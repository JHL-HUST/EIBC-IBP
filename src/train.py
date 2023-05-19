import argparse
import glob
import json
import os
import random
import sys
import datetime
import math

import numpy as np
import torch
from tqdm import tqdm

import data_util
import text_classification
import attacks
import vocabulary


def triplet_loss(xs,
                 synonyms,
                 embeddings,
                 device,
                 nb=8,
                 margin=0.5,
                 gamma=0):
    xs = xs.squeeze(-1)  # [b, n]
    synonyms = synonyms.squeeze(-1)  # [b, n, k]
    mask = xs.ne(0.0)
    xs = torch.masked_select(xs, mask)  # [b*n]
    if gamma > 0:
        synonyms = torch.masked_select(synonyms, mask.unsqueeze(-1).repeat(
            1, 1, math.ceil(nb*gamma))).view(-1, synonyms.size(-1))  # [b*n, k]
    else:
        synonyms = torch.masked_select(
            synonyms, mask.unsqueeze(-1).repeat(1, 1, 9)).view(-1, synonyms.size(-1))  # [b*n, k]
    vocab_size = embeddings.size()[0]
    # [b*n, k] filter where synonym is [UNK]
    pos_mask = torch.ge(synonyms, 1).float()
    anchor = embeddings[xs]  # [b*n, d]
    positive = embeddings[synonyms]  # [b*n, k, d]

    non_synonyms = torch.randint(low=1,
                                 high=vocab_size,
                                 size=(nb, ),
                                 dtype=torch.long).to(device).detach()
    non_synonyms = non_synonyms.unsqueeze(0)  # [1, k_n]
    negtive = embeddings[non_synonyms]  # [1, k_n, d]

    nonsyn_dis = torch.norm((anchor.unsqueeze(1) - negtive), dim=-1, p=1)
    nonsyn_dis = torch.mean(torch.min(torch.zeros_like(
        nonsyn_dis) + margin, other=nonsyn_dis), dim=-1)
    syn_dis = (anchor.unsqueeze(1) - positive) * pos_mask.unsqueeze(-1)
    syn_dis = torch.norm(syn_dis, dim=-2, p=float('inf'))  # [b*n, 1, d]
    # you can use the following code to test large norm of distance metric
    # with torch.no_grad():
    #     normalize = torch.norm(syn_dis, dim=-1, p=float('inf'), keepdim=True)
    #     normalize = torch.where(normalize==0, torch.full_like(normalize, 1), normalize)
    # syn_dis = (torch.norm(syn_dis/ normalize, dim=-1, p=1, keepdim=True) * normalize).squeeze(-1) #[b*n, 1, d]
    syn_dis = torch.norm(syn_dis, dim=-1, p=1)

    tpl_loss = syn_dis - nonsyn_dis + margin
    # [b*n, ]

    tpl_loss = torch.mean(torch.max(torch.zeros_like(
        tpl_loss, dtype=torch.float), tpl_loss))

    return tpl_loss, torch.mean(syn_dis), torch.mean(nonsyn_dis)


def ibp_cert_loss(upper, lower, targets):
    margin = upper - torch.gather(lower, 1, targets.view(-1, 1))
    margin = margin.scatter(1, targets.view(-1, 1), 0)
    return torch.nn.functional.cross_entropy(margin, targets.squeeze(-1))


def train(task_class,
          model,
          train_data,
          num_epochs,
          lr,
          device,
          alpha,
          nb,
          only_emb=False,
          only_model=False,
          use_lrsch_in_fulltrain=True,
          test_data=None,
          cert_frac=0.0,
          initial_cert_frac=0.0,
          cert_eps=1.0,
          initial_cert_eps=0.0,
          non_cert_train_epochs=0,
          full_train_epochs=0,
          batch_size=1,
          epochs_per_save=1,
          clip_grad_norm=0,
          weight_decay=0,
          load_emb=True,
          out_dir=None,
          load_emb_dir=None):
    if (load_emb):
        load_fn = os.path.join(load_emb_dir, 'model-checkpoint-19.pth')
        print('Loading model from %s.' % load_fn)
        state_dict = dict(torch.load(load_fn))
        model.embs.weight.copy_(state_dict['embs.weight'])
        # model.init_weight()
    print('Training model')
    sys.stdout.flush()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    zero_stats = {'epoch': 0, 'clean_acc': 0.0, 'cert_acc': 0.0}
    all_epoch_stats = {
        "loss": {
            "total": [],
            "clean": [],
            "cert": []
        },
        "cert": {
            "frac": [],
            "eps": []
        },
        "acc": {
            "train": {
                "clean": [],
                "cert": []
            },
            "dev": {
                "clean": [],
                "cert": []
            },
            "best_dev": {
                "clean": [zero_stats],
                "cert": [zero_stats]
            }
        },
        "total_epochs": num_epochs,
    }
    # Create all batches now and pin them in memory
    data = train_data.get_loader(batch_size)
    # Linearly increase the weight of adversarial loss over all the epochs to end up at the final desired fraction
    if cert_frac > 0.0:
        cert_schedule = torch.tensor(np.linspace(
            initial_cert_frac, cert_frac,
            num_epochs - full_train_epochs - non_cert_train_epochs),
            dtype=torch.float,
            device=device)
        eps_schedule = torch.tensor(np.linspace(
            initial_cert_eps, cert_eps,
            num_epochs - full_train_epochs - non_cert_train_epochs),
            dtype=torch.float,
            device=device)
    else:
        cert_schedule = torch.tensor(np.linspace(
            0, 0,
            num_epochs - full_train_epochs - non_cert_train_epochs),
            dtype=torch.float,
            device=device)
        eps_schedule = torch.tensor(np.linspace(
            0, 0,
            num_epochs - full_train_epochs - non_cert_train_epochs),
            dtype=torch.float,
            device=device)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=full_train_epochs, verbose=True)
    for t in range(num_epochs):
        model.train()
        if t < non_cert_train_epochs:
            cur_cert_frac = 0.0
            cur_cert_eps = 0.0
        else:
            cur_cert_frac = cert_schedule[t - non_cert_train_epochs] if t - \
                non_cert_train_epochs < len(cert_schedule) else cert_schedule[-1]
            cur_cert_eps = eps_schedule[t - non_cert_train_epochs] if t - \
                non_cert_train_epochs < len(eps_schedule) else eps_schedule[-1]
        if t > num_epochs - full_train_epochs and use_lrsch_in_fulltrain:
            lr_scheduler.step()
        epoch = {
            "total_loss": 0.0,
            "clean_loss": 0.0,
            "cert_loss": 0.0,
            "num_correct": 0,
            "num_cert_correct": 0,
            "num": 0,
            "clean_acc": 0,
            "cert_acc": 0,
            "triplet_bound_loss": 0,
            "dev": {},
            "best_dev": {},
            "cert_frac": cur_cert_frac if isinstance(cur_cert_frac, float) else cur_cert_frac.item(),
            "cert_eps": cur_cert_eps if isinstance(cur_cert_eps, float) else cur_cert_eps.item(),
            "epoch": t,
            "sys_loss": 0.0,
            "nonsys_loss": 0.0,
            "cur_cert_frac": 0.0
        }
        step = 0
        for batch_idx, batch in enumerate(data):
            batch = data_util.dict_batch_to_device(batch, device)
            optimizer.zero_grad()
            step += 1
            if cur_cert_frac > 0.0:
                if only_emb:
                    embeddings = model.get_embeddings()
                    triplet, syss, nonsys = triplet_loss(batch['x'].val,
                                                         batch['x'].choice_mat,
                                                         embeddings,
                                                         device,
                                                         nb=nb,
                                                         margin=alpha,
                                                         gamma=OPTS.gamma)
                    epoch["triplet_bound_loss"] += (triplet).item()
                    epoch["sys_loss"] += syss.item()
                    epoch["nonsys_loss"] += nonsys.item()
                    loss = triplet
                elif only_model:
                    out = model.forward(
                        batch, cert_eps=cur_cert_eps, compute_bounds=True)
                    logits = out.val
                    clean_loss = torch.nn.functional.cross_entropy(
                        logits, batch['y'].to(torch.int64).squeeze(-1))
                    epoch["clean_loss"] += clean_loss.item()

                    cert_loss = ibp_cert_loss(
                        out.ub, out.lb, batch['y'].to(torch.int64))

                    epoch["cert_loss"] += cert_loss.item()
                    epoch["cur_cert_frac"] = cur_cert_frac.item()

                    loss = cur_cert_frac * cert_loss + (1.0 - cur_cert_frac) * clean_loss
                else:
                    embeddings = model.get_embeddings()
                    triplet, syss, nonsys = triplet_loss(batch['x'].val,
                                                         batch['x'].choice_mat,
                                                         embeddings,
                                                         device,
                                                         nb=nb,
                                                         margin=alpha,
                                                         gamma=OPTS.gamma)
                    epoch["triplet_bound_loss"] += (triplet).item()
                    epoch["sys_loss"] += syss.item()
                    epoch["nonsys_loss"] += nonsys.item()

                    out = model.forward(
                        batch, cert_eps=cur_cert_eps, compute_bounds=True, freeze_embs=True)
                    logits = out.val
                    clean_loss = torch.nn.functional.cross_entropy(
                        logits, batch['y'].to(torch.int64).squeeze(-1))
                    epoch["clean_loss"] += clean_loss.item()

                    cert_loss = ibp_cert_loss(
                        out.ub, out.lb, batch['y'].to(torch.int64))

                    epoch["cert_loss"] += cert_loss.item()
                    epoch["cur_cert_frac"] = cur_cert_frac.item()

                    loss = cur_cert_frac * cert_loss + (1.0 - cur_cert_frac) * clean_loss + triplet

            else:
                # Bypass computing bounds during training
                if only_emb:
                    embeddings = model.get_embeddings()
                    triplet, syss, nonsys = triplet_loss(batch['x'].val,
                                                         batch['x'].choice_mat,
                                                         embeddings,
                                                         device,
                                                         nb=nb,
                                                         margin=alpha,
                                                         gamma=OPTS.gamma)
                    epoch["triplet_bound_loss"] += (triplet).item()
                    epoch["sys_loss"] += syss.item()
                    epoch["nonsys_loss"] += nonsys.item()
                    loss = triplet
                elif only_model:
                    out = model.forward(
                        batch, cert_eps=cur_cert_eps, compute_bounds=False)
                    logits = out
                    clean_loss = torch.nn.functional.cross_entropy(
                        logits, batch['y'].to(torch.int64).squeeze(-1))

                    loss = clean_loss
                else:
                    embeddings = model.get_embeddings()
                    triplet, syss, nonsys = triplet_loss(batch['x'].val,
                                                         batch['x'].choice_mat,
                                                         embeddings,
                                                         device,
                                                         nb=nb,
                                                         margin=alpha,
                                                         gamma=OPTS.gamma)
                    epoch["triplet_bound_loss"] += (triplet).item()
                    epoch["sys_loss"] += syss.item()
                    epoch["nonsys_loss"] += nonsys.item()

                    out = model.forward(
                        batch, cert_eps=cur_cert_eps, compute_bounds=False, freeze_embs=True)
                    logits = out
                    clean_loss = torch.nn.functional.cross_entropy(
                        logits, batch['y'].to(torch.int64).squeeze(-1))

                    loss = clean_loss + triplet

            epoch["total_loss"] += loss.item()
            epoch["num"] += len(batch['y'])
            if not only_emb:
                # targets = torch.tensor(batch['y'], dtype=torch.int64)
                num_correct, num_cert_correct = task_class.num_correct(out, batch['y'].to(torch.int64).squeeze(-1))
                epoch["num_correct"] += num_correct
                epoch["num_cert_correct"] += num_cert_correct
            loss.backward()
            if any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters()):
                nan_params = [
                    p.name for p in model.parameters()
                    if p.grad is not None and torch.isnan(p.grad).any()
                ]
                print('NaN found in gradients: %s' %
                      nan_params, file=sys.stderr)
            else:
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_grad_norm)
                optimizer.step()
        if cert_frac > 0.0:
            print(
                "Epoch {epoch:>3}: train loss: {total_loss:.6f}, clean_loss: {clean_loss:.6f}, cert_loss: {cert_loss:.6f}, triplet_bound_loss:{triplet_bound_loss:.6f}, cur_frac:{cur_cert_frac:.6f}"
                .format(**epoch))
        else:
            print(
                "Epoch {epoch:>3}: train loss: {total_loss:.6f} triplet_bound_loss:{triplet_bound_loss:.6f}, clean_loss:{clean_loss:.6f}, sys_loss:{sys_loss:.6f}, nonsys_loss:{nonsys_loss:.6f}, cur_frac:{cur_cert_frac:.6f}"
                .format(**epoch))
        sys.stdout.flush()

        epoch["clean_acc"] = 100.0 * epoch["num_correct"] / epoch["num"]
        acc_str = "  Train accuracy: {num_correct}/{num} = {clean_acc:.2f}".format(
            **epoch)
        if cert_frac > 0.0:
            epoch["cert_acc"] = 100.0 * \
                epoch["num_cert_correct"] / epoch["num"]
            acc_str += ", certified {num_cert_correct}/{num} = {cert_acc:.2f}".format(
                **epoch)
        print(acc_str)
        if test_data and not only_emb:
            dev_results = test(task_class,
                               model,
                               "Dev",
                               test_data,
                               device,
                               batch_size=batch_size)
            epoch['dev'] = dev_results
            all_epoch_stats['acc']['dev']['clean'].append(
                dev_results['clean_acc'])
            all_epoch_stats['acc']['dev']['cert'].append(
                dev_results['cert_acc'])

        all_epoch_stats["loss"]['total'].append(epoch["total_loss"])
        all_epoch_stats["loss"]['clean'].append(epoch["clean_loss"])
        all_epoch_stats["loss"]['cert'].append(epoch["cert_loss"])
        all_epoch_stats['cert']['frac'].append(epoch["cert_frac"])
        all_epoch_stats['cert']['eps'].append(epoch["cert_eps"])
        all_epoch_stats["acc"]['train']['clean'].append(epoch["clean_acc"])
        all_epoch_stats["acc"]['train']['cert'].append(epoch["cert_acc"])
        with open(os.path.join(out_dir, "run_stats.json"), "w") as outfile:
            json.dump(epoch, outfile)
        with open(os.path.join(out_dir, "all_epoch_stats.json"),
                  "w") as outfile:
            json.dump(all_epoch_stats, outfile)
        if (epochs_per_save and
                (t + 1) % epochs_per_save == 0) or t == num_epochs - 1:
            model_save_path = os.path.join(out_dir,
                                           "model-checkpoint-{}.pth".format(t))
            print('Saving model to %s' % model_save_path)
            torch.save(model.state_dict(), model_save_path)

    return model


def test(task_class,
         model,
         name,
         dataset,
         device,
         batch_size=1,
         adversary=None,
         attack_batch=1000):
    model.eval()
    results = {
        'name': name,
        'num_total': 0,
        'num_correct': 0,
        'num_cert_correct': 0,
        'clean_acc': 0.0,
        'cert_acc': 0.0,
        'clean_loss': 0.0,
        "bound_loss": 0.0,
        "sys_loss": 0.0,
        "nonsys_loss": 0.0
    }
    data = dataset.get_loader(batch_size)
    with torch.no_grad():
        for batch in data:
            batch = data_util.dict_batch_to_device(batch, device)
            out = model.forward(batch, cert_eps=1.0, compute_bounds=True)
            # targets = torch.tensor(batch['y'], dtype=torch.int64)
            results['clean_loss'] += torch.nn.functional.cross_entropy(
                out.val, batch['y'].to(torch.int64).squeeze(-1)).item()
            num_correct, num_cert_correct = task_class.num_correct(
                out, batch['y'].to(torch.int64).squeeze(-1))
            results["num_correct"] += num_correct
            results["num_cert_correct"] += num_cert_correct
            results['num_total'] += len(batch['y'])

    results['clean_acc'] = 100 * results['num_correct'] / results['num_total']
    results[
        'cert_acc'] = 100 * results['num_cert_correct'] / results['num_total']
    out_str = "  {name} clean_loss = {clean_loss:.2f}; accuracy: {num_correct}/{num_total} = {clean_acc:.2f}, certified {num_cert_correct}/{num_total} = {cert_acc:.2f}, bound_loss={bound_loss:.2f}, sys_loss={sys_loss:.2f}, nonsys_loss={nonsys_loss:.2f}".format(
        **results)
    print(out_str)

    if adversary:
        success_attack_count = 0
        fail_count = 0
        unchanged_sample_count = 0
        with tqdm(data) as batch_loop:
            for batch_idx, batch in enumerate(batch_loop):
                batch = data_util.dict_batch_to_device(batch, device)
                x = batch['x']
                # adv_sentence = str(sentence.val)
                label = batch['y']
                # adv_label = int(model.query([sentence], [label])[1][0])
                out = model.forward(batch, cert_eps=0, compute_bounds=True)
                num_correct, num_cert_correct = task_class.num_correct(
                    out, batch['y'].to(torch.int64).squeeze(-1))
                if num_correct:
                    success, _, _ = adversary.run(
                        batch['x'].val, batch['x'].choice_mat, label)
                    if success:
                        success_attack_count += 1
                    else:
                        fail_count += 1
                else:
                    unchanged_sample_count += 1
        model_acc_before_attack = 1.0 - unchanged_sample_count / attack_batch
        model_acc_after_attack = 1.0 - \
            (unchanged_sample_count + success_attack_count) / attack_batch
        out_str = "model acc before attack = {:.4f}; model acc after attack = {:.4f}".format(
            model_acc_before_attack*100, model_acc_after_attack*100)
        print(out_str)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['bow', 'cnn', 'textcnn'])
    parser.add_argument('dataset', choices=['YELP', 'IMDB', 'SST2'])
    parser.add_argument('out_dir', help='Directory to store and load output')
    # Model
    parser.add_argument('--hidden-size', '-d', type=int, default=200)
    parser.add_argument('--kernel-size',
                        '-k',
                        type=int,
                        default=3,
                        help='Kernel size, for CNN convolutions and pooling')
    parser.add_argument('--pool',
                        choices=['max', 'mean', 'attn'],
                        default='max')
    parser.add_argument('--no-wordvec-layer',
                        action='store_true',
                        help="Don't apply linear transform to word vectors")
    parser.add_argument('--early-ibp',
                        action='store_true',
                        help="Do to_interval_bounded directly on base word vectors")
    parser.add_argument('--no-relu-wordvec',
                        action='store_true',
                        help="Don't do ReLU after word vector transform")
    parser.add_argument('--unfreeze-wordvec',
                        action='store_true',
                        help="Don't freeze word vectors")
    parser.add_argument('--glove',
                        choices=vocabulary.GLOVE_CONFIGS,
                        default='840B.300d')
    # Adversary
    parser.add_argument('--adversary',
                        '-a',
                        choices=['genetic'],
                        default=None,
                        help='Which adversary to test on')
    parser.add_argument('--adv-num-epochs', type=int, default=10)
    parser.add_argument('--adv-num-tries', type=int, default=2)
    parser.add_argument('--adv-pop-size', type=int, default=60)
    parser.add_argument('--vocab-size', type=int, default=0)
    parser.add_argument('--max_seq_length', type=int, default=50000)

    # Training
    parser.add_argument('--num-epochs', '-T', type=int, default=1)
    parser.add_argument('--learning-rate', '-r', type=float, default=1e-3)
    parser.add_argument('--dropout-prob', type=float, default=0.2)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--clip-grad-norm', type=float, default=0.25)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument(
        '--cert-frac',
        '-c',
        type=float,
        default=0.0,
        help='Fraction of loss devoted to certified loss term.')
    parser.add_argument(
        '--initial-cert-frac',
        type=float,
        default=0.0,
        help='If certified loss is being used, where the linear scale for it begins'
    )
    parser.add_argument(
        '--cert-eps',
        type=float,
        default=1.0,
        help='Max scaling factor for the interval bounds of the attack words to be used'
    )
    parser.add_argument(
        '--initial-cert-eps',
        type=float,
        default=0.0,
        help='If certified loss is being used, where the linear scale for its epsilon begins'
    )
    parser.add_argument(
        '--full-train-epochs',
        type=int,
        default=0,
        help='If specified use full cert_frac and cert_eps for this many epochs at the end'
    )
    parser.add_argument(
        '--non-cert-train-epochs',
        type=int,
        default=0,
        help='If specified train this many epochs regularly in beginning')
    parser.add_argument(
        '--epochs-per-save',
        type=int,
        default=0,
        help='How often to save model; 0 to only save final model')
    parser.add_argument(
        '--save-best-only',
        action='store_true',
        help='Only save best dev epochs (based on cert acc if cert_frac > 0, clean acc else)'
    )

    # Data and files
    parser.add_argument('--test',
                        action='store_true',
                        help='Evaluate on test set')
    parser.add_argument('--data-cache-dir',
                        '-D',
                        help='Where to load cached dataset and glove')
    parser.add_argument('--neighbor-file',
                        type=str,
                        default=data_util.NEIGHBOR_FILE)
    parser.add_argument('--glove-dir', type=str, default=vocabulary.GLOVE_DIR)
    parser.add_argument('--imdb-dir',
                        type=str,
                        default=text_classification.IMDB_DIR)
    parser.add_argument('--yelp-dir',
                        type=str,
                        default=text_classification.YELP_DIR)
    parser.add_argument('--sst2-dir',
                        type=str,
                        default=text_classification.SST2_DIR)
    parser.add_argument('--prepend-null',
                        action='store_true',
                        help='If true add UNK token to sequences')
    parser.add_argument('--normalize-word-vecs',
                        action='store_true',
                        help='If true normalize word vectors')
    parser.add_argument(
        '--downsample-to',
        type=int,
        default=None,
        help='Downsample train and dev data to this many examples')
    parser.add_argument(
        '--downsample-shard',
        type=int,
        default=0,
        help='Downsample starting at this multiple of downsample_to')
    parser.add_argument('--truncate-to',
                        type=int,
                        default=None,
                        help='Truncate examples to this max length')
    # Loading
    parser.add_argument('--load-emb',
                        action='store_true',
                        help='If load embedding layer parameters from load-emb-dir')
    parser.add_argument('--load-dir', '-L', help='Where to load checkpoint')
    parser.add_argument(
        '--load-emb-dir', help='Where to load checkpoint for embedding layer')
    parser.add_argument('--load-ckpt',
                        type=int,
                        default=None,
                        help='Which checkpoint to load')
    # random seed
    parser.add_argument('--seed_value', type=int, default=123456)

    # triplet bound loss
    parser.add_argument("--nb", default=8, type=int, help="num of negtive samples in triplet loss.")
    parser.add_argument("--alpha",
                        default=6.0,
                        type=float,
                        help="hyperparameter for triplet bound loss.")
    parser.add_argument('--only-emb',
                        action='store_true',
                        help="If only train the embedding layer.")
    parser.add_argument('--only-model',
                        action='store_true',
                        help="If only train the model without embedding layer.")
    parser.add_argument('--use_lrsch_in_fulltrain',
                        action='store_true',
                        help='If use lrscheduler in full train epochs')
    
    parser.add_argument('--gamma',
                        default=0.0,
                        type=float,
                        help="hyperparameter to control unseen substitutions. 0 to disable unseen substitutions.")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    # Set seed
    random.seed(OPTS.seed_value)
    np.random.seed(OPTS.seed_value)
    torch.manual_seed(OPTS.seed_value)
    os.environ['PYTHONHASHSEED'] = str(OPTS.seed_value)
    torch.cuda.manual_seed(OPTS.seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Loading dataset.')
    if not os.path.exists(OPTS.out_dir):
        os.makedirs(OPTS.out_dir)
    with open(os.path.join(OPTS.out_dir, 'log.txt'), 'w') as f:
        print(sys.argv, file=f)
        print(OPTS, file=f)
    if OPTS.data_cache_dir:
        if not os.path.exists(OPTS.data_cache_dir):
            os.makedirs(OPTS.data_cache_dir)
    train_data, test_data, word_mat, attack_surface = text_classification.load_datasets_v2(
        device, OPTS)
    print('Initializing model.')
    model = text_classification.load_model(word_mat, device, OPTS)

    starttime = datetime.datetime.now()

    if OPTS.num_epochs > 0:
        train(text_classification,
              model,
              train_data,
              OPTS.num_epochs,
              OPTS.learning_rate,
              device,
              test_data=test_data,
              cert_frac=OPTS.cert_frac,
              initial_cert_frac=OPTS.initial_cert_frac,
              cert_eps=OPTS.cert_eps,
              initial_cert_eps=OPTS.initial_cert_eps,
              batch_size=OPTS.batch_size,
              epochs_per_save=OPTS.epochs_per_save,
              clip_grad_norm=OPTS.clip_grad_norm,
              weight_decay=OPTS.weight_decay,
              full_train_epochs=OPTS.full_train_epochs,
              non_cert_train_epochs=OPTS.non_cert_train_epochs,
              alpha=OPTS.alpha,
              nb=OPTS.nb,
              use_lrsch_in_fulltrain=OPTS.use_lrsch_in_fulltrain,
              only_emb=OPTS.only_emb,
              only_model=OPTS.only_model,
              out_dir=OPTS.out_dir,
              load_emb=OPTS.load_emb,
              load_emb_dir=OPTS.load_emb_dir)
        print('Training finished.')

    endtime = datetime.datetime.now()
    print('Training model takes {:s}'.format(str(endtime - starttime)))

    print('Testing model.')
    train_results = test(text_classification,
                         model,
                         'Train',
                         train_data,
                         device,
                         batch_size=OPTS.batch_size)
    adversary = None
    if OPTS.adversary == 'genetic':
        adversary = attacks.GAAdversary(
            attack_surface, model, iterations_num=OPTS.adv_num_epochs, pop_max_size=OPTS.adv_pop_size, device=device)
    dev_results = test(text_classification,
                       model,
                       'Dev',
                       test_data,
                       device,
                       adversary=adversary,
                       batch_size=OPTS.batch_size,
                       attack_batch=OPTS.downsample_to)
    results = {'train': train_results, 'dev': dev_results}
    with open(os.path.join(OPTS.out_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    OPTS = parse_args()
    main()
