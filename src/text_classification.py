"""IBP text classification model."""
import itertools
import glob
import json
import numpy as np
import os
import pickle
import random
import bz2
import math

from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import attacks
import data_util
import ibp
import vocabulary

LOSS_FUNC = nn.BCEWithLogitsLoss()
IMDB_DIR = 'data/aclImdb'
YELP_DIR = 'data/yelp'
SST2_DIR = 'data/sst2'
COUNTER_FITTED_FILE = 'data/counter-fitted-vectors.txt'


class AdversarialModel(nn.Module):

    def __init__(self):
        super(AdversarialModel, self).__init__()


def attention_pool(x, mask, layer):
    """Attention pooling

  Args:
    x: batch of inputs, shape (B, n, h)
    mask: binary mask, shape (B, n)
    layer: Linear layer mapping h -> 1
  Returns:
    pooled version of x, shape (B, h)
  """
    attn_raw = layer(x).squeeze(2)  # B, n, 1 -> B, n
    attn_raw = ibp.add(attn_raw, (1 - mask) * -1e20)
    attn_logsoftmax = ibp.log_softmax(attn_raw, 1)
    attn_probs = ibp.activation(torch.exp, attn_logsoftmax)  # B, n
    return ibp.bmm(attn_probs.unsqueeze(1),
                   x).squeeze(1)  # B, 1, n x B, n, h -> B, h


class CNNModel(AdversarialModel):
    """Convolutional neural network.
    Here is the overall architecture:
    1) Rotate word vectors
    2) One convolutional layer
    3) Max/mean pool across all time
    4) Predict with MLP
    """

    def __init__(self,
                 word_vec_size,
                 hidden_size,
                 kernel_size,
                 word_mat,
                 pool='max',
                 dropout=0.2,
                 no_wordvec_layer=False,
                 early_ibp=False,
                 relu_wordvec=True,
                 unfreeze_wordvec=False,
                 num_classes=2):
        super(CNNModel, self).__init__()
        cnn_padding = (kernel_size - 1) // 2  # preserves size
        self.pool = pool
        # Ablations
        self.no_wordvec_layer = no_wordvec_layer
        self.early_ibp = early_ibp
        self.relu_wordvec = relu_wordvec
        self.unfreeze_wordvec = unfreeze_wordvec
        # End ablations
        self.embs = ibp.Embedding.from_pretrained(
            word_mat, freeze=not self.unfreeze_wordvec)
        if no_wordvec_layer:
            self.conv1 = ibp.Conv1d(word_vec_size,
                                    hidden_size,
                                    kernel_size,
                                    padding=cnn_padding)
        else:
            self.linear_input = ibp.Linear(word_vec_size, hidden_size)
            self.conv1 = ibp.Conv1d(hidden_size,
                                    hidden_size,
                                    kernel_size,
                                    padding=cnn_padding)
        if self.pool == 'attn':
            self.attn_pool = ibp.Linear(hidden_size, 1)
        self.dropout = ibp.Dropout(dropout)
        self.fc_hidden = ibp.Linear(hidden_size, hidden_size)
        self.fc_output = ibp.Linear(hidden_size, num_classes)

    def forward(self, batch, compute_bounds=True, cert_eps=1.0, freeze_embs=False):
        """
        Args:
            batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
            compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
            cert_eps: Scaling factor for interval bounds of the input
        """
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']

        x_vecs = self.embs(x)  # B, n, d

        if freeze_embs:
            x_vecs = x_vecs.detach()

        if self.early_ibp and isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer and not self.relu_wordvec:
            z = x_vecs
        else:
            z = ibp.activation(F.relu, x_vecs)  # B, n, h
        z_masked = z * mask.unsqueeze(-1)  # B, n, h
        z_cnn_in = z_masked.permute(0, 2, 1)  # B, h, n
        c1 = ibp.activation(F.relu, self.conv1(z_cnn_in))  # B, h, n
        c1 = c1 * mask.unsqueeze(1)  # B, h, n
        if self.pool == 'mean':
            fc_in = ibp.sum(c1 / lengths.to(dtype=torch.float).view(-1, 1, 1), 2)  # B, h
        elif self.pool == 'attn':
            fc_in = attention_pool(c1.permute(0, 2, 1), mask, self.attn_pool)  # B, h
        else:
            # zero-masking works b/c ReLU guarantees that everything is >= 0
            fc_in = ibp.pool(torch.max, c1, 2)  # B, h
        fc_in = self.dropout(fc_in)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        return output

    def input_to_embs(self, batch, compute_bounds=True):
        # Embedding
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']

        x = self.embs(x)  # dim: (batch_size, max_seq_len, embedding_size)
        return x, mask, lengths

    def embs_to_logit(self, x_vecs, mask, lengths, cert_eps=1.0):
        if self.early_ibp and isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer and not self.relu_wordvec:
            z = x_vecs
        else:
            z = ibp.activation(F.relu, x_vecs)  # B, n, h
        z_masked = z * mask.unsqueeze(-1)  # B, n, h
        z_cnn_in = z_masked.permute(0, 2, 1)  # B, h, n
        c1 = ibp.activation(F.relu, self.conv1(z_cnn_in))  # B, h, n
        c1 = c1 * mask.unsqueeze(1)  # B, h, n
        if self.pool == 'mean':
            fc_in = ibp.sum(c1 / lengths.to(dtype=torch.float).view(-1, 1, 1),
                            2)  # B, h
        elif self.pool == 'attn':
            fc_in = attention_pool(c1.permute(0, 2, 1), mask,
                                   self.attn_pool)  # B, h
        else:
            # zero-masking works b/c ReLU guarantees that everything is >= 0
            fc_in = ibp.pool(torch.max, c1, 2)  # B, h
        fc_in = self.dropout(fc_in)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        return output

    def query_from_ids(self, x):
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        logits = self.input_to_logit(x).detach().cpu().numpy()
        return logits

    def get_embeddings(self):
        return self.embs.weight


class TextCNNModel(AdversarialModel):
    """Convolutional neural network.
    Here is the overall architecture:
    1) Rotate word vectors
    2) One convolutional layer
    3) Max/mean pool across all time
    4) Predict with MLP
    """

    def __init__(self,
                 word_vec_size,
                 hidden_size,
                 kernel_size,
                 word_mat,
                 pool='max',
                 dropout=0.2,
                 no_wordvec_layer=True,
                 early_ibp=False,
                 relu_wordvec=False,
                 unfreeze_wordvec=True,
                 num_classes=2):
        super(TextCNNModel, self).__init__()
        cnn_padding = (kernel_size - 1) // 2  # preserves size
        self.pool = pool
        self.filter_sizes = (2, 3, 4)
        # Ablations
        self.no_wordvec_layer = no_wordvec_layer
        self.early_ibp = early_ibp
        self.relu_wordvec = relu_wordvec
        self.unfreeze_wordvec = unfreeze_wordvec
        # End ablations
        self.embs = ibp.Embedding.from_pretrained(
            word_mat, freeze=not self.unfreeze_wordvec)
        if no_wordvec_layer:
            self.convs = nn.ModuleList([
                ibp.Conv1d(word_vec_size,
                           hidden_size,
                           kernel_size,
                           padding='same') for kernel_size in self.filter_sizes
            ])
        else:
            self.linear_input = ibp.Linear(word_vec_size, hidden_size)
            self.convs = nn.ModuleList([
                ibp.Conv1d(hidden_size,
                           hidden_size,
                           kernel_size,
                           padding='same') for kernel_size in self.filter_sizes
            ])
        if self.pool == 'attn':
            self.attn_pool = ibp.Linear(hidden_size, 1)
        self.dropout = ibp.Dropout(dropout)
        self.fc_hidden = ibp.Linear(hidden_size * len(self.filter_sizes),
                                    hidden_size * len(self.filter_sizes))
        self.fc_output = ibp.Linear(hidden_size * len(self.filter_sizes), num_classes)

    def forward(self, batch, compute_bounds=True, cert_eps=1.0, freeze_embs=False):
        """
        Args:
            batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
            compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
            cert_eps: Scaling factor for interval bounds of the input
        """
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']

        x_vecs = self.embs(x)  # B, n, d

        if freeze_embs:
            x_vecs = x_vecs.detach()

        if self.early_ibp and isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer and not self.relu_wordvec:
            z = x_vecs
        else:
            z = ibp.activation(F.relu, x_vecs)  # B, n, h

        z_masked = z * mask.unsqueeze(-1)  # B, n, h
        z_cnn_in = z_masked.permute(0, 2, 1)  # B, h, n
        c = ibp.cat([self.conv_and_pool(z_cnn_in, conv, mask, self.pool)
            for conv in self.convs], 1)
        # B, h
        fc_in = self.dropout(c)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        return output

    def conv_and_pool(self, x, conv, mask, pool='max'):
        x = ibp.activation(F.relu, conv(x))
        x = x * mask.unsqueeze(1)
        if pool == 'mean':
            if isinstance(x, ibp.IntervalBoundedTensor):
                x = ibp.sum(x / torch.tensor(
                    x.val.size(2), dtype=torch.float).cuda().view(-1, 1, 1),
                            2)  # B, h
            else:
                x = ibp.sum(x / torch.tensor(
                    x.size(2), dtype=torch.float).cuda().view(-1, 1, 1),
                            2)  # B, h
        elif pool == 'attn':
            x = attention_pool(x.permute(0, 2, 1), mask,
                               self.attn_pool)  # B, h
        else:
            # zero-masking works b/c ReLU guarantees that everything is >= 0
            x = ibp.pool(torch.max, x, 2)  # B, h
        return x

    def input_to_embs(self, batch, compute_bounds=True):
        # Embedding
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']
        x = self.embs(x)  # dim: (batch_size, max_seq_len, embedding_size)
        return x, mask, lengths

    def embs_to_logit(self, x_vecs, mask, lengths, cert_eps=1.0):
        if self.early_ibp and isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer and not self.relu_wordvec:
            z = x_vecs
        else:
            z = ibp.activation(F.relu, x_vecs)  # B, n, h

        z_masked = z * mask.unsqueeze(-1)  # B, n, h
        z_cnn_in = z_masked.permute(0, 2, 1)  # B, h, n
        c = ibp.cat([
            self.conv_and_pool(z_cnn_in, conv, mask, self.pool)
            for conv in self.convs
        ], 1)
        # B, h
        fc_in = self.dropout(c)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        return output

    def input_to_logits(self, batch):

        x = batch

        mask = torch.ones(x.size()).to(x.device)


        cert_eps=0.0

        x_vecs = self.embs(x)  # B, n, d
        if self.early_ibp and isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer and not self.relu_wordvec:
            z = x_vecs
        else:
            z = ibp.activation(F.relu, x_vecs)  # B, n, h

        z_masked = z * mask.unsqueeze(-1)  # B, n, h
        z_cnn_in = z_masked.permute(0, 2, 1)  # B, h, n
        c = ibp.cat([self.conv_and_pool(z_cnn_in, conv, mask, self.pool)
            for conv in self.convs], 1)
        # B, h
        fc_in = self.dropout(c)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        return output

    def query_from_ids(self, x):
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        logits = self.input_to_logit(x).detach().cpu().numpy()
        return logits

    def get_embeddings(self):
        return self.embs.weight

    def init_weight(self):
        for conv in self.convs:
            torch.nn.init.normal_(conv.weight, std=0.1)
            torch.nn.init.constant_(conv.bias, 0.1)
        torch.nn.init.xavier_uniform_(self.fc_hidden.weight)
        torch.nn.init.constant_(self.fc_hidden.bias, 0.1)
        torch.nn.init.xavier_uniform_(self.fc_output.weight)
        torch.nn.init.constant_(self.fc_output.bias, 0.1)
        torch.nn.init.xavier_uniform_(self.linear_input.weight)
        torch.nn.init.constant_(self.linear_input.bias, 0.1)
    
    def query(self, x, labels):
        logits = self.input_to_logits(x).detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        return logits, predictions


class BOWModel(AdversarialModel):
    """Bag of word vectors + MLP."""

    def __init__(self,
                 word_vec_size,
                 hidden_size,
                 word_mat,
                 pool='max',
                 dropout=0.2,
                 no_wordvec_layer=False):
        super(BOWModel, self).__init__()
        self.pool = pool
        self.no_wordvec_layer = no_wordvec_layer
        self.embs = ibp.Embedding.from_pretrained(word_mat)
        if no_wordvec_layer:
            self.linear_hidden = ibp.Linear(word_vec_size, hidden_size)
        else:
            self.linear_input = ibp.Linear(word_vec_size, hidden_size)
            self.linear_hidden = ibp.Linear(hidden_size, hidden_size)
        self.linear_output = ibp.Linear(hidden_size, 1)
        self.dropout = ibp.Dropout(dropout)
        if self.pool == 'attn':
            self.attn_pool = ibp.Linear(hidden_size, 1)

    def forward(self, batch, compute_bounds=True, cert_eps=1.0):
        """Forward pass of BOWModel.

    Args:
      batch: A batch dict from a TextClassificationDataset with the following keys:
        - x: tensor of word vector indices, size (B, n, 1)
        - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
        - lengths: lengths of sequences, size (B,)
      compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
      cert_eps: Scaling factor for interval bounds of the input
    """
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']

        x_vecs = self.embs(x)  # B, n, d
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer:
            z1 = x_vecs
        else:
            z1 = ibp.activation(F.relu, x_vecs)
        z1_masked = z1 * mask.unsqueeze(-1)  # B, n, h
        if self.pool == 'mean':
            z1_pooled = ibp.sum(z1_masked /
                                lengths.to(dtype=torch.float).view(-1, 1, 1),
                                1)  # B, h
        elif self.pool == 'attn':
            z1_pooled = attention_pool(z1_masked, mask, self.attn_pool)
        else:  # max
            # zero-masking works b/c ReLU guarantees that everything is >= 0
            z1_pooled = ibp.pool(torch.max, z1_masked, 1)  # B, h
        z1_pooled = self.dropout(z1_pooled)
        z2 = ibp.activation(F.relu, self.linear_hidden(z1_pooled))  # B, h
        z2 = self.dropout(z2)
        output = self.linear_output(z2)  # B, 1
        return output


def load_datasets(device, opts):
    """
  Loads text classification datasets given opts on the device and returns the dataset.
  If a data cache is specified in opts and the cached data there is of the same class
    as the one specified in opts, uses the cache. Otherwise reads from the raw dataset
    files specified in OPTS.
  Returns:
    - train_data:  EntailmentDataset - Processed training dataset
    - dev_data: Optional[EntailmentDataset] - Processed dev dataset if raw dev data was found or
        dev_frac was specified in opts
    - word_mat: torch.Tensor
    - attack_surface: AttackSurface - defines the adversarial attack surface
  """
    data_class_dict = {'IMDB': IMDBDataset, 'YELP': YELPDataset}
    data_class = data_class_dict[opts.dataset]
    try:
        with open(
                os.path.join(opts.data_cache_dir,
                             opts.dataset + '_train_data.pkl'),
                'rb') as infile:
            train_data = pickle.load(infile)
            if not isinstance(train_data, data_class):
                raise Exception("Cached dataset of wrong class: {}".format(
                    type(train_data)))
        with open(
                os.path.join(opts.data_cache_dir,
                             opts.dataset + '_dev_data.pkl'), 'rb') as infile:
            dev_data = pickle.load(infile)
            if not isinstance(dev_data, data_class):
                raise Exception("Cached dataset of wrong class: {}".format(
                    type(train_data)))
        with open(
                os.path.join(opts.data_cache_dir,
                             opts.dataset + '_word_mat.pkl'), 'rb') as infile:
            word_mat = pickle.load(infile)
        with open(
                os.path.join(opts.data_cache_dir,
                             opts.dataset + '_attack_surface.pkl'),
                'rb') as infile:
            attack_surface = pickle.load(infile)
        print("Loaded data from {}.".format(opts.data_cache_dir))
    except Exception:
        attack_surface = attacks.WordSubstitutionAttackSurface.from_file(opts.neighbor_file)
        print('Reading dataset.')
        if opts.dataset == 'IMDB':
            raw_data = data_class.get_raw_data(
                opts.imdb_dir,
                test=opts.test)
        elif opts.dataset == 'YELP':
            raw_data = data_class.get_raw_data(
                opts.yelp_dir,
                test=opts.test)
        word_set = raw_data.get_word_set(attack_surface)
        vocab, word_mat = vocabulary.Vocabulary.read_word_vecs(
            word_set, opts.glove_dir, opts.glove, device)
        train_data = data_class.from_raw_data(
            raw_data.train_data,
            vocab,
            attack_surface,
            downsample_to=opts.downsample_to,
            downsample_shard=opts.downsample_shard,
            truncate_to=opts.truncate_to)
        dev_data = data_class.from_raw_data(
            raw_data.dev_data,
            vocab,
            attack_surface,
            downsample_to=opts.downsample_to,
            downsample_shard=opts.downsample_shard,
            truncate_to=opts.truncate_to)
        if opts.data_cache_dir:
            with open(
                    os.path.join(opts.data_cache_dir,
                                 opts.dataset + '_train_data.pkl'),
                    'wb') as outfile:
                pickle.dump(train_data, outfile)
            with open(
                    os.path.join(opts.data_cache_dir,
                                 opts.dataset + '_dev_data.pkl'),
                    'wb') as outfile:
                pickle.dump(dev_data, outfile)
            with open(
                    os.path.join(opts.data_cache_dir,
                                 opts.dataset + '_word_mat.pkl'),
                    'wb') as outfile:
                pickle.dump(word_mat, outfile)
            with open(
                    os.path.join(opts.data_cache_dir,
                                 opts.dataset + '_attack_surface.pkl'),
                    'wb') as outfile:
                pickle.dump(attack_surface, outfile)
            # with open(os.path.join(opts.data_cache_dir, 'attack_surface_id_dict.pkl'), 'wb') as outfile:
            #   pickle.dump(attack_surface_id_dict, outfile)
    return train_data, dev_data, word_mat, attack_surface


def load_datasets_v2(device, opts):
    """
  Loads text classification datasets given opts on the device and returns the dataset.
  If a data cache is specified in opts and the cached data there is of the same class
    as the one specified in opts, uses the cache. Otherwise reads from the raw dataset
    files specified in OPTS.
  Returns:
    - train_data:  EntailmentDataset - Processed training dataset
    - dev_data: Optional[EntailmentDataset] - Processed dev dataset if raw dev data was found or
        dev_frac was specified in opts
    - word_mat: torch.Tensor
    - attack_surface: AttackSurface - defines the adversarial attack surface
  """
    data_class_dict = {'IMDB': IMDBDataset, 'YELP': YELPDataset, 'SST2':SST2Dataset}
    data_class = data_class_dict[opts.dataset]
    try:
        with open(
                os.path.join(opts.data_cache_dir,
                             opts.dataset + '_train_data.pkl'),
                'rb') as infile:
            train_data = pickle.load(infile)
            if not isinstance(train_data, data_class):
                raise Exception("Cached dataset of wrong class: {}".format(
                    type(train_data)))
        with open(
                os.path.join(opts.data_cache_dir,
                             opts.dataset + '_dev_data.pkl'), 'rb') as infile:
            dev_data = pickle.load(infile)
            if not isinstance(dev_data, data_class):
                raise Exception("Cached dataset of wrong class: {}".format(
                    type(train_data)))
        with open(
                os.path.join(opts.data_cache_dir,
                             opts.dataset + '_word_mat.pkl'), 'rb') as infile:
            word_mat = pickle.load(infile)
        with open(
                os.path.join(opts.data_cache_dir,
                             opts.dataset + '_attack_surface.pkl'),
                'rb') as infile:
            attack_surface = pickle.load(infile)
        print("Loaded data from {}.".format(opts.data_cache_dir))
    except Exception:
        attack_surface = attacks.WordSubstitutionAttackSurface.from_file(opts.neighbor_file)
        print('Reading dataset.')
        if opts.dataset == 'IMDB':
            raw_data = data_class.get_raw_data(
                opts.imdb_dir,
                test=opts.test)
        elif opts.dataset == 'YELP':
            raw_data = data_class.get_raw_data(
                opts.yelp_dir,
                test=opts.test)
        elif opts.dataset == 'SST2':
            raw_data = data_class.get_raw_data(
                opts.sst2_dir,
                test=opts.test)
        if opts.vocab_size:
            vocab = init_dict(raw_data, opts.vocab_size)
            word_set = raw_data.get_word_set_v3(attack_surface, vocab)
            vocab, word_mat = vocabulary.Vocabulary.read_word_vecs(
                word_set, opts.glove_dir, opts.glove, device)
        else:
            word_set = raw_data.get_word_set_v2(attack_surface)
            vocab, word_mat = vocabulary.Vocabulary.read_word_vecs(
                word_set, opts.glove_dir, opts.glove, device)
        train_data = data_class.from_raw_data_v2(
            raw_data.train_data,
            vocab,
            attack_surface,
            downsample_to=opts.downsample_to,
            downsample_shard=opts.downsample_shard,
            truncate_to=opts.truncate_to,
            max_seq_length=opts.max_seq_length,
            gamma=opts.gamma)
        dev_data = data_class.from_raw_data_v2(
            raw_data.dev_data,
            vocab,
            attack_surface,
            downsample_to=opts.downsample_to,
            downsample_shard=opts.downsample_shard,
            truncate_to=opts.truncate_to,
            max_seq_length=opts.max_seq_length,
            gamma=0)
        if opts.data_cache_dir:
            with open(
                    os.path.join(opts.data_cache_dir,
                                 opts.dataset + '_train_data.pkl'),
                    'wb') as outfile:
                pickle.dump(train_data, outfile)
            with open(
                    os.path.join(opts.data_cache_dir,
                                 opts.dataset + '_dev_data.pkl'),
                    'wb') as outfile:
                pickle.dump(dev_data, outfile)
            with open(
                    os.path.join(opts.data_cache_dir,
                                 opts.dataset + '_word_mat.pkl'),
                    'wb') as outfile:
                pickle.dump(word_mat, outfile)
            with open(
                    os.path.join(opts.data_cache_dir,
                                 opts.dataset + '_attack_surface.pkl'),
                    'wb') as outfile:
                pickle.dump(attack_surface, outfile)
            # with open(os.path.join(opts.data_cache_dir, 'attack_surface_id_dict.pkl'), 'wb') as outfile:
            #   pickle.dump(attack_surface_id_dict, outfile)
    return train_data, dev_data, word_mat, attack_surface


def num_correct(model_output, targets):
    """
  Given the output of model and gold labels returns number of correct and certified correct
  predictions
  Args:
    - model_output: output of the model, could be ibp.IntervalBoundedTensor or torch.Tensor
    - targets: torch.Tensor, should be of size 1 per sample, 1 for positive 0 for negative
  Returns:
    - num_correct: int - number of correct predictions from the actual model output
    - num_cert_correct - number of bounds-certified correct predictions if the model_output was an
        IntervalBoundedTensor, 0 otherwise.
  """
    if isinstance(model_output, ibp.IntervalBoundedTensor):
        logits = model_output.val
        upper = model_output.ub
        lower = model_output.lb
        margin = upper - torch.gather(lower, 1, targets.view(-1, 1))
        margin = margin.scatter(1, targets.view(-1, 1), 0)
        predicted = torch.max(margin.data, 1)[1]
        num_cert_correct = (predicted == targets).float().sum().item()
    else:
        logits = model_output
        num_cert_correct = 0
    predicted = torch.max(logits.data, 1)[1]
    num_correct = (predicted == targets).float().sum().item()
    return num_correct, num_cert_correct


def load_model(word_mat, device, opts):
    """
  Try to load a model on the device given the word_mat and opts.
  Tries to load a model from the given or latest checkpoint if specified in the opts.
  Otherwise instantiates a new model on the device.
  """
    if opts.model == 'bow':
        model = BOWModel(vocabulary.GLOVE_CONFIGS[opts.glove]['size'],
                         opts.hidden_size,
                         word_mat,
                         pool=opts.pool,
                         dropout=opts.dropout_prob,
                         no_wordvec_layer=opts.no_wordvec_layer).to(device)
    elif opts.model == 'cnn':
        model = CNNModel(
            vocabulary.GLOVE_CONFIGS[opts.glove]['size'],
            opts.hidden_size,
            opts.kernel_size,
            word_mat,
            pool=opts.pool,
            dropout=opts.dropout_prob,
            no_wordvec_layer=opts.no_wordvec_layer,
            early_ibp=opts.early_ibp,
            relu_wordvec=not opts.no_relu_wordvec,
            unfreeze_wordvec=opts.unfreeze_wordvec).to(device)
    elif opts.model == 'textcnn':
        model = TextCNNModel(
            vocabulary.GLOVE_CONFIGS[opts.glove]['size'],
            opts.hidden_size,
            opts.kernel_size,
            word_mat,
            pool=opts.pool,
            dropout=opts.dropout_prob,
            no_wordvec_layer=opts.no_wordvec_layer,
            early_ibp=opts.early_ibp,
            relu_wordvec=not opts.no_relu_wordvec,
            unfreeze_wordvec=opts.unfreeze_wordvec).to(device)
    if opts.load_dir:
        try:
            if opts.load_ckpt is None:
                load_fn = sorted(
                    glob.glob(
                        os.path.join(opts.load_dir, 'model-checkpoint-[0-9]+.pth')))[-1]
                print(load_fn)
                exit()
            else:
                load_fn = os.path.join(opts.load_dir, 'model-checkpoint-%d.pth' % opts.load_ckpt)
            print('Loading model from %s.' % load_fn)
            state_dict = dict(torch.load(load_fn))
            # state_dict['embs.weight'] = model.embs.weight
            model.load_state_dict(state_dict)
            print('Finished loading model.')
        except Exception as ex:
            print("Couldn't load model, starting anew: {}".format(ex))
    return model


class RawClassificationDataset(data_util.RawDataset):
    """
  Dataset that only holds x,y as (str, str) tuples
  """

    def get_word_set(self, attack_surface):
		# use vocabulary of counter-fitted word embeddings
        with open(COUNTER_FITTED_FILE) as f:
            counter_vocab = set([line.split(' ')[0] for line in f])
        word_set = set()
        for x, y in self.data:
            words = [w.lower() for w in x.split(' ')]
            for w in words:
                word_set.add(w)
            try:
                swaps = attack_surface.get_swaps(words)
                for cur_swaps in swaps:
                    for w in cur_swaps:
                        word_set.add(w)
            except KeyError:
                # For now, ignore things not in attack surface
                # If we really need them, later code will throw an error
                pass
        return word_set & counter_vocab

    def get_word_set_v2(self, attack_surface):
		# use vocabulary of glove word embeddings
        word_set = set()
        for x, y in self.data:
            words = [w.lower() for w in x.split(' ')]
            for w in words:
                word_set.add(w)
            try:
                swaps = attack_surface.get_swaps(words)
                for cur_swaps in swaps:
                    for w in cur_swaps:
                        word_set.add(w)
            except KeyError:
                # For now, ignore things not in attack surface
                # If we really need them, later code will throw an error
                pass
        return word_set

    def get_word_set_v3(self, attack_surface, vocab):
		# keep words with vocab_size constraint
        raise NotImplementedError
        with open(COUNTER_FITTED_FILE) as f:
            counter_vocab = set([line.split(' ')[0] for line in f])
        word_set = set()
        for x, y in self.data:
            words = [w.lower() for w in x.split(' ')]
            for w in words:
                if w in vocab or w in counter_vocab:
                    word_set.add(w)
            try:
                swaps = attack_surface.get_swaps(words)
                for cur_swaps in swaps:
                    for w in cur_swaps:
                        word_set.add(w)
            except KeyError:
                # For now, ignore things not in attack surface
                # If we really need them, later code will throw an error
                pass
        return word_set


class TextClassificationDataset(data_util.ProcessedDataset):
    """
  Dataset that holds processed example dicts
  """

    @classmethod
    def from_raw_data(cls,
                      raw_data,
                      vocab,
                      attack_surface=None,
                      truncate_to=None,
                      downsample_to=None,
                      downsample_shard=0):
    # delete words without synonym
        if downsample_to:
            raw_data = raw_data[downsample_shard *
                                downsample_to:(downsample_shard + 1) *
                                downsample_to]
        examples = []
        for x, y in raw_data:
            all_words = [w.lower() for w in x.split()]
            if attack_surface:
                all_swaps = attack_surface.get_swaps(all_words)
                words = [w for w in all_words if w in vocab]
                swaps = [s for w, s in zip(all_words, all_swaps) if w in vocab]
                choices = [[w] + cur_swaps
                           for w, cur_swaps in zip(words, swaps)]
            else:
                words = [w for w in all_words
                         if w in vocab]  # Delete UNK words
            if truncate_to:
                words = words[:truncate_to]
            word_idxs = [vocab.get_index(w) for w in words]
            x_torch = torch.tensor(word_idxs).view(1, -1, 1)  # (1, T, d)
            if attack_surface:
                choices_word_idxs = [
                    torch.tensor([vocab.get_index(c) for c in c_list],
                                 dtype=torch.long) for c_list in choices
                ]
                if any(0 in c.view(-1).tolist() for c in choices_word_idxs):
                    raise ValueError("UNK tokens found")
                choices_torch = pad_sequence(
                    choices_word_idxs,
                    batch_first=True).unsqueeze(2).unsqueeze(0)  # (1, T, C, 1)
                choices_mask = (choices_torch.squeeze(-1) !=
                                0).long()  # (1, T, C)
            else:
                choices_torch = x_torch.view(1, -1, 1, 1)  # (1, T, 1, 1)
                choices_mask = torch.ones_like(x_torch.view(1, -1, 1))
            mask_torch = torch.ones((1, len(word_idxs)))
            x_bounded = ibp.DiscreteChoiceTensor(x_torch, choices_torch,
                                                 choices_mask, mask_torch)
            y_torch = torch.tensor(y, dtype=torch.float).view(1, 1)
            lengths_torch = torch.tensor(len(word_idxs)).view(1)
            examples.append(
                dict(x=x_bounded,
                     y=y_torch,
                     mask=mask_torch,
                     lengths=lengths_torch))
        return cls(raw_data, vocab, examples)

    @classmethod
    def from_raw_data_v2(cls,
                         raw_data,
                         vocab,
                         attack_surface=None,
                         truncate_to=None,
                         downsample_to=None,
                         downsample_shard=0,
                         max_seq_length=10000,
                         gamma=0):
	# keep all words
        with open(COUNTER_FITTED_FILE) as f:
            counter_vocab = set([line.split(' ')[0] for line in f])
        print('max_seq_length: %d' % max_seq_length)
        if downsample_to:
            raw_data = raw_data[downsample_shard *
                                downsample_to:(downsample_shard + 1) *
                                downsample_to]
        examples = []
        for x, y in raw_data:
            all_words = [w.lower() for w in x.split()]
            if len(all_words) > max_seq_length - 2:
                all_words = all_words[:(max_seq_length - 2)]
            if attack_surface:
                all_swaps = attack_surface.get_swaps(all_words)
                words = [w for w in all_words]
                swaps = [s for w, s in zip(all_words, all_swaps)]
                choices=[]
                for w, cur_swaps in zip(words, swaps):
                    if gamma > 0 and len(cur_swaps) > 0:
                        random.shuffle(cur_swaps)
                        choices.append([w] + cur_swaps[:math.ceil(len(cur_swaps)*0.3)])
                    else:
                        choices.append([w] + cur_swaps)
            else:
                words = [w for w in all_words if w in vocab]  # Delete UNK words
            if truncate_to:
                words = words[:truncate_to]
            word_idxs = [vocab.get_index(w) for w in words]
            x_torch = torch.tensor(word_idxs).view(1, -1, 1)  # (1, T, d)
            if attack_surface:
                choices_word_idxs = [torch.tensor([vocab.get_index(c) for c in c_list], dtype=torch.long) for c_list in choices]
                choices_mask = [torch.tensor([1 for c in c_list], dtype=torch.long) for c_list in choices]
                # if any(0 in c.view(-1).tolist() for c in choices_word_idxs):
                #   raise ValueError("UNK tokens found")
                choices_torch = pad_sequence(
                    choices_word_idxs,
                    batch_first=True).unsqueeze(2).unsqueeze(0)  # (1, T, C, 1)
                choices_mask = pad_sequence(
                    choices_mask,
                    batch_first=True).unsqueeze(2).unsqueeze(0).squeeze(-1)
                # choices_mask = (choices_torch.squeeze(-1) != 0).long()  # (1, T, C)
            else:
                choices_torch = x_torch.view(1, -1, 1, 1)  # (1, T, 1, 1)
                choices_mask = torch.ones_like(x_torch.view(1, -1, 1))
            mask_torch = torch.ones((1, len(word_idxs)))
            x_bounded = ibp.DiscreteChoiceTensor(x_torch, choices_torch,
                                                 choices_mask, mask_torch)
            y_torch = torch.tensor(y, dtype=torch.float).view(1, 1)
            lengths_torch = torch.tensor(len(word_idxs)).view(1)
            examples.append(
                dict(x=x_bounded,
                     y=y_torch,
                     mask=mask_torch,
                     lengths=lengths_torch))
        return cls(raw_data, vocab, examples)

    @staticmethod
    def example_len(example):
        return example['x'].shape[1]

    @staticmethod
    def collate_examples(examples):
        """
    Turns a list of examples into a workable batch:
    """
        if len(examples) == 1:
            return examples[0]
        B = len(examples)
        max_len = max(ex['x'].shape[1] for ex in examples)
        x_vals = []
        choice_mats = []
        choice_masks = []
        y = torch.zeros((B, 1))
        lengths = torch.zeros((B, ), dtype=torch.long)
        masks = torch.zeros((B, max_len))
        for i, ex in enumerate(examples):
            x_vals.append(ex['x'].val)
            choice_mats.append(ex['x'].choice_mat)
            choice_masks.append(ex['x'].choice_mask)
            cur_len = ex['x'].shape[1]
            masks[i, :cur_len] = 1
            y[i, 0] = ex['y']
            lengths[i] = ex['lengths'][0]
        x_vals = data_util.multi_dim_padded_cat(x_vals, 0).long()
        choice_mats = data_util.multi_dim_padded_cat(choice_mats, 0).long()
        choice_masks = data_util.multi_dim_padded_cat(choice_masks, 0).long()
        return {
            'x': ibp.DiscreteChoiceTensor(x_vals, choice_mats, choice_masks,
                                          masks),
            'y': y,
            'mask': masks,
            'lengths': lengths
        }


def init_dict(raw_data, vocab_size=0):
    """
    The most frequently occurring words in the data set constitute the dictionary.
    Words that do not appear in the dictionary are all mapped to `UNK` with word id 0.
    """
    tokenizer = Tokenizer()
    train_text = [x for x, y in raw_data.train_data]
    tokenizer.fit_on_texts(train_text)
    dic = dict()
    dic["UNK"] = 0
    for word, idx in tokenizer.word_index.items():
        if not vocab_size or idx <= vocab_size:
            dic[word] = idx
    print(len(dic))
    return dic


class IMDBDataset(TextClassificationDataset):
    """
  Dataset that holds the IMDB sentiment classification data
  """

    @classmethod
    def read_text(cls, imdb_dir, split):
        if split == 'test':
            subdir = 'test'
        else:
            subdir = 'train'
        pos_path = os.path.join(imdb_dir, subdir + '/pos')
        neg_path = os.path.join(imdb_dir, subdir + '/neg')
        pos_files = [
            pos_path + '/' + x for x in os.listdir(pos_path)
            if x.endswith('.txt')
        ]
        neg_files = [
            neg_path + '/' + x for x in os.listdir(neg_path)
            if x.endswith('.txt')
        ]
        data = []
        num_words = 0
        for fn in tqdm(pos_files):
            label = 1
            with open(fn) as f:
                x_raw = f.readlines()[0].strip().replace('<br />', ' ')
                x_toks = word_tokenize(x_raw)
                num_words += len(x_toks)
                data.append((' '.join(x_toks), label))
        num_pos = sum(y for x, y in data)
        for fn in tqdm(neg_files):
            label = 0
            with open(fn) as f:
                x_raw = f.readlines()[0].strip().replace('<br />', ' ')
                x_toks = word_tokenize(x_raw)
                num_words += len(x_toks)
                data.append((' '.join(x_toks), label))
        num_neg = sum(1 - y for x, y in data)
        avg_words = num_words / len(data)
        print('Read %d examples (+%d/-%d), average length %d words' %
                (len(data), num_pos, num_neg, avg_words))
        random.shuffle(data)
        return data

    @classmethod
    def get_raw_data(cls, imdb_dir, test=False):
        train_data = cls.read_text(imdb_dir, 'train')
        test_data = cls.read_text(imdb_dir, 'test')
        return RawClassificationDataset(train_data, test_data)


class YELPDataset(TextClassificationDataset):
    """
    Dataset that holds the YELP classification data
    """

    @classmethod
    def read_text(cls, yelp_dir, split):
        data = []
        path = os.path.join(yelp_dir, 'sentiment.' + split)
        num_words = 0
        with open(path + '.0') as f:
            for line in f:
                x_raw = line.strip().replace('<br />', ' ')
                x_toks = word_tokenize(x_raw)
                num_words += len(x_toks)
                data.append((' '.join(x_toks), 0))
        with open(path + '.1') as f:
            for line in f:
                x_raw = line.strip().replace('<br />', ' ')
                x_toks = word_tokenize(x_raw)
                num_words += len(x_toks)
                data.append((' '.join(x_toks), 1))
        num_pos = sum(y for x, y in data)
        num_neg = sum(1 - y for x, y in data)
        avg_words = num_words / len(data)
        print('Read %d examples (+%d, -%d), average length %d words' %
              (len(data), num_pos, num_neg, avg_words))
        return data

    @classmethod
    def get_raw_data(cls, yelp_dir, test=False):
        print('Processing Yelp dataset')
        train_data = cls.read_text(yelp_dir, 'train')
        if test:
            dev_data = cls.read_text(yelp_dir, 'test')
        else:
            dev_data = cls.read_text(yelp_dir, 'dev')
        return RawClassificationDataset(train_data, dev_data)


class SST2Dataset(TextClassificationDataset):
    """
    Dataset that holds the SST2 classification data
    """

    @classmethod
    def read_text(cls, sst2_dir, split):
        data = []
        with open(sst2_dir + "/%s.tsv" % split, "r", encoding="utf-8") as fp:
            lines = fp.readlines()[1:]
            for line in lines:
                line = line.split('\t')
                label = int(line[1])
                x_raw = line[0].lower().strip().replace('<br />', ' ')
                x_toks = word_tokenize(x_raw)
                data.append((' '.join(x_toks), label))

        return data

    @classmethod
    def get_raw_data(cls, sst2_dir, test=False):
        print('Processing SST2 dataset')
        train_data = cls.read_text(sst2_dir, 'train')
        test_data = cls.read_text(sst2_dir, 'dev')
        return RawClassificationDataset(train_data, test_data)