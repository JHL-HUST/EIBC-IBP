"""Defines an attack surface."""
import collections
import json
import sys
import torch
import numpy as np

OPTS = None

DEFAULT_MAX_LOG_P_DIFF = -5.0  # Maximum difference in log p for swaps.

class AttackSurface(object):
  def get_swaps(self, words):
    """Return valid substitutions for each position in input |words|."""
    raise NotImplementedError

class WordSubstitutionAttackSurface(AttackSurface):
  def __init__(self, neighbors):
    self.neighbors = neighbors

  @classmethod
  def from_file(cls, neighbors_file):
    with open(neighbors_file) as f:
      return cls(json.load(f))

  def get_swaps(self, words):
    swaps = []
    for i in range(len(words)):
      if words[i] in self.neighbors: 
        swaps.append(self.neighbors[words[i]])
      else:
        swaps.append([])
    return swaps
  
  def to_id_tensor_dict(self, vocab, device):
    res = dict()
    for word in vocab.word2index.keys():
      if word in self.neighbors:
        swaps = self.neighbors[word]
      else:
        swaps = []
      res[vocab.get_index(word)] = torch.tensor([vocab.get_index(s) for s in swaps], dtype=torch.long).to(device)
    return res
      

class LMConstrainedAttackSurface(AttackSurface):
  """WordSubstitutionAttackSurface with language model constraint."""
  def __init__(self, neighbors, lm_scores, min_log_p_diff=DEFAULT_MAX_LOG_P_DIFF):
    self.neighbors = neighbors
    self.lm_scores = lm_scores
    self.min_log_p_diff = min_log_p_diff

  @classmethod
  def from_files(cls, neighbors_file, lm_file):
    with open(neighbors_file) as f:
      neighbors = json.load(f)
    with open(lm_file) as f:
      lm_scores = {}
      cur_sent = None
      for line in f:
        toks = line.strip().split('\t')
        if len(toks) == 2:
          cur_sent = toks[1].lower()
          lm_scores[cur_sent] = collections.defaultdict(dict)
        else:
          word_idx, word, score = int(toks[1]), toks[2], float(toks[3])
          lm_scores[cur_sent][word_idx][word] = score
    return cls(neighbors, lm_scores)

  def get_swaps(self, words):
    swaps = []
    words = [word.lower() for word in words]
    s = ' '.join(words)
    if s not in self.lm_scores:
      raise KeyError('Unrecognized sentence "%s"' % s)
    for i in range(len(words)):
      if i in self.lm_scores[s]:
        cur_swaps = []
        orig_score = self.lm_scores[s][i][words[i]]
        for swap, score in self.lm_scores[s][i].items():
          if swap == words[i]: continue
          if swap not in self.neighbors[words[i]]: continue
          if score - orig_score >= self.min_log_p_diff:
            cur_swaps.append(swap)
        swaps.append(cur_swaps)
      else:
        swaps.append([])
    return swaps

class Adversary(object):
    """An Adversary tries to fool a model on a given example."""

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25, device=None):
        self.synonym_selector = synonym_selector
        self.target_model = target_model
        self.max_perturbed_percent = max_perturbed_percent
        self.device = device

    def run(self, model, dataset, device, opts=None):
        """Run adversary on a dataset.
        Args:
        model: a TextClassificationModel.
        dataset: a TextClassificationDataset.
        device: torch device.
        Returns: pair of
        - list of 0-1 adversarial loss of same length as |dataset|
        - list of list of adversarial examples (each is just a text string)
        """
        raise NotImplementedError

    def _softmax(self, x):
        orig_shape = x.shape
        if len(x.shape) > 1:
            _c_matrix = np.max(x, axis=1)
            _c_matrix = np.reshape(_c_matrix, [_c_matrix.shape[0], 1])
            _diff = np.exp(x - _c_matrix)
            x = _diff / np.reshape(np.sum(_diff, axis=1), [_c_matrix.shape[0], 1])
        else:
            _c = np.max(x)
            _diff = np.exp(x - _c)
            x = _diff / np.sum(_diff)
        assert x.shape == orig_shape
        return x

    def check_diff(self, sentence, perturbed_sentence):
        words = sentence
        perturbed_words = perturbed_sentence
        diff_count = 0
        if len(words) != len(perturbed_words):
            raise RuntimeError("Length changed after attack.")
        for i in range(len(words)):
            if words[i] != perturbed_words[i]:
                diff_count += 1
        return diff_count

class GAAdversary(Adversary):
    """  GA attack method.  """

    def __init__(self, synonym_selector, target_model, iterations_num=20, pop_max_size=60, max_perturbed_percent=0.25, device=None):
        super(GAAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)
        self.max_iters = iterations_num
        self.pop_size = pop_max_size
        self.temp = 0.3
        self.device = device

    def predict_batch(self, sentences): # Done
        x = sentences
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        tem, _ = self.target_model.query(x, None)
        tem = self._softmax(tem)
        return tem

    def predict(self, sentence): # Done
        x = [sentence]
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        tem, _ = self.target_model.query(x, None)
        tem = self._softmax(tem[0])
        return tem

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_best_replacement(self, pos, x_cur, x_orig, ori_label, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_x_list = [self.do_replace(
            x_cur, pos, w) if x_orig[pos] != w else x_cur for w in replace_list]
        new_x_preds = self.predict_batch(new_x_list)

        new_x_scores = 1 - new_x_preds[:, ori_label]
        orig_score = 1 - self.predict(x_cur)[ori_label]
        new_x_scores = new_x_scores - orig_score

        if (np.max(new_x_scores) > 0):
            return new_x_list[np.argsort(new_x_scores)[-1]]
        return x_cur
    
    def perturb(self, x_cur, x_orig, neigbhours, w_select_probs, ori_label):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(np.array(x_orig) != np.array(x_cur)) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        replace_list = neigbhours[rand_idx]
        return self.select_best_replacement(rand_idx, x_cur, x_orig, ori_label, replace_list)

    def generate_population(self, x_orig, neigbhours_list, w_select_probs, ori_label, pop_size):
        return [self.perturb(x_orig, x_orig, neigbhours_list, w_select_probs, ori_label) for _ in range(pop_size)]

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def check_return(self, perturbed_words, ori_words, ori_label):
        perturbed_text = perturbed_words
        clean_text = ori_words
        if self.check_diff(clean_text, perturbed_text) / len(ori_words) > self.max_perturbed_percent:
            return False, clean_text, ori_label
        else:
            x = torch.tensor([perturbed_text], dtype=torch.long).to(self.device)
            adv_label = self.target_model.query(x, [ori_label])[1][0]
            assert (adv_label != ori_label)
            return True, perturbed_text, adv_label

    def run(self, x_orig, neigbhours, ori_label):

        # x_orig = np.array(sentence.split())
        
        x_orig = x_orig.squeeze(-1).squeeze(0).tolist()
        neigbhours = neigbhours.squeeze(-1).squeeze(0)
        x_len = len(x_orig)
        neigbhours_list = []
        ori_label = int(ori_label.detach().cpu().item())
        for i in range(x_len):
            ns = neigbhours[i][1:]
            mask = ns.ne(0.0)
            ns = torch.masked_select(ns, mask)
            neigbhours_list.append(ns.tolist())
        
            
        neighbours_len = [len(x) for x in neigbhours_list]
        w_select_probs = []
        for pos in range(x_len):
            if neighbours_len[pos] == 0:
                w_select_probs.append(0)
            else:
                w_select_probs.append(min(neighbours_len[pos], 10))

        if np.sum(w_select_probs) == 0:
            return False, x_orig, ori_label
            
        w_select_probs = w_select_probs / np.sum(w_select_probs)

        pop = self.generate_population(
            x_orig, neigbhours_list, w_select_probs, ori_label, self.pop_size)
        for i in range(self.max_iters):
            pop_preds = self.predict_batch(pop)
            pop_scores = 1 - pop_preds[:, ori_label]

            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]

            logits = np.exp(pop_scores / self.temp)
            select_probs = logits / np.sum(logits)

            if np.argmax(pop_preds[top_attack, :]) != ori_label:
                return self.check_return(pop[top_attack], x_orig, ori_label)
            elite = [pop[top_attack]]  # elite
            # print(select_probs.shape)
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)

            childs = [self.crossover(pop[parent1_idx[i]],
                                     pop[parent2_idx[i]])
                      for i in range(self.pop_size-1)]
            childs = [self.perturb(
                x, x_orig, neigbhours_list, w_select_probs, ori_label) for x in childs]
            pop = elite + childs

        return False, x_orig, ori_label
