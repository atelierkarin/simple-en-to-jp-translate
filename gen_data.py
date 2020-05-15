import sys
sys.path.append('..')
import os
from common.np import *
from common.utils import preprocess
from common.utils_jp import preprocess as preprocess_jp

class GenData:
  def __init__(self):
    self.word_to_id_q, self.id_to_word_q = {}, {}
    self.word_to_id_a, self.id_to_word_a = {}, {}

  def split_list_by_value(self, l, target=None, split_front=False):
    if target == None:
      return l[:]
    
    res = []
    phrase = []
    for idx, val in enumerate(l):
      if split_front:
        if val == target:
          if (phrase != []): res.append(phrase[:])
          phrase = []
        phrase.append(val)
      else:
        phrase.append(val)
        if val == target:
          if (phrase != []): res.append(phrase[:])
          phrase = []

    if (phrase != []): res.append(phrase[:])
    return res

  def load_data(self, file_name='data.txt', seed=1984):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name

    if not os.path.exists(file_path):
      print('No file: %s' % file_name)
      return None

    questions, answers = [], []

    for line in open(file_path, 'r', encoding='utf-8'):
      idx = line.find(';')
      questions.append(line[:idx] + " _")
      answers.append("＿" + line[idx+1:])

    corpus_q, self.word_to_id_q, self.id_to_word_q = preprocess(" ".join(questions))
    corpus_a, self.word_to_id_a, self.id_to_word_a = preprocess_jp("".join(answers), skip_symbol=False)

    if " " in self.word_to_id_q:
      empty_space_id_in_q = self.word_to_id_q[" "]
    else:
      empty_space_id_in_q = len(self.word_to_id_q)
      self.id_to_word_q[empty_space_id_in_q] = " "
      self.word_to_id_q[" "] = empty_space_id_in_q
    
    if " " in self.word_to_id_a:
      empty_space_id_in_a = self.word_to_id_a[" "]
    else:
      empty_space_id_in_a = len(self.word_to_id_a)
      self.id_to_word_a[empty_space_id_in_a] = " "
      self.word_to_id_a[" "] = empty_space_id_in_a

    corpus_questions = self.split_list_by_value(corpus_q, self.word_to_id_q["_"])
    corpus_answers = self.split_list_by_value(corpus_a, self.word_to_id_a["＿"], True)

    q_max = max(len(x) for x in corpus_questions)
    a_max = max(len(x) for x in corpus_answers)

    # create numpy array
    x = np.zeros((len(corpus_questions), q_max), dtype=np.int)
    x.fill(empty_space_id_in_q)
    t = np.zeros((len(corpus_answers), a_max), dtype=np.int)
    t.fill(empty_space_id_in_a)

    for i, sentence in enumerate(corpus_questions):
      for j, c in enumerate(sentence):
        x[i][j] = c
    for i, sentence in enumerate(corpus_answers):
      for j, c in enumerate(sentence):
        t[i][j] = c
    
    # shuffle
    indices = np.arange(len(x))
    if seed is not None:
      np.random.seed(seed)
    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 10% for validation set
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)

  def get_vocab(self):
    return self.word_to_id_q, self.word_to_id_a, self.id_to_word_q, self.id_to_word_a
