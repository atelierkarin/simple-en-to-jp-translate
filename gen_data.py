import sys
sys.path.append('..')
import os
import io
import zipfile
import pickle

from common.np import *
from common.utils import preprocess
from common.utils_jp import preprocess as preprocess_jp

class GenData:
  def __init__(self):
    self.x_train, self.x_test, self.t_train, self.t_test = None, None, None, None
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
          if (phrase != []):
            res.append(phrase[:])
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
      answers.append(" ＿" + line[idx+1:])
    
    print("{} questions and {} answers found in text file.".format(len(questions), len(answers)))

    corpus_q, self.word_to_id_q, self.id_to_word_q = preprocess(" ".join(questions))
    corpus_a, self.word_to_id_a, self.id_to_word_a = preprocess_jp("".join(answers), skip_symbol=False)

    print("Corpus done. Len of corpus : {} questions and {} answers.".format(len(corpus_q), len(corpus_a)))

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
    
    print("x len, t len = {}, {}".format(len(x), len(t)))
    
    # shuffle
    indices = np.arange(len(x))
    if seed is not None:
      np.random.seed(seed)
    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 10% for validation set
    split_at = len(x) - len(x) // 10
    (self.x_train, self.x_test) = x[:split_at], x[split_at:]
    (self.t_train, self.t_test) = t[:split_at], t[split_at:]

    self.save_corpus(file_name)

    return (self.x_train, self.t_train), (self.x_test, self.t_test)

  def get_vocab(self):
    return self.word_to_id_q, self.word_to_id_a, self.id_to_word_q, self.id_to_word_a
  
  def unzip_and_gen_data(self, force=False):
    fname = "realdata.txt"
    is_data_exist = os.path.isfile(fname)

    if not is_data_exist or force:
      content = ""
      with zipfile.ZipFile("jpn-eng.zip") as myzip:
        with io.TextIOWrapper(myzip.open("jpn.txt"), encoding="utf-8") as f:
          for line in f.readlines():
            d = line.split("\t")
            content += d[0]+";"+d[1]+"\n"
      
      with open(fname, mode='w', encoding='utf-8') as f:
        f.write(content)
  
  def save_corpus(self, file_name="data.txt"):
    data = {
      "train_data": ((self.x_train, self.t_train), (self.x_test, self.t_test)),
      "vocab": (self.word_to_id_q, self.word_to_id_a, self.id_to_word_q, self.id_to_word_a)
    }
    pkl_filename = "corpus_" + file_name.replace(".txt", "") + ".pkl"

    with open(pkl_filename, 'wb') as f:
      pickle.dump(data, f)
  
  def load_corpus(self, file_name='data.txt', seed=1984):
    pkl_filename = "corpus_" + file_name.replace(".txt", "") + ".pkl"
    try:
      with open(pkl_filename, 'rb') as f:
        data = pickle.load(f)
      (self.x_train, self.t_train), (self.x_test, self.t_test) = data["train_data"]
      self.word_to_id_q, self.word_to_id_a, self.id_to_word_q, self.id_to_word_a = data["vocab"]
      return (self.x_train, self.t_train), (self.x_test, self.t_test)
    except FileNotFoundError:
      print("Pickle file not found, read text file...")
      return self.load_data(file_name=file_name, seed=seed)