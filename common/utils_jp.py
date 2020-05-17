# coding: utf-8
import sys
sys.path.append('..')
import os
import re
from common.np import *
from janome.tokenizer import Tokenizer

def preprocess(text, custom_dic=None, skip_symbol=True):
  text = text.lower()
  text = re.sub("[\f\n\r\t\v]","", text)
  text = re.sub("　","", text)
  if skip_symbol:
    text = re.sub("…","", text)
    text = re.sub("・・・","", text)
    text = re.sub("。","", text)
    text = re.sub("！","", text)
    text = re.sub("？","", text)
    text = re.sub("[「」]","", text)
    text = re.sub("[【】]","", text)
  text = [re.sub("[（）]","", text)]  

  if custom_dic != None:
    t = Tokenizer(custom_dic, udic_type="simpledic", udic_enc="utf8")
  else:
    t = Tokenizer()

  words_list = []
  for word in text:
    token_list = t.tokenize(word, wakati=True)
    token_list = ["N" if x.isdigit() else x for x in token_list]
    token_list = ["ﾜﾗ" if x == "ｗ" else x for x in token_list]
    token_list = ["ﾜﾗ" if x == "ｗｗ" else x for x in token_list]
    token_list.remove(' ')
    words_list.append(token_list)
  
  word_to_id = {}
  id_to_word = {}

  for words in words_list:
    for word in words:
      if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word
  
  corpus = np.array([word_to_id[w] for w in words])

  return corpus, word_to_id, id_to_word