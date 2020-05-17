import sys
sys.path.append("..")
from common import config
config.GPU = True

from gen_data import GenData
from common.optimizer import Adam
from common.trainer import Trainer
from eval_tools import eval_seq2seq
#from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
from common.gpu import to_cpu, to_gpu

d = GenData()
(x_train, t_train), (x_test, t_test) = d.load_corpus(file_name="data.txt")
word_to_id_q, word_to_id_a, id_to_word_q, id_to_word_a = d.get_vocab()

is_reverse = True
if is_reverse:
  x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# ハイパーパラメータ設定
vocab_size_x = len(word_to_id_q)
vocab_size_t = len(word_to_id_a)
wordvec_size = 100
hidden_size = 100
batch_size = 5
max_epoch = 300
max_grad = 5.0

model = PeekySeq2seq(vocab_size_x, wordvec_size, hidden_size, vocab_size_t)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
  print("epoch {}".format(epoch))
  x_train = to_gpu(x_train)
  t_train = to_gpu(t_train)
  trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

  correct_num = 0
  for i in range(len(x_test)):
    question, correct = x_test[[i]], t_test[[i]]
    verbose = i < 5
    correct_num += eval_seq2seq(model, question, correct, id_to_word_q, id_to_word_a, verbose, is_reverse)
  acc = float(correct_num) / len(x_test)
  acc_list.append(acc)
  print("val acc %.3f%%" % (acc * 100))