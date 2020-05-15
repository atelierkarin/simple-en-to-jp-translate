import sys
sys.path.append("..")
from gen_data import GenData

from common.optimizer import Adam
from common.trainer import Trainer
from seq2seq import Seq2seq

def eval_seq2seq(model, question, correct, id_to_word_q, id_to_word_a, verbos=False, is_reverse=False):
  correct = correct.flatten()
  start_id = correct[0]
  correct = correct[1:]
  guess = model.generate(question, start_id, len(correct))

  if is_reverse:
    question = question[0]
    question = question[::-1]

  question = ' '.join([id_to_word_q[int(c)] for c in question.flatten()])
  correct = ''.join([id_to_word_a[int(c)] for c in correct])
  guess = ''.join([id_to_word_a[int(c)] for c in guess])

  print("{}: {} = {}? -> {}".format(question, correct, guess, guess == correct))

  return 1 if guess == correct else 0

d = GenData()
(x_train, t_train), (x_test, t_test) = d.load_data("data.txt", seed=1984)
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

model = Seq2seq(vocab_size_x, wordvec_size, hidden_size, vocab_size_t)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
  print("epoch {}".format(epoch))
  trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

  correct_num = 0
  for i in range(len(x_test)):
    question, correct = x_test[[i]], t_test[[i]]
    verbose = i < 5
    correct_num += eval_seq2seq(model, question, correct, id_to_word_q, id_to_word_a, verbose, is_reverse)
  acc = float(correct_num) / len(x_test)
  acc_list.append(acc)
  print("val acc %.3f%%" % (acc * 100))