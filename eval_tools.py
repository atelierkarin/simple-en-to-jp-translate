import sys
sys.path.append("..")

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
  
  if verbos:
    print("{}: {} = {}? -> {}".format(question, correct, guess, guess == correct))

  return 1 if guess == correct else 0