import sys
sys.path.append("..")
from common.time_layers import *
from seq2seq.seq2seq import Seq2seq, Encoder
from attention.attention_layer import TimeAttention

class AttentionEncoder(Encoder):
  def forward(self, xs):
    xs = self.embed.forward(xs)
    hs = self.lstm.forward(xs)
    return hs
  
  def backward(self, dhs):
    dout = self.lstm.backward(dhs)
    dout = self.embed.backward(dout)
    return dout

class AttentionDecoder:
  def __init__(self, vocab_size, wordvec_size, hidden_size):
    V, D, H = vocab_size, wordvec_size, hidden_size

    embed_W = (0.01 * np.random.randn(V, D)).astype("f")    

    lstm_Wx1 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
    lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b1 = np.zeros(4 * H).astype("f")

    affine_W = (np.random.randn(2 * H, V) / np.sqrt(2 * H)).astype("f")
    affine_b = np.zeros(V).astype("f")

    self.embed = TimeEmbedding(embed_W)
    self.lstm = TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True)
    self.attention = TimeAttention()
    self.affine = TimeAffine(affine_W, affine_b)
    layers = [self.embed, self.lstm, self.attention, self.affine]

    self.params, self.grads = [], []
    for layer in layers:
      self.params += layer.params
      self.grads += layer.grads
    
  def forward(self, xs, enc_hs):
    h = enc_hs[:, -1]
    self.lstm.set_state(h)

    out = self.embed.forward(xs)
    dec_hs = self.lstm.forward(out)
    c = self.attention.forward(enc_hs, dec_hs)
    out = np.concatenate([c, dec_hs], axis=2)
    score = self.affine.forward(out)

    return score
  
  def backward(self, dscore):
    dout = self.affine.backward(dscore)
    N, T, H2 = dout.shape
    H = H2 // 2

    dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
    denc_hs, ddec_hs1 = self.attention.backward(dc)
    ddec_hs = ddec_hs0 + ddec_hs1

    dout = self.lstm.backward(ddec_hs)

    dh = self.lstm.dh
    denc_hs[:, -1] += dh

    self.embed.backward(dout)
    return denc_hs
  
  def generate(self, enc_hs, start_id, sample_size):
    sampled = []
    sample_id = start_id
    h = enc_hs[:, -1]
    self.lstm.set_state(h)

    for _ in range(sample_size):
      x = np.array([sample_id]).reshape((1, 1))

      out = self.embed.forward(x)
      dec_hs = self.lstm.forward(out)
      c = self.attention.forward(enc_hs, dec_hs)
      out = np.concatenate([c, dec_hs], axis=2)
      score = self.affine.forward(out)

      sample_id = np.argmax(score.flatten())
      sampled.append(int(sample_id))
    
    return sampled

class AttentionSeq2seq(Seq2seq):
  def __init__(self, vocab_size, wordvec_size, hidden_size, vocab_size_decoder=None):
    VE, D, H = vocab_size, wordvec_size, hidden_size
    VD = vocab_size if vocab_size_decoder == None else vocab_size_decoder
    self.encoder = AttentionEncoder(VE, D, H)
    self.decoder = AttentionDecoder(VD, D, H)
    self.softmax = TimeSoftmaxWithLoss()

    self.params = self.encoder.params + self.decoder.params
    self.grads = self.encoder.grads + self.decoder.grads