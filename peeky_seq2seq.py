import sys
sys.path.append("..")
from common.time_layers import *
from seq2seq import Seq2seq, Encoder

class PeekySeq2seq(Seq2seq):
  def __init__(self, vocab_size, wordvec_size, hidden_size, vocab_size_decoder=None):
    VE, D, H = vocab_size, wordvec_size, hidden_size
    VD = vocab_size if vocab_size_decoder == None else vocab_size_decoder
    self.encoder = Encoder(VE, D, H)
    self.decoder = PeekyDecoder(VD, D, H)
    self.softmax = TimeSoftmaxWithLoss()

    self.params = self.encoder.params + self.decoder.params
    self.grads = self.encoder.grads + self.decoder.grads

class PeekyDecoder:
  def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
    V, D, H = vocab_size, wordvec_size, hidden_size

    embed_W = (0.01 * np.random.randn(V, D)).astype("f")    

    # サイズはhと単語ベクトル
    lstm_Wx1 = (np.random.randn(H + D, 4 * H) / np.sqrt(H + D)).astype("f")
    lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b1 = np.zeros(4 * H).astype("f")

    # サイズはhとLSTM後の
    affine_W = (np.random.randn(H + H, V) / np.sqrt(H + H)).astype("f")
    affine_b = np.zeros(V).astype("f")

    self.embed = TimeEmbedding(embed_W)
    self.lstm = TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True)
    self.affine = TimeAffine(affine_W, affine_b)

    self.params, self.grads = [], []

    for layer in (self.embed, self.lstm, self.affine):
      self.params += layer.params
      self.grads += layer.grads
    self.cache = None
  
  def forward(self, xs, h, train_flg=True):
    N, T = xs.shape
    N, H = h.shape
    self.lstm.set_state(h)

    out = self.embed.forward(xs)

    # LSTMはタイムレイヤーなので、hそのままではできず、時系列サイズまでに拡張
    hs = np.repeat(h, T, axis=0).reshape(N, T, H)
    # そのままEmbed後のと結合
    out = np.concatenate((hs, out), axis=2)
    out = self.lstm.forward(out)

    # そのままLSTM後のと結合
    out = np.concatenate((hs, out), axis=2)
    score = self.affine.forward(out)

    self.cache = H
    return score
  
  def backward(self, dscore):
    H = self.cache

    dout = self.affine.backward(dscore)
    # 結合の逆伝播なので分割、dhs0はdh計算用
    dout, dhs0 = dout[:, :, H:], dout[:, :, :H]

    dout = self.lstm.backward(dout)
    # 結合の逆伝播なので分割、dhs1はdh計算用
    dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]

    self.embed.backward(dembed)

    dhs = dhs0 + dhs1
    dh = self.lstm.dh + np.sum(dhs, axis=1)

    return dh
  
  def generate(self, h, start_id, sample_size):
    sampled = []
    sample_id = start_id
    self.lstm.set_state(h)

    H = h.shape[1]
    # 学習ではなく単純に予測なのでNは1、Tも1
    peeky_h = h.reshape(1, 1, H)
    for _ in range(sample_size):
      x = np.array(sample_id).reshape((1, 1))
      out = self.embed.forward(x)
      out = np.concatenate((peeky_h, out), axis=2)
      out = self.lstm.forward(out)
      out = np.concatenate((peeky_h, out), axis=2)
      score = self.affine.forward(out)

      sample_id = np.argmax(score.flatten())
      sampled.append(int(sample_id))
    
    return sampled