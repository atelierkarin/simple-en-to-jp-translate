import sys
sys.path.append("..")
from common.time_layers import *
from common.base_model import BaseModel

class Seq2seq(BaseModel):
  def __init__(self, vocab_size, wordvec_size, hidden_size, vocab_size_decoder=None):
    VE, D, H = vocab_size, wordvec_size, hidden_size
    VD = vocab_size if vocab_size_decoder == None else vocab_size_decoder
    self.encoder = Encoder(VE, D, H)
    self.decoder = Decoder(VD, D, H)
    self.softmax = TimeSoftmaxWithLoss()

    self.params = self.encoder.params + self.decoder.params
    self.grads = self.encoder.grads + self.decoder.grads
  
  def forward(self, xs, ts):
    decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

    h = self.encoder.forward(xs)
    score = self.decoder.forward(decoder_xs, h)
    loss = self.softmax.forward(score, decoder_ts)

    return loss
  
  def backward(self, dout=1):
    dout = self.softmax.backward(dout)
    dh = self.decoder.backward(dout)
    dout = self.encoder.backward(dh)
    return dout
  
  def generate(self, xs, start_id, sample_size):
    h = self.encoder.forward(xs)
    sampled = self.decoder.generate(h, start_id, sample_size)
    return sampled

class Encoder:
  def __init__(self, vocab_size, wordvec_size, hidden_size):
    V, D, H = vocab_size, wordvec_size, hidden_size

    embed_W = (0.01 * np.random.randn(V, D)).astype("f")

    lstm_Wx1 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
    lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b1 = np.zeros(4 * H).astype("f")

    self.embed = TimeEmbedding(embed_W)
    self.lstm = TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=False)

    self.params = self.embed.params + self.lstm.params
    self.grads = self.embed.grads + self.lstm.grads
    self.hs = None
  
  def forward(self, xs):
    xs = self.embed.forward(xs)
    hs = self.lstm.forward(xs)
    self.hs = hs

    # 最後のタイムレイヤーのhなので-1
    return hs[:, -1, :]
  
  def backward(self, dh):
    # dhは最後のタイムレイヤーのdhなので-1
    dhs = np.zeros_like(self.hs)
    dhs[:, -1, :] = dh

    dout = self.lstm.backward(dhs)
    dout = self.embed.backward(dout)
    return dout

class Decoder:
  def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
    V, D, H = vocab_size, wordvec_size, hidden_size

    embed_W = (0.01 * np.random.randn(V, D)).astype("f")    

    lstm_Wx1 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
    lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b1 = np.zeros(4 * H).astype("f")

    affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype("f")
    affine_b = np.zeros(V).astype("f")

    self.layers = [
      TimeEmbedding(embed_W),
      TimeDropout(dropout_ratio),
      TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
      TimeDropout(dropout_ratio),
      TimeAffine(affine_W, affine_b)
    ]

    self.embed = self.layers[0]
    self.lstm = [self.layers[x] for x in [2]]
    self.drops = [self.layers[x] for x in [1,3]]
    self.affine = self.layers[4]

    self.params, self.grads = [], []

    for layer in self.layers:
      self.params += layer.params
      self.grads += layer.grads
  
  def predict(self, xs, train_flg=False):
    for layer in self.drops:
      layer.train_flg = train_flg
    for layer in self.layers:
      xs = layer.forward(xs)
    return xs
  
  def forward(self, xs, h, train_flg=True):
    self.lstm[0].set_state(h)
    score = self.predict(xs, train_flg)
    return score
  
  def backward(self, dout):
    for layer in reversed(self.layers):
      dout = layer.backward(dout)
    dh = self.lstm[0].dh

    return dh
  
  def generate(self, h, start_id, sample_size):
    sampled = []
    sample_id = start_id
    self.lstm[0].set_state(h)

    for _ in range(sample_size):
      x = np.array(sample_id).reshape((1, 1))
      score = self.predict(x)

      sample_id = np.argmax(score.flatten())
      sampled.append(int(sample_id))
    
    return sampled