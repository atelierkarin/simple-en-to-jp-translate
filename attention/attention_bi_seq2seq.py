import sys
sys.path.append("..")
from common.time_layers import *
from seq2seq.seq2seq import Seq2seq
from attention.attention_layer import TimeAttention
from attention.attention_seq2seq import AttentionDecoder

# 双方向RNN + 単方向RNN
class AttentionBiEncoder:
  def __init__(self, vocab_size, wordvec_size, hidden_size):
    V, D, H = vocab_size, wordvec_size, hidden_size

    embed_W = (0.01 * np.random.randn(V, D)).astype("f")

    lstm_Wx1 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
    lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b1 = np.zeros(4 * H).astype("f")

    lstm_Wx2 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
    lstm_Wh2 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b2 = np.zeros(4 * H).astype("f")

    lstm_Wx3 = (np.random.randn(2 * H, 4 * H) / np.sqrt(2 * H)).astype("f")
    lstm_Wh3 = (np.random.randn(H, 4 * H) / np.sqrt(2 * H)).astype("f")
    lstm_b3 = np.zeros(4 * H).astype("f")

    self.embed = TimeEmbedding(embed_W)
    self.lstm1 = TimeBiLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, lstm_Wx2, lstm_Wh2, lstm_b2, stateful=False)
    self.lstm2 = TimeLSTM(lstm_Wx3, lstm_Wh3, lstm_b3, stateful=False)

    self.params = self.embed.params + self.lstm1.params + self.lstm2.params
    self.grads = self.embed.grads + self.lstm1.grads + self.lstm2.grads
    self.hs = None
  
  def forward(self, xs):
    xs = self.embed.forward(xs)
    hs = self.lstm1.forward(xs)
    hs = self.lstm2.forward(hs)
    return hs
  
  def backward(self, dhs):
    dout = self.lstm2.backward(dhs)
    dout = self.lstm1.backward(dout)
    dout = self.embed.backward(dout)
    return dout

class AttentionBiDecoder:
  def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
    V, D, H = vocab_size, wordvec_size, hidden_size

    embed_W = (0.01 * np.random.randn(V, D)).astype("f")    

    lstm_Wx1 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
    lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b1 = np.zeros(4 * H).astype("f")

    affine_W1 = (np.random.randn(2 * H, H) / np.sqrt(2 * H)).astype("f")
    affine_b1 = np.zeros(H).astype("f")

    lstm_Wx2 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_Wh2 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b2 = np.zeros(4 * H).astype("f")

    affine_W2 = (np.random.randn(2 * H, V) / np.sqrt(2 * H)).astype("f")
    affine_b2 = np.zeros(V).astype("f")

    self.embed = TimeEmbedding(embed_W)
    self.lstm1 = TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True)
    self.lstm2 = TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True)
    self.attention1 = TimeAttention()
    self.attention2 = TimeAttention()
    self.affine1 = TimeAffine(affine_W1, affine_b1)
    self.affine2 = TimeAffine(affine_W2, affine_b2)
    self.dropout1, self.dropout2, self.dropout3, self.dropout4, self.dropout5 = TimeDropout(), TimeDropout(), TimeDropout(), TimeDropout(), TimeDropout()
    # layers = [self.embed, self.lstm1, self.attention1, self.affine1, self.lstm2, self.attention2, self.affine2]
    layers = [self.embed, self.dropout1, self.lstm1, self.attention1, self.dropout2, self.affine1, self.dropout3, self.lstm2, self.attention2, self.dropout4, self.affine2, self.dropout5]

    self.params, self.grads = [], []
    for layer in layers:
      self.params += layer.params
      self.grads += layer.grads
    
  def forward(self, xs, enc_hs):
    h = enc_hs[:, -1]
    self.lstm1.set_state(h)
    self.lstm2.set_state(np.zeros(h.shape, dtype="f"))

    # Embedding
    out = self.embed.forward(xs)

    # Dropout 1
    out = self.dropout1.forward(out)

    # LSTM 1
    dec_hs_a = self.lstm1.forward(out)

    # Attention 1
    c_a = self.attention1.forward(enc_hs, dec_hs_a)

    # Dropout 2
    c_a = self.dropout2.forward(c_a)

    # Affine 1
    out = np.concatenate([c_a, dec_hs_a], axis=2)
    out = self.affine1.forward(out)

    # Dropout 3
    out = self.dropout3.forward(out)

    # LSTM 2
    dec_hs_b = self.lstm2.forward(out)
    
    # Attention 2
    c_b = self.attention2.forward(enc_hs, dec_hs_b)

    # Dropout 4
    c_b = self.dropout4.forward(c_b)

    # Affine 2
    out = np.concatenate([c_b, dec_hs_b], axis=2)
    score = self.affine2.forward(out)

    # Dropout 5
    score = self.dropout5.forward(score)

    return score
  
  def backward(self, dscore):

    # Dropout 5
    dscore = self.dropout5.backward(dscore)

    # Affine 2
    dout = self.affine2.backward(dscore)
    N, T, H2 = dout.shape
    H = H2 // 2
    dc_b, ddec_hs_b0 = dout[:, :, :H], dout[:, :, H:]

    # Dropout 4
    dc_b = self.dropout4.backward(dc_b)

    # Attention 2
    denc_hs0, ddec_hs_b1 = self.attention2.backward(dc_b)
    ddec_hs_b = ddec_hs_b0 + ddec_hs_b1

    # LSTM 2
    # print("Backward ddec_hs_b size = {}".format(ddec_hs_b.shape))
    dout = self.lstm2.backward(ddec_hs_b)

    # Dropout 3
    dout = self.dropout3.backward(dout)

    # Affine 1
    dout = self.affine1.backward(dout)
    dc_a, ddec_hs_a0 = dout[:, :, :H], dout[:, :, H:]

    # Dropout 2
    dc_a = self.dropout2.backward(dc_a)

    # Attention 1
    denc_hs1, ddec_hs_a1 = self.attention1.backward(dc_a)    

    # LSTM 1
    ddec_hs_a = ddec_hs_a0 + ddec_hs_a1
    dout = self.lstm1.backward(ddec_hs_a)

    # Dropout 1
    dout = self.dropout1.backward(dout)

    # Embedding
    self.embed.backward(dout)

    dh = self.lstm1.dh
    denc_hs = denc_hs0 + denc_hs1
    denc_hs[:, -1] += dh

    return denc_hs
  
  def generate(self, enc_hs, start_id, sample_size):
    sampled = []
    sample_id = start_id
    h = enc_hs[:, -1]
    self.lstm1.set_state(h)
    self.lstm2.set_state(np.zeros(h.shape, dtype="f"))

    for _ in range(sample_size):
      x = np.array([sample_id]).reshape((1, 1))

      # Embedding
      out = self.embed.forward(x)

      # LSTM 1
      dec_hs_a = self.lstm1.forward(out)

      # Attention 1
      c_a = self.attention1.forward(enc_hs, dec_hs_a)

      # Affine 1
      out = np.concatenate([c_a, dec_hs_a], axis=2)
      out = self.affine1.forward(out)

      # LSTM 2
      dec_hs_b = self.lstm2.forward(out)
      
      # Attention 2
      c_b = self.attention2.forward(enc_hs, dec_hs_b)

      # Affine 2
      out = np.concatenate([c_b, dec_hs_b], axis=2)
      score = self.affine2.forward(out)

      sample_id = np.argmax(score.flatten())
      sampled.append(int(sample_id))
    
    return sampled

# 双方向RNN + 単方向RNN
class AttentionBiSeq2seq(Seq2seq):
  def __init__(self, vocab_size, wordvec_size, hidden_size, vocab_size_decoder=None):
    VE, D, H = vocab_size, wordvec_size, hidden_size
    VD = vocab_size if vocab_size_decoder == None else vocab_size_decoder
    self.encoder = AttentionBiEncoder(VE, D, H)
    #self.decoder = AttentionDecoder(VD, D, H)
    self.decoder = AttentionBiDecoder(VD, D, H)
    self.softmax = TimeSoftmaxWithLoss()

    self.params = self.encoder.params + self.decoder.params
    self.grads = self.encoder.grads + self.decoder.grads