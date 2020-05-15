import sys
sys.path.append("..")
from common.np import *  # import numpy as np
from common.layers import *
from common.functions import sigmoid

class RNN:
  def __init__(self, Wx, Wh, b):
    self.params = [Wx, Wh, b]
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.cache = None
  
  def forward(self, x, h_prev):
    Wx, Wh, b = self.params
    t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
    h_next = np.tanh(t)

    self.cache = (x, h_prev, h_next)
    return h_next
  
  def backward(self, dh_next):
    Wx, Wh, b = self.params
    x, h_prev, h_next = self.cache

    dt = dh_next * (1 - h_next ** 2)
    db = np.sum(dt, axis=0)

    dWh = np.dot(h_prev.T, dt)
    dh_prev = np.dot(dt, Wh.T)

    dWx = np.dot(x.T, dt)
    dx = np.dot(dt, Wx.T)

    self.grads[0][...] = dWx
    self.grads[1][...] = dWh
    self.grads[2][...] = db

    return dx, dh_prev

class TimeRNN:
  def __init__(self, Wx, Wh, b, stateful=False):
    self.params = [Wx, Wh, b]
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.layers = None

    self.h, self.dh = None, None
    self.stateful = stateful

  def set_state(self, h):
    self.h = h
  
  def reset_state(self):
    self.h = None
  
  def forward(self, xs):
    Wx, Wh, b = self.params

    # N はパッチサイズ、Tは時系列データのサイズ、Dは入力ベクトルの次元数
    N, T, D = xs.shape
    D, H = Wx.shape

    self.layers = []
    hs = np.empty((N, T, H), dtype="f")

    if not self.stateful or self.h is None:
      self.h = np.zeros((N, H), dtype="f")
    
    for t in range(T):
      layer = RNN(*self.params)
      self.h = layer.forward(xs[:, t, :], self.h)
      hs[:, t, :] = self.h
      self.layers.append(layer)
    
    return hs
  
  def backward(self, dhs):
    Wx, Wh, b = self.params

    N, T, H = dhs.shape
    D, H = Wx.shape

    dxs = np.empty((N, T, D), dtype="f")
    dh = 0
    grads = [0, 0, 0]

    for t in reversed(range(T)):
      layer = self.layers[t]
      dx, dh = layer.backward(dhs[:, t, :] + dh)
      dxs[:, t, :] = dx

      for i, grad in enumerate(layer.grads):
        grads[i] += grad
    
    for i, grad in enumerate(grads):
      self.grads[i][...] = grad
    
    self.dh = dh
    
    return dxs

class TimeEmbedding:
  def __init__(self, W):
    self.params = [W]
    self.grads = [np.zeros_like(W)]
    self.layers = None
    self.W = W
  
  def forward(self, xs):
    N, T = xs.shape
    V, D = self.W.shape

    out = np.empty((N, T, D), dtype='f')
    self.layers = []

    for t in range(T):
      layer = Embedding(self.W)
      out[:, t, :] = layer.forward(xs[:, t])
      self.layers.append(layer)

    return out
  
  def backward(self, dout):
    N, T, D = dout.shape

    grad = 0
    for t in range(T):
      layer = self.layers[t]
      layer.backward(dout[:, t, :])
      grad += layer.grads[0]
    
    self.grads[0][...] = grad
    return None

class TimeAffine:
  def __init__(self, W, b):
    self.params = [W, b]
    self.grads = [np.zeros_like(W), np.zeros_like(b)]
    self.x = None
  
  def forward(self, x):
    N, T, D = x.shape
    W, b = self.params

    rx = x.reshape(N*T, -1)
    out = np.dot(rx, W) + b
    self.x = x
    return out.reshape(N, T, -1)
  
  def backward(self, dout):
    x = self.x
    N, T, D = x.shape
    W, b = self.params

    dout = dout.reshape(N*T, -1)
    rx = x.reshape(N*T, -1)

    db = np.sum(dout, axis=0)
    dW = np.dot(rx.T, dout)
    dx = np.dot(dout, W.T)

    dx = dx.reshape(*x.shape)

    self.grads[0][...] = dW
    self.grads[1][...] = db
    return dx

class TimeSoftmaxWithLoss:
  def __init__(self):
    self.params = []
    self.grads = []
    self.cache = None
    self.ignore_label = -1
  
  def forward(self, xs, ts):
    N, T, V = xs.shape

    if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
      ts = ts.argmax(axis=2)
    
    mask = (ts != self.ignore_label)

    # バッチ分と時系列分をまとめる（reshape）
    xs = xs.reshape(N * T, V)
    ts = ts.reshape(N * T)
    mask = mask.reshape(N * T)

    ys = softmax(xs)
    ls = np.log(ys[np.arange(N * T), ts])
    ls *= mask  # ignore_labelに該当するデータは損失を0にする
    loss = -np.sum(ls)
    loss /= mask.sum()

    self.cache = (ts, ys, mask, (N, T, V))
    return loss
  
  def backward(self, dout=1):
    ts, ys, mask, (N, T, V) = self.cache

    dx = ys
    dx[np.arange(N * T), ts] -= 1
    dx *= dout
    dx /= mask.sum()
    dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

    dx = dx.reshape((N, T, V))

    return dx

class LSTM:
  def __init__(self, Wx, Wh, b):
    self.params = [Wx, Wh, b]
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.cache = None
  
  def forward(self, x, h_prev, c_prev):
    Wx, Wh, b = self.params
    N, H = h_prev.shape

    # 4つ分の重みをまとめてアフィン変換
    A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

    # そしてそれぞれを分割
    f = A[:, :H]
    g = A[:, H:2*H]
    i = A[:, 2*H:3*H]
    o = A[:, 3*H:4*H]

    # ゲート計算
    f = sigmoid(f)
    g = np.tanh(g)
    i = sigmoid(i)
    o = sigmoid(o)

    # 伝播
    c_next = f * c_prev + g * i
    h_next = o * np.tanh(c_next)

    self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
    return h_next, c_next
  
  def backward(self, dh_next, dc_next):
    Wx, Wh, b = self.params
    x, h_prev, c_prev, i, f, g, o, c_next = self.cache

    tanh_c_next = np.tanh(c_next)

    # dh_nextから逆伝播
    dh_next_back1 = dh_next * o
    dh_next_back2 = dh_next_back1 * (1 - tanh_c_next ** 2)

    # dc_nextの逆伝播と合流
    ds = dh_next_back2 + dc_next

    # dc_prevを計算
    dc_prev = ds * f

    # ゲートそれぞれの逆伝播
    df = c_prev * ds
    dg = i * ds
    di = g * ds
    do = dh_next * tanh_c_next

    # 活性化関数の逆伝播
    df *= f * (1 - f)
    dg *= (1 - g ** 2)
    di *= i * (1 - i)
    do *= o * (1 - o)

    # 分割の逆伝播＝結合
    dA = np.hstack((df, dg, di, do))

    # アフィン変換の逆伝播
    dWh = np.dot(h_prev.T, dA)
    dh_prev = np.dot(dA, Wh.T)

    dWx = np.dot(x.T, dA)
    dx = np.dot(dA, Wx.T)

    db = np.sum(dA, axis=0)

    self.grads[0][...] = dWx
    self.grads[1][...] = dWh
    self.grads[2][...] = db

    return dx, dh_prev, dc_prev

class TimeLSTM:
  def __init__(self, Wx, Wh, b, stateful=True):
    self.params = [Wx, Wh, b]
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.layers = None

    self.h, self.c, self.dh = None, None, None
    self.stateful = stateful
  
  def forward(self, xs):
    Wx, Wh, b = self.params
    N, T, D = xs.shape
    H = Wh.shape[0]

    self.layers = []
    hs = np.empty((N, T, H), dtype="f")

    if not self.stateful or self.h is None:
      self.h = np.zeros((N, H), dtype="f")
    if not self.stateful or self.c is None:
      self.c = np.zeros((N, H), dtype="f")
    
    for t in range(T):
      layer = LSTM(*self.params)
      self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
      hs[:, t, :] = self.h

      self.layers.append(layer)
    
    return hs

  def backward(self, dhs):
    Wx, Wh, b = self.params
    N, T, H = dhs.shape
    D = Wx.shape[0]

    dxs = np.empty((N, T, D), dtype="f")
    dh, dc = 0, 0

    grads = [0, 0, 0]
    
    for t in reversed(range(T)):
      layer = self.layers[t]
      dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
      dxs[:, t, :] = dx

      for i, grad in enumerate(layer.grads):
        grads[i] += grad
    
    for i, grad in enumerate(grads):
      self.grads[i][...] = grad

    self.dh = dh
    return dxs

  def set_state(self, h, c=None):
    self.h, self.c = h, c
  
  def reset_state(self):
    self.h, self.c = None, None

class TimeDropout:
  def __init__(self, dropout_ratio=0.5):
    self.params, self.grads = [], []
    self.dropout_ratio = dropout_ratio
    self.mask = None
    self.train_flg = True

  def forward(self, xs):
    if self.train_flg:
      flg = np.random.rand(*xs.shape) > self.dropout_ratio
      scale = 1 / (1.0 - self.dropout_ratio)
      self.mask = flg.astype(np.float32) * scale
      return xs * self.mask
    else:
      return xs

  def backward(self, dout):
    return dout * self.mask