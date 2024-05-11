"""BPFの実装."""

import math
import wave
import matplotlib.pyplot as plt
import numpy as np

def convolution(x, h):
    """畳み込み演算をする.

    Args:
        x (_np.array[1]_): _入力信号_
        h (_np.array[1]_): _単位インパルス応答_

    Returns:
        _np.array[1]_: _畳み込み後の信号_
    """
    # 出力yの初期化
    y = np.zeros(len(x) + len(h) - 1, dtype = float)

    # xとhをゼロ詰めによって拡張
    hzero = np.hstack([h, np.zeros(len(x) - 1)])
    xzero = np.hstack([x, np.zeros(len(h) - 1)])

    # 重ね合わせ
    for n in range(0, len(y)):
        for k in range(0, n + 1):
            y[n] += hzero[k] * xzero[n - k]
    return y

def sinc(x):
    """sinc関数."""
    if x == 0.0:
        return 1.0
    else:
        return np.sin(x) / x

def createBSF(fe1, fe2, delta):
    """BSFのフィルタを設計する.

    Args:
        fe1 (_float_): _エッジ周波数(低)_
        fe2 (_float_): _エッジ周波数(高)_
        delta (_float_): _遷移帯域幅_

    Returns:
        _np.array[1]_: _単位インパルス応答_
    """

    # フィルタ係数の数を算出
    N = round(3.1 / delta) - 1
    # 係数の数を奇数にそろえる
    if (N + 1) % 2 == 0:
        N += 1
    N = int(N)

    # フィルタ係数を求める(逆変換)
    h = [sinc(math.pi * i) - 2 * fe2 * sinc(2 * math.pi * fe2 * i) + 2 *
         fe1 * sinc(2 * math.pi * fe1 * i) for i in range(-N / 2, N / 2 + 1)]

    # ハニング窓をかける
    hanningWindow = np.hanning(N + 1)
    for i in range(len(h)):
        h[i] *= hanningWindow[i]

    return h

if __name__ == '__main__':
    wf = wave.open("./audio/white-noise-44100hz.wav", "r")
    fs = wf.getframerate()
    byte = wf.getsampwidth()
    chn = wf.getnchannels()
    x_t = wf.readframes(-1)
    x = np.frombuffer(x_t, dtype="int16") / (2.0 ** 8) ** byte

    delta = 1000.0 / fs
    fe1 = 1000.0 / fs
    fe2 = 3000.0 / fs
    # BSFを設計
    h = createBSF(fe1, fe2, delta)
    np.fft(h, fs)

    # フィルタをかける
    y = convolution(x, h)