import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
import scipy.signal as signal
import soundfile as sf

def convolution(input, filter):
    """
    Do the convolution.

    Parameters
    -------------
    input[array] : 入力データ
    filter[array] : filter

    Returns
    ------------
    output[array] : 出力
    """
    # 畳み込みを行う(https://data-analytics.fun/2021/11/23/understanding-convolution/)
    output = np.zeros(len(input)+len(filter)-1)
    # inputの配列の前後に0を追加。
    add = np.zeros(len(filter)-1)
    input = np.concatenate((add, input))
    input = np.concatenate((input, add))
    # フィルタ配列を反転
    filter = filter[::-1]
    # 今回は一次元だから内積でいい？
    for i in range(len(output)):
        output[i] = np.dot(input[i:i+len(filter)],filter)
    return output


def main():
    # ハイパスフィルタを実装
    input = np.array([1,2,3,4,5,6,7,8,9,10])
    filter = np.array([1,2,3,4])
    a = convolution(input,filter)
    print(a)


if __name__ == "__main__":
    main()
