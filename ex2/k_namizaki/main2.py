import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
import scipy.signal as signal
import soundfile as sf

def convolution(input, filter):
    """
    _summary_
    Do the convolution.

    Parameters
    -------------
    input : 入力データ
    filter : filter

    Returns
    ------------
    output : 出力
    """
    # 畳み込みを行う(https://data-analytics.fun/2021/11/23/understanding-convolution/)
    output = np.zeros(len(input)+len(filter)-1)
    # inputの配列の前後に0を追加。
    add = np.zeros(len(filter)-1)
    input = np.concatenate((add, input))
    input = np.concatenate((input, add))
    # フィルタ配列を反転
    filter = filter[::-1]
    # 今回は一次元だから内積でいい
    for i in range(len(output)):
        output[i] = np.dot(input[i:i+len(filter)],filter)
    return output


def hpf(fs, fc, N):
    """
    _summary_
    make the high pass filter.

    Parameters
    ----------
    fs : サンプル周波数
    fc : カットオフ周波数
    N : 次数

    Returns
    -------
    filter : フィルター
    """
    omega_c = 2.0 * np.pi * (fc/fs)
    window = signal.windows.hann(2*N+1)

    # 理想ハイパスフィルタを作成
    n = np.arange(1,N+1)
    ideal_fil = np.zeros(2*N+1)
    ideal_fil_half = -2 * (fc/fs) * np.sin(omega_c * n) / (omega_c * n)
    ideal_fil[0:N] = ideal_fil_half[::-1]
    ideal_fil[N] = 1 - 2 * (fc/fs)
    ideal_fil[N+1:2*N+1] = ideal_fil_half

    # 窓関数をかける
    filter = ideal_fil * window
    return filter


def main():
    parser = argparse.ArgumentParser(
        description="wavfileからスペクトログラムを描画し、"
    )
    #parser.add_argument("-file", help="ファイルを入力")
    parser.add_argument("-cut", help="カットオフ周波数", default = 5000)
    parser.add_argument("-n", help="次数", default = 64)
    args = parser.parse_args()

    # データを読み込み
    data, rate = sf.read(r"C:\Users\kyskn\B4Lecture-2024\ex2\k_namizaki\20240428173142.wav")
    fc = args.cut
    N = args.n

    # filterの作成
    filter = hpf(rate, fc, N)

    # フィルタの振幅特性を計算
    mag = np.abs(np.fft.fft(filter, 1024))
    mag_db = 20 * np.log10(mag)
    # 周波数軸の計算
    freq_axis = np.linspace(0, rate, 1024)
    # フィルタの位相特性を計算(unwrapなしだと2pi分が毎回巻き戻されてしまう)
    phase = np.unwrap(np.angle(np.fft.fft(filter, 1024)))
    # 角度に変更
    phase = phase * 180/np.pi

    # filterの振幅を描画
    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis[:len(mag)//2], mag_db[:len(mag)//2])
    plt.title("high pass filter")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid(True)
    plt.show()
    # filterの位相を描画
    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis[:len(mag)//2], phase[:len(mag)//2])
    plt.title("high pass filter")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Digree [°]")
    plt.grid(True)
    plt.show()

    # dataとhpfを畳み込み
    filteredData = convolution(data,filter)

    # 入力と出力の大きさをそろえる
    length = len(data)
    filteredData = filteredData[0:length]
    # フィルター前のスペクトログラム分析
    f, t, Sxx = signal.spectrogram(data, rate)

    # フィルター前のスペクトログラムの描画
    plt.pcolormesh(t, f, 10*np.log(Sxx), shading="gouraud") #intensityを修正
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    cbar = plt.colorbar() #カラーバー表示のため追加
    cbar.ax.set_ylabel("Intensity [dB]") #カラーバーの名称表示のため追加
    plt.show()

    # フィルター後のスペクトログラム分析
    f, t, Sxx = signal.spectrogram(filteredData, rate)

    # フィルター後のスペクトログラムの描画
    plt.pcolormesh(t, f, 10*np.log(Sxx), shading="gouraud") #intensityを修正
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    cbar = plt.colorbar() #カラーバー表示のため追加
    cbar.ax.set_ylabel("Intensity [dB]") #カラーバーの名称表示のため追加
    plt.show()

if __name__ == "__main__":
    main()
