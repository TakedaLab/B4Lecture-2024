"""This module generates a spectrogram and performs an inverse transform.

It reads an input WAV file and generates a spectrogram.
The script then performs
an inverse transform to obtain the original signal.

"""
import argparse # 引数の解析

import matplotlib.pyplot as plt # グラフ描画
import numpy as np # 線形代数
import scipy.fftpack as fftpack # フーリエ変換
import scipy.io.wavfile as wavfile # 音声読み込み
import scipy.signal as signal # 窓関数


def parse_args():
    """Retrieve variables from the command prompt.""" # コマンドプロンプトから変数を取り出す
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    parser.add_argument("--input-file", type=str, required=True, help="input wav file") # wavファイルをstr型で取り出す
    """add nfft.""" # NFFT（不等間隔離散フーリエ変換）
    parser.add_argument("--nfft", type=int, default=1024, help="number of FFT points") # NFFTの間隔（窓の幅）をint型で取り出す、デフォルト1024
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="number of samples between successive STFT columns", # 連続的なSTFT（短時間フーリエ変換）列間のサンプル数（窓の移動幅）をint型で取り出す、デフォルト512
    )
    parser.add_argument(
        "--window", type=str, default="hann", help="window function type" # 窓関数のタイプをstrで取り出す、デフォルト"hann"（山の形）
    )
    return parser.parse_args()


def main():
    """Generate spectrograms and inverse transforms of audio signals."""  # 音声信号のスペクトログラムと逆変換を生成
    args = parse_args() # 引数情報

    # 音声ファイルを読み込む
    rate, data = wavfile.read(args.input_file) # rate=32000:サンプリング周波数, data:信号の値
    data = np.array(data, dtype=float) # 信号の値を、要素float型のnumpy.array形式にする

    # 波形をプロットする
    time = np.arange(0, len(data)) / rate # 横軸（時間）の範囲内で、等差数列をndarrayとして生成
    plt.figure()
    plt.plot(time, data) # 横軸：時間　縦軸：信号
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.title("Time Signal")
    plt.savefig("Time_Signal.png") # 表示（時間信号）

    # STFTのそれぞれのパラメータ
    nfft = args.nfft # デフォルト1024
    hop_length = args.hop_length # デフォルト512
    window = args.window # 窓関数、デフォルト"hann"
    window_func = signal.get_window(window, nfft) # 与えられたタイプ・長さの窓関数を生成

    # スペクトログラムの計算
    spectrogram = np.zeros(
        # (正の周波数のみ対象, 時間の分解能？)
        (1 + nfft // 2, ((len(data) - nfft) // hop_length) + 1), dtype=np.complex128 # 0配列を生成、"//"は小数切り捨てで整数を返す除算
    )
    for i in range(spectrogram.shape[1]): # shapeは各次元のサイズ、3次元の場合は（奥行,縦,横）だが今回は2次元（行,列）
        segment = data[i * hop_length : i * hop_length + nfft] * window_func # [短時間区間]範囲内の信号値に窓関数を掛ける
        spectrum = fftpack.fft(segment, n=nfft, axis=0)[: 1 + nfft // 2] # fft(x:フーリエ変換する配列, n:フーリエ変換の長さ, axis:fftが計算される軸), [開始位置:終了位置]はスライス
        spectrogram[:, i] = spectrum # おそらく縦方向ごとに計算してスペクトログラムに代入している

    # スペクトログラムの描画
    plt.figure() # 描画領域全体を生成する
    plt.imshow(
        20 * np.log10(np.abs(spectrogram)), # 画像データ。振幅とデシベルの変換式
        origin="lower", # origin:[0,0]を左上に置くか左下に置くか
        aspect="auto", # aspect:軸のアス比, autoで軸に収まるように歪む
        cmap="jet", # ColorMap
        extent=(0, len(data) / rate, 0, spectrogram.shape[0]) # (x開始, x終了, y開始, y終了)
    )
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar() # 図にカラーバーを表示
    plt.title("Spectrogram")
    plt.savefig("Spectrogram.png")

    # 逆変換の計算
    time_signal = np.zeros(len(data)) # 時間信号用の0配列を用意
    for i in range(spectrogram.shape[1]):
        spectrum = spectrogram[:, i] # スペクトログラムの、縦は最初から最後まで、横軸に関してはi地点を抽出？
        segment = fftpack.ifft(spectrum, n=nfft, axis=0) # よく見たらスペクトログラムの描画の逆をたどっているだけ
        segment = np.real(segment) * window_func # np.real()は実数部分を返す
        time_signal[i * hop_length : i * hop_length + nfft] += segment # 時間信号の[]の範囲に計算したやつを代入している、[]はおそらく窓を掛ける範囲？

    # 逆変換した波形のプロット
    plt.figure()
    plt.plot(time, time_signal)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.title("Inverse Transform")
    plt.savefig("Inverse_Transform.png")


if __name__ == "__main__":
    main()
