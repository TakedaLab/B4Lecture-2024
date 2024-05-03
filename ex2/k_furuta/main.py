"""音声波形にバンドパスフィルタを適用するプログラム."""

import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import ex1


def convolve(A: np.ndarray, B: np.ndarray):
    """畳み込みを行う関数.

    フーリエ変換によって得た周波数成分を掛け合わせて、逆フーリエ変換を行うことで畳み込みを行う.
    Parameters
    -------------
    A : ndarray, shape=(time_of_A)
        畳み込みを行う数列
    B : ndarray, shape=(time_of_B)
        畳み込みを行う数列
    Returns
    ------------
    spectrogram : ndarray, shape=(time_of_A + time_of_B - 1)
        畳み込み結果
    """
    # 畳み込み後と同じサイズにする
    extended_A = np.pad(A, (0, B.shape[0] - 1))
    extended_B = np.pad(B, (0, A.shape[0] - 1))

    # fftで周波数成分を計算
    frequency_A = np.fft.fft(extended_A)
    frequency_B = np.fft.fft(extended_B)

    # 畳み込み結果の周波数を計算
    frequency_result = frequency_A * frequency_B

    # 周波数から畳み込み結果を復元
    result = np.fft.ifft(frequency_result)

    # 誤差によって虚数部が発生しているので切り捨てて出力
    return np.real(result)


def low_pass_filter(low: int, rate: int, sz: int):
    """ローパスフィルタを作成する関数.

    Parameters
    -------------
    low : int
        閾値となる周波数
    rate : int
        信号のサンプリングレート
    sz : int
        フィルタのサイズ(偶数の場合は奇数に合わせられる)
    Returns
    ------------
    spectrogram : ndarray, shape=(sz)
        畳み込み結果
    """
    # 周波数に対する閾値の割合を計算
    low = low / (rate / 2)
    # サイズに合わせて系列を計算
    time = np.arange(-(sz // 2), sz // 2 + 1)
    # sin(n*r_c)/n*piの系列を計算
    lpf = np.sinc(low * time) * low
    # 最後にハニング窓を掛ける
    lpf *= np.hanning(sz)
    return lpf


def parse_args():
    """コマンドプロントから引数を受け取るための関数."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="name of input wav file"
    )
    parser.add_argument("--nfft", type=int, default=1024, help="number of FFT points")
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="number of samples between successive STFT columns",
    )
    parser.add_argument(
        "--low", type=int, default=2000, help="left end point of the bandpass filter"
    )
    parser.add_argument(
        "--high", type=int, default=5000, help="right end point of the bandpass filter"
    )
    parser.add_argument(
        "--filter-length", type=int, default=101, help="size of bandpass filter"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 引数を受け取る
    args = parse_args()

    # 　入力音源の受け取り
    signal, rate = librosa.load(args.input_file, sr=None)

    # highとlowのローパスフィルタからバンドパスフィルタを作成(窓関数法)
    lpf_low = low_pass_filter(args.low, rate, args.filter_length)
    lpf_high = low_pass_filter(args.high, rate, args.filter_length)
    bpf = lpf_high - lpf_low

    # 周波数位相特性を計算
    response = np.fft.fft(bpf)

    # 作成したFIRと音源の畳み込み
    filtered_signal = convolve(signal, bpf)

    # フィルタ前後のスペクトログラムの作成
    original_spectrogram = ex1.create_spectrogram(signal, args.nfft, args.hop_length)
    filtered_spectrogram = ex1.create_spectrogram(
        filtered_signal, args.nfft, args.hop_length
    )

    # 同じスケールで表示させるために元の音源のスペクトログラムの最大と最小を取得
    vmax = np.max(20 * np.log10(np.abs(original_spectrogram)))
    vmin = np.min(20 * np.log10(np.abs(original_spectrogram)))

    # グラフサイズと空白の調整
    plt.figure(figsize=(7, 8))
    plt.subplots_adjust(wspace=0.4, hspace=1.0)

    # 周波数特性プロット用配列
    freq = np.arange(response.shape[0] // 2)

    # 周波数特性の表示
    plt.subplot(411)
    # デシベルと振幅の変換 振幅:x -> デシベル:20*log_10(x)
    plt.plot(
        freq * rate / 2 / freq.shape[0],
        20 * np.log10(np.abs(response))[: args.filter_length // 2],
    )
    plt.title("Frequency responce of bandpass filter")
    plt.xlabel("Frequency [hz]")
    plt.ylabel("Amplitude[db]")

    # プロット範囲
    extent = (0, rate / 2, -np.pi, np.pi)

    # 周波数特性の表示
    plt.subplot(412)
    # デシベルと振幅の変換 振幅:x -> デシベル:20*log_10(x)
    plt.plot(
        freq * rate / 2 / freq.shape[0], np.angle(response)[: args.filter_length // 2]
    )
    plt.title("Phase responce of bandpass filter")
    plt.xlabel("Frequency [hz]")
    plt.ylabel("phase[rad]")

    # x軸とy軸の表示範囲
    # x軸は音声の時間
    # y軸はサンプリング周波数の半分
    extent = (0, len(signal) / rate, 0, rate / 2)

    # 元の音源のスペクトログラムを表示
    # デシベルと振幅の変換 振幅:x -> デシベル:20*log_10(x)
    plt.subplot(413)
    plt.imshow(
        20 * np.log10(np.abs(original_spectrogram))[: args.nfft // 2],
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title("Spectrogram of original signal")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [hz]")

    # フィルタ後のスペクトログラムを表示
    # デシベルと振幅の変換 振幅:x -> デシベル:20*log_10(x)
    plt.subplot(414)
    plt.imshow(
        20 * np.log10(np.abs(filtered_spectrogram))[: args.nfft // 2],
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title("Spectrogram of filtered signal")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [hz]")

    # 画像として保存するときはコメントアウトを外す
    plt.savefig("result.png")

    plt.show()

    # 音源の復元
    sf.write(f"filtered_{args.input_file}", filtered_signal, rate)
