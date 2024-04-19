"""This module generates a spectrogram and performs an inverse transform.

It reads an input WAV file and generates a spectrogram.
The script then performs
an inverse transform to obtain the original signal.

"""

import argparse  # 引数の解析

import matplotlib.pyplot as plt  # グラフ描画
import numpy as np  # 線形代数
import scipy.fftpack as fftpack  # フーリエ変換
import scipy.io.wavfile as wavfile  # 音声読み込み
import scipy.signal as signal  # 窓関数


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    # wavファイル名
    parser.add_argument("--input-file", type=str, required=True, help="input wav file")
    """add nfft."""
    # NFFTの間隔（窓の幅）
    parser.add_argument("--nfft", type=int, default=1024, help="number of FFT points")
    # 連続的なSTFT列間のサンプル数（窓の移動幅）
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="number of samples between successive STFT columns",
    )
    # 窓関数の種類, デフォルト"hann"（山の形）
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        help="window function type",
    )
    return parser.parse_args()


def main():
    """Generate spectrograms and inverse transforms of audio signals."""
    args = parse_args()  # 引数情報

    # 音声ファイルを読み込む, rate:サンプリング周波数, data:信号のデータ
    rate, data = wavfile.read(args.input_file)
    data = np.array(data, dtype=float)

    # 出力の整形, 縦にグラフを並べる
    axs = plt.subplots(3, 2, gridspec_kw=dict(width_ratios=[9, 1]), figsize=(7, 7))[1]
    axs[0][1].axis("off")
    axs[1][1].axis("off")
    axs[2][1].axis("off")

    # 波形をプロットする, 横軸範囲は録音時間[sec]
    time = np.arange(0, len(data)) / rate
    axs[0][0].plot(time, data)
    axs[0][0].set_xlabel("Time [sec]")
    axs[0][0].set_ylabel("Amplitude")
    axs[0][0].set_title("Time Signal")

    # STFTのそれぞれのパラメータ
    nfft = args.nfft
    hop_length = args.hop_length
    # 与えられたタイプ・長さの窓関数を生成
    window = args.window
    window_func = signal.get_window(window, nfft)

    # スペクトログラムの計算
    spectrogram = np.zeros(
        # (正の周波数のみを対象, 時間の分解能), 複素数型
        (1 + nfft // 2, ((len(data) - nfft) // hop_length) + 1),
        dtype=np.complex128,
    )
    for i in range(spectrogram.shape[1]):
        # [短時間区間] の信号値に窓関数を掛ける
        segment = data[i * hop_length : i * hop_length + nfft] * window_func
        # fft(x:フーリエ変換する配列, n:フーリエ変換の長さ, axis:fftが計算される軸)
        spectrum = fftpack.fft(segment, n=nfft, axis=0)[: 1 + nfft // 2]
        spectrogram[:, i] = spectrum

    # スペクトログラムの描画
    im = axs[1][0].imshow(
        20 * np.log10(np.abs(spectrogram)),  # 画像データ, 振幅とデシベルの変換
        origin="lower",  # [0,0]を左上に置くか左下に置くか
        aspect="auto",  # 軸のアス比, autoで軸に収まるように歪む
        cmap="jet",  # ColorMap
        extent=(  # x軸y軸のメモリ指定
            0,
            len(data) / rate,
            0,
            spectrogram.shape[0],
        ),
    )
    axs[1][0].set_xlabel("Time [sec]")
    axs[1][0].set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=axs[1][1])
    axs[1][0].set_title("Spectrogram")

    # 逆変換の計算
    time_signal = np.zeros(len(data))
    for i in range(spectrogram.shape[1]):
        spectrum = spectrogram[:, i]
        segment = fftpack.ifft(spectrum, n=nfft, axis=0)
        segment = np.real(segment) * window_func  # np.real()は実数部分を返す
        time_signal[i * hop_length : i * hop_length + nfft] += segment

    # 逆変換した波形のプロット
    axs[2][0].plot(time, time_signal)
    axs[2][0].set_xlabel("Time [sec]")
    axs[2][0].set_ylabel("Amplitude")
    axs[2][0].set_title("Inverse Transform")

    plt.subplots_adjust(left=0.2, hspace=0.7)
    plt.savefig("result.png")


if __name__ == "__main__":
    main()
