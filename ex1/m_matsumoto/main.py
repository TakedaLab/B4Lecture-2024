"""This module generates a spectrogram and performs an inverse transform.

It reads an input WAV file and generates a spectrogram.
The script then performs
an inverse transform to obtain the original signal.

"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
This module parse variables from execution options.
"""


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    parser.add_argument(
        # params
        "--input-file",
        type=str,
        required=True,
        help="input wav file",
    )
    """add nfft."""
    parser.add_argument(
        # params
        "--nfft",
        type=int,
        default=1024,
        help="number of FFT points",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="number of samples between successive STFT columns",
    )
    parser.add_argument(
        "--window", type=str, default="hann", help="window function type"
    )
    return parser.parse_args()


"""
Description shows on the above.
"""


def main():
    """Generate spectrograms and inverse transforms of audio signals."""
    # 標準入力からオプションを読み取る．
    args = parse_args()

    # 入力音声ファイルを読み込む
    rate, data = wavfile.read(args.input_file)

    # 入力音声ファイルをnumpyの配列にする．
    data = np.array(data, dtype=float)

    # 時刻の定義
    time = np.arange(0, len(data)) / rate

    """STFT(short-time Fourier transform)のそれぞれのパラメータ"""
    nfft = args.nfft  # 窓の大きさ
    hop_length = args.hop_length  # 窓をずらす大きさ
    window_func = signal.get_window(args.window, nfft)  # 窓関数の取得

    """スペクトログラムの計算"""
    # スペクトグラムの初期値
    spectrogram = np.zeros(
        # 配列の形
        (
            # 観測する周波数の個数
            1 + nfft // 2,
            # 時間区切りする個数
            (len(data) - nfft) // hop_length + 1,
        ),
        # 複素数
        dtype=np.complex128,
    )
    for i in range(
        spectrogram.shape[1]
    ):  # 1次元目は周波数が配列になっているため，.shape[1]を取得
        # 1次元目は周波数なので，全てを表現する : ，2次元目が時間を表現する．
        spectrogram[:, i] = fftpack.fft(
            # 窓を付与させて，そのスペクトグラムの値を格納している．
            # E203 whitespace before ':'
            data[i * hop_length : i * hop_length + nfft] * window_func,
            # フーリエ変換を行う範囲．つまり，窓の大きさである．
            n=nfft,
            # 配列の作用させる方向．
            axis=0,
            # スペクトグラムに付与する．
        )[: 1 + nfft // 2]

    """逆変換の計算"""
    time_signal = np.zeros(len(data))  # 初期値の設定
    # 時間でループさせて，逆変換を行う．
    for i in range(spectrogram.shape[1]):
        # 窓の大きさの範囲にそのスペクトログラムを適応させる．振幅は実数部を描画すれば良い．
        # E203 whitespace before ':'
        time_signal[i * hop_length : i * hop_length + nfft] += (
            np.real(
                fftpack.ifft(
                    # i番目のスペクトログラムを取得する．
                    spectrogram[:, i],
                    # フーリエ変換を行う範囲．つまり，窓の大きさ
                    n=nfft,
                    # 配列の作用させる方向
                    axis=0,
                )
            )
            * window_func
        )  # 窓関数

    # オブジェクト指向描画
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    # グラフ1: 時間と振幅
    axs[0].plot(time, data)
    axs[0].set_xlabel("Time [sec]")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Input Signal")

    # グラフ2: スペクトログラム
    im = axs[1].imshow(
        # デシベルへの変換
        20 * np.log10(np.abs(spectrogram)),
        origin="lower",
        aspect="auto",
        cmap="jet",
    )
    axs[1].set_xlabel("Hop Times[count]")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_title("Spectrogram")
    divider = make_axes_locatable(axs[1])
    ax_cb = divider.new_horizontal(size="2%", pad=-0.05)
    fig.add_axes(ax_cb)
    _ = fig.colorbar(im, ax=axs[1], pad=0.1, shrink=0.8, cax=ax_cb)

    # グラフ3: 逆変換された信号
    axs[2].plot(time, time_signal)
    axs[2].set_xlabel("Time [sec]")
    axs[2].set_ylabel("Amplitude")
    axs[2].set_title("Inverse Transform")

    # グラフ間の間隔を調整
    plt.subplots_adjust(hspace=0.5)

    # 描画
    plt.show()


if __name__ == "__main__":
    main()
