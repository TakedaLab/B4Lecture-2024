"""This module generates a spectrogram and performs an inverse transform.

It reads an input WAV file and generates a spectrogram.
The script then performs
an inverse transform to obtain the original signal.

"""

import argparse  # 引数の解析

import matplotlib.pyplot as plt  # グラフ描画
import numpy as np  # 線形代数
import scipy.fftpack as fftpack  # フーリエ変換
import scipy.signal as signal  # 窓関数
import soundfile as sf


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
    # 周波数通過域の指定
    parser.add_argument(
        "--f1",
        type=int,
        default="5000",
        help="f1[Hz]以下を通過域とする",
    )
    parser.add_argument(
        "--f2",
        type=int,
        default="10000",
        help="f2[Hz]以上を通過域とする",
    )
    return parser.parse_args()


# 波形をプロットする, 横軸範囲は録音時間[sec]
def show_wave(data: np.ndarray, samplerate: int, title: str = "Time Signal") -> None:
    """波形を時間領域でプロットする

    Args:
        data (np.ndarray): 信号のデータ
        samplerate (int): サンプリング周波数
        title (str, optional): 図のタイトル. Defaults to "Time Signal".

    Returns:
        None
    """
    time = np.arange(0, len(data)) / samplerate

    plt.figure()
    plt.plot(time, data)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.title(title)

    plt.savefig(title + ".png")

    return None


def calc_spectrogram(
    data: np.ndarray,
    nfft: int,
    hop_length: int,
    window: str,
) -> np.ndarray:
    """スペクトログラムの計算

    Args:
        data (np.ndarray): 信号のデータ
        nfft (int): nfftの間隔
        hop_length (int): 窓の移動幅
        window (str): 窓の種類

    Returns:
        np.ndarray: スペクトログラム
    """
    # 与えられたタイプ・長さの窓関数を生成
    window_func = signal.get_window(window, nfft)

    # スペクトログラムの計算
    spectrogram = np.zeros(
        # (↑FFTの出力データ数=FFTに入れるデータ長の半分, →時間の分解能), 複素数型
        (1 + nfft // 2, ((len(data) - nfft) // hop_length) + 1),
        dtype=np.complex128,
    )

    for i in range(spectrogram.shape[1]):  # 時間の分解能ぶんの範囲計算
        # [短時間区間] の信号値に窓関数を掛ける
        segment = data[i * hop_length : i * hop_length + nfft] * window_func
        # fft(x:フーリエ変換する配列, n:フーリエ変換の長さ, axis:fftが計算される軸)
        spectrum = fftpack.fft(segment, n=nfft, axis=0)[: 1 + nfft // 2]

        spectrogram[:, i] = spectrum

    return spectrogram


# スペクトログラムの描画
def show_spectrogram(
    spectrogram: np.ndarray, samplerate: int, len_data: int, title: str = "Spectrogram"
) -> None:
    """スペクトログラムを描画し, 画像を保存する

    Args:
        spectrogram (np.ndarray): スペクトログラム
        samplerate (int): サンプリング周波数
        len_data (int): 信号のデータ長
        title (str, optional): 図のタイトル. Defaults to "Spectrogram".

    Returns:
        None
    """
    amp = np.abs(spectrogram)
    amp_nonzero = np.where(amp == 0, np.nan, amp)
    plt.figure()
    plt.imshow(
        20 * np.log10(amp_nonzero),  # 画像データ, 振幅とデシベルの変換
        origin="lower",  # [0,0]を左上に置くか左下に置くか
        aspect="auto",  # 軸のアス比, autoで軸に収まるように歪む
        cmap="jet",  # ColorMap
        extent=(  # x軸y軸のメモリ指定
            0,
            len_data / samplerate,
            0,
            samplerate / 2,
        ),
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.title(title)

    plt.savefig(title + ".png")

    return None


# 逆変換の計算
def calc_inv_signal(
    len_data: int, spectrogram: np.ndarray, nfft: int, hop_length: int, window: str
) -> np.ndarray:
    """スペクトログラムから時間信号を復元する

    Args:
        len_data (int): 信号のデータ長
        spectrogram (np.ndarray): スペクトログラム
        nfft (int): nfftの間隔
        hop_length (int): 窓のずらし幅
        window (str): 窓の種類

    Returns:
        np.ndarray: 時間信号
    """
    # 与えられたタイプ・長さの窓関数を生成
    window_func = signal.get_window(window, nfft)

    inv_time_signal = np.zeros(len_data)
    for i in range(spectrogram.shape[1]):
        spectrum = spectrogram[:, i]
        segment = fftpack.ifft(spectrum, n=nfft, axis=0)
        segment = np.real(segment) * window_func  # np.real()は実数部分を返す
        inv_time_signal[i * hop_length : i * hop_length + nfft] += segment

    return inv_time_signal


def plot_any(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "plot",
    save: bool = False,
) -> None:
    """グラフ描画（保存）を行う

    Args:
        x (np.ndarray): x軸
        y (np.ndarray): プロットするデータ
        xlabel (str, optional): x軸のラベル名. Defaults to "x".
        ylabel (str, optional): y軸のラベル名. Defaults to "y".
        title (str, optional): 図のタイトル. Defaults to "plot".
        save (bool, optional): pngとして保存するか否か. Defaults to False.

    Returns:
        None
    """
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save:
        plt.savefig(title + ".png")
    plt.show()
    return None


def main():
    """Generate spectrograms and inverse transforms of audio signals."""
    args = parse_args()  # 引数情報

    # 音声ファイルを読み込む, data:信号のデータ, samplerate:サンプリング周波数
    data, samplerate = sf.read(args.input_file)

    # 時間領域のプロット
    show_wave(data, samplerate)

    # argsの取得
    nfft = args.nfft
    hop_length = args.hop_length
    window = args.window
    # window_func = signal.get_window(window, nfft)

    # スペクトログラムの計算
    spectrogram = calc_spectrogram(data, nfft, hop_length, window)

    # スペクトログラムの描画
    show_spectrogram(spectrogram, samplerate, len(data))

    # 逆変換の計算
    inv_signal = calc_inv_signal(len(data), spectrogram, nfft, hop_length, window)

    # 逆変換した波形のプロット
    show_wave(inv_signal, samplerate, "inv Time Signal")


if __name__ == "__main__":
    main()
