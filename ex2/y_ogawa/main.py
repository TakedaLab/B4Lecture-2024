"""ローパスフィルタを実装する."""

import argparse

import librosa
import matplotlib.pyplot as plt  # グラフ作成用
import numpy as np  # 線形代数
import soundfile as sf

import ex1


# 畳み込みの計算を実装
def conv(x: np.ndarray, h: np.ndarray):
    """畳み込み演算を行う.

    Args:
        x: 入力信号
        h: インパルス応答

    Returns:
        y: 畳み込みの結果を返す
    """
    # 出力配列を用意
    y = np.zeros(len(x) + len(h) - 1)

    # 畳み込みの実施
    for i in range(len(x)):
        y[i : i + len(h)] += x[i] * h

    return y


#   コマンドラインから引数を受け取る
def parse_args():
    """引数の取得を行う.

    filename: 読み込むファイル名
    cutoff_freq: カットオフ周波数
    fft_size: 高速フーリエ変換の窓の大きさ
    """
    parser = argparse.ArgumentParser(description="LPF")
    parser.add_argument("--filename", type=str, required=True, help="name of file")
    parser.add_argument("--cutoff_freq", type=float, default=5000, help="cutoff freq")
    parser.add_argument("--fft_size", type=int, default=1024, help="size of FFT")
    return parser.parse_args()


#   LPFの作成
def make_LPF(cutoff_freq, sr):
    """LPFの作成を行う.

    Args:
        cutoff_freq: カットオフ周波数
        sr: サンプリング周波数

    Returns:
        lpf: ローパスフィルタを返す
    """
    tap_number = 101  # タップの数
    time = np.arange(-(tap_number // 2), tap_number // 2 + 1)
    # インパルス応答
    lpf = (np.sinc(2 * cutoff_freq / sr * time)) * 2 * cutoff_freq / sr
    lpf *= np.hanning(tap_number)  # ハニング窓を掛ける
    return lpf


#   周波数特性、位相特性のプロット
def plot_filter_response(LPF, sr, fft_size):
    """周波数特性、位相特性をプロットする.

    Args:
        LPF: フィルター
        sr: サンプリング周波数
        fft_size: 高速フーリエ変換の窓の大きさ
    """
    # フーリエ変換をして周波数領域へ
    lpf_freq_response = np.fft.fft(LPF, fft_size)

    # 周波数軸を生成
    plot_freq_size = len(lpf_freq_response) // 2
    freq = np.arange(plot_freq_size)

    # プロット
    fig = plt.figure()
    # 周波数特性の図示
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(
        freq * sr / 2 / freq.shape[0],
        20 * np.log(np.abs(lpf_freq_response))[:plot_freq_size],
        label="LPF Amplitude Response",
    )
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude (dB)")
    # 位相特性の図示
    ax2 = fig.add_subplot(3, 3, 7)
    ax2.plot(
        freq * sr / 2 / freq.shape[0],
        np.unwrap(np.angle(lpf_freq_response)[:plot_freq_size]),
        label="LPF Phase Response",
    )
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (rad)")


# 　スペクトログラムの図示
def plot_spectrogram(spectrogram, sr, y, title_name, locate):
    """スペクトログラムを描画する.

    Args:
        spectrogram: sスペクトログラムの計算結果
        sr: サンプリング周波数
        y: 波形の値
        title_name: グラフのタイトル
        locate: グラフの位置
    """
    # 単位をdbに変換する 真数条件を満たすためにわずかな値を加えた
    spectrogram_db = 20 * np.log(np.abs(spectrogram).T + 1e-10)

    extent = (0.1, y.size / sr, 0, sr / 2)  # 縦軸はナイキスト周波数まで
    plt.subplot(3, 3, locate)
    # フーリエ変換の実数部分だけをみるので半分だけ
    plt.imshow(
        spectrogram_db[: int(len(spectrogram) / 2)],
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="jet",
    )
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar()
    plt.title(title_name)


def main():
    """LPFでフィルタリングし、スペクトログラムを表示."""
    #   引数を受け取る
    args = parse_args()

    y, sr = librosa.load(
        args.filename, sr=None
    )  # 音声データを取得　srはサンプリング周波数
    LPF = make_LPF(args.cutoff_freq, sr)  # LPFを実装
    plot_filter_response(LPF, sr, args.fft_size)  # LPFの特性を図示
    filter_data = conv(y, LPF)  # 畳み込みの計算
    # スペクトログラムを作成する
    plot_spectrogram(
        ex1.makeSpectrogram(y.size, y, sr, 1024, 512), sr, y, "origin supectrogram", 3
    )
    plot_spectrogram(
        ex1.makeSpectrogram(filter_data.size, filter_data, sr, 1024, 512),
        sr,
        filter_data,
        "filtered spectrogram",
        9,
    )
    plt.savefig("Figure.png")
    plt.show()

    sf.write("filtered.wav", np.abs(filter_data), sr)


if __name__ == "__main__":
    main()
