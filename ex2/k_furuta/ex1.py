"""音声波形からスペクトログラムを作成し、それを逆変換によって復元するプログラム.

元の音声データ、スペクトログラム、復元された音声データをグラフで表示する
コマンドライン引数でファイルを指定する(python3 main.py FILE_NAME)
"""

import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def create_spectrogram(signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """音声波形からスペクトログラムを作成する.

    Parameters
    -------------
    signal : ndarray
        波形データ
    n_fft : int
        fftを行う窓の幅
    hop_length : int
        窓ごとの間隔
    Returns
    ------------
    spectrogram : ndarray, shape=(frequency, time)
        波形データから計算されたスペクトログラム
    """
    n_frames = 1 + (len(signal) - n_fft) // hop_length  # フレーム数を決定
    # hop_lengthだけずらしながら切り出した行列を作成 shape=(n_fft, n_frames)
    framed_signal = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_fft, n_frames),
        strides=(signal.strides[0], signal.strides[0] * hop_length),
    )
    # ブロードキャストするために，窓関数に軸を追加
    windowed_signal = framed_signal * np.hanning(n_fft)[:, np.newaxis]
    # 0番目の軸に沿ってfftをかける
    spectrogram = np.fft.fft(windowed_signal, axis=0)

    return spectrogram


# スペクトログラムから音声波形を復元
def inverse_spectrogram(
    spectrogram: np.ndarray, n_fft: int, hop_length: int
) -> np.ndarray:
    """音声波形からスペクトログラムを作成する.

    Parameters
    -------------
    spectrogram : ndarray
        スペクトログラム
    n_fft : int
        fftを行う窓の幅
    hop_length : int
        窓ごとの間隔
    Returns
    ------------
    signal : ndarray , shape=(time)
        スペクトログラムから計算された波形データ
    """
    n_frames = spectrogram.shape[1]
    output_length = n_fft + hop_length * (n_frames - 1)
    signal = np.zeros(output_length)

    framed_signal = np.fft.ifft(spectrogram, axis=0)

    # for文除去できるらしい
    for i in range(n_frames):
        frame = framed_signal[:, i]
        start = i * hop_length
        # ifftの実数部分のみ利用(丸め誤差で微小な虚部が発生している、結果には影響しない)
        signal[start : start + n_fft] += frame.real * np.hanning(n_fft)
    return signal


if __name__ == "__main__":
    # argparserでファイル名を読み込む
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)

    filename = parser.parse_args().filename
    signal, rate = librosa.load(filename, sr=None)

    # パラメータ
    n_fft = 1024
    hop_length = 512

    # スペクトログラムの生成
    spectrogram = create_spectrogram(signal, n_fft, hop_length)

    # 逆変換
    reconstructed_signal = inverse_spectrogram(spectrogram, n_fft, hop_length)

    # スペクトログラムと信号の表示
    time = np.arange(0, len(signal)) / rate  # 音声波形をプロットするための時間データ

    plt.subplots_adjust(wspace=0.4, hspace=1.0)
    plt.subplot(221)
    plt.plot(time, signal)
    plt.title("Original Audio Signal")
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")

    plt.subplot(122)
    # x軸とy軸の表示範囲
    # x軸は音声の時間
    # サンプリング定理よりy軸はサンプリング周波数の半分？
    extent = (0, len(signal) / rate, 0, rate / 2)
    # 実数信号なので周波数はsymmetricになっているため半分だけ表示
    # デシベルと振幅の変換 振幅:x -> デシベル:20*log_10(x)
    plt.imshow(
        20 * np.log10(np.abs(spectrogram)[: n_fft // 2, :]),
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )

    plt.title("Spectrogram")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [hz]")

    plt.subplot(223)
    plt.plot(time, reconstructed_signal)
    plt.title("Restored Audio Signal")
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")

    plt.show()

    # 復元した波形をwavファイルに再構成(なぜかノイズが発生している)
    sf.write(f"reconstructed_{filename}", reconstructed_signal, rate)
