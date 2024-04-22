"""
This code draws a spectrogram from a wavfile and returns it to the original.

FFT a wavfile into a spectrogram.
IFFT the spectrogram back to the original data.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
import scipy.signal as signal
import soundfile as sf



def main():
    """
    Draws a spectrogram from a wavfile and returns it to the original.

    FFT a wavfile into a spectrogram.
    IFFT the spectrogram back to the original data.
    """
    parser = argparse.ArgumentParser(
        description="wavfileからスペクトログラムを描画し、元に戻す"
    )
    parser.add_argument("-file", help="ファイルを入力")
    args = parser.parse_args()
    # データを読み込み
    data, rate = sf.read(args.file)

    # 波形データをplot
    time = np.arange(0, len(data)) / rate
    plt.plot(time, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude[unknown]")
    plt.title("Before")
    plt.show()

    # FFTのパラメータ
    N = 1024  # FFTサイズ
    shift = N // 2  # フレーム間のシフト量

    window = signal.get_window("hann", N)  # 窓関数

    # スペクトログラムの計算
    # スペクトログラムを格納(dtype=complexじゃないと複素数を実数に格納しようとしていることになってしまう)
    spectrogram = np.zeros(
        [N // 2, len(data) // shift - (N // shift - 1)], dtype=complex
    )

    for i in range(spectrogram.shape[1]):
        # 窓関数をかける
        segment = data[i * shift : i * shift + N] * window
        # fft
        spectrum = fftpack.fft(segment, n=N)[: N // 2]
        # スペクトログラムに格納
        spectrogram[:, i] = spectrum

    # デシベル変換
    dB_spectrogram = np.zeros([N // 2, len(data) // shift - (N // shift - 1)])
    eps = np.finfo(float).eps  # ゼロ除算を回避するための微小な値
    dB_spectrogram = 20 * np.log10(np.abs(spectrogram) + eps)

    # スペクトログラムの表示
    plt.figure()
    # imshowは2次元配列をカラーマップで表示(originは原点の位置)
    plt.imshow(
        dB_spectrogram,
        origin="lower",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram")
    plt.colorbar()
    plt.show()

    # 逆フーリエ変換を使ってオリジナルの波形データを再構築
    after_data = np.zeros(len(data))
    for i in range(spectrogram.shape[1]):
        spectrum = spectrogram[:, i]
        segment = fftpack.ifft(spectrum, n=N)
        segment = np.real(segment) * window
        after_data[i * shift : i * shift + N] += segment

    # オリジナルの波形データをplot
    plt.figure()
    plt.plot(time, after_data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude[unknown]")
    plt.title("After")
    plt.show()


if __name__ == "__main__":
    main()
