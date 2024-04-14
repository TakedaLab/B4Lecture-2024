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


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    parser.add_argument("--input-file", type=str, required=True, help="input wav file")
    """add nfft."""
    parser.add_argument("--nfft", type=int, default=1024, help="number of FFT points")
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


def main():
    """Generate spectrograms and inverse transforms of audio signals."""
    args = parse_args()

    # 音声ファイルを読み込む
    rate, data = wavfile.read(args.input_file)
    data = np.array(data, dtype=float)

    # 波形をプロットする
    time = np.arange(0, len(data)) / rate
    plt.plot(time, data)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.show()

    # STFTのそれぞれのパラメータ
    nfft = args.nfft
    hop_length = args.hop_length
    window = args.window
    window_func = signal.get_window(window, nfft)

    # スペクトログラムの計算
    spectrogram = np.zeros(
        (1 + nfft // 2, (len(data) - nfft) // hop_length + 1), dtype=np.complex128
    )
    for i in range(spectrogram.shape[1]):
        segment = data[i * hop_length : i * hop_length + nfft] * window_func
        spectrum = fftpack.fft(segment, n=nfft, axis=0)[: 1 + nfft // 2]
        spectrogram[:, i] = spectrum

    # スペクトログラムの描画
    plt.figure()
    plt.imshow(
        20 * np.log10(np.abs(spectrogram)), origin="lower", aspect="auto", cmap="jet"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()

    # 逆変換の計算
    time_signal = np.zeros(len(data))
    for i in range(spectrogram.shape[1]):
        spectrum = spectrogram[:, i]
        segment = fftpack.ifft(spectrum, n=nfft, axis=0)
        segment = np.real(segment) * window_func
        time_signal[i * hop_length : i * hop_length + nfft] += segment

    # 逆変換した波形のプロット
    plt.figure()
    plt.plot(time, time_signal)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.title("Inverse Transform")

    plt.show()


if __name__ == "__main__":
    main()
