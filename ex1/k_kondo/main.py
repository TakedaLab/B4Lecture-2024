"""STFTとその逆変換の実行と結果の描画"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile


def STFT(data, Lw, step):
    """短時間フーリエ変換の実行

    Args:
        data (_np.array_): _入力信号_
        Lw (_int_): _窓幅_
        step (_int_): _切り出し幅_

    Returns:
        _np.array_: _スペクトログラム_
    """
    wavelength = data.shape[0]
    win = np.hanning(Lw)
    Mf = Lw // 2 + 1
    Nf = int(np.ceil((wavelength - Lw + step) / step))
    S = np.empty([Mf, Nf], dtype=np.complex128)
    for i in range(Nf):
        start = int(i * step)
        end = int(start + Lw)
        segment = (
            data[start:end]
            if end <= wavelength
            else np.append(data[start:], np.zeros(end - wavelength))
        )
        S[:, i] = np.fft.rfft(segment * win, n=Lw, axis=0)
    return S


def ISTFT(S, Lw, step):
    """逆変換の実行

    Args:
        S (_np.array_): _スペクトログラム_
        Lw (_int_): _窓幅_
        step (_int_): _切り出し幅_

    Returns:
        _np.array_: _音声信号_
    """
    Nf = S.shape[1]
    wavelength = int((Nf - 1) * step + Lw)

    win = np.hanning(Lw)
    x = np.zeros(wavelength)

    for i in range(Nf):
        start = int(i * step)
        end = start + Lw
        x[start:end] += np.fft.irfft(S[:, i], n=Lw) * win

    return x


def main():
    # コマンドプロンプトからファイル名を受け取り
    args = sys.argv
    if len(args) == 2:
        file_name = args[1]
    else:
        print("Usage: python main.py <file_name>")

    rate, data = wavfile.read(file_name)
    data = np.array(data, dtype=float)
    time = np.arange(0, len(data)) / rate

    # 波形を表示
    plt.plot(time, data)
    plt.xlabel("Time[sec]")
    plt.ylabel("Amplitude")
    plt.title("original_signal")

    length_window = 512
    step = length_window / 4

    spectrogrum = STFT(data, length_window, step)

    # スペクトログラムを表示
    P = 20 * np.log10(np.abs(spectrogrum))
    plt.figure()
    plt.imshow(P, origin="lower", aspect="auto")
    plt.xlabel("Time[sec]")
    plt.ylabel("Frequency[Hz]")
    plt.colorbar()
    plt.title("spectgrum")

    time_signal = ISTFT(spectrogrum, length_window, step)

    # 逆変換後の波形を表示
    time_adjusted = np.arange(0, len(time_signal)) * step / rate
    plt.figure()
    plt.plot(time_adjusted, time_signal)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.title("inverse transform")
    plt.show()


if __name__ == "__main__":
    main()
