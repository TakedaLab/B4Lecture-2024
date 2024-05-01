"""Execute BEF(Band Eliminate Filter)."""

import filter
import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile


def spectrogram(
    ax, fig, index: int, data: np.ndarray, sr: int, label: bool = True
) -> None:
    """Draw Spectrogram.

    Args:
        ax
        fig
        index (int): ax, fig index
        data (np.ndarray): wave data
        sr (int): wave samplerate
    """
    f, t, s = scipy.signal.spectrogram(data, sr)
    if label:
        ax[index].set_xlabel("Time [sec]")
        ax[index].set_ylabel("Frequency [Hz]")
    fig.colorbar(
        ax[index].pcolormesh(t, f, 20 * np.log10(s), vmax=1e-6, cmap="CMRmap"),
        ax=ax[index],
        aspect=5,
        location="right",
    )


def soundwave(
    ax, index: int, data: np.ndarray, samplerate: int, label: bool = True
) -> None:
    """Draw SoundWave.

    Args:
        ax
        index (int): ax index
        data (np.ndarray): wavedata
        samplerate (int): wave sample rate
    """
    if label:
        ax[index].set_xlabel("Time")
        ax[index].set_ylabel("Magnitude")
    ax[index].plot(np.arange(0, len(data)) / samplerate, data)


if __name__ == "__main__":
    fig, ax = plt.subplots(4, 1, layout="constrained", sharex=True)
    fig2, ax2 = plt.subplots(3, 1, layout="constrained")

    INPUT = "./my_voice.wav"
    OUTPUT = "./out.wav"
    WAVE = "./wave.png"
    FILTER = "./filter.png"
    TAP = 201
    LEFT = 2000
    RIGHT = 5000

    data, rate = soundfile.read(INPUT)

    ax[0].set_title("Original Wave")
    soundwave(ax, 0, data, rate)
    ax[2].set_title("Original spectrogram")
    spectrogram(ax, fig, 2, data, rate)  # フィルタリング前のスペクトログラムを出力

    # フィルタリング処理
    bef = filter.bef(LEFT, RIGHT, TAP, rate)  # BEF
    bef_fft = np.fft.fft(bef)  # BEF周波数
    freq = np.fft.fftfreq(TAP, d=1.0 / rate)

    # BEF
    # Amp
    ax2[0].set_title("BEF (Magnitude)")
    ax2[0].set_xlabel("Frequency[Hz]")
    ax2[0].set_ylabel("Amplitude[db]")
    ax2[0].plot(freq[: TAP // 2], np.abs(bef_fft[: TAP // 2]))
    # Phase
    ax2[1].set_title("BEF (Angle)")
    ax2[1].set_xlabel("Frequency[Hz]")
    ax2[1].set_ylabel("Angle[rad]")
    ax2[1].plot(freq[: TAP // 2], np.unwrap(np.angle(bef_fft[: TAP // 2])))
    # Time
    ax2[2].set_title("Filter preview")
    ax2[2].set_xlabel("The number of the sample")
    ax2[2].set_ylabel("Amplitude[db]")
    ax2[2].plot(np.arange(len(bef)), bef)

    conv_result = filter.conv(data, bef)  # 音源とフィルタを畳み込み
    ax[1].set_title("Filtered wave")
    soundwave(ax, 1, conv_result, rate)  # フィルタリング後の波形を出力
    ax[3].set_title("Filtered spectrogram")
    spectrogram(
        ax, fig, 3, conv_result, rate
    )  # フィルタリング後のスペクトログラムを出力

    plt.show()
    fig.savefig(WAVE)
    fig2.savefig(FILTER)
    plt.clf()
    plt.close()
    soundfile.write(file=OUTPUT, data=conv_result, samplerate=rate)
