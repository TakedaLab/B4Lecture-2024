import sys
import wave
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal


def get_filename():
    # コマンドライン引数を取得
    args = sys.argv  # args=["main.py", str(FILENAME)]

    if len(args) == 2:
        return args[1]
    else:
        # コマンドライン引数に過不足がある場合, USAGEを表示
        print("USAGE: main.py FILENAME")
        return


def get_wavedata(filename: str):    
    # Wave_read objectを取得
    wave_read_obj = wave.open(filename, mode="rb")

    # バイナリデータを取得
    wavedata_bin = wave_read_obj.readframes(-1)

    # バイナリデータをndarrayに変換
    if wave_read_obj.getsampwidth() == 2:
        wavedata_ndarray = np.frombuffer(wavedata_bin, dtype="int16")
    elif wave_read_obj.getsampwidth() == 4:
        wavedata_ndarray = np.frombuffer(wavedata_bin, dtype="int32")
    else:
        print("Wave read object sample width error")
        return

    # サンプル周波数を取得
    framerate = wave_read_obj.getframerate()  # [Hz]
    sampling_rate = 1 / framerate  # [s]

    return wavedata_ndarray, sampling_rate


def plot_waveform(
    wavedata_ndarray: np.ndarray, sampling_rate: float, picture_name_option: str = ""
):
    # 波形を表示する準備
    x_val = np.arange(len(wavedata_ndarray)) * sampling_rate  # 横軸の定義
    plt.plot(x_val, wavedata_ndarray)
    plt.title("Waveform{0}".format(picture_name_option))
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")

    # 画像として保存
    plt.savefig("waveform{0}.png".format(picture_name_option))

    # 波形を表示
    plt.show()


def get_spectrogram(wavedata_ndarray: np.ndarray, n_samples: int, skip_width: int):
    # 空の行列を準備
    spectrogram = np.zeros(
        (1 + (len(wavedata_ndarray) - n_samples) // skip_width, n_samples // 2),
        dtype=np.complex128,
    )

    # 窓関数
    window = signal.hann(n_samples)

    for i in range(1 + (len(wavedata_ndarray) - n_samples) // skip_width):
        # fftを行う領域を切り出し、窓関数をかける
        sample = wavedata_ndarray[i * skip_width : i * skip_width + n_samples] * window

        # スペクトルを計算
        fft_sample = np.fft.fft(sample, n=n_samples, axis=0)[: n_samples // 2]  # fft
        spectrogram[i] = fft_sample  # 記録

    return spectrogram


def plot_spectrogram(
    spectrogram: np.ndarray, sampling_rate: float, n_samples: int, skip_width: int
):
    # スペクトログラムを表示する準備
    amp = np.abs(spectrogram)  # 振幅を計算
    amp_nonzero = np.where(
        amp == 0, np.nan, amp
    )  # divide by zero encountered in log10を回避するため0をnanに置き換え
    x_val = np.arange(len(spectrogram)) * sampling_rate * skip_width  # 横軸の定義
    y_val = np.fft.fftfreq(n_samples, d=sampling_rate)[:n_samples // 2]  # 縦軸の定義
    plt.pcolor(x_val, y_val, 20 * np.log10(amp_nonzero).T, shading="auto", cmap="jet")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram")
    plt.colorbar()

    # 画像として保存
    plt.savefig("spectrogram.png")

    # スペクトログラムを表示
    plt.show()


def restore_waveform(spectrogram: np.ndarray, n_samples: int, skip_width: int):
    # 空の配列を準備
    restore_waveform_ndarray = np.zeros((len(spectrogram) - 1) * skip_width + n_samples)

    # 窓関数
    window = signal.hann(n_samples)

    for i in range(len(spectrogram)):
        # フーリエ逆変換
        spectol = spectrogram[i]  # スペクトル切り出し
        wave_piece = np.fft.ifft(spectol, n=n_samples, axis=0).real  # ifft
        wave_piece = wave_piece * window  # 窓関数をかける
        restore_waveform_ndarray[
            i * skip_width : i * skip_width + n_samples
        ] += wave_piece

    return restore_waveform_ndarray


def main():
    # 音声ファイル名を取得
    filename = get_filename()

    # 音声ファイル名を取得できなければ終了
    if filename is None:
        return

    # 波形のndarrayを取得
    wavedata_ndarray, sampling_rate = get_wavedata(filename)

    # 波形のndarrayを取得できなければ終了
    if wavedata_ndarray is None:
        return

    # 波形を出力
    plot_waveform(wavedata_ndarray, sampling_rate, picture_name_option="_original")

    # パラメータ
    N_SAMPLES = 1024
    SKIP_WIDTH = N_SAMPLES // 2

    # スペクトログラムの計算
    spectrogram = get_spectrogram(wavedata_ndarray, N_SAMPLES, SKIP_WIDTH)

    # スペクトログラムの表示
    plot_spectrogram(spectrogram, sampling_rate, N_SAMPLES, SKIP_WIDTH)

    # フーリエ逆変換で波形を再計算
    restore_wavedata_ndarray = restore_waveform(spectrogram, N_SAMPLES, SKIP_WIDTH)

    # 再計算した波形を出力
    plot_waveform(
        restore_wavedata_ndarray, sampling_rate, picture_name_option="_restored"
    )


if __name__ == "__main__":
    main()
