"""畳み込み演算の実装とディジタルフィルタの設計を行い、フィルタリングの影響をスペクトログラムと音声ファイルで確認する.

wave       : 音声ファイルの取得
matplotlib : グラフやスペクトログラムの描画
numpy      : 行列,fft
scipy      : 窓関数
ex1        : 課題1で作成した関数
"""

import wave

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ex1 import get_filename, get_wavedata, calculate_spectrogram, plot_spectrogram


def convolution(data1: np.ndarray, data2: np.ndarray):
    """畳み込み演算.

    -> 畳み込み演算結果: np.ndarray

    if 0 <= n < len(datai): x[n]=data[n]
    else: x[n] = 0
    とする.
    """
    n1 = len(data1)
    n2 = len(data2)
    # data1の方が配列長が大きくなるよう調整
    if len(data1) < len(data2):
        data1, data2 = data2, data1
        n1, n2 = n2, n1

    # data2の順番を逆転
    data2_reversed = data2[::-1]

    # 演算結果の配列を定義
    output = np.zeros(n1 + n2 - 1)

    for i in range(n1 + n2 - 1):
        temp = (
            data1[max(0, i - n2 + 1) : min(i + 1, n1)]
            * data2_reversed[max(0, n2 - i - 1) : min(n2, n1 + n2 - i - 1)]
        )  # x1[k]✕x2[n-k]
        output[i] = np.sum(temp)  # 総和をとる

    return output


def calc_low_pass_filer(fc: int, n_impulse_response: int, sampling_rate: float):
    """ディジタルフィルタ(LPF)の実装.

    -> LPFのインパルス応答: np.ndarray
    """
    # omega_cを計算
    omega_c = 2 * fc * sampling_rate
    # h[n] = (omega_c / pi) * sinc(omega_c * n)
    impulse_response = omega_c * np.sinc(
        np.arange(-n_impulse_response, n_impulse_response + 1) * omega_c
    )

    # ハニング窓をh[n]にかける
    window = signal.hann(2 * n_impulse_response + 1)
    impulse_response *= window

    return impulse_response


def plot_frequency_response(
    impulse_response: np.ndarray, n_impulse_response: int, sampling_rate: float
):
    """振幅特性と位相特性を描画.

    -> None
    """
    len_impulse_response = 2 * n_impulse_response + 1
    # インパルス応答をfftして周波数応答を求める
    frequency_resp = np.fft.fft(impulse_response, n=len_impulse_response, axis=0)[
        : len_impulse_response // 2
    ]

    # 横軸の定義
    x_val = np.fft.fftfreq(len_impulse_response, d=sampling_rate)[
        : len_impulse_response // 2
    ]

    # 振幅特性を描画
    amp = np.abs(frequency_resp)  # 振幅を計算
    amp_nonzero = np.where(amp == 0, np.nan, amp)  # 0をnanに置換
    plt.plot(x_val, 20 * np.log10(amp_nonzero))
    plt.title("Amplitude Spectol")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.savefig("amplitude_spectol.png")
    plt.show()

    # 位相特性を描画
    angle = np.angle(frequency_resp)  # 位相を計算
    plt.plot(x_val, angle)
    plt.title("Phase Spectol")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [rad]")
    plt.savefig("phase_spectol.png")
    plt.show()

    return


def output_wav(wave_data: np.ndarray, sampling_rate: float, file_name: str):
    """波形データをwavファイルに変換.

    -> None
    """
    # 波形データをint16に変換
    data = wave_data.astype(np.int16)

    with wave.open(file_name, "w") as wf:
        wf.setnchannels(1)  # チャンネル数
        wf.setsampwidth(2)  # サンプル幅
        wf.setframerate(round(1 / sampling_rate))  # サンプリング周波数
        wf.writeframes(data)  # データ
        wf.close()

    return


def main():
    """main関数.

    -> None
    """
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

    # パラメータ
    FC = 10000
    N_IMPULSE_RESPONSE = 512

    # LPFのインパルス応答を取得
    impulse_response = calc_low_pass_filer(FC, N_IMPULSE_RESPONSE, sampling_rate)

    # 周波数特性を描画
    plot_frequency_response(impulse_response, N_IMPULSE_RESPONSE, sampling_rate)

    # 入力波形とインパルス応答を畳み込み
    filtered_wavedata = convolution(wavedata_ndarray, impulse_response)[
        N_IMPULSE_RESPONSE:-N_IMPULSE_RESPONSE
    ]

    # パラメータ
    N_SAMPLES = 1024
    SKIP_WIDTH = N_SAMPLES // 2

    # 入力波形とフィルタリング結果のスペクトログラムを計算、表示
    spectrogram = calculate_spectrogram(wavedata_ndarray, N_SAMPLES, SKIP_WIDTH)
    plot_spectrogram(
        spectrogram,
        sampling_rate,
        N_SAMPLES,
        SKIP_WIDTH,
        name_option="_original",
    )

    filtered_spectrogram = calculate_spectrogram(
        filtered_wavedata, N_SAMPLES, SKIP_WIDTH
    )
    plot_spectrogram(
        filtered_spectrogram,
        sampling_rate,
        N_SAMPLES,
        SKIP_WIDTH,
        name_option="_filtered",
    )

    # フィルタリングされた波形をwavファイルに出力
    output_wav(filtered_wavedata, sampling_rate, "filtered_sample.wav")

    return


if __name__ == "__main__":
    main()
