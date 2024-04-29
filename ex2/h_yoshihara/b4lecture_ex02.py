# -*- coding: utf-8 -*-
"""
音声データにハイパスフィルタをかけるコード.
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
import soundfile
from pydub import AudioSegment


def stereo_to_mono(audio_name):
    """ステレオ音声データをモノラルに変換."""
    sound = AudioSegment.from_wav(audio_name)
    mono_audio = sound.set_channels(1)
    mono_audio.export("mono_shuffle.wav", format="wav")
    returned_audio_path = "mono_shuffle.wav"
    return returned_audio_path


def sinc(x):
    """sinc関数、numpyのやつだとなぜか上手くいかなかった(?)ので自作."""
    if x == 0.0:
        return 1.0
    else:
        return np.sin(x) / x


def hpf_filter(edge_f, delta):
    """フィルタの作成."""
    # フィルタ係数が奇数になるように調整
    coefficient = round(3.1 / delta) - 1
    if (coefficient + 1) % 2 == 0:
        coefficient += 1
    coefficient = int(coefficient)

    # 遷移帯域幅を満たすフィルタ係数の導出
    b = []
    for i in range(int(-coefficient / 2), int(coefficient / 2) + 1):
        x = math.pi * i
        b.append(sinc(x) - 2.0 * edge_f * sinc(2.0 * edge_f * x))

    # 窓関数法
    window = sp.windows.hann(coefficient + 1)
    for i in range(len(b)):
        b[i] *= window[i]
    return np.array(b)


def conv(input_x, filter_h):
    """ハイパスフィルタをかける."""
    conv_y = np.zeros(len(input_x) + len(filter_h))  # 出力用の信号
    for i in range(len(input_x)):
        conv_y[i : i + len(filter_h)] += input_x[i] * filter_h
    return conv_y


def spec_display(wave_data, data_length, wave_samplerate):
    """前回作成した、スペクトログラムを表示する関数."""
    fft_data = []  # フーリエ変換後のデータを逐次保存
    f_width = 1024  # 切り出すフレームの幅
    overlap = 512  # オーバーラップするフレーム幅（今回は50%で）
    cut_surplus = f_width - overlap  # 切り出していくフレーム間の幅

    wd = sp.windows.hamming(
        f_width
    )  # 切り出すフレーム幅で窓関数を用意（今回はハミング窓）

    cut_start = 0  # 切り出していくフレーム位置
    while cut_start + f_width <= data_length:
        frame = wave_data[cut_start : cut_start + f_width]  # 音声データから切り出し
        fft_frame = np.fft.fft(
            frame * wd
        )  # 切り出したデータに窓関数をかけ、フーリエ変換
        fft_data.append(fft_frame)  # フーリエ変換後のデータを保存
        cut_start += cut_surplus  # 切り出し開始位置を更新

    fft_data = np.array(fft_data)  # np.absを使いたいので
    fft_abs = np.abs(
        fft_data[:, :cut_surplus]
    ).T  # オーバーラップ部分を省きつつ絶対値をとる

    spectrogram = 20 * (np.log10(fft_abs))  # 20*log(振幅)でデシベルに変換できるらしい?
    fig, ax = plt.subplots()
    im = ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        extent=(0, data_length / wave_samplerate, 0, wave_samplerate // 2),
        cmap="plasma",
    )
    plt.colorbar(im, aspect=3, location="right")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.show()
    plt.close()

    return None


if __name__ == "__main__":
    stereo_audio = "月まで届け.wav"
    audio_path = stereo_to_mono(stereo_audio)  # ステレオ音声をモノラルに

    data, samplerate = soundfile.read(
        audio_path
    )  # オーディオデータとサンプリングレートの取り出し
    N = len(data)  # オーディオデータのサイズを取得

    # 今回使用した音声データの波形表示
    time = np.arange(0, N) / samplerate  # 音声データの時間を取得

    fe = 5000.0 / samplerate  # 今回は5000Hz以上を通す想定
    delta = 100.0 / samplerate  # 遷移帯域幅の設定

    filter = hpf_filter(fe, delta)  # ハイパスフィルタの作成

    filtered_data = conv(data, filter)  # 音声データをフィルタにかける
    filtered_time = np.arange(0, len(filtered_data)) / samplerate

    # フィルタをかける前後の音声データの波形の比較
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 3), tight_layout=True)
    ax1.plot(time, data, label="original")
    ax1.plot(filtered_time, filtered_data, label="filtered")
    ax1.set(
        title="Signal comparison",
        xlabel="Time[s]",
        ylabel="Magnitude",
        xlim=(0, time[-1]),
        ylim=(-1, 1),
    )
    ax1.legend()
    fig1.savefig("wave_comparison.png")

    # 今回使用したフィルタの表示のための下準備
    filter_freq = np.fft.rfft(filter)
    amp = np.abs(filter_freq)
    fil_phase = np.unwrap(np.angle(filter_freq))
    frequency = np.linspace(0, samplerate / 2, len(fil_phase)) / 1000

    # 今回したフィルタの周波数特性を表示
    fig2, ax2 = plt.subplots(2, 1, figsize=(6, 6), tight_layout=True, sharex=True)
    ax2[0].plot(frequency, amp)
    ax2[0].set(
        title="Frequency response of BPF",
        ylabel="Amplitude[dB]",
    )
    ax2[1].plot(frequency, fil_phase)
    ax2[1].set(
        xlabel="Frequency[kHz]",
        ylabel="Phase[rad]",
    )
    fig2.savefig("frequency_response.png")

    spec_display(
        data, N, samplerate
    )  # フィルタをかける前の音声データのスペクトログラム
    spec_display(
        filtered_data, N, samplerate
    )  # フィルタをかけた後の音声データのスペクトログラム
