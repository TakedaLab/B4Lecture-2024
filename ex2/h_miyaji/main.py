"""This module BEFフィルタを窓関数法で作成し、フィルタリングを行う.

It reads an input WAV file and フィルタリング

"""

import matplotlib.pyplot as plt  # グラフ描画
import myfunc as mf
import numpy as np  # 線形代数
import scipy.fftpack as fftpack  # フーリエ変
import scipy.signal as signal  # 窓関数
import soundfile as sf  # 音声読み込み


def conv(x: np.ndarray, h: np.ndarray, mode: str = "circ") -> np.ndarray:
    """2つのndarray配列に対して畳み込み演算を実行する

    Args:
        x (np.ndarray): 畳み込み演算の対象1
        h (np.ndarray): 畳み込み演算の対象2
        mode (str, optional): "circ" -> 循環畳み込み（Defaults）, "line" -> 線形畳み込み

    Returns:
        np.ndarray: 畳み込み演算の結果
    """
    # 循環畳み込みの場合 lenが異なればエラー
    if mode == "circ" and (len(x) != len(h)):
        print("Error: len(x) and len(h) must be equal")
        return -1

    # 畳み込み演算
    y = np.zeros(len(x) + len(h) - 1)
    for i in range(len(x)):
        y[i : i + len(h)] = x[i] * h

    # 線形・循環の場合分け
    if mode == "line":
        return y
    elif mode == "circ":
        n = len(x)
        y_over = y[n:]
        y[: n - 1] = y[: n - 1] + y_over
        return y[:n]


def conv_new(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """2つのndarray配列に対して畳み込み演算を実行する

    Args:
        x (np.ndarray): 畳み込み演算の対象1
        h (np.ndarray): 畳み込み演算の対象2

    Returns:
        np.ndarray: 畳み込み演算の結果
    """
    y = fftpack.fft(x) * fftpack.fft(h)
    return fftpack.ifft(y)


def make_bef(
    size: int, f1: float, f2: float, window: str, wlen: int, samplerate: int
) -> np.ndarray:
    """帯域消去フィルタの作成

    Args:
        size (int): フィルタのサイズ
        f1 (float): 通過域指定0 ~ f1[Hz]
        f2 (float): 通過域指定f2 ~ サンプリング周波数[Hz]
        window (str): 窓関数の種類
        wlen (int): 窓の長さ
        samplerate (int): サンプリング周波数

    Returns:
        np.ndarray: BEFの単位インパルス応答
    """
    # 与えられたタイプ・長さの窓関数を生成
    window_func = signal.get_window(window, wlen)

    time = size // samplerate  # 計測時間[s]
    N = wlen // 2  # 窓の長さの半分

    # フィルタ振幅特性の決定
    H_filter = np.zeros(size)
    H_filter[: int(f1 * time)] = 1
    H_filter[int(f2 * time) :] = 1

    # TODO: 確認用：理想のフィルタ振幅特性
    # freq = np.linspace(0, samplerate, size)
    # mf.plot_any(freq, H_filter, "Freq", "Amp", "Amp of filter")

    # 単位インパルス応答
    h_filter = fftpack.ifft(H_filter, n=size, axis=0)

    # TODO: 確認用：理想のフィルタに対する単位インパルス応答
    # mf.plot_any(freq[:200], h_filter[:200], "Time", "Amp", "impulse")

    # 時間シフト
    h_filter = np.roll(h_filter, N)

    # TODO: 確認用：時間シフト後
    # mf.plot_any(freq[:200], h_filter[:200], "Time", "Amp", "after roll")

    # 単位インパルス * 窓関数
    h_filter[:wlen] = h_filter[:wlen] * window_func
    h_filter[wlen + 1 :] = 0

    # TODO: 確認用：窓関数乗算後
    # mf.plot_any(freq[:200], h_filter[:200], "Time", "Amp", "h * window")

    # 時間シフト
    h_filter = fftpack.ifftshift(h_filter)
    h_filter = np.roll(h_filter, -N)

    # TODO: 確認用：中央に寄せた後
    # mf.plot_any(freq, h_filter, "Time", "Amp", "after timeshift")
    # mf.plot_any(
    #    freq[size // 2 - N : size // 2 + N],
    #    h_filter[size // 2 - N : size // 2 + N],
    #    "Time",
    #    "Amp",
    #    "after timeshift",
    # )
    return h_filter


def show_freq_responce(h: np.ndarray, samplerate: int) -> None:
    """周波数特性を図示する

    Args:
        h (np.ndarray): 単位インパルス応答
        samplerate (int): サンプリング周波数[Hz]

    Returns:
        None
    """
    # 単位インパルス応答->周波数応答
    h = fftpack.fftshift(h)
    H = fftpack.fft(h)

    # 振幅特性[dB]
    amp = 20 * np.log10(np.abs(H))

    # 位相特性[rad]
    deg = np.unwrap(np.angle(H, deg=True))

    # グラフ描画
    freq = np.linspace(0, samplerate, h.size)
    plt.figure()

    # 振幅特性
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(freq[: h.size // 2], amp[: h.size // 2], label="振幅特性")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude (dB)")

    # 位相特性
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(freq[: h.size // 2], deg[: h.size // 2], label="位相特性")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Degree")
    plt.subplots_adjust(hspace=0.7)

    plt.savefig("freq_responce.png")

    return None


def main():
    """入力信号をBEFでフィルタリングする"""
    args = mf.parse_args()  # 引数情報

    # 音声ファイルを読み込む, data:信号のデータ, samplerate:サンプリング周波数
    data, samplerate = sf.read(args.input_file)

    # 時間領域のプロット
    # mf.show_wave(data, samplerate)

    # argsの取得
    nfft = args.nfft
    hop_length = args.hop_length
    window = args.window
    f1 = args.f1
    f2 = args.f2
    # window_func = signal.get_window(window, nfft)

    # BEF作成
    h_bef = make_bef(len(data), f1, f2, window, 161, samplerate)

    # 畳み込み（失敗）
    # data_filtered = conv(data, h_bef)

    # TODO: 周波数領域から計算してみる（成功）
    # h = fftpack.fftshift(h_bef)
    # H = fftpack.fft(h)
    # H_data_filtered = fftpack.fft(data) * H
    # data_filtered = fftpack.ifft(H_data_filtered)

    # TODO: 畳み込み関数の中身を周波数領域計算にする（成功）
    h = fftpack.fftshift(h_bef)
    data_filtered = conv_new(data, h)

    # TODO: debug フィルタリング後の波形を表示
    print(data_filtered.shape)
    print(data_filtered.dtype)
    mf.show_wave(data_filtered, samplerate, "filtered wave")

    # 周波数特性の図示
    show_freq_responce(h_bef, samplerate)

    # スペクトログラムの計算
    spectrogram_original = mf.calc_spectrogram(data, nfft, hop_length, window)
    spectrogram_filtered = mf.calc_spectrogram(data_filtered, nfft, hop_length, window)

    # スペクトログラムの描画
    mf.show_spectrogram(
        spectrogram_original, samplerate, len(data), "Spectrogram original"
    )
    mf.show_spectrogram(
        spectrogram_filtered, samplerate, len(data_filtered), "Spectrogram filtered"
    )

    # フィルタリングされた信号のwavファイル作成
    sf.write("filtered_rec.wav", np.abs(data_filtered), samplerate)


if __name__ == "__main__":
    main()
