import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile


def STFT(data, Lw, step):
    # STFTの実行
    # data 信号,　Lw 窓幅, step 切り出し幅
    l = data.shape[0]
    win = np.hanning(Lw)
    Mf = Lw//2 + 1
    Nf = int(np.ceil((l-Lw+step)/step))
    S = np.empty([Mf, Nf], dtype=np.complex128)
    for i in range(Nf):
        start = int(i * step)
        end = int(start + Lw)
        segment = data[start:end] if end <= l else np.append(data[start:], np.zeros(end - l))
        S[:, i] = np.fft.rfft(segment * win, n=Lw, axis=0)
    return S

def ISTFT(S, Lw, step):
    # 逆変換の実行
    # S スペクトログラム
    Mf, Nf = S.shape
    l = int((Nf - 1) * step + Lw)

    win = np.hanning(Lw)
    x = np.zeros(l)

    for i in range(Nf):
        start = int(i * step)
        end = start + Lw
        x[start:end] += np.fft.irfft(S[:, i], n=Lw) * win

    return x

def main():
    # コマンドプロンプトからファイル名を受け取り
    args = sys.argv
    if(len(args) == 2):
        file_name = args[1]
    else:
        print("Usage: python main.py <file_name>")

    rate, data = wavfile.read(file_name)
    data = np.array(data, dtype=float)
    time = np.arange(0, len(data))/rate

    # 波形を表示
    plt.plot(time, data)
    plt.xlabel("Time[sec]")
    plt.ylabel("Amplitude")
    plt.title("original_signal")
    
    length_window = 512
    step = length_window/4

    spectrogrum = STFT(data, length_window, step)
    
    # スペクトログラムを表示
    P = 20*np.log10(np.abs(spectrogrum))
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
