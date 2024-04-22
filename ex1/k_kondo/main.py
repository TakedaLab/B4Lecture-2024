import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile

def STFT(data, Lw, step):
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
    args = sys.argv
    file_name = args[1]
    

    rate, data = wavfile.read(file_name)
    data = np.array(data, dtype=float)

    time = np.arange(0, len(data))/rate
    plt.plot(time, data)
    plt.xlabel("Time[sec]")
    plt.ylabel("Amplitude")
    plt.title("original_signal")
    
    length_window = 512
    step = length_window/4

    spectrogrum = STFT(data, length_window, step)
    
    P = 20*np.log10(np.abs(spectrogrum))
    plt.figure()
    plt.imshow(P, origin="lower", aspect="auto")
    plt.xlabel("Time[sec]")
    plt.ylabel("Frequency[Hz]")
    plt.colorbar()
    plt.title("spectgrum")

    time_signal = ISTFT(spectrogrum, length_window, step)

    time_adjusted = np.arange(0, len(time_signal)) * step / rate
    plt.figure()
    plt.plot(time_adjusted, time_signal)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.title("inverse transform")
    plt.show()

if __name__ == "__main__":
    main()