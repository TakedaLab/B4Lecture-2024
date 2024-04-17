import numpy as np
import matplotlib.pyplot as plt
import librosa
import sys

def create_spectrogram(signal, n_fft, hop_length):
    n_frames = 1 + int((len(signal) - n_fft) / hop_length)
    spectrogram = np.zeros((n_fft, n_frames), dtype=np.complex_)

    for i in range(n_frames):
        frame = signal[i * hop_length:i * hop_length + n_fft]
        frame = np.pad(frame, (0, max(0, n_fft - len(frame))), mode='constant')
        windowed = frame * np.hanning(n_fft)
        fft_result = np.fft.fft(windowed)
        spectrogram[:, i] = fft_result

    return spectrogram

def inverse_spectrogram(spectrogram, n_fft, hop_length):
    n_frames = spectrogram.shape[1]
    output_length = n_fft + hop_length * (n_frames - 1)
    signal = np.zeros(output_length)

    for i in range(n_frames):
        frame = np.fft.ifft(spectrogram[:, i])
        start = i * hop_length
        signal[start:start + n_fft] += frame.real * np.hanning(n_fft)  # ifftの実数部分のみ利用

    return signal

# ファイルを読み込む
args = sys.argv
if len(args)==1:
    print("usage:main.py INPUT_FILE")
    exit()
filename = args[1]
signal, sr = librosa.load(filename, sr=None)
time = np.arange(0, len(signal)) / sr

# パラメータ
n_fft = 2048
hop_length = 512

# スペクトログラムの生成
spectrogram = create_spectrogram(signal, n_fft, hop_length)

# 逆変換
reconstructed_signal = inverse_spectrogram(spectrogram, n_fft, hop_length)

# スペクトログラムの表示
plt.subplots_adjust(wspace=0.4, hspace=1.0)
plt.subplot(221)
plt.plot(time, signal)
plt.title("Original Audio Signal")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")

plt.subplot(122)
plt.imshow(20 * np.log10(np.abs(spectrogram)[:n_fft // 2, :]), aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time Frame')
plt.ylabel('Frequency Bin')

plt.subplot(223)
plt.plot(time, reconstructed_signal)
plt.title("Restored Audio Signal")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")

plt.show()
