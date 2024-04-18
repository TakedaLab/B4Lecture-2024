import numpy as np
import matplotlib.pyplot as plt
import librosa
import sys

def create_spectrogram(signal, n_fft, hop_length):
    '''
    音声波形からスペクトログラムを作成する
    
    Parameters
    -------------
    signal : ndarray
        波形データ
    n_fft : int
        fftを行う窓の幅
    hop_length : int
        窓ごとの間隔

    Returns
    ------------
    spectrogram : ndarray
        波形データから計算されたスペクトログラム
    '''
    n_frames = 1 + int((len(signal) - n_fft) / hop_length) #窓のサイズと窓ごとの間隔でフレーム数を決定
    spectrogram = np.zeros((n_fft, n_frames), dtype=np.complex_) #データを格納するndarrayを作成

    #フレームごとに窓関数を掛けてスペクトルを計算
    for i in range(n_frames):
        frame = signal[i * hop_length:i * hop_length + n_fft]
        frame = np.pad(frame, (0, max(0, n_fft - len(frame))), mode='constant') #フレームが足りない場合は0埋め
        windowed = frame * np.hanning(n_fft)
        fft_result = np.fft.fft(windowed)
        spectrogram[:, i] = fft_result

    return spectrogram

#スペクトログラムから音声波形を復元
def inverse_spectrogram(spectrogram, n_fft, hop_length):
    '''
    音声波形からスペクトログラムを作成する
    
    Parameters
    -------------
    spectrogram : ndarray
        スペクトログラム
    n_fft : int
        fftを行う窓の幅
    hop_length : int
        窓ごとの間隔

    Returns
    ------------
    signal : ndarray
        スペクトログラムから計算された波形データ
    '''
    n_frames = spectrogram.shape[1]
    output_length = n_fft + hop_length * (n_frames - 1)
    signal = np.zeros(output_length)

    for i in range(n_frames):
        frame = np.fft.ifft(spectrogram[:, i])
        start = i * hop_length
        # ifftの実数部分のみ利用(丸め誤差で微小な虚部が発生しているため、結果には影響しない)
        signal[start:start + n_fft] += frame.real * np.hanning(n_fft)  
    return signal

# コマンドライン引数ファイルを読み込む
args = sys.argv
if len(args)==1:
    #ファイル名未指定の場合のエラー処理
    print("usage:main.py INPUT_FILE")
    exit()

filename = args[1]
signal, rate = librosa.load(filename, sr=None)

# パラメータ
n_fft = 1024
hop_length = 512

# スペクトログラムの生成
spectrogram = create_spectrogram(signal, n_fft, hop_length)

# 逆変換
reconstructed_signal = inverse_spectrogram(spectrogram, n_fft, hop_length)

# スペクトログラムと信号の表示
time = np.arange(0, len(signal)) / rate # 音声波形をプロットするための時間データ

plt.subplots_adjust(wspace=0.4, hspace=1.0)
plt.subplot(221)
plt.plot(time, signal)
plt.title("Original Audio Signal")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")

plt.subplot(122)
#x軸とy軸の表示範囲
#  x軸は音声の時間
#  サンプリング定理よりy軸はサンプリング周波数の半分？
extent = (0, len(signal)/rate, 0, rate/2)
#実数信号なので周波数はsymmetricになっているため半分だけ表示
#デシベルと振幅の変換 振幅:x -> デシベル:20*log_10(x) 
plt.imshow(20 * np.log10(np.abs(spectrogram)[:n_fft // 2, :]), extent=extent, aspect='auto', origin='lower', cmap='viridis')

plt.title('Spectrogram')
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [hz]')

plt.subplot(223)
plt.plot(time, reconstructed_signal)
plt.title("Restored Audio Signal")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")

plt.show()
