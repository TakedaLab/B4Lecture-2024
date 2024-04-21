import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import scipy.fftpack as fftpack

def main():
    # データを読み込み
    rate, data = wavfile.read(r"C:\Users\kyskn\B4Lecture-2024\k_namizaki\reco2mono.wav")

    # 波形データをplot
    time = np.arange(0, len(data)) / rate # 0~最後までを周波数で割る
    plt.plot(time, data)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude[unknown]")
    plt.title("Before")
    plt.show()

    # FFTのパラメータ
    nfft = 1024  # FFTサイズ
    hop_length = 512  # フレーム間のシフト量
    
    window =  signal.get_window("hann", nfft)

    # スペクトログラムの計算
    # スペクトログラムの初期化 np.zeros(行,列)
    spectrogram = np.zeros(
        (1 + nfft // 2, (len(data) - nfft) // hop_length + 1), dtype=np.complex128
    )
    # spectrogramの列数だけ回す
    for i in range(spectrogram.shape[1]):
        # 分割したものに窓関数をかける
        segment = data[i * hop_length : i * hop_length + nfft] * window
        # fftする[0 : 1 + nfft // 2]分だけ取り出す
        spectrum = fftpack.fft(segment, n=nfft)[: 1 + nfft // 2]
        # spectrogrumのi列目に保管
        spectrogram[:, i] = spectrum
        
    # スペクトログラムの描画
    
    plt.figure()
    # plt.imshow()
    plt.imshow(
        20 * np.log10(np.abs(spectrogram)), origin="lower", aspect="auto", cmap="jet"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()

    # 逆フーリエ変換を使ってオリジナルの波形データを再構築
    after_data = np.zeros(len(data), dtype=np.float64)
    # 0から len(data)-nfft-1 までの整数を、hop_length分シフトしながら区切る
    for i in range(0, len(data)-nfft, hop_length):
        # spectrogramの i//hop_length列目を保管
        spectrum = spectrogram[:, i//hop_length]
        # 逆フーリエ
        frame = np.fft.irfft(spectrum, n=nfft)
        # i:i+nfft の範囲に、波形データframeを加算
        after_data[i:i+nfft] += frame

    # オリジナルの波形データをplot
    plt.figure()
    plt.plot(time, after_data[:len(time)])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude[unknown]")
    plt.title("After")
    plt.show()

if __name__ == "__main__":
    main()
