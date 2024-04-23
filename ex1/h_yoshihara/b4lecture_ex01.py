from pydub import AudioSegment
import soundfile
import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sp
import copy


#音声ファイルがステレオなのでモノラルに変換
stereo_audio = '月まで届け.wav'  #音声ファイルのパス
sound = AudioSegment.from_wav(stereo_audio)
mono_audio = sound.set_channels(1)  #ステレオ音声をモノラルの変換
mono_audio.export("mono_shuffle.wav", format="wav")  #変換後の音声を保存

audio_path = 'mono_shuffle.wav'
data, samplerate = soundfile.read(audio_path)  #オーディオデータとサンプリングレートの取り出し
N = len(data) #オーディオデータのサイズを取得

time = np.arange(0, N) / samplerate  #音声データの時間を取得
plt.plot(time, data)  #音声ファイルの波形を描写
plt.xlabel("Time[s]")
plt.ylabel("Amplitude")
plt.show()
plt.close()


fft_data = []  #フーリエ変換後のデータを逐次保存
fs = 1024  #切り出すフレームの幅
overlap = 512  #オーバーラップするフレーム幅（今回は50%で）
cut_surplus = fs - overlap  #切り出していくフレーム間の幅

wd = sp.hamming(fs)  #切り出すフレーム幅で窓関数を用意（今回はハミング窓）

cut_start = 0  #切り出していくフレーム位置
while cut_start + fs <= N:
  frame = data[cut_start:cut_start + fs]  #音声データから切り出し
  fft_frame = np.fft.fft(frame * wd)  #切り出したデータに窓関数をかけ、フーリエ変換
  fft_data.append(fft_frame)  #フーリエ変換後のデータを保存
  cut_start += cut_surplus  #切り出し開始位置を更新

fft_data_copy = copy.deepcopy(fft_data) #逆変換用
fft_data = np.array(fft_data)  #np.absを使いたいので
fft_abs = np.abs(fft_data[:, :cut_surplus]).T  #オーバーラップ部分を省きつつ絶対値をとる

spectrogram = 20 * (np.log10(fft_abs))  #20*log(振幅)でデシベルに変換できるらしい?
fig, ax = plt.subplots()
im = ax.imshow(spectrogram,
               origin="lower",
               aspect="auto",
               extent=(0, N / samplerate, 0, samplerate // 2),
               cmap="plasma")
plt.colorbar(im, aspect=3, location="right")
plt.xlabel("Time[s]")
plt.ylabel("Frequency[Hz]")
plt.show()
plt.close()


ifft_data = np.zeros(N)  #逆変換後のデータを逐次保存
cut_start = 0  #切り出し開始位置
for fft_frame in fft_data_copy:
  ifft_frame = np.fft.ifft(fft_frame) / wd  #逆変換し、窓関数で割る
  ifft_data[cut_start:cut_start + cut_surplus] += np.real(ifft_frame[:cut_surplus])  #逆変換後は虚数部分が邪魔になったので削除
  cut_start += cut_surplus  #切り出し開始位置更新

plt.plot(time, ifft_data)
plt.xlabel("Time[s]")
plt.ylabel("Amplitude")
plt.show()