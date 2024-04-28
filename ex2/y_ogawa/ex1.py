"""

wav形式の音声ファイルの波形、スペクトログラムを作成する.

逆変換によって元の信号も出力する。
"""

import sys

import librosa  # 音声読み込み用
import matplotlib.pyplot as plt  # グラフ作成用
import numpy as np


# スペクトログラムを作成する
def makeSpectrogram(waveSize, wave, sr, flameSize, overlap):
    """短時間フーリエ変換を行い、スペクトログラムを計算します.

    Args:
        waveSize: フレーム数
        wave: 波形の値
        sr: サンプリング周波数
        flameSize: フレームの大きさ
        overlap: オーバーラップさせるサンプル数

    Returns:
        短時間フーリエ変換の結果を格納したものを返す
    """
    spectrogram = []  # 保持用
    window = np.hanning(flameSize)  # 窓関数としてハニング窓を採用
    flameStart = int(
        sr / 10
    )  # 波形の初めの部分が0となってしまうため0.1秒分スキップした
    while (
        flameStart + flameSize <= waveSize
    ):  # フレームの端が波のサンプル数を超えるまで繰り返す
        cutWave = wave[
            flameStart : flameStart + flameSize
        ]  # 音声波形から短時間区間を切り出す
        fft = np.fft.fft(window * cutWave)  # 窓関数をかけて高速フーリエ変換
        spectrogram.append(fft[0:flameSize])  # 結果を格納する
        flameStart += overlap  # 次のフレーム
    return spectrogram


# 変換して元の波形にする
def inverseWave(waveSize, spectrogram, sr, flameSize, overlap):
    """逆変換を行い、結果を返します.

    Args:
        waveSize: フレーム数
        Spectrogram: スペクトログラム
        sr: サンプリング周波数
        flameSize: フレームの大きさ
        overlap: オーバーラップさせるサンプル数

    Returns:
        逆変換の結果を格納したものを返す
    """
    iWave = np.zeros(waveSize)  # 保持用の配列を波のフレーム数だけ用意する
    flameStart = int(sr / 10)  # スキップした分だけ遅らせる
    window = np.hanning(flameSize)  # 窓関数としてハニング窓を採用
    i = 0
    while (
        flameStart + flameSize <= waveSize
    ):  # フレームの端が波のサンプル数を超えるまで繰り返す
        frame = np.fft.ifft(spectrogram[i]) / window  # スペクトログラムを逆変換する
        iWave[flameStart : flameStart + overlap] += np.real(
            frame[:overlap]
        )  # 実数部分を窓関数で割ってiWaveに格納する
        flameStart += overlap  # 次のフレーム
        i += 1  # 次の配列
    return iWave


def main():
    """コマンドラインからファイル名を受け取り、音声ファイルの読み込み、スペクトログラムの作成、逆変換を行い、それぞれの結果をプロットして表示する.

    Args:
        なし

    Returns:
        なし
    """
    args = sys.argv
    filename = args[1]  # filename をコマンドラインから取得
    y, sr = librosa.load(filename, sr=None)  # 音声データを取得　srはサンプリング周波数
    time = np.arange(0, y.size / sr, 1 / sr)  # 時間軸を配列に格納

    # 元の波形を表示する
    plt.subplot(3, 1, 1)
    plt.title("original wave")
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.plot(time, y)

    # スペクトログラムを作成する
    flameSize = 1024  # フレームサイズ
    overlap = int(flameSize / 2)  # オーバーラップ数
    spectrogram = makeSpectrogram(y.size, y, sr, flameSize, overlap)
    spectrogram_db = 20 * np.log(np.abs(spectrogram).T)  # 単位をdbに変換する

    # スペクトログラムを表示する
    extent = (0.1, y.size / sr, 0, sr / 2)  # 縦軸はナイキスト周波数まで
    plt.subplot(3, 1, 2)
    # フーリエ変換の実数部分だけをみるので半分だけ
    plt.imshow(
        spectrogram_db[: int(len(spectrogram) / 2)],
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="jet",
    )
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar()
    plt.title("Spectrogram")

    # 逆変換後の波形を表示する
    iWave = inverseWave(y.size, spectrogram, sr, flameSize, overlap)
    plt.subplot(3, 1, 3)
    plt.title("inverse wave")
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.plot(time, iWave)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
