"""
Convert Stereo to Monoral in Wav.
- Sampling Rate as 16kHz.
- Allows more than 2 channel.
- Requires Input&Output path.
"""
import soundfile as sf
import numpy as np
import resampy

from arg_reader import io_path


"""
Description shows on the first line. 
"""
def main(strInputPath:str, strOutputPath:str, intSamplingRate:int=16_000) -> None:
    # 元の音声ファイルを読み込む
    data, samplerate = sf.read(strInputPath)

    # ステレオからモノラルに変換する
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # サンプリングレートを16kHzに変換する
    target_rate = intSamplingRate
    if samplerate != target_rate:
        data = resampy.resample(data, samplerate, target_rate)

    # 16kHz、モノラルのwavファイルを書き出す
    sf.write(strOutputPath, data, target_rate)


if __name__ == "__main__":
    args = io_path(True, True)
    
    main(args.input_path, args.output_path)