#!/bin/bash
# This Module Convert .m4a file to .wav file

# コマンドライン引数からファイル名を取得
input_file=$1
output_file=$2

# ffmpegが存在するかチェック
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg could not be found. Please install ffmpeg first."
    exit 1
fi

# 引数がない場合はエラーメッセージを表示して終了
if [ -z "$input_file" ] || [ -z "$output_file" ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

# m4aファイルをwavファイルに変換
# -vn: ビデオストリームなし
# -ar: 16kHz
# -ac: channel
# -ab: bit rate
# -f: wavファイル
ffmpeg -i "$input_file" -vn -ar 16000 -ac 1 -ab 192k -f wav "$output_file"

# 変換が成功したかどうかを確認
if [ $? -eq 0 ]; then
    echo "Conversion succeeded: $input_file => $output_file"
else
    echo "Conversion failed"
fi