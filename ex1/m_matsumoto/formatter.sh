#!/bin/bash

# 引数の数を確認
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_files>"
    exit 1
fi

# ワイルドカードを展開して、引数として渡されたすべてのファイルに対して処理を行う
for input_file in "$@"; do
    echo "####################"
    echo "<<<<<$input_file>>>>>"
    # isortコマンドを実行
    # echo "###I###"
    isort "$input_file"
    # echo "###I###"

    # blackコマンドを実行
    # echo "###B###"
    black "$input_file"
    # echo "###B###"
    
    # flake8コマンドを実行
    # echo "###F###"
    flake8 "$input_file"
    # echo "###F###"
    echo "####################"
done
