#!/bin/bash
# 上記の行はシェバン（shebang）と呼ばれ、シェルのパスを指定します。

export OMP_NUM_THREADS=1

# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace6/venvs/py3venv/bin/activate

# Pythonスクリプトの実行
python3 /home/k_namizaki/workspace6/B4Lecture-2024/ex7/k_namizaki/main7.py --path_to_truth /home/k_namizaki/workspace6/B4Lecture-2024/ex7/test_truth.csv
# 終了ステータスを出力（オプション）
echo "Python script executed with exit code $?"
