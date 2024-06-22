#!/bin/bash
# 上記の行はシェバン（shebang）と呼ばれ、シェルのパスを指定します。

export OMP_NUM_THREADS=1

# セットアップで作成した自分のvirtualenv環境をロード
source /work6/k_namizaki/.venv/bin/activate

python3 /work6/k_namizaki/B4Lecture-2024/ex9/k_namizaki/main9.py datadir=/work6/k_namizaki/B4Lecture-2024/ex9/k_namizaki/data

# 終了ステータスを出力（オプション）
echo "Python script executed with exit code $?"