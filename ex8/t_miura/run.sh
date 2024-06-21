#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace6/venvs/py3venv/bin/activate
# 使用スレッド数を1に指定する場合
export OMP_NUM_THREADS=1
# 作成したPythonスクリプトを実行
python main.py main --z_dim 2 --h_dim 400 --drop_rate 0.2 --learning_rate 0.001 --train_size_rate 0.8
