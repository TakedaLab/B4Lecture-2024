#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source $h_yoshihara/workspace/venvs/py3venv/bin/activate
# 使用スレッド数を1に指定する場合
export OMP_NUM_THREADS=1
# 作成したPythonスクリプトを実行
python main.py