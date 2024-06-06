#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace5/venvs/py3venv/bin/activate
# 作成したPythonスクリプトを実行
python -u main.py --path_to_truth ../test_truth.csv