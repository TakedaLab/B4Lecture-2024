#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/workspace5/ex9/venvs/py3venv/bin/activate
# 作成したPythonスクリプトを実行
set HYDRA_FULL_ERROR=1
python -u main.py