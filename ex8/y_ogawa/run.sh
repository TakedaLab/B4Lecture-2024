#!/bin/bash
source venv/bin/activate

# 仮想環境内のPythonインタープリターを使用する
python main.py main --z_dim 2 --h_dim 400 --drop_rate 0.2 --learning_rate 0.001 --train_size_rate 0.8
