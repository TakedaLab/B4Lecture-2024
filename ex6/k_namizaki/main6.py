"""
データの読み込み
必要なデータはpickleで配布する
読み込み方法や階層構造はヒントを参照
出力系列の尤度を計算
ForwardアルゴリズムとViterbiアルゴリズムを実装
出力系列ごとにどのモデルから生成されたか推定
正解ラベルと比較
混同行列 (Confusion Matrix) を作成
正解率 （Accuracy） を算出
アルゴリズムごとの計算時間を比較
"""

import pickle

def main():
    data = pickle.load(open(r"ex6\data1.pickle", "rb"))
    print(data)