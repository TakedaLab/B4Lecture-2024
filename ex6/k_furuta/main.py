"""出力系列から複数のHMMのうちどのモデルから出力されたものか推定するプログラム."""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    file_name: str,
    class_names=None,
):
    """
    プロット混同行列をプロットする関数.

    Parameter
    -------------
    true_labels : np.ndarray, shape=(sample,)
        正解ラベル
    predicted_labels : np.ndarray, shape=(sample,)
        予想ラベル
    file_name : str
        保存するファイル名
    class_names : list[str]
        プロットする際の列名
    """
    # 混同行列の計算
    cm = confusion_matrix(true_labels, predicted_labels)

    # クラス名の設定
    if class_names is None:
        class_names = np.unique(true_labels)

    # 混同行列のプロット
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    # タイトルとラベルの設定
    plt.title(
        f"Confusion Matrix (Acc. {np.sum(true_labels==predicted_labels)/true_labels.shape[0]*100}%)"
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # プロットの表示
    plt.savefig(file_name)


def forward(output, model_pi, model_a, model_b):
    """
    forwardアルゴリズムを用いてoutputが出力される確率を求める.

    Parameter
    -------------
    output : list[int]
        出力系列
    model_pi : list[int]
        初期状態の確率
    model_a : list[list[int]]
        状態遷移確率
    model_b : list[list[int]]
        出力確率
    Returns
    -------------
    probability : float
        モデルから計算されたoutputが出力される確率
    """
    # 扱いやすくするためにnp.ndarrayに変形
    model_pi = np.array(model_pi).reshape(-1)
    model_a = np.array(model_a)
    model_b = np.array(model_b)

    # 時間ごとに計算
    res = model_pi
    for i in range(len(output)):
        if i != 0:
            # 状態確立と状態遷移確率のドット積をとる(i!=1)
            res = res @ model_a

        # b[out[i]]とアダマール積をとる
        res = res * model_b[:, output[i]]

    # 状態の合計を返す
    return np.sum(res)


def viterbi(output, model_pi, model_a, model_b):
    """
    viterbiアルゴリズムを用いてoutputが出力される確率を求める.

    Parameter
    -------------
    output : list[int]
        出力系列
    model_pi : list[int]
        初期状態の確率
    model_a : list[list[int]]
        状態遷移確率
    model_b : list[list[int]]
        出力確率
    Returns
    -------------
    probability : float
        モデルから計算されたoutputが出力される確率
    """
    # 扱いやすくするためにnp.ndarrayに変形
    model_pi = np.array(model_pi).reshape(-1)
    model_a = np.array(model_a)
    model_b = np.array(model_b)

    # 時間ごとに計算
    res = model_pi
    for i in range(len(output)):
        if i != 0:
            # 状態確立と状態遷移確率のアダマール積をとる(i!=1)
            res = res.reshape(-1, 1) * model_a
            # 列ごとに最大値をとる
            res = np.max(res, axis=0)

        # b[out[i]]とアダマール積をとる
        res = res * model_b[:, output[i]]

    # maxを返す
    return np.max(res)


def parse_args():
    """コマンドプロントから引数を受け取るための関数."""
    parser = argparse.ArgumentParser(description="Plot PCA from data")
    parser.add_argument(
        "--input-file", type=str, required=True, help="Name of input csv file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 引数の受け取り
    args = parse_args()
    # ファイルパスとファイル名の受け取り
    file_path = args.input_file
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # モデルをロード
    with open("../data1.pickle", mode="rb") as file:
        data_dict = pickle.load(file)
        # モデルとデータを代入しておく
        output_data = data_dict["output"]
        label_data = data_dict["answer_models"]
        model_data = data_dict["models"]

    # 各出力ごとに各モデルで確立を計算して行列に積める(forward)
    model_kinds = len(model_data["A"])
    output_kinds = len(output_data)
    forward_result = np.zeros(shape=(model_kinds, output_kinds))
    for i in range(model_kinds):
        for j in range(output_kinds):
            forward_result[i, j] = forward(
                output_data[j],
                model_data["PI"][i],
                model_data["A"][i],
                model_data["B"][i],
            )

    # argmaxの結果とプロット
    forward_result = np.argmax(forward_result, axis=0)
    plot_confusion_matrix(label_data, forward_result, f"forward_{file_name}.png")

    # 各出力ごとに各モデルで確立を計算して行列に積める(viterbi)
    viterbi_result = np.zeros(shape=(model_kinds, output_kinds))
    for i in range(model_kinds):
        for j in range(output_kinds):
            viterbi_result[i, j] = viterbi(
                output_data[j],
                model_data["PI"][i],
                model_data["A"][i],
                model_data["B"][i],
            )
    # argmaxの結果とプロット
    viterbi_result = np.argmax(viterbi_result, axis=0)
    plot_confusion_matrix(label_data, viterbi_result, f"viterbi_{file_name}.png")
