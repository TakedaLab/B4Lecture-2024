"""forwardとviterbiアルゴリズムを実行し、処理時間を計測する."""

import argparse
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_data():
    """pickleファイルからデータを読み込む."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    data = pickle.load(open(args.filename, "rb"))

    # データの中身をnumpy配列に変換
    data["output"] = np.array(data["output"])
    data["models"]["PI"] = np.array(data["models"]["PI"])
    data["models"]["A"] = np.array(data["models"]["A"])
    data["models"]["B"] = np.array(data["models"]["B"])

    return args.filename, data


def forward(output, init_prob, trans_prob, output_prob):
    """forwardアルゴリズムを実行する.

    Args:
        output (np.ndarray): 出力系列 [p, t]
        init_prob (np.ndarray): 初期確率 [k, l, l]
        trans_prob (np.ndarray): 状態遷移確率行列 [k, l, l]
        output_prob (np.ndarray): 出力確率 [k, l, n]

    Returns:
        np.ndarray: 予測結果 [p]
    """
    # 出力系列の数(p)、状態数(l)、出力記号数(n)、モデル数(k)を取得
    p, t = output.shape
    k, l, n = output_prob.shape

    # forwardアルゴリズムの実行
    forward_result = np.zeros(p)
    for i in range(p):
        alpha = init_prob[:, :, 0] * output_prob[:, :, output[i, 0]]
        for j in range(1, t):
            alpha = (
                np.sum(trans_prob * alpha[:, :, np.newaxis], axis=1)
                * output_prob[:, :, output[i, j]]
            )

        # 尤度が最大となる状態を取得
        forward_result[i] = np.argmax(np.sum(alpha, axis=1))

    return forward_result


def viterbi(output, init_prob, trans_prob, output_prob):
    """viterbiアルゴリズムを実行する.

    Args:
        output (np.ndarray): 出力系列 [p, t]
        init_prob (np.ndarray): 初期確率 [k, l, l]
        trans_prob (np.ndarray): 状態遷移確率行列 [k, l, l]
        output_prob (np.ndarray): 出力確率 [k, l, n]

    Returns:
        np.ndarray: 予測結果 [p]
    """
    # 出力系列の数(p)、状態数(l)、出力記号数(n)、モデル数(k)を取得
    p, t = output.shape
    k, l, n = output_prob.shape
    # viterbiアルゴリズムの実行
    viterbi_result = np.zeros(p)
    for i in range(p):
        delta = init_prob[:, :, 0] * output_prob[:, :, output[i, 0]]
        for j in range(1, t):
            delta = (
                np.max(trans_prob * delta[:, :, np.newaxis], axis=1)
                * output_prob[:, :, output[i, j]]
            )

        # 尤度が最大となる状態を取得
        viterbi_result[i] = np.argmax(np.sum(delta, axis=1))

    return viterbi_result


def plot(predict, answer, algorithm, time):
    """結果をプロットする.

    Args:
        predict (np.ndarray): 予測結果 [p]
        answer (np.ndarray): 正解ラベル [p]
        algorithm (str): アルゴリズム名
        time (float): 処理時間
    """
    # 混合行列の作成
    cm = confusion_matrix(answer, predict)
    cm_df = pd.DataFrame(cm, columns=np.unique(answer), index=np.unique(answer))

    # 正解率の計算
    accuracy = np.sum(predict - answer == 0) / len(answer) * 100

    # プロット
    ax = sns.heatmap(
        cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, vmin=0, vmax=2
    )
    ax.set_title(f"{algorithm} (accuracy: {accuracy:.2f}%, time: {time:.2f}s)")
    ax.set_xlabel("Predicted model")
    ax.set_ylabel("Actual model")

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 1, 2])


def main():
    """forwardとviterbiを実行し、処理時間を計測."""
    # データの読み込み
    filename, data = load_data()
    answer_models = np.array(data["answer_models"])
    output = np.array(data["output"])
    init_prob = np.array(data["models"]["PI"])
    trans_prob = np.array(data["models"]["A"])
    output_prob = np.array(data["models"]["B"])

    # forwardとviterbiの実行及び処理時間の計測
    start_time = time.time()
    forward_result = forward(output, init_prob, trans_prob, output_prob)
    end_time = time.time()
    forward_time = end_time - start_time
    print("forward_time: ", forward_time, "s")

    start_time = time.time()
    viterbi_result = viterbi(output, init_prob, trans_prob, output_prob)
    end_time = time.time()
    viterbi_time = end_time - start_time
    print("viterbi_time: ", viterbi_time, "s")

    # 結果のプロット
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot(forward_result, answer_models, "forward", forward_time)
    plt.subplot(1, 2, 2)
    plot(viterbi_result, answer_models, "viterbi", viterbi_time)
    plt.tight_layout
    title = filename.replace("../", "").replace(".pickle", "")
    plt.savefig(f"{title}.png")
    plt.show()


if __name__ == "__main__":
    main()
