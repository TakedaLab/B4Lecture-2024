"""pickleのファイルを読み込みHMMの予測を行う."""

import argparse
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def parse_args():
    """引数の取得を行う.

    filename : 読み込むファイル名
    n_cluster : クラスター数
    """
    parser = argparse.ArgumentParser(description="HMMの予測")
    parser.add_argument("--filename", type=str, required=True, help="name of file")
    return parser.parse_args()


def load_pickle(filename: str) -> np.ndarray:
    """
    pickleファイルの読み込みを行う.

    filename : 読み込むファイル名

    return
    answer_models (np.ndarray): 正解ラベル
    output (np.ndarray): 出力ラベル
    PI (np.ndarray): 初期確率
    A (np.ndarray): 遷移確率
    B (np.ndarray): 出力確率
    """
    # pickleファイルの読み込み
    data = pickle.load(open(filename, "rb"))
    answer_models = np.array(data["answer_models"])
    output = np.array(data["output"])
    PI = np.array(data["models"]["PI"])
    A = np.array(data["models"]["A"])
    B = np.array(data["models"]["B"])

    return answer_models, output, PI, A, B


# Forwardアルゴリズム
def forward(
    output: np.ndarray, PI: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """
    Forwardアルゴリズムを実行する.

    output (np.ndarray): 出力ラベル
    PI (np.ndarray): 初期確率
    A (np.ndarray): 遷移確率
    B (np.ndarray): 出力確率
    """
    # n_samples : 出力系列数(p), time: 遷移回数(s)
    n_samples, time = output.shape

    # n_model: モデルの数(k), n_state: 状態数(l), n_output: 出力の状態数(n)
    n_model, n_state, n_output = B.shape

    probability = np.zeros((n_model, n_samples))

    # Forwardアルゴリズム
    for k in range(n_model):
        for p in range(n_samples):
            # alphaの初期化
            alpha = np.zeros((time, n_state))
            alpha[0, :] = PI[k].T[0] * B[k, :, output[p, 0]]

            # alphaの計算
            for t in range(1, time):
                # alphaの更新
                alpha[t, :] = (
                    # 前の状態の確率と遷移確率の積
                    np.dot(alpha[t - 1, :], A[k, :, :])
                    * B[k, :, output[p, t]]
                )

            # 確率の計算
            probability[k, p] = np.sum(alpha[-1, :])

    return probability


# Viterbiアルゴリズム
def viterbi(
    output: np.ndarray, PI: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """
    Viterbiアルゴリズムを実行する.

    output (np.ndarray): 出力ラベル
    PI (np.ndarray): 初期確率
    A (np.ndarray): 遷移確率
    B (np.ndarray): 出力確率
    """
    # n_samples : 出力系列数(p), time: 遷移回数(s)
    n_samples, time = output.shape

    # n_model: モデルの数(k), n_state: 状態数(l), n_output: 出力の状態数(n)
    n_model, n_state, n_output = B.shape

    probability = np.zeros((n_model, n_samples))

    # Viterbiアルゴリズム
    for k in range(n_model):
        for p in range(n_samples):
            # deltaの初期化
            delta = np.zeros((time, n_state))
            delta[0, :] = PI[k].T[0] * B[k, :, output[p, 0]]

            # deltaの計算
            for t in range(1, time):
                # deltaの更新
                delta[t, :] = (
                    # 前の状態の確率と遷移確率の積
                    np.max(delta[t - 1, :, np.newaxis] * A[k, :, :], axis=0)
                    * B[k, :, output[p, t]]
                )

            # 確率の計算
            probability[k, p] = np.max(delta[-1, :])

    return probability


def plot_confusion_matirx(
    answer: np.ndarray, pred: np.ndarray, algorithm: str, time: float
):
    """
    混同行列のプロットを行う.

    answer (np.ndarray): 正解ラベル
    pred (np.ndarray): 予測ラベル
    algorithm (str): アルゴリズム名
    time (float): 実行時間
    """
    n_class = np.unique(answer)
    cm = confusion_matrix(answer, pred)
    cm_df = pd.DataFrame(cm, columns=n_class, index=n_class)
    # 正解率の計算
    acc = np.sum(pred - answer == 0) / len(answer) * 100
    # 混合行列のプロット
    ax = sns.heatmap(
        cm_df, annot=True, cbar=True, cmap="Blues", fmt="d", vmin=0, vmax=20
    )

    # カラーバーの取得と整数ティックの設定
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.arange(0, 21, 5))

    plt.title(algorithm + f"\nAccuracy: {acc}%\n" + f"time: {time}s")
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")


def main():
    """pickleのファイルを読み込みHMMの予測を行う."""
    # 引数を受け取る
    args = parse_args()

    # pickleファイルの読み込み
    answer_models, output, PI, A, B = load_pickle(args.filename)

    # Forwardアルゴリズム
    forward_start = time.time()
    probability_forward = forward(output, PI, A, B)
    forward_end = time.time()
    forward_time = round(forward_end - forward_start, 3)

    # Viterbiアルゴリズム
    viterbi_start = time.time()
    probability_viterbi = viterbi(output, PI, A, B)
    viterbi_end = time.time()
    viterbi_time = round(viterbi_end - viterbi_start, 3)

    # 混同行列のプロット
    plt.figure(figsize=(11, 6))
    plt.subplot(121)
    plot_confusion_matirx(
        answer_models, np.argmax(probability_forward, axis=0), "Forward", forward_time
    )
    plt.subplot(122)
    plot_confusion_matirx(
        answer_models, np.argmax(probability_viterbi, axis=0), "Viterbi", viterbi_time
    )
    plt.tight_layout()
    title = args.filename.replace("../", "").replace(".pickle", "")
    plt.savefig(title + ".png")
    plt.show()


if __name__ == "__main__":
    main()
