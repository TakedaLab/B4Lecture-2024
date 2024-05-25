"""
pickleデータの読み込み

ForwardアルゴリズムとViterbiアルゴリズムを実装
出力系列ごとにどのモデルから生成されたか推定
正解ラベルと比較
混同行列 (Confusion Matrix) を作成
正解率 （Accuracy） を算出
アルゴリズムごとの計算時間を比較
"""

import argparse
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def parse_args():
    """
    Get Arguments.

    Returns
    -------
    parser.parse_args() : 引数を返す
    """
    parser = argparse.ArgumentParser(
        description="GMMを用いてデータのフィッティングを行う"
    )
    parser.add_argument(
        "-file",
        help="ファイルを入力",
        default=r"C:\Users\kyskn\B4Lecture-2024\ex6\data1.pickle",
        type=str,
    )
    return parser.parse_args()


def forward(output, trans_prob, out_prob, init_prob):
    """
    フォワードアルゴリズムをする

    Parameters
    ----------
    output : np.array[p, t]
        出力系列
    trans_prob : np.array[k, l, l]
        状態遷移確率
    out_prob : np.array[k, l, n]
        出力確率
    init_prob : np.array[k, l, 1]
        初期状態確率

    Returns
    -------
    forward_prob : np.array[p]
        フォワードアルゴリズムの結果
    """
    # p:実践回数, t:遷移回数
    p, t = output.shape
    # k:モデル数, l:状態数, n:出力記号数
    k, l, n = out_prob.shape

    forward_prob = np.zeros(p)

    # outputのi番目について
    for i in range(p):
        # alphaの初期化[k,l] = [k,l] * [k,l]
        alpha = init_prob[:, :, 0] * out_prob[:, :, output[i, 0]]
        for j in range(t):
            # alphaを最後のt回目まで回す{sum([k,l,newでl] * [k,l,l]) = [k,l]} * [k,l]
            alpha = (
                np.sum(alpha[:, :, np.newaxis] * trans_prob, axis=1)
                * out_prob[:, :, output[i, j]]
            )
        # P[k]（それぞれのモデルである確率）
        P = np.sum(alpha, axis=1)
        # outputのi番目が何のモデルの確率が一番高いか（argmaxでどの配列の要素が最大か取得）
        forward_prob[i] = np.argmax(P)

    return forward_prob


def viterbi(output, trans_prob, out_prob, init_prob):
    """
    viterbiアルゴリズムをする

    Parameters
    ----------
    output : np.array[p, t]
        出力系列
    trans_prob : np.array[k, l, l]
        状態遷移確率
    out_prob : np.array[k, l, n]
        出力確率
    init_prob : np.array[k, l, 1]
        初期状態確率

    Returns
    -------
    viterbi_prob : np.array[p]
        viterbiアルゴリズムの結果
    """
    # p:実践回数, t:遷移回数
    p, t = output.shape
    # k:モデル数, l:状態数, n:出力記号数
    k, l, n = out_prob.shape

    viterbi_prob = np.zeros(p)

    # outputのi番目について
    for i in range(p):
        # alphaの初期化[k,l] = [k,l] * [k,l]
        alpha = init_prob[:, :, 0] * out_prob[:, :, output[i, 0]]
        for j in range(t):
            # alphaを最後のt回目まで回す{sum([k,l,newでl] * [k,l,l]) = [k,l]} * [k,l]
            alpha = (
                np.sum(alpha[:, :, np.newaxis] * trans_prob, axis=1)
                * out_prob[:, :, output[i, j]]
            )
        # P[k]（それぞれのモデルである確率）
        P = np.max(alpha, axis=1)
        # outputのi番目が何のモデルの確率が一番高いか（argmaxでどの配列の要素が最大か取得）
        viterbi_prob[i] = np.argmax(P)

    return viterbi_prob


def plot_confusion(true, forward_prob, viterbi_prob):
    """
    混同行列を作成

    Parameters
    ----------
    true : np.array[p]
        正解ラベル
    forward_prob : np.array[p]
        フォワードアルゴリズムの結果
    viterbi_prob : np.array[p]
        viterbiアルゴリズムの結果
    """
    # Confusion Matrixを計算
    cm1 = confusion_matrix(true, forward_prob)
    cm2 = confusion_matrix(true, viterbi_prob)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # ヒートマップとしてConfusion Matrixを表示
    # forward
    forward_accurancy = 100 * accuracy_score(true, forward_prob)
    sns.heatmap(
        cm1,
        annot=True,
        cmap="Blues",
        fmt="g",
        xticklabels=["1", "2", "3", "4", "5"],
        yticklabels=["1", "2", "3", "4", "5"],
        ax=axs[0],
    )
    axs[0].set_xlabel("Predicted model", fontsize=16)
    axs[0].set_ylabel("Actual model", fontsize=16)
    axs[0].set_title(f"Forward algorithm\n(Acc.{forward_accurancy:.2f}%)", fontsize=16)

    # viterbi
    viterbi_accurancy = 100 * accuracy_score(true, viterbi_prob)
    sns.heatmap(
        cm2,
        annot=True,
        cmap="Blues",
        fmt="g",
        xticklabels=["1", "2", "3", "4", "5"],
        yticklabels=["1", "2", "3", "4", "5"],
        ax=axs[1],
    )
    axs[1].set_xlabel("Predicted model", fontsize=16)
    axs[1].set_ylabel("Actual model", fontsize=16)
    axs[1].set_title(f"Viterbi algorithm\n(Acc.{viterbi_accurancy:.2f}%)", fontsize=16)

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    # pickleからデータを取り出す
    data = pickle.load(open(args.file, "rb"))
    answer_models = np.array(data["answer_models"])
    output = np.array(data["output"])
    init_prob = np.array(data["models"]["PI"])
    trans_prob = np.array(data["models"]["A"])
    out_prob = np.array(data["models"]["B"])

    # フォワードアルゴリズムとviterbiアルゴリズムを実行（時間も計測）
    start_time = time.time()
    forward_prob = forward(output, trans_prob, out_prob, init_prob)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("forward処理時間:", elapsed_time, "秒")

    start_time = time.time()
    viterbi_prob = viterbi(output, trans_prob, out_prob, init_prob)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("viterbi処理時間:", elapsed_time, "秒")

    # プロットする
    plot_confusion(answer_models, forward_prob, viterbi_prob)


if __name__ == "__main__":
    main()
