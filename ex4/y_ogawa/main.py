"""主成分分析を行う."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """引数の取得を行う.

    filename : 読み込むファイル名
    """
    parser = argparse.ArgumentParser(description="主成分分析を行う")
    parser.add_argument("--filename", type=str, required=True, help="name of file")
    return parser.parse_args()


def make_scatter(data: np.ndarray, dim: int, title: str):
    """散布図を表示.

    Args:
        data : データ
        dim : 次元
    """
    # 次元が1の時の散布図描画
    if dim == 1:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
        y = np.zeros(len(data))
        # 散布図プロット
        ax.scatter(data, y, color="r", label="Data", marker=".")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=10)
        scale_min = min(data)
        scale_max = max(data)
        plt.xlim(scale_min, scale_max)
        plt.ylim(scale_min, scale_max)
        plt.savefig(title + "result.png")
        plt.show()

    # 次元が2の時の散布図描画
    if dim == 2:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
        # 散布図プロット
        ax.scatter(data[:, 0], data[:, 1], color="r", label="Data", marker=".")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=10)
        scale_min = min(min(data[:, 0]), min(data[:, 1]))
        scale_max = max(max(data[:, 0]), max(data[:, 1]))
        plt.xlim(scale_min, scale_max)
        plt.ylim(scale_min, scale_max)
        plt.savefig(title + "result.png")
        plt.show()

    # 次元が3の時の散布図描画
    if dim == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection="3d")
        # 散布図プロット
        ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            color="r",
            label="Data",
            marker=".",
        )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=10)
        scale_min = min(min(data[:, 0]), min(data[:, 1]), min(data[:, 2]))
        scale_max = max(max(data[:, 0]), max(data[:, 1]), max(data[:, 2]))
        plt.xlim(scale_min, scale_max)
        plt.ylim(scale_min, scale_max)
        plt.savefig(title + "result.png")
        plt.show()


def load_csv(filename: str) -> np.ndarray:
    """csvファイルを読み込み、データを独立変数と従属変数に分割する.

    Args:
        filename : 読み込むcsvファイル名
    Returns:
        data : 読み込んだデータ
    """
    data = np.loadtxt(filename, delimiter=",")  # CSVファイルの中身をNDArrayに格納
    return data


def standard_data(data: np.ndarray) -> np.ndarray:
    """データを標準化する.

    Args:
        data (np.ndarray): データ
    Returns:
        str_data (np.ndarray): 標準化したデータ
    """
    mean = np.mean(data, axis=0)  # 列ごとの平均を求める
    std = np.std(data, axis=0)  # 列ごとの標準偏差を求める
    std_data = (data - mean) / std  # 標準化の計算
    return std_data


def calc_contribution(eigen_vals: np.ndarray, dim: int):
    """寄与率、累積寄与率の計算.

    Args:
        eigen_vals (np.ndarray): 固有値
        dim (int): 次元
    Returns:
        contribution (np.ndarray): 寄与率
        sum_contribution (np.ndarray): 累積寄与率
        n_components (int): 累積寄与率が90%以上になるときの次元数
    """
    contribution = eigen_vals / sum(eigen_vals)
    sum_contribution = np.zeros(dim)
    n_components = 1
    for i in range(dim):
        sum_contribution[i] = np.sum(contribution[: i + 1])
        if sum_contribution[i] <= 0.9:
            n_components = i + 2
    return contribution, sum_contribution, n_components


def make_baseline(
    eigen_vec: np.ndarray, contribution: np.ndarray, dim: int, data: np.ndarray
):
    """基底を描画.

    Args:
        eigen_vec (np.ndarray): 基底ベクトル
        contribution (np.ndarray): 寄与率
        dim (int): 次元
        data (np.ndarray): データ
    """
    # 次元が2の時の基底描画
    if dim == 2:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
        # 散布図プロット
        ax.scatter(data[:, 0], data[:, 1], color="r", label="Data", marker=".")
        ax.axline(
            (0, 0),
            (eigen_vec[0, 0], eigen_vec[1, 0]),
            color="y",
            label="PC1_contribution: " + str(round(contribution[0], 2)),
        )
        ax.axline(
            (0, 0),
            (eigen_vec[0, 1], eigen_vec[1, 1]),
            color="b",
            label="PC2_contribution: " + str(round(contribution[1], 2)),
        )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("baseline_result")
        ax.legend(loc="upper left", fontsize=10)
        plt.savefig("baseline_result.png")
        plt.show()

    # 次元が3の時の散布図描画
    if dim == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection="3d")
        # 散布図プロット
        ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            color="r",
            label="Data",
            marker=".",
        )
        # 媒介変数
        t = np.linspace(-1, 1, 1000)
        x1 = t * eigen_vec[0, 0]
        y1 = t * eigen_vec[1, 0]
        z1 = t * eigen_vec[2, 0]
        x2 = t * eigen_vec[0, 1]
        y2 = t * eigen_vec[1, 1]
        z2 = t * eigen_vec[2, 1]
        x3 = t * eigen_vec[0, 2]
        y3 = t * eigen_vec[1, 2]
        z3 = t * eigen_vec[2, 2]
        ax.plot(
            x1,
            y1,
            z1,
            color="y",
            label="PC1_contribution: " + str(round(contribution[0], 2)),
        )
        ax.plot(
            x2,
            y2,
            z2,
            color="b",
            label="PC2_contribution: " + str(round(contribution[1], 2)),
        )
        ax.plot(
            x3,
            y3,
            z3,
            color="g",
            label="PC3_contribution: " + str(round(contribution[2], 2)),
        )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title("baseline_result")
        ax.legend(loc="upper left", fontsize=10)
        plt.savefig("baseline_result.png")
        plt.show()


def main():
    """読み込んだデータで主成分分析."""
    # 引数を受け取る
    args = parse_args()
    data = load_csv(args.filename)  # データの受け取り
    dim = len(data[0])  # 次元数

    # 散布図の表示
    make_scatter(data, dim, args.filename + "_plot")

    # データの標準化
    std_data = standard_data(data)
    trans_std_data = std_data.T

    # 共分散行列を求める
    cov_data = np.cov(trans_std_data)

    # 固有値と固有ベクトルを求める
    eigen_vals, eigen_vec = np.linalg.eig(cov_data)

    # 大きい順に並び変える
    index = np.argsort(eigen_vals)[::-1]
    sort_eigen_vals = eigen_vals[index]
    sort_eigen_vec = eigen_vec[index]

    # 寄与率、累積寄与率を求める
    contribution, sum_contribution, n_components = calc_contribution(
        sort_eigen_vals, dim
    )

    # 基底を表示
    make_baseline(sort_eigen_vec, contribution, dim, data)

    # 次元削減を行う
    components = sort_eigen_vec[:, :n_components]
    pca = np.dot(components.T, data.T)

    # 圧縮後のプロット
    if n_components == 2:
        make_scatter(pca.T, n_components, args.filename + "_pca")

    # 次元削減量を表示
    print("reduction: " + str(dim) + " -> " + str(n_components))


if __name__ == "__main__":
    main()
