"""
This code is to do the Principal Component Analysis.

Do the standardization.
Find eigenvalues and eigenvectors of the covariance matrix.
Transform matrix.
Calculate the contribution ratio.
Plot data.
"""

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def parse_args() -> Any:
    """
    Get Arguments.

    Returns
    -------
    parser.parse_args() : 引数を返す
    """
    parser = argparse.ArgumentParser(description="最小二乗法を用いて回帰分析を行う。")
    parser.add_argument(
        "-file",
        help="ファイルを入力",
        default=r"C:\Users\kyskn\B4Lecture-2024\ex4\data1.csv",
        type=str,
    )
    parser.add_argument("-n", help="次数", default=1, type=int)
    parser.add_argument("-normal", help="正則化係数", default=0, type=int)
    return parser.parse_args()


def plot2d(data: ArrayLike, Eigenvectors: ArrayLike, rate: ArrayLike) -> None:
    """
    Plot in 2 dimensions.

    Parameters
    ----------
    data (array-like): データ
    Eigenvectors (array-like): 固有ベクトル
    rate (array-like): 寄与率
    """
    # グラフと実際の点を描画
    fig, ax = plt.subplots()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.scatter(data[:, 0], data[:, 1], color="b", label="data1")

    # 固有ベクトルをプロット
    vec1 = Eigenvectors.T[0]
    ax.plot(
        [-2 * vec1[0], 2 * vec1[0]],
        [-2 * vec1[1], 2 * vec1[1]],
        color="red",
        label=round(rate[0], 2),
    )
    vec2 = Eigenvectors.T[1]
    ax.plot(
        [-2 * vec2[0], 2 * vec2[0]],
        [-2 * vec2[1], 2 * vec2[1]],
        color="green",
        label=round(rate[1], 2),
    )
    ax.legend()
    plt.show()


def plot3d(data: ArrayLike, Eigenvectors: ArrayLike, rate: ArrayLike) -> None:
    """
    Plot in 3 dimensions.

    Parameters
    ----------
    data (array-like): データ
    Eigenvectors (array-like): 固有ベクトル
    rate (array-like): 寄与率
    """
    # グラフと実際の点を描画
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="b", label="data2")

    vec1 = Eigenvectors.T[0]
    ax.plot(
        [-2 * vec1[0], 2 * vec1[0]],
        [-2 * vec1[1], 2 * vec1[1]],
        [-2 * vec1[2], 2 * vec1[2]],
        color="r",
        label=round(rate[0], 2),
    )
    vec2 = Eigenvectors.T[1]
    ax.plot(
        [-2 * vec2[0], 2 * vec2[0]],
        [-2 * vec2[1], 2 * vec2[1]],
        [-2 * vec2[2], 2 * vec2[2]],
        color="g",
        label=round(rate[1], 2),
    )
    vec3 = Eigenvectors.T[2]
    ax.plot(
        [-2 * vec3[0], 2 * vec3[0]],
        [-2 * vec3[1], 2 * vec3[1]],
        [-2 * vec3[2], 2 * vec3[2]],
        color="y",
        label=round(rate[2], 2),
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def plot_comp(Y: ArrayLike) -> None:
    """
    Plot the data after compression.

    Parameters
    ----------
    Y (array-like): 変換後のデータ
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.scatter(Y[:, 0], Y[:, 1], color="b", label="compression data")
    ax.legend()
    plt.show()


# 標準化
def standardization(data: ArrayLike) -> ArrayLike:
    """
    Do the standardization.

    Parameters
    ----------
    data (ArrayLike): データ

    Returns
    -------
    st_data (ArrayLike): 標準化データ
    """
    # x = (x' - μ)/σ
    # np.mean(array_, axis = 0) # 列ごとに集計した平均
    # np.std(array_, axis = 0) # 列ごとに集計した標本標準偏差
    st_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return st_data


def transformation_matrix(data: ArrayLike, Eigenvectors: ArrayLike) -> ArrayLike:
    """
    Transform matrix.

    Parameters
    ----------
    data (ArrayLike): データ
    Eigenvectors (ArrayLike): 固有ベクトル

    Returns
    -------
    Y (ArrayLike): 変換後のデータ
    """
    Y = Eigenvectors.T @ data.T
    return Y.T


def cul_rate(Eigenvalues: ArrayLike) -> ArrayLike:
    """
    Calculate the contribution ratio.

    Parameters
    ----------
    Eigenvalues (ArrayLike): 固有ベクトル

    Returns
    -------
    sorted_rate (ArrayLike): 大きい順の寄与率
    """
    rate = Eigenvalues / sum(Eigenvalues)
    sorted_rate = np.sort(rate)[::-1]
    return sorted_rate


def compression(rate: ArrayLike) -> ArrayLike:
    """
    Compress the dimension to a cumulative contribution ratio of 90%.

    Parameters
    ----------
    rate (ArrayLike): 寄与率

    Returns
    -------
    cum_rate (ArrayLike): 累積寄与率
    """
    # 累積寄与率90%以上とまで次元圧縮
    ac_rate = 0.0
    i = 0
    cum_rate = []
    while ac_rate < 0.9:
        ac_rate += rate[i]
        print(ac_rate)
        cum_rate.append(ac_rate)
        i += 1
    return cum_rate


def main() -> None:
    """Do the Principal Component Analysis."""
    args = parse_args()
    data = np.loadtxt(args.file, delimiter=",", dtype="float")
    # 標準化
    st_data = standardization(data)
    # 共分散行列(bias=1で標本共分散)
    cov_data = np.cov(st_data, rowvar=0, bias=1)
    # 共分散行列の固有値と固有ベクトル
    eig_data = np.linalg.eig(cov_data)
    Eigenvalues = eig_data[0]
    Eigenvectors = eig_data[1]
    print(Eigenvectors)
    # 返還後の行列
    Y = transformation_matrix(data, Eigenvectors)
    # 寄与率
    rate = cul_rate(Eigenvalues)
    # データそれぞれ
    if len(data[0]) == 2:
        plot2d(data, Eigenvectors, rate)
    elif len(data[0]) == 3:
        plot3d(data, Eigenvectors, rate)
        plot_comp(Y)
    else:
        cum_rate = compression(rate)
        print("contribution rate", rate)
        print("Cumulative contribution rate", cum_rate)
        print("圧縮前の次元は", len(Y[0]), "圧縮後の次元は", len(cum_rate))


if __name__ == "__main__":
    main()
