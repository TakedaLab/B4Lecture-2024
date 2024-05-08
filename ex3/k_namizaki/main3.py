"""
This code is to do regression analyzed using the least squares method.

Calculate weights.
Plot data.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
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
        default=r"C:\Users\kyskn\B4Lecture-2024\ex3\k_namizaki\data3.csv",
        type=str,
    )
    parser.add_argument("-n", help="次数", default=1, type=int)
    parser.add_argument("-normal", help="正則化係数", default=0, type=int)
    return parser.parse_args()


from numpy.typing import ArrayLike

def plot2d(x: ArrayLike, y: ArrayLike, w: ArrayLike):
    """
    Plot in 2 dimensions.

    Parameters
    ----------
    x (array-like): xのデータ
    y (array-like): yのデータ
    w (array-like): 重み
    """
    # 解答グラフ作成
    x_ans = np.linspace(np.min(x), np.max(x), 100)
    # np.poly1d()は、最高次数の係数から始めないとダメ
    f = np.poly1d(w[::-1])
    y_ans = f(x_ans)

    # グラフと実際の点を描画
    fig, ax = plt.subplots()
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.scatter(x, y, label="Observed data")
    ax.plot(x_ans, y_ans, label="ans", color="r")
    ax.legend()
    plt.show()


def plot3d(x, y, z, w, degree):
    """
    Plot in 3 dimensions.

    Parameters
    ----------
    x (array-like): xのデータ
    y (array-like): yのデータ
    z (array-like): zのデータ
    w (array-like): 重み
    degree : 次数
    """
    # 解答グラフ作成
    x_ans = np.linspace(np.min(x), np.max(x), 100)
    y_ans = np.linspace(np.min(y), np.max(y), 100)
    # meshgrid必須
    X, Y = np.meshgrid(x_ans, y_ans)
    # np.poly1d()は、最高次数の係数から始めないとダメ
    f = np.poly1d(w[: degree + 1][::-1])
    # 長さを同じにしつつ、1の係数部分を消すために0をappend
    g = np.poly1d(np.append(w[degree + 1 :][::-1], 0))
    # z00 = (...x0^2 + x0 + 1) + (... y0^2 + y0 + 0)
    z_ans = f(X) + g(Y)

    # グラフと実際の点を描画
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, label="Observed data")
    ax.plot_surface(X, Y, z_ans, label="ans", color="r", alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def weight2d(x, y, degree, normal):
    """
    Calculate weights from 2 sets of data.

    Parameters
    ----------
    x (array-like): xのデータ
    y (array-like): yのデータ
    degree : 次数
    normal : 正則化係数

    Returns
    -------
    w (array-like): 重み
    """
    # X = 1+x+x^2+...
    X = np.zeros((len(x), degree + 1))
    Y = y
    # x[:, np.newaxis] はxを縦ベクトルにする
    # np.arange(degree + 1) は、0 から degree までの整数の配列を生成
    # 最後に[np.newaxis, :] を使って、この配列を行ベクトルに変形
    X = x[:, np.newaxis] ** np.arange(degree + 1)[np.newaxis, :]
    # w = (X.T @ X + λ @ I)^-1 @ X.T @ Y
    w = np.zeros(degree + 1)
    Im = np.identity(len(X[0]))
    w = np.linalg.inv(X.T @ X + normal * Im) @ X.T @ Y
    return w


def weight3d(x, y, z, degree, normal):
    """
    Calculate weights from 3 sets of data.

    Parameters
    ----------
    x (array-like): xのデータ
    y (array-like): yのデータ
    z (array-like): zのデータ
    degree : 次数
    normal : 正則化係数

    Returns
    -------
    w (array-like): 重み
    """
    # X = 1+x+x^2+...+y+y^2+...
    X = np.zeros((len(x), 2 * degree + 1))
    Z = z
    # x[:, np.newaxis] はxを縦ベクトルにする
    # np.arange(degree + 1) は、0 から degree までの整数の配列を生成
    # [np.newaxis, :] を使って、整数の配列を行ベクトルに変形
    X = x[:, np.newaxis] ** np.arange(degree + 1)[np.newaxis, :]
    #　Yも同様に
    Y = y[:, np.newaxis] ** np.arange(1, degree + 1)[np.newaxis, :]
    # XとYの行列を水平に結合
    X = np.hstack([X,Y])
    # w = (X.T @ X + λ @ I)^-1 @ X.T @ Y
    w = np.zeros(2 * degree + 1)
    Im = np.identity(len(X[0]))
    w = np.linalg.inv(X.T @ X + normal * Im) @ X.T @ Z
    return w


def main():
    """Do the regression analyzed using the least squares method."""
    args = parse_args()
    data = np.loadtxt(args.file, comments="x1", delimiter=",", dtype="float")
    degree = args.n
    normal = args.normal

    if len(data[0]) == 2:
        # データ収納
        x = data[:, 0]
        y = data[:, 1]
        # 重みwを計算
        w = weight2d(x, y, degree, normal)
        print(w)
        plot2d(x, y, w)
    elif len(data[0]) == 3:
        # データ収納
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        # 重みwを計算
        w = weight3d(x, y, z, degree, normal)
        print(w)
        plot3d(x, y, z, w, degree)


if __name__ == "__main__":
    main()
