"""
最小二乗法を用いて回帰分析を行う．
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    """
    引数を取得する

    Returns
    -------
    parser.parse_args() : 引数を返す
    """
    parser = argparse.ArgumentParser(
        description="最小二乗法を用いて回帰分析を行う。"
    )
    parser.add_argument(
        "-file",
        help="ファイルを入力",
        default=r"C:\Users\kyskn\B4Lecture-2024\ex3\k_namizaki\data3.csv",
        type = str
    )
    parser.add_argument("-n", help="次数", default=1, type=int)
    parser.add_argument("-normal", help="正則化係数", default=0, type=int)
    return parser.parse_args()

def plot2d(x,y,w):
    """
    2次元のプロットを行う

    Parameters
    ----------
    x : xのデータ
    y : yのデータ
    w : 重み
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
    ax.scatter(x, y, label = "Observed data")
    ax.plot(x_ans, y_ans, label="ans")
    ax.legend()
    plt.show()

def plot3d(x, y, z, w, degree):
    """
    3次元のプロットを行う

    Parameters
    ----------
    x : xのデータ
    y : yのデータ
    z : zのデータ
    w : 重み
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
    ax.plot_surface(X, Y, z_ans, label="ans",)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def weight2d(x, y, degree, normal):
    """
    2つのデータから重みを計算

    Parameters
    ----------
    x : xのデータ
    y : yのデータ
    degree : 次数
    normal : 正則化係数

    Returns
    -------
    w : 重み
    """
    # X = 1+x+x^2+...
    X = np.zeros((len(x), degree + 1))
    Y = y
    for i in range(degree + 1):
        for j in range(len(x)):
            X[j][i] = x[j] ** i
            j += 1
        i += 1
    # w = (X.T @ X + λ @ I)^-1 @ X.T @ Y
    w = np.zeros(degree + 1)
    I = np.identity(len(X[0]))
    w = np.linalg.inv(X.T @ X + normal * I) @ X.T @ Y
    return w

def weight3d(x, y, z, degree, normal):
    """
    3つのデータから重みを計算

    Parameters
    ----------
    x : xのデータ
    y : yのデータ
    z : zのデータ
    degree : 次数
    normal : 正則化係数

    Returns
    -------
    w : 重み
    """
    # X = 1+x+x^2+...+y+y^2+...
    X = np.zeros((len(x), 2*degree + 1))
    Z = z
    # 0~次数まで回す
    for i in range(degree + 1):
        for j in range(len(x)):
            X[j][i] = x[j] ** i
            j += 1
        i += 1
    # 1~次数まで回す
    for i in range(degree):
        for j in range(len(y)):
            X[j][i + degree + 1] = y[j] ** (i + 1)
            j += 1
        i += 1
    # w = (X.T @ X + λ @ I)^-1 @ X.T @ Y
    w = np.zeros(2*degree + 1)
    I = np.identity(len(X[0]))
    w = np.linalg.inv(X.T @ X + normal * I) @ X.T @ Z
    return w

def main():
    """
    最小二乗法を用いて回帰分析を行う．
    """
    args = parse_args()
    data = np.loadtxt(
        args.file,
        comments="x1",
        delimiter=",",
        dtype="float"
    )
    degree = args.n
    normal = args.normal

    if len(data[0]) == 2:
        #データ収納
        x = data[:, 0]
        y = data[:, 1]
        # 重みwを計算
        w = weight2d(x, y, degree, normal)
        print(w)
        plot2d(x,y,w)
    elif len(data[0]) == 3:
        #データ収納
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        # 重みwを計算
        w = weight3d(x, y, z, degree, normal)
        print(w)
        plot3d(x, y, z, w, degree)


if __name__ == "__main__":
    main()
