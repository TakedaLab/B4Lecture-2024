"""最小二乗法を用いた回帰分析."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """引数の取得を行う.

    filename : 読み込むファイル名
    degree : 多項式回帰の次数
    normalization : 正則化係数(デフォルトはなし)
    """
    parser = argparse.ArgumentParser(description="線形回帰を行う")
    parser.add_argument("--filename", type=str, required=True, help="name of file")
    parser.add_argument("--degree", type=int, required=True, help="number of degree")
    parser.add_argument(
        "--normalization", type=int, default=0, help="normalization factor"
    )
    return parser.parse_args()


def load_csv(filename):
    """csvファイルを読み込み、データを独立変数と従属変数に分割する.

    Args:
        filename : 読み込むcsvファイル名

    Returns:
        x_data : 独立変数
        y_data : 従属変数
    """
    csvdata = np.loadtxt(
        filename, delimiter=",", skiprows=1
    )  # CSVファイルの中身をNDArrayに格納
    x_size = len(csvdata[0]) - 1  # 独立変数の数
    x_data = csvdata[:, :x_size]  # 独立変数のNDArray
    y_data = csvdata[:, x_size]  # 従属変数のNDArray
    return x_data, y_data


def function(x1, x2, w, degree):
    """独立変数が２変数のときの回帰式.

    Args:
        x1 : 1つめの独立変数
        x2 : 2つめの独立変数
        w : 重み
        degree : 次数

    Returns:
        h : 回帰式
    """
    # w0+w1*x1+w2*x1^2+...+wd*x1^d
    f = np.poly1d(w[: degree + 1][::-1])
    # w(d+1)*x2+w(d+2)*x2^2+...
    g = np.poly1d(np.insert(w[degree + 1 :], 0, 0)[::-1])
    h = f(x1) + g(x2)
    return h


def make_graph(x_data, y_data, w, degree):
    """散布図と回帰のグラフを表示.

    Args:
        x_data : 独立変数
        y_data : 従属変数
        w : 重み
        degree : 次元
    """
    x_size = len(x_data[0])  # 独立変数の数

    # 独立変数が1つの時のグラフ描画
    if x_size == 1:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
        # 散布図プロット
        ax.scatter(x_data[:, 0], y_data, color="r", label="Actual Values", marker=".")
        f = np.poly1d(w[::-1])  # 回帰式
        x_axis = np.linspace(np.min(x_data[:, 0]), np.max(x_data[:, 0]), 1000)
        y = f(x_axis)
        ax.plot(x_axis, y, label="expected line")
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Linear Regression degree=" + str(degree))
        plt.savefig("result.png")
        plt.show()

    # 独立変数が2つの時の散布図描画
    if x_size == 2:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection="3d")
        # 散布図プロット
        ax.scatter(
            x_data[:, 0],
            x_data[:, 1],
            y_data,
            color="r",
            label="Actual Values",
            marker=".",
        )
        x1_axis = np.linspace(np.min(x_data[:, 0]), np.max(x_data[:, 0]), 1000)
        x2_axis = np.linspace(np.min(x_data[:, 1]), np.max(x_data[:, 1]), 1000)
        X1, X2 = np.meshgrid(x1_axis, x2_axis)
        y = function(X1, X2, w, degree)
        ax.plot_surface(X1, X2, y, label="expected surface", alpha=0.5)
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.set_title("Linear Regression degree=" + str(degree))
        plt.savefig("result.png")
        plt.show()


def calc_weight(x_data, y_data, degree, normalization):
    """重みを計算する.

    Args:
        x_data : 独立変数
        y_data : 従属変数
        degree : 次数
        normalization : 正則化係数

    Returns:
        w : 重み
    """
    w = np.zeros(len(x_data[0]) * degree + 1)
    # 多項式回帰用に新たな行列を用意
    new_x_data = np.zeros((len(x_data), len(x_data[0]) * degree + 1))
    if len(x_data[0]) == 1:
        for i in range(degree + 1):
            for j in range(len(x_data)):
                new_x_data[j][i] = x_data[j][0] ** i
                j += 1
            i += 1
    if len(x_data[0]) == 2:
        for i in range(degree + 1):
            for j in range(len(x_data)):
                new_x_data[j][i] = x_data[j][0] ** i
                j += 1
            i += 1
        for i in range(degree):
            for j in range(len(x_data)):
                new_x_data[j][i + degree + 1] = x_data[j][1] ** (i + 1)
                j += 1
            i += 1
    # 正規方程式で重みを計算(正規化するときは係数が適用される)
    w = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(new_x_data.T, new_x_data)
                + normalization * np.identity(len(new_x_data[0]))
            ),
            new_x_data.T,
        ),
        y_data,
    )
    return w


def main():
    """
    読み込んだデータで線形回帰
    """
    # 引数を受け取る
    args = parse_args()
    # 重みを計算
    w = calc_weight(
        load_csv(args.filename)[0],
        load_csv(args.filename)[1],
        args.degree,
        args.normalization,
    )
    # グラフの表示
    make_graph(load_csv(args.filename)[0], load_csv(args.filename)[1], w, args.degree)


if __name__ == "__main__":
    main()
