# -*- coding: utf-8 -*-
"""最小二乗法を用いて回帰分析を行う."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def least_squire_method(data_size, dimension, csv_data, polynomial, switch, alpha):
    """最小二乗法を行う関数.

    Args:
        dimension : データの次数
        csv_data : csvのデータ
        data_size : csvデータのサイズ
        polynomial : 変数1つに対して何項にするか
    """
    list_value_x = np.zeros((data_size, dimension - 1))  # x1,x2...の値を格納
    list_value_y = np.zeros(data_size)  # yの値を格納

    # csvのデータを"x1,x2,..."と"y"で分ける
    for i in range(data_size):
        list_value_y[i] = csv_data[i, dimension - 1]
        for j in range(dimension - 1):
            list_value_x[i, j] += csv_data[i, j]

    # 次数、項数によって説明変数の次数が変化
    num_of_variable = (dimension - 1) * polynomial
    value_to_x = np.zeros((data_size, num_of_variable))

    # x^2, x^3, ... を保存
    for i in range(list_value_x.shape[0]):
        for j in range(dimension - 1):
            for k in range(polynomial):
                value_to_x[i, polynomial * j + k] += list_value_x[i, j] ** (k + 1)

    # 平均
    mean_x = np.mean(value_to_x, axis=0)
    mean_y = list_value_y.mean()

    X = value_to_x.T @ value_to_x

    if switch == 0:
        # リッジ回帰を適用しない
        weight = np.linalg.inv(X) @ value_to_x.T @ list_value_y
    else:
        # リッジ回帰を適用
        i = np.eye(len(X))
        weight = np.linalg.inv(X + alpha * i) @ value_to_x.T @ list_value_y

    w0 = mean_y
    for i in range(len(weight)):
        w0 -= weight[i] * mean_x[i]
    weight = np.insert(weight, 0, w0)

    return weight


def func(x, weight):
    """2次元グラフのyの値を導出.

    Args:
        x : そのまま
        dimension : データの次数
    """
    y = 0
    for i in range(len(weight)):
        y += (x**i) * weight[i]
    return y


def func_3d(x, y, weight):
    """3次元グラフのzの値を導出.

    Args:
        x : そのまま
        y : そのまま
        dimension : データの次数
    """
    z = 0
    poly = (len(weight) - 1) // 2
    for i in range(len(weight)):
        if i <= poly:
            z += (x**i) * weight[i]
        else:
            z += (y ** (i - poly)) * weight[i]
    return z


def label_make(weight):
    """2次元グラフの凡例を作るための関数.

    Args:
        weight : 導出した重み
    """
    weight = np.round(weight, decimals=2)
    text = str(weight[0])
    for i in range(len(weight) - 1):
        if weight[i + 1] > 0:
            text += "+" + str(weight[i + 1]) + "x^" + str(i + 1)
        else:
            text += str(weight[i + 1]) + "x^" + str(i + 1)
    return text


def label_make_3d(weight):
    """3次元グラフの凡例を作るための関数.

    Args:
        weight : 導出した重み
    """
    weight = np.round(weight, decimals=2)
    poly = (len(weight) - 1) // 2

    text = str(weight[0])
    for i in range(len(weight) - 1):
        if i < poly:
            if weight[i + 1] > 0:
                text += "+" + str(weight[i + 1]) + "x^" + str(i + 1)
            else:
                text += str(weight[i + 1]) + "x^" + str(i + 1)
        else:
            if weight[i + 1] > 0:
                text += "+" + str(weight[i + 1]) + "y^" + str(i - poly + 1)
            else:
                text += str(weight[i + 1]) + "y^" + str(i - poly + 1)
    return text


if __name__ == "__main__":
    colorlist = ["#FFA500"]

    csv_data = pd.read_csv("data2.csv")

    data_array = csv_data.values.astype(float)
    data_size = data_array.shape[0]  # csvデータのデータ数
    dimension = data_array.shape[1]  # csvデータの次元数
    polynomial = 15  # 説明変数の次数
    switch = 1  # 正則化するかどうか
    alpha = 10

    # 重みの計算
    weight = least_squire_method(
        data_size, dimension, data_array, polynomial, switch, alpha
    )
    print(weight)

    # 正則化の比較のときにコメントアウトを外す
    # weight1 = least_squire_method(data_size, dimension, data_array, polynomial, 0, alpha)
    # weight2 = least_squire_method(data_size, dimension, data_array, polynomial, 1, alpha)

    N = 200  # 描写の際の細かさ
    fig1 = plt.figure()

    # 2D描写
    if dimension == 2:
        ax1 = fig1.add_subplot(111)
        ax1.scatter(csv_data["x1"], csv_data["x2"], color=colorlist)

        min_x = int(csv_data.min()["x1"] - 0.5)
        max_x = int(csv_data.max()["x1"] + 0.5)
        p = np.linspace(min_x, max_x, N)
        ax1.plot(p, [func(p[k], weight) for k in range(N)], label=label_make(weight))
        ax1.legend(loc="upper center", fontsize="xx-small")  # フォントサイズ
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

    # 3D描写
    if dimension == 3:
        fig1 = plt.figure()
        ax1 = Axes3D(fig1)
        fig1.add_axes(ax1)
        ax1.scatter(csv_data["x1"], csv_data["x2"], csv_data["x3"], color=colorlist)

        min_x = int(csv_data.min()["x1"] - 0.5)
        max_x = int(csv_data.max()["x1"] + 0.5)
        min_y = int(csv_data.min()["x2"] - 0.5)
        max_y = int(csv_data.max()["x2"] + 0.5)
        p = np.linspace(min_x, max_x, N)
        q = np.linspace(min_y, max_y, N)
        X, Y = np.meshgrid(p, q)
        Z = func_3d(X, Y, weight)
        ax1.plot_wireframe(X, Y, Z, label=label_make_3d(weight))
        ax1.legend(loc="upper center", fontsize="xx-small")  # フォントサイズ
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")

    # fig1.savefig("overfitting.png")
