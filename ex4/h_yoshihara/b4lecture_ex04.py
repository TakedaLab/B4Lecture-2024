# -*- coding: utf-8 -*-
"""最小二乗法を用いて回帰分析を行う."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def eigenvector(data_array, data_size, dim):
    """固有値、固有ベクトルを求める関数.

    Args:
        data_array : csvのデータ
        data_size : csvデータのサイズ
        dim : csvデータの次元数
    """
    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)

    stanardization = np.zeros((data_size, dim))
    for i in range(data_size):
        for j in range(dim):
            stanardization[i, j] = (data_array[i, j] - mean_data[j]) / std_data[j]

    covariance = (stanardization.T @ stanardization) / data_size

    vector = LA.eig(covariance)

    sort_vector = vector[1][:, np.argsort(-vector[0])]
    sort_value = sorted(vector[0], reverse=True)

    return sort_value, sort_vector


def contribute_rate(eigen):
    """寄与率の計算を行う関数.

    Args:
        eigen : 固有値のリスト
    """
    rate = eigen / np.sum(eigen)

    # 寄与率の確認
    # print(np.sum(vector))
    # print(rate)

    return rate


if __name__ == "__main__":
    colorlist = ["#FFA500"]

    csv_data = pd.read_csv("data3.csv", header=None)

    data_array = csv_data.values.astype(float)
    data_size = data_array.shape[0]  # csvデータのデータ数
    dimension = data_array.shape[1]  # csvデータの次元数

    # 固有値、固有ベクトル
    eigen, vector = eigenvector(data_array, data_size, dimension)

    # 固有値確認用
    # print(eigen)
    # print(vector)

    rate = contribute_rate(eigen)  # 寄与率を求める

    N = 200  # 描写の際の細かさ
    fig1 = plt.figure()

    # 2D描写
    if dimension == 2:
        ax1 = fig1.add_subplot(111)
        ax1.scatter(data_array[:, 0], data_array[:, 1], color=colorlist)

        x = np.linspace(np.min(data_array[:, 0]), np.max(data_array[:, 1]), 100)
        x_y = vector[1] / vector[0]
        ax1.plot(x, x_y[0] * x, color="blue", label=str(rate[0]))
        ax1.plot(x, x_y[1] * x, color="red", label=str(rate[1]))
        plt.legend(loc="upper left", fontsize=9)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")

        fig1.savefig("data1_vector.png")

    # 3D描写
    if dimension == 3:
        ax1 = Axes3D(fig1)
        fig1.add_axes(ax1)
        ax1.scatter(
            data_array[:, 0], data_array[:, 1], data_array[:, 2], color=colorlist
        )

        x = np.linspace(np.min(data_array[:, 0]), np.max(data_array[:, 1]), 100)
        x_y = vector[1] / vector[0]
        x_z = vector[2] / vector[0]

        ax1.plot(x, x_y[0] * x, x_z[0] * x, color="blue", label=str(rate[0]))
        ax1.plot(x, x_y[1] * x, x_z[1] * x, color="red", label=str(rate[1]))
        ax1.plot(x, x_y[2] * x, x_z[2] * x, color="green", label=str(rate[2]))
        plt.legend(loc="upper left", fontsize=9)
        ax1.view_init(azim=0, elev=90)  # 仰角、俯角の変更

        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("x3")

        # 2次元に次元圧縮
        data_2d = data_array @ vector
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.scatter(data_2d[:, 0], data_2d[:, 1], color=colorlist, label="data")

        plt.legend(loc="upper left", fontsize=9)
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")

        fig1.savefig("data2_vector_0_90.png")
        fig2.savefig("data2_2d.png")

    # 4次元以上のデータに対して、どこまで次元圧縮できるか
    else:
        print(rate)

        rate_sum = 0.0  # 累積寄与率
        count = 0  # 採用する主成分数
        for i in rate:
            rate_sum += i
            count += 1
            if rate_sum >= 0.9:
                print(rate_sum)  # 90％を超えた時の累積寄与率
                print(count)
                break

    # fig1.savefig("data3_vector.png")
