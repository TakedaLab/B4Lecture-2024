# -*- coding: utf-8 -*-
"""GMMを用いてデータのフィッティングを行う."""

import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def gaussian(x, mu, sigma):
    """ガウス分布を求める関数.

    Args:
        x : 入力
        mu : 平均ベクトル
        sigma : 分散共分散行列
    """
    dim = x.shape[0]
    det_sigma = LA.det(sigma)
    inv_sigma = LA.inv(sigma)

    d = np.sqrt(((2 * np.pi) ** dim) * det_sigma)
    n = np.exp((-0.5) * (x - mu).T @ inv_sigma @ (x - mu))

    gauss = n / d
    return gauss


def mix_gaussian(data_array, mu, sigma, pi):
    """混合ガウス分布を求める関数.

    Args:
        data_array : csvデータ
        mu : 平均ベクトル
        sigma : 分散共分散行列
        pi : 各ガウス分布の重み
    """
    num_gauss = pi.shape[0]
    data_size = data_array.shape[0]

    mix_gauss = np.zeros((num_gauss, data_size))
    for i in range(num_gauss):
        mix_gauss[i] = [pi[i] * gaussian(x, mu[i], sigma[i]) for x in data_array]
    return mix_gauss


def log_likelihood(data_array, mu, sigma, pi):
    """対数尤度関数を求める関数.

    Args:
        data_array : csvデータ
        mu : 平均ベクトル
        sigma : 分散共分散行列
        pi : 各ガウス分布の重み
    """
    likelihood = np.sum(mix_gaussian(data_array, mu, sigma, pi), axis=0)
    loglikelihood= np.sum(np.log(likelihood))
    return loglikelihood


def em_algorithm(data_array, mu, sigma, pi):
    """EMアルゴリズムによって各パラメータを更新する関数.

    Args:
        data_array : csvデータ
        mu : 平均ベクトル
        sigma : 分散共分散行列
        pi : 各ガウス分布の重み
    """
    data_size = data_array.shape[0]

    mix_gauss = mix_gaussian(data_array, mu, sigma, pi)
    gamma = mix_gauss / np.sum(mix_gauss, axis=0)[np.newaxis, :]

    nk = np.sum(gamma, axis=1)

    # 平均ベクトルの更新
    mu = (gamma @ data_array) / nk[:, np.newaxis]

    # 分散共分散行列の更新
    diff_x = data_array - mu[:, np.newaxis, :]
    for i in range(len(nk)):
        sigma[i] = gamma[i] * diff_x[i].T @ diff_x[i]
    sigma = sigma / nk[:, np.newaxis, np.newaxis]

    # 重みの更新
    pi = nk / data_size

    return mu, sigma, pi


if __name__ == "__main__":
    colorlist = ["#FFA500"]

    csv_data = pd.read_csv("data3.csv", header=None)

    data_array = csv_data.values.astype(float)
    data_size = data_array.shape[0]  # csvデータのデータ数
    dimension = data_array.shape[1]  # csvデータの次元数

    cluster = 3  # クラスター数の設定

    # 初期パラメータ設定
    mu = np.random.randn(cluster, dimension)
    sigma = np.array([np.identity(dimension) for i in range(cluster)])
    pi = np.ones(cluster) / cluster

    thr = 0.0001  # 収束条件

    loglikelihood = [log_likelihood(data_array, mu, sigma, pi)]

    while True:
        mu, sigma, pi = em_algorithm(data_array, mu, sigma, pi)
        # 更新したパラメータによる対数尤度関数を保存
        loglikelihood.append(log_likelihood(data_array, mu, sigma, pi))
        if np.abs(loglikelihood[-1] - loglikelihood[-2]) < thr:
            break

    N = 200  # 描写の際の細かさ

    # 対数尤度関数の推移を描写
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(
        np.arange(0, len(loglikelihood), 1),
        loglikelihood,
        label="log_likelihood"
    )
    plt.legend(loc="upper left", fontsize=9)
    ax1.set_xlabel("repetition")
    ax1.set_ylabel("log-likelihood")
    fig1.savefig("data3_loglikelihood.png")

    # 2D描写
    if dimension == 1:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.scatter(
            data_array, np.zeros(data_size), color='blue', label="data"
        )
        ax2.scatter(
            mu, np.zeros(mu.shape[0]), color='red', label="centroid"
        )
        x_min = np.min(data_array[:, 0] - 0.5)
        x_max = np.max(data_array[:, 0] + 0.5)
        p = np.linspace(x_min, x_max, N)[:, np.newaxis]
        mix_gauss = np.sum(mix_gaussian(p, mu, sigma, pi), axis=0)
        ax2.plot(p, mix_gauss, label="gaussian")
        plt.legend(loc="upper left", fontsize=9)
        ax2.set_xlabel("x")
        ax2.set_ylabel("probability density")
        ax2.set_ylim(0,)
        fig2.savefig("data1_gaussian.png")

    # 3D描写
    if dimension == 2:
        x_min = int(np.min(data_array[:, 0]) - 1.5)
        x_max = int(np.max(data_array[:, 0]) + 1.5)
        y_min = int(np.min(data_array[:, 1]) - 1.5)
        y_max = int(np.max(data_array[:, 1]) + 1.5)
        x = np.linspace(x_min, x_max, N)
        y = np.linspace(y_min, y_max, N)
        X, Y = np.meshgrid(x, y)
        xy = np.c_[X.ravel(), Y.ravel()]
        pdf = np.sum(mix_gaussian(xy, mu, sigma, pi), axis=0)
        pdf = pdf.reshape(X.shape)

        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        fig2.add_axes(ax2)
        ax2.scatter(
            data_array[:, 0], data_array[:, 1], np.zeros(data_size), color=colorlist, label="data"
        )
        ax2.plot_surface(X, Y, pdf, rstride=1, cstride=1, cmap='viridis')
        plt.legend(loc="upper left", fontsize=9)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("probability density")
        ax2.set_zlim(0,)
        fig2.savefig("data3_gaussian.png")
        ax2.view_init(azim=0, elev=0)
        fig2.savefig("data3_gaussian_0_0.png")
        ax2.view_init(azim=90, elev=0)
        fig2.savefig("data3_gaussian_90_0.png")
        ax2.view_init(azim=0, elev=90)
        fig2.savefig("data3_gaussian_0_90.png")

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.scatter(
            data_array[:, 0], data_array[:, 1], label="data", color='blue'
        )
        ax3.scatter(
            mu[:, 0], mu[:, 1], label="Centroid", color="red"
        )
        cset = ax3.contour(X, Y, pdf, cmap=cm.jet)
        plt.legend(loc="upper left", fontsize=9)
        fig3.savefig("data3_contour.png")
