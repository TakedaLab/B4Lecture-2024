"""EMアルゴリズムによるGMMの最尤推定."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """コマンドプロンプトからデータとクラスター数を取得."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("num_components")
    args = parser.parse_args()

    data = np.loadtxt(args.filename, delimiter=",")
    num_components = int(args.num_components)

    return data, num_components


def plot_scatter(data, title):
    """散布図をプロット."""
    # 1次元の場合
    if data.ndim == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data, np.zeros(len(data)), marker=".")
        ax.set_xlabel("X")
        ax.set_title(title)

    # 2次元の場合
    elif data.ndim == 2 and data.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[:, 0], data[:, 1], marker=".")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)

    # 3次元の場合
    elif data.ndim == 2 and data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

    # データが3次元以外の場合
    else:
        print("Data dimension not supported")

    # プロットの表示
    plt.show()


def initialize(data, num_components):
    """初期値をランダムに設定.

    Args:
        data (np.ndarray): 入力データ
        num_components (int): クラスター数

    Returns:
        mean(np.ndarray): 平均ベクトル
        cov(np.ndarray): 分散共分散行列
        weight(np.ndarray): 各クラスターの重み
    """
    # 平均ベクトル
    # クラスター数×次元数の配列を用意
    mean = np.zeros((num_components, data.shape[1]))
    # それぞれの軸からランダムな要素を選択し、平均とする
    for k in range(num_components):
        for i in range(data.shape[1]):
            mean[k, i] = np.random.choice(data[i, :])

    # 分散共分散行列
    # 次元数×次元数の単位行列をクラスター数分用意
    eye_dim = np.eye(data.shape[1])
    cov_list = [eye_dim + 1e-6 for k in range(num_components)]
    cov = np.array(cov_list)

    # クラスターの重み
    # クラスター数分の乱数を生成し、重みとする
    weight = np.random.random(num_components)
    weight /= np.sum(weight)

    return mean, cov, weight


def calc_pdf(x, mean, cov):
    """多変量正規分布の確率密度関数を計算.

    Args:
        x (np.ndarray): 入力データ
        mean (np.ndarray): 平均ベクトル
        cov (np.ndarray): 分散共分散行列

    Returns:
        np.ndarray: 確率密度関数
    """
    A = float(1 / ((2 * np.pi) ** (x.size / 2)))
    B = float(1 / np.linalg.det(cov) ** 0.5)
    C = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        C[i] = -0.5 * (x[i] - mean).T @ np.linalg.inv(cov) @ (x[i] - mean)

    pdf = A * B * np.exp(C)
    return pdf


def E_step(data, num_components, mean, cov, weight):
    """平均、分散共分散行列、重みを用いて、負担率を計算.

    Args:
        data (np.ndarray): データ
        num_components (int): クラスター数
        mean (np.ndarray): 平均ベクトル
        cov (np.ndarray): 分散共分散行列
        weight (np.ndarray): クラスターの重み

    Returns:
        np.ndarray: 負担率
    """
    responsibility = np.zeros((data.shape[0], num_components))
    for k in range(num_components):
        responsibility[:, k] = weight[k] * calc_pdf(data, mean[k], cov[k])
        # print("responsibility_pre = {}".format(responsibility), end="\n\n")
    responsibility /= np.sum(responsibility, axis=1, keepdims=True)
    # print("responsibility = {}".format(responsibility), end="\n\n")
    return responsibility


def M_step(data, num_components, responsibility):
    """負担率を用いて、平均、分散共分散行列、重みを更新.

    Args:
        data (np.ndarray): データ
        num_components (int): クラスター数
        responsibility (np.ndarray): 負担率

    Returns:
        mean (np.ndarray): 平均ベクトル
        cov (np.ndarray): 分散共分散行列
        weight (np.ndarray): クラスターの重みS
    """
    Nk = np.sum(responsibility, axis=0)
    mean = np.zeros((num_components, data.shape[1]))
    for k in range(num_components):
        mean[k, :] = np.sum(responsibility[:, k, np.newaxis] * data, axis=0) / Nk[k]

    cov = np.zeros((num_components, data.shape[1], data.shape[1]))
    for k in range(num_components):
        cov[k] = (
            (responsibility[:, k, np.newaxis] * (data - mean[k])).T
            @ (data - mean[k])
            / Nk[k]
        )

    weight = Nk / data.size

    return mean, cov, weight


def EM_algorithm(data, num_components, mean, cov, weight):
    """EMアルゴリズムの実行.

    Args:
        data (np.ndarray): データ
        num_components (int): クラスター数
        mean (np.ndarray): 平均ベクトル
        cov (np.ndarray): 分散共分散行列
        weight (np.ndarray): クラスターの重み

    Returns:
        mean (np.ndarray): 最適化された平均ベクトル
        cov (np.ndarray): 最適化された分散共分散行列
        weight (np.ndarray): 最適化されたクラスターの重み
        log_likelihood (list): 対数尤度
    """
    max_iter = 100
    log_likelihood = []

    for i in range(max_iter):
        responsibility = E_step(data, num_components, mean, cov, weight)
        mean, cov, weight = M_step(data, num_components, responsibility)
        print(
            "mean = {}\n, cov = {}\n, weight = {}\n".format(mean, cov, weight),
            end="\n\n",
        )
        print("responsibility = {}".format(responsibility), end="\n\n")

        pdf = np.zeros((len(data), num_components))
        for k in range(num_components):
            pdf[:, k] = weight[k] * calc_pdf(data, mean[k], cov[k])
        log_likelihood.append(np.sum(np.log(np.sum(pdf, axis=0))))
        print("log_likelihood = {}".format(log_likelihood), end="\n\n")

        if i > 1 and np.abs(log_likelihood[-1] - log_likelihood[-2]) < 1e-6:
            print("end", i)
            break

    return mean, cov, weight, log_likelihood


def plot_log_likelihood(log_likelihood):
    """対数尤度をプロット."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(log_likelihood)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log likelihood")
    ax.set_title("Log likelihood vs Iteration")
    plt.show()


def main():
    """EMアルゴリズムを実行し、結果をプロット."""
    data, num_components = parse_args()
    # plot_scatter(data, "data")

    mean, cov, weight = initialize(data, num_components)
    mean, cov, weight, log_likelihood = EM_algorithm(
        data, num_components, mean, cov, weight
    )

    plot_log_likelihood(log_likelihood)


if __name__ == "__main__":
    main()
