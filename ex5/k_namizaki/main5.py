"""This code is Fitting of data using GMM."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from numpy.typing import ArrayLike
import scipy.stats as stats



def parse_args() -> Any:
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
        default=r"C:\Users\kyskn\B4Lecture-2024\ex5\data2.csv",
        type=str,
    )
    parser.add_argument("-k", help="クラスター数", default=3, type=int)
    return parser.parse_args()


def plot_data(data: ArrayLike, D: int):
    """
    Plot data.

    Parameters
    ----------
    data : ArrayLike (N,D)
        読み込んだcsvデータ
    D : int
        データの次元
    """
    fig, ax = plt.subplots()
    if D == 1:
        zeros = np.zeros(len(data))
        ax.scatter(data, zeros, marker=".")
        ax.set_title("sample data")
        ax.set_xlabel("x")
        plt.show()
    elif D == 2:
        ax.scatter(data[:, 0], data[:, 1])
        ax.set_title("sample data")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()


def visualize(
    data: ArrayLike, pi: ArrayLike, mu: ArrayLike, sigma: ArrayLike, K: int, D: int
):
    """
    Visualize contour (line).

    Parameters
    ----------
    data : ArrayLike (N,D)
        読み込んだcsvデータ
    pi : ArrayLike (K,)
        混合率
    mu : ArrayLike (K,D)
        平均ベクトル
    sigma : ArrayLike (K,D,D)
        分散共分散行列
    K : int
        クラスター数
    D : int
        データの次元数
    """
    fig, ax = plt.subplots()
    # １次元の場合
    if D == 1:
        min_value = min(data)
        max_value = max(data)
        x = np.arange(min_value - 1, max_value + 1, 0.01)
        Z = cul_gmm(x, pi, mu, sigma, K)
        Z = np.sum(Z, axis=1)
        ax.plot(x, Z, color="r")
        zeros = np.zeros(len(data))
        ax.scatter(data, zeros, marker=".")
        for k in range(K):
            ax.scatter(mu[k], 0, c="red", marker="x")
        ax.set_title("contour (line) K = " + str(K))
        ax.set_xlabel("x1")
    elif D == 2:
        # データ生成
        min_value = min(data[:, 0])
        max_value = max(data[:, 0])
        x = np.arange(min_value - 0.5, max_value + 0.5, 0.01)
        min_value = min(data[:, 1])
        max_value = max(data[:, 1])
        y = np.arange(min_value - 0.5, max_value + 0.5, 0.01)
        X, Y = np.meshgrid(x, y)

        z = np.c_[X.ravel(), Y.ravel()]
        Z = cul_gmm(z, pi, mu, sigma, K)
        # 分割で保持されているから一列にまとめないといけない
        Z = np.sum(Z, axis=1)
        shape = X.shape
        Z = Z.reshape(shape)
        # 等高線グラフ描画
        ax.contour(X, Y, Z)
        ax.scatter(data[:, 0], data[:, 1])
        for k in range(K):
            ax.scatter(mu[k, 0], mu[k, 1], c="red", marker="x")
        ax.set_title("contour (line) K = " + str(K))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    plt.show()


def initialization(K: int, D: int):
    """
    initialize.

    Parameters
    ----------
    K : int
        クラスター数
    D : int
        データの次元数

    Returns
    -------
    pi : ArrayLike (K,)
        混合率
    mu : ArrayLike (K,D)
        平均ベクトル
    sigma : ArrayLike (K,D,D)
        分散共分散行列
    """
    # 平均は標準ガウス分布から生成
    mu = np.random.randn(K, D)
    # 分散共分散行列はD*D単位行列をK個
    sigma = np.tile(np.eye(D), (K, 1, 1))
    # 重みは一様分布から生成
    pi = np.ones(K) / K
    return pi, mu, sigma


def cul_gmm(data: ArrayLike, pi: ArrayLike, mu: ArrayLike, sigma: ArrayLike, K: int):
    """
    Calculate gmm.

    Parameters
    ----------
    data : ArrayLike (N,D)
        読み込んだcsvデータ
    pi : ArrayLike (K,)
        混合率
    mu : ArrayLike (K,D)
        平均ベクトル
    sigma : ArrayLike (K,D,D)
        分散共分散行列
    K : int
        クラスター数

    Returns
    -------
    gmm : ArrayLike
        gmm
    """
    gmm = np.array(
        [
            pi[k] * stats.multivariate_normal.pdf(data, mean=mu[k], cov=sigma[k])
            for k in range(K)
        ]
    ).T
    return gmm


def e_step(data: ArrayLike, pi: ArrayLike, mu: ArrayLike, sigma: ArrayLike, K: int):
    """
    Do e_step.

    Parameters
    ----------
    data : ArrayLike (N,D)
        読み込んだcsvデータ
    pi : ArrayLike (K,)
        混合率
    mu : ArrayLike (K,D)
        平均ベクトル
    sigma : ArrayLike (K,D,D)
        分散共分散行列
    K : int
        クラスター数

    Returns
    -------
    r : ArrayLike (N,K)
        負担率
    """
    # GMMの確率密度関数を計算
    gmm = cul_gmm(data, pi, mu, sigma, K)
    # 対数領域で負担率を計算
    eps = np.finfo(float).eps
    # axis=1 は行方向に合計を計算し、keepdims=True は次元を保持することを意味します。これにより、各行の合計が列ベクトルとして保持されます
    log_r = np.log(gmm) - np.log(np.sum(gmm, 1, keepdims=True) + eps)
    # 対数領域から元に戻す
    r = np.exp(log_r)
    # np.expでオーバーフローを起こしている可能性があるためnanを置換しておく
    r[np.isnan(r)] = 1.0 / (K)
    # 更新
    return r


def m_step(data: ArrayLike, r: ArrayLike, K: int, N: int, D: int):
    """
    Do m_step.

    Parameters
    ----------
    data : ArrayLike (N,D)
        読み込んだcsvデータ
    r : ArrayLike (N,K)
        負担率
    K : int
        クラスター数
    N : int
        データの数
    D : int
        データの次元数

    Returns
    -------
    pi : ArrayLike (K,)
        混合率
    mu : ArrayLike (K,D)
        平均ベクトル
    sigma : ArrayLike (K,D,D)
        分散共分散行列
    """
    # まずは N_k を計算しておく
    N_k = np.sum(r, 0)  # (K)
    # 最適なpiを計算して更新する
    pi = N_k / N  # (K)
    if D == 1:
        # 最適なmuを計算して更新する
        mu = (r.T @ data) / (N_k)  # (K, )
        # 最適なsigmaを計算して更新する
        res_error = np.tile(data[:, None], (1, K)).T - np.tile(
            mu[:, None], (1, N)
        )  # (K, N)
        sigma = ((r.T * res_error) @ res_error.T) / (N_k[:, None])  # (K, K)
        sigma = sigma[:, 0]  # 最初の列を取り出す (K, )
    elif D == 2:
        # 最適なmuを計算して更新する
        mu = (r.T @ data) / (N_k[:, None] + np.spacing(1))  # (K, D)
        # 最適なsigmaを計算して更新する
        r_tile = np.tile(r[:, :, None], (1, 1, D)).transpose(1, 2, 0)  # (K, D, N)
        res_error = np.tile(data[:, :, None], (1, 1, K)).transpose(2, 1, 0) - np.tile(
            mu[:, :, None], (1, 1, N)
        )  # (K, D, N)
        sigma = ((r_tile * res_error) @ res_error.transpose(0, 2, 1)) / (
            N_k[:, None, None] + np.spacing(1)
        )  # (K, D, D)
    return pi, mu, sigma


def main() -> None:
    """Fitting of data using GMM."""
    args = parse_args()
    data = np.loadtxt(args.file, delimiter=",", dtype="float")
    K = args.k
    rota_max = 1000
    thr = 0.0001
    # 次元数Nはshapeでは、一列の時に困る
    # 入力データXのサイズは(N, D)
    D = data.shape[1] if len(data.shape) > 1 else 1
    N = len(data)
    # 散布図をプロットして概形を確認
    plot_data(data, D)

    eps = np.finfo(float).eps
    # 初期化
    pi, mu, sigma = initialization(K, D)
    # 各イテレーションの対数尤度を記録するためのリスト
    log_likelihood_list = []
    # 対数尤度の初期値を計算
    log_likelihood_list.append(
        np.mean(np.log(np.sum(cul_gmm(data, pi, mu, sigma, K), 1) + eps))
    )
    for i in range(rota_max):
        # Eステップの実行
        r = e_step(data, pi, mu, sigma, K)
        # Mステップの実行
        pi, mu, sigma = m_step(data, r, K, N, D)
        # 今回のイテレーションの対数尤度を記録する
        gmm = cul_gmm(data, pi, mu, sigma, K)
        log_likelihood_list.append(np.mean(np.log(np.sum(gmm, 1) + eps)))
        # 前回の対数尤度からの増加幅を出力する
        print(
            "Log-likelihood gap: "
            + str(round(np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]), 4))
        )
        # もし収束条件を満たした場合，もしくは最大更新回数に到達した場合は更新停止して可視化を行う
        if (np.abs(log_likelihood_list[i] - log_likelihood_list[i + 1]) < thr) or (
            i == rota_max - 1
        ):
            print(f"EM algorithm has stopped after {i + 1} iteraions.")
            visualize(data, pi, mu, sigma, K, D)
            fig, ax = plt.subplots()
            ax.plot(log_likelihood_list)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Log likelihood(mean)")
            plt.show()
            break


if __name__ == "__main__":
    main()
