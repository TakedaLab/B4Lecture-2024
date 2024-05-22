"""GMMを用いたデータフィッティングを行う."""

import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

import ex4


def parse_args():
    """引数の取得を行う.

    filename : 読み込むファイル名
    n_cluster : クラスター数
    """
    parser = argparse.ArgumentParser(description="GMMを用いたマッピングを行う")
    parser.add_argument("--filename", type=str, required=True, help="name of file")
    parser.add_argument(
        "--n_components", type=int, required=True, help="number of component"
    )
    return parser.parse_args()


def random_initialize_para(data: np.ndarray, n_components: int) -> np.ndarray:
    """ランダムでパラメータを初期化する.

    Args:
        data (np.ndarray): データ
        n_components (int): 混合数

    Returns:
        weights: 混合係数
        means: 平均
        covariances: 共分散行列
    """
    n_features = data.shape[1]

    # 混合係数をランダムに初期化 (正規化する)
    weights = np.random.rand(n_components)
    weights /= weights.sum()

    # 平均値をランダムに初期化
    # means = np.random.rand(n_components, n_features) * (
    #     data.max(axis=0) - data.min(axis=0)
    # ) + data.min(axis=0)

    # KMeansで初期化
    kmeans = KMeans(n_clusters=n_components, random_state=0).fit(data)
    means = kmeans.cluster_centers_

    # 共分散行列のランダム初期化（数値安定性のため小さな値を加える）
    # covariances = np.zeros((n_components, n_features, n_features))
    # for i in range(n_components):
    #     covariances[i] = np.eye(n_features) * (np.random.rand() + 1e-6)
    covariances = np.eye(n_features) * (np.random.rand(n_components, 1, 1) + 1e-6)
    return weights, means, covariances


def estep(
    data: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """Eステップの実行.

    Args:
        data (np.ndarray): データ
        weights (np.ndarray): 混合係数
        means (np.ndarray): 平均値
        covariances (np.ndarray): 共分散行列

    Returns:
        responsibilities (np.ndarray): 負担率
        responsibilities_sum (np.ndarray): 負担率の合計
    """
    n_samples = data.shape[0]
    n_components = weights.shape[0]

    # 負担率を計算する
    responsibilities = np.zeros((n_samples, n_components))
    for k in range(n_components):
        responsibilities[:, k] = weights[k] * multivariate_normal.pdf(
            data, mean=means[k], cov=covariances[k]
        )
    responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= responsibilities_sum

    return responsibilities, responsibilities_sum


def mstep(data: np.ndarray, responsibilities: np.ndarray) -> np.ndarray:
    """Mステップの実行.

    Args:
        data (np.ndarray): データ
        responsibilities (np.ndarray): 負担率

    Returns:
        weights (np.ndarray): 再計算後の混合係数
        means (np.ndarray): 再計算後の平均
        covariances (np.ndarray): 再計算後の共分散行列
    """
    n_samples, n_features = data.shape
    n_components = responsibilities.shape[1]

    N_k = responsibilities.sum(axis=0)

    means = np.zeros((n_components, n_features))
    covariances = np.zeros((n_components, n_features, n_features))
    weights = np.zeros(n_components)

    # 平均の再計算
    for k in range(n_components):
        means[k, :] = np.sum(responsibilities[:, k, np.newaxis] * data, axis=0) / N_k[k]

    # 共分散行列の再計算
    for k in range(n_components):
        # データポイントと平均の差分を計算
        diff = data - means[k, :]

        # データポイントと平均の差分を重み付けし、行列の積を計算
        weighted_diff = np.dot((responsibilities[:, k, np.newaxis] * diff).T, diff)

        # 共分散行列を計算し、N_k で割る
        covariances[k] = weighted_diff / N_k[k]

    # 混合係数の再計算
    weights = N_k / np.sum(N_k, axis=0)

    return weights, means, covariances


def EMstep(
    data: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray
):
    """EMアルゴリズムの実行.

    Args:
        data (np.ndarray): データ
        weights (np.ndarray): 混合係数
        means (np.ndarray): 平均
        covariances (np.ndarray): 共分散行列

    Returns:
        log_likelihood (list): 対数尤度のリスト
        weights (np.ndarray): 再計算後の混合係数
        means (np.ndarray): 再計算後の平均
        covariances (np.ndarray): 再計算後の共分散行列
    """
    # n_samples = data.shape[0]

    # EMアルゴリズムの試行回数
    max_try = 100

    # 対数尤度を格納する
    log_likelihood_list = []

    for j in range(max_try):
        # Eステップを実行する
        responsibilities, responsibilities_sum = estep(
            data, weights, means, covariances
        )

        # Mステップを実行する
        weights, means, covariances = mstep(data, responsibilities)

        # log_likelihood = 0
        # for i in range(n_samples):
        #     log_likelihood += np.log(responsibilities_sum[i])

        # log_likelihood_list.append(log_likelihood[0])

        log_likelihood = np.sum(np.log(responsibilities_sum))
        log_likelihood_list.append(log_likelihood)

        # 収束判定
        if j > 0 and np.abs(log_likelihood_list[-1] - log_likelihood_list[-2]) < 1e-4:
            break

    return log_likelihood_list, weights, means, covariances


def make_likelihoodgraph(log_likelihood_list: list):
    """再計算回数と対数尤度の関係をプロットする.

    Args:
        log_likelihood_list (list): 対数尤度
    """
    # 対数尤度のプロット
    plt.plot(np.arange(len(log_likelihood_list)), log_likelihood_list)
    plt.xlabel("Iterations")
    plt.ylabel("Log Likelihood")
    plt.title("Log Likelihood by Iterations")

    # 横軸を整数にする
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig("likelihood.png")
    plt.show()


def plot_gmm(
    data: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray
):
    """混合ガウス分布をプロットする.

    Args:
        data (np.ndarray): データ
        weights (np.ndarray): 混合係数
        means (np.ndarray): 平均
        covariances (np.ndarray): 共分散行列
    """
    n_components = weights.shape[0]

    # 1次元データの場合
    if data.shape[1] == 1:
        x = np.linspace(data.min(), data.max(), 1000)
        y = np.zeros_like(x)

        # 混合ガウス分布の確率密度関数を計算
        for k in range(n_components):
            y += weights[k] * multivariate_normal.pdf(
                x, mean=means[k], cov=covariances[k]
            )
        zeros = np.zeros(len(data))

        # 確率密度関数をプロット
        plt.plot(x, y, color="b", label="GMM")

        # 散布図をプロット
        plt.scatter(data, zeros, color="r", label="Data", marker=".")

        # ガウス分布の平均を表示
        plt.scatter(
            means,
            np.zeros(n_components),
            color="g",
            marker="x",
            label="Mean of Gaussian Distribution",
        )

        # 軸ラベルを表示
        plt.xlabel("x")
        plt.ylabel("Probability Density")

        # タイトルを表示
        plt.title("GMM K = " + str(n_components))

        # 凡例を表示
        plt.legend(loc="upper left", fontsize=10)
        plt.savefig("GMM.png")
        plt.show()

    # 2次元データの場合
    elif data.shape[1] == 2:
        x = np.linspace(data[:, 0].min(), data[:, 0].max(), 1000)
        y = np.linspace(data[:, 1].min(), data[:, 1].max(), 1000)
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        Z = np.zeros_like(X)

        # 混合ガウス分布の確率密度関数を計算
        for k in range(n_components):
            Z += weights[k] * multivariate_normal.pdf(
                pos, mean=means[k], cov=covariances[k]
            )

        # 確率密度関数をプロット
        CS = plt.contour(X, Y, Z)

        # カラーバーの設定
        plt.colorbar(CS, aspect=8)

        # 散布図をプロット
        plt.scatter(data[:, 0], data[:, 1], color="r", label="Data", marker=".")

        # ガウス分布の平均を表示
        plt.scatter(
            means[:, 0],
            means[:, 1],
            color="g",
            marker="x",
            label="Mean of Gaussian Distribution",
        )

        # 軸ラベルを表示
        plt.xlabel("x1")
        plt.ylabel("x2")

        # タイトルを表示
        plt.title("GMM K = " + str(n_components))

        # 凡例を表示
        plt.legend(loc="upper left", fontsize=10)
        plt.savefig("GMM.png")
        plt.show()

        # ヒートマップを作成
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # 3Dグラフ
        SF = ax.plot_surface(X, Y, Z, cmap="viridis")

        # 軸ラベルを表示
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Probability Density")

        # タイトルを表示
        plt.title("GMM K = " + str(n_components))

        # カラーバーの設定
        cbar = fig.colorbar(SF, aspect=8)
        cbar.ax.set_ylabel("probability density")

        plt.savefig("GMM_heatmap.png")
        plt.show()


def main():
    """読み込んだデータでGMMを用いたデータフィッティング."""
    # 引数を受け取る
    args = parse_args()

    # データを受け取る
    data = ex4.load_csv(args.filename)
    file_title = args.filename.replace(".csv", "")

    # データの次元数を取得
    dim = data.ndim
    if dim == 1:
        # 縦ベクトルにする
        data = data.reshape(-1, 1)

    # 散布図を作成
    ex4.make_scatter(data, dim, file_title + "_plot")

    # ランダムにパラメータの初期化を行う
    weights, means, covariances = random_initialize_para(data, args.n_components)

    # EMアルゴリズムを実行する
    log_likelihood_list, weights, means, covariances = EMstep(
        data, weights, means, covariances
    )

    # 対数尤度をグラフにする
    make_likelihoodgraph(log_likelihood_list)

    # 混合ガウス分布のプロット
    plot_gmm(data, weights, means, covariances)


if __name__ == "__main__":
    main()
