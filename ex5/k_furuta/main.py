"""EMアルゴリズムを用いてデータをGMMで回帰分析するプログラム."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    """ガウス分布のパラメーター計算と可視化を行うプログラム."""

    def __init__(self, n_components, max_iter=100, tol=1e-6):
        """ガウス分布のハイパーパラメータを初期化する関数.

        Parameter
        -------------
        n_components : int
            ガウス分布の数
        mat_iter : int
            イテレーションの最大数
        tol : float
            対数尤度の収束条件
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, file_name):
        """ガウス分布のパラメータを収束するまで更新する関数.

        Parameter
        -------------
        self.n_components : int
            <self>のガウス分布の数
        self.mat_iter : int
            <self>のイテレーションの最大数
        self.tol : float
            <self>の対数尤度の収束条件
        X : np.ndarray, shape=(number of sample, dimension of data)
            対象となるデータ(ベクトルはエラーが起きる)
        file_name : str
            イテレーションごとの対数尤度(保存する画像の名前はhistory_ファイル名.png)
        """
        n_samples, n_features = X.shape

        # 初期化
        np.random.seed(42)
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X, rowvar=False)] * self.n_components)
        self.weights_ = np.ones(self.n_components) / self.n_components

        # 対数尤度は初めは0としておく
        log_likelihood = 0
        self.log_likelihood_ = []

        for _ in range(self.max_iter):
            # Eステップ
            responsibilities = self._estimate_responsibilities(X)

            # Mステップ
            self._m_step(X, responsibilities)

            # 対数尤度の計算
            new_log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_.append(new_log_likelihood)

            # 更新後の差がtol以下なら収束
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood
        
        # 対数尤度の推移をプロット
        plt.figure()
        plt.plot(np.array(self.log_likelihood_)/n_samples)
        # 軸の名前と凡例を付けて保存
        plt.title("")
        plt.xlabel("Iteration")
        plt.ylabel("Log likelihood")
        plt.savefig(f"history_{file_name}.png")

    def _estimate_responsibilities(self, X):
        """ガウス分布の負担率を計算する関数.

        Parameter
        -------------
        self.n_components : int
            <self>のガウス分布の数
        self.mat_iter : int
            <self>のイテレーションの最大数
        self.tol : float
            <self>の対数尤度の収束条件
        X : np.ndarray, shape=(number of sample, dimension of data)
            対象となるデータ
        Return
        -------------
        responsibilities : np.ndarray, shape=(number of sample, n_components)
            負担率
        """
        # 負担率の計算
        responsibilities = np.zeros((X.shape[0], self.n_components))
        # 各クラスタごとにpiによる重み付き確率密度関数を計算
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, self.means_[k], self.covariances_[k]
            )
        # サンプルごとに負担率の和が1になるように割る
        sum_responsibilities = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities /= sum_responsibilities
        return responsibilities

    def _m_step(self, X, responsibilities):
        """負担率から新しいガウス分布のパラメータを求める関数.

        Parameter
        -------------
        self.n_components : int
            <self>のガウス分布の数
        self.mat_iter : int
            <self>のイテレーションの最大数
        self.tol : float
            <self>の対数尤度の収束条件
        X : np.ndarray, shape=(number of sample, dimension of data)
            対象となるデータ
        """
        # パラメータN_kを計算
        N_k = responsibilities.sum(axis=0)
        # 講義資料の式に沿って計算
        self.weights_ = N_k / X.shape[0]
        self.means_ = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]
        self.covariances_ = np.zeros((self.n_components, X.shape[1], X.shape[1]))
        # 分散共分散行列はクラスタごとに計算
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = (
                np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k]
            )

    def _compute_log_likelihood(self, X):
        """パラメータから負の対数尤度を計算する関数.

        Parameter
        -------------
        self.n_components : int
            <self>のガウス分布の数
        self.mat_iter : int
            <self>のイテレーションの最大数
        self.tol : float
            <self>の対数尤度の収束条件
        X : np.ndarray, shape=(number of sample, dimension of data)
            対象となるデータ
        """
        # scipyの関数を用いて対数尤度を求める
        log_likelihood = 0
        for k in range(self.n_components):
            # piによる重みつきの確率密度の和を求める
            log_likelihood += self.weights_[k] * multivariate_normal.pdf(
                X, self.means_[k], self.covariances_[k]
            )
        # 最後に対数をとる
        return np.log(log_likelihood).sum()

    def visualize(self, X, file_name):
        """データとfit済みのパラメータから結果をプロットする関数.

        Parameter
        -------------
        X : np.ndarray, shape=(number of sample, dimension of data)
            対象となるデータ
        file_name : str
            読み込んだデータのファイル名(保存する画像の名前はresult_ファイル名.png)
        """
        # 1,2次元の場合にそれぞれ表示用関数を呼び出す
        # 3次元以上の場合は何もしない
        if X.shape[1] == 1:
            self._visualize_1d(X, file_name)
        elif X.shape[1] == 2:
            self._visualize_2d(X, file_name)
        else:
            pass

    def _visualize_1d(self, X, file_name):
        """データとfit済みのパラメータから結果をプロットする関数.

        Parameter
        -------------
        X : np.ndarray, shape=(number of sample, dimension of data)
            対象となるデータ
        file_name : str
            読み込んだデータのファイル名(保存する画像の名前はresult_ファイル名.png)
        """
        plt.figure()
        # 出力範囲内で適当な間隔でxを計算しておく
        x = np.linspace(np.min(X), np.max(X), 1000)
        y = np.zeros_like(x)

        # ガウス分布のパラメータからその地点での確率密度を計算する
        for k in range(self.n_components):
            y += self.weights_[k] * multivariate_normal.pdf(
                x, self.means_[k], self.covariances_[k]
            )

        # 1次元のデータは見ずらいと思ったのでヒストグラムで表示(正確な位置が失われるのでよくないかもしれない)
        plt.hist(X, bins=30, density=True, alpha=0.5, color="gray")
        # GMMの確率密度をプロット
        plt.plot(x, y, label="GMM Likelihood")
        # 軸の名前と凡例を付けて保存
        plt.title("1D Gaussian Mixture Model")
        plt.xlabel("Data")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f"result_{file_name}.png")

    def _visualize_2d(self, X, file_name):
        """データとfit済みのパラメータから結果をプロットする関数.

        Parameter
        -------------
        X : np.ndarray, shape=(number of sample, dimension of data)
            対象となるデータ
        file_name : str
            読み込んだデータのファイル名(保存する画像の名前はresult_ファイル名.png)
        """
        plt.figure()
        # データの散布図をプロット
        plt.scatter(X[:, 0], X[:, 1], s=4, label="Data")

        # 表示範囲内で適当な間隔で取ったx,yの直積をとる
        x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
        X_grid, Y_grid = np.meshgrid(x, y)
        pos = np.empty(X_grid.shape + (2,))
        pos[:, :, 0] = X_grid
        pos[:, :, 1] = Y_grid

        # 直積ごとにガウス分布のパラメータからその地点での確率密度を計算する
        Z = np.zeros(X_grid.shape)
        for k in range(self.n_components):
            rv = multivariate_normal(self.means_[k], self.covariances_[k])
            Z += self.weights_[k] * rv.pdf(pos)

        # 等高線を表示
        plt.contour(X_grid, Y_grid, Z, levels=10, cmap="viridis")
        # 軸の名前と凡例を付けて保存
        plt.title("2D Gaussian Mixture Model")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.savefig(f"result_{file_name}.png")


def parse_args():
    """コマンドプロントから引数を受け取るための関数."""
    parser = argparse.ArgumentParser(description="Plot PCA from data")
    parser.add_argument(
        "--input-file", type=str, required=True, help="Name of input csv file"
    )
    parser.add_argument(
        "--n-components", type=int, required=True, help="Number of gaussian"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100, help="Maximum number of iteration"
    )
    parser.add_argument("--tol", type=float, default=1e-6, help="convergence condition")
    return parser.parse_args()


if __name__ == "__main__":
    # 引数の受け取り
    args = parse_args()
    # ファイルパスとファイル名の受け取り
    file_path = args.input_file
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # GMMのハイパーパラメータの受け取り
    n_components = args.n_components
    max_iter = args.max_iter
    tol = args.tol

    # csvファイルの読み込み
    data = np.genfromtxt(file_path, delimiter=",")
    # データが1次元でも2次元と同じ処理をするため、形状を(x,)から(x,1)に変形
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # 初期化
    gmm = GMM(n_components=n_components, max_iter=max_iter, tol=tol)
    # パラメータの学習
    gmm.fit(data, file_name)
    # 結果の表示
    gmm.visualize(data, file_name)
