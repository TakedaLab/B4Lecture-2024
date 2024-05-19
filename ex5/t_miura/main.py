"""GMMを用いたデータのフィッティングを行う.

argparse   : コマンドライン引数
matplotlib : グラフのプロット
numpy      : 行列計算
pandas     : csvファイルの読み込み
scipy      : ガウス分布の計算
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def get_args():
    """コマンドライン引数の取得.

    -> ファイル名: str, 混合数: int
    """
    # コマンドライン引数を取得
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="csv filename")
    parser.add_argument("n_class", type=int, help="the number of class")
    args = parser.parse_args()
    filename, n_class = args.filename, args.n_class

    if filename[-4:] != ".csv":
        raise (ValueError("filename must be csv file"))

    return filename, n_class


def plot_log_likelihood(filename: str, log_likelihood_list: list):
    """対数尤度の変化をプロット.

    -> None
    """
    plt.plot(log_likelihood_list)
    plt_name = filename.replace(".csv", "_log-likelihood")
    plt.title(plt_name)
    plt.xlabel("steps")
    plt.ylabel("log-likelihood")
    plt.savefig(plt_name + ".png")
    plt.show()

    return


class GMM:
    """混合ガウスモデルによるデータのフィッティングを行う."""
    def __init__(self, filename: str, n_class: int):
        """コンストラクタ.

        -> None
        """
        self.filename = filename
        self.n_class = n_class
        self.data = None
        self.n_dim = None
        self.n_sample = None
        self.data_max = None
        self.data_min = None
        self.param = None
        self.pi_n = None
        self.gamma = None
        return

    def get_csvdata(self):
        """csvファイルを読み取り、ndarrayに変換.

        -> None
        """
        # pandasでcsvファイルを読み取り
        df_pd = pd.read_csv(f"../{self.filename}", header=None)

        # pd.DataFrameをnp.ndarrayに変換
        self.data = np.array(df_pd)

        # サンプル数と次元数を確認
        self.n_sample, self.n_dim = self.data.shape

        # 最大値最小値を確認
        self.data_max = np.max(self.data, axis=0)
        self.data_min = np.min(self.data, axis=0)

        return

    def plot_dispersal_chart(self, distribution_flag: bool):
        """散布図と混合分布を描画.

        -> None
        """
        if self.data is None:
            self.get_csvdata()

        if self.n_dim > 2:
            print("this program cannot plot more than 3-dimention data")
            return

        # 散布図を描画
        if self.n_dim == 1:
            plt.scatter(
                self.data[:, 0],
                np.zeros(self.n_sample),
                label="Observed data",
                alpha=0.5,
            )
        else:
            plt.scatter(self.data[:, 0], self.data[:, 1], label="Observed data")

        # 混合分布を描画
        if distribution_flag:
            if self.param is None:
                raise (ValueError("param is not set"))

            PLOT_COUNT = 1000

            if self.n_dim == 1:
                # 平均をプロット
                plt.scatter(
                    self.param[0][:, 0],
                    np.zeros(self.n_class),
                    label="Centroid",
                    color="red",
                    marker="x",
                )

                # 混合分布を計算
                mixed_dist_x = (
                    np.arange(PLOT_COUNT)
                    * (self.data_max[0] - self.data_min[0])
                    / (PLOT_COUNT - 1)
                    + self.data_min[0]
                )
                mixed_dist_y = np.zeros(PLOT_COUNT)
                temp = mixed_dist_x.reshape((PLOT_COUNT, 1))
                for i in range(self.n_class):
                    mixed_dist_y += (
                        stats.multivariate_normal.pdf(
                            temp,
                            mean=self.param[0][i],
                            cov=self.param[1][i],
                            allow_singular=True,
                        )
                        * self.param[2][i]
                    )

                # 混合分布をプロット
                plt.plot(mixed_dist_x, mixed_dist_y, label="GMM", color="orange")

            else:
                # 平均をプロット
                plt.scatter(
                    self.param[0][:, 0],
                    self.param[0][:, 1],
                    label="Centroid",
                    color="red",
                    marker="x",
                )

                # 混合分布を計算
                x = (
                    np.arange(PLOT_COUNT)
                    * (self.data_max[0] - self.data_min[0])
                    / (PLOT_COUNT - 1)
                    + self.data_min[0]
                )
                y = (
                    np.arange(PLOT_COUNT)
                    * (self.data_max[1] - self.data_min[1])
                    / (PLOT_COUNT - 1)
                    + self.data_min[1]
                )
                X, Y = np.meshgrid(x, y)
                temp = np.stack([X.ravel(), Y.ravel()]).T
                z = np.zeros(PLOT_COUNT**2)
                for i in range(self.n_class):
                    z += (
                        stats.multivariate_normal.pdf(
                            temp,
                            mean=self.param[0][i],
                            cov=self.param[1][i],
                            allow_singular=True,
                        )
                        * self.param[2][i]
                    )
                Z = z.reshape((PLOT_COUNT, PLOT_COUNT))

                # 混合分布をプロット
                plt.contour(X, Y, Z, levels=10)

        # 軸ラベル等の設定
        plt_name = self.filename.replace(".csv", "")
        if distribution_flag:
            plt_name += "_mixed_dist"
        plt.title(plt_name)
        plt.xlabel("X1")
        if self.n_dim == 1:
            if distribution_flag:
                plt.ylabel("value of GMM")
            else:
                plt.ylabel("")
        else:
            plt.ylabel("X2")
        plt.legend()

        # プロットの保存と表示
        plt.savefig(plt_name + ".png")
        plt.show()

        return

    def decede_initial_param(self, seed: int = 0):
        """パラメータの初期値を決定.

        -> None
        """
        # シード値を設定
        np.random.seed(seed)
        # 平均ベクトルをランダムで決定
        mu = np.random.rand(self.n_class, self.n_dim)
        for i in range(self.n_dim):
            mu[:, i] = (
                mu[:, i] * (self.data_max[i] - self.data_min[i]) + self.data_min[i]
            )

        # 分散共分散行列をすべて基本行列で決定
        sigma = np.stack([np.eye(self.n_dim)] * self.n_class)

        # 各ガウス分布の重みをランダムで決定
        temp2 = np.random.rand(self.n_class)
        pi = temp2 / np.sum(temp2)

        # 各パラメータを設定
        self.param = [mu, sigma, pi]

        # param更新処理
        self.reset_param()

        return

    def reset_param(self):
        """paramが更新されたときpi_nとgammaを削除.

        -> None
        """
        self.pi_n = None
        self.gamma = None

        return

    def calc_pi_n(self):
        """pi_nを計算.

        -> None
        """
        # 計算過程を記録する配列を定義
        self.pi_n = np.zeros((self.n_sample, self.n_class))

        for i in range(self.n_class):
            self.pi_n[:, i] = (
                stats.multivariate_normal.pdf(
                    self.data,
                    mean=self.param[0][i],
                    cov=self.param[1][i],
                    allow_singular=True,
                )
                * self.param[2][i]
            )

        return

    def calc_log_likelihood(self):
        """対数尤度の計算.

        -> 対数尤度: float
        """
        # pi_nを計算
        if self.pi_n is None:
            self.calc_pi_n()

        # 対数尤度を計算
        temp = np.sum(self.pi_n, axis=1)
        log_likelihood = np.sum(np.log(temp))

        return log_likelihood

    def E_step(self):
        """Eステップの実行.

        -> None
        """
        # pi_nを計算
        if self.pi_n is None:
            self.calc_pi_n()

        # 負担率を計算
        temp = np.sum(self.pi_n, axis=1).reshape((self.n_sample, 1))
        self.gamma = self.pi_n / temp

        return

    def M_step(self):
        """Mステップの実行.

        -> None
        """
        # Eステップが実行されたか確認
        if self.gamma is None:
            raise (ValueError("E-step was not complited"))

        # N_kを計算
        n = np.sum(self.gamma, axis=0)

        # 平均ベクトルを再計算
        mu = np.zeros((self.n_class, self.n_dim))

        for k in range(self.n_class):
            mu[k] = np.sum(self.gamma[:, k : k + 1] * self.data, axis=0) / n[k]

        # 分散共分散行列を再計算
        sigma = np.zeros((self.n_class, self.n_dim, self.n_dim))
        for k in range(self.n_class):
            x_sub_mu = self.data - self.param[0][k]
            sigma[k] = x_sub_mu.T @ (self.gamma[:, k : k + 1] * x_sub_mu) / n[k]

        # 各ガウス分布の重みを再計算
        pi = n / self.n_sample

        # パラメータを更新
        self.param = [mu, sigma, pi]
        self.reset_param()

        return

    def EM_algorithm(self, E: float = 1e-4):
        """EMアルゴリズムの実行.

        -> ステップ毎の対数尤度: list
        """
        # パラメータの初期化
        if self.param is None:
            self.decede_initial_param()

        # 対数尤度を記録する配列を定義
        log_likelihood_list = []

        # EMアルゴリズム
        l_old = -float("inf")
        l_new = self.calc_log_likelihood()
        log_likelihood_list.append(l_new)

        while l_new - l_old >= E:
            self.E_step()
            self.M_step()
            l_old = l_new
            l_new = self.calc_log_likelihood()
            log_likelihood_list.append(l_new)

        return log_likelihood_list


def main():
    """main関数.

    -> None
    """
    # コマンドライン引数の取得
    filename, n_class = get_args()

    # GMMインスタンスの生成
    gmm = GMM(filename, n_class)

    # csvファイルの読み込み
    gmm.get_csvdata()

    # データの散布図をプロット
    gmm.plot_dispersal_chart(False)

    # EMアルゴリズムを実行
    log_likelihood_list = gmm.EM_algorithm()

    # データと混合分布をプロット
    gmm.plot_dispersal_chart(True)

    # 対数尤度の変化をプロット
    plot_log_likelihood(filename, log_likelihood_list)


if __name__ == "__main__":
    main()
