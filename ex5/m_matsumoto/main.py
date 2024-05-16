"""Do GMM Fitting."""

import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import multivariate_normal


class clsGMM:
    """Do GMM Fitting."""

    def __init__(self, str_csv_path: str, is_csv_header: bool = False) -> None:
        """Read csv.

        Args:
            str_csv_path (str): csv file path
            is_csv_header (bool, optional): csv has header. Defaults to False.
        """
        try:
            self.df_raw: pl.DataFrame = pl.read_csv(
                str_csv_path, has_header=is_csv_header, encoding="utf8"
            )
            # 絶対パス
            self.path_csv_abs_path: Path = Path(str_csv_path).resolve()
            os.makedirs(self.path_csv_abs_path.stem, exist_ok=True)
        except Exception as e:
            raise ValueError(
                f"""
        csv file must with only float data.
        input: {str_csv_path}
        error: {e}"""
            )

        self.n, self.dim = self.df_raw.shape

        assert 1 <= self.dim <= 2, f"dimention must be 1<={self.dim}<=2"

    def em(
        self, k: int, EPSILON: float = 10e-3, MAX_LOOP: int = 100
    ) -> Tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
        """EM algorithm.

        Args:
            k (int): number of cluster
            EPSILON (float, optional): use for ending condition. Defaults to 10e-3.
            MAX_LOOP (int): max loop

        Returns:
            Tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
            ls_Ls: Likelihood by updating
            pi: mixing coefficients
            mu: mean
            sigma: variance-covariance matrix
        """
        # initializer
        x: np.ndarray = np.zeros((self.dim, self.n))
        for i, c in enumerate(self.df_raw.columns):
            x[i] = self.df_raw[c]
        mu: np.ndarray = np.random.rand(k, self.dim)
        sigma: np.ndarray = np.array([np.eye(self.dim) for _ in range(k)])
        pi: np.ndarray = np.random.rand(k)
        pi /= np.sum(pi, axis=0)  # 確率として扱うため，正規化する．
        ls_Ls: list[float] = [-float("inf")]

        # update
        for _ in range(MAX_LOOP):
            # gamma
            gamma: np.ndarray = np.zeros((k, self.n))
            for i in range(self.n):
                denominator: float = 0.0
                for j in range(k):
                    nominator: float = pi[j] * multivariate_normal(
                        mean=mu[j, :], cov=sigma[j, :, :]
                    ).pdf(x[:, i])
                    denominator += nominator
                    gamma[j, i] = nominator
                gamma[:, i] /= denominator

            # Nk
            Nk: np.ndarray = np.sum(gamma, axis=1)

            # pi
            pi = Nk / np.sum(Nk, axis=0)

            # mu
            mu = np.zeros_like(mu)
            for i in range(k):
                for j in range(self.n):
                    mu[i, :] += gamma[i, j] * x[:, j]
                mu[i, :] /= Nk[i]

            # sigma
            sigma = np.zeros_like(sigma)
            for i in range(k):
                for j in range(self.n):
                    dev: np.ndarray = x[:, j] - mu[i, :]
                    sigma[i, :, :] += gamma[i, j] * np.outer(dev, dev)
                sigma[i, :] /= Nk[i]

            # L
            L: float = 0.0
            for i in range(self.n):
                tmp: float = 0.0
                for j in range(k):
                    tmp += pi[j] * multivariate_normal(
                        mean=mu[j, :], cov=sigma[j, :, :]
                    ).pdf(x[:, i])
                L += np.log(tmp)
            ls_Ls.append(L)
            if ls_Ls[-1] - ls_Ls[-2] < EPSILON:
                print(_)
                break
        return ls_Ls[1:], pi, mu, sigma

    def likelihood_plot(self, ls_Ls: list[float]) -> None:
        """Plot Likelihood.

        Args:
            ls_Ls (list[float]): Likelihood by updating.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, len(ls_Ls)), ls_Ls, marker="o", linestyle="-", color="b")
        plt.xlabel("Update Times")
        plt.ylabel("L")
        plt.title("L by updating")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.path_csv_abs_path.stem, "likelihood.png"))
        plt.show()

    def gaussian_plot(self, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Plot Gaussian.

        Args:
            pi (np.ndarray): soft clustering.
            mu (np.ndarray): mean
            sigma (np.ndarray): variance-covariance matrix
        """
        if self.dim == 1:

            # scatter
            plt.scatter(
                self.df_raw["column_1"], [0] * self.n, marker="o", s=30, alpha=0.1
            )
            # mean
            for i in range(pi.shape[0]):
                w: float = pi[i]
                mean = mu[i]
                cov = sigma[i]
                # mean
                plt.plot(mean, 0, "x", label=f"mean: {i}")
                # gauss
                x_range: np.ndarray = np.linspace(
                    min(self.df_raw["column_1"]), max(self.df_raw["column_1"]), 100
                )
                plt.plot(
                    x_range,
                    w * multivariate_normal(mean, cov).pdf(x_range),
                    label=str(i),
                )
                plt.grid(True)
                plt.legend()

        if self.dim == 2:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(121, projection="3d")
            ax.scatter(
                self.df_raw["column_1"],
                self.df_raw["column_2"],
                [0] * self.n,
                color="blue",
                s=10,
            )
            for i, w in enumerate(pi):
                mean = mu[i]
                cov = sigma[i]
                x_range = np.linspace(
                    min(self.df_raw["column_1"]), max(self.df_raw["column_1"]), 100
                )
                y_range = np.linspace(
                    min(self.df_raw["column_2"]), max(self.df_raw["column_2"]), 100
                )
                X, Y = np.meshgrid(x_range, y_range)
                pos = np.dstack((X, Y))
                Z = w * multivariate_normal(mean, cov).pdf(pos)
                ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Probability Density")
                ax.set_title("3D Gaussian Distributions")

        plt.savefig(os.path.join(self.path_csv_abs_path.stem, "gaussian.png"))
        plt.show()

    def scatter_plot(self, is_draw=True) -> None:
        """Scatter Plot."""
        if self.dim == 1:
            plt.figure()
            plt.scatter(
                self.df_raw["column_1"], [0] * self.n, marker="o", s=30, alpha=0.1
            )
            plt.yticks([])  # Y軸を消す
            plt.title(self.path_csv_abs_path.stem)
            plt.xlabel("$X$")
            # plt.ylabel("$Y$")
        if self.dim == 2:
            plt.figure()
            plt.scatter(
                self.df_raw["column_1"],
                self.df_raw["column_2"],
                marker="o",
                s=30,
                alpha=0.1,
            )
            # plt.yticks([])
            plt.title(self.path_csv_abs_path.stem)
            plt.xlabel("$X$")
            plt.ylabel("$Y$")

        if is_draw:
            plt.savefig(os.path.join(self.path_csv_abs_path.stem, "scatter.png"))
            plt.show()


def kadai(str_csv_path: str, k: int) -> None:
    """Do kadai.

    Args:
        str_csv_path (str): csv path
        k (int): number of cluster.
    """
    gmm = clsGMM(str_csv_path)
    gmm.scatter_plot()
    l, p, m, s = gmm.em(k, 0.1)
    gmm.gaussian_plot(p, m, s)
    gmm.likelihood_plot(l)


if __name__ == "__main__":

    SEED = 1000
    np.random.seed(SEED)
    kadai("../data1.csv", 2)
    kadai("../data2.csv", 3)
    kadai("../data3.csv", 5)
