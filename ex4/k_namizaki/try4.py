import argparse
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

def parse_args() -> Any:
    """
    Get Arguments.

    Returns
    -------
    parser.parse_args() : 引数を返す
    """
    parser = argparse.ArgumentParser(description="最小二乗法を用いて回帰分析を行う。")
    parser.add_argument(
        "-file",
        help="ファイルを入力",
        default=r"C:\Users\kyskn\B4Lecture-2024\ex4\data2.csv",
        type=str,
    )
    return parser.parse_args()

class PrincipalComponentAnalysis:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = None
        self.st_data = None
        self.cov_data = None
        self.eig_data = None
        self.Eigenvalues = None
        self.Eigenvectors = None
        self.Y = None
        self.rate = None

    def standardization(self, data: ArrayLike) -> ArrayLike:
        st_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        return st_data

    def transformation_matrix(self, data: ArrayLike) -> ArrayLike:
        Y = self.Eigenvectors.T @ data.T
        return Y.T

    def cul_rate(self) -> ArrayLike:
        rate = self.Eigenvalues / sum(self.Eigenvalues)
        sorted_rate = np.sort(rate)[::-1]
        return sorted_rate

    def compression(self) -> ArrayLike:
        ac_rate = 0.0
        i = 0
        cum_rate = []
        while ac_rate < 0.9:
            ac_rate += self.rate[i]
            cum_rate.append(ac_rate)
            i += 1
        return cum_rate

    def load_data(self) -> None:
        self.data = np.loadtxt(self.file_path, delimiter=",", dtype="float")

    def pca(self) -> None:
        self.st_data = self.standardization(self.data)
        self.cov_data = np.cov(self.st_data, rowvar=0, bias=1)
        self.eig_data = np.linalg.eig(self.cov_data)
        self.Eigenvalues = self.eig_data[0]
        self.Eigenvectors = self.eig_data[1]
        self.Y = self.transformation_matrix(self.data)
        self.rate = self.cul_rate()

    def plot2d(self) -> None:
        fig, ax = plt.subplots()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.scatter(self.data[:, 0], self.data[:, 1], color="b", label="data1")

        vec1 = self.Eigenvectors.T[0]
        ax.plot(
            [-2 * vec1[0], 2 * vec1[0]],
            [-2 * vec1[1], 2 * vec1[1]],
            color="red",
            label=round(self.rate[0], 2),
        )
        vec2 = self.Eigenvectors.T[1]
        ax.plot(
            [-2 * vec2[0], 2 * vec2[0]],
            [-2 * vec2[1], 2 * vec2[1]],
            color="green",
            label=round(self.rate[1], 2),
        )
        ax.legend()
        plt.show()

    def plot3d(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], color="b", label="data2")

        vec1 = self.Eigenvectors.T[0]
        ax.plot(
            [-2 * vec1[0], 2 * vec1[0]],
            [-2 * vec1[1], 2 * vec1[1]],
            [-2 * vec1[2], 2 * vec1[2]],
            color="r",
            label=round(self.rate[0], 2),
        )
        vec2 = self.Eigenvectors.T[1]
        ax.plot(
            [-2 * vec2[0], 2 * vec2[0]],
            [-2 * vec2[1], 2 * vec2[1]],
            [-2 * vec2[2], 2 * vec2[2]],
            color="g",
            label=round(self.rate[1], 2),
        )
        vec3 = self.Eigenvectors.T[2]
        ax.plot(
            [-2 * vec3[0], 2 * vec3[0]],
            [-2 * vec3[1], 2 * vec3[1]],
            [-2 * vec3[2], 2 * vec3[2]],
            color="y",
            label=round(self.rate[2], 2),
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    def plot_comp(self) -> None:
        fig, ax = plt.subplots()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.scatter(self.Y[:, 0], self.Y[:, 1], color="b", label="compression data")
        ax.legend()
        plt.show()


def main() -> None:
    args = parse_args()
    pca = PrincipalComponentAnalysis(args.file)
    pca.load_data()
    pca.pca()
    if len(pca.data[0]) == 2:
        pca.plot2d()
    elif len(pca.data[0]) == 3:
        pca.plot3d()
        pca.plot_comp()
    else:
        cum_rate = pca.compression()
        print("Contribution rate", pca.rate)
        print("Cumulative contribution rate", cum_rate)
        print("Original:", len(pca.Y[0]), "Compressed:", len(cum_rate))


if __name__ == "__main__":
    main()
