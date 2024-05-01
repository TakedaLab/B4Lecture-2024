"""Read CSV and Do Linear Regression."""

import csv
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_csv(str_path: str, int_skip_header: int = 1) -> np.ndarray[float]:
    """Return numpy array with float.

    Args:
        str_path (str): csv file path
        int_skip_header (int, optional): lines of header(s). Defaults to 1.

    Returns:
        np.ndarray[csv.Any, np.dtype[csv.Any]]: float numpy array
    """
    return np.genfromtxt(str_path, delimiter=",", skip_header=1).astype(float)


class clsRegression1:

    def __init__(self, data: np.ndarray) -> None:
        if data.shape[1] != 2:
            raise ValueError(f"data must be (x, 2) but got {data.shape}")

        self.x: np.ndarray = data[:, 0].reshape(-1, 1)
        self.y: np.ndarray = data[:, 1].reshape(-1, 1)

    def draw(self, is_expect: bool, N: int = 1, alphas: List[float] = []) -> None:
        DIVIDE = 100
        plt.scatter(
            self.x, self.y, facecolor="None", edgecolors="red", label="Observed"
        )

        if is_expect:
            # X軸
            x_axis = np.linspace(min(self.x), max(self.x), DIVIDE)
            # print(self.x)
            # 0~N次のxの配列
            X: np.ndarray = np.concatenate([self.x**i for i in range(0, N + 1)], axis=1)
            print(X)
            # 正規方程式
            beta = self._beta(X, self.y, 0.0)
            # print(beta)
            # 正則化をせずに描画
            plt.plot(x_axis, self._expect(beta, x_axis), label=self._equation(beta))

            for alpha in alphas:
                # 正規方程式
                beta = self._beta(X, self.y, alpha)
                # 正則化をせずに描画
                plt.plot(x_axis, self._expect(beta, x_axis), label=self._equation(beta))
                plt.title("Regression")
                plt.xlabel("x")
                plt.ylabel("y")
            plt.legend()
        plt.show()

    def _expect(self, beta: np.ndarray, x_axis: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x_axis)
        for i, b in enumerate(beta):
            y += b * x_axis**i

        return y

    def _equation(self, beta: np.ndarray) -> str:
        ret = "y="
        for i in range(len(beta)):
            b = beta[i][0]

            if i == 0:
                ret += f"{b:+.2f}"
            else:
                ret += f"{b:+.2f}$x^{i}$"

        return ret

    def _beta(self, X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:

        return np.linalg.inv(X.T @ X + (alpha * np.identity(X.shape[1]))) @ X.T @ y


if __name__ == "__main__":
    data1 = load_csv(
        r"/Users/mattsunkun/research/bm-rinko/B4Lecture-2024/ex3/data1.csv"
    )

    reg1 = clsRegression1(data1)

    reg1.draw(True, 1, [])
