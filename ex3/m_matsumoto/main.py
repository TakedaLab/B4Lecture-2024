"""Read CSV and Do Linear Regression."""

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
    """Regression with 1 Independent Valuable."""

    def __init__(self, data: np.ndarray) -> None:
        """Initialize.

        Args:
            data (np.ndarray): data= (x, y)

        Raises:
            ValueError: x has to be 1 dimention
        """
        if data.shape[1] != 2:
            raise ValueError(f"data must be (x, 2) but got {data.shape}")

        self.x: np.ndarray = data[:, 0].reshape(-1, 1)
        self.y: np.ndarray = data[:, 1].reshape(-1, 1)

    def draw(
        self,
        is_expect: bool,
        N: int = -1,
        alphas: List[float] = [0.0],
        label: bool = True,
    ) -> None:
        """Draw Graph.

        Args:
            is_expect (bool): Do Linear Regression.
            N (int, optional): N-Polinominal Regression. Without regression, needless.
            alphas (List[float], optional): Uses for calculating Beta.
                0.0 is neccesary for without regularized regression.
            label (bool): Graph with label.
        """
        # x_axis division num.
        DIVIDE = 100
        plt.scatter(
            self.x, self.y, facecolor="None", edgecolors="red", label="Observed"
        )

        # 回帰描画するとき
        if is_expect:
            # X軸
            x_axis = np.linspace(min(self.x), max(self.x), DIVIDE)
            # 0~N次のxの配列
            X: np.ndarray = np.concatenate([self.x**i for i in range(0, N + 1)], axis=1)
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
    data1 = load_csv("../data2.csv")

    reg1 = clsRegression1(data1)

    reg1.draw(True, 3, [0.0, 1.0])
