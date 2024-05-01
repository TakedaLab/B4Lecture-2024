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
            data (np.ndarray): data = (x, y)

        Raises:
            ValueError: x has to be 2 dimention
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
        is_label: bool = True,
    ) -> None:
        """Draw Graph.

        Args:
            is_expect (bool): Do Regression.
            N (int, optional): N-Polinominal Regression. Without regression, needless.
            alphas (List[float], optional): Uses for calculating Beta.
                0.0 is neccesary for without regularized regression.
            is_label (bool): Graph with label.
        """
        # x_axis division num.
        DIVIDE = 100
        # データプロット
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

        if is_label:
            plt.title("Regression")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
        else:
            plt.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )
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


class clsRegression2:
    """Regression with 2 Independent Valuables."""

    def __init__(self, data: np.ndarray) -> None:
        """Initialize.

        Args:
            data (np.ndarray): data = (x1, x2, y)

        Raises:
            ValueError: x has to be 3 dimention
        """
        if data.shape[1] != 3:
            raise ValueError(f"data must be (x, 3) but got {data.shape}")

        self.x1: np.ndarray = data[:, 0].reshape(-1, 1)
        self.x2: np.ndarray = data[:, 1].reshape(-1, 1)
        self.y: np.ndarray = data[:, 2].reshape(-1, 1)

    def draw(
        self,
        is_expect: bool,
        N1: int = -1,
        N2: int = -1,
        alphas: List[float] = [0.0],
        is_label: bool = True,
    ) -> None:
        """Draw Graph.

        Args:
            is_expect (bool): Do Regression
            Nx (int, optional): N-Polinominal Regression. Without regression, needless.
            alphas (List[float], optional): Uses for calculating Beta.
                0.0 is neccesary for without regularized regression.
            is_label (bool): Graph with label.
        """
        # x_axes division num.
        DIVIDE = 100
        FIGSIZE = (6, 6)
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(121, projection="3d")

        # グラフ作成

        # データプロット
        ax.scatter(self.x1, self.x2, self.y, marker="o", c="red", label="Observed")

        if is_expect:
            x1_axis: np.ndarray = np.linspace(min(self.x1), max(self.x2), DIVIDE)
            x2_axis: np.ndarray = np.linspace(min(self.x2), max(self.x2), DIVIDE)
            x1_mesh, x2_mesh = np.meshgrid(x1_axis, x2_axis)
            # 0~Nx次の配列
            X: np.ndarray = np.concatenate(
                # 0次元は一つあれば良い．ため，最初だけrangeを0から開始．
                [self.x1**i for i in range(0, N1 + 1)]
                + [self.x2**i for i in range(1, N2 + 1)],
                axis=1,
            )
            # print(X.shape)
            # exit()
            # print(x1_axis)
            # print(x2_axis)
            # exit()

            for alpha in alphas:
                # 正規方程式
                beta = self._beta(X, self.y, alpha)
                ax.plot_wireframe(
                    x1_mesh,
                    x2_mesh,
                    self._expect(beta, N1, N2, x1_mesh, x2_mesh),
                    label=self._equation(beta, N1, N2),
                )

        if is_label:
            ax.set_title("Regression")
            ax.set_xlabel("$x_{1}$")
            ax.set_ylabel("$x_{2}$")
            ax.set_zlabel("y")
            ax.legend()
        else:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )
        plt.show()

    def _expect(
        self,
        beta: np.ndarray,
        N1: int,
        N2: int,
        x1_mesh: np.ndarray,
        x2_mesh: np.ndarray,
    ) -> np.ndarray:
        if len(beta) - 1 != N1 + N2:
            raise ValueError(
                f"""sum of the dimentions has to be beta.
                len(beta) - 1 == N1 + N2
                but
                {len(beta)-1} == {N1+N2}"""
            )

        y = np.zeros_like(x1_mesh)
        for i in range(N1):
            # print(y.shape, beta[i].shape, x1_mesh.shape)
            # exit()
            y += beta[i] * x1_mesh**i
        for i in range(1, N2 + 1):
            y += beta[N1 + i] * x2_mesh**i

        return y

    def _equation(self, beta: np.ndarray, N1: int, N2: int) -> np.ndarray:
        if len(beta) - 1 != N1 + N2:
            raise ValueError(
                f"""sum of the dimentions has to be beta.
                len(beta) - 1 == N1 + N2
                but
                {len(beta)-1} == {N1+N2}"""
            )

        ret = "y="
        for i in range(N1 + 1):
            b = beta[i][0]
            if i == 0:
                ret += f"{b:+.2f}"
            else:
                ret += f"{b:+.2f}$x_{1}^{i}$"

        for i in range(1, N2 + 1):
            b = beta[N1 + i][0]
            ret += f"{b:+.2f}$x_{2}^{i}$"

        return ret

    def _beta(self, X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:

        return np.linalg.inv(X.T @ X + (alpha * np.identity(X.shape[1]))) @ X.T @ y


if __name__ == "__main__":
    # data1 = load_csv("../data2.csv")

    # reg1 = clsRegression1(data1)

    # reg1.draw(False, 3, [0.0, 1.0], True)

    data3 = load_csv("../data3.csv")

    # グラフ描画の関係があるから，同一クラスはやめた．
    reg2 = clsRegression2(data3)

    reg2.draw(True, 1, 2, [0.0, 10.0], True)
