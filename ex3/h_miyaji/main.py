"""最小二乗法を用いて回帰分析を行う."""

import matplotlib.pyplot as plt  # グラフ描画
import numpy as np  # 線形代数
from mpl_toolkits.mplot3d import Axes3D


def load_data(filename: str) -> np.ndarray:
    """csvファイルからndarray配列を作成.

    Args:
        filename (str): csvファイルの名前

    Returns:
        np.ndarray: csvファイル内のデータ
    """
    data_set = np.loadtxt(fname=filename, dtype="float", delimiter=",", skiprows=1)
    return data_set


def plot_scatter_diag(
    dataset: np.ndarray,
    title: str = "scatter diagram",
    label: str = "Observed data",
    xlabel: str = "x1",
    ylabel: str = "x2",
    zlabel: str = "x3",
):
    """新たにグラフ描画領域を作り、散布図をプロットする.

    Args:
        dataset (np.ndarray): csvデータ
        dim (int, optional): グラフの次元数（2 or 3）. Defaults to 2.
        title (str, optional): グラフタイトル. Defaults to "scatter diagram".
        xlabel (str, optional): X軸のラベル名. Defaults to "x1".
        ylabel (str, optional): Y軸のラベル名. Defaults to "x2".
        zlabel (str, optional): Z軸のラベル名. Defaults to "x3".

    Returns:
        ax: Axisオブジェクト
    """

    fig = plt.figure()

    dim = len(dataset[0])

    x1 = dataset[:, 0]
    x2 = dataset[:, 1]

    if dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(x1, x2, label=label, color="b", marker=".")
    elif dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        x3 = dataset[:, 2]
        ax.scatter(x1, x2, x3, label=label, color="b", marker=".")
        ax.set_zlabel(zlabel)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    # plt.show()
    return ax


def plot_reg_model(
    dataset: np.ndarray,
    ax,
    dim: int = 2,
):
    """散布図に回帰モデルを重ねて表示する.

    Args:
        dataset (np.ndarray): 回帰モデルのデータ
        ax (Axis, optional): プロットを追加したい場所を示すAxisオブジェクト.
        dim (int, optional): 2Dグラフ or 3Dグラフ. Defaults to 2.


    Returns:
        ax: Axisオブジェクト
    """

    # if ax is None:
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111, projection="3d")
    #    ax.grid()
    #    col = "blue"
    # else:
    #    col = "orange"

    # labelつくりたい

    if dim == 2:
        ax.scatter(dataset[0], dataset[1], color="orange", alpha=0.1, marker=".")
    elif dim == 3:
        ax.scatter(
            dataset[0], dataset[1], dataset[2], color="orange", alpha=0.1, marker="."
        )

    ax.legend()
    plt.show()
    return ax


def calc_obj_var(dataset: np.ndarray) -> np.ndarray:

    # 目的変数
    y = dataset[:, -1].reshape(-1, 1)
    return y


def calc_ind_var(dataset: np.ndarray, N: int) -> np.ndarray:

    # 独立変数
    # 行列X作成
    X = np.ones((len(dataset), 1), dtype=float)
    # X = np.hstack((X, dataset[:, 0].reshape(-1, 1)))

    # 散布図の軸数（x-y / x-y-z）
    dim = len(dataset[0])

    if dim == 2:  # x-y：次元数分だけ増やす
        for i in range(N):
            X = np.hstack((X, dataset[:, 0].reshape(-1, 1) ** (i + 1)))

    elif dim == 3:  # x-y-z：xとyを増やす
        for i in range(N):
            X = np.hstack((X, dataset[:, 0].reshape(-1, 1) ** (i + 1)))
            X = np.hstack((X, dataset[:, 1].reshape(-1, 1) ** (i + 1)))

    return X


def calc_beta_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    # 行列betaを0で初期化　TODO: いらない？
    # beta = np.zeros((1 + N, 1))

    # beta計算　TODO: ここに正規化を入れる？
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    return beta


def calc_e(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    # 行列eを0で初期化　TODO: いらない？
    # e = np.zeros((len(y), 1))

    # e 計算
    e = y - (X @ beta)
    return e


def calc_axis_x(
    dataset: np.ndarray, beta: np.ndarray, N: int, point: int = 10000
) -> list:

    # 行列X作成
    # point = len(dataset)
    axis_x = np.ones((point, 1), dtype=float)
    # X = np.hstack((X, dataset[:, 0].reshape(-1, 1)))

    x = np.linspace(min(dataset[:, 0]), max(dataset[:, 0]), point)

    # 散布図の軸数（x-y / x-y-z）
    dim = len(dataset[0])

    if dim == 2:  # x-y：次元数分だけ増やす
        for i in range(N):
            axis_x = np.hstack((axis_x, x.reshape(-1, 1) ** (i + 1)))
        fx = axis_x @ beta
        model_dataset = [x, fx]

    elif dim == 3:  # x-y-z：xとyを次元数分だけ増やす
        y = np.linspace(min(dataset[:, 1]), max(dataset[:, 1]), point)
        XX, YY = np.meshgrid(x, y)
        fx = np.zeros((point, point))
        fx += beta[0]
        for i in range(N):
            fx += (XX ** (i + 1)) * beta[(2 * i) + 1]
            fx += (YY ** (i + 1)) * beta[(2 * i) + 2]
        model_dataset = [XX, YY, fx]

    return model_dataset


def create_reg_model(dataset: np.ndarray, N: int, point: int = 10000) -> list:

    # dim = len(dataset[0])

    y = calc_obj_var(dataset)
    X = calc_ind_var(dataset, N)

    beta = calc_beta_ols(y, X)

    reg_model = calc_axis_x(dataset, beta, N, point)

    print(reg_model[0].shape)
    print(reg_model[1].shape)

    return reg_model


def main():
    """csvファイルからデータを読み込み, 回帰分析を行う."""

    # csvファイル読み込み
    data1 = load_data("data1.csv")
    data2 = load_data("data2.csv")
    data3 = load_data("data3.csv")

    # 散布図のプロット
    ax1 = plot_scatter_diag(data1)
    ax2 = plot_scatter_diag(data2)
    ax3 = plot_scatter_diag(data3)

    # ep = calc_e(y, X, beta)

    reg_model1 = create_reg_model(data1, N=1)
    reg_model2 = create_reg_model(data2, N=3)
    reg_model3 = create_reg_model(data3, N=2, point=100)

    plot_reg_model(reg_model1, ax=ax1)
    plot_reg_model(reg_model2, ax=ax2)
    plot_reg_model(reg_model3, ax=ax3, dim=3)
    # plt.show()


if __name__ == "__main__":
    main()
