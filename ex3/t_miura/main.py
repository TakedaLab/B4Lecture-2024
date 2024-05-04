"""データに対して、最小二乗法を用いた回帰分析を行う.

sys        : コマンドライン引数
numpy      : 行列
matplotlib : 散布図の描画
pandas     : csvの読み取り
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_args():
    """コマンドライン引数を取得.

    -> csvファイル名: str, 次元数: int, 正則化係数: float
    """
    # コマンドライン引数を取得
    args = sys.argv  # args = ["main.py", FILENAME, N_DIM, REGULARIZATION_FACTOR]

    if len(args) == 4:
        filename = args[1]
        n_dim = int(args[2])
        regularization_factor = float(args[3])
        return filename, n_dim, regularization_factor
    else:
        # コマンドライン引数に過不足がある場合, USAGEを表示
        print("USAGE: main.py FILENAME N_DIM REGULARIZATION_FACTOR")
        return [None] * 3


def csv2ndarray(filename: str):
    """csvファイルを読み取り、ndarrayに変換.

    -> データ: np.ndarray, カラム名: tuple
    """
    # pandasでcsvファイルを読み取り
    df_pd = pd.read_csv(filename)  # 一行目のカラム名はdfのカラム名になる

    # pd.DataFrameをnp.ndarrayに変換
    df_np = np.array(df_pd)

    return df_np, tuple(df_pd.columns)


def plot_dispersal_chart(
    data: np.ndarray,
    filename: str,
    columns_name: tuple,
    linedata: np.ndarray = None,
    linename: str = None,
):
    """散布図を描画.

    -> None
    """
    if data.shape[1] == 2:
        # データの散布図をプロット
        plt.scatter(data[:, 0], data[:, 1], label="Observed data")

        # 回帰式を描画
        if linedata is not None:
            plt.plot(linedata[:, 0], linedata[:, 1], label=linename, color="orange")

        # ラベルを設定
        plt.title(filename.replace(".csv", ""))
        plt.xlabel(columns_name[0])
        plt.ylabel(columns_name[1])
        plt.legend()  # 凡例の表示

    elif data.shape[1] == 3:
        # 3dプロットの準備
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1,projection = "3d")

        # データの散布図をプロット
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], label="Observed data")

        # 回帰式を描画
        if linedata is not None:
            n_plot = int(np.sqrt(len(linedata)))
            x_new = linedata[::n_plot, 0]
            y_new = linedata[:n_plot, 1]
            x_new, y_new = np.meshgrid(x_new, y_new)
            z_new = linedata[:, 2].reshape((n_plot, n_plot)).T
            surf = ax.plot_surface(
                x_new,
                y_new,
                z_new,
                label=linename,
                color="orange",
                alpha=0.3,
            )
            surf._facecolors2d = surf._facecolor3d  # ラベル関連の操作
            surf._edgecolors2d = surf._edgecolor3d  # ラベル関連の操作

        # ラベルを設定
        ax.set_title(filename.replace(".csv", ""))
        ax.set_xlabel(columns_name[0])
        ax.set_ylabel(columns_name[1])
        ax.set_zlabel(columns_name[2])
        ax.legend()

    # 散布図の保存と描画
    if linedata is None:
        temp = ".png"
    else:
        temp = "_regression.png"
    plt.savefig(filename.replace(".csv", "") + temp)  # 画像の保存
    plt.show()  # 描画


def calc_regression(
    data: np.ndarray, n_dim: int, regularization_factor: float, columns_name: tuple
):
    """リッジ回帰により回帰式を計算し、それを表す配列を返す.

    -> 回帰式を表す配列: np.ndarray
    """
    # データの次元数を確認
    n_data, data_dim = data.shape

    # X,yを定義
    y = data[:, -1]
    X = np.zeros((1 + n_dim * (data_dim - 1), n_data), dtype=np.float64)
    X[0] = 1

    for x_idx in range(data_dim - 1):
        xi = data[:, x_idx]
        for x_dim in range(1, n_dim + 1):
            X[x_idx * n_dim + x_dim] = xi**x_dim

    X = X.T

    # wを計算. np.linalg.invは逆行列を計算する関数

    w = (
        np.linalg.inv(
            X.T @ X + regularization_factor * np.identity(1 + n_dim * (data_dim - 1))
        )
        @ X.T
        @ y
    )

    # 回帰式の数式表現
    linename = columns_name[-1] + "=" + str(round(w[0], 2))
    for x_idx in range(data_dim - 1):
        x_str = columns_name[x_idx]
        for x_dim in range(1, n_dim + 1):
            linename += "+{0}*{1}^{2}".format(
                round(w[x_idx * n_dim + x_dim], 2), x_str, x_dim
            )

    # 回帰式のプロット用配列を作成
    N_PLOT = 100
    data_d = None

    # 各軸毎に100点をプロット
    for x_idx in range(data_dim - 1):
        xi_min = np.min(data[:, x_idx])
        xi_max = np.max(data[:, x_idx])
        xi_data_d = np.arange(N_PLOT) * (xi_max - xi_min) / N_PLOT + xi_min
        xi_data_d = xi_data_d.reshape(N_PLOT, 1)

        if data_d is None:
            data_d = xi_data_d
        else:
            data_d = np.concatenate(
                [
                    np.repeat(data_d, N_PLOT, axis=0),
                    np.tile(xi_data_d, (N_PLOT ** x_idx, 1))
                ],
                axis=1,
            )

    # Xを計算
    X_d = np.zeros(
        (1 + n_dim * (data_dim - 1), N_PLOT ** (data_dim - 1)), dtype=np.float64
    )
    X_d[0] = 1

    for x_idx in range(data_dim - 1):
        xi = data_d[:, x_idx]
        for x_dim in range(1, n_dim + 1):
            X_d[x_idx * n_dim + x_dim] = xi**x_dim

    X_d = X_d.T

    # yを求める
    y_d = X_d @ w
    y_d = y_d.reshape(len(y_d), 1)

    # data_dに結合
    linedata = np.concatenate([data_d, y_d], axis=1)

    return linedata, linename


def main():
    """main関数.

    -> None
    """
    # コマンドライン引数を取得
    filename, n_dim, regularization_factor = get_args()

    # コマンドライン引数を取得できなければ終了
    if filename is None:
        return

    # csvファイルを読み取り
    data, columns_name = csv2ndarray(filename)

    # 散布図を描画
    plot_dispersal_chart(data, filename, columns_name)

    # 回帰式を求める
    linedata, linename = calc_regression(
        data, n_dim, regularization_factor, columns_name
    )

    # 回帰式を描画
    plot_dispersal_chart(
        data, filename, columns_name, linedata=linedata, linename=linename
    )

    return


if __name__ == "__main__":
    main()