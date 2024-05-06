"""コマンドライン引数で指定した次数の基底関数で回帰直線するプログラム."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import cartesian


def extend_data(data, dim):
    """コマンドライン引数で指定した累乗で拡張した特徴量を生成する関数.

    Parameters
    -------------
    data : np.ndarray, shape=(the number of sample, the kinds of data)
        特徴量の二次元配列
    dim : int
        各変数の次元数([1,2]ならば1,x^1,y^1,y^2)
    Returns
    ------------
    extended_data : ndarray, shape=(the number of sample, 1 + sum(dim))
        拡張されたデータ
    """
    # 切片用の配列を追加
    extended_data = np.ones((data.shape[0], 1))

    for i in range(len(dim)):
        for j in range(dim[i]):
            # 指定された各次元について累乗を計算
            extended_data = np.concatenate(
                [extended_data, data[:, i].reshape(-1, 1) ** (j + 1)], axis=1
            )

    return extended_data


def calc_regression(data, dim, coeff):
    """正規方程式を解いて重みを計算する関数.

    Parameters
    -------------
    data : np.ndarray, shape=(the number of sample, the kinds of data)
        特徴量と推定するデータの二次元配列
    dim : int
        各変数の次元数([1,2]ならば1,x^1,y^1,y^2)
    coeff : int
        正規化の重み(0なら正規化しない)
    Returns
    ------------
    weight : ndarray, shape=(1 + sum(dim))
        回帰直線の係数
    """
    # dataを特徴量と推定量に分ける
    feature = data[:, :-1]
    target = data[:, -1]

    # 指定された次数に拡張する
    extended_feature = extend_data(feature, dim)

    # 正規化行列を解いて重みを計算
    weight = (
        np.linalg.inv(
            extended_feature.T @ extended_feature
            + coeff * np.eye(extended_feature.shape[1])
        )
        @ extended_feature.T
        @ target
    )

    return weight


def parse_args():
    """コマンドプロントから引数を受け取るための関数."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="Name of input csv file"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        required=True,
        nargs="*",
        help="Upper limit on the order of the basis functions",
    )
    parser.add_argument(
        "--number-of-point",
        type=int,
        default=100,
        help="Number of points on the regression curve",
    )
    parser.add_argument(
        "--coeff", type=float, default=0.1, help="Parameters of ridge regression"
    )
    return parser.parse_args()


def visalize(data, weight, dim, points):
    """実データと回帰直線を表示する関数.

    4次元以上のデータ(特徴量が3つ以上のデータ)においてはプロットしないこととする

    Parameters
    -------------
    data : np.ndarray, shape=(the number of sample, the kinds of data)
        特徴量の二次元配列
    weight : ndarray, shape=(1 + sum(dim))
        回帰直線の係数
    dim : int
        各変数の次元数([1,2]ならば1,x^1,y^1,y^2)
    points : int
        回帰結果の表示点の数
    """
    # 特徴量と推定量に分割
    feature = data[:, :-1]
    target = data[:, -1]

    if len(dim) == 1:
        # プロット用の変数
        x_0 = np.min(feature[:, 0]) + np.arange(0, points, 1) / points * (
            np.max(feature[:, 0]) - np.min(feature[:, 0])
        )
        # 拡張して特徴量を用意
        variable = extend_data(x_0.reshape(-1, 1), dim)
        # 重みから回帰結果の計算
        predict = variable @ weight

        # 結果と散布図の作成
        # 実データの散布図の表示
        plt.scatter(feature[:, 0], target)
        # 回帰結果の表示
        plt.plot(x_0, predict, color="orange")
        # plt.show()

        # 画像化する際にはコメントアウトを外す
        plt.savefig("regression.png")

    elif len(dim) == 2:
        # プロット用の変数
        x_0 = np.min(feature[:, 0]) + np.arange(0, points, 1) / points * (
            np.max(feature[:, 0]) - np.min(feature[:, 0])
        )
        x_1 = np.min(feature[:, 1]) + np.arange(0, points, 1) / points * (
            np.max(feature[:, 1]) - np.min(feature[:, 1])
        )

        # 直積を作成
        product = cartesian((x_0, x_1))
        # 直積を拡張して特徴量を用意
        variable = extend_data(product, dim)
        # 重みから回帰結果の計算
        predict = (variable @ weight).reshape(points, points).T

        # 結果と散布図の作成
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # 散布図の表示
        ax.scatter(feature[:, 0], feature[:, 1], target)
        # 回帰結果の表示
        X_0, X_1 = np.meshgrid(x_0, x_1)
        ax.plot_wireframe(X_0, X_1, predict, color="orange")
        # plt.show()

        # 画像化する際にはコメントアウトを外す
        plt.savefig("regression.png")


if __name__ == "__main__":
    # 引数の受け取り
    args = parse_args()
    dimension = args.dimension
    number_of_point = args.number_of_point
    file_path = args.input_file
    coeff = args.coeff

    # csvファイルの読み込み
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

    # 指定した次数が間違っている場合エラー
    if data.shape[1] != len(dimension) + 1:
        print(
            "[error] The dimension of the argument"
            "and the dimension of the feature are different"
        )
        exit(1)

    # 重みを計算
    weight = calc_regression(data, dimension, coeff)

    # 重みから回帰される格子と実データの散布図を表示
    visalize(data, weight, dimension, number_of_point)
