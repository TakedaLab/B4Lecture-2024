"""主成分分析して主成分を見つけ、次元圧縮を行う.

argparse   : コマンドライン引数
bisect     : 二分探索
matplotlib : グラフの描画
numpy      : 行列計算
pandas     : csvの読み込み
"""

import argparse
import bisect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_args():
    """コマンドライン引数の取得.

    -> filename: str, comp_method: str
    """
    # コマンドライン引数を取得
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="csv filename")
    parser.add_argument("--comp_method", default="none", help="none or 2d or 90%")
    args = parser.parse_args()
    filename, comp_method = args.filename, args.comp_method

    if filename[-4:] != ".csv":
        raise (ValueError("filename must be csv file"))
    if comp_method not in {"none", "2d", "90%"}:
        raise (ValueError("comp_method muse be none or 2d or 90%"))

    return filename, comp_method


def csv2ndarray(filename: str):
    """csvファイルを読み取り、ndarrayに変換.

    -> データ: np.ndarray
    """
    # pandasでcsvファイルを読み取り
    df_pd = pd.read_csv(f"../{filename}", header=None)

    # pd.DataFrameをnp.ndarrayに変換
    df_np = np.array(df_pd)

    return df_np


def plot_dispersal_chart(
    data: np.ndarray,
    filename: str,
    columns_name: tuple = ("X1", "X2", "X3"),
    base_vec: np.ndarray = None,
):
    """散布図を描画.

    -> None
    """
    if data.shape[1] == 2:
        # データの散布図をプロット
        plt.scatter(data[:, 0], data[:, 1], label="Observed data")

        # 回帰式を描画
        if base_vec is not None:
            plt.axline(
                (0, 0),
                (base_vec[0, 0], base_vec[1, 0]),
                label="1st component",
                color="orange",
            )
            plt.axline(
                (0, 0),
                (base_vec[0, 1], base_vec[1, 1]),
                label="2nd component",
                color="green",
            )

            # ラベルを設定
            plt.title(filename.replace(".csv", "_bace"))
        else:
            plt.title(filename.replace(".csv", ""))
        plt.xlabel(columns_name[0])
        plt.ylabel(columns_name[1])
        plt.legend()  # 凡例の表示
        # 散布図の保存と描画
        if base_vec is None:
            temp = ".png"
        else:
            temp = "_bace.png"
        plt.gca().set_aspect("equal")  # アスペクト比の設定
        plt.savefig(filename.replace(".csv", temp))  # 画像の保存
        plt.show()  # 描画
    elif data.shape[1] == 3:
        # 3dプロットの準備
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        # データの散布図をプロット
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], label="Observed data")

        # データの最大最小値を計算
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)

        # 基底を描画
        if base_vec is not None:
            label = ["1st", "2nd", "3rd"]
            color = ["orange", "green", "red"]
            for i in range(3):
                # 各軸毎に媒介変数の範囲を決定
                vec_i = base_vec[:, i]
                vec_i = np.where(vec_i == 0, 1e-15, vec_i)  # zero dividの回避
                param1 = data_min / vec_i
                param2 = data_max / vec_i
                param_higher = min([max(param1[j], param2[j]) for j in range(3)])
                param_lower = max([min(param1[j], param2[j]) for j in range(3)])
                ax.plot(
                    (vec_i[0] * param_lower, vec_i[0] * param_higher),
                    (vec_i[1] * param_lower, vec_i[1] * param_higher),
                    (vec_i[2] * param_lower, vec_i[2] * param_higher),
                    label=label[i] + " component",
                    color=color[i],
                )
            # ラベルを設定
            ax.set_title(filename.replace(".csv", "_bace"))
        else:
            ax.set_title(filename.replace(".csv", ""))
        ax.set_xlabel(columns_name[0])
        ax.set_ylabel(columns_name[1])
        ax.set_zlabel(columns_name[2])
        ax.legend()
        # 散布図の保存と描画
        if base_vec is None:
            temp = ".png"
        else:
            temp = "_bace.png"
        ax.set_box_aspect(
            (
                data_max[0] - data_min[0],
                data_max[1] - data_min[1],
                data_max[2] - data_min[2],
            )
        )
        plt.savefig(filename.replace(".csv", temp))  # 画像の保存
        plt.show()  # 描画


def pca(data: np.ndarray, comp_method: str, filename: str):
    """主成分分析と寄与率、圧縮次元数の表示.

    -> 主成分得点: np.ndarray
    """
    n = data.shape[0]  # サンプル数の取得

    # データの標準化
    m = np.mean(data, axis=0)  # 平均を計算
    std = np.std(data, axis=0)  # 標準偏差を計算
    x = (data - m) / std  # 標準化
    x = x.T  # xを転置

    # 共分散行列を求める
    sigma = (x @ x.T) / n

    # 固有値,固有ベクトルを求める
    eigenvalue, a = np.linalg.eig(sigma)

    # 基底のプロット
    plot_dispersal_chart(data, filename, base_vec=a)

    # 寄与率の計算
    contribution_rate = eigenvalue / np.sum(eigenvalue)
    print("Conrtibution Rate =", contribution_rate.tolist())

    # 次元削減
    if comp_method == "none":
        # 次元削減しない
        n_dim = a.shape[1]
    elif comp_method == "2d":
        # 2次元まで削減
        n_dim = 2
    elif comp_method == "90%":
        # 累積寄与率が0.9を超える次元まで削減
        cumsum_cont = np.cumsum(contribution_rate)
        print("Cumsum Conrtibution Rate =", cumsum_cont.tolist())
        n_dim = bisect.bisect(cumsum_cont, 0.9) + 1

    # 次元圧縮
    print(f"Compress to {n_dim} dimentions")
    a = a[:, :n_dim]

    # 主成分得点の計算
    y = a.T @ x

    return y.T


def main():
    """main関数.

    -> None
    """
    # コマンドライン引数の取得
    filename, comp_method = get_args()

    # csvファイルの読み取り
    data_np = csv2ndarray(filename)

    # 散布図のプロット
    plot_dispersal_chart(data_np, filename)

    # 主成分分析
    pca_data = pca(data_np, comp_method, filename)

    # 主成分得点のプロット
    plot_dispersal_chart(
        pca_data, "PCA_" + filename, ("1st component", "2nd component", "3rd component")
    )


if __name__ == "__main__":
    main()
