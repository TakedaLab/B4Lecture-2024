"""Do PCA(Principal Component Analysis)."""

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class clsPCA:
    """Do PCA from csv file."""

    def __init__(self, str_csv_path: str, is_csv_header: bool = False) -> None:
        """Do PCA from csv file.

        Args:
            str_csv_path (str): csv file with only float data.
        """
        try:
            self.df_raw: pl.DataFrame = (
                # csv読み込み
                pl.read_csv(str_csv_path, has_header=is_csv_header, encoding="utf8")
            )
        except Exception as e:
            raise ValueError(
                f"""
        csv file must with only float data.
        input: {str_csv_path}
        error: {e}"""
            )

        # 標準化
        self.df_norm: pl.DataFrame = self.df_raw.select(
            (pl.all() - pl.all().mean()) / pl.all().std()
        )

        # numpyにする．
        self.np_norm: np.ndarray = self.df_norm.to_numpy().T
        # 共分散行列を求める．
        self.np_S: np.ndarray = np.cov(self.np_norm)

        # 固有値と固有ベクトルを求める．
        self.np_eig_val: np.ndarray
        self.np_eig_vec: np.ndarray
        self.np_eig_val, self.np_eig_vec = np.linalg.eig(self.np_S)
        # 固有値を昇順にする．
        np_sorted_indices: np.ndarray = np.argsort(-self.np_eig_val)
        self.np_eig_val = self.np_eig_val[np_sorted_indices]
        self.np_eig_vec = self.np_eig_vec[:, np_sorted_indices]

        # PCA
        self.np_A: np.ndarray = self.np_eig_vec
        self.np_Y: np.ndarray = self.np_A.T @ self.np_norm

        # 寄与率
        self.np_contributions: np.ndarray = self.np_eig_val / np.sum(self.np_eig_val)
        self.np_cumulative_contributions: np.ndarray = np.cumsum(self.np_contributions)


if __name__ == "__main__":
    # _pca = clsPCA("pre-data3.csv", is_csv_header=True)
    pca1 = clsPCA("../data1.csv")
    pca2 = clsPCA("../data2.csv")
    pca3 = clsPCA("../data3.csv")

    def plot(is_kadai1: bool) -> None:
        """Plot."""
        str_kadai: str
        if is_kadai1:
            str_kadai = "kadai1"
        else:
            str_kadai = "kadai2"

        os.makedirs(f"./{str_kadai}", exist_ok=True)
        # 2次元プロット
        df1: pl.DataFrame = pca1.df_raw
        fig1: plt.Figure
        ax1: plt.Axes
        fig1, ax1 = plt.subplots()
        ax1.scatter(df1["column_1"], df1["column_2"])
        if not is_kadai1:
            for i in range(len(pca1.np_A)):
                ax1.quiver(0, 0, pca1.np_A[i, 0], pca1.np_A[i, 1], color="r")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        title1: str = f"{str_kadai}/data1"
        ax1.set_title(title1)
        fig1.savefig(
            f"{title1}.png", dpi=300, bbox_inches="tight"
        )  # 画像を余白ギリギリで保存．

        # 3次元プロット
        df2: pl.DataFrame = pca2.df_raw
        fig2: plt.Figure = plt.figure()
        ax2: plt.Axes3D = fig2.add_subplot(111, projection="3d")
        ax2.scatter(df2["column_1"], df2["column_2"], df2["column_3"])
        for i in range(len(pca2.np_A)):
            ax2.quiver(
                0, 0, 0, pca2.np_A[i, 0], pca2.np_A[i, 1], pca2.np_A[i, 2], color="r"
            )

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        title2: str = f"{str_kadai}/data2/"
        os.makedirs(title2, exist_ok=True)
        for elev in range(30, 30 + 1):  # zからの仰角
            for azim in range(0, 360, 30):  # x-yの角度
                title2_angle: str = os.path.join(title2, f"{elev}-{azim}")
                ax2.view_init(elev, azim)  # 角度設定
                ax2.set_title(title2_angle)
                fig2.savefig(f"{title2_angle}.png", dpi=300, bbox_inches="tight")

        # plt.show()

    def kadai1() -> None:
        """Scatter Plot."""
        plot(True)
        return

    def kadai2() -> None:
        """Calculate Contribution and Plot."""
        print(f"---data1.csvの寄与率---\n{pca1.np_contributions}\n")
        print(f"---data2.csvの寄与率---\n{pca2.np_contributions}\n")
        print(f"---data3.csvの寄与率---\n{pca3.np_contributions}\n")
        plot(False)
        return

    def kadai3() -> None:
        """Dimention Compression."""
        # data3において，累積寄与率を90%以上とする．
        int_pca_num: int = np.argmax(pca3.np_cumulative_contributions >= 0.9) + 1
        print("---data3.csvの累積寄与率を90%以上とした時---")
        print(f"採用する主成分の数: {int_pca_num}")
        print(f"圧縮できる次元の数: {pca3.np_norm.shape[0]-int_pca_num}")

        # data2の2次元プロット
        os.makedirs("./kadai3", exist_ok=True)
        df: pl.DataFrame = pca2.df_raw
        fig: plt.Figure
        ax: plt.Axes
        for x, y in itertools.combinations(range(3), 2):
            fig, ax = plt.subplots()
            ax.scatter(df[f"column_{x+1}"], df[f"column_{y+1}"])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            title: str = f"kadai3/{x}-{y}"
            ax.set_title(title)
            fig.savefig(f"{title}.png", dpi=300, bbox_inches="tight")

    kadai1()
    kadai2()
    kadai3()
