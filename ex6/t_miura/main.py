"""隠れマルコフモデルによるモデルの予測を行う.

argparse   : コマンドライン引数の取得
datetime   : 計算時間の計測
pickle     : pickleファイルの読み込み
matplotlib : 混同行列のプロット
numpy      : 行列計算
"""

import argparse
import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np


def get_args():
    """コマンドライン引数の取得.

    -> ファイル名: str
    """
    # コマンドライン引数を取得
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="pickle filename")
    args = parser.parse_args()
    filename = args.filename

    if filename[-7:] != ".pickle":
        raise (ValueError("filename must be pickle file"))

    return filename


class HMM:
    """隠れマルコフモデルによるモデル予測."""

    def __init__(self, filename: str):
        """コンストラクタ.

        -> None
        """
        self.filename = filename
        self.answer_models = None
        self.output = None
        self.PI = None
        self.A = None
        self.AT = None
        self.B = None
        self.n_sample = None
        self.n_models = None
        self.n_state = None
        self.n_output = None
        self.last_time = None
        self.result_forward = None
        self.result_viterbi = None

        return

    def load_picklefile(self):
        """pickleファイルの読み込み.

        -> None
        """
        # pickleファイルの読み込み
        data = pickle.load(open(f"../{self.filename}", "rb"))

        # データの分解
        self.answer_models = np.array(data["answer_models"])
        self.output = np.array(data["output"])
        self.PI = np.array(data["models"]["PI"])
        self.A = np.array(data["models"]["A"])
        self.B = np.array(data["models"]["B"])

        # 各データサイズの確認
        self.n_sample, self.last_time = self.output.shape
        self.n_models, self.n_state, self.n_output = self.B.shape

        return

    def forward_algorithm(self, p: int):
        """forwardアルゴリズムによる各モデルのP(0|M)の計算.

        -> 各モデルのP(0|M): np.ndarray
        """
        # データの読み込み確認
        if self.PI is None:
            self.load_picklefile()

        # Aを転置
        AT = self.A.transpose(0, 2, 1)

        # 初期条件
        alpha = self.PI * self.B[:, :, self.output[p, 0]: self.output[p, 0] + 1]

        # 漸化式
        for s in range(1, self.last_time):
            alpha = (AT @ alpha) * self.B[:, :, self.output[p, s]: self.output[p, s] + 1]

        # 各モデルのP(0|M)を算出
        p_0_m = np.sum(alpha, axis=1).flatten()

        return p_0_m

    def viterbi_algorithm(self, p: int):
        """viterbiアルゴリズムによるP(M|0)の計算.

        -> 各モデルのP(M|0): np.ndarray
        """
        # データの読み込み確認
        if self.PI is None:
            self.load_picklefile()

        # 初期条件
        delta = self.PI * self.B[:, :, self.output[p, 0]: self.output[p, 0] + 1]

        # 漸化式
        for s in range(1,self.last_time):
            delta = (
                np.max(delta * self.A, axis=1).reshape((self.n_models, self.n_state, 1))
                * self.B[:, :, self.output[p, s]: self.output[p, s] + 1]
            )

        # 各モデルのP(M|0)を算出
        p_m_0 = np.max(delta, axis=1).flatten()

        return p_m_0

    def predict_model(self):
        """モデルの予測.

        -> forwardアルゴリズムの計算時間: datetime.timedelta, viterbiアルゴリズムの計算時間: datetime.timedelta
        """
        # データの読み込み確認
        if self.n_sample is None:
            self.load_picklefile()

        # 結果を記録する配列を定義
        self.result_forward = np.full(self.n_sample, None)
        self.result_viterbi = np.full(self.n_sample, None)

        # 各サンプルに対して両アルゴリズムを実行
        # forwardアルゴリズム
        start = datetime.datetime.now()
        for p in range(self.n_sample):
            p_0_m = self.forward_algorithm(p)
            self.result_forward[p] = np.argmax(p_0_m)
        time_forward = datetime.datetime.now() - start

        # viterbiアルゴリズム
        start = datetime.datetime.now()
        for p in range(self.n_sample):
            p_m_0 = self.viterbi_algorithm(p)
            self.result_viterbi[p] = np.argmax(p_m_0)
        time_viterbi = datetime.datetime.now() - start

        return time_forward, time_viterbi


    def plot_prediction(self):
        """２つのモデルの予想の混同行列と正解率をプロット.

        -> None
        """
        # 混同行列を計算
        confusion_matrix_forward = np.zeros((self.n_models, self.n_models))
        confusion_matrix_viterbi = np.zeros((self.n_models, self.n_models))

        for p in range(self.n_sample):
            confusion_matrix_forward[self.answer_models[p], self.result_forward[p]] += 1
            confusion_matrix_viterbi[self.answer_models[p], self.result_viterbi[p]] += 1

        # 正解率を計算
        accuracy_forward = np.trace(confusion_matrix_forward) / self.n_sample
        accuracy_viterbi = np.trace(confusion_matrix_viterbi) / self.n_sample

        # 混同行列と正解率をプロット
        # forward
        plt.subplot(121)
        plt.imshow(confusion_matrix_forward, cmap="Greys")
        value_max = np.max(confusion_matrix_forward)
        for x in range(self.n_models):
            for y in range(self.n_models):
                if confusion_matrix_forward[x, y] <= value_max / 2:
                    plt.text(
                        y,
                        x,
                        int(confusion_matrix_forward[x, y]),
                        horizontalalignment='center',
                        verticalalignment='center',
                    )
                else:
                    plt.text(
                        y,
                        x,
                        int(confusion_matrix_forward[x, y]),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color="w",
                    )

        plt.title(f"Forward Algorithm\nAccurancy={accuracy_forward}")
        plt.xlabel("Predicted Model")
        plt.ylabel("Answer Model")

        # viterbi
        plt.subplot(122)
        plt.imshow(confusion_matrix_viterbi, cmap="Greys")
        value_max = np.max(confusion_matrix_viterbi)
        for x in range(self.n_models):
            for y in range(self.n_models):
                if confusion_matrix_viterbi[x, y] <= value_max / 2:
                    plt.text(
                        y,
                        x,
                        int(confusion_matrix_viterbi[x, y]),
                        horizontalalignment='center',
                        verticalalignment='center',
                    )
                else:
                    plt.text(
                        y,
                        x,
                        int(confusion_matrix_viterbi[x, y]),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color="w"
                    )

        plt.title(f"Viterbi Algorithm\nAccurancy={accuracy_viterbi}")
        plt.xlabel("Predicted Model")
        plt.ylabel("Answer Model")

        # プロットの保存と表示
        plt.savefig(self.filename.replace(".pickle", ".png"))
        plt.show()

        return


def main():
    """main関数.

    -> None
    """

    # コマンドライン引数の取得
    filename = get_args()

    # hmmインスタンスの生成
    hmm = HMM(filename)

    # ファイルの読み込み
    hmm.load_picklefile()

    # モデルの予測
    time_forward, time_viterbi = hmm.predict_model()

    # アルゴリズム毎の計算時間を表示
    print(f"forward : {time_forward.seconds}.{time_forward.microseconds} [s]")
    print(f"viterbi : {time_viterbi.seconds}.{time_viterbi.microseconds} [s]")

    # 混同行列と正解率をプロット
    hmm.plot_prediction()

    return


if __name__ == "__main__":
    main()
