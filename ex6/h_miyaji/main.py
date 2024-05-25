"""This module makes predictions of HMM."""

import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


class HMM:
    """Define several HMMs and make predictions for the model.

    The Forward and Viterbi algorithm can be performed.
    It can also display the prediction results in a confusion matrix.
    """

    def __init__(self, file_name: str) -> None:
        """Initialize HMMs.

        Args:
            file_name (str): pickle file name.
        """
        # load pickle data
        data = pickle.load(open(file_name, "rb"))

        # set HMMs
        self.answer_models: np.ndarray = np.array(data["answer_models"])  # (p, )
        self.output: np.ndarray = np.array(data["output"])  # (p, t)
        self.PI: np.ndarray = np.array(data["models"]["PI"])  # (k, l, 1)
        self.A: np.ndarray = np.array(data["models"]["A"])  # (k, l, l)
        self.B: np.ndarray = np.array(data["models"]["B"])  # (k, l, n)

        # set constant
        self.OUT_NUM: int = self.output.shape[0]  # p
        self.OUT_LEN: int = self.output.shape[1]  # t
        self.MODEL_NUM: int = self.B.shape[0]  # k
        self.STATE_NUM: int = self.B.shape[1]  # l
        self.SYMBOL_NUM: int = self.B.shape[2]  # n

    def forward_algorithm(self) -> np.ndarray:
        """Make predictions of HMM with the Forward algorithm.

        Returns:
            np.ndarray: the results of HMM predictions (p, ).
        """
        alpha = np.zeros((self.MODEL_NUM, self.STATE_NUM, self.OUT_LEN))  # (k, l, t)
        probability = np.zeros((self.OUT_NUM, self.MODEL_NUM))  # (p, k)
        predicted_model = np.zeros(self.OUT_NUM, dtype=int)  # (p, )

        for p in range(self.OUT_NUM):
            # (k, l, 0) = (k, l, 0) * (k, l, o_0)
            alpha[:, :, 0] = self.PI[:, :, 0] * self.B[:, :, self.output[p, 0]]

            for t in range(self.OUT_LEN - 1):  # t -> 1 ~ t
                # (k, l1, T+1) = np.sum( (k, l0, ax, T)*(k, l0, l1), axis=l0) * (k, l1, out[t+1]) -> (k, l1)
                alpha[:, :, t + 1] = np.sum(
                    alpha[:, :, t][:, :, np.newaxis] * self.A,
                    axis=1,
                )
                alpha[:, :, t + 1] *= self.B[:, :, self.output[p, t + 1]]
            # (P, k) = np.sum(k, l, T-1, axis=l) -> (k, )
            probability[p, :] = np.sum(alpha[:, :, self.OUT_LEN - 1], axis=1)
            predicted_model[p] = np.argmax(probability[p, :])
        return predicted_model

    def viterbi_algorithm(self) -> np.ndarray:
        """Make predictions of HMM with the Viterbi algorithm.

        Returns:
            np.ndarray: the results of HMM predictions (p, ).
        """
        delta = np.zeros((self.MODEL_NUM, self.STATE_NUM, self.OUT_LEN))  # (k, l, t)
        probability = np.zeros((self.OUT_NUM, self.MODEL_NUM))  # (p, k)
        predicted_model = np.zeros(self.OUT_NUM, dtype=int)  # (p, )

        for p in range(self.OUT_NUM):
            # (k, l, 0) = (k, l, 0) * (k, l, o_0)
            delta[:, :, 0] = self.PI[:, :, 0] * self.B[:, :, self.output[p, 0]]

            for t in range(self.OUT_LEN - 1):  # t -> 1 ~ t
                # (k, l1, T+1) = np.max( (k, l0, ax, T)*(k, l0, l1), axis=l0) * (k, l1, out[t+1]) -> (k, l1)
                delta[:, :, t + 1] = np.max(
                    delta[:, :, t][:, :, np.newaxis] * self.A,
                    axis=1,
                )
                delta[:, :, t + 1] *= self.B[:, :, self.output[p, t + 1]]
            # (P, k) = np.max(k, l, T-1, axis=l) -> (k, )
            probability[p, :] = np.max(delta[:, :, self.OUT_LEN - 1], axis=1)
            predicted_model[p] = np.argmax(probability[p, :])
        return predicted_model

    def display_result(
        self, predicted_model: np.ndarray, run_time: float, algorithm_name: str, ax
    ) -> None:
        """Display the prediction results in a confusion matrix.

        Args:
            predicted_model (np.ndarray): the results of HMM predictions (p, ).
            run_time (float): algorithm run time.
            algorithm_name (str): the name of algorithm used for predictions.
            ax (Axes): Axes object indicating where to display the figure.

        Returns:
            None
        """
        cm = confusion_matrix(self.answer_models, predicted_model)
        accuracy = accuracy_score(self.answer_models, predicted_model) * 100
        sns.heatmap(cm, ax=ax, annot=True, cmap="gray_r", cbar=False, square=True)
        ax.set_xlabel("Predicted model")
        ax.set_ylabel("Actual model")
        ax.set_title(
            f"{algorithm_name} algorithm\n(ACC. {accuracy:2.0f}%, time. {run_time:.3f}s)"
        )
        return None


def parse_args() -> argparse.Namespace:
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="makes predictions of HMM.")

    # pickle file name
    parser.add_argument(
        "--input-file", type=str, required=True, help="input pickle file"
    )

    return parser.parse_args()


def main():
    """Load pickle file and makes predictions of HMM."""
    # get argument
    args = parse_args()
    file_name = args.input_file  # pickle file name

    # make an instance of HMM
    hmm = HMM(file_name)

    # forward algorithm
    forward_start = time.time()
    forward_predicted = hmm.forward_algorithm()
    forward_end = time.time()
    forward_time = forward_end - forward_start

    # viterbi algorithm
    viterbi_start = time.time()
    viterbi_predicted = hmm.viterbi_algorithm()
    viterbi_end = time.time()
    viterbi_time = viterbi_end - viterbi_start

    # display confusion matrix
    _, axes = plt.subplots(1, 2, tight_layout=True)
    hmm.display_result(forward_predicted, forward_time, "Forward", axes[0])
    hmm.display_result(viterbi_predicted, viterbi_time, "Viterbi", axes[1])
    plt.savefig(os.path.join("h_miyaji", "figs", f"result_{file_name[2:-7]}.png"))
    plt.show()


if __name__ == "__main__":
    main()
