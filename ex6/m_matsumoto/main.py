"""Do HMM."""

import os
import pickle
import time
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


class clsHMM:
    """Do HMM."""

    KEY_ANSWER_MODELS: str = "answer_models"
    KEY_OUTPUT: str = "output"
    KEY_MODELS: str = "models"
    KEY_A: str = "A"
    KEY_B: str = "B"
    KEY_PI: str = "PI"

    def __init__(self, str_pickle_path: str) -> None:
        """Do Initialize.

        Args:
            str_pickle_path (str): pickle file path
        """
        try:
            data: dict = pickle.load(open(str_pickle_path, "rb"))
            self.answer_model: np.ndarray = np.array(data[self.KEY_ANSWER_MODELS])
            self.output: np.ndarray = np.array(data[self.KEY_OUTPUT])
            self.A: np.ndarray = np.array(data[self.KEY_MODELS][self.KEY_A])
            self.B: np.ndarray = np.array(data[self.KEY_MODELS][self.KEY_B])
            self.PI: np.ndarray = np.array(data[self.KEY_MODELS][self.KEY_PI])
            print(f"clsHMM Initialized with the file: {str_pickle_path}")
        except Exception as e:
            raise NameError(
                f"cannot load data from {str_pickle_path} with the Error: \n {e}"
            )

    def forward(self) -> Tuple[np.ndarray, float]:
        """Predict model with the method "forword".

        Returns:
            np.ndarray: predict model sequence
            float: execution time(ms)
        """
        print("Start forward")
        return self._predict(np.sum)

    def viterbi(self) -> Tuple[np.ndarray, float]:
        """Predict model with the method "viterbi".

        Returns:
            np.ndarray: predict model sequence
            float: execution time(ms)
        """
        print("Start viterbi")
        return self._predict(np.max)

    def _predict(
        self,
        sigma: Callable[[np.ndarray, Optional[Union[int, tuple, None]]], np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        st: float = time.time()
        p, t = self.output.shape
        predicted_model_sequence: np.ndarray = np.zeros(p)
        for i in range(p):
            alpha: list = [np.zeros_like(self.PI) for _ in range(t)]
            alpha[0] = self.PI[:, :, 0] * self.B[:, :, self.output[i, 0]]
            for j in range(t - 1):
                alpha[j + 1] = (
                    sigma(
                        np.tile(alpha[j][:, :, np.newaxis], (1, 1, self.A.shape[2]))
                        * self.A,
                        axis=1,
                    )
                    * self.B[:, :, self.output[i, j + 1]]
                )
            P: np.ndarray = sigma(alpha[t - 1], axis=1)
            predicted_model_sequence[i] = np.argmax(P)
        ed: float = time.time()
        flt_execution_time: float = ed - st
        print(f"Execution time: {flt_execution_time}")
        return predicted_model_sequence, flt_execution_time


def display_hmm(str_pickle_path: str) -> None:
    """Do HMM and write result to png file.

    Args:
        str_pickle_path (str): pickle file path
    """
    # Do HMM
    hmm = clsHMM(str_pickle_path)
    # forward
    np_forward_predict, flt_forward_time = hmm.forward()
    # viterbi
    np_viterbi_predict, flt_viterbi_time = hmm.viterbi()

    # display results
    plt.rcParams["figure.figsize"] = (13, 6)
    fig = plt.figure()
    # forward
    plt.subplot(121)
    acc_forward = 100 * accuracy_score(hmm.answer_model, np_forward_predict)
    cm = confusion_matrix(hmm.answer_model, np_forward_predict)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")
    plt.title(
        f"Forward algorithm\n(Acc. {acc_forward:.0f}%)\n(Time: {flt_forward_time:.3f}ms)"
    )
    # viterbi
    plt.subplot(122)
    acc_viterbi = 100 * accuracy_score(hmm.answer_model, np_viterbi_predict)
    cm = confusion_matrix(hmm.answer_model, np_viterbi_predict)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")
    plt.title(
        f"Viterbi algorithm\n(Acc. {acc_viterbi:.0f}%)\n(Time: {flt_viterbi_time:.3f}ms)"
    )

    # writing file
    str_output_file: str = (
        f"{os.path.splitext(os.path.basename(str_pickle_path))[0]}.png"
    )
    str_output_folder: str = "fig"
    os.makedirs(str_output_folder, exist_ok=True)
    fig.savefig(os.path.join(str_output_folder, str_output_file))


if __name__ == "__main__":

    display_hmm("../data1.pickle")
    display_hmm("../data2.pickle")
    display_hmm("../data3.pickle")
    display_hmm("../data4.pickle")
