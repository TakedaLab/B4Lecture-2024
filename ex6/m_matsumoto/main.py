"""Do HMM."""

import pickle
from typing import Callable

import numpy as np


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
        except Exception as e:
            raise NameError(
                f"cannot load data from {str_pickle_path} with the Error: \n {e}"
            )

    def forward(self) -> np.ndarray:
        """Predict model with the method "forword".

        Returns:
            np.ndarray: predict model sequence
        """
        return self._predict(np.sum)
        p, t = self.output.shape
        predicted_model_sequence: np.ndarray = np.zeros(p)
        for i in range(p):
            alpha: list = [np.zeros_like(self.PI) for _ in range(t)]
            alpha[0] = self.PI[:, :, 0] * self.B[:, :, self.output[i, 0]]
            for j in range(t - 1):
                alpha[j + 1] = (
                    np.sum(
                        np.tile(alpha[j][:, :, np.newaxis], (1, 1, self.A.shape[2]))
                        * self.A,
                        axis=1,
                    )
                    * self.B[:, :, self.output[i, j + 1]]
                )
            P: np.ndarray = np.sum(alpha[t - 1], axis=1)
            predicted_model_sequence[i] = np.argmax(P)

        return predicted_model_sequence

    def viterbi(self) -> np.ndarray:
        """Predict model with the method "viterbi".

        Returns:
            np.ndarray: predict model sequence
        """
        return self._predict(np.max)
        p, t = self.output.shape
        predicted_model_sequence: np.ndarray = np.zeros(p)
        for i in range(p):
            alpha: list = [np.zeros_like(self.PI) for _ in range(t)]
            alpha[0] = self.PI[:, :, 0] * self.B[:, :, self.output[i, 0]]
            for j in range(t - 1):
                alpha[j + 1] = (
                    np.max(
                        np.tile(alpha[j][:, :, np.newaxis], (1, 1, self.A.shape[2]))
                        * self.A,
                        axis=1,
                    )
                    * self.B[:, :, self.output[i, j + 1]]
                )
            P: np.ndarray = np.max(alpha[t - 1], axis=1)
            predicted_model_sequence[i] = np.argmax(P)
        return predicted_model_sequence

    def _predict(
        self, sigma: Callable[[np.ndarray, int or tuple or None], np.ndarray]
    ) -> np.ndarray:
        """Predict.

        Args:
            sigma (Callable): np.sum or np.max

        Returns:
            np.ndarray: predict model sequence
        """
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
        return predicted_model_sequence


if __name__ == "__main__":

    hmm1 = clsHMM("../data1.pickle")
    a = hmm1.forward()
    b = hmm1.viterbi()
    print(a)
    print(b)
    hmm2 = clsHMM("../data2.pickle")
    hmm3 = clsHMM("../data3.pickle")
    hmm4 = clsHMM("../data4.pickle")
