# -*- coding: utf-8 -*-
"""データからモデルの予測を行うプログラム."""

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def forward_algorithm(pi, A, B, output):
    """forwardアルゴリズムを行う関数.

    Args:
        pi : 初期確率
        A : 状態遷移確率行列
        B : 出力確率
        output : 出力系列
    """
    N = A.shape[0]
    T = len(output)

    # Forward確率行列
    alpha = np.zeros((T, N))

    # 初期化 (t = 0)
    alpha[0, :] = pi[:, 0] * B[:, output[0]]

    # 漸化式
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1, :] * A[:, j]) * B[j, output[t]]

    # 観測シーケンスの確率
    log_prob = np.log(np.sum(alpha[T - 1, :]))

    return log_prob


def viterbi_algorithm(pi, A, B, output):
    """viterbiアルゴリズムを行う関数.

    Args:
        pi : 初期確率
        A : 状態遷移確率行列
        B : 出力確率
        output : 出力系列
    """
    N = A.shape[0]
    T = len(output)

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # 初期化
    delta[0, :] = pi[:, 0] * B[:, output[0]]

    # 漸化式
    for t in range(1, T):
        for j in range(N):
            max_val = np.max(delta[t - 1, :] * A[:, j])
            delta[t, j] = max_val * B[j, output[t]]
            psi[t, j] = np.argmax(delta[t - 1, :] * A[:, j])

    return np.max(delta[T - 1, :])


if __name__ == "__main__":
    file_num = 4
    data = pickle.load(open(f"data{file_num}.pickle", "rb"))

    keys = data.keys()
    outputs = data.get("output")
    answer_models = data.get("answer_models")
    model = data.get("models")
    num_models = len(model["PI"])

    # print(data)
    # print(keys)
    # print(output[0].shape)
    # print(model)

    predicted_models_forward = []
    predicted_models_viterbi = []

    # Forward_algorithmの実行
    start_forward = time.perf_counter()
    for output in outputs:
        log_likelihoods = []
        for model_idx in range(num_models):
            pi = model["PI"][model_idx]
            A = model["A"][model_idx]
            B = model["B"][model_idx]

            log_likelihood = forward_algorithm(pi, A, B, output)
            log_likelihoods.append(log_likelihood)

        best_model_forward = np.argmax(log_likelihoods)
        predicted_models_forward.append(best_model_forward)
    end_forward = time.perf_counter()

    # Viterbi_algorithmの実行
    start_viterbi = time.perf_counter()
    for output in outputs:
        max_probs = []
        for model_idx in range(num_models):
            pi = model["PI"][model_idx]
            A = model["A"][model_idx]
            B = model["B"][model_idx]

            max_prob = viterbi_algorithm(pi, A, B, output)
            max_probs.append(max_prob)

        best_model_viterbi = np.argmax(max_probs)
        predicted_models_viterbi.append(best_model_viterbi)
    end_viterbi = time.perf_counter()

    predicted_models_forward = np.array(predicted_models_forward)
    predicted_models_viterbi = np.array(predicted_models_viterbi)

    accuracy_forward = np.mean(predicted_models_forward == answer_models)
    accuracy_viterbi = np.mean(predicted_models_viterbi == answer_models)

    print(f"Forward_algorithm 正解率 : {accuracy_forward * 100:.2f}%")
    print(f"Viterbi_algorithm 正解率 : {accuracy_viterbi * 100:.2f}%")

    print(f"Forward_algorithm 処理時間 : {{:.2f}}".format(end_forward - start_forward))
    print(f"Viterbi_algorithm 処理時間 : {{:.2f}}".format(end_viterbi - start_viterbi))

    fig1 = plt.figure(figsize=(20, 8))

    # Forward_algorithmの表示
    ax1 = fig1.add_subplot(121)
    a = answer_models.tolist()
    p = predicted_models_forward.tolist()
    cm = confusion_matrix(a, p)
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=True, yticklabels=True)
    ax1.set_title(f"Forward algorithm\nAcc. {accuracy_forward * 100:.2f}%")
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_xlabel("Predicted models")
    ax1.set_ylabel("True models")

    # Viterbi_algorithmの表示
    ax2 = fig1.add_subplot(122)
    a = answer_models.tolist()
    p = predicted_models_viterbi.tolist()
    cm = confusion_matrix(a, p)
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=True, yticklabels=True)
    ax2.set_title(f"Viterbi algorithm\nAcc. {accuracy_viterbi * 100:.2f}%")
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.set_xlabel("Predicted models")
    ax2.set_ylabel("True models")

    fig1.savefig(f"cm_{file_num}.png")
