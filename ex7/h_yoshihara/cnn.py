#!/usr/bin/env python
# -*- coding: utf-8 -*-


r"""
CNNを用いて音声データの分類を行うプログラム.

特徴量；MFCCの平均（0次項含まず）
識別器；CNN
実行コマンド
python .\baseline_cnn.py --path_to_truth ..\test_truth.csv
0.95
0.9533333333333334
0.9466666666666667
0.96
0.9633333333333334
"""

from __future__ import division, print_function

import argparse
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def my_CNN(input_dim, output_dim):
    """
    CNNモデルの構築.

    Args:
        shape: 入力の形
        output_dim: 出力次元
    Returns:
        model: 定義済みモデル
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 1), padding="same", input_shape=input_dim))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation("softmax"))

    model.summary()

    return model


def load_data(path, root):
    """
    指定されたパスから音声データを取り出す関数.

    flake8対策。
    Args:
        path: 読み込むファイルのパス
        root: ルートディレクトリ
    Returns:
        y: 音声データ
    """
    return librosa.load(os.path.join(root, path))[0]


def feature_extraction(path_list):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す.

    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """
    data = [load_data(path, root) for path in path_list]
    features = np.array(
        [np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data]
    )

    return features


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.Blues):
    """
    予測結果の混合行列をプロット.

    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    Returns:
        Nothing
    """
    predicted_value = accuracy_score(ground_truth, predict)
    cm = confusion_matrix(predict, ground_truth)

    sums = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / sums

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111)
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(f"CNN Result\nAcc. {predicted_value * 100:.2f}%")
    ax.set_ylabel("Predicted")
    ax.set_xlabel("Ground truth")
    num_classes = len(np.unique(ground_truth))  # ground_truthに含まれるクラスの数
    ax.set_xticks(np.arange(num_classes) + 0.5)
    ax.set_yticks(np.arange(num_classes) + 0.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    fig.savefig("result.png")
    fig.show()


def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する.

    Args:
        paths: テストする音声ファイルリスト
        outputs:
    Returns:
        Nothing
    """
    with open("result.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


def main():
    """main関数."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス"
    )
    args = parser.parse_args()

    training_path = os.path.join(root, "training.csv")
    test_path = os.path.join(root, "test_truth.csv")

    print(f"Training file path: {training_path}")
    print(f"Test file path: {test_path}")

    # データの読み込み
    training = pd.read_csv(os.path.join(root, "training.csv"))
    test = pd.read_csv(os.path.join(root, "test_truth.csv"))

    # 学習データの特徴抽出
    X_train = feature_extraction(training["path"].values)
    X_test = feature_extraction(test["path"].values)

    # 正解ラベルをone-hotベクトルに変換
    Y_train = np_utils.to_categorical(y=training["label"], num_classes=10)

    # 学習データを学習データとバリデーションデータに分割
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
        random_state=20200616,
    )

    X_train = X_train[..., np.newaxis, np.newaxis]
    X_test = X_test[..., np.newaxis, np.newaxis]
    X_validation = X_validation[..., np.newaxis, np.newaxis]

    # モデルの構築
    model = my_CNN((X_train.shape[1], 1, 1), output_dim=10)

    # モデルの学習基準の設定
    model.compile(
        loss="categorical_crossentropy", optimizer=SGD(lr=0.002), metrics=["accuracy"]
    )

    # モデルの学習
    model.fit(X_train, Y_train, batch_size=32, epochs=500, verbose=1)

    # モデル構成，学習した重みの保存
    model.save(os.path.join(root, "keras_model/my_model.h5"))

    # バリデーションセットによるモデルの評価
    score = model.evaluate(X_validation, Y_validation, verbose=0)
    print("Validation accuracy: ", score[1])

    # 予測結果
    predict = model.predict(X_test)
    predicted_values = np.argmax(predict, axis=1)

    # テストデータに対して推論した結果の保存
    write_result(test["path"].values, predicted_values)

    # テストデータに対する正解ファイルが指定されていれば評価を行う
    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth["label"].values
        plot_confusion_matrix(predicted_values, truth_values)
        print("Test accuracy: ", accuracy_score(truth_values, predicted_values))


if __name__ == "__main__":
    main()
