#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""メルスペクトログラムに対してCNNを適用してクラスタリングを行うモデルの学習を行うプログラム."""

from __future__ import division, print_function

import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


def my_CNN(input_shape, output_dim):
    """CNNモデルの構築.

    Args:
        input_shape: 入力の形
        output_dim: 出力次元
    Returns:
        model: 定義済みモデル
    """
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(input_shape[0], input_shape[0], 1),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5)),
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5)),
    model.add(Dense(output_dim, activation="softmax"))
    model.summary()  # モデル構成の表示
    return model


def load_data(path):
    """ファイルへのパスからデータをロードする関数."""
    return librosa.load(path)[0]


def feature_extraction(path_list):
    """wavファイルのリストから特徴抽出を行い，リストで返す.

    扱う特徴量はメルスペクトログラム
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """
    data = list(map(load_data, path_list))
    features = []
    for y in data:
        mel_spectrogram = librosa.feature.melspectrogram(y=y)  # メルスペクトログラム
        mel_spectrogram_dB = librosa.power_to_db(
            mel_spectrogram, ref=np.max
        )  # デシベル単位に変換
        mel_spectrogram_dB_resize = Image.fromarray(mel_spectrogram_dB).resize(
            (128, 128)
        )  # 時間方向に拡大して正方形画像とする
        mel_spectrogram_dB_square = np.asarray(
            mel_spectrogram_dB_resize
        )  # np.ndarrayに戻す
        features.append(mel_spectrogram_dB_square)  # featuresにappend
    features = np.stack(features)  # shape=(sample, 128, 128)になる
    return features


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.Blues):
    """予測結果の混合行列をプロット.

    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    Returns:
        Nothing
    """
    cm = confusion_matrix(ground_truth, predict)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel("Ground truth")
    plt.xlabel("Predicted")
    plt.savefig("result.png")


def write_result(paths, outputs):
    """結果をcsvファイルで保存する.

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
    """モデルの学習と保存を行うプログラム."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス"
    )
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv("training.csv")
    test = pd.read_csv("test.csv")

    # 学習データの特徴抽出
    X_train = feature_extraction(training["path"].values)
    X_test = feature_extraction(test["path"].values)

    # 正解ラベルをone-hotベクトルに変換
    Y_train = to_categorical(training["label"], num_classes=10)

    # 学習データを学習データとバリデーションデータに分割
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=20200616
    )

    # モデルの構築
    model = my_CNN(input_shape=X_train.shape[1:], output_dim=10)

    # モデルの学習基準の設定
    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(learning_rate=0.002),
        metrics=["accuracy"],
    )

    # モデルの学習
    model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1)

    # モデル構成，学習した重みの保存
    model.save("keras_model/my_model.h5")

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


def summary():
    """学習済みモデルからレポート作成用データのグラフや不正解ラベルを出力."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス"
    )
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv("training.csv")
    test = pd.read_csv("test.csv")

    # 学習データの特徴抽出
    X_train = feature_extraction(training["path"].values)
    X_test = feature_extraction(test["path"].values)

    # 正解ラベルをone-hotベクトルに変換
    Y_train = to_categorical(training["label"], num_classes=10)

    # 学習データを学習データとバリデーションデータに分割
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=20200616
    )

    # 学習済みモデルの読み込み
    model = load_model("keras_model/my_model.h5")

    # 予測結果
    predict = model.predict(X_test)
    predicted_values = np.argmax(predict, axis=1)

    # テストデータに対して推論した結果の保存
    write_result(test["path"].values, predicted_values)

    # テストデータに対する正解ファイルが指定されていれば評価を行う
    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth["label"].values
        print(test["path"].values[predicted_values != truth_values])
        plot_confusion_matrix(predicted_values, truth_values)
        print("Test accuracy: ", accuracy_score(truth_values, predicted_values))


if __name__ == "__main__":
    summary()
    # main()
