#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識.
ベースラインスクリプト: Pytorch Lightning版
特徴量: MFCCの平均*5
識別器: MLP
"""
import argparse
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchaudio
import torchmetrics
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, random_split

# root = ../ex7
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class my_MLP(pl.LightningModule):
    """MLPによるパターン認識モデル."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """モデルの初期化を行う.

        Args:
            input_dim (int): モデルの入力次元数.
            output_dim (int): モデルの出力次元数.
        """
        super().__init__()
        self.model = self.create_model(input_dim, output_dim)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.confm = torchmetrics.ConfusionMatrix(10, normalize="true")
        self.validation_step_outputs = []  # validationデータの認識結果
        self.test_step_outputs = []  # testデータの認識結果

    def create_model(self, input_dim: int, output_dim: int) -> torch.nn.Sequential:
        """MLPモデルを作成する.

        Args:
            input_dim (int): モデルの入力次元数.
            output_dim (int): モデルの出力次元数.

        Returns:
            torch.nn.Sequential: 定義済みモデル.
        """
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, output_dim),
            torch.nn.Softmax(dim=-1),
        )
        # モデル構成の表示
        print(model)
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/acc",
            self.train_acc(pred, y),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("val/loss", loss, prog_bar=True, logger=True)
        self.log("val/acc", self.val_acc(pred, y), prog_bar=True, logger=True)
        self.validation_step_outputs.append(
            {"pred": torch.argmax(pred, dim=-1), "target": y}
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("test/loss", loss, prog_bar=True, logger=True)
        self.log("test/acc", self.test_acc(pred, y), prog_bar=True, logger=True)
        self.test_step_outputs.append({"pred": torch.argmax(pred, dim=-1), "target": y})
        return {"pred": torch.argmax(pred, dim=-1), "target": y}

    def validation_epoch_end(self, outputs) -> None:
        # validationデータの混同行列を tensorboard に出力
        preds = torch.cat([tmp["pred"] for tmp in self.validation_step_outputs])
        targets = torch.cat([tmp["target"] for tmp in self.validation_step_outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(
            confusion_matrix.cpu().numpy(), index=range(10), columns=range(10)
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="gray_r").get_figure()
        plt.rcParams["font.size"] = 14
        plt.title(f"Acc. = {self.val_acc.compute():.4f}", fontsize=20)
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Ground truth", fontsize=20)
        plt.gca().spines["right"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        # plt.savefig(os.path.join(root, "h_miyaji", "figs", "result_validation.png"))
        plt.close(fig_)
        self.logger.experiment.add_figure(
            "Confusion matrix (val)", fig_, self.current_epoch
        )
        self.validation_step_outputs.clear()
        self.val_acc.reset()

    def test_epoch_end(self, outputs) -> None:
        # testデータの混同行列を tensorboard に出力
        preds = torch.cat([tmp["pred"] for tmp in self.test_step_outputs])
        targets = torch.cat([tmp["target"] for tmp in self.test_step_outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(
            confusion_matrix.cpu().numpy(), index=range(10), columns=range(10)
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Blues").get_figure()
        plt.rcParams["font.size"] = 14
        plt.title(f"Acc. = {self.test_acc.compute():.4f}", fontsize=20)
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Ground truth", fontsize=20)
        plt.gca().spines["right"].set_visible(True)
        plt.gca().spines["left"].set_visible(True)
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["bottom"].set_visible(True)
        plt.savefig(os.path.join(root, "h_miyaji", "figs", "result_test.png"))
        plt.close(fig_)
        self.logger.experiment.add_figure(
            "Confusion matrix (test)", fig_, self.current_epoch
        )

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.002)
        return self.optimizer


class FSDD(Dataset):
    """認識に使用するデータセット."""

    def __init__(self, path_list: list, label: list) -> None:
        """データセットの初期化を行う.

        Args:
            path_list (list): データのパスリスト.
            label (list): データの正解ラベル.
        """
        super().__init__()
        self.features = self.feature_extraction(path_list)
        self.label = label

    def feature_extraction(self, path_list: list) -> torch.Tensor:
        """wavファイルのリストから特徴抽出を行う.

        Args:
            path_list (list): 特徴抽出するファイルのパスリスト.

        Returns:
            torch.Tensor: 特徴量(MFCC平均).
        """
        n_mfcc = 13  # MFCC13次元
        datasize = len(path_list)  # ファイルパスの個数

        # 特徴量を保存する配列(datasize, MFCC次元数*5)
        features = torch.zeros(datasize, (n_mfcc - 1) * 5)

        # waveform -> MFCC
        transform = torchaudio.transforms.MFCC(
            n_mfcc=13, melkwargs={"n_mels": 64, "n_fft": 512}
        )

        for i, path in enumerate(path_list):
            # data.shape==(channel,time)
            data, _ = torchaudio.load(os.path.join(root, path))
            mfcc = transform(data[0])[1:, :]
            mean_all = torch.mean(mfcc, axis=1)

            # MFCCの時間方向の次元が7未満の場合, パディングする
            if mfcc.shape[1] < 7:
                pad_width = 7 - mfcc.shape[1]
                mfcc = torch.nn.functional.pad(
                    mfcc, (0, pad_width), mode="constant", value=0.0
                )

            # 時間方向にデータを4分割し, それぞれで平均をとる
            split = [
                int(mfcc.shape[1] * 0.25),
                int(mfcc.shape[1] * 0.5),
                int(mfcc.shape[1] * 0.75),
            ]
            mean_1 = torch.mean(mfcc[:, : split[0]], axis=1)
            mean_2 = torch.mean(mfcc[:, split[0] + 1 : split[1]], axis=1)
            mean_3 = torch.mean(mfcc[:, split[1] + 1 : split[2]], axis=1)
            mean_4 = torch.mean(mfcc[:, split[2] + 1 :], axis=1)
            features[i] = torch.cat((mean_all, mean_1, mean_2, mean_3, mean_4))

        return features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.label[index]


def main():
    """wavファイルに対して認識モデルを作成し, 学習とテストを行う."""

    # 引数取得
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス"
    )
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv(os.path.join(root, "training.csv"))

    # Dataset の作成
    train_dataset = FSDD(training["path"].values, training["label"].values)

    # Train/Validation 分割
    val_size = int(len(train_dataset) * 0.2)  # dataset内の20%をテストデータとする
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], torch.Generator().manual_seed(20200616)
    )

    if args.path_to_truth:  # 正解ファイルが実行時に指定されていたとき
        # Test Dataset の作成
        test = pd.read_csv(args.path_to_truth)
        test_dataset = FSDD(test["path"].values, test["label"].values)
    else:
        test_dataset = None

    # DataModule の作成
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        num_workers=4,
    )

    # モデルの構築 # 出力が10次元（0~9）
    model = my_MLP(input_dim=train_dataset[0][0].shape[0], output_dim=10)

    # 学習の設定
    trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1)

    # モデルの学習
    trainer.fit(model=model, datamodule=datamodule)

    # バリデーション
    trainer.validate(model=model, datamodule=datamodule)

    if args.path_to_truth:
        # テスト
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
