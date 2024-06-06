#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識を行う.

特徴量; MFCCの平均 (0次項含まず), PCAで次元削減
識別器; MLP
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
from torch.utils.data import DataLoader, Dataset, random_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class my_MLP(pl.LightningModule):
    """MLPモデルを構築するクラス."""

    def __init__(self, input_dim: int, output_dim: int):
        """インスタンス."""
        super().__init__()
        self.model = self.create_model(input_dim, output_dim)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.confm = torchmetrics.ConfusionMatrix(10, normalize="true")
        self.test_step_outputs = []

    def create_model(self, input_dim: int, output_dim: int):
        """モデルを構築する."""
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, output_dim),
            torch.nn.Softmax(dim=-1),
        )

        return model

    def forward(self, x):
        """順伝播."""
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_id=None):
        """トレーニングステップの実装."""
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
        """バリデーションステップの実装."""
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("val/acc", self.val_acc(pred, y), prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_id=None):
        """テストステップの実装."""
        x, y = batch
        pred = self.forward(x)
        self.log("test/acc", self.test_acc(pred, y), prog_bar=True, logger=True)
        self.test_step_outputs.append({"pred": torch.argmax(pred, dim=-1), "target": y})

        return {"pred": torch.argmax(pred, dim=-1), "target": y}

    def on_test_epoch_end(self) -> None:
        """テストエポック終了時の処理."""
        # 混同行列を tensorboard に出力
        preds = torch.cat([tmp["pred"] for tmp in self.test_step_outputs])
        targets = torch.cat([tmp["target"] for tmp in self.test_step_outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(
            confusion_matrix.cpu().numpy(), index=range(10), columns=range(10)
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Blues").get_figure()
        plt.title(f"Acc. = {self.test_acc.compute():.2f}", fontsize=20)
        plt.xlabel("Predicted", fontsize=15)
        plt.ylabel("True", fontsize=15)
        plt.savefig("100result_4.png")
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def configure_optimizers(self):
        """オプティマイザの設定."""
        self.optimizer = torch.optim.Adam(self.model.parameters())

        return self.optimizer


class FSDD(Dataset):
    """音声データからデータセットを作成するクラス."""

    def __init__(self, data, label) -> None:
        """インスタンス."""
        super().__init__()
        self.data = data
        self.label = label

    def feature_extraction(self, data):
        """
        音声データから特徴抽出を行いリストで返す.

        扱う特徴量はMFCC64次元の平均 (0次は含めない)
        """
        transform = torchaudio.transforms.MFCC(
            n_mfcc=64, melkwargs={"n_mels": 64, "n_fft": 512}
        )
        mfcc_features = transform(data)
        # フレームごとに計算されたMFCCを結合して64次元の特徴量を得る
        features = torch.mean(mfcc_features[:, 1:], axis=2)
        features = features.flatten()
        return features

    def __len__(self):
        """len関数."""
        return len(self.data)

    def __getitem__(self, index):
        """getitem関数."""
        features = self.feature_extraction(self.data[index])
        label = torch.tensor(self.label[index])

        return features, label


class FSDDDataModule(pl.LightningDataModule):
    """データモジュールを構築するクラス."""

    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset=None,
        batch_size=32,
        num_workers=4,
    ):
        """インスタンス."""
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        """トレーニングデータローダーを返す."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """バリデーションデータローダーを返す."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """テストデータローダーを返す."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def add_whitenoise(data, noise_factor=0.01):
    """ホワイトノイズを加える."""
    noise = torch.randn_like(data) * noise_factor
    noise_data = data + noise

    return noise_data


def time_shift(data, shift_factor=0.3):
    """時間シフトを行う."""
    # シフトするサンプル数を計算
    shift_samples = int(data.size(1) * shift_factor)

    # ゼロパディングしてシフト
    if shift_samples >= 0:
        shifted_data = torch.cat(
            [torch.zeros((1, shift_samples)), data[:, :-shift_samples]], dim=1
        )
    else:
        shifted_data = torch.cat(
            [data[:, -shift_samples:], torch.zeros((1, -shift_samples))], dim=1
        )

    return shifted_data


def time_masking(data, mask_length):
    """時間マスキングを行う."""
    # 0ではない地点を取得
    non_zero_indices = torch.nonzero(data[0]).flatten()
    start = np.random.choice(non_zero_indices[: len(non_zero_indices) - mask_length])
    # 選択された位置から mask_length 分のデータを無音で埋める
    data[:, start : start + mask_length] = 0
    return data


def expand_data(path_list, labels):
    """データセットを拡張する."""
    expanded_data = []
    max_sequence_length = 0
    for path, label in zip(path_list, labels):
        data, sample_rate = torchaudio.load("../" + path)
        sequence_length = data.size(1)
        max_sequence_length = max(max_sequence_length, sequence_length)

    for path, label in zip(path_list, labels):
        data, sample_rate = torchaudio.load("../" + path)
        # ゼロパディングを適用して音声データの長さを揃える
        if data.size(1) < max_sequence_length:
            padding = torch.zeros((1, max_sequence_length - data.size(1)))
            data = torch.cat([data, padding], dim=1)
        insert_data = data.clone()
        expanded_data.append((insert_data, int(label)))
        whitenoisy_data = add_whitenoise(data)
        expanded_data.append((whitenoisy_data, int(label)))
        shift_data = time_shift(data)
        expanded_data.append((shift_data, int(label)))
        time_mask_data = time_masking(data, int(len(data[0]) * 0.05))
        expanded_data.append((time_mask_data, int(label)))

    # データとラベルをそれぞれ別々のリストに分割
    expanded_data_list = [pair[0] for pair in expanded_data]
    expanded_label_list = [pair[1] for pair in expanded_data]

    # データとラベルのリストをテンソルに変換
    expanded_data_tensor = torch.stack(expanded_data_list)
    expanded_label_tensor = torch.tensor(expanded_label_list)

    return expanded_data_tensor, expanded_label_tensor


def make_testdata(path_list, labels):
    """テストデータを作成する."""
    test_data = []
    max_sequence_length = 0
    for path, label in zip(path_list, labels):
        data, sample_rate = torchaudio.load("../" + path)
        sequence_length = data.size(1)
        max_sequence_length = max(max_sequence_length, sequence_length)
    for path, label in zip(path_list, labels):
        data, sample_rate = torchaudio.load("../" + path)
        # ゼロパディングを適用して音声データの長さを揃える
        if data.size(1) < max_sequence_length:
            padding = torch.zeros((1, max_sequence_length - data.size(1)))
            data = torch.cat([data, padding], dim=1)
        test_data.append((data, int(label)))

    # データとラベルをそれぞれ別々のリストに分割
    test_data_list = [pair[0] for pair in test_data]
    test_label_list = [pair[1] for pair in test_data]

    # データとラベルのリストをテンソルに変換
    test_data_tensor = torch.stack(test_data_list)
    test_label_tensor = torch.tensor(test_label_list)

    return test_data_tensor, test_label_tensor


def main():
    """メイン関数."""
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス"
    )
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv(os.path.join(root, "training.csv"))

    # Dataset の作成
    expanded_data = expand_data(training["path"].values, training["label"].values)
    train_dataset = FSDD(expanded_data[0], expanded_data[1])

    # Train/Validation 分割
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], torch.Generator().manual_seed(20200616)
    )

    if args.path_to_truth:
        # Test Dataset の作成
        test = pd.read_csv(args.path_to_truth)
        test_data = make_testdata(test["path"].values, test["label"].values)
        test_dataset = FSDD(test_data[0], test_data[1])
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

    # モデルの構築
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
