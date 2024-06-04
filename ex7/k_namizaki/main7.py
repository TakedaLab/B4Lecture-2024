#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
機械学習をしてみる.

バッチ正規化を追加
MLPモデルの層を増加
Adamに変更
ホワイトノイズデータ増殖
B4輪講最終課題 パターン認識に挑戦してみよう
ベースラインスクリプト(Pytorch Lightning版)
特徴量；MFCCの平均（0次項含まず）
識別器；MLP
"""

import argparse
import os

import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchaudio
import torchmetrics
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class my_MLP(pl.LightningModule):
    """MLPを作る."""

    def __init__(self, input_dim, output_dim):
        """init."""
        super().__init__()
        # モデルを作成
        self.model = self.create_model(input_dim, output_dim)
        # 損失関数の設定
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # トレーニング、検証、テストの正確度を計測するメトリクス
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        # 混同行列の設定
        self.confm = torchmetrics.ConfusionMatrix(10, normalize="true")

    def create_model(self, input_dim, output_dim):
        """
        MLPモデルの構築.

        Args:
            input_dim: 入力の形
            output_dim: 出力次元
        Returns:
            model: 定義済みモデル
        """
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),  # 入力層から隠れ層への線形変換
            torch.nn.BatchNorm1d(512),  # バッチ正規化
            torch.nn.ReLU(),  # 活性化関数ReLU
            torch.nn.Dropout(0.2),  # ドロップアウト
            torch.nn.Linear(512, 256),  # 隠れ層から隠れ層への線形変換
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, output_dim),  # 隠れ層から出力層への線形変換
            torch.nn.Softmax(dim=-1),  # 出力の正規化
        )
        # モデル構成の表示
        print(model)
        return model

    def forward(self, x):
        """モデルの前向き計算."""
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_id=None):
        """Train step."""
        x, y = batch
        pred = self.forward(x)
        # 損失の計算
        loss = self.loss_fn(pred, y)
        # 損失をログに記録
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
        )
        # トレーニング精度をログに記録
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
        """Validate step."""
        x, y = batch
        pred = self.forward(x)
        # 損失の計算
        loss = self.loss_fn(pred, y)
        # 検証精度をログに記録
        self.log("val/acc", self.val_acc(pred, y), prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=None):
        """Test step."""
        x, y = batch
        pred = self.forward(x)
        # 損失の計算
        # loss = self.loss_fn(pred, y)
        # テスト精度をログに記録
        self.log("test/acc", self.test_acc(pred, y), prog_bar=True, logger=True)
        return {"pred": torch.argmax(pred, dim=-1), "target": y}

    def test_epoch_end(self, outputs) -> None:
        """混同行列を tensorboard に出力."""
        preds = torch.cat([tmp["pred"] for tmp in outputs])
        targets = torch.cat([tmp["target"] for tmp in outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(
            confusion_matrix.cpu().numpy(), index=range(10), columns=range(10)
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="gray_r").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def configure_optimizers(self):
        """Configure optimizers."""
        # 最適化手法の設定（Adam）
        self.optimizer = torch.optim.Adam(self.model.parameters())
        return self.optimizer


class FSDD(Dataset):
    """FSDDを作る."""

    def __init__(self, path_list, label) -> None:
        """init."""
        super().__init__()
        # 特徴抽出を実行
        self.features = self.feature_extraction(path_list)
        self.label = label

    def feature_extraction(self, path_list):
        """
        wavファイルのリストから特徴抽出を行いリストで返す.

        扱う特徴量はMFCC13次元の平均（0次は含めない）
        Args:
            path_list: 特徴抽出するファイルのパスリスト
        Returns:
            features: 特徴量
        """
        n_mfcc = 13
        datasize = len(path_list)
        features = torch.zeros(datasize, n_mfcc)
        transform = torchaudio.transforms.MFCC(
            n_mfcc=13, melkwargs={"n_mels": 64, "n_fft": 512}
        )
        for i, path in enumerate(path_list):
            # データ読み込み (data.shape == (channel, time))
            data, _ = torchaudio.load(os.path.join(root, path))
            # 特徴量の計算 (MFCC の平均)
            features[i] = torch.mean(transform(data[0]), axis=1)
        return features

    def __len__(self):
        """データセットのサイズを返す."""
        return self.features.shape[0]

    def __getitem__(self, index):
        """指定されたインデックスの特徴量とラベルを返す."""
        return self.features[index], self.label[index]


def augment_and_expand_dataset(dataset):
    """データを拡張増殖."""
    augmented_features = []
    augmented_labels = []
    for features, label in dataset:
        # まず[features]
        augmented_features.append(features)
        augmented_labels.append(label)
        # [features],[white_noise]
        augmented_data = white_data(features)
        augmented_features.append(augmented_data)
        augmented_labels.append(label)
    augmented_features = torch.stack(augmented_features)
    augmented_labels = torch.tensor(augmented_labels)
    return augmented_features, augmented_labels


def white_data(features):
    """ホワイトノイズデータを作る."""
    # ホワイトノイズの追加
    noise = torch.randn_like(features) * 0.005
    augmented_features = features + noise
    return augmented_features


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス"
    )
    args = parser.parse_args()

    # トレーニングデータの読み込み
    training = pd.read_csv(os.path.join(root, "training.csv"))

    # Dataset の作成
    train_dataset = FSDD(training["path"].values, training["label"].values)

    # Train(学習用)/Validation(一時的なテスト用) 分割
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        torch.Generator().manual_seed(20240605),  # 乱数シードの設定
    )

    # train_datasetの水増し
    augmented_features, augmented_labels = augment_and_expand_dataset(train_dataset)
    train_dataset = torch.utils.data.TensorDataset(augmented_features, augmented_labels)

    if args.path_to_truth:
        # 本物のテストデータの読み込み
        test = pd.read_csv(args.path_to_truth)
        # Test Dataset の作成
        test_dataset = FSDD(test["path"].values, test["label"].values)
    else:
        test_dataset = None

    # DataModule の作成
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=32,  # バッチサイズの設定
        num_workers=4,  # データローディング時のワーカースレッド数の設定
    )

    # モデルの構築
    model = my_MLP(
        input_dim=train_dataset[0][0].shape[0], output_dim=10  # 入力次元の設定
    )  # 出力次元の設定（クラス数）

    # 学習の設定
    trainer = pl.Trainer(max_epochs=100, gpus=1)  # エポック数とGPUの設定

    # モデルの学習
    trainer.fit(model=model, datamodule=datamodule)

    # バリデーション
    trainer.validate(model=model, datamodule=datamodule)

    if args.path_to_truth:
        # テスト
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
