#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
単一数字発話の認識.

特徴量：MFCC
識別機：CNN
"""


import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchaudio
import torchmetrics
from torch.utils.data import Dataset, random_split

import net


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class my_MLP(pl.LightningModule):
    """モデルの構築."""

    def __init__(self, input_dim: list, output_dim: int):
        """インスタンス."""
        super().__init__()
        self.model = self.create_model(input_dim, output_dim)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.confm = torchmetrics.ConfusionMatrix(10, normalize="true")

        # on_test_epoch_endのためにいる
        self.test_step_outputs = []

    def create_model(self, input_dim: list, output_dim: int):
        """モデルの構築."""
        model = net.Net(input_dim, output_dim)
        # モデル構成の表示
        print(model)
        return model

    def forward(self, x):
        """順伝搬の実装."""
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_id=None):
        """学習ステップの実装."""
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
            self.train_acc(pred,y),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_id=None):
        """バリテーションステップの実装."""
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val/acc",
            self.val_acc(pred,y),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=None):
        """テストステップの実装."""
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("test/acc", self.test_acc(pred, y), prog_bar=True, logger=True)
        # on_test_epoch_endのためにいる
        self.test_step_outputs.append({"pred": torch.argmax(pred, dim=-1), "target": y})
        return {"pred": torch.argmax(pred, dim=-1), "target": y}

    def on_test_epoch_end(self) -> None:
        """テスト終了時の実装."""
        # 混同行列を tensorboard に出力
        preds = torch.cat([tmp["pred"] for tmp in self.test_step_outputs])
        targets = torch.cat([tmp["target"] for tmp in self.test_step_outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(10), columns=range(10))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="gray_r").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def configure_optimizers(self):
        """optimizerの実装."""
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.002)
        return self.optimizer


class FSDD(Dataset):
    """データセットの構築."""
    def __init__(self, path_list, label) -> None:
        """インスタンス."""
        super().__init__()
        self.features = self.feature_extraction(path_list)
        self.label = label

    def feature_extraction(self, path_list):
        """wavファイルのリストから特徴抽出を行いリストで返す.

        特徴量：log-melスペクトル(64x32次元)
        """
        n_mfcc = 64
        n_time = 32
        datasize = len(path_list)
        features = torch.zeros(datasize, 1, n_mfcc, n_time)
        transform = torchaudio.transforms.MFCC(
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={"n_mels":64, "n_fft":512, "hop_length": 64},
        )

        for i, path in enumerate(path_list):
            data, _ = torchaudio.load(os.path.join(root, path))
            mfcc = transform(data[0])
            mfcc = (mfcc - torch.mean(mfcc)) / torch.std(mfcc)
            n_step = mfcc.shape[1]
            if n_step <= n_time:
                features[i, 0, :, : n_step] = mfcc
            else:
                for j in range(n_time):
                    features[i, 0, :, j] = torch.mean(
                        mfcc[:, n_step * j // n_time: n_step * (j + 1) // n_time],
                        axis=1,
                    )

        return features

    def __len__(self):
        """len関数."""
        return self.features.shape[0]

    def __getitem__(self, index):
        """getitem関数."""
        return self.features[index], self.label[index]


def main():
    """main関数."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス")
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv(os.path.join(root, "training.csv"))

    # Dataset の作成
    train_dataset = FSDD(training["path"].values, training["label"].values)

    # Train/Validation 分割
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        torch.Generator().manual_seed(20200616),
    )

    if args.path_to_truth:
        # Test Dataset の作成
        test = pd.read_csv(os.path.join(root, args.path_to_truth))
        test_dataset = FSDD(test["path"].values, test["label"].values)
    else:
        test_dataset = None

    # DataModule の作成
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=8,
        num_workers=1,
    )

    # モデルの構築
    model = my_MLP(
        input_dim=train_dataset[0][0][0].shape,
        output_dim=10,
    )

    # 学習の設定
    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
    )

    # モデルの学習
    trainer.fit(model=model, datamodule=datamodule)

    # バリデーション
    trainer.validate(model=model, datamodule=datamodule)

    # テスト
    if args.path_to_truth:
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
