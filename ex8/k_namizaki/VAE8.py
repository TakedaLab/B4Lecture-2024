# -*- coding: utf-8 -*-
"""This file is for you to implement VAE. Add variables as needed."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    """Set the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)  # この関数をモデル定義やデータローダーの初期化前に呼び出す


MNIST_SIZE = 28  # MNIST画像は28x28ピクセル


class VAE(nn.Module):
    """VAE model."""

    def __init__(self, z_dim, h_dim, drop_rate):
        """Set constructors.

        Parameters
        ----------
        z_dim : int
            潜在変数の次元数。
        h_dim : int
            隠れ層の次元数。
        drop_rate : float
            過学習を防ぐためのドロップアウト率。
        """
        super(VAE, self).__init__()
        self.eps = np.spacing(1)  # ゼロ除算を防ぐための小さな定数
        self.x_dim = MNIST_SIZE * MNIST_SIZE  # フラット化された画像サイズ（28*28）
        self.z_dim = z_dim  # 潜在変数の次元数
        self.h_dim = h_dim  # 隠れ層の次元数
        self.drop_rate = drop_rate  # ドロップアウト率

        # エンコーダの層
        self.enc_fc1 = nn.Linear(self.x_dim, self.h_dim)  # 全結合層1
        self.enc_fc2 = nn.Linear(self.h_dim, int(self.h_dim / 2))  # 全結合層2
        self.enc_fc3_mean = nn.Linear(int(self.h_dim / 2), z_dim)  # 潜在変数の平均の層
        self.enc_fc3_logvar = nn.Linear(
            int(self.h_dim / 2), z_dim
        )  # 潜在変数の対数分散の層

        # デコーダの層
        self.dec_fc1 = nn.Linear(z_dim, int(self.h_dim / 2))  # 全結合層1
        self.dec_fc2 = nn.Linear(int(self.h_dim / 2), self.h_dim)  # 全結合層2
        self.dec_drop = nn.Dropout(self.drop_rate)  # ドロップアウト層
        self.dec_fc3 = nn.Linear(self.h_dim, self.x_dim)  # 画像を再構築するための出力層

    def encoder(self, x):
        """入力画像を潜在空間にエンコードする.

        Parameters
        ----------
        x : torch.Tensor
            入力画像テンソル。

        Returns
        -------
        mean : torch.Tensor
            潜在変数の平均。
        logvar : torch.Tensor
            潜在変数の対数分散。
        """
        h = x.view(-1, self.x_dim)  # テンソルを２次元にリサイズ
        h = F.relu(self.enc_fc1(h))  # 最初の全結合層をReLU活性化関数で通過
        h = F.relu(self.enc_fc2(h))  # 2番目の全結合層をReLU活性化関数で通過

        mean = self.enc_fc3_mean(h)  # 平均
        logvar = self.enc_fc3_logvar(h)  # 対数分散

        return mean, logvar

    def sample_z(self, mean, logvar, device):
        """平均と対数分散で定義された分布から潜在変数zをサンプリングする.

        Parameters
        ----------
        mean : torch.Tensor
            潜在変数の平均。
        logvar : torch.Tensor
            潜在変数の対数分散。

        Returns
        -------
        z : torch.Tensor
            サンプリングされた潜在変数。
        """
        epsilon = torch.randn(mean.shape, device=device)  # 標準正規分布からε
        # 0.5はlog_varがlog(sigma^2)で、使いたいのはlog(sigma)だから
        z = mean + epsilon * torch.exp(0.5 * logvar)  # 潜在変数をサンプリング
        return z

    def decoder(self, z):
        """潜在変数zを画像空間にデコードする.

        Parameters
        ----------
        z : torch.Tensor
            潜在変数。

        Returns
        -------
        reconstruction : torch.Tensor
            再構築された画像。
        """
        h = F.relu(self.dec_fc1(z))  # 最初の全結合層をReLU活性化関数で通過
        h = F.relu(self.dec_fc2(h))  # 2番目の全結合層をReLU活性化関数で通過
        h = self.dec_drop(h)  # ドロップアウト層を通過
        y = torch.sigmoid(
            self.dec_fc3(h)
        )  # 出力層をシグモイド活性化関数で通過し、出力
        return y

    def forward(self, x, device):
        """ネットワークを前向きに通過させる.

        Parameters
        ----------
        x : torch.Tensor
            入力画像テンソル。

        Returns
        -------
        [KL, reconstruction] : list
            KLダイバージェンスと再構築誤差を含むリスト。
        z : torch.Tensor
            潜在変数。
        y : torch.Tensor
            再構築された画像。
        """
        x = x.to(device)
        mean, logvar = self.encoder(x)  # エンコーダで平均と対数分散を計算
        # リパラメトリゼーショントリックを用いて潜在変数をサンプリング
        z = self.sample_z(mean, logvar, device)
        y = self.decoder(z)  # デコーダで潜在変数から再構築された画像を生成
        # KLダイバージェンスを計算
        KL = 0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
        # 再構築誤差を計算
        reconstruction = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )
        # KLダイバージェンスと再構築誤差、潜在変数、再構築された画像を返す
        return [KL, reconstruction], z, y