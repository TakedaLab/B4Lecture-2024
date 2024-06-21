# -*- coding: utf-8 -*-
"""This file is for you to implement VAE. Add variables as needed."""

import numpy as np
import torch
import torch.nn as nn

import torch
import torch.nn.functional as F

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
        self.enc_fc3_logvar = nn.Linear(int(self.h_dim / 2), z_dim)  # 潜在変数の対数分散の層

<<<<<<< HEAD
        # デコーダの層
        self.dec_fc1 = nn.Linear(z_dim, int(self.h_dim / 2))  # 全結合層1
        self.dec_fc2 = nn.Linear(int(self.h_dim / 2), self.h_dim)  # 全結合層2
        self.dec_drop = nn.Dropout(self.drop_rate)  # ドロップアウト層
        self.dec_fc3 = nn.Linear(self.h_dim, self.x_dim)  # 画像を再構築するための出力層

    def encoder(self, x):
        """入力画像を潜在空間にエンコードする。

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
        """# ToDo: Implement the encoder."""
        h = x.view(-1, self.x_dim)  # テンソルを２次元にリサイズ
        h = F.relu(self.enc_fc1(x))  # 最初の全結合層をReLU活性化関数で通過
        h = F.relu(self.enc_fc2(h))  # 2番目の全結合層をReLU活性化関数で通過

        # どのサイトでもこのままだけど、これで平均と対数分散が分かれているのか？
        mean = self.enc_fc3_mean(h)  # 平均
        logvar = self.enc_fc3_logvar(h)  # 対数分散

        return mean, logvar

    def sample_z(self, mean, logvar):
        """平均と対数分散で定義された分布から潜在変数zをサンプリングする。

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
        """# ToDo: Implement a function to sample latent variables."""
        epsilon = torch.randn_like(logvar)  # 標準正規分布からε
        # 0.5はlog_varがlog(sigma^2)で、使いたいのはlog(sigma)だから
        z = mean + epsilon * torch.exp(0.5 * logvar)  # 潜在変数をサンプリング
        return z

    def decoder(self, z):
        """潜在変数zを画像空間にデコードする。

        Parameters
        ----------
        z : torch.Tensor
            潜在変数。

        Returns
        -------
        reconstruction : torch.Tensor
            再構築された画像。
        """
        """# ToDo: Implement the decoder."""
        # TODO: デコーダのロジックを実装する
        h = F.relu(self.dec_fc1(z))  # 最初の全結合層をReLU活性化関数で通過
        h = F.relu(self.dec_fc2(h))  # 2番目の全結合層をReLU活性化関数で通過
        h = self.dec_drop(h)  # ドロップアウト層を通過
        y = torch.sigmoid(self.dec_fc3(z))  # 出力層をシグモイド活性化関数で通過し、再構築された画像を出力
        return y


    def forward(self, x):
        """ネットワークを前向きに通過させる。

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
        """# ToDo: Implement the forward function to return the following variables."""
        # TODO: 次の変数を返すためのforward関数を実装する
        mean, logvar = self.encoder(x)  # エンコーダで平均と対数分散を計算
        z = self.sample_z(mean, logvar)  # リパラメトリゼーショントリックを用いて潜在変数をサンプリング
        y = self.decoder(z)  # デコーダで潜在変数から再構築された画像を生成
        KL = 0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))  # KLダイバージェンスを計算
        # 資料のramda_sigumaがyの部分
        reconstruction = torch.sum(x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps))  # 再構築誤差を計算(クロスエントロピー？)
        return [KL, reconstruction], z, y  # KLダイバージェンスと再構築誤差、潜在変数、再構築された画像を返す
=======
    def encoder(self, x: torch.Tensor):
        """# ToDo: Implement the encoder."""
        x = x.view(-1, self.x_dim)
        x = nn.functional.relu(self.enc_fc1(x))
        x = nn.functional.relu(self.enc_fc2(x))
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)

    def sample_z(self, mean: torch.Tensor, log_var: torch.Tensor, device: torch.device):
        """# ToDo: Implement a function to sample latent variables."""
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def decoder(self, z: torch.Tensor):
        """# ToDo: Implement the decoder."""
        z = nn.functional.relu(self.dec_fc1(z))
        z = nn.functional.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))

    def forward(self, x: torch.Tensor, device: torch.device):
        """# ToDo: Implement the forward function to return the following variables."""
        x = x.to(device)
        mean, log_var = self.encoder(x)
        z = self.sample_z(mean, log_var, device)
        y = self.decoder(z)
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        reconstruction = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )
        return [KL, reconstruction], z, y
>>>>>>> e80e126c6833a04fcf3e898485e9784d49b85aa3
