#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""モデルを構築するためのtorch.nn.Moduleを継承したNetクラスを定義.

main.pyで使用.
"""

import torch


class ResBlock(torch.nn.Module):
    """スキップ接続のグループ化."""

    def __init__(self, n_chans):
        """インスタンス."""
        super(ResBlock, self).__init__()
        self.conv = torch.nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batchnorm = torch.nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        torch.nn.init.constant_(self.batchnorm.weight, 0.5)
        torch.nn.init.zeros_(self.batchnorm.bias)

    def forward(self, x):
        """順伝搬を実装."""
        out = self.conv(x)
        out = self.batchnorm(out)
        out = torch.nn.ReLU()(out)
        return out + x


class Net(torch.nn.Module):
    """モデルを構築するためのクラス."""

    def __init__(self, input_dim: list, output_dim: int) -> None:
        """インスタンス."""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=64)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.resblock1 = torch.nn.Sequential(*(2 * [ResBlock(64)]))
        self.conv2 = torch.nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=16)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.input_liner_dim = input_dim[0] * input_dim[1] * 16 // (4**2)
        self.liner_in = torch.nn.Linear(self.input_liner_dim, 256)
        self.liner_out = torch.nn.Linear(256, output_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x) -> torch.tensor:
        """順伝搬を実装."""
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = torch.nn.ReLU()(out)
        out = self.pool1(out)
        out = self.resblock1(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = torch.nn.ReLU()(out)
        out = self.pool2(out)
        out = out.view(-1, self.input_liner_dim)
        out = self.liner_in(out)
        out = torch.nn.ReLU()(out)
        out = torch.nn.Dropout(0.2)(out)
        out = self.liner_out(out)
        out = self.softmax(out)
        return out
