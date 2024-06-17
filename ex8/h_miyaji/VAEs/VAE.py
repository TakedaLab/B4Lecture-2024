# -*- coding: utf-8 -*-
"""This file is for you to implement VAE. Add variables as needed."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_SIZE = 28


class VAE(nn.Module):
    """VAE model."""

    def __init__(self, z_dim, h_dim, drop_rate):
        """Set constructors.

        Parameters
        ----------
        z_dim : int
            Dimensions of the latent variable.
        h_dim : int
            Dimensions of the hidden layer.
        drop_rate : float
            Dropout rate.
        """
        super(VAE, self).__init__()
        self.eps = np.spacing(1)
        self.x_dim = MNIST_SIZE * MNIST_SIZE  # The image in MNIST is 28×28
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.drop_rate = drop_rate

        self.enc_fc1 = nn.Linear(self.x_dim, self.h_dim)
        self.enc_fc2 = nn.Linear(self.h_dim, int(self.h_dim / 2))
        self.enc_fc3_mean = nn.Linear(int(self.h_dim / 2), z_dim)
        self.enc_fc3_logvar = nn.Linear(int(self.h_dim / 2), z_dim)

        self.dec_fc1 = nn.Linear(z_dim, int(self.h_dim / 2))
        self.dec_fc2 = nn.Linear(int(self.h_dim / 2), self.h_dim)
        self.dec_drop = nn.Dropout(self.drop_rate)
        self.dec_fc3 = nn.Linear(self.h_dim, self.x_dim)

    def encoder(self, x):
        """Encode the input data.

        Parameters
        ----------
        x : torch.tensor
            Input data (batch size, x_dim).

        Returns
        -------
        mean : torch.tensor
            Mean (batch size, z_dim).
        log_var : torch.tensor
            Logarithm of variance (batch size, z_dim).
        """
        x = x.view(-1, self.x_dim)  # x.shapeを自動調整
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))  # 活性化関数に通す
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)

    def sample_z(self, mean, log_var, device):
        """Sample latent variables.

        Parameters
        ----------
        mean : torch.tensor
            Mean (batch size, z_dim).
        log_var : torch.tensor
            Logarithm of variance (batch size, z_dim).
        device : torch.device
            "cuda" (GPU is available) or "cpu" (otherwise).

        Returns
        -------
        z : torch.tensor
            Latent variable (batch size, z_dim)
        """
        epsilon = torch.randn(mean.shape, device=device)  # ε~N(0, 1)を生成
        z = mean + epsilon * torch.exp(0.5 * log_var)
        return z

    def decoder(self, z):
        """Decode the latent variable.

        Parameters
        ----------
        z : torch.tensor
            Latent variable (batch size, z_dim).

        Returns
        -------
        y : torch.tensor
            Output data (batch size, x_dim).
        """
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))

    def forward(self, x, device):
        """Return KL divergence, reconstruction loss, latent variable, and output data.

        Parameters
        ----------
        x : torch.tensor
            Input data (batch size, x_dim).
        device : torch.device
            "cuda" (GPU is available) or "cpu" (otherwise).

        Returns
        -------
        list
            List of KL divergence and reconstruction loss.
        z : torch.tensor
            Latent variable (batch size, z_dim).
        y : torch.tensor
            Output data (batch size, x_dim).
        """
        x = x.to(device)  # xをGPU/CPUに転送
        mean, log_var = self.encoder(x)
        z = self.sample_z(mean, log_var, device)
        y = self.decoder(z)

        # KL = 1/2 * sum(1 + log(σ^2) - σ^2 - μ^2)
        KL = 0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean**2)

        # x*log(y) + (1-x)*log(1-y)
        reconstruction = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )

        return [KL, reconstruction], z, y
