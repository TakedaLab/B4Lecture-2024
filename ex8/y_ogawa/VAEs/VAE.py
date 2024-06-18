#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file is for you to implement VAE. Add variables as needed."""

import numpy as np
import torch
import torch.nn as nn

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
        """Implement the encoder."""
        x = x.view(-1, self.x_dim)
        h = nn.functional.relu(self.enc_fc1(x))
        h = nn.functional.relu(self.enc_fc2(h))
        mean = self.enc_fc3_mean(h)
        logvar = self.enc_fc3_logvar(h)
        return mean, logvar

    def sample_z(self, mean, logvar, device):
        """Implement a function to sample latent variables."""
        epsilon = torch.randn(mean.shape, device=device)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        return z

    def decoder(self, z):
        """Implement the decoder."""
        h = nn.functional.relu(self.dec_fc1(z))
        h = nn.functional.relu(self.dec_fc2(h))
        h = self.dec_drop(h)
        y = torch.sigmoid(self.dec_fc3(h))
        return y

    def forward(self, x, device):
        """Implement the forward function to return the following variables."""
        x = x.to(device)
        mean, logvar = self.encoder(x)
        z = self.sample_z(mean, logvar, device)
        y = self.decoder(z)
        KL = 0.5 * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar))
        reconstruction = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )
        return [KL, reconstruction], z, y
