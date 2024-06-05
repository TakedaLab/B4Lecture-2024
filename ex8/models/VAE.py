# -*- coding: utf-8 -*-
"""This file is for you to implement VAE."""

import numpy as np

# import torch
import torch.nn as nn

# import torch.nn.functional as F

MNIST_SIZE = 28


class VAE(nn.Module):
    """VAE model."""

    def __init__(self, z_dim, h_dim, drop_rate):
        """
        Set constructors.

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

    def encoder(self):
        """
        # ToDo: Implement the encoder
        ※引数は適宜追加してください
        """

    def sample_z(self):
        """
        # ToDo: Implement the sampling of the latent variable
        ※引数は適宜追加してください
        """

    def decoder(self):
        """
        # ToDo: Implement the decoder
        ※引数は適宜追加してください
        """

    def forward(self):
        """
        # ToDo: Implement the forward pass
        ※引数は適宜追加し、以下のreturn文に沿う形で結果を返すようにしてください
        """
        # return [KL, reconstruction], z, y
