# -*- coding: utf-8 -*-
"""This file is used to visualize the results."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import ArtistAnimation
from sklearn.decomposition import PCA

DIR = "images_4dim"  # 画像を保存するディレクトリ


class Visualize:
    """Visualize the results of the VAE model."""

    def __init__(self, z_dim, h_dim, dataloader_test, model, device):
        """Set constructors.

        Parameters
        ----------
        z_dim : int
            Dimensions of the latent variable.
        h_dim : int
            Dimensions of the hidden layer.
        dataloader_test : torch.utils.data.dataloader.DataLoader
            DataLoader for the test data.
        model : VAE
            VAE model.
        device : torch.device
            "cuda" if GPU is available, or "cpu" otherwise.
        """
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.dataloader_test = dataloader_test
        self.model = model
        self.device = device

    def createDirectories(self):
        """Create directories for storing images."""
        os.makedirs(f"./{DIR}/reconstruction", exist_ok=True)
        os.makedirs(f"./{DIR}/latent_space", exist_ok=True)
        os.makedirs(f"./{DIR}/lattice_point", exist_ok=True)
        os.makedirs(f"./{DIR}/walkthrough", exist_ok=True)

    def reconstruction(self):
        """Visualize the reconstructed images."""
        for num_batch, data in enumerate(self.dataloader_test):
            fig, axes = plt.subplots(2, 10, figsize=(20, 4))
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
            for i, im in enumerate(data[0].view(-1, 28, 28)[:10]):
                axes[0][i].imshow(im, "gray")
            _, _, y = self.model(data[0], self.device)
            y = y.cpu().detach().numpy().reshape(-1, 28, 28)
            for i, im in enumerate(y[:10]):
                axes[1][i].imshow(im, "gray")
            fig.savefig(f"./{DIR}/reconstruction/z_{self.z_dim}_{num_batch}.png")
            plt.close(fig)

    def convert_to_two_dim(self, z):
        """PCAを適用し, 潜在空間の次元数を2まで削減する."""
        autoscaled_z = (z - z.mean()) / z.std()
        pca = PCA(n_components=2)
        z_dim2 = pca.fit_transform(autoscaled_z)
        return z_dim2

    def latent_space(self):
        """Visualize latent space."""
        cm = plt.get_cmap("tab10")
        for num_batch, data in enumerate(self.dataloader_test):
            fig_plot, ax_plot = plt.subplots(figsize=(9, 9))
            fig_scatter, ax_scatter = plt.subplots(figsize=(9, 9))
            _, z, _ = self.model(data[0], self.device)
            z = z.cpu().detach().numpy()
            if self.z_dim != 2:
                z = self.convert_to_two_dim(z)
            for k in range(10):
                cluster_indexes = np.where(data[1].cpu().detach().numpy() == k)[0]
                ax_plot.plot(
                    z[cluster_indexes, 0], z[cluster_indexes, 1], "o", ms=4, color=cm(k)
                )
                ax_scatter.scatter(
                    z[cluster_indexes, 0],
                    z[cluster_indexes, 1],
                    marker=f"${k}$",
                    color=cm(k),
                )
            fig_plot.savefig(
                f"./{DIR}/latent_space/z_{self.z_dim}_{num_batch}_plot.png"
            )
            fig_scatter.savefig(
                f"./{DIR}/latent_space/z_{self.z_dim}_{num_batch}_scatter.png"
            )
            plt.close(fig_plot)
            plt.close(fig_scatter)

    def lattice_point(self):
        """Visualize latent space generated from artificial lattice point."""
        # The size of Z must be (Batch size, z_dim)
        num_image = 25  # How many images per row (column)
        x = np.linspace(-2, 2, num_image)
        y = np.linspace(-2, 2, num_image)
        z_x, z_y = np.meshgrid(x, y)  # (num_image, num_image)

        Z = np.zeros((num_image * num_image, self.z_dim))

        # 潜在変数の0次元目と1次元目以外を0に固定して、2次元分のみ動かしてできた出力をイメージ画像とする
        Z[:, 0] = z_x.flatten()
        Z[:, 1] = z_y.flatten()

        Z = torch.tensor(Z, dtype=torch.float).to(self.device)  # tensor変換

        y = self.model.decoder(Z).cpu().detach().numpy().reshape(-1, 28, 28)
        fig, axes = plt.subplots(num_image, num_image, figsize=(9, 9))
        for i in range(num_image):
            for j in range(num_image):
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                axes[i][j].imshow(y[num_image * (num_image - 1 - i) + j], "gray")
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f"./{DIR}/lattice_point/z_{self.z_dim}.png")
        plt.close(fig)

    def walkthrough(self):
        """Create animations of the reconstructed images by walking through the latent space."""
        self.step = 50  # Step size of the animation
        self.z11 = torch.tensor([-3, 0], dtype=torch.float)
        self.z12 = torch.tensor([3, 0], dtype=torch.float)
        self.z21 = torch.tensor([-3, 3], dtype=torch.float)
        self.z22 = torch.tensor([3, -3], dtype=torch.float)
        self.z31 = torch.tensor([0, 3], dtype=torch.float)
        self.z32 = torch.tensor([0, -3], dtype=torch.float)
        self.z41 = torch.tensor([3, 3], dtype=torch.float)
        self.z42 = torch.tensor([-3, -3], dtype=torch.float)
        z1_list = [self.z11, self.z21, self.z31, self.z41]
        z2_list = [self.z12, self.z22, self.z32, self.z42]

        z1_to_z2_list = []
        y1_to_y2_list = []
        # Store latent variables which are linearly changed from a start point to goal point
        for z1, z2 in zip(z1_list, z2_list):
            if self.z_dim > 2:
                pad_width = self.z_dim - 2
                z1 = torch.nn.functional.pad(
                    z1, (0, pad_width), mode="constant", value=0.0
                )
                z2 = torch.nn.functional.pad(
                    z2, (0, pad_width), mode="constant", value=0.0
                )

            z1_to_z2_list.append(
                torch.cat(
                    [
                        ((z1 * ((self.step - i) / self.step)) + (z2 * (i / self.step)))
                        for i in range(self.step)
                    ]
                )
                .reshape(self.step, self.z_dim)
                .to(self.device)
            )
        # Store the output of each latent variable from the decoder
        for z1_to_z2 in z1_to_z2_list:
            y1_to_y2_list.append(
                self.model.decoder(z1_to_z2).cpu().detach().numpy().reshape(-1, 28, 28)
            )
        # Create gif animations
        for n in range(len(y1_to_y2_list)):
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_xticks([])
            ax.set_yticks([])
            images = []
            for _, im in enumerate(y1_to_y2_list[n]):
                images.append([ax.imshow(im, "gray")])
            animation = ArtistAnimation(
                fig, images, interval=100, blit=True, repeat_delay=1000
            )
            animation.save(
                f"./{DIR}/walkthrough/z_{self.z_dim}_{n}.gif", writer="pillow"
            )
            plt.close(fig)
