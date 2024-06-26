"""Train the diffusion model for image denoising."""

import logging
import os
from typing import Any, Dict

import diffusers
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

os.environ["HYDRA_FULL_ERROR"] = "1"


class DiffusionModel(pl.LightningModule):
    """Diffusion model for image denoising."""

    def __init__(
        self,
        model: torch.nn.Module,  # Noise prediction model
        criterion: torch.nn.Module,  # Loss function
        optimizer: torch.optim.Optimizer,  # Optimizer
        num_epochs: int,  # max num epochs
        num_timesteps: int,  # Time steps of the diffusion
        noise_schedule: str,  # Noise scheduler type
        noise_schedule_kwargs: Dict[str, Any],  # Arguments for noise scheduler
        num_samples: tuple,  # Number of samples for visualization
        image_size: tuple,  # Image size
        every_n_epochs: int,  # Visualization interval
    ) -> None:
        """Initialize the diffusion model."""
        super(DiffusionModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.num_timesteps = num_timesteps

        if noise_schedule == "linear":
            beta = torch.linspace(
                noise_schedule_kwargs["start"],
                noise_schedule_kwargs["end"],
                num_timesteps,
                device=model.device,
            )
            alpha = 1.0 - beta
            alpha_prod = alpha.cumprod(dim=0)

            self.register_buffer("beta", beta)
            self.register_buffer("alpha", alpha)
            self.register_buffer("alpha_prod", alpha_prod)

        self.num_samples = tuple(num_samples)
        self.image_size = tuple(image_size)
        self.every_n_epochs = every_n_epochs

        # GIF作成用
        self.num_epochs = num_epochs
        self.generated_images = []

    def configure_optimizers(self):
        """Configure optimizer."""
        return self.optimizer

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward noise prediction model.

        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            t (torch.Tensor): Time step (B,)

        Returns:
            torch.Tensor: Predicted noise (B, C, H, W)
        """
        return self.model(x, t).sample

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward process of the diffusion model. x_t ~ q(x_t|x_0).

        Args:
            x0 (torch.Tensor): Clean image x_0 (B, C, H, W)
            t (torch.Tensor): Time step (B,)
            noise (torch.Tensor, optional): Noise tensor. Defaults to None.

        Returns:
            torch.Tensor: Noisy image x_t (B, C, H, W)
        """
        if noise is None:  # noiseのデータが無い場合
            # x0と同じサイズで平均0/分散1の正規分布乱数を作成
            noise = torch.randn_like(x0)
        a = self.alpha_prod[t].view(-1, 1, 1, 1)
        return x0 * a.sqrt() + noise * (1 - a).sqrt()  # noise付加の定義式

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Inverse process of the diffusion model. x_t ~ p(x_t|x_{t+1}).

        Args:
            x (torch.Tensor): Noisy image x_{t+1} (B, C, H, W)
            t (torch.Tensor): Time step (B,)

        Returns:
            torch.Tensor: Denoised image x_t (B, C, H, W)
        """
        # ↓tensorに保存されている勾配値を保持しない設定にできる
        with torch.no_grad():
            noise_pred = self.forward(x, t)  # noise予測値
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)  # 1-beta(np.linspace)
            alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)

            # ノイズ除去の定義式
            x = (
                x - noise_pred * (1 - alpha_t) / (1 - alpha_prod_t).sqrt()
            ) / alpha_t.sqrt()
            if t[0].item() != 0:
                x = x + torch.randn_like(x) * (1 - alpha_t).sqrt()
            return x

    def training_step(self, batch, batch_idx):
        """Training 1 step.

        Args:
            batch (tuple): Input batch
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Loss
        """
        # 画像読み込み
        images = batch["images"]
        images = images.to(self.device)

        # 0~1000(timesteps)のrandintをとる
        t = torch.randint(
            0, self.num_timesteps, (images.size(0),), device=self.device
        ).long()

        noise = torch.randn_like(images)  # randomノイズ生成
        noisy_images = self.q_sample(images, t, noise)  # ノイズ画像にするステップ

        outputs = self.forward(noisy_images, t)  # 出力=forwardした結果(予測ノイズ？)
        loss = self.criterion(outputs, noise)  # loss=noiseと出力の差
        self.log("train_loss", loss, prog_bar=True)  # log出力

        return loss

    def generate(self, num_timesteps, shape):
        """Generate samples from the diffusion model."""
        x = torch.randn(shape).to(self.device)
        for t in tqdm(range(num_timesteps - 1, -1, -1)):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t)
        return x

    def on_train_epoch_end(self):
        """Generate images at the end of each epoch."""
        if self.current_epoch % self.every_n_epochs == self.every_n_epochs - 1:
            logging.info("Generating Images...")

            # 画像生成
            generated_image = self.generate(
                self.num_timesteps,
                (self.num_samples[0] * self.num_samples[1],) + self.image_size,
            )
            generated_image = (generated_image + 1) / 2
            generated_image = (
                generated_image.reshape(self.num_samples + self.image_size)
                .permute([2, 0, 3, 1, 4])
                .flatten(-4, -3)
                .flatten(-2, -1)
            )
            # Tensorboard出力
            self.logger.experiment.add_image(
                "Generated Images",
                generated_image,
                self.current_epoch,
                dataformats="HW" if generated_image.ndim == 2 else "CHW",
            )

            # GIF作成用
            self.generated_images.append(transforms.ToPILImage()(generated_image.cpu()))
            logging.info("Done.")

        if self.current_epoch == self.num_epochs - 1:
            # モデルの成長過程GIF
            logging.info("Generating GIF Images...")
            gif_path = os.path.join("figs", "generated_images.gif")
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)

            self.generated_images[0].save(
                gif_path,
                save_all=True,
                append_images=self.generated_images[1:],
                duration=200,  # フレーム数(ms)/1画像
                loop=0,  # 無限ループ
            )

            # 最終結果の逆拡散課程GIF
            self.generate_gif()
            logging.info("Done. GIF saved to figs")

    def generate_gif(self):
        """Generate GIF images at the end of train."""
        # GIF作成用
        images = []

        times = np.linspace(1, self.num_timesteps, 20)

        for t in times:
            generated_image = self.generate(
                int(t),
                (self.num_samples[0] * self.num_samples[1],) + self.image_size,
            )
            generated_image = (generated_image + 1) / 2
            generated_image = (
                generated_image.reshape(self.num_samples + self.image_size)
                .permute([2, 0, 3, 1, 4])
                .flatten(-4, -3)
                .flatten(-2, -1)
            )

            # tensor -> PIL 画像
            pil_image = transforms.ToPILImage()(generated_image.cpu())
            images.append(pil_image)
        images.append(pil_image)

        # GIF保存
        gif_path = os.path.join(
            "figs", f"generated_images_epoch_{self.current_epoch+1}_loop.gif"
        )
        gif_path_stop = os.path.join(
            "figs", f"generated_images_epoch_{self.current_epoch+1}_stop.gif"
        )
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)

        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=100,  # フレーム数(ms)/1画像
            loop=0,
        )
        images[0].save(
            gif_path_stop,
            save_all=True,
            append_images=images[1:],
            duration=100,  # フレーム数(ms)/1画像
        )


@hydra.main(config_path="conf", config_name="default.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the diffusion model."""
    # fix seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # prepare dataset
    train_dataset = load_dataset(
        "huggan/smithsonian_butterflies_subset", split="train", cache_dir=cfg.datadir
    )
    # preprocess
    preprocess = transforms.Compose(
        [
            transforms.Resize(cfg.plot.image_size[-2:]),  # Resize
            transforms.ToTensor(),  # ToTensor
            transforms.Normalize(
                [0.5], [0.5]
            ),  # Normalize to [-1, 1]　平均0.5, 標準偏差0.5で標準化
        ]
    )

    def transform(examples):
        """Transform function for the dataset."""
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    train_dataset.set_transform(transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4
    )

    model = diffusers.UNet2DModel(**cfg.model)
    criterion = nn.MSELoss()  # 判断するための条件や基準：平均二乗誤差
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer)

    diffmodel = DiffusionModel(
        model,
        criterion,
        optimizer,
        cfg.train.num_epochs,
        **cfg.diffusion,
        **cfg.plot,
    )

    diffmodel.save_hyperparameters(cfg)

    # configure logger
    tb_logger = TensorBoardLogger(outdir)
    # train
    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        accelerator="gpu",
        devices=1,
        logger=tb_logger,
    )
    trainer.fit(diffmodel, train_loader)

    # save model
    torch.save(diffmodel.state_dict(), outdir + "/model.pth")


if __name__ == "__main__":
    main()
