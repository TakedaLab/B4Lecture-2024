"""Train the diffusion model for image denoising."""

import logging
from typing import Any, Dict

import diffusers
import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class DiffusionModel(pl.LightningModule):
    """画像のノイズ除去のための拡散モデル."""

    def __init__(
        self,
        model: torch.nn.Module,  # ノイズ予測モデル
        criterion: torch.nn.Module,  # 損失関数
        optimizer: torch.optim.Optimizer,  # オプティマイザー
        num_timesteps: int,  # 拡散のタイムステップ数
        noise_schedule: str,  # ノイズスケジュールの種類
        noise_schedule_kwargs: Dict[str, Any],  # ノイズスケジュールの引数
        num_samples: tuple,  # 可視化のためのサンプル数
        image_size: tuple,  # 画像サイズ
        every_n_epochs: int,  # 可視化のインターバル
    ) -> None:
        """拡散モデルの初期化."""
        super(DiffusionModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.num_timesteps = num_timesteps

        self.generated_images = []  # 生成された画像を保存するリスト

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

    def configure_optimizers(self):
        """オプティマイザーを設定."""
        return self.optimizer

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ノイズ予測モデルのフォワード.

        Args:
            x (torch.Tensor): 入力画像 (B, C, H, W)
            t (torch.Tensor): タイムステップ (B,)

        Returns:
            torch.Tensor: 予測されたノイズ (B, C, H, W)
        """
        return self.model(x, t).sample

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """拡散モデルのフォワードプロセス。 x_t ~ q(x_t|x_0).

        Args:
            x0 (torch.Tensor): クリーンな画像 x_0 (B, C, H, W)
            t (torch.Tensor): タイムステップ (B,)
            noise (torch.Tensor, optional): ノイズテンソル。デフォルトはNone。

        Returns:
            torch.Tensor: ノイズのある画像 x_t (B, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(
                x0
            )  # ノイズがNoneの場合、新たにランダムなノイズを生成
        # タイムステップに対応するアルファ値(形状を [batch_size, 1, 1, 1] に変換)
        alpha_t = self.alpha_prod[t].view(-1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1)  # タイムステップに対応するベータ値
        x_t = (
            torch.sqrt(alpha_t) * x0 + torch.sqrt(beta_t) * noise
        )  # 拡散プロセスを適用してノイズを追加
        return x_t  # ノイズのある画像を返す

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """拡散モデルの逆プロセス。 x_t ~ p(x_t|x_{t+1}).

        Args:
            x (torch.Tensor): ノイズのある画像 x_{t+1} (B, C, H, W)
            t (torch.Tensor): タイムステップ (B,)

        Returns:
            torch.Tensor: ノイズ除去された画像 x_t (B, C, H, W)
        """
        with torch.no_grad():  # これがないとGPU不足でエラーでる
            noise_pred = self.forward(x, t)  # ノイズを予測
            beta_t = self.beta[t].view(-1, 1, 1, 1)  # タイムステップに対応するベータ値
            alpha_t = self.alpha[t].view(
                -1, 1, 1, 1
            )  # タイムステップに対応するアルファ値
            alpha_prod_t = self.alpha_prod[t].view(
                -1, 1, 1, 1
            )  # タイムステップに対応する累積アルファ値
            x = (x - (beta_t / torch.sqrt(1 - alpha_prod_t)) * noise_pred) / (
                torch.sqrt(alpha_t)
            )  # ノイズを除去
            if t[0].item() != 0:
                x = x + torch.randn_like(x) * torch.sqrt(
                    beta_t
                )  # 最初のタイムステップ以外の場合、追加ノイズを加える
        return x  # ノイズ除去された画像を返す

    def training_step(self, batch, batch_idx):
        """トレーニングの1ステップ.

        Args:
            batch (tuple): 入力バッチ
            batch_idx (int): バッチインデックス

        Returns:
            torch.Tensor: 損失
        """
        images = batch["images"]  # バッチから画像を取得
        images = images.to(self.device)  # デバイスに移動
        batch_size = images.size(0)  # バッチサイズを取得
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=self.device
        ).long()
        # random integer(最小、最大（含まない）、テンソルの形状) -> ランダムなタイムステップを生成
        # .long(): 生成されたテンソルを長整数型 (int64) に変換。タイムステップは整数である必要があるため、この変換が必要
        noise = torch.randn_like(images)  # 画像と同じサイズのランダムノイズを生成
        noisy_images = self.q_sample(images, t, noise)  # 画像にノイズを加える
        noise_pred = self.forward(noisy_images, t)  # ノイズを予測
        loss = self.criterion(noise_pred, noise)  # criterion(平均二乗誤差)で損失を計算
        self.log("train_loss", loss, prog_bar=True)  # 損失をログに記録
        return loss  # 損失を返す

    def generate(self, num_timesteps, shape):
        """拡散モデルからサンプルを生成."""
        x = torch.randn(shape).to(self.device)
        # tqdm は、ループの進行状況を表示するためのツール
        for t in tqdm(range(num_timesteps - 1, -1, -1)):
            # full(テンソルサイズ、埋める値) -> タイムステップの値を持つテンソルを生成
            t = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t)
        return x

    def generate_and_save_gif(self, images, filename):
        """画像のリストからGIFを生成し、保存する.

        Args:
            images (List[torch.Tensor]): 画像のリスト (各画像は (C, H, W) 形式)
            filename (str, optional): 保存するGIFのファイル名。デフォルトは 'output.gif'。
        """
        # PyTorchテンソルをPILイメージに変換し、値を[0, 255]にスケーリング
        images_pil = [
            Image.fromarray((img.permute(1, 2, 0) * 255).byte().cpu().numpy())
            for img in images
        ]

        # GIFを保存
        images_pil[0].save(
            filename, save_all=True, append_images=images_pil[1:], duration=100, loop=0
        )

    def on_train_epoch_end(self):
        """各エポック終了時に画像を生成."""
        if self.current_epoch % self.every_n_epochs == self.every_n_epochs - 1:
            logging.info("Generating Images...")
            generated_image = self.generate(
                self.num_timesteps,
                (self.num_samples[0] * self.num_samples[1],) + self.image_size,
            )
            # [-1, 1] の範囲から [0, 1] の範囲に変換
            generated_image = (generated_image + 1) / 2
            generated_image = (
                generated_image.reshape(
                    self.num_samples + self.image_size
                )  # (num_samples[0], num_samples[1], image_size[0], image_size[1], channels)
                .permute(
                    [2, 0, 3, 1, 4]
                )  # [image_size[0], num_samples[0], image_size[1], num_samples[1], channels]になる
                .flatten(-4, -3)  # 画像が横方向に連続
                .flatten(-2, -1)  # 画像が縦方向に連続
            )  # 画像を適切な形状に変換
            # add_image メソッドは、指定された画像をTensorBoardに追加
            self.logger.experiment.add_image(
                "Generated Images",
                generated_image,
                self.current_epoch,
                dataformats="HW" if generated_image.ndim == 2 else "CHW",
            )
            self.generated_images.append(generated_image)
            logging.info("Done.")

    def on_train_end(self):
        """訓練終了時に全ての生成画像をGIFとして保存."""
        logging.info("Saving generated images as a GIF...")
        # 画像のリストからGIFを生成し、保存
        self.generate_and_save_gif(self.generated_images, "final_generated_images.gif")


# HydraというPythonの設定管理ライブラリを使用して、設定ファイルから簡単に設定を読み込むためのもの
@hydra.main(config_path="conf", config_name="default.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """拡散モデルのトレーニング."""
    # シードを固定
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # 出力ディレクトリを取得
    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # データセットの準備
    train_dataset = load_dataset(
        "huggan/smithsonian_butterflies_subset", split="train", cache_dir=cfg.datadir
    )
    # 前処理
    preprocess = transforms.Compose(
        [
            transforms.Resize(cfg.plot.image_size[-2:]),  # サイズ変更
            transforms.ToTensor(),  # テンソルに変換
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] に正規化
        ]
    )

    def transform(examples):
        """データセットを変換."""
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    train_dataset.set_transform(transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )

    model = diffusers.UNet2DModel(**cfg.model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer)

    diffmodel = DiffusionModel(
        model,
        criterion,
        optimizer,
        **cfg.diffusion,
        **cfg.plot,
    )

    diffmodel.save_hyperparameters(cfg)

    # ロガーの設定
    tb_logger = TensorBoardLogger(outdir)
    # トレーニング
    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs, devices=1, logger=tb_logger, accelerator="gpu"
    )
    trainer.fit(diffmodel, train_loader)

    # モデルの保存
    torch.save(diffmodel.state_dict(), outdir + "/model.pth")


if __name__ == "__main__":
    main()
