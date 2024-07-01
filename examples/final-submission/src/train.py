import os
import sys
import yaml
from typing import Any
import numpy as np
from time import gmtime, strftime

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from utils import get_model, get_loss, get_scheduler, get_metric
from dataset import VolumeDataset


class SegmentationModule(pl.LightningModule):
    """
    Main training module that defines model archirecture, loss function, metric,
    optimizer, constructs training and validation dataloaders.
    """

    def __init__(self, config: Any, mode="train"):
        super().__init__()
        self.validation_step_outputs = []
        self.hparams.update(config)
        self.save_hyperparameters(config)

        self.net = get_model(**self.hparams["model"])
        self.loss = get_loss(**self.hparams["loss"])
        self.metric = get_metric(**self.hparams["metric"])

        val_transform = A.Compose(
            [
                A.Lambda(image=lambda img, **kwargs: img.astype(np.float32) / 255.0),
                A.Resize(128, 320),
                A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )

        train_transform = A.Compose(
            [
                A.Lambda(image=lambda img, **kwargs: img.astype(np.float32) / 255.0),
                A.Resize(128, 320),
                A.Blur(blur_limit=10, p=0.5),
                A.HorizontalFlip(),
                A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )

        data_path_train = config["data_paths"]["train_data_dir"]
        data_path_val = config["data_paths"]["val_data_dir"]

        if mode == "train":
            val_dataset = VolumeDataset(data_path_val, val_transform)
            train_dataset = VolumeDataset(data_path_train, train_transform)
            self.train_data = train_dataset
            print("Train dataset", len(train_dataset))
            self.val_data = val_dataset
            print("Val dataset", len(val_dataset))
        else:
            val_dataset = VolumeDataset(data_path_val, val_transform)
            self.val_data = val_dataset

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        predict = self.forward(image)

        loss = self.loss(predict, mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        predict = self.forward(image)

        val_loss = self.loss(predict, mask)
        val_metric = self.metric(predict, mask)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False)

        output = {"val_loss": val_loss, "val_metric": val_metric}

        self.validation_step_outputs.append(output)

        # return output

    def on_validation_epoch_end(self):
        val_loss = 0
        val_metric = 0

        outputs = self.validation_step_outputs

        for output in outputs:
            val_loss += output["val_loss"]
            val_metric += output["val_metric"]

        val_loss /= len(outputs)
        val_metric /= len(outputs)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_metric", val_metric, on_step=False, on_epoch=True, prog_bar=False)

        output = {"val_loss": val_loss, "val_metric": val_metric}
        self.validation_step_outputs.clear()

        # return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams["lr"])
        scheduler = get_scheduler(optimizer, **self.hparams["scheduler"])
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
        )


def main():
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = SegmentationModule(config)
    time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    checkpoint_dirpath = os.path.join(
        "checkpoints/",
        config["name"] + "_" + time,
    )
    os.mkdir(checkpoint_dirpath)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath, **config["model_checkpoint"]
    )

    trainer = pl.Trainer(
        logger=model.logger,
        precision=32,
        accelerator="gpu",
        devices=1,
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback],
        log_every_n_steps=2,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
