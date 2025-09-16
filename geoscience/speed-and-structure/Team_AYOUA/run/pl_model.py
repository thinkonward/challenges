from logging import getLogger
from pathlib import Path
from typing import Dict, List, Literal

import torch
import torchvision.transforms.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader

from run.init.dataset import init_datasets_from_config
from run.init.forwarder import Forwarder
from run.init.model import init_model_from_config
from run.init.optimizer import init_optimizer_from_config

from run.init.scheduler import init_scheduler_from_config

from torchmetrics import MeanAbsolutePercentageError
logger = getLogger(__name__)



class PLModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg.copy()
        self.save_embed = self.cfg.training.save_embed
       
        self.mape = MeanAbsolutePercentageError()
        
        logger.info("creating model")
        model = init_model_from_config(cfg.model)
        self.forwarder = Forwarder(cfg.forwarder, model)
        
        logger.info("loading metadata")
        raw_datasets = init_datasets_from_config(cfg.dataset)        
        self.datasets = raw_datasets   
        
        bs_per_gpu   = cfg.training.batch_size 
        grad_accum   = cfg.training.accumulate_grad_batches
        self.cfg.scheduler.num_steps_per_epoch = (
            len(self.datasets["train"]) // (bs_per_gpu * grad_accum)
        )
        logger.info(
            f"number of training samples: {len(self.datasets["train"])}"
        )
        logger.info(
            f"training steps per epoch: {self.cfg.scheduler.num_steps_per_epoch}"
        )
    def on_train_epoch_start(self):
        pass

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        additional_info = {}
       
        _, loss = self.forwarder.forward(
            batch, phase="train", epoch=self.current_epoch, **additional_info
        )

        self.log(
            "train/loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "lr",
            self.lr_schedulers().get_last_lr()[0], 
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )

        return loss
    

    def _evaluation_step2(self, batch: Dict[str, Tensor], phase: Literal["val", "test"]):
        preds, loss = self.forwarder.forward(
            batch, phase=phase, epoch=self.current_epoch
        )
        
        v_true = batch["label"]
        self.mape(preds, v_true)
        self.log_dict(
            {
                f"{phase}/loss": loss,
                f"{phase}/mape": self.mape,  # Automatically handles epoch aggregation
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        output = {
            "loss": loss,
            
        }
        return output
    
    
    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) :
       
        return self._evaluation_step2(batch, phase="val")
            
    

    
    
    
    def _end_process2(self,  phase: str="val"):
        
        if phase != "test" and self.trainer.global_rank == 0:
            weights_filepath = Path(self.cfg.out_dir) / "weights"
            if not weights_filepath.exists():
                weights_filepath.mkdir(exist_ok=True)
            weights_path = str(weights_filepath / "model_weights.pth")
            logger.info(f"Extracting and saving weights: {weights_path}")
            torch.save(self.forwarder.model.state_dict(), weights_path)
       

    def on_validation_epoch_end(self) -> None:
       
        self._end_process2( "val")
 

   

    def configure_optimizers(self):
        model = self.forwarder.model
       

        optimizer = init_optimizer_from_config(self.cfg.optimizer, model, return_cls=False)
        scheduler = init_scheduler_from_config(self.cfg.scheduler, optimizer)

        if scheduler is None:
            return [optimizer]
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1,     
                "monitor": "val_loss" if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None,
                "strict": True, 
            },
            }
       

    def on_before_zero_grad(self, *args, **kwargs):
        self.forwarder.ema.update(self.forwarder.model.parameters())

    def _dataloader(self, phase: str) -> DataLoader:
        logger.info(f"{phase} data loader called")
        dataset = self.datasets[phase]

        batch_size = self.cfg.training.batch_size
        num_workers = self.cfg.training.num_workers

        num_gpus = self.cfg.training.num_gpus
        if phase != "train":
            batch_size = self.cfg.training.batch_size_test
        batch_size //= num_gpus
        num_workers //= num_gpus

        drop_last = True if self.cfg.training.drop_last and phase == "train" else False
        shuffle = phase == "train"

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(phase="train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(phase="val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(phase="test")
