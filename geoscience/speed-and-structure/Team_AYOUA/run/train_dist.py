import logging
import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from run.pl_model import PLModel

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False


def main(cfg: DictConfig, pl_model: type) -> Path:
    seed_everything(cfg.training.seed)
    out_dir = Path(cfg.out_dir).resolve()
    
    if cfg.test_model is not None:
        # Only run test with the given model weights
        is_test_mode = True
    else:
        # Run full training
        is_test_mode = False

    # init experiment logger
    if not cfg.training.use_wandb or is_test_mode:
        pl_logger = False
    else:
        pl_logger = WandbLogger(
            project=cfg.training.project_name,
            save_dir=str(out_dir),
            name=Path(out_dir).name,
        )

    # init lightning model
    model = pl_model(cfg)
    if cfg.training.load_from is not None:
        model.forwarder.model.load_state_dict(torch.load(cfg.training.load_from),strict=True)
        logger.info(f"Loaded model weights from {cfg.training.load_from}")
    os.makedirs(out_dir / "weights", exist_ok=True)
    # set callbacks
    if cfg.dataset.num_folds == -1:
        checkpoint_cb = ModelCheckpoint(
                dirpath=str(out_dir / "weights"),
                verbose=True,
                save_on_train_epoch_end=True,
                save_last=True,
            )
    else:
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(out_dir / "weights"),
            verbose=True,
            monitor=cfg.training.monitor,
            mode=cfg.training.monitor_mode,
            save_top_k=1,
            save_last=True,
        )

    # init trainer
    def _init_trainer(resume=True):
        
        return Trainer(
            # env
            devices=[0,1],
            accelerator="gpu",
            default_root_dir=str(out_dir),
            precision=16 if cfg.training.use_amp else 32,
            max_epochs=cfg.training.epoch,
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            sync_batchnorm=cfg.training.sync_batchnorm,
            callbacks=[checkpoint_cb],
            logger=pl_logger,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0 if is_test_mode else -1,
            strategy="ddp_find_unused_parameters_true"
            
        )

    trainer = _init_trainer()

    
    resume_from = cfg.training.resume_from
    if is_test_mode:
        trainer.test(model)
    else:
        if resume_from is not None:
            trainer.fit(model,ckpt_path=resume_from)
        else:
            trainer.fit(model)

        
        if trainer.global_rank == 0:
         
            # save ema weights
            weights_path = str(Path(checkpoint_cb.dirpath) / "model_weights_ema.pth")
            model.forwarder.ema.store()
            model.forwarder.ema.copy_to()
            logger.info(f"Extracting and saving EMA weights: {weights_path}")
            torch.save(model.forwarder.model.state_dict(), weights_path)
            model.forwarder.ema.restore()

    # return path to checkpoints directory
    if checkpoint_cb.dirpath is not None:
        return Path(checkpoint_cb.dirpath)


def prepare_env() -> None:
    # Disable PIL's debug logs
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    # move to original directory
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    # set PYTHONPATH if not set for possible launching of DDP processes
    os.environ.setdefault("PYTHONPATH", ".")


@hydra.main(config_path="conf", config_name="config")
def entry(cfg: DictConfig) -> None:
    
    prepare_env()
    main(cfg, PLModel)


if __name__ == "__main__":
    
    entry()
