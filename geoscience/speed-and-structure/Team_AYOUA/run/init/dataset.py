from logging import getLogger
from typing import Dict

import pandas as pd
from omegaconf import DictConfig

from src.datasets.fwi import FWIDataset
 

logger = getLogger(__name__)


def init_datasets_from_config(cfg: DictConfig):
    if cfg.type == "fwi":
        datasets = get_fwi_dataset(
            num_folds=cfg.num_folds,
            test_fold=cfg.test_fold,
            val_fold=cfg.val_fold,
            seed=cfg.seed,
            num_records=cfg.num_records,
            phase=cfg.phase,
            cfg=cfg,
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {cfg.type}")

    return datasets


def get_fwi_dataset(
    num_folds: int,
    test_fold: int,
    val_fold: int,
    seed: int = 7,
    num_records: int = 0,
    phase: str = "train",
    cfg=None,
) -> Dict[str, FWIDataset]:
    logger.info("creating dataframe...")
    df = FWIDataset.create_dataframe(
        num_folds,
        seed,
        num_records,
        data_path=f"data/prep_train",
        meta_data_path=cfg.meta_path,
    )

    test_df = FWIDataset.create_dataframe(
        -1,
        seed,
        num_records,
        data_path="data/test",
    ).sort_values(by="img_id")
    
    train_df = df[(df["fold"] != val_fold) & (df["fold"] != test_fold)]
   
    val_df = df[df["fold"] == val_fold]
    train_dataset = FWIDataset(train_df, phase="train", cfg=cfg)
    val_dataset = FWIDataset(val_df, phase="train", cfg=cfg)
    test_dataset = FWIDataset(test_df, phase="test", cfg=cfg)
    
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset,
                }
    return datasets
