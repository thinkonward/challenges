import glob
import random
import warnings
from logging import getLogger
from pathlib import Path
import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

# --- GLOBAL CONSTANTS FOR NORMALIZATION ---
# These are the global mean and std
GLOBAL_MEAN = -6.080667e-6
GLOBAL_STD = 0.01550384


class FWIDataset(Dataset):
    """
    Dataset class for Full Waveform Inversion (FWI) data.
    Handles data loading, optional spectrogram conversion, normalization, and coarse dropout augmentation.
    """

    @classmethod
    def create_dataframe(
        cls,
        num_folds: int = 5,
        seed: int = 7,
        num_records: int = 0,
        data_path: str = "data",
        meta_data_path: str = None,
    ) -> pd.DataFrame:
        logger.info(f"loading {data_path}")
        
        if meta_data_path is not None:
            df = pd.read_csv(meta_data_path)
            df["data_path"] = df["img_id"].map(lambda x: f"{data_path}/{x}_input.npy")
            df["anno_path"] = df["img_id"].map(lambda x: f"{data_path}/{x}_target.npy")
        else:
            
            df = pd.DataFrame({"data_path": glob.glob(f"{data_path}/*.npy")})
            df["img_id"] = df["data_path"].map(lambda x: Path(x).name[:-4])
        if num_folds > 0:
            n_splits = num_folds
            shuffle = True

            kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
            X = df["img_id"].values
            fold = -np.ones(len(df), dtype=int)
            for i, (_, indices) in enumerate(kfold.split(X)):
                fold[indices] = i

            df["fold"] = fold
        else:
            df["fold"] = -1

        if num_records:
            df = df.iloc[::num_records]

        return df

    def __init__(
        self,
        df: pd.DataFrame,
        phase="train",
        cfg=None,
    ) -> None:
        self.df = df.copy()
        self.ids = df['img_id']
        self.phase = phase
        self.cfg = cfg
        
        
            
        
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        
        _id = self.ids.iloc[index]
        seis_path = self.df.data_path.iloc[index]
        
        seis = np.load(seis_path)
        if seis is None:
            logger.warning(f"Failed to load seismic data from {seis_path}")
            return None
        
        # Default: global Z-score using pre-calculated global mean/std
        seis = (seis - GLOBAL_MEAN) / GLOBAL_STD
        
        seis = torch.from_numpy(seis).float()
        
        
        
            
        if self.phase == "test":
            res = {
                "image_id": _id,
                "image": seis,
                "label": -1,
            }
            return res
        
        vel_path = self.df.anno_path.iloc[index]
        vel = np.load(vel_path)
        vel = torch.from_numpy(vel).float()
        if self.cfg.get("filp", 0)>0:
            
            if np.random.random() < self.cfg["filp"]:
                x=x[::-1, :, ::-1]
                y= y[::-1, :]
        res = {
            "image_id": _id,
            "image": seis,
            "label": vel,
        }

        return res