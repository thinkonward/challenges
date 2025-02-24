from mmengine.config import ConfigDict
from .dataset import TrainSliceDataset, ValDataset, Train_25D_Dataset
from .transform import rescale_volume, train_data_trainsform, infer_data_transform

def build_dataset(cfg: ConfigDict):
    if cfg.type == "TrainSliceDataset":
        dataset = TrainSliceDataset(cfg)
    if cfg.type == "Train_25D_Dataset":
        dataset = Train_25D_Dataset(cfg)
    elif cfg.type == "ValDataset":
        dataset = ValDataset(cfg)
    else:
        raise NotImplementedError("Only TrainSliceDataset and ValDataset supported now")
    return dataset

__all__ = [
    "TrainSliceDataset", "ValDataset", "build_dataset", "rescale_volume",
    "train_data_trainsform", "infer_data_transform", "Train_25D_Dataset"
]