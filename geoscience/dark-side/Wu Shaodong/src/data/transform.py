import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np



def rescale_volume(seismic, low=0, high=100):
    """
    Rescaling 3D seismic volumes 0-255 range, clipping values between low and high percentiles
    """
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255

    return seismic

infer_data_transform = [
                        A.Normalize(mean=0.485, std=0.229, normalization="standard", p=1.0),
                        ToTensorV2(p=1.0)
]

train_data_trainsform = [
                        # 
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.Normalize(mean=0.485, std=0.229, normalization="standard", p=1.0),
                        ToTensorV2(p=1.0)
]