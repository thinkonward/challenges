import os
import numpy as np
import warnings

import torch
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

# Suppress warnings
warnings.simplefilter("ignore")


def joint_augment(image, mask, mode='train'):
    """
    Applies joint and separate augmentations to image and mask.

    Args:
        image (np.ndarray): shape (5, 10000, 32)
        mask  (np.ndarray): shape (300, 1259)
        mode  (str): 'train' or 'val'/'test'

    Returns:
        Tuple of (augmented image, augmented mask)
    """
    if mode != 'train':
        return image, mask

    # 1.Joint horizontal reversal (image & velocity)
    if np.random.random() < 0.5:
        image = image[:, :, ::-1]
        mask =  mask[::-1, ...]

    return image, mask

class SeismicDataset(Dataset):
    """
    PyTorch Dataset for loading seismic data and their corresponding seismic velocity models.

    This dataset class handles:
    - Loading seismic input data from multiple source coordinates as numpy arrays.
    - Preprocessing seismic data by removing multiples (scaling).
    - Loading corresponding seismic velocity models (velocity masks) for train/validation modes.
    - Applying optional joint augmentations to input data and velocity models.
    - Resizing seismic data and velocity models to fixed dimensions required by the model.
    - Supporting separate logic for 'test' mode where velocity models are not available.

    Args:
        data (pd.DataFrame): DataFrame containing metadata such as image paths and IDs.
        mode (str): One of 'train', 'val', or 'test'. Controls velocity model loading and return format.
        transform (callable, optional): A function to perform joint augmentation on seismic data and velocity models.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(index): Returns a sample (seismic_data, velocity_model) tuple or dict (seismic_data and metadata) for testing.

    Returns:
        - In 'train'/'val' mode: Tuple (seismic_data_tensor, velocity_model_tensor)
            - image(seismic_data_tensor): FloatTensor of seismic input data resized to (5, 10000, 31).
            - mask (velocity_model_tensor): FloatTensor of corresponding seismic velocity model (300, 1259).
        - In 'test' mode: Dictionary with keys:
            - 'image': FloatTensor seismic input data resized to (5, 10000, 31).
            - 'id': String image identifier.
    """

    def __init__(self, data, mode, transform=None):
        self.data = data
        self.mode = mode
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, item):
        """
        Retrieves a sample (image and mask if applicable) for the given index.

        Args:
            item (int): Index of the sample to retrieve.

        Returns:
            - In 'test' mode: A dictionary with keys:
                - `image`: Transformed image.
                - `id`: Image ID without file extension.
            - In 'train'/'val' mode: A tuple (`image`, `mask`).
                - `image`: Transformed image.
                - `mask`: Transformed segmentation mask.
        """

        record = self.data.iloc[item]
        image_path = record["image_path"]
        image_id = record["image_id"]

        # Load input data from multiple source coordinates
        source_coordinates = [1, 75, 150, 225, 300]
        image = [
            np.load(os.path.join(image_path, f"receiver_data_src_{i}.npy"))
            for i in source_coordinates
        ]

        # Preprocessing: remove multiples (scale data)
        image = np.stack(image) / 1e-3

        if self.mode == 'test':
            image = image.copy()
            image = torch.FloatTensor(image)

            image = F.interpolate(image.unsqueeze(0), size=(10000, 31), mode="area").squeeze(0)
            return {"image": image, "id": image_id}

        else:
            # Load mask (seismic velocity model)
            mask_path = os.path.join(image_path, "vp_model.npy")
            mask = np.load(mask_path)

            if self.transform:
                # Apply augmentations jointly to image and mask
                image, mask = joint_augment(image, mask, mode=self.mode)

            image = image.copy()
            mask = mask.copy()

            image = torch.FloatTensor(image)
            mask = torch.FloatTensor(mask)

            image = F.interpolate(image.unsqueeze(0), size=(10000, 31), mode="area").squeeze(0)

            return image, mask

            


    