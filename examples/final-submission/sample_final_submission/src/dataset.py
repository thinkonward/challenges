import os
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset


class VolumeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        volumes_dir = os.path.join(data_dir, "volumes")
        volume_filenames_list = os.listdir(volumes_dir)
        volume_filenames_list.sort()
        labels_dir = os.path.join(data_dir, "labels")
        label_filenames_list = os.listdir(labels_dir)
        label_filenames_list.sort()

        self.transform = transform
        self.seismic_list = list()
        self.labels_list = list()

        for volume_filename, label_filename in tqdm(
            zip(volume_filenames_list, label_filenames_list),
            total=len(volume_filenames_list),
            desc="Loading dataset",
        ):
            full_vol_path = os.path.join(volumes_dir, volume_filename)
            full_lb_path = os.path.join(labels_dir, label_filename)

            self.seismic_list.append(
                np.load(full_vol_path, mmap_mode="r", allow_pickle=True)
            )
            self.labels_list.append(
                np.load(full_lb_path, mmap_mode="r", allow_pickle=True)
            )

        self.volume_shape = self.seismic_list[0].shape

    def _convert_to_tensor(self, image, mask):
        image_dtype = torch.float32

        image = torch.from_numpy(np.array(image, dtype="float32")).type(image_dtype)
        mask = torch.from_numpy(np.array(mask, dtype="uint8")).type(torch.long)

        return image, mask

    def __len__(self):
        return len(self.seismic_list) * self.volume_shape[0]

    def __getitem__(self, index):
        volume_num = index // self.volume_shape[0]
        index_num = index % self.volume_shape[0]

        image = self.seismic_list[volume_num][index_num].T
        mask = np.array(self.labels_list[volume_num][index_num].T, dtype="long")

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return self._convert_to_tensor(image, mask)
