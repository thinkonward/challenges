from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np
import glob
import os
from mmengine.config import ConfigDict
from natsort import natsorted
import pandas as pd
from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from .transform import rescale_volume, train_data_trainsform


class ValDataset:
    def __init__(
        self,
        cfg: ConfigDict,
    ):
        self.root_dir = cfg.root_dir
        self.txt_file = cfg.txt_file

        self._get_data_list()

    def _get_data_list(self,):
        self.volume_ids = np.loadtxt(self.txt_file, dtype=str).tolist()
        self.data_file_paths = []
        self.label_file_paths = []
        for volume_id in self.volume_ids:
            self.data_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismicCubes_RFC_fullstack_*.npy")[0])
            self.label_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/fault_segments_*.npy")[0])

    def _load_data(self, idx):
        # 这里只做数据加载和数据增强，shape转换和to_device在trainer层来做
        # 这里加载的是整个3D volume，若训练按照切片来，则在val时需要做切片
        volume = np.load(self.data_file_paths[idx], allow_pickle=True, mmap_mode="r+")
        label = np.load(self.label_file_paths[idx], allow_pickle=True, mmap_mode="r+")
        
        # volume = rescale_volume(volume)
        volume = volume.astype(np.float32)
        label = label.astype(np.float32)
        return {"volume" : volume, "label": label}

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, idx):
        return self._load_data(idx)


class TrainSliceDataset(ValDataset):
    def __init__(
        self,
        cfg: ConfigDict,
    ):
        super().__init__(cfg)

        self.num_slices = 300
        self._get_data_list()
        self._create_slice_menu()
        self._get_transforms()

    def _create_slice_menu(self):
        menu = pd.DataFrame({"volume": self.data_file_paths,
                             "label": self.label_file_paths})

        # TODO: train in 300 * 1259 slices
        menu["axis"] = [["h", "w"] for _ in range(len(self.data_file_paths))]
        menu["idx"] = [
            [str(s) for s in list(range(self.num_slices))]
            for _ in range(len(self.data_file_paths))
        ]
        menu = menu.explode("axis")
        menu = menu.explode("idx")
        menu = menu.reset_index(drop=True)
        self.slice_menu = menu

    def __len__(self):
        return len(self.slice_menu)
    
    def _get_transforms(self,):
        # baseline不用任何数据增强，且使用mmseg不需要保证图像尺寸是32的整数倍
        self.data_transform = A.Compose(train_data_trainsform)

    def _load_data(self, idx):
        # 300 * 300 * 1259
        volume = np.load(self.slice_menu["volume"][idx], allow_pickle=True, mmap_mode="r+")
        label = np.load(self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+")
        # volume = rescale_volume(volume)
        if self.slice_menu["axis"][idx] == "h":
            slice = volume[:, int(self.slice_menu["idx"][idx]), :]
            label = label[:, int(self.slice_menu["idx"][idx]), :]
        elif self.slice_menu["axis"][idx] == "w":
            slice = volume[int(self.slice_menu["idx"][idx]), ...]
            label = label[int(self.slice_menu["idx"][idx]), ...]

        # to HWC and tensor， trainer中只要to_device即可
        slice, label = slice.astype(np.float32), label.astype(np.float32)
        data = self.data_transform(image=slice, mask=label)
        slice, label = data["image"], data["mask"].unsqueeze(0)
        return {"slice" : slice, "label": label}
    


class Train_25D_Dataset(ValDataset):
    def __init__(
        self,
        cfg: ConfigDict,
    ):
        super().__init__(cfg)
        self.near_slice_shuffle = cfg.near_slice_shuffle
        self.num_slices = 300
        self._get_data_list()
        self._create_slice_menu()
        self._get_transforms()

    def _create_slice_menu(self):
        menu = pd.DataFrame({"volume": self.data_file_paths,
                             "label": self.label_file_paths})

        # TODO: train in 300 * 1259 slices
        menu["axis"] = [["h", "w"] for _ in range(len(self.data_file_paths))]
        menu["idx"] = [
            [str(s) for s in list(range(self.num_slices))]
            for _ in range(len(self.data_file_paths))
        ]
        menu = menu.explode("axis")
        menu = menu.explode("idx")
        menu = menu.reset_index(drop=True)
        self.slice_menu = menu

    def __len__(self):
        return len(self.slice_menu)
    
    def _get_transforms(self,):
        # baseline不用任何数据增强，且使用mmseg不需要保证图像尺寸是32的整数倍
        self.data_transform = A.Compose(train_data_trainsform)


    def _load_data(self, idx):
        # 300 * 300 * 1259
        volume = np.load(self.slice_menu["volume"][idx], allow_pickle=True, mmap_mode="r+")
        label = np.load(self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+")
        # volume = rescale_volume(volume)
        if self.slice_menu["axis"][idx] == "h":
            cur_slice = int(self.slice_menu["idx"][idx])
            if cur_slice == 0:
                slice = volume[:, [cur_slice, cur_slice, cur_slice + 1], :]
            elif cur_slice == (self.num_slices - 1):
                slice = volume[:, [cur_slice - 1, cur_slice, cur_slice], :]
            else:
                slice = volume[:, [cur_slice - 1, cur_slice, cur_slice + 1], :]
            slice = slice.transpose(0, 2, 1)
            label = label[:, cur_slice, :]
        elif self.slice_menu["axis"][idx] == "w":
            cur_slice = int(self.slice_menu["idx"][idx])
            if cur_slice == 0:
                slice = volume[[cur_slice, cur_slice, cur_slice + 1], :, :]
            elif cur_slice == (self.num_slices - 1):
                slice = volume[[cur_slice - 1, cur_slice, cur_slice], :, :]
            else:
                slice = volume[[cur_slice - 1, cur_slice, cur_slice + 1], :, :]
            slice = slice.transpose(1, 2, 0)
            label = label[cur_slice, :, :]

        # if self.near_slice_shuffle:
        #     if np.random.random() < 0.5:
        #         slice = slice[:, :, ::-1]

        # to HWC and tensor， trainer中只要to_device即可
        slice, label = slice.astype(np.float32), label.astype(np.float32)
        data = self.data_transform(image=slice, mask=label)
        slice, label = data["image"], data["mask"].unsqueeze(0)
        return {"slice" : slice, "label": label}

    # def _load_data(self, idx):
    #     # 300 * 300 * 1259
    #     volume = np.load(self.slice_menu["volume"][idx], allow_pickle=True, mmap_mode="r+")
    #     label = np.load(self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+")
    #     # volume = rescale_volume(volume)
    #     if self.slice_menu["axis"][idx] == "h":
    #         cur_slice = int(self.slice_menu["idx"][idx])
    #         if cur_slice == 0:
    #             slice = volume[:, [cur_slice, cur_slice, cur_slice, cur_slice + 1, cur_slice + 2], :]
    #         elif cur_slice == 1:
    #             slice = volume[:, [cur_slice-1, cur_slice-1, cur_slice, cur_slice + 1, cur_slice + 2], :]
    #         elif cur_slice == (self.num_slices - 2):
    #             slice = volume[:, [cur_slice - 2, cur_slice - 1, cur_slice, cur_slice+1, cur_slice+1], :]
    #         elif cur_slice == (self.num_slices - 1):
    #             slice = volume[:, [cur_slice-2, cur_slice - 1, cur_slice, cur_slice, cur_slice], :]
    #         else:
    #             slice = volume[:, [cur_slice - 2, cur_slice - 1, cur_slice, cur_slice+1, cur_slice+2], :]
    #         slice = slice.transpose(0, 2, 1)
    #         label = label[:, cur_slice, :]
    #     elif self.slice_menu["axis"][idx] == "w":
    #         cur_slice = int(self.slice_menu["idx"][idx])
    #         if cur_slice == 0:
    #             slice = volume[[cur_slice, cur_slice, cur_slice, cur_slice + 1, cur_slice + 2], :, :]
    #         elif cur_slice == 1:
    #             slice = volume[[cur_slice-1, cur_slice-1, cur_slice, cur_slice + 1, cur_slice + 2], :, :]
    #         elif cur_slice == (self.num_slices - 2):
    #             slice = volume[[cur_slice - 2, cur_slice - 1, cur_slice, cur_slice+1, cur_slice+1], :, :]
    #         elif cur_slice == (self.num_slices - 1):
    #             slice = volume[[cur_slice-2, cur_slice - 1, cur_slice, cur_slice, cur_slice], :, :]
    #         else:
    #             slice = volume[[cur_slice - 2, cur_slice - 1, cur_slice, cur_slice+1, cur_slice+2], :, :]
    #         slice = slice.transpose(1, 2, 0)
    #         label = label[cur_slice, :, :]

    #     # to HWC and tensor， trainer中只要to_device即可
    #     slice, label = slice.astype(np.float32), label.astype(np.float32)
    #     data = self.data_transform(image=slice, mask=label)
    #     slice, label = data["image"], data["mask"].unsqueeze(0)
    #     return {"slice" : slice, "label": label}

    # def _load_data(self, idx):
    #     # 300 * 300 * 1259
    #     volume = np.load(self.slice_menu["volume"][idx], allow_pickle=True, mmap_mode="r+")
    #     label = np.load(self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+")
    #     # volume = rescale_volume(volume)
    #     if self.slice_menu["axis"][idx] == "h":
    #         cur_slice = int(self.slice_menu["idx"][idx])
    #         idx_list = self._gen_slice_idx_list(cur_slice)
    #         slice = volume[:, idx_list, :]
    #         slice = slice.transpose(0, 2, 1)
    #         label = label[:, cur_slice, :]
    #     elif self.slice_menu["axis"][idx] == "w":
    #         cur_slice = int(self.slice_menu["idx"][idx])
    #         idx_list = self._gen_slice_idx_list(cur_slice)
    #         slice = volume[idx_list, :, :]
    #         slice = slice.transpose(1, 2, 0)
    #         label = label[cur_slice, :, :]

    #     # to HWC and tensor， trainer中只要to_device即可
    #     slice, label = slice.astype(np.float32), label.astype(np.float32)
    #     data = self.data_transform(image=slice, mask=label)
    #     slice, label = data["image"], data["mask"].unsqueeze(0)
    #     return {"slice" : slice, "label": label}


# class TrainHWZSliceDataset(ValDataset):
#     def __init__(
#         self,
#         cfg: ConfigDict,
#     ):
#         super().__init__(cfg)

#         self.num_slices = 300
#         self._get_data_list()
#         self._create_slice_menu()
#         self._get_transforms()

#     def _create_slice_menu(self):
#         menu = pd.DataFrame({"volume": self.data_file_paths,
#                              "label": self.label_file_paths})

#         # TODO: train in 300 * 1259 slices
#         menu["axis"] = [["h", "w", "z"] for _ in range(len(self.data_file_paths))]
#         menu["idx"] = [
#             [str(s) for s in list(range(self.num_slices))]
#             for _ in range(len(self.data_file_paths))
#         ]
#         menu = menu.explode("axis")
#         menu = menu.explode("idx")
#         menu = menu.reset_index(drop=True)
#         self.slice_menu = menu

#     def __len__(self):
#         return len(self.slice_menu)
    
#     def _get_transforms(self,):
#         # baseline不用任何数据增强，且使用mmseg不需要保证图像尺寸是32的整数倍
#         self.data_transform = A.Compose(train_data_trainsform)

#     def _load_data(self, idx):
#         # 300 * 300 * 1259
#         volume = np.load(self.slice_menu["volume"][idx], allow_pickle=True, mmap_mode="r+")
#         label = np.load(self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+")
#         # volume = rescale_volume(volume)
#         if self.slice_menu["axis"][idx] == "h":
#             slice = volume[:, int(self.slice_menu["idx"][idx]), :]
#             label = label[:, int(self.slice_menu["idx"][idx]), :]
#         elif self.slice_menu["axis"][idx] == "w":
#             slice = volume[int(self.slice_menu["idx"][idx]), ...]
#             label = label[int(self.slice_menu["idx"][idx]), ...]

#         # to HWC and tensor， trainer中只要to_device即可
#         slice, label = slice.astype(np.float32), label.astype(np.float32)
#         data = self.data_transform(image=slice, mask=label)
#         slice, label = data["image"], data["mask"].unsqueeze(0)
#         return {"slice" : slice, "label": label}


