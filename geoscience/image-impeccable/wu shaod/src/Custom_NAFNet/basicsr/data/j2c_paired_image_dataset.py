# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import numpy as np
import glob
import os
from natsort import natsorted
import pandas as pd


# diagonal_elements = array[np.arange(300), np.arange(300)]
# low=2, high=98
# low=0, high=100
def rescale_volume(seismic, low=0, high=100):
    """
    Rescaling 3D seismic volumes 0-255 range, clipping values between low and high percentiles
    """
    import time
    st = time.time()
    # minval = np.percentile(seismic, low)
    # maxval = np.percentile(seismic, high)
    minval = seismic.min()
    maxval = seismic.max()
    ed = time.time()
    print("percentile", ed-st)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255
    print("clip", time.time()-ed)
    return seismic

# 77692237
class SliceServer:
    def __init__(
        self,
        root_dir,
        txt_file,
        num_slices,
        train_flag=False,
    ):
        self.root_dir = root_dir
        self.num_slices = num_slices
        self.txt_file = txt_file
        self.train_flag = train_flag

        # Gather volume IDs
        self.volume_ids = np.loadtxt(self.txt_file, dtype=str).tolist()
        self.data_file_paths = []
        self.label_file_paths = []
        for volume_id in self.volume_ids:
            data_files = natsorted(os.listdir(f"{self.root_dir}/{volume_id}/"))

            self.data_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismic_w_noise_*.npy")[0])
            self.label_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismicCubes_RFC_fullstack_*.npy")[0])

        # Create the dataframe for slices
        self.dataframe = self.create_slice_menu()

    def create_slice_menu(self):
        # Create a dataframe with matched data and label paths
        menu = pd.DataFrame({"data": self.data_file_paths,
                             "label": self.label_file_paths})

        # Add axis info for inline and crossline
        # TODO: train in 1259 * 300 slices
        menu["axis"] = [["i", "x"] for _ in range(len(self.data_file_paths))]

        # [i, :, :], [:, i, :],  [0.25, 0.5, 0.75]
        # Add all possible slices in the volume
        if not self.train_flag:
            menu["idx"] = [
                [str(s) for s in list((np.array([0.25, 0.5, 0.75]) * self.num_slices).astype(np.int32))]
                for _ in range(len(self.data_file_paths))
            ]
        else:
            menu["idx"] = [
                [str(s) for s in list(range(self.num_slices))]
                for _ in range(len(self.data_file_paths))
            ]

        # Explode the dataframe to create dataset for each slice
        menu = menu.explode("axis")
        menu = menu.explode("idx")
        menu = menu.reset_index(drop=True)

        return menu

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, column):
        return self.dataframe[column]


# class SliceServer:
#     def __init__(
#         self,
#         root_dir,
#         txt_file,
#         num_slices,
#         train_flag=False,
#     ):
#         self.root_dir = root_dir
#         self.num_slices = num_slices
#         self.txt_file = txt_file
#         self.train_flag = train_flag

#         # Gather volume IDs
#         self.volume_ids = np.loadtxt(self.txt_file, dtype=str).tolist()
#         self.data_file_paths = []
#         self.label_file_paths = []
#         for volume_id in self.volume_ids:
#             data_files = natsorted(os.listdir(f"{self.root_dir}/{volume_id}/"))

#             self.data_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismic_w_noise_*.npy")[0])
#             self.label_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismicCubes_RFC_fullstack_*.npy")[0])

#         # Create the dataframe for slices
#         self.dataframe = self.create_slice_menu()

#     def create_slice_menu(self):
#         # Create a dataframe with matched data and label paths
#         menu = pd.DataFrame({"data": self.data_file_paths,
#                              "label": self.label_file_paths})

#         # Add axis info for inline and crossline
#         # TODO: train in 1259 * 300 slices
#         if not self.train_flag:
#             menu["axis"] = [["i", "x"] for _ in range(len(self.data_file_paths))]
#         else:
#             menu["axis"] = [["i", "x", "s1", "s2", "s3", "s4"] for _ in range(len(self.data_file_paths))]

#         # [i, :, :], [:, i, :],  [0.25, 0.5, 0.75]
#         # Add all possible slices in the volume
#         if not self.train_flag:
#             menu["idx"] = [
#                 [str(s) for s in list((np.array([0.25, 0.5, 0.75]) * self.num_slices).astype(np.int32))]
#                 for _ in range(len(self.data_file_paths))
#             ]
#         else:
#             menu["idx"] = [
#                 [str(s) for s in list(range(self.num_slices))]
#                 for _ in range(len(self.data_file_paths))
#             ]

#         # Explode the dataframe to create dataset for each slice
#         menu = menu.explode("axis")
#         menu = menu.explode("idx")
#         menu = menu.reset_index(drop=True)

#         #删除小于256的斜行
#         menu = menu.drop(menu[((menu['axis']=="s1") & (menu['idx'].astype(int)<256)) | 
#                               ((menu['axis']=="s2") & (menu['idx'].astype(int)<256)) |
#                               ((menu['axis']=="s3") & (menu['idx'].astype(int)<256)) |
#                               ((menu['axis']=="s4") & (menu['idx'].astype(int)<256))].index)
#         menu = menu.reset_index(drop=True)
#         return menu

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, column):
#         return self.dataframe[column]


# class SliceServer:
#     def __init__(
#         self,
#         root_dir,
#         txt_file,
#         num_slices,
#         train_flag=False,
#     ):
#         self.root_dir = root_dir
#         self.num_slices = num_slices
#         self.txt_file = txt_file
#         self.train_flag = train_flag

#         # Gather volume IDs
#         self.volume_ids = np.loadtxt(self.txt_file, dtype=str).tolist()
#         self.data_file_paths = []
#         self.label_file_paths = []
#         for volume_id in self.volume_ids:
#             data_files = natsorted(os.listdir(f"{self.root_dir}/{volume_id}/"))

#             self.data_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismic_w_noise_*.npy")[0])
#             self.label_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismicCubes_RFC_fullstack_*.npy")[0])

#         # Create the dataframe for slices
#         self.dataframe = self.create_slice_menu()

#     def create_slice_menu(self):
#         # Create a dataframe with matched data and label paths
#         menu = pd.DataFrame({"data": self.data_file_paths,
#                              "label": self.label_file_paths})

#         # Add axis info for inline and crossline
#         # TODO: train in 1259 * 300 slices
#         if not self.train_flag:
#             menu["axis"] = [["i", "x"] for _ in range(len(self.data_file_paths))]
#         else:
#             menu["axis"] = [["i", "x", "y"] for _ in range(len(self.data_file_paths))]

#         # [i, :, :], [:, i, :],  [0.25, 0.5, 0.75]
#         # Add all possible slices in the volume
#         if not self.train_flag:
#             menu["idx"] = [
#                 [str(s) for s in list((np.array([0.25, 0.5, 0.75]) * self.num_slices).astype(np.int32))]
#                 for _ in range(len(self.data_file_paths))
#             ]
#         else:
#             menu["idx"] = [
#                 [str(s) for s in list(range(self.num_slices))]
#                 for _ in range(len(self.data_file_paths))
#             ]

#         # Explode the dataframe to create dataset for each slice
#         menu = menu.explode("axis")
#         menu = menu.explode("idx")
#         menu = menu.reset_index(drop=True)

#         return menu

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, column):
#         return self.dataframe[column]


# class SliceServer:
#     def __init__(
#         self,
#         root_dir,
#         txt_file,
#         num_slices,
#         train_flag=False,
#     ):
#         self.root_dir = root_dir
#         self.num_slices = num_slices
#         self.txt_file = txt_file
#         self.train_flag = train_flag

#         # Gather volume IDs
#         self.volume_ids = np.loadtxt(self.txt_file, dtype=str).tolist()
#         self.data_file_paths = []
#         self.label_file_paths = []
#         for volume_id in self.volume_ids:
#             data_files = natsorted(os.listdir(f"{self.root_dir}/{volume_id}/"))

#             self.data_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismic_w_noise_*.npy")[0])
#             self.label_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismicCubes_RFC_fullstack_*.npy")[0])

#         # Create the dataframe for slices
#         self.dataframe = self.create_slice_menu()

#     def create_slice_menu(self):
#         # Create a dataframe with matched data and label paths
#         menu = pd.DataFrame({"data": self.data_file_paths,
#                              "label": self.label_file_paths})

#         # Add axis info for inline and crossline
#         # TODO: train in 1259 * 300 slices
#         if not self.train_flag:
#             menu["axis"] = [["i", "x"] for _ in range(len(self.data_file_paths))]
#         else:
#             menu["axis"] = [["y"] for _ in range(len(self.data_file_paths))]

#         # [i, :, :], [:, i, :],  [0.25, 0.5, 0.75]
#         # Add all possible slices in the volume
#         if not self.train_flag:
#             menu["idx"] = [
#                 [str(s) for s in list((np.array([0.25, 0.5, 0.75]) * self.num_slices).astype(np.int32))]
#                 for _ in range(len(self.data_file_paths))
#             ]
#         else:
#             menu["idx"] = [
#                 [str(s) for s in list(range(1259))]
#                 for _ in range(len(self.data_file_paths))
#             ]

#         # Explode the dataframe to create dataset for each slice
#         menu = menu.explode("axis")
#         menu = menu.explode("idx")
#         menu = menu.reset_index(drop=True)

#         return menu

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, column):
#         return self.dataframe[column]


class CustomDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        txt_file,
        num_slices=300,
        train_flag=False
    ):
        self.root_dir = root_dir
        self.num_slices = num_slices
        self.txt_file = txt_file
        self.train_flag=train_flag
        self.slice_menu = SliceServer(
            self.root_dir,
            self.txt_file,
            self.num_slices,
            self.train_flag
        )

    def __len__(self):
        return len(self.slice_menu)

    def __getitem__(self, idx):
        # adjust label shape
        # import time
        # st = time.time()
        data = np.load(self.slice_menu["data"][idx], allow_pickle=True, mmap_mode="r+")
        label = np.load(self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+")
        # data = rescale_volume(data)
        # label = rescale_volume(label)
        # ed = time.time()
        # print(ed-st)
        # if data.shape != label.shape:
        #     label = label.T
        # # trans to shape (300, 300, 1259)
        # if data.shape[0] == 1259:
        #     data = data.transpose(1, 2, 0)
        #     label = label.transpose(1, 2, 0)
        # elif data.shape[1] == 1259:
        #     data = data.transpose(0, 2, 1)
        #     label = label.transpose(0, 2, 1)

        if self.slice_menu["axis"][idx] == "i":
            data = data[..., int(self.slice_menu["idx"][idx])]
            label = label[..., int(self.slice_menu["idx"][idx])]
        elif self.slice_menu["axis"][idx] == "x":
            data = data[:, int(self.slice_menu["idx"][idx]), :]
            label = label[:, int(self.slice_menu["idx"][idx]), :]
        elif self.slice_menu["axis"][idx] == "s1":
            iii = int(self.slice_menu["idx"][idx])
            x_idx = np.arange(0, iii)
            y_idx = -x_idx[::-1] - 1
            data = data[:, x_idx, y_idx]
            label = label[:, x_idx, y_idx]
        elif self.slice_menu["axis"][idx] == "s2":
            iii = int(self.slice_menu["idx"][idx])
            y_idx = np.arange(0, iii)
            x_idx = -y_idx[::-1] - 1
            data = data[:, x_idx, y_idx]
            label = label[:, x_idx, y_idx]
        elif self.slice_menu["axis"][idx] == "s3":
            iii = int(self.slice_menu["idx"][idx])
            x_idx = np.arange(-iii, 0)
            y_idx = x_idx
            data = data[:, x_idx, y_idx]
            label = label[:, x_idx, y_idx]
        elif self.slice_menu["axis"][idx] == "s4":
            iii = int(self.slice_menu["idx"][idx])
            x_idx = np.arange(0, iii)
            y_idx = x_idx[::-1]
            data = data[:, x_idx, y_idx]
            label = label[:, x_idx, y_idx]
        else:# 300 * 300
            # iii = int((float(self.slice_menu["idx"][idx]) / 300) * 1259)
            data =data[int(self.slice_menu["idx"][idx]), ...]
            label = label[int(self.slice_menu["idx"][idx]), ...]

        # data = rescale_volume(data)
        # label = rescale_volume(label)

        # HWC
        data = data[:, :, np.newaxis]
        label = label[:, :, np.newaxis]
        # data = torch.from_numpy(data).long()
        # label = torch.from_numpy(label).long()

        return self.slice_menu["data"][idx], self.slice_menu["label"][idx], data, label


# class CustomDataset(data.Dataset):
#     def __init__(
#         self,
#         root_dir,
#         txt_file,
#         num_slices=300,
#         train_flag=False
#     ):
#         self.root_dir = root_dir
#         self.num_slices = num_slices
#         self.txt_file = txt_file
#         self.train_flag=train_flag
#         self.slice_menu = SliceServer(
#             self.root_dir,
#             self.txt_file,
#             self.num_slices,
#             self.train_flag
#         )

#     def __len__(self):
#         return len(self.slice_menu)

#     def __getitem__(self, idx):
#         # adjust label shape
#         # import time
#         # st = time.time()
#         data = np.load(self.slice_menu["data"][idx], allow_pickle=True, mmap_mode="r+")
#         label = np.load(self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+")
#         # data = rescale_volume(data)
#         # label = rescale_volume(label)
#         # ed = time.time()
#         # print(ed-st)
#         # if data.shape != label.shape:
#         #     label = label.T
#         # # trans to shape (300, 300, 1259)
#         # if data.shape[0] == 1259:
#         #     data = data.transpose(1, 2, 0)
#         #     label = label.transpose(1, 2, 0)
#         # elif data.shape[1] == 1259:
#         #     data = data.transpose(0, 2, 1)
#         #     label = label.transpose(0, 2, 1)

#         if self.slice_menu["axis"][idx] == "i":
#             cur_i = int(self.slice_menu["idx"][idx])
#             if cur_i == 0: cur_i += 1
#             elif cur_i == 299: cur_i -= 1
#             data = data[..., cur_i - 1 : cur_i + 2]
#             label = label[..., cur_i - 1 : cur_i + 2]
#         elif self.slice_menu["axis"][idx] == "x":
#             cur_i = int(self.slice_menu["idx"][idx])
#             if cur_i == 0: cur_i += 1
#             elif cur_i == 299: cur_i -= 1
#             data = data[:, cur_i - 1 : cur_i + 2, :]
#             label = label[:, cur_i - 1 : cur_i + 2, :]
#             data = data.transpose(0, 2, 1)
#             label = label.transpose(0, 2, 1)  # 1259, 3, 300   to    1259, 300, 3
#         else:# 300 * 300
#             # iii = int((float(self.slice_menu["idx"][idx]) / 300) * 1259)
#             data =data[int(self.slice_menu["idx"][idx]), ...]
#             label = label[int(self.slice_menu["idx"][idx]), ...]

#         # data = rescale_volume(data)
#         # label = rescale_volume(label)

#         # HWC
#         # data = data[:, :, np.newaxis]
#         # label = label[:, :, np.newaxis]
#         # data = torch.from_numpy(data).long()
#         # label = torch.from_numpy(label).long()

#         return self.slice_menu["data"][idx], self.slice_menu["label"][idx], data, label

class J2CPairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(J2CPairedImageDataset, self).__init__()
        self.opt = opt
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.root_dir = opt["root_dir"]
        self.txt_file = opt["txt_file"]
        self.num_slices = opt["num_slices"]
        self.train_flag=(self.opt['phase'] == 'train')
        self.custom_dataset = CustomDataset(root_dir=self.root_dir,
                                            txt_file=self.txt_file,
                                            num_slices=self.num_slices,
                                            train_flag=self.train_flag)

    def __getitem__(self, index):
        scale = self.opt['scale']
        lq_path, gt_path, img_lq, img_gt = self.custom_dataset[index]
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']  # h, w  [512, 300]
            # padding
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'], vflip=self.opt['use_vflip'])

        # TODO: color space transform
        # TODO: 3D data transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.custom_dataset)
