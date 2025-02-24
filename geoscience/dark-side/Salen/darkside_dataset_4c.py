import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler
#from volumentations import *
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
#from skimage.metrics import structural_similarity as ssim
import os
import glob
import random
import numpy as np
#import pandas as pd
import re
#import monai
#import scipy.ndimage

def downsample_volume(volume, target_size, order=1):
    # Calculate the zoom factors for each dimension
    zoom_factors = [n / o for n, o in zip(target_size, volume.shape)]
    # Apply the downsampling
    return scipy.ndimage.zoom(volume, zoom_factors, order=order)  # order=1 for linear interpolation

def rand_bbox_3d(patch_size=(128, 128, 128)):
    
    x_min = random.randint(0, 300-patch_size[0])
    y_min = random.randint(0, 300-patch_size[1])
    z_min = random.randint(0, 1259-patch_size[2])

    x_max = x_min + patch_size[0]
    y_max = y_min + patch_size[1]
    z_max = z_min + patch_size[2]

    return x_min, x_max, y_min, y_max, z_min, z_max

class DarksideTrainDataset(Dataset):
    def __init__(self, transform=None, n_repeats=1, patch_size=(128, 128, 128), meta='data_indexes/train.txt'):

        with open(meta, 'r') as f:
            self.vol_filenames = [i[:-1] for i in f.readlines()]
            f.close()

        self.transform = transform
        self.n_repeats = n_repeats
        self.patch_size = patch_size

    def __len__(self):
        
        return len(self.vol_filenames) * self.n_repeats

    def __getitem__(self, idx):

        idx = idx % len(self.vol_filenames)

        data = np.lib.format.open_memmap(self.vol_filenames[idx], mode='r')
        label = np.lib.format.open_memmap(self.vol_filenames[idx].replace('seismicCubes_RFC_fullstack', 'fault_segments_upk'), mode='r')
        
        
        x_min, x_max, y_min, y_max, z_min, z_max = rand_bbox_3d(self.patch_size)

        seismic = data[x_min:x_max, y_min:y_max, z_min:z_max].copy()
        mask = label[x_min:x_max, y_min:y_max, z_min:z_max].copy()
        
        if random.random() > 0.5:
            seismic, mask = np.flip(seismic, axis=0), np.flip(mask, axis=0)
        if random.random() > 0.5:
            seismic, mask = np.flip(seismic, axis=1), np.flip(mask, axis=1)
        if random.random() > 0.5:
            seismic, mask = np.rot90(seismic, axes=(0, 1)), np.rot90(mask, axes=(0, 1))
        
        #Feature Engineering
        grad = np.gradient(seismic, axis=2).astype('float')
        edge_grad = (np.roll(grad, 1, axis=2) * grad <= 0).astype('float')
        edge_raw = (np.roll(seismic, 1, axis=2) * seismic <= 0).astype('float')
        
        # rescale volume
        seismic = seismic / 94.64
        grad = grad / 39.04
        
        seismic = np.nan_to_num(np.stack([seismic, grad, edge_grad, edge_raw], axis=0), nan=0.0, posinf=0.0, neginf=0.0).clip(-1, 1)
        
        seismic = torch.from_numpy(seismic)
        mask = torch.from_numpy(mask.copy())
        
        # When done, it's good practice to close the memmap (optional in many cases)
        del data, label   # This will close the memmap file

        # apply volumentations aug
        if self.transform:
            seismic = self.transform(image=seismic)['image']
        
        return seismic.float(), mask.unsqueeze(0).float()




class DarksideValidDataset(Dataset):

    def __init__(self, transform=None, patch_size=(128, 128, 128), meta='data_indexes/valid.txt'):

        with open(meta, 'r') as f:
            all_vol_filenames = [i[:-1] for i in f.readlines()]
            f.close()
        
        all_slices = []
        for x_min in list(range(0, 300-patch_size[0], patch_size[0]))+[300-patch_size[0]]:
            for y_min in list(range(0, 300-patch_size[1], patch_size[1]))+[300-patch_size[1]]:
                for z_min in list(range(0, 1259-patch_size[2], patch_size[2]))+[1259-patch_size[2]]:
                    x_max = x_min + patch_size[0]
                    y_max = y_min + patch_size[1]
                    z_max = z_min + patch_size[2]
                    all_slices.append((x_min, x_max, y_min, y_max, z_min, z_max))
        
        self.vol_filenames = []
        for vol in all_vol_filenames:
            for slc in all_slices:
                self.vol_filenames.append((vol, slc))
                
        print(f'Total valid sample count: {len(self.vol_filenames)}')
                
        self.transform = transform
        self.patch_size = patch_size
                    

    def __len__(self):
        
        return len(self.vol_filenames)

    def __getitem__(self, idx):
        
        filename, slc = self.vol_filenames[idx]
        data = np.lib.format.open_memmap(filename, mode='r')
        label = np.lib.format.open_memmap(filename.replace('seismicCubes_RFC_fullstack', 'fault_segments_upk'), mode='r')
        
        x_min, x_max, y_min, y_max, z_min, z_max = slc

        seismic = data[x_min:x_max, y_min:y_max, z_min:z_max].copy()
        
        #Feature Engineering
        grad = np.gradient(seismic, axis=2).astype('float')
        edge_grad = (np.roll(grad, 1, axis=2) * grad <= 0).astype('float')
        edge_raw = (np.roll(seismic, 1, axis=2) * seismic <= 0).astype('float')
        
        # rescale volume
        seismic = seismic / 94.64
        grad = grad / 39.04
        
        seismic = np.nan_to_num(np.stack([seismic, grad, edge_grad, edge_raw], axis=0), nan=0.0, posinf=0.0, neginf=0.0).clip(-1, 1)
        
        seismic = torch.from_numpy(seismic)
        mask = torch.from_numpy(label[x_min:x_max, y_min:y_max, z_min:z_max].copy())
        
        # When done, it's good practice to close the memmap (optional in many cases)
        del data, label   # This will close the memmap file

        # apply volumentations aug
        if self.transform:
            seismic = self.transform(image=seismic)['image']
        
        return seismic.float(), mask.unsqueeze(0).float()