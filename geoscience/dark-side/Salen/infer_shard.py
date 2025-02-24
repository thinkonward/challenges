import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import glob
import random
import numpy as np
#import pandas as pd
import re
import segmentation_models_pytorch_3d as smp

from darkside_utils import create_submission

parser = argparse.ArgumentParser()
parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
parser.add_argument('--port', default=12345, type=int, help='DDP Port')
parser.add_argument('--lr', default=4e-4, type=float, help='Initial learning rate')
parser.add_argument('--eta_min', default=1e-7, type=float, help='Cosine minimum lr')
parser.add_argument('--sub', default=94.64, type=float, help='Input subtract factor')
parser.add_argument('--width', default=3, type=int, help='A Width Multiply Factor to Original SegResNet')
parser.add_argument('--depth', default=1, type=int, help='A Depth Multiply Factor to Original SegResNet')
parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
#parser.add_argument('--fold', default=0, type=int, help='Number of kfold')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--patch_size', default='192_192_640', type=str, help='Patch size')
parser.add_argument('--val_interv', default=1, type=int, help='Valid Interval')
parser.add_argument('--log_dir', default='./runs', type=str, help='Directory for tensorboard logs')
parser.add_argument('--noeval',default=False,action='store_true',help='Turn off Eval Process')

parser.add_argument('--shard', default='0_2', type=str, help='infer shard')
args = parser.parse_args()



rank = 'cuda'

model = smp.Unet(
        encoder_name="efficientnet-b4", # choose encoder, e.g. resnet34
        in_channels=4,                  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
        encoder_weights=None,
        classes=1,                      # model output channels (number of classes in your dataset)
    ).to(rank)
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('pretrained.pth', map_location=rank).items()})
model.eval()
print('loaded')

all_test_vols = sorted(glob.glob('test_data/*/*.npy'))[int(args.shard.split('_')[0]):int(args.shard.split('_')[1])]
len(all_test_vols)

shape = (300, 300, 1259)
stride=(24, 24, 80)
patch_size=list(map(int, args.patch_size.split('_')))


all_slices = []
for x_min in list(range(0, shape[-3]-patch_size[0], stride[0]))+[shape[-3]-patch_size[0]]:
    for y_min in list(range(0, shape[-2]-patch_size[1], stride[1]))+[shape[-2]-patch_size[1]]:
        for z_min in list(range(0, shape[-1]-patch_size[2], stride[2]))+[shape[-1]-patch_size[2]]:
            x_max = x_min + patch_size[0]
            y_max = y_min + patch_size[1]
            z_max = z_min + patch_size[2]

            all_slices.append((x_min, x_max, y_min, y_max, z_min, z_max))

all_arr = {i: np.load(i).astype(np.float16) for i in tqdm(all_test_vols)}

all_overlap = {i: np.zeros(shape, dtype=np.int16) for i in tqdm(all_test_vols)}
all_pred = {i: np.zeros(shape, dtype=np.float32) for i in tqdm(all_test_vols)}

def tta(array, method='h', back=False):
    
    if method == 'h':
        return np.flip(array, axis=0)
    elif method == 'v':
        return np.flip(array, axis=1)
    elif method == 'hv':
        return np.flip(np.flip(array, axis=1), axis=0)
        
    elif method == 'r':
        if not back:
            return np.rot90(array, axes=(0, 1))
        else:
            return np.rot90(array, axes=(0, 1), k=-1)

    elif method == 'hr':
        if not back:
            array = np.flip(array, axis=0)
            return np.rot90(array, axes=(0, 1))
        else:
            return np.flip(np.rot90(array, axes=(0, 1), k=-1), axis=0)

    elif method == 'vr':
        if not back:
            array = np.flip(array, axis=1)
            return np.rot90(array, axes=(0, 1))
        else:
            return np.flip(np.rot90(array, axes=(0, 1), k=-1), axis=1)

    elif method == 'hvr':
        if not back:
            array = np.flip(np.flip(array, axis=1), axis=0)
            return np.rot90(array, axes=(0, 1))
        else:
            array = np.flip(np.flip(array, axis=1), axis=0)
            return np.rot90(array, axes=(0, 1), k=-1)

    elif not method:
        return array
    
    else:
        raise

class InferDataset(Dataset):
    def __init__(self, all_arrays):

        #self.all_arrays = all_arrays
        self.all_arr_slc = []
        for arr in all_arrays:
            for aug in [None, 'h', 'v', 'hv', 'r', 'hr', 'vr', 'hvr']:
            #for aug in [None, 'h', 'v', 'r']:
                for slc in all_slices:
                    self.all_arr_slc.append((arr, slc, aug))

    def __len__(self):
        
        return len(self.all_arr_slc)

    def __getitem__(self, idx):

        name, (x_min, x_max, y_min, y_max, z_min, z_max), aug_ = self.all_arr_slc[idx]
        input_array = all_arr[name]
        seismic = input_array[x_min:x_max, y_min:y_max, z_min:z_max].copy()

        seismic = tta(seismic, method=aug_, back=False)
                
        #Feature Engineering
        grad = np.gradient(seismic, axis=2).astype('float')
        edge_grad = (np.roll(grad, 1, axis=2) * grad <= 0).astype('float')
        edge_raw = (np.roll(seismic, 1, axis=2) * seismic <= 0).astype('float')
        
        # rescale volume
        seismic = seismic / 94.64
        grad = grad / 39.04
        
        seismic = np.nan_to_num(np.stack([seismic, grad, edge_grad, edge_raw], axis=0), nan=0.0, posinf=0.0, neginf=0.0).clip(-1, 1)

        seismic = torch.from_numpy(seismic).float()

        return seismic

ds = InferDataset(all_test_vols)
dl = DataLoader(ds, 
              batch_size=1, 
              num_workers=16, 
              pin_memory=True,
              drop_last=False,
              shuffle=False,
           )

model = torch.compile(model)

for dl_idx, seismic in enumerate(tqdm(dl)):
    name, (x_min, x_max, y_min, y_max, z_min, z_max), aug_ = ds.all_arr_slc[dl_idx]
    with torch.no_grad():
        pred = model(seismic.cuda()).squeeze().sigmoid().cpu().numpy()
        
    pred = tta(pred, method=aug_, back=True)

    all_overlap[name][x_min:x_max, y_min:y_max, z_min:z_max] += 1
    all_pred[name][x_min:x_max, y_min:y_max, z_min:z_max] += pred

for idx, nam in enumerate(tqdm(all_test_vols)):

    outputs = all_pred[nam] / all_overlap[nam]

    save_dir = nam.replace('test_data/', 'test_data/infered_results/')
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    np.save(save_dir, (outputs > 0.5).astype('uint8'))