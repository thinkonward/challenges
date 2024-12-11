import numpy as np
import h5py

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import pickle
import redis
redis_cache = redis.StrictRedis(host='localhost', port=6379, db=0)

class ObsidianDataset(Dataset):
    def __init__(self, cfg, data, transforms=None, is_training=False):
        self.cfg = cfg
        self.data = data
        self.transforms = transforms
        self.is_training = is_training

        self.class_info()
    
    def __len__(self):
        return len(self.data)

    def class_info(self):
        self.classes = sorted(self.data.label.unique())
        self.n_classes = len(self.classes)
        self.class_to_idx = {cl: i for i, cl in enumerate(self.classes)}

        print(self.class_to_idx)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image = cv2.imread(row.image_path)[:, :, ::-1]

        if self.transforms:
            transformed = self.transforms(image=image)

            image = transformed['image']

            if image.dtype==torch.uint8: 
                image = image / image.max()
        
        label = np.zeros((self.n_classes,), dtype=np.float32)
        idx = self.class_to_idx[row.label]
        label[idx] = 1.

        label = torch.as_tensor(label)

        return {
            'images': image,
            'labels': label,
            'ids': row.image_path
        }


import zstandard as zstd
zstd_compression_level = 3
def load_and_cache_data_zstd(cache_key, data=None):
    
    if redis_cache.exists(cache_key):
        compressed_data = redis_cache.get(cache_key)
        decompressed_data = zstd.decompress(compressed_data)
        cached_data = pickle.loads(decompressed_data)
        return cached_data
    else:
        serialized_data = pickle.dumps(data)
        compressor = zstd.ZstdCompressor(level=zstd_compression_level)
        compressed_data = compressor.compress(serialized_data)
        redis_cache.set(cache_key, compressed_data)

load_and_cache_data = load_and_cache_data_zstd


class ImpeccableDataset(Dataset):
    def __init__(self, cfg, data, transforms=None, is_training=False):
        self.cfg = cfg
        self.data = data
        self.transforms = transforms
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def minmax(self, array, _min=None, _max=None):
        if _min==None:
            _min = array.min()
        if _max==None:
            _max = array.max()    
        
        return (array - _min) / (_max - _min)

    def load_raw_data(self, row, key):
        denoised_path = row.denoised
        noised_path = row.noised

        denoised_vol = h5py.File(denoised_path)['volume']
        noised_vol = h5py.File(noised_path)['volume']
        
        if row['view']=='i':
            noised = noised_vol[:, :, [row.idx]].T
            denoised = denoised_vol[:, :, [row.idx]].T
        else:
            noised = noised_vol[:, [row.idx]].transpose(1,2,0)
            denoised = denoised_vol[:, [row.idx]].transpose(1,2,0)

        noised = self.minmax(noised, _min=row.min_noise, _max=row.max_noise)
        denoised = self.minmax(denoised, _min=row.min_denoise, _max=row.max_denoise)

        load_and_cache_data(f"{key}_denoised", denoised)
        load_and_cache_data(f"{key}_noised", noised)

        return denoised, noised

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        key = f"{row.folder}_{row['view']}_{row.idx}"

        if redis_cache.exists(f"{key}_noised"):
            denoised = load_and_cache_data(f"{key}_denoised")
            noised = load_and_cache_data(f"{key}_noised")
        else:
            denoised, noised = self.load_raw_data(row, key)

        if self.is_training and np.random.random() < 0.5:
            noised = noised[:, ::-1].copy()
            denoised = denoised[:, ::-1].copy()

        noised_flip = noised[:, ::-1].copy()

        noised_ = np.zeros((1, self.cfg.image_info.height, self.cfg.image_info.width), dtype=np.float32)
        noised_[:, :300*1, :1259] = noised
        noised = noised_

        noised_ = np.zeros((1, self.cfg.image_info.height, self.cfg.image_info.width), dtype=np.float32)
        noised_[:, :300*1, :1259] = noised_flip
        noised_flip = noised_

        return {
            'noised': noised,
            'noised_flip': noised_flip,
            'denoised': denoised,
            'ids': f"{key}"
        }


class ImpeccableTestDataset(Dataset):
    def __init__(self, cfg, data, transforms=None, is_training=False):
        self.cfg = cfg
        self.data = data
        self.transforms = transforms
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def minmax(self, array, _min=None, _max=None):
        if _min==None:
            _min = array.min()
        if _max==None:
            _max = array.max()    
        
        return (array - _min) / (_max - _min)

    def load_raw_data(self, row, key):
        noised_path = row.noised

        noised_vol = h5py.File(noised_path)['volume']
        
        if row['view']=='i':
            noised = noised_vol[:, :, [row.idx]].T
        else:
            noised = noised_vol[:, [row.idx]].transpose(1,2,0)
            
        noised = self.minmax(noised, _min=row.min_noise, _max=row.max_noise)
        
        load_and_cache_data(f"{key}_noised", noised)

        return noised

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        key = f"{row.folder}_{row['view']}_{row.idx}"

        #if redis_cache.exists(f"{key}_noised"):
            #noised = load_and_cache_data(f"{key}_noised")
        #else:
        noised = self.load_raw_data(row, key)

        noised_flip = noised[:, ::-1].copy()

        noised_ = np.zeros((1, self.cfg.image_info.height, self.cfg.image_info.width), dtype=np.float32)
        noised_[:, :300*1, :1259] = noised
        noised = noised_

        noised_ = np.zeros((1, self.cfg.image_info.height, self.cfg.image_info.width), dtype=np.float32)
        noised_[:, :300*1, :1259] = noised_flip
        noised_flip = noised_

        return {
            'noised': noised,
            'noised_flip': noised_flip,
            'ids': f"{key}"
        }