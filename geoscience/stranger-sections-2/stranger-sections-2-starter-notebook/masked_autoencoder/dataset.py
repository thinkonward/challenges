import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
import numpy as np
import glob
import cv2
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir, img_size, transform, target_transform):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'image')
        self.label_folder = os.path.join(root_dir, 'label')
        self.image_filenames = [file for file in os.listdir(self.image_folder) if file.endswith(('.JPG','.jpg','.jpeg', '.JPEG'))]
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        image_id = os.path.splitext(self.image_filenames[idx])[0]  # Extract ID without extension
        label_name = os.path.join(self.label_folder, f'{image_id}_gt.npy')
        resizer = v2.Compose([v2.Resize((self.img_size))])
        image = Image.open(img_name).convert('RGB')
        image = resizer(image)
        image = self.transform(image)
        label = np.load(label_name)
        label = torch.from_numpy(label)
        label = resizer(label)
        label = self.target_transform(label)
        return image, label