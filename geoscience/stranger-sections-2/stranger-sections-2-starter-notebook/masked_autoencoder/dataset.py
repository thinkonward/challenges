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
from torch.utils import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision import transforms, VisionDataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform, target_transform):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, 'image')
        self.label_folder = os.path.join(root_dir, 'label')
        self.image_filenames = [file for file in os.listdir(self.image_folder) if file.endswith(('.JPG','.jpg','.jpeg', '.JPEG'))]
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        image_id = os.path.splitext(self.image_filenames[idx])[0]  # Extract ID without extension
        label_name = os.path.join(self.label_folder, f'{image_id}_gt.npy')
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        label = np.load(label_name)
        label = torch.from_numpy(label)
        label = self.target_transform(label)
        return image, label