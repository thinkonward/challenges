import os
import random
import torch
import h5py
from torch.utils.data import DataLoader, Dataset,ConcatDataset
import numpy as np


class H5Dataset(Dataset):
    def __init__(self, filepath, slice_offset=0, transpose = False, flip_vertical = False, flip_horizontal = False):
        # Open the HDF5 file
        self.filepath = filepath
        self.slice_offset = slice_offset
        self.transpose = transpose
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        
        # Get the data shape
        with h5py.File(filepath, 'r') as h5file:
            length = h5file['data'].shape[0]
            self.num_slices = ((length - slice_offset) // 200) * 200
        
    def __len__(self):
        return self.num_slices // 200
    
    def __getitem__(self, idx):
        start = idx * 200 + self.slice_offset
        end = start + 200

        # Open the HDF5 file and load the data
        with h5py.File(self.filepath, 'r') as h5file:
            data = h5file['data'][start:end]
            label = h5file['label'][start:end]

        # Normalize data and labels
        data = (data - 223.07626342773438) / 23.740541458129883
        label = (label - 0.0021280869841575623) / 100.11495208740234
        
        # Reshape the data and labels
        data = data.reshape(1, 200, 300, 300)
        label = label.reshape(1, 200, 300, 300)
        
        # Apply random transpose with given probability
        if self.transpose:
            data = data.transpose(0, 1, 3, 2)  # Swap last two dimensions
            label = label.transpose(0, 1, 3, 2)
        
        # Apply random vertical flip with given probability
        if self.flip_vertical:
            data = np.flip(data, axis=2)  # Flip along height
            label = np.flip(label, axis=2)
        
        # Apply random horizontal flip with given probability
        if self.flip_horizontal:
            data = np.flip(data, axis=3)  # Flip along width
            label = np.flip(label, axis=3)
        
        # Ensure data and labels are copied to prevent negative strides
        data = data.copy()
        label = label.copy()

        # Convert data and label to PyTorch tensors
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return data, label



# Initialize the datasets with different offsets

def dataset_offset(slice_offset, transpose, flip_vertical, flip_horizontal, start, end, increment):
    # Define the base path relative to the current directory (h5py folder is within the final_submission directory)
    h5_base_path = os.path.join(os.getcwd(), "h5py/")
    
    dataset_offset = ConcatDataset([
        H5Dataset(os.path.join(h5_base_path, f"original_image-impeccable-train-data-part{i}.h5"), 
                  slice_offset=slice_offset, transpose=transpose, 
                  flip_vertical=flip_vertical, flip_horizontal=flip_horizontal)
        for i in range(start, end + 1, increment)
    ])
    return dataset_offset