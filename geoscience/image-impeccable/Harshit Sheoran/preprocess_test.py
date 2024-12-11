import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os, sys, copy, time

import cv2
from PIL import Image
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

import h5py

print("Creating Test Data .h5 files")

min_noise, mean_noise, max_noise, folders = [], [], [], []

for folder in tqdm(glob('./data/test_data/*')):
    for file in glob(folder+"/*"):
        if '.npy' not in file: continue
        output_file = file.replace('.npy', '.h5')
        
        #if os.path.exists(output_file):
            #continue
        
        vol = np.load(file, allow_pickle=1)
        
        if vol.shape[0]!=1259:
            vol = vol.T
        
        output_file = file.replace('.npy', '.h5')
        hf = h5py.File(f"{output_file}", 'w')
        hf.create_dataset('volume', data=vol.astype(np.float16), compression="lzf")
        
        min_noise.append(vol.min())
        mean_noise.append(vol.mean())
        max_noise.append(vol.max())
        folders.append(folder.split('/')[-1])

        #break
    #break

test_minmax = pd.DataFrame({'min_noise': min_noise, 'mean_noise': mean_noise, 'max_noise': max_noise, 'folder': folders})

sample_sub = np.load('./data/image-impeccable-submission-sample.npz', allow_pickle=1)
test_keys = np.array(list(sample_sub.keys()))

DAT = {'noised': [], 'view': [], 'idx': []}
for key in test_keys:
    folder = '_'.join(key.split('.')[0].split('_')[:-1])
    file = glob(f"./data/test_data/{folder}/*.npy")[0].replace('.npy', '.h5')
    DAT['noised'].append(file)
    DAT['view'].append(key.split('-')[-1].split('_')[0])
    DAT['idx'].append((int(key.split('_')[-1])+1) * 75)
    
data = pd.DataFrame(DAT).reset_index(drop=True)

#print(data)

data['folder'] = data.noised.apply(lambda x: x.split('/')[3])

#print(data)

data = pd.merge(data, test_minmax, on='folder')

#print(data, test_minmax)

data.to_csv('./data/processed_test1.csv', index=False)

print("Saved!")