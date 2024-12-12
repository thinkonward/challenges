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

min_noise, mean_noise, max_noise, min_denoise, mean_denoise, max_denoise, folders = [], [], [], [], [], [], []

for folder in tqdm(glob('./data/train_data/*')):
    for file in glob(folder+"/*"):
        if '.npy' not in file: continue
        is_noisy = 1 if 'w_noise_vol_' in file else 0

        output_file = file.replace('.npy', '.h5')
        
        #if os.path.exists(output_file):
            #continue
        
        vol = np.load(file, allow_pickle=1)
        
        if vol.shape[0]!=1259:
            vol = vol.T
        
        output_file = file.replace('.npy', '.h5')
        hf = h5py.File(f"{output_file}", 'w')
        hf.create_dataset('volume', data=vol.astype(np.float16), compression="lzf")
        
        if is_noisy:
            min_noise.append(vol.min())
            mean_noise.append(vol.mean())
            max_noise.append(vol.max())
        else:
            min_denoise.append(vol.min())
            mean_denoise.append(vol.mean())
            max_denoise.append(vol.max())

    folders.append(folder.split('/')[-1])

    #break

train_minmax = pd.DataFrame({'min_noise': min_noise, 'mean_noise': mean_noise, 'max_noise': max_noise,
                            'min_denoise': min_denoise, 'mean_denoise': mean_denoise, 'max_denoise': max_denoise,
                            'folder': folders})

DAT = {'noised': [], 'denoised': [], 'ssim': []}

def minmax(array):
    return (array - array.min()) / (array.max() - array.min())

for folder in tqdm(glob('./data/train_data/*')):
    
    denoised_path, noised_path = sorted(glob(folder+'/*.h5'))
    denoised = h5py.File(denoised_path)['volume']
    noised = h5py.File(noised_path)['volume']
    
    if not np.array_equal(denoised.shape, noised.shape):
        print('not equal shape', folder)
    
    ssims = []
    for idx in [75, 150, 225]:
        score = ssim(minmax(denoised[:, idx]), minmax(noised[:, idx]), data_range=1)
        ssims.append(score)
        score = ssim(minmax(denoised[:, :, idx]), minmax(noised[:, :, idx]), data_range=1)
        ssims.append(score)
    
    DAT['noised'].append(noised_path)
    DAT['denoised'].append(denoised_path)
    DAT['ssim'].append(np.mean(ssims))
    
    #break
    
data = pd.DataFrame(DAT)


from sklearn.model_selection import *

folds = KFold(n_splits=4).split(data)

data['fold'] = -1
for F, fold in enumerate(folds):
    data.loc[fold[1], 'fold'] = F


data['folder'] = data.noised.apply(lambda x: x.split('/')[-2])


data2 = pd.merge(data, train_minmax, on='folder')


data2.to_csv('./data/train_master1.csv', index=False)



data = pd.read_csv('./data/train_master1.csv')

new_rows = []
for i, row in tqdm(data.iterrows(), total=len(data)):
    for view in ['i', 'x']:
        idxs = np.unique(np.arange(0, 300, 5).tolist() + [75, 150, 225])
        #idxs = np.unique(np.arange(0, 150, 10).tolist() + [75,])
        for idx in idxs:
            this_row = row.copy()
            this_row['view'] = view
            this_row['idx'] = idx
            new_rows.append(this_row)
    
    #break

data = pd.DataFrame(new_rows).reset_index(drop=True)
    
data = data[data.ssim > 0.7].reset_index(drop=True)

data.to_csv('./data/processed_train1.csv', index=False)