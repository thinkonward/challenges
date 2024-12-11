#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('--input_dir', default='train', type=str, help='Directory with volumes')
parser.add_argument('--output_dir_axis_0', default='train_img_axis_0', type=str, help='Directory to save images taken from axis 0')
parser.add_argument('--output_dir_axis_1', default='train_img_axis_1', type=str, help='Directory to save images taken from axis 1')
parser.add_argument('--has_label', default=1, choices=[0, 1], type=int, help='Train data has label: 1, test data does not: 0')
parser.add_argument('--n_volumes', default=-1, type=int, help='Number of volumes to process. Process all volumes in the input dir: -1, single first volume: 1, etc.')
args = parser.parse_args() # pass empty list to run in notebook using default arg values: parser.parse_args([])
for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# 9 pairs of noisy/clean volumes visually don't match - excluding them
mismatch = ['42673698', '77692226', '77692237', '77692243', '77692246', '77702634', '77702638', '89194322', '89194324']
dirs = sorted(glob.glob(os.path.join(args.input_dir, '*/')))
dirs = [d for d in dirs if d.split('/')[-2] not in mismatch]
if args.n_volumes != -1:
    dirs = dirs[:args.n_volumes]
n_dirs = len(dirs)
print('N volumes:', n_dirs)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_volume(path, data_type):
    """
    Create images from slices of a single volume over axes 0 and 1.
    There are 300 images over each axis.

    Parameters:
    path : str
        Path to a directory where volume file (.npy) is located
    data_type : str
        Type of the volume: "noisy" or "clean".
        Processing is exactly the same for noisy/clean volumes.
        The purpose of this parameter is to switch file name template (wildcard) 
            specific for noisy/clean volumes.
    """
    assert data_type in ['noisy', 'clean'], ('Unknown data type: %s' % data_type)
    # wildcard just contains part of a file name unique for noisy/clean files
    if data_type == 'noisy':
        wildcard='*noise*.npy'
    else:
        wildcard='*RFC*.npy'

    # create subdir to save images for a given volume
    volume_id = path.split('/')[-2] # '42487393'
    image_dir_0 = os.path.join(args.output_dir_axis_0, volume_id) # 'train_img_axis_0/42487393'
    image_dir_1 = os.path.join(args.output_dir_axis_1, volume_id) # 'train_img_axis_1/42487393'
    os.makedirs(image_dir_0, exist_ok=True)
    os.makedirs(image_dir_1, exist_ok=True)

    file_in = sorted(glob.glob(os.path.join(path, wildcard)))[0]
    volume = np.load(file_in, allow_pickle=True)

    # transpose to shape (300, 300, 1259) if needed
    shape = volume.shape
    if shape[0] == 1259:
        volume = volume.T
    elif shape[1] == 1259:
        print('Skipped volume:', path)
        return

    # min-max normalize
    volume = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)

    # create images for axis 0
    for i in range(volume.shape[0]):
        name_out = ('img_' + volume_id + '_%03d_%s.png' % (i, data_type))
        path_out = os.path.join(image_dir_0, name_out)
        # get slice to save as image
        image_out = volume[i, :, :]
        # save
        Image.fromarray(image_out).save(path_out, format='PNG')
        print('Axis: 0    Image: %03d' % i, end='\r')

    # create images for axis 1
    for i in range(volume.shape[1]):
        name_out = ('img_' + volume_id + '_%03d_%s.png' % (i, data_type))
        path_out = os.path.join(image_dir_1, name_out)
        # get slice to save as image
        image_out = volume[:, i, :]
        # save
        Image.fromarray(image_out).save(path_out, format='PNG')
        print('Axis: 1    Image: %03d' % i, end='\r')



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

for counter, path in enumerate(dirs):
    print('Noisy volume: %03d of %d' % (counter, n_dirs))
    process_volume(path, data_type='noisy')    
    if args.has_label:
        print('Clean volume: %03d of %d' % (counter, n_dirs))
        process_volume(path, data_type='clean')

print('\nDone')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


