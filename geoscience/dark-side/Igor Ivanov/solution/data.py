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
parser.add_argument('--n_volumes', default=-1, type=int, help='Number of volumes to process. All volumes: -1, single first: 1, etc.')
args = parser.parse_args() # pass empty list to run in notebook using default arg values: parser.parse_args([])
for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_volume(path, data_type, output_dir_axis_0, output_dir_axis_1):
    """
    Create images from slices of a single volume over 2 axes.

    Parameters:
    path : str
        Path to a directory where volume file (.npy) is located
    data_type : str
        Type of the volume: "input" or "label".
    output_dir_axis_0 : str
        Output dir for axis 0
    output_dir_axis_1 : str
        Output dir for axis 1
    """
    assert data_type in ['input', 'label'], ('Unknown data type: %s' % data_type)
    # wildcard just contains part of a file name unique for input/label files
    if data_type == 'input':
        wildcard = 'seismicCubes*.npy'        
    else:
        wildcard = 'fault*.npy'

    volume_id = path.split('/')[-2]
    image_dir_0 = os.path.join(output_dir_axis_0, volume_id)
    image_dir_1 = os.path.join(output_dir_axis_1, volume_id)
    os.makedirs(image_dir_0, exist_ok=True)
    os.makedirs(image_dir_1, exist_ok=True)

    file_in = sorted(glob.glob(os.path.join(path, wildcard)))[0]
    volume = np.load(file_in, allow_pickle=True)

    if volume.shape != (300, 300, 1259):
        print('Skipped volume:', path)
        return

    if volume_id in to_transpose and data_type == 'input':
        volume = np.transpose(volume, [1, 0, 2])
        print('Volume was transposed:', volume_id)

    # min-max normalize
    if data_type == 'input':
        volume = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)
    else:
        volume = volume.astype(np.uint8)

    # create images for axis 0
    for i in range(volume.shape[0]):
        print('Axis: 0    Image: %04d' % i, end='\r')
        name_out = ('img_' + volume_id + '_%04d_%s.png' % (i, data_type))
        path_out = os.path.join(image_dir_0, name_out)
        # get slice to save as image
        image_out = volume[i, :, :]
        # if there is no positive class in label - do not save label image
        if data_type == 'label' and len(np.unique(image_out)) == 1:
            continue
        # save
        Image.fromarray(image_out).save(path_out, format='PNG')

    # create images for axis 1
    for i in range(volume.shape[1]):
        print('Axis: 1    Image: %04d' % i, end='\r')
        name_out = ('img_' + volume_id + '_%04d_%s.png' % (i, data_type))
        path_out = os.path.join(image_dir_1, name_out)
        # get slice to save as image
        image_out = volume[:, i, :]
        # if there is no positive class in label - do not save label image
        if data_type == 'label' and len(np.unique(image_out)) == 1:
            continue
        # save
        Image.fromarray(image_out).save(path_out, format='PNG')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# 8 volumes have no positive pixels in their labels
to_exclude = [
    '2023-10-05_37fd5dd2', '2023-10-05_407bcfc6', '2023-10-05_74a447f1', '2023-10-05_795448d4', 
    '2023-10-05_aa1525c4', '2023-10-05_ba087996', '2023-10-05_c923057a', '2023-10-05_eff8c5d7', 
]

# 12 volumes must be transposed to [1, 0, 2]
to_transpose = [
    '2024-06-10_1a4e5680', '2024-06-10_1b9a0096', '2024-06-10_2bd82c05', '2024-06-10_3b118e17', 
    '2024-06-10_971ac6dd', '2024-06-10_43537d46', '2024-06-10_662066f4', '2024-06-10_b7c329be', 
    '2024-06-10_bfd43f22', '2024-06-10_c952ed24', '2024-06-10_cec3da7f', '2024-06-10_eb45f27e',
]

dirs = sorted(glob.glob(os.path.join(args.input_dir, '*/')))
dirs = [d for d in dirs if d.split('/')[-2] not in to_exclude]
if args.n_volumes != -1:
    dirs = dirs[:args.n_volumes]
n_dirs = len(dirs)
print('N volumes:', n_dirs)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

for counter, path in enumerate(dirs):
    print('Input volume: %04d of %d' % (counter, n_dirs))
    process_volume(path, 'input', args.output_dir_axis_0, args.output_dir_axis_1)
    if args.has_label:
        print('Label volume: %04d of %d' % (counter, n_dirs))
        process_volume(path, 'label', args.output_dir_axis_0, args.output_dir_axis_1)

print('\nDone')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


