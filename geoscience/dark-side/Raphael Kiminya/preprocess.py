import numpy as np, pandas as pd, h5py
import os, glob
from sys import getsizeof
from tqdm import tqdm

def rescale_volume(seismic, low=2, high=98):
    """
    Rescaling 3D seismic volumes 0-255 range, clipping values between low and high percentiles
    """
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval))
    seismic = np.clip(seismic, 0, 1) * 255
    seismic = seismic.astype(np.uint8)
    return seismic


def process_volume(path,out_dir):
    x = y = None
    for pi in glob.glob(f'{p}/*.npy'):
        if 'fault_segments' in pi:
            y = np.load(pi)
        if 'seismicCubes' in pi:
            x = np.load(pi)

    x = rescale_volume(x)
    p1 = f"{out_dir}/{p.split('/')[-1]}"
    os.makedirs(p1,exist_ok=True)
    with h5py.File(f'{p1}/x.h5', 'w') as h5file:
        h5file.create_dataset('x', data=x, chunks=True, compression="gzip")
    if y is not None:
        np.savez_compressed(f"{p1}/y.npz",y)


DATA_DIR = 'data'

paths = glob.glob(f'{DATA_DIR}/train/*')
out_dir = f'{DATA_DIR}/train_norm'
os.makedirs(out_dir,exist_ok=True)
for p in tqdm(paths):
    process_volume(p,out_dir)


paths = glob.glob(f'{DATA_DIR}/test/*')
out_dir = f'{DATA_DIR}/test_norm'
os.makedirs(out_dir,exist_ok=True)
for p in tqdm(paths):
    process_volume(p,out_dir)
