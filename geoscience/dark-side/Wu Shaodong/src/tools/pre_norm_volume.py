import os
import numpy as np
from natsort import natsorted
import multiprocessing


def rescale_volume(seismic, low=2, high=98):
    """
    Rescaling 3D seismic volumes 0-255 range, clipping values between low and high percentiles
    """
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255
    
    return seismic


SRC_TRAIN_DATA_ROOT = r"/root/autodl-tmp/Dark_side_of_the_volume/data/train_data/"
DST_TRAIN_DATA_ROOT = r"/root/autodl-tmp/Dark_side_of_the_volume/data/train_data_normed_2_98"
PROCESS_THREAD_NUM = 16

os.makedirs(DST_TRAIN_DATA_ROOT, exist_ok=True)

def process(test_id):
    files = natsorted(os.listdir(f"{SRC_TRAIN_DATA_ROOT}/{test_id}"))
    # assert "noise" in files[1]
    # assert "fullstack" in files[0]
    data = np.load(os.path.join(SRC_TRAIN_DATA_ROOT, test_id, files[1]), allow_pickle=True, mmap_mode="r+")
    label = np.load(os.path.join(SRC_TRAIN_DATA_ROOT, test_id,files[0]), allow_pickle=True, mmap_mode="r+")

    data = data.astype(np.float32)
    label = label.astype(np.uint8)

    save_dir = os.path.join(DST_TRAIN_DATA_ROOT, test_id)

    os.makedirs(save_dir, exist_ok=True)
    data = rescale_volume(data)

    np.save(f"{save_dir}/{files[1]}", data)
    np.save(f"{save_dir}/{files[0]}", label)


with multiprocessing.Pool(processes = PROCESS_THREAD_NUM) as pool:
    test_id_s = natsorted(os.listdir(SRC_TRAIN_DATA_ROOT))
    print(test_id_s)
    result = pool.map(process, test_id_s)