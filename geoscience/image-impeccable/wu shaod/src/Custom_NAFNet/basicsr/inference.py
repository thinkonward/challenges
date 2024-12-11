import logging
import torch
from os import path as osp
import os
import numpy as np
import glob
import tqdm
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs, img2tensor)
from basicsr.utils.options import dict2str

def rescale_volume(seismic, low=0, high=100):
    """
    Rescaling 3D seismic volumes 0-255 range, clipping values between low and high percentiles
    """
    print("in normed data")
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)

    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255

    return seismic

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # test data- dir root
    test_data_root = opt["test_dir"]
    test_res_save_root = opt["test_res_save_root"]

    # test_data_root = "../../data/train_images_new_shape"
    # test_res_save_root = "../../data/val_res_z"

    os.makedirs(test_res_save_root, exist_ok=True)
    # create test dataset and dataloader

    # create model
    opt['dist'] = False
    model = create_model(opt)

    # val_txt = "../../data/train_txt/val_f0.txt"
    # cases = np.loadtxt(val_txt, dtype=str).tolist()
    # for test_case in tqdm.tqdm(cases):
    #     print(test_case)
    for test_case in tqdm.tqdm(os.listdir(test_data_root)):
        test_data = glob.glob(f"{test_data_root}/{test_case}/*.npy")[0]
        test_data = glob.glob(f"{test_data_root}/{test_case}/seismic_w_noise_*.npy")[0]
        npy_data = np.load(test_data, allow_pickle=True, mmap_mode="r+")
        npy_data = rescale_volume(npy_data)
        C, H, W = npy_data.shape
        res = np.zeros_like(npy_data)
        res_cnt = np.zeros_like(npy_data)

        # test Z slice
        # for i in range(C):
        #     test_slice = npy_data[i, :, :]
        #     test_slice = test_slice[:, :, np.newaxis]
        #     tensor_data = img2tensor([test_slice], bgr2rgb=True, float32=True)[0]
        #     model.feed_data(data={'lq': tensor_data.unsqueeze(dim=0)})
        #     model.test()
        #     visuals = model.get_current_visuals()
        #     pred = visuals['result']
        #     pred = pred.squeeze()
        #     res[i, :, :] = pred
        #     res_cnt[i, :, :] += 1.0

        # test H slice
        for i in range(H):
            test_slice = npy_data[:, i, :]
            test_slice = test_slice[:, :, np.newaxis]
            tensor_data = img2tensor([test_slice], bgr2rgb=True, float32=True)[0]
            model.feed_data(data={'lq': tensor_data.unsqueeze(dim=0)})
            model.test()
            visuals = model.get_current_visuals()
            pred = visuals['result']
            pred = pred.squeeze().numpy()
            res[:, i, :] += pred
            res_cnt[:, i, :] += 1.0

        # test H slice hflip
        for i in range(H):
            test_slice = npy_data[:, i, :]
            test_slice = test_slice[:, ::-1]
            test_slice = test_slice[:, :, np.newaxis]
            tensor_data = img2tensor([test_slice.copy()], bgr2rgb=True, float32=True)[0]
            model.feed_data(data={'lq': tensor_data.unsqueeze(dim=0)})
            model.test()
            visuals = model.get_current_visuals()
            pred = visuals['result']
            pred = pred.squeeze().numpy()
            pred = pred[:, ::-1]
            res[:, i, :] += pred
            res_cnt[:, i, :] += 1.0

        # # test H slice vflip
        # for i in range(H):
        #     test_slice = npy_data[:, i, :]
        #     test_slice = test_slice[::-1, :]
        #     test_slice = test_slice[:, :, np.newaxis]
        #     tensor_data = img2tensor([test_slice.copy()], bgr2rgb=True, float32=True)[0]
        #     model.feed_data(data={'lq': tensor_data.unsqueeze(dim=0)})
        #     model.test()
        #     visuals = model.get_current_visuals()
        #     pred = visuals['result']
        #     pred = pred.squeeze().numpy()
        #     pred = pred[::-1, :]
        #     res[:, i, :] = pred
        #     res_cnt[:, i, :] += 1.0

        # test W slice
        for i in range(W):
            test_slice = npy_data[:, :, i]
            test_slice = test_slice[:, :, np.newaxis]
            tensor_data = img2tensor([test_slice], bgr2rgb=True, float32=True)[0]
            model.feed_data(data={'lq': tensor_data.unsqueeze(dim=0)})
            model.test()
            visuals = model.get_current_visuals()
            pred = visuals['result']
            pred = pred.squeeze().numpy()
            res[:, :, i] += pred
            res_cnt[:, :, i] += 1.0

        # test W slice hflip
        for i in range(W):
            test_slice = npy_data[:, :, i]
            test_slice = test_slice[:, :, np.newaxis]
            test_slice = test_slice[:, ::-1]
            tensor_data = img2tensor([test_slice.copy()], bgr2rgb=True, float32=True)[0]
            model.feed_data(data={'lq': tensor_data.unsqueeze(dim=0)})
            model.test()
            visuals = model.get_current_visuals()
            pred = visuals['result']
            pred = pred.squeeze().numpy()
            pred = pred[:, ::-1]
            res[:, :, i] += pred
            res_cnt[:, :, i] += 1.0

        # # test W slice vflip
        # for i in range(W):
        #     test_slice = npy_data[:, :, i]
        #     test_slice = test_slice[:, :, np.newaxis]
        #     test_slice = test_slice[::-1, :]
        #     tensor_data = img2tensor([test_slice.copy()], bgr2rgb=True, float32=True)[0]
        #     model.feed_data(data={'lq': tensor_data.unsqueeze(dim=0)})
        #     model.test()
        #     visuals = model.get_current_visuals()
        #     pred = visuals['result']
        #     pred = pred.squeeze().numpy()
        #     pred = pred[::-1, :]
        #     res[:, :, i] = pred
        #     res_cnt[:, :, i] += 1.0

        final_res = res / res_cnt
        save_dir = f"{test_res_save_root}/{test_case}/"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/infe_res.npy", final_res)
        


if __name__ == '__main__':
    main()
