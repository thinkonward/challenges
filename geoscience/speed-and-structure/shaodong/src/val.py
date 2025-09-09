import os
import numpy as np
import argparse
import glob
import torch
from pathlib import Path
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
import numpy as np
from detectron2.config import LazyConfig, instantiate
from tqdm import tqdm
from data.dataset import SASDataset




@torch.no_grad()
def do_val(data_root, val_txt, model_list):
    val_ds = SASDataset(data_root, val_txt, mode="test")
    test_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)
    # model.eval()
    prog_bar = tqdm(test_loader)
    pred_vel_list = []
    vel_list = []
    for iii, batch in enumerate(prog_bar):
        seis = batch["seis"].cuda()
        vel = batch["vel"].cuda()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            
            pred_vel = sum([md(seis) for md in model_list]) / len(model_list)
            pred_vel = pred_vel.float()

            seis = seis.flip(dims=(1, -1))
            pred_vel_hflp = sum([md(seis) for md in model_list]) / len(model_list)
            pred_vel_hflp = pred_vel_hflp.flip(dims=(-1,)).float()

            pred_vel = (pred_vel + pred_vel_hflp) / 2.0
            pred_vel_list.append(pred_vel)
            vel_list.append(vel)
    pred_vel_list = torch.concatenate(pred_vel_list)
    vel_list = torch.concatenate(vel_list)
    loss = F.l1_loss(pred_vel_list.squeeze(), vel_list.squeeze(), reduction="none") / torch.abs(vel_list.squeeze())

    print(loss.shape)
    return loss.mean()


def main():
    data_root = r"./data/train_data/"
    val_txt = r"./train_txt/val_f0.txt"

    configs = [
        # # tiny s_at_8 ohem2
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",

        # # base s_at_6
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",

        # # base s_at_8
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",

    ]
    
    ckpt_paths = [
        # tiny s_at_8 ohem2
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed2.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed5.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed9.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed10.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed14.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed3487.pth",

        # base s_at_6
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed4.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed6.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed7.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed9.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed10.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed11.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed12.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed13.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed14.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed15.pth",


        # base s_at_8
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed2.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed3.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed4.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed5.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed7.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed10.pth",

    ]

    def create_model(cfg, ck_path, ):
        cfg = LazyConfig.load(cfg)
        model = instantiate(cfg.model)
        load_from = ck_path
        state_dict = torch.load(load_from, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {
            k.replace("_orig_mod.module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }
        load_result = model.load_state_dict(state_dict, strict=False)
        print(load_result)
        model = model.cuda()
        model.eval()
        return model
    
    model_list = []
    for cfg, ck_path in zip(configs, ckpt_paths):
        model = create_model(cfg, ck_path)
        model_list.append(model)

    score = do_val(data_root, val_txt, model_list)

    print(f"Validation score: {score:.6f}") 


if __name__ == "__main__":
    main()
