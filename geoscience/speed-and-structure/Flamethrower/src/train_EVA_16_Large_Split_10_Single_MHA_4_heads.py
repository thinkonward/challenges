
import os
from glob import glob
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate

import os
import gc
import csv

import math
import shutil
import pickle
import random
import logging
import argparse
import warnings
import pprint

import sys
from anytree import Node, RenderTree
from typing import Dict, List

from utils import *
from muon import MuonWithAuxAdam

from copy import deepcopy
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import yaml

from tqdm import tqdm
from tabulate import tabulate

import torch
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
import timm


# Suppress warnings
warnings.simplefilter("ignore")

from dataset import SeismicDataset
from engine import train_one_epoch, valid_one_epoch

from models.model_EVA_MHA_Handler import ModelEMA
from models.model_EVA_16_Large_Split_10_Single_MHA_4_heads import (
    EVA16_MHA as EVA16_Model)


# Check CUDA availability
torch.cuda.is_available()
seed = 42
#Data generator randomness
def seed_worker(worker_id):
    worker_seed = seed 
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Your predefined args class
class Args:
    epochs = 30
    lr = 1e-4
    ema_decay = 0.99
    batch_size = 1
    accumulation_steps = 2
    num_workers = os.cpu_count()
    Encoder = "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k"
    name = "EVA_16_Large_Split_10_Single_MHA_4_heads"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sub_dir = "../Submissions"
    T_0 = 100
    min_lr = 5e-5
    swa_start = 200
    val_batch_size = 16
    trained = False
    seed = 42


def train_and_validate(args, train_df, valid_df):
    """
    Train and validate the EVA16 model with given training and validation datasets.

    This function sets seeds for reproducibility, initializes the model and its EMA counterpart,
    creates DataLoaders with proper seeding and batching, defines the loss, optimizer, and
    learning rate scheduler (a sequential scheduler combining cosine annealing and a constant LR),
    and runs training and validation epochs with early stopping based on validation MAPE.

    Args:
        args (Namespace): Contains configuration parameters including:
            - seed (int): Random seed for reproducibility.
            - device (str): Device for model training (e.g., 'cuda').
            - Encoder (str): Model encoder name.
            - batch_size (int): Batch size for training.
            - val_batch_size (int): Batch size for validation.
            - num_workers (int): Number of workers for DataLoader.
            - lr (float): Learning rate.
            - T_0 (int): Number of epochs for cosine annealing scheduler.
            - min_lr (float): Minimum learning rate for scheduler.
            - epochs (int): Total number of epochs to train.
            - ema_decay (float): Decay factor for EMA model.
            - fold (int): Fold index for cross-validation.
            - sub_dir (str): Directory for saving checkpoints.
            - name (str): Name prefix for saved checkpoint files.
        train_df (pd.DataFrame): Training dataset metadata and paths.
        valid_df (pd.DataFrame): Validation dataset metadata and paths.

    Returns:
        None
    """
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    model = EVA16_Model(encoder=args.Encoder, pretrained=True).to(args.device)
    ema_model = ModelEMA(model, decay=args.ema_decay, device=args.device)
    
    train_loader = DataLoader(
        SeismicDataset(train_df, 'train', transform=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    valid_loader = DataLoader(
        SeismicDataset(valid_df, 'val', transform=False),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    print(f"Train: {len(train_loader.dataset)}\nValidation: {len(valid_loader.dataset)}")

    criterion = torch.nn.L1Loss()
    optimizer = MuonWithAuxAdam(
        model, lr=args.lr, weight_decay=0.01, momentum=0.95, betas=(0.9, 0.95)
    )

    # Number of steps for the cosine annealing phase
    cosine_epochs = 25
    cosine_steps = cosine_epochs * len(train_loader)

    # 1. CosineAnnealingLR for the initial phase
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.T_0, eta_min=args.min_lr
    )

    # 2. Constant LR after cosine phase (LR reduced by factor 0.02)
    constant_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.02  # multiply base LR by 0.02
    )

    # 3. Sequential scheduler to switch between cosine and constant LR
    scheduler = SequentialLR(
        optimizer,
        schedulers=[cosine_scheduler, constant_scheduler],
        milestones=[cosine_steps]  # switch after cosine_steps
    )
    
    early_stopping = EarlyStopping(patience=3, delta=0)

    best_mape = float('inf')
    best_epoch = 0
    start_epoch = 1
    results_table = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_mae = train_one_epoch(args, model, ema_model, train_loader, criterion, optimizer, scheduler, epoch)
        valid_mae, valid_mape = valid_one_epoch(args, model, ema_model, valid_loader, criterion, epoch)
    
        results_table.append([epoch, np.mean(train_mae), valid_mape])
        print(tabulate(results_table, headers=["Epoch", "Train Loss", "Valid MAPE"], tablefmt="grid"))

        if valid_mape < best_mape:
            best_mape = valid_mape
            best_epoch = epoch
            save_checkpoint(ema_model,
                            args.sub_dir, 
                            args.name, 
                            fold=args.fold)

        early_stopping(valid_mape)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}, best epoch was {best_epoch}")
            break

    del model, optimizer, scheduler, train_loader, valid_loader
    torch.cuda.empty_cache()


def main():
    """
    Main entry point: parses arguments, loads data, and starts training.

    Command line arguments:
        --fold (int): Fold number for cross-validation.
        --sub_dir (str): Directory to save checkpoints.
        --data_dir (str): Directory containing training CSV files.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--sub_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    
    args_cli = parser.parse_args()

    args = Args()
    args.fold = args_cli.fold
    args.sub_dir = args_cli.sub_dir
    args.data_dir = args_cli.data_dir

    train_df = pd.read_csv(f"{args.data_dir}/train_fold{args.fold}.csv")
    valid_df = pd.read_csv(f"{args.data_dir}/val_fold{args.fold}.csv")

    train_and_validate(args, train_df, valid_df)


if __name__ == "__main__":
    main()

