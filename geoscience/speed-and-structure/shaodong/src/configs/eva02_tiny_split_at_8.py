# ruff: noqa
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from detectron2.config import LazyCall as L

from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    CosineParamScheduler,
)

from data.dataset import SASDataset
from models.custom_eva_tiny_split_at_8 import SASModel
from optim.muon import MuonWithAuxAdam


# model config
model = L(SASModel)(pretrained=True)
optimizer = L(MuonWithAuxAdam)(
    lr=20e-5,
    weight_decay=0.01,
    momentum=0.95,
    betas=(0.9, 0.95)
 )


# data config
def build_dataset(data_root, txt_file, mode):
    dataset = SASDataset(
        data_root=data_root,
        data_txt=txt_file,
        mode=mode,
    )
    return dataset

def build_data_loader(dataset, batch_size, num_workers, training=True):
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=training, drop_last=training)
    else:
        sampler = None
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=training and not dist.is_initialized(),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=training,
        persistent_workers=True,
        sampler=sampler,
        pin_memory=True,
    )


dataloader = dict(
    train=L(build_data_loader)(
        dataset=build_dataset(data_root="./data/train_data/", 
                              txt_file="./train_txt/train_f0.txt", 
                              mode="train"),
        batch_size=8,
        num_workers=8,
        training=True,
    ),
    val=L(build_data_loader)(
        dataset=build_dataset(data_root="./data/train_data/", 
                              txt_file="./train_txt/val_f0.txt", 
                              mode="test"),
        batch_size=16,
        num_workers=1,
        training=False,
    ),
)

max_epochs = 80
lr_multiplier = L(CompositeParamScheduler)(
    schedulers=[
        L(ConstantParamScheduler)(value=1),
        L(CosineParamScheduler)(start_value=1, end_value=0.001),
    ],
    lengths=[0.5, 0.5],
    interval_scaling=["rescaled", "rescaled"],
)

train = dict(
    device="cuda",
    max_epochs=max_epochs,
    log_interval=5,
    checkpoint_interval=200,
    eval_interval=1,
    log_buffer_size=1,
    clip_grad=False,
    seed=3,
    compile=dict(
        mode="reduce-overhead",
    ),
    # compile=None,
)