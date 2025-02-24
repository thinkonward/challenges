import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler
#from volumentations import *
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
#from skimage.metrics import structural_similarity as ssim

import os
import glob
import random
import numpy as np
#import pandas as pd
import re
#import monai
#import scipy.ndimage

from darkside_dataset_4c import DarksideTrainDataset, DarksideValidDataset
import segmentation_models_pytorch_3d as smp


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)#'12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def downsample_volume(volume, target_size, order=1):
    # Calculate the zoom factors for each dimension
    zoom_factors = [n / o for n, o in zip(target_size, volume.shape)]
    # Apply the downsampling
    return scipy.ndimage.zoom(volume, zoom_factors, order=order)  # order=1 for linear interpolation

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def train_one_epoch(model, dataloader, optimizer, loss_function, scaler, scheduler, args, rank):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", disable=(rank!=0))
    
    for images, masks in pbar:
        images, masks = images.to(rank), masks.to(rank)
        #images, masks = mosaic_augmentation_resize(images, masks)
        
        optimizer.zero_grad()

        with autocast():  # Automatic mixed-precision
            outputs = model(images)
            loss = loss_function(outputs, masks)
            
        scaler.scale(loss).backward()

        if args.norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.norm, norm_type=2)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step the scheduler at every iteration
        scheduler.step()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())

        # NaN Detection
        if not np.isfinite(loss.item()):
            torch.save(images, 'nan_input.pth')
            torch.save(masks, 'nan_label.pth')
            torch.save(outputs, 'nan_pred.pth')
            raise ValueError('NaN Detected, Exit Training')

        #break

    epoch_loss = running_loss / len(dataloader.dataset)
    return torch.tensor(epoch_loss).to(rank)

def validate(model, dataloader, loss_function, rank, args):
    model.eval()
    running_loss = 0.0
    running_ssim = 0.0
    #dice_total = 0.0
    #iou_total = 0.0
    pbar = tqdm(dataloader, desc="Validating", disable=(rank!=0))
    dice_metric = monai.metrics.DiceMetric(ignore_empty=False)
    
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(rank), masks.to(rank)
            #images = images.to(rank)

            with autocast():  # Automatic mixed-precision
                outputs = model(images)
                loss = loss_function(outputs, masks)#nn.functional.binary_cross_entropy_with_logits(outputs, masks)
            
            running_ssim += dice_metric(outputs.sigmoid()>0.5, masks).sum()
            #print(dice_metric(outputs.sigmoid()>0.5, masks))
            
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_ssim = running_ssim / len(dataloader.dataset)
    
    return torch.tensor(epoch_loss).to(rank), torch.tensor(epoch_ssim).to(rank)#, epoch_dice, epoch_iou

def main(rank, world_size, args):
    
    #print(1)
    #print([int(i) for i in args.patch_size.split('_')])
    
    args.epochs = args.epochs // args.val_interv
    args.warmup_epochs = args.warmup_epochs // args.val_interv
    
    setup(rank, world_size, args.port)
    seed_torch(45)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    model = smp.Unet(
        encoder_name="efficientnet-b4", # choose encoder, e.g. resnet34
        in_channels=4,                  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
        encoder_weights='imagenet',
        classes=1,                      # model output channels (number of classes in your dataset)
        #decoder_attention_type='scse',
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    #model = torch.compile(model)
    #model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in torch.load('runs/w_3_d_1_warm_add_epoch_200_4e-4/best.pth').items()})
    #model.load_state_dict(torch.load('runs/w_3_d_1_warm_add_epoch_200_4e-4/best.pth', map_location=f'cuda:{rank}'))
    
    
    # Example of how to use the dataset and dataloader
    train_dataset = DarksideTrainDataset(patch_size=list(map(int, args.patch_size.split('_'))), 
                                        n_repeats=args.val_interv)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              sampler=train_sampler, 
                              persistent_workers=True,
                              num_workers=8, 
                              pin_memory=True,
                              drop_last=True)
    if not args.noeval:
        val_dataset = DarksideValidDataset(patch_size=list(map(int, args.patch_size.split('_'))))
        
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(val_dataset, 
                                batch_size=args.batch_size, 
                                sampler=val_sampler, 
                                persistent_workers=True,
                                num_workers=8, 
                                pin_memory=True)
    
    # Loss, Optimizer, Scheduler, and AMP Setup
    loss_function = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.000101111313663464)
    #scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=args.eta_min)  # Assuming train_loader is defined, T_max is total steps
    
    steps_per_epoch = len(train_loader)
    total_training_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    remaining_steps = total_training_steps - warmup_steps
    
    if args.warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=args.eta_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=args.eta_min)
    
    
    scaler = GradScaler()  # For AMP

    epochs = args.epochs
    #model_save_path = args.model_save_path
    log_dir = args.log_dir

    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
        best_score = 0
    
    for epoch in range(epochs):
        # Train

        train_sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, scaler, scheduler, args, rank)
            
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss = train_loss.item()

        # Validate
        #if (epoch+1) % args.val_interv == 0 and not args.noeval:
        if not args.noeval:
            val_loss, val_ssim = validate(model, val_loader, loss_function, rank, args)
            #print(val_loss, val_l1_loss)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_ssim, op=dist.ReduceOp.SUM)
            val_loss, val_ssim = val_loss.item(), val_ssim.item()


        
        if rank == 0:

            print('Epoch:{} Train Loss:{}'.format(epoch*args.val_interv, train_loss))
            writer.add_scalar('Loss/train', train_loss, epoch*args.val_interv)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch*args.val_interv)
             

            #if (epoch+1) % args.val_interv == 0 and not args.noeval:
            if not args.noeval:
                print('Epoch:{} Eval Loss:{} Eval Dice:{}'.format(epoch*args.val_interv, val_loss, val_ssim))
                writer.add_scalar('Loss/val', val_loss, epoch*args.val_interv)
                writer.add_scalar('Dice/val', val_ssim, epoch*args.val_interv)

        
                # Save best model
                if val_ssim > best_score:
                    best_score = val_ssim
                    torch.save(model.state_dict(), args.log_dir+'/best.pth')
                    
            if args.noeval:
                torch.save(model.state_dict(), args.log_dir+'/epoch_{}.pth'.format(epoch*args.val_interv))
        
    cleanup()
    if rank == 0:
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
    parser.add_argument('--port', default=12345, type=int, help='DDP Port')
    parser.add_argument('--lr', default=4e-4, type=float, help='Initial learning rate')
    parser.add_argument('--eta_min', default=1e-7, type=float, help='Cosine minimum lr')
    parser.add_argument('--sub', default=94.64, type=float, help='Input subtract factor')
    parser.add_argument('--width', default=1, type=int, help='A Width Multiply Factor to Original SegResNet')
    parser.add_argument('--depth', default=1, type=int, help='A Depth Multiply Factor to Original SegResNet')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='Number of warmup epochs')
    #parser.add_argument('--fold', default=0, type=int, help='Number of kfold')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--patch_size', default='192 192 592', type=str, help='Patch size')
    parser.add_argument('--norm', default=0., type=float, help='Gradient Norm Type')
    parser.add_argument('--val_interv', default=1, type=int, help='Valid Interval')
    parser.add_argument('--log_dir', default='./runs', type=str, help='Directory for tensorboard logs')
    parser.add_argument('--noeval',default=False,action='store_true',help='Turn off Eval Process')

    
    args = parser.parse_args()
    world_size = args.world_size

    torch.multiprocessing.spawn(main, args=(world_size, args, ), nprocs=world_size)