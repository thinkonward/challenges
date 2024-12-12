import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from data_aug import dataset_offset
from unet3d import *
import losses
import h5py
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchmetrics.functional.image import structural_similarity_index_measure
import gc
import yaml

# Load configuration from YAML
with open('scripts/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set training parameters from the configuration file
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
num_epochs = config['training']['num_epochs']
warmup_epochs = config['training']['warmup_epochs']
weight_decay = float(config['training']['weight_decay'])
eta_min = float(config['training']['eta_min'])

betas = tuple(config['optimizer']['betas'])
eps = float(config['optimizer']['eps'])
start = config['data']['start']
end = config['data']['end']
warmup_multiplier = config['scheduler']['warmup_multiplier']

# Prepare datasets
dataset_0 = dataset_offset(slice_offset=0, transpose=False, flip_vertical=False, flip_horizontal=False, start=start, end=end, increment=1)
dataset_1 = dataset_offset(slice_offset=50, transpose=True, flip_vertical=True, flip_horizontal=True, start=start, end=end, increment=1)
dataset_2 = dataset_offset(slice_offset=100, transpose=False, flip_vertical=True, flip_horizontal=True, start=start, end=end, increment=1)
dataset_3 = dataset_offset(slice_offset=0, transpose=False, flip_vertical=True, flip_horizontal=False, start=start, end=end, increment=1)
dataset_4 = dataset_offset(slice_offset=60, transpose=True, flip_vertical=False, flip_horizontal=True, start=start, end=end, increment=1)
dataset_5 = dataset_offset(slice_offset=120, transpose=True, flip_vertical=True, flip_horizontal=True, start=start, end=end, increment=1)

# Concatenate datasets
combined_dataset = ConcatDataset([dataset_0, dataset_1, dataset_2, dataset_3, dataset_4, dataset_5])

# Create DataLoader
train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
print("Data loaded")

# Define the model
model = ResidualUNetSE3D(in_channels=1, out_channels=1, f_maps=[16, 32, 64, 128, 256, 512, 1024])
print("Model loaded successfully")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model).to(device)
print(f"Using {torch.cuda.device_count()} GPUs")

# Set seeds for reproducibility
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Define optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=eta_min)
scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_multiplier, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

# Define SSIM loss function
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        data_range = img2.max() - img2.min()
        if data_range < 0.01:
            data_range = 0.01
        ssim = structural_similarity_index_measure(img1, img2, data_range=data_range)
        return 1 - ssim

# Define combined loss function
def loss_forward(image1, image2):
    ssim_loss = SSIMLoss()
    mse_loss = nn.MSELoss()
    criterion_edge = losses.EdgeLoss()

    loss1 = ssim_loss(image1, image2)
    loss2 = mse_loss(image1, image2)
    loss_edge = criterion_edge(image1, image2)

    total_loss = loss1 + loss2 + loss_edge
    return total_loss

# Memory cleanup function
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Training loop
train_losses = []
best_train_loss = float('inf')

for epoch in range(num_epochs):
    free_memory()

    epoch_start_time = time.time()
    train_loss = 0.0
    model.train()

    for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        slice_loss = sum(loss_forward(outputs[:, :, :, idx, :], labels[:, :, :, idx, :]) for idx in range(0, 300, 3)) / 100
        slice_loss += sum(loss_forward(outputs[:, :, :, :, idx], labels[:, :, :, :, idx]) for idx in range(0, 300, 3)) / 100
        slice_loss += sum(loss_forward(outputs[:, :, idx, :, :], labels[:, :,idx, :, :]) for idx in range(0, 200, 4)) / 50
        slice_loss.backward()
        optimizer.step()

        train_loss += slice_loss.item()

        del outputs, labels, images, slice_loss
        free_memory()

    average_train_loss = train_loss / len(train_dataloader)
    train_losses.append(average_train_loss)

    scheduler.step()

    print(f"Epoch {epoch + 1} - Loss: {average_train_loss:.6f}, LR: {scheduler.get_lr()[0]:.6e}, Time: {time.time() - epoch_start_time:.2f}s")

    # Save checkpoint
    if epoch % 10 == 0:
        model_path = f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Checkpoint saved: {model_path}")

    # Save best model
    if average_train_loss < best_train_loss:
        best_train_loss = average_train_loss
        model_path = f'best_model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved: {model_path}")
