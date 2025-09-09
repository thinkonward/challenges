import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import gc
from utils import *

def train_one_epoch(args, model, ema_model, dataloader, criterion, optimizer, scheduler, epoch):
    """
    Train the model for one epoch with mixed precision and optional EMA updating.

    This function performs a complete training pass over the provided dataloader, including:
    - Forward pass
    - Loss computation (using a custom MAPE loss)
    - Backpropagation with gradient scaling for mixed precision
    - Gradient clipping to avoid exploding gradients
    - Optimizer step and scheduler update
    - Optional Exponential Moving Average (EMA) model update
    - Logging of training loss using tqdm progress bar

    Args:
        args (Namespace): Configuration including device information.
        model (torch.nn.Module): Model to train.
        ema_model (torch.nn.Module or None): EMA model wrapper for stable evaluation (optional).
        dataloader (DataLoader): Provides batches of (images, masks) for training.
        criterion (callable): Loss function (not directly used here since mape_loss is used).
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler to update per batch.
        epoch (int): Current epoch index, used for logging.

    Returns:
        float: Average training loss over all batches in the epoch.

    Notes:
        - Uses torch.cuda.amp for mixed precision to speed up training and reduce memory.
        - Gradient clipping with max norm 3.0 to stabilize training.
        - Loss computed via custom MAPE loss function (`mape_loss`).
    """
    model.train()
    losses = []
    scaler = GradScaler()  # For mixed precision

    optimizer.zero_grad()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} | Train", dynamic_ncols=True, leave=True, mininterval=5.0)

    for step, (images, masks) in pbar:
        images = images.to(args.device)
        masks = masks.to(args.device)

        with autocast():  # Mixed precision context
            y_pred = model(images)
            loss = mape_loss(y_pred, masks)  # Use your custom MAPE loss here

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # Clip gradients

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        if ema_model is not None:
            ema_model.update(model)  # Update EMA weights if provided

        losses.append(loss.item())

        pbar.set_postfix({
            "Epoch": epoch,
            "Train MAPE Loss": loss.item(),
        })
        pbar.update(1)

    return np.mean(losses)


@torch.no_grad()
def valid_one_epoch(args, model, ema_model, dataloader, criterion, epoch):
    """
    Validate the model for one epoch, computing average loss and MAPE score.

    This function:
    - Evaluates model performance on the validation dataset without gradient computation.
    - Uses mixed precision inference for efficiency.
    - Supports using an EMA model for predictions if provided.
    - Calculates batch-wise MAPE metric.
    - Aggregates all predictions and ground truths for further analysis.
    - Clears CUDA cache after evaluation.

    Args:
        args (Namespace): Configuration including device information.
        model (torch.nn.Module): Model to evaluate.
        ema_model (torch.nn.Module or None): EMA model for stable evaluation (optional).
        dataloader (DataLoader): Validation data loader providing batches.
        criterion (callable): Loss function (not directly used here since mape_loss is used).
        epoch (int): Current epoch index, for logging.

    Returns:
        tuple:
            - float: Average validation loss over all batches.
            - float: Average MAPE score over all samples in validation.

    Notes:
        - Uses torch.no_grad() to disable gradients for faster evaluation.
        - Uses a custom MAPE loss and metric (`mape_loss`, `calculate_mape`).
    """
    model.eval()
    losses = []
    mape_list = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} | Validation",
                leave=False, dynamic_ncols=True, position=0, ncols=80, mininterval=5.0)

    for step, (images, masks) in pbar:
        images = images.to(args.device)
        masks = masks.to(args.device)

        with autocast():
            if ema_model is not None:
                y_pred = ema_model.module(images)  # EMA wrapped model call
            else:
                y_pred = model(images)

            loss = mape_loss(y_pred, masks)

        # Compute MAPE per sample in batch
        for i in range(images.size(0)):
            y_pred_single = y_pred[i]
            mask = masks[i]
            mape = calculate_mape(mask.cpu().numpy(), y_pred_single.cpu().numpy())
            mape_list.append(mape)

        losses.append(loss.item())

        pbar.set_postfix({
            "Epoch": epoch,
            "Val MAPE Loss": np.mean(losses),
            "MAPE Score": np.mean(mape_list),
        })

    gc.collect()
    torch.cuda.empty_cache()

    return np.mean(losses), np.mean(mape_list)
