from typing import Tuple, Dict

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for training segmentation models.

    This loss computes the Dice coefficient between the predicted and true masks,
    which is particularly useful for imbalanced datasets.

    Args:
        smooth (float, optional): Smoothing factor to prevent division by zero. Defaults to 1e-7.
    """

    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Dice Loss.

        Args:
            y_pred (torch.Tensor): Predicted masks with shape (N, C, H, W).
            y_true (torch.Tensor): Ground truth masks with shape (N, C, H, W).

        Returns:
            torch.Tensor: Computed Dice Loss.
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} vs y_true shape {y_true.shape}")

        # Flatten the tensors to shape (N, -1)
        y_pred = y_pred.flatten(1)
        y_true = y_true.flatten(1)

        # Compute intersection and union
        intersection = (y_pred * y_true).sum(dim=1)
        union = y_pred.sum(dim=1) + y_true.sum(dim=1)

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined training loss and return with metrics.

    Combines Dice Loss and Binary Cross Entropy (BCE) Loss for robust training.

    Args:
        pred (torch.Tensor): Model predictions before sigmoid activation with shape (N, C, H, W).
        target (torch.Tensor): Ground truth masks with shape (N, C, H, W).

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]:
            - Total loss (Dice Loss + BCE Loss).
            - Dictionary containing individual loss metrics.
    """
    dice_loss_fn = DiceLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # Compute Dice Loss
    dice_loss = dice_loss_fn(torch.sigmoid(pred), target)

    # Compute BCE Loss
    bce_loss = bce_loss_fn(pred, target)

    # Combine losses with equal weighting
    total_loss = 0.5 * dice_loss + 0.5 * bce_loss

    metrics = {
        'train_dice_loss': dice_loss.item(),
        'train_bce_loss': bce_loss.item(),
    }

    return total_loss, metrics


def compute_2d_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute 2D Dice score.

    Args:
        pred (torch.Tensor): Predicted masks after sigmoid activation with shape (N, C, H, W).
        target (torch.Tensor): Ground truth masks with shape (N, C, H, W).

    Returns:
        float: Computed 2D Dice score.
    """
    dice_loss_fn = DiceLoss()
    dice_score = 1 - dice_loss_fn(pred, target)
    return dice_score.item()


def calculate_dice_3d_gpu(pred_volume: torch.Tensor, true_volume: torch.Tensor) -> torch.Tensor:
    """
    Calculate 3D Dice score efficiently on GPU.

    Args:
        pred_volume (torch.Tensor): Predicted 3D masks with shape (D, H, W).
        true_volume (torch.Tensor): Ground truth 3D masks with shape (D, H, W).

    Returns:
        torch.Tensor: Computed 3D Dice score.
    """
    if pred_volume.shape != true_volume.shape:
        raise ValueError(f"Shape mismatch: pred_volume shape {pred_volume.shape} vs true_volume shape {true_volume.shape}")

    # Ensure tensors are contiguous for optimal memory access
    pred_volume = pred_volume.contiguous()
    true_volume = true_volume.contiguous()

    # Flatten the tensors
    pred_flat = pred_volume.view(-1)
    true_flat = true_volume.view(-1)

    # Compute intersection and union
    intersection = (pred_flat * true_flat).sum()
    pred_sum = pred_flat.sum()
    true_sum = true_flat.sum()

    # Compute Dice coefficient
    dice = (2. * intersection + 1e-7) / (pred_sum + true_sum + 1e-7)

    return dice
