"""
Inference Script for EVA-16 Seismic Models with Weighted Fold Ensembling

This script:
1. Loads multiple EVA-16 model variants trained across multiple folds.
2. Performs inference on the holdout test dataset using:
   - Weighted mean ensembling of multiple models within each fold.
   - Mean aggregation of predictions across folds.
3. Saves predictions into a `.npz` submission file.

"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.models.model_EVA_16_Large_Split_10_Multi_MHA_4_heads import (
    EVA16_MHA as Eva_16_Multi_Split_10_MHA_4_heads)
from src.models.model_EVA_16_Large_Split_10_Multi_MHA_2_heads import (
    EVA16_MHA as Eva_16_Split_10_MHA_2_heads)
from src.models.model_EVA_16_Large_Split_9_Multi_MHA_4_heads import (
    EVA16_MHA as Eva_16_Split_9_MHA_4_heads)
from src.models.model_EVA_16_Large_Split_10_Single_MHA_4_heads import (
    EVA16_MHA as Eva_16_Single_Split_10_MHA_4_heads)

from src.dataset import SeismicDataset
from src.utils import *


# ------------------------------
# ✅ Inference Configuration
# ------------------------------
class args:
    """
    Static configuration for inference.

    Attributes:
        device (str): 'cuda' if GPU available, else 'cpu'.
        ckpt_root (str): Path to the root folder containing model checkpoints.
        sub_path (str): Path to save the final submission file (.npz).
        test_path (str): Path to the CSV file for test dataset.
        num_workers (int): Number of CPU workers for DataLoader.
        num_folds (int): Number of folds used for ensembling.
        val_batch_size (int): Batch size used during inference.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    ckpt_root = "../speed_structure_checkpoints/Checkpoints"
    sub_path = "../Submissions/Final_holdout_test_sub.npz"
    test_path = "../data/test.csv"
    num_workers = os.cpu_count()
    num_folds = 5
    val_batch_size = 16


# ------------------------------
# ✅ Create EVA Model
# ------------------------------
@torch.no_grad()
def create_eva_model(args, model_name, fold):
    """
    Create and load a trained EVA model for a given fold.

    Args:
        args (args): Inference configuration object.
        model_name (str): Model variant name (key in model_root dict).
        fold (int): Fold index to load the corresponding checkpoint.

    Returns:
        torch.nn.Module: Loaded EVA model in evaluation mode.
    """
    # Map model names to checkpoint folder paths
    model_root = {
        "Multi_Split_10_MHA_4_heads": f"{args.ckpt_root}/EVA_16_Large_Split_10_Dual_MHA_4_heads",
        "Split_10_MHA_2_heads": f"{args.ckpt_root}/EVA_16_Large_Split_10_Dual_MHA_2_heads",
        "Split_9_MHA_4_heads": f"{args.ckpt_root}/EVA_16_Large_Split_9_Dual_MHA_4_heads",
        "Single_Split_10_MHA_4_heads": f"{args.ckpt_root}/EVA_16_Large_Split_10_Single_MHA_4_heads",
    }

    model_fold_path = os.path.basename(model_root[model_name])
    checkpoint_path = os.path.join(model_root[model_name], f"{model_fold_path}_fold_{fold}.pt")

    # Instantiate the correct model variant
    if model_name == 'Multi_Split_10_MHA_4_heads':
        model = Eva_16_Multi_Split_10_MHA_4_heads(
            encoder="eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
            pretrained=True,
            mode='test'
        )
    elif model_name == 'Split_10_MHA_2_heads':
        model = Eva_16_Split_10_MHA_2_heads(
            encoder="eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
            pretrained=True,
            mode='test'
        )
    elif model_name == 'Split_9_MHA_4_heads':
        model = Eva_16_Split_9_MHA_4_heads(
            encoder="eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
            pretrained=True,
            mode='test'
        )
    elif model_name == 'Single_Split_10_MHA_4_heads':
        model = Eva_16_Single_Split_10_MHA_4_heads(
            encoder="eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
            pretrained=True,
            mode='test'
        )

    # Load EMA model weights from checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['ema_model'])

    # Send to device & eval mode
    model = model.to(args.device)
    model.eval()

    return model

@torch.no_grad()
# @torch.autocast()
def load_all_folds_models(args):
    """
    Load all fold-specific model sets into memory.

    For each fold, loads a specific combination of three EVA-16 variants
    based on fold number. Fold 0 uses a slightly different variant set.

    Args:
        args (args): Inference configuration.

    Returns:
        list[list[torch.nn.Module]]: List of folds, each containing 3 loaded models.
    """
    folds_models = []

    for fold in range(args.num_folds):
        print(fold)
        if fold in [1, 2, 3, 4]:
            folds_models.append([
                create_eva_model(args, 'Multi_Split_10_MHA_4_heads', fold).half(),
                create_eva_model(args, 'Split_10_MHA_2_heads', fold).half(),
                create_eva_model(args, 'Single_Split_10_MHA_4_heads', fold).half(),
            ])
        elif fold == 0:
            folds_models.append([
                create_eva_model(args, 'Multi_Split_10_MHA_4_heads', fold).half(),
                create_eva_model(args, 'Split_9_MHA_4_heads', fold).half(),
                create_eva_model(args, 'Single_Split_10_MHA_4_heads', fold).half(),
            ])
            
        print('folds_models class', len(folds_models))

    return folds_models


@torch.no_grad()
# @torch.autocast()
def inference_folds(folds_models, dataloader, args, per_fold_model_weights=None):
    """
    Run inference with weighted mean across models per fold, then
    average predictions across folds.

    Args:
        folds_models (list[list[torch.nn.Module]]): Nested list of models per fold.
        dataloader (DataLoader): Test DataLoader.
        args (args): Inference configuration.
        per_fold_model_weights (list[list[float]]): Model weights per fold for intra-fold ensembling.
    """
    device = args.device

    # Ensure all models are in eval mode
    for fold_models in folds_models:
        for m in fold_models:
            m.eval()
    print('folds_models', len(folds_models))
    input('done')

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predicting', ncols=150)

    for step, data in pbar:
        images = data["image"].to(device)
        ids = data["id"]

        folds_preds = []

        for fold_idx, fold_models in enumerate(tqdm(folds_models, desc="Folds")):
            preds_list = []

            for model in tqdm(fold_models, desc=f"Models in fold {fold_idx}", leave=False):
                with autocast():
                    preds = model(images)  # (B, ...)
                    preds_list.append(preds.unsqueeze(0))  # (1, B, ...)

            stacked_preds = torch.cat(preds_list, dim=0)  # (n_models, B, ...)
            model_weights_tensor = torch.tensor(per_fold_model_weights[fold_idx], device=device)
            model_weights_tensor /= model_weights_tensor.sum()  # normalize

            w = model_weights_tensor.view(-1, *([1] * (stacked_preds.ndim - 1)))
            weighted_mean = torch.sum(stacked_preds * w, dim=0)  # (B, ...)
            folds_preds.append(weighted_mean.unsqueeze(0))  # (1, B, ...)

        folds_preds = torch.cat(folds_preds, dim=0)  # (n_folds, B, ...)
        batch_preds = torch.mean(folds_preds, dim=0)  # Final mean across folds

        for idx in range(len(images)):
            sample_id = ids[idx]
            y_pred = batch_preds[idx].cpu().numpy().astype(np.float64).squeeze()
            create_submission(sample_id, y_pred, args.sub_path)

        gc.collect()
        torch.cuda.empty_cache()
        
    print(f"{args.sub_path} has been created")


def inference_submission(args, test_df):
    """
    Main inference routine:
    1. Load all models for each fold.
    2. Create test DataLoader.
    3. Run inference across all folds and models.

    Args:
        args (args): Inference configuration.
        test_df (pd.DataFrame): Test CSV with image IDs and metadata.
    """
    models = load_all_folds_models(args)

    test_dataset = SeismicDataset(test_df, 'test', transform=False)
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=args.val_batch_size,
        shuffle=False
    )

    per_fold_model_weights = [
        [0.8, 0.5, 0.3],
        [0.8, 0.5, 0.3],
        [0.8, 0.5, 0.3],
        [0.8, 0.5, 0.3],
        [0.8, 0.5, 0.3],
    ]
    inference_folds(models, test_loader, args, per_fold_model_weights)

    torch.cuda.empty_cache()


def main():
    """
    Entry point:
    - Reads the test CSV.
    - Runs inference submission generation.
    """
    test_df = pd.read_csv(args.test_path)
    inference_submission(args, test_df)


if __name__ == "__main__":
    main()
