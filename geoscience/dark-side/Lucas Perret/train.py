import os
import argparse
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.callback import BestMetricsCallback
from src.data_loading import create_train_val_dataloaders
from src.model import SeismicFaultDetector
from src.utils import load_ckpt, save_config


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Seismic Fault Detection Training Script")

    # Basic arguments
    parser.add_argument("root_dir", type=str, help="Root directory containing 2d slices data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_det_algo", action="store_true", help="Use detection algorithm")

    # Available architectures
    AVAILABLE_ARCHITECTURES = ["unet", "unetpp", "umamba"]

    # Model architecture arguments
    parser.add_argument(
        "--archi",
        type=str,
        default="unetpp",
        choices=AVAILABLE_ARCHITECTURES,
        help="Model architecture to use",
    )
    parser.add_argument("--num_classes", type=int, default=1, help="Number of output classes")
    parser.add_argument(
        "--input_size",
        nargs=2,
        type=int,
        default=[320, 1280],
        help="Input size as height width (e.g. --input_size 320 1280)",
    )
    parser.add_argument(
        "--nchans",
        type=int,
        default=7,
        help="Number of input channels (must be odd)",
    )

    # Unet/Unetpp model parameter
    parser.add_argument(
        "--encoder_name",
        type=str,
        default=None,
        help="Encoder backbone to use (if applicable)",
    )

    # Umamba model parameters
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="Model size for models that support it (e.g., umamba)",
    )

    # Cross-validation configuration
    parser.add_argument(
        "--excluded_vol",
        type=str,
        nargs="+",
        default=None,
        help="List of volume IDs to exclude from training and validation",
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        default=6,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific folds to train on (default: all folds)",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=40,
        help="Number of bins for stratified split",
    )

    parser.add_argument(
        "--val_interval",
        type=float,
        default=1.0,
        help="Validation interval in epochs",
    )

    # Checkpoint saving configuration
    parser.add_argument(
        "--ckpt_optim_interval",
        type=float,
        default=None,
        help="Save checkpoint with optimizer state every N steps (as fraction of epoch)",
    )

    # Training configuration
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Precision for training (e.g., 16-mixed, 32)",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default=None,
        help="Compiler settings for the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.8,
        help="Gamma value for the exponential LR scheduler",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability for models that support it",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=0.5,
        help="Gradient clip val for training",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=5,
        help="Early stopping patience",
    )

    # Data processing
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Apply training augmentation",
    )

    # Axis arguments
    parser.add_argument(
        "--val_axis",
        type=str,
        required=True,
        choices=["x", "y", "z", "xy"],
        help="Axis to use for validation",
    )
    parser.add_argument(
        "--train_axis",
        type=str,
        choices=["x", "y", "z", "xy"],
        default=None,
        help="Axis to use for training (default: same as val_axis)",
    )

    # Checkpoint handling
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to load",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        default=False,
        help="Restore the training from checkpoint",
    )

    # For fast debug
    parser.add_argument(
        "--frac",
        type=float,
        default=None,
        help="Fraction of volumes to use (between 0 and 1)",
    )

    # New Arguments: Device configuration
    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        help=(
            "Accelerator to use for training. Options: 'cpu', 'gpu', 'tpu', 'hpu', 'mps', 'auto', or leave None for automatic selection."
        ),
    )

    return parser.parse_args()


def validate_args(
    nchans: int,
    encoder_name: Optional[str],
    model_size: Optional[str]
) -> None:
    """
    Validates specific command-line arguments.

    Args:
        nchans (int): Number of input channels.
        encoder_name (Optional[str]): Encoder backbone name.
        model_size (Optional[str]): Model size.

    Raises:
        ValueError: If nchans is not odd or if encoder_name and model_size conditions are not met.
    """
    if nchans % 2 != 1:
        raise ValueError(f"nchans must be odd, got {nchans}")

    if (encoder_name is None) == (model_size is None):
        raise ValueError("Exactly one of encoder_name or model_size must be specified")


def create_directory_path(
    archi: str,
    encoder_name: Optional[str],
    model_size: Optional[str],
    nchans: int,
    val_axis: str
) -> str:
    """
    Creates the directory path for saving checkpoints.

    Args:
        archi (str): Model architecture.
        encoder_name (Optional[str]): Encoder backbone name.
        model_size (Optional[str]): Model size.
        nchans (int): Number of input channels.
        val_axis (str): Validation axis.

    Returns:
        str: Directory path.
    """
    model_identifier = f"_{encoder_name}" if encoder_name else f"_{model_size}"
    dirpath = (
        f"checkpoints_{archi}{model_identifier}_nchans{nchans}_"
        f"val_axis{val_axis}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"dirpath={dirpath}")
    return dirpath


def prepare_data(
    df_path: str,
    frac: Optional[float],
    seed: int,
    excluded_vol: Optional[List[str]]
) -> pd.DataFrame:
    """
    Loads and preprocesses the data.

    Args:
        df_path (str): Path to the parquet index file.
        frac (Optional[float]): Fraction of volumes to use.
        seed (int): Random seed.
        excluded_vol (Optional[List[str]]): List of volume IDs to exclude.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_parquet(df_path)
    print(f"Initial DataFrame shape: {df.shape}")

    # Optionally select fraction of volumes for fast training debug
    if frac is not None:
        if not 0 < frac <= 1:
            raise ValueError(f"frac must be between 0 and 1, got {frac}")

        all_volumes = sorted(df["sample_id"].unique())
        n_volumes = len(all_volumes)
        n_volumes_to_keep = int(n_volumes * frac)

        np.random.seed(seed)
        selected_volumes = np.random.choice(all_volumes, size=n_volumes_to_keep, replace=False)
        df = df[df["sample_id"].isin(selected_volumes)].copy()

        print(f"\nRandomly selected {frac:.1%} of volumes:")
        print(f"- Initial number of volumes: {n_volumes}")
        print(f"- Selected number of volumes: {n_volumes_to_keep}")
        print(f"- DataFrame shape after selection: {df.shape}")

    # Remove samples without labels
    df = df[df["has_labels"]].reset_index(drop=True)
    print(f"DataFrame shape after removing unlabelled volumes: {df.shape}")

    # Handle excluded volumes if specified
    if excluded_vol:
        for vol_id in excluded_vol:
            vol_pixels = df[df["sample_id"] == vol_id]["positive_pixels"].sum() // 3
            print(f"Excluding volume {vol_id} with {vol_pixels:,} total positive pixels")
        df = df[~df["sample_id"].isin(excluded_vol)].copy()
        print(f"DataFrame shape after exclusion: {df.shape}")

    # Identify empty volumes
    empty_volumes = [
        vol_id
        for vol_id in df["sample_id"].unique()
        if df[df["sample_id"] == vol_id]["positive_pixels"].sum() == 0
    ]

    if empty_volumes:
        print(f"\nFound {len(empty_volumes)} empty volumes")
    else:
        print("\nNo empty volumes found")

    return df


def create_stratification_bins(
    df: pd.DataFrame,
    n_bins: int
) -> pd.DataFrame:
    """
    Stratifies the data into bins based on positive pixels.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        n_bins (int): Number of bins for stratification.

    Returns:
        pd.DataFrame: Volume statistics with assigned bins.
    """
    volume_stats = df[df["axis"] == "x"].groupby("sample_id")["positive_pixels"].sum().reset_index()
    volume_stats["bin"], _ = pd.qcut(
        volume_stats["positive_pixels"],
        q=n_bins,
        labels=[f"bin_{i}" for i in range(n_bins)],
        retbins=True,
    )
    return volume_stats


def main() -> None:
    """
    Main function to execute the training pipeline.
    """
    args = parse_args()
    validate_args(args.nchans, args.encoder_name, args.model_size)
    dirpath = create_directory_path(
        archi=args.archi,
        encoder_name=args.encoder_name,
        model_size=args.model_size,
        nchans=args.nchans,
        val_axis=args.val_axis
    )
    df_path = os.path.join(args.root_dir, 'dataset.parquet')
    df = prepare_data(
        df_path=df_path,
        frac=args.frac,
        seed=args.seed,
        excluded_vol=args.excluded_vol
    )
    volume_stats = create_stratification_bins(df, args.n_bins)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed)
    folds_to_train = args.folds if args.folds is not None else list(range(args.nfolds))

    print("\nSetting up cross-validation folds...")
    fold_info: Dict[int, Any] = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(volume_stats, volume_stats["bin"])):
        if fold_idx not in folds_to_train:
            continue

        print("\n" + "=" * 50)
        print(f"Training Fold {fold_idx + 1}/{args.nfolds}")
        print("=" * 50)

        train_volumes = volume_stats.iloc[train_idx]["sample_id"].tolist()
        val_volumes = volume_stats.iloc[val_idx]["sample_id"].tolist()

        train_parts = sorted(df[df["sample_id"].isin(train_volumes)]["data_part"].unique())
        val_parts = sorted(df[df["sample_id"].isin(val_volumes)]["data_part"].unique())

        print(f"\nTrain volumes: {len(train_volumes)} across {len(train_parts)} data parts")
        print(f"Val volumes: {len(val_volumes)} across {len(val_parts)} data parts")

        fold_info[fold_idx] = {
            "train_parts": train_parts,
            "val_parts": val_parts,
            "train_volumes": train_volumes,
            "val_volumes": val_volumes,
            "best_epoch": "N/A",
            "best_val_score": "N/A",
            "best_threshold": "N/A",
            "train_bin_distribution": volume_stats.iloc[train_idx]["bin"].value_counts().to_dict(),
            "val_bin_distribution": volume_stats.iloc[val_idx]["bin"].value_counts().to_dict(),
        }

        train_loader, val_loader = create_train_val_dataloaders(
            root_dir=args.root_dir,
            df=df,
            train_volumes=train_volumes,
            val_volumes=val_volumes,
            train_axis=args.train_axis,
            val_axis=args.val_axis,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            apply_augmentation=args.aug,
            nchans=args.nchans,
        )

        model = SeismicFaultDetector(
            args.archi,
            args.val_axis,
            args.nchans,
            args.num_classes,
            args.lr,
            args.scheduler_gamma,
            encoder_name=args.encoder_name,
            model_size=args.model_size,
            input_size=args.input_size,
            dropout=args.dropout,
            _compile=args.compile,
        )

        if args.ckpt:
            print(f"\nLoading checkpoint from: {args.ckpt}")
            load_ckpt(model, args.ckpt, load_optim=False)

        callbacks = [
            ModelCheckpoint(
                monitor="val_dice_3d",
                dirpath=dirpath,
                filename=f"fold{fold_idx}-best-model-{{epoch:02d}}-{{val_dice_3d:.4f}}",
                save_top_k=1,
                mode="max",
                save_weights_only=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                monitor="val_dice_3d",
                patience=args.early_stop,
                mode="max",
                verbose=True,
            ),
            BestMetricsCallback(
                fold_idx=fold_idx,
                fold_info=fold_info,
                train_parts=train_parts,
                val_parts=val_parts,
                train_volumes=train_volumes,
                val_volumes=val_volumes,
                dirpath=dirpath,
                args=args
            ),
        ]

        if args.ckpt_optim_interval is not None:
            steps_per_epoch = len(train_loader)
            save_every_n_steps = math.ceil(steps_per_epoch * args.ckpt_optim_interval)
            print(f"ckpt_optim_interval={args.ckpt_optim_interval}, save_every_n_steps={save_every_n_steps}")
            callbacks.append(
                ModelCheckpoint(
                    dirpath=dirpath,
                    filename="last",
                    every_n_train_steps=save_every_n_steps,
                    save_top_k=1,
                    save_weights_only=False,
                )
            )

        trainer_kwargs = {
            "max_epochs": args.epochs,
            "accelerator": 'auto',
            "precision": args.precision,
            "callbacks": callbacks,
            "logger": True,
            "log_every_n_steps": 10,
            "val_check_interval": args.val_interval,
            "gradient_clip_val": args.gradient_clip_val,
        }

        trainer = pl.Trainer(**trainer_kwargs)

        trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt if args.restore else None)

    save_config(args, dirpath, fold_info)

    print("\nTraining completed for all selected folds!")
    print("\nSummary of results:")
    for fold_idx in sorted(fold_info.keys()):
        info = fold_info[fold_idx]
        print(f"\nFold {fold_idx}:")
        print(f"Best epoch: {info['best_epoch']}")
        print(f"Best val score: {info['best_val_score']}")
        print(f"Best threshold: {info['best_threshold']}")

    print(f"Root results dir: {dirpath}")


if __name__ == "__main__":
    main()
