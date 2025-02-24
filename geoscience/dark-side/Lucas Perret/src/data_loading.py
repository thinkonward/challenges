from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from torch.utils.data import DataLoader

from .dataset import SeismicDataset, seismic_collate_fn


def check_train_val_contamination(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """
    Verify that there is absolutely no overlap between train and validation volumes.
    Raises ValueError if any contamination is found.

    Args:
        train_df (pd.DataFrame): DataFrame containing training data.
        val_df (pd.DataFrame): DataFrame containing validation data.

    Raises:
        ValueError: If overlapping volumes are found between training and validation sets.
    """
    train_volumes = set(train_df["sample_id"].unique())
    val_volumes = set(val_df["sample_id"].unique())
    common_volumes = train_volumes.intersection(val_volumes)

    if common_volumes:
        overlap_details = []
        for volume in sorted(common_volumes):
            train_parts = sorted(
                train_df[train_df["sample_id"] == volume]["data_part"].unique()
            )
            val_parts = sorted(
                val_df[val_df["sample_id"] == volume]["data_part"].unique()
            )
            overlap_details.append(f"\nVolume {volume}:")
            overlap_details.append(f"    - In train parts: {train_parts}")
            overlap_details.append(f"    - In val parts: {val_parts}")

        error_message = (
            f"Found {len(common_volumes)} volumes present in both train and "
            f"validation sets!\nOverlapping volumes: {sorted(common_volumes)}"
            f"\nDetailed overlap:"
            f"\n{''.join(overlap_details)}"
            f"\nTotal train volumes: {len(train_volumes)}"
            f"\nTotal val volumes: {len(val_volumes)}"
        )
        raise ValueError(error_message)

    print("\nCross-contamination check passed ✓")
    print(f"Train volumes: {len(train_volumes)}")
    print(f"Val volumes: {len(val_volumes)}")
    print(f"Intersection: {len(common_volumes)} volumes")


def filter_data_by_axis(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """
    Filter DataFrame based on the specified axis.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        axis (str): The axis to filter by ('x', 'y', 'z', or 'xy').

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if axis == "xy":
        filtered_df = df[df["axis"].isin(["x", "y"])].copy()
    else:
        filtered_df = df[df["axis"] == axis].copy()
    return filtered_df


def create_train_val_dataloaders(
    root_dir: Union[str, Path],
    df: pd.DataFrame,
    train_volumes: List[str],
    val_volumes: List[str],
    train_axis: str,
    val_axis: str,
    batch_size: int = 32,
    num_workers: int = 4,
    apply_augmentation: bool = False,
    nchans: int = 9,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        root_dir (Union[str, Path]): Root directory containing the data.
        df (pd.DataFrame): DataFrame containing dataset information.
        train_volumes (List[str]): List of volume identifiers for training.
        val_volumes (List[str]): List of volume identifiers for validation.
        train_axis (str): Axis to filter training data by ('x', 'y', 'z', or 'xy').
        val_axis (str): Axis to filter validation data by ('x', 'y', 'z', or 'xy').
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
        apply_augmentation (bool, optional): Whether to apply data augmentation. Defaults to False.
        nchans (int, optional): Number of channels in the data. Defaults to 9.

    Raises:
        ValueError: If `train_axis` or `val_axis` is not one of 'x', 'y', 'z', or 'xy'.

    Returns:
        Tuple[DataLoader, DataLoader]: Tuple containing the training and validation DataLoaders.
    """
    valid_axes = {"x", "y", "z", "xy"}

    if val_axis not in valid_axes:
        raise ValueError(f"val_axis must be one of {valid_axes}, got '{val_axis}'.")

    if train_axis not in valid_axes:
        raise ValueError(f"train_axis must be one of {valid_axes}, got '{train_axis}'.")

    print("\nCreating dataloaders")
    print(f"Validation axis: {val_axis}")
    print(f"Training axis: {train_axis}")

    # Filter data based on specified axes
    val_df = filter_data_by_axis(df, val_axis)
    train_df = filter_data_by_axis(df, train_axis)

    # Create separate DataFrames for train and validation volumes
    train_df = train_df[train_df["sample_id"].isin(train_volumes)].copy()
    val_df = val_df[val_df["sample_id"].isin(val_volumes)].copy()

    # Filter out samples without labels
    for mode, current_df in [("train", train_df), ("valid", val_df)]:
        labeled_df = current_df[current_df["has_labels"] == True].copy()
        if mode == "train":
            train_df = labeled_df
        else:
            val_df = labeled_df

    print(
        f"Train set: {len(train_df)} frames from "
        f"{len(train_df['sample_id'].unique())} volumes"
    )
    print(
        f"Val set: {len(val_df)} frames from "
        f"{len(val_df['sample_id'].unique())} volumes"
    )

    # Verify no overlap between train and validation sets
    check_train_val_contamination(train_df, val_df)

    # Initialize datasets
    train_dataset = SeismicDataset(
        root_dir=root_dir,
        df=train_df,
        mode="train",
        apply_augmentation=apply_augmentation,
        nchans=nchans,
    )

    val_dataset = SeismicDataset(
        root_dir=root_dir,
        df=val_df,
        mode="valid",
        nchans=nchans,
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seismic_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=seismic_collate_fn,
    )

    return train_loader, val_loader
