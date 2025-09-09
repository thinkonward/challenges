import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils import (
    create_ids, 
    create_df,
    extract_features,
    cluster_velocity_models,
    predict_cluster_velocity_models
)

def main(args):
    """
    Prepare stratified K-Fold CSV files for training and validation.

    Steps:
    1. Load training datasets from original and extended folders.
    2. Perform K-Means clustering on training data to identify seismic groups.
    3. Assign seismic groups to extended training data using the trained cluster model.
    4. Use StratifiedKFold to split the original training data ensuring balanced seismic groups.
    5. Append extended datasets to training folds.
    6. Save train and validation CSVs per fold.

    Args:
        args (Namespace): Command-line arguments containing:
            - dataset (str): Path to root dataset folder containing train folders.
            - num_folds (int): Number of stratified folds (default 5).
            - seed (int): Random seed for reproducibility (default 42).
    """

    # Step 1: Construct paths to original and extended training datasets
    train_path = os.path.join(args.dataset, "train_data")
    train_ex_path_1 = os.path.join(args.dataset, "train_extended_1")
    train_ex_path_2 = os.path.join(args.dataset, "train_extended_2")

    # Load data into DataFrames for original and extended train datasets
    train_df = create_df(train_path)
    train_ex_df_1 = create_df(train_ex_path_1)
    train_ex_df_2 = create_df(train_ex_path_2)

    # Step 2: Cluster velocity models to identify seismic groups on original train data
    kmeans, train_df['seismic_group'] = cluster_velocity_models(
        list(train_df['image_path'])
    )

    # Step 3: Predict seismic groups for extended training datasets using cluster model
    train_ex_df_1['seismic_group'] = predict_cluster_velocity_models(
        kmeans, train_ex_df_1['image_path']
    )
    train_ex_df_2['seismic_group'] = predict_cluster_velocity_models(
        kmeans, train_ex_df_2['image_path']
    )

    # Step 4: Perform stratified K-Fold split on original train data to balance seismic groups
    skf = StratifiedKFold(
        n_splits=args.num_folds,
        shuffle=True,
        random_state=args.seed
    )
    for fold, (train_index, val_index) in enumerate(
        skf.split(train_df, y=train_df['seismic_group'])
    ):
        train_df.loc[val_index, 'fold'] = int(fold)

    # Step 5 & 6: For each fold:
    for FOLD in range(args.num_folds):
        # Select validation and training samples for the current fold
        val_l_df = train_df[train_df['fold'] == FOLD]
        train_l_df = train_df[train_df['fold'] != FOLD]

        # Append the extended datasets to the training set for better coverage
        train_l_df = pd.concat([train_l_df, train_ex_df_1, train_ex_df_2])

        # Save fold-specific train and validation CSV files
        train_l_df.to_csv(os.path.join(args.dataset, f"train_fold{FOLD}.csv"), index=False)
        val_l_df.to_csv(os.path.join(args.dataset, f"val_fold{FOLD}.csv"), index=False)

    print(f"[INFO] Saved {args.num_folds} stratified train and validation CSV folds at {args.dataset}")


if __name__ == "__main__":
    """
    Script entry point.

    Usage example:
        python prepare_folds.py --dataset /path/to/dataset --num_folds 5 --seed 42

    Arguments:
        --dataset: Path to dataset root folder containing training data subfolders.
        --num_folds: Number of stratified folds to create (default: 5).
        --seed: Random seed for reproducibility (default: 42).
    """
    parser = argparse.ArgumentParser(description="Create stratified folds CSVs for training.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for StratifiedKFold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
