import os
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch

from src.predict_utils import (
    build_final_volumes_dir_name,
    predict_single_checkpoint,
    ensemble_volumes_and_save,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse and organize command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate fault predictions for test data'
    )

    # Data Input Arguments
    data_group = parser.add_argument_group('Data Inputs')
    data_group.add_argument("root_dir", type=str, help="Root directory containing 2D slices data")
    data_group.add_argument(
        '--vol_filter',
        type=str,
        nargs='*',
        default=None,
        help='List of sample_ids to filter the dataset'
    )

    # Checkpoint and Axis Arguments
    checkpoint_group = parser.add_argument_group('Model Checkpoints')
    checkpoint_group.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        required=True,
        help='Paths to model checkpoint files (.ckpt)'
    )
    checkpoint_group.add_argument(
        '--axes',
        type=str,
        nargs='+',
        required=True,
        help='Axes to process for each checkpoint (e.g. x/y/z/xy)'
    )

    # Processing Configuration Arguments
    config_group = parser.add_argument_group('Processing Configuration')
    config_group.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for processing'
    )
    config_group.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for DataLoader'
    )
    config_group.add_argument(
        '--save_threshold',
        type=float,
        default=0.5,
        help='Threshold to apply to averaged predictions (default: 0.5)'
    )
    config_group.add_argument(
        '--min_mean_conf',
        type=float,
        default=None,
        help=(
            'Minimum mean confidence required. If the mean confidence is below this, '
            'the predicted volume is set to zero (default: None)'
        )
    )

    # Output / Behavior Arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--submission_path',
        type=str,
        default='submission.npz',
        help='Path to the final submission file (default: submission.npz)'
    )
    output_group.add_argument(
        '--save_probas',
        action='store_true',
        default=False,
        help='Save raw probability volumes into predictions_probas/{model_name}/{axis}'
    )
    output_group.add_argument(
        '--save_final',
        action='store_true',
        default=False,
        help='Save final thresholded volumes into a combined predictions folder'
    )

    # Performance Optimization Arguments
    optimization_group = parser.add_argument_group('Performance Optimization')
    optimization_group.add_argument(
        '--compile',
        action='store_true',
        default=False,
        help='Compile the model for optimized performance'
    )
    optimization_group.add_argument(
        '--cpu',
        action='store_true',
        help='Force inference on CPU'
    )
    optimization_group.add_argument(
        '--dtype',
        type=str,
        choices=['float16', 'float32', 'bf16'],
        default='bf16',
        help='Data type for inference (default: bf16)'
    )

    # Additional Behavior Arguments
    behavior_group = parser.add_argument_group('Additional Behavior')
    behavior_group.add_argument(
        '--force_prediction',
        action='store_true',
        default=False,
        help='Re-predict even if volumes are already predicted'
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to generate fault predictions for test data.
    """
    args = parse_arguments()

    # Setup logger - removed
    # logger = setup_logging(verbose=args.verbose)

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Basic validation
    if len(args.checkpoints) != len(args.axes):
        raise ValueError(
            "Number of checkpoints must match number of axes specifications."
        )

    # Count total axes across all specs
    total_axes = 0
    for axis_spec in args.axes:
        if axis_spec.lower() == 'xy':
            total_axes += 2
        else:
            total_axes += 1

    # If there's only one model and one axis, but min_mean_conf is set, warn user
    if len(args.checkpoints) == 1 and total_axes == 1 and args.min_mean_conf is not None:
        print(
            f"Warning: You set a min_mean_conf ({args.min_mean_conf}) but there's only one model and one axis. "
            "Confidence-based filtering will be ignored."
        )

    # Load dataset index
    print("Loading dataset index...")
    df_path = os.path.join(args.root_dir, 'dataset.parquet')
    full_df = pd.read_parquet(df_path)
    print(f"Total samples in dataset: {len(full_df)}")

    # Filter volumes if specified
    if args.vol_filter:
        print(
            f"Applying volume filter with {len(args.vol_filter)} sample_id(s)."
        )
        initial_count = len(full_df)
        filtered_df = full_df[full_df['sample_id'].isin(args.vol_filter)].copy()
        final_count = len(filtered_df)
        missing_samples = set(args.vol_filter) - set(filtered_df['sample_id'].unique())

        print(
            f"Number of samples after filtering: {final_count} "
            f"(filtered out {initial_count - final_count} samples)."
        )
        if missing_samples:
            print(
                "Warning: The following sample_id(s) were not found in the dataset and will be ignored: "
                f"{', '.join(missing_samples)}"
            )
        full_df = filtered_df
    else:
        print("No volume filter applied. Processing all samples.")

    print(f"Total samples to process: {len(full_df)}")

    # Predict for each checkpoint & axis specification
    for ckpt_path_str, axis_spec in zip(args.checkpoints, args.axes):
        checkpoint_path = Path(ckpt_path_str)
        # For 'xy', we do x, then y
        if axis_spec.lower() == 'xy':
            axis_list = ['x', 'y']
        else:
            axis_list = [axis_spec.lower()]

        for ax in axis_list:
            print(f"Checkpoint: {checkpoint_path.name}, Axis: {ax}")
            predict_single_checkpoint(
                checkpoint_path=checkpoint_path,
                axis=ax,
                full_df=full_df,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                save_probas=args.save_probas,
                force_prediction=args.force_prediction,
                root_dir=args.root_dir,
                dtype=args.dtype,
                compile_model=args.compile,
                cpu=args.cpu,
                device=device
            )

    print("All checkpoints processed.")

    # Collect the prediction directories for ensembling
    prediction_dirs: List[Path] = []
    for ckpt_path_str, axis_spec in zip(args.checkpoints, args.axes):
        checkpoint_path = Path(ckpt_path_str)
        model_name = checkpoint_path.stem

        if axis_spec.lower() == 'xy':
            sub_axes = ['x', 'y']
        else:
            sub_axes = [axis_spec.lower()]

        for sub_ax in sub_axes:
            pred_dir = checkpoint_path.parent / 'predictions_probas' / model_name / sub_ax
            print(f"Looking for predictions in {pred_dir}")
            if pred_dir.exists():
                prediction_dirs.append(pred_dir)

    print(
        f"Found {len(prediction_dirs)} relevant prediction directories for ensembling."
    )

    # Optionally build final volumes directory name
    final_volumes_dir = None
    if args.save_final:
        final_dir_name = build_final_volumes_dir_name(args.checkpoints, args.axes)
        final_volumes_dir = Path(final_dir_name)
        print(f"Final volumes dir will be '{final_dir_name}'")

    # Ensemble volumes & create submission
    ensemble_volumes_and_save(
        all_predictions=prediction_dirs,
        dataset_index=full_df,
        output_path=Path(args.submission_path),
        save_threshold=args.save_threshold,
        device=device,
        min_mean_conf=args.min_mean_conf,
        save_final_volumes=args.save_final,
        final_volumes_dir=final_volumes_dir
    )

    print("Ensembling and submission creation completed successfully.")
    print(f"Submission file: {args.submission_path}")


if __name__ == '__main__':
    main()
