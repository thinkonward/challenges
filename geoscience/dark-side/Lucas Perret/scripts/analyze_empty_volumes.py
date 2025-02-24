#!/usr/bin/env python3
"""
Module to analyze empty volumes and volumes without labels in seismic datasets.
"""

from typing import Dict, List
import argparse
import pandas as pd


def analyze_empty_volumes(df: pd.DataFrame, verbose: bool = False) -> Dict[str, List[str]]:
    """
    Analyze volumes in dataset to identify those without labels and those that are empty.

    Args:
        df: Pandas DataFrame containing the dataset information with columns:
            - axis: Axis identifier
            - sample_id: Volume identifier
            - positive_pixels: Number of positive pixels
            - has_labels: Boolean indicating presence of labels
            - data_part: Dataset partition identifier
            - format: Volume format
            - volume_x, volume_y, volume_z: Volume dimensions
        verbose: If True, prints detailed analysis information

    Returns:
        Dict containing lists of volume IDs:
            - 'no_labels_vol': List of volume IDs without labels
            - 'empty_vol': List of volume IDs that are empty but have labels
    """
    # Filter to single axis to avoid counting same volume multiple times
    df_single_axis = df[df['axis'] == 'z']

    # Aggregate volume information
    volume_pixels = df_single_axis.groupby('sample_id').agg({
        'positive_pixels': 'sum',
        'has_labels': 'first',
        'data_part': 'first',
        'format': 'first',
        'volume_x': 'first',
        'volume_y': 'first',
        'volume_z': 'first'
    })

    # Initialize return dictionary
    result_dict: Dict[str, List[str]] = {
        'no_labels_vol': [],
        'empty_vol': []
    }

    # Analyze volumes without labels
    no_label_volumes = volume_pixels[~volume_pixels['has_labels']]

    if verbose:
        print("\nAnalyzing volumes in dataset...")
        print("\n=== Volumes Without Labels Analysis ===")
        print(f"\nTotal volumes without labels: {len(no_label_volumes)}")

    if not no_label_volumes.empty:
        result_dict['no_labels_vol'] = list(no_label_volumes.index)

        if verbose:
            print("\nDetails by data part:")
            print("-" * 80)

            # Group and display information for volumes without labels
            for data_part, group in no_label_volumes.groupby('data_part'):
                print(f"\nData Part: {data_part}")
                print(f"Number of volumes without labels in this part: {len(group)}")
                print("\nVolumes without labels:")

                for idx, row in group.iterrows():
                    print(f"  - Sample ID: {idx}")
                    print(f"    Format: {row['format']}")
                    print(
                        f"    Volume dimensions: "
                        f"{row['volume_x']}x{row['volume_y']}x{row['volume_z']}"
                    )
                    print()

    # Analyze empty volumes (volumes with labels but no positive pixels)
    volumes_with_labels = volume_pixels[volume_pixels['has_labels']]
    empty_volumes = volumes_with_labels[volumes_with_labels['positive_pixels'] == 0]

    if verbose:
        print("\n=== Empty Volumes Analysis ===")
        print(f"\nTotal empty volumes (with labels): {len(empty_volumes)}")

    if not empty_volumes.empty:
        result_dict['empty_vol'] = list(empty_volumes.index)

        if verbose:
            print("\nDetails by data part:")
            print("-" * 80)

            # Group and display information for empty volumes
            for data_part, group in empty_volumes.groupby('data_part'):
                print(f"\nData Part: {data_part}")
                print(f"Number of empty volumes in this part: {len(group)}")
                print("\nEmpty volumes:")

                for idx, row in group.iterrows():
                    print(f"  - Sample ID: {idx}")
                    print(f"    Format: {row['format']}")
                    print(
                        f"    Volume dimensions: "
                        f"{row['volume_x']}x{row['volume_y']}x{row['volume_z']}"
                    )
                    print()

            # Display copy-pasteable list of empty volume IDs
            print("\nCopy-pasteable list of empty volume IDs:")
            print(f"empty_volumes = {repr(list(empty_volumes.index))}")

    # Print summary counts regardless of empty status when verbose
    if verbose:
        print("\nSummary Counts:")
        print(f"Total volumes without labels: {len(result_dict['no_labels_vol'])}")
        print(f"Total empty volumes: {len(result_dict['empty_vol'])}")

    return result_dict


def main() -> None:
    """
    Main function to parse arguments and execute the analysis.
    """
    parser = argparse.ArgumentParser(
        description='Analyze empty volumes in seismic dataset'
    )
    parser.add_argument(
        'df_path',
        type=str,
        help='Path to the parquet index file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Enable detailed output printing'
    )
    args = parser.parse_args()

    # Load the DataFrame
    df = pd.read_parquet(args.df_path)

    # Execute analysis and get results
    results = analyze_empty_volumes(df, verbose=args.verbose)

    if args.verbose:
        print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
