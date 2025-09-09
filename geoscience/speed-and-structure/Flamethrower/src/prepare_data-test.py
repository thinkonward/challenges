import os
import argparse
import pandas as pd

from utils import (
    create_ids, 
    create_df,
)

from glob import glob

def main(args):
    """
    Main function for preparing the test dataset CSV file.

    Steps:
        1. Construct the path to the 'test_data' directory inside the dataset root.
        2. Generate a DataFrame containing metadata or file paths for the test set.
        3. Save the generated DataFrame as 'test.csv' inside the dataset root.

    Args:
        args (Namespace): Command-line arguments containing:
            - dataset (str): Path to the root folder containing 'test_data'.
    """

    # Step 1: Build absolute path to the folder containing test images/data
    test_path = os.path.join(args.dataset, "test_data")
    test_path = '../../../../challenge_speed_and_structure/speed-and-structure-holdout-dataset-private'
    
    print(len(glob(test_path+'/*')))
    
    

    # Step 2: Create DataFrame from test folder
    # `create_df()` scans the given directory, collects relevant file info (e.g., file names, IDs),
    # and returns a DataFrame ready for the dataloader to use.
    test_df = create_df(test_path)

    # Step 3: Save the DataFrame as 'test.csv' for inference pipelines
    test_csv_path = os.path.join(args.dataset, "test.csv")
    test_csv_path = "../data/test.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"[INFO] Test CSV saved at: {test_csv_path}")


if __name__ == "__main__":
    """
    Script entry point.

    Command-line usage example:
        python prepare_data_test.py --dataset /path/to/dataset

    Arguments:
        --dataset: Path to the dataset root containing 'test_data' folder.
    """
    parser = argparse.ArgumentParser(description="Prepare test data CSV from dataset folder.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset root containing 'test_data' folder"
    )

    args = parser.parse_args()
    main(args)
