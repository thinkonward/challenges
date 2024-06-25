import numpy as np
from typing import Literal
from skimage.metrics import structural_similarity as ssim
import os
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision.transforms import v2


def scoring(prediction_path, ground_truth_path):
    """Scoring function. Use scikit-image implementation of Structural Similarity Index:
       https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

    Parameters:
        prediction_path: path of perdiction .npz file
        ground_truth_path: path of ground truth .npz file

    Returns:
        score: -1 to 1 structural similarity index
    """

    ground_truth = np.load(ground_truth_path)
    prediction = np.load(prediction_path)

    ssim_scores = sorted(
        [
            ssim(ground_truth[key], prediction[key], data_range=255)
            for key in ground_truth.files
        ]
    )

    trimmed_scores = ssim_scores[1:-1]

    score = np.mean(trimmed_scores)

    return score


def create_submission(seismic_filenames: list, prediction: list, submission_path: str):
    """Function to create submission file out of all test predictions in one list

    Parameters:
        seismic_filenames: list of survey .npy filenames used for perdiction
        prediction: list with 3D np.ndarrays of predicted missing parts
        submission_path: path to save submission

    Returns:
        None
    """

    submission = dict({})
    for sample_name, sample_prediction in zip(seismic_filenames, prediction):
        vol_id = sample_name.split(".")[0]
        i_slices_index = (
            np.array([0.25, 0.5, 0.75]) * sample_prediction.shape[0]
        ).astype(int)
        i_slices_names = [f"{vol_id}_gt.npy-i_{n}" for n in range(0, 3)]
        i_slices = [sample_prediction[s, :, :].astype(np.uint8) for s in i_slices_index]
        submission.update(dict(zip(i_slices_names, i_slices)))

        x_slices_index = (
            np.array([0.25, 0.5, 0.75]) * sample_prediction.shape[1]
        ).astype(int)
        x_slices_names = [f"{vol_id}_gt.npy-x_{n}" for n in range(0, 3)]
        x_slices = [sample_prediction[:, s, :].astype(np.uint8) for s in x_slices_index]
        submission.update(dict(zip(x_slices_names, x_slices)))

    np.savez(submission_path, **submission)


def create_single_submission(
    seismic_filename: str, prediction: np.ndarray, submission_path: str
):
    """Function to create submission file out of one test prediction at time

    Parameters:
        seismic_filename: filename of survey .npy used for perdiction
        prediction: 3D np.ndarray of predicted missing part
        submission_path: path to save submission

    Returns:
        None
    """

    try:
        submission = dict(np.load(submission_path))
    except:
        submission = dict({})

    i_slices_index = (np.array([0.25, 0.5, 0.75]) * prediction.shape[0]).astype(int)
    i_slices_names = [f"{seismic_filename}-i_{n}" for n in range(0, 3)]
    i_slices = [prediction[s, :, :].astype(np.uint8) for s in i_slices_index]
    submission.update(dict(zip(i_slices_names, i_slices)))

    x_slices_index = (np.array([0.25, 0.5, 0.75]) * prediction.shape[1]).astype(int)
    x_slices_names = [f"{seismic_filename}-x_{n}" for n in range(0, 3)]
    x_slices = [prediction[:, s, :].astype(np.uint8) for s in x_slices_index]
    submission.update(dict(zip(x_slices_names, x_slices)))

    np.savez(submission_path, **submission)


def rescale_volume(seismic, low=0, high=100):
    """
    Rescaling 3D seismic volumes 0-255 range, clipping values between low and high percentiles
    """
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255

    return seismic


class SliceServer:
    def __init__(
        self,
        root_dir,
        num_slices,
        cheesecake_factory_mode,
        limit,
        data_prefix,
        label_prefix,
        pretraining,
    ):
        self.root_dir = root_dir
        self.num_slices = num_slices
        self.cheesecake_factory_mode = cheesecake_factory_mode
        self.limit = limit
        self.data_prefix = data_prefix
        self.label_prefix = label_prefix
        self.pretraining = pretraining
        self.exclude_dirs = {".ipynb_checkpoints"}  # Set of directories to exclude

        # Gather volume IDs
        self.volume_ids = [
            d
            for path, dirs, files in os.walk(self.root_dir)
            for d in dirs
            if d not in self.exclude_dirs
        ]

        # Or alternatively:
        # self.volume_ids = [d for path, dirs, files in os.walk(self.root_dir) for d in dirs if d.isdigit()]

        # Gather all file names and filter out irrelevant ones
        self.volume_filenames = [
            file for path, dirs, files in os.walk(self.root_dir) for file in files
        ]

        # Filter data and label filenames and their corresponding volume IDs
        self.data_filenames = sorted(
            [
                file
                for file in self.volume_filenames
                if file.startswith(self.data_prefix)
            ]
        )
        self.label_filenames = sorted(
            [
                file
                for file in self.volume_filenames
                if file.startswith(self.label_prefix)
            ]
        )

        self.data_file_paths = []
        self.label_file_paths = []

        for volume_id in self.volume_ids:
            data_files = [
                data_file for data_file in self.data_filenames if volume_id in data_file
            ]
            label_files = [
                label_file
                for label_file in self.label_filenames
                if volume_id in label_file
            ]

            if data_files and label_files:
                data_file = data_files[0]
                label_file = label_files[0]
                self.data_file_paths.append(f"{self.root_dir}/{volume_id}/{data_file}")
                self.label_file_paths.append(
                    f"{self.root_dir}/{volume_id}/{label_file}"
                )

        # Create the dataframe for slices
        self.dataframe = self.create_slice_menu()

    def create_slice_menu(self):
        # Create a dataframe with matched data and label paths
        if not self.pretraining:
            menu = pd.DataFrame(
                {"data": self.data_file_paths, "label": self.label_file_paths}
            )
        else:
            menu = pd.DataFrame({"data": self.data_file_paths})

        # Add axis info for inline and crossline
        menu["axis"] = [["i", "x"] for _ in range(len(self.data_file_paths))]

        # Add all possible slices in the volume
        menu["idx"] = [
            [str(s) for s in list(range(self.num_slices))]
            for _ in range(len(self.data_file_paths))
        ]

        # Explode the dataframe to create dataset for each slice
        menu = menu.explode("axis")
        menu = menu.explode("idx")
        menu = menu.reset_index(drop=True)

        # Limit the menu if not in cheesecake factory mode
        if not self.cheesecake_factory_mode:
            menu = menu[: self.limit]

        return menu

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, column):
        return self.dataframe[column]


class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_slices=300,
        cheesecake_factory_mode=True,
        limit=None,
        data_prefix="",
        label_prefix="",
        pretraining=False,
    ):
        self.root_dir = root_dir
        self.num_slices = num_slices
        self.cheesecake_factory_mode = cheesecake_factory_mode
        self.limit = limit
        self.data_prefix = data_prefix
        self.label_prefix = label_prefix
        self.pretraining = pretraining
        self.slice_menu = SliceServer(
            self.root_dir,
            self.num_slices,
            self.cheesecake_factory_mode,
            self.limit,
            self.data_prefix,
            self.label_prefix,
            self.pretraining,
        )

    def __len__(self):
        return len(self.slice_menu)

    def __getitem__(self, idx):
        if not self.pretraining:
            if self.slice_menu["axis"][idx] == "i":
                data = np.load(
                    self.slice_menu["data"][idx], allow_pickle=True, mmap_mode="r+"
                )[int(self.slice_menu["idx"][idx]), ...]
                label = np.load(
                    self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+"
                )[int(self.slice_menu["idx"][idx]), ...]
            else:
                data = np.load(
                    self.slice_menu["data"][idx], allow_pickle=True, mmap_mode="r+"
                )[:, int(self.slice_menu["idx"][idx]), :]
                label = np.load(
                    self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+"
                )[:, int(self.slice_menu["idx"][idx]), :]

            data = data[np.newaxis, :, :]
            label = label[np.newaxis, :, :]

            data = torch.from_numpy(data).long()
            label = torch.from_numpy(label).long()

            return data, label

        else:
            if self.slice_menu["axis"][idx] == "i":
                data = np.load(
                    self.slice_menu["data"][idx], allow_pickle=True, mmap_mode="r+"
                )[int(self.slice_menu["idx"][idx]), ...]
            else:
                data = np.load(
                    self.slice_menu["data"][idx], allow_pickle=True, mmap_mode="r+"
                )[:, int(self.slice_menu["idx"][idx]), :]

            data = data[np.newaxis, :, :]

            data = torch.from_numpy(data).long()

            return data
