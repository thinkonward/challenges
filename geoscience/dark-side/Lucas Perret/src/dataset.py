from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .aug import SeismicSequenceAugmentation


class SeismicDataset(Dataset):
    """
    PyTorch Dataset for Seismic Fault Detection.

    Args:
        root_dir (Union[str, Path]): Root directory containing 2d slices data.
        df (pd.DataFrame): DataFrame containing metadata for the dataset.
        mode (str, optional): Mode of the dataset ('train', 'valid', 'inference'). Defaults to 'valid'.
        apply_augmentation (bool, optional): Whether to apply data augmentation. Defaults to False.
        nchans (int, optional): Number of input channels. Must be odd. Defaults to 1.

    Raises:
        ValueError: If `nchans` is not odd.
        ValueError: If `mode` is not one of 'train', 'valid', or 'inference'.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        df: pd.DataFrame,
        mode: str = 'valid',
        apply_augmentation: bool = False,
        nchans: int = 1
    ):
        if nchans % 2 != 1:
            raise ValueError(f"nchans must be odd, got {nchans}")

        if mode not in ['train', 'valid', 'inference']:
            raise ValueError(f"Mode must be 'train', 'valid' or 'inference', got {mode}")

        self.root_dir: Path = Path(root_dir)
        self.nchans: int = nchans
        self.radius: int = nchans // 2
        self.mode: str = mode
        self.apply_augmentation: bool = apply_augmentation and mode == 'train'
        self.df: pd.DataFrame = df

        # Initialize augmentation if needed
        self.augmenter: Optional[SeismicSequenceAugmentation] = SeismicSequenceAugmentation(p=0.5) if self.apply_augmentation else None

        # Print final dataset summary
        print("\nFinal dataset composition:")
        for part in sorted(self.df['data_part'].unique()):
            part_df = self.df[self.df['data_part'] == part]
            n_volumes = part_df['sample_id'].nunique()
            n_frames = len(part_df)
            print(f"- {part}: {n_frames} frames from {n_volumes} volumes")

    def _get_neighbor_indices(self, frame_idx: int, n_frames: int) -> List[int]:
        """
        Determine indices of neighboring frames to use.

        Args:
            frame_idx (int): Current frame index.
            n_frames (int): Total number of frames in the volume.

        Returns:
            List[int]: List of neighboring frame indices.
        """
        neighbor_indices = []
        for offset in range(-self.radius, self.radius + 1):
            neighbor_idx = frame_idx + offset
            if 0 <= neighbor_idx < n_frames:
                neighbor_indices.append(neighbor_idx)
            else:
                neighbor_indices.append(0 if neighbor_idx < 0 else n_frames - 1)
        return neighbor_indices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], str, int, str]:
        """
        Get a single frame and its neighbors.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], str, int, str]:
                - Seismic frames tensor.
                - Label frame tensor or None.
                - Sample ID.
                - Frame index.
                - Axis.
        """
        frame_info = self.df.iloc[idx]

        sample_id: str = frame_info['sample_id']
        axis: str = frame_info['axis']

        # Get volume info for determining neighbors
        volume_frames = self.df[
            (self.df['sample_id'] == sample_id) &
            (self.df['axis'] == axis)
        ].sort_values('frame_idx').copy()

        n_frames: int = len(volume_frames)
        frame_position: int = frame_info['frame_idx']

        # Get indices of frames to load
        neighbor_indices: List[int] = self._get_neighbor_indices(frame_position, n_frames)

        # Load seismic frames
        seismic_frames: List[np.ndarray] = []
        for n_idx in neighbor_indices:
            try:
                nframe = volume_frames[volume_frames['frame_idx'] == n_idx].iloc[0]
                frame = np.load(self.root_dir / nframe['frame_path'])
                seismic_frames.append(frame)
            except IndexError:
                raise IndexError(f"Frame index {n_idx} not found for sample_id {sample_id} and axis {axis}.")
            except FileNotFoundError:
                raise FileNotFoundError(f"Frame file {nframe['frame_path']} not found.")

        seismic_frames = np.stack(seismic_frames)

        # Load label if available and required
        if self.mode != 'inference' and frame_info['has_labels']:
            try:
                label_frame = np.load(self.root_dir / frame_info['label_path'])
                if frame_info['label_type'] == 'fault':
                    label_frame = (label_frame > 0).astype(np.float32)
            except FileNotFoundError:
                raise FileNotFoundError(f"Label file {frame_info['label_path']} not found.")
        else:
            label_frame = None

        if self.apply_augmentation and label_frame is not None and self.augmenter is not None:
            # Ensure that the augmenter returns numpy arrays
            seismic_frames, label_frame = self.augmenter(seismic_frames, label_frame)
            if isinstance(seismic_frames, np.ndarray):
                seismic_frames = torch.from_numpy(seismic_frames).float()
            elif isinstance(seismic_frames, torch.Tensor):
                seismic_frames = seismic_frames.float()
            else:
                raise TypeError(f"Augmenter returned unsupported type for seismic_frames: {type(seismic_frames)}")

            if label_frame is not None:
                if isinstance(label_frame, np.ndarray):
                    label_frame = torch.from_numpy(label_frame).float()
                elif isinstance(label_frame, torch.Tensor):
                    label_frame = label_frame.float()
                else:
                    raise TypeError(f"Augmenter returned unsupported type for label_frame: {type(label_frame)}")
        else:
            seismic_frames = torch.from_numpy(seismic_frames).float()
            if label_frame is not None:
                label_frame = torch.from_numpy(label_frame).float()

        return seismic_frames, label_frame, sample_id, frame_info['frame_idx'], axis

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.df)


def seismic_collate_fn(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], str, int, str]]) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str], List[int], List[str]]:
    """
    Custom collate function for SeismicDataset.

    Args:
        batch (List[Tuple[torch.Tensor, Optional[torch.Tensor], str, int, str]]): List of data samples.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], List[str], List[int], List[str]]:
            - Batched seismic frames tensor.
            - Batched mask frames tensor or None.
            - List of sample IDs.
            - List of frame indices.
            - List of axes.
    """
    seismic_frames, mask_frames, sample_ids, frame_indices, axes = zip(*batch)
    seismic_batch = default_collate(seismic_frames)
    sample_ids_batch = list(sample_ids)
    frame_indices_batch = list(frame_indices)
    axes_batch = list(axes)

    if all(mask is None for mask in mask_frames):
        mask_batch = None
    else:
        # Filter out None masks before collating
        masks = [mask for mask in mask_frames if mask is not None]
        mask_batch = default_collate(masks)

    return seismic_batch, mask_batch, sample_ids_batch, frame_indices_batch, axes_batch
