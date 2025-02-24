from typing import Optional, Tuple
import albumentations as A

import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore", message=".*Rotate could work incorrectly.*")


class SeismicSequenceAugmentation:
    """
    A class to perform spatial augmentations on seismic frame sequences.

    Attributes:
        spatial_transform (A.ReplayCompose): A composed set of spatial transformations.
    """

    def __init__(self, p: float = 0.5):
        """
        Initializes the SeismicSequenceAugmentation with specified probability.

        Args:
            p (float): Probability of applying the augmentation. Default is 0.5.
        """
        self.spatial_transform = A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=(180, 180), p=0.5),
            ]
        )

    def __call__(
        self,
        seismic_frames: np.ndarray,
        label_frame: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies spatial augmentations to a sequence of seismic frames and an optional label frame.

        Args:
            seismic_frames (np.ndarray): A numpy array of seismic frames.
            label_frame (Optional[np.ndarray]): A numpy array of the label frame. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Augmented seismic frames and optionally the augmented label frame.
        """
        central_idx = len(seismic_frames) // 2

        # Process the central seismic frame
        central_frame = seismic_frames[central_idx].astype(np.float32)
        if central_frame.ndim == 2:
            central_frame = central_frame[..., np.newaxis]

        # Process the label frame if provided
        if label_frame is not None:
            label_frame = label_frame.astype(np.float32)
            if label_frame.ndim == 2:
                label_frame = label_frame[..., np.newaxis]

        # Apply spatial transformations to the central frame and label
        augmented = self.spatial_transform(image=central_frame, mask=label_frame)

        # Initialize list to hold augmented frames
        augmented_frames = []

        for i, frame in enumerate(seismic_frames):
            frame = frame.astype(np.float32)
            if frame.ndim == 2:
                frame = frame[..., np.newaxis]

            if i == central_idx:
                # Use the augmented central frame and label
                aug_frame = augmented['image'].squeeze()
                if label_frame is not None:
                    label_frame = augmented['mask'].squeeze()
            else:
                # Replay the same transformations on other frames
                aug_frame = self.spatial_transform.replay(
                    augmented['replay'], image=frame
                )['image'].squeeze()

            augmented_frames.append(aug_frame)

        # Stack all augmented frames into a single numpy array
        augmented_frames = np.stack(augmented_frames)

        # Convert numpy arrays to PyTorch tensors
        augmented_frames_tensor = torch.from_numpy(augmented_frames).float()
        label_frame_tensor = (
            torch.from_numpy(label_frame).float() if label_frame is not None else None
        )

        return augmented_frames_tensor, label_frame_tensor
