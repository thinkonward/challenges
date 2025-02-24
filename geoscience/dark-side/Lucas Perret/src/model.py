from typing import Tuple, Optional, Dict, Any
from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl

from .model_factory import create_model
from .image_processing import pad_to_size, unpad_image
from .loss import (
    compute_loss,
    compute_2d_dice,
    calculate_dice_3d_gpu
)


class SeismicFaultDetector(pl.LightningModule):
    """
    PyTorch Lightning Module for Seismic Fault Detection.

    Args:
        archi (str): Model architecture.
        val_axis (str): Axis to use for validation ('x', 'y', 'z', 'xy').
        nchans (int): Number of input channels.
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate for the optimizer.
        scheduler_gamma (float): Gamma value for the exponential LR scheduler.
        encoder_name (str, optional): Encoder backbone name. Defaults to None.
        encoder_weights (str, optional): Encoder backbone weights (for unet and unetpp). Defaults to imagenet.
        model_size (str, optional): Model size. Defaults to None.
        input_size (Tuple[int, int], optional): Input size as (height, width). Defaults to (320, 1280).
        dropout (float, optional): Dropout probability. Defaults to 0.5.
        _compile (Any, optional): Compilation mode for the model. Defaults to None.
    """

    def __init__(
        self,
        archi: str,
        val_axis: str,
        nchans: int,
        num_classes: int,
        learning_rate: float,
        scheduler_gamma: float,
        encoder_name: Optional[str] = None,
        encoder_weights: Optional[str] = 'imagenet',
        model_size: Optional[str] = None,
        input_size: Tuple[int, int] = (320, 1280),
        dropout: float = 0.5,
        _compile: Optional[Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.best_epoch: int = 0
        self.best_val_score: float = 0.0
        self.best_threshold: float = 0.5
        self.axis: str = val_axis
        self.learning_rate: float = learning_rate
        self.scheduler_gamma: float = scheduler_gamma
        self.input_size: Tuple[int, int] = input_size

        # Fixed dimensions for all axes
        self.volume_shape: Tuple[int, int, int] = (300, 300, 1259)

        # Axis-specific configurations
        self.expected_frames_x: Optional[int] = None
        self.expected_frames_y: Optional[int] = None
        self.expected_frames_z: Optional[int] = None

        if self.axis in ['x', 'xy']:
            self.expected_frames_x = self.volume_shape[0]  # 300
        if self.axis in ['y', 'xy']:
            self.expected_frames_y = self.volume_shape[1]  # 300
        if self.axis == 'z':
            self.expected_frames_z = self.volume_shape[2]  # 1259

        # Create model
        self.model = create_model(
            archi=archi,
            nchans=nchans,
            num_classes=num_classes,
            axis=val_axis,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            model_size=model_size,
            input_size=input_size,
            dropout=dropout,
        )

        # Compile model if requested
        if _compile is not None:
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(self.model, mode=_compile)

        self.validation_thresholds: np.ndarray = np.arange(0.1, 1.0, 0.025)
        self.volumes: Dict[str, Any] = {}
        self.threshold_scores: Dict[float, list] = defaultdict(list)
        self.threshold_scores_x: Optional[Dict[float, list]] = defaultdict(list) if self.axis == 'xy' else None
        self.threshold_scores_y: Optional[Dict[float, list]] = defaultdict(list) if self.axis == 'xy' else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step processing a single batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch containing seismic data and masks.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        seismic, mask, _, _, _ = batch

        # Pad according to architecture requirements
        seismic, mask, _ = pad_to_size(seismic, self.input_size, mask=mask)
        mask = mask.unsqueeze(1)

        # Forward pass
        pred = self(seismic)

        loss, _ = compute_loss(pred, mask)

        # Calculate and log 2D Dice
        with torch.no_grad():
            dice_2d = compute_2d_dice(pred, mask)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_dice_2d', dice_2d, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, list, list, list],
        batch_idx: int
    ) -> None:
        """
        Validation step processing a single batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, list, list, list]): Batch containing seismic data, masks,
                                                                          sample IDs, frame indices, and axes.
            batch_idx (int): Batch index.
        """
        seismic, mask, sample_ids, frame_indices, axes = batch

        # Pad according to architecture requirements
        seismic, mask, padding = pad_to_size(seismic, self.input_size, mask=mask)
        mask = mask.unsqueeze(1)

        # Forward pass
        pred = torch.sigmoid(self(seismic))

        # Remove padding
        pred = unpad_image(pred, padding)
        mask = unpad_image(mask, padding)

        # Process each sample in the batch
        for i in range(len(sample_ids)):
            sample_id = sample_ids[i]
            frame_idx = int(frame_indices[i])
            current_axis = axes[i]

            pred_frame = pred[i, 0]
            true_frame = mask[i, 0]

            # Initialize volume data if needed
            if sample_id not in self.volumes:
                if self.axis == 'xy':
                    self.volumes[sample_id] = {
                        'x': {
                            'pred_frames': torch.zeros(self.volume_shape, dtype=torch.float32, device=pred.device),
                            'true_frames': torch.zeros(self.volume_shape, dtype=torch.float32, device=true_frame.device),
                            'frame_count': 0
                        },
                        'y': {
                            'pred_frames': torch.zeros(self.volume_shape, dtype=torch.float32, device=pred.device),
                            'true_frames': torch.zeros(self.volume_shape, dtype=torch.float32, device=true_frame.device),
                            'frame_count': 0
                        }
                    }
                else:
                    self.volumes[sample_id] = {
                        'pred_frames': torch.zeros(self.volume_shape, dtype=torch.float32, device=pred.device),
                        'true_frames': torch.zeros(self.volume_shape, dtype=torch.float32, device=true_frame.device),
                        'frame_count': 0
                    }

            # Store frame according to axis
            if self.axis == 'xy':
                volume_data = self.volumes[sample_id][current_axis]
            else:
                volume_data = self.volumes[sample_id]

            if current_axis == 'x':
                volume_data['pred_frames'][frame_idx, :, :] = pred_frame
                volume_data['true_frames'][frame_idx, :, :] = true_frame
            elif current_axis == 'y':
                volume_data['pred_frames'][:, frame_idx, :] = pred_frame
                volume_data['true_frames'][:, frame_idx, :] = true_frame
            else:  # axis == 'z'
                volume_data['pred_frames'][:, :, frame_idx] = pred_frame
                volume_data['true_frames'][:, :, frame_idx] = true_frame

            volume_data['frame_count'] += 1

            # Process completed volumes
            if self.axis == 'xy':
                x_complete = self.volumes[sample_id]['x']['frame_count'] == self.expected_frames_x
                y_complete = self.volumes[sample_id]['y']['frame_count'] == self.expected_frames_y

                if x_complete and y_complete:
                    # Calculate scores for X axis
                    self._calculate_threshold_scores(
                        volume_data=self.volumes[sample_id]['x'],
                        thresh_scores=self.threshold_scores_x
                    )

                    # Calculate scores for Y axis
                    self._calculate_threshold_scores(
                        volume_data=self.volumes[sample_id]['y'],
                        thresh_scores=self.threshold_scores_y
                    )

                    # Calculate scores for averaged predictions
                    averaged_preds = (
                        self.volumes[sample_id]['x']['pred_frames'] +
                        self.volumes[sample_id]['y']['pred_frames']
                    ) / 2
                    self._calculate_threshold_scores(
                        volume_data={
                            'pred_frames': averaged_preds,
                            'true_frames': self.volumes[sample_id]['x']['true_frames']
                        },
                        thresh_scores=self.threshold_scores
                    )

                    # Cleanup
                    del self.volumes[sample_id]
                    torch.cuda.empty_cache()

            else:  # Single axis mode
                expected_frames = self._get_expected_frames()

                if volume_data['frame_count'] == expected_frames:
                    # Calculate scores for each threshold
                    self._calculate_threshold_scores(
                        volume_data=volume_data,
                        thresh_scores=self.threshold_scores
                    )

                    # Cleanup
                    del self.volumes[sample_id]
                    torch.cuda.empty_cache()

    def on_validation_epoch_start(self) -> None:
        """
        Initialize storage for validation epoch.
        """
        self.volumes: Dict[str, Any] = {}
        self.threshold_scores: Dict[float, list] = defaultdict(list)
        if self.axis == 'xy':
            self.threshold_scores_x: Dict[float, list] = defaultdict(list)
            self.threshold_scores_y: Dict[float, list] = defaultdict(list)
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        """
        Calculate and log validation metrics at the end of the validation epoch.
        """
        if not self.threshold_scores:
            self.log('val_dice_3d', 0.0, prog_bar=True)
            return

        # Calculate final score based on threshold_scores (either from averaged predictions in xy mode or single axis)
        threshold_means = {thresh: np.mean(scores) for thresh, scores in self.threshold_scores.items()}
        best_threshold, best_score = max(threshold_means.items(), key=lambda x: x[1])

        # Update best score if needed
        if best_score > self.best_val_score:
            self.best_epoch = self.current_epoch
            self.best_val_score = best_score
            self.best_threshold = best_threshold

        # Log main validation metric (used for early stopping)
        self.log('val_dice_3d', best_score, prog_bar=True, sync_dist=True)

        # If using xy mode, calculate and log individual axis scores
        if self.axis == 'xy':
            # Calculate X axis scores
            threshold_means_x = {thresh: np.mean(scores) for thresh, scores in self.threshold_scores_x.items()}
            best_threshold_x, best_score_x = max(threshold_means_x.items(), key=lambda x: x[1])
            self.log('val_dice_3d_x', best_score_x, prog_bar=True, sync_dist=True)

            # Calculate Y axis scores
            threshold_means_y = {thresh: np.mean(scores) for thresh, scores in self.threshold_scores_y.items()}
            best_threshold_y, best_score_y = max(threshold_means_y.items(), key=lambda x: x[1])
            self.log('val_dice_3d_y', best_score_y, prog_bar=True, sync_dist=True)

            print(f"\nBest X threshold: {best_threshold_x:.3f} (score: {best_score_x:.3f})")
            print(f"Best Y threshold: {best_threshold_y:.3f} (score: {best_score_y:.3f})")

        print(f"Best validation threshold: {best_threshold:.3f} (score: {best_score:.3f})")
        print(f"Best overall threshold: {self.best_threshold:.3f} (score: {self.best_val_score:.3f})")

        # Clear storage
        self.volumes.clear()
        self.threshold_scores.clear()
        if self.axis == 'xy':
            self.threshold_scores_x.clear()
            self.threshold_scores_y.clear()
        torch.cuda.empty_cache()

    def configure_optimizers(self) -> Tuple[list, list]:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Tuple[list, list]: Optimizer and scheduler configurations.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.scheduler_gamma
            ),
            "interval": "epoch",
            "frequency": 1
        }
        return [optimizer], [scheduler]

    def _get_expected_frames(self) -> int:
        """
        Retrieve the expected number of frames based on the axis.

        Returns:
            int: Expected number of frames.
        """
        if self.axis == 'x':
            return self.expected_frames_x
        elif self.axis == 'y':
            return self.expected_frames_y
        elif self.axis == 'z':
            return self.expected_frames_z
        else:
            raise ValueError(f"Unknown axis: {self.axis}")

    def _calculate_threshold_scores(
        self,
        volume_data: Dict[str, torch.Tensor],
        thresh_scores: Dict[float, list]
    ) -> None:
        """
        Calculate Dice scores for each threshold and update threshold scores.

        Args:
            volume_data (Dict[str, torch.Tensor]): Dictionary containing predicted and true frames.
            thresh_scores (Dict[float, list]): Dictionary to store Dice scores for each threshold.
        """
        for thresh in self.validation_thresholds:
            thresh_tensor = torch.tensor(thresh, device=volume_data['pred_frames'].device)
            thresholded_pred = (volume_data['pred_frames'] > thresh_tensor).float()
            dice = calculate_dice_3d_gpu(thresholded_pred, volume_data['true_frames'])
            thresh_scores[thresh].append(dice.item())
