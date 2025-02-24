from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .utils import save_config


class BestMetricsCallback(Callback):
    """
    Callback to track and save the best validation metrics during training.

    Attributes:
        fold_idx (int): Index of the current fold.
        fold_info (Dict[int, Dict[str, Any]]): Information about each fold.
        train_parts (Any): Training parts information.
        val_parts (Any): Validation parts information.
        dirpath (str): Directory path to save configurations.
        args (Any): Arguments or configurations.
        train_volumes (Optional[Any]): Training volumes information.
        val_volumes (Optional[Any]): Validation volumes information.
        best_val_score (float): Best validation score observed.
    """

    def __init__(
        self,
        fold_idx: int,
        fold_info: Dict[int, Dict[str, Any]],
        train_parts: Any,
        val_parts: Any,
        dirpath: str,
        args: Any,
        train_volumes: Optional[Any] = None,
        val_volumes: Optional[Any] = None,
    ) -> None:
        """
        Initializes the BestMetricsCallback.

        Args:
            fold_idx (int): Index of the current fold.
            fold_info (Dict[int, Dict[str, Any]]): Information about each fold.
            train_parts (Any): Training parts information.
            val_parts (Any): Validation parts information.
            dirpath (str): Directory path to save configurations.
            args (Any): Arguments or configurations.
            train_volumes (Optional[Any], optional): Training volumes information. Defaults to None.
            val_volumes (Optional[Any], optional): Validation volumes information. Defaults to None.
        """
        super().__init__()
        self.fold_idx: int = fold_idx
        self.fold_info: Dict[int, Dict[str, Any]] = fold_info
        self.train_parts: Any = train_parts
        self.val_parts: Any = val_parts
        self.train_volumes: Optional[Any] = train_volumes
        self.val_volumes: Optional[Any] = val_volumes
        self.dirpath: str = dirpath
        self.args: Any = args
        self.best_val_score: float = float('-inf')

        self._update_fold_info()
        save_config(self.args, self.dirpath, self.fold_info)

    def _update_fold_info(self) -> None:
        """
        Updates the fold information with initial values if the fold is not already present.
        """
        if self.fold_idx not in self.fold_info:
            self.fold_info[self.fold_idx] = {}

        current_info: Dict[str, Union[Any, str]] = {
            'train_parts': self.train_parts,
            'val_parts': self.val_parts,
            'train_volumes': self.train_volumes,
            'val_volumes': self.val_volumes,
            'best_epoch': 'N/A',
            'best_val_score': 'N/A',
            'best_threshold': 'N/A',
        }

        self.fold_info[self.fold_idx].update(current_info)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called when the validation loop ends. Checks and updates the best validation score.

        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The model being trained.
        """
        current_score: Optional[float] = trainer.callback_metrics.get('val_dice_3d')

        if current_score is not None and current_score > self.best_val_score:
            self.best_val_score = current_score

            best_threshold = getattr(pl_module, 'best_threshold', 'N/A')

            self.fold_info[self.fold_idx].update(
                {
                    'best_epoch': trainer.current_epoch,
                    'best_val_score': float(current_score),
                    'best_threshold': best_threshold,
                }
            )

            save_config(self.args, self.dirpath, self.fold_info)
