import json
from typing import Any, Dict, List, Optional
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import queue
import threading

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import SeismicDataset, seismic_collate_fn
from .image_processing import pad_to_size, unpad_image
from .model import SeismicFaultDetector


def load_checkpoint_config(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load the configuration dictionary for a given checkpoint.

    Args:
        checkpoint_path (Path): Path to the model checkpoint.

    Returns:
        Dict[str, Any]: Configuration dictionary loaded from the associated config.json.

    Raises:
        FileNotFoundError: If the config.json file does not exist.
    """
    config_path = checkpoint_path.parent / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config file found for checkpoint: {checkpoint_path}"
        )

    with config_path.open('r') as f:
        config = json.load(f)

    return config


def load_model(
    checkpoint_path: Path,
    config: Dict[str, Any],
    device: str
) -> SeismicFaultDetector:
    """
    Load the SeismicFaultDetector model from a checkpoint.

    Supports both Lightning checkpoints and model-only checkpoints.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.
        config (Dict[str, Any]): Model configuration dictionary.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        SeismicFaultDetector: Loaded and configured model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SeismicFaultDetector(
        archi=config['archi'],
        val_axis=config['val_axis'],
        nchans=config['nchans'],
        num_classes=config['num_classes'],
        learning_rate=config['lr'],
        scheduler_gamma=config['scheduler_gamma'],
        encoder_name=config['encoder_name'],
        encoder_weights=None,
        model_size=config['model_size'],
        input_size=config.get('input_size', (320, 1280)),
        dropout=0.0,
    )

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Lightning checkpoint
        state_dict = checkpoint['state_dict']
        new_state_dict = {
            k.replace('model._orig_mod.', 'model.'): v
            for k, v in state_dict.items()
        }
    else:
        # Model-only checkpoint
        new_state_dict = {'model.' + k: v for k, v in checkpoint.items()}

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model.to(device)

    return model


def build_final_volumes_dir_name(
    checkpoints: List[str],
    axes: List[str]
) -> str:
    """
    Build a combined name for the final volumes directory when --save_final is used.

    For each checkpoint, we retrieve its stem (file name without extension),
    and for each axis specification, we expand 'xy' into 'x-y'.

    Example:
        checkpoints = ['fold0-best.ckpt', 'fold1-best.ckpt']
        axes = ['xy', 'xy']
        => 'predictions_fold0-best_x-y-fold1-best_x-y'

    Args:
        checkpoints (List[str]): List of checkpoint file paths.
        axes (List[str]): Corresponding axes specification for each checkpoint.

    Returns:
        str: The combined directory name, e.g. 'predictions_fold0-best_x-y-fold1-best_x-y'.
    """
    result_parts = []
    for ckpt_path_str, axis_spec in zip(checkpoints, axes):
        ckpt_stem = Path(ckpt_path_str).stem
        # Expand 'xy' => 'x-y', else keep as is
        if axis_spec.lower() == 'xy':
            axes_joined = 'x-y'
        else:
            axes_joined = axis_spec.lower()
        result_parts.append(f"{ckpt_stem}_{axes_joined}")

    combined_name = "-".join(result_parts)
    return f"predictions_{combined_name}"


def get_model_name_from_checkpoint(checkpoint_path: Path) -> str:
    """
    Extract the model name from the checkpoint file name (without extension).

    Example:
        If checkpoint file is 'fold0-best-model.ckpt', this returns 'fold0-best-model'.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.

    Returns:
        str: Model name (the file name without '.ckpt').
    """
    return checkpoint_path.stem


def filter_samples_to_predict(
    df: pd.DataFrame,
    checkpoint_path: Path,
    axis: str,
    force_prediction: bool,
    save_probas: bool
) -> pd.DataFrame:
    """
    Filter the DataFrame to retain only samples that have not been predicted yet,
    unless force_prediction is True, in which case no filtering is applied.

    Args:
        df (pd.DataFrame): DataFrame containing sample information.
        checkpoint_path (Path): Path to the model checkpoint.
        axis (str): Axis being processed ('x', 'y', or 'z').
        force_prediction (bool): If True, ignore existing predictions and predict again.
        save_probas (bool): Whether the raw probability volumes are being saved
                            (to 'predictions_probas/{model_name}/{axis}') or not.

    Returns:
        pd.DataFrame: Filtered DataFrame with samples pending prediction.
    """
    if force_prediction:
        print(
            f"Force prediction is enabled. All samples will be processed for axis '{axis}'."
        )
        return df

    model_name = get_model_name_from_checkpoint(checkpoint_path)
    # If saving probas, use predictions_probas dir. Otherwise, same logic but still skip if found.
    predictions_dir = (
        checkpoint_path.parent / 'predictions_probas' / model_name / axis
    )

    if not predictions_dir.exists():
        print(
            f"Directory {predictions_dir} does not exist. "
            f"Processing all samples for checkpoint '{checkpoint_path.name}' and axis '{axis}'."
        )
        remaining_df = df.copy()
    else:
        predicted_samples = {
            sample_id for sample_id in df['sample_id'].unique()
            if (predictions_dir / sample_id / 'volume.npy').exists()
        }

        print(
            f"Found {len(predicted_samples)} already predicted sample(s) "
            f"for checkpoint '{checkpoint_path.name}' (axis {axis}). Skipping these."
        )
        remaining_df = df[~df['sample_id'].isin(predicted_samples)].copy()

    print(
        f"Number of samples to process for checkpoint '{checkpoint_path.name}' "
        f"(axis {axis}): {len(remaining_df)}"
    )
    print(f"Sample IDs to process for axis '{axis}': {remaining_df['sample_id'].unique()}")

    return remaining_df


class AsyncVolumeSaver:
    """
    Asynchronous saver for saving volume predictions to disk.
    """

    def __init__(
        self,
        dataset: SeismicDataset,
        axis: str = 'z',
        volume_output_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize the AsyncVolumeSaver.

        Args:
            dataset (SeismicDataset): Dataset being processed.
            axis (str, optional): Axis being processed ('x', 'y', or 'z').
            volume_output_dir (Optional[Path], optional): Directory to save volume predictions.
        """
        self.dataset = dataset
        self.axis = axis
        self.volume_output_dir = volume_output_dir

        if self.volume_output_dir:
            self.volume_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Volume outputs will be saved to: {self.volume_output_dir}")

        self.expected_frames = 1259 if axis == 'z' else 300
        self.completed_volumes = 0
        self.queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()

        self.volume_buffers: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros((300, 300, 1259), dtype=np.float32)
        )
        self.frames_received: Dict[str, set] = defaultdict(set)

        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def add_prediction(self, sample_id: str, frame_idx: int, pred: np.ndarray) -> None:
        """
        Add a prediction to the queue for asynchronous writing.

        Args:
            sample_id (str): Identifier for the sample.
            frame_idx (int): Index of the frame.
            pred (np.ndarray): Prediction array for the frame.
        """
        self.queue.put((sample_id, frame_idx, pred))

    def finish(self) -> None:
        """
        Signal the writer thread to finish processing and wait for it to terminate.
        """
        self.stop_event.set()
        self.writer_thread.join()

    def _writer_loop(self) -> None:
        """
        Internal method running in a separate thread to handle writing predictions.
        """
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                sample_id, frame_idx, pred = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.axis == 'x':
                self.volume_buffers[sample_id][frame_idx, :, :] = pred
            elif self.axis == 'y':
                self.volume_buffers[sample_id][:, frame_idx, :] = pred
            else:  # axis == 'z'
                self.volume_buffers[sample_id][:, :, frame_idx] = pred

            self.frames_received[sample_id].add(frame_idx)

            # Once all frames are received, save the volume
            if len(self.frames_received[sample_id]) == self.expected_frames:
                volume = self.volume_buffers[sample_id]
                if self.volume_output_dir:
                    output_sample_dir = self.volume_output_dir / sample_id
                    output_sample_dir.mkdir(parents=True, exist_ok=True)
                    np.save(output_sample_dir / 'volume.npy', volume)

                self.completed_volumes += 1
                print(f"Completed volume {self.completed_volumes}: {sample_id}")

                del self.volume_buffers[sample_id]
                del self.frames_received[sample_id]


def batchwise_predict(
    model: SeismicFaultDetector,
    loader: DataLoader,
    saver: AsyncVolumeSaver,
    device: str,
    use_cpu: bool = False,
    input_size: Optional[List[int]] = None,
    dtype: str = 'bf16'
) -> None:
    """
    Perform batchwise prediction on the given data loader and send predictions
    to the asynchronous saver.

    Args:
        model (SeismicFaultDetector): The fault detection model.
        loader (DataLoader): DataLoader for the dataset.
        saver (AsyncVolumeSaver): Saver instance for saving predictions.
        device (str): Device to perform inference on ('cuda' or 'cpu').
        use_cpu (bool, optional): Force CPU usage for inference.
        input_size (Optional[List[int]], optional): Desired input size (height, width).
        dtype (str, optional): Data type for inference ('float16', 'float32', 'bf16').

    Raises:
        ValueError: If an unsupported dtype is provided.
    """
    print("Starting batchwise prediction...")
    progress_bar = tqdm(loader, desc="Processing batches", leave=True)

    dtype_mapping = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bf16': torch.bfloat16
    }
    if dtype not in dtype_mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Supported dtypes: {list(dtype_mapping.keys())}"
        )
    torch_dtype = dtype_mapping[dtype]

    for batch in progress_bar:
        seismic, _, sample_ids, frame_indices, axis = batch
        seismic = seismic.to(device)

        with torch.no_grad():
            if device == 'cpu':
                autocast_device = 'cpu'
                autocast_dtype = torch.bfloat16
            else:
                autocast_device = 'cuda'
                autocast_dtype = torch_dtype

            with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype):
                padded_seismic = []
                paddings = []
                for img in seismic:
                    img = img.unsqueeze(0)
                    padded_img, _, padding = pad_to_size(img, input_size)
                    padded_seismic.append(padded_img)
                    paddings.append(padding)

                padded_seismic = torch.cat(padded_seismic)
                preds = model(padded_seismic)
                preds = torch.sigmoid(preds)

            preds = preds.to(torch.float32)  # Ensure consistency
            for pred, padding, sample_id, frame_idx in zip(
                preds, paddings, sample_ids, frame_indices
            ):
                pred = unpad_image(pred.squeeze(), padding).cpu().numpy()
                saver.add_prediction(sample_id, frame_idx, pred)

    print("Batchwise prediction completed.")


def predict_single_checkpoint(
    checkpoint_path: Path,
    axis: str,
    full_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    save_probas: bool,
    force_prediction: bool,
    root_dir: str,
    dtype: str,
    compile_model: bool,
    cpu: bool,
    device: str
) -> None:
    """
    Predict volumes for a single checkpoint on a specified axis.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.
        axis (str): Axis to process ('x', 'y', or 'z').
        full_df (pd.DataFrame): DataFrame containing all samples.
        batch_size (int): Batch size for processing.
        num_workers (int): Number of workers for DataLoader.
        save_probas (bool): Whether to save raw probability volumes.
        force_prediction (bool): If True, ignore existing predictions.
        root_dir (str): Root directory containing 2D slices data.
        dtype (str): Data type for inference.
        compile_model (bool): If True, compile the model for optimized performance.
        cpu (bool): If True, force inference on CPU.
        device (str): Device to perform computations on ('cuda' or 'cpu').
    """
    config = load_checkpoint_config(checkpoint_path)
    input_size = config.get('input_size', [320, 1280])

    print(f"Processing checkpoint: {checkpoint_path.name}")
    print(f"Axis: {axis}")
    print(f"Using input size: {input_size}")

    # Decide which samples to predict
    remaining_df = filter_samples_to_predict(
        df=full_df,
        checkpoint_path=checkpoint_path,
        axis=axis,
        force_prediction=force_prediction,
        save_probas=save_probas
    )

    axis_df = remaining_df[remaining_df['axis'] == axis]
    print(
        f"Number of remaining samples after filtering for axis '{axis}': {len(axis_df)}"
    )
    print(f"Sample IDs to process for axis '{axis}': {axis_df['sample_id'].unique()}")

    if axis_df.empty:
        print("No samples to process for this checkpoint/axis combination.")
        return

    dataset = SeismicDataset(
        root_dir=root_dir,
        df=axis_df,
        mode='inference',
        nchans=config['nchans']
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=seismic_collate_fn
    )

    model = load_model(checkpoint_path, config, device)
    if compile_model:
        model = torch.compile(model)

    model_name = get_model_name_from_checkpoint(checkpoint_path)
    volume_output_dir = None
    if save_probas:
        volume_output_dir = checkpoint_path.parent / "predictions_probas" / model_name / axis
        volume_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving raw probability volumes to: {volume_output_dir}")

    saver = AsyncVolumeSaver(
        dataset=dataset,
        axis=axis,
        volume_output_dir=volume_output_dir
    )

    batchwise_predict(
        model=model,
        loader=loader,
        saver=saver,
        device=device,
        use_cpu=cpu,
        input_size=input_size,
        dtype=dtype
    )
    saver.finish()
    print(f"Finished processing checkpoint: {checkpoint_path.name} for axis: {axis}")


def ensemble_volumes_and_save(
    all_predictions: List[Path],
    dataset_index: pd.DataFrame,
    output_path: Path,
    save_threshold: float,
    device: str = 'cpu',
    min_mean_conf: Optional[float] = None,
    save_final_volumes: bool = False,
    final_volumes_dir: Optional[Path] = None
) -> None:
    """
    Ensemble predictions from multiple models, compute confidence metrics,
    and optionally save final thresholded volumes.

    Args:
        all_predictions (List[Path]): List of directories containing model predictions.
        dataset_index (pd.DataFrame): DataFrame containing dataset information.
        output_path (Path): Path to save the final submission file.
        save_threshold (float): Threshold to apply to averaged predictions.
        device (str, optional): Device for computations ('cpu' or 'cuda'). Defaults to 'cpu'.
        min_mean_conf (Optional[float], optional): Minimum mean confidence required.
            If below, the predicted volume is set to zero. Defaults to None.
        save_final_volumes (bool, optional): If True, saves the final thresholded volumes
            to final_volumes_dir.
        final_volumes_dir (Optional[Path], optional): Directory to save final thresholded volumes.
    """
    sample_ids = sorted(dataset_index['sample_id'].unique())

    if output_path.exists():
        output_path.unlink()
        print(f"Existing file {output_path} has been removed.")

    if save_final_volumes and final_volumes_dir:
        final_volumes_dir.mkdir(parents=True, exist_ok=True)
        print(f"Final thresholded volumes will be saved to: {final_volumes_dir}")

    print(
        "Ensembling volumes from these prediction directories:"
        f"\n{[str(p) for p in all_predictions]}"
    )

    # For each sample_id, gather predictions across all models
    for sample_id in sample_ids:
        volumes = []
        for prediction_dir in all_predictions:
            volume_path = prediction_dir / sample_id / 'volume.npy'
            if volume_path.exists():
                volumes.append(torch.from_numpy(np.load(volume_path)).to(device))

        if not volumes:
            print(f"Warning: No predictions found for sample {sample_id}. Skipping.")
            continue

        stacked_volumes = torch.stack(volumes, dim=0)
        avg_volume = torch.mean(stacked_volumes, dim=0)
        std_volume = torch.std(stacked_volumes, dim=0)

        # Confidence metrics
        distance_from_middle = torch.abs(avg_volume - 0.5) * 2  # [0, 1]
        agreement_score = 1 - std_volume
        confidence_scores = distance_from_middle * agreement_score

        # Thresholding
        binary_volume = (avg_volume > save_threshold)
        active_points_confidence = confidence_scores[binary_volume]
        total_confidence = torch.sum(active_points_confidence).item()
        mean_confidence = (
            torch.mean(active_points_confidence).item()
            if torch.any(binary_volume)
            else 0.0
        )
        num_fault_points = torch.sum(binary_volume).item()

        print(f"\nConfidence metrics for {sample_id}:")
        print(f"- Number of fault points: {num_fault_points}")
        print(f"- Total confidence score: {total_confidence:.2f}")
        print(f"- Mean confidence per point: {mean_confidence:.4f}")

        if min_mean_conf is not None and mean_confidence < min_mean_conf:
            print(
                f"Mean confidence ({mean_confidence:.4f}) below min_mean_conf ({min_mean_conf}). "
                f"Setting volume {sample_id} to zero."
            )
            binary_volume = torch.zeros_like(binary_volume)
            num_fault_points = 0

        # Save in submission
        binary_volume = binary_volume.cpu().numpy().astype(np.uint16)
        create_submission(sample_id, binary_volume, str(output_path), append=True)

        # Optionally save final thresholded volume
        if save_final_volumes and final_volumes_dir:
            sample_dir = final_volumes_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.save(sample_dir / 'volume.npy', binary_volume)

    print("Ensembling and processing of volumes completed.")


def create_submission(
    sample_id: str, prediction: np.ndarray, submission_path: str, append: bool = True
):
    """
    Function to create submission file out of one test prediction at a time.

    Parameters:
        sample_id (str): ID of the survey used for prediction.
        prediction (np.ndarray): Binary 3D np.ndarray of predicted faults.
        submission_path (str): Path to save submission.
        append (bool, optional): Whether to append prediction to existing .npz or create new one. Defaults to True.

    Returns:
        None
    """
    if append:
        try:
            submission = dict(np.load(submission_path))
        except FileNotFoundError:
            print("File not found, new submission will be created.")
            submission = dict({})
    else:
        submission = dict({})

    # Positive value coordinates
    coordinates = np.stack(np.where(prediction == 1)).T
    coordinates = coordinates.astype(np.uint16)

    submission.update({sample_id: coordinates})

    np.savez(submission_path, **submission)
