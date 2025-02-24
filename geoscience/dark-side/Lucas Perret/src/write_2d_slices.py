"""Process 3D seismic volumes into 2D slices.

This script takes 3D seismic data volumes and converts them into 2D slices along
specified axes (x, y, and/or z). It supports processing of both training and testing
data parts, with options to filter by data mode (train/test). The script handles
multiple data formats and can process fault or horizon labels if present.

The output includes:
- 2D slice files (.npy format) organized by sample and axis
- A dataset index saved as a parquet file with metadata for all slices
"""

import argparse
import multiprocessing as mp
import re
import shutil
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_data_parts(
    root_dir: Optional[Union[str, Path]] = None,
    data_parts: Optional[List[Union[str, Path]]] = None,
    mode: Optional[str] = None,
) -> List[Path]:
    """Return a list of data part directories, optionally filtered by mode (train/test).

    Args:
        root_dir: Root directory containing data part folders.
        data_parts: List of specific data part paths to process.
        mode: If specified, only returns data parts matching the mode ('train' or 'test').

    Returns:
        List of Path objects for each data part directory.

    Raises:
        ValueError: If neither root_dir nor data_parts is provided, or if no matching
            data parts are found.
    """
    if root_dir is not None:
        root_dir = Path(root_dir)
        data_parts = sorted(
            [
                d
                for d in root_dir.glob("*")
                if d.is_dir() and "dark-side" in d.name.lower() and (
                    not mode
                    or (mode == "train" and "train" in d.name.lower())
                    or (mode == "test" and "test" in d.name.lower())
                )
            ]
        )
        if not data_parts:
            raise ValueError(f"No data parts found in {root_dir}")
    elif data_parts is not None:
        data_parts = [Path(dp) for dp in data_parts]
    else:
        raise ValueError("Either root_dir or data_parts must be provided")

    return data_parts


def get_volume_paths(data_parts: List[Path]) -> Dict[str, Dict[str, Union[Path, str]]]:
    """Scan data parts for seismic volumes and their corresponding labels.

    Args:
        data_parts: List of data part directories to scan.

    Returns:
        Dictionary mapping sample IDs to volume information containing:
        - seismic: Path to seismic data file
        - label: Path to label file (if exists)
        - data_part: Name of the data part
        - format: Data format ('dark-side' or 'every-layer')
        - label_type: Type of labels ('fault' or 'horizon')
    """
    print("Scanning for seismic volumes in data parts...")
    volumes = {}

    for part_path in data_parts:
        sample_dirs = [d for d in part_path.iterdir() if d.is_dir()]
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name

            seismic_files = list(sample_dir.glob("seismicCubes_RFC_fullstack_*.npy"))
            fault_files = list(sample_dir.glob("fault_segments_*.npy"))

            if seismic_files:  # dark-side format
                volumes[f"{sample_id}"] = {
                    "seismic": seismic_files[0],
                    "label": fault_files[0] if fault_files else None,
                    "data_part": part_path.name,
                    "format": "dark-side",
                    "label_type": "fault",
                }
            else:  # Check for every-layer format
                files = list(sample_dir.glob("*.npy"))
                blocks = {}

                for f in files:
                    match = re.search(r"block-(\d+)", f.name)
                    if match:
                        block_num = match.group(1)
                        if block_num not in blocks:
                            blocks[block_num] = {"data_part": part_path.name}

                        if "seismic" in f.name:
                            blocks[block_num]["seismic"] = f
                        elif "horizon_labels" in f.name:
                            blocks[block_num]["label"] = f

                for block_num, block_info in blocks.items():
                    if "seismic" in block_info:
                        volume_id = f"{sample_id}_block{block_num}"
                        block_info["format"] = "every-layer"
                        block_info["label_type"] = "horizon"
                        volumes[volume_id] = block_info

    print(f"Found {len(volumes)} seismic volumes")
    return volumes


def get_slice(volume: np.ndarray, idx: int, axis: str) -> np.ndarray:
    """Extract a 2D slice from a 3D volume along specified axis.

    Args:
        volume: 3D numpy array to slice
        idx: Index along the specified axis
        axis: Axis along which to slice ('x', 'y', or 'z')

    Returns:
        2D numpy array representing the slice
    """
    if axis == "z":
        return volume[..., idx].copy()
    if axis == "y":
        return volume[:, idx, :].copy()
    return volume[idx, :, :].copy()


def process_volume(volume_info: tuple, output_dir: Union[str, Path], axes: List[str]) -> List[Dict]:
    sample_id, paths = volume_info
    output_dir = Path(output_dir)

    seismic = np.load(paths["seismic"], allow_pickle=True).astype(np.float32)
    label = np.load(paths["label"], allow_pickle=True) if paths.get("label") else None

    frame_data = []

    for axis in axes:
        volume_slice_dir = output_dir / paths["data_part"] / sample_id / f"slices_{axis}"
        seismic_slice_dir = volume_slice_dir / "seismic"
        label_slice_dir = volume_slice_dir / "label" if paths.get("label") else None

        seismic_slice_dir.mkdir(parents=True, exist_ok=True)
        if label_slice_dir:
            label_slice_dir.mkdir(parents=True, exist_ok=True)

        slice_dim = {"x": 0, "y": 1, "z": 2}[axis]
        n_slices = seismic.shape[slice_dim]

        for idx in range(n_slices):
            seismic_slice = get_slice(seismic, idx, axis)
            label_slice = get_slice(label, idx, axis) if label is not None else None

            seismic_path = seismic_slice_dir / f"slice_{idx:04d}.npy"
            label_path = label_slice_dir / f"slice_{idx:04d}.npy" if label_slice_dir else None

            np.save(seismic_path, seismic_slice)
            if label_slice is not None:
                np.save(label_path, label_slice)

            positive_pixels = int(np.sum(label_slice == 1)) if label_slice is not None else 0

            rel_seismic_path = str(
                Path(paths["data_part"]) / sample_id / f"slices_{axis}/seismic/slice_{idx:04d}.npy"
            )
            rel_label_path = (
                str(Path(paths["data_part"]) / sample_id / f"slices_{axis}/label/slice_{idx:04d}.npy")
                if label_path
                else None
            )

            frame_info = {
                "sample_id": sample_id,
                "data_part": paths["data_part"],
                "format": paths["format"],
                "label_type": paths["label_type"],
                "has_labels": label is not None,
                "axis": axis,
                "frame_idx": idx,
                "frame_path": rel_seismic_path,
                "label_path": rel_label_path,
                "frame_height": seismic_slice.shape[0],
                "frame_width": seismic_slice.shape[1],
                "volume_x": seismic.shape[0],
                "volume_y": seismic.shape[1],
                "volume_z": seismic.shape[2],
                "positive_pixels": positive_pixels,
            }

            frame_data.append(frame_info)

    return frame_data


def write_2d_slices(
    root_dir: Optional[Union[str, Path]] = None,
    data_parts: Optional[List[Union[str, Path]]] = None,
    output_dir: Union[str, Path] = "./2d_slices",
    axes: List[str] = ["z"],
    num_workers: Optional[int] = None,
    mode: Optional[str] = None,
) -> None:
    """Create 2D slices from 3D seismic volumes along specified axes.

    Args:
        root_dir: Root directory containing data parts
        data_parts: List of specific data part paths to process
        output_dir: Directory where slices and metadata will be saved
        axes: List of axes along which to slice ('x', 'y', 'z')
        num_workers: Number of parallel worker processes
        mode: If specified, only process data parts matching mode ('train' or 'test')

    Raises:
        ValueError: If invalid axes are specified
    """
    print("Setting up directories and scanning data...")

    for axis in axes:
        if axis not in ["x", "y", "z"]:
            raise ValueError(f"Invalid axis: {axis}. Must be one of ['x', 'y', 'z']")

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Existing output directory '{output_dir}' found and has been removed.")

    data_parts = get_data_parts(root_dir, data_parts, mode)
    volumes = get_volume_paths(data_parts)

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    all_frame_data = []
    stats = {"total_frames": {axis: 0 for axis in axes}, "formats": {}, "label_types": {}}

    process_fn = partial(process_volume, output_dir=output_dir, axes=axes)

    print("Processing volumes...")
    with tqdm(total=len(volumes), desc=f"Processing {len(volumes)} volumes along axes {','.join(axes)}") as pbar:
        if num_workers == 1:
            # Sequential processing without multiprocessing
            for volume_info in volumes.items():
                frame_data_list = process_fn(volume_info)
                all_frame_data.extend(frame_data_list)

                if frame_data_list:
                    first_frame = frame_data_list[0]
                    stats["formats"][first_frame["format"]] = stats["formats"].get(first_frame["format"], 0) + 1
                    stats["label_types"][first_frame["label_type"]] = (
                        stats["label_types"].get(first_frame["label_type"], 0) + 1
                    )

                    for frame in frame_data_list:
                        stats["total_frames"][frame["axis"]] += 1

                    pbar.set_postfix({"frames": len(all_frame_data), "volume": first_frame["sample_id"]})
                pbar.update(1)
        else:
            # Use multiprocessing
            with mp.Pool(num_workers) as pool:
                for frame_data_list in pool.imap_unordered(process_fn, volumes.items()):
                    all_frame_data.extend(frame_data_list)

                    if frame_data_list:
                        first_frame = frame_data_list[0]
                        stats["formats"][first_frame["format"]] = stats["formats"].get(first_frame["format"], 0) + 1
                        stats["label_types"][first_frame["label_type"]] = (
                            stats["label_types"].get(first_frame["label_type"], 0) + 1
                        )

                        for frame in frame_data_list:
                            stats["total_frames"][frame["axis"]] += 1

                        pbar.set_postfix({"frames": len(all_frame_data), "volume": first_frame["sample_id"]})
                    pbar.update(1)

    print("Creating dataset dataframe...")
    df = pd.DataFrame(all_frame_data)
    columns = [
        "sample_id",
        "data_part",
        "format",
        "label_type",
        "has_labels",
        "axis",
        "frame_idx",
        "frame_path",
        "label_path",
        "frame_height",
        "frame_width",
        "volume_x",
        "volume_y",
        "volume_z",
        "positive_pixels",
    ]
    df = df[columns]

    parquet_path = output_dir / "dataset.parquet"
    df.to_parquet(parquet_path)
    print(f"Dataset index saved to: {parquet_path}")

    print("\nProcessing complete:")
    print(f"- Processed {len(volumes):,} volumes")
    print(f"- Total frames processed: {len(all_frame_data):,}")
    print("- Frames per axis:")
    for axis, count in stats["total_frames"].items():
        print(f"  {axis}: {count:,}")
    print(f"- Output directory: {output_dir}")
    print("\nComplete DataFrame:")
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 3D seismic volumes to 2D slices")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root-dir", type=str, help="Root directory containing data parts")
    group.add_argument("--data-parts", type=str, nargs="+", help="List of data part paths")

    parser.add_argument("--output-dir", type=str, default="./2d_slices", help="Output directory for 2D slices")
    parser.add_argument(
        "--axes", type=str, nargs="+", default=["z"], choices=["x", "y", "z"],
        help="Axes along which to slice"
    )
    parser.add_argument("--num-workers", type=int, help="Number of worker processes")
    parser.add_argument("--mode", type=str, choices=["train", "test"], help="Process only train or test data parts")

    args = parser.parse_args()

    write_2d_slices(
        root_dir=args.root_dir,
        data_parts=args.data_parts,
        output_dir=args.output_dir,
        axes=args.axes,
        num_workers=args.num_workers,
        mode=args.mode,
    )
