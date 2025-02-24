#!/usr/bin/env python3
"""
Visualization script for seismic dataset.

This script provides an interactive visualization tool for exploring seismic data volumes.
Users can navigate through different frames and volumes using keyboard controls.

Controls:
- Left/Right arrows: Navigate through frames
- A/Z: Navigate through volumes
- Q: Quit visualization

Usage:
    python visualize_seismic.py /path/to/root_dir /path/to/dataframe.parquet --axis z
"""

import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

# Add parent directory to system path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import SeismicDataset  # Assurez-vous que ce chemin est correct


def get_default_display_size(frame_shape: Tuple[int, int], axis: str) -> Tuple[int, int]:
    """
    Determine default display size based on frame shape and visualization axis.

    Args:
        frame_shape (Tuple[int, int]): Shape of the frame (height, width).
        axis (str): Visualization axis ('x', 'y', or 'z').

    Returns:
        Tuple[int, int]: Display width and height.
    """
    height, width = frame_shape
    if axis in ['x', 'y']:
        # Vertical stacking for x and y axes
        disp_width = int(width * 1.2)  # Slightly wider for separators
        disp_height = int(height * 3.5)  # 3 stacked images + text space
    else:  # axis 'z'
        # Horizontal layout for z axis
        disp_width = int(width * 3.5)  # 3 side-by-side images + separators
        disp_height = int(height * 1.2)  # Slightly taller for text

    # Limit maximum size to avoid display issues
    max_dimension = 1200
    if disp_width > max_dimension or disp_height > max_dimension:
        scale = max_dimension / max(disp_width, disp_height)
        disp_width = int(disp_width * scale)
        disp_height = int(disp_height * scale)

    return disp_width, disp_height


def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """
    Normalize slice data to the 0-255 range.

    Args:
        slice_data (np.ndarray): Input array to normalize.

    Returns:
        np.ndarray: Normalized array as uint8.
    """
    slice_min, slice_max = slice_data.min(), slice_data.max()
    if slice_max - slice_min == 0:
        return np.zeros_like(slice_data, dtype=np.uint8)
    normalized = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
    return normalized


def create_separator(shape: Tuple[int, int, int], vertical: bool = True) -> np.ndarray:
    """
    Create a separator line between visualizations.

    Args:
        shape (Tuple[int, int, int]): Shape of the original image (H, W, C).
        vertical (bool, optional): If True, create a vertical separator; else horizontal. Defaults to True.

    Returns:
        np.ndarray: Separator image.
    """
    if vertical:
        separator = np.zeros((shape[0], 10, shape[2]), dtype=np.uint8)
    else:
        separator = np.zeros((10, shape[1], shape[2]), dtype=np.uint8)
    separator[:] = [0, 0, 255]  # Red color in BGR format
    return separator


def visualize_dataset(
    dataset: SeismicDataset,
    start_idx: int = 0,
    display_size: Optional[Tuple[int, int]] = None
) -> None:
    """
    Interactive visualization of the seismic dataset.

    Users can navigate through frames and volumes using keyboard controls.

    Args:
        dataset (SeismicDataset): SeismicDataset instance to visualize.
        start_idx (int, optional): Starting frame index. Defaults to 0.
        display_size (Optional[Tuple[int, int]], optional): Tuple of (width, height) for display.
            If None, defaults are calculated based on frame shape and axis. Defaults to None.
    """
    # Ensure DataFrame is sorted by sample_id and frame_idx
    dataset.df = dataset.df.sort_values(['sample_id', 'frame_idx']).reset_index(drop=True)

    # Initialize tracking of last visited frame for each volume
    last_visited_frames: Dict[str, int] = {}
    current_idx = start_idx

    while True:
        # Fetch frame information
        frame_info = dataset.df.iloc[current_idx]
        current_sample = frame_info['sample_id']

        # Update last visited frame for current volume
        last_visited_frames[current_sample] = current_idx

        # Retrieve seismic data and label
        seismic, label, sample_id, frame_idx, axis = dataset[current_idx]
        seismic = seismic.squeeze(0).numpy()

        # Normalize seismic data for visualization
        viz = normalize_slice(seismic)
        viz_rgb = cv2.cvtColor(viz, cv2.COLOR_GRAY2RGB)

        # Handle label overlay if available
        if label is not None:
            label = label.numpy()
            if frame_info['label_type'] == 'fault':
                # Red overlay for faults
                overlay = viz_rgb.copy()
                overlay[label > 0] = [0, 0, 255]  # BGR format
                viz_with_overlay = cv2.addWeighted(viz_rgb, 0.7, overlay, 0.3, 0)
                mask_viz = np.zeros_like(viz_rgb)
                mask_viz[label > 0] = [255, 255, 255]
            else:  # label_type == 'horizon'
                label_norm = normalize_slice(label)
                horizon_viz = cv2.applyColorMap(label_norm, cv2.COLORMAP_JET)
                viz_with_overlay = cv2.addWeighted(viz_rgb, 0.7, horizon_viz, 0.3, 0)
                mask_viz = horizon_viz
        else:
            viz_with_overlay = viz_rgb.copy()
            mask_viz = viz_rgb.copy()

        # Set default display size if not specified
        if display_size is None:
            display_size = get_default_display_size(viz_rgb.shape[:2], axis)

        # Assemble images based on axis
        if axis in ['x', 'y']:
            # Vertical stacking for x and y axes
            viz_final = np.vstack((
                viz_with_overlay,
                create_separator(viz_rgb.shape, vertical=False),
                viz_rgb,
                create_separator(viz_rgb.shape, vertical=False),
                mask_viz
            ))
        else:
            # Horizontal layout for z axis
            viz_final = np.hstack((
                viz_with_overlay,
                create_separator(viz_rgb.shape, vertical=True),
                viz_rgb,
                create_separator(viz_rgb.shape, vertical=True),
                mask_viz
            ))

        # Resize for display
        current_height, current_width = viz_final.shape[:2]
        scale_factor = min(display_size[0] / current_width, display_size[1] / current_height)
        new_width = int(current_width * scale_factor)
        new_height = int(current_height * scale_factor)
        viz_final_resized = cv2.resize(viz_final, (new_width, new_height))

        # Add labels to the visualization
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (255, 255, 255)  # White color in BGR

        if axis in ['x', 'y']:
            # Position text for vertical layout
            section_height = new_height // 3
            x_pos = 10
            overlay_text = 'Overlay' if label is not None else 'Raw'
            cv2.putText(viz_final_resized, overlay_text,
                       (x_pos, section_height // 6), font, font_scale, font_color, font_thickness)
            cv2.putText(viz_final_resized, 'Raw Data',
                       (x_pos, section_height + section_height // 6), font, font_scale, font_color, font_thickness)
            label_text = frame_info['label_type'].capitalize() if label is not None else 'Raw'
            cv2.putText(viz_final_resized, label_text,
                       (x_pos, 2 * section_height + section_height // 6), font, font_scale, font_color, font_thickness)
        else:
            # Position text for horizontal layout
            section_width = new_width // 3
            y_pos = 20
            overlay_text = 'Overlay' if label is not None else 'Raw'
            cv2.putText(viz_final_resized, overlay_text,
                       (section_width // 4, y_pos), font, font_scale, font_color, font_thickness)
            cv2.putText(viz_final_resized, 'Raw Data',
                       (section_width + 10 + section_width // 4, y_pos), font, font_scale, font_color, font_thickness)
            label_text = frame_info['label_type'].capitalize() if label is not None else 'Raw'
            cv2.putText(viz_final_resized, label_text,
                       (2 * section_width + 20 + section_width // 4, y_pos), font, font_scale, font_color, font_thickness)

        # Display the visualization
        cv2.imshow('Seismic Data Viewer', viz_final_resized)
        cv2.setWindowProperty('Seismic Data Viewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        # Update window title with frame information
        current_volume = dataset.df[dataset.df['sample_id'] == sample_id]
        n_frames = len(current_volume)
        title = (
            f'Volume: {sample_id} | '
            f'Frame: {frame_idx}/{n_frames - 1} | '
            f'Shape: {seismic.shape[0]}x{seismic.shape[1]} | '
            f'Axis: {axis} | '
            f'Type: {frame_info["label_type"].capitalize() if label is not None else "Raw"} | '
            f'Min: {seismic.min():.2f}, Max: {seismic.max():.2f}'
        )
        cv2.setWindowTitle('Seismic Data Viewer', title)

        # Handle keyboard input for navigation
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 81:  # Left arrow key
            current_idx = max(0, current_idx - 1)
        elif key == 83:  # Right arrow key
            current_idx = min(len(dataset) - 1, current_idx + 1)
        elif key == ord('a'):  # Navigate to previous volume
            try:
                current_sample = dataset.df.iloc[current_idx]['sample_id']
                # Get current volume's first frame index
                curr_vol_start = dataset.df[dataset.df['sample_id'] == current_sample].index[0]

                # If not at the start of the dataset
                if curr_vol_start > 0:
                    # Get the previous volume's ID
                    prev_vol = dataset.df.iloc[curr_vol_start - 1]['sample_id']

                    # Use last visited frame if available, else last frame of previous volume
                    if prev_vol in last_visited_frames:
                        current_idx = last_visited_frames[prev_vol]
                    else:
                        # Get all frames from previous volume
                        prev_frames = dataset.df[dataset.df['sample_id'] == prev_vol].index
                        current_idx = prev_frames[-1]
                        last_visited_frames[prev_vol] = current_idx
            except (IndexError, KeyError):
                pass

        elif key == ord('z'):  # Navigate to next volume
            try:
                current_sample = dataset.df.iloc[current_idx]['sample_id']
                # Get current volume's last frame index
                curr_vol_end = dataset.df[dataset.df['sample_id'] == current_sample].index[-1]

                # If not at the end of the dataset
                if curr_vol_end < len(dataset.df) - 1:
                    # Get the next volume's ID
                    next_vol = dataset.df.iloc[curr_vol_end + 1]['sample_id']

                    # Use last visited frame if available, else first frame of next volume
                    if next_vol in last_visited_frames:
                        current_idx = last_visited_frames[next_vol]
                    else:
                        # Get all frames from next volume
                        next_frames = dataset.df[dataset.df['sample_id'] == next_vol].index
                        current_idx = next_frames[0]
                        last_visited_frames[next_vol] = current_idx
            except (IndexError, KeyError):
                pass


def main() -> None:
    """
    Main entry point for the visualization script.

    Parses command-line arguments, loads the dataset, and starts the visualization.
    """
    parser = argparse.ArgumentParser(description='Visualize seismic dataset')
    parser.add_argument('root_dir', type=str,
                        help='Root directory containing cached data')
    parser.add_argument('df_path', type=str,
                        help='Path to the parquet DataFrame file')
    parser.add_argument('--axis', type=str, default='z',
                        choices=['x', 'y', 'z'],
                        help='Axis for visualization (default: z)')
    parser.add_argument('--display-width', type=int, default=None,
                        help='Display width in pixels')
    parser.add_argument('--display-height', type=int, default=None,
                        help='Display height in pixels')

    args = parser.parse_args()

    # Load and filter DataFrame based on the specified axis
    try:
        df = pd.read_parquet(args.df_path)
    except Exception as e:
        print(f"Error loading DataFrame from {args.df_path}: {e}")
        sys.exit(1)

    filtered_df = df[df['axis'] == args.axis].copy()

    # Create dataset instance
    dataset = SeismicDataset(
        root_dir=args.root_dir,
        df=filtered_df,
        mode='valid',
        nchans=1
    )

    print(f"\nDataset size: {len(dataset):,} frames")
    print("\nControls:")
    print("- Left/Right arrows: Navigate through frames")
    print("- A/Z: Navigate through volumes")
    print("- Q: Quit visualization")

    # Set display size if provided
    display_size: Optional[Tuple[int, int]] = None
    if args.display_width is not None and args.display_height is not None:
        display_size = (args.display_width, args.display_height)

    # Start visualization
    visualize_dataset(dataset, display_size=display_size)


if __name__ == '__main__':
    main()
