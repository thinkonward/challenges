"""
utils.py

This module provides utility functions for setting up logging, ensuring reproducibility,
saving configurations, and loading checkpoints. These utilities are essential for maintaining consistency and facilitating
model training and inference in deep learning workflows.
"""

import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from argparse import Namespace
import logging
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy, DeepSpeedStrategy


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up the logging configuration.

    Args:
        verbose (bool, optional): If True, set logging level to DEBUG. Defaults to False.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def set_seed(seed: int, use_det_algo: bool = False) -> None:
    """
    Set the random seed for various libraries to ensure reproducibility.

    This function configures the seed for Python's random module, NumPy, and PyTorch.
    It also adjusts PyTorch's backend settings to promote deterministic behavior.

    Args:
        seed (int): The seed value to be set.
        use_det_algo (bool, optional): If True, enforces the use of deterministic algorithms in PyTorch.
            Defaults to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    if use_det_algo:
        torch.use_deterministic_algorithms(True)
    print("Seed set to:", seed)


def save_config(
    args: Namespace,
    dirpath: str,
    fold_info: Dict[int, Dict[str, Any]]
) -> None:
    """
    Save configuration and fold results to a JSON file.

    This function serializes the command-line arguments and fold information,
    adds a timestamp, and writes the combined configuration to a JSON file
    in the specified directory.

    Args:
        args (Namespace): Parsed command-line arguments.
        dirpath (str): Directory path where the configuration file will be saved.
        fold_info (Dict[int, Dict[str, Any]]): Dictionary containing information for each fold.

    Raises:
        Exception: If an error occurs while writing the configuration file.
    """
    config_dir = Path(dirpath)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Convert args to dictionary
    config = vars(args).copy()

    # Add timestamp
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Add fold information with a descriptive key
    config['fold_results'] = fold_info

    # Save to JSON file
    config_path = config_dir / 'config.json'
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")


def load_ckpt(
    model: torch.nn.Module,
    ckpt_path: str,
    exclude_modules: Optional[List[str]] = None,
    load_optim: bool = False
) -> Tuple[int, int, int, Dict[str, Any]]:
    """
    Load a checkpoint into a model, optionally excluding certain modules and loading optimizer states.

    This function loads the model's state dictionary from a checkpoint file, allowing for the exclusion
    of specified modules. It can also load optimizer and scheduler states if desired.

    Args:
        model (torch.nn.Module): The model instance into which the checkpoint will be loaded.
        ckpt_path (str): Path to the checkpoint file.
        exclude_modules (Optional[List[str]], optional): List of substrings. If a key in the state_dict
            contains any of these substrings, it will be excluded from loading. Defaults to None.
        load_optim (bool, optional): If True, attempts to load the optimizer and scheduler states.
            Defaults to False.

    Returns:
        Tuple[int, int, int, Dict[str, Any]]:
            - Number of modules successfully loaded.
            - Number of modules missing in the checkpoint.
            - Number of unexpected modules found in the checkpoint.
            - Dictionary containing optimizer and scheduler states if `load_optim` is True.

    Raises:
        ValueError: If invalid GPU indices are provided when loading optimizer states.
    """
    # Load the complete checkpoint (including optimizer states if load_optim is True)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)  # Fallback if 'state_dict' key is missing

    # Create sets of model and checkpoint keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Filter out keys that match any of the excluded substrings
    excluded_keys = set()
    if exclude_modules:
        for key in checkpoint_keys:
            if any(substr in key for substr in exclude_modules):
                excluded_keys.add(key)

    # Identify loaded, missing, and unexpected keys
    filtered_checkpoint_keys = checkpoint_keys - excluded_keys
    loaded_keys = model_keys.intersection(filtered_checkpoint_keys)
    missing_keys = model_keys - filtered_checkpoint_keys
    unexpected_keys = filtered_checkpoint_keys - model_keys

    # Create a cleaned state_dict excluding the specified modules
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key in model_keys and key not in excluded_keys:
            cleaned_state_dict[key] = value

    # Load the cleaned state_dict into the model
    model.load_state_dict(cleaned_state_dict, strict=False)

    # Initialize dictionary to hold optimizer and scheduler states
    optim_states: Dict[str, Any] = {}
    if load_optim:
        try:
            if 'optimizer_states' in checkpoint:
                optim_states['optimizer_state'] = checkpoint['optimizer_states'][0]

            if 'lr_schedulers' in checkpoint:
                optim_states['scheduler_state'] = checkpoint['lr_schedulers'][0]

            # Save the current learning rate if available
            if 'optimizer_states' in checkpoint:
                param_groups = checkpoint['optimizer_states'][0].get('param_groups', [])
                if param_groups:
                    optim_states['learning_rate'] = param_groups[0].get('lr', None)

            # Save the current epoch if available
            if 'epoch' in checkpoint:
                optim_states['epoch'] = checkpoint['epoch']

        except Exception as e:
            print(f"\nWarning: Couldn't load optimizer states: {str(e)}")

    # Log the results of the checkpoint loading process
    print("\nCheckpoint loading results:")
    print(f"Successfully loaded {len(loaded_keys)} modules:")
    for key in sorted(loaded_keys):
        print(f"  ✓ {key}")

    if missing_keys:
        print(f"\nMissing {len(missing_keys)} modules (initialized to default values):")
        for key in sorted(missing_keys):
            print(f"  × {key}")

    if unexpected_keys:
        print(f"\nFound {len(unexpected_keys)} unexpected modules in checkpoint (ignored):")
        for key in sorted(unexpected_keys):
            print(f"  ? {key}")

    if excluded_keys:
        print(f"\nExcluded {len(excluded_keys)} modules as requested:")
        for key in sorted(excluded_keys):
            print(f"  ! {key}")

    if load_optim:
        if optim_states:
            print("\nOptimizer states loaded:")
            for key in optim_states:
                print(f"  ✓ {key}")
        else:
            print("\nNo optimizer states found in checkpoint")

    return len(loaded_keys), len(missing_keys), len(unexpected_keys), optim_states
