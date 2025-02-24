import sys
import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any, Tuple

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from src.model_factory import create_model


def parse_checkpoint(ckpt_path: Path, device: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load checkpoint and extract model state dict and config parameters.
    Handles the renaming of model components for compatibility.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Extract model state dict and clean up compiled model paths if present
    state_dict = checkpoint['state_dict']

    # Clean up state dict keys:
    # 1. Remove 'model._orig_mod.' prefix (from compiled models)
    # 2. Remove 'model.' prefix
    # 3. Handle renaming of encoder/decoder components
    clean_state_dict = {}
    for k, v in state_dict.items():
        # Remove compilation prefixes
        k = k.replace('model._orig_mod.', '')
        k = k.replace('model.', '')

        # Handle component renaming
        k = k.replace('base_encoder', 'base_model.encoder')
        k = k.replace('base_decoder', 'base_model.decoder')
        k = k.replace('base_segmentation_head', 'base_model.segmentation_head')

        clean_state_dict[k] = v.to(device)

    # Extract relevant config parameters
    config = {
        'archi': checkpoint['hyper_parameters']['archi'],
        'val_axis': checkpoint['hyper_parameters']['val_axis'],
        'nchans': checkpoint['hyper_parameters']['nchans'],
        'num_classes': checkpoint['hyper_parameters']['num_classes'],
        'lr': checkpoint['hyper_parameters']['learning_rate'],
        'scheduler_gamma': checkpoint['hyper_parameters']['scheduler_gamma'],
        'optim_name': checkpoint['hyper_parameters']['optim_name'],
        'encoder_name': checkpoint['hyper_parameters'].get('encoder_name'),
        'model_size': checkpoint['hyper_parameters'].get('model_size'),
        'input_size': checkpoint['hyper_parameters']['input_size'],
        'dropout': checkpoint['hyper_parameters'].get('dropout', 0.2),
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'best_val_score': checkpoint.get('best_val_score', None),
    }

    return clean_state_dict, config


def validate_model_loading(state_dict: Dict[str, Any], config: Dict[str, Any], device: str) -> bool:
    """
    Test if the extracted model can be properly loaded and run.
    Returns True if validation succeeds, raises exception otherwise.
    """
    print("Creating model with config...")
    # Create model with config
    model = create_model(
        archi=config['archi'],
        nchans=config['nchans'],
        num_classes=config['num_classes'],
        axis=config['val_axis'],
        encoder_name=config['encoder_name'],
        model_size=config['model_size'],
        input_size=config['input_size'],
        dropout=config['dropout'],
    )

    model = model.to(device)

    print("Loading state dict...")
    # Load state dict
    model.load_state_dict(state_dict)

    print("Testing with random input...")
    # Test with random input
    model.eval()
    test_input = torch.randn(1, config['nchans'], *config['input_size'], device=device)
    with torch.no_grad():
        _ = model(test_input)

    print("Model validation successful!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert Lightning checkpoints to model-only checkpoints')
    parser.add_argument('checkpoints', nargs='+', type=str, help='Path(s) to checkpoint file(s)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                       help='Device to use for model loading and validation')
    args = parser.parse_args()

    for ckpt_path in args.checkpoints:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            continue

        print(f"\nProcessing: {ckpt_path}")

        # Extract model and config
        state_dict, config = parse_checkpoint(ckpt_path, args.device)

        # Print checkpoint info
        print("\nCheckpoint Info:")
        print(f"Architecture: {config['archi']}")
        print(f"Validation Axis: {config['val_axis']}")
        print(f"Input Channels: {config['nchans']}")
        print(f"Input Size: {config['input_size']}")
        print(f"Epoch: {config['epoch']}")
        print(f"Best Val Score: {config['best_val_score']}")

        # Validate model loading
        print("\nValidating model loading...")
        validate_model_loading(state_dict, config, args.device)

        # Save model-only checkpoint
        model_path = ckpt_path.with_suffix('.model.ckpt')
        torch.save(state_dict, model_path)
        print(f"\nSaved model-only checkpoint to: {model_path}")


if __name__ == '__main__':
    main()
