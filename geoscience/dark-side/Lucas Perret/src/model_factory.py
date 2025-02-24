from typing import Tuple, Optional

import segmentation_models_pytorch as smp

from .smp_with_dropout import SMPWithDropout


def create_model(
    archi: str,
    nchans: int,
    num_classes: int,
    axis: str,
    encoder_name: Optional[str] = None,
    encoder_weights: Optional[str] = "imagenet",
    model_size: Optional[str] = None,
    input_size: Tuple[int, int] = (320, 1280),
    dropout: float = 0.5,
) -> SMPWithDropout:
    """
    Creates and returns a seismic fault detection model based on the specified architecture.

    Args:
        archi (str): Model architecture to use ('umamba', 'unet', 'unetpp').
        nchans (int): Number of input channels.
        num_classes (int): Number of output classes.
        axis (str): Axis to use for validation ('x', 'y', 'z', 'xy').
        encoder_name (Optional[str], optional): Encoder backbone name. Required for 'unet' and 'unetpp'. Defaults to None.
        encoder_weights (str, optional): Encoder backbone weights (for unet and unetpp). Defaults to 'imagenet'.
        model_size (Optional[str], optional): Model size. Required for 'umamba'. Defaults to None.
        input_size (Tuple[int, int], optional): Input size as (height, width). Defaults to (320, 1280).
        dropout (float, optional): Dropout probability. Defaults to 0.5.

    Returns:
        SMPWithDropout: Configured model instance.

    Raises:
        ValueError: If both or neither of `encoder_name` and `model_size` are specified.
        ValueError: If `archi` is 'umamba' but `model_size` is not provided.
        ValueError: If `archi` is 'unet' or 'unetpp' but `encoder_name` is not provided.
        ValueError: If an unknown architecture is specified.
    """
    if (encoder_name is None) == (model_size is None):
        raise ValueError("Exactly one of encoder_name or model_size must be specified")

    if archi == "umamba":
        from nnunetv2.nets.UMambaEnc_2d import create_umamba_model

        if model_size is None:
            raise ValueError("umamba requires model_size")

        model = create_umamba_model(
            input_size=input_size,
            input_channels=nchans,
            num_classes=num_classes,
            model_size=model_size,
            dropout_p=dropout,
        )
    elif archi in {"unet", "unetpp"}:

        if encoder_name is None:
            raise ValueError(f"{archi} requires encoder_name")

        model_class = smp.Unet if archi == "unet" else smp.UnetPlusPlus

        model = SMPWithDropout(
            model_class=model_class,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=nchans,
            classes=num_classes,
            max_dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {archi}")

    return model
