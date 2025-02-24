from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def pad_to_size(
    image: torch.Tensor,
    target_size: Tuple[int, int],
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int, int, int]]:
    """
    Pad images to match the target size.

    This function pads the input image and an optional mask to ensure they meet the specified
    target dimensions. Padding is applied symmetrically on all sides using reflection mode.

    Args:
        image (torch.Tensor): Input tensor image with shape [B, C, H, W].
        target_size (Tuple[int, int]): Desired size as (target_height, target_width).
        mask (Optional[torch.Tensor], optional): Optional mask tensor to pad similarly.
            Defaults to None.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int, int, int]]:
            - Padded image tensor.
            - Padded mask tensor if provided, otherwise None.
            - Padding applied as (pad_w1, pad_w2, pad_h1, pad_h2).
    """
    # Extract current height and width from the image tensor
    current_height, current_width = image.shape[-2:]
    target_height, target_width = target_size

    # Calculate required padding for height and width
    pad_height = max(0, target_height - current_height)
    pad_width = max(0, target_width - current_width)

    # Initialize padding values
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # Apply padding only if necessary
    if pad_height > 0 or pad_width > 0:
        # Calculate symmetric padding for height
        pad_h1 = pad_height // 2
        pad_h2 = pad_height - pad_h1

        # Calculate symmetric padding for width
        pad_w1 = pad_width // 2
        pad_w2 = pad_width - pad_w1

        # Define the padding tuple (left, right, top, bottom)
        padding = (pad_w1, pad_w2, pad_h1, pad_h2)

        # Apply padding to the image using reflection mode
        image = F.pad(image, padding, mode='reflect')

        # Apply the same padding to the mask if provided
        if mask is not None:
            # Unsqueeze to add a batch dimension, pad, then squeeze back
            mask = F.pad(mask.unsqueeze(0), padding, mode='reflect').squeeze(0)

    return image, mask, padding


def unpad_image(
    image: torch.Tensor,
    padding: Tuple[int, int, int, int]
) -> torch.Tensor:
    """
    Remove padding from an image.

    This function removes the specified padding from the input image tensor.

    Args:
        image (torch.Tensor): Padded tensor image with shape [B, C, H, W] or similar.
        padding (Tuple[int, int, int, int]): Padding values as (pad_w1, pad_w2, pad_h1, pad_h2).

    Returns:
        torch.Tensor: Unpadded image tensor.
    """
    # If no padding was applied, return the image as is
    if not any(padding):
        return image

    pad_w1, pad_w2, pad_h1, pad_h2 = padding

    # Determine the slicing indices for height
    if pad_h2 > 0:
        h_start, h_end = pad_h1, -pad_h2
    else:
        h_start, h_end = pad_h1, None

    # Determine the slicing indices for width
    if pad_w2 > 0:
        w_start, w_end = pad_w1, -pad_w2
    else:
        w_start, w_end = pad_w1, None

    # Slice the image tensor to remove padding
    unpadded_image = image[..., h_start:h_end, w_start:w_end]

    return unpadded_image
