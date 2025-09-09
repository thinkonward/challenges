import torch
from torch import nn
import timm
import torch.nn.functional as F

from .model_EVA_MHA_Handler import EVA16_Speed_large_model

    
class EVA16_MHA(EVA16_Speed_large_model):
    """
    Variant of EVA16_Speed_large_model configured for single channel fusion.

    This subclass modifies the default parameters to:
    - Use `split_at=10` (default split index for transformer blocks).
    - Use `channel_fusion='single'` for a simpler channel fusion strategy.
    - Accept `fusion_num_heads=None` since fusion MHA is not used in single fusion mode.
    - Keep 4 heads for channel multi-head attention by default.

    Args:
        encoder (str): Name of the encoder model to instantiate.
        pretrained (bool): Whether to use pretrained weights for the encoder.
        mode (str): Mode of operation, e.g., 'train' or 'test'.
        split_at (int): Index to split transformer blocks between per-channel and fused processing.
        channel_num_heads (int): Number of heads for channel multi-head attention.
        fusion_num_heads (int or None): Number of heads for fusion attention; None if not used.
        channel_fusion (str): Channel fusion strategy, 'single' or 'dual'. Defaults to 'single'.

    Notes:
        - This class uses the single-channel fusion approach, which employs a simpler
          multi-head attention mechanism over channels without fusion MHA.
        - Fusion_num_heads is set to None by default because it is not required in 'single' mode.
        - Inherits all other functionality from EVA16_Speed_large_model.
    """

    def __init__(self, encoder, pretrained=True, 
                 mode='train',
                 split_at=10, 
                 channel_num_heads=4,
                 fusion_num_heads=None, 
                 channel_fusion='single'):
        super().__init__(encoder, 
                         pretrained, 
                         mode,
                         split_at,
                         channel_num_heads,
                         fusion_num_heads, 
                         channel_fusion)
