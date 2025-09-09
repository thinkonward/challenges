import torch
from torch import nn
import timm
import torch.nn.functional as F

from .model_EVA_MHA_Handler import EVA16_Speed_large_model

    
class EVA16_MHA(EVA16_Speed_large_model):
    """
    EVA16 model variant with dual channel fusion Multi-Head Attention (MHA).

    This subclass customizes EVA16_Speed_large_model to use a dual channel fusion strategy,
    which applies two levels of multi-head attention:
      - Channel-to-channel attention with `channel_num_heads`.
      - Fusion attention with `fusion_num_heads`.

    Default parameters are set to:
      - `split_at=10`: Transformer block split point between per-channel and fused processing.
      - `channel_num_heads=4`: Number of attention heads for channel MHA.
      - `fusion_num_heads=2`: Number of attention heads for fusion MHA.
      - `channel_fusion='dual'`: Enables dual fusion mechanism.

    Args:
        encoder (str): Name of the encoder backbone model.
        pretrained (bool): Whether to load pretrained weights.
        mode (str): Mode of operation, e.g., 'train' or 'test'.
        split_at (int): Index to split transformer blocks into per-channel and fusion processing.
        channel_num_heads (int): Number of heads for the channel multi-head attention.
        fusion_num_heads (int): Number of heads for fusion multi-head attention.
        channel_fusion (str): Type of channel fusion to use ('dual' for this class).

    Notes:
        - Uses complex dual fusion attention to integrate information across channels.
        - Inherits all other behaviors from EVA16_Speed_large_model.
    """
    def __init__(self, encoder, pretrained=True, 
                 mode='train',
                 split_at=10, 
                 channel_num_heads=4,
                 fusion_num_heads=2, 
                 channel_fusion='dual'):
        super().__init__(encoder, 
                         pretrained, 
                         mode,
                         split_at,
                         channel_num_heads,
                         fusion_num_heads, 
                         channel_fusion)
