import torch
from torch import nn
import timm
import torch.nn.functional as F

from .model_EVA_MHA_Handler import EVA16_Speed_large_model

    
class EVA16_MHA(EVA16_Speed_large_model):
    """
    EVA16 model variant with dual channel fusion Multi-Head Attention (MHA).

    This model inherits from EVA16_Speed_large_model and uses a dual fusion attention
    mechanism to integrate information across multiple channels. It applies two
    multi-head attention layers:
      - Channel-wise attention with `channel_num_heads` heads.
      - Fusion attention with `fusion_num_heads` heads.

    Parameters:
        encoder (str): Name of the encoder backbone model.
        pretrained (bool): Load pretrained weights if True.
        mode (str): Operating mode, 'train' or 'test'.
        split_at (int): Transformer block index at which to split per-channel and fused processing.
        channel_num_heads (int): Number of heads for channel multi-head attention.
        fusion_num_heads (int): Number of heads for fusion multi-head attention.
        channel_fusion (str): Type of channel fusion; 'dual' enables dual MHA fusion.

    Defaults:
        split_at = 10
        channel_num_heads = 2
        fusion_num_heads = 2
        channel_fusion = 'dual'

    Inherits all other behaviors from EVA16_Speed_large_model.
    """
    def __init__(self, encoder, pretrained=True, 
                 mode='train',
                 split_at=10, 
                 channel_num_heads=2,
                 fusion_num_heads=2, 
                 channel_fusion='dual'):
        super().__init__(encoder, 
                         pretrained, 
                         mode,
                         split_at,
                         channel_num_heads,
                         fusion_num_heads, 
                         channel_fusion)
