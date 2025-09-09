import torch
from torch import nn
import timm
import torch.nn.functional as F

from .model_EVA_MHA_Handler import EVA16_Speed_large_model

    
class EVA16_MHA(EVA16_Speed_large_model):
    """
    Specialized variant of EVA16_Speed_large_model with modified transformer configuration.

    This subclass inherits from EVA16_Speed_large_model and overrides default parameters
    to use a different split index and increased multi-head attention heads for channel fusion.

    Args:
        encoder (str): Name of the encoder model to use from timm.
        pretrained (bool): Whether to load pretrained weights for the encoder backbone.
        mode (str): Mode of operation, either 'train' or 'test'.
        split_at (int): Transformer block split index between per-channel and fused processing.
                        Defaults to 9 (vs. 10 in parent).
        channel_num_heads (int): Number of attention heads for channel attention.
                                 Defaults to 4.
        fusion_num_heads (int): Number of attention heads for fusion attention.
                                Defaults to 4 (vs. 2 in parent).
        channel_fusion (str): Type of channel fusion method, either 'dual' or 'single'.
                              Defaults to 'dual'.

    Notes:
        - The difference from the base class is primarily in the default `split_at`,
          and the number of heads for channel and fusion multi-head attention layers.
        - All other behavior and methods are inherited unchanged.
    """

    def __init__(self, encoder, pretrained=True, 
                 mode='train',
                 split_at=9, 
                 channel_num_heads=4,
                 fusion_num_heads=4, 
                 channel_fusion='dual'):
        super().__init__(encoder, 
                         pretrained, 
                         mode,
                         split_at,
                         channel_num_heads,
                         fusion_num_heads, 
                         channel_fusion)
