from copy import deepcopy
from types import MethodType


import torch
from torch import nn
import timm
import torch.nn.functional as F


def process_input(x):
    """
    Preprocess and reshape seismic input tensor for model compatibility.

    This function reshapes and permutes the input tensor to break down large spatial
    dimensions, then interpolates to the required spatial size suitable for model input.

    Args:
        x (torch.Tensor): Input seismic tensor of shape (N, 5, 10000, 31), where
                          N = batch size,
                          5 = number of source coordinates,
                          10000 = spatial dimension (timesteps),
                          31 = spatial dimension (receivers).

    Returns:
        torch.Tensor: Processed tensor of shape (N, 5, 560, 560), resized spatially
                      to match the model's expected input size.

    Steps:
        1. Reshape from (N, 5, 10000, 31) to (N, 5, 100, 100, 31) to break down
           the large spatial dimension (10000 ~ 100 * 100).
        2. Permute dimensions to reorder axes as (N, 5, 100, 31, 100) for correct alignment.
        3. Flatten last two spatial dimensions into one: (N, 5, 100, 3100).
        4. Resize/interpolate spatial dimensions to (560, 560) using bilinear interpolation,
           matching model's input size and patch size constraints.

    Note:
        The function assumes input dimensions are compatible with the reshaping operations.
    """
    N = x.size(0)
    # Reshape to separate large spatial dimension into smaller chunks
    x = x.reshape(N, 5, 100, 100, 31)

    # Permute to reorder axes for proper alignment before flattening
    x = x.permute(0, 1, 2, 4, 3)

    # Flatten last two spatial dimensions
    x = x.reshape(N, 5, 100, 3100)

    # Interpolate to desired spatial size (560 x 560) using bilinear mode
    x = F.interpolate(x, size=(560, 560), mode="bilinear", align_corners=False)
    return x



class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


        
class ChannelFusionDual_MHA(nn.Module):
    """
    Dual Multi-Head Attention module for channel fusion with positional encoding.

    This module performs attention-based fusion of multiple input channels using:
    1. Cross-channel attention to capture relationships between channels.
    2. Fusion attention with a learnable query token to combine channel features per spatial token.

    Args:
        dim (int): Dimension of the input feature embeddings.
        num_channels (int): Number of input channels.
        channel_num_heads (int): Number of attention heads for cross-channel attention.
        fusion_num_heads (int): Number of attention heads for fusion attention.
        dropout (float): Dropout rate applied in attention layers.

    Input:
        x (torch.Tensor): Input tensor of shape (B, C, T, D) where
                          B = batch size,
                          C = number of channels,
                          T = number of spatial tokens,
                          D = embedding dimension.

    Returns:
        torch.Tensor: Output tensor of shape (B, T, D), representing fused channel information per token.

    Details:
        - Adds learnable positional embeddings to input channels.
        - Applies multi-head self-attention across channels per token.
        - Applies fusion attention using a learnable query token to aggregate channel information.
        - Layer normalization is used before and after attention layers.
    """
    def __init__(self, dim, num_channels=5, channel_num_heads=4, 
                 fusion_num_heads=2, dropout=0):
        super().__init__()
        self.num_channels = num_channels
        self.dim = dim

        # Positional embedding for channel order
        self.channel_pos = nn.Parameter(torch.randn(1, num_channels, dim))

        # Channel-to-channel attention
        self.cross_channel_mha = nn.MultiheadAttention(embed_dim=dim, 
                                                       num_heads=channel_num_heads, 
                                                       dropout=dropout,
                                                       batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        # Fusion query token (one per spatial token)
        self.fusion_query = nn.Parameter(torch.randn(1, 1, dim))

        # Channel fusion attention with learnable query
        self.fusion_mha = nn.MultiheadAttention(embed_dim=dim, 
                                                num_heads=fusion_num_heads, 
                                                dropout=dropout, 
                                                batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass of the dual MHA channel fusion.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, T, D).

        Returns:
            torch.Tensor: Fused output tensor with shape (B, T, D).
        """
        B, C, T, D = x.shape
        assert C == self.num_channels and D == self.dim

        # Reshape input to (B*T, C, D) for cross-channel attention per token
        x = x.permute(0, 2, 1, 3).reshape(B * T, C, D)
        x = x + self.channel_pos  # Add positional embedding to channels

        # Cross-channel multi-head attention with residual connection
        residual = x
        x = self.norm1(x)
        x, _ = self.cross_channel_mha(x, x, x)
        x = residual + x

        # Fusion multi-head attention using a learnable query token per token
        query = self.fusion_query.expand(B * T, -1, -1)  # (B*T, 1, D)
        fused, _ = self.fusion_mha(query, x, x)  # Output shape: (B*T, 1, D)
        fused = fused.squeeze(1).view(B, T, D)  # Reshape to (B, T, D)

        # Final normalization
        fused = self.norm2(fused)
        return fused


class ChannelFusionSingle_MHA(nn.Module):
    """
    Single Multi-Head Attention module for channel fusion.

    This module treats input channels as a sequence and performs self-attention
    across channels per spatial token, then averages the attended channel features.

    Args:
        dim (int): Dimension of the input feature embeddings.
        num_channels (int): Number of input channels.
        channel_num_heads (int): Number of attention heads.
        dropout (float): Dropout rate for attention layer.

    Input:
        x (torch.Tensor): Input tensor of shape (B, C, T, D) where
                          B = batch size,
                          C = number of channels,
                          T = number of spatial tokens,
                          D = embedding dimension.

    Returns:
        torch.Tensor: Output tensor of shape (B, T, D), representing fused channel features.

    Details:
        - Applies layer normalization before attention.
        - Applies multi-head self-attention treating channels as sequence tokens.
        - Uses residual connection and averages across channels for fusion.
    """
    def __init__(self, dim, num_channels=5, channel_num_heads=4, dropout=0.0):
        super().__init__()
        self.num_channels = num_channels

        self.mha = nn.MultiheadAttention(embed_dim=dim, 
                                         num_heads=channel_num_heads, 
                                         dropout=dropout, 
                                         batch_first=True)
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        """
        Forward pass of the single MHA channel fusion.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, T, D).

        Returns:
            torch.Tensor: Fused output tensor with shape (B, T, D).
        """
        B, C, T, D = x.shape

        # Reshape input to (B*T, C, D) treating channels as sequence dimension
        x = x.permute(0, 2, 1, 3).reshape(B * T, C, D)

        # Pre-norm and multi-head self-attention with residual connection
        residual = x
        x = self.norm(x)
        attn_out, _ = self.mha(x, x, x)
        x = residual + attn_out

        # Average over channels for fusion
        fused = x.mean(dim=1)  # Shape: (B*T, D)
        fused = fused.view(B, T, D)
        return fused

    

class EVA16_Speed_large_model(nn.Module):
    """
    Large EVA16-based model for seismic velocity (speed) prediction with channel-wise multi-head attention fusion.

    This model is designed to process seismic data consisting of multiple channels,
    applying a Vision Transformer backbone (EVA16) with a two-stage processing:
    - Per-channel transformer blocks on separated channels.
    - Fusion of channels via multi-head attention (dual or single type).
    - Further transformer blocks on fused channel representations.
    - Final prediction head producing velocity maps.

    Supports test-time augmentation (horizontal flipping) for improved robustness.

    Args:
        encoder (str): Name of the encoder model to use from timm.
        pretrained (bool): Whether to load pretrained weights for the encoder backbone.
        mode (str): Mode of operation, either 'train' or 'test'. Influences TTA behavior.
        split_at (int): Index at which to split the transformer blocks between per-channel and fused processing.
        channel_num_heads (int): Number of attention heads used in channel fusion.
        fusion_num_heads (int): Number of attention heads in fusion attention (used in dual fusion).
        channel_fusion (str): Type of channel fusion method, either 'dual' or 'single'.

    Attributes:
        backbone (nn.Module): Vision Transformer backbone (EVA16) from timm.
        head (nn.Linear): Final linear layer predicting seismic velocity.
        global_min_y (float): Minimum velocity value for normalization.
        global_max_y (float): Maximum velocity value for normalization.
        channel_fusion_mha (nn.Module): Channel fusion module (dual or single multi-head attention).
        split_at (int): Transformer block split index.

    Methods:
        forward(x_in): Forward pass returning predicted velocity map.
        tta_forward(x_in): Forward pass with horizontal flip test-time augmentation.

    Input:
        x_in (torch.Tensor): Input tensor of shape (N, 5, 10000, 31) representing
                             batch size N, 5 channels, spatial dimensions 10000x31.

    Output:
        torch.Tensor: Predicted velocity map tensor of shape approximately (N, 1, 300, 1259)
                      with values scaled to the expected velocity range [global_min_y, global_max_y].
    """

    def __init__(self, encoder, pretrained=True, 
                 mode='train',
                 split_at=10, 
                 channel_num_heads=4,
                 fusion_num_heads=2,      
                 channel_fusion='dual'):
        super().__init__()
        
        # Create the Vision Transformer backbone with single input channel for each channel separately
        backbone = timm.create_model(
            encoder,
            pretrained=pretrained,
            dynamic_img_size=True,
            in_chans=1,
        )
        
        self.backbone = backbone
        self.head = nn.Linear(1024, 300)  # final head mapping transformer output dim to velocity depth
        
        # Data-specific velocity normalization bounds
        self.global_min_y = 1.5
        self.global_max_y = 4.5
        
        self.mode = mode
        self.split_at = split_at
        
        # Select channel fusion strategy
        if channel_fusion == "single":
            self.channel_fusion_mha = ChannelFusionSingle_MHA(dim=1024, channel_num_heads=channel_num_heads)
        elif channel_fusion == "dual":
            self.channel_fusion_mha = ChannelFusionDual_MHA(dim=1024, 
                                                           channel_num_heads=channel_num_heads,
                                                           fusion_num_heads=fusion_num_heads)
        else:
            raise ValueError(f"Unsupported channel_fusion type: {channel_fusion}")
        
    def tta_forward(self, x_in):
        """
        Perform test-time augmentation by horizontal flipping and averaging predictions.

        Args:
            x_in (torch.Tensor): Input tensor (N, 5, 10000, 31).

        Returns:
            torch.Tensor: Velocity prediction after TTA.
        """
        N = x_in.size(0)
        
        # Flip horizontally (last dimension)
        x_in = torch.flip(x_in, dims=[-1])
        
        # Process input (reshape & interpolate)
        x = process_input(x_in)  # Output shape: (N, 5, 560, 560)
        N, C, H, W = x.size()
        
        # Patch embedding on each channel individually
        x = self.backbone.patch_embed(x.reshape(N * C, 1, H, W))
        x, rot_pos_embed = self.backbone._pos_embed(x)
        
        # Per-channel transformer blocks
        for blk in self.backbone.blocks[:self.split_at]:
            x = blk(x, rope=rot_pos_embed)
        
        # Reshape for fusion
        x = x.reshape(N, 5, *x.size()[1:])  # (N, 5, tokens, dim)
        
        # Fuse channels using attention
        x = self.channel_fusion_mha(x)
        
        # Remaining transformer blocks on fused tokens
        for blk in self.backbone.blocks[self.split_at:]:
            x = blk(x, rope=rot_pos_embed)
        
        x = self.backbone.norm(x)
        x = x[:, 1:]  # Remove CLS token
        
        # Final head and reshape
        x = self.head(x)
        x = x.permute(0, 2, 1)  # (N, depth, width)
        
        # Interpolate to target size
        x = F.interpolate(x.unsqueeze(1), size=(300, 1259), mode="area")
        
        # Scale outputs to velocity range
        x = torch.sigmoid(x.float())
        x = x * (self.global_max_y - self.global_min_y) + self.global_min_y
        
        # Flip back to original orientation
        x = torch.flip(x, dims=[-2])
        
        return x
    
    def forward(self, x_in):
        """
        Forward pass through the model.

        Args:
            x_in (torch.Tensor): Input tensor of shape (N, 5, 10000, 31).

        Returns:
            torch.Tensor: Predicted velocity map tensor.
        """
        N = x_in.size(0)
        
        x = process_input(x_in)  # reshape and interpolate input to (N, 5, 560, 560)
        N, C, H, W = x.size()
        
        # Patch embedding on each channel separately
        x = self.backbone.patch_embed(x.reshape(N * C, 1, H, W))
        x, rot_pos_embed = self.backbone._pos_embed(x)
        
        # Process initial transformer blocks per channel
        for blk in self.backbone.blocks[:self.split_at]:
            x = blk(x, rope=rot_pos_embed)
        
        # Reshape for channel fusion
        x = x.reshape(N, 5, *x.size()[1:])  # (N, 5, tokens, dim)
        
        # Fuse channels with attention
        x = self.channel_fusion_mha(x)
        
        # Process remaining transformer blocks on fused tokens
        for blk in self.backbone.blocks[self.split_at:]:
            x = blk(x, rope=rot_pos_embed)
        
        x = self.backbone.norm(x)
        x = x[:, 1:]  # remove CLS token
        
        # Final linear head to produce velocity depth
        x = self.head(x)
        x = x.permute(0, 2, 1)
        
        # Interpolate to fixed size
        x = F.interpolate(x.unsqueeze(1), size=(300, 1259), mode="area")
        
        # Scale outputs to velocity range
        x = torch.sigmoid(x.float())
        x_final = x * (self.global_max_y - self.global_min_y) + self.global_min_y
        
        # Apply test time augmentation if in test mode
        if self.mode == 'test':
            p1 = self.tta_forward(x_in)
            x_final = torch.quantile(torch.stack([x_final, p1]), q=0.5, dim=0)
        
        return x_final
