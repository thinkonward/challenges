
from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn as nn
from monai.networks.blocks import UpSample, SubpixelUpsample
import torch.nn.functional as F
import timm


class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.Identity,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.conv= nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride, 
            padding=padding, 
            bias=False,
        )
        self.norm = norm_layer(out_channels) if norm_layer != nn.Identity else nn.Identity()
        self.act= act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.Tanh(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1), 
            nn.Sigmoid(),
            )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention2d(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule2d(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class DecoderBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
        dropout: float = -1.0,
    ):
        super().__init__()

        # Upsample block
        if upsample_mode == "pixelshuffle":
            self.upsample= SubpixelUpsample(
                spatial_dims= 2,
                in_channels= in_channels,
                scale_factor= scale_factor,
            )
        else:
            self.upsample = UpSample(
                spatial_dims= 2,
                in_channels= in_channels,
                out_channels= in_channels,
                scale_factor= scale_factor,
                mode= upsample_mode,
            )

        if intermediate_conv:
            k= 3
            c= skip_channels if skip_channels != 0 else in_channels
            self.intermediate_conv = nn.Sequential(
                ConvBnAct2d(c, c, k, k//2),
                ConvBnAct2d(c, c, k, k//2),
                )
        else:
            self.intermediate_conv= None

        self.attention1 = Attention2d(
            name= attention_type, 
            in_channels= in_channels + skip_channels,
            )

        self.conv1 = ConvBnAct2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )

        self.conv2 = ConvBnAct2d(
            out_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )
        self.attention2 = Attention2d(
            name= attention_type, 
            in_channels= out_channels,
            )

        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None 

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.intermediate_conv is not None:
            if skip is not None:
                skip = self.intermediate_conv(skip)
            else:
                x = self.intermediate_conv(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class UnetDecoder2d(nn.Module):
    """
    Unet decoder.
    Source: https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self,
        encoder_channels: Tuple[int],
        skip_channels: Tuple[int] = None,
        decoder_channels: Tuple = (256, 128, 64, 32),
        scale_factors: Tuple = (2,2,2,2),
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = "scse",
        intermediate_conv: bool = True,
        upsample_mode: str = "pixelshuffle",
        dropout: float = -1.0,
    ):
        super().__init__()
        
        if len(encoder_channels) == 4:
            decoder_channels= decoder_channels[1:]
        self.decoder_channels= decoder_channels
        
        if skip_channels is None:
            skip_channels= list(encoder_channels[1:]) + [0]

        # Build decoder blocks
        in_channels= [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList()

        for i, (ic, sc, dc) in enumerate(zip(in_channels, skip_channels, decoder_channels)):
            self.blocks.append(
                DecoderBlock2d(
                    ic, sc, dc, 
                    norm_layer= norm_layer,
                    attention_type= attention_type,
                    intermediate_conv= intermediate_conv,
                    upsample_mode= upsample_mode,
                    scale_factor= scale_factors[i],
                    dropout= dropout
                    )
            )

    def forward(self, feats: List[torch.Tensor]):
        res= [feats[0]]
        feats= feats[1:]

        # Decoder blocks
        for i, b in enumerate(self.blocks):
            skip= feats[i] if i < len(feats) else None
            res.append(
                b(res[-1], skip=skip),
                )
            
        return res

class SegmentationHead2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: Tuple[int] = (2,2),
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        self.conv= nn.Conv2d(
            in_channels, out_channels, kernel_size= kernel_size,
            padding= kernel_size//2
        )
        self.upsample = UpSample(
            spatial_dims= 2,
            in_channels= out_channels,
            out_channels= out_channels,
            scale_factor= scale_factor,
            mode= mode,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x
        

class Net(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        fuse_ch: int = 64,
        one_channel: bool = False,
        norm_layer: str = "identity",
        dropout: float = -1.0,
        y_min_max_norm: bool = False,
        horizontal_tta: bool = False,
    ):
        super().__init__()
        
        # Encoder
        self.backbone= timm.create_model(
            backbone,
            in_chans= 1 if one_channel else 5,
            pretrained= pretrained,
            features_only= True,
            drop_path_rate=0.0,
            )
        ecs= [_["num_chs"] for _ in self.backbone.feature_info][::-1] # [768, 512, 256, 128]

        if norm_layer == "bn":
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == "gn":
            self.norm_layer = nn.GroupNorm
        elif norm_layer == "ln":
            self.norm_layer = nn.LayerNorm
        elif norm_layer == "in":
            self.norm_layer = nn.InstanceNorm2d
        else:
            self.norm_layer = nn.Identity
        # Decoder
        self.decoder= UnetDecoder2d(
            encoder_channels= ecs,
            norm_layer= self.norm_layer,
            dropout= dropout,
        )

        self.seg_head= SegmentationHead2d(
            in_channels= 4*fuse_ch, # self.decoder.decoder_channels[-1], # 32
            out_channels= 1,
            scale_factor= 1,
        )
        
        # 1×1 convs to reduce all feature maps to fuse_ch channels
        if backbone == "caformer_b36.sail_in22k_ft_in1k":
            self.reduce0 = nn.Conv2d(768, fuse_ch, kernel_size=1)
            self.reduce1 = nn.Conv2d(128, fuse_ch, kernel_size=1)
            self.reduce2 = nn.Conv2d( 64, fuse_ch, kernel_size=1)
            self.reduce3 = nn.Conv2d( 32, fuse_ch, kernel_size=1)
        elif backbone == "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384":
            self.reduce0 = nn.Conv2d(1536, fuse_ch, kernel_size=1)
            self.reduce1 = nn.Conv2d(128, fuse_ch, kernel_size=1)
            self.reduce2 = nn.Conv2d( 64, fuse_ch, kernel_size=1)
            self.reduce3 = nn.Conv2d( 32, fuse_ch, kernel_size=1)
        else:
            raise NotImplementedError

        self.y_min_max_norm = y_min_max_norm
        if self.y_min_max_norm:
            self.final_activation = nn.Sigmoid()

        self.horizontal_tta = horizontal_tta 

    def _forward(self, batch):
        x= batch # (B, 5, 1024, 32)

        # Encoder
        feats = self.backbone(x)

        feats = feats[::-1] # [torch.Size([4, 768, 32, 1]), torch.Size([4, 512, 64, 2]), torch.Size([4, 256, 128, 4]), torch.Size([4, 128, 256, 8])]

        # Decoder
        feats = self.decoder(feats) 

        target_size = (1259, 300)
        reduced = []
        for f, reduce_conv in zip(feats, [self.reduce0, self.reduce1, 
                                          self.reduce2, self.reduce3]):
            # 1) reduce channels
            x = reduce_conv(f)                     # (B, fuse_ch, Hi, Wi)
            # 2) upsample spatially
            x = F.interpolate(x, size=target_size, 
                              mode='bilinear', align_corners=False)
            reduced.append(x)                      # (B, fuse_ch, 1259, 300)

        # 3) fuse (here: concat along channels → (B,4·fuse_ch,1259,300))
        x = torch.cat(reduced, dim=1)
        # # 4) project to one channel + squeeze
        x = self.seg_head(x).squeeze(1)

        # transpose
        x= torch.transpose(x, -1, -2)

        # y_min_max_norm
        if self.y_min_max_norm:
            x = self.final_activation(x)
    
        return x

    def forward(self, x):
        # apply TTA by flipping the input
        if self.training or not self.horizontal_tta:
            return self._forward(x)

        # apply TTA by flipping the input
        res_1 = self._forward(x)
        res_2 = self._forward(torch.flip(x, dims=[1, 3]))
        res_2 = torch.flip(res_2, dims=[1])
        return (res_1 + res_2) / 2

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

class EnsembleModel(nn.Module):
    def __init__(self, models, ensemble_method="mean"):
        super().__init__()
        self.models = nn.ModuleList(models).eval()
        self.ensemble_method = ensemble_method

    def forward(self, x):
        output = []

        for m in self.models:
            logits = m(x)
            output.append(logits)

        output = torch.stack(output)
        if self.ensemble_method == "median":
            output = torch.quantile(output, 0.5, dim=0)
        elif self.ensemble_method == "mean":
            output = torch.mean(output, dim=0)
        else:
            raise NotImplementedError
        return output