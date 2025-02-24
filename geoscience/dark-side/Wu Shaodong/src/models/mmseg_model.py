from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from mmseg.registry import MODELS
from mmseg.utils import ConfigType
from mmseg.models.utils import resize
from mmengine.config import Config
from mmengine.config import ConfigDict


class MMSegModel(nn.Module):
    def __init__(self,
                 cfg: ConfigDict) -> None:
        super().__init__()
        backbone_cfg         = cfg.backbone
        decode_head_cfg      = cfg.decode_head
        auxiliary_head_cfg   = cfg.auxiliary_head if "auxiliary_head" in cfg else None
        self.pretrained_path = cfg.pretrained_path
        
        self.backbone = MODELS.build(backbone_cfg)
        self._init_decode_head(decode_head_cfg)
        self._init_auxiliary_head(auxiliary_head_cfg)

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        # loss用不到，直接删除
        del self.decode_head.loss_decode
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)
            # loss用不到，直接删除
            del self.auxiliary_head.loss_decode
    
    def forward(self, inputs: Tensor) -> List[Tensor]:
        ori_shape = inputs.shape[2:]
        x = self.backbone(inputs)
        seg_logits = self.decode_head(x)
        seg_logits = resize(input=seg_logits,
                            size=ori_shape,
                            mode='bilinear',
                            align_corners=self.align_corners)
        
        if hasattr(self, "auxiliary_head"):
            seg_logits_aux = self.auxiliary_head(x)
            seg_logits_aux = resize(input=seg_logits_aux,
                                    size=ori_shape,
                                    mode='bilinear',
                                    align_corners=self.align_corners)

        if hasattr(self, "auxiliary_head") and self.training:
            return {
                "seg_logits_aux": seg_logits_aux, 
                "seg_logits": seg_logits
            }
        else:
            return {
                "seg_logits": seg_logits
            }
    

if __name__ == "__main__":
    model_cfg = dict(
        model=dict(
            name = "MMSegModel",
            backbone=dict(
                type='mmpretrain.ConvNeXt',
                arch='base',
                out_indices=[0, 1, 2, 3],
                drop_path_rate=0.4,
                layer_scale_init_value=1.0,
                gap_before_final_norm=False,
            ),
            decode_head=dict(
                type='UPerHead',
                in_channels=[128, 256, 512, 1024],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
            ),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=384,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
            ),
            pretrained_path     = None,
        )
    )
    
    cfg = Config(model_cfg)
    
    mmseg_model = MMSegModel(cfg=cfg, pretrained_path=None).to("cuda:0")
    input = torch.rand(size=(2, 3, 1258, 1258), dtype=torch.float32, device="cuda:0")
    print(mmseg_model.training)
    print(input.shape)
    output = mmseg_model(input)
    print(output.shape)