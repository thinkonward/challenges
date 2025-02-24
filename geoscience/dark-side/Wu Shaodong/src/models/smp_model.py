from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmseg.models.utils import resize
from mmengine.config import Config
from mmengine.config import ConfigDict
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init


class SMPUnetModel(nn.Module):
    # 使用smp的head则需要将head的activation改为identity，然后再使用nn.CronssEntropyLoss
    def __init__(self,
                 cfg: ConfigDict) -> None:
        super().__init__()

        encoder_cfg             = cfg.encoder
        decoder_cfg             = cfg.decoder
        self.pretrained_path    = cfg.pretrained_path

        self._create_encoder(encoder_cfg)
        self._create_decoder(decoder_cfg)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )
    
    def _create_encoder(self, encoder_cfg: ConfigDict):
        # timm-xxx
        # tu-xxx
        self.encoder = get_encoder(name         = encoder_cfg.encoder_name,
                                   in_channels  = encoder_cfg.in_channels,
                                   depth        = encoder_cfg.encoder_depth,
                                   weights      = encoder_cfg.encoder_weights,
        )

    def _create_decoder(self, decoder_cfg: ConfigDict):
        self.decoder = UnetDecoder(encoder_channels    = self.encoder.out_channels,
                                   decoder_channels    = decoder_cfg.decoder_channels,
                                   n_blocks            = self.encoder._depth,
                                   use_batchnorm       = decoder_cfg.decoder_use_batchnorm,
                                   center              = False,
                                   attention_type      = decoder_cfg.decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(in_channels   = decoder_cfg.decoder_channels[-1],
                                                  out_channels  = decoder_cfg.num_classes,
                                                  activation    = "identity",
                                                  kernel_size   = 3,
        )
        self.align_corners = decoder_cfg.align_corners

    def _preprocess(self, inputs: torch.Tensor):
        _, _, h1, w1 = inputs.shape
        h2, w2 = h1 % 32, w1 % 32
        padding_w = 32 - w2
        padding_h = 32 - h2
        inputs = F.pad(inputs, (0, padding_w, 0, padding_h), mode='constant', value=0)
        return inputs

    def _postprocess(self, inputs: torch.Tensor, ori_shape):
        inputs = inputs[:, :, :ori_shape[0], :ori_shape[1]]
        return inputs.contiguous()

    def forward(self, inputs: Tensor) -> List[Tensor]:
        ori_shape = inputs.shape[2:]
        inputs = self._preprocess(inputs)
        self.check_input_shape(inputs)
        features = self.encoder(inputs)
        decoder_output = self.decoder(*features)

        seg_logits = self.segmentation_head(decoder_output)
        seg_logits = self._postprocess(seg_logits, ori_shape)
        # seg_logits = resize(input=seg_logits,
        #                     size=ori_shape,
        #                     mode='bilinear',
        #                     align_corners=self.align_corners)
        return {
                "seg_logits": seg_logits
            }

if __name__ == "__main__":
    # 目前timm中只有efficientnet等基本的CNN模型支持多层特征输出类，
    # 其他的模型比如VIT、Convext等只能创建好已有的模型类，自己改forward(不同模型得定制)
    model_cfg = dict(
        model=dict(
            encoder=dict(
                encoder_name    = 'tu-tf_efficientnetv2_s',
                in_channels     = 3,
                encoder_depth   = 5,
                encoder_weights = None,
            ),
            decoder=dict(
                decoder_name            = 'UnetDecoder',
                decoder_channels        = (256, 128, 64, 32, 16),
                decoder_use_batchnorm   = False,
                decoder_attention_type  = None,
                num_classes             = 1,
                align_corners           = False
            ),
            pretrained_path     = None,
        )
    )
    
    cfg = Config(model_cfg)
    
    smp_model = SMPUnetModel(cfg=cfg).to("cuda:0")
    # Unet输入需要是32的整数倍
    input = torch.rand(size=(8, 3, 256, 256), dtype=torch.float32, device="cuda:0")
    print(input.shape)
    output = smp_model(input)
    print(output.shape)