import torch
from torch import nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp

import logging
logging.getLogger('timm').setLevel(logging.WARNING)

class SMPModel(nn.Module):
    def __init__(self, name='resnet50', pretrained=True, in_channels=1, num_classes=1):
        super(SMPModel, self).__init__()
        
        self.segmentor = smp.UnetPlusPlus(encoder_name=f'tu-{name}', in_channels=in_channels, classes=num_classes)
        
    def forward(self, inp, crop=True):
        inp = torch.nan_to_num(inp, 0, 0, 0)
        
        bs, c, h, w = inp.shape
        
        masks = self.segmentor(inp)
        
        if crop:
            masks = masks[:, :, :300*1, :1259]
        
        masks = masks.sigmoid()
        #masks = masks.tanh()
        
        masks = torch.nan_to_num(masks, 0, 0, 0)
        
        return masks, None