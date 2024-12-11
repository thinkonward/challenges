import torch
from torch import nn
import torch.nn.functional as F
import timm

class TimmClsModel(nn.Module):
    def __init__(self, name='resnet50', pretrained=True, in_chans=3, num_classes=1, global_pool=''):
        super().__init__()

        self.backbone = timm.create_model(model_name=name, pretrained=pretrained, in_chans=in_chans, 
        num_classes=0, global_pool=global_pool)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, inp):
        pass