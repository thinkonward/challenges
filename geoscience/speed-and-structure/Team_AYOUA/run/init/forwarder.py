from typing import Dict, Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from torch import Tensor
from torch_ema import ExponentialMovingAverage
import itertools
from src.losses.losses import *

def get_transform_pair(hflip):
    def pre(img):
        if hflip:
            img=torch.flip(img, dims=(1, 3))
        return img

    def post(pred):
        
        if hflip:
            pred =torch.flip(pred, dims=(2,))
        return pred

    return pre, post


TTA_COMBINATIONS = [
    get_transform_pair(*args) for args in itertools.product([False, True])
]


class Forwarder(nn.Module):
    def __init__(self, cfg: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.ema = None
        self.cfg = cfg
        
        self.hybrid_loss = HybridLoss()
        self.use_amp = cfg.use_amp
        
    def hybrid_loss_func(
        self,
        logits,
        labels,
    ):
        if len(logits) == 0:
            return 0
        loss = self.hybrid_loss(logits, labels)
        return loss
    
    def forward(
        self, batch: Dict[str, Tensor], phase: str, epoch=None
    ) -> Tuple[Tensor, Tensor]:
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.997)
        inputs = batch["image"]
        if phase != "test":
            labels = batch["label"]          
        else:
            labels = None
           
        if phase == "train":            
            with torch.set_grad_enabled(True):
                logits = self.model(inputs)
        else:
            if phase == "test":
                with self.ema.average_parameters():
                    logits = self.model(inputs)
            elif phase == "val":
                logits = self.model(inputs)
        
        if phase!="test" :
            per_element_l1_loss = F.l1_loss(logits, labels, reduction='none')
            per_sample_l1_loss = per_element_l1_loss.mean(dim=list(range(1, per_element_l1_loss.ndim)))
            l1_loss = per_sample_l1_loss.mean()
            loss = (l1_loss*self.cfg.loss.get("l1_weight",1.0)+ \
            self.hybrid_loss_func(logits, labels)*self.cfg.loss.get("hybrid_weight",0) ) 
                                  
            return (
                logits,
                loss,
            )
        else: return (
                logits,
                torch.tensor(-1),
            )
    
    @torch.no_grad()
    def predict(self, batch: Dict[str,Tensor],phase="test"):
        out,_ = self.forward(batch,phase)
        pred = out.unsqueeze(1)
        return pred
    
    @torch.no_grad()
    def predict_tta(self, batch: Dict[str,Tensor],phase="test") -> Tensor:
        
        img = batch["image"]
        pred = torch.zeros(
            img.size(0), 1, 300, 1259, dtype=torch.float32, device=img.device
        )
        for i,(pre, post) in enumerate(TTA_COMBINATIONS):
            batch["image"]=pre(img)
            out,_ = self.forward(batch,phase)
            out = out.unsqueeze(1)
            out = post(out)
            pred += out
        pred /= len(TTA_COMBINATIONS)
        return pred
    