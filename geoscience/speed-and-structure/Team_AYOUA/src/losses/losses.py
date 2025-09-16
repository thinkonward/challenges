import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha 
        self.beta = beta   

        self.mae_loss = nn.L1Loss() 
        self.ssim_loss_metric = StructuralSimilarityIndexMeasure(data_range=1.0) 

    def forward(self, prediction, target):
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        min_val = 1.5
        max_val = 4.5
        normalized_prediction = (prediction - min_val) / (max_val - min_val)
        normalized_target = (target - min_val) / (max_val - min_val)
        normalized_prediction = torch.clamp(normalized_prediction, 0.0, 1.0)
        normalized_target = torch.clamp(normalized_target, 0.0, 1.0)

        mae = self.mae_loss(prediction, target)
        ssim = 1.0 - self.ssim_loss_metric(normalized_prediction, normalized_target)
        total_loss = self.alpha * mae + self.beta * ssim
        return total_loss