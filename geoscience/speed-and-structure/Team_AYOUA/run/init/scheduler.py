import math
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
    LRScheduler,
)
from torch.optim.optimizer import Optimizer
from transformers import get_cosine_schedule_with_warmup
from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    CosineParamScheduler,
)
from fvcore.common.param_scheduler import CompositeParamScheduler, ConstantParamScheduler, CosineParamScheduler
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineLambda:
    def __init__(
        self,
        warmup_steps: int,
        cycle_steps: int,
        decay_scale: float,
        exponential_warmup: bool = False,
    ):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.decay_scale = decay_scale
        self.exponential_warmup = exponential_warmup

    def __call__(self, epoch: int):
        if epoch < self.warmup_steps:
            if self.exponential_warmup:
                return self.decay_scale * pow(
                    self.decay_scale, -epoch / self.warmup_steps
                )
            ratio = epoch / self.warmup_steps
        else:
            ratio = (
                1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.cycle_steps)
            ) / 2
        return self.decay_scale + (1 - self.decay_scale) * ratio
class ConstantThenCosineLambda:
    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        decay_scale: float = 1e-3,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_scale = decay_scale

    def __call__(self, step: int):
        if step < self.warmup_steps:
            return 1.0  # Constant LR during warmup
        else:
            progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.decay_scale + (1.0 - self.decay_scale) * cosine_decay

class FVCoreSchedulerWrapper:
    def __init__(self, scheduler, total_steps):
        self.scheduler = scheduler
        self.total_steps = total_steps

    def __call__(self, step):
        t = step / self.total_steps
        t = min(t, 1.0)  # Clip to [0, 1]
        return self.scheduler(t)

def init_scheduler_from_config(cfg: DictConfig, optimizer: Optimizer) -> Optional[LRScheduler]:
    if cfg.type is None:
        return None
    elif cfg.type == "step_lr":
        return StepLR(optimizer, step_size=cfg.lr_decay_steps, gamma=cfg.lr_decay_rate)
    elif cfg.type == "exponential_lr":
        return ExponentialLR(optimizer, gamma=cfg.lr_decay_rate)
    elif cfg.type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.lr_decay_rate,
            patience=cfg.patience,
            verbose=True,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == "cosine_annealing_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min
        )
    elif cfg.type == "cosine_annealing":
        return CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)
    elif cfg.type == "cosine_warmup":
        warmup_steps = cfg.max_epochs * cfg.warmup_steps_ratio
        cycle_steps = cfg.max_epochs - warmup_steps
        lr_lambda = WarmupCosineLambda(warmup_steps, cycle_steps, cfg.lr_decay_scale)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    elif cfg.type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.num_steps_per_epoch * cfg.max_epochs * cfg.warmup_steps_ratio
            ),
            num_training_steps=int(cfg.num_steps_per_epoch * cfg.max_epochs),
            num_cycles=cfg.num_cycles,
        )
        return scheduler
    elif cfg.type == "constant_cosine":
        warmup_steps = int(cfg.max_epochs * cfg.num_steps_per_epoch * cfg.warmup_steps_ratio)
        total_steps = int(cfg.max_epochs * cfg.num_steps_per_epoch)
        print("total_steps =",total_steps)
        print("warmup_steps =",warmup_steps)
        lr_lambda = ConstantThenCosineLambda(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_scale=cfg.lr_decay_scale,
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}")
