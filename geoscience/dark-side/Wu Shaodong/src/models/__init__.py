from .mmseg_model import MMSegModel
from .smp_model import SMPUnetModel

from mmengine.config import ConfigDict

def build_model(cfg: ConfigDict):
    if cfg.type == "MMSegModel":
        model = MMSegModel(cfg)
    elif cfg.type == "SMPUnetModel":
        model = SMPUnetModel(cfg)
    else:
        raise NotImplementedError("Only MMSegModel and SMPUnetModel supported now")
    return model

__all__ = [
    "MMSegModel", "SMPUnetModel", "build_model",
]