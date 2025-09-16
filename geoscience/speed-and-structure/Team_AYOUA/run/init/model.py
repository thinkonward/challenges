from logging import getLogger
import importlib
import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = getLogger(__name__)


def init_model_from_config(cfg: DictConfig):
    """
    Initialize a model from a Hydra/OmegaConf config.
    cfg.model.name: name of the model module (without .py)
    cfg.model.params: dict of arguments passed to the model constructor
    """
    
    model = get_model(cfg.name,cfg.params) 
    return model


def get_model(model_name: str, model_params: DictConfig = None) -> nn.Module:
    """
    Factory function to dynamically import and return a model.
    Assumes the model file defines a class `FWIModel`.
    """
    try:
        module_path = f"src.models.{model_name}"
        module = importlib.import_module(module_path)

        model_class = getattr(module, "FWIModel")
        return  model_class(**model_params) if model_params else model_class()

    except ModuleNotFoundError:
        raise ValueError(f"Model file 'src/models/{model_name}.py' not found.")
    except AttributeError:
        raise ValueError(f"'FWIModel' class not found in 'src/models/{model_name}.py'.")
