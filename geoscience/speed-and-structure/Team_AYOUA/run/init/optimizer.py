from omegaconf import DictConfig
from torch.optim import SGD, Adam, AdamW
from torch_optimizer import AdaBelief, RAdam

from src.optimizers.muon import MuonWithAuxAdam

def init_optimizer_from_config(cfg, model, return_cls=True):
    """
    Initialize an optimizer with configurable learning rates for different model components.

    Args:
        cfg (DictConfig): Configuration containing optimizer parameters.
        model (nn.Module): Model with components (e.g., encoder, decoder, heads).
        return_cls (bool): If True, returns optimizer class and arguments; otherwise, returns optimizer instance.

    Returns:
        Optimizer or tuple: The optimizer instance or (class, kwargs).
    """
   
    optimizers = {
        "adam": Adam,
        "adamw": AdamW,
        "sgd": SGD,
        "radam": RAdam,
        "adabelief": AdaBelief,
        "muon": MuonWithAuxAdam,
        
    }

    if cfg.type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {cfg.type}")

    opt_cls = optimizers[cfg.type]
    
    # Common optimizer parameters
    kwargs = {
        "lr": cfg.lr,
        "weight_decay": cfg.get("weight_decay", 0),
    }
    if cfg.type in ["adam", "adamw", "radam", "adabelief"]:
        kwargs.update({
            "betas": (cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)),
            "eps": cfg.get("eps", 1e-8),
            "amsgrad": cfg.get("amsgrad", False),
        })
    elif cfg.type == "sgd":
        kwargs.update({
            "momentum": cfg.get("momentum", 0.9),
        })
    elif cfg.type == "muon":
        kwargs.update({
            "momentum": cfg.get("momentum", 0.95),
            "betas": (cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
            
        })

    # Define keys for different model components and their corresponding learning rates
    component_keys = {
        "encoder": ["encoder", "enc"],
        "decoder": ["decoder", "dec", "dec1", "dec2", "dec3","dec4","dec5","dec6"],
        "head": ["segmentation_head", "classification_head", "head","final_conv"],
    }

    param_groups = []
    for component, keys in component_keys.items():
        for key in keys:
            attr = getattr(model, key, None)
            if attr is not None: 
                param_groups.append({
                    "params": list(getattr(model, key).parameters()),
                    "lr": cfg.get(f"{component}_lr", cfg.lr),  # Use specific LR if available, else default
                })
    
    # Include remaining parameters not in predefined components
    other_params = [
        param for name, param in model.named_parameters()
        if not any(any(key in name for key in keys) for keys in component_keys.values())
    ]
    if other_params:
        param_groups.append({"params": other_params, "lr": cfg.get("other_lr", cfg.lr)})
    if cfg.type == "muon":
        # MuonWithAuxAdam requires the full model to inspect parameter names
        return (opt_cls, {"model": model, **kwargs}) if return_cls else opt_cls(model=model, **kwargs)
    else:
        kwargs["params"] = param_groups if param_groups else [{"params": model.parameters(), "lr": cfg.lr}]
        return (opt_cls, kwargs) if return_cls else opt_cls(**kwargs)
    