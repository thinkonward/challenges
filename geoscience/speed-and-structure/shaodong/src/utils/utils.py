import torch
import torch.nn as nn
from copy import deepcopy


@torch.no_grad()
def get_total_norm(model, norm_type=2):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(torch.linalg.vector_norm(p.grad, norm_type))
    total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
    return total_norm


def to_gpu(data):
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)
    elif isinstance(data, dict):
        return {key: to_gpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_gpu(value) for value in data]
    elif isinstance(data, str):
        return data
    elif isinstance(data, int):
        return data
    elif data is None:
        return data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
