# From:
# 1. https://github.com/KellerJordan/Muon
# 2. https://github.com/MoonshotAI/Moonlight

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    # 5 steps
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonWithAuxAdam(Optimizer):
    """Muon with Adam for ViT."""
    def __init__(
        self,
        model,
        # shared params
        lr=1e-3,
        weight_decay=0.01,
        # muon params
        momentum=0.95,
        # adam params
        betas=(0.9, 0.95),
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        super(MuonWithAuxAdam, self).__init__(model.parameters(), defaults)
        param2name = {}
        for name, param in model.named_parameters():
            param2name[param] = name
        self.param2name = param2name

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # dispatch
                # if (
                #     self.param2name[p].startswith("head.")
                #     or self.param2name[p].startswith("_orig_mod.module.head.")
                #     or self.param2name[p].startswith("_orig_mod.head.")
                # ):
                #     self.step_adam(group, p)
                # elif p.ndim == 1:
                if p.ndim == 1:
                    # bias, norm, etc.
                    self.step_adam(group, p)
                elif p.ndim == 2:
                    # linear weight
                    self.step_muon(group, p)
                elif p.ndim == 3:
                    # cls_token, pos_embed in ViT
                    self.step_adam(group, p)
                elif p.ndim == 4:
                    # weight of patch embedding
                    # it's a Linear, but using Adam works better
                    # self.step_muon(group, p)
                    self.step_adam(group, p)
                else:
                    raise ValueError(f"Unsupported parameter dimension: {p.ndim}")

        return loss

    def step_muon(self, group, p):
        state = self.state[p]
        grad = p.grad

        # State initialization
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)

        lr = group["lr"]
        weight_decay = group["weight_decay"]
        nesterov = True

        momentum = group["momentum"]
        momentum_buffer = state["momentum_buffer"]
        momentum_buffer.lerp_(grad, 1 - momentum)

        update = grad.lerp_(momentum_buffer, momentum) if nesterov else momentum_buffer
        if p.ndim == 4:
            update = update.view(len(update), -1)
            update = zeropower_via_newtonschulz5(update)
            update = update.view(p.shape)
        else:
            update = zeropower_via_newtonschulz5(update)
        update.mul_(0.2 * (max(grad.size(-2), grad.size(-1)) ** 0.5))

        p.mul_(1 - lr * weight_decay)
        p.add_(update, alpha=-lr)

    def step_adam(self, group, p):
        state = self.state[p]
        grad = p.grad

        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["step"] += 1

        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        step = state["step"]

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        # update
        p.mul_(1 - lr * weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = bias_correction2**0.5

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        p.addcdiv_(exp_avg, denom, value=-step_size)
