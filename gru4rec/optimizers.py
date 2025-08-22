"""Custom optimizers used by GRU4Rec."""

import torch
from torch.optim import Optimizer


class IndexedAdagradM(Optimizer):
    """Adagrad with momentum supporting sparse indexed updates."""

    def __init__(self, params, lr=0.05, momentum=0.0, eps=1e-6):
        """Configure optimizer hyper-parameters."""
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"] = torch.full_like(
                    p, 0, memory_format=torch.preserve_format
                )
                if momentum > 0:
                    state["mom"] = torch.full_like(
                        p, 0, memory_format=torch.preserve_format
                    )

    def share_memory(self):
        """Share state tensors across processes."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"].share_memory_()
                if group["momentum"] > 0:
                    state["mom"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                clr = group["lr"]
                momentum = group["momentum"]
                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()
                    accs = state["acc"][grad_indices] + grad_values.pow(2)
                    state["acc"].index_copy_(0, grad_indices, accs)
                    accs.add_(group["eps"]).sqrt_().mul_(-1 / clr)
                    if momentum > 0:
                        moma = state["mom"][grad_indices]
                        moma.mul_(momentum).add_(grad_values / accs)
                        state["mom"].index_copy_(0, grad_indices, moma)
                        p.index_add_(0, grad_indices, moma)
                    else:
                        p.index_add_(0, grad_indices, grad_values / accs)
                else:
                    state["acc"].add_(grad.pow(2))
                    accs = state["acc"].add(group["eps"])
                    accs.sqrt_()
                    if momentum > 0:
                        mom = state["mom"]
                        mom.mul_(momentum).addcdiv_(grad, accs, value=-clr)
                        p.add_(mom)
                    else:
                        p.addcdiv_(grad, accs, value=-clr)
        return loss
