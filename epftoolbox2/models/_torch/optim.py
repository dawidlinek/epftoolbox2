"""Adam for the trainer: one learning rate per member of a stack and an active mask, in the research trainer's arithmetic.

Adam is elementwise, so one Adam over stacked parameters (members on dimension 0) is one Adam per member, provided each member
has its own learning rate and a stopped member doesn't move: `StackedAdam.step(active)` leaves the parameters of an inactive
member as they are, so its result doesn't depend on how long the others train. `torch.optim.Adam` can hold neither. The update
is stk.py's, operation for operation (`m = b1 m + (1 - b1) g`, `v = b2 v + (1 - b2) g^2`, `p - lr (m / c1) / (sqrt(v / c2) +
eps)`), and the loop's `Adam` is the same update on one member's parameters.
"""

from typing import List, Optional, Sequence

import torch


def _moments(param, grad, m, v, b1: float, b2: float, weight_decay: float) -> None:
    if weight_decay:
        grad = grad + weight_decay * param  # Adam's L2 term, as torch.optim.Adam(weight_decay=...)
    m.mul_(b1).add_(grad, alpha=1 - b1)
    v.mul_(b2).addcmul_(grad, grad, value=1 - b2)


def _direction(m, v, c1: float, c2: float, eps: float):
    return (m / c1) / ((v / c2).sqrt() + eps)


class StackedAdam:
    """Adam over stacked parameters (members on dimension 0) with a learning rate per member (`lrs`, one per slot)."""

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        lrs: Sequence[float],
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params: List[torch.Tensor] = list(params)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        first = self.params[0]
        self.slots = int(first.shape[0])
        if len(lrs) != self.slots:
            raise ValueError(f"StackedAdam: {len(lrs)} learning rates for {self.slots} members")
        self.lrs = torch.as_tensor(list(lrs), dtype=first.dtype, device=first.device)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, active: Optional[torch.Tensor] = None) -> None:
        """One step for every member; with `active` (bool, one per slot) an inactive member's parameters don't change."""
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            shape = (self.slots,) + (1,) * (p.dim() - 1)
            _moments(p, p.grad, m, v, self.b1, self.b2, self.weight_decay)
            moved = p - self.lrs.view(shape) * _direction(m, v, c1, c2, self.eps)
            p.copy_(moved if active is None else torch.where(active.view(shape), moved, p))


class Adam:
    """The same update on one member's own (not stacked) parameters, with one learning rate: the loop's optimiser."""

    def __init__(
        self, params: Sequence[torch.Tensor], lr: float, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay=0.0
    ):
        self.params: List[torch.Tensor] = [p for p in params if p.requires_grad]
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.lr = (
            torch.tensor(float(lr), dtype=self.params[0].dtype, device=self.params[0].device) if self.params else None
        )
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            _moments(p, p.grad, m, v, self.b1, self.b2, self.weight_decay)
            p.copy_(p - self.lr * _direction(m, v, c1, c2, self.eps))
