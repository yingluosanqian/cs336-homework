
import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import math


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
    norm_type: int = 2,
):
    all_norm = torch.stack(
        [p.grad.data.detach() for p in parameters if p.grad is not None]).norm(norm_type)
    if all_norm > max_l2_norm:
        clip_coef = max_l2_norm / (all_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
