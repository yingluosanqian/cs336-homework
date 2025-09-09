

import torch
import torch.nn as torchnn
from .nn import utils
from collections.abc import Callable, Iterable
from typing import Optional
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        gradient_clipping: Optional[float] = None,
    ):
        beta_1, beta_2 = betas
        defaults = {
            'lr': lr,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)

        self.gradient_clipping = gradient_clipping

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            alpha = group['lr']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            if self.gradient_clipping is not None:
                utils.gradient_clipping(
                    group['params'], self.gradient_clipping)
            for p in group['params']:
                p: torchnn.Parameter
                if p.grad is None:
                    continue
                grad = p.grad.data
                state: dict = self.state[p]
                # Persistent state
                t = state.get('t', 0) + 1
                m = beta_1 * \
                    state.get('m', torch.zeros_like(p)) + (1 - beta_1) * grad
                v = beta_2 * state.get('v', torch.zeros_like(p)) + \
                    (1 - beta_2) * (grad * grad)
                alpha_t = alpha * \
                    math.sqrt(1 - pow(beta_2, t)) / (1 - math.pow(beta_1, t))
                # Update state
                state['m'] = m
                state['v'] = v
                state['t'] = t

                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= alpha * weight_decay * p.data

        return loss


def lr_cosine_schedule(
    current_iter: int,
    max_learning_rate: float,
    min_learning_rate: float,
    Tw: int,
    Tc: int,
) -> float:
    '''
    Args:
        current_iter: Current iteration (0-indexed).
        max_learning_rate: Maximum learning rate.
        min_learning_rate: Minimum learning rate.
        Tw: Number of warmup iterations.
        Tc: Total number of iterations in the cosine cycle (including warmup).
    Returns:
        Learning rate at the current iteration.
    '''
    if current_iter < Tw:
        return max_learning_rate * current_iter / Tw
    elif Tw <= current_iter <= Tc:
        progress = (current_iter - Tw) / (Tc - Tw)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    else:
        return min_learning_rate
