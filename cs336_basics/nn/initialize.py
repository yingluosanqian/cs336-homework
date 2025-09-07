
import torch
from torch import Tensor, LongTensor


def init_linear_weights(shape: tuple, d_in: int, d_out: int, *args, **kwargs) -> Tensor:
    sigma = (2 / (d_in + d_out))**0.5
    empty_tensor = torch.empty(shape, *args, **kwargs)
    return torch.nn.init.trunc_normal_(empty_tensor, mean=0, std=sigma, a=-3*sigma, b=3*sigma)


def init_embedding_weights(shape: tuple, *args, **kwargs) -> Tensor:
    empty_tensor = torch.empty(shape, *args, **kwargs)
    return torch.nn.init.trunc_normal_(empty_tensor, mean=0, std=1, a=-3, b=3)


def init_rmsnorm_weights(shape: tuple, *args, **kwargs) -> Tensor:
    return torch.ones(shape, *args, **kwargs)
