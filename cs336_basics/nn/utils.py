
import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np
import numpy.typing as npt
import os
import typing


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


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
):
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    dataset_length = len(dataset)
    # Random starting indices for each sequence in the batch
    start_indices = np.random.randint(
        0, dataset_length - context_length, size=batch_size)
    x_batch = np.stack(
        [dataset[i:i + context_length] for i in start_indices])
    y_batch = np.stack(
        [dataset[i + 1:i + context_length + 1] for i in start_indices])
    x_batch_tensor = torch.tensor(x_batch, dtype=torch.long, device=device)
    y_batch_tensor = torch.tensor(y_batch, dtype=torch.long, device=device)
    return x_batch_tensor, y_batch_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    '''
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The current training iteration.
        out (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
            The file path or file-like object to save the checkpoint to.
    '''
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> int:
    '''
    Args:
        model (torch.nn.Module): The model to load the state dict into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state dict into.
        checkpoint_path (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
            The file path or file-like object to load the checkpoint from.

    Returns:
        int: The iteration number stored in the checkpoint.
    '''
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration
