

import torch
import numpy as np


class CS336Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: np.memmap,
        ctx: int,
        seed: int = 0,
    ):
        super().__init__()
        self.data = data
        self.ctx = ctx
        self.length = len(data) - ctx
        # Dedicated RNG to avoid external interference; default seed is fixed.
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, index):
        # Ignore incoming index and pick a random start to avoid full-data shuffle overhead.
        start = self.rng.integers(0, self.length)
        x = self.data[start: start + self.ctx].astype(np.int64)
        y = self.data[start + 1: start + self.ctx + 1].astype(np.int64)
        return x, y

    def __len__(self):
        return self.length
