from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class Traj1dDS(Dataset):
    def __init__(self, ic, target):
        self.ic, self.target = ic, target

    def __len__(self):
        return len(self.ic)

    def __getitem__(self, i):
        return (torch.from_numpy(self.ic[i]).float().unsqueeze(-1),
                torch.from_numpy(self.target[i]).float())


class Phase1dDS(Dataset):
    def __init__(self, ic, phase):
        self.ic, self.phase = ic, phase

    def __len__(self):
        return len(self.ic)

    def __getitem__(self, i):
        return (torch.from_numpy(self.ic[i]).float().unsqueeze(0),
                torch.from_numpy(self.phase[i]).float())


def relative_l2_loss(pred, true):
    B = pred.shape[0]
    diff = (pred - true).reshape(B, -1)
    norm_true = true.reshape(B, -1)
    return (diff.norm(dim=1) / (norm_true.norm(dim=1) + 1e-8)).mean()