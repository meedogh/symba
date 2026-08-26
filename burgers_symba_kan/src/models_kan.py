from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkovTransitionBasis(nn.Module):
    def __init__(self, order=4, grid=16, self_loop=0.5):
        super().__init__()
        G = int(grid)
        G = max(G, 2 * order + 2)          # grid must resolve 'order' steps of the walk
        self.G = G
        self.order = int(order)

        P = torch.zeros(G, G)
        p = float(self_loop)
        for i in range(G):
            P[i, i] = p
            P[i, (i + 1) % G] = 1.0 - p
        assert torch.allclose(P.sum(dim=1), torch.ones(G), atol=1e-6)

        rows = []
        M = torch.eye(G)
        for k in range(0, order + 1):      # P**0 = I gives the constant basis
            diag = M.diag()
            off = M.diagonal(offset=1).contiguous()
            off = torch.cat([off, torch.zeros(G - off.numel())])   # pad to length G
            rows.append(torch.stack([diag, off], dim=0))           # (2, G)
            if k < order:
                M = M @ P
        basis = torch.cat(rows, dim=0)                             # ((order+1)*2, G)
        self.register_buffer("basis", basis)                       # non-learnable kernel

    @property
    def nb(self):
        return self.basis.shape[0]

    def forward(self, x):
        """x: (...) in [-1, 1] -> ((order+1)*2, ...) Markov basis values."""
        G = self.G
        s = (x.clamp(-1.0, 1.0) + 1.0) / 2.0 * (G - 1)             # continuous state index
        lo = torch.floor(s).long().clamp(0, G - 1)
        hi = torch.ceil(s).long().clamp(0, G - 1)
        frac = (s - lo.float()).unsqueeze(0)                       # (1, ...)
        b_lo = self.basis[:, lo]                                   # (nb, ...)
        b_hi = self.basis[:, hi]
        return b_lo * (1.0 - frac) + b_hi * frac