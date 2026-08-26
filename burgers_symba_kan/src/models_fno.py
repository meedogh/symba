from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.w = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, S = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_ch, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m] = torch.einsum("bix,iox->box", x_ft[:, :, :m], self.w[:, :, :m])
        return torch.fft.irfft(out_ft, n=S, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, modes, width, in_channels, out_channels, n_layers=4):
        super().__init__()
        self.fc0 = nn.Linear(in_channels + 1, width)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.w_layers = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (B, S, in_channels)
        B, S, _ = x.shape
        grid = torch.linspace(0, 1, S, device=x.device).view(1, S, 1).repeat(B, 1, 1)
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        for spec, w in zip(self.spectral, self.w_layers):
            x = F.gelu(spec(x) + w(x))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).permute(0, 2, 1)  # (B, out_channels, S)


class PhaseNet1d(nn.Module):
    def __init__(self, S, T_out, base_ch=16):
        super().__init__()
        self.T_out = T_out
        self.enc = nn.Sequential(
            nn.Conv1d(1, base_ch, 5, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(base_ch, base_ch, 5, stride=2, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(base_ch, base_ch * 2, 5, stride=2, padding=2, padding_mode="circular"), nn.GELU(),
        )
        out_len = max(8, S // 16)
        self.pool = nn.AdaptiveAvgPool1d(out_len)
        self.head = nn.Sequential(
            nn.Linear(base_ch * 2 * out_len, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, T_out),
        )

    def forward(self, u0):
        z = self.pool(self.enc(u0)).flatten(1)
        return self.head(z)