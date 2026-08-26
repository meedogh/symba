from __future__ import annotations

import math

import numpy as np
import torch

BURGERS_L = 20.0
BURGERS_NU = 0.02
BURGERS_DT_OUT = 5e-4 * 300 
DEALIAS_FRAC = 2.0 / 3.0

def fourier_translate_1d(field, c):
    B, S = field.shape
    f_hat = torch.fft.fft(field.to(torch.complex64), dim=-1)
    k = torch.fft.fftfreq(S, d=1.0 / S).to(field.device).view(1, S)
    phase = -2 * math.pi * (k * c.view(B, 1)) / S
    shift_factor = torch.exp(1j * phase)
    if S % 2 == 0:
        nyq = S // 2
        is_int = torch.isclose(c, torch.round(c), atol=1e-4)
        keep = is_int.view(B, 1).to(shift_factor.dtype)
        shift_factor = shift_factor.clone()
        shift_factor[:, nyq:nyq + 1] *= keep
    out = torch.fft.ifft(f_hat * shift_factor, dim=-1)
    imag_res = out.imag.abs().max().item()
    if imag_res > 1e-2:
        print(f"WARNING: projector imaginary residual {imag_res:.2e} -- check input.")
    return out.real


def estimate_shift_1d(f0, f1):
    B, S = f0.shape
    F0 = torch.fft.fft(f0.to(torch.complex64), dim=-1)
    F1 = torch.fft.fft(f1.to(torch.complex64), dim=-1)
    R = F1 * torch.conj(F0)
    R = R / (R.abs() + 1e-8)
    r = torch.fft.ifft(R, dim=-1).real
    peak_val, peak_idx = r.max(dim=1)
    idx_m1 = (peak_idx - 1) % S
    idx_p1 = (peak_idx + 1) % S
    r_m1 = r.gather(1, idx_m1.unsqueeze(1)).squeeze(1)
    r_p1 = r.gather(1, idx_p1.unsqueeze(1)).squeeze(1)
    denom = r_m1 - 2.0 * peak_val + r_p1
    frac = torch.where(denom.abs() > 1e-8, 0.5 * (r_m1 - r_p1) / denom,
                       torch.zeros_like(denom))
    frac = frac.clamp(-0.5, 0.5)
    d = peak_idx.float() + frac
    d = torch.where(d > S // 2, d - S, d)
    return d, peak_val

def bandlimit(u, frac=DEALIAS_FRAC):
    S = u.shape[-1]
    keep = int(frac * S / 2)
    U = torch.fft.rfft(u, dim=-1)
    U[:, keep + 1:] = 0
    return torch.fft.irfft(U, n=S, dim=-1)


def fourier_shift_exact(u, c_px):
    return fourier_translate_1d(bandlimit(u), c_px)


def fourier_resample(u, lam, upsample=16):
    B, S = u.shape
    dev = u.device
    if not torch.is_tensor(lam):
        lam = torch.full((B,), float(lam), device=dev)
    lam = lam.float().to(dev)
    Uh = torch.fft.rfft(u, dim=-1)
    P = upsample * S
    pad = torch.zeros(B, P // 2 + 1, dtype=torch.complex64, device=dev)
    pad[:, :Uh.shape[-1]] = Uh
    fine = torch.fft.irfft(pad, n=P, dim=-1) * (P / S)
    pos = lam.view(B, 1) * torch.arange(S, device=dev, dtype=torch.float32).view(1, -1) % S
    fidx = pos * upsample
    i0 = torch.floor(fidx).long() % P
    frac = fidx - torch.floor(fidx)
    out = fine.gather(1, i0) * (1 - frac) + fine.gather(1, (i0 + 1) % P) * frac
    return out


def _spectral_omega(S, dx):
    return 2 * math.pi * torch.fft.rfftfreq(S, d=dx).to(torch.float64)

def colehopf_to_logphi(u, nu, dx):
    S = u.shape[-1]
    L = S * dx
    om = _spectral_omega(S, dx).to(u.device)
    um = u - u.mean(dim=-1, keepdim=True)
    U = torch.fft.rfft(um.to(torch.float64), dim=-1)
    U[:, -1] = 0 
    inv = torch.zeros_like(om, dtype=torch.complex128)
    nz = om > 1e-12
    inv[nz] = 1.0 / (1j * om[nz])
    I = torch.fft.irfft(U * inv.view(1, -1), n=S, dim=-1)
    lp = (-I / L).to(torch.float32)
    lp = lp - lp.mean(dim=-1, keepdim=True)
    return lp, u.mean(dim=-1)


def colehopf_to_u(lp, nu, dx, mean_u):
    S = lp.shape[-1]
    L = S * dx
    om = _spectral_omega(S, dx).to(lp.device)
    lpraw = (lp.to(torch.float64)) * (L / (2 * nu))
    dlpraw = torch.fft.irfft(1j * om.view(1, -1) * torch.fft.rfft(lpraw, dim=-1), n=S, dim=-1)
    return (-2 * nu * dlpraw).to(torch.float32) + mean_u.reshape(-1, 1)


def reflect_u(u):
    return -torch.roll(torch.flip(u, dims=[-1]), 1, dims=-1)


def reflect_lp(lp):
    return torch.roll(torch.flip(lp, dims=[-1]), 1, dims=-1)