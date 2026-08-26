from __future__ import annotations

import math

import numpy as np
import torch


def make_kdv_dataset(N, S, T, L=60.0, dt_out=0.08, c_range=(0.4, 2.2), n_images=4, seed=0):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, L, S, endpoint=False)
    c = rng.uniform(*c_range, size=N).astype(np.float32)
    x0 = rng.uniform(0, L, size=N).astype(np.float32)
    times = np.arange(T + 1) * dt_out  # t=0..T

    def soliton(xx, t, cc, xx0):
        u = np.zeros_like(xx)
        for n in range(-n_images, n_images + 1):
            arg = (np.sqrt(cc) / 2.0) * (xx - cc * t - (xx0 + n * L))
            u += (cc / 2.0) * (1.0 / np.cosh(np.clip(arg, -40, 40))) ** 2
        return u

    data = np.zeros((N, T + 1, S), dtype=np.float32)
    for i in range(N):
        for ti, t in enumerate(times):
            data[i, ti] = soliton(x, t, c[i], x0[i])
    return data, {"L": L, "dt_out": dt_out, "c": c, "x0": x0}


def _burgers_step(u_hat, k, nu, dt):
    dealias = (torch.abs(k) < (2 / 3) * k.abs().max()).to(u_hat.dtype)

    def nl(uh):
        u = torch.fft.ifft(uh, dim=-1).real
        flux_hat = torch.fft.fft((0.5 * u ** 2).to(torch.complex64), dim=-1)
        return -1j * k * flux_hat * dealias

    decay = torch.exp(-nu.view(-1, 1) * (k.view(1, -1) ** 2) * dt / 2)
    u_hat = u_hat * decay
    k1 = nl(u_hat)
    k2 = nl(u_hat + dt / 2 * k1)
    k3 = nl(u_hat + dt / 2 * k2)
    k4 = nl(u_hat + dt * k3)
    u_hat = u_hat + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    u_hat = u_hat * decay
    return u_hat


def make_burgers_dataset(N, S, T, L=20.0, nu=0.02, dt=5e-4, substeps=300,
                         n_modes=6, flow_mag_range=(0.8, 1.8), osc_scale=0.4,
                         seed=0, device="cpu", batch_size=200):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, L, S, endpoint=False)
    k = 2 * math.pi * torch.fft.fftfreq(S, d=L / S).to(device)

    mag = rng.uniform(*flow_mag_range, size=N)
    sign = rng.choice([-1, 1], size=N)
    mean_flow = (mag * sign).astype(np.float32)
    u0_all = np.zeros((N, S), dtype=np.float32)
    for i in range(N):
        u = np.full(S, mean_flow[i])
        for m in range(1, n_modes + 1):
            amp = osc_scale * rng.uniform(-1, 1) / m
            phase = rng.uniform(0, 2 * np.pi)
            u += amp * np.sin(m * 2 * np.pi * x / L + phase)
        u0_all[i] = u

    data = np.zeros((N, T + 1, S), dtype=np.float32)
    data[:, 0] = u0_all
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        u0_b = torch.from_numpy(u0_all[start:end]).float().to(device)
        u_hat = torch.fft.fft(u0_b.to(torch.complex64), dim=-1)
        nu_b = torch.full((end - start,), nu, device=device)
        for t in range(1, T + 1):
            for _ in range(substeps):
                u_hat = _burgers_step(u_hat, k, nu_b, dt)
            data[start:end, t] = torch.fft.ifft(u_hat, dim=-1).real.cpu().numpy()
    return data, {"L": L, "nu": nu, "dt_out": dt * substeps, "mean_flow": mean_flow}