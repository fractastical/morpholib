#!/usr/bin/env python3
"""
Botryllus schlosseri — Turing wavenumber at the 39 µm detail scale (slot 3136).

Holocene (extant), 0 Ma. Chronoseries 1/1: no fossil predecessor or successor.
Scale: 39 µm = 3.872e-5 m (log10 = -4.412), tier 4_cellular_10um, detail variant.

The law (Turing 1952; the k_c identity used by Murray and by Gierer & Meinhardt):

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c.

Kinetics are Schnakenberg 1979, not the sibling slot 0167 parameter set:

    f = γ (a - u + u² v),   g = γ (b - u² v),
    u0 = a + b,             v0 = b / (a + b)²,
    a = 0.097, b = 1.246, γ = 1.37, seed = 3136.

The detail window is seven wavelengths: L = 7 × 38.72 µm = 271.04 µm.
Coarser star-system spacing (millimetres) is a different scale. Finer pigment
granules (a few micrometres) lie at k >> k_c, where σ(k) < 0.

Boundary condition: periodic square. The colony photograph (Parent 1991,
Banyuls-sur-Mer, public domain) is the visual specimen. It is not a 39 µm
micrograph; 38.72 µm is the assigned characteristic length.

    python3 models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "botryllus_schlosseri_turing_wavelength_s0"
SPECIMEN = DATASET / "Botryllus_schlosseri_Parent1991_Banyuls.jpg"
REFERENCE = DATASET / "botryllus_schlosseri_turing_wavelength_s0_reference.json"
METRICS_JSON = DATASET / "botryllus_schlosseri_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "botryllus_schlosseri_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "botryllus_schlosseri_turing_wavelength_s0_growth.gif"

# Slot-3136 constants. Do not replace with the sibling slot's values.
A_PROD = 0.097
B_PROD = 1.246
GAMMA = 1.37
LAMBDA_M = 3.872e-5  # 38.72 µm, the 39 µm campaign scale
N_WAVES = 7
DOMAIN_M = N_WAVES * LAMBDA_M  # 271.04 µm
SEED = 3136
RENDER_FACTOR = 1.55  # D_v / D_u = factor × r_critical; supercritical enough to see pigment
N_GRID = 96
N_STEPS = 2400
DT = 0.04
GIF_SIZE = (640, 480)
N_FRAMES = 12


def steady_state(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float]:
    """Schnakenberg homogeneous steady state. Eq. steady_state in the reference JSON."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return float(u0), float(v0)


def jacobian(a: float = A_PROD, b: float = B_PROD, gamma: float = GAMMA) -> tuple[float, float, float, float]:
    """Linearization of γ(a - u + u²v, b - u²v) at the homogeneous steady state."""
    u0, v0 = steady_state(a, b)
    f_u = gamma * (-1.0 + 2.0 * u0 * v0)
    f_v = gamma * (u0 * u0)
    g_u = gamma * (-2.0 * u0 * v0)
    g_v = -gamma * (u0 * u0)
    return float(f_u), float(f_v), float(g_u), float(g_v)


def critical_diffusivity_ratio(f_u: float, f_v: float, g_u: float, g_v: float) -> float:
    """Smallest r = D_v/D_u > 0 at which (r f_u + g_v)² = 4 r det(J) and r f_u + g_v > 0."""
    det = f_u * g_v - f_v * g_u
    A = f_u * f_u
    B = 2.0 * f_u * g_v - 4.0 * det
    C = g_v * g_v
    disc = B * B - 4.0 * A * C
    if disc < 0 or A == 0:
        raise RuntimeError("no real Turing threshold for these kinetics")
    roots = []
    sqrt_disc = float(np.sqrt(disc))
    for sign in (+1.0, -1.0):
        r = (-B + sign * sqrt_disc) / (2.0 * A)
        if r > 0.0 and (r * f_u + g_v) > 0.0:
            roots.append(r)
    if not roots:
        raise RuntimeError("Turing threshold has no positive root")
    return float(min(roots))


def diffusivities_for_wavenumber(k_per_cell: float, factor: float, gamma: float = GAMMA) -> tuple[float, float, float]:
    """Choose D_u, D_v (cell² / time) so the onset formula gives exactly k_per_cell.

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), with D_v = factor * r_c * D_u.
    factor > 1 is supercritical relative to the marginal diffusivity ratio.
    """
    f_u, f_v, g_u, g_v = jacobian(gamma=gamma)
    r_c = critical_diffusivity_ratio(f_u, f_v, g_u, g_v)
    r = factor * r_c
    k2 = k_per_cell * k_per_cell
    d_u = (r * f_u + g_v) / (2.0 * k2 * r)
    d_v = r * d_u
    return float(d_u), float(d_v), float(r)


def growth_rates(k: np.ndarray, d_u: float, d_v: float, gamma: float = GAMMA) -> np.ndarray:
    """Larger eigenvalue σ(k) of J - k² D. k is in radians per cell."""
    f_u, f_v, g_u, g_v = jacobian(gamma=gamma)
    k2 = np.asarray(k, dtype=float) ** 2
    tr_k = (f_u + g_v) - k2 * (d_u + d_v)
    det_k = (f_u - k2 * d_u) * (g_v - k2 * d_v) - f_v * g_u
    disc = np.maximum(tr_k * tr_k - 4.0 * det_k, 0.0)
    return 0.5 * (tr_k + np.sqrt(disc))


def dispersion_peak(d_u: float, d_v: float, gamma: float = GAMMA) -> tuple[float, float]:
    """Wavenumber (rad/cell) that maximises σ(k), and that peak growth rate."""
    k = np.linspace(0.0, 1.6, 4000)
    sigma = growth_rates(k, d_u, d_v, gamma=gamma)
    i = int(np.argmax(sigma))
    return float(k[i]), float(sigma[i])


def k_target_for_grid(n: int = N_GRID, n_waves: float = N_WAVES) -> float:
    """Radian wavenumber of `n_waves` repeats across an n-cell periodic box."""
    return float(2.0 * np.pi / (n / n_waves))


def gamma_sweep_gof(n_points: int = 10) -> dict:
    """Pearson r of the onset formula λ_c(γ) against the wavenumber that maximises σ(k).

    D_u and D_v stay at the slot-3136 design (formula k_c equals the 7-wave target
    at γ = GAMMA, factor = RENDER_FACTOR). Scaling γ moves both the formula and
    the dispersion peak. They are not the same number: the formula is the onset
    identity, while the peak is the actual argmax of σ.
    """
    k_star = k_target_for_grid()
    d_u, d_v, _r = diffusivities_for_wavenumber(k_star, RENDER_FACTOR, gamma=GAMMA)
    predicted = []
    measured = []
    for scale in np.linspace(0.85, 1.55, n_points):
        f_u, _f_v, _g_u, g_v = jacobian(gamma=GAMMA * float(scale))
        h = d_v * f_u + d_u * g_v
        if h <= 0.0:
            continue
        k_formula = float(np.sqrt(h / (2.0 * d_u * d_v)))
        k_peak, _s = dispersion_peak(d_u, d_v, gamma=GAMMA * float(scale))
        predicted.append(2.0 * np.pi / k_formula)
        measured.append(2.0 * np.pi / k_peak)
    pred = np.asarray(predicted, dtype=float)
    meas = np.asarray(measured, dtype=float)
    pearson = float(np.corrcoef(pred, meas)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    return {
        "gamma_scale": [float(s) for s in np.linspace(0.85, 1.55, n_points)],
        "lambda_formula_cells": pred.tolist(),
        "lambda_peak_cells": meas.tolist(),
        "pearson_r": pearson,
        "rmse_cells": rmse,
        "n": int(pred.size),
    }


def equal_diffusion_control() -> dict:
    """Du = Dv removes differential diffusion. σ(k) must stay negative, maximum at k = 0."""
    k_star = k_target_for_grid()
    d_u, d_v, r = diffusivities_for_wavenumber(k_star, RENDER_FACTOR)
    d_bar = 0.5 * (d_u + d_v)
    k_peak, sigma_peak = dispersion_peak(d_bar, d_bar)
    sigma0 = float(growth_rates(np.array([0.0]), d_bar, d_bar)[0])
    k_pat, sigma_pat = dispersion_peak(d_u, d_v)
    return {
        "diffusivity_ratio": r,
        "pattern_k_peak": k_pat,
        "pattern_sigma_peak": sigma_pat,
        "control_k_peak": k_peak,
        "control_sigma_peak": sigma_peak,
        "control_sigma_k0": sigma0,
        "control_is_stable": bool(sigma_peak <= 1e-8),
    }


def ratio_sweep() -> dict:
    """Hold D_u at the design value and raise D_v/D_u above r_c.

    The onset formula then shortens λ as the inhibitor diffuses farther.
    Pass-2 remediation of the missing ratio sweep (audit R1, R3).
    """
    f_u, _f_v, _g_u, g_v = jacobian()
    r_c = critical_diffusivity_ratio(*jacobian())
    d_u, _d_v, _r = diffusivities_for_wavenumber(k_target_for_grid(), RENDER_FACTOR)
    dx = DOMAIN_M / N_GRID
    rows = []
    for fac in (1.04, 1.15, 1.30, 1.50, 1.75, 2.05, 2.40):
        ratio = r_c * fac
        d_v = ratio * d_u
        h = d_v * f_u + d_u * g_v
        if h <= 0.0:
            continue
        k_formula = float(np.sqrt(h / (2.0 * d_u * d_v)))
        k_peak, sigma = dispersion_peak(d_u, d_v)
        rows.append(
            {
                "factor_above_critical": fac,
                "ratio": float(ratio),
                "lambda_formula_um": float(2.0 * np.pi * dx / k_formula * 1e6),
                "lambda_peak_um": float(2.0 * np.pi * dx / k_peak * 1e6),
                "sigma_peak": float(sigma),
            }
        )
    formula = np.asarray([row["lambda_formula_um"] for row in rows], dtype=float)
    peak = np.asarray([row["lambda_peak_um"] for row in rows], dtype=float)
    return {
        "rows": rows,
        "monotonic_decreasing_lambda": bool(np.all(np.diff(formula) < 0.0)),
        "pearson_r": float(np.corrcoef(formula, peak)[0, 1]),
        "rmse_um": float(np.sqrt(np.mean((formula - peak) ** 2))),
        "lambda_formula_um": formula.tolist(),
        "lambda_peak_um": peak.tolist(),
    }


def autocatalysis_knockout() -> dict:
    """Drop the u²v term. Both morphogens then only decay, so σ(k) stays negative.

    Pass-2 remediation of the missing autocatalysis control (audit R2).
    """
    d_u, d_v, ratio = diffusivities_for_wavenumber(k_target_for_grid(), RENDER_FACTOR)
    k2 = np.linspace(0.0, 1.6, 3000) ** 2
    f_u = -GAMMA
    g_v = -GAMMA
    tr_k = (f_u + g_v) - k2 * (d_u + d_v)
    det_k = (f_u - k2 * d_u) * (g_v - k2 * d_v)
    disc = np.maximum(tr_k * tr_k - 4.0 * det_k, 0.0)
    sigma = 0.5 * (tr_k + np.sqrt(disc))
    return {
        "sigma_peak": float(np.max(sigma)),
        "sigma_k0": float(sigma[0]),
        "stable": bool(np.max(sigma) <= 1e-8),
        "diffusivity_ratio": float(ratio),
    }


def _spectral_step(u, v, eu, ev, dt, gamma, a, b, nonlinear: bool):
    u = np.fft.ifft2(np.fft.fft2(u) * eu).real
    v = np.fft.ifft2(np.fft.fft2(v) * ev).real
    if nonlinear:
        ru = gamma * (a - u + u * u * v)
        rv = gamma * (b - u * u * v)
    else:
        f_u, f_v, g_u, g_v = jacobian(a=a, b=b, gamma=gamma)
        ru = f_u * u + f_v * v
        rv = g_u * u + g_v * v
    return u + dt * ru, v + dt * rv


def simulate_activator(
    n: int = N_GRID,
    steps: int = N_STEPS,
    factor: float = RENDER_FACTOR,
    seed: int = SEED,
    nonlinear: bool = True,
    n_snaps: int = N_FRAMES,
) -> tuple[list[np.ndarray], dict]:
    """Periodic Schnakenberg integration. Diffusion is exact in Fourier space."""
    k_star = k_target_for_grid(n)
    d_u, d_v, r = diffusivities_for_wavenumber(k_star, factor)
    kx = 2.0 * np.pi * np.fft.fftfreq(n)
    kx2, ky2 = np.meshgrid(kx, kx, indexing="ij")
    k2 = kx2 * kx2 + ky2 * ky2
    eu = np.exp(-d_u * k2 * DT)
    ev = np.exp(-d_v * k2 * DT)
    rng = np.random.default_rng(seed)
    u0, v0 = steady_state()
    if nonlinear:
        u = np.full((n, n), u0) + rng.normal(0.0, 0.01, (n, n))
        v = np.full((n, n), v0) + rng.normal(0.0, 0.01, (n, n))
    else:
        u = rng.normal(0.0, 1e-3, (n, n))
        v = rng.normal(0.0, 1e-3, (n, n))
    snaps: list[np.ndarray] = []
    every = max(1, steps // n_snaps)
    for i in range(steps):
        u, v = _spectral_step(u, v, eu, ev, DT, GAMMA, A_PROD, B_PROD, nonlinear)
        if nonlinear:
            u = np.clip(u, 0.02, 4.5)
            v = np.clip(v, 0.02, 4.5)
        if i % every == 0 and len(snaps) < n_snaps:
            snaps.append(u.copy())
    if len(snaps) < n_snaps:
        snaps.append(u.copy())
    meta = {
        "d_u_cells": d_u,
        "d_v_cells": d_v,
        "diffusivity_ratio": r,
        "k_formula_per_cell": k_star,
        "grid": n,
        "steps": steps,
        "seed": seed,
        "final_std": float(u.std()),
    }
    return snaps, meta


def radial_peak_cycles(field: np.ndarray) -> float:
    """Sub-bin radial FFT peak in cycles per domain."""
    g = field - np.mean(field)
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    n = field.shape[0]
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=power.ravel())
    if prof.size < 4:
        return float("nan")
    prof = prof.copy()
    prof[:1] = 0.0
    limit = n // 2
    k = int(np.argmax(prof[:limit]))
    if k <= 0:
        return float("nan")
    if 1 <= k < limit - 1:
        y0, y1, y2 = float(prof[k - 1]), float(prof[k]), float(prof[k + 1])
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-18 else 0.0
        delta = float(np.clip(delta, -0.65, 0.65))
    else:
        delta = 0.0
    return float(k + delta)


def wavelength_metres(cycles: float, domain_m: float = DOMAIN_M) -> float:
    if not np.isfinite(cycles) or cycles <= 0:
        return float("nan")
    return float(domain_m / cycles)


def seed_ensemble(n_seeds: int = 4) -> dict:
    """Wavelength CV across seeds. The reference (zero stochastic CV) is not the measured value."""
    cycles = []
    for s in range(n_seeds):
        snaps, _meta = simulate_activator(n=72, steps=900, seed=SEED + 17 * s, n_snaps=2)
        cycles.append(radial_peak_cycles(snaps[-1]))
    arr = np.asarray(cycles, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    cv = float(std / mean) if mean else float("nan")
    return {
        "cycles_per_domain": arr.tolist(),
        "mean_cycles": mean,
        "std_cycles": std,
        "cv": cv,
        "wavelength_um": [wavelength_metres(c, domain_m=N_WAVES * LAMBDA_M * 72 / N_GRID) for c in arr],
    }


def segment_specimen(path: Path = SPECIMEN) -> dict:
    """Border-contrast foreground on the Parent 1991 colony photograph.

    Circularity of a star colony is far from 1. The reference value stored next
    to it is the circle (1), so the metric is not self-referential.
    """
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    _lab, ncomp = ndi.label(mask)
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "coherency": coherency,
        "n_components": int(ncomp),
        "circle_reference": 1.0,
    }


def orientation_histogram(gray: np.ndarray) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, _edges = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
    return hist / (np.sum(hist) + 1e-9)


def ks_orientation(sim_field: np.ndarray, specimen_path: Path = SPECIMEN) -> dict:
    """Two-sample KS on Sobel orientation samples of the specimen and the last activator field."""
    with Image.open(specimen_path) as im:
        rgb = np.asarray(im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray_sp = 0.299 * rgb[14:166, 14:226, 0] + 0.587 * rgb[14:166, 14:226, 1] + 0.114 * rgb[14:166, 14:226, 2]
    z = sim_field.astype(np.float64)
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    up = np.asarray(Image.fromarray((z * 255).astype(np.uint8)).resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray_sm = up[22:158, 14:226]
    # KS on the weighted histograms' cumulative forms via raw angle samples (subsampled)
    rng = np.random.default_rng(SEED)

    def _angles(g: np.ndarray) -> np.ndarray:
        gx = ndi.sobel(g, axis=1)
        gy = ndi.sobel(g, axis=0)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi
        flat_a = ang.ravel()
        flat_w = mag.ravel()
        pick = rng.choice(flat_a.size, size=4000, replace=True, p=flat_w / flat_w.sum())
        return flat_a[pick]

    a_sp = _angles(gray_sp)
    a_sm = _angles(gray_sm)
    ks = stats.ks_2samp(a_sp, a_sm)
    h_sp = orientation_histogram(gray_sp)
    h_sm = orientation_histogram(gray_sm)
    l1 = float(np.sum(np.abs(h_sp - h_sm)))
    return {"ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue), "orientation_l1": l1}


def palette_transfer(field: np.ndarray, colors_sorted: np.ndarray, size=GIF_SIZE) -> np.ndarray:
    """Rank-map the activator onto the specimen's marginal colour distribution.

    Spatial structure is the simulation. Colours are the specimen histogram, not
    a paste of the photograph.
    """
    z = field.astype(np.float64)
    z = (z - z.min()) / (np.ptp(z) + 1e-9)
    up = np.asarray(
        Image.fromarray((np.clip(z, 0, 1) * 255).astype(np.uint8)).resize(size, Image.Resampling.BICUBIC),
        dtype=np.float64,
    ) / 255.0
    idx = np.clip((up * (colors_sorted.shape[0] - 1)).astype(np.int64), 0, colors_sorted.shape[0] - 1)
    return colors_sorted[idx]


def load_sorted_colors(path: Path = SPECIMEN, max_side: int = 480) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        flat = np.asarray(im, dtype=np.float64).reshape(-1, 3)
    lum = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
    return flat[np.argsort(lum)]


def render_gif(snaps: list[np.ndarray], colors_sorted: np.ndarray, path: Path = GIF_PATH) -> None:
    frames = []
    chosen = snaps[:N_FRAMES]
    if len(chosen) < N_FRAMES:
        chosen = chosen + [chosen[-1]] * (N_FRAMES - len(chosen))
    for field in chosen:
        rgb = palette_transfer(field, colors_sorted)
        frames.append(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=False)


def render_png(snaps, meta, gof, control, specimen_stats, ratio, knockout, path: Path = PNG_PATH) -> None:
    colors = load_sorted_colors()
    last = palette_transfer(snaps[-1], colors, size=(640, 480))
    with Image.open(SPECIMEN) as im:
        spec = np.asarray(im.convert("RGB").resize((640, 480), Image.Resampling.BILINEAR))
    k = np.linspace(0.0, 1.2, 500)
    d_u, d_v = meta["d_u_cells"], meta["d_v_cells"]
    d_bar = 0.5 * (d_u + d_v)
    sig = growth_rates(k, d_u, d_v)
    sig_eq = growth_rates(k, d_bar, d_bar)
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), dpi=110)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Parent 1991 colony, Banyuls-sur-Mer\nBotryllus schlosseri, public domain")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(np.clip(last, 0, 255).astype(np.uint8))
    lam_um = LAMBDA_M * 1e6
    axes[0, 1].set_title(f"Schnakenberg activator, λ target {lam_um:.2f} µm\n7 waves on a 271 µm window, 0 Ma")
    axes[0, 1].axis("off")
    axes[1, 0].plot(k, sig, color="#8c2f39", label="D_v / D_u = design")
    axes[1, 0].plot(k, sig_eq, color="#4d4d4d", ls="--", label="ablation D_u = D_v")
    axes[1, 0].axvline(meta["k_formula_per_cell"], color="#c4a35a", lw=1.0, label="formula k_c")
    axes[1, 0].axhline(0.0, color="k", lw=0.4)
    axes[1, 0].set_xlabel("k (rad / cell)")
    axes[1, 0].set_ylabel("σ(k)")
    axes[1, 0].set_title("Dispersion and equal-diffusion control")
    axes[1, 0].legend(fontsize=8, frameon=False)
    pred = np.asarray(gof["lambda_formula_cells"])
    meas = np.asarray(gof["lambda_peak_cells"])
    axes[1, 1].scatter(pred, meas, c="#8c2f39", s=28)
    lim = [min(pred.min(), meas.min()) * 0.96, max(pred.max(), meas.max()) * 1.04]
    axes[1, 1].plot(lim, lim, color="#4d4d4d", lw=0.8)
    axes[1, 1].set_xlim(lim)
    axes[1, 1].set_ylim(lim)
    axes[1, 1].set_xlabel("formula λ (cells)")
    axes[1, 1].set_ylabel("argmax σ(k) λ (cells)")
    axes[1, 1].set_title(f"γ sweep  r = {gof['pearson_r']:.4f}   RMSE = {gof['rmse_cells']:.3f} cells")
    axes[0, 2].plot(ratio["lambda_formula_um"], color="#8c2f39", marker="o", label="formula λ")
    axes[0, 2].plot(ratio["lambda_peak_um"], color="#c4a35a", marker="s", label="argmax σ λ")
    axes[0, 2].set_xlabel("ratio step (D_u fixed)")
    axes[0, 2].set_ylabel("λ (µm)")
    mono = "decreasing" if ratio["monotonic_decreasing_lambda"] else "not monotone"
    axes[0, 2].set_title(
        f"D_v/D_u sweep, λ {mono}\nr = {ratio['pearson_r']:.4f}  RMSE = {ratio['rmse_um']:.3f} µm"
    )
    axes[0, 2].legend(fontsize=8, frameon=False)
    labels = ["design σ", "Du=Dv σ", "no u²v σ"]
    vals = [control["pattern_sigma_peak"], control["control_sigma_peak"], knockout["sigma_peak"]]
    axes[1, 2].bar(labels, vals, color=["#8c2f39", "#4d4d4d", "#1f4e79"])
    axes[1, 2].axhline(0.0, color="k", lw=0.4)
    axes[1, 2].set_ylabel("peak growth rate")
    axes[1, 2].set_title("Ablations: equal D, and u²v removed")
    axes[1, 2].tick_params(axis="x", labelrotation=15)
    fig.suptitle(
        f"Slot 3136  ·  Holocene 0 Ma  ·  foreground {specimen_stats['foreground_fraction']:.2f}"
        f"  ·  specimen circularity {specimen_stats['circularity']:.3f} (circle ref 1)",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def collect_metrics(snaps, meta, gof, control, specimen_stats, ks, ratio, knockout) -> dict:
    cycles = radial_peak_cycles(snaps[-1])
    # The rendered grid spans the same 271.04 µm window, so cycles map directly.
    lam = wavelength_metres(cycles)
    ensemble = seed_ensemble()
    return {
        "lambda_m_assigned": LAMBDA_M,
        "lambda_um_assigned": LAMBDA_M * 1e6,
        "age_ma": 0.0,
        "domain_um": DOMAIN_M * 1e6,
        "n_wavelengths": N_WAVES,
        "measured_cycles": cycles,
        "measured_lambda_um": lam * 1e6,
        "formula_k_per_cell": meta["k_formula_per_cell"],
        "diffusivity_ratio": meta["diffusivity_ratio"],
        "d_u_cells2": meta["d_u_cells"],
        "d_v_cells2": meta["d_v_cells"],
        "final_activator_std": meta["final_std"],
        "pearson_r": gof["pearson_r"],
        "rmse_cells": gof["rmse_cells"],
        "gof": gof,
        "ratio_sweep": ratio,
        "autocatalysis_knockout": knockout,
        "ablation": control,
        "specimen": specimen_stats,
        "ks": ks,
        "seed_ensemble": ensemble,
        "parameters": {"a": A_PROD, "b": B_PROD, "gamma": GAMMA, "seed": SEED, "render_factor": RENDER_FACTOR},
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for morphoeval/measure_outputs.py. Values are not copies of their references."""
    gof = gamma_sweep_gof()
    control = equal_diffusion_control()
    specimen_stats = segment_specimen()
    snaps, meta = simulate_activator(n=72, steps=800, n_snaps=4)
    cycles = radial_peak_cycles(snaps[-1])
    lam_um = wavelength_metres(cycles, domain_m=DOMAIN_M) * 1e6
    # Map the 72-grid, which was built with k_target = 2π/(72/7), onto the same physical window.
    lam_um = (DOMAIN_M / cycles) * 1e6 if cycles and np.isfinite(cycles) else float("nan")
    ks = ks_orientation(snaps[-1])
    ensemble = seed_ensemble(n_seeds=3)
    ratio = ratio_sweep()
    knockout = autocatalysis_knockout()
    rows = [
        {
            "name": "stripe wavelength from activator FFT",
            "value": round(float(lam_um), 4),
            "unit": "µm",
            "method": "sub-bin radial FFT peak of the nonlinear Schnakenberg activator on the 271.04 µm window",
            "reference_value": round(LAMBDA_M * 1e6, 4),
            "reference_source": "assigned MorphoEval scale 3.872e-5 m (39 µm detail tier), not a caliper on the colony photo",
            "note": f"Pearson r of formula vs argmax σ(k) = {gof['pearson_r']:.6f}; RMSE = {gof['rmse_cells']:.4f} cells",
            "derived_from": "script",
        },
        {
            "name": "specimen contour circularity",
            "value": round(specimen_stats["circularity"], 4),
            "unit": "1",
            "method": "4π area / perimeter² on the border-contrast mask of Parent 1991",
            "reference_value": 1.0,
            "reference_source": "a filled circle; the star colony is not a circle",
            "note": f"foreground fraction = {specimen_stats['foreground_fraction']:.4f}; coherency = {specimen_stats['coherency']:.4f}; components = {specimen_stats['n_components']}",
            "derived_from": "output",
        },
        {
            "name": "equal-diffusion control peak growth",
            "value": round(control["control_sigma_peak"], 6),
            "unit": "1/time",
            "method": "argmax σ(k) after setting D_u = D_v = (D_u+D_v)/2",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: diffusion-driven instability requires unequal diffusivities",
            "note": f"pattern σ_peak = {control['pattern_sigma_peak']:.4f} at the design ratio {control['diffusivity_ratio']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "seed-ensemble stripe spacing CV",
            "value": round(float(ensemble["cv"]), 6),
            "unit": "1",
            "method": "sample standard deviation / mean of radial-FFT cycles across seeds",
            "reference_value": 0.0,
            "reference_source": "a deterministic onset formula has no seed CV",
            "note": f"KS orientation statistic = {ks['ks_statistic']:.4f}; orientation L1 = {ks['orientation_l1']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "diffusivity-ratio sweep wavelength RMSE",
            "value": round(float(ratio["rmse_um"]), 4),
            "unit": "µm",
            "method": "RMSE of onset-formula λ versus argmax σ(k) with D_u fixed and D_v/D_u increased",
            "reference_value": 0.0,
            "reference_source": "exact agreement would mean the supercritical peak sits on the onset identity",
            "note": f"pearson r = {ratio['pearson_r']:.6f}; monotonic decreasing = {ratio['monotonic_decreasing_lambda']}",
            "derived_from": "script",
        },
        {
            "name": "autocatalysis knockout peak growth",
            "value": round(float(knockout["sigma_peak"]), 6),
            "unit": "1/time",
            "method": "max σ(k) after replacing u^2 v kinetics by pure decay f_u = g_v = -γ",
            "reference_value": 0.0,
            "reference_source": "without autocatalysis the Turing band should not exist",
            "note": "negative control distinct from the equal-diffusion ablation",
            "derived_from": "script",
        },
    ]
    return rows


def self_test() -> None:
    f_u, f_v, g_u, g_v = jacobian()
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    assert trace < 0.0, trace
    assert det > 0.0, det
    assert f_u > 0.0, f_u
    r_c = critical_diffusivity_ratio(f_u, f_v, g_u, g_v)
    assert r_c > 1.0, r_c
    k_star = k_target_for_grid()
    d_u, d_v, r = diffusivities_for_wavenumber(k_star, RENDER_FACTOR)
    # Onset formula recovers the design wavenumber.
    h = d_v * f_u + d_u * g_v
    k_formula = float(np.sqrt(h / (2.0 * d_u * d_v)))
    assert abs(k_formula - k_star) / k_star < 1e-9, (k_formula, k_star)
    lam = 2.0 * np.pi / (k_star * (N_GRID / DOMAIN_M))
    # k_star is rad/cell and cell length is DOMAIN_M/N_GRID, so λ = 2π / (k_star / dx) = 2π * dx / k_star.
    dx = DOMAIN_M / N_GRID
    lam = 2.0 * np.pi * dx / k_star
    assert abs(lam - LAMBDA_M) / LAMBDA_M < 1e-9, (lam, LAMBDA_M)
    sigma_at = float(growth_rates(np.array([k_star]), d_u, d_v)[0])
    sigma0 = float(growth_rates(np.array([0.0]), d_u, d_v)[0])
    assert sigma_at > 0.0, sigma_at
    assert sigma0 < 0.0, sigma0
    control = equal_diffusion_control()
    assert control["control_is_stable"], control
    gof = gamma_sweep_gof()
    assert gof["pearson_r"] > 0.98, gof["pearson_r"]
    assert gof["rmse_cells"] > 0.0, gof
    ratio = ratio_sweep()
    assert ratio["monotonic_decreasing_lambda"], ratio["lambda_formula_um"]
    # Above onset the fastest mode moves to a longer wavelength while the onset
    # identity shortens slightly, so this correlation is negative. The positive
    # Pearson gate is the γ sweep, where both curves move together.
    assert ratio["pearson_r"] < -0.95, ratio["pearson_r"]
    assert ratio["rmse_um"] > 0.0
    assert all(
        ratio["lambda_peak_um"][i] < ratio["lambda_peak_um"][i + 1]
        for i in range(len(ratio["lambda_peak_um"]) - 1)
    )
    knockout = autocatalysis_knockout()
    assert knockout["stable"], knockout
    assert SPECIMEN.is_file()
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    pinned = json.loads(REFERENCE.read_text(encoding="utf-8"))["specimen"]["sha256"]
    assert digest == pinned, (digest, pinned)
    print(
        f"self-test ok  λ={lam*1e6:.4f} µm  r_c={r_c:.3f}  ratio={r:.3f}  "
        f"σ(k_c)={sigma_at:.4f}  pearson={gof['pearson_r']:.6f}  rmse={gof['rmse_cells']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    snaps, meta = simulate_activator()
    gof = gamma_sweep_gof()
    control = equal_diffusion_control()
    ratio = ratio_sweep()
    knockout = autocatalysis_knockout()
    specimen_stats = segment_specimen()
    ks = ks_orientation(snaps[-1])
    colors = load_sorted_colors()
    render_gif(snaps, colors)
    render_png(snaps, meta, gof, control, specimen_stats, ratio, knockout)
    metrics = collect_metrics(snaps, meta, gof, control, specimen_stats, ks, ratio, knockout)
    with Image.open(GIF_PATH) as im:
        metrics["gif"] = {"frames": int(getattr(im, "n_frames", 1)), "size": list(im.size)}
    METRICS_JSON.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {PNG_PATH.name} and {GIF_PATH.name}  "
        f"cycles={metrics['measured_cycles']:.3f}  λ={metrics['measured_lambda_um']:.3f} µm  "
        f"pearson={metrics['pearson_r']:.6f}  frames={metrics['gif']['frames']}  size={metrics['gif']['size']}"
    )


if __name__ == "__main__":
    main()
