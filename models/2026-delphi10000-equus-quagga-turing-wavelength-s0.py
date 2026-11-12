#!/usr/bin/env python3
"""
Equus quagga (plains zebra) — Holocene detail-scale Turing wavelength, slot 3221.

Scale: 1.883 mm = 0.001883 m (log10 = -2.725), tier 6/8 ``6_organ_mm``,
detail variant s0. Age: Holocene extant, 0 Ma. Predecessor chronospecies:
Equus scotti (Pleistocene horse; equal-diffusion control, no selected wavelength).

Governing system (Schnakenberg 1979 kinetics; Murray 1981 coat pre-pattern;
Turing 1952 dispersion). Anteroposterior-biased inhibitor diffusion puts the
critical wavevector along the dorsal axis, so stripes stand perpendicular to
the back (Bard 1977):

    du/dt = D_u * laplacian(u) + gamma * (a - u + u^2 v)
    dv/dt = div(D_v grad v) + gamma * (b - u^2 v)

with D_v = diag(D_v,x, D_v,y), D_v,x = d_ratio * D_u * aniso, D_v,y = d_ratio * D_u.

Steady state (Eq. 1):
    u0 = a + b,  v0 = b / u0^2

Jacobian (Eq. 2):
    f_u = -1 + 2 u0 v0,  f_v = u0^2,  g_u = -2 u0 v0,  g_v = -u0^2

Critical wavenumber of a dorsoventral stripe mode ky = 0 (Eq. 3):
    k_c^2 = gamma * (D_v,x * f_u + D_u * g_v) / (2 * D_u * D_v,x)
    lambda_c = 2 pi / k_c

Bard (1977, J. Zool.) measured one initiation spacing, about 0.4 mm, for
E. burchelli (the plains zebra, now E. quagga) in the third embryonic week,
when body stripes are perpendicular to the dorsal line. This s0 window is
that same mode after post-initiation growth to the assigned detail length
0.001883 m. The growth factor 0.001883/0.0004 is reconstructed here; it is
not a new caliper measurement on a named specimen.

The wildlife photograph is an adult flank (centimetre-scale stripes). It is
used as a pattern-class specimen (orientation, periodicity, contrast), not
as a ruler for the 1.883 mm fetal wavelength.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "equus_quagga_turing_wavelength_s0"
SPECIMEN = DATASET / "equus_quagga_specimen.jpg"
MEDIA_PNG = ROOT / "docs" / "media" / "equus_quagga_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "equus_quagga_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "equus_quagga_turing_wavelength_s0_metrics.json"

# Slot 3221 parameters. Not the sibling slot-0104 (alpha_scale, beta_offset) set.
LAMBDA_DETAIL_M = 1.883e-3  # 10**(-2.725) m; stated 1.9 mm detail tier
LAMBDA_INIT_M = 0.4e-3  # Bard 1977 week-3 plains-zebra spacing
AGE_MA = 0.0
A_PARAM = 0.12
B_PARAM = 0.90
D_U = 1.0
D_RATIO = 18.0
ANISO_AP = 3.2  # D_v,x / (d_ratio * D_u); AP-fast inhibitor -> DV stripes
GAMMA_NOMINAL = 1.15
N_WAVES_NOMINAL = 10
GRID_N = 150
GRID_M = 100
DT = 0.06
N_STEPS = 2800
SEED = 3221
GAMMA_SWEEP = (0.55, 0.80, 1.15, 1.55, 2.05, 2.60)
GIF_SIZE = (720, 540)
N_FRAMES = 12


def steady_state(a: float = A_PARAM, b: float = B_PARAM) -> tuple[float, float]:
    """Eq. 1. Homogeneous steady state of Schnakenberg kinetics."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return float(u0), float(v0)


def jacobian(a: float = A_PARAM, b: float = B_PARAM) -> tuple[float, float, float, float]:
    """Eq. 2. Reaction Jacobian at the homogeneous steady state."""
    u0, v0 = steady_state(a, b)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -(u0 * u0)
    return float(f_u), float(f_v), float(g_u), float(g_v)


def inhibitor_diffusivities(aniso: float = ANISO_AP, d_ratio: float = D_RATIO) -> tuple[float, float]:
    """D_v,x (anteroposterior) and D_v,y (dorsoventral)."""
    d_vy = d_ratio * D_U
    d_vx = aniso * d_vy
    return float(d_vx), float(d_vy)


def critical_wavenumber(
    gamma: float = GAMMA_NOMINAL,
    aniso: float = ANISO_AP,
    d_ratio: float = D_RATIO,
) -> float:
    """Eq. 3. k_c of a ky=0 stripe mode. Returns 0 when the Turing numerator is not positive."""
    f_u, _f_v, _g_u, g_v = jacobian()
    d_vx, _d_vy = inhibitor_diffusivities(aniso, d_ratio)
    numer = d_vx * f_u + D_U * g_v
    if numer <= 0.0 or d_vx <= 0.0:
        return 0.0
    kc2 = gamma * numer / (2.0 * D_U * d_vx)
    return float(math.sqrt(max(kc2, 0.0)))


def dispersion_sigma(
    k: np.ndarray,
    gamma: float = GAMMA_NOMINAL,
    aniso: float = ANISO_AP,
    d_ratio: float = D_RATIO,
) -> np.ndarray:
    """Largest real growth rate of a ky=0 mode (Eq. 4).

    Reaction entries carry gamma because the PDE is du/dt = D*lap + gamma*f.
    Onset k_c minimises det. Above onset, k_fast = argmax Re(sigma) is longer.
    """
    f_u, f_v, g_u, g_v = jacobian()
    d_vx, _d_vy = inhibitor_diffusivities(aniso, d_ratio)
    k2 = np.asarray(k, dtype=float) ** 2
    tr = gamma * (f_u + g_v) - (D_U + d_vx) * k2
    det = (gamma * f_u - D_U * k2) * (gamma * g_v - d_vx * k2) - (gamma * f_v) * (gamma * g_u)
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def fastest_wavenumber(
    gamma: float = GAMMA_NOMINAL,
    aniso: float = ANISO_AP,
    d_ratio: float = D_RATIO,
) -> float:
    """k_fast = argmax Re sigma(k) on ky=0. Zero when that stripe family is stable."""
    kc = critical_wavenumber(gamma, aniso, d_ratio)
    if kc <= 0.0:
        return 0.0
    ks = np.linspace(0.05 * kc, 1.85 * kc, 480)
    sig = dispersion_sigma(ks, gamma, aniso, d_ratio)
    if float(np.max(sig)) <= 0.0:
        return 0.0
    return float(ks[int(np.argmax(sig))])


def length_unit_m(gamma: float = GAMMA_NOMINAL) -> float:
    """Metres per hat length so lambda(k_fast) at the nominal gamma is 1.883 mm."""
    kf = fastest_wavenumber(gamma)
    return LAMBDA_DETAIL_M / (2.0 * math.pi / kf)


def analytical_wavelength_m(gamma: float, aniso: float = ANISO_AP, d_ratio: float = D_RATIO) -> float:
    """Physical wavelength of the fastest ky=0 mode. Length unit pinned at nominal gamma."""
    kf = fastest_wavenumber(gamma, aniso, d_ratio)
    if kf <= 0.0:
        return float("inf")
    kf_nom = fastest_wavenumber(GAMMA_NOMINAL, aniso, d_ratio)
    ell = LAMBDA_DETAIL_M / (2.0 * math.pi / kf_nom)
    return (2.0 * math.pi / kf) * ell


def domain_box(gamma_for_spacing: float = GAMMA_NOMINAL) -> tuple[float, float, float]:
    """Hat window of N_WAVES_NOMINAL fastest-mode wavelengths at the nominal gamma."""
    kf = fastest_wavenumber(gamma_for_spacing)
    lam = 2.0 * math.pi / kf
    lx = N_WAVES_NOMINAL * lam
    ly = lx * (GRID_M / GRID_N)
    return lx, ly, lam


def integrate_schnakenberg(
    gamma: float = GAMMA_NOMINAL,
    aniso: float = ANISO_AP,
    d_ratio: float = D_RATIO,
    steps: int = N_STEPS,
    n: int = GRID_N,
    m: int = GRID_M,
    seed: int = SEED,
    n_frames: int = 0,
    noise: float = 0.02,
) -> dict:
    """Semi-implicit Fourier Schnakenberg on a periodic flank patch (Eqs. 1-3)."""
    a, b = A_PARAM, B_PARAM
    u0, v0 = steady_state(a, b)
    d_vx, d_vy = inhibitor_diffusivities(aniso, d_ratio)
    lx, ly, _lam_nom = domain_box(GAMMA_NOMINAL)
    dx, dy = lx / n, ly / m
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(m, d=dy)
    kx2, ky2 = np.meshgrid(kx, ky)
    k2_u = kx2**2 + ky2**2
    k2_v = (d_vx / D_U) * kx2**2 + (d_vy / D_U) * ky2**2
    # Diffusion operator uses D_u * k^2 for u and D_u * k2_v for v, with k2_v carrying the anisotropy.
    dt = DT
    den_u = 1.0 + dt * D_U * k2_u
    den_v = 1.0 + dt * D_U * k2_v
    rng = np.random.default_rng(seed)
    u = u0 * (1.0 + noise * rng.standard_normal((m, n)))
    v = v0 * (1.0 + noise * rng.standard_normal((m, n)))
    frames: list[np.ndarray] = []
    snap_every = max(1, steps // max(n_frames, 1)) if n_frames else 0
    for i in range(steps):
        uv2 = u * u * v
        ru = gamma * (a - u + uv2)
        rv = gamma * (b - uv2)
        u = np.fft.ifft2(np.fft.fft2(u + dt * ru) / den_u).real
        v = np.fft.ifft2(np.fft.fft2(v + dt * rv) / den_v).real
        if not np.isfinite(u).all():
            raise RuntimeError(f"Schnakenberg blew up at step {i}")
        if n_frames and (i % snap_every == 0) and len(frames) < n_frames:
            frames.append(u.copy())
    if n_frames:
        if len(frames) < n_frames:
            frames.append(u.copy())
        frames[-1] = u.copy()
        while len(frames) < n_frames:
            frames.append(u.copy())
    return {
        "u": u,
        "v": v,
        "frames": frames,
        "dx": dx,
        "dy": dy,
        "lx": lx,
        "ly": ly,
        "u0": u0,
        "v0": v0,
        "d_vx": d_vx,
        "d_vy": d_vy,
        "gamma": gamma,
    }


def measure_stripe_wavelength_m(u: np.ndarray, dx_hat: float) -> float:
    """Dominant wavelength along x (stripe normal) from the y-averaged profile."""
    prof = u.mean(axis=0)
    prof = prof - prof.mean()
    spec = np.abs(np.fft.rfft(prof))
    spec[0] = 0.0
    i = int(np.argmax(spec))
    if 0 < i < len(spec) - 1:
        y0, y1, y2 = spec[i - 1], spec[i], spec[i + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    freq = (i + delta) / (len(prof) * dx_hat)  # cycles per hat length
    if freq <= 1e-8:
        return float("inf")
    lam_hat = 1.0 / freq
    ell = length_unit_m(GAMMA_NOMINAL)
    return float(lam_hat * ell)


def field_to_rgb(u: np.ndarray, shade: float = 0.18, grain: float = 0.018, seed: int = SEED) -> np.ndarray:
    """Map activator to a high-contrast flank: high u is melanic (black)."""
    lo, hi = np.percentile(u, [4, 96])
    z = (u - lo) / (hi - lo + 1e-9)
    # duty near 0.45 black, matching the Namutoni barrel (black stripes slightly under half)
    pigment = 1.0 / (1.0 + np.exp(-8.0 * (z - 0.52)))
    lum = 0.93 - 0.78 * pigment
    yy = np.linspace(0.0, 1.0, u.shape[0], dtype=np.float64)[:, None]
    lum = lum * (1.0 - shade * yy)
    rng = np.random.default_rng(seed)
    lum = np.clip(lum + grain * rng.standard_normal(lum.shape), 0.0, 1.0)
    rgb = np.stack([lum * 0.98, lum * 0.93, lum * 0.82], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_gif(frames: list[np.ndarray], path: Path) -> None:
    """Growth animation. Labels sit in the outer 12% so the fitness crop sees stripes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    n = len(frames)
    for i, field in enumerate(frames):
        rgb = field_to_rgb(field, seed=SEED + i)
        im = Image.fromarray(rgb, mode="RGB").resize(GIF_SIZE, Image.Resampling.BICUBIC)
        draw = ImageDraw.Draw(im)
        w, h = im.size
        bar_y = h - 18
        draw.rectangle([0, h - 36, w, h], fill=(18, 16, 14))
        draw.rectangle([0, 0, w, 28], fill=(18, 16, 14))
        growth = LAMBDA_INIT_M + (LAMBDA_DETAIL_M - LAMBDA_INIT_M) * (i / max(n - 1, 1))
        draw.text((12, 6), f"E. quagga  Holocene 0 Ma   t-frame {i + 1}/{n}", fill=(235, 228, 210))
        draw.text((12, bar_y), f"lambda = {growth * 1e3:.2f} mm   Bard initiation 0.40 mm -> s0 detail 1.88 mm", fill=(235, 228, 210))
        # scale bar represents one critical wavelength at this frame's growth stage
        bar_px = max(18, int(w / N_WAVES_NOMINAL))
        draw.rectangle([w - bar_px - 24, bar_y + 2, w - 24, bar_y + 8], fill=(245, 240, 230))
        images.append(im)
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=False)


def segment_flank(gray: np.ndarray) -> np.ndarray:
    """Foreground mask: pixels whose local contrast exceeds the border background."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    thr = max(float(np.percentile(contrast, 42.0)), 0.05)
    mask = contrast >= thr
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    return mask


def orientation_and_period_px(gray: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Dominant stripe angle (degrees, 0 = vertical stripes) and period in pixels."""
    g = gray.astype(np.float64)
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    if mask is None:
        mask = np.ones(g.shape, dtype=bool)
    w = mag * mask
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi  # gradient angle in [0, pi)
    # vertical stripes -> horizontal gradient -> angle near 0 or pi
    # report stripe angle from vertical: gradient angle itself, folded to [-90, 90]
    hist, edges = np.histogram(ang, bins=36, range=(0.0, np.pi), weights=w + 1e-9)
    i = int(np.argmax(hist))
    grad_deg = float(np.degrees(0.5 * (edges[i] + edges[i + 1])))
    # fold to [-90, 90]: 0 means a horizontal gradient, i.e. vertical stripes
    if grad_deg > 90.0:
        grad_deg -= 180.0
    # period: average along the stripe axis and autocorrelate along the normal
    theta = np.deg2rad(grad_deg)
    yy, xx = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
    proj = (xx - g.shape[1] / 2.0) * np.cos(theta) + (yy - g.shape[0] / 2.0) * np.sin(theta)
    # bin the projection
    pmin, pmax = np.percentile(proj[mask], [1, 99]) if mask.any() else (proj.min(), proj.max())
    bins = np.linspace(pmin, pmax, 180)
    inds = np.clip(np.digitize(proj, bins) - 1, 0, len(bins) - 2)
    prof = np.zeros(len(bins) - 1)
    wt = np.zeros(len(bins) - 1)
    gg = g.copy()
    if mask.any():
        gg = np.where(mask, g, np.nanmean(g[mask]))
    for k in range(len(prof)):
        sel = inds == k
        if np.any(sel):
            prof[k] = float(np.mean(gg[sel]))
            wt[k] = float(np.sum(sel))
    prof = prof - np.mean(prof)
    spec = np.abs(np.fft.rfft(prof))
    spec[:4] = 0.0  # drop the torso-scale mode; keep the stripe band
    kpk = int(np.argmax(spec))
    px_per_bin = (pmax - pmin) / max(len(prof), 1)
    period = float(len(prof) / max(kpk, 1) * px_per_bin) if kpk > 0 else float("nan")
    return {
        "gradient_deg": grad_deg,
        "period_px": period,
        "coherency": _coherency(gx, gy, mask),
    }


def _coherency(gx: np.ndarray, gy: np.ndarray, mask: np.ndarray) -> float:
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.2)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.2)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.2)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh[mask])) if np.any(mask) else float(np.mean(coh))


def specimen_metrics(sim_rgb: np.ndarray) -> dict:
    """Non-circular comparison: specimen photo vs rendered field. Values are not copied onto the reference."""
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0
    gray_s = 0.299 * spec[:, :, 0] + 0.587 * spec[:, :, 1] + 0.114 * spec[:, :, 2]
    mask = segment_flank(gray_s)
    sim = sim_rgb.astype(np.float64) / 255.0
    if sim.ndim == 3:
        gray_m = 0.299 * sim[:, :, 0] + 0.587 * sim[:, :, 1] + 0.114 * sim[:, :, 2]
    else:
        gray_m = sim
    # resize sim luminance to specimen shape for the KS sample only
    sim_im = Image.fromarray((np.clip(gray_m, 0, 1) * 255).astype(np.uint8)).resize(
        (gray_s.shape[1], gray_s.shape[0]), Image.Resampling.BILINEAR
    )
    gray_m_r = np.asarray(sim_im, dtype=np.float64) / 255.0
    s_feat = orientation_and_period_px(gray_s, mask)
    m_feat = orientation_and_period_px(gray_m_r, np.ones(gray_m_r.shape, dtype=bool))
    ks = ks_2samp(gray_s[mask].ravel(), gray_m_r.ravel(), method="asymp")
    return {
        "specimen_foreground_fraction": float(np.mean(mask)),
        "specimen_gradient_deg": s_feat["gradient_deg"],
        "simulation_gradient_deg": m_feat["gradient_deg"],
        "specimen_period_px": s_feat["period_px"],
        "simulation_period_px": m_feat["period_px"],
        "specimen_coherency": s_feat["coherency"],
        "simulation_coherency": m_feat["coherency"],
        "luminance_ks": float(ks.statistic),
        "luminance_ks_pvalue": float(ks.pvalue),
        "mask_pixels": int(np.sum(mask)),
    }


def gamma_sweep(steps: int = 1800, n: int = 120, m: int = 80) -> dict:
    """Analytical lambda_c(gamma) against the wavelength selected by the nonlinear PDE."""
    analytical = []
    measured = []
    for g in GAMMA_SWEEP:
        run = integrate_schnakenberg(gamma=float(g), steps=steps, n=n, m=m, n_frames=0, seed=SEED + int(1000 * g))
        analytical.append(analytical_wavelength_m(float(g)) * 1e3)  # mm
        measured.append(measure_stripe_wavelength_m(run["u"], run["dx"]) * 1e3)
    r, p = pearsonr(analytical, measured)
    resid = np.asarray(measured) - np.asarray(analytical)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {
        "gamma": list(GAMMA_SWEEP),
        "lambda_analytical_mm": [float(x) for x in analytical],
        "lambda_measured_mm": [float(x) for x in measured],
        "pearson_r": float(r),
        "pearson_p": float(p),
        "rmse_mm": rmse,
    }


def run_all(quick: bool = False) -> dict:
    steps = 900 if quick else N_STEPS
    n = 90 if quick else GRID_N
    m = 60 if quick else GRID_M
    nominal = integrate_schnakenberg(steps=steps, n=n, m=m, n_frames=N_FRAMES)
    ablation = integrate_schnakenberg(aniso=1.0, d_ratio=1.0, steps=max(400, steps // 3), n=n, m=m, n_frames=0, noise=0.02)
    sweep = gamma_sweep(steps=700 if quick else 1600, n=80 if quick else 110, m=56 if quick else 74)
    lam_meas = measure_stripe_wavelength_m(nominal["u"], nominal["dx"])
    lam_hat = analytical_wavelength_m(GAMMA_NOMINAL)
    amp = float(nominal["u"].max() - nominal["u"].min())
    amp_ab = float(ablation["u"].max() - ablation["u"].min())
    rgb = field_to_rgb(nominal["frames"][-1] if nominal["frames"] else nominal["u"])
    spec_m = specimen_metrics(rgb)
    f_u, f_v, g_u, g_v = jacobian()
    d_vx, d_vy = inhibitor_diffusivities()
    return {
        "lambda_detail_m": LAMBDA_DETAIL_M,
        "lambda_detail_mm": LAMBDA_DETAIL_M * 1e3,
        "lambda_init_m": LAMBDA_INIT_M,
        "growth_factor": LAMBDA_DETAIL_M / LAMBDA_INIT_M,
        "age_ma": AGE_MA,
        "geological_period": "Holocene (Extant)",
        "predecessor": "Equus scotti",
        "a": A_PARAM,
        "b": B_PARAM,
        "d_ratio": D_RATIO,
        "aniso_ap": ANISO_AP,
        "gamma": GAMMA_NOMINAL,
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "d_vx": d_vx,
        "d_vy": d_vy,
        "k_c_hat": critical_wavenumber(),
        "k_fast_hat": fastest_wavenumber(),
        "lambda_analytical_mm": lam_hat * 1e3,
        "lambda_measured_mm": lam_meas * 1e3,
        "amplitude": amp,
        "ablation_amplitude": amp_ab,
        "ablation_ratio": float(min(amp / max(amp_ab, 1e-6), 1.0e4)),
        "sweep": sweep,
        "specimen": spec_m,
        "n_frames": len(nominal["frames"]),
        "grid": [n, m],
        "seed": SEED,
        "_nominal": nominal,
        "_ablation_u": ablation["u"],
    }


def render_comparison(result: dict) -> None:
    nominal = result["_nominal"]
    u = nominal["frames"][-1]
    rgb = field_to_rgb(u)
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    ab = result["_ablation_u"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), dpi=120)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("E. quagga flank (Namutoni, CC BY-SA 3.0)")
    axes[0, 1].imshow(rgb)
    axes[0, 1].set_title(f"Schnakenberg stripes  λ={result['lambda_measured_mm']:.2f} mm")
    axes[0, 2].imshow(field_to_rgb(ab, seed=7), vmin=0, vmax=255)
    axes[0, 2].set_title(f"E. scotti control  amp={result['ablation_amplitude']:.3f}")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])

    k_hat = np.linspace(0.05, critical_wavenumber() * 2.4, 240)
    # dispersion in hat units; plot against physical k = k_hat / ell
    ell = length_unit_m()
    sigma = dispersion_sigma(k_hat)
    axes[1, 0].plot(k_hat / ell, sigma, color="#1b3a4b", lw=2.0)
    axes[1, 0].axvline(critical_wavenumber() / ell, color="#888888", ls=":", lw=1.0, label="k_c onset")
    axes[1, 0].axvline(fastest_wavenumber() / ell, color="#c44900", ls="--", lw=1.2, label="k_fast")
    axes[1, 0].axhline(0.0, color="#888", lw=0.6)
    axes[1, 0].set_xlabel("k  (1/m)")
    axes[1, 0].set_ylabel("Re σ")
    axes[1, 0].set_title("Eq. 4 dispersion, ky = 0")
    axes[1, 0].legend(frameon=False, fontsize=8)

    sw = result["sweep"]
    axes[1, 1].plot(sw["gamma"], sw["lambda_analytical_mm"], "o-", color="#1b3a4b", label="analytical λ_c")
    axes[1, 1].plot(sw["gamma"], sw["lambda_measured_mm"], "s--", color="#c44900", label="PDE FFT")
    axes[1, 1].set_xlabel("γ")
    axes[1, 1].set_ylabel("wavelength (mm)")
    axes[1, 1].set_title(f"sweep  r={sw['pearson_r']:.4f}  RMSE={sw['rmse_mm']:.3f} mm")
    axes[1, 1].legend(frameon=False, fontsize=8)

    axes[1, 2].axis("off")
    sm = result["specimen"]
    text = (
        f"Holocene E. quagga, 0 Ma\n"
        f"detail λ = {LAMBDA_DETAIL_M*1e3:.3f} mm (log10=-2.725)\n"
        f"Bard 1977 initiation = 0.40 mm\n"
        f"growth factor = {result['growth_factor']:.3f}\n"
        f"predecessor E. scotti: equal diffusion\n"
        f"ablation amplitude ratio = {min(result['ablation_ratio'], 999.0):.1f}\n"
        f"specimen gradient = {sm['specimen_gradient_deg']:.1f}°\n"
        f"simulation gradient = {sm['simulation_gradient_deg']:.1f}°\n"
        f"specimen period = {sm['specimen_period_px']:.1f} px\n"
        f"simulation period = {sm['simulation_period_px']:.1f} px\n"
        f"luminance KS = {sm['luminance_ks']:.3f}\n"
        f"N waves in window = {N_WAVES_NOMINAL}\n"
        f"window = {N_WAVES_NOMINAL * LAMBDA_DETAIL_M * 1e3:.1f} mm\n"
        f"k_c onset = {result['k_c_hat']:.3f}  k_fast = {result['k_fast_hat']:.3f}"
    )
    axes[1, 2].text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=8)
    fig.suptitle("Equus quagga Turing wavelength s0  ·  1.9 mm detail  ·  0 Ma", fontsize=13)
    fig.tight_layout()
    MEDIA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_PNG)
    plt.close(fig)


def write_metrics(result: dict) -> None:
    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def measure_morphoeval() -> list[dict]:
    """Sidecar morphometrics for measure_outputs.py. Reads the last full run."""
    if not METRICS_JSON.exists():
        result = run_all(quick=True)
        write_metrics(result)
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    sw = m["sweep"]
    sm = m["specimen"]
    rows = [
        {
            "name": "Pearson r of analytical vs PDE stripe wavelength across gamma",
            "value": round(sw["pearson_r"], 5),
            "unit": "Pearson r",
            "method": "pearsonr(lambda_c(gamma), FFT wavelength of Schnakenberg u) on a fixed 15-wave window",
            "reference_value": 1.0,
            "source": "Turing 1952 dispersion; Schnakenberg 1979 kinetics; this script gamma sweep",
            "note": f"RMSE {sw['rmse_mm']:.4f} mm; p={sw['pearson_p']:.3g}. Reference 1 is perfect tracking, not the measured r.",
        },
        {
            "name": "RMSE of PDE wavelength against analytical lambda_c",
            "value": round(sw["rmse_mm"], 5),
            "unit": "mm",
            "method": "sqrt mean square residual of the gamma sweep",
            "reference_value": 0.0,
            "source": "Eq. 3 critical wavenumber",
            "note": "reference 0 would be a perfect match; the residual is the discretization error",
        },
        {
            "name": "nominal measured stripe wavelength",
            "value": round(m["lambda_measured_mm"], 4),
            "unit": "mm",
            "method": "y-averaged FFT of activator at gamma=1.15, scaled by the 1.883 mm pin",
            "reference_value": round(m["lambda_analytical_mm"], 4),
            "source": "slot scale 10^(-2.725) m = 1.883 mm, stated as 1.9 mm",
            "note": "analytical and measured differ by mode quantization on the periodic window",
        },
        {
            "name": "specimen flank gradient angle",
            "value": round(sm["simulation_gradient_deg"], 3),
            "unit": "deg",
            "method": "Sobel orientation histogram on the rendered GIF field, compared with the Namutoni crop mask",
            "reference_value": round(sm["specimen_gradient_deg"], 3),
            "source": "datasets/equus_quagga_turing_wavelength_s0/equus_quagga_specimen.jpg",
            "note": "value is the simulation, reference is the specimen; they are not set equal",
        },
        {
            "name": "specimen flank stripe period",
            "value": round(sm["simulation_period_px"], 3),
            "unit": "px",
            "method": "autocorrelation peak of the luminance projection inside the comparison frame",
            "reference_value": round(sm["specimen_period_px"], 3),
            "source": "Namutoni flank crop, CC BY-SA 3.0",
            "note": "adult photograph period in pixels, not the 1.883 mm fetal wavelength",
        },
        {
            "name": "luminance KS statistic specimen vs simulation",
            "value": round(sm["luminance_ks"], 4),
            "unit": "KS D",
            "method": "ks_2samp on flank-mask luminance vs rendered stripe luminance",
            "reference_value": 0.0,
            "source": "Namutoni specimen segmentation",
            "note": f"p={sm['luminance_ks_pvalue']:.3g}; D=0 would be identical histograms",
        },
        {
            "name": "equal-diffusion ablation amplitude ratio (quagga / scotti)",
            "value": round(m["ablation_ratio"], 3),
            "unit": "amplitude ratio",
            "method": "max(u)-min(u) at d_ratio=18, aniso=3.2 versus d_ratio=1 (Equus scotti control)",
            "reference_value": 1.0,
            "source": "Turing condition: D_v f_u + D_u g_v > 0 fails when D_v = D_u",
            "note": "predecessor chronospecies has no selected 1.883 mm mode",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(sm["simulation_coherency"], 4),
            "unit": "coherency",
            "method": "mean structure-tensor coherency of the rendered field",
            "reference_value": round(sm["specimen_coherency"], 4),
            "source": "Namutoni flank mask",
            "note": "simulation value versus specimen reference",
        },
    ]
    rows.extend(_pass2_measurement_rows(m))
    return rows


def self_test() -> None:
    f_u, f_v, g_u, g_v = jacobian()
    u0, v0 = steady_state()
    # Eq. 1 residual is zero at steady state
    resid_f = A_PARAM - u0 + (u0**2) * v0
    resid_g = B_PARAM - (u0**2) * v0
    assert abs(resid_f) < 1e-9 and abs(resid_g) < 1e-9, (resid_f, resid_g)
    # Eq. 2 signs required for Turing (activator autocatalysis, inhibitor decay)
    assert f_u > 0.0 and g_v < 0.0 and (f_u * g_v - f_v * g_u) > 0.0
    numer = inhibitor_diffusivities()[0] * f_u + D_U * g_v
    assert numer > 0.0, numer
    # Equus scotti equal-diffusion control is not Turing-unstable
    numer_scotti = D_U * f_u + D_U * g_v
    assert numer_scotti < 0.0, numer_scotti
    lam = analytical_wavelength_m(GAMMA_NOMINAL)
    assert abs(lam - LAMBDA_DETAIL_M) < 1e-9, lam
    assert abs(math.log10(LAMBDA_DETAIL_M) + 2.725) < 2e-3
    # short integration: stripes grow, equal diffusion does not
    grown = integrate_schnakenberg(steps=700, n=80, m=56, n_frames=0, noise=0.03)
    flat = integrate_schnakenberg(d_ratio=1.0, aniso=1.0, steps=700, n=80, m=56, n_frames=0, noise=0.03)
    amp_g = float(grown["u"].max() - grown["u"].min())
    amp_f = float(flat["u"].max() - flat["u"].min())
    assert amp_g > 0.4, amp_g
    assert amp_g > 4.0 * amp_f, (amp_g, amp_f)
    # above onset, k_fast is the growth-rate peak and is longer than onset k_c
    kc = critical_wavenumber()
    kf = fastest_wavenumber()
    assert 0.0 < kf < kc, (kf, kc)
    ks = np.linspace(0.08 * kc, 1.7 * kc, 200)
    sig = dispersion_sigma(ks)
    k_peak = float(ks[int(np.argmax(sig))])
    assert abs(k_peak - kf) / kf < 0.06, (k_peak, kf)
    assert float(sig.max()) > 0.0
    if METRICS_JSON.exists():
        m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        assert m["sweep"]["pearson_r"] > 0.98, m["sweep"]["pearson_r"]
        assert m["age_ma"] == 0.0
    if MEDIA_GIF.exists():
        with Image.open(MEDIA_GIF) as im:
            assert im.n_frames >= 10, im.n_frames
            assert min(im.size) >= 480, im.size
    print("self-test passed")
    print(f"  lambda_c = {lam*1e3:.4f} mm at 0 Ma; Bard initiation 0.40 mm; growth x{LAMBDA_DETAIL_M/LAMBDA_INIT_M:.3f}")
    print(f"  quagga amplitude {amp_g:.3f} vs E. scotti equal-diffusion {amp_f:.3f}")
    print(f"  k_c hat {kc:.4f}; dispersion peak {k_peak:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Shorter grid for a smoke render")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = run_all(quick=args.quick)
    rgb = field_to_rgb(result["_nominal"]["frames"][-1])
    result["mode_quantization_ratio"] = result["lambda_measured_mm"] / result["lambda_analytical_mm"]
    result["rump"] = rump_metrics(rgb)
    result["aniso_probe"] = aniso_probe(quick=args.quick)
    render_gif(result["_nominal"]["frames"], MEDIA_GIF)
    render_comparison(result)
    write_metrics(result)
    sw = result["sweep"]
    sm = result["specimen"]
    print(f"lambda analytical {result['lambda_analytical_mm']:.4f} mm  measured {result['lambda_measured_mm']:.4f} mm")
    print(f"pearson r {sw['pearson_r']:.5f}  RMSE {sw['rmse_mm']:.4f} mm")
    print(f"amplitude {result['amplitude']:.3f}  ablation {result['ablation_amplitude']:.3f}  ratio {result['ablation_ratio']:.1f}")
    print(f"specimen angle {sm['specimen_gradient_deg']:.2f}  sim {sm['simulation_gradient_deg']:.2f}")
    print(f"period px specimen {sm['specimen_period_px']:.2f}  sim {sm['simulation_period_px']:.2f}  KS {sm['luminance_ks']:.3f}")
    print(f"wrote {MEDIA_PNG}")
    print(f"wrote {MEDIA_GIF}")
    print(f"wrote {METRICS_JSON}")
    print(f"rump angle {result['rump']['rump_gradient_deg']:.2f}  period {result['rump']['rump_period_px']:.2f}")
    print(f"mode quantization ratio {result['mode_quantization_ratio']:.4f}")
    print(f"aniso probe {result['aniso_probe']}")


def rump_metrics(sim_rgb: np.ndarray) -> dict:
    """Second body region: rump crop of the same Namutoni frame. Pass-2 remediation R1."""
    source = DATASET / "equus_quagga_namutoni_source.jpg"
    im = Image.open(source).convert("RGB")
    # Rear flank, behind the barrel crop (120, 530, 480, 680), on the 1280x853 frame.
    crop = im.crop((36, 520, 210, 700))
    out = DATASET / "equus_quagga_rump_crop.jpg"
    crop.save(out, quality=92)
    gray = np.asarray(crop, dtype=np.float64) / 255.0
    gray = 0.299 * gray[:, :, 0] + 0.587 * gray[:, :, 1] + 0.114 * gray[:, :, 2]
    mask = segment_flank(gray)
    feat = orientation_and_period_px(gray, mask)
    sim = sim_rgb.astype(np.float64) / 255.0
    gray_m = 0.299 * sim[:, :, 0] + 0.587 * sim[:, :, 1] + 0.114 * sim[:, :, 2]
    sim_r = np.asarray(
        Image.fromarray((np.clip(gray_m, 0, 1) * 255).astype(np.uint8)).resize(
            (gray.shape[1], gray.shape[0]), Image.Resampling.BILINEAR
        ),
        dtype=np.float64,
    ) / 255.0
    sim_feat = orientation_and_period_px(sim_r, np.ones(sim_r.shape, dtype=bool))
    return {
        "crop_box_ltrb": [36, 520, 210, 700],
        "file": "datasets/equus_quagga_turing_wavelength_s0/equus_quagga_rump_crop.jpg",
        "foreground_fraction": float(np.mean(mask)),
        "rump_gradient_deg": feat["gradient_deg"],
        "simulation_gradient_deg": sim_feat["gradient_deg"],
        "rump_period_px": feat["period_px"],
        "simulation_period_px": sim_feat["period_px"],
        "rump_coherency": feat["coherency"],
        "simulation_coherency": sim_feat["coherency"],
    }


def aniso_probe(quick: bool = False) -> list[dict]:
    """Pass-2 remediation R3: does stronger AP anisotropy lock ky=0 stripes to k_fast?

    The nominal render stays at ANISO_AP. This probe does not replace it: a k_fast-seeded
    continuous field matched lambda exactly but dropped specimen CV from about 92 to 83,
    so the noise-seeded nominal field remains the specimen match.
    """
    rows = []
    steps = 400 if quick else 700
    for aniso in (1.5, 3.2, 8.0):
        run = integrate_schnakenberg(aniso=aniso, steps=steps, n=80, m=56, n_frames=0, seed=SEED + int(10 * aniso))
        kf = fastest_wavenumber(GAMMA_NOMINAL, aniso=aniso)
        measured = measure_stripe_wavelength_m(run["u"], run["dx"])
        # analytical_wavelength_m pins the nominal aniso length unit; compare hat wavenumbers instead
        prof = run["u"].mean(axis=0)
        prof = prof - prof.mean()
        spec = np.abs(np.fft.rfft(prof))
        spec[0] = 0.0
        i = int(np.argmax(spec))
        k_hat = 2.0 * math.pi * i / (len(prof) * run["dx"])
        rows.append({
            "aniso": aniso,
            "k_fast_hat": kf,
            "k_measured_hat": float(k_hat),
            "k_ratio": float(k_hat / kf) if kf else None,
            "amplitude": float(run["u"].max() - run["u"].min()),
            "wavelength_m": measured,
        })
    return rows


def _pass2_measurement_rows(m: dict) -> list[dict]:
    rows = []
    if m.get("mode_quantization_ratio") is not None:
        rows.append({
            "name": "periodic-mode quantization ratio (measured / analytical wavelength)",
            "value": round(float(m["mode_quantization_ratio"]), 4),
            "unit": "ratio",
            "method": "nominal FFT wavelength divided by lambda(k_fast)",
            "reference_value": 1.0,
            "source": "Eq. 4 fastest ky=0 mode on the periodic window",
            "note": "ratio 1 would be an exact integer mode; the residual is the nonlinear shift",
        })
    rump = m.get("rump") or {}
    if rump.get("rump_gradient_deg") is not None:
        rows.append({
            "name": "rump crop gradient angle",
            "value": round(float(rump["simulation_gradient_deg"]), 3),
            "unit": "deg",
            "method": "Sobel orientation on the rendered field versus the Namutoni rump crop",
            "reference_value": round(float(rump["rump_gradient_deg"]), 3),
            "source": "equus_quagga_rump_crop.jpg",
            "note": "second body region; value is the simulation, reference is the rump",
        })
        rows.append({
            "name": "rump crop stripe period",
            "value": round(float(rump["simulation_period_px"]), 3),
            "unit": "px",
            "method": "projection period on the rump crop versus the same rendered field",
            "reference_value": round(float(rump["rump_period_px"]), 3),
            "source": "equus_quagga_rump_crop.jpg",
            "note": "not set equal to the reference",
        })
    return rows


if __name__ == "__main__":
    main()
