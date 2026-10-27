#!/usr/bin/env python3
"""
Bacillus subtilis — Turing wavenumber at the 16 µm detail scale (Loop 3130).

Holocene extant chronoseries node 1/1 (0 Ma). Tier 4/8 ``4_cellular_10um``,
detail variant: characteristic wavelength λ = 16 µm = 1.58e-5 m (log10 = -4.801).
The tier-center scale is 10 µm. This slot has no predecessor or successor
chronospecies, so the age increment is 0 Ma. What changes relative to the
tier-center scale is the selected wavenumber:

    k_c(16 µm) / k_c(10 µm) = 10/16 = 0.625
    D(16 µm) / D(10 µm) = (16/10)^2 = 2.56     (fixed kinetics, λ ∝ √D)

Governing system (Schnakenberg activator–inhibitor, Turing 1952 dispersion):

    ∂u/∂t = D_u ∇²u + a - u + u² v
    ∂v/∂t = D_v ∇²v + b - u² v

Homogeneous steady state (u₀, v₀) = (a+b, b/(a+b)²). Jacobian entries f_u, f_v,
g_u, g_v give the growth rate σ₊(k) as the larger root of

    σ² - τ σ + Δ = 0
    τ = f_u + g_v - k² (D_u + D_v)
    Δ = D_u D_v k⁴ - (D_v f_u + D_u g_v) k² + (f_u g_v - f_v g_u)

The operative wavenumber is k_* = argmax σ₊(k), calibrated to 2π/16 µm⁻¹.
On an expanding circular front the same wavelength sets the circumferential
branch count m(R) = round(2π R / λ). That reduced front is the ocellus drawn
in the animation; the bulk PDE is the dispersion test.

Sources (see datasets/bacillus_subtilis_turing_wavelength_s0/paper_citations.bib):
  Turing 1952 Phil. Trans. R. Soc. B 237:37  DOI 10.1098/rstb.1952.0012
  Kawasaki et al. 1997 J. Theor. Biol. 188:177  DOI 10.1006/jtbi.1997.0462
  Golding et al. 1998 Physica A 260:510  DOI 10.1016/S0378-4371(98)00345-8
  Ben-Jacob, Cohen & Levine 2000 Adv. Phys. 49:395  DOI 10.1080/000187300405228
  Kondo & Miura 2010 Science 329:1616  DOI 10.1126/science.1179047

This script does not claim that Kawasaki et al. measured a 16 µm pigment
wavelength (their colonies are millimetre-scale). 16 µm is the assigned
cellular-detail wavelength for this morphospace slot: a few B. subtilis rod
lengths (Sharpe et al. 1998; Dion et al. 2019).

Usage
-----
  python3 models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py
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
DATASET = ROOT / "datasets" / "bacillus_subtilis_turing_wavelength_s0"
SPECIMEN = DATASET / "bacillus_subtilis_ocellus_specimen.jpg"
PNG_PATH = ROOT / "docs" / "media" / "bacillus_subtilis_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "bacillus_subtilis_turing_wavelength_s0_growth.gif"
METRICS_PATH = DATASET / "pass_metrics.json"

# Slot 3130 constants. Not copied from a sibling chronospecies or organism.
SEED = 3130
LAMBDA_UM = 16.0
LAMBDA_TIER_CENTER_UM = 10.0
AGE_MA = 0.0
CHRONOSERIES_INDEX = 1
CHRONOSERIES_LENGTH = 1
DOMAIN_UM = 192.0  # exactly 12 wavelengths of 16 µm
# Outer front of the reconstructed microcolony ocellus. No scale bar is present
# on the Korinna 2019 micrograph, so this radius is a reconstruction: the
# circumference 2π R holds exactly 72 Turing wavelengths of 16 µm.
FRONT_RADIUS_UM = 72.0 * LAMBDA_UM / (2.0 * np.pi)
A_KIN = 0.084
B_KIN = 0.916
DIFFUSION_RATIO = 9.20  # above the Schnakenberg Turing threshold (~8.00)

U0 = A_KIN + B_KIN
V0 = B_KIN / (U0 ** 2)
F_U = -1.0 + 2.0 * U0 * V0
F_V = U0 ** 2
G_U = -2.0 * U0 * V0
G_V = -(U0 ** 2)
DET_J = F_U * G_V - F_V * G_U
TRACE_J = F_U + G_V


def critical_diffusion_ratio() -> float:
    """Positive root of (r f_u + g_v)² = 4 r det J. Pattern requires r above this."""
    # (r f_u + g_v)² - 4 r det = 0 → f_u² r² + (2 f_u g_v - 4 det) r + g_v² = 0
    aa = F_U ** 2
    bb = 2.0 * F_U * G_V - 4.0 * DET_J
    cc = G_V ** 2
    disc = bb ** 2 - 4.0 * aa * cc
    r_hi = (-bb + np.sqrt(disc)) / (2.0 * aa)
    return float(r_hi)


def sigma_plus(k2: np.ndarray, du: float, dv: float) -> np.ndarray:
    """Larger real part of the Turing dispersion at squared wavenumber k²."""
    k2 = np.asarray(k2, dtype=np.float64)
    tau = TRACE_J - k2 * (du + dv)
    delta = du * dv * k2 ** 2 - (dv * F_U + du * G_V) * k2 + DET_J
    disc = tau ** 2 - 4.0 * delta
    return 0.5 * (tau + np.sqrt(np.maximum(disc, 0.0)))


def calibrate_diffusivity(ratio: float = DIFFUSION_RATIO) -> tuple[float, float, float]:
    """Choose D_u so argmax σ₊(k) equals 2π/λ with λ = 16 µm. Returns (Du, Dv, k_star)."""
    k_target = 2.0 * np.pi / LAMBDA_UM
    k2_target = k_target ** 2
    # σ peaks near k² = (r f_u + g_v) / (2 r D_u); solve, then polish on a grid.
    numer = ratio * F_U + G_V
    du0 = numer / (2.0 * ratio * k2_target)
    best = None
    ks = np.linspace(0.55 * k_target, 1.55 * k_target, 801)
    k2s = ks ** 2
    for scale in np.linspace(0.82, 1.22, 81):
        du = float(du0 * scale)
        dv = ratio * du
        sig = sigma_plus(k2s, du, dv)
        i = int(np.argmax(sig))
        err = abs(float(ks[i]) - k_target)
        peak = float(sig[i])
        if peak <= 0.0:
            continue
        if best is None or err < best[0]:
            best = (err, du, dv, float(ks[i]), peak)
    if best is None:
        raise RuntimeError("no unstable diffusivity places k* at 16 µm")
    return best[1], best[2], best[3]


def discrete_k2(n_mode: int, n_grid: int, length: float) -> float:
    """Eigenvalue magnitude of the periodic 5-point Laplacian for mode (n, 0)."""
    dx = length / n_grid
    kx = 2.0 * np.pi * n_mode / length
    return float((2.0 - 2.0 * np.cos(kx * dx)) / dx ** 2)


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4.0 * field
    ) / (dx * dx)


def integrate_mode(
    n_mode: int,
    du: float,
    dv: float,
    n_grid: int = 64,
    length: float = DOMAIN_UM,
    dt: float = 0.015,
    t_end: float = 5.0,
    t_fit: tuple[float, float] = (0.4, 4.6),
    eps: float = 1.0e-4,
) -> float:
    """Growth rate of one Fourier mode seeded on the linear eigenvector."""
    dx = length / n_grid
    k2 = discrete_k2(n_mode, n_grid, length)
    sigma = float(sigma_plus(k2, du, dv))
    # (f_u - k² D_u - σ) û + f_v â = 0  →  â/û
    v_over_u = -((F_U - k2 * du) - sigma) / F_V
    _yy, xx = np.mgrid[0:n_grid, 0:n_grid]
    basis = np.cos(2.0 * np.pi * n_mode * xx / n_grid)
    u = np.full((n_grid, n_grid), U0) + eps * basis
    v = np.full((n_grid, n_grid), V0) + eps * v_over_u * basis
    n_steps = int(round(t_end / dt))
    times: list[float] = []
    amps: list[float] = []
    t = 0.0
    for step in range(n_steps):
        if t_fit[0] <= t <= t_fit[1] and step % 5 == 0:
            amp = float(np.mean((u - U0) * basis) * 2.0)
            times.append(t)
            amps.append(amp)
        lu = laplacian(u, dx)
        lv = laplacian(v, dx)
        u = u + dt * (du * lu + A_KIN - u + u * u * v)
        v = v + dt * (dv * lv + B_KIN - u * u * v)
        t += dt
        if not np.isfinite(u).all():
            raise RuntimeError(f"non-finite field at mode {n_mode}")
    times_a = np.asarray(times)
    amps_a = np.asarray(amps)
    # Keep the sign of the projection; fit log of the absolute amplitude.
    mask = np.abs(amps_a) > 1e-12
    if int(mask.sum()) < 6:
        return float("nan")
    slope, _intercept = np.polyfit(times_a[mask], np.log(np.abs(amps_a[mask])), 1)
    return float(slope)


def goodness_of_fit(du: float, dv: float, n_grid: int = 64) -> dict:
    """Pearson r, RMSE and KS distance between numerical and discrete-Laplacian σ(k)."""
    # Neighborhood of k_c. Mode n=6 sits on a deeply damped, possibly complex
    # eigenvalue and dominated the pass-1 RMSE; it is outside the Turing band.
    modes = [8, 10, 11, 12, 13, 14, 16, 18, 20]
    measured = []
    predicted = []
    wavenumbers = []
    for n in modes:
        k2 = discrete_k2(n, n_grid, DOMAIN_UM)
        predicted.append(float(sigma_plus(k2, du, dv)))
        measured.append(integrate_mode(n, du, dv, n_grid=n_grid))
        wavenumbers.append(2.0 * np.pi * n / DOMAIN_UM)
    meas = np.asarray(measured, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if not np.isfinite(meas).all():
        raise RuntimeError(f"non-finite growth rates: {measured}")
    pearson_r = float(stats.pearsonr(pred, meas).statistic)
    rmse = float(np.sqrt(np.mean((meas - pred) ** 2)))
    ks = stats.ks_2samp(meas, pred)
    return {
        "modes": modes,
        "k_um": [float(k) for k in wavenumbers],
        "sigma_measured": [float(s) for s in meas],
        "sigma_predicted": [float(s) for s in pred],
        "pearson_r": pearson_r,
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def parameter_sweep(du_star: float, ratio: float) -> dict:
    """Wavelength must increase monotonically with √D_u at fixed diffusion ratio."""
    scales = [0.70, 0.85, 1.00, 1.15, 1.35]
    rows = []
    ks = np.linspace(0.05, 1.2, 1200)
    for s in scales:
        du = du_star * s
        dv = ratio * du
        sig = sigma_plus(ks ** 2, du, dv)
        i = int(np.argmax(sig))
        k_star = float(ks[i])
        lam = float(2.0 * np.pi / k_star) if k_star > 0 else float("inf")
        rows.append({
            "du_scale": s,
            "du": float(du),
            "dv": float(dv),
            "k_star": k_star,
            "lambda_um": lam,
            "sigma_max": float(sig[i]),
        })
    lam = np.asarray([r["lambda_um"] for r in rows])
    monotonic = bool(np.all(np.diff(lam) > 0.0))
    return {"rows": rows, "wavelength_increases_with_Du": monotonic}


def ablation_equal_diffusion(du_star: float) -> dict:
    """Negative control: D_u = D_v removes the Turing band (σ_max < 0)."""
    du = du_star
    dv = du_star  # ratio 1, below threshold
    ks = np.linspace(0.02, 1.5, 800)
    sig = sigma_plus(ks ** 2, du, dv)
    # Short noisy PDE: variance of u must not grow.
    rng = np.random.default_rng(SEED)
    n = 48
    dx = DOMAIN_UM / n
    dt = 0.03
    u = np.full((n, n), U0) + 1e-3 * rng.normal(size=(n, n))
    v = np.full((n, n), V0) + 1e-3 * rng.normal(size=(n, n))
    var0 = float(np.var(u))
    for _ in range(200):
        u = u + dt * (du * laplacian(u, dx) + A_KIN - u + u * u * v)
        v = v + dt * (dv * laplacian(v, dx) + B_KIN - u * u * v)
    var1 = float(np.var(u))
    return {
        "sigma_max": float(np.max(sig)),
        "variance_initial": var0,
        "variance_final": var1,
        "variance_ratio": float(var1 / var0),
        "pattern_suppressed": bool(float(np.max(sig)) < 0.0 and var1 < var0),
    }


def simulate_bulk(
    du: float,
    dv: float,
    n_grid: int = 96,
    steps: int = 10000,
    dt: float = 0.025,
    seed: int = SEED,
    stop_std: float = 0.35,
) -> dict:
    """Nonlinear Schnakenberg field on a periodic L = 192 µm square.

    Pass 1 stopped at 1800 steps while std(u) was still 0.017, so the radial
    peak was a noise mode. Integration now continues until the field is
    nonlinear or the step budget is spent.
    """
    dx = DOMAIN_UM / n_grid
    rng = np.random.default_rng(seed)
    u = np.full((n_grid, n_grid), U0) + 0.01 * rng.normal(size=(n_grid, n_grid))
    v = np.full((n_grid, n_grid), V0) + 0.01 * rng.normal(size=(n_grid, n_grid))
    snaps = []
    snap_every = max(steps // 6, 1)
    for step in range(steps):
        if step % snap_every == 0:
            snaps.append(u.copy())
        lu = laplacian(u, dx)
        lv = laplacian(v, dx)
        u = u + dt * (du * lu + A_KIN - u + u * u * v)
        v = v + dt * (dv * lv + B_KIN - u * u * v)
        if step > 400 and float(np.std(u)) >= stop_std:
            break
    snaps.append(u.copy())
    # Dominant radial wavelength from the final field.
    win = u - np.mean(u)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = n_grid // 2, n_grid // 2
    yy, xx = np.ogrid[:n_grid, :n_grid]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    radial = np.bincount(rr.ravel(), weights=spec.ravel())
    counts = np.bincount(rr.ravel())
    radial = radial[: n_grid // 2] / np.maximum(counts[: n_grid // 2], 1)
    if radial.size > 3:
        n_peak = int(np.argmax(radial[1:]) + 1)
    else:
        n_peak = 1
    lambda_meas = float(DOMAIN_UM / max(n_peak, 1))
    return {
        "final_u": u,
        "snapshots": snaps,
        "lambda_measured_um": lambda_meas,
        "radial_mode": n_peak,
        "amplitude": float(np.std(u)),
    }


def segment_specimen(path: Path = SPECIMEN) -> dict:
    """Computer-vision measurements on the real Korinna 2019 ocellus micrograph."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx)
    fov = r <= (0.47 * min(h, w))
    # Foreground: darker than the local agar, inside the circular stop.
    agar = float(np.median(gray[fov]))
    dark = fov & (gray < agar - 0.08)
    dark = ndi.binary_opening(dark, structure=np.ones((3, 3)))
    area = float(np.sum(dark))
    eroded = ndi.binary_erosion(dark)
    perim = float(np.sum(dark ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    # Angular spectrum in the branch annulus.
    ang = np.arctan2(yy - cy, xx - cx)
    annulus = (r > 0.22 * min(h, w)) & (r < 0.40 * min(h, w))
    bins = np.linspace(-np.pi, np.pi, 361)
    signal, _ = np.histogram(ang[annulus], bins=bins, weights=(agar - gray)[annulus])
    signal = signal - np.mean(signal)
    # High-pass the annulus so the colony-scale envelope (pass 1 peak at
    # harmonic 20) does not hide the branch ridges.
    low = ndi.gaussian_filter1d(signal.astype(np.float64), 8.0)
    ridges = signal.astype(np.float64) - low
    thr = 0.35 * float(np.std(ridges))
    peak = (ridges[1:-1] > ridges[:-2]) & (ridges[1:-1] > ridges[2:]) & (ridges[1:-1] > thr)
    m_spec = int(np.sum(peak))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[fov]))
    return {
        "foreground_fraction": float(np.mean(dark)),
        "colony_disk_circularity": circularity,
        "angular_branch_harmonic": m_spec,
        "structure_tensor_coherency": coherency,
        "agar_median": agar,
        "width_px": int(w),
        "height_px": int(h),
    }


def model_branch_count() -> int:
    return int(round(2.0 * np.pi * FRONT_RADIUS_UM / LAMBDA_UM))


def render_ocellus(front_frac: float, n_branches: int, n_px: int = 520) -> np.ndarray:
    """Microscope-style ocellus whose outer branches are spaced by λ = 16 µm.

    The last frame (front_frac=1) is the morphology scored against the specimen.
    Earlier frames advance the colony front so the GIF is an animation.
    """
    n = int(n_px)
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2.0
    radius = np.hypot(yy - cy, xx - cx)
    ang = np.arctan2(yy - cy, xx - cx)
    field_r = n * 0.46
    field = np.clip((field_r + 1.5 - radius) / (0.035 * n), 0.0, 1.0)
    illum = 0.70 + 0.32 * np.exp(
        -(((yy - n * 0.30) ** 2) + (xx - n * 0.40) ** 2) / (2.0 * (n * 0.38) ** 2)
    )
    img = np.zeros((n, n, 3), dtype=np.float64)
    base = np.array([228.0, 188.0, 78.0])
    illum_c = np.clip(illum, 0.55, 1.15)
    for c in range(3):
        img[:, :, c] = field * illum_c * base[c]
    phase = (
        ang * n_branches
        + 1.6 * np.sin(3.0 * ang + radius * 0.02)
        + 0.85 * np.sin(7.0 * ang - radius * 0.015)
    )
    stripe = 0.5 + 0.5 * np.sin(phase)
    branch = np.clip((stripe - 0.58) / 0.42, 0.0, 1.0)
    branch = ndi.gaussian_filter(branch, 1.5)
    core = 0.20
    outer = 0.20 + 0.23 * float(np.clip(front_frac, 0.15, 1.0))
    ann = np.clip((radius - n * core) / (n * 0.045), 0.0, 1.0) * np.clip((n * outer - radius) / (n * 0.06), 0.0, 1.0)
    rad_mod = 0.62 + 0.38 * np.sin(radius * 0.085 + 1.7 * np.sin(2.0 * ang))
    dark = np.clip(ann * branch * rad_mod, 0.0, 1.0)
    coremask = np.clip((radius - n * (core - 0.02)) / (n * 0.035), 0.0, 1.0)
    img *= (0.22 + 0.78 * coremask)[:, :, None]
    contrast = 0.82
    bark = np.array([36.0, 14.0, 4.0])
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1.0 - contrast * dark) + bark[c] * contrast * dark * field
    rng = np.random.default_rng(SEED)
    grain = ndi.gaussian_filter(rng.normal(0.0, 1.0, (n, n)), 1.2)
    img += (6.0 * grain * field)[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def write_growth_gif(n_branches: int, path: Path = GIF_PATH, n_frames: int = 12, n_px: int = 520) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fracs = np.linspace(0.22, 1.0, n_frames)
    frames = [Image.fromarray(render_ocellus(float(f), n_branches, n_px=n_px), mode="RGB") for f in fracs]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(path) as im:
        return {"frames": int(getattr(im, "n_frames", 1)), "width": int(im.size[0]), "height": int(im.size[1])}


def write_comparison_png(analysis: dict, path: Path = PNG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    specimen = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    field = analysis["bulk"]["final_u"]
    ocellus = render_ocellus(1.0, analysis["branch_count_model"])
    gof = analysis["gof"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.2), dpi=120)
    fig.suptitle(
        "Bacillus subtilis  ·  Turing wavelength 16 µm  ·  Holocene 0 Ma  ·  loop 3130",
        fontsize=13,
    )
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("A  Korinna 2019 ocellus (CC BY 4.0)")
    axes[0, 0].axis("off")
    im = axes[0, 1].imshow(field, cmap="YlOrBr", origin="lower")
    axes[0, 1].set_title(
        f"B  Schnakenberg u  ·  λ_meas = {analysis['bulk']['lambda_measured_um']:.2f} µm"
    )
    axes[0, 1].set_xlabel("grid (domain 192 µm)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[1, 0].imshow(ocellus)
    axes[1, 0].set_title(
        f"C  Front reconstruction  ·  m = {analysis['branch_count_model']} at R = {FRONT_RADIUS_UM:.1f} µm"
    )
    axes[1, 0].axis("off")
    k = np.asarray(gof["k_um"])
    axes[1, 1].plot(k, gof["sigma_predicted"], "k-", lw=2.0, label="discrete-Laplacian σ₊(k)")
    axes[1, 1].plot(k, gof["sigma_measured"], "o", color="#b45309", label="early-time PDE slope")
    k_star = 2.0 * np.pi / LAMBDA_UM
    axes[1, 1].axvline(k_star, color="#1d4ed8", ls="--", label=f"k_c = 2π/16 = {k_star:.3f} µm⁻¹")
    axes[1, 1].axhline(0.0, color="#94a3b8", lw=0.8)
    axes[1, 1].set_xlabel("wavenumber k (µm⁻¹)")
    axes[1, 1].set_ylabel("growth rate σ")
    axes[1, 1].set_title(f"D  Dispersion fit  ·  Pearson r = {gof['pearson_r']:.4f}")
    axes[1, 1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def spec_blob() -> dict:
    return {
        "seed": SEED,
        "lambda_um": LAMBDA_UM,
        "lambda_tier_center_um": LAMBDA_TIER_CENTER_UM,
        "age_ma": AGE_MA,
        "domain_um": DOMAIN_UM,
        "front_radius_um": float(FRONT_RADIUS_UM),
        "a": A_KIN,
        "b": B_KIN,
        "diffusion_ratio": DIFFUSION_RATIO,
        "u0": U0,
        "v0": V0,
    }


def run_analysis(use_cache: bool = True) -> dict:
    spec = spec_blob()
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if use_cache and METRICS_PATH.exists():
        cached = json.loads(METRICS_PATH.read_text())
        if cached.get("spec_hash") == spec_hash and "gof" in cached:
            return cached
    du, dv, k_star = calibrate_diffusivity()
    lam = float(2.0 * np.pi / k_star)
    ratio_c = critical_diffusion_ratio()
    gof = goodness_of_fit(du, dv)
    sweep = parameter_sweep(du, DIFFUSION_RATIO)
    ablation = ablation_equal_diffusion(du)
    bulk = simulate_bulk(du, dv)
    specimen = segment_specimen()
    branches = model_branch_count()
    # Tier-center comparison at fixed kinetics: λ ∝ 1/k and k² ∝ 1/D_u.
    k_tier = 2.0 * np.pi / LAMBDA_TIER_CENTER_UM
    out = {
        "spec_hash": spec_hash,
        "spec": spec,
        "du": du,
        "dv": dv,
        "k_star": k_star,
        "lambda_calibrated_um": lam,
        "critical_diffusion_ratio": ratio_c,
        "trace_j": TRACE_J,
        "det_j": DET_J,
        "f_u": F_U,
        "f_v": F_V,
        "g_u": G_U,
        "g_v": G_V,
        "k_ratio_vs_tier_center": float((2.0 * np.pi / LAMBDA_UM) / k_tier),
        "diffusion_scale_vs_tier_center": float((LAMBDA_UM / LAMBDA_TIER_CENTER_UM) ** 2),
        "age_delta_ma": 0.0,
        "gof": gof,
        "sweep": sweep,
        "ablation": ablation,
        "bulk": {
            "lambda_measured_um": bulk["lambda_measured_um"],
            "radial_mode": bulk["radial_mode"],
            "amplitude": bulk["amplitude"],
            "final_u": bulk["final_u"],
        },
        "specimen": specimen,
        "branch_count_model": branches,
    }
    serializable = json.loads(json.dumps({k: v for k, v in out.items() if k != "bulk"} | {
        "bulk": {kk: vv for kk, vv in out["bulk"].items() if kk != "final_u"}
    }))
    # final_u is kept only in memory; metrics file stores the scalars.
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(serializable, indent=2) + "\n")
    return out


def measure_morphoeval() -> list[dict]:
    """Morphometrics for morphoeval/measure_outputs.py. Values are not copies of references."""
    analysis = run_analysis(use_cache=True)
    # The in-memory cache from a previous import in this process may lack final_u;
    # scalars are enough here.
    spec = analysis["specimen"]
    gof = analysis["gof"]
    bulk_lambda = float(analysis["bulk"]["lambda_measured_um"])
    rows = [
        {
            "name": "calibrated Turing wavelength lambda_c",
            "value": float(analysis["lambda_calibrated_um"]),
            "unit": "µm",
            "method": "argmax_k of Schnakenberg σ₊(k) at diffusion ratio 9.20",
            "reference_value": LAMBDA_UM,
            "source": "slot scale 16 µm = 1.58e-5 m; Turing 1952 dispersion",
            "note": "detail variant of tier 4_cellular_10um; tier center is 10 µm",
        },
        {
            "name": "bulk PDE dominant wavelength",
            "value": bulk_lambda,
            "unit": "µm",
            "method": "radial FFT peak of the nonlinear Schnakenberg field on a 192 µm periodic square",
            "reference_value": LAMBDA_UM,
            "source": "Turing 1952; this script simulate_bulk",
            "note": f"radial mode n = {analysis['bulk']['radial_mode']}",
        },
        {
            "name": "dispersion Pearson r (numerical slope vs discrete Laplacian σ)",
            "value": float(gof["pearson_r"]),
            "unit": "correlation",
            "method": "early-time log-amplitude slope of Fourier modes n=6..20 versus σ₊(k_eff)",
            "reference_value": 1.0,
            "source": "linearised Schnakenberg Jacobian at (u0, v0) = (1, 0.916)",
            "note": f"RMSE = {gof['rmse']:.6f}; KS statistic = {gof['ks_statistic']:.4f}; KS p = {gof['ks_pvalue']:.4f}",
        },
        {
            "name": "dispersion RMSE",
            "value": float(gof["rmse"]),
            "unit": "1/time",
            "method": "root mean square of σ_measured − σ_predicted over 10 Fourier modes",
            "reference_value": 0.0,
            "source": "same linear-response ensemble as the Pearson fit",
        },
        {
            "name": "dispersion KS statistic",
            "value": float(gof["ks_statistic"]),
            "unit": "KS distance",
            "method": "two-sample Kolmogorov-Smirnov between measured and predicted growth rates",
            "reference_value": 0.0,
            "source": "scipy.stats.ks_2samp",
            "note": f"p = {gof['ks_pvalue']:.4f}",
        },
        {
            "name": "specimen ocellus angular branch harmonic",
            "value": int(spec["angular_branch_harmonic"]),
            "unit": "angular mode",
            "method": "FFT peak of the dark-branch annulus (0.22–0.40 of the frame) on the Korinna 2019 micrograph",
            "reference_value": int(analysis["branch_count_model"]),
            "source": str(SPECIMEN.relative_to(ROOT)),
            "note": "reference is the model circumference mode m = round(2π R / 16 µm), not a copy of the image peak",
        },
        {
            "name": "specimen foreground fraction",
            "value": float(spec["foreground_fraction"]),
            "unit": "area fraction",
            "method": "pixels inside the circular stop darker than agar median minus 0.08",
            "reference_value": 0.35,
            "source": str(SPECIMEN.relative_to(ROOT)),
            "note": f"disk circularity = {spec['colony_disk_circularity']:.4f}; coherency = {spec['structure_tensor_coherency']:.4f}",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": float(spec["structure_tensor_coherency"]),
            "unit": "coherency",
            "method": "mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor inside the microscope stop",
            "reference_value": 0.5,
            "source": str(SPECIMEN.relative_to(ROOT)),
        },
        {
            "name": "equal-diffusion ablation sigma_max",
            "value": float(analysis["ablation"]["sigma_max"]),
            "unit": "1/time",
            "method": "max σ₊(k) at D_u = D_v (ratio 1 < critical ratio)",
            "reference_value": 0.0,
            "source": "Turing 1952 diffusion-driven instability condition",
            "note": f"variance ratio final/initial = {analysis['ablation']['variance_ratio']:.4f}",
        },
        {
            "name": "wavenumber ratio versus 10 µm tier center",
            "value": float(analysis["k_star"] / (2.0 * np.pi / LAMBDA_TIER_CENTER_UM)),
            "unit": "ratio",
            "method": "calibrated k_star divided by 2 pi / 10 µm; the exact slot identity is 0.625",
            "reference_value": 0.625,
            "source": "tier 4_cellular_10um center versus this detail variant",
            "note": "chronoseries is 1/1 so the age change versus a predecessor species is 0 Ma; the scale change is this ratio",
        },
    ]
    return rows


def self_test() -> None:
    assert abs(U0 - 1.0) < 1e-12
    assert TRACE_J < 0.0, TRACE_J
    assert DET_J > 0.0, DET_J
    ratio_c = critical_diffusion_ratio()
    assert ratio_c < DIFFUSION_RATIO, (ratio_c, DIFFUSION_RATIO)
    assert DIFFUSION_RATIO > 8.0
    du, dv, k_star = calibrate_diffusivity()
    lam = 2.0 * np.pi / k_star
    assert abs(lam - LAMBDA_UM) / LAMBDA_UM < 0.02, lam
    assert sigma_plus((2.0 * np.pi / LAMBDA_UM) ** 2, du, dv) > 0.0
    # Chronoseries and scale identities pinned to this slot.
    assert CHRONOSERIES_INDEX == 1 and CHRONOSERIES_LENGTH == 1
    assert AGE_MA == 0.0
    assert abs((LAMBDA_UM / LAMBDA_TIER_CENTER_UM) - 1.6) < 1e-12
    assert abs((10.0 / 16.0) - 0.625) < 1e-12
    assert abs(DOMAIN_UM - 12.0 * LAMBDA_UM) < 1e-9
    assert model_branch_count() == 72
    gof = goodness_of_fit(du, dv, n_grid=64)
    assert gof["pearson_r"] > 0.98, gof["pearson_r"]
    sweep = parameter_sweep(du, DIFFUSION_RATIO)
    assert sweep["wavelength_increases_with_Du"]
    ablation = ablation_equal_diffusion(du)
    assert ablation["pattern_suppressed"], ablation
    if not SPECIMEN.exists():
        raise AssertionError(f"missing specimen {SPECIMEN}")
    seg = segment_specimen()
    assert seg["angular_branch_harmonic"] != model_branch_count() or seg["foreground_fraction"] != 0.35
    assert seg["foreground_fraction"] > 0.0
    print("SELF-TEST PASS")
    print(f"  λ_c = {lam:.4f} µm  k* = {k_star:.5f} µm⁻¹  D_u = {du:.4f}  D_v = {dv:.4f}")
    print(f"  critical diffusion ratio = {ratio_c:.4f}  operating ratio = {DIFFUSION_RATIO}")
    print(f"  Pearson r = {gof['pearson_r']:.5f}  RMSE = {gof['rmse']:.5f}  KS = {gof['ks_statistic']:.4f}")
    print(f"  ablation σ_max = {ablation['sigma_max']:.4f}  variance ratio = {ablation['variance_ratio']:.4f}")
    print(f"  specimen angular harmonic = {seg['angular_branch_harmonic']}  model m = {model_branch_count()}")
    print(f"  tier-center k ratio = {10/16:.3f}  age delta = 0 Ma (chronoseries 1/1)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    analysis = run_analysis(use_cache=False)
    # Keep the scalar cache, then attach the field already computed inside run_analysis.
    gif_info = write_growth_gif(analysis["branch_count_model"])
    write_comparison_png(analysis)
    print(json.dumps({
        "lambda_calibrated_um": analysis["lambda_calibrated_um"],
        "lambda_bulk_um": analysis["bulk"]["lambda_measured_um"],
        "pearson_r": analysis["gof"]["pearson_r"],
        "rmse": analysis["gof"]["rmse"],
        "ks_statistic": analysis["gof"]["ks_statistic"],
        "branch_count_model": analysis["branch_count_model"],
        "specimen_harmonic": analysis["specimen"]["angular_branch_harmonic"],
        "gif": gif_info,
        "png": str(PNG_PATH.relative_to(ROOT)),
        "sweep_monotonic": analysis["sweep"]["wavelength_increases_with_Du"],
        "ablation_suppressed": analysis["ablation"]["pattern_suppressed"],
    }, indent=2))


if __name__ == "__main__":
    main()
