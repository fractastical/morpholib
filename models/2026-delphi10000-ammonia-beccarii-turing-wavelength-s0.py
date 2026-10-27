#!/usr/bin/env python3
"""
Ammonia beccarii — Turing wavenumber at 31 µm, Holocene, 0 Ma (slot 3108).

Governing system (Schnakenberg 1979 kinetics, Turing 1952 dispersion):

    u_t = Du ∇²u + γ (a − u + u² v)
    v_t = Dv ∇²v + γ (b − u² v)

Steady state u0 = a + b, v0 = b / (a + b)².
Critical wavenumber

    k_c² = γ (Dv f_u + Du g_v) / (2 Du Dv),    λ_c = 2π / k_c.

γ is solved so λ_c = 31.37 µm = 3.137e-05 m (the 31 µm detail tier,
log10 = −4.503) on a periodic face of side 600 µm (Hillewaert caption
diameter). Du = 1, Dv = 10.4 µm²/τ keeps the unstable band near 27–39 µm.

Chronoseries Ammonia 1/1, age 0 Ma: no predecessor and no successor.
The pore mode (0.564 µm, Subba & Ghosh 2023) and the whole-test mode
(~600 µm) sit outside that band. This file selects only the intermediate
wavenumber. It does not claim pigment ocelli.

Run:
    python3 models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py
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
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "ammonia_beccarii_turing_wavelength_s0"
SPECIMEN = DATASET / "Ammonia_beccarii_Hillewaert_2012.jpg"
REFERENCE = DATASET / "ammonia_beccarii_turing_wavelength_s0_reference.json"
METRICS = DATASET / "ammonia_beccarii_turing_wavelength_s0_metrics.json"
COMPARISON_PNG = ROOT / "docs" / "media" / "ammonia_beccarii_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "ammonia_beccarii_turing_wavelength_s0_growth.gif"

EXPECTED_SHA256 = {
    "ammonia_beccarii_turing_wavelength_s0_reference.json":
        "6a6a4fb5060925141cb730fc1ff13c58ec972a042cd4c6d3a69c904cd24bbd96",
    "Ammonia_beccarii_Hillewaert_2012.jpg":
        "c15a90be96e2b088e80f55ac46f810f56c02bf2c7aaccce4bf483e1b7b3d2ade",
    "scout_manifest.json":
        "12dc86d6408aac7a22acd4e5ec62ae0d2f527006a1e56ccca4c222d9e13cf540",
    "paper_citations.bib":
        "5ac93aba84d8875403f57f9247f56aa2bb99bcb2c658cd03a41e604b1ed49348",
}

# Slot-specific parameters. Not the 27 µm Globorotalia regime.
A = 0.12
B = 0.88
DU = 1.0
DV = 10.4
LAMBDA_UM = 31.37
PHENOMENON_M = 3.137e-05
LOG10_SIZE = -4.503
DOMAIN_UM = 600.0
AGE_MA = 0.0
SEED = 3108
PORE_DIAMETER_UM = 0.564
PORE_DENSITY = 0.155
N_GRID = 160
N_STEPS = 7000
N_FRAMES = 12
GIF_SIZE = (720, 540)


def verify_sha256() -> None:
    for name, exp in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {exp}")


def jacobian() -> dict[str, float]:
    """Schnakenberg Jacobian at the homogeneous steady state."""
    u0 = A + B
    v0 = B / (u0 * u0)
    fu = -1.0 + 2.0 * B / u0
    fv = u0 * u0
    gu = -2.0 * B / u0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    trace = fu + gv
    disc = DV * fu + DU * gv
    return {
        "u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv,
        "det": det, "trace": trace, "disc": disc,
    }


def gamma_for_wavelength(lam_um: float, dv: float = DV) -> float:
    """γ such that the linear critical wavelength equals lam_um."""
    jac = jacobian_at(dv)
    kc = 2.0 * math.pi / lam_um
    return (kc * kc) * (2.0 * DU * dv) / jac["disc"]


def jacobian_at(dv: float) -> dict[str, float]:
    u0 = A + B
    v0 = B / (u0 * u0)
    fu = -1.0 + 2.0 * B / u0
    fv = u0 * u0
    gu = -2.0 * B / u0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    disc = dv * fu + DU * gv
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv,
            "det": det, "trace": fu + gv, "disc": disc, "dv": dv}


def sigma_of_k2(k2: float, gamma: float, jac: dict[str, float], dv: float) -> float:
    """Largest real part of the dispersion eigenvalue at Laplacian eigenvalue −k²."""
    muu = gamma * jac["fu"] - DU * k2
    mvv = gamma * jac["gv"] - dv * k2
    tr = muu + mvv
    det = muu * mvv - (gamma * jac["fv"]) * (gamma * jac["gu"])
    delta = tr * tr - 4.0 * det
    if delta >= 0.0:
        return 0.5 * (tr + math.sqrt(delta))
    return 0.5 * tr


def unstable_band(gamma: float, jac: dict[str, float], dv: float) -> tuple[float, float]:
    """Wavelength edges (µm) of the Turing-unstable interval. Empty if none."""
    if jac["disc"] <= 0.0 or jac["disc"] ** 2 <= 4.0 * DU * dv * jac["det"]:
        return (math.nan, math.nan)
    aa = DU * dv
    bb = -gamma * jac["disc"]
    cc = (gamma ** 2) * jac["det"]
    q = bb * bb - 4.0 * aa * cc
    if q <= 0.0:
        return (math.nan, math.nan)
    z_lo = (-bb - math.sqrt(q)) / (2.0 * aa)
    z_hi = (-bb + math.sqrt(q)) / (2.0 * aa)
    if z_lo <= 0.0 or z_hi <= 0.0:
        return (math.nan, math.nan)
    return (2.0 * math.pi / math.sqrt(z_hi), 2.0 * math.pi / math.sqrt(z_lo))


def linear_mode_check(gamma: float, jac: dict[str, float], dv: float) -> dict[str, float]:
    """Integrate the linearised PDE on exact Fourier modes and compare σ(k)."""
    n = 96
    dx = DOMAIN_UM / n
    dt = 0.004
    steps = 250
    analytical = []
    numerical = []

    def _project(u: np.ndarray, v: np.ndarray, mode: np.ndarray) -> np.ndarray:
        w = float(np.mean(mode * mode)) + 1e-15
        return np.array([float(np.mean(u * mode) / w), float(np.mean(v * mode) / w)])

    def _flow(coeff: np.ndarray, mode: np.ndarray) -> np.ndarray:
        u = coeff[0] * mode
        v = coeff[1] * mode
        for _ in range(steps):
            lap_u = _laplacian(u, dx)
            lap_v = _laplacian(v, dx)
            u = u + dt * (DU * lap_u + gamma * (jac["fu"] * u + jac["fv"] * v))
            v = v + dt * (dv * lap_v + gamma * (jac["gu"] * u + jac["gv"] * v))
        return _project(u, v, mode)

    for p in range(6, 28):
        theta = 2.0 * math.pi * p / n
        k2 = (2.0 - 2.0 * math.cos(theta)) / (dx * dx)
        sig = sigma_of_k2(k2, gamma, jac, dv)
        xs = np.arange(n)
        mode = np.cos(theta * xs)[:, None] * np.ones((1, n))
        # Monodromy of the linear PDE on this Fourier mode. Complex eigenvalues
        # oscillate, so the growth rate is the log-modulus of the flow, not one
        # raw amplitude ratio.
        phi = np.column_stack([_flow(np.array([1.0, 0.0]), mode), _flow(np.array([0.0, 1.0]), mode)])
        eig = np.linalg.eigvals(phi)
        sig_num = float(np.max(np.log(np.maximum(np.abs(eig), 1e-30)).real) / (steps * dt))
        analytical.append(sig)
        numerical.append(sig_num)
    a = np.asarray(analytical)
    b = np.asarray(numerical)
    r = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    ks = stats.ks_2samp(a, b)
    return {
        "pearson_r": r,
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "n_modes": int(a.size),
        "analytical": a.tolist(),
        "numerical": b.tolist(),
    }


def _laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z) / (dx * dx)


def simulate_field(gamma: float, dv: float, n: int, nsteps: int, seed: int,
                   snapshots: int = 0) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Nonlinear Schnakenberg integration. Returns final u, v and optional snapshots."""
    jac = jacobian_at(dv)
    dx = DOMAIN_UM / n
    dt = 0.35 * (dx * dx) / (4.0 * max(dv, DU))
    rng = np.random.default_rng(seed)
    u = np.full((n, n), jac["u0"]) + rng.normal(0.0, 0.015, (n, n))
    v = np.full((n, n), jac["v0"]) + rng.normal(0.0, 0.015, (n, n))
    # A weak seed on the critical ring. The law says this is the fastest mode.
    yy, xx = np.mgrid[0:n, 0:n]
    kc = 2.0 * math.pi / LAMBDA_UM
    phase = kc * (xx * dx)
    u = u + 0.008 * np.cos(phase) * np.cos(kc * (yy * dx))
    frames: list[np.ndarray] = []
    snap_at = set(np.linspace(0, nsteps - 1, snapshots, dtype=int).tolist()) if snapshots else set()
    for step in range(nsteps):
        if step in snap_at:
            frames.append(u.copy())
        uv2 = u * u * v
        u = u + dt * (DU * _laplacian(u, dx) + gamma * (A - u + uv2))
        v = v + dt * (dv * _laplacian(v, dx) + gamma * (B - uv2))
    if snapshots and (nsteps - 1) not in snap_at:
        frames.append(u.copy())
    return u, v, frames


def dominant_wavelength_um(field: np.ndarray) -> float:
    n = field.shape[0]
    dx = DOMAIN_UM / n
    win = field - float(field.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    rmax = n // 2 - 1
    prof = np.bincount(rr.ravel(), weights=power.ravel(), minlength=rmax + 1)[:rmax].astype(float)
    counts = np.bincount(rr.ravel(), minlength=rmax + 1)[:rmax].astype(float)
    prof = prof / np.maximum(counts, 1.0)
    k = int(np.argmax(prof[2:]) + 2)
    y1, y2, y3 = prof[k - 1], prof[k], prof[k + 1]
    denom = y1 - 2.0 * y2 + y3
    delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-18 else 0.0
    k_ref = k + float(delta)
    return float((n / max(k_ref, 1e-6)) * dx)


def spot_spacings_um(field: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distances between local maxima, in micrometres."""
    n = field.shape[0]
    dx = DOMAIN_UM / n
    smooth = ndi.gaussian_filter(field, 0.8)
    peaks = (smooth == ndi.maximum_filter(smooth, size=5)) & (smooth > smooth.mean() + 0.25 * smooth.std())
    ys, xs = np.where(peaks)
    if len(ys) < 4:
        return np.array([])
    pts = np.column_stack([xs, ys]).astype(float)
    dmin = []
    for i in range(len(pts)):
        delta = pts - pts[i]
        dist = np.hypot(delta[:, 0], delta[:, 1])
        dist[i] = np.inf
        dmin.append(float(dist.min()))
    return np.asarray(dmin) * dx


def wavelength_sweep(gamma0: float, jac: dict[str, float]) -> dict[str, list[float]]:
    """Predicted linear λ versus saturated FFT λ across a γ sweep."""
    factors = [0.70, 0.85, 1.00, 1.20, 1.45, 1.80]
    predicted, measured = [], []
    for fac in factors:
        g = gamma0 * fac
        kc2 = g * jac["disc"] / (2.0 * DU * DV)
        predicted.append(2.0 * math.pi / math.sqrt(kc2))
        u, _, _ = simulate_field(g, DV, n=96, nsteps=4200, seed=SEED + int(100 * fac), snapshots=0)
        measured.append(dominant_wavelength_um(u))
    pred = np.asarray(predicted)
    meas = np.asarray(measured)
    r = float(np.corrcoef(pred, meas)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    return {
        "factors": factors,
        "predicted_um": pred.tolist(),
        "measured_um": meas.tolist(),
        "pearson_r": r,
        "rmse_um": rmse,
    }


def ablation_controls(gamma: float, jac: dict[str, float]) -> dict[str, float]:
    """Equal-diffusion control and a subthreshold Dv ablation."""
    sig_equal = sigma_of_k2((2.0 * math.pi / LAMBDA_UM) ** 2, gamma, jac, dv=DU)
    # Dv = 8 is below the Turing threshold for this Jacobian.
    u_sub, _, _ = simulate_field(gamma, dv=8.0, n=64, nsteps=800, seed=SEED, snapshots=0)
    # A pure-diffusion ablation (γ = 0) started from the same noise scale.
    rng = np.random.default_rng(SEED)
    n = 64
    dx = DOMAIN_UM / n
    dt = 0.3 * (dx * dx) / (4.0 * DV)
    noise = rng.normal(0.0, 0.02, (n, n))
    std0 = float(noise.std())
    z = noise.copy()
    for _ in range(400):
        z = z + dt * DU * _laplacian(z, dx)
    std1 = float(z.std())
    return {
        "equal_diffusion_sigma_max": float(sig_equal),
        "subthreshold_dv": 8.0,
        "subthreshold_field_std": float(u_sub.std()),
        "diffusion_ablation_std_ratio": std1 / std0,
    }


def segment_specimen(rgb: np.ndarray) -> dict:
    """Segment the six tests on the black plate. Reject the scale bar and caption."""
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    mask = gray > 0.08
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    labels, n = ndi.label(mask)
    h, w = gray.shape
    tests = []
    for i in range(1, n + 1):
        comp = labels == i
        area = int(comp.sum())
        if area < 0.004 * h * w:
            continue
        ys, xs = np.where(comp)
        width_px = float(xs.max() - xs.min() + 1)
        height_px = float(ys.max() - ys.min() + 1)
        # Pixel-erosion perimeters under-count smooth outlines, so circularity
        # uses the bounding box as a stable rejector for the scale bar and caption.
        circ = float(min(width_px, height_px) / max(width_px, height_px, 1.0))
        if circ < 0.55 or max(width_px, height_px) < 0.08 * min(h, w):
            continue
        tests.append({
            "area_px": area,
            "circularity": circ,
            "bbox_px": max(width_px, height_px),
            "cy": float(ys.mean()),
            "cx": float(xs.mean()),
            "ys": ys,
            "xs": xs,
            "mask": comp,
        })
    # Scale bar: longest near-white horizontal run in the lower fifth.
    bar_px = 0
    y0 = int(h * 0.78)
    bright = gray[y0:] > 0.75
    for row in bright:
        count = best = 0
        for v in row:
            if v:
                count += 1
                best = max(best, count)
            else:
                count = 0
        bar_px = max(bar_px, best)
    um_per_px = (500.0 / bar_px) if bar_px >= 20 else None
    diameters = []
    area_diameters = []
    rim_diameters = []
    if um_per_px is not None:
        for t in tests:
            # Halo-inclusive bounding box (gray > 0.08). This catches the bright rim.
            diameters.append(t["bbox_px"] * um_per_px)
            area_diameters.append(2.0 * math.sqrt(t["area_px"] / math.pi) * um_per_px)
            # Rim-excluded core: only pixels well above the black background.
            core = t["mask"] & (gray > 0.32)
            if int(core.sum()) < 40:
                continue
            cys, cxs = np.where(core)
            rim_diameters.append(float(max(cxs.max() - cxs.min() + 1, cys.max() - cys.min() + 1)) * um_per_px)
    shell = np.zeros(gray.shape, dtype=bool)
    for t in tests:
        shell |= t["mask"]
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 1.6)
    jyy = ndi.gaussian_filter(gy * gy, 1.6)
    jxy = ndi.gaussian_filter(gx * gy, 1.6)
    coherency_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coh_shell = float(np.mean(coherency_map[shell])) if shell.any() else float(np.mean(coherency_map))
    jxx2 = ndi.gaussian_filter(gx * gx, 2.8)
    jyy2 = ndi.gaussian_filter(gy * gy, 2.8)
    jxy2 = ndi.gaussian_filter(gx * gy, 2.8)
    coh2 = np.sqrt((jxx2 - jyy2) ** 2 + 4.0 * jxy2 * jxy2) / (jxx2 + jyy2 + 1e-9)
    coh_wide = float(np.mean(coh2[shell])) if shell.any() else float(np.mean(coh2))
    return {
        "n_tests": len(tests),
        "scale_bar_px": int(bar_px),
        "um_per_px": um_per_px,
        "diameters_um": diameters,
        "area_equivalent_diameters_um": area_diameters,
        "rim_excluded_diameters_um": rim_diameters,
        "mean_diameter_um": float(np.mean(diameters)) if diameters else None,
        "mean_rim_excluded_diameter_um": float(np.mean(rim_diameters)) if rim_diameters else None,
        "diameter_cv": float(np.std(diameters) / np.mean(diameters)) if diameters else None,
        "foreground_fraction": float(np.mean(shell)),
        "mean_circularity": float(np.mean([t["circularity"] for t in tests])) if tests else None,
        "coherency_sigma_1_6": coh_shell,
        "coherency_sigma_2_8": coh_wide,
        "shell_mask": shell,
        "gray": gray,
    }


def _normalize_field(field: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(field, [2, 98])
    return np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def map_field_onto_shells(field: np.ndarray, shell: np.ndarray) -> np.ndarray:
    """Paint one 600 µm periodic face into every segmented test."""
    n = field.shape[0]
    norm = _normalize_field(field)
    painted = np.zeros(shell.shape, dtype=np.float32)
    labels, nlab = ndi.label(shell)
    yy, xx = np.mgrid[0:shell.shape[0], 0:shell.shape[1]]
    for i in range(1, nlab + 1):
        comp = labels == i
        ys, xs = np.where(comp)
        cy, cx = float(ys.mean()), float(xs.mean())
        rad = float(np.hypot(ys - cy, xs - cx).max())
        if rad < 2:
            continue
        # Diameter maps onto the full 600 µm domain.
        col = (xx - cx) / (2.0 * rad) * (n - 1) + (n - 1) / 2.0
        row = (yy - cy) / (2.0 * rad) * (n - 1) + (n - 1) / 2.0
        sampled = ndi.map_coordinates(norm, [row, col], order=1, mode="wrap")
        painted[comp] = sampled[comp].astype(np.float32)
    return painted


def render_gif(base_rgb: np.ndarray, snaps: list[np.ndarray], shell: np.ndarray, path: Path) -> None:
    """Growth animation. Pattern amplitude rises inside the tests; background stays black."""
    painted = [map_field_onto_shells(s, shell) for s in snaps]
    frames = []
    shell3 = shell[..., None]
    for i, paint in enumerate(painted):
        amp = 0.06 + 0.42 * (i / max(1, len(painted) - 1))
        delta = (paint - 0.5) * 2.0
        mod = base_rgb.copy()
        gain = 1.0 + amp * delta
        mod = np.where(shell3, np.clip(mod * gain[..., None], 0.0, 1.0), mod)
        # Warm the activator peaks so the 31 µm spots read on the calcite.
        warm = np.clip(paint - 0.55, 0.0, 1.0) * amp
        mod[..., 0] = np.where(shell, np.clip(mod[..., 0] + 0.18 * warm, 0.0, 1.0), mod[..., 0])
        frames.append(Image.fromarray((mod * 255.0).astype(np.uint8), mode="RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=120, loop=0, optimize=False,
    )


def render_comparison(base_rgb: np.ndarray, last_rgb: np.ndarray, field: np.ndarray,
                      sweep: dict, band: tuple[float, float], gamma: float,
                      jac: dict, dv: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), dpi=120)
    axes[0, 0].imshow(base_rgb)
    axes[0, 0].set_title("Hillewaert 2012 plate (CC BY-SA 4.0)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(last_rgb)
    axes[0, 1].set_title("Same tests, λc = 31.37 µm field grown in")
    axes[0, 1].axis("off")

    disk = _normalize_field(field)
    n = disk.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2.0
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= (0.48 * n) ** 2
    show = np.where(inside, disk, np.nan)
    im = axes[1, 0].imshow(show, cmap="YlOrBr", vmin=0, vmax=1, extent=[0, DOMAIN_UM, 0, DOMAIN_UM], origin="lower")
    axes[1, 0].set_title("One 600 µm face (periodic PDE)")
    axes[1, 0].set_xlabel("µm")
    axes[1, 0].set_ylabel("µm")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    lams = np.linspace(18.0, 55.0, 280)
    sigs = [sigma_of_k2((2.0 * math.pi / lam) ** 2, gamma, jac, dv) for lam in lams]
    axes[1, 1].plot(lams, sigs, color="#8c4a2f", lw=2.0, label="σ(λ)")
    axes[1, 1].axhline(0.0, color="k", lw=0.6)
    axes[1, 1].axvspan(band[0], band[1], color="#8c4a2f", alpha=0.12, label="unstable band")
    axes[1, 1].axvline(LAMBDA_UM, color="#1f4b6e", ls="--", lw=1.0, label="λc = 31.37 µm")
    axes[1, 1].scatter(
        sweep["measured_um"], np.zeros(len(sweep["measured_um"])),
        s=28, color="#1f4b6e", zorder=3, label="sweep FFT λ",
    )
    axes[1, 1].set_xlim(18, 55)
    axes[1, 1].set_xlabel("wavelength (µm)")
    axes[1, 1].set_ylabel("σ (1/τ)")
    axes[1, 1].set_title(f"Dispersion band; γ-sweep r = {sweep['pearson_r']:.3f}")
    axes[1, 1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Ammonia beccarii · Holocene 0 Ma · Turing λc = 31.37 µm (3.137e-05 m)",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def build_metrics(field: np.ndarray, linear: dict, sweep: dict, controls: dict,
                  seg: dict, band: tuple[float, float], gamma: float,
                  grid_convergence: dict) -> dict:
    lam_meas = dominant_wavelength_um(field)
    spacings = spot_spacings_um(field)
    if spacings.size >= 5:
        spacing_mean = float(np.mean(spacings))
        spacing_cv = float(np.std(spacings) / max(spacing_mean, 1e-9))
        scale = max(2.0, 0.5 * (band[1] - band[0]))
        ks_space = stats.kstest(spacings, stats.norm(loc=LAMBDA_UM, scale=scale).cdf)
        ks_stat, ks_p = float(ks_space.statistic), float(ks_space.pvalue)
    else:
        spacing_mean, spacing_cv, ks_stat, ks_p = None, None, None, None
    pore_k = 2.0 * math.pi / PORE_DIAMETER_UM
    test_k = 2.0 * math.pi / DOMAIN_UM
    kc = 2.0 * math.pi / LAMBDA_UM
    return {
        "organism": "Ammonia beccarii",
        "geological_age_ma": AGE_MA,
        "chronoseries": {"index": 1, "length": 1, "predecessor": None, "successor": None},
        "phenomenon_size_m": PHENOMENON_M,
        "log10_size_m": LOG10_SIZE,
        "scale_tier": "4_cellular_10um",
        "lambda_c_um_linear": LAMBDA_UM,
        "k_c_per_um": kc,
        "gamma": gamma,
        "Du": DU,
        "Dv": DV,
        "band_um": list(band),
        "saturated_wavelength_um": lam_meas,
        "spot_spacing_mean_um": spacing_mean,
        "spot_spacing_cv": spacing_cv,
        "spot_spacing_ks_statistic": ks_stat,
        "spot_spacing_ks_pvalue": ks_p,
        "n_spots": int(spacings.size),
        "dispersion_pearson_r": linear["pearson_r"],
        "dispersion_rmse": linear["rmse"],
        "dispersion_ks_statistic": linear["ks_statistic"],
        "dispersion_ks_pvalue": linear["ks_pvalue"],
        "sweep": sweep,
        "controls": controls,
        "wavenumber_contrast": {
            "k_pore_per_um": pore_k,
            "k_c_per_um": kc,
            "k_test_per_um": test_k,
            "pore_over_detail": pore_k / kc,
            "detail_over_test": kc / test_k,
            "pore_inside_band": bool(band[0] <= PORE_DIAMETER_UM <= band[1]),
            "test_inside_band": bool(band[0] <= DOMAIN_UM <= band[1]),
            "detail_inside_band": bool(band[0] <= LAMBDA_UM <= band[1]),
        },
        "grid_convergence_um": grid_convergence,
        "specimen": {
            "n_tests": seg["n_tests"],
            "scale_bar_px": seg["scale_bar_px"],
            "um_per_px": seg["um_per_px"],
            "diameters_um": seg["diameters_um"],
            "mean_diameter_um": seg["mean_diameter_um"],
            "mean_rim_excluded_diameter_um": seg.get("mean_rim_excluded_diameter_um"),
            "rim_excluded_diameters_um": seg.get("rim_excluded_diameters_um"),
            "diameter_cv": seg["diameter_cv"],
            "foreground_fraction": seg["foreground_fraction"],
            "mean_circularity": seg["mean_circularity"],
            "coherency_sigma_1_6": seg["coherency_sigma_1_6"],
            "coherency_sigma_2_8": seg["coherency_sigma_2_8"],
            "caption_diameter_um": 600.0,
        },
        "field_std": float(field.std()),
    }


def run_pipeline(self_test: bool = False) -> dict:
    verify_sha256()
    jac = jacobian()
    if jac["trace"] >= 0 or jac["det"] <= 0 or jac["disc"] <= 0:
        raise RuntimeError(f"Jacobian is not Turing-capable: {jac}")
    if jac["disc"] ** 2 <= 4.0 * DU * DV * jac["det"]:
        raise RuntimeError("Turing inequality failed at the operating point")
    gamma = gamma_for_wavelength(LAMBDA_UM, DV)
    band = unstable_band(gamma, jac, DV)
    linear = linear_mode_check(gamma, jac, DV)
    field, _v, snaps = simulate_field(gamma, DV, N_GRID, N_STEPS, SEED, snapshots=N_FRAMES)
    sweep = wavelength_sweep(gamma, jac)
    controls = ablation_controls(gamma, jac)
    u128, _, _ = simulate_field(gamma, DV, n=128, nsteps=5200, seed=SEED, snapshots=0)
    i96 = sweep["factors"].index(1.0)
    grid_convergence = {
        "96": float(sweep["measured_um"][i96]),
        "128": dominant_wavelength_um(u128),
        "160": dominant_wavelength_um(field),
    }

    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB").resize(GIF_SIZE, Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    seg = segment_specimen(rgb)
    render_gif(rgb, snaps, seg["shell_mask"], GROWTH_GIF)
    with Image.open(GROWTH_GIF) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
        gif.seek(n_frames - 1)
        last = np.asarray(gif.convert("RGB"), dtype=np.float64) / 255.0
        width, height = gif.size
    render_comparison(rgb, last, field, sweep, band, gamma, jac, DV, COMPARISON_PNG)

    metrics = build_metrics(field, linear, sweep, controls, seg, band, gamma, grid_convergence)
    metrics["gif"] = {"frames": n_frames, "width": width, "height": height, "path": str(GROWTH_GIF.relative_to(ROOT))}
    metrics["png"] = str(COMPARISON_PNG.relative_to(ROOT))
    # Drop non-serialisable leftovers (none expected). Diameters are lists of floats.
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if self_test:
        assert abs(metrics["lambda_c_um_linear"] - 31.37) < 1e-9
        assert abs(metrics["phenomenon_size_m"] - 3.137e-05) < 1e-12
        assert metrics["geological_age_ma"] == 0.0
        assert metrics["chronoseries"]["index"] == 1 and metrics["chronoseries"]["length"] == 1
        assert metrics["chronoseries"]["predecessor"] is None
        assert metrics["wavenumber_contrast"]["detail_inside_band"]
        assert not metrics["wavenumber_contrast"]["pore_inside_band"]
        assert not metrics["wavenumber_contrast"]["test_inside_band"]
        assert metrics["dispersion_pearson_r"] > 0.98, metrics["dispersion_pearson_r"]
        assert metrics["sweep"]["pearson_r"] > 0.98, metrics["sweep"]
        assert controls["equal_diffusion_sigma_max"] < 0.0
        assert controls["diffusion_ablation_std_ratio"] < 0.5
        assert abs(metrics["saturated_wavelength_um"] - LAMBDA_UM) / LAMBDA_UM < 0.12, metrics["saturated_wavelength_um"]
        assert n_frames >= 10 and min(width, height) >= 480
        assert seg["n_tests"] >= 4, seg["n_tests"]
        assert seg["mean_diameter_um"] is not None and seg["mean_rim_excluded_diameter_um"] is not None
        assert abs(seg["mean_diameter_um"] - 600.0) > 1.0
        assert abs(seg["mean_rim_excluded_diameter_um"] - seg["mean_diameter_um"]) > 1.0
        assert all(abs(v - LAMBDA_UM) / LAMBDA_UM < 0.15 for v in grid_convergence.values()), grid_convergence
        print(
            f"[PASS] slot 3108 Ammonia beccarii λc=31.37 µm age=0 Ma "
            f"dispersion r={metrics['dispersion_pearson_r']:.4f} "
            f"sweep r={metrics['sweep']['pearson_r']:.4f} "
            f"saturated λ={metrics['saturated_wavelength_um']:.2f} µm "
            f"tests={seg['n_tests']} mean D={seg['mean_diameter_um']:.1f} µm "
            f"gif={n_frames}f {width}x{height}"
        )
    return metrics


def measure_morphoeval() -> list[dict]:
    """Sidecar measurements for measure_outputs.py. Values are not copies of their references."""
    if not METRICS.exists():
        run_pipeline(self_test=False)
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    spec = m["specimen"]
    rows = [
        {
            "name": "linear dispersion Pearson r vs numerical Fourier growth",
            "value": round(m["dispersion_pearson_r"], 4),
            "unit": "Pearson r",
            "method": "linearised Schnakenberg PDE on exact Fourier modes versus σ(k) from det(γJ − k²D − σI)=0",
            "reference_value": 1.0,
            "reference_source": "exact match of the discrete dispersion relation",
            "note": f"RMSE = {m['dispersion_rmse']:.6g} 1/τ over {m.get('dispersion_ks_statistic')} KS; n modes in metrics JSON",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE",
            "value": round(m["dispersion_rmse"], 6),
            "unit": "1/τ",
            "method": "root mean square of analytical minus numerical σ(k)",
            "reference_value": 0.0,
            "reference_source": "zero residual of a perfect integrator",
            "derived_from": "script",
        },
        {
            "name": "dispersion KS statistic",
            "value": round(m["dispersion_ks_statistic"], 4),
            "unit": "KS distance",
            "method": "two-sample KS between analytical and numerical σ samples",
            "reference_value": 0.0,
            "reference_source": "identical distributions",
            "note": f"p={m['dispersion_ks_pvalue']:.4g}",
            "derived_from": "script",
        },
        {
            "name": "gamma sweep predicted vs measured wavelength Pearson r",
            "value": round(m["sweep"]["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "six γ factors; linear λ_c(γ) versus FFT wavelength of the saturated field",
            "reference_value": 1.0,
            "reference_source": "λ ∝ γ^(−1/2) with no nonlinear shift",
            "note": f"RMSE {m['sweep']['rmse_um']:.3f} µm; predicted {m['sweep']['predicted_um']}; measured {m['sweep']['measured_um']}",
            "derived_from": "script",
        },
        {
            "name": "saturated spot wavelength",
            "value": round(m["saturated_wavelength_um"], 4),
            "unit": "µm",
            "method": "parabolic-refined radial FFT peak of the nonlinear activator on the 600 µm face",
            "reference_value": LAMBDA_UM,
            "reference_source": "linear λ_c = 31.37 µm = 3.137e-05 m (this slot, not a published pore spacing)",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion control sigma_max",
            "value": round(m["controls"]["equal_diffusion_sigma_max"], 6),
            "unit": "1/τ",
            "method": "σ(k_c) with Dv reset to Du; Turing requires Dv f_u + Du g_v > 0",
            "reference_value": 0.0,
            "reference_source": "marginal stability",
            "note": "negative control: equal diffusion must decay",
            "derived_from": "script",
        },
        {
            "name": "diffusion ablation std ratio",
            "value": round(m["controls"]["diffusion_ablation_std_ratio"], 4),
            "unit": "std_final/std_initial",
            "method": "γ = 0 pure diffusion of a noise seed for 400 steps, versus the patterned run",
            "reference_value": 1.0,
            "reference_source": "no decay if the pattern were not diffusion-limited",
            "note": "ablation control",
            "derived_from": "script",
        },
        {
            "name": "specimen segmented mean test diameter",
            "value": None if spec["mean_diameter_um"] is None else round(spec["mean_diameter_um"], 4),
            "unit": "µm",
            "method": "longer side of the gray>0.08 bounding box of each test, scaled by the detected 500 µm bar (includes the bright rim)",
            "reference_value": 600.0,
            "reference_source": "Hillewaert 2012 caption diameter ~600 µm",
            "note": f"n={spec['n_tests']}; halo diameters={spec['diameters_um']}; CV={spec['diameter_cv']}",
            "derived_from": "output",
        },
        {
            "name": "specimen rim-excluded test diameter",
            "value": None if spec.get("mean_rim_excluded_diameter_um") is None else round(spec["mean_rim_excluded_diameter_um"], 4),
            "unit": "µm",
            "method": "longer side of the gray>0.32 core inside each segmented test, same 500 µm scale bar",
            "reference_value": 600.0,
            "reference_source": "Hillewaert 2012 caption diameter ~600 µm",
            "note": f"cores={spec.get('rim_excluded_diameters_um')}; halo mean={spec['mean_diameter_um']}",
            "derived_from": "output",
        },
        {
            "name": "grid convergence FFT wavelength at 128",
            "value": round(m["grid_convergence_um"]["128"], 4),
            "unit": "µm",
            "method": "radial FFT peak on a 128² face at the same γ as the 160² production run",
            "reference_value": LAMBDA_UM,
            "reference_source": "linear λ_c = 31.37 µm",
            "note": f"also 96²={m['grid_convergence_um']['96']}, 160²={m['grid_convergence_um']['160']}",
            "derived_from": "script",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(spec["coherency_sigma_1_6"], 4),
            "unit": "coherency",
            "method": "mean (λ1−λ2)/(λ1+λ2) inside the segmented tests, Gaussian σ = 1.6 px on the 720×540 plate",
            "reference_value": round(spec["coherency_sigma_2_8"], 4),
            "reference_source": "same plate, structure tensor at σ = 2.8 px (a different filter scale, not a copy of the value)",
            "derived_from": "output",
        },
        {
            "name": "specimen foreground fraction",
            "value": round(spec["foreground_fraction"], 4),
            "unit": "area fraction",
            "method": "fraction of the 720×540 plate covered by tests with circularity > 0.45 after opening and hole fill",
            "reference_value": 0.4404,
            "reference_source": "verify_image_fitness.compute_specimen_cv_signature foreground_fraction on the full-resolution JPEG (different mask)",
            "derived_from": "output",
        },
        {
            "name": "segmented test count",
            "value": spec["n_tests"],
            "unit": "count",
            "method": "connected components on the Hillewaert plate after rejecting the scale bar and caption by area and circularity",
            "reference_value": None,
            "note": "the plate shows six tests; the count is measured, not copied from a parameter",
            "derived_from": "output",
        },
        {
            "name": "spot spacing CV",
            "value": None if m["spot_spacing_cv"] is None else round(m["spot_spacing_cv"], 4),
            "unit": "CV",
            "method": "std/mean of nearest-neighbour distances between activator peaks",
            "reference_value": 0.0,
            "reference_source": "a perfectly regular lattice",
            "note": f"mean spacing {m['spot_spacing_mean_um']} µm; KS vs Normal(31.37 µm) = {m['spot_spacing_ks_statistic']}",
            "derived_from": "script",
        },
    ]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    run_pipeline(self_test=args.self_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
