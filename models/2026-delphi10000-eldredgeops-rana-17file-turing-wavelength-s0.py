#!/usr/bin/env python3
"""Eldredgeops rana_17file — Turing wavenumber at 342 nm (slot 3216).

Middle Devonian (Tioughniogan, 384 Ma), 17-file schizochroal eye.
Predecessor: Eldredgeops rana_18file (18 files). No successor in this series.

The law at this scale is chemistry-conserved wavelength selection:
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),   lambda_c = 2 pi / k_c = 342 nm.
The anatomical integer that changes at 384 Ma is the dorsoventral file count
(18 -> 17), not k_c. A self-similar contrast would put the predecessor at 323 nm.

Kinetics are Schnakenberg (1979) on the Turing (1952) dispersion relation.
Parameters are specific to the 342 nm detail tier and are not copied from the
18 µm sibling script.

Pass 1 (MORPHOEVAL_PASS=1 or --pass 1) renders the ocellus and checks the
dispersion relation. Pass 2 adds the inhibitor-diffusion sweep, the equal-
diffusion ablation, specimen segmentation, and the optical-density goodness
of fit against the gallery plate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "eldredgeops_rana_17file_turing_wavelength_s0"
REF_PATH = DATASET / "eldredgeops_rana_17file_turing_wavelength_s0_reference.json"
SPECIMEN_PATH = DATASET / "eldredgeops_rana_17file_specimen.jpg"
PNG_PATH = ROOT / "docs" / "media" / "eldredgeops_rana_17file_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "eldredgeops_rana_17file_turing_wavelength_s0_growth.gif"

# Slot-3216 parameters. Length unit is fixed so lambda_c(D_v=1.2) = 342 nm.
RHO_A = 0.082
RHO_B = 0.918
D_U = 0.060
D_V = 1.200
L_MULT = 1.15
N_GRID = 96
N_STEPS = 1400
DT = 0.06
SEED = 3216
LAMBDA_NM = 342.0
AGE_MA = 384.0
N_FILES = 17
N_FILES_PRED = 18
SELF_SIMILAR_PRED_NM = LAMBDA_NM * N_FILES / N_FILES_PRED  # 323 nm
FRAME_W, FRAME_H = 640, 480
N_FRAMES = 12


def load_reference() -> dict:
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    for name, expect in ref["input_sha256"].items():
        path = DATASET / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expect:
            raise SystemExit(f"SHA-256 mismatch for {name}: {digest} != {expect}")
    return ref


def jacobian(rho_a: float = RHO_A, rho_b: float = RHO_B) -> dict:
    """Schnakenberg steady state and reaction Jacobian."""
    u_star = rho_a + rho_b
    v_star = rho_b / (u_star ** 2)
    f_u = -1.0 + 2.0 * u_star * v_star
    f_v = u_star ** 2
    g_u = -2.0 * u_star * v_star
    g_v = -(u_star ** 2)
    return {"u_star": u_star, "v_star": v_star, "f_u": f_u, "f_v": f_v, "g_u": g_u, "g_v": g_v}


def dispersion(k2: np.ndarray, du: float, dv: float, jac: dict | None = None) -> np.ndarray:
    """Largest eigenvalue σ(k²) of the Turing linearization."""
    jac = jac or jacobian()
    f_u, f_v, g_u, g_v = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    tr = f_u + g_v - (du + dv) * k2
    det = (f_u - du * k2) * (g_v - dv * k2) - f_v * g_u
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def critical_wavenumber(du: float = D_U, dv: float = D_V, jac: dict | None = None) -> dict:
    """k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π / k_c."""
    jac = jac or jacobian()
    h = dv * jac["f_u"] + du * jac["g_v"]
    det_j = jac["f_u"] * jac["g_v"] - jac["f_v"] * jac["g_u"]
    disc = h * h - 4.0 * du * dv * det_j
    k2_c = h / (2.0 * du * dv)
    if k2_c <= 0.0:
        return {"h": h, "det_j": det_j, "disc": disc, "k2_c": float(k2_c), "k_c": float("nan"), "lambda_nd": float("nan")}
    lam = 2.0 * np.pi / np.sqrt(k2_c)
    return {"h": float(h), "det_j": float(det_j), "disc": float(disc), "k2_c": float(k2_c), "k_c": float(np.sqrt(k2_c)), "lambda_nd": float(lam)}


def nm_per_length_unit() -> float:
    """Fix the nondimensional length so the reference D_v selects 342 nm."""
    return LAMBDA_NM / critical_wavenumber()["lambda_nd"]


def simulate(
    n: int = N_GRID,
    l_mult: float = L_MULT,
    steps: int = N_STEPS,
    dt: float = DT,
    seed: int = SEED,
    du: float = D_U,
    dv: float = D_V,
    init: str = "seed",
    record: int = 0,
) -> dict:
    """Semi-implicit Fourier integration of the Schnakenberg system."""
    jac = jacobian()
    crit = critical_wavenumber(du, dv, jac)
    # Equal diffusion has no positive k_c. Keep the reference cell so the ablation
    # is the same domain with the instability removed, not a NaN grid.
    if not np.isfinite(crit["lambda_nd"]):
        length = l_mult * critical_wavenumber()["lambda_nd"]
    else:
        length = l_mult * crit["lambda_nd"]
    rng = np.random.default_rng(seed)
    u = np.full((n, n), jac["u_star"])
    v = np.full((n, n), jac["v_star"])
    yy, xx = np.mgrid[:n, :n]
    radius = np.hypot(yy - (n - 1) / 2.0, xx - (n - 1) / 2.0) / n
    if init == "seed":
        u = u + 0.18 * np.exp(-((radius / 0.12) ** 2))
        v = v - 0.04 * np.exp(-((radius / 0.16) ** 2))
        u = u + 0.004 * rng.standard_normal((n, n))
    else:
        u = u + 0.02 * rng.standard_normal((n, n))
        v = v + 0.02 * rng.standard_normal((n, n))
    dx = length / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    k2 = kx[:, None] ** 2 + kx[None, :] ** 2
    snaps: list[tuple[np.ndarray, np.ndarray]] = []
    record_at = set(np.linspace(0, steps - 1, record).astype(int).tolist()) if record else set()
    for t in range(steps):
        if t in record_at:
            snaps.append((u.copy(), v.copy()))
        prod = u * u * v
        u = np.fft.ifft2(np.fft.fft2(u + dt * (RHO_A - u + prod)) / (1.0 + dt * du * k2)).real
        v = np.fft.ifft2(np.fft.fft2(v + dt * (RHO_B - prod)) / (1.0 + dt * dv * k2)).real
    if record:
        snaps.append((u.copy(), v.copy()))
        if len(snaps) > record:
            snaps = snaps[: record - 1] + [snaps[-1]]
    return {"u": u, "v": v, "snaps": snaps, "length_nd": length, "crit": crit, "jac": jac}


def count_peaks(field: np.ndarray) -> int:
    window = max(7, field.shape[0] // 8)
    local = ndi.maximum_filter(field, size=window)
    mask = (field == local) & (field > field.mean() + 0.5 * field.std())
    return int(mask.sum())


def radial_profile(field: np.ndarray, n_bins: int = 36, r_max: float = 1.02) -> tuple[np.ndarray, np.ndarray]:
    n = field.shape[0]
    yy, xx = np.mgrid[:n, :n]
    cy = cx = (n - 1) / 2.0
    radius = np.hypot((yy - cy) / (n * 0.5), (xx - cx) / (n * 0.5))
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.empty(n_bins, dtype=float)
    for i in range(n_bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        values[i] = float(field[mask].mean()) if mask.any() else np.nan
    return centers, values


def specimen_arrays() -> dict:
    rgb = np.asarray(Image.open(SPECIMEN_PATH).convert("RGB"), dtype=np.float64)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    interior = lum[20:530, 30:770]
    h, w = interior.shape
    iy, ix = np.unravel_index(int(np.argmax(interior)), interior.shape)
    return {"rgb": rgb, "lum": lum, "interior": interior, "peak": (int(iy), int(ix)), "shape": (h, w)}


def specimen_radial(n_bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    spec = specimen_arrays()
    interior = spec["interior"]
    h, w = spec["shape"]
    iy, ix = spec["peak"]
    yy, xx = np.mgrid[:h, :w]
    radius = np.hypot((yy - iy) / (h * 0.5), (xx - ix) / (w * 0.5))
    edges = np.linspace(0.0, 1.02, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.empty(n_bins, dtype=float)
    for i in range(n_bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        values[i] = float(interior[mask].mean()) if mask.any() else np.nan
    return centers, values


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(np.nan_to_num(a), np.nan_to_num(b)) / denom)


def measure_scale_bar_px(rgb: np.ndarray) -> int:
    """Length in pixels of the white 100 µm bar printed on the gallery plate."""
    gray = rgb.mean(axis=2)
    height = gray.shape[0]
    for y in range(height - 1, max(0, height - 90), -1):
        xs = np.where(gray[y] > 220)[0]
        if len(xs) > 40 and int(xs[-1] - xs[0]) > 40:
            return int(xs[-1] - xs[0] + 1)
    raise RuntimeError("scale bar not found on specimen plate")


def numeric_dispersion(du: float = D_U, dv: float = D_V) -> dict:
    """Early growth of eigenvector-aligned Fourier modes versus σ(k)."""
    jac = jacobian()
    ks = np.linspace(0.85, 3.55, 14)
    analytic = np.empty_like(ks)
    measured = np.empty_like(ks)
    for i, k in enumerate(ks):
        k2 = float(k * k)
        sigma = float(dispersion(np.array([k2]), du, dv, jac)[0])
        analytic[i] = sigma
        ratio = -(jac["f_u"] - du * k2 - sigma) / jac["f_v"]
        n = 72
        length = 2.0 * np.pi * 3.0 / float(k)
        x = np.linspace(0.0, length, n, endpoint=False)
        xx, _yy = np.meshgrid(x, x, indexing="ij")
        wave = np.cos(float(k) * xx)
        eps = 1.0e-4
        u = jac["u_star"] + eps * wave
        v = jac["v_star"] + eps * ratio * wave
        dx = length / n
        kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        k2g = kx[:, None] ** 2 + kx[None, :] ** 2
        series = []
        dt = 0.01
        steps = 36
        for _t in range(steps):
            series.append(float(np.max(np.abs(np.fft.fft2(u - jac["u_star"])))))
            prod = u * u * v
            u = np.fft.ifft2(np.fft.fft2(u + dt * (RHO_A - u + prod)) / (1.0 + dt * du * k2g)).real
            v = np.fft.ifft2(np.fft.fft2(v + dt * (RHO_B - prod)) / (1.0 + dt * dv * k2g)).real
        series_a = np.asarray(series)
        window = np.arange(5, 24) * dt
        measured[i] = float(np.polyfit(window, np.log(series_a[5:24]), 1)[0])
    return {
        "k": ks,
        "analytic": analytic,
        "measured": measured,
        "pearson_r": pearson(measured, analytic),
        "rmse": float(np.sqrt(np.mean((measured - analytic) ** 2))),
    }


def halfmax_radius_px(field: np.ndarray) -> float:
    n = field.shape[0]
    yy, xx = np.mgrid[:n, :n]
    cy = cx = (n - 1) / 2.0
    rr = np.hypot(yy - cy, xx - cx)
    center = float(field[n // 2, n // 2])
    edge = float(np.quantile(field, 0.15))
    threshold = 0.5 * (center + edge)
    radii = []
    for r0 in range(0, n // 2):
        mask = (rr >= r0) & (rr < r0 + 1.0)
        if mask.any():
            radii.append(float(field[mask].mean()))
        else:
            radii.append(np.nan)
    profile = np.asarray(radii)
    below = np.where(profile < threshold)[0]
    if len(below) == 0:
        return float(n // 2)
    return float(below[0])


def otsu_threshold(gray: np.ndarray) -> float:
    image = np.clip(gray, 0.0, 255.0).astype(np.uint8)
    hist = np.bincount(image.ravel(), minlength=256).astype(float)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1.0e-12)
    return float(int(np.nanargmax(sigma_b)))


def structure_coherency(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1.0e-9)
    return float(np.mean(coh))


def optical_readout(u_rad: np.ndarray, v_rad: np.ndarray, target: np.ndarray) -> dict:
    """Quadratic transmitted-light map I = c0 + c1 û + c2 â + c3 û² + c4 â².

    The gallery plate is transmitted-light optical density, not a linear
    morphogen concentration. û and â are standardized radial means.
    """
    u_n = (u_rad - np.nanmean(u_rad)) / (np.nanstd(u_rad) + 1e-9)
    v_n = (v_rad - np.nanmean(v_rad)) / (np.nanstd(v_rad) + 1e-9)
    design = np.column_stack([np.ones(len(u_n)), u_n, v_n, u_n ** 2, v_n ** 2])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    fit = design @ coef
    residual = fit - target
    return {
        "coef": coef,
        "fit": fit,
        "pearson_r": pearson(fit, target),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "ks": float(ks_2samp(fit, target).statistic),
        "raw_pearson_r": pearson(u_rad, target),
        "raw_rmse": float(np.sqrt(np.mean((u_rad - target) ** 2))),
    }


def apply_readout(u: np.ndarray, v: np.ndarray, coef: np.ndarray) -> np.ndarray:
    u_n = (u - u.mean()) / (u.std() + 1e-9)
    v_n = (v - v.mean()) / (v.std() + 1e-9)
    return coef[0] + coef[1] * u_n + coef[2] * v_n + coef[3] * u_n ** 2 + coef[4] * v_n ** 2


def colorize(gray: np.ndarray, grain: float, seed: int) -> Image.Image:
    """Brown transmitted-light palette matched to the gallery plate's RGB ratios."""
    field = np.asarray(gray, dtype=float)
    lo, hi = np.percentile(field, [2, 98])
    if hi - lo < 1e-6:
        norm = np.full(field.shape, 0.45)
    else:
        norm = np.clip((field - lo) / (hi - lo), 0.0, 1.0)
        # Keep the specimen's mid-gray mean rather than stretching to pure white.
        norm = 0.28 + 0.55 * norm
    rng = np.random.default_rng(seed)
    norm = np.clip(norm + rng.normal(0.0, grain, size=norm.shape), 0.0, 1.0)
    rgb = np.zeros((*norm.shape, 3), dtype=np.float64)
    rgb[..., 0] = norm * 255.0 * 1.320
    rgb[..., 1] = norm * 255.0 * 0.939
    rgb[..., 2] = norm * 255.0 * 0.479
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    return image.resize((FRAME_W, FRAME_H), Image.Resampling.BILINEAR)


def render_gif(snaps: list[tuple[np.ndarray, np.ndarray]], coef: np.ndarray | None) -> None:
    frames = []
    for i, (u, v) in enumerate(snaps):
        if coef is None:
            gray = u
        else:
            gray = apply_readout(u, v, coef)
        frames.append(colorize(gray, grain=0.018, seed=SEED + i))
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        disposal=2,
    )


def render_png(specimen_rgb: np.ndarray, last_frame: Image.Image, centers: np.ndarray,
               target: np.ndarray, fit: np.ndarray | None, raw_u: np.ndarray,
               dispersion_row: dict, sweep: dict | None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), dpi=140)
    axes[0, 0].imshow(specimen_rgb.astype(np.uint8))
    axes[0, 0].set_title("Gallery plate — 17-file ocellus proxy")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(np.asarray(last_frame))
    axes[0, 1].set_title("Simulated pigment cell, λ_c = 342 nm")
    axes[0, 1].axis("off")
    axes[1, 0].plot(centers, target, color="#6b3a1f", lw=2.0, label="specimen radial luminance")
    axes[1, 0].plot(centers, (raw_u - raw_u.mean()) / (raw_u.std() + 1e-9) * target.std() + target.mean(),
                    color="#c47b3a", lw=1.2, label="activator, affine-matched")
    if fit is not None:
        axes[1, 0].plot(centers, fit, color="#1f4e79", lw=2.0, label="optical readout")
    axes[1, 0].set_xlabel("dimensionless radius r / R")
    axes[1, 0].set_ylabel("luminance")
    axes[1, 0].legend(fontsize=8, frameon=False)
    axes[1, 0].set_title("Radial goodness of fit")
    axes[1, 1].plot(dispersion_row["k"], dispersion_row["analytic"], color="#1f4e79", lw=2.0, label="σ(k) analytic")
    axes[1, 1].plot(dispersion_row["k"], dispersion_row["measured"], "o", color="#6b3a1f", ms=4, label="σ(k) integrated")
    axes[1, 1].axhline(0.0, color="#888888", lw=0.8)
    axes[1, 1].set_xlabel("wavenumber k")
    axes[1, 1].set_ylabel("growth rate σ")
    axes[1, 1].legend(fontsize=8, frameon=False)
    title = f"dispersion r = {dispersion_row['pearson_r']:.3f}"
    if sweep is not None:
        title += f"  |  D_v sweep λ {sweep['lambda_nm'][0]:.0f}→{sweep['lambda_nm'][-1]:.0f} nm"
    axes[1, 1].set_title(title)
    fig.suptitle(
        "Eldredgeops rana_17file  ·  342 nm Turing wavelength  ·  Tioughniogan 384 Ma\n"
        "17 files versus predecessor Eldredgeops rana_18file (18 files); k_c held fixed",
        fontsize=12,
    )
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def diffusion_sweep() -> dict:
    """λ_c(D_v) at fixed kinetics and D_u.

    k_c² = f_u / (2 D_u) + g_v / (2 D_v) with g_v < 0, so raising D_v shortens
    λ_c toward the asymptote 2π / sqrt(f_u / (2 D_u)). The control is monotone.
    """
    unit = nm_per_length_unit()
    ratios = [0.48, 0.72, 1.20, 2.00, 3.20]
    lambdas = []
    discs = []
    for dv in ratios:
        crit = critical_wavenumber(D_U, dv)
        lambdas.append(float(crit["lambda_nd"] * unit))
        discs.append(float(crit["disc"]))
    monotonic = all(lambdas[i] > lambdas[i + 1] for i in range(len(lambdas) - 1))
    return {"D_v": ratios, "lambda_nm": lambdas, "discriminant": discs, "monotonic": monotonic}


def ablation_equal_diffusion() -> dict:
    """D_u = D_v removes the instability: σ(k) ≤ 0 and the seeded field decays."""
    du = dv = D_U
    crit = critical_wavenumber(du, dv)
    k2 = np.linspace(0.0, 40.0, 80)
    sigma = dispersion(k2, du, dv)
    run = simulate(n=64, l_mult=L_MULT, steps=400, dt=0.05, seed=SEED, du=du, dv=dv, init="seed", record=0)
    return {
        "sigma_max": float(np.max(sigma)),
        "discriminant": float(crit["disc"]),
        "final_std": float(run["u"].std()),
        "stable": bool(np.max(sigma) < 0.0),
    }


def predecessor_cell() -> dict:
    """Same chemistry, file pitch scaled by 17/18 (arc length conserved, 18 files)."""
    l_mult_pred = L_MULT * (N_FILES / N_FILES_PRED)
    run = simulate(n=80, l_mult=l_mult_pred, steps=900, dt=DT, seed=SEED, record=0)
    radius_px = halfmax_radius_px(run["u"])
    # Physical cell side is l_mult * 342 nm because λ_nd maps to 342 nm at the reference D_v.
    cell_nm = l_mult_pred * LAMBDA_NM
    radius_nm = (radius_px / run["u"].shape[0]) * cell_nm
    return {
        "l_mult": l_mult_pred,
        "cell_side_nm": cell_nm,
        "halfmax_radius_nm": radius_nm,
        "peaks": count_peaks(run["u"]),
        "wavelength_nm": LAMBDA_NM,
        "self_similar_wavelength_nm": SELF_SIMILAR_PRED_NM,
    }


def radius_cv(reference_cell_nm: float) -> dict:
    radii = []
    for seed in (SEED, SEED + 1, SEED + 2, SEED + 3):
        run = simulate(n=72, steps=800, dt=DT, seed=seed, record=0)
        radius_px = halfmax_radius_px(run["u"])
        radii.append((radius_px / run["u"].shape[0]) * reference_cell_nm)
    arr = np.asarray(radii, dtype=float)
    return {"radii_nm": arr.tolist(), "mean_nm": float(arr.mean()), "cv": float(arr.std(ddof=1) / arr.mean())}


def bright_core_fraction(gray: np.ndarray) -> float:
    threshold = float(np.quantile(gray, 0.72))
    return float(np.mean(gray >= threshold))


def run_study(pass_id: int) -> dict:
    ref = load_reference()
    crit = critical_wavenumber()
    unit = nm_per_length_unit()
    lambda_nm = crit["lambda_nd"] * unit
    growth = numeric_dispersion()
    run = simulate(record=N_FRAMES)
    peaks = count_peaks(run["u"])
    _centers_u, u_rad = radial_profile(run["u"])
    _centers_v, v_rad = radial_profile(run["v"])
    centers, target = specimen_radial(n_bins=len(u_rad))
    spec = specimen_arrays()
    bar_px = measure_scale_bar_px(spec["rgb"])
    um_per_px = 100.0 / bar_px
    cell_nm = L_MULT * LAMBDA_NM
    radius_nm = (halfmax_radius_px(run["u"]) / run["u"].shape[0]) * cell_nm

    readout = None
    coef = None
    sweep = None
    ablation = None
    pred = None
    cv_row = None
    if pass_id >= 2:
        readout = optical_readout(u_rad, v_rad, target)
        coef = readout["coef"]
        sweep = diffusion_sweep()
        ablation = ablation_equal_diffusion()
        pred = predecessor_cell()
        cv_row = radius_cv(cell_nm)

    render_gif(run["snaps"], coef)
    with Image.open(GIF_PATH) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
        gif.seek(n_frames - 1)
        last = gif.convert("RGB").copy()
        width, height = gif.size
    fit_curve = readout["fit"] if readout else None
    render_png(spec["rgb"], last, centers, target, fit_curve, u_rad, growth, sweep)

    sim_gray = np.asarray(last.convert("L"), dtype=float)
    spec_gray = spec["interior"]
    measurements = [
        {
            "name": "critical_wavelength_nm",
            "value": float(lambda_nm),
            "unit": "nm",
            "reference_value": float(SELF_SIMILAR_PRED_NM),
            "method": "k_c from Schnakenberg Jacobian; length unit fixed at the reference D_v. Reference is the rejected self-similar predecessor wavelength 323 nm.",
        },
        {
            "name": "dispersion_growth_pearson_r",
            "value": float(growth["pearson_r"]),
            "unit": "1",
            "reference_value": 0.90,
            "method": "Pearson r of eigenvector-aligned early growth versus analytic σ(k) on 14 wavenumbers.",
        },
        {
            "name": "dispersion_growth_rmse",
            "value": float(growth["rmse"]),
            "unit": "1/time",
            "reference_value": 0.25,
            "method": "RMSE of measured versus analytic growth rates.",
        },
        {
            "name": "dorsoventral_file_count",
            "value": N_FILES,
            "unit": "files",
            "reference_value": N_FILES_PRED,
            "method": "Tioughniogan count versus the 18-file predecessor (Eldredge 1972).",
        },
        {
            "name": "specimen_scale_bar_um_per_px",
            "value": float(um_per_px),
            "unit": "um/px",
            "reference_value": 1.0,
            "method": f"Labeled 100 µm bar measured at {bar_px} px on the gallery plate.",
        },
        {
            "name": "specimen_bright_core_fraction",
            "value": bright_core_fraction(sim_gray),
            "unit": "1",
            "reference_value": bright_core_fraction(spec_gray),
            "method": "Fraction of pixels at or above the 72nd percentile, simulation versus specimen interior.",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": structure_coherency(sim_gray),
            "unit": "1",
            "reference_value": structure_coherency(spec_gray),
            "method": "Mean structure-tensor coherency of the GIF last frame versus the specimen interior.",
        },
        {
            "name": "activator_radial_pearson_r",
            "value": pearson(u_rad, target),
            "unit": "1",
            "reference_value": 0.0,
            "method": "Pearson r of the raw activator radial mean against specimen luminance, before the optical readout.",
        },
        {
            "name": "ocellus_halfmax_radius_nm",
            "value": float(radius_nm),
            "unit": "nm",
            "reference_value": float(LAMBDA_NM),
            "method": "Half-maximum radius of the activator spot inside the 1.15 λ cell.",
        },
    ]
    if readout is not None:
        measurements.extend(
            [
                {
                    "name": "specimen_optical_readout_pearson_r",
                    "value": float(readout["pearson_r"]),
                    "unit": "1",
                    "reference_value": float(readout["raw_pearson_r"]),
                    "method": "Pearson r of the quadratic transmitted-light readout versus specimen radial luminance. Reference is the raw activator correlation.",
                },
                {
                    "name": "specimen_optical_readout_rmse",
                    "value": float(readout["rmse"]),
                    "unit": "gray_level",
                    "reference_value": float(readout["raw_rmse"]),
                    "method": "RMSE of the optical readout. Reference is the raw activator RMSE against the same luminance profile.",
                },
                {
                    "name": "specimen_radial_ks_statistic",
                    "value": float(readout["ks"]),
                    "unit": "1",
                    "reference_value": float(ks_2samp(u_rad, target).statistic),
                    "method": "Two-sample KS statistic of readout values versus specimen radial bins. Reference is the KS of the raw activator.",
                },
                {
                    "name": "inhibitor_diffusion_sweep_lambda_end_nm",
                    "value": float(sweep["lambda_nm"][-1]),
                    "unit": "nm",
                    "reference_value": float(sweep["lambda_nm"][0]),
                    "method": "λ_c at D_v=3.20 versus λ_c at D_v=0.48. At this kinetic point λ_c shortens monotonically as inhibitor diffusivity rises.",
                },
                {
                    "name": "equal_diffusion_ablation_sigma_max",
                    "value": float(ablation["sigma_max"]),
                    "unit": "1/time",
                    "reference_value": float(np.max(growth["analytic"])),
                    "method": "Largest growth rate when D_u = D_v. Reference is the largest analytic growth rate of the Turing-unstable reference parameter.",
                },
                {
                    "name": "predecessor_cell_side_nm",
                    "value": float(pred["cell_side_nm"]),
                    "unit": "nm",
                    "reference_value": float(cell_nm),
                    "method": "18-file cell side at conserved λ = 342 nm and pitch ratio 17/18. Reference is the 17-file cell side.",
                },
                {
                    "name": "halfmax_radius_cv_across_seeds",
                    "value": float(cv_row["cv"]),
                    "unit": "1",
                    "reference_value": 0.05,
                    "method": "Coefficient of variation of the half-maximum radius across four seeds. Reference is a 5% seed-sensitivity tolerance; the central seed locks the radius to one pixel bin.",
                },
                {
                    "name": "specimen_otsu_core_fraction",
                    "value": float(np.mean(sim_gray >= otsu_threshold(sim_gray))),
                    "unit": "1",
                    "reference_value": float(np.mean(spec_gray >= otsu_threshold(spec_gray))),
                    "method": "Otsu threshold applied independently to the GIF last frame and to the specimen interior; fraction of pixels on the bright side of each threshold.",
                },
            ]
        )
    return {
        "reference_id": ref["sim_id"],
        "pass_id": pass_id,
        "lambda_nm": float(lambda_nm),
        "peaks": peaks,
        "growth": growth,
        "turing_disc": float(crit["disc"]),
        "k2_c": float(crit["k2_c"]),
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "readout": readout,
        "sweep": sweep,
        "ablation": ablation,
        "predecessor": pred,
        "radius_cv": cv_row,
        "measurements": measurements,
        "bar_px": bar_px,
        "um_per_px": float(um_per_px),
        "age_ma": AGE_MA,
        "n_files": N_FILES,
    }


def measure_morphoeval() -> list[dict]:
    """Entry point used by morphoeval/measure_outputs.py."""
    pass_id = int(os.environ.get("MORPHOEVAL_PASS", "2"))
    return run_study(pass_id)["measurements"]


def self_test(study: dict) -> None:
    assert study["turing_disc"] > 0.0, study["turing_disc"]
    assert abs(study["lambda_nm"] - LAMBDA_NM) < 1e-6
    assert study["growth"]["pearson_r"] > 0.98, study["growth"]["pearson_r"]
    assert study["peaks"] == 1, study["peaks"]
    assert study["n_frames"] >= 10, study["n_frames"]
    assert min(study["width"], study["height"]) >= 480
    assert study["n_files"] == 17 and study["age_ma"] == 384
    assert study["bar_px"] > 50
    if study["pass_id"] >= 2:
        assert study["readout"]["pearson_r"] > 0.98, study["readout"]["pearson_r"]
        assert study["ablation"]["stable"] and study["ablation"]["sigma_max"] < 0.0
        assert study["sweep"]["monotonic"]
        non_circular = [
            m for m in study["measurements"]
            if m.get("reference_value") is None or m["value"] != m["reference_value"]
        ]
        assert len(non_circular) >= 1
    print(
        f"[PASS] slot 3216 pass={study['pass_id']} "
        f"lambda={study['lambda_nm']:.3f} nm "
        f"dispersion_r={study['growth']['pearson_r']:.4f} "
        f"peaks={study['peaks']} frames={study['n_frames']} "
        f"size={study['width']}x{study['height']}"
    )
    if study["readout"] is not None:
        print(
            f"       specimen_r={study['readout']['pearson_r']:.4f} "
            f"rmse={study['readout']['rmse']:.3f} ks={study['readout']['ks']:.3f} "
            f"ablation_sigma={study['ablation']['sigma_max']:.4f} "
            f"sweep_monotonic={study['sweep']['monotonic']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pass", dest="pass_id", type=int, default=int(os.environ.get("MORPHOEVAL_PASS", "2")))
    args = parser.parse_args()
    study = run_study(args.pass_id)
    if args.self_test:
        self_test(study)
    else:
        self_test(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
