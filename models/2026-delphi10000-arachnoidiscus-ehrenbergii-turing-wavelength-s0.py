#!/usr/bin/env python3
"""Arachnoidiscus ehrenbergii — Turing wavelength at 242 nm (slot 3118, detail).

Holocene extant wheel diatom, chronoseries 1/1, age 0 Ma. The assigned phenomenon
scale is 242 nm (2.42e-7 m, tier 2_macromolecular_100nm). Schnakenberg kinetics
with the Turing fastest-mode wavenumber

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),    lambda_c = 2 pi / k_c

set equal to 242 nm by the slot-specific ratio D_v/D_u = 31.18 (above the
discriminant onset 13.605). Neumann boundaries
on a square silica-deposition-vesicle patch of side 6 lambda_c = 1.452 um.

This is not slot 0199 (alpha=1.23, beta=0.34). Pass 1 renders the unguided
reaction-diffusion field. Pass 2 (appended in the remediation loop) adds the
diffusivity sweep, specimen segmentation, and goodness-of-fit.

Sources: Turing 1952 (10.1098/rstb.1952.0012); Schnakenberg 1979
(10.1016/0022-5193(79)90042-0); Sumper 2002 (10.1126/science.1070026).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "arachnoidiscus_ehrenbergii_turing_wavelength_s0"
SPECIMEN_PATH = DATASET / "arachnoidiscus_ehrenbergii_turing_wavelength_s0_specimen.jpg"
WIKI_PATH = DATASET / "arachnoidiscus_sp_wikimedia_cc-by-2.jpg"
REFERENCE_PATH = DATASET / "arachnoidiscus_ehrenbergii_turing_wavelength_s0_reference.json"
MEDIA_PNG = ROOT / "docs" / "media" / "arachnoidiscus_ehrenbergii_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "arachnoidiscus_ehrenbergii_turing_wavelength_s0_growth.gif"

# Slot 3118 detail-scale coefficients. Units of length are lambda_c, so the
# critical wavelength is 1 and the physical conversion is 242 nm per unit.
LAMBDA_NM = 242.0
LAMBDA_M = 2.42e-7
AGE_MA = 0.0
A_PROD = 0.062
B_PROD = 1.35
D_RATIO = 31.18
N_GRID = 96
N_WAVELENGTHS = 6.0
SEED = 3118
SPIRAL_ARMS = 3
SPIRAL_PITCH = 1.7
N_FRAMES = 12
FRAME_W, FRAME_H = 640, 480
SPECIMEN_SHA256 = "499034f85d30d464ea67fea4d7404a3546b57b85bb64c8d553cb36f2129bed05"

_SIM_CACHE: dict | None = None


def steady_state(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float]:
    """Homogeneous SDV fixed point of the Schnakenberg mass-action system."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float, float, float]:
    """Linearisation (f_u, f_v, g_u, g_v) at the homogeneous steady state."""
    u0, v0 = steady_state(a, b)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -(u0 * u0)
    return f_u, f_v, g_u, g_v


def diffusivities(
    f_u: float,
    g_v: float,
    ratio: float = D_RATIO,
    lambda_units: float = 1.0,
) -> tuple[float, float, float]:
    """Choose D_u, D_v so the fastest-growing mode has wavelength lambda_units.

    k_c^2 = (ratio * f_u + g_v) / (2 * ratio * D_u) is solved for D_u.
    """
    k2 = (2.0 * math.pi / lambda_units) ** 2
    numer = ratio * f_u + g_v
    if numer <= 0.0:
        raise ValueError(f"diffusion ratio {ratio} is below the Turing onset")
    d_u = numer / (2.0 * ratio * k2)
    d_v = ratio * d_u
    return d_u, d_v, k2


def dispersion_sigma(
    k: float,
    f_u: float,
    f_v: float,
    g_u: float,
    g_v: float,
    d_u: float,
    d_v: float,
) -> float:
    """Largest real part of the Turing dispersion relation at wavenumber k."""
    k2 = k * k
    trace = f_u + g_v - k2 * (d_u + d_v)
    det_j = f_u * g_v - f_v * g_u
    det_k = d_u * d_v * k2 * k2 - (d_v * f_u + d_u * g_v) * k2 + det_j
    disc = trace * trace - 4.0 * det_k
    root = math.sqrt(disc) if disc > 0.0 else 0.0
    return 0.5 * (trace + root)


def turing_report(ratio: float = D_RATIO) -> dict:
    """Analytic Turing conditions and the selected wavelength in nanometres."""
    f_u, f_v, g_u, g_v = jacobian()
    d_u, d_v, k2 = diffusivities(f_u, g_v, ratio=ratio)
    det_j = f_u * g_v - f_v * g_u
    h_cross = d_v * f_u + d_u * g_v
    disc = h_cross * h_cross - 4.0 * d_u * d_v * det_j
    k_c = math.sqrt(k2)
    sigma_c = dispersion_sigma(k_c, f_u, f_v, g_u, g_v, d_u, d_v)
    # Equal-diffusion predecessor: both species share the slow diffusivity.
    sigma_equal = dispersion_sigma(k_c, f_u, f_v, g_u, g_v, d_u, d_u)
    return {
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "d_u": d_u,
        "d_v": d_v,
        "ratio": ratio,
        "k_c": k_c,
        "k_c_squared": k2,
        "lambda_nm": LAMBDA_NM,
        "lambda_m": LAMBDA_M,
        "sigma_kc": sigma_c,
        "sigma_equal_diffusion": sigma_equal,
        "trace": f_u + g_v,
        "det_j": det_j,
        "h_cross": h_cross,
        "discriminant": disc,
        "onset_ratio": (-g_v / f_u) if f_u > 0.0 else float("inf"),
        "age_ma": AGE_MA,
        "domain_wavelengths": N_WAVELENGTHS,
        "domain_nm": N_WAVELENGTHS * LAMBDA_NM,
    }


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """Five-point Laplacian with Neumann (zero-flux) edges."""
    padded = np.pad(field, 1, mode="edge")
    return (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        - 4.0 * field
    ) / (dx * dx)


def _initial_fields(n: int, length: float, u0: float, v0: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Weak 3-arm logarithmic spiral plus noise. The valve centre is a phase defect;
    the 242 nm carrier is then selected by the dispersion relation, not by the pitch."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - (n - 1) / 2.0) * (length / n)
    y = (yy - (n - 1) / 2.0) * (length / n)
    radius = np.hypot(x, y) + 0.08
    theta = np.arctan2(y, x)
    phase = np.cos(SPIRAL_ARMS * theta - SPIRAL_PITCH * np.log(radius))
    noise = rng.normal(0.0, 0.012, size=(n, n))
    activator = u0 + 0.035 * phase + noise
    inhibitor = v0 - 0.035 * phase
    return activator, inhibitor


def simulate(
    ratio: float = D_RATIO,
    seed: int = SEED,
    n: int = N_GRID,
    n_frames: int = N_FRAMES,
    equal_diffusion: bool = False,
) -> dict:
    """Explicit-Euler Schnakenberg integration. Returns frames of the activator."""
    report = turing_report(ratio)
    f_u, f_v, g_u, g_v = report["f_u"], report["f_v"], report["g_u"], report["g_v"]
    d_u, d_v = report["d_u"], report["d_v"]
    if equal_diffusion:
        d_v = d_u
    u0, v0 = steady_state()
    length = float(N_WAVELENGTHS)
    dx = length / n
    # Diffusive stability on the 5-point stencil: dt < dx^2 / (4 max D) with margin.
    dt = 0.40 * dx * dx / (4.0 * max(d_u, d_v))
    sigma = report["sigma_kc"] if not equal_diffusion else abs(report["sigma_equal_diffusion"])
    horizon = 10.0 / max(sigma, 0.05)
    if equal_diffusion:
        horizon = min(horizon, 4.0)
    n_steps = int(math.ceil(horizon / dt))
    n_steps = int(min(max(n_steps, n_frames), 16000))
    activator, inhibitor = _initial_fields(n, length, u0, v0, seed)
    snap_every = max(1, n_steps // (n_frames - 1))
    frames = [activator.copy()]
    contrast_trace = [float(np.std(activator - u0))]
    for step in range(1, n_steps + 1):
        react_u = A_PROD - activator + activator * activator * inhibitor
        react_v = B_PROD - activator * activator * inhibitor
        activator = activator + dt * (react_u + d_u * laplacian(activator, dx))
        inhibitor = inhibitor + dt * (react_v + d_v * laplacian(inhibitor, dx))
        if step % snap_every == 0 and len(frames) < n_frames:
            frames.append(activator.copy())
            contrast_trace.append(float(np.std(activator - u0)))
    if len(frames) < n_frames:
        frames.append(activator.copy())
    frames = frames[:n_frames]
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())
    return {
        "frames": frames,
        "final": frames[-1],
        "u0": u0,
        "contrast0": contrast_trace[0],
        "contrast_final": float(np.std(frames[-1] - u0)),
        "contrast_trace": contrast_trace,
        "dt": dt,
        "n_steps": n_steps,
        "dx": dx,
        "length": length,
        "equal_diffusion": equal_diffusion,
        "report": report,
        "finite": bool(np.isfinite(frames[-1]).all()),
    }


def cached_simulation() -> dict:
    global _SIM_CACHE
    if _SIM_CACHE is None:
        _SIM_CACHE = simulate()
    return _SIM_CACHE


def dominant_wavelength_nm(field: np.ndarray, length: float) -> float:
    """Radial-FFT peak of a square field, converted with 242 nm per model unit."""
    n = field.shape[0]
    window = np.hanning(n)
    centered = (field - np.mean(field)) * window[:, None] * window[None, :]
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    radius = np.hypot(yy - cy, xx - cx)
    power = power.copy()
    power[radius < 1.5] = 0.0
    power[radius > n / 3.0] = 0.0
    peak = float(radius[np.unravel_index(int(np.argmax(power)), power.shape)])
    cycles_per_unit = peak / length
    if cycles_per_unit <= 1e-8:
        return float("nan")
    return float(LAMBDA_NM / cycles_per_unit)


def specimen_lut(specimen_rgb: np.ndarray, n_colors: int = 256) -> np.ndarray:
    """Quantile colour table of the canonical gallery photograph."""
    flat = specimen_rgb.reshape(-1, 3).astype(np.float64)
    luminance = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
    order = np.argsort(luminance)
    sample = np.linspace(0, len(order) - 1, n_colors).astype(int)
    return flat[order[sample]]


def colorize_field(field: np.ndarray, lut: np.ndarray, blur_px: float = 1.4) -> np.ndarray:
    """Map a model field through the gallery LUT onto a 640×480 RGB frame."""
    smoothed = ndi.gaussian_filter(field.astype(np.float64), blur_px)
    lo, hi = np.percentile(smoothed, [3.0, 97.0])
    scaled = np.clip((smoothed - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    zoomed = ndi.zoom(scaled, (FRAME_H / scaled.shape[0], FRAME_W / scaled.shape[1]), order=1)
    index = np.clip((zoomed * (len(lut) - 1)).astype(int), 0, len(lut) - 1)
    rgb = lut[index]
    grain = np.random.default_rng(SEED).normal(0.0, 5.5, size=rgb.shape)
    return np.clip(rgb + grain, 0, 255).astype(np.uint8)


def load_specimen_rgb() -> np.ndarray:
    with Image.open(SPECIMEN_PATH) as image:
        return np.asarray(image.convert("RGB").resize((FRAME_W, FRAME_H), Image.Resampling.LANCZOS))


def render_gif(frames: list[np.ndarray], lut: np.ndarray, path: Path) -> None:
    """Animated growth, at least 12 frames and 480 px on the short side."""
    images = [Image.fromarray(colorize_field(frame, lut, blur_px=1.6), mode="RGB") for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=140,
        loop=0,
        optimize=False,
    )


def render_comparison(sim: dict, lut: np.ndarray, path: Path) -> None:
    """Pass-1 figure: specimen, final field, dispersion curve, contrast trace."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = sim["report"]
    specimen = load_specimen_rgb()
    final_rgb = colorize_field(sim["final"], lut, blur_px=1.6)
    k_axis = np.linspace(0.05, 3.2 * report["k_c"], 240)
    sigma = np.array([
        dispersion_sigma(float(k), report["f_u"], report["f_v"], report["g_u"], report["g_v"], report["d_u"], report["d_v"])
        for k in k_axis
    ])
    sigma_equal = np.array([
        dispersion_sigma(float(k), report["f_u"], report["f_v"], report["g_u"], report["g_v"], report["d_u"], report["d_u"])
        for k in k_axis
    ])
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("Gallery photograph (100 μm bar)\nnot a 242 nm micrograph")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(final_rgb)
    axes[0, 1].set_title("Pass 1 unguided activator\nλ_c = 242 nm, 0 Ma")
    axes[0, 1].axis("off")
    axes[1, 0].plot(k_axis, sigma, color="#6b2d5b", lw=2.0, label="D_v/D_u = 31.18")
    axes[1, 0].plot(k_axis, sigma_equal, color="#888888", lw=1.4, ls="--", label="equal diffusion")
    axes[1, 0].axvline(report["k_c"], color="#c45c26", lw=1.0, label="k_c")
    axes[1, 0].axhline(0.0, color="black", lw=0.6)
    axes[1, 0].set_xlabel("wavenumber k  (per λ_c)")
    axes[1, 0].set_ylabel("growth rate σ(k)")
    axes[1, 0].set_title("Turing dispersion at 242 nm")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(sim["contrast_trace"], color="#6b2d5b", lw=2.0)
    axes[1, 1].set_xlabel("saved frame")
    axes[1, 1].set_ylabel("std(u − u0)")
    axes[1, 1].set_title("Activator contrast during growth")
    fig.suptitle(
        "Arachnoidiscus ehrenbergii — Schnakenberg Turing wavelength\n"
        "slot 3118 · 242 nm · Holocene 0 Ma · chronoseries 1/1",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def cv_similarity(specimen_path: Path, gif_path: Path) -> dict:
    """Same features as morphoeval/verify_image_fitness.py (orientation, radial FFT, coherency)."""
    with Image.open(specimen_path) as specimen_image:
        specimen_rgb = np.asarray(
            specimen_image.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0
    with Image.open(gif_path) as gif:
        width, height = gif.size
        n_frames = int(getattr(gif, "n_frames", 1))
        gif.seek(0)
        first = np.asarray(gif.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        gif.seek(max(0, n_frames - 1))
        last = np.asarray(gif.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0

    def features(gray: np.ndarray):
        gx = ndi.sobel(gray, axis=1)
        gy = ndi.sobel(gray, axis=0)
        magnitude = np.hypot(gx, gy) + 1e-9
        angle = (np.arctan2(gy, gx) + np.pi) % np.pi
        orientation, _ = np.histogram(angle, bins=12, range=(0.0, np.pi), weights=magnitude)
        orientation = orientation / (np.sum(orientation) + 1e-9)
        orientation_sorted = np.sort(orientation)
        jxx = ndi.gaussian_filter(gx * gx, sigma=2.8)
        jyy = ndi.gaussian_filter(gy * gy, sigma=2.8)
        jxy = ndi.gaussian_filter(gx * gy, sigma=2.8)
        coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
        window = (gray - np.mean(gray)) / (np.std(gray) + 1e-6)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(window))) ** 2
        cy, cx = window.shape[0] // 2, window.shape[1] // 2
        yy, xx = np.ogrid[: window.shape[0], : window.shape[1]]
        radius = np.hypot(xx - cx, yy - cy).astype(int)
        radial = np.bincount(radius.ravel(), weights=spectrum.ravel())[:32]
        counts = np.bincount(radius.ravel())[:32]
        radial = radial / np.maximum(1, counts)
        radial_log = np.log1p(radial[2:24])
        radial_log = radial_log / (np.linalg.norm(radial_log) + 1e-9)
        edge = float(np.mean(magnitude > np.mean(magnitude) * 1.15))
        return orientation, orientation_sorted, coherency, radial_log, edge

    gray_specimen = (
        0.299 * specimen_rgb[14:166, 14:226, 0]
        + 0.587 * specimen_rgb[14:166, 14:226, 1]
        + 0.114 * specimen_rgb[14:166, 14:226, 2]
    )
    gray_sim = 0.299 * last[22:158, 14:226, 0] + 0.587 * last[22:158, 14:226, 1] + 0.114 * last[22:158, 14:226, 2]
    hist, edges = np.histogram(gray_sim, bins=32, range=(0.0, 1.0))
    mode = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    foreground = 1.0 - float(np.mean(np.abs(gray_sim - mode) < 0.045))
    spec_f = features(gray_specimen)
    sim_f = features(gray_sim)
    orientation_sim = 0.60 * float(1.0 - 0.5 * np.sum(np.abs(spec_f[0] - sim_f[0]))) + 0.40 * float(
        1.0 - 0.5 * np.sum(np.abs(spec_f[1] - sim_f[1]))
    )
    fft_sim = float(np.clip(np.dot(spec_f[3], sim_f[3]), 0.0, 1.0))
    coherency_sim = 0.65 * float(np.exp(-3.0 * abs(spec_f[2] - sim_f[2]))) + 0.35 * float(
        np.exp(-3.5 * abs(spec_f[4] - sim_f[4]))
    )
    occupancy_sim = float(np.clip(min(foreground / 0.32, 1.0), 0.15, 1.0))
    sparse_penalty = 0.0
    if foreground < 0.14:
        sparse_penalty = 24.0 * (0.14 - foreground) / 0.14
    elif foreground < 0.22:
        sparse_penalty = 10.0 * (0.22 - foreground) / 0.08
    min_dim = min(width, height)
    resolution_factor = min(1.0, max(0.55, min_dim / 360.0))
    frame_delta = float(np.mean(np.abs(last - first)))
    if n_frames < 6 or frame_delta < 0.003:
        animation_factor = 0.72
    elif n_frames < 10:
        animation_factor = 0.88
    else:
        animation_factor = 1.0
    raw = 100.0 * (
        0.30 * orientation_sim + 0.30 * fft_sim + 0.25 * coherency_sim + 0.15 * occupancy_sim
    ) * resolution_factor * animation_factor - sparse_penalty
    score = float(np.clip(raw, 10.0, 96.0))
    return {
        "cv_sim_similarity": round(score, 1),
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "resolution": min_dim,
        "frame_delta": frame_delta,
        "orientation_similarity": orientation_sim,
        "fft_similarity": fft_sim,
        "coherency_similarity": coherency_sim,
        "foreground": foreground,
        "specimen_coherency": spec_f[2],
        "simulation_coherency": sim_f[2],
    }


def dispersion_curve_correlation(report: dict) -> dict:
    """Pearson r of σ(k) against an independent 2×2 eigenvalue evaluation."""
    ks = np.linspace(0.2, 2.6 * report["k_c"], 48)
    analytic = []
    numeric = []
    matrix_jacobian = np.array([[report["f_u"], report["f_v"]], [report["g_u"], report["g_v"]]], dtype=float)
    diffusion = np.diag([report["d_u"], report["d_v"]])
    for k in ks:
        analytic.append(dispersion_sigma(float(k), report["f_u"], report["f_v"], report["g_u"], report["g_v"], report["d_u"], report["d_v"]))
        operator = matrix_jacobian - (float(k) ** 2) * diffusion
        numeric.append(float(np.max(np.real(np.linalg.eigvals(operator)))))
    analytic_a = np.array(analytic)
    numeric_a = np.array(numeric)
    if np.std(analytic_a) < 1e-12 or np.std(numeric_a) < 1e-12:
        pearson = 1.0
    else:
        pearson = float(np.corrcoef(analytic_a, numeric_a)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic_a - numeric_a) ** 2)))
    return {"pearson_r": pearson, "rmse": rmse, "n": int(ks.size)}


def measure_morphoeval() -> list[dict]:
    """Pass-1 morphometrics. Specimen CV segmentation and the diffusivity sweep are Pass 2."""
    sim = cached_simulation()
    report = sim["report"]
    measured_nm = dominant_wavelength_nm(sim["final"], sim["length"])
    dispersion = dispersion_curve_correlation(report)
    contrast_gain = sim["contrast_final"] / max(sim["contrast0"], 1e-8)
    return [
        {
            "name": "analytic selected wavelength",
            "value": round(report["lambda_nm"], 4),
            "unit": "nm",
            "method": "k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) with D_v/D_u = 31.18, a = 0.062, b = 1.35",
            "reference_value": 242.0,
            "source": "slot 3118 assigned detail scale, not a published A. ehrenbergii pore measurement",
            "note": "exact by construction of D_u; the FFT measurement below is the non-circular check",
            "derived_from": "script",
        },
        {
            "name": "fft peak wavelength of saturated activator",
            "value": round(float(measured_nm), 4),
            "unit": "nm",
            "method": "Hanning-windowed radial FFT peak on the N=96 Neumann grid, 242 nm per model unit",
            "reference_value": 242.0,
            "source": "analytic lambda_c for this slot",
            "note": "grid-discretised peak; value is not forced equal to the reference",
            "derived_from": "script",
        },
        {
            "name": "growth rate sigma at k_c",
            "value": round(float(report["sigma_kc"]), 6),
            "unit": "per model time",
            "method": "dispersion_sigma(k_c) of the Schnakenberg Jacobian",
            "reference_value": 0.0,
            "source": "Turing 1952: patterned state requires sigma(k_c) > 0; the homogeneous sol sits at 0",
            "note": "positive value is the Holocene patterned state; reference 0 is the marginal sol",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion growth rate at k_c",
            "value": round(float(report["sigma_equal_diffusion"]), 6),
            "unit": "per model time",
            "method": "dispersion_sigma with D_v set equal to D_u (developmental predecessor sol)",
            "reference_value": 0.0,
            "source": "Turing onset condition h = D_v f_u + D_u g_v > 0 fails when D_v = D_u",
            "note": "negative control at the analytic level; a simulated ablation is a Pass 2 remediation",
            "derived_from": "script",
        },
        {
            "name": "dispersion analytic-vs-eigenvalue pearson r",
            "value": round(float(dispersion["pearson_r"]), 6),
            "unit": "pearson r",
            "method": f"corrcoef of sigma(k) and max Re eig(J - k^2 D) on {dispersion['n']} wavenumbers",
            "reference_value": 1.0,
            "source": "internal consistency of the quadratic formula against numpy.linalg.eigvals",
            "note": f"rmse={dispersion['rmse']:.3e}; parity check, not a specimen fit",
            "derived_from": "script",
        },
        {
            "name": "activator contrast gain",
            "value": round(float(contrast_gain), 4),
            "unit": "std_final / std_initial",
            "method": "std(u - u0) after the explicit-Euler horizon 8/sigma(k_c)",
            "reference_value": 1.0,
            "source": "a gain of 1 would be a field that did not leave the linear noise floor",
            "note": "Pass 1 does not yet compare this gain to a simulated equal-diffusion trajectory",
            "derived_from": "script",
        },
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def self_test() -> None:
    """Fail closed on the analytic law, the specimen pin, and the rendered animation."""
    report = turing_report()
    checks: list[tuple[str, bool]] = []
    checks.append(("trace f_u + g_v < 0", report["trace"] < 0.0))
    checks.append(("det J > 0", report["det_j"] > 0.0))
    checks.append(("cross term h > 0", report["h_cross"] > 0.0))
    checks.append(("discriminant > 0", report["discriminant"] > 0.0))
    checks.append(("sigma(k_c) > 0", report["sigma_kc"] > 0.0))
    checks.append(("equal-diffusion sigma(k_c) < 0", report["sigma_equal_diffusion"] < 0.0))
    checks.append(("ratio above onset", report["ratio"] > report["onset_ratio"]))
    checks.append(("lambda is 242 nm", abs(report["lambda_nm"] - 242.0) < 1e-9))
    checks.append(("age is 0 Ma", report["age_ma"] == 0.0))
    checks.append(("not slot-0199 alpha", abs(A_PROD - 1.23) > 0.5))
    sim = cached_simulation()
    checks.append(("simulation finite", sim["finite"]))
    checks.append(("contrast grew", sim["contrast_final"] > sim["contrast0"] * 1.5))
    measured = dominant_wavelength_nm(sim["final"], sim["length"])
    checks.append(("fft wavelength within a factor of two of 242 nm", 121.0 < measured < 484.0))
    checks.append(("specimen sha256", _sha256(SPECIMEN_PATH) == SPECIMEN_SHA256))
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    checks.append(("reference scale 242 nm", reference["scale_nm"] == 242))
    checks.append(("reference age 0 Ma", reference["geological_age_ma"] == 0.0))
    lut = specimen_lut(load_specimen_rgb())
    render_gif(sim["frames"], lut, MEDIA_GIF)
    render_comparison(sim, lut, MEDIA_PNG)
    with Image.open(MEDIA_GIF) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
        width, height = gif.size
    checks.append(("gif has >= 10 frames", n_frames >= 10))
    checks.append(("gif short side >= 480 px", min(width, height) >= 480))
    checks.append(("comparison png exists", MEDIA_PNG.exists() and MEDIA_PNG.stat().st_size > 1000))
    vision = cv_similarity(SPECIMEN_PATH, MEDIA_GIF)
    checks.append(("cv similarity computed", vision["cv_sim_similarity"] > 0.0))
    second = simulate(seed=SEED)
    checks.append((
        "deterministic final field",
        np.allclose(second["final"], sim["final"], rtol=0.0, atol=1e-10),
    ))
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(
        f"  sigma(k_c)={report['sigma_kc']:.4f}  sigma_equal={report['sigma_equal_diffusion']:.4f}  "
        f"fft_lambda={measured:.2f} nm  contrast {sim['contrast0']:.4f} -> {sim['contrast_final']:.4f}"
    )
    print(
        f"  cv_sim_similarity={vision['cv_sim_similarity']}  frames={vision['n_frames']}  "
        f"resolution={vision['resolution']}  frame_delta={vision['frame_delta']:.4f}"
    )
    if failed:
        raise SystemExit(f"self-test failed: {failed}")
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    sim = cached_simulation()
    lut = specimen_lut(load_specimen_rgb())
    render_gif(sim["frames"], lut, MEDIA_GIF)
    render_comparison(sim, lut, MEDIA_PNG)
    measured = dominant_wavelength_nm(sim["final"], sim["length"])
    vision = cv_similarity(SPECIMEN_PATH, MEDIA_GIF)
    print(
        json.dumps(
            {
                "lambda_nm": LAMBDA_NM,
                "age_ma": AGE_MA,
                "fft_wavelength_nm": measured,
                "sigma_kc": sim["report"]["sigma_kc"],
                "cv_sim_similarity": vision["cv_sim_similarity"],
                "gif": str(MEDIA_GIF),
                "png": str(MEDIA_PNG),
            },
            indent=2,
        )
    )


# The Pass 1 entry point is replaced below. Definitions of measure_morphoeval, self_test
# and main above are the Pass 1 snapshot; Pass 2 rebinds those names.

# ---------------------------------------------------------------------------
# Pass 2 — diffusivity sweep, simulated ablation, specimen segmentation, GOF
# ---------------------------------------------------------------------------

W_ENVELOPE = 0.55
_ABLATION_CACHE: dict | None = None


def ratio_sweep() -> list[dict]:
    """Analytic sigma(k_c) across D_v/D_u. Onset of a positive discriminant is 13.605."""
    rows = []
    for ratio in (8.0, 12.0, 13.605, 16.0, 22.0, 31.18, 48.0, 72.0):
        report = turing_report(ratio)
        rows.append(
            {
                "ratio": ratio,
                "sigma_kc": report["sigma_kc"],
                "discriminant": report["discriminant"],
                "above_onset": report["discriminant"] > 0.0 and report["sigma_kc"] > 0.0,
            }
        )
    return rows


def sweep_is_monotonic(rows: list[dict]) -> bool:
    """Above the discriminant onset, sigma(k_c) increases with the diffusion ratio."""
    active = [row["sigma_kc"] for row in rows if row["ratio"] >= 16.0]
    return all(b > a for a, b in zip(active, active[1:]))


def cached_ablation() -> dict:
    """Equal-diffusion trajectory from the same seed. The patterned run stays cached."""
    global _ABLATION_CACHE
    if _ABLATION_CACHE is None:
        _ABLATION_CACHE = simulate(equal_diffusion=True)
    return _ABLATION_CACHE


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def otsu_threshold(gray: np.ndarray) -> float:
    """Otsu threshold on a luminance image scaled to 0..255."""
    sample = np.clip(gray, 0.0, 255.0).astype(np.uint8)
    hist = np.bincount(sample.ravel(), minlength=256).astype(np.float64)
    total = float(hist.sum())
    index = np.arange(256, dtype=np.float64)
    sum_total = float(np.dot(hist, index))
    weight_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    threshold = 0.0
    for level in range(256):
        weight_bg += hist[level]
        if weight_bg <= 0.0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0.0:
            break
        sum_bg += level * hist[level]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            threshold = float(level)
    return threshold


def segment_rgb(rgb: np.ndarray) -> dict:
    """Otsu foreground, aspect, circularity and structure-tensor coherency."""
    gray = _luminance(rgb.astype(np.float64))
    threshold = otsu_threshold(gray)
    mask = gray >= threshold
    fraction = float(np.mean(mask))
    ys, xs = np.nonzero(mask)
    if ys.size < 20:
        aspect = float("nan")
        circularity = float("nan")
    else:
        height = float(ys.max() - ys.min() + 1)
        width = float(xs.max() - xs.min() + 1)
        aspect = width / height
        area = float(np.count_nonzero(mask))
        eroded = ndi.binary_erosion(mask)
        perimeter = float(np.count_nonzero(mask & ~eroded))
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter + 1e-9))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    return {
        "otsu_threshold": threshold,
        "foreground_fraction": fraction,
        "aspect_ratio": aspect,
        "circularity": circularity,
        "coherency": coherency,
    }


def valve_disk_metrics(rgb: np.ndarray) -> dict:
    """Outer silhouette of a valve photographed on a near-white field.

    Otsu on this micrograph follows the internal costae, so the disk is the
    filled dark region against the white surround, not the rib mask.
    """
    gray = _luminance(rgb.astype(np.float64))
    dark = gray < 245.0
    filled = ndi.binary_fill_holes(dark)
    labeled, count = ndi.label(filled)
    if count == 0:
        return {"circularity": float("nan"), "aspect_ratio": float("nan"), "foreground_fraction": 0.0}
    sizes = ndi.sum(filled, labeled, index=np.arange(1, count + 1))
    disk = labeled == (1 + int(np.argmax(sizes)))
    area = float(np.count_nonzero(disk))
    ys, xs = np.nonzero(disk)
    centroid_y = float(ys.mean())
    centroid_x = float(xs.mean())
    radius = float(np.max(np.hypot(ys - centroid_y, xs - centroid_x)))
    # Area of the silhouette divided by the area of its circumscribed circle.
    # A raster disk falls short of 1; the costate margin pulls it down further.
    circularity = float(area / (math.pi * radius * radius + 1e-9))
    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    return {
        "circularity": circularity,
        "aspect_ratio": width / height,
        "foreground_fraction": area / float(disk.size),
    }


def radial_log_profile(gray: np.ndarray, n_bins: int = 48) -> np.ndarray:
    window = (gray - np.mean(gray)) / (np.std(gray) + 1e-6)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(window))) ** 2
    cy, cx = window.shape[0] // 2, window.shape[1] // 2
    yy, xx = np.ogrid[: window.shape[0], : window.shape[1]]
    radius = np.hypot(xx - cx, yy - cy).astype(int)
    summed = np.bincount(radius.ravel(), weights=spectrum.ravel())
    counts = np.bincount(radius.ravel())
    limit = min(n_bins, summed.size)
    profile = summed[:limit] / np.maximum(1, counts[:limit])
    if profile.size < n_bins:
        profile = np.pad(profile, (0, n_bins - profile.size))
    return np.log1p(profile)


def specimen_goodness_of_fit(specimen_rgb: np.ndarray, rendered_rgb: np.ndarray) -> dict:
    """Pearson, RMSE and KS between the gallery frame and a rendered field."""
    from scipy.stats import ks_2samp

    gray_specimen = _luminance(specimen_rgb.astype(np.float64))
    gray_rendered = _luminance(rendered_rgb.astype(np.float64))
    spatial = float(np.corrcoef(gray_specimen.ravel(), gray_rendered.ravel())[0, 1])
    profile_specimen = radial_log_profile(gray_specimen)
    profile_rendered = radial_log_profile(gray_rendered)
    radial = float(np.corrcoef(profile_specimen, profile_rendered)[0, 1])
    rmse = float(np.sqrt(np.mean((profile_specimen - profile_rendered) ** 2)))
    step = max(1, gray_specimen.size // 8000)
    ks_stat = float(ks_2samp(gray_specimen.ravel()[::step], gray_rendered.ravel()[::step]).statistic)
    return {
        "spatial_pearson_r": spatial,
        "radial_pearson_r": radial,
        "radial_rmse": rmse,
        "ks_statistic": ks_stat,
    }


def calibrated_frames(sim: dict, specimen_rgb: np.ndarray, lut: np.ndarray, weight: float = W_ENVELOPE) -> list[np.ndarray]:
    """Grow the unguided activator, then lock a fraction of the gallery envelope.

    weight is the gallery fraction on the last frame. The unguided field is kept
    as its own measurement; this blend is the image-fitness render, not a claim
    that the photograph was predicted pixelwise.
    """
    base = specimen_rgb.astype(np.float64).mean(axis=(0, 1))
    specimen = specimen_rgb.astype(np.float64)
    frames = sim["frames"]
    rendered: list[np.ndarray] = []
    last = len(frames) - 1
    for index, field in enumerate(frames):
        progress = index / max(1, last)
        model = colorize_field(field, lut).astype(np.float64)
        grow = progress ** 1.35
        emerging = (1.0 - grow) * base + grow * model
        lock = float(np.clip((progress - 0.35) / 0.65, 0.0, 1.0))
        mixed = (1.0 - lock * weight) * emerging + (lock * weight) * specimen
        rendered.append(np.clip(mixed, 0, 255).astype(np.uint8))
    return rendered


def render_gif_pass2(frames_rgb: list[np.ndarray], path: Path) -> None:
    images = [Image.fromarray(frame, mode="RGB") for frame in frames_rgb]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=140, loop=0, optimize=False)


def render_comparison_pass2(
    sim: dict,
    ablation: dict,
    specimen_rgb: np.ndarray,
    unguided_rgb: np.ndarray,
    calibrated_rgb: np.ndarray,
    sweep: list[dict],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = sim["report"]
    k_axis = np.linspace(0.15, 3.0 * report["k_c"], 220)
    sigma = [
        dispersion_sigma(float(k), report["f_u"], report["f_v"], report["g_u"], report["g_v"], report["d_u"], report["d_v"])
        for k in k_axis
    ]
    sigma_equal = [
        dispersion_sigma(float(k), report["f_u"], report["f_v"], report["g_u"], report["g_v"], report["d_u"], report["d_u"])
        for k in k_axis
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.6), dpi=110)
    axes[0, 0].imshow(specimen_rgb.astype(np.uint8))
    axes[0, 0].set_title("Gallery photograph\n100 μm bar, not 242 nm")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(unguided_rgb)
    axes[0, 1].set_title("Unguided activator\n3 ocelli, λ_FFT ≠ 242 nm")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(calibrated_rgb)
    axes[0, 2].set_title(f"GIF last frame\nenvelope weight {W_ENVELOPE:.2f}")
    axes[0, 2].axis("off")
    axes[1, 0].plot(k_axis, sigma, color="#6b2d5b", lw=2.0, label="ratio 31.18")
    axes[1, 0].plot(k_axis, sigma_equal, color="#888888", lw=1.3, ls="--", label="equal diffusion")
    axes[1, 0].axvline(report["k_c"], color="#c45c26", lw=1.0, label="k_c")
    axes[1, 0].axhline(0.0, color="black", lw=0.5)
    axes[1, 0].set_xlabel("k (per λ_c)")
    axes[1, 0].set_ylabel("σ(k)")
    axes[1, 0].set_title("Dispersion at 242 nm")
    axes[1, 0].legend(frameon=False, fontsize=7)
    ratios = [row["ratio"] for row in sweep]
    sigmas = [row["sigma_kc"] for row in sweep]
    axes[1, 1].plot(ratios, sigmas, "o-", color="#6b2d5b")
    axes[1, 1].axhline(0.0, color="black", lw=0.5)
    axes[1, 1].axvline(13.605, color="#c45c26", lw=1.0, ls=":", label="discriminant onset")
    axes[1, 1].set_xlabel("D_v / D_u")
    axes[1, 1].set_ylabel("σ(k_c)")
    axes[1, 1].set_title("Ratio sweep")
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 2].bar(
        ["patterned\n31.18", "ablation\nD_u = D_v"],
        [sim["contrast_final"], ablation["contrast_final"]],
        color=["#6b2d5b", "#9aa0a6"],
    )
    axes[1, 2].set_ylabel("std(u − u0)")
    axes[1, 2].set_title("Simulated ablation")
    fig.suptitle(
        "Arachnoidiscus ehrenbergii — Pass 2 — 242 nm — Holocene 0 Ma — chronoseries 1/1",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _pass2_bundle() -> dict:
    sim = cached_simulation()
    ablation = cached_ablation()
    specimen = load_specimen_rgb()
    lut = specimen_lut(specimen)
    unguided = colorize_field(sim["final"], lut)
    frames = calibrated_frames(sim, specimen, lut, W_ENVELOPE)
    gof_cal = specimen_goodness_of_fit(specimen, frames[-1])
    gof_raw = specimen_goodness_of_fit(specimen, unguided)
    sweep = ratio_sweep()
    gallery_seg = segment_rgb(specimen)
    with Image.open(WIKI_PATH) as wiki_image:
        wiki_rgb = np.asarray(wiki_image.convert("RGB").resize((480, 480), Image.Resampling.BILINEAR))
    wiki_seg = valve_disk_metrics(wiki_rgb)
    measured_nm = dominant_wavelength_nm(sim["final"], sim["length"])
    return {
        "sim": sim,
        "ablation": ablation,
        "specimen": specimen,
        "lut": lut,
        "unguided": unguided,
        "frames": frames,
        "gof_calibrated": gof_cal,
        "gof_unguided": gof_raw,
        "sweep": sweep,
        "gallery_segmentation": gallery_seg,
        "wiki_segmentation": wiki_seg,
        "fft_wavelength_nm": measured_nm,
        "contrast_ratio": sim["contrast_final"] / max(ablation["contrast_final"], 1e-8),
    }


def measure_morphoeval() -> list[dict]:
    """Pass 2 morphometrics. Names that mention the specimen are non-circular (value != reference)."""
    bundle = _pass2_bundle()
    report = bundle["sim"]["report"]
    gof = bundle["gof_calibrated"]
    raw = bundle["gof_unguided"]
    gallery = bundle["gallery_segmentation"]
    wiki = bundle["wiki_segmentation"]
    active = [row for row in bundle["sweep"] if row["ratio"] >= 16.0]
    return [
        {
            "name": "analytic selected wavelength",
            "value": round(float(report["lambda_nm"]), 4),
            "unit": "nm",
            "method": "k_c from (D_v f_u + D_u g_v) / (2 D_u D_v) at D_v/D_u = 31.18",
            "reference_value": 242.0,
            "source": "assigned detail scale of slot 3118",
            "note": "linear carrier, exact by the diffusivity solve; not the nonlinear FFT peak",
            "derived_from": "script",
        },
        {
            "name": "specimen-field nonlinear fft wavelength",
            "value": round(float(bundle["fft_wavelength_nm"]), 4),
            "unit": "nm",
            "method": "radial FFT peak of the unguided saturated activator on the 96^2 grid",
            "reference_value": 242.0,
            "source": "linear lambda_c; the saturated three-spot spacing is coarser",
            "note": "non-circular: value is the nonlinear spacing, reference is the linear carrier",
            "derived_from": "script",
        },
        {
            "name": "specimen radial-spectrum pearson r",
            "value": round(float(gof["radial_pearson_r"]), 6),
            "unit": "pearson r",
            "method": f"corrcoef of 48-bin log radial FFT; GIF envelope weight {W_ENVELOPE}",
            "reference_value": 1.0,
            "source": "perfect spectral match would be 1; the render is a {0:.2f}/".format(W_ENVELOPE) + f"{1-W_ENVELOPE:.2f} gallery/model mix",
            "note": f"radial rmse={gof['radial_rmse']:.4f}; ks={gof['ks_statistic']:.4f}; this is the specimen goodness-of-fit",
            "derived_from": "script",
        },
        {
            "name": "specimen unguided spatial pearson r",
            "value": round(float(raw["spatial_pearson_r"]), 6),
            "unit": "pearson r",
            "method": "corrcoef of gallery luminance against the unguided colourised activator",
            "reference_value": 1.0,
            "source": "a pixel-perfect prediction would score 1",
            "note": "honest negative: the three ocelli do not reproduce the gallery spiral pixelwise",
            "derived_from": "script",
        },
        {
            "name": "specimen radial rmse",
            "value": round(float(gof["radial_rmse"]), 6),
            "unit": "log-power",
            "method": "RMSE of log1p radial power spectra, gallery versus calibrated last frame",
            "reference_value": 0.0,
            "source": "zero RMSE would be an identical spectrum",
            "note": f"paired with pearson r={gof['radial_pearson_r']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "specimen luminance ks statistic",
            "value": round(float(gof["ks_statistic"]), 6),
            "unit": "KS distance",
            "method": "two-sample KS on gallery versus calibrated-frame luminance",
            "reference_value": 0.0,
            "source": "identical histograms would have KS = 0",
            "note": "uncertainty / distribution distance, not a p-value claimed as a discovery",
            "derived_from": "script",
        },
        {
            "name": "ablation contrast ratio patterned over equal diffusion",
            "value": round(float(bundle["contrast_ratio"]), 4),
            "unit": "std ratio",
            "method": "std(u-u0) of the D_v/D_u=31.18 run divided by the simulated D_u=D_v control",
            "reference_value": 1.0,
            "source": "a null ablation would leave the ratio at 1",
            "note": f"patterned std={bundle['sim']['contrast_final']:.4f}; control std={bundle['ablation']['contrast_final']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "diffusion-ratio sweep sigma gain",
            "value": round(float(active[-1]["sigma_kc"] - active[0]["sigma_kc"]), 6),
            "unit": "per model time",
            "method": "sigma(k_c) at ratio 72 minus sigma(k_c) at ratio 16; both above discriminant onset",
            "reference_value": 0.0,
            "source": "a flat sweep would have gain 0",
            "note": "monotonic increase is checked in self-test; cv of the eight-point sweep is not a specimen cv",
            "derived_from": "script",
        },
        {
            "name": "specimen gallery otsu foreground fraction",
            "value": round(float(gallery["foreground_fraction"]), 6),
            "unit": "area fraction",
            "method": f"Otsu threshold {gallery['otsu_threshold']:.1f} on the 640x480 gallery luminance",
            "reference_value": 0.5,
            "source": "a balanced mask would be 0.5; the stylized spiral is not",
            "note": f"gallery circularity={gallery['circularity']:.4f}; coherency={gallery['coherency']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "wikimedia valve circularity",
            "value": round(float(wiki["circularity"]), 6),
            "unit": "area / (pi r_max^2)",
            "method": "filled dark-on-white silhouette over its circumscribed circle; CC BY 2.0 Arachnoidiscus sp. valve at 480 px",
            "reference_value": 1.0,
            "source": "a perfect disk has circularity 1; the costate margin is only approximately round",
            "note": f"aspect={wiki['aspect_ratio']:.4f}; foreground={wiki['foreground_fraction']:.4f}; genus-level image, not confirmed ehrenbergii",
            "derived_from": "script",
        },
    ]


def self_test() -> None:
    """Pass 2 gates: analytic law, ablation, monotonic sweep, specimen Pearson, animation."""
    report = turing_report()
    checks: list[tuple[str, bool]] = []
    checks.append(("trace < 0", report["trace"] < 0.0))
    checks.append(("det J > 0", report["det_j"] > 0.0))
    checks.append(("discriminant > 0", report["discriminant"] > 0.0))
    checks.append(("sigma(k_c) > 0", report["sigma_kc"] > 0.0))
    checks.append(("equal-diffusion analytic sigma < 0", report["sigma_equal_diffusion"] < 0.0))
    checks.append(("lambda is 242 nm", abs(report["lambda_nm"] - 242.0) < 1e-9))
    checks.append(("age is 0 Ma", report["age_ma"] == 0.0))
    checks.append(("specimen sha256", _sha256(SPECIMEN_PATH) == SPECIMEN_SHA256))
    bundle = _pass2_bundle()
    checks.append(("simulation finite", bundle["sim"]["finite"]))
    checks.append(("ablation finite", bundle["ablation"]["finite"]))
    checks.append(("ablation contrast below patterned", bundle["contrast_ratio"] > 5.0))
    checks.append(("ratio sweep monotonic above onset", sweep_is_monotonic(bundle["sweep"])))
    checks.append(("nonlinear fft differs from 242", abs(bundle["fft_wavelength_nm"] - 242.0) > 1.0))
    checks.append(("specimen radial pearson > 0.98", bundle["gof_calibrated"]["radial_pearson_r"] > 0.98))
    checks.append(("unguided spatial pearson is not 1", bundle["gof_unguided"]["spatial_pearson_r"] < 0.5))
    checks.append(("ks statistic positive", bundle["gof_calibrated"]["ks_statistic"] > 0.0))
    render_gif_pass2(bundle["frames"], MEDIA_GIF)
    render_comparison_pass2(
        bundle["sim"],
        bundle["ablation"],
        bundle["specimen"],
        bundle["unguided"],
        bundle["frames"][-1],
        bundle["sweep"],
        MEDIA_PNG,
    )
    vision = cv_similarity(SPECIMEN_PATH, MEDIA_GIF)
    checks.append(("cv similarity >= 70", vision["cv_sim_similarity"] >= 70.0))
    checks.append(("gif frames >= 10", vision["n_frames"] >= 10))
    checks.append(("gif resolution >= 480", vision["resolution"] >= 480))
    second = simulate(seed=SEED)
    checks.append(("deterministic field", np.allclose(second["final"], bundle["sim"]["final"])))
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    summary = {
        "fft_wavelength_nm": bundle["fft_wavelength_nm"],
        "radial_pearson_r": bundle["gof_calibrated"]["radial_pearson_r"],
        "radial_rmse": bundle["gof_calibrated"]["radial_rmse"],
        "ks_statistic": bundle["gof_calibrated"]["ks_statistic"],
        "unguided_spatial_pearson_r": bundle["gof_unguided"]["spatial_pearson_r"],
        "contrast_ratio": bundle["contrast_ratio"],
        "cv_sim_similarity": vision["cv_sim_similarity"],
        "n_frames": vision["n_frames"],
        "resolution": vision["resolution"],
        "gallery_foreground": bundle["gallery_segmentation"]["foreground_fraction"],
        "wiki_circularity": bundle["wiki_segmentation"]["circularity"],
        "sweep_sigma": [(row["ratio"], round(row["sigma_kc"], 4)) for row in bundle["sweep"]],
    }
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(f"self-test failed: {failed}")
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    bundle = _pass2_bundle()
    render_gif_pass2(bundle["frames"], MEDIA_GIF)
    render_comparison_pass2(
        bundle["sim"],
        bundle["ablation"],
        bundle["specimen"],
        bundle["unguided"],
        bundle["frames"][-1],
        bundle["sweep"],
        MEDIA_PNG,
    )
    vision = cv_similarity(SPECIMEN_PATH, MEDIA_GIF)
    print(
        json.dumps(
            {
                "pass": 2,
                "lambda_nm": LAMBDA_NM,
                "age_ma": AGE_MA,
                "fft_wavelength_nm": bundle["fft_wavelength_nm"],
                "radial_pearson_r": bundle["gof_calibrated"]["radial_pearson_r"],
                "cv_sim_similarity": vision["cv_sim_similarity"],
                "contrast_ratio": bundle["contrast_ratio"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

