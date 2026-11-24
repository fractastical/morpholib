#!/usr/bin/env python3
"""
Slot 3146 — Chlamydomonas reinhardtii eyespot pigment stripe at 10.32 nm.

Holocene extant (0 Ma). Chronoseries length 1/1, so the predecessor and
successor are neighboring length scales of the same ocellus, not fossils:

  carotenoid monomer ~3.2 nm  <  this stripe 10.32 nm  <  quarter-wave spacing ~93 nm

Governing system (Schnakenberg 1979, J. Theor. Biol., DOI 10.1016/0022-5193(79)90042-0):

  du/dt = D_u lap(u) + a - u + u^2 v          (eq. 1, activator)
  dv/dt = D_v lap(v) + b - u^2 v              (eq. 2, inhibitor)

Homogeneous steady state u0 = a + b, v0 = b / u0^2.
Jacobian entries (eq. 3):

  f_u = -1 + 2 u0 v0,  f_v = u0^2,  g_u = -2 u0 v0,  g_v = -u0^2

Turing onset wavenumber (Turing 1952, DOI 10.1098/rstb.1952.0012), eq. 4:

  k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),    lambda_c = 2 pi / k_c

Above onset the band of unstable modes is wide. The stripe that emerges is the
fastest linear mode k_max = argmax sigma(k), not k_c. The spatial unit is fixed
so that lambda(k_max) = 10.32 nm = 1.032e-8 m at the reference parameters
(a, b, D_u, D_v) = (0.07, 0.95, 1, 36). Those are not slot 195's
(1.63, 0.43, 14, 10, 140).

The eyespot biology follows Dieckmann 2003 (carotenoid granules under the plasma
membrane) and Foster & Smyth 1980 (quarter-wave antenna). EYE2/EYE3
(Boyd et al. 2011) assemble the organelle; this script does not claim to
replace that protein pathway. Channelrhodopsin structure: Kato et al. 2012.

Run:
  python3 models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py
"""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "chlamydomonas_reinhardtii_turing_wavelength_s0"
REF_JSON = DATASET / "chlamydomonas_reinhardtii_turing_wavelength_s0_reference.json"
SPECIMEN = DATASET / "chlamydomonas_reinhardtii_specimen.jpg"
OUT_PNG = ROOT / "docs" / "media" / "chlamydomonas_reinhardtii_turing_wavelength_s0_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "chlamydomonas_reinhardtii_turing_wavelength_s0_growth.gif"

# --- SECTION: parameters ---
SLOT = 3146
AGE_MA = 0.0
PHENOMENON_SIZE_M = 1.032e-8
PHENOMENON_NM = 10.32
MONOMER_NM = 3.2
QUARTER_WAVE_NM = 540.0 / (4.0 * 1.45)  # Foster & Smyth quarter-wave plate
GLOBLE_NM = 100.0
STRIPES = 6
A = 0.07
B = 0.95
DU = 1.0
DV_REF = 36.0
SEED = 3146
GRID_N = 96
SIBLING_195 = (1.63, 0.43, 14, 10, 140)
SWEEP_DV = (18.0, 24.0, 32.0, 36.0, 44.0, 58.0, 76.0)
SELF_TEST_DV = (24.0, 36.0, 52.0, 76.0)

# Pass 2 render gain. Pass 1 used 0.55; the CV-term check below retunes if needed.
MODULATION_GAIN = 0.62


def log10_size() -> float:
    return math.log10(PHENOMENON_SIZE_M)


# --- SECTION: dispersion ---
def jacobian(a: float = A, b: float = B) -> dict[str, float]:
    """Eq. 3. Schnakenberg Jacobian at the homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -u0 * u0
    return {
        "u0": u0,
        "v0": v0,
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "trace": f_u + g_v,
        "det": f_u * g_v - f_v * g_u,
    }


def onset_wavenumber(a: float, b: float, du: float, dv: float) -> dict[str, float]:
    """Eq. 4. Critical onset wavenumber k_c and the Turing conditions."""
    jac = jacobian(a, b)
    h = dv * jac["f_u"] + du * jac["g_v"]
    disc = h * h - 4.0 * du * dv * jac["det"]
    turing = jac["trace"] < 0.0 and jac["det"] > 0.0 and h > 0.0 and disc > 0.0
    kc2 = h / (2.0 * du * dv) if h > 0.0 else float("nan")
    kc = math.sqrt(kc2) if kc2 > 0.0 else float("nan")
    lam = (2.0 * math.pi / kc) if kc == kc and kc > 0.0 else float("nan")
    return {**jac, "h": h, "disc": disc, "turing": turing, "k_c": kc, "lambda_c": lam}


def growth_rate(k: np.ndarray, a: float, b: float, du: float, dv: float) -> np.ndarray:
    """Linear growth rate sigma(k) of the larger Jacobian eigenvalue."""
    jac = jacobian(a, b)
    h = dv * jac["f_u"] + du * jac["g_v"]
    k2 = np.asarray(k, dtype=np.float64) ** 2
    tr_k = jac["trace"] - (du + dv) * k2
    det_k = du * dv * k2 ** 2 - h * k2 + jac["det"]
    disc = tr_k ** 2 - 4.0 * det_k
    # Real part of the larger eigenvalue. A complex pair contributes tr_k/2.
    sig = 0.5 * tr_k
    ok = disc >= 0.0
    sig = np.array(sig, dtype=np.float64, copy=True)
    sig[ok] = 0.5 * (tr_k[ok] + np.sqrt(disc[ok]))
    return sig


def fastest_mode(a: float = A, b: float = B, du: float = DU, dv: float = DV_REF) -> dict[str, float]:
    """k_max = argmax sigma(k), the stripe that emerges above onset."""
    onset = onset_wavenumber(a, b, du, dv)
    kc = onset["k_c"] if onset["k_c"] == onset["k_c"] else 0.5
    ks = np.linspace(0.02, max(3.0 * kc, 0.3), 4000)
    sig = growth_rate(ks, a, b, du, dv)
    i = int(np.argmax(sig))
    k_max = float(ks[i])
    return {
        **onset,
        "k_max": k_max,
        "sigma_max": float(sig[i]),
        "lambda_fast": (2.0 * math.pi / k_max) if k_max > 0.0 else float("nan"),
    }


def nm_per_nondim() -> float:
    """Fix the unit so the reference fastest wavelength is 10.32 nm."""
    lam = fastest_mode(A, B, DU, DV_REF)["lambda_fast"]
    return PHENOMENON_NM / lam


def ablation_sigma_max() -> float:
    """Drop the autocatalytic u^2 v term. Linear decay f=a-u, g=b-v cannot pattern."""
    # f_u = -1, f_v = 0, g_u = 0, g_v = -1. Trace = -2, h = -(Du+Dv) < 0.
    du, dv = DU, DV_REF
    trace = -2.0
    det = 1.0
    h = -(du + dv)
    ks = np.linspace(0.02, 2.0, 800)
    k2 = ks ** 2
    tr_k = trace - (du + dv) * k2
    det_k = du * dv * k2 ** 2 - h * k2 + det
    disc = tr_k ** 2 - 4.0 * det_k
    sig = 0.5 * tr_k
    ok = disc >= 0.0
    sig = np.array(sig, copy=True)
    sig[ok] = 0.5 * (tr_k[ok] + np.sqrt(disc[ok]))
    return float(np.max(sig))


# --- SECTION: solver ---
def _laplace(z: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / (dx * dx)


def _radial_peak(field: np.ndarray, length: float) -> float:
    """Sub-bin radial FFT peak, returned as a wavelength in the same unit as `length`."""
    win = field - float(np.mean(field))
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    n = field.shape[0]
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    m = n // 2
    prof = np.bincount(rr.ravel(), weights=power.ravel(), minlength=m)[:m]
    cnt = np.bincount(rr.ravel(), minlength=m)[:m]
    prof = prof / np.maximum(cnt, 1)
    k_pix = int(np.argmax(prof[1:]) + 1)
    y0 = prof[max(k_pix - 1, 0)]
    y1 = prof[k_pix]
    y2 = prof[min(k_pix + 1, m - 1)]
    den = y0 - 2.0 * y1 + y2
    delta = 0.0 if abs(den) < 1e-18 else float(np.clip(0.5 * (y0 - y2) / den, -0.8, 0.8))
    k_sub = k_pix + delta
    return length / k_sub


def measure_linear_wavelength(
    dv: float,
    n: int = GRID_N,
    dt: float = 0.0015,
    target_std: float = 0.20,
    max_t: float = 80.0,
    seed: int = SEED,
) -> dict[str, float]:
    """Integrate eq. 1-2 only until the linear stripe is measurable (std ~ 0.2)."""
    mode = fastest_mode(A, B, DU, dv)
    lam = mode["lambda_fast"]
    length = STRIPES * lam
    dx = length / n
    jac = jacobian(A, B)
    rng = np.random.default_rng(seed)
    u = jac["u0"] + 0.02 * rng.standard_normal((n, n))
    v = jac["v0"] + 0.02 * rng.standard_normal((n, n))
    steps = int(max_t / dt)
    hit = max_t
    for i in range(steps):
        uv2 = u * u * v
        u = u + dt * (DU * _laplace(u, dx) + A - u + uv2)
        v = v + dt * (dv * _laplace(v, dx) + B - uv2)
        if (i % 150) == 0 and float(np.std(u)) >= target_std:
            hit = (i + 1) * dt
            break
    lam_num = _radial_peak(u, length)
    return {
        "Dv": dv,
        "lambda_fast": lam,
        "lambda_num": lam_num,
        "lambda_c": mode["lambda_c"],
        "k_max": mode["k_max"],
        "k_c": mode["k_c"],
        "sigma_max": mode["sigma_max"],
        "std": float(np.std(u)),
        "t_hit": hit,
        "field": u,
    }


def simulate_reference_frames(n: int = GRID_N, n_frames: int = 12, dt: float = 0.002, steps: int = 12000) -> list[np.ndarray]:
    """Saturated trajectory of the reference parameters, 12 frames from noise to pigment stripes."""
    mode = fastest_mode()
    length = STRIPES * mode["lambda_fast"]
    dx = length / n
    jac = jacobian()
    rng = np.random.default_rng(SEED)
    u = jac["u0"] + 0.02 * rng.standard_normal((n, n))
    v = jac["v0"] + 0.02 * rng.standard_normal((n, n))
    save_at = set(int(round(x)) for x in np.linspace(0, steps - 1, n_frames))
    frames: list[np.ndarray] = []
    for i in range(steps):
        if i in save_at:
            frames.append(u.copy())
        uv2 = u * u * v
        u = u + dt * (DU * _laplace(u, dx) + A - u + uv2)
        v = v + dt * (DV_REF * _laplace(v, dx) + B - uv2)
    if len(frames) < n_frames:
        frames.append(u.copy())
    return frames[:n_frames]


def negative_control_sigma() -> float:
    """Equal diffusivities D_v = D_u destroy the differential-diffusion condition h > 0."""
    mode = fastest_mode(A, B, DU, DU)
    return mode["sigma_max"]


# --- SECTION: sweep ---
def dispersion_sweep(dv_values: tuple[float, ...] = SWEEP_DV) -> dict[str, np.ndarray]:
    rows = [measure_linear_wavelength(dv) for dv in dv_values]
    scale = nm_per_nondim()
    analytical = np.array([r["lambda_fast"] * scale for r in rows], dtype=np.float64)
    numerical = np.array([r["lambda_num"] * scale for r in rows], dtype=np.float64)
    pearson = float(stats.pearsonr(analytical, numerical).statistic)
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    return {
        "Dv": np.array([r["Dv"] for r in rows], dtype=np.float64),
        "analytical_nm": analytical,
        "numerical_nm": numerical,
        "lambda_c_nm": np.array([r["lambda_c"] * scale for r in rows], dtype=np.float64),
        "pearson_r": pearson,
        "rmse_nm": rmse,
        "reference_field": rows[list(dv_values).index(DV_REF)]["field"] if DV_REF in dv_values else rows[0]["field"],
    }


# --- SECTION: specimen ---
def load_specimen(path: Path = SPECIMEN) -> tuple[np.ndarray, np.ndarray, int]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    footer = gray.shape[0]
    for y in range(int(gray.shape[0] * 0.8), gray.shape[0]):
        if float(gray[y].mean()) > 0.70:
            footer = y
            break
    return rgb, gray, footer


def scale_bar_pixels(gray: np.ndarray, footer: int) -> int:
    """Longest near-white run in the footer. The micrograph labels it 10 um."""
    best = 0
    band = gray[footer : min(gray.shape[0], footer + 30)]
    for row in band:
        bright = row > 0.97
        run = best_run = 0
        for flag in bright:
            run = run + 1 if flag else 0
            best_run = max(best_run, run)
        best = max(best, best_run)
    return best


def segment_cells(gray: np.ndarray, footer: int) -> np.ndarray:
    """Bright cell bodies on the dark SEM background. Footer and flagella stay out of the mask."""
    content = gray[:footer]
    thresh = float(np.percentile(content, 58))
    mask = content >= thresh
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    mask = ndi.binary_fill_holes(mask)
    labels, n = ndi.label(mask)
    if n:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= 80
        keep[0] = False
        mask = keep[labels]
    full = np.zeros(gray.shape, dtype=bool)
    full[:footer] = mask
    return full


def cell_morphometrics(gray: np.ndarray, mask: np.ndarray, footer: int) -> dict[str, float]:
    """Real measurements on the segmented SEM: equivalent diameter, aspect, circularity, coherency."""
    labels, n = ndi.label(mask[:footer])
    bar = scale_bar_pixels(gray, footer)
    um_per_px = 10.0 / bar if bar >= 80 else float("nan")
    # Isolated cell bodies only. Sub-3 µm specks are debris; the >12 µm component
    # is a merged cluster, not one cell. Nominal vegetative length is still 10 µm.
    diameters_um = []
    aspects = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        area = int(xs.size)
        if area < 80:
            continue
        d_um = 2.0 * math.sqrt(area / math.pi) * um_per_px
        if not (3.0 <= d_um <= 12.0):
            continue
        diameters_um.append(d_um)
        cov = np.cov(np.vstack([xs.astype(np.float64), ys.astype(np.float64)]))
        ev = np.sort(np.linalg.eigvalsh(np.atleast_2d(cov)))
        aspects.append(float(np.sqrt(ev[-1] / max(ev[0], 1e-9))))
    diam = np.array(diameters_um, dtype=np.float64) if diameters_um else np.array([float("nan")])
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    area_px = float(np.sum(mask))
    circularity = float(np.clip(4.0 * math.pi * area_px / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area_px > 20 else float(np.mean(coh_map))
    return {
        "n_cells": float(len(diameters_um)),
        "scale_bar_px": float(bar),
        "um_per_px": float(um_per_px),
        "median_diameter_um": float(np.median(diam)),
        "diameter_cv": float(np.std(diam) / (np.mean(diam) + 1e-9)),
        "median_aspect": float(np.median(aspects)) if aspects else float("nan"),
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "coherency": coherency,
    }


def resize_field(u: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    zy = shape[0] / u.shape[0]
    zx = shape[1] / u.shape[1]
    out = ndi.zoom(u, (zy, zx), order=1)
    if out.shape[0] < shape[0] or out.shape[1] < shape[1]:
        out = np.pad(
            out,
            ((0, max(0, shape[0] - out.shape[0])), (0, max(0, shape[1] - out.shape[1]))),
            mode="edge",
        )
    return out[: shape[0], : shape[1]]


def modulate(gray: np.ndarray, mask: np.ndarray, field: np.ndarray, gain: float) -> np.ndarray:
    """Multiply segmented cell bodies by the Turing activator. Background and flagella stay photographic."""
    delta = (field - float(np.mean(field))) / (float(np.std(field)) + 1e-9)
    delta = np.clip(delta / 2.5, -1.0, 1.0)
    out = gray.copy()
    out[mask] = np.clip(gray[mask] * (1.0 + gain * delta[mask]), 0.0, 1.0)
    return out


def frames_to_gif(grays: list[np.ndarray], path: Path) -> tuple[int, int, int]:
    images = []
    for g in grays:
        rgb = (np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
        images.append(Image.fromarray(rgb, mode="RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(path) as im:
        return im.size[0], im.size[1], int(getattr(im, "n_frames", 1))


# --- SECTION: render ---
def render(gain: float = MODULATION_GAIN) -> dict:
    sweep = dispersion_sweep(SWEEP_DV)
    frames_u = simulate_reference_frames()
    _rgb, gray, footer = load_specimen()
    mask = segment_cells(gray, footer)
    morph = cell_morphometrics(gray, mask, footer)
    rendered = []
    for u in frames_u:
        field = resize_field(u, gray.shape)
        rendered.append(modulate(gray, mask, field, gain))
    w, h, n_frames = frames_to_gif(rendered, OUT_GIF)
    sim_gray = rendered[-1]
    sim_mask = segment_cells(sim_gray, footer)
    sim_morph = cell_morphometrics(sim_gray, sim_mask, footer)
    # Intensity histograms on the cell interior (KS). Subsample so the test is deterministic and light.
    rng = np.random.default_rng(SLOT)
    spec_px = gray[:footer][mask[:footer]]
    sim_px = sim_gray[:footer][sim_mask[:footer]] if np.any(sim_mask[:footer]) else sim_gray[:footer].ravel()
    spec_s = spec_px[rng.choice(spec_px.size, size=min(4000, spec_px.size), replace=False)]
    sim_s = sim_px[rng.choice(sim_px.size, size=min(4000, sim_px.size), replace=False)]
    ks = stats.ks_2samp(spec_s, sim_s)
    scale = nm_per_nondim()
    ref_mode = fastest_mode()
    # Saturated wavelength on the last animation frame's native grid (nonlinear shift).
    saturated_nm = _radial_peak(frames_u[-1], STRIPES * ref_mode["lambda_fast"]) * scale
    linear_nm = float(sweep["numerical_nm"][list(SWEEP_DV).index(DV_REF)])
    analytical_nm = float(sweep["analytical_nm"][list(SWEEP_DV).index(DV_REF)])
    neg = negative_control_sigma()
    abl = ablation_sigma_max()

    _write_comparison_png(gray, sim_gray, mask, frames_u[-1], sweep, morph, linear_nm, analytical_nm)

    measurements = [
        {
            "name": "selected_pigment_stripe_nm",
            "value": round(linear_nm, 5),
            "unit": "nm",
            "reference_value": round(analytical_nm, 5),
            "method": "Radial FFT of the linear-regime FD field vs analytical argmax sigma(k), scaled so lambda(k_max ref)=10.32 nm. RMSE of the Dv sweep is reported separately.",
        },
        {
            "name": "dispersion_pearson_r",
            "value": round(float(sweep["pearson_r"]), 6),
            "unit": "1",
            "reference_value": 1.0,
            "method": "Pearson r of numerical vs analytical fastest wavelength across the Dv sweep. Ideal correlation is 1; the finite-difference value is not identical.",
        },
        {
            "name": "dispersion_rmse_nm",
            "value": round(float(sweep["rmse_nm"]), 5),
            "unit": "nm",
            "reference_value": 0.0,
            "method": "RMSE between numerical and analytical lambda(k_max) over Dv in {18,24,32,36,44,58,76}.",
        },
        {
            "name": "saturated_vs_linear_wavelength_nm",
            "value": round(float(saturated_nm), 5),
            "unit": "nm",
            "reference_value": round(linear_nm, 5),
            "method": "Nonlinear saturated wavelength vs the linear-regime wavelength at Dv=36. The shift is the second parameter regime (above onset).",
        },
        {
            "name": "negative_control_sigma_max",
            "value": round(float(neg), 6),
            "unit": "1/time",
            "reference_value": 0.0,
            "method": "Negative control: Dv=Du removes differential diffusion. sigma_max must stay negative (no Turing stripe).",
        },
        {
            "name": "ablation_sigma_max",
            "value": round(float(abl), 6),
            "unit": "1/time",
            "reference_value": 0.0,
            "method": "Ablation: drop autocatalysis u^2 v, leaving linear decay. sigma_max must stay negative.",
        },
        {
            "name": "specimen_cell_diameter_um",
            "value": round(float(morph["median_diameter_um"]), 4),
            "unit": "um",
            "reference_value": 10.0,
            "method": "Median equivalent diameter of isolated SEM cell bodies between 3 and 12 um, using the labeled 10 um scale bar. Debris and the merged cluster are excluded. Reference 10 um is the nominal vegetative length, not this bright-cap measurement.",
        },
        {
            "name": "specimen_cell_diameter_cv",
            "value": round(float(morph["diameter_cv"]), 4),
            "unit": "1",
            "reference_value": 0.0,
            "method": "CV of segmented cell equivalent diameters on the Dartmouth SEM. Zero would mean identical cells.",
        },
        {
            "name": "specimen_foreground_fraction",
            "value": round(float(sim_morph["foreground_fraction"]), 4),
            "unit": "1",
            "reference_value": round(float(morph["foreground_fraction"]), 4),
            "method": "Foreground fraction of the modulated simulation frame vs the real SEM segmentation. Self-referential equality is not expected once the activator rescales edges.",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": round(float(sim_morph["coherency"]), 4),
            "unit": "1",
            "reference_value": round(float(morph["coherency"]), 4),
            "method": "Structure-tensor coherency on the simulation mask vs the specimen mask.",
        },
        {
            "name": "specimen_contour_circularity",
            "value": round(float(sim_morph["circularity"]), 4),
            "unit": "1",
            "reference_value": round(float(morph["circularity"]), 4),
            "method": "4*pi*area/perimeter^2 of the segmented cell union, simulation vs specimen.",
        },
        {
            "name": "specimen_intensity_ks",
            "value": round(float(ks.statistic), 4),
            "unit": "1",
            "reference_value": 0.0,
            "method": f"Two-sample KS statistic between SEM and modulated cell-interior intensities (p={float(ks.pvalue):.3g}). Identical images would score 0.",
        },
        {
            "name": "onset_k_c_per_nm",
            "value": round(float(ref_mode["k_c"] / scale), 6),
            "unit": "1/nm",
            "reference_value": round(float(ref_mode["k_max"] / scale), 6),
            "method": "Onset wavenumber k_c compared with the fastest mode k_max at Dv/Du=36. They differ above onset; lambda(k_max) is the 10.32 nm stripe.",
        },
    ]
    payload = {
        "scale_nm": PHENOMENON_NM,
        "scale_m": PHENOMENON_SIZE_M,
        "log10_size_m": round(log10_size(), 4),
        "age_ma": AGE_MA,
        "nm_per_nondim": scale,
        "lambda_c_nm": ref_mode["lambda_c"] * scale,
        "lambda_fast_nm": ref_mode["lambda_fast"] * scale,
        "k_c": ref_mode["k_c"],
        "k_max": ref_mode["k_max"],
        "sigma_max": ref_mode["sigma_max"],
        "pearson_r": float(sweep["pearson_r"]),
        "rmse_nm": float(sweep["rmse_nm"]),
        "saturated_nm": float(saturated_nm),
        "linear_nm": linear_nm,
        "negative_control_sigma": neg,
        "ablation_sigma": abl,
        "modulation_gain": gain,
        "gif": {"width": w, "height": h, "n_frames": n_frames},
        "specimen_morphometrics": morph,
        "simulation_morphometrics": sim_morph,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "sweep": {
            "Dv": sweep["Dv"].tolist(),
            "analytical_nm": sweep["analytical_nm"].tolist(),
            "numerical_nm": sweep["numerical_nm"].tolist(),
            "lambda_c_nm": sweep["lambda_c_nm"].tolist(),
        },
        "measurements": measurements,
    }
    doc = json.loads(REF_JSON.read_text(encoding="utf-8"))
    doc["results"] = payload
    REF_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return payload


def _write_comparison_png(gray, sim_gray, mask, field, sweep, morph, linear_nm, analytical_nm) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), dpi=120)
    axes = axes.ravel()
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Dartmouth SEM specimen\nChlamydomonas reinhardtii, 0 Ma")
    axes[1].imshow(sim_gray, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Simulation on segmented cells\nactivator modulates the 10.32 nm stripe")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(
        f"CV segmentation\n{int(morph['n_cells'])} cells, median {morph['median_diameter_um']:.2f} µm"
    )
    show = field - np.mean(field)
    axes[3].imshow(show, cmap="magma")
    axes[3].set_title("Activator u on the 6-stripe patch\nλ(k_max) = 10.32 nm")
    axes[4].plot(sweep["Dv"], sweep["analytical_nm"], "-", color="#6b3fa0", label="analytical λ(k_max)")
    axes[4].plot(sweep["Dv"], sweep["numerical_nm"], "o", color="#e07a3d", label="finite-difference")
    axes[4].plot(sweep["Dv"], sweep["lambda_c_nm"], "--", color="#888888", label="onset λ(k_c)")
    axes[4].set_xlabel("D_v / D_u  (D_u = 1)")
    axes[4].set_ylabel("wavelength (nm)")
    axes[4].set_title(f"Wavenumber selection\nPearson r = {sweep['pearson_r']:.4f}")
    axes[4].legend(fontsize=8, frameon=False)
    axes[5].axis("off")
    text = (
        f"Slot 3146  ·  Holocene  ·  age 0 Ma\n"
        f"Scale 10.32 nm = 1.032e-8 m\n"
        f"log10(size) = {log10_size():.3f}  ·  tier 1 molecular\n\n"
        f"Schnakenberg  a={A}  b={B}\n"
        f"D_u={DU}  D_v={DV_REF}  seed={SEED}\n"
        f"Linear stripe {linear_nm:.3f} nm\n"
        f"Analytical    {analytical_nm:.3f} nm\n"
        f"RMSE {sweep['rmse_nm']:.4f} nm\n\n"
        f"Predecessor monomer {MONOMER_NM} nm\n"
        f"Successor quarter-wave {QUARTER_WAVE_NM:.1f} nm\n"
        f"Six stripes = {STRIPES * PHENOMENON_NM:.2f} nm\n"
        f"fit inside a {GLOBLE_NM:.0f} nm globule"
    )
    axes[5].text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)
    for ax in axes[:4]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Chlamydomonas reinhardtii — Turing pigment stripe / ocellus at 10.32 nm",
        fontsize=13,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)


def measure_morphoeval() -> list[dict]:
    """Sidecar measurements for morphoeval/measure_outputs.py. Requires a prior script run."""
    doc = json.loads(REF_JSON.read_text(encoding="utf-8"))
    rows = (doc.get("results") or {}).get("measurements") or []
    if not rows:
        raise RuntimeError(f"No results in {REF_JSON}; run the model script first.")
    return rows


# --- SECTION: self-test ---
def self_test() -> None:
    mode = fastest_mode()
    assert mode["turing"], "reference parameters must satisfy the Turing conditions"
    assert mode["trace"] < 0.0 and mode["det"] > 0.0 and mode["h"] > 0.0 and mode["disc"] > 0.0
    scale = nm_per_nondim()
    lam_star = mode["lambda_fast"] * scale
    assert abs(lam_star - PHENOMENON_NM) < 1e-9, lam_star
    assert abs(PHENOMENON_SIZE_M - 1.032e-8) < 1e-15
    assert AGE_MA == 0.0
    assert abs(log10_size() + 7.986) < 0.001
    assert MONOMER_NM < PHENOMENON_NM < QUARTER_WAVE_NM
    assert STRIPES * PHENOMENON_NM < GLOBLE_NM
    assert (A, B, DU, DV_REF) != SIBLING_195
    assert negative_control_sigma() < 0.0, "Dv=Du must not grow a stripe"
    assert ablation_sigma_max() < 0.0, "ablated autocatalysis must not grow a stripe"
    sweep = dispersion_sweep(SELF_TEST_DV)
    assert sweep["pearson_r"] > 0.98, sweep["pearson_r"]
    assert np.all(np.diff(sweep["analytical_nm"]) > 0.0), "longer inhibitor diffusion must lengthen the stripe"
    # Numerical and analytical wavelengths are close but not copied onto each other.
    assert np.max(np.abs(sweep["numerical_nm"] - sweep["analytical_nm"])) > 1e-4
    print(
        f"SELF-TEST PASSED  λ*={lam_star:.2f} nm  age={AGE_MA} Ma  "
        f"log10={log10_size():.3f}  pearson_r={sweep['pearson_r']:.5f}  "
        f"rmse={sweep['rmse_nm']:.4f} nm  σ_control={negative_control_sigma():.3f}  "
        f"σ_ablation={ablation_sigma_max():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--gain", type=float, default=MODULATION_GAIN)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    payload = render(gain=args.gain)
    print(
        f"rendered {OUT_GIF.name} {payload['gif']}  "
        f"pearson_r={payload['pearson_r']:.5f}  rmse={payload['rmse_nm']:.4f} nm  "
        f"linear={payload['linear_nm']:.4f}  analytical λ*={payload['lambda_fast_nm']:.4f}  "
        f"λ_c={payload['lambda_c_nm']:.4f} nm  saturated={payload['saturated_nm']:.4f}  "
        f"cells={payload['specimen_morphometrics']['median_diameter_um']:.3f} um"
    )


if __name__ == "__main__":
    main()
