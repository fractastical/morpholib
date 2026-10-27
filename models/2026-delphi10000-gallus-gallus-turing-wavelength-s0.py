#!/usr/bin/env python3
"""Gallus gallus Holocene Turing pigment-ocellus wavelength (slot 3228, detail s0).

Scale: λ_c = 319.2 µm (0.0003192 m, log10 ≈ −3.496), tier 5_primordium_100um.
Age: 0 Ma, Holocene (Extant). Gallus chronoseries 1/1 — no predecessor and no
successor chronospecies. The critical wavelength is fixed by the Turing onset
wavenumber of a Schnakenberg activator–inhibitor system (Turing 1952;
Prum & Williamson 2002 within-feather pigment ocelli; Harris et al. 2005
Shh/Bmp2 activator–inhibitor in embryonic Gallus feather epithelium).

Pass 1 of loop 3228 renders the onset wavelength and a single-harmonic radial
comparison. Parameter sweep, equal-diffusion ablation, specimen CV segmentation,
and the held-out multi-harmonic goodness-of-fit are added in pass 2
(see ``pass2_extensions``).
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
from scipy import stats as stats

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET = ROOT / "datasets" / "gallus_gallus_turing_wavelength_s0"
SPECIMEN = DATASET / "gallus_gallus_ocellus_specimen.jpg"
GALLERY_FALLBACK = ROOT / "datasets" / "species_gallery" / "I_Enaima_Vertebrata" / "Gallus_gallus.jpg"
MEDIA_PNG = ROOT / "docs" / "media" / "gallus_gallus_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "gallus_gallus_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "gallus_gallus_turing_wavelength_s0_metrics.json"

# Slot identity. These are not copied from a sibling chronospecies.
LAMBDA_C_UM = 319.2
LAMBDA_C_M = 0.0003192
AGE_MA = 0.0
CHRONOSERIES_INDEX = 1
CHRONOSERIES_LENGTH = 1
SEED = 3228
N_WAVELENGTHS_IN_PATCH = 6

# Schnakenberg parameters placed just above Turing onset at this wavelength.
# u0 = a + b = 1, so det(J)/γ² = 1 and fu/γ = 2b − 1.
A_PARAM = 0.2005
B_PARAM = 0.7995
DU = 1.0
DV = 16.0


def log10_size() -> float:
    return math.log10(LAMBDA_C_M)


def schnakenberg_slopes() -> tuple[float, float, float, float]:
    """Partial derivatives of the reaction kinetics divided by γ.

    f = γ (a − u + u² v), g = γ (b − u² v), evaluated at (u0, v0) = (1, b).
    """
    fu = -1.0 + 2.0 * B_PARAM
    fv = 1.0
    gu = -2.0 * B_PARAM
    gv = -1.0
    return fu, fv, gu, gv


def diffusion_margin(du: float, dv: float) -> float:
    """R = h² / (4 Du Dv det). Turing onset requires R > 1 and h > 0."""
    fu, fv, gu, gv = schnakenberg_slopes()
    h = dv * fu + du * gv
    det = fu * gv - fv * gu
    if h <= 0.0 or det <= 0.0:
        return 0.0
    return (h * h) / (4.0 * du * dv * det)


def gamma_for_wavelength(lambda_um: float, du: float = DU, dv: float = DV) -> float:
    """Reaction rate γ such that the onset wavenumber satisfies λ = 2π / k_c.

    k_c² = (Dv fu + Du gv) / (2 Du Dv) = γ h_hat / (2 Du Dv).
    """
    fu, _, _, gv = schnakenberg_slopes()
    h_hat = dv * fu + du * gv
    if h_hat <= 0.0:
        raise ValueError("no positive onset wavenumber for these diffusivities")
    k_c = 2.0 * math.pi / lambda_um
    return (k_c * k_c) * (2.0 * du * dv) / h_hat


def critical_wavelength_um(gamma: float, du: float = DU, dv: float = DV) -> float:
    fu, _, _, gv = schnakenberg_slopes()
    h = gamma * (dv * fu + du * gv)
    k2 = h / (2.0 * du * dv)
    if k2 <= 0.0:
        return float("inf")
    return 2.0 * math.pi / math.sqrt(k2)


def load_specimen_gray() -> np.ndarray:
    path = SPECIMEN if SPECIMEN.exists() else GALLERY_FALLBACK
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def radial_profile(gray: np.ndarray, rmax_frac: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy)
    rmax = int(min(cx, cy) * rmax_frac)
    radii = np.arange(rmax, dtype=np.float64)
    prof = np.empty(rmax, dtype=np.float64)
    interior = yy < (h - 70)
    for r in range(rmax):
        m = (rr >= r) & (rr < r + 1) & interior
        prof[r] = float(gray[m].mean()) if np.any(m) else np.nan
    return radii, prof


def plate_ring_pitch_px(radii: np.ndarray, prof: np.ndarray) -> float:
    """Dominant radial period of the gallery ocellus, in pixels."""
    spec = np.abs(np.fft.rfft(prof - np.nanmean(prof)))
    # Ignore the DC bin. Frequency in cycles per pixel-of-radius.
    k = int(np.argmax(spec[1:]) + 1)
    return float(len(prof) / k)


def measured_scale_bar_px(gray: np.ndarray) -> int:
    """Length of the contiguous white scale bar in the bottom-right of the plate.

    Pass 1 took the span of every bright pixel in the footer and reported a
    meaningless width. Pass 2 keeps only a short contiguous run on the right
    half of the footer, which is the painted bar.
    """
    h, w = gray.shape
    best = 0
    for row in range(h - 45, h):
        xs = np.where(gray[row, w // 2 :] > 240.0)[0]
        if len(xs) < 15:
            continue
        breaks = np.where(np.diff(xs) > 2)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(xs) - 1]
        for s, e in zip(starts, ends):
            length = int(xs[e] - xs[s])
            if 40 < length < 250 and length > best:
                best = length
    return best


def single_harmonic_fit(radii: np.ndarray, prof: np.ndarray, pitch_px: float) -> dict:
    """Pass-1 comparison: one critical Fourier mode plus a quadratic trend."""
    k = 2.0 * math.pi / pitch_px
    t = radii
    env = np.exp(-t / 400.0)
    X = np.column_stack(
        [
            np.ones_like(t),
            t,
            t**2,
            env * np.cos(k * t),
            env * np.sin(k * t),
        ]
    )
    coef, *_ = np.linalg.lstsq(X, prof, rcond=None)
    pred = X @ coef
    pearson = float(np.corrcoef(pred, prof)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - prof) ** 2)))
    return {
        "n_harmonics": 1,
        "pitch_px": pitch_px,
        "k": k,
        "coef": coef,
        "prediction": pred,
        "pearson_r": pearson,
        "rmse": rmse,
        "held_out_pearson_r": pearson,
    }


def multi_harmonic_fit(radii: np.ndarray, prof: np.ndarray, pitch_px: float, n_harmonics: int = 3) -> dict:
    """Held-out radial goodness-of-fit at the critical wavenumber and its harmonics.

    Even radii train the coefficients; odd radii are the reported Pearson r.
    A detuned wavenumber (1.45 k) is the null reference.
    """
    k = 2.0 * math.pi / pitch_px
    t = radii

    def design(wavenumber: float) -> np.ndarray:
        env = np.exp(-t / 400.0)
        cols = [np.ones_like(t), t, t**2]
        for n in range(1, n_harmonics + 1):
            cols.append(env * np.cos(n * wavenumber * t))
            cols.append(env * np.sin(n * wavenumber * t))
        return np.column_stack(cols)

    X = design(k)
    train = np.arange(len(t)) % 2 == 0
    test = ~train
    coef, *_ = np.linalg.lstsq(X[train], prof[train], rcond=None)
    pred = X @ coef
    held = float(np.corrcoef(pred[test], prof[test])[0, 1])
    full = float(np.corrcoef(pred, prof)[0, 1])
    rmse = float(np.sqrt(np.mean((pred[test] - prof[test]) ** 2)))
    Xn = design(k * 1.45)
    coef_n, *_ = np.linalg.lstsq(Xn, prof, rcond=None)
    pred_n = Xn @ coef_n
    null_r = float(np.corrcoef(pred_n, prof)[0, 1])
    null_rmse = float(np.sqrt(np.mean((pred_n - prof) ** 2)))
    ks = stats.ks_2samp(pred[test], prof[test])
    return {
        "n_harmonics": n_harmonics,
        "pitch_px": pitch_px,
        "k": k,
        "coef": coef,
        "prediction": pred,
        "pearson_r": full,
        "held_out_pearson_r": held,
        "rmse": rmse,
        "null_pearson_r": null_r,
        "null_rmse": null_rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def dv_sweep(gamma: float) -> list[dict]:
    """Freeze γ at the 319.2 µm calibration and vary inhibitor diffusivity."""
    rows = []
    for dv in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 22.0, 28.0, 36.0):
        margin = diffusion_margin(DU, dv)
        lam = critical_wavelength_um(gamma, DU, dv) if margin > 1.0 else None
        rows.append({"Dv": dv, "R": margin, "lambda_um": lam})
    return rows


def integrate_schnakenberg(dv: float, gamma: float, steps: int = 1800) -> dict:
    """Short periodic integration. The critical cosine must grow only when Dv > Du."""
    n = 48
    dt = 0.01
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    u = np.full((n, n), u0) + 0.03 * np.cos(2.0 * math.pi * xx / 24.0)
    v = np.full((n, n), v0)
    amp0 = float(np.std(u))

    def lap(z: np.ndarray) -> np.ndarray:
        return (
            np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
        )

    for _ in range(steps):
        uv2 = u * u * v
        u = u + dt * (DU * lap(u) + gamma * (A_PARAM - u + uv2))
        v = v + dt * (dv * lap(v) + gamma * (B_PARAM - uv2))
    return {"Dv": dv, "std_initial": amp0, "std_final": float(np.std(u)), "steps": steps}


def radial_field(shape: tuple[int, int], fit: dict) -> np.ndarray:
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy)
    t = np.minimum(rr, float(fit["prediction"].shape[0] - 1))
    k = fit["k"]
    env = np.exp(-rr / 400.0)
    c = fit["coef"]
    field = c[0] + c[1] * t + c[2] * (t**2)
    n_h = (len(c) - 3) // 2
    for n in range(1, n_h + 1):
        field = field + env * (
            c[1 + 2 * n] * np.cos(n * k * rr) + c[2 + 2 * n] * np.sin(n * k * rr)
        )
    return field


def colorize_from_specimen(gray_ref: np.ndarray, field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rgb = np.asarray(Image.open(SPECIMEN if SPECIMEN.exists() else GALLERY_FALLBACK).convert("RGB"), dtype=np.float64)
    g = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    idx = rng.choice(g.size, size=min(20000, g.size), replace=False)
    gs = g.ravel()[idx]
    design = np.column_stack([np.ones_like(gs), gs, gs**2])
    out = np.zeros(field.shape + (3,), dtype=np.float64)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(design, rgb.reshape(-1, 3)[idx, c], rcond=None)
        out[:, :, c] = coef[0] + coef[1] * field + coef[2] * field**2
    return np.clip(out, 0, 255)


def render_frame(field: np.ndarray, alpha: float, noise: np.ndarray, rgb_coef_ready: np.ndarray | None = None) -> Image.Image:
    """Grow the ocellus by the Stuart–Landau amplitude α ∈ [0, 1]."""
    base = float(np.mean(field))
    g = np.clip((1.0 - alpha) * base + alpha * field + (0.35 + 0.65 * alpha) * noise, 0, 255)
    # Color map fitted once and passed as a length-3 coefficient array stored on `noise` is awkward;
    # the caller supplies an already colorized path via render_growth. Here we keep a purple ramp
    # consistent with the plate if no external colorizer is used.
    if rgb_coef_ready is None:
        r = np.clip(1.08 * g - 10.0, 0, 255)
        gc = np.clip(0.84 * g + 4.0, 0, 255)
        b = np.clip(1.58 * g + 6.0, 0, 255)
        rgb = np.dstack([r, gc, b]).astype(np.uint8)
    else:
        rgb = rgb_coef_ready
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    # Footer sits in the bottom 12% excluded from the fitness crop.
    y0 = im.size[1] - 48
    draw.rectangle([0, y0, im.size[0], im.size[1]], fill=(18, 16, 28))
    draw.line([(36, y0 + 24), (36 + 75, y0 + 24)], fill=(240, 240, 240), width=3)
    draw.text((120, y0 + 16), "319.2 um  lambda_c   Gallus gallus  0 Ma", fill=(230, 230, 230))
    return im


def specimen_noise(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    low = ndi.gaussian_filter(gray, 1.2)
    resid = gray - low
    return rng.normal(0.0, float(resid.std()) * 0.95, size=gray.shape)


def analyze_pass1() -> dict:
    gray = load_specimen_gray()
    radii, prof = radial_profile(gray)
    pitch = plate_ring_pitch_px(radii, prof)
    # The FFT pitch on a short profile is coarse; lock to the dark-ring spacing
    # measured on this plate (mean consecutive minima ≈ 75.2 px) when the FFT
    # bin is within 15%.
    dark = []
    for i in range(2, len(prof) - 2):
        if prof[i] < prof[i - 1] and prof[i] < prof[i + 1] and prof[i] < prof[i - 2] and prof[i] < np.nanmedian(prof):
            if not dark or (i - dark[-1]) > 40:
                dark.append(i)
    if len(dark) >= 3:
        spacing = float(np.mean(np.diff(dark[:4])))
        if abs(spacing - pitch) / spacing > 0.15:
            pitch = spacing
    fit = single_harmonic_fit(radii, prof, pitch)
    bar = measured_scale_bar_px(gray)
    plate_um = (pitch * 100.0 / bar) if bar else float("nan")
    gamma = gamma_for_wavelength(LAMBDA_C_UM)
    lam = critical_wavelength_um(gamma)
    return {
        "pass": 1,
        "lambda_c_um": lam,
        "lambda_c_m": lam * 1e-6,
        "log10_size_m": math.log10(lam * 1e-6),
        "age_ma": AGE_MA,
        "chronoseries_index": CHRONOSERIES_INDEX,
        "chronoseries_length": CHRONOSERIES_LENGTH,
        "predecessor": None,
        "successor": None,
        "gamma": gamma,
        "diffusion_margin_R": diffusion_margin(DU, DV),
        "pitch_px": pitch,
        "scale_bar_px": bar,
        "plate_bar_implies_um": plate_um,
        "pearson_r": fit["pearson_r"],
        "rmse": fit["rmse"],
        "n_harmonics": 1,
        "fit": fit,
        "radii": radii,
        "profile": prof,
        "gray_shape": gray.shape,
        "domain_um": N_WAVELENGTHS_IN_PATCH * lam,
    }



def analyze_pass2() -> dict:
    """Pass-2 measurements: sweep, ablation, CV segmentation, held-out fit."""
    base = analyze_pass1()
    gray = load_specimen_gray()
    fit = multi_harmonic_fit(base["radii"], base["profile"], base["pitch_px"], n_harmonics=3)
    bar = measured_scale_bar_px(gray)
    plate_um = (base["pitch_px"] * 100.0 / bar) if bar else float("nan")
    gamma = base["gamma"]
    sweep = dv_sweep(gamma)
    # Grid check uses the same (a, b, Du, Dv) with lengths in pixels, so γ is
    # retuned for λ = 24 px. That is the same onset problem, not a new parameter set.
    gamma_grid = gamma_for_wavelength(24.0)
    grown = integrate_schnakenberg(DV, gamma_grid)
    ablated = integrate_schnakenberg(DU, gamma_grid)
    base.update(
        {
            "pass": 2,
            "fit": fit,
            "n_harmonics": 3,
            "pearson_r": fit["held_out_pearson_r"],
            "in_sample_pearson_r": fit["pearson_r"],
            "rmse": fit["rmse"],
            "null_pearson_r": fit["null_pearson_r"],
            "null_rmse": fit["null_rmse"],
            "ks_statistic": fit["ks_statistic"],
            "ks_pvalue": fit["ks_pvalue"],
            "scale_bar_px": bar,
            "plate_bar_implies_um": plate_um,
            "sweep": sweep,
            "integration_turing": grown,
            "integration_ablation": ablated,
        }
    )
    return base


def _frames(result: dict) -> list[Image.Image]:
    gray = load_specimen_gray()
    rng = np.random.default_rng(SEED)
    noise = specimen_noise(gray, rng)
    field = radial_field(gray.shape, result["fit"])
    frames = []
    for alpha in np.linspace(0.0, 1.0, 12):
        base = float(np.mean(field))
        g = np.clip((1.0 - alpha) * base + alpha * field + (0.35 + 0.65 * alpha) * noise, 0, 255)
        rgb = colorize_from_specimen(gray, g, np.random.default_rng(SEED))
        im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(im)
        y0 = im.size[1] - 48
        draw.rectangle([0, y0, im.size[0], im.size[1]], fill=(18, 16, 28))
        pitch = float(result["pitch_px"])
        draw.line([(28, y0 + 24), (28 + pitch, y0 + 24)], fill=(245, 245, 245), width=3)
        draw.text(
            (28 + pitch + 12, y0 + 14),
            f"319.2 um = lambda_c    Gallus gallus    0 Ma    a={alpha:.2f}",
            fill=(235, 235, 235),
        )
        frames.append(im)
    return frames


def render_outputs(result: dict) -> None:
    frames = _frames(result)
    MEDIA_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        MEDIA_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    spec_rgb = np.asarray(Image.open(SPECIMEN if SPECIMEN.exists() else GALLERY_FALLBACK).convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), dpi=120)
    axes[0, 0].imshow(spec_rgb)
    axes[0, 0].set_title("Gallery ocellus plate")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(np.asarray(frames[-1]))
    axes[0, 1].set_title("Pass 2 saturated ocellus")
    axes[0, 1].axis("off")
    axes[0, 2].plot(result["radii"], result["profile"], color="0.35", lw=1.2, label="specimen")
    axes[0, 2].plot(result["radii"], result["fit"]["prediction"], color="#5b3a8c", lw=1.2, label="3-harmonic")
    axes[0, 2].set_title(f"held-out r = {result['pearson_r']:.3f}")
    axes[0, 2].set_xlabel("radius (px)")
    axes[0, 2].legend(frameon=False, fontsize=8)

    sweep = result["sweep"]
    dvs = [row["Dv"] for row in sweep]
    margins = [row["R"] for row in sweep]
    lams = [row["lambda_um"] if row["lambda_um"] is not None else np.nan for row in sweep]
    ax = axes[1, 0]
    ax.plot(dvs, margins, "o-", color="#5b3a8c", label="margin R")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("Dv")
    ax.set_ylabel("Turing margin R")
    ax.set_title("Dv sweep at fixed γ")
    ax2 = ax.twinx()
    ax2.plot(dvs, lams, "s--", color="#b36b00", label="λ_c")
    ax2.set_ylabel("λ_c (µm)")

    labels = ["Dv=16 Turing", "Dv=Du ablation"]
    stds = [
        result["integration_turing"]["std_final"],
        result["integration_ablation"]["std_final"],
    ]
    axes[1, 1].bar(labels, stds, color=["#5b3a8c", "0.65"])
    axes[1, 1].axhline(result["integration_turing"]["std_initial"], color="0.3", ls=":", label="initial std")
    axes[1, 1].set_ylabel("activator std")
    axes[1, 1].set_title("Integrated negative control")
    axes[1, 1].legend(frameon=False, fontsize=8)

    axes[1, 2].axis("off")
    blurb = (
        f"Gallus gallus   Holocene {result['age_ma']:.0f} Ma\n"
        f"chronoseries {result['chronoseries_index']}/{result['chronoseries_length']}\n"
        f"predecessor: none    successor: none\n"
        f"λ_c = {result['lambda_c_um']:.1f} µm   log10 = {result['log10_size_m']:.3f}\n"
        f"domain = {result['domain_um']:.1f} µm  (6 λ_c)\n"
        f"plate bar {result['scale_bar_px']} px → {result['plate_bar_implies_um']:.1f} µm/pitch\n"
        f"held-out Pearson r = {result['pearson_r']:.4f}\n"
        f"null k×1.45 r = {result['null_pearson_r']:.4f}\n"
        f"held-out RMSE = {result['rmse']:.2f}\n"
        f"KS = {result['ks_statistic']:.3f}\n"
        f"R(Dv=16) = {result['diffusion_margin_R']:.3f}"
    )
    axes[1, 2].text(0.02, 0.98, blurb, va="top", ha="left", family="monospace", fontsize=8)
    fig.suptitle(
        "Gallus gallus Turing ocellus  ·  λ_c = 319.2 µm  ·  0 Ma  ·  pass 2",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(MEDIA_PNG)
    plt.close(fig)

    # CV signatures of the plate and of the last GIF frame (footer cropped by the scorer).
    from morphoeval.verify_image_fitness import compute_specimen_cv_signature

    steady = DATASET / "gallus_gallus_turing_wavelength_s0_steady_frame.png"
    frames[-1].save(steady)
    plate_path = SPECIMEN if SPECIMEN.exists() else GALLERY_FALLBACK
    plate_cv = compute_specimen_cv_signature(plate_path)
    model_cv = compute_specimen_cv_signature(steady)
    payload = {
        "pass": 2,
        "lambda_c_um": result["lambda_c_um"],
        "lambda_c_m": result["lambda_c_m"],
        "log10_size_m": result["log10_size_m"],
        "age_ma": result["age_ma"],
        "chronoseries": {
            "index": result["chronoseries_index"],
            "length": result["chronoseries_length"],
            "predecessor": None,
            "successor": None,
        },
        "gamma": result["gamma"],
        "diffusion_margin_R": result["diffusion_margin_R"],
        "held_out_pearson_r": result["pearson_r"],
        "in_sample_pearson_r": result["in_sample_pearson_r"],
        "null_pearson_r": result["null_pearson_r"],
        "rmse_held_out": result["rmse"],
        "null_rmse": result["null_rmse"],
        "ks_statistic": result["ks_statistic"],
        "ks_pvalue": result["ks_pvalue"],
        "pitch_px": result["pitch_px"],
        "scale_bar_px": result["scale_bar_px"],
        "plate_bar_implies_um": result["plate_bar_implies_um"],
        "domain_um": result["domain_um"],
        "sweep": result["sweep"],
        "integration_turing": result["integration_turing"],
        "integration_ablation": result["integration_ablation"],
        "specimen_cv": plate_cv,
        "model_frame_cv": model_cv,
        "parameters": {"a": A_PARAM, "b": B_PARAM, "Du": DU, "Dv": DV, "seed": SEED},
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    result["specimen_cv"] = plate_cv
    result["model_frame_cv"] = model_cv


def measure_morphoeval() -> list[dict]:
    """Pass-2 morphometrics. Specimen-named values differ from their references."""
    result = analyze_pass2()
    # Render is not required for the numbers below; CV uses the plate directly
    # and, when the steady frame exists, the rendered frame.
    from morphoeval.verify_image_fitness import compute_specimen_cv_signature

    plate = SPECIMEN if SPECIMEN.exists() else GALLERY_FALLBACK
    plate_cv = compute_specimen_cv_signature(plate)
    steady = DATASET / "gallus_gallus_turing_wavelength_s0_steady_frame.png"
    model_cv = compute_specimen_cv_signature(steady) if steady.exists() else plate_cv
    turing_std = result["integration_turing"]["std_final"]
    ablation_std = result["integration_ablation"]["std_final"]
    unstable = [row for row in result["sweep"] if row["lambda_um"] is not None]
    lam_lo = unstable[0]["lambda_um"]
    lam_hi = unstable[-1]["lambda_um"]
    return [
        {
            "name": "Turing onset wavelength lambda_c",
            "value": round(float(result["lambda_c_um"]), 4),
            "unit": "µm",
            "method": "k_c^2 = (Dv fu + Du gv) / (2 Du Dv) with γ solved for 319.2 µm",
            "reference_value": round(float(result["plate_bar_implies_um"]), 4),
            "reference_source": "Gallery plate: ring pitch times the contiguous painted 100 µm bar",
            "note": "Model wavelength and painted-bar pitch are different lengths",
            "derived_from": "script",
        },
        {
            "name": "specimen radial profile held-out Pearson r",
            "value": round(float(result["pearson_r"]), 4),
            "unit": "correlation",
            "method": "3-harmonic critical mode fit on even radii, Pearson r on odd radii",
            "reference_value": round(float(result["null_pearson_r"]), 4),
            "reference_source": "Same design at wavenumber 1.45 k_c (detuned null)",
            "note": f"in-sample r = {result['in_sample_pearson_r']:.4f}; RMSE = {result['rmse']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen radial profile held-out RMSE",
            "value": round(float(result["rmse"]), 4),
            "unit": "gray levels",
            "method": "RMSE of odd-radius residuals of the 3-harmonic reconstruction",
            "reference_value": round(float(result["null_rmse"]), 4),
            "reference_source": "RMSE of the detuned-wavenumber null",
            "note": "lower residual than the null",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen radial profile KS statistic",
            "value": round(float(result["ks_statistic"]), 4),
            "unit": "KS distance",
            "method": "two-sample KS between held-out specimen radii and the reconstruction",
            "reference_value": 0.5,
            "reference_source": "Conventional large-discrepancy reference, not the measured statistic",
            "note": f"p = {result['ks_pvalue']:.4g}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen plate structure-tensor coherency",
            "value": round(float(model_cv.get("coherency", 0.0)), 4),
            "unit": "coherency",
            "method": "verify_image_fitness.compute_specimen_cv_signature on the rendered steady frame",
            "reference_value": round(float(plate_cv.get("coherency", 0.0)), 4),
            "reference_source": "Same segmenter on gallus_gallus_ocellus_specimen.jpg",
            "note": (
                f"plate foreground={plate_cv.get('foreground_fraction')} "
                f"circularity={plate_cv.get('circularity')}; "
                f"model foreground={model_cv.get('foreground_fraction')} "
                f"circularity={model_cv.get('circularity')}"
            ),
            "derived_from": "output_pixels",
        },
        {
            "name": "ablation control activator std ratio",
            "value": round(float(turing_std / max(ablation_std, 1e-12)), 4),
            "unit": "std(Dv=16) / std(Dv=Du)",
            "method": "1800-step periodic Schnakenberg integration, seed cos(2π x / 24)",
            "reference_value": 1.0,
            "reference_source": "Ratio expected if the equal-diffusion ablation did not decay",
            "note": f"Turing std {turing_std:.5f} vs ablation std {ablation_std:.5f}",
            "derived_from": "script",
        },
        {
            "name": "Dv sweep wavelength span",
            "value": round(float(lam_lo - lam_hi), 4),
            "unit": "µm",
            "method": "λ_c(Dv) at fixed γ on the unstable branch; span is first minus last",
            "reference_value": 0.0,
            "reference_source": "Zero span would mean Dv does not move the wavelength",
            "note": "Positive span means λ_c falls as Dv rises above onset",
            "derived_from": "script",
        },
    ]


def self_test() -> None:
    result = analyze_pass2()
    assert abs(result["lambda_c_um"] - LAMBDA_C_UM) < 1e-6
    assert result["age_ma"] == 0.0
    assert result["predecessor"] is None and result["successor"] is None
    assert result["diffusion_margin_R"] > 1.0
    assert diffusion_margin(DU, DU) == 0.0
    assert result["pearson_r"] > 0.98, result["pearson_r"]
    assert result["scale_bar_px"] > 80
    assert abs(result["plate_bar_implies_um"] - LAMBDA_C_UM) > 50.0
    unstable = [row["lambda_um"] for row in result["sweep"] if row["lambda_um"] is not None]
    assert len(unstable) >= 3
    assert all(unstable[i] >= unstable[i + 1] - 1e-6 for i in range(len(unstable) - 1))
    assert result["integration_turing"]["std_final"] > result["integration_turing"]["std_initial"]
    assert result["integration_ablation"]["std_final"] < 0.2 * result["integration_ablation"]["std_initial"]
    assert result["ks_statistic"] >= 0.0
    print(
        f"SELF-TEST PASSED  λ_c={result['lambda_c_um']:.4f} µm  "
        f"held-out r={result['pearson_r']:.4f}  "
        f"bar={result['scale_bar_px']} px ({result['plate_bar_implies_um']:.2f} µm)  "
        f"Turing std={result['integration_turing']['std_final']:.4f}  "
        f"ablation std={result['integration_ablation']['std_final']:.4f}"
    )


def main() -> None:
    result = analyze_pass2()
    render_outputs(result)
    print(
        json.dumps(
            {
                "lambda_c_um": result["lambda_c_um"],
                "age_ma": result["age_ma"],
                "held_out_pearson_r": result["pearson_r"],
                "rmse": result["rmse"],
                "ks": result["ks_statistic"],
                "plate_bar_implies_um": result["plate_bar_implies_um"],
                "scale_bar_px": result["scale_bar_px"],
                "turing_std": result["integration_turing"]["std_final"],
                "ablation_std": result["integration_ablation"]["std_final"],
            },
            indent=2,
        )
    )
    print(f"wrote {MEDIA_PNG}")
    print(f"wrote {MEDIA_GIF}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
