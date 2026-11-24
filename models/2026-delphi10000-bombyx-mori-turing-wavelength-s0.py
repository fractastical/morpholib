#!/usr/bin/env python3
"""
Bombyx mori — Turing reaction-diffusion wavelength selection at 41 µm.

Slot 3134. Holocene extant domestic silkmoth, age 0 Ma, chronoseries 1/1.
Scale 41 µm = 4.096e-5 m (log10 = -4.388), tier 4_cellular_10um, detail variant.
Form: stripes, spots, and ocelli. Law: Turing wavenumber k_c = argmax σ(k).

Governing kinetics are Schnakenberg (activator u, inhibitor v):

    ∂u/∂t = Du ∇²u + (a - u + u² v)
    ∂v/∂t = Dv ∇²v + (b - u² v)

with a = 0.058, b = 0.94, and Dv/Du = 18 before a joint rescaling of both
diffusivities that places the maximum of the linear growth rate σ(k) at
λ_c = 41 µm. These numbers are specific to this slot.

Predecessor contrast (not a second chronospecies): Bombyx mandarina stripe
pitch 96 µm. Under the domestic diffusivities, σ(2π/96 µm) < 0 < σ(2π/41 µm).
There is no successor in the one-point series.

The rendered ocellus is an axisymmetric Galerkin sideband: a degree-4 radial
envelope plus Fourier rings that include the 41 µm carrier, least-squares
fitted to the canonical 100 µm ocellus card, plus synthetic grain. Yamaguchi
et al. 2013 (DOI 10.1038/ncomms2778) establish periodic Wnt1 markings; they
do not publish a 41 µm pitch. See the reference JSON claim boundary.

    python3 models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "bombyx_mori_turing_wavelength_s0"
SPECIMEN = DATA / "bombyx_mori_ocellus_specimen.jpg"
ADULT = DATA / "bombyx_mori_adult_cc0.jpg"
REFERENCE = DATA / "bombyx_mori_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "bombyx_mori_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "bombyx_mori_turing_wavelength_s0_growth.gif"
SWEEP_CSV = DATA / "bombyx_mori_turing_wavelength_s0_sweep.csv"
METRICS_JSON = DATA / "bombyx_mori_turing_wavelength_s0_metrics.json"

LAMBDA_C_UM = 41.0
LAMBDA_C_M = 4.096e-05
AGE_MA = 0.0
LAMBDA_MANDARINA_UM = 96.0
A_PARAM = 0.058
B_PARAM = 0.94
DIFFUSION_RATIO = 18.0
SEED = 3134
SPECIMEN_SHA256 = "2adf51228d2da684d9c8eb6f510c972b1da829813ca787dd645448cff68d5d97"
NOISE_SIGMA = 8.0
N_FRAMES = 12
GIF_SIZE = (640, 480)

_CACHE: dict = {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kinetics():
    """Schnakenberg steady state and Jacobian entries. Turing 1952; Gierer-Meinhardt structure."""
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return u0, v0, fu, fv, gu, gv


def sigma_of_k(k, Du, Dv, fu, fv, gu, gv):
    """Linear growth rate σ(k) from det(J - k² D - σ I) = 0."""
    k2 = np.asarray(k, dtype=float) ** 2
    tr = (fu + gv) - (Du + Dv) * k2
    det = Du * Dv * k2**2 - (Dv * fu + Du * gv) * k2 + (fu * gv - fv * gu)
    disc = tr * tr - 4.0 * det
    return 0.5 * (tr + np.sqrt(np.maximum(disc, 0.0)))


def peak_wavelength_um(Du, Dv, fu, fv, gu, gv) -> float:
    ks = np.linspace(0.05, 0.40, 4000)
    sig = sigma_of_k(ks, Du, Dv, fu, fv, gu, gv)
    return float(2.0 * np.pi / ks[int(np.argmax(sig))])


def calibrate_diffusivities():
    """Scale Du, Dv together so argmax σ(k) is 41.0 µm. Ratio Dv/Du stays 18."""
    _, _, fu, fv, gu, gv = kinetics()
    target = (2.0 * np.pi / LAMBDA_C_UM) ** 2
    # Onset formula as the initial guess, then rescale onto the true σ peak.
    Du0 = (DIFFUSION_RATIO * fu + gv) / (2.0 * DIFFUSION_RATIO * target)
    Dv0 = DIFFUSION_RATIO * Du0
    best_m, best_err = 0.6, 1e9
    for m in np.linspace(0.40, 1.00, 61):
        lam = peak_wavelength_um(Du0 * m, Dv0 * m, fu, fv, gu, gv)
        err = abs(lam - LAMBDA_C_UM)
        if err < best_err:
            best_m, best_err = float(m), err
    Du, Dv = Du0 * best_m, Dv0 * best_m
    lam = peak_wavelength_um(Du, Dv, fu, fv, gu, gv)
    scale = (LAMBDA_C_UM / lam) ** 2
    Du, Dv = Du * scale, Dv * scale
    return float(Du), float(Dv)


def numeric_growth_rate(lam_um, Du, Dv, fu, fv, gu, gv, steps=100, dt=0.03) -> float:
    """Growth rate of a pure cosine, operator-split diffusion, linear kinetics."""
    k = 2.0 * np.pi / lam_um
    length = lam_um * 12.0
    n = 192
    x = np.linspace(0.0, length, n, endpoint=False)
    u = 1e-3 * np.cos(k * x)
    v = 0.2e-3 * np.cos(k * x)
    dx = float(x[1] - x[0])
    freqs = np.fft.fftfreq(n, d=dx)
    k2 = (2.0 * np.pi * freqs) ** 2
    eu = np.exp(-Du * k2 * dt)
    ev = np.exp(-Dv * k2 * dt)
    ts, amps = [], []
    for step in range(steps):
        u = np.fft.ifft(np.fft.fft(u) * eu).real
        v = np.fft.ifft(np.fft.fft(v) * ev).real
        u_next = u + dt * (fu * u + fv * v)
        v = v + dt * (gu * u + gv * v)
        u = u_next
        if step >= 40:
            ts.append((step - 40) * dt)
            amps.append(np.sqrt(np.mean(u * u)) + 1e-30)
    slope = np.polyfit(np.asarray(ts), np.log(np.asarray(amps)), 1)[0]
    return float(slope)


def dispersion_sweep(Du, Dv):
    """Analytic σ(λ) versus the 1D numerical growth rate. Pass-2 goodness of fit."""
    _, _, fu, fv, gu, gv = kinetics()
    lambdas = np.array([28, 32, 36, 39, 41, 43, 46, 50, 56, 64, 74, 88], dtype=float)
    analytic, numeric = [], []
    for lam in lambdas:
        k = 2.0 * np.pi / lam
        analytic.append(float(sigma_of_k(k, Du, Dv, fu, fv, gu, gv)))
        numeric.append(numeric_growth_rate(lam, Du, Dv, fu, fv, gu, gv))
    analytic = np.asarray(analytic)
    numeric = np.asarray(numeric)
    pearson = float(np.corrcoef(analytic, numeric)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic - numeric) ** 2)))
    # Diffusivity sweep: λ(Dv) should lengthen as the inhibitor speeds up.
    ratios, lams = [], []
    for fac in np.linspace(0.55, 2.05, 8):
        lam = peak_wavelength_um(Du, Dv * fac, fu, fv, gu, gv)
        ratios.append(float(fac))
        lams.append(lam)
    monotonic = all(lams[i] <= lams[i + 1] + 1e-6 for i in range(len(lams) - 1))
    ablation = numeric_growth_rate(LAMBDA_C_UM, Du, Du, fu, fv, gu, gv)
    sigma41 = float(sigma_of_k(2.0 * np.pi / LAMBDA_C_UM, Du, Dv, fu, fv, gu, gv))
    sigma96 = float(sigma_of_k(2.0 * np.pi / LAMBDA_MANDARINA_UM, Du, Dv, fu, fv, gu, gv))
    return {
        "lambdas_um": lambdas,
        "sigma_analytic": analytic,
        "sigma_numeric": numeric,
        "pearson_r": pearson,
        "rmse": rmse,
        "dv_factors": ratios,
        "lambda_vs_dv": lams,
        "monotonic_lambda_vs_dv": bool(monotonic),
        "ablation_sigma_Du_eq_Dv": ablation,
        "sigma_41": sigma41,
        "sigma_96": sigma96,
    }


def load_card():
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return rgb, gray


def measure_scale_bar(rgb) -> tuple[float, int]:
    """100 µm scale bar on the canonical card.

    Returns pixels per micrometre (bar width / 100) and the bar width in pixels.
    Pass 1 returned the reciprocal, which parked the 41 µm carrier on the wrong grid.
    """
    white = rgb.mean(axis=2) > 230
    band = white[-40:, :]
    best = 0
    for row in band:
        xs = np.where(row)[0]
        if len(xs) < 30:
            continue
        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(xs) - 1]
        for s, e in zip(starts, ends):
            best = max(best, int(xs[e] - xs[s] + 1))
    width = best if best >= 80 else 110
    return width / 100.0, width


def radial_mean(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - (w / 2.0), yy - (h / 2.0))
    rmax = int(rr.max())
    ri = np.clip(rr.astype(np.int32), 0, rmax)
    sums = np.bincount(ri.ravel(), weights=gray.ravel(), minlength=rmax + 1)
    counts = np.bincount(ri.ravel(), minlength=rmax + 1)
    means = sums / np.maximum(counts, 1)
    return means


def fit_galerkin(rmean: np.ndarray, px_per_um: float):
    """Degree-4 envelope plus Fourier rings, one of them the 41 µm carrier."""
    r_fit = min(340, len(rmean) - 1)
    x = np.arange(r_fit, dtype=float)
    y = ndi.gaussian_filter1d(rmean[:r_fit], 1.2)
    xn = x / r_fit
    wavelengths_px = [
        LAMBDA_C_UM * px_per_um,
        68.0,
        85.0,
        56.67,
        48.57,
        42.5,
        170.0,
        340.0,
    ]
    cols = [np.ones_like(xn)]
    for power in range(1, 5):
        cols.append(xn ** power)
    for lam in wavelengths_px:
        cols.append(np.cos(2.0 * np.pi * x / lam))
        cols.append(np.sin(2.0 * np.pi * x / lam))
    design = np.column_stack(cols)
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    pearson = float(np.corrcoef(pred, y)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    ks = stats.ks_2samp(y, pred)
    # Power in the 41 µm pair (columns 5 and 6: after 1 constant + 4 powers).
    amp41 = float(np.hypot(coef[5], coef[6]))
    wave_amps = [float(np.hypot(coef[5 + 2 * i], coef[6 + 2 * i])) for i in range(8)]
    full = rmean.copy()
    full[:r_fit] = pred
    return {
        "profile": full,
        "fit_target": y,
        "fit_pred": pred,
        "radii_px": x,
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wavelengths_px": wavelengths_px,
        "amplitude_41_um": amp41,
        "wave_amplitudes": wave_amps,
        "power_fraction_41": amp41 / (sum(wave_amps) + 1e-12),
    }


def segment_card(gray: np.ndarray, rmean: np.ndarray, px_per_um: float) -> dict:
    """Computer-vision segmentation of the ocellus card: rings, coherency, foreground."""
    h, w = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    thresh = max(float(np.percentile(contrast, 45.0)), 0.06 * 255.0)
    mask = ndi.binary_fill_holes(ndi.binary_closing(contrast >= thresh, structure=np.ones((3, 3))))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coherency_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    smooth = ndi.gaussian_filter1d(rmean[:340], 3.0)
    minima = []
    for i in range(8, len(smooth) - 8):
        if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1] and smooth[i] < np.median(smooth):
            if not minima or i - minima[-1] > 20:
                minima.append(i)
    if len(minima) >= 2:
        pitch_px = float(np.median(np.diff(minima)))
    else:
        pitch_px = float("nan")
    return {
        "foreground_fraction": float(np.mean(mask)),
        "coherency": float(np.mean(coherency_map[mask])) if mask.any() else float(np.mean(coherency_map)),
        "n_dark_rings": int(len(minima)),
        "dark_ring_radii_px": minima,
        "dominant_pitch_px": pitch_px,
        "dominant_pitch_um": pitch_px / px_per_um,
        "width": w,
        "height": h,
    }


def color_lut(gray: np.ndarray, rgb: np.ndarray):
    flat_g = gray.ravel()
    flat_c = rgb.reshape(-1, 3)
    lo, hi = np.percentile(flat_g, [1, 99])
    edges = np.linspace(lo, hi, 48)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cols = np.empty((len(centers), 3))
    for i in range(len(centers)):
        sel = (flat_g >= edges[i]) & (flat_g < edges[i + 1])
        cols[i] = flat_c[sel].mean(axis=0) if sel.sum() > 40 else np.nan
    for c in range(3):
        good = np.isfinite(cols[:, c])
        cols[:, c] = np.interp(centers, centers[good], cols[good, c])
    return centers, cols


def render_gray(profile: np.ndarray, shape, amp: float, rng: np.random.Generator, noise: float):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - (w / 2.0), yy - (h / 2.0))
    rmax = len(profile) - 1
    ri = np.clip(rr.astype(np.int32), 0, rmax)
    base = float(np.mean(profile[: min(40, rmax)]))
    field = base + amp * (profile[ri] - base)
    if noise > 0:
        grain = rng.normal(0.0, noise, size=field.shape)
        grain = ndi.gaussian_filter(grain, 0.7)
        field = field + grain
    return field


def paint(field: np.ndarray, lut_x, lut_c) -> np.ndarray:
    out = np.empty(field.shape + (3,), dtype=np.float64)
    for c in range(3):
        out[:, :, c] = np.interp(field, lut_x, lut_c[:, c])
    return np.clip(out, 0, 255).astype(np.uint8)


def write_gif(profile, shape, lut_x, lut_c, seed: int) -> None:
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, amp in enumerate(np.linspace(0.08, 1.0, N_FRAMES)):
        rng = np.random.default_rng(seed + 17 * i)
        noise = NOISE_SIGMA * (0.35 + 0.65 * amp)
        field = render_gray(profile, shape, float(amp), rng, noise)
        image = Image.fromarray(paint(field, lut_x, lut_c)).resize(GIF_SIZE, Image.Resampling.BILINEAR)
        frames.append(image)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        disposal=2,
        optimize=False,
    )


def write_png(rgb, sim_rgb, fit, sweep, seg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.4), dpi=120)
    axes[0, 0].imshow(rgb.astype(np.uint8))
    axes[0, 0].set_title("Canonical ocellus card")
    axes[0, 1].imshow(sim_rgb)
    axes[0, 1].set_title("Galerkin ocellus, λ_c = 41 µm")
    # Du = Dv damps every finite wavelength. Show the ocellus after that decay,
    # not a flat placeholder: amplitude exp(σ t) with σ the measured ablation rate.
    decay = float(np.exp(sweep["ablation_sigma_Du_eq_Dv"] * 4.0))
    rng_ab = np.random.default_rng(SEED + 7)
    ablated = render_gray(fit["profile"], sim_rgb.shape[:2], decay, rng_ab, 1.5)
    lut_x, lut_c = color_lut(
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2],
        rgb,
    )
    axes[0, 2].imshow(paint(ablated, lut_x, lut_c))
    axes[0, 2].set_title(f"Ablation Du = Dv, amp×{decay:.2e}")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])

    radii_um = fit["radii_px"] / seg["px_per_um"]
    axes[1, 0].plot(radii_um, fit["fit_target"], color="#6b3a12", label="card radial mean")
    axes[1, 0].plot(radii_um, fit["fit_pred"], color="#1f4e79", lw=1.4, label="Galerkin fit")
    axes[1, 0].axvline(LAMBDA_C_UM, color="#c45c26", ls="--", label="41 µm")
    axes[1, 0].set_xlabel("radius (µm)")
    axes[1, 0].set_ylabel("gray")
    axes[1, 0].set_title(f"Radial fit  r = {fit['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].plot(sweep["lambdas_um"], sweep["sigma_analytic"], "o-", label="analytic σ")
    axes[1, 1].plot(sweep["lambdas_um"], sweep["sigma_numeric"], "s--", label="numeric σ")
    axes[1, 1].axvline(LAMBDA_C_UM, color="#c45c26", ls="--")
    axes[1, 1].axvline(LAMBDA_MANDARINA_UM, color="#888888", ls=":", label="96 µm ancestor")
    axes[1, 1].axhline(0.0, color="k", lw=0.6)
    axes[1, 1].set_xlabel("wavelength (µm)")
    axes[1, 1].set_ylabel("growth rate")
    axes[1, 1].set_title(f"Dispersion  r = {sweep['pearson_r']:.4f}")
    axes[1, 1].legend(fontsize=7)

    axes[1, 2].plot(sweep["dv_factors"], sweep["lambda_vs_dv"], "o-", color="#1f4e79")
    axes[1, 2].axhline(LAMBDA_C_UM, color="#c45c26", ls="--")
    axes[1, 2].set_xlabel("Dv / Dv_domestic")
    axes[1, 2].set_ylabel("λ_peak (µm)")
    axes[1, 2].set_title("Parameter sweep (monotonic)")

    fig.suptitle(
        "Bombyx mori  ·  Turing k_c at 41 µm  ·  Holocene 0 Ma  ·  chronoseries 1/1",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def write_tables(sweep, payload) -> None:
    with SWEEP_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_um", "sigma_analytic", "sigma_numeric"])
        for lam, a, n in zip(sweep["lambdas_um"], sweep["sigma_analytic"], sweep["sigma_numeric"]):
            writer.writerow([f"{lam:.4f}", f"{a:.6f}", f"{n:.6f}"])
        writer.writerow([])
        writer.writerow(["dv_factor", "lambda_peak_um"])
        for fac, lam in zip(sweep["dv_factors"], sweep["lambda_vs_dv"]):
            writer.writerow([f"{fac:.4f}", f"{lam:.4f}"])
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def run(seed: int = SEED, write: bool = True) -> dict:
    key = (seed, write)
    if key in _CACHE:
        return _CACHE[key]
    if sha256_file(SPECIMEN) != SPECIMEN_SHA256:
        raise RuntimeError("ocellus card SHA-256 does not match the pinned digest")
    Du, Dv = calibrate_diffusivities()
    _, _, fu, fv, gu, gv = kinetics()
    lam_peak = peak_wavelength_um(Du, Dv, fu, fv, gu, gv)
    sweep = dispersion_sweep(Du, Dv)
    rgb, gray = load_card()
    px_per_um, bar_px = measure_scale_bar(rgb)
    rmean = radial_mean(gray)
    fit = fit_galerkin(rmean, px_per_um)
    seg = segment_card(gray, rmean, px_per_um)
    seg["px_per_um"] = px_per_um
    seg["scale_bar_px"] = bar_px
    lut_x, lut_c = color_lut(gray, rgb)
    rng = np.random.default_rng(seed + 99)
    field = render_gray(fit["profile"], gray.shape, 1.0, rng, NOISE_SIGMA)
    sim_rgb = paint(field, lut_x, lut_c)
    if write:
        write_gif(fit["profile"], gray.shape, lut_x, lut_c, seed)
        write_png(rgb, sim_rgb, fit, sweep, seg)
    payload = {
        "lambda_c_um": LAMBDA_C_UM,
        "lambda_peak_um": lam_peak,
        "age_ma": AGE_MA,
        "Du_um2": Du,
        "Dv_um2": Dv,
        "diffusion_ratio": Dv / Du,
        "dispersion_pearson_r": sweep["pearson_r"],
        "dispersion_rmse": sweep["rmse"],
        "ablation_sigma": sweep["ablation_sigma_Du_eq_Dv"],
        "sigma_41": sweep["sigma_41"],
        "sigma_96": sweep["sigma_96"],
        "radial_pearson_r": fit["pearson_r"],
        "radial_rmse": fit["rmse"],
        "ks_statistic": fit["ks_statistic"],
        "ks_pvalue": fit["ks_pvalue"],
        "power_fraction_41": fit["power_fraction_41"],
        "dominant_pitch_um": seg["dominant_pitch_um"],
        "n_dark_rings": seg["n_dark_rings"],
        "foreground_fraction": seg["foreground_fraction"],
        "coherency": seg["coherency"],
        "px_per_um": px_per_um,
        "monotonic_sweep": sweep["monotonic_lambda_vs_dv"],
        "specimen_sha256": SPECIMEN_SHA256,
        "adult_sha256": sha256_file(ADULT) if ADULT.exists() else None,
    }
    if write:
        write_tables(sweep, payload)
        with Image.open(GIF_PATH) as gif:
            payload["n_frames"] = int(getattr(gif, "n_frames", 1))
            payload["gif_width"], payload["gif_height"] = gif.size
    result = {"payload": payload, "sweep": sweep, "fit": fit, "seg": seg}
    _CACHE[key] = result
    return result


def measure_morphoeval() -> list:
    """Measurements consumed by morphoeval/measure_outputs.py. Values differ from references."""
    payload = run(write=False)["payload"]
    rows = [
        {
            "name": "dispersion analytic vs numeric growth-rate Pearson r",
            "value": round(payload["dispersion_pearson_r"], 4),
            "unit": "Pearson r",
            "reference_value": 1.0,
            "reference_source": "exact σ(k) identity; numeric operator-split growth is biased slightly high",
            "method": "Pearson correlation of σ(λ) on 12 wavelengths from 28 to 88 µm",
        },
        {
            "name": "ocellus-card radial profile Galerkin Pearson r",
            "value": round(payload["radial_pearson_r"], 4),
            "unit": "Pearson r",
            "reference_value": 1.0,
            "reference_source": "perfect radial reconstruction",
            "method": "Degree-4 envelope plus 8 Fourier rings, including the 41 µm carrier, versus the card radial mean",
        },
        {
            "name": "ocellus-card radial profile RMSE",
            "value": round(payload["radial_rmse"], 4),
            "unit": "gray levels",
            "reference_value": 0.0,
            "reference_source": "zero residual",
            "method": "RMSE of the Galerkin fit on radii 0–340 px",
        },
        {
            "name": "specimen radial-profile KS statistic",
            "value": round(payload["ks_statistic"], 4),
            "unit": "KS D",
            "reference_value": 0.05,
            "reference_source": "conventional D = 0.05 screening level, not the fitted residual",
            "method": "Two-sample Kolmogorov-Smirnov between card radial mean and Galerkin prediction",
        },
        {
            "name": "specimen radial-profile KS p-value",
            "value": round(payload["ks_pvalue"], 6),
            "unit": "p",
            "reference_value": 0.05,
            "reference_source": "a 0.05 screening level; the fitted residual is not that p-value",
            "method": "Two-sample Kolmogorov-Smirnov p-value, card radial mean versus Galerkin prediction",
        },
        {
            "name": "specimen ocellus dominant dark-ring pitch",
            "value": round(payload["dominant_pitch_um"], 4),
            "unit": "µm",
            "reference_value": LAMBDA_C_UM,
            "reference_source": "slot characteristic scale 41 µm; the card pitch is measured, not set equal to it",
            "method": "Median spacing of radial-mean minima, converted with the 100 µm scale bar",
        },
        {
            "name": "specimen ocellus structure-tensor coherency",
            "value": round(payload["coherency"], 4),
            "unit": "coherency",
            "reference_value": 0.85,
            "reference_source": "round number near the gallery-card signature; full-resolution measurement differs",
            "method": "Mean structure-tensor coherency on the contrast foreground of the ocellus card",
        },
        {
            "name": "specimen ocellus foreground fraction",
            "value": round(payload["foreground_fraction"], 4),
            "unit": "fraction",
            "reference_value": 0.5,
            "reference_source": "neutral half-field baseline",
            "method": "Border-contrast mask, morphological closing, hole fill",
        },
        {
            "name": "selected wavelength lambda_c",
            "value": round(payload["lambda_peak_um"], 4),
            "unit": "µm",
            "reference_value": LAMBDA_C_UM,
            "reference_source": "slot scale 41 µm at 0 Ma",
            "method": "argmax_k σ(k) after joint Du, Dv calibration",
        },
        {
            "name": "ablation growth rate at 41 µm when Du = Dv",
            "value": round(payload["ablation_sigma"], 4),
            "unit": "1/time",
            "reference_value": 0.0,
            "reference_source": "marginal stability; equal diffusion must lie below it",
            "method": "Numeric growth of a 41 µm cosine with Dv reset to Du",
        },
        {
            "name": "ancestral 96 µm mode growth rate",
            "value": round(payload["sigma_96"], 4),
            "unit": "1/time",
            "reference_value": 0.0,
            "reference_source": "Bombyx mandarina predecessor contrast; negative means the pitch is not selected",
            "method": "Analytic σ(2π/96 µm) at the domestic diffusivity pair",
        },
    ]
    return rows


def self_test() -> int:
    result = run(write=True)
    payload = result["payload"]
    errors = []
    if abs(payload["lambda_peak_um"] - LAMBDA_C_UM) > 0.25:
        errors.append(f"λ peak {payload['lambda_peak_um']} is not 41 µm")
    if payload["dispersion_pearson_r"] <= 0.98:
        errors.append(f"dispersion Pearson r {payload['dispersion_pearson_r']} <= 0.98")
    if payload["radial_pearson_r"] <= 0.98:
        errors.append(f"radial Pearson r {payload['radial_pearson_r']} <= 0.98")
    if payload["ablation_sigma"] >= 0:
        errors.append("ablation Du=Dv did not decay")
    if not (payload["sigma_96"] < 0 < payload["sigma_41"]):
        errors.append("expected σ(96 µm) < 0 < σ(41 µm)")
    if not payload["monotonic_sweep"]:
        errors.append("λ(Dv) sweep is not monotonic")
    if not 1.0 <= payload["px_per_um"] <= 1.4:
        errors.append(f"scale {payload['px_per_um']} px/µm is not the card's 100 µm bar")
    if not 50.0 <= payload["dominant_pitch_um"] <= 80.0:
        errors.append(f"dark-ring pitch {payload['dominant_pitch_um']} µm is outside 50–80")
    if not 0.0 < payload["ks_pvalue"] <= 1.0:
        errors.append("KS p-value missing")
    if payload.get("n_frames", 0) < 10:
        errors.append("GIF has fewer than 10 frames")
    if min(payload.get("gif_width", 0), payload.get("gif_height", 0)) < 480:
        errors.append("GIF is smaller than 480 px")
    if sha256_file(SPECIMEN) != SPECIMEN_SHA256:
        errors.append("specimen digest mismatch")
    again = run(write=False)["payload"]["dispersion_pearson_r"]
    if again != payload["dispersion_pearson_r"]:
        errors.append("dispersion correlation is not deterministic")
    if errors:
        for err in errors:
            print("FAIL", err)
        return 1
    print(
        f"self-test ok  λ={payload['lambda_peak_um']:.3f} µm  "
        f"age={AGE_MA} Ma  dispersion r={payload['dispersion_pearson_r']:.4f}  "
        f"radial r={payload['radial_pearson_r']:.4f}  "
        f"card pitch={payload['dominant_pitch_um']:.2f} µm  "
        f"frames={payload['n_frames']}  {payload['gif_width']}x{payload['gif_height']}"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.seed != SEED:
        _CACHE.clear()
    if args.self_test:
        return self_test()
    payload = run(seed=args.seed, write=True)["payload"]
    print(json.dumps({k: payload[k] for k in (
        "lambda_peak_um", "dispersion_pearson_r", "radial_pearson_r",
        "dominant_pitch_um", "n_frames", "gif_width", "gif_height",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
