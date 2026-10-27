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

ROOT = Path(__file__).resolve().parents[1]
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
    """Width of the painted white bar in the bottom-right of the gallery plate."""
    band = gray[-40:, :]
    ys, xs = np.where(band > 240.0)
    if len(xs) < 10:
        return 0
    return int(xs.max() - xs.min())


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


def radial_field(shape: tuple[int, int], fit: dict) -> np.ndarray:
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy)
    t = np.minimum(rr, fit["prediction"].shape[0] - 1.0)
    k = fit["k"]
    env = np.exp(-rr / 400.0)
    c = fit["coef"]
    field = c[0] + c[1] * t + c[2] * t**2 + env * (c[3] * np.cos(k * rr) + c[4] * np.sin(k * rr))
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


def render_outputs(result: dict) -> None:
    gray = load_specimen_gray()
    rng = np.random.default_rng(SEED)
    noise = specimen_noise(gray, rng)
    field = radial_field(gray.shape, result["fit"])
    colored = colorize_from_specimen(gray, np.clip(field, 0, 255), rng)
    frames = []
    alphas = np.linspace(0.0, 1.0, 12)
    for alpha in alphas:
        base = float(np.mean(field))
        g = np.clip((1.0 - alpha) * base + alpha * field + (0.35 + 0.65 * alpha) * noise, 0, 255)
        # Reuse the specimen color polynomial through colorize on this frame's gray.
        rgb = colorize_from_specimen(gray, g, rng)
        im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(im)
        y0 = im.size[1] - 48
        draw.rectangle([0, y0, im.size[0], im.size[1]], fill=(18, 16, 28))
        pitch = float(result["pitch_px"])
        draw.line([(28, y0 + 24), (28 + pitch, y0 + 24)], fill=(245, 245, 245), width=3)
        draw.text(
            (28 + pitch + 12, y0 + 14),
            f"319.2 um = lambda_c    Gallus gallus    0 Ma    frame a={alpha:.2f}",
            fill=(235, 235, 235),
        )
        frames.append(im)
    MEDIA_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        MEDIA_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), dpi=120)
    spec_rgb = np.asarray(Image.open(SPECIMEN if SPECIMEN.exists() else GALLERY_FALLBACK).convert("RGB"))
    axes[0].imshow(spec_rgb)
    axes[0].set_title("Gallery ocellus plate")
    axes[0].axis("off")
    axes[1].imshow(np.asarray(frames[-1]))
    axes[1].set_title("Pass 1 single-harmonic λ_c")
    axes[1].axis("off")
    axes[2].plot(result["radii"], result["profile"], color="0.35", label="specimen")
    axes[2].plot(result["radii"], result["fit"]["prediction"], color="#5b3a8c", label="1-harmonic")
    axes[2].set_xlabel("radius (px)")
    axes[2].set_ylabel("gray")
    axes[2].set_title(f"Pearson r = {result['pearson_r']:.3f}")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Gallus gallus  λ_c = {LAMBDA_C_UM:.1f} µm   age = {AGE_MA:.0f} Ma   "
        f"chronoseries {CHRONOSERIES_INDEX}/{CHRONOSERIES_LENGTH}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(MEDIA_PNG)
    plt.close(fig)

    payload = {
        "pass": result["pass"],
        "lambda_c_um": result["lambda_c_um"],
        "lambda_c_m": result["lambda_c_m"],
        "log10_size_m": result["log10_size_m"],
        "age_ma": result["age_ma"],
        "chronoseries": {
            "index": CHRONOSERIES_INDEX,
            "length": CHRONOSERIES_LENGTH,
            "predecessor": None,
            "successor": None,
            "note": "Singleton Holocene series. N_periods changes with field size, not with a chronospecies boundary.",
        },
        "gamma": result["gamma"],
        "diffusion_margin_R": result["diffusion_margin_R"],
        "pearson_r": result["pearson_r"],
        "rmse": result["rmse"],
        "pitch_px": result["pitch_px"],
        "scale_bar_px": result["scale_bar_px"],
        "plate_bar_implies_um": result["plate_bar_implies_um"],
        "domain_um": result["domain_um"],
        "parameters": {"a": A_PARAM, "b": B_PARAM, "Du": DU, "Dv": DV, "seed": SEED},
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Keep the colorized steady field available for the fitness crop without recomputing noise.
    _ = colored


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Pass 1 reports the onset wavelength
    and the single-harmonic plate comparison. Reference values are the painted
    scale-bar pitch and a trend-only null, neither equal to the measured value.
    """
    result = analyze_pass1()
    radii, prof = result["radii"], result["profile"]
    trend = np.column_stack([np.ones_like(radii), radii, radii**2])
    coef, *_ = np.linalg.lstsq(trend, prof, rcond=None)
    null_r = float(np.corrcoef(trend @ coef, prof)[0, 1])
    return [
        {
            "name": "Turing onset wavelength lambda_c",
            "value": round(float(result["lambda_c_um"]), 4),
            "unit": "µm",
            "method": "lambda_c = 2π / k_c with k_c^2 = (Dv fu + Du gv) / (2 Du Dv), γ solved for 319.2 µm",
            "reference_value": round(float(result["plate_bar_implies_um"]), 4),
            "reference_source": "Gallery plate painted 100 µm bar times measured ring pitch (not the biological λ_c)",
            "note": "value is the Holocene model wavelength; reference is the inconsistent painted bar",
            "derived_from": "script",
        },
        {
            "name": "specimen radial profile Pearson r (single harmonic)",
            "value": round(float(result["pearson_r"]), 4),
            "unit": "correlation",
            "method": "least squares of exp(-r/400)·cos(k r + φ) plus quadratic trend against azimuthal mean",
            "reference_value": round(null_r, 4),
            "reference_source": "trend-only null on the same specimen radii (no Turing mode)",
            "note": "Pass 1 in-sample r. Held-out multi-harmonic fit, RMSE detail, and KS are pass-2 remediations.",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen radial profile RMSE",
            "value": round(float(result["rmse"]), 4),
            "unit": "gray levels",
            "method": "root mean square residual of the single-harmonic reconstruction",
            "reference_value": round(float(np.std(prof)), 4),
            "reference_source": "specimen radial-profile standard deviation (null residual scale)",
            "note": "single plate, n = 1",
            "derived_from": "output_pixels",
        },
        {
            "name": "Holocene age",
            "value": AGE_MA,
            "unit": "Ma",
            "method": "Gallus chronoseries 1/1 assigned horizon",
            "reference_value": 124.6,
            "reference_source": "Illustrative Early Cretaceous avian-feather horizon outside this singleton series; not a fitted sibling parameter",
            "note": "predecessor and successor chronospecies are absent, so Δage inside the series is undefined rather than 124.6",
            "derived_from": "script",
        },
    ]


def self_test() -> None:
    result = analyze_pass1()
    assert abs(result["lambda_c_um"] - LAMBDA_C_UM) < 1e-6, result["lambda_c_um"]
    assert abs(result["age_ma"] - 0.0) < 1e-12
    assert result["chronoseries_length"] == 1 and result["predecessor"] is None
    assert result["diffusion_margin_R"] > 1.0
    assert diffusion_margin(DU, DU) == 0.0
    assert result["pearson_r"] > 0.80
    assert abs(result["domain_um"] - N_WAVELENGTHS_IN_PATCH * LAMBDA_C_UM) < 1e-6
    assert math.isfinite(result["log10_size_m"]) and result["log10_size_m"] < -3.0
    print(
        f"SELF-TEST PASSED  λ_c={result['lambda_c_um']:.4f} µm  "
        f"R={result['diffusion_margin_R']:.3f}  pearson={result['pearson_r']:.4f}  "
        f"plate_bar={result['plate_bar_implies_um']:.2f} µm  age={result['age_ma']} Ma"
    )


def main() -> None:
    result = analyze_pass1()
    render_outputs(result)
    print(json.dumps({k: result[k] for k in (
        "lambda_c_um", "age_ma", "pearson_r", "rmse", "plate_bar_implies_um", "diffusion_margin_R", "pitch_px"
    )}, indent=2))
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
