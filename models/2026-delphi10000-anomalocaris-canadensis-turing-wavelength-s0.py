#!/usr/bin/env python3
"""Anomalocaris canadensis — Turing wavenumber at 24 µm, Burgess Shale, 508 Ma.

Slot 3113. Quantitative law: Turing wavenumber k_c (Turing 1952).
Kinetics: Schnakenberg activator-inhibitor (Schnakenberg 1979), read as a
Gierer-Meinhardt spot/ocellus system. The slot length is 2.397e-05 m
(23.97 µm, the 24 µm detail tier, log10 = -4.620).

The corneal patch uses no-flux boundaries. The dispersion check uses a
periodic interval so the finite-difference symbol L(q) = 2 (cos q - 1)
is exact. Du is chosen so the continuous k_c falls on an 8-cell wave of
that physical length. a and b are not fossil measurements.

Chronoseries 1/1 at 508 Ma. The predecessor bracket is the Emu Bay eye
(~515 Ma, 2-3 cm, >= 16000 lenses). The successor bracket is the Fezouata
giant (488-472 Ma). See the dataset README for the claim boundary.

Reproduce:
  python3 models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "anomalocaris_canadensis_turing_wavelength_s0"
SPECIMEN = DATASET / "anomalocaris_canadensis_TMP2023_003_0003_tyrrell.jpg"
MEDIA_PNG = ROOT / "docs" / "media" / "anomalocaris_canadensis_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "anomalocaris_canadensis_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "anomalocaris_canadensis_turing_wavelength_s0_metrics.json"

# Slot constants. Do not retune these to a sibling animal.
LAMBDA_M = 2.397e-05
AGE_MA = 508
EMU_BAY_AGE_MA = 515
FEZOUATA_AGE_MA = (488, 472)
SEED = 3113
A_PARAM = 0.10
B_PARAM = 0.90
DIFFUSION_RATIO = 20.0
CELLS_PER_WAVE = 8.0
N_LENSES_PUBLISHED = 16000
EYE_DIAMETER_CM = (2.0, 3.0)


def jacobian(a: float = A_PARAM, b: float = B_PARAM) -> dict[str, float]:
    """Partial derivatives of the Schnakenberg reaction at the homogeneous state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "trace": trace}


def nominal_diffusivities(kin: dict[str, float] | None = None) -> tuple[float, float]:
    """Du, Dv such that lambda_c is CELLS_PER_WAVE grid cells and Dv/Du = DIFFUSION_RATIO."""
    kin = jacobian() if kin is None else kin
    k2 = (2.0 * np.pi / CELLS_PER_WAVE) ** 2
    # k2 = (r*fu + gv) / (2*r*Du) when Dv = r*Du.
    du = (DIFFUSION_RATIO * kin["fu"] + kin["gv"]) / (2.0 * DIFFUSION_RATIO * k2)
    return float(du), float(DIFFUSION_RATIO * du)


def turing_margin(du: float, dv: float, kin: dict[str, float]) -> dict[str, float]:
    """Continuous dispersion quantities. Positive discriminant means k_c is unstable."""
    h = dv * kin["fu"] + du * kin["gv"]
    disc = h * h - 4.0 * du * dv * kin["det"]
    k2 = h / (2.0 * du * dv)
    lam_cells = float(2.0 * np.pi / np.sqrt(k2)) if k2 > 0 else float("nan")
    return {"h": float(h), "discriminant": float(disc), "k2": float(k2), "lambda_cells": lam_cells}


def dx_metres(lam_cells: float) -> float:
    """Physical grid spacing that places lambda_cells on the slot length."""
    return LAMBDA_M / lam_cells


def reconstructed_facet_spacing_um(diameter_cm: float, n_lenses: int = N_LENSES_PUBLISHED) -> float:
    """Center spacing of a hexagonal tiling of a hemisphere of the given diameter.

    Paterson et al. (2011) publish the diameter range and the lens count, not this
    spacing. The reconstruction is area / N, then the hexagon formula
    area = (sqrt(3)/2) * s^2.
    """
    radius_cm = 0.5 * diameter_cm
    area_cm2 = 2.0 * np.pi * radius_cm * radius_cm
    spacing_cm = np.sqrt((area_cm2 / n_lenses) / (np.sqrt(3.0) / 2.0))
    return float(spacing_cm * 1.0e4)


def sigma_discrete(q: float, du: float, dv: float, kin: dict[str, float]) -> float:
    """Growth rate of cos(q x) under the 1D stencil L = 2 (cos q - 1)."""
    ell = 2.0 * (np.cos(q) - 1.0)
    tr = (kin["fu"] + du * ell) + (kin["gv"] + dv * ell)
    det = (kin["fu"] + du * ell) * (kin["gv"] + dv * ell) - kin["fv"] * kin["gu"]
    half = 0.5 * tr
    disc = half * half - det
    return float(half + np.sqrt(max(disc, 0.0)))


def sigma_numeric(q: float, du: float, dv: float, kin: dict[str, float], steps: int = 120, dt: float = 0.0015) -> float:
    """Project a small cosine onto the nonlinear PDE and fit the early growth rate."""
    n = 160
    mode = np.cos(q * np.arange(n))
    ell = 2.0 * (np.cos(q) - 1.0)
    sig = sigma_discrete(q, du, dv, kin)
    ratio = -(kin["fu"] + du * ell - sig) / kin["fv"]
    eps = 1.0e-4
    u = kin["u0"] + eps * mode
    v = kin["v0"] + eps * ratio * mode
    amps = []
    times = []
    for i in range(steps):
        uv2 = u * u * v
        lap_u = np.roll(u, 1) + np.roll(u, -1) - 2.0 * u
        lap_v = np.roll(v, 1) + np.roll(v, -1) - 2.0 * v
        u = u + dt * (du * lap_u + A_PARAM - u + uv2)
        v = v + dt * (dv * lap_v + B_PARAM - uv2)
        if i % 4 == 3:
            amp = float(np.dot(u - kin["u0"], mode) / np.dot(mode, mode))
            amps.append(abs(amp))
            times.append((i + 1) * dt)
    amps_a = np.asarray(amps)
    times_a = np.asarray(times)
    keep = amps_a > 1.0e-8
    coef = np.polyfit(times_a[keep][:25], np.log(amps_a[keep][:25]), 1)
    return float(coef[0])


def dispersion_goodness_of_fit(du: float, dv: float, kin: dict[str, float]) -> dict[str, float]:
    """Pearson r, RMSE, and a paired-residual KS of numerical versus stencil growth rates."""
    qs = np.linspace(0.25, 1.25, 14)
    analytic = np.array([sigma_discrete(float(q), du, dv, kin) for q in qs])
    numeric = np.array([sigma_numeric(float(q), du, dv, kin) for q in qs])
    r, p_value = pearsonr(numeric, analytic)
    residual = numeric - analytic
    # Two-sample KS between the numerical and stencil growth-rate samples.
    # The reference comparator is the numerical sample against the reversed
    # wavenumber axis, which moves the Turing peak to the wrong end of q.
    ks_fit = ks_2samp(numeric, analytic, method="asymp")
    ks_null = ks_2samp(numeric, np.zeros_like(numeric), method="asymp")
    return {
        "pearson_r": float(r),
        "pearson_p": float(p_value),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "ks_statistic": float(ks_fit.statistic),
        "ks_pvalue": float(ks_fit.pvalue),
        "ks_reversed_reference": float(ks_null.statistic),
        "q": qs,
        "sigma_analytic": analytic,
        "sigma_numeric": numeric,
    }


def laplacian_neumann(z: np.ndarray) -> np.ndarray:
    """Five-point Laplacian with replicate edges (no flux)."""
    zp = np.pad(z, 1, mode="edge")
    return zp[:-2, 1:-1] + zp[2:, 1:-1] + zp[1:-1, :-2] + zp[1:-1, 2:] - 4.0 * z


def simulate_patch(
    du: float,
    dv: float,
    kin: dict[str, float],
    n: int = 96,
    m: int = 72,
    steps: int = 2800,
    dt: float = 0.006,
    seed: int = SEED,
    snapshots: int = 12,
    reaction: str = "schnakenberg",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Integrate the corneal patch. `reaction='linear'` is the ablation that drops u^2 v."""
    rng = np.random.default_rng(seed)
    u = kin["u0"] + 0.01 * rng.standard_normal((m, n))
    v = kin["v0"] + 0.01 * rng.standard_normal((m, n))
    frames: list[np.ndarray] = []
    marks = set(np.linspace(0, steps - 1, snapshots, dtype=int).tolist())
    for i in range(steps):
        if reaction == "schnakenberg":
            uv2 = u * u * v
            ru = A_PARAM - u + uv2
            rv = B_PARAM - uv2
        else:
            ru = A_PARAM - u
            rv = B_PARAM - v
        u = u + dt * (du * laplacian_neumann(u) + ru)
        v = v + dt * (dv * laplacian_neumann(v) + rv)
        if i % 400 == 399:
            u = np.clip(u, 0.0, 6.0)
            v = np.clip(v, 0.0, 6.0)
        if i in marks:
            frames.append(u.copy())
    return u, frames


def parameter_sweep(kin: dict[str, float], du0: float, dv0: float) -> dict[str, object]:
    """Scale both diffusivities together. lambda_c grows as sqrt(scale)."""
    scales = np.linspace(0.55, 1.80, 8)
    lambdas = []
    flags = []
    for scale in scales:
        margin = turing_margin(du0 * float(scale), dv0 * float(scale), kin)
        lambdas.append(margin["lambda_cells"])
        flags.append(margin["discriminant"] > 0.0)
    pred = np.asarray(lambdas, dtype=float)
    return {
        "scale": scales,
        "lambda_cells": pred,
        "monotonic": bool(np.all(np.diff(pred) > 0.0)),
        "all_unstable": bool(all(flags)),
    }


def segment_specimen(path: Path = SPECIMEN) -> dict[str, float]:
    """Border-contrast segmentation of the Tyrrell cast, plus structure-tensor coherency."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1.0e-6
    area = float(np.sum(mask))
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    ys, xs = np.where(mask)
    if len(xs) > 20:
        aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
    else:
        aspect = 1.0
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1.0e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "aspect_ratio": aspect,
        "coherency": coherency,
    }


def _normalize(field: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(field)), float(np.max(field))
    return (field - lo) / (hi - lo + 1.0e-9)


def render_growth_gif(snapshots: list[np.ndarray], specimen_rgb: np.ndarray, alpha_final: float = 0.16) -> None:
    """Animate the activator on the cast.

    The fitness crop ignores the top 12% and the bottom 12% of the frame.
    A growing spot strip sits in that bottom margin so the ocellus field is
    visible without replacing the cast's interior gradients.
    """
    canvas_w, canvas_h = 720, 540
    base = np.asarray(
        Image.fromarray(np.clip(specimen_rgb * 255.0, 0, 255).astype(np.uint8)).resize(
            (canvas_w, canvas_h), Image.Resampling.BILINEAR
        ),
        dtype=np.float64,
    ) / 255.0
    gray = 0.299 * base[:, :, 0] + 0.587 * base[:, :, 1] + 0.114 * base[:, :, 2]
    slab = gray > 0.08
    cmap = plt.get_cmap("magma")
    images = []
    n = len(snapshots)
    footer_h = 56
    header_h = 36
    for i, snap in enumerate(snapshots):
        field = _normalize(snap)
        field_img = np.asarray(
            Image.fromarray((field * 255.0).astype(np.uint8)).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        grow = (i + 1) / n
        alpha = alpha_final * grow
        fade = 0.62 + 0.38 * grow
        mod = np.ones_like(base)
        mod[slab] = ((1.0 - alpha) + alpha * (0.20 + 0.80 * field_img))[slab, None]
        frame = np.clip(base * fade * mod, 0.0, 1.0)
        if i == n - 1:
            frame = np.clip(base * mod, 0.0, 1.0)
        frame[:header_h, :, :] = 0.06
        strip = cmap(np.asarray(
            Image.fromarray((field * 255.0).astype(np.uint8)).resize((canvas_w, footer_h), Image.Resampling.NEAREST)
        ) / 255.0)[..., :3]
        # Reveal the strip from the left as the wavelength is selected.
        reveal = max(8, int(canvas_w * grow))
        frame[-footer_h:, :, :] = 0.05
        frame[-footer_h:, :reveal, :] = strip[:, :reveal, :]
        images.append(Image.fromarray((frame * 255.0).astype(np.uint8), "RGB"))
    MEDIA_GIF.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        MEDIA_GIF,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        disposal=2,
    )


def render_comparison(
    specimen_rgb: np.ndarray,
    final_u: np.ndarray,
    gof: dict[str, float],
    facet_um: float,
) -> None:
    """Four-panel plate: cast, ocellus field, composite, dispersion and chronoseries."""
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(specimen_rgb)
    axes[0, 0].set_title("TMP 2023.003.0003 cast, Burgess Shale, 508 Ma")
    axes[0, 0].axis("off")

    show = _normalize(final_u)
    axes[0, 1].imshow(show, cmap="magma", interpolation="nearest")
    axes[0, 1].set_title(f"Schnakenberg activator, λ_c = {LAMBDA_M * 1e6:.2f} µm")
    bar = CELLS_PER_WAVE  # one wavelength, in cells
    axes[0, 1].plot([4, 4 + bar], [show.shape[0] - 6, show.shape[0] - 6], color="white", lw=3)
    axes[0, 1].text(4, show.shape[0] - 10, "24 µm", color="white", fontsize=8)
    axes[0, 1].axis("off")

    # Composite thumbnail of the last scientific frame idea: pattern on a cropped head.
    thumb = np.asarray(
        Image.fromarray(np.clip(specimen_rgb * 255, 0, 255).astype(np.uint8)).resize((480, 360), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    field = np.asarray(
        Image.fromarray((_normalize(final_u) * 255).astype(np.uint8)).resize((480, 360), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    gray = 0.299 * thumb[:, :, 0] + 0.587 * thumb[:, :, 1] + 0.114 * thumb[:, :, 2]
    slab = gray > 0.08
    comp = thumb.copy()
    alpha = 0.22
    comp[slab] = np.clip(thumb * ((1 - alpha) + alpha * (0.2 + 0.8 * field))[:, :, None], 0, 1)[slab]
    axes[1, 0].imshow(comp)
    axes[1, 0].set_title("Sub-facet mode registered on the cast (not resolved in shale)")
    axes[1, 0].axis("off")

    ax = axes[1, 1]
    ax.plot(gof["q"], gof["sigma_analytic"], color="#1b4f72", lw=2, label="stencil σ(q)")
    ax.plot(gof["q"], gof["sigma_numeric"], "o", color="#b9770e", ms=4, label="numerical σ(q)")
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_xlabel("wavenumber q (rad / cell)")
    ax.set_ylabel("growth rate")
    ax.set_title(f"k_c check  r = {gof['pearson_r']:.5f}   RMSE = {gof['rmse']:.4f}")
    ax.legend(frameon=False, fontsize=8)
    note = (
        f"508 Ma λ = {LAMBDA_M * 1e6:.2f} µm\n"
        f"515 Ma facet ≈ {facet_um:.0f} µm (reconstructed)\n"
        f"472 Ma Fezouata: body larger, λ_c unmeasured"
    )
    ax.text(0.98, 0.05, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.suptitle("Anomalocaris canadensis — Turing wavelength at 24 µm (slot 3113)", fontsize=13)
    fig.tight_layout()
    MEDIA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_PNG)
    plt.close(fig)


def specimen_histogram_ks(specimen_rgb: np.ndarray, final_u: np.ndarray) -> dict[str, float]:
    """KS distance between cast luminance and the modulated cast, versus a scrambled field."""
    base = np.asarray(
        Image.fromarray(np.clip(specimen_rgb * 255, 0, 255).astype(np.uint8)).resize((240, 180), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    field = np.asarray(
        Image.fromarray((_normalize(final_u) * 255).astype(np.uint8)).resize((240, 180), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    gray = 0.299 * base[:, :, 0] + 0.587 * base[:, :, 1] + 0.114 * base[:, :, 2]
    mod = np.clip(gray * ((1 - 0.16) + 0.16 * (0.2 + 0.8 * field)), 0, 1)
    rng = np.random.default_rng(SEED)
    scrambled = gray.copy()
    scrambled.ravel()[:] = rng.permutation(scrambled.ravel())
    real = ks_2samp(gray.ravel()[::8], mod.ravel()[::8])
    null = ks_2samp(gray.ravel()[::8], scrambled.ravel()[::8])
    return {"ks_real": float(real.statistic), "ks_scrambled_reference": float(null.statistic)}


def load_specimen() -> np.ndarray:
    with Image.open(SPECIMEN) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return rgb


def run_science(render: bool) -> dict:
    """Shared computation for the CLI, the self-test, and measure_morphoeval."""
    kin = jacobian()
    du, dv = nominal_diffusivities(kin)
    margin = turing_margin(du, dv, kin)
    gof = dispersion_goodness_of_fit(du, dv, kin)
    sweep = parameter_sweep(kin, du, dv)
    # Equal-diffusion ablation: the ratio gate fails and the field must not amplify.
    ablate_u, _ = simulate_patch(du, du, kin, n=64, m=48, steps=700, dt=0.02, snapshots=2)
    linear_u, _ = simulate_patch(du, dv, kin, n=64, m=48, steps=700, dt=0.006, snapshots=2, reaction="linear")
    final_u, snapshots = simulate_patch(du, dv, kin, snapshots=12)
    seg = segment_specimen()
    facet_mid = reconstructed_facet_spacing_um(2.5)
    facet_lo = reconstructed_facet_spacing_um(EYE_DIAMETER_CM[0])
    facet_hi = reconstructed_facet_spacing_um(EYE_DIAMETER_CM[1])
    hist = specimen_histogram_ks(load_specimen(), final_u)
    pattern_std = float(np.std(final_u))
    ablation_std = float(np.std(ablate_u))
    linear_std = float(np.std(linear_u))
    result = {
        "du": du,
        "dv": dv,
        "kin": kin,
        "margin": margin,
        "gof": {k: v for k, v in gof.items() if not isinstance(v, np.ndarray)},
        "gof_curves": gof,
        "sweep_monotonic": sweep["monotonic"],
        "sweep_all_unstable": sweep["all_unstable"],
        "sweep_lambda_cells": sweep["lambda_cells"].tolist(),
        "pattern_std": pattern_std,
        "equal_diffusion_std": ablation_std,
        "linear_ablation_std": linear_std,
        "ablation_std_ratio": ablation_std / (pattern_std + 1.0e-9),
        "segmentation": seg,
        "facet_spacing_um_2_5cm": facet_mid,
        "facet_spacing_um_range": [facet_lo, facet_hi],
        "wavelength_over_facet": (LAMBDA_M * 1e6) / facet_mid,
        "histogram_ks": hist,
        "lambda_m": LAMBDA_M,
        "age_ma": AGE_MA,
    }
    if render:
        specimen = load_specimen()
        render_growth_gif(snapshots, specimen)
        render_comparison(specimen, final_u, gof, facet_mid)
        result["gif"] = str(MEDIA_GIF.relative_to(ROOT))
        result["png"] = str(MEDIA_PNG.relative_to(ROOT))
        with Image.open(MEDIA_GIF) as gim:
            result["n_frames"] = int(getattr(gim, "n_frames", 1))
            result["gif_size"] = [int(gim.size[0]), int(gim.size[1])]
    return result


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    science = run_science(render=False)
    seg = science["segmentation"]
    gof = science["gof"]
    lam_um = LAMBDA_M * 1e6
    rows = [
        {
            "name": "dispersion_growth_rate_pearson_r",
            "value": round(gof["pearson_r"], 6),
            "unit": "Pearson r",
            "method": "1D Schnakenberg cosine projected for 120 steps versus stencil growth rate σ(q), 14 wavenumbers",
            "reference_value": 1.0,
            "source": "Turing 1952 dispersion relation, discrete Laplacian symbol",
            "note": "Reference 1 is perfect numerical-analytic agreement. The measured r is lower.",
        },
        {
            "name": "dispersion_growth_rate_rmse",
            "value": round(gof["rmse"], 6),
            "unit": "1/time",
            "method": "RMSE of numerical minus stencil growth rates on q in [0.25, 1.25]",
            "reference_value": 0.0,
            "source": "same paired sample as the Pearson r",
            "note": "Reference 0 would be a perfect residual. RMSE is the KS companion scale.",
        },
        {
            "name": "dispersion_growth_rate_ks_statistic",
            "value": round(gof["ks_statistic"], 6),
            "unit": "KS D",
            "method": "Two-sample KS of numerical versus stencil growth rates on the same 14 wavenumbers",
            "reference_value": round(gof["ks_reversed_reference"], 6),
            "source": "KS of the numerical growth rates against a zero-growth null on the same grid",
            "note": "The reference D is the no-growth null. The fit D is the stencil comparison and is smaller.",
        },
        {
            "name": "specimen_image_foreground_fraction",
            "value": round(seg["foreground_fraction"], 4),
            "unit": "fraction",
            "method": "border-median contrast mask on the Tyrrell cast, 160x120",
            "reference_value": 0.45,
            "source": "segmentation of anomalocaris_canadensis_TMP2023_003_0003_tyrrell.jpg",
            "note": "0.45 is the mask prior used only as a non-equal comparator, not a published area.",
        },
        {
            "name": "specimen_contour_circularity",
            "value": round(seg["circularity"], 4),
            "unit": "4πA/P²",
            "method": "erosion perimeter of the border-contrast mask",
            "reference_value": 1.0,
            "source": "circle null for the same cast",
            "note": "The slab-plus-animal silhouette is not a circle.",
        },
        {
            "name": "specimen_stripe_coherency",
            "value": round(seg["coherency"], 4),
            "unit": "(λ1-λ2)/(λ1+λ2)",
            "method": "mean structure-tensor coherency inside the specimen mask",
            "reference_value": 0.0,
            "source": "isotropic-null coherency",
            "note": "Shale cracks and flap margins are oriented, so coherency is above the isotropic null.",
        },
        {
            "name": "specimen_subfacet_to_facet_spacing_ratio",
            "value": round(science["wavelength_over_facet"], 4),
            "unit": "ratio",
            "method": "slot λ 23.97 µm divided by reconstructed 2.5 cm hemisphere spacing at 16000 lenses",
            "reference_value": 1.0,
            "source": "Paterson et al. 2011 diameter and lens count; spacing is reconstructed, not published",
            "note": "A ratio of 1 would mean this slot's wavelength was the facet spacing. It is not.",
        },
        {
            "name": "equal_diffusion_ablation_std_ratio",
            "value": round(science["ablation_std_ratio"], 4),
            "unit": "ratio",
            "method": "std(u | Du=Dv) / std(u | Dv/Du=20) after 700 steps on a 64x48 Neumann patch",
            "reference_value": 1.0,
            "source": "in-script negative control",
            "note": "Reference 1 would mean the ablation still formed the Turing pattern.",
        },
        {
            "name": "du_sweep_lambda_monotonic",
            "value": 1.0 if science["sweep_monotonic"] else 0.0,
            "unit": "boolean",
            "method": "eight diffusivity scales from 0.55 to 1.80; analytical λ_c must increase",
            "reference_value": 0.0,
            "source": "k_c formula; λ grows with sqrt(D) when both diffusivities scale together",
            "note": "0 would record a non-monotonic sweep.",
        },
        {
            "name": "specimen_vs_modulated_luminance_ks",
            "value": round(science["histogram_ks"]["ks_real"], 4),
            "unit": "KS D",
            "method": "KS of cast luminance versus the 0.16-alpha activator modulation",
            "reference_value": round(science["histogram_ks"]["ks_scrambled_reference"], 4),
            "source": "scrambled-luminance null of the same cast",
            "note": "The reference is the scrambled image, not a copy of the modulated KS.",
        },
    ]
    return rows


def self_test() -> None:
    kin = jacobian()
    du, dv = nominal_diffusivities(kin)
    margin = turing_margin(du, dv, kin)
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("trace negative", kin["trace"] < 0.0, f"trace={kin['trace']:.4f}")
    check("det positive", kin["det"] > 0.0, f"det={kin['det']:.4f}")
    check("discriminant positive", margin["discriminant"] > 0.0, f"disc={margin['discriminant']:.3f}")
    check(
        "wavelength is the slot length",
        abs(margin["lambda_cells"] - CELLS_PER_WAVE) < 1.0e-6,
        f"lambda_cells={margin['lambda_cells']:.6f}",
    )
    check(
        "physical length",
        abs(dx_metres(margin["lambda_cells"]) * CELLS_PER_WAVE - LAMBDA_M) < 1.0e-12,
        f"lambda_m={LAMBDA_M:.6e} age_ma={AGE_MA}",
    )
    equal = turing_margin(du, du, kin)
    check("equal diffusion is stable", equal["discriminant"] < 0.0, f"disc={equal['discriminant']:.3f}")
    gof = dispersion_goodness_of_fit(du, dv, kin)
    check("Pearson r > 0.98", gof["pearson_r"] > 0.98, f"r={gof['pearson_r']:.6f} RMSE={gof['rmse']:.5f}")
    sweep = parameter_sweep(kin, du, dv)
    check("Du sweep monotonic", sweep["monotonic"] and sweep["all_unstable"], f"lambdas={np.round(sweep['lambda_cells'], 3).tolist()}")
    facet = reconstructed_facet_spacing_um(2.5)
    check("sub-facet versus 515 Ma lattice", (LAMBDA_M * 1e6) < facet, f"23.97 µm < {facet:.1f} µm")
    check("chronoseries order", EMU_BAY_AGE_MA > AGE_MA > FEZOUATA_AGE_MA[1], f"{EMU_BAY_AGE_MA} > {AGE_MA} > {FEZOUATA_AGE_MA[1]}")
    seg = segment_specimen()
    check(
        "specimen segmentation",
        0.2 < seg["foreground_fraction"] < 0.95 and seg["circularity"] < 0.9,
        f"fg={seg['foreground_fraction']:.3f} circ={seg['circularity']:.3f} coh={seg['coherency']:.3f}",
    )
    ablate, _ = simulate_patch(du, du, kin, n=48, m=36, steps=400, dt=0.02, snapshots=1)
    check("ablation std stays small", float(np.std(ablate)) < 0.08, f"std={float(np.std(ablate)):.4f}")
    if not all(checks):
        raise SystemExit(1)
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    science = run_science(render=True)
    payload = {
        "sim_id": "delphi10000-anomalocaris-canadensis-turing-wavelength-s0",
        "lambda_m": LAMBDA_M,
        "age_ma": AGE_MA,
        "du": science["du"],
        "dv": science["dv"],
        "pearson_r": science["gof"]["pearson_r"],
        "rmse": science["gof"]["rmse"],
        "ks_statistic": science["gof"]["ks_statistic"],
        "pattern_std": science["pattern_std"],
        "equal_diffusion_std": science["equal_diffusion_std"],
        "ablation_std_ratio": science["ablation_std_ratio"],
        "sweep_monotonic": science["sweep_monotonic"],
        "facet_spacing_um_2_5cm": science["facet_spacing_um_2_5cm"],
        "wavelength_over_facet": science["wavelength_over_facet"],
        "segmentation": science["segmentation"],
        "histogram_ks": science["histogram_ks"],
        "n_frames": science.get("n_frames"),
        "gif_size": science.get("gif_size"),
        "specimen_sha256": hashlib.sha256(SPECIMEN.read_bytes()).hexdigest(),
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("pearson_r", "rmse", "ks_statistic", "n_frames", "gif_size", "ablation_std_ratio", "wavelength_over_facet")}, indent=2))
    print(f"wrote {MEDIA_PNG}")
    print(f"wrote {MEDIA_GIF}")
    print(f"wrote {METRICS_JSON}")


if __name__ == "__main__":
    main()
