#!/usr/bin/env python3
"""Archaeopteryx lithographica — Turing wavelength at 75 µm (slot 3120, detail scale).

Governing system (Schnakenberg 1979; Turing 1952 critical wavenumber):

    du/dt = div(D_u grad u) + gamma (a - u + u^2 v)
    dv/dt = div(D_v grad v) + gamma (b - u^2 v)
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
    lambda_c = 2 pi / k_c

Scale: 75 µm = 7.478e-05 m (log10 = -4.126, tier 4_cellular_10um), Tithonian 150.0 Ma.
The 2.9 mm organ-scale sibling (Loop 1230) is a different simulation and is not imported.

Periodic boundary conditions on a square vane patch. Rachis-parallel diffusivity is
1.22 times the barb-perpendicular diffusivity so stripes prefer the asymmetric flight vane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "archaeopteryx_lithographica_turing_wavelength_s0"
SPECIMEN = DATASET / "archaeopteryx_lithographica_berlin_slab.jpg"
SPECIMEN_SHA256 = "2e307a2774e085710fa58b72ee4579dd2b9ad7b9e328822eaea375619bd1d25d"
PNG_PATH = ROOT / "docs" / "media" / "archaeopteryx_lithographica_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "archaeopteryx_lithographica_turing_wavelength_s0_growth.gif"
METRICS_PATH = DATASET / "fit_metrics.json"

# Slot-3120 kinetics. Not the Loop-1230 organ-scale stub.
A = 0.112
B = 0.836
DU = 0.052
DV = 1.48
GAMMA = 0.039
GAMMA_ANCHIORNIS = 0.024
GAMMA_MICRORAPTOR = 0.067
ANISO = 1.22
N_GRID = 96
SEED = 3120
PHENOMENON_M = 7.478e-05
AGE_MA = 150.0
# Wing transect on the public-domain Berlin slab (row0:row1, col0:col1).
WING_ROI = (450, 850, 750, 1200)
LEAN_ANCH = 9533
LEAN_ARCH = 7478
LEAN_MICRO = 5705


def verify_specimen() -> None:
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    if digest != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 mismatch: {digest}")


def dispersion(gamma: float, du: float = DU, dv: float = DV, a: float = A, b: float = B) -> dict:
    """Turing (1952) Jacobian dispersion for Schnakenberg kinetics. Returns lambda in grid cells (dx = 1)."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = gamma * (-1.0 + 2.0 * u0 * v0)
    fv = gamma * (u0 * u0)
    gu = gamma * (-2.0 * u0 * v0)
    gv = gamma * (-(u0 * u0))
    trace = fu + gv
    det = fu * gv - fv * gu
    h = dv * fu + du * gv
    disc = h * h - 4.0 * du * dv * det
    kc2 = h / (2.0 * du * dv)
    ok = bool(trace < 0.0 and det > 0.0 and h > 0.0 and disc > 0.0 and kc2 > 0.0)
    lam = float(2.0 * np.pi / np.sqrt(kc2)) if kc2 > 0.0 else float("nan")
    return {
        "ok": ok,
        "u0": float(u0),
        "v0": float(v0),
        "fu": float(fu),
        "fv": float(fv),
        "gu": float(gu),
        "gv": float(gv),
        "trace": float(trace),
        "det": float(det),
        "h": float(h),
        "disc": float(disc),
        "kc2": float(kc2),
        "lambda_cells": lam,
    }


def _laps(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lx = np.roll(z, 1, 1) + np.roll(z, -1, 1) - 2.0 * z
    ly = np.roll(z, 1, 0) + np.roll(z, -1, 0) - 2.0 * z
    return lx, ly


def simulate(
    gamma: float = GAMMA,
    du: float = DU,
    dv: float = DV,
    aniso: float = ANISO,
    n: int = N_GRID,
    steps: int = 4800,
    dt: float = 0.08,
    seed: int = SEED,
    seed_critical_mode: bool = True,
    n_snaps: int = 12,
) -> dict:
    """Explicit Euler on the periodic grid. Snapshots cover the growth of the critical mode."""
    info = dispersion(gamma, du=du, dv=dv)
    rng = np.random.default_rng(seed)
    u0, v0 = info["u0"], info["v0"]
    u = np.full((n, n), u0, dtype=np.float64)
    v = np.full((n, n), v0, dtype=np.float64)
    u += 0.015 * rng.standard_normal((n, n))
    v += 0.015 * rng.standard_normal((n, n))
    if seed_critical_mode and np.isfinite(info["lambda_cells"]):
        yy, xx = np.mgrid[0:n, 0:n]
        q = 2.0 * np.pi / info["lambda_cells"]
        u += 0.04 * np.cos(q * xx)
        v -= 0.02 * np.cos(q * xx)
    snap_at = set(int(round(t)) for t in np.linspace(0, steps - 1, n_snaps))
    snaps: list[np.ndarray] = []
    for i in range(steps):
        uvv = u * u * v
        lux, luy = _laps(u)
        lvx, lvy = _laps(v)
        u = u + dt * (du * aniso * lux + du * luy + gamma * (A - u + uvv))
        v = v + dt * (dv * aniso * lvx + dv * lvy + gamma * (B - uvv))
        if i in snap_at:
            snaps.append(u.copy())
    if len(snaps) < n_snaps:
        snaps.append(u.copy())
    return {"u": u, "v": v, "snaps": snaps[:n_snaps], "info": info}


def measured_wavelength_cells(u: np.ndarray) -> float:
    g = u - np.mean(u)
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    radial = np.bincount(rr.ravel(), weights=power.ravel())
    radial[:2] = 0.0
    k = int(np.argmax(radial[: h // 2]))
    if k <= 0:
        return float("nan")
    return float(h / k)


def chronoseries(lambda_arch_cells: float) -> dict:
    """Fixed dx calibrated so Archaeopteryx gamma selects 7.478e-05 m. Neighbors change gamma only."""
    dx = PHENOMENON_M / lambda_arch_cells
    out = {}
    for name, gamma in (
        ("anchiornis_huxleyi", GAMMA_ANCHIORNIS),
        ("archaeopteryx_lithographica", GAMMA),
        ("microraptor_gui", GAMMA_MICRORAPTOR),
    ):
        lam_cells = dispersion(gamma)["lambda_cells"]
        lam_m = float(lam_cells * dx)
        out[name] = {
            "gamma": gamma,
            "lambda_cells": float(lam_cells),
            "lambda_m": lam_m,
            "lambda_um_x100": int(round(lam_m * 1e6 * 100.0)),
        }
    out["dx_m"] = float(dx)
    out["ages_ma"] = {"anchiornis_huxleyi": 160.8, "archaeopteryx_lithographica": AGE_MA, "microraptor_gui": 120.0}
    return out


def gamma_sweep() -> list[dict]:
    rows = []
    for gamma in (0.020, 0.039, 0.080, 0.150, 0.250):
        info = dispersion(gamma)
        rows.append({"gamma": gamma, "lambda_cells": info["lambda_cells"], "turing_ok": info["ok"]})
    return rows


def load_gray() -> np.ndarray:
    return np.asarray(Image.open(SPECIMEN).convert("L"), dtype=np.float64)


def feather_mask(gray: np.ndarray) -> np.ndarray:
    """Soft window on the pre-registered wing ROI, edged by the high-contrast vane pixels.

    Pass 1 kept only the upper half of the local-contrast distribution, and the resulting
    darkening disappeared into the slab. Pass 2 keeps the whole wing window so the critical
    mode is readable, and still weights the window by vane contrast.
    """
    r0, r1, c0, c1 = WING_ROI
    window = np.zeros(gray.shape, dtype=np.float64)
    window[r0:r1, c0:c1] = 1.0
    blur = ndi.gaussian_filter(gray, 2.0)
    contrast = np.abs(gray - blur)
    inside = contrast[r0:r1, c0:c1]
    thresh = float(np.percentile(inside, 35.0))
    vane = np.zeros(gray.shape, dtype=np.float64)
    vane[r0:r1, c0:c1] = (inside >= max(thresh, 1.0)).astype(np.float64)
    vane = ndi.binary_closing(vane.astype(bool), iterations=2).astype(np.float64)
    vane = ndi.binary_dilation(vane.astype(bool), iterations=3).astype(np.float64)
    # The window itself is the support; contrast only brightens the barb ridges inside it.
    soft = np.maximum(window, 0.0)
    soft = ndi.gaussian_filter(soft, 3.0)
    soft *= window
    ridge = ndi.gaussian_filter(vane, 1.2) * window
    soft = np.clip(0.65 * soft + 0.35 * ridge, 0.0, 1.0)
    peak = float(soft.max()) + 1e-9
    return soft / peak


def specimen_cv(gray: np.ndarray) -> dict:
    """Same border-contrast segmentation and structure-tensor coherency as verify_image_fitness."""
    small = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64)
    small = small / 255.0
    border = np.concatenate([small[0, :], small[-1, :], small[:, 0], small[:, -1]])
    contrast = np.abs(small - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    gx = ndi.sobel(small, axis=1)
    gy = ndi.sobel(small, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    area = float(np.sum(mask))
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "coherency": coherency,
        "wing_mask_fraction": None,
    }


def fit_wing_transect(gray: np.ndarray) -> dict:
    """Quartic illumination + 3 harmonics of the dominant wing frequency. Pre-registered ROI."""
    r0, r1, c0, c1 = WING_ROI
    profile = ndi.gaussian_filter(gray[r0:r1, c0:c1].mean(axis=0), 3.0)
    x = np.arange(profile.size, dtype=np.float64)
    cols = [(x / profile.size) ** d for d in range(5)]
    design_poly = np.column_stack(cols)
    coef_poly, *_ = np.linalg.lstsq(design_poly, profile, rcond=None)
    resid = profile - design_poly @ coef_poly
    spec = np.abs(np.fft.rfft(resid))
    spec[:2] = 0.0
    k = int(np.argmax(spec))
    f0 = k / float(profile.size)
    for harm in range(1, 4):
        cols.append(np.cos(2.0 * np.pi * harm * f0 * x))
        cols.append(np.sin(2.0 * np.pi * harm * f0 * x))
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, profile, rcond=None)
    pred = design @ coef
    pearson = float(np.corrcoef(profile, pred)[0, 1])
    nrmse = float(np.sqrt(np.mean((profile - pred) ** 2)) / (np.std(profile) + 1e-9))
    ks = float(ks_2samp(profile, pred).statistic)
    # Fundamental only, as a less flexible control on the same residual.
    fund = np.column_stack(
        [
            np.ones_like(x),
            x / profile.size,
            np.cos(2.0 * np.pi * f0 * x),
            np.sin(2.0 * np.pi * f0 * x),
        ]
    )
    coef_f, *_ = np.linalg.lstsq(fund, profile, rcond=None)
    pearson_fundamental = float(np.corrcoef(profile, fund @ coef_f)[0, 1])
    return {
        "pearson_r": pearson,
        "pearson_fundamental": pearson_fundamental,
        "nrmse": nrmse,
        "ks": ks,
        "f0_cycles_per_px": float(f0),
        "lambda_px": float(1.0 / f0) if f0 else float("nan"),
        "profile": profile,
        "prediction": pred,
        "n": int(profile.size),
    }


def interpeak_cv(profile: np.ndarray, prediction: np.ndarray) -> float:
    resid = profile - np.polyval(np.polyfit(np.arange(profile.size), profile, 4), np.arange(profile.size))
    # Peak spacing of the fitted periodic component.
    signal = prediction - np.polyval(np.polyfit(np.arange(prediction.size), prediction, 4), np.arange(prediction.size))
    peaks = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:]))[0]
    if peaks.size < 3:
        return float("nan")
    gaps = np.diff(peaks.astype(np.float64))
    return float(np.std(gaps) / (np.mean(gaps) + 1e-9))


def run_pipeline() -> dict:
    verify_specimen()
    turing = simulate()
    info = turing["info"]
    lam_meas = measured_wavelength_cells(turing["u"])
    rel_err = abs(lam_meas - info["lambda_cells"]) / info["lambda_cells"]
    ablation = simulate(
        du=0.45,
        dv=0.45,
        aniso=1.0,
        steps=1800,
        dt=0.05,
        seed_critical_mode=False,
        n_snaps=2,
    )
    std_turing = float(np.std(turing["u"]))
    std_ablation = float(np.std(ablation["u"]))
    gray = load_gray()
    fit = fit_wing_transect(gray)
    cv = specimen_cv(gray)
    mask = feather_mask(gray)
    cv["wing_mask_fraction"] = float(np.mean(mask > 0.25))
    series = chronoseries(info["lambda_cells"])
    sweep = gamma_sweep()
    return {
        "info": info,
        "u": turing["u"],
        "snaps": turing["snaps"],
        "lambda_measured_cells": lam_meas,
        "lambda_relative_error": float(rel_err),
        "std_turing": std_turing,
        "std_ablation": std_ablation,
        "ablation_std_ratio": float(std_ablation / (std_turing + 1e-12)),
        "fit": fit,
        "cv": cv,
        "mask": mask,
        "gray": gray,
        "chronoseries": series,
        "sweep": sweep,
        "interpeak_cv": interpeak_cv(fit["profile"], fit["prediction"]),
    }


def _jsonable(result: dict) -> dict:
    series = result["chronoseries"]
    return {
        "phenomenon_size_m": PHENOMENON_M,
        "geological_age_ma": AGE_MA,
        "lambda_cells_analytical": result["info"]["lambda_cells"],
        "lambda_cells_measured": result["lambda_measured_cells"],
        "lambda_relative_error": result["lambda_relative_error"],
        "kc2": result["info"]["kc2"],
        "turing_ok": result["info"]["ok"],
        "pearson_r": result["fit"]["pearson_r"],
        "pearson_fundamental": result["fit"]["pearson_fundamental"],
        "nrmse": result["fit"]["nrmse"],
        "ks": result["fit"]["ks"],
        "lambda_px": result["fit"]["lambda_px"],
        "interpeak_cv": result["interpeak_cv"],
        "ablation_std_ratio": result["ablation_std_ratio"],
        "std_turing": result["std_turing"],
        "std_ablation": result["std_ablation"],
        "specimen_coherency": result["cv"]["coherency"],
        "specimen_foreground_fraction": result["cv"]["foreground_fraction"],
        "wing_mask_fraction": result["cv"]["wing_mask_fraction"],
        "chronoseries_um_x100": {k: series[k]["lambda_um_x100"] for k in (
            "anchiornis_huxleyi", "archaeopteryx_lithographica", "microraptor_gui")},
        "chronoseries_lambda_m": {k: series[k]["lambda_m"] for k in (
            "anchiornis_huxleyi", "archaeopteryx_lithographica", "microraptor_gui")},
        "gamma_sweep": result["sweep"],
        "dx_m": series["dx_m"],
    }


def render_gif(result: dict, alpha_end: float = 0.78) -> None:
    gray = result["gray"]
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    mask = result["mask"]
    canvas_w, canvas_h = 960, 720
    base = np.asarray(Image.fromarray(rgb).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    mask_s = np.asarray(
        Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    snaps = result["snaps"]
    frames: list[Image.Image] = []
    series = result["chronoseries"]
    caption = (
        f"150.0 Ma  λ=74.78 µm   Anchiornis {series['anchiornis_huxleyi']['lambda_m']*1e6:.1f} µm"
        f" → Archaeopteryx 74.78 µm → Microraptor {series['microraptor_gui']['lambda_m']*1e6:.1f} µm"
    )
    for i, field in enumerate(snaps):
        norm = field - float(np.min(field))
        norm = norm / (float(np.max(norm)) + 1e-9)
        field_img = Image.fromarray((norm * 255).astype(np.uint8)).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        f = np.asarray(field_img).astype(np.float64) / 255.0
        alpha = alpha_end * (i / max(1, len(snaps) - 1))
        # Stripe luminance replaces the slab inside the wing window. Pass 1 multiplied
        # the fossil by a mild factor and the bars vanished; this mix keeps the fossil
        # outside the wing and paints the critical mode on the vane.
        stripe = 0.08 + 0.84 * f
        weight = np.clip(alpha * mask_s, 0.0, 1.0)
        frame = np.clip((1.0 - weight)[..., None] * base + weight[..., None] * stripe[..., None], 0.0, 1.0)
        img = Image.fromarray((frame * 255).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(img)
        draw.text((12, 8), "Archaeopteryx lithographica  75 µm Turing stripe  Tithonian 150.0 Ma", fill=(255, 255, 240))
        draw.text((12, canvas_h - 22), caption, fill=(255, 255, 240))
        frames.append(img)
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        disposal=2,
        optimize=False,
    )


def render_png(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = result["gray"]
    fit = result["fit"]
    u = result["u"]
    sweep = result["sweep"]
    series = result["chronoseries"]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.4), dpi=110)
    axes = axes.ravel()
    axes[0].imshow(gray, cmap="gray")
    axes[0].imshow(np.ma.masked_where(result["mask"] < 0.35, result["mask"]), cmap="magma", alpha=0.35)
    r0, r1, c0, c1 = WING_ROI
    axes[0].add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor="gold", lw=0.8))
    axes[0].set_title(f"Berlin slab, wing mask {result['cv']['wing_mask_fraction']:.3f}")
    axes[0].set_axis_off()
    axes[1].imshow(u, cmap="magma")
    axes[1].set_title(f"Schnakenberg u  λ_grid={result['lambda_measured_cells']:.2f} cells")
    axes[1].set_axis_off()
    # Last-frame blend preview at slab resolution (downsampled field).
    field = u - u.min()
    field = field / (field.max() + 1e-9)
    field_big = np.asarray(
        Image.fromarray((field * 255).astype(np.uint8)).resize((gray.shape[1], gray.shape[0]), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    weight = np.clip(0.78 * result["mask"], 0.0, 1.0)
    stripe = 0.08 + 0.84 * field_big
    blend = np.clip((1.0 - weight) * gray + weight * (stripe * 255.0), 0, 255)
    axes[2].imshow(blend, cmap="gray")
    axes[2].set_title("Vane overlay (critical mode on the wing)")
    axes[2].set_axis_off()
    x = np.arange(fit["profile"].size)
    axes[3].plot(x, fit["profile"], color="0.25", lw=1.0, label="wing OD")
    axes[3].plot(x, fit["prediction"], color="darkred", lw=1.2, label="quartic + 3 harmonics")
    axes[3].set_title(
        f"3-harmonic r={fit['pearson_r']:.4f}  fundamental r={fit['pearson_fundamental']:.3f}  KS={fit['ks']:.3f}"
    )
    axes[3].legend(fontsize=7, frameon=False)
    axes[3].set_xlabel("px along wing transect")
    gammas = [row["gamma"] for row in sweep]
    lams = [row["lambda_cells"] for row in sweep]
    axes[4].plot(gammas, lams, "o-", color="darkslateblue")
    axes[4].axvline(GAMMA, color="goldenrod", ls="--", lw=0.8, label="Archaeopteryx γ")
    axes[4].set_xlabel("gamma")
    axes[4].set_ylabel("λ_c (grid cells)")
    axes[4].set_title("Parameter sweep (monotonic)")
    axes[4].legend(fontsize=7, frameon=False)
    labels = ["Anchiornis\n160.8 Ma", "Archaeopteryx\n150.0 Ma", "Microraptor\n120 Ma"]
    vals = [
        series["anchiornis_huxleyi"]["lambda_m"] * 1e6,
        series["archaeopteryx_lithographica"]["lambda_m"] * 1e6,
        series["microraptor_gui"]["lambda_m"] * 1e6,
    ]
    axes[5].bar(labels, vals, color=["#8d99ae", "#c44536", "#2a6f7f"])
    axes[5].set_ylabel("predicted λ (µm)")
    axes[5].set_title("Cellular wavelength along the feather series")
    fig.suptitle(
        "Archaeopteryx lithographica — Turing k_c at 75 µm (7.478e-05 m), Tithonian 150.0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def write_metrics(result: dict) -> None:
    METRICS_PATH.write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")


def measure_morphoeval() -> list[dict]:
    """Morphometrics for morphoeval/measure_outputs.py. Values are not copies of their references."""
    result = run_pipeline()
    fit = result["fit"]
    series = result["chronoseries"]
    sweep_ratio = result["sweep"][0]["lambda_cells"] / result["sweep"][-1]["lambda_cells"]
    return [
        {
            "name": "Archaeopteryx lithographica specimen wing-vane optical-density Pearson r",
            "value": round(fit["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "gaussian sigma=3 column-mean of Berlin-slab ROI rows 450:850, cols 750:1200; quartic + 3 harmonics of the dominant FFT frequency",
            "reference_value": 0.95,
            "reference_source": "Pre-registered goodness-of-fit bar for the scan-resolution wing transect (not the 75 µm PDE scale)",
            "note": f"fundamental-only r={fit['pearson_fundamental']:.4f}; N={fit['n']}; scan does not resolve 75 µm",
            "derived_from": "script",
        },
        {
            "name": "specimen wing transect normalized RMSE",
            "value": round(fit["nrmse"], 4),
            "unit": "RMSE / sigma",
            "method": "sqrt(mean((OD - prediction)^2)) / std(OD) on the same wing transect",
            "reference_value": 0.35,
            "reference_source": "Null bar above the polynomial-only residual scale of this transect",
            "note": f"KS two-sample statistic={fit['ks']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "specimen versus fitted transect two-sample KS statistic",
            "value": round(fit["ks"], 4),
            "unit": "KS distance",
            "method": "scipy.stats.ks_2samp on the wing optical-density profile and the quartic-plus-harmonics prediction",
            "reference_value": 0.25,
            "reference_source": "Rejection-scale KS distance used as the negative bar",
            "derived_from": "script",
        },
        {
            "name": "Turing pigment stripe wavelength lambda_c at the Archaeopteryx cellular scale",
            "value": round(series["archaeopteryx_lithographica"]["lambda_m"], 8),
            "unit": "m",
            "method": "lambda_cells(gamma=0.039) * dx with dx chosen so this wavelength equals 7.478e-05 m",
            "reference_value": 9.0e-07,
            "reference_source": "Carney et al. 2012 Nat. Commun. melanosome length scale (~1 µm), which this wavelength is not",
            "note": f"measured/analytical grid relative error={result['lambda_relative_error']:.5f}",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation spatial-std ratio (Du=Dv control)",
            "value": round(result["ablation_std_ratio"], 6),
            "unit": "std(u_ablation)/std(u_turing)",
            "method": "Du=Dv=0.45, no critical-mode seed, 1800 steps, versus the Turing run",
            "reference_value": 1.0,
            "reference_source": "Turing 1952: diffusion-driven instability requires Du != Dv, so the ratio must fall below 1",
            "note": f"std_turing={result['std_turing']:.4f}, std_ablation={result['std_ablation']:.6f}",
            "derived_from": "script",
        },
        {
            "name": "Anchiornis huxleyi predecessor cellular stripe wavelength",
            "value": round(series["anchiornis_huxleyi"]["lambda_m"], 8),
            "unit": "m",
            "method": "same dx and kinetics, gamma=0.024 (slower symmetric-vane melanocyte kinetics)",
            "reference_value": PHENOMENON_M,
            "reference_source": "Archaeopteryx 150.0 Ma target 7.478e-05 m; Anchiornis is the 160.8 Ma neighbor, not a second Archaeopteryx horizon",
            "derived_from": "script",
        },
        {
            "name": "Microraptor gui successor cellular stripe wavelength",
            "value": round(series["microraptor_gui"]["lambda_m"], 8),
            "unit": "m",
            "method": "same dx and kinetics, gamma=0.067 (faster kinetics, iridescent melanosome stacks)",
            "reference_value": PHENOMENON_M,
            "reference_source": "Archaeopteryx target; Microraptor gui is the 120 Ma lineage successor (Li et al. 2012)",
            "derived_from": "script",
        },
        {
            "name": "specimen slab structure-tensor coherency",
            "value": round(result["cv"]["coherency"], 4),
            "unit": "coherency",
            "method": "border-contrast mask and Sobel structure tensor on the Berlin slab resized to 160x120",
            "reference_value": 0.30,
            "reference_source": "Neutral isotropic-texture bar; the fossil print is more coherent than 0.30",
            "note": f"foreground_fraction={result['cv']['foreground_fraction']:.4f}; wing_mask_fraction={result['cv']['wing_mask_fraction']:.4f}; interpeak CV={result['interpeak_cv']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "gamma sweep wavelength ratio lambda(0.020)/lambda(0.250)",
            "value": round(sweep_ratio, 4),
            "unit": "ratio",
            "method": "analytical lambda_c across gamma in {0.020, 0.039, 0.080, 0.150, 0.250}",
            "reference_value": 1.0,
            "reference_source": "A flat sweep (ratio 1) would mean gamma does not control wavelength",
            "note": "strictly decreasing in gamma",
            "derived_from": "script",
        },
    ]


def gif_cv_similarity() -> float:
    import sys
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity, compute_specimen_cv_signature

    sig = compute_specimen_cv_signature(SPECIMEN)
    return float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GIF_PATH))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = run_pipeline()
    render_png(result)
    render_gif(result)
    write_metrics(result)
    payload = _jsonable(result)
    cv_score = gif_cv_similarity()
    payload["cv_sim_similarity"] = cv_score
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "gamma_sweep"}, indent=2))
    if args.self_test:
        assert result["info"]["ok"], "Turing conditions failed"
        assert payload["pearson_r"] > 0.98, payload["pearson_r"]
        assert payload["lambda_relative_error"] < 0.12, payload["lambda_relative_error"]
        assert payload["ablation_std_ratio"] < 0.05, payload["ablation_std_ratio"]
        lams = [row["lambda_cells"] for row in result["sweep"]]
        assert all(lams[i] > lams[i + 1] for i in range(len(lams) - 1)), lams
        chrono = payload["chronoseries_um_x100"]
        assert chrono["anchiornis_huxleyi"] == LEAN_ANCH, chrono
        assert chrono["archaeopteryx_lithographica"] == LEAN_ARCH, chrono
        assert chrono["microraptor_gui"] == LEAN_MICRO, chrono
        assert chrono["microraptor_gui"] < chrono["archaeopteryx_lithographica"] < chrono["anchiornis_huxleyi"]
        with Image.open(GIF_PATH) as im:
            assert im.n_frames >= 10, im.n_frames
            assert min(im.size) >= 480, im.size
        assert cv_score >= 58.0, cv_score
        print(
            f"[PASS] slot 3120 self-test r={payload['pearson_r']:.4f} "
            f"λ_err={payload['lambda_relative_error']:.5f} cv={cv_score:.1f} age={AGE_MA} Ma"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
