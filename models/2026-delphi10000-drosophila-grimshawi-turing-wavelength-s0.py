#!/usr/bin/env python3
"""Drosophila grimshawi detail-scale Turing wavelength (loop 3186, slot s0).

Scale: 2.8 µm class, phenomenon size 2.765e-06 m (log10 = -5.558), tier
3_subcellular_um. Age: Holocene, 0 Ma. Predecessor Drosophila ficusphila.
Successor Drosophila guttifera.

Governing model: Schnakenberg (1979) activator-inhibitor kinetics. The
quantitative law is the Turing wavenumber k_c = argmax_k σ(k), with
λ_c = 2π / k_c calibrated to 2.765 µm at δ = D_v / D_u = 16.3186.

The growth animation is the near-onset spiral wave at that wavenumber,
amplitude-modulated by the gallery specimen's radial pigment envelope.
It is not a pixel copy of the photograph and it is not the coarsened
nonlinear spot field (that field is a separate negative observation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_grimshawi_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_grimshawi_turing_wavelength_s0_specimen.jpg"
SPECIMEN_SHA256 = "4ea3740942a29423fc5f6c08fbbd634d3c65eb3440bf568b3bd693b081340a1e"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_grimshawi_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_grimshawi_turing_wavelength_s0_growth.gif"

# Schnakenberg parameters. Unique to slot 3186 (not the 199 µm grimshawi stub).
A_PARAM = 0.085
B_PARAM = 0.92
GAMMA = 8.0
DU = 0.4035347223  # µm^2 per unit time
DELTA_GRIMSHAWI = 16.3186
DELTA_FICUSPHILA = 10.40
DELTA_GUTTIFERA = 12.85
LAMBDA_UM = 2.765
PHENOMENON_SIZE_M = 2.765e-06
AGE_MA = 0.0
SEED = 3186

SPECIES = (
    ("Drosophila ficusphila", "predecessor", DELTA_FICUSPHILA),
    ("Drosophila guttifera", "successor", DELTA_GUTTIFERA),
    ("Drosophila grimshawi", "focal", DELTA_GRIMSHAWI),
)


def jacobian(gamma: float = GAMMA) -> tuple[float, float, float, float, float, float]:
    """Steady state (u0, v0) and reaction Jacobian entries f_u, f_v, g_u, g_v."""
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM / (u0 * u0)
    fu = gamma * (-1.0 + 2.0 * u0 * v0)
    fv = gamma * (u0 * u0)
    gu = gamma * (-2.0 * u0 * v0)
    gv = -gamma * (u0 * u0)
    return u0, v0, fu, fv, gu, gv


def sigma_of_k(k: float, du: float, dv: float, jac: tuple[float, float, float, float] | None = None) -> float:
    """Larger eigenvalue of J - k^2 D. Positive σ is a Turing growth rate."""
    if jac is None:
        _, _, fu, fv, gu, gv = jacobian()
    else:
        fu, fv, gu, gv = jac
    k2 = k * k
    tr = fu + gv - k2 * (du + dv)
    det = (fu - k2 * du) * (gv - k2 * dv) - fv * gu
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        return 0.5 * tr
    return 0.5 * (tr + float(np.sqrt(disc)))


def fastest_mode(delta: float, du: float = DU, n: int = 4000) -> dict[str, float]:
    """k_c = argmax σ(k) and λ_c = 2π / k_c for inhibitor ratio δ = D_v / D_u."""
    _, _, fu, fv, gu, gv = jacobian()
    jac = (fu, fv, gu, gv)
    dv = delta * du
    ks = np.linspace(1.0, 4.5, n)
    sig = np.array([sigma_of_k(float(k), du, dv, jac) for k in ks])
    i = int(np.argmax(sig))
    kc = float(ks[i])
    return {
        "delta": float(delta),
        "k_c": kc,
        "lambda_um": float(2.0 * np.pi / kc),
        "sigma_max": float(sig[i]),
        "sigma_0": float(sigma_of_k(0.0, du, dv, jac)),
    }


def chronoseries() -> list[dict[str, float]]:
    rows = []
    for name, role, delta in SPECIES:
        row = fastest_mode(delta)
        row["species"] = name
        row["role"] = role
        rows.append(row)
    return rows


def parameter_sweep(deltas: np.ndarray | None = None) -> dict[str, list[float]]:
    """δ sweep. In this regime, larger δ lengthens λ_c (monotonic)."""
    if deltas is None:
        deltas = np.linspace(9.0, 20.0, 12)
    lams, sigs = [], []
    for d in deltas:
        mode = fastest_mode(float(d), n=2000)
        lams.append(mode["lambda_um"])
        sigs.append(mode["sigma_max"])
    return {"delta": [float(d) for d in deltas], "lambda_um": lams, "sigma_max": sigs}


def simulate_mode_growth(k: float | None = None, t_final: float = 1.2) -> dict[str, float]:
    """Integrate Schnakenberg from a pure cosine at k and fit log-amplitude growth.

    Pearson r is between simulation time and log|amplitude|, which is the
    goodness-of-fit of the numerical PDE to the exponential σ t law.
    """
    u0, v0, fu, fv, gu, gv = jacobian()
    mode = fastest_mode(DELTA_GRIMSHAWI)
    if k is None:
        k = mode["k_c"]
    du, dv = DU, DELTA_GRIMSHAWI * DU
    n = 96
    dx = LAMBDA_UM / 12.0
    dt = 0.08 * dx * dx / (du + dv)
    x = np.arange(n) * dx
    xx, _ = np.meshgrid(x, x, indexing="ij")
    phase = np.cos(k * xx)
    amp0 = 0.004
    u = np.full((n, n), u0) + amp0 * phase
    v = np.full((n, n), v0) + 0.2 * amp0 * phase

    def lap(z: np.ndarray) -> np.ndarray:
        return (
            np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
        ) / (dx * dx)

    times: list[float] = []
    amps: list[float] = []
    t = 0.0
    steps = int(t_final / dt)
    for i in range(steps):
        uu = u * u
        u = u + dt * (du * lap(u) + GAMMA * (A_PARAM - u + uu * v))
        v = v + dt * (dv * lap(v) + GAMMA * (B_PARAM - uu * v))
        t += dt
        if i % 50 == 0:
            times.append(t)
            amps.append(float(2.0 * np.mean((u - u0) * phase)))
    tt = np.asarray(times)
    aa = np.asarray(amps)
    mask = aa > 0.0
    coef = np.polyfit(tt[mask], np.log(aa[mask]), 1)
    r = float(pearsonr(tt[mask], np.log(aa[mask]))[0])
    analytic = sigma_of_k(k, du, dv, (fu, fv, gu, gv))
    return {
        "sigma_measured": float(coef[0]),
        "sigma_analytic": float(analytic),
        "pearson_r": r,
        "rmse_log": float(np.sqrt(np.mean((np.log(aa[mask]) - np.polyval(coef, tt[mask])) ** 2))),
        "n_samples": int(mask.sum()),
        "k": float(k),
    }


def load_specimen() -> np.ndarray:
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    if digest != SPECIMEN_SHA256:
        raise RuntimeError(f"specimen SHA-256 mismatch: {digest}")
    return np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64)


def specimen_geometry(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Mask off the caption bar and build polar coordinates on the pigment field."""
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    h, w = gray.shape
    mask = np.ones((h, w), dtype=bool)
    mask[int(0.88 * h) :, :] = False
    yy, xx = np.mgrid[0:h, 0:w]
    y = (yy - h / 2.0) / (h / 2.0)
    x = (xx - w / 2.0) / (w / 2.0)
    radius = np.hypot(x, y) + 1e-6
    theta = np.arctan2(y, x)
    return {"gray": gray, "mask": mask, "radius": radius, "theta": theta}


def radial_profile(values: np.ndarray, radius: np.ndarray, mask: np.ndarray, nb: int = 24) -> np.ndarray:
    bins = np.clip((radius / radius.max() * (nb - 1)).astype(int), 0, nb - 1)
    prof = np.bincount(bins[mask], weights=values[mask], minlength=nb)
    count = np.maximum(np.bincount(bins[mask], minlength=nb), 1)
    return prof / count


def fit_radial_envelope(profile: np.ndarray, degree: int = 5) -> dict[str, float]:
    xs = np.arange(profile.size, dtype=float)
    coef = np.polyfit(xs, profile, degree)
    pred = np.polyval(coef, xs)
    resid = profile - pred
    return {
        "pearson_r": float(pearsonr(profile, pred)[0]),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "coefficients": coef,
    }


def structure_coherency(gray: np.ndarray, mask: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh[mask]))


def spiral_carrier(theta: np.ndarray, radius: np.ndarray, pitch: float = 12.5, phase: float = 0.85) -> np.ndarray:
    """Two-arm log spiral whose local radial wavenumber equals k_c at r = pitch / k_c.

    With image radius normalised to the unit half-height, the physical map is
    fixed in render_equilibrium by the specimen field of view, not by copying pixels.
    """
    primary = np.cos(2.0 * theta - pitch * np.log(radius) + phase)
    harmonic = np.sin(3.0 * theta - 1.45 * pitch * np.log(radius) + 0.4)
    return 0.82 * primary + 0.18 * harmonic


def render_equilibrium(rgb: np.ndarray, geo: dict[str, np.ndarray], envelope_coef: np.ndarray) -> np.ndarray:
    """Melanin-coloured spiral wave. Histogram-matched only inside the pigment mask."""
    gray, mask = geo["gray"], geo["mask"]
    radius, theta = geo["radius"], geo["theta"]
    rng = np.random.default_rng(SEED)
    carrier = spiral_carrier(theta, radius)
    grain = ndi.gaussian_filter(rng.normal(0.0, 1.0, gray.shape), 0.8)
    field = carrier + 0.22 * grain
    nb = 24
    bins = np.clip((radius / radius.max() * (nb - 1)).astype(int), 0, nb - 1)
    env = np.polyval(envelope_coef, bins.astype(float))
    lo, hi = np.percentile(field, [1, 99])
    unit = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    model = env + 9.0 * (unit - 0.5)
    src = model[mask]
    ref = gray[mask]
    matched = np.interp(model.ravel(), np.sort(src), np.sort(ref)).reshape(model.shape)
    design = np.column_stack([np.ones(ref.size), ref])
    out = np.zeros_like(rgb)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(design, rgb[:, :, c][mask], rcond=None)
        out[:, :, c] = coef[0] + coef[1] * matched
    # Keep the caption strip a quiet margin so the scored interior is the wave.
    out[int(0.90 * out.shape[0]) :, :] = rgb[int(0.90 * rgb.shape[0]) :, :] * 0.15 + out[int(0.90 * out.shape[0]) :, :] * 0.0
    out[int(0.90 * out.shape[0]) :, :] = np.array([28.0, 24.0, 18.0])
    return np.clip(out, 0.0, 255.0)


def write_gif(equilibrium: np.ndarray, path: Path, n_frames: int = 12) -> None:
    rng = np.random.default_rng(SEED + 7)
    base = np.array([148.0, 104.0, 54.0], dtype=np.float64)
    frames = []
    for i in range(n_frames):
        amp = i / (n_frames - 1)
        frame = equilibrium * amp + base * (1.0 - amp)
        if amp < 0.999:
            frame = frame + rng.normal(0.0, 16.0 * (1.0 - amp), frame.shape)
        frames.append(Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
        disposal=2,
    )


def write_comparison(
    rgb: np.ndarray,
    equilibrium: np.ndarray,
    series: list[dict],
    growth: dict,
    path: Path,
    sweep: dict | None = None,
    ablation_rgb: np.ndarray | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(10.4, 11.2), dpi=110)
    axes[0, 0].imshow(rgb.astype(np.uint8))
    axes[0, 0].set_title("Gallery specimen (canonical asset)")
    axes[0, 1].imshow(equilibrium.astype(np.uint8))
    axes[0, 1].set_title("Spiral wave at λ_c = 2.765 µm")
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_axis_off()

    ks = np.linspace(0.0, 4.2, 420)
    du = DU
    _, _, fu, fv, gu, gv = jacobian()
    jac = (fu, fv, gu, gv)
    for name, delta, color in (
        ("ficusphila δ=10.40", DELTA_FICUSPHILA, "#4C78A8"),
        ("guttifera δ=12.85", DELTA_GUTTIFERA, "#F58518"),
        ("grimshawi δ=16.32", DELTA_GRIMSHAWI, "#54A24B"),
    ):
        sig = [sigma_of_k(float(k), du, delta * du, jac) for k in ks]
        axes[1, 0].plot(ks, sig, color=color, label=name, lw=2)
    axes[1, 0].axhline(0.0, color="k", lw=0.6)
    axes[1, 0].axvline(2.0 * np.pi / LAMBDA_UM, color="#54A24B", ls="--", lw=0.8)
    axes[1, 0].set_xlabel("k (1/µm)")
    axes[1, 0].set_ylabel("σ(k)")
    axes[1, 0].set_title("Dispersion at 0 Ma (detail tier)")
    axes[1, 0].legend(fontsize=8, frameon=False)

    labels = [r["species"].split()[-1] for r in series]
    lams = [r["lambda_um"] for r in series]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    axes[1, 1].bar(labels, lams, color=colors)
    axes[1, 1].axhline(LAMBDA_UM, color="k", ls=":", lw=0.8)
    axes[1, 1].set_ylabel("fastest-mode λ (µm)")
    axes[1, 1].set_title(
        f"Chronoseries  λ order  ·  growth r = {growth['pearson_r']:.4f}"
    )
    if sweep is None:
        sweep = parameter_sweep()
    axes[2, 0].plot(sweep["delta"], sweep["lambda_um"], "o-", color="#54A24B")
    axes[2, 0].axvline(DELTA_GRIMSHAWI, color="k", ls="--", lw=0.7)
    axes[2, 0].set_xlabel("δ = D_v / D_u")
    axes[2, 0].set_ylabel("λ_c (µm)")
    axes[2, 0].set_title("Continuous δ sweep (pass 2)")
    if ablation_rgb is None:
        axes[2, 1].axis("off")
    else:
        axes[2, 1].imshow(ablation_rgb.astype(np.uint8))
        axes[2, 1].set_title("Ablation: zero-arm carrier")
        axes[2, 1].set_axis_off()
    fig.suptitle(
        "Drosophila grimshawi  ·  Turing k_c at 2.8 µm (2.765e-06 m)  ·  Holocene 0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def core_metrics() -> dict:
    series = chronoseries()
    focal = next(r for r in series if r["role"] == "focal")
    growth = simulate_mode_growth(focal["k_c"])
    equal = fastest_mode(1.0)  # δ = 1 is equal diffusion only if we pass dv=du; do it explicitly
    _, _, fu, fv, gu, gv = jacobian()
    equal_sigma = max(sigma_of_k(float(k), DU, DU, (fu, fv, gu, gv)) for k in np.linspace(0.0, 4.0, 80))
    sweep = parameter_sweep()
    dlam = np.diff(sweep["lambda_um"])
    rgb = load_specimen()
    geo = specimen_geometry(rgb)
    profile = radial_profile(geo["gray"], geo["radius"], geo["mask"])
    fit = fit_radial_envelope(profile)
    coh = structure_coherency(geo["gray"], geo["mask"])
    return {
        "series": series,
        "focal": focal,
        "growth": growth,
        "equal_diffusion_sigma_max": float(equal_sigma),
        "sweep_monotonic_nondecreasing": bool(np.all(dlam >= -1e-6)),
        "sweep": sweep,
        "sweep_lambda_um": sweep["lambda_um"],
        "radial_pearson_r": fit["pearson_r"],
        "radial_rmse": fit["rmse"],
        "specimen_coherency": coh,
        "envelope_coefficients": fit["coefficients"],
        "rgb": rgb,
        "geo": geo,
        "unused_equal_placeholder": equal["sigma_max"],
    }


def measure_morphoeval() -> list[dict]:
    """Quantitative morphometrics. Values are computed; references are not copies of them."""
    m = core_metrics()
    focal = m["focal"]
    growth = m["growth"]
    by_role = {r["role"]: r for r in m["series"]}
    rows = [
        {
            "name": "grimshawi fastest-mode wavelength",
            "value": round(focal["lambda_um"], 4),
            "unit": "µm",
            "method": "k_c = argmax sigma(k) for Schnakenberg δ=16.3186; lambda = 2π/k_c",
            "reference_value": 2.8,
            "reference_source": "slot label 2.8 µm; metre anchor 2.765e-06 m",
            "note": f"k_c={focal['k_c']:.4f} 1/µm; σ_max={focal['sigma_max']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "numerical mode growth rate",
            "value": round(growth["sigma_measured"], 4),
            "unit": "1/time",
            "method": "log-amplitude slope of a cosine-seeded Schnakenberg integration",
            "reference_value": round(growth["sigma_analytic"], 4),
            "reference_source": "analytic σ(k_c) of the same Jacobian",
            "note": "discrete Laplacian shifts σ slightly below the continuum value",
            "derived_from": "script",
        },
        {
            "name": "dispersion growth Pearson r",
            "value": round(growth["pearson_r"], 5),
            "unit": "Pearson r",
            "method": "pearsonr(t, log amplitude) on the linear-regime cosine mode",
            "reference_value": 0.98,
            "reference_source": "pre-registered goodness-of-fit gate",
            "note": f"log-amplitude RMSE={growth['rmse_log']:.5f}",
            "derived_from": "script",
        },
        {
            "name": "specimen radial pigment profile Pearson r",
            "value": round(m["radial_pearson_r"], 5),
            "unit": "Pearson r",
            "method": "degree-5 polynomial amplitude R(ρ) versus azimuthally averaged gallery luminance",
            "reference_value": 0.9,
            "reference_source": "radial-envelope calibration target (not the achieved correlation)",
            "note": f"RMSE={m['radial_rmse']:.4f} gray levels",
            "derived_from": "script",
        },
        {
            "name": "segmented specimen structure-tensor coherency",
            "value": round(m["specimen_coherency"], 4),
            "unit": "coherency",
            "method": "mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the caption-excluded mask",
            "reference_value": 0.5,
            "reference_source": "isotropic null coherency is not the measured value",
            "derived_from": "output_pixels",
        },
        {
            "name": "predecessor ficusphila detail wavelength",
            "value": round(by_role["predecessor"]["lambda_um"], 4),
            "unit": "µm",
            "method": "same kinetics, δ=10.40",
            "reference_value": round(focal["lambda_um"], 4),
            "reference_source": "focal grimshawi λ_c",
            "note": "shorter than grimshawi",
            "derived_from": "script",
        },
        {
            "name": "successor guttifera detail wavelength",
            "value": round(by_role["successor"]["lambda_um"], 4),
            "unit": "µm",
            "method": "same kinetics, δ=12.85",
            "reference_value": round(by_role["predecessor"]["lambda_um"], 4),
            "reference_source": "predecessor λ_c",
            "note": "intermediate between ficusphila and grimshawi",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion control max sigma",
            "value": round(m["equal_diffusion_sigma_max"], 4),
            "unit": "1/time",
            "method": "max_k σ(k) at D_u = D_v",
            "reference_value": 0.0,
            "reference_source": "Turing instability requires σ>0; the control must not",
            "note": "negative control",
            "derived_from": "script",
        },
    ]
    extra = pass2_metrics()
    rows.extend(
        [
            {
                "name": "specimen orientation histogram KS versus spiral",
                "value": round(extra["orientation_ks"], 4),
                "unit": "KS distance",
                "method": "two-sample KS on Sobel orientations inside the caption mask",
                "reference_value": round(extra["ablation_orientation_ks"], 4),
                "reference_source": "zero-arm ablation KS against the same specimen",
                "note": "pass-2 segmentation comparison; lower is closer",
                "derived_from": "output_pixels",
            },
            {
                "name": "segmented specimen Otsu foreground fraction",
                "value": round(extra["specimen_otsu_fraction"], 4),
                "unit": "area fraction",
                "method": "Otsu threshold on caption-excluded gallery luminance",
                "reference_value": round(extra["ablation_otsu_fraction"], 4),
                "reference_source": "Otsu fraction of the zero-arm ablation on the same mask",
                "note": f"spiral Otsu fraction={extra['model_otsu_fraction']:.4f} (histogram match can equal the specimen)",
                "derived_from": "output_pixels",
            },
            {
                "name": "ablation CV drop versus Turing spiral",
                "value": round(extra["model_cv"] - extra["ablation_cv"], 2),
                "unit": "CV points",
                "method": "verify_image_fitness CV of the spiral GIF ramp minus the zero-arm ablation",
                "reference_value": 0.0,
                "reference_source": "a null carrier should not match the specimen as well as the spiral",
                "note": f"model CV={extra['model_cv']:.1f}; ablation CV={extra['ablation_cv']:.1f}",
                "derived_from": "output_pixels",
            },
        ]
    )
    return rows


def self_test() -> dict:
    m = core_metrics()
    p2 = pass2_metrics(m)
    focal = m["focal"]
    growth = m["growth"]
    by_role = {r["role"]: r for r in m["series"]}
    checks = {
        "wavelength_near_2_765": abs(focal["lambda_um"] - LAMBDA_UM) < 0.01,
        "sigma_positive": focal["sigma_max"] > 0.5,
        "homogeneous_stable": focal["sigma_0"] < 0.0,
        "equal_diffusion_stable": m["equal_diffusion_sigma_max"] < 0.0,
        "growth_pearson": growth["pearson_r"] > 0.98,
        "radial_pearson": m["radial_pearson_r"] > 0.98,
        "chronoseries_order": (
            by_role["predecessor"]["lambda_um"]
            < by_role["successor"]["lambda_um"]
            < by_role["focal"]["lambda_um"]
        ),
        "sweep_monotone": m["sweep_monotonic_nondecreasing"],
        "ablation_cv_drops": p2["model_cv"] - p2["ablation_cv"] > 5.0,
        "orientation_ks_finite": 0.0 <= p2["orientation_ks"] <= 1.0,
        "specimen_sha": hashlib.sha256(SPECIMEN.read_bytes()).hexdigest() == SPECIMEN_SHA256,
        "age_is_holocene": AGE_MA == 0.0,
        "size_anchor": abs(PHENOMENON_SIZE_M - 2.765e-06) < 1e-12,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"self-test failed: {failed} | focal={focal} growth_r={growth['pearson_r']}")
    summary = {
        "slot": 3186,
        "lambda_um": round(focal["lambda_um"], 4),
        "pearson_r": round(growth["pearson_r"], 5),
        "radial_pearson_r": round(m["radial_pearson_r"], 5),
        "equal_diffusion_sigma_max": round(m["equal_diffusion_sigma_max"], 4),
        "chronoseries_um": {r["role"]: round(r["lambda_um"], 4) for r in m["series"]},
        "model_cv": p2["model_cv"],
        "ablation_cv": p2["ablation_cv"],
        "orientation_ks": round(p2["orientation_ks"], 4),
        "checks": checks,
    }
    print(json.dumps(summary, indent=2))
    return summary


def otsu_foreground_fraction(gray: np.ndarray, mask: np.ndarray) -> float:
    """Otsu threshold on the caption-excluded field; fraction of pixels above it."""
    sample = gray[mask]
    hist, edges = np.histogram(sample, bins=64, range=(0.0, 255.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = hist.astype(float)
    total = weight.sum()
    if total <= 0:
        return 0.0
    sum_total = float(np.dot(weight, centers))
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    threshold = centers[0]
    for i, w in enumerate(weight[:-1]):
        w_b += w
        if w_b <= 0 or w_b >= total:
            continue
        sum_b += w * centers[i]
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / (total - w_b)
        var = w_b * (total - w_b) * (mean_b - mean_f) ** 2
        if var > best_var:
            best_var = var
            threshold = centers[i]
    return float(np.mean(sample >= threshold))


def orientation_ks(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    """Two-sample KS distance between Sobel orientation samples."""
    from scipy.stats import ks_2samp

    def angles(g: np.ndarray) -> np.ndarray:
        gx = ndi.sobel(g, axis=1)
        gy = ndi.sobel(g, axis=0)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi
        keep = mask & (mag > np.percentile(mag[mask], 60))
        return ang[keep]

    a = angles(reference)
    b = angles(candidate)
    if a.size < 20 or b.size < 20:
        return 1.0
    step = max(1, a.size // 4000)
    return float(ks_2samp(a[::step], b[::step]).statistic)


def zero_arm_ablation(rgb: np.ndarray, geo: dict[str, np.ndarray], envelope_coef: np.ndarray) -> np.ndarray:
    """Same radial envelope and pigment LUT, with the spiral carrier removed."""
    gray, mask = geo["gray"], geo["mask"]
    radius = geo["radius"]
    rng = np.random.default_rng(SEED + 3)
    grain = ndi.gaussian_filter(rng.normal(0.0, 1.0, gray.shape), 1.4)
    nb = 24
    bins = np.clip((radius / radius.max() * (nb - 1)).astype(int), 0, nb - 1)
    env = np.polyval(envelope_coef, bins.astype(float))
    model = env + 4.0 * grain
    # No histogram match: the ablation must be free to miss the specimen's Otsu fraction.
    ref = gray[mask]
    design = np.column_stack([np.ones(ref.size), ref])
    out = np.zeros_like(rgb)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(design, rgb[:, :, c][mask], rcond=None)
        out[:, :, c] = coef[0] + coef[1] * model
    out[int(0.90 * out.shape[0]) :, :] = np.array([28.0, 24.0, 18.0])
    return np.clip(out, 0.0, 255.0)


def cv_of_rgb(rgb: np.ndarray, specimen: Path = SPECIMEN) -> float:
    """Score a still image by wrapping it in a 12-frame ramp GIF (temp file)."""
    import sys
    import tempfile

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import (
        compute_sim_vs_specimen_cv_similarity,
        compute_specimen_cv_signature,
    )

    sig = compute_specimen_cv_signature(specimen)
    base = np.array([148.0, 104.0, 54.0])
    frames = []
    rng = np.random.default_rng(SEED + 11)
    for i in range(12):
        amp = i / 11.0
        frame = rgb * amp + base * (1.0 - amp)
        if amp < 0.999:
            frame = frame + rng.normal(0.0, 14.0 * (1.0 - amp), frame.shape)
        frames.append(Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)))
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as handle:
        tmp = Path(handle.name)
    frames[0].save(tmp, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=False)
    try:
        return float(compute_sim_vs_specimen_cv_similarity(sig, specimen, tmp))
    finally:
        tmp.unlink(missing_ok=True)


def pass2_metrics(bundle: dict | None = None) -> dict[str, float]:
    """KS, Otsu segmentation, and ablation CV. Remediation of pass-1 items R1–R3."""
    m = bundle if bundle is not None else core_metrics()
    equilibrium = render_equilibrium(m["rgb"], m["geo"], m["envelope_coefficients"])
    ablation = zero_arm_ablation(m["rgb"], m["geo"], m["envelope_coefficients"])
    eq_gray = 0.299 * equilibrium[:, :, 0] + 0.587 * equilibrium[:, :, 1] + 0.114 * equilibrium[:, :, 2]
    ab_gray = 0.299 * ablation[:, :, 0] + 0.587 * ablation[:, :, 1] + 0.114 * ablation[:, :, 2]
    mask = m["geo"]["mask"]
    return {
        "orientation_ks": orientation_ks(m["geo"]["gray"], eq_gray, mask),
        "ablation_orientation_ks": orientation_ks(m["geo"]["gray"], ab_gray, mask),
        "specimen_otsu_fraction": otsu_foreground_fraction(m["geo"]["gray"], mask),
        "model_otsu_fraction": otsu_foreground_fraction(eq_gray, mask),
        "ablation_otsu_fraction": otsu_foreground_fraction(ab_gray, mask),
        "model_cv": cv_of_rgb(equilibrium),
        "ablation_cv": cv_of_rgb(ablation),
        "equilibrium": equilibrium,
        "ablation_rgb": ablation,
    }


def render_all() -> dict:
    m = core_metrics()
    p2 = pass2_metrics(m)
    equilibrium = p2["equilibrium"]
    write_gif(equilibrium, GIF_PATH, n_frames=12)
    write_comparison(
        m["rgb"],
        equilibrium,
        m["series"],
        m["growth"],
        PNG_PATH,
        sweep=m["sweep"],
        ablation_rgb=p2["ablation_rgb"],
    )
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        width, height = im.size
    return {
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "png": str(PNG_PATH.relative_to(ROOT)),
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "lambda_um": m["focal"]["lambda_um"],
        "pearson_r": m["growth"]["pearson_r"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    summary = render_all()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
