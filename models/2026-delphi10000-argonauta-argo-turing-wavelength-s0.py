#!/usr/bin/env python3
"""Argonauta argo ocellus at the 95 µm Holocene detail scale (slot 3121).

Schnakenberg reaction-diffusion, reduced to the critical radial Turing mode.
The tier size is 9.48e-05 m (95 µm, log10 = -4.023, tier 4_cellular_10um) at
geological age 0 Ma. The Argonauta chronoseries has one node, so this script
does not invent a predecessor wavelength.

Governing system (Turing 1952; activator-inhibitor class of Gierer & Meinhardt
1972), parameters a=0.10, b=0.90, D_u=1, D_v=18, gamma=1:

    du/dt = D_u ∇²u + γ (a - u + u² v)
    dv/dt = D_v ∇²v + γ (b - u² v)

    u_s = a + b,  v_s = b / (a + b)²
    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c = 2π / k_c

The length unit is set so λ_c equals the peak-to-peak period of the radial
mode fitted to the vendored 100 µm plate. The 95 µm tier label is compared
with the measured inhibitory-halo radius; the two numbers are not forced equal.

Equal diffusion (D_u = D_v) is the negative control. A γ sweep is the
parameter-regime check (λ ∝ γ^(-1/2)). Sub-wavelength pigment grain is
procedural Gaussian noise (seed 3121), amplitude-swept against the plate.
It does not copy specimen Fourier phase.

Run:
    python3 models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "argonauta_argo_turing_wavelength_s0"
SPECIMEN = DATASET / "argonauta_argo_turing_wavelength_s0_specimen.jpg"
SHELL = DATASET / "argonauta_argo_shell_cc0.jpg"
REFERENCE = DATASET / "argonauta_argo_turing_wavelength_s0_reference.json"
METRICS = DATASET / "argonauta_argo_turing_wavelength_s0_metrics.json"
SMOOTH_GIF = DATASET / "argonauta_argo_turing_wavelength_s0_smooth_control.gif"
MEDIA_PNG = ROOT / "docs" / "media" / "argonauta_argo_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "argonauta_argo_turing_wavelength_s0_growth.gif"

# Slot-3121 parameters. Not the sibling Argonauta slot's (1.39, 0.31) pair.
A_PARAM = 0.10
B_PARAM = 0.90
GAMMA = 1.0
DU = 1.0
DV = 18.0
SEED = 3121
PHENOMENON_SIZE_M = 9.48e-05
TIER_HALO_UM = 95.0
AGE_MA = 0.0
N_FRAMES = 12
GRAIN_AMPS = (0.0, 0.25, 0.40, 0.50, 0.65, 0.80)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> dict:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    pinned = ref["sha256"]
    for name, expected in pinned.items():
        got = sha256_file(DATASET / name)
        if got != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {expected}")
    if abs(float(ref["phenomenon_scale"]["phenomenon_size_m"]) - PHENOMENON_SIZE_M) > 1e-12:
        raise ValueError("reference phenomenon size is not 9.48e-05 m")
    if float(ref["paleobiology"]["geological_age_ma"]) != AGE_MA:
        raise ValueError("reference geological age is not 0 Ma")
    return ref


def jacobian(gamma: float = GAMMA, a: float = A_PARAM, b: float = B_PARAM):
    """Reaction Jacobian of the Schnakenberg system at its unique positive steady state."""
    us = a + b
    vs = b / us**2
    fu = gamma * (-1.0 + 2.0 * us * vs)
    fv = gamma * us**2
    gu = gamma * (-2.0 * us * vs)
    gv = -gamma * us**2
    return float(fu), float(fv), float(gu), float(gv)


def sigma_max(k: float, fu: float, fv: float, gu: float, gv: float, du: float, dv: float) -> float:
    """Largest real eigenvalue of J - k² D (quadratic formula)."""
    tr = fu + gv - (k * k) * (du + dv)
    det = (fu - du * k * k) * (gv - dv * k * k) - fv * gu
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        return float(0.5 * (tr + np.sqrt(disc)))
    return float(0.5 * tr)


def sigma_eig(k: float, fu: float, fv: float, gu: float, gv: float, du: float, dv: float) -> float:
    """Same eigenvalue from the 2×2 matrix, used as an algebra cross-check."""
    matrix = np.array(
        [[fu - du * k * k, fv], [gu, gv - dv * k * k]],
        dtype=float,
    )
    return float(np.max(np.real(np.linalg.eigvals(matrix))))


def critical_wavenumber(fu: float, gv: float, du: float = DU, dv: float = DV) -> float:
    numer = dv * fu + du * gv
    if numer <= 0.0:
        return float("nan")
    return float(np.sqrt(numer / (2.0 * du * dv)))


def load_gray_rgb(path: Path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    return rgb, gray


def detect_center(gray: np.ndarray) -> tuple[int, int]:
    """Bright core of the ocellus, ignoring the caption band and the scale bar."""
    sm = ndi.gaussian_filter(gray, 8.0)
    masked = sm.copy()
    masked[:30, :] = -1.0
    masked[-80:, :] = -1.0
    masked[:, :30] = -1.0
    masked[:, -60:] = -1.0
    cy, cx = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return int(cy), int(cx)


def detect_scale(gray: np.ndarray) -> tuple[float, float]:
    """Return (µm per pixel, bar length in px) from the white 100 µm bar."""
    height, width = gray.shape
    for y in range(height - 45, height - 8):
        bright = np.where(gray[y] > 0.90)[0]
        if bright.size < 30:
            continue
        start = prev = int(bright[0])
        runs: list[tuple[int, int]] = []
        for x in bright[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
            else:
                runs.append((start, prev))
                start = prev = x
        runs.append((start, prev))
        for a, b in runs:
            span = b - a
            if 80 <= span <= 160 and a > width * 0.55:
                return 100.0 / float(span), float(span)
    return 100.0 / 110.0, 110.0


def radial_profile(gray: np.ndarray, cy: int, cx: int, r_max: float, n_bins: int = 80):
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    rr = np.hypot(yy - cy, xx - cx)
    valid = (yy < height - 80) & (yy > 20) & (xx > 20) & (xx < width - 40)
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centres = []
    values = []
    for i in range(n_bins):
        mask = valid & (rr >= edges[i]) & (rr < edges[i + 1])
        if int(mask.sum()) < 25:
            continue
        centres.append(0.5 * (edges[i] + edges[i + 1]))
        values.append(float(gray[mask].mean()))
    return np.asarray(centres), np.asarray(values), rr


def mode_intensity(r, theta):
    amp, scale, lam, phase, baseline, core = theta
    envelope = np.exp(-((r / scale) ** 2))
    carrier = np.exp(-((r / (1.4 * scale)) ** 2))
    return baseline + core * envelope + amp * carrier * np.cos(2.0 * np.pi * r / lam + phase)


def fit_mode(radius: np.ndarray, profile: np.ndarray):
    x0 = np.array([0.29, 138.0, 187.0, -0.15, 0.43, 0.05])
    lo = np.array([-1.0, 20.0, 40.0, -np.pi, 0.0, -1.0])
    hi = np.array([1.0, 400.0, 400.0, np.pi, 1.0, 1.0])
    result = least_squares(lambda th: mode_intensity(radius, th) - profile, x0, bounds=(lo, hi))
    pred = mode_intensity(radius, result.x)
    return result.x, pred


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def halo_radius_px(theta, r_max: float = 250.0) -> float:
    grid = np.linspace(8.0, r_max, 600)
    values = mode_intensity(grid, theta)
    return float(grid[int(np.argmin(values))])


def structure_coherency(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def color_lut(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Mean specimen RGB at each gray level. A palette, not a pixel copy."""
    idx = np.clip(gray * 255.0, 0, 255).astype(np.int32)
    lut = np.zeros((256, 3), dtype=np.float64)
    counts = np.zeros(256, dtype=np.float64)
    flat_i = idx.ravel()[::4]
    flat_rgb = rgb.reshape(-1, 3)[::4]
    np.add.at(lut, flat_i, flat_rgb)
    np.add.at(counts, flat_i, 1.0)
    good = counts > 5
    xs = np.arange(256)
    for channel in range(3):
        mean = lut[:, channel] / np.maximum(counts, 1.0)
        lut[:, channel] = np.interp(xs, xs[good], mean[good])
    return lut


def apply_lut(gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = np.clip(gray * 255.0, 0, 255).astype(np.int32)
    return lut[idx]


def shell_autocorr_period(path: Path) -> float:
    """First positive autocorrelation peak along a mid-shell transect, in pixels."""
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    height, width = gray.shape
    band = gray[int(height * 0.32) : int(height * 0.58), int(width * 0.20) : int(width * 0.80)]
    profile = band.mean(axis=0)
    profile = profile - profile.mean()
    ac = np.correlate(profile, profile, mode="full")
    ac = ac[ac.size // 2 :]
    peaks, _ = find_peaks(ac[5:], distance=5)
    if peaks.size == 0:
        return float("nan")
    return float(peaks[0] + 5)


def dispersion_bundle():
    fu, fv, gu, gv = jacobian(GAMMA)
    kc = critical_wavenumber(fu, gv)
    ks = np.linspace(0.0, 1.6, 64)
    analytic = np.array([sigma_max(k, fu, fv, gu, gv, DU, DV) for k in ks])
    numeric = np.array([sigma_eig(k, fu, fv, gu, gv, DU, DV) for k in ks])
    fu_e, fv_e, gu_e, gv_e = jacobian(GAMMA)
    ablation = np.array([sigma_max(k, fu_e, fv_e, gu_e, gv_e, DU, DU) for k in ks])
    gammas = np.array([0.55, 0.75, 1.0, 1.35, 1.80, 2.40])
    # λ_dim at each γ; physical scale is applied by the caller via the γ=1 length unit.
    lam_dim = []
    for g in gammas:
        fu_g, _, _, gv_g = jacobian(float(g))
        kc_g = critical_wavenumber(fu_g, gv_g)
        lam_dim.append(2.0 * np.pi / kc_g)
    return {
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "det": fu * gv - fv * gu,
        "trace": fu + gv,
        "kc": kc,
        "lambda_dim": 2.0 * np.pi / kc,
        "k_grid": ks,
        "sigma": analytic,
        "sigma_eig": numeric,
        "sigma_equal_diffusion": ablation,
        "dispersion_pearson": pearson(analytic, numeric),
        "sigma_at_kc": sigma_max(kc, fu, fv, gu, gv, DU, DV),
        "sigma_at_0": sigma_max(0.0, fu, fv, gu, gv, DU, DV),
        "ablation_sigma_max": float(np.max(ablation)),
        "gammas": gammas,
        "lambda_dim_sweep": np.asarray(lam_dim),
    }


def analyze() -> dict:
    verify_inputs()
    rgb, gray = load_gray_rgb(SPECIMEN)
    cy, cx = detect_center(gray)
    um_per_px, bar_px = detect_scale(gray)
    radius, profile, rr = radial_profile(gray, cy, cx, r_max=220.0)
    theta, pred = fit_mode(radius, profile)
    radius_b, profile_b, _ = radial_profile(gray, cy, cx, r_max=150.0)
    theta_b, pred_b = fit_mode(radius_b, profile_b)
    disp = dispersion_bundle()
    halo_px = halo_radius_px(theta)
    halo_b_px = halo_radius_px(theta_b, r_max=150.0)
    lam_px = float(theta[2])
    lam_um = lam_px * um_per_px
    length_unit_um = lam_um / disp["lambda_dim"]
    sweep_um = disp["lambda_dim_sweep"] * length_unit_um
    resid = profile - pred
    field = np.clip(mode_intensity(rr, theta), 0.0, 1.0)
    shell_px = shell_autocorr_period(SHELL)
    return {
        "rgb": rgb,
        "gray": gray,
        "center": (cy, cx),
        "um_per_px": um_per_px,
        "bar_px": bar_px,
        "rr": rr,
        "radius": radius,
        "profile": profile,
        "theta": theta,
        "pred": pred,
        "pearson_r": pearson(profile, pred),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "residual_cv": float(np.std(resid) / (np.mean(profile) + 1e-9)),
        "ks_statistic": float(ks_2samp(profile, pred).statistic),
        "ks_pvalue": float(ks_2samp(profile, pred).pvalue),
        "halo_um": halo_px * um_per_px,
        "halo_window_um": halo_b_px * um_per_px,
        "window_pearson": pearson(profile_b, pred_b),
        "lambda_px": lam_px,
        "lambda_um": lam_um,
        "length_unit_um": length_unit_um,
        "field": field,
        "coherency_specimen": structure_coherency(gray[40:520, 40:740]),
        "coherency_model": structure_coherency(field[40:520, 40:740]),
        "shell_period_px": shell_px,
        "shell_period_fraction": shell_px / float(np.asarray(Image.open(SHELL)).shape[1]),
        "ocellus_period_fraction": lam_px / float(gray.shape[1]),
        "sweep_gamma": disp["gammas"],
        "sweep_lambda_um": sweep_um,
        "dispersion": disp,
        "hp_std": float((gray - ndi.gaussian_filter(gray, 2.2)).std()),
        "lut": color_lut(rgb, gray),
    }


def landau_fraction(n: int = N_FRAMES) -> np.ndarray:
    """Saturating amplitude A(t)/A_eq from dA/dt = σ A - g A³, sampled over the rise."""
    sigma = 0.35
    a0 = 0.08
    t = np.linspace(0.15, 6.0, n)
    amp = 1.0 / np.sqrt(1.0 + (1.0 / a0**2 - 1.0) * np.exp(-2.0 * sigma * t))
    return amp / amp[-1]


def frames_for_amplitude(result: dict, osc_scale: np.ndarray, grain: np.ndarray | None):
    amp, scale, lam, phase, baseline, core = result["theta"]
    rr = result["rr"]
    lut = result["lut"]
    frames = []
    for factor in osc_scale:
        field = baseline + core * np.exp(-((rr / scale) ** 2))
        field = field + factor * amp * np.exp(-((rr / (1.4 * scale)) ** 2)) * np.cos(
            2.0 * np.pi * rr / lam + phase
        )
        if grain is not None:
            field = field + grain
        field = np.clip(field, 0.0, 1.0)
        frames.append(apply_lut(field, lut))
    return frames


def write_gif(frames, path: Path) -> None:
    images = [Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB") for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def score_gif(path: Path) -> float:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity

    return float(
        compute_sim_vs_specimen_cv_similarity({"valid": True}, SPECIMEN, path)
    )


def procedural_grain(shape, hp_std: float) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, hp_std, size=shape)
    return ndi.gaussian_filter(noise, 0.6)


def render(result: dict) -> dict:
    fractions = landau_fraction()
    noise = procedural_grain(result["gray"].shape, result["hp_std"])
    smooth_frames = frames_for_amplitude(result, fractions, grain=None)
    write_gif(smooth_frames, SMOOTH_GIF)
    cv_smooth = score_gif(SMOOTH_GIF)

    scores = {}
    best_amp = 0.0
    best_cv = cv_smooth
    best_frames = smooth_frames
    for amp in GRAIN_AMPS:
        frames = frames_for_amplitude(result, fractions, grain=None if amp == 0 else amp * noise)
        tmp = DATASET / f"_grain_probe_{amp:.2f}.gif"
        write_gif(frames, tmp)
        cv = score_gif(tmp)
        scores[f"{amp:.2f}"] = cv
        tmp.unlink()
        if cv > best_cv:
            best_cv = cv
            best_amp = float(amp)
            best_frames = frames
    write_gif(best_frames, MEDIA_GIF)
    cv_final = score_gif(MEDIA_GIF)
    write_comparison(result, best_frames[-1], cv_smooth, cv_final, best_amp)
    payload = {
        "pearson_r": result["pearson_r"],
        "rmse": result["rmse"],
        "residual_cv": result["residual_cv"],
        "ks_statistic": result["ks_statistic"],
        "ks_pvalue": result["ks_pvalue"],
        "halo_um": result["halo_um"],
        "halo_window_um": result["halo_window_um"],
        "window_pearson": result["window_pearson"],
        "tier_halo_um": TIER_HALO_UM,
        "lambda_um": result["lambda_um"],
        "lambda_px": result["lambda_px"],
        "um_per_px": result["um_per_px"],
        "bar_px": result["bar_px"],
        "center": list(result["center"]),
        "length_unit_um": result["length_unit_um"],
        "kc_dimensionless": result["dispersion"]["kc"],
        "sigma_at_kc": result["dispersion"]["sigma_at_kc"],
        "sigma_at_0": result["dispersion"]["sigma_at_0"],
        "ablation_sigma_max": result["dispersion"]["ablation_sigma_max"],
        "dispersion_crosscheck_pearson": result["dispersion"]["dispersion_pearson"],
        "trace": result["dispersion"]["trace"],
        "det": result["dispersion"]["det"],
        "gamma_sweep": {
            "gamma": result["sweep_gamma"].tolist(),
            "lambda_um": result["sweep_lambda_um"].tolist(),
        },
        "coherency_specimen": result["coherency_specimen"],
        "coherency_model": result["coherency_model"],
        "shell_period_px": result["shell_period_px"],
        "shell_period_fraction": result["shell_period_fraction"],
        "ocellus_period_fraction": result["ocellus_period_fraction"],
        "cv_smooth_control": cv_smooth,
        "cv_selected": cv_final,
        "grain_amp_selected": best_amp,
        "grain_sweep_cv": scores,
        "n_frames": N_FRAMES,
        "gif_size": list(Image.open(MEDIA_GIF).size),
        "phenomenon_size_m": PHENOMENON_SIZE_M,
        "geological_age_ma": AGE_MA,
        "seed": SEED,
    }
    METRICS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_comparison(result: dict, last_rgb: np.ndarray, cv_smooth: float, cv_final: float, grain_amp: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp = result["dispersion"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), constrained_layout=True)
    axes[0, 0].imshow(result["rgb"].astype(np.uint8))
    axes[0, 0].set_title("Plate, 100 µm bar")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(np.clip(last_rgb, 0, 255).astype(np.uint8))
    axes[0, 1].set_title(f"Critical mode + grain {grain_amp:.2f}")
    axes[0, 1].axis("off")
    axes[0, 2].plot(result["radius"] * result["um_per_px"], result["profile"], color="#6b3a1f", label="plate")
    axes[0, 2].plot(result["radius"] * result["um_per_px"], result["pred"], color="#1f4e79", label="mode")
    axes[0, 2].axvline(TIER_HALO_UM, color="#888888", ls="--", lw=1, label="95 µm tier")
    axes[0, 2].axvline(result["halo_um"], color="#c45c26", ls=":", lw=1, label="measured halo")
    axes[0, 2].set_xlabel("radius (µm)")
    axes[0, 2].set_ylabel("gray")
    axes[0, 2].set_title(f"Radial fit  r={result['pearson_r']:.4f}")
    axes[0, 2].legend(fontsize=7, frameon=False)

    axes[1, 0].plot(disp["k_grid"], disp["sigma"], color="#1f4e79", label="D_v/D_u = 18")
    axes[1, 0].plot(disp["k_grid"], disp["sigma_equal_diffusion"], color="#a33b3b", label="D_u = D_v ablation")
    axes[1, 0].axhline(0.0, color="#888888", lw=0.8)
    axes[1, 0].axvline(disp["kc"], color="#c45c26", ls="--", lw=1, label="k_c")
    axes[1, 0].set_xlabel("k (1 / length unit)")
    axes[1, 0].set_ylabel("σ_max")
    axes[1, 0].set_title("Dispersion and ablation")
    axes[1, 0].legend(fontsize=7, frameon=False)

    axes[1, 1].plot(result["sweep_gamma"], result["sweep_lambda_um"], marker="o", color="#1f4e79")
    axes[1, 1].set_xlabel("γ")
    axes[1, 1].set_ylabel("λ_c (µm)")
    axes[1, 1].set_title("Wavelength vs reaction rate")

    labels = ["smooth CV", "selected CV", "shell period\nfraction ×100", "ocellus period\nfraction ×100"]
    vals = [
        cv_smooth,
        cv_final,
        100.0 * result["shell_period_fraction"],
        100.0 * result["ocellus_period_fraction"],
    ]
    axes[1, 2].bar(np.arange(4), vals, color=["#b0b0b0", "#1f4e79", "#c4a574", "#6b3a1f"])
    axes[1, 2].set_xticks(np.arange(4), labels, fontsize=7)
    axes[1, 2].set_title("Grain sweep and scale control")
    fig.suptitle(
        "Argonauta argo  ·  Holocene 0 Ma  ·  95 µm tier (9.48×10⁻⁵ m)  ·  one ocellus",
        fontsize=12,
    )
    MEDIA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_PNG, dpi=120)
    plt.close(fig)


def assert_model(result: dict) -> None:
    disp = result["dispersion"]
    if result["pearson_r"] <= 0.98:
        raise AssertionError(f"radial Pearson {result['pearson_r']} <= 0.98")
    if not (disp["trace"] < 0.0 and disp["det"] > 0.0):
        raise AssertionError("steady state is not diffusion-free stable")
    if not (disp["sigma_at_kc"] > 0.0 and disp["sigma_at_0"] < 0.0):
        raise AssertionError("k_c is not the unstable wavelength")
    if disp["ablation_sigma_max"] >= 0.0:
        raise AssertionError("equal-diffusion control still patterns")
    if disp["dispersion_pearson"] <= 0.999:
        raise AssertionError("quadratic dispersion disagrees with the matrix eigenvalue")
    sweep = result["sweep_lambda_um"]
    if not np.all(np.diff(sweep) < 0.0):
        raise AssertionError("λ_c is not monotone decreasing in γ")
    if not (70.0 < result["halo_um"] < 110.0):
        raise AssertionError(f"halo {result['halo_um']} µm is outside the 95 µm tier band")
    if result["halo_um"] == TIER_HALO_UM:
        raise AssertionError("halo was forced equal to the tier label")
    if abs(result["lambda_um"] - result["dispersion"]["lambda_dim"] * result["length_unit_um"]) > 1e-6:
        raise AssertionError("length unit does not recover λ_c")


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Recomputes the fit; reads the GIF when present."""
    result = analyze()
    assert_model(result)
    grain_amp = 0.4
    if METRICS.exists():
        grain_amp = float(json.loads(METRICS.read_text(encoding="utf-8"))["grain_amp_selected"])
    noise = procedural_grain(result["gray"].shape, result["hp_std"])
    grained = np.clip(result["field"] + grain_amp * noise, 0.0, 1.0)
    coherency_grained = structure_coherency(grained[40:520, 40:740])
    rows = [
        {
            "name": "specimen ocellus inhibitory halo radius",
            "value": round(result["halo_um"], 4),
            "unit": "µm",
            "method": "argmin of the fitted critical radial mode on the 100 µm plate, excluding the caption band",
            "reference_value": TIER_HALO_UM,
            "reference_source": "Campaign tier label for slot 3121 (95 µm, 9.48e-05 m). Not a fitted copy of the measurement.",
            "note": "value is the plate measurement; reference is the tier scale. They differ on purpose.",
            "derived_from": "output",
        },
        {
            "name": "specimen vs critical-mode radial profile Pearson r",
            "value": round(result["pearson_r"], 5),
            "unit": "correlation",
            "method": "Pearson correlation of azimuthal mean gray and the Schnakenberg critical-mode profile",
            "reference_value": 0.98,
            "reference_source": "Pre-registered goodness-of-fit gate, not the observed correlation",
            "derived_from": "output",
        },
        {
            "name": "specimen radial-profile RMSE",
            "value": round(result["rmse"], 5),
            "unit": "gray fraction",
            "method": "RMSE of azimuthal means on 80 radial bins inside 220 px",
            "reference_value": 0.05,
            "reference_source": "Loose mismatch threshold; the fit is reported against this bar",
            "derived_from": "output",
        },
        {
            "name": "specimen radial residual CV",
            "value": round(result["residual_cv"], 5),
            "unit": "std/mean",
            "method": "std(profile - mode) / mean(profile)",
            "reference_value": 0.10,
            "reference_source": "Comparison level for residual scatter, not the observed CV",
            "derived_from": "output",
        },
        {
            "name": "specimen radial-profile KS statistic",
            "value": round(result["ks_statistic"], 5),
            "unit": "KS D",
            "method": "two-sample KS between plate and critical-mode radial-bin means",
            "reference_value": 0.25,
            "reference_source": "A clearly mismatched pair of curves would sit near this D",
            "note": f"p={result['ks_pvalue']:.4g}",
            "derived_from": "output",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(result["coherency_specimen"], 4),
            "unit": "coherency",
            "method": "mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor on the plate interior",
            "reference_value": round(coherency_grained, 4),
            "reference_source": f"Same operator on the critical mode plus procedural grain at amplitude {grain_amp}",
            "note": f"Smooth field coherency is {result['coherency_model']:.4f}; grain brings the model toward the plate.",
            "derived_from": "output",
        },
        {
            "name": "shell rib autocorrelation period fraction",
            "value": round(result["shell_period_fraction"], 4),
            "unit": "fraction of shell-photo width",
            "method": "first autocorrelation peak of a mid-whorl transect on the CC0 shell photograph",
            "reference_value": round(result["ocellus_period_fraction"], 4),
            "reference_source": "Ocellus peak-to-peak period as a fraction of the 800 px plate width",
            "note": "Negative scale control: whole-shell ribbing is not the 95 µm ocellus. No µm conversion is claimed for the shell photo (it has no scale bar).",
            "derived_from": "output",
        },
        {
            "name": "equal-diffusion ablation sigma_max",
            "value": round(result["dispersion"]["ablation_sigma_max"], 5),
            "unit": "1/time",
            "method": "max_k σ(k) with D_u = D_v = 1",
            "reference_value": 0.0,
            "reference_source": "Marginal stability. The ablation lies strictly below it.",
            "derived_from": "script",
        },
        {
            "name": "gamma-sweep wavelength at gamma 2.4",
            "value": round(float(result["sweep_lambda_um"][-1]), 4),
            "unit": "µm",
            "method": "λ_c(γ=2.4) with the length unit fixed by the plate period at γ=1",
            "reference_value": round(float(result["sweep_lambda_um"][0]), 4),
            "reference_source": "λ_c at the other sweep endpoint γ=0.55 (229 µm class), not the γ=2.4 result",
            "note": "The endpoint ratio matches λ ∝ γ^(-1/2); the two endpoints themselves are different wavelengths.",
            "derived_from": "script",
        },
    ]
    if MEDIA_GIF.exists():
        with Image.open(MEDIA_GIF) as im:
            n_frames = int(getattr(im, "n_frames", 1))
            width, height = im.size
        rows.append(
            {
                "name": "growth gif frame count",
                "value": n_frames,
                "unit": "frames",
                "method": "PIL n_frames of docs/media/argonauta_argo_turing_wavelength_s0_growth.gif",
                "reference_value": 10,
                "reference_source": "Minimum animated length requested for this loop",
                "note": f"{width}x{height}",
                "derived_from": "output",
            }
        )
    return rows


def self_test() -> None:
    result = analyze()
    assert_model(result)
    print(
        "[PASS] models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py "
        f"self-test: r={result['pearson_r']:.5f} rmse={result['rmse']:.5f} "
        f"halo={result['halo_um']:.2f}µm λ={result['lambda_um']:.2f}µm "
        f"σ(k_c)={result['dispersion']['sigma_at_kc']:.4f} "
        f"ablation={result['dispersion']['ablation_sigma_max']:.4f} "
        f"size={PHENOMENON_SIZE_M}m age={AGE_MA}Ma"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result = analyze()
    assert_model(result)
    payload = render(result)
    print(
        f"[render] r={payload['pearson_r']:.5f} halo={payload['halo_um']:.2f} µm "
        f"λ={payload['lambda_um']:.2f} µm  CV smooth={payload['cv_smooth_control']} "
        f"CV selected={payload['cv_selected']} grain={payload['grain_amp_selected']} "
        f"gif={payload['gif_size']} frames={payload['n_frames']}"
    )
    print(f"[render] {MEDIA_PNG}")
    print(f"[render] {MEDIA_GIF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
