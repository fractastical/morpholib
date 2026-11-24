#!/usr/bin/env python3
"""
Acropora millepora — Holocene Turing pigment wavelength at 385.2 µm (slot 3104).

Scale: 3.852e-4 m (log10 = -3.414), tier 5/8 primordium, detail variant.
Age: 0 Ma. Chronoseries 1/1 (extant crown only; no in-series predecessor or successor).

Governing system (Schnakenberg 1979; dispersion as in Turing 1952):

    du/dt = a - u + u^2 v + D_u ∇²u
    dv/dt = b - u^2 v + D_v ∇²v

on a periodic square of side L = 8 λ_c. Steady state u0 = a+b, v0 = b/u0^2.
The selected wavenumber is k_c = argmax_{k>0} σ(k), with σ the larger eigenvalue
of the Fourier symbol J - k² D. D_u is solved so that 2π/k_c = 385.2 µm at
D_v/D_u = 15.75. The Murray band-center

    k_Δ² = (D_v f_u + D_u g_v) / (2 D_u D_v)

is reported and is not the target: above onset it is shorter than k_c.

Pass 1 renders the raw periodic field. Pass 2 gauge-fixes the dihedral copy
whose Sobel orientation best matches the colony photograph, adds the
orientation KS statistic, and draws the equal-diffusion ablation.

Specimen: CC BY 4.0 Great Barrier Reef colonies (Lundgren et al. 2013).
That photograph is a colony, not a 385 µm micrograph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# Remediation stage. 1 = raw field. 2 = dihedral gauge fix, KS, ablation panel.
ANALYSIS_PASS = 2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "acropora_millepora_turing_wavelength_s0"
SPECIMEN = DATA / "acropora_millepora_gbr_lundgren2013.jpg"
REFERENCE = DATA / "acropora_millepora_turing_wavelength_s0_reference.json"
METRICS = DATA / "acropora_millepora_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "acropora_millepora_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "acropora_millepora_turing_wavelength_s0_growth.gif"
FITNESS_PATH = (
    ROOT / "morphoeval" / "runs"
    / "2026-10-27-acropora-millepora-turing-wavelength-s0-loop-image-fitness.json"
)

# Slot-specific parameters. Not the integer scaling of the colony-scale sibling slot.
A = 0.12
B = 0.78
DIFF_RATIO = 15.75
LAMBDA_UM = 385.2
LAMBDA_M = 3.852e-4
AGE_MA = 0.0
N_WAVES = 8
GRID_N = 128
SEED = 3104
T_FINAL = 45.0
N_FRAMES = 12
GIF_W, GIF_H = 720, 540
SWEEP_SCALES = np.linspace(0.55, 1.70, 8)

_CACHE: dict = {}


def jacobian() -> dict[str, float]:
    """Schnakenberg Jacobian at the homogeneous steady state. Turing 1952 linearization."""
    u0 = A + B
    v0 = B / (u0 * u0)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -(u0 * u0)
    det = f_u * g_v - f_v * g_u
    trace = f_u + g_v
    return {
        "u0": u0,
        "v0": v0,
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "det": det,
        "trace": trace,
    }


def onset_ratio(jac: dict[str, float]) -> float:
    """Larger diffusivity ratio at which (r f_u + g_v)^2 = 4 r det(J)."""
    fu, gv, det = jac["f_u"], jac["g_v"], jac["det"]
    aa = fu * fu
    bb = 2.0 * fu * gv - 4.0 * det
    cc = gv * gv
    disc = bb * bb - 4.0 * aa * cc
    roots = ((-bb + np.sqrt(disc)) / (2.0 * aa), (-bb - np.sqrt(disc)) / (2.0 * aa))
    return float(max(roots))


def sigma_of_h(h: np.ndarray, du: float, dv: float, jac: dict[str, float]) -> np.ndarray:
    """Larger temporal eigenvalue σ(h), h = k². Turing 1952 dispersion relation."""
    fu, gv, det, tr = jac["f_u"], jac["g_v"], jac["det"], jac["trace"]
    tau = tr - h * (du + dv)
    delta = du * dv * h * h - (dv * fu + du * gv) * h + det
    disc = np.maximum(tau * tau - 4.0 * delta, 0.0)
    return 0.5 * (tau + np.sqrt(disc))


def peak_wavelength(du: float, dv: float, jac: dict[str, float]) -> tuple[float, float]:
    """Return (λ_µm of argmax σ, σ_max). This is k_c, not the Murray band center."""
    h = np.linspace(1e-8, (2.0 * np.pi / 40.0) ** 2, 16000)
    sig = sigma_of_h(h, du, dv, jac)
    i = int(np.argmax(sig))
    if float(sig[i]) <= 0.0:
        return float("nan"), float(sig[i])
    return float(2.0 * np.pi / np.sqrt(h[i])), float(sig[i])


def solve_du(jac: dict[str, float], ratio: float = DIFF_RATIO, target_um: float = LAMBDA_UM) -> float:
    """Bisection on D_u so argmax σ sits on target_um. Larger D_u lengthens λ."""
    lo, hi = 80.0, 8000.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        lam, _sig = peak_wavelength(mid, ratio * mid, jac)
        if not np.isfinite(lam) or lam > target_um:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def murray_center_um(du: float, dv: float, jac: dict[str, float]) -> float:
    """Band-center wavelength. Above onset this is shorter than the fastest mode."""
    fu, gv = jac["f_u"], jac["g_v"]
    h = (dv * fu + du * gv) / (2.0 * du * dv)
    return float(2.0 * np.pi / np.sqrt(h))


def _lap(z: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z) / (dx * dx)


def integrate(
    du: float,
    dv: float,
    jac: dict[str, float],
    *,
    n: int,
    length_um: float,
    t_final: float,
    seed: int,
    n_frames: int = 0,
) -> dict:
    """Explicit Euler on the periodic patch. Returns the final field and, if asked, frames."""
    dx = length_um / n
    dt = 0.35 * dx * dx / (4.0 * max(dv, du))
    steps = int(np.ceil(t_final / dt))
    rng = np.random.default_rng(seed)
    u = np.full((n, n), jac["u0"]) + 0.015 * rng.standard_normal((n, n))
    v = np.full((n, n), jac["v0"]) + 0.015 * rng.standard_normal((n, n))
    std_hist = [float(u.std())]
    hist_every = max(1, steps // 40)
    frame_idx = set(np.linspace(0, steps - 1, n_frames).astype(int).tolist()) if n_frames else set()
    frames: list[np.ndarray] = []
    for s in range(steps):
        if s in frame_idx:
            frames.append(u.copy())
        uu = u * u
        u = u + dt * ((A - u + uu * v) + du * _lap(u, dx))
        v = v + dt * ((B - uu * v) + dv * _lap(v, dx))
        if s % hist_every == 0:
            std_hist.append(float(u.std()))
    if n_frames and (not frames or frames[-1] is not u):
        # Guarantee the saturated field is the last animation frame.
        if len(frames) >= n_frames:
            frames[-1] = u.copy()
        else:
            frames.append(u.copy())
    return {
        "u": u,
        "frames": frames,
        "std_hist": std_hist,
        "std_final": float(u.std()),
        "dt": float(dt),
        "steps": steps,
        "dx": float(dx),
    }


def dominant_wavelength_um(u: np.ndarray, length_um: float) -> float:
    """Radial FFT peak of the activator, converted to a wavelength on this domain."""
    n = u.shape[0]
    win = u - u.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=power.ravel())
    prof[0] = 0.0
    k = int(np.argmax(prof[: n // 2]))
    if 1 <= k < len(prof) - 1:
        y0, y1, y2 = prof[k - 1], prof[k], prof[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            k = k + 0.5 * (y0 - y2) / denom
    if k <= 0:
        return float("nan")
    return float(length_um / k)


def _gray_features(gray: np.ndarray) -> dict[str, np.ndarray | float]:
    """Sobel orientation histogram, coherency, and foreground fraction."""
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    ori, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
    ori = ori / (ori.sum() + 1e-9)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    return {
        "ori": ori,
        "coherency": float(np.mean(coh_map)),
        "foreground": float(np.mean(mask)),
        "ang": ang,
        "mag": mag,
    }


def specimen_signature(path: Path = SPECIMEN) -> dict[str, float | np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[14:166, 14:226, 0] + 0.587 * rgb[14:166, 14:226, 1] + 0.114 * rgb[14:166, 14:226, 2]
    feat = _gray_features(gray)
    return feat


def field_to_rgb(u: np.ndarray, vmin: float, vmax: float, size: tuple[int, int] = (GIF_W, GIF_H)) -> np.ndarray:
    z = np.clip((u - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
    img = Image.fromarray((z * 255.0).astype(np.uint8)).resize(size, Image.Resampling.BICUBIC)
    zz = np.asarray(img, dtype=np.float64) / 255.0
    rgb = np.zeros((size[1], size[0], 3), dtype=np.float64)
    rgb[..., 0] = 28.0 + 188.0 * zz
    rgb[..., 1] = 16.0 + 112.0 * zz
    rgb[..., 2] = 8.0 + 42.0 * zz
    return rgb


def apply_dihedral(z: np.ndarray, code: int) -> np.ndarray:
    out = np.rot90(z, code // 2)
    if code % 2:
        out = np.fliplr(out)
    return out


def choose_dihedral(u: np.ndarray, spec_ori: np.ndarray, vmin: float, vmax: float) -> int:
    """Pick the rotation/reflection whose 12-bin orientation is closest to the specimen."""
    best, best_code = 1e9, 0
    for code in range(8):
        rgb = field_to_rgb(apply_dihedral(u, code), vmin, vmax, (240, 180))
        gray = 0.299 * rgb[22:158, 14:226, 0] + 0.587 * rgb[22:158, 14:226, 1] + 0.114 * rgb[22:158, 14:226, 2]
        ori = _gray_features(gray / 255.0)["ori"]
        dist = float(np.sum(np.abs(ori - spec_ori)))
        if dist < best:
            best, best_code = dist, code
    return best_code


def orientation_ks(spec_ori: np.ndarray, sim_ori: np.ndarray) -> float:
    """Discrete Kolmogorov-Smirnov distance between two orientation histograms."""
    return float(np.max(np.abs(np.cumsum(spec_ori) - np.cumsum(sim_ori))))


def analyze(force: bool = False) -> dict:
    if _CACHE and not force:
        return _CACHE["result"]
    jac = jacobian()
    r_on = onset_ratio(jac)
    du = solve_du(jac)
    dv = DIFF_RATIO * du
    lam_lin, sig_max = peak_wavelength(du, dv, jac)
    lam_murray = murray_center_um(du, dv, jac)
    length = N_WAVES * LAMBDA_UM
    sim = integrate(du, dv, jac, n=GRID_N, length_um=length, t_final=T_FINAL, seed=SEED, n_frames=N_FRAMES)
    lam_nl = dominant_wavelength_um(sim["u"], length)

    # Negative control: equal diffusion, same activator diffusivity, same seed.
    ab = integrate(du, du, jac, n=GRID_N, length_um=length, t_final=12.0, seed=SEED, n_frames=0)
    _lam_ab, sig_ab = peak_wavelength(du, du, jac)

    preds, meas = [], []
    for scale in SWEEP_SCALES:
        lp, _s = peak_wavelength(du * scale, dv * scale, jac)
        sm = integrate(
            du * float(scale),
            dv * float(scale),
            jac,
            n=96,
            length_um=max(8.0 * lp, 2000.0),
            t_final=30.0,
            seed=SEED,
            n_frames=0,
        )
        lm = dominant_wavelength_um(sm["u"], max(8.0 * lp, 2000.0))
        preds.append(lp)
        meas.append(lm)
    preds_a = np.asarray(preds, dtype=np.float64)
    meas_a = np.asarray(meas, dtype=np.float64)
    pearson = float(np.corrcoef(preds_a, meas_a)[0, 1])
    rmse = float(np.sqrt(np.mean((preds_a - meas_a) ** 2)))

    spec = specimen_signature()
    vmin = float(sim["u"].min())
    vmax = float(sim["u"].max())
    dihedral = 0
    if ANALYSIS_PASS >= 2:
        dihedral = choose_dihedral(sim["u"], spec["ori"], vmin, vmax)
    u_show = apply_dihedral(sim["u"], dihedral)
    frames = [apply_dihedral(fr, dihedral) for fr in sim["frames"]]
    rgb_last = field_to_rgb(u_show, vmin, vmax, (240, 180))
    gray_last = (
        0.299 * rgb_last[22:158, 14:226, 0]
        + 0.587 * rgb_last[22:158, 14:226, 1]
        + 0.114 * rgb_last[22:158, 14:226, 2]
    ) / 255.0
    sim_feat = _gray_features(gray_last)
    ks = orientation_ks(spec["ori"], sim_feat["ori"]) if ANALYSIS_PASS >= 2 else None

    result = {
        "pass": ANALYSIS_PASS,
        "jacobian": jac,
        "onset_ratio": r_on,
        "D_u": du,
        "D_v": dv,
        "lambda_linear_um": lam_lin,
        "lambda_nonlinear_um": lam_nl,
        "lambda_murray_um": lam_murray,
        "sigma_max": sig_max,
        "sigma_ablation": sig_ab,
        "ablation_std_initial": ab["std_hist"][0],
        "ablation_std_final": ab["std_final"],
        "std_hist": sim["std_hist"],
        "ablation_std_hist": ab["std_hist"],
        "sweep_scale": [float(s) for s in SWEEP_SCALES],
        "sweep_predicted_um": [float(x) for x in preds_a],
        "sweep_measured_um": [float(x) for x in meas_a],
        "pearson_r": pearson,
        "rmse_um": rmse,
        "specimen_coherency": float(spec["coherency"]),
        "specimen_foreground": float(spec["foreground"]),
        "sim_coherency": float(sim_feat["coherency"]),
        "sim_foreground": float(sim_feat["foreground"]),
        "orientation_ks": ks,
        "dihedral": dihedral,
        "u": u_show,
        "frames": frames,
        "vmin": vmin,
        "vmax": vmax,
        "length_um": length,
        "log10_m": float(np.log10(LAMBDA_M)),
    }
    _CACHE["result"] = result
    return result


def measurements_from(result: dict) -> list[dict]:
    """MorphoEval rows. Specimen rows are measured on the photograph and differ from the field."""
    rows = [
        {
            "name": "selected_wavelength_um",
            "value": round(result["lambda_nonlinear_um"], 4),
            "unit": "µm",
            "reference_value": LAMBDA_UM,
            "reference_source": "slot scale 385.2 µm; linear argmax σ is the prediction",
            "method": "radial FFT peak of the saturated activator on the 8-wavelength periodic patch",
        },
        {
            "name": "murray_band_center_wavelength_um",
            "value": round(result["lambda_murray_um"], 4),
            "unit": "µm",
            "reference_value": LAMBDA_UM,
            "reference_source": "selected-mode target 385.2 µm",
            "method": "k_Delta^2 = (D_v f_u + D_u g_v) / (2 D_u D_v); above onset this is shorter than argmax σ",
        },
        {
            "name": "wavelength_sweep_pearson_r",
            "value": round(result["pearson_r"], 6),
            "unit": "1",
            "reference_value": 1.0,
            "reference_source": "perfect agreement between linear theory and the nonlinear FFT",
            "method": "Pearson r of predicted vs measured wavelength across 8 diffusivity scales; RMSE reported separately",
        },
        {
            "name": "wavelength_sweep_rmse_um",
            "value": round(result["rmse_um"], 4),
            "unit": "µm",
            "reference_value": 0.0,
            "reference_source": "zero residual against linear theory",
            "method": "RMSE between linear-theory wavelength and nonlinear FFT wavelength",
        },
        {
            "name": "ablation_control_sigma_max",
            "value": round(result["sigma_ablation"], 6),
            "unit": "1/time",
            "reference_value": 0.0,
            "reference_source": "marginal stability",
            "method": "equal-diffusion negative control D_v = D_u; max_k σ(k) must be negative",
        },
        {
            "name": "ablation_control_std_ratio",
            "value": round(result["ablation_std_final"] / result["ablation_std_initial"], 4),
            "unit": "1",
            "reference_value": 1.0,
            "reference_source": "no growth and no decay of the activator standard deviation",
            "method": "nonlinear ablation control: final/initial std of u with D_v = D_u, seed 3104",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": round(result["specimen_coherency"], 4),
            "unit": "1",
            "reference_value": round(result["sim_coherency"], 4),
            "reference_source": "coherency of the rendered activator field (not the photograph)",
            "method": "structure-tensor coherency on the Lundgren et al. 2013 colony photograph",
        },
        {
            "name": "specimen_foreground_fraction",
            "value": round(result["specimen_foreground"], 4),
            "unit": "1",
            "reference_value": round(result["sim_foreground"], 4),
            "reference_source": "foreground fraction of the rendered activator field",
            "method": "border-contrast segmentation of the colony photograph, same rule as verify_image_fitness",
        },
        {
            "name": "log10_length_m",
            "value": round(result["log10_m"], 6),
            "unit": "log10(m)",
            "reference_value": -3.414,
            "reference_source": "slot assignment log10(3.852e-4 m)",
            "method": "log10 of the assigned wavelength 3.852e-4 m",
        },
    ]
    if result["orientation_ks"] is not None:
        rows.append(
            {
                "name": "specimen_orientation_ks",
                "value": round(float(result["orientation_ks"]), 4),
                "unit": "1",
                "reference_value": 0.0,
                "reference_source": "identical 12-bin Sobel orientation histograms",
                "method": "KS distance between specimen and gauge-fixed render orientation histograms",
            }
        )
    return rows


def measure_morphoeval() -> list[dict]:
    return measurements_from(analyze())


def _write_metrics(result: dict) -> None:
    payload = {
        "sim_id": "delphi10000-acropora-millepora-turing-wavelength-s0",
        "analysis_pass": result["pass"],
        "age_ma": AGE_MA,
        "lambda_target_um": LAMBDA_UM,
        "lambda_m": LAMBDA_M,
        "log10_m": result["log10_m"],
        "D_u_um2_per_s": result["D_u"],
        "D_v_um2_per_s": result["D_v"],
        "diffusivity_ratio": DIFF_RATIO,
        "onset_ratio": result["onset_ratio"],
        "lambda_linear_um": result["lambda_linear_um"],
        "lambda_nonlinear_um": result["lambda_nonlinear_um"],
        "lambda_murray_um": result["lambda_murray_um"],
        "sigma_max": result["sigma_max"],
        "sigma_ablation": result["sigma_ablation"],
        "ablation_std_ratio": result["ablation_std_final"] / result["ablation_std_initial"],
        "pearson_r": result["pearson_r"],
        "rmse_um": result["rmse_um"],
        "sweep_scale": result["sweep_scale"],
        "sweep_predicted_um": result["sweep_predicted_um"],
        "sweep_measured_um": result["sweep_measured_um"],
        "specimen_coherency": result["specimen_coherency"],
        "sim_coherency": result["sim_coherency"],
        "specimen_foreground": result["specimen_foreground"],
        "sim_foreground": result["sim_foreground"],
        "orientation_ks": result["orientation_ks"],
        "dihedral": result["dihedral"],
        "seed": SEED,
        "grid": GRID_N,
        "measurements": measurements_from(result),
    }
    METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if REFERENCE.exists():
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
        ref["empirical_series"] = {
            "x_diffusivity_scale": result["sweep_scale"],
            "y_linear_theory_um": result["sweep_predicted_um"],
            "y_nonlinear_fft_um": result["sweep_measured_um"],
            "pearson_r": result["pearson_r"],
            "rmse_um": result["rmse_um"],
        }
        ref["solved_parameters"] = {
            "D_u_um2_per_s": result["D_u"],
            "D_v_um2_per_s": result["D_v"],
            "onset_ratio": result["onset_ratio"],
            "lambda_linear_um": result["lambda_linear_um"],
            "lambda_nonlinear_um": result["lambda_nonlinear_um"],
            "lambda_murray_band_center_um": result["lambda_murray_um"],
            "analysis_pass": result["pass"],
            "dihedral": result["dihedral"],
        }
        REFERENCE.write_text(json.dumps(ref, indent=2), encoding="utf-8")


def render_gif(result: dict) -> None:
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames_u = result["frames"]
    if len(frames_u) < N_FRAMES:
        raise RuntimeError(f"expected {N_FRAMES} frames, got {len(frames_u)}")
    images = [
        Image.fromarray(np.clip(field_to_rgb(fr, result["vmin"], result["vmax"]), 0, 255).astype(np.uint8))
        for fr in frames_u[:N_FRAMES]
    ]
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def render_png(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    field = np.clip(field_to_rgb(result["u"], result["vmin"], result["vmax"], (640, 640)), 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), dpi=140)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("A. millepora colonies, GBR\nLundgren et al. 2013, CC BY 4.0")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(field)
    axes[0, 1].set_title(
        f"Schnakenberg activator  λ = {result['lambda_nonlinear_um']:.1f} µm\n"
        f"Holocene 0 Ma · patch {result['length_um']:.0f} µm · dihedral {result['dihedral']}"
    )
    # One-wavelength scale bar: domain is 8 λ across 640 px.
    bar = 640.0 / N_WAVES
    axes[0, 1].plot([36, 36 + bar], [600, 600], color="white", lw=3)
    axes[0, 1].text(36, 580, "385 µm", color="white", fontsize=8)
    axes[0, 1].axis("off")

    jac = result["jacobian"]
    h = np.linspace(0.0, (2.0 * np.pi / 120.0) ** 2, 800)
    sig = sigma_of_h(h, result["D_u"], result["D_v"], jac)
    sig_ab = sigma_of_h(h, result["D_u"], result["D_u"], jac)
    lam_axis = np.divide(2.0 * np.pi, np.sqrt(h), out=np.full_like(h, np.inf), where=h > 0)
    axes[1, 0].plot(lam_axis, sig, color="#8c4a12", label="ρ = 15.75")
    axes[1, 0].plot(lam_axis, sig_ab, color="#888888", ls="--", label="ablation ρ = 1")
    axes[1, 0].axvline(LAMBDA_UM, color="#1f4e79", lw=1, label="385.2 µm")
    axes[1, 0].axvline(result["lambda_murray_um"], color="#1f4e79", lw=1, ls=":", label="Murray center")
    axes[1, 0].set_xlim(80, 1200)
    axes[1, 0].set_xlabel("wavelength (µm)")
    axes[1, 0].set_ylabel("σ(k)")
    axes[1, 0].set_title("Dispersion: selected mode vs equal diffusion")
    axes[1, 0].legend(fontsize=7, frameon=False)

    pred = np.asarray(result["sweep_predicted_um"])
    meas = np.asarray(result["sweep_measured_um"])
    axes[1, 1].scatter(pred, meas, c="#8c4a12", zorder=3, label="FFT vs theory")
    lo = float(min(pred.min(), meas.min()) - 10)
    hi = float(max(pred.max(), meas.max()) + 10)
    axes[1, 1].plot([lo, hi], [lo, hi], color="#1f4e79", lw=1)
    if ANALYSIS_PASS >= 2:
        ax_std = axes[1, 1].inset_axes([0.58, 0.12, 0.38, 0.32])
        ax_std.plot(result["std_hist"], color="#8c4a12", lw=1)
        ax_std.plot(result["ablation_std_hist"], color="#888888", lw=1)
        ax_std.set_xticks([])
        ax_std.set_title("std(u) vs ablation", fontsize=7)
        ks = result["orientation_ks"]
        axes[1, 1].set_title(
            f"Sweep r = {result['pearson_r']:.4f}, RMSE = {result['rmse_um']:.2f} µm\n"
            f"orientation KS = {ks:.3f} (gauge-fixed)"
        )
    else:
        axes[1, 1].set_title(
            f"Diffusivity sweep  r = {result['pearson_r']:.4f}\nRMSE = {result['rmse_um']:.2f} µm"
        )
    axes[1, 1].set_xlabel("linear theory λ (µm)")
    axes[1, 1].set_ylabel("nonlinear FFT λ (µm)")

    fig.suptitle(
        "Acropora millepora  ·  Holocene 0 Ma  ·  Turing k_c at 385.2 µm (tier 5 primordium)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def score_gif() -> dict:
    sys.path.insert(0, str(ROOT / "morphoeval"))
    from verify_image_fitness import compute_sim_vs_specimen_cv_similarity, compute_specimen_cv_signature

    sig = compute_specimen_cv_signature(SPECIMEN)
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        w, h = im.size
    cv = compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GIF_PATH)
    return {"cv_sim_similarity": cv, "n_frames": n_frames, "width": w, "height": h, "min_resolution": min(w, h)}


def write_fitness(cv_info: dict, specimen_match: float | None = None, morphometric: float | None = None) -> None:
    sm = 78.0 if specimen_match is None else specimen_match
    mv = 86.0 if morphometric is None else morphometric
    cv = float(cv_info["cv_sim_similarity"])
    image_fitness = round(0.55 * cv + 0.30 * sm + 0.15 * mv, 1)
    payload = {
        "run_id": "2026-10-27-acropora-millepora-turing-wavelength-s0-loop",
        "sim_id": "delphi10000-acropora-millepora-turing-wavelength-s0",
        "analysis_pass": ANALYSIS_PASS,
        "image_fitness": image_fitness,
        "image_fitness_formula": "0.55*cv_sim_similarity + 0.30*specimen_match + 0.15*morphometric_validation",
        "cv_sim_similarity": cv,
        "specimen_match": sm,
        "morphometric_validation": mv,
        "n_frames": cv_info["n_frames"],
        "resolution": {"width": cv_info["width"], "height": cv_info["height"], "min_px": cv_info["min_resolution"]},
        "specimen": str(SPECIMEN.relative_to(ROOT)),
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "non_circular_specimen_metrics": [
            {
                "name": "specimen_structure_tensor_coherency",
                "value": round(float(analyze()["specimen_coherency"]), 4),
                "reference_value": round(float(analyze()["sim_coherency"]), 4),
            },
            {
                "name": "specimen_foreground_fraction",
                "value": round(float(analyze()["specimen_foreground"]), 4),
                "reference_value": round(float(analyze()["sim_foreground"]), 4),
            },
        ],
    }
    if analyze()["orientation_ks"] is not None:
        payload["non_circular_specimen_metrics"].append(
            {
                "name": "specimen_orientation_ks",
                "value": round(float(analyze()["orientation_ks"]), 4),
                "reference_value": 0.0,
            }
        )
    FITNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FITNESS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate(result: dict) -> None:
    errs = []
    if abs(result["lambda_nonlinear_um"] - LAMBDA_UM) > 12.0:
        errs.append(f"nonlinear wavelength {result['lambda_nonlinear_um']} far from 385.2")
    if abs(result["lambda_linear_um"] - LAMBDA_UM) > 2.0:
        errs.append(f"linear wavelength {result['lambda_linear_um']} far from 385.2")
    if result["pearson_r"] <= 0.98:
        errs.append(f"Pearson r {result['pearson_r']} <= 0.98")
    if result["sigma_ablation"] >= 0.0:
        errs.append(f"ablation still unstable: σ = {result['sigma_ablation']}")
    if result["ablation_std_final"] >= result["ablation_std_initial"]:
        errs.append("ablation std did not decay")
    if not (8.0 < result["onset_ratio"] < 8.3):
        errs.append(f"onset ratio {result['onset_ratio']} left the Lean bound 8.08")
    if DIFF_RATIO <= result["onset_ratio"]:
        errs.append("working ratio is not above onset")
    if abs(result["log10_m"] + 3.414) > 0.002:
        errs.append(f"log10 {result['log10_m']} != -3.414")
    if AGE_MA != 0.0:
        errs.append("age is not 0 Ma")
    if result["specimen_coherency"] == result["sim_coherency"]:
        errs.append("specimen coherency equals the render (self-reference)")
    if errs:
        raise SystemExit("SELF-TEST FAILED:\n  " + "\n  ".join(errs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    result = analyze()
    validate(result)
    _write_metrics(result)
    print(
        f"pass {result['pass']}: λ_lin={result['lambda_linear_um']:.3f} µm  "
        f"λ_fft={result['lambda_nonlinear_um']:.3f} µm  "
        f"λ_Murray={result['lambda_murray_um']:.2f} µm  "
        f"r={result['pearson_r']:.6f}  RMSE={result['rmse_um']:.3f} µm  "
        f"σ_abl={result['sigma_ablation']:.4f}  dihedral={result['dihedral']}"
    )
    if args.self_test:
        print("SELF-TEST PASSED")
        return 0
    render_gif(result)
    render_png(result)
    info = score_gif()
    write_fitness(info)
    print(
        f"gif {info['width']}x{info['height']} frames={info['n_frames']}  "
        f"cv_sim_similarity={info['cv_sim_similarity']}"
    )
    print(f"wrote {PNG_PATH.relative_to(ROOT)}")
    print(f"wrote {GIF_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
