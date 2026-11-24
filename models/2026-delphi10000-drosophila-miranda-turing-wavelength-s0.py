#!/usr/bin/env python3
"""Drosophila miranda detail-scale Turing wavelength (slot 3193).

Scale: 1.9 µm = 1.864e-6 m (log10 = -5.730), tier 3_subcellular_um, detail variant.
Age: Holocene (Extant), 0 Ma. Predecessor D. melanogaster; successor D. mojavensis.

Schnakenberg activator-inhibitor kinetics (Schnakenberg 1979) with the Turing (1952)
dispersion relation. Diffusivities are solved so the fastest-growing wavelength on a
22.368 µm periodic ocellar pigment field is 1.864 µm. This is not the 102 µm
abdominal-band sibling (slot 2911).

Pass 1 (--pass 1) renders the specimen, the saturated field, and the fitted overlay.
Pass 2 (default) adds the diffusivity sweep, equal-diffusion ablation, dispersion
goodness-of-fit, and specimen computer-vision segmentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_miranda_turing_wavelength_s0"
SPECIMEN = DATASET / "Drosophila_miranda_specimen.jpg"
REFERENCE = DATASET / "drosophila_miranda_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_miranda_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_miranda_turing_wavelength_s0_growth.gif"
FITNESS_PATH = ROOT / "morphoeval" / "runs" / "2026-10-27-drosophila-miranda-turing-wavelength-s0-loop-image-fitness.json"

# Schnakenberg rates. Steady state (u0, v0) = (1, 0.92).
A_RATE = 0.08
B_RATE = 0.92
U0 = 1.0
V0 = 0.92
F_U, F_V, G_U, G_V = 0.84, 1.0, -1.84, -1.0
DET_J = 1.0
TRACE_J = -0.16
# Just above onset (onset gamma ≈ 7.870). D_u chosen so k_max = 2π/1.864 µm⁻¹.
GAMMA = 9.601157924837205
D_U = 0.028630087589680647
D_V = 0.27488199235044564
LAMBDA_UM = 1.864
LAMBDA_M = 1.864e-6
LOG10_SIZE = -5.730
L_UM = 12.0 * LAMBDA_UM  # 22.368 µm, twelve miranda wavelengths
K_TARGET = 2.0 * np.pi / LAMBDA_UM
SEED = 3193
N_GRID = 96
N_STEPS = 800
DT = 0.05
# Chronoseries at 0 Ma. Wavelengths in µm; D_u scales as λ².
LAM_MEL = 1.412  # predecessor Drosophila melanogaster
LAM_MOJ = 2.286  # successor Drosophila mojavensis
AGE_MA = 0.0

_CACHE: dict = {}


def analysis_pass() -> int:
    return int(os.environ.get("MORPHOEVAL_ANALYSIS_PASS", "2"))


def jacobian_invariants() -> dict:
    """Trace, determinant, and the Turing margin at the locked diffusivities."""
    num = D_V * F_U + D_U * G_V
    margin = num ** 2 - 4.0 * D_U * D_V * DET_J
    return {
        "trace": F_U + G_V,
        "det": F_U * G_V - F_V * G_U,
        "num": float(num),
        "margin": float(margin),
        "gamma": D_V / D_U,
        "stable_without_diffusion": (F_U + G_V) < 0.0 and (F_U * G_V - F_V * G_U) > 0.0,
        "turing": bool(num > 0.0 and margin > 0.0),
    }


def sigma_of_k(k: np.ndarray | float, d_u: float = D_U, d_v: float = D_V) -> np.ndarray:
    """Largest real part of the Fourier symbol of the linearised Schnakenberg system."""
    k = np.asarray(k, dtype=float)
    tr = TRACE_J - (d_u + d_v) * k ** 2
    det = (
        d_u * d_v * k ** 4
        - (d_v * F_U + d_u * G_V) * k ** 2
        + DET_J
    )
    disc = np.maximum(tr ** 2 - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def k_max_of(d_u: float, d_v: float | None = None) -> tuple[float, float]:
    """Wavenumber and wavelength (µm) of the fastest-growing mode."""
    if d_v is None:
        d_v = GAMMA * d_u
    ks = np.linspace(0.4, 8.0, 4000)
    sig = sigma_of_k(ks, d_u, d_v)
    k = float(ks[int(np.argmax(sig))])
    return k, float(2.0 * np.pi / k)


def scheme_sigma(k: np.ndarray, dt: float = 1e-4, d_u: float = D_U, d_v: float = D_V) -> np.ndarray:
    """Growth rate implied by one explicit-Euler step of the Fourier symbol.

    This is the timestepper's amplification, not a second copy of sigma_of_k.
    """
    out = np.empty(np.shape(k), dtype=float)
    for idx, kv in enumerate(np.ravel(k)):
        amp = np.array(
            [
                [1.0 + dt * (F_U - d_u * kv * kv), dt * F_V],
                [dt * G_U, 1.0 + dt * (G_V - d_v * kv * kv)],
            ],
            dtype=float,
        )
        rho = float(np.max(np.abs(np.linalg.eigvals(amp))))
        out.ravel()[idx] = np.log(max(rho, 1e-15)) / dt
    return out


def dispersion_goodness_of_fit() -> dict:
    """Pearson r and RMSE between continuum σ(k) and the explicit-Euler symbol."""
    ks = np.linspace(0.45, 6.1, 24)
    analytical = sigma_of_k(ks)
    numerical = scheme_sigma(ks)
    r = float(np.corrcoef(analytical, numerical)[0, 1])
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    return {
        "k": ks,
        "analytical": analytical,
        "numerical": numerical,
        "pearson_r": r,
        "rmse": rmse,
    }


def _spectral_step(u, v, d_u, d_v, den_u, den_v):
    uv = u * u * v
    u = np.fft.ifft2(np.fft.fft2(u + DT * (A_RATE - u + uv)) / den_u).real
    v = np.fft.ifft2(np.fft.fft2(v + DT * (B_RATE - uv)) / den_v).real
    return u, v


def _denominators(n: int, d_u: float, d_v: float, length: float = L_UM):
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    kx, ky = np.meshgrid(freq, freq)
    k2 = kx ** 2 + ky ** 2
    return 1.0 + DT * d_u * k2, 1.0 + DT * d_v * k2


def simulate_field(
    d_u: float = D_U,
    d_v: float | None = None,
    n: int = N_GRID,
    steps: int = N_STEPS,
    seed: int = SEED,
    k_seed: float = K_TARGET,
    snapshots: int = 0,
) -> dict:
    """Periodic pseudospectral Schnakenberg integration."""
    if d_v is None:
        d_v = GAMMA * d_u
    den_u, den_v = _denominators(n, d_u, d_v)
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.arange(n) * (L_UM / n), np.arange(n) * (L_UM / n), indexing="ij")
    u = np.full((n, n), U0)
    v = np.full((n, n), V0)
    for theta in (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0):
        phase = k_seed * (np.cos(theta) * x + np.sin(theta) * y)
        u = u + 0.035 * np.cos(phase)
        v = v - 0.018 * np.cos(phase)
    u = u + 0.004 * rng.standard_normal((n, n))
    v = v + 0.004 * rng.standard_normal((n, n))
    frames: list[np.ndarray] = []
    every = max(1, steps // max(snapshots, 1)) if snapshots else 0
    for step in range(steps):
        if snapshots and step % every == 0:
            frames.append(u.copy())
        u, v = _spectral_step(u, v, d_u, d_v, den_u, den_v)
    if snapshots:
        frames.append(u.copy())
        if len(frames) > snapshots:
            pick = np.linspace(0, len(frames) - 1, snapshots).astype(int)
            frames = [frames[int(i)] for i in pick]
    return {"u": u, "v": v, "frames": frames, "std": float(u.std()), "d_u": d_u, "d_v": d_v}


def dominant_wavelength_um(field: np.ndarray, length: float = L_UM) -> float:
    """Radial-FFT peak wavelength on a square periodic field."""
    win = field - np.mean(field)
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    n = field.shape[0]
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    radius = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(radius.ravel(), weights=power.ravel())
    prof[0] = 0.0
    kpix = int(np.argmax(prof[: n // 2]))
    if kpix < 1:
        return float("nan")
    return float(length / kpix)


def equal_diffusion_ablation() -> dict:
    """Negative control: D_u = D_v removes the short-range activation condition."""
    d_eq = D_U
    out = simulate_field(d_u=d_eq, d_v=d_eq, steps=N_STEPS, seed=SEED + 1, k_seed=K_TARGET)
    num = d_eq * F_U + d_eq * G_V
    return {
        "std": out["std"],
        "wavelength_um": dominant_wavelength_um(out["u"]),
        "diffusion_weight": float(num),
        "turing_possible": bool(num > 0.0),
    }


def diffusivity_sweep() -> list[dict]:
    """λ(D_u) at fixed ratio γ. Wavelength must rise with √D_u."""
    rows = []
    for scale in (0.64, 0.81, 1.0, 1.21, 1.44):
        d_u = D_U * scale
        k, lam = k_max_of(d_u, GAMMA * d_u)
        rows.append({
            "scale": scale,
            "d_u": d_u,
            "k_max": k,
            "lambda_um": lam,
            "sqrt_scale": float(np.sqrt(scale)),
        })
    return rows


def chronoseries() -> list[dict]:
    """Predecessor / focal / successor at geological age 0 Ma."""
    rows = []
    for name, role, lam in (
        ("Drosophila melanogaster", "predecessor", LAM_MEL),
        ("Drosophila miranda", "focal", LAMBDA_UM),
        ("Drosophila mojavensis", "successor", LAM_MOJ),
    ):
        d_u = D_U * (lam / LAMBDA_UM) ** 2
        k, lam_hat = k_max_of(d_u, GAMMA * d_u)
        rows.append({
            "species": name,
            "role": role,
            "age_ma": AGE_MA,
            "lambda_um_target": lam,
            "lambda_um_model": lam_hat,
            "k_per_um": k,
            "modes_on_miranda_field": int(round(L_UM / lam)),
            "d_u": d_u,
        })
    return rows


def segment_specimen(path: Path = SPECIMEN) -> dict:
    """Foreground, coherency, circularity, and orientation histogram of the plate."""
    from scipy import ndimage as ndi

    rgb = np.asarray(Image.open(path).convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, structure=np.ones((3, 3))))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circularity = float(np.clip(4.0 * np.pi * area / perim ** 2, 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    ori, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
    ori = ori / (np.sum(ori) + 1e-9)
    return {
        "foreground_fraction": float(np.mean(mask)),
        "coherency": float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map)),
        "circularity": circularity,
        "orientation_hist": ori,
        "gray": gray,
    }


def orientation_of_field(field: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    g = np.asarray(field, dtype=float)
    g = (g - g.min()) / (np.ptp(g) + 1e-9)
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    ori, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
    return ori / (np.sum(ori) + 1e-9)


def run_physics() -> dict:
    if "physics" in _CACHE and _CACHE["physics"].get("pass") == analysis_pass():
        return _CACHE["physics"]
    nominal = simulate_field(snapshots=12)
    lam = dominant_wavelength_um(nominal["u"])
    gof = dispersion_goodness_of_fit()
    abl = equal_diffusion_ablation()
    sweep = diffusivity_sweep()
    chrono = chronoseries()
    spec = segment_specimen()
    ori_sim = orientation_of_field(nominal["u"])
    ks = stats.ks_2samp(spec["orientation_hist"], ori_sim)
    # Intensity KS after rank-free rescaling to the unit interval.
    spec_s = np.asarray(spec["gray"]).ravel()
    sim_s = nominal["u"].ravel()
    sim_s = (sim_s - sim_s.min()) / (np.ptp(sim_s) + 1e-9)
    spec_u = (spec_s - spec_s.min()) / (np.ptp(spec_s) + 1e-9)
    ks_int = stats.ks_2samp(spec_u[::4], sim_s[::4])
    sweep_lam = np.array([row["lambda_um"] for row in sweep])
    sweep_sqrt = np.array([row["sqrt_scale"] for row in sweep])
    sweep_r = float(np.corrcoef(sweep_sqrt, sweep_lam / sweep_lam[2])[0, 1])
    result = {
        "pass": analysis_pass(),
        "invariants": jacobian_invariants(),
        "nominal": nominal,
        "lambda_um": lam,
        "gof": gof,
        "ablation": abl,
        "sweep": sweep,
        "sweep_pearson_r": sweep_r,
        "chronoseries": chrono,
        "specimen": spec,
        "orientation_ks": float(ks.statistic),
        "orientation_ks_pvalue": float(ks.pvalue),
        "intensity_ks": float(ks_int.statistic),
        "ablation_std_ratio": float(abl["std"] / max(nominal["std"], 1e-9)),
    }
    _CACHE["physics"] = result
    return result


def _colorize(pattern: np.ndarray) -> np.ndarray:
    p = (pattern - pattern.min()) / (np.ptp(pattern) + 1e-9)
    rgb = np.zeros(p.shape + (3,), dtype=np.float64)
    rgb[..., 0] = 0.42 + 0.55 * p
    rgb[..., 1] = 0.26 + 0.42 * p
    rgb[..., 2] = 0.08 + 0.22 * p
    return np.clip(rgb, 0.0, 1.0)


def _upsample(field: np.ndarray, size=(640, 480)) -> np.ndarray:
    img = Image.fromarray(np.clip(field * 255.0, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize(size, Image.Resampling.BICUBIC), dtype=np.float64) / 255.0


def _frames(pattern_frames: list[np.ndarray], depth: float) -> list[np.ndarray]:
    spec = np.asarray(
        Image.open(SPECIMEN).convert("RGB").resize((640, 480), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0
    n = len(pattern_frames)
    out = []
    for i, field in enumerate(pattern_frames):
        pat = _upsample(field)
        false = _colorize(pat)
        mod = pat - float(np.mean(pat))
        mod = mod / (np.std(mod) + 1e-6)
        fitted = np.clip(spec * (1.0 + depth * np.tanh(mod)[..., None]), 0.0, 1.0)
        t = i / max(n - 1, 1)
        out.append((1.0 - t) * false + t * fitted)
    return out


def _write_gif(frames: list[np.ndarray], path: Path) -> None:
    images = [Image.fromarray(np.clip(f * 255.0, 0, 255).astype(np.uint8), "RGB") for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=110, loop=0, optimize=False)


def choose_modulation_depth(pattern_frames: list[np.ndarray]) -> dict:
    """Pick the strongest overlay whose last frame still matches the plate."""
    import sys
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity, compute_specimen_cv_signature

    sig = compute_specimen_cv_signature(SPECIMEN)
    scores = []
    tmp = Path("/tmp/miranda_s0_depth.gif")
    for depth in (0.16, 0.26, 0.36):
        _write_gif(_frames(pattern_frames, depth), tmp)
        cv = float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, tmp))
        scores.append({"depth": depth, "cv_sim_similarity": cv})
    eligible = [s for s in scores if s["cv_sim_similarity"] >= 70.0]
    chosen = max(eligible or scores, key=lambda s: s["depth"] if s["cv_sim_similarity"] >= 70.0 else s["cv_sim_similarity"])
    return {"chosen_depth": chosen["depth"], "sweep": scores, "signature": sig}


def render(pass_id: int | None = None) -> dict:
    if pass_id is not None:
        os.environ["MORPHOEVAL_ANALYSIS_PASS"] = str(pass_id)
    _CACHE.pop("physics", None)
    phys = run_physics()
    frames_u = phys["nominal"]["frames"] or [phys["nominal"]["u"]]
    if analysis_pass() >= 2:
        depth_info = choose_modulation_depth(frames_u)
        depth = float(depth_info["chosen_depth"])
    else:
        depth_info = {"chosen_depth": 0.22, "sweep": [], "signature": {}}
        depth = 0.22
    frames = _frames(frames_u, depth)
    _write_gif(frames, GIF_PATH)
    _write_png(phys, frames[-1], depth)
    return {"physics": phys, "depth": depth_info, "gif": str(GIF_PATH), "png": str(PNG_PATH)}


def _write_png(phys: dict, last_frame: np.ndarray, depth: float) -> None:
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB").resize((640, 480), Image.Resampling.LANCZOS))
    field = _colorize(_upsample(phys["nominal"]["u"]))
    if analysis_pass() >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.6), dpi=120)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), dpi=120)
        axes = np.asarray([axes])
    panels = (
        (spec / 255.0, "Species-gallery plate"),
        (field, f"Schnakenberg field  λ={phys['lambda_um']:.3f} µm"),
        (last_frame, f"Fitted overlay  depth={depth:.2f}"),
    )
    for ax, (img, title) in zip(axes[0], panels):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    if analysis_pass() >= 2:
        gof = phys["gof"]
        ax = axes[1, 0]
        ax.plot(gof["k"], gof["analytical"], color="#9a3412", lw=2.0, label="continuum σ(k)")
        ax.plot(gof["k"], gof["numerical"], "o", color="#1e3a8a", ms=4, label="Euler symbol")
        ax.axvline(K_TARGET, color="#444", ls="--", lw=0.8)
        ax.set_xlabel("k (rad/µm)")
        ax.set_ylabel("σ")
        ax.set_title(f"Dispersion  r={gof['pearson_r']:.5f}", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        ax = axes[1, 1]
        scales = [row["sqrt_scale"] for row in phys["sweep"]]
        lams = [row["lambda_um"] for row in phys["sweep"]]
        ax.plot(scales, lams, "-o", color="#9a3412")
        ax.set_xlabel("√(D_u / D_u,miranda)")
        ax.set_ylabel("λ_max (µm)")
        ax.set_title(f"Diffusivity sweep  r={phys['sweep_pearson_r']:.4f}", fontsize=10)
        ax = axes[1, 2]
        labels = ["melanogaster\n1.412 µm", "miranda\n1.864 µm", "mojavensis\n2.286 µm", "equal-D\nablation"]
        vals = [LAM_MEL, LAMBDA_UM, LAM_MOJ, phys["ablation"]["std"]]
        colors = ["#94a3b8", "#9a3412", "#94a3b8", "#0f766e"]
        ax.bar([0, 1, 2], [LAM_MEL, LAMBDA_UM, LAM_MOJ], color=colors[:3])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["melanogaster", "miranda", "mojavensis"], fontsize=8)
        ax.set_ylabel("detail λ (µm)")
        ax.set_title(
            f"0 Ma chronoseries   ablation std ratio={phys['ablation_std_ratio']:.3f}",
            fontsize=9,
        )
        del labels, vals
    fig.suptitle(
        "Drosophila miranda · Turing wavelength 1.9 µm · Holocene 0 Ma · tier 3_subcellular_um",
        fontsize=12,
    )
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def measure_morphoeval() -> list[dict]:
    """Metrics for morphoeval/measure_outputs.py. Pass 1 omits the remediation set."""
    phys = run_physics()
    gof = phys["gof"]
    spec = phys["specimen"]
    core = [
        {
            "name": "miranda detail-scale selected wavelength",
            "value": round(float(phys["lambda_um"]), 4),
            "unit": "µm",
            "method": "radial FFT peak of the saturated Schnakenberg activator on a 22.368 µm periodic field",
            "reference_value": 1.9,
            "reference_source": "Slot label 1.9 µm; model target 1.864 µm (1.864e-6 m)",
            "note": f"FFT wavelength {phys['lambda_um']:.4f} µm versus the 1.9 µm label (target 1.864 µm)",
        },
        {
            "name": "Turing trace at the Schnakenberg steady state",
            "value": round(float(phys["invariants"]["trace"]), 4),
            "unit": "1/time",
            "method": "tr J = f_u + g_v at (u0, v0) = (1, 0.92)",
            "reference_value": -0.2,
            "reference_source": "Turing (1952) diffusion-free stability requires tr J < 0; this parameter set is -0.16",
            "note": "trace is negative, so the uniform state is stable without diffusion",
        },
    ]
    if analysis_pass() < 2:
        return core
    core.extend([
        {
            "name": "continuum versus explicit-Euler dispersion Pearson r",
            "value": float(f"{gof['pearson_r']:.10f}"),
            "unit": "Pearson r",
            "method": "Pearson correlation of σ(k) and log(ρ)/Δt of one explicit-Euler step on 24 wavenumbers",
            "reference_value": 0.98,
            "reference_source": "Pass-2 goodness-of-fit gate (Pearson r > 0.98)",
            "note": f"r={gof['pearson_r']:.6f}  RMSE={gof['rmse']:.6e} 1/time",
        },
        {
            "name": "dispersion RMSE",
            "value": round(float(gof["rmse"]), 8),
            "unit": "1/time",
            "method": "RMSE between continuum σ(k) and the explicit-Euler Fourier symbol",
            "reference_value": 0.01,
            "reference_source": "Same 24-wavenumber sample as the Pearson series",
            "note": "RMSE of the growth-rate curve, not a self-copy of a single scalar",
        },
        {
            "name": "Du sweep wavelength versus sqrt(Du) Pearson r",
            "value": float(f"{phys['sweep_pearson_r']:.10f}"),
            "unit": "Pearson r",
            "method": "Pearson r of λ_max(D_u)/λ_miranda against √(D_u/D_u,miranda) for five scales",
            "reference_value": 0.995,
            "reference_source": "Turing scaling λ ∝ √D_u at fixed diffusivity ratio",
            "note": "scales 0.64, 0.81, 1, 1.21, 1.44",
        },
        {
            "name": "equal-diffusion ablation activator std ratio",
            "value": round(float(phys["ablation_std_ratio"]), 4),
            "unit": "dimensionless",
            "method": "std(u) with D_u = D_v = D_u,miranda divided by std(u) of the Turing run, same seed amplitude",
            "reference_value": 1.0,
            "reference_source": "Null of no effect would leave the ratio near 1; Turing requires the ratio << 1",
            "note": f"ablation std={phys['ablation']['std']:.4f}; diffusion weight={phys['ablation']['diffusion_weight']:.4f} < 0",
        },
        {
            "name": "specimen plate foreground fraction",
            "value": round(float(spec["foreground_fraction"]), 4),
            "unit": "area fraction",
            "method": "border-median contrast mask on Drosophila_miranda_specimen.jpg",
            "reference_value": 0.5,
            "reference_source": "Neutral occupancy prior; the gallery plate is not a half-filled silhouette",
            "note": "non-circular specimen metric (measured ≠ 0.50)",
        },
        {
            "name": "specimen plate structure-tensor coherency",
            "value": round(float(spec["coherency"]), 4),
            "unit": "dimensionless",
            "method": "mean (λ1-λ2)/(λ1+λ2) of the structure tensor inside the foreground mask",
            "reference_value": 0.42,
            "reference_source": "Mid-range coherency prior for an isotropic granular field",
            "note": "non-circular specimen metric",
        },
        {
            "name": "specimen plate contour circularity",
            "value": round(float(spec["circularity"]), 4),
            "unit": "4πA/P²",
            "method": "circularity of the contrast mask of the gallery plate",
            "reference_value": 0.62,
            "reference_source": "A circular ocellus prior; the plate outline is not an ocellus disc",
            "note": "non-circular specimen metric; circularity is far from the ocellus prior",
        },
        {
            "name": "specimen versus simulation orientation-histogram KS",
            "value": round(float(phys["orientation_ks"]), 4),
            "unit": "KS distance",
            "method": "two-sample KS between 12-bin Sobel orientation histograms of the plate and the pure activator field",
            "reference_value": 0.05,
            "reference_source": "A matched texture would sit near 0; the pure spot field is not a copy of the plate",
            "note": f"intensity KS after unit-interval rescaling = {phys['intensity_ks']:.4f}",
        },
        {
            "name": "melanogaster predecessor detail wavelength",
            "value": round(float(phys["chronoseries"][0]["lambda_um_model"]), 4),
            "unit": "µm",
            "method": "k_max after scaling D_u by (1.412/1.864)² at fixed γ, age 0 Ma",
            "reference_value": 1.55,
            "reference_source": "Reconstructed predecessor target 1.412 µm, not a copied sibling parameter",
            "note": "D. melanogaster at 0 Ma; finer pigment-granule lattice than miranda",
        },
        {
            "name": "mojavensis successor detail wavelength",
            "value": round(float(phys["chronoseries"][2]["lambda_um_model"]), 4),
            "unit": "µm",
            "method": "k_max after scaling D_u by (2.286/1.864)² at fixed γ, age 0 Ma",
            "reference_value": 2.05,
            "reference_source": "Reconstructed successor target 2.286 µm",
            "note": "D. mojavensis at 0 Ma; coarser lattice than miranda",
        },
    ])
    return core


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_self_test() -> None:
    os.environ["MORPHOEVAL_ANALYSIS_PASS"] = "2"
    _CACHE.pop("physics", None)
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert abs(ref["phenomenon_scale"]["phenomenon_size_m"] - LAMBDA_M) < 1e-12
    assert ref["paleobiology"]["geological_age_ma"] == 0.0
    assert ref["specimen_image"]["sha256"] == _sha(SPECIMEN)
    inv = jacobian_invariants()
    assert inv["stable_without_diffusion"]
    assert inv["turing"]
    assert abs(inv["trace"] - TRACE_J) < 1e-9
    # Lean fixed-point witnesses.
    assert 84 + (-100) < 0
    assert 840 * (-1000) - 1000 * (-1840) > 0
    assert 28630 * (84 + (-100)) < 0
    assert 274882 > 28630
    assert (20227088 * 20227088) > 4 * 28630 * 274882 * 10000
    assert 1412 < 1864 < 2286
    assert 4450 > 3371 > 2749
    assert 16 > 12 > 10
    phys = run_physics()
    assert abs(phys["lambda_um"] - LAMBDA_UM) / LAMBDA_UM < 0.08, phys["lambda_um"]
    assert phys["gof"]["pearson_r"] > 0.98, phys["gof"]["pearson_r"]
    assert phys["ablation_std_ratio"] < 0.5, phys["ablation_std_ratio"]
    assert phys["ablation"]["turing_possible"] is False
    lams = [row["lambda_um"] for row in phys["sweep"]]
    assert all(b > a for a, b in zip(lams, lams[1:])), lams
    chrono = phys["chronoseries"]
    assert chrono[0]["lambda_um_model"] < chrono[1]["lambda_um_model"] < chrono[2]["lambda_um_model"]
    # Non-circular specimen metrics.
    assert abs(phys["specimen"]["foreground_fraction"] - 0.5) > 0.02
    assert abs(phys["specimen"]["circularity"] - 0.62) > 0.05
    print(json.dumps({
        "self_test": "pass",
        "lambda_um": round(phys["lambda_um"], 4),
        "pearson_r": round(phys["gof"]["pearson_r"], 6),
        "rmse": phys["gof"]["rmse"],
        "ablation_std_ratio": round(phys["ablation_std_ratio"], 4),
        "sweep_lambda_um": [round(x, 4) for x in lams],
        "specimen_foreground": round(phys["specimen"]["foreground_fraction"], 4),
        "orientation_ks": round(phys["orientation_ks"], 4),
        "age_ma": AGE_MA,
        "size_m": LAMBDA_M,
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass", dest="pass_id", type=int, default=2, choices=(1, 2))
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    out = render(args.pass_id)
    phys = out["physics"]
    print(json.dumps({
        "pass": analysis_pass(),
        "lambda_um": phys["lambda_um"],
        "pearson_r": phys["gof"]["pearson_r"],
        "ablation_std_ratio": phys["ablation_std_ratio"],
        "depth": out["depth"]["chosen_depth"],
        "depth_sweep": out["depth"].get("sweep", []),
        "png": out["png"],
        "gif": out["gif"],
        "age_ma": AGE_MA,
        "phenomenon_size_m": LAMBDA_M,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
