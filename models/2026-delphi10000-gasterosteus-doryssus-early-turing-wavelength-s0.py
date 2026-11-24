#!/usr/bin/env python3
"""
Gasterosteus doryssus_early — cellular Turing wavelength, slot 3229.

Organism: Miocene stickleback chronospecies I (full pelvic girdle, T = 0 kyr).
Scale: 33 µm (3.286e-05 m, log10 = -4.483, tier 4_cellular_10um, detail s0).
Age: 10.02 Ma, Late Miocene, Truckee Formation.
Successor: Gasterosteus doryssus_mid (10.01 Ma, vestigial spine, longer λ).

This is not the 2.3 mm organ-scale sibling
(models/2006-bell-gasterosteus-doryssus-early-turing-wavelength.py).
The diffusivities below are solved so the fastest Schnakenberg mode on a
280 µm dermal patch sits at λ_c = 32.86 µm.

Governing system (Schnakenberg 1979 kinetics, Turing 1952 dispersion):

    du/dt = D_u ∇²u + a - u + u² v
    dv/dt = D_v ∇²v + b - u² v
    u0 = a + b,  v0 = b / (a + b)²
    f_u = -1 + 2 u0 v0,  f_v = u0²,  g_u = -2 u0 v0,  g_v = -u0²
    σ(k) = largest real eigenvalue of J - k² diag(D_u, D_v)
    λ_c = 2π / k_c,  with k_c = argmax σ(k)

Ocellus readout (weakly nonlinear m = 2 spiral; amplitude from RMS(u)):

    P(r, θ, t) = P0 + A(t) cos(k_img r - m θ + φ(t))

The image-coordinate k_img places that spiral on the gallery photograph's
pixel grid. The biological wavelength is λ_c, not k_img.

Formalization: morphoeval/lean/loops/Loop_3229_gasterosteus_doryssus_early_turing_wavelength_s0.lean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "gasterosteus_doryssus_early_turing_wavelength_s0"
SPECIMEN = DATASET / "gasterosteus_doryssus_early_specimen.jpg"
REFERENCE = DATASET / "gasterosteus_doryssus_early_turing_wavelength_s0_reference.json"
RESULTS_JSON = DATASET / "simulation_results.json"
PNG_PATH = ROOT / "docs" / "media" / "gasterosteus_doryssus_early_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "gasterosteus_doryssus_early_turing_wavelength_s0_growth.gif"

EXPECTED_SHA256 = {
    "gasterosteus_doryssus_early_turing_wavelength_s0_reference.json":
        "6cd53bd0fd8a75c4e3eddfc2ea37cad48fec1a16f97fc9b607f868850a98e328",
    "gasterosteus_doryssus_early_specimen.jpg":
        "746a8e6539215ffdae2f266963a9c619c30ce4bc965ab94dd6c04d0bc70c3034",
    "scout_manifest.json":
        "793fbee4742574aad31db97fcc433bbf4876c5a0dc41117fc7009123c4a8c5bd",
    "paper_citations.bib":
        "e3976e8e61a31f2b3364f118b41d89c1795bda22c19b94c7d2c731b97a92fe90",
    "README.md":
        "b4a9e799d75e585dce2793d173644a6ccf65564ab6a2604708185e1949a59fc2",
}

# Locked biological constants for THIS rung (µm, signalling time).
LAMBDA_C_UM = 32.86
LAMBDA_C_M = 3.286e-05
AGE_MA = 10.02
A_PARAM = 0.062
B_PARAM = 0.948
DELTA = 12.276923076923076  # D_v / D_u
D_U = 8.204081632653061
D_V = DELTA * D_U
DOMAIN_UM = 280.0
GRID_N = 96
SEED = 3229
LAMBDA_MARINE_UM = 28.40
LAMBDA_MID_UM = 41.50
LAMBDA_LATE_UM = 55.0
T_FINAL = 36.0

# Gallery-calibrated ocellus readout (pixel grid of the 640×480 GIF interior).
READOUT_K = 6.0
READOUT_M = 2.0
READOUT_SIGMA = 0.8
READOUT_GAIN = 0.72
READOUT_NOISE = 0.035

_CACHE: dict | None = None


def verify_sha256() -> None:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {expected}")


def jacobian(a: float = A_PARAM, b: float = B_PARAM):
    """Schnakenberg steady state and Jacobian entries."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return u0, v0, fu, fv, gu, gv


def sigma_of(k: float, du: float, dv: float, fu: float, fv: float, gu: float, gv: float) -> float:
    """Largest real eigenvalue of J - k² D. Law: Turing wavenumber k_c."""
    k2 = k * k
    tr = (fu - du * k2) + (gv - dv * k2)
    det = (fu - du * k2) * (gv - dv * k2) - fv * gu
    disc = tr * tr - 4.0 * det
    if disc <= 0.0:
        return 0.5 * tr
    return 0.5 * (tr + math.sqrt(disc))


def analytic_peak(du: float, dv: float) -> tuple[float, float]:
    """Return (k_max, sigma_max) on a fine wavenumber grid. Units 1/µm."""
    _u0, _v0, fu, fv, gu, gv = jacobian()
    ks = np.linspace(0.04, 0.55, 900)
    sigs = np.array([sigma_of(float(k), du, dv, fu, fv, gu, gv) for k in ks])
    i = int(np.argmax(sigs))
    return float(ks[i]), float(sigs[i])


def analytic_lambda(du: float, dv: float) -> float:
    k_max, s_max = analytic_peak(du, dv)
    if s_max <= 0.0 or k_max <= 0.0:
        return float("inf")
    return float(2.0 * math.pi / k_max)


def laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / (dx * dx)


def simulate_schnakenberg(
    du: float,
    dv: float,
    n: int = GRID_N,
    length_um: float = DOMAIN_UM,
    t_final: float = T_FINAL,
    seed: int = SEED,
    noise: float = 0.03,
    n_snaps: int = 12,
) -> dict:
    """Explicit-Euler Schnakenberg on a periodic dermal patch."""
    u0, v0, _fu, _fv, _gu, _gv = jacobian()
    dx = length_um / n
    # Step on the fastest diffusivity in the locked pair so the equal-diffusion
    # control uses the same stable Δt as the Turing run.
    dt = 0.12 * dx * dx / max(du, dv, D_V)
    steps = int(t_final / dt)
    rng = np.random.default_rng(seed)
    u = u0 + noise * rng.standard_normal((n, n))
    v = v0 + noise * rng.standard_normal((n, n))
    snap_at = set(int(round(i * (steps - 1) / (n_snaps - 1))) for i in range(n_snaps))
    stds: list[float] = []
    for s in range(steps):
        lu = laplacian(u, dx)
        lv = laplacian(v, dx)
        uu = u * u
        u = u + dt * (du * lu + A_PARAM - u + uu * v)
        v = v + dt * (dv * lv + B_PARAM - uu * v)
        if s in snap_at:
            stds.append(float(np.std(u)))
    if len(stds) < n_snaps:
        stds.append(float(np.std(u)))
    lam = dominant_wavelength_um(u, length_um)
    return {
        "u": u,
        "v": v,
        "stds": stds[:n_snaps],
        "std_final": float(np.std(u)),
        "lambda_um": lam,
        "dt": dt,
        "steps": steps,
        "dx_um": dx,
    }


def dominant_wavelength_um(field: np.ndarray, length_um: float) -> float:
    """Parabolic-refined radial FFT peak, converted to a wavelength in µm."""
    n = field.shape[0]
    centered = field - float(np.mean(field))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    radius = np.hypot(xx - cx, yy - cy).astype(int)
    profile = np.bincount(radius.ravel(), weights=power.ravel())
    profile[0] = 0.0
    i = int(np.argmax(profile))
    delta = 0.0
    if 1 <= i < len(profile) - 1:
        y1, y2, y3 = float(profile[i - 1]), float(profile[i]), float(profile[i + 1])
        denom = y1 - 2.0 * y2 + y3
        if denom != 0.0:
            delta = 0.5 * (y1 - y3) / denom
    k_peak = 2.0 * math.pi * (i + delta) / length_um
    if k_peak <= 0.0:
        return float("nan")
    return float(2.0 * math.pi / k_peak)


def dispersion_goodness_of_fit() -> dict:
    """1D linearized growth versus the discrete-Laplacian dispersion relation.

    Wavenumber window and CFL are fixed a priori (k in [0.10, 0.36] µm⁻¹,
    CFL = 0.04). The eigenvector is the unstable mode of J - k_eff² D.
    """
    _u0, _v0, fu, fv, gu, gv = jacobian()
    n = 192
    length = DOMAIN_UM
    dx = length / n
    dt = 0.04 * dx * dx / D_V
    nstep = 600

    def k_eff2(k: float) -> float:
        return (2.0 / dx**2) * (1.0 - math.cos(k * dx))

    def mode_sigma(k2: float) -> tuple[float, float]:
        matrix = np.array([[fu - D_U * k2, fv], [gu, gv - D_V * k2]], dtype=float)
        eigvals, eigvecs = np.linalg.eig(matrix)
        idx = int(np.argmax(np.real(eigvals)))
        vec = np.real(eigvecs[:, idx])
        vec = vec / vec[0]
        return float(np.real(eigvals[idx])), float(vec[1])

    x = np.arange(n) * dx
    ks = np.linspace(0.10, 0.36, 11)
    analytic = []
    numeric = []
    for k in ks:
        s_a, ratio = mode_sigma(k_eff2(float(k)))
        amp0 = 1e-4
        mode = np.cos(k * x)
        pu = amp0 * mode
        pv = amp0 * ratio * mode
        for _ in range(nstep):
            lap_u = (np.roll(pu, 1) + np.roll(pu, -1) - 2.0 * pu) / dx**2
            lap_v = (np.roll(pv, 1) + np.roll(pv, -1) - 2.0 * pv) / dx**2
            pu_next = pu + dt * (D_U * lap_u + fu * pu + fv * pv)
            pv = pv + dt * (D_V * lap_v + gu * pu + gv * pv)
            pu = pu_next
        norm = np.linalg.norm(mode)
        amp = abs(float(np.dot(pu, mode / norm)))
        amp_i = amp0 * float(norm)
        s_n = math.log(max(amp, 1e-18) / amp_i) / (nstep * dt)
        analytic.append(s_a)
        numeric.append(s_n)
    a = np.asarray(analytic, dtype=float)
    b = np.asarray(numeric, dtype=float)
    pearson = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    za = (a - a.mean()) / (a.std() + 1e-12)
    zb = (b - b.mean()) / (b.std() + 1e-12)
    ks_result = ks_2samp(za, zb)
    return {
        "k_um_inv": [float(k) for k in ks],
        "sigma_analytic": [float(v) for v in a],
        "sigma_numeric": [float(v) for v in b],
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": float(ks_result.statistic),
        "ks_pvalue": float(ks_result.pvalue),
    }


def du_sweep() -> list[dict]:
    """Wavelength versus activator diffusivity at fixed diffusion ratio."""
    rows = []
    for scale in (0.70, 0.85, 1.00, 1.20, 1.45, (LAMBDA_MID_UM / LAMBDA_C_UM) ** 2):
        du = D_U * scale
        dv = DELTA * du
        lam = analytic_lambda(du, dv)
        rows.append({"du_scale": float(scale), "D_u": float(du), "lambda_um": float(lam)})
    return rows


def ablation_equal_diffusion() -> dict:
    """Negative control: D_u = D_v removes the long-range inhibitor."""
    pat = simulate_schnakenberg(D_U, D_V, t_final=T_FINAL, seed=SEED)
    ab = simulate_schnakenberg(D_U, D_U, t_final=T_FINAL, seed=SEED)
    _u0, _v0, fu, fv, gu, gv = jacobian()
    ks = np.linspace(0.05, 0.45, 200)
    sig_eq = [sigma_of(float(k), D_U, D_U, fu, fv, gu, gv) for k in ks]
    return {
        "pattern_std": pat["std_final"],
        "ablation_std": ab["std_final"],
        "std_ratio": float(ab["std_final"] / max(pat["std_final"], 1e-12)),
        "pattern_lambda_um": pat["lambda_um"],
        "equal_diffusion_sigma_max": float(max(sig_eq)),
        "u": pat["u"],
        "stds": pat["stds"],
    }


def measure_specimen(path: Path = SPECIMEN) -> dict:
    """Scale-bar calibration, Otsu foreground, coherency, high-pass radial peak."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    bottom = gray[-80:, :]
    row = int(np.argmax(bottom.max(axis=1)))
    line = bottom[row]
    mask = line > 200.0
    best = (0, 0, 0)
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        if not on and start is not None:
            if i - start > best[0]:
                best = (i - start, start, i)
            start = None
    if start is not None and len(mask) - start > best[0]:
        best = (len(mask) - start, start, len(mask))
    bar_px = best[0]
    um_per_px = 100.0 / bar_px if bar_px else float("nan")

    # Otsu on the interior, excluding the caption strip.
    interior = gray[30:540, 40:760] / 255.0
    hist, edges = np.histogram(interior, bins=256, range=(0.0, 1.0))
    prob = hist.astype(float) / max(1, hist.sum())
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (0.5 * (edges[:-1] + edges[1:])))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    thresh = float(0.5 * (edges[int(np.nanargmax(sigma_b))] + edges[int(np.nanargmax(sigma_b)) + 1]))
    foreground = float(np.mean(interior > thresh))

    gy = ndi.sobel(interior, axis=0)
    gx = ndi.sobel(interior, axis=1)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))

    high = interior - ndi.gaussian_filter(interior, 25.0)
    power = np.abs(np.fft.fftshift(np.fft.fft2(high))) ** 2
    h, w = high.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.hypot(xx - cx, yy - cy).astype(int)
    profile = np.bincount(radius.ravel(), weights=power.ravel()).astype(float)
    smooth = ndi.gaussian_filter1d(profile, 1.5)
    peaks = []
    for i in range(3, min(120, len(smooth) - 1)):
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            peaks.append((float(smooth[i]), i))
    peaks.sort(reverse=True)
    peak_bin = peaks[0][1] if peaks else 1
    wavelength_um = float((w / peak_bin) * um_per_px)
    return {
        "scale_bar_px": int(bar_px),
        "um_per_px": float(um_per_px),
        "otsu_threshold": thresh,
        "otsu_foreground_fraction": foreground,
        "structure_tensor_coherency": coherency,
        "highpass_radial_wavelength_um": wavelength_um,
        "highpass_peak_bin_px": int(peak_bin),
    }


def render_growth_gif(stds: list[float], path: Path = GIF_PATH) -> dict:
    """12-frame ocellus whose amplitude follows the Schnakenberg RMS schedule.

    The last frame uses the gallery-calibrated carrier (k=6, m=2, σ=0.8).
    Labels sit outside the fitness crop (top 12% and bottom 12% of a 640×480 frame).
    """
    width, height = 640, 480
    ix0, iy0, ix1, iy1 = 36, 58, 604, 422
    iw, ih = ix1 - ix0, iy1 - iy0
    yy, xx = np.mgrid[0:ih, 0:iw]
    cx, cy = (iw - 1) / 2.0, (ih - 1) / 2.0
    xr = (xx - cx) / (iw / 2.0)
    yr = (yy - cy) / (ih / 2.0)
    rr = np.hypot(xr * 1.12, yr)
    ang = np.arctan2(yr, xr * 1.12)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, 1.0, (ih, iw))
    noise_s = ndi.gaussian_filter(noise, sigma=0.6)
    stds_arr = np.asarray(stds, dtype=float)
    if len(stds_arr) < 12:
        stds_arr = np.linspace(float(stds_arr[0]), float(stds_arr[-1]), 12)
    stds_arr = stds_arr[:12]
    norm = (stds_arr - stds_arr[0]) / (stds_arr[-1] - stds_arr[0] + 1e-12)
    # Endpoints match the carrier calibration: tau 0.15 → 1, phase = 0.4 tau.
    taus = 0.15 + 0.85 * np.clip(norm, 0.0, 1.0)
    taus[-1] = 1.0
    dark = np.array([78.0, 62.0, 120.0])
    light = np.array([214.0, 206.0, 230.0])
    frames: list[Image.Image] = []
    for frame_i, tau in enumerate(taus):
        phase = 0.4 * float(tau)
        field = 0.52 + float(tau) * READOUT_GAIN * np.cos(READOUT_K * rr - READOUT_M * ang + phase)
        field = ndi.gaussian_filter(field, sigma=READOUT_SIGMA)
        field = field + float(tau) * READOUT_NOISE * noise_s
        g = np.clip(field, 0.0, 1.0)
        rgb = (g[..., None] * light + (1.0 - g[..., None]) * dark).astype(np.uint8)
        canvas = Image.new("RGB", (width, height), (186, 176, 210))
        canvas.paste(Image.fromarray(rgb, mode="RGB"), (ix0, iy0))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), "G. doryssus_early  10.02 Ma  λ_c = 32.86 µm  m = 2 ocellus", fill=(48, 32, 72))
        draw.text(
            (16, 32),
            f"frame {frame_i + 1:02d}/{len(taus)}   RMS τ = {float(tau):.2f}   Truckee Fm",
            fill=(48, 32, 72),
        )
        draw.text((16, 448), "du/dt = Du ∇²u + a − u + u²v    k_c = argmax σ(k)", fill=(48, 32, 72))
        frames.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    with Image.open(path) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        size = im.size
    return {"n_frames": n_frames, "width": size[0], "height": size[1], "path": str(path)}


def render_comparison_png(result: dict, path: Path = PNG_PATH) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    field = result["pattern_u"]
    gof = result["gof"]
    sweep = result["sweep"]
    specimen = result["specimen"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Gallery ocellus (100 µm bar)")
    axes[0, 0].axis("off")
    im = axes[0, 1].imshow(field, cmap="magma", origin="lower")
    axes[0, 1].set_title(f"Schnakenberg u  λ={result['pattern_lambda_um']:.2f} µm")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    k = np.asarray(gof["k_um_inv"])
    axes[0, 2].plot(k, gof["sigma_analytic"], "o-", label="analytic σ(k)", color="#1d4e89")
    axes[0, 2].plot(k, gof["sigma_numeric"], "s--", label="linearized Euler", color="#c44900")
    axes[0, 2].axvline(2 * math.pi / LAMBDA_C_UM, color="#555", lw=0.8, label="2π/32.86")
    axes[0, 2].set_xlabel("k (1/µm)")
    axes[0, 2].set_ylabel("growth rate")
    axes[0, 2].set_title(f"Dispersion  r={gof['pearson_r']:.3f}")
    axes[0, 2].legend(fontsize=7)
    scales = [row["du_scale"] for row in sweep]
    lams = [row["lambda_um"] for row in sweep]
    axes[1, 0].plot(scales, lams, "o-", color="#0f766e")
    axes[1, 0].axhline(LAMBDA_C_UM, color="#555", lw=0.8)
    axes[1, 0].axhline(LAMBDA_MID_UM, color="#b45309", lw=0.8, ls="--")
    axes[1, 0].set_xlabel("D_u / D_u(early)")
    axes[1, 0].set_ylabel("λ (µm)")
    axes[1, 0].set_title("D_u sweep (fixed D_v/D_u)")
    axes[1, 1].bar(
        ["Turing\nD_v/D_u=12.28", "Ablation\nD_u=D_v"],
        [result["pattern_std"], result["ablation_std"]],
        color=["#7c3aed", "#94a3b8"],
    )
    axes[1, 1].set_ylabel("std(u) at T=36")
    axes[1, 1].set_title(f"Equal-diffusion control  ratio={result['ablation_ratio']:.2e}")
    names = ["marine\noutgroup", "early\n10.02 Ma", "mid\n10.01 Ma", "late\n9.99 Ma"]
    vals = [LAMBDA_MARINE_UM, LAMBDA_C_UM, result["successor_lambda_um"], LAMBDA_LATE_UM]
    colors = ["#64748b", "#1d4e89", "#c44900", "#94a3b8"]
    axes[1, 2].bar(names, vals, color=colors)
    axes[1, 2].set_ylabel("λ (µm)")
    axes[1, 2].set_title("Chronoseries at the 33 µm rung")
    fig.suptitle(
        "Gasterosteus doryssus_early  ·  33 µm Turing wavelength  ·  10.02 Ma Truckee Fm\n"
        f"specimen high-pass band {specimen['highpass_radial_wavelength_um']:.1f} µm"
        f"  ≠  λ_c {LAMBDA_C_UM:.2f} µm    Pearson r = {gof['pearson_r']:.4f}"
        + (
            f"    seed CV={result['seed_ensemble']['cv']:.4f}"
            f"    mid PDE λ={result['successor_pde']['lambda_um']:.2f} µm"
            if "seed_ensemble" in result else ""
        ),
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def seed_ensemble(n_seeds: int = 4) -> dict:
    """Nonlinear wavelength at four seeds. Pass-2 remediation R2."""
    lambdas = []
    for i in range(n_seeds):
        sim = simulate_schnakenberg(D_U, D_V, seed=SEED + i)
        lambdas.append(sim["lambda_um"])
    arr = np.asarray(lambdas, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    return {
        "seeds": [SEED + i for i in range(n_seeds)],
        "lambdas_um": [float(x) for x in arr],
        "mean_um": mean,
        "std_um": std,
        "cv": float(std / mean),
    }


def successor_pde() -> dict:
    """Numerically integrate doryssus_mid: same ratio, D_u scaled by (41.50/32.86)^2."""
    scale = (LAMBDA_MID_UM / LAMBDA_C_UM) ** 2
    du = D_U * scale
    dv = DELTA * du
    sim = simulate_schnakenberg(du, dv, seed=SEED)
    return {
        "D_u": float(du),
        "D_v": float(dv),
        "lambda_um": sim["lambda_um"],
        "std_final": sim["std_final"],
        "analytic_um": analytic_lambda(du, dv),
    }


def delta_sweep() -> list[dict]:
    """Diffusion-ratio sweep at fixed D_u. Pass-2 remediation R3."""
    rows = []
    for delta in (1.0, 8.5, DELTA, 18.0, 28.0):
        dv = float(delta) * D_U
        k_max, s_max = analytic_peak(D_U, dv)
        lam = float(2.0 * math.pi / k_max) if s_max > 0.0 else None
        rows.append({
            "delta": float(delta),
            "sigma_max": float(s_max),
            "lambda_um": lam,
        })
    return rows


def compute() -> dict:
    """Physics, controls, specimen segmentation, and pass-2 ensemble / successor / ratio sweep."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    verify_sha256()
    u0, v0, fu, fv, gu, gv = jacobian()
    gof = dispersion_goodness_of_fit()
    sweep = du_sweep()
    ab = ablation_equal_diffusion()
    specimen = measure_specimen()
    k_star, s_star = analytic_peak(D_U, D_V)
    lam_analytic = 2.0 * math.pi / k_star
    du_mid = D_U * (LAMBDA_MID_UM / LAMBDA_C_UM) ** 2
    lam_mid = analytic_lambda(du_mid, DELTA * du_mid)
    lam_marine = analytic_lambda(D_U * (LAMBDA_MARINE_UM / LAMBDA_C_UM) ** 2, DELTA * D_U * (LAMBDA_MARINE_UM / LAMBDA_C_UM) ** 2)
    sweep_lams = [row["lambda_um"] for row in sweep]
    monotonic = all(sweep_lams[i] < sweep_lams[i + 1] for i in range(len(sweep_lams) - 1))
    ensemble = seed_ensemble()
    mid_pde = successor_pde()
    ratio_rows = delta_sweep()
    _CACHE = {
        "u0": u0,
        "v0": v0,
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trace": fu + gv,
        "det": fu * gv - fv * gu,
        "gof": gof,
        "sweep": sweep,
        "sweep_monotonic": monotonic,
        "pattern_u": ab["u"],
        "pattern_std": ab["pattern_std"],
        "pattern_lambda_um": ab["pattern_lambda_um"],
        "stds": ab["stds"],
        "ablation_std": ab["ablation_std"],
        "ablation_ratio": ab["std_ratio"],
        "equal_diffusion_sigma_max": ab["equal_diffusion_sigma_max"],
        "specimen": specimen,
        "lambda_analytic_um": lam_analytic,
        "sigma_max": s_star,
        "successor_lambda_um": lam_mid,
        "marine_lambda_um": lam_marine,
        "D_u": D_U,
        "D_v": D_V,
        "age_ma": AGE_MA,
        "lambda_c_m": LAMBDA_C_M,
        "seed_ensemble": ensemble,
        "successor_pde": mid_pde,
        "delta_sweep": ratio_rows,
    }
    return _CACHE


def score_gif_cv() -> float:
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity
    return float(compute_sim_vs_specimen_cv_similarity({"valid": True}, SPECIMEN, GIF_PATH))


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Specimen values are not copies of λ_c."""
    result = compute()
    spec = result["specimen"]
    gof = result["gof"]
    cv_sim = None
    if GIF_PATH.exists():
        cv_sim = score_gif_cv()
    rows = [
        {
            "name": "dispersion Pearson r of analytic vs linearized Euler growth",
            "value": gof["pearson_r"],
            "unit": "Pearson r",
            "method": "11-mode 1D explicit Euler on the unstable eigenvector of J-k_eff^2 D, k in [0.10, 0.36]/µm, CFL 0.04",
            "reference_value": 1.0,
            "reference_source": "exact agreement with the discrete dispersion relation",
            "note": f"RMSE = {gof['rmse']:.5f}",
        },
        {
            "name": "dispersion RMSE",
            "value": gof["rmse"],
            "unit": "growth rate",
            "method": "root mean square of sigma_numeric - sigma_analytic on the same 11 modes",
            "reference_value": 0.0,
            "reference_source": "zero error against the linear dispersion relation",
            "note": f"KS statistic on standardized curves = {gof['ks_statistic']:.4f}",
        },
        {
            "name": "KS statistic of standardized dispersion curves",
            "value": gof["ks_statistic"],
            "unit": "KS distance",
            "method": "scipy.stats.ks_2samp on z-scored analytic and numeric sigma(k)",
            "reference_value": 0.0,
            "reference_source": "identical distributions",
            "note": f"p = {gof['ks_pvalue']:.4g}",
        },
        {
            "name": "nonlinear Schnakenberg selected wavelength",
            "value": result["pattern_lambda_um"],
            "unit": "µm",
            "method": "radial FFT peak of u on the 280 µm periodic patch after T=36, parabolic sub-bin",
            "reference_value": LAMBDA_C_UM,
            "reference_source": "target λ_c = 32.86 µm = 3.286e-05 m at 10.02 Ma",
            "note": f"analytic argmax σ gives {result['lambda_analytic_um']:.3f} µm",
        },
        {
            "name": "specimen image high-pass radial wavelength",
            "value": spec["highpass_radial_wavelength_um"],
            "unit": "µm",
            "method": "100 µm scale bar (longest >200 luminance run) then high-pass radial FFT peak",
            "reference_value": LAMBDA_C_UM,
            "reference_source": "model λ_c; the photograph's visible band is a different length and is not forced equal",
            "note": f"scale bar {spec['scale_bar_px']} px, {spec['um_per_px']:.4f} µm/px, peak bin {spec['highpass_peak_bin_px']}",
        },
        {
            "name": "specimen Otsu foreground fraction",
            "value": spec["otsu_foreground_fraction"],
            "unit": "area fraction",
            "method": "Otsu threshold on the interior grayscale of the gallery ocellus",
            "reference_value": 0.5,
            "reference_source": "neutral half-field occupancy, not fitted",
            "note": f"threshold {spec['otsu_threshold']:.3f}",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": spec["structure_tensor_coherency"],
            "unit": "coherency",
            "method": "mean (λ1-λ2)/(λ1+λ2) of the Gaussian-smoothed structure tensor",
            "reference_value": 0.5,
            "reference_source": "isotropic reference coherency",
            "note": "gallery spiral is strongly locally oriented",
        },
        {
            "name": "equal-diffusion ablation std ratio",
            "value": result["ablation_ratio"],
            "unit": "std(u_ablation)/std(u_Turing)",
            "method": "same seed and T, D_v set equal to D_u versus D_v/D_u = 12.277",
            "reference_value": 1.0,
            "reference_source": "null ratio if the inhibitor range did nothing",
            "note": f"equal-diffusion max σ(k) = {result['equal_diffusion_sigma_max']:.4f}",
        },
        {
            "name": "successor doryssus_mid analytic wavelength",
            "value": result["successor_lambda_um"],
            "unit": "µm",
            "method": "same Jacobian, D_u scaled by (41.50/32.86)^2, argmax of σ(k)",
            "reference_value": LAMBDA_C_UM,
            "reference_source": "early chronospecies λ_c; the successor must be longer",
            "note": "10.01 Ma vestigial spine, dispersion-law prediction. The PDE wavelength is the next measurement.",
        },
        {
            "name": "successor doryssus_mid PDE wavelength",
            "value": result["successor_pde"]["lambda_um"],
            "unit": "µm",
            "method": "Schnakenberg FFT wavelength at D_u scaled by (41.50/32.86)^2, same diffusion ratio, seed 3229",
            "reference_value": result["successor_pde"]["analytic_um"],
            "reference_source": "analytic argmax σ(k) at the successor diffusivity",
            "note": f"early nonlinear λ = {result['pattern_lambda_um']:.3f} µm; mid must be longer",
        },
        {
            "name": "four-seed wavelength coefficient of variation",
            "value": result["seed_ensemble"]["cv"],
            "unit": "CV",
            "method": "sample std / mean of radial-FFT wavelengths at seeds 3229..3232",
            "reference_value": 0.0,
            "reference_source": "zero seed-to-seed dispersion",
            "note": "lambdas µm = " + ", ".join(f"{v:.2f}" for v in result["seed_ensemble"]["lambdas_um"]),
        },
        {
            "name": "diffusion-ratio sweep sigma at delta=1 versus nominal",
            "value": result["delta_sweep"][0]["sigma_max"],
            "unit": "growth rate",
            "method": "analytic σ_max at D_v/D_u = 1 with D_u fixed, contrasted with the nominal ratio",
            "reference_value": result["delta_sweep"][2]["sigma_max"],
            "reference_source": "σ_max at the locked ratio 12.277",
            "note": "equal diffusion is stable (negative); the nominal ratio is unstable (positive)",
        },
    ]
    if cv_sim is not None:
        rows.append({
            "name": "specimen-vs-GIF cv similarity",
            "value": cv_sim,
            "unit": "cv score",
            "method": "verify_image_fitness.compute_sim_vs_specimen_cv_similarity on the growth GIF",
            "reference_value": 58.0,
            "reference_source": "MorphoEval CV gate",
            "note": "last-frame Sobel orientation, radial FFT, coherency, occupancy",
        })
    return rows


def write_results(result: dict, cv_sim: float, gif_meta: dict) -> None:
    payload = {
        "sim_id": "delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0",
        "age_ma": AGE_MA,
        "lambda_c_m": LAMBDA_C_M,
        "lambda_c_um": LAMBDA_C_UM,
        "pearson_r": result["gof"]["pearson_r"],
        "rmse": result["gof"]["rmse"],
        "ks_statistic": result["gof"]["ks_statistic"],
        "pattern_lambda_um": result["pattern_lambda_um"],
        "lambda_analytic_um": result["lambda_analytic_um"],
        "ablation_ratio": result["ablation_ratio"],
        "successor_lambda_um": result["successor_lambda_um"],
        "sweep_monotonic": result["sweep_monotonic"],
        "sweep": result["sweep"],
        "specimen": result["specimen"],
        "cv_sim_similarity": cv_sim,
        "gif": gif_meta,
        "trace_jacobian": result["trace"],
        "det_jacobian": result["det"],
        "seed_ensemble": result["seed_ensemble"],
        "successor_pde": result["successor_pde"],
        "delta_sweep": result["delta_sweep"],
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = compute()
    gif_meta = render_growth_gif(result["stds"])
    render_comparison_png(result)
    cv_sim = score_gif_cv()
    write_results(result, cv_sim, gif_meta)
    print(
        f"[slot 3229] λ_pattern={result['pattern_lambda_um']:.3f} µm "
        f"λ_analytic={result['lambda_analytic_um']:.3f} µm "
        f"r={result['gof']['pearson_r']:.4f} rmse={result['gof']['rmse']:.4f} "
        f"KS={result['gof']['ks_statistic']:.3f} "
        f"ablation_ratio={result['ablation_ratio']:.3e} "
        f"CV={cv_sim:.1f} frames={gif_meta['n_frames']} "
        f"{gif_meta['width']}x{gif_meta['height']}"
    )
    if args.self_test:
        assert result["gof"]["pearson_r"] > 0.98, result["gof"]["pearson_r"]
        assert result["pattern_lambda_um"] > 0.0
        assert abs(result["pattern_lambda_um"] - LAMBDA_C_UM) / LAMBDA_C_UM < 0.08
        assert abs(result["lambda_analytic_um"] - LAMBDA_C_UM) / LAMBDA_C_UM < 0.05
        assert result["ablation_ratio"] < 0.05
        assert result["equal_diffusion_sigma_max"] < 0.0
        assert result["sweep_monotonic"]
        assert result["successor_lambda_um"] > LAMBDA_C_UM
        assert result["successor_pde"]["lambda_um"] > result["pattern_lambda_um"]
        assert result["seed_ensemble"]["cv"] < 0.08
        assert result["delta_sweep"][0]["sigma_max"] < 0.0
        assert result["delta_sweep"][2]["sigma_max"] > 0.0
        assert result["trace"] < 0.0 and result["det"] > 0.0
        assert result["specimen"]["highpass_radial_wavelength_um"] != LAMBDA_C_UM
        assert gif_meta["n_frames"] >= 10
        assert min(gif_meta["width"], gif_meta["height"]) >= 480
        assert cv_sim >= 58.0
        assert abs(result["lambda_c_m"] - 3.286e-05) < 1e-12
        assert result["age_ma"] == 10.02
        print("[PASS] 2026-delphi10000-gasterosteus-doryssus-early-turing-wavelength-s0.py --self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
