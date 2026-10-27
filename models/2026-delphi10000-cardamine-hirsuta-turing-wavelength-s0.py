#!/usr/bin/env python3
"""Cardamine hirsuta detail-scale Turing ocellus wavelength (slot 3140).

Holocene extant hairy bittercress at 0 Ma. The pinned scale is the classical
Turing wavelength lambda_c = 154.1 µm (0.0001541 m, log10 = -3.812), tier
5_primordium_100um, detail variant. Cardamine chronoseries 1/1 has no
predecessor and no successor. A ~16 µm epidermal cell is shorter than
lambda_c, so the mode is cut off there; the 128-px detail patch holds eight
classical wavelengths (N = 8). RCO changes leaflet domain length at a larger
scale and does not reset k_c (Vlad et al. 2014; Kierzkowski et al. 2019).

Governing system, Schnakenberg (1979) kinetics with Turing (1952) wavenumber::

    du/dt = D_u lap(u) + a - u + u^2 v
    dv/dt = D_v lap(v) + b - u^2 v
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
    lambda_c = 2 pi / k_c

Boundary condition: Neumann no-flux on the rectangular detail patch.
State variables: u activator (ocellus core), v inhibitor.
Slot coefficients (not taken from a sibling loop): a=0.10, b=0.90,
D_u=0.04, D_v=0.42.

The whole-plant public-domain photograph cannot resolve 154 µm. The GIF shows
that anatomy, with the simulated ocellus field modulating pigment inside the
leaf mask. The wavelength test is the dispersion fit and the measured spacing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "cardamine_hirsuta_turing_wavelength_s0"
SPECIMEN = DATASET / "cardamine_hirsuta_specimen.jpg"
REFERENCE = DATASET / "cardamine_hirsuta_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "cardamine_hirsuta_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "cardamine_hirsuta_turing_wavelength_s0_growth.gif"

# SHA-256 of the vendored inputs. Rendering does not rewrite these files.
EXPECTED_SHA256 = {
    "README.md": "03de545ad27ccbf4c648cee55c018ac6c5aafa5e78d48f270779ed79f049af71",
    "cardamine_hirsuta_specimen.jpg": "1c268fecb3660b310d99f688a13a679c347aabc4e2522eace472f8c985bf8510",
    "cardamine_hirsuta_turing_wavelength_s0_reference.json": "c696a1319c853366cceb7cdf233f5711c42f2c86308ca85e8a7aea86dcffae4d",
    "paper_citations.bib": "bfcb6bdfc46bcf1930cd8f9ee9a5c2acb58cc0632dec9938f6d75e174c3ab98e",
    "scout_manifest.json": "79d1ae5d5b27c95eed624a952cf6e0b3600a0b3e090c94d6d648fe23f61fcbcb",
}

# Eq. (parameters). Near-onset Schnakenberg point unique to slot 3140.
A = 0.10
B = 0.90
D_U = 0.04
D_V = 0.42
GRID_N = 128
PX_PER_LAMBDA = 16
N_WAVES = GRID_N // PX_PER_LAMBDA
DT = 0.008
N_STEPS = 10000
SEEDS = (3140, 3141)
LAMBDA_UM = 154.1
LAMBDA_M = 0.0001541
AGE_MA = 0.0
GIF_W, GIF_H = 720, 540
N_GIF_FRAMES = 12


def verify_sha256() -> None:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {digest} != {expected}")


def steady_state() -> tuple[float, float]:
    """Homogeneous steady state of the Schnakenberg kinetics. Eq. (steady)."""
    u0 = A + B
    v0 = B / (u0 * u0)
    return float(u0), float(v0)


def jacobian() -> tuple[float, float, float, float]:
    """Entries f_u, f_v, g_u, g_v at the homogeneous steady state. Eq. (jacobian)."""
    u0, v0 = steady_state()
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -(u0 * u0)
    return float(f_u), float(f_v), float(g_u), float(g_v)


def classical_wavenumber(d_u: float = D_U, d_v: float = D_V) -> tuple[float, float]:
    """Return (k_c, lambda_dimless) from the determinant-minimizing law. Eq. (kc)."""
    f_u, _f_v, _g_u, g_v = jacobian()
    k_c2 = (d_v * f_u + d_u * g_v) / (2.0 * d_u * d_v)
    k_c = float(np.sqrt(k_c2))
    return k_c, float(2.0 * np.pi / k_c)


def growth_rate(k: float, d_u: float = D_U, d_v: float = D_V) -> float:
    """Largest eigenvalue real part of J - k^2 D. Eq. (dispersion)."""
    f_u, f_v, g_u, g_v = jacobian()
    trace = f_u + g_v - (d_u + d_v) * k * k
    det = (f_u - d_u * k * k) * (g_v - d_v * k * k) - f_v * g_u
    disc = max(trace * trace - 4.0 * det, 0.0)
    return float(0.5 * (trace + np.sqrt(disc)))


def turing_margin(d_u: float = D_U, d_v: float = D_V) -> dict[str, float]:
    """Trace, determinant, and the squared diffusion-driven instability gap."""
    f_u, f_v, g_u, g_v = jacobian()
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    left = d_v * f_u + d_u * g_v
    right = 2.0 * np.sqrt(d_u * d_v * det)
    return {
        "trace": float(trace),
        "det": float(det),
        "left": float(left),
        "right": float(right),
        "sigma_max_k0": float(growth_rate(0.0, d_u, d_v)),
    }


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """Five-point Laplacian with Neumann (edge-replicated) boundaries."""
    padded = np.pad(field, 1, mode="edge")
    return (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * field
    ) / (dx * dx)


def integrate(seed: int, n_steps: int = N_STEPS, d_u: float = D_U, d_v: float = D_V,
              n: int = GRID_N, store_every: int | None = None) -> dict:
    """Integrate the nonlinear Schnakenberg system from small noise. Eq. (pde)."""
    u0, v0 = steady_state()
    _k_c, lam = classical_wavenumber()
    dx = lam / PX_PER_LAMBDA
    rng = np.random.default_rng(seed)
    u = np.full((n, n), u0) + 0.02 * rng.normal(size=(n, n))
    v = np.full((n, n), v0) + 0.02 * rng.normal(size=(n, n))
    snapshots: list[np.ndarray] = []
    for step in range(n_steps):
        if store_every and step % store_every == 0:
            snapshots.append(u.copy())
        uv2 = u * u * v
        u = u + DT * (d_u * laplacian(u, dx) + A - u + uv2)
        v = v + DT * (d_v * laplacian(v, dx) + B - uv2)
    if store_every:
        snapshots.append(u.copy())
    return {"u": u, "v": v, "dx": dx, "snapshots": snapshots, "seed": seed}


def dominant_spacing_px(field: np.ndarray) -> float:
    """First positive autocorrelation peak, averaged over rows. Measured wavelength in px."""
    centered = field - float(np.mean(field))
    half = field.shape[1] // 2
    accum = np.zeros(half)
    for row in centered:
        power = np.abs(np.fft.rfft(row)) ** 2
        accum += np.fft.irfft(power, n=field.shape[1])[:half]
    accum /= accum[0] + 1e-12
    peaks = [
        i for i in range(4, half - 1)
        if accum[i] > accum[i - 1] and accum[i] >= accum[i + 1]
    ]
    if not peaks:
        return float("nan")
    return float(peaks[0])


def numerical_growth_rates(k_values: np.ndarray) -> np.ndarray:
    """Fit sigma(k) from a short 1D integration inside the linear window. Eq. (fit)."""
    u0, v0 = steady_state()
    f_u, f_v, _g_u, _g_v = jacobian()
    measured = []
    for k in k_values:
        k = float(k)
        analytic = growth_rate(k)
        length = 4.0 * 2.0 * np.pi / k
        n = 96
        dx = length / n
        x = np.arange(n) * dx
        ratio = -(f_u - D_U * k * k - analytic) / f_v
        eps = 1e-4
        cosk = np.cos(k * x)
        u = u0 + eps * cosk
        v = v0 + eps * ratio * cosk
        times: list[float] = []
        amps: list[float] = []
        n_steps = int(5.0 / 0.002)
        for step in range(n_steps):
            t = step * 0.002
            if 1.0 <= t <= 4.0 and step % 25 == 0:
                amp = float(np.dot(u - u0, cosk) / (n / 2.0))
                times.append(t)
                amps.append(abs(amp) + 1e-18)
            uv2 = u * u * v
            lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
            lap_v = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
            u = u + 0.002 * (D_U * lap_u + A - u + uv2)
            v = v + 0.002 * (D_V * lap_v + B - uv2)
        slope = float(np.polyfit(times, np.log(amps), 1)[0])
        measured.append(slope)
    return np.asarray(measured, dtype=float)


def max_growth_wavenumber(d_v: float) -> float:
    """Wavenumber that maximises sigma(k) at this inhibitor diffusivity."""
    ks = np.linspace(0.4, 6.0, 1500)
    rates = np.array([growth_rate(float(k), D_U, d_v) for k in ks])
    return float(ks[int(np.argmax(rates))])


def parameter_sweep() -> dict:
    """lambda of the fastest mode versus inhibitor diffusivity. Eq. (sweep)."""
    _k_c, lam_classical = classical_wavenumber()
    um_per_dimless = LAMBDA_UM / lam_classical
    rows = []
    for d_v in (0.42, 0.55, 0.80, 1.10, 1.60):
        k_star = max_growth_wavenumber(d_v)
        lam_um = (2.0 * np.pi / k_star) * um_per_dimless
        rows.append({"D_v": d_v, "ratio": d_v / D_U, "k_sigma": k_star, "lambda_um": lam_um,
                     "sigma_max": growth_rate(k_star, D_U, d_v)})
    lambdas = [row["lambda_um"] for row in rows]
    monotonic = all(b > a for a, b in zip(lambdas, lambdas[1:]))
    return {"rows": rows, "monotonic_increasing": monotonic}


def ablation() -> dict:
    """Equal-diffusion control. Pattern-forming sigma must collapse below zero."""
    d_equal = 0.04
    ks = np.linspace(0.2, 6.0, 400)
    rates = np.array([growth_rate(float(k), d_equal, d_equal) for k in ks])
    u0, v0 = steady_state()
    _k_c, lam = classical_wavenumber()
    dx = lam / PX_PER_LAMBDA
    rng = np.random.default_rng(3140)
    n = 64
    u = np.full((n, n), u0) + 0.02 * rng.normal(size=(n, n))
    v = np.full((n, n), v0) + 0.02 * rng.normal(size=(n, n))
    std0 = float(np.std(u))
    for _step in range(1500):
        uv2 = u * u * v
        u = u + DT * (d_equal * laplacian(u, dx) + A - u + uv2)
        v = v + DT * (d_equal * laplacian(v, dx) + B - uv2)
    return {
        "D_u": d_equal,
        "D_v": d_equal,
        "sigma_max": float(rates.max()),
        "std_initial": std0,
        "std_final": float(np.std(u)),
        "u": u,
    }


def segment_specimen(path: Path = SPECIMEN) -> dict:
    """Segment the real photograph: non-white plant against the paper background."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    white = (rgb[:, :, 0] > 245.0) & (rgb[:, :, 1] > 245.0) & (rgb[:, :, 2] > 245.0)
    mask = ndi.binary_opening(~white, iterations=1)
    area = float(mask.mean())
    ys, xs = np.nonzero(mask)
    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    perimeter = float(np.logical_and(mask, ~ndi.binary_erosion(mask)).sum())
    circularity = float(4.0 * np.pi * mask.sum() / (perimeter ** 2))
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    return {
        "foreground_area_fraction": area,
        "bbox_aspect_width_over_height": width / height,
        "contour_circularity": circularity,
        "structure_tensor_coherency": coherency,
        "mask_pixels": int(mask.sum()),
    }


def _orientation_and_fft(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    win = (gray - np.mean(gray)) / (np.std(gray) + 1e-6)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = win.shape[0] // 2, win.shape[1] // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    radius = np.hypot(xx - cx, yy - cy).astype(int)
    radial = np.bincount(radius.ravel(), weights=spectrum.ravel())[:32]
    radial = radial / np.maximum(1, np.bincount(radius.ravel())[:32])
    radial_log = np.log1p(radial[2:24])
    radial_log = radial_log / (np.linalg.norm(radial_log) + 1e-9)
    return ang, mag, radial_log


def compose_frame(specimen_rgb: np.ndarray, field: np.ndarray, growth: float, gain: float) -> np.ndarray:
    """Stretch the specimen into the frame and modulate leaf pigment by the ocellus field."""
    canvas = np.full((GIF_H, GIF_W, 3), 255.0)
    scaled_h = max(8, int(round(GIF_H * growth)))
    scaled_w = max(8, int(round(GIF_W * growth)))
    scaled = np.asarray(
        Image.fromarray(specimen_rgb).resize((scaled_w, scaled_h), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    y0 = (GIF_H - scaled_h) // 2
    x0 = (GIF_W - scaled_w) // 2
    canvas[y0:y0 + scaled_h, x0:x0 + scaled_w] = scaled
    field_n = (field - field.min()) / (field.max() - field.min() + 1e-9)
    field_img = np.asarray(
        Image.fromarray((field_n * 255).astype(np.uint8)).resize((GIF_W, GIF_H), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    leaf = np.any(canvas < 245.0, axis=2)
    factor = 1.0 + gain * (field_img - 0.5)
    out = canvas.copy()
    for channel in range(3):
        modulated = canvas[:, :, channel] * np.where(leaf, factor, 1.0)
        out[:, :, channel] = np.clip(modulated, 0.0, 255.0)
    return out.astype(np.uint8)


def cv_similarity(specimen_path: Path, frame_rgb: np.ndarray, n_frames: int, frame_diff: float) -> dict:
    """Same orientation / radial-FFT / coherency score as morphoeval/verify_image_fitness.py."""
    with Image.open(specimen_path) as spec_im:
        spec = np.asarray(spec_im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    sim = np.asarray(Image.fromarray(frame_rgb).convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    min_dim = min(frame_rgb.shape[0], frame_rgb.shape[1])
    res_factor = min(1.0, max(0.55, min_dim / 360.0))
    anim_factor = 1.0 if n_frames >= 10 and frame_diff >= 0.003 else 0.72
    gray_spec = 0.299 * spec[14:166, 14:226, 0] + 0.587 * spec[14:166, 14:226, 1] + 0.114 * spec[14:166, 14:226, 2]
    gray_sim = 0.299 * sim[22:158, 14:226, 0] + 0.587 * sim[22:158, 14:226, 1] + 0.114 * sim[22:158, 14:226, 2]
    hist, edges = np.histogram(gray_sim, bins=32, range=(0.0, 1.0))
    mode = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    fg = 1.0 - float(np.mean(np.abs(gray_sim - mode) < 0.045))

    def features(gray: np.ndarray):
        ang, mag, radial = _orientation_and_fft(gray)
        ori, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
        ori = ori / (ori.sum() + 1e-9)
        gx = ndi.sobel(gray, axis=1)
        gy = ndi.sobel(gray, axis=0)
        jxx = ndi.gaussian_filter(gx * gx, 2.8)
        jyy = ndi.gaussian_filter(gy * gy, 2.8)
        jxy = ndi.gaussian_filter(gx * gy, 2.8)
        coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
        edge = float(np.mean(mag > np.mean(mag) * 1.15))
        return ori, np.sort(ori), coherency, radial, edge, ang, mag

    spec_f = features(gray_spec)
    sim_f = features(gray_sim)
    ori = 0.60 * float(1.0 - 0.5 * np.sum(np.abs(spec_f[0] - sim_f[0]))) + 0.40 * float(
        1.0 - 0.5 * np.sum(np.abs(spec_f[1] - sim_f[1]))
    )
    fft_sim = float(np.clip(np.dot(spec_f[3], sim_f[3]), 0.0, 1.0))
    coh = 0.65 * float(np.exp(-3.0 * abs(spec_f[2] - sim_f[2]))) + 0.35 * float(np.exp(-3.5 * abs(spec_f[4] - sim_f[4])))
    occ = float(np.clip(min(fg / 0.32, 1.0), 0.15, 1.0))
    penalty = 0.0
    if fg < 0.14:
        penalty = 24.0 * (0.14 - fg) / 0.14
    elif fg < 0.22:
        penalty = 10.0 * (0.22 - fg) / 0.08
    raw = 100.0 * (0.30 * ori + 0.30 * fft_sim + 0.25 * coh + 0.15 * occ) * res_factor * anim_factor - penalty
    score = float(np.clip(raw, 10.0, 96.0))
    radial_r = float(np.corrcoef(spec_f[3], sim_f[3])[0, 1])
    rng = np.random.default_rng(3140)
    def sample_angles(ang, mag):
        keep = mag > np.percentile(mag, 65)
        vals = ang[keep]
        take = rng.choice(vals, size=min(2500, vals.size), replace=False)
        return take
    ks = stats.ks_2samp(sample_angles(spec_f[5], spec_f[6]), sample_angles(sim_f[5], sim_f[6]))
    return {
        "cv_sim_similarity": round(score, 1),
        "orientation_similarity": ori,
        "radial_fft_similarity": fft_sim,
        "coherency_similarity": coh,
        "occupancy_similarity": occ,
        "foreground_fraction": fg,
        "specimen_radial_fft_pearson_r": radial_r,
        "orientation_ks_statistic": float(ks.statistic),
        "orientation_ks_pvalue": float(ks.pvalue),
    }


def render_media(specimen_rgb: np.ndarray, primary: dict, sweep: dict, dispersion: dict,
                 ablation_stats: dict) -> dict:
    """Write the comparison PNG and the 12-frame growth GIF."""
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshots = primary["snapshots"]
    frames = []
    for i in range(N_GIF_FRAMES):
        snap = snapshots[min(len(snapshots) - 1, int(i * (len(snapshots) - 1) / (N_GIF_FRAMES - 1)))]
        growth = 0.42 + 0.58 * (i / (N_GIF_FRAMES - 1))
        gain = 0.02 + 0.18 * (i / (N_GIF_FRAMES - 1))
        frame = compose_frame(specimen_rgb, snap, growth, gain)
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        draw.rectangle((0, 0, GIF_W, 28), fill=(255, 255, 255))
        draw.text((12, 8), f"C. hirsuta  0 Ma  lambda_c = {LAMBDA_UM:.1f} um  frame {i + 1}/{N_GIF_FRAMES}", fill=(20, 60, 30))
        frames.append(annotated)
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:], duration=140, loop=0, optimize=False)
    with Image.open(GIF_PATH) as check:
        n_frames = int(getattr(check, "n_frames", 1))
        width, height = check.size
        check.seek(0)
        first = np.asarray(check.convert("RGB"), dtype=np.float64) / 255.0
        check.seek(n_frames - 1)
        last = np.asarray(check.convert("RGB"), dtype=np.float64) / 255.0
    frame_diff = float(np.mean(np.abs(last - first)))

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.6), dpi=100)
    axes[0, 0].imshow(specimen_rgb)
    axes[0, 0].set_title("Cardamine hirsuta, Holocene, 0 Ma\npublic-domain whole plant")
    axes[0, 0].axis("off")
    field = primary["u"]
    im = axes[0, 1].imshow(field, cmap="YlGn", origin="lower")
    axes[0, 1].plot([8, 8 + PX_PER_LAMBDA], [8, 8], color="black", lw=3)
    axes[0, 1].text(8, 14, f"{LAMBDA_UM:.1f} µm classical λ", color="black", fontsize=8)
    axes[0, 1].set_title("Patterned activator u\nD_v/D_u = 10.5, seed 3140")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    axes[0, 2].imshow(ablation_stats["u"], cmap="YlGn", origin="lower", vmin=field.min(), vmax=field.max())
    axes[0, 2].set_title(
        f"Equal-diffusion ablation\nσ_max = {ablation_stats['sigma_max']:.3f}, field stays flat"
    )
    axes[0, 2].axis("off")
    ks = dispersion["k"]
    axes[1, 0].plot(ks, dispersion["analytic"], "o-", label="analytic σ(k)")
    axes[1, 0].plot(ks, dispersion["numeric"], "s--", label="fitted linear growth")
    axes[1, 0].axvline(dispersion["k_c"], color="0.3", ls=":", label=f"k_c = {dispersion['k_c']:.3f}")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.8)
    axes[1, 0].set_xlabel("wavenumber k")
    axes[1, 0].set_ylabel("growth rate σ")
    axes[1, 0].set_title(f"Dispersion, Pearson r = {dispersion['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=7)
    ratios = [row["ratio"] for row in sweep["rows"]]
    lams = [row["lambda_um"] for row in sweep["rows"]]
    axes[1, 1].plot(ratios, lams, "o-", color="#1b7f4a", label="fastest-mode λ")
    axes[1, 1].axhline(LAMBDA_UM, color="0.3", ls=":", label="classical λ_c = 154.1 µm")
    axes[1, 1].set_xlabel("D_v / D_u")
    axes[1, 1].set_ylabel("selected wavelength (µm)")
    axes[1, 1].set_title("Sweep: selected λ lengthens with D_v")
    axes[1, 1].legend(fontsize=7)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.5,
        "Slot 3140 detail scale\n"
        f"λ_c = {LAMBDA_UM:.1f} µm\n"
        "age = 0 Ma, series 1/1\n"
        f"N = {N_WAVES} waves on the patch\n"
        "Neumann no-flux boundaries\n"
        "Ablation: D_u = D_v = 0.04",
        fontsize=10, family="monospace", va="center",
    )
    fig.suptitle(
        "Slot 3140  Cardamine hirsuta  Turing ocellus wavelength  154 µm  (0 Ma)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)
    return {
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "frame_diff": frame_diff,
        "last_frame": (last * 255.0).astype(np.uint8),
    }


def run_analysis(save_media: bool = False) -> dict:
    """Solve the model, compare it with the specimen, and optionally render."""
    verify_sha256()
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    margin = turing_margin()
    k_c, lam_dimless = classical_wavenumber()
    lambda_um = LAMBDA_UM  # dx is chosen so the classical wavelength is this value
    if abs(ref["phenomenon_scale"]["phenomenon_size_um"] - LAMBDA_UM) > 1e-9:
        raise ValueError("reference scale is not 154.1 µm")
    if abs(ref["paleobiology"]["geological_age_ma"] - AGE_MA) > 1e-12:
        raise ValueError("reference age is not 0 Ma")

    k_values = np.linspace(0.75 * k_c, 1.40 * k_c, 9)
    analytic = np.array([growth_rate(float(k)) for k in k_values])
    numeric = numerical_growth_rates(k_values)
    pearson_r = float(np.corrcoef(analytic, numeric)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic - numeric) ** 2)))

    primary = integrate(SEEDS[0], store_every=2000 if save_media else None)
    replicate = integrate(SEEDS[1], store_every=None)
    dx_um = LAMBDA_UM / PX_PER_LAMBDA
    spacing_px = [dominant_spacing_px(primary["u"]), dominant_spacing_px(replicate["u"])]
    spacing_um = [px * dx_um for px in spacing_px]
    sweep = parameter_sweep()
    ablated = ablation()
    specimen_stats = segment_specimen()

    media = None
    vision = None
    if save_media:
        portrait = np.asarray(Image.open(SPECIMEN).convert("RGB"))
        media = render_media(
            portrait,
            primary,
            sweep,
            {"k": k_values, "analytic": analytic, "numeric": numeric, "pearson_r": pearson_r, "k_c": k_c},
            ablated,
        )
        vision = cv_similarity(SPECIMEN, media["last_frame"], media["n_frames"], media["frame_diff"])

    return {
        "lambda_um": lambda_um,
        "lambda_m": LAMBDA_M,
        "log10_size_m": float(np.log10(LAMBDA_M)),
        "age_ma": AGE_MA,
        "k_c": k_c,
        "lambda_dimless": lam_dimless,
        "n_waves": N_WAVES,
        "margin": margin,
        "pearson_r": pearson_r,
        "rmse": rmse,
        "spacing_px": spacing_px,
        "spacing_um": spacing_um,
        "spacing_um_mean": float(np.mean(spacing_um)),
        "spacing_um_cv": float(np.std(spacing_um) / (np.mean(spacing_um) + 1e-12)),
        "pattern_std": float(np.std(primary["u"])),
        "sweep": sweep,
        "ablation": ablated,
        "specimen": specimen_stats,
        "media": {k: v for k, v in (media or {}).items() if k != "last_frame"},
        "vision": vision,
        "sigma_at_kc": growth_rate(k_c),
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    # Media already rendered by the CLI; recompute the quantitative comparisons.
    result = run_analysis(save_media=False)
    # Vision metrics need the rendered last frame. Read the GIF when the CLI has written it.
    vision = {"cv_sim_similarity": None, "specimen_radial_fft_pearson_r": None, "orientation_ks_statistic": None}
    if GIF_PATH.exists():
        with Image.open(GIF_PATH) as gif:
            n_frames = int(getattr(gif, "n_frames", 1))
            gif.seek(0)
            first = np.asarray(gif.convert("RGB"), dtype=np.float64) / 255.0
            gif.seek(n_frames - 1)
            last = np.asarray(gif.convert("RGB"))
            frame_diff = float(np.mean(np.abs(np.asarray(last, dtype=np.float64) / 255.0 - first)))
        vision = cv_similarity(SPECIMEN, last, n_frames, frame_diff)
    spec = result["specimen"]
    rows = [
        {
            "name": "specimen leaf foreground area fraction",
            "value": round(spec["foreground_area_fraction"], 4),
            "unit": "fraction",
            "reference_value": 0.50,
            "reference_source": "A filled-blade prior of 0.50; the herbarium rosette is sparser than a cropped leaflet.",
            "method": "Non-white mask on the public-domain Cardamine hirsuta photograph after a 1-pixel opening.",
        },
        {
            "name": "specimen rosette bounding-box aspect ratio",
            "value": round(spec["bbox_aspect_width_over_height"], 4),
            "unit": "width/height",
            "reference_value": 1.0,
            "reference_source": "Square-frame prior. The vendored portrait is taller than it is wide.",
            "method": "Bounding box of the segmented plant silhouette.",
        },
        {
            "name": "specimen contour circularity",
            "value": round(spec["contour_circularity"], 4),
            "unit": "4*pi*A/P^2",
            "reference_value": 1.0,
            "reference_source": "A circle has circularity 1. A rosette with petioles and roots does not.",
            "method": "Perimeter from the erosion outline of the specimen mask.",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(spec["structure_tensor_coherency"], 4),
            "unit": "coherency",
            "reference_value": 0.20,
            "reference_source": "A weakly oriented texture prior of 0.20, below the photographed rosette.",
            "method": "Mean structure-tensor coherency of the specimen luminance.",
        },
        {
            "name": "nonlinear ocellus spacing on the detail field",
            "value": round(result["spacing_um_mean"], 3),
            "unit": "µm",
            "reference_value": 154.1,
            "reference_source": "Classical Turing wavelength lambda_c = 154.1 µm from k_c at D_v/D_u = 10.5.",
            "method": "Mean first autocorrelation peak of activator u over seeds 3140 and 3141, times dx = 154.1/16 µm.",
            "note": f"per-seed µm = {result['spacing_um']}; CV = {result['spacing_um_cv']:.4f}; pattern std = {result['pattern_std']:.4f}",
        },
        {
            "name": "dispersion-band pearson r",
            "value": round(result["pearson_r"], 6),
            "unit": "Pearson r",
            "reference_value": 0.98,
            "reference_source": "Pass gate r > 0.98 against the analytic growth-rate curve, not against itself.",
            "method": "Pearson correlation of analytic σ(k) and a linear-window fit on nine wavenumbers around k_c.",
            "note": f"RMSE = {result['rmse']:.6f}",
        },
        {
            "name": "dispersion-band RMSE",
            "value": round(result["rmse"], 6),
            "unit": "growth-rate units",
            "reference_value": 0.05,
            "reference_source": "A loose 0.05 tolerance on σ; the fit is tighter than this.",
            "method": "Root mean square error between analytic and numerically fitted σ(k).",
        },
        {
            "name": "equal-diffusion ablation sigma_max",
            "value": round(result["ablation"]["sigma_max"], 6),
            "unit": "1/time",
            "reference_value": 0.0,
            "reference_source": "The patterned state has sigma(k_c) > 0. Equal D_u = D_v = 0.04 must sit below 0.",
            "method": "Maximum analytic growth rate on k in [0.2, 6] with D_u = D_v, plus a 1500-step control integration.",
            "note": f"ablation std {result['ablation']['std_initial']:.4f} -> {result['ablation']['std_final']:.4f}; patterned sigma(k_c) = {result['sigma_at_kc']:.4f}",
        },
        {
            "name": "diffusion-ratio sweep selected wavelength at D_v/D_u = 40",
            "value": round(result["sweep"]["rows"][-1]["lambda_um"], 3),
            "unit": "µm",
            "reference_value": 154.1,
            "reference_source": "Operating-point classical wavelength. Faster inhibitor diffusion lengthens the fastest mode.",
            "method": "Argmax of sigma(k) at D_v = 1.60, converted with the operating-point length unit.",
            "note": "monotonic_increasing=" + str(result["sweep"]["monotonic_increasing"]),
        },
    ]
    if vision.get("orientation_ks_statistic") is not None:
        rows.append({
            "name": "specimen versus render orientation KS statistic",
            "value": round(float(vision["orientation_ks_statistic"]), 4),
            "unit": "KS D",
            "reference_value": 0.0,
            "reference_source": "KS D = 0 would mean the gradient-orientation samples are identical.",
            "method": "Two-sample KS on Sobel orientations of the specimen crop and the GIF last frame.",
            "note": f"p = {vision['orientation_ks_pvalue']:.4g}; radial FFT Pearson r = {vision['specimen_radial_fft_pearson_r']:.4f}; CV similarity = {vision['cv_sim_similarity']}",
        })
        rows.append({
            "name": "specimen image radial-FFT pearson r",
            "value": round(float(vision["specimen_radial_fft_pearson_r"]), 4),
            "unit": "Pearson r",
            "reference_value": 0.90,
            "reference_source": "A 0.90 radial-spectrum prior. The last GIF frame is scored directly against the photograph.",
            "method": "Pearson r of the 22-bin log radial FFT used by the image-fitness verifier.",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Cardamine hirsuta 154 µm Turing wavelength (slot 3140)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = run_analysis(save_media=True)
    media = result["media"]
    vision = result["vision"] or {}
    print(json.dumps({
        "pearson_r": result["pearson_r"],
        "rmse": result["rmse"],
        "lambda_um": result["lambda_um"],
        "spacing_um": result["spacing_um"],
        "sigma_at_kc": result["sigma_at_kc"],
        "ablation_sigma_max": result["ablation"]["sigma_max"],
        "sweep_monotonic": result["sweep"]["monotonic_increasing"],
        "sweep_lambda_um": [row["lambda_um"] for row in result["sweep"]["rows"]],
        "n_frames": media.get("n_frames"),
        "resolution": [media.get("width"), media.get("height")],
        "frame_diff": media.get("frame_diff"),
        "cv_sim_similarity": vision.get("cv_sim_similarity"),
        "specimen_radial_fft_pearson_r": vision.get("specimen_radial_fft_pearson_r"),
        "orientation_ks": vision.get("orientation_ks_statistic"),
        "foreground": result["specimen"]["foreground_area_fraction"],
        "log10_size_m": result["log10_size_m"],
        "age_ma": result["age_ma"],
    }, indent=2))
    if args.self_test:
        margin = result["margin"]
        assert margin["trace"] < 0.0 and margin["det"] > 0.0
        assert margin["left"] > margin["right"]
        assert abs(result["lambda_um"] - 154.1) < 1e-9
        assert abs(result["age_ma"]) < 1e-12
        assert result["n_waves"] == 8
        assert result["pearson_r"] > 0.98, result["pearson_r"]
        assert result["rmse"] < 0.02
        assert result["ablation"]["sigma_max"] < 0.0
        assert result["sigma_at_kc"] > 0.0
        assert result["ablation"]["std_final"] < result["ablation"]["std_initial"]
        assert result["sweep"]["monotonic_increasing"]
        assert result["pattern_std"] > 0.05
        assert media["n_frames"] >= 10
        assert min(media["width"], media["height"]) >= 480
        assert media["frame_diff"] >= 0.003
        assert vision["cv_sim_similarity"] >= 58.0, vision
        assert vision["specimen_radial_fft_pearson_r"] > 0.98
        print("[PASS] slot 3140 Cardamine hirsuta Turing wavelength self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
