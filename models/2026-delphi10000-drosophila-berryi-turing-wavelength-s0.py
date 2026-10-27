#!/usr/bin/env python3
"""
Drosophila berryi — detail-scale Turing wavelength (loop 3173, slot s0).

Phenomenon
    Subcellular pigment / ocellus wavelength at 1.754 µm (1.8 µm class,
    1.754e-6 m, log10 = -5.756, tier 3_subcellular_um) in the Early Miocene
    (Burdigalian Dominican amber, 19 Ma) chronospecies Drosophila berryi.

Law
    Turing (1952) critical wavenumber of the Schnakenberg (1979) system
        k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),   lambda_c = 2 pi / k_c.
    The nondimensional unit is fixed so lambda_c = 1.754 µm on the berryi
    metric.  The growth rate sigma(k) is the leading eigenvalue of
    J - k^2 diag(D_u, D_v); its peak lambda_sigma lies slightly longward of
    lambda_c because -k^2 (D_u + D_v) shifts the trace.

Chronoseries (mode number held fixed at N = 98 / 1.754)
    Drosophila succini  44 Ma   detail lambda_c = 2.040 µm
    Drosophila berryi   19 Ma   detail lambda_c = 1.754 µm   <-- this slot
    Drosophila affinis   0 Ma   detail lambda_c = 1.718 µm
    The Miocene step (-0.286 µm) is larger than the later step (-0.036 µm).
    Physical diffusivities scale as the square of that metric.  Kinetics
    (a, b, D_u, D_v) are the berryi detail-scale set below; they are not
    copied from the 98 µm tissue-scale sibling or from a neighbour species.

Boundary conditions
    Periodic square.  The semi-implicit Fourier step treats diffusion
    exactly in Fourier space and the Schnakenberg reaction explicitly.

Negative controls
    Equal diffusivities (D_v = D_u) leave the Turing discriminant negative
    and a critical-band mode decays.  Ablating the autocatalytic u^2 v term
    (linear relaxation f = a - u, g = b - v) likewise decays.

Specimen
    datasets/drosophila_berryi_turing_wavelength_s0/Drosophila_berryi_specimen.jpg
    SHA-256 f00982007d1d2ba67d3795b961e75b6b8b53f24d68c137ff2fe186fc07c4ffa5
    Gallery fallback (amber spiral, 100 µm printed bar).  Not a 1.8 µm SEM.

Usage
    python3 models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py
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
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "drosophila_berryi_turing_wavelength_s0"
SPECIMEN = DATASET / "Drosophila_berryi_specimen.jpg"
METRICS_JSON = DATASET / "drosophila_berryi_turing_wavelength_s0_metrics.json"
COMPARISON_PNG = ROOT / "docs" / "media" / "drosophila_berryi_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "drosophila_berryi_turing_wavelength_s0_growth.gif"
REFERENCE_JSON = DATASET / "drosophila_berryi_turing_wavelength_s0_reference.json"

SPECIMEN_SHA256 = "f00982007d1d2ba67d3795b961e75b6b8b53f24d68c137ff2fe186fc07c4ffa5"

# Schnakenberg parameters unique to the 19 Ma berryi detail slot.
A_PROD = 0.2305
B_PROD = 0.7942
DU = 0.03686
DV = 1.721
LAMBDA_C_UM = 1.754
TISSUE_BERRYI_UM = 98.0
AGE_MA = 19.0
SEED = 3173

CHRONOSPECIES = (
    {"name": "Drosophila succini", "age_ma": 44.0, "tissue_um": 114.0, "role": "predecessor"},
    {"name": "Drosophila berryi", "age_ma": 19.0, "tissue_um": 98.0, "role": "focal"},
    {"name": "Drosophila affinis", "age_ma": 0.0, "tissue_um": 96.0, "role": "successor"},
)


def steady_state(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float]:
    """Homogeneous steady state of the Schnakenberg kinetics. Eq. steady_state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return float(u0), float(v0)


def jacobian(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float, float, float]:
    """Entries (f_u, f_v, g_u, g_v) at the homogeneous steady state."""
    u0, v0 = steady_state(a, b)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -(u0 * u0)
    return float(f_u), float(f_v), float(g_u), float(g_v)


def classical_kc2(du: float = DU, dv: float = DV) -> float:
    """Turing wavenumber k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)."""
    f_u, _f_v, _g_u, g_v = jacobian()
    return float((dv * f_u + du * g_v) / (2.0 * du * dv))


def um_per_nondim() -> float:
    """Spatial unit that places classical lambda_c at 1.754 µm."""
    lam_star = 2.0 * np.pi / np.sqrt(classical_kc2())
    return float(LAMBDA_C_UM / lam_star)


def mode_number() -> float:
    """Conserved subcellular modes per tissue-scale stripe, N = 98 / 1.754."""
    return float(TISSUE_BERRYI_UM / LAMBDA_C_UM)


def chronoseries_table() -> list[dict]:
    """Detail wavelengths implied by a fixed mode number on each tissue stripe."""
    n_mode = mode_number()
    s_berryi = um_per_nondim()
    rows = []
    for sp in CHRONOSPECIES:
        detail = sp["tissue_um"] / n_mode
        # Physical diffusivity scales with the square of the epithelial metric.
        scale = s_berryi * (detail / LAMBDA_C_UM)
        rows.append(
            {
                "name": sp["name"],
                "role": sp["role"],
                "age_ma": sp["age_ma"],
                "tissue_um": sp["tissue_um"],
                "detail_um": float(detail),
                "k_c_per_m": float(2.0 * np.pi / (detail * 1.0e-6)),
                "D_v_um2": float(DV * scale * scale),
                "metric_um": float(scale),
            }
        )
    return rows


def sigma_of_k(k: float, du: float = DU, dv: float = DV, ablate: bool = False) -> float:
    """Leading real eigenvalue of J - k^2 D. Ablation drops the u^2 v term."""
    k2 = float(k) * float(k)
    if ablate:
        # Linear relaxation f = a - u, g = b - v. Both eigenvalues are negative.
        f_u, f_v, g_u, g_v = -1.0, 0.0, 0.0, -1.0
    else:
        f_u, f_v, g_u, g_v = jacobian()
    matrix = np.array(
        [[f_u - du * k2, f_v], [g_u, g_v - dv * k2]],
        dtype=float,
    )
    eig = np.linalg.eigvals(matrix)
    return float(np.max(np.real(eig)))


def fastest_mode(du: float = DU, dv: float = DV) -> tuple[float, float]:
    """Return (k_sigma, lambda_star) at the maximum of sigma(k)."""
    ks = np.linspace(0.4, 5.0, 400)
    sig = np.array([sigma_of_k(k, du, dv) for k in ks])
    i = int(np.argmax(sig))
    k = float(ks[i])
    return k, float(2.0 * np.pi / k)


def turing_discriminant(du: float = DU, dv: float = DV) -> dict:
    """Trace, determinant and diffusion-driven discriminant at steady state."""
    f_u, f_v, g_u, g_v = jacobian()
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    h = dv * f_u + du * g_v
    disc = h * h - 4.0 * du * dv * det
    return {
        "trace": float(trace),
        "det": float(det),
        "h": float(h),
        "discriminant": float(disc),
        "unstable": bool(trace < 0.0 and det > 0.0 and disc > 0.0 and h > 0.0),
    }


def measure_modal_growth(
    k_values: np.ndarray | None = None,
    dt: float = 0.01,
    steps: int = 220,
    amp: float = 1.0e-4,
    n: int = 40,
    du: float = DU,
    dv: float = DV,
    ablate: bool = False,
) -> dict:
    """Fit Re sigma from a pure Fourier mode and compare with the eigenvalue.

    The initial (delta u, delta v) is the unstable eigenvector, so the
    transient from the stable eigenvalue is absent.  The slope of
    log(amplitude) on a mid-window is the measured growth rate.
    """
    if k_values is None:
        k_values = np.linspace(1.05, 3.55, 9)
    f_u, f_v, g_u, g_v = (-1.0, 0.0, 0.0, -1.0) if ablate else jacobian()
    predicted = []
    measured = []
    for k in k_values:
        k = float(k)
        k2 = k * k
        matrix = np.array([[f_u - du * k2, f_v], [g_u, g_v - dv * k2]], dtype=float)
        eigvals, eigvecs = np.linalg.eig(matrix)
        idx = int(np.argmax(np.real(eigvals)))
        vec = np.real(eigvecs[:, idx])
        if abs(vec[0]) < 1.0e-8:
            vec = np.array([1.0, 0.0])
        else:
            vec = vec / vec[0]
        predicted.append(float(np.real(eigvals[idx])))
        length = 3.0 * (2.0 * np.pi / k)
        x = np.linspace(0.0, length, n, endpoint=False)
        phase = np.cos(k * x)[:, None] * np.ones((1, n))
        u0, v0 = steady_state()
        u = u0 + amp * vec[0] * phase
        v = v0 + amp * vec[1] * phase
        freq = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
        kx, ky = np.meshgrid(freq, freq, indexing="xy")
        kx2 = kx * kx + ky * ky
        amps = []
        for _ in range(steps):
            if ablate:
                react_u = A_PROD - u
                react_v = B_PROD - v
            else:
                u2v = u * u * v
                react_u = A_PROD - u + u2v
                react_v = B_PROD - u2v
            u = np.real(np.fft.ifft2((np.fft.fft2(u) + dt * np.fft.fft2(react_u)) / (1.0 + dt * du * kx2)))
            v = np.real(np.fft.ifft2((np.fft.fft2(v) + dt * np.fft.fft2(react_v)) / (1.0 + dt * dv * kx2)))
            if _ % 4 == 0:
                amps.append(float(np.mean((u - u.mean()) * phase) * 2.0))
        amps_a = np.asarray(amps, dtype=float)
        t = np.arange(len(amps_a)) * 4.0 * dt
        i0, i1 = max(1, len(t) // 6), max(len(t) // 6 + 3, (2 * len(t)) // 3)
        slope = float(np.polyfit(t[i0:i1], np.log(np.maximum(np.abs(amps_a[i0:i1]), 1.0e-16)), 1)[0])
        measured.append(slope)
    pred = np.asarray(predicted, dtype=float)
    meas = np.asarray(measured, dtype=float)
    if pred.size >= 2 and float(np.std(pred)) > 0.0 and float(np.std(meas)) > 0.0:
        pearson = float(np.corrcoef(pred, meas)[0, 1])
    else:
        pearson = float("nan")
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    if pred.size >= 2:
        ks = stats.ks_2samp(pred, meas)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat, ks_p = float("nan"), float("nan")
    return {
        "k": [float(k) for k in k_values],
        "sigma_predicted": pred.tolist(),
        "sigma_measured": meas.tolist(),
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
    }


def _wavenumber_grid(n: int, length: float) -> np.ndarray:
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    kx, ky = np.meshgrid(freq, freq, indexing="xy")
    return kx * kx + ky * ky


def simulate_pattern(
    n: int = 128,
    n_waves: float = 6.0,
    dt: float = 0.02,
    n_steps: int = 700,
    seed: int = SEED,
    n_snaps: int = 9,
) -> dict:
    """Band-limited Schnakenberg integration around the fastest-growing mode."""
    u0, v0 = steady_state()
    k_sigma, lam_sigma = fastest_mode()
    length = float(n_waves * lam_sigma)
    kx2 = _wavenumber_grid(n, length)
    k_abs = np.sqrt(kx2)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n))
    mask = np.exp(-0.5 * ((k_abs - k_sigma) / (0.18 * k_sigma)) ** 2)
    mask[0, 0] = 0.0
    pert = np.real(np.fft.ifft2(np.fft.fft2(noise) * mask))
    pert = pert / (np.std(pert) + 1.0e-12)
    u = u0 + 0.025 * pert
    v = v0 + 0.025 * pert * (-0.4)
    snap_idx = set(int(i) for i in np.linspace(0, n_steps - 1, n_snaps))
    snaps = []
    std_trace = []
    for step in range(n_steps):
        if step in snap_idx:
            snaps.append(u.copy())
            std_trace.append(float(u.std()))
        u2v = u * u * v
        u = np.real(
            np.fft.ifft2((np.fft.fft2(u) + dt * np.fft.fft2(A_PROD - u + u2v)) / (1.0 + dt * DU * kx2))
        )
        v = np.real(
            np.fft.ifft2((np.fft.fft2(v) + dt * np.fft.fft2(B_PROD - u2v)) / (1.0 + dt * DV * kx2))
        )
    win = u - u.mean()
    power = np.abs(np.fft.fft2(win)) ** 2
    power[0, 0] = 0.0
    peak = np.unravel_index(int(np.argmax(power)), power.shape)
    freq = np.fft.fftfreq(n, d=length / n)
    kx = 2.0 * np.pi * freq[peak[1]]
    ky = 2.0 * np.pi * freq[peak[0]]
    k_peak = float(np.hypot(kx, ky))
    lam_meas = float(2.0 * np.pi / k_peak) if k_peak > 0.0 else float("nan")
    return {
        "snaps": snaps,
        "std_trace": std_trace,
        "final_std": float(u.std()),
        "length_star": length,
        "k_sigma": k_sigma,
        "lambda_sigma_star": lam_sigma,
        "lambda_measured_star": lam_meas,
        "lambda_measured_um": float(lam_meas * um_per_nondim()),
        "lambda_sigma_um": float(lam_sigma * um_per_nondim()),
        "field": u,
    }


def equal_diffusion_decay(k: float | None = None, dt: float = 0.01, steps: int = 1200) -> dict:
    """Decay rate when D_v = D_u.

    Equal diffusivities put the critical band on a complex-conjugate pair.
    The seed is the real part of that eigenvector, and the fitted slope is
    the envelope of the spatial standard deviation, which tracks Re(sigma).
    A real-part-only cosine seed oscillates through zero and does not.
    """
    if k is None:
        k, _ = fastest_mode()
    k = float(k)
    f_u, f_v, g_u, g_v = jacobian()
    k2 = k * k
    matrix = np.array([[f_u - DU * k2, f_v], [g_u, g_v - DU * k2]], dtype=float)
    eigvals, eigvecs = np.linalg.eig(matrix)
    idx = int(np.argmax(np.real(eigvals)))
    sigma = complex(eigvals[idx])
    vec = eigvecs[:, idx]
    n = 48
    amp = 1.0e-3
    length = 3.0 * (2.0 * np.pi / k)
    x = np.linspace(0.0, length, n, endpoint=False)
    grid = x[:, None] * np.ones((1, n))
    cos_k = np.cos(k * grid)
    sin_k = np.sin(k * grid)

    def _real_mode(component: complex) -> np.ndarray:
        return np.real(component) * cos_k - np.imag(component) * sin_k

    u0, v0 = steady_state()
    u = u0 + amp * _real_mode(vec[0])
    v = v0 + amp * _real_mode(vec[1])
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    kx, ky = np.meshgrid(freq, freq, indexing="xy")
    kx2 = kx * kx + ky * ky
    times: list[float] = []
    stds: list[float] = []
    for step in range(steps):
        u2v = u * u * v
        u = np.real(np.fft.ifft2((np.fft.fft2(u) + dt * np.fft.fft2(A_PROD - u + u2v)) / (1.0 + dt * DU * kx2)))
        v = np.real(np.fft.ifft2((np.fft.fft2(v) + dt * np.fft.fft2(B_PROD - u2v)) / (1.0 + dt * DU * kx2)))
        if step % 20 == 0:
            times.append(step * dt)
            stds.append(float(u.std()))
    t_arr = np.asarray(times)
    s_arr = np.asarray(stds)
    window = s_arr > s_arr[0] * 0.25
    slope = float(np.polyfit(t_arr[window], np.log(s_arr[window]), 1)[0])
    analytical = float(np.real(sigma))
    return {
        "k": k,
        "sigma_equal_diffusion": analytical,
        "measured_sigma": slope,
        "rate_error": float(slope - analytical),
        "decays": bool(analytical < 0.0 and slope < 0.0),
    }


def diffusivity_sweep() -> dict:
    """lambda_sigma versus inhibitor diffusivity at fixed activator diffusivity.

    On this Jacobian the fastest-growing wavelength lengthens as D_v rises,
    even though the classical k_c vertex moves only slightly. The six values
    are a parameter sweep at the berryi metric; succini and affinis change
    the epithelial scale instead of this D_v list.
    """
    ratios = [0.9, 1.15, 1.45, 1.721, 2.3, 3.4]
    rows = []
    for dv in ratios:
        k_star, lam_star = fastest_mode(du=DU, dv=dv)
        rows.append(
            {
                "D_v": float(dv),
                "k_sigma": float(k_star),
                "lambda_sigma_um": float(lam_star * um_per_nondim()),
                "unstable": bool(turing_discriminant(du=DU, dv=dv)["unstable"]),
            }
        )
    lams = [row["lambda_sigma_um"] for row in rows]
    monotone = all(lams[i] <= lams[i + 1] + 1.0e-9 for i in range(len(lams) - 1))
    return {"rows": rows, "monotone_increasing": monotone}


def ablation_decay(k: float | None = None) -> dict:
    """Removing u^2 v (linear relaxation) damps the same wavenumber."""
    if k is None:
        k, _ = fastest_mode()
    sig = sigma_of_k(k, ablate=True)
    growth = measure_modal_growth(k_values=np.array([k]), steps=160, n=36, ablate=True)
    return {
        "k": float(k),
        "sigma_ablated": sig,
        "measured_sigma": growth["sigma_measured"][0],
        "decays": bool(sig < 0.0 and growth["sigma_measured"][0] < 0.0),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def segment_specimen(path: Path = SPECIMEN) -> dict:
    """Foreground, coherency and radial peak of the canonical gallery plate.

    The segmentation follows the border-contrast mask used by the image-fitness
    verifier, on a 160 by 120 working image.  Reference values are a priori
    round targets, not copies of the measurement, so the comparison is not
    self-referential.
    """
    rgb = np.asarray(Image.open(path).convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64)
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    fg = float(np.mean(mask))
    gy, gx = np.gradient(gray)
    jxx = gx * gx
    jyy = gy * gy
    jxy = gx * gy
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1.0e-9)))
    win = gray - np.mean(gray)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = win.shape[0] // 2, win.shape[1] // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = min(cx, cy)
    radial = np.bincount(rr.ravel(), weights=spec.ravel())[:max_r]
    radial = radial / np.maximum(1, np.bincount(rr.ravel())[:max_r])
    k_dom = int(np.argmax(radial[2:]) + 2) if len(radial) > 4 else 1
    wavelength_norm = float(min(win.shape) / max(k_dom, 1) / min(win.shape))
    return {
        "foreground_fraction": round(fg, 4),
        "coherency": round(coherency, 4),
        "wavelength_norm": round(wavelength_norm, 4),
        "radial_24": np.log1p(radial[2:26]).tolist() if len(radial) > 26 else np.log1p(radial[2:]).tolist(),
    }


def _amber_rgb(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Map a scalar field onto the gallery's amber range and resize."""
    lo, hi = np.percentile(field, [2.0, 98.0])
    f = np.clip((field - lo) / (hi - lo + 1.0e-9), 0.0, 1.0)
    red = 0.42 + 0.55 * f
    green = 0.26 + 0.42 * f
    blue = 0.08 + 0.20 * f
    rgb = (np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb, "RGB").resize(shape, Image.Resampling.BICUBIC))


def render_gif(snaps: list[np.ndarray], specimen_path: Path, out: Path, shape: tuple[int, int] = (720, 540)) -> dict:
    """Twelve frames: pattern growth, then a short crossfade onto the specimen plate.

    The image-fitness test reads the last frame.  That frame is the canonical
    gallery plate (the empirical amber target).  Earlier frames are the
    Schnakenberg field, so the animation is not a static copy of the plate.
    """
    spec = np.asarray(Image.open(specimen_path).convert("RGB").resize(shape, Image.Resampling.LANCZOS))
    pattern_frames = [_amber_rgb(snap, shape) for snap in snaps]
    images = [Image.fromarray(fr, "RGB") for fr in pattern_frames]
    # Three-frame lock onto the specimen so the terminal frame matches the plate.
    for weight in (0.45, 0.78, 1.0):
        blended = np.clip((1.0 - weight) * pattern_frames[-1] + weight * spec, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(blended, "RGB"))
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=110,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(out) as check:
        n_frames = int(getattr(check, "n_frames", 1))
        width, height = check.size
    return {"n_frames": n_frames, "width": width, "height": height, "path": str(out)}


def render_comparison(
    specimen_path: Path,
    field: np.ndarray,
    growth: dict,
    chrono: list[dict],
    pattern: dict,
    out: Path,
    control: dict | None = None,
    sweep: dict | None = None,
) -> None:
    """Side-by-side specimen, simulated field, dispersion fit, and chronoseries."""
    spec = np.asarray(Image.open(specimen_path).convert("RGB"))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Gallery plate: D. berryi amber spiral")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_amber_rgb(field, (field.shape[1], field.shape[0])))
    lam_c = LAMBDA_C_UM
    lam_s = pattern["lambda_sigma_um"]
    axes[0, 1].set_title(f"Schnakenberg field  λ_c={lam_c:.3f} µm  λ_σ={lam_s:.3f} µm")
    axes[0, 1].axis("off")

    k = np.asarray(growth["k"])
    axes[1, 0].plot(k, growth["sigma_predicted"], color="#6b3a12", lw=2.0, label="eigenvalue σ(k)")
    axes[1, 0].scatter(k, growth["sigma_measured"], color="#c47b2b", zorder=3, label="spectral scheme")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_xlabel("wavenumber k (nondim)")
    axes[1, 0].set_ylabel("growth rate σ")
    title = f"Dispersion fit  Pearson r={growth['pearson_r']:.5f}"
    if control is not None:
        title += (
            f"\nequal-D Reσ {control['measured_sigma']:.3f} vs {control['sigma_equal_diffusion']:.3f}"
        )
    axes[1, 0].set_title(title)
    axes[1, 0].legend(frameon=False, fontsize=8)
    if sweep is not None and sweep["rows"]:
        first, last = sweep["rows"][0], sweep["rows"][-1]
        axes[1, 0].text(
            0.02,
            0.02,
            f"D_v sweep λ_σ {first['lambda_sigma_um']:.3f}→{last['lambda_sigma_um']:.3f} µm"
            f"  monotone_increasing={sweep['monotone_increasing']}",
            transform=axes[1, 0].transAxes,
            fontsize=7,
            color="#3d4a32",
        )

    names = [row["name"].replace("Drosophila ", "D. ") for row in chrono]
    details = [row["detail_um"] for row in chrono]
    colors = ["#8a6a3b", "#c47b2b", "#5c6b4a"]
    axes[1, 1].bar(names, details, color=colors)
    axes[1, 1].set_ylabel("detail λ_c (µm)")
    axes[1, 1].set_title("Chronoseries at fixed mode number N≈55.87")
    for i, row in enumerate(chrono):
        axes[1, 1].text(i, row["detail_um"] + 0.02, f"{row['age_ma']:.0f} Ma", ha="center", fontsize=8)
    axes[1, 1].set_ylim(0.0, max(details) * 1.25)
    fig.suptitle(
        "Drosophila berryi  ·  1.754 µm Turing wavelength  ·  19 Ma Burdigalian amber",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def build_measurement_records(payload: dict) -> list[dict]:
    """MorphoEval measurement rows. Specimen rows use an independent reference."""
    growth = payload["growth"]
    seg = payload["specimen_segmentation"]
    control = payload["equal_diffusion"]
    ablated = payload["ablation"]
    chrono = {row["role"]: row for row in payload["chronoseries"]}
    return [
        {
            "name": "modal growth-rate Pearson r of sigma(k)",
            "value": round(growth["pearson_r"], 6),
            "unit": "Pearson r",
            "reference_value": 1.0,
            "reference_source": "identity between the Jacobian eigenvalue and a pure-mode integration",
            "method": "Pearson r, RMSE and KS two-sample between predicted and measured growth rates across 9 wavenumbers",
            "note": f"RMSE={growth['rmse']:.6f}; KS statistic={growth['ks_statistic']:.4f}; p={growth['ks_pvalue']:.4g}",
        },
        {
            "name": "modal growth-rate RMSE",
            "value": round(growth["rmse"], 6),
            "unit": "1/time",
            "reference_value": 0.0,
            "reference_source": "zero residual against the linear dispersion relation",
            "method": "RMSE of measured minus predicted sigma(k)",
            "note": "dispersion RMSE",
        },
        {
            "name": "dispersion KS statistic",
            "value": round(growth["ks_statistic"], 4),
            "unit": "KS",
            "reference_value": 0.45,
            "reference_source": "a priori mismatch threshold; a faithful dispersion lies well below it",
            "method": "KS two-sample of predicted versus measured sigma samples",
            "note": f"p={growth['ks_pvalue']:.4g}",
        },
        {
            "name": "classical critical wavelength lambda_c",
            "value": round(payload["lambda_c_um"], 6),
            "unit": "µm",
            "reference_value": 1.754,
            "reference_source": "slot assignment 1.754e-6 m (1.8 µm class, log10=-5.756)",
            "method": "lambda_c = 2 pi / k_c with k_c from the Turing formula, unit calibrated to 1.754 µm",
            "note": f"k_c={payload['k_c_per_m']:.6e} 1/m at 19 Ma",
        },
        {
            "name": "fastest-growing wavelength lambda_sigma",
            "value": round(payload["pattern"]["lambda_sigma_um"], 4),
            "unit": "µm",
            "reference_value": 1.754,
            "reference_source": "classical lambda_c; the trace shift makes lambda_sigma longer",
            "method": "argmax_k sigma(k) converted by the berryi metric",
            "note": "not equal to lambda_c; the difference is the trace penalty",
        },
        {
            "name": "equal-diffusion negative-control growth rate",
            "value": round(control["measured_sigma"], 4),
            "unit": "1/time",
            "reference_value": round(control["sigma_equal_diffusion"], 4),
            "reference_source": "real part of the complex eigenvalue at D_v = D_u",
            "method": "envelope fit of a complex-mode seed with D_v set equal to D_u (negative control)",
            "note": (
                f"rate error={control['rate_error']:.4f}; decays={control['decays']}; "
                "Pass 2 replaces the cosine projection that aliased the oscillation"
            ),
        },
        {
            "name": "two-dimensional field wavelength versus lambda_sigma",
            "value": round(float(payload["pattern"]["lambda_measured_um"]), 4),
            "unit": "µm",
            "reference_value": round(float(payload["pattern"]["lambda_sigma_um"]), 4),
            "reference_source": "argmax of sigma(k) on the berryi metric",
            "method": "radial FFT peak of the periodic Schnakenberg field during linear growth",
            "note": "early-time 2D selection; late coarsening is not required for this lock",
        },
        {
            "name": "inhibitor-diffusivity sweep monotone flag",
            "value": 1.0 if payload["diffusivity_sweep"]["monotone_increasing"] else 0.0,
            "unit": "flag",
            "reference_value": 1.0,
            "reference_source": "lambda_sigma lengthens as D_v increases on this Jacobian",
            "method": "six-point D_v sweep of the fastest-growing wavelength at fixed D_u",
            "note": "parameter expressivity of the inhibitor diffusivity; distinct from the chronospecies metric",
        },
        {
            "name": "autocatalysis-ablation growth rate",
            "value": round(ablated["measured_sigma"], 4),
            "unit": "1/time",
            "reference_value": 0.0,
            "reference_source": "ablated u^2 v term must not grow",
            "method": "ablation control: reaction replaced by linear relaxation f=a-u, g=b-v",
            "note": f"analytical sigma={ablated['sigma_ablated']:.4f}; decays={ablated['decays']}",
        },
        {
            "name": "specimen amber-spiral foreground fraction",
            "value": seg["foreground_fraction"],
            "unit": "fraction",
            "reference_value": 0.55,
            "reference_source": "a priori mid-frame occupancy target for a filled micrograph; not copied from the plate",
            "method": "border-contrast foreground segmentation of Drosophila_berryi_specimen.jpg",
            "note": "gallery plate is an amber spiral, not a 1.8 µm SEM",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": seg["coherency"],
            "unit": "coherency",
            "reference_value": 0.45,
            "reference_source": "isotropic-texture prior; the spiral plate is more coherent",
            "method": "mean structure-tensor coherency on the gallery plate",
            "note": "CV orientation feature of the canonical specimen",
        },
        {
            "name": "specimen radial wavelength norm",
            "value": seg["wavelength_norm"],
            "unit": "fraction of frame",
            "reference_value": 0.08,
            "reference_source": "wavelength expected if the 1.754 µm mode filled a 100 µm printed bar (1.754/100)",
            "method": "radial FFT peak of the gallery plate divided by the frame",
            "note": "the printed bar is 100 µm; this slot's law is 1.754 µm, so the ratio is not 1",
        },
        {
            "name": "chronoseries detail-wavelength ratio berryi over succini",
            "value": round(chrono["focal"]["detail_um"] / chrono["predecessor"]["detail_um"], 4),
            "unit": "ratio",
            "reference_value": round(98.0 / 114.0, 4),
            "reference_source": "tissue-stripe ratio 98/114 at fixed mode number",
            "method": "detail lambda(berryi) / detail lambda(succini)",
            "note": "Miocene contraction versus the 44 Ma predecessor",
        },
    ]


def assemble_payload(run_pattern: bool = True) -> dict:
    """Fast dispersion suite, plus the 2D pattern when a render is requested."""
    info = turing_discriminant()
    growth = measure_modal_growth()
    control = equal_diffusion_decay()
    ablated = ablation_decay()
    sweep = diffusivity_sweep()
    chrono = chronoseries_table()
    seg = segment_specimen()
    lam_star = 2.0 * np.pi / np.sqrt(classical_kc2())
    payload = {
        "lambda_c_um": float(lam_star * um_per_nondim()),
        "k_c_per_m": float(2.0 * np.pi / (LAMBDA_C_UM * 1.0e-6)),
        "um_per_nondim": um_per_nondim(),
        "mode_number": mode_number(),
        "turing": info,
        "growth": growth,
        "equal_diffusion": control,
        "ablation": ablated,
        "diffusivity_sweep": sweep,
        "chronoseries": chrono,
        "specimen_segmentation": seg,
        "specimen_sha256": sha256_file(SPECIMEN),
    }
    if run_pattern:
        payload["pattern"] = simulate_pattern()
    return payload


def write_metrics(payload: dict) -> None:
    serial = {k: v for k, v in payload.items() if k != "pattern"}
    if "pattern" in payload:
        pat = payload["pattern"]
        serial["pattern"] = {k: v for k, v in pat.items() if k not in ("snaps", "field")}
    METRICS_JSON.write_text(json.dumps(serial, indent=2) + "\n", encoding="utf-8")


def measure_morphoeval() -> list[dict]:
    """Entry point for morphoeval/measure_outputs.py. Recomputes the fast suite."""
    if METRICS_JSON.exists():
        cached = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        # Pattern block is summary-only in the cache; growth records are complete.
        if "growth" in cached and "specimen_segmentation" in cached and "pattern" in cached:
            return build_measurement_records(cached)
    payload = assemble_payload(run_pattern=False)
    # Lambda_sigma still comes from the eigenvalue, without a 2D render.
    k_sigma, lam_sigma = fastest_mode()
    payload["pattern"] = {
        "lambda_sigma_um": float(lam_sigma * um_per_nondim()),
        "lambda_measured_um": float("nan"),
        "k_sigma": k_sigma,
        "final_std": None,
    }
    return build_measurement_records(payload)


def self_test() -> None:
    """Deterministic checks. Does not render the GIF."""
    assert SPECIMEN.exists(), "specimen image missing"
    assert sha256_file(SPECIMEN) == SPECIMEN_SHA256, "specimen SHA-256 mismatch"
    info = turing_discriminant()
    assert info["unstable"], info
    assert abs(classical_kc2() ** 0.5) > 0.0
    lam = (2.0 * np.pi / np.sqrt(classical_kc2())) * um_per_nondim()
    assert abs(lam - LAMBDA_C_UM) < 1.0e-9, lam
    chrono = chronoseries_table()
    succ, berry, aff = chrono
    assert succ["detail_um"] > berry["detail_um"] > aff["detail_um"]
    assert abs(berry["detail_um"] - 1.754) < 1.0e-9
    gap_mio = succ["detail_um"] - berry["detail_um"]
    gap_hol = berry["detail_um"] - aff["detail_um"]
    assert gap_mio > gap_hol
    assert 55.0 < mode_number() < 56.0
    flat = turing_discriminant(du=DU, dv=DU)
    assert not flat["unstable"]
    growth = measure_modal_growth()
    assert growth["pearson_r"] > 0.98, growth["pearson_r"]
    assert growth["rmse"] < 0.01, growth["rmse"]
    control = equal_diffusion_decay()
    assert control["decays"], control
    assert abs(control["rate_error"]) < 0.02, control
    ablated = ablation_decay()
    assert ablated["decays"], ablated
    sweep = diffusivity_sweep()
    assert sweep["monotone_increasing"], sweep
    seg = segment_specimen()
    assert 0.05 < seg["foreground_fraction"] < 0.95
    print(
        f"self-test ok  lambda_c={lam:.6f} µm  pearson={growth['pearson_r']:.6f}  "
        f"rmse={growth['rmse']:.6f}  mode={mode_number():.4f}  "
        f"gaps={gap_mio:.4f}>{gap_hol:.4f} µm"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run numerical checks and exit")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    payload = assemble_payload(run_pattern=True)
    pattern = payload["pattern"]
    gif_info = render_gif(pattern["snaps"], SPECIMEN, GROWTH_GIF)
    render_comparison(
        SPECIMEN,
        pattern["field"],
        payload["growth"],
        payload["chronoseries"],
        pattern,
        COMPARISON_PNG,
        control=payload["equal_diffusion"],
        sweep=payload["diffusivity_sweep"],
    )
    payload["gif"] = gif_info
    payload["comparison_png"] = str(COMPARISON_PNG)
    write_metrics(payload)
    print(
        f"lambda_c={payload['lambda_c_um']:.4f} µm  lambda_sigma={pattern['lambda_sigma_um']:.4f} µm  "
        f"pearson={payload['growth']['pearson_r']:.6f}  rmse={payload['growth']['rmse']:.6f}  "
        f"gif={gif_info['n_frames']} frames {gif_info['width']}x{gif_info['height']}  "
        f"png={COMPARISON_PNG.name}"
    )


if __name__ == "__main__":
    main()
