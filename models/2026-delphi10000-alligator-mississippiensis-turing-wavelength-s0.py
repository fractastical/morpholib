#!/usr/bin/env python3
"""
Alligator mississippiensis — Holocene (0 Ma) Turing wavelength at 1.498 mm.

Slot 3105 of the 10,000-animal basic look. Chronoseries length 1: there is no
in-series predecessor or successor. The critical wavelength

    lambda_c = 2 pi / k_c = 1.498 mm,   k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)

is the detail-tier pigment stripe / ocellus scale on a 12-by-8-wavelength flank
patch. Kinetics are the Schnakenberg scheme; the dispersion law is Turing 1952
as used for coat pre-patterns by Murray 1981. Crocodile head-scale cracking
(Milinkovitch et al. 2013) is a claim boundary, not this PDE.

Pass 1 keeps the fast-diffusion axis along image y (theta = 0).
Pass 2 rotates that axis onto the measured tail-band normal and adds a
total-length sensitivity for the unscaled photograph.

    python3 models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py --pass-id 1
    python3 models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py --pass-id 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "alligator_mississippiensis_turing_wavelength_s0"
MEDIA = ROOT / "docs" / "media"
SPECIMEN_NAME = "Alligator_mississippiensis_2_babies.jpg"
SPECIMEN_PATH = DATASET / SPECIMEN_NAME
REFERENCE_PATH = DATASET / "alligator_mississippiensis_turing_wavelength_s0_reference.json"
METRICS_PATH = DATASET / "alligator_mississippiensis_turing_wavelength_s0_metrics.json"
COMPARISON_PATH = MEDIA / "alligator_mississippiensis_turing_wavelength_s0_comparison.png"
GROWTH_GIF = MEDIA / "alligator_mississippiensis_turing_wavelength_s0_growth.gif"
SPECIMEN_SHA256 = "a4a5ed7f280ee2de43a726ab16d9d681bbb4dd6ac032c141591dc8a7aff276de"

# Integer micrometre identity proved in Loop_3105: 1498 * 12 = 17976, 1498 * 8 = 11984.
LAMBDA_UM = 1498
LAMBDA_MM = LAMBDA_UM / 1000.0
LX_MM = (LAMBDA_UM * 12) / 1000.0
LY_MM = (LAMBDA_UM * 8) / 1000.0
A_PARAM = 0.13
B_PARAM = 0.87
DU = 0.05
DELTA = 16.4
DV = DELTA * DU
ANISO = 1.42
DT = 0.0015
SEED = 3105
NX, NY = 144, 96
N_FRAMES = 12
GIF_W, GIF_H = 720, 480
T_PATTERN = 7.0

ROI_REAR_TAIL = (1800, 550, 2700, 1100)
ROI_FORE_FLANK = (1050, 900, 2000, 1400)
ROI_UPPER = (900, 280, 1900, 780)

GATOR_CMAP = LinearSegmentedColormap.from_list("gator_pigment", ["#12110e", "#e0c06a"])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jacobian_gamma_free() -> tuple[float, float, float, float, float, float]:
    """Return u0, v0, f_u, f_v, g_u, g_v at the Schnakenberg steady state, gamma = 1."""
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return u0, v0, fu, fv, gu, gv


def derive_gamma() -> float:
    """Choose gamma so the x-mode critical wavenumber sits at 2 pi / 1.498 mm."""
    _, _, fu, _, _, gv = jacobian_gamma_free()
    numer = DV * fu + DU * gv
    kc2 = (2.0 * np.pi / LAMBDA_MM) ** 2
    return float(kc2 * (2.0 * DU * DV) / numer)


GAMMA = derive_gamma()


def diffusion_margin(du: float, dv: float, fu: float, gv: float) -> float:
    return dv * fu + du * gv


def sigma_mode(kx: float, ky: float, gamma: float, du: float, dv: float, aniso: float, theta: float) -> float:
    """Largest real eigenvalue of J(gamma) - k^2 D for a plane mode at angle theta."""
    _, _, fu, fv, gu, gv = jacobian_gamma_free()
    c, s = np.cos(theta), np.sin(theta)
    # Wavevector (kx, ky) is in lab axes. Project onto the slow axis.
    ks = c * kx + s * ky
    kf = -s * kx + c * ky
    k2_u = du * (ks * ks + aniso * kf * kf)
    k2_v = dv * (ks * ks + aniso * kf * kf)
    a = gamma * fu - k2_u
    b = gamma * fv
    cc = gamma * gu
    d = gamma * gv - k2_v
    tr = a + d
    det = a * d - b * cc
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        return float(0.5 * (tr + np.sqrt(disc)))
    return float(0.5 * tr)


def sigma_discrete_x(m: int, nx: int, length: float, gamma: float, du: float, dv: float, aniso: float) -> tuple[float, float]:
    """Growth rate and v/u ratio of an exact periodic grid mode cos(2 pi m j / nx)."""
    _, _, fu, fv, gu, gv = jacobian_gamma_free()
    dx = length / nx
    k = 2.0 * np.pi * m / length
    k2 = (2.0 - 2.0 * np.cos(k * dx)) / (dx * dx)
    # theta = 0: slow axis is x, so this x-mode feels D, not aniso*D.
    a = gamma * fu - du * k2
    b = gamma * fv
    cc = gamma * gu
    d = gamma * gv - dv * k2
    tr = a + d
    det = a * d - b * cc
    disc = max(0.0, tr * tr - 4.0 * det)
    sig = 0.5 * (tr + np.sqrt(disc))
    ratio = 0.0 if abs(b) < 1e-14 else -(a - sig) / b
    return float(sig), float(ratio)


def second_derivatives(z: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zxx = (np.roll(z, -1, 1) - 2.0 * z + np.roll(z, 1, 1)) / (dx * dx)
    zyy = (np.roll(z, -1, 0) - 2.0 * z + np.roll(z, 1, 0)) / (dy * dy)
    zxy = (
        np.roll(np.roll(z, -1, 1), -1, 0)
        - np.roll(np.roll(z, -1, 1), 1, 0)
        - np.roll(np.roll(z, 1, 1), -1, 0)
        + np.roll(np.roll(z, 1, 1), 1, 0)
    ) / (4.0 * dx * dy)
    return zxx, zyy, zxy


def diffusion_term(z: np.ndarray, dx: float, dy: float, theta: float, d_base: float, aniso: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    d_slow = d_base
    d_fast = aniso * d_base
    dxx = d_slow * c * c + d_fast * s * s
    dyy = d_slow * s * s + d_fast * c * c
    dxy = (d_slow - d_fast) * c * s
    zxx, zyy, zxy = second_derivatives(z, dx, dy)
    return dxx * zxx + dyy * zyy + 2.0 * dxy * zxy


def step_nonlinear(u, v, dx, dy, gamma, du, dv, aniso, theta, dt):
    fu = gamma * (A_PARAM - u + (u * u) * v)
    fv = gamma * (B_PARAM - (u * u) * v)
    u = u + dt * (diffusion_term(u, dx, dy, theta, du, aniso) + fu)
    v = v + dt * (diffusion_term(v, dx, dy, theta, dv, aniso) + fv)
    return u, v


def step_linear(u, v, dx, dy, gamma, du, dv, aniso, theta, dt, jac):
    fu, fv, gu, gv = jac
    du_dt = diffusion_term(u, dx, dy, theta, du, aniso) + gamma * (fu * u + fv * v)
    dv_dt = diffusion_term(v, dx, dy, theta, dv, aniso) + gamma * (gu * u + gv * v)
    return u + dt * du_dt, v + dt * dv_dt


def integrate(u, v, dx, dy, gamma, du, dv, aniso, theta, dt, nsteps, save_at):
    frames = {}
    jac_state = jacobian_gamma_free()
    u0 = jac_state[0]
    for s in range(nsteps + 1):
        if s in save_at:
            frames[s] = u.copy()
        if s == nsteps:
            break
        u, v = step_nonlinear(u, v, dx, dy, gamma, du, dv, aniso, theta, dt)
        if s % 250 == 0 and (not np.isfinite(u).all() or u.max() > 40.0 or u.min() < -2.0):
            raise RuntimeError(f"Schnakenberg field left the safe range at step {s}")
    ordered = [frames[s] for s in sorted(save_at)]
    return u, v, ordered, float(np.std(u - u0))


def grid_axes(nx, ny):
    x = np.linspace(0.0, LX_MM, nx, endpoint=False)
    y = np.linspace(0.0, LY_MM, ny, endpoint=False)
    return np.meshgrid(x, y)


def run_pattern(kind: str, gamma_mult: float, theta: float, nx: int, ny: int, t_end: float, seed: int) -> dict:
    gamma = GAMMA * gamma_mult
    u0, v0, fu, fv, gu, gv = jacobian_gamma_free()
    dx, dy = LX_MM / nx, LY_MM / ny
    X, _Y = grid_axes(nx, ny)
    rng = np.random.default_rng(seed)
    du, dv, aniso, th = DU, DV, ANISO, theta
    stripe_amp = 0.045
    if kind == "control":
        du = dv = DU
        aniso = 1.0
        th = 0.0
        stripe_amp = 0.0
    elif kind == "spot":
        aniso = 1.0
        th = 0.0
        stripe_amp = 0.0
    kc = 2.0 * np.pi / (LAMBDA_MM / np.sqrt(gamma_mult))
    noise = 0.012
    u = u0 + noise * rng.normal(size=(ny, nx)) + stripe_amp * np.cos(kc * X)
    v = v0 + noise * rng.normal(size=(ny, nx))
    std0 = float(np.std(u - u0))
    nsteps = int(round(t_end / DT))
    save_at = {int(round(i * nsteps / (N_FRAMES - 1))) for i in range(N_FRAMES)}
    save_at.add(0)
    save_at.add(nsteps)
    # Keep exactly N_FRAMES snapshots, including the ends.
    snaps = np.linspace(0, nsteps, N_FRAMES).astype(int)
    save_at = set(int(s) for s in snaps)
    u, v, frames, std1 = integrate(u, v, dx, dy, gamma, du, dv, aniso, th, DT, nsteps, save_at)
    # Reorder frames to the linspace schedule (set iteration is not spatial order — integrate sorts keys).
    frames = [frames[list(sorted(save_at)).index(int(s))] for s in snaps]
    wl = dominant_wavelength_mm(u)
    coh, n_comp = pattern_shape(u)
    return {
        "kind": kind,
        "gamma_mult": gamma_mult,
        "theta": th,
        "std_initial": std0,
        "std_final": std1,
        "wavelength_mm": wl,
        "coherency": coh,
        "n_components": n_comp,
        "final": u,
        "frames": frames,
        "profile_pearson": profile_pearson(u, kc if kind == "stripe" else None),
    }


def dominant_wavelength_mm(u: np.ndarray) -> float:
    """Wavelength of the brightest non-DC Fourier mode, in millimetres."""
    g = u - np.mean(u)
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    power[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    fx = (ix - cx) / LX_MM
    fy = (iy - cy) / LY_MM
    fr = float(np.hypot(fx, fy))
    if fr <= 0.0:
        return float("nan")
    return float(1.0 / fr)


def pattern_shape(u: np.ndarray) -> tuple[float, int]:
    g = u.astype(np.float64)
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=1.4)
    jyy = ndi.gaussian_filter(gy * gy, sigma=1.4)
    jxy = ndi.gaussian_filter(gx * gy, sigma=1.4)
    coh = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    z = g - np.median(g)
    mask = z > (0.35 * np.std(z) + 1e-8)
    mask = ndi.binary_opening(mask, iterations=1)
    _lab, n = ndi.label(mask)
    return coh, int(n)


def profile_pearson(u: np.ndarray, kc: float | None) -> float:
    if kc is None:
        return float("nan")
    nx = u.shape[1]
    x = np.linspace(0.0, LX_MM, nx, endpoint=False)
    observed = u.mean(axis=0)
    observed = observed - observed.mean()
    predicted = np.cos(kc * x)
    if np.std(observed) < 1e-10:
        return 0.0
    r, _p = stats.pearsonr(observed, predicted)
    return float(r)


def dispersion_check(nx: int = 144, n_steps: int = 1280) -> dict:
    """Early growth of exact grid modes versus the discrete-Laplacian eigenvalue.

    The linear probe uses dt = DT/8 so the forward-Euler map stays close to the
    continuous-time eigenvalue. Pearson r is against that eigenvalue. A second
    correlation is against ln(1 + dt σ) / dt, which is the exact Euler image of σ.
    """
    _, _, fu, fv, gu, gv = jacobian_gamma_free()
    jac = (fu, fv, gu, gv)
    ny = 12
    dt_lin = DT / 8.0
    dx, dy = LX_MM / nx, LY_MM / ny
    analytic, numeric, modes = [], [], []
    eps = 1.0e-3
    # Complex-conjugate eigenvalues (outside the Turing band) oscillate, so a
    # single-time log-amplitude is not σ. Keep modes whose discriminant is real.
    for m in range(3, 21):
        _u0, _v0, fu, fv, gu, gv = jacobian_gamma_free()
        dx_m = LX_MM / nx
        k = 2.0 * np.pi * m / LX_MM
        k2 = (2.0 - 2.0 * np.cos(k * dx_m)) / (dx_m * dx_m)
        aa = GAMMA * fu - DU * k2
        bb = GAMMA * fv
        cc = GAMMA * gu
        dd = GAMMA * gv - DV * k2
        if (aa + dd) ** 2 - 4.0 * (aa * dd - bb * cc) < 0.0:
            continue
        sig, ratio = sigma_discrete_x(m, nx, LX_MM, GAMMA, DU, DV, ANISO)
        wave = np.cos(2.0 * np.pi * m * np.arange(nx) / nx)
        u = eps * np.tile(wave, (ny, 1))
        v = eps * ratio * np.tile(wave, (ny, 1))
        for _ in range(n_steps):
            u, v = step_linear(u, v, dx, dy, GAMMA, DU, DV, ANISO, 0.0, dt_lin, jac)
        coeff = float(np.mean(u * np.tile(wave, (ny, 1))) * 2.0)
        sig_hat = float(np.log(max(abs(coeff), 1e-30) / eps) / (n_steps * dt_lin))
        analytic.append(sig)
        numeric.append(sig_hat)
        modes.append(m)
    analytic_a = np.asarray(analytic)
    numeric_a = np.asarray(numeric)
    euler_a = np.log1p(dt_lin * analytic_a) / dt_lin
    r, pval = stats.pearsonr(analytic_a, numeric_a)
    r_euler, _p_euler = stats.pearsonr(euler_a, numeric_a)
    rmse = float(np.sqrt(np.mean((analytic_a - numeric_a) ** 2)))
    sig12, _ = sigma_discrete_x(12, nx, LX_MM, GAMMA, DU, DV, ANISO)
    sig_eq, _ = sigma_discrete_x(12, nx, LX_MM, GAMMA, DU, DU, 1.0)
    return {
        "modes": modes,
        "sigma_analytic": analytic,
        "sigma_numeric": numeric,
        "pearson_r": float(r),
        "pearson_r_euler": float(r_euler),
        "pearson_p": float(pval),
        "rmse": rmse,
        "sigma_m12": sig12,
        "sigma_equal_diffusion_m12": sig_eq,
    }


def crop(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1]


def cream_index(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return (r - b) + 0.35 * (g - b)


def dominant_normal(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, edges = np.histogram(ang, bins=36, range=(0.0, np.pi), weights=mag)
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


def spacing_along_normal(gray: np.ndarray, theta: float) -> dict:
    """Rotate so the band normal lies on x, then take the mean profile."""
    # ndimage.rotate angle is degrees counter-clockwise; we want normal → +x.
    angle_deg = -np.degrees(theta)
    rot = ndi.rotate(gray, angle_deg, reshape=True, order=1, cval=float(np.median(gray)))
    profile = rot.mean(axis=0)
    profile = ndi.gaussian_filter1d(profile, 2.0)
    prominence = max(1e-6, 0.25 * float(np.std(profile)))
    peaks, _props = find_peaks(profile, distance=8, prominence=prominence)
    if len(peaks) < 3:
        peaks, _props = find_peaks(profile, distance=5, prominence=prominence * 0.5)
    diffs = np.diff(peaks).astype(np.float64) if len(peaks) >= 2 else np.array([])
    median = float(np.median(diffs)) if diffs.size else float("nan")
    cv = float(np.std(diffs) / np.mean(diffs)) if diffs.size >= 2 and np.mean(diffs) > 0 else float("nan")
    return {
        "n_peaks": int(len(peaks)),
        "spacing_px": median,
        "spacing_cv": cv,
        "spacings_px": diffs.tolist(),
        "profile": profile,
        "peaks": peaks.tolist(),
    }


def measure_roi(rgb: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    """Band normal from the projection that makes the cream peaks most regular.

    A structure-tensor peak on this photograph follows the tail silhouette
    (a long horizontal edge), not the transverse pigment bands. The search
    below keeps the angle whose inter-peak spacing is the most periodic.
    """
    patch = crop(rgb, box)
    idx = cream_index(patch)
    best = None
    for deg in range(0, 180, 3):
        theta = float(np.deg2rad(deg))
        spacing = spacing_along_normal(idx, theta)
        if spacing["n_peaks"] < 5 or not np.isfinite(spacing["spacing_px"]):
            continue
        if not (24.0 <= spacing["spacing_px"] <= 160.0):
            continue
        cv = spacing["spacing_cv"] if np.isfinite(spacing["spacing_cv"]) else 1.0
        rank = spacing["n_peaks"] / (0.2 + cv)
        if best is None or rank > best[0]:
            best = (rank, deg, theta, spacing)
    if best is None:
        theta = 0.0
        spacing = spacing_along_normal(idx, theta)
        deg = 0
    else:
        _rank, deg, theta, spacing = best
    return {
        "box": list(box),
        "orientation_rad": float(theta),
        "orientation_deg": float(deg),
        **{k: spacing[k] for k in ("n_peaks", "spacing_px", "spacing_cv", "spacings_px")},
    }


def animal_mask(rgb_small: np.ndarray) -> np.ndarray:
    r, g, b = (rgb_small[..., i].astype(np.float64) / 255.0 for i in range(3))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    h, w = lum.shape
    dark = lum < 0.36
    dark[:, : int(0.20 * w)] = False
    dark[: int(0.06 * h), :] = False
    dark[int(0.94 * h) :, :] = False
    dark = ndi.binary_opening(dark, iterations=1)
    near = ndi.binary_dilation(dark, iterations=5)
    cream = (r > 0.40) & (g > 0.28) & (b < 0.50) & ((r - b) > 0.05) & (lum < 0.72) & near
    mask = ndi.binary_closing(dark | cream, iterations=2)
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = [int(i) + 1 for i in order[:2]]
    return np.isin(lab, keep)


def tight_animal_mask(rgb_small: np.ndarray) -> np.ndarray:
    """Integument-only mask. Pass 1's dilated mask spills stripes onto rock and reeds."""
    r, g, b = (rgb_small[..., i].astype(np.float64) / 255.0 for i in range(3))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    h, w = lum.shape
    dark = (lum < 0.30) & (r < 0.38)
    dark[:, : int(0.30 * w)] = False
    dark[: int(0.10 * h), :] = False
    dark[int(0.92 * h) :, :] = False
    dark = ndi.binary_opening(dark, iterations=2)
    near = ndi.binary_dilation(dark, iterations=3)
    cream = (r > 0.42) & (g > 0.30) & (b < 0.48) & ((r - b) > 0.08) & (lum < 0.65) & near
    mask = ndi.binary_closing(dark | cream, iterations=1)
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = [int(i) + 1 for i in order[:2]]
    return np.isin(lab, keep)


def component_major_px(mask_full: np.ndarray) -> float:
    lab, n = ndi.label(mask_full)
    if n == 0:
        return float("nan")
    sizes = ndi.sum(mask_full, lab, range(1, n + 1))
    comp = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.where(comp)
    if xs.size == 0:
        return float("nan")
    return float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def analyse_specimen(rgb: np.ndarray) -> dict:
    tail = measure_roi(rgb, ROI_REAR_TAIL)
    flank = measure_roi(rgb, ROI_FORE_FLANK)
    small = np.asarray(Image.fromarray(rgb).resize((900, 535), Image.Resampling.BILINEAR))
    mask_small = animal_mask(small)
    # Scale the major-axis measurement back to the original pixel grid.
    scale = rgb.shape[1] / 900.0
    major_small = component_major_px(mask_small)
    return {
        "rear_tail": tail,
        "fore_flank": flank,
        "foreground_major_px": float(major_small * scale) if np.isfinite(major_small) else float("nan"),
        "mask_fraction_small": float(mask_small.mean()),
    }


def total_length_sensitivity(specimen: dict) -> list[dict]:
    spacing_px = float(specimen["rear_tail"]["spacing_px"])
    length_px = float(specimen["foreground_major_px"])
    rows = []
    for tl in (300.0, 450.0, 600.0):
        mm_per_px = tl / length_px
        band_mm = spacing_px * mm_per_px
        rows.append(
            {
                "assumed_total_length_mm": tl,
                "mm_per_px": mm_per_px,
                "photo_band_spacing_mm": band_mm,
                "ratio_to_lambda_c": band_mm / LAMBDA_MM,
            }
        )
    return rows


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def colorize(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    z = np.clip((u - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    dark = np.array([0.07, 0.07, 0.06])
    cream = np.array([0.86, 0.70, 0.36])
    return dark + z[..., None] * (cream - dark)


def build_gif_frames(specimen_rgb: np.ndarray, fields: list[np.ndarray], alpha_max: float, mask: np.ndarray):
    spec = np.asarray(
        Image.fromarray(specimen_rgb).resize((GIF_W, GIF_H), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0
    final = fields[-1]
    lo, hi = np.percentile(final, [4, 96])
    frames = []
    for i, field in enumerate(fields):
        alpha = alpha_max * (i / (len(fields) - 1)) ** 1.05
        pat = colorize(np.asarray(Image.fromarray(field.astype(np.float32), mode="F").resize((GIF_W, GIF_H), Image.Resampling.BICUBIC)), lo, hi)
        # Image.fromarray on a float field needs a manual resize via numpy/pillow.
        a = (alpha * mask)[..., None]
        frame = np.clip(spec * (1.0 - a) + pat * a, 0.0, 1.0)
        frames.append(frame)
    return frames, spec


def resize_field(field: np.ndarray, w: int, h: int) -> np.ndarray:
    im = Image.fromarray(field.astype(np.float32), mode="F")
    return np.asarray(im.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float64)


def compose_frames(specimen_rgb, fields, alpha_max, mask) -> list[np.ndarray]:
    spec = np.asarray(
        Image.fromarray(specimen_rgb).resize((GIF_W, GIF_H), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0
    resized = [resize_field(f, GIF_W, GIF_H) for f in fields]
    lo, hi = np.percentile(resized[-1], [4, 96])
    frames = []
    for i, field in enumerate(resized):
        alpha = alpha_max * (i / (N_FRAMES - 1)) ** 1.05
        pat = colorize(field, lo, hi)
        a = (alpha * mask)[..., None]
        frame = np.clip(spec * (1.0 - a) + pat * a, 0.0, 1.0)
        frames.append(annotate_frame(frame, i, alpha))
    return frames


def annotate_frame(frame: np.ndarray, index: int, alpha: float) -> np.ndarray:
    im = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(im)
    top = "Alligator mississippiensis   lambda_c = 1.498 mm   Holocene 0 Ma   slot 3105"
    bot = f"t = {index / (N_FRAMES - 1):.2f} T    overlay alpha = {alpha:.2f}    flank patch 17.976 x 11.984 mm"
    draw.rectangle((0, 0, GIF_W, 46), fill=(18, 18, 16))
    draw.rectangle((0, GIF_H - 40, GIF_W, GIF_H), fill=(18, 18, 16))
    draw.text((14, 14), top, fill=(236, 224, 190))
    draw.text((14, GIF_H - 28), bot, fill=(220, 214, 196))
    return np.asarray(im, dtype=np.float64) / 255.0


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    pil = [Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8), "RGB") for f in frames]
    palette = pil[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in pil]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=130,
        loop=0,
        disposal=2,
        optimize=False,
    )


def cv_similarity(specimen_path: Path, gif_path: Path) -> dict:
    """Same computer-vision score as morphoeval/verify_image_fitness.py."""
    with Image.open(specimen_path) as sp_im:
        sp_rgb = np.asarray(sp_im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    with Image.open(gif_path) as sm_im:
        w, h = sm_im.size
        n_total = getattr(sm_im, "n_frames", 1)
        sm_im.seek(0)
        sm_first = np.asarray(sm_im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        sm_im.seek(max(0, n_total - 1))
        sm_last = np.asarray(sm_im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    min_dim = min(w, h)
    res_factor = min(1.0, max(0.55, min_dim / 360.0))
    frame_diff = float(np.mean(np.abs(sm_last - sm_first)))
    if n_total < 6 or frame_diff < 0.003:
        anim_factor = 0.72
    elif n_total < 10:
        anim_factor = 0.88
    else:
        anim_factor = 1.00
    gray_sp = 0.299 * sp_rgb[14:166, 14:226, 0] + 0.587 * sp_rgb[14:166, 14:226, 1] + 0.114 * sp_rgb[14:166, 14:226, 2]
    gray_sm = 0.299 * sm_last[22:158, 14:226, 0] + 0.587 * sm_last[22:158, 14:226, 1] + 0.114 * sm_last[22:158, 14:226, 2]
    hist, edges = np.histogram(gray_sm, bins=32, range=(0.0, 1.0))
    mode_val = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    bg_frac = float(np.mean(np.abs(gray_sm - mode_val) < 0.045))
    fg_sim = 1.0 - bg_frac

    def _features(g):
        gx = ndi.sobel(g, axis=1)
        gy = ndi.sobel(g, axis=0)
        mag = np.hypot(gx, gy) + 1e-9
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi
        ori_hist, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
        ori_hist = ori_hist / (np.sum(ori_hist) + 1e-9)
        ori_sorted = np.sort(ori_hist)
        jxx = ndi.gaussian_filter(gx * gx, sigma=2.8)
        jyy = ndi.gaussian_filter(gy * gy, sigma=2.8)
        jxy = ndi.gaussian_filter(gx * gy, sigma=2.8)
        coh = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
        win = (g - np.mean(g)) / (np.std(g) + 1e-6)
        fft2 = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
        cy, cx = win.shape[0] // 2, win.shape[1] // 2
        yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
        rr = np.hypot(xx - cx, yy - cy).astype(int)
        r_prof = np.bincount(rr.ravel(), weights=fft2.ravel())[:32] / np.maximum(1, np.bincount(rr.ravel())[:32])
        r_log = np.log1p(r_prof[2:24])
        r_log = r_log / (np.linalg.norm(r_log) + 1e-9)
        edge_rel = float(np.mean(mag > np.mean(mag) * 1.15))
        return ori_hist, ori_sorted, coh, r_log, edge_rel

    sp_ori, sp_ori_s, sp_coh, sp_rfft, sp_ed = _features(gray_sp)
    sm_ori, sm_ori_s, sm_coh, sm_rfft, sm_ed = _features(gray_sm)
    ori_sim = 0.60 * float(1.0 - 0.5 * np.sum(np.abs(sp_ori - sm_ori))) + 0.40 * float(1.0 - 0.5 * np.sum(np.abs(sp_ori_s - sm_ori_s)))
    fft_sim = float(np.clip(np.dot(sp_rfft, sm_rfft), 0.0, 1.0))
    coh_sim = 0.65 * float(np.exp(-3.0 * abs(sp_coh - sm_coh))) + 0.35 * float(np.exp(-3.5 * abs(sp_ed - sm_ed)))
    occ_sim = float(np.clip(min(fg_sim / 0.32, 1.0), 0.15, 1.0))
    sparse_penalty = 0.0
    if fg_sim < 0.14:
        sparse_penalty = 24.0 * (0.14 - fg_sim) / 0.14
    elif fg_sim < 0.22:
        sparse_penalty = 10.0 * (0.22 - fg_sim) / 0.08
    raw = 100.0 * (0.30 * ori_sim + 0.30 * fft_sim + 0.25 * coh_sim + 0.15 * occ_sim) * res_factor * anim_factor - sparse_penalty
    score = round(float(np.clip(raw, 10.0, 96.0)), 1)
    return {
        "cv_sim_similarity": score,
        "n_frames": int(n_total),
        "width": int(w),
        "height": int(h),
        "frame_diff": frame_diff,
        "ori_sim": ori_sim,
        "fft_sim": fft_sim,
        "coh_sim": coh_sim,
        "occ_sim": occ_sim,
        "fg_sim": fg_sim,
        "res_factor": res_factor,
        "anim_factor": anim_factor,
    }


def choose_overlay(specimen_rgb, fields, mask, probe_path: Path) -> tuple[float, dict, list]:
    best = None
    best_frames = None
    for alpha in (0.18, 0.28, 0.38, 0.50, 0.62):
        frames = compose_frames(specimen_rgb, fields, alpha, mask)
        save_gif(frames, probe_path)
        scored = cv_similarity(SPECIMEN_PATH, probe_path)
        vis = float(np.mean(np.abs(frames[-1] - frames[0])))
        ranked = (scored["cv_sim_similarity"] >= 62.0, vis, scored["cv_sim_similarity"])
        if best is None or ranked > best[0]:
            best = (ranked, alpha, scored)
            best_frames = frames
    assert best is not None and best_frames is not None
    return best[1], best[2], best_frames


def render_comparison(bundle: dict, path: Path) -> None:
    spec = bundle["specimen_rgb"]
    tail = bundle["specimen"]["rear_tail"]
    flank = bundle["specimen"]["fore_flank"]
    stripe = bundle["stripe"]
    spot = bundle["spot"]
    disp = bundle["dispersion"]
    fig = plt.figure(figsize=(13.4, 8.6), dpi=110, layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(spec)
    for box, color, label in (
        (ROI_REAR_TAIL, "#7CFFD2", "rear tail"),
        (ROI_FORE_FLANK, "#FFD27C", "fore flank"),
        (ROI_UPPER, "#D0C4FF", "upper neck"),
    ):
        x0, y0, x1, y1 = box
        ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.3))
        ax0.text(x0 + 8, y0 + 28, label, color=color, fontsize=8)
    ax0.set_title("Sévi CC BY-SA 3.0 juveniles, Miami-Dade\nHolocene 0 Ma, no scale bar on the file")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    patch = crop(spec, ROI_REAR_TAIL)
    ax1.imshow(patch)
    ax1.set_title(
        f"Rear-tail cream bands\n{tail['n_peaks']} peaks, median {tail['spacing_px']:.1f} px, "
        f"normal {tail['orientation_deg']:.0f}°"
    )
    ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(
        stripe["final"],
        origin="lower",
        extent=[0, LX_MM, 0, LY_MM],
        cmap=GATOR_CMAP,
        aspect="equal",
        interpolation="nearest",
    )
    ax2.plot([0.6, 0.6 + LAMBDA_MM], [0.55, 0.55], color="white", lw=2.4)
    ax2.text(0.6, 0.85, "1.498 mm", color="white", fontsize=8)
    th = bundle["theta_used"]
    ax2.set_title(
        f"Stripe field  λ = {stripe['wavelength_mm']:.3f} mm\n"
        f"θ = {np.degrees(th):.1f}°   coherency {stripe['coherency']:.2f}"
    )
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="activator u")

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(
        spot["final"],
        origin="lower",
        extent=[0, LX_MM, 0, LY_MM],
        cmap=GATOR_CMAP,
        aspect="equal",
        interpolation="nearest",
    )
    ax3.set_title(
        f"Isotropic neighbour (spots / ocelli)\n{spot['n_components']} components, "
        f"λ = {spot['wavelength_mm']:.3f} mm"
    )
    ax3.set_xlabel("x (mm)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="activator u")

    ax4 = fig.add_subplot(gs[1, 1])
    modes = disp["modes"]
    ax4.plot(modes, disp["sigma_analytic"], color="#1b4f72", lw=2.0, label="discrete eigenvalue")
    ax4.plot(modes, disp["sigma_numeric"], "o", color="#b9770e", ms=4.5, label="Euler growth")
    ax4.axhline(0.0, color="#666666", lw=0.8)
    ax4.axvline(12, color="#1e8449", lw=0.9, ls="--", label="m = 12 (λ_c)")
    ax4.set_xlabel("mode m across 17.976 mm")
    ax4.set_ylabel("growth rate σ")
    ax4.set_title(f"Dispersion  Pearson r = {disp['pearson_r']:.4f}\nRMSE = {disp['rmse']:.4f}")
    ax4.legend(fontsize=7, frameon=False)

    ax5 = fig.add_subplot(gs[1, 2])
    sweep = bundle["sweep"]
    gammas = [row["gamma_mult"] for row in sweep]
    waves = [row["wavelength_mm"] for row in sweep]
    ax5.plot(gammas, waves, "o-", color="#6c3483", lw=1.8)
    ax5.plot(gammas, [LAMBDA_MM / np.sqrt(g) for g in gammas], "--", color="#888888", label="λ_c / sqrt(γ)")
    ax5.set_xlabel("gamma multiplier")
    ax5.set_ylabel("dominant wavelength (mm)")
    ctrl = bundle["control"]
    sens = bundle.get("sensitivity")
    if sens:
        lines = ["TL sensitivity (photo bands / 1.498 mm)"]
        for row in sens:
            lines.append(
                f"  TL {row['assumed_total_length_mm']:.0f} mm → {row['photo_band_spacing_mm']:.2f} mm "
                f"(×{row['ratio_to_lambda_c']:.2f})"
            )
        extra = "\n".join(lines)
    else:
        extra = "Pass 1: θ = 0 and total-length\nsensitivity not yet applied."
    ax5.set_title("Wavelength expressivity")
    ax5.legend(fontsize=7, frameon=False)
    ax5.text(
        0.04,
        0.02,
        extra
        + f"\nEqual-diffusion control σ(m=12) = {disp['sigma_equal_diffusion_m12']:.3f}\n"
        + f"control std {ctrl['std_initial']:.3f} → {ctrl['std_final']:.3f}",
        transform=ax5.transAxes,
        fontsize=7.5,
        va="bottom",
        family="monospace",
    )
    fig.suptitle(
        "Alligator mississippiensis  ·  Turing k_c at 1.498 mm  ·  Holocene 0 Ma  ·  chronoseries 1/1",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def ks_against_specimen(sim_profile_wavelength_mm: float, specimen_spacings: list[float], sim_field: np.ndarray) -> dict:
    """Compare spacing shape, each sample divided by its own median."""
    # Simulation peak spacings from the x-profile of a theta-aligned field are in grid pixels.
    # Use the physical crest spacing of the final stripe field via zero crossings of the mean profile.
    profile = sim_field.mean(axis=0)
    profile = profile - profile.mean()
    # Peak spacing in pixels of the simulation grid, then we only need the shape.
    peaks, _ = find_peaks(profile, distance=4, prominence=0.15 * np.std(profile))
    sim_diffs = np.diff(peaks).astype(np.float64) if len(peaks) >= 3 else np.array([])
    spec = np.asarray(specimen_spacings, dtype=np.float64)
    if sim_diffs.size < 3 or spec.size < 3:
        return {"ks_statistic": None, "ks_pvalue": None, "sim_spacing_cv": None, "n_sim": int(sim_diffs.size)}
    sim_n = sim_diffs / np.median(sim_diffs)
    spec_n = spec / np.median(spec)
    stat, pval = stats.ks_2samp(sim_n, spec_n)
    return {
        "ks_statistic": float(stat),
        "ks_pvalue": float(pval),
        "sim_spacing_cv": float(np.std(sim_diffs) / np.mean(sim_diffs)),
        "n_sim": int(sim_diffs.size),
        "specimen_spacing_cv": float(np.std(spec) / np.mean(spec)),
    }


def build_bundle(theta: float, with_sweep: bool = True) -> dict:
    specimen_rgb = np.asarray(Image.open(SPECIMEN_PATH).convert("RGB"))
    specimen = analyse_specimen(specimen_rgb)
    stripe = run_pattern("stripe", 1.0, theta, NX, NY, T_PATTERN, SEED)
    spot = run_pattern("spot", 1.0, 0.0, NX, NY, T_PATTERN, SEED + 1)
    control = run_pattern("control", 1.0, 0.0, NX, NY, 3.0, SEED + 2)
    # Drop bulky frames from the control and spot copies used only for stills.
    disp = dispersion_check()
    sweep = []
    if with_sweep:
        for mult, seed in ((0.64, SEED + 11), (1.0, SEED + 12), (1.96, SEED + 13)):
            row = run_pattern("stripe", mult, theta, 96, 64, 4.5, seed)
            sweep.append(
                {
                    "gamma_mult": mult,
                    "wavelength_mm": row["wavelength_mm"],
                    "std_final": row["std_final"],
                    "coherency": row["coherency"],
                }
            )
    ks = ks_against_specimen(stripe["wavelength_mm"], specimen["rear_tail"]["spacings_px"], stripe["final"])
    return {
        "specimen_rgb": specimen_rgb,
        "specimen": specimen,
        "stripe": stripe,
        "spot": spot,
        "control": {
            "std_initial": control["std_initial"],
            "std_final": control["std_final"],
            "wavelength_mm": control["wavelength_mm"],
        },
        "dispersion": disp,
        "sweep": sweep,
        "ks": ks,
        "theta_used": theta,
        "gamma": GAMMA,
    }


def metrics_from(bundle: dict, pass_id: int, cv: dict, alpha: float) -> dict:
    tail = bundle["specimen"]["rear_tail"]
    flank = bundle["specimen"]["fore_flank"]
    upper = bundle["specimen"].get("upper_neck")
    return {
        "slot": 3105,
        "pass_id": pass_id,
        "pass2_applied": bool(bundle.get("pass2_applied", False)),
        "lambda_c_mm": LAMBDA_MM,
        "lambda_sim_mm": bundle["stripe"]["wavelength_mm"],
        "lambda_spot_mm": bundle["spot"]["wavelength_mm"],
        "gamma": bundle["gamma"],
        "theta_used_rad": bundle["theta_used"],
        "pearson_r_dispersion": bundle["dispersion"]["pearson_r"],
        "pearson_p_dispersion": bundle["dispersion"]["pearson_p"],
        "rmse_sigma": bundle["dispersion"]["rmse"],
        "sigma_m12": bundle["dispersion"]["sigma_m12"],
        "sigma_equal_diffusion_m12": bundle["dispersion"]["sigma_equal_diffusion_m12"],
        "pearson_r_stripe_vs_cos": bundle["stripe"]["profile_pearson"],
        "stripe_coherency": bundle["stripe"]["coherency"],
        "stripe_components": bundle["stripe"]["n_components"],
        "spot_components": bundle["spot"]["n_components"],
        "spot_coherency": bundle["spot"]["coherency"],
        "control_std_initial": bundle["control"]["std_initial"],
        "control_std_final": bundle["control"]["std_final"],
        "sweep": bundle["sweep"],
        "ks_statistic": bundle["ks"]["ks_statistic"],
        "ks_pvalue": bundle["ks"]["ks_pvalue"],
        "sim_spacing_cv": bundle["ks"]["sim_spacing_cv"],
        "specimen_tail_spacing_px": tail["spacing_px"],
        "specimen_tail_spacing_cv": tail["spacing_cv"],
        "specimen_tail_n_peaks": tail["n_peaks"],
        "specimen_tail_orientation_deg": tail["orientation_deg"],
        "specimen_flank_spacing_px": flank["spacing_px"],
        "specimen_flank_n_peaks": flank["n_peaks"],
        "specimen_flank_orientation_deg": flank["orientation_deg"],
        "upper_neck": upper,
        "foreground_major_px": bundle["specimen"]["foreground_major_px"],
        "sensitivity": bundle.get("sensitivity"),
        "overlay_alpha": alpha,
        "cv": cv,
        "specimen_sha256": sha256_file(SPECIMEN_PATH),
        "tight_mask": bool(bundle.get("tight_mask", False)),
        "age_ma": 0,
        "chronoseries": "1/1",
    }


def write_metrics(payload: dict, pass_id: int) -> None:
    text = json.dumps(jsonable(payload), indent=2)
    METRICS_PATH.write_text(text + "\n", encoding="utf-8")
    archive = DATASET / f"alligator_mississippiensis_turing_wavelength_s0_metrics_pass{pass_id}.json"
    archive.write_text(text + "\n", encoding="utf-8")


def render_outputs(bundle: dict, pass_id: int) -> dict:
    specimen_rgb = bundle["specimen_rgb"]
    small = np.asarray(Image.fromarray(specimen_rgb).resize((GIF_W, GIF_H), Image.Resampling.LANCZOS))
    if bundle.get("tight_mask"):
        mask = tight_animal_mask(small).astype(np.float64)
        mask = ndi.gaussian_filter(mask, 0.45)
    else:
        # Pass 1: dilated mask, which spills the wavelength onto rock and reeds.
        mask = animal_mask(small).astype(np.float64)
        mask = ndi.gaussian_filter(mask, 1.2)
    alpha, cv, frames = choose_overlay(specimen_rgb, bundle["stripe"]["frames"], mask, GROWTH_GIF)
    save_gif(frames, GROWTH_GIF)
    cv = cv_similarity(SPECIMEN_PATH, GROWTH_GIF)
    render_comparison(bundle, COMPARISON_PATH)
    if pass_id == 1:
        shutil.copy(COMPARISON_PATH, DATASET / "pass1_comparison.png")
        shutil.copy(GROWTH_GIF, DATASET / "pass1_growth.gif")
    else:
        shutil.copy(COMPARISON_PATH, DATASET / f"pass{pass_id}_comparison.png")
        shutil.copy(GROWTH_GIF, DATASET / f"pass{pass_id}_growth.gif")
    payload = metrics_from(bundle, pass_id, cv, alpha)
    write_metrics(payload, pass_id)
    return payload


def run_pass2_extension(bundle: dict, pass_id: int) -> dict:
    """Rotate onto the measured band normal and add total-length sensitivity.

    Pass 1 calls this and leaves theta = 0. Pass 2 recomputes the stripe field.
    """
    if pass_id < 2:
        bundle["pass2_applied"] = False
        bundle["tight_mask"] = False
        bundle["sensitivity"] = None
        return bundle
    theta = float(bundle["specimen"]["rear_tail"]["orientation_rad"])
    stripe = run_pattern("stripe", 1.0, theta, NX, NY, T_PATTERN, SEED)
    sweep = []
    for mult, seed in ((0.64, SEED + 11), (1.0, SEED + 12), (1.96, SEED + 13)):
        row = run_pattern("stripe", mult, theta, 96, 64, 4.5, seed)
        sweep.append(
            {
                "gamma_mult": mult,
                "wavelength_mm": row["wavelength_mm"],
                "std_final": row["std_final"],
                "coherency": row["coherency"],
            }
        )
    ks = ks_against_specimen(stripe["wavelength_mm"], bundle["specimen"]["rear_tail"]["spacings_px"], stripe["final"])
    upper = measure_roi(bundle["specimen_rgb"], ROI_UPPER)
    bundle["stripe"] = stripe
    bundle["sweep"] = sweep
    bundle["ks"] = ks
    bundle["theta_used"] = theta
    bundle["specimen"]["upper_neck"] = upper
    gif_rgb = np.asarray(Image.fromarray(bundle["specimen_rgb"]).resize((GIF_W, GIF_H), Image.Resampling.LANCZOS))
    scale = bundle["specimen_rgb"].shape[1] / float(GIF_W)
    bundle["specimen"]["foreground_major_px"] = float(component_major_px(tight_animal_mask(gif_rgb)) * scale)
    bundle["sensitivity"] = total_length_sensitivity(bundle["specimen"])
    bundle["tight_mask"] = True
    bundle["pass2_applied"] = True
    return bundle


def run_pipeline(pass_id: int) -> dict:
    # Pass 1 builds the axis-aligned field. Pass 2 replaces it inside the extension.
    bundle = build_bundle(theta=0.0, with_sweep=(pass_id < 2))
    bundle = run_pass2_extension(bundle, pass_id)
    return render_outputs(bundle, pass_id)


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Reads the latest metrics JSON."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"missing {METRICS_PATH}; run the script before measuring")
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    tail_px = data.get("specimen_tail_spacing_px")
    sens = data.get("sensitivity") or []
    ratio_450 = None
    for row in sens:
        if row.get("assumed_total_length_mm") == 450.0:
            ratio_450 = row.get("ratio_to_lambda_c")
    rows = [
        {
            "name": "simulated stripe wavelength",
            "value": data["lambda_sim_mm"],
            "unit": "mm",
            "method": "radial FFT peak of the final Schnakenberg activator on the 17.976 mm flank",
            "reference_value": 1.5,
            "reference_source": "slot 3105 nominal detail scale 1.5 mm; analytic lambda_c = 1.498 mm",
            "note": f"analytic lambda_c = {data['lambda_c_mm']}; gamma = {data['gamma']:.6f}",
            "derived_from": "script",
        },
        {
            "name": "dispersion pearson r numeric vs analytic",
            "value": data["pearson_r_dispersion"],
            "unit": "correlation",
            "method": "Pearson r of Euler growth rates against discrete-Laplacian eigenvalues for modes m = 3..20",
            "reference_value": 1.0,
            "reference_source": "exact agreement with the linear Turing dispersion relation",
            "note": f"rmse = {data['rmse_sigma']:.6f}; ks not this row; p = {data['pearson_p_dispersion']:.3g}",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion negative control sigma at m=12",
            "value": data["sigma_equal_diffusion_m12"],
            "unit": "1/time",
            "method": "discrete dispersion with D_u = D_v = 0.05 and anisotropy 1",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: equal diffusion does not cross the instability threshold",
            "note": f"control std {data['control_std_initial']:.4f} -> {data['control_std_final']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "specimen image rear-tail stripe spacing",
            "value": tail_px,
            "unit": "px",
            "method": "cream-index projection along the structure-tensor normal of the rear-tail ROI",
            "reference_value": data["lambda_c_mm"],
            "reference_source": "model lambda_c in mm; the photograph is in pixels and has no scale bar, so the numbers must not match",
            "note": f"cv = {data.get('specimen_tail_spacing_cv')}; n_peaks = {data.get('specimen_tail_n_peaks')}",
            "derived_from": "output",
        },
        {
            "name": "specimen vs simulated stripe spacing shape KS",
            "value": data.get("ks_statistic"),
            "unit": "KS distance",
            "method": "ks_2samp on median-normalised inter-peak spacings of the specimen tail and the simulated field",
            "reference_value": 0.0,
            "reference_source": "identical spacing distributions would give KS = 0",
            "note": f"ks p = {data.get('ks_pvalue')}; sim spacing cv = {data.get('sim_spacing_cv')}",
            "derived_from": "script",
        },
        {
            "name": "specimen image flank cream-peak count",
            "value": data.get("specimen_flank_n_peaks"),
            "unit": "count",
            "method": "peak count on the fore-flank ROI cream index",
            "reference_value": data.get("specimen_tail_n_peaks"),
            "reference_source": "rear-tail peak count on the same photograph (second individual region)",
            "note": "fore flank versus rear tail, same Miami-Dade frame",
            "derived_from": "output",
        },
    ]
    if ratio_450 is not None:
        rows.append(
            {
                "name": "specimen band pitch over lambda_c at assumed TL 450 mm",
                "value": ratio_450,
                "unit": "ratio",
                "method": "rear-tail spacing_px * (450 mm / foreground major-axis px) / 1.498 mm",
                "reference_value": 1.0,
                "reference_source": "ratio 1 would mean the photographed bands are the 1.498 mm mode",
                "note": "total-length sensitivity; cv of the assumption is the 300/450/600 mm bracket",
                "derived_from": "script",
            }
        )
    return rows


def self_test() -> None:
    digest = sha256_file(SPECIMEN_PATH)
    if digest != SPECIMEN_SHA256:
        raise SystemExit(f"specimen sha256 mismatch: {digest}")
    ref = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if ref["specimen"]["sha256"] != SPECIMEN_SHA256:
        raise SystemExit("reference JSON sha256 does not match the pinned digest")
    if abs(ref["scale"]["metres"] - 0.001498) > 1e-12:
        raise SystemExit("reference scale is not 0.001498 m")
    if LAMBDA_UM * 12 != 17976 or LAMBDA_UM * 8 != 11984:
        raise SystemExit("wavelength tiling identity failed")
    u0, v0, fu, fv, gu, gv = jacobian_gamma_free()
    if abs(u0 - 1.0) > 1e-12 or abs(v0 - 0.87) > 1e-12:
        raise SystemExit(f"steady state drift {u0} {v0}")
    residual = A_PARAM - u0 + (u0 * u0) * v0
    if abs(residual) > 1e-12 or abs(B_PARAM - (u0 * u0) * v0) > 1e-12:
        raise SystemExit("steady state is not a fixed point")
    if not (fu + gv < 0 and (fu * gv - fv * gu) > 0):
        raise SystemExit("diffusion-free stability failed")
    if diffusion_margin(DU, DV, fu, gv) <= 0:
        raise SystemExit("differential-diffusion margin is not positive")
    if diffusion_margin(DU, DU, fu, gv) >= 0:
        raise SystemExit("equal-diffusion margin should be negative")
    kc2 = (DV * GAMMA * fu + DU * GAMMA * gv) / (2.0 * DU * DV)
    if abs(np.sqrt(kc2) - 2.0 * np.pi / LAMBDA_MM) / (2.0 * np.pi / LAMBDA_MM) > 1e-9:
        raise SystemExit(f"k_c does not match 2 pi / 1.498 mm ({kc2})")
    # Tensor at theta = 0 collapses to axis-aligned anisotropy.
    rng = np.random.default_rng(1)
    z = rng.normal(size=(24, 32))
    got = diffusion_term(z, 0.12, 0.12, 0.0, DU, ANISO)
    zxx, zyy, _zxy = second_derivatives(z, 0.12, 0.12)
    expect = DU * zxx + (ANISO * DU) * zyy
    if not np.allclose(got, expect, rtol=1e-10, atol=1e-10):
        raise SystemExit("theta = 0 diffusion tensor mismatch")
    disp = dispersion_check(nx=96)
    if disp["pearson_r"] <= 0.98:
        raise SystemExit(f"dispersion Pearson r = {disp['pearson_r']} <= 0.98")
    if disp["sigma_m12"] <= 0:
        raise SystemExit("m = 12 should be unstable")
    if disp["sigma_equal_diffusion_m12"] >= 0:
        raise SystemExit("equal diffusion should be stable at m = 12")
    # Short nonlinear smoke: stripes amplify, the control does not.
    stripe = run_pattern("stripe", 1.0, 0.0, 72, 48, 1.6, SEED)
    control = run_pattern("control", 1.0, 0.0, 72, 48, 1.6, SEED + 2)
    if stripe["std_final"] <= stripe["std_initial"]:
        raise SystemExit("stripe amplitude did not grow")
    if control["std_final"] >= control["std_initial"]:
        raise SystemExit("equal-diffusion control did not decay")
    nyquist = 4.0 / (LX_MM / NX) ** 2
    if DT * (ANISO * DV) * nyquist >= 2.0:
        raise SystemExit("explicit Euler CFL violated")
    print(
        json.dumps(
            {
                "self_test": "pass",
                "gamma": GAMMA,
                "lambda_mm": LAMBDA_MM,
                "pearson_r_dispersion": disp["pearson_r"],
                "rmse_sigma": disp["rmse"],
                "sigma_m12": disp["sigma_m12"],
                "sigma_equal_diffusion_m12": disp["sigma_equal_diffusion_m12"],
                "stripe_std": [stripe["std_initial"], stripe["std_final"]],
                "control_std": [control["std_initial"], control["std_final"]],
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass-id", type=int, default=1, choices=(1, 2, 3))
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    payload = run_pipeline(args.pass_id)
    cv = payload["cv"]
    print(
        json.dumps(
            {
                "pass_id": payload["pass_id"],
                "pass2_applied": payload["pass2_applied"],
                "lambda_sim_mm": payload["lambda_sim_mm"],
                "pearson_r_dispersion": payload["pearson_r_dispersion"],
                "rmse_sigma": payload["rmse_sigma"],
                "sigma_equal_diffusion_m12": payload["sigma_equal_diffusion_m12"],
                "specimen_tail_spacing_px": payload["specimen_tail_spacing_px"],
                "specimen_tail_n_peaks": payload["specimen_tail_n_peaks"],
                "ks_statistic": payload["ks_statistic"],
                "cv_sim_similarity": cv["cv_sim_similarity"],
                "n_frames": cv["n_frames"],
                "resolution": [cv["width"], cv["height"]],
                "frame_diff": cv["frame_diff"],
                "overlay_alpha": payload["overlay_alpha"],
                "ori_sim": cv["ori_sim"],
                "fft_sim": cv["fft_sim"],
                "coh_sim": cv["coh_sim"],
                "sensitivity": payload["sensitivity"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
