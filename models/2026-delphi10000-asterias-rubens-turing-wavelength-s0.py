#!/usr/bin/env python3
"""
Asterias rubens — Turing reaction-diffusion wavelength at 115.2 µm (slot 3123).

Holocene (extant), 0 Ma. Catalog length 0.0001152 m (log10 ≈ −3.9386; tier
5/8 ``5_primordium_100um``, detail variant). Aristotelian place: II. Anaima
(Invertebrata) → Ostrakoderma → Asterias. Form: stripes, spots and ocelli.
Law: Turing wavenumber k_c. Cohort: chemical Turing / reaction-diffusion.

The chronoseries for this slot is a single point (Holocene, 1/1). There is no
earlier chronospecies in the series. What changes at this scale, relative to
the adjacent cellular tier (~10 µm pigment cell / photoreceptor) and the
adjacent organ tier (~1 mm arm-tip optic cushion), is that a diffusion-driven
wavenumber first fits inside an epithelial patch: the 576.0 µm Neumann domain
holds exactly five critical wavelengths, one per pentaradial sector.

Governing system
----------------
Schnakenberg (1979) J. Theor. Biol. 81:389–400 kinetics, linearised as in
Turing (1952) Phil. Trans. R. Soc. B 237:37–72 and arranged as the
critical-wavenumber formula used by Murray (1981) J. Theor. Biol. 88:161–199:

    ∂u/∂t = D_u ∇²u + a − u + u² v          (activator)
    ∂v/∂t = D_v ∇²v + b − u² v              (substrate)
    k_c²  = (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c   = 2π / k_c = 115.2 µm

Parameters below are specific to this 115.2 µm detail slot. They are not the
slot-154 template (α=1.23, β=0.25).

Specimen
--------
Hans Hillewaert, CC BY-SA 4.0, Wikimedia Commons File:Asterias_rubens.jpg,
Belgian continental shelf, studio photograph. The photograph is an organism-scale
phenotype. It is not a 115.2 µm micrograph and carries no scale bar; absolute
tubercle spacing in micrometres is not read off the plate. The GIF ends on that
photograph. The microscopic wavelength is the RD field in the comparison figure.

Usage
-----
    python3 models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SPECIMEN = ROOT / "datasets" / "asterias_rubens_turing_wavelength_s0" / "asterias_rubens_specimen.jpg"
PNG_PATH = ROOT / "docs" / "media" / "asterias_rubens_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "asterias_rubens_turing_wavelength_s0_growth.gif"

# --- slot-3123 parameters (do not alias a sibling slot) ---------------------
SLOT = 3123
SEED = 3123
AGE_MA = 0.0
LAMBDA_UM = 115.2                 # 0.0001152 m
LAMBDA_M = 1.152e-4
N_WAVES = 5                       # one critical wavelength per pentaradial sector
DOMAIN_UM = N_WAVES * LAMBDA_UM   # 576.0 µm
A_PROD = 0.072                    # basal activator production (1/τ)
B_PROD = 0.918                    # substrate feed (1/τ)
# Just above the Schnakenberg onset root r≈7.487, so the only unstable
# Neumann mode on the 576 µm patch is the p=5 wave (λ=115.2 µm). Farther
# above onset the σ peak drifts longward of the catalog wavenumber.
DIFF_RATIO = 7.62                 # D_v / D_u
GRID_N = 96
T_FINAL = 320.0
N_FRAMES = 12
GIF_W, GIF_H = 840, 666
# Pass 1 keeps the reaction on the full square. Pass 2 sets this True so the
# pentaradial body planform, scaled onto the 576 µm patch, masks the reaction.
BODY_MASK = False

_SCIENCE: dict | None = None


def steady_state(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float]:
    """Schnakenberg homogeneous steady state (u*, v*) = (a+b, b/(a+b)²)."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float, float, float]:
    """Reaction Jacobian (f_u, f_v, g_u, g_v) at the homogeneous steady state.

    f = a − u + u²v,  g = b − u²v
    f_u = −1 + 2uv, f_v = u², g_u = −2uv, g_v = −u².
    """
    u0, v0 = steady_state(a, b)
    uv = u0 * v0
    f_u = -1.0 + 2.0 * uv
    f_v = u0 * u0
    g_u = -2.0 * uv
    g_v = -u0 * u0
    return f_u, f_v, g_u, g_v


def turing_coefficients(diff_ratio: float = DIFF_RATIO) -> dict:
    """Critical wavenumber and diffusivities that place λ_c at 115.2 µm.

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), with D_v = r D_u, solved for D_u
    once k_c = 2π / λ_c is fixed by the catalog scale. Murray (1981) arrangement
    of the Turing (1952) dispersion condition.
    """
    f_u, f_v, g_u, g_v = jacobian()
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    drive = diff_ratio * f_u + g_v  # sign of (D_v f_u + D_u g_v) / D_u
    k_c = 2.0 * math.pi / LAMBDA_UM
    k_c2 = k_c * k_c
    d_u = drive / (2.0 * diff_ratio * k_c2)
    d_v = diff_ratio * d_u
    margin = drive * drive - 4.0 * diff_ratio * det
    return {
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "trace": trace,
        "det": det,
        "drive": drive,
        "margin": margin,
        "k_c": k_c,
        "k_c2": k_c2,
        "D_u": d_u,
        "D_v": d_v,
        "lambda_um": LAMBDA_UM,
        "diff_ratio": diff_ratio,
    }


def sigma_of_k(k: np.ndarray, coeff: dict, scale: float = 1.0) -> np.ndarray:
    """Larger eigenvalue σ(k) of the reaction-diffusion Jacobian.

    σ = ½ (tr + √(tr² − 4 det)), with
    tr  = f_u + g_v − k² (D_u + D_v)
    det = (f_u − D_u k²)(g_v − D_v k²) − f_v g_u.
    ``scale`` multiplies both diffusivities (λ ∝ √scale).
    """
    k2 = np.asarray(k, dtype=float) ** 2
    d_u = coeff["D_u"] * scale
    d_v = coeff["D_v"] * scale
    f_u, f_v, g_u, g_v = coeff["f_u"], coeff["f_v"], coeff["g_u"], coeff["g_v"]
    tr = (f_u + g_v) - k2 * (d_u + d_v)
    det = (f_u - d_u * k2) * (g_v - d_v * k2) - f_v * g_u
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def _dt_for(dx: float, d_u: float, d_v: float) -> float:
    return 0.35 * (dx * dx) / (4.0 * max(d_u, d_v))


def measured_growth_rate(k: float, coeff: dict, scale: float = 1.0, t_window: float = 36.0) -> float:
    """FTCS growth rate of a pure cosine mode on a periodic line (numerical σ)."""
    k = float(k)
    if k <= 0.0:
        return float(sigma_of_k(np.array([0.0]), coeff, scale=scale)[0])
    n_per = 4
    n = 128
    length = n_per * 2.0 * math.pi / k
    dx = length / n
    x = np.arange(n) * dx
    d_u = coeff["D_u"] * scale
    d_v = coeff["D_v"] * scale
    f_u, f_v, g_u, g_v = coeff["f_u"], coeff["f_v"], coeff["g_u"], coeff["g_v"]
    jac = np.array(
        [[f_u - d_u * k * k, f_v], [g_u, g_v - d_v * k * k]],
        dtype=float,
    )
    ev, vecs = np.linalg.eig(jac)
    idx = int(np.argmax(np.real(ev)))
    vec = np.real(vecs[:, idx])
    vec = vec / (np.max(np.abs(vec)) + 1e-15)
    u0, v0 = steady_state()
    eps = 1.0e-3
    mode = np.cos(k * x)
    u = u0 + eps * vec[0] * mode
    v = v0 + eps * vec[1] * mode
    dt = min(0.0025, _dt_for(dx, d_u, d_v))
    steps = max(12, int(t_window / dt))

    def amplitude(z: np.ndarray) -> float:
        return float(np.dot(z - z.mean(), mode) / (0.5 * n))

    a0 = amplitude(u)
    a_kin, b_kin = A_PROD, B_PROD
    for _ in range(steps):
        lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
        lap_v = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
        uv = u * u * v
        u = u + dt * (d_u * lap_u + a_kin - u + uv)
        v = v + dt * (d_v * lap_v + b_kin - uv)
    a1 = amplitude(u)
    if abs(a0) < 1e-14 or abs(a1) < 1e-18:
        return 0.0
    return math.log(abs(a1 / a0)) / (steps * dt)


def dispersion_comparison(coeff: dict, n_k: int = 12) -> dict:
    """Analytical σ(k) against the 1D FTCS growth rate, plus the equal-D control."""
    k_c = coeff["k_c"]
    ks = k_c * np.linspace(0.45, 1.70, n_k)
    analytical = sigma_of_k(ks, coeff, scale=1.0)
    numerical = np.array([measured_growth_rate(float(k), coeff, scale=1.0) for k in ks])
    pearson = float(np.corrcoef(analytical, numerical)[0, 1])
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    ks_stat = float(stats.ks_2samp(analytical, numerical).statistic)
    # Equal diffusion: r = 1 destroys the drive term whenever trace < 0.
    d_u = coeff["D_u"]
    ab = dict(coeff)
    ab["D_u"] = d_u
    ab["D_v"] = d_u
    ab["diff_ratio"] = 1.0
    ab_analytical = sigma_of_k(ks, ab, scale=1.0)
    # The helper multiplies coeff diffusivities; pass a coeff whose D_v == D_u.
    ab_numerical = np.array([measured_growth_rate(float(k), ab, scale=1.0, t_window=12.0) for k in ks])
    return {
        "k": ks,
        "sigma_analytical": analytical,
        "sigma_numerical": numerical,
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": ks_stat,
        "ablation_sigma_analytical_max": float(np.max(ab_analytical)),
        "ablation_sigma_numerical_max": float(np.max(ab_numerical)),
    }


def diffusivity_sweep(coeff: dict) -> dict:
    """λ(s) for D → s D. Analytical law λ = λ_c √s, checked by numerical σ peaks."""
    scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    lam_ref = LAMBDA_UM * np.sqrt(scales)
    lam_num = []
    for s in scales:
        k_expect = coeff["k_c"] / math.sqrt(float(s))
        trial = k_expect * np.linspace(0.72, 1.28, 9)
        sig = np.array([measured_growth_rate(float(k), coeff, scale=float(s), t_window=28.0) for k in trial])
        i = int(np.argmax(sig))
        if 0 < i < len(trial) - 1:
            y0, y1, y2 = sig[i - 1], sig[i], sig[i + 1]
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            # trial is geometric-linear in k; interpolate in index space
            k_hat = float(np.interp(i + delta, np.arange(len(trial)), trial))
        else:
            k_hat = float(trial[i])
        lam_num.append(2.0 * math.pi / k_hat)
    lam_num_a = np.array(lam_num)
    pearson = float(np.corrcoef(lam_ref, lam_num_a)[0, 1])
    return {
        "scale": scales,
        "lambda_analytical_um": lam_ref,
        "lambda_numerical_um": lam_num_a,
        "pearson_r": pearson,
        "ratio_s4_over_s1": float(lam_num_a[-1] / lam_num_a[2]),
    }


def laplacian_neumann(z: np.ndarray, dx: float) -> np.ndarray:
    zp = np.pad(z, 1, mode="edge")
    return (zp[2:, 1:-1] + zp[:-2, 1:-1] + zp[1:-1, 2:] + zp[1:-1, :-2] - 4.0 * z) / (dx * dx)


def band_limited_noise(n: int, dx: float, k_c: float, rng: np.random.Generator, width: float = 0.28) -> np.ndarray:
    white = rng.standard_normal((n, n))
    spec = np.fft.fft2(white)
    freq = np.fft.fftfreq(n, d=dx) * 2.0 * math.pi
    kx, ky = np.meshgrid(freq, freq)
    kr = np.sqrt(kx * kx + ky * ky)
    window = np.exp(-0.5 * ((kr - k_c) / (width * k_c)) ** 2)
    window[0, 0] = 0.0
    field = np.fft.ifft2(spec * window).real
    field = field / (np.std(field) + 1e-12)
    return field


def simulate_field(
    coeff: dict,
    body_mask: np.ndarray | None = None,
    seed: int = SEED,
    t_final: float | None = None,
) -> dict:
    """Integrate the Schnakenberg system on the 576 µm Neumann patch."""
    n = GRID_N
    dx = DOMAIN_UM / n
    d_u, d_v = coeff["D_u"], coeff["D_v"]
    # Explicit FTCS, held under the 2D Neumann stability bound.
    dt = min(0.0072, 0.72 * (dx * dx) / (4.0 * max(d_u, d_v)))
    horizon = T_FINAL if t_final is None else t_final
    steps = max(2, int(horizon / dt))
    u0, v0 = steady_state()
    rng = np.random.default_rng(seed)
    noise_u = band_limited_noise(n, dx, coeff["k_c"], rng)
    noise_v = band_limited_noise(n, dx, coeff["k_c"], rng)
    # Weak pentaradial bias (Asteroidea), far below the critical-mode amplitude.
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2.0
    theta = np.arctan2(yy - cy, xx - cx)
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (n / 2.0)
    pent = 0.15 * np.cos(5.0 * theta) * np.exp(-((radius - 0.45) ** 2) / 0.08)
    u = u0 + 0.025 * noise_u + 0.0005 * pent
    v = v0 - 0.004 * noise_u + 0.012 * noise_v
    if body_mask is None:
        mask = np.ones((n, n), dtype=float)
    else:
        mask = body_mask.astype(float)
        if mask.shape != (n, n):
            mask = np.array(
                Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize((n, n), Image.Resampling.BOX)
            )
            mask = (mask > 127).astype(float)
        # Keep a one-pixel rim so Neumann edges stay defined.
        mask = ndi.binary_dilation(mask > 0.5, iterations=1).astype(float)
    snap_idx = set(int(round(i * (steps - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES))
    snaps = []
    a_kin, b_kin = A_PROD, B_PROD
    for s in range(steps):
        if s in snap_idx:
            snaps.append(u.copy())
        uv = u * u * v
        react_u = (a_kin - u + uv) * mask
        react_v = (b_kin - uv) * mask
        # Outside the epithelium, relax back to the homogeneous state (no pattern).
        relax = 1.0 - mask
        u = u + dt * (d_u * laplacian_neumann(u, dx) + react_u - 2.5 * relax * (u - u0))
        v = v + dt * (d_v * laplacian_neumann(v, dx) + react_v - 2.5 * relax * (v - v0))
    if len(snaps) < N_FRAMES:
        snaps.append(u.copy())
    snaps = snaps[:N_FRAMES]
    lam = dominant_wavelength_um(u, dx)
    return {"u": u, "v": v, "snaps": snaps, "dx": dx, "dt": dt, "steps": steps, "lambda_um": lam, "mask": mask}


def dominant_wavelength_um(field: np.ndarray, dx: float) -> float:
    """Sub-bin radial-FFT wavelength on a square Neumann field (µm)."""
    win = field - np.mean(field)
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    yy, xx = np.ogrid[: power.shape[0], : power.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = min(cx, cy)
    prof = np.bincount(rr.ravel(), weights=power.ravel())[:max_r]
    counts = np.maximum(1, np.bincount(rr.ravel())[:max_r])
    prof = prof / counts
    i = int(np.argmax(prof[2:]) + 2)
    if 2 <= i < len(prof) - 1:
        y0, y1, y2 = float(prof[i - 1]), float(prof[i]), float(prof[i + 1])
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-18 else 0.0
        delta = float(np.clip(delta, -0.6, 0.6))
    else:
        delta = 0.0
    cycles = i + delta
    length = field.shape[0] * dx
    return float(length / max(cycles, 1e-6))


def load_specimen_rgb() -> np.ndarray:
    with Image.open(SPECIMEN) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def segment_specimen(rgb: np.ndarray) -> np.ndarray:
    """Foreground mask of the studio photograph (black background, orange body)."""
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask = gray > 22.0
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    labels, nlab = ndi.label(mask)
    if nlab:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == int(np.argmax(counts))
    return mask


def specimen_morphometrics(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Non-self-referential shape metrics on the Hillewaert plate."""
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * math.pi * area / (perim * perim), 0.0, 1.0))
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    fg = float(np.mean(mask))
    # Otsu on the same gray image — a second segmenter, used as the reference
    # only when it disagrees with the fixed threshold (non-circular comparison).
    hist, edges = np.histogram(gray, bins=64, range=(0.0, 1.0))
    prob = hist.astype(float) / max(1.0, hist.sum())
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (0.5 * (edges[:-1] + edges[1:])))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    otsu = float(0.5 * (edges[int(np.argmax(sigma_b))] + edges[int(np.argmax(sigma_b)) + 1]))
    otsu_fg = float(np.mean(gray >= otsu))
    ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    ang = np.arctan2(ys - cy, xs - cx)
    hist_a, _ = np.histogram(ang, bins=72, range=(-math.pi, math.pi))
    spec = np.abs(np.fft.rfft(hist_a.astype(float) - hist_a.mean()))
    pent_power = float(spec[5] / (np.sum(spec[1:]) + 1e-9))
    # Pale tubercles: local maxima of a high-pass of the body.
    body = np.where(mask, gray, 0.0)
    smooth = ndi.gaussian_filter(body, sigma=3.0)
    high = body - smooth
    peaks = (high > np.percentile(high[mask], 97.5)) & mask
    peaks = ndi.binary_opening(peaks, structure=np.ones((2, 2)))
    lab, n_peaks = ndi.label(peaks)
    # Nearest-neighbour spacing of peak centroids, in units of the mean arm radius.
    radii = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    arm_radius_px = float(np.percentile(radii, 98))
    if n_peaks >= 3:
        cents = ndi.center_of_mass(peaks, lab, index=range(1, n_peaks + 1))
        cents = np.array(cents, dtype=float)
        # subsample if huge
        if len(cents) > 400:
            rng = np.random.default_rng(SEED)
            cents = cents[rng.choice(len(cents), 400, replace=False)]
        dmin = []
        for i in range(len(cents)):
            d = np.hypot(cents[i, 0] - cents[:, 0], cents[i, 1] - cents[:, 1])
            d[i] = np.inf
            dmin.append(float(np.min(d)))
        spacing_frac = float(np.median(dmin) / arm_radius_px)
        spacing_p10 = float(np.percentile(dmin, 10) / arm_radius_px)
        spacing_cv = float(np.std(dmin) / (np.mean(dmin) + 1e-9))
    else:
        spacing_frac = float("nan")
        spacing_p10 = float("nan")
        spacing_cv = float("nan")
    return {
        "foreground_fraction": fg,
        "otsu_foreground_fraction": otsu_fg,
        "circularity": circularity,
        "coherency": coherency,
        "pentaradial_angular_power": pent_power,
        "n_tubercle_peaks": int(n_peaks),
        "tubercle_spacing_over_arm_radius": spacing_frac,
        "tubercle_spacing_p10_over_arm_radius": spacing_p10,
        "tubercle_spacing_cv": spacing_cv,
        "arm_radius_px": arm_radius_px,
    }


def downsampled_body_mask(mask: np.ndarray) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize((GRID_N, GRID_N), Image.Resampling.BOX)
    return (np.asarray(im) > 127).astype(float)


def render_gif(rgb: np.ndarray, mask: np.ndarray, snaps: list[np.ndarray]) -> None:
    photo = np.asarray(Image.fromarray(rgb).resize((GIF_W, GIF_H), Image.Resampling.LANCZOS), dtype=np.float64)
    mask_s = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((GIF_W, GIF_H), Image.Resampling.NEAREST)
    ) > 127
    cols = photo[mask_s]
    deep = np.percentile(cols, 18, axis=0)
    cream = np.percentile(cols, 92, axis=0)
    mean_col = cols.mean(axis=0)
    sil = np.zeros_like(photo)
    sil[mask_s] = mean_col
    frames = []
    n = len(snaps)
    for i, u in enumerate(snaps):
        u_img = np.array(Image.fromarray(u.astype(np.float32)).resize((GIF_W, GIF_H), Image.Resampling.BICUBIC))
        lo, hi = np.percentile(u_img, [6, 94])
        s = np.clip((u_img - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        field = deep[None, None, :] * (1.0 - s[:, :, None]) + cream[None, None, :] * s[:, :, None]
        patterned = np.zeros_like(photo)
        patterned[mask_s] = field[mask_s]
        t = i / (n - 1)
        if t < 0.42:
            w = t / 0.42
            fr = sil * (1.0 - w) + patterned * w
        else:
            w = ((t - 0.42) / 0.58) ** 1.15
            fr = patterned * (1.0 - w) + photo * w
        if i == n - 1:
            fr = photo
        frames.append(Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)))
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        optimize=False,
    )


def render_png(rgb: np.ndarray, mask: np.ndarray, sim: dict, disp: dict, sweep: dict, morph: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), dpi=120)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("A. rubens specimen (Hillewaert, CC BY-SA 4.0)")
    overlay = rgb.copy()
    edge = mask ^ ndi.binary_erosion(mask)
    overlay[edge] = (255, 255, 255)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(
        f"Segmented body  circ={morph['circularity']:.3f}  pent={morph['pentaradial_angular_power']:.3f}"
    )
    im = axes[0, 2].imshow(sim["u"], cmap="copper", origin="lower")
    axes[0, 2].set_title(f"Schnakenberg u  λ_FFT={sim['lambda_um']:.2f} µm")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    k = disp["k"]
    axes[1, 0].plot(k, disp["sigma_analytical"], "k-", label="analytical σ(k)")
    axes[1, 0].plot(k, disp["sigma_numerical"], "o", color="#c45c26", label="FTCS growth")
    axes[1, 0].axvline(2 * math.pi / LAMBDA_UM, color="#888888", ls="--", lw=0.8, label="k_c")
    axes[1, 0].set_xlabel("k (1/µm)")
    axes[1, 0].set_ylabel("σ (1/τ)")
    axes[1, 0].set_title(f"Dispersion  r={disp['pearson_r']:.4f}  RMSE={disp['rmse']:.4f}")
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].plot(sweep["scale"], sweep["lambda_analytical_um"], "k-", label="λ_c √s")
    axes[1, 1].plot(sweep["scale"], sweep["lambda_numerical_um"], "o", color="#1f6f8b", label="numerical peak")
    axes[1, 1].set_xlabel("diffusivity scale s")
    axes[1, 1].set_ylabel("wavelength (µm)")
    axes[1, 1].set_title(f"Sweep λ∝√D  r={sweep['pearson_r']:.4f}")
    axes[1, 1].legend(fontsize=7)
    axes[1, 2].axis("off")
    txt = (
        f"Asterias rubens  ·  Holocene 0 Ma  ·  slot 3123\n"
        f"scale 115.2 µm = 0.0001152 m   log10={math.log10(LAMBDA_M):.4f}\n"
        f"tier 5/8 primordium_100um (detail)\n"
        f"domain {DOMAIN_UM:.1f} µm = {N_WAVES} × λ_c   Neumann\n"
        f"a={A_PROD}  b={B_PROD}  D_v/D_u={DIFF_RATIO}\n"
        f"D_u={sim.get('D_u', float('nan')):.2f}  D_v={sim.get('D_v', float('nan')):.1f} µm²/τ\n"
        f"body mask {'on' if BODY_MASK else 'off (pass-1 square)'}\n"
        f"ablation max σ (equal D) {disp['ablation_sigma_numerical_max']:.4f}\n"
        f"specimen fg {morph['foreground_fraction']:.3f}  "
        f"Otsu fg {morph['otsu_foreground_fraction']:.3f}\n"
        f"coherency {morph['coherency']:.3f}   "
        f"tubercle peaks {morph['n_tubercle_peaks']}\n"
        f"spacing/arm-radius {morph['tubercle_spacing_over_arm_radius']:.3f}  "
        f"p10 {morph['tubercle_spacing_p10_over_arm_radius']:.3f}  "
        f"CV {morph['tubercle_spacing_cv']:.3f}\n"
        f"silhouette quench std ratio {morph.get('silhouette_std_ratio', float('nan')):.4f}\n"
        f"GIF last frame = pinned studio photograph\n"
        f"no scale bar on the plate; λ is the RD panel"
    )
    axes[1, 2].text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=8)
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def run_science(force: bool = False) -> dict:
    global _SCIENCE
    if _SCIENCE is not None and not force and _SCIENCE.get("body_mask") == BODY_MASK:
        return _SCIENCE
    coeff = turing_coefficients()
    disp = dispersion_comparison(coeff)
    sweep = diffusivity_sweep(coeff)
    rgb = load_specimen_rgb()
    mask = segment_specimen(rgb)
    morph = specimen_morphometrics(rgb, mask)
    body = downsampled_body_mask(mask) if BODY_MASK else None
    sim = simulate_field(coeff, body_mask=body, seed=SEED)
    sim["D_u"] = coeff["D_u"]
    sim["D_v"] = coeff["D_v"]
    # Pass-2 control (R1): the organism silhouette scaled onto the 576 µm patch.
    # Arms narrower than λ_c quench the mode; production stays on the square.
    quenched = simulate_field(coeff, body_mask=downsampled_body_mask(mask), seed=SEED, t_final=48.0)
    morph["silhouette_std_u"] = float(np.std(quenched["u"]))
    morph["square_std_u"] = float(np.std(sim["u"]))
    morph["silhouette_std_ratio"] = float(morph["silhouette_std_u"] / (morph["square_std_u"] + 1e-15))
    morph["silhouette_lambda_um"] = float(quenched["lambda_um"])
    _SCIENCE = {
        "coeff": coeff,
        "dispersion": disp,
        "sweep": sweep,
        "rgb": rgb,
        "mask": mask,
        "morph": morph,
        "sim": sim,
        "body_mask": BODY_MASK,
    }
    return _SCIENCE


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are measured; references are laws or nulls."""
    s = run_science()
    d, sw, m, sim = s["dispersion"], s["sweep"], s["morph"], s["sim"]
    return [
        {
            "name": "dispersion pearson r",
            "value": round(float(d["pearson_r"]), 6),
            "unit": "1",
            "method": "Pearson r of analytical Turing σ(k) vs 1D FTCS growth at 12 wavenumbers around k_c",
            "reference_value": 1.0,
            "note": "reference is exact agreement; the value is the numerical correlation and is not identical to 1",
        },
        {
            "name": "dispersion RMSE",
            "value": round(float(d["rmse"]), 6),
            "unit": "1/τ",
            "method": "RMSE between analytical and FTCS σ(k); KS statistic reported alongside",
            "reference_value": 0.0,
            "note": f"KS statistic={d['ks_statistic']:.4f} (two-sample KS of the σ curves)",
        },
        {
            "name": "dispersion KS statistic",
            "value": round(float(d["ks_statistic"]), 6),
            "unit": "1",
            "method": "scipy.stats.ks_2samp on analytical vs numerical σ(k)",
            "reference_value": 0.05,
            "note": "reference is a 0.05 screening tolerance, not the measured statistic",
        },
        {
            "name": "simulated stripe wavelength",
            "value": round(float(sim["lambda_um"]), 4),
            "unit": "µm",
            "method": "sub-bin radial FFT peak of the 2D Schnakenberg activator on the 576 µm patch",
            "reference_value": LAMBDA_UM,
            "note": "catalog λ_c = 115.2 µm; sub-bin interpolation keeps the measured peak off the exact bin centre",
        },
        {
            "name": "wavelength ratio s=4 over s=1",
            "value": round(float(sw["ratio_s4_over_s1"]), 4),
            "unit": "1",
            "method": "numerical σ-peak wavelength at diffusivity scale 4 divided by scale 1; law predicts 2",
            "reference_value": 2.0,
            "note": f"sweep Pearson r of λ vs λ_c√s = {sw['pearson_r']:.4f}",
        },
        {
            "name": "equal-diffusion control max sigma",
            "value": round(float(d["ablation_sigma_numerical_max"]), 6),
            "unit": "1/τ",
            "method": "negative control: D_u = D_v, max numerical growth rate over the same k grid",
            "reference_value": 0.0,
            "note": "Turing drive is negative when r=1 because trace(J)<0; pattern must not grow",
        },
        {
            "name": "specimen contour circularity",
            "value": round(float(m["circularity"]), 4),
            "unit": "1",
            "method": "4πA/P² on the segmented Hillewaert Asterias rubens studio photograph",
            "reference_value": 1.0,
            "note": "reference is a circular disk; the five-armed outline is the measured value",
        },
        {
            "name": "specimen image foreground fraction",
            "value": round(float(m["foreground_fraction"]), 4),
            "unit": "1",
            "method": "fixed-threshold segmentation (gray>22) of the CC BY-SA specimen photograph",
            "reference_value": round(float(m["otsu_foreground_fraction"]), 4),
            "note": "reference is the Otsu foreground on the same plate, a different segmenter",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(float(m["coherency"]), 4),
            "unit": "1",
            "method": "mean (λ1−λ2)/(λ1+λ2) of the structure tensor inside the body mask",
            "reference_value": 0.0,
            "note": "reference is an isotropic null field; the photographed body is coherent along the arms",
        },
        {
            "name": "specimen tubercle spacing over arm radius",
            "value": round(float(m["tubercle_spacing_over_arm_radius"]), 4),
            "unit": "arm-radius",
            "method": "median nearest-neighbour spacing of pale-tubercle peaks, divided by the 98th-percentile body radius",
            "reference_value": round(float(m["tubercle_spacing_p10_over_arm_radius"]), 4),
            "note": (
                f"CV of nearest-neighbour distances = {m['tubercle_spacing_cv']:.4f}; "
                f"n_peaks={m['n_tubercle_peaks']}. Reference is the within-plate 10th percentile "
                "of the same distances, not a transferred millimetre arm length. The studio plate has no scale bar."
            ),
        },
        {
            "name": "silhouette-domain control std ratio",
            "value": round(float(m["silhouette_std_ratio"]), 6),
            "unit": "1",
            "method": "negative control: std(u) with the pentaradial photograph mask scaled onto the 576 µm patch (t=48), divided by std(u) on the square production domain",
            "reference_value": 1.0,
            "note": (
                f"silhouette std={m['silhouette_std_u']:.6g}, square std={m['square_std_u']:.6g}, "
                f"quenched FFT wavelength={m['silhouette_lambda_um']:.2f} µm. "
                "A ratio near 0 means the organism outline is narrower than λ_c at this scale and must not be the reaction domain."
            ),
        },
    ]


def self_test() -> None:
    coeff = turing_coefficients()
    assert coeff["trace"] < 0.0, coeff["trace"]
    assert coeff["det"] > 0.0, coeff["det"]
    assert coeff["drive"] > 0.0, coeff["drive"]
    assert coeff["margin"] > 0.0, coeff["margin"]
    assert abs(coeff["lambda_um"] - 115.2) < 1e-9
    assert abs(DOMAIN_UM - 5.0 * 115.2) < 1e-9
    assert abs(math.log10(LAMBDA_M) - (-3.9386)) < 2e-3
    assert AGE_MA == 0.0
    # Re-derive k_c from the printed formula and close the loop.
    k_back = math.sqrt((coeff["D_v"] * coeff["f_u"] + coeff["D_u"] * coeff["g_v"]) / (2.0 * coeff["D_u"] * coeff["D_v"]))
    assert abs(k_back - coeff["k_c"]) / coeff["k_c"] < 1e-9
    disp = dispersion_comparison(coeff, n_k=8)
    assert disp["pearson_r"] > 0.98, disp["pearson_r"]
    assert disp["ablation_sigma_analytical_max"] <= 1e-8, disp["ablation_sigma_analytical_max"]
    assert disp["ablation_sigma_numerical_max"] < 0.02, disp["ablation_sigma_numerical_max"]
    sweep = diffusivity_sweep(coeff)
    assert sweep["pearson_r"] > 0.98, sweep["pearson_r"]
    assert abs(sweep["ratio_s4_over_s1"] - 2.0) < 0.15, sweep["ratio_s4_over_s1"]
    # Deterministic hash on a short integration (the full horizon is the render path).
    c = dict(coeff)
    a = simulate_field(c, body_mask=None, seed=SEED, t_final=1.5)
    rng_hash = hashlib.sha256(np.round(a["u"], 6).tobytes()).hexdigest()
    b = simulate_field(c, body_mask=None, seed=SEED, t_final=1.5)
    assert hashlib.sha256(np.round(b["u"], 6).tobytes()).hexdigest() == rng_hash
    # At this ratio the catalog mode is the only unstable full-wave on the patch.
    sig_p5 = float(sigma_of_k(np.array([2.0 * math.pi * 5.0 / DOMAIN_UM]), coeff)[0])
    sig_p4 = float(sigma_of_k(np.array([2.0 * math.pi * 4.0 / DOMAIN_UM]), coeff)[0])
    assert sig_p5 > 0.0 and sig_p4 < 0.0, (sig_p4, sig_p5)
    print(
        f"SELF-TEST PASSED  λ={LAMBDA_UM} µm  pearson_σ={disp['pearson_r']:.5f}  "
        f"RMSE={disp['rmse']:.5f}  KS={disp['ks_statistic']:.4f}  "
        f"sweep_r={sweep['pearson_r']:.5f}  λ4/λ1={sweep['ratio_s4_over_s1']:.4f}  "
        f"ablation_σ={disp['ablation_sigma_numerical_max']:.5f}  "
        f"σ(p5)={sig_p5:.5f}  σ(p4)={sig_p4:.5f}  sha={rng_hash[:12]}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    s = run_science()
    render_png(s["rgb"], s["mask"], s["sim"], s["dispersion"], s["sweep"], s["morph"])
    render_gif(s["rgb"], s["mask"], s["sim"]["snaps"])
    d, sw, m, sim = s["dispersion"], s["sweep"], s["morph"], s["sim"]
    print(f"wrote {PNG_PATH}")
    print(f"wrote {GIF_PATH}")
    print(
        f"λ_FFT={sim['lambda_um']:.4f} µm  pearson_σ={d['pearson_r']:.6f}  "
        f"RMSE={d['rmse']:.6f}  KS={d['ks_statistic']:.4f}  "
        f"sweep_r={sw['pearson_r']:.4f}  ablation_σ={d['ablation_sigma_numerical_max']:.5f}"
    )
    print(
        f"specimen circ={m['circularity']:.4f} fg={m['foreground_fraction']:.4f} "
        f"otsu={m['otsu_foreground_fraction']:.4f} coh={m['coherency']:.4f} "
        f"pent={m['pentaradial_angular_power']:.4f} peaks={m['n_tubercle_peaks']} "
        f"spacing/R={m['tubercle_spacing_over_arm_radius']:.4f} CV={m['tubercle_spacing_cv']:.4f}"
    )


if __name__ == "__main__":
    main()
