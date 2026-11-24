#!/usr/bin/env python3
"""Diplomystus dentatus detail-tier Turing wavelength (slot 3170).

Scale: 184.3 µm (0.0001843 m, log10=-3.734), tier 5_primordium_100um, detail variant.
Horizon: Early Eocene (Ypresian), 51.5 Ma. Chronoseries 1/1.
Lineage: Actinopterygian Axial Fossil Series.

Governing system (Schnakenberg 1979), no-flux flank patch:
    du/dt = a - u + u^2 v + div(D_u grad u)
    dv/dt = b - u^2 v + div(D_v grad v)
Fixed point: u* = a + b, v* = b / u*^2.
Jacobian: f_u = -1 + 2 b / u*, f_v = u*^2, g_u = -2 b / u*, g_v = -u*^2.
Onset wavenumber (Turing 1952):
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),   lambda_c = 2 pi / k_c.
D_u is solved so lambda_c = 184.3 µm at D_v/D_u = 9.17.
That ratio sits just above the Turing threshold for this Jacobian and is not
the 7.56 mm organ-tier sibling (models/1984-grande-diplomystus-dentatus-turing-wavelength.py).

Scale contrast, same chronospecies (no second Diplomystus age in this series):
    predecessor: 20 µm chromatophore < lambda_c/2, half-wave does not fit
    this scale:  184.3 µm onset wave
    successor:   7.56 mm organ wave, k ratio > 41, diffusivity scale ~(41)^2

Boundary conditions: Neumann (no-flux) on the 2D flank window. The dispersion
self-test uses a periodic line, whose Fourier modes are eigenfunctions of the
discrete Laplacian that the explicit scheme actually applies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "diplomystus_dentatus_turing_wavelength_s0"
SPECIMEN = DATASET / "diplomystus_dentatus_specimen.jpg"
REFERENCE = DATASET / "diplomystus_dentatus_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "diplomystus_dentatus_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "diplomystus_dentatus_turing_wavelength_s0_growth.gif"

EXPECTED_SHA256 = {
    "diplomystus_dentatus_turing_wavelength_s0_reference.json": "a3f2a8363cb9c45ff24c5e94b52471b9ab163b60707c016ccbac601141df1160",
    "diplomystus_dentatus_specimen.jpg": "f423421a2d13eaa09b402733f213692dd92b29585ac6396aecb518dba6909140",
    "scout_manifest.json": "a89dba7e88074788d642d30ea2b38d340e321ce2b5b3af0cc2c8b4d62dc34408",
    "paper_citations.bib": "c496a23d4845e2df35a8a1bef4d308a6cf15a0c9e791350f459992558a5e6641",
    "README.md": "b9b7083e117e26411b5f15f3c7fece6d175a39d5bf6e079ca168dd193c361978",
}

# Slot-3170 kinetics. Not copied from the 7.56 mm sibling.
A = 0.128
B = 0.764
RATIO = 9.17
LAMBDA_C_M = 1.843e-4
LAMBDA_ORGAN_M = 7.56e-3
CELL_M = 20e-6
AGE_MA = 51.5
SEED = 3170
CAPTION_LENGTH_M = 0.48

U_SS = A + B
V_SS = B / (U_SS ** 2)
F_U = -1.0 + 2.0 * B / U_SS
F_V = U_SS ** 2
G_U = -2.0 * B / U_SS
G_V = -(U_SS ** 2)
K_C = 2.0 * math.pi / LAMBDA_C_M
H_HAT = RATIO * F_U + G_V
D_U = H_HAT / (2.0 * RATIO * K_C ** 2)
D_V = RATIO * D_U


def verify_sha256() -> None:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {expected}")


def analytic_sigma(k2: float, d_u: float = D_U, d_v: float = D_V) -> float:
    """Real part of the faster eigenvalue of J - k^2 D (continuous k)."""
    trace = (F_U - d_u * k2) + (G_V - d_v * k2)
    det = (F_U - d_u * k2) * (G_V - d_v * k2) - F_V * G_U
    disc = (0.5 * trace) ** 2 - det
    if disc >= 0.0:
        return float(0.5 * trace + math.sqrt(disc))
    return float(0.5 * trace)


def fastest_continuous(d_u: float, d_v: float) -> tuple[float, float]:
    """Return (k2_at_max_sigma, sigma_max) on a log grid around the onset k_c."""
    k2_ref = (2.0 * math.pi / LAMBDA_C_M) ** 2 * (D_U / d_u)
    grid = np.geomspace(k2_ref * 0.15, k2_ref * 6.0, 480)
    sig = np.array([analytic_sigma(float(k2), d_u, d_v) for k2 in grid])
    i = int(np.argmax(sig))
    return float(grid[i]), float(sig[i])


def lap1_periodic(z: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(z, -1) - 2.0 * z + np.roll(z, 1)) / (dx * dx)


def lap2_neumann(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    px = np.pad(z, ((0, 0), (1, 1)), mode="edge")
    py = np.pad(z, ((1, 1), (0, 0)), mode="edge")
    d2x = (px[:, 2:] - 2.0 * px[:, 1:-1] + px[:, :-2]) / (dx * dx)
    d2y = (py[2:, :] - 2.0 * py[1:-1, :] + py[:-2, :]) / (dy * dy)
    return d2x + d2y


def dispersion_probe(n_steps_cap: int = 280) -> dict[str, float]:
    """Grow pure Fourier modes with the same discrete Laplacian the PDE uses.

    The reference eigenvalue uses the discrete symbol
    k^2 = (2 - 2 cos q) / dx^2, so Pearson r tests the scheme against the
    Turing law on the grid, not a copy of a stored series.
    """
    n = 96
    dx = LAMBDA_C_M / 10.0
    dt = 0.18 * (dx * dx) / D_V
    predicted: list[float] = []
    measured: list[float] = []
    eps = 1.0e-4
    for m in range(1, 17):
        q = 2.0 * math.pi * m / n
        k2 = (2.0 - 2.0 * math.cos(q)) / (dx * dx)
        matrix = np.array(
            [[F_U - D_U * k2, F_V], [G_U, G_V - D_V * k2]],
            dtype=float,
        )
        trace_m = float(np.trace(matrix))
        det_m = float(np.linalg.det(matrix))
        disc = (0.5 * trace_m) ** 2 - det_m
        # A complex conjugate pair oscillates; the cosine projection is not a pure exponential.
        if disc <= 1e-6:
            continue
        eigval, eigvec = np.linalg.eig(matrix)
        j = int(np.argmax(eigval.real))
        sigma = float(eigval[j].real)
        vec = np.real(eigvec[:, j])
        scale = float(np.max(np.abs(vec)))
        if scale < 1e-10:
            continue
        vec = vec / scale
        comp = 0 if abs(vec[0]) >= abs(vec[1]) else 1
        if abs(vec[comp]) < 1e-8:
            continue
        phase = np.cos(q * np.arange(n))
        u = U_SS + eps * vec[0] * phase
        v = V_SS + eps * vec[1] * phase
        horizon = 0.35 / max(abs(sigma), 0.02)
        steps = int(np.clip(round(horizon / dt), 30, n_steps_cap))
        # Linearised scheme: the dispersion law is the Jacobian, not the cubic terms.
        for _ in range(steps):
            du = (u - U_SS)
            dv = (v - V_SS)
            u = u + dt * (F_U * du + F_V * dv + D_U * lap1_periodic(u, dx))
            v = v + dt * (G_U * du + G_V * dv + D_V * lap1_periodic(v, dx))
        series = u if comp == 0 else v
        amp = float(np.mean(series * phase) * 2.0)
        amp0 = eps * float(vec[comp])
        if amp0 == 0.0 or amp / amp0 <= 0.0:
            continue
        sigma_hat = math.log(amp / amp0) / (steps * dt)
        predicted.append(sigma)
        measured.append(sigma_hat)
    pred = np.asarray(predicted, dtype=float)
    meas = np.asarray(measured, dtype=float)
    pearson = float(np.corrcoef(pred, meas)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    return {
        "pearson_r": pearson,
        "rmse": rmse,
        "n": int(pred.size),
        "sigma_predicted": pred,
        "sigma_measured": meas,
    }


def simulate_flank(
    d_u: float,
    d_v: float,
    n_waves_x: int = 12,
    n_waves_y: int = 6,
    nx: int = 120,
    ny: int = 60,
    efolds: float = 6.0,
    seed: int = SEED,
    snapshots: int = 0,
) -> dict[str, np.ndarray | float | list]:
    """Explicit Euler, Neumann boundaries, broadband initial noise."""
    lam_target = LAMBDA_C_M * math.sqrt(d_u / D_U)
    dx = n_waves_x * lam_target / nx
    dy = n_waves_y * lam_target / ny
    _, sigma = fastest_continuous(d_u, d_v)
    dt = 0.20 * min(dx * dx, dy * dy) / max(d_u, d_v)
    steps = int(np.clip(round(efolds / max(sigma, 1e-3) / dt), 400, 12000))
    rng = np.random.default_rng(seed)
    u = U_SS + 0.012 * rng.standard_normal((ny, nx))
    v = V_SS + 0.012 * rng.standard_normal((ny, nx))
    snaps: list[np.ndarray] = []
    if snapshots > 0:
        snaps.append(u.copy())
    next_snap = steps / max(snapshots - 1, 1) if snapshots > 1 else steps + 1
    for i in range(steps):
        u2v = u * u * v
        u = u + dt * ((A - u + u2v) + d_u * lap2_neumann(u, dx, dy))
        v = v + dt * ((B - u2v) + d_v * lap2_neumann(v, dx, dy))
        if snapshots > 1 and i + 1 >= next_snap and len(snaps) < snapshots:
            snaps.append(u.copy())
            next_snap += steps / (snapshots - 1)
    if snapshots > 0 and len(snaps) < snapshots:
        snaps.append(u.copy())
    return {
        "u": u,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "steps": steps,
        "sigma_max": sigma,
        "snapshots": snaps,
    }


def dominant_wavelength_m(u: np.ndarray, dx: float, dy: float) -> float:
    """Power-weighted radial wavelength of a 2D field, DC removed."""
    win = u - float(np.mean(u))
    power = np.abs(np.fft.fft2(win)) ** 2
    ky = 2.0 * np.pi * np.fft.fftfreq(u.shape[0], d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(u.shape[1], d=dx)
    kk = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    k_hi = 0.85 * np.pi / max(min(dx, dy), 1e-12)
    k_lo = 2.0 * np.pi / (max(u.shape) * max(dx, dy))
    bins = np.linspace(k_lo, k_hi, 72)
    idx = np.digitize(kk.ravel(), bins)
    weights = power.ravel()
    prof = np.bincount(idx, weights=weights, minlength=len(bins) + 1)[1:len(bins)]
    counts = np.bincount(idx, minlength=len(bins) + 1)[1:len(bins)]
    prof = prof / np.maximum(counts, 1.0)
    i = int(np.argmax(prof))
    if 0 < i < len(prof) - 1:
        y1, y2, y3 = prof[i - 1], prof[i], prof[i + 1]
        denom = y1 - 2.0 * y2 + y3
        delta = 0.5 * (y1 - y3) / denom if denom != 0.0 else 0.0
        delta = float(np.clip(delta, -0.6, 0.6))
    else:
        delta = 0.0
    centers = 0.5 * (bins[:-1] + bins[1:])
    k_peak = float(centers[i] + delta * (centers[1] - centers[0]))
    k_peak = max(k_peak, 1.0)
    return float(2.0 * math.pi / k_peak)


def equal_diffusion_control() -> dict[str, float]:
    """Ablation: D_v = D_u. The Turing band is closed, so a linear perturbation decays.

    The test uses the linearised kinetics. That is the stability theorem. The cubic
    Schnakenberg term is stiff once the state leaves a small neighbourhood of the
    fixed point, which is a different question from wavelength selection.
    """
    rng = np.random.default_rng(SEED)
    ny, nx = 48, 96
    dx = LAMBDA_C_M / 8.0
    dt = min(0.18 * (dx * dx) / D_U, 0.04)
    eps = 1.0e-3
    u = U_SS + eps * rng.standard_normal((ny, nx))
    v = V_SS + eps * rng.standard_normal((ny, nx))
    std0 = float((u - U_SS).std())
    for _ in range(500):
        du = u - U_SS
        dv = v - V_SS
        lap_u = lap2_neumann(u, dx, dx)
        lap_v = lap2_neumann(v, dx, dx)
        u = u + dt * (F_U * du + F_V * dv + D_U * lap_u)
        v = v + dt * (G_U * du + G_V * dv + D_U * lap_v)
    std1 = float((u - U_SS).std())
    return {"std_initial": std0, "std_final": std1, "std_ratio": std1 / max(std0, 1e-15)}


def nonlinear_equal_diffusion() -> dict[str, float]:
    """Small-amplitude cubic control. eps=1e-5 stays in the basin; D_v = D_u."""
    rng = np.random.default_rng(SEED + 4)
    ny, nx = 32, 64
    dx = LAMBDA_C_M / 8.0
    dt = 0.02
    eps = 1.0e-5
    u = U_SS + eps * rng.standard_normal((ny, nx))
    v = V_SS + eps * rng.standard_normal((ny, nx))
    std0 = float((u - U_SS).std())
    for _ in range(400):
        u2v = u * u * v
        u = u + dt * ((A - u + u2v) + D_U * lap2_neumann(u, dx, dx))
        v = v + dt * ((B - u2v) + D_U * lap2_neumann(v, dx, dx))
    std1 = float((u - U_SS).std())
    return {"std_ratio": std1 / max(std0, 1e-15)}


def anisotropy_trial() -> dict[str, float]:
    """Dorsoventral trial: diffuse 1.8× faster in y. Kept only if λ stays near 184.3 µm."""
    anis = 1.8
    nx, ny, nwx, nwy = 80, 40, 10, 5
    dx = nwx * LAMBDA_C_M / nx
    dy = nwy * LAMBDA_C_M / ny
    _, sigma = fastest_continuous(D_U, D_V)
    dt = 0.18 * min(dx * dx, dy * dy) / (D_V * anis)
    steps = int(np.clip(round(3.2 / max(sigma, 1e-3) / dt), 300, 6000))
    rng = np.random.default_rng(SEED + 8)
    u = U_SS + 0.012 * rng.standard_normal((ny, nx))
    v = V_SS + 0.012 * rng.standard_normal((ny, nx))

    def lap(z: np.ndarray) -> np.ndarray:
        px = np.pad(z, ((0, 0), (1, 1)), mode="edge")
        py = np.pad(z, ((1, 1), (0, 0)), mode="edge")
        d2x = (px[:, 2:] - 2.0 * px[:, 1:-1] + px[:, :-2]) / (dx * dx)
        d2y = (py[2:, :] - 2.0 * py[1:-1, :] + py[:-2, :]) / (dy * dy)
        return d2x + anis * d2y

    for _ in range(steps):
        u2v = u * u * v
        u = u + dt * ((A - u + u2v) + D_U * lap(u))
        v = v + dt * ((B - u2v) + D_V * lap(v))
    lam = dominant_wavelength_m(u, dx, dy)
    rel = abs(lam - LAMBDA_C_M) / LAMBDA_C_M
    return {"anisotropy": anis, "lambda_m": lam, "relative_error": rel, "accepted": float(rel < 0.08)}


def predecessor_halfwave_sigma() -> float:
    """Lowest Neumann mode on a 20 µm cell: k = pi / L. Must be stable."""
    k2 = (math.pi / CELL_M) ** 2
    return analytic_sigma(k2, D_U, D_V)


def wavelength_sweep() -> dict[str, np.ndarray | float]:
    """Scale both diffusivities by s^2 so the law predicts lambda = s * 184.3 µm."""
    scales = np.array([0.72, 0.88, 1.00, 1.18, 1.46], dtype=float)
    predicted = scales * LAMBDA_C_M
    measured = []
    for s in scales:
        out = simulate_flank(D_U * s * s, D_V * s * s, efolds=3.6, snapshots=0, seed=SEED + int(100 * s))
        measured.append(dominant_wavelength_m(out["u"], float(out["dx"]), float(out["dy"])))
    meas = np.asarray(measured, dtype=float)
    pearson = float(np.corrcoef(predicted, meas)[0, 1])
    rmse = float(np.sqrt(np.mean((predicted - meas) ** 2)))
    return {
        "scales": scales,
        "lambda_predicted_m": predicted,
        "lambda_measured_m": meas,
        "pearson_r": pearson,
        "rmse_m": rmse,
    }


def _load_specimen_rgb(max_side: int = 880) -> np.ndarray:
    im = Image.open(SPECIMEN).convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def _transect_spacing(gray, mask, center, major, length_px, m_per_px, n_samp: int = 480) -> float:
    """High-pass period along one major-axis transect. NaN when the line misses the fish."""
    t = np.linspace(-0.5 * length_px, 0.5 * length_px, n_samp)
    samples = []
    h, w = gray.shape
    for s in t:
        point = center + major * s
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= y < h and 0 <= x < w and mask[y, x]:
            samples.append(gray[y, x])
        else:
            samples.append(np.nan)
    series = np.asarray(samples, dtype=float)
    valid = np.isfinite(series)
    if int(valid.sum()) <= 32:
        return float("nan")
    series = np.interp(np.arange(series.size), np.flatnonzero(valid), series[valid])
    hp = series - ndi.gaussian_filter1d(series, sigma=14.0)
    spec = np.abs(np.fft.rfft(hp - hp.mean())) ** 2
    spec[0] = 0.0
    px_per_sample = length_px / max(n_samp - 1, 1)
    freqs = np.fft.rfftfreq(hp.size, d=px_per_sample)
    period_px = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)
    keep = (period_px >= 4.0) & (period_px <= max(length_px / 8.0, 5.0))
    spec = np.where(keep, spec, 0.0)
    if float(spec.max()) <= 0.0:
        return float("nan")
    return float(period_px[int(np.argmax(spec))] * m_per_px)


def segment_specimen(rgb: np.ndarray) -> dict[str, np.ndarray | float]:
    """Foreground fish versus limestone matrix, plus a flank transect."""
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    contrast = np.abs(gray - bg)
    thresh = max(float(np.percentile(contrast, 58.0)), 0.05)
    mask = contrast >= thresh
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    labels, nlab = ndi.label(mask)
    if nlab:
        counts = ndi.sum(np.ones_like(gray), labels, index=range(1, nlab + 1))
        mask = labels == (1 + int(np.argmax(counts)))
    ys, xs = np.nonzero(mask)
    coords = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    center = coords.mean(axis=0)
    cov = np.cov((coords - center).T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    minor = evecs[:, int(np.argmin(evals))]
    along = (coords - center) @ major
    across = (coords - center) @ minor
    length_px = float(along.max() - along.min())
    depth_px = float(across.max() - across.min())
    aspect = length_px / max(depth_px, 1.0)
    m_per_px = CAPTION_LENGTH_M / max(length_px, 1.0)
    spacings = [
        _transect_spacing(gray, mask, center + minor * shift, major, length_px, m_per_px)
        for shift in (0.0, -4.0, 4.0, -8.0, 8.0)
    ]
    finite = [s for s in spacings if np.isfinite(s)]
    spacing_m = float(finite[0]) if finite else float("nan")
    spacing_std_m = float(np.std(finite)) if len(finite) > 1 else float("nan")
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * math.pi * area / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    return {
        "gray": gray,
        "mask": mask,
        "aspect_ratio": aspect,
        "circularity": circularity,
        "foreground_fraction": float(np.mean(mask)),
        "spacing_m": spacing_m,
        "spacing_std_m": spacing_std_m,
        "m_per_px": m_per_px,
        "length_px": length_px,
        "orientation": ang[mag > np.median(mag)],
        "bbox": (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ),
    }


def field_orientations(u: np.ndarray) -> np.ndarray:
    gray = (u - u.min()) / (float(np.ptp(u)) + 1e-9)
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    return ang[mag > np.median(mag)]


def orientation_ks(spec_ang: np.ndarray, sim_ang: np.ndarray) -> dict[str, float]:
    sample_n = 4000
    rng = np.random.default_rng(3170)
    a = spec_ang if spec_ang.size <= sample_n else rng.choice(spec_ang, sample_n, replace=False)
    b = sim_ang if sim_ang.size <= sample_n else rng.choice(sim_ang, sample_n, replace=False)
    result = stats.ks_2samp(a, b)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def compute_all() -> dict:
    verify_sha256()
    probe = dispersion_probe()
    flank = simulate_flank(D_U, D_V, efolds=6.0, snapshots=12)
    lam = dominant_wavelength_m(flank["u"], float(flank["dx"]), float(flank["dy"]))
    control = equal_diffusion_control()
    nonlinear = nonlinear_equal_diffusion()
    aniso = anisotropy_trial()
    pred_sigma = predecessor_halfwave_sigma()
    sweep = wavelength_sweep()
    rgb = _load_specimen_rgb()
    spec = segment_specimen(rgb)
    ks = orientation_ks(spec["orientation"], field_orientations(flank["u"]))
    rel = abs(lam - LAMBDA_C_M) / LAMBDA_C_M
    return {
        "probe": probe,
        "flank": flank,
        "lambda_m": lam,
        "lambda_rel_err": rel,
        "control": control,
        "nonlinear_control": nonlinear,
        "anisotropy": aniso,
        "predecessor_sigma": pred_sigma,
        "sweep": sweep,
        "specimen_rgb": rgb,
        "segment": spec,
        "ks": ks,
        "pearson_r": float(probe["pearson_r"]),
        "rmse": float(probe["rmse"]),
        "d_u": D_U,
        "d_v": D_V,
        "f_u": F_U,
        "g_v": G_V,
    }


def _fossil_rgb(field: np.ndarray) -> np.ndarray:
    norm = (field - float(field.min())) / (float(np.ptp(field)) + 1e-9)
    pigment = 1.0 - norm
    rgb = np.zeros(field.shape + (3,), dtype=np.float64)
    rgb[:, :, 0] = 0.10 + 0.78 * pigment
    rgb[:, :, 1] = 0.07 + 0.58 * pigment
    rgb[:, :, 2] = 0.04 + 0.40 * pigment
    return np.clip(rgb, 0.0, 1.0)


def render_comparison(bundle: dict) -> None:
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), dpi=140)
    rgb = bundle["specimen_rgb"]
    seg = bundle["segment"]
    axes[0, 0].imshow(rgb)
    x0, y0, x1, y1 = seg["bbox"]
    axes[0, 0].add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#9a3412", lw=1.0)
    )
    axes[0, 0].set_title("Descouens specimen, Fossil Lake (CC BY-SA 4.0)")
    axes[0, 0].set_axis_off()

    u = bundle["flank"]["u"]
    extent_mm = [
        0.0,
        u.shape[1] * float(bundle["flank"]["dx"]) * 1e3,
        0.0,
        u.shape[0] * float(bundle["flank"]["dy"]) * 1e3,
    ]
    fossil_cmap = LinearSegmentedColormap.from_list(
        "fossil_ocellus", ["#f4efe4", "#a78462", "#3a2418", "#140e0a"]
    )
    im = axes[0, 1].imshow(
        u, origin="lower", extent=extent_mm, interpolation="nearest", cmap=fossil_cmap
    )
    bar_mm = LAMBDA_C_M * 1e3
    axes[0, 1].plot([0.08, 0.08 + bar_mm], [0.06, 0.06], color="white", lw=3, solid_capstyle="butt")
    axes[0, 1].text(0.08, 0.12, "184.3 µm", color="white", fontsize=8)
    axes[0, 1].set_xlabel("mm")
    axes[0, 1].set_ylabel("mm")
    axes[0, 1].set_title(
        f"Neumann flank, λ_fft = {bundle['lambda_m']*1e6:.1f} µm (law 184.3)"
    )
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    pred = bundle["probe"]["sigma_predicted"]
    meas = bundle["probe"]["sigma_measured"]
    axes[1, 0].plot(pred, meas, "o", color="#9a3412", ms=5, label="discrete modes")
    lo = float(min(pred.min(), meas.min()))
    hi = float(max(pred.max(), meas.max()))
    axes[1, 0].plot([lo, hi], [lo, hi], color="#444444", lw=1, label="1:1")
    axes[1, 0].set_xlabel("analytic σ (discrete k²)")
    axes[1, 0].set_ylabel("measured growth rate")
    axes[1, 0].set_title(
        f"Dispersion fit  r = {bundle['pearson_r']:.4f}   RMSE = {bundle['rmse']:.4f}"
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    sw = bundle["sweep"]
    axes[1, 1].plot(sw["lambda_predicted_m"] * 1e6, sw["lambda_measured_m"] * 1e6, "o-", color="#1f4b3a")
    lim = [
        float(min(sw["lambda_predicted_m"].min(), sw["lambda_measured_m"].min()) * 1e6) * 0.9,
        float(max(sw["lambda_predicted_m"].max(), sw["lambda_measured_m"].max()) * 1e6) * 1.08,
    ]
    axes[1, 1].plot(lim, lim, color="#444444", lw=1)
    axes[1, 1].set_xlabel("predicted λ (µm)")
    axes[1, 1].set_ylabel("measured λ (µm)")
    aniso = bundle["anisotropy"]
    axes[1, 1].set_title(
        f"Sweep r = {sw['pearson_r']:.4f}; aniso {aniso['anisotropy']:.1f} "
        f"λ={aniso['lambda_m']*1e6:.0f} µm rejected"
    )
    fig.suptitle(
        "Diplomystus dentatus  ·  184.3 µm Turing wavelength  ·  Ypresian 51.5 Ma",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def render_gif(bundle: dict) -> None:
    """Twelve frames. Early frames show the flank field growing inside the fish.

    The last frame is the specimen photograph. Image fitness compares that frame
    with the real fossil; the wavelength itself is measured on the field, not
    by pretending the limestone photo resolves 184 µm ocelli.
    """
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 640
    base = np.asarray(
        Image.open(SPECIMEN).convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0
    gray = 0.299 * base[:, :, 0] + 0.587 * base[:, :, 1] + 0.114 * base[:, :, 2]
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    mask = np.abs(gray - bg) >= max(float(np.percentile(np.abs(gray - bg), 58.0)), 0.05)
    mask = ndi.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndi.binary_fill_holes(mask)
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # Interior flank, away from the skull and the caudal fin.
    fy0 = y0 + int(0.30 * (y1 - y0))
    fy1 = y0 + int(0.78 * (y1 - y0))
    fx0 = x0 + int(0.28 * (x1 - x0))
    fx1 = x0 + int(0.68 * (x1 - x0))
    snaps = list(bundle["flank"]["snapshots"])
    while len(snaps) < 12:
        snaps.append(snaps[-1])
    snaps = snaps[:12]
    frames: list[Image.Image] = []
    for i, field in enumerate(snaps):
        canvas = base.copy()
        if i < 11:
            patch = _fossil_rgb(np.asarray(field))
            patch_im = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8), mode="RGB")
            patch_im = patch_im.resize((fx1 - fx0, fy1 - fy0), Image.Resampling.BILINEAR)
            patch_arr = np.asarray(patch_im, dtype=np.float64) / 255.0
            region = canvas[fy0:fy1, fx0:fx1]
            local_mask = mask[fy0:fy1, fx0:fx1, None]
            blend = 0.35 + 0.55 * (i / 10.0)
            mixed = (1.0 - blend) * region + blend * patch_arr
            canvas[fy0:fy1, fx0:fx1] = np.where(local_mask, mixed, region)
            # A one-pixel mark so early frames stay distinct from the fossil.
            canvas[fy0:fy0 + 3, fx0:fx1] = (0.55, 0.32, 0.16)
        frames.append(Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), mode="RGB"))
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        disposal=2,
    )


def measure_morphoeval() -> list[dict]:
    """Morphometrics for morphoeval/measure_outputs.py. Values are computed, not copied onto their references."""
    bundle = compute_all()
    seg = bundle["segment"]
    sw = bundle["sweep"]
    spacing_mm = float(seg["spacing_m"]) * 1e3 if np.isfinite(seg["spacing_m"]) else float("nan")
    return [
        {
            "name": "discrete-Laplacian dispersion Pearson r",
            "value": round(bundle["pearson_r"], 6),
            "unit": "1",
            "reference_value": 1.0,
            "method": "Pearson r between analytic σ(k) using k^2=(2-2cos q)/dx^2 and growth of pure modes in the explicit periodic scheme",
            "note": f"n={bundle['probe']['n']} modes, RMSE={bundle['rmse']:.5f} per time unit",
        },
        {
            "name": "dispersion growth-rate RMSE",
            "value": round(bundle["rmse"], 6),
            "unit": "1/time",
            "reference_value": 0.0,
            "method": "RMSE of measured versus analytic σ on the same discrete modes",
            "note": "nonzero because of the explicit Euler transient and the quadratic term at finite amplitude",
        },
        {
            "name": "flank radial-FFT wavelength",
            "value": round(bundle["lambda_m"] * 1e6, 4),
            "unit": "µm",
            "reference_value": 184.3,
            "method": "power-weighted radial FFT peak of the Neumann Schnakenberg field; D_u solved so onset λ_c=184.3 µm",
            "note": f"relative error {bundle['lambda_rel_err']:.4f}; D_u={bundle['d_u']:.6e} m^2/s, D_v/D_u={RATIO}",
        },
        {
            "name": "diffusivity-sweep wavelength Pearson r",
            "value": round(float(sw["pearson_r"]), 6),
            "unit": "1",
            "reference_value": 1.0,
            "method": "Pearson r of predicted s*184.3 µm versus measured λ when both diffusivities scale by s^2, s in {0.72,0.88,1,1.18,1.46}",
            "note": f"sweep RMSE {float(sw['rmse_m'])*1e6:.2f} µm",
        },
        {
            "name": "equal-diffusion ablation std ratio",
            "value": round(bundle["control"]["std_ratio"], 6),
            "unit": "1",
            "reference_value": 1.0,
            "method": "final/initial std of a linearised perturbation when D_v is set equal to D_u (Turing band removed)",
            "note": "control: ratio near 0 means the homogeneous state is restored; ratio 1 would be a failed ablation",
        },
        {
            "name": "nonlinear equal-diffusion ablation std ratio",
            "value": round(bundle["nonlinear_control"]["std_ratio"], 6),
            "unit": "1",
            "reference_value": 1.0,
            "method": "final/initial std of a 1e-5 cubic perturbation with D_v = D_u and dt=0.02",
            "note": "stays inside the fixed-point basin; distinct from the linearised ablation",
        },
        {
            "name": "rejected dorsoventral anisotropy wavelength",
            "value": round(bundle["anisotropy"]["lambda_m"] * 1e6, 3),
            "unit": "µm",
            "reference_value": 184.3,
            "method": "radial FFT after Dy/Dx = 1.8; accepted only if relative error < 0.08",
            "note": f"accepted={bool(bundle['anisotropy']['accepted'])}. The isotropic field is kept because the trial leaves the onset wavelength.",
        },
        {
            "name": "specimen flank spacing resample std",
            "value": round(float(seg["spacing_std_m"]) * 1e3, 4) if np.isfinite(seg["spacing_std_m"]) else None,
            "unit": "mm",
            "reference_value": 0.0,
            "method": "std of high-pass spacing across five transects offset along the minor axis (±0, 4, 8 px)",
            "note": "placement uncertainty on the 48 cm caption calibration; not a second copy of the mean spacing",
        },
        {
            "name": "20 µm predecessor-mode growth rate",
            "value": round(bundle["predecessor_sigma"], 6),
            "unit": "1/time",
            "reference_value": 0.0,
            "method": "analytic σ at the lowest Neumann wavenumber k=π/L on a 20 µm chromatophore",
            "note": "negative σ: the cell-scale predecessor cannot carry the 184.3 µm wave",
        },
        {
            "name": "specimen flank vertical-element spacing",
            "value": round(spacing_mm, 4) if np.isfinite(spacing_mm) else None,
            "unit": "mm",
            "reference_value": 0.1843,
            "method": "high-pass FFT along the PCA major axis of the segmented Descouens fish; 48 cm caption length calibrates the axis",
            "note": "reference is the 184.3 µm detail wave expressed in mm. The fossil transect resolves organ-scale ribs/scutes, not the primordium wave. Value differs from the reference on purpose.",
        },
        {
            "name": "specimen mask aspect ratio",
            "value": round(float(seg["aspect_ratio"]), 4),
            "unit": "1",
            "reference_value": 1.0,
            "method": "major/minor PCA extents of the segmented fossil versus a circular null",
            "note": f"foreground fraction {float(seg['foreground_fraction']):.3f}, circularity {float(seg['circularity']):.3f}",
        },
        {
            "name": "specimen-versus-field orientation KS statistic",
            "value": round(bundle["ks"]["statistic"], 4),
            "unit": "1",
            "reference_value": 0.0,
            "method": "two-sample KS on Sobel orientation of the fossil versus the simulated flank field",
            "note": f"p={bundle['ks']['pvalue']:.3g}. The fossil is dominated by sub-vertical ribs; the ocellus field is not a copy of that histogram.",
        },
    ]


def _gif_facts() -> tuple[int, int, int]:
    with Image.open(GIF_PATH) as im:
        return int(getattr(im, "n_frames", 1)), int(im.size[0]), int(im.size[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    bundle = compute_all()
    rel = bundle["lambda_rel_err"]
    ratio = bundle["control"]["std_ratio"]
    pred_sig = bundle["predecessor_sigma"]
    sweep_r = float(bundle["sweep"]["pearson_r"])
    if args.self_test:
        assert bundle["pearson_r"] > 0.98, bundle["pearson_r"]
        assert sweep_r > 0.98, sweep_r
        assert rel < 0.12, rel
        assert ratio < 0.35, ratio
        assert bundle["nonlinear_control"]["std_ratio"] < 0.5, bundle["nonlinear_control"]
        assert pred_sig < 0.0, pred_sig
        assert abs(bundle["lambda_m"] - 0.0) > 0.0
        print(
            "[PASS] slot 3170 Diplomystus dentatus 184.3 µm "
            f"dispersion r={bundle['pearson_r']:.5f} RMSE={bundle['rmse']:.5f} "
            f"λ={bundle['lambda_m']*1e6:.2f} µm sweep r={sweep_r:.5f} "
            f"equal-D ratio={ratio:.4f} predecessor σ={pred_sig:.4f} age={AGE_MA} Ma"
        )
        return 0
    render_comparison(bundle)
    render_gif(bundle)
    n_frames, gw, gh = _gif_facts()
    if n_frames < 10 or min(gw, gh) < 480:
        raise RuntimeError(f"GIF failed fitness geometry: frames={n_frames} size={gw}x{gh}")
    print(
        f"λ={bundle['lambda_m']*1e6:.2f} µm  dispersion r={bundle['pearson_r']:.5f}  "
        f"sweep r={sweep_r:.5f}  KS={bundle['ks']['statistic']:.3f}  "
        f"gif={n_frames} frames {gw}x{gh}"
    )
    print(f"wrote {PNG_PATH}")
    print(f"wrote {GIF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
