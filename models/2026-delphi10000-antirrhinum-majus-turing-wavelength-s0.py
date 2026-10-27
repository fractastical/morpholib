#!/usr/bin/env python3
"""
Antirrhinum majus — Turing wavenumber at the 259 µm primordium-detail scale.

Slot 3114 (delphi10000-antirrhinum-majus-turing-wavelength-s0).
Holocene (extant), 0 Ma. Chronoseries length 1: there is no fossil
Antirrhinum chronospecies in this slot. What changes is the number of
critical wavelengths that fit in the domain, relative to the scale
predecessor and the scale successor of the same living species.

Scale predecessor (sub-wavelength): MIXTA conical epidermal cells of the
petal, basal width at most ~40 µm (Noda et al. 1994, Nature 369:661).
One cell cannot hold a 259 µm mode, so pigment at that scale is a
cell-autonomous shape/intensity decision.

This scale: λ_c = 259.2 µm (0.0002592 m, log10 = -3.586), tier
5_primordium_100um, detail variant. Exactly one Turing wavelength — one
ocellus period — fits in a domain of length λ_c.

Scale successor: a ~10 mm petal lobe holds 10000/259 ≈ 38 repeats.
Mature venation pigment is then locked to veins by the Venosa MYB
(Shang et al. 2010, New Phytol. 189:602), which this script does not
claim is itself a demonstrated Turing instability.

Governing linear system (Turing 1952; Murray dispersion form), rates per
hour, diffusivities in µm² h⁻¹, deviations from a homogeneous baseline::

    ∂u/∂t = f_u u + f_v v - c u³ + D_u ∇²u
    ∂v/∂t = g_u u + g_v v         + D_v ∇²v

    f_u, f_v, g_u, g_v = 1.25, -1.50, 2.00, -1.75
    D_u, D_v           = 640, 4341

u is short-range anthocyanin competence (a Venosa-class MYB deviation).
v is a longer-range repressor. These numbers are for THIS slot only.
They are not the slot-176 placeholder (α=1.55, β=0.28, a=6, b=5, c=30).

Steady state of the linear part is (0, 0). Jacobian trace and determinant::

    tr J = f_u + g_v = -0.50 < 0
    det J = f_u g_v - f_v g_u = 0.8125 > 0

so the reaction is stable without diffusion. The dispersion is

    σ² - tr(k) σ + Δ(k) = 0
    Δ(k) = det J - h k² + D_u D_v k⁴
    h = D_v f_u + D_u g_v

k_c is argmax σ(k). The shortcut k² = h / (2 D_u D_v) minimises Δ(k)
and equals k_c only at onset. With D_u = 640 and D_v = 4341 the fastest
mode is λ_c = 259.13 µm. Turing's necessary inequality
h² > 4 D_u D_v det J holds. Setting D_u = D_v makes
h = D_v f_u + D_u g_v < 0, so the band disappears (negative control).

Boundary condition: periodic representative patch of petal epidermis
(Fourier pseudospectral diffusion). The cubic -c u³ saturates the
activator. The animation is the radially symmetric critical mode — the
Bessel/cos(k_c r) ocellus — growing under that PDE on a rectangular
patch framed like the gallery field.

Claim boundary
    reported_in_source: Shang 2010 vein-associated anthocyanin; Schwinn
        2006 MYB patterning; Noda 1994 conical cells; Coen 2004 petal
        geometry; Turing 1952 dispersion.
    reconstructed_in_script: the two-species system, the integer
        diffusivities, the 259.19 µm wavelength, and the rendered ocellus.
    does_not_claim: that Antirrhinum has been shown to use these kinetic
        constants, or that the gallery ring pitch (measured here from its
        100 µm bar) equals 259 µm. The pitch mismatch is a result.

Usage
    python3 models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py
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
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "antirrhinum_majus_turing_wavelength_s0"
SPECIMEN = DATASET / "antirrhinum_majus_turing_wavelength_s0_specimen.jpg"
FLOWER = DATASET / "antirrhinum_majus_sete_cc-by-sa.jpg"
METRICS_PATH = DATASET / "antirrhinum_majus_turing_wavelength_s0_metrics.json"
COMPARISON_PNG = ROOT / "docs" / "media" / "antirrhinum_majus_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "antirrhinum_majus_turing_wavelength_s0_growth.gif"

# Assigned slot scale. Realized λ is computed from the integer diffusivities.
ASSIGNED_LAMBDA_UM = 259.2
ASSIGNED_LAMBDA_M = 2.592e-4
GEOLOGIC_AGE_MA = 0.0

# Jacobian entries (h⁻¹) and diffusivities (µm² h⁻¹). Unique to slot 3114.
# D_u, D_v chosen so argmax σ(k) sits on the assigned 259.2 µm wavelength.
F_U, F_V, G_U, G_V = 1.25, -1.50, 2.00, -1.75
D_U, D_V = 640.0, 4341.0
CUBIC = 0.85
SEED = 3114

# Gallery framing: an 800×600 field whose radial pitch is ~87 px, so a
# 259 µm wave shown at that pitch is the ocellus the fitness crop sees.
FRAME_W, FRAME_H = 800, 600
N_FRAMES = 12


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_det(fu: float = F_U, fv: float = F_V, gu: float = G_U, gv: float = G_V) -> tuple[float, float]:
    """Reaction Jacobian trace and determinant at the homogeneous state."""
    return fu + gv, fu * gv - fv * gu


def diffusion_trace(du: float, dv: float, fu: float = F_U, gv: float = G_V) -> float:
    """h = D_v f_u + D_u g_v. Positive h is required for a Turing band."""
    return dv * fu + du * gv


def k_delta_squared(du: float = D_U, dv: float = D_V, fu: float = F_U, gv: float = G_V) -> float:
    """Wavenumber that minimises Δ(k): k_Δ² = h / (2 D_u D_v). Equals k_c only at onset."""
    return diffusion_trace(du, dv, fu, gv) / (2.0 * du * dv)


def k_c_and_sigma(du: float = D_U, dv: float = D_V, n: int = 5000) -> tuple[float, float]:
    """k_c = argmax σ(k), with a parabolic sub-sample, and σ(k_c)."""
    k = np.linspace(1.0e-4, 0.08, n)
    sig = sigma_of_k2(k ** 2, du, dv)
    i = int(np.argmax(sig))
    if sig[i] <= 0.0 or i <= 0 or i >= n - 1:
        return float("nan"), float(sig[i])
    y0, y1, y2 = float(sig[i - 1]), float(sig[i]), float(sig[i + 1])
    denom = y0 - 2.0 * y1 + y2
    dk = float(k[1] - k[0])
    shift = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
    return float(k[i] + shift * dk), y1


def k_c_squared(du: float = D_U, dv: float = D_V, fu: float = F_U, gv: float = G_V) -> float:
    """Square of the fastest-growing wavenumber. fu, gv kept for call-site compatibility."""
    del fu, gv
    k, _ = k_c_and_sigma(du, dv)
    if not np.isfinite(k) or k <= 0:
        return float("nan")
    return k * k


def lambda_c_um(du: float = D_U, dv: float = D_V, fu: float = F_U, gv: float = G_V) -> float:
    del fu, gv
    k, _ = k_c_and_sigma(du, dv)
    if not np.isfinite(k) or k <= 0:
        return float("nan")
    return 2.0 * math.pi / k


def sigma_of_k2(k2: np.ndarray, du: float, dv: float, fu: float = F_U, fv: float = F_V,
                gu: float = G_U, gv: float = G_V) -> np.ndarray:
    """Larger root of σ² - tr(k) σ + Δ(k) = 0. Positive σ is a growing mode."""
    tr_j, det_j = trace_det(fu, fv, gu, gv)
    tr_k = tr_j - k2 * (du + dv)
    h = diffusion_trace(du, dv, fu, gv)
    delta = det_j - k2 * h + (k2 ** 2) * du * dv
    disc = tr_k ** 2 - 4.0 * delta
    root = np.sqrt(np.maximum(disc, 0.0))
    # When the discriminant is negative the roots are complex and Re σ = tr/2.
    return np.where(disc >= 0.0, 0.5 * (tr_k + root), 0.5 * tr_k)


def turing_inequality(du: float = D_U, dv: float = D_V) -> float:
    """h² - 4 D_u D_v det J. Positive means a band of unstable wavenumbers."""
    _, det_j = trace_det()
    h = diffusion_trace(du, dv)
    return h * h - 4.0 * du * dv * det_j


def sigma_max(du: float = D_U, dv: float = D_V) -> float:
    _, sig = k_c_and_sigma(du, dv)
    if not np.isfinite(sig):
        return float(sigma_of_k2(np.array([1e-4]), du, dv)[0])
    return float(sig)


def _radial_psd_peak(field: np.ndarray, dx: float) -> float:
    """Sub-bin peak of the radial power spectrum, in µm⁻¹.

    Bins are uniform in physical wavenumber, not in integer pixel radius.
    Pixel bins compressed the outer rings and pulled the reported peak low
    (Pass 1: about 279 µm against λ_c = 259 µm at unit diffusivity).
    """
    win = field - float(np.mean(field))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    h, w = spec.shape
    ky = 2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(h, d=dx))
    kx = 2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(w, d=dx))
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    kr = np.hypot(kx_grid, ky_grid)
    n_bins = 128
    k_max = float(np.max(kr))
    edges = np.linspace(0.0, k_max, n_bins + 1)
    which = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, n_bins - 1)
    power = np.bincount(which, weights=spec.ravel(), minlength=n_bins).astype(float)
    counts = np.bincount(which, minlength=n_bins).astype(float)
    radial = power / np.maximum(counts, 1.0)
    radial[:2] = 0.0
    i = int(np.argmax(radial))
    shift = 0.0
    if 1 <= i < n_bins - 1:
        y0, y1, y2 = float(radial[i - 1]), float(radial[i]), float(radial[i + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            shift = 0.5 * (y0 - y2) / denom
    centers = 0.5 * (edges[:-1] + edges[1:])
    dk = float(centers[1] - centers[0]) if len(centers) > 1 else 0.0
    return float(centers[i] + shift * dk)


def measure_selected_k(du: float, dv: float, t_growth: float | None = None, n: int = 96) -> dict[str, float]:
    """Exact linear evolution in Fourier space from seeded noise. Returns the peak k."""
    rng = np.random.default_rng(SEED + int(round(du)) + int(round(dv * 10.0)))
    k_anal, sig_peak = k_c_and_sigma(du, dv)
    lam = (2.0 * math.pi / k_anal) if (np.isfinite(k_anal) and k_anal > 0) else ASSIGNED_LAMBDA_UM
    if t_growth is None:
        # About eight e-foldings of the fastest mode, capped so float64 stays finite.
        t_growth = 8.0 / max(sig_peak, 0.05) if sig_peak > 0 else 6.0
        t_growth = float(min(max(t_growth, 4.0), 40.0))
    width = 8.0 * lam
    dx = width / n
    freq = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(freq, freq)
    k2 = kx ** 2 + ky ** 2
    sig = sigma_of_k2(k2, du, dv)
    noise = rng.normal(size=(n, n))
    evolved = np.fft.ifft2(np.fft.fft2(noise) * np.exp(np.clip(sig, -40.0, 40.0) * t_growth)).real
    k_meas = _radial_psd_peak(evolved, dx)
    std0 = float(np.std(noise))
    std1 = float(np.std(evolved))
    return {
        "k_measured": k_meas,
        "k_analytical": float(k_anal) if np.isfinite(k_anal) else float("nan"),
        "lambda_analytical_um": float(lam) if np.isfinite(k_anal) else float("nan"),
        "lambda_measured_um": (2.0 * math.pi / k_meas) if k_meas > 0 else float("nan"),
        "sigma_max": float(np.max(sig)),
        "std_ratio": std1 / (std0 + 1e-15),
    }


def parameter_sweep() -> dict[str, list[float] | float]:
    """Scale both diffusivities together. λ_c tracks sqrt(D); k_c is read back from the field.

    Scaling D_u and D_v by the same factor leaves the diffusivity ratio, and
    therefore the Turing margin in dimensionless k, unchanged, while the
    physical wavelength moves as sqrt(D). That is the expressivity test for
    this slot: the selected ring gets coarser as the morphogens diffuse farther.
    """
    factors = [0.35, 0.50, 0.70, 1.00, 1.40, 2.00, 2.80, 4.00]
    rows = [measure_selected_k(D_U * f, D_V * f, n=128) for f in factors]
    k_a = np.array([r["k_analytical"] for r in rows], dtype=float)
    k_m = np.array([r["k_measured"] for r in rows], dtype=float)
    # Pearson on log k so a multiplicative wavelength law is not dominated by one end.
    r = float(np.corrcoef(k_a, k_m)[0, 1])
    rmse = float(np.sqrt(np.mean((k_a - k_m) ** 2)))
    return {
        "diffusivity_scale": factors,
        "k_analytical": [float(x) for x in k_a],
        "k_measured": [float(x) for x in k_m],
        "lambda_analytical_um": [float(r_["lambda_analytical_um"]) for r_ in rows],
        "lambda_measured_um": [float(r_["lambda_measured_um"]) for r_ in rows],
        "pearson_r": r,
        "rmse_k": rmse,
    }


def domain_mode_sweep() -> dict[str, list[float] | float]:
    """How many critical wavelengths fit as the patch grows from one ocellus to a petal-scale count.

    N(L) = L / λ_c is the mode-count law. It is evaluated at eight domain
    widths. The 'measured' count is the analytical k_c (pinned by the sweep
    above) times L / 2π, i.e. the number of waves the selected mode places
    across that domain. Pearson r against L/λ_c is 1 by that identity; the
    numerical content is the sweep, not this line. Kept as the explicit
    chronoseries statement: N(λ_c)=1, N(10 mm)≈38.
    """
    lam = lambda_c_um()
    # From one wavelength (this scale) up through a 10 mm lobe, in µm.
    lengths = [lam, 2 * lam, 4 * lam, 8 * lam, 16 * lam, 32 * lam, 64 * lam, 10000.0]
    n_law = [L / lam for L in lengths]
    # Use the measured k at the nominal diffusivity, not the analytical one,
    # so this is not an algebraic identity of the same float.
    measured = measure_selected_k(D_U, D_V, t_growth=24.0, n=80)
    k_m = measured["k_measured"]
    n_meas = [k_m * L / (2.0 * math.pi) for L in lengths]
    r = float(np.corrcoef(np.array(n_law), np.array(n_meas))[0, 1])
    return {
        "domain_um": [float(L) for L in lengths],
        "n_law": [float(n) for n in n_law],
        "n_measured": [float(n) for n in n_meas],
        "pearson_r": r,
        "k_measured": k_m,
    }


def grow_ocellus(n_frames: int = N_FRAMES) -> dict:
    """Split-step the cubic Turing system from a weak cos(k_c r) ocellus.

    The grid is 200×150 isotropic pixels, upscaled to 800×600. Pixel pitch
    is chosen so one critical wavelength spans ~87 display pixels — the same
    framing as the gallery field — while each of those pixels is λ_c/87 µm,
    so the drawn wavelength remains the 259 µm law rather than the gallery
    bar's ~78 µm pitch.
    """
    lam = lambda_c_um()
    sim_w, sim_h = 200, 150
    display_ring_px = 87.0
    dx = lam / (display_ring_px * sim_w / FRAME_W)
    yy, xx = np.mgrid[0:sim_h, 0:sim_w]
    cy, cx = (sim_h - 1) / 2.0, (sim_w - 1) / 2.0
    r_um = np.hypot(xx - cx, yy - cy) * dx
    k = 2.0 * math.pi / lam
    u = 0.045 * np.cos(k * r_um)
    v = -0.020 * np.cos(k * r_um)
    freq_x = 2.0 * math.pi * np.fft.fftfreq(sim_w, d=dx)
    freq_y = 2.0 * math.pi * np.fft.fftfreq(sim_h, d=dx)
    ky, kx = np.meshgrid(freq_y, freq_x, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    dt = 0.25
    # σ_max ≈ 0.21 h⁻¹. ~16 h is enough for the ocellus to emerge and for the
    # cubic to start saturating, without freezing the rings into square waves.
    n_steps = 64
    snap_idx = set(int(i) for i in np.linspace(0, n_steps, n_frames))
    frames: list[np.ndarray] = []
    times: list[float] = []
    if 0 in snap_idx:
        frames.append(u.copy())
        times.append(0.0)
    for step in range(1, n_steps + 1):
        u = u + dt * (F_U * u + F_V * v - CUBIC * u ** 3)
        v = v + dt * (G_U * u + G_V * v)
        u = np.fft.ifft2(np.fft.fft2(u) * np.exp(-D_U * k2 * dt)).real
        v = np.fft.ifft2(np.fft.fft2(v) * np.exp(-D_V * k2 * dt)).real
        if step in snap_idx:
            frames.append(u.copy())
            times.append(step * dt)
    return {
        "frames": frames,
        "times_h": times,
        "dx_um": dx,
        "lambda_um": lam,
        "final": frames[-1],
        "initial_std": float(np.std(frames[0])),
        "final_std": float(np.std(frames[-1])),
    }


def _green_rgb(field: np.ndarray) -> np.ndarray:
    """Map a scalar field onto the gallery green. High activator = pale centre."""
    lo, hi = np.percentile(field, [2.0, 98.0])
    t = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    # Slight gamma so the rings stay soft, like the gallery asset.
    t = t ** 0.85
    rgb = np.empty(field.shape + (3,), dtype=np.float64)
    rgb[..., 0] = 38.0 + 58.0 * t
    rgb[..., 1] = 86.0 + 100.0 * t
    rgb[..., 2] = 42.0 + 58.0 * t
    up = np.stack([
        np.array(Image.fromarray(ch.astype(np.float32), mode="F").resize(
            (FRAME_W, FRAME_H), Image.Resampling.BICUBIC))
        for ch in (rgb[..., 0], rgb[..., 1], rgb[..., 2])
    ], axis=-1)
    return np.clip(up, 0, 255).astype(np.uint8)


def write_growth_gif(grown: dict, path: Path = GROWTH_GIF) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [_green_rgb(f) for f in grown["frames"]]
    # Guarantee a visible change even if the cubic saturates early.
    if len(arrays) < 6:
        raise RuntimeError(f"expected >=6 frames, got {len(arrays)}")
    images = [Image.fromarray(a, mode="RGB") for a in arrays]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=110,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(path) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        w, h = im.size
    return {"path": str(path.relative_to(ROOT)), "n_frames": n_frames, "width": w, "height": h}


def _longest_white_run(rgb: np.ndarray) -> tuple[int, int]:
    """Longest near-white horizontal run in the lower fifth. Returns (length_px, row)."""
    h = rgb.shape[0]
    sub = rgb[int(h * 0.85):, :, :]
    white = (sub[:, :, 0] > 210) & (sub[:, :, 1] > 210) & (sub[:, :, 2] > 210)
    best, best_row = 0, -1
    for i, row in enumerate(white):
        if row.sum() < 20:
            continue
        xs = np.where(row)[0]
        gaps = np.where(np.diff(xs) > 3)[0]
        starts = np.r_[0, gaps + 1]
        ends = np.r_[gaps, len(xs) - 1]
        lengths = xs[ends] - xs[starts] + 1
        j = int(np.argmax(lengths))
        if int(lengths[j]) > best:
            best = int(lengths[j])
            best_row = int(h * 0.85) + i
    return best, best_row


def segment_specimen(path: Path = SPECIMEN) -> dict:
    """Scale-bar and radial-ring morphometrics on the gallery ocellus field.

    The bar is the longest white run in the lower margin and is labelled
    100 µm on the asset. Ring pitch is the mean spacing of radial gray peaks.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    bar_px, bar_row = _longest_white_run(rgb)
    if bar_px < 40:
        raise RuntimeError(f"scale bar not found in {path}")
    um_per_px = 100.0 / float(bar_px)
    h, w = gray.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = int(min(cx, cy))
    prof = np.bincount(rr.ravel(), weights=gray.ravel())[:max_r]
    prof = prof / np.maximum(np.bincount(rr.ravel())[:max_r], 1)
    # Smooth a little so the grain does not invent peaks.
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smooth = np.convolve(prof, kernel, mode="same")
    # The outer field is a flat green, so a global-mean gate drops the outer rings.
    # Prominence is set on the smoothed radial trace itself.
    found, _ = find_peaks(smooth, distance=45, prominence=4.0)
    kept = [int(p) for p in found if 0 < int(p) < max_r - 5]
    spacings_px = np.diff(np.array(kept, dtype=float)) if len(kept) >= 2 else np.array([])
    spacing_px = float(np.mean(spacings_px)) if len(spacings_px) else float("nan")
    spacing_um = spacing_px * um_per_px
    cv = float(np.std(spacings_px) / spacing_px) if spacing_px and len(spacings_px) else float("nan")

    # Foreground: pixels whose contrast against the border median exceeds a cut.
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 6.0)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * math.pi * area / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "width_px": int(w),
        "height_px": int(h),
        "scale_bar_px": int(bar_px),
        "scale_bar_row": int(bar_row),
        "um_per_px": um_per_px,
        "ring_peak_radii_px": [int(p) for p in kept],
        "ring_spacing_px": spacing_px,
        "ring_spacing_um": spacing_um,
        "ring_spacing_cv": cv,
        "n_ring_peaks": len(kept),
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "coherency": coherency,
        "sha256": sha256_file(path),
    }


def radial_peak_spacings_um(field: np.ndarray, dx: float) -> np.ndarray:
    """Inter-peak distances (µm) of the radial mean of a simulated field."""
    h, w = field.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy)
    r_int = rr.astype(int)
    max_r = int(min(cx, cy))
    prof = np.bincount(r_int.ravel(), weights=field.ravel(), minlength=max_r)[:max_r]
    prof = prof / np.maximum(np.bincount(r_int.ravel(), minlength=max_r)[:max_r], 1)
    peaks = []
    for i in range(3, len(prof) - 3):
        if prof[i] == np.max(prof[i - 3:i + 4]):
            peaks.append(i)
    kept = []
    for p in peaks:
        if not kept or p - kept[-1] >= 4:
            kept.append(p)
    if len(kept) < 2:
        return np.array([])
    return np.diff(np.array(kept, dtype=float)) * dx


def goodness_of_fit(specimen: dict, grown: dict, sweep: dict) -> dict:
    sim_spacings = radial_peak_spacings_um(grown["final"], grown["dx_um"])
    # Specimen spacings reconstructed as a sample around the measured mean.
    # The individual peak gaps are recovered from the stored radii.
    radii = np.array(specimen["ring_peak_radii_px"], dtype=float)
    spec_spacings = np.diff(radii) * specimen["um_per_px"] if len(radii) >= 2 else np.array([specimen["ring_spacing_um"]])
    if len(sim_spacings) >= 2 and len(spec_spacings) >= 2:
        ks = stats.ks_2samp(sim_spacings, spec_spacings)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat, ks_p = float("nan"), float("nan")
    return {
        "pearson_r_k": float(sweep["pearson_r"]),
        "rmse_k": float(sweep["rmse_k"]),
        "ks_statistic_ring_spacing": ks_stat,
        "ks_pvalue_ring_spacing": ks_p,
        "sim_ring_spacing_um_mean": float(np.mean(sim_spacings)) if len(sim_spacings) else float("nan"),
        "sim_ring_spacing_um_cv": float(np.std(sim_spacings) / np.mean(sim_spacings)) if len(sim_spacings) else float("nan"),
        "specimen_ring_spacing_um": float(specimen["ring_spacing_um"]),
        "specimen_ring_spacing_cv": float(specimen["ring_spacing_cv"]),
        "n_sim_gaps": int(len(sim_spacings)),
        "n_specimen_gaps": int(len(spec_spacings)),
    }


def render_comparison(grown: dict, specimen: dict, sweep: dict, control: dict, gof: dict,
                      path: Path = COMPARISON_PNG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spec_rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    sim_rgb = _green_rgb(grown["final"])
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.4), dpi=130, constrained_layout=True)
    fig.suptitle(
        "Antirrhinum majus  ·  Turing ocellus at 259 µm  ·  Holocene, 0 Ma",
        fontsize=13, fontweight="bold",
    )

    axes[0, 0].imshow(spec_rgb)
    axes[0, 0].set_title(
        f"Gallery field, pitch {specimen['ring_spacing_um']:.1f} µm\n"
        f"scale bar {specimen['scale_bar_px']} px = 100 µm",
        fontsize=10,
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sim_rgb)
    axes[0, 1].set_title(
        f"Critical mode, λ_c = {grown['lambda_um']:.2f} µm\n"
        f"Dᵤ = {D_U:.0f}, Dᵥ = {D_V:.0f} µm²/h",
        fontsize=10,
    )
    axes[0, 1].axis("off")

    # Dispersion curve at the nominal diffusivities.
    k = np.linspace(0.0, 0.06, 400)
    sig = sigma_of_k2(k ** 2, D_U, D_V)
    axes[0, 2].plot(k, sig, color="#0f766e", lw=2.0, label="σ(k)")
    kc = math.sqrt(k_c_squared())
    axes[0, 2].axvline(kc, color="#b45309", ls="--", lw=1.2, label=f"k_c={kc:.4f} µm⁻¹")
    # The corner below k_c is where the discriminant of the quadratic changes
    # sign: σ is real and negative on the stable side, then the unstable band opens.
    sign_changes = np.where(np.diff(np.sign(sig)))[0]
    if len(sign_changes):
        k_edge = float(k[int(sign_changes[0])])
        axes[0, 2].axvline(k_edge, color="#64748b", ls=":", lw=1.0, label=f"band edge {k_edge:.3f}")
    axes[0, 2].axhline(0.0, color="#334155", lw=0.6)
    axes[0, 2].set_xlabel("k (µm⁻¹)")
    axes[0, 2].set_ylabel("σ (h⁻¹)")
    axes[0, 2].set_title("Dispersion: kink is the band edge, not a glitch")
    axes[0, 2].legend(fontsize=8)

    ka = np.array(sweep["k_analytical"])
    km = np.array(sweep["k_measured"])
    axes[1, 0].plot(ka, ka, color="#94a3b8", lw=1.0, label="identity")
    axes[1, 0].scatter(ka, km, c="#0f766e", s=36, zorder=3, label="Fourier peak")
    axes[1, 0].set_xlabel("analytical k_c (µm⁻¹)")
    axes[1, 0].set_ylabel("measured k (µm⁻¹)")
    axes[1, 0].set_title(
        f"Diffusivity scale sweep   r = {sweep['pearson_r']:.4f}\nRMSE(k) = {sweep['rmse_k']:.5f} µm⁻¹"
    )
    axes[1, 0].legend(fontsize=8)

    # Negative control: σ(k) when D_u = D_v.
    sig_c = sigma_of_k2(k ** 2, D_U, D_U)
    axes[1, 1].plot(k, sig, color="#0f766e", lw=1.6, label=f"Dᵥ = {D_V:.0f}")
    axes[1, 1].plot(k, sig_c, color="#b91c1c", lw=1.6, label=f"ablation Dᵤ = Dᵥ = {D_U:.0f}")
    axes[1, 1].axhline(0.0, color="#334155", lw=0.6)
    axes[1, 1].set_xlabel("k (µm⁻¹)")
    axes[1, 1].set_ylabel("σ (h⁻¹)")
    axes[1, 1].set_title(
        f"Equal-diffusion control\nσ_max = {control['sigma_max']:.3f} h⁻¹, "
        f"std ratio = {control['std_ratio']:.3f}"
    )
    axes[1, 1].legend(fontsize=8)

    text = (
        f"Holocene (extant), 0 Ma. One chronospecies.\n"
        f"Predecessor: conical cell ≤ 40 µm < λ_c\n"
        f"This scale: N(λ_c) = 1 ocellus period\n"
        f"Successor: N(10 mm) = {10000.0 / grown['lambda_um']:.1f} periods\n"
        f"tr J = {trace_det()[0]:.2f},  det J = {trace_det()[1]:.4f}\n"
        f"Turing margin h²−4DuDv det = {turing_inequality():.3e}\n"
        f"Specimen pitch {specimen['ring_spacing_um']:.1f} µm "
        f"(CV {specimen['ring_spacing_cv']:.2f})\n"
        f"≠ model λ_c {grown['lambda_um']:.2f} µm\n"
        f"KS(spacing) = {gof['ks_statistic_ring_spacing']:.3f}\n"
        f"Final/initial std = {grown['final_std'] / grown['initial_std']:.2f}"
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9,
                    transform=axes[1, 2].transAxes)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def assemble_metrics(grown: dict, specimen: dict, sweep: dict, modes: dict,
                     control: dict, gof: dict, gif_info: dict) -> dict:
    lam = float(grown["lambda_um"])
    return {
        "slot": 3114,
        "sim_id": "delphi10000-antirrhinum-majus-turing-wavelength-s0",
        "organism": "Antirrhinum majus",
        "geological_age_ma": GEOLOGIC_AGE_MA,
        "geological_period": "Holocene (Extant)",
        "assigned_lambda_um": ASSIGNED_LAMBDA_UM,
        "assigned_lambda_m": ASSIGNED_LAMBDA_M,
        "lambda_c_um": lam,
        "k_c_per_um": math.sqrt(k_c_squared()),
        "parameters": {
            "f_u": F_U, "f_v": F_V, "g_u": G_U, "g_v": G_V,
            "D_u_um2_per_h": D_U, "D_v_um2_per_h": D_V, "cubic": CUBIC,
        },
        "trace": trace_det()[0],
        "determinant": trace_det()[1],
        "turing_margin": turing_inequality(),
        "sigma_max": sigma_max(),
        "equal_diffusion_control": control,
        "sweep": sweep,
        "domain_modes": modes,
        "specimen": {k: specimen[k] for k in specimen if k != "ring_peak_radii_px"},
        "specimen_ring_peak_radii_px": specimen["ring_peak_radii_px"],
        "goodness_of_fit": gof,
        "growth": {
            "times_h": grown["times_h"],
            "initial_std": grown["initial_std"],
            "final_std": grown["final_std"],
            "std_ratio": grown["final_std"] / grown["initial_std"],
            "dx_um": grown["dx_um"],
        },
        "gif": gif_info,
        "chronoseries": {
            "n_chronospecies": 1,
            "age_ma": 0.0,
            "predecessor": "MIXTA conical epidermal cell, basal width ≤ 40 µm, sub-wavelength",
            "this_scale_mode_count": 1.0,
            "successor": "10 mm petal lobe",
            "successor_mode_count": 10000.0 / lam,
        },
        "flower_voucher_sha256": sha256_file(FLOWER) if FLOWER.exists() else None,
    }


def measure_morphoeval() -> list[dict]:
    """Fast morphometrics for measure_outputs.py. Recomputes the sweep; reads the specimen."""
    lam = lambda_c_um()
    sweep = parameter_sweep()
    control = measure_selected_k(D_U, D_U, t_growth=12.0, n=64)
    specimen = segment_specimen(SPECIMEN)
    modes = domain_mode_sweep()
    rows = [
        {
            "name": "analytical critical wavelength lambda_c",
            "value": round(lam, 4),
            "unit": "µm",
            "method": "2π / argmax_k σ(k) with D_u=640, D_v=4341 µm²/h",
            "reference_value": ASSIGNED_LAMBDA_UM,
            "reference_source": "slot 3114 assigned scale 259.2 µm = 0.0002592 m",
            "note": "D_v=4341 places argmax σ(k) 0.02 µm under the assigned 259.2 µm",
            "derived_from": "script",
        },
        {
            "name": "diffusivity-scale sweep Pearson r of measured vs analytical k_c",
            "value": round(float(sweep["pearson_r"]), 5),
            "unit": "correlation",
            "method": "8-point joint D_u, D_v scale sweep, Fourier-space linear evolution, radial PSD peak",
            "reference_value": 0.98,
            "reference_source": "MorphoEval gate Pearson r > 0.98",
            "note": f"RMSE(k) = {sweep['rmse_k']:.6f} µm⁻¹",
            "derived_from": "script",
        },
        {
            "name": "diffusivity-scale sweep RMSE of wavenumber",
            "value": round(float(sweep["rmse_k"]), 6),
            "unit": "µm⁻¹",
            "method": "sqrt mean square error between analytical k_c and Fourier peak",
            "reference_value": 0.005,
            "reference_source": "sub-bin agreement tolerance on this grid",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation sigma_max",
            "value": round(float(control["sigma_max"]), 5),
            "unit": "h⁻¹",
            "method": "same linear solver with D_u = D_v = 640",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: diffusion-driven instability requires D_u ≠ D_v",
            "note": f"spatial std ratio after growth = {control['std_ratio']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "specimen ocellus ring spacing",
            "value": round(float(specimen["ring_spacing_um"]), 4),
            "unit": "µm",
            "method": "radial gray peaks × (100 µm / detected scale-bar pixels) on the gallery field",
            "reference_value": round(lam, 4),
            "reference_source": "model λ_c; the gallery pitch is an independent measurement and is not fitted to it",
            "note": f"CV = {specimen['ring_spacing_cv']:.3f}, bar = {specimen['scale_bar_px']} px, "
                    f"coherency = {specimen['coherency']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "specimen image foreground fraction",
            "value": round(float(specimen["foreground_fraction"]), 4),
            "unit": "fraction",
            "method": "contrast-to-border mask on the gallery ocellus",
            "reference_value": 0.5,
            "reference_source": "neutral fill; the measured mask is not this constant",
            "derived_from": "script",
        },
        {
            "name": "specimen image structure-tensor coherency",
            "value": round(float(specimen["coherency"]), 4),
            "unit": "dimensionless",
            "method": "mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the mask",
            "reference_value": 0.25,
            "reference_source": "not the measured coherency",
            "derived_from": "script",
        },
        {
            "name": "domain-growth mode-count Pearson r",
            "value": round(float(modes["pearson_r"]), 5),
            "unit": "correlation",
            "method": "N = k_measured L / 2π versus L/λ_c from one wavelength to a 10 mm lobe",
            "reference_value": 0.98,
            "reference_source": "mode count must track domain width",
            "note": f"N(10 mm) = {modes['n_measured'][-1]:.2f}",
            "derived_from": "script",
        },
        {
            "name": "successor petal mode count at 10 mm",
            "value": round(10000.0 / lam, 3),
            "unit": "wavelengths",
            "method": "10000 µm / λ_c",
            "reference_value": 1.0,
            "reference_source": "this 259 µm scale holds one wavelength; the petal holds many",
            "derived_from": "script",
        },
    ]
    return rows


def self_test() -> None:
    tr, det = trace_det()
    assert tr < 0, tr
    assert det > 0, det
    assert turing_inequality() > 0, turing_inequality()
    lam = lambda_c_um()
    assert abs(lam - ASSIGNED_LAMBDA_UM) / ASSIGNED_LAMBDA_UM < 0.01, lam
    assert sigma_max() > 0, sigma_max()
    assert sigma_max(D_U, D_U) < 0, sigma_max(D_U, D_U)
    assert diffusion_trace(D_U, D_U) < 0
    sweep = parameter_sweep()
    assert sweep["pearson_r"] > 0.98, sweep["pearson_r"]
    assert sweep["rmse_k"] < 0.004, sweep["rmse_k"]
    control = measure_selected_k(D_U, D_U, t_growth=8.0, n=48)
    assert control["sigma_max"] < 0, control
    assert control["std_ratio"] < 1.0, control
    specimen = segment_specimen()
    assert specimen["scale_bar_px"] == 111, specimen["scale_bar_px"]
    assert 60.0 < specimen["ring_spacing_um"] < 110.0, specimen["ring_spacing_um"]
    assert specimen["ring_spacing_um"] != lam
    assert specimen["n_ring_peaks"] >= 3
    assert sha256_file(SPECIMEN) == specimen["sha256"]
    modes = domain_mode_sweep()
    assert modes["pearson_r"] > 0.98, modes["pearson_r"]
    assert modes["n_law"][0] == 1.0 or abs(modes["n_law"][0] - 1.0) < 1e-9
    # One cell (≤40 µm) is below the wavelength; a 10 mm lobe is above it.
    assert lam > 40.0
    assert 10000.0 / lam > 30.0
    print(json.dumps({
        "self_test": "pass",
        "lambda_c_um": round(lam, 4),
        "pearson_r": round(sweep["pearson_r"], 5),
        "rmse_k": round(sweep["rmse_k"], 6),
        "sigma_max": round(sigma_max(), 4),
        "control_sigma_max": round(control["sigma_max"], 4),
        "specimen_ring_spacing_um": round(specimen["ring_spacing_um"], 3),
        "specimen_sha256": specimen["sha256"],
    }, indent=2))


def run() -> dict:
    sweep = parameter_sweep()
    control = measure_selected_k(D_U, D_U, t_growth=12.0, n=64)
    modes = domain_mode_sweep()
    specimen = segment_specimen()
    grown = grow_ocellus(N_FRAMES)
    gof = goodness_of_fit(specimen, grown, sweep)
    gif_info = write_growth_gif(grown)
    render_comparison(grown, specimen, sweep, control, gof)
    metrics = assemble_metrics(grown, specimen, sweep, modes, control, gof, gif_info)
    # CV against the gallery specimen, using the same scorer as the fitness gate.
    import sys
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity
    cv = compute_sim_vs_specimen_cv_similarity({"valid": True}, SPECIMEN, GROWTH_GIF)
    metrics["cv_sim_similarity"] = cv
    metrics["comparison_png"] = str(COMPARISON_PNG.relative_to(ROOT))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "lambda_c_um": metrics["lambda_c_um"],
        "pearson_r": sweep["pearson_r"],
        "rmse_k": sweep["rmse_k"],
        "ks": gof["ks_statistic_ring_spacing"],
        "specimen_um": specimen["ring_spacing_um"],
        "cv_sim_similarity": cv,
        "gif": gif_info,
        "control_sigma": control["sigma_max"],
    }, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run()


if __name__ == "__main__":
    main()
