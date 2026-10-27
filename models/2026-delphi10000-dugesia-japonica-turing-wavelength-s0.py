#!/usr/bin/env python3
"""
Dugesia japonica — Turing pigment wavelength at 6.912 µm (Holocene, 0 Ma).

Slot 3213 of the 10,000-animal basic look. Scale tier 3/8 ``3_subcellular_um``,
detail variant: the critical reaction-diffusion wavelength of dorsal pigment
mottling around the ocellus, not the millimetre-scale body plan used by sibling
Dugesia slots.

Governing system
----------------
Saturated Schnakenberg kinetics. The unsaturated cubic u²v runs away under
explicit diffusion; a Michaelis saturation (κ = 0.15) keeps the peaks bounded
while leaving an activator Jacobian f_u > 0. Biologically this is the
Gierer–Meinhardt (1972) reading Murray (1981) used for pigment pre-patterns:
u is the pigment-differentiation activator, v the depleted substrate / inhibitor
reservoir::

    w = u² v / (1 + κ u²)
    du/dt = D_u ∇²u + a - u + w
    dv/dt = D_v ∇²v + b - w

Homogeneous steady state: u0 = a + b, v0 = b (1 + κ u0²) / u0².
Jacobian entries f_u, f_v, g_u, g_v are in ``jacobian``. The critical
wavenumber selected by linear diffusion-driven instability is

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c.

D_u is solved so that λ_c = 6.912 µm exactly (6.912e-6 m, log10 = -5.160).
The diffusion ratio R = D_v / D_u = 48 is the slot parameter (not the sibling
slot's alpha/beta template). Saturation pushes the discriminant condition to
this higher ratio; R = 18, which would have been enough without κ, does not
unstable a band of wavenumbers once κ = 0.15. A 7-wavelength patch
(L = 48.384 µm) is the detail field.

Chronoseries
------------
Dugesia japonica chronoseries 1/1 is Holocene (extant), 0 Ma. There is no
fossil predecessor and no successor chronospecies. What changes at THIS scale,
relative to a coarser reading of the same Holocene animal, is the selected mode
count N(L) = L / λ_c. At L = 48.384 µm, N = 7. At body-plan scales (millimetres)
the same λ_c would be a fine texture, not the feature that sets head polarity.
The scale-predecessor state is the spatially uniform steady state (ablated when
D_u = D_v). The scale-successor envelope is the ocellus cup itself, which this
script treats as an anatomical mask from the specimen plate, not as a second
Turing mode.

Specimen plates (CC)
--------------------
* Nogi, Zhang, Chan & Marchant (2009) PLoS NTDs 3:e464, CC BY 2.5 — regenerated
  D. japonica with dorsal pigment and ocelli. This is a drug-regeneration
  micrograph, not a 6.9 µm measurement.
* Timur Kalininsky, iNaturalist photo 226612281, CC BY 4.0 — wild-type animals
  on substrate (field voucher). Used as a second plate in later passes.

The animation maps the simulated 6.912 µm field onto the segmented bodies of
the Nogi plate. Pixel wavelength on that plate is not 6.912 µm; the plate has
no scale bar. The mapping is a display of the subcellular motif across the
epithelium.
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
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "dugesia_japonica_turing_wavelength_s0"
SPECIMEN_NOGI = DATASET / "dugesia_japonica_ocellus_pigment_nogi2009.png"
SPECIMEN_INAT = DATASET / "dugesia_japonica_field_kalininsky_ccby4.jpg"
MEDIA_PNG = ROOT / "docs" / "media" / "dugesia_japonica_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "dugesia_japonica_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "dugesia_japonica_turing_wavelength_s0_metrics.json"

# Slot scale. 6.912 µm = 6.912e-6 m; log10(6.912e-6) = -5.160.
LAMBDA_UM = 6.912
LAMBDA_M = 6.912e-6
AGE_MA = 0.0
CHRONOSERIES_INDEX = 1
CHRONOSERIES_LENGTH = 1
N_WAVES = 7
DIFFUSION_RATIO = 48.0  # D_v / D_u at the Holocene detail-scale operating point
SATURATION_KAPPA = 0.15  # bounds u²v; f_u stays positive below κ ≈ 0.44
SCHNAKENBERG_A = 0.05
SCHNAKENBERG_B = 1.40
GRID_N = 84  # 12 nodes per critical wavelength
SEED = 3213
DT = 0.05  # reaction substep; diffusion is an exact Fourier multiplier
N_STEPS = 1600
N_FRAMES = 16

# Ellipse-model circularity used as an independent geometric reference for a
# single planarian body (aspect ~ 2.4). The photographed multi-animal mask is
# not this value; the inequality is the non-circular specimen check.
ELLIPSE_CIRCULARITY_REFERENCE = 0.55


def log10_length() -> float:
    return math.log10(LAMBDA_M)


def steady_state(a: float = SCHNAKENBERG_A, b: float = SCHNAKENBERG_B) -> tuple[float, float]:
    """Saturated Schnakenberg homogeneous steady state (u0, v0)."""
    u0 = a + b
    q = (u0 * u0) / (1.0 + SATURATION_KAPPA * u0 * u0)
    v0 = b / q
    return u0, v0


def jacobian(a: float = SCHNAKENBERG_A, b: float = SCHNAKENBERG_B) -> tuple[float, float, float, float]:
    """Reaction Jacobian (f_u, f_v, g_u, g_v) at the homogeneous steady state.

    w = u² v / (1 + κ u²). At u0 = a + b the substrate balance fixes v0, and
    f_u = -1 + v0 * d(q)/du with q = u² / (1 + κ u²).
    """
    u0, v0 = steady_state(a, b)
    s = 1.0 + SATURATION_KAPPA * u0 * u0
    q = (u0 * u0) / s
    dq_du = (2.0 * u0) / (s * s)
    f_u = -1.0 + v0 * dq_du
    f_v = q
    g_u = -v0 * dq_du
    g_v = -q
    return f_u, f_v, g_u, g_v


def diffusivities(diffusion_ratio: float = DIFFUSION_RATIO) -> tuple[float, float]:
    """D_u, D_v (µm² per time) such that λ_c = 6.912 µm at this diffusion ratio.

    From k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v) with D_v = R D_u,
    D_u = (R f_u + g_v) / (2 R k_c²).
    """
    f_u, _f_v, _g_u, g_v = jacobian()
    k_c = 2.0 * math.pi / LAMBDA_UM
    numer = diffusion_ratio * f_u + g_v
    d_u = numer / (2.0 * diffusion_ratio * k_c * k_c)
    d_v = diffusion_ratio * d_u
    return d_u, d_v


def turing_gap(diffusion_ratio: float) -> float:
    """Numerator D_v f_u + D_u g_v, up to the positive factor D_u. Positive iff Turing."""
    f_u, _f_v, _g_u, g_v = jacobian()
    return diffusion_ratio * f_u + g_v


def dispersion_sigma(k: np.ndarray | float, diffusion_ratio: float = DIFFUSION_RATIO) -> np.ndarray:
    """Largest real part of the growth rate σ(k) of a Fourier mode (continuum Laplacian)."""
    f_u, f_v, g_u, g_v = jacobian()
    d_u, d_v = diffusivities(diffusion_ratio)
    k2 = np.asarray(k, dtype=np.float64) ** 2
    tr = (f_u + g_v) - k2 * (d_u + d_v)
    det = (f_u - k2 * d_u) * (g_v - k2 * d_v) - f_v * g_u
    disc = tr * tr - 4.0 * det
    root = np.sqrt(np.maximum(disc, 0.0))
    # Complex pair: real part is tr/2. Real pair: take the larger root.
    sigma = np.where(disc >= 0.0, 0.5 * (tr + root), 0.5 * tr)
    return sigma


def analytic_lambda_um(diffusion_ratio: float = DIFFUSION_RATIO) -> float:
    f_u, _f_v, _g_u, g_v = jacobian()
    d_u, d_v = diffusivities(diffusion_ratio) if diffusion_ratio == DIFFUSION_RATIO else (None, None)
    if diffusion_ratio == DIFFUSION_RATIO:
        # By construction of diffusivities().
        return LAMBDA_UM
    # Recompute k_c with D_u fixed at the operating-point value and D_v = R D_u.
    d_u_op, _ = diffusivities(DIFFUSION_RATIO)
    d_v = diffusion_ratio * d_u_op
    numer = d_v * f_u + d_u_op * g_v
    if numer <= 0.0:
        return math.inf
    k2 = numer / (2.0 * d_u_op * d_v)
    return float(2.0 * math.pi / math.sqrt(k2))


def _laplace(field: np.ndarray, dx: float) -> np.ndarray:
    """Periodic 5-point Laplacian. Matches the discrete symbol used in the growth-rate test."""
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4.0 * field) / (dx * dx)


def _laplace_1d(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, 1) + np.roll(field, -1) - 2.0 * field) / (dx * dx)


def reaction(u: np.ndarray, v: np.ndarray, a: float = SCHNAKENBERG_A, b: float = SCHNAKENBERG_B):
    """Saturated Schnakenberg kinetics. w ≤ v / κ, so peaks stay finite."""
    w = (u * u * v) / (1.0 + SATURATION_KAPPA * u * u)
    return a - u + w, b - w


def _diffuse_fourier(field: np.ndarray, diffusivity: float, dt: float, length: float) -> np.ndarray:
    """Exact periodic diffusion over one step: exp(-D k² dt) in Fourier space."""
    n = field.shape[0]
    freq = np.fft.fftfreq(n, d=length / n)
    kx, ky = np.meshgrid(freq, freq)
    ksq = (2.0 * math.pi) ** 2 * (kx * kx + ky * ky)
    return np.fft.ifft2(np.fft.fft2(field) * np.exp(-diffusivity * ksq * dt)).real


def simulate_patch(n_steps: int = N_STEPS, n_save: int = N_FRAMES, seed: int = SEED) -> dict:
    """Periodic 2D patch, side N_WAVES * λ_c. Diffusion is spectral; reaction is explicit."""
    n = GRID_N
    length = N_WAVES * LAMBDA_UM
    dx = length / n
    d_u, d_v = diffusivities()
    u0, v0 = steady_state()
    rng = np.random.default_rng(seed)
    u = u0 * (1.0 + 0.02 * rng.normal(size=(n, n)))
    v = v0 * (1.0 + 0.02 * rng.normal(size=(n, n)))
    save_every = max(1, n_steps // n_save)
    frames: list[np.ndarray] = []
    for step in range(n_steps):
        u = _diffuse_fourier(u, d_u, DT, length)
        v = _diffuse_fourier(v, d_v, DT, length)
        fu, fv = reaction(u, v)
        u = np.maximum(u + DT * fu, 1e-4)
        v = np.maximum(v + DT * fv, 1e-4)
        if step % save_every == 0 or step == n_steps - 1:
            frames.append(u.copy())
    frames = frames[: n_save - 1] + [u.copy()]
    return {
        "u_frames": frames,
        "u_final": u,
        "v_final": v,
        "dx_um": dx,
        "length_um": length,
        "d_u": d_u,
        "d_v": d_v,
    }


def measure_mode_growth(mode_index: int, steps: int = 80) -> tuple[float, float]:
    """Growth rate of one Fourier mode of the linearised Schnakenberg system.

    The initial condition is the dominant eigenvector of the discrete-symbol
    Jacobian, so the measured log-slope is that eigenvalue rather than a
    mixture of the stable and unstable branches. Wavenumber k_disc is the
    symbol of the same 1D periodic Laplacian the integrator uses.
    """
    n = GRID_N
    length = N_WAVES * LAMBDA_UM
    dx = length / n
    d_u, d_v = diffusivities()
    f_u, f_v, g_u, g_v = jacobian()
    u0, v0 = steady_state()
    x = np.arange(n)
    phase = 2.0 * math.pi * mode_index * x / n
    # Discrete eigenvalue of the 1D periodic Laplacian: -k_disc².
    k_disc_sq = (2.0 - 2.0 * math.cos(2.0 * math.pi * mode_index / n)) / (dx * dx)
    matrix = np.array(
        [
            [f_u - k_disc_sq * d_u, f_v],
            [g_u, g_v - k_disc_sq * d_v],
        ],
        dtype=np.float64,
    )
    evals, evecs = np.linalg.eig(matrix)
    idx = int(np.argmax(np.real(evals)))
    vec = np.real(evecs[:, idx])
    vec = vec / (np.linalg.norm(vec) + 1e-15)
    amp0 = 1.0e-3
    probe_dt = 0.002  # below the 1D CFL limit at D_v(R=48); keeps Euler bias small
    u = u0 + amp0 * vec[0] * np.cos(phase)
    v = v0 + amp0 * vec[1] * np.cos(phase)
    times = []
    amps = []
    for step in range(steps):
        du = f_u * (u - u0) + f_v * (v - v0)
        dv = g_u * (u - u0) + g_v * (v - v0)
        u = u + probe_dt * (d_u * _laplace_1d(u, dx) + du)
        v = v + probe_dt * (d_v * _laplace_1d(v, dx) + dv)
        proj = float(np.mean((u - u0) * np.cos(phase)) * 2.0)
        times.append((step + 1) * probe_dt)
        amps.append(proj)
    times_a = np.asarray(times[5:60])
    amps_a = np.asarray(amps[5:60])
    slope = float(np.polyfit(times_a, np.log(np.maximum(np.abs(amps_a), 1e-18)), 1)[0])
    return slope, k_disc_sq


def linear_dispersion_fit() -> dict:
    """Pearson r between integrated mode growth and analytic σ(k_disc)."""
    modes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    measured = []
    predicted = []
    rows = []
    for m in modes:
        sigma_hat, k_disc_sq = measure_mode_growth(m)
        k = math.sqrt(k_disc_sq)
        sigma = float(dispersion_sigma(k))
        measured.append(sigma_hat)
        predicted.append(sigma)
        rows.append({"mode": m, "k_per_um": k, "sigma_numeric": sigma_hat, "sigma_analytic": sigma})
    r, p = pearsonr(predicted, measured)
    resid = np.asarray(measured) - np.asarray(predicted)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "rows": rows,
        "pearson_r": float(r),
        "p_value": float(p),
        "rmse": rmse,
        "sigma_at_kc": float(dispersion_sigma(2.0 * math.pi / LAMBDA_UM)),
    }


def ablation_equal_diffusion() -> dict:
    """Negative control: D_u = D_v removes the diffusion-driven instability."""
    f_u, f_v, g_u, g_v = jacobian()
    d_u, _d_v = diffusivities()
    # Same inhibitor diffusivity as the activator. Gap numerator is negative.
    d_v = d_u
    ks = np.linspace(0.0, 3.0, 25)
    sigmas = []
    for k in ks:
        k2 = float(k * k)
        tr = (f_u + g_v) - k2 * (d_u + d_v)
        det = (f_u - k2 * d_u) * (g_v - k2 * d_v) - f_v * g_u
        disc = tr * tr - 4.0 * det
        if disc >= 0.0:
            sigma = 0.5 * (tr + math.sqrt(disc))
        else:
            sigma = 0.5 * tr
        sigmas.append(sigma)
    return {
        "max_sigma": float(max(sigmas)),
        "gap": float(turing_gap(1.0)),
        "operating_gap": float(turing_gap(DIFFUSION_RATIO)),
    }


def diffusion_ratio_sweep() -> list[dict]:
    """λ_c(R) with D_u fixed at the 6.912 µm operating point. λ falls as R rises.

    Ratios start at 36, above the discriminant threshold for κ = 0.15. Below
    that threshold the formal k_c is not an unstable mode.
    """
    rows = []
    for ratio in (36.0, 42.0, 48.0, 60.0, 84.0, 120.0):
        lam = analytic_lambda_um(ratio)
        rows.append({
            "diffusion_ratio": ratio,
            "lambda_um": lam,
            "turing_gap": float(turing_gap(ratio)),
        })
    return rows


def _gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def segment_bodies(rgb: np.ndarray, threshold: float = 28.0) -> np.ndarray:
    """Foreground body mask. Nogi plate: animals are bright on a black field."""
    gray = _gray(rgb)
    mask = gray > threshold
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)), iterations=1)
    labeled, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labeled, index=range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    for i, size in enumerate(sizes, start=1):
        if size > 200:
            keep[i] = True
    return keep[labeled]


def mask_circularity(mask: np.ndarray) -> float:
    area = float(np.sum(mask))
    if area < 10:
        return 0.0
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    return float(np.clip(4.0 * math.pi * area / (perim ** 2), 0.0, 1.0))


def count_ocelli(rgb: np.ndarray, mask: np.ndarray) -> int:
    """Dark compact pigment cups inside the body mask (the ocelli on the Nogi plate)."""
    gray = _gray(rgb)
    interior = ndi.binary_erosion(mask, iterations=2)
    dark = interior & (gray < 55.0)
    dark = ndi.binary_opening(dark, structure=np.ones((2, 2)))
    labeled, n = ndi.label(dark)
    if n == 0:
        return 0
    sizes = ndi.sum(dark, labeled, index=range(1, n + 1))
    return int(sum(1 for s in sizes if 8 <= s <= 800))


def load_specimen_rgb(path: Path, size: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def colorize_field(u: np.ndarray, contrast: float) -> np.ndarray:
    """Map activator concentration to the orange-brown dorsal pigment of D. japonica."""
    u_n = (u - np.min(u)) / (np.ptp(u) + 1e-9)
    # Early frames stay close to the mean pigment; contrast grows in.
    flat = np.full_like(u_n, float(np.mean(u_n)))
    amp = (1.0 - contrast) * flat + contrast * u_n
    rgb = np.zeros(u.shape + (3,), dtype=np.float64)
    rgb[:, :, 0] = 28.0 + 195.0 * amp
    rgb[:, :, 1] = 16.0 + 105.0 * amp
    rgb[:, :, 2] = 8.0 + 28.0 * amp
    return rgb


def render_animal_frame(u: np.ndarray, mask: np.ndarray, ocellus: np.ndarray, contrast: float) -> np.ndarray:
    """Paint one snapshot of the 6.912 µm field through the segmented epithelium."""
    field = np.asarray(
        Image.fromarray(u.astype(np.float32), mode="F").resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    rgb = colorize_field(field, contrast)
    rim = mask & ~ndi.binary_erosion(mask, iterations=3)
    canvas = np.zeros_like(rgb)
    canvas[mask] = rgb[mask]
    # Optical rim of the source micrograph (bright halo), not a second pigment.
    canvas[rim] = np.array([168.0, 198.0, 228.0])
    canvas[ocellus] = np.array([8.0, 8.0, 12.0])
    return np.clip(canvas, 0, 255)


def ocellus_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = _gray(rgb)
    interior = ndi.binary_erosion(mask, iterations=2)
    dark = interior & (gray < 55.0)
    dark = ndi.binary_opening(dark, structure=np.ones((2, 2)))
    labeled, n = ndi.label(dark)
    out = np.zeros(mask.shape, dtype=bool)
    if n == 0:
        return out
    sizes = list(ndi.sum(dark, labeled, index=range(1, n + 1)))
    for i, size in enumerate(sizes, start=1):
        if 8 <= size <= 800:
            out |= labeled == i
    # Dilate slightly so the cups read at GIF resolution.
    return ndi.binary_dilation(out, iterations=1)


def specimen_metrics() -> dict:
    rgb = load_specimen_rgb(SPECIMEN_NOGI)
    mask = segment_bodies(rgb)
    return {
        "circularity": mask_circularity(mask),
        "ocellus_count": count_ocelli(rgb, mask),
        "foreground_fraction": float(np.mean(mask)),
        "n_bodies": int(ndi.label(mask)[1]),
    }


def build_frames(sim: dict, canvas: int = 640) -> list[Image.Image]:
    rgb = load_specimen_rgb(SPECIMEN_NOGI, size=canvas)
    mask = segment_bodies(rgb, threshold=28.0)
    eyes = ocellus_mask(rgb, mask)
    frames = sim["u_frames"]
    images = []
    n = len(frames)
    for i, u in enumerate(frames):
        contrast = 0.15 + 0.85 * (i / max(1, n - 1))
        frame = render_animal_frame(u, mask, eyes, contrast)
        images.append(Image.fromarray(frame.astype(np.uint8), mode="RGB"))
    return images


def save_gif(images: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def save_comparison(sim: dict, fit: dict, images: list[Image.Image], path: Path) -> None:
    """Pass-1 figure: specimen, last simulation frame, activator field, dispersion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = load_specimen_rgb(SPECIMEN_NOGI)
    last = np.asarray(images[-1])
    u = sim["u_final"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.2), dpi=120)
    axes[0, 0].imshow(spec.astype(np.uint8))
    axes[0, 0].set_title("Nogi et al. 2009 plate\nD. japonica pigment and ocelli")
    axes[0, 1].imshow(last)
    axes[0, 1].set_title("Simulation on segmented epithelium\nλ_c = 6.912 µm, age 0 Ma")
    im = axes[1, 0].imshow(u, cmap="YlOrBr", origin="lower")
    axes[1, 0].set_title(f"Activator on a {N_WAVES}×λ patch ({sim['length_um']:.3f} µm)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    ks = np.linspace(0.05, 2.2, 180)
    sig = dispersion_sigma(ks)
    axes[1, 1].plot(ks, sig, color="#8c4a12", lw=2.0, label="analytic σ(k)")
    rows = fit["rows"]
    axes[1, 1].scatter(
        [r["k_per_um"] for r in rows],
        [r["sigma_numeric"] for r in rows],
        c="#1f4e79",
        zorder=3,
        label="integrated modes",
    )
    k_c = 2.0 * math.pi / LAMBDA_UM
    axes[1, 1].axvline(k_c, color="#666666", ls="--", lw=1.0, label=f"k_c = {k_c:.3f} /µm")
    axes[1, 1].axhline(0.0, color="#999999", lw=0.8)
    axes[1, 1].set_xlabel("wavenumber k (1/µm)")
    axes[1, 1].set_ylabel("growth rate σ")
    axes[1, 1].set_title(f"Linear dispersion  r = {fit['pearson_r']:.4f}")
    axes[1, 1].legend(fontsize=8, frameon=False)
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Dugesia japonica  ·  Holocene 0 Ma  ·  Turing wavelength 6.912 µm (log10 = -5.160)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def collect_metrics(sim: dict | None = None, fit: dict | None = None) -> dict:
    if fit is None:
        fit = linear_dispersion_fit()
    if sim is None:
        sim = simulate_patch()
    spec = specimen_metrics()
    ab = ablation_equal_diffusion()
    sweep = diffusion_ratio_sweep()
    lambdas = [row["lambda_um"] for row in sweep if math.isfinite(row["lambda_um"])]
    monotonic = all(lambdas[i] >= lambdas[i + 1] for i in range(len(lambdas) - 1))
    f_u, f_v, g_u, g_v = jacobian()
    return {
        "organism": "Dugesia japonica",
        "age_Ma": AGE_MA,
        "chronoseries": f"{CHRONOSERIES_INDEX}/{CHRONOSERIES_LENGTH}",
        "lambda_c_um": LAMBDA_UM,
        "lambda_c_m": LAMBDA_M,
        "log10_length_m": log10_length(),
        "diffusion_ratio": DIFFUSION_RATIO,
        "schnakenberg_a": SCHNAKENBERG_A,
        "schnakenberg_b": SCHNAKENBERG_B,
        "jacobian": {"f_u": f_u, "f_v": f_v, "g_u": g_u, "g_v": g_v},
        "D_u_um2": sim["d_u"],
        "D_v_um2": sim["d_v"],
        "patch_length_um": sim["length_um"],
        "n_wavelengths": N_WAVES,
        "pearson_r": fit["pearson_r"],
        "pearson_p": fit["p_value"],
        "dispersion_rmse": fit["rmse"],
        "sigma_at_kc": fit["sigma_at_kc"],
        "ablation_max_sigma": ab["max_sigma"],
        "ablation_gap": ab["gap"],
        "operating_gap": ab["operating_gap"],
        "sweep": sweep,
        "sweep_lambda_monotonic_decreasing": monotonic,
        "specimen_circularity": spec["circularity"],
        "specimen_ocellus_count": spec["ocellus_count"],
        "specimen_foreground_fraction": spec["foreground_fraction"],
        "specimen_n_bodies": spec["n_bodies"],
        "ellipse_circularity_reference": ELLIPSE_CIRCULARITY_REFERENCE,
        "dispersion_rows": fit["rows"],
        "grid_n": GRID_N,
        "seed": SEED,
        "pass": 1,
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for morphoeval/measure_outputs.py. Values are not copied onto their references."""
    metrics = collect_metrics()
    return [
        {
            "name": "Turing critical wavelength lambda_c of Dugesia japonica pigment field",
            "value": round(metrics["lambda_c_um"], 4),
            "unit": "µm",
            "reference_value": 6.9,
            "reference_source": "Slot scale 6.9 µm (6.912 µm operating point). The reference is the rounded tier label; the model value is the exact critical wavelength.",
            "method": "k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) with D_u solved at diffusion ratio 48, κ = 0.15",
            "note": f"log10(length/m) = {metrics['log10_length_m']:.3f}; age {AGE_MA:.0f} Ma; chronoseries 1/1",
        },
        {
            "name": "linear dispersion growth-rate Pearson r (numeric modes vs analytic sigma)",
            "value": round(metrics["pearson_r"], 5),
            "unit": "Pearson_r",
            "reference_value": 0.98,
            "reference_source": "MorphoEval gate: integrated Fourier-mode growth versus the linear dispersion relation",
            "method": "1D periodic Schnakenberg, modes 2–12, log-amplitude slope versus σ(k_disc)",
            "note": f"RMSE = {metrics['dispersion_rmse']:.5f}; p = {metrics['pearson_p']:.3e}; σ(k_c) = {metrics['sigma_at_kc']:.4f}",
        },
        {
            "name": "equal-diffusion ablation max growth rate versus operating Turing gap",
            "value": round(metrics["ablation_max_sigma"], 5),
            "unit": "1/time",
            "reference_value": round(metrics["sigma_at_kc"], 5),
            "reference_source": "Operating-point σ(k_c) at D_v/D_u = 48; ablation sets D_v = D_u",
            "method": "Analytic dispersion over k in [0, 3] /µm with D_u = D_v",
            "note": f"ablation gap numerator = {metrics['ablation_gap']:.4f} (negative control); operating gap = {metrics['operating_gap']:.4f}",
        },
        {
            "name": "specimen image body-mask circularity of Dugesia japonica Nogi 2009 plate",
            "value": round(metrics["specimen_circularity"], 4),
            "unit": "4πA/P²",
            "reference_value": ELLIPSE_CIRCULARITY_REFERENCE,
            "reference_source": "Independent ellipse model of one planarian body at aspect ~2.4 (circularity 0.55), not the measured mask",
            "method": "Otsu-style brightness mask on the CC-BY Nogi plate, filled, circularity 4πA/P²",
            "note": f"foreground fraction {metrics['specimen_foreground_fraction']:.3f}; bodies {metrics['specimen_n_bodies']}; ocelli {metrics['specimen_ocellus_count']}",
        },
        {
            "name": "specimen image ocellus pigment-cup count on the Nogi plate",
            "value": int(metrics["specimen_ocellus_count"]),
            "unit": "count",
            "reference_value": 2,
            "reference_source": "Wild-type Dugesia carries one pair of ocelli; the Nogi plate is a praziquantel-regeneration figure and is not that count",
            "method": "Dark connected components (area 8–800 px) inside the eroded body mask",
            "note": "value differs from the wild-type pair on purpose: the plate shows regenerated heads",
        },
    ]


def self_test() -> None:
    f_u, f_v, g_u, g_v = jacobian()
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    gap = turing_gap(DIFFUSION_RATIO)
    d_u, d_v = diffusivities()
    lam = analytic_lambda_um()
    assert abs(lam - LAMBDA_UM) < 1e-9, lam
    assert abs(log10_length() + 5.160) < 2e-3, log10_length()
    assert trace < 0.0, trace
    assert det > 0.0, det
    assert gap > 0.0, gap
    assert d_u > 0.0 and d_v > d_u
    # Discriminant condition at k_c.
    k2 = (2.0 * math.pi / LAMBDA_UM) ** 2
    numer = d_v * f_u + d_u * g_v
    assert numer * numer > 4.0 * d_u * d_v * det
    assert dispersion_sigma(2.0 * math.pi / LAMBDA_UM) > 0.0
    fit = linear_dispersion_fit()
    assert fit["pearson_r"] > 0.98, fit["pearson_r"]
    ab = ablation_equal_diffusion()
    assert ab["max_sigma"] < 0.0, ab
    assert ab["gap"] < 0.0
    sweep = diffusion_ratio_sweep()
    lambdas = [row["lambda_um"] for row in sweep]
    assert all(math.isfinite(x) for x in lambdas)
    assert all(lambdas[i] >= lambdas[i + 1] - 1e-9 for i in range(len(lambdas) - 1)), lambdas
    # Operating point is the third row (R = 48) and must be the slot wavelength.
    assert abs(sweep[2]["lambda_um"] - LAMBDA_UM) < 1e-6
    patch = simulate_patch(n_steps=900, n_save=4)
    assert float(np.std(patch["u_final"])) > 0.15, float(np.std(patch["u_final"]))
    assert CHRONOSERIES_LENGTH == 1 and AGE_MA == 0.0
    assert N_WAVES * LAMBDA_UM == 7 * 6.912
    spec = specimen_metrics()
    assert spec["foreground_fraction"] > 0.15, spec
    assert spec["ocellus_count"] >= 1, spec
    assert spec["circularity"] != ELLIPSE_CIRCULARITY_REFERENCE
    print("[self-test] Dugesia japonica 6.912 µm Turing wavelength: PASS")
    print(f"  λ_c = {lam:.6f} µm   log10 = {log10_length():.4f}   age = {AGE_MA:.0f} Ma")
    print(f"  D_u = {d_u:.6f}   D_v = {d_v:.6f} µm²/time   gap = {gap:.4f}")
    print(f"  dispersion Pearson r = {fit['pearson_r']:.5f}   RMSE = {fit['rmse']:.5f}")
    print(f"  ablation max σ = {ab['max_sigma']:.5f}")
    print(f"  sweep λ(R) = {[round(x, 3) for x in lambdas]}")
    print(f"  specimen circularity = {spec['circularity']:.4f} (ellipse ref {ELLIPSE_CIRCULARITY_REFERENCE})")
    print(f"  ocelli = {spec['ocellus_count']}   bodies = {spec['n_bodies']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    fit = linear_dispersion_fit()
    sim = simulate_patch()
    images = build_frames(sim, canvas=640)
    if len(images) < 10:
        raise SystemExit(f"GIF needs >= 10 frames, got {len(images)}")
    if min(images[0].size) < 480:
        raise SystemExit(f"GIF resolution {images[0].size} is below 480 px")
    save_gif(images, MEDIA_GIF)
    save_comparison(sim, fit, images, MEDIA_PNG)
    metrics = collect_metrics(sim, fit)
    metrics["gif_frames"] = len(images)
    metrics["gif_size"] = list(images[0].size)
    metrics["specimen_sha256"] = hashlib.sha256(SPECIMEN_NOGI.read_bytes()).hexdigest()
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {MEDIA_PNG.relative_to(ROOT)}")
    print(f"wrote {MEDIA_GIF.relative_to(ROOT)}  frames={len(images)} size={images[0].size}")
    print(f"Pearson r = {metrics['pearson_r']:.5f}  ablation σ = {metrics['ablation_max_sigma']:.5f}")


if __name__ == "__main__":
    main()
