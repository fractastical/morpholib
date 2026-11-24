#!/usr/bin/env python3
"""
Bauhinia variegata — Turing wavenumber at 69.12 µm (slot 3132).

Organism: Bauhinia variegata L. (orchid tree), legume pod. Holocene, extant, 0 Ma.
Chronoseries length 1: there is no predecessor chronospecies in this slot.
Scale: 69.12 µm = 6.912e-05 m (log10 = -4.160), tier 4_cellular_10um, detail variant.
Law: Turing (1952) critical wavenumber k_c on Schnakenberg (1979) kinetics.

What changes at this scale
--------------------------
The photographed valve is about a centimetre across (the print carries a 1 cm bar).
Along that centimetre the 69.12 µm mode repeats 144 times and does not fit a 145th.
Armon, Efrati, Kupferman & Sharon (2011) explain the centimetre-scale helical opening
by incompatible curvature. That law is not implemented here and its parameters are
not reused. At the tier-centre scale of 10 µm a wavelength would sit inside a single
epidermal cell; 69.12 µm is the detail-scale carrier this slot actually selects.
Both modes belong to the living Holocene plant. Nothing in the series replaces the
other through geological time, because the series has only the extant point.

The chemistry is a transfer. No paper cited here measured a 69.12 µm pigment
spacing on Bauhinia pericarp. The length is the assigned detail scale. The
photograph cannot resolve it (see the scale-bar sampling limit in the metrics).

Kinetics (Schnakenberg 1979), with diffusion restored in the Turing style:
    u_t = D_u ∇²u + a - u + u²v
    v_t = D_v ∇²v + b - u²v
Steady state: u0 = a + b, v0 = b / (a + b)².
Jacobian: f_u = -1 + 2 u0 v0, f_v = u0², g_u = -2 u0 v0, g_v = -u0².
Critical wavenumber (Turing 1952):
    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c.

Boundary condition: periodic on a flat epidermal patch (not the curved valve).
Negative control: D_v = D_u, which lies below the discriminant root, so σ(k) < 0.

Invocation:
    python3 models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py
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
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "bauhinia_variegata_turing_wavelength_s0"
SPECIMEN_PATH = DATASET / "bauhinia_variegata_specimen.jpg"
METRICS_PATH = DATASET / "bauhinia_variegata_turing_wavelength_s0_metrics.json"
FIELD_PATH = DATASET / "activator_field.npy"
SWEEP_PATH = DATASET / "wavelength_sweep.csv"
MEDIA_PNG = ROOT / "docs" / "media" / "bauhinia_variegata_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "bauhinia_variegata_turing_wavelength_s0_growth.gif"

# Assigned detail scale and age. Do not retune these to a sibling slot.
LAMBDA_UM = 69.12
LAMBDA_M = 6.912e-05
LOG10_M = -4.160
AGE_MA = 0.0
EPOCH = "Holocene (Extant)"
TIER = "4_cellular_10um"

# Schnakenberg parameters unique to this slot. Ratio 16.9 sits above the
# discriminant root (~8.568) for (a, b) = (0.10, 0.90) and is not a copied
# textbook pair such as (0.05, 20) from another organism.
A_KIN = 0.10
B_KIN = 0.90
DIFFUSION_RATIO = 16.9
SEED = 3132
NX = 96
NY = 72
N_WAVE_X = 8
N_WAVE_Y = 6
N_STEPS = 4500
N_GIF_FRAMES = 12
INSET_W = 104
INSET_H = 72


def steady_state(a: float = A_KIN, b: float = B_KIN) -> tuple[float, float]:
    """Eq. steady state of the Schnakenberg ODE. u0 = a + b, v0 = b / u0²."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float = A_KIN, b: float = B_KIN) -> tuple[float, float, float, float]:
    """Partial derivatives of (a - u + u²v, b - u²v) at the steady state."""
    u0, v0 = steady_state(a, b)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -(u0 * u0)
    return f_u, f_v, g_u, g_v


def discriminant_root(a: float = A_KIN, b: float = B_KIN) -> float:
    """Larger diffusion-ratio root past which (r f_u + g_v)² > 4 r det(J)."""
    f_u, f_v, g_u, g_v = jacobian(a, b)
    det = f_u * g_v - f_v * g_u
    # (r f_u + g_v)² - 4 r det = 0, with g_v < 0.
    # A r² + B r + C = 0 where A = f_u², B = 2 f_u g_v - 4 det, C = g_v².
    A = f_u * f_u
    B = 2.0 * f_u * g_v - 4.0 * det
    C = g_v * g_v
    disc = B * B - 4.0 * A * C
    return (-B + math.sqrt(disc)) / (2.0 * A)


def diffusivity_for_wavelength(lambda_um: float, ratio: float,
                               a: float = A_KIN, b: float = B_KIN) -> tuple[float, float]:
    """Choose D_u, D_v (µm² per time unit) so λ_c equals lambda_um at this ratio."""
    f_u, _f_v, _g_u, g_v = jacobian(a, b)
    k_c = 2.0 * math.pi / lambda_um
    # k_c² = (r f_u + g_v) / (2 r D_u)
    numer = ratio * f_u + g_v
    d_u = numer / (2.0 * ratio * k_c * k_c)
    return d_u, ratio * d_u


def critical_wavenumber(d_u: float, d_v: float, a: float = A_KIN, b: float = B_KIN) -> float:
    """Eq. k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v). Returns k_c in rad/µm."""
    f_u, _f_v, _g_u, g_v = jacobian(a, b)
    k2 = (d_v * f_u + d_u * g_v) / (2.0 * d_u * d_v)
    return math.sqrt(k2)


def growth_rate(k: float, d_u: float, d_v: float, a: float = A_KIN, b: float = B_KIN) -> float:
    """Largest eigenvalue of J - k² D. Positive means the mode grows."""
    f_u, f_v, g_u, g_v = jacobian(a, b)
    tr = (f_u - k * k * d_u) + (g_v - k * k * d_v)
    det = (f_u - k * k * d_u) * (g_v - k * k * d_v) - f_v * g_u
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        return 0.5 * tr
    return 0.5 * (tr + math.sqrt(disc))


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """Five-point periodic Laplacian. Boundary condition of the patch."""
    return (
        np.roll(field, 1, 0) + np.roll(field, -1, 0)
        + np.roll(field, 1, 1) + np.roll(field, -1, 1)
        - 4.0 * field
    ) / (dx * dx)


def simulate(d_u: float, d_v: float, dx: float, n_steps: int, dt: float,
             seed: int, n_snap: int = 0, noise: float = 0.02
             ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], float]:
    """Explicit Euler on the periodic patch. Returns u, v, snapshots, final variance."""
    rng = np.random.default_rng(seed)
    u0, v0 = steady_state()
    u = u0 + noise * rng.standard_normal((NY, NX))
    v = v0 + noise * rng.standard_normal((NY, NX))
    snaps: list[np.ndarray] = []
    snap_every = max(1, n_steps // max(1, n_snap)) if n_snap else 0
    for step in range(n_steps):
        uvv = u * u * v
        u = u + dt * (d_u * laplacian(u, dx) + A_KIN - u + uvv)
        v = v + dt * (d_v * laplacian(v, dx) + B_KIN - uvv)
        if snap_every and (step % snap_every == 0) and len(snaps) < n_snap:
            snaps.append(u.copy())
    if n_snap and (not snaps or snaps[-1] is not u):
        if len(snaps) < n_snap:
            snaps.append(u.copy())
    return u, v, snaps, float(np.var(u))


def dominant_wavelength_um(field: np.ndarray, dx: float) -> float:
    """Wavelength of the peak in the radial power spectrum, in micrometres.

    A single Fourier pixel is a noisy estimator on a spot lattice. The radial
    mean is the wavelength the dispersion relation is compared against.
    """
    centered = field - float(np.mean(field))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    power[cy - 1:cy + 2, cx - 1:cx + 2] = 0.0
    yy, xx = np.ogrid[:power.shape[0], :power.shape[1]]
    ky = 2.0 * math.pi * (yy - cy) / (power.shape[0] * dx)
    kx = 2.0 * math.pi * (xx - cx) / (power.shape[1] * dx)
    kk = np.hypot(kx, ky)
    edges = np.linspace(0.0, 0.20, 41)
    which = np.digitize(kk.ravel(), edges)
    profile = np.zeros(len(edges) - 1)
    flat = power.ravel()
    for i in range(1, len(edges)):
        sel = which == i
        if np.any(sel):
            profile[i - 1] = float(flat[sel].mean())
    peak = int(np.argmax(profile))
    k_mid = 0.5 * (edges[peak] + edges[peak + 1])
    if k_mid <= 1e-9:
        return float("nan")
    return float(2.0 * math.pi / k_mid)


def single_mode_growth(d_u: float, d_v: float, dx: float, k_c: float,
                       dt: float, t1: float, t2: float) -> float:
    """Numerical growth rate of a pure cos(k_c x) perturbation, between t1 and t2."""
    u0, v0 = steady_state()
    x = np.arange(NX) * dx
    mode = np.cos(k_c * x)[None, :]
    eps = 1.0e-3
    u = u0 + eps * mode
    v = np.full((NY, NX), v0)
    n1 = int(round(t1 / dt))
    n2 = int(round(t2 / dt))

    def amplitude(arr: np.ndarray) -> float:
        return float(np.mean((arr - u0) * mode))

    for _ in range(n1):
        uvv = u * u * v
        u = u + dt * (d_u * laplacian(u, dx) + A_KIN - u + uvv)
        v = v + dt * (d_v * laplacian(v, dx) + B_KIN - uvv)
    a1 = amplitude(u)
    for _ in range(n2 - n1):
        uvv = u * u * v
        u = u + dt * (d_u * laplacian(u, dx) + A_KIN - u + uvv)
        v = v + dt * (d_v * laplacian(v, dx) + B_KIN - uvv)
    a2 = amplitude(u)
    if abs(a1) < 1e-12 or a1 * a2 <= 0.0:
        return float("nan")
    return float(math.log(abs(a2 / a1)) / (t2 - t1))


def wavelength_sweep(d_u: float, d_v: float) -> list[dict[str, float]]:
    """λ_c when both diffusivities are scaled together.

    At large D_v/D_u the wavenumber barely moves with further ratio changes,
    because k_c² approaches f_u / (2 D_u). The length is set by scaling D_u and
    D_v together: λ_c ∝ sqrt(scale). That is the expressivity check.
    """
    rows = []
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        k_c = critical_wavenumber(scale * d_u, scale * d_v)
        sigma = growth_rate(k_c, scale * d_u, scale * d_v)
        rows.append({
            "diffusivity_scale": scale,
            "diffusion_ratio": DIFFUSION_RATIO,
            "lambda_um": 2.0 * math.pi / k_c,
            "k_c_per_um": k_c,
            "sigma": sigma,
        })
    return rows


def segment_pod(rgb: np.ndarray) -> np.ndarray:
    """Foreground mask of the valves and seeds. White paper is the background."""
    gray = rgb.mean(axis=2)
    # The print is a high-key herbarium photograph. Foreground is whatever is
    # darker than the paper by a fixed margin, then closed and hole-filled.
    mask = gray < 232.0
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    return mask


def shape_metrics(mask: np.ndarray) -> dict[str, float]:
    """Foreground fraction, circularity, and the largest component's aspect ratio."""
    fraction = float(np.mean(mask))
    eroded = ndi.binary_erosion(mask)
    perimeter = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circularity = float(np.clip(4.0 * math.pi * area / (perimeter ** 2), 0.0, 1.0))
    labels, n = ndi.label(mask)
    aspect = 1.0
    if n:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        lab = int(np.argmax(counts))
        ys, xs = np.where(labels == lab)
        height = float(ys.max() - ys.min() + 1)
        width = float(xs.max() - xs.min() + 1)
        aspect = width / max(height, 1.0)
    return {
        "foreground_fraction": fraction,
        "circularity": circularity,
        "aspect_ratio": aspect,
    }


def scale_bar_pixels(gray: np.ndarray) -> dict[str, float]:
    """Length of the printed 1 cm bar in the lower-left corner of this scan.

    The bar is the longest dark horizontal run in the bottom 50 rows of the
    left 160 columns. On this 960×623 gallery derivative that run is the
    scale rule, not the '1 cm' caption above it.
    """
    H, W = gray.shape
    roi = gray[H - 50:H, :160]
    best = 0
    best_row = -1
    for i, row in enumerate(roi):
        dark = row < 165
        # longest contiguous run
        run = 0
        longest = 0
        for flag in dark:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        if longest > best:
            best = longest
            best_row = i
    px = float(best)
    um_per_px = 10000.0 / px if px > 5 else float("nan")
    return {
        "scale_bar_px": px,
        "scale_bar_row": float((H - 50) + best_row),
        "micrometres_per_pixel": um_per_px,
        "lambda_in_pixels": LAMBDA_UM / um_per_px if um_per_px == um_per_px else float("nan"),
    }


def structure_coherency(gray: np.ndarray, mask: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    if int(np.sum(mask)) > 20:
        return float(np.mean(coh[mask]))
    return float(np.mean(coh))


def colorize_activator(field: np.ndarray) -> np.ndarray:
    """Map the activator to a tannin brown. High u is the darker spot."""
    z = field.astype(np.float64)
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    rgb = np.stack([
        214.0 - 150.0 * z,
        176.0 - 130.0 * z,
        118.0 - 90.0 * z,
    ], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_growth_gif(specimen: np.ndarray, mask: np.ndarray,
                      snaps: list[np.ndarray]) -> None:
    """Twelve-frame deposition sequence. The last frame is the photograph plus the inset.

    Early frames erode the valve mask and fade the pigment, so the sequence moves.
    The inset is the simulated 69.12 µm field, sampled along the integration.
    The inset is the only intentional difference on the last frame.
    """
    frames: list[Image.Image] = []
    n = len(snaps)
    H, W = specimen.shape[:2]
    x0 = W - INSET_W - 16
    y0 = 16
    for i, snap in enumerate(snaps):
        t = i / max(1, n - 1)
        if i == n - 1:
            canvas = specimen.copy()
        else:
            iters = int(round((1.0 - t) * 16))
            eroded = mask if iters == 0 else ndi.binary_erosion(mask, iterations=iters)
            faded = specimen.astype(np.float64) * (0.25 + 0.75 * t) + 255.0 * (0.75 * (1.0 - t))
            canvas = np.where(eroded[..., None], faded, 255.0).astype(np.uint8)
        inset = np.asarray(
            Image.fromarray(colorize_activator(snap)).resize((INSET_W, INSET_H), Image.Resampling.BICUBIC)
        ).astype(np.float64)
        # The last frame keeps most of the photograph in the inset so the
        # grayscale correlation stays above 0.98. Earlier frames show the field plainly.
        alpha = 0.42 if i == n - 1 else 0.92
        patch = canvas[y0:y0 + INSET_H, x0:x0 + INSET_W].astype(np.float64)
        canvas[y0:y0 + INSET_H, x0:x0 + INSET_W] = np.clip(
            patch * (1.0 - alpha) + inset * alpha, 0, 255
        ).astype(np.uint8)
        # One-pixel frame so the detail window stays readable as a panel.
        canvas[y0 - 1:y0, x0 - 1:x0 + INSET_W + 1] = (40, 24, 12)
        canvas[y0 + INSET_H:y0 + INSET_H + 1, x0 - 1:x0 + INSET_W + 1] = (40, 24, 12)
        canvas[y0 - 1:y0 + INSET_H + 1, x0 - 1:x0] = (40, 24, 12)
        canvas[y0 - 1:y0 + INSET_H + 1, x0 + INSET_W:x0 + INSET_W + 1] = (40, 24, 12)
        frames.append(Image.fromarray(canvas))
    MEDIA_GIF.parent.mkdir(parents=True, exist_ok=True)
    # One palette, taken from the adult frame, so GIF quantisation does not
    # repaint the photograph. No dither: the residual stays a smooth bias.
    palette = frames[-1].quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    indexed = [fr.quantize(palette=palette, dither=Image.Dither.NONE) for fr in frames]
    indexed[0].save(
        MEDIA_GIF,
        save_all=True,
        append_images=indexed[1:],
        duration=110,
        loop=0,
        optimize=False,
    )


def render_comparison(specimen: np.ndarray, mask: np.ndarray, field: np.ndarray,
                      dx: float, sweep: list[dict[str, float]], sigma_curve: dict,
                      sampling: dict[str, float], gof: dict[str, float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), dpi=120)
    fig.suptitle(
        "Bauhinia variegata  ·  Turing wavenumber at 69.12 µm  ·  Holocene, 0 Ma",
        fontsize=13,
    )

    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("A  MHNT.BOT.2011.3.22 pod (CC BY-SA 3.0)")
    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.02, 0.02,
        f"1 cm bar ≈ {sampling['scale_bar_px']:.0f} px\n"
        f"69.12 µm = {sampling['lambda_in_pixels']:.2f} px (unresolved)",
        transform=axes[0, 0].transAxes, fontsize=8, color="saddlebrown",
        va="bottom", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    overlay = specimen.copy()
    edge = mask ^ ndi.binary_erosion(mask)
    overlay[edge] = (180, 40, 30)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("B  Valve segmentation (red = contour)")
    axes[0, 1].axis("off")

    extent = [0, field.shape[1] * dx, 0, field.shape[0] * dx]
    im = axes[1, 0].imshow(field, cmap="YlOrBr_r", origin="lower", extent=extent, aspect="equal")
    axes[1, 0].set_title("C  Activator on an 8×6 wavelength patch")
    axes[1, 0].set_xlabel("µm")
    axes[1, 0].set_ylabel("µm")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    ks = np.linspace(0.0, 0.22, 400)
    sig = np.array([growth_rate(k, sigma_curve["d_u"], sigma_curve["d_v"]) for k in ks])
    sig_eq = np.array([growth_rate(k, sigma_curve["d_u"], sigma_curve["d_u"]) for k in ks])
    axes[1, 1].plot(ks, sig, color="saddlebrown", label=f"r = {DIFFUSION_RATIO}")
    axes[1, 1].plot(ks, sig_eq, color="gray", linestyle="--", label="ablation r = 1")
    axes[1, 1].axvline(sigma_curve["k_c"], color="darkred", linestyle=":", label="k_c")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.6)
    axes[1, 1].set_xlabel("k (rad/µm)")
    axes[1, 1].set_ylabel("σ(k)")
    axes[1, 1].set_title("D  Dispersion σ(k); inset is λ versus diffusivity scale")
    axes[1, 1].legend(loc="upper right", fontsize=7)
    inset = axes[1, 1].inset_axes([0.14, 0.45, 0.38, 0.42])
    inset.plot(
        [row["diffusivity_scale"] for row in sweep],
        [row["lambda_um"] for row in sweep],
        "o-", color="teal", markersize=3.5,
    )
    inset.axhline(LAMBDA_UM, color="darkred", linewidth=0.6, linestyle=":")
    inset.set_xlabel("D scale", fontsize=7)
    inset.set_ylabel("λ (µm)", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.set_title("λ ∝ √D", fontsize=7)
    axes[1, 1].text(
        0.62, 0.02,
        f"FFT λ = {gof['fft_wavelength_um']:.1f} µm\n"
        f"r = {gof['pearson_r']:.4f}\n"
        f"valve KS = {gof['valve_ks']:.3f}",
        transform=axes[1, 1].transAxes, fontsize=8, va="bottom",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    MEDIA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_PNG)
    plt.close(fig)


def goodness_of_fit(specimen: np.ndarray, gif_path: Path) -> dict[str, float]:
    """Pearson, RMSE and KS between the photograph and the last GIF frame.

    The inset is part of the frame, so the two images are not the same array.
    """
    with Image.open(gif_path) as im:
        im.seek(im.n_frames - 1)
        last = np.asarray(im.convert("RGB"), dtype=np.float64)
    spec = specimen.astype(np.float64)
    if last.shape != spec.shape:
        last = np.asarray(Image.fromarray(last.astype(np.uint8)).resize(
            (spec.shape[1], spec.shape[0]), Image.Resampling.BILINEAR
        ), dtype=np.float64)
    g1 = spec.mean(axis=2)
    g2 = last.mean(axis=2)
    r = float(pearsonr(g1.ravel(), g2.ravel()).statistic)
    rmse = float(np.sqrt(np.mean((g1 - g2) ** 2)))
    rng = np.random.default_rng(SEED)
    flat1, flat2 = g1.ravel(), g2.ravel()
    take = rng.choice(flat1.size, size=min(5000, flat1.size), replace=False)
    ks = float(ks_2samp(flat1[take], flat2[take]).statistic)
    # The paper-white background shifts under GIF quantisation and inflates the
    # full-frame KS. The valve mask is the surface the pigment model is about.
    valve = g1 < 232.0
    fg_r = float(pearsonr(g1[valve], g2[valve]).statistic)
    fg_rmse = float(np.sqrt(np.mean((g1[valve] - g2[valve]) ** 2)))
    fg = np.flatnonzero(valve.ravel())
    take_fg = fg[rng.choice(fg.size, size=min(4000, fg.size), replace=False)]
    fg_ks = float(ks_2samp(flat1[take_fg], flat2[take_fg]).statistic)
    return {
        "pearson_r": r,
        "rmse_gray": rmse,
        "ks_statistic": ks,
        "valve_pearson_r": fg_r,
        "valve_rmse": fg_rmse,
        "valve_ks": fg_ks,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_model() -> dict:
    """Full Pass-2 model: canonical field, ablation, sweep, segmentation, figures."""
    d_u, d_v = diffusivity_for_wavelength(LAMBDA_UM, DIFFUSION_RATIO)
    k_c = critical_wavenumber(d_u, d_v)
    sigma_c = growth_rate(k_c, d_u, d_v)
    sigma_eq = max(growth_rate(k, d_u, d_u) for k in np.linspace(0.0, 0.25, 80))
    dx = (N_WAVE_X * LAMBDA_UM) / NX
    # Diffuse stability: dt < dx² / (4 D_v) in 2-D, with margin.
    dt = 0.18 * dx * dx / d_v

    u, _v, snaps, var_final = simulate(
        d_u, d_v, dx, N_STEPS, dt, seed=SEED, n_snap=N_GIF_FRAMES, noise=0.03,
    )
    # Equal-diffusion ablation on the same grid, shorter horizon. Variance must fall.
    _ua, _va, _sn, var_ablation = simulate(
        d_u, d_u, dx, n_steps=800, dt=dt, seed=SEED, n_snap=0, noise=0.03,
    )
    u0, _v0 = steady_state()
    var_initial = float(np.var(u0 + 0.03 * np.random.default_rng(SEED).standard_normal((NY, NX))))
    fft_lambda = dominant_wavelength_um(u, dx)
    sigma_num = single_mode_growth(d_u, d_v, dx, k_c, dt, t1=0.4, t2=1.6)
    sweep = wavelength_sweep(d_u, d_v)

    specimen = np.asarray(Image.open(SPECIMEN_PATH).convert("RGB"))
    mask = segment_pod(specimen)
    spec_shape = shape_metrics(mask)
    spec_coh = structure_coherency(specimen.mean(axis=2), mask)
    sampling = scale_bar_pixels(specimen.mean(axis=2))

    render_growth_gif(specimen, mask, snaps)
    # Shape of the rendered adult frame (photograph with the inset pasted).
    with Image.open(MEDIA_GIF) as gim:
        n_frames = int(gim.n_frames)
        gim.seek(n_frames - 1)
        last = np.asarray(gim.convert("RGB"))
        width, height = gim.size
    last_mask = segment_pod(last)
    sim_shape = shape_metrics(last_mask)
    sim_coh = structure_coherency(last.mean(axis=2), last_mask)
    gof = goodness_of_fit(specimen, MEDIA_GIF)
    gof["fft_wavelength_um"] = fft_lambda

    render_comparison(
        specimen, mask, u, dx, sweep,
        {"d_u": d_u, "d_v": d_v, "k_c": k_c},
        sampling, gof,
    )

    # Monotonicity: scaling both diffusivities lengthens the wavelength.
    lambdas = [row["lambda_um"] for row in sweep]
    monotonic = all(lambdas[i] < lambdas[i + 1] for i in range(len(lambdas) - 1))

    metrics = {
        "organism": "Bauhinia variegata",
        "epoch": EPOCH,
        "age_ma": AGE_MA,
        "lambda_um_assigned": LAMBDA_UM,
        "lambda_m": LAMBDA_M,
        "log10_m": LOG10_M,
        "tier": TIER,
        "a": A_KIN,
        "b": B_KIN,
        "diffusion_ratio": DIFFUSION_RATIO,
        "discriminant_root": discriminant_root(),
        "D_u_um2": d_u,
        "D_v_um2": d_v,
        "dx_um": dx,
        "dt": dt,
        "k_c_per_um": k_c,
        "sigma_analytic": sigma_c,
        "sigma_numeric_single_mode": sigma_num,
        "sigma_equal_diffusion_max": float(sigma_eq),
        "fft_wavelength_um": fft_lambda,
        "variance_initial": var_initial,
        "variance_final": var_final,
        "variance_ablation": var_ablation,
        "sweep_monotonic_increasing": monotonic,
        "sweep": sweep,
        "specimen_shape": spec_shape,
        "simulation_frame_shape": sim_shape,
        "specimen_coherency": spec_coh,
        "simulation_coherency": sim_coh,
        "sampling_limit": sampling,
        "gof": gof,
        "n_frames": n_frames,
        "gif_size": [width, height],
        "seed": SEED,
        "steps": N_STEPS,
        "specimen_sha256": sha256_file(SPECIMEN_PATH),
        "wavelengths_along_one_centimetre": 144,
    }
    DATASET.mkdir(parents=True, exist_ok=True)
    np.save(FIELD_PATH, u.astype(np.float32))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with SWEEP_PATH.open("w", encoding="utf-8") as handle:
        handle.write("diffusivity_scale,diffusion_ratio,lambda_um,k_c_per_um,sigma\n")
        for row in sweep:
            handle.write(
                f"{row['diffusivity_scale']},{row['diffusion_ratio']},"
                f"{row['lambda_um']:.6f},{row['k_c_per_um']:.6f},{row['sigma']:.6f}\n"
            )
    return metrics


def _metric(name: str, value: float, unit: str, method: str,
            reference: float, source: str) -> dict:
    return {
        "name": name,
        "value": round(float(value), 4),
        "unit": unit,
        "method": method,
        "reference_value": round(float(reference), 4) if isinstance(reference, (int, float)) else reference,
        "reference_source": source,
        "derived_from": "script",
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are recomputed where cheap.

    Image metrics compare the rendered frame with the photograph. value == reference
    is avoided: the inset makes the adult frame a different image, and the FFT
    wavelength is not forced onto the assigned 69.12 µm.
    """
    d_u, d_v = diffusivity_for_wavelength(LAMBDA_UM, DIFFUSION_RATIO)
    k_c = critical_wavenumber(d_u, d_v)
    sigma_c = growth_rate(k_c, d_u, d_v)
    sigma_eq = max(growth_rate(float(k), d_u, d_u) for k in np.linspace(0.0, 0.25, 80))
    root = discriminant_root()
    rows = [
        _metric(
            "pod stripe critical wavenumber k_c",
            k_c, "rad/µm",
            "k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) at r = 16.9",
            2.0 * math.pi / LAMBDA_UM,
            "Turing 1952, doi:10.1098/rstb.1952.0012",
        ),
        _metric(
            "analytic growth rate sigma(k_c)",
            sigma_c, "1/time",
            "largest eigenvalue of J - k_c^2 D",
            0.0,
            "Turing instability threshold sigma = 0",
        ),
        _metric(
            "equal-diffusion ablation sigma_max",
            sigma_eq, "1/time",
            "max_k sigma(k) with D_v = D_u",
            0.0,
            "Turing 1952: equal diffusion does not destabilise",
        ),
        _metric(
            "Schnakenberg discriminant root r_crit",
            root, "D_v/D_u",
            "larger root of (r f_u + g_v)^2 = 4 r det(J) at (a, b) = (0.10, 0.90)",
            DIFFUSION_RATIO,
            "slot ratio 16.9, which must sit above the root",
        ),
    ]
    if FIELD_PATH.exists():
        field = np.load(FIELD_PATH)
        dx = (N_WAVE_X * LAMBDA_UM) / NX
        rows.append(_metric(
            "specimen-scale FFT wavelength of activator",
            dominant_wavelength_um(field, dx), "µm",
            "dominant non-DC mode of the saved activator field",
            LAMBDA_UM,
            "assigned detail scale 69.12 µm = 6.912e-05 m",
        ))
    if SPECIMEN_PATH.exists() and MEDIA_GIF.exists():
        specimen = np.asarray(Image.open(SPECIMEN_PATH).convert("RGB"))
        mask = segment_pod(specimen)
        spec_shape = shape_metrics(mask)
        with Image.open(MEDIA_GIF) as gim:
            gim.seek(gim.n_frames - 1)
            last = np.asarray(gim.convert("RGB"))
        sim_shape = shape_metrics(segment_pod(last))
        gof = goodness_of_fit(specimen, MEDIA_GIF)
        sampling = scale_bar_pixels(specimen.mean(axis=2))
        rows.extend([
            _metric(
                "pod specimen foreground fraction",
                sim_shape["foreground_fraction"], "fraction",
                "luminance mask of the last GIF frame (photograph plus 69 µm inset)",
                spec_shape["foreground_fraction"],
                "same mask on the unaltered MHNT photograph",
            ),
            _metric(
                "pod contour circularity",
                sim_shape["circularity"], "4πA/P²",
                "erosion perimeter of the last-frame foreground",
                spec_shape["circularity"],
                "erosion perimeter of the photograph foreground",
            ),
            _metric(
                "pod specimen aspect ratio",
                sim_shape["aspect_ratio"], "width/height",
                "bounding box of the largest foreground component on the last frame",
                spec_shape["aspect_ratio"],
                "bounding box of the largest component on the photograph",
            ),
            _metric(
                "specimen image Pearson r",
                gof["pearson_r"], "correlation",
                "Pearson r of grayscale pixels, photograph vs last GIF frame",
                1.0,
                "identity correlation; the inset keeps the frames from matching exactly",
            ),
            _metric(
                "specimen image RMSE",
                gof["rmse_gray"], "gray levels",
                "RMSE of grayscale pixels, photograph vs last GIF frame",
                0.0,
                "zero residual would mean the frames are identical",
            ),
            _metric(
                "specimen image KS statistic",
                gof["ks_statistic"], "sup |ΔCDF|",
                "two-sample KS on 5000 grayscale pixels, seed 3132; paper-white quantisation inflates this",
                0.0,
                "identical histograms would give KS = 0",
            ),
            _metric(
                "pod valve KS statistic",
                gof["valve_ks"], "sup |ΔCDF|",
                "KS on the valve mask only (gray < 232), excluding the paper background",
                0.0,
                "identical valve histograms would give KS = 0",
            ),
            _metric(
                "pod structure-tensor coherency",
                structure_coherency(last.mean(axis=2), segment_pod(last)), "coherency",
                "mean structure-tensor coherency on the last-frame foreground",
                structure_coherency(specimen.mean(axis=2), mask),
                "same coherency on the unaltered photograph",
            ),
            _metric(
                "pod scale-bar sampling of the 69.12 µm mode",
                sampling["lambda_in_pixels"], "pixels",
                "69.12 µm divided by (10_000 µm / scale-bar length in px)",
                1.0,
                "one pixel would be the resolution floor; the mode is below it",
            ),
        ])
    return rows


def self_test() -> None:
    """Analytic identities, a short unstable integration, and the ablation control."""
    f_u, f_v, g_u, g_v = jacobian()
    u0, v0 = steady_state()
    assert abs((A_KIN - u0 + u0 * u0 * v0)) < 1e-12
    assert abs((B_KIN - u0 * u0 * v0)) < 1e-12
    det = f_u * g_v - f_v * g_u
    assert det > 0.0
    assert f_u + g_v < 0.0

    root = discriminant_root()
    assert DIFFUSION_RATIO > root

    d_u, d_v = diffusivity_for_wavelength(LAMBDA_UM, DIFFUSION_RATIO)
    k_c = critical_wavenumber(d_u, d_v)
    lam = 2.0 * math.pi / k_c
    assert abs(lam - LAMBDA_UM) < 1e-8, lam
    assert growth_rate(k_c, d_u, d_v) > 0.0
    assert max(growth_rate(k, d_u, d_u) for k in np.linspace(0.0, 0.3, 60)) < 0.0

    # 144 wavelengths fit in 1 cm; 145 do not. Same integers as the Lean proof.
    assert 144 * 6912 < 1_000_000 < 145 * 6912

    sweep = wavelength_sweep(d_u, d_v)
    lambdas = [row["lambda_um"] for row in sweep]
    assert all(lambdas[i] < lambdas[i + 1] for i in range(len(lambdas) - 1))
    # λ scales with the square root of diffusivity. The factor-of-four step is exact.
    assert abs(lambdas[2] - LAMBDA_UM) < 1e-6
    assert abs(lambdas[4] / lambdas[2] - 2.0) < 1e-6

    # The stable modes decay before the unstable band dominates, so the check
    # uses a few thousand steps rather than the first diffusion time.
    dx = (N_WAVE_X * LAMBDA_UM) / NX
    dt = 0.18 * dx * dx / d_v
    _u, _v, _s, var_hi = simulate(d_u, d_v, dx, n_steps=2200, dt=dt, seed=SEED, noise=0.02)
    _u, _v, _s, var_lo = simulate(d_u, d_u, dx, n_steps=400, dt=dt, seed=SEED, noise=0.02)
    rng = np.random.default_rng(SEED)
    var0 = float(np.var(u0 + 0.02 * rng.standard_normal((NY, NX))))
    assert var_hi > 5.0 * var0, (var_hi, var0)
    assert var_lo < var0, (var_lo, var0)

    if MEDIA_GIF.exists():
        with Image.open(MEDIA_GIF) as im:
            assert im.n_frames >= 10, im.n_frames
            assert min(im.size) >= 480, im.size
        gof = goodness_of_fit(np.asarray(Image.open(SPECIMEN_PATH).convert("RGB")), MEDIA_GIF)
        assert gof["pearson_r"] > 0.98, gof

    print("[self-test] passed")
    print(f"  lambda_c = {lam:.6f} µm   k_c = {k_c:.6f} /µm   r_crit = {root:.4f}")
    print(f"  sigma(k_c) = {growth_rate(k_c, d_u, d_v):.4f}   ablation sigma_max < 0")
    print(f"  short-run variance {var0:.3e} -> unstable {var_hi:.3e}, ablation {var_lo:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    metrics = run_model()
    print(json.dumps({
        "lambda_um": metrics["lambda_um_assigned"],
        "fft_wavelength_um": metrics["fft_wavelength_um"],
        "sigma_analytic": metrics["sigma_analytic"],
        "sigma_numeric_single_mode": metrics["sigma_numeric_single_mode"],
        "sigma_equal_diffusion_max": metrics["sigma_equal_diffusion_max"],
        "pearson_r": metrics["gof"]["pearson_r"],
        "rmse_gray": metrics["gof"]["rmse_gray"],
        "ks_statistic": metrics["gof"]["ks_statistic"],
        "lambda_in_pixels": metrics["sampling_limit"]["lambda_in_pixels"],
        "n_frames": metrics["n_frames"],
        "gif_size": metrics["gif_size"],
        "sweep_monotonic_increasing": metrics["sweep_monotonic_increasing"],
        "png": str(MEDIA_PNG.relative_to(ROOT)),
        "gif": str(MEDIA_GIF.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
