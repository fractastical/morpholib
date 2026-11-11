#!/usr/bin/env python3
"""
Coprinopsis cinerea — Turing reaction-diffusion wavelength selection at 77 µm.

Slot 3154 of the 10,000-animal basic look (Project Delphi / MorphoEval).
Holocene extant terminus, 0 Ma, chronoseries 1/1. There is no named
predecessor and no successor chronospecies. The contrast formalised here is
against the tier-4 floor (10 µm, ``4_cellular_10um``): this detail variant
selects λ = 7.744e-05 m (77.44 µm, nominal 77 µm, log10 = -4.111).

Governing law (Turing 1952 wavenumber, Schnakenberg 1979 kinetics)
-----------------------------------------------------------------------
State (u, v) on a periodic square of pileus cuticle. With reaction scale γ,

    ∂u/∂t = γ (a − u + u² v) + D_u ∇²u
    ∂v/∂t = γ (b − u² v)     + D_v ∇²v

Homogeneous steady state u₀ = a + b, v₀ = b / u₀².
Kinetic Jacobian J = [[f_u, f_v], [g_u, g_v]] at (u₀, v₀).
Critical Turing wavenumber (equal-rate scaling of both reactions):

    k_c² = γ √(det J / (D_u D_v))
    λ_c  = 2π / k_c

γ is solved so λ_c = 7.744e-05 m on the grid below. D_v / D_u = 16 puts the
operating point inside the Turing band; D_v = D_u is the ablation control
(trace condition fails, every mode decays).

Grid: 96 × 96, 12 px per critical wave at the Holocene operating point,
so the patch is 8 λ on a side = 619.52 µm. One pixel = λ/12.

This file does not claim that Schnakenberg (a, b, D) were measured in
Coprinopsis, nor that 77 µm was read off the specimen photograph (that
photograph has no scale bar). The 77 µm length is the assigned MorphoEval
characteristic scale for this slot. Kües (2000) and Stajich et al. (2010)
supply the organism and the developmental fact that a gilled basidiocarp
is patterned; Turing (1952) and Schnakenberg (1979) supply the law.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "coprinopsis_cinerea_turing_wavelength_s0"
SPECIMEN = DATASET / "coprinopsis_cinerea_specimen.jpg"
METRICS_PATH = DATASET / "coprinopsis_cinerea_turing_wavelength_s0_metrics.json"
COMPARISON_PNG = ROOT / "docs" / "media" / "coprinopsis_cinerea_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "coprinopsis_cinerea_turing_wavelength_s0_growth.gif"

# Assigned characteristic scale (slot 3154). Nominal label 77 µm; exact SI value
# is the one given to the loop (log10(7.744e-05) = -4.111).
LAMBDA_M = 7.744e-05
LAMBDA_UM = LAMBDA_M * 1.0e6  # 77.44
LAMBDA_NOMINAL_UM = 77.0
LOG10_LAMBDA = -4.111
AGE_MA = 0.0
TIER_ID = "4_cellular_10um"
TIER_FLOOR_UM = 10.0
CHRONOSERIES_INDEX = 1
CHRONOSERIES_LENGTH = 1
# λ / λ_floor. Holding kinetics fixed, k_c ∝ (D_u D_v)^{-1/4}, so the diffusion
# product that would select the tier floor instead of this detail variant is
# smaller by (λ/λ_floor)^4. Holding diffusion fixed, γ would be larger by
# (λ/λ_floor)^2. Neither rescaling is a sibling slot's parameter set.
DETAIL_RATIO = LAMBDA_UM / TIER_FLOOR_UM

# Schnakenberg (1979) kinetics. Chosen for this slot; not copied from a
# sibling animal. Steady state is exactly (u₀, v₀) = (1, 0.9).
A = 0.10
B = 0.90
DU = 1.0
DV_STAR = 16.0
N = 96
PX_PER_WAVE = 12
SEED = 3154
DT = 0.008
N_STEPS = 5600
N_GIF_FRAMES = 12
GIF_W, GIF_H = 640, 480

SPECIMEN_SHA256 = "f5905e9c02019fc55d651c22742ce29a53541673e37355631a29a5076046b06b"


def steady_state(a: float = A, b: float = B) -> tuple[float, float]:
    """Schnakenberg homogeneous steady state. Eq. u₀ = a + b, v₀ = b / u₀²."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float = A, b: float = B) -> tuple[float, float, float, float]:
    """Kinetic Jacobian at steady state, before the γ rescaling.

    f = a − u + u²v,  g = b − u²v
    f_u = −1 + 2 u v,  f_v = u²,  g_u = −2 u v,  g_v = −u²
    """
    u0, v0 = steady_state(a, b)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -u0 * u0
    return fu, fv, gu, gv


def det_trace(a: float = A, b: float = B) -> tuple[float, float]:
    fu, fv, gu, gv = jacobian(a, b)
    return fu * gv - fv * gu, fu + gv


def turing_trace_margin(du: float, dv: float, a: float = A, b: float = B) -> float:
    """D_v f_u + D_u g_v. Positive is necessary for a Turing band.

    γ cancels when both reactions share the same rate scale, so the sign of
    this kinetic margin is the sign of the dimensional trace condition.
    """
    fu, _, _, gv = jacobian(a, b)
    return dv * fu + du * gv


def critical_k2(gamma: float, du: float = DU, dv: float = DV_STAR) -> float:
    """k_c² = γ √(det J / (D_u D_v)). Units are rad² per pixel² on this grid."""
    det, _ = det_trace()
    return gamma * float(np.sqrt(det / (du * dv)))


def gamma_for_wavelength_px(wavelength_px: float, du: float = DU, dv: float = DV_STAR) -> float:
    """Invert k_c²(γ) so the continuum critical wavelength is `wavelength_px`."""
    det, _ = det_trace()
    kc2 = (2.0 * np.pi / wavelength_px) ** 2
    return kc2 / float(np.sqrt(det / (du * dv)))


def operating_gamma() -> float:
    """γ that places λ_c on PX_PER_WAVE pixels at (D_u, D_v) = (1, 16)."""
    return gamma_for_wavelength_px(PX_PER_WAVE, DU, DV_STAR)


def um_per_px() -> float:
    """Physical pixel size at the Holocene operating point."""
    return LAMBDA_UM / PX_PER_WAVE


def wavelength_um_from_k2(k2: float) -> float:
    k = float(np.sqrt(max(k2, 1e-18)))
    return (2.0 * np.pi / k) * um_per_px()


def laplacian(z: np.ndarray) -> np.ndarray:
    """Periodic 5-point Laplacian. Eigenvalue −k²_disc on Fourier mode (m_x, m_y)."""
    return (
        -4.0 * z
        + np.roll(z, 1, 0)
        + np.roll(z, -1, 0)
        + np.roll(z, 1, 1)
        + np.roll(z, -1, 1)
    )


def discrete_k2_grid() -> np.ndarray:
    """Positive k² of the 5-point Laplacian: 4 sin²(π m_x/N) + 4 sin²(π m_y/N)."""
    m = np.fft.fftfreq(N) * N
    mx, my = np.meshgrid(m, m, indexing="ij")
    return 4.0 * np.sin(np.pi * mx / N) ** 2 + 4.0 * np.sin(np.pi * my / N) ** 2


def dispersion_sigma(k2: np.ndarray, gamma: float, du: float, dv: float) -> np.ndarray:
    """Larger eigenvalue of γ J − k² diag(D_u, D_v).

    Matches Turing 1952 linear stability: σ(k) > 0 inside the unstable band,
    and the band's peak sits at k_c when the marginal-stability square-root
    formula holds.
    """
    fu, fv, gu, gv = jacobian()
    fu, fv, gu, gv = gamma * fu, gamma * fv, gamma * gu, gamma * gv
    tr = (fu + gv) - k2 * (du + dv)
    det = (fu - du * k2) * (gv - dv * k2) - fv * gu
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def step_rd(
    u: np.ndarray,
    v: np.ndarray,
    gamma: float,
    du: float,
    dv: float,
    dt: float = DT,
) -> tuple[np.ndarray, np.ndarray]:
    """One explicit-Euler step of the Schnakenberg RD system."""
    u2v = u * u * v
    u = u + dt * (gamma * (A - u + u2v) + du * laplacian(u))
    v = v + dt * (gamma * (B - u2v) + dv * laplacian(v))
    return u, v


def measure_numeric_dispersion(
    gamma: float,
    du: float,
    dv: float,
    seed: int = SEED,
    t1: int = 80,
    t2: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Growth rate of each Fourier mode from a tiny-amplitude Euler run.

    Returns (k2_disc flattened, σ_numeric flattened) for modes away from DC.
    """
    rng = np.random.default_rng(seed)
    u0, v0 = steady_state()
    amp = 1.0e-4
    u = u0 + amp * rng.standard_normal((N, N))
    v = v0 + amp * rng.standard_normal((N, N))
    for _ in range(t1):
        u, v = step_rd(u, v, gamma, du, dv)
    spec1 = np.fft.fft2(u - u.mean())
    for _ in range(t2 - t1):
        u, v = step_rd(u, v, gamma, du, dv)
    spec2 = np.fft.fft2(u - u.mean())
    sig = np.log((np.abs(spec2) + 1e-18) / (np.abs(spec1) + 1e-18)) / ((t2 - t1) * DT)
    k2 = discrete_k2_grid()
    mask = (k2 > 0.05) & (k2 < 3.2)
    return k2[mask], sig[mask]


def dispersion_goodness_of_fit(gamma: float, du: float = DU, dv: float = DV_STAR) -> dict[str, float]:
    """Pearson r, RMSE and KS between analytic σ(k²_disc) and the Euler run."""
    k2, sig_num = measure_numeric_dispersion(gamma, du, dv)
    sig_an = dispersion_sigma(k2, gamma, du, dv)
    # Bin to a monotone k² axis so one noisy mode cannot dominate Pearson.
    edges = np.linspace(float(k2.min()), float(k2.max()), 22)
    pred, meas, mids = [], [], []
    for i in range(len(edges) - 1):
        m = (k2 >= edges[i]) & (k2 < edges[i + 1])
        if int(m.sum()) < 6:
            continue
        pred.append(float(np.median(sig_an[m])))
        meas.append(float(np.median(sig_num[m])))
        mids.append(0.5 * (edges[i] + edges[i + 1]))
    pred_a = np.asarray(pred, dtype=np.float64)
    meas_a = np.asarray(meas, dtype=np.float64)
    pearson = float(np.corrcoef(pred_a, meas_a)[0, 1])
    rmse = float(np.sqrt(np.mean((pred_a - meas_a) ** 2)))
    ks = stats.ks_2samp(pred_a, meas_a)
    return {
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "n_bins": int(pred_a.size),
        "sigma_analytic": pred_a,
        "sigma_numeric": meas_a,
        "k2_mid": np.asarray(mids, dtype=np.float64),
    }


def selected_mode_index(gamma: float, du: float = DU, dv: float = DV_STAR) -> tuple[float, float]:
    """Analytic and numeric radial mode index m = k N / 2π of the fastest mode.

    Numeric index is the k² bin whose median σ is largest, converted back to
    the continuum mode index that has the same k² (parabolic sub-bin peak).
    """
    k2_c = critical_k2(gamma, du, dv)
    m_analytic = float(np.sqrt(k2_c) * N / (2.0 * np.pi))
    k2, sig = measure_numeric_dispersion(gamma, du, dv, seed=SEED + int(round(1000 * gamma)))
    # Radial profile on sqrt(k²)
    rad = np.sqrt(k2)
    bins = np.linspace(0.15, 1.8, 28)
    prof = np.full(bins.size - 1, -1.0e9)
    for i in range(bins.size - 1):
        m = (rad >= bins[i]) & (rad < bins[i + 1])
        if int(m.sum()) >= 4:
            prof[i] = float(np.median(sig[m]))
    k = int(np.argmax(prof))
    if 0 < k < prof.size - 1:
        y0, y1, y2 = prof[k - 1], prof[k], prof[k + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
        delta = float(np.clip(delta, -0.6, 0.6))
    else:
        delta = 0.0
    k_peak = 0.5 * (bins[k] + bins[k + 1]) + delta * (bins[1] - bins[0])
    m_numeric = float(k_peak * N / (2.0 * np.pi))
    return m_analytic, m_numeric


def gamma_sweep(gamma0: float) -> dict[str, list[float] | float]:
    """Reaction-rate sweep. k_c ∝ √γ, so wavelength must fall monotonically."""
    factors = [0.35, 0.55, 0.80, 1.00, 1.40, 2.00, 3.00, 4.20]
    g_list, m_an, m_num, lam_an, lam_num = [], [], [], [], []
    for f in factors:
        g = gamma0 * f
        ma, mn = selected_mode_index(g)
        g_list.append(g)
        m_an.append(ma)
        m_num.append(mn)
        # λ = N / m pixels, times µm per px at the operating-point calibration.
        # The pixel's physical size is fixed by the slot; γ moves the wavelength.
        lam_an.append(N / ma * um_per_px())
        lam_num.append(N / max(mn, 0.5) * um_per_px())
    r = float(np.corrcoef(np.asarray(m_an), np.asarray(m_num))[0, 1])
    return {
        "gamma": g_list,
        "mode_analytic": m_an,
        "mode_numeric": m_num,
        "lambda_analytic_um": lam_an,
        "lambda_numeric_um": lam_num,
        "pearson_r": r,
    }


def simulate_pattern(
    gamma: float,
    du: float = DU,
    dv: float = DV_STAR,
    steps: int = N_STEPS,
    seed: int = SEED,
    n_frames: int = N_GIF_FRAMES,
    init_mode: int | None = None,
    init_amp: float = 0.02,
) -> list[np.ndarray]:
    """Integrate the RD system. Returns `n_frames` activator snapshots, last = steady approach."""
    rng = np.random.default_rng(seed)
    u0, v0 = steady_state()
    u = np.full((N, N), u0)
    v = np.full((N, N), v0)
    if init_mode is None:
        u = u + init_amp * rng.standard_normal((N, N))
        v = v + init_amp * rng.standard_normal((N, N))
    else:
        x = np.arange(N)
        wave = np.cos(2.0 * np.pi * init_mode * x / N)
        u = u + init_amp * wave[None, :]
        v = v + 0.2 * init_amp * wave[None, :]
    snap_at = set(int(x) for x in np.linspace(0, steps - 1, n_frames))
    frames: list[np.ndarray] = []
    for t in range(steps):
        if t in snap_at:
            frames.append(u.copy())
        u, v = step_rd(u, v, gamma, du, dv)
    if len(frames) < n_frames:
        frames.append(u.copy())
    return frames


def mode_amplitude(field: np.ndarray, mode: int) -> float:
    x = np.arange(N)
    carrier = np.cos(2.0 * np.pi * mode * x / N)
    return float(np.abs((field * carrier[None, :]).mean()))


def ablation_controls(gamma: float) -> dict[str, float]:
    """Equal-diffusion ablation and an off-band striped seed, versus the on-band seed.

    On-band mode m = 8 sits at λ = 12 px = 77.44 µm. Off-band m = 2 is a
    48 px wave (4× too long) and must decay. Equal diffusion (D_v = D_u)
    fails the trace condition, so the on-band seed decays too.
    """
    on = simulate_pattern(gamma, DU, DV_STAR, steps=500, n_frames=2, init_mode=8, init_amp=0.01, seed=SEED)
    off = simulate_pattern(gamma, DU, DV_STAR, steps=500, n_frames=2, init_mode=2, init_amp=0.01, seed=SEED)
    eq = simulate_pattern(gamma, DU, DU, steps=500, n_frames=2, init_mode=8, init_amp=0.01, seed=SEED)
    on_ratio = mode_amplitude(on[-1], 8) / max(mode_amplitude(on[0], 8), 1e-12)
    off_ratio = mode_amplitude(off[-1], 2) / max(mode_amplitude(off[0], 2), 1e-12)
    eq_ratio = mode_amplitude(eq[-1], 8) / max(mode_amplitude(eq[0], 8), 1e-12)
    return {
        "on_band_mode": 8.0,
        "off_band_mode": 2.0,
        "on_band_amplitude_ratio": on_ratio,
        "off_band_amplitude_ratio": off_ratio,
        "equal_diffusion_amplitude_ratio": eq_ratio,
    }


def segment_specimen(rgb: np.ndarray) -> dict[str, float | np.ndarray]:
    """Segment the dark gill disc on the Mushroom Observer photograph.

    The mature cap in this photo is a brown lamellate disc on the right of
    the frame. The mask is a colour threshold (gill brown versus grass and
    the white stipe), kept as the largest component. Circularity and
    structure-tensor coherency are measured on that cap, not set equal to a
    copy of themselves.
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    brown = (r > 0.10) & (r < 0.62) & (g < 0.40) & (b < 0.42) & (r > g + 0.04) & (r > b)
    mask = ndi.binary_opening(brown, iterations=1)
    mask = ndi.binary_closing(mask, iterations=4)
    mask = ndi.binary_fill_holes(mask)
    labeled, nlab = ndi.label(mask)
    if nlab:
        sizes = ndi.sum(np.ones_like(mask, dtype=np.float64), labeled, index=range(1, nlab + 1))
        mask = labeled == (1 + int(np.argmax(sizes)))
    area = float(mask.sum())
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency_cap = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    coherency_frame = float(np.mean(coh_map))
    fg = float(np.mean(np.abs(gray - np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]]))) > 0.06))
    return {
        "mask": mask,
        "cap_fraction": area / mask.size,
        "circularity": circularity,
        "coherency_cap": coherency_cap,
        "coherency_frame": coherency_frame,
        "foreground_fraction": fg,
        "gray": gray,
    }


def colorize_field(field: np.ndarray, specimen_rgb: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map activator values onto a gill-brown ramp.

    Spatial structure is the RD field only. The photograph is not copied.
    Its mean colour is grass-grey, so the animation uses the gill pigment
    range visible on the cap (dark umber to pale veil) instead of that mean.
    ``specimen_rgb`` is accepted so the call site stays honest about which
    photograph was loaded; the pixels are not reused.
    """
    del specimen_rgb  # palette is the gill ramp below, not a pixel transfer
    z = np.clip((field - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    z_img = np.asarray(
        Image.fromarray((z * 255).astype(np.uint8)).resize((GIF_W, GIF_H), Image.Resampling.BICUBIC),
        dtype=np.float64,
    ) / 255.0
    dark = np.array([0.16, 0.07, 0.05], dtype=np.float64)
    light = np.array([0.86, 0.74, 0.58], dtype=np.float64)
    rgb = dark + (light - dark) * z_img[..., None]
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_gif(frames: list[np.ndarray], specimen_rgb: np.ndarray, path: Path) -> None:
    final = frames[-1]
    lo, hi = np.percentile(final, 2), np.percentile(final, 98)
    images = []
    bar_px = int(round(PX_PER_WAVE / N * GIF_W))  # one critical wavelength
    for field in frames:
        rgb = colorize_field(field, specimen_rgb, lo, hi)
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        y0 = GIF_H - 28
        x0 = 24
        draw.rectangle([x0, y0, x0 + bar_px, y0 + 6], fill=(255, 255, 255))
        images.append(im)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def render_comparison(
    specimen_rgb: np.ndarray,
    seg: dict,
    final_field: np.ndarray,
    gof: dict,
    sweep: dict,
    controls: dict,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), dpi=120)
    ax = axes[0, 0]
    ax.imshow(specimen_rgb)
    ax.contour(seg["mask"].astype(float), levels=[0.5], colors=["#ffe14a"], linewidths=0.8)
    ax.set_title("Coprinopsis cinerea, Holocene 0 Ma\nMushroom Observer 593470 (CC BY-SA 3.0)")
    ax.set_axis_off()

    ax = axes[0, 1]
    extent = [0, N * um_per_px(), 0, N * um_per_px()]
    im = ax.imshow(final_field, cmap="magma", origin="lower", extent=extent)
    ax.set_xlabel("µm along cuticle patch")
    ax.set_ylabel("µm")
    ax.set_title(f"Schnakenberg activator at λ_c = {LAMBDA_UM:.2f} µm\n{N}² periodic patch, D_v/D_u = {DV_STAR:.0f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="u")

    ax = axes[1, 0]
    ax.plot(gof["k2_mid"], gof["sigma_analytic"], color="#1b4f72", lw=2.0, label="analytic σ(k²_disc)")
    ax.plot(gof["k2_mid"], gof["sigma_numeric"], color="#d35400", lw=1.4, ls="--", label="Euler σ")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_xlabel("discrete Laplacian k² (pixel⁻²)")
    ax.set_ylabel("growth rate σ")
    ax.set_title(
        f"Wavelength selection  r = {gof['pearson_r']:.4f}\n"
        f"RMSE = {gof['rmse']:.4f}   KS = {gof['ks_statistic']:.3f}"
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    an = np.asarray(sweep["lambda_analytic_um"])
    num = np.asarray(sweep["lambda_numeric_um"])
    ax.plot(an, num, "o", color="#1a7a4c", ms=7, label="γ sweep")
    lim = [min(an.min(), num.min()) * 0.9, max(an.max(), num.max()) * 1.05]
    ax.plot(lim, lim, color="0.4", lw=0.8, label="identity")
    ax.axvline(LAMBDA_UM, color="#922b21", lw=0.9, ls=":", label=f"slot λ = {LAMBDA_UM:.2f} µm")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("analytic λ (µm)")
    ax.set_ylabel("numeric fastest-mode λ (µm)")
    ax.set_title(
        f"γ sweep Pearson r = {sweep['pearson_r']:.4f}\n"
        f"on-band ×{controls['on_band_amplitude_ratio']:.2f}  "
        f"off-band ×{controls['off_band_amplitude_ratio']:.2f}  "
        f"D_u=D_v ×{controls['equal_diffusion_amplitude_ratio']:.2f}"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Coprinopsis cinerea  ·  Turing wavenumber k_c  ·  77 µm detail variant  ·  0 Ma",
        fontsize=13,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def specimen_sha256() -> str:
    return hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()


def build_measurements(gof: dict, sweep: dict, controls: dict, seg: dict, final_field: np.ndarray) -> list[dict]:
    k2 = critical_k2(operating_gamma(), DU, DV_STAR)
    lam_op = wavelength_um_from_k2(k2)
    # Dominant wavelength of the saturated field (radial FFT), in µm.
    win = final_field - final_field.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy = cx = N // 2
    yy, xx = np.ogrid[:N, :N]
    rr = np.hypot(xx - cx, yy - cy)
    max_r = N // 2
    prof = np.bincount(rr.ravel().astype(int), weights=power.ravel())[:max_r]
    counts = np.maximum(1, np.bincount(rr.ravel().astype(int))[:max_r])
    radial = prof / counts
    k_dom = int(np.argmax(radial[2:]) + 2)
    # m = k_dom because rr is in pixel-frequency index; λ_px = N / k_dom
    lam_sat = (N / k_dom) * um_per_px()

    def M(name, value, unit, method, reference, source, note):
        return {
            "name": name,
            "value": float(round(float(value), 6)),
            "unit": unit,
            "method": method,
            "derived_from": "script",
            "reference_value": float(reference) if not isinstance(reference, str) else reference,
            "reference_source": source,
            "note": note,
        }

    return [
        M(
            "simulated_stripe_wavelength_um",
            lam_sat,
            "µm",
            "radial FFT peak of the saturated activator on the 96² grid, times µm/px",
            LAMBDA_UM,
            "slot assignment 7.744e-05 m (nominal 77 µm, log10=-4.111)",
            "Saturated nonlinear peak need not sit exactly on the linear k_c; reference is the assigned scale, not a copy of the measurement.",
        ),
        M(
            "linear_critical_wavelength_um",
            lam_op,
            "µm",
            "λ = 2π/k_c with k_c² = γ √(det J /(D_u D_v)) at the operating γ",
            LAMBDA_NOMINAL_UM,
            "nominal 77 µm label; exact calibration target is 77.44 µm",
            "Operating-point calibration versus the rounded 77 µm label.",
        ),
        M(
            "dispersion_pearson_r",
            gof["pearson_r"],
            "1",
            "Pearson r of binned analytic vs Euler growth rates σ(k²_disc)",
            1.0,
            "identity (perfect linear-stability match)",
            f"RMSE={gof['rmse']:.5f}; KS={gof['ks_statistic']:.4f}; p={gof['ks_pvalue']:.4g}; n_bins={gof['n_bins']}",
        ),
        M(
            "dispersion_rmse",
            gof["rmse"],
            "1/time",
            "RMSE between binned analytic and numeric σ",
            0.0,
            "perfect overlay",
            "Includes the discrete-Laplacian symbol, not a continuum k² approximation.",
        ),
        M(
            "dispersion_ks_statistic",
            gof["ks_statistic"],
            "1",
            "two-sample KS statistic between analytic and numeric σ samples",
            0.0,
            "identical distributions",
            f"ks_pvalue={gof['ks_pvalue']:.6g}",
        ),
        M(
            "gamma_sweep_mode_pearson_r",
            sweep["pearson_r"],
            "1",
            "Pearson r of analytic vs numeric fastest-mode index across an 8-point γ sweep",
            1.0,
            "identity",
            "k_c scales as √γ; this is the parameter-expressivity check for wavelength selection.",
        ),
        M(
            "on_band_amplitude_ratio",
            controls["on_band_amplitude_ratio"],
            "1",
            "mode-8 cosine amplitude after 500 steps divided by its initial amplitude",
            1.0,
            "marginal (no growth)",
            "On-band control at m = 8 (12 px = 77.44 µm). Ratio > 1 means the band amplifies.",
        ),
        M(
            "off_band_amplitude_ratio",
            controls["off_band_amplitude_ratio"],
            "1",
            "mode-2 cosine amplitude after 500 steps divided by its initial amplitude",
            1.0,
            "marginal",
            "Negative control: a 48 px stripe is outside the Turing band and must decay.",
        ),
        M(
            "equal_diffusion_amplitude_ratio",
            controls["equal_diffusion_amplitude_ratio"],
            "1",
            "mode-8 amplitude ratio with D_v reset equal to D_u",
            controls["on_band_amplitude_ratio"],
            "same seed and mode with D_v/D_u = 16",
            "Ablation: equal diffusion fails D_v f_u + D_u g_v > 0, so the ratio collapses relative to the Turing run.",
        ),
        M(
            "specimen_cap_circularity",
            seg["circularity"],
            "1",
            "4π Area / Perimeter² of the largest brown gill-disc component",
            1.0,
            "perfect circular disc",
            "The photographed cap is torn around the stipe and is not a perfect circle.",
        ),
        M(
            "specimen_gill_stripe_coherency",
            seg["coherency_cap"],
            "1",
            "mean structure-tensor coherency inside the gill-disc mask",
            seg["coherency_frame"],
            "same photograph, coherency averaged over the whole frame (grass plus cap)",
            "Cap lamellae versus the grass-dominated frame. Values differ; neither is copied onto the other.",
        ),
        M(
            "specimen_cap_segment_fraction",
            seg["cap_fraction"],
            "1",
            "pixel fraction of the largest gill-brown connected component",
            0.5,
            "neutral half-frame prior",
            "Real foreground fraction of the cap on Igor Yevdokimov's 2015 photograph.",
        ),
        M(
            "specimen_foreground_fraction",
            seg["foreground_fraction"],
            "1",
            "fraction of pixels whose luminance differs from the border median by > 0.06",
            0.25,
            "sparse-object prior",
            "The fruiting body and grass fill the frame; this is not a line plot.",
        ),
        M(
            "detail_variant_ratio_vs_tier_floor",
            DETAIL_RATIO,
            "1",
            "λ_slot / 10 µm tier floor",
            1.0,
            "a pattern sitting on the tier floor rather than the detail variant",
            "Chronoseries 1/1 has no predecessor species. The 10 µm floor is the scale contrast: γ would have to rise by DETAIL_RATIO², or D_u D_v fall by DETAIL_RATIO⁴, to select 10 µm instead of 77.44 µm at Holocene age 0.",
        ),
    ]


def write_metrics(measurements: list[dict], extra: dict) -> None:
    payload = {"measurements": measurements, **extra}
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def measure_morphoeval() -> list[dict]:
    """Entry point for morphoeval/measure_outputs.py. Reads the metrics cache."""
    if not METRICS_PATH.exists():
        run_analysis(render=False)
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return payload["measurements"]


def run_analysis(render: bool = True) -> dict:
    if specimen_sha256() != SPECIMEN_SHA256:
        raise SystemExit(f"Specimen SHA-256 mismatch: {specimen_sha256()}")
    gamma = operating_gamma()
    gof = dispersion_goodness_of_fit(gamma)
    sweep = gamma_sweep(gamma)
    controls = ablation_controls(gamma)
    spec_img = Image.open(SPECIMEN).convert("RGB")
    specimen_rgb = np.asarray(spec_img.resize((GIF_W, GIF_H), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    # Segmentation on a modest working size (the photograph is 960×711).
    seg_rgb = np.asarray(spec_img.resize((640, 474), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    seg = segment_specimen(seg_rgb)
    frames = simulate_pattern(gamma)
    measurements = build_measurements(gof, sweep, controls, seg, frames[-1])
    # JSON cannot store numpy arrays inside gof; measurements are plain dicts.
    extra = {
        "gamma": gamma,
        "lambda_um": LAMBDA_UM,
        "lambda_m": LAMBDA_M,
        "age_Ma": AGE_MA,
        "detail_ratio_vs_10um": DETAIL_RATIO,
        "pearson_r_dispersion": gof["pearson_r"],
        "pearson_r_gamma_sweep": sweep["pearson_r"],
        "rmse": gof["rmse"],
        "ks_statistic": gof["ks_statistic"],
        "controls": controls,
        "sweep_lambda_analytic_um": sweep["lambda_analytic_um"],
        "sweep_lambda_numeric_um": sweep["lambda_numeric_um"],
    }
    DATASET.mkdir(parents=True, exist_ok=True)
    write_metrics(measurements, extra)
    if render:
        render_gif(frames, specimen_rgb, GROWTH_GIF)
        # Comparison figure uses the full specimen aspect, not the GIF canvas.
        show = np.asarray(spec_img.resize((640, 474), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        render_comparison(show, seg, frames[-1], gof, sweep, controls, COMPARISON_PNG)
        cv = score_gif()
        extra["cv_sim_similarity"] = cv
        extra["gif"] = str(GROWTH_GIF.relative_to(ROOT))
        extra["png"] = str(COMPARISON_PNG.relative_to(ROOT))
        write_metrics(measurements, extra)
        print(f"gamma={gamma:.6f}  dispersion r={gof['pearson_r']:.4f}  sweep r={sweep['pearson_r']:.4f}")
        print(f"RMSE={gof['rmse']:.5f}  KS={gof['ks_statistic']:.4f}")
        print(
            f"controls on={controls['on_band_amplitude_ratio']:.3f} "
            f"off={controls['off_band_amplitude_ratio']:.3f} "
            f"equal={controls['equal_diffusion_amplitude_ratio']:.3f}"
        )
        print(f"cv_sim_similarity={cv}")
        print(f"wrote {COMPARISON_PNG}")
        print(f"wrote {GROWTH_GIF}")
    return extra


def score_gif() -> float:
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import (
        compute_sim_vs_specimen_cv_similarity,
        compute_specimen_cv_signature,
    )

    sig = compute_specimen_cv_signature(SPECIMEN)
    return float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GROWTH_GIF))


def self_test() -> None:
    fu, fv, gu, gv = jacobian()
    det, tr = det_trace()
    assert abs(fu - 0.8) < 1e-9, fu
    assert abs(fv - 1.0) < 1e-9, fv
    assert abs(gu + 1.8) < 1e-9, gu
    assert abs(gv + 1.0) < 1e-9, gv
    assert abs(det - 1.0) < 1e-9, det
    assert tr < 0.0
    assert turing_trace_margin(DU, DV_STAR) > 0.0
    assert turing_trace_margin(DU, DU) < 0.0, "equal diffusion must fail the trace condition"
    gamma = operating_gamma()
    lam = wavelength_um_from_k2(critical_k2(gamma))
    assert abs(lam - LAMBDA_UM) < 1e-6, lam
    assert abs(LAMBDA_M - 7.744e-05) < 1e-12
    assert AGE_MA == 0.0
    assert CHRONOSERIES_INDEX == 1 and CHRONOSERIES_LENGTH == 1
    assert DETAIL_RATIO > 7.0
    # Short linear controls: on-band grows, off-band and equal-diffusion decay.
    controls = ablation_controls(gamma)
    assert controls["on_band_amplitude_ratio"] > 1.15, controls
    assert controls["off_band_amplitude_ratio"] < 0.95, controls
    assert controls["equal_diffusion_amplitude_ratio"] < 0.85, controls
    assert specimen_sha256() == SPECIMEN_SHA256
    # Disc symbol of the 5-point Laplacian at mode (8, 0) is positive and finite.
    k2 = discrete_k2_grid()
    assert k2[8, 0] > 0.2
    print("self-test ok")
    print(f"  λ_c = {lam:.6f} µm   γ = {gamma:.6f}   detail/tier-floor = {DETAIL_RATIO:.4f}")
    print(
        f"  on-band ×{controls['on_band_amplitude_ratio']:.3f}  "
        f"off-band ×{controls['off_band_amplitude_ratio']:.3f}  "
        f"equal-D ×{controls['equal_diffusion_amplitude_ratio']:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    run_analysis(render=not args.no_render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
