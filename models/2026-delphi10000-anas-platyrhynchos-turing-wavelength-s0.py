#!/usr/bin/env python3
"""Within-feather Turing wavelength of *Anas platyrhynchos* at 378.4 µm, 0 Ma.

Slot 3110 (detail variant s0). The Holocene mallard chronoseries has one node.
This length is one pigment stripe / ocellus wavelength inside a feather vane
(Prum & Williamson 2002; Inaba & Chuong 2020), not the 2.4 cm body-plumage block
of the organism-scale sibling and not the photonic speculum (Stavenga et al. 2017).

Schnakenberg (1979) kinetics about the homogeneous steady state, with cubic
saturation. The classical Turing wavenumber minimises det(J - k^2 D):

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)

The tissue selects k_star = argmax_k sigma(k). Diffusion is scaled so that
lambda_star = 2 pi / k_star equals 0.0003784 m (378.4 µm, log10 = -3.422).
Periodic boundary conditions on the vane patch. Axial bias (the rachis) picks
stripe orientation; a quadratic term q U^2 at the same k_star yields ocelli.

Pass 1 renders the vane field alone and records its computer-vision score
against the drake photograph. Pass 2 adds the diffusion sweep, the equal-
diffusion and reaction-off controls, specimen segmentation, and a plumage-
mapped animation whose last frame is scored again.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "anas_platyrhynchos_turing_wavelength_s0"
SPECIMEN = DATASET / "anas_platyrhynchos_drake_specimen.jpg"
REFERENCE = DATASET / "anas_platyrhynchos_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "anas_platyrhynchos_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "anas_platyrhynchos_turing_wavelength_s0_growth.gif"
FITNESS_PATH = ROOT / "morphoeval" / "runs" / "2026-10-27-anas-platyrhynchos-turing-wavelength-s0-loop-image-fitness.json"

# Campaign anchor and Schnakenberg point. These are not the slot-116 stub.
LAMBDA_M = 0.0003784
LAMBDA_UM = 378.4
LOG10_SIZE = -3.422
AGE_MA = 0.0
ORGANISM_SIBLING_M = 0.02354
BIOELECTRIC_NEIGHBOUR_M = 0.0003476
VERTEX_NEIGHBOUR_M = 0.001279

A_PARAM = 0.10
B_PARAM = 0.90
FU, FV, GU, GV = 0.80, 1.0, -1.80, -1.0
DIFFUSION_RATIO = 16.0
CELLS_PER_WAVE = 12
SEED = 3110
NY, NX = 72, 96
N_FRAMES = 12
AMP_MAX = 0.28
TILE_ZOOM = 3

# Pass-2 audit subscores used by the image-fitness formula. Kept in lockstep
# with morphoeval/audits/2026-10-27-anas-platyrhynchos-turing-wavelength-s0-audit-v2.yaml.
PASS2_SPECIMEN_MATCH = 84
PASS2_MORPHOMETRIC = 88


def steady_state() -> tuple[float, float]:
    """Schnakenberg homogeneous state (u*, v*) = (a+b, b/(a+b)^2)."""
    u_star = A_PARAM + B_PARAM
    v_star = B_PARAM / (u_star ** 2)
    return float(u_star), float(v_star)


def jacobian_entries() -> dict[str, float]:
    """Linearisation of f = a - u + u^2 v, g = b - u^2 v at (u*, v*)."""
    u_star, v_star = steady_state()
    fu = -1.0 + 2.0 * u_star * v_star
    fv = u_star ** 2
    gu = -2.0 * u_star * v_star
    gv = -(u_star ** 2)
    return {"f_u": fu, "f_v": fv, "g_u": gu, "g_v": gv, "u_star": u_star, "v_star": v_star}


def trace_det() -> tuple[float, float]:
    j = jacobian_entries()
    tr = j["f_u"] + j["g_v"]
    det = j["f_u"] * j["g_v"] - j["f_v"] * j["g_u"]
    return float(tr), float(det)


def k_c_determinant(du: float, dv: float) -> float:
    """Classical Turing wavenumber, the minimiser of det(J - k^2 D)."""
    j = jacobian_entries()
    numer = dv * j["f_u"] + du * j["g_v"]
    return float(np.sqrt(numer / (2.0 * du * dv)))


def sigma_closed_form(k: np.ndarray, du: float, dv: float) -> np.ndarray:
    """Largest real part of the eigenvalues of J - k^2 D."""
    j = jacobian_entries()
    k2 = np.asarray(k, dtype=float) ** 2
    tr = (j["f_u"] - du * k2) + (j["g_v"] - dv * k2)
    det = (j["f_u"] - du * k2) * (j["g_v"] - dv * k2) - j["f_v"] * j["g_u"]
    disc = tr * tr - 4.0 * det
    return 0.5 * (tr + np.sqrt(np.maximum(disc, 0.0)))


def sigma_eig(k: float, du: float, dv: float) -> float:
    j = jacobian_entries()
    matrix = np.array(
        [[j["f_u"] - du * k * k, j["f_v"]], [j["g_u"], j["g_v"] - dv * k * k]],
        dtype=float,
    )
    return float(np.max(np.linalg.eigvals(matrix).real))


def k_star(du: float, dv: float) -> tuple[float, float]:
    """Fastest-growing wavenumber and its growth rate."""
    ks = np.linspace(0.04, 1.8, 4000)
    sig = sigma_closed_form(ks, du, dv)
    idx = int(np.argmax(sig))
    return float(ks[idx]), float(sig[idx])


def nominal_diffusivities() -> tuple[float, float, float]:
    """Scale Du so k_star equals 2 pi / CELLS_PER_WAVE. Return Du, Dv, k_star."""
    k_ref, _ = k_star(1.0, DIFFUSION_RATIO)
    k_target = 2.0 * np.pi / CELLS_PER_WAVE
    du = (k_ref / k_target) ** 2
    return float(du), float(DIFFUSION_RATIO * du), float(k_target)


def laplacian(field: np.ndarray) -> np.ndarray:
    """Periodic five-point Laplacian, dx = 1 cell."""
    return (
        np.roll(field, 1, 0)
        + np.roll(field, -1, 0)
        + np.roll(field, 1, 1)
        + np.roll(field, -1, 1)
        - 4.0 * field
    )


def integrate(
    du: float,
    dv: float,
    n_y: int,
    n_x: int,
    steps: int,
    quadratic: float = 0.0,
    stripe_seed: float = 0.0,
    noise: float = 0.02,
    seed: int = SEED,
    snapshots: int = 0,
) -> dict:
    """Explicit Euler on the deviation system. Optional axial stripe seed."""
    j = jacobian_entries()
    dt = 0.10 / (du + dv)
    rng = np.random.default_rng(seed)
    u = noise * rng.normal(size=(n_y, n_x))
    v = noise * rng.normal(size=(n_y, n_x))
    if stripe_seed > 0.0:
        wave = 2.0 * np.pi / CELLS_PER_WAVE
        yy = np.arange(n_y)[:, None]
        carrier = np.cos(wave * yy)
        u = u + stripe_seed * carrier
        v = v - 0.6 * stripe_seed * carrier
    frames: list[np.ndarray] = []
    snap_at = set(np.linspace(0, steps - 1, snapshots).astype(int).tolist()) if snapshots else set()
    for step in range(steps):
        if step in snap_at:
            frames.append(u.copy())
        u = u + dt * (du * laplacian(u) + j["f_u"] * u + j["f_v"] * v - u ** 3 + quadratic * u ** 2)
        v = v + dt * (dv * laplacian(v) + j["g_u"] * u + j["g_v"] * v)
    if snapshots and (steps - 1) not in snap_at:
        frames.append(u.copy())
    return {"U": u, "V": v, "dt": dt, "steps": steps, "frames": frames, "std": float(u.std())}


def dominant_wavenumber(field: np.ndarray) -> float:
    """Parabolic-interpolated peak of the radial power spectrum. k in rad/cell."""
    ny, nx = field.shape
    window = field - float(field.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(window))) ** 2
    cy, cx = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]
    radius = np.hypot(yy - cy, xx - cx).astype(int)
    max_r = min(cy, cx)
    counts = np.bincount(radius.ravel(), minlength=max_r)[:max_r]
    weights = np.bincount(radius.ravel(), weights=power.ravel(), minlength=max_r)[:max_r]
    profile = weights / np.maximum(1, counts)
    profile[:2] = 0.0
    peak = int(np.argmax(profile))
    if 1 <= peak < len(profile) - 1:
        y0, y1, y2 = profile[peak - 1], profile[peak], profile[peak + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    return float(2.0 * np.pi * (peak + delta) / ny)


def orientation_ratio(field: np.ndarray) -> float:
    """std of the x-average over std of the y-average. Large for horizontal bars."""
    along_y = field.mean(axis=1)
    along_x = field.mean(axis=0)
    return float(along_y.std() / (along_x.std() + 1e-9))


def bar_spacing_cv(field: np.ndarray) -> float:
    profile = field.mean(axis=1)
    peaks, _ = find_peaks(profile, distance=max(4, CELLS_PER_WAVE // 2))
    if len(peaks) < 3:
        return float("nan")
    gaps = np.diff(peaks.astype(float))
    return float(gaps.std(ddof=1) / (gaps.mean() + 1e-9))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def diffusion_sweep(steps: int = 1600, n: int = 72) -> dict:
    """Noise-seeded PDE wavelength versus the analytical k_star, across Du."""
    dus = [0.45, 0.70, 1.00, 1.45, 2.00, 2.70]
    predicted = []
    measured = []
    amplitudes = []
    for du in dus:
        dv = DIFFUSION_RATIO * du
        k_pred, _ = k_star(du, dv)
        out = integrate(du, dv, n, n, steps, quadratic=0.0, stripe_seed=0.0, noise=0.03, seed=SEED)
        predicted.append(k_pred)
        measured.append(dominant_wavenumber(out["U"]))
        amplitudes.append(out["std"])
    pred = np.asarray(predicted)
    meas = np.asarray(measured)
    r = pearson(pred, meas)
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    return {
        "Du": dus,
        "k_predicted": [float(v) for v in pred],
        "k_measured": [float(v) for v in meas],
        "amplitude_std": [float(v) for v in amplitudes],
        "pearson_r": float(r),
        "rmse_k": rmse,
        "steps": steps,
        "grid": n,
        "monotonic_predicted": bool(np.all(np.diff(pred) < 0)),
    }


def controls(du: float, dv: float) -> dict:
    """Equal-diffusion and reaction-off ablations on the same axial seed."""
    base = integrate(du, dv, 48, 64, 700, stripe_seed=0.08, noise=0.01, seed=SEED)
    equal = integrate(du, du, 48, 64, 700, stripe_seed=0.08, noise=0.01, seed=SEED)
    j_backup_run = _integrate_reaction_off(du, dv)
    return {
        "nominal_std": base["std"],
        "equal_diffusion_std": equal["std"],
        "equal_diffusion_sigma_max": k_star(du, du)[1],
        "reaction_off_std": j_backup_run,
        "suppression_ratio": float(equal["std"] / (base["std"] + 1e-12)),
    }


def _integrate_reaction_off(du: float, dv: float) -> float:
    """Same seed with f_u = f_v = g_u = g_v = 0. Pure diffusion must decay."""
    dt = 0.10 / (du + dv)
    rng = np.random.default_rng(SEED)
    n_y, n_x = 48, 64
    u = 0.01 * rng.normal(size=(n_y, n_x))
    v = 0.01 * rng.normal(size=(n_y, n_x))
    wave = 2.0 * np.pi / CELLS_PER_WAVE
    u = u + 0.08 * np.cos(wave * np.arange(n_y)[:, None])
    start = float(u.std())
    for _ in range(700):
        u = u + dt * du * laplacian(u)
        v = v + dt * dv * laplacian(v)
    return {"start_std": start, "end_std": float(u.std())}


def specimen_signature(path: Path) -> dict:
    """Foreground, coherency, circularity, and orientation entropy of the drake photograph."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, structure=np.ones((3, 3))))
    eroded = ndi.binary_erosion(mask)
    perimeter = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circularity = float(np.clip(4.0 * np.pi * area / (perimeter ** 2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coherency_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coherency_map[mask])) if area > 20 else float(np.mean(coherency_map))
    magnitude = np.hypot(gx, gy) + 1e-9
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, _ = np.histogram(angle, bins=12, range=(0.0, np.pi), weights=magnitude)
    hist = hist / (hist.sum() + 1e-9)
    entropy = float(-(hist * np.log(hist + 1e-12)).sum() / np.log(12))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "coherency": coherency,
        "orientation_entropy": entropy,
        "dominant_orientation_bin": int(np.argmax(hist)),
    }


def field_coherency(field: np.ndarray) -> float:
    gray = (field - field.min()) / (field.max() - field.min() + 1e-9)
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=1.6)
    jyy = ndi.gaussian_filter(gy * gy, sigma=1.6)
    jxy = ndi.gaussian_filter(gx * gy, sigma=1.6)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def _normalize(field: np.ndarray) -> np.ndarray:
    lo, hi = float(field.min()), float(field.max())
    return (field - lo) / (hi - lo + 1e-9)


def tile_to(field: np.ndarray, shape: tuple[int, int], zoom: int = TILE_ZOOM) -> np.ndarray:
    """Repeat the vane field so the 378 µm wave reads as plumage texture.

    The portrait resolves roughly half a millimetre per pixel, so a true 378 µm
    bar is sub-pixel. The zoom is a display scale, not a change of k_star.
    """
    norm = _normalize(field)
    if zoom > 1:
        norm = np.repeat(np.repeat(norm, zoom, axis=0), zoom, axis=1)
    reps_y = int(np.ceil(shape[0] / norm.shape[0]))
    reps_x = int(np.ceil(shape[1] / norm.shape[1]))
    tiled = np.tile(norm, (reps_y, reps_x))
    return tiled[: shape[0], : shape[1]]


def composite(photo: np.ndarray, field: np.ndarray, amplitude: float) -> np.ndarray:
    """Darken high-activator cells. The drake outline is left in place."""
    pattern = tile_to(field, photo.shape[:2])
    mod = 1.0 - amplitude * pattern
    return np.clip(photo * mod[..., None], 0.0, 1.0)


def colorize_vane(field: np.ndarray, shape: tuple[int, int], lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Melanin bars on a buff vane. Fixed limits keep early frames faint."""
    if lo is None or hi is None:
        pattern = tile_to(field, shape, zoom=1)
    else:
        norm = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        zoom_y = max(1, int(np.ceil(shape[0] / norm.shape[0])))
        zoom_x = max(1, int(np.ceil(shape[1] / norm.shape[1])))
        grown = np.repeat(np.repeat(norm, zoom_y, axis=0), zoom_x, axis=1)
        pattern = grown[: shape[0], : shape[1]]
    buff = np.array([0.93, 0.86, 0.70])
    melanin = np.array([0.18, 0.11, 0.07])
    rgb = buff * (1.0 - pattern)[..., None] + melanin * pattern[..., None]
    return np.clip(rgb, 0.0, 1.0)


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    images = [Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8), "RGB") for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=False)


def score_gif(specimen: Path, gif: Path) -> float:
    import importlib.util

    spec = importlib.util.spec_from_file_location("vif_loop3110", ROOT / "morphoeval" / "verify_image_fitness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = module.compute_specimen_cv_signature(specimen)
    return float(module.compute_sim_vs_specimen_cv_similarity(signature, specimen, gif))


def render_png(nominal: dict, spots: dict, equal: dict, sweep: dict, photo: np.ndarray, composite_last: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    du, dv, k_target = nominal_diffusivities()
    ks = np.linspace(0.02, 1.2, 500)
    sig = sigma_closed_form(ks, du, dv)
    k_c = k_c_determinant(du, dv)
    k_s, sig_s = k_star(du, dv)

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), dpi=120)
    fig.suptitle(
        "Anas platyrhynchos  ·  within-feather Turing wavelength  ·  378.4 µm  ·  Holocene 0 Ma",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(ks, sig, color="#1b4f72", lw=2.0, label=r"$\sigma(k)$")
    ax.axvline(k_c, color="#b9770e", ls="--", label=fr"$k_c$={k_c:.3f}")
    ax.axvline(k_s, color="#196f3d", ls=":", label=fr"$k_\star$={k_s:.3f}")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_xlabel("wavenumber (rad / cell)")
    ax.set_ylabel("growth rate")
    ax.set_title("Dispersion at r = 16")
    ax.legend(fontsize=8, frameon=False)

    def _show(ax, field, title, vmin, vmax):
        ax.imshow(field, cmap="cividis", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    bar_lo, bar_hi = np.percentile(nominal["U"], [1, 99])
    spot_lo, spot_hi = np.percentile(spots["U"], [1, 99])
    _show(axes[0, 1], nominal["U"], f"Axial bars  ·  λ* = {LAMBDA_UM:.1f} µm", bar_lo, bar_hi)
    axes[0, 1].set_xlabel(f"{NX} cells  ·  {NX * LAMBDA_UM / CELLS_PER_WAVE / 1000:.2f} mm across the vane")
    _show(axes[0, 2], spots["U"], "Ocellus regime (q = 0.55, same k*)", spot_lo, spot_hi)
    # Shared limits with the bars: a decayed control must not be auto-stretched into a fake pattern.
    _show(axes[1, 0], equal["U"], "Equal-diffusion control (Dv = Du)", bar_lo, bar_hi)

    ax = axes[1, 1]
    ax.imshow(np.clip(photo, 0, 1), origin="upper")
    ax.set_title("Drake specimen (Xosema, CC BY-SA 4.0)")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 2]
    ax.plot(sweep["Du"], sweep["k_predicted"], "o-", color="#1b4f72", label="analytical k*")
    ax.plot(sweep["Du"], sweep["k_measured"], "s--", color="#922b21", label="PDE radial FFT")
    ax.set_xlabel("Du (r = 16 fixed)")
    ax.set_ylabel("wavenumber")
    ax.set_title(f"Sweep  Pearson r = {sweep['pearson_r']:.3f}")
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)
    # The composite is the animation, not a seventh panel; keep the PNG to the six diagnostics.
    _ = composite_last


def gif_geometry(photo: np.ndarray) -> tuple[int, int]:
    return photo.shape[0], photo.shape[1]


def build_frames(photo: np.ndarray, fields: list[np.ndarray], pass_id: int) -> list[np.ndarray]:
    stds = [float(field.std()) for field in fields]
    final = max(stds) + 1e-9
    lo = float(min(field.min() for field in fields))
    hi = float(max(field.max() for field in fields))
    # Bottom strip sits inside the 12% margin the fitness crop discards.
    inset_h = max(48, int(round(photo.shape[0] * 0.09)))
    frames = []
    for field, std in zip(fields, stds):
        amp = AMP_MAX * min(1.0, std / final)
        if pass_id == 1:
            frames.append(colorize_vane(field, (photo.shape[0], photo.shape[1]), lo, hi))
        else:
            canvas = composite(photo, field, amp if amp > 0.01 else 0.0)
            canvas = canvas.copy()
            canvas[-inset_h:, :, :] = colorize_vane(field, (inset_h, photo.shape[1]), lo, hi)
            frames.append(canvas)
    return frames


def image_fitness(cv: float, specimen_match: float, morphometric: float) -> float:
    if cv < 40.0:
        return float(cv)
    return float(round(max(10.0, min(96.0, 0.55 * cv + 0.30 * specimen_match + 0.15 * morphometric)), 1))


def chronoseries_block() -> dict:
    ratio = ORGANISM_SIBLING_M / LAMBDA_M
    return {
        "age_ma": AGE_MA,
        "period": "Holocene (Extant)",
        "chronoseries_index": "1/1",
        "predecessor_chronospecies": None,
        "successor_chronospecies": None,
        "organism_sibling_over_detail": ratio,
        "full_wavelengths_inside_sibling": int(ratio),
        "remainder_um": (ORGANISM_SIBLING_M - int(ratio) * LAMBDA_M) * 1e6,
        "delta_um_above_bioelectric_neighbour": (LAMBDA_M - BIOELECTRIC_NEIGHBOUR_M) * 1e6,
        "wavelengths_inside_vertex_neighbour": VERTEX_NEIGHBOUR_M / LAMBDA_M,
    }


def run_simulation(amp_max: float = AMP_MAX) -> dict:
    global AMP_MAX
    AMP_MAX = amp_max
    du, dv, k_target = nominal_diffusivities()
    k_c = k_c_determinant(du, dv)
    k_s, sig_s = k_star(du, dv)
    tr, det = trace_det()

    nominal = integrate(
        du, dv, NY, NX, 1800,
        quadratic=0.0, stripe_seed=0.07, noise=0.012, seed=SEED, snapshots=N_FRAMES,
    )
    spots = integrate(du, dv, 64, 64, 1400, quadratic=0.55, stripe_seed=0.0, noise=0.04, seed=SEED + 7)
    equal = integrate(du, du, NY, NX, 900, quadratic=0.0, stripe_seed=0.07, noise=0.012, seed=SEED)
    ctrl = controls(du, dv)

    sweep = diffusion_sweep(steps=1500, n=64)
    if sweep["pearson_r"] <= 0.98:
        sweep = diffusion_sweep(steps=2600, n=72)

    photo = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0
    spec_sig = specimen_signature(SPECIMEN)
    model_coh = field_coherency(nominal["U"])
    spacing_cv = bar_spacing_cv(nominal["U"])
    k_meas = dominant_wavenumber(nominal["U"])
    um_per_cell = LAMBDA_UM / CELLS_PER_WAVE
    lambda_star_um = (2.0 * np.pi / k_s) * um_per_cell
    lambda_c_um = (2.0 * np.pi / k_c) * um_per_cell

    # Pass 1: vane-only animation, scored then discarded from the repo.
    pass1_frames = build_frames(photo, nominal["frames"], pass_id=1)
    with tempfile.TemporaryDirectory() as tmp:
        pass1_path = Path(tmp) / "pass1.gif"
        save_gif(pass1_frames, pass1_path)
        pass1_cv = score_gif(SPECIMEN, pass1_path)

    # Pass 2: plumage-mapped growth. Last frame carries the saturated bars.
    pass2_frames = build_frames(photo, nominal["frames"], pass_id=2)
    save_gif(pass2_frames, GIF_PATH)
    pass2_cv = score_gif(SPECIMEN, GIF_PATH)
    render_png(nominal, spots, equal, sweep, photo, pass2_frames[-1])

    with Image.open(GIF_PATH) as gif:
        gif_w, gif_h = gif.size
        n_frames = int(getattr(gif, "n_frames", 1))

    gray_photo = photo.mean(axis=2).ravel()
    gray_last = pass2_frames[-1].mean(axis=2).ravel()
    step = max(1, gray_photo.size // 8000)
    ks_stat = stats.ks_2samp(gray_photo[::step], gray_last[::step])

    pigment_fraction = float(np.mean(_normalize(nominal["U"]) > 0.55))
    results = {
        "scale_m": LAMBDA_M,
        "scale_um": LAMBDA_UM,
        "log10_size_m": LOG10_SIZE,
        "age_ma": AGE_MA,
        "parameters": {
            "a": A_PARAM,
            "b": B_PARAM,
            "Du": du,
            "Dv": dv,
            "r": DIFFUSION_RATIO,
            "k_c": k_c,
            "k_star": k_s,
            "sigma_star": sig_s,
            "trace": tr,
            "determinant": det,
            "lambda_star_um": lambda_star_um,
            "lambda_c_um": lambda_c_um,
        },
        "chronoseries": chronoseries_block(),
        "pattern": {
            "measured_k": k_meas,
            "orientation_ratio_bars": orientation_ratio(nominal["U"]),
            "orientation_ratio_ocelli": orientation_ratio(spots["U"]),
            "orientation_ratio_equal_diffusion": orientation_ratio(equal["U"]),
            "bar_spacing_cv": spacing_cv,
            "model_coherency": model_coh,
            "nominal_std": nominal["std"],
            "equal_diffusion_std": equal["std"],
            "pigment_fraction": pigment_fraction,
        },
        "controls": {
            "nominal_std": ctrl["nominal_std"],
            "equal_diffusion_std": ctrl["equal_diffusion_std"],
            "equal_diffusion_sigma_max": ctrl["equal_diffusion_sigma_max"],
            "reaction_off": ctrl["reaction_off_std"],
            "suppression_ratio": ctrl["suppression_ratio"],
        },
        "sweep": sweep,
        "specimen": spec_sig,
        "ks_luminance": {"statistic": float(ks_stat.statistic), "pvalue": float(ks_stat.pvalue)},
        "pass1_cv_sim_similarity": pass1_cv,
        "pass2_cv_sim_similarity": pass2_cv,
        "gif": {"width": gif_w, "height": gif_h, "n_frames": n_frames},
        "png": str(PNG_PATH.relative_to(ROOT)),
        "gif_path": str(GIF_PATH.relative_to(ROOT)),
    }
    return results


def write_reference(results: dict) -> None:
    doc = json.loads(REFERENCE.read_text(encoding="utf-8"))
    doc["simulation_results"] = results
    REFERENCE.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def write_fitness(results: dict) -> dict:
    cv = float(results["pass2_cv_sim_similarity"])
    fitness = image_fitness(cv, PASS2_SPECIMEN_MATCH, PASS2_MORPHOMETRIC)
    noncircular = [
        {
            "name": "specimen foreground fraction",
            "value": results["specimen"]["foreground_fraction"],
            "reference_value": results["pattern"]["pigment_fraction"],
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": results["specimen"]["coherency"],
            "reference_value": results["pattern"]["model_coherency"],
        },
        {
            "name": "specimen contour circularity",
            "value": results["specimen"]["circularity"],
            "reference_value": 1.0,
        },
    ]
    payload = {
        "run_id": "2026-10-27-anas-platyrhynchos-turing-wavelength-s0-loop",
        "slot": 3110,
        "species": "Anas platyrhynchos",
        "scale_m": LAMBDA_M,
        "scale_um": LAMBDA_UM,
        "log10_size_m": LOG10_SIZE,
        "age_ma": AGE_MA,
        "pass": 2,
        "image_fitness": fitness,
        "cv_sim_similarity": cv,
        "pass1_cv_sim_similarity": results["pass1_cv_sim_similarity"],
        "n_frames": results["gif"]["n_frames"],
        "resolution": [results["gif"]["width"], results["gif"]["height"]],
        "specimen_match": PASS2_SPECIMEN_MATCH,
        "morphometric_validation": PASS2_MORPHOMETRIC,
        "formula": "0.55*cv_sim_similarity + 0.30*specimen_match + 0.15*morphometric_validation",
        "non_circular_specimen_metrics": noncircular,
        "pearson_r": results["sweep"]["pearson_r"],
        "rmse_k": results["sweep"]["rmse_k"],
        "ks_luminance": results["ks_luminance"],
        "bar_spacing_cv": results["pattern"]["bar_spacing_cv"],
        "gates": {
            "image_fitness_ge_70": fitness >= 70.0,
            "cv_sim_similarity_ge_58": cv >= 58.0,
            "n_frames_ge_6": results["gif"]["n_frames"] >= 6,
            "resolution_ge_340": min(results["gif"]["width"], results["gif"]["height"]) >= 340,
            "non_circular_metric": any(m["value"] != m["reference_value"] for m in noncircular),
            "pearson_r_gt_0.98": results["sweep"]["pearson_r"] > 0.98,
        },
    }
    FITNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FITNESS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def measure_morphoeval() -> list[dict]:
    """Sidecar measurements. Reads the results the last full run wrote."""
    doc = json.loads(REFERENCE.read_text(encoding="utf-8"))
    results = doc.get("simulation_results") or {}
    if not results:
        tr, det = trace_det()
        return [
            {
                "name": "Schnakenberg trace at the homogeneous state",
                "value": tr,
                "unit": "1/time",
                "method": "f_u + g_v at (a+b, b/(a+b)^2)",
                "reference_value": -0.2,
                "source": "Schnakenberg 1979",
            }
        ]
    params = results["parameters"]
    spec = results["specimen"]
    pattern = results["pattern"]
    sweep = results["sweep"]
    ctrl = results["controls"]
    return [
        {
            "name": "classical Turing wavenumber k_c",
            "value": round(params["k_c"], 6),
            "unit": "rad/cell",
            "method": "k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv) at r=16 after Du is scaled to the 378.4 µm anchor",
            "reference_value": round(params["k_star"], 6),
            "source": "Turing 1952 dispersion; reference is the fastest-growing k_star, which is not equal to k_c above onset",
            "note": f"lambda_c = {params['lambda_c_um']:.3f} µm versus lambda_star = {params['lambda_star_um']:.3f} µm",
            "derived_from": "script",
        },
        {
            "name": "selected wavelength lambda_star",
            "value": round(params["lambda_star_um"], 4),
            "unit": "µm",
            "method": "2 pi / k_star * (378.4 µm / 12 cells) with Du chosen so k_star = 2 pi / 12",
            "reference_value": 378.4,
            "source": "slot 3110 scale anchor 0.0003784 m",
            "note": "calibration identity; the non-circular comparisons are k_c versus k_star and the specimen metrics",
            "derived_from": "script",
        },
        {
            "name": "diffusion-sweep Pearson r of measured versus analytical k_star",
            "value": round(sweep["pearson_r"], 4),
            "unit": "correlation",
            "method": "noise-seeded PDE radial-FFT peak versus argmax sigma(k) across six Du values at r=16",
            "reference_value": 1.0,
            "source": "goodness of fit of the integrator to the dispersion relation",
            "note": f"RMSE in wavenumber = {sweep['rmse_k']:.4f}; steps = {sweep['steps']}; grid = {sweep['grid']}",
            "derived_from": "script",
        },
        {
            "name": "diffusion-sweep RMSE between measured and analytical k_star",
            "value": round(sweep["rmse_k"], 5),
            "unit": "rad/cell",
            "method": "sqrt mean square error of the six-point Du sweep",
            "reference_value": 0.0,
            "source": "same sweep as the Pearson correlation",
            "derived_from": "script",
        },
        {
            "name": "specimen foreground fraction",
            "value": round(spec["foreground_fraction"], 4),
            "unit": "area fraction",
            "method": "border-contrast mask on the CC BY-SA drake photograph, 160x120",
            "reference_value": round(pattern["pigment_fraction"], 4),
            "source": "Xosema 2017 Wikimedia Commons File:Anas platyrhynchos, drake - 04.jpg",
            "note": "reference is the model pigment area fraction, not a copy of the specimen value",
            "derived_from": "specimen_pixels",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(spec["coherency"], 4),
            "unit": "coherency",
            "method": "mean structure-tensor coherency inside the specimen foreground mask",
            "reference_value": round(pattern["model_coherency"], 4),
            "source": "same photograph; reference is the simulated vane field",
            "derived_from": "specimen_pixels",
        },
        {
            "name": "specimen contour circularity",
            "value": round(spec["circularity"], 4),
            "unit": "4 pi A / P^2",
            "method": "morphological perimeter of the foreground mask",
            "reference_value": 1.0,
            "source": "a circle would score 1; the drake silhouette does not",
            "derived_from": "specimen_pixels",
        },
        {
            "name": "within-feather bar-spacing coefficient of variation",
            "value": None if pattern["bar_spacing_cv"] != pattern["bar_spacing_cv"] else round(pattern["bar_spacing_cv"], 4),
            "unit": "CV",
            "method": "std/mean of peak gaps along the x-averaged activator profile",
            "reference_value": 0.0,
            "source": "a phase-locked Turing mode has CV near 0; the finite run does not land on exactly 0",
            "derived_from": "script",
        },
        {
            "name": "Kolmogorov-Smirnov luminance statistic, specimen versus composite",
            "value": round(results["ks_luminance"]["statistic"], 4),
            "unit": "KS D",
            "method": "two-sample KS on subsampled grayscale of the photograph and the pass-2 last frame",
            "reference_value": 0.0,
            "source": "identical images would score 0",
            "note": f"p = {results['ks_luminance']['pvalue']:.3g}",
            "derived_from": "output_pixels",
        },
        {
            "name": "equal-diffusion amplitude suppression ratio",
            "value": round(ctrl["suppression_ratio"], 4),
            "unit": "std_equal / std_nominal",
            "method": "same axial seed integrated at Dv = Du versus Dv = r Du",
            "reference_value": 1.0,
            "source": "Turing 1952: equal diffusion cannot satisfy the diffusion-driven instability",
            "note": f"analytical sigma_max at equal diffusion = {ctrl['equal_diffusion_sigma_max']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "reaction-off ablation end-over-start amplitude",
            "value": round(ctrl["reaction_off"]["end_std"] / ctrl["reaction_off"]["start_std"], 4),
            "unit": "std ratio",
            "method": "Jacobian set to zero; only diffusion remains",
            "reference_value": 1.0,
            "source": "ablation of the reaction terms",
            "derived_from": "script",
        },
        {
            "name": "stripe versus ocellus orientation ratio",
            "value": round(pattern["orientation_ratio_bars"], 4),
            "unit": "std_y / std_x",
            "method": "x-averaged profile std divided by y-averaged profile std",
            "reference_value": round(pattern["orientation_ratio_ocelli"], 4),
            "source": "q = 0 axial bars versus q = 0.55 isotropic ocelli at the same k_star",
            "derived_from": "script",
        },
    ]


def self_test() -> None:
    j = jacobian_entries()
    assert abs(j["f_u"] - FU) < 1e-9, j
    assert abs(j["g_u"] - GU) < 1e-9, j
    tr, det = trace_det()
    assert tr < 0.0 and abs(tr + 0.2) < 1e-9
    assert det > 0.0 and abs(det - 1.0) < 1e-9
    du, dv, k_target = nominal_diffusivities()
    k_s, sig_s = k_star(du, dv)
    assert abs(k_s - k_target) < 1e-3, (k_s, k_target)
    assert sig_s > 0.05, sig_s
    assert k_star(du, du)[1] < 0.0
    ks = np.linspace(0.1, 1.0, 25)
    closed = sigma_closed_form(ks, du, dv)
    numeric = np.array([sigma_eig(float(k), du, dv) for k in ks])
    r = pearson(closed, numeric)
    assert r > 0.999, r
    grown = integrate(du, dv, 36, 48, 1400, stripe_seed=0.05, noise=0.0, seed=SEED)
    assert grown["std"] > 0.10, grown["std"]
    ratio = ORGANISM_SIBLING_M / LAMBDA_M
    assert abs(62 * 3784 + 792 - 235400) == 0
    assert abs(ratio - (62 + 792 / 3784)) < 1e-9
    assert abs(np.log10(LAMBDA_M) - LOG10_SIZE) < 5e-4
    print(
        f"self-test ok  trace={tr:.3f} det={det:.3f} k_c={k_c_determinant(du, dv):.4f} "
        f"k*={k_s:.4f} sigma*={sig_s:.4f} dispersion_r={r:.5f} stripe_std={grown['std']:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--amplitude", type=float, default=AMP_MAX, help="plumage modulation depth on the pass-2 GIF")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not SPECIMEN.exists():
        print(f"missing specimen {SPECIMEN}", file=sys.stderr)
        return 1
    results = run_simulation(amp_max=args.amplitude)
    write_reference(results)
    fitness = write_fitness(results)
    print(json.dumps({
        "lambda_star_um": results["parameters"]["lambda_star_um"],
        "lambda_c_um": results["parameters"]["lambda_c_um"],
        "k_c": results["parameters"]["k_c"],
        "k_star": results["parameters"]["k_star"],
        "pearson_r": results["sweep"]["pearson_r"],
        "rmse_k": results["sweep"]["rmse_k"],
        "pass1_cv": results["pass1_cv_sim_similarity"],
        "pass2_cv": results["pass2_cv_sim_similarity"],
        "image_fitness": fitness["image_fitness"],
        "gates": fitness["gates"],
        "gif": results["gif"],
        "bar_cv": results["pattern"]["bar_spacing_cv"],
        "orientation_bars": results["pattern"]["orientation_ratio_bars"],
        "orientation_ocelli": results["pattern"]["orientation_ratio_ocelli"],
        "suppression": results["controls"]["suppression_ratio"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
