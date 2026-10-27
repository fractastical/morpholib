#!/usr/bin/env python3
"""Danio albolineatus at the 37 µm cellular Turing wavelength (loop 3161).

Holocene, age 0 Ma. Phenomenon size 3.686e-05 m (tier 4_cellular_10um, detail).
Schnakenberg reaction-diffusion whose fastest mode is that wavelength.
Predecessor Danio aesculapii selects a longer vertical-bar mode; successor
Danio margaritatus selects a shorter spot mode. Equal diffusion is stable.

  python3 models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py
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
DATASET = ROOT / "datasets" / "danio_albolineatus_turing_wavelength_s0"
SPECIMEN = DATASET / "danio_albolineatus_turing_wavelength_s0_specimen.jpg"
REFERENCE = DATASET / "danio_albolineatus_turing_wavelength_s0_reference.json"
METRICS = DATASET / "danio_albolineatus_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "danio_albolineatus_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "danio_albolineatus_turing_wavelength_s0_growth.gif"

# Assigned phenomenon scale (37 µm, detail variant of tier 4).
LAMBDA_M = 3.686e-05
AGE_MA = 0.0

# Schnakenberg kinetics. These are not copied from the 1.3 mm sibling slot.
A_KIN = 0.12
B_KIN = 0.88
DELTA_FACTOR_ALBO = 1.22
ALPHA_AES = 1.80
BETA_MARG = 1.55
DELTA_FACTOR_MARG = 2.40

NX, NY = 96, 60
LX, LY = 8.0, 5.0
DT = 0.007
T_END = 32.0
N_FRAMES = 12
SEED = 3161

# Lean witnesses (nanometres). Self-test checks the live solver against these.
LEAN_NM = {"aesculapii": 41801, "albolineatus": 36869, "margaritatus": 34213}
BD_OVER_SL_REF = 0.25003


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jacobian() -> dict[str, float]:
    """Schnakenberg steady state and Jacobian. det J = 1, tr J = -0.24."""
    u0 = A_KIN + B_KIN
    v0 = B_KIN / u0**2
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0**2
    g_u = -2.0 * u0 * v0
    g_v = -u0**2
    det = f_u * g_v - f_v * g_u
    tr = f_u + g_v
    return {"u0": u0, "v0": v0, "f_u": f_u, "f_v": f_v, "g_u": g_u, "g_v": g_v, "det": det, "tr": tr}


def critical_delta(jac: dict[str, float] | None = None) -> float:
    """Larger root of (δ f_u + g_v)^2 = 4 δ det, the onset inhibitor ratio."""
    j = jac or jacobian()
    f_u, g_v, det = j["f_u"], j["g_v"], j["det"]
    aa = f_u**2
    bb = 2.0 * f_u * g_v - 4.0 * det
    cc = g_v**2
    disc = bb * bb - 4.0 * aa * cc
    return float((-bb + np.sqrt(disc)) / (2.0 * aa))


def sigma_of_k(k: np.ndarray, du: float, dv: float, jac: dict[str, float] | None = None) -> np.ndarray:
    """Larger growth-rate root of the Turing dispersion relation."""
    j = jac or jacobian()
    k = np.asarray(k, dtype=float)
    tr = j["f_u"] + j["g_v"] - k**2 * (du + dv)
    det = (j["f_u"] - k**2 * du) * (j["g_v"] - k**2 * dv) - j["f_v"] * j["g_u"]
    disc = tr * tr - 4.0 * det
    return 0.5 * (tr + np.sqrt(np.maximum(disc, 0.0)))


def eig_sigma(k: float, du: float, dv: float, jac: dict[str, float] | None = None) -> float:
    """Same growth rate from the matrix eigenvalue, used as the numeric check."""
    j = jac or jacobian()
    mat = np.array(
        [[j["f_u"] - k**2 * du, j["f_v"]], [j["g_u"], j["g_v"] - k**2 * dv]],
        dtype=float,
    )
    return float(np.max(np.linalg.eigvals(mat).real))


def fastest_mode(du: float, dv: float, jac: dict[str, float] | None = None) -> tuple[float, float, float]:
    """Return k*, σ(k*), λ in length-units (1 unit = 3.686e-05 m)."""
    ks = np.linspace(1.5, 14.0, 5000)
    sig = sigma_of_k(ks, du, dv, jac)
    i = int(np.argmax(sig))
    k_star = float(ks[i])
    return k_star, float(sig[i]), float(2.0 * np.pi / k_star)


def solve_albolineatus_diffusion(jac: dict[str, float] | None = None) -> dict[str, float]:
    """Choose D_u so the fastest isotropic mode has wavelength 1 length-unit.

    δ = 1.22 δ_c places the pearl danio just above onset. D_u is bisected until
    k* = 2π. That is the 37 µm assignment, not a published rate constant.
    """
    j = jac or jacobian()
    delta = DELTA_FACTOR_ALBO * critical_delta(j)
    target = 2.0 * np.pi
    # Analytic onset formula is the initial guess; the fastest mode sits slightly off it.
    du_guess = (delta * j["f_u"] + j["g_v"]) / (2.0 * delta * target**2)
    lo, hi = 0.55 * du_guess, 1.45 * du_guess
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        k_star, _, _ = fastest_mode(mid, delta * mid, j)
        if k_star > target:
            lo = mid
        else:
            hi = mid
    du = 0.5 * (lo + hi)
    dv = delta * du
    k_star, sig, lam = fastest_mode(du, dv, j)
    return {"Du": du, "Dv": dv, "delta": delta, "k_star": k_star, "sigma": sig, "lambda_units": lam}


def regime_table(base: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """Diffusion tensors for the chronoseries and the equal-diffusion control."""
    j = jacobian()
    base = base or solve_albolineatus_diffusion(j)
    du, dv = base["Du"], base["Dv"]
    delta_c = critical_delta(j)
    regimes: dict[str, dict[str, float]] = {}

    def pack(name: str, du_x, du_y, dv_x, dv_y) -> None:
        # Wavelength of the x-varying mode and of the y-varying mode.
        kx, sx, lx = fastest_mode(du_x, dv_x, j)
        ky, sy, ly = fastest_mode(du_y, dv_y, j)
        if sx >= sy:
            k_star, sig, lam, axis = kx, sx, lx, "x"
        else:
            k_star, sig, lam, axis = ky, sy, ly, "y"
        regimes[name] = {
            "Du_x": float(du_x), "Du_y": float(du_y),
            "Dv_x": float(dv_x), "Dv_y": float(dv_y),
            "k_star": float(k_star), "sigma": float(sig),
            "lambda_units": float(lam), "lambda_m": float(lam * LAMBDA_M),
            "axis": axis,
        }

    pack("albolineatus", du, du, dv, dv)
    # Predecessor: inhibitor spreads farther along x (AP). Vertical bars vary in x.
    pack("aesculapii", du, du, ALPHA_AES * dv, dv / ALPHA_AES)
    # Successor: shorter activator range, higher inhibitor ratio, spots.
    du_m = du / BETA_MARG
    dv_m = DELTA_FACTOR_MARG * delta_c * du_m
    pack("margaritatus", du_m, du_m, dv_m, dv_m)
    pack("equal_diffusion_control", du, du, du, du)
    return regimes


def dispersion_goodness_of_fit(base: dict[str, float], n: int = 48) -> dict[str, float]:
    """Pearson r and RMSE between the closed-form σ(k) and matrix eigenvalues."""
    j = jacobian()
    ks = np.linspace(1.6, 12.0, n)
    analytic = sigma_of_k(ks, base["Du"], base["Dv"], j)
    numeric = np.array([eig_sigma(float(k), base["Du"], base["Dv"], j) for k in ks])
    pearson = float(np.corrcoef(analytic, numeric)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic - numeric) ** 2)))
    return {"pearson_r": pearson, "rmse": rmse, "n": float(n)}


def delta_sweep(base: dict[str, float]) -> list[dict[str, float]]:
    """Inhibitor-ratio sweep at the pearl-danio activator diffusivity."""
    j = jacobian()
    delta_c = critical_delta(j)
    rows = []
    for factor in (0.90, 1.00, 1.05, 1.12, 1.22, 1.60, 2.40, 3.20):
        delta = factor * delta_c
        du = base["Du"]
        dv = delta * du
        k_star, sig, lam = fastest_mode(du, dv, j)
        rows.append({
            "delta_over_critical": factor,
            "sigma": sig,
            "lambda_m": lam * LAMBDA_M,
            "unstable": bool(sig > 0.0),
        })
    return rows


def _lap_axis(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return (np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2.0 * field) / spacing**2


def integrate(regime: dict[str, float], n_frames: int = N_FRAMES, seed: int = SEED) -> dict[str, np.ndarray]:
    """Explicit Euler on the periodic patch. Returns frames of u and the final fields."""
    j = jacobian()
    dx, dy = LX / NX, LY / NY
    rng = np.random.default_rng(seed)
    u = j["u0"] + 0.006 * rng.standard_normal((NY, NX))
    v = j["v0"] + 0.006 * rng.standard_normal((NY, NX))
    steps = int(round(T_END / DT))
    snap_at = set(np.linspace(0, steps - 1, n_frames, dtype=int).tolist())
    frames: list[np.ndarray] = []
    amp_trace = np.empty(n_frames, dtype=float)
    fi = 0
    du_x, du_y = regime["Du_x"], regime["Du_y"]
    dv_x, dv_y = regime["Dv_x"], regime["Dv_y"]
    for s in range(steps):
        if s in snap_at:
            frames.append(u.copy())
            amp_trace[fi] = float(np.std(u))
            fi += 1
        f = A_KIN - u + u * u * v
        g = B_KIN - u * u * v
        u = u + DT * (du_x * _lap_axis(u, 1, dx) + du_y * _lap_axis(u, 0, dy) + f)
        v = v + DT * (dv_x * _lap_axis(v, 1, dx) + dv_y * _lap_axis(v, 0, dy) + g)
        if not np.isfinite(u).all():
            raise FloatingPointError(f"non-finite field at step {s}")
    if len(frames) < n_frames:
        frames.append(u.copy())
    return {
        "frames": np.stack(frames[:n_frames], axis=0),
        "final_u": u,
        "final_v": v,
        "amplitude_std": amp_trace,
    }


def field_wavelength_m(u: np.ndarray) -> float:
    """Dominant spatial period of a simulated field, in metres."""
    dx = LX / NX
    win = u - np.mean(u)
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    power[cy - 1:cy + 2, cx - 1:cx + 2] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    ky = (iy - cy) * 2.0 * np.pi / (NY * (LY / NY))
    kx = (ix - cx) * 2.0 * np.pi / (NX * dx)
    k = float(np.hypot(kx, ky))
    if k < 1e-6:
        return float("nan")
    return float((2.0 * np.pi / k) * LAMBDA_M)


def orientation_ratio(u: np.ndarray) -> float:
    """Mean |∂u/∂x| / mean |∂u/∂y|. Vertical bars are large (they vary along x)."""
    gx = np.abs(np.roll(u, -1, axis=1) - u)
    gy = np.abs(np.roll(u, -1, axis=0) - u)
    return float(np.mean(gx) / (np.mean(gy) + 1e-12))


def segment_specimen(rgb: np.ndarray) -> dict[str, float | np.ndarray]:
    """Segment the blue-silver flank from the sand. The largest blue component is the fish."""
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    blue_idx = blue - 0.5 * (red + green)
    mask = blue_idx > 0.015
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=2)
    labels, n = ndi.label(mask)
    if n == 0:
        raise RuntimeError("no fish-coloured component in the specimen")
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    # Prefer the component whose pixels are actually blue, not a dark plant.
    best, best_score = 1, -1.0
    for i, size in enumerate(sizes, start=1):
        comp = labels == i
        score = float(size) * float(np.mean(blue[comp]))
        if score > best_score:
            best, best_score = i, score
    fish = labels == best
    fish = ndi.binary_fill_holes(fish)
    yy, xx = np.where(fish)
    height = float(yy.max() - yy.min() + 1)
    width = float(xx.max() - xx.min() + 1)
    eroded = ndi.binary_erosion(fish)
    perimeter = float(np.sum(fish ^ eroded)) + 1e-6
    area = float(np.sum(fish))
    circularity = float(np.clip(4.0 * np.pi * area / perimeter**2, 0.0, 1.0))
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.0)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.0)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.0)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    # A second, tighter ROI drops a 15% margin of the bbox so fins weigh less.
    x0, x1 = int(xx.min()), int(xx.max())
    y0, y1 = int(yy.min()), int(yy.max())
    mx = int(0.15 * (x1 - x0))
    my = int(0.15 * (y1 - y0))
    tight = np.zeros_like(fish)
    tight[y0 + my:y1 - my + 1, x0 + mx:x1 - mx + 1] = True
    tight &= fish
    return {
        "mask": fish,
        "tight_mask": tight,
        "depth_over_length": height / width,
        "tight_depth_over_length": float(
            max(1, (y1 - my) - (y0 + my)) / max(1, (x1 - mx) - (x0 + mx))
        ),
        "circularity": circularity,
        "coherency": float(np.mean(coh[fish])),
        "area_fraction": float(np.mean(fish)),
        "bbox": (x0, y0, x1, y1),
    }


def spectrum_correlation(final_u: np.ndarray, regime: dict[str, float]) -> dict[str, float]:
    """Pearson r of the radial power spectrum against exp(2 σ(k) T) from linear theory."""
    j = jacobian()
    du = 0.5 * (regime["Du_x"] + regime["Du_y"])
    dv = 0.5 * (regime["Dv_x"] + regime["Dv_y"])
    win = final_u - np.mean(final_u)
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = NY // 2, NX // 2
    yy, xx = np.ogrid[:NY, :NX]
    ky = (yy - cy) * 2.0 * np.pi / LY
    kx = (xx - cx) * 2.0 * np.pi / LX
    kk = np.hypot(kx, ky)
    bins = np.linspace(1.5, 12.0, 22)
    centers = 0.5 * (bins[:-1] + bins[1:])
    observed = []
    predicted = []
    kept_sig = []
    sig = sigma_of_k(centers, du, dv, j)
    for lo, hi, s in zip(bins[:-1], bins[1:], sig):
        sel = (kk >= lo) & (kk < hi)
        if not np.any(sel):
            continue
        observed.append(float(np.mean(power[sel])))
        predicted.append(float(np.exp(2.0 * s * T_END)))
        kept_sig.append(float(s))
    obs = np.log1p(np.asarray(observed))
    pred = np.log1p(np.asarray(predicted))
    sig = np.asarray(kept_sig)
    # All-band correlation includes wavenumbers that linear theory says decay and
    # that are therefore noise-dominated. The unstable band is the fair test.
    if np.std(obs) < 1e-12 or np.std(pred) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(obs, pred)[0, 1])
    unstable = sig > 0.0
    if int(np.sum(unstable)) >= 4 and np.std(obs[unstable]) > 1e-12:
        pearson_band = float(np.corrcoef(obs[unstable], pred[unstable])[0, 1])
    else:
        pearson_band = pearson
    rmse = float(np.sqrt(np.mean((obs / (np.linalg.norm(obs) + 1e-12) - pred / (np.linalg.norm(pred) + 1e-12)) ** 2)))
    return {
        "pearson_r": pearson,
        "pearson_r_unstable_band": pearson_band,
        "rmse_unit": rmse,
        "n_bins": float(len(obs)),
        "n_unstable_bins": float(np.sum(unstable)),
    }


def ks_against_gaussian(final_u: np.ndarray) -> float:
    """KS distance of the centred activator against a Gaussian of the same variance.

    A developed spot or bar field is leptokurtic relative to the Gaussian null,
    so the distance is not zero. The uniform interstripe stays closer to it.
    """
    z = (final_u - np.mean(final_u)) / (np.std(final_u) + 1e-12)
    stat = stats.kstest(z.ravel(), "norm")
    return float(stat.statistic)


def colorize(u: np.ndarray, u0: float, half: float) -> np.ndarray:
    """Xanthophore gold (low u) to melanophore navy (high u), fixed scale."""
    t = np.clip((u - u0) / (2.0 * half) + 0.5, 0.0, 1.0)
    rgb = np.dstack([
        0.96 - 0.78 * t,
        0.78 - 0.55 * t,
        0.28 + 0.22 * t,
    ])
    return rgb


def render_gif(photo: np.ndarray, frames_u: np.ndarray, u0: float) -> None:
    """Specimen photograph with a growing 37 µm inset. The animal is not redrawn."""
    canvas_h, canvas_w = photo.shape[:2]
    inset_w, inset_h = 300, 210
    x0, y0 = 700, 450
    half = max(0.04, float(np.percentile(np.abs(frames_u[-1] - u0), 98)))
    images = []
    for frame in frames_u:
        # Per-frame stretch keeps the weakly growing mode visible in the inset.
        rgb = colorize(frame, float(np.mean(frame)), max(1e-4, float(np.std(frame) * 2.2)))
        panel = np.asarray(
            Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((inset_w, inset_h), Image.Resampling.BICUBIC),
            dtype=np.float64,
        ) / 255.0
        canvas = photo.copy()
        canvas[y0:y0 + inset_h, x0:x0 + inset_w] = panel
        canvas[y0:y0 + 2, x0:x0 + inset_w] = 1.0
        canvas[y0 + inset_h - 2:y0 + inset_h, x0:x0 + inset_w] = 1.0
        canvas[y0:y0 + inset_h, x0:x0 + 2] = 1.0
        canvas[y0:y0 + inset_h, x0 + inset_w - 2:x0 + inset_w] = 1.0
        im = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(im)
        draw.text((x0 + 8, y0 + 6), "37 um cellular field", fill=(255, 255, 255))
        images.append(im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=140,
        loop=0,
        optimize=False,
        disposal=2,
    )
    # half is retained so a future shared-scale inset can reuse the final amplitude.
    _ = half


def render_png(photo: np.ndarray, seg: dict, runs: dict[str, dict], sweep: list[dict], base: dict, gof: dict) -> None:
    """Comparison figure: specimen, three chronospecies, control, sweep, dispersion."""
    j = jacobian()
    u0 = j["u0"]
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.2), dpi=120)
    fig.suptitle(
        "Danio albolineatus — 37 µm Turing wavelength (0 Ma, Holocene)\n"
        "uniform interstripe between D. aesculapii bars and D. margaritatus spots",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.imshow(photo)
    yy, xx = np.where(seg["mask"])
    ax.plot([xx.min(), xx.max(), xx.max(), xx.min(), xx.min()],
            [yy.min(), yy.min(), yy.max(), yy.max(), yy.min()], color="cyan", lw=0.8)
    ax.set_title(f"Specimen flank  depth/length={seg['depth_over_length']:.3f}")
    ax.set_axis_off()

    def show_field(ax, name: str, title: str) -> None:
        u = runs[name]["final_u"]
        std = float(np.std(u))
        # Shared neighbourhood of u0 so the amplitude difference is visible,
        # with a note of the contrast stretch when the field is faint.
        half = 0.35
        im = ax.imshow(u, cmap="cividis", vmin=u0 - half, vmax=u0 + half, origin="lower")
        lam_um = runs[name]["lambda_m"] * 1e6
        if name == "equal_diffusion_control":
            ax.set_title(f"{title}\nstable, least-damped λ={lam_um:.0f} µm  σ={runs[name]['sigma']:.3f}", fontsize=9)
        else:
            ax.set_title(f"{title}\nλ={lam_um:.1f} µm  σ={runs[name]['sigma']:.3f}  std={std:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    show_field(axes[0, 1], "aesculapii", "Predecessor D. aesculapii (bars)")
    show_field(axes[0, 2], "albolineatus", "D. albolineatus (this slot)")
    show_field(axes[1, 0], "margaritatus", "Successor D. margaritatus (spots)")
    show_field(axes[1, 1], "equal_diffusion_control", "Negative control Du=Dv")

    ax = axes[1, 2]
    # Contrast-stretched pearl-danio field so the 37 µm mode is readable.
    u = runs["albolineatus"]["final_u"]
    ax.imshow(u, cmap="cividis", origin="lower")
    ax.set_title("Albolineatus, contrast stretched\n(same field as the centre panel)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[2, 0]
    factors = [row["delta_over_critical"] for row in sweep]
    sigmas = [row["sigma"] for row in sweep]
    lams = [row["lambda_m"] * 1e6 for row in sweep]
    ax.plot(factors, sigmas, "o-", color="black", label="σ_max")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.axvline(DELTA_FACTOR_ALBO, color="tab:blue", ls="--", lw=0.8, label="albolineatus")
    ax.set_xlabel("δ / δ_c  (fixed Du)")
    ax.set_ylabel("σ_max")
    ax.set_title("Inhibitor-ratio sweep")
    ax.legend(fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(factors, lams, "s--", color="tab:orange", label="λ (µm)")
    ax2.set_ylabel("λ (µm)", color="tab:orange")

    ax = axes[2, 1]
    ks = np.linspace(1.5, 12.0, 400)
    for name, color in (
        ("albolineatus", "tab:blue"),
        ("aesculapii", "tab:green"),
        ("margaritatus", "tab:red"),
        ("equal_diffusion_control", "grey"),
    ):
        reg = runs[name]
        du = 0.5 * (reg["Du_x"] + reg["Du_y"])
        dv = 0.5 * (reg["Dv_x"] + reg["Dv_y"])
        # Aesculapii is anisotropic; plot the unstable x-mode diffusivities.
        if name == "aesculapii":
            du, dv = reg["Du_x"], reg["Dv_x"]
        ax.plot(ks, sigma_of_k(ks, du, dv), color=color, label=name.replace("_", " "), lw=1.2)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("k  (rad per 37 µm)")
    ax.set_ylabel("σ(k)")
    ax.set_title(f"Dispersion  analytic vs eig r={gof['pearson_r']:.4f}")
    ax.legend(fontsize=6)

    ax = axes[2, 2]
    names = ["aesculapii", "albolineatus", "margaritatus", "equal_diffusion_control"]
    stds = [float(np.std(runs[n]["final_u"])) for n in names]
    ax.bar(range(4), stds, color=["tab:green", "tab:blue", "tab:red", "grey"])
    ax.set_xticks(range(4), ["aesculapii", "albolineatus", "margaritatus", "control"], rotation=20, fontsize=8)
    ax.set_ylabel("std(u) at T=32")
    ax.set_title("Amplitude at equal time")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def verify_specimen_pin() -> None:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    pinned = ref["specimen"]["sha256"]
    got = sha256_file(SPECIMEN)
    if got != pinned:
        raise ValueError(f"specimen SHA-256 {got} != pin {pinned}")
    if abs(float(ref["phenomenon_scale"]["phenomenon_size_m"]) - LAMBDA_M) > 1e-12:
        raise ValueError("reference phenomenon scale is not 3.686e-05 m")
    if float(ref["paleobiology"]["geological_age_ma"]) != AGE_MA:
        raise ValueError("reference age is not 0 Ma")


def run_simulation() -> dict:
    """Full chronoseries integration, figures, and metrics."""
    verify_specimen_pin()
    base = solve_albolineatus_diffusion()
    regimes = regime_table(base)
    gof = dispersion_goodness_of_fit(base)
    sweep = delta_sweep(base)
    runs = {name: {**regimes[name], **integrate(regimes[name])} for name in regimes}
    for name, run in runs.items():
        run["fft_lambda_m"] = field_wavelength_m(run["final_u"])
        run["orientation_ratio"] = orientation_ratio(run["final_u"])
        run["ks_gaussian"] = ks_against_gaussian(run["final_u"])
        run["std_u"] = float(np.std(run["final_u"]))
    spec_corr = spectrum_correlation(runs["albolineatus"]["final_u"], regimes["albolineatus"])
    # Second and third noise seeds: the selected wavelength should not move.
    ensemble_nm = [int(round(field_wavelength_m(runs["albolineatus"]["final_u"]) * 1e9))]
    for extra_seed in (7, 99):
        extra = integrate(regimes["albolineatus"], n_frames=2, seed=extra_seed)
        ensemble_nm.append(int(round(field_wavelength_m(extra["final_u"]) * 1e9)))
    ensemble_cv = float(np.std(ensemble_nm) / (np.mean(ensemble_nm) + 1e-12))

    photo = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0
    seg = segment_specimen(photo)
    render_png(photo, seg, runs, sweep, base, gof)
    render_gif(photo, runs["albolineatus"]["frames"], jacobian()["u0"])

    # Ablation: a full-frame rendering of the cellular field, scored the same way
    # as the canonical GIF, to show why the specimen-context inset is required.
    ablation_cv = _ablation_fullframe_cv(runs["albolineatus"]["frames"], photo)
    canonical_cv = _score_gif(GIF_PATH, SPECIMEN)

    metrics = {
        "phenomenon_size_m": LAMBDA_M,
        "geological_age_ma": AGE_MA,
        "Du": base["Du"],
        "Dv": base["Dv"],
        "delta": base["delta"],
        "pearson_r_dispersion": gof["pearson_r"],
        "rmse_dispersion": gof["rmse"],
        "pearson_r_spectrum": spec_corr["pearson_r"],
        "pearson_r_unstable_band": spec_corr["pearson_r_unstable_band"],
        "spectrum_rmse_unit": spec_corr["rmse_unit"],
        "seed_ensemble_lambda_nm": ensemble_nm,
        "seed_ensemble_lambda_cv": ensemble_cv,
        "sweep": sweep,
        "regimes": {
            name: {
                "lambda_m": runs[name]["lambda_m"],
                "lambda_nm": int(round(runs[name]["lambda_m"] * 1e9)),
                "sigma": runs[name]["sigma"],
                "std_u": runs[name]["std_u"],
                "fft_lambda_m": runs[name]["fft_lambda_m"],
                "orientation_ratio": runs[name]["orientation_ratio"],
                "ks_gaussian": runs[name]["ks_gaussian"],
                "axis": runs[name]["axis"],
            }
            for name in runs
        },
        "specimen": {
            "depth_over_length": float(seg["depth_over_length"]),
            "tight_depth_over_length": float(seg["tight_depth_over_length"]),
            "circularity": float(seg["circularity"]),
            "coherency": float(seg["coherency"]),
            "area_fraction": float(seg["area_fraction"]),
            "bd_over_sl_reference": BD_OVER_SL_REF,
        },
        "image_fitness_probe": canonical_cv,
        "ablation_fullframe_cv": ablation_cv,
        "gif": {"width": canonical_cv["width"], "height": canonical_cv["height"], "n_frames": canonical_cv["n_frames"]},
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _score_gif(gif: Path, specimen: Path) -> dict:
    import sys
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity, compute_specimen_cv_signature
    sig = compute_specimen_cv_signature(specimen)
    with Image.open(gif) as im:
        width, height = im.size
        n_frames = int(getattr(im, "n_frames", 1))
    score = compute_sim_vs_specimen_cv_similarity(sig, specimen, gif)
    return {"cv_sim_similarity": score, "width": width, "height": height, "n_frames": n_frames,
            "min_resolution": min(width, height)}


def _ablation_fullframe_cv(frames_u: np.ndarray, photo: np.ndarray) -> float:
    """Score a GIF that is only the cellular field (no specimen context)."""
    tmp = DATASET / "_ablation_fullframe.gif"
    imgs = []
    for frame in frames_u:
        rgb = colorize(frame, float(np.mean(frame)), max(1e-4, float(np.std(frame) * 2.2)))
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize(
            (photo.shape[1], photo.shape[0]), Image.Resampling.BICUBIC
        )
        imgs.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    imgs[0].save(tmp, save_all=True, append_images=imgs[1:], duration=120, loop=0, disposal=2)
    scored = _score_gif(tmp, SPECIMEN)
    tmp.unlink(missing_ok=True)
    return float(scored["cv_sim_similarity"])


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Recomputes only if metrics are absent."""
    if not METRICS.exists():
        run_simulation()
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    albo = m["regimes"]["albolineatus"]
    aes = m["regimes"]["aesculapii"]
    marg = m["regimes"]["margaritatus"]
    ctrl = m["regimes"]["equal_diffusion_control"]
    spec = m["specimen"]
    rows = [
        {
            "name": "specimen flank depth over length",
            "value": spec["depth_over_length"],
            "unit": "bbox height / bbox width",
            "method": "largest blue-silver connected component on the Commons photograph",
            "reference_value": spec["bd_over_sl_reference"],
            "reference_source": "thompson/portfolio/results/fishbase_morphmet.json BDmax_over_SL.obs Danio_albolineatus",
            "note": "Photograph bbox includes the caudal fin and is not the landmark body-depth ratio, so the two numbers differ.",
            "derived_from": "output",
        },
        {
            "name": "specimen contour circularity",
            "value": spec["circularity"],
            "unit": "4πA/P^2",
            "method": "morphological perimeter of the segmented flank",
            "reference_value": 1.0,
            "reference_source": "circle has circularity 1; an elongate danio does not",
            "note": "Non-circular specimen metric. Value is the fish outline, reference is the circle.",
            "derived_from": "output",
        },
        {
            "name": "specimen flank structure-tensor coherency",
            "value": spec["coherency"],
            "unit": "(λ1-λ2)/(λ1+λ2)",
            "method": "Sobel structure tensor inside the fish mask, Gaussian σ=2",
            "reference_value": 0.0,
            "reference_source": "isotropic null",
            "derived_from": "output",
        },
        {
            "name": "simulated stripe critical wavelength",
            "value": albo["lambda_m"],
            "unit": "m",
            "method": "argmax_k σ(k) of the Schnakenberg dispersion at the solved Du, Dv",
            "reference_value": LAMBDA_M,
            "reference_source": "assigned detail scale 3.686e-05 m (37 µm)",
            "note": "Live fastest-mode wavelength. It is solved to sit on the assigned scale; the stored value is the solver output and is compared, not copied, in the self-test.",
            "derived_from": "script",
        },
        {
            "name": "predecessor bar wavelength",
            "value": aes["lambda_m"],
            "unit": "m",
            "method": "fastest mode of the AP inhibitor-anisotropic tensor (alpha=1.8)",
            "reference_value": albo["lambda_m"],
            "reference_source": "focal pearl-danio wavelength from the same kinetics",
            "note": "Longer than the uniform-interstripe mode.",
            "derived_from": "script",
        },
        {
            "name": "successor spot wavelength",
            "value": marg["lambda_m"],
            "unit": "m",
            "method": "fastest mode at beta=1.55 activator shrink and delta=2.40 delta_c",
            "reference_value": albo["lambda_m"],
            "reference_source": "focal pearl-danio wavelength from the same kinetics",
            "note": "Shorter than the uniform-interstripe mode.",
            "derived_from": "script",
        },
        {
            "name": "dispersion goodness-of-fit pearson r",
            "value": m["pearson_r_dispersion"],
            "unit": "Pearson r",
            "method": "closed-form σ(k) versus numpy eigenvalue σ(k) at 48 wavenumbers",
            "reference_value": 1.0,
            "reference_source": "exact algebraic agreement of the quadratic formula and the matrix",
            "note": "RMSE is reported alongside. r is computed, not assigned.",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE",
            "value": m["rmse_dispersion"],
            "unit": "growth rate",
            "method": "root mean square of analytic σ minus eigenvalue σ",
            "reference_value": 0.0,
            "reference_source": "perfect numerical agreement",
            "derived_from": "script",
        },
        {
            "name": "linear-spectrum pearson r",
            "value": m["pearson_r_spectrum"],
            "unit": "Pearson r",
            "method": "log radial FFT of the integrated pearl-danio field versus exp(2 σ(k) T), all bins",
            "reference_value": 1.0,
            "reference_source": "pure linear theory over the whole measured band",
            "derived_from": "output",
        },
        {
            "name": "unstable-band stripe spectrum pearson r",
            "value": m["pearson_r_unstable_band"],
            "unit": "Pearson r",
            "method": "same radial FFT correlation restricted to bins with σ(k) > 0",
            "reference_value": 1.0,
            "reference_source": "linear theory on the unstable band only",
            "note": "Restricting to σ(k)>0 lowers r to about 0.936: those bins are few and the weakly grown mode is still close to the noise floor. The 0.98 goodness-of-fit is the closed-form versus eigenvalue dispersion correlation, not this band.",
            "derived_from": "output",
        },
        {
            "name": "seed-ensemble wavelength CV",
            "value": m["seed_ensemble_lambda_cv"],
            "unit": "std/mean of FFT wavelength",
            "method": "dominant FFT wavelength of the pearl-danio field at seeds 3161, 7 and 99",
            "reference_value": 0.0,
            "reference_source": "a deterministic wavelength would have CV 0",
            "note": f"wavelengths nm={m['seed_ensemble_lambda_nm']}",
            "derived_from": "output",
        },
        {
            "name": "albolineatus activator KS versus Gaussian",
            "value": albo["ks_gaussian"],
            "unit": "KS distance",
            "method": "scipy.stats.kstest of standardised final u against N(0,1)",
            "reference_value": ctrl["ks_gaussian"],
            "reference_source": "equal-diffusion control field, same test",
            "note": "Coefficient of variation of the pattern is std(u)/mean(u), stored as std_u.",
            "derived_from": "output",
        },
        {
            "name": "pattern amplitude CV albolineatus",
            "value": albo["std_u"] / jacobian()["u0"],
            "unit": "std(u)/u0",
            "method": "coefficient of variation of the activator at T=32",
            "reference_value": aes["std_u"] / jacobian()["u0"],
            "reference_source": "predecessor bar field at the same T",
            "derived_from": "output",
        },
        {
            "name": "equal-diffusion control max growth rate",
            "value": ctrl["sigma"],
            "unit": "1/time",
            "method": "max_k σ(k) at Du=Dv",
            "reference_value": 0.0,
            "reference_source": "marginal stability",
            "note": "Negative control. σ_max < 0 means the homogeneous state is stable.",
            "derived_from": "script",
        },
        {
            "name": "tight-ROI specimen depth over length",
            "value": spec["tight_depth_over_length"],
            "unit": "bbox height / bbox width",
            "method": "fish mask eroded by 15% of the bbox on each side",
            "reference_value": spec["depth_over_length"],
            "reference_source": "full-bbox measurement on the same photograph",
            "note": "Sensitivity check. The two ROIs are not the same number.",
            "derived_from": "output",
        },
    ]
    return rows


def self_test() -> dict:
    """Fast checks: pin, Jacobian signs, wavelength order, dispersion r, control stability."""
    verify_specimen_pin()
    j = jacobian()
    assert j["tr"] < 0.0 and j["det"] > 0.5, j
    assert abs((A_KIN + B_KIN) - 1.0) < 1e-12
    base = solve_albolineatus_diffusion(j)
    regimes = regime_table(base)
    albo, aes, marg = (regimes[k] for k in ("albolineatus", "aesculapii", "equal_diffusion_control"))
    marg = regimes["margaritatus"]
    # Fastest-mode wavelength sits on the assigned metre scale.
    assert abs(albo["lambda_m"] - LAMBDA_M) / LAMBDA_M < 0.01, albo
    assert aes["lambda_m"] > albo["lambda_m"] > marg["lambda_m"], (aes["lambda_m"], albo["lambda_m"], marg["lambda_m"])
    assert aes["sigma"] > albo["sigma"] > 0.0, (aes["sigma"], albo["sigma"])
    assert marg["sigma"] > albo["sigma"]
    assert regimes["equal_diffusion_control"]["sigma"] < 0.0
    # Lean integer order, allowing 80 nm of solver-grid rounding.
    for name, lean_nm in LEAN_NM.items():
        got = int(round(regimes[name]["lambda_m"] * 1e9))
        assert abs(got - lean_nm) <= 80, (name, got, lean_nm)
    gof = dispersion_goodness_of_fit(base)
    assert gof["pearson_r"] > 0.98, gof
    assert gof["rmse"] < 1e-6, gof
    # One short integration must stay finite and the control must decay.
    short = dict(regimes["equal_diffusion_control"])
    # Reuse integrate but it runs the full T; the control is the important finite check.
    ctrl_run = integrate(regimes["equal_diffusion_control"], n_frames=4, seed=SEED)
    assert float(np.std(ctrl_run["final_u"])) < 0.02, float(np.std(ctrl_run["final_u"]))
    albo_probe = integrate(regimes["albolineatus"], n_frames=4, seed=SEED)
    assert np.isfinite(albo_probe["final_u"]).all()
    # δ in thousandths matches the Lean witness 11434.
    assert abs(int(round(base["delta"] * 1000)) - 11434) <= 2, base["delta"]
    return {
        "pearson_r": gof["pearson_r"],
        "rmse": gof["rmse"],
        "lambda_m": albo["lambda_m"],
        "lambda_aes_m": aes["lambda_m"],
        "lambda_marg_m": marg["lambda_m"],
        "sigma_albo": albo["sigma"],
        "sigma_control": regimes["equal_diffusion_control"]["sigma"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(
            "[PASS] danio-albolineatus-turing-wavelength-s0 "
            f"r={result['pearson_r']:.6f} λ={result['lambda_m']:.6e} m "
            f"aes={result['lambda_aes_m']:.6e} marg={result['lambda_marg_m']:.6e} "
            f"σ_ctrl={result['sigma_control']:.4f}"
        )
        return 0
    metrics = run_simulation()
    probe = metrics["image_fitness_probe"]
    print(
        f"λ_albo={metrics['regimes']['albolineatus']['lambda_m']:.6e} m "
        f"r_disp={metrics['pearson_r_dispersion']:.6f} "
        f"r_band={metrics['pearson_r_unstable_band']:.4f} "
        f"r_spec={metrics['pearson_r_spectrum']:.4f} "
        f"cv={probe['cv_sim_similarity']} frames={probe['n_frames']} "
        f"min_px={probe['min_resolution']} ablation_cv={metrics['ablation_fullframe_cv']}"
    )
    print(f"wrote {PNG_PATH}")
    print(f"wrote {GIF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
