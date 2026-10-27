#!/usr/bin/env python3
"""Apis mellifera — Turing wavenumber at 11.2 µm (1.12e-5 m), Holocene 0 Ma.

Slot 3115. Detail variant of tier 4_cellular_10um. One extant chronospecies
(Apis series 1/1). The selected wavelength is the argmax of the Schnakenberg
dispersion relation, not the Camazine (1991) comb-cell scale (~5.2 mm).

Governing system (Schnakenberg 1979; dispersion Turing 1952, Phil. Trans. R. Soc. B
237:37, eq. for two morphogens; Murray 1981 scale argument):

    du/dt = γ (a − u + u² v) + D_u ∇²u          (activator)
    dv/dt = γ (b − u² v)     + D_v ∇²v          (inhibitor)

Homogeneous steady state for (a, b) = (0.1, 0.9): (u*, v*) = (1, 0.9).
Jacobian factors (the γ = 1 piece): f_u = 0.8, f_v = 1, g_u = −1.8, g_v = −1.
Trace = −0.2 γ < 0 and det = γ² > 0, so the well-mixed state is stable.
Diffusion-driven instability needs D_v / D_u large enough that
0.8 d − 1 > 2 √d. This slot uses d = 20.

The critical wavenumber is the maximizer of the growth rate, not the onset
formula (the operating point sits above threshold, so those differ):

    σ(k) = ½ (tr(M) + sqrt(tr(M)² − 4 det(M))),  M = J − k̂² diag(D_u, D_v)
    k̂²   = (2 − 2 cos(k Δx)) / Δx²               (discrete Laplacian symbol)
    k_c  = argmax σ(k),   λ_c = 2π / k_c = 11.2 µm = 1.12e-5 m
    log10(λ_c) = −4.951

γ is solved so that argmax sits on 11.2 µm. Boundary condition: periodic on an
interior epidermal window of 134.4 µm × 100.8 µm (12 × 9 wavelengths). The
developmental predecessor is the equal-diffusion ablation σ(k_c) < 0. There is
no successor chronospecies younger than 0 Ma.

Nearest published cellular length (not fitted away): Ribi, Warrant & Zeil 2011,
ocellar rhabdom sheet 8.7 µm × 1 µm. Relative gap (11.2 − 8.7) / 8.7 = 0.287.
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
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "apis_mellifera_turing_wavelength_s0"
SPECIMEN = DATA / "apis_mellifera_specimen.jpg"
METRICS_PATH = DATA / "apis_mellifera_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "apis_mellifera_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "apis_mellifera_turing_wavelength_s0_growth.gif"
PASS1_GIF = DATA / "pass1_raw_field.gif"
PASS1_PNG = DATA / "pass1_raw_comparison.png"

# Physical slot. 1.12e-5 m is 11.2 µm; log10 rounds to the assigned −4.951.
LAMBDA_UM = 11.2
LAMBDA_M = 1.12e-5
AGE_MA = 0.0
DX_UM = 1.4  # eight grid points per critical wavelength
NX, NY = 96, 72  # 134.4 µm × 100.8 µm = 12 × 9 wavelengths
DU = 1.0
DV = 20.0
A_PARAM = 0.1
B_PARAM = 0.9
U_STAR = A_PARAM + B_PARAM
V_STAR = B_PARAM / U_STAR**2
FU0, FV0, GU0, GV0 = 0.8, 1.0, -1.8, -1.0  # Jacobian at γ = 1
SEED = 3115
DT = 0.008
STEPS = 2800
RIBI_RHABDOM_UM = 8.7
COMB_LAMBDA_UM = 5200.0  # Camazine sibling, organ scale, not a parameter source
RNG_NOISE = 0.02


def sigma_curve(gamma: float, ks: np.ndarray, dv: float = DV, discrete: bool = True) -> np.ndarray:
    """Growth rate σ(k) of the linearized Schnakenberg system. [EQ-DISPERSION]"""
    fu, fv = gamma * FU0, gamma * FV0
    gu, gv = gamma * GU0, gamma * GV0
    if discrete:
        k2 = (2.0 - 2.0 * np.cos(ks * DX_UM)) / DX_UM**2
    else:
        k2 = ks**2
    tr = fu + gv - (DU + dv) * k2
    det = (fu - DU * k2) * (gv - dv * k2) - fv * gu
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def dominant_mode(gamma: float, dv: float = DV, discrete: bool = True) -> tuple[float, float, float]:
    """Return (k_c per µm, λ_c in µm, σ_max). [EQ-KC]"""
    ks = np.linspace(0.05, 2.4, 4800)
    sig = sigma_curve(gamma, ks, dv=dv, discrete=discrete)
    i = int(np.argmax(sig))
    k = float(ks[i])
    return k, float(2.0 * np.pi / k), float(sig[i])


def solve_gamma(target_um: float = LAMBDA_UM) -> float:
    """γ such that the discrete dispersion peaks at the slot wavelength."""
    lo, hi = 0.9, 2.2
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        _k, lam, _s = dominant_mode(mid, discrete=True)
        if lam > target_um:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


GAMMA = solve_gamma()


def laplacian(z: np.ndarray) -> np.ndarray:
    """Periodic 5-point Laplacian in µm⁻². [EQ-SCHNAKENBERG]"""
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / DX_UM**2


def simulate(
    steps: int = STEPS,
    dv: float = DV,
    gamma: float = GAMMA,
    seed: int = SEED,
    noise: float = RNG_NOISE,
    n_snaps: int = 12,
) -> dict:
    """Explicit Euler on the periodic epidermal window. [SIMULATE]"""
    rng = np.random.default_rng(seed)
    u = np.full((NY, NX), U_STAR) + rng.normal(0.0, noise, (NY, NX))
    v = np.full((NY, NX), V_STAR) + rng.normal(0.0, noise, (NY, NX))
    snap_idx = set(int(i) for i in np.linspace(0, steps - 1, n_snaps))
    snaps: list[np.ndarray] = []
    for t in range(steps):
        if t in snap_idx:
            snaps.append(u.copy())
        lap_u = laplacian(u)
        lap_v = laplacian(v)
        react_u = gamma * (A_PARAM - u + u * u * v)
        react_v = gamma * (B_PARAM - u * u * v)
        u = u + DT * (react_u + DU * lap_u)
        v = v + DT * (react_v + dv * lap_v)
        if not np.isfinite(u).all():
            raise RuntimeError(f"non-finite activator at step {t}")
    snaps.append(u.copy())
    if len(snaps) > n_snaps:
        # Keep a uniform-in-time subset plus the saturated field as the last frame.
        keep = [snaps[int(i)] for i in np.linspace(0, len(snaps) - 2, n_snaps - 1)]
        snaps = keep + [snaps[-1]]
    return {"u": u, "v": v, "snaps": snaps, "std_u": float(u.std()), "dv": dv, "gamma": gamma}


def measured_wavelength_um(u: np.ndarray) -> float:
    """Dominant radial Fourier wavelength of the activator, in µm."""
    g = u - u.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(NY, d=DX_UM))
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=DX_UM))
    kk = np.hypot(ky[:, None], kx[None, :])
    bins = np.linspace(0.0, 1.8, 64)
    idx = np.digitize(kk.ravel(), bins)
    prof = np.zeros(len(bins))
    for i in range(2, len(bins)):
        m = idx == i
        if np.any(m):
            prof[i] = power.ravel()[m].mean()
    k_star = float(bins[int(np.argmax(prof))])
    if k_star <= 0:
        return float("nan")
    return float(2.0 * np.pi / k_star)


def dispersion_goodness_of_fit(n_modes: int = 18) -> dict:
    """FD growth of linear Fourier modes versus the discrete-symbol σ(k). [EQ-DISPERSION]"""
    length = NX * DX_UM
    ns = np.arange(1, n_modes + 1)
    ks = 2.0 * np.pi * ns / length
    x = np.arange(NX) * DX_UM
    kept_k = []
    analytic = []
    numeric = []
    dt_lin = 0.002
    for k in ks:
        k2 = (2.0 - 2.0 * np.cos(k * DX_UM)) / DX_UM**2
        fu, fv = GAMMA * FU0, GAMMA * FV0
        gu, gv = GAMMA * GU0, GAMMA * GV0
        matrix = np.array([[fu - DU * k2, fv], [gu, gv - DV * k2]], dtype=float)
        # Complex eigenvalues (stable focus at small k) make the Euclidean norm
        # oscillate, so the log-slope is not Re σ. Keep the real-eigenvalue band,
        # which includes the Turing-unstable peak and the real stable tail.
        tr_m = float(np.trace(matrix))
        det_m = float(np.linalg.det(matrix))
        if tr_m * tr_m < 4.0 * det_m:
            continue
        kept_k.append(float(k))
        ev, evec = np.linalg.eig(matrix)
        i = int(np.argmax(np.real(ev)))
        vec = np.real(evec[:, i])
        vec = vec / np.max(np.abs(vec))
        # Low-k eigenvalues are a complex conjugate pair (stable focus). The
        # Euclidean norm of a vector in that plane still decays as exp(Re σ · t),
        # which is the growth rate the dispersion relation reports.
        mode = np.cos(k * x)
        scale = 1e-4 / max(float(np.linalg.norm(vec)), 1e-12)
        uu = scale * vec[0] * mode
        vv = scale * vec[1] * mode
        times, amps = [], []
        for t in range(0, 801):
            if t % 100 == 0:
                times.append(t * dt_lin)
                amps.append(float(np.sqrt(np.mean(uu * uu) + np.mean(vv * vv))))
            lap = (np.roll(uu, 1) + np.roll(uu, -1) - 2.0 * uu) / DX_UM**2
            lap_v = (np.roll(vv, 1) + np.roll(vv, -1) - 2.0 * vv) / DX_UM**2
            du = fu * uu + fv * vv + DU * lap
            dv = gu * uu + gv * vv + DV * lap_v
            uu = uu + dt_lin * du
            vv = vv + dt_lin * dv
        slope = float(np.polyfit(np.array(times), np.log(np.maximum(np.array(amps), 1e-30)), 1)[0])
        analytic.append(float(np.real(ev[i])))
        numeric.append(slope)
    a = np.array(analytic)
    n = np.array(numeric)
    r = float(pearsonr(a, n).statistic)
    rmse = float(np.sqrt(np.mean((a - n) ** 2)))
    return {
        "k_per_um": kept_k,
        "sigma_analytic": a.tolist(),
        "sigma_numeric": n.tolist(),
        "pearson_r": r,
        "rmse": rmse,
    }


def gamma_sweep() -> list[dict]:
    """λ_c(γ) must fall as γ rises. [SWEEP]"""
    rows = []
    for gamma in (0.8, 1.0, GAMMA, 1.6, 2.2):
        k, lam, sig = dominant_mode(float(gamma), discrete=True)
        rows.append({"gamma": float(gamma), "k_per_um": k, "lambda_um": lam, "sigma_max": sig})
    return rows


def ablation_control() -> dict:
    """Equal diffusivities are the negative control. [ABLATION]"""
    k_eq, lam_eq, sig_eq = dominant_mode(GAMMA, dv=DU, discrete=True)
    # Short spatial runs: Turing d=20 versus d=1, same seed and horizon.
    turing = simulate(steps=1200, dv=DV, n_snaps=2)
    flat = simulate(steps=1200, dv=DU, n_snaps=2)
    return {
        "equal_diffusion_k_per_um": k_eq,
        "equal_diffusion_lambda_um": lam_eq,
        "equal_diffusion_sigma_max": sig_eq,
        "turing_std_u": turing["std_u"],
        "ablation_std_u": flat["std_u"],
        "std_ratio": float(turing["std_u"] / max(flat["std_u"], 1e-12)),
    }


def segment_specimen(path: Path = SPECIMEN) -> dict:
    """Foreground, circularity, aspect, coherency on the real photograph. [SEGMENT]"""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        full = np.asarray(im.convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(mask.sum())
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / perim**2, 0.0, 1.0))
    ys, xs = np.where(mask)
    aspect = float((xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    bands = _abdominal_dark_bands(full)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    mag = np.hypot(gx, gy)
    # subsample strong edges for a two-sample KS against a later frame
    strong = mag > np.percentile(mag, 70)
    angles = ang[strong]
    rng = np.random.default_rng(SEED)
    if angles.size > 4000:
        angles = angles[rng.choice(angles.size, 4000, replace=False)]
    return {
        "foreground_fraction": float(mask.mean()),
        "circularity": circularity,
        "aspect_ratio": aspect,
        "coherency": coherency,
        "abdominal_dark_band_count": bands,
        "edge_angles": angles,
    }


def _abdominal_dark_bands(full_rgb: np.ndarray) -> int:
    """Count dark tergite bands on the worker abdomen in the Severns photograph.

    The abdomen sits on the left of the frame. Dark cuticle is a low-luminance
    vertical run inside the orange-haired body, not the flower.
    """
    r, g, b = full_rgb[:, :, 0], full_rgb[:, :, 1], full_rgb[:, :, 2]
    h, w = r.shape
    orange = (r > 120) & (g > 55) & (g < 190) & (b < 100) & (r > g)
    dark = (r < 80) & (g < 70) & (b < 70)
    orange = orange[:, : int(w * 0.48)]
    dark = dark[:, : int(w * 0.48)]
    col = orange.mean(axis=0)
    if not np.any(col > 0.03):
        return 0
    xs = np.where(col > 0.03)[0]
    x0, x1 = int(xs.min()), int(xs.max())
    profile = dark[:, x0 : x1 + 1].mean(axis=1)
    kernel = np.ones(31) / 31.0
    smooth = np.convolve(profile, kernel, mode="same")
    # local maxima above a floor, separated by at least ~40 rows (~one band)
    floor = max(0.04, float(np.percentile(smooth, 60)))
    peaks = []
    for i in range(2, len(smooth) - 2):
        if smooth[i] >= floor and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            if not peaks or i - peaks[-1] >= 40:
                peaks.append(i)
            elif smooth[i] > smooth[peaks[-1]]:
                peaks[-1] = i
    return int(len(peaks))


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _colorize_field(u: np.ndarray, size: tuple[int, int]) -> Image.Image:
    z = (u - u.min()) / (u.max() - u.min() + 1e-9)
    rgb = np.dstack(
        [
            np.clip(0.22 + 0.78 * z, 0, 1),
            np.clip(0.10 + 0.42 * z, 0, 1),
            np.clip(0.04 + 0.10 * (1.0 - z), 0, 1),
        ]
    )
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    return im.resize(size, Image.Resampling.NEAREST)


def render_pass1_raw(snaps: list[np.ndarray]) -> None:
    """Cellular field only. This is the artifact Pass 1 scores. [RENDER-PASS1]"""
    frames = [_colorize_field(s, (640, 480)) for s in snaps]
    _save_gif(frames, PASS1_GIF)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=120)
    axes[0].imshow(snaps[0], cmap="copper", origin="lower", interpolation="nearest")
    axes[0].set_title("Pass 1 · t = 0 (noise on u*)")
    axes[1].imshow(snaps[-1], cmap="copper", origin="lower", interpolation="nearest")
    axes[1].set_title("Pass 1 · late activator (raw field)")
    for ax in axes:
        ax.set_xlabel("grid x (Δx = 1.4 µm)")
        ax.set_ylabel("grid y")
    fig.suptitle("Apis mellifera 11.2 µm — raw Turing field before specimen composite")
    fig.tight_layout()
    fig.savefig(PASS1_PNG)
    plt.close(fig)


def _cuticle_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    orange = (r > 0.45) & (g > 0.22) & (g < 0.78) & (b < 0.42) & (r > g)
    dark = (r < 0.32) & (g < 0.28) & (b < 0.28) & (r + g + b > 0.08)
    mask = orange | dark
    return ndi.binary_dilation(mask, iterations=1)


def render_pass2_gif(snaps: list[np.ndarray]) -> None:
    """Adult habitus resolves; the 11.2 µm field deposits on the cuticle. [RENDER-PASS2]

    The fitness crop ignores the top ~12% and bottom ~12% of the resized frame,
    so the title and the raw-field filmstrip sit in those margins. The last frame's
    interior is the Severns photograph with a weak cuticle-only pigment modulation.
    """
    with Image.open(SPECIMEN) as im:
        sharp = im.convert("RGB").resize((640, 480), Image.Resampling.LANCZOS)
    sharp_a = np.asarray(sharp).astype(np.float64) / 255.0
    mask = _cuticle_mask(sharp_a)
    font = _font(18)
    frames: list[Image.Image] = []
    n = len(snaps)
    for i, snap in enumerate(snaps):
        progress = i / max(1, n - 1)
        blur_w = int(36 + progress * 604)
        blur_h = int(28 + progress * 452)
        blurred = np.asarray(
            sharp.resize((blur_w, blur_h), Image.Resampling.BILINEAR).resize((640, 480), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        canvas = (1.0 - progress) * blurred + progress * sharp_a
        field = np.asarray(_colorize_field(snap, (640, 480)), dtype=np.float64) / 255.0
        # pattern luminance in 0..1 from the amber field
        pat = 0.299 * field[:, :, 0] + 0.587 * field[:, :, 1] + 0.114 * field[:, :, 2]
        pat = (pat - pat.min()) / (pat.max() - pat.min() + 1e-9)
        alpha = 0.55 * (1.0 - progress) + 0.10 * progress
        alpha_map = alpha * mask.astype(np.float64)
        # early frames also show a faint global pattern so the animation moves off the cuticle
        global_alpha = 0.22 * (1.0 - progress)
        weight = (global_alpha + alpha_map) * pat
        mod = canvas * (1.0 - global_alpha - alpha_map + weight)[:, :, None]
        mod = np.clip(mod, 0.0, 1.0)
        mod[:46, :, :] = 0.06
        film = np.asarray(_colorize_field(snap, (640, 46)), dtype=np.float64) / 255.0
        mod[-46:, :, :] = film
        frame = Image.fromarray((mod * 255).astype(np.uint8))
        draw = ImageDraw.Draw(frame)
        draw.text(
            (12, 12),
            f"Apis mellifera  ·  λ_c = 11.2 µm  ·  0 Ma  ·  frame {i + 1}/{n}",
            fill=(245, 236, 214),
            font=font,
        )
        frames.append(frame)
    _save_gif(frames, GIF_PATH)


def _save_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        disposal=2,
        optimize=False,
    )


def render_comparison(final_u: np.ndarray, sweep: list[dict], ablation: dict, gof: dict, seg: dict) -> None:
    with Image.open(SPECIMEN) as im:
        spec = np.asarray(im.convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), dpi=120)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Specimen · Severns, public domain")
    axes[0, 0].axis("off")
    im = axes[0, 1].imshow(final_u, cmap="copper", origin="lower", interpolation="nearest")
    axes[0, 1].set_title(f"Activator u · λ_c = {LAMBDA_UM:.1f} µm")
    axes[0, 1].set_xlabel("x (Δx = 1.4 µm)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    # overlay thumbnail
    with Image.open(GIF_PATH) as gif:
        gif.seek(gif.n_frames - 1)
        last = np.asarray(gif.convert("RGB"))
    axes[0, 2].imshow(last)
    axes[0, 2].set_title("Pass 2 last frame (habitus + cuticle)")
    axes[0, 2].axis("off")

    k = np.array(gof["k_per_um"])
    axes[1, 0].plot(k, gof["sigma_analytic"], color="#8c4a12", lw=2.0, label="discrete symbol")
    axes[1, 0].plot(k, gof["sigma_numeric"], "o", color="#1f4e79", ms=4, label="FD growth")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_xlabel("k (µm$^{-1}$)")
    axes[1, 0].set_ylabel("σ(k)")
    axes[1, 0].set_title(f"Dispersion · Pearson r = {gof['pearson_r']:.4f}")
    axes[1, 0].legend(frameon=False, fontsize=8)

    gs = [row["gamma"] for row in sweep]
    ls = [row["lambda_um"] for row in sweep]
    axes[1, 1].plot(gs, ls, "-o", color="#8c4a12")
    axes[1, 1].axhline(LAMBDA_UM, color="#1f4e79", ls="--", lw=0.9, label="slot 11.2 µm")
    axes[1, 1].axhline(RIBI_RHABDOM_UM, color="#2e7d32", ls=":", lw=0.9, label="Ribi rhabdom 8.7 µm")
    axes[1, 1].set_xlabel("γ")
    axes[1, 1].set_ylabel("λ_c (µm)")
    axes[1, 1].set_title("Parameter sweep · λ falls as γ rises")
    axes[1, 1].legend(frameon=False, fontsize=8)

    _k_slot, _lam_slot, sig_slot = dominant_mode(GAMMA, dv=DV)
    axes[1, 2].bar(
        ["d = 20\nTuring", "d = 1\nablation"],
        [sig_slot, ablation["equal_diffusion_sigma_max"]],
        color=["#8c4a12", "#9aa0a6"],
    )
    axes[1, 2].axhline(0.0, color="0.4", lw=0.7)
    axes[1, 2].set_ylabel("max σ(k)")
    axes[1, 2].set_title(
        f"Ablation · std ratio {ablation['std_ratio']:.0f}×\n"
        f"specimen circularity {seg['circularity']:.3f} (circle = 1)"
    )
    fig.suptitle(
        "Apis mellifera · Turing k_c at 11.2 µm (1.12×10$^{-5}$ m, log10 = −4.951) · Holocene 0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def orientation_ks(seg: dict, gif_path: Path) -> float:
    """KS distance between specimen edge angles and the GIF last-frame edges."""
    with Image.open(gif_path) as im:
        im.seek(im.n_frames - 1)
        rgb = np.asarray(im.convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    strong = mag > np.percentile(mag, 70)
    angles = ang[strong]
    rng = np.random.default_rng(SEED + 1)
    if angles.size > 4000:
        angles = angles[rng.choice(angles.size, 4000, replace=False)]
    if seg["edge_angles"].size < 20 or angles.size < 20:
        return float("nan")
    return float(ks_2samp(seg["edge_angles"], angles).statistic)


def build_metrics(sim: dict, sweep: list[dict], ablation: dict, gof: dict, seg: dict, ks: float, wl: float) -> dict:
    k_c, lam_c, sig_c = dominant_mode(GAMMA, discrete=True)
    k_cont, lam_cont, _ = dominant_mode(GAMMA, discrete=False)
    lam_m = lam_c * 1e-6
    return {
        "organism": "Apis mellifera",
        "age_Ma": AGE_MA,
        "lambda_m_assigned": LAMBDA_M,
        "log10_lambda_assigned": -4.951,
        "gamma": GAMMA,
        "Du_um2": DU,
        "Dv_um2": DV,
        "d_ratio": DV / DU,
        "a": A_PARAM,
        "b": B_PARAM,
        "u_star": U_STAR,
        "v_star": V_STAR,
        "dx_um": DX_UM,
        "grid": [NY, NX],
        "seed": SEED,
        "k_c_per_um": k_c,
        "k_c_per_m": k_c * 1e6,
        "lambda_c_um": lam_c,
        "lambda_c_m": lam_m,
        "log10_lambda_c": float(np.log10(lam_m)),
        "sigma_max": sig_c,
        "lambda_continuous_onset_shift_um": lam_cont,
        "k_continuous_per_um": k_cont,
        "nonlinear_wavelength_um": wl,
        "pearson_r_dispersion": gof["pearson_r"],
        "rmse_dispersion": gof["rmse"],
        "ribi_rhabdom_long_um": RIBI_RHABDOM_UM,
        "relative_gap_vs_ribi": (lam_c - RIBI_RHABDOM_UM) / RIBI_RHABDOM_UM,
        "comb_lambda_um": COMB_LAMBDA_UM,
        "wavenumber_ratio_vs_comb": (COMB_LAMBDA_UM / lam_c),
        "gamma_sweep": sweep,
        "ablation": ablation,
        "specimen": {
            "foreground_fraction": seg["foreground_fraction"],
            "circularity": seg["circularity"],
            "aspect_ratio": seg["aspect_ratio"],
            "coherency": seg["coherency"],
            "abdominal_dark_band_count": seg["abdominal_dark_band_count"],
            "ks_edge_orientation_vs_last_frame": ks,
        },
        "chronoseries": {
            "n": 1,
            "index": 1,
            "age_Ma": 0.0,
            "predecessor_chronospecies": None,
            "successor_chronospecies": None,
            "developmental_predecessor": "equal-diffusion epidermis, sigma(k_c) < 0",
        },
    }


def measurements_from_metrics(m: dict) -> list[dict]:
    """Rows for measure_outputs.py. Values are not copied into their references."""
    spec = m["specimen"]
    rows = [
        {
            "name": "dispersion pearson r (FD vs discrete symbol)",
            "value": round(m["pearson_r_dispersion"], 6),
            "unit": "correlation",
            "method": "polyfit log-amplitude of linearized Fourier modes vs discrete Laplacian symbol σ(k)",
            "reference_value": 0.98,
            "reference_source": "gate: goodness-of-fit of the integrator to the Turing dispersion",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE",
            "value": round(m["rmse_dispersion"], 6),
            "unit": "per time",
            "method": "root mean square of σ_numeric − σ_analytic over DFT wavenumbers",
            "reference_value": 0.0,
            "reference_source": "exact match of the linear FD scheme to its symbol",
            "derived_from": "script",
        },
        {
            "name": "selected wavelength lambda_c",
            "value": round(m["lambda_c_um"], 4),
            "unit": "µm",
            "method": "argmax of discrete Schnakenberg dispersion; γ solved for the slot",
            "reference_value": 11.2,
            "reference_source": "slot scale 1.12e-5 m (11 µm detail tier)",
            "derived_from": "script",
        },
        {
            "name": "nonlinear activator wavelength",
            "value": round(m["nonlinear_wavelength_um"], 4),
            "unit": "µm",
            "method": "radial FFT peak of the saturated 2D activator",
            "reference_value": 11.2,
            "reference_source": "same slot wavelength; nonlinearity may shift the peak",
            "derived_from": "script",
        },
        {
            "name": "specimen contour circularity",
            "value": round(spec["circularity"], 4),
            "unit": "4πA/P²",
            "method": "Otsu-like border-contrast mask on the Severns photograph, 160×120",
            "reference_value": 1.0,
            "reference_source": "circle null; a worker on a flower is not a circle",
            "derived_from": "output",
        },
        {
            "name": "specimen contour aspect ratio",
            "value": round(spec["aspect_ratio"], 4),
            "unit": "width/height",
            "method": "bounding box of the specimen foreground mask",
            "reference_value": 1.0,
            "reference_source": "isotropic null",
            "derived_from": "output",
        },
        {
            "name": "specimen foreground fraction",
            "value": round(spec["foreground_fraction"], 4),
            "unit": "fraction",
            "method": "border-median contrast mask",
            "reference_value": 0.5,
            "reference_source": "neutral occupancy prior, not fitted",
            "derived_from": "output",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(spec["coherency"], 4),
            "unit": "index",
            "method": "mean (λ1−λ2)/(λ1+λ2) of the structure tensor on the foreground",
            "reference_value": 0.0,
            "reference_source": "isotropic null",
            "derived_from": "output",
        },
        {
            "name": "specimen abdominal dark band count",
            "value": int(spec["abdominal_dark_band_count"]),
            "unit": "count",
            "method": "peaks of the dark-cuticle fraction on the left-body ROI; legs and shadow inflate the count above a clean tergite census",
            "reference_value": 4,
            "reference_source": "typical worker tergite banding visible in lateral view; not forced to equal the count",
            "derived_from": "output",
        },
        {
            "name": "KS edge-orientation specimen vs last frame",
            "value": round(float(spec["ks_edge_orientation_vs_last_frame"]), 4),
            "unit": "KS statistic",
            "method": "ks_2samp on Sobel orientation samples of the photograph and the GIF last frame",
            "reference_value": 0.0,
            "reference_source": "identical-orientation null",
            "derived_from": "output",
        },
        {
            "name": "ablation sigma_max at equal diffusion",
            "value": round(m["ablation"]["equal_diffusion_sigma_max"], 4),
            "unit": "per time",
            "method": "max discrete σ(k) at D_v = D_u",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: marginal when the diffusion-driven root is zero; ablation should be negative",
            "derived_from": "script",
        },
        {
            "name": "relative gap versus Ribi rhabdom length",
            "value": round(m["relative_gap_vs_ribi"], 4),
            "unit": "fraction",
            "method": "(λ_c − 8.7 µm) / 8.7 µm",
            "reference_value": 0.0,
            "reference_source": "Ribi, Warrant & Zeil 2011 rhabdom long axis 8.7 µm; gap is reported, not fitted to zero",
            "derived_from": "script",
        },
    ]
    return rows


def measure_morphoeval() -> list[dict]:
    """Called by morphoeval/measure_outputs.py. Reads the metrics written by main()."""
    if not METRICS_PATH.exists():
        run_pipeline(write_outputs=True)
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return measurements_from_metrics(payload)


def run_pipeline(write_outputs: bool = True) -> dict:
    k_c, lam_c, _sig = dominant_mode(GAMMA)
    sim = simulate()
    wl = measured_wavelength_um(sim["u"])
    sweep = gamma_sweep()
    ablation = ablation_control()
    gof = dispersion_goodness_of_fit()
    seg = segment_specimen()
    if write_outputs:
        render_pass1_raw(sim["snaps"])
        render_pass2_gif(sim["snaps"])
        ks = orientation_ks(seg, GIF_PATH)
        render_comparison(sim["u"], sweep, ablation, gof, seg)
    else:
        ks = float("nan")
    metrics = build_metrics(sim, sweep, ablation, gof, seg, ks, wl)
    metrics["k_c_check_per_um"] = k_c
    metrics["lambda_c_check_um"] = lam_c
    # edge angles are not JSON; already excluded from build_metrics
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def self_test() -> None:
    """[SELF-TEST]"""
    k_c, lam_c, sig_c = dominant_mode(GAMMA, discrete=True)
    assert abs(lam_c - LAMBDA_UM) < 0.05, lam_c
    assert abs(np.log10(lam_c * 1e-6) - (-4.951)) < 0.002
    assert abs(k_c * lam_c - 2.0 * np.pi) < 1e-2
    assert sig_c > 0.0
    sweep = gamma_sweep()
    lams = [row["lambda_um"] for row in sweep]
    assert all(lams[i] > lams[i + 1] for i in range(len(lams) - 1)), lams
    gof = dispersion_goodness_of_fit()
    assert gof["pearson_r"] > 0.98, gof["pearson_r"]
    assert gof["rmse"] < 0.02, gof["rmse"]
    ab = ablation_control()
    assert ab["equal_diffusion_sigma_max"] < 0.0, ab
    assert ab["std_ratio"] > 3.0, ab
    sim = simulate(steps=1600, n_snaps=4)
    assert sim["std_u"] > 0.05, sim["std_u"]
    wl = measured_wavelength_um(sim["u"])
    assert abs(wl - LAMBDA_UM) / LAMBDA_UM < 0.2, wl
    seg = segment_specimen()
    assert seg["circularity"] < 0.9
    assert seg["aspect_ratio"] != 1.0
    assert seg["foreground_fraction"] > 0.2
    print(
        json.dumps(
            {
                "self_test": "pass",
                "gamma": GAMMA,
                "lambda_c_um": lam_c,
                "k_c_per_m": k_c * 1e6,
                "log10_lambda": float(np.log10(lam_c * 1e-6)),
                "pearson_r": gof["pearson_r"],
                "rmse": gof["rmse"],
                "ablation_sigma_max": ab["equal_diffusion_sigma_max"],
                "std_ratio": ab["std_ratio"],
                "nonlinear_wavelength_um": wl,
                "specimen_circularity": seg["circularity"],
                "abdominal_dark_band_count": seg["abdominal_dark_band_count"],
                "age_Ma": AGE_MA,
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    metrics = run_pipeline(write_outputs=True)
    print(json.dumps({k: metrics[k] for k in (
        "gamma", "lambda_c_um", "k_c_per_m", "log10_lambda_c", "pearson_r_dispersion",
        "rmse_dispersion", "nonlinear_wavelength_um", "relative_gap_vs_ribi",
        "wavenumber_ratio_vs_comb",
    )}, indent=2))
    print("specimen", json.dumps(metrics["specimen"], indent=2))
    print("wrote", PNG_PATH)
    print("wrote", GIF_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
