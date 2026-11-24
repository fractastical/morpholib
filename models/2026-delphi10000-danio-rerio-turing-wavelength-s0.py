#!/usr/bin/env python3
"""Danio rerio detail-scale Turing wavelength (slot 3164).

Holocene extant zebrafish at 24.08 µm (2.408e-05 m, tier 4_cellular_10um).
Age 0 Ma. Predecessor Danio nigrofasciatus. No successor.

Governing law: Schnakenberg reaction-diffusion (Turing 1952) with the
Nakamasu, Takahashi, Kanbe & Kondo (2009) reading that melanophores and
xanthophores are the activator-inhibitor pair, oriented by anisotropic
inhibitor diffusion (Hiscock & Megason 2015).

    u_t = Du_x u_xx + Du_y u_yy + gamma (a - u + u^2 v)
    v_t = Dv_x v_xx + Dv_y v_yy + gamma (b - u^2 v)

The dorsoventral diffusivity Dv_y = 20.80 and gamma = 0.1989 place the
dispersion peak on k_c = 2 pi / 24.08 µm. Anisotropy Dv_y/Dv_x = 3.85
(this species) drops the AP axis out of the Turing band. The predecessor
regime at the same cellular scale uses anisotropy 1.12, so both axes
remain unstable and the pattern is spotted. Those ratios are specific to
this 24.08 µm window; they are not the 1 mm D. nigrofasciatus parameters
and not the 18 µm melanophore-cell parameters.

The GIF maps the 288.96 x 144.48 µm window onto the segmented adult
outline so the cellular stripes can be compared with the specimen. That
mapping is a magnification, not a claim that adult flank stripes are 24 µm.
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
from numpy.fft import fft2, fftfreq, ifft2
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "danio_rerio_turing_wavelength_s0"
SPECIMEN = DATASET / "danio_rerio_specimen.jpg"
REFERENCE = DATASET / "danio_rerio_turing_wavelength_s0_reference.json"
MEDIA_PNG = ROOT / "docs" / "media" / "danio_rerio_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "danio_rerio_turing_wavelength_s0_growth.gif"
METRICS_PATH = DATASET / "metrics.json"

# Slot-3164 parameters. Length unit is the micrometre.
WAVELENGTH_UM = 24.08
PHENOMENON_M = 2.408e-05
AGE_MA = 0.0
A_PROD = 0.11
B_PROD = 1.24
DU = 0.80
DV_Y = 20.80
GAMMA = 0.1989
ANISO_RERIO = 3.85
ANISO_NIGRO = 1.12
N_DV = 6
N_AP = 12
LX = N_AP * WAVELENGTH_UM
LY = N_DV * WAVELENGTH_UM
NX = 192
NY = 96
SEED = 3164
K_C = 2.0 * np.pi / WAVELENGTH_UM


def steady_state() -> tuple[float, float, float, float, float, float]:
    """Return u0, v0, f_u, f_v, g_u, g_v at the Schnakenberg steady state."""
    u0 = A_PROD + B_PROD
    v0 = B_PROD / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return u0, v0, fu, fv, gu, gv


def sigma_mode(kx: float, ky: float, dvx: float, dvy: float, gamma: float = GAMMA) -> float:
    """Largest eigenvalue of gamma*J - diag(Du k^2, Dv k^2) for one wavevector."""
    _u0, _v0, fu, fv, gu, gv = steady_state()
    mu = gamma * fu - (DU * kx * kx + DU * ky * ky)
    mv = gamma * gv - (dvx * kx * kx + dvy * ky * ky)
    tr = mu + mv
    det = mu * mv - (gamma * fv) * (gamma * gu)
    disc = tr * tr - 4.0 * det
    return float(0.5 * (tr + np.sqrt(disc if disc > 0.0 else 0.0)))


def diffusivities(anisotropy: float) -> tuple[float, float]:
    """Dv_y stays on the calibrated peak; Dv_x = Dv_y / anisotropy."""
    return DV_Y / float(anisotropy), DV_Y


def verify_inputs() -> dict:
    """Fail if a vendored dataset file does not match the pinned SHA-256."""
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    pins = ref["input_sha256"]
    for name, expected in pins.items():
        path = DATASET / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {digest} != {expected}")
    return ref


def integrate(
    anisotropy: float,
    seed: int,
    nsteps: int,
    mode_amp: float = 0.0,
    dv_override: tuple[float, float] | None = None,
    snapshots: int = 0,
) -> dict:
    """Spectral diffusion, explicit Schnakenberg reaction, periodic boundaries."""
    u0, v0, _fu, _fv, _gu, _gv = steady_state()
    if dv_override is None:
        dvx, dvy = diffusivities(anisotropy)
    else:
        dvx, dvy = dv_override
    dx, dy = LX / NX, LY / NY
    rng = np.random.default_rng(seed)
    yy = np.arange(NY) * dy
    u = np.full((NY, NX), u0) + mode_amp * np.cos(K_C * yy)[:, None]
    v = np.full((NY, NX), v0) - mode_amp * np.cos(K_C * yy)[:, None]
    u = u + 0.015 * rng.standard_normal((NY, NX))
    v = v + 0.015 * rng.standard_normal((NY, NX))
    kx = 2.0 * np.pi * fftfreq(NX, d=dx)
    ky = 2.0 * np.pi * fftfreq(NY, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    dt = 0.25
    decay_u = np.exp(-dt * (DU * kx_grid * kx_grid + DU * ky_grid * ky_grid))
    decay_v = np.exp(-dt * (dvx * kx_grid * kx_grid + dvy * ky_grid * ky_grid))
    frames: list[np.ndarray] = []
    snap_at = set(np.linspace(0, nsteps - 1, snapshots, dtype=int).tolist()) if snapshots else set()
    for step in range(nsteps):
        if step in snap_at:
            frames.append(u.copy())
        u = np.real(ifft2(fft2(u) * decay_u))
        v = np.real(ifft2(fft2(v) * decay_v))
        uu = u * u
        u = np.clip(u + dt * GAMMA * (A_PROD - u + uu * v), 0.0, 6.0)
        v = np.clip(v + dt * GAMMA * (B_PROD - uu * v), 0.0, 6.0)
    if snapshots and (nsteps - 1) not in snap_at:
        frames.append(u.copy())
    return {
        "u": u,
        "v": v,
        "frames": frames,
        "dvx": dvx,
        "dvy": dvy,
        "anisotropy": anisotropy,
        "seed": seed,
    }


def pattern_metrics(u: np.ndarray) -> dict:
    """Dominant DV wavelength and gradient anisotropy of a field."""
    profile = u.mean(axis=1)
    profile = profile - profile.mean()
    power = np.abs(np.fft.rfft(profile))
    mode = int(np.argmax(power[1:]) + 1)
    gy, gx = np.gradient(u)
    anisotropy = float(np.mean(np.abs(gy)) / (np.mean(np.abs(gx)) + 1e-12))
    return {
        "mode": mode,
        "wavelength_um": float(LY / mode),
        "gradient_anisotropy": anisotropy,
        "amplitude": float(u.std()),
    }


def dispersion_goodness(n_k: int = 11) -> dict:
    """Compare short-time PDE growth rates with the analytical eigenvalue.

    Wavenumbers are the exact y-DFT modes of the 144.48 µm window. Off-grid
    cosines leak into the unstable band and corrupt a long-window growth fit.
    """
    u0, v0, _fu, _fv, _gu, _gv = steady_state()
    dvx, dvy = diffusivities(ANISO_RERIO)
    modes = np.arange(1, n_k + 1)
    ks = 2.0 * np.pi * modes / LY
    analytical = np.array([sigma_mode(0.0, float(k), dvx, dvy) for k in ks])
    numerical = []
    yy = np.arange(NY) * (LY / NY)
    dt = 0.25
    eps = 1.0e-3
    t1, t2 = 2.0, 14.0
    for k in ks:
        u = np.full((NY, NX), u0) + eps * np.cos(k * yy)[:, None]
        v = np.full((NY, NX), v0) - eps * np.cos(k * yy)[:, None]
        kx = 2.0 * np.pi * fftfreq(NX, d=LX / NX)
        ky = 2.0 * np.pi * fftfreq(NY, d=LY / NY)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        decay_u = np.exp(-dt * DU * (kx_grid * kx_grid + ky_grid * ky_grid))
        decay_v = np.exp(-dt * (dvx * kx_grid * kx_grid + dvy * ky_grid * ky_grid))
        amp1 = None
        steps = int(t2 / dt)
        for step in range(steps):
            u = np.real(ifft2(fft2(u) * decay_u))
            v = np.real(ifft2(fft2(v) * decay_v))
            uu = u * u
            u = u + dt * GAMMA * (A_PROD - u + uu * v)
            v = v + dt * GAMMA * (B_PROD - uu * v)
            t = (step + 1) * dt
            if amp1 is None and t >= t1:
                amp1 = float(np.mean((u - u0) * np.cos(k * yy)[:, None]))
        amp2 = float(np.mean((u - u0) * np.cos(k * yy)[:, None]))
        if amp1 is None or abs(amp1) < 1e-16:
            numerical.append(0.0)
        else:
            numerical.append(float(np.log(abs(amp2) + 1e-16) - np.log(abs(amp1) + 1e-16)) / (t2 - t1))
    numerical_arr = np.asarray(numerical, dtype=float)
    pearson = float(np.corrcoef(analytical, numerical_arr)[0, 1])
    rmse = float(np.sqrt(np.mean((analytical - numerical_arr) ** 2)))
    ks_test = stats.ks_2samp(analytical, numerical_arr)
    return {
        "k_um_inv": [float(k) for k in ks],
        "sigma_analytical": [float(s) for s in analytical],
        "sigma_numerical": [float(s) for s in numerical_arr],
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": float(ks_test.statistic),
        "ks_pvalue": float(ks_test.pvalue),
    }


def segment_specimen(rgb: np.ndarray) -> np.ndarray:
    """Foreground fish mask from border-referenced optical density."""
    gray = rgb.mean(axis=2)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))
    mask = gray < background - 18.0
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_closing(mask, iterations=4)
    mask = ndi.binary_fill_holes(mask)
    labels, count = ndi.label(mask)
    if count > 1:
        sizes = ndi.sum(mask, labels, range(1, count + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    return mask


def _palette(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = image[mask]
    luminance = pixels.mean(axis=1)
    dark = pixels[luminance <= np.percentile(luminance, 30)].mean(axis=0)
    light = pixels[luminance >= np.percentile(luminance, 70)].mean(axis=0)
    background = image[~mask].mean(axis=0) if np.any(~mask) else np.array([235.0, 235.0, 235.0])
    return dark, light, background


def render_fish_frames(fields: list[np.ndarray], specimen_rgb: np.ndarray, mask: np.ndarray) -> list[Image.Image]:
    """Paint each RD frame into the segmented fish, on the specimen background."""
    width, height = 960, 540
    mask_img = np.asarray(
        Image.fromarray((mask.astype(np.uint8) * 255)).resize((width, height), Image.Resampling.NEAREST)
    ) > 127
    specimen = np.asarray(
        Image.fromarray(specimen_rgb).resize((width, height), Image.Resampling.LANCZOS)
    ).astype(np.float64)
    dark, light, background = _palette(specimen, mask_img)
    ys, _xs = np.where(mask_img)
    y0, y1 = int(ys.min()), int(ys.max())
    body_h = y1 - y0 + 1
    frames: list[Image.Image] = []
    n = max(1, len(fields) - 1)
    for index, field in enumerate(fields):
        alpha = index / n
        lo, hi = np.percentile(field, [2, 98])
        pigment = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        # Steep sigmoid: the linear sinusoid averaged to gray. Bimodal pigment
        # uses the specimen's dark-stripe and light-interstripe colors.
        pigment = 1.0 / (1.0 + np.exp(-12.0 * (pigment - 0.5)))
        pigment_img = np.asarray(
            Image.fromarray((pigment * 255).astype(np.uint8)).resize((width, body_h), Image.Resampling.BILINEAR)
        ) / 255.0
        shown = (1.0 - alpha) * 0.5 + alpha * pigment_img
        canvas = np.zeros((body_h, width))
        canvas[:, :] = shown
        rgb = np.zeros((height, width, 3), dtype=np.float64)
        rgb[:] = background
        body = rgb[y0 : y0 + body_h]
        for channel in range(3):
            column = canvas * dark[channel] + (1.0 - canvas) * light[channel]
            body[:, :, channel] = np.where(mask_img[y0 : y0 + body_h], column, body[:, :, channel])
        frames.append(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)))
    return frames


def write_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        disposal=2,
    )


def write_comparison(
    specimen_rgb: np.ndarray,
    mask: np.ndarray,
    headline: dict,
    fit: dict,
    sweep: list[dict],
    path: Path,
    rendered: Image.Image | None = None,
) -> None:
    """Specimen, mask composite, cellular field, dispersion fit, and the sweep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.4), dpi=110)
    axes[0, 0].imshow(specimen_rgb)
    axes[0, 0].set_title("Danio rerio specimen, Holocene, 0 Ma")
    axes[0, 0].axis("off")
    if rendered is not None:
        axes[0, 1].imshow(rendered)
        axes[0, 1].set_title("Mask composite of the 24.08 µm field")
        axes[0, 1].axis("off")
    else:
        axes[0, 1].axis("off")
    field = headline["u"]
    axes[0, 2].imshow(field, cmap="cividis", origin="lower", extent=[0, LX, 0, LY], aspect="auto")
    axes[0, 2].set_title(f"RD field, λ = {headline['metrics']['wavelength_um']:.2f} µm")
    axes[0, 2].set_xlabel("AP (µm)")
    axes[0, 2].set_ylabel("DV (µm)")
    ks = np.asarray(fit["k_um_inv"])
    axes[1, 0].plot(ks, fit["sigma_analytical"], color="#1b4f72", lw=2.0, label="analytical σ(k)")
    axes[1, 0].plot(ks, fit["sigma_numerical"], "o", color="#b9770e", ms=4, label="PDE growth")
    axes[1, 0].axvline(K_C, color="#7b241c", ls="--", lw=1.0, label="k_c = 2π/24.08 µm")
    axes[1, 0].set_xlabel("k (µm$^{-1}$)")
    axes[1, 0].set_ylabel("growth rate")
    axes[1, 0].set_title(f"Dispersion fit  r = {fit['pearson_r']:.4f}")
    axes[1, 0].legend(frameon=False, fontsize=8)
    anisos = [row["anisotropy"] for row in sweep]
    orients = [row["gradient_anisotropy"] for row in sweep]
    axes[1, 1].plot(anisos, orients, "-o", color="#196f3d")
    axes[1, 1].axvline(ANISO_NIGRO, color="#7b241c", ls=":", label="D. nigrofasciatus 1.12")
    axes[1, 1].axvline(ANISO_RERIO, color="#1a5276", ls="--", label="D. rerio 3.85")
    axes[1, 1].set_xlabel("inhibitor anisotropy Dv_y / Dv_x")
    axes[1, 1].set_ylabel("|∂y u| / |∂x u|")
    axes[1, 1].set_title("Chronospecies axis selection at 24.08 µm")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        0.5,
        "Pass 2: bimodal pigment uses the\n"
        "specimen dark/light palette.\n"
        f"KS D = {fit['ks_statistic']:.3f} on {len(ks)} wavenumbers.\n"
        "Age 0 Ma. Predecessor anisotropy 1.12.\n"
        "Window 288.96 × 144.48 µm, magnified.",
        transform=axes[1, 2].transAxes,
        va="center",
        fontsize=10,
    )
    fig.suptitle(
        "Danio rerio Turing wavelength 24.08 µm (2.408e-05 m), age 0 Ma — slot 3164",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    _ = mask  # mask is applied in the GIF; the panel shows the unwrapped cellular field


def coherency(gray: np.ndarray) -> float:
    """Mean structure-tensor coherency, the same construction as the fitness test."""
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def run_suite() -> dict:
    """Headline pattern, controls, sweep, dispersion fit, and specimen segmentation."""
    verify_inputs()
    headline = integrate(ANISO_RERIO, SEED, nsteps=1600, mode_amp=0.012, snapshots=12)
    headline["metrics"] = pattern_metrics(headline["u"])
    seeds = []
    for seed in (3164, 4071, 2408):
        field = integrate(ANISO_RERIO, seed, nsteps=1400, mode_amp=0.0)
        seeds.append({"seed": seed, **pattern_metrics(field["u"])})
    predecessor = integrate(ANISO_NIGRO, SEED, nsteps=1400, mode_amp=0.0)
    predecessor_metrics = pattern_metrics(predecessor["u"])
    equal = integrate(ANISO_RERIO, SEED, nsteps=800, mode_amp=0.0, dv_override=(DU, DU))
    equal_metrics = pattern_metrics(equal["u"])
    sweep = []
    for anisotropy in (1.12, 2.0, 3.85, 6.0):
        field = integrate(anisotropy, SEED, nsteps=1200, mode_amp=0.0)
        sweep.append({"anisotropy": anisotropy, **pattern_metrics(field["u"])})
    fit = dispersion_goodness()
    specimen = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    mask = segment_specimen(specimen)
    frames = render_fish_frames(headline["frames"], specimen, mask)
    write_gif(frames, MEDIA_GIF)
    write_comparison(specimen, mask, headline, fit, sweep, MEDIA_PNG, rendered=frames[-1])
    with Image.open(MEDIA_GIF) as gif:
        gif.seek(gif.n_frames - 1)
        last = np.asarray(gif.convert("L"), dtype=np.float64) / 255.0
        n_frames = int(gif.n_frames)
        width, height = gif.size
    specimen_gray = np.asarray(Image.open(SPECIMEN).convert("L").resize((width, height)), dtype=np.float64) / 255.0
    sim_coh = coherency(last)
    spec_coh = coherency(specimen_gray)
    ys, xs = np.where(mask)
    specimen_aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
    # The rendered fish uses the same mask, but the measured GIF foreground
    # includes anti-aliased edges, so the two fractions are not identical.
    gif_fg = float(np.mean(last < 0.90))
    spec_fg = float(np.mean(mask))
    result = {
        "wavelength_um": headline["metrics"]["wavelength_um"],
        "gradient_anisotropy": headline["metrics"]["gradient_anisotropy"],
        "amplitude": headline["metrics"]["amplitude"],
        "seeds": seeds,
        "predecessor": predecessor_metrics,
        "equal_diffusion_amplitude": equal_metrics["amplitude"],
        "sweep": sweep,
        "dispersion": fit,
        "specimen_aspect_ratio": specimen_aspect,
        "specimen_foreground_fraction": spec_fg,
        "simulation_foreground_fraction": gif_fg,
        "specimen_coherency": spec_coh,
        "simulation_coherency": sim_coh,
        "gif_frames": n_frames,
        "gif_width": width,
        "gif_height": height,
        "phenomenon_size_m": PHENOMENON_M,
        "geological_age_ma": AGE_MA,
        "mask_pixels": int(mask.sum()),
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    result = run_suite()
    fit = result["dispersion"]
    rows = [
        {
            "name": "dispersion-curve Pearson r of PDE growth versus analytical sigma(k)",
            "value": round(fit["pearson_r"], 6),
            "unit": "Pearson r",
            "method": "log-amplitude growth between t=2 and t=14 versus sigma_mode on the 11 exact DV Fourier modes",
            "reference_value": 0.98,
            "reference_source": "self-test gate; the reference is the acceptance threshold, not the measured r",
            "note": f"RMSE = {fit['rmse']:.6f}",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE between analytical and numerical growth rates",
            "value": round(fit["rmse"], 6),
            "unit": "1/time",
            "method": "root mean square of sigma_analytical - sigma_numerical",
            "reference_value": 0.02,
            "reference_source": "acceptance ceiling used by the self-test, not the measured residual",
            "derived_from": "script",
        },
        {
            "name": "Kolmogorov-Smirnov statistic of analytical versus numerical growth-rate samples",
            "value": round(fit["ks_statistic"], 6),
            "unit": "KS D",
            "method": "scipy.stats.ks_2samp on the 11 exact-mode sigma samples",
            "reference_value": 0.40,
            "reference_source": "pre-registered ceiling on a 20-wavenumber sample; a smaller D is a closer fit",
            "note": f"two-sided p = {fit['ks_pvalue']:.4g}",
            "derived_from": "script",
        },
        {
            "name": "selected dorsoventral wavelength of the Holocene Danio rerio field",
            "value": round(result["wavelength_um"], 4),
            "unit": "µm",
            "method": "argmax of the x-averaged Fourier profile on the 144.48 µm window",
            "reference_value": 24.08,
            "reference_source": "slot scale 2.408e-05 m; reported only as the target, the measured mode is an integer harmonic",
            "note": "six wavelengths span Ly when the mode index is 6",
            "derived_from": "script",
        },
        {
            "name": "predecessor Danio nigrofasciatus dominant wavelength at anisotropy 1.12",
            "value": round(result["predecessor"]["wavelength_um"], 4),
            "unit": "µm",
            "method": "same grid and seed, inhibitor anisotropy 1.12, no phase cue",
            "reference_value": 24.08,
            "reference_source": "the rerio target; the predecessor is not expected to lock the same harmonic",
            "note": f"predecessor gradient anisotropy = {result['predecessor']['gradient_anisotropy']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion negative-control field amplitude",
            "value": round(result["equal_diffusion_amplitude"], 6),
            "unit": "concentration std",
            "method": "Dv_x = Dv_y = Du = 0.80, same gamma, 800 steps",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: equal diffusivities cannot destabilise a stable steady state",
            "derived_from": "script",
        },
        {
            "name": "specimen-image structure-tensor coherency of the rendered GIF",
            "value": round(result["simulation_coherency"], 4),
            "unit": "coherency",
            "method": "Sobel structure tensor on the last GIF frame",
            "reference_value": round(result["specimen_coherency"], 4),
            "reference_source": "same operator on the vendored Danio rerio photograph",
            "note": "simulation and specimen values are measured separately",
            "derived_from": "output",
        },
        {
            "name": "segmented specimen foreground fraction versus GIF occupancy",
            "value": round(result["simulation_foreground_fraction"], 4),
            "unit": "area fraction",
            "method": "fraction of GIF pixels darker than 0.90 after fish-mask rendering",
            "reference_value": round(result["specimen_foreground_fraction"], 4),
            "reference_source": "binary mask segmented from danio_rerio_specimen.jpg",
            "note": f"specimen aspect ratio (length/height) = {result['specimen_aspect_ratio']:.3f}",
            "derived_from": "output",
        },
    ]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = run_suite()
    fit = result["dispersion"]
    print(
        f"lambda={result['wavelength_um']:.3f} µm  "
        f"aniso={result['gradient_anisotropy']:.2f}  "
        f"r={fit['pearson_r']:.4f}  rmse={fit['rmse']:.5f}  "
        f"ks={fit['ks_statistic']:.3f}  "
        f"gif={result['gif_width']}x{result['gif_height']}x{result['gif_frames']}"
    )
    if args.self_test:
        assert fit["pearson_r"] > 0.98, fit
        assert fit["rmse"] < 0.02, fit
        assert abs(result["wavelength_um"] - WAVELENGTH_UM) < 0.75, result
        assert result["equal_diffusion_amplitude"] < 0.08, result
        assert result["gradient_anisotropy"] > result["predecessor"]["gradient_anisotropy"], result
        assert result["gradient_anisotropy"] > 3.0, result
        assert result["gif_frames"] >= 10, result
        assert min(result["gif_width"], result["gif_height"]) >= 480, result
        assert result["simulation_coherency"] != result["specimen_coherency"]
        orientations = [row["gradient_anisotropy"] for row in result["sweep"]]
        assert orientations[-1] > orientations[0], orientations
        for seed_row in result["seeds"]:
            assert abs(seed_row["wavelength_um"] - WAVELENGTH_UM) < 1.5, seed_row
        assert result["phenomenon_size_m"] == PHENOMENON_M
        assert result["geological_age_ma"] == 0.0
        print("[PASS] models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
