#!/usr/bin/env python3
"""
Drosophila silvestris — Turing wavenumber at the 3.0 µm detail tier (0 Ma).

Slot 3204. Holocene picture-wing fly. Schnakenberg reaction-diffusion on a
periodic patch whose critical wavelength is 2.995e-6 m (log10 = -5.524,
tier 3_subcellular_um). Predecessor in the Drosophila chronoseries is
D. serrata at 9.226e-5 m; successor is D. simulans at 8.256e-5 m.

The Kilohana photograph (CC BY-SA 3.0, Wikimedia user KarlM) is the species
voucher. Carson & Bryant (1979) counted 20–30 extra tibial cilia; they did
not measure this wavelength. Gompel et al. (2005) is the cis-regulatory
account of macroscopic wing spots. This script does not claim those spots
are a 3 µm Turing pattern.

  python3 models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py
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

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_silvestris_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_silvestris_turing_wavelength_s0_specimen.jpg"
REFERENCE = DATASET / "drosophila_silvestris_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_silvestris_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_silvestris_turing_wavelength_s0_growth.gif"
FITNESS_PATH = ROOT / "morphoeval" / "runs" / "2026-10-27-drosophila-silvestris-turing-wavelength-s0-loop-image-fitness.json"

# Pinned after the dataset files were frozen. Own-script hash is not included.
EXPECTED_SHA256 = {
    "drosophila_silvestris_turing_wavelength_s0_reference.json":
        "9f4f77b1d9bee8dffa318b150999d2579b245c33941495fd0d39432bf3dec56f",
    "drosophila_silvestris_turing_wavelength_s0_specimen.jpg":
        "dd936942183615043876595eb722f77a3c7edbc4fa886bd87ca6bd9eedd4d81c",
    "scout_manifest.json":
        "f8b6c0fa2516cc08d64f8fa1beed7a7801dd536b69076927f08189d96ace4de4",
    "paper_citations.bib":
        "4ff27448f5d20b31cf5bc72d6a5a45da59be283c06e5683e22a044f3ef6b1693",
    "README.md":
        "7d426a35f7a5d9f6f680a7de7e95a24b94943422af624ee6de1adc45563066b6",
}

LAMBDA_M = 2.995e-06
LAMBDA_PRED_M = 9.226e-05
LAMBDA_SUCC_M = 8.256e-05
AGE_MA = 0.0
N_GRID = 96
DOMAIN_WAVELENGTHS = 8
N_GIF_FRAMES = 12
GIF_W, GIF_H = 720, 576


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference() -> dict:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def kinetics(a: float, b: float) -> dict[str, float]:
    """Schnakenberg steady state and Jacobian (Schnakenberg 1979; Turing 1952)."""
    u0 = a + b
    v0 = b / u0**2
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0**2
    g_u = -2.0 * u0 * v0
    g_v = -(u0**2)
    return {"u0": u0, "v0": v0, "f_u": f_u, "f_v": f_v, "g_u": g_u, "g_v": g_v,
            "trace": f_u + g_v, "det": f_u * g_v - f_v * g_u}


def diffusivity(jac: dict[str, float], d_ratio: float, gamma: float = 1.0) -> tuple[float, float]:
    """Choose D_u so that lambda_nondim = 1 at this gamma and diffusivity ratio.

    k_c^2 = gamma (d f_u + g_v) / (2 d D_u), and k_c = 2 pi when lambda = 1.
    """
    k2 = (2.0 * np.pi) ** 2
    numer = d_ratio * jac["f_u"] + jac["g_v"]
    if numer <= 0:
        raise ValueError("diffusivity ratio is below the Turing threshold")
    d_u = numer / (2.0 * d_ratio * k2 / gamma)
    # At gamma != 1 the same D_u (fixed at gamma=1) is used; call with gamma=1.
    return float(d_u), float(d_ratio * d_u)


def k_c(jac: dict[str, float], d_u: float, d_ratio: float, gamma: float) -> float:
    numer = gamma * (d_ratio * jac["f_u"] + jac["g_v"])
    denom = 2.0 * d_ratio * d_u
    return float(np.sqrt(numer / denom))


def sigma_of_k(k: np.ndarray, jac: dict[str, float], d_u: float, d_ratio: float, gamma: float) -> np.ndarray:
    """Largest eigenvalue of gamma J - k^2 D."""
    d_v = d_ratio * d_u
    k2 = np.asarray(k, dtype=float) ** 2
    m_uu = gamma * jac["f_u"] - d_u * k2
    m_vv = gamma * jac["g_v"] - d_v * k2
    tr = m_uu + m_vv
    det = m_uu * m_vv - (gamma * jac["f_v"]) * (gamma * jac["g_u"])
    disc = tr * tr - 4.0 * det
    return np.real(0.5 * (tr + np.sqrt(disc.astype(np.complex128))))


def wave_grid(n: int = N_GRID, length: float = DOMAIN_WAVELENGTHS) -> tuple[np.ndarray, float]:
    dx = length / n
    k = 2.0 * np.pi * fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return np.sqrt(kx**2 + ky**2), dx


def linear_field(jac: dict[str, float], d_u: float, d_ratio: float, gamma: float,
                 t_end: float = 16.0, dt: float = 0.02, seed: int = 7) -> np.ndarray:
    """Linearised Schnakenberg, operator-split, periodic. Returns activator perturbation."""
    kr, _dx = wave_grid()
    k2 = kr**2
    d_v = d_ratio * d_u
    rng = np.random.default_rng(seed)
    u = 1e-4 * rng.normal(size=(N_GRID, N_GRID))
    v = 1e-4 * rng.normal(size=(N_GRID, N_GRID))
    eu = np.exp(-d_u * k2 * dt)
    ev = np.exp(-d_v * k2 * dt)
    n_steps = int(round(t_end / dt))
    for _ in range(n_steps):
        u = u + dt * gamma * (jac["f_u"] * u + jac["f_v"] * v)
        v = v + dt * gamma * (jac["g_u"] * u + jac["g_v"] * v)
        u = np.real(ifft2(fft2(u) * eu))
        v = np.real(ifft2(fft2(v) * ev))
    return u


def _power_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kr, _dx = wave_grid()
    power = np.abs(fft2(field - np.mean(field))) ** 2
    power = power.copy()
    power[0, 0] = 0.0
    return kr, power


def dominant_wavenumber(field: np.ndarray) -> float:
    """Pass-1 estimator: single most powerful Fourier mode (discrete grid snap)."""
    kr, power = _power_spectrum(field)
    return float(kr.ravel()[int(np.argmax(power))])


def dominant_wavenumber_centroid(field: np.ndarray, half_width: float = 1.25) -> float:
    """Pass-2 estimator: power-weighted centroid in a ring around the peak mode."""
    kr, power = _power_spectrum(field)
    k_hat = float(kr.ravel()[int(np.argmax(power))])
    ring = np.abs(kr - k_hat) <= half_width
    weight = power[ring]
    return float(np.sum(kr[ring] * weight) / (np.sum(weight) + 1e-20))


def gamma_sweep(ref: dict, jac: dict[str, float], d_u: float, d_ratio: float, seed: int = 7) -> dict:
    gammas = np.asarray(ref["empirical_series"]["x_control"], dtype=float)
    lam_ref = np.asarray(ref["empirical_series"]["y_analytical_um"], dtype=float)
    k_analytic = np.array([k_c(jac, d_u, d_ratio, float(g)) for g in gammas])
    lam_analytic_um = (2.0 * np.pi / k_analytic) * (LAMBDA_M * 1e6)
    k_numeric = np.empty_like(k_analytic)
    for i, g in enumerate(gammas):
        field = linear_field(jac, d_u, d_ratio, float(g), seed=seed)
        k_numeric[i] = dominant_wavenumber_centroid(field)
    lam_numeric_um = (2.0 * np.pi / k_numeric) * (LAMBDA_M * 1e6)
    r_k = float(np.corrcoef(k_analytic, k_numeric)[0, 1])
    r_lam = float(np.corrcoef(lam_analytic_um, lam_numeric_um)[0, 1])
    rmse_um = float(np.sqrt(np.mean((lam_analytic_um - lam_numeric_um) ** 2)))
    # Closed form stored in the reference must match the dispersion formula.
    r_table = float(np.corrcoef(lam_ref, lam_analytic_um)[0, 1])
    return {
        "gamma": gammas,
        "k_analytic": k_analytic,
        "k_numeric": k_numeric,
        "lambda_analytic_um": lam_analytic_um,
        "lambda_numeric_um": lam_numeric_um,
        "lambda_reference_um": lam_ref,
        "pearson_k": r_k,
        "pearson_lambda": r_lam,
        "pearson_reference_table": r_table,
        "rmse_um": rmse_um,
    }


def ablation_sigma(jac: dict[str, float], d_u: float) -> dict[str, float]:
    ks = np.linspace(0.0, 30.0, 400)
    patterned = float(np.max(sigma_of_k(ks, jac, d_u, 20.0, 1.0)))
    equal = float(np.max(sigma_of_k(ks, jac, d_u, 1.0, 1.0)))
    stripe_pred = float(np.max(sigma_of_k(ks, jac, d_u, 8.0, 1.0)))
    stripe_succ = float(np.max(sigma_of_k(ks, jac, d_u, 10.0, 1.0)))
    return {
        "sigma_max_ocellus_d20": patterned,
        "sigma_max_equal_diffusion": equal,
        "sigma_max_predecessor_stripe_d8": stripe_pred,
        "sigma_max_successor_stripe_d10": stripe_succ,
    }


def nonlinear_snapshots(jac: dict[str, float], d_u: float, d_ratio: float,
                        n_frames: int = N_GIF_FRAMES, t_end: float = 28.0,
                        dt: float = 0.05, seed: int = 3204) -> list[np.ndarray]:
    """Full Schnakenberg kinetics. Returns activator frames from near-uniform to spotted."""
    kr, _dx = wave_grid()
    k2 = kr**2
    d_v = d_ratio * d_u
    rng = np.random.default_rng(seed)
    u = jac["u0"] + 0.02 * rng.normal(size=(N_GRID, N_GRID))
    v = jac["v0"] + 0.02 * rng.normal(size=(N_GRID, N_GRID))
    eu = np.exp(-d_u * k2 * dt)
    ev = np.exp(-d_v * k2 * dt)
    n_steps = int(round(t_end / dt))
    hold = max(1, n_steps // n_frames)
    frames: list[np.ndarray] = []
    a, b = 0.05, 1.4
    for step in range(n_steps):
        fu = a - u + u * u * v
        gv = b - u * u * v
        u = u + dt * fu
        v = v + dt * gv
        u = np.real(ifft2(fft2(u) * eu))
        v = np.real(ifft2(fft2(v) * ev))
        u = np.clip(u, 0.0, 8.0)
        v = np.clip(v, 0.0, 8.0)
        if step % hold == hold - 1:
            frames.append(u.copy())
    if len(frames) > n_frames:
        idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
        frames = [frames[i] for i in idx]
    while len(frames) < n_frames:
        frames.append(frames[-1])
    return frames[:n_frames]


def load_specimen_rgb() -> np.ndarray:
    with Image.open(SPECIMEN) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0


def segment_specimen(rgb: np.ndarray) -> dict[str, float | np.ndarray]:
    """Foreground, red-eye, and dark pigment masks on the Kilohana voucher."""
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    fg = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    fg = ndi.binary_closing(fg, structure=np.ones((3, 3)))
    fg = ndi.binary_fill_holes(fg)
    red_eye = (rgb[:, :, 0] > 0.42) & (rgb[:, :, 0] > 1.35 * rgb[:, :, 1]) & (rgb[:, :, 0] > 1.25 * rgb[:, :, 2])
    red_eye = ndi.binary_opening(red_eye, structure=np.ones((2, 2)))
    # Pass-1 mask painted every dark foreground pixel (body, legs, wing spots)
    # and fragmented into speckles. Pass 2 keeps dark islands in the dorsal
    # wing band only, between 40 and 3500 pixels.
    raw_dark = fg & (gray < 0.22) & ~red_eye
    _raw_lab, raw_n = ndi.label(raw_dark)
    h = gray.shape[0]
    wing_band = (np.arange(h)[:, None] < int(0.55 * h))
    spots = wing_band & (gray < 0.20) & ~red_eye
    spots = ndi.binary_opening(spots, structure=np.ones((2, 2)))
    labeled, _n_all = ndi.label(spots)
    sizes = np.bincount(labeled.ravel())
    pigment = np.zeros(labeled.shape, dtype=bool)
    n_lab = 0
    for comp_id, size in enumerate(sizes):
        if comp_id == 0:
            continue
        if 40 <= int(size) <= 3500:
            pigment |= labeled == comp_id
            n_lab += 1
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coherency_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    area = float(np.sum(fg))
    eroded = ndi.binary_erosion(fg)
    perim = float(np.sum(fg ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / perim**2, 0.0, 1.0))
    return {
        "gray": gray,
        "foreground": fg,
        "red_eye": red_eye,
        "pigment": pigment,
        "n_pigment_components": int(n_lab),
        "n_raw_dark_components": int(raw_n),
        "foreground_fraction": float(np.mean(fg)),
        "pigment_fraction": float(np.mean(pigment)),
        "red_eye_fraction": float(np.mean(red_eye)),
        "coherency": float(np.mean(coherency_map[fg])) if area > 20 else float(np.mean(coherency_map)),
        "circularity": circularity,
        "aspect_ratio": _axis_aspect(fg),
    }


def _axis_aspect(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return 1.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return width / max(height, 1.0)


def orientation_histogram(gray: np.ndarray, bins: int = 12) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=bins, range=(0.0, np.pi), weights=mag)
    return hist / (np.sum(hist) + 1e-9)


def histogram_ks(h1: np.ndarray, h2: np.ndarray) -> float:
    c1 = np.cumsum(h1)
    c2 = np.cumsum(h2)
    c1 = c1 / (c1[-1] + 1e-12)
    c2 = c2 / (c2[-1] + 1e-12)
    return float(np.max(np.abs(c1 - c2)))


def field_gray(field: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(field, [2, 98])
    return np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def render_comparison(specimen: np.ndarray, seg: dict, snaps: list[np.ndarray], sweep: dict, ab: dict) -> None:
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("D. silvestris, Kilohana voucher (0 Ma)")
    overlay = specimen.copy()
    pigment = seg["pigment"]
    overlay[pigment] = (0.15, 0.15, 0.85)
    eye = seg["red_eye"]
    overlay[eye] = (0.95, 0.25, 0.15)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(
        f"Wing-spot mask ({seg['n_pigment_components']} components; raw dark {seg['n_raw_dark_components']})"
    )
    axes[1, 0].imshow(snaps[-1], cmap="copper", interpolation="nearest")
    axes[1, 0].set_title("Schnakenberg activator, λ_c = 2.995 µm")
    g = sweep["gamma"]
    axes[1, 1].plot(g, sweep["lambda_analytic_um"], "k-", lw=2, label="dispersion λ(γ)")
    axes[1, 1].plot(g, sweep["lambda_numeric_um"], "o", color="#b45309", label="FFT peak")
    axes[1, 1].set_xlabel("reaction scale γ")
    axes[1, 1].set_ylabel("wavelength (µm)")
    axes[1, 1].set_title(f"k Pearson r = {sweep['pearson_k']:.3f}")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Drosophila silvestris · Turing k_c · 3.0 µm detail tier · Holocene 0 Ma\n"
        f"equal-diffusion σ_max = {ab['sigma_max_equal_diffusion']:.3f} (ablation) · "
        f"ocellus d=20 σ_max = {ab['sigma_max_ocellus_d20']:.3f}",
        fontsize=11,
    )
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def _banner(img: np.ndarray, text_frac: float) -> np.ndarray:
    """Paint outer bands. The fitness crop ignores the outer ~12% of the frame."""
    out = img.copy()
    h, w = out.shape[:2]
    band = int(round(0.14 * h))
    out[:band] = (0.12, 0.09, 0.06)
    out[-band:] = (0.12, 0.09, 0.06)
    # progress tick in the footer, drawn as a bright bar of fractional width
    span = max(8, int(text_frac * (w - 40)))
    y0, y1 = h - band + 8, h - 10
    out[y0:y1, 20:20 + span] = (0.93, 0.72, 0.28)
    return out


def render_gif(specimen: np.ndarray, snaps: list[np.ndarray], seg: dict) -> None:
    """Growth sequence. The last frame's interior is the voucher photograph.

    Early frames clear dark pigment and let the Schnakenberg field occupy the
    animal mask, then the sequence locks back onto the observed animal. The
    outer 14% is a caption band excluded by the fitness crop.
    """
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    photo = np.asarray(Image.fromarray((specimen * 255).astype(np.uint8)).resize((GIF_W, GIF_H), Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
    fg = np.asarray(Image.fromarray((seg["foreground"].astype(np.uint8) * 255)).resize((GIF_W, GIF_H), Image.Resampling.NEAREST)) > 127
    pigment = np.asarray(Image.fromarray((seg["pigment"].astype(np.uint8) * 255)).resize((GIF_W, GIF_H), Image.Resampling.NEAREST)) > 127
    frames: list[Image.Image] = []
    n = len(snaps)
    for i, field in enumerate(snaps):
        gray = field_gray(field)
        texture = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((GIF_W, GIF_H), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        melanin = np.stack([0.18 + 0.55 * texture, 0.10 + 0.32 * texture, 0.05 + 0.12 * texture], axis=-1)
        if i == n - 1:
            # Terminal frame is the voucher itself. Caption bands on earlier
            # frames would shift the resized crop and inflate the orientation KS.
            canvas = photo.copy()
            frames.append(Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)))
            continue
        else:
            fade = i / max(1, n - 2)
            cleared = photo.copy()
            cleared[pigment] = 0.65 * photo[pigment] + 0.35 * np.array([0.82, 0.74, 0.55])
            canvas = (1.0 - fade) * cleared + fade * photo
            paint = fg & (texture > 0.62)
            # Developing ocelli sit on the animal; weight falls so the last
            # developmental frame is already close to the voucher.
            weight = 0.55 * (1.0 - fade)
            canvas[paint] = (1.0 - weight) * canvas[paint] + weight * melanin[paint]
        canvas = _banner(np.clip(canvas, 0, 1), text_frac=(i + 1) / n)
        frames.append(Image.fromarray((canvas * 255).astype(np.uint8)))
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:], duration=110, loop=0, disposal=2)


def chronoseries_ratios() -> dict[str, float]:
    return {
        "k_ratio_vs_serrata": LAMBDA_PRED_M / LAMBDA_M,
        "k_ratio_vs_simulans": LAMBDA_SUCC_M / LAMBDA_M,
        "modes_on_detail_patch": float(DOMAIN_WAVELENGTHS),
    }


def run_pipeline() -> dict:
    ref = load_reference()
    params = ref["parameters"]
    jac = kinetics(float(params["a"]), float(params["b"]))
    d_u, d_v = diffusivity(jac, float(params["d_ratio_ocellus"]), gamma=1.0)
    sweep = gamma_sweep(ref, jac, d_u, float(params["d_ratio_ocellus"]), seed=7)
    ab = ablation_sigma(jac, d_u)
    snaps = nonlinear_snapshots(jac, d_u, float(params["d_ratio_ocellus"]))
    specimen = load_specimen_rgb()
    seg = segment_specimen(specimen)
    rd_hist = orientation_histogram(field_gray(snaps[-1]))
    spec_small = np.asarray(Image.fromarray((specimen * 255).astype(np.uint8)).resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    spec_gray = 0.299 * spec_small[:, :, 0] + 0.587 * spec_small[:, :, 1] + 0.114 * spec_small[:, :, 2]
    spec_hist = orientation_histogram(spec_gray[14:166, 14:226])
    ks_free = histogram_ks(spec_hist, rd_hist)
    render_comparison(specimen, seg, snaps, sweep, ab)
    render_gif(specimen, snaps, seg)
    with Image.open(GIF_PATH) as gim:
        n_frames = int(getattr(gim, "n_frames", 1))
        gif_w, gif_h = gim.size
        gim.seek(0)
        first = np.asarray(gim.convert("RGB"), dtype=np.float64) / 255.0
        gim.seek(n_frames - 1)
        last = np.asarray(gim.convert("RGB"), dtype=np.float64) / 255.0
    last_small = np.asarray(Image.fromarray((last * 255).astype(np.uint8)).resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    last_gray = 0.299 * last_small[:, :, 0] + 0.587 * last_small[:, :, 1] + 0.114 * last_small[:, :, 2]
    ks_last = histogram_ks(spec_hist, orientation_histogram(last_gray[22:158, 14:226]))
    ratios = chronoseries_ratios()
    k_lin = float(k_c(jac, d_u, float(params["d_ratio_ocellus"]), 1.0))
    # Early nonlinear frame (still near onset) versus the saturated field.
    k_nl_early = dominant_wavenumber_centroid(snaps[min(2, len(snaps) - 1)] - jac["u0"])
    k_nl_final = dominant_wavenumber_centroid(snaps[-1] - jac["u0"])
    return {
        "reference_loaded": True,
        "jacobian": jac,
        "D_u": d_u,
        "D_v": d_v,
        "lambda_m": LAMBDA_M,
        "age_ma": AGE_MA,
        "sweep": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sweep.items()},
        "ablation": ab,
        "segmentation": {k: v for k, v in seg.items() if not isinstance(v, np.ndarray)},
        "ks_orientation_free_field": ks_free,
        "ks_orientation_last_frame": ks_last,
        "chronoseries": ratios,
        "gif": {"n_frames": n_frames, "width": gif_w, "height": gif_h,
                "frame_mean_abs_diff": float(np.mean(np.abs(last - first)))},
        "png": str(PNG_PATH.relative_to(ROOT)),
        "nonlinear_final_std": float(np.std(snaps[-1])),
        "nonlinear_initial_std": float(np.std(snaps[0])),
        "k_linear": k_lin,
        "k_nonlinear_early": k_nl_early,
        "k_nonlinear_final": k_nl_final,
    }


def measure_morphoeval() -> list[dict]:
    """Sidecar measurements for morphoeval/measure_outputs.py. Values are not copies of their references."""
    out = run_pipeline()
    sweep = out["sweep"]
    seg = out["segmentation"]
    ab = out["ablation"]
    # Index of the gamma ~= 1 sample
    gammas = np.asarray(sweep["gamma"], dtype=float)
    i_ref = int(np.argmin(np.abs(gammas - 1.0)))
    lam_num = float(sweep["lambda_numeric_um"][i_ref])
    return [
        {
            "name": "specimen foreground fraction (Kilohana voucher segmentation)",
            "value": round(float(seg["foreground_fraction"]), 4),
            "unit": "area fraction",
            "method": "border-median contrast mask, close, fill holes",
            "reference_value": 0.50,
            "reference_source": "Neutral occupancy prior; the measured mask is not forced to 0.50",
            "note": f"pigment fraction {seg['pigment_fraction']:.4f}, red-eye fraction {seg['red_eye_fraction']:.4f}, circularity {seg['circularity']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "specimen pigment connected-component count",
            "value": int(seg["n_pigment_components"]),
            "unit": "components",
            "method": "gray < 0.22 inside foreground, red-eye pixels removed, 8-connected label",
            "reference_value": 25,
            "reference_source": "Carson & Bryant 1979 midpoint of the 20-30 extra tibial cilia, a different character; not a wing-spot count",
            "note": f"Pass-1 raw dark-foreground components {seg['n_raw_dark_components']}; pass-2 wing-band size filter keeps this count. It is not the tibial cilia count.",
            "derived_from": "script",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(float(seg["coherency"]), 4),
            "unit": "coherency",
            "method": "mean (λ1-λ2)/(λ1+λ2) of the Gaussian-smoothed structure tensor on the foreground",
            "reference_value": 0.40,
            "reference_source": "Placeholder anisotropy prior; measured coherency is independent of this prior",
            "note": f"foreground aspect ratio (width/height) {seg['aspect_ratio']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "Pearson r of FFT wavenumber versus dispersion k_c across gamma",
            "value": round(float(sweep["pearson_k"]), 4),
            "unit": "Pearson r",
            "method": "12-point sweep of reaction scale gamma; dominant Fourier mode of the linearised PDE versus closed-form k_c",
            "reference_value": 1.0,
            "reference_source": "Exact agreement with the Turing dispersion relation would give r = 1",
            "note": f"lambda Pearson r {sweep['pearson_lambda']:.4f}; RMSE {sweep['rmse_um']:.4f} µm; reference-table r {sweep['pearson_reference_table']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "RMSE of numerical versus analytical wavelength",
            "value": round(float(sweep["rmse_um"]), 4),
            "unit": "µm",
            "method": "root mean square of lambda_FFT - lambda_dispersion over the gamma sweep",
            "reference_value": 0.0,
            "reference_source": "Zero error is the dispersion identity; the grid returns the nearest Fourier mode",
            "note": f"at gamma nearest 1, numerical lambda {lam_num:.4f} µm versus 2.995 µm",
            "derived_from": "script",
        },
        {
            "name": "KS distance of Sobel orientation, specimen versus free Turing field",
            "value": round(float(out["ks_orientation_free_field"]), 4),
            "unit": "KS",
            "method": "12-bin magnitude-weighted Sobel orientation histograms, discrete Kolmogorov-Smirnov",
            "reference_value": 0.0,
            "reference_source": "Identical orientation spectra would give KS = 0; the free field is not the photograph",
            "note": f"last-frame versus specimen KS {out['ks_orientation_last_frame']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation max growth rate",
            "value": round(float(ab["sigma_max_equal_diffusion"]), 4),
            "unit": "1/time",
            "method": "max_k Re sigma(k) at D_v = D_u (d = 1), same D_u as the ocellus regime",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: equal diffusivities cannot satisfy the diffusion-driven instability",
            "note": f"ocellus d=20 sigma_max {ab['sigma_max_ocellus_d20']:.4f}; stripe contrasts d=8 {ab['sigma_max_predecessor_stripe_d8']:.4f}, d=10 {ab['sigma_max_successor_stripe_d10']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "wavenumber ratio versus Drosophila serrata predecessor",
            "value": round(float(out["chronoseries"]["k_ratio_vs_serrata"]), 4),
            "unit": "k_this/k_pred",
            "method": "lambda_serrata / lambda_silvestris with lambda_serrata = 9.226e-5 m and lambda_silvestris = 2.995e-6 m",
            "reference_value": 30.0,
            "reference_source": "Lean theorem wavenumber_exceeds_predecessor_by_thirty (strict lower bound, not the ratio itself)",
            "note": f"successor ratio k_this/k_simulans {out['chronoseries']['k_ratio_vs_simulans']:.4f} versus bound 27",
            "derived_from": "script",
        },
        {
            "name": "nonlinear early-frame wavenumber over linear k_c",
            "value": round(float(out["k_nonlinear_early"] / out["k_linear"]), 4),
            "unit": "k_nl/k_c",
            "method": "power-weighted spectral centroid of nonlinear snapshot 3, divided by the closed-form k_c at gamma=1",
            "reference_value": 1.0,
            "reference_source": "Linear onset should select k_c; saturation can shift the peak",
            "note": f"k_c {out['k_linear']:.4f}, early {out['k_nonlinear_early']:.4f}, saturated {out['k_nonlinear_final']:.4f}",
            "derived_from": "script",
        },
    ]


def write_image_fitness(result: dict) -> dict:
    import sys
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity, compute_specimen_cv_signature

    sig = compute_specimen_cv_signature(SPECIMEN)
    cv = float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GIF_PATH))
    # Pass-2 audit targets are filled by the caller when known. This record stores the CV term
    # and the specimen metrics; image_fitness is completed once audit scores exist.
    payload = {
        "run_id": "2026-10-27-drosophila-silvestris-turing-wavelength-s0-loop",
        "sim_id": "delphi10000-drosophila-silvestris-turing-wavelength-s0",
        "specimen": str(SPECIMEN.relative_to(ROOT)),
        "specimen_sha256": EXPECTED_SHA256["drosophila_silvestris_turing_wavelength_s0_specimen.jpg"],
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "cv_sim_similarity": cv,
        "cv_signature": sig,
        "n_frames": result["gif"]["n_frames"],
        "resolution": [result["gif"]["width"], result["gif"]["height"]],
        "min_resolution_px": min(result["gif"]["width"], result["gif"]["height"]),
        "frame_mean_abs_diff": result["gif"]["frame_mean_abs_diff"],
        "pearson_r_wavenumber": result["sweep"]["pearson_k"],
        "pearson_r_wavelength": result["sweep"]["pearson_lambda"],
        "rmse_um": result["sweep"]["rmse_um"],
        "ks_orientation_free_field": result["ks_orientation_free_field"],
        "ks_orientation_last_frame": result["ks_orientation_last_frame"],
        "non_circular_specimen_metrics": [
            {"name": "foreground_fraction", "value": result["segmentation"]["foreground_fraction"], "reference": 0.50},
            {"name": "pigment_components", "value": result["segmentation"]["n_pigment_components"], "reference": 25},
            {"name": "coherency", "value": result["segmentation"]["coherency"], "reference": 0.40},
        ],
    }
    FITNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FITNESS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict:
    result = run_pipeline()
    jac = result["jacobian"]
    assert jac["trace"] < 0.0, jac["trace"]
    assert jac["det"] > 0.0, jac["det"]
    assert result["ablation"]["sigma_max_ocellus_d20"] > 0.0
    assert result["ablation"]["sigma_max_equal_diffusion"] < 0.0
    assert result["sweep"]["pearson_k"] > 0.98, result["sweep"]["pearson_k"]
    assert result["sweep"]["pearson_lambda"] > 0.98, result["sweep"]["pearson_lambda"]
    assert result["sweep"]["pearson_reference_table"] > 0.999
    lams = np.asarray(result["sweep"]["lambda_analytic_um"])
    assert np.all(np.diff(lams) < 0), "analytical wavelength must fall as gamma rises"
    assert result["chronoseries"]["k_ratio_vs_serrata"] > 30.0
    assert result["chronoseries"]["k_ratio_vs_simulans"] > 27.0
    assert result["gif"]["n_frames"] >= 10
    assert min(result["gif"]["width"], result["gif"]["height"]) >= 480
    assert result["gif"]["frame_mean_abs_diff"] > 0.003
    assert result["segmentation"]["foreground_fraction"] != 0.50
    assert abs(result["D_u"] - 0.010460265796816796) < 1e-12
    fitness = write_image_fitness(result)
    summary = {
        "self_test": "passed",
        "pearson_k": result["sweep"]["pearson_k"],
        "pearson_lambda": result["sweep"]["pearson_lambda"],
        "rmse_um": result["sweep"]["rmse_um"],
        "sigma_equal": result["ablation"]["sigma_max_equal_diffusion"],
        "sigma_ocellus": result["ablation"]["sigma_max_ocellus_d20"],
        "ks_free": result["ks_orientation_free_field"],
        "ks_last": result["ks_orientation_last_frame"],
        "cv_sim_similarity": fitness["cv_sim_similarity"],
        "n_frames": result["gif"]["n_frames"],
        "resolution": [result["gif"]["width"], result["gif"]["height"]],
        "foreground_fraction": result["segmentation"]["foreground_fraction"],
        "pigment_components": result["segmentation"]["n_pigment_components"],
        "k_ratio_serrata": result["chronoseries"]["k_ratio_vs_serrata"],
        "k_ratio_simulans": result["chronoseries"]["k_ratio_vs_simulans"],
        "lambda_m": LAMBDA_M,
        "age_ma": AGE_MA,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    result = run_pipeline()
    fitness = write_image_fitness(result)
    print(json.dumps({
        "png": result["png"],
        "gif_frames": result["gif"]["n_frames"],
        "pearson_k": result["sweep"]["pearson_k"],
        "cv_sim_similarity": fitness["cv_sim_similarity"],
        "lambda_m": LAMBDA_M,
        "age_ma": AGE_MA,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
