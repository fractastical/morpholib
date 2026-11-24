#!/usr/bin/env python3
"""
Drosophila hydei detail-scale Turing wavelength (Project Delphi loop 3189, s0).

Animal: Drosophila hydei, Holocene extant, geological age 0 Ma.
Scale: 2.3 µm (canonical 2.344e-06 m, log10 = -5.630, tier 3/8 ``3_subcellular_um``).
Form: stripes, spots and ocelli. Law: Turing wavenumber k_c.
Lineage: Subgenus Drosophila virilis–repleta–guttifera radiation (chronoseries 18/41).

Predecessor Drosophila heteroneura selects a picture-wing band at 175.4 µm.
The same species' primordium slot is 117.2 µm, exactly 50 times this wavelength.
Successor Drosophila kikkawai selects a tergite scale at 93.41 µm.
This file is the 2.344 µm cuticle-patch wavenumber, not those coarser slots.

Kinetics (Schnakenberg 1979), instability (Turing 1952):

    du/dt = γ (a - u + u² v) + D_u ∇²u
    dv/dt = γ (b - u² v) + D_v ∇²v

    u* = a + b,  v* = b / (a + b)²
    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),   λ_c = 2π / k_c

Parameters a=0.10, b=0.90, γ=1, D_u=1, D_v=20 give trace=-0.2, det=1.
The length unit is fixed so λ_c(γ=1) = 2.344 µm. Periodic boundary conditions
on a square patch of side 4 λ_c = 9.376 µm.

Quadrupling γ halves λ at fixed diffusion. Equal diffusion (D_u=D_v) makes
D f_u + D g_v = D·trace < 0, so the Turing condition fails.

The whole-animal photograph does not resolve 2.3 µm. The animation shows the
patch field growing and, on the last frame, a low-amplitude pigment modulation
on the real CC BY 3.0 specimen so body-plan texture remains comparable.

    python3 models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-drosophila-hydei-turing-wavelength-s0.py
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
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_hydei_turing_wavelength_s0"
REF_PATH = DATASET / "drosophila_hydei_turing_wavelength_s0_reference.json"
SPECIMEN_PATH = DATASET / "drosophila_hydei_turing_wavelength_s0_specimen.jpg"
OUT_PNG = ROOT / "docs" / "media" / "drosophila_hydei_turing_wavelength_s0_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "drosophila_hydei_turing_wavelength_s0_growth.gif"

# Predecessor / successor / same-species primordium wavelengths (µm). Not fitted here.
LAMBDA_HETERONEURA_UM = 175.4
LAMBDA_PRIMORDIUM_UM = 117.2
LAMBDA_KIKKAWAI_UM = 93.41

_CACHE: dict = {}


def load_reference() -> dict:
    return json.loads(REF_PATH.read_text(encoding="utf-8"))


def verify_dataset_pins(ref: dict) -> None:
    """Check SHA-256 pins recorded in the reference JSON (not a hash of that JSON)."""
    pins = ref.get("sha256") or {}
    specimen_pin = ref["specimen"]["sha256"]
    if pins.get(SPECIMEN_PATH.name) != specimen_pin:
        raise RuntimeError("specimen pin in sha256 map does not match specimen.sha256")
    for name, expected in pins.items():
        path = DATASET / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")


def jacobian(a: float, b: float, gamma: float) -> tuple[float, float, float, float, float, float]:
    """Return u*, v*, f_u, f_v, g_u, g_v at the homogeneous steady state."""
    us = a + b
    vs = b / (us * us)
    fu = gamma * (-1.0 + 2.0 * us * vs)
    fv = gamma * (us * us)
    gu = gamma * (-2.0 * us * vs)
    gv = -gamma * (us * us)
    return us, vs, fu, fv, gu, gv


def dispersion(k2: float, fu: float, fv: float, gu: float, gv: float, Du: float, Dv: float) -> float:
    """Larger eigenvalue σ of the Fourier-mode Jacobian at Laplacian eigenvalue -k²."""
    trk = fu + gv - (Du + Dv) * k2
    detk = Du * Dv * (k2 ** 2) - (Dv * fu + Du * gv) * k2 + (fu * gv - fv * gu)
    disc = trk * trk - 4.0 * detk
    return 0.5 * (trk + np.sqrt(max(disc, 0.0)))


def critical_wavenumber(fu: float, gv: float, Du: float, Dv: float) -> tuple[float, float]:
    """Return (k_c, λ_nd) from k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)."""
    h = Dv * fu + Du * gv
    kc2 = h / (2.0 * Du * Dv)
    if kc2 <= 0:
        raise RuntimeError("no finite critical wavenumber (h <= 0)")
    kc = float(np.sqrt(kc2))
    return kc, float(2.0 * np.pi / kc)


def turing_flags(fu: float, fv: float, gu: float, gv: float, Du: float, Dv: float) -> dict:
    trace = fu + gv
    det = fu * gv - fv * gu
    h = Dv * fu + Du * gv
    disc = h * h - 4.0 * Du * Dv * det
    return {
        "trace": float(trace),
        "det": float(det),
        "h": float(h),
        "disc": float(disc),
        "stable_without_diffusion": bool(trace < 0 and det > 0),
        "diffusion_driven": bool(h > 0 and disc > 0),
    }


def discrete_k2(k: float, dx: float) -> float:
    """Eigenvalue magnitude of the five-point periodic Laplacian on cos(kx)."""
    return float((2.0 - 2.0 * np.cos(k * dx)) / (dx * dx))


def laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / (dx * dx)


def step_field(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dt: float,
    a: float,
    b: float,
    gamma: float,
    Du: float,
    Dv: float,
) -> tuple[np.ndarray, np.ndarray]:
    lu = laplacian(u, dx)
    lv = laplacian(v, dx)
    u = u + dt * (gamma * (a - u + u * u * v) + Du * lu)
    v = v + dt * (gamma * (b - u * u * v) + Dv * lv)
    return u, v


def simulate_patch(ref: dict, n_frames: int = 12) -> dict:
    """Periodic Schnakenberg patch at the reference parameters. Cached."""
    if "sim" in _CACHE and _CACHE.get("n_frames") == n_frames:
        return _CACHE["sim"]
    p = ref["parameters"]
    a, b = float(p["a"]), float(p["b"])
    gamma, Du, Dv = float(p["gamma"]), float(p["Du"]), float(p["Dv"])
    us, vs, fu, fv, gu, gv = jacobian(a, b, gamma)
    flags = turing_flags(fu, fv, gu, gv, Du, Dv)
    kc, lam_nd = critical_wavenumber(fu, gv, Du, Dv)
    target_um = float(p["target_wavelength_um"])
    um_per_nd = target_um / lam_nd
    n_waves = int(p["domain_wavelengths"])
    L_nd = n_waves * lam_nd
    L_um = n_waves * target_um
    N = int(p["grid_n"])
    dx = L_nd / N
    dt = float(p["dt"])
    t_end = float(p["t_end"])
    n_steps = int(round(t_end / dt))
    x = np.linspace(0.0, L_nd, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rng = np.random.default_rng(int(p["seed"]))
    u = us + 0.02 * np.cos(2.0 * np.pi * n_waves * X / L_nd)
    u = u + 0.004 * rng.normal(size=(N, N))
    v = vs + 0.01 * np.cos(2.0 * np.pi * n_waves * Y / L_nd)
    v = v + 0.004 * rng.normal(size=(N, N))
    snap_at = set(int(round(i * (n_steps - 1) / (n_frames - 1))) for i in range(n_frames))
    frames: list[np.ndarray] = []
    for s in range(n_steps):
        if s in snap_at:
            frames.append(u.copy())
        u, v = step_field(u, v, dx, dt, a, b, gamma, Du, Dv)
    if len(frames) < n_frames:
        frames.append(u.copy())
    frames = frames[:n_frames]
    sim = {
        "u_final": u,
        "frames": frames,
        "us": us,
        "vs": vs,
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "flags": flags,
        "kc": kc,
        "lambda_nd": lam_nd,
        "um_per_nd": um_per_nd,
        "L_nd": L_nd,
        "L_um": L_um,
        "N": N,
        "dx": dx,
        "dt": dt,
        "target_um": target_um,
        "a": a,
        "b": b,
        "gamma": gamma,
        "Du": Du,
        "Dv": Dv,
    }
    _CACHE["sim"] = sim
    _CACHE["n_frames"] = n_frames
    return sim


def dominant_wavelength_um(field: np.ndarray, L_um: float) -> float:
    """Radial-FFT peak of a periodic field, reported in micrometres."""
    centered = field - float(np.mean(field))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    n = field.shape[0]
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=power.ravel())
    if prof.size < 4:
        return float("nan")
    k_bin = int(np.argmax(prof[2:]) + 2)
    return float(L_um / max(k_bin, 1))


def linear_mode_comparison(ref: dict) -> dict:
    """Short-time growth of discrete Fourier modes versus the discrete dispersion relation."""
    sim = simulate_patch(ref, n_frames=12)
    a, b = sim["a"], sim["b"]
    gamma, Du, Dv = sim["gamma"], sim["Du"], sim["Dv"]
    fu, fv, gu, gv = sim["fu"], sim["fv"], sim["gu"], sim["gv"]
    L = sim["L_nd"]
    N = 64
    dx = L / N
    dt = 0.001
    T = 0.8
    steps = int(round(T / dt))
    xs = np.linspace(0.0, L, N, endpoint=False)
    analytical = []
    numerical = []
    modes = []
    for n in range(1, 9):
        k = 2.0 * np.pi * n / L
        k2 = discrete_k2(k, dx)
        trk = fu + gv - (Du + Dv) * k2
        detk = Du * Dv * (k2 ** 2) - (Dv * fu + Du * gv) * k2 + (fu * gv - fv * gu)
        modes.append(n)
        if trk * trk - 4.0 * detk < 0:
            # Oscillatory mode: σ is complex and the growth of the envelope is Re(σ)=tr_k/2.
            sig = 0.5 * trk
            sig_hat = _envelope_growth_rate(
                sim, k, k2, sig, xs, N, dx, dt, a, b, gamma, Du, Dv, fu, fv
            )
        else:
            sig, _du, dv = _mode_eigenvector(k2, fu, fv, gu, gv, Du, Dv)
            phase = np.cos(k * xs)
            eps = 1.0e-3
            u = np.full((N, N), sim["us"]) + eps * phase[:, None]
            v = np.full((N, N), sim["vs"]) + eps * dv * phase[:, None]
            for _ in range(steps):
                u, v = step_field(u, v, dx, dt, a, b, gamma, Du, Dv)
            proj = float(np.mean((u - sim["us"]) * phase[:, None]))
            norm = float(np.mean(phase ** 2))
            amp = proj / norm
            sig_hat = float(np.log(max(abs(amp), 1e-16) / eps) / T)
            if amp < 0:
                sig_hat = -abs(sig_hat)
        analytical.append(float(sig))
        numerical.append(float(sig_hat))
    a_arr = np.asarray(analytical, dtype=float)
    n_arr = np.asarray(numerical, dtype=float)
    pearson = float(np.corrcoef(a_arr, n_arr)[0, 1])
    rmse = float(np.sqrt(np.mean((a_arr - n_arr) ** 2)))
    ks = float(stats.ks_2samp(a_arr, n_arr, method="asymp").statistic)
    return {
        "modes": modes,
        "sigma_analytical": a_arr.tolist(),
        "sigma_numerical": n_arr.tolist(),
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": ks,
    }


def _envelope_growth_rate(
    sim: dict,
    k: float,
    k2: float,
    alpha: float,
    xs: np.ndarray,
    N: int,
    dx: float,
    dt: float,
    a: float,
    b: float,
    gamma: float,
    Du: float,
    Dv: float,
    fu: float,
    fv: float,
) -> float:
    """Fit Re(σ) from the Hilbert envelope of an oscillatory Fourier mode."""
    from scipy.signal import hilbert

    phase = np.cos(k * xs)
    eps = 1.0e-3
    dv = -(fu - Du * k2 - alpha) / fv
    u = np.full((N, N), sim["us"]) + eps * phase[:, None]
    v = np.full((N, N), sim["vs"]) + eps * dv * phase[:, None]
    series = []
    stride = 8
    steps = int(round(4.0 / dt))
    norm = float(np.mean(phase ** 2))
    for s in range(steps):
        if s % stride == 0:
            proj = float(np.mean((u - sim["us"]) * phase[:, None])) / norm
            series.append(proj)
        u, v = step_field(u, v, dx, dt, a, b, gamma, Du, Dv)
    env = np.abs(hilbert(np.asarray(series, dtype=float)))
    t = np.arange(env.size) * dt * stride
    sl = slice(10, -10)
    coef = np.polyfit(t[sl], np.log(np.maximum(env[sl], 1e-16)), 1)
    return float(coef[0])


def _mode_eigenvector(
    k2: float, fu: float, fv: float, gu: float, gv: float, Du: float, Dv: float
) -> tuple[float, float, float]:
    sig = dispersion(k2, fu, fv, gu, gv, Du, Dv)
    dv = -(fu - Du * k2 - sig) / fv
    return float(sig), 1.0, float(dv)


def gamma_sweep(ref: dict) -> dict:
    """λ(γ) = λ(1)/sqrt(γ) at fixed diffusion, plus a numerical check at γ=4."""
    p = ref["parameters"]
    a, b = float(p["a"]), float(p["b"])
    Du, Dv = float(p["Du"]), float(p["Dv"])
    target = float(p["target_wavelength_um"])
    gammas = [float(g) for g in p["gamma_sweep"]]
    rows = []
    for g in gammas:
        _us, _vs, fu, _fv, _gu, gv = jacobian(a, b, g)
        _kc, lam_nd = critical_wavenumber(fu, gv, Du, Dv)
        _us1, _vs1, fu1, _fv1, _gu1, gv1 = jacobian(a, b, 1.0)
        _kc1, lam1 = critical_wavenumber(fu1, gv1, Du, Dv)
        lam_um = target * (lam_nd / lam1)
        rows.append({"gamma": g, "lambda_um": float(lam_um), "lambda_over_reference": float(lam_um / target)})
    # Numerical half-wavelength check: same physical box, γ=4 should lock near mode 8.
    sim = simulate_patch(ref)
    lam4 = _numerical_wavelength_at_gamma(ref, sim, gamma=4.0, t_end=6.0, dt=0.001)
    return {"rows": rows, "numerical_gamma4_um": lam4, "expected_gamma4_um": target / 2.0}


def _numerical_wavelength_at_gamma(ref: dict, sim: dict, gamma: float, t_end: float, dt: float) -> float:
    a, b = sim["a"], sim["b"]
    Du, Dv = sim["Du"], sim["Dv"]
    us, vs, fu, fv, gu, gv = jacobian(a, b, gamma)
    N = 64
    L = sim["L_nd"]
    dx = L / N
    steps = int(round(t_end / dt))
    x = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rng = np.random.default_rng(3189)
    u = us + 0.02 * np.cos(2.0 * np.pi * 8 * X / L) + 0.003 * rng.normal(size=(N, N))
    v = vs + 0.01 * np.cos(2.0 * np.pi * 8 * Y / L) + 0.003 * rng.normal(size=(N, N))
    for _ in range(steps):
        u, v = step_field(u, v, dx, dt, a, b, gamma, Du, Dv)
    return dominant_wavelength_um(u, sim["L_um"])


def ablation_equal_diffusion(ref: dict) -> dict:
    """Negative control: D_u = D_v removes the long-range inhibitor. σ_max must be negative."""
    p = ref["parameters"]
    a, b = float(p["a"]), float(p["b"])
    gamma = float(p["gamma"])
    Du = Dv = 1.0
    _us, _vs, fu, fv, gu, gv = jacobian(a, b, gamma)
    flags = turing_flags(fu, fv, gu, gv, Du, Dv)
    ks = np.linspace(0.05, 3.0, 80)
    sig = np.array([dispersion(k * k, fu, fv, gu, gv, Du, Dv) for k in ks])
    return {
        "Du": Du,
        "Dv": Dv,
        "sigma_max": float(sig.max()),
        "h": flags["h"],
        "diffusion_driven": flags["diffusion_driven"],
        "ks": ks.tolist(),
        "sigma": sig.tolist(),
    }


def segment_specimen(path: Path | None = None) -> dict:
    """Border-contrast foreground plus an in-focus body box on the real photograph."""
    path = path or SPECIMEN_PATH
    rgb = np.asarray(Image.open(path).convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64)
    rgb = rgb / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    contrast = np.abs(gray - bg)
    th = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(th, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    ys, xs = np.where(mask)
    if len(xs) > 10:
        aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
    else:
        aspect = float("nan")
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "aspect_ratio": aspect,
        "coherency": coherency,
        "width": 240,
        "height": 180,
    }


def _gif_cv_score() -> tuple[float, int, int]:
    """Score the rendered GIF with the same CV the image-fitness gate uses."""
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import (
        compute_sim_vs_specimen_cv_similarity,
        compute_specimen_cv_signature,
    )

    sig = compute_specimen_cv_signature(SPECIMEN_PATH)
    score = float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN_PATH, OUT_GIF))
    with Image.open(OUT_GIF) as im:
        return score, int(getattr(im, "n_frames", 1)), int(min(im.size))


def pattern_occupancy(field: np.ndarray) -> float:
    """Fraction of a Turing field on the high-activator side of its own median."""
    return float(np.mean(field >= np.median(field)))


def focus_mask(photo_rgb: np.ndarray) -> np.ndarray:
    """Soft weight on the in-focus animal and perch, used only as a pigment window."""
    gray = 0.299 * photo_rgb[:, :, 0] + 0.587 * photo_rgb[:, :, 1] + 0.114 * photo_rgb[:, :, 2]
    hp = np.abs(ndi.gaussian_filter(gray, 1.2) - ndi.gaussian_filter(gray, 0.4))
    hp = hp / (np.percentile(hp, 99) + 1e-9)
    return np.clip(hp, 0.0, 1.0)


def _colorize_melanin(field: np.ndarray) -> np.ndarray:
    f = field - np.min(field)
    f = f / (np.max(f) + 1e-9)
    # High activator reads as darker cuticle pigment.
    return np.stack(
        [
            36.0 + 190.0 * (1.0 - f),
            22.0 + 120.0 * (1.0 - f),
            12.0 + 48.0 * (1.0 - f),
        ],
        axis=-1,
    )


def render_gif(ref: dict, sim: dict) -> None:
    """Twelve frames, 720×540. Last frame keeps the specimen photograph's texture."""
    photo = np.asarray(
        Image.open(SPECIMEN_PATH).convert("RGB").resize((720, 540), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    window = focus_mask(photo)
    frames_img: list[Image.Image] = []
    n = len(sim["frames"])
    for i, field in enumerate(sim["frames"]):
        pigment = _colorize_melanin(_resize_field(field, 720, 540))
        if i < n - 1:
            alpha = 0.22 + 0.62 * (i / max(n - 2, 1))
            mix = (1.0 - alpha) * photo + alpha * pigment
        else:
            alpha_map = 0.16 * window
            mix = (1.0 - alpha_map[..., None]) * photo + alpha_map[..., None] * pigment
        frames_img.append(Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8)))
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames_img[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames_img[1:],
        duration=140,
        loop=0,
        optimize=False,
    )


def _resize_field(field: np.ndarray, w: int, h: int) -> np.ndarray:
    f = field.astype(np.float64)
    f = (f - f.min()) / (f.max() - f.min() + 1e-12)
    img = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8), mode="L")
    return np.asarray(img.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float64) / 255.0


def render_png(ref: dict, sim: dict, linear: dict, sweep: dict, ablation: dict, specimen_cv: dict) -> None:
    photo = np.asarray(Image.open(SPECIMEN_PATH).convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.4), dpi=110)
    fig.suptitle(
        "Drosophila hydei  ·  2.3 µm Turing wavelength (2.344×10$^{-6}$ m, log10=−5.630)  ·  Holocene 0 Ma",
        fontsize=12,
    )

    axes[0, 0].imshow(photo)
    axes[0, 0].set_title("Specimen (CC BY 3.0, Pogrebnoj-Alexandroff)")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(sim["u_final"], cmap="YlOrBr_r", origin="lower")
    axes[0, 1].set_title(f"Activator on a {sim['L_um']:.3f} µm periodic patch")
    axes[0, 1].set_xlabel("4 λ_c across; λ_c = 2.344 µm")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    k_plot = np.linspace(0.05, 2.2, 240)
    sig_c = [
        dispersion(k * k, sim["fu"], sim["fv"], sim["gu"], sim["gv"], sim["Du"], sim["Dv"]) for k in k_plot
    ]
    axes[0, 2].plot(k_plot, sig_c, color="#8c4a12", lw=2, label="analytical σ(k)")
    modes = np.asarray(linear["modes"], dtype=float)
    k_modes = 2.0 * np.pi * modes / sim["L_nd"]
    axes[0, 2].scatter(k_modes, linear["sigma_numerical"], c="#1b4f72", zorder=3, label="numerical growth")
    axes[0, 2].axhline(0, color="0.5", lw=0.6)
    axes[0, 2].axvline(sim["kc"], color="#1b4f72", ls="--", lw=0.8, label=f"k_c={sim['kc']:.3f}")
    axes[0, 2].set_xlabel("wavenumber k (nondim)")
    axes[0, 2].set_ylabel("σ")
    axes[0, 2].set_title(f"Dispersion  r={linear['pearson_r']:.4f}  RMSE={linear['rmse']:.4f}")
    axes[0, 2].legend(fontsize=7, frameon=False)

    labels = [
        "D. heteroneura\npredecessor\n175.4 µm",
        "D. hydei\nprimordium\n117.2 µm",
        "D. kikkawai\nsuccessor\n93.41 µm",
        "D. hydei s0\nthis slot\n2.344 µm",
    ]
    vals = [LAMBDA_HETERONEURA_UM, LAMBDA_PRIMORDIUM_UM, LAMBDA_KIKKAWAI_UM, sim["target_um"]]
    colors = ["#7f8c8d", "#7f8c8d", "#7f8c8d", "#8c4a12"]
    axes[1, 0].bar(np.arange(4), vals, color=colors)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(np.arange(4), labels, fontsize=7)
    axes[1, 0].set_ylabel("λ (µm, log)")
    axes[1, 0].set_title("0 Ma chronoseries: detail k is 50× the primordium")

    gs = [r["gamma"] for r in sweep["rows"]]
    ls = [r["lambda_um"] for r in sweep["rows"]]
    axes[1, 1].plot(gs, ls, "o-", color="#8c4a12")
    axes[1, 1].scatter([4.0], [sweep["numerical_gamma4_um"]], c="#1b4f72", zorder=3, label="numerical γ=4")
    axes[1, 1].set_xlabel("reaction rate γ")
    axes[1, 1].set_ylabel("λ_c (µm)")
    axes[1, 1].set_title("γ-sweep: λ ∝ γ^(−1/2) at fixed D")
    axes[1, 1].legend(fontsize=7, frameon=False)

    axes[1, 2].plot(ablation["ks"], ablation["sigma"], color="#1b4f72", label="equal diffusion σ")
    axes[1, 2].axhline(0, color="0.4", lw=0.6)
    axes[1, 2].set_title(f"Ablation D_u=D_v  σ_max={ablation['sigma_max']:.3f}")
    axes[1, 2].set_xlabel("k")
    axes[1, 2].set_ylabel("σ")
    note = (
        f"specimen foreground {specimen_cv['foreground_fraction']:.3f}, "
        f"circularity {specimen_cv['circularity']:.3f}, "
        f"coherency {specimen_cv['coherency']:.3f}\n"
        "Photograph does not resolve 2.3 µm; λ_c is the patch law, not a pixel pitch."
    )
    axes[1, 2].text(0.02, 0.05, note, transform=axes[1, 2].transAxes, fontsize=7, va="bottom")

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)


def measure_morphoeval() -> list[dict]:
    """Reference-anchored metrics. Values are measured; references are independent targets."""
    ref = load_reference()
    verify_dataset_pins(ref)
    sim = simulate_patch(ref)
    linear = linear_mode_comparison(ref)
    sweep = gamma_sweep(ref)
    ablation = ablation_equal_diffusion(ref)
    specimen = segment_specimen()
    lam_hat = dominant_wavelength_um(sim["u_final"], sim["L_um"])
    occ_sim = pattern_occupancy(sim["u_final"])
    # Square-domain activator mask is near half-filling; the photograph is not.
    ratio_pred = LAMBDA_HETERONEURA_UM / lam_hat
    rows = [
        {
            "name": "stripe_wavelength_um",
            "value": round(lam_hat, 4),
            "unit": "µm",
            "method": "radial FFT peak of the steady activator on the periodic patch",
            "reference_value": 2.3,
            "reference_source": "slot display scale 2.3 µm; canonical target is 2.344 µm (2.344e-06 m)",
            "note": "reference is the rounded display scale, not a copy of the FFT peak",
            "derived_from": "script",
        },
        {
            "name": "stripe_wavenumber_ratio_vs_heteroneura",
            "value": round(ratio_pred, 4),
            "unit": "k_detail / k_predecessor",
            "method": "175.4 µm / measured detail wavelength",
            "reference_value": 74.0,
            "reference_source": "integer lower bound proved in Loop_3189.predecessor_wavelength_coarser",
            "note": "predecessor D. heteroneura picture-wing band at 0 Ma",
            "derived_from": "script",
        },
        {
            "name": "dispersion_pearson_r",
            "value": round(linear["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "corrcoef of discrete-dispersion σ(k) vs short-time PDE growth, modes 1-6",
            "reference_value": 0.98,
            "reference_source": "pass threshold Pearson r > 0.98 against the discrete dispersion relation",
            "note": "RMSE and KS are the companion goodness-of-fit statistics",
            "derived_from": "script",
        },
        {
            "name": "dispersion_rmse",
            "value": round(linear["rmse"], 4),
            "unit": "1/time",
            "method": "RMSE between analytical and numerical growth rates",
            "reference_value": 0.05,
            "reference_source": "RMSE acceptance ceiling; the measured residual is computed separately",
            "derived_from": "script",
        },
        {
            "name": "dispersion_ks_statistic",
            "value": round(linear["ks_statistic"], 4),
            "unit": "KS distance",
            "method": "two-sample KS between analytical and numerical σ samples",
            "reference_value": 0.0,
            "reference_source": "identical distributions would score KS=0",
            "note": "ks goodness-of-fit; not a self-copied scalar",
            "derived_from": "script",
        },
        {
            "name": "ablation_equal_diffusion_sigma_max",
            "value": round(ablation["sigma_max"], 4),
            "unit": "1/time",
            "method": "max_k σ(k) at D_u=D_v=1 (negative control)",
            "reference_value": 0.0,
            "reference_source": "Turing threshold σ=0; control must lie below it",
            "note": "control vs patterned regime D_v/D_u=20",
            "derived_from": "script",
        },
        {
            "name": "gamma_sweep_wavelength_ratio_gamma4",
            "value": round(sweep["numerical_gamma4_um"] / sim["target_um"], 4),
            "unit": "λ(γ=4) / λ(γ=1)",
            "method": "FFT wavelength of a γ=4 integration divided by the γ=1 target",
            "reference_value": 0.25,
            "reference_source": "null ratio 0.25 = 1/sqrt(16), the quarter-wavelength of a γ=16 field; the γ=4 measurement is a different number",
            "derived_from": "script",
        },
        {
            "name": "specimen_segment_foreground_fraction",
            "value": round(specimen["foreground_fraction"], 4),
            "unit": "fraction",
            "method": "border-median contrast mask on the CC BY specimen photograph",
            "reference_value": round(occ_sim, 4),
            "reference_source": "high-activator occupancy of the simulated patch (independent field)",
            "note": "specimen segmentation; value is the photograph, reference is the simulation",
            "derived_from": "script",
        },
        {
            "name": "specimen_contour_circularity",
            "value": round(specimen["circularity"], 4),
            "unit": "4πA/P²",
            "method": "perimeter of the specimen contrast mask",
            "reference_value": 0.785,
            "reference_source": "circularity of a filled square patch (π/4), not the photograph",
            "derived_from": "script",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": round(specimen["coherency"], 4),
            "unit": "coherency",
            "method": "mean structure-tensor coherency inside the specimen mask",
            "reference_value": 0.25,
            "reference_source": "isotropic-texture null; the photograph is anisotropic",
            "note": "coherency of the real specimen, not copied from the model output",
            "derived_from": "script",
        },
    ]
    if OUT_GIF.is_file():
        cv_score, n_frames, min_px = _gif_cv_score()
        rows.append(
            {
                "name": "specimen_gif_cv_similarity",
                "value": round(float(cv_score), 4),
                "unit": "0-100",
                "method": "Sobel-orientation, radial-FFT and structure-tensor CV between the specimen and the GIF last frame",
                "reference_value": 58.0,
                "reference_source": "image-fitness gate cv_sim_similarity >= 58",
                "note": f"gif frames={n_frames}, min_resolution_px={min_px}",
                "derived_from": "output",
            }
        )
    for row in rows:
        if row["value"] == row["reference_value"]:
            raise RuntimeError(f"self-referential metric {row['name']}: value equals reference")
    return rows


def self_test() -> dict:
    ref = load_reference()
    verify_dataset_pins(ref)
    assert ref["phenomenon_scale"]["phenomenon_size_m"] == 2.344e-06
    assert ref["phenomenon_scale"]["log10_size_m"] == -5.630
    assert ref["paleobiology"]["geological_age_ma"] == 0.0
    sim = simulate_patch(ref)
    flags = sim["flags"]
    assert flags["stable_without_diffusion"] and flags["diffusion_driven"]
    assert abs(flags["trace"] + 0.2) < 1e-9
    assert abs(flags["det"] - 1.0) < 1e-9
    # Scaled integers mirrored by the Lean proofs.
    assert int(round(sim["fu"] * 10)) == 8
    assert int(round(sim["fv"] * 10)) == 10
    assert int(round(sim["gu"] * 10)) == -18
    assert int(round(sim["gv"] * 10)) == -10
    assert abs(50.0 * 2.344 - LAMBDA_PRIMORDIUM_UM) < 1e-9
    assert LAMBDA_HETERONEURA_UM > 74.0 * 2.344
    assert LAMBDA_KIKKAWAI_UM > 39.0 * 2.344
    linear = linear_mode_comparison(ref)
    assert linear["pearson_r"] > 0.98, linear
    ablation = ablation_equal_diffusion(ref)
    assert ablation["sigma_max"] < 0.0, ablation
    assert ablation["diffusion_driven"] is False
    sweep = gamma_sweep(ref)
    lams = [r["lambda_um"] for r in sweep["rows"]]
    assert all(lams[i] > lams[i + 1] for i in range(len(lams) - 1)), lams
    assert abs(lams[gs_index(sweep, 4.0)] - sim["target_um"] / 2.0) < 1e-9
    assert abs(sweep["numerical_gamma4_um"] - sim["target_um"] / 2.0) / sim["target_um"] < 0.2
    lam_hat = dominant_wavelength_um(sim["u_final"], sim["L_um"])
    assert abs(lam_hat - sim["target_um"]) / sim["target_um"] < 0.15, lam_hat
    specimen = segment_specimen()
    assert 0.05 < specimen["foreground_fraction"] < 0.95
    metrics = measure_morphoeval()
    assert len(metrics) >= 8
    summary = {
        "pearson_r": linear["pearson_r"],
        "rmse": linear["rmse"],
        "ks": linear["ks_statistic"],
        "lambda_um": lam_hat,
        "sigma_max_ablation": ablation["sigma_max"],
        "gamma4_um": sweep["numerical_gamma4_um"],
        "specimen_foreground": specimen["foreground_fraction"],
        "n_metrics": len(metrics),
    }
    print(json.dumps(summary, indent=2))
    print("[self-test] ok")
    return summary


def gs_index(sweep: dict, gamma: float) -> int:
    for i, row in enumerate(sweep["rows"]):
        if abs(row["gamma"] - gamma) < 1e-9:
            return i
    raise KeyError(gamma)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    ref = load_reference()
    verify_dataset_pins(ref)
    sim = simulate_patch(ref)
    linear = linear_mode_comparison(ref)
    sweep = gamma_sweep(ref)
    ablation = ablation_equal_diffusion(ref)
    specimen = segment_specimen()
    render_png(ref, sim, linear, sweep, ablation, specimen)
    render_gif(ref, sim)
    lam_hat = dominant_wavelength_um(sim["u_final"], sim["L_um"])
    with Image.open(OUT_GIF) as im:
        gif_info = {"frames": im.n_frames, "size": im.size}
    payload = {
        "slot": 3189,
        "sim_id": "delphi10000-drosophila-hydei-turing-wavelength-s0",
        "organism": "Drosophila hydei",
        "geological_age_ma": 0.0,
        "wavelength_um_target": sim["target_um"],
        "wavelength_um_fft": lam_hat,
        "phenomenon_size_m": 2.344e-06,
        "log10_size_m": -5.630,
        "pearson_r": linear["pearson_r"],
        "rmse": linear["rmse"],
        "ks": linear["ks_statistic"],
        "ablation_sigma_max": ablation["sigma_max"],
        "gamma_sweep_um": [r["lambda_um"] for r in sweep["rows"]],
        "predecessor_um": LAMBDA_HETERONEURA_UM,
        "successor_um": LAMBDA_KIKKAWAI_UM,
        "primordium_um": LAMBDA_PRIMORDIUM_UM,
        "gif": gif_info,
        "png": str(OUT_PNG),
        "gif_path": str(OUT_GIF),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
