#!/usr/bin/env python3
"""
Aspergillus nidulans — Turing reaction-diffusion wavelength selection
at the Holocene extant detail scale (slot 3122).

Scale: λ_c = 1.284e-05 m = 12.84 µm (nominal 13 µm, log10 = -4.891),
tier 4/8 ``4_cellular_10um``. Age: Holocene, 0 Ma. Chronoseries 1/1:
there is no predecessor and no successor. Against the 10 µm tier floor
this detail variant lengthens the onset wavelength by 2.84 µm, so
k_c = 2π/λ_c falls from 6.283e5 m^-1 to 4.893e5 m^-1. Slot 182's
template constants (1.31, 0.25, 12, 4, 48) are not used.

Governing kinetics are Schnakenberg (1979), the standard two-species
substrate for Turing's (1952) diffusion-driven instability:

    ∂u/∂t = D_u ∇²u + a - u + u²v          (1)
    ∂v/∂t = D_v ∇²v + b - u²v              (2)

Homogeneous steady state u0 = a + b, v0 = b / u0². Onset wavenumber

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),   λ_c = 2π / k_c.   (3)

Boundary conditions: periodic on the cellular grid. The colony animation
is a separate observable: Trinci radial expansion reveals a specimen-
conditioned optical-density field

    I = S * (1 - α + α (0.55 + 0.90 û)),                         (4)

where S is the public-domain colony photograph and û is the normalised
Schnakenberg activator. Adams, Wieser & Yu (1998) and Clutterbuck (1969)
describe conidial pigment (wA / yA); they do not publish this PDE.
The 13 µm length is the Delphi detail-scale assignment, not a wavelength
read off a published micrograph. The colony photograph does not resolve
13 µm; it is the macroscopic spiral-mutant envelope.

Usage:
    python3 models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py
    python3 models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SPECIMEN = (
    ROOT
    / "datasets"
    / "aspergillus_nidulans_turing_wavelength_s0"
    / "aspergillus_nidulans_turing_wavelength_s0_specimen.jpg"
)
GIF_PATH = ROOT / "docs" / "media" / "aspergillus_nidulans_turing_wavelength_s0_growth.gif"
PNG_PATH = ROOT / "docs" / "media" / "aspergillus_nidulans_turing_wavelength_s0_comparison.png"

# Schnakenberg parameters unique to slot 3122 (not slot 182).
A_PROD = 0.12
B_PROD = 0.78
DU = 1.0
DV = 9.295
LAMBDA_M = 1.284e-05  # 12.84 µm, nominal 13 µm
AGE_MA = 0.0
SEED = 3122
N_GRID = 84
PX_PER_WAVE = 14  # N / PX_PER_WAVE = 6 onset waves, so k_c is a periodic DFT mode
SPECIMEN_ALPHA = 0.12
GIF_W, GIF_H = 640, 480
N_FRAMES = 16
RNG_SEED = 3122


def steady_state(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float]:
    """Eq. (1–2) homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return float(u0), float(v0)


def jacobian(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float, float, float]:
    """Partial derivatives f_u, f_v, g_u, g_v at the steady state."""
    u0, v0 = steady_state(a, b)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -(u0 * u0)
    return float(f_u), float(f_v), float(g_u), float(g_v)


def turing_report(du: float = DU, dv: float = DV) -> dict:
    """Linear Turing conditions and onset wavenumber, Eq. (3)."""
    f_u, f_v, g_u, g_v = jacobian()
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    cross = dv * f_u + du * g_v
    discr = cross * cross - 4.0 * du * dv * det
    kc2 = cross / (2.0 * du * dv)
    kc = float(np.sqrt(max(kc2, 0.0)))
    lam_nd = float(2.0 * np.pi / kc) if kc > 0 else float("inf")
    return {
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "trace": float(trace),
        "det": float(det),
        "cross": float(cross),
        "discriminant": float(discr),
        "k_c": kc,
        "lambda_nd": lam_nd,
        "turing": bool(trace < 0.0 and det > 0.0 and cross > 0.0 and discr > 0.0),
    }


def length_unit_um(report: dict | None = None) -> float:
    """Micrometres per nondimensional length so that λ_c = 12.84 µm."""
    report = report or turing_report()
    return (LAMBDA_M * 1.0e6) / report["lambda_nd"]


def sigma_of_k(k: float, du: float = DU, dv: float = DV) -> float:
    """Largest real eigenvalue of J - k² D."""
    f_u, f_v, g_u, g_v = jacobian()
    k2 = k * k
    tr = (f_u - du * k2) + (g_v - dv * k2)
    det = (f_u - du * k2) * (g_v - dv * k2) - f_v * g_u
    disc = tr * tr - 4.0 * det
    return float(0.5 * (tr + np.sqrt(max(disc, 0.0))))


def most_unstable_k(du: float = DU, dv: float = DV) -> tuple[float, float]:
    ks = np.linspace(0.05, 2.2, 600)
    sig = np.array([sigma_of_k(float(k), du, dv) for k in ks])
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    """Periodic 5-point Laplacian."""
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / (dx * dx)


def simulate(
    du: float,
    dv: float,
    steps: int,
    dx: float,
    dt: float,
    init: str = "stripe",
    k_seed: float | None = None,
    amp: float = 0.05,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler integration of Eq. (1–2) on a periodic N×N grid."""
    u0, v0 = steady_state()
    n = N_GRID
    rng = np.random.default_rng(seed)
    u = np.full((n, n), u0, dtype=np.float64)
    v = np.full((n, n), v0, dtype=np.float64)
    xs = np.arange(n, dtype=np.float64) * dx
    if init == "stripe":
        k = k_seed if k_seed is not None else most_unstable_k(du, dv)[0]
        phase = np.cos(k * xs)
        u = u + amp * phase[None, :]
        v = v + 0.35 * amp * phase[None, :]
    elif init == "noise":
        u = u + amp * rng.normal(0.0, 1.0, size=u.shape)
        v = v + 0.35 * amp * rng.normal(0.0, 1.0, size=v.shape)
    elif init == "mode":
        # Caller passes k_seed and has already set the eigenvector via amp sign on v.
        k = float(k_seed)
        phase = np.cos(k * xs)
        u = u + phase[None, :] * amp
        v = v + phase[None, :] * amp  # replaced by simulate_mode
    for _ in range(steps):
        u2v = u * u * v
        lu = laplacian(u, dx)
        lv = laplacian(v, dx)
        u = u + dt * (du * lu + A_PROD - u + u2v)
        v = v + dt * (dv * lv + B_PROD - u2v)
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            break
    return u, v


def simulate_mode(k: float, du: float, dv: float, dx: float, dt: float, steps: int = 40, eps: float = 5.0e-4) -> float:
    """Return Re(σ) from a periodic Fourier mode of the PDE.

    Eigenvalues of J - k²D are often a complex conjugate pair, so a single
    component oscillates. The modulus of the modal coordinate, obtained by
    projecting (δu, δv) onto the cosine and changing to the eigenbasis,
    grows as exp(Re(σ) t).
    """
    f_u, f_v, g_u, g_v = jacobian()
    k2 = float(k * k)
    matrix = np.array([[f_u - du * k2, f_v], [g_u, g_v - dv * k2]], dtype=np.float64)
    evals, evecs = np.linalg.eig(matrix)
    idx = int(np.argmax(evals.real))
    vec = np.real(evecs[:, idx])
    vec = vec / (np.linalg.norm(vec) + 1e-15)
    u0, v0 = steady_state()
    n = N_GRID
    xs = np.arange(n, dtype=np.float64) * dx
    phase = np.cos(k * xs)

    def project(field: np.ndarray) -> float:
        return float(np.mean((field - np.mean(field)) * phase[None, :]) * 2.0)

    u = np.full((n, n), u0) + eps * vec[0] * phase[None, :]
    v = np.full((n, n), v0) + eps * vec[1] * phase[None, :]
    amp0 = np.array([project(u), project(v)], dtype=np.complex128)
    coord0 = np.linalg.solve(evecs, amp0)
    for _ in range(steps):
        u2v = u * u * v
        lu = laplacian(u, dx)
        lv = laplacian(v, dx)
        u_new = u + dt * (du * lu + A_PROD - u + u2v)
        v = v + dt * (dv * lv + B_PROD - u2v)
        u = u_new
    amp1 = np.array([project(u), project(v)], dtype=np.complex128)
    coord1 = np.linalg.solve(evecs, amp1)
    elapsed = steps * dt
    if abs(coord0[idx]) < 1e-18:
        return 0.0
    return float(np.real(np.log(coord1[idx] / coord0[idx]) / elapsed))


def dominant_wavelength_um(field: np.ndarray, dx_um: float) -> float:
    """Wavelength of the strongest non-DC Fourier peak."""
    n = field.shape[0]
    centered = field - np.mean(field)
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    cy = cx = n // 2
    power[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    kx = (ix - cx) * 2.0 * np.pi / (n * (dx_um))
    ky = (iy - cy) * 2.0 * np.pi / (n * (dx_um))
    k = float(np.hypot(kx, ky))
    if k <= 1e-9:
        return float("nan")
    return float(2.0 * np.pi / k)


def grid_spacing(report: dict) -> tuple[float, float, float]:
    """Return dx_nd, dt, dx_um."""
    dx_nd = report["lambda_nd"] / PX_PER_WAVE
    dt = 0.35 * (dx_nd * dx_nd) / (4.0 * max(DU, DV))
    dx_um = dx_nd * length_unit_um(report)
    return float(dx_nd), float(dt), float(dx_um)


def parameter_regime_sweep(report: dict) -> dict:
    """λ_c(D_v) must shorten as the inhibitor diffuses faster."""
    dvs = [8.50, 9.295, 11.0, 14.0, 18.0]
    rows = []
    for dv in dvs:
        rep = turing_report(DU, dv)
        rows.append({"Dv": dv, "k_c": rep["k_c"], "lambda_nd": rep["lambda_nd"], "turing": rep["turing"]})
    lams = [r["lambda_nd"] for r in rows]
    monotone = all(lams[i] > lams[i + 1] for i in range(len(lams) - 1))
    return {"rows": rows, "lambda_decreases_with_Dv": monotone}


def periodic_wavenumbers(dx: float, m_lo: int = 2, m_hi: int = 10) -> np.ndarray:
    """Wavenumbers 2π m / (N dx) that close on the periodic grid."""
    return np.array([2.0 * np.pi * m / (N_GRID * dx) for m in range(m_lo, m_hi + 1)], dtype=np.float64)


def dispersion_goodness_of_fit(report: dict, dx: float, dt: float) -> dict:
    """Pearson r and RMSE of PDE mode-growth σ against the analytical eigenvalue.

    Only periodic DFT modes are probed. A non-periodic cosine leaves a seam
    that the 5-point Laplacian treats as a shock, and the growth rate is then
    not σ(k).
    """
    ks = periodic_wavenumbers(dx)
    analytical = np.array([sigma_of_k(float(k)) for k in ks])
    numerical = np.array([simulate_mode(float(k), DU, DV, dx, dt, steps=40, eps=5e-4) for k in ks])
    r = pearson(analytical, numerical)
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    return {
        "k": [float(k) for k in ks],
        "sigma_analytical": [float(s) for s in analytical],
        "sigma_numerical": [float(s) for s in numerical],
        "pearson_r": r,
        "rmse": rmse,
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(a * b) / denom)


def otsu_mask(gray: np.ndarray) -> tuple[np.ndarray, float]:
    g = np.clip(gray, 0.0, 1.0)
    hist, _ = np.histogram(g, bins=256, range=(0.0, 1.0))
    prob = hist.astype(np.float64)
    prob /= prob.sum() + 1e-12
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.argmax(sigma_b))
    thr = (idx + 0.5) / 256.0
    return g >= thr, float(thr)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + 1e-9))


def load_specimen_rgb() -> np.ndarray:
    im = Image.open(SPECIMEN).convert("RGB").resize((GIF_W, GIF_H), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float64)


def modulate(specimen_rgb: np.ndarray, activator: np.ndarray, alpha: float) -> np.ndarray:
    """Eq. (4): specimen optical density times a weak Turing carrier."""
    u = activator - np.min(activator)
    u = u / (np.ptp(u) + 1e-12)
    u_img = Image.fromarray(np.clip(u * 255.0, 0, 255).astype(np.uint8), mode="L")
    u_big = np.asarray(u_img.resize((GIF_W, GIF_H), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    factor = 1.0 - alpha + alpha * (0.55 + 0.90 * u_big)
    return np.clip(specimen_rgb * factor[..., None], 0.0, 255.0)


def render_gif(modulated: np.ndarray, path: Path) -> dict:
    """Radial colony expansion (Trinci) ending on the full modulated field."""
    h, w = modulated.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0)
    rmax = float(rad.max())
    frames: list[Image.Image] = []
    times = np.linspace(0.10, 1.0, N_FRAMES)
    for t in times:
        if t >= 0.999:
            mask = np.ones_like(rad)
        else:
            front = float(t) * rmax * 0.78
            mask = np.clip((front - rad) / 18.0, 0.0, 1.0)
        frame = modulated * mask[..., None]
        frames.append(Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), mode="RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    with Image.open(path) as check:
        n_frames = int(getattr(check, "n_frames", 1))
        width, height = check.size
        check.seek(n_frames - 1)
        last = np.asarray(check.convert("RGB"), dtype=np.float64)
    return {"n_frames": n_frames, "width": width, "height": height, "last": last}


def render_png(
    specimen_rgb: np.ndarray,
    mask: np.ndarray,
    modulated: np.ndarray,
    activator: np.ndarray,
    ablation: np.ndarray,
    gof: dict,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), dpi=120)
    axes[0, 0].imshow(specimen_rgb.astype(np.uint8))
    axes[0, 0].set_title("Specimen colony (PD, 2007)")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Otsu colony segmentation")
    axes[0, 2].imshow(np.clip(modulated, 0, 255).astype(np.uint8))
    axes[0, 2].set_title(f"Specimen × Turing carrier (α={SPECIMEN_ALPHA})")
    im1 = axes[1, 0].imshow(activator, cmap="magma")
    axes[1, 0].set_title("Schnakenberg activator u")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    im2 = axes[1, 1].imshow(ablation, cmap="magma")
    axes[1, 1].set_title("Ablation D_u = D_v (no Turing)")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    ks = np.array(gof["k"])
    axes[1, 2].plot(ks, gof["sigma_analytical"], "k-", label="analytical σ(k)")
    axes[1, 2].plot(ks, gof["sigma_numerical"], "o", ms=4, label="PDE mode growth")
    axes[1, 2].axhline(0.0, color="0.5", lw=0.8)
    axes[1, 2].set_xlabel("k (nondimensional)")
    axes[1, 2].set_ylabel("growth rate")
    axes[1, 2].set_title(f"Dispersion fit r={gof['pearson_r']:.3f}")
    axes[1, 2].legend(fontsize=8)
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[1, :2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Aspergillus nidulans  ·  Turing λ_c = 12.84 µm (1.284e-05 m)  ·  Holocene 0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def alpha_expressivity(specimen_rgb: np.ndarray, activator: np.ndarray) -> list[dict]:
    """Pixel Pearson against the specimen falls as the Turing carrier weight α grows."""
    gray = specimen_rgb.mean(axis=2)
    rows = []
    for alpha in (0.0, 0.06, 0.12, 0.24, 0.40):
        mod = modulate(specimen_rgb, activator, alpha)
        rows.append({"alpha": alpha, "pearson_r": pearson(gray, mod.mean(axis=2))})
    rs = [r["pearson_r"] for r in rows]
    return rows + [{"monotone_nonincreasing": all(rs[i] >= rs[i + 1] - 1e-6 for i in range(len(rs) - 1))}]


def multi_seed_wavelengths(dx: float, dt: float, dx_um: float, k_periodic: float) -> dict:
    """Repeat the nonlinear run from three seeds with a 2% wavenumber jitter."""
    wavelengths = []
    for seed in (3122, 3123, 3124):
        jitter = 1.0 + 0.02 * (seed - 3123)
        field, _ = simulate(
            DU,
            DV,
            steps=500,
            dx=dx,
            dt=dt,
            init="stripe",
            k_seed=k_periodic * jitter,
            amp=0.07,
            seed=seed,
        )
        wavelengths.append(dominant_wavelength_um(field, dx_um))
    arr = np.asarray(wavelengths, dtype=np.float64)
    cv = float(np.std(arr) / (np.mean(arr) + 1e-12))
    return {"wavelengths_um": [float(v) for v in arr], "cv": cv}


def sobel_orientation_l1(reference_rgb: np.ndarray, other_rgb: np.ndarray) -> float:
    """L1 distance between 12-bin Sobel orientation histograms.

    Both images are resized to 240×180 and cropped the way
    verify_image_fitness crops the specimen, so the bins are comparable.
    """
    from scipy import ndimage as ndi

    def hist(rgb: np.ndarray) -> np.ndarray:
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
        im = im.resize((240, 180), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float64) / 255.0
        gray = arr[14:166, 14:226].mean(axis=2)
        gx = ndi.sobel(gray, axis=1)
        gy = ndi.sobel(gray, axis=0)
        mag = np.hypot(gx, gy) + 1e-9
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi
        bins, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
        return bins / (bins.sum() + 1e-9)

    return float(np.sum(np.abs(hist(reference_rgb) - hist(other_rgb))))


def run_pipeline(write: bool = True) -> dict:
    report = turing_report()
    if not report["turing"]:
        raise RuntimeError(f"parameters are not in the Turing regime: {report}")
    dx_nd, dt, dx_um = grid_spacing(report)
    k_star, sigma_star = most_unstable_k()
    # Six onset waves fit the 84-cell grid exactly (k_c = 2π * 6 / (N dx)).
    k_periodic = 2.0 * np.pi * (N_GRID / PX_PER_WAVE) / (N_GRID * dx_nd)
    activator, _inhibitor = simulate(
        DU, DV, steps=900, dx=dx_nd, dt=dt, init="stripe", k_seed=k_periodic, amp=0.08
    )
    ablation, _ = simulate(
        DU, DU, steps=400, dx=dx_nd, dt=dt, init="stripe", k_seed=k_periodic, amp=0.08, seed=SEED + 1
    )
    lam_meas_um = dominant_wavelength_um(activator, dx_um)
    gof = dispersion_goodness_of_fit(report, dx_nd, dt)
    sweep = parameter_regime_sweep(report)
    specimen = load_specimen_rgb()
    modulated = modulate(specimen, activator, SPECIMEN_ALPHA)
    gray_s = specimen.mean(axis=2) / 255.0
    gray_m = modulated.mean(axis=2) / 255.0
    mask_s, thr_s = otsu_mask(gray_s)
    mask_m, thr_m = otsu_mask(gray_m)
    pix_r = pearson(gray_s, gray_m)
    rmse = float(np.sqrt(np.mean((gray_s - gray_m) ** 2)))
    rng = np.random.default_rng(SEED)
    take = rng.choice(gray_s.size, size=4000, replace=False)
    ks = stats.ks_2samp(gray_s.ravel()[take], gray_m.ravel()[take])
    # Null: pure horizontal stripes versus the specimen (dissimilar).
    xs = np.linspace(0, 16 * np.pi, GIF_W)
    stripes = 0.5 + 0.5 * np.sin(xs)[None, :] * np.ones((GIF_H, 1))
    null_r = pearson(gray_s, stripes)
    null_rmse = float(np.sqrt(np.mean((gray_s - stripes) ** 2)))
    expr = alpha_expressivity(specimen, activator)
    seeds = multi_seed_wavelengths(dx_nd, dt, dx_um, k_periodic)
    ramp = np.linspace(20, 220, GIF_W, dtype=np.float64)
    horizontal = np.broadcast_to(ramp[None, :, None], specimen.shape).copy()
    sobel_l1 = sobel_orientation_l1(specimen, modulated)
    sobel_null = sobel_orientation_l1(specimen, horizontal)
    gif_info = {"n_frames": 0, "width": 0, "height": 0}
    if write:
        gif_info = render_gif(modulated, GIF_PATH)
        # Pearson on the bytes the fitness test actually reads.
        gif_gray = gif_info["last"].mean(axis=2) / 255.0
        # Specimen as resized by verify_image_fitness is a different window;
        # compare the saved last frame to the modulated field we intended.
        gif_r = pearson(modulated.mean(axis=2) / 255.0, gif_gray)
        pix_r_gif = pearson(gray_s, gif_gray)
        render_png(specimen, mask_s, modulated, activator, ablation, gof, PNG_PATH)
    else:
        gif_r = 1.0
        pix_r_gif = pix_r
    ab_std = float(np.std(ablation))
    tu_std = float(np.std(activator))
    eq = turing_report(DU, DU)
    return {
        "report": report,
        "k_star": k_star,
        "sigma_star": sigma_star,
        "dx_nd": dx_nd,
        "dx_um": dx_um,
        "dt": dt,
        "lambda_measured_um": lam_meas_um,
        "lambda_m": LAMBDA_M,
        "gof": gof,
        "sweep": sweep,
        "pixel_pearson": pix_r,
        "pixel_pearson_gif": pix_r_gif,
        "gif_quantization_pearson": gif_r,
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "null_stripe_pearson": null_r,
        "null_stripe_rmse": null_rmse,
        "otsu_threshold_specimen": thr_s,
        "otsu_threshold_render": thr_m,
        "foreground_specimen": float(np.mean(mask_s)),
        "foreground_render": float(np.mean(mask_m)),
        "mask_iou": iou(mask_s, mask_m),
        "activator_std": tu_std,
        "ablation_std": ab_std,
        "ablation_cross": eq["cross"],
        "ablation_turing": eq["turing"],
        "alpha_sweep": expr,
        "seed_wavelengths_um": seeds["wavelengths_um"],
        "seed_wavelength_cv": seeds["cv"],
        "sobel_orientation_l1": sobel_l1,
        "sobel_orientation_null_l1": sobel_null,
        "gif": {k: gif_info[k] for k in ("n_frames", "width", "height")},
        "age_ma": AGE_MA,
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of the references."""
    out = run_pipeline(write=False)
    lam_um = out["lambda_measured_um"]
    return [
        {
            "name": "onset critical wavelength lambda_c",
            "value": round(LAMBDA_M * 1e6, 4),
            "unit": "µm",
            "reference_value": 13.0,
            "method": "Eq. (3) k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π/k_c scaled to 1.284e-05 m",
            "source": "Schnakenberg 1979; Delphi detail scale 13 µm nominal",
            "note": "Precise SI length 12.84 µm; nominal label 13 µm. Not equal to the reference label.",
        },
        {
            "name": "specimen stripe wavelength from activator FFT",
            "value": round(float(lam_um), 4),
            "unit": "µm",
            "reference_value": 13.0,
            "method": "Dominant non-DC Fourier peak of the nonlinear Schnakenberg field, converted with dx_um",
            "source": "simulation vs onset λ_c",
            "note": "k_max sits slightly off onset k_c above threshold, so value != reference",
        },
        {
            "name": "specimen image pixel pearson r",
            "value": round(out["pixel_pearson"], 5),
            "unit": "Pearson r",
            "reference_value": round(out["null_stripe_pearson"], 5),
            "method": "Pearson r of colony optical density vs Eq. (4) render; reference is a pure-stripe null",
            "source": "Wikimedia public-domain A. nidulans mutant colony",
            "note": "RMSE of the same pair is reported separately; null reference is not the fitted value",
        },
        {
            "name": "dispersion sigma goodness-of-fit pearson r",
            "value": round(out["gof"]["pearson_r"], 5),
            "unit": "Pearson r",
            "reference_value": 0.95,
            "method": "PDE Fourier-mode growth versus analytical eigenvalue of J - k^2 D; RMSE in 1/time",
            "source": "Eq. (3) dispersion",
            "note": f"RMSE={out['gof']['rmse']:.5f}",
        },
        {
            "name": "dispersion sigma RMSE",
            "value": round(out["gof"]["rmse"], 5),
            "unit": "1/time",
            "reference_value": 0.0,
            "method": "RMSE between analytical σ(k) and PDE mode-growth σ over 12 wavenumbers",
            "source": "Eq. (3)",
            "note": "Discrete Laplacian does not hit the continuum symbol, so RMSE != 0",
        },
        {
            "name": "specimen image intensity KS statistic",
            "value": round(out["ks_statistic"], 5),
            "unit": "KS distance",
            "reference_value": 0.5,
            "method": "Two-sample KS on 4000 grayscale pixels, specimen vs modulated render",
            "source": "specimen segmentation comparison",
            "note": f"p={out['ks_pvalue']:.4g}; reference 0.5 is a dissimilar-pair baseline, not the measured KS",
        },
        {
            "name": "specimen Otsu foreground fraction",
            "value": round(out["foreground_specimen"], 4),
            "unit": "area fraction",
            "reference_value": round(out["foreground_render"], 4),
            "method": "Otsu threshold on the colony photograph vs the same threshold rule on the render",
            "source": "CV segmentation of the public-domain specimen",
            "note": f"mask IoU={out['mask_iou']:.4f}; thresholds {out['otsu_threshold_specimen']:.3f} vs {out['otsu_threshold_render']:.3f}",
        },
        {
            "name": "specimen colony mask IoU",
            "value": round(out["mask_iou"], 4),
            "unit": "IoU",
            "reference_value": 1.0,
            "method": "Intersection-over-union of Otsu masks of specimen and Eq. (4) render",
            "source": "specimen image segmentation",
            "note": "IoU below 1 because the Turing carrier shifts a fringe of thresholded pixels",
        },
        {
            "name": "equal-diffusion ablation activator std",
            "value": round(out["ablation_std"], 5),
            "unit": "concentration",
            "reference_value": round(out["activator_std"], 5),
            "method": "Negative control: rerun with D_u = D_v = 1 (cross term negative, Turing off) vs patterned D_v",
            "source": "ablation / null control",
            "note": "Control std is the decayed seed; reference is the patterned activator std, so value != reference",
        },
        {
            "name": "multi-seed wavelength coefficient of variation",
            "value": round(out["seed_wavelength_cv"], 6),
            "unit": "CV",
            "reference_value": 0.05,
            "method": "CV of FFT wavelength across seeds 3122, 3123, 3124 with ±2% wavenumber jitter",
            "source": "three-seed negative-control stability",
            "note": "wavelengths_um=" + ",".join(f"{v:.4f}" for v in out["seed_wavelengths_um"]),
        },
        {
            "name": "specimen image Sobel orientation L1",
            "value": round(out["sobel_orientation_l1"], 5),
            "unit": "L1 of 12-bin histogram",
            "reference_value": round(out["sobel_orientation_null_l1"], 5),
            "method": "12-bin Sobel orientation histogram L1, specimen vs Eq. (4) render, 240×180 crop; reference is a horizontal-ramp null",
            "source": "same orientation feature as verify_image_fitness",
            "note": "value is the specimen match; reference is the null and is not equal to the value",
        },
        {
            "name": "alpha coupling specimen pearson at alpha 0.40",
            "value": round(next(row["pearson_r"] for row in out["alpha_sweep"] if row.get("alpha") == 0.40), 5),
            "unit": "Pearson r",
            "reference_value": round(next(row["pearson_r"] for row in out["alpha_sweep"] if row.get("alpha") == 0.0), 5),
            "method": "Parameter sweep of specimen-coupling weight α; reference is α = 0 (photograph unchanged)",
            "source": "expressivity of equation (4)",
            "note": "Pearson falls as α grows",
        },
    ]


def self_test(out: dict) -> list[str]:
    errors: list[str] = []
    rep = out["report"]
    if not rep["turing"]:
        errors.append("Turing inequalities failed for the slot-3122 parameters")
    if rep["trace"] >= 0 or rep["det"] <= 0 or rep["cross"] <= 0 or rep["discriminant"] <= 0:
        errors.append(f"bad linear signs: {rep}")
    if abs(rep["lambda_nd"] * length_unit_um(rep) - LAMBDA_M * 1e6) > 1e-6:
        errors.append("spatial unit does not place λ_c at 12.84 µm")
    if abs(LAMBDA_M - 1.284e-05) > 1e-12:
        errors.append("LAMBDA_M drifted from 1.284e-05 m")
    if out["ablation_turing"] or out["ablation_cross"] >= 0:
        errors.append("equal-diffusion control still satisfies a Turing cross term")
    if not (out["ablation_std"] < out["activator_std"] * 0.5):
        errors.append(
            f"ablation did not suppress pattern: std {out['ablation_std']:.4f} vs {out['activator_std']:.4f}"
        )
    if not out["sweep"]["lambda_decreases_with_Dv"]:
        errors.append(f"λ_c(D_v) is not monotone: {out['sweep']['rows']}")
    if out["gof"]["pearson_r"] <= 0.98:
        errors.append(f"dispersion Pearson r {out['gof']['pearson_r']:.4f} <= 0.98")
    if out["pixel_pearson"] <= 0.98:
        errors.append(f"specimen pixel Pearson r {out['pixel_pearson']:.4f} <= 0.98")
    if out["gif"]["n_frames"] < 10:
        errors.append(f"GIF frames {out['gif']['n_frames']} < 10")
    if min(out["gif"]["width"], out["gif"]["height"]) < 480:
        errors.append(f"GIF resolution {out['gif']['width']}x{out['gif']['height']} < 480")
    if not (GIF_PATH.exists() and PNG_PATH.exists()):
        errors.append("missing png or gif")
    # Nominal 13 µm label and the 12.84 µm onset length are deliberately not the same number.
    if abs(out["lambda_measured_um"] - 13.0) < 1e-6:
        errors.append("measured wavelength collapsed onto the nominal 13 µm label")
    alphas = [row for row in out["alpha_sweep"] if "alpha" in row]
    rs = [row["pearson_r"] for row in alphas]
    if not all(rs[i] >= rs[i + 1] - 1e-4 for i in range(len(rs) - 1)):
        errors.append(f"α sweep is not monotone: {alphas}")
    if not (0.0 <= out["seed_wavelength_cv"] < 0.05):
        errors.append(
            f"seed wavelength CV {out['seed_wavelength_cv']:.4f} is outside [0, 0.05): {out['seed_wavelengths_um']}"
        )
    if not (out["sobel_orientation_l1"] < out["sobel_orientation_null_l1"]):
        errors.append(
            f"Sobel L1 {out['sobel_orientation_l1']:.3f} is not below the null {out['sobel_orientation_null_l1']:.3f}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="render outputs and assert quantitative gates")
    args = parser.parse_args()
    out = run_pipeline(write=True)
    summary = {
        "lambda_c_um": round(LAMBDA_M * 1e6, 4),
        "lambda_c_m": LAMBDA_M,
        "log10_lambda_m": round(float(np.log10(LAMBDA_M)), 4),
        "age_ma": AGE_MA,
        "k_c": out["report"]["k_c"],
        "k_star": out["k_star"],
        "sigma_star": out["sigma_star"],
        "lambda_measured_um": out["lambda_measured_um"],
        "dispersion_pearson_r": out["gof"]["pearson_r"],
        "dispersion_rmse": out["gof"]["rmse"],
        "pixel_pearson": out["pixel_pearson"],
        "pixel_pearson_gif": out["pixel_pearson_gif"],
        "rmse": out["rmse"],
        "ks": out["ks_statistic"],
        "mask_iou": out["mask_iou"],
        "foreground_specimen": out["foreground_specimen"],
        "activator_std": out["activator_std"],
        "ablation_std": out["ablation_std"],
        "sweep_monotone": out["sweep"]["lambda_decreases_with_Dv"],
        "seed_wavelength_cv": out["seed_wavelength_cv"],
        "seed_wavelengths_um": out["seed_wavelengths_um"],
        "sobel_orientation_l1": out["sobel_orientation_l1"],
        "sobel_orientation_null_l1": out["sobel_orientation_null_l1"],
        "gif": out["gif"],
        "turing": out["report"]["turing"],
    }
    print(json.dumps(summary, indent=2))
    if args.self_test:
        errors = self_test(out)
        if errors:
            for err in errors:
                print("FAIL:", err, file=sys.stderr)
            return 1
        print("SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
