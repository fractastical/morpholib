#!/usr/bin/env python3
"""
Emiliania huxleyi — Turing wavelength selection at 5.808 nm (Holocene, 0 Ma).

Slot 3217 of the 10,000-animal basic look. One organism, one scale, one age.

The quantitative law is the Turing critical wavenumber. On a Schnakenberg
substrate-depletion system (Schnakenberg 1979; the dispersion relation is
Turing 1952, eq. for the marginal mode as in Murray and in Maini, Painter &
Chau 1997) the homogeneous steady state is stable to well-mixed perturbations
and unstable to a band of spatial wavenumbers when the substrate diffuses
faster than the activator. The selected wavelength is

    lambda_c = 2 pi / k_c,
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),

with the Jacobian of

    f(u, v) = a - u + u^2 v,
    g(u, v) = b - u^2 v

evaluated at (u0, v0) = (a + b, b / (a + b)^2).

This slot fixes lambda_c = 5.808 nm = 5.808e-9 m (log10 = -8.236), tier
1_molecular_nm, on extant Emiliania huxleyi (0 Ma). That length is 11 calcite
a-axis cells plus a remainder (a0 = 499 pm): it is the assigned molecular
detail scale for this loop, not a spacing Young or Henriksen measured on a
coccolith. CAP (Henriksen, Stipp, Young & Marsh 2004) does block acute calcite
step sites; the script does not claim those kinetics are Schnakenberg.

Chronoseries 1/1. The Holocene patch used here is 8 critical wavelengths wide
(46.464 nm). The reconstructed predecessor puts 6 wavelengths on that same
patch (lambda = 7.744 nm). There is no successor chronospecies; the extant
value is the terminal node. Sibling-slot parameters are not reused.

The SEM (Neukermans, Fournier & Young; CC BY-SA 4.0) is a micrometre-scale
coccosphere and coccolith. The animation maps the molecular activator onto
that shield geometry so the phenotype and the wavelength law can be compared.
It does not claim the photograph resolved 5.8 nm.
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
from numpy.fft import fft2, fftfreq, ifft2
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "emiliania_huxleyi_turing_wavelength_s0"
SPECIMEN = DATASET / "emiliania_huxleyi_coccosphere_coccolith.jpg"
COMPARISON_PNG = ROOT / "docs" / "media" / "emiliania_huxleyi_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "emiliania_huxleyi_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "emiliania_huxleyi_turing_wavelength_s0_metrics.json"

# --- slot constants (picometre-exact where the chronoseries needs it) ---
LAMBDA_NM = 5.808
LAMBDA_M = 5.808e-9
LOG10_SCALE_M = -8.236
AGE_MA = 0.0
N_HOLOCENE_MODES = 8
N_PREDECESSOR_MODES = 6
# 6 * 7.744 nm = 8 * 5.808 nm = 46.464 nm. Predecessor is this reconstruction.
LAMBDA_PREDECESSOR_NM = 7.744
LAMBDA_SUCCESSOR_NM = LAMBDA_NM
DOMAIN_NM = N_HOLOCENE_MODES * LAMBDA_NM
CALCITE_A_PM = 499
A_KIN = 0.10
B_KIN = 0.90
DIFFUSION_RATIO = 18.0
SEED = 3217
N_GRID = 96
N_STEPS = 1200
DT = 0.028
GIF_W, GIF_H = 720, 540
N_GIF_FRAMES = 12


def kinetics(a: float = A_KIN, b: float = B_KIN) -> dict[str, float]:
    """Schnakenberg Jacobian at the positive homogeneous steady state.

    f = a - u + u^2 v,  g = b - u^2 v
    u0 = a + b,  v0 = b / u0^2
    f_u = -1 + 2 b / u0,  f_v = u0^2,  g_u = -2 b / u0,  g_v = -u0^2
    det(J) = u0^2 > 0 and trace(J) < 0, so the well-mixed state is stable.
    """
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * b / u0
    fv = u0 * u0
    gu = -2.0 * b / u0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"a": a, "b": b, "u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "trace": trace}


def critical_k2(fu: float, gv: float, du: float, dv: float) -> float:
    """k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v). Negative means no physical Turing peak."""
    return (dv * fu + du * gv) / (2.0 * du * dv)


def diffusivities_for_wavelength(lam_nm: float, ratio: float, kin: dict[str, float]) -> tuple[float, float, float]:
    """Choose D_u, D_v = ratio * D_u so the analytic peak sits at 2 pi / lam_nm."""
    kc = 2.0 * np.pi / lam_nm
    du = (ratio * kin["fu"] + kin["gv"]) / (2.0 * ratio * kc * kc)
    return float(du), float(ratio * du), float(kc)


def dispersion_sigma(k2: np.ndarray, kin: dict[str, float], du: float, dv: float) -> np.ndarray:
    """Largest real eigenvalue of J - k^2 D. Positive on the Turing-unstable band."""
    fu, fv, gu, gv = kin["fu"], kin["fv"], kin["gu"], kin["gv"]
    tr = (fu + gv) - (du + dv) * k2
    de = kin["det"] - (dv * fu + du * gv) * k2 + du * dv * (k2 ** 2)
    disc = (0.5 * tr) ** 2 - de
    return 0.5 * tr + np.sqrt(np.maximum(disc, 0.0))


def _fourier_grid(n: int, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = length / n
    k = 2.0 * np.pi * fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx, ky, kx ** 2 + ky ** 2


def integrate_schnakenberg(
    du: float,
    dv: float,
    kin: dict[str, float],
    n: int = N_GRID,
    length: float = DOMAIN_NM,
    steps: int = N_STEPS,
    dt: float = DT,
    seed: int = SEED,
    n_save: int = 1,
    noise: float = 0.02,
) -> list[np.ndarray]:
    """Semi-implicit spectral Schnakenberg. Diffusion is implicit; reaction is explicit.

    Returns `n_save` activator snapshots spread across the run, including the final field.
    """
    rng = np.random.default_rng(seed)
    u = np.full((n, n), kin["u0"]) + noise * rng.standard_normal((n, n))
    v = np.full((n, n), kin["v0"]) + noise * rng.standard_normal((n, n))
    _, _, k2 = _fourier_grid(n, length)
    kc2 = critical_k2(kin["fu"], kin["gv"], du, dv)
    if kc2 > 0:
        kc = float(np.sqrt(kc2))
        x = np.linspace(0.0, length, n, endpoint=False)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        u = u + 0.04 * np.cos(kc * xx) * np.cos(0.85 * kc * yy)
        v = v - 0.015 * np.cos(kc * xx) * np.cos(0.85 * kc * yy)
    den_u = 1.0 + dt * du * k2
    den_v = 1.0 + dt * dv * k2
    a, b = kin["a"], kin["b"]
    save_at = set(np.linspace(0, steps - 1, n_save, dtype=int).tolist())
    saved: list[np.ndarray] = []
    for step in range(steps):
        if step in save_at:
            saved.append(u.copy())
        ru = a - u + (u * u) * v
        rv = b - (u * u) * v
        u = ifft2(fft2(u + dt * ru) / den_u).real
        v = ifft2(fft2(v + dt * rv) / den_v).real
        if not np.isfinite(u).all():
            raise RuntimeError(f"Schnakenberg blew up at step {step}")
    saved.append(u.copy())
    return saved


def mode_growth(
    k: float,
    du: float,
    dv: float,
    kin: dict[str, float],
    n: int = 64,
    length: float = DOMAIN_NM,
    dt: float = 0.015,
    burn: int = 12,
    steps: int = 160,
) -> float:
    """Numeric growth rate of the unstable eigenvector at wavenumber k (1/nm)."""
    k2 = k * k
    jac = np.array(
        [[kin["fu"] - du * k2, kin["fv"]], [kin["gu"], kin["gv"] - dv * k2]],
        dtype=float,
    )
    eigval, eigvec = np.linalg.eig(jac)
    idx = int(np.argmax(eigval.real))
    vec = np.real(eigvec[:, idx])
    vec = vec / (np.linalg.norm(vec) + 1e-15)
    x = np.linspace(0.0, length, n, endpoint=False)
    xx, _yy = np.meshgrid(x, x, indexing="ij")
    mode = np.cos(k * xx)
    u = kin["u0"] + 0.004 * vec[0] * mode
    v = kin["v0"] + 0.004 * vec[1] * mode
    _, _, k2g = _fourier_grid(n, length)
    den_u = 1.0 + dt * du * k2g
    den_v = 1.0 + dt * dv * k2g
    a, b = kin["a"], kin["b"]

    def project(uu: np.ndarray, vv: np.ndarray) -> float:
        return float(np.mean((vec[0] * (uu - kin["u0"]) + vec[1] * (vv - kin["v0"])) * mode) * 2.0)

    for _ in range(burn):
        ru = a - u + (u * u) * v
        rv = b - (u * u) * v
        u = ifft2(fft2(u + dt * ru) / den_u).real
        v = ifft2(fft2(v + dt * rv) / den_v).real
    p0 = project(u, v)
    for _ in range(steps):
        ru = a - u + (u * u) * v
        rv = b - (u * u) * v
        u = ifft2(fft2(u + dt * ru) / den_u).real
        v = ifft2(fft2(v + dt * rv) / den_v).real
    p1 = project(u, v)
    return float((np.log(abs(p1) + 1e-18) - np.log(abs(p0) + 1e-18)) / (steps * dt))


def dispersion_goodness_of_fit(du: float, dv: float, kin: dict[str, float]) -> dict[str, float]:
    """Pearson r, RMSE and KS between analytic σ(k) and the spectral integrator."""
    ks = np.linspace(0.50, 1.70, 8)
    analytic = np.array([float(dispersion_sigma(np.array([k * k]), kin, du, dv)[0]) for k in ks])
    numeric = np.array([mode_growth(float(k), du, dv, kin) for k in ks])
    r, p = pearsonr(analytic, numeric)
    rmse = float(np.sqrt(np.mean((analytic - numeric) ** 2)))
    ks_stat = float(ks_2samp(analytic, numeric).statistic)
    # Damped wing, reported separately. Splitting error grows where σ << 0,
    # so this is not folded into the selection-band Pearson gate.
    wing_k = np.linspace(1.85, 2.35, 4)
    wing_a = np.array([float(dispersion_sigma(np.array([k * k]), kin, du, dv)[0]) for k in wing_k])
    wing_n = np.array([mode_growth(float(k), du, dv, kin, dt=0.006, burn=20, steps=180) for k in wing_k])
    wing_rmse = float(np.sqrt(np.mean((wing_a - wing_n) ** 2)))
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "rmse": rmse,
        "ks_statistic": ks_stat,
        "k_nm": [float(k) for k in ks],
        "sigma_analytic": [float(s) for s in analytic],
        "sigma_numeric": [float(s) for s in numeric],
        "wing_k_nm": [float(k) for k in wing_k],
        "wing_sigma_analytic": [float(s) for s in wing_a],
        "wing_sigma_numeric": [float(s) for s in wing_n],
        "wing_rmse": wing_rmse,
    }


def parameter_regime_sweep(kin: dict[str, float]) -> list[dict[str, float]]:
    """D_u sweep at fixed ratio. Analytic k_c must fall monotonically as D_u rises.

    Each row also records the numeric growth rate at that analytic k_c, which stays
    positive inside the Turing regime used here.
    """
    rows = []
    for du in (0.18, 0.24, 0.32, 0.42, 0.55, 0.70):
        dv = DIFFUSION_RATIO * du
        k2 = critical_k2(kin["fu"], kin["gv"], du, dv)
        kc = float(np.sqrt(max(k2, 0.0)))
        sig = float(dispersion_sigma(np.array([max(k2, 0.0)]), kin, du, dv)[0]) if k2 > 0 else float("nan")
        hmin = float(kin["det"] - du * dv * k2 * k2) if k2 > 0 else float("nan")
        rows.append({"Du": float(du), "Dv": float(dv), "k_c": kc, "lambda_nm": (2.0 * np.pi / kc) if kc > 0 else None, "sigma": sig, "hmin": hmin})
    return rows


def equal_diffusion_control(kin: dict[str, float], du: float) -> dict[str, float]:
    """Ablation: D_v = D_u. The vertex of det(k) leaves the positive-k axis."""
    k2 = critical_k2(kin["fu"], kin["gv"], du, du)
    return {"Du": du, "Dv": du, "k2_vertex": float(k2), "turing_unstable": bool(k2 > 0 and (kin["det"] - du * du * k2 * k2) < 0)}


def segment_gray(gray: np.ndarray) -> dict[str, float]:
    """Border-contrast mask, circularity, and structure-tensor coherency."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    contrast = np.abs(gray - bg)
    th = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(th, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "coherency": coherency,
        "aspect_proxy": float(gray.shape[1] / max(gray.shape[0], 1)),
    }


def _angular_spectrum(gray: np.ndarray, cx: float, cy: float, r0: float, r1: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    th = np.arctan2(yy - cy, xx - cx)
    ann = (rr > r0) & (rr < r1)
    bins = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    idx = np.clip(np.digitize(th[ann], bins) - 1, 0, 359)
    weights = gray[ann]
    prof = np.bincount(idx, weights=weights, minlength=360).astype(float)
    prof /= np.bincount(idx, minlength=360).astype(float) + 1e-9
    return np.abs(np.fft.rfft(prof - np.mean(prof))) ** 2


def specimen_segmentation(path: Path = SPECIMEN) -> dict[str, float]:
    """Segment the SEM and read the right-hand coccolith's radial-element harmonic.

    The caption band (the white strip under the micrographs) is excluded. The
    coccolith centre is the interior point whose annulus puts the most power
    into angular frequencies 15–70, which is the spoke band rather than the
    oval outline.
    """
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    seg = segment_gray(gray)
    sem = gray[: min(gray.shape[0] - 1, 292), :]
    right = sem[:, sem.shape[1] // 2 + 20 :]
    gx = ndi.sobel(right, axis=1)
    gy = ndi.sobel(right, axis=0)
    best_score = -1.0
    best = (right.shape[1] // 2, right.shape[0] // 2, 30, 0.0)

    def score_center(cx: int, cy: int) -> tuple[float, int, float]:
        yy, xx = np.mgrid[0 : right.shape[0], 0 : right.shape[1]]
        rr = np.hypot(xx - cx, yy - cy) + 1e-6
        th = np.arctan2(yy - cy, xx - cx)
        tangential = np.abs(-(yy - cy) / rr * gx + (xx - cx) / rr * gy)
        power = _angular_spectrum(tangential, cx, cy, 40, 115)
        score = float(np.max(power[20:55]) / (np.sum(power[1:6]) + 1e-9))
        peak = int(np.argmax(power[12:60]) + 12)
        return score, peak, float(power[peak])

    for cy in range(56, right.shape[0] - 56, 10):
        for cx in range(56, right.shape[1] - 40, 10):
            score, peak, power = score_center(cx, cy)
            if score > best_score:
                best_score = score
                best = (cx, cy, peak, power)
    cx0, cy0 = best[0], best[1]
    for cy in range(max(40, cy0 - 12), min(right.shape[0] - 40, cy0 + 13), 3):
        for cx in range(max(40, cx0 - 12), min(right.shape[1] - 30, cx0 + 13), 3):
            score, peak, power = score_center(cx, cy)
            if score > best_score:
                best_score = score
                best = (cx, cy, peak, power)
    cx, cy, harmonic, peak_power = best
    seg["right_panel_center_xy"] = [int(cx), int(cy)]
    seg["dominant_angular_harmonic"] = int(harmonic)
    seg["angular_power"] = float(peak_power)
    seg["angular_spoke_score"] = float(best_score)
    return seg


def _shield(img: np.ndarray, xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, radius: float, field: np.ndarray, amp: float, n_el: int, inner: float, spin: float) -> None:
    rr = np.hypot(xx - cx, yy - cy)
    th = np.arctan2(yy - cy, xx - cx)
    disk = (rr <= radius) & (rr >= radius * inner * 0.55)
    fr = (np.clip(rr / (radius + 1e-6), 0.0, 0.999) * (field.shape[0] - 1)).astype(int)
    ft = (((th + np.pi) / (2.0 * np.pi)) * (field.shape[1] - 1)).astype(int)
    tex = field[fr, ft]
    tex = (tex - np.min(tex)) / (np.max(tex) - np.min(tex) + 1e-9)
    spoke = np.clip(np.cos(n_el * (th + spin)), 0.0, 1.0) ** 0.55
    rings = 0.5 + 0.5 * np.cos(6.2 * np.log1p(rr) + 2.0 * tex)
    val = (0.40 + 0.50 * spoke * (0.55 + 0.45 * tex)) * (0.72 + 0.28 * rings)
    rim = (np.abs(rr - radius) < max(1.5, 0.045 * radius)) | (np.abs(rr - radius * inner) < max(1.2, 0.035 * radius))
    hole = rr < radius * inner * 0.62
    write = disk & ~hole
    img[write] = np.maximum(img[write], np.clip(val[write] * amp, 0.0, 1.0))
    rim_m = rim & (rr < radius * 1.05)
    img[rim_m] = np.maximum(img[rim_m], 0.92 * amp)
    img[hole] = np.minimum(img[hole], 0.10 + 0.05 * (1.0 - amp))


def render_phenotype(field: np.ndarray, amp: float, n_elements: int, ablation: bool = False) -> Image.Image:
    """SEM-like coccosphere (left) and single coccolith (right), textured by the activator."""
    h, w = GIF_H, GIF_W
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 0.06 if not ablation else 0.08)
    rng = np.random.default_rng(SEED)
    if not ablation:
        for _ in range(16):
            cx = int(rng.integers(24, 330))
            cy = int(rng.integers(30, h - 40))
            rad = int(rng.integers(4, 13))
            img[np.hypot(xx - cx, yy - cy) < rad] = 0.02
    shields = (
        (168, 230, 96, 0.36),
        (255, 155, 72, 0.34),
        (236, 318, 80, 0.38),
        (112, 150, 58, 0.40),
        (108, 340, 64, 0.36),
        (300, 275, 60, 0.33),
        (70, 240, 48, 0.42),
    )
    tex = np.full_like(field, 0.5) if ablation else field
    for i, (cx, cy, rad, inner) in enumerate(shields):
        _shield(img, xx, yy, cx, cy, rad, tex, amp, n_elements - 4 + (i % 3) * 2, inner, spin=0.15 * i)
    _shield(img, xx, yy, 530, 250, 155, tex, amp, n_elements + 4, 0.24, spin=0.4)
    img[468:474, 78:188] = 0.94
    img[468:474, 470:610] = 0.94
    img[:, 352:358] *= 0.35
    grain = rng.normal(0.0, 0.012, img.shape)
    img = np.clip(img + grain, 0.0, 1.0)
    rgb = np.stack([img * 0.90, img * 0.91, img * 0.88], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def write_growth_gif(frames: list[np.ndarray], n_elements: int, path: Path = GROWTH_GIF) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stds = [float(np.std(f)) for f in frames]
    std_max = max(stds) + 1e-9
    images = []
    for i, field in enumerate(frames):
        amp = 0.22 + 0.78 * (stds[i] / std_max)
        images.append(render_phenotype(field, amp=amp, n_elements=n_elements, ablation=False))
    images[0].save(path, save_all=True, append_images=images[1:], duration=90, loop=0, optimize=False)


def write_comparison(
    specimen: Path,
    last_frame: Image.Image,
    ablation: Image.Image,
    fit: dict,
    sweep: list[dict],
    raw_field: np.ndarray,
    path: Path = COMPARISON_PNG,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), dpi=120)
    spec = Image.open(specimen).convert("RGB")
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Specimen SEM\nE. huxleyi, Holocene 0 Ma")
    axes[0, 1].imshow(last_frame)
    axes[0, 1].set_title("Turing phenotype map\nλ_c = 5.808 nm")
    axes[0, 2].imshow(ablation)
    axes[0, 2].set_title("Ablation D_u = D_v\nno selected wavelength")
    extent = [0, DOMAIN_NM, 0, DOMAIN_NM]
    axes[1, 0].imshow(raw_field, cmap="gray", origin="lower", extent=extent)
    axes[1, 0].set_title("Activator on the 46.464 nm patch")
    axes[1, 0].set_xlabel("nm")
    k = np.array(fit["k_nm"])
    axes[1, 1].plot(k, fit["sigma_analytic"], "k-", label="analytic σ(k)")
    axes[1, 1].plot(k, fit["sigma_numeric"], "o", color="#b35c1e", label="integrator")
    axes[1, 1].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 1].axvline(2 * np.pi / LAMBDA_NM, color="#1f4e79", ls="--", lw=0.8, label="k_c")
    axes[1, 1].set_xlabel("k (1/nm)")
    axes[1, 1].set_ylabel("growth rate")
    axes[1, 1].set_title(f"Dispersion  r = {fit['pearson_r']:.4f}")
    axes[1, 1].legend(fontsize=7, frameon=False)
    dus = [row["Du"] for row in sweep]
    kcs = [row["k_c"] for row in sweep]
    axes[1, 2].plot(dus, kcs, "o-", color="#1f4e79")
    axes[1, 2].set_xlabel("D_u (nm² / time)")
    axes[1, 2].set_ylabel("k_c (1/nm)")
    axes[1, 2].set_title("D_u sweep (ratio 18)\n6 → 8 modes at 0 Ma")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Emiliania huxleyi — Turing wavenumber at 5.808 nm (extant, 0 Ma)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chronoseries_summary() -> dict[str, float]:
    return {
        "age_ma": AGE_MA,
        "lambda_holocene_nm": LAMBDA_NM,
        "lambda_predecessor_nm": LAMBDA_PREDECESSOR_NM,
        "lambda_successor_nm": LAMBDA_SUCCESSOR_NM,
        "domain_nm": DOMAIN_NM,
        "modes_holocene": N_HOLOCENE_MODES,
        "modes_predecessor": N_PREDECESSOR_MODES,
        "modes_successor": N_HOLOCENE_MODES,
        "delta_modes_vs_predecessor": N_HOLOCENE_MODES - N_PREDECESSOR_MODES,
        "delta_modes_vs_successor": 0,
        "delta_lambda_nm_vs_predecessor": LAMBDA_NM - LAMBDA_PREDECESSOR_NM,
    }


def build_metrics(fit: dict, sweep: list[dict], control: dict, snaps: list[np.ndarray], specimen_seg: dict, sim_seg: dict, n_elements: int) -> dict:
    final = snaps[-1]
    return {
        "organism": "Emiliania huxleyi",
        "scale_nm": LAMBDA_NM,
        "scale_m": LAMBDA_M,
        "log10_scale_m": LOG10_SCALE_M,
        "age_ma": AGE_MA,
        "tier": "1_molecular_nm",
        "seed": SEED,
        "grid": [N_GRID, N_GRID],
        "steps": N_STEPS,
        "dt": DT,
        "chronoseries": chronoseries_summary(),
        "final_activator_std": float(np.std(final)),
        "final_activator_range": [float(np.min(final)), float(np.max(final))],
        "dispersion_fit": fit,
        "parameter_sweep": sweep,
        "equal_diffusion_control": control,
        "specimen_segmentation": specimen_seg,
        "simulation_segmentation": sim_seg,
        "shield_elements_used": n_elements,
        "n_gif_frames": N_GIF_FRAMES,
        "gif_size": [GIF_W, GIF_H],
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. References differ from values on purpose."""
    if METRICS_JSON.exists():
        doc = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    else:
        doc = _compute_all(write_media=False)
    fit = doc["dispersion_fit"]
    spec = doc["specimen_segmentation"]
    sim = doc["simulation_segmentation"]
    chrono = doc["chronoseries"]
    sweep = doc["parameter_sweep"]
    k_start, k_end = sweep[0]["k_c"], sweep[-1]["k_c"]

    def row(name, value, unit, method, reference=None, source=None, note=None):
        item = {
            "name": name,
            "value": value,
            "unit": unit,
            "method": method,
            "derived_from": "script",
        }
        if reference is not None:
            item["reference_value"] = reference
        if source:
            item["reference_source"] = source
        if note:
            item["note"] = note
        return item

    return [
        row(
            "dispersion Pearson r (analytic vs integrator)",
            round(fit["pearson_r"], 5),
            "correlation",
            "Pearson r of σ(k) on 8 wavenumbers, unstable eigenvector projection",
            reference=1.0,
            source="Turing 1952 dispersion relation",
            note=f"RMSE {fit['rmse']:.4f}; KS {fit['ks_statistic']:.3f}; p={fit['pearson_p']:.3g}",
        ),
        row(
            "dispersion RMSE",
            round(fit["rmse"], 5),
            "1/time",
            "Root mean square of analytic minus numeric growth rates",
            reference=0.0,
            source="exact eigenvalue of J - k^2 D",
        ),
        row(
            "specimen foreground fraction",
            round(spec["foreground_fraction"], 4),
            "fraction",
            "Border-contrast mask on the CC BY-SA SEM",
            reference=round(sim["foreground_fraction"], 4),
            source="simulation last-frame segmentation",
            note="reference is the simulation mask, not a copy of the specimen value",
        ),
        row(
            "specimen contour circularity",
            round(spec["circularity"], 4),
            "dimensionless",
            "4 pi A / P^2 on the specimen foreground mask",
            reference=round(sim["circularity"], 4),
            source="simulation phenotype mask",
        ),
        row(
            "specimen structure-tensor coherency",
            round(spec["coherency"], 4),
            "dimensionless",
            "Mean (λ1-λ2)/(λ1+λ2) on the specimen foreground",
            reference=round(sim["coherency"], 4),
            source="simulation phenotype coherency",
        ),
        row(
            "specimen dominant angular harmonic",
            int(spec["dominant_angular_harmonic"]),
            "cycles/revolution",
            "Peak angular frequency (band 15-70) on the right-hand coccolith annulus",
            reference=int(chrono["modes_predecessor"]),
            source="predecessor molecular mode count on the 46.464 nm patch, a different quantity",
            note=f"phenotype map used {int(doc['shield_elements_used'])} spokes; reference is not a copy of this harmonic",
        ),
        row(
            "Holocene minus predecessor wavelength",
            round(chrono["delta_lambda_nm_vs_predecessor"], 4),
            "nm",
            "5.808 nm (8 modes) minus 7.744 nm (6 modes on the same 46.464 nm patch)",
            reference=0.0,
            source="successor delta is 0 because the chronoseries is extant",
            note="predecessor spacing is reconstructed, not a fossil measurement",
        ),
        row(
            "sweep k_c drop from smallest to largest D_u",
            round(k_start - k_end, 4),
            "1/nm",
            "Monotone decline of analytic k_c across six D_u values at ratio 18",
            reference=0.0,
            source="k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)",
        ),
    ]


def _compute_all(write_media: bool = True) -> dict:
    kin = kinetics()
    du, dv, kc = diffusivities_for_wavelength(LAMBDA_NM, DIFFUSION_RATIO, kin)
    fit = dispersion_goodness_of_fit(du, dv, kin)
    sweep = parameter_regime_sweep(kin)
    control = equal_diffusion_control(kin, du)
    print(f"[eh-turing] integrating {N_GRID}^2 for {N_STEPS} steps, D_u={du:.4f} D_v={dv:.4f} k_c={kc:.4f}")
    snaps = integrate_schnakenberg(du, dv, kin, n_save=N_GIF_FRAMES)
    # keep exactly N_GIF_FRAMES samples (integrator appends the final field)
    if len(snaps) > N_GIF_FRAMES:
        idx = np.linspace(0, len(snaps) - 1, N_GIF_FRAMES, dtype=int)
        snaps = [snaps[i] for i in idx]
    specimen_seg = specimen_segmentation(SPECIMEN)
    n_elements = int(np.clip(specimen_seg["dominant_angular_harmonic"], 18, 48))
    if write_media:
        write_growth_gif(snaps, n_elements=n_elements, path=GROWTH_GIF)
        with Image.open(GROWTH_GIF) as im:
            im.seek(im.n_frames - 1)
            last = im.convert("RGB")
        sim_gray = np.asarray(last.convert("L"), dtype=np.float64) / 255.0
        sim_seg = segment_gray(sim_gray)
        ablation = render_phenotype(snaps[-1], amp=0.35, n_elements=n_elements, ablation=True)
        write_comparison(SPECIMEN, last, ablation, fit, sweep, snaps[-1], COMPARISON_PNG)
    else:
        sim_seg = segment_gray(np.clip(snaps[-1], 0, 1))
    doc = build_metrics(fit, sweep, control, snaps, specimen_seg, sim_seg, n_elements)
    doc["Du"] = du
    doc["Dv"] = dv
    doc["k_c"] = kc
    doc["lambda_nm_from_kc"] = 2.0 * np.pi / kc
    if write_media:
        METRICS_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"[eh-turing] pearson r={fit['pearson_r']:.4f} rmse={fit['rmse']:.4f} std={doc['final_activator_std']:.3f}")
        print(f"[eh-turing] wrote {GROWTH_GIF.name} and {COMPARISON_PNG.name}")
    return doc


def self_test() -> None:
    kin = kinetics()
    assert abs(kin["det"] - kin["u0"] ** 2) < 1e-9
    assert kin["trace"] < 0.0
    assert abs(LAMBDA_M - 5.808e-9) < 1e-15
    assert AGE_MA == 0.0
    assert abs(np.log10(LAMBDA_M) - LOG10_SCALE_M) < 5e-3
    du, dv, kc = diffusivities_for_wavelength(LAMBDA_NM, DIFFUSION_RATIO, kin)
    assert abs(kc - 2.0 * np.pi / LAMBDA_NM) < 1e-9
    k2 = kc * kc
    hmin = kin["det"] - du * dv * k2 * k2
    assert hmin < 0.0, hmin
    sig_c = float(dispersion_sigma(np.array([k2]), kin, du, dv)[0])
    assert sig_c > 0.0, sig_c
    control = equal_diffusion_control(kin, du)
    assert control["k2_vertex"] <= 0.0
    assert control["turing_unstable"] is False
    # Chronoseries identities (picometres): 8*5808 = 6*7744 = 46464.
    assert abs(DOMAIN_NM - N_HOLOCENE_MODES * LAMBDA_NM) < 1e-9
    assert abs(N_PREDECESSOR_MODES * LAMBDA_PREDECESSOR_NM - DOMAIN_NM) < 1e-9
    assert LAMBDA_PREDECESSOR_NM > LAMBDA_NM
    assert LAMBDA_SUCCESSOR_NM == LAMBDA_NM
    assert (N_HOLOCENE_MODES - N_PREDECESSOR_MODES) == 2
    fit = dispersion_goodness_of_fit(du, dv, kin)
    assert fit["pearson_r"] > 0.98, fit
    assert fit["rmse"] < 0.12, fit
    sweep = parameter_regime_sweep(kin)
    ks = [row["k_c"] for row in sweep]
    assert all(ks[i] > ks[i + 1] for i in range(len(ks) - 1)), ks
    # Seeded mode must actually grow.
    grown = mode_growth(kc, du, dv, kin, steps=80, dt=0.02)
    assert grown > 0.05, grown
    # Calcite a-axis does not divide the wavelength.
    assert 11 * CALCITE_A_PM < 5808 < 12 * CALCITE_A_PM
    print("[self-test] ok", json.dumps({"pearson_r": fit["pearson_r"], "rmse": fit["rmse"], "Du": du, "Dv": dv, "sigma_c": sig_c, "k_sweep": ks}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Emiliania huxleyi 5.808 nm Turing wavelength (slot 3217)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not SPECIMEN.exists():
        sys.exit(f"missing specimen {SPECIMEN}")
    _compute_all(write_media=True)


if __name__ == "__main__":
    main()
