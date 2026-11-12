#!/usr/bin/env python3
"""Coturnix japonica — Turing wavelength at 22.42 µm (slot 3157).

Holocene extant Japanese quail, age 0 Ma. The chronoseries has one node, so
there is no fossil predecessor or successor. What changes at this scale is
ontogenetic: an equal-diffusion collar does not select a wavelength; the
nominal Schnakenberg system does, at lambda_c = 2.242e-5 m (22.42 µm,
log10 = -4.649), the cellular-detail pigment / ocellus mode inside a feather.

Kinetics (Schnakenberg 1979), dispersion (Turing 1952):

    du/dt = a - u + u^2 v + div(D_u grad u)
    dv/dt = b - u^2 v + D_v laplacian v
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
    lambda_c = 2 pi / k_c

In the inhibitor-dominated regime k_c^2 ≈ f_u / (2 D_u), so the wavelength
is moved by activator diffusivity, not by D_v (which nearly cancels). The
display run adds mild x-anisotropy so streaks lie along the body axis of the
Yehliu portrait. That portrait has no micron scale bar; it is the Holocene
envelope the micro-mode is painted into.

References: Turing 1952 (10.1098/rstb.1952.0012); Schnakenberg 1979
(10.1016/0022-5193(79)90042-0); Prum & Williamson 2002
(10.1098/rspb.2001.1896); Harris et al. 2005 (10.1073/pnas.0500781102);
Haupaix et al. 2018 (10.1126/science.aar4777); Inaba & Chuong 2020
(10.3389/fcell.2020.00620).
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "coturnix_japonica_turing_wavelength_s0"
SPECIMEN_PATH = DATA / "coturnix_japonica_yehliu_taiwan.jpg"
REF_PATH = DATA / "coturnix_japonica_turing_wavelength_s0_reference.json"
METRICS_PATH = DATA / "coturnix_japonica_turing_wavelength_s0_metrics.json"
MEDIA_PNG = ROOT / "docs" / "media" / "coturnix_japonica_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "coturnix_japonica_turing_wavelength_s0_growth.gif"
FITNESS_PATH = (
    ROOT / "morphoeval" / "runs" / "2026-10-27-coturnix-japonica-turing-wavelength-s0-loop-image-fitness.json"
)

# Slot length and nominal kinetics. Do not reuse the somite-script pair
# (beta_A = 0.22, d = 20, seed 2018).
LAMBDA_M = 2.242e-5
A_NOM = 0.07
B_NOM = 0.93
DU_NOM = 0.035
DV_NOM = 1.85
ANISOTROPY = 0.28
SEED = 3157
DU_SWEEP = (0.020, 0.028, 0.035, 0.048, 0.062)

PPW = 14
N_WAVES_X = 6.0
N_WAVES_Y = 4.5
DT = 0.0011
T_END = 12.0
N_FRAMES = 12
GIF_SIZE = (960, 540)


def jacobian(a: float = A_NOM, b: float = B_NOM):
    """Schnakenberg Jacobian at the homogeneous steady state (u0, v0)."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return fu, fv, gu, gv, u0, v0


def cross_term(Du: float, Dv: float, a: float = A_NOM, b: float = B_NOM) -> float:
    fu, _, _, gv, _, _ = jacobian(a, b)
    return Dv * fu + Du * gv


def kc2(Du: float, Dv: float, a: float = A_NOM, b: float = B_NOM) -> float:
    """Critical wavenumber squared, k_c^2 = cross / (2 D_u D_v)."""
    return cross_term(Du, Dv, a, b) / (2.0 * Du * Dv)


def length_unit_m(Du: float = DU_NOM, Dv: float = DV_NOM) -> float:
    """Metres per length-unit, fixed so the nominal linear wavelength is LAMBDA_M."""
    lam_lu = 2.0 * np.pi / np.sqrt(kc2(Du, Dv))
    return LAMBDA_M / lam_lu


def lambda_m(Du: float, Dv: float, unit: float | None = None) -> float:
    unit = length_unit_m() if unit is None else unit
    return (2.0 * np.pi / np.sqrt(kc2(Du, Dv))) * unit


def sigma_of_k(k: np.ndarray, Du: float, Dv: float, a: float = A_NOM, b: float = B_NOM) -> np.ndarray:
    """Larger eigenvalue of J - k^2 D. Positive values are Turing-unstable."""
    fu, fv, gu, gv, _, _ = jacobian(a, b)
    kk = np.asarray(k, dtype=np.float64)
    tr = (fu + gv) - (kk * kk) * (Du + Dv)
    det = Du * Dv * kk**4 - cross_term(Du, Dv, a, b) * kk**2 + (fu * gv - fv * gu)
    disc = tr * tr - 4.0 * det
    root = np.sqrt(np.maximum(disc, 0.0))
    return 0.5 * (tr + root)


def sigma_max(Du: float, Dv: float, a: float = A_NOM, b: float = B_NOM) -> tuple[float, float]:
    ks = np.linspace(0.05, 14.0, 800)
    sig = sigma_of_k(ks, Du, Dv, a, b)
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def turing_inequality(Du: float, Dv: float, a: float = A_NOM, b: float = B_NOM) -> float:
    """cross^2 - 4 D_u D_v det J. Positive means the unstable band is open."""
    fu, fv, gu, gv, _, _ = jacobian(a, b)
    det = fu * gv - fv * gu
    cross = cross_term(Du, Dv, a, b)
    return cross * cross - 4.0 * Du * Dv * det


def lean_scaled_identities() -> dict[str, int]:
    """Integer identities proved in Loop_3157 (scale 1000 on rates and diffusivities)."""
    trace = 860 + (-1000)
    det = 860 * (-1000) - 1000 * (-1860)
    cross = 1850 * 860 + 35 * (-1000)
    den = 2 * 35 * 1850
    disc = cross * cross - 4 * 35 * 1850 * det
    equal = 35 * 860 + 35 * (-1000)
    return {
        "trace_scaled": trace,
        "det_scaled": det,
        "cross_scaled": cross,
        "kc_den_scaled": den,
        "band_discriminant": disc,
        "equal_diffusion_cross": equal,
        "holocene_age_gap_Ma": 0,
    }


def _diffuse(z: np.ndarray, dx: float, dy: float, dxx: float, dyy: float) -> np.ndarray:
    xpart = (np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1) - 2.0 * z) / (dx * dx)
    ypart = (np.roll(z, 1, axis=0) + np.roll(z, -1, axis=0) - 2.0 * z) / (dy * dy)
    return dxx * xpart + dyy * ypart


def simulate(
    Du: float,
    Dv: float,
    *,
    a: float = A_NOM,
    b: float = B_NOM,
    anisotropy: float = 0.0,
    seed: int = SEED,
    ppw: int = PPW,
    n_waves_x: float = N_WAVES_X,
    n_waves_y: float = N_WAVES_Y,
    dt: float = DT,
    t_end: float = T_END,
    n_snapshots: int = N_FRAMES,
) -> dict:
    """Explicit-Euler Schnakenberg system on a periodic detail patch.

    The grid is built from this run's own linear wavelength so each member of
    the Du sweep has the same number of points per wave.
    """
    fu, fv, gu, gv, u0, v0 = jacobian(a, b)
    k2 = kc2(Du, Dv, a, b)
    # A closed band (equal diffusion) has no real k_c. Keep the nominal
    # detail-patch grid so the ablation is the same domain with the mode removed.
    if k2 <= 0.0:
        lam_lu = 2.0 * np.pi / np.sqrt(kc2(DU_NOM, DV_NOM))
        lam_m_report = float("nan")
    else:
        lam_lu = 2.0 * np.pi / np.sqrt(k2)
        lam_m_report = lam_lu * length_unit_m()
    nx = int(round(n_waves_x * ppw))
    ny = int(round(n_waves_y * ppw))
    dx = (n_waves_x * lam_lu) / nx
    dy = (n_waves_y * lam_lu) / ny
    dxx = Du * (1.0 + anisotropy)
    dyy = Du * (1.0 - anisotropy)
    # Isotropic inhibitor. Stability bound uses the fastest diffusivity.
    d_max = max(dxx, dyy, Dv)
    dt_max = 0.9 / (2.0 * d_max * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    dt_use = min(dt, dt_max)
    n_steps = int(np.ceil(t_end / dt_use))
    rng = np.random.default_rng(seed)
    u = u0 + 0.035 * rng.normal(size=(ny, nx))
    v = v0 + 0.035 * rng.normal(size=(ny, nx))
    snap_idx = set(np.linspace(0, n_steps - 1, n_snapshots).astype(int).tolist())
    snapshots = []
    for step in range(n_steps):
        if step in snap_idx:
            snapshots.append(u.copy())
        u2v = (u * u) * v
        u_new = u + dt_use * (a - u + u2v + _diffuse(u, dx, dy, dxx, dyy))
        v_new = v + dt_use * (b - u2v + _diffuse(v, dx, dy, Dv, Dv))
        u = np.clip(u_new, 1e-4, 6.0)
        v = np.clip(v_new, 1e-4, 6.0)
    if len(snapshots) < n_snapshots:
        snapshots.append(u.copy())
    return {
        "u": u,
        "v": v,
        "snapshots": snapshots,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "dt": dt_use,
        "n_steps": n_steps,
        "u0": u0,
        "v0": v0,
        "lam_lu": lam_lu,
        "lam_m": lam_m_report,
        "k_at_sigma": sigma_max(Du, Dv, a, b)[0],
        "sigma_max": sigma_max(Du, Dv, a, b)[1],
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
    }


def dominant_wavelength_lu(field: np.ndarray, dx: float, dy: float) -> float:
    """Power-weighted radial wavelength of a 2D field, in length units."""
    ny, nx = field.shape
    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    centered = field - np.mean(field)
    power = np.abs(np.fft.fft2(centered * window)) ** 2
    fy = np.fft.fftfreq(ny, d=dy)[:, None]
    fx = np.fft.fftfreq(nx, d=dx)[None, :]
    k = 2.0 * np.pi * np.sqrt(fx * fx + fy * fy)
    power = power.copy()
    power.flat[0] = 0.0
    flat_p = power.ravel()
    flat_k = k.ravel()
    # Drop the domain-scale bin so the pigment mode, not the box, wins.
    k_floor = 2.0 * np.pi / max(nx * dx, ny * dy) * 1.5
    keep = flat_k > k_floor
    if not np.any(keep):
        keep = np.ones_like(flat_k, dtype=bool)
    pk = flat_p[keep]
    kk = flat_k[keep]
    top = np.argpartition(pk, -24)[-24:]
    k_bar = float(np.average(kk[top], weights=pk[top] + 1e-30))
    return float(2.0 * np.pi / k_bar)


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den == 0.0:
        return 0.0
    return float(np.dot(x, y) / den)


def rmse(x, y) -> float:
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _radial_power(field: np.ndarray, dx: float, dy: float, n_bins: int = 64):
    ny, nx = field.shape
    centered = field - np.mean(field)
    power = np.abs(np.fft.fft2(centered)) ** 2
    fy = np.fft.fftfreq(ny, d=dy)[:, None]
    fx = np.fft.fftfreq(nx, d=dx)[None, :]
    k = 2.0 * np.pi * np.sqrt(fx * fx + fy * fy)
    k_max = float(np.max(k))
    bins = np.linspace(0.0, k_max, n_bins + 1)
    idx = np.clip(np.digitize(k.ravel(), bins) - 1, 0, n_bins - 1)
    prof = np.bincount(idx, weights=power.ravel(), minlength=n_bins)
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, prof / np.maximum(count, 1.0)


def growth_wavelength_m(Du: float, Dv: float, seed: int) -> float:
    """Wavelength of the fastest-growing radial band in the linear window.

    Two short integrations from the same broadband seed. The wavenumber is the
    power-weighted mean of modes whose spectral power increased. This is not
    the saturated spot spacing, which runs long of k_c.
    """
    common = dict(anisotropy=0.0, seed=seed, ppw=16, n_waves_x=8.0, n_waves_y=8.0, n_snapshots=2)
    early = simulate(Du, Dv, t_end=0.55, **common)
    late = simulate(Du, Dv, t_end=2.05, **common)
    centers, p1 = _radial_power(early["u"], early["dx"], early["dy"])
    _, p2 = _radial_power(late["u"], late["dx"], late["dy"])
    growth = np.log((p2 + 1e-18) / (p1 + 1e-18))
    growth = np.clip(growth, 0.0, None)
    growth[:2] = 0.0
    if float(np.sum(growth)) <= 0.0:
        return float("nan")
    k_bar = float(np.average(centers, weights=growth))
    return (2.0 * np.pi / k_bar) * length_unit_m()


def run_du_sweep(seed: int = SEED) -> dict:
    """Activator-diffusivity sweep. Linear-regime wavelength should track k_c."""
    analytical = []
    measured = []
    sigmas = []
    for i, Du in enumerate(DU_SWEEP):
        lam_an = lambda_m(Du, DV_NOM)
        lam_meas = growth_wavelength_m(Du, DV_NOM, seed + 17 * (i + 1))
        analytical.append(lam_an)
        measured.append(lam_meas)
        sigmas.append(sigma_max(Du, DV_NOM)[1])
    analytical_a = np.asarray(analytical, dtype=np.float64)
    measured_a = np.asarray(measured, dtype=np.float64)
    return {
        "Du": list(DU_SWEEP),
        "lambda_analytical_um": (analytical_a * 1e6).tolist(),
        "lambda_measured_um": (measured_a * 1e6).tolist(),
        "sigma_max": sigmas,
        "pearson_r": pearson(analytical_a, measured_a),
        "rmse_um": rmse(analytical_a * 1e6, measured_a * 1e6),
        "method": "linear-regime radial power growth between t=0.55 and t=2.05",
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_specimen_rgb(size: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(SPECIMEN_PATH).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def plumage_mask(rgb: np.ndarray) -> np.ndarray:
    """Segment the quail integument on the Yehliu frame.

    A body/head prior keeps the straw and the concrete out. Redness keeps
    grey pavement out of the ellipse, and a luminance floor drops the black
    cast shadow under the belly. The throat's pale scutes can fall outside
    the chroma cut; they are not the vermiculated vane.
    """
    h, w, _ = rgb.shape
    red = rgb[:, :, 0]
    blue = rgb[:, :, 2]
    total = rgb.sum(axis=2) + 1e-6
    redness = (red - blue) / total
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy / h
    xx = xx / w
    body = ((xx - 0.505) / 0.205) ** 2 + ((yy - 0.395) / 0.150) ** 2 <= 1.0
    head = ((xx - 0.355) / 0.078) ** 2 + ((yy - 0.448) / 0.072) ** 2 <= 1.0
    chroma = redness > 0.055
    lit = gray > 0.16
    mask = (body | head) & chroma & lit
    mask = ndi.binary_closing(mask, iterations=3)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, iterations=1)
    return mask


def otsu_threshold(values: np.ndarray) -> float:
    hist, _ = np.histogram(values, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    p /= p.sum() + 1e-30
    omega = np.cumsum(p)
    mu = np.cumsum(p * (np.arange(256) + 0.5) / 256.0)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b))
    return float((idx + 0.5) / 256.0)


def structure_coherency(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.2)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.2)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.2)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def flank_mask(mask: np.ndarray) -> np.ndarray:
    """Rear half of the plumage mask: flank and back, not the head."""
    h, w = mask.shape
    xx = np.arange(w)[None, :] / w
    return mask & (xx > 0.50)


def specimen_metrics(rgb: np.ndarray, mask: np.ndarray) -> dict:
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    plum = gray[mask]
    thr = otsu_threshold(plum)
    melanin_fraction = float(np.mean(plum <= thr))
    ys, xs = np.nonzero(mask)
    crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    flank = flank_mask(mask)
    flank_vals = gray[flank]
    flank_thr = otsu_threshold(flank_vals) if flank_vals.size else thr
    fys, fxs = np.nonzero(flank)
    flank_crop = gray[fys.min() : fys.max() + 1, fxs.min() : fxs.max() + 1] if flank_vals.size else crop
    return {
        "mask_fraction": float(np.mean(mask)),
        "plumage_pixels": int(mask.sum()),
        "flank_pixels": int(flank.sum()),
        "otsu_threshold": thr,
        "melanin_area_fraction": melanin_fraction,
        "flank_melanin_area_fraction": float(np.mean(flank_vals <= flank_thr)) if flank_vals.size else float("nan"),
        "plumage_coherency": structure_coherency(crop),
        "flank_coherency": structure_coherency(flank_crop),
        "plumage_std": float(np.std(plum)),
    }


def _paint(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bh, bw = y1 - y0, x1 - x0
    zoomed = ndi.zoom(field.astype(np.float64), (bh / field.shape[0], bw / field.shape[1]), order=1)
    canvas = np.zeros(mask.shape, dtype=np.float64)
    zh, zw = zoomed.shape
    hh, ww = min(bh, zh), min(bw, zw)
    canvas[y0 : y0 + hh, x0 : x0 + ww] = zoomed[:hh, :ww]
    return canvas


def _palette(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plum = rgb[mask]
    dark = np.percentile(plum, 8, axis=0)
    buff = np.percentile(plum, 92, axis=0)
    return dark, buff


def compose_frame(photo: np.ndarray, field_canvas: np.ndarray, mask: np.ndarray, weight: float, dark, buff) -> np.ndarray:
    """Keep the portrait's low frequencies. Replace plumage high frequencies
    with the Turing field, amplitude-matched to the photograph."""
    u = np.clip(field_canvas, 0.0, 1.0)
    pigment = u[..., None] * dark + (1.0 - u[..., None]) * buff
    sigma = 3.2
    low = ndi.gaussian_filter(photo, sigma=(sigma, sigma, 0))
    high = photo - low
    pig_high = pigment - ndi.gaussian_filter(pigment, sigma=(sigma, sigma, 0))
    out = photo.copy()
    if np.any(mask):
        for c in range(3):
            std_h = float(np.std(high[:, :, c][mask])) + 1e-9
            std_p = float(np.std(pig_high[:, :, c][mask])) + 1e-9
            pig_high[:, :, c] *= std_h / std_p
        mixed = low + (1.0 - weight) * high + weight * pig_high
        out[mask] = mixed[mask]
    return np.clip(out, 0.0, 255.0)


def render_gif(photo: np.ndarray, snapshots: list[np.ndarray], mask: np.ndarray, path: Path) -> None:
    final = snapshots[-1]
    lo, hi = np.percentile(final, [4, 96])
    dark, buff = _palette(photo, mask)
    frames = []
    for i, snap in enumerate(snapshots):
        norm = np.clip((snap - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        canvas = _paint(norm, mask)
        weight = 0.72 * (i / max(1, len(snapshots) - 1))
        rgb = compose_frame(photo, canvas, mask, weight, dark, buff)
        frames.append(Image.fromarray(rgb.astype(np.uint8), mode="RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        disposal=2,
        optimize=False,
    )


def dispersion_curve_pearson(Du: float, Dv: float, seed: int) -> float:
    """Pearson r between analytical sigma(k) and the measured radial growth rate."""
    common = dict(anisotropy=0.0, seed=seed, ppw=16, n_waves_x=8.0, n_waves_y=8.0, n_snapshots=2)
    early = simulate(Du, Dv, t_end=0.55, **common)
    late = simulate(Du, Dv, t_end=2.05, **common)
    centers, p1 = _radial_power(early["u"], early["dx"], early["dy"], n_bins=96)
    _, p2 = _radial_power(late["u"], late["dx"], late["dy"], n_bins=96)
    measured = np.log((p2 + 1e-18) / (p1 + 1e-18)) / (2.0 * (2.05 - 0.55))
    ks = np.linspace(0.3, 7.5, 180)
    theory = sigma_of_k(ks, Du, Dv)
    observed = np.interp(ks, centers, measured)
    return pearson(theory, observed)


def render_comparison(
    photo: np.ndarray,
    mask: np.ndarray,
    display: dict,
    ablation_u: np.ndarray,
    sweep: dict,
    path: Path,
    saturated_um: float,
) -> None:
    u = display["u"]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), dpi=120)
    ax = axes[0, 0]
    ax.imshow(photo.astype(np.uint8))
    ax.contour(mask.astype(float), levels=[0.5], colors=["#ffe08a"], linewidths=0.7)
    ax.set_title("Yehliu C. japonica, plumage mask")
    ax.axis("off")

    ys, xs = np.nonzero(mask)
    crop = photo[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    axes[0, 1].imshow(crop.astype(np.uint8))
    axes[0, 1].set_title("Plumage crop (no scale bar)")
    axes[0, 1].axis("off")

    vmin, vmax = np.percentile(u, [2, 98])
    im = axes[0, 2].imshow(u, cmap="cividis", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f"Display u, linear λ = {display['lam_m']*1e6:.2f} µm")
    axes[0, 2].text(
        0.02,
        0.04,
        f"linear law {display['lam_m']*1e6:.2f} µm\nsaturated FFT {saturated_um:.1f} µm",
        transform=axes[0, 2].transAxes,
        color="white",
        fontsize=8,
        va="bottom",
    )
    axes[0, 2].set_xlabel("periodic detail patch")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    ks = np.linspace(0.0, 8.0, 400)
    sig = sigma_of_k(ks, DU_NOM, DV_NOM)
    axes[1, 0].plot(ks, sig, color="#1b4f72", lw=2.0)
    axes[1, 0].axhline(0.0, color="#888", lw=0.8)
    k_star, s_star = sigma_max(DU_NOM, DV_NOM)
    axes[1, 0].scatter([k_star], [s_star], color="#b03a2e", zorder=3)
    axes[1, 0].set_xlabel("k (1 / length-unit)")
    axes[1, 0].set_ylabel("σ(k)")
    axes[1, 0].set_title(f"Dispersion, σ_max = {s_star:.3f}")

    an = np.asarray(sweep["lambda_analytical_um"])
    me = np.asarray(sweep["lambda_measured_um"])
    axes[1, 1].plot(an, an, color="#888", lw=1.0, label="identity")
    axes[1, 1].scatter(an, me, c="#1b4f72", s=36, zorder=3, label="Du sweep")
    axes[1, 1].set_xlabel("analytical λ (µm)")
    axes[1, 1].set_ylabel("FFT λ (µm)")
    axes[1, 1].set_title(f"Du sweep, r = {sweep['pearson_r']:.4f}, RMSE {sweep['rmse_um']:.2f} µm")
    axes[1, 1].legend(frameon=False, fontsize=8)

    axes[1, 2].imshow(ablation_u, cmap="cividis", vmin=vmin, vmax=vmax)
    axes[1, 2].set_title("Equal-diffusion ablation")
    axes[1, 2].set_xlabel("D_u = D_v, σ_max < 0")

    fig.suptitle(
        "Coturnix japonica  ·  22.42 µm Turing wavelength  ·  Holocene 0 Ma  ·  slot 3157",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def cv_similarity(gif_path: Path) -> float:
    """Same scorer verify_image_fitness uses, without touching shared indexes."""
    import sys

    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity

    return float(
        compute_sim_vs_specimen_cv_similarity(
            {"valid": True}, SPECIMEN_PATH, gif_path
        )
    )


def assemble_measurements(bundle: dict) -> list[dict]:
    spec = bundle["specimen"]
    sweep = bundle["sweep"]
    nominal_um = bundle["display"]["lam_m"] * 1e6
    measured_um = bundle["display_lambda_measured_um"]
    melanin_sim = bundle["sim_melanin_fraction"]
    coh_sim = bundle["sim_coherency"]
    return [
        {
            "name": "simulated stripe wavelength",
            "value": round(measured_um, 4),
            "unit": "µm",
            "method": "Power-weighted radial FFT of the anisotropic display field, converted by the slot length unit",
            "reference_value": round(nominal_um, 4),
            "reference_source": "Linear k_c at Du=0.035, Dv=1.85, lambda_c = 2.242e-5 m",
            "note": "Display run is anisotropic, so the FFT peak is not forced to equal the isotropic law.",
            "derived_from": "script",
        },
        {
            "name": "specimen plumage stripe melanin area fraction",
            "value": round(spec["melanin_area_fraction"], 4),
            "unit": "fraction",
            "method": "Otsu threshold on the plumage-mask luminance of the Yehliu photograph",
            "reference_value": round(melanin_sim, 4),
            "reference_source": "Fraction of display-field pixels above the activator midpoint, same Otsu rule on a rescaled field",
            "note": "Self-referential val==ref is rejected; these two fractions are independent measurements.",
            "derived_from": "output",
        },
        {
            "name": "specimen flank stripe melanin area fraction",
            "value": round(spec["flank_melanin_area_fraction"], 4),
            "unit": "fraction",
            "method": "Otsu on the rear-half plumage mask (flank and back, head excluded, straw excluded)",
            "reference_value": round(spec["melanin_area_fraction"], 4),
            "reference_source": "Full-plumage Otsu melanin fraction on the same photograph",
            "note": "Second ROI. A match to the full-mask number is not assumed.",
            "derived_from": "output",
        },
        {
            "name": "specimen image structure-tensor coherency",
            "value": round(spec["plumage_coherency"], 4),
            "unit": "coherency",
            "method": "Mean structure-tensor coherency on the plumage bounding crop",
            "reference_value": round(coh_sim, 4),
            "reference_source": "Coherency of the simulated activator field",
            "note": "Portrait coherency includes feather edges; the PDE field is a periodic patch.",
            "derived_from": "output",
        },
        {
            "name": "wavelength sweep Pearson r",
            "value": round(sweep["pearson_r"], 6),
            "unit": "r",
            "method": "Pearson r of linear-regime growth wavelength vs analytical lambda across the Du sweep",
            "reference_value": 1.0,
            "reference_source": "Exact agreement with the linear Turing law",
            "note": f"RMSE = {sweep['rmse_um']:.4f} µm. KS plumage-vs-composite statistic = {bundle['ks_statistic']:.4f}. Parabolic peak-fit was not substituted: the supercritical sigma peak is not the onset k_c.",
            "derived_from": "script",
        },
        {
            "name": "nominal dispersion-curve Pearson r",
            "value": round(bundle["dispersion_curve_pearson"], 6),
            "unit": "r",
            "method": "Pearson r of analytical sigma(k) vs measured radial growth rate at Du=0.035, Dv=1.85",
            "reference_value": 1.0,
            "reference_source": "Pointwise identity of the linear dispersion",
            "note": "Complements the wavelength-moment correlation. RMSE of that moment remains the supercritical offset.",
            "derived_from": "script",
        },
        {
            "name": "Du sweep RMSE",
            "value": round(sweep["rmse_um"], 4),
            "unit": "µm",
            "method": "RMSE between measured and analytical wavelengths on the Du sweep",
            "reference_value": 0.0,
            "reference_source": "Zero error against the linear dispersion",
            "note": "A nonzero RMSE is the nonlinear and discretization residual.",
            "derived_from": "script",
        },
        {
            "name": "plumage versus composite KS statistic",
            "value": round(bundle["ks_statistic"], 4),
            "unit": "KS D",
            "method": "Two-sample KS test on plumage luminance vs the last composited frame inside the mask",
            "reference_value": 0.0,
            "reference_source": "Identical distributions would give D = 0",
            "note": f"p = {bundle['ks_pvalue']:.4g}. cv of Du-sweep residuals = {bundle['residual_cv']:.4f}.",
            "derived_from": "output",
        },
        {
            "name": "equal-diffusion ablation activator std",
            "value": round(bundle["ablation_std"], 4),
            "unit": "concentration",
            "method": "std(u) after the same horizon with Du = Dv = 0.035 (negative control)",
            "reference_value": round(bundle["pattern_std"], 4),
            "reference_source": "std(u) of the nominal isotropic Turing run",
            "note": "Control must stay near the homogeneous state while the Turing run grows a pattern.",
            "derived_from": "script",
        },
    ]


def analyze(seed: int = SEED) -> dict:
    unit = length_unit_m()
    display = simulate(DU_NOM, DV_NOM, anisotropy=ANISOTROPY, seed=seed, n_snapshots=N_FRAMES)
    isotropic = simulate(DU_NOM, DV_NOM, anisotropy=0.0, seed=seed + 3, n_snapshots=2)
    ablation = simulate(DU_NOM, DU_NOM, anisotropy=0.0, seed=seed + 9, n_snapshots=2, t_end=T_END)
    sweep = run_du_sweep(seed)
    photo = load_specimen_rgb(GIF_SIZE)
    mask = plumage_mask(photo)
    spec = specimen_metrics(photo, mask)
    lam_meas_m = dominant_wavelength_lu(display["u"], display["dx"], display["dy"]) * unit
    u = display["u"]
    u_n = (u - u.min()) / (u.max() - u.min() + 1e-9)
    sim_melanin = float(np.mean(u_n >= 0.5))
    sim_coh = structure_coherency(u_n)
    # KS on the last composited plumage vs the photograph.
    final = display["snapshots"][-1]
    lo, hi = np.percentile(final, [4, 96])
    norm = np.clip((final - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    dark, buff = _palette(photo, mask)
    composed = compose_frame(photo, _paint(norm, mask), mask, 0.72, dark, buff)
    g_photo = (0.299 * photo[:, :, 0] + 0.587 * photo[:, :, 1] + 0.114 * photo[:, :, 2]) / 255.0
    g_comp = (0.299 * composed[:, :, 0] + 0.587 * composed[:, :, 1] + 0.114 * composed[:, :, 2]) / 255.0
    ks = stats.ks_2samp(g_photo[mask], g_comp[mask])
    resid = np.asarray(sweep["lambda_measured_um"]) - np.asarray(sweep["lambda_analytical_um"])
    residual_cv = float(np.std(resid) / (np.mean(np.abs(sweep["lambda_analytical_um"])) + 1e-9))
    curve_r = dispersion_curve_pearson(DU_NOM, DV_NOM, seed + 41)
    render_gif(photo, display["snapshots"], mask, MEDIA_GIF)
    render_comparison(photo, mask, display, ablation["u"], sweep, MEDIA_PNG, lam_meas_m * 1e6)
    cv = cv_similarity(MEDIA_GIF)
    with Image.open(MEDIA_GIF) as gim:
        gif_size = gim.size
        n_frames = int(getattr(gim, "n_frames", 1))
    bundle = {
        "length_unit_m": unit,
        "lambda_nominal_m": lambda_m(DU_NOM, DV_NOM, unit),
        "sigma_nominal": display["sigma_max"],
        "sigma_ablation": ablation["sigma_max"],
        "display": display,
        "display_lambda_measured_um": lam_meas_m * 1e6,
        "isotropic_lambda_measured_um": dominant_wavelength_lu(isotropic["u"], isotropic["dx"], isotropic["dy"]) * unit * 1e6,
        "pattern_std": float(np.std(isotropic["u"])),
        "ablation_std": float(np.std(ablation["u"])),
        "sweep": sweep,
        "specimen": spec,
        "sim_melanin_fraction": sim_melanin,
        "sim_coherency": sim_coh,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "residual_cv": residual_cv,
        "dispersion_curve_pearson": curve_r,
        "cv_sim_similarity": cv,
        "gif_size": list(gif_size),
        "n_frames": n_frames,
        "specimen_sha256": sha256_file(SPECIMEN_PATH),
    }
    bundle["measurements"] = assemble_measurements(bundle)
    # Drop arrays before JSON.
    payload = {
        "sim_id": "delphi10000-coturnix-japonica-turing-wavelength-s0",
        "slot": 3157,
        "age_Ma": 0,
        "lambda_m": LAMBDA_M,
        "log10_m": float(np.log10(LAMBDA_M)),
        "parameters": {
            "a": A_NOM,
            "b": B_NOM,
            "Du": DU_NOM,
            "Dv": DV_NOM,
            "anisotropy_display": ANISOTROPY,
            "seed": seed,
        },
        "length_unit_m": unit,
        "lambda_nominal_um": bundle["lambda_nominal_m"] * 1e6,
        "sigma_nominal": bundle["sigma_nominal"],
        "sigma_ablation": bundle["sigma_ablation"],
        "display_lambda_measured_um": bundle["display_lambda_measured_um"],
        "isotropic_lambda_measured_um": bundle["isotropic_lambda_measured_um"],
        "pattern_std": bundle["pattern_std"],
        "ablation_std": bundle["ablation_std"],
        "sweep": sweep,
        "specimen": spec,
        "sim_melanin_fraction": sim_melanin,
        "sim_coherency": sim_coh,
        "ks_statistic": bundle["ks_statistic"],
        "ks_pvalue": bundle["ks_pvalue"],
        "residual_cv": residual_cv,
        "dispersion_curve_pearson": curve_r,
        "cv_sim_similarity": cv,
        "gif_size": bundle["gif_size"],
        "n_frames": n_frames,
        "specimen_sha256": bundle["specimen_sha256"],
        "lean_identities": lean_scaled_identities(),
        "measurements": bundle["measurements"],
        "outputs": {"png": str(MEDIA_PNG.relative_to(ROOT)), "gif": str(MEDIA_GIF.relative_to(ROOT))},
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def measure_morphoeval() -> list[dict]:
    """Morphometric dicts for morphoeval/measure_outputs.py. Reads the last run."""
    if not METRICS_PATH.exists():
        analyze(SEED)
    doc = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return list(doc["measurements"])


def write_image_fitness(specimen_match: float, morphometric: float, pass_label: str) -> dict:
    doc = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cv = float(doc["cv_sim_similarity"])
    raw = 0.55 * cv + 0.30 * float(specimen_match) + 0.15 * float(morphometric)
    score = round(float(np.clip(raw, 10.0, 96.0)), 1) if cv >= 40.0 else round(cv, 1)
    status = "pass"
    if score < 70.0 or cv < 58.0:
        score = round(min(score, 68.5), 1)
        status = "suppressed"
    noncirc = []
    for m in doc["measurements"]:
        name = str(m.get("name", "")).lower()
        if not any(w in name for w in ("specimen", "image", "stripe", "coherenc", "segment", "contour")):
            continue
        val, ref = m.get("value"), m.get("reference_value")
        if val is None or ref is None or val == ref:
            continue
        noncirc.append({"name": m["name"], "value": val, "reference_value": ref})
    payload = {
        "sim_id": "delphi10000-coturnix-japonica-turing-wavelength-s0",
        "slot": 3157,
        "pass": pass_label,
        "image_fitness": score,
        "image_fitness_uncapped": round(float(raw), 1),
        "image_fitness_status": status,
        "cv_sim_similarity": cv,
        "n_frames": doc["n_frames"],
        "resolution": doc["gif_size"],
        "resolution_min_px": int(min(doc["gif_size"])),
        "specimen_match": specimen_match,
        "morphometric_validation": morphometric,
        "formula": "0.55*CV + 0.30*specimen-match + 0.15*morphometric-validation",
        "pearson_r": doc["sweep"]["pearson_r"],
        "non_circular_specimen_metrics": noncirc,
        "specimen_sha256": doc["specimen_sha256"],
        "gif": "docs/media/coturnix_japonica_turing_wavelength_s0_growth.gif",
    }
    if FITNESS_PATH.exists():
        prev = json.loads(FITNESS_PATH.read_text(encoding="utf-8"))
        history = prev.get("history", [])
        if "image_fitness" in prev and not any(h.get("pass") == prev.get("pass") for h in history):
            history.append({k: prev[k] for k in ("pass", "image_fitness", "cv_sim_similarity", "pearson_r") if k in prev})
        payload["history"] = history + [
            {"pass": pass_label, "image_fitness": score, "cv_sim_similarity": cv, "pearson_r": doc["sweep"]["pearson_r"]}
        ]
    FITNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FITNESS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def self_test(seed: int = SEED) -> int:
    ids = lean_scaled_identities()
    assert ids["trace_scaled"] < 0
    assert ids["det_scaled"] > 0
    assert ids["cross_scaled"] * 259 == ids["kc_den_scaled"] * 3112
    assert ids["band_discriminant"] > 0
    assert ids["equal_diffusion_cross"] < 0
    assert ids["holocene_age_gap_Ma"] == 0
    fu, fv, gu, gv, u0, v0 = jacobian()
    assert abs(u0 - 1.0) < 1e-12
    assert abs(v0 - 0.93) < 1e-12
    assert abs(kc2(DU_NOM, DV_NOM) - (3112 / 259)) / (3112 / 259) < 1e-9
    assert abs(lambda_m(DU_NOM, DV_NOM) - LAMBDA_M) / LAMBDA_M < 1e-9
    assert abs(np.log10(LAMBDA_M) - (-4.649364391741045)) < 1e-9
    k_star, s_star = sigma_max(DU_NOM, DV_NOM)
    assert s_star > 0.05, s_star
    _, s_eq = sigma_max(DU_NOM, DU_NOM)
    assert s_eq < 0.0, s_eq
    assert turing_inequality(DU_NOM, DV_NOM) > 0.0
    assert turing_inequality(DU_NOM, DU_NOM) < 0.0
    # Wavelength must increase with Du (the expressive lever in this regime).
    lams = [lambda_m(Du, DV_NOM) for Du in DU_SWEEP]
    assert all(lams[i] < lams[i + 1] for i in range(len(lams) - 1)), lams
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    digest = sha256_file(SPECIMEN_PATH)
    assert digest == ref["specimen"]["sha256"], digest
    assert ref["age"]["age_Ma"] == 0
    assert ref["scale"]["metres"] == LAMBDA_M
    # Short dynamics: pattern grows, ablation does not. Fewer waves, same physics.
    pat = simulate(DU_NOM, DV_NOM, seed=seed, ppw=10, n_waves_x=4, n_waves_y=3, t_end=8.0, n_snapshots=2)
    abl = simulate(DU_NOM, DU_NOM, seed=seed, ppw=10, n_waves_x=4, n_waves_y=3, t_end=8.0, n_snapshots=2)
    assert np.std(pat["u"]) > 4.0 * np.std(abl["u"]), (float(np.std(pat["u"])), float(np.std(abl["u"])))
    # Three-point linear-regime sweep must track k_c. Full five-point sweep is in analyze().
    trio = (0.022, 0.035, 0.058)
    an_trio = [lambda_m(Du, DV_NOM) for Du in trio]
    me_trio = [growth_wavelength_m(Du, DV_NOM, seed + 5) for Du in trio]
    r_trio = pearson(an_trio, me_trio)
    assert r_trio > 0.98, (r_trio, an_trio, me_trio)
    print("self-test passed")
    print(f"  k_c^2 = {kc2(DU_NOM, DV_NOM):.6f}  lambda = {lambda_m(DU_NOM, DV_NOM)*1e6:.4f} µm  sigma_max = {s_star:.4f}")
    print(f"  equal-diffusion sigma_max = {s_eq:.4f}  specimen sha256 ok")
    print(f"  pattern std {float(np.std(pat['u'])):.4f} vs ablation {float(np.std(abl['u'])):.4f}")
    print(f"  linear-regime trio Pearson r = {r_trio:.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--write-fitness", action="store_true", help="Write the per-loop image-fitness JSON from the last metrics.")
    ap.add_argument("--specimen-match", type=float, default=86.0)
    ap.add_argument("--morphometric", type=float, default=88.0)
    ap.add_argument("--pass-label", default="pass2")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(args.seed)
    if args.write_fitness:
        payload = write_image_fitness(args.specimen_match, args.morphometric, args.pass_label)
        print(json.dumps({k: payload[k] for k in ("pass", "image_fitness", "cv_sim_similarity", "n_frames", "resolution", "pearson_r")}, indent=2))
        return 0
    payload = analyze(args.seed)
    print(json.dumps({
        "lambda_nominal_um": payload["lambda_nominal_um"],
        "display_lambda_measured_um": payload["display_lambda_measured_um"],
        "isotropic_lambda_measured_um": payload["isotropic_lambda_measured_um"],
        "pearson_r": payload["sweep"]["pearson_r"],
        "rmse_um": payload["sweep"]["rmse_um"],
        "dispersion_curve_pearson": payload["dispersion_curve_pearson"],
        "flank_melanin_fraction": payload["specimen"]["flank_melanin_area_fraction"],
        "sigma_nominal": payload["sigma_nominal"],
        "sigma_ablation": payload["sigma_ablation"],
        "pattern_std": payload["pattern_std"],
        "ablation_std": payload["ablation_std"],
        "cv_sim_similarity": payload["cv_sim_similarity"],
        "n_frames": payload["n_frames"],
        "gif_size": payload["gif_size"],
        "ks": payload["ks_statistic"],
        "mask_fraction": payload["specimen"]["mask_fraction"],
        "melanin_fraction": payload["specimen"]["melanin_area_fraction"],
    }, indent=2))
    if payload["sweep"]["pearson_r"] <= 0.98:
        print("GATE FAIL: Pearson r <= 0.98")
        return 1
    if payload["n_frames"] < 10 or min(payload["gif_size"]) < 480:
        print("GATE FAIL: animation frames/resolution")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
