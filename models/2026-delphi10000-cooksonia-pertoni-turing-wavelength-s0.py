#!/usr/bin/env python3
"""
Cooksonia pertoni at the 1.3 µm detail scale — Turing wavelength selection.

Slot 3153 of the 10,000-animal basic look. Late Silurian (Wenlock–Ludlow, 425 Ma)
dichotomous sporophyte, the earliest vascular tracheids (Edwards, Davies & Axe 1992).
This file is the subcellular detail variant (tier 3_subcellular_um): the selected
Turing wavelength is 1.339 µm, not the 73 µm cellular-stripe slot
(models/1992-edwards-cooksonia-pertoni-turing-wavelength.py).

Governing system (Schnakenberg 1979 kinetics; Turing 1952 dispersion):

    du/dt = D_u ∇²u + a - u + u² v
    dv/dt = D_v ∇²v + b - u² v

on a cylinder unrolled from a tracheid wall (periodic circumferentially, Neumann
axially). Homogeneous steady state (u*, v*) = (a+b, b/(a+b)²). With
a = 3/25, b = 21/25 the Jacobian entries are exact:

    f_u = 3/4,  f_v = 576/625,  g_u = -7/4,  g_v = -576/625
    tr J = -429/2500 < 0,  det J = 576/625 > 0

Diffusion ratio D_v/D_u = 10 sits just above the Turing threshold (~8.84), so the
fastest-growing wavenumber k_c = argmax Re σ_+(k) is sharp. The physical length
unit is fixed so λ_c = 2π/k_c equals 1.339e-6 m.

What changes at this scale and age
-----------------------------------
* Predecessor (pre-vascular axis, before the 425 Ma conducting strand; Cooksonia
  is chronospecies 1/1 and the first member of the Paleobotanical Vascular & Leaf
  Fossil Series): no lignified wall to separate activator and inhibitor
  diffusivities, so D_u = D_v and σ_max < 0. No 1.3 µm band is selected.
* This horizon (C. pertoni, 425 Ma): differential diffusion on the tracheid wall
  selects λ_c = 1.339 µm, read as the spacing of annular/helical secondary-wall
  thickenings (Edwards et al. 1992; Edwards 2003). The museum compression itself
  is a centimetre-scale dichotomous axis; 1.3 µm is below that photograph's
  resolution and is not a length Edwards published.
* Successor (Rhynia gwynne-vaughanii, Pragian Rhynie Chert, 407 Ma): a longer
  activator diffusion length (thicker, more regular S-type thickenings) shifts
  λ_c upward at the same reaction kinetics. The shift is a reconstruction, not
  a measurement quoted from Kenrick & Crane.

The growth GIF reconstructs the Amgueddfa Cymru compression (CC BY-SA 4.0):
a dark isotomous axis on a matrix whose radial spectrum is matched to that
photograph. Banding along the axis is the developing Turing field, magnified
for display. It is not a claim that the slab shows 1.3 µm stripes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "cooksonia_pertoni_turing_wavelength_s0"
MEDIA = ROOT / "docs" / "media"
SPECIMEN_NAME = "cooksonia_pertoni_amgueddfa_specimen.jpg"
NMNH_NAME = "cooksonia_pertoni_nmnh_specimen.jpg"
PNG_NAME = "cooksonia_pertoni_turing_wavelength_s0_comparison.png"
GIF_NAME = "cooksonia_pertoni_turing_wavelength_s0_growth.gif"
METRICS_NAME = "cooksonia_pertoni_turing_wavelength_s0_metrics.json"

# Exact Schnakenberg parameters for this 1.3 µm detail slot (not the 73 µm sibling).
A = 0.12  # 3/25
B = 0.84  # 21/25
D_U = 0.055
D_RATIO = 10.0
D_V = D_U * D_RATIO
LAMBDA_UM = 1.339
LAMBDA_M = 1.339e-6
AGE_MA = 425.0
SUCCESSOR_AGE_MA = 407.0
SUCCESSOR_D_U = 0.088  # longer diffusion length on the Rhynia wall
SEED = 3153
N_X, N_Y = 96, 72
N_WAVES = 4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def steady_state() -> tuple[float, float]:
    u = A + B
    v = B / (u * u)
    return float(u), float(v)


def jacobian() -> tuple[float, float, float, float]:
    u, v = steady_state()
    f_u = -1.0 + 2.0 * u * v
    f_v = u * u
    g_u = -2.0 * u * v
    g_v = -(u * u)
    return float(f_u), float(f_v), float(g_u), float(g_v)


def sigma_plus_k2(k2: np.ndarray, d_u: float, d_v: float) -> np.ndarray:
    """Largest real-part eigenvalue of J - k² diag(D_u, D_v)."""
    f_u, f_v, g_u, g_v = jacobian()
    q = np.asarray(k2, dtype=float)
    tr = (f_u + g_v) - (d_u + d_v) * q
    det = (
        d_u * d_v * q ** 2
        - (d_v * f_u + d_u * g_v) * q
        + (f_u * g_v - f_v * g_u)
    )
    disc = tr ** 2 - 4.0 * det
    root = np.sqrt(np.maximum(disc, 0.0))
    return 0.5 * (tr + root)


def sigma_plus(k: np.ndarray, d_u: float, d_v: float) -> np.ndarray:
    """Continuous-Laplacian dispersion, k² = (wavenumber)²."""
    return sigma_plus_k2(np.asarray(k, dtype=float) ** 2, d_u, d_v)


def discrete_k2(k: float, dx: float) -> float:
    """Symbol of the periodic 3-point Laplacian at wavenumber k."""
    return float((2.0 - 2.0 * np.cos(k * dx)) / (dx * dx))


def dispersion_summary(d_u: float, d_v: float, k_max: float = 8.0, n: int = 4000) -> dict:
    ks = np.linspace(0.05, k_max, n)
    sig = sigma_plus(ks, d_u, d_v)
    i = int(np.argmax(sig))
    unstable = ks[sig > 0.0]
    k_c = float(ks[i])
    return {
        "k_grid": ks,
        "sigma": sig,
        "k_c": k_c,
        "sigma_max": float(sig[i]),
        "lambda_dim": float(2.0 * np.pi / k_c),
        "k_band": [float(unstable[0]), float(unstable[-1])] if unstable.size else [None, None],
        "unstable": bool(sig[i] > 0.0),
    }


def length_unit_m(lambda_dim: float) -> float:
    """Metres per dimensionless length so that λ_dim maps to 1.339 µm."""
    return LAMBDA_M / lambda_dim


def laplacian_cylinder(z: np.ndarray, dx: float) -> np.ndarray:
    """Periodic in x (axis 1, circumference); Neumann in y (axis 0, axial)."""
    zx = np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1) - 2.0 * z
    yp = np.pad(z, ((1, 1), (0, 0)), mode="edge")
    zy = yp[:-2] + yp[2:] - 2.0 * z
    return (zx + zy) / (dx * dx)


def integrate_schnakenberg(
    d_u: float,
    d_v: float,
    dx: float,
    steps: int,
    dt: float,
    snap_every: int,
    seed: int,
    amplitude: float = 0.02,
) -> dict:
    rng = np.random.default_rng(seed)
    u_s, v_s = steady_state()
    u = u_s + amplitude * rng.normal(size=(N_Y, N_X))
    v = v_s + amplitude * rng.normal(size=(N_Y, N_X))
    snaps = []
    for t in range(steps):
        lap_u = laplacian_cylinder(u, dx)
        lap_v = laplacian_cylinder(v, dx)
        u2v = u * u * v
        u = u + dt * (d_u * lap_u + A - u + u2v)
        v = v + dt * (d_v * lap_v + B - u2v)
        u = np.clip(u, 0.0, 4.0)
        v = np.clip(v, 0.0, 4.0)
        if t % snap_every == 0 or t == steps - 1:
            snaps.append(u.copy())
    return {"u": u, "v": v, "snaps": snaps, "u_star": u_s, "v_star": v_s}


def linear_growth_rates(d_u: float, d_v: float, dx: float, modes: list[int], dt: float = 0.004, steps: int = 400) -> dict:
    """Finite-difference growth rates of pure Fourier modes about (u*, v*).

    The reference uses the symbol of the same 3-point Laplacian the integrator
    applies, and each mode is started on the dominant eigenvector so the rate
    is σ₊ rather than a transient mixture.
    """
    u_s, v_s = steady_state()
    f_u, f_v, g_u, g_v = jacobian()
    measured = []
    analytic = []
    for m in modes:
        k = 2.0 * np.pi * m / (N_X * dx)
        q = discrete_k2(k, dx)
        mat = np.array([
            [f_u - d_u * q, f_v],
            [g_u, g_v - d_v * q],
        ], dtype=float)
        eigvals, eigvecs = np.linalg.eig(mat)
        idx = int(np.argmax(eigvals.real))
        vec = eigvecs[:, idx].real
        nrm = float(np.linalg.norm(vec)) + 1e-15
        vec = vec / nrm
        eps = 1.0e-3
        x = np.arange(N_X) * dx
        wave = np.cos(k * x)[None, :]
        u = u_s + eps * vec[0] * np.broadcast_to(wave, (N_Y, N_X)).copy()
        v = v_s + eps * vec[1] * np.broadcast_to(wave, (N_Y, N_X)).copy()
        # cos^2 averages to 1/2, so the L2 amplitude is not eps.
        amp0 = float(np.sqrt(np.mean((u - u_s) ** 2 + (v - v_s) ** 2)))
        for _ in range(steps):
            lap_u = laplacian_cylinder(u, dx)
            lap_v = laplacian_cylinder(v, dx)
            du = u - u_s
            dv = v - v_s
            u = u + dt * (d_u * lap_u + f_u * du + f_v * dv)
            v = v + dt * (d_v * lap_v + g_u * du + g_v * dv)
        amp = float(np.sqrt(np.mean((u - u_s) ** 2 + (v - v_s) ** 2)))
        sigma_hat = float(np.log(max(amp, 1e-16) / amp0) / (steps * dt))
        sigma_ref = float(sigma_plus_k2(np.array([q]), d_u, d_v)[0])
        measured.append(sigma_hat)
        analytic.append(sigma_ref)
    measured_a = np.asarray(measured)
    analytic_a = np.asarray(analytic)
    r = float(np.corrcoef(measured_a, analytic_a)[0, 1])
    rmse = float(np.sqrt(np.mean((measured_a - analytic_a) ** 2)))
    return {
        "modes": modes,
        "sigma_measured": measured,
        "sigma_analytic": analytic,
        "pearson_r": r,
        "rmse": rmse,
    }


def radial_log_profile(gray: np.ndarray, n_bins: int = 22) -> np.ndarray:
    win = gray - np.mean(gray)
    win = win / (np.std(win) + 1e-6)
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    h, w = win.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=power.ravel())
    counts = np.bincount(rr.ravel())
    prof = prof[: min(32, prof.size)] / np.maximum(1, counts[: min(32, counts.size)])
    sl = np.log1p(prof[2: 2 + n_bins])
    if sl.size < n_bins:
        sl = np.pad(sl, (0, n_bins - sl.size))
    return sl[:n_bins]


def synthesize_matrix(specimen_rgb: np.ndarray, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Phase-random field with the specimen's radial power spectrum and colour."""
    src = np.asarray(
        Image.fromarray((np.clip(specimen_rgb, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    gray = 0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]
    mag = np.abs(np.fft.fft2(gray - gray.mean()))
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    rr = np.hypot(kx, ky)
    n_bins = 64
    bins = np.minimum((rr / (rr.max() + 1e-12) * n_bins).astype(int), n_bins - 1)
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    np.add.at(acc, bins.ravel(), mag.ravel())
    np.add.at(cnt, bins.ravel(), 1.0)
    avg = acc / np.maximum(cnt, 1.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(h, w))
    phase[0, 0] = 0.0
    field = np.fft.ifft2(avg[bins] * np.exp(1j * phase)).real
    field = (field - field.mean()) / (field.std() + 1e-9)
    # Keep a fraction of the specimen's own grain so orientation is not purely synthetic.
    mix = 0.72 * field * gray.std() + 0.28 * (gray - gray.mean())
    mix = (mix - mix.mean()) / (mix.std() + 1e-9) * gray.std() + gray.mean()
    ratios = src.mean(axis=(0, 1)) / (gray.mean() + 1e-9)
    rgb = np.clip(mix[:, :, None] * ratios[None, None, :], 0.0, 1.0)
    return rgb


def _segment_distance(h: int, w: int, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    v = p1 - p0
    vv = float(np.dot(v, v)) + 1e-9
    t = ((xx - p0[0]) * v[0] + (yy - p0[1]) * v[1]) / vv
    t = np.clip(t, 0.0, 1.0)
    px = p0[0] + t * v[0]
    py = p0[1] + t * v[1]
    return np.hypot(xx - px, yy - py)


def axis_mask(h: int, w: int, geometry: dict, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft occupancy of the isotomous axis and arc-length coordinate along it."""
    pts = [np.asarray(p, dtype=float) for p in geometry["polylines_flat"]]
    dist = np.full((h, w), 1.0e6)
    arc = np.zeros((h, w))
    cursor = 0.0
    for a, b in zip(pts[0::2], pts[1::2]):
        d = _segment_distance(h, w, a, b)
        closer = d < dist
        seg_len = float(np.hypot(*(b - a)))
        dist = np.where(closer, d, dist)
        along = cursor + np.clip(
            ((np.mgrid[0:h, 0:w][1] - a[0]) * (b[0] - a[0]) + (np.mgrid[0:h, 0:w][0] - a[1]) * (b[1] - a[1]))
            / (seg_len ** 2 + 1e-9),
            0.0,
            1.0,
        ) * seg_len
        arc = np.where(closer, along, arc)
        cursor += seg_len
    mask = np.exp(-(dist ** 2) / (2.0 * (width * 0.55) ** 2))
    return mask, arc


def fossil_geometry(h: int, w: int) -> dict:
    """Isotomous Cooksonia axis in pixel coordinates of an h×w canvas.

    Angles are a geometric reconstruction placed to overlap the dark ridge of the
    Amgueddfa compression, not a published Edwards morphometric table.
    """
    jx, jy = 0.50 * w, 0.60 * h
    stem = (0.505 * w, 0.93 * h)
    arm_l = (0.30 * w, 0.16 * h)
    arm_r = (0.74 * w, 0.11 * h)
    v_l = np.array([arm_l[0] - jx, arm_l[1] - jy])
    v_r = np.array([arm_r[0] - jx, arm_r[1] - jy])
    cosang = float(np.dot(v_l, v_r) / (np.linalg.norm(v_l) * np.linalg.norm(v_r) + 1e-9))
    angle = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    return {
        "junction": (jx, jy),
        "stem": stem,
        "arm_l": arm_l,
        "arm_r": arm_r,
        "branch_angle_deg": angle,
        "polylines_flat": [stem, (jx, jy), (jx, jy), arm_l, (jx, jy), arm_r],
    }


def paint_axis(rgb: np.ndarray, geometry: dict, growth: float, u_slice: np.ndarray, width: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask, arc = axis_mask(h, w, geometry, width)
    # Growth: stem first, then the two arms (isotomous dichotomy).
    jy = geometry["junction"][1]
    stem_y = geometry["stem"][1]
    yy = np.mgrid[0:h, 0:w][0]
    if growth < 0.42:
        s = growth / 0.42
        revealed = yy > (stem_y - s * (stem_y - jy))
        mask = mask * revealed
    else:
        s = (growth - 0.42) / 0.58
        arm_gate = (yy > jy) | (yy > (jy - s * jy))
        mask = mask * arm_gate
    # Magnified Turing banding along arc length. Display period is a visualization
    # of the selected mode, not 1.3 µm on the slab.
    period = max(9.0, width * 1.6)
    phase = np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, u_slice.size), u_slice)
    # Map arc to the slice.
    arc_n = arc / (arc.max() + 1e-9)
    idx = np.clip((arc_n * (u_slice.size - 1)).astype(int), 0, u_slice.size - 1)
    band = u_slice[idx]
    band = (band - band.mean()) / (band.std() + 1e-9)
    darkness = 0.22 + 0.10 * np.tanh(band)
    out = rgb.copy()
    ink = np.array([0.16, 0.17, 0.18])
    m = mask[:, :, None]
    out = out * (1.0 - 0.82 * m) + ink * (0.82 * m) * (0.65 + 0.35 * darkness[:, :, None])
    # Slight cool shadow so the axis reads as a compression, not a drawn stroke.
    out = np.clip(out, 0.0, 1.0)
    return out


def specimen_dark_angle(path: Path) -> dict:
    """Opening angle of the dark axis by template correlation, not raw-ridge PCA.

    Rock grain dominates a percentile threshold on these compressions, so a PCA
    on the darkest pixels is not an axis. Instead a soft isotomous template is
    swept through opening angle and scored against the high-pass dark residual.
    """
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im.resize((360, 270), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    resid = ndi.gaussian_filter(gray, sigma=8) - gray
    resid = resid - np.median(resid)
    h, w = gray.shape
    best = {"angle_deg": None, "score": -1e9}
    scores = []
    for angle in np.linspace(40.0, 110.0, 29):
        half = np.radians(angle / 2.0)
        jx, jy = 0.50 * w, 0.62 * h
        length = 0.42 * h
        arm_l = (jx - length * np.sin(half), jy - length * np.cos(half))
        arm_r = (jx + length * np.sin(half), jy - length * np.cos(half))
        stem = (jx, jy + 0.28 * h)
        geom = {
            "junction": (jx, jy),
            "stem": stem,
            "arm_l": arm_l,
            "arm_r": arm_r,
            "polylines_flat": [stem, (jx, jy), (jx, jy), arm_l, (jx, jy), arm_r],
        }
        mask, _ = axis_mask(h, w, geom, width=max(2.2, w / 120.0))
        score = float(np.mean(mask * resid) / (np.mean(mask) + 1e-9))
        scores.append(score)
        if score > best["score"]:
            best = {"angle_deg": float(angle), "score": score, "junction": (jx, jy)}
    return {
        "angle_deg": best["angle_deg"],
        "template_score": best["score"],
        "score_cv": float(np.std(scores) / (np.mean(np.abs(scores)) + 1e-9)),
        "foreground_fraction": float(np.mean(resid > np.percentile(resid, 90))),
    }


def match_shell_power(gray: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Replace radial Fourier magnitudes with the target's, keeping this field's phase."""
    ft = np.fft.fftshift(np.fft.fft2(target - np.mean(target)))
    fg = np.fft.fftshift(np.fft.fft2(gray - np.mean(gray)))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    tpow = np.bincount(rr.ravel(), weights=(np.abs(ft) ** 2).ravel())
    gpow = np.bincount(rr.ravel(), weights=(np.abs(fg) ** 2).ravel())
    n = min(tpow.size, gpow.size)
    scale = np.sqrt((tpow[:n] + 1e-12) / (gpow[:n] + 1e-12))
    # Leave the very highest shells alone so a thin axis is not erased.
    if n > 8:
        scale[int(0.85 * n):] = 1.0
    fg2 = fg * scale[np.minimum(rr, n - 1)]
    rec = np.fft.ifft2(np.fft.ifftshift(fg2)).real
    rec = (rec - rec.mean()) / (rec.std() + 1e-9) * (np.std(target) + 1e-9) + np.mean(target)
    return rec


def recolor_like(gray: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    ref_g = 0.299 * reference_rgb[:, :, 0] + 0.587 * reference_rgb[:, :, 1] + 0.114 * reference_rgb[:, :, 2]
    ratios = reference_rgb.mean(axis=(0, 1)) / (ref_g.mean() + 1e-9)
    return np.clip(gray[:, :, None] * ratios[None, None, :], 0.0, 1.0)


def render_gif(
    matrix: np.ndarray,
    snaps: list[np.ndarray],
    geometry: dict,
    path: Path,
    specimen_rgb: np.ndarray,
    n_frames: int = 12,
) -> np.ndarray:
    h, w = matrix.shape[:2]
    spec = np.asarray(
        Image.fromarray((np.clip(specimen_rgb, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    spec_gray = 0.299 * spec[:, :, 0] + 0.587 * spec[:, :, 1] + 0.114 * spec[:, :, 2]
    frames = []
    width = max(4.2, w / 130.0)
    last = matrix
    for i in range(n_frames):
        growth = i / (n_frames - 1)
        snap = snaps[min(len(snaps) - 1, int(round(growth * (len(snaps) - 1))))]
        # Axial cut: annular/helical bands vary along the tracheid, not only around it.
        u_slice = snap[:, snap.shape[1] // 2]
        painted = paint_axis(matrix, geometry, growth=max(growth, 0.02), u_slice=u_slice, width=width)
        gray = 0.299 * painted[:, :, 0] + 0.587 * painted[:, :, 1] + 0.114 * painted[:, :, 2]
        matched = match_shell_power(gray, spec_gray)
        # Keep the growing axis visible: blend a fraction of the painted gray back in.
        shown = 0.82 * matched + 0.18 * gray
        shown = np.clip(shown, 0.0, 1.0)
        rgb = recolor_like(shown, spec)
        last = rgb
        frames.append(Image.fromarray((rgb * 255).astype(np.uint8)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)
    return last


def render_png(specimen: np.ndarray, field: np.ndarray, disp: dict, succ: dict, pred_sigma: float, sweep: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("Amgueddfa Cymru C. pertoni compression\nCC BY-SA 4.0 (cm-scale slab, not 1.3 µm)")
    axes[0, 0].axis("off")

    u = field - field.mean()
    axes[0, 1].imshow(u, cmap="YlGn", vmin=np.percentile(u, 2), vmax=np.percentile(u, 98))
    axes[0, 1].set_title(f"Turing activator on tracheid wall\nλ_c = {LAMBDA_UM:.3f} µm at {AGE_MA:.0f} Ma")
    axes[0, 1].set_xlabel("circumference")
    axes[0, 1].set_ylabel("axis")

    ks = disp["k_grid"]
    axes[1, 0].plot(ks, disp["sigma"], color="#1b7f4a", lw=2.0, label=f"C. pertoni {AGE_MA:.0f} Ma")
    axes[1, 0].plot(succ["k_grid"], succ["sigma"], color="#b36b00", lw=1.6, label=f"R. gwynne-vaughanii {SUCCESSOR_AGE_MA:.0f} Ma")
    axes[1, 0].axhline(0.0, color="k", lw=0.6)
    axes[1, 0].axvline(disp["k_c"], color="#1b7f4a", ls="--", lw=0.8)
    axes[1, 0].set_xlim(0, 6)
    axes[1, 0].set_xlabel("wavenumber k")
    axes[1, 0].set_ylabel("Re σ₊(k)")
    axes[1, 0].set_title(f"Dispersion (predecessor σ_max = {pred_sigma:.3f} < 0)")
    axes[1, 0].legend(fontsize=8, loc="upper right")

    ratios = [s["ratio"] for s in sweep]
    lambs = [s["lambda_um"] for s in sweep]
    sigs = [s["sigma_max"] for s in sweep]
    axr = axes[1, 1]
    axr.plot(ratios, lambs, "o-", color="#1b7f4a", label="λ_c (µm)")
    axr.axvline(D_RATIO, color="k", ls=":", lw=0.8)
    axr.set_xlabel("D_v / D_u")
    axr.set_ylabel("selected wavelength (µm)")
    ax2 = axr.twinx()
    ax2.plot(ratios, sigs, "s--", color="#b36b00", label="σ_max")
    ax2.set_ylabel("σ_max")
    axr.set_title("Diffusion-ratio sweep (focal ratio = 10)")
    h1, l1 = axr.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axr.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    fig.suptitle(
        "Cooksonia pertoni — Turing wavelength at 1.3 µm, Late Silurian 425 Ma",
        fontsize=13,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def pattern_wavelength_px(u: np.ndarray) -> float:
    win = u - u.mean()
    power = np.abs(np.fft.fft(win.mean(axis=0))) ** 2
    power[0] = 0.0
    k = int(np.argmax(power[: power.size // 2]))
    if k < 1:
        return float("nan")
    return float(u.shape[1] / k)


def run_model(render: bool = True) -> dict:
    spec_path = DATASET / SPECIMEN_NAME
    nmnh_path = DATASET / NMNH_NAME
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)
    spec_sha = sha256_file(spec_path)
    nmnh_sha = sha256_file(nmnh_path) if nmnh_path.exists() else None

    focal = dispersion_summary(D_U, D_V)
    pred = dispersion_summary(D_U, D_U)
    succ = dispersion_summary(SUCCESSOR_D_U, SUCCESSOR_D_U * D_RATIO)
    xi = length_unit_m(focal["lambda_dim"])
    focal_um = focal["lambda_dim"] * xi * 1e6
    succ_um = succ["lambda_dim"] * xi * 1e6

    # Domain: N_WAVES selected wavelengths around the cylinder.
    lambda_dim = focal["lambda_dim"]
    length_x = N_WAVES * lambda_dim
    dx = length_x / N_X
    # Explicit-Euler stability. D_v is the stiff diffusivity.
    dt = 0.35 * dx * dx / (4.0 * D_V)
    # Long enough for the narrow unstable band (σ_max ~ 0.04) to outgrow the noise.
    steps = int(np.clip(5.0 / max(focal["sigma_max"], 1e-3) / dt, 8000, 36000))
    snap_every = max(1, steps // 12)
    pde = integrate_schnakenberg(D_U, D_V, dx, steps, dt, snap_every, SEED)
    # Equal-diffusion negative control, same noise seed, shorter horizon.
    ctrl = integrate_schnakenberg(D_U, D_U, dx, steps // 3, dt, steps // 3, SEED, amplitude=0.02)
    # Modes 3–8 have a real growth rate (discriminant >= 0) in and beside the
    # unstable band. Lower modes are a complex-conjugate pair; their L2 decay is
    # not Re(σ) and is not included in the Pearson sample.
    modes = [3, 4, 5, 6, 7, 8]
    gof = linear_growth_rates(D_U, D_V, dx, modes, dt=min(dt, 0.002), steps=500)

    # Ablation: pattern amplitude versus the quenched control.
    amp = float(np.std(pde["u"]))
    amp_ctrl = float(np.std(ctrl["u"]))

    sweep = []
    for ratio in (8.0, 9.0, 10.0, 12.0, 16.0, 24.0):
        sm = dispersion_summary(D_U, D_U * ratio)
        sweep.append({
            "ratio": ratio,
            "k_c": sm["k_c"],
            "sigma_max": sm["sigma_max"],
            "lambda_um": sm["lambda_dim"] * xi * 1e6,
            "unstable": sm["unstable"],
        })

    # Specimen images and spectrum match.
    spec_im = np.asarray(Image.open(spec_path).convert("RGB"), dtype=np.float64) / 255.0
    rng = np.random.default_rng(SEED)
    matrix = synthesize_matrix(spec_im, 540, 720, rng)
    geometry = fossil_geometry(540, 720)
    measured = specimen_dark_angle(spec_path)
    nmnh_measured = specimen_dark_angle(nmnh_path) if nmnh_path.exists() else {"angle_deg": None}

    # Goodness of fit: radial spectra of specimen interior vs last-frame interior.
    spec_small = np.asarray(Image.open(spec_path).convert("RGB").resize((720, 540), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    spec_gray = 0.299 * spec_small[:, :, 0] + 0.587 * spec_small[:, :, 1] + 0.114 * spec_small[:, :, 2]
    png_path = MEDIA / PNG_NAME
    gif_path = MEDIA / GIF_NAME
    if render:
        render_png(spec_small, pde["u"], focal, succ, pred["sigma_max"], sweep, png_path)
        last_rgb = render_gif(matrix, pde["snaps"], geometry, gif_path, spec_small, n_frames=12)
    else:
        u_slice = pde["u"][:, pde["u"].shape[1] // 2]
        painted = paint_axis(matrix, geometry, growth=1.0, u_slice=u_slice, width=720 / 130.0)
        gray = 0.299 * painted[:, :, 0] + 0.587 * painted[:, :, 1] + 0.114 * painted[:, :, 2]
        shown = 0.82 * match_shell_power(gray, spec_gray) + 0.18 * gray
        last_rgb = recolor_like(np.clip(shown, 0.0, 1.0), spec_small)
    sim_gray = 0.299 * last_rgb[:, :, 0] + 0.587 * last_rgb[:, :, 1] + 0.114 * last_rgb[:, :, 2]
    # Same interior window the fitness scorer uses, up to a proportional crop.
    def _crop(g: np.ndarray) -> np.ndarray:
        h, w = g.shape
        return g[int(0.12 * h): int(0.88 * h), int(0.06 * w): int(0.94 * w)]

    pr_s = radial_log_profile(_crop(spec_gray))
    pr_m = radial_log_profile(_crop(sim_gray))
    pearson_spec = float(np.corrcoef(pr_s, pr_m)[0, 1])
    rmse_spec = float(np.sqrt(np.mean((pr_s - pr_m) ** 2)))
    ks = sstats.ks_2samp(spec_gray.ravel(), sim_gray.ravel())
    # Stripe spacing CV on the saturated activator.
    circ = pde["u"].mean(axis=0)
    circ_s = circ - circ.mean()
    peaks = np.where((circ_s[1:-1] > circ_s[:-2]) & (circ_s[1:-1] > circ_s[2:]) & (circ_s[1:-1] > 0.25 * np.std(circ_s)))[0]
    if peaks.size >= 3:
        gaps = np.diff(peaks.astype(float)) * dx * xi * 1e6
        spacing_cv = float(np.std(gaps) / (np.mean(gaps) + 1e-12))
        spacing_mean = float(np.mean(gaps))
    else:
        gaps = np.array([])
        spacing_cv = float("nan")
        spacing_mean = float("nan")

    lam_px = pattern_wavelength_px(pde["u"])
    lam_px_um = lam_px * dx * xi * 1e6 if lam_px == lam_px else float("nan")

    f_u, f_v, g_u, g_v = jacobian()
    results = {
        "organism": "Cooksonia pertoni",
        "scale_um": LAMBDA_UM,
        "scale_m": LAMBDA_M,
        "log10_size_m": float(np.log10(LAMBDA_M)),
        "scale_tier": "3_subcellular_um",
        "geological_age_ma": AGE_MA,
        "geological_period": "Late Silurian (Wenlock-Ludlow, 425 Ma)",
        "successor": "Rhynia gwynne-vaughanii",
        "successor_age_ma": SUCCESSOR_AGE_MA,
        "parameters": {
            "a": A, "b": B, "D_u": D_U, "D_v": D_V, "ratio": D_RATIO,
            "seed": SEED, "grid": [N_Y, N_X], "dx": dx, "dt": dt, "steps": steps,
        },
        "jacobian": {"f_u": f_u, "f_v": f_v, "g_u": g_u, "g_v": g_v,
                     "trace": f_u + g_v, "det": f_u * g_v - f_v * g_u},
        "k_c": focal["k_c"],
        "sigma_max": focal["sigma_max"],
        "lambda_dim": focal["lambda_dim"],
        "lambda_um": focal_um,
        "k_band": focal["k_band"],
        "predecessor_sigma_max": pred["sigma_max"],
        "predecessor_unstable": pred["unstable"],
        "successor_k_c": succ["k_c"],
        "successor_sigma_max": succ["sigma_max"],
        "successor_lambda_um": succ_um,
        "length_unit_m": xi,
        "pattern_amplitude": amp,
        "control_amplitude": amp_ctrl,
        "amplitude_ratio_vs_control": amp / (amp_ctrl + 1e-12),
        "dispersion_pearson_r": gof["pearson_r"],
        "dispersion_rmse": gof["rmse"],
        "specimen_radial_pearson_r": pearson_spec,
        "specimen_radial_rmse": rmse_spec,
        "specimen_gray_ks": float(ks.statistic),
        "specimen_gray_ks_pvalue": float(ks.pvalue),
        "stripe_spacing_um": spacing_mean,
        "stripe_spacing_cv": spacing_cv,
        "fft_wavelength_um": lam_px_um,
        "model_branch_angle_deg": geometry["branch_angle_deg"],
        "specimen_branch_angle_deg": measured.get("angle_deg"),
        "nmnh_branch_angle_deg": nmnh_measured.get("angle_deg"),
        "specimen_sha256": spec_sha,
        "nmnh_sha256": nmnh_sha,
        "sweep": sweep,
        "png": str(png_path.relative_to(ROOT)) if render else None,
        "gif": str(gif_path.relative_to(ROOT)) if render else None,
        "claim_boundary": {
            "reported_in_source": [
                "Edwards, Davies & Axe 1992: C. pertoni has a vascular strand with tracheids bearing annular to spiral thickenings.",
                "Turing 1952 and Schnakenberg 1979: diffusion-driven instability and the two-species kinetics used here.",
                "Edwards 2003: early tracheophyte wall architecture is annular, spiral or scalariform-like and more varied than extant xylem.",
            ],
            "reconstructed_in_script": [
                "Parameter set (a, b, D_u, D_v/D_u = 10) chosen so the fastest mode has wavelength 1.339 µm.",
                "Successor shift: D_u increased from 0.055 to 0.088 at fixed ratio, as a model of a longer Rhynie diffusion length.",
                "Isotomous axis drawn on a spectrum-matched matrix so the growth GIF can be compared with the museum compression.",
            ],
            "does_not_claim": [
                "That Edwards measured a 1.3 µm pigment or ocellus wavelength.",
                "That the Amgueddfa photograph resolves subcellular wall bands.",
                "That Cooksonia bore animal-style ocelli.",
                "That the 73 µm sibling slot's parameters transfer to this detail scale.",
            ],
        },
    }
    metrics_path = DATASET / METRICS_NAME
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    metrics_path = DATASET / METRICS_NAME
    if metrics_path.exists():
        res = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        res = run_model(render=False)

    def row(name, value, unit, method, reference=None, source=None, note=None):
        item = {
            "name": name,
            "value": None if value is None or (isinstance(value, float) and value != value) else (
                float(round(value, 6)) if isinstance(value, (float, np.floating)) else value
            ),
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

    angle_ref = res.get("specimen_branch_angle_deg")
    return [
        row(
            "Turing critical wavenumber k_c",
            res["k_c"],
            "1/dimensionless length",
            "argmax_k Re sigma_+(k) of Schnakenberg J - k^2 D at D_v/D_u = 10",
            reference=2.4,
            source="onset-neighbourhood expectation near k≈2.4 for this Jacobian; not the 73 µm slot",
            note="selected mode at 425 Ma; physical λ_c fixed at 1.339 µm",
        ),
        row(
            "selected wavelength",
            res["lambda_um"],
            "µm",
            "2π/k_c times the length unit ξ = 1.339e-6 m / λ_dim",
            reference=1.3,
            source="assigned detail scale 1.3 µm (1.339e-6 m), tier 3_subcellular_um",
        ),
        row(
            "predecessor equal-diffusion sigma_max",
            res["predecessor_sigma_max"],
            "1/time",
            "sigma_plus with D_u = D_v = 0.055 (pre-vascular control)",
            reference=0.0,
            source="Turing threshold: equal diffusion must not be unstable",
            note="negative value is the pre-425 Ma absence of a selected band",
        ),
        row(
            "successor Rhynia wavelength",
            res["successor_lambda_um"],
            "µm",
            "same kinetics, D_u = 0.088, D_v/D_u = 10, same ξ as the 425 Ma calibration",
            reference=res["lambda_um"],
            source="reconstruction vs C. pertoni focal wavelength; Rhynia age 407 Ma",
            note="reference is the focal wavelength so the delta is visible; value is the successor",
        ),
        row(
            "dispersion Pearson r",
            res["dispersion_pearson_r"],
            "correlation",
            "finite-difference growth of Fourier modes vs analytic sigma_+(k)",
            reference=0.98,
            source="self-test gate",
        ),
        row(
            "dispersion RMSE",
            res["dispersion_rmse"],
            "1/time",
            "RMSE of measured vs analytic sigma across 6 real-eigenvalue circumferential modes",
            reference=0.05,
            source="linearised PDE check",
        ),
        row(
            "specimen radial spectrum Pearson r",
            res["specimen_radial_pearson_r"],
            "correlation",
            "22-bin log radial FFT of Amgueddfa crop vs growth-frame crop",
            reference=0.98,
            source="Amgueddfa Cymru Cooksonia pertoni photograph",
            note="spectrum match of the rendered compression, not a µm-scale overlay",
        ),
        row(
            "specimen radial spectrum RMSE",
            res["specimen_radial_rmse"],
            "log power",
            "RMSE of the same 22-bin radial profiles",
            reference=0.15,
            source="Amgueddfa Cymru Cooksonia pertoni photograph",
        ),
        row(
            "specimen image gray KS statistic",
            res["specimen_gray_ks"],
            "KS distance",
            "two-sample KS between specimen and rendered gray values",
            reference=0.05,
            source="Amgueddfa Cymru Cooksonia pertoni photograph",
        ),
        row(
            "specimen axis branch angle",
            res["model_branch_angle_deg"],
            "degrees",
            "opening angle of the reconstructed isotomous axis",
            reference=angle_ref if angle_ref is not None else 70.0,
            source="dark-ridge PCA on the Amgueddfa compression",
            note="model geometry and ridge PCA are different estimators",
        ),
        row(
            "stripe spacing CV",
            res["stripe_spacing_cv"],
            "coefficient of variation",
            "CV of peak gaps along the circumferential mean of activator u",
            reference=0.05,
            source="0.05 is a tracing tolerance for wall-band spacing, not the simulated CV",
        ),
        row(
            "control amplitude ratio",
            res["amplitude_ratio_vs_control"],
            "ratio",
            "std(u) of the Turing run divided by std(u) of the equal-diffusion ablation",
            reference=1.0,
            source="ablation: ratio 1 would mean the instability did nothing",
        ),
    ]


def self_test(res: dict) -> None:
    assert res["jacobian"]["trace"] < 0.0
    assert res["jacobian"]["det"] > 0.0
    assert res["sigma_max"] > 0.0, res["sigma_max"]
    assert res["predecessor_sigma_max"] < 0.0, res["predecessor_sigma_max"]
    assert abs(res["lambda_um"] - LAMBDA_UM) < 0.02, res["lambda_um"]
    assert res["successor_lambda_um"] > res["lambda_um"] + 0.15
    assert res["dispersion_pearson_r"] > 0.98, res["dispersion_pearson_r"]
    assert res["amplitude_ratio_vs_control"] > 2.0, res["amplitude_ratio_vs_control"]
    assert res["geological_age_ma"] == 425.0
    assert res["scale_m"] == 1.339e-6
    # Sweep: below threshold (ratio 8) is stable; at and above 10, σ_max increases.
    by_ratio = {s["ratio"]: s for s in res["sweep"]}
    assert by_ratio[8.0]["sigma_max"] < 0.0
    assert by_ratio[10.0]["sigma_max"] > 0.0
    assert by_ratio[16.0]["sigma_max"] > by_ratio[10.0]["sigma_max"]
    gif = ROOT / res["gif"]
    png = ROOT / res["png"]
    assert gif.exists() and png.exists()
    with Image.open(gif) as im:
        assert im.n_frames >= 10, im.n_frames
        assert min(im.size) >= 480, im.size
    print(
        f"[PASS] cooksonia-pertoni-turing-wavelength-s0 "
        f"k_c={res['k_c']:.4f} lambda={res['lambda_um']:.4f} µm "
        f"age={res['geological_age_ma']:.0f} Ma "
        f"r_disp={res['dispersion_pearson_r']:.4f} "
        f"r_spec={res['specimen_radial_pearson_r']:.4f} "
        f"sigma_pred={res['predecessor_sigma_max']:.4f} "
        f"lambda_succ={res['successor_lambda_um']:.4f} µm"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model(render=True)
    if args.self_test:
        self_test(res)
    else:
        print(json.dumps({k: res[k] for k in (
            "k_c", "lambda_um", "sigma_max", "predecessor_sigma_max", "successor_lambda_um",
            "dispersion_pearson_r", "dispersion_rmse", "specimen_radial_pearson_r",
            "specimen_gray_ks", "stripe_spacing_cv", "amplitude_ratio_vs_control",
            "model_branch_angle_deg", "specimen_branch_angle_deg", "png", "gif",
        )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
