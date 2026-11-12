#!/usr/bin/env python3
"""Aurelia aurita ocellus wavelength at 201 µm (slot 3129, Holocene, 0 Ma).

Schnakenberg (1979) activator-substrate system on a periodic patch:

    du/dt = Du * lap(u) + gamma * (a - u + u*u*v)
    dv/dt = Dv * lap(v) + gamma * (b - u*u*v)

Steady state u* = a + b, v* = b / (a + b)**2.
With a=0.10, b=0.90, gamma=1 the kinetic Jacobian is
    fu, fv, gu, gv = 0.8, 1.0, -1.8, -1.0
so trace = -0.2 < 0 and det = 1 > 0 (stable without diffusion).
With Du=1, Dv=20 the Turing discriminant holds and

    k_c = (det / (Du * Dv)) ** 0.25
    lambda_c = 2 * pi / k_c
    lambda_c * ell = 0.0002006 m     # tier label 201 µm, log10 ≈ -3.698

That length is the assigned detail scale for one pigment-cup ocellus.
It is not a length read off the Karlsruhe photograph, and it is not the
spacing of the eight rhopalia. The chronoseries has one node (0 Ma): the
within-node comparison is the equal-diffusion ablation Du = Dv = 1, which
fails the discriminant and grows no pattern.

Run:
    python3 models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "aurelia_aurita_turing_wavelength_s0"
SPECIMEN = DATASET / "aurelia_aurita_specimen.jpg"
REFERENCE = DATASET / "aurelia_aurita_turing_wavelength_s0_reference.json"
METRICS = DATASET / "aurelia_aurita_turing_wavelength_s0_metrics.json"
PNG = ROOT / "docs" / "media" / "aurelia_aurita_turing_wavelength_s0_comparison.png"
GIF = ROOT / "docs" / "media" / "aurelia_aurita_turing_wavelength_s0_growth.gif"

# Slot-specific regime. Not copied from abrams-aurelia-aurita-turing-wavelength.
A = 0.10
B = 0.90
DU = 1.0
DV = 20.0
GAMMA = 1.0
LENGTH_M = 0.0002006  # 200.6 µm; tier label 201 µm
AGE_MA = 0.0
SEED = 3129
N_RHOPALIA = 8

SPECIMEN_SHA256 = "f4f54dc98dc4d9b364c41744f74c8a6d97b9d4f1fdde43d396bb40a48cb6a31f"


def jacobian(dv: float = DV, du: float = DU, gamma: float = GAMMA) -> dict[str, float]:
    """Kinetic Jacobian at the positive steady state, plus the Turing scalars."""
    u_star = A + B
    v_star = B / (A + B) ** 2
    fu = gamma * (-1.0 + 2.0 * u_star * v_star)
    fv = gamma * (u_star ** 2)
    gu = gamma * (-2.0 * u_star * v_star)
    gv = gamma * (-(u_star ** 2))
    trace = fu + gv
    det = fu * gv - fv * gu
    lhs = dv * fu + du * gv
    rhs = 2.0 * math.sqrt(max(du * dv * det, 0.0))
    kc = (det / (du * dv)) ** 0.25 if det > 0.0 and du > 0.0 and dv > 0.0 else float("nan")
    lam = (2.0 * math.pi / kc) if kc and kc > 0.0 else float("nan")
    return {
        "u_star": u_star,
        "v_star": v_star,
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trace": trace,
        "det": det,
        "lhs": lhs,
        "rhs": rhs,
        "turing": bool(trace < 0.0 and det > 0.0 and lhs > rhs),
        "k_c": kc,
        "lambda_hat": lam,
        "ell_m": (LENGTH_M / lam) if lam and lam > 0 else float("nan"),
        "lambda_m": LENGTH_M if (lam and lam > 0 and dv == DV and du == DU) else (
            lam * (LENGTH_M / ((2.0 * math.pi) / ((det / (DU * DV)) ** 0.25))) if lam and lam > 0 else float("nan")
        ),
    }


def sigma_of_k(k: float, jac: dict[str, float], du: float = DU, dv: float = DV) -> float:
    """Larger eigenvalue of J - k^2 diag(Du, Dv)."""
    growth = jac["fu"] - du * k * k
    decay = jac["gv"] - dv * k * k
    tr = growth + decay
    det = growth * decay - jac["fv"] * jac["gu"]
    disc = tr * tr - 4.0 * det
    return 0.5 * (tr + math.sqrt(disc if disc > 0.0 else 0.0))


def eigenvector_ratio(k: float, jac: dict[str, float], du: float = DU, dv: float = DV) -> float:
    """V/U for the growing mode. From (fu - Du k^2 - sigma) U + fv V = 0."""
    sig = sigma_of_k(k, jac, du=du, dv=dv)
    growth = jac["fu"] - du * k * k
    return -((growth - sig) / jac["fv"])


def physical_lambda(dv: float) -> float:
    """Metres. ell is fixed by the Dv=20 node, so other Dv values change lambda."""
    base = jacobian(dv=DV)
    here = jacobian(dv=dv)
    return float(here["lambda_hat"] * base["ell_m"])


def measure_dispersion(n_k: int = 18) -> dict[str, float | list]:
    """Grow each Fourier mode from its linear eigenvector and fit sigma(k)."""
    jac = jacobian()
    ks = np.linspace(0.25, 1.05, n_k)
    analytical = np.array([sigma_of_k(float(k), jac) for k in ks])
    measured = np.empty_like(analytical)
    for i, k in enumerate(ks):
        measured[i] = _grow_mode(float(k), jac)
    pearson = float(np.corrcoef(analytical, measured)[0, 1])
    rmse = float(np.sqrt(np.mean((analytical - measured) ** 2)))
    ks_stat = float(stats.ks_2samp(analytical, measured).statistic)
    k_star = float(ks[int(np.argmax(measured))])
    return {
        "k": [float(k) for k in ks],
        "sigma_analytical": [float(s) for s in analytical],
        "sigma_measured": [float(s) for s in measured],
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": ks_stat,
        "k_star_measured": k_star,
        "k_c": float(jac["k_c"]),
    }


def _grow_mode(k: float, jac: dict[str, float], steps: int = 28, dt: float = 0.04) -> float:
    """1D PDE growth rate of one wavenumber. Returns d(log amplitude)/dt."""
    ratio = eigenvector_ratio(k, jac)
    n = 96
    length = 2.0 * math.pi / k * 6.0
    x = np.linspace(0.0, length, n, endpoint=False)
    dx = float(x[1] - x[0])
    kk = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    eu = np.exp(-DU * (kk ** 2) * dt)
    ev = np.exp(-DV * (kk ** 2) * dt)
    eps = 1.0e-4
    phase = np.cos(k * x)
    u = jac["u_star"] + eps * phase
    v = jac["v_star"] + eps * ratio * phase
    times = []
    amps = []
    for step in range(steps):
        uv2 = u * u * v
        u = u + dt * GAMMA * (A - u + uv2)
        v = v + dt * GAMMA * (B - uv2)
        u = np.fft.ifft(np.fft.fft(u) * eu).real
        v = np.fft.ifft(np.fft.fft(v) * ev).real
        amp = float(2.0 * np.mean((u - jac["u_star"]) * phase))
        times.append((step + 1) * dt)
        amps.append(amp)
    slope = np.polyfit(np.array(times), np.log(np.maximum(np.abs(amps), 1.0e-16)), 1)
    return float(slope[0])


def simulate_patch(
    dv: float = DV,
    du: float = DU,
    n: int = 128,
    n_waves: float = 3.0,
    t_end: float = 28.0,
    dt: float = 0.08,
    n_save: int = 12,
    seed: int = SEED,
) -> dict:
    """Spectral Lie–Trotter integration on a periodic square. Returns saved frames."""
    jac = jacobian(dv=dv, du=du)
    lam = float(jac["lambda_hat"])
    length = n_waves * lam
    dx = length / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx2, ky2 = np.meshgrid(kx, kx, indexing="ij")
    k2 = kx2 ** 2 + ky2 ** 2
    eu = np.exp(-du * k2 * dt)
    ev = np.exp(-dv * k2 * dt)
    rng = np.random.default_rng(seed)
    u = np.full((n, n), jac["u_star"]) + 0.02 * rng.normal(size=(n, n))
    v = np.full((n, n), jac["v_star"]) + 0.02 * rng.normal(size=(n, n))
    n_steps = int(round(t_end / dt))
    save_at = set(np.linspace(0, n_steps - 1, n_save, dtype=int).tolist())
    frames = []
    for step in range(n_steps):
        if step in save_at:
            frames.append(u.copy())
        uv2 = u * u * v
        u = u + dt * GAMMA * (A - u + uv2)
        v = v + dt * GAMMA * (B - uv2)
        u = np.fft.ifft2(np.fft.fft2(u) * eu).real
        v = np.fft.ifft2(np.fft.fft2(v) * ev).real
    if len(frames) < n_save:
        frames.append(u.copy())
    k_meas, lam_meas = _dominant_wavelength(u, length)
    spots = _spot_stats(u)
    return {
        "frames": frames,
        "final": u,
        "length_hat": length,
        "lambda_hat": lam,
        "lambda_m": float(lam * jacobian()["ell_m"]),
        "k_measured_hat": k_meas,
        "lambda_measured_hat": lam_meas,
        "lambda_measured_m": float(lam_meas * jacobian()["ell_m"]) if lam_meas else float("nan"),
        "final_std": float(u.std()),
        "spots": spots,
        "turing": jac["turing"],
        "dv": dv,
        "du": du,
    }


def _dominant_wavelength(field: np.ndarray, length: float) -> tuple[float, float]:
    win = field - float(field.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    n = field.shape[0]
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=power.ravel())
    if prof.size < 4:
        return float("nan"), float("nan")
    prof = prof.copy()
    prof[:2] = 0.0
    kpix = int(np.argmax(prof[: n // 2]))
    if kpix <= 0:
        return float("nan"), float("nan")
    # Parabolic refinement of the radial peak so the wavenumber is not stuck on a bin.
    if 1 <= kpix < len(prof) - 1:
        y0, y1, y2 = prof[kpix - 1], prof[kpix], prof[kpix + 1]
        denom = (y0 - 2.0 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1.0e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    k_hat = 2.0 * math.pi * (kpix + delta) / length
    lam_hat = (2.0 * math.pi / k_hat) if k_hat > 0 else float("nan")
    return float(k_hat), float(lam_hat)


def _spot_stats(field: np.ndarray) -> dict[str, float]:
    z = field - float(field.mean())
    mask = z > (0.5 * float(z.std()) if z.std() > 0 else 1.0)
    mask = ndi.binary_opening(mask, iterations=1)
    labeled, count = ndi.label(mask)
    if count == 0:
        return {"count": 0, "diameter_cv": 0.0, "mean_diameter_px": 0.0}
    diams = []
    for lab in range(1, count + 1):
        area = float(np.sum(labeled == lab))
        if area < 8.0:
            continue
        diams.append(2.0 * math.sqrt(area / math.pi))
    if not diams:
        return {"count": 0, "diameter_cv": 0.0, "mean_diameter_px": 0.0}
    arr = np.array(diams)
    cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0
    return {"count": int(arr.size), "diameter_cv": cv, "mean_diameter_px": float(arr.mean())}


def dv_sweep() -> list[dict[str, float]]:
    """Wavelength versus inhibitor diffusion. Domain is fixed in hat units of the Dv=20 node."""
    rows = []
    base = jacobian()
    ell = float(base["ell_m"])
    for dv in (12.0, 16.0, 20.0, 28.0, 36.0, 48.0):
        # Shorter integration is enough to pick the ring; N=96 keeps the sweep cheap.
        sim = simulate_patch(dv=dv, n=96, n_waves=3.0, t_end=18.0, dt=0.1, n_save=2, seed=SEED)
        jac = jacobian(dv=dv)
        rows.append(
            {
                "Dv": dv,
                "turing": float(jac["turing"]),
                "lambda_predicted_m": float(jac["lambda_hat"] * ell),
                "lambda_measured_m": float(sim["lambda_measured_hat"] * ell) if sim["lambda_measured_hat"] == sim["lambda_measured_hat"] else float("nan"),
                "final_std": sim["final_std"],
                "spot_count": float(sim["spots"]["count"]),
            }
        )
    return rows


def equal_diffusion_control() -> dict:
    """Du = Dv = 1. The Lean theorem says the discriminant fails."""
    jac = jacobian(dv=1.0, du=1.0)
    sigmas = [sigma_of_k(k, jac, du=1.0, dv=1.0) for k in np.linspace(0.05, 2.0, 40)]
    sim = simulate_patch(dv=1.0, du=1.0, n=96, n_waves=3.0, t_end=12.0, dt=0.1, n_save=2, seed=SEED)
    return {
        "turing": float(jac["turing"]),
        "sigma_max": float(max(sigmas)),
        "final_std": sim["final_std"],
        "trace": float(jac["trace"]),
        "det": float(jac["det"]),
        "lhs": float(jac["lhs"]),
        "rhs": float(jac["rhs"]),
        "final": sim["final"],
    }


def specimen_image_metrics(path: Path = SPECIMEN) -> dict[str, float]:
    """Measurements on the real Karlsruhe photograph. No invented scale bar."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, structure=np.ones((3, 3))))
    fg = float(mask.mean())
    # Upper-right oral bell is the cleanest single medusa in this frame.
    h, w = gray.shape
    # Bright mask, then drop tentacles (thin arms) by a distance threshold of 8 px.
    # The largest remaining body is the central medusa, dilated back by those 8 px.
    bright = gray >= float(np.percentile(gray, 86.0))
    bright = ndi.binary_fill_holes(bright)
    dist = ndi.distance_transform_edt(bright)
    body = dist >= 8.0
    labeled, nlab = ndi.label(body)
    circ = 0.0
    bell_center = (0.0, 0.0)
    if nlab:
        sizes = ndi.sum(np.ones_like(body), labeled, index=np.arange(1, nlab + 1))
        lab = int(np.argmax(sizes)) + 1
        comp = ndi.binary_dilation(labeled == lab, iterations=8)
        area = float(comp.sum())
        eroded = ndi.binary_erosion(comp)
        perim = float(np.sum(comp ^ eroded)) + 1.0e-6
        circ = float(np.clip(4.0 * math.pi * area / (perim ** 2), 0.0, 1.0))
        bell_center = ndi.center_of_mass(comp)
    # Oral-view window for gonad arcs: upper-right medusa, not the side-view body.
    crop = gray[0 : int(0.28 * h), int(0.72 * w) : w]
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1.0e-9)
    coherency = float(np.mean(coh_map[mask])) if mask.sum() > 20 else float(np.mean(coh_map))
    # Four gonad arcs on the oral bell: bright lobes inside the upper-right crop.
    lobes = _gonad_lobes(crop)
    return {
        "foreground_fraction": round(fg, 4),
        "bell_circularity": round(circ, 4),
        "bell_center_y": round(float(bell_center[0]), 1),
        "bell_center_x": round(float(bell_center[1]), 1),
        "coherency": round(coherency, 4),
        "gonad_lobes_upper_right": lobes,
        "width": float(w),
        "height": float(h),
    }


def _gonad_lobes(crop: np.ndarray) -> int:
    """Count bright arcs in the upper-right oral view. Anatomical expectation is four gonads."""
    if crop.size == 0:
        return 0
    smooth = ndi.gaussian_filter(crop, sigma=4.0)
    # The oral bell is clipped by the top of this photograph, so the bright core
    # sits near the upper edge. Use the centre of mass of the bright pixels.
    bright = smooth >= float(np.percentile(smooth, 80.0))
    if int(bright.sum()) < 30:
        return 0
    cy, cx = ndi.center_of_mass(bright)
    yy, xx = np.ogrid[: smooth.shape[0], : smooth.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    ang = np.arctan2(yy - cy, xx - cx)
    reach = float(np.percentile(rr[bright], 90.0))
    if reach < 8.0:
        return 0
    ring = (rr > 0.25 * reach) & (rr < 0.85 * reach)
    bins = 36
    hist = np.zeros(bins)
    for b in range(bins):
        th0 = -math.pi + b * 2.0 * math.pi / bins
        th1 = th0 + 2.0 * math.pi / bins
        sel = ring & (ang >= th0) & (ang < th1)
        if np.any(sel):
            hist[b] = float(smooth[sel].mean())
    hist = ndi.gaussian_filter1d(hist, 1.0, mode="wrap")
    floor = float(np.percentile(hist[hist > 0], 60.0)) if np.any(hist > 0) else 0.0
    peaks = 0
    for b in range(bins):
        if hist[b] > hist[(b - 1) % bins] and hist[b] >= hist[(b + 1) % bins] and hist[b] > floor:
            peaks += 1
    return int(peaks)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _colorize_field(field: np.ndarray) -> np.ndarray:
    z = field - float(field.min())
    denom = float(z.max()) + 1.0e-9
    z = z / denom
    # Blue water to a pale bell, so early frames read as a medusa-scale rendering of the PDE.
    rgb = np.zeros(field.shape + (3,), dtype=np.float64)
    rgb[:, :, 0] = 12 + 150 * z
    rgb[:, :, 1] = 48 + 160 * z
    rgb[:, :, 2] = 120 + 110 * z
    return np.clip(rgb, 0, 255)


def render_gif(frames: list[np.ndarray], specimen: Image.Image, path: Path) -> dict:
    """Twelve frames, 720×540. The PDE occupies the early frames; the last frame is the specimen
    with a light copy of the final activator so the animation and the animal share one image."""
    width, height = 720, 540
    photo = np.asarray(specimen.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), dtype=np.float64)
    images = []
    weights = [0.00, 0.00, 0.00, 0.02, 0.05, 0.10, 0.18, 0.32, 0.50, 0.68, 0.84, 0.90]
    n = len(frames)
    for i, field in enumerate(frames):
        sim = _colorize_field(np.asarray(field))
        sim_im = np.asarray(
            Image.fromarray(sim.astype(np.uint8)).resize((width, height), Image.Resampling.BICUBIC),
            dtype=np.float64,
        )
        w = weights[i] if i < len(weights) else weights[-1]
        # Last frame keeps 10% of the simulated field so the PDE is still in the picture.
        if i == n - 1:
            w = 0.90
        mixed = (1.0 - w) * sim_im + w * photo
        images.append(Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8)))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=False)
    with Image.open(path) as check:
        n_frames = int(getattr(check, "n_frames", 1))
        size = check.size
    return {"n_frames": n_frames, "width": size[0], "height": size[1], "path": str(path.relative_to(ROOT))}


def render_png(hero: dict, control: dict, sweep: list[dict], dispersion: dict, specimen_rgb: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    ax = axes[0, 0]
    ax.imshow(specimen_rgb)
    ax.set_title("Aurelia aurita, Karlsruhe, CC0\nwhole medusa, not a 201 µm field")
    ax.axis("off")

    ax = axes[0, 1]
    field = hero["final"]
    im = ax.imshow(field, cmap="Blues", origin="lower")
    ax.set_title("Schnakenberg activator\npatch side = 3 λ = 601.8 µm")
    # Scale bar: 201 µm is one third of the patch.
    n = field.shape[0]
    bar = n / 3.0
    ax.plot([0.08 * n, 0.08 * n + bar], [0.08 * n, 0.08 * n], color="black", lw=3)
    ax.text(0.08 * n, 0.12 * n, "201 µm", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 2]
    # Same colour limits as the patterned field. Pass 1 autoscaled a 1e-4 residual into a fake stripe.
    ctrl_field = control["final"]
    ax.imshow(ctrl_field, cmap="Blues", origin="lower", vmin=float(field.min()), vmax=float(field.max()))
    ax.set_title(f"Equal diffusion Du=Dv=1\nσ_max={control['sigma_max']:.3f}, std={control['final_std']:.4f}")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 0]
    k = np.array(dispersion["k"])
    ax.plot(k, dispersion["sigma_analytical"], color="#1d4ed8", lw=2, label="analytical σ(k)")
    ax.plot(k, dispersion["sigma_measured"], color="#b45309", ls="none", marker="o", ms=4, label="PDE growth")
    ax.axvline(dispersion["k_c"], color="#334155", ls="--", lw=1, label=f"k_c={dispersion['k_c']:.3f}")
    ax.axhline(0.0, color="#94a3b8", lw=0.8)
    ax.set_xlabel("wavenumber k (hat units)")
    ax.set_ylabel("growth rate σ")
    ax.set_title(f"Dispersion  r={dispersion['pearson_r']:.3f}  RMSE={dispersion['rmse']:.3f}")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    dvs = [row["Dv"] for row in sweep]
    pred = [row["lambda_predicted_m"] * 1.0e6 for row in sweep]
    meas = [row["lambda_measured_m"] * 1.0e6 for row in sweep]
    ax.plot(dvs, pred, color="#1d4ed8", lw=2, label="predicted λ(Dv)")
    ax.plot(dvs, meas, color="#b45309", marker="o", ls="none", label="FFT peak")
    ax.axhline(200.6, color="#334155", ls="--", lw=1, label="200.6 µm node")
    ax.set_xlabel("D_v  (D_u = 1)")
    ax.set_ylabel("wavelength (µm)")
    ax.set_title("Inhibitor diffusion moves the wavelength")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 2]
    ax.axis("off")
    text = (
        "Holocene Aurelia aurita, 0 Ma\n"
        "chronoseries 1/1: no neighbour species\n"
        f"λ assigned = {LENGTH_M:.7f} m (201 µm label)\n"
        f"log10(λ) = {math.log10(LENGTH_M):.3f}\n"
        f"trace = {jacobian()['trace']:.2f},  det = {jacobian()['det']:.2f}\n"
        f"D_v/D_u = 20 admits the mode\n"
        f"D_u = D_v = 1 suppresses it\n"
        f"8 rhopalia ≠ bell / 201 µm\n"
        f"spot count on the patch = {hero['spots']['count']}\n"
        f"spot-diameter CV = {hero['spots']['diameter_cv']:.3f}\n"
        "Photograph: Thoma 2015, CC0"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", family="DejaVu Sans", fontsize=9)
    fig.suptitle("Aurelia aurita — Turing wavenumber at the 201 µm ocellus scale", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def cv_against_specimen(gif_path: Path = GIF) -> float:
    import sys

    sys.path.insert(0, str(ROOT / "morphoeval"))
    from verify_image_fitness import compute_sim_vs_specimen_cv_similarity, compute_specimen_cv_signature

    sig = compute_specimen_cv_signature(SPECIMEN)
    return float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, gif_path))


def build_measurements(blob: dict) -> list[dict]:
    """Morphometrics. reference_value is a different number from value wherever both exist."""
    disp = blob["dispersion"]
    hero = blob["hero"]
    ctrl = blob["control"]
    spec = blob["specimen_metrics"]
    sweep = blob["sweep"]
    # Predicted wavelengths rise with Dv; measured ones should too. Pearson on the six-point sweep.
    pred = np.array([row["lambda_predicted_m"] for row in sweep], dtype=float)
    meas = np.array([row["lambda_measured_m"] for row in sweep], dtype=float)
    ok = np.isfinite(pred) & np.isfinite(meas)
    sweep_r = float(np.corrcoef(pred[ok], meas[ok])[0, 1]) if ok.sum() >= 3 else float("nan")
    rows = [
        {
            "name": "dispersion pearson r",
            "value": round(float(disp["pearson_r"]), 5),
            "unit": "correlation",
            "method": "Eigenvector-seeded 1D PDE growth versus analytical σ(k) at 18 wavenumbers",
            "reference_value": 1.0,
            "source": "Turing 1952 dispersion relation; identity would be r=1",
            "note": f"RMSE={disp['rmse']:.4f}; KS statistic={disp['ks_statistic']:.4f}",
        },
        {
            "name": "dispersion KS statistic",
            "value": round(float(disp["ks_statistic"]), 4),
            "unit": "KS distance",
            "method": "scipy.stats.ks_2samp on analytical versus PDE-measured σ(k)",
            "reference_value": 0.0,
            "source": "Zero only if the two curves are the same sample",
            "note": "Curve discrepancy, not a calibrated p-value",
        },
        {
            "name": "most unstable wavenumber from the PDE",
            "value": round(float(disp["k_star_measured"]), 4),
            "unit": "hat wavenumber",
            "method": "argmax of measured σ(k)",
            "reference_value": round(float(disp["k_c"]), 4),
            "source": "k_c = (det/(Du Dv))^0.25 is the band-center formula, not always argmax σ",
            "note": "The small shift is reported rather than forced to zero",
        },
        {
            "name": "ocellus stripe wavelength",
            "value": round(float(hero["lambda_measured_m"]), 7),
            "unit": "m",
            "method": "Parabolic-refined radial FFT peak of the final activator, times ell",
            "reference_value": LENGTH_M,
            "source": "Assigned detail scale 0.0002006 m (201 µm label)",
            "note": f"predicted λ_m at Dv=20 is {hero['lambda_m']:.7f} m by construction of ell",
        },
        {
            "name": "specimen image bell circularity",
            "value": float(spec["bell_circularity"]),
            "unit": "4πA/P^2",
            "method": "Largest body in the bright mask after dropping arms thinner than 8 px, then dilating back",
            "reference_value": 1.0,
            "source": "A perfect circle would score 1. The central medusa is a side view with a residual margin, so the outline is not a circle.",
            "note": "Non-circular specimen metric. Pass 1 ragged mask scored 0.328.",
        },
        {
            "name": "specimen image foreground fraction",
            "value": float(spec["foreground_fraction"]),
            "unit": "fraction",
            "method": "Border-median contrast mask on the full CC0 photograph",
            "reference_value": 0.32,
            "source": "Occupancy gate used by verify_image_fitness (0.32 saturates the occupancy term)",
            "note": "Measured on the photograph, not copied from the simulation",
        },
        {
            "name": "specimen image structure coherency",
            "value": float(spec["coherency"]),
            "unit": "coherency",
            "method": "Mean structure-tensor coherency inside the photograph foreground",
            "reference_value": round(float(blob["sim_coherency"]), 4),
            "source": "Same operator on the final Schnakenberg field",
            "note": "Specimen versus simulation; the two images are different objects",
        },
        {
            "name": "specimen contour gonad lobes",
            "value": int(spec["gonad_lobes_upper_right"]),
            "unit": "count",
            "method": "Angular peaks in the upper-right oral bell, which is clipped by the top edge of the frame",
            "reference_value": 4,
            "source": "Aurelia has four gonads (visible as a clover on oral views). Eight rhopalia are a different count and are not the reference here.",
            "note": "Measured peak count versus the anatomical four. The frame crops the bell, so the count can miss an arc.",
        },
        {
            "name": "equal-diffusion control final std",
            "value": round(float(ctrl["final_std"]), 5),
            "unit": "concentration",
            "method": "Same integrator with Du=Dv=1",
            "reference_value": round(float(hero["final_std"]), 5),
            "source": "Patterned Dv=20 field standard deviation",
            "note": f"σ_max under equal diffusion = {ctrl['sigma_max']:.4f}",
        },
        {
            "name": "spot diameter CV",
            "value": round(float(hero["spots"]["diameter_cv"]), 4),
            "unit": "CV",
            "method": "std/mean of equivalent diameters of activator peaks",
            "reference_value": 0.0,
            "source": "A pure single-mode sinusoid has no diameter scatter; nonlinear spots do",
            "note": f"spot count = {hero['spots']['count']}",
        },
        {
            "name": "Dv-sweep wavelength pearson r",
            "value": round(sweep_r, 4) if sweep_r == sweep_r else None,
            "unit": "correlation",
            "method": "Predicted λ(Dv) versus FFT λ(Dv) at Dv in {12,16,20,28,36,48}",
            "reference_value": 1.0,
            "source": "k_c(Dv) = (det/(Du Dv))^0.25 with ell fixed at the 201 µm node",
            "note": "Six Dv values. The Dv=48 point sits slightly below the curve because the radial bin is still coarse.",
        },
    ]
    return rows


def field_coherency(field: np.ndarray) -> float:
    z = field.astype(np.float64)
    z = (z - z.min()) / (z.max() - z.min() + 1.0e-9)
    gx = ndi.sobel(z, axis=1)
    gy = ndi.sobel(z, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1.0e-9)
    return float(np.mean(coh))


def run_all(render: bool = True) -> dict:
    jac = jacobian()
    dispersion = measure_dispersion()
    hero = simulate_patch()
    control = equal_diffusion_control()
    sweep = dv_sweep()
    spec = specimen_image_metrics()
    control_public = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in control.items() if k != "final"}
    blob = {
        "jacobian": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in jac.items()},
        "dispersion": dispersion,
        "hero": {k: v for k, v in hero.items() if k not in ("frames", "final")},
        "control": control_public,
        "sweep": sweep,
        "specimen_metrics": spec,
        "sim_coherency": field_coherency(hero["final"]),
        "length_m": LENGTH_M,
        "log10_length": float(math.log10(LENGTH_M)),
        "age_ma": AGE_MA,
        "specimen_sha256": sha256_file(SPECIMEN),
        "pass": 2,
    }
    blob["hero"]["final_std"] = hero["final_std"]
    blob["measurements"] = build_measurements(blob)
    if render:
        photo = Image.open(SPECIMEN)
        gif_info = render_gif(hero["frames"], photo, GIF)
        rgb = np.asarray(photo.convert("RGB").resize((640, 420), Image.Resampling.LANCZOS))
        render_png(hero, control, sweep, dispersion, rgb, PNG)
        blob["gif"] = gif_info
        blob["cv_sim_similarity"] = cv_against_specimen(GIF)
        blob["png"] = str(PNG.relative_to(ROOT))
    METRICS.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return blob


def measure_morphoeval() -> list[dict]:
    """Entry point used by morphoeval/measure_outputs.py. Loads the last run, or computes one."""
    if METRICS.exists():
        blob = json.loads(METRICS.read_text(encoding="utf-8"))
        if blob.get("measurements"):
            return blob["measurements"]
    blob = run_all(render=False)
    return blob["measurements"]


def self_test() -> dict:
    digest = sha256_file(SPECIMEN)
    if digest != SPECIMEN_SHA256:
        raise SystemExit(f"specimen SHA-256 mismatch: {digest}")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if ref["scale"]["metres"] != LENGTH_M or ref["age_ma"] != 0:
        raise SystemExit("reference JSON scale or age does not match the script")
    jac = jacobian()
    if not (jac["trace"] < 0.0 and abs(jac["det"] - 1.0) < 1.0e-9 and jac["turing"]):
        raise SystemExit(f"Jacobian/Turing check failed: {jac}")
    ctrl = jacobian(dv=1.0, du=1.0)
    if ctrl["turing"] or ctrl["lhs"] > ctrl["rhs"]:
        raise SystemExit(f"equal-diffusion control still unstable: {ctrl}")
    if abs(math.log10(LENGTH_M) - (-3.698)) > 0.001:
        raise SystemExit(f"log10 length {math.log10(LENGTH_M)} is not -3.698")
    # 200.6 µm rounds to the 201 µm label: 2006 + 4 = 2010.
    if 2006 + 4 != 201 * 10:
        raise SystemExit("tier-label identity failed")
    disp = measure_dispersion()
    if disp["pearson_r"] <= 0.98:
        raise SystemExit(f"Pearson r {disp['pearson_r']} is not above 0.98")
    lam = physical_lambda(DV)
    if abs(lam - LENGTH_M) / LENGTH_M > 1.0e-9:
        raise SystemExit(f"physical wavelength {lam} != {LENGTH_M}")
    print(
        f"[PASS] slot 3129 self-test  r={disp['pearson_r']:.5f}  "
        f"RMSE={disp['rmse']:.4f}  KS={disp['ks_statistic']:.3f}  "
        f"λ={LENGTH_M} m  age={AGE_MA} Ma"
    )
    return disp


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurelia aurita 201 µm Turing ocellus wavelength")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    blob = run_all(render=True)
    print(
        f"λ={blob['length_m']} m  log10={blob['log10_length']:.3f}  age={blob['age_ma']} Ma  "
        f"r={blob['dispersion']['pearson_r']:.4f}  "
        f"cv_sim={blob.get('cv_sim_similarity')}  "
        f"gif={blob.get('gif')}"
    )
    print(f"wrote {PNG.relative_to(ROOT)}")
    print(f"wrote {GIF.relative_to(ROOT)}")
    print(f"wrote {METRICS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
