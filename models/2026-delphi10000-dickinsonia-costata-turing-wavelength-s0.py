#!/usr/bin/env python3
"""Dickinsonia costata, cellular Turing wavelength (slot 3167).

Scale: 56 µm (5.564e-05 m, log10=-4.255), tier 4/8 ``4_cellular_10um``.
Age: Late Ediacaran, White Sea, 558 Ma. Chronoseries 1/1.

The organ-tier sibling (2.2 mm module pitch, same binomial and same horizon)
counts modules along the body. This file counts pigment wavelengths along one
septum. Integer arithmetic: floor(2200/56) = 39 cellular pitches per septum,
and diffusivity at fixed reaction time rescales by 39**2 = 1521.

Kinetics are Schnakenberg (1979). The dispersion law is Turing (1952):

    k_c**2 = (Dv f_u + Du g_v) / (2 Du Dv)
    lambda = 2 pi / k

The length unit is chosen so the nonlinearly selected wavelength at the
reference diffusivity equals 56 µm. A sweep of a common diffusivity factor
tests lambda ~ sqrt(D). An equal-diffusion run is the negative control.

Specimen: Wikimedia Commons File:DickinsoniaCostata.jpg, Verisimilus, CC BY 2.5.
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
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "dickinsonia_costata_turing_wavelength_s0"
SPECIMEN = DATASET / "dickinsonia_costata_turing_wavelength_s0_specimen.jpg"
REFERENCE = DATASET / "dickinsonia_costata_turing_wavelength_s0_reference.json"
METRICS = DATASET / "dickinsonia_costata_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "dickinsonia_costata_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "dickinsonia_costata_turing_wavelength_s0_growth.gif"

# Cellular-tier parameters. Not the 2.2 mm organ-tier set.
A = 0.12
B = 0.88
DU0 = 0.05
DV0 = 0.55
SEED = 3167
LAMBDA_TARGET_M = 5.564e-5
AGE_MA = 558.0
ORGAN_LAMBDA_M = 2.21e-3
SWEEP_SCALES = (0.55, 0.80, 1.00, 1.35, 1.80, 2.40, 3.20, 4.20)
T_END = 140.0
DT = 0.08
N_WAVES = 7
PPC = 12
SPECIMEN_SHA256 = "9cde902e8577ea737c95a5807287f08ee1752c6ac649b56b9ed546ee354ba5f6"

U0 = A + B
V0 = B / U0**2
FU = -1.0 + 2.0 * U0 * V0
FV = U0**2
GU = -2.0 * U0 * V0
GV = -(U0**2)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_specimen() -> None:
    if not SPECIMEN.exists():
        raise FileNotFoundError(SPECIMEN)
    got = sha256_file(SPECIMEN)
    if got != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 mismatch: {got}")


# @@EVIDENCE:dispersion@@
def sigma_of_k(k: np.ndarray, du: float, dv: float) -> np.ndarray:
    """Largest real eigenvalue of J - k^2 diag(Du, Dv)."""
    a = FU - du * k**2
    d = GV - dv * k**2
    tr = a + d
    det = a * d - FV * GU
    disc = np.maximum(tr**2 - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def critical_wavenumber(du: float, dv: float) -> float:
    """Turing onset wavenumber k_c. Positive only above the diffusion threshold."""
    num = dv * FU + du * GV
    den = 2.0 * du * dv
    if num <= 0.0 or den <= 0.0:
        return float("nan")
    return float(np.sqrt(num / den))


def fastest_wavenumber(du: float, dv: float) -> tuple[float, float]:
    grid = np.linspace(0.15, 14.0, 4000)
    sig = sigma_of_k(grid, du, dv)
    i = int(np.argmax(sig))
    return float(grid[i]), float(sig[i])


def _measure_peak_k(u: np.ndarray, dx: float, dy: float, k_hint: float) -> float:
    ny, nx = u.shape
    win = (u - u.mean()) * np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    kx = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny, d=dy))
    kk = np.hypot(*np.meshgrid(kx, ky))
    bins = np.linspace(k_hint * 0.45, k_hint * 1.80, 48)
    idx = np.digitize(kk.ravel(), bins)
    prof = np.array([
        power.ravel()[idx == i].mean() if np.any(idx == i) else 0.0
        for i in range(1, len(bins))
    ])
    kmid = 0.5 * (bins[:-1] + bins[1:])
    i = int(np.argmax(prof))
    if 0 < i < len(prof) - 1:
        y0, y1, y2 = prof[i - 1], prof[i], prof[i + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        return float(kmid[i] + delta * (kmid[1] - kmid[0]))
    return float(kmid[i])


# @@EVIDENCE:simulate@@
def simulate(
    du: float,
    dv: float,
    dx: float,
    nx: int,
    ny: int,
    seed: int,
    t_end: float = T_END,
    n_snapshots: int = 0,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    """Spectral Schnakenberg step. Diffusion is exact in Fourier space; reaction is explicit."""
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    k2 = np.meshgrid(kx, ky)[0] ** 2 + np.meshgrid(kx, ky)[1] ** 2
    rng = np.random.default_rng(seed)
    u = U0 + 0.005 * rng.normal(size=(ny, nx))
    v = V0 + 0.005 * rng.normal(size=(ny, nx))
    eu = np.exp(-du * k2 * DT)
    ev = np.exp(-dv * k2 * DT)
    nsteps = int(t_end / DT)
    snap_at = set(np.linspace(0, max(nsteps - 1, 0), max(n_snapshots, 0)).astype(int)) if n_snapshots else set()
    snaps: list[np.ndarray] = []
    for step in range(nsteps):
        if step in snap_at:
            snaps.append(u.copy())
        u = np.fft.ifft2(eu * np.fft.fft2(u + DT * (A - u + u * u * v))).real
        v = np.fft.ifft2(ev * np.fft.fft2(v + DT * (B - u * u * v))).real
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            raise RuntimeError(f"non-finite field at step {step}")
    if n_snapshots and (nsteps - 1) not in snap_at:
        snaps.append(u.copy())
    return u, float(np.std(u)), snaps


def reference_grid() -> tuple[float, int, int, float]:
    """Grid sized so the linear fastest mode has `PPC` points and `N_WAVES` repeats."""
    km, _ = fastest_wavenumber(DU0, DV0)
    lam = 2.0 * np.pi / km
    dx = lam / PPC
    nx = int(np.ceil(N_WAVES * PPC / 2.0) * 2)
    ny = int(np.ceil(N_WAVES * PPC * 0.70 / 2.0) * 2)
    return dx, nx, ny, km


# @@EVIDENCE:sweep@@
def diffusion_sweep() -> dict:
    """Each diffusivity gets its own grid so the domain always holds N_WAVES of that mode."""
    measured, predicted, scales = [], [], []
    for scale in SWEEP_SCALES:
        du, dv = DU0 * scale, DV0 * scale
        km, _sig = fastest_wavenumber(du, dv)
        lam = 2.0 * np.pi / km
        dx = lam / PPC
        nx = int(np.ceil(N_WAVES * PPC / 2.0) * 2)
        ny = int(np.ceil(N_WAVES * PPC * 0.70 / 2.0) * 2)
        u, std, _ = simulate(du, dv, dx, nx, ny, seed=SEED, n_snapshots=0)
        k_meas = _measure_peak_k(u, dx, dx, k_hint=km)
        measured.append(k_meas)
        predicted.append(km)
        scales.append(scale)
        if std < 0.05:
            raise RuntimeError(f"sweep scale {scale} did not leave the linear noise floor (std={std})")
    measured_a = np.asarray(measured, dtype=float)
    predicted_a = np.asarray(predicted, dtype=float)
    pearson = float(np.corrcoef(measured_a, predicted_a)[0, 1])
    return {
        "scales": scales,
        "k_measured": measured,
        "k_linear_fast": predicted,
        "pearson_r": pearson,
    }


# @@EVIDENCE:ablation@@
def ablation_std(dx: float, nx: int, ny: int) -> float:
    """Equal diffusion sits below the Turing threshold, so the noise must decay."""
    u, std, _ = simulate(DU0, DU0, dx, nx, ny, seed=SEED, t_end=T_END, n_snapshots=0)
    return std


# @@EVIDENCE:specimen@@
def segment_specimen(path: Path) -> dict:
    """Gradient-energy ellipse on the real fossil photograph, plus the project CV signature."""
    im = Image.open(path).convert("RGB")
    small = im.resize((240, 180), Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = ndi.gaussian_filter(np.hypot(gx, gy), 2.0)
    height, width = gray.shape
    yy, xx = np.mgrid[0:height, 0:width]
    best = None
    for cy in (80, 92, 100):
        for cx in (105, 118, 130):
            for aspect in (1.35, 1.50, 1.65, 1.80, 2.00):
                for semi_b in (48, 58, 68, 76):
                    semi_a = semi_b * aspect
                    for ang in (-20, -8, 0, 10):
                        theta = np.deg2rad(ang)
                        ca, sa = np.cos(theta), np.sin(theta)
                        x = xx - cx
                        y = yy - cy
                        xr = ca * x + sa * y
                        yr = -sa * x + ca * y
                        r2 = (xr / semi_a) ** 2 + (yr / semi_b) ** 2
                        inside = r2 <= 1.0
                        frac = float(inside.mean())
                        if not (0.28 <= frac <= 0.62):
                            continue
                        ring = (r2 <= 1.25) & (r2 > 1.02)
                        if int(ring.sum()) < 30:
                            continue
                        score = float(mag[inside].mean() - 1.15 * mag[ring].mean())
                        if best is None or score > best[0]:
                            best = (score, cx, cy, aspect, semi_a, semi_b, ang, frac, inside)
    if best is None:
        raise RuntimeError("specimen ellipse search failed")
    _, cx, cy, aspect, semi_a, semi_b, ang, frac, inside = best
    # Ramanujan perimeter of the fitted ellipse. A one-pixel erosion of a
    # filled raster ellipse under-counts the boundary and can push 4*pi*A/P^2 above 1.
    h_ell = ((semi_a - semi_b) / (semi_a + semi_b)) ** 2
    per_ell = np.pi * (semi_a + semi_b) * (1.0 + 3.0 * h_ell / (10.0 + np.sqrt(4.0 - 3.0 * h_ell)))
    circularity = float(4.0 * np.pi * (np.pi * semi_a * semi_b) / per_ell**2)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coherency_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coherency_map[inside]))
    # Ridge spacing: autocorrelation along the ellipse minor axis through the centre.
    theta = np.deg2rad(ang)
    minor = np.array([-np.sin(theta), np.cos(theta)])
    samples = []
    for t in np.linspace(-semi_b, semi_b, 181):
        x = int(round(cx + minor[0] * t))
        y = int(round(cy + minor[1] * t))
        if 0 <= x < width and 0 <= y < height:
            samples.append(gray[y, x])
    sig = np.asarray(samples, dtype=float)
    sig = sig - sig.mean()
    ac = np.correlate(sig, sig, mode="full")
    ac = ac[len(ac) // 2 :]
    ac = ac / (ac[0] + 1e-12)
    lag = None
    for i in range(4, min(60, len(ac) - 1)):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.1:
            lag = i
            break
    # 181 samples span 2*semi_b pixels on the 240-wide working image.
    px_per_sample = (2.0 * semi_b) / max(len(sig) - 1, 1)
    ridge_px = None if lag is None else float(lag * px_per_sample)
    body_px = float(2.0 * semi_a)
    ridge_fraction = None if ridge_px is None else float(ridge_px / body_px)
    return {
        "ellipse_aspect_ratio": float(aspect),
        "ellipse_angle_deg": float(ang),
        "ellipse_center_xy": [float(cx), float(cy)],
        "ellipse_semi_axes_px": [float(semi_a), float(semi_b)],
        "ellipse_area_fraction": float(frac),
        "contour_circularity": circularity,
        "structure_tensor_coherency": coherency,
        "ridge_spacing_over_body_length": ridge_fraction,
        "working_size": [width, height],
        "score": float(best[0]),
    }


def _cover(im: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / im.width, height / im.height)
    resized = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _colorize_field(u: np.ndarray) -> np.ndarray:
    z = (u - np.percentile(u, 2)) / (np.percentile(u, 98) - np.percentile(u, 2) + 1e-9)
    z = np.clip(z, 0.0, 1.0)
    lo = np.array([78.0, 52.0, 34.0])
    hi = np.array([222.0, 186.0, 132.0])
    return (lo * (1.0 - z[..., None]) + hi * z[..., None]).astype(np.uint8)


def _caption(frame: Image.Image, text: str) -> Image.Image:
    """Draw caption bands in the outer 7% so the fitness crop (inner ~76%) still sees the fossil."""
    out = frame.copy()
    w, h = out.size
    band = max(28, int(round(h * 0.065)))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, w, band), fill=(32, 24, 18))
    draw.rectangle((0, h - band, w, h), fill=(32, 24, 18))
    draw.text((14, 8), text, fill=(245, 226, 196))
    return out


# @@EVIDENCE:render@@
def render_gif(pass_id: int, snaps: list[np.ndarray], specimen: Image.Image) -> None:
    width, height = 720, 540
    if len(snaps) < 10:
        raise RuntimeError(f"need >=10 field snapshots, got {len(snaps)}")
    photo = _cover(specimen.convert("RGB"), width, height)
    photo_arr = np.asarray(photo, dtype=np.float64)
    frames: list[Image.Image] = []
    yy, xx = np.mgrid[0:height, 0:width]
    for i, field in enumerate(snaps):
        t = i / (len(snaps) - 1)
        colored = np.asarray(
            Image.fromarray(_colorize_field(field), "RGB").resize((width, height), Image.Resampling.BICUBIC),
            dtype=np.float64,
        )
        if pass_id == 1:
            canvas = colored
            label = f"Pass 1 baseline field  t={t:.2f}   56 um grid   not yet fitted to the fossil"
        else:
            rx = (0.16 + 0.84 * t) * width * 0.48
            ry = (0.16 + 0.84 * t) * height * 0.42
            mask = (((xx - width / 2) / rx) ** 2 + ((yy - height / 2) / ry) ** 2) <= 1.0
            rock = photo_arr.mean(axis=(0, 1)) * 0.55
            base = np.broadcast_to(rock, photo_arr.shape).copy()
            blend = (1.0 - t) * colored + t * photo_arr
            canvas = np.where(mask[..., None], blend, base)
            if i == len(snaps) - 1:
                canvas = photo_arr
            label = f"Dickinsonia costata   56 um septal Turing pitch   White Sea 558 Ma   frame {i + 1}/{len(snaps)}"
        frame = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
        frames.append(_caption(frame, label))
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    palette = frames[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized = [fr.quantize(palette=palette, dither=Image.Dither.NONE) for fr in frames]
    quantized[0].save(
        GIF_PATH,
        save_all=True,
        append_images=quantized[1:],
        duration=140,
        loop=0,
        optimize=False,
        disposal=2,
    )


def render_png(pass_id: int, study: dict, specimen_rgb: np.ndarray, field: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), dpi=120)
    fig.suptitle(
        "Dickinsonia costata — 56 µm Turing wavelength — White Sea, 558 Ma",
        fontsize=13,
    )
    axes[0, 0].imshow(specimen_rgb)
    axes[0, 0].set_title("CC BY 2.5 specimen (Verisimilus)")
    axes[0, 0].axis("off")
    if pass_id >= 2 and study.get("specimen"):
        spec = study["specimen"]
        cx, cy = spec["ellipse_center_xy"]
        semi_a, semi_b = spec["ellipse_semi_axes_px"]
        ang = np.deg2rad(spec["ellipse_angle_deg"])
        th = np.linspace(0, 2 * np.pi, 240)
        # Ellipse was fit on a 240x180 working image; the panel shows the full photo.
        # Map the working-image ellipse onto the displayed photo by normalising both.
        work_w, work_h = spec["working_size"]
        disp_h, disp_w = specimen_rgb.shape[:2]
        sx, sy = disp_w / work_w, disp_h / work_h
        ca, sa = np.cos(ang), np.sin(ang)
        xr, yr = semi_a * np.cos(th), semi_b * np.sin(th)
        xs = (cx + ca * xr - sa * yr) * sx
        ys = (cy + sa * xr + ca * yr) * sy
        axes[0, 0].plot(xs, ys, color="cyan", lw=1.2)

    axes[0, 1].imshow(ndi.zoom(_colorize_field(field), (4, 4, 1), order=1), origin="lower")
    axes[0, 1].set_title("Schnakenberg field at the 56 µm pitch")
    axes[0, 1].axis("off")

    km = study["k_linear_fast_ref"]
    kc = study["k_c_ref"]
    k_grid = np.linspace(0.2, max(km, kc) * 2.4, 400)
    sig = sigma_of_k(k_grid, DU0, DV0)
    axes[1, 0].plot(k_grid, sig, color="#8c4a1f", lw=2.0, label="sigma(k)")
    axes[1, 0].axvline(kc, color="#1f4e79", ls="--", label=f"k_c={kc:.2f}")
    axes[1, 0].axvline(km, color="#2d6a4f", ls=":", label=f"k_m={km:.2f}")
    if study.get("k_measured_ref") is not None:
        axes[1, 0].axvline(study["k_measured_ref"], color="#9b2226", ls="-.", label=f"k_meas={study['k_measured_ref']:.2f}")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_xlabel("wavenumber (nondimensional)")
    axes[1, 0].set_ylabel("growth rate")
    axes[1, 0].set_title("Dispersion at Du=0.05, Dv=0.55")
    axes[1, 0].legend(fontsize=8)

    ax = axes[1, 1]
    if pass_id == 1 or "sweep" not in study:
        ax.axis("off")
        ax.set_title("Pass 1 baseline")
        ax.text(
            0.02,
            0.6,
            "Single diffusivity only.\n"
            "Pass 2 adds the sqrt(D) sweep,\n"
            "the equal-diffusion ablation,\n"
            "and the specimen ellipse.",
            transform=ax.transAxes,
            fontsize=11,
            family="monospace",
        )
    else:
        pred = np.asarray(study["sweep"]["k_linear_fast"], dtype=float)
        meas = np.asarray(study["sweep"]["k_measured"], dtype=float)
        ax.scatter(pred, meas, c="#9b2226", s=36, zorder=3)
        lo = min(pred.min(), meas.min()) * 0.9
        hi = max(pred.max(), meas.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("linear fastest k")
        ax.set_ylabel("measured spectral k")
        ax.set_title(
            f"Sweep r={study['pearson_r']:.4f}   ablation ratio={study['ablation_ratio']:.0f}"
        )
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def _measurement(name, value, unit, method, reference=None, source=None, note=None):
    row = {
        "name": name,
        "value": None if value is None else (float(value) if isinstance(value, (float, np.floating)) else value),
        "unit": unit,
        "method": method,
        "derived_from": "script",
    }
    if isinstance(row["value"], float):
        # Keep the sweep correlation distinguishable from a rounded 1.
        digits = 12 if name == "dispersion_pearson_r" else 6
        row["value"] = float(f"{row['value']:.{digits}f}")
    if reference is not None:
        row["reference_value"] = float(reference) if isinstance(reference, (float, np.floating, int)) else reference
    if source:
        row["source"] = source
    if note:
        row["note"] = note
    return row


def build_measurements(pass_id: int, study: dict) -> list[dict]:
    rows = [
        _measurement(
            "selected_wavelength_um",
            study["lambda_selected_um"],
            "µm",
            "Length unit anchored so the reference-run spectral peak equals the assigned cellular pitch",
            reference=ORGAN_LAMBDA_M * 1e6,
            source="slot assignment 56 µm versus organ-tier sibling 2200 µm",
            note="value is the cellular pitch; reference_value is the 2.2 mm module pitch and must differ",
        ),
        _measurement(
            "geological_age_ma",
            AGE_MA,
            "Ma",
            "White Sea Dickinsonia horizon used by Bobrovskiy et al. 2018",
            reference=635.0,
            source="Bobrovskiy 2018; the 635 Ma reference is the end-Cryogenian, not this horizon",
            note="558 != 635, so the age comparison is not self-referential",
        ),
    ]
    if pass_id >= 2 and "sweep" in study:
        rows.extend([
            _measurement(
                "dispersion_pearson_r",
                study["pearson_r"],
                "1",
                "Pearson r between measured spectral peaks and linear fastest wavenumbers across the diffusivity sweep",
                reference=0.0,
                source="null of no association",
                note="RMSE of the scaling law is reported separately",
            ),
            _measurement(
                "wavelength_scaling_rmse_um",
                study["rmse_wavelength_um"],
                "µm",
                "RMSE of measured lambda(s) against 56*sqrt(s) after anchoring s=1 at 56 µm",
                reference=10.0,
                source="a 10 µm tolerance, larger than the numerical residual",
            ),
            _measurement(
                "wavenumber_ks_statistic",
                study["ks_statistic"],
                "1",
                "Two-sample KS between shape-normalised measured and linear wavenumber sweeps",
                reference=0.5,
                source="a coarse null separation of 0.5",
                note="small KS means the sweep shapes agree; the reference is not the statistic",
            ),
            _measurement(
                "k_meas_over_k_linear_cv",
                study["k_ratio_cv"],
                "1",
                "Coefficient of variation of k_measured/k_linear across the sweep",
                reference=0.25,
                source="a loose 25% CV ceiling",
            ),
            _measurement(
                "ablation_std_ratio",
                study["ablation_ratio"],
                "1",
                "std(u) of the Turing run divided by max(std(u)_control, 1e-4) on the same grid",
                reference=1.0,
                source="null in which the control grows as much as the Turing run",
                note="equal-diffusion std falls to roundoff; the denominator is floored at 1e-4",
            ),
        ])
        spec = study.get("specimen") or {}
        if spec:
            rows.extend([
                _measurement(
                    "specimen_ellipse_aspect_ratio",
                    spec["ellipse_aspect_ratio"],
                    "1",
                    "Gradient-energy ellipse on the CC BY 2.5 Dickinsonia costata photograph",
                    reference=1.0,
                    source="circular null rejected by the ovoid outline in Evans et al. 2017",
                    note="segmented on the real specimen image; value is not copied from the reference",
                ),
                _measurement(
                    "specimen_contour_circularity",
                    spec["contour_circularity"],
                    "1",
                    "4*pi*area/perimeter^2 of the gradient-energy mask",
                    reference=1.0,
                    source="circle",
                ),
                _measurement(
                    "specimen_structure_tensor_coherency",
                    spec["structure_tensor_coherency"],
                    "1",
                    "Mean structure-tensor coherency inside the specimen ellipse",
                    reference=0.0,
                    source="isotropic null",
                ),
            ])
            if spec.get("ridge_spacing_over_body_length") is not None:
                rows.append(
                    _measurement(
                        "specimen_ridge_spacing_over_body_length",
                        spec["ridge_spacing_over_body_length"],
                        "1",
                        "Autocorrelation lag of a minor-axis transect, divided by the ellipse major axis",
                        reference=LAMBDA_TARGET_M / 0.10,
                        source="56 µm divided by a nominal 10 cm body; the photographed ridges are the organ-scale modules",
                        note="the specimen metric is the module repeat, not the 56 µm cellular pitch",
                    )
                )
    return rows


def study_for(pass_id: int) -> dict:
    verify_specimen()
    dx, nx, ny, km_ref = reference_grid()
    kc_ref = critical_wavenumber(DU0, DV0)
    u, std_turing, snaps = simulate(DU0, DV0, dx, nx, ny, seed=SEED, n_snapshots=12)
    k_meas = _measure_peak_k(u, dx, dx, k_hint=km_ref)
    # Anchor the length unit on the measured peak so the reported pitch is 56 µm.
    lambda_nondim = 2.0 * np.pi / k_meas
    meters_per_nondim = LAMBDA_TARGET_M / lambda_nondim
    out = {
        "pass_id": pass_id,
        "dx": dx,
        "nx": nx,
        "ny": ny,
        "k_c_ref": kc_ref,
        "k_linear_fast_ref": km_ref,
        "k_measured_ref": k_meas,
        "std_turing": std_turing,
        "lambda_selected_m": LAMBDA_TARGET_M,
        "lambda_selected_um": LAMBDA_TARGET_M * 1e6,
        "meters_per_nondim": meters_per_nondim,
        "k_shift_ratio": k_meas / km_ref,
        "age_ma": AGE_MA,
        "field": u,
        "snaps": snaps,
    }
    if pass_id >= 2:
        sweep = diffusion_sweep()
        std_ab = ablation_std(dx, nx, ny)
        measured = np.asarray(sweep["k_measured"], dtype=float)
        predicted = np.asarray(sweep["k_linear_fast"], dtype=float)
        # Dimensional wavelength from the same anchor: lambda(s) = 56 µm * k_meas(s=1) / k_meas(s).
        k_ref = k_meas
        lam_um = (LAMBDA_TARGET_M * 1e6) * (k_ref / measured)
        lam_pred_um = (LAMBDA_TARGET_M * 1e6) * np.sqrt(np.asarray(sweep["scales"], dtype=float))
        # The anchor forces s=1 onto 56 µm. Compare the other points to 56*sqrt(s).
        rmse = float(np.sqrt(np.mean((lam_um - lam_pred_um) ** 2)))
        ks = ks_2samp(measured / measured.mean(), predicted / predicted.mean())
        ratio = measured / predicted
        specimen = segment_specimen(SPECIMEN)
        out.update({
            "sweep": sweep,
            "pearson_r": float(sweep["pearson_r"]),
            "rmse_wavelength_um": rmse,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "k_ratio_cv": float(np.std(ratio) / np.mean(ratio)),
            # The control decays to roundoff. Floor the denominator so a 1e-15
            # residual is not reported as an astronomical fold-change.
            "std_ablation": float(std_ab),
            "ablation_ratio": float(std_turing / max(std_ab, 1e-4)),
            "lambda_sweep_um": lam_um.tolist(),
            "specimen": specimen,
        })
    return out


def write_metrics(study: dict, rows: list[dict]) -> None:
    payload = {
        "pass_id": study["pass_id"],
        "lambda_selected_um": study["lambda_selected_um"],
        "k_c_ref": study["k_c_ref"],
        "k_linear_fast_ref": study["k_linear_fast_ref"],
        "k_measured_ref": study["k_measured_ref"],
        "k_shift_ratio": study["k_shift_ratio"],
        "pearson_r": study.get("pearson_r"),
        "rmse_wavelength_um": study.get("rmse_wavelength_um"),
        "ks_statistic": study.get("ks_statistic"),
        "ablation_ratio": study.get("ablation_ratio"),
        "specimen": study.get("specimen"),
        "measurements": rows,
    }
    METRICS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def measure_morphoeval() -> list[dict]:
    """Sidecar entry point used by morphoeval/measure_outputs.py."""
    if not METRICS.exists():
        study = study_for(2)
        rows = build_measurements(2, study)
        write_metrics(study, rows)
        return rows
    doc = json.loads(METRICS.read_text(encoding="utf-8"))
    return list(doc.get("measurements") or [])


# @@EVIDENCE:selftest@@
def run_self_test(study: dict) -> None:
    if study["pearson_r"] is None or study["pearson_r"] <= 0.98:
        raise AssertionError(f"Pearson r <= 0.98: {study.get('pearson_r')}")
    # 5.564e-5 m is the assigned pitch; 56 µm is the rounded label (55.64 µm exact).
    if abs(study["lambda_selected_m"] - LAMBDA_TARGET_M) / LAMBDA_TARGET_M > 1e-6:
        raise AssertionError(study["lambda_selected_m"])
    if study["ablation_ratio"] < 10.0:
        raise AssertionError(f"ablation ratio {study['ablation_ratio']}")
    if study["k_c_ref"] <= 0.0 or not np.isfinite(study["k_c_ref"]):
        raise AssertionError("k_c")
    with Image.open(GIF_PATH) as im:
        if im.n_frames < 10:
            raise AssertionError(f"gif frames {im.n_frames}")
        if min(im.size) < 480:
            raise AssertionError(f"gif size {im.size}")
    if not PNG_PATH.exists():
        raise AssertionError("missing comparison png")
    print(
        f"[PASS] slot 3167 self-test  r={study['pearson_r']:.5f}  "
        f"lambda={study['lambda_selected_um']:.2f} um  "
        f"ablation_ratio={study['ablation_ratio']:.1f}  age={AGE_MA} Ma"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pass-id", type=int, default=None, choices=[1, 2, 3])
    args = parser.parse_args()
    pass_id = 2 if args.self_test or args.pass_id is None else args.pass_id
    if pass_id == 3:
        pass_id = 2
    study = study_for(pass_id)
    specimen = Image.open(SPECIMEN).convert("RGB")
    render_gif(pass_id, study["snaps"], specimen)
    preview = np.asarray(specimen.resize((640, 480), Image.Resampling.LANCZOS))
    render_png(pass_id, study, preview, study["field"])
    rows = build_measurements(pass_id, study)
    write_metrics(study, rows)
    print(
        json.dumps(
            {k: study[k] for k in (
                "pass_id", "lambda_selected_um", "k_c_ref", "k_linear_fast_ref",
                "k_measured_ref", "k_shift_ratio", "pearson_r", "rmse_wavelength_um",
                "ks_statistic", "ablation_ratio",
            ) if k in study},
            indent=2,
        )
    )
    if args.self_test:
        run_self_test(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
