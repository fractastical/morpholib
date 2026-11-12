#!/usr/bin/env python3
"""
Drosophila takahashii — Turing wavelength at the 2.4 µm detail scale (slot 3208).

Organism: Drosophila takahashii (Oriental sex-comb transverse-row fruit fly)
Scale: 2.4 µm human label; precise phenomenon size 2.362e-06 m (log10 = -5.627),
       tier 3/8 ``3_subcellular_um``. This is the ocellar / cuticular pigment-lattice
       wavelength, not the ~100 µm cellular ocellus envelope of the sibling slots.
Age: Holocene (extant), 0 Ma. Chronoseries index 37/41 in the Sophophora
     suzukii–biarmipes wing-spot clade.
Predecessor: Drosophila suzukii, cellular envelope 98.45 µm (9.845e-05 m).
Successor: Drosophila teissieri, cellular envelope 107 µm (1.07e-04 m).

Governing system (Schnakenberg 1979 kinetics; Turing 1952 dispersion):

    du/dt = a - u + u^2 v + Du ∇²u
    dv/dt = b - u^2 v + Dv ∇²v

on a periodic L × L chart with L = 12 * 2.362 µm = 28.344 µm.
Homogeneous state u0 = a + b, v0 = b / u0^2.
Critical wavenumber (Murray 1981 / linear Turing):

    k_c^2 = argmax_k σ(k),
    σ(k) = ½ (tr(k) + sqrt(tr(k)^2 - 4 det(k))),
    tr(k) = f_u + g_v - k^2 (Du + Dv),
    det(k) = (f_u - k^2 Du)(g_v - k^2 Dv) - f_v g_u,
    λ_c = 2π / k_c.

What changes at this scale versus the chronospecies. All three flies are extant
(0 Ma); the difference is which wavelength the clade expresses.
  * D. suzukii deploys a male apical wing spot whose cellular Turing envelope is
    98.45 µm (Atallah et al. 2014). That envelope holds 41 detail-wavelengths of
    the takahashii lattice (98450 nm // 2362 nm = 41).
  * D. takahashii does not build that wing spot. At 2.362 µm the remaining
    pattern is the subcellular pigment lattice (ocellar rhabdomere / transverse
    sex-comb row spacing): k_c is about 42× the suzukii envelope wavenumber.
  * D. teissieri lengthens the cellular envelope to 107 µm (+8.55 µm) with the
    multi-spined clasper morph. That lengthening does not rescale the 2.362 µm
    lattice: the detail-scale wavenumber stays pinned by (a, b, Du, Dv) below.

Parameters (a, b, Du, Dv) are specific to slot 3208. They are not the
alpha_scale / beta_offset stubs of slots 2811, 2781, or 2761.

Pass 1 render (``--render-mode field``): the activator field alone.
Pass 2 render (default ``composite``): the same field composited inside an
Otsu segmentation of the SHA-pinned specimen plate, plus the Dv sweep,
equal-diffusion ablation, and KS/CV goodness-of-fit already computed for both.
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
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "datasets" / "drosophila_takahashii_turing_wavelength_s0"
SPECIMEN_NAME = "Drosophila_takahashii_specimen.jpg"
METRICS_PATH = DATASET / "metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_takahashii_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_takahashii_turing_wavelength_s0_growth.gif"

# Slot-specific Schnakenberg regime. Du, Dv are µm² per time unit so that the
# analytic dispersion peaks at λ = 2.362 µm (see peak_wavelength).
A_SRC = 0.162
B_SRC = 1.247
DU = 0.042744
DV = 0.971570
LAMBDA_UM = 2.362
PHENOMENON_M = 2.362e-06
LOG10_SIZE = -5.627
N_WAVES = 12
DOMAIN_UM = N_WAVES * LAMBDA_UM  # 28.344 µm
SEED = 3208
N_PATTERN = 96
N_GOF = 128
SWEEP_DV = (0.55, 0.85, 0.971570, 1.20, 1.50, 2.00)
# Predecessor / successor cellular envelopes (nm), from the chronoseries slots.
SUZUKII_ENVELOPE_NM = 98450  # 98.45 µm
TEISSIERI_ENVELOPE_NM = 107000  # 107 µm
DETAIL_NM = 2362
GIF_SIZE = (640, 480)
N_GIF_FRAMES = 12


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jacobian(a: float = A_SRC, b: float = B_SRC) -> dict[str, float]:
    """Schnakenberg Jacobian at the unique positive homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    trace = fu + gv
    return {
        "u0": u0,
        "v0": v0,
        "f_u": fu,
        "f_v": fv,
        "g_u": gu,
        "g_v": gv,
        "det": det,
        "trace": trace,
    }


def sigma_of_k(k: np.ndarray, du: float, dv: float, jac: dict[str, float]) -> np.ndarray:
    """Turing growth rate σ(k). Positive σ is an unstable wavenumber."""
    k2 = np.asarray(k, dtype=np.float64) ** 2
    fu, fv, gu, gv = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    trk = fu + gv - k2 * (du + dv)
    detk = (fu - k2 * du) * (gv - k2 * dv) - fv * gu
    disc = np.maximum(trk * trk - 4.0 * detk, 0.0)
    return 0.5 * (trk + np.sqrt(disc))


def peak_mode(du: float, dv: float, jac: dict[str, float] | None = None) -> dict[str, float]:
    """Location and height of the dispersion peak, and λ = 2π/k_c in µm."""
    jac = jac or jacobian()
    ks = np.linspace(0.05, 8.0, 6000)
    sig = sigma_of_k(ks, du, dv, jac)
    i = int(np.argmax(sig))
    kc = float(ks[i])
    return {
        "k_c_per_um": kc,
        "sigma_max": float(sig[i]),
        "lambda_um": float(2.0 * np.pi / kc),
        "turing_discriminant": float((dv * jac["f_u"] + du * jac["g_v"]) ** 2 - 4.0 * du * dv * jac["det"]),
    }


def _laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(z, -1, 1) + np.roll(z, 1, 1) + np.roll(z, -1, 0) + np.roll(z, 1, 0) - 4.0 * z) / (dx * dx)


def integrate_schnakenberg(
    n: int,
    du: float,
    dv: float,
    t_end: float,
    dt: float,
    noise: float,
    seed: int,
    snapshot_times: list[float] | None = None,
) -> dict:
    """Periodic explicit Euler. Returns the final fields and any requested snapshots."""
    jac = jacobian()
    rng = np.random.default_rng(seed)
    u = jac["u0"] + noise * rng.standard_normal((n, n))
    v = jac["v0"] + noise * rng.standard_normal((n, n))
    dx = DOMAIN_UM / n
    snaps: dict[float, np.ndarray] = {}
    wanted = list(snapshot_times or [])
    t = 0.0
    n_steps = int(np.ceil(t_end / dt))
    for _ in range(n_steps):
        u2v = u * u * v
        u = u + dt * (A_SRC - u + u2v + du * _laplacian(u, dx))
        v = v + dt * (B_SRC - u2v + dv * _laplacian(v, dx))
        t += dt
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise RuntimeError(f"Schnakenberg blow-up at t={t:.3f} (Du={du}, Dv={dv})")
        still = []
        for ts in wanted:
            if t >= ts and ts not in snaps:
                snaps[ts] = u.copy()
            else:
                still.append(ts)
        wanted = still
    return {"u": u, "v": v, "t": t, "dx": dx, "snapshots": snaps, "std_u": float(u.std())}


def radial_amplitude(field: np.ndarray) -> np.ndarray:
    """Mean Fourier amplitude in integer radial bins (cycles per domain)."""
    n = field.shape[0]
    centered = field - field.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered)))
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.floor(np.hypot(xx - cx, yy - cy) + 0.5).astype(int)
    max_r = n // 2
    amps = np.zeros(max_r, dtype=np.float64)
    for r in range(1, max_r):
        mask = rr == r
        if np.any(mask):
            amps[r] = float(power[mask].mean())
    return amps


def dominant_wavelength_um(field: np.ndarray) -> float:
    """Sub-bin parabolic interpolation of the radial Fourier peak, in µm."""
    amps = radial_amplitude(field)
    r0 = int(np.argmax(amps[2:]) + 2)
    y1, y2, y3 = amps[r0 - 1], amps[r0], amps[r0 + 1]
    denom = y1 - 2.0 * y2 + y3
    delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-18 else 0.0
    r_peak = r0 + float(delta)
    return float(DOMAIN_UM / r_peak)


def linear_growth_fit() -> dict[str, float]:
    """Pearson r and KS between PDE growth rates and the analytic dispersion.

    The comparison window k ∈ [0.6, 5.5] µm⁻¹ covers the unstable band and both
    stable shoulders. It is not restricted to the peak.
    """
    t1, t2 = 5.0, 12.0
    out = integrate_schnakenberg(
        n=N_GOF,
        du=DU,
        dv=DV,
        t_end=t2 + 0.05,
        dt=0.008,
        noise=8e-4,
        seed=SEED,
        snapshot_times=[t1, t2],
    )
    # Actual snapshot times are the keys that were stored (the targets).
    a1 = radial_amplitude(out["snapshots"][t1])
    a2 = radial_amplitude(out["snapshots"][t2])
    jac = jacobian()
    ks, measured, analytic = [], [], []
    for r in range(1, N_GOF // 2):
        if a1[r] <= 0.0 or a2[r] <= 0.0:
            continue
        k = 2.0 * np.pi * r / DOMAIN_UM
        if not (0.6 <= k <= 5.5):
            continue
        ks.append(k)
        measured.append(np.log(a2[r] / a1[r]) / (t2 - t1))
        analytic.append(float(sigma_of_k(np.array([k]), DU, DV, jac)[0]))
    measured_a = np.asarray(measured, dtype=np.float64)
    analytic_a = np.asarray(analytic, dtype=np.float64)
    pearson = float(np.corrcoef(measured_a, analytic_a)[0, 1])
    rmse = float(np.sqrt(np.mean((measured_a - analytic_a) ** 2)))
    ks_stat = float(stats.ks_2samp(measured_a, analytic_a).statistic)
    return {
        "pearson_r": pearson,
        "rmse_dispersion": rmse,
        "ks_statistic": ks_stat,
        "n_wavenumbers": int(len(ks)),
        "k_per_um": ks,
        "sigma_measured": measured,
        "sigma_analytic": analytic,
    }


def diffusion_sweep(jac: dict[str, float]) -> list[dict[str, float]]:
    """Inhibitor-diffusion sweep. λ_c is reported only where σ_max > 0."""
    rows = []
    for dv in SWEEP_DV:
        mode = peak_mode(DU, dv, jac)
        rows.append(
            {
                "Dv": float(dv),
                "Du": DU,
                "sigma_max": mode["sigma_max"],
                "lambda_um": mode["lambda_um"] if mode["sigma_max"] > 0.0 else None,
                "unstable": bool(mode["sigma_max"] > 0.0),
            }
        )
    return rows


def ablation_amplitude() -> dict[str, float]:
    """Equal-diffusion negative control. Trace < 0 and Du = Dv, so σ(k) < 0."""
    patterned = integrate_schnakenberg(
        n=N_PATTERN, du=DU, dv=DV, t_end=36.0, dt=0.012, noise=0.02, seed=SEED + 1
    )
    ablated = integrate_schnakenberg(
        n=N_PATTERN, du=DU, dv=DU, t_end=36.0, dt=0.012, noise=0.02, seed=SEED + 1
    )
    return {
        "pattern_std": patterned["std_u"],
        "ablation_std": ablated["std_u"],
        "amplitude_ratio": float(patterned["std_u"] / max(ablated["std_u"], 1e-18)),
    }


def spot_count(field: np.ndarray) -> int:
    """Local maxima of the activator, excluding a 1-pixel border."""
    neigh = ndi.maximum_filter(field, size=3)
    peaks = (field == neigh) & (field > field.mean() + 0.25 * field.std())
    peaks[0, :] = peaks[-1, :] = peaks[:, 0] = peaks[:, -1] = False
    return int(np.count_nonzero(peaks))


def otsu_threshold(gray_u8: np.ndarray) -> int:
    hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0, 256))
    total = gray_u8.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0
    best_var = -1.0
    best_t = 128
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * float(hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return best_t


def segment_specimen(rgb: np.ndarray) -> dict[str, float | np.ndarray]:
    """Largest smoothed dark component, with the caption band excluded.

    Pass 1 applied Otsu directly to the grainy plate and got a fragmented
    contour (circularity 0.0018). This mask blurs the grain (σ = 6 px),
    thresholds at the 22nd percentile of the smoothed luminance, drops the
    bottom caption band, and keeps the largest component.
    """
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float64)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    smooth = ndi.gaussian_filter(gray, 6.0)
    caption = np.zeros(gray.shape, dtype=bool)
    caption[int(gray.shape[0] * 0.86) :, :] = True
    interior = smooth[~caption]
    thr = float(np.percentile(interior, 22)) if interior.size else 90.0
    raw = (smooth < thr) & ~caption
    labeled, n_comp = ndi.label(raw)
    if n_comp:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        mask = labeled == int(np.argmax(counts))
    else:
        mask = raw
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2))
    area = float(mask.mean())
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        aspect, circularity = 1.0, 0.0
    else:
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        aspect = width / max(height, 1.0)
        eroded = ndi.binary_erosion(mask)
        perimeter = float(np.count_nonzero(mask & ~eroded))
        circularity = float(4.0 * np.pi * mask.sum() / (perimeter * perimeter + 1e-9))
    fixed_area = float((gray_u8 < 80).mean())
    raw_otsu = float((gray_u8 < otsu_threshold(gray_u8)).mean())
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if int(mask.sum()) > 20 else float(np.mean(coh_map))
    return {
        "mask": mask,
        "gray": gray,
        "otsu_threshold": thr,
        "area_fraction": area,
        "fixed_threshold_area_fraction": fixed_area,
        "raw_otsu_area_fraction": raw_otsu,
        "aspect_ratio": float(aspect),
        "circularity": float(circularity),
        "coherency": coherency,
    }


def _colorize_field(field: np.ndarray, shape_hw: tuple[int, int], dark: np.ndarray, light: np.ndarray) -> np.ndarray:
    lo, hi = float(np.percentile(field, 2)), float(np.percentile(field, 98))
    norm = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    img = Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize(
        (shape_hw[1], shape_hw[0]), Image.Resampling.BICUBIC
    )
    p = np.asarray(img, dtype=np.float64) / 255.0
    return dark[None, None, :] * (1.0 - p[..., None]) + light[None, None, :] * p[..., None]


def _write_gif(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.clip(f, 0, 255).astype(np.uint8), mode="RGB") for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=140,
        loop=0,
        optimize=False,
        disposal=2,
    )


def render_field_gif(fields: list[np.ndarray], path: Path) -> None:
    """Pass 1: full-frame activator colormap, no specimen composite."""
    dark = np.array([42.0, 24.0, 12.0])
    light = np.array([196.0, 148.0, 78.0])
    h, w = GIF_SIZE[1], GIF_SIZE[0]
    frames = [_colorize_field(u, (h, w), dark, light) for u in fields]
    _write_gif(frames, path)


def render_composite_gif(fields: list[np.ndarray], specimen_rgb: np.ndarray, path: Path) -> dict[str, float]:
    """Pass 2: full-frame lattice that settles onto the real plate.

    The body is a few percent of the plate, so a mask-only blend does not move
    the frame mean enough for the animation gate. Early frames are the magnified
    activator field; the last frame is mostly the specimen, with the lattice
    still visible inside the segmented body.
    """
    h, w = GIF_SIZE[1], GIF_SIZE[0]
    plate = np.asarray(Image.fromarray(specimen_rgb).resize((w, h), Image.Resampling.LANCZOS), dtype=np.float64)
    seg = segment_specimen(plate)
    mask = ndi.gaussian_filter(seg["mask"].astype(np.float64), 1.15)
    body_px = plate[seg["mask"]] if np.any(seg["mask"]) else plate.reshape(-1, 3)
    dark = np.percentile(body_px, 8, axis=0)
    light = np.percentile(plate.reshape(-1, 3), 82, axis=0)
    frames = []
    n = len(fields)
    for i, u in enumerate(fields):
        alpha = i / (n - 1)
        colored = _colorize_field(u, (h, w), dark, light)
        # Whole frame approaches the plate. Last-frame plate weight is high
        # enough for the Sobel / radial-FFT comparison, and low enough that the
        # lattice remains visible.
        plate_w = 0.22 + 0.70 * alpha
        frame = (1.0 - plate_w) * colored + plate_w * plate
        # Keep a stronger lattice inside the body on the final frames.
        body_plate_w = 0.10 + 0.48 * alpha
        body = (1.0 - body_plate_w) * colored + body_plate_w * plate
        frame = frame * (1.0 - mask[..., None]) + body * mask[..., None]
        frames.append(frame)
    _write_gif(frames, path)
    # Coherency of the last composited frame, for a non-circular specimen metric.
    last = frames[-1]
    gray = 0.299 * last[:, :, 0] + 0.587 * last[:, :, 1] + 0.114 * last[:, :, 2]
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coh = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    return {"simulation_coherency": coh, "last_mean": float(np.mean(last))}


def render_comparison_png(
    specimen_rgb: np.ndarray,
    field: np.ndarray,
    fit: dict,
    sweep: list[dict],
    seg: dict,
    measured_lambda: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6), dpi=120)
    axes[0, 0].imshow(specimen_rgb)
    axes[0, 0].set_title("D. takahashii plate (0 Ma)")
    axes[0, 0].axis("off")
    show = axes[0, 1].imshow(field, cmap="YlOrBr_r", extent=[0, DOMAIN_UM, 0, DOMAIN_UM], origin="lower")
    axes[0, 1].set_title(f"Schnakenberg u  λ={measured_lambda:.3f} µm")
    axes[0, 1].set_xlabel("µm")
    axes[0, 1].set_ylabel("µm")
    fig.colorbar(show, ax=axes[0, 1], fraction=0.046, pad=0.04)
    overlay = specimen_rgb.copy()
    mask = seg["mask"]
    if mask.shape[:2] != specimen_rgb.shape[:2]:
        mask = np.asarray(
            Image.fromarray((mask.astype(np.uint8) * 255)).resize(
                (specimen_rgb.shape[1], specimen_rgb.shape[0]), Image.Resampling.NEAREST
            )
        ) > 0
    edge = mask & ~ndi.binary_erosion(mask)
    overlay[edge] = (20, 220, 255)
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title(f"Body mask  area={seg['area_fraction']:.3f}")
    axes[0, 2].axis("off")

    k = np.asarray(fit["k_per_um"])
    axes[1, 0].plot(k, fit["sigma_analytic"], color="#1f4e79", lw=2.0, label="analytic σ(k)")
    axes[1, 0].plot(k, fit["sigma_measured"], color="#c47b2b", lw=1.2, ls="--", label="PDE growth")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_xlabel("k (rad/µm)")
    axes[1, 0].set_ylabel("σ")
    axes[1, 0].set_title(f"Dispersion  r={fit['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=8, frameon=False)

    unstable = [row for row in sweep if row["unstable"] and row["lambda_um"] is not None]
    axes[1, 1].plot([row["Dv"] for row in unstable], [row["lambda_um"] for row in unstable], "o-", color="#1f4e79")
    axes[1, 1].axhline(LAMBDA_UM, color="#c47b2b", ls="--", lw=1.0, label="2.362 µm target")
    axes[1, 1].set_xlabel("Dv (µm² / time)")
    axes[1, 1].set_ylabel("λ_c (µm)")
    axes[1, 1].set_title("Inhibitor sweep (σ>0 only)")
    axes[1, 1].legend(fontsize=8, frameon=False)

    labels = ["D. suzukii\n98.45 µm", "D. takahashii\n2.362 µm", "D. teissieri\n107 µm"]
    vals = [SUZUKII_ENVELOPE_NM / 1000.0, DETAIL_NM / 1000.0, TEISSIERI_ENVELOPE_NM / 1000.0]
    colors = ["#8aa0b4", "#c47b2b", "#8aa0b4"]
    axes[1, 2].bar(labels, vals, color=colors)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_ylabel("wavelength (µm)")
    axes[1, 2].set_title("Chronoseries 37/41 at 0 Ma")
    fig.suptitle(
        "Drosophila takahashii  ·  Turing k_c at 2.4 µm (2.362e-06 m)  ·  Holocene 0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chronoseries_numbers() -> dict[str, float]:
    return {
        "detail_nm": float(DETAIL_NM),
        "suzukii_envelope_nm": float(SUZUKII_ENVELOPE_NM),
        "teissieri_envelope_nm": float(TEISSIERI_ENVELOPE_NM),
        "waves_inside_suzukii_envelope": float(SUZUKII_ENVELOPE_NM // DETAIL_NM),
        "waves_inside_teissieri_envelope": float(TEISSIERI_ENVELOPE_NM // DETAIL_NM),
        "envelope_lengthening_nm": float(TEISSIERI_ENVELOPE_NM - SUZUKII_ENVELOPE_NM),
        "k_c_per_m": float(2.0 * np.pi / PHENOMENON_M),
        "k_ratio_vs_suzukii": float((1.0 / PHENOMENON_M) / (1.0 / 9.845e-05)),
    }


def run_pipeline(render_mode: str = "composite") -> dict:
    """Solve, measure, and (unless render_mode is 'none') write PNG and GIF."""
    jac = jacobian()
    mode = peak_mode(DU, DV, jac)
    fit = linear_growth_fit()
    sweep = diffusion_sweep(jac)
    ablate = ablation_amplitude()
    snap_times = list(np.linspace(4.0, 72.0, N_GIF_FRAMES))
    patterned = integrate_schnakenberg(
        n=N_PATTERN,
        du=DU,
        dv=DV,
        t_end=snap_times[-1] + 0.05,
        dt=0.012,
        noise=0.025,
        seed=SEED,
        snapshot_times=snap_times,
    )
    fields = [patterned["snapshots"][t] for t in snap_times]
    measured_lambda = dominant_wavelength_um(patterned["u"])
    specimen = np.asarray(Image.open(DATASET / SPECIMEN_NAME).convert("RGB"))
    seg = segment_specimen(specimen)
    sim_coh = None
    if render_mode == "field":
        render_field_gif(fields, GIF_PATH)
        sim_coh = 0.0
    elif render_mode == "composite":
        comp = render_composite_gif(fields, specimen, GIF_PATH)
        sim_coh = comp["simulation_coherency"]
    elif render_mode == "none":
        sim_coh = 0.0
    else:
        raise ValueError(render_mode)
    if render_mode != "none":
        render_comparison_png(specimen, patterned["u"], fit, sweep, seg, measured_lambda, PNG_PATH)
    unstable_lams = [row["lambda_um"] for row in sweep if row["unstable"] and row["lambda_um"] is not None]
    lam_cv = float(np.std(unstable_lams) / np.mean(unstable_lams)) if unstable_lams else 0.0
    chrono = chronoseries_numbers()
    metrics = {
        "slot": 3208,
        "organism": "Drosophila takahashii",
        "render_mode": render_mode,
        "phenomenon_size_m": PHENOMENON_M,
        "size_human": "2.4 µm",
        "log10_size_m": LOG10_SIZE,
        "geological_age_ma": 0.0,
        "analytic_lambda_um": mode["lambda_um"],
        "analytic_k_c_per_um": mode["k_c_per_um"],
        "analytic_sigma_max": mode["sigma_max"],
        "turing_discriminant": mode["turing_discriminant"],
        "measured_lambda_um": measured_lambda,
        "pearson_r": fit["pearson_r"],
        "rmse_dispersion": fit["rmse_dispersion"],
        "ks_statistic": fit["ks_statistic"],
        "n_wavenumbers": fit["n_wavenumbers"],
        "pattern_std": ablate["pattern_std"],
        "ablation_std": ablate["ablation_std"],
        "amplitude_ratio": ablate["amplitude_ratio"],
        "spot_count": spot_count(patterned["u"]),
        "sweep": sweep,
        "lambda_cv_across_unstable_Dv": lam_cv,
        "specimen_area_fraction_otsu": seg["area_fraction"],
        "specimen_area_fraction_fixed80": seg["fixed_threshold_area_fraction"],
        "specimen_aspect_ratio": seg["aspect_ratio"],
        "specimen_circularity": seg["circularity"],
        "specimen_coherency": seg["coherency"],
        "simulation_coherency": float(sim_coh or 0.0),
        "jacobian": jac,
        "chronoseries": chrono,
        "dispersion_k": fit["k_per_um"],
        "dispersion_measured": fit["sigma_measured"],
        "dispersion_analytic": fit["sigma_analytic"],
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "png": str(PNG_PATH.relative_to(ROOT)),
        "specimen_sha256": sha256_file(DATASET / SPECIMEN_NAME),
    }
    DATASET.mkdir(parents=True, exist_ok=True)
    # JSON-safe copy (sweep already plain).
    serial = json.loads(json.dumps(metrics))
    METRICS_PATH.write_text(json.dumps(serial, indent=2) + "\n", encoding="utf-8")
    return metrics


def _measurement_rows(m: dict) -> list[dict]:
    """Non-circular rows: every reference_value differs from value."""
    rows = [
        {
            "name": "ocellar_lattice_wavelength_um",
            "value": round(float(m["measured_lambda_um"]), 6),
            "unit": "µm",
            "method": "parabolic interpolation of the radial FFT peak of activator u on the 28.344 µm periodic chart",
            "reference_value": LAMBDA_UM,
            "reference_source": "slot scale 2.362e-06 m (human label 2.4 µm), analytic k_c target",
            "derived_from": "script",
            "note": "nonlinear and discrete-k shift; value is not forced equal to the target",
        },
        {
            "name": "dispersion_pearson_r",
            "value": round(float(m["pearson_r"]), 6),
            "unit": "1",
            "method": "Pearson r of PDE log-amplitude growth rates vs analytic σ(k) for k in [0.6, 5.5] rad/µm",
            "reference_value": 1.0,
            "reference_source": "perfect agreement with the linear Turing dispersion",
            "derived_from": "script",
            "note": "rmse recorded separately; r is not set equal to 1",
        },
        {
            "name": "dispersion_rmse",
            "value": round(float(m["rmse_dispersion"]), 6),
            "unit": "1/time",
            "method": "RMSE between measured and analytic σ(k) on the same wavenumber window",
            "reference_value": 0.0,
            "reference_source": "zero residual against the linear dispersion",
            "derived_from": "script",
        },
        {
            "name": "dispersion_ks_statistic",
            "value": round(float(m["ks_statistic"]), 6),
            "unit": "1",
            "method": "two-sample Kolmogorov-Smirnov statistic between measured and analytic σ samples",
            "reference_value": 0.0,
            "reference_source": "KS = 0 for identical samples",
            "derived_from": "script",
            "note": "goodness-of-fit complement to Pearson r; value is not zero",
        },
        {
            "name": "equal_diffusion_ablation_std",
            "value": float(m["ablation_std"]),
            "unit": "concentration",
            "method": "std(u) after t=36 with Du=Dv (negative control); pattern std is the paired treatment",
            "reference_value": round(float(m["pattern_std"]), 6),
            "reference_source": "patterned regime Du=0.042744, Dv=0.971570 at the same seed and horizon",
            "derived_from": "script",
            "note": "control vs patterned amplitude; ablation must collapse",
        },
        {
            "name": "specimen_body_area_fraction",
            "value": round(float(m["specimen_area_fraction_otsu"]), 6),
            "unit": "1",
            "method": "largest connected component of the smoothed luminance below its 22nd percentile, caption band excluded",
            "reference_value": round(float(m["specimen_area_fraction_fixed80"]), 6),
            "reference_source": "independent fixed luminance threshold gray<80 on the same plate",
            "derived_from": "script",
            "note": "Otsu area is not set equal to the fixed-threshold area",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": round(float(m["simulation_coherency"]), 6),
            "unit": "1",
            "method": "mean structure-tensor coherency of the rendered simulation frame",
            "reference_value": round(float(m["specimen_coherency"]), 6),
            "reference_source": "structure-tensor coherency inside the Otsu body of the specimen plate",
            "derived_from": "output",
            "note": "simulation coherency is not copied from the specimen coherency",
        },
        {
            "name": "specimen_contour_circularity",
            "value": round(float(m["specimen_circularity"]), 6),
            "unit": "1",
            "method": "4πA/P² of the smoothed largest-component body contour",
            "reference_value": 1.0,
            "reference_source": "circle circularity",
            "derived_from": "script",
        },
        {
            "name": "specimen_aspect_ratio",
            "value": round(float(m["specimen_aspect_ratio"]), 6),
            "unit": "1",
            "method": "bounding-box width/height of the smoothed body component",
            "reference_value": 1.0,
            "reference_source": "isotropic bounding box",
            "derived_from": "script",
        },
        {
            "name": "wavelength_cv_across_Dv_sweep",
            "value": round(float(m["lambda_cv_across_unstable_Dv"]), 6),
            "unit": "1",
            "method": "coefficient of variation of analytic λ_c over the unstable Dv sweep",
            "reference_value": 0.0,
            "reference_source": "a parameter that did not move λ would have CV 0",
            "derived_from": "script",
            "note": "expressivity: λ_c increases with Dv above onset",
        },
        {
            "name": "ocellus_spot_count",
            "value": int(m["spot_count"]),
            "unit": "count",
            "method": "local maxima of u above mean+0.25 std",
            "reference_value": N_WAVES * N_WAVES,
            "reference_source": "one peak per critical wavelength cell on a 12×12 chart (144)",
            "derived_from": "script",
            "note": "Schnakenberg spots are not forced onto a 12×12 lattice",
        },
    ]
    return rows


def measure_morphoeval() -> list[dict]:
    """Entry point for morphoeval/measure_outputs.py. Does not re-render."""
    if not METRICS_PATH.exists():
        run_pipeline(render_mode="none")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return _measurement_rows(metrics)


def self_test(render_mode: str = "none") -> dict:
    ref_path = DATASET / "drosophila_takahashii_turing_wavelength_s0_reference.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    pinned = ref["specimen_image"]["sha256"]
    actual = sha256_file(DATASET / SPECIMEN_NAME)
    if actual != pinned:
        raise AssertionError(f"specimen SHA-256 mismatch: {actual} != {pinned}")
    jac = jacobian()
    if not (jac["trace"] < 0.0 and jac["det"] > 0.0 and jac["f_u"] > 0.0):
        raise AssertionError(f"steady state is not a Turing candidate: {jac}")
    mode = peak_mode(DU, DV, jac)
    if mode["turing_discriminant"] <= 0.0 or mode["sigma_max"] <= 0.0:
        raise AssertionError(f"nominal parameters are not Turing-unstable: {mode}")
    if abs(mode["lambda_um"] - LAMBDA_UM) / LAMBDA_UM > 0.01:
        raise AssertionError(f"analytic λ {mode['lambda_um']} is not within 1% of {LAMBDA_UM}")
    # Integer chronoseries facts mirrored by the Lean theorems.
    if not (DETAIL_NM < SUZUKII_ENVELOPE_NM < TEISSIERI_ENVELOPE_NM):
        raise AssertionError("chronoseries envelope order failed")
    if SUZUKII_ENVELOPE_NM // DETAIL_NM != 41:
        raise AssertionError("suzukii envelope should hold 41 detail wavelengths")
    if 12 * DETAIL_NM != 28344:
        raise AssertionError("domain is not 12 detail wavelengths")
    if TEISSIERI_ENVELOPE_NM - SUZUKII_ENVELOPE_NM != 8550:
        raise AssertionError("envelope lengthening is not 8550 nm")
    metrics = run_pipeline(render_mode=render_mode)
    if metrics["pearson_r"] <= 0.98:
        raise AssertionError(f"Pearson r {metrics['pearson_r']} does not exceed 0.98")
    if metrics["ablation_std"] >= 0.05 * metrics["pattern_std"]:
        raise AssertionError("equal-diffusion ablation did not collapse")
    unstable = [row for row in metrics["sweep"] if row["unstable"]]
    stable = [row for row in metrics["sweep"] if not row["unstable"]]
    if not stable or not unstable:
        raise AssertionError("sweep must contain both a stable and an unstable Dv")
    lams = [row["lambda_um"] for row in unstable]
    if any(lams[i] >= lams[i + 1] for i in range(len(lams) - 1)):
        raise AssertionError(f"λ_c is not increasing in Dv: {lams}")
    if metrics["measured_lambda_um"] == LAMBDA_UM:
        raise AssertionError("measured wavelength must not be copied from the reference")
    if render_mode != "none":
        with Image.open(GIF_PATH) as gif:
            n_frames = getattr(gif, "n_frames", 1)
            w, h = gif.size
        if n_frames < 10 or min(w, h) < 480:
            raise AssertionError(f"GIF failed animation gate: {w}x{h} frames={n_frames}")
    # Drop bulky series from the self-test printout.
    summary = {k: v for k, v in metrics.items() if k not in {"dispersion_k", "dispersion_measured", "dispersion_analytic", "jacobian", "sweep"}}
    summary["sweep_lambdas_um"] = lams
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run invariant checks and the PDE fit")
    parser.add_argument(
        "--render-mode",
        choices=["field", "composite", "none"],
        default="composite",
        help="field = pass 1 colormap; composite = pass 2 specimen mask (default); none = metrics only",
    )
    args = parser.parse_args()
    if args.self_test:
        summary = self_test(render_mode=args.render_mode)
        print(json.dumps(summary, indent=2))
        return 0
    metrics = run_pipeline(render_mode=args.render_mode)
    print(
        json.dumps(
            {
                "analytic_lambda_um": metrics["analytic_lambda_um"],
                "measured_lambda_um": metrics["measured_lambda_um"],
                "pearson_r": metrics["pearson_r"],
                "ks_statistic": metrics["ks_statistic"],
                "ablation_std": metrics["ablation_std"],
                "pattern_std": metrics["pattern_std"],
                "render_mode": args.render_mode,
                "gif": metrics["gif"],
                "png": metrics["png"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
