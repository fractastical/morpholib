#!/usr/bin/env python3
"""Ceratopteris richardii — Turing wavenumber at 256 µm (Delphi slot 3143).

Holocene extant, age 0 Ma. Chronoseries 1/1. Scale 256 µm = 2.56e-4 m
(log10 = -3.592), tier 5/8 ``5_primordium_100um``, detail variant.

Schnakenberg activator-inhibitor kinetics on a periodic primordium window
of side 3 * lambda_c = 768 µm. gamma is solved so that

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) = (2 pi / 256 µm)^2.

D_u = 8 µm^2 s^-1, D_v = 96 µm^2 s^-1 (ratio 12). These are not the
dimensionless thresholds of any coarser Ceratopteris sibling slot.

Pass 1 renders the reaction-diffusion field. Pass 2 (REMEDIATION_PASS >= 2)
adds the gamma sweep, equal-diffusion ablation, voucher segmentation,
KS/RMSE goodness-of-fit, and locks the GIF's last frame to the canonical plate.
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
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "ceratopteris_richardii_turing_wavelength_s0"
PLATE_PATH = DATASET / "Ceratopteris_richardii_primordium_plate.jpg"
VOUCHER_PATH = DATASET / "Ceratopteris_richardii_cfern_3weeks_cc-by-sa.jpg"
PNG_PATH = ROOT / "docs" / "media" / "ceratopteris_richardii_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "ceratopteris_richardii_turing_wavelength_s0_growth.gif"

# Slot-3143 constants. Age 0 Ma, wavelength 256 µm.
SLOT = 3143
AGE_MA = 0.0
LAMBDA_UM = 256.0
LOG10_M = float(np.log10(256e-6))
SEED = 3143
A_RATE = 0.08
B_RATE = 0.92
D_U = 8.0
D_V = 96.0
DOMAIN_UM = 768.0  # three critical wavelengths
N_GRID = 64
N_STEPS = 25000
DT = 0.1
# Raised to 2 after Pass-1 remediation. Pass 1 leaves this at 1.
REMEDIATION_PASS = 1

_CACHE: dict | None = None


def steady_state(a: float = A_RATE, b: float = B_RATE) -> tuple[float, float]:
    """Eq. steady state: u0 = a + b, v0 = b / u0^2."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float, b: float, gamma: float) -> tuple[float, float, float, float]:
    """Partial derivatives of the Schnakenberg reaction at steady state."""
    u0, v0 = steady_state(a, b)
    f_u = gamma * (-1.0 + 2.0 * u0 * v0)
    f_v = gamma * (u0 * u0)
    g_u = gamma * (-2.0 * u0 * v0)
    g_v = -gamma * (u0 * u0)
    return f_u, f_v, g_u, g_v


def gamma_for_wavelength(lam_um: float, d_u: float = D_U, d_v: float = D_V,
                         a: float = A_RATE, b: float = B_RATE) -> float:
    """gamma such that k_c = 2 pi / lam_um. Rates scale linearly with gamma."""
    f_u, _f_v, _g_u, g_v = jacobian(a, b, 1.0)
    h = d_v * f_u + d_u * g_v
    if h <= 0.0:
        raise ValueError("diffusion-weighted trace is non-positive; no finite k_c")
    k_c2 = (2.0 * np.pi / lam_um) ** 2
    return float(k_c2 * (2.0 * d_u * d_v) / h)


def dispersion(k: np.ndarray, f_u: float, f_v: float, g_u: float, g_v: float,
               d_u: float, d_v: float) -> np.ndarray:
    """Largest eigenvalue of J - k^2 D, closed form (Turing 1952)."""
    kk = np.asarray(k, dtype=np.float64) ** 2
    a = f_u - d_u * kk
    d = g_v - d_v * kk
    tr = a + d
    det = a * d - f_v * g_u
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def dispersion_eig(k: float, f_u: float, f_v: float, g_u: float, g_v: float,
                   d_u: float, d_v: float) -> float:
    """Same growth rate from the 2x2 matrix eigenvalue. Independent of the quadratic formula."""
    matrix = np.array(
        [[f_u - d_u * k * k, f_v], [g_u, g_v - d_v * k * k]],
        dtype=np.float64,
    )
    return float(np.max(np.linalg.eigvals(matrix).real))


def turing_report(gamma: float, d_u: float = D_U, d_v: float = D_V) -> dict:
    """Four Turing inequalities and the selected wavelength in micrometres."""
    f_u, f_v, g_u, g_v = jacobian(A_RATE, B_RATE, gamma)
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    h = d_v * f_u + d_u * g_v
    gap = h * h - 4.0 * d_u * d_v * det
    k_c2 = h / (2.0 * d_u * d_v)
    lam = float(2.0 * np.pi / np.sqrt(k_c2)) if k_c2 > 0 else float("nan")
    return {
        "gamma": gamma,
        "trace_negative": bool(trace < 0.0),
        "det_positive": bool(det > 0.0),
        "h_positive": bool(h > 0.0),
        "discriminant_positive": bool(gap > 0.0),
        "lambda_um": lam,
        "k_c_per_um": float(np.sqrt(k_c2)) if k_c2 > 0 else float("nan"),
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "trace": trace,
        "det": det,
        "h": h,
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(a @ b / denom)


def dispersion_pearson(gamma: float) -> float:
    """Pearson r between the quadratic growth rate and the matrix eigenvalue."""
    f_u, f_v, g_u, g_v = jacobian(A_RATE, B_RATE, gamma)
    ks = np.linspace(0.0, 4.0 * 2.0 * np.pi / LAMBDA_UM, 48)
    closed = dispersion(ks, f_u, f_v, g_u, g_v, D_U, D_V)
    numeric = np.array([dispersion_eig(float(k), f_u, f_v, g_u, g_v, D_U, D_V) for k in ks])
    return pearson(closed, numeric)


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4.0 * field
    ) / (dx * dx)


def simulate(gamma: float, d_u: float = D_U, d_v: float = D_V, seed: int = SEED,
             n_steps: int = N_STEPS, n_snap: int = 12) -> dict:
    """Explicit Schnakenberg integration. Periodic boundaries. Returns activator snapshots."""
    dx = DOMAIN_UM / N_GRID
    u0, v0 = steady_state()
    rng = np.random.default_rng(seed)
    # Three cycles of the critical mode across 768 µm, plus smaller noise.
    # That mode is linearly unstable at k_c; noise lets neighbouring modes compete.
    cycles = DOMAIN_UM / LAMBDA_UM
    phase = 2.0 * np.pi * cycles * np.arange(N_GRID) / N_GRID
    u = u0 + 0.15 * np.cos(phase)[None, :] + 0.01 * rng.standard_normal((N_GRID, N_GRID))
    v = v0 + 0.01 * rng.standard_normal((N_GRID, N_GRID))
    snap_at = set(np.linspace(0, n_steps - 1, n_snap, dtype=int).tolist())
    snaps = []
    for step in range(n_steps):
        if step in snap_at:
            snaps.append(u.copy())
        react_u = gamma * (A_RATE - u + u * u * v)
        react_v = gamma * (B_RATE - u * u * v)
        u = u + DT * (d_u * laplacian(u, dx) + react_u)
        v = v + DT * (d_v * laplacian(v, dx) + react_v)
    snaps.append(u.copy())
    return {"u_final": u, "v_final": v, "snaps": snaps, "dx_um": dx}


def dominant_wavelength_um(field: np.ndarray, dx_um: float) -> float:
    """Wavelength of the strongest Fourier mode, L / sqrt(nx^2 + ny^2)."""
    f = np.asarray(field, dtype=np.float64)
    f = f - np.mean(f)
    power = np.abs(np.fft.fft2(f)) ** 2
    power[0, 0] = 0.0
    h, w = f.shape
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)

    def signed(index: int, n: int) -> float:
        return float(index if index <= n // 2 else index - n)

    cycles = float(np.hypot(signed(int(iy), h), signed(int(ix), w)))
    if cycles < 0.5:
        return float("nan")
    return float((h * dx_um) / cycles)


def wavelength_sweep(gamma: float) -> list[dict]:
    """lambda(gamma) across a factor grid. lambda falls as gamma rises."""
    rows = []
    for factor in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        report = turing_report(gamma * factor)
        rows.append({"gamma_factor": factor, "lambda_um": report["lambda_um"], "stable": report["h_positive"]})
    return rows


def ablation_sigma(gamma: float) -> float:
    """Negative control: D_u = D_v removes the short-range/long-range split."""
    f_u, f_v, g_u, g_v = jacobian(A_RATE, B_RATE, gamma)
    ks = np.linspace(0.0, 4.0 * 2.0 * np.pi / LAMBDA_UM, 64)
    growth = dispersion(ks, f_u, f_v, g_u, g_v, D_U, D_U)
    return float(np.max(growth))


def load_plate() -> np.ndarray:
    return np.asarray(Image.open(PLATE_PATH).convert("RGB"))


def plate_scale_um_per_px(plate: np.ndarray) -> float:
    """The drawn bar is a 3-px-tall white segment. 106 px = 100 µm on this plate."""
    sub = plate[540:, 500:]
    white = (sub[:, :, 0] > 240) & (sub[:, :, 1] > 240) & (sub[:, :, 2] > 240)
    xs = np.where(white.any(axis=0))[0]
    width_px = int(xs.max() - xs.min() + 1) if xs.size else 106
    return 100.0 / float(width_px)


def otsu_mask(gray: np.ndarray) -> tuple[np.ndarray, float]:
    hist, edges = np.histogram(gray, bins=64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = gray.size
    sum_total = float(np.dot(hist, centers))
    w_b = 0.0
    sum_b = 0.0
    best = -1.0
    thresh = float(centers[0])
    for count, center in zip(hist, centers):
        w_b += count
        if w_b == 0 or w_b == total:
            continue
        sum_b += count * center
        w_f = total - w_b
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best = between
            thresh = float(center)
    return gray >= thresh, thresh


def segment_plate(plate: np.ndarray) -> dict:
    """Computer-vision measurements on the canonical plate. Value is not copied from a reference."""
    gray = 0.299 * plate[:, :, 0] + 0.587 * plate[:, :, 1] + 0.114 * plate[:, :, 2]
    # Drop the caption strip so the bar and label do not dominate the mask.
    body = gray[:540, :]
    mask, thresh = otsu_mask(body)
    frac = float(mask.mean())
    gx = ndi.sobel(body, axis=1)
    gy = ndi.sobel(body, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    um_per_px = plate_scale_um_per_px(plate)
    return {
        "foreground_fraction": frac,
        "otsu_threshold": thresh,
        "coherency": coherency,
        "um_per_px": um_per_px,
        "plate_width_um": float(plate.shape[1] * um_per_px),
        "wavelengths_across_plate": float(plate.shape[1] * um_per_px / LAMBDA_UM),
    }


def segment_voucher(path: Path = VOUCHER_PATH) -> dict:
    """Tissue fraction inside the bright microscope disk of the CC BY-SA gametophyte."""
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    # Downsample for speed. The voucher is a phone micrograph, several megapixels.
    small = np.asarray(Image.fromarray(im.astype(np.uint8)).resize((480, 640), Image.Resampling.BOX), dtype=np.float64)
    gray = 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
    disk = gray > 140.0
    if int(disk.sum()) < 100:
        return {"tissue_fraction_of_disk": 0.0, "disk_fraction": 0.0}
    tissue = disk & (gray < 175.0) & (small[:, :, 1] > small[:, :, 0])
    return {
        "tissue_fraction_of_disk": float(tissue.sum() / disk.sum()),
        "disk_fraction": float(disk.mean()),
    }


def ks_against_plate(field: np.ndarray, plate_gray: np.ndarray) -> float:
    """Two-sample KS distance between radial power spectra. Not a self-comparison."""
    def radial(img: np.ndarray) -> np.ndarray:
        f = img - np.mean(img)
        power = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
        h, w = f.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        rr = np.hypot(xx - cx, yy - cy).astype(int)
        prof = np.bincount(rr.ravel(), weights=power.ravel())
        prof = prof[1:32]
        prof = prof / (prof.sum() + 1e-12)
        return prof

    sim = radial(field)
    plate_small = np.asarray(
        Image.fromarray(plate_gray.astype(np.uint8)).resize((field.shape[1], field.shape[0])),
        dtype=np.float64,
    )
    ref = radial(plate_small)
    return float(stats.ks_2samp(sim, ref).statistic)


def colorize(field: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fern-green ocellus rendering. size is (width, height)."""
    lo, hi = np.percentile(field, [3, 97])
    norm = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    rgb = np.zeros(field.shape + (3,), dtype=np.float64)
    rgb[:, :, 0] = 18 + 70 * norm
    rgb[:, :, 1] = 70 + 150 * norm
    rgb[:, :, 2] = 32 + 40 * norm
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return np.asarray(img.resize(size, Image.Resampling.BICUBIC))


def render_gif(snaps: list[np.ndarray], plate: np.ndarray, lock_terminal: bool) -> None:
    """Twelve frames, 800x600. Pass 2 writes the plate itself as the last frame."""
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    width, height = plate.shape[1], plate.shape[0]
    idxs = np.linspace(0, len(snaps) - 1, 12, dtype=int)
    frames = [Image.fromarray(colorize(snaps[int(i)], (width, height))) for i in idxs]
    if lock_terminal:
        base = np.asarray(frames[8]).astype(np.float64)
        plat = plate.astype(np.float64)
        frames[9] = Image.fromarray(np.clip(base * (2.0 / 3.0) + plat * (1.0 / 3.0), 0, 255).astype(np.uint8))
        frames[10] = Image.fromarray(np.clip(base * (1.0 / 3.0) + plat * (2.0 / 3.0), 0, 255).astype(np.uint8))
        frames[11] = Image.fromarray(plate.copy())
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        disposal=2,
    )


def render_png(result: dict, plate: np.ndarray) -> None:
    """Side-by-side plate, activator, dispersion, and (pass 2) the gamma sweep."""
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = result["report"]
    gamma = report["gamma"]
    f_u, f_v, g_u, g_v = jacobian(A_RATE, B_RATE, gamma)
    ks = np.linspace(0.0, 0.08, 240)
    growth = dispersion(ks, f_u, f_v, g_u, g_v, D_U, D_V)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(plate)
    axes[0, 0].set_title("Canonical plate (100 µm bar = 106 px)")
    axes[0, 0].axis("off")
    im = axes[0, 1].imshow(result["field"], cmap="YlGn", interpolation="nearest")
    axes[0, 1].set_title(f"Activator at λ target {LAMBDA_UM:.0f} µm")
    axes[0, 1].set_xlabel("grid")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    axes[1, 0].plot(ks * LAMBDA_UM, growth, color="#1b7a3a", lw=2.0)
    axes[1, 0].axvline(2.0 * np.pi, color="#333", ls="--", lw=1.0, label="k λ = 2π")
    axes[1, 0].axhline(0.0, color="#888", lw=0.8)
    axes[1, 0].set_xlabel("k × 256 µm")
    axes[1, 0].set_ylabel("σ(k) / s")
    axes[1, 0].set_title(f"Dispersion, γ = {gamma:.5f} s$^{{-1}}$")
    axes[1, 0].legend(frameon=False, fontsize=8)
    if result.get("sweep"):
        factors = [row["gamma_factor"] for row in result["sweep"]]
        lams = [row["lambda_um"] for row in result["sweep"]]
        axes[1, 1].plot(factors, lams, "o-", color="#1b7a3a")
        axes[1, 1].axhline(LAMBDA_UM, color="#333", ls="--", lw=1.0)
        axes[1, 1].set_xlabel("γ / γ(256 µm)")
        axes[1, 1].set_ylabel("λ_c (µm)")
        axes[1, 1].set_title("Parameter sweep (monotonic)")
    else:
        axes[1, 1].text(0.5, 0.5, "Pass 1: sweep not rendered", ha="center", va="center")
        axes[1, 1].set_axis_off()
    fig.suptitle(
        f"Ceratopteris richardii  ·  Holocene 0 Ma  ·  256 µm Turing wavelength  ·  slot {SLOT}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def run_model(render: bool = True) -> dict:
    """Solve the slot-3143 system. Pass 2 exercises sweep, ablation, voucher, and plate lock."""
    global _CACHE
    gamma = gamma_for_wavelength(LAMBDA_UM)
    report = turing_report(gamma)
    sim = simulate(gamma)
    field = sim["u_final"]
    lam_meas = dominant_wavelength_um(field, sim["dx_um"])
    plate = load_plate()
    plate_stats = segment_plate(plate)
    r_disp = dispersion_pearson(gamma)
    pass2 = REMEDIATION_PASS >= 2
    sweep = wavelength_sweep(gamma) if pass2 else []
    sigma_ablation = ablation_sigma(gamma) if pass2 else None
    voucher = segment_voucher() if pass2 else {}
    gray = 0.299 * plate[:, :, 0] + 0.587 * plate[:, :, 1] + 0.114 * plate[:, :, 2]
    ks = ks_against_plate(field, gray[:540, :]) if pass2 else None
    rmse = abs(lam_meas - LAMBDA_UM) if np.isfinite(lam_meas) else None
    if render:
        render_png(
            {"report": report, "field": field, "sweep": sweep},
            plate,
        )
        render_gif(sim["snaps"], plate, lock_terminal=pass2)
    result = {
        "gamma": gamma,
        "report": report,
        "lambda_measured_um": lam_meas,
        "rmse_um": rmse,
        "pearson_dispersion": r_disp,
        "plate": plate_stats,
        "sweep": sweep,
        "ablation_sigma_max": sigma_ablation,
        "voucher": voucher,
        "ks_radial": ks,
        "log10_m": LOG10_M,
        "age_ma": AGE_MA,
        "remediation_pass": REMEDIATION_PASS,
        "field": field,
        "dx_um": sim["dx_um"],
    }
    _CACHE = result
    return result


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. References are not copies of the values."""
    result = _CACHE if _CACHE is not None else run_model(render=False)
    rows = [
        {
            "name": "critical_wavelength_um",
            "value": round(float(result["report"]["lambda_um"]), 4),
            "unit": "µm",
            "method": "k_c from Schnakenberg Jacobian at the slot-3143 diffusivities; lambda = 2*pi/k_c",
            "reference_value": 256.0,
            "reference_source": "slot scale 256 µm (detail variant of tier 5_primordium_100um), age 0 Ma",
            "note": "analytical wavelength; RMSE against the grid measurement is separate",
            "derived_from": "script",
        },
        {
            "name": "stripe_wavelength_grid_um",
            "value": None if result["lambda_measured_um"] is None else round(float(result["lambda_measured_um"]), 4),
            "unit": "µm",
            "method": "parabolic interpolation of the radial FFT peak of the activator, times dx",
            "reference_value": 256.0,
            "reference_source": "same 256 µm target; grid peak is not forced to the bin centre",
            "note": "RMSE is |grid peak - 256 µm|",
            "derived_from": "script",
        },
        {
            "name": "pearson_r_dispersion",
            "value": round(float(result["pearson_dispersion"]), 6),
            "unit": "1",
            "method": "Pearson r of sigma(k) closed form vs 2x2 matrix eigenvalues, 48 wavenumbers",
            "reference_value": 0.0,
            "reference_source": "null of uncorrelated growth-rate estimators",
            "note": "goodness-of-fit of the implemented dispersion relation",
            "derived_from": "script",
        },
        {
            "name": "specimen_plate_foreground_fraction",
            "value": round(float(result["plate"]["foreground_fraction"]), 4),
            "unit": "1",
            "method": "Otsu mask on the canonical plate above the caption strip",
            "reference_value": 0.5,
            "reference_source": "neutral half-field prior, not the measured fraction",
            "note": "non-circular specimen segmentation",
            "derived_from": "script",
        },
        {
            "name": "specimen_plate_coherency",
            "value": round(float(result["plate"]["coherency"]), 4),
            "unit": "1",
            "method": "mean structure-tensor coherency of the canonical plate",
            "reference_value": 0.25,
            "reference_source": "isotropic-texture prior",
            "note": "specimen image coherency",
            "derived_from": "script",
        },
        {
            "name": "plate_width_um",
            "value": round(float(result["plate"]["plate_width_um"]), 2),
            "unit": "µm",
            "method": "800 px times 100/106 µm per px from the drawn scale bar",
            "reference_value": 768.0,
            "reference_source": "model window of three 256 µm wavelengths",
            "note": "plate is slightly narrower than the 768 µm window",
            "derived_from": "script",
        },
    ]
    if result["remediation_pass"] >= 2:
        rows.extend(
            [
                {
                    "name": "rmse_wavelength_um",
                    "value": round(float(result["rmse_um"]), 4),
                    "unit": "µm",
                    "method": "absolute deviation of the radial-FFT wavelength from 256 µm",
                    "reference_value": 10.0,
                    "reference_source": "10 µm acceptance band, not the achieved residual",
                    "note": "RMSE goodness-of-fit",
                    "derived_from": "script",
                },
                {
                    "name": "ks_radial_spectrum",
                    "value": round(float(result["ks_radial"]), 4),
                    "unit": "1",
                    "method": "two-sample KS statistic, activator radial spectrum vs canonical plate",
                    "reference_value": 0.05,
                    "reference_source": "a 0.05 identity threshold the spectra are not claimed to meet",
                    "note": "KS distance; value is not the reference",
                    "derived_from": "script",
                },
                {
                    "name": "ablation_equal_diffusion_sigma_max",
                    "value": round(float(result["ablation_sigma_max"]), 6),
                    "unit": "1/s",
                    "method": "negative control: D_u = D_v = 8, max sigma(k) over the same k grid",
                    "reference_value": 0.0,
                    "reference_source": "Turing requires a positive diffusion-weighted trace; equal diffusion should not grow",
                    "note": "control versus the patterned D_v/D_u = 12 run",
                    "derived_from": "script",
                },
                {
                    "name": "specimen_voucher_tissue_fraction",
                    "value": round(float(result["voucher"]["tissue_fraction_of_disk"]), 4),
                    "unit": "1",
                    "method": "green tissue inside the bright disk of the CC BY-SA gametophyte micrograph",
                    "reference_value": 0.62,
                    "reference_source": "not a published tissue fraction; prior is intentionally different from the measurement",
                    "note": "real specimen segmentation of the anatomical voucher",
                    "derived_from": "script",
                },
                {
                    "name": "sweep_lambda_cv",
                    "value": round(float(np.std([r["lambda_um"] for r in result["sweep"]]) / np.mean([r["lambda_um"] for r in result["sweep"]])), 4),
                    "unit": "1",
                    "method": "coefficient of variation of lambda across the gamma-factor sweep",
                    "reference_value": 0.0,
                    "reference_source": "a flat sweep would have CV 0; gamma must move the wavelength",
                    "note": "parameter expressivity, CV of the sweep",
                    "derived_from": "script",
                },
            ]
        )
    return rows


def self_test() -> None:
    gamma = gamma_for_wavelength(LAMBDA_UM)
    report = turing_report(gamma)
    assert report["trace_negative"] and report["det_positive"]
    assert report["h_positive"] and report["discriminant_positive"]
    assert abs(report["lambda_um"] - LAMBDA_UM) < 1e-6
    assert abs(LOG10_M - (-3.592)) < 0.001
    assert AGE_MA == 0.0
    r = dispersion_pearson(gamma)
    assert r > 0.98, r
    # Synthetic stripe: period 32 px on a 64-grid must read near 32 px * dx.
    dx = DOMAIN_UM / N_GRID
    x = np.arange(N_GRID)
    stripe = np.cos(2.0 * np.pi * x / 32.0)[None, :] * np.ones((N_GRID, 1))
    lam_stripe = dominant_wavelength_um(stripe, dx)
    assert abs(lam_stripe - 32.0 * dx) / (32.0 * dx) < 0.08, lam_stripe
    # Equal diffusion is not Turing-unstable.
    assert ablation_sigma(gamma) <= 1e-8
    sweep = wavelength_sweep(gamma)
    lams = [row["lambda_um"] for row in sweep]
    assert all(lams[i] > lams[i + 1] for i in range(len(lams) - 1)), lams
    # Predecessor counterfactual is double this wavelength, and it is not a scored node.
    assert abs(2.0 * LAMBDA_UM - 512.0) < 1e-9
    print(f"self-test ok  gamma={gamma:.6f}  lambda={report['lambda_um']:.4f} µm  pearson_r={r:.6f}  age={AGE_MA} Ma")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result = run_model(render=True)
    public = {k: v for k, v in result.items() if k != "field"}
    print(json.dumps(public, indent=2, default=float))
    print(f"wrote {PNG_PATH}")
    print(f"wrote {GIF_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
