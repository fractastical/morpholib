#!/usr/bin/env python3
"""
Dasypus novemcinctus — Turing reaction-diffusion wavelength at 4.108 µm (slot 3166).

Scale: 4.1 µm (4.108e-06 m, log10 = -5.386, tier 3_subcellular_um, detail variant).
Age: Holocene (Extant), 0 Ma. Chronoseries 1/1 (no predecessor or successor chronospecies).

Governing system (Schnakenberg 1979 kinetics; Turing 1952 dispersion):
    du/dt = a - u + u^2 v + D_u ∇²u
    dv/dt = b - u^2 v + D_v ∇²v
    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c = 2π / k_c = 4.108 µm

D_u and D_v are solved from that identity at diffusion ratio 20. They are not
the parameters of the 273 µm sibling slot. Periodic boundary conditions on a
square patch of side 12 λ_c.

Specimen: Rhododendrites, CC BY-SA 4.0, La Fortuna, Costa Rica, 2024-12-11.
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

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "dasypus_novemcinctus_turing_wavelength_s0"
SPECIMEN = DATASET / "dasypus_novemcinctus_specimen.jpg"
REFERENCE = DATASET / "dasypus_novemcinctus_turing_wavelength_s0_reference.json"
MEDIA_PNG = ROOT / "docs" / "media" / "dasypus_novemcinctus_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "dasypus_novemcinctus_turing_wavelength_s0_growth.gif"
SPECIMEN_SHA256 = "7f1ba0e3ab8bdb136b0e6ca8dc0e2b85669420f6c3499a9968fa9f19aa002ad5"

# Assigned detail-scale wavelength and Holocene age. Not a Chen/Vickaryous measurement.
LAMBDA_UM = 4.108
LAMBDA_M = 4.108e-6
AGE_MA = 0.0
PRIMORDIUM_UM = 273.0
MODE_PACKING = int(PRIMORDIUM_UM * 1000 // (LAMBDA_UM * 1000))  # 66

# Schnakenberg production rates, unique to this 4.108 µm reconstruction.
A_PROD = 0.10
B_PROD = 0.90
DIFFUSION_RATIO = 20.0

GRID_N = 96
N_WAVELENGTHS = 12
N_STEPS = 1500
DT = 0.05
SEED = 3166
N_FRAMES = 12
GIF_W, GIF_H = 640, 480
SWEEP_SCALES = (0.64, 0.81, 1.0, 1.21, 1.44, 1.69)
SEED_REPLICATES = (3166, 3167, 3168, 3169)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_specimen() -> str:
    """Pin the vendored CC BY-SA photograph before any measurement uses it."""
    if not SPECIMEN.exists():
        raise FileNotFoundError(SPECIMEN)
    digest = sha256_file(SPECIMEN)
    if digest != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 mismatch: {digest}")
    return digest


def schnakenberg_coefficients(diffusion_ratio: float = DIFFUSION_RATIO, lambda_um: float = LAMBDA_UM) -> dict:
    """Jacobian at the homogeneous state and diffusivities that place k_c at lambda_um.

    Eq. (1) u0 = a + b, v0 = b / u0^2.                         Schnakenberg 1979
    Eq. (2) f_u = -1 + 2 u0 v0, f_v = u0^2, g_u = -2 u0 v0, g_v = -u0^2.
    Eq. (3) k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π / k_c.   Turing 1952
    With D_v = δ D_u the identity fixes D_u = (δ f_u + g_v) / (2 δ k_c^2).
    """
    u0 = A_PROD + B_PROD
    v0 = B_PROD / u0**2
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0**2
    g_u = -2.0 * u0 * v0
    g_v = -u0**2
    det = f_u * g_v - f_v * g_u
    trace = f_u + g_v
    k_c = 2.0 * np.pi / lambda_um
    d_u = (diffusion_ratio * f_u + g_v) / (2.0 * diffusion_ratio * k_c**2)
    d_v = diffusion_ratio * d_u
    return {
        "a": A_PROD,
        "b": B_PROD,
        "u0": float(u0),
        "v0": float(v0),
        "f_u": float(f_u),
        "f_v": float(f_v),
        "g_u": float(g_u),
        "g_v": float(g_v),
        "det": float(det),
        "trace": float(trace),
        "k_c": float(k_c),
        "lambda_um": float(lambda_um),
        "D_u": float(d_u),
        "D_v": float(d_v),
        "diffusion_ratio": float(diffusion_ratio),
        "age_ma": AGE_MA,
    }


def growth_rate(k: float, coeff: dict) -> float:
    """Larger eigenvalue of J - k² D. Positive inside the Turing band."""
    f_u = coeff["f_u"] - coeff["D_u"] * k * k
    g_v = coeff["g_v"] - coeff["D_v"] * k * k
    trace = f_u + g_v
    det = f_u * g_v - coeff["f_v"] * coeff["g_u"]
    disc = max(trace * trace - 4.0 * det, 0.0)
    return 0.5 * (trace + disc**0.5)


def simulate(
    coeff: dict | None = None,
    n: int = GRID_N,
    n_steps: int = N_STEPS,
    n_frames: int = N_FRAMES,
    seed: int = SEED,
) -> dict:
    """Fourier-split explicit Schnakenberg integration, periodic boundaries."""
    coeff = coeff or schnakenberg_coefficients()
    domain_um = N_WAVELENGTHS * coeff["lambda_um"]
    dx = domain_um / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx_grid**2 + ky_grid**2
    eu = np.exp(-coeff["D_u"] * k2 * DT)
    ev = np.exp(-coeff["D_v"] * k2 * DT)
    rng = np.random.default_rng(seed)
    u = coeff["u0"] + 0.01 * rng.standard_normal((n, n))
    v = coeff["v0"] + 0.01 * rng.standard_normal((n, n))
    frame_idx = set(np.linspace(0, n_steps - 1, n_frames).astype(int).tolist())
    frames: list[np.ndarray] = []
    for step in range(n_steps):
        u = np.fft.ifft2(np.fft.fft2(u) * eu).real
        v = np.fft.ifft2(np.fft.fft2(v) * ev).real
        uv2 = u * u * v
        u = u + DT * (coeff["a"] - u + uv2)
        v = v + DT * (coeff["b"] - uv2)
        if step in frame_idx:
            frames.append(u.copy())
    return {
        "u": u,
        "v": v,
        "frames": frames,
        "dx_um": float(dx),
        "domain_um": float(domain_um),
        "std_u": float(u.std()),
        "coeff": coeff,
    }


def dominant_wavelength_um(field: np.ndarray, dx_um: float) -> float:
    """Radial-power peak of the activator, with a parabolic sub-bin correction."""
    centered = field - np.mean(field)
    power = np.abs(np.fft.fft2(centered)) ** 2
    power[0, 0] = 0.0
    n = field.shape[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx_um)
    ky = 2.0 * np.pi * np.fft.fftfreq(field.shape[1], d=dx_um)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    kr = np.sqrt(kx_grid**2 + ky_grid**2)
    k_max = float(kr.max())
    n_bins = 72
    edges = np.linspace(0.0, k_max, n_bins + 1)
    which = np.clip(np.digitize(kr.ravel(), edges) - 1, 0, n_bins - 1)
    weights = np.bincount(which, weights=power.ravel(), minlength=n_bins)
    counts = np.bincount(which, minlength=n_bins)
    profile = weights / np.maximum(counts, 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Ignore the domain-scale mode; the selected band sits near one wavelength.
    domain = n * dx_um
    mask = centers > (2.0 * np.pi / (domain / 3.0))
    masked = np.where(mask, profile, 0.0)
    peak = int(np.argmax(masked))
    if 0 < peak < n_bins - 1:
        y0, y1, y2 = profile[peak - 1], profile[peak], profile[peak + 1]
        denom = (y0 - 2.0 * y1 + y2)
        shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        shift = float(np.clip(shift, -0.5, 0.5))
    else:
        shift = 0.0
    k_peak = float(centers[peak] + shift * (edges[1] - edges[0]))
    return float(2.0 * np.pi / max(k_peak, 1e-9))


def _specimen_arrays(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verify_specimen()
    with Image.open(SPECIMEN) as im:
        rgb = np.asarray(im.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    # Mid-frequency carrier keeps carapace edges; it is not a pixel copy of the file.
    from scipy import ndimage as ndi

    carrier = gray - ndi.gaussian_filter(gray, sigma=2.4)
    return rgb, gray, carrier


def render_growth_gif(frames: list[np.ndarray], out_path: Path) -> None:
    """Twelve-frame growth animation. Last frame carries the saturated spot lattice."""
    from scipy import ndimage as ndi

    title_h, footer_h = 48, 36
    ix0, iy0 = 12, title_h
    ix1, iy1 = GIF_W - 12, GIF_H - footer_h
    iw, ih = ix1 - ix0, iy1 - iy0
    rgb, _gray, carrier = _specimen_arrays(iw, ih)
    last = frames[-1]
    vmin, vmax = np.percentile(last, [2, 98])
    pil_frames: list[Image.Image] = []
    for i, field in enumerate(frames):
        tau = i / max(1, len(frames) - 1)
        up = np.asarray(
            Image.fromarray(field.astype(np.float32), mode="F").resize((iw, ih), Image.Resampling.BICUBIC),
            dtype=np.float64,
        )
        pat = np.clip((up - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
        pat = ndi.gaussian_filter(pat, sigma=0.6)
        keratin = np.stack(
            (
                0.45 + 0.42 * pat,
                0.34 + 0.30 * pat,
                0.22 + 0.10 * pat,
            ),
            axis=-1,
        )
        mixed = 0.52 * rgb + 0.48 * keratin
        mixed = np.clip(mixed + (0.55 + 0.45 * tau) * carrier[..., None], 0.0, 1.0)
        canvas = Image.new("RGB", (GIF_W, GIF_H), (16, 20, 16))
        canvas.paste(Image.fromarray((mixed * 255.0).astype(np.uint8), mode="RGB"), (ix0, iy0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, GIF_W, title_h - 1], fill=(12, 16, 12))
        draw.text((14, 6), "Dasypus novemcinctus  Holocene  0 Ma  lambda_c = 4.108 um", fill=(245, 224, 170))
        draw.text((14, 26), f"Schnakenberg activator  frame {i + 1}/{len(frames)}  t/T = {tau:.2f}", fill=(220, 220, 210))
        draw.rectangle([0, GIF_H - footer_h, GIF_W, GIF_H], fill=(12, 16, 12))
        draw.text((14, GIF_H - 24), "k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)   periodic 12-wavelength patch", fill=(176, 196, 168))
        pil_frames.append(canvas)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=140,
        loop=0,
        optimize=False,
    )


def render_comparison_png(field: np.ndarray, measured_lambda_um: float, sweep: list[dict], out_path: Path) -> None:
    """Specimen, activator, carapace overlay, and the diffusion-scale sweep."""
    verify_specimen()
    with Image.open(SPECIMEN) as im:
        spec = np.asarray(im.convert("RGB"))
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), dpi=120)
    axes = axes.ravel()
    axes[0].imshow(spec)
    axes[0].set_title("Specimen (Rhododendrites, CC BY-SA 4.0)")
    vmin, vmax = np.percentile(field, [2, 98])
    axes[1].imshow(field, cmap="YlOrBr", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title(f"Activator u  (saturated λ ≈ {measured_lambda_um:.2f} µm)")
    up = np.asarray(
        Image.fromarray(np.clip((field - vmin) / (vmax - vmin + 1e-9), 0, 1).astype(np.float32), mode="F").resize(
            (spec.shape[1], spec.shape[0]), Image.Resampling.BICUBIC
        ),
        dtype=np.float64,
    )
    overlay = spec.astype(np.float64) / 255.0
    overlay = np.clip(overlay * (0.45 + 0.7 * up[..., None]), 0.0, 1.0)
    axes[2].imshow(overlay)
    axes[2].set_title("Pattern on the carapace photograph")
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    pred = np.array([row["predicted_um"] for row in sweep])
    meas = np.array([row["measured_um"] for row in sweep])
    axes[3].scatter(pred, meas, c="#8a5a2b", s=36, zorder=3)
    lo = float(min(pred.min(), meas.min()) * 0.92)
    hi = float(max(pred.max(), meas.max()) * 1.08)
    axes[3].plot([lo, hi], [lo, hi], color="#888888", lw=1, label="1:1")
    axes[3].set_xlim(lo, hi)
    axes[3].set_ylim(lo, hi)
    axes[3].set_xlabel("Analytic λ (µm)")
    axes[3].set_ylabel("Measured λ (µm)")
    axes[3].set_title("Diffusion-scale sweep")
    axes[3].set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"Dasypus novemcinctus  ·  4.108 µm Turing wavelength  ·  Holocene 0 Ma  ·  target λ = {LAMBDA_UM:.3f} µm",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _scaled_coeff(scale: float) -> dict:
    """Multiply both diffusivities by `scale`. Wavelength then scales as sqrt(scale)."""
    coeff = dict(schnakenberg_coefficients())
    coeff["D_u"] *= scale
    coeff["D_v"] *= scale
    h = coeff["D_v"] * coeff["f_u"] + coeff["D_u"] * coeff["g_v"]
    k_c = (h / (2.0 * coeff["D_u"] * coeff["D_v"])) ** 0.5
    coeff["k_c"] = float(k_c)
    coeff["lambda_um"] = float(2.0 * np.pi / k_c)
    return coeff


def parameter_sweep(scales: tuple[float, ...] = SWEEP_SCALES, n_steps: int = 1200) -> list[dict]:
    """R1: measured wavelength versus the analytic square-root law in D."""
    rows = []
    for scale in scales:
        coeff = _scaled_coeff(scale)
        sim = simulate(coeff, n_steps=n_steps, n_frames=2, seed=SEED)
        measured = dominant_wavelength_um(sim["u"], sim["dx_um"])
        rows.append(
            {
                "scale": float(scale),
                "predicted_um": float(coeff["lambda_um"]),
                "measured_um": float(measured),
                "std_u": float(sim["std_u"]),
            }
        )
    return rows


def ablation_control(n_steps: int = 800) -> dict:
    """R2: equal diffusivities remove the Turing band. Amplitude should collapse."""
    coeff = dict(schnakenberg_coefficients())
    coeff["D_v"] = coeff["D_u"]
    sim = simulate(coeff, n_steps=n_steps, n_frames=2, seed=7)
    return {"ablated_std_u": float(sim["std_u"]), "D_u": float(coeff["D_u"]), "D_v": float(coeff["D_v"])}


def seed_replicates(seeds: tuple[int, ...] = SEED_REPLICATES, n_steps: int = 1200) -> list[float]:
    """Repeated nominal runs for the coefficient of variation of the measured wavelength."""
    coeff = schnakenberg_coefficients()
    measured = []
    for seed in seeds:
        sim = simulate(coeff, n_steps=n_steps, n_frames=2, seed=seed)
        measured.append(dominant_wavelength_um(sim["u"], sim["dx_um"]))
    return measured


def goodness_of_fit(sweep: list[dict], replicates: list[float], patterned_std: float, ablated_std: float) -> dict:
    """R4: Pearson r, RMSE, KS distance of the amplitude collapse, and replicate CV."""
    from scipy import stats

    predicted = np.array([row["predicted_um"] for row in sweep], dtype=float)
    measured = np.array([row["measured_um"] for row in sweep], dtype=float)
    pearson_r = float(np.corrcoef(predicted, measured)[0, 1])
    rmse = float(np.sqrt(np.mean((measured - predicted) ** 2)))
    # KS between the sweep residuals and a zero-centered normal with the same spread.
    # A large statistic means the red-shift is systematic, not a symmetric noise cloud.
    # Two-sample KS of the measured wavelengths against the analytic predictions.
    # n = 6, so the p-value is coarse; the statistic is the reported distance.
    ks = stats.ks_2samp(measured, predicted)
    reps = np.array(replicates, dtype=float)
    cv = float(reps.std(ddof=1) / reps.mean()) if reps.mean() else float("nan")
    return {
        "pearson_r": pearson_r,
        "rmse_um": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wavelength_cv": cv,
        "amplitude_ratio": float(patterned_std / max(ablated_std, 1e-12)),
        "nonlinear_shift_ratio": float(measured[list(SWEEP_SCALES).index(1.0)] / LAMBDA_UM),
    }


def segment_specimen() -> dict:
    """R3: segment the vendored photograph. Counts are measurements, not the literature target."""
    from scipy import ndimage as ndi
    from scipy.signal import find_peaks

    verify_specimen()
    with Image.open(SPECIMEN) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    moss = (green > red + 12.0) & (green > blue + 8.0)
    warm = (red > green - 8.0) & (red > 90.0) & ~moss
    warm = ndi.binary_opening(warm, iterations=1)
    warm = ndi.binary_closing(warm, iterations=3)
    warm = ndi.binary_fill_holes(warm)
    labels, n_lab = ndi.label(warm)
    if n_lab == 0:
        raise RuntimeError("specimen segmentation found no foreground")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    mask = labels == int(np.argmax(counts))
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2))
    area = float(mask.sum())
    eroded = ndi.binary_erosion(mask)
    perimeter = float(np.logical_xor(mask, eroded).sum())
    circularity = float(4.0 * np.pi * area / (perimeter**2 + 1e-9))
    ys, xs = np.where(mask)
    aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coherency_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(coherency_map[mask].mean())
    y0 = int(ys.min())
    y1 = int(ys.min() + 0.62 * (ys.max() - ys.min()))
    x0 = int(xs.min() + 0.08 * (xs.max() - xs.min()))
    x1 = int(xs.min() + 0.55 * (xs.max() - xs.min()))
    band_crop = np.abs(ndi.sobel(gray[y0:y1, x0:x1], axis=0))
    profile = ndi.gaussian_filter1d(band_crop.mean(axis=1), 2.5)
    peaks, _props = find_peaks(profile, prominence=float(np.std(profile)) * 0.50, distance=8)
    # Shoulder-tile spacing from the first autocorrelation peak past the pixel scale.
    sx0 = int(xs.min() + 0.48 * (xs.max() - xs.min()))
    sx1 = int(xs.min() + 0.78 * (xs.max() - xs.min()))
    sy0 = int(ys.min() + 0.25 * (ys.max() - ys.min()))
    sy1 = int(ys.min() + 0.62 * (ys.max() - ys.min()))
    signal = gray[sy0:sy1, sx0:sx1].mean(axis=0)
    signal = signal - signal.mean()
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    autocorr[0] = 0.0
    space_peaks, _ = find_peaks(autocorr, distance=5)
    spacing_px = float(space_peaks[space_peaks > 8][0]) if np.any(space_peaks > 8) else float("nan")
    return {
        "specimen_foreground_fraction": float(mask.mean()),
        "specimen_contour_circularity": circularity,
        "specimen_aspect_ratio": aspect,
        "specimen_structure_tensor_coherency": coherency,
        "specimen_movable_band_count": int(len(peaks)),
        "specimen_osteoderm_spacing_px": spacing_px,
    }


def run_analysis(render: bool = False) -> dict:
    """Nominal integration plus the Pass 2 sweep, ablation, segmentation, and fit."""
    coeff = schnakenberg_coefficients()
    sim = simulate(coeff)
    measured = dominant_wavelength_um(sim["u"], sim["dx_um"])
    sweep = parameter_sweep()
    ablated = ablation_control()
    replicates = seed_replicates()
    fit = goodness_of_fit(sweep, replicates, sim["std_u"], ablated["ablated_std_u"])
    specimen = segment_specimen()
    if render:
        render_growth_gif(sim["frames"], MEDIA_GIF)
        render_comparison_png(sim["u"], measured, sweep, MEDIA_PNG)
    return {
        "coeff": coeff,
        "measured_wavelength_um": measured,
        "sigma_kc": growth_rate(coeff["k_c"], coeff),
        "sigma_0": growth_rate(0.0, coeff),
        "std_u": sim["std_u"],
        "sweep": sweep,
        "ablated": ablated,
        "replicates_um": replicates,
        "fit": fit,
        "specimen": specimen,
        "mode_packing": MODE_PACKING,
        "age_ma": AGE_MA,
        "lambda_m": LAMBDA_M,
        "n_frames": len(sim["frames"]),
    }


def _metric(name, value, unit, method, reference, source, note, derived="script"):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "method": method,
        "reference_value": reference,
        "reference_source": source,
        "note": note,
        "derived_from": derived,
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Measured values are not copied onto the references."""
    analysis = run_analysis(render=False)
    fit = analysis["fit"]
    spec = analysis["specimen"]
    return [
        _metric(
            "saturated_activator_wavelength_um",
            round(analysis["measured_wavelength_um"], 4),
            "µm",
            "Parabolic peak of the radial Fourier power of the saturated activator",
            LAMBDA_UM,
            "Linear Turing wavelength pinned to the 4.108 µm detail rung",
            "Finite-amplitude spots sit long of linear k_c. value != reference.",
        ),
        _metric(
            "sweep_pearson_r",
            round(fit["pearson_r"], 6),
            "1",
            "Pearson correlation of measured versus analytic wavelength across six diffusion scales",
            0.98,
            "Pass gate is r > 0.98; the reported r is the measured correlation",
            "Reference is the gate, not the measurement.",
        ),
        _metric(
            "sweep_rmse_um",
            round(fit["rmse_um"], 4),
            "µm",
            "RMSE of measured versus analytic wavelength on the diffusion-scale sweep",
            0.0,
            "Zero RMSE would be an exact match to the linear law",
            "The offset is the nonlinear red-shift, reported rather than removed.",
        ),
        _metric(
            "wavelength_replicate_cv",
            round(fit["wavelength_cv"], 4),
            "1",
            "Sample CV of the saturated wavelength across four RNG seeds",
            0.0,
            "A deterministic field with no seed dependence would have CV 0",
            "Non-zero CV is the replicate scatter.",
        ),
        _metric(
            "ablation_ks_statistic",
            round(fit["ks_statistic"], 4),
            "1",
            "Two-sample KS statistic between measured and analytic wavelengths on the sweep",
            0.0,
            "KS = 0 would mean the measured and analytic samples share a distribution",
            "n = 6, so this is a distance, not a precise p-value.",
        ),
        _metric(
            "ablation_amplitude_ratio",
            round(fit["amplitude_ratio"], 4),
            "1",
            "std(u) with D_v/D_u = 20 divided by std(u) with D_u = D_v",
            1.0,
            "Equal diffusivities are the negative control; a ratio of 1 would mean the control failed",
            "The patterned state must be much larger than the ablated state.",
        ),
        _metric(
            "specimen_foreground_fraction",
            round(spec["specimen_foreground_fraction"], 4),
            "1",
            "Largest warm-tone component on the CC BY-SA photograph after moss rejection",
            1.0,
            "A frame-filling mask would have fraction 1",
            "The animal does not fill the frame.",
            "output",
        ),
        _metric(
            "specimen_contour_circularity",
            round(spec["specimen_contour_circularity"], 4),
            "1",
            "4 pi Area / Perimeter^2 of the smoothed foreground mask",
            1.0,
            "A circle has circularity 1",
            "The carapace silhouette is not circular.",
            "output",
        ),
        _metric(
            "specimen_structure_tensor_coherency",
            round(spec["specimen_structure_tensor_coherency"], 4),
            "1",
            "Mean structure-tensor coherency on the foreground mask",
            0.0,
            "An isotropic field has coherency 0",
            "Bands and tile edges raise coherency above 0.",
            "output",
        ),
        _metric(
            "specimen_movable_band_count",
            spec["specimen_movable_band_count"],
            "count",
            "Peaks in the vertical Sobel profile of the dorsal carapace crop",
            9,
            "The species is named for nine movable bands; Vickaryous & Hall 2006 describe the banded shield",
            "The peak counter is not a manual band census. value is whatever the profile yields.",
            "output",
        ),
        _metric(
            "specimen_osteoderm_spacing_px",
            round(spec["specimen_osteoderm_spacing_px"], 4),
            "px",
            "First autocorrelation peak beyond 8 px on the shoulder-tile crop",
            0.0,
            "Organism-scale photograph pixels, not the 4.108 µm model wavelength",
            "Spacing is in photo pixels and is not converted to 4.1 µm.",
            "output",
        ),
    ]


def self_test() -> dict:
    verify_specimen()
    coeff = schnakenberg_coefficients()
    if not (coeff["trace"] < 0.0 and coeff["det"] > 0.0):
        raise AssertionError(f"homogeneous state not stable: {coeff}")
    if growth_rate(coeff["k_c"], coeff) <= 0.0 or growth_rate(0.0, coeff) >= 0.0:
        raise AssertionError("Turing band signs failed")
    if MODE_PACKING != 66 or AGE_MA != 0.0 or LAMBDA_M != 4.108e-6:
        raise AssertionError("scale, age, or packing constant mismatch")
    k_back = ((coeff["D_v"] * coeff["f_u"] + coeff["D_u"] * coeff["g_v"]) / (2.0 * coeff["D_u"] * coeff["D_v"])) ** 0.5
    lam_back = 2.0 * np.pi / k_back
    if abs(lam_back - LAMBDA_UM) / LAMBDA_UM > 1e-6:
        raise AssertionError(f"dispersion identity failed: {lam_back}")
    analysis = run_analysis(render=False)
    fit = analysis["fit"]
    if not (fit["pearson_r"] > 0.98):
        raise AssertionError(f"Pearson r {fit['pearson_r']} is not above 0.98")
    if fit["amplitude_ratio"] < 20.0:
        raise AssertionError(f"ablation failed, amplitude ratio {fit['amplitude_ratio']}")
    if analysis["specimen"]["specimen_foreground_fraction"] <= 0.05:
        raise AssertionError("specimen segmentation collapsed")
    result = {
        "lambda_um": lam_back,
        "pearson_r": fit["pearson_r"],
        "rmse_um": fit["rmse_um"],
        "ks_statistic": fit["ks_statistic"],
        "wavelength_cv": fit["wavelength_cv"],
        "amplitude_ratio": fit["amplitude_ratio"],
        "nonlinear_shift_ratio": fit["nonlinear_shift_ratio"],
        "band_count": analysis["specimen"]["specimen_movable_band_count"],
        "mode_packing": MODE_PACKING,
        "age_ma": AGE_MA,
        "D_u": coeff["D_u"],
        "D_v": coeff["D_v"],
    }
    if MEDIA_GIF.exists():
        with Image.open(MEDIA_GIF) as im:
            n_frames = getattr(im, "n_frames", 1)
            w, h = im.size
        if n_frames < 10 or min(w, h) < 480:
            raise AssertionError(f"GIF gate failed: {n_frames} frames, {w}x{h}")
        result["gif_frames"] = n_frames
        result["gif_min_px"] = min(w, h)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Dasypus novemcinctus 4.108 µm Turing wavelength (slot 3166)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        info = self_test()
        print("[PASS] slot 3166 self-test", json.dumps(info))
        return 0
    out = run_analysis(render=True)
    printable = {
        "measured_wavelength_um": out["measured_wavelength_um"],
        "linear_lambda_um": out["coeff"]["lambda_um"],
        "sigma_kc": out["sigma_kc"],
        "sigma_0": out["sigma_0"],
        "std_u": out["std_u"],
        "ablated_std_u": out["ablated"]["ablated_std_u"],
        "fit": out["fit"],
        "specimen": out["specimen"],
        "sweep": out["sweep"],
        "replicates_um": out["replicates_um"],
        "mode_packing": out["mode_packing"],
        "age_ma": out["age_ma"],
        "lambda_m": out["lambda_m"],
        "D_u": out["coeff"]["D_u"],
        "D_v": out["coeff"]["D_v"],
    }
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
