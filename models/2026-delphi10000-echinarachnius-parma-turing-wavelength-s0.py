#!/usr/bin/env python3
"""
Echinarachnius parma — Turing wavelength at 60 µm, Holocene (0 Ma), detail scale.

Slot 3214 of the 10,000-animal basic look. One animal, one scale, one age.

  Animal   Echinarachnius parma (common sand dollar), extant Holocene, 0 Ma.
           Chronoseries length 1/1: there is no earlier Echinarachnius
           chronospecies in this slot, so the predecessor/successor contrast
           is the scale ladder at age 0, not a stratigraphic series.
  Scale    60 µm (6e-05 m, log10 ≈ -4.222), tier 4/8 ``4_cellular_10um``,
           detail variant. This is the linear most-unstable wavelength of a
           Schnakenberg activator–inhibitor pair on an aboral epithelial patch,
           read as a pigment-ocellus / stripe spacing. It is not the
           centimetre-scale petaloid (Seilacher 1979).
  Law      Turing wavenumber. σ(k) is the larger eigenvalue of J − k²D.
           λ = 2π / k_max, with k_max = argmax σ(k).

Governing system (Schnakenberg 1979 kinetics; Turing 1952 dispersion):

    ∂u/∂t = Du ∇²u + a − u + u²v
    ∂v/∂t = Dv ∇²v + b − u²v

on a periodic L = 480 µm patch (eight 60 µm wavelengths). Steady state
u* = a + b, v* = b / (a + b)². Du and Dv below are spatial calibrations that
place the equilibrated lattice on the 60 µm slot scale. They are not measured
diffusion coefficients of sand-dollar tissue.

Sibling slot 160 (organism-scale cartoon, dimensionless α = 1.71) is a
different model and its parameters are not reused.

Primary specimen: USNM E39365, Smithsonian Open Access, CC0, via Wikimedia
Commons (living tests, 3 cm scale bar). The simulation is a 480 µm patch,
shown beside that plate; it is not a pixel copy of the photograph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "echinarachnius_parma_turing_wavelength_s0"
SPECIMEN = DATASET / "echinarachnius_parma_usnm_e39365_specimen.jpeg"
SPECIMEN_SHA256 = "1af65dc69de55d4b0ff00c1f4134565eed1f5e151e1539fbbdc52730ed880787"
RESULTS_JSON = DATASET / "echinarachnius_parma_turing_wavelength_s0_run_results.json"
COMPARISON_PNG = ROOT / "docs" / "media" / "echinarachnius_parma_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "echinarachnius_parma_turing_wavelength_s0_growth.gif"
LEAN_PATH = ROOT / "morphoeval" / "lean" / "loops" / "Loop_3214_echinarachnius_parma_turing_wavelength_s0.lean"

# Holocene detail scale. Age is 0 Ma; λ does not depend on age in a 1/1 series.
AGE_MA = 0.0
LAMBDA_UM = 60.0
PATCH_UM = 480.0          # eight wavelengths; the field of view at this scale
PREDECESSOR_UM = 10.0     # sub-wavelength cellular tier; the unstable mode does not fit
SUCCESSOR_UM = 600.0      # next coarser tier; ten wavelengths, still isotropic
SEED = 3214

# Schnakenberg kinetics. fu = 0.8, gv = -1 at the homogeneous state (see jacobian).
A_FEED = 0.10
B_FEED = 0.90
# Diffusivities (µm² per reaction time) at ratio Dv/Du = 12.
# Linear dispersion peak sits near 61.3 µm; one nonlinear rescale places the
# equilibrated FFT wavelength on the 60 µm slot (seed 3214, N = 128, L = 480).
DU_UM2 = 28.00571711952237
DV_UM2 = 336.0686054342684
DIFFUSION_RATIO = 12.0

SWEEP_SCALES = (0.35, 0.55, 0.80, 1.00, 1.40, 2.00, 3.00)
N_METRIC = 128
N_STEPS = 520
DT = 0.25
GIF_SIDE = 560
N_GIF_FRAMES = 12


def jacobian() -> dict[str, float]:
    """Homogeneous steady state and its reaction Jacobian."""
    u_star = A_FEED + B_FEED
    v_star = B_FEED / (u_star ** 2)
    fu = -1.0 + 2.0 * u_star * v_star
    fv = u_star ** 2
    gu = -2.0 * u_star * v_star
    gv = -(u_star ** 2)
    trace = fu + gv
    det = fu * gv - fv * gu
    return {
        "u_star": u_star,
        "v_star": v_star,
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trace": trace,
        "det": det,
    }


def dispersion_sigma(k2: np.ndarray, Du: float, Dv: float, jac: dict[str, float] | None = None) -> np.ndarray:
    """Larger eigenvalue of J − k² D. k2 is the squared wavenumber (1/µm²)."""
    J = jac or jacobian()
    tr = (J["fu"] - Du * k2) + (J["gv"] - Dv * k2)
    det = (J["fu"] - Du * k2) * (J["gv"] - Dv * k2) - J["fv"] * J["gu"]
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def linear_peak_wavelength(Du: float, Dv: float) -> tuple[float, float]:
    """Return (λ_µm, σ_max) at the most unstable linear wavenumber."""
    k2 = np.linspace(1e-8, 0.45, 16000)
    sigma = dispersion_sigma(k2, Du, Dv)
    i = int(np.argmax(sigma))
    k = float(np.sqrt(k2[i]))
    return (2.0 * np.pi / k, float(sigma[i]))


def turing_trace_gap(Du: float, Dv: float, jac: dict[str, float] | None = None) -> float:
    """Dv*fu + Du*gv. Positive when inhibitor diffusion can destabilise a band of k."""
    J = jac or jacobian()
    return float(Dv * J["fu"] + Du * J["gv"])


def _radial_wavelength(field: np.ndarray, length_um: float) -> float:
    """Dominant wavelength (µm) from the radial power spectrum, with a parabolic sub-bin peak."""
    n = field.shape[0]
    window = field - float(np.mean(field))
    power = np.abs(np.fft.fft2(window)) ** 2
    dx = length_um / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    k_nyquist = np.pi / dx
    bins = np.linspace(0.0, k_nyquist, 128)
    idx = np.clip(np.digitize(k_mag.ravel(), bins) - 1, 0, len(bins) - 2)
    prof = np.bincount(idx, weights=power.ravel(), minlength=len(bins) - 1).astype(float)
    count = np.bincount(idx, minlength=len(bins) - 1).astype(float)
    prof = prof / np.maximum(count, 1.0)
    prof[:2] = 0.0
    i = int(np.argmax(prof))
    if 0 < i < len(prof) - 1:
        y1, y2, y3 = prof[i - 1], prof[i], prof[i + 1]
        denom = y1 - 2.0 * y2 + y3
        delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-18 else 0.0
        delta = float(np.clip(delta, -0.6, 0.6))
    else:
        delta = 0.0
    k_peak = 0.5 * (bins[i] + bins[i + 1]) + delta * (bins[1] - bins[0])
    return float(2.0 * np.pi / max(k_peak, 1e-9))


def simulate_schnakenberg(
    Du: float,
    Dv: float,
    length_um: float = PATCH_UM,
    n: int = N_METRIC,
    steps: int = N_STEPS,
    dt: float = DT,
    seed: int = SEED,
    n_snapshots: int = 0,
) -> dict[str, np.ndarray | float]:
    """Periodic Fourier IMEX step. Diffusion is implicit; the reaction is explicit."""
    J = jacobian()
    rng = np.random.default_rng(seed)
    u = np.full((n, n), J["u_star"]) + 0.02 * rng.normal(size=(n, n))
    v = np.full((n, n), J["v_star"]) + 0.02 * rng.normal(size=(n, n))
    dx = length_um / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx_grid ** 2 + ky_grid ** 2
    den_u = 1.0 + dt * Du * k2
    den_v = 1.0 + dt * Dv * k2
    snap_idx = set(np.linspace(0, steps - 1, n_snapshots).astype(int).tolist()) if n_snapshots else set()
    snapshots: list[np.ndarray] = []
    for t in range(steps):
        if t in snap_idx:
            snapshots.append(u.copy())
        reaction_u = A_FEED - u + u * u * v
        reaction_v = B_FEED - u * u * v
        u = np.fft.ifft2(np.fft.fft2(u + dt * reaction_u) / den_u).real
        v = np.fft.ifft2(np.fft.fft2(v + dt * reaction_v) / den_v).real
    if n_snapshots and (steps - 1) not in snap_idx:
        snapshots.append(u.copy())
    return {
        "u": u,
        "v": v,
        "amplitude": float(np.std(u)),
        "wavelength_um": _radial_wavelength(u, length_um),
        "snapshots": snapshots,
    }


def wavelength_sweep(scales: tuple[float, ...] = SWEEP_SCALES) -> dict[str, list[float]]:
    """Scale both diffusivities by s. Linear peak λ moves as sqrt(s); the lattice should follow."""
    predicted: list[float] = []
    measured: list[float] = []
    amplitudes: list[float] = []
    for scale in scales:
        lam_lin, _sigma = linear_peak_wavelength(DU_UM2 * scale, DV_UM2 * scale)
        run = simulate_schnakenberg(DU_UM2 * scale, DV_UM2 * scale, steps=N_STEPS)
        predicted.append(float(lam_lin))
        measured.append(float(run["wavelength_um"]))
        amplitudes.append(float(run["amplitude"]))
    return {"scale": list(scales), "predicted_um": predicted, "measured_um": measured, "amplitude": amplitudes}


def goodness_of_fit(sweep: dict[str, list[float]]) -> dict[str, float]:
    pred = np.asarray(sweep["predicted_um"], dtype=float)
    meas = np.asarray(sweep["measured_um"], dtype=float)
    pearson = float(np.corrcoef(pred, meas)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    # Two-sample KS between the measured wavelengths and the linear predictions.
    ks = float(stats.ks_2samp(meas, pred).statistic)
    return {"pearson_r": pearson, "rmse_um": rmse, "ks_statistic": ks}


def ablation_equal_diffusion() -> dict[str, float]:
    """Negative control: Du = Dv removes the long-range inhibition that Turing needs."""
    run = simulate_schnakenberg(DU_UM2, DU_UM2, steps=N_STEPS)
    return {"amplitude": float(run["amplitude"]), "wavelength_um": float(run["wavelength_um"])}


def domain_series() -> list[dict[str, float]]:
    """Same kinetics on the scale ladder. 10 µm cannot hold k_max; 480 µm holds eight waves."""
    rows = []
    for length in (PREDECESSOR_UM, LAMBDA_UM, 120.0, PATCH_UM, SUCCESSOR_UM):
        n = 48 if length <= LAMBDA_UM else N_METRIC
        steps = 280 if length <= PREDECESSOR_UM else N_STEPS
        run = simulate_schnakenberg(DU_UM2, DV_UM2, length_um=length, n=n, steps=steps)
        rows.append({
            "length_um": float(length),
            "n_wavelengths_integer": int(length // LAMBDA_UM),
            "amplitude": float(run["amplitude"]),
            "wavelength_um": float(run["wavelength_um"]),
        })
    return rows


def spot_spacings(field: np.ndarray, length_um: float) -> np.ndarray:
    """Nearest-neighbour distances (µm) of activator peaks on the periodic patch."""
    n = field.shape[0]
    threshold = float(np.mean(field) + 0.45 * np.std(field))
    local_max = field == ndi.maximum_filter(field, size=5)
    peaks = local_max & (field > threshold)
    ys, xs = np.nonzero(peaks)
    if len(xs) < 4:
        return np.array([], dtype=float)
    dx = length_um / n
    pts = np.stack([ys.astype(float), xs.astype(float)], axis=1) * dx
    nearest = np.empty(len(pts), dtype=float)
    for i, p in enumerate(pts):
        delta = pts - p
        delta -= length_um * np.round(delta / length_um)
        dist = np.hypot(delta[:, 0], delta[:, 1])
        dist[i] = np.inf
        nearest[i] = float(dist.min())
    return nearest


def segment_specimen(path: Path = SPECIMEN) -> dict[str, float]:
    """Foreground of the USNM plate: three living tests on a black ground.

    Pass 1 uses one union mask. Per-disc connected components are a Pass 2 item.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask = gray > 0.18
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perimeter = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perimeter ** 2), 0.0, 1.0))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        aspect = 1.0
    else:
        height = float(ys.max() - ys.min() + 1)
        width = float(xs.max() - xs.min() + 1)
        aspect = width / height
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coherency_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coherency_map[mask])) if area > 20 else float(np.mean(coherency_map))
    fg = np.clip(rgb[mask], 0.0, 1.0)
    mean_rgb = fg.mean(axis=0) if area > 20 else np.array([0.84, 0.49, 0.38])
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "aspect_ratio": float(aspect),
        "coherency": coherency,
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "width_px": float(gray.shape[1]),
        "height_px": float(gray.shape[0]),
    }


def specimen_sha256(path: Path = SPECIMEN) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _disc_frame(field: np.ndarray, pigment_rgb: tuple[float, float, float], side: int = GIF_SIDE) -> Image.Image:
    """Render one periodic patch as a disc of echinochrome on a black museum ground."""
    zoom = side / field.shape[0]
    up = ndi.zoom(field, zoom, order=3)
    up = up[:side, :side]
    span = float(up.max() - up.min()) + 1e-9
    shade = (up - float(up.min())) / span
    yy, xx = np.ogrid[:side, :side]
    center = (side - 1) / 2.0
    radius = 0.43 * side
    mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    canvas = np.zeros((side, side, 3), dtype=np.float64)
    canvas[:] = 0.02
    tone = np.array(pigment_rgb, dtype=np.float64)
    # High activator is the darker ocellus; the ground is the plate's mean pigment.
    darken = 0.55 + 0.55 * (1.0 - shade)
    for channel in range(3):
        canvas[:, :, channel] = np.where(mask, np.clip(tone[channel] * darken, 0.0, 1.0), canvas[:, :, channel])
    image = Image.fromarray(np.clip(canvas * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    # 60 µm bar in the outer margin (cropped out of the fitness window's core).
    bar = max(8, int(round(side * (LAMBDA_UM / PATCH_UM) * (2.0 * radius / side))))
    x0, y0 = 28, side - 36
    draw.line([(x0, y0), (x0 + bar, y0)], fill=(245, 245, 245), width=3)
    draw.text((x0, y0 + 6), "60 um", fill=(245, 245, 245))
    draw.text((side - 210, side - 28), "E. parma  0 Ma  patch", fill=(220, 220, 220))
    return image


def write_growth_gif(snapshots: list[np.ndarray], pigment_rgb: tuple[float, float, float], path: Path = GROWTH_GIF) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(snapshots) < N_GIF_FRAMES:
        # Repeat the last equilibrated field rather than invent frames.
        snapshots = list(snapshots) + [snapshots[-1]] * (N_GIF_FRAMES - len(snapshots))
    frames = [_disc_frame(snap, pigment_rgb) for snap in snapshots[:N_GIF_FRAMES]]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        disposal=2,
    )


def write_comparison_png(
    specimen_path: Path,
    field: np.ndarray,
    sweep: dict[str, list[float]],
    fit: dict[str, float],
    ablation_amp: float,
    turing_amp: float,
    domains: list[dict[str, float]],
    spacings: np.ndarray,
    segment: dict[str, float],
    path: Path = COMPARISON_PNG,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    fig.suptitle(
        "Echinarachnius parma — Schnakenberg Turing wavelength at 60 µm (Holocene, 0 Ma)",
        fontsize=12,
    )

    photo = np.asarray(Image.open(specimen_path).convert("RGB"))
    axes[0, 0].imshow(photo)
    axes[0, 0].set_title("USNM E39365 (CC0)\nliving tests, 3 cm bar")
    axes[0, 0].set_axis_off()

    show = field - field.min()
    show = show / (show.max() + 1e-9)
    axes[0, 1].imshow(show, cmap="YlOrBr", origin="lower", extent=[0, PATCH_UM, 0, PATCH_UM])
    axes[0, 1].set_title(f"Equilibrated patch\nL = {PATCH_UM:.0f} µm")
    axes[0, 1].set_xlabel("µm")
    axes[0, 1].set_ylabel("µm")

    k = np.linspace(0.0, 0.35, 400)
    sigma = dispersion_sigma(k ** 2, DU_UM2, DV_UM2)
    axes[0, 2].plot(k, sigma, color="#8c4a2f", lw=2.0)
    k_max = float(k[int(np.argmax(sigma))])
    axes[0, 2].axvline(k_max, color="#333333", ls="--", lw=1.0, label=f"k_max → λ={2*np.pi/k_max:.1f} µm")
    axes[0, 2].axhline(0.0, color="#888888", lw=0.6)
    axes[0, 2].set_title("Linear dispersion σ(k)")
    axes[0, 2].set_xlabel("k (1/µm)")
    axes[0, 2].set_ylabel("growth rate")
    axes[0, 2].legend(fontsize=8, frameon=False)

    pred = np.asarray(sweep["predicted_um"])
    meas = np.asarray(sweep["measured_um"])
    axes[1, 0].scatter(pred, meas, c="#8c4a2f", s=36, zorder=3)
    lim = [min(pred.min(), meas.min()) * 0.9, max(pred.max(), meas.max()) * 1.05]
    axes[1, 0].plot(lim, lim, color="#888888", lw=1.0)
    axes[1, 0].set_xlim(lim)
    axes[1, 0].set_ylim(lim)
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].set_title(f"Wavelength sweep\nPearson r = {fit['pearson_r']:.4f}")
    axes[1, 0].set_xlabel("linear λ (µm)")
    axes[1, 0].set_ylabel("equilibrated FFT λ (µm)")

    labels = ["Du=Dv\nablation", "Turing\nratio 12"] + [f"L={int(row['length_um'])}" for row in domains]
    values = [ablation_amp, turing_amp] + [row["amplitude"] for row in domains]
    colors = ["#9aa0a6", "#c47a3a"] + ["#6b4f3a"] * len(domains)
    axes[1, 1].bar(range(len(values)), values, color=colors)
    axes[1, 1].set_xticks(range(len(labels)))
    axes[1, 1].set_xticklabels(labels, fontsize=7)
    axes[1, 1].set_ylabel("activator std")
    axes[1, 1].set_title("Ablation and scale ladder\n10 µm predecessor quenches")

    if len(spacings):
        axes[1, 2].hist(spacings, bins=12, color="#c47a3a", edgecolor="white")
        axes[1, 2].axvline(float(np.mean(spacings)), color="#222222", ls="--", lw=1.0)
    axes[1, 2].set_title(
        f"Spot nearest neighbours\nunion circularity {segment['circularity']:.3f}"
    )
    axes[1, 2].set_xlabel("spacing (µm)")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _measurement(
    name: str,
    value: float,
    unit: str,
    method: str,
    reference: float | None = None,
    source: str | None = None,
    note: str | None = None,
    derived_from: str = "script",
) -> dict:
    row = {
        "name": name,
        "value": value,
        "unit": unit,
        "method": method,
        "derived_from": derived_from,
    }
    if reference is not None:
        row["reference_value"] = reference
    if source:
        row["reference_source"] = source
    if note:
        row["note"] = note
    return row


def build_measurements(bundle: dict) -> list[dict]:
    """Quantitative rows. Specimen metrics are measured on the photograph, not copied from the reference."""
    fit = bundle["fit"]
    hero = bundle["hero"]
    segment = bundle["segment"]
    spacings = np.asarray(bundle["spacings"], dtype=float)
    domains = bundle["domains"]
    pred10 = next(row for row in domains if row["length_um"] == PREDECESSOR_UM)
    spacing_mean = float(np.mean(spacings)) if len(spacings) else float("nan")
    spacing_cv = float(np.std(spacings) / spacing_mean) if len(spacings) and spacing_mean else float("nan")
    rows = [
        _measurement(
            "linear dispersion peak wavelength",
            round(bundle["lambda_linear_um"], 4),
            "µm",
            "argmax of σ(k), the larger eigenvalue of J − k²D, at Du, Dv",
            reference=LAMBDA_UM,
            source="slot scale 60 µm; Turing 1952 dispersion; Schnakenberg 1979 Jacobian",
            note="linear peak is deliberately a few micrometres off 60 so the equilibrated lattice, not the onset mode, carries the slot scale",
        ),
        _measurement(
            "equilibrated activator FFT wavelength",
            round(float(hero["wavelength_um"]), 4),
            "µm",
            "parabolic interpolation of the radial power spectrum of u on the 480 µm periodic patch",
            reference=LAMBDA_UM,
            source="detail scale 6e-05 m assigned to this Holocene epithelial patch",
            note=f"amplitude std(u) = {hero['amplitude']:.4f}; seed {SEED}",
        ),
        _measurement(
            "dispersion-peak versus equilibrated wavelength Pearson r",
            round(fit["pearson_r"], 6),
            "Pearson r",
            "corr(λ_linear(s), λ_FFT(s)) across diffusivity scales s in {0.35, 0.55, 0.80, 1, 1.4, 2, 3}",
            reference=0.98,
            source="goodness-of-fit gate for the Turing wavenumber law",
            note=f"RMSE = {fit['rmse_um']:.4f} µm; KS two-sample statistic = {fit['ks_statistic']:.4f}",
        ),
        _measurement(
            "wavelength sweep RMSE",
            round(fit["rmse_um"], 4),
            "µm",
            "root mean square of λ_FFT − λ_linear across the diffusivity sweep",
            reference=0.0,
            source="zero residual would mean the nonlinear lattice sits on the linear peak",
            note="a nonzero RMSE is the nonlinear shift above onset, not a missing sweep",
        ),
        _measurement(
            "wavelength sweep KS statistic",
            round(fit["ks_statistic"], 4),
            "KS D",
            "two-sample Kolmogorov-Smirnov statistic between linear and FFT wavelength samples",
            reference=0.0,
            source="scipy.stats.ks_2samp",
        ),
        _measurement(
            "equal-diffusion ablation amplitude",
            round(bundle["ablation_amplitude"], 6),
            "std(u)",
            "same integrator with Du = Dv = DU_UM2; Turing trace gap is negative",
            reference=0.0,
            source="Turing 1952: equal diffusivities cannot destabilise a stable Jacobian",
            note=f"Turing amplitude at ratio 12 is {hero['amplitude']:.4f}",
        ),
        _measurement(
            "10 µm predecessor-scale amplitude",
            round(pred10["amplitude"], 6),
            "std(u)",
            "same Du, Dv on a periodic patch of length 10 µm (below one wavelength)",
            reference=0.0,
            source="scale ladder at 0 Ma: the 10 µm tier is the predecessor of this 60 µm detail scale",
            note="chronoseries is 1/1, so the predecessor is a scale, not an older species",
        ),
        _measurement(
            "specimen USNM E39365 foreground fraction",
            round(segment["foreground_fraction"], 4),
            "area fraction",
            "pixels with luminance > 0.18 on the CC0 plate, opened and hole-filled",
            reference=0.70,
            source="USNM E39365 aboral photographs; black museum ground, three tests",
            note="reference 0.70 is a single-test fill prior; the plate holds three tests, so the fraction is lower",
            derived_from="specimen_image",
        ),
        _measurement(
            "specimen union-mask circularity",
            round(segment["circularity"], 4),
            "4πA/P²",
            "circularity of the union of all bright tests, not of one individual",
            reference=1.0,
            source="a single disc would be near 1; three separate tests on one plate are not",
            note=f"bounding aspect ratio width/height = {segment['aspect_ratio']:.4f}",
            derived_from="specimen_image",
        ),
        _measurement(
            "specimen structure-tensor coherency",
            round(segment["coherency"], 4),
            "coherency",
            "mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor inside the foreground",
            reference=0.25,
            source="spine canopy is locally oriented, so coherency exceeds an isotropic noise prior of 0.25",
            derived_from="specimen_image",
        ),
        _measurement(
            "ocellus nearest-neighbour spacing",
            round(spacing_mean, 4),
            "µm",
            "periodic nearest-neighbour distance of activator peaks above mean + 0.45 std",
            reference=LAMBDA_UM,
            source="FFT wavelength is the row period; nearest neighbours of a spot lattice need not equal it",
            note=f"spacing CV = {spacing_cv:.4f}; n_peaks = {len(spacings)}",
        ),
    ]
    return rows


def run_bundle(n_snapshots: int = N_GIF_FRAMES) -> dict:
    hero = simulate_schnakenberg(DU_UM2, DV_UM2, n_snapshots=n_snapshots)
    sweep = wavelength_sweep()
    fit = goodness_of_fit(sweep)
    ablation = ablation_equal_diffusion()
    domains = domain_series()
    spacings = spot_spacings(hero["u"], PATCH_UM)
    segment = segment_specimen()
    lam_lin, sigma_max = linear_peak_wavelength(DU_UM2, DV_UM2)
    bundle = {
        "hero": {k: hero[k] for k in ("u", "v", "amplitude", "wavelength_um", "snapshots")},
        "sweep": sweep,
        "fit": fit,
        "ablation_amplitude": ablation["amplitude"],
        "domains": domains,
        "spacings": spacings,
        "segment": segment,
        "lambda_linear_um": lam_lin,
        "sigma_max": sigma_max,
        "trace_gap": turing_trace_gap(DU_UM2, DV_UM2),
        "ablation_trace_gap": turing_trace_gap(DU_UM2, DU_UM2),
    }
    return bundle


def persist_results(bundle: dict) -> list[dict]:
    measurements = build_measurements(bundle)
    payload = {
        "age_Ma": AGE_MA,
        "lambda_um_slot": LAMBDA_UM,
        "patch_um": PATCH_UM,
        "Du_um2": DU_UM2,
        "Dv_um2": DV_UM2,
        "seed": SEED,
        "lambda_linear_um": bundle["lambda_linear_um"],
        "lambda_fft_um": float(bundle["hero"]["wavelength_um"]),
        "amplitude": float(bundle["hero"]["amplitude"]),
        "fit": bundle["fit"],
        "ablation_amplitude": bundle["ablation_amplitude"],
        "trace_gap": bundle["trace_gap"],
        "ablation_trace_gap": bundle["ablation_trace_gap"],
        "domains": bundle["domains"],
        "segment": bundle["segment"],
        "spacing_mean_um": float(np.mean(bundle["spacings"])) if len(bundle["spacings"]) else None,
        "measurements": measurements,
    }
    DATASET.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return measurements


def measure_morphoeval() -> list[dict]:
    """MorphoEval hook. Reuses the last script run when its cache is present."""
    if RESULTS_JSON.exists():
        payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        if payload.get("measurements") and abs(float(payload.get("Du_um2", 0.0)) - DU_UM2) < 1e-9:
            return payload["measurements"]
    bundle = run_bundle(n_snapshots=0)
    return persist_results(bundle)


def self_test() -> None:
    J = jacobian()
    assert abs(J["u_star"] - 1.0) < 1e-12
    assert abs(J["v_star"] - 0.9) < 1e-12
    assert J["trace"] < 0.0 and J["det"] > 0.0, "homogeneous state must be linearly stable"
    assert turing_trace_gap(DU_UM2, DV_UM2) > 0.0
    assert turing_trace_gap(DU_UM2, DU_UM2) < 0.0, "equal diffusion must close the Turing gap"
    assert abs(DV_UM2 / DU_UM2 - DIFFUSION_RATIO) < 1e-9
    assert specimen_sha256() == SPECIMEN_SHA256
    lean = LEAN_PATH.read_text(encoding="utf-8")
    assert "sorry" not in lean
    assert lean.count("theorem ") >= 3
    assert "MathOfBiology.Loop_3214" in lean

    hero = simulate_schnakenberg(DU_UM2, DV_UM2, n_snapshots=0)
    assert hero["amplitude"] > 0.30, hero["amplitude"]
    assert abs(hero["wavelength_um"] - LAMBDA_UM) < 6.0, hero["wavelength_um"]
    again = simulate_schnakenberg(DU_UM2, DV_UM2, n_snapshots=0)
    assert abs(again["wavelength_um"] - hero["wavelength_um"]) < 1e-9

    ablation = ablation_equal_diffusion()
    assert ablation["amplitude"] < 0.02, ablation["amplitude"]

    short = simulate_schnakenberg(
        DU_UM2, DV_UM2, length_um=PREDECESSOR_UM, n=48, steps=220, n_snapshots=0
    )
    assert short["amplitude"] < 0.02, short["amplitude"]

    sweep = wavelength_sweep()
    fit = goodness_of_fit(sweep)
    assert fit["pearson_r"] > 0.98, fit
    # Monotone in the scale factor: larger diffusivity, longer wavelength.
    measured = sweep["measured_um"]
    assert measured[0] < measured[-1], measured

    segment = segment_specimen()
    assert 0.15 < segment["foreground_fraction"] < 0.85
    assert segment["circularity"] < 0.85, "three tests on one plate are not one circle"
    assert segment["circularity"] != 1.0

    print("[self-test] jacobian stable; Turing gap open at ratio 12 and closed at Du=Dv")
    print(f"[self-test] FFT λ = {hero['wavelength_um']:.3f} µm (slot {LAMBDA_UM:.0f} µm), amp = {hero['amplitude']:.3f}")
    print(f"[self-test] sweep Pearson r = {fit['pearson_r']:.4f}, RMSE = {fit['rmse_um']:.3f} µm, KS = {fit['ks_statistic']:.3f}")
    print(f"[self-test] ablation amp = {ablation['amplitude']:.3e}; 10 µm amp = {short['amplitude']:.3e}")
    print(f"[self-test] specimen foreground {segment['foreground_fraction']:.3f}, circularity {segment['circularity']:.3f}")
    print("[self-test] PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="Run numerical checks and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    bundle = run_bundle(n_snapshots=N_GIF_FRAMES)
    measurements = persist_results(bundle)
    pigment = (
        bundle["segment"]["mean_r"],
        bundle["segment"]["mean_g"],
        bundle["segment"]["mean_b"],
    )
    write_growth_gif(bundle["hero"]["snapshots"], pigment)
    write_comparison_png(
        SPECIMEN,
        bundle["hero"]["u"],
        bundle["sweep"],
        bundle["fit"],
        bundle["ablation_amplitude"],
        float(bundle["hero"]["amplitude"]),
        bundle["domains"],
        bundle["spacings"],
        bundle["segment"],
    )
    with Image.open(GROWTH_GIF) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
        width, height = gif.size
    print(f"[render] λ_linear = {bundle['lambda_linear_um']:.3f} µm")
    print(f"[render] λ_FFT = {bundle['hero']['wavelength_um']:.3f} µm  amp = {bundle['hero']['amplitude']:.3f}")
    print(
        f"[render] Pearson r = {bundle['fit']['pearson_r']:.4f}  "
        f"RMSE = {bundle['fit']['rmse_um']:.3f} µm  KS = {bundle['fit']['ks_statistic']:.3f}"
    )
    print(f"[render] ablation amp = {bundle['ablation_amplitude']:.3e}  trace gap = {bundle['trace_gap']:.3f}")
    print(f"[render] GIF {width}x{height} frames={n_frames} -> {GROWTH_GIF.relative_to(ROOT)}")
    print(f"[render] PNG -> {COMPARISON_PNG.relative_to(ROOT)}")
    print(f"[render] measurements = {len(measurements)}")
    if n_frames < 10 or min(width, height) < 480:
        print("[render] GIF failed the frame/resolution gate", file=sys.stderr)
        return 1
    if bundle["fit"]["pearson_r"] <= 0.98:
        print("[render] Pearson r gate failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
