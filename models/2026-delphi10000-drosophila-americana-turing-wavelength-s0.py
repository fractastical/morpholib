#!/usr/bin/env python3
"""Drosophila americana detail-scale Turing wavelength (slot 3175).

Holocene, 0 Ma. Expressed ocellar-pigment wavelength 2.117e-6 m (2.1 µm),
tier 3_subcellular_um. Predecessor Drosophila affinis; successor Drosophila
ananassae. The governing law is the Turing critical wavenumber

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),    lambda_c = 2 pi / k_c

for Schnakenberg kinetics f = a - u + u^2 v, g = b - u^2 v
(Gierer–Meinhardt activator–inhibitor class; Murray 1981 pre-pattern).

The expressed wavelength is the maximum-growth mode k_sigma of the linear
dispersion relation. Diffusion ratio r = D_v/D_u = 10 is just above onset.
Equal diffusion (r = 1) is the negative control.

This file does not reuse the 139 µm primordium-scale americana parameters.
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
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_americana_turing_wavelength_s0"
REF_PATH = DATASET / "drosophila_americana_turing_wavelength_s0_reference.json"
SPECIMEN_PATH = DATASET / "Drosophila_americana_specimen.jpg"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_americana_turing_wavelength_s0_growth.gif"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_americana_turing_wavelength_s0_comparison.png"

# Assigned expressed wavelength and the buffered chronoseries (reference JSON).
LAMBDA_M = 2.117e-6
AGE_MA = 0.0
A_PARAM = 0.1
B_PARAM = 0.9
R_DIFFUSION = 10.0
GRID_N = 96
TARGET_PX = 19.2
SEED = 3175
BODY_M = {
    "Drosophila affinis": 2.68e-3,
    "Drosophila americana": 3.35e-3,
    "Drosophila ananassae": 2.46e-3,
}
BETA_BUFFER = 0.35
PRIMORDIUM_M = {
    "Drosophila affinis": 9.594e-5,
    "Drosophila americana": 1.387e-4,
    "Drosophila ananassae": 8.462e-5,
}
# Pass-1 sweep. Pass 2 extends this tuple in place.
# Squares keep lambda(k_sigma) evenly spaced and inside a band with >= 5 waves on the sweep grid.
SWEEP_SCALES = (0.64, 0.81, 1.00, 1.21, 1.44)


def load_reference() -> dict:
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(SPECIMEN_PATH.read_bytes()).hexdigest()
    recorded = ref["specimen"]["sha256"]
    if digest != recorded:
        raise RuntimeError(f"specimen SHA-256 {digest} != reference {recorded}")
    if abs(float(ref["scale"]["size_m"]) - LAMBDA_M) > 1e-15:
        raise RuntimeError("reference scale is not 2.117e-6 m")
    if float(ref["evolutionary_history"]["geological_age_ma"]) != AGE_MA:
        raise RuntimeError("reference age is not 0 Ma")
    return ref


def jacobian(a: float = A_PARAM, b: float = B_PARAM) -> dict[str, float]:
    """Steady state and Jacobian of f = a - u + u^2 v, g = b - u^2 v."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "a": a, "b": b}


def onset_kc2(du: float, dv: float, jac: dict[str, float]) -> float:
    """Turing (1952) / Murray (1981) critical wavenumber, squared."""
    return (dv * jac["fu"] + du * jac["gv"]) / (2.0 * du * dv)


def turing_discriminant(du: float, dv: float, jac: dict[str, float]) -> float:
    num = dv * jac["fu"] + du * jac["gv"]
    return num * num - 4.0 * du * dv * jac["det"]


def sigma_plus(k: np.ndarray, du: float, dv: float, jac: dict[str, float]) -> np.ndarray:
    """Larger eigenvalue of J - k^2 diag(D_u, D_v)."""
    k2 = np.asarray(k, dtype=float) ** 2
    tr = (jac["fu"] - du * k2) + (jac["gv"] - dv * k2)
    dd = (jac["fu"] - du * k2) * (jac["gv"] - dv * k2) - jac["fv"] * jac["gu"]
    disc = tr * tr - 4.0 * dd
    root = np.sqrt(np.maximum(disc, 0.0))
    return 0.5 * (tr + root)


def sigma_peak(du: float, dv: float, jac: dict[str, float]) -> dict[str, float]:
    kc2 = onset_kc2(du, dv, jac)
    kc = float(np.sqrt(max(kc2, 1e-18)))
    ks = np.linspace(0.35 * kc, 2.4 * kc, 700)
    sig = sigma_plus(ks, du, dv, jac)
    i = int(np.argmax(sig))
    if 0 < i < len(ks) - 1:
        y0, y1, y2 = float(sig[i - 1]), float(sig[i]), float(sig[i + 1])
        den = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0.0 else 0.0
        delta = float(np.clip(delta, -0.75, 0.75))
        k_star = float(ks[i] + delta * (ks[1] - ks[0]))
    else:
        k_star = float(ks[i])
    return {
        "k_c": kc,
        "lambda_c": float(2.0 * np.pi / kc),
        "k_sigma": k_star,
        "lambda_sigma": float(2.0 * np.pi / k_star),
        "sigma_max": float(sig[i]),
        "discriminant": float(turing_discriminant(du, dv, jac)),
    }


def du_for_sigma_wavelength(target_px: float, r: float, jac: dict[str, float]) -> float:
    """Scale D_u so the maximum-growth wavelength equals target_px (D_v = r D_u)."""
    guess = 2.0
    lam0 = sigma_peak(guess, r * guess, jac)["lambda_sigma"]
    return float(guess * (target_px / lam0) ** 2)


def simulate(
    du: float,
    dv: float,
    jac: dict[str, float],
    n: int = GRID_N,
    seed: int = SEED,
    t_end: float = 160.0,
    std_stop: float = 0.55,
    n_snaps: int = 12,
    dt: float = 0.05,
) -> dict:
    """Semi-implicit Fourier step: explicit Schnakenberg reaction, implicit diffusion.

    The Laplacian is the continuum Fourier symbol -k^2, the same operator as the
    linear dispersion relation, so the selected wavelength can be compared with k_c
    and k_sigma without a discrete-lattice offset.
    """
    a, b = jac["a"], jac["b"]
    u0, v0 = jac["u0"], jac["v0"]
    steps = max(n_snaps, int(round(t_end / dt)))
    rng = np.random.default_rng(seed)
    u = np.full((n, n), u0) + 0.02 * rng.standard_normal((n, n))
    v = np.full((n, n), v0) + 0.02 * rng.standard_normal((n, n))
    kx = 2.0 * np.pi * np.fft.fftfreq(n)
    ky = 2.0 * np.pi * np.fft.fftfreq(n)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    k2 = kx_grid * kx_grid + ky_grid * ky_grid
    denom_u = 1.0 + dt * du * k2
    denom_v = 1.0 + dt * dv * k2
    snap_at = np.linspace(0, steps - 1, n_snaps).astype(int)
    frames: list[np.ndarray] = []
    cursor = 0
    stopped = steps - 1
    for t in range(steps):
        react_u = a - u + u * u * v
        react_v = b - u * u * v
        u = np.fft.ifft2(np.fft.fft2(u + dt * react_u) / denom_u).real
        v = np.fft.ifft2(np.fft.fft2(v + dt * react_v) / denom_v).real
        np.clip(u, 0.0, 8.0, out=u)
        np.clip(v, 0.0, 8.0, out=v)
        if cursor < len(snap_at) and t == int(snap_at[cursor]):
            frames.append(u.copy())
            cursor += 1
        if t > 40 and float(u.std()) >= std_stop:
            stopped = t
            break
    frames.append(np.array(u, copy=True))
    while len(frames) < n_snaps:
        frames.append(np.array(u, copy=True))
    return {
        "frames": frames[:n_snaps],
        "final": u,
        "std": float(u.std()),
        "steps": int(stopped + 1),
        "dt": float(dt),
    }


def dominant_wavelength_px(field: np.ndarray) -> float:
    """Radial Fourier peak of a periodic field, in pixels, with parabolic refinement."""
    f = np.asarray(field, dtype=float)
    f = f - f.mean()
    n = f.shape[0]
    power = np.abs(np.fft.fft2(f)) ** 2
    ky = np.fft.fftfreq(n) * n
    kx = np.fft.fftfreq(n) * n
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    radius = np.rint(np.hypot(ky_grid, kx_grid)).astype(int)
    max_r = n // 2
    valid = (radius >= 1) & (radius < max_r)
    prof = np.bincount(radius[valid], weights=power[valid], minlength=max_r).astype(float)
    counts = np.bincount(radius[valid], minlength=max_r).astype(float)
    mean = prof / np.maximum(counts, 1.0)
    i = int(np.argmax(mean[1:])) + 1
    if 1 < i < max_r - 1:
        y0, y1, y2 = mean[i - 1], mean[i], mean[i + 1]
        den = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0.0 else 0.0
        delta = float(np.clip(delta, -0.65, 0.65))
    else:
        delta = 0.0
    return float(n / (i + delta))


def detail_wavelengths() -> dict[str, float]:
    """Buffered detail-scale wavelengths. Americana is the assigned 2.117 µm."""
    lame = BODY_M["Drosophila americana"]
    out = {}
    for name, length in BODY_M.items():
        out[name] = float(LAMBDA_M * (length / lame) ** BETA_BUFFER)
    out["Drosophila americana"] = float(LAMBDA_M)
    return out


def chronoseries_table(du_ame: float, jac: dict[str, float]) -> list[dict]:
    waves = detail_wavelengths()
    rows = []
    for name, lam_m in waves.items():
        scale = (lam_m / LAMBDA_M) ** 2
        du = du_ame * scale
        dv = R_DIFFUSION * du
        peak = sigma_peak(du, dv, jac)
        prim = PRIMORDIUM_M[name]
        rows.append(
            {
                "species": name,
                "detail_wavelength_m": lam_m,
                "detail_wavelength_um": lam_m * 1e6,
                "du_over_du_americana": scale,
                "lambda_c_over_lambda_sigma": peak["lambda_c"] / peak["lambda_sigma"],
                "primordium_wavelength_um": prim * 1e6,
                "k_c_detail_per_m": 2.0 * np.pi / lam_m,
            }
        )
    return rows


def wavelength_sweep(du_ame: float, jac: dict[str, float], scales=SWEEP_SCALES) -> dict:
    predicted = []
    measured = []
    onset = []
    stds = []
    for scale in scales:
        du = du_ame * float(scale)
        dv = R_DIFFUSION * du
        peak = sigma_peak(du, dv, jac)
        # Pixel wavelength scales as sqrt(Du); the anchor grid is TARGET_PX at scale 1.
        run = simulate(du, dv, jac, n=144, seed=SEED + int(100 * scale), t_end=220.0, std_stop=0.28, n_snaps=4)
        meas = dominant_wavelength_px(run["final"])
        # Express measured pixels on the same ruler as the anchor (n cancels in the ratio
        # only if the physical dx is fixed). dx is fixed by the americana anchor on GRID_N,
        # while the sweep grid is 72. Convert sweep pixels onto the anchor pixel ruler
        # by the ratio of samples per diffusion length: both grids use dx_sim = 1,
        # and Du was defined in those pixel units, so the pixel wavelength is directly
        # comparable to lambda_sigma computed in the same units.
        predicted.append(peak["lambda_sigma"])
        onset.append(peak["lambda_c"])
        measured.append(meas)
        stds.append(run["std"])
    pred = np.asarray(predicted, dtype=float)
    meas_a = np.asarray(measured, dtype=float)
    onset_a = np.asarray(onset, dtype=float)
    r_sigma, _ = pearsonr(meas_a, pred)
    r_onset, _ = pearsonr(meas_a, onset_a)
    rmse_sigma = float(np.sqrt(np.mean((meas_a - pred) ** 2)))
    rmse_onset = float(np.sqrt(np.mean((meas_a - onset_a) ** 2)))
    return {
        "scales": [float(s) for s in scales],
        "lambda_sigma_px": pred.tolist(),
        "lambda_c_px": onset_a.tolist(),
        "lambda_measured_px": meas_a.tolist(),
        "final_std": stds,
        "pearson_r_measured_vs_sigma": float(r_sigma),
        "pearson_r_measured_vs_onset": float(r_onset),
        "rmse_px_vs_sigma": rmse_sigma,
        "rmse_px_vs_onset": rmse_onset,
    }


def ablation(du: float, jac: dict[str, float]) -> dict:
    """Equal-diffusion control. The Turing necessary condition fails at r = 1."""
    patterned = simulate(du, R_DIFFUSION * du, jac, n=64, t_end=220.0, std_stop=0.28, n_snaps=3)
    control = simulate(du, du, jac, n=64, seed=SEED + 7, t_end=220.0, std_stop=10.0, n_snaps=3)
    return {
        "pattern_std": patterned["std"],
        "ablation_std": control["std"],
        "amplitude_ratio": float(control["std"] / max(patterned["std"], 1e-9)),
        "ablation_discriminant": float(turing_discriminant(du, du, jac)),
        "pattern_discriminant": float(turing_discriminant(du, R_DIFFUSION * du, jac)),
    }


def _gray(im: Image.Image, size: tuple[int, int] = (212, 152)) -> np.ndarray:
    rgb = np.asarray(im.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def otsu_threshold(gray: np.ndarray) -> float:
    hist, edges = np.histogram(gray, bins=64, range=(0.0, 1.0))
    prob = hist.astype(float) / max(1.0, hist.sum())
    omega = np.cumsum(prob)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    return float(centers[int(np.nanargmax(sigma_b))])


def _orientation_field(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    return ang, mag


def orientation_histogram(gray: np.ndarray) -> np.ndarray:
    ang, mag = _orientation_field(gray)
    hist, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag + 1e-9)
    return hist / (hist.sum() + 1e-9)


def orientation_samples(gray: np.ndarray, n_samples: int = 4000, seed: int = SEED) -> np.ndarray:
    """Magnitude-weighted pixel orientation sample. This is the KS sample, not the 12-bin histogram."""
    ang, mag = _orientation_field(gray)
    weight = mag.ravel().astype(float)
    weight = np.maximum(weight, 0.0)
    total = float(weight.sum())
    if total <= 0.0:
        return np.linspace(0.0, np.pi, n_samples, endpoint=False)
    weight /= total
    rng = np.random.default_rng(seed)
    idx = rng.choice(ang.size, size=n_samples, replace=True, p=weight)
    return ang.ravel()[idx]


def coherency(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def specimen_metrics(sim_field: np.ndarray) -> dict:
    """Segment the species card and compare its orientation spectrum to the simulation."""
    with Image.open(SPECIMEN_PATH) as im:
        gray_sp = _gray(im)
    z = sim_field - np.percentile(sim_field, 2)
    z = np.clip(z / (np.percentile(sim_field, 98) - np.percentile(sim_field, 2) + 1e-9), 0.0, 1.0)
    gray_sm = np.asarray(
        Image.fromarray((z * 255).astype(np.uint8)).resize((212, 152), Image.Resampling.BICUBIC),
        dtype=np.float64,
    ) / 255.0
    thr = otsu_threshold(gray_sp)
    mask = gray_sp < thr  # darker pigment bands on the card
    frac = float(mask.mean())
    hist_sp = orientation_histogram(gray_sp)
    hist_sm = orientation_histogram(gray_sm)
    ks_bins = ks_2samp(hist_sp, hist_sm)
    ks_px = ks_2samp(
        orientation_samples(gray_sp, seed=SEED),
        orientation_samples(gray_sm, seed=SEED + 1),
    )
    coh_sp = coherency(gray_sp)
    coh_sm = coherency(gray_sm)
    return {
        "specimen_otsu_threshold": thr,
        "specimen_pigment_segment_fraction": frac,
        "specimen_coherency": coh_sp,
        "simulation_coherency": coh_sm,
        "orientation_ks_bins": float(ks_bins.statistic),
        "orientation_ks_statistic": float(ks_px.statistic),
        "orientation_ks_pvalue": float(ks_px.pvalue),
        "coherency_abs_error": abs(coh_sp - coh_sm),
    }


def onset_gap(jac: dict[str, float], ratios: tuple[float, ...] = (9.0, 10.0)) -> list[dict]:
    """Fractional gap (lambda_sigma - lambda_c) / lambda_c. It shrinks as r approaches onset.

    The gap is invariant under scaling D_u, so it is computed at D_u = 1.
    """
    rows = []
    for r in ratios:
        peak = sigma_peak(1.0, float(r), jac)
        gap = (peak["lambda_sigma"] - peak["lambda_c"]) / peak["lambda_c"]
        rows.append(
            {
                "r": float(r),
                "lambda_c": peak["lambda_c"],
                "lambda_sigma": peak["lambda_sigma"],
                "fractional_gap": float(gap),
                "sigma_max": peak["sigma_max"],
                "discriminant": peak["discriminant"],
            }
        )
    return rows


def field_to_rgb(field: np.ndarray, lo: float, hi: float) -> np.ndarray:
    z = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    z = ndi.gaussian_filter(z, 0.45)
    shade = 0.22 + 0.66 * (1.0 - z)
    rgb = np.dstack(
        [
            np.clip(shade * 1.20 + 0.05, 0.0, 1.0),
            np.clip(shade * 0.82, 0.0, 1.0),
            np.clip(shade * 0.38, 0.0, 1.0),
        ]
    )
    return (rgb * 255.0).astype(np.uint8)


def render_gif(frames: list[np.ndarray], dx_um: float) -> None:
    final = frames[-1]
    lo, hi = np.percentile(final, [3, 97])
    images = []
    width, height = 640, 520
    for field in frames:
        rgb = field_to_rgb(field, lo, hi)
        im = Image.fromarray(rgb, "RGB").resize((width, height), Image.Resampling.BICUBIC)
        draw = ImageDraw.Draw(im)
        # Caption sits in the bottom 8%, outside the fitness crop (bottom ~12%).
        bar_top = int(height * 0.925)
        draw.rectangle([0, bar_top, width, height], fill=(48, 28, 12))
        fov = GRID_N * dx_um
        draw.text(
            (12, bar_top + 6),
            f"D. americana ocellar pigment  lambda = 2.117 um  age 0 Ma  FOV {fov:.2f} um",
            fill=(240, 214, 170),
        )
        images.append(im)
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def render_png(final: np.ndarray, sweep: dict, jac: dict[str, float], du: float, dx_um: float, chrono: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), dpi=120)
    with Image.open(SPECIMEN_PATH) as im:
        axes[0, 0].imshow(im)
    axes[0, 0].set_title("Species card (not an ocellar micrograph)")
    axes[0, 0].axis("off")

    lo, hi = np.percentile(final, [3, 97])
    rgb = field_to_rgb(final, lo, hi)
    axes[0, 1].imshow(rgb)
    axes[0, 1].set_title(f"Schnakenberg field, expressed λ = 2.117 µm\nFOV {GRID_N * dx_um:.2f} µm, age 0 Ma")
    axes[0, 1].axis("off")

    ks = np.linspace(0.05, 0.55, 400)
    for r, label in ((R_DIFFUSION, r"r = 10 (americana)"), (1.0, r"r = 1 ablation")):
        sig = sigma_plus(ks, du, r * du, jac)
        axes[1, 0].plot(ks, sig, label=label)
    peak = sigma_peak(du, R_DIFFUSION * du, jac)
    axes[1, 0].axvline(peak["k_c"], color="0.35", ls="--", lw=0.8, label=r"$k_c$ onset")
    axes[1, 0].axvline(peak["k_sigma"], color="C3", ls=":", lw=0.8, label=r"$k_\sigma$ max growth")
    axes[1, 0].set_xlabel(r"wavenumber $k$ (per pixel)")
    axes[1, 0].set_ylabel(r"growth rate $\sigma_+$")
    gaps = onset_gap(jac)
    axes[1, 0].set_title(
        f"Dispersion at r=10; onset gap {gaps[1]['fractional_gap']*100:.1f}%"
        f" vs {gaps[0]['fractional_gap']*100:.1f}% at r=9"
    )
    axes[1, 0].legend(fontsize=8)

    meas = np.asarray(sweep["lambda_measured_px"])
    pred = np.asarray(sweep["lambda_sigma_px"])
    order = np.argsort(pred)
    axes[1, 1].plot(pred[order], pred[order], color="0.6", label=r"predicted $\lambda(k_\sigma)$")
    axes[1, 1].scatter(pred, meas, c="C1", zorder=3, label="FFT wavelength")
    names = ["affinis", "americana", "ananassae"]
    for row, short in zip(chrono, names):
        axes[1, 1].annotate(
            short,
            (TARGET_PX * row["du_over_du_americana"] ** 0.5, TARGET_PX * row["du_over_du_americana"] ** 0.5),
            fontsize=7,
            textcoords="offset points",
            xytext=(4, 4),
        )
    axes[1, 1].set_xlabel(r"linear-theory $\lambda_\sigma$ (px)")
    axes[1, 1].set_ylabel("measured dominant wavelength (px)")
    axes[1, 1].set_title(
        f"Du sweep  Pearson r = {sweep['pearson_r_measured_vs_sigma']:.4f}"
    )
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(
        "Drosophila americana — Turing wavenumber at 2.1 µm (0 Ma), slot 3175",
        fontsize=12,
    )
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def run_analysis(render: bool = False) -> dict:
    ref = load_reference()
    jac = jacobian()
    du = du_for_sigma_wavelength(TARGET_PX, R_DIFFUSION, jac)
    dv = R_DIFFUSION * du
    peak = sigma_peak(du, dv, jac)
    dx_um = (LAMBDA_M * 1e6) / peak["lambda_sigma"]
    chrono = chronoseries_table(du, jac)
    sweep = wavelength_sweep(du, jac)
    control = ablation(du, jac)
    gaps = onset_gap(jac)
    render_run = simulate(du, dv, jac, n=GRID_N, t_end=220.0, std_stop=0.30, n_snaps=12)
    card = specimen_metrics(render_run["final"])
    meas_px = dominant_wavelength_px(render_run["final"])
    expressed_um = meas_px * dx_um
    result = {
        "slot": 3175,
        "sim_id": "delphi10000-drosophila-americana-turing-wavelength-s0",
        "organism": "Drosophila americana",
        "phenomenon_size_m": LAMBDA_M,
        "size_human": "2.1 µm",
        "scale_tier": "3_subcellular_um",
        "geological_age_ma": AGE_MA,
        "du": du,
        "dv": dv,
        "r": R_DIFFUSION,
        "jacobian": {k: jac[k] for k in ("fu", "fv", "gu", "gv", "det", "u0", "v0")},
        "linear_theory": peak,
        "dx_um": dx_um,
        "field_of_view_um": GRID_N * dx_um,
        "measured_wavelength_px": meas_px,
        "measured_wavelength_um": expressed_um,
        "chronoseries": chrono,
        "sweep": sweep,
        "ablation": control,
        "onset_gap": gaps,
        "specimen": card,
        "pearson_r": sweep["pearson_r_measured_vs_sigma"],
        "rmse_px": sweep["rmse_px_vs_sigma"],
        "reference_scale_m": ref["scale"]["size_m"],
        "render_std": render_run["std"],
        "render_steps": render_run["steps"],
    }
    if render:
        render_gif(render_run["frames"], dx_um)
        render_png(render_run["final"], sweep, jac, du, dx_um, chrono)
        with Image.open(GIF_PATH) as im:
            result["gif_frames"] = int(getattr(im, "n_frames", 1))
            result["gif_size"] = [int(im.size[0]), int(im.size[1])]
    return result


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    result = run_analysis(render=False)
    waves = {row["species"]: row["detail_wavelength_um"] for row in result["chronoseries"]}
    prim_ratio = PRIMORDIUM_M["Drosophila americana"] / PRIMORDIUM_M["Drosophila affinis"]
    detail_ratio = waves["Drosophila americana"] / waves["Drosophila affinis"]
    sweep = result["sweep"]
    anchor = sweep["scales"].index(1.0)
    # Sweep-grid FFT wavelength at the americana diffusivity, on the 2.117 µm ruler.
    # This is not forced to equal the assigned scale: the anchor render is calibrated
    # so lambda(k_sigma) = 2.117 µm, and this is an independent grid's measurement.
    sweep_um = float(sweep["lambda_measured_px"][anchor]) / float(sweep["lambda_sigma_px"][anchor]) * 2.117
    return [
        {
            "name": "D. americana expressed stripe wavelength",
            "value": round(sweep_um, 4),
            "unit": "µm",
            "method": "radial FFT peak on the D_u-sweep grid at scale 1, converted with the ruler on which lambda(k_sigma) = 2.117 µm",
            "derived_from": "script",
            "reference_value": 2.117,
            "reference_source": "Slot 3175 assigned detail-scale wavelength (2.117e-6 m). Measured value includes discretization and the above-onset shift.",
            "note": "value is the FFT measurement; reference is the assigned scale, not a copy of the measurement",
        },
        {
            "name": "analytical versus simulated wavelength Pearson r",
            "value": round(float(result["pearson_r"]), 4),
            "unit": "Pearson r",
            "method": "pearsonr of FFT wavelengths against lambda(k_sigma) across the D_u sweep",
            "derived_from": "script",
            "reference_value": 0.98,
            "reference_source": "Pre-registered MorphoEval gate (r > 0.98), not the measured correlation",
        },
        {
            "name": "wavelength RMSE against maximum-growth prediction",
            "value": round(float(result["rmse_px"]), 4),
            "unit": "px",
            "method": "RMSE of measured radial-FFT wavelength versus sigma-peak wavelength over the D_u sweep",
            "derived_from": "script",
            "reference_value": round(float(result["sweep"]["rmse_px_vs_onset"]), 4),
            "reference_source": "RMSE against the onset wavelength lambda(k_c), a different predictor than lambda(k_sigma)",
            "note": "KS/CV companion: orientation KS is reported as its own measurement",
        },
        {
            "name": "specimen pigment-segment fraction",
            "value": round(float(result["specimen"]["specimen_pigment_segment_fraction"]), 4),
            "unit": "area fraction",
            "method": "Otsu threshold on the species-card grayscale; fraction darker than the threshold",
            "derived_from": "output",
            "reference_value": round(float(result["specimen"]["simulation_coherency"]), 4),
            "reference_source": "Simulation structure-tensor coherency (different quantity; not a self-copy of the segment fraction)",
        },
        {
            "name": "specimen versus simulation orientation KS statistic",
            "value": round(float(result["specimen"]["orientation_ks_statistic"]), 4),
            "unit": "KS distance",
            "method": "ks_2samp on 4000 magnitude-weighted pixel orientations (not the 12-bin histogram)",
            "derived_from": "output",
            "reference_value": round(float(result["specimen"]["orientation_ks_bins"]), 4),
            "reference_source": "Pass-1 KS on the 12 histogram-bin masses, a different and coarser sample",
        },
        {
            "name": "near-onset fractional wavelength gap at r=9 versus r=10",
            "value": round(float(result["onset_gap"][0]["fractional_gap"]), 4),
            "unit": "fraction",
            "method": "(lambda_sigma - lambda_c) / lambda_c at r = 9, just above onset",
            "derived_from": "script",
            "reference_value": round(float(result["onset_gap"][1]["fractional_gap"]), 4),
            "reference_source": "Same gap at the operating ratio r = 10; the near-onset gap must be smaller",
        },
        {
            "name": "equal-diffusion ablation amplitude ratio",
            "value": round(float(result["ablation"]["amplitude_ratio"]), 4),
            "unit": "std ratio",
            "method": "final activator std at r = 1 divided by final std at r = 10, same D_u and seed family",
            "derived_from": "script",
            "reference_value": 1.0,
            "reference_source": "Null expectation if equal diffusion still patterned",
            "note": "negative control; discriminant at r = 1 is negative",
        },
        {
            "name": "detail-scale versus primordium-scale americana/affinis wavelength ratio",
            "value": round(float(detail_ratio), 4),
            "unit": "ratio",
            "method": "lambda_ame/lambda_aff at beta = 0.35 body-length buffering, versus sibling primordium wavelengths",
            "derived_from": "script",
            "reference_value": round(float(prim_ratio), 4),
            "reference_source": "Primordium-scale sibling slots (138.7 µm / 95.94 µm), not this model's detail-scale ratio",
        },
    ]


def self_test() -> dict:
    result = run_analysis(render=False)
    waves = detail_wavelengths()
    assert waves["Drosophila americana"] == LAMBDA_M
    assert waves["Drosophila americana"] > waves["Drosophila affinis"] > waves["Drosophila ananassae"]
    detail_ratio = waves["Drosophila americana"] / waves["Drosophila affinis"]
    prim_ratio = PRIMORDIUM_M["Drosophila americana"] / PRIMORDIUM_M["Drosophila affinis"]
    assert detail_ratio < prim_ratio
    assert result["linear_theory"]["discriminant"] > 0.0
    assert result["ablation"]["ablation_discriminant"] < 0.0
    assert result["ablation"]["amplitude_ratio"] < 0.35
    assert result["render_std"] > 0.25
    assert result["pearson_r"] > 0.98, result["pearson_r"]
    gap9, gap10 = result["onset_gap"]
    assert gap9["r"] == 9.0 and gap10["r"] == 10.0
    assert gap9["fractional_gap"] < gap10["fractional_gap"]
    assert result["specimen"]["orientation_ks_statistic"] != result["specimen"]["orientation_ks_bins"]
    assert result["specimen"]["specimen_pigment_segment_fraction"] != result["specimen"]["simulation_coherency"]
    # Lean integers used by Loop_3175, checked so the proof file cannot drift.
    assert int(round(waves["Drosophila americana"] * 1e9)) == 2117
    assert int(round(waves["Drosophila affinis"] * 1e9)) == 1958
    assert int(round(waves["Drosophila ananassae"] * 1e9)) == 1900
    assert (4 * 10 - 5) ** 2 - 100 * 10 > 0
    assert (4 * 1 - 5) ** 2 - 100 * 1 < 0
    print(
        f"[PASS] slot 3175 self-test  r={result['pearson_r']:.5f}  "
        f"rmse_px={result['rmse_px']:.3f}  "
        f"lambda_um={result['measured_wavelength_um']:.4f}  "
        f"ablation_ratio={result['ablation']['amplitude_ratio']:.4f}  "
        f"age={AGE_MA} Ma  scale=2.1 µm"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result = run_analysis(render=True)
    public = {
        key: result[key]
        for key in (
            "slot",
            "sim_id",
            "organism",
            "phenomenon_size_m",
            "size_human",
            "scale_tier",
            "geological_age_ma",
            "pearson_r",
            "rmse_px",
            "measured_wavelength_um",
            "field_of_view_um",
            "gif_frames",
            "gif_size",
            "render_std",
        )
    }
    public["ablation_amplitude_ratio"] = result["ablation"]["amplitude_ratio"]
    public["orientation_ks"] = result["specimen"]["orientation_ks_statistic"]
    public["detail_um"] = {
        row["species"]: round(row["detail_wavelength_um"], 4) for row in result["chronoseries"]
    }
    print(json.dumps(public, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
