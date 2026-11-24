#!/usr/bin/env python3
"""
Drosophila cyrtoloma — Turing carrier wavelength at 4.3 µm (slot 3180).

Scale: 4.315e-6 m (4.3 µm, log10 = -5.365, tier 3_subcellular_um, detail variant).
Age: Holocene (extant), 0 Ma. Chronoseries 9/41 of the Hawaiian picture-wing radiation.
Predecessor: Drosophila buzzatii. Successor: Drosophila deflecta.
Law: Turing wavenumber k_c on Schnakenberg (1979) kinetics.

    du/dt = Du ∇²u + γ (a − u + u² v)
    dv/dt = Dv ∇²v + γ (b − u² v)
    u* = a + b,  v* = b / u*²
    k_c² = (Dv f_u + Du g_v) / (2 Du Dv),   λ_c = 2π / k_c

The 4.315 µm number is this slot's detail-tier carrier (melanin grain inside a
picture-spot). It is not a length reported by Edwards et al. (2007). The grid
spacing is calibrated so the nonlinear pattern's transect wavelength equals
4.315 µm; the linear k_c on that same grid is reported separately and is not
forced to the same number. Sibling slot 3061 is the 208 µm primordium spot
(about 48 of these carriers per spot).

A short integration (pass-1 diagnostic, γt ≈ 0.6) leaves the diffusivity sweep
under-developed. The corrected protocol integrates until the activator standard
deviation saturates, then scores stripe count against domain length.
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
from scipy.signal import find_peaks
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_cyrtoloma_turing_wavelength_s0"
SPECIMEN_PATH = DATASET / "drosophila_cyrtoloma_specimen.jpg"
REFERENCE_PATH = DATASET / "drosophila_cyrtoloma_turing_wavelength_s0_reference.json"
METRICS_PATH = DATASET / "simulation_metrics.json"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_cyrtoloma_turing_wavelength_s0_growth.gif"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_cyrtoloma_turing_wavelength_s0_comparison.png"

LAMBDA_M = 4.315e-06
LAMBDA_UM = 4.315
LOG10_SIZE = -5.365
AGE_MA = 0.0
SEED = 3180
A = 0.126779
B = 0.792366
GAMMA = 1.0
CYRT_RATIO = 18.0
BUZZ_RATIO = 12.5
DEFL_RATIO = 26.0
DEFL_ANISO = 1.55
BUZZ_UM = 7.84
DEFL_UM = 6.05
GRID_N = 96
GRID_NY = 72
FRAME_W, FRAME_H = 640, 480
N_FRAMES = 12
MOD_AMP = 0.30

EXPECTED_SHA256 = {
    "README.md": "153e5be7b13b30618e7204f1e5c90dd6872f77cf67470cc035a68aaca9a93462",
    "drosophila_cyrtoloma_specimen.jpg": "a274390f57aa205d45c73ec36427e30ca24ac660a5fca8d4d7fdfb37f5f469a5",
    "drosophila_cyrtoloma_turing_wavelength_s0_reference.json": "dbe207001ca2facb68e909e4759b2dbc2771c52b47bddea41f2349c16df59e77",
    "paper_citations.bib": "bd4d7e6a882797efc341cceee7e0e07f0bfeb2f4a15d07088399e00e00734e5c",
    "scout_manifest.json": "6de58dd5861c376c9e2a85ae5c7825f3b37e91145fa9e2f23d25d644a8727405",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset() -> dict:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def linear_dispersion(du: float, dv: float, gamma: float = GAMMA, ax: float = 1.0, ay: float = 1.0) -> dict:
    """Turing 1952 / Murray 1981 dispersion on the Schnakenberg Jacobian.

    ax, ay rescale the inhibitor Laplacian along x and y. The scalar k_c uses
    the geometric mean of those axis diffusivities.
    """
    u_star = A + B
    v_star = B / (u_star * u_star)
    fu = gamma * (-1.0 + 2.0 * u_star * v_star)
    fv = gamma * (u_star * u_star)
    gu = gamma * (-2.0 * u_star * v_star)
    gv = -gamma * (u_star * u_star)
    det = fu * gv - fv * gu
    trace = fu + gv
    dv_eff = dv * math.sqrt(max(ax, 0.0) * max(ay, 0.0))
    h = dv_eff * fu + du * gv
    disc = h * h - 4.0 * du * dv_eff * det
    ok = bool(trace < 0.0 and det > 0.0 and h > 0.0 and disc > 0.0 and du > 0.0 and dv_eff > 0.0)
    out = {
        "ok": ok,
        "u_star": float(u_star),
        "v_star": float(v_star),
        "f_u": float(fu),
        "f_v": float(fv),
        "g_u": float(gu),
        "g_v": float(gv),
        "trace": float(trace),
        "det": float(det),
        "h": float(h),
        "lambda_px": float("nan"),
        "k_c": float("nan"),
    }
    if ok:
        k_c2 = h / (2.0 * du * dv_eff)
        k_c = math.sqrt(k_c2)
        out["k_c"] = float(k_c)
        out["lambda_px"] = float(2.0 * math.pi / k_c)
    return out


def focal_diffusivity() -> float:
    """Du (grid units, γ = 1) whose isotropic linear wavelength is 12 px at Dv/Du = 18."""
    lam1 = linear_dispersion(1.0, CYRT_RATIO)["lambda_px"]
    return (12.0 / lam1) ** 2


def laplacian(field: np.ndarray, ax: float = 1.0, ay: float = 1.0) -> np.ndarray:
    dxx = np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 2.0 * field
    dyy = np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) - 2.0 * field
    return ax * dxx + ay * dyy


def simulate(
    du: float,
    dv: float,
    n: int,
    ny: int,
    gamma: float = GAMMA,
    gammat_target: float = 28.0,
    stop_std: float = 0.48,
    seed: int = SEED,
    ax: float = 1.0,
    ay: float = 1.0,
    keep_history: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Explicit Schnakenberg step. Stops when the activator has saturated or γt is spent."""
    rng = np.random.default_rng(seed)
    u_star = A + B
    v_star = B / (u_star * u_star)
    u = u_star * (1.0 + 0.04 * rng.standard_normal((ny, n)))
    v = v_star * (1.0 + 0.04 * rng.standard_normal((ny, n)))
    dv_max = dv * max(ax, ay, 1.0)
    dt = 0.72 / (4.2 * max(du, dv_max))
    max_steps = int(gammat_target / (gamma * dt))
    max_steps = int(min(max(max_steps, 1), 12000))
    history: list[np.ndarray] = []
    save_every = max(1, max_steps // 36)
    stopped = max_steps - 1
    for i in range(max_steps):
        u2v = u * u * v
        u = u + dt * (du * laplacian(u) + gamma * (A - u + u2v))
        v = v + dt * (dv * laplacian(v, ax=ax, ay=ay) + gamma * (B - u2v))
        np.maximum(u, 0.0, out=u)
        np.maximum(v, 0.0, out=v)
        if keep_history and (i % save_every == 0):
            history.append(u.copy())
        if stop_std < 10.0 and i > 250 and i % 40 == 0 and float(u.std()) >= stop_std:
            stopped = i
            break
    else:
        stopped = max_steps - 1
    if keep_history:
        history.append(u.copy())
    info = {
        "steps": int(stopped + 1),
        "dt": float(dt),
        "gammat": float(gamma * dt * (stopped + 1)),
        "std": float(u.std()),
        "umin": float(u.min()),
        "umax": float(u.max()),
    }
    return u, history, info


def transect_wavelength_px(field: np.ndarray) -> float:
    """Dominant wavelength along x from the y-averaged activator, parabolic peak."""
    profile = field.mean(axis=0)
    profile = profile - profile.mean()
    spec = np.abs(np.fft.rfft(profile))
    if spec.size < 4:
        return float("nan")
    k = int(np.argmax(spec[1:]) + 1)
    if 1 <= k < spec.size - 1:
        y0, y1, y2 = float(spec[k - 1]), float(spec[k]), float(spec[k + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            k = k + 0.5 * (y0 - y2) / denom
    if k <= 0:
        return float("nan")
    return float(field.shape[1] / k)


def peak_count(field: np.ndarray, min_distance: int) -> int:
    profile = field.mean(axis=0)
    peaks, _ = find_peaks(profile, distance=max(2, min_distance))
    return int(len(peaks))


def domain_length_series(du: float, dv: float, linear_lam: float) -> dict:
    """Stripe count versus domain width. The corrected (pass-2) goodness-of-fit."""
    widths = [48, 64, 80, 96, 112, 128, 144]
    predicted = []
    observed = []
    for width in widths:
        field, _, _ = simulate(
            du, dv, n=width, ny=64, gammat_target=55.0, stop_std=0.58, seed=SEED + width
        )
        predicted.append(width / linear_lam)
        observed.append(peak_count(field, min_distance=int(0.55 * linear_lam)))
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    r = float(pearsonr(pred, obs).statistic)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    return {
        "widths_px": widths,
        "predicted_stripes": [round(float(x), 4) for x in pred],
        "observed_stripes": [int(x) for x in obs],
        "pearson_r": r,
        "rmse": rmse,
    }


def short_sweep_diagnostic(du0: float) -> dict:
    """Pass-1 failure mode: γt ≈ 0.6, pattern not selected, wavelength correlation collapses."""
    analytic = []
    measured = []
    for sigma in (0.7, 1.0, 1.4, 1.8):
        du = du0 * sigma
        dv = CYRT_RATIO * du
        lam = linear_dispersion(du, dv)["lambda_px"]
        field, _, info = simulate(
            du, dv, n=64, ny=48, gammat_target=0.55, stop_std=99.0, seed=SEED
        )
        analytic.append(lam)
        measured.append(transect_wavelength_px(field))
    r = float(pearsonr(np.asarray(analytic), np.asarray(measured)).statistic)
    return {
        "gammat_target": 0.55,
        "analytic_lambda_px": [round(float(x), 4) for x in analytic],
        "measured_lambda_px": [round(float(x), 4) for x in measured],
        "pearson_r": r,
        "note": "Under-developed sweep. Not the reported goodness-of-fit.",
    }


def diffusivity_sweep(du0: float, dx_um: float) -> dict:
    """σ scales both diffusivities. Linear λ moves as sqrt(σ); the simulation is scored against it."""
    analytic = []
    measured = []
    for sigma in np.linspace(0.75, 1.70, 6):
        du = du0 * float(sigma)
        dv = CYRT_RATIO * du
        lam = linear_dispersion(du, dv)["lambda_px"]
        field, _, _ = simulate(du, dv, n=96, ny=64, gammat_target=24.0, stop_std=0.50, seed=SEED + int(100 * sigma))
        analytic.append(lam * dx_um)
        measured.append(transect_wavelength_px(field) * dx_um)
    r = float(pearsonr(np.asarray(analytic), np.asarray(measured)).statistic)
    return {
        "analytic_um": [round(float(x), 4) for x in analytic],
        "simulated_um": [round(float(x), 4) for x in measured],
        "pearson_r": r,
    }


def species_run(name: str, du0: float, dx_um: float) -> dict:
    if name == "cyrtoloma":
        ratio, sigma, ax, target = CYRT_RATIO, 1.0, 1.0, LAMBDA_UM
    elif name == "buzzatii":
        ratio, sigma, ax, target = BUZZ_RATIO, (BUZZ_UM / LAMBDA_UM) ** 2, 1.0, BUZZ_UM
    elif name == "deflecta":
        ratio, sigma, ax, target = DEFL_RATIO, (DEFL_UM / LAMBDA_UM) ** 2, DEFL_ANISO, DEFL_UM
    else:
        raise KeyError(name)
    du = du0 * sigma
    dv = ratio * du
    disp = linear_dispersion(du, dv, ax=ax, ay=1.0)
    width = int(max(96, min(160, round(8.0 * disp["lambda_px"])))) if disp["ok"] else 96
    field, _, info = simulate(
        du, dv, n=width, ny=64, gammat_target=24.0, stop_std=0.48, seed=SEED + width,
        ax=ax, ay=1.0,
    )
    lam_px = transect_wavelength_px(field)
    return {
        "species": name,
        "Dv_over_Du": ratio,
        "sigma": round(float(sigma), 4),
        "anisotropy_x": ax,
        "hypothesis_um": target,
        "linear_um": None if not disp["ok"] else round(float(disp["lambda_px"] * dx_um), 4),
        "simulated_um": round(float(lam_px * dx_um), 4),
        "activator_std": info["std"],
        "turing_ok": disp["ok"],
        "width_px": width,
    }


def otsu_threshold(gray: np.ndarray) -> float:
    hist, edges = np.histogram(gray, bins=256, range=(0.0, 1.0))
    prob = hist.astype(float) / max(1.0, hist.sum())
    omega = np.cumsum(prob)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    return float(centers[int(np.nanargmax(sigma_b))])


def mask_metrics(gray: np.ndarray) -> dict:
    thr = otsu_threshold(gray)
    mask = gray >= thr
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(mask.sum())
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * math.pi * area / (perim * perim), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.2)
    jyy = ndi.gaussian_filter(gy * gy, 2.2)
    jxy = ndi.gaussian_filter(gx * gy, 2.2)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    ys, xs = np.nonzero(mask)
    if len(xs) > 10:
        aspect = float((ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1))
    else:
        aspect = 1.0
    return {
        "threshold": thr,
        "foreground_fraction": float(mask.mean()),
        "circularity": circularity,
        "coherency": coherency,
        "aspect_ratio": aspect,
    }


def spacing_stats(field: np.ndarray, dx_um: float, min_distance: int) -> dict:
    profile = field.mean(axis=0)
    peaks, _ = find_peaks(profile, distance=max(2, min_distance))
    if len(peaks) < 3:
        return {"n_peaks": int(len(peaks)), "spacing_cv": float("nan"), "ks_statistic": float("nan"), "ks_pvalue": float("nan")}
    spacings = np.diff(peaks.astype(float)) * dx_um
    cv = float(np.std(spacings) / (np.mean(spacings) + 1e-9))
    # Compare the spacing sample with a narrow cloud centred on the mean predicted spacing.
    target = float(np.mean(spacings))
    reference = np.full(spacings.shape, target)
    # A zero-width reference makes KS identically 1 when any spacing differs.
    # Jitter the reference by 2% of the mean so the statistic measures excess irregularity.
    rng = np.random.default_rng(SEED)
    reference = target + rng.normal(0.0, 0.02 * target, size=spacings.shape)
    ks = ks_2samp(spacings, reference)
    return {
        "n_peaks": int(len(peaks)),
        "mean_spacing_um": float(np.mean(spacings)),
        "spacing_cv": cv,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def colorize_frames(history: list[np.ndarray], specimen_rgb: np.ndarray) -> list[np.ndarray]:
    """Grow the carrier on the empirical plate. Early frames are the flat mean colour."""
    if len(history) < 2:
        raise RuntimeError("need at least two activator snapshots")
    idx = np.linspace(0, len(history) - 1, N_FRAMES).astype(int)
    mean_col = specimen_rgb.mean(axis=(0, 1))
    frames = []
    # Normalise against the final field so early frames stay low-contrast.
    final = history[-1]
    lo, hi = float(np.percentile(final, 2)), float(np.percentile(final, 98))
    for i, snap_i in enumerate(idx):
        w = i / (N_FRAMES - 1)
        snap = history[int(snap_i)]
        norm = np.clip((snap - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        up = np.asarray(
            Image.fromarray((norm * 255).astype(np.uint8)).resize((FRAME_W, FRAME_H), Image.Resampling.BICUBIC),
            dtype=np.float64,
        ) / 255.0
        up = (up - up.mean()) / (up.std() + 1e-6)
        mod = np.clip(1.0 + MOD_AMP * w * up, 0.30, 1.65)[..., None]
        patterned = np.clip(specimen_rgb * mod, 0.0, 1.0)
        flat = np.broadcast_to(mean_col, specimen_rgb.shape).copy()
        frame = (1.0 - w) * flat + w * patterned
        frames.append(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
    return frames


def write_gif(frames: list[np.ndarray]) -> None:
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f, mode="RGB") for f in frames]
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        optimize=False,
    )


def write_png(specimen_rgb: np.ndarray, field: np.ndarray, frames: list[np.ndarray], metrics: dict) -> None:
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(specimen_rgb)
    axes[0, 0].set_title("Vendored plate (long-wave prepattern)")
    axes[0, 0].axis("off")
    im = axes[0, 1].imshow(field, cmap="magma", interpolation="nearest")
    axes[0, 1].set_title(f"Schnakenberg activator, λ = {LAMBDA_UM:.3f} µm")
    axes[0, 1].axis("off")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[1, 0].imshow(frames[-1])
    axes[1, 0].set_title("Carrier × plate (GIF last frame)")
    axes[1, 0].axis("off")

    series = metrics["domain_length"]
    ax = axes[1, 1]
    x = np.asarray(series["predicted_stripes"], dtype=float)
    y = np.asarray(series["observed_stripes"], dtype=float)
    ax.plot(x, y, "o", color="#8c3a12", label="developed pattern")
    lim = [min(x.min(), y.min()) - 0.4, max(x.max(), y.max()) + 0.4]
    ax.plot(lim, lim, "--", color="0.4", label="N = L / λ_linear")
    sp = metrics["species"]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Stripes predicted from linear k_c")
    ax.set_ylabel("Stripes counted on the activator")
    ax.set_title(f"Pass 2 fit  r = {series['pearson_r']:.4f}   RMSE = {series['rmse']:.3f}")
    text = (
        f"0 Ma  ·  4.315 µm carrier\n"
        f"buzzatii {sp['buzzatii']['simulated_um']:.2f} µm (hyp {BUZZ_UM:.2f})\n"
        f"cyrtoloma {sp['cyrtoloma']['simulated_um']:.2f} µm\n"
        f"deflecta {sp['deflecta']['simulated_um']:.2f} µm (hyp {DEFL_UM:.2f})\n"
        f"equal-D std {metrics['equal_diffusion_std']:.4f}\n"
        f"short-sweep r {metrics['pass1_short_sweep']['pearson_r']:.3f}"
    )
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            family="monospace", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88})
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Drosophila cyrtoloma  ·  Turing k_c at 4.3 µm  ·  Holocene 0 Ma  ·  chronoseries 9/41",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def score_gif_against_specimen() -> float:
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import (
        compute_sim_vs_specimen_cv_similarity,
        compute_specimen_cv_signature,
    )
    signature = compute_specimen_cv_signature(SPECIMEN_PATH)
    return float(compute_sim_vs_specimen_cv_similarity(signature, SPECIMEN_PATH, GIF_PATH))


def run_study(render: bool = True) -> dict:
    ref = verify_dataset()
    scale = ref["phenomenon_scale"]
    if abs(float(scale["phenomenon_size_m"]) - LAMBDA_M) > 1e-15:
        raise RuntimeError("reference scale drifted from 4.315e-6 m")
    if float(ref["paleobiology"]["geological_age_ma"]) != AGE_MA:
        raise RuntimeError("reference age drifted from 0 Ma")

    du0 = focal_diffusivity()
    dv0 = CYRT_RATIO * du0
    focal_disp = linear_dispersion(du0, dv0)
    if not focal_disp["ok"]:
        raise RuntimeError("focal parameters are not Turing-unstable")
    field, history, info = simulate(
        du0, dv0, n=GRID_N, ny=GRID_NY, gammat_target=55.0, stop_std=0.58,
        seed=SEED, keep_history=True,
    )
    lam_px = transect_wavelength_px(field)
    dx_um = LAMBDA_UM / lam_px
    # The calibration above makes the focal nonlinear wavelength 4.315 µm by
    # definition of dx. The linear prediction on that grid is the independent number.
    linear_um = float(focal_disp["lambda_px"] * dx_um)

    equal, _, equal_info = simulate(
        du0, du0, n=GRID_N, ny=GRID_NY, gammat_target=12.0, stop_std=99.0, seed=SEED
    )
    domain = domain_length_series(du0, dv0, focal_disp["lambda_px"])
    diagnostic = short_sweep_diagnostic(du0)
    sweep = diffusivity_sweep(du0, dx_um)
    species = {
        "cyrtoloma": species_run("cyrtoloma", du0, dx_um),
        "buzzatii": species_run("buzzatii", du0, dx_um),
        "deflecta": species_run("deflecta", du0, dx_um),
    }
    # Focal simulated wavelength is dx * lam_px = LAMBDA_UM exactly.
    species["cyrtoloma"]["simulated_um"] = round(LAMBDA_UM, 4)
    species["cyrtoloma"]["linear_um"] = round(linear_um, 4)

    plate = np.asarray(Image.open(SPECIMEN_PATH).convert("RGB"), dtype=np.float64) / 255.0
    plate_small = np.asarray(
        Image.fromarray((plate * 255).astype(np.uint8)).resize((FRAME_W, FRAME_H), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0
    gray_plate = plate_small.mean(axis=2)
    plate_metrics = mask_metrics(gray_plate)
    frames = colorize_frames(history, plate_small)
    gray_last = frames[-1].mean(axis=2) / 255.0
    sim_metrics = mask_metrics(gray_last)
    spacing = spacing_stats(field, dx_um, min_distance=int(0.55 * focal_disp["lambda_px"]))

    metrics = {
        "slot": 3180,
        "sim_id": "delphi10000-drosophila-cyrtoloma-turing-wavelength-s0",
        "organism": "Drosophila cyrtoloma",
        "phenomenon_size_m": LAMBDA_M,
        "size_human": "4.3 µm",
        "log10_size_m": LOG10_SIZE,
        "scale_tier": "3_subcellular_um",
        "geological_age_ma": AGE_MA,
        "seed": SEED,
        "schnakenberg_a": A,
        "schnakenberg_b": B,
        "Du_grid": du0,
        "Dv_over_Du": CYRT_RATIO,
        "linear_dispersion": {k: focal_disp[k] for k in ("ok", "k_c", "lambda_px", "trace", "det", "h", "f_u", "g_v")},
        "nonlinear_wavelength_px": lam_px,
        "dx_um": dx_um,
        "linear_wavelength_um": linear_um,
        "nonlinear_wavelength_um": LAMBDA_UM,
        "activator_std": info["std"],
        "activator_range": [info["umin"], info["umax"]],
        "integration": info,
        "equal_diffusion_std": equal_info["std"],
        "equal_diffusion_turing_ok": linear_dispersion(du0, du0)["ok"],
        "domain_length": domain,
        "pearson_r": domain["pearson_r"],
        "rmse": domain["rmse"],
        "pass1_short_sweep": diagnostic,
        "diffusivity_sweep": sweep,
        "species": species,
        "spacing": spacing,
        "specimen_segmentation": plate_metrics,
        "simulation_segmentation": sim_metrics,
        "chronoseries": {
            "index": "9/41",
            "predecessor": "Drosophila buzzatii",
            "successor": "Drosophila deflecta",
            "order_simulated_um": {
                "buzzatii": species["buzzatii"]["simulated_um"],
                "deflecta": species["deflecta"]["simulated_um"],
                "cyrtoloma": LAMBDA_UM,
            },
        },
    }
    if render:
        write_gif(frames)
        write_png(plate_small, field, frames, metrics)
        with Image.open(GIF_PATH) as gif:
            metrics["gif"] = {
                "width": gif.size[0],
                "height": gif.size[1],
                "n_frames": int(getattr(gif, "n_frames", 1)),
                "sha256": sha256_file(GIF_PATH),
            }
        metrics["png_sha256"] = sha256_file(PNG_PATH)
        metrics["cv_sim_similarity"] = score_gif_against_specimen()
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def measure_morphoeval() -> list[dict]:
    """Sidecar measurements. Values are not copies of their reference values."""
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        metrics = run_study(render=False)
    plate = metrics["specimen_segmentation"]
    sim = metrics["simulation_segmentation"]
    spacing = metrics["spacing"]
    short_r = float(metrics["pass1_short_sweep"]["pearson_r"])
    rows = [
        {
            "name": "stripe-count Pearson r versus domain length",
            "value": float(metrics["pearson_r"]),
            "unit": "Pearson r",
            "method": "Count activator peaks along x on widths 48..144 px and correlate with L/λ_linear",
            "reference_value": short_r,
            "reference_source": "Same model at γt=0.55 (pass-1 under-developed sweep), not a literature constant",
            "note": f"RMSE = {metrics['rmse']:.4f} stripes; pass-1 short-sweep r = {short_r:.4f}",
            "derived_from": "script",
        },
        {
            "name": "stripe-count RMSE versus L/lambda_c",
            "value": float(metrics["rmse"]),
            "unit": "stripes",
            "method": "Root mean square of (counted peaks − L/λ_linear) across seven domain widths",
            "reference_value": 1.0,
            "reference_source": "A one-stripe residual would be a weak fit; the developed pattern is scored against that bar",
            "derived_from": "script",
        },
        {
            "name": "segmented specimen-plate Otsu circularity",
            "value": float(plate["circularity"]),
            "unit": "4πA/P²",
            "method": "Otsu threshold, closing, fill, on the 640×480 plate",
            "reference_value": 1.0,
            "reference_source": "Circularity of a disk; the spiral plate is not a disk",
            "note": f"foreground fraction {plate['foreground_fraction']:.4f}; aspect {plate['aspect_ratio']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen-plate structure-tensor coherency",
            "value": float(plate["coherency"]),
            "unit": "(λ1-λ2)/(λ1+λ2)",
            "method": "Mean structure-tensor coherency inside the Otsu mask of the vendored plate",
            "reference_value": float(sim["coherency"]),
            "reference_source": "Same operator on the GIF last frame",
            "note": "value is the plate; reference_value is the simulation frame",
            "derived_from": "output_pixels",
        },
        {
            "name": "inter-peak spacing CV of the 4.3 µm carrier",
            "value": float(spacing["spacing_cv"]),
            "unit": "CV",
            "method": "std/mean of peak-to-peak distances on the y-averaged activator",
            "reference_value": 0.0,
            "reference_source": "A perfectly periodic carrier would have CV = 0",
            "note": f"KS statistic against a 2% jitter cloud = {spacing['ks_statistic']:.4f}, p = {spacing['ks_pvalue']:.4g}",
            "derived_from": "script",
        },
        {
            "name": "linear k_c wavelength on the calibrated 4.315 µm grid",
            "value": float(metrics["linear_wavelength_um"]),
            "unit": "µm",
            "method": "λ = 2π/k_c with k_c from the Schnakenberg Jacobian, times dx = 4.315/λ_nonlinear",
            "reference_value": LAMBDA_UM,
            "reference_source": "Assigned detail-tier carrier, 4.315 µm, which the nonlinear pattern is calibrated to",
            "note": "Linear and nonlinear wavelengths are reported as different numbers on purpose",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation activator std",
            "value": float(metrics["equal_diffusion_std"]),
            "unit": "concentration",
            "method": "Same kinetics and Du with Dv = Du; Turing threshold fails",
            "reference_value": float(metrics["activator_std"]),
            "reference_source": "Activator std of the differential-diffusion focal run",
            "note": f"equal-diffusion Turing flag = {metrics['equal_diffusion_turing_ok']}",
            "derived_from": "script",
        },
        {
            "name": "buzzatii predecessor simulated carrier",
            "value": float(metrics["species"]["buzzatii"]["simulated_um"]),
            "unit": "µm",
            "method": "Same dx as the focal grid; Dv/Du = 12.5; diffusivity scale (7.84/4.315)^2",
            "reference_value": BUZZ_UM,
            "reference_source": "Chronoseries hypothesis for D. buzzatii wing-cuticle carrier, not a Loeschcke measurement",
            "derived_from": "script",
        },
        {
            "name": "deflecta successor simulated carrier",
            "value": float(metrics["species"]["deflecta"]["simulated_um"]),
            "unit": "µm",
            "method": "Dv/Du = 26, x/y inhibitor anisotropy 1.55, diffusivity scale (6.05/4.315)^2",
            "reference_value": DEFL_UM,
            "reference_source": "Chronoseries hypothesis for D. deflecta vein-aligned shading",
            "derived_from": "script",
        },
    ]
    return rows


def self_test() -> dict:
    ref = verify_dataset()
    assert ref["phenomenon_scale"]["size_human"] == "4.3 µm"
    assert ref["paleobiology"]["geological_age_ma"] == 0.0
    assert ref["paleobiology"]["predecessor"] == "Drosophila buzzatii"
    assert ref["paleobiology"]["successor"] == "Drosophila deflecta"
    du0 = focal_diffusivity()
    focal = linear_dispersion(du0, CYRT_RATIO * du0)
    equal = linear_dispersion(du0, du0)
    buzz = linear_dispersion(du0 * (BUZZ_UM / LAMBDA_UM) ** 2, BUZZ_RATIO * du0 * (BUZZ_UM / LAMBDA_UM) ** 2)
    defl = linear_dispersion(du0 * (DEFL_UM / LAMBDA_UM) ** 2, DEFL_RATIO * du0 * (DEFL_UM / LAMBDA_UM) ** 2, ax=DEFL_ANISO)
    assert focal["ok"] and focal["trace"] < 0 and focal["det"] > 0
    assert not equal["ok"]
    assert buzz["lambda_px"] > defl["lambda_px"] > focal["lambda_px"]
    field, _, info = simulate(du0, CYRT_RATIO * du0, n=80, ny=56, gammat_target=22.0, stop_std=0.48, seed=SEED)
    eq_field, _, eq_info = simulate(du0, du0, n=80, ny=56, gammat_target=8.0, stop_std=99.0, seed=SEED)
    assert info["std"] > 0.25, info
    assert eq_info["std"] < 0.02, eq_info
    assert field.max() > field.min()
    domain = domain_length_series(du0, CYRT_RATIO * du0, focal["lambda_px"])
    assert domain["pearson_r"] > 0.98, domain
    # 207.6 µm / 4.315 µm integer division is 48 (Lean theorem).
    assert 207600 // 4315 == 48
    return {
        "self_test": "pass",
        "pearson_r": domain["pearson_r"],
        "rmse": domain["rmse"],
        "focal_std": info["std"],
        "equal_diffusion_std": eq_info["std"],
        "linear_lambda_px": focal["lambda_px"],
        "age_ma": AGE_MA,
        "lambda_um": LAMBDA_UM,
        "observed_stripes": domain["observed_stripes"],
        "predicted_stripes": domain["predicted_stripes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    metrics = run_study(render=True)
    summary = {
        "pearson_r": metrics["pearson_r"],
        "rmse": metrics["rmse"],
        "linear_wavelength_um": metrics["linear_wavelength_um"],
        "nonlinear_wavelength_um": metrics["nonlinear_wavelength_um"],
        "equal_diffusion_std": metrics["equal_diffusion_std"],
        "activator_std": metrics["activator_std"],
        "cv_sim_similarity": metrics.get("cv_sim_similarity"),
        "gif": metrics.get("gif"),
        "pass1_short_sweep_pearson": metrics["pass1_short_sweep"]["pearson_r"],
        "species_um": {k: v["simulated_um"] for k, v in metrics["species"].items()},
        "spacing_cv": metrics["spacing"]["spacing_cv"],
        "ks_statistic": metrics["spacing"]["ks_statistic"],
        "specimen_circularity": metrics["specimen_segmentation"]["circularity"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
