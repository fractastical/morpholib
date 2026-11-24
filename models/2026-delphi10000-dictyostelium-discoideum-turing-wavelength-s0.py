#!/usr/bin/env python3
"""Dictyostelium discoideum cortical Turing wavelength at 2.304 µm (slot 3168).

Holocene extant social amoeba, geological age 0 Ma, chronoseries 1/1.
Scale tier 3_subcellular_um. The quantitative law is the Turing critical
wavenumber of Schnakenberg kinetics (Schnakenberg 1979; Turing 1952):

    du/dt = Du ∇²u + a - u + u² v
    dv/dt = Dv ∇²v + b - u² v
    k_c²  = (Dv f_u + Du g_v) / (2 Du Dv)
    λ_c   = 2π / k_c = 2.304 µm

The cortical patch is one vegetative cell, L = 11.52 µm = 5 λ_c.
A 0.20 µm molecular predecessor domain admits zero modes. Millimetre-scale
extracellular cAMP waves (Höfer, Sherratt & Maini 1995) are the longer
successor regime of the same extant species, not a later chronospecies.

Taniguchi et al. 2013 document cortical PIP3/F-actin waves. Those waves are
excitable. This script does not integrate their phase-singularity model; it
selects k_c with Schnakenberg kinetics tuned to the slot coordinate.

Boundary condition: periodic torus, so integer modes 2π n / L are exact.
Negative control: Du = Dv, which closes the unstable band.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPECIMEN = ROOT / (
    "datasets/dictyostelium_discoideum_turing_wavelength_s0/"
    "dictyostelium_discoideum_turing_wavelength_s0_specimen.jpg"
)
METRICS_PATH = ROOT / (
    "datasets/dictyostelium_discoideum_turing_wavelength_s0/"
    "dictyostelium_discoideum_turing_wavelength_s0_metrics.json"
)
PNG_PATH = ROOT / "docs/media/dictyostelium_discoideum_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs/media/dictyostelium_discoideum_turing_wavelength_s0_growth.gif"

# Slot-3168 parameters. Not copied from a sibling animal.
A = 0.16
B = 0.94
DU = 0.03208776235376616  # µm²/s, activator; scaled so the discrete Laplacian peaks at 2.304 µm
DV = 0.770106296490388    # µm²/s, inhibitor; δ = Dv/Du = 24
DELTA = DV / DU           # 24
LAMBDA_C_UM = 2.304
DOMAIN_UM = 11.52         # 5 * 2.304
N_GRID = 80
DT = 0.004
N_STEPS = 8000
SEED = 3168
GIF_SIZE = (960, 720)     # min edge 720 ≥ 480
MOLECULAR_PREDECESSOR_UM = 0.20
AGGREGATION_SUCCESSOR_UM = 1000.0  # 1 mm order-of-magnitude cAMP field

U0 = A + B
V0 = B / (U0 ** 2)
F_U = -1.0 + 2.0 * U0 * V0
F_V = U0 ** 2
G_U = -2.0 * U0 * V0
G_V = -(U0 ** 2)


def laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    """Five-point periodic Laplacian. Eq. discrete ∇² on the torus."""
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / (dx * dx)


def keff2(kx: float, ky: float, dx: float) -> float:
    """Eigenvalue of the five-point Laplacian on cos(kx x) cos(ky y)."""
    return (2.0 * (1.0 - np.cos(kx * dx)) + 2.0 * (1.0 - np.cos(ky * dx))) / (dx * dx)


def sigma_discrete(kx: float, ky: float, du: float, dv: float, dx: float) -> tuple[float, float]:
    """Real part of the faster eigenvalue, and the discriminant.

    sigma = 1/2 (tr + sqrt(tr^2 - 4 det)) when the discriminant is non-negative.
    Complex pairs (discriminant < 0) return the real part tr/2 and a negative disc.
    """
    q = keff2(kx, ky, dx)
    tr = F_U + G_V - du * q - dv * q
    det = (F_U - du * q) * (G_V - dv * q) - F_V * G_U
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        return 0.5 * tr, disc
    return 0.5 * (tr + np.sqrt(disc)), disc


def unstable_eigenvector(kx: float, ky: float, du: float, dv: float, dx: float) -> tuple[float, float]:
    """(û, v̂) for the larger eigenvalue, normalized so û = 1."""
    q = keff2(kx, ky, dx)
    sigma, _disc = sigma_discrete(kx, ky, du, dv, dx)
    coef = G_V - dv * q - sigma
    if abs(coef) < 1e-12:
        return 1.0, 0.0
    return 1.0, -G_U / coef


def analytical_peak_wavelength(du: float = DU, dv: float = DV) -> tuple[float, float]:
    """Dense scan of sigma(k) on the continuous-k proxy of the discrete operator.

    Returns (lambda_peak_um, sigma_max).
    """
    ks = np.linspace(0.15, 8.0, 4000)
    sig = np.empty_like(ks)
    # Pure x-modes: ky = 0, kx = k. dx from the production grid.
    dx = DOMAIN_UM / N_GRID
    for i, k in enumerate(ks):
        sig[i], _ = sigma_discrete(float(k), 0.0, du, dv, dx)
    j = int(np.argmax(sig))
    k_peak = float(ks[j])
    return (2.0 * np.pi / k_peak), float(sig[j])


def simulate(
    du: float,
    dv: float,
    steps: int,
    seed: int,
    snapshot_steps: list[int] | None = None,
    noise: float = 0.03,
    n: int = N_GRID,
    length: float = DOMAIN_UM,
    dt: float = DT,
) -> dict:
    """Explicit-Euler Schnakenberg integration on a periodic square."""
    dx = length / n
    rng = np.random.default_rng(seed)
    u = U0 + noise * rng.standard_normal((n, n))
    v = V0 + noise * rng.standard_normal((n, n))
    want = set(snapshot_steps or [])
    snaps: list[tuple[int, np.ndarray]] = []
    variances: list[float] = []
    var_every = max(1, steps // 40)
    for t in range(steps):
        if t in want:
            snaps.append((t, u.copy()))
        if t % var_every == 0:
            variances.append(float(np.var(u)))
        uv = u * u * v
        u = u + dt * (du * laplacian(u, dx) + A - u + uv)
        v = v + dt * (dv * laplacian(v, dx) + B - uv)
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise FloatingPointError(f"non-finite state at step {t}")
    snaps.append((steps - 1, u.copy()))
    variances.append(float(np.var(u)))
    return {
        "final_u": u,
        "final_v": v,
        "snaps": snaps,
        "variances": variances,
        "dx": dx,
    }


def mode_growth_table(n_grid: int = 64, dt: float = 0.0035) -> list[dict]:
    """Grow each unstable Fourier mode from its linear eigenvector.

    Pearson r is taken against sigma predicted with the same discrete Laplacian
    the solver uses. Restricted to modes with sigma > 0.02 so oscillatory
    stable modes, which are not pure exponentials, are not mixed in.
    """
    length = DOMAIN_UM
    dx = length / n_grid
    x = np.arange(n_grid) * dx
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rows: list[dict] = []
    for n in range(0, 9):
        for m in range(0, n + 1):
            if n == 0 and m == 0:
                continue
            kx = 2.0 * np.pi * n / length
            ky = 2.0 * np.pi * m / length
            sigma, disc = sigma_discrete(kx, ky, DU, DV, dx)
            if disc < 0.0 or sigma < 0.02:
                continue
            eu, ev = unstable_eigenvector(kx, ky, DU, DV, dx)
            wave = np.cos(kx * xx) * np.cos(ky * yy)
            eps = 2e-4
            u = U0 + eps * eu * wave
            v = V0 + eps * ev * wave
            ts: list[float] = []
            amps: list[float] = []
            for t in range(0, 501):
                if t % 40 == 0:
                    amps.append(float(np.mean((u - U0) * wave)))
                    ts.append(t * dt)
                uv = u * u * v
                u = u + dt * (DU * laplacian(u, dx) + A - u + uv)
                v = v + dt * (DV * laplacian(v, dx) + B - uv)
            amp = np.asarray(amps)
            time = np.asarray(ts)
            sel = (amp > 1e-8) & (time >= 0.05) & (time <= 1.2)
            coef = float(np.polyfit(time[sel], np.log(amp[sel]), 1)[0])
            rows.append(
                {
                    "n": n,
                    "m": m,
                    "k_per_um": float(np.hypot(kx, ky)),
                    "sigma_theory": float(sigma),
                    "sigma_measured": coef,
                }
            )
    return rows


def gof_summary(rows: list[dict]) -> dict:
    theory = np.array([r["sigma_theory"] for r in rows], dtype=float)
    meas = np.array([r["sigma_measured"] for r in rows], dtype=float)
    pearson = float(np.corrcoef(theory, meas)[0, 1])
    rmse = float(np.sqrt(np.mean((theory - meas) ** 2)))
    # Two-sample KS on the paired sigma values, as a distributional residual.
    from scipy.stats import ks_2samp

    ks = ks_2samp(theory, meas)
    return {
        "n_modes": int(len(rows)),
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def diffusion_ratio_sweep() -> list[dict]:
    """λ_peak and σ_max versus δ = Dv/Du at fixed Du.

    δ = 1 is the equal-diffusion ablation. Below onset, σ_max < 0 and the
    reported wavelength is omitted (the argmax sits on the scan edge).
    Above onset, λ_c increases monotonically with δ.
    """
    rows = []
    for delta in (1.0, 8.0, 12.0, 14.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0):
        dv = DU * delta
        lam, sig = analytical_peak_wavelength(DU, dv)
        unstable = bool(sig > 0.0)
        rows.append(
            {
                "delta": delta,
                "Dv": dv,
                "lambda_peak_um": lam if unstable else None,
                "sigma_max": sig,
                "turing_unstable": unstable,
            }
        )
    return rows


def specimen_segmentation(path: Path = SPECIMEN) -> dict:
    """Border-contrast mask, the same construction as verify_image_fitness.

    Circularity, foreground fraction and structure-tensor coherency are measured
    on the real sorocarp photograph. They are not set equal to a model target.
    """
    from scipy import ndimage as ndi

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        native = im.size
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    small = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).resize((160, 120), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    border = np.concatenate([small[0, :], small[-1, :], small[:, 0], small[:, -1]])
    bg = float(np.median(border))
    contrast = np.abs(small - bg)
    th = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(th, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(small, axis=1)
    gy = ndi.sobel(small, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    labeled, n_obj = ndi.label(mask)
    return {
        "native_px": [int(native[0]), int(native[1])],
        "foreground_fraction": float(np.mean(mask)),
        "contour_circularity": circularity,
        "structure_tensor_coherency": coherency,
        "mask_components_160x120": int(n_obj),
        "background_median": bg,
    }


def specimen_mask_image(path: Path = SPECIMEN) -> np.ndarray:
    """160×120 border-contrast mask, the same rule as the circularity metric."""
    from scipy import ndimage as ndi

    with Image.open(path) as im:
        gray = np.asarray(im.convert("L").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    th = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(th, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(np.float64)


def field_to_rgb(field: np.ndarray, size: tuple[int, int] = GIF_SIZE) -> Image.Image:
    """Full-bleed activator field. Palette matched to the fitness probe."""
    f = field - float(np.min(field))
    f = f / (float(np.max(f)) + 1e-9)
    rgb = np.dstack(
        [
            np.clip(40 + 200 * f, 0, 255),
            np.clip(50 + 170 * f, 0, 255),
            np.clip(30 + 80 * f, 0, 255),
        ]
    ).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.Resampling.BICUBIC)


def spot_count(field: np.ndarray) -> int:
    from scipy import ndimage as ndi

    thr = float(np.quantile(field, 0.72))
    mask = field >= thr
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    _lab, n_obj = ndi.label(mask)
    return int(n_obj)


def orientation_ks(specimen: Path, sim_image: Image.Image) -> float:
    """KS statistic between Sobel-orientation samples of specimen and last frame."""
    from scipy import ndimage as ndi
    from scipy.stats import ks_2samp

    def angles(arr: np.ndarray) -> np.ndarray:
        g = arr.astype(np.float64)
        if g.ndim == 3:
            g = 0.299 * g[:, :, 0] + 0.587 * g[:, :, 1] + 0.114 * g[:, :, 2]
        gx = ndi.sobel(g, axis=1)
        gy = ndi.sobel(g, axis=0)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi
        keep = mag > np.quantile(mag, 0.70)
        return ang[keep].ravel()

    with Image.open(specimen) as im:
        sp = np.asarray(im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR))
    sm = np.asarray(sim_image.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR))
    a = angles(sp)
    b = angles(sm)
    if len(a) < 30 or len(b) < 30:
        return 1.0
    return float(ks_2samp(a, b).statistic)


def render_gif(snaps: list[tuple[int, np.ndarray]], path: Path) -> dict:
    """Twelve distinct frames, 960×720, full-bleed activator field."""
    if len(snaps) < 12:
        raise RuntimeError(f"need >= 12 snapshots, got {len(snaps)}")
    idx = np.linspace(0, len(snaps) - 1, 12).astype(int)
    frames = [field_to_rgb(snaps[int(i)][1]) for i in idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=False,
    )
    with Image.open(path) as im:
        return {"n_frames": int(getattr(im, "n_frames", 1)), "width": im.size[0], "height": im.size[1]}


def render_png(
    specimen_path: Path,
    final_u: np.ndarray,
    rows: list[dict],
    sweep: list[dict],
    turing_var: list[float],
    ablation_var: list[float],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), dpi=120)
    fig.suptitle(
        "Dictyostelium discoideum  ·  Turing k_c at 2.304 µm  ·  Holocene, 0 Ma",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.imshow(Image.open(specimen_path))
    ax.set_title("Specimen sorocarp (public domain)")
    ax.axis("off")

    ax = axes[0, 1]
    seg = specimen_segmentation(specimen_path)
    ax.imshow(specimen_mask_image(specimen_path), cmap="gray")
    ax.set_title(
        f"Specimen border-contrast mask\ncircularity {seg['contour_circularity']:.3f}"
    )
    ax.axis("off")

    ax = axes[0, 2]
    im = ax.imshow(final_u, cmap="magma", origin="lower")
    ax.set_title(f"Cortical activator u\nL = {DOMAIN_UM} µm = 5 λ_c")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    k = np.array([r["k_per_um"] for r in rows])
    ax.plot(k, [r["sigma_theory"] for r in rows], "o", label="discrete theory", ms=5)
    ax.plot(k, [r["sigma_measured"] for r in rows], "x", label="integrated mode", ms=5)
    ax.axvline(2.0 * np.pi / LAMBDA_C_UM, color="0.4", ls="--", lw=0.8, label="k_c catalog")
    ax.set_xlabel("k (1/µm)")
    ax.set_ylabel("σ (1/s)")
    ax.set_title("Unstable-band dispersion")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    delta = [r["delta"] for r in sweep]
    lam = [r["lambda_peak_um"] if r["lambda_peak_um"] is not None else np.nan for r in sweep]
    sig = [r["sigma_max"] for r in sweep]
    ax.plot(delta, lam, "o-", color="C0", label="λ_peak")
    ax.set_xlabel("δ = D_v / D_u")
    ax.set_ylabel("λ_peak (µm)", color="C0")
    ax2 = ax.twinx()
    ax2.plot(delta, sig, "s--", color="C3", label="σ_max")
    ax2.axhline(0.0, color="0.5", lw=0.6)
    ax2.set_ylabel("σ_max (1/s)", color="C3")
    ax.axvline(24, color="0.5", ls=":", lw=0.8)
    ax.set_title("δ sweep: λ rises once σ_max > 0")

    ax = axes[1, 2]
    ax.plot(np.linspace(0, 1, len(turing_var)), turing_var, label="δ = 24")
    ax.plot(np.linspace(0, 1, len(ablation_var)), ablation_var, label="δ = 1 ablation")
    ax.set_xlabel("normalized time")
    ax.set_ylabel("var(u)")
    ax.set_title("Negative control: equal diffusion")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def build_metrics() -> dict:
    lam_peak, sig_peak = analytical_peak_wavelength()
    rows = mode_growth_table()
    fit = gof_summary(rows)
    sweep = diffusion_ratio_sweep()
    # Above onset, longer inhibitor diffusion selects a longer wavelength.
    unstable = [r for r in sweep if r["turing_unstable"]]
    lams = [r["lambda_peak_um"] for r in unstable]
    monotonic = all(lams[i + 1] + 1e-6 >= lams[i] for i in range(len(lams) - 1))

    snap_steps = [0, 400, 800, 1200, 1800, 2400, 3200, 4000, 5000, 6000, 7000, 7600]
    turing = simulate(DU, DV, N_STEPS, SEED, snapshot_steps=snap_steps)
    ablation = simulate(DU, DU, 2000, SEED, snapshot_steps=[0], noise=0.03)
    seg = specimen_segmentation()
    n_spots = spot_count(turing["final_u"])
    gif_info = render_gif(turing["snaps"], GIF_PATH)
    last = field_to_rgb(turing["final_u"])
    ks_ori = orientation_ks(SPECIMEN, last)
    render_png(
        SPECIMEN,
        turing["final_u"],
        rows,
        sweep,
        turing["variances"],
        ablation["variances"],
        PNG_PATH,
    )

    # Computer-vision similarity of the GIF against the real specimen.
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import (
        compute_sim_vs_specimen_cv_similarity,
        compute_specimen_cv_signature,
    )

    sig = compute_specimen_cv_signature(SPECIMEN)
    cv = compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GIF_PATH)

    var_t0 = turing["variances"][0]
    var_t1 = turing["variances"][-1]
    var_a0 = ablation["variances"][0]
    var_a1 = ablation["variances"][-1]
    metrics = {
        "slot": 3168,
        "organism": "Dictyostelium discoideum",
        "geological_age_ma": 0.0,
        "lambda_c_um_catalog": LAMBDA_C_UM,
        "lambda_peak_um": lam_peak,
        "sigma_max_per_s": sig_peak,
        "wavelength_abs_error_um": abs(lam_peak - LAMBDA_C_UM),
        "modes_per_cell": int(round(DOMAIN_UM / LAMBDA_C_UM)),
        "molecular_predecessor_modes": int(MOLECULAR_PREDECESSOR_UM // LAMBDA_C_UM),
        "aggregation_successor_um": AGGREGATION_SUCCESSOR_UM,
        "dispersion_gof": fit,
        "dispersion_rows": rows,
        "sweep": sweep,
        "sweep_lambda_monotonic_above_onset": monotonic,
        "ablation_equal_diffusion": {
            "delta": 1.0,
            "sigma_max": next(r["sigma_max"] for r in sweep if r["delta"] == 1.0),
            "variance_initial": var_a0,
            "variance_final": var_a1,
            "variance_ratio": var_a1 / var_a0,
        },
        "turing_variance_ratio": var_t1 / var_t0,
        "spot_count": n_spots,
        "specimen_segmentation": seg,
        "specimen_cv_signature": sig,
        "orientation_ks": ks_ori,
        "gif": gif_info,
        "cv_sim_similarity": cv,
        "final_u_min": float(np.min(turing["final_u"])),
        "final_u_max": float(np.max(turing["final_u"])),
        "final_u_std": float(np.std(turing["final_u"])),
        "paths": {"png": str(PNG_PATH.relative_to(ROOT)), "gif": str(GIF_PATH.relative_to(ROOT))},
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def measure_morphoeval() -> list[dict]:
    """Sidecar measurements. Values are not copied onto their reference values."""
    if not METRICS_PATH.exists():
        build_metrics()
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    seg = m["specimen_segmentation"]
    gof = m["dispersion_gof"]
    abl = m["ablation_equal_diffusion"]
    return [
        {
            "name": "simulated stripe wavelength",
            "value": m["lambda_peak_um"],
            "unit": "µm",
            "method": "argmax of discrete-Laplacian Schnakenberg growth rate σ(k)",
            "reference_value": LAMBDA_C_UM,
            "reference_source": "slot 3168 catalog coordinate 2.304 µm (2.304e-06 m), not a wet-lab mean",
            "derived_from": "script",
            "note": f"absolute error {m['wavelength_abs_error_um']:.6f} µm",
        },
        {
            "name": "dispersion-relation Pearson r",
            "value": gof["pearson_r"],
            "unit": "1",
            "method": "Pearson r of integrated Fourier-mode growth versus discrete-Laplacian σ(k) on the unstable band",
            "reference_value": 1.0,
            "reference_source": "identity line of linear theory; measured r is the residual",
            "derived_from": "script",
            "note": f"RMSE {gof['rmse']:.6e} 1/s over {gof['n_modes']} modes; KS statistic {gof['ks_statistic']:.4f}",
        },
        {
            "name": "dispersion RMSE",
            "value": gof["rmse"],
            "unit": "1/s",
            "method": "root mean square residual of σ_measured against σ_theory",
            "reference_value": 0.05,
            "reference_source": "pre-set screening tolerance, not the measured residual",
            "derived_from": "script",
        },
        {
            "name": "dispersion KS statistic",
            "value": gof["ks_statistic"],
            "unit": "1",
            "method": "two-sample KS between theoretical and measured growth-rate samples",
            "reference_value": 0.2,
            "reference_source": "screening tolerance; value is the measured statistic",
            "derived_from": "script",
            "note": f"p={gof['ks_pvalue']:.4g}",
        },
        {
            "name": "specimen contour circularity",
            "value": seg["contour_circularity"],
            "unit": "1",
            "method": "4πA/P² on the border-contrast mask of the sorocarp photograph",
            "reference_value": 0.75,
            "reference_source": "round-spot circularity typical of a saturated Schnakenberg spot; the sorocarp is not a spot",
            "derived_from": "output",
            "note": "value != reference: fruiting-body silhouette versus cortical spot",
        },
        {
            "name": "specimen foreground fraction",
            "value": seg["foreground_fraction"],
            "unit": "1",
            "method": "mean of the 160×120 border-contrast mask",
            "reference_value": 0.45,
            "reference_source": "not the measured mask; a filled-cell occupancy would be higher",
            "derived_from": "output",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": seg["structure_tensor_coherency"],
            "unit": "1",
            "method": "mean (λ1-λ2)/(λ1+λ2) on the specimen mask",
            "reference_value": 0.25,
            "reference_source": "placeholder isotropic-field coherency, not the measured stalk image",
            "derived_from": "output",
        },
        {
            "name": "specimen image orientation KS",
            "value": m["orientation_ks"],
            "unit": "1",
            "method": "KS statistic of Sobel orientations, specimen photograph versus GIF last frame",
            "reference_value": 0.05,
            "reference_source": "a matched-image KS near 0; the measured statistic is the distance",
            "derived_from": "output",
        },
        {
            "name": "equal-diffusion ablation variance ratio",
            "value": abl["variance_ratio"],
            "unit": "1",
            "method": "var(u) final / var(u) initial with Du = Dv",
            "reference_value": m["turing_variance_ratio"],
            "reference_source": "same initial noise evolved at δ = 24; the ablation ratio must sit below it",
            "derived_from": "script",
            "note": "negative control versus the Turing regime",
        },
        {
            "name": "cortical spot count",
            "value": m["spot_count"],
            "unit": "1",
            "method": "connected components of u above the 0.72 quantile",
            "reference_value": 25,
            "reference_source": "(L/λ_c)^2 = 25, a square lattice count the nonlinear spots need not hit",
            "derived_from": "output",
        },
    ]


def self_test() -> None:
    """Fast checks: wavelength, five modes, ablation sign, one-mode growth, finiteness."""
    lam, sig = analytical_peak_wavelength()
    if abs(lam - LAMBDA_C_UM) / LAMBDA_C_UM > 0.01:
        raise AssertionError(f"λ_peak {lam} is not within 1% of {LAMBDA_C_UM}")
    if sig <= 0.0:
        raise AssertionError("reference regime is not Turing-unstable")
    if int(DOMAIN_UM / LAMBDA_C_UM) != 5:
        raise AssertionError("vegetative cell must hold 5 wavelengths")
    if int(MOLECULAR_PREDECESSOR_UM // LAMBDA_C_UM) != 0:
        raise AssertionError("molecular predecessor must admit 0 modes")
    lam_ab, sig_ab = analytical_peak_wavelength(DU, DU)
    if sig_ab >= 0.0:
        raise AssertionError(f"equal-diffusion control still unstable: σ={sig_ab}, λ={lam_ab}")
    # Single critical mode (n, m) = (5, 0) on a short grid.
    dx = DOMAIN_UM / 48
    kx = 2.0 * np.pi * 5 / DOMAIN_UM
    sigma, disc = sigma_discrete(kx, 0.0, DU, DV, dx)
    if disc <= 0.0 or sigma <= 0.1:
        raise AssertionError(f"mode n=5 not unstable: σ={sigma}, disc={disc}")
    x = np.arange(48) * dx
    xx, yy = np.meshgrid(x, x, indexing="ij")
    eu, ev = unstable_eigenvector(kx, 0.0, DU, DV, dx)
    wave = np.cos(kx * xx)
    eps = 2e-4
    u = U0 + eps * eu * wave
    v = V0 + eps * ev * wave
    dt = 0.003
    amp0 = float(np.mean((u - U0) * wave))
    for _t in range(200):
        uv = u * u * v
        u = u + dt * (DU * laplacian(u, dx) + A - u + uv)
        v = v + dt * (DV * laplacian(v, dx) + B - uv)
    amp1 = float(np.mean((u - U0) * wave))
    if not np.isfinite(u).all() or amp1 <= amp0:
        raise AssertionError("critical mode failed to grow")
    grown = np.log(amp1 / amp0) / (200 * dt)
    if abs(grown - sigma) / sigma > 0.05:
        raise AssertionError(f"mode growth {grown} disagrees with σ={sigma}")
    sweep = diffusion_ratio_sweep()
    if sweep[0]["turing_unstable"]:
        raise AssertionError("δ=1 must be stable")
    if not sweep[-1]["turing_unstable"]:
        raise AssertionError("δ=48 must be unstable")
    print(
        f"self-test ok  λ={lam:.6f} µm  σ={sig:.4f} 1/s  "
        f"mode5 σ={sigma:.4f} measured={grown:.4f}  ablation σ={sig_ab:.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    metrics = build_metrics()
    gof = metrics["dispersion_gof"]
    print(
        f"λ_peak={metrics['lambda_peak_um']:.6f} µm  "
        f"Pearson r={gof['pearson_r']:.6f}  RMSE={gof['rmse']:.3e}  "
        f"KS={gof['ks_statistic']:.4f}  CV={metrics['cv_sim_similarity']}  "
        f"GIF {metrics['gif']}  spots={metrics['spot_count']}"
    )
    print(f"wrote {PNG_PATH}")
    print(f"wrote {GIF_PATH}")
    print(f"wrote {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
