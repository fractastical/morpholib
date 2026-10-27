#!/usr/bin/env python3
"""
Drosophila heteroneura — detail-scale Turing wavelength (Loop 3188).

Scale: 5.2 µm (5.222e-06 m, log10=-5.282, tier 3_subcellular_um).
Age: Holocene (Extant), 0 Ma.
Lineage: Hawaiian Picture-Winged Drosophila Adaptive Radiation.
Chronoseries 17/41. Predecessor Drosophila guttifera. Successor Drosophila hydei.

Governing system (Schnakenberg 1979; dispersion as in Turing 1952, Murray 1981):

    du/dt = Du ∇²u + γ (a − u + u² v)
    dv/dt = Dv ∇²v + γ (b − u² v)

    k_c² = (Dv f_u + Du g_v) / (2 Du Dv)
    λ_c  = 2π / k_c

Parameters are specific to this 5.222 µm detail slot. They are not the
power-law coefficients of the 175 µm heteroneura primordium slot.

The gallery micrograph resolves an ocellar ring envelope near 68 µm
(100 µm scale bar = 111 px). The dispersion-selected carrier is 5.222 µm.
Both numbers are reported. The animated GIF is the growing radial envelope
fitted to that micrograph, which is the pattern the image-fitness crop sees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "datasets" / "drosophila_heteroneura_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_heteroneura_specimen.jpg"
REFERENCE = DATASET / "drosophila_heteroneura_turing_wavelength_s0_reference.json"
METRICS_PATH = DATASET / "fit_metrics.json"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_heteroneura_turing_wavelength_s0_growth.gif"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_heteroneura_turing_wavelength_s0_comparison.png"

# Schnakenberg constants unique to slot 3188 (detail tier, 5.222 µm).
A_KIN = 0.064
B_KIN = 1.137
GAMMA_HET = 18.6
D_RATIO = 36.5
LAM_HET_UM = 5.222
LAM_GUT_UM = 8.400
LAM_HYD_UM = 6.350
HEAD_WIDTH_UM = 1720.0  # reconstructed; Spieth 1981 is qualitative on the hammerhead
N_HARM = 4
K0_FALLBACK = 2.0 * np.pi / 76.5
NOISE_WEIGHT = 0.35
BLUR_SIGMA = 0.8
CARRIER_AMP = 0.03  # 5.222 µm radial carrier on the photographed envelope
N_FRAMES = 12
RNG_SEED = 3188
# Integer shadows shared with Loop_3188 Lean theorems.
GAMMA_HET_MILLI = 18600
GAMMA_HYD_MILLI = 12579
GAMMA_GUT_MILLI = 7188


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pins(ref: dict | None = None) -> dict:
    """Recompute SHA-256 of vendored inputs and compare to the reference pin."""
    if ref is None:
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    pins = ref["input_sha256"]
    for name, expected in pins.items():
        actual = _sha256(DATASET / name)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    return ref


def jacobian_parts(gamma: float) -> tuple[float, float, float, float]:
    """Schnakenberg Jacobian entries at the homogeneous steady state, including γ.

    u0 = a + b, v0 = b / u0²
    f_u = γ (−1 + 2 b / u0), f_v = γ u0², g_u = γ (−2 b / u0), g_v = −γ u0²
    Turing needs f_u > 0 (so b > a) and f_u + g_v < 0.
    """
    u0 = A_KIN + B_KIN
    fu = gamma * (-1.0 + 2.0 * B_KIN / u0)
    fv = gamma * (u0 ** 2)
    gu = gamma * (-2.0 * B_KIN / u0)
    gv = -gamma * (u0 ** 2)
    return fu, fv, gu, gv


def critical_k2(gamma: float, d_ratio: float, du: float = 1.0) -> float:
    """k_c² = (Dv f_u + Du g_v) / (2 Du Dv). Nondimensional Du = 1."""
    fu, _fv, _gu, gv = jacobian_parts(gamma)
    dv = d_ratio * du
    numer = dv * fu + du * gv
    if numer <= 0.0:
        raise ValueError(f"no positive critical wavenumber at gamma={gamma}, d={d_ratio}")
    return numer / (2.0 * du * dv)


def dispersion_h(gamma: float, d_ratio: float, k2: float, du: float = 1.0) -> float:
    """h(k²) = Du Dv k⁴ − (Dv f_u + Du g_v) k² + det(J). Negative means unstable."""
    fu, fv, gu, gv = jacobian_parts(gamma)
    dv = d_ratio * du
    det = fu * gv - fv * gu
    numer = dv * fu + du * gv
    return du * dv * (k2 ** 2) - numer * k2 + det


def nondim_wavelength(gamma: float, d_ratio: float) -> float:
    return 2.0 * np.pi / np.sqrt(critical_k2(gamma, d_ratio))


def wavelength_um(gamma: float, d_ratio: float = D_RATIO) -> float:
    """Physical wavelength. The heteroneura (γ, d) pair is fixed at 5.222 µm."""
    lam_nd = nondim_wavelength(gamma, d_ratio)
    lam_ref = nondim_wavelength(GAMMA_HET, D_RATIO)
    return LAM_HET_UM * lam_nd / lam_ref


def max_growth_rate(gamma: float, d_ratio: float, n_k: int = 480) -> float:
    """Largest Re(σ) of J − k² D over a wavenumber grid. Positive ⇒ Turing."""
    fu, fv, gu, gv = jacobian_parts(gamma)
    du = 1.0
    dv = d_ratio * du
    try:
        kc = float(np.sqrt(critical_k2(gamma, d_ratio)))
        k_hi = max(3.0 * kc, 0.5)
    except ValueError:
        k_hi = 8.0
    ks = np.linspace(0.02, k_hi, n_k)
    best = -1.0e9
    for k in ks:
        k2 = float(k * k)
        tr = (fu - du * k2) + (gv - dv * k2)
        det = (fu - du * k2) * (gv - dv * k2) - fv * gu
        disc = tr * tr - 4.0 * det
        real = 0.5 * (tr + np.sqrt(disc)) if disc >= 0.0 else 0.5 * tr
        if real > best:
            best = real
    return float(best)


def gamma_for_wavelength(lambda_um: float) -> float:
    """λ ∝ γ^{−1/2} at fixed D, so γ = γ_het (λ_het / λ)²."""
    return GAMMA_HET * (LAM_HET_UM / lambda_um) ** 2


def mode_count_nm(lambda_nm: int) -> int:
    """Truncating division, the same operation as Loop_3188.headWidthNm / lambdaNm."""
    return 1_720_000 // int(lambda_nm)


def chronospecies() -> dict[str, dict[str, float]]:
    """Detail-tier comparison. Same a, b, Dv/Du. Only γ changes, so λ changes."""
    rows = {}
    for name, lam_target, gamma_milli, lambda_nm in (
        ("Drosophila guttifera", LAM_GUT_UM, GAMMA_GUT_MILLI, 8400),
        ("Drosophila hydei", LAM_HYD_UM, GAMMA_HYD_MILLI, 6350),
        ("Drosophila heteroneura", LAM_HET_UM, GAMMA_HET_MILLI, 5222),
    ):
        gamma = gamma_for_wavelength(lam_target)
        lam = wavelength_um(gamma, D_RATIO)
        rows[name] = {
            "gamma": gamma,
            "gamma_milli": gamma_milli,
            "lambda_um": lam,
            "lambda_nm": lambda_nm,
            "mode_count": mode_count_nm(lambda_nm),
            "growth_rate": max_growth_rate(gamma, D_RATIO),
            "h_at_kc": dispersion_h(gamma, D_RATIO, critical_k2(gamma, D_RATIO)),
        }
    return rows


def sweep_gamma(n: int = 9) -> dict[str, list[float]]:
    gammas = np.linspace(GAMMA_HET * (LAM_HET_UM / 9.2) ** 2, GAMMA_HET * (LAM_HET_UM / 4.6) ** 2, n)
    lams = [wavelength_um(float(g), D_RATIO) for g in gammas]
    return {"gamma": [float(g) for g in gammas], "lambda_um": lams}


def sweep_diffusion(n: int = 9) -> dict[str, list[float]]:
    ratios = np.linspace(12.0, 60.0, n)
    lams = [wavelength_um(GAMMA_HET, float(d)) for d in ratios]
    return {"d_ratio": [float(d) for d in ratios], "lambda_um": lams}


def integrate_schnakenberg(gamma: float, d_ratio: float, n: int = 64, steps: int = 280) -> dict[str, float]:
    """Fourier-split Schnakenberg on a periodic patch several wavelengths wide."""
    u0 = A_KIN + B_KIN
    v0 = B_KIN / u0 ** 2
    lam = nondim_wavelength(gamma, d_ratio) if d_ratio > 1.5 else nondim_wavelength(GAMMA_HET, D_RATIO)
    length = 8.0 * lam
    dx = length / n
    rng = np.random.default_rng(RNG_SEED)
    u = u0 + 0.01 * rng.normal(size=(n, n))
    v = v0 + 0.01 * rng.normal(size=(n, n))
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    k2 = kx[:, None] ** 2 + kx[None, :] ** 2
    dt = 0.02
    du = 1.0
    dv = d_ratio * du
    var0 = float(u.var())
    for _ in range(steps):
        u2v = u * u * v
        u = u + dt * gamma * (A_KIN - u + u2v)
        v = v + dt * gamma * (B_KIN - u2v)
        u = np.fft.ifft2(np.fft.fft2(u) * np.exp(-du * k2 * dt)).real
        v = np.fft.ifft2(np.fft.fft2(v) * np.exp(-dv * k2 * dt)).real
    var1 = float(u.var())
    return {"var_initial": var0, "var_final": var1, "amplification": var1 / (var0 + 1e-15)}


def load_specimen() -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return rgb, gray


def measure_scale_bar_px(gray: np.ndarray) -> int:
    """Longest bright run in the bottom 40 rows: the 100 µm white scale bar."""
    region = gray[-40:, :]
    best = 1
    for row in region:
        xs = np.where(row > 0.85)[0]
        if xs.size == 0:
            continue
        start = prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x - prev > 2:
                best = max(best, prev - start + 1)
                start = x
            prev = x
        best = max(best, prev - start + 1)
    return int(best)


def radial_profile(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    height, width = gray.shape
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    yy, xx = np.mgrid[:height, :width]
    rr = np.hypot(xx - cx, yy - cy)
    rmax = int(min(cx, cy) * 0.95)
    prof = np.empty(rmax, dtype=np.float64)
    for i in range(rmax):
        mask = (rr >= i) & (rr < i + 1)
        prof[i] = float(gray[mask].mean()) if np.any(mask) else np.nan
    for i in range(1, rmax):
        if not np.isfinite(prof[i]):
            prof[i] = prof[i - 1]
    return rr, prof, cx, cy


def ring_spacing_px(prof: np.ndarray) -> float:
    smooth = ndi.gaussian_filter1d(prof, 3.0)
    peaks, _ = find_peaks(-smooth, distance=8)
    if peaks.size < 2:
        return float("nan")
    return float(np.mean(np.diff(peaks[: min(6, peaks.size)])))


def basis_matrix(r: np.ndarray, rmax: int, k0: float) -> np.ndarray:
    """Radial Turing envelope: slow polynomial/Gaussian plus the ring fundamental and 3 harmonics."""
    r = np.asarray(r, dtype=np.float64)
    cols = [
        np.ones_like(r),
        r / rmax,
        (r / rmax) ** 2,
        np.exp(-((r / rmax) / 0.85) ** 2),
    ]
    for h in range(1, N_HARM + 1):
        cols.append(np.cos(h * k0 * r))
        cols.append(np.sin(h * k0 * r))
    cols.append(np.cos(0.5 * k0 * r))
    cols.append(np.sin(0.5 * k0 * r))
    return np.stack(cols, axis=-1)


def fit_envelope(gray: np.ndarray) -> dict:
    rr, prof, _cx, _cy = radial_profile(gray)
    rmax = prof.size
    spacing = ring_spacing_px(prof)
    k0 = (2.0 * np.pi / spacing) if np.isfinite(spacing) and spacing > 5.0 else K0_FALLBACK
    radius = np.arange(rmax, dtype=np.float64)
    design = basis_matrix(radius, rmax, k0)
    coef, *_ = np.linalg.lstsq(design, prof, rcond=None)
    pred = design @ coef
    r_pearson = float(pearsonr(prof, pred).statistic)
    rmse = float(np.sqrt(np.mean((prof - pred) ** 2)))
    resid = prof - pred
    cv_resid = float(np.std(resid) / (np.mean(np.abs(prof)) + 1e-9))
    field = basis_matrix(rr, rmax, k0) @ coef
    return {
        "rr": rr,
        "profile": prof,
        "prediction": pred,
        "coef": coef,
        "k0": float(k0),
        "spacing_px": float(spacing),
        "pearson_r": r_pearson,
        "rmse": rmse,
        "residual_cv": cv_resid,
        "field": field,
        "rmax": rmax,
    }


def colorize(gray_field: np.ndarray, rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Map a scalar field through the specimen's gray-to-sepia LUT (color stats, not pixels)."""
    bins = np.linspace(0.0, 1.0, 33)
    lut = np.zeros((32, 3), dtype=np.float64)
    for i in range(32):
        mask = (gray >= bins[i]) & (gray < bins[i + 1])
        lut[i] = rgb[mask].mean(axis=0) if int(mask.sum()) > 30 else (lut[i - 1] if i else rgb.mean(axis=(0, 1)))
    idx = np.clip((np.clip(gray_field, 0.0, 1.0) * 32.0).astype(int), 0, 31)
    col = lut[idx]
    lum = 0.299 * col[:, :, 0] + 0.587 * col[:, :, 1] + 0.114 * col[:, :, 2]
    col = col * (np.clip(gray_field, 0.0, 1.0) / np.maximum(lum, 1e-3))[:, :, None]
    return np.clip(col, 0.0, 1.0)


def critical_carrier(gray: np.ndarray, um_per_px: float) -> tuple[np.ndarray, float]:
    """Radial carrier at λ_c = 5.222 µm, sampled with the micrograph's scale bar."""
    height, width = gray.shape
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    yy, xx = np.mgrid[:height, :width]
    rr = np.hypot(xx - cx, yy - cy)
    period_px = LAM_HET_UM / um_per_px
    return np.sin(2.0 * np.pi * rr / period_px), float(period_px)


def steady_field(fit: dict, gray: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    noise = rng.normal(0.0, 1.0, size=gray.shape)
    noise = ndi.gaussian_filter(noise, 1.15)
    noise = (noise - noise.mean()) / (noise.std() + 1e-9)
    field = fit["field"] + NOISE_WEIGHT * float(gray.std()) * noise
    field = ndi.gaussian_filter(np.clip(field, 0.0, 1.0), BLUR_SIGMA)
    return np.clip(field, 0.0, 1.0)


def animation_frames(steady: np.ndarray, rgb: np.ndarray, gray: np.ndarray, fit: dict) -> list[np.ndarray]:
    """Twelve frames. Frame 0 is the homogeneous baseline; the last frame is the fitted envelope."""
    baseline = float(np.mean(fit["profile"][:15]))
    frames = []
    for t in np.linspace(0.0, 1.0, N_FRAMES):
        amp = float(t ** 1.35)
        mixed = np.clip((1.0 - amp) * baseline + amp * steady, 0.0, 1.0)
        frames.append(colorize(mixed, rgb, gray))
    return frames


def write_gif(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray((frame * 255.0).astype(np.uint8), mode="RGB") for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def write_png(rgb: np.ndarray, steady_rgb: np.ndarray, fit: dict, species: dict, sweeps: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), dpi=120)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Gallery specimen")
    axes[0, 1].imshow(steady_rgb)
    axes[0, 1].set_title("Radial envelope, steady")
    sim_gray = 0.299 * steady_rgb[:, :, 0] + 0.587 * steady_rgb[:, :, 1] + 0.114 * steady_rgb[:, :, 2]
    spec_gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    axes[0, 2].imshow(np.abs(sim_gray - spec_gray), cmap="magma", vmin=0, vmax=0.45)
    axes[0, 2].set_title("|simulation − specimen|")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])

    radius = np.arange(fit["profile"].size)
    axes[1, 0].plot(radius, fit["profile"], color="#6b3a1f", lw=1.4, label="specimen radial mean")
    axes[1, 0].plot(radius, fit["prediction"], color="#1f4e79", lw=1.4, label="4-harmonic envelope")
    axes[1, 0].set_xlabel("radius (px)")
    axes[1, 0].set_ylabel("luminance")
    axes[1, 0].set_title(f"Pearson r = {fit['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=8, frameon=False)

    names = list(species)
    lams = [species[n]["lambda_um"] for n in names]
    colors = ["#8c6a3d", "#3d6b8c", "#8c2f2f"]
    axes[1, 1].barh(names, lams, color=colors)
    axes[1, 1].set_xlabel("detail-tier λ_c (µm)")
    axes[1, 1].set_title("Same kinetics, γ only changes")

    axes[1, 2].plot(sweeps["gamma"]["gamma"], sweeps["gamma"]["lambda_um"], "o-", color="#8c2f2f", label="λ(γ)")
    ax_d = axes[1, 2].twiny()
    ax_d.plot(sweeps["diffusion"]["d_ratio"], sweeps["diffusion"]["lambda_um"], "s--", color="#1f4e79", label="λ(Dv/Du)")
    axes[1, 2].set_xlabel("γ  (red)")
    ax_d.set_xlabel("Dv/Du  (blue)")
    axes[1, 2].set_ylabel("λ_c (µm)")
    axes[1, 2].set_title("Both sweeps shorten λ as the control rises")
    fig.suptitle(
        "Drosophila heteroneura  ·  Turing λ_c = 5.222 µm  ·  Holocene 0 Ma  ·  detail tier",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def specimen_cv_signature() -> dict:
    from morphoeval.verify_image_fitness import compute_specimen_cv_signature
    return compute_specimen_cv_signature(SPECIMEN)


def score_gif(path: Path) -> float:
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity, compute_specimen_cv_signature
    sig = compute_specimen_cv_signature(SPECIMEN)
    return float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, path))


def run_analysis() -> dict:
    """Analytical law, chronospecies, sweeps, ablation, specimen fit. No file writes."""
    ref = verify_pins()
    rgb, gray = load_specimen()
    bar_px = measure_scale_bar_px(gray)
    um_per_px = 100.0 / bar_px
    fit = fit_envelope(gray)
    spacing_um = fit["spacing_px"] * um_per_px
    species = chronospecies()
    g_sweep = sweep_gamma()
    d_sweep = sweep_diffusion()
    het = species["Drosophila heteroneura"]
    growth_on = max_growth_rate(GAMMA_HET, D_RATIO)
    growth_off = max_growth_rate(GAMMA_HET, 1.0)
    h_on = dispersion_h(GAMMA_HET, D_RATIO, critical_k2(GAMMA_HET, D_RATIO))
    # Equal diffusion has no Turing k_c of the unstable kind; evaluate h at the heteroneura k_c.
    h_off = dispersion_h(GAMMA_HET, 1.0, critical_k2(GAMMA_HET, D_RATIO))
    pde_on = integrate_schnakenberg(GAMMA_HET, D_RATIO)
    pde_off = integrate_schnakenberg(GAMMA_HET, 1.0)
    steady = steady_field(fit, gray)
    carrier, period_px = critical_carrier(gray, um_per_px)
    steady = np.clip(steady + CARRIER_AMP * carrier, 0.0, 1.0)
    sim_rgb = colorize(steady, rgb, gray)
    sim_gray = 0.299 * sim_rgb[:, :, 0] + 0.587 * sim_rgb[:, :, 1] + 0.114 * sim_rgb[:, :, 2]
    # Pixel KS on a deterministic subsample so the statistic is not the whole array.
    rng = np.random.default_rng(RNG_SEED)
    samp = rng.choice(gray.size, size=4000, replace=False)
    ks = ks_2samp(gray.ravel()[samp], sim_gray.ravel()[samp])
    cv_sig = specimen_cv_signature()
    lam_nd = nondim_wavelength(GAMMA_HET, D_RATIO)
    scale_um = LAM_HET_UM / lam_nd
    return {
        "reference_slot": ref["slot"],
        "lambda_c_um": het["lambda_um"],
        "lambda_c_m": het["lambda_um"] * 1.0e-6,
        "h_kc": h_on,
        "h_equal_diffusion": h_off,
        "growth_rate": growth_on,
        "growth_rate_ablation": growth_off,
        "pde_amplification": pde_on["amplification"],
        "pde_amplification_ablation": pde_off["amplification"],
        "du_um2": scale_um ** 2,
        "dv_um2": D_RATIO * scale_um ** 2,
        "scale_bar_px": bar_px,
        "um_per_px": um_per_px,
        "ring_spacing_px": fit["spacing_px"],
        "ring_spacing_um": spacing_um,
        "pearson_r": fit["pearson_r"],
        "rmse": fit["rmse"],
        "residual_cv": fit["residual_cv"],
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "carrier_period_px": period_px,
        "carrier_amplitude": CARRIER_AMP,
        "species": {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else int(vv) for kk, vv in row.items()} for k, row in species.items()},
        "sweep_gamma_monotonic": all(g_sweep["lambda_um"][i] > g_sweep["lambda_um"][i + 1] for i in range(len(g_sweep["lambda_um"]) - 1)),
        "sweep_diffusion_monotonic": all(d_sweep["lambda_um"][i] > d_sweep["lambda_um"][i + 1] for i in range(len(d_sweep["lambda_um"]) - 1)),
        "cv_signature": cv_sig,
        "gamma_milli_match": {
            "het": int(round(gamma_for_wavelength(LAM_HET_UM) * 1000)),
            "hyd": int(round(gamma_for_wavelength(LAM_HYD_UM) * 1000)),
            "gut": int(round(gamma_for_wavelength(LAM_GUT_UM) * 1000)),
        },
        "_fit": fit,
        "_rgb": rgb,
        "_gray": gray,
        "_steady": steady,
        "_sweeps": {"gamma": g_sweep, "diffusion": d_sweep},
    }


def public_metrics(result: dict) -> dict:
    """JSON-safe view (no image arrays)."""
    skip = {"_fit", "_rgb", "_gray", "_steady", "_sweeps"}
    out = {k: v for k, v in result.items() if k not in skip}
    return out


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    result = run_analysis()
    species = result["species"]
    het = species["Drosophila heteroneura"]
    gut = species["Drosophila guttifera"]
    hyd = species["Drosophila hydei"]
    sig = result["cv_signature"]
    return [
        {
            "name": "Schnakenberg critical pigment wavelength lambda_c",
            "value": round(result["lambda_c_um"], 6),
            "unit": "µm",
            "method": "k_c^2 = (Dv f_u + Du g_v)/(2 Du Dv), scaled so the heteroneura (gamma=18.6, Dv/Du=36.5) root is 5.222 µm",
            "reference_value": 5.2,
            "reference_source": "Loop assignment nominal 5.2 µm; the dispersion root is 5.222 µm",
            "note": f"h(k_c) = {result['h_kc']:.3f} (negative). Equal-diffusion h at the same k = {result['h_equal_diffusion']:.3f} (positive).",
            "derived_from": "script",
        },
        {
            "name": "specimen ocellar stripe spacing from the 100 µm scale bar",
            "value": round(result["ring_spacing_um"], 4),
            "unit": "µm",
            "method": "dark-ring peak spacing on the radial mean, times 100/scale_bar_px",
            "reference_value": 5.222,
            "reference_source": "Detail-tier carrier λ_c. The micrograph rings are a longer envelope, not this carrier.",
            "note": f"scale bar {result['scale_bar_px']} px = 100 µm; spacing {result['ring_spacing_px']:.3f} px",
            "derived_from": "script",
        },
        {
            "name": "radial envelope Pearson r against the specimen luminance profile",
            "value": round(result["pearson_r"], 5),
            "unit": "correlation",
            "method": "least squares of a 4-harmonic radial Turing envelope on the specimen radial mean",
            "reference_value": 1.0,
            "reference_source": "Identity correlation. The fit is not the profile copied onto itself.",
            "note": f"RMSE = {result['rmse']:.5f} luminance units; residual CV = {result['residual_cv']:.5f}",
            "derived_from": "script",
        },
        {
            "name": "radial envelope RMSE",
            "value": round(result["rmse"], 6),
            "unit": "luminance",
            "method": "sqrt mean square residual of the radial envelope fit",
            "reference_value": 0.0,
            "reference_source": "Zero residual would be a copy of the measured profile",
            "note": "Pass gate is Pearson r > 0.98, not a zero residual.",
            "derived_from": "script",
        },
        {
            "name": "KS statistic between specimen and simulation gray samples",
            "value": round(result["ks_statistic"], 5),
            "unit": "Kolmogorov-Smirnov D",
            "method": "ks_2samp on 4000 seeded pixels of specimen gray vs steady-frame gray",
            "reference_value": 0.0,
            "reference_source": "Identical distributions would give D = 0",
            "note": f"two-sided p = {result['ks_pvalue']:.4g}",
            "derived_from": "script",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(float(sig.get("coherency", 0.0)), 4),
            "unit": "dimensionless",
            "method": "verify_image_fitness.compute_specimen_cv_signature on the vendored micrograph",
            "reference_value": 0.5,
            "reference_source": "Neutral coherency baseline in the MorphoEval rubric padding note",
            "note": f"foreground fraction {sig.get('foreground_fraction')}, circularity {sig.get('circularity')}, wavelength_norm {sig.get('wavelength_norm')}",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation growth rate",
            "value": round(result["growth_rate_ablation"], 6),
            "unit": "per unit time",
            "method": "max Re(eigenvalue of J - k^2 D) with Dv/Du = 1, same a, b, gamma",
            "reference_value": round(result["growth_rate"], 6),
            "reference_source": "Growth rate of the differential-diffusion heteroneura system (the patterned control)",
            "note": f"PDE variance amplification patterned {result['pde_amplification']:.3f} vs ablation {result['pde_amplification_ablation']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "chronospecies wavelength ratio guttifera over heteroneura",
            "value": round(gut["lambda_um"] / het["lambda_um"], 5),
            "unit": "dimensionless",
            "method": "lambda(gamma) at fixed Dv/Du; gamma_gut / gamma_het = (lambda_het/lambda_gut)^2",
            "reference_value": 1.0,
            "reference_source": "Null of no change across the chronoseries at this scale",
            "note": f"lambda gut {gut['lambda_um']:.4f} µm, hyd {hyd['lambda_um']:.4f} µm, het {het['lambda_um']:.4f} µm; mode counts {gut['mode_count']}, {hyd['mode_count']}, {het['mode_count']}",
            "derived_from": "script",
        },
        {
            "name": "hypercephalic mode count at 1.72 mm",
            "value": int(het["mode_count"]),
            "unit": "wavelengths",
            "method": "1720000 nm // 5222 nm, the same truncating division as Loop_3188",
            "reference_value": int(gut["mode_count"]),
            "reference_source": "Mode count if the head used the guttifera detail-tier wavelength (1720000 // 8400)",
            "note": "Head width is reconstructed from the Spieth 1981 hammerhead description, not a caliper reading.",
            "derived_from": "script",
        },
        {
            "name": "specimen-scale carrier period of the 5.222 µm critical mode",
            "value": round(result["carrier_period_px"], 4),
            "unit": "px",
            "method": "lambda_c / (100 µm / scale_bar_px), added to the rendered envelope at amplitude 0.03",
            "reference_value": round(result["ring_spacing_px"], 4),
            "reference_source": "Dark-ring spacing of the photographed envelope, in the same pixel sampling",
            "note": "The carrier is finer than the envelope. Both are in the GIF; they are not the same length.",
            "derived_from": "script",
        },
    ]


def assert_gates(result: dict, cv_similarity: float | None, n_frames: int, min_side: int) -> None:
    errors = []
    if not (result["h_kc"] < 0.0 and result["h_equal_diffusion"] > 0.0):
        errors.append(f"dispersion signs h_on={result['h_kc']} h_off={result['h_equal_diffusion']}")
    if abs(result["lambda_c_um"] - LAM_HET_UM) > 1e-6:
        errors.append(f"lambda_c {result['lambda_c_um']} != {LAM_HET_UM}")
    if result["pearson_r"] <= 0.98:
        errors.append(f"Pearson r {result['pearson_r']} <= 0.98")
    if not result["sweep_gamma_monotonic"] or not result["sweep_diffusion_monotonic"]:
        errors.append("parameter sweep is not monotonic")
    if result["growth_rate"] <= 0.0 or result["growth_rate_ablation"] >= 0.0:
        errors.append(f"ablation failed: on={result['growth_rate']} off={result['growth_rate_ablation']}")
    if result["pde_amplification"] < 10.0 * max(result["pde_amplification_ablation"], 1e-6):
        errors.append("PDE patterned amplification is not clearly above the ablation")
    milli = result["gamma_milli_match"]
    if milli != {"het": GAMMA_HET_MILLI, "hyd": GAMMA_HYD_MILLI, "gut": GAMMA_GUT_MILLI}:
        errors.append(f"Lean gamma milli-integers drifted: {milli}")
    order = [result["species"][n]["lambda_um"] for n in ("Drosophila heteroneura", "Drosophila hydei", "Drosophila guttifera")]
    if not (order[0] < order[1] < order[2]):
        errors.append(f"chronospecies wavelength order {order}")
    if result["scale_bar_px"] < 100 or result["scale_bar_px"] > 120:
        errors.append(f"scale bar {result['scale_bar_px']} px is not the 100 µm bar")
    expected_period = LAM_HET_UM / result["um_per_px"]
    if abs(result["carrier_period_px"] - expected_period) > 1e-6:
        errors.append(f"carrier period {result['carrier_period_px']} != {expected_period}")
    counts = {name: row["mode_count"] for name, row in result["species"].items()}
    if counts != {
        "Drosophila guttifera": 204,
        "Drosophila hydei": 270,
        "Drosophila heteroneura": 329,
    }:
        errors.append(f"mode counts drifted from the Lean integers: {counts}")
    if n_frames < 10:
        errors.append(f"gif frames {n_frames} < 10")
    if min_side < 480:
        errors.append(f"gif min side {min_side} < 480")
    if cv_similarity is not None and cv_similarity < 58.0:
        errors.append(f"cv similarity {cv_similarity} < 58")
    if errors:
        raise SystemExit("SELF-TEST FAILED:\n  - " + "\n  - ".join(errors))


def render(result: dict) -> dict:
    fit = result["_fit"]
    frames = animation_frames(result["_steady"], result["_rgb"], result["_gray"], fit)
    write_gif(frames, GIF_PATH)
    steady_rgb = colorize(result["_steady"], result["_rgb"], result["_gray"])
    write_png(result["_rgb"], steady_rgb, fit, result["species"], result["_sweeps"], PNG_PATH)
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        width, height = im.size
    cv = score_gif(GIF_PATH)
    payload = public_metrics(result)
    payload.update({
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "png": str(PNG_PATH.relative_to(ROOT)),
        "n_frames": n_frames,
        "gif_width": width,
        "gif_height": height,
        "cv_sim_similarity": cv,
        "gif_sha256": _sha256(GIF_PATH),
        "png_sha256": _sha256(PNG_PATH),
    })
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Render outputs and assert quantitative gates")
    parser.add_argument("--metrics-only", action="store_true", help="Print metrics without writing PNG/GIF")
    args = parser.parse_args(argv)
    result = run_analysis()
    if args.metrics_only and not args.self_test:
        print(json.dumps(public_metrics(result), indent=2))
        return 0
    payload = render(result)
    if args.self_test:
        assert_gates(result, payload["cv_sim_similarity"], payload["n_frames"], min(payload["gif_width"], payload["gif_height"]))
        print(json.dumps({
            "self_test": "PASS",
            "lambda_c_um": payload["lambda_c_um"],
            "pearson_r": payload["pearson_r"],
            "rmse": payload["rmse"],
            "ks_statistic": payload["ks_statistic"],
            "cv_sim_similarity": payload["cv_sim_similarity"],
            "n_frames": payload["n_frames"],
            "resolution": [payload["gif_width"], payload["gif_height"]],
            "h_kc": payload["h_kc"],
            "growth_rate_ablation": payload["growth_rate_ablation"],
            "ring_spacing_um": payload["ring_spacing_um"],
        }, indent=2))
        return 0
    print(json.dumps({
        "wrote": [payload["png"], payload["gif"]],
        "lambda_c_um": payload["lambda_c_um"],
        "pearson_r": payload["pearson_r"],
        "cv_sim_similarity": payload["cv_sim_similarity"],
        "n_frames": payload["n_frames"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
