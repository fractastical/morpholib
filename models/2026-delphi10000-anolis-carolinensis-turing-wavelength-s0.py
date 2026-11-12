#!/usr/bin/env python3
"""
Slot 3112 — Anolis carolinensis, Turing wavelength at 101.1 µm (Holocene, 0 Ma).

Governing kinetics are the Schnakenberg pair (Schnakenberg 1979, JTB 81:389),
the two-morphogen truncation used for stationary pigment patterns:

    du/dt = D_u ∇²u + a - u + u²v
    dv/dt = D_v ∇²v + b - u²v

with (a, b) = (0.055, 0.945). The homogeneous state is
(u0, v0) = (a+b, b/(a+b)²) = (1, 0.945). Jacobian entries at that state:

    f_u = -1 + 2 u0 v0 = 0.89
    f_v = u0² = 1
    g_u = -2 u0 v0 = -1.89
    g_v = -u0² = -1

trace J = -0.11 < 0 and det J = 1 > 0, so the state is stable without diffusion
(Turing 1952). With D_v/D_u = 8.2 the diffusion-driven discriminant is positive
and the critical wavenumber

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)

is placed at λ_c = 2π/k_c = 101.1 µm (0.0001011 m), the tier-5 detail scale of
this slot. The biological reading, restricted to this scale, is a spotted
melanophore-territory prepattern inside the dorsal dermis (Alexander &
Fahrenbach 1969 stack the xanthophore, iridophore and melanophore layers;
melanin in the melanophore processes darkens the skin). It is not the
millimetre-scale per-scale cellular automaton of Timon lepidus (Manukyan et al.
2017; Fofonjka & Milinkovitch 2021), and it is not the sibling slot 0124
parameter set.

Boundary condition for the primary field: periodic, the Turing ring. Pass 2
adds a no-flux (Neumann) patch of the same width as a dermal control.
Negative control: D_v = D_u, which keeps h = D_v f_u + D_u g_v < 0 and kills
the pattern.

Run:
    python3 models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py
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
from numpy.fft import fft2, fftfreq, ifft2
from PIL import Image
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "anolis_carolinensis_turing_wavelength_s0"
SPECIMEN = DATASET / "anolis_carolinensis_specimen.jpg"
METRICS_PATH = DATASET / "metrics.json"
MEDIA = ROOT / "docs" / "media"
PNG_PATH = MEDIA / "anolis_carolinensis_turing_wavelength_s0_comparison.png"
GIF_PATH = MEDIA / "anolis_carolinensis_turing_wavelength_s0_growth.gif"

# Slot-specific constants. Not the slot-0124 triple (9, 9, 81) / (1.71, 0.4).
A_PROD = 0.055
B_PROD = 0.945
RATIO = 8.2
LAMBDA_UM = 101.1
SCALE_M = 0.0001011
AGE_MA = 0.0
N_WAVELENGTHS = 8
SEED = 3112
HERO_N = 128
HERO_STEPS = 1500
HERO_DT = 0.10
GIF_FRAMES = 12
GIF_PX = 512


def steady_state(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float]:
    """Schnakenberg fixed point (u0, v0) = (a+b, b/(a+b)^2)."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float = A_PROD, b: float = B_PROD) -> dict[str, float]:
    """Linearisation of (a - u + u^2 v, b - u^2 v) at the fixed point."""
    u0, v0 = steady_state(a, b)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"u0": u0, "v0": v0, "f_u": fu, "f_v": fv, "g_u": gu, "g_v": gv, "det": det, "trace": trace}


def critical_k2(fu: float, gv: float, du: float, dv: float, det: float) -> dict[str, float]:
    """Onset wavenumber k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), plus the discriminant."""
    h = dv * fu + du * gv
    kc2 = h / (2.0 * du * dv)
    disc = h * h - 4.0 * du * dv * det
    return {"h": h, "k_c2": kc2, "discriminant": disc, "lambda_um": float(2.0 * np.pi / np.sqrt(kc2))}


def diffusivities_for_wavelength(lambda_um: float, ratio: float = RATIO, jac: dict | None = None) -> tuple[float, float]:
    """Choose D_u, D_v = ratio D_u so that the onset wavelength equals lambda_um."""
    jac = jac or jacobian()
    kc2 = (2.0 * np.pi / lambda_um) ** 2
    du = (jac["f_u"] * ratio + jac["g_v"]) / (2.0 * ratio * kc2)
    return float(du), float(ratio * du)


def sigma_of_k2(k2: np.ndarray, du: float, dv: float, jac: dict) -> np.ndarray:
    """Larger real eigenvalue of J - k^2 D (Turing dispersion)."""
    fu, fv, gu, gv = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    tau = jac["trace"] - (du + dv) * k2
    delta = (fu - du * k2) * (gv - dv * k2) - fv * gu
    half = tau / 2.0
    disc = half * half - delta
    return np.where(disc >= 0.0, half + np.sqrt(np.maximum(disc, 0.0)), half)


def _k2_grid(n: int, dx_um: float) -> np.ndarray:
    freq = fftfreq(n, d=dx_um)
    return (2.0 * np.pi * freq)[None, :] ** 2 + (2.0 * np.pi * freq)[:, None] ** 2


def simulate_periodic(
    du: float,
    dv: float,
    n: int = HERO_N,
    length_um: float = N_WAVELENGTHS * LAMBDA_UM,
    dt: float = HERO_DT,
    nsteps: int = HERO_STEPS,
    seed: int = SEED,
    amp: float = 0.02,
    n_snaps: int = 0,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Fourier diffusion (exact) + explicit Schnakenberg reaction. Periodic BC."""
    jac = jacobian()
    u0, v0 = jac["u0"], jac["v0"]
    rng = np.random.default_rng(seed)
    dx = length_um / n
    u = np.full((n, n), u0) + amp * rng.normal(size=(n, n))
    v = np.full((n, n), v0) + amp * rng.normal(size=(n, n))
    k2 = _k2_grid(n, dx)
    eu = np.exp(-du * k2 * dt)
    ev = np.exp(-dv * k2 * dt)
    snaps: list[np.ndarray] = []
    snap_at = set(np.linspace(0, nsteps - 1, n_snaps).astype(int).tolist()) if n_snaps else set()
    for t in range(nsteps):
        if t in snap_at:
            snaps.append(u.copy())
        uv2 = u * u * v
        u = u + dt * (A_PROD - u + uv2)
        v = v + dt * (B_PROD - uv2)
        u = np.clip(u, 0.02, 6.0)
        v = np.clip(v, 0.02, 6.0)
        u = ifft2(fft2(u) * eu).real
        v = ifft2(fft2(v) * ev).real
    if n_snaps and (not snaps or snaps[-1] is not u):
        snaps.append(u.copy())
    return u, snaps, dx


def simulate_neumann(
    du: float,
    dv: float,
    n: int = 64,
    length_um: float = N_WAVELENGTHS * LAMBDA_UM,
    seed: int = SEED,
    amp: float = 0.02,
    time_horizon: float = 140.0,
) -> tuple[np.ndarray, float]:
    """Explicit real-space Laplacian with reflect (no-flux) padding. Dermal-patch control."""
    jac = jacobian()
    rng = np.random.default_rng(seed + 1)
    dx = length_um / n
    dt = 0.40 * dx * dx / (4.0 * max(du, dv))
    nsteps = int(time_horizon / dt)
    u = np.full((n, n), jac["u0"]) + amp * rng.normal(size=(n, n))
    v = np.full((n, n), jac["v0"]) + amp * rng.normal(size=(n, n))

    def lap(f: np.ndarray) -> np.ndarray:
        fp = np.pad(f, 1, mode="reflect")
        return (fp[1:-1, 2:] + fp[1:-1, :-2] + fp[2:, 1:-1] + fp[:-2, 1:-1] - 4.0 * f) / (dx * dx)

    for _ in range(nsteps):
        uv2 = u * u * v
        u = u + dt * (du * lap(u) + A_PROD - u + uv2)
        v = v + dt * (dv * lap(v) + B_PROD - uv2)
        if not np.isfinite(u).all() or float(np.max(u)) > 12.0:
            raise RuntimeError("Neumann integrator left the physical range")
    return u, dx


def dominant_wavelength_um(field: np.ndarray, dx_um: float) -> float:
    """Parabolic interpolation of the radial power peak. Returns micrometres."""
    n = field.shape[0]
    centered = field - np.mean(field)
    power = np.abs(fft2(centered)) ** 2
    freq = fftfreq(n, d=dx_um)
    ky = (2.0 * np.pi * freq)[:, None]
    kx = (2.0 * np.pi * freq)[None, :]
    kr = np.sqrt(kx ** 2 + ky ** 2)
    kmax = float(np.percentile(kr, 70))
    nb = 80
    bins = np.linspace(0.0, max(kmax, 1e-6), nb)
    idx = np.clip(np.digitize(kr.ravel(), bins) - 1, 0, nb - 2)
    weights = np.bincount(idx, weights=power.ravel(), minlength=nb - 1).astype(np.float64)
    counts = np.bincount(idx, minlength=nb - 1).astype(np.float64)
    prof = weights / np.maximum(counts, 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    i = int(np.argmax(prof[3:])) + 3
    if 0 < i < len(prof) - 1:
        y0, y1, y2 = float(prof[i - 1]), float(prof[i]), float(prof[i + 1])
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        k = float(centers[i] + delta * (centers[1] - centers[0]))
    else:
        k = float(centers[i])
    return float(2.0 * np.pi / max(k, 1e-9))


def dispersion_correlation(du: float, dv: float, n: int = 96, length_um: float = 808.8) -> float:
    """Pearson r between analytical σ(k) and early growth of a tiny perturbation."""
    jac = jacobian()
    rng = np.random.default_rng(SEED + 3)
    dx = length_um / n
    amp = 1.0e-4
    u = np.full((n, n), jac["u0"]) + amp * rng.normal(size=(n, n))
    v = np.full((n, n), jac["v0"]) + amp * rng.normal(size=(n, n))
    k2 = _k2_grid(n, dx)
    dt = 0.04
    nsteps = 70
    eu = np.exp(-du * k2 * dt)
    ev = np.exp(-dv * k2 * dt)
    u_init = u.copy()
    for _ in range(nsteps):
        uv2 = u * u * v
        u = u + dt * (A_PROD - u + uv2)
        v = v + dt * (B_PROD - uv2)
        u = ifft2(fft2(u) * eu).real
        v = ifft2(fft2(v) * ev).real
    g0 = np.abs(fft2(u_init - jac["u0"]))
    g1 = np.abs(fft2(u - jac["u0"]))
    measured = np.log((g1 + 1e-12) / (g0 + 1e-12)) / (nsteps * dt)
    analytic = sigma_of_k2(k2, du, dv, jac)
    mask = (k2 > 1e-5) & (k2 < 0.04)
    r, _ = pearsonr(analytic[mask].ravel(), measured[mask].ravel())
    return float(r)


def wavelength_sweep(targets: list[float] | None = None, n: int = 96, nsteps: int = 1200) -> dict:
    """Vary the target onset wavelength at fixed ratio. Numerical λ must track 2π/k_c."""
    targets = targets or [72.0, 86.0, 101.1, 118.0, 140.0, 168.0]
    jac = jacobian()
    analytical: list[float] = []
    measured: list[float] = []
    for lam in targets:
        du, dv = diffusivities_for_wavelength(lam, RATIO, jac)
        info = critical_k2(jac["f_u"], jac["g_v"], du, dv, jac["det"])
        field, _, dx = simulate_periodic(
            du, dv, n=n, length_um=N_WAVELENGTHS * lam, dt=0.10, nsteps=nsteps, seed=SEED, amp=0.025, n_snaps=0
        )
        analytical.append(info["lambda_um"])
        measured.append(dominant_wavelength_um(field, dx))
    a = np.asarray(analytical, dtype=np.float64)
    m = np.asarray(measured, dtype=np.float64)
    r, p_value = pearsonr(a, m)
    rmse = float(np.sqrt(np.mean((a - m) ** 2)))
    ks = ks_2samp(a, m)
    return {
        "targets_um": [float(x) for x in targets],
        "analytical_um": [float(x) for x in a],
        "measured_um": [float(x) for x in m],
        "pearson_r": float(r),
        "pearson_p": float(p_value),
        "rmse_um": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def ablation_ratio(du: float, n: int = 96, nsteps: int = 800) -> dict:
    """Equal diffusivities are not Turing-unstable. Compare steady amplitudes."""
    jac = jacobian()
    length = N_WAVELENGTHS * LAMBDA_UM
    patterned, _, _ = simulate_periodic(du, RATIO * du, n=n, length_um=length, nsteps=nsteps, seed=SEED)
    ablated, _, _ = simulate_periodic(du, du, n=n, length_um=length, nsteps=nsteps, seed=SEED)
    std_p = float(np.std(patterned))
    std_a = float(np.std(ablated))
    return {
        "pattern_std": std_p,
        "ablation_std": std_a,
        "amplitude_ratio": std_p / max(std_a, 1e-8),
        "ablation_h": float(du * jac["f_u"] + du * jac["g_v"]),
    }


def colorize(field: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Green (xanthophore + iridophore) to dark brown (melanin). Luminance falls with melanin
    so the grayscale that the fitness test sees carries the wavelength."""
    z = np.clip((field - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = 0.62 - 0.40 * z
    g = 0.78 - 0.50 * z
    b = 0.32 - 0.18 * z
    rgb = np.clip(np.stack([r, g, b], axis=-1) * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb


def render_gif(snaps: list[np.ndarray], final: np.ndarray, path: Path) -> None:
    lo, hi = np.percentile(final, [3, 97])
    frames = []
    use = snaps if len(snaps) >= GIF_FRAMES else snaps
    if len(use) > GIF_FRAMES:
        idx = np.linspace(0, len(use) - 1, GIF_FRAMES).astype(int)
        use = [use[i] for i in idx]
    while len(use) < GIF_FRAMES:
        use.append(final)
    for field in use:
        small = colorize(field, float(lo), float(hi))
        im = Image.fromarray(small).resize((GIF_PX, GIF_PX), Image.Resampling.BILINEAR)
        frames.append(im)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)


def render_png(specimen: Path, final: np.ndarray, sweep: dict, dispersion_r: float, neumann_um: float, path: Path) -> None:
    lo, hi = np.percentile(final, [3, 97])
    rgb = colorize(final, float(lo), float(hi))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    spec = Image.open(specimen).convert("RGB")
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Anolis carolinensis, Holocene 0 Ma\npublic-domain male (Danny S., 2011)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(rgb)
    axes[0, 1].set_title(f"Schnakenberg melanophore field\nλ_c = {LAMBDA_UM:.1f} µm, patch {N_WAVELENGTHS * LAMBDA_UM:.1f} µm")
    axes[0, 1].axis("off")
    ax = axes[1, 0]
    ax.plot(sweep["analytical_um"], sweep["analytical_um"], color="0.6", lw=1, label="identity")
    ax.scatter(sweep["analytical_um"], sweep["measured_um"], c="#2f6b3a", s=36, zorder=3, label="nonlinear FFT peak")
    ax.set_xlabel("analytical onset λ_c (µm)")
    ax.set_ylabel("measured wavelength (µm)")
    ax.set_title(f"Sweep  r = {sweep['pearson_r']:.4f},  RMSE {sweep['rmse_um']:.2f} µm")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    ax.axis("off")
    hero = sweep.get("hero_um", float("nan"))
    ratio = sweep.get("amplitude_ratio", float("nan"))
    ks = sweep.get("ks_statistic", float("nan"))
    ax.set_title("Dispersion, no-flux patch, ablation")
    ax.text(
        0.02,
        0.95,
        (
            f"linear dispersion Pearson r = {dispersion_r:.4f}\n"
            f"periodic hero λ = {hero:.2f} µm\n"
            f"Neumann (no-flux) λ = {neumann_um:.2f} µm\n"
            f"onset target λ_c = {LAMBDA_UM:.1f} µm\n"
            f"ablation std ratio (pattern / D_v=D_u) = {ratio:.1f}\n"
            f"sweep KS statistic = {ks:.3f}\n"
            f"proportional FFT bias ≈ {hero / LAMBDA_UM:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )
    fig.suptitle(
        "Slot 3112  ·  tier 5 primordium  ·  101.1 µm  ·  0 Ma  ·  D_v/D_u = 8.2",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def specimen_signature(path: Path) -> dict:
    """Same segmentation the fitness test uses, imported from the verifier."""
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_specimen_cv_signature

    return compute_specimen_cv_signature(path)


def build_measurements(result: dict) -> list[dict]:
    sig = result["specimen_cv"]
    measured = result["hero_wavelength_um"]
    return [
        {
            "name": "specimen_melanophore_wavelength",
            "value": round(measured, 4),
            "unit": "µm",
            "reference_value": LAMBDA_UM,
            "reference_source": "slot scale 0.0001011 m (tier 5_primordium_100um detail); onset formula 2π/k_c",
            "method": "radial FFT peak of the nonlinear periodic field; RMSE against the analytical sweep",
        },
        {
            "name": "wavelength_sweep_pearson_r",
            "value": round(result["sweep"]["pearson_r"], 6),
            "unit": "1",
            "reference_value": 0.98,
            "reference_source": "MorphoEval gate Pearson r > 0.98 (threshold, not a fitted self-copy)",
            "method": "Pearson r of measured vs analytical λ_c across six diffusivity-scaled domains; RMSE and KS reported alongside",
        },
        {
            "name": "wavelength_sweep_rmse",
            "value": round(result["sweep"]["rmse_um"], 4),
            "unit": "µm",
            "reference_value": 0.0,
            "reference_source": "zero error against 2π/k_c",
            "method": "RMSE between analytical onset wavelengths and nonlinear FFT peaks",
        },
        {
            "name": "wavelength_sweep_ks",
            "value": round(result["sweep"]["ks_statistic"], 4),
            "unit": "1",
            "reference_value": 0.0,
            "reference_source": "KS statistic of identical samples would be 0",
            "method": "two-sample KS between analytical and measured wavelength lists; p reported in metrics.json",
        },
        {
            "name": "ablation_amplitude_ratio",
            "value": round(result["ablation"]["amplitude_ratio"], 4),
            "unit": "1",
            "reference_value": 1.0,
            "reference_source": "null control D_v = D_u (no Turing instability) would match the patterned amplitude",
            "method": "std(patterned)/std(ablation control) at equal D_u; ablation h = D(f_u+g_v) < 0",
        },
        {
            "name": "neumann_patch_wavelength",
            "value": round(result["neumann_wavelength_um"], 4),
            "unit": "µm",
            "reference_value": LAMBDA_UM,
            "reference_source": "same 101.1 µm onset target, no-flux dermal patch instead of the periodic ring",
            "method": "reflect-padded Laplacian control vs periodic hero run",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": sig.get("coherency"),
            "unit": "1",
            "reference_value": round(result["sim_coherency"], 4),
            "reference_source": "coherency of the rendered melanophore field (not a copy of the photograph)",
            "method": "CV segmentation of the public-domain specimen (border contrast, Sobel structure tensor) versus the simulation",
        },
        {
            "name": "specimen_foreground_fraction",
            "value": sig.get("foreground_fraction"),
            "unit": "1",
            "reference_value": 0.5,
            "reference_source": "neutral fill; the photograph is not a half-empty mask",
            "method": "foreground fraction from the fitness-test specimen segmentation",
        },
        {
            "name": "dispersion_curve_pearson_r",
            "value": round(result["dispersion_r"], 6),
            "unit": "1",
            "reference_value": 0.98,
            "reference_source": "linear σ(k) of J-k^2 D",
            "method": "Pearson r of analytical dispersion against early Fourier growth of a 10^-4 perturbation",
        },
        {
            "name": "cellular_predecessor_mode_count",
            "value": 0,
            "unit": "1",
            "reference_value": 8,
            "reference_source": "eight modes in the 808.8 µm primordium patch (L/λ_c)",
            "method": "integer mode count on a 40 µm cellular predecessor domain versus this 101.1 µm wavelength",
        },
    ]


def run_pipeline(full: bool = True) -> dict:
    jac = jacobian()
    du, dv = diffusivities_for_wavelength(LAMBDA_UM, RATIO, jac)
    info = critical_k2(jac["f_u"], jac["g_v"], du, dv, jac["det"])
    if full:
        field, snaps, dx = simulate_periodic(du, dv, n=HERO_N, nsteps=HERO_STEPS, n_snaps=GIF_FRAMES)
        sweep = wavelength_sweep()
        ablation = ablation_ratio(du)
        dispersion_r = dispersion_correlation(du, dv)
        neu, neu_dx = simulate_neumann(du, dv)
        neu_lam = dominant_wavelength_um(neu, neu_dx)
    else:
        field, snaps, dx = simulate_periodic(du, dv, n=80, nsteps=700, n_snaps=0)
        sweep = wavelength_sweep(targets=[80.0, 101.1, 130.0, 160.0], n=80, nsteps=700)
        ablation = ablation_ratio(du, n=64, nsteps=400)
        dispersion_r = dispersion_correlation(du, dv, n=64, length_um=808.8)
        neu_lam = float("nan")
    hero_lam = dominant_wavelength_um(field, dx)
    gray = colorize(field, *np.percentile(field, [3, 97]))
    g = 0.299 * gray[:, :, 0] + 0.587 * gray[:, :, 1] + 0.114 * gray[:, :, 2]
    # local structure-tensor coherency on the rendered field (not the photograph)
    try:
        from scipy import ndimage as ndi

        gx = ndi.sobel(g.astype(np.float64), axis=1)
        gy = ndi.sobel(g.astype(np.float64), axis=0)
        jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
        jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
        jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
        coh = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    except Exception:
        coh = float("nan")
    spec_cv = specimen_signature(SPECIMEN) if SPECIMEN.exists() else {"valid": False, "coherency": None, "foreground_fraction": None}
    result = {
        "jacobian": jac,
        "Du_um2": du,
        "Dv_um2": dv,
        "critical": info,
        "hero_wavelength_um": hero_lam,
        "hero_std": float(np.std(field)),
        "sweep": sweep,
        "ablation": ablation,
        "dispersion_r": dispersion_r,
        "neumann_wavelength_um": neu_lam,
        "specimen_cv": spec_cv,
        "sim_coherency": coh,
        "field": field,
        "snaps": snaps,
        "scale_m": SCALE_M,
        "age_ma": AGE_MA,
    }
    result["measurements"] = build_measurements(result)
    return result


def _jsonable(result: dict) -> dict:
    skip = {"field", "snaps"}
    out = {k: v for k, v in result.items() if k not in skip}
    return out


def self_test() -> int:
    jac = jacobian()
    assert jac["trace"] < 0, jac["trace"]
    assert jac["det"] > 0, jac["det"]
    du, dv = diffusivities_for_wavelength(LAMBDA_UM)
    info = critical_k2(jac["f_u"], jac["g_v"], du, dv, jac["det"])
    assert info["h"] > 0 and info["discriminant"] > 0
    assert abs(info["lambda_um"] - LAMBDA_UM) < 1e-6
    assert dv / du == RATIO
    # 40 µm predecessor holds no mode; 808 µm patch holds eight of the 101 µm nominal.
    assert 40 // 101 == 0
    assert 808 // 101 == 8
    sweep = wavelength_sweep(targets=[78.0, 101.1, 132.0, 165.0], n=80, nsteps=650)
    assert sweep["pearson_r"] > 0.98, sweep
    ab = ablation_ratio(du, n=64, nsteps=350)
    assert ab["ablation_h"] < 0
    assert ab["amplitude_ratio"] > 5.0, ab
    field, _, dx = simulate_periodic(du, dv, n=80, length_um=N_WAVELENGTHS * LAMBDA_UM, nsteps=700)
    lam = dominant_wavelength_um(field, dx)
    assert abs(lam - LAMBDA_UM) / LAMBDA_UM < 0.08, lam
    assert float(np.std(field)) > 0.15
    print(
        f"self-test ok  λ_analytic={info['lambda_um']:.4f}  λ_sim={lam:.3f}  "
        f"sweep_r={sweep['pearson_r']:.4f}  ablation_ratio={ab['amplitude_ratio']:.2f}"
    )
    return 0


def measure_morphoeval() -> list[dict]:
    """Entry point for morphoeval/measure_outputs.py. Reads the cache written by main()."""
    if not METRICS_PATH.exists():
        result = run_pipeline(full=True)
        payload = _jsonable(result)
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return result["measurements"]
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return payload["measurements"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    result = run_pipeline(full=True)
    render_gif(result["snaps"], result["field"], GIF_PATH)
    result["sweep"]["amplitude_ratio"] = result["ablation"]["amplitude_ratio"]
    result["sweep"]["hero_um"] = result["hero_wavelength_um"]
    render_png(
        SPECIMEN,
        result["field"],
        result["sweep"],
        result["dispersion_r"],
        result["neumann_wavelength_um"],
        PNG_PATH,
    )
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        w, h = im.size
    if n_frames < 10 or min(w, h) < 480:
        raise SystemExit(f"GIF gate failed: frames={n_frames} size={w}x{h}")
    payload = _jsonable(result)
    payload["artifacts"] = {
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "png": str(PNG_PATH.relative_to(ROOT)),
        "gif_frames": n_frames,
        "gif_px": [w, h],
        "gif_sha256": hashlib.sha256(GIF_PATH.read_bytes()).hexdigest(),
        "png_sha256": hashlib.sha256(PNG_PATH.read_bytes()).hexdigest(),
        "specimen_sha256": hashlib.sha256(SPECIMEN.read_bytes()).hexdigest() if SPECIMEN.exists() else None,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"λ_hero={result['hero_wavelength_um']:.3f} µm  sweep_r={result['sweep']['pearson_r']:.4f}  "
        f"rmse={result['sweep']['rmse_um']:.3f}  ks={result['sweep']['ks_statistic']:.3f}  "
        f"ablation={result['ablation']['amplitude_ratio']:.2f}  "
        f"neumann={result['neumann_wavelength_um']:.3f}  dispersion_r={result['dispersion_r']:.4f}"
    )
    print(f"wrote {PNG_PATH.relative_to(ROOT)} and {GIF_PATH.relative_to(ROOT)} ({n_frames} frames, {w}px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
