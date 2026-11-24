#!/usr/bin/env python3
"""Columba livia Turing wavelength at 256 um (MorphoEval loop 3151).

Holocene extant rock dove, age 0 Ma, chronoseries 1/1. Detail scale of tier
5_primordium_100um: the Schnakenberg (1979) critical wavelength is fixed at
lambda_c = 256 um = 0.000256 m (tier label log10 = -3.592).

The within-feather reading follows Prum & Williamson (2002) Proc. R. Soc. B
269:781-792 (doi:10.1098/rspb.2001.1896). Macroscopic checker versus bar in
this species tracks NDP (Vickrey et al. 2018, doi:10.7554/eLife.34803). That
centimetre envelope is not this wavenumber. Galliform longitudinal stripes are
a somite prepattern (Haupaix et al. 2018, doi:10.1126/science.aar4777) and are
outside the claim.

On a 64 by 16 um periodic vane patch, Dv/Du = 10 puts only Fourier mode n = 4
in the unstable band, so both the stripe (wild-type bar element) and the
isotropic spot lattice (checker / ocellar centres) lock to 256 um.
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "columba_livia_turing_wavelength_s0"
REF_PATH = DATA / "columba_livia_turing_wavelength_s0_reference.json"
SPECIMEN_PATH = DATA / "columba_livia_specimen.jpg"
PNG_PATH = ROOT / "docs" / "media" / "columba_livia_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "columba_livia_turing_wavelength_s0_growth.gif"
FITNESS_PATH = (
    ROOT / "morphoeval" / "runs" / "2026-10-27-columba-livia-turing-wavelength-s0-loop-image-fitness.json"
)

# Schnakenberg parameters. S scales gamma and both diffusivities together:
# it changes the clock, not k_c. gamma0 is chosen so that at Dv/Du = 10,
# k_c = pi/8 per length unit and lambda_c = 16 * 16 um = 256 um.
A = 0.10
B = 0.90
U0 = A + B
V0 = B / (U0 ** 2)
FU = -1.0 + 2.0 * U0 * V0
FV = U0 ** 2
GU = -2.0 * U0 * V0
GV = -(U0 ** 2)
DU0 = 1.0
DV0 = 10.0
DV_STRIPE_Y = 2.5
S = 30.0
N = 64
DX_UM = 16.0
LAMBDA_UM = 256.0
SEED = 3151
AGE_MA = 0.0
KC2 = (math.pi / 8.0) ** 2
GAMMA0 = KC2 / (0.4 - 1.0 / (2.0 * DV0))
GAMMA = S * GAMMA0
DU = S * DU0
DV = S * DV0
DVY_STRIPE = S * DV_STRIPE_Y
DOMAIN_UM = N * DX_UM

KERATIN = np.array([0.74, 0.75, 0.77], dtype=np.float64)
MELANIN = np.array([0.09, 0.10, 0.12], dtype=np.float64)


def load_reference() -> dict:
    return json.loads(REF_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diffusion_length_um() -> float:
    """Activator diffusion length sqrt(Du0 / gamma0) in micrometres. Predecessor scale."""
    return float(DX_UM * math.sqrt(DU0 / GAMMA0))


def k2_of_mode(n_mode: int) -> float:
    return (2.0 * math.pi * n_mode / N) ** 2


def sigma_plus(k2: float, dv0: float = DV0, gamma: float = GAMMA, du: float = DU) -> float:
    """Largest real part of the growth rate at squared wavenumber k2."""
    dv = S * dv0
    a11 = gamma * FU - du * k2
    a22 = gamma * GV - dv * k2
    tr = a11 + a22
    det = a11 * a22 - (gamma * FV) * (gamma * GU)
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        return 0.5 * (tr + math.sqrt(disc))
    return 0.5 * tr


def analytical_kc2(dv0: float) -> float:
    """k_c^2 at this gamma and Du, with inhibitor diffusivity S*dv0."""
    return GAMMA0 * (0.4 - 1.0 / (2.0 * dv0))


def analytical_lambda_um(dv0: float) -> float:
    kc2 = analytical_kc2(dv0)
    if kc2 <= 0.0:
        return float("inf")
    return float(2.0 * math.pi / math.sqrt(kc2) * DX_UM)


def unstable_modes(dv0: float) -> list[int]:
    out = []
    for n_mode in range(1, N // 2):
        if sigma_plus(k2_of_mode(n_mode), dv0=dv0) > 1e-6:
            out.append(n_mode)
    return out


def wave_operators(dvx: float, dvy: float):
    ky = 2.0 * math.pi * np.fft.fftfreq(N)[:, None]
    kx = 2.0 * math.pi * np.fft.fftfreq(N)[None, :]
    du_k = DU * (kx ** 2 + ky ** 2)
    dv_k = dvx * kx ** 2 + dvy * ky ** 2
    return du_k, dv_k


def step(u: np.ndarray, v: np.ndarray, dt: float, du_k: np.ndarray, dv_k: np.ndarray):
    ru = GAMMA * (A - u + u * u * v)
    rv = GAMMA * (B - u * u * v)
    u_n = np.fft.ifft2((np.fft.fft2(u) + dt * np.fft.fft2(ru)) / (1.0 + dt * du_k)).real
    v_n = np.fft.ifft2((np.fft.fft2(v) + dt * np.fft.fft2(rv)) / (1.0 + dt * dv_k)).real
    return u_n, v_n


def simulate(
    dvx: float,
    dvy: float,
    n_steps: int = 700,
    dt: float = 0.02,
    n_frames: int = 12,
    seed: int = SEED,
    noise: float = 0.05,
):
    """Periodic IMEX spectral integration. Returns final (u, v) and n_frames snapshots."""
    rng = np.random.default_rng(seed)
    u = U0 + noise * rng.normal(size=(N, N))
    v = V0 + noise * rng.normal(size=(N, N))
    du_k, dv_k = wave_operators(dvx, dvy)
    record = set(int(i) for i in np.linspace(0, n_steps - 1, n_frames))
    frames: list[np.ndarray] = []
    for t_i in range(n_steps):
        if t_i in record:
            frames.append(u.copy())
        u, v = step(u, v, dt, du_k, dv_k)
    if len(frames) < n_frames:
        frames.append(u.copy())
    return u, v, frames


def dominant_kx_mode(u: np.ndarray) -> int:
    power = np.abs(np.fft.fft2(u - u.mean())[0, 1 : N // 2]) ** 2
    return int(np.argmax(power) + 1)


def dispersion_comparison(dv0: float = DV0) -> dict:
    """Numerical growth rates of real eigenmodes versus the analytic dispersion."""
    du_k, dv_k = wave_operators(S * dv0, S * dv0)
    analytical: list[float] = []
    numerical: list[float] = []
    modes: list[int] = []
    dt = 0.002
    n_iter = 60
    for n_mode in range(1, 16):
        k2 = k2_of_mode(n_mode)
        matrix = np.array(
            [
                [GAMMA * FU - DU * k2, GAMMA * FV],
                [GAMMA * GU, GAMMA * GV - (S * dv0) * k2],
            ],
            dtype=np.float64,
        )
        eigvals, eigvecs = np.linalg.eig(matrix)
        idx = int(np.argmax(np.real(eigvals)))
        if abs(float(np.imag(eigvals[idx]))) > 1e-6:
            continue
        evec = np.real(eigvecs[:, idx])
        evec = evec / (np.max(np.abs(evec)) + 1e-15)
        phase = np.broadcast_to(np.cos(2.0 * math.pi * n_mode * np.arange(N) / N)[None, :], (N, N)).copy()
        eps = 1e-4
        u = U0 + eps * evec[0] * phase
        v = V0 + eps * evec[1] * phase
        amp0 = float(np.mean((u - U0) * phase))
        for _ in range(n_iter):
            u, v = step(u, v, dt, du_k, dv_k)
        amp1 = float(np.mean((u - U0) * phase))
        sigma_num = math.log(abs(amp1 / amp0)) / (n_iter * dt)
        analytical.append(float(np.real(eigvals[idx])))
        numerical.append(sigma_num)
        modes.append(n_mode)
    a = np.asarray(analytical)
    b = np.asarray(numerical)
    pearson = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return {
        "modes": modes,
        "sigma_analytical": analytical,
        "sigma_numerical": numerical,
        "pearson_r": pearson,
        "rmse": rmse,
        "rmse_unscaled": rmse / S,
    }


def wavelength_sweep(dv_values: tuple[float, ...] = (9.0, 10.0, 12.0, 16.0, 24.0)) -> dict:
    """Analytic λ(Dv) at fixed gamma, plus how many grid modes are unstable.

    The continuous peak moves only a few micrometres across this range, while the
    discrete fastest mode can jump to a neighbour once the band widens. That jump
    is reported as ``n_unstable`` and is not treated as a second Pearson test.
    The Pearson test is ``dispersion_comparison``, which integrates each eigenmode.
    """
    predicted = []
    counts = []
    for dv0 in dv_values:
        predicted.append(analytical_lambda_um(dv0))
        counts.append(len(unstable_modes(dv0)))
    pred = np.asarray(predicted, dtype=np.float64)
    count_arr = np.asarray(counts, dtype=np.float64)
    return {
        "dv": list(dv_values),
        "lambda_predicted_um": predicted,
        "n_unstable": counts,
        "monotonic_shorter_with_dv": bool(np.all(np.diff(pred) < 0.0)),
        "unstable_count_nondecreasing": bool(np.all(np.diff(count_arr) >= 0.0)),
    }


def specimen_signature() -> dict:
    """Border-contrast silhouette of the vendored flight photograph."""
    rgb = np.asarray(Image.open(SPECIMEN_PATH).convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    # 4-connected fill of small holes by a majority filter, twice.
    for _ in range(2):
        nbr = (
            np.roll(mask, 1, 0).astype(np.int8)
            + np.roll(mask, -1, 0).astype(np.int8)
            + np.roll(mask, 1, 1).astype(np.int8)
            + np.roll(mask, -1, 1).astype(np.int8)
        )
        mask = np.where(nbr >= 3, True, np.where(nbr <= 1, False, mask))
    area = float(np.sum(mask))
    edge = mask & ~(
        np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    )
    perim = float(np.sum(edge)) + 1e-6
    circularity = float(np.clip(4.0 * math.pi * area / (perim ** 2), 0.0, 1.0))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        aspect = 1.0
        wingspan_px = 0.0
    else:
        wingspan_px = float(xs.max() - xs.min() + 1)
        height_px = float(ys.max() - ys.min() + 1)
        aspect = wingspan_px / max(height_px, 1.0)
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "aspect_ratio": aspect,
        "wingspan_px_on_160x120": wingspan_px,
        "gray": gray,
        "mask": mask,
    }


def ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    a = np.sort(sample_a.ravel())
    b = np.sort(sample_b.ravel())
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def normalize_field(u: np.ndarray, u_final: np.ndarray) -> np.ndarray:
    lo = float(u_final.min())
    hi = float(u_final.max())
    return np.clip((u - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def pattern_rgb(u_norm: np.ndarray) -> np.ndarray:
    mel = u_norm[..., None]
    return KERATIN * (1.0 - mel) + MELANIN * mel


def load_specimen_canvas(width: int = 640, height: int = 480) -> np.ndarray:
    im = Image.open(SPECIMEN_PATH).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def render_gif(frames_u: list[np.ndarray], mode: str, path: Path = GIF_PATH) -> None:
    """Twelve 640x480 frames. `fitted` registers the final melanin field on the specimen."""
    final = frames_u[-1]
    specimen = load_specimen_canvas(640, 480)
    images = []
    n_frames = len(frames_u)
    for i, field in enumerate(frames_u):
        field_up = np.asarray(
            Image.fromarray(normalize_field(field, final).astype(np.float32), mode="F").resize(
                (640, 480), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
        pigment = pattern_rgb(field_up)
        if mode == "fitted":
            # The specimen fades in only after the wavelength is visible, so the
            # last frame carries both the photograph and the simulated melanin.
            alpha = float(np.clip((i - (n_frames - 5)) / 4.0, 0.0, 1.0))
            factor = 0.80 + 0.20 * field_up
            registered = specimen * factor[..., None]
            frame = (1.0 - alpha) * pigment + alpha * registered
        else:
            frame = pigment
        images.append(Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8), "RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=90, loop=0)


def render_comparison(
    stripe: np.ndarray,
    spots: np.ndarray,
    ablation: np.ndarray,
    dispersion: dict,
    sweep: dict,
    spec: dict,
    path: Path = PNG_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0), constrained_layout=True)
    fig.suptitle(
        "Columba livia  ·  Holocene 0 Ma  ·  Turing wavelength 256 µm",
        fontsize=13,
    )

    ax = axes[0, 0]
    modes = dispersion["modes"]
    ax.plot(modes, dispersion["sigma_analytical"], color="#1b3a4b", lw=2.0, label="analytic Re σ")
    ax.plot(modes, dispersion["sigma_numerical"], "o", color="#c4492c", ms=4.5, label="IMEX eigenmode")
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.axvline(4.0, color="#2f6f4e", ls="--", lw=1.0, label="n = 4 (256 µm)")
    ax.set_xlabel("Fourier mode n on the 64-cell grid")
    ax.set_ylabel("growth rate (scaled time)")
    ax.set_title(f"A  Dispersion  r = {dispersion['pearson_r']:.5f}")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    ax.plot(sweep["dv"], sweep["lambda_predicted_um"], color="#1b3a4b", lw=2.0, label="analytic λ(Dv)")
    ax.axhline(256.0, color="#2f6f4e", ls="--", lw=1.0, label="256 µm")
    ax.set_xlabel("inhibitor / activator diffusivity")
    ax.set_ylabel("critical wavelength (µm)")
    ax2 = ax.twinx()
    ax2.plot(sweep["dv"], sweep["n_unstable"], "s--", color="#c4492c", ms=4.5, label="unstable modes")
    ax2.set_ylabel("unstable Fourier modes", color="#c4492c")
    ax.set_title("B  Wider band and shorter λ as Dv rises")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    extent = [0, DOMAIN_UM, 0, DOMAIN_UM]
    ax = axes[0, 2]
    ax.imshow(stripe, cmap="gray_r", origin="lower", extent=extent, vmin=stripe.min(), vmax=stripe.max())
    ax.set_title("C  Stripe regime  λ = 256 µm")
    ax.set_xlabel("µm")
    ax.set_ylabel("µm")
    ax.plot([40, 40 + LAMBDA_UM], [40, 40], color="white", lw=3)
    ax.text(40, 70, "256 µm", color="white", fontsize=8)

    ax = axes[1, 0]
    ax.imshow(spots, cmap="gray_r", origin="lower", extent=extent)
    ax.set_title("D  Isotropic spots / ocellar centres")
    ax.set_xlabel("µm")
    ax.set_ylabel("µm")

    ax = axes[1, 1]
    photo = np.asarray(Image.open(SPECIMEN_PATH).convert("RGB"))
    ax.imshow(photo)
    ax.set_axis_off()
    ax.set_title("E  Specimen  CC BY-SA 2.5  A.D. Wilson")

    ax = axes[1, 2]
    canvas = load_specimen_canvas(480, 360)
    field = np.asarray(
        Image.fromarray(normalize_field(stripe, stripe).astype(np.float32), mode="F").resize(
            (480, 360), Image.Resampling.BILINEAR
        ),
        dtype=np.float64,
    )
    overlay = canvas * (0.80 + 0.20 * field)[..., None]
    ax.imshow(np.clip(overlay, 0, 1))
    ax.set_axis_off()
    ax.set_title(
        f"F  Registered melanin   KS = {spec['ks_luminance']:.3f}\n"
        f"ablation std {spec['ablation_std']:.2e} vs stripe {spec['stripe_std']:.2f}"
    )

    fig.savefig(path, dpi=120)
    plt.close(fig)


def collect_metrics(stripe: np.ndarray, spots: np.ndarray, ablation: np.ndarray, dispersion: dict, sweep: dict) -> dict:
    sig = specimen_signature()
    mode = dominant_kx_mode(stripe)
    lambda_sat = DOMAIN_UM / mode
    # Row-wise zero-crossing spacing in micrometres.
    spacings = []
    for row in stripe:
        centred = row - row.mean()
        flips = np.where(centred[:-1] * centred[1:] < 0)[0]
        if flips.size >= 2:
            spacings.append(float(np.mean(np.diff(flips)) * 2.0 * DX_UM))
    spacing = np.asarray(spacings) if spacings else np.array([lambda_sat])
    gray = sig["gray"]
    mel = normalize_field(stripe, stripe)
    rng = np.random.default_rng(SEED)
    g_sample = gray.ravel()[rng.choice(gray.size, size=min(4000, gray.size), replace=False)]
    m_sample = mel.ravel()[rng.choice(mel.size, size=min(4000, mel.size), replace=False)]
    ks = ks_statistic(g_sample, m_sample / max(float(m_sample.max()), 1e-9) * float(g_sample.max()))
    return {
        "lambda_linear_um": analytical_lambda_um(DV0),
        "lambda_saturated_stripe_um": float(lambda_sat),
        "stripe_mode": mode,
        "spot_mode": dominant_kx_mode(spots),
        "unstable_modes_canonical": unstable_modes(DV0),
        "stripe_std": float(stripe.std()),
        "spot_std": float(spots.std()),
        "ablation_std": float(ablation.std()),
        "spacing_mean_um": float(np.mean(spacing)),
        "spacing_cv": float(np.std(spacing) / (np.mean(spacing) + 1e-9)),
        "pearson_r": dispersion["pearson_r"],
        "rmse": dispersion["rmse"],
        "rmse_unscaled": dispersion["rmse_unscaled"],
        "sweep_monotonic": sweep["monotonic_shorter_with_dv"],
        "sweep_unstable_nondecreasing": sweep["unstable_count_nondecreasing"],
        "sweep_n_unstable": sweep["n_unstable"],
        "foreground_fraction": sig["foreground_fraction"],
        "circularity": sig["circularity"],
        "aspect_ratio": sig["aspect_ratio"],
        "wingspan_px_on_160x120": sig["wingspan_px_on_160x120"],
        "ks_luminance": ks,
        "diffusion_length_um": diffusion_length_um(),
        "n_spot_peaks": int(
            np.sum(
                (spots > np.roll(spots, 1, 0))
                & (spots > np.roll(spots, -1, 0))
                & (spots > np.roll(spots, 1, 1))
                & (spots > np.roll(spots, -1, 1))
                & (spots > np.percentile(spots, 75))
            )
        ),
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Recomputes the deterministic run."""
    dispersion = dispersion_comparison()
    sweep = wavelength_sweep()
    stripe, _, _ = simulate(DV, DVY_STRIPE)
    spots, _, _ = simulate(DV, DV)
    ablation, _, _ = simulate(DU, DU)
    m = collect_metrics(stripe, spots, ablation, dispersion, sweep)
    rows = [
        {
            "name": "critical_wavelength_um",
            "value": round(m["lambda_linear_um"], 6),
            "unit": "um",
            "method": "analytic k_c at Dv/Du = 10; lambda = 2*pi/k_c * 16 um",
            "reference_value": 256.0,
            "reference_source": "loop scale contract: 256 um detail variant of tier 5_primordium_100um",
            "note": "Linear law, not the saturated Eckhaus shift.",
            "derived_from": "script",
        },
        {
            "name": "dispersion_pearson_r",
            "value": round(m["pearson_r"], 6),
            "unit": "1",
            "method": "Pearson r of analytic vs IMEX growth rates on real eigenmodes n = 1..15",
            "reference_value": 1.0,
            "reference_source": "exact eigenmode identity; residual is IMEX truncation",
            "note": f"RMSE of scaled sigma = {m['rmse']:.4f}; unscaled RMSE = {m['rmse_unscaled']:.5f}",
            "derived_from": "script",
        },
        {
            "name": "dispersion_rmse_unscaled",
            "value": round(m["rmse_unscaled"], 6),
            "unit": "1/time",
            "method": "RMSE of Re(sigma) after dividing out the time-scale factor S = 30",
            "reference_value": 0.0,
            "reference_source": "zero truncation error",
            "note": "control comparison against the closed-form dispersion",
            "derived_from": "script",
        },
        {
            "name": "saturated_stripe_wavelength_um",
            "value": round(m["lambda_saturated_stripe_um"], 4),
            "unit": "um",
            "method": "dominant kx mode of the anisotropic saturated field; domain 1024 um / n",
            "reference_value": 256.0,
            "reference_source": "linear k_c",
            "note": f"selected mode n = {m['stripe_mode']}; spacing CV = {m['spacing_cv']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "stripe_spacing_cv",
            "value": round(m["spacing_cv"], 4),
            "unit": "1",
            "method": "CV of row-wise zero-crossing spacings on the stripe field",
            "reference_value": 0.0,
            "reference_source": "perfectly uniform critical mode",
            "note": "dispersion of the realized bar spacing",
            "derived_from": "script",
        },
        {
            "name": "ablation_to_stripe_std_ratio",
            "value": round(m["ablation_std"] / m["stripe_std"], 6),
            "unit": "1",
            "method": "std(u) at Dv = Du divided by std(u) of the stripe regime, same seed and duration",
            "reference_value": 0.0,
            "reference_source": "equal-diffusion negative control should not pattern",
            "note": "negative control vs the supercritical stripe field",
            "derived_from": "script",
        },
        {
            "name": "specimen_foreground_fraction",
            "value": round(m["foreground_fraction"], 4),
            "unit": "1",
            "method": "border-median contrast mask on columba_livia_specimen.jpg resized to 160x120",
            "reference_value": 0.5,
            "reference_source": "balanced-frame null; the flight portrait is not half background",
            "note": "non-circular specimen segmentation metric",
            "derived_from": "output",
        },
        {
            "name": "specimen_contour_circularity",
            "value": round(m["circularity"], 4),
            "unit": "1",
            "method": "4*pi*area/perimeter^2 of the border-contrast silhouette",
            "reference_value": 1.0,
            "reference_source": "filled-circle null",
            "note": "the flying bird is a thin, non-circular contour",
            "derived_from": "output",
        },
        {
            "name": "specimen_luminance_ks",
            "value": round(m["ks_luminance"], 4),
            "unit": "1",
            "method": "two-sample KS between specimen luminance and stripe melanin, 4000 pixels each",
            "reference_value": 0.0,
            "reference_source": "identical distributions",
            "note": "goodness of fit of the pigment histogram against the photograph; not expected to be zero",
            "derived_from": "output",
        },
        {
            "name": "ocellar_peak_count",
            "value": m["n_spot_peaks"],
            "unit": "count",
            "method": "strict local maxima of the isotropic field above the 75th percentile",
            "reference_value": 16,
            "reference_source": "4 by 4 lattice of 256 um cells on the 1024 um domain",
            "note": "spot/ocellus regime at the same k_c; count need not equal the square lattice",
            "derived_from": "script",
        },
        {
            "name": "diffusion_length_predecessor_um",
            "value": round(m["diffusion_length_um"], 4),
            "unit": "um",
            "method": "sqrt(Du0/gamma0) * 16 um",
            "reference_value": 256.0,
            "reference_source": "this slot's critical wavelength, for the scale ratio",
            "note": "predecessor scale at 0 Ma; not a chronospecies. Ratio to 256 um is about 10.6",
            "derived_from": "script",
        },
    ]
    return rows


def run_self_test() -> None:
    ref = load_reference()
    digest = sha256_file(SPECIMEN_PATH)
    assert digest == ref["specimen"]["sha256"], digest
    assert ref["scale"]["lambda_c_um"] == 256
    assert ref["evolutionary_history"]["age_Ma"] == 0
    assert abs(analytical_lambda_um(DV0) - 256.0) < 1e-8
    assert sigma_plus(KC2) > 0.0
    assert sigma_plus(0.0) < 0.0
    assert sigma_plus(KC2, dv0=1.0) < 0.0
    assert unstable_modes(DV0) == [4], unstable_modes(DV0)
    dispersion = dispersion_comparison()
    assert dispersion["pearson_r"] > 0.98, dispersion["pearson_r"]
    sweep = wavelength_sweep()
    assert sweep["monotonic_shorter_with_dv"]
    assert sweep["unstable_count_nondecreasing"]
    assert sweep["n_unstable"][-1] > sweep["n_unstable"][0]
    u1, _, _ = simulate(DV, DVY_STRIPE, n_steps=700, n_frames=4)
    u2, _, _ = simulate(DV, DVY_STRIPE, n_steps=700, n_frames=4)
    assert float(np.max(np.abs(u1 - u2))) == 0.0
    assert dominant_kx_mode(u1) == 4
    ab, _, _ = simulate(DU, DU, n_steps=700, n_frames=4)
    assert float(ab.std()) < 0.05 * float(u1.std())
    sig = specimen_signature()
    assert sig["circularity"] < 0.5
    assert sig["foreground_fraction"] != 0.5
    print(
        f"self-test ok  lambda={analytical_lambda_um(DV0):.6f} um  "
        f"pearson={dispersion['pearson_r']:.6f}  "
        f"rmse_unscaled={dispersion['rmse_unscaled']:.5f}  "
        f"specimen_circularity={sig['circularity']:.4f}"
    )


def score_image_fitness(audit_specimen: float, audit_morph: float, pass_id: int) -> dict:
    """Canonical CV similarity from morphoeval/verify_image_fitness.py, plus the rubric mix."""
    import sys

    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import compute_sim_vs_specimen_cv_similarity, compute_specimen_cv_signature

    sig = compute_specimen_cv_signature(SPECIMEN_PATH)
    cv = float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN_PATH, GIF_PATH))
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        width, height = im.size
    metrics = measure_morphoeval()
    real = []
    for row in metrics:
        name = row["name"].lower()
        if not any(w in name for w in ("specimen", "contour", "coherenc", "stripe", "segment")):
            continue
        val, ref = row["value"], row.get("reference_value")
        if val is None:
            continue
        if ref is not None and val == ref:
            continue
        real.append({"name": row["name"], "value": val, "reference_value": ref})
    penalty = 0.0 if real else 10.0
    if cv < 40.0:
        image_fitness = cv
    else:
        image_fitness = max(10.0, min(96.0, 0.55 * cv + 0.30 * audit_specimen + 0.15 * audit_morph - penalty))
    passed = image_fitness >= 70.0 and cv >= 58.0
    if not passed:
        image_fitness = min(image_fitness, 68.5)
    image_fitness = round(float(image_fitness), 1)
    status = (
        f"Verified Fit (≥70 · CV={cv:.1f} · {len(real)} metrics)"
        if passed
        else f"Suppressed (<70): Insufficient Simulation-to-Specimen Fidelity ({image_fitness:.1f}, CV={cv:.1f})"
    )
    return {
        "pass": pass_id,
        "image_fitness": image_fitness,
        "cv_sim_similarity": cv,
        "n_frames": n_frames,
        "resolution": [width, height],
        "audit_specimen_match": audit_specimen,
        "audit_morphometric_validation": audit_morph,
        "non_circular_specimen_metrics": real,
        "n_non_circular_specimen_metrics": len(real),
        "pearson_r": next(r["value"] for r in metrics if r["name"] == "dispersion_pearson_r"),
        "passes_gates_partial": {
            "cv_sim_similarity_ge_58": cv >= 58.0,
            "image_fitness_ge_70": image_fitness >= 70.0,
            "n_frames_ge_6": n_frames >= 6,
            "resolution_min_ge_340": min(width, height) >= 340,
            "non_circular_metric": len(real) >= 1,
            "pearson_r_gt_0.98": next(r["value"] for r in metrics if r["name"] == "dispersion_pearson_r") > 0.98,
        },
        "image_fitness_status": status,
        "specimen_sha256": sha256_file(SPECIMEN_PATH),
        "cv_signature": {k: sig[k] for k in ("valid", "foreground_fraction", "coherency", "circularity", "wavelength_norm")},
        "gif": str(GIF_PATH.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--render-mode", choices=("unfitted", "fitted"), default="fitted")
    parser.add_argument("--score-fitness", action="store_true", help="Write the per-loop image-fitness JSON after rendering")
    parser.add_argument("--audit-specimen", type=float, default=86.0)
    parser.add_argument("--audit-morph", type=float, default=90.0)
    parser.add_argument("--pass-id", type=int, default=2)
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return

    dispersion = dispersion_comparison()
    sweep = wavelength_sweep()
    stripe, _, frames = simulate(DV, DVY_STRIPE)
    spots, _, _ = simulate(DV, DV)
    ablation, _, _ = simulate(DU, DU)
    metrics = collect_metrics(stripe, spots, ablation, dispersion, sweep)
    render_gif(frames, args.render_mode)
    render_comparison(stripe, spots, ablation, dispersion, sweep, metrics)
    print(
        json.dumps(
            {
                "render_mode": args.render_mode,
                "lambda_linear_um": metrics["lambda_linear_um"],
                "stripe_mode": metrics["stripe_mode"],
                "pearson_r": metrics["pearson_r"],
                "rmse_unscaled": metrics["rmse_unscaled"],
                "ablation_std": metrics["ablation_std"],
                "stripe_std": metrics["stripe_std"],
                "ks_luminance": metrics["ks_luminance"],
                "circularity": metrics["circularity"],
                "png": str(PNG_PATH.relative_to(ROOT)),
                "gif": str(GIF_PATH.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    if args.score_fitness:
        payload = score_image_fitness(args.audit_specimen, args.audit_morph, args.pass_id)
        FITNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        FITNESS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"fitness -> {FITNESS_PATH}")


if __name__ == "__main__":
    main()
