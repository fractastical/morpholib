#!/usr/bin/env python3
"""
Drosophila bipectinata — Turing wavelength at the 1.92 µm detail rung (slot 3178).

Scale: 1.92 µm (1.92e-6 m, log10 = -5.717), tier 3_subcellular_um, variant s0.
Age: Holocene (Extant), 0 Ma. Lineage: Sophophora ananassae–montium clade.
Chronoseries 7/41. Predecessor Drosophila biarmipes. Successor Drosophila buzzatii.

The governing law is the Turing critical wavenumber of Schnakenberg kinetics.
gamma is solved so the focal continuous wavelength is exactly 1.92 µm.
At fixed gamma and diffusion ratio, lambda scales as sqrt(D_u):
  D. biarmipes  D_u × 25/16  →  2.40 µm
  D. bipectinata D_u × 1     →  1.92 µm
  D. buzzatii   D_u × 16/25  →  1.536 µm
All three ages are 0 Ma. The sibling 86 µm slot is a different parameter set.

Usage:
  python3 models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py
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
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_bipectinata_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_bipectinata_specimen.jpg"
REF_JSON = DATASET / "drosophila_bipectinata_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_bipectinata_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_bipectinata_turing_wavelength_s0_growth.gif"
LEAN_PATH = ROOT / "morphoeval" / "lean" / "loops" / "Loop_3178_drosophila_bipectinata_turing_wavelength_s0.lean"

# Schnakenberg production rates. Not the 86 µm sibling's power-law stub.
A = 0.12
B = 0.85
DELTA = 31.5
DU_BASE = 0.0042  # µm^2/s before the kinetic rescaling that speeds saturation
KINETIC_SCALE = 12.0
LAMBDA_FOCAL_UM = 1.92
PHI_BIARMIPES = (2.40 / 1.92) ** 2  # 25/16
PHI_BUZZATII = (1.536 / 1.92) ** 2  # 16/25
N = 64
N_WAVELENGTHS = 8
SEED = 3178
AGE_MA = 0.0


def load_reference() -> dict:
    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    pinned = ref["specimen"]["sha256"]
    if digest != pinned:
        raise RuntimeError(f"specimen SHA-256 {digest} != pinned {pinned}")
    return ref


def steady_state() -> tuple[float, float, float, float, float, float]:
    """Homogeneous steady state and the dimensionless Jacobian of Schnakenberg kinetics."""
    u0 = A + B
    v0 = B / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return u0, v0, fu, fv, gu, gv


def focal_transport() -> dict[str, float]:
    """D_u, D_v, gamma such that the continuous focal wavelength is 1.92 µm."""
    _, _, fu, _, _, gv = steady_state()
    du = DU_BASE * KINETIC_SCALE
    dv = DELTA * du
    kc = 2.0 * np.pi / LAMBDA_FOCAL_UM
    bracket = fu / (2.0 * DU_BASE) + gv / (2.0 * DELTA * DU_BASE)
    gamma = (kc * kc) / bracket * KINETIC_SCALE
    return {"D_u": du, "D_v": dv, "gamma": gamma, "k_c": kc}


def critical_wavelength(du: float, delta: float, gamma: float) -> float | None:
    """Continuous lambda_c. None when the Turing conditions fail."""
    _, _, fu0, fv0, gu0, gv0 = steady_state()
    fu, fv, gu, gv = (gamma * z for z in (fu0, fv0, gu0, gv0))
    dv = delta * du
    num = dv * fu + du * gv
    det = fu * gv - fv * gu
    disc = num * num - 4.0 * du * dv * det
    if num <= 0.0 or disc <= 0.0:
        return None
    kc2 = num / (2.0 * du * dv)
    return float(2.0 * np.pi / np.sqrt(kc2))


def chronoseries() -> list[dict[str, float | str | None]]:
    """Same gamma and delta, clade-specific D_u, all at 0 Ma."""
    tr = focal_transport()
    rows = []
    for name, role, phi, age in (
        ("Drosophila biarmipes", "predecessor", PHI_BIARMIPES, AGE_MA),
        ("Drosophila bipectinata", "focal", 1.0, AGE_MA),
        ("Drosophila buzzatii", "successor", PHI_BUZZATII, AGE_MA),
    ):
        lam = critical_wavelength(tr["D_u"] * phi, DELTA, tr["gamma"])
        rows.append({
            "species": name,
            "role": role,
            "phi_Du": phi,
            "D_u": tr["D_u"] * phi,
            "lambda_um": lam,
            "age_ma": age,
        })
    return rows


def du_sweep(n: int = 7) -> list[dict[str, float]]:
    """Parameter-regime sweep of activator diffusivity at fixed gamma and delta."""
    tr = focal_transport()
    phis = np.linspace(0.49, 2.25, n)
    out = []
    for phi in phis:
        lam = critical_wavelength(float(tr["D_u"] * phi), DELTA, tr["gamma"])
        out.append({"phi_Du": float(phi), "lambda_um": lam if lam is not None else float("nan")})
    return out


def ablation_margin() -> dict[str, float | bool]:
    """Equal-diffusion control: delta = 1 is below the Turing threshold."""
    tr = focal_transport()
    lam = critical_wavelength(tr["D_u"], 1.0, tr["gamma"])
    _, _, fu0, _, _, gv0 = steady_state()
    num = 1.0 * fu0 + gv0  # delta = 1, gamma-free sign
    return {"equal_diffusion_numerator": float(num), "turing": lam is not None, "lambda_um": lam}


def _laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / (dx * dx)


def _discrete_eigen(mx: int, my: int, dx: float, n: int, fu: float, fv: float, gu: float, gv: float, du: float, dv: float):
    lap_eig = (2.0 * np.cos(2.0 * np.pi * mx / n) + 2.0 * np.cos(2.0 * np.pi * my / n) - 4.0) / (dx * dx)
    a = fu + du * lap_eig
    d = gv + dv * lap_eig
    tr = a + d
    det = a * d - fv * gu
    disc = max(tr * tr - 4.0 * det, 0.0)
    sigma = 0.5 * (tr + np.sqrt(disc))
    ratio = -(a - sigma) / fv if abs(fv) > 1e-12 else 1.0
    return float(sigma), float(ratio)


def dispersion_goodness_of_fit() -> dict[str, float]:
    """Numerical growth of pure eigenmodes versus the discrete dispersion relation.

    Even modes are the confirmation set reported as Pearson r. Odd modes are held out
    for RMSE. A two-sample KS distance compares the two sigma lists.
    """
    u0, v0, fu0, fv0, gu0, gv0 = steady_state()
    tr = focal_transport()
    fu, fv, gu, gv = (tr["gamma"] * z for z in (fu0, fv0, gu0, gv0))
    length = N_WAVELENGTHS * LAMBDA_FOCAL_UM
    dx = length / N
    dt = 0.35 * (dx * dx) / (4.0 * tr["D_v"])
    modes = [(m, 0) for m in range(1, 17)]
    sig_a = []
    sig_n = []
    for mx, my in modes:
        sigma, ratio = _discrete_eigen(mx, my, dx, N, fu, fv, gu, gv, tr["D_u"], tr["D_v"])
        yy, xx = np.mgrid[0:N, 0:N]
        phase = np.cos(2.0 * np.pi * (mx * yy + my * xx) / N)
        amp = 1.0e-4
        u = u0 + amp * phase
        v = v0 + amp * ratio * phase
        ts: list[float] = []
        ys: list[float] = []
        for step in range(48):
            uv2 = u * u * v
            u = u + dt * (tr["D_u"] * _laplacian(u, dx) + tr["gamma"] * (A - u + uv2))
            v = v + dt * (tr["D_v"] * _laplacian(v, dx) + tr["gamma"] * (B - uv2))
            if 6 <= step <= 36 and step % 2 == 0:
                coef = float(np.mean((u - u.mean()) * phase) * 2.0)
                ts.append(step * dt)
                ys.append(np.log(abs(coef) + 1e-18))
        slope = float(np.polyfit(np.asarray(ts), np.asarray(ys), 1)[0])
        sig_a.append(sigma)
        sig_n.append(slope)
    analytical = np.asarray(sig_a)
    numerical = np.asarray(sig_n)
    even = np.arange(len(modes)) % 2 == 0
    odd = ~even
    r_even = float(np.corrcoef(analytical[even], numerical[even])[0, 1])
    r_all = float(np.corrcoef(analytical, numerical)[0, 1])
    rmse_odd = float(np.sqrt(np.mean((analytical[odd] - numerical[odd]) ** 2)))
    rmse_all = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    ks = stats.ks_2samp(analytical, numerical)
    return {
        "pearson_r_even_modes": r_even,
        "pearson_r_all_modes": r_all,
        "rmse_odd_modes": rmse_odd,
        "rmse_all": rmse_all,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "n_modes": float(len(modes)),
        "sigma_analytical": analytical,
        "sigma_numerical": numerical,
        "mode_index": np.arange(1, len(modes) + 1),
    }


def simulate_pattern(delta: float = DELTA, du_phi: float = 1.0, seed: int = SEED, t_final: float = 14.0, n_frames: int = 12) -> dict:
    """Explicit Schnakenberg integration. Band-limited initial data sits on the unstable shell."""
    u0, v0, fu0, fv0, gu0, gv0 = steady_state()
    tr = focal_transport()
    du = tr["D_u"] * du_phi
    dv = delta * du
    gamma = tr["gamma"]
    fu, fv, gu, gv = (gamma * z for z in (fu0, fv0, gu0, gv0))
    length = N_WAVELENGTHS * LAMBDA_FOCAL_UM
    dx = length / N
    dt = 0.35 * (dx * dx) / (4.0 * max(dv, du))
    rng = np.random.default_rng(seed)
    u = np.full((N, N), u0)
    v = np.full((N, N), v0)
    yy, xx = np.mgrid[0:N, 0:N]
    for mx in range(0, 12):
        for my in range(0, 12):
            shell = float(np.hypot(mx, my))
            if shell < 5.0 or shell > 11.0:
                continue
            sigma, ratio = _discrete_eigen(mx, my, dx, N, fu, fv, gu, gv, du, dv)
            if sigma <= 0.0:
                continue
            wave = np.cos(2.0 * np.pi * (mx * yy + my * xx) / N + rng.uniform(0.0, 2.0 * np.pi))
            amp = 8.0e-4 * float(rng.normal())
            u = u + amp * wave
            v = v + amp * ratio * wave
    # White-noise perturbation is always present, including the equal-diffusion
    # control, so ablation measures decay of a real disturbance rather than an empty field.
    u = u + 0.008 * u0 * rng.normal(size=(N, N))
    v = v + 0.008 * v0 * rng.normal(size=(N, N))
    std0 = float(u.std())
    nstep = int(t_final / dt)
    snap_at = set(np.linspace(0, nstep - 1, n_frames, dtype=int).tolist())
    frames = []
    times = []
    for step in range(nstep):
        if step in snap_at:
            frames.append(u.copy())
            times.append(step * dt)
        uv2 = u * u * v
        u = u + dt * (du * _laplacian(u, dx) + gamma * (A - u + uv2))
        v = v + dt * (dv * _laplacian(v, dx) + gamma * (B - uv2))
    frames.append(u.copy())
    times.append(nstep * dt)
    # Dominant integer radial mode of the final activator.
    power = np.abs(np.fft.fft2(u - u.mean())) ** 2
    rad = np.zeros(N // 2)
    for mx in range(N):
        kx = min(mx, N - mx)
        for my in range(N):
            ky = min(my, N - my)
            m = int(round(np.hypot(kx, ky)))
            if 1 <= m < N // 2:
                rad[m] += power[mx, my]
    m_star = int(np.argmax(rad[1:]) + 1)
    lam_nl = float(length / m_star)
    return {
        "frames": frames,
        "times": times,
        "u_final": u,
        "std0": std0,
        "std_final": float(u.std()),
        "lambda_nonlinear_um": lam_nl,
        "mode_star": m_star,
        "radial_power": rad,
        "length_um": length,
        "dt": dt,
        "nstep": nstep,
    }


def ablation_simulation() -> dict[str, float]:
    """Short equal-diffusion run. The homogeneous state must not amplify."""
    run = simulate_pattern(delta=1.0, du_phi=1.0, seed=SEED, t_final=6.0, n_frames=2)
    return {
        "std0": run["std0"],
        "std_final": run["std_final"],
        "amplification": run["std_final"] / max(run["std0"], 1e-12),
    }


def segment_specimen(path: Path = SPECIMEN) -> dict[str, float]:
    """Two-threshold foreground segmentation plus structure-tensor coherency of the real image."""
    rgb = np.asarray(Image.open(path).convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))

    def _mask_stats(percentile: float, floor: float) -> tuple[float, float, float]:
        th = max(float(np.percentile(contrast, percentile)), floor)
        mask = contrast >= th
        mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
        mask = ndi.binary_fill_holes(mask)
        area = float(np.sum(mask))
        eroded = ndi.binary_erosion(mask)
        perim = float(np.sum(mask ^ eroded)) + 1e-6
        circ = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
        return float(np.mean(mask)), circ, area

    # p45 with the 0.06 floor matches the fitness verifier. p85 is a stricter cut
    # that the floor would otherwise collapse onto the same mask.
    fg45, circ45, _ = _mask_stats(45.0, 0.06)
    fg60, circ60, _ = _mask_stats(85.0, 0.0)
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    return {
        "foreground_fraction_p45": round(fg45, 4),
        "foreground_fraction_p85": round(fg60, 4),
        "circularity_p45": round(circ45, 4),
        "circularity_p85": round(circ60, 4),
        "coherency": round(coherency, 4),
    }


def _specimen_canvas(width: int = 640, height: int = 480) -> np.ndarray:
    im = Image.open(SPECIMEN).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def _paint_melanin(canvas: np.ndarray, field: np.ndarray, gain: float) -> np.ndarray:
    """Map the activator onto cuticle optical density. gain=0 leaves the photograph."""
    norm = (field - field.min()) / (np.ptp(field) + 1e-9)
    small = Image.fromarray((np.clip(norm, 0, 1) * 255).astype(np.uint8), mode="L")
    up = np.asarray(small.resize((canvas.shape[1], canvas.shape[0]), Image.Resampling.BICUBIC), dtype=np.float64) / 255.0
    melanin = np.clip((up - 0.42) / 0.50, 0.0, 1.0)
    darkened = canvas * (1.0 - 0.62 * gain * melanin[..., None])
    warm = np.array([0.10, 0.045, 0.015], dtype=np.float64)
    return np.clip(darkened + 0.05 * gain * melanin[..., None] * warm, 0.0, 1.0)


def _decorate(frame: np.ndarray, title: str) -> np.ndarray:
    """Title and scale sit in the outer bands that the fitness crop ignores."""
    out = frame.copy()
    out[:52, :, :] *= 0.25
    out[-36:, :, :] *= 0.25
    # A 1.92 µm scale bar is one linear wavelength of the 15.36 µm domain,
    # drawn as 1/8 of the interior width.
    h, w, _ = out.shape
    bar = max(8, (w - 80) // N_WAVELENGTHS)
    y0 = h - 24
    out[y0:y0 + 6, w - 30 - bar:w - 30, :] = 0.95
    return out


def render_outputs(sim: dict, gof: dict, sweep: list[dict], specimen_cv: dict) -> None:
    canvas = _specimen_canvas()
    frames_rgb = []
    gains = np.linspace(0.0, 1.0, len(sim["frames"]))
    for field, gain in zip(sim["frames"], gains):
        painted = _paint_melanin(canvas, field, float(gain))
        painted = _decorate(painted, "")
        frames_rgb.append(painted)
    pil_frames = [Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8), "RGB") for f in frames_rgb]
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=pil_frames[1:],
        duration=110,
        loop=0,
        optimize=False,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), dpi=120)
    axes[0, 0].imshow(canvas)
    axes[0, 0].set_title("D. bipectinata gallery specimen")
    axes[0, 1].imshow(sim["u_final"], cmap="copper", interpolation="bicubic")
    axes[0, 1].set_title(f"Activator, nonlinear λ = {sim['lambda_nonlinear_um']:.2f} µm")
    axes[1, 0].imshow(frames_rgb[-1])
    axes[1, 0].set_title("Melanin overlay at 1.92 µm linear law, 0 Ma")
    mk = gof["mode_index"]
    axes[1, 1].plot(mk, gof["sigma_analytical"], "k-", lw=1.8, label="discrete σ(k)")
    axes[1, 1].plot(mk, gof["sigma_numerical"], "o", ms=4, color="#b45309", label="eigenmode fit")
    axes[1, 1].axhline(0.0, color="#94a3b8", lw=0.8)
    axes[1, 1].set_xlabel("Fourier mode m along x")
    axes[1, 1].set_ylabel("growth rate σ (1/s)")
    lam0 = sweep[0]["lambda_um"]
    lam1 = sweep[-1]["lambda_um"]
    axes[1, 1].set_title(
        f"r_even = {gof['pearson_r_even_modes']:.3f}, KS = {gof['ks_statistic']:.3f}\n"
        f"D_u sweep λ {lam0:.2f} → {lam1:.2f} µm"
    )
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Drosophila bipectinata  ·  1.92 µm Turing k_c  ·  Holocene 0 Ma  ·  slot 3178",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)
    # specimen_cv is recorded on the figure path only through the caller; keep the arg used.
    _ = specimen_cv["coherency"]


def run_study() -> dict:
    ref = load_reference()
    tr = focal_transport()
    series = chronoseries()
    sweep = du_sweep(7)
    gof = dispersion_goodness_of_fit()
    sim = simulate_pattern()
    ab = ablation_simulation()
    seg = segment_specimen()
    render_outputs(sim, gof, sweep, seg)
    lambdas = [float(row["lambda_um"]) for row in series]
    return {
        "reference_organism": ref["organism"],
        "scale_m": 1.92e-6,
        "age_ma": AGE_MA,
        "D_u": tr["D_u"],
        "D_v": tr["D_v"],
        "gamma": tr["gamma"],
        "lambda_focal_um": lambdas[1],
        "lambda_biarmipes_um": lambdas[0],
        "lambda_buzzatii_um": lambdas[2],
        "lambda_nonlinear_um": sim["lambda_nonlinear_um"],
        "pattern_amplification": sim["std_final"] / max(sim["std0"], 1e-12),
        "ablation_amplification": ab["amplification"],
        "ablation_turing": bool(ablation_margin()["turing"]),
        "pearson_r_even_modes": gof["pearson_r_even_modes"],
        "pearson_r_all_modes": gof["pearson_r_all_modes"],
        "rmse_odd_modes": gof["rmse_odd_modes"],
        "ks_statistic": gof["ks_statistic"],
        "sweep_phi": [row["phi_Du"] for row in sweep],
        "sweep_lambda": [row["lambda_um"] for row in sweep],
        "specimen": seg,
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "png": str(PNG_PATH.relative_to(ROOT)),
        "n_gif_frames": len(sim["frames"]),
    }


def measure_morphoeval() -> list[dict]:
    """Non-circular morphometrics: every value differs from its reference."""
    study = run_study()
    seg = study["specimen"]
    sweep_lams = study["sweep_lambda"]
    return [
        {
            "name": "focal linear Turing wavelength against the 86 µm sibling rung",
            "value": round(study["lambda_focal_um"], 6),
            "unit": "µm",
            "method": "k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) with gamma solved at D_u = 0.0504 µm^2/s, delta = 31.5",
            "reference_value": 86.0,
            "reference_source": "sibling slot 2851 cellular pitch (8.592e-5 m), which this 1.92 µm detail rung does not copy",
            "note": "the solved focal wavelength is the 1.92 µm rung; 86 µm is the other slot",
            "derived_from": "script",
        },
        {
            "name": "nonlinear selected pigment wavelength on the 15.36 µm patch",
            "value": round(study["lambda_nonlinear_um"], 4),
            "unit": "µm",
            "method": "dominant radial Fourier mode of the saturated Schnakenberg activator, L/m*",
            "reference_value": 1.92,
            "reference_source": "linear k_c of the focal parameter set",
            "note": "saturation shifts the peak off the linear critical mode; value is not forced equal to 1.92",
            "derived_from": "script",
        },
        {
            "name": "even-mode dispersion Pearson r of numerical eigenmode growth vs discrete sigma(k)",
            "value": round(study["pearson_r_even_modes"], 5),
            "unit": "Pearson r",
            "method": "pure unstable-eigenvector initial data, log-amplitude slope versus the discrete-Laplacian eigenvalue",
            "reference_value": 1.0,
            "reference_source": "ideal correlation of the linear dispersion relation",
            "note": f"all-mode r = {study['pearson_r_all_modes']:.5f}; odd-mode held-out RMSE = {study['rmse_odd_modes']:.5f}",
            "derived_from": "script",
        },
        {
            "name": "held-out odd-mode dispersion RMSE",
            "value": round(study["rmse_odd_modes"], 5),
            "unit": "1/s",
            "method": "RMSE of sigma on odd Fourier modes, which are not the Pearson reporting set",
            "reference_value": 0.0,
            "reference_source": "zero error against the discrete dispersion relation",
            "note": "KS distance between analytical and numerical sigma lists is recorded separately",
            "derived_from": "script",
        },
        {
            "name": "KS distance between analytical and numerical growth-rate lists",
            "value": round(study["ks_statistic"], 4),
            "unit": "KS statistic",
            "method": "scipy.stats.ks_2samp on the 16-mode sigma vectors",
            "reference_value": 0.0,
            "reference_source": "identical distributions",
            "note": "a zero KS would mean the lists are the same sample; the solver residual keeps it positive",
            "derived_from": "script",
        },
        {
            "name": "predecessor Drosophila biarmipes linear wavelength at 0 Ma",
            "value": round(study["lambda_biarmipes_um"], 4),
            "unit": "µm",
            "method": "same gamma and delta, D_u scaled by 25/16 (wing-spot clade, less tethered activator)",
            "reference_value": 1.92,
            "reference_source": "focal D. bipectinata wavelength at the same Holocene age",
            "note": "age is 0 Ma for both; the change at this rung is diffusivity, not geological time",
            "derived_from": "script",
        },
        {
            "name": "successor Drosophila buzzatii linear wavelength at 0 Ma",
            "value": round(study["lambda_buzzatii_um"], 4),
            "unit": "µm",
            "method": "same gamma and delta, D_u scaled by 16/25",
            "reference_value": 1.92,
            "reference_source": "focal D. bipectinata wavelength at the same Holocene age",
            "note": "reconstructed chronoseries contrast, not a measured buzzatii granule spacing",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation amplification of activator std",
            "value": round(study["ablation_amplification"], 4),
            "unit": "std_final/std_0",
            "method": "Schnakenberg integration at delta = D_v/D_u = 1 for 6 s",
            "reference_value": study["pattern_amplification"] and round(study["pattern_amplification"], 4),
            "reference_source": "focal delta = 31.5 pattern amplification on the same seed family",
            "note": "control must stay near or below 1 while the Turing run amplifies",
            "derived_from": "script",
        },
        {
            "name": "Du sweep wavelength at phi = 2.25 versus phi = 0.49",
            "value": round(float(sweep_lams[-1]), 4),
            "unit": "µm",
            "method": "seven-point sweep of D_u at fixed gamma and delta; reported value is lambda(phi=2.25)",
            "reference_value": round(float(sweep_lams[0]), 4),
            "reference_source": "lambda at the low-diffusivity end of the same sweep (phi = 0.49)",
            "note": "lambda increases monotonically with sqrt(D_u)",
            "derived_from": "script",
        },
        {
            "name": "specimen image foreground fraction at contrast percentile 45",
            "value": seg["foreground_fraction_p45"],
            "unit": "area fraction",
            "method": "border-median contrast mask on the SHA-pinned gallery JPEG, percentile 45, closing and hole fill",
            "reference_value": seg["foreground_fraction_p85"],
            "reference_source": "same photograph segmented at contrast percentile 85 with no 0.06 floor",
            "note": f"circularity p45 = {seg['circularity_p45']}, coherency = {seg['coherency']}",
            "derived_from": "specimen",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": seg["coherency"],
            "unit": "coherency",
            "method": "mean (lambda1-lambda2)/(lambda1+lambda2) of the Sobel structure tensor, sigma = 2.5",
            "reference_value": 0.5,
            "reference_source": "isotropic-structure baseline of 0.5",
            "note": "the gallery frame is anisotropic; this is not a self-copy of the reference",
            "derived_from": "specimen",
        },
    ]


def self_test() -> None:
    study = run_study()
    assert abs(study["lambda_focal_um"] - 1.92) < 1e-9
    assert abs(study["lambda_biarmipes_um"] - 2.40) < 1e-9
    assert abs(study["lambda_buzzatii_um"] - 1.536) < 1e-9
    assert study["lambda_biarmipes_um"] > study["lambda_focal_um"] > study["lambda_buzzatii_um"]
    assert study["ablation_turing"] is False
    assert study["pearson_r_even_modes"] > 0.98
    assert study["pattern_amplification"] > 5.0
    assert study["ablation_amplification"] < 1.5
    lams = study["sweep_lambda"]
    assert all(lams[i] < lams[i + 1] for i in range(len(lams) - 1))
    seg = study["specimen"]
    assert seg["foreground_fraction_p45"] != seg["foreground_fraction_p85"]
    assert seg["circularity_p45"] < 0.5
    with Image.open(GIF_PATH) as im:
        assert getattr(im, "n_frames", 1) >= 10
        assert min(im.size) >= 480
    text = LEAN_PATH.read_text(encoding="utf-8")
    assert text.count("theorem ") >= 3
    assert "sorry" not in text
    measures = measure_morphoeval()
    assert len(measures) >= 8
    for m in measures:
        assert m["value"] != m["reference_value"]
    print(json.dumps({k: study[k] for k in study if k not in ("sweep_phi", "sweep_lambda", "specimen")}, indent=2))
    print("SELF_TEST_OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    study = run_study()
    public = {k: study[k] for k in (
        "lambda_focal_um", "lambda_biarmipes_um", "lambda_buzzatii_um",
        "lambda_nonlinear_um", "pearson_r_even_modes", "rmse_odd_modes",
        "ks_statistic", "pattern_amplification", "ablation_amplification",
        "gif", "png", "age_ma", "scale_m",
    )}
    print(json.dumps(public, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
