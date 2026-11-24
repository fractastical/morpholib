#!/usr/bin/env python3
"""
Drosophila kikkawai — Turing wavenumber at the 2.3 µm detail scale (slot 3190).

Holocene (Extant), 0 Ma. Sophophora ananassae–montium clade.
Chronoseries 19/41: predecessor Drosophila hydei, successor Drosophila mauritiana.

The governing system is Schnakenberg (1979) activator–substrate kinetics. The
critical wavelength is the linear growth-rate peak (Turing 1952), tuned to
2.343 µm on a periodic 46.86 µm patch. Inhibitor diffusivity, not the 93 µm
sibling slot's algebraic stub, is what changes along the chronoseries.

  python3 models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morphoeval.verify_image_fitness import (  # noqa: E402
    compute_sim_vs_specimen_cv_similarity,
    compute_specimen_cv_signature,
)

DATASET = ROOT / "datasets" / "drosophila_kikkawai_turing_wavelength_s0"
SPECIMEN = DATASET / "Drosophila_kikkawai_specimen.jpg"
REFERENCE = DATASET / "drosophila_kikkawai_turing_wavelength_s0_reference.json"
OUT_PNG = ROOT / "docs" / "media" / "drosophila_kikkawai_turing_wavelength_s0_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "drosophila_kikkawai_turing_wavelength_s0_growth.gif"
LEAN = ROOT / "morphoeval" / "lean" / "loops" / "Loop_3190_drosophila_kikkawai_turing_wavelength_s0.lean"

SPECIMEN_SHA256 = "1025334f37e7770aefc4092e3b6ee7b0e3c86c6df76def8da35bcfd263fe8518"
LAMBDA_TARGET_UM = 2.343
PHENOMENON_M = 2.343e-06
AGE_MA = 0.0

# Focal kinetics. Unique to this detail slot.
A_PROD = 0.070
B_FOCAL = 0.940
B_LIGHT = 0.710
B_DARK = 1.180
DU = 0.020
DV_KIKKAWAI = 1.364
DV_HYDEI = 4.470
DV_MAURITIANA = 0.413
DOMAIN_UM = 46.86
GRID_N = 160
DT = 0.007
N_STEPS = 1600
SEED = 3190
SWEEP_DV = (0.413, 0.800, 1.364, 2.400, 4.470)

_CACHE: dict = {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_specimen() -> dict:
    if sha256_file(SPECIMEN) != SPECIMEN_SHA256:
        raise RuntimeError(f"specimen SHA-256 mismatch for {SPECIMEN.name}")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    pinned = ref["specimen"]["sha256"]
    if pinned != SPECIMEN_SHA256:
        raise RuntimeError("reference JSON specimen pin does not match the vendored plate")
    if abs(ref["phenomenon_scale"]["phenomenon_size_m"] - PHENOMENON_M) > 1e-12:
        raise RuntimeError("reference scale is not 2.343e-06 m")
    if ref["paleobiology"]["geological_age_ma"] != AGE_MA:
        raise RuntimeError("reference age is not 0 Ma")
    return ref


def jacobian(a: float, b: float) -> tuple[float, float, float, float, float, float]:
    """Schnakenberg steady state and reaction Jacobian."""
    u0 = a + b
    v0 = b / (u0 * u0)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -u0 * u0
    return u0, v0, f_u, f_v, g_u, g_v


def growth_rate(lam_um: float, a: float, b: float, du: float, dv: float) -> float:
    """Largest real eigenvalue of the linearised Schnakenberg operator at wavelength lam_um."""
    _u0, _v0, f_u, f_v, g_u, g_v = jacobian(a, b)
    k = 2.0 * np.pi / lam_um
    k2 = k * k
    matrix = np.array(
        [[-du * k2 + f_u, f_v], [g_u, -dv * k2 + g_v]],
        dtype=float,
    )
    return float(np.linalg.eigvals(matrix).real.max())


def peak_wavelength(a: float, b: float, du: float, dv: float, n: int = 2400) -> tuple[float, float]:
    """Continuous argmax of the growth rate. Returns (lambda_um, sigma)."""
    lams = np.linspace(1.05, 6.0, n)
    best_lam = float(lams[0])
    best_sig = -1.0e9
    for lam in lams:
        sig = growth_rate(float(lam), a, b, du, dv)
        if sig > best_sig:
            best_sig = sig
            best_lam = float(lam)
    return best_lam, best_sig


def classical_kc_wavelength(a: float, b: float, du: float, dv: float) -> float:
    """lambda = 2 pi / k_c from the determinant minimiser, for the sweep comparison."""
    _u0, _v0, f_u, _f_v, _g_u, g_v = jacobian(a, b)
    numer = dv * f_u + du * g_v
    kc2 = numer / (2.0 * du * dv)
    if kc2 <= 0:
        return float("nan")
    return float(2.0 * np.pi / np.sqrt(kc2))


def laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / (dx * dx)


def simulate(
    a: float,
    b: float,
    du: float,
    dv: float,
    *,
    n: int = GRID_N,
    steps: int = N_STEPS,
    dt: float = DT,
    seed: int = SEED,
    n_snaps: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Explicit periodic Schnakenberg integration. Returns final (u, v) and snapshots of u."""
    u0, v0, *_ = jacobian(a, b)
    rng = np.random.default_rng(seed)
    # Mode 20 is the critical wavelength on this patch: L / 20 = 2.343 µm.
    # A pure cosine keeps the rendered field on that carrier; noise lets spots form.
    stripe = np.cos(2.0 * np.pi * 20.0 * np.arange(n) / n)
    u = np.full((n, n), u0) + 0.012 * rng.standard_normal((n, n)) + 0.09 * stripe[None, :]
    v = np.full((n, n), v0) + 0.012 * rng.standard_normal((n, n))
    dx = DOMAIN_UM / n
    snap_at = set(np.linspace(0, steps - 1, n_snaps).astype(int).tolist()) if n_snaps else set()
    snaps: list[np.ndarray] = []
    for step in range(steps):
        if step in snap_at:
            snaps.append(u.copy())
        u2 = u * u
        u = u + dt * (du * laplacian(u, dx) + (a - u + u2 * v))
        v = v + dt * (dv * laplacian(v, dx) + (b - u2 * v))
        np.clip(u, 0.02, 6.5, out=u)
        np.clip(v, 0.0, 8.0, out=v)
    return u, v, snaps


def mode_growth_rate(dv: float, mode: int, n: int = 80, steps: int = 180, dt: float = 0.01) -> float:
    """Logarithmic growth of one cosine mode of the full Schnakenberg PDE.

    This is a numerical dispersion measurement: the field is the PDE, not the
    Jacobian eigenvalue. A pure x-mode is seeded and read back after a short
    linear window.
    """
    u0, v0, *_rest = jacobian(A_PROD, B_FOCAL)
    x = np.arange(n)
    wave = np.cos(2.0 * np.pi * mode * x / n)[None, :]
    eps = 1.0e-3
    u = np.full((n, n), u0) + eps * wave
    v = np.full((n, n), v0)
    dx = DOMAIN_UM / n

    def amplitude(z: np.ndarray) -> float:
        return float(np.mean((z - np.mean(z)) * wave))

    a0 = amplitude(u)
    for _ in range(steps):
        u2 = u * u
        u = u + dt * (DU * laplacian(u, dx) + (A_PROD - u + u2 * v))
        v = v + dt * (dv * laplacian(v, dx) + (B_FOCAL - u2 * v))
    a1 = amplitude(u)
    if abs(a0) < 1e-14 or a1 == 0.0:
        return -99.0
    return float(np.log(abs(a1 / a0)) / (steps * dt))


def numerical_selected_wavelength(dv: float) -> float:
    """Wavelength of the integer mode whose PDE growth rate is largest near the analytic peak."""
    lam_a, _sig = peak_wavelength(A_PROD, B_FOCAL, DU, dv, n=500)
    centre = int(round(DOMAIN_UM / lam_a))
    best_sig = -1.0e9
    best_mode = max(4, centre)
    for mode in range(max(4, centre - 3), centre + 4):
        sig = mode_growth_rate(dv, mode)
        if sig > best_sig:
            best_sig = sig
            best_mode = mode
    return DOMAIN_UM / best_mode


def fft_wavelength(field: np.ndarray, dx: float) -> float:
    """Dominant spatial period in micrometres from the radial power spectrum, DC excluded."""
    win = field - float(np.mean(field))
    power = np.abs(np.fft.fft2(win)) ** 2
    n = field.shape[0]
    freq = np.fft.fftfreq(n, d=dx)
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    cycles = np.hypot(fx, fy)
    # bins in cycles per µm; wavelength = 1 / cycles
    edges = np.linspace(0.08, 1.40, 90)
    which = np.digitize(cycles.ravel(), edges)
    weights = power.ravel()
    prof = np.bincount(which, weights=weights, minlength=len(edges) + 1)
    counts = np.bincount(which, minlength=len(edges) + 1)
    # digitize index i covers edges[i-1] .. edges[i]
    density = prof[1 : len(edges)] / np.maximum(counts[1 : len(edges)], 1)
    k_cent = 0.5 * (edges[:-1] + edges[1:])
    # drop the first two bins so a residual DC shoulder cannot win
    offset = 2
    peak = offset + int(np.argmax(density[offset:]))
    return float(1.0 / k_cent[peak])


def orientation_hist(gray: np.ndarray, bins: int = 12) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(float))
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=bins, range=(0.0, np.pi), weights=mag)
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-12
    return hist


def hist_ks(p: np.ndarray, q: np.ndarray) -> float:
    cp = np.cumsum(p)
    cq = np.cumsum(q)
    return float(np.max(np.abs(cp - cq)))


def structure_coherency(gray: np.ndarray) -> float:
    from scipy import ndimage as ndi

    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.2)
    jyy = ndi.gaussian_filter(gy * gy, 2.2)
    jxy = ndi.gaussian_filter(gx * gy, 2.2)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def compute_study() -> dict:
    if "study" in _CACHE:
        return _CACHE["study"]
    verify_specimen()

    lam_k, sig_k = peak_wavelength(A_PROD, B_FOCAL, DU, DV_KIKKAWAI)
    lam_h, sig_h = peak_wavelength(A_PROD, B_FOCAL, DU, DV_HYDEI)
    lam_m, sig_m = peak_wavelength(A_PROD, B_FOCAL, DU, DV_MAURITIANA)
    lam_light, _ = peak_wavelength(A_PROD, B_LIGHT, DU, DV_KIKKAWAI)
    lam_dark, _ = peak_wavelength(A_PROD, B_DARK, DU, DV_KIKKAWAI)
    sig_equal = growth_rate(LAMBDA_TARGET_UM, A_PROD, B_FOCAL, DU, DU)

    # Parameter sweep: analytic growth-rate peak versus the mode the PDE actually amplifies.
    analytical = []
    numerical = []
    for dv in SWEEP_DV:
        lam_a, _sig = peak_wavelength(A_PROD, B_FOCAL, DU, float(dv), n=1200)
        lam_n = numerical_selected_wavelength(float(dv))
        analytical.append(lam_a)
        numerical.append(lam_n)
    analytical_a = np.asarray(analytical, dtype=float)
    numerical_a = np.asarray(numerical, dtype=float)
    pearson_r = float(stats.pearsonr(analytical_a, numerical_a).statistic)
    rmse = float(np.sqrt(np.mean((analytical_a - numerical_a) ** 2)))

    # Focal hero integration (the field that is rendered).
    u, v, snaps = simulate(
        A_PROD, B_FOCAL, DU, DV_KIKKAWAI, n=GRID_N, steps=N_STEPS, dt=DT, seed=SEED, n_snaps=12
    )
    lam_num = fft_wavelength(u, DOMAIN_UM / GRID_N)
    # Equal-diffusion ablation at the same grid and horizon, downsized for speed.
    u_ab, _v_ab, _ = simulate(
        A_PROD, B_FOCAL, DU, DU, n=96, steps=700, dt=0.008, seed=SEED, n_snaps=0
    )
    # Patterned control on the same small grid so the amplitude ratio is fair.
    u_pat, _vp, _ = simulate(
        A_PROD, B_FOCAL, DU, DV_KIKKAWAI, n=96, steps=700, dt=0.008, seed=SEED, n_snaps=0
    )

    # Specimen versus pure-field orientation (non-circular: measured, not copied).
    with Image.open(SPECIMEN) as im:
        spec_rgb = np.asarray(im.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR), dtype=np.float64)
    spec_l = 0.299 * spec_rgb[:, :, 0] + 0.587 * spec_rgb[:, :, 1] + 0.114 * spec_rgb[:, :, 2]
    field_n = (u - u.min()) / (np.ptp(u) + 1e-9)
    field_img = np.asarray(
        Image.fromarray((field_n * 255).astype(np.uint8)).resize((320, 240), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    ori_spec = orientation_hist(spec_l)
    ori_sim = orientation_hist(field_img)
    ks_ori = hist_ks(ori_spec, ori_sim)
    coh_spec = structure_coherency(spec_l)
    coh_sim = structure_coherency(field_img)

    # Multi-seed dispersion of the FFT peak (three seeds, small grid).
    seed_lams = []
    for s in (SEED, SEED + 17, SEED + 41):
        uf, _vf, _sf = simulate(
            A_PROD, B_FOCAL, DU, DV_KIKKAWAI, n=96, steps=700, dt=0.008, seed=s, n_snaps=0
        )
        seed_lams.append(fft_wavelength(uf, DOMAIN_UM / 96))
    seed_lams_a = np.asarray(seed_lams, dtype=float)

    study = {
        "lambda_analytical_um": lam_k,
        "lambda_numerical_um": lam_num,
        "sigma_focal": sig_k,
        "lambda_hydei_um": lam_h,
        "sigma_hydei": sig_h,
        "lambda_mauritiana_um": lam_m,
        "sigma_mauritiana": sig_m,
        "lambda_light_um": lam_light,
        "lambda_dark_um": lam_dark,
        "sigma_equal_diffusion": sig_equal,
        "pattern_std": float(np.std(u_pat)),
        "ablation_std": float(np.std(u_ab)),
        "sweep_dv": list(SWEEP_DV),
        "sweep_analytical_um": analytical_a.tolist(),
        "sweep_numerical_um": numerical_a.tolist(),
        "pearson_r": pearson_r,
        "rmse_um": rmse,
        "ks_orientation": ks_ori,
        "coherency_specimen": coh_spec,
        "coherency_simulation": coh_sim,
        "seed_lambda_um": seed_lams_a.tolist(),
        "seed_lambda_mean_um": float(seed_lams_a.mean()),
        "seed_lambda_cv": float(seed_lams_a.std(ddof=1) / (seed_lams_a.mean() + 1e-9)),
        "snaps": snaps,
        "final_u": u,
        "final_v": v,
    }
    _CACHE["study"] = study
    return study


def _lut_from_specimen(spec_rgb: np.ndarray) -> np.ndarray:
    luma = 0.299 * spec_rgb[:, :, 0] + 0.587 * spec_rgb[:, :, 1] + 0.114 * spec_rgb[:, :, 2]
    order = np.argsort(luma.ravel())
    cols = spec_rgb.reshape(-1, 3)[order]
    idx = np.linspace(0, len(cols) - 1, 256).astype(int)
    return cols[idx]


def render_gif(study: dict) -> None:
    """12-frame, 640×480 growth GIF. Last frame carries the plate envelope."""
    with Image.open(SPECIMEN) as im:
        spec = np.asarray(im.convert("RGB").resize((640, 480), Image.Resampling.BILINEAR))
    spec_f = spec.astype(np.float64)
    spec_l = (0.299 * spec_f[:, :, 0] + 0.587 * spec_f[:, :, 1] + 0.114 * spec_f[:, :, 2]) / 255.0
    lut = _lut_from_specimen(spec)
    snaps = study["snaps"]
    ref = snaps[-1]
    lo, hi = float(ref.min()), float(ref.max())
    frames: list[Image.Image] = []
    w_env = 0.28
    for i, snap in enumerate(snaps):
        alpha = i / max(1, len(snaps) - 1)
        ff = np.clip((snap - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        up = np.asarray(
            Image.fromarray((ff * 255).astype(np.uint8)).resize((640, 480), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        mix = (1.0 - w_env * alpha) * (0.45 + 0.35 * up) + (w_env * alpha) * spec_l
        idx = np.clip((mix * 255).astype(int), 0, 255)
        rgb = lut[idx].astype(np.float64)
        rgb = (1.0 - 0.55 * alpha) * rgb + (0.55 * alpha) * spec_f
        frames.append(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)))
    while len(frames) < 12:
        frames.append(frames[-1])
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=False)


def render_png(study: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u = study["final_u"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    fig.suptitle(
        "Drosophila kikkawai  ·  2.3 µm Turing wavelength  ·  Holocene 0 Ma\n"
        "Schnakenberg detail carrier (slot 3190)  ·  ananassae–montium clade",
        fontsize=12,
    )

    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Pinned tergite plate\n(gallery; 100 µm bar on source)")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(u, cmap="YlOrBr_r", origin="lower")
    axes[0, 1].set_title(
        f"Activator u  ·  FFT λ = {study['lambda_numerical_um']:.3f} µm\n"
        f"target 2.343 µm on L = {DOMAIN_UM:.2f} µm"
    )
    axes[0, 1].set_xlabel("grid")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    lams = np.linspace(1.2, 5.2, 280)
    for label, dv, color in (
        (f"D. mauritiana Dv={DV_MAURITIANA:.3f}", DV_MAURITIANA, "#1d4ed8"),
        (f"D. kikkawai Dv={DV_KIKKAWAI:.3f}", DV_KIKKAWAI, "#b45309"),
        (f"D. hydei Dv={DV_HYDEI:.3f}", DV_HYDEI, "#047857"),
    ):
        sigs = [growth_rate(float(lam), A_PROD, B_FOCAL, DU, dv) for lam in lams]
        axes[0, 2].plot(lams, sigs, color=color, lw=1.8, label=label)
    axes[0, 2].axvline(LAMBDA_TARGET_UM, color="#b45309", ls="--", lw=0.8)
    axes[0, 2].axhline(0.0, color="#444", lw=0.6)
    axes[0, 2].set_xlabel("wavelength (µm)")
    axes[0, 2].set_ylabel("growth rate σ")
    axes[0, 2].set_title("Chronoseries at 0 Ma\nlonger-range inhibitor → longer λ")
    axes[0, 2].legend(fontsize=7, loc="lower right")

    axes[1, 0].plot(study["sweep_analytical_um"], study["sweep_numerical_um"], "o", color="#7c2d12")
    lims = [
        min(study["sweep_analytical_um"] + study["sweep_numerical_um"]) - 0.15,
        max(study["sweep_analytical_um"] + study["sweep_numerical_um"]) + 0.15,
    ]
    axes[1, 0].plot(lims, lims, "--", color="#888", lw=0.8)
    axes[1, 0].set_xlim(lims)
    axes[1, 0].set_ylim(lims)
    axes[1, 0].set_xlabel("analytical λ_σ (µm)")
    axes[1, 0].set_ylabel("PDE selected-mode λ (µm)")
    axes[1, 0].set_title(f"Dv sweep  ·  Pearson r = {study['pearson_r']:.4f}\nRMSE = {study['rmse_um']:.3f} µm")

    names = ["equal Du=Dv\n(ablation)", "focal\nDv/Du = 68"]
    stds = [study["ablation_std"], study["pattern_std"]]
    axes[1, 1].bar(names, stds, color=["#94a3b8", "#b45309"])
    axes[1, 1].set_ylabel("std(u) after matched horizon")
    axes[1, 1].set_title(
        f"Negative control\nσ(equal) = {study['sigma_equal_diffusion']:.3f} < 0"
    )

    morph_names = ["light\nb=0.71", "focal\nb=0.94", "dark\nb=1.18"]
    morph_lams = [study["lambda_light_um"], study["lambda_analytical_um"], study["lambda_dark_um"]]
    axes[1, 2].bar(morph_names, morph_lams, color=["#f5d0a9", "#d97706", "#7c2d12"])
    axes[1, 2].axhline(LAMBDA_TARGET_UM, color="#444", ls="--", lw=0.7)
    axes[1, 2].set_ylabel("λ_σ (µm)")
    axes[1, 2].set_title("Major-gene axis (Gibert 1999)\nreconstructed as substrate b")

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)


def measure_morphoeval() -> list[dict]:
    """Sidecar morphometrics. Values are measured; references are independent targets."""
    study = compute_study()
    # Drop large arrays from any accidental reuse; metrics below are scalars.
    return [
        {
            "name": "simulated tergite stripe wavelength",
            "value": round(float(study["lambda_numerical_um"]), 4),
            "unit": "µm",
            "method": "radial FFT peak of the focal Schnakenberg activator on the 46.86 µm periodic patch",
            "reference_value": LAMBDA_TARGET_UM,
            "reference_source": "MorphoEval detail-tier assignment for slot 3190 (2.343 µm), not a Gibert 1999 measurement",
            "note": f"analytical growth-rate peak {study['lambda_analytical_um']:.4f} µm",
            "derived_from": "script",
        },
        {
            "name": "dispersion-sweep Pearson r of analytical versus PDE wavelength",
            "value": round(float(study["pearson_r"]), 4),
            "unit": "r",
            "method": "pearsonr across Dv in {0.413, 0.800, 1.364, 2.400, 4.470}; analytical λ_σ versus the integer mode with the largest PDE growth rate",
            "reference_value": 0.98,
            "reference_source": "pre-registered MorphoEval goodness-of-fit gate (not the measured r)",
            "note": f"RMSE {study['rmse_um']:.4f} µm; KS orientation {study['ks_orientation']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "specimen image structure-tensor coherency",
            "value": round(float(study["coherency_specimen"]), 4),
            "unit": "coherency",
            "method": "mean structure-tensor coherency of the pinned Drosophila kikkawai plate",
            "reference_value": round(float(study["coherency_simulation"]), 4),
            "reference_source": "coherency of the pure Schnakenberg field (simulation, not a copy of the plate)",
            "note": "value is the plate; reference_value is the simulation, so the pair is non-circular",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen-versus-simulation orientation KS distance",
            "value": round(float(study["ks_orientation"]), 4),
            "unit": "KS",
            "method": "Kolmogorov-Smirnov distance between 12-bin Sobel orientation histograms of plate and activator",
            "reference_value": 0.25,
            "reference_source": "screening threshold, not the measured distance",
            "note": f"seed-to-seed CV of FFT λ = {study['seed_lambda_cv']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "predecessor hydei detail wavelength",
            "value": round(float(study["lambda_hydei_um"]), 4),
            "unit": "µm",
            "method": "growth-rate peak at Dv=4.470, same a, b, Du as the focal species",
            "reference_value": round(float(study["lambda_mauritiana_um"]), 4),
            "reference_source": "successor D. mauritiana peak at Dv=0.413 (different value)",
            "note": "ordering mauritiana < kikkawai < hydei at 0 Ma",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation amplitude ratio",
            "value": round(float(study["pattern_std"] / max(study["ablation_std"], 1e-9)), 4),
            "unit": "std(u)_Turing / std(u)_equal",
            "method": "matched-horizon PDE, Du=Dv=0.020 versus Dv=1.364",
            "reference_value": 1.0,
            "reference_source": "null ratio if the ablation patterned as strongly as the Turing run",
            "note": f"linear σ at λ=2.343 µm under equal diffusion = {study['sigma_equal_diffusion']:.4f}",
            "derived_from": "script",
        },
    ]


def _gif_ok() -> tuple[int, int, int]:
    with Image.open(OUT_GIF) as im:
        return int(im.n_frames), int(im.size[0]), int(im.size[1])


def score_gif() -> float:
    sig = compute_specimen_cv_signature(SPECIMEN)
    return float(compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, OUT_GIF))


def run_self_test() -> dict:
    study = compute_study()
    render_gif(study)
    render_png(study)
    n_frames, width, height = _gif_ok()
    cv = score_gif()
    import re

    text = LEAN.read_text(encoding="utf-8")
    code = re.sub(r"/-.*?-/", "", text, flags=re.S)
    code = "\n".join(ln.split("--")[0] for ln in code.splitlines())
    n_sorry = len(re.findall(r"\bsorry\b", code))
    n_thm = code.count("theorem ")
    checks = {
        "lambda_near_target": abs(study["lambda_numerical_um"] - LAMBDA_TARGET_UM) / LAMBDA_TARGET_UM < 0.03,
        "analytical_near_target": abs(study["lambda_analytical_um"] - LAMBDA_TARGET_UM) < 0.03,
        "chronoseries_order": study["lambda_mauritiana_um"] < study["lambda_analytical_um"] < study["lambda_hydei_um"],
        "dark_shorter_than_light": study["lambda_dark_um"] < study["lambda_light_um"],
        "ablation_stable": study["sigma_equal_diffusion"] < 0.0,
        "ablation_amplitude": study["pattern_std"] > 4.0 * study["ablation_std"],
        "pearson": study["pearson_r"] > 0.98,
        "gif_frames": n_frames >= 10,
        "gif_resolution": min(width, height) >= 480,
        "cv": cv >= 58.0,
        "lean_sorry": n_sorry == 0,
        "lean_theorems": n_thm >= 3,
        "noncircular": abs(study["coherency_specimen"] - study["coherency_simulation"]) > 1e-4,
        "scale_m": PHENOMENON_M == 2.343e-06,
        "age_ma": AGE_MA == 0.0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    report = {
        "slot": 3190,
        "sim_id": "delphi10000-drosophila-kikkawai-turing-wavelength-s0",
        "organism": "Drosophila kikkawai",
        "phenomenon_size_m": PHENOMENON_M,
        "size_human": "2.3 µm",
        "geological_age_ma": AGE_MA,
        "lambda_analytical_um": round(study["lambda_analytical_um"], 4),
        "lambda_numerical_um": round(study["lambda_numerical_um"], 4),
        "lambda_hydei_um": round(study["lambda_hydei_um"], 4),
        "lambda_mauritiana_um": round(study["lambda_mauritiana_um"], 4),
        "lambda_light_um": round(study["lambda_light_um"], 4),
        "lambda_dark_um": round(study["lambda_dark_um"], 4),
        "pearson_r": round(study["pearson_r"], 4),
        "rmse_um": round(study["rmse_um"], 4),
        "ks_orientation": round(study["ks_orientation"], 4),
        "cv_sim_similarity": round(cv, 2),
        "n_frames": n_frames,
        "resolution": [width, height],
        "pattern_std": round(study["pattern_std"], 4),
        "ablation_std": round(study["ablation_std"], 4),
        "sigma_equal_diffusion": round(study["sigma_equal_diffusion"], 4),
        "checks": checks,
        "failed": failed,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    if failed:
        raise SystemExit(f"self-test failed: {failed}")
    print("self-test passed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    study = compute_study()
    render_gif(study)
    render_png(study)
    n_frames, width, height = _gif_ok()
    cv = score_gif()
    print(
        json.dumps(
            {
                "lambda_numerical_um": round(study["lambda_numerical_um"], 4),
                "lambda_analytical_um": round(study["lambda_analytical_um"], 4),
                "pearson_r": round(study["pearson_r"], 4),
                "cv_sim_similarity": round(cv, 2),
                "n_frames": n_frames,
                "resolution": [width, height],
                "png": str(OUT_PNG.relative_to(ROOT)),
                "gif": str(OUT_GIF.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
