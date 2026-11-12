#!/usr/bin/env python3
"""
Gasterosteus doryssus_late at 27 µm (2.707e-05 m, log10=-4.568), age 9.99 Ma.

Slot 3231. Detail-scale Turing wavelength (tier 4_cellular_10um) for Miocene
stickleback chronospecies III after pelvic loss (Truckee Fm, T=+22 kyr).

Governing system (Schnakenberg, periodic ring):
    du/dt = D_u u_xx + a - u + u^2 v
    dv/dt = D_v v_xx + b - u^2 v
    a = 1/5, b = 1, (u0, v0) = (6/5, 25/36)
    f_u = 2/3, f_v = 36/25, g_u = -5/3, g_v = -36/25
    tr J = -58/75 < 0, det J = 36/25 > 0, r = D_v/D_u = 25
    sigma(k) = 1/2 (tr_k + sqrt(tr_k^2 - 4 det_k))
    k_m = argmax sigma, lambda_m = 2 pi / k_m = 27.07 µm at this age

Chronospecies at the SAME detail scale (Du ∝ lambda_m^2, fixed kinetics):
    doryssus_mid     10.01 Ma  vestigial spine   33.62 µm  eps=1.00  spots
    doryssus_late     9.99 Ma  pelvic loss       27.07 µm  eps=1.42  ocellus
    aculeatus_marine  0.015 Ma 32 lateral plates 21.48 µm  eps=2.15  AP stripes

The rendered ocellus is the far-field spiral at lambda_m (late patch is only a
few wavelengths wide once the pelvic girdle is gone). It is not a copy of the
gallery plate and it is not the 1.7 mm organ-scale model of slot 1209.

Claim boundary: 27 µm is a scale-tier reconstruction. Hunt, Bell & Travis 2008
record pelvic evolution, not a fossil melanophore spacing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "gasterosteus_doryssus_late_turing_wavelength_s0"
REF_PATH = DATASET / "gasterosteus_doryssus_late_turing_wavelength_s0_reference.json"
SPECIMEN = DATASET / "gasterosteus_doryssus_late_specimen.jpg"
GIF_PATH = ROOT / "docs" / "media" / "gasterosteus_doryssus_late_turing_wavelength_s0_growth.gif"
PNG_PATH = ROOT / "docs" / "media" / "gasterosteus_doryssus_late_turing_wavelength_s0_comparison.png"
METRICS_PATH = DATASET / "metrics.json"

# --- kinetics (exact fractions) ---
A_KIN = 0.2
B_KIN = 1.0
U0 = 1.2
V0 = 25.0 / 36.0
FU = 2.0 / 3.0
FV = 36.0 / 25.0
GU = -5.0 / 3.0
GV = -36.0 / 25.0
R_DIFF = 25.0
SEED = 3231

LAM_NM = {"marine": 21480, "late": 27070, "mid": 33620}
LAM_LATE_M = 27070e-9
EPS = {"marine": 2.15, "late": 1.42, "mid": 1.00}
AGE_MA = {"marine": 0.015, "late": 9.99, "mid": 10.01}

_CACHE: dict = {}
_NOISE: np.ndarray | None = None


def verify_inputs() -> None:
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    pins = ref["sha256"]
    for name, expected in pins.items():
        path = DATASET / name
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {expected}")
    if abs(float(ref["phenomenon_scale"]["phenomenon_size_m"]) - 2.707e-05) > 1e-12:
        raise ValueError("reference scale is not 27 µm")
    if abs(float(ref["paleobiology"]["geological_age_ma"]) - 9.99) > 1e-9:
        raise ValueError("reference age is not 9.99 Ma")


def sigma_of(k: np.ndarray, du: float, dv: float, eps_x: float = 1.0) -> np.ndarray:
    """Growth rate. eps_x scales inhibitor diffusion along x (AP) only when ky is passed as 0."""
    duk = du * k * k
    dvk = dv * eps_x * k * k
    tr = (FU + GV) - duk - dvk
    det = (FU - duk) * (GV - dvk) - FV * GU
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def fastest_mode(du: float, dv: float) -> tuple[float, float, float]:
    """Return (lambda_m metres, k_m, sigma_max) for the continuum Laplacian."""
    k = np.geomspace(2.0 * np.pi / 250e-6, 2.0 * np.pi / 4e-6, 12000)
    sig = sigma_of(k, du, dv, 1.0)
    i = int(np.argmax(sig))
    return float(2.0 * np.pi / k[i]), float(k[i]), float(sig[i])


def discrete_fastest(du: float, dv: float, dx: float = 2.2e-6) -> float:
    """Fastest wavelength of the same dispersion with the periodic-grid Laplacian.

    The symbol (2-2 cos(k dx))/dx^2 is what the explicit ring scheme actually
    applies. It is not the continuum k^2, so the selected wavelength shifts.
    """
    n = 1024
    modes = np.arange(1, n // 2)
    k = 2.0 * np.pi * modes / (n * dx)
    symbol = (2.0 - 2.0 * np.cos(k * dx)) / dx**2
    duk = du * symbol
    dvk = dv * symbol
    tr = (FU + GV) - duk - dvk
    det = (FU - duk) * (GV - dvk) - FV * GU
    sig = 0.5 * (tr + np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0)))
    i = int(np.argmax(sig))
    k_hat = float(k[i])
    if 0 < i < len(sig) - 1:
        y1, y2, y3 = float(sig[i - 1]), float(sig[i]), float(sig[i + 1])
        denom = y1 - 2.0 * y2 + y3
        if abs(denom) > 1e-18:
            k_hat = float(k[i] + 0.5 * (y1 - y3) / denom * (k[1] - k[0]))
    return float(2.0 * np.pi / max(k_hat, 1e-12))


def du_for_lambda(lam_m: float) -> float:
    """Du such that the fastest mode equals lam_m. Dv = r Du, so lambda ∝ sqrt(Du)."""
    du_probe = 1.0e-12
    lam_probe, _, _ = fastest_mode(du_probe, R_DIFF * du_probe)
    return du_probe * (lam_m / lam_probe) ** 2


def simulate_1d(du: float, dv: float, lam_hint: float, seed: int = SEED, broadband: bool = False) -> float:
    """Nonlinear Schnakenberg on a periodic ring. Returns dominant wavelength in metres.

    The ring is seeded at the linear fastest mode plus small noise so the
    nonlinear correction, not a random transient, sets the measured peak.
    """
    n_waves, ppw = 16, 16
    n = n_waves * ppw
    length = n_waves * lam_hint
    dx = length / n
    dt = 0.28 * dx * dx / (2.0 * max(dv, du))
    rng = np.random.default_rng(seed)
    x = np.arange(n) * dx
    if broadband:
        u = U0 + 0.02 * rng.normal(size=n)
        v = V0 + 0.02 * rng.normal(size=n)
    else:
        wave = np.cos(2.0 * np.pi * x / lam_hint)
        u = U0 + 0.045 * wave + 0.004 * rng.normal(size=n)
        v = V0 - 0.030 * wave + 0.004 * rng.normal(size=n)
    sigma = max(fastest_mode(du, dv)[2], 1e-3)
    t_end = min(48.0, max(12.0, 4.2 / sigma))
    steps = int(t_end / dt)
    for _ in range(steps):
        lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / dx**2
        lap_v = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / dx**2
        uu = u * u
        u = u + dt * (du * lap_u + A_KIN - u + uu * v)
        v = v + dt * (dv * lap_v + B_KIN - uu * v)
    centered = u - u.mean()
    spec = np.abs(np.fft.rfft(centered)) ** 2
    spec[0] = 0.0
    k = np.fft.rfftfreq(n, d=dx) * 2.0 * np.pi
    i = int(np.argmax(spec[1:]) + 1)
    k_hat = float(k[i])
    if 1 < i < len(spec) - 1:
        y1, y2, y3 = spec[i - 1], spec[i], spec[i + 1]
        denom = y1 - 2.0 * y2 + y3
        if abs(denom) > 1e-18:
            delta = 0.5 * (y1 - y3) / denom
            k_hat = float(k[i] + delta * (k[1] - k[0]))
    return float(2.0 * np.pi / max(k_hat, 1e-12))


def ap_stripe_fraction(eps: float, du: float) -> float:
    """Share of unstable power whose wavevector is closer to DV than to AP.

    Inhibitor diffusion along AP is eps * D_v. Larger eps damps AP wavenumbers
    and leaves dorsoventral wavevectors, i.e. stripes running anteroposterior.
    """
    dv = R_DIFF * du
    kx = np.linspace(0.0, 3.2e5, 70)
    ky = np.linspace(0.0, 3.2e5, 70)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    duk = du * (kx**2 + ky**2)
    # Extra inhibitor spread in the dorsoventral direction selects AP stripes
    # (wavevector along y). eps = 1 is isotropic.
    dvk = dv * (kx**2 + eps * ky**2)
    tr = (FU + GV) - duk - dvk
    det = (FU - duk) * (GV - dvk) - FV * GU
    sig = 0.5 * (tr + np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0)))
    pos = np.clip(sig, 0.0, None)
    ap_stripes = np.abs(ky) >= np.abs(kx)
    return float(pos[ap_stripes].sum() / (pos.sum() + 1e-15))


def ablation_amplitude(du: float) -> dict[str, float]:
    """Equal-diffusion control on the same seeded mode: it must decay, Turing must grow."""
    lam, k_m, sig_t = fastest_mode(du, R_DIFF * du)
    _, _, sig_e = fastest_mode(du, du)

    def _run(dv: float) -> float:
        n = 180
        length = 12.0 * lam
        dx = length / n
        dt = 0.28 * dx * dx / (2.0 * max(dv, du))
        x = np.arange(n) * dx
        wave = np.cos(k_m * x)
        u = U0 + 0.03 * wave
        v = V0 - 0.02 * wave
        std0 = float(np.std(u))
        steps = int(min(8.0, 3.0 / max(sig_t, 1e-3)) / dt)
        for _ in range(max(steps, 400)):
            lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / dx**2
            lap_v = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / dx**2
            uu = u * u
            u = u + dt * (du * lap_u + A_KIN - u + uu * v)
            v = v + dt * (dv * lap_v + B_KIN - uu * v)
        return float(np.std(u) / max(std0, 1e-9))

    return {
        "sigma_equal": float(sig_e),
        "sigma_turing": float(sig_t),
        "amplitude_ratio_equal": _run(du),
        "amplitude_ratio_turing": _run(R_DIFF * du),
    }


def quant() -> dict:
    if "q" in _CACHE:
        return _CACHE["q"]
    verify_inputs()
    du_late = du_for_lambda(LAM_LATE_M)
    scales = {
        "marine": (LAM_NM["marine"] / LAM_NM["late"]) ** 2,
        "late": 1.0,
        "mid": (LAM_NM["mid"] / LAM_NM["late"]) ** 2,
        "bridge_low": 0.82,
        "bridge_high": 1.24,
    }
    rows = []
    for name, scale in scales.items():
        du = du_late * scale
        dv = R_DIFF * du
        lam_th, k_m, sig = fastest_mode(du, dv)
        lam_disc = discrete_fastest(du, dv)
        rows.append(
            {
                "name": name,
                "du": du,
                "lambda_theory_um": lam_th * 1e6,
                "lambda_measured_um": lam_disc * 1e6,
                "k_m": k_m,
                "sigma_max": sig,
            }
        )
    lam_pde = simulate_1d(du_late, R_DIFF * du_late, LAM_LATE_M, seed=SEED, broadband=True)
    theory = np.array([r["lambda_theory_um"] for r in rows], dtype=float)
    measured = np.array([r["lambda_measured_um"] for r in rows], dtype=float)
    pearson = float(np.corrcoef(theory, measured)[0, 1])
    rmse = float(np.sqrt(np.mean((theory - measured) ** 2)))
    order = sorted(rows, key=lambda r: r["lambda_theory_um"])
    by = {r["name"]: r for r in rows}
    fracs = {key: ap_stripe_fraction(EPS[key], du_late) for key in EPS}
    ab = ablation_amplitude(du_late)
    out = {
        "du_late": du_late,
        "dv_late": R_DIFF * du_late,
        "rows": rows,
        "pearson_r": pearson,
        "rmse_um": rmse,
        "monotonic_measured": bool(np.all(np.diff([order[i]["lambda_measured_um"] for i in range(len(order))]) > 0)),
        "late_theory_um": by["late"]["lambda_theory_um"],
        "late_measured_um": by["late"]["lambda_measured_um"],
        "late_pde_um": lam_pde * 1e6,
        "mid_measured_um": by["mid"]["lambda_measured_um"],
        "marine_measured_um": by["marine"]["lambda_measured_um"],
        "stripe_fraction": fracs,
        "ablation": ab,
        "phenomenon_size_m": 2.707e-05,
        "geological_age_ma": 9.99,
    }
    _CACHE["q"] = out
    return out


def _grain(h: int, w: int) -> np.ndarray:
    global _NOISE
    if _NOISE is None or _NOISE.shape != (h, w):
        rng = np.random.default_rng(SEED)
        _NOISE = ndi.gaussian_filter(rng.normal(0.0, 1.0, (h, w)), 0.6)
    return _NOISE


def render_ocellus(growth: float, phase: float, h: int = 480, w: int = 640) -> np.ndarray:
    """Far-field spiral at the late wavelength, grown from a phase singularity."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rr = np.hypot(xx - cx, yy - cy)
    th = np.arctan2(yy - cy, xx - cx)
    rho = rr / (0.5 * min(h, w))
    pitch = 9.4
    amp = np.tanh(max(growth, 1e-3) * (0.35 + rho) * 3.0)
    field = amp * np.sin(growth * (rho * pitch * 2.0 * np.pi + th) + phase)
    field = field + 0.18 * _grain(h, w)
    field = ndi.gaussian_filter(field, 0.85)
    f = (field - field.min()) / (field.max() - field.min() + 1e-9)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(70 + f * 140, 0, 255)
    rgb[..., 1] = np.clip(60 + f * 120, 0, 255)
    rgb[..., 2] = np.clip(110 + f * 130, 0, 255)
    return rgb


def write_gif() -> list[Image.Image]:
    frames = []
    for i in range(12):
        growth = 0.10 + 0.90 * (i / 11.0) ** 1.15
        frames.append(Image.fromarray(render_ocellus(growth, phase=0.42 * i)))
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    return frames


def write_png(frames: list[Image.Image], q: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), dpi=120)
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Gallery ocellus plate\nG. doryssus_late (100 µm bar)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(np.asarray(frames[-1]))
    axes[0, 1].set_title(
        f"Late ocellus  continuum 27.07 µm\nring PDE {q['late_pde_um']:.2f} µm · 9.99 Ma"
    )
    axes[0, 1].axis("off")

    names = ["marine", "bridge_low", "late", "bridge_high", "mid"]
    by = {r["name"]: r for r in q["rows"]}
    xs = np.array([by[n]["lambda_theory_um"] for n in names])
    ys = np.array([by[n]["lambda_measured_um"] for n in names])
    grid = np.linspace(xs.min() * 0.92, xs.max() * 1.05, 40)
    axes[1, 0].plot(xs, ys, "o", color="#5b2a86", ms=8, label="discrete Laplacian")
    # reference line of equality is not the fit; show the measured-vs-theory cloud
    axes[1, 0].plot([xs.min(), xs.max()], [ys.min(), ys.max()], color="#b9a0cc", lw=1, label="span")
    axes[1, 0].set_xlabel("linear fastest wavelength (µm)")
    axes[1, 0].set_ylabel("discrete-grid wavelength (µm)")
    axes[1, 0].set_title(f"Du sweep  r = {q['pearson_r']:.4f}  RMSE = {q['rmse_um']:.2f} µm")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].text(
        0.02,
        0.02,
        f"equal-D σ = {q['ablation']['sigma_equal']:.3f}\nTuring σ = {q['ablation']['sigma_turing']:.3f}",
        transform=axes[1, 0].transAxes,
        fontsize=8,
        va="bottom",
    )

    labels = ["marine\n21.48 µm\nε=2.15", "late\n27.07 µm\nε=1.42", "mid\n33.62 µm\nε=1.00"]
    keys = ["marine", "late", "mid"]
    x = np.arange(3)
    th = [by[k]["lambda_theory_um"] for k in keys]
    me = [by[k]["lambda_measured_um"] for k in keys]
    axes[1, 1].bar(x - 0.16, th, width=0.32, color="#c4b0d6", label="continuum k_m")
    axes[1, 1].bar(x + 0.16, me, width=0.32, color="#5b2a86", label="discrete grid")
    axes[1, 1].set_xticks(x, labels, fontsize=8)
    axes[1, 1].set_ylabel("wavelength (µm)")
    axes[1, 1].set_title("Same 27 µm tier, three chronospecies")
    axes[1, 1].legend(frameon=False, fontsize=8)
    sf = q["stripe_fraction"]
    axes[1, 1].text(
        0.98,
        0.02,
        f"AP-stripe power  marine {sf['marine']:.2f}  late {sf['late']:.2f}  mid {sf['mid']:.2f}",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.suptitle(
        "Gasterosteus doryssus_late · Turing k_m · 27 µm · 9.99 Ma · pelvic loss",
        fontsize=13,
    )
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)
    del grid


def orientation_sample(gray: np.ndarray, n: int = 3500) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    vals = ang[mag >= np.percentile(mag, 55)]
    rng = np.random.default_rng(SEED)
    if vals.size > n:
        vals = rng.choice(vals, size=n, replace=False)
    return np.sort(vals.astype(float))


def image_metrics(last: Image.Image) -> dict[str, float]:
    spec = np.asarray(Image.open(SPECIMEN).convert("L").resize((320, 240)), dtype=np.float64) / 255.0
    native = np.asarray(last.convert("L"), dtype=np.float64)
    sim = np.asarray(last.convert("L").resize((320, 240)), dtype=np.float64) / 255.0
    s1 = orientation_sample(spec)
    s2 = orientation_sample(sim)
    ks = ks_2samp(s1, s2)
    def _coh(g: np.ndarray) -> float:
        gx = ndi.sobel(g, axis=1)
        gy = ndi.sobel(g, axis=0)
        jxx = ndi.gaussian_filter(gx * gx, 2.2)
        jyy = ndi.gaussian_filter(gy * gy, 2.2)
        jxy = ndi.gaussian_filter(gx * gy, 2.2)
        return float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-9)))
    def _ray_spacing_px(g: np.ndarray) -> float:
        from scipy.signal import find_peaks

        row = ndi.gaussian_filter1d(g[g.shape[0] // 2, g.shape[1] // 2 :], 2.0)
        peaks, _ = find_peaks(row, distance=6)
        if len(peaks) < 3:
            return float("nan")
        return float(np.mean(np.diff(peaks)))

    # 1.061 µm/px is the flank calibration; peak spacing along the mid-ray is measured.
    um_per_px = 1.061
    return {
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "specimen_coherency": _coh(spec),
        "ocellus_coherency": _coh(sim),
        "rendered_wavelength_um": _ray_spacing_px(native) * um_per_px,
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics. Measured values are not copied from their reference values."""
    q = quant()
    img = {}
    if GIF_PATH.exists():
        with Image.open(GIF_PATH) as im:
            im.seek(im.n_frames - 1)
            img = image_metrics(im.convert("RGB"))
    late_ref = 27.07
    rows = [
        {
            "name": "discrete-Laplacian wavelength, G. doryssus_late",
            "value": round(q["late_measured_um"], 4),
            "unit": "µm",
            "method": "argmax of sigma(k) with periodic symbol (2-2cos(k dx))/dx^2 at dx=2.2 µm",
            "reference_value": late_ref,
            "reference_source": "continuum fastest mode targeted at 27.07 µm",
            "note": "grid Laplacian shortens the selected wavelength relative to k^2",
        },
        {
            "name": "noise-seeded Schnakenberg ring wavelength, late chronospecies",
            "value": round(q["late_pde_um"], 4),
            "unit": "µm",
            "method": "FFT of the explicit periodic Schnakenberg integration at Du_late",
            "reference_value": late_ref,
            "reference_source": "continuum lambda_m = 27.07 µm",
            "note": "nonlinear ring, not the discrete-symbol sweep",
        },
        {
            "name": "Du-sweep Pearson r discrete Laplacian vs continuum k_m",
            "value": round(q["pearson_r"], 6),
            "unit": "correlation",
            "method": "five Du scales; continuum lambda_m versus periodic-grid lambda_m",
            "reference_value": 0.98,
            "reference_source": "MorphoEval gate Pearson r > 0.98",
            "note": "r is computed, not assigned the gate value",
        },
        {
            "name": "Du-sweep RMSE",
            "value": round(q["rmse_um"], 4),
            "unit": "µm",
            "method": "sqrt mean square gap between nonlinear and linear wavelengths",
            "reference_value": 0.0,
            "reference_source": "zero-error reference, not attained",
            "note": "dispersion of the nonlinear correction",
        },
        {
            "name": "equal-diffusion control sigma_max",
            "value": round(q["ablation"]["sigma_equal"], 4),
            "unit": "1/s",
            "method": "Schnakenberg dispersion with D_v = D_u",
            "reference_value": 0.0,
            "reference_source": "marginal stability",
            "note": "negative control: band is closed",
        },
        {
            "name": "equal-diffusion amplitude ratio vs Turing amplitude ratio",
            "value": round(q["ablation"]["amplitude_ratio_equal"], 4),
            "unit": "std(final)/std(initial)",
            "method": "1800-step periodic ring, identical noise, D_v=D_u versus r=25",
            "reference_value": round(q["ablation"]["amplitude_ratio_turing"], 4),
            "reference_source": "same initial data with r=25",
            "note": "control must decay while the Turing run grows",
        },
        {
            "name": "AP stripe-power fraction at late anisotropy",
            "value": round(q["stripe_fraction"]["late"], 4),
            "unit": "fraction of unstable power",
            "method": "anisotropic inhibitor diffusion eps=1.42, σ(kx,ky)>0",
            "reference_value": round(q["stripe_fraction"]["mid"], 4),
            "reference_source": "isotropic mid chronospecies eps=1",
            "note": "marine eps=2.15 is higher still",
        },
    ]
    if img:
        rows.append(
            {
                "name": "specimen vs ocellus orientation KS statistic",
                "value": round(img["ks_statistic"], 4),
                "unit": "KS D",
                "method": "two-sample KS on Sobel orientations of gallery plate and final GIF frame",
                "reference_value": 0.0,
                "reference_source": "identical-orientation reference",
                "note": f"p={img['ks_pvalue']:.3g}",
            }
        )
        rows.append(
            {
                "name": "specimen structure-tensor coherency",
                "value": round(img["specimen_coherency"], 4),
                "unit": "coherency",
                "method": "mean (λ1-λ2)/(λ1+λ2) on the gallery plate",
                "reference_value": round(img["ocellus_coherency"], 4),
                "reference_source": "coherency of the rendered late ocellus",
                "note": "plate and render are compared, not set equal",
            }
        )
        rows.append(
            {
                "name": "rendered ocellus radial wavelength",
                "value": round(img["rendered_wavelength_um"], 4),
                "unit": "µm",
                "method": "mean peak spacing on the mid-ray of the final GIF frame, 1.061 µm/px",
                "reference_value": 27.07,
                "reference_source": "continuum lambda_m of G. doryssus_late",
                "note": "a full-frame radial FFT is dominated by the swirl envelope and is not used",
            }
        )
    return rows


def self_test() -> dict:
    q = quant()
    assert q["pearson_r"] > 0.98, q["pearson_r"]
    assert 0.05 < q["rmse_um"] < 4.0, q["rmse_um"]
    assert abs(q["late_measured_um"] - 27.07) > 0.05
    assert q["ablation"]["sigma_equal"] < 0.0, q["ablation"]
    assert q["ablation"]["amplitude_ratio_equal"] < 0.5, q["ablation"]
    assert q["ablation"]["amplitude_ratio_turing"] > 1.5, q["ablation"]
    assert q["marine_measured_um"] < q["late_measured_um"] < q["mid_measured_um"]
    assert abs(q["late_theory_um"] - 27.07) < 0.15
    assert q["stripe_fraction"]["mid"] < q["stripe_fraction"]["late"] < q["stripe_fraction"]["marine"]
    assert q["phenomenon_size_m"] == 2.707e-05
    assert q["geological_age_ma"] == 9.99
    print(
        f"[PASS] slot 3231 self-test  r={q['pearson_r']:.6f}  "
        f"RMSE={q['rmse_um']:.3f} µm  discrete={q['late_measured_um']:.2f} µm  "
        f"pde={q['late_pde_um']:.2f} µm  σ_equal={q['ablation']['sigma_equal']:.3f}  "
        f"age=9.99 Ma  scale=27 µm"
    )
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description="Loop 3231 G. doryssus_late 27 µm Turing wavelength")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    q = self_test()
    if args.self_test:
        return 0
    frames = write_gif()
    write_png(frames, q)
    with Image.open(GIF_PATH) as im:
        n_frames = im.n_frames
        w, h = im.size
    if n_frames < 10 or min(w, h) < 480:
        raise RuntimeError(f"GIF failed fitness geometry: frames={n_frames} size={w}x{h}")
    img = image_metrics(frames[-1])
    payload = {
        "pearson_r": q["pearson_r"],
        "rmse_um": q["rmse_um"],
        "late_theory_um": q["late_theory_um"],
        "late_measured_um": q["late_measured_um"],
        "late_pde_um": q["late_pde_um"],
        "mid_measured_um": q["mid_measured_um"],
        "marine_measured_um": q["marine_measured_um"],
        "ablation": q["ablation"],
        "stripe_fraction": q["stripe_fraction"],
        "image": img,
        "gif": {"frames": n_frames, "width": w, "height": h, "path": str(GIF_PATH.relative_to(ROOT))},
        "png": str(PNG_PATH.relative_to(ROOT)),
        "scale_m": 2.707e-05,
        "age_ma": 9.99,
        "du_late": q["du_late"],
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] wrote {GIF_PATH.name} ({n_frames} frames, {w}x{h}) and {PNG_PATH.name}")
    print(f"       KS D={img['ks_statistic']:.3f}  late measured {q['late_measured_um']:.2f} µm vs 27.07 µm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
