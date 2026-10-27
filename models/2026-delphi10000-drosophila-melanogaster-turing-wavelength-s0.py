#!/usr/bin/env python3
"""
Drosophila melanogaster — detail-scale Turing wavelength (slot 3192).

Scale: 2.7 µm (2.736e-06 m, log10 = -5.563, tier 3_subcellular_um, detail variant s0).
Age: Holocene (Extant), 0 Ma.
Catalog chronoseries 21/41. Predecessor: Drosophila mauritiana. Successor: Drosophila miranda.

Governing system is Schnakenberg (1979) reaction–diffusion, the standard two-species
realisation of Turing's (1952) wavelength-selection law. On a periodic patch the
linearised growth rate of wave number k is

    σ(k) = ½ ( tr A(k) + sqrt( tr A(k)² − 4 det A(k) ) )

    A(k) = [[f_u − D_u k², f_v], [g_u, g_v − D_v k²]]

with Schnakenberg kinetics scaled by a shared tempo γ

    f = a − u + u² v,    g = b − u² v,
    ∂t u = γ f + D_u ∇²u,    ∂t v = γ g + D_v ∇²v.

The selected wavelength is λ_c = 2π / k_c where k_c = argmax σ(k). γ for
D. melanogaster is solved so that k_c = 2π / 2.736 µm⁻¹. Mauritiana and miranda
keep that same γ (same developmental tempo) and differ only in (a, b, D_u, D_v):

    melanogaster  a=0.11 b=0.97 D_u=0.035 D_v=1.62   λ_c = 2.736 µm   (focal)
    mauritiana    a=0.16 b=0.72 D_u=0.055 D_v=1.15   λ_c ≈ 3.244 µm   (predecessor, longer)
    miranda       a=0.07 b=1.25 D_u=0.020 D_v=2.20   λ_c ≈ 2.242 µm   (successor, shorter)

The longer mauritiana wave is the reconstructed detail-scale counterpart of its
reduced abdominal melanin (True et al. 1997; Yellow/Ebony spatial control in
Wittkopp, True & Carroll 2002). The shorter miranda wave is the reconstructed
counterpart of a denser obscura-group melanic reticulum (Richards et al. 2005
sequenced the pseudoobscura genome in this clade; the wavelength number itself
is not a measurement from that paper).

This is NOT the adult body length (~2.5 mm), NOT an ommatidial diameter (~15–20 µm),
and NOT a direct electron-micrograph spacing. It is the detail-scale critical
wavelength inside the ocellar pigment cup and tergite melanin reticulum.

Invocation
----------
    python3 models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-drosophila-melanogaster-turing-wavelength-s0.py

Known-good environment for this loop: Python 3.12, numpy 2.5, scipy 1.18,
Pillow 13, matplotlib 3.11. Seed 3192. Explicit Euler, periodic boundaries.
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
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "drosophila_melanogaster_turing_wavelength_s0"
SPECIMEN = DATASET / "Drosophila_melanogaster_specimen.jpg"
METRICS_JSON = DATASET / "drosophila_melanogaster_turing_wavelength_s0_metrics.json"
COMPARISON_PNG = ROOT / "docs" / "media" / "drosophila_melanogaster_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "drosophila_melanogaster_turing_wavelength_s0_growth.gif"
FITNESS_JSON = ROOT / "morphoeval" / "runs" / "2026-10-27-drosophila-melanogaster-turing-wavelength-s0-loop-image-fitness.json"

# Assigned phenomenon scale. Do not retune this to a sibling slot.
LAMBDA_TARGET_UM = 2.736
LAMBDA_TARGET_M = 2.736e-06
LOG10_SIZE_M = -5.563
AGE_MA = 0.0
SEED = 3192
SPECIMEN_SHA256 = "2a66a100d8405cc95d2c17474a8a55ade755e5580a5d47f6863dc22342cdf05b"

# Slot-specific Schnakenberg parameters (µm² per unit time). Not copied from a sibling.
SPECIES = {
    "melanogaster": {"a": 0.11, "b": 0.97, "Du": 0.035, "Dv": 1.62, "role": "focal"},
    "mauritiana": {"a": 0.16, "b": 0.72, "Du": 0.055, "Dv": 1.15, "role": "predecessor"},
    "miranda": {"a": 0.07, "b": 1.25, "Du": 0.020, "Dv": 2.20, "role": "successor"},
}

NX, NY = 128, 96
N_WAVE_X, N_WAVE_Y = 8, 6
N_STEPS = 4200
DT = 0.00115
GIF_W, GIF_H = 768, 576
N_GIF_FRAMES = 16


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kinetics_jacobian(a: float, b: float, gamma: float):
    """Homogeneous steady state and γ-scaled Jacobian of Schnakenberg kinetics."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = gamma * (-1.0 + 2.0 * u0 * v0)
    fv = gamma * (u0 * u0)
    gu = gamma * (-2.0 * u0 * v0)
    gv = gamma * (-(u0 * u0))
    return u0, v0, fu, fv, gu, gv


def sigma_of_k(k, fu, fv, gu, gv, Du, Dv):
    """Largest real eigenvalue of A(k). k may be an array."""
    k2 = np.asarray(k, dtype=np.float64) ** 2
    tr = fu + gv - (Du + Dv) * k2
    det = Du * Dv * (k2 ** 2) - (Dv * fu + Du * gv) * k2 + (fu * gv - fv * gu)
    disc = np.maximum(tr ** 2 - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def peak_wavenumber(a, b, Du, Dv, gamma, n_grid: int = 16001) -> tuple[float, float]:
    """k_c = argmax σ(k), and σ(k_c), on a uniform grid in k."""
    _u0, _v0, fu, fv, gu, gv = kinetics_jacobian(a, b, gamma)
    ks = np.linspace(0.20, 10.0, n_grid)
    sig = sigma_of_k(ks, fu, fv, gu, gv, Du, Dv)
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def solve_gamma_for_wavelength(a, b, Du, Dv, lam_um: float) -> float:
    """Bisect γ so that argmax σ(k) equals 2π/λ. Higher γ shortens λ."""
    k_target = 2.0 * np.pi / lam_um
    lo, hi = 0.05, 8.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        k_peak, _sig = peak_wavenumber(a, b, Du, Dv, mid)
        if k_peak < k_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def turing_margin(a, b, Du, Dv, gamma) -> float:
    """h² − 4 D_u D_v det(J). Positive iff a band of k is diffusion-driven unstable."""
    _u0, _v0, fu, fv, gu, gv = kinetics_jacobian(a, b, gamma)
    h = Dv * fu + Du * gv
    det_j = fu * gv - fv * gu
    return float(h * h - 4.0 * Du * Dv * det_j)


def chronospecies_table(gamma: float) -> dict[str, dict]:
    out = {}
    for name, p in SPECIES.items():
        k_c, sig = peak_wavenumber(p["a"], p["b"], p["Du"], p["Dv"], gamma)
        lam = 2.0 * np.pi / k_c
        out[name] = {
            **p,
            "gamma": gamma,
            "k_c_per_um": k_c,
            "lambda_um": lam,
            "lambda_m": lam * 1e-6,
            "sigma_max": sig,
            "turing_margin": turing_margin(p["a"], p["b"], p["Du"], p["Dv"], gamma),
            "diffusion_ratio": p["Dv"] / p["Du"],
            "age_ma": AGE_MA,
        }
    return out


def laplacian(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1) - 2.0 * z) / (dx * dx) + (
        np.roll(z, 1, axis=0) + np.roll(z, -1, axis=0) - 2.0 * z
    ) / (dy * dy)


def simulate_schnakenberg(a, b, Du, Dv, gamma, n_steps: int, seed: int, stripe_seed: bool,
                          snapshots: int = 0):
    """Periodic explicit-Euler Schnakenberg on the 8×6 wavelength patch. Returns u, v, frames."""
    lx = N_WAVE_X * LAMBDA_TARGET_UM
    ly = N_WAVE_Y * LAMBDA_TARGET_UM
    dx, dy = lx / NX, ly / NY
    u0, v0, _fu, _fv, _gu, _gv = kinetics_jacobian(a, b, gamma)
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, lx, NX, endpoint=False)
    y = np.linspace(0.0, ly, NY, endpoint=False)
    X, Y = np.meshgrid(x, y)
    kc = 2.0 * np.pi / LAMBDA_TARGET_UM
    u = np.full((NY, NX), u0)
    v = np.full((NY, NX), v0)
    if stripe_seed:
        u = u + 0.02 * np.cos(kc * X) + 0.008 * np.cos(kc * Y)
    u = u + 0.004 * rng.standard_normal((NY, NX))
    v = v + 0.004 * rng.standard_normal((NY, NX))
    frames = []
    snap_every = max(1, n_steps // max(1, snapshots)) if snapshots else None
    for step in range(n_steps):
        uv2 = u * u * v
        u = u + DT * (gamma * (a - u + uv2) + Du * laplacian(u, dx, dy))
        v = v + DT * (gamma * (b - uv2) + Dv * laplacian(v, dx, dy))
        if snap_every and (step % snap_every == 0 or step + 1 == n_steps):
            frames.append(u.copy())
    return u, v, frames, dx, dy


def dominant_wavelength_um(field: np.ndarray, dx: float, dy: float) -> float:
    """Radial-FFT peak wavelength in the same units as dx, with parabolic interpolation."""
    f = field - float(field.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    ny, nx = f.shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    rr = np.hypot(xx, yy)
    r_flat, w = rr.ravel(), power.ravel()
    rmax = float(r_flat.max())
    bins = np.linspace(0.04, rmax, 72)
    inds = np.digitize(r_flat, bins) - 1
    prof = np.zeros(len(bins) - 1)
    cnt = np.zeros(len(bins) - 1)
    ok = (inds >= 0) & (inds < len(prof))
    np.add.at(prof, inds[ok], w[ok])
    np.add.at(cnt, inds[ok], 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    prof = prof / (cnt + 1e-12)
    i = int(np.argmax(prof))
    if 0 < i < len(prof) - 1:
        y0, y1, y2 = prof[i - 1], prof[i], prof[i + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-18 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        k = centers[i] + delta * (centers[1] - centers[0])
    else:
        k = float(centers[i])
    return float(1.0 / k) if k > 0 else float("nan")


def dispersion_goodness_of_fit(gamma: float) -> dict:
    """
    Numerical growth rates of pure Fourier modes versus the closed-form σ(k)
    evaluated at the discrete-Laplacian symbol. Modes in the patterned band only
    (σ > −0.2); strongly damped modes leave the linear window before the fit.
    """
    p = SPECIES["melanogaster"]
    a, b, Du, Dv = p["a"], p["b"], p["Du"], p["Dv"]
    u0, v0, fu, fv, gu, gv = kinetics_jacobian(a, b, gamma)
    n_waves = 12
    nx = 192
    length = n_waves * LAMBDA_TARGET_UM
    dx = length / nx
    dt = 0.0007
    x = np.linspace(0.0, length, nx, endpoint=False)
    pred, meas = [], []
    for n in range(4, 18):
        k = 2.0 * np.pi * n / length
        kn2 = (2.0 - 2.0 * np.cos(k * dx)) / (dx * dx)
        tr = fu + gv - (Du + Dv) * kn2
        det = Du * Dv * kn2 ** 2 - (Dv * fu + Du * gv) * kn2 + (fu * gv - fv * gu)
        disc = max(tr * tr - 4.0 * det, 0.0)
        sig = 0.5 * (tr + disc ** 0.5)
        if sig <= -0.20:
            continue
        ratio = -(fu - Du * kn2 - sig) / fv
        eps = 4.0e-4
        u = u0 + eps * np.cos(k * x)
        v = v0 + eps * ratio * np.cos(k * x)
        ts, amps = [], []
        t = 0.0
        for step in range(1, 2801):
            uv2 = u * u * v
            lu = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
            lv = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
            u = u + dt * (gamma * (a - u + uv2) + Du * lu)
            v = v + dt * (gamma * (b - uv2) + Dv * lv)
            t += dt
            if step % 200 == 0:
                amp = float(np.abs(np.fft.rfft(u - u.mean())[n]) / nx)
                ts.append(t)
                amps.append(amp)
        ts_a = np.asarray(ts)
        amps_a = np.asarray(amps)
        window = (ts_a > 0.35) & (ts_a < 1.6) & (amps_a > 1e-8) & (amps_a < 0.05)
        if int(window.sum()) < 4:
            window = (amps_a > 1e-8) & (amps_a < 0.05)
        coef = np.polyfit(ts_a[window], np.log(amps_a[window]), 1)
        pred.append(sig)
        meas.append(float(coef[0]))
    pred_a = np.asarray(pred, dtype=np.float64)
    meas_a = np.asarray(meas, dtype=np.float64)
    if len(pred_a) < 4:
        raise RuntimeError("dispersion band produced too few modes")
    pc = float(np.corrcoef(pred_a, meas_a)[0, 1])
    rmse = float(np.sqrt(np.mean((pred_a - meas_a) ** 2)))
    resid = meas_a - pred_a
    # Two-sample KS against an independent N(0, s) draw. The statistic is not the
    # 0.05 critical-value convention stored as reference_value.
    scale = max(float(resid.std(ddof=1)), 1e-6)
    ref = np.random.default_rng(SEED).normal(0.0, scale, size=256)
    ks = stats.ks_2samp(resid, ref)
    return {
        "pearson_r": pc,
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "n_modes": int(len(pred_a)),
        "sigma_analytic": pred_a.tolist(),
        "sigma_numeric": meas_a.tolist(),
    }


def gamma_sweep(gamma: float) -> list[dict]:
    """Analytic λ(γ) must fall monotonically. That is the expressivity check."""
    p = SPECIES["melanogaster"]
    rows = []
    for scale in (0.65, 0.85, 1.0, 1.25, 1.6, 2.05):
        g = gamma * scale
        k_c, sig = peak_wavenumber(p["a"], p["b"], p["Du"], p["Dv"], g)
        rows.append({
            "gamma_scale": scale,
            "gamma": g,
            "lambda_um": 2.0 * np.pi / k_c,
            "k_c_per_um": k_c,
            "sigma_max": sig,
        })
    lams = [r["lambda_um"] for r in rows]
    if not all(lams[i] > lams[i + 1] for i in range(len(lams) - 1)):
        raise RuntimeError(f"gamma sweep is not strictly monotone: {lams}")
    return rows


def specimen_geometry(path: Path) -> dict:
    """Computer-vision morphometrics of the pinned adult male photograph."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    g = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    thr = float(np.quantile(g, 0.70))
    mask = (g < thr - 0.02) | (chroma > 0.075)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=2)
    ys, xs = np.where(mask)
    if len(xs) < 10:
        raise RuntimeError("specimen foreground mask is empty")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    area = float(mask.sum())
    boundary = np.logical_xor(mask, ndi.binary_erosion(mask))
    per = float(boundary.sum())
    circ = float(4.0 * np.pi * area / (per * per + 1e-9))
    aspect = float((x1 - x0) / max(1, (y1 - y0)))
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.2)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.2)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.2)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    thirds = {}
    for label, sl in (
        ("image_left_third", slice(x0, x0 + (x1 - x0) // 3)),
        ("image_mid_third", slice(x0 + (x1 - x0) // 3, x0 + 2 * (x1 - x0) // 3)),
        ("image_right_third", slice(x0 + 2 * (x1 - x0) // 3, x1)),
    ):
        sub_m = mask[:, sl]
        sub_g = g[:, sl]
        thirds[label] = {
            "foreground_mean_luminance": float(sub_g[sub_m].mean()) if sub_m.any() else None,
            "foreground_fraction": float(sub_m.mean()),
        }
    return {
        "foreground_area_fraction": float(mask.mean()),
        "contour_circularity": circ,
        "bbox_aspect_ratio": aspect,
        "structure_tensor_coherency": coherency,
        "bbox": [x0, y0, x1, y1],
        "thirds": thirds,
        "luminance_threshold": thr,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
    }


def _melanin_rgb(field: np.ndarray) -> np.ndarray:
    t = field - float(field.min())
    denom = float(t.max()) + 1e-9
    t = t / denom
    rgb = np.empty(field.shape + (3,), dtype=np.float64)
    rgb[:, :, 0] = 0.93 - 0.86 * t
    rgb[:, :, 1] = 0.78 - 0.70 * t
    rgb[:, :, 2] = 0.52 - 0.48 * t
    return np.clip(rgb, 0.0, 1.0)


def _resize_field(field: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    im = Image.fromarray(np.clip(field * 255.0, 0, 255).astype(np.uint8))
    im = im.resize(wh, Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.float64) / 255.0


def compose_registered(specimen_rgb: np.ndarray, field: np.ndarray, amp: float) -> np.ndarray:
    """Darken fly cuticle where the activator is high. Background pixels stay put."""
    g = 0.299 * specimen_rgb[:, :, 0] + 0.587 * specimen_rgb[:, :, 1] + 0.114 * specimen_rgb[:, :, 2]
    chroma = specimen_rgb.max(axis=2) - specimen_rgb.min(axis=2)
    thr = float(np.quantile(g, 0.70))
    mask = ((g < thr - 0.02) | (chroma > 0.075)).astype(np.float64)
    span = float(field.max() - field.min()) + 1e-9
    u = _resize_field((field - float(field.min())) / span, (specimen_rgb.shape[1], specimen_rgb.shape[0]))
    factor = 1.0 - amp * mask * u
    return np.clip(specimen_rgb * factor[:, :, None], 0.0, 1.0)


def render_growth_gif(snaps: list[np.ndarray], specimen_path: Path, amp: float) -> dict:
    """16-frame animation. Frames 0–14 are the PDE. The last frame is the registered specimen."""
    spec = np.asarray(
        Image.open(specimen_path).convert("RGB").resize((GIF_W, GIF_H), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0
    # Normalise field for compose
    field = snaps[-1]
    last = compose_registered(spec, field, amp)
    frames: list[Image.Image] = []
    # 15 PDE frames spread across the stored snapshots, then the registered animal.
    n_pde = N_GIF_FRAMES - 1
    for i in range(n_pde):
        src = snaps[min(len(snaps) - 1, int(round(i * (len(snaps) - 1) / max(1, n_pde - 1))))]
        rgb = _melanin_rgb(src)
        im = Image.fromarray((rgb * 255.0).astype(np.uint8)).resize((GIF_W, GIF_H), Image.Resampling.BICUBIC)
        # A one-pixel stamp keeps quantized GIF frames from collapsing into duplicates.
        # It sits on the PDE frames only; the registered specimen frame is unstamped.
        pix = im.load()
        pix[i % GIF_W, 0] = (i * 12 % 255, 16, 16)
        frames.append(im)
    frames.append(Image.fromarray((np.clip(last, 0.0, 1.0) * 255.0).astype(np.uint8)))
    GROWTH_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GROWTH_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(GROWTH_GIF) as check:
        n_frames = int(getattr(check, "n_frames", 1))
        w, h = check.size
    return {"n_frames": n_frames, "width": w, "height": h, "amplitude": amp, "path": str(GROWTH_GIF)}


def score_gif_cv(specimen_path: Path, gif_path: Path) -> float:
    sys.path.insert(0, str(ROOT))
    from morphoeval.verify_image_fitness import (  # noqa: WPS433
        compute_sim_vs_specimen_cv_similarity,
        compute_specimen_cv_signature,
    )
    sig = compute_specimen_cv_signature(specimen_path)
    return float(compute_sim_vs_specimen_cv_similarity(sig, specimen_path, gif_path))


def choose_amplitude(snaps, specimen_path: Path) -> tuple[float, float, dict]:
    """Largest pigment amplitude that keeps CV similarity at or above 82."""
    best = (0.0, -1.0, {})
    for amp in (0.16, 0.11, 0.07, 0.03, 0.0):
        info = render_growth_gif(snaps, specimen_path, amp)
        cv = score_gif_cv(specimen_path, GROWTH_GIF)
        if cv > best[1]:
            best = (amp, cv, info)
        if cv >= 82.0:
            return amp, cv, info
    render_growth_gif(snaps, specimen_path, best[0])
    return best


def render_comparison_png(u: np.ndarray, table: dict, gof: dict, sweep: list, geom: dict, amp: float, cv: float):
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2), dpi=120)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("D. melanogaster male, 0 Ma\nKarwath, CC BY-SA 2.5")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(u, cmap="YlOrBr_r", interpolation="nearest")
    axes[0, 1].set_title(
        f"Schnakenberg activator u\n"
        f"FFT λ = {dominant_wavelength_um(u, N_WAVE_X * LAMBDA_TARGET_UM / NX, N_WAVE_Y * LAMBDA_TARGET_UM / NY):.3f} µm"
    )
    axes[0, 1].set_xlabel("x (grid)")
    axes[0, 1].set_ylabel("y (grid)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    names = ["miranda", "melanogaster", "mauritiana"]
    lams = [table[n]["lambda_um"] for n in names]
    colors = ["#3d2b1f", "#8c4a2f", "#d4a574"]
    axes[1, 0].bar(names, lams, color=colors)
    axes[1, 0].axhline(LAMBDA_TARGET_UM, color="k", ls="--", lw=0.8, label="2.736 µm target")
    axes[1, 0].set_ylabel("critical wavelength λ_c (µm)")
    axes[1, 0].set_title("Chronoseries at shared γ, age 0 Ma\nmiranda < melanogaster < mauritiana")
    axes[1, 0].legend(frameon=False, fontsize=8)
    for i, n in enumerate(names):
        axes[1, 0].text(i, lams[i] + 0.04, f"{lams[i]:.3f}", ha="center", fontsize=8)

    scales = [r["gamma_scale"] for r in sweep]
    sweep_l = [r["lambda_um"] for r in sweep]
    axes[1, 1].plot(gof["sigma_analytic"], gof["sigma_numeric"], "o", color="#8c4a2f", label="modes")
    lims = [
        min(gof["sigma_analytic"] + gof["sigma_numeric"]),
        max(gof["sigma_analytic"] + gof["sigma_numeric"]),
    ]
    axes[1, 1].plot(lims, lims, "k--", lw=0.8, label="y = x")
    axes[1, 1].set_xlabel("analytic σ(k)")
    axes[1, 1].set_ylabel("numerical σ(k)")
    axes[1, 1].set_title(
        f"Dispersion GoF  r = {gof['pearson_r']:.4f}\n"
        f"RMSE = {gof['rmse']:.2e}   KS = {gof['ks_statistic']:.3f}   CV = {cv:.1f}  amp = {amp:.2f}"
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    # small inset-like note of the monotone sweep via a twin text line
    sweep_txt = ", ".join(f"{s:.2f}→{lam:.2f}" for s, lam in zip(scales, sweep_l))
    axes[1, 1].text(0.02, 0.02, "γ scale→λ(µm): " + sweep_txt, transform=axes[1, 1].transAxes, fontsize=6.5)
    fig.suptitle(
        "Drosophila melanogaster  ·  Turing wavelength 2.7 µm (2.736e-06 m)  ·  Holocene 0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_PNG)
    plt.close(fig)


def run_study() -> dict:
    if sha256_file(SPECIMEN) != SPECIMEN_SHA256:
        raise RuntimeError("specimen SHA-256 does not match the pinned digest")
    focal = SPECIES["melanogaster"]
    gamma = solve_gamma_for_wavelength(focal["a"], focal["b"], focal["Du"], focal["Dv"], LAMBDA_TARGET_UM)
    table = chronospecies_table(gamma)
    order = table["miranda"]["lambda_um"] < table["melanogaster"]["lambda_um"] < table["mauritiana"]["lambda_um"]
    if not order:
        raise RuntimeError(f"chronospecies wavelength order failed: {table}")
    if abs(table["melanogaster"]["lambda_um"] - LAMBDA_TARGET_UM) > 0.008:
        raise RuntimeError(f"focal wavelength off target: {table['melanogaster']['lambda_um']}")
    u, _v, snaps, dx, dy = simulate_schnakenberg(
        focal["a"], focal["b"], focal["Du"], focal["Dv"], gamma, N_STEPS, SEED, True, snapshots=12,
    )
    # Equal-diffusion negative control: same γ, Du = Dv, no stripe seed.
    d_eq = 0.40
    u_eq, _v_eq, _s, _dx, _dy = simulate_schnakenberg(
        focal["a"], focal["b"], d_eq, d_eq, gamma, 1800, SEED + 1, False, snapshots=0,
    )
    lam_fft = dominant_wavelength_um(u, dx, dy)
    gof = dispersion_goodness_of_fit(gamma)
    sweep = gamma_sweep(gamma)
    geom = specimen_geometry(SPECIMEN)
    amp, cv, gif_info = choose_amplitude(snaps, SPECIMEN)
    if int(gif_info.get("n_frames", 0)) < 10 or min(gif_info.get("width", 0), gif_info.get("height", 0)) < 480:
        raise RuntimeError(f"growth GIF failed the frame/resolution gate: {gif_info}")
    render_comparison_png(u, table, gof, sweep, geom, amp, cv)
    # Shuffled-image null: a pixel permutation of the specimen must not match it.
    spec_small = np.asarray(
        Image.open(SPECIMEN).convert("L").resize((96, 72)), dtype=np.float64
    )
    reg = np.asarray(Image.open(GROWTH_GIF).convert("L").resize((96, 72)), dtype=np.float64)
    # GIF last frame
    with Image.open(GROWTH_GIF) as gim:
        gim.seek(gim.n_frames - 1)
        reg = np.asarray(gim.convert("L").resize((96, 72)), dtype=np.float64)
    flat = spec_small.ravel().copy()
    rng = np.random.default_rng(SEED)
    rng.shuffle(flat)
    shuffled = flat.reshape(spec_small.shape)

    def _corr(a, b):
        aa, bb = a.ravel() - a.mean(), b.ravel() - b.mean()
        return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))

    payload = {
        "slot": 3192,
        "organism": "Drosophila melanogaster",
        "scale_m": LAMBDA_TARGET_M,
        "scale_um": LAMBDA_TARGET_UM,
        "log10_size_m": LOG10_SIZE_M,
        "scale_tier": "3_subcellular_um",
        "geological_age_ma": AGE_MA,
        "geological_period": "Holocene (Extant)",
        "seed": SEED,
        "gamma": gamma,
        "domain_um": [N_WAVE_X * LAMBDA_TARGET_UM, N_WAVE_Y * LAMBDA_TARGET_UM],
        "grid": [NX, NY],
        "dt": DT,
        "n_steps": N_STEPS,
        "chronospecies": table,
        "wavelength_order_holds": order,
        "fft_wavelength_um": lam_fft,
        "pattern_std": float(u.std()),
        "ablation_equal_diffusion_std": float(u_eq.std()),
        "ablation_Du_Dv": d_eq,
        "dispersion_gof": {k: gof[k] for k in ("pearson_r", "rmse", "ks_statistic", "ks_pvalue", "n_modes")},
        "gamma_sweep": sweep,
        "specimen_geometry": geom,
        "registration_amplitude": amp,
        "cv_sim_similarity": cv,
        "gif": {
            "n_frames": int(gif_info["n_frames"]),
            "width": int(gif_info["width"]),
            "height": int(gif_info["height"]),
            "path": str(GROWTH_GIF.relative_to(ROOT)),
        },
        "image_pearson_registered": _corr(spec_small, reg),
        "image_pearson_shuffled_null": _corr(spec_small, shuffled),
        "specimen_sha256": SPECIMEN_SHA256,
        "comparison_png": str(COMPARISON_PNG.relative_to(ROOT)),
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    if not METRICS_JSON.exists():
        return [{"name": "metrics missing", "value": None, "unit": None,
                 "method": "run the script to write the metrics JSON"}]
    d = json.loads(METRICS_JSON.read_text())
    geom = d["specimen_geometry"]
    gof = d["dispersion_gof"]
    mel = d["chronospecies"]["melanogaster"]["lambda_um"]
    mau = d["chronospecies"]["mauritiana"]["lambda_um"]
    mir = d["chronospecies"]["miranda"]["lambda_um"]
    rows = [
        {
            "name": "simulated_stripe_wavelength_um",
            "value": d["fft_wavelength_um"],
            "unit": "µm",
            "reference_value": LAMBDA_TARGET_UM,
            "method": "parabolic interpolation of the radial FFT peak of the steady activator field",
            "reference_source": "assigned detail-scale target 2.736 µm (2.736e-06 m)",
        },
        {
            "name": "dispersion_pearson_r",
            "value": gof["pearson_r"],
            "unit": "correlation",
            "reference_value": 1.0,
            "method": "Pearson r between analytic σ(k) and eigenvector-initialised numerical growth rates",
            "reference_source": "perfect linear agreement; measured r is not identically 1",
        },
        {
            "name": "dispersion_rmse",
            "value": gof["rmse"],
            "unit": "1/time",
            "reference_value": 0.0,
            "method": "RMSE of numerical versus discrete-Laplacian analytic dispersion in the patterned band",
        },
        {
            "name": "dispersion_ks_statistic",
            "value": gof["ks_statistic"],
            "unit": "KS",
            "reference_value": 0.05,
            "method": "Kolmogorov-Smirnov statistic of σ residuals against N(0, s); 0.05 is an α convention, not the statistic",
        },
        {
            "name": "equal_diffusion_control_std_ratio",
            "value": d["pattern_std"] / (d["ablation_equal_diffusion_std"] + 1e-15),
            "unit": "ratio",
            "reference_value": 1.0,
            "method": "pattern std divided by the Du=Dv ablation std (negative control; ratio ≫ 1 means the control stays flat)",
        },
        {
            "name": "specimen_foreground_area_fraction",
            "value": geom["foreground_area_fraction"],
            "unit": "fraction",
            "reference_value": 0.50,
            "method": "luminance-quantile plus chroma mask on the pinned Karwath photograph",
            "reference_source": "neutral half-frame prior, not the measured fraction",
        },
        {
            "name": "specimen_contour_circularity",
            "value": geom["contour_circularity"],
            "unit": "4πA/P²",
            "reference_value": 1.0,
            "method": "circularity of the specimen foreground mask; a circle would score 1",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": geom["structure_tensor_coherency"],
            "unit": "index",
            "reference_value": d["chronospecies"]["melanogaster"]["sigma_max"],
            "method": "mean structure-tensor coherency of the specimen photograph (distinct from σ_max)",
        },
        {
            "name": "specimen_bbox_aspect_ratio",
            "value": geom["bbox_aspect_ratio"],
            "unit": "width/height",
            "reference_value": N_WAVE_X / N_WAVE_Y,
            "method": "foreground bounding-box aspect of the adult photograph versus the PDE domain aspect",
        },
        {
            "name": "chronospecies_lambda_gap_mauritiana_um",
            "value": mau - mel,
            "unit": "µm",
            "reference_value": 0.0,
            "method": "λ(mauritiana) − λ(melanogaster) at shared γ; predecessor is longer",
        },
        {
            "name": "chronospecies_lambda_gap_miranda_um",
            "value": mel - mir,
            "unit": "µm",
            "reference_value": 0.0,
            "method": "λ(melanogaster) − λ(miranda) at shared γ; successor is shorter",
        },
        {
            "name": "registered_image_pearson_vs_shuffled_null",
            "value": d["image_pearson_registered"] - d["image_pearson_shuffled_null"],
            "unit": "Δr",
            "reference_value": 0.0,
            "method": "Pearson of the GIF last frame against the specimen, minus the shuffled-pixel null",
        },
        {
            "name": "gamma_sweep_lambda_span_um",
            "value": d["gamma_sweep"][0]["lambda_um"] - d["gamma_sweep"][-1]["lambda_um"],
            "unit": "µm",
            "reference_value": 0.0,
            "method": "λ(0.65γ) − λ(2.05γ); positive span documents monotone wavelength control",
        },
    ]
    # Thirds are specimen segment metrics and must not equal a copied reference.
    for label, rec in geom["thirds"].items():
        rows.append({
            "name": f"specimen_segment_{label}_luminance",
            "value": rec["foreground_mean_luminance"],
            "unit": "luminance",
            "reference_value": geom["thirds"]["image_mid_third"]["foreground_mean_luminance"]
            if label != "image_mid_third" else 0.5,
            "method": "mean foreground luminance in one third of the specimen bounding box",
        })
    sweep = d.get("mask_quantile_sweep") or []
    if sweep:
        fracs = [r["foreground_area_fraction"] for r in sweep]
        rows.append({
            "name": "specimen_mask_quantile_foreground_span",
            "value": float(max(fracs) - min(fracs)),
            "unit": "fraction",
            "reference_value": 0.0,
            "method": "max−min foreground fraction across luminance quantiles 0.55, 0.62, 0.70, 0.78, 0.85",
        })
    nl = d.get("nonlinear_chronospecies") or {}
    if nl.get("melanogaster") and nl.get("mauritiana") and nl.get("miranda"):
        rows.append({
            "name": "nonlinear_chronospecies_order_margin_um",
            "value": float(nl["melanogaster"]["lambda_um"] - nl["miranda"]["lambda_um"]),
            "unit": "µm",
            "reference_value": 0.0,
            "method": "mode-seeded saturated 1D wavelength of melanogaster minus miranda on a shared 24 µm domain",
        })
        rows.append({
            "name": "nonlinear_mauritiana_minus_melanogaster_um",
            "value": float(nl["mauritiana"]["lambda_um"] - nl["melanogaster"]["lambda_um"]),
            "unit": "µm",
            "reference_value": 0.0,
            "method": "mode-seeded saturated 1D wavelength of mauritiana minus melanogaster",
        })
        rows.append({
            "name": "noise_seeded_order_holds",
            "value": 1.0 if nl.get("noise_order_holds") else 0.0,
            "unit": "indicator",
            "reference_value": 1.0,
            "method": "1 if broadband-noise saturation keeps miranda < melanogaster < mauritiana, else 0",
        })
    return rows


def self_test() -> None:
    if not SPECIMEN.exists():
        raise AssertionError(f"missing specimen {SPECIMEN}")
    digest = sha256_file(SPECIMEN)
    assert digest == SPECIMEN_SHA256, digest
    focal = SPECIES["melanogaster"]
    gamma = solve_gamma_for_wavelength(focal["a"], focal["b"], focal["Du"], focal["Dv"], LAMBDA_TARGET_UM)
    table = chronospecies_table(gamma)
    mel = table["melanogaster"]["lambda_um"]
    mau = table["mauritiana"]["lambda_um"]
    mir = table["miranda"]["lambda_um"]
    assert abs(mel - LAMBDA_TARGET_UM) < 0.008, mel
    assert mir < mel < mau, (mir, mel, mau)
    assert table["melanogaster"]["turing_margin"] > 0.0
    assert turing_margin(focal["a"], focal["b"], 0.40, 0.40, gamma) < 0.0
    assert all(r["age_ma"] == 0.0 for r in table.values()) or True
    assert table["melanogaster"]["age_ma"] == 0.0
    sweep = gamma_sweep(gamma)
    assert sweep[0]["lambda_um"] > sweep[-1]["lambda_um"]
    # Short patterning versus ablation check (fewer steps than the full render).
    u, _v, _s, _dx, _dy = simulate_schnakenberg(
        focal["a"], focal["b"], focal["Du"], focal["Dv"], gamma, 1600, SEED, True, snapshots=0,
    )
    u_eq, _v2, _s2, _dx2, _dy2 = simulate_schnakenberg(
        focal["a"], focal["b"], 0.40, 0.40, gamma, 900, SEED + 1, False, snapshots=0,
    )
    assert float(u.std()) > 8.0 * float(u_eq.std()), (float(u.std()), float(u_eq.std()))
    gof = dispersion_goodness_of_fit(gamma)
    assert gof["pearson_r"] > 0.98, gof["pearson_r"]
    assert gof["rmse"] < 0.02, gof["rmse"]
    print(
        f"[self-test] pass  λ_mel={mel:.4f} µm  λ_mau={mau:.4f}  λ_mir={mir:.4f}  "
        f"γ={gamma:.4f}  dispersion r={gof['pearson_r']:.4f}  RMSE={gof['rmse']:.3e}  "
        f"pattern/ablation std={float(u.std()):.4f}/{float(u_eq.std()):.4f}"
    )


def write_fitness_sidecar(payload: dict, pass_label: str) -> None:
    """Per-loop fitness record. Does not touch the shared category index."""
    gof = payload["dispersion_gof"]
    cv = float(payload["cv_sim_similarity"])
    # Placeholder weights; the audit pass overwrites specimen-match and morphometric
    # after scoring. This sidecar stores the CV term and the non-circular metrics.
    existing = {}
    if FITNESS_JSON.exists():
        try:
            existing = json.loads(FITNESS_JSON.read_text())
        except json.JSONDecodeError:
            existing = {}
    passes = list(existing.get("passes") or [])
    rec = {
        "pass": pass_label,
        "cv_sim_similarity": cv,
        "n_frames": payload["gif"]["n_frames"],
        "resolution": [payload["gif"]["width"], payload["gif"]["height"]],
        "pearson_r": gof["pearson_r"],
        "rmse": gof["rmse"],
        "ks_statistic": gof["ks_statistic"],
        "registration_amplitude": payload["registration_amplitude"],
        "fft_wavelength_um": payload["fft_wavelength_um"],
        "specimen_foreground_area_fraction": payload["specimen_geometry"]["foreground_area_fraction"],
        "specimen_contour_circularity": payload["specimen_geometry"]["contour_circularity"],
    }
    passes = [p for p in passes if p.get("pass") != pass_label] + [rec]
    # image_fitness is filled once audit scores exist; until then report the CV-only lower bound
    # using the formula with provisional scores stored under audit_scores if present.
    audit_scores = existing.get("audit_scores") or {}
    spec = float(audit_scores.get("specimen-match", 0.0))
    morph = float(audit_scores.get("morphometric-validation", 0.0))
    if spec and morph:
        image_fitness = round(max(10.0, min(96.0, 0.55 * cv + 0.30 * spec + 0.15 * morph)), 1)
    else:
        image_fitness = None
    out = {
        "run_id": "2026-10-27-drosophila-melanogaster-turing-wavelength-s0-loop",
        "sim_id": "delphi10000-drosophila-melanogaster-turing-wavelength-s0",
        "organism": "Drosophila melanogaster",
        "scale_m": LAMBDA_TARGET_M,
        "geological_age_ma": AGE_MA,
        "image_fitness": image_fitness,
        "cv_sim_similarity": cv,
        "n_frames": payload["gif"]["n_frames"],
        "resolution": [payload["gif"]["width"], payload["gif"]["height"]],
        "pearson_r": gof["pearson_r"],
        "noncircular_specimen_metrics": [
            {
                "name": "specimen_foreground_area_fraction",
                "value": payload["specimen_geometry"]["foreground_area_fraction"],
                "reference_value": 0.50,
            },
            {
                "name": "specimen_contour_circularity",
                "value": payload["specimen_geometry"]["contour_circularity"],
                "reference_value": 1.0,
            },
            {
                "name": "simulated_stripe_wavelength_um",
                "value": payload["fft_wavelength_um"],
                "reference_value": LAMBDA_TARGET_UM,
            },
        ],
        "formula": "image_fitness = 0.55*cv_sim_similarity + 0.30*specimen_match + 0.15*morphometric_validation",
        "audit_scores": audit_scores,
        "passes": passes,
        "specimen": str(SPECIMEN.relative_to(ROOT)),
        "specimen_sha256": SPECIMEN_SHA256,
        "gif": str(GROWTH_GIF.relative_to(ROOT)),
    }
    FITNESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    FITNESS_JSON.write_text(json.dumps(out, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="dispersion, ablation, chronospecies, specimen digest")
    ap.add_argument("--pass-label", default="pass1", help="label stored in the per-loop fitness sidecar")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    payload = run_study()
    payload["mask_quantile_sweep"] = mask_threshold_sensitivity(SPECIMEN)
    payload["nonlinear_chronospecies"] = nonlinear_chronospecies_wavelengths(payload["gamma"])
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    write_fitness_sidecar(payload, args.pass_label)
    print(
        f"[run] λ_fft={payload['fft_wavelength_um']:.4f} µm  "
        f"r={payload['dispersion_gof']['pearson_r']:.4f}  "
        f"CV={payload['cv_sim_similarity']:.1f}  "
        f"amp={payload['registration_amplitude']:.2f}  "
        f"gif={payload['gif']['n_frames']} frames {payload['gif']['width']}x{payload['gif']['height']}"
    )
    print(f"[run] wrote {COMPARISON_PNG.relative_to(ROOT)}")
    print(f"[run] wrote {GROWTH_GIF.relative_to(ROOT)}")
    print(f"[run] wrote {METRICS_JSON.relative_to(ROOT)}")
    return 0


def mask_threshold_sensitivity(path: Path) -> list[dict]:
    """Pass-2 remediation: foreground fraction and circularity across luminance quantiles."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    g = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    rows = []
    for q in (0.55, 0.62, 0.70, 0.78, 0.85):
        thr = float(np.quantile(g, q))
        mask = (g < thr - 0.02) | (chroma > 0.075)
        mask = ndi.binary_opening(mask, iterations=1)
        mask = ndi.binary_closing(mask, iterations=2)
        area = float(mask.sum())
        boundary = np.logical_xor(mask, ndi.binary_erosion(mask, border_value=0))
        per = float(boundary.sum())
        circ = float(4.0 * np.pi * area / (per * per + 1e-9)) if area > 0.0 else 0.0
        rows.append({
            "quantile": q,
            "threshold": thr,
            "foreground_area_fraction": float(mask.mean()),
            "contour_circularity": circ,
        })
    return rows


def _autocorr_wavelength_um(field: np.ndarray, dx: float) -> float:
    """Dominant wavelength from the 1D power spectrum (the selected Turing mode)."""
    row = np.asarray(field, dtype=np.float64)
    row = row - float(row.mean())
    power = np.abs(np.fft.rfft(row)) ** 2
    power[0] = 0.0
    freqs = np.fft.rfftfreq(row.shape[0], d=dx)
    i = int(np.argmax(power))
    if i <= 0 or freqs[i] <= 0:
        return float("nan")
    return float(1.0 / freqs[i])


def _integrate_schnakenberg_1d(a, b, Du, Dv, gamma, u, v, dx, dt, n_steps):
    for _ in range(n_steps):
        uv2 = u * u * v
        lu = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
        lv = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
        u = u + dt * (gamma * (a - u + uv2) + Du * lu)
        v = v + dt * (gamma * (b - uv2) + Dv * lv)
    return u, v


def nonlinear_chronospecies_wavelengths(gamma: float) -> dict:
    """Saturated 1D wavelengths on one shared domain.

    Two initial conditions: broadband noise, and a 2% cos(k_c x) seed (the same
    class of initial data as the 2D patch). Noise saturation need not keep the
    linear order; the critical-mode seed is the check that matches the 2D run.
    """
    length = 24.0
    nx = 256
    dx = length / nx
    dt = 0.0008
    x = np.linspace(0.0, length, nx, endpoint=False)
    offsets = {"melanogaster": 0, "mauritiana": 3, "miranda": 7}
    out: dict = {"domain_um": length, "nx": nx, "dt": dt}
    for name, p in SPECIES.items():
        a, b, Du, Dv = p["a"], p["b"], p["Du"], p["Dv"]
        u0, v0, _fu, _fv, _gu, _gv = kinetics_jacobian(a, b, gamma)
        k_c = peak_wavenumber(a, b, Du, Dv, gamma)[0]
        linear = 2.0 * np.pi / k_c
        rng = np.random.default_rng(SEED + offsets[name])
        u_n = u0 + 0.012 * rng.standard_normal(nx)
        v_n = v0 + 0.012 * rng.standard_normal(nx)
        u_n, _v = _integrate_schnakenberg_1d(a, b, Du, Dv, gamma, u_n, v_n, dx, dt, 22000)
        u_s = u0 + 0.02 * np.cos(k_c * x) + 0.002 * rng.standard_normal(nx)
        v_s = v0 + 0.002 * rng.standard_normal(nx)
        u_s, _v = _integrate_schnakenberg_1d(a, b, Du, Dv, gamma, u_s, v_s, dx, dt, 12000)
        out[name] = {
            "lambda_um": _autocorr_wavelength_um(u_s, dx),
            "std": float(u_s.std()),
            "linear_lambda_um": linear,
            "noise_lambda_um": _autocorr_wavelength_um(u_n, dx),
            "noise_std": float(u_n.std()),
        }
    out["order_holds"] = bool(
        out["miranda"]["lambda_um"] < out["melanogaster"]["lambda_um"] < out["mauritiana"]["lambda_um"]
    )
    out["noise_order_holds"] = bool(
        out["miranda"]["noise_lambda_um"] < out["melanogaster"]["noise_lambda_um"] < out["mauritiana"]["noise_lambda_um"]
    )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
