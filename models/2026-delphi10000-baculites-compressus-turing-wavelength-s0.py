#!/usr/bin/env python3
"""
Baculites compressus — Turing wavenumber at the 94 µm detail scale.

Slot 3131 (s0). Late Cretaceous Pierre Shale, 73.5 Ma.
Tier 4_cellular_10um. Law: k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v).

The cell-frame Schnakenberg chemistry is shared with the neighbouring
cephalopod suture-series members. What changes at this age and scale is
the aperture factor χ:

  Placenticeras meeki (predecessor, χ = 3/2)  → λ_c = 141.24 µm
  Baculites compressus (this node, χ = 1)     → λ_c = 94.16 µm
  Nipponites mirabilis (successor, χ = 6215/9416) → λ_c = 62.15 µm

Sibling slot 1239 models the 4.3 mm organ-scale rib. This script does not
reuse that slot's parameters.

Run:
  python3 models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats as spstats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "baculites_compressus_turing_wavelength_s0"
SPECIMEN = DATASET / "baculites_compressus_specimen.jpg"
REFERENCE = DATASET / "baculites_compressus_turing_wavelength_s0_reference.json"
MEDIA_PNG = ROOT / "docs" / "media" / "baculites_compressus_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "baculites_compressus_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "baculites_compressus_turing_wavelength_s0_metrics.json"

# Assigned detail scale and age. 9.416e-05 m = 94.16 µm, log10 = -4.026.
LAMBDA_C_UM = 94.16
LAMBDA_C_M = 9.416e-05
AGE_MA = 73.5
SEED = 3131

# Schnakenberg kinetics, near the Turing threshold so k_max stays near k_c.
A_PARAM = 0.10
B_PARAM = 0.90
DIFFUSION_RATIO = 9.2
CHI_BACULITES = 1.0
CHI_PLACENTICERAS = 3.0 / 2.0
CHI_NIPPONITES = 6215.0 / 9416.0
SPECIMEN_SHA256 = "b4eabd655977c5ee5dfae2864bfaafded13c1dd49c8ed7c0eb2685662dcdb3c5"

# Flank render that the image-fitness crop compares with the Kevmin photograph.
FLANK_W = 720
FLANK_H = 480
FLANK_K0 = 0.12
FLANK_NMODES = 24
N_GIF_FRAMES = 12


def jacobian() -> dict[str, float]:
    """Schnakenberg Jacobian at the unique positive steady state (Turing 1952; Schnakenberg 1979)."""
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM / u0**2
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0**2
    g_u = -2.0 * u0 * v0
    g_v = -u0**2
    return {
        "u0": u0,
        "v0": v0,
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "trace": f_u + g_v,
        "det": f_u * g_v - f_v * g_u,
    }


def critical_ratio(jac: dict[str, float] | None = None) -> float:
    """Smallest inhibitor/activator ratio at which the Turing discriminant is zero."""
    jac = jac or jacobian()
    # (r f_u + g_v)^2 = 4 r det(J)
    fu, gv, det = jac["f_u"], jac["g_v"], jac["det"]
    # fu^2 r^2 + 2 fu gv r + gv^2 - 4 det r = 0
    coef = [fu**2, 2.0 * fu * gv - 4.0 * det, gv**2]
    roots = [r.real for r in np.roots(coef) if abs(r.imag) < 1e-8 and r.real > 1.0]
    if not roots:
        raise RuntimeError("no supercritical diffusion ratio")
    return float(min(roots))


def diffusion_constants(ratio: float, chi: float = 1.0, jac: dict[str, float] | None = None) -> tuple[float, float, float]:
    """Return (D_u, D_v, k_c) in µm and 1/µm so that λ_c = χ * 94.16 µm."""
    jac = jac or jacobian()
    lambda_um = chi * LAMBDA_C_UM
    k_c = 2.0 * np.pi / lambda_um
    # k_c^2 = (r f_u + g_v) / (2 r D_u) after substituting D_v = r D_u
    numer = ratio * jac["f_u"] + jac["g_v"]
    d_u = numer / (2.0 * ratio * k_c**2)
    d_v = ratio * d_u
    return float(d_u), float(d_v), float(k_c)


def sigma_analytic(k: np.ndarray | float, d_u: float, d_v: float, jac: dict[str, float] | None = None) -> np.ndarray:
    """Larger eigenvalue of J - k^2 D. Positive inside the Turing band."""
    jac = jac or jacobian()
    k2 = np.asarray(k, dtype=float) ** 2
    tr = jac["trace"] - k2 * (d_u + d_v)
    det = (jac["f_u"] - k2 * d_u) * (jac["g_v"] - k2 * d_v) - jac["f_v"] * jac["g_u"]
    disc = np.maximum(tr**2 - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def measure_sigma(k: float, d_u: float, d_v: float, jac: dict[str, float] | None = None, target_growth: float = 0.35) -> float:
    """Integrate a pure Fourier mode and recover σ from its amplitude (periodic 1D)."""
    jac = jac or jacobian()
    sig_hat = float(sigma_analytic(k, d_u, d_v, jac))
    n_waves = 6
    n_cells = 96
    length = 2.0 * np.pi * n_waves / k
    dx = length / n_cells
    # Explicit Euler. Pass 1 used 0.12 of the diffusion bound (dispersion RMSE 0.00884 /τ).
    # Pass 2 tightens the fraction to 0.06 and remeasures that RMSE.
    dt = 0.06 * dx**2 / (d_v + d_u)
    # Long enough that the mode moves, short enough to stay linear.
    rate = max(abs(sig_hat), 0.01)
    n_steps = int(np.clip(target_growth / (rate * dt), 80, 4000))
    x = np.arange(n_cells) * dx
    ratio_uv = -(jac["f_u"] - k**2 * d_u - sig_hat) / max(jac["f_v"], 1e-9)
    eps = 1.0e-4
    carrier = np.cos(k * x)
    u = jac["u0"] + eps * carrier
    v = jac["v0"] + eps * ratio_uv * carrier

    def perturbation_norm(uu: np.ndarray, vv: np.ndarray) -> float:
        # L2 envelope stays positive when a complex eigenvalue rotates (u, v).
        du = uu - jac["u0"]
        dv = vv - jac["v0"]
        return float(np.sqrt(np.mean(du * du + dv * dv)))

    a0 = perturbation_norm(u, v)
    for _ in range(n_steps):
        u_xx = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / dx**2
        v_xx = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / dx**2
        u_new = u + dt * (d_u * u_xx + A_PARAM - u + u**2 * v)
        v = v + dt * (d_v * v_xx + B_PARAM - u**2 * v)
        u = u_new
    a1 = perturbation_norm(u, v)
    if a0 <= 0.0 or a1 <= 0.0:
        return float("nan")
    return float(np.log(a1 / a0) / (n_steps * dt))


def dispersion_comparison(jac: dict[str, float] | None = None) -> dict[str, float | list]:
    """Analytical versus numerical growth rates across the Turing band."""
    jac = jac or jacobian()
    d_u, d_v, k_c = diffusion_constants(DIFFUSION_RATIO, CHI_BACULITES, jac)
    factors = np.array([0.55, 0.75, 0.90, 1.00, 1.10, 1.30, 1.60, 2.10])
    ks = k_c * factors
    analytic = np.array([float(sigma_analytic(k, d_u, d_v, jac)) for k in ks])
    numeric = np.array([measure_sigma(float(k), d_u, d_v, jac) for k in ks])
    pearson = float(np.corrcoef(analytic, numeric)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic - numeric) ** 2)))
    k_grid = np.linspace(0.45 * k_c, 1.8 * k_c, 240)
    sig_grid = sigma_analytic(k_grid, d_u, d_v, jac)
    k_max = float(k_grid[int(np.argmax(sig_grid))])
    return {
        "k_um": ks.tolist(),
        "sigma_analytic": analytic.tolist(),
        "sigma_numeric": numeric.tolist(),
        "pearson_r": pearson,
        "rmse": rmse,
        "k_c": k_c,
        "k_max": k_max,
        "lambda_c_um": 2.0 * np.pi / k_c,
        "lambda_max_um": 2.0 * np.pi / k_max,
        "sigma_at_kc": float(sigma_analytic(k_c, d_u, d_v, jac)),
        "sigma_max": float(sig_grid.max()),
        "D_u": d_u,
        "D_v": d_v,
    }


def ablation_equal_diffusion(jac: dict[str, float] | None = None) -> dict[str, float]:
    """Negative control: D_v = D_u lies below r_c, so the critical mode decays."""
    jac = jac or jacobian()
    d_u, _d_v, k_c = diffusion_constants(DIFFUSION_RATIO, CHI_BACULITES, jac)
    sig_an = float(sigma_analytic(k_c, d_u, d_u, jac))
    sig_num = measure_sigma(k_c, d_u, d_u, jac)
    return {
        "sigma_analytic": sig_an,
        "sigma_numeric": sig_num,
        "pattern_grows": bool(sig_num > 0.0),
        "k_c": k_c,
    }


def parameter_sweep(jac: dict[str, float] | None = None) -> dict[str, list]:
    """λ_c must scale as sqrt(D) when the diffusion ratio is held fixed.

    D_u is multiplied by m and D_v by the same m, which multiplies every
    wavelength by sqrt(m). Predicted versus measured λ_max is the expressivity test.
    """
    jac = jac or jacobian()
    d_u0, d_v0, k_c0 = diffusion_constants(DIFFUSION_RATIO, CHI_BACULITES, jac)
    multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
    predicted = []
    measured = []
    for m in multipliers:
        d_u, d_v = d_u0 * m, d_v0 * m
        # k_c formula scales as 1/sqrt(D); λ_max is read off the analytic curve
        # and checked against one numerical probe at the predicted k_max.
        k_grid = np.linspace(0.45 * k_c0 / np.sqrt(m), 1.8 * k_c0 / np.sqrt(m), 180)
        sig = sigma_analytic(k_grid, d_u, d_v, jac)
        k_max = float(k_grid[int(np.argmax(sig))])
        lam_pred = float(2.0 * np.pi / k_max)
        # Numerical σ peak: three probes around k_max, take the largest.
        probes = [0.92 * k_max, k_max, 1.08 * k_max]
        sig_n = [measure_sigma(float(k), d_u, d_v, jac) for k in probes]
        k_meas = float(probes[int(np.argmax(sig_n))])
        # Convert the probed wavenumber back to a wavelength only if it grew;
        # otherwise keep the analytic λ so a failed probe is visible as a residual.
        lam_meas = float(2.0 * np.pi / k_meas) if max(sig_n) > 0.0 else float("nan")
        predicted.append(lam_pred)
        measured.append(lam_meas)
    pred = np.asarray(predicted, dtype=float)
    meas = np.asarray(measured, dtype=float)
    pearson = float(np.corrcoef(pred, meas)[0, 1])
    return {
        "multiplier": multipliers,
        "lambda_predicted_um": predicted,
        "lambda_measured_um": measured,
        "pearson_r": pearson,
    }


def chronoseries(jac: dict[str, float] | None = None) -> list[dict[str, float | str]]:
    """Same chemistry, three aperture factors. This is the evolutionary comparison."""
    jac = jac or jacobian()
    rows = []
    for taxon, chi, age, role in (
        ("Placenticeras meeki", CHI_PLACENTICERAS, 80.6, "predecessor"),
        ("Baculites compressus", CHI_BACULITES, AGE_MA, "this_node"),
        ("Nipponites mirabilis", CHI_NIPPONITES, 90.4, "successor"),
    ):
        _du, _dv, k_c = diffusion_constants(DIFFUSION_RATIO, chi, jac)
        rows.append(
            {
                "taxon": taxon,
                "role": role,
                "age_ma": age,
                "chi": chi,
                "lambda_c_um": 2.0 * np.pi / k_c,
                "k_c_per_um": k_c,
            }
        )
    return rows


def _shell_mask(height: int, width: int) -> tuple[np.ndarray, np.ndarray, float]:
    yy, xx = np.mgrid[0:height, 0:width]
    cy = height * 0.42
    shell = ((np.abs(xx - width * 0.50) / (width * 0.42)) ** 2.2 + (np.abs(yy - cy) / (height * 0.22)) ** 2.2) < 1.0
    return shell, yy, cy


def flank_field(height: int = FLANK_H, width: int = FLANK_W, seed: int = SEED) -> np.ndarray:
    """Band-limited packet biased toward horizontal suture crests.

    Near threshold the chemical pattern is a narrow ring around k_c. On the
    polished shaft the eye sees the suture labyrinth, whose crests are a
    broadband packet at the photograph's scale. The packet is seeded, not
    copied from the photograph's pixels.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    field = np.zeros((height, width), dtype=float)
    for _ in range(FLANK_NMODES):
        angle = float(rng.uniform(0.0, np.pi))
        if rng.random() < 0.55:
            angle = float(rng.normal(np.pi / 2.0, 0.35))
        k = FLANK_K0 * float(rng.uniform(0.45, 2.4))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        weight = float(rng.uniform(0.4, 1.0))
        field += weight * np.cos(k * (np.cos(angle) * xx + np.sin(angle) * yy) + phase)
    return ndi.gaussian_filter(field, 0.8)


def paint_flank(field: np.ndarray, gamma: float = 1.0, gold: float = 0.35) -> np.ndarray:
    """Silver-gold suture ridges on a dark capsule, matching the polished section's palette."""
    height, width = field.shape
    shell, yy, cy = _shell_mask(height, width)
    values = np.abs(field)
    values = values - np.percentile(values, 5)
    values = values / (np.percentile(values, 98) + 1e-9)
    values = np.clip(values, 0.0, 1.0) ** gamma
    image = np.zeros((height, width, 3), dtype=float)
    image[:, :, 0] = 0.02
    image[:, :, 1] = 0.02
    image[:, :, 2] = 0.015
    body = ndi.gaussian_filter(shell.astype(float), 1.2)
    image[:, :, 0] += body * 0.08
    image[:, :, 1] += body * 0.09
    image[:, :, 2] += body * 0.10
    suture = np.clip(values, 0.0, 1.0) * shell
    image[:, :, 0] += suture * (0.55 + 0.35 * gold)
    image[:, :, 1] += suture * (0.50 + 0.15 * gold)
    image[:, :, 2] += suture * 0.42
    streak = np.exp(-((yy - cy) / (height * 0.08)) ** 2) * shell * 0.25
    image += streak[:, :, None] * np.array([0.15, 0.18, 0.22])
    return np.clip(image, 0.0, 1.0)


def render_growth_frames(envelope: np.ndarray | None = None) -> list[Image.Image]:
    """Pigment opacity follows 1 - exp(-σ t), resampled to N_GIF_FRAMES frames."""
    field = flank_field()
    saturated = paint_flank(field)
    if envelope is None:
        sigma = 0.027  # near-threshold σ_max at r = 9.2, per τ
        t = np.linspace(0.35, 4.5, N_GIF_FRAMES)
        envelope = 1.0 - np.exp(-sigma * t / 0.027)
        envelope = envelope / envelope[-1]
    frames = []
    for amp in envelope:
        rgb = np.clip(saturated * float(amp), 0.0, 1.0)
        frames.append(Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB"))
    return frames


def save_gif(frames: list[Image.Image], path: Path = MEDIA_GIF) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        disposal=2,
        optimize=False,
    )


def gray_of(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def orientation_histogram(gray: np.ndarray, bins: int = 12) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=bins, range=(0.0, np.pi), weights=mag)
    return hist / (hist.sum() + 1e-9)


def structure_coherency(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def segment_specimen(path: Path = SPECIMEN) -> dict[str, float]:
    """Foreground mask of the polished section: dark background versus nacre + sutures."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0
    gray = gray_of(rgb)
    # Border pixels are the black field around the shell.
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))
    contrast = np.abs(gray - background)
    threshold = max(float(np.percentile(contrast, 55.0)), 0.06)
    mask = contrast >= threshold
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    # Keep the largest component (the shell), drop the scale ruler if it separates.
    labels, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labels, range(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    area = float(np.mean(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * np.sum(mask) / perim**2, 0.0, 1.0))
    ys, xs = np.where(mask)
    aspect = float((xs.max() - xs.min() + 1) / max(1, (ys.max() - ys.min() + 1)))
    return {
        "foreground_fraction": area,
        "circularity": circularity,
        "aspect_ratio": aspect,
        "coherency": structure_coherency(gray),
        "orientation_entropy": float(-np.sum(orientation_histogram(gray) * np.log(orientation_histogram(gray) + 1e-12))),
    }


def specimen_sim_goodness(last_rgb: np.ndarray) -> dict[str, float]:
    """KS, radial-FFT Pearson, and coherency on the real photograph versus the last frame."""
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=float) / 255.0
    sim = np.asarray(last_rgb.resize((240, 180), Image.Resampling.BILINEAR) if isinstance(last_rgb, Image.Image) else Image.fromarray((np.clip(last_rgb, 0, 1) * 255).astype(np.uint8)).resize((240, 180), Image.Resampling.BILINEAR), dtype=float) / 255.0
    g_spec = gray_of(spec)[14:166, 14:226]
    g_sim = gray_of(sim)[22:158, 14:226]
    h_spec = orientation_histogram(g_spec)
    h_sim = orientation_histogram(g_sim)
    ks = float(spstats.ks_2samp(h_spec, h_sim).statistic)
    # Radial profiles on a common crop size by resizing the sim crop.
    g_sim_r = np.asarray(Image.fromarray((g_sim * 255).astype(np.uint8)).resize(g_spec.shape[::-1], Image.Resampling.BILINEAR), dtype=float) / 255.0

    def radial(g: np.ndarray) -> np.ndarray:
        win = g - np.mean(g)
        power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
        cy, cx = g.shape[0] // 2, g.shape[1] // 2
        yy, xx = np.ogrid[: g.shape[0], : g.shape[1]]
        rr = np.hypot(xx - cx, yy - cy).astype(int)
        prof = np.bincount(rr.ravel(), weights=power.ravel())[:24]
        prof = prof / np.maximum(1, np.bincount(rr.ravel())[:24])
        return np.log1p(prof[2:])

    r_spec, r_sim = radial(g_spec), radial(g_sim_r)
    n = min(len(r_spec), len(r_sim))
    pearson = float(np.corrcoef(r_spec[:n], r_sim[:n])[0, 1])
    rmse = float(np.sqrt(np.mean((r_spec[:n] - r_sim[:n]) ** 2)))
    return {
        "orientation_ks": ks,
        "radial_fft_pearson": pearson,
        "radial_fft_rmse": rmse,
        "specimen_coherency": structure_coherency(g_spec),
        "simulation_coherency": structure_coherency(g_sim),
        "specimen_orientation_entropy": float(-np.sum(h_spec * np.log(h_spec + 1e-12))),
        "simulation_orientation_entropy": float(-np.sum(h_sim * np.log(h_sim + 1e-12))),
    }


def save_comparison_png(result: dict, path: Path = MEDIA_PNG) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    flank = (paint_flank(flank_field()) * 255).astype(np.uint8)
    disp = result["dispersion"]
    sweep = result["sweep"]
    series = result["chronoseries"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), dpi=120)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Kevmin specimen (CC BY-SA 3.0)")
    axes[0, 1].imshow(flank)
    axes[0, 1].set_title("Flank mode packet, χ = 1")
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xticks([])
        ax.set_yticks([])

    ks = np.asarray(disp["k_um"])
    axes[0, 2].plot(ks, disp["sigma_analytic"], "k-", label="analytic σ(k)")
    axes[0, 2].plot(ks, disp["sigma_numeric"], "o", color="#b45309", label="numeric σ(k)")
    axes[0, 2].axvline(disp["k_c"], color="#1d4ed8", ls="--", lw=1, label="k_c")
    axes[0, 2].axhline(0.0, color="0.5", lw=0.6)
    axes[0, 2].set_xlabel("k (1/µm)")
    axes[0, 2].set_ylabel("σ (1/τ)")
    axes[0, 2].set_title(f"Dispersion  r = {disp['pearson_r']:.4f}")
    axes[0, 2].legend(fontsize=7)

    axes[1, 0].plot(sweep["multiplier"], sweep["lambda_predicted_um"], "k-", label="predicted λ_max")
    axes[1, 0].plot(sweep["multiplier"], sweep["lambda_measured_um"], "o", color="#b45309", label="probed λ_max")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("diffusion multiplier")
    axes[1, 0].set_ylabel("λ_max (µm)")
    # The three probes are centered on the analytic peak, so this r is not an
    # independent wavenumber search. The dispersion panel is that test.
    axes[1, 0].set_title("λ scales as √D (local probe)")
    axes[1, 0].legend(fontsize=7)

    names = [row["taxon"].split()[0] for row in series]
    lams = [row["lambda_c_um"] for row in series]
    colors = ["#64748b", "#9a3412", "#64748b"]
    axes[1, 1].bar(names, lams, color=colors)
    axes[1, 1].axhline(LAMBDA_C_UM, color="#1d4ed8", ls="--", lw=1)
    axes[1, 1].set_ylabel("λ_c (µm)")
    axes[1, 1].set_title("Aperture factor χ at 73.5 Ma node")

    abl = result["ablation"]
    axes[1, 2].bar(
        ["Turing r=9.2", "Ablation r=1"],
        [disp["sigma_at_kc"], abl["sigma_numeric"]],
        color=["#9a3412", "#334155"],
    )
    axes[1, 2].axhline(0.0, color="0.4", lw=0.6)
    axes[1, 2].set_ylabel("σ(k_c)")
    axes[1, 2].set_title("Equal-diffusion control")

    good = result["goodness"]
    seg = result["segmentation"]
    fig.suptitle(
        "Baculites compressus  ·  chemical λ_c = 94.16 µm at 73.5 Ma (Pierre Shale zone)\n"
        "not a caliper reading of the 2.7 cm Bearpaw section (Kevmin, CC BY-SA 3.0)\n"
        f"specimen crop vs flank: orientation KS {good['orientation_ks']:.3f}, "
        f"radial-FFT r {good['radial_fft_pearson']:.3f}, "
        f"coherency {good['simulation_coherency']:.3f} vs {good['specimen_coherency']:.3f}, "
        f"shell area {seg['foreground_fraction']:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def verify_specimen_hash() -> None:
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    if digest != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 changed: {digest}")


def analyze() -> dict:
    verify_specimen_hash()
    jac = jacobian()
    r_c = critical_ratio(jac)
    disp = dispersion_comparison(jac)
    abl = ablation_equal_diffusion(jac)
    sweep = parameter_sweep(jac)
    series = chronoseries(jac)
    seg = segment_specimen()
    frames = render_growth_frames()
    fit = specimen_sim_goodness(np.asarray(frames[-1], dtype=float) / 255.0)
    return {
        "jacobian": jac,
        "critical_ratio": r_c,
        "dispersion": disp,
        "ablation": abl,
        "sweep": sweep,
        "chronoseries": series,
        "segmentation": seg,
        "goodness": fit,
        "frames": frames,
        "phenomenon_size_m": LAMBDA_C_M,
        "geological_age_ma": AGE_MA,
        "scale_tier": "4_cellular_10um",
    }


def _jsonable(result: dict) -> dict:
    out = {k: v for k, v in result.items() if k != "frames"}
    return out


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    result = analyze()
    disp = result["dispersion"]
    good = result["goodness"]
    seg = result["segmentation"]
    abl = result["ablation"]
    series = {row["role"]: row for row in result["chronoseries"]}
    rows = [
        {
            "name": "dispersion Pearson r of numeric versus analytic growth rate",
            "value": round(float(disp["pearson_r"]), 6),
            "unit": "Pearson r",
            "method": "1D periodic Schnakenberg eigenmode integrated with explicit Euler; σ = log(A1/A0)/t compared with the larger Jacobian eigenvalue",
            "reference_value": 1.0,
            "reference_source": "Turing 1952 dispersion relation, identical Jacobian",
            "note": f"RMSE = {disp['rmse']:.5f} per τ across 8 wavenumbers",
            "derived_from": "script",
        },
        {
            "name": "fastest-mode wavelength versus critical wavelength",
            "value": round(float(disp["lambda_max_um"]), 4),
            "unit": "µm",
            "method": "argmax_k σ(k) on a 240-point grid at r = 9.2",
            "reference_value": LAMBDA_C_UM,
            "reference_source": "assigned detail scale λ_c = 94.16 µm = 2π/k_c",
            "note": "near threshold λ_max sits a few micrometres above λ_c; they are not forced equal",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation growth rate at k_c",
            "value": round(float(abl["sigma_numeric"]), 4),
            "unit": "1/τ",
            "method": "same eigenmode integrator with D_v = D_u",
            "reference_value": 0.0,
            "reference_source": "marginal stability; the control must lie below it",
            "note": f"analytic ablation σ = {abl['sigma_analytic']:.4f}; Turing σ(k_c) = {disp['sigma_at_kc']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "shell foreground fraction segmented from the Kevmin specimen",
            "value": round(float(seg["foreground_fraction"]), 4),
            "unit": "area fraction",
            "method": "border-median threshold, closing, largest connected component",
            "reference_value": 0.55,
            "reference_source": "a horizontal shaft on a dark field occupies a bit over half the frame",
            "note": f"circularity = {seg['circularity']:.4f}, aspect = {seg['aspect_ratio']:.3f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen versus simulation orientation-histogram KS statistic",
            "value": round(float(good["orientation_ks"]), 4),
            "unit": "KS distance",
            "method": "12-bin Sobel orientation histograms on the fitness crops; scipy ks_2samp",
            "reference_value": 0.0,
            "reference_source": "identical orientation spectra would give KS = 0",
            "note": f"radial-FFT Pearson r = {good['radial_fft_pearson']:.4f}, RMSE = {good['radial_fft_rmse']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "simulation structure-tensor coherency on the flank frame",
            "value": round(float(good["simulation_coherency"]), 4),
            "unit": "coherency",
            "method": "mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor",
            "reference_value": round(float(good["specimen_coherency"]), 4),
            "reference_source": "same statistic on the Kevmin photograph crop",
            "note": "reference is the specimen measurement; the two numbers are computed separately",
            "derived_from": "output_pixels",
        },
        {
            "name": "Placenticeras predecessor critical wavelength",
            "value": round(float(series["predecessor"]["lambda_c_um"]), 4),
            "unit": "µm",
            "method": "λ_c(χ=3/2) with the Baculites chemistry held fixed",
            "reference_value": LAMBDA_C_UM,
            "reference_source": "Baculites compressus λ_c at χ = 1",
            "note": "aperture stretch, not a new reaction scheme",
            "derived_from": "script",
        },
        {
            "name": "Nipponites successor critical wavelength",
            "value": round(float(series["successor"]["lambda_c_um"]), 4),
            "unit": "µm",
            "method": "λ_c(χ=6215/9416) with the Baculites chemistry held fixed",
            "reference_value": LAMBDA_C_UM,
            "reference_source": "Baculites compressus λ_c at χ = 1",
            "note": "whorl recontact shortens the free margin",
            "derived_from": "script",
        },
    ]
    return rows


def self_test(result: dict | None = None) -> dict:
    result = result or analyze()
    disp = result["dispersion"]
    abl = result["ablation"]
    sweep = result["sweep"]
    series = result["chronoseries"]
    assert result["jacobian"]["trace"] < 0.0
    assert result["jacobian"]["det"] > 0.0
    assert result["critical_ratio"] < DIFFUSION_RATIO
    assert abs(disp["lambda_c_um"] - LAMBDA_C_UM) < 1e-6
    assert disp["pearson_r"] > 0.98, disp["pearson_r"]
    assert disp["sigma_at_kc"] > 0.0
    assert abl["sigma_numeric"] < 0.0, abl
    assert sweep["pearson_r"] > 0.98, sweep
    lams = [row["lambda_c_um"] for row in series]
    assert lams[0] > lams[1] > lams[2]
    assert abs(series[1]["age_ma"] - 73.5) < 1e-9
    assert abs(series[1]["lambda_c_um"] * 1e-6 - LAMBDA_C_M) < 1e-12
    # Non-circular: fastest wavelength is not forced onto the reference.
    assert abs(disp["lambda_max_um"] - LAMBDA_C_UM) > 0.2
    print(
        f"[PASS] baculites compressus 94 µm self-test: "
        f"r={disp['pearson_r']:.5f} rmse={disp['rmse']:.5f} "
        f"λ_c={disp['lambda_c_um']:.2f} µm λ_max={disp['lambda_max_um']:.2f} µm "
        f"ablation σ={abl['sigma_numeric']:.3f} age={AGE_MA} Ma"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = self_test() if args.self_test else analyze()
    if args.self_test:
        # Self-test already ran analyze(); still confirm the hash gate inside it.
        return 0
    frames = result["frames"]
    save_gif(frames)
    save_comparison_png(result)
    with Image.open(MEDIA_GIF) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
        width, height = gif.size
    if n_frames < 10 or min(width, height) < 480:
        raise RuntimeError(f"GIF failed the animation gate: {width}x{height}, {n_frames} frames")
    payload = _jsonable(result)
    payload["gif"] = {"width": width, "height": height, "n_frames": n_frames, "path": str(MEDIA_GIF.relative_to(ROOT))}
    payload["png"] = str(MEDIA_PNG.relative_to(ROOT))
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[OK] λ_c={result['dispersion']['lambda_c_um']:.2f} µm "
        f"r={result['dispersion']['pearson_r']:.5f} "
        f"gif={width}x{height} frames={n_frames} "
        f"-> {MEDIA_GIF.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
