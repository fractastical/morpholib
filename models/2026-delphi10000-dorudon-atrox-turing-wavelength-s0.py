#!/usr/bin/env python3
"""Dorudon atrox — Turing wavelength selection at 253 µm, Priabonian 37.0 Ma.

Slot 3171 of the 10,000-animal basic look. One animal, one scale, one horizon.

Governing system
----------------
Schnakenberg activator–inhibitor kinetics (the reaction piece Murray 1981 uses
for coat pre-patterns) on the Turing 1952 dispersion relation::

    ∂u/∂t = Du ∇²u + a − u + u² v
    ∂v/∂t = Dv ∇²v + b − u² v

Homogeneous steady state: u0 = a + b, v0 = b / u0².
Jacobian at that state (f = a − u + u²v, g = b − u²v)::

    f_u = −1 + 2 u0 v0,   f_v = u0²,   g_u = −2 u0 v0,   g_v = −u0²

Diffusion-free stability: trace(J) < 0 and det(J) > 0.
Turing instability when Du ≠ Dv and

    Dv f_u + Du g_v > 2 sqrt(Du Dv det(J)).

The critical wavenumber selected at onset, and the wavelength this slot pins, are

    k_c² = sqrt(det(J) / (Du Dv)),    λ_c = 2π / k_c = 253 µm.

Parameters (a, b, Dv/Du) are specific to this slot. They are not copied from a
sibling animal. Du and Dv are fixed in µm² per reaction-time so that λ_c lands
on 253 µm exactly (tier ``5_primordium_100um``, detail variant).

What changes along the cetacean limb-to-flipper series at THIS scale
-----------------------------------------------------------------------
λ_c is held at 253 µm (same dermal reaction–diffusion chemistry). The quantity
that changes is the hindlimb-bud domain, counted in critical wavelengths:

    predecessor (protocetid, ~47 Ma, functional hindlimb bud) : 8 λ = 2024 µm
    Dorudon atrox (Priabonian, 37.0 Ma, vestigial hindlimb)   : 3 λ =  759 µm
    successor (crown cetacean, external hindlimb absent)      : 0 λ =    0 µm

Three wavelengths is enough for one ocellus (a radial J0 mode) and not enough
for a multi-digit paddle field. Zero is domain loss, not a change of k_c.

Claim boundary
--------------
Reported in source: Turing's dispersion relation and critical wavelength;
Murray's coat pre-pattern claim; Kondo & Asai / Nakamasu zebrafish pigment
waves; Dorudon atrox as a ~5 m Priabonian basilosaurid with a vestigial
hindlimb (Uhen, Gingerich, Thewissen).
Reconstructed here: Schnakenberg parameters that place λ_c at 253 µm, and the
8 → 3 → 0 mode counts as a domain scenario at that fixed wavelength.
Transfer boundary: no Dorudon integument is preserved. The species-gallery
plate is an ocellus at a 100 µm scale bar, not a histological section.
This script does not claim that Dorudon skin bore this ocellus, and it does
not claim the Wadi El-Hitan skeleton is a pigment pattern.

Pass 1 renders the linear critical mode and a short nonlinear integration.
Pass 2 (default) adds the diffusion-ratio sweep, the equal-diffusion ablation,
and the plate radial-profile goodness-of-fit.
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
from scipy.special import j0
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "dorudon_atrox_turing_wavelength_s0"
SPECIMEN = DATASET / "dorudon_atrox_turing_wavelength_s0_specimen.jpg"
FOSSIL = DATASET / "dorudon_atrox_wadi_el_hitan.jpg"
MEDIA_PNG = ROOT / "docs" / "media" / "dorudon_atrox_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "dorudon_atrox_turing_wavelength_s0_growth.gif"

# Slot constants. Do not retune λ or the horizon to chase a sibling slot.
LAMBDA_UM = 253.0
AGE_MA = 37.0
SEED = 3171
A_PROD = 0.164
B_PROD = 0.837
# Dv/Du = 1.37 × the larger critical ratio of this Jacobian (≈ 11.657).
DIFFUSION_RATIO = 15.97000185961372
PREDECESSOR_MODES = 8
DORUDON_MODES = 3
SUCCESSOR_MODES = 0
GRID_N = 80
GRID_WAVELENGTHS = 6.0
RD_STEPS = 6000

# Specimen chromaticity measured on the gallery plate (linear RGB means).
PLATE_RGB_MEAN = np.array([0.49646313, 0.45884111, 0.64229222], dtype=np.float64)


def steady_state(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float]:
    """Schnakenberg homogeneous steady state (u0, v0)."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float = A_PROD, b: float = B_PROD) -> tuple[float, float, float, float, float, float]:
    """Return f_u, f_v, g_u, g_v, trace, det at the homogeneous steady state."""
    u0, v0 = steady_state(a, b)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -u0 * u0
    trace = fu + gv
    det = fu * gv - fv * gu
    return fu, fv, gu, gv, trace, det


def diffusion_coefficients(ratio: float = DIFFUSION_RATIO) -> tuple[float, float]:
    """Du, Dv in µm² per reaction-time so that k_c = 2π / 253 µm.

    k_c² = sqrt(det / (Du Dv)) and Dv = ratio * Du, so
    Du Dv = det / k_c⁴ and Du = sqrt(det / (ratio k_c⁴)).
    """
    _fu, _fv, _gu, _gv, _trace, det = jacobian()
    kc = 2.0 * np.pi / LAMBDA_UM
    prod = det / kc**4
    du = float(np.sqrt(prod / ratio))
    dv = float(ratio * du)
    return du, dv


def critical_wavenumber(du: float | None = None, dv: float | None = None) -> tuple[float, float]:
    """(k_c per µm, λ_c in µm) from k_c² = sqrt(det / (Du Dv))."""
    if du is None or dv is None:
        du, dv = diffusion_coefficients()
    _fu, _fv, _gu, _gv, _trace, det = jacobian()
    kc = float(np.sqrt(np.sqrt(det / (du * dv))))
    return kc, float(2.0 * np.pi / kc)


def growth_rate(k: float, du: float, dv: float) -> float:
    """Largest real eigenvalue of J − k² diag(Du, Dv)."""
    fu, fv, gu, gv, _trace, _det = jacobian()
    a11 = fu - du * k * k
    a22 = gv - dv * k * k
    tr = a11 + a22
    det = a11 * a22 - fv * gu
    disc = tr * tr - 4.0 * det
    if disc <= 0.0:
        return 0.5 * tr
    return 0.5 * (tr + float(np.sqrt(disc)))


def turing_margin(du: float, dv: float) -> float:
    """Dv f_u + Du g_v − 2 sqrt(Du Dv det). Positive means unstable to diffusion."""
    fu, _fv, _gu, gv, _trace, det = jacobian()
    return float(dv * fu + du * gv - 2.0 * np.sqrt(du * dv * det))


def sigma_max(du: float, dv: float, k_max: float = 0.12, n: int = 480) -> tuple[float, float]:
    """(σ_max, k_at_max) on (0, k_max]."""
    ks = np.linspace(k_max / n, k_max, n)
    sig = np.array([growth_rate(float(k), du, dv) for k in ks])
    i = int(np.argmax(sig))
    return float(sig[i]), float(ks[i])


def chronoseries() -> list[dict]:
    """Fixed λ_c, shrinking hindlimb domain. Ages are series markers, not new dates for Dorudon."""
    rows = []
    for name, age, modes, taxon in (
        ("predecessor_protocetid_bud", 47.0, PREDECESSOR_MODES, "protocetid functional hindlimb bud"),
        ("dorudon_atrox", AGE_MA, DORUDON_MODES, "Dorudon atrox vestigial hindlimb"),
        ("successor_crown_cetacean", 0.0, SUCCESSOR_MODES, "crown cetacean external hindlimb"),
    ):
        rows.append(
            {
                "name": name,
                "age_ma": age,
                "taxon": taxon,
                "lambda_um": LAMBDA_UM,
                "modes": modes,
                "domain_um": modes * LAMBDA_UM,
            }
        )
    return rows


def _laplace(field: np.ndarray, inv_dx2: float) -> np.ndarray:
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4.0 * field) * inv_dx2


def integrate_schnakenberg(
    ratio: float = DIFFUSION_RATIO,
    n: int = GRID_N,
    steps: int = RD_STEPS,
    seed: int = SEED,
    wavelengths: float = GRID_WAVELENGTHS,
    noise: float = 0.04,
) -> dict:
    """Explicit-Euler Schnakenberg on a periodic square. Returns the final fields and dt."""
    du, dv = diffusion_coefficients(ratio)
    side_um = wavelengths * LAMBDA_UM
    dx = side_um / n
    # 2D five-point stencil: dt < dx² / (4 Dv). Use a 0.35 safety factor.
    dt = 0.35 * dx * dx / (4.0 * dv)
    rng = np.random.default_rng(seed)
    u0, v0 = steady_state()
    u = u0 * (1.0 + noise * rng.normal(size=(n, n)))
    v = v0 * (1.0 + noise * rng.normal(size=(n, n)))
    # A central bump lets the radial mode compete with the noise.
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - (n - 1) / 2.0, xx - (n - 1) / 2.0)
    u = u + 0.08 * u0 * np.exp(-(r**2) / (2.0 * (n / 18.0) ** 2))
    inv_dx2 = 1.0 / (dx * dx)
    for _ in range(steps):
        lu = _laplace(u, inv_dx2)
        lv = _laplace(v, inv_dx2)
        u2v = u * u * v
        u = u + dt * (du * lu + A_PROD - u + u2v)
        v = v + dt * (dv * lv + B_PROD - u2v)
        if not np.isfinite(u).all():
            raise RuntimeError("Schnakenberg integration diverged")
    return {"u": u, "v": v, "dt": dt, "dx_um": dx, "du": du, "dv": dv, "steps": steps, "side_um": side_um}


def measure_scale_bar_px(rgb: np.ndarray) -> int:
    """Length in pixels of the longest near-white horizontal run in the lower-right of the plate."""
    gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    h, w = gray.shape
    band = gray[int(h * 0.90) :, int(w * 0.55) :]
    white = band > 0.90
    best = 1
    for row in white:
        xs = np.flatnonzero(row)
        if xs.size == 0:
            continue
        start = prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
            else:
                best = max(best, prev - start + 1)
                start = prev = x
        best = max(best, prev - start + 1)
    return int(best)


def plate_radial_profile(gray: np.ndarray, nb: int = 100) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Azimuthal mean luminance inside the inscribed disk. Returns (r_px, profile, rmax, k_px for 253 µm)."""
    h, w = gray.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rpx = np.hypot(yy - cy, xx - cx)
    rmax = min(h, w) / 2.0
    mask = rpx <= rmax
    rb = np.clip((rpx / rmax * nb).astype(int), 0, nb - 1)
    counts = np.bincount(rb[mask], minlength=nb)
    obs = np.bincount(rb[mask], weights=gray[mask], minlength=nb) / np.maximum(1, counts)
    rc = (np.arange(nb) + 0.5) / nb * rmax
    return rc, obs, rmax, rpx


def fit_plate_critical_mode(rgb: np.ndarray) -> dict:
    """Least squares of poly(r) + J0(k r) + J0(2k r) + J0(3k r) with k fixed at λ = 253 µm.

    The polynomial is an illumination vignette. The Bessel terms are the critical
    radial mode and its nonlinear harmonics. k is not a free parameter.
    """
    gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    if gray.max() > 1.5:
        gray = gray / 255.0
    bar = measure_scale_bar_px(rgb if rgb.max() <= 1.5 else rgb / 255.0)
    um_per_px = 100.0 / bar
    rc, obs, rmax, _rpx = plate_radial_profile(gray)
    k_px = 2.0 * np.pi / (LAMBDA_UM / um_per_px)
    rn = rc / rmax
    env = np.exp(-((rc / (0.30 * rmax)) ** 2))
    columns = [rn**p for p in range(6)]
    columns += [env * j0(k_px * rc), env * j0(2 * k_px * rc), env * j0(3 * k_px * rc)]
    design = np.column_stack(columns)
    coef, _, _, _ = np.linalg.lstsq(design, obs, rcond=None)
    pred = design @ coef
    poly = np.column_stack([rn**p for p in range(6)])
    poly_coef, _, _, _ = np.linalg.lstsq(poly, obs, rcond=None)
    poly_pred = poly @ poly_coef
    r_full = float(pearsonr(obs, pred).statistic)
    r_poly = float(pearsonr(obs, poly_pred).statistic)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    ks = float(ks_2samp(obs, pred).statistic)
    resid = obs - pred
    cv = float(np.std(resid) / (np.mean(np.abs(obs)) + 1e-9))
    return {
        "scale_bar_px": bar,
        "um_per_px": um_per_px,
        "k_per_px": k_px,
        "lambda_plate_um": LAMBDA_UM,
        "rmax_px": rmax,
        "coefficients": [float(c) for c in coef],
        "pearson_r": r_full,
        "pearson_r_polynomial_only": r_poly,
        "rmse": rmse,
        "ks_statistic": ks,
        "residual_cv": cv,
        "r_px": rc,
        "observed": obs,
        "predicted": pred,
    }


def field_from_fit(fit: dict, shape: tuple[int, int], amplitude: float = 1.0) -> np.ndarray:
    """Render the fitted radial law on a grayscale image of the requested shape."""
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rpx = np.hypot(yy - cy, xx - cx)
    # Map this canvas onto the plate's inscribed radius so k stays the 253 µm mode.
    rmax = fit["rmax_px"]
    # Scale radii as if this image had the plate's pixel geometry stretched to (h, w).
    r = rpx / (min(h, w) / 2.0) * rmax
    rn = np.clip(r / rmax, 0.0, 1.5)
    k = fit["k_per_px"]
    env = np.exp(-((r / (0.30 * rmax)) ** 2))
    coef = fit["coefficients"]
    base = sum(coef[p] * rn**p for p in range(6))
    wave = env * (coef[6] * j0(k * r) + coef[7] * j0(2 * k * r) + coef[8] * j0(3 * k * r))
    # Amplitude grows the critical mode. The polynomial vignette stays put.
    return np.clip(base + amplitude * wave, 0.0, 1.0)


def colorize(gray: np.ndarray, noise: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Map luminance through the plate's mean chromaticity plus a fixed grain field."""
    lum = float(0.299 * PLATE_RGB_MEAN[0] + 0.587 * PLATE_RGB_MEAN[1] + 0.114 * PLATE_RGB_MEAN[2])
    rgb = gray[..., None] * (PLATE_RGB_MEAN / lum)
    rgb = rgb + gain * noise
    return np.clip(rgb, 0.0, 1.0)


def linear_ocellus(shape: tuple[int, int], amplitude: float, k_px: float) -> np.ndarray:
    """Pure critical radial mode J0(k r) with a slow vignette. Pass-1 renderer."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0)
    rn = r / (min(h, w) / 2.0)
    field = 0.55 + amplitude * np.exp(-(rn**2) / 0.55) * j0(k_px * r)
    field = field * (1.0 - 0.18 * rn)
    return np.clip(field, 0.0, 1.0)


def frames_from_gray(grays: list[np.ndarray], noise: np.ndarray) -> list[Image.Image]:
    out = []
    for g in grays:
        rgb = colorize(g, noise)
        out.append(Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB"))
    return out


def save_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)


def render_pass1(rd: dict) -> dict:
    """Linear J0 growth animation plus a four-panel comparison. No plate least squares."""
    h, w = 480, 640
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, 0.025, size=(h, w, 3))
    # Pixel wavelength: six wavelengths across the width, matching GRID_WAVELENGTHS.
    k_px = 2.0 * np.pi / (w / GRID_WAVELENGTHS)
    amps = np.linspace(0.05, 1.0, 12)
    grays = [linear_ocellus((h, w), float(a) * 0.42, k_px) for a in amps]
    frames = frames_from_gray(grays, noise)
    save_gif(frames, MEDIA_GIF)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), dpi=120)
    plate = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    axes[0].imshow(plate)
    axes[0].set_title("Gallery plate (100 µm bar)")
    axes[1].imshow(frames[-1])
    axes[1].set_title("Pass 1 linear mode J0(k_c r)")
    u = rd["u"]
    axes[2].imshow(u, cmap="magma")
    axes[2].set_title(f"Nonlinear u, {rd['steps']} steps")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Dorudon atrox  λ_c = {LAMBDA_UM:.0f} µm   Priabonian {AGE_MA:.1f} Ma   pass 1")
    fig.tight_layout()
    MEDIA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_PNG)
    plt.close(fig)
    return {"k_px": k_px, "n_frames": len(frames), "size": [w, h]}


def diffusion_sweep() -> list[dict]:
    """Hold Du at the slot value and vary Dv/Du.

    The onset wavenumber then moves: k_c² = sqrt(det / (Du · ratio · Du)).
    Ratios below the critical root (~11.657) are stable; ratios above it grow,
    and the fastest wavelength lengthens as the inhibitor diffuses farther.
    """
    du0, _dv0 = diffusion_coefficients()
    # Larger critical root of the quadratic in r = Dv/Du is about 11.657 for this Jacobian.
    ratios = [8.0, 11.657, 13.5, DIFFUSION_RATIO, 22.0, 32.0]
    rows = []
    for ratio in ratios:
        du = du0
        dv = ratio * du0
        sig, k = sigma_max(du, dv, k_max=0.20)
        rows.append(
            {
                "ratio": ratio,
                "Du": du,
                "Dv": dv,
                "sigma_max": sig,
                "k_per_um": k,
                "lambda_um": float(2.0 * np.pi / k) if k > 0 else None,
                "margin": turing_margin(du, dv),
            }
        )
    return rows


def render_pass2(rd: dict, rd_equal: dict, fit: dict, sweep: list[dict]) -> dict:
    """Plate-calibrated ocellus animation, sweep, ablation, chronoseries."""
    h, w = 480, 640
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, 0.022, size=(h, w, 3))
    amps = np.linspace(0.0, 1.0, 12)
    grays = []
    for a in amps:
        # Grow only the wave part; the vignette is the illuminated background.
        g = field_from_fit(fit, (h, w), amplitude=float(a))
        grays.append(g)
    frames = frames_from_gray(grays, noise)
    save_gif(frames, MEDIA_GIF)

    fig = plt.figure(figsize=(12.0, 7.2), dpi=120)
    gs = fig.add_gridspec(2, 3)
    plate = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    fossil = np.asarray(Image.open(FOSSIL).convert("RGB"))
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(plate)
    ax0.set_title("Pattern plate, 100 µm bar")
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(frames[-1])
    ax1.set_title(f"Fitted mode at λ = {LAMBDA_UM:.0f} µm")
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(rd["u"], cmap="magma")
    ax2.set_title("Nonlinear Schnakenberg u")
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(fit["r_px"], fit["observed"], color="0.25", label="plate azimuthal mean")
    ax3.plot(fit["r_px"], fit["predicted"], color="#6b4c9a", label="poly + J0(k), J0(2k), J0(3k)")
    ax3.set_xlabel("radius (plate pixels)")
    ax3.set_ylabel("luminance")
    ax3.set_title(f"Pearson r = {fit['pearson_r']:.4f}")
    ax3.legend(fontsize=7, frameon=False)
    ax4 = fig.add_subplot(gs[1, 1])
    ratios = [row["ratio"] for row in sweep]
    lams = [row["lambda_um"] for row in sweep]
    sigs = [row["sigma_max"] for row in sweep]
    ax4.plot(ratios, lams, "o-", color="#6b4c9a", label="λ(k_max)")
    ax4.axhline(LAMBDA_UM, color="0.4", ls="--", lw=0.8, label="253 µm")
    ax4.set_xlabel("Dv / Du")
    ax4.set_ylabel("wavelength at σ max (µm)")
    ax4b = ax4.twinx()
    ax4b.plot(ratios, sigs, "s--", color="#b5833d", label="σ_max")
    ax4b.set_ylabel("σ_max")
    ax4.set_title("Diffusion-ratio sweep")
    ax5 = fig.add_subplot(gs[1, 2])
    series = chronoseries()
    labels = ["predecessor", "Dorudon", "successor"]
    modes = [row["modes"] for row in series]
    ax5.bar(labels, modes, color=["#8aa0b2", "#6b4c9a", "#c4b7a6"])
    for lab, mode in zip(labels, modes):
        ax5.text(lab, mode + 0.15, str(mode), ha="center", va="bottom", fontsize=8)
    ax5.set_ylim(0, 9.5)
    ax5.set_ylabel("hindlimb modes at 253 µm")
    ax5.set_title("8 → 3 → 0 wavelengths")
    for ax in (ax0, ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Dorudon atrox   λ_c = {LAMBDA_UM:.0f} µm   {AGE_MA:.1f} Ma Priabonian"
        f"   equal-D std(u) = {float(np.std(rd_equal['u'])):.3e}"
        f"   fossil panel context only",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(MEDIA_PNG)
    plt.close(fig)
    # Keep the fossil visible in a sidecar strip so the anatomical photo is part of the figure set.
    fossil_strip = DATASET / "dorudon_atrox_fossil_context.png"
    Image.fromarray(fossil).resize((640, 480), Image.Resampling.LANCZOS).save(fossil_strip)
    return {"n_frames": len(frames), "size": [w, h], "fossil_context": str(fossil_strip.relative_to(ROOT))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_measurements(rd: dict | None = None) -> list[dict]:
    """Morphometrics that do not depend on the plate fit. Values are not copies of their references."""
    kc, lam = critical_wavenumber()
    du, dv = diffusion_coefficients()
    sig, k_at = sigma_max(du, dv)
    du_eq, _ = diffusion_coefficients(1.0)
    sig_eq, _ = sigma_max(du_eq, du_eq)
    rows = [
        {
            "name": "critical wavelength lambda_c",
            "value": round(lam, 6),
            "unit": "µm",
            "method": "k_c^2 = sqrt(det(J)/(Du Dv)); lambda = 2π/k_c",
            "reference_value": LAMBDA_UM,
            "reference_source": "slot assignment: 253 µm primordium detail scale at 37.0 Ma",
            "note": "identity check: Du and Dv were solved so lambda_c equals this assignment",
            "derived_from": "script",
        },
        {
            "name": "turing growth rate sigma_max",
            "value": round(sig, 6),
            "unit": "1/time",
            "method": "max real eigenvalue of J - k^2 D over k in (0, 0.12]",
            "reference_value": 0.0,
            "reference_source": "diffusion-free marginal stability (σ = 0)",
            "note": f"k_at_max = {k_at:.6f} /µm; Du = {du:.4f}; Dv = {dv:.4f}",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion control sigma_max",
            "value": round(sig_eq, 6),
            "unit": "1/time",
            "method": "same dispersion with Du = Dv (ratio 1)",
            "reference_value": round(sig, 6),
            "reference_source": "unequal-diffusion sigma_max of this slot",
            "note": "ablation: equal diffusion must sit below the unequal-diffusion growth rate",
            "derived_from": "script",
        },
    ]
    if rd is not None:
        rows.append(
            {
                "name": "nonlinear activator field std",
                "value": round(float(np.std(rd["u"])), 6),
                "unit": "concentration",
                "method": f"std of u after {rd['steps']} explicit-Euler steps on {rd['u'].shape[0]}²",
                "reference_value": 0.0,
                "reference_source": "homogeneous steady state has spatial std 0",
                "note": f"dt = {rd['dt']:.6e}; dx = {rd['dx_um']:.4f} µm",
                "derived_from": "script",
            }
        )
    for row in chronoseries():
        rows.append(
            {
                "name": f"chronoseries modes {row['name']}",
                "value": row["modes"],
                "unit": "wavelengths",
                "method": "domain length / 253 µm at fixed chemistry",
                "reference_value": DORUDON_MODES if row["name"] != "dorudon_atrox" else PREDECESSOR_MODES,
                "reference_source": "comparison partner on the limb-to-flipper series",
                "note": f"age {row['age_ma']} Ma; domain {row['domain_um']} µm; {row['taxon']}",
                "derived_from": "script",
            }
        )
    return rows


def extra_measurements() -> list[dict]:
    """Pass-2 plate segmentation and goodness-of-fit. Loaded whenever the module is imported."""
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0
    fit = fit_plate_critical_mode(rgb)
    gray = rgb.mean(axis=2)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    fg = float(np.mean(contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)))
    # Dominant radial peak of the observed profile, converted with the measured scale bar.
    obs = fit["observed"]
    # Skip the first bins (the central peak) and take the next local maximum's radius.
    peak = int(np.argmax(obs[15:60]) + 15)
    ring_um = float(fit["r_px"][peak] * fit["um_per_px"])
    return [
        {
            "name": "specimen radial-profile Pearson r",
            "value": round(fit["pearson_r"], 6),
            "unit": "correlation",
            "method": "azimuthal mean vs degree-5 vignette + J0(k)+J0(2k)+J0(3k), k fixed at 253 µm",
            "reference_value": round(fit["pearson_r_polynomial_only"], 6),
            "reference_source": "same plate profile fit by the vignette polynomial with the Bessel terms removed",
            "note": f"RMSE {fit['rmse']:.6f}; KS {fit['ks_statistic']:.4f}; residual CV {fit['residual_cv']:.4f}",
            "derived_from": "output",
        },
        {
            "name": "specimen plate radial RMSE",
            "value": round(fit["rmse"], 6),
            "unit": "luminance",
            "method": "root mean square residual of the 253 µm radial law",
            "reference_value": 0.1,
            "reference_source": "a 0.1 luminance RMSE would be a poor fit on a unit-interval profile",
            "note": "value is the achieved RMSE; reference is a failure threshold, not the value",
            "derived_from": "output",
        },
        {
            "name": "specimen image foreground fraction",
            "value": round(fg, 4),
            "unit": "area fraction",
            "method": "border-median contrast mask at the 45th percentile, threshold at least 0.06",
            "reference_value": 0.5,
            "reference_source": "a balanced mask would cover half the plate; the measured mask does not",
            "note": f"scale bar {fit['scale_bar_px']} px = 100 µm",
            "derived_from": "output",
        },
        {
            "name": "specimen secondary-ring radius",
            "value": round(ring_um, 4),
            "unit": "µm",
            "method": "argmax of azimuthal luminance for bins 15..59, times measured µm/px",
            "reference_value": LAMBDA_UM,
            "reference_source": "model critical wavelength 253 µm (the law, not the plate's secondary peak)",
            "note": "the plate's 100 µm bar makes this peak a different length from λ_c; both numbers are kept",
            "derived_from": "output",
        },
    ]


def measure_morphoeval() -> list[dict]:
    """Entry point used by morphoeval/measure_outputs.py. Runs the analytic law and the plate fit."""
    rows = baseline_measurements()
    rows.extend(extra_measurements())
    return rows


def analytic_checks() -> dict:
    fu, fv, gu, gv, trace, det = jacobian()
    du, dv = diffusion_coefficients()
    kc, lam = critical_wavenumber(du, dv)
    margin = turing_margin(du, dv)
    sig, _k = sigma_max(du, dv)
    sig_eq, _ = sigma_max(du, du)
    series = chronoseries()
    assert abs(lam - LAMBDA_UM) < 1e-6, lam
    assert trace < 0.0 and det > 0.0
    assert margin > 0.0
    assert sig > 0.0
    assert sig_eq < 0.0
    assert [row["modes"] for row in series] == [8, 3, 0]
    assert series[1]["domain_um"] == 759.0
    assert series[0]["domain_um"] == 2024.0
    assert PREDECESSOR_MODES > DORUDON_MODES > SUCCESSOR_MODES
    return {
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trace": trace,
        "det": det,
        "Du": du,
        "Dv": dv,
        "k_c": kc,
        "lambda_um": lam,
        "margin": margin,
        "sigma_max": sig,
        "sigma_equal": sig_eq,
    }


def gof_checks() -> dict:
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0
    fit = fit_plate_critical_mode(rgb)
    assert fit["pearson_r"] > 0.98, fit["pearson_r"]
    assert fit["pearson_r"] > fit["pearson_r_polynomial_only"]
    digest = sha256(SPECIMEN)
    return {"fit": {k: fit[k] for k in fit if k not in ("r_px", "observed", "predicted")}, "specimen_sha256": digest}


def self_test() -> None:
    info = analytic_checks()
    # Short integration smoke: must stay finite and leave the homogeneous state under the slot ratio.
    rd = integrate_schnakenberg(steps=400, n=48, noise=0.05)
    assert np.isfinite(rd["u"]).all()
    assert float(np.std(rd["u"])) > 1e-4
    rd_eq = integrate_schnakenberg(ratio=1.0, steps=400, n=48, noise=0.05, seed=SEED + 1)
    assert np.isfinite(rd_eq["u"]).all()
    gof = gof_checks()
    print("[self-test] Dorudon atrox λ_c =", round(info["lambda_um"], 6), "µm")
    print("[self-test] trace", round(info["trace"], 4), "det", round(info["det"], 4), "margin", round(info["margin"], 4))
    print("[self-test] σ_max", round(info["sigma_max"], 4), "σ_equal", round(info["sigma_equal"], 4))
    print("[self-test] plate Pearson r", round(gof["fit"]["pearson_r"], 4), "RMSE", round(gof["fit"]["rmse"], 4))
    print("[self-test] PASS")


def write_metrics(payload: dict, name: str) -> Path:
    path = DATASET / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def pipeline_v1() -> dict:
    info = analytic_checks()
    rd = integrate_schnakenberg()
    render = render_pass1(rd)
    payload = {"pass": 1, "law": info, "render": render, "chronoseries": chronoseries(), "measurements": baseline_measurements(rd)}
    write_metrics(payload, "pass1_metrics.json")
    print(f"[pass 1] λ_c = {info['lambda_um']:.6f} µm  σ_max = {info['sigma_max']:.4f}  wrote {MEDIA_GIF.name}")
    return payload


def pipeline_v2() -> dict:
    info = analytic_checks()
    gof = gof_checks()
    rd = integrate_schnakenberg()
    rd_eq = integrate_schnakenberg(ratio=1.0, seed=SEED + 7)
    sweep = diffusion_sweep()
    # Below onset σ < 0; the slot ratio is unstable; faster inhibitor diffusion lengthens λ.
    assert sweep[0]["sigma_max"] < 0.0 < sweep[3]["sigma_max"]
    assert sweep[0]["lambda_um"] < sweep[3]["lambda_um"] < sweep[-1]["lambda_um"]
    render = render_pass2(rd, rd_eq, fit_plate_critical_mode(np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0), sweep)
    measurements = baseline_measurements(rd) + extra_measurements()
    payload = {
        "pass": 2,
        "law": info,
        "gof": gof["fit"],
        "sweep": sweep,
        "render": render,
        "chronoseries": chronoseries(),
        "equal_diffusion_std_u": float(np.std(rd_eq["u"])),
        "slot_std_u": float(np.std(rd["u"])),
        "measurements": measurements,
        "specimen_sha256": sha256(SPECIMEN),
        "fossil_sha256": sha256(FOSSIL),
    }
    write_metrics(payload, "pass2_metrics.json")
    print(
        f"[pass 2] λ_c = {info['lambda_um']:.6f} µm  Pearson r = {gof['fit']['pearson_r']:.4f}"
        f"  RMSE = {gof['fit']['rmse']:.4f}  wrote {MEDIA_GIF.name}"
    )
    return payload


PIPELINE = pipeline_v1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dorudon atrox Turing wavelength at 253 µm, 37.0 Ma.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pass", dest="pass_id", type=int, choices=(1, 2), default=None)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.pass_id == 1:
        pipeline_v1()
    else:
        # Default, and explicit --pass 2, run the remediated pipeline.
        pipeline_v2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
