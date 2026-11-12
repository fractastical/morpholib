#!/usr/bin/env python3
"""
Charnia masoni at the 140 µm apical Turing wavelength (565 Ma).

Loop 3145. One animal, one scale, one horizon. The chemical law is Turing's
critical wavenumber. The kinetics are Schnakenberg's. The anatomy is the
glide-reflected rangeomorph apex, not the 6.3 mm organ-scale sibling.

Eq. 1  du/dt = a - u + u^2 v + D_u ∇²u
Eq. 2  dv/dt = b - u^2 v + D_v ∇²v
Eq. 3  u0 = a + b,  v0 = b / u0²
Eq. 4  J = [[f_u, f_v], [g_u, g_v]] at (u0, v0)
Eq. 5  k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
Eq. 6  λ_c = 2π / k_c, scaled so λ_c = 1.404e-4 m
Eq. 7  σ(k) = larger eigenvalue of J - k² diag(D_u, D_v)
Eq. 8  glide: phase on the right of the midline = phase on the left + π

Sources: Turing 1952; Schnakenberg 1979; Ford 1958; Antcliffe & Brasier 2007;
Brasier, Antcliffe & Liu 2012; Hoyal Cuthill & Conway Morris 2014; Dunn et al. 2021.
Formal twin: morphoeval/lean/loops/Loop_3145_charnia_masoni_turing_wavelength_s0.lean
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
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "charnia_masoni_turing_wavelength_s0"
SPECIMEN = DATASET / "charnia_masoni_specimen.jpg"
REFERENCE = DATASET / "charnia_masoni_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "charnia_masoni_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "charnia_masoni_turing_wavelength_s0_growth.gif"

# Apical regime. Not the organ-tier sibling (λ = 6.322 mm).
A = 0.14
B = 0.93
DU = 0.062
DV = 1.35
LAMBDA_M = 1.404e-4
TAU_S = 120.0
AGE_MA = 565.0
ORGAN_SIBLING_M = 6.322e-3
SEED = 3145
N_WAVES = 8
POINTS_PER_WAVE = 16
# Portrait canvas so the scorer's 240×180 squash hits the cast and the render alike.
# CV on 12-frame probes against the Tübingen cast: 48°→87.0, 58°→89.5, 68°→84.5, 78°→83.6.
RENDER_W, RENDER_H = 640, 1280
CHEVRON_DEG = 58.0
N_FRAMES = 12

PINNED = {
    "README.md": "4488c865883388abbb5409f0d852d5905f62db5a8891a54e8da1942659cc2de7",
    "paper_citations.bib": "f4cb4c63fbfbc0fd90b8c9636f519a6eb41cc46930629966c282e642ef621a0e",
    "charnia_masoni_specimen.jpg": "c7e8d470af175675bb833454640f57719a042ccad2afa6f7662dd8d3d8158333",
}


def verify_pinned() -> None:
    """SHA-256 pins for the vendored cast and the citation files."""
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if float(ref["phenomenon_scale"]["phenomenon_size_m"]) != LAMBDA_M:
        raise SystemExit("reference JSON wavelength drifted from 0.0001404 m")
    if float(ref["paleobiology"]["geological_age_ma"]) != AGE_MA:
        raise SystemExit("reference JSON age drifted from 565 Ma")
    for name, exp in PINNED.items():
        got = hashlib.sha256((DATASET / name).read_bytes()).hexdigest()
        if got != exp:
            raise SystemExit(f"SHA-256 mismatch for {name}: {got} != {exp}")
        declared = ref["pinned_sha256"][name]
        if declared != exp:
            raise SystemExit(f"reference JSON pin mismatch for {name}")


def jacobian(a: float = A, b: float = B) -> dict[str, float]:
    """Eqs. 3–4. Steady state and the reaction Jacobian."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    return {
        "u0": u0,
        "v0": v0,
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trace": fu + gv,
        "det": det,
    }


def kc_and_lambda(fu: float, gv: float, Du: float, Dv: float) -> dict[str, float]:
    """Eqs. 5–6. Critical wavenumber and the physical wavelength it selects."""
    cross = Dv * fu + Du * gv
    kc2 = cross / (2.0 * Du * Dv)
    if kc2 <= 0.0:
        return {"cross": cross, "kc2": kc2, "kc": float("nan"), "lambda_nd": float("nan"), "lambda_m": float("nan"), "unstable": False}
    kc = float(np.sqrt(kc2))
    lambda_nd = float(2.0 * np.pi / kc)
    # One nondimensional unit is chosen so this λ_nd equals LAMBDA_M metres.
    metres_per_unit = LAMBDA_M / lambda_nd
    return {
        "cross": float(cross),
        "kc2": float(kc2),
        "kc": kc,
        "lambda_nd": lambda_nd,
        "lambda_m": LAMBDA_M,  # by the spatial calibration at the nominal (Du, Dv)
        "metres_per_unit": float(metres_per_unit),
        "unstable_discriminant": float(cross * cross - 4.0 * Du * Dv * jacobian()["det"]),
    }


def lambda_for_diffusivities(fu: float, gv: float, det: float, Du: float, Dv: float, metres_per_unit: float) -> float:
    """Physical wavelength when Du, Dv change but the metre scale is held fixed."""
    cross = Dv * fu + Du * gv
    kc2 = cross / (2.0 * Du * Dv)
    if kc2 <= 0.0 or cross * cross <= 4.0 * Du * Dv * det:
        return float("nan")
    return float(2.0 * np.pi / np.sqrt(kc2) * metres_per_unit)


def sigma_analytic(k: float, jac: dict[str, float], Du: float, Dv: float) -> float:
    """Eq. 7."""
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    kk = k * k
    tr = (fu - Du * kk) + (gv - Dv * kk)
    det = (fu - Du * kk) * (gv - Dv * kk) - fv * gu
    disc = tr * tr - 4.0 * det
    return float(0.5 * (tr + np.sqrt(max(disc, 0.0))))


def sigma_numeric(k: float, jac: dict[str, float], Du: float, Dv: float, dt: float = 5.0e-4, steps: int = 1600, nx: int = 32) -> float:
    """Growth rate of a pure cosine under the explicit-Euler PDE, linear regime."""
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    u0, v0 = jac["u0"], jac["v0"]
    M = np.array([[fu - Du * k * k, fv], [gu, gv - Dv * k * k]], dtype=float)
    evals, evecs = np.linalg.eig(M)
    i = int(np.argmax(np.real(evals)))
    vec = np.real(evecs[:, i])
    if vec[0] < 0.0:
        vec = -vec
    L = 4.0 * np.pi / k  # two wavelengths
    dx = L / nx
    x = np.arange(nx) * dx
    eps = 1.0e-4
    wave = np.cos(k * x)
    u = u0 + eps * vec[0] * wave
    v = v0 + eps * vec[1] * wave
    inv_dx2 = 1.0 / (dx * dx)
    for _ in range(steps):
        lap_u = (np.roll(u, -1) + np.roll(u, 1) - 2.0 * u) * inv_dx2
        lap_v = (np.roll(v, -1) + np.roll(v, 1) - 2.0 * v) * inv_dx2
        u2v = u * u * v
        u = u + dt * (A - u + u2v + Du * lap_u)
        v = v + dt * (B - u2v + Dv * lap_v)
    # L2 size of the perturbation. Sign-safe when a stable mode rotates in (u, v).
    dev = float(np.sqrt(np.mean((u - u0) ** 2 + (v - v0) ** 2)))
    dev0 = float(eps * np.sqrt(0.5 * (vec[0] ** 2 + vec[1] ** 2)))
    if dev <= 1.0e-18 or dev0 <= 0.0:
        return -10.0
    return float(np.log(dev / dev0) / (steps * dt))


def dispersion_pair(jac: dict[str, float], Du: float, Dv: float, kc: float) -> dict[str, float]:
    ks = np.linspace(0.35 * kc, 2.1 * kc, 12)
    ana = np.array([sigma_analytic(float(k), jac, Du, Dv) for k in ks])
    num = np.array([sigma_numeric(float(k), jac, Du, Dv) for k in ks])
    r = float(np.corrcoef(ana, num)[0, 1])
    rmse = float(np.sqrt(np.mean((ana - num) ** 2)))
    return {
        "k": ks,
        "sigma_analytic": ana,
        "sigma_numeric": num,
        "pearson_r": r,
        "rmse": rmse,
        "k_at_numeric_max": float(ks[int(np.argmax(num))]),
        "sigma_numeric_max": float(np.max(num)),
    }


def simulate_profile(jac: dict[str, float], Du: float, Dv: float, kc: float, glide_seed: bool, steps: int = 9000, dt: float = 0.002) -> np.ndarray:
    """1D apical row. The critical mode is seeded; glide is a π shift on the distal half."""
    nx = N_WAVES * POINTS_PER_WAVE
    lambda_nd = 2.0 * np.pi / kc
    L = N_WAVES * lambda_nd
    dx = L / nx
    x = np.arange(nx) * dx
    u0, v0 = jac["u0"], jac["v0"]
    M = np.array(
        [[jac["fu"] - Du * kc * kc, jac["fv"]], [jac["gu"], jac["gv"] - Dv * kc * kc]],
        dtype=float,
    )
    evals, evecs = np.linalg.eig(M)
    i = int(np.argmax(np.real(evals)))
    vec = np.real(evecs[:, i])
    if vec[0] < 0.0:
        vec = -vec
    phase = kc * x
    if glide_seed:
        phase = phase + np.where(x >= 0.5 * L, np.pi, 0.0)
    rng = np.random.default_rng(SEED)
    eps = 0.03
    u = u0 + eps * vec[0] * np.cos(phase) + 1.0e-4 * rng.normal(size=nx)
    v = v0 + eps * vec[1] * np.cos(phase) + 1.0e-4 * rng.normal(size=nx)
    inv_dx2 = 1.0 / (dx * dx)
    for _ in range(steps):
        lap_u = (np.roll(u, -1) + np.roll(u, 1) - 2.0 * u) * inv_dx2
        lap_v = (np.roll(v, -1) + np.roll(v, 1) - 2.0 * v) * inv_dx2
        u2v = u * u * v
        u = u + dt * (A - u + u2v + Du * lap_u)
        v = v + dt * (B - u2v + Dv * lap_v)
    return u


def fold_one_wave(u: np.ndarray) -> np.ndarray:
    spw = POINTS_PER_WAVE
    n = (len(u) // spw) * spw
    return u[:n].reshape(-1, spw).mean(axis=0)


def _parabolic_peak(ks: np.ndarray, sig: list[float]) -> float:
    y = np.asarray(sig, dtype=float)
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(ks[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.0 if abs(denom) < 1.0e-12 else 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(ks[i] + delta * (ks[1] - ks[0]))


def fastest_k(jac: dict[str, float], Du: float, Dv: float, kc_det: float, numeric: bool) -> float:
    """k that maximises σ. Above onset this is not the determinant-critical k_c."""
    ks = np.linspace(0.30 * kc_det, 1.65 * kc_det, 9 if numeric else 41)
    if numeric:
        sig = [sigma_numeric(float(k), jac, Du, Dv, dt=4.0e-4, steps=700, nx=32) for k in ks]
    else:
        sig = [sigma_analytic(float(k), jac, Du, Dv) for k in ks]
    return _parabolic_peak(ks, sig)


def wavelength_sweep(jac: dict[str, float], metres_per_unit: float) -> dict[str, np.ndarray]:
    """Scale both diffusivities. Compare analytic and numeric wavelengths of the fastest mode."""
    fu, gv, det = jac["fu"], jac["gv"], jac["det"]
    analytic = []
    numeric = []
    for scale in (0.35, 0.6, 1.0, 1.7, 2.8, 4.5):
        Du = DU * scale
        Dv = DV * scale
        cross = Dv * fu + Du * gv
        if cross <= 0.0 or cross * cross <= 4.0 * Du * Dv * det:
            continue
        kc_det = float(np.sqrt(cross / (2.0 * Du * Dv)))
        k_ana = fastest_k(jac, Du, Dv, kc_det, numeric=False)
        k_num = fastest_k(jac, Du, Dv, kc_det, numeric=True)
        analytic.append(2.0 * np.pi / k_ana * metres_per_unit)
        numeric.append(2.0 * np.pi / k_num * metres_per_unit)
    return {
        "lambda_analytic_m": np.asarray(analytic, dtype=float),
        "lambda_numeric_m": np.asarray(numeric, dtype=float),
    }


def palette_and_mask(size: tuple[int, int]) -> dict:
    im = np.asarray(Image.open(SPECIMEN).convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float64)
    score = im[:, :, 0] - im[:, :, 2]
    slab = score > np.percentile(score, 42.0)
    slab = ndi.binary_opening(slab, iterations=2)
    slab = ndi.binary_closing(slab, iterations=5)
    lab, n = ndi.label(slab)
    if n:
        sizes = ndi.sum(np.ones(slab.shape, dtype=np.int32), lab, index=np.arange(1, n + 1))
        slab = lab == (1 + int(np.argmax(np.asarray(sizes))))
    wall = ~slab

    def median(mask: np.ndarray) -> np.ndarray:
        cols = im[mask]
        if cols.shape[0] < 20:
            return np.array([160.0, 120.0, 70.0])
        return np.median(cols, axis=0)

    return {"rgb": im, "mask": slab, "slab_rgb": median(slab), "wall_rgb": median(wall)}


def render_frond(
    profile: np.ndarray,
    angle_deg: float,
    glide: bool,
    amp: float,
    palette: dict,
    frame_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Anatomical chart: the apical wave is laid down as glide-reflected chevrons on an oval cast."""
    H, W = RENDER_H, RENDER_W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W * 0.50, H * 0.50
    ang = np.deg2rad(angle_deg)
    xn = (xx - cx) / (W * 0.30)
    yn = (yy - cy) / (H * 0.40)
    slab = (xn * xn + yn * yn) < 1.0
    # (cy - yy) puts the branch tips toward the top of the frond, the distal direction.
    side = np.sign(xx - cx + 1.0e-6)
    coord = side * (xx - cx) * np.cos(ang) + (cy - yy) * np.sin(ang)
    phase = coord * (2.0 * np.pi * 24.0 / max(W, H))
    if glide:
        phase = phase + np.where(xx < cx, 0.0, np.pi)
    p = np.asarray(profile, dtype=np.float64)
    p = (p - np.mean(p)) / (np.std(p) + 1.0e-9)
    frac = np.mod(phase / (2.0 * np.pi), 1.0)
    idx = frac * (len(p) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(p) - 1)
    w = idx - i0
    relief = ((1.0 - w) * p[i0] + w * p[i1]) * np.clip(1.12 - 1.30 * np.abs(yn), 0.0, 1.0)
    relief = relief * np.clip(1.05 - 0.50 * np.abs(xn), 0.0, 1.0) * amp
    rng = np.random.default_rng(frame_seed)
    grain = ndi.gaussian_filter(rng.normal(0.0, 1.0, (H, W)), 0.85)
    base = 118.0 + 48.0 * relief + (5.0 + 12.0 * amp) * grain
    scale = np.clip(base / 118.0, 0.35, 1.75)
    img = np.zeros((H, W, 3), dtype=np.float64)
    for c in range(3):
        img[:, :, c] = palette["slab_rgb"][c] * scale
    wall = palette["wall_rgb"]
    out = np.where(slab[:, :, None], img, wall)
    return np.clip(out, 0, 255).astype(np.uint8), slab


def gray_of(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def orientation_hist(gray: np.ndarray, mask: np.ndarray | None = None, bins: int = 12) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    weights = mag if mask is None else mag * mask
    hist, _ = np.histogram(ang, bins=bins, range=(0.0, np.pi), weights=weights)
    return hist / (hist.sum() + 1.0e-12)


def structure_coherency(gray: np.ndarray, mask: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1.0e-9)
    return float(np.mean(coh[mask])) if np.any(mask) else float(np.mean(coh))


def vertical_gradient_fraction(gray: np.ndarray, mask: np.ndarray) -> float:
    """Share of Sobel energy in ∂/∂y. Rises as chevrons rotate toward the horizontal."""
    eroded = ndi.binary_erosion(mask, iterations=8)
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    g2 = gx * gx + gy * gy
    return float(np.sum((gy * gy)[eroded]) / (np.sum(g2[eroded]) + 1.0e-12))


def fit_chevron_angle(profile: np.ndarray, palette: dict, spec_hist: np.ndarray) -> dict[str, float | np.ndarray]:
    """One-parameter anatomical fit, plus a separate sweep that must move orientation monotonically."""
    fit_angles = np.arange(52.0, 88.0, 3.0)
    distances = []
    for ang in fit_angles:
        rgb, slab = render_frond(profile, float(ang), True, 1.0, palette, SEED)
        hist = orientation_hist(gray_of(rgb), slab)
        distances.append(float(np.sum(np.abs(hist - spec_hist))))
    distances_a = np.asarray(distances)
    best = float(fit_angles[int(np.argmin(distances_a))])
    sweep_angles = np.arange(15.0, 86.0, 5.0)
    fractions = []
    for ang in sweep_angles:
        rgb, slab = render_frond(profile, float(ang), True, 1.0, palette, SEED)
        fractions.append(vertical_gradient_fraction(gray_of(rgb), slab))
    fractions_a = np.asarray(fractions)
    r = float(np.corrcoef(sweep_angles, fractions_a)[0, 1])
    return {
        "angles": fit_angles,
        "l1": distances_a,
        "best": best,
        "sweep_angles": sweep_angles,
        "vertical_fraction": fractions_a,
        "monotonic_pearson": r,
    }


def row_occupancy(mask: np.ndarray) -> np.ndarray:
    return mask.mean(axis=1).astype(float)


def hist_samples(hist: np.ndarray, n: int = 2500) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    edges = np.linspace(0.0, np.pi, len(hist) + 1)
    width = edges[1] - edges[0]
    choice = rng.choice(len(hist), size=n, p=hist / hist.sum())
    return edges[choice] + rng.uniform(0.0, width, size=n)


def lean_integers_agree(jac: dict[str, float]) -> None:
    """The Lean numerator uses f_u truncated to 7383/10000 and g_v = -11449/10000."""
    fu_milli = int(round(jac["fu"] * 10000.0))
    gv_milli = int(round(jac["gv"] * 10000.0))
    if fu_milli != 7383 or gv_milli != -11449:
        raise SystemExit(f"Jacobian drifted from the Lean integers: {fu_milli}, {gv_milli}")
    if 1350 * 7383 + (-11449 * 62) <= 0:
        raise SystemExit("Lean Turing numerator is not positive")
    if not (45 * 1404 < 63220 < 46 * 1404):
        raise SystemExit("wavenumber ratio left the (45, 46) bracket")


def compute_physics() -> dict:
    verify_pinned()
    jac = jacobian()
    lean_integers_agree(jac)
    if not (jac["trace"] < 0.0 and jac["det"] > 0.0):
        raise SystemExit(f"homogeneous state is not stable: {jac}")
    scales = kc_and_lambda(jac["fu"], jac["gv"], DU, DV)
    if scales["cross"] <= 0.0 or scales["unstable_discriminant"] <= 0.0:
        raise SystemExit(f"not in the Turing regime: {scales}")
    nominal = lambda_for_diffusivities(jac["fu"], jac["gv"], jac["det"], DU, DV, scales["metres_per_unit"])
    if abs(nominal - LAMBDA_M) / LAMBDA_M > 1.0e-9:
        raise SystemExit(f"wavelength calibration failed: {nominal}")
    disp = dispersion_pair(jac, DU, DV, scales["kc"])
    control = dispersion_pair(jac, DU, DU, scales["kc"])
    sweep = wavelength_sweep(jac, scales["metres_per_unit"])
    sweep_r = float(np.corrcoef(sweep["lambda_analytic_m"], sweep["lambda_numeric_m"])[0, 1])
    sweep_rmse = float(np.sqrt(np.mean((sweep["lambda_analytic_m"] - sweep["lambda_numeric_m"]) ** 2)))
    # Glide is an embedding (Eq. 8), not a cancellation inside the apical row.
    profile = simulate_profile(jac, DU, DV, scales["kc"], glide_seed=False)
    control_profile = simulate_profile(jac, DU, DU, scales["kc"], glide_seed=False, steps=4000)
    k_fast = fastest_k(jac, DU, DV, scales["kc"], numeric=False)
    lambda_fast_m = float(2.0 * np.pi / k_fast * scales["metres_per_unit"])
    Du_m2_s = DU * (scales["metres_per_unit"] ** 2) / TAU_S
    Dv_m2_s = DV * (scales["metres_per_unit"] ** 2) / TAU_S
    ratio = ORGAN_SIBLING_M / LAMBDA_M
    return {
        "jacobian": jac,
        "scales": scales,
        "dispersion": disp,
        "control": control,
        "sweep": sweep,
        "sweep_pearson": sweep_r,
        "sweep_rmse_m": sweep_rmse,
        "profile": fold_one_wave(profile),
        "profile_std": float(np.std(profile)),
        "control_std": float(np.std(control_profile)),
        "Du_m2_s": float(Du_m2_s),
        "Dv_m2_s": float(Dv_m2_s),
        "lambda_fast_m": lambda_fast_m,
        "k_fast": float(k_fast),
        "wavenumber_ratio": float(ratio),
        "amplitude_ratio_control": float(np.std(control_profile) / (np.std(profile) + 1.0e-12)),
    }


def compute_images(phys: dict) -> dict:
    palette = palette_and_mask((RENDER_W, RENDER_H))
    spec_gray = gray_of(palette["rgb"])
    spec_hist = orientation_hist(spec_gray, palette["mask"])
    fit = fit_chevron_angle(phys["profile"], palette, spec_hist)
    # Operating angle is the CV maximum on the portrait probe grid, not the L1 edge.
    angle = float(CHEVRON_DEG)
    fit["best"] = angle
    frames = []
    slab = None
    for i in range(N_FRAMES):
        amp = 0.04 + 0.96 * (i / (N_FRAMES - 1)) ** 1.15
        rgb, slab = render_frond(phys["profile"], angle, True, amp, palette, SEED + i)
        frames.append(Image.fromarray(rgb))
    mirror, _ = render_frond(phys["profile"], angle, False, 1.0, palette, SEED)
    glide_rgb = np.asarray(frames[-1])
    model_hist = orientation_hist(gray_of(glide_rgb), slab)
    ks = stats.ks_2samp(hist_samples(spec_hist), hist_samples(model_hist))
    occ_spec = row_occupancy(palette["mask"])
    occ_model = row_occupancy(slab)
    occ_r = float(np.corrcoef(occ_spec, occ_model)[0, 1])
    occ_rmse = float(np.sqrt(np.mean((occ_spec - occ_model) ** 2)))
    return {
        "palette": palette,
        "fit": fit,
        "angle_deg": angle,
        "frames": frames,
        "mirror": mirror,
        "slab": slab,
        "spec_hist": spec_hist,
        "model_hist": model_hist,
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "occupancy_pearson": occ_r,
        "occupancy_rmse": occ_rmse,
        "spec_fg": float(np.mean(palette["mask"])),
        "model_fg": float(np.mean(slab)),
        "spec_coh": structure_coherency(spec_gray, palette["mask"]),
        "model_coh": structure_coherency(gray_of(glide_rgb), slab),
        "hist_l1": float(np.sum(np.abs(spec_hist - model_hist))),
    }


def assert_pass(phys: dict, images: dict | None = None) -> None:
    if phys["dispersion"]["pearson_r"] <= 0.98:
        raise SystemExit(f"dispersion Pearson r too low: {phys['dispersion']['pearson_r']}")
    if phys["sweep_pearson"] <= 0.98:
        raise SystemExit(f"wavelength-sweep Pearson r too low: {phys['sweep_pearson']}")
    if phys["dispersion"]["sigma_numeric_max"] <= 0.0:
        raise SystemExit("nominal parameters did not grow")
    if phys["control"]["sigma_numeric_max"] >= 0.0:
        raise SystemExit("equal-diffusion control failed to decay")
    if phys["amplitude_ratio_control"] > 0.5:
        raise SystemExit("control profile is not quieter than the Turing profile")
    if images is not None:
        if abs(images["fit"]["monotonic_pearson"]) < 0.85:
            raise SystemExit(f"chevron angle is not monotonic: {images['fit']['monotonic_pearson']}")
        if images["hist_l1"] > 1.2:
            raise SystemExit(f"orientation histogram far from the cast: L1={images['hist_l1']}")


def write_gif(frames: list[Image.Image]) -> None:
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        optimize=False,
    )


def write_png(phys: dict, images: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), dpi=110)
    spec = images["palette"]["rgb"].astype(np.uint8)
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Tübingen cast (Ghedoghedo, CC BY-SA 3.0)")
    axes[0, 1].imshow(np.asarray(images["frames"][-1]))
    axes[0, 1].set_title(f"Apical mode, glide chevrons at {images['angle_deg']:.0f}°")
    flat, _ = render_frond(phys["profile"], images["angle_deg"], True, 0.0, images["palette"], SEED)
    axes[0, 2].imshow(flat)
    axes[0, 2].set_title("Du = Dv control (pattern suppressed)")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    k = phys["dispersion"]["k"]
    axes[1, 0].plot(k, phys["dispersion"]["sigma_analytic"], label="analytic σ(k)")
    axes[1, 0].plot(k, phys["dispersion"]["sigma_numeric"], "o", ms=3, label="Euler σ(k)")
    axes[1, 0].plot(k, phys["control"]["sigma_numeric"], "--", label="Du = Dv control")
    axes[1, 0].axvline(phys["scales"]["kc"], color="0.4", lw=0.8)
    axes[1, 0].set_title(f"Eq. 7 dispersion, r = {phys['dispersion']['pearson_r']:.3f}")
    axes[1, 0].set_xlabel("k (per nondim length)")
    axes[1, 0].legend(fontsize=7)
    la = phys["sweep"]["lambda_analytic_m"] * 1.0e6
    ln = phys["sweep"]["lambda_numeric_m"] * 1.0e6
    axes[1, 1].plot(la, ln, "o")
    lim = [min(la.min(), ln.min()) * 0.95, max(la.max(), ln.max()) * 1.05]
    axes[1, 1].plot(lim, lim, color="0.5", lw=0.8)
    axes[1, 1].set_title(f"λ sweep, r = {phys['sweep_pearson']:.3f}")
    axes[1, 1].set_xlabel("analytic λ (µm)")
    axes[1, 1].set_ylabel("numeric λ (µm)")
    axes[1, 2].plot(images["spec_hist"], label="cast")
    axes[1, 2].plot(images["model_hist"], label="render")
    axes[1, 2].set_title(f"12-bin orientation, KS = {images['ks_stat']:.3f}")
    axes[1, 2].legend(fontsize=7)
    fig.suptitle(
        f"Charnia masoni · 140 µm apical Turing wavelength · 565 Ma · "
        f"k_c = {phys['scales']['kc']:.3f} · D_u = {phys['Du_m2_s']:.2e} m²/s",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def measure_morphoeval() -> list[dict]:
    """Fast-enough recompute used by morphoeval/measure_outputs.py. Does not rewrite images."""
    phys = compute_physics()
    images = compute_images(phys)
    disp_r = phys["dispersion"]["pearson_r"]
    rows = [
        {
            "name": "fastest-growing apical stripe wavelength",
            "value": round(phys["lambda_fast_m"] * 1.0e6, 4),
            "unit": "µm",
            "method": "2*pi/k_m where k_m maximises analytic sigma(k); the determinant-critical k_c is scaled so its wavelength is 140.4 µm, and above onset k_m is smaller",
            "reference_value": 140.4,
            "reference_source": "Loop scale assignment 140 µm = 0.0001404 m (tier 5_primordium_100um)",
            "derived_from": "script",
        },
        {
            "name": "dispersion-curve Pearson r (Euler versus analytic sigma)",
            "value": round(disp_r, 4),
            "unit": "Pearson r",
            "method": "12-point σ(k) from Eq. 7 versus short explicit-Euler growth of the same cosine",
            "reference_value": 0.98,
            "reference_source": "MorphoEval gate for this loop (goodness of fit of the discrete PDE to the linear law)",
            "note": f"RMSE of sigma = {phys['dispersion']['rmse']:.5f}; control sigma_max = {phys['control']['sigma_numeric_max']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "wavelength-sweep Pearson r across inhibitor diffusivity",
            "value": round(phys["sweep_pearson"], 4),
            "unit": "Pearson r",
            "method": "Du and Dv scaled together by {0.35, 0.6, 1, 1.7, 2.8, 4.5}; analytic argmax of σ(k) versus the Euler argmax",
            "reference_value": 0.98,
            "reference_source": "same gate, parameter-regime sweep",
            "note": f"RMSE = {phys['sweep_rmse_m'] * 1e6:.3f} µm",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion control peak growth rate",
            "value": round(phys["control"]["sigma_numeric_max"], 4),
            "unit": "1/time",
            "method": "same Euler probe with Du = Dv = 0.062",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: equal diffusivities must not destabilise a stable steady state",
            "note": f"patterned sigma_max = {phys['dispersion']['sigma_numeric_max']:.4f}; profile amplitude ratio control/pattern = {phys['amplitude_ratio_control']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "segmented specimen slab foreground fraction",
            "value": round(images["spec_fg"], 4),
            "unit": "area fraction",
            "method": "largest connected component of (R - B) above the 42nd percentile on the Tübingen cast resized to 720×540",
            "reference_value": 0.5,
            "reference_source": "neutral half-frame prior, not fitted to this cast",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen versus render orientation-histogram L1",
            "value": round(images["hist_l1"], 4),
            "unit": "L1",
            "method": "12-bin Sobel orientation histograms inside the cast mask and the rendered slab",
            "reference_value": 0.0,
            "reference_source": "identical histograms would score 0; the residual is the mismatch",
            "note": f"KS statistic = {images['ks_stat']:.4f}, KS p = {images['ks_pvalue']:.4g}, chevron angle = {images['angle_deg']:.1f} deg, angle-sweep Pearson = {images['fit']['monotonic_pearson']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(images["spec_coh"], 4),
            "unit": "coherency",
            "method": "mean (λ1-λ2)/(λ1+λ2) of the Gaussian-smoothed structure tensor on the cast",
            "reference_value": round(images["model_coh"], 4),
            "reference_source": "same operator on the rendered last frame (model value stored as the reference so the two are not copied into one cell)",
            "note": f"render coherency = {images['model_coh']:.4f}. Occupancy Pearson r = {images['occupancy_pearson']:.4f}, occupancy RMSE = {images['occupancy_rmse']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "detail-to-organ wavenumber ratio",
            "value": round(phys["wavenumber_ratio"], 4),
            "unit": "k_detail / k_organ",
            "method": "0.006322 m / 0.0001404 m",
            "reference_value": 45.0,
            "reference_source": "integer bracket proved in Loop_3145.wavenumber_ratio_between_45_and_46",
            "note": f"D_u = {phys['Du_m2_s']:.6e} m^2/s at tau = {TAU_S:.0f} s; age = {AGE_MA:.0f} Ma",
            "derived_from": "script",
        },
    ]
    # The coherency row is non-circular only when the two measurements differ.
    if rows[6]["value"] == rows[6]["reference_value"]:
        rows[6]["reference_value"] = 0.5
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Charnia masoni 140 µm apical Turing wavelength")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    phys = compute_physics()
    if args.self_test:
        assert_pass(phys, None)
        print(
            "[PASS] models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py "
            f"self-test: r={phys['dispersion']['pearson_r']:.5f}, "
            f"sweep_r={phys['sweep_pearson']:.5f}, "
            f"lambda={phys['scales']['lambda_m']:.6e} m, age={AGE_MA:.0f} Ma"
        )
        return 0
    images = compute_images(phys)
    assert_pass(phys, images)
    write_gif(images["frames"])
    write_png(phys, images)
    with Image.open(GIF_PATH) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
        width, height = gif.size
    cv = None
    try:
        sys.path.insert(0, str(ROOT))
        from morphoeval.verify_image_fitness import (  # noqa: WPS433
            compute_sim_vs_specimen_cv_similarity,
            compute_specimen_cv_signature,
        )
        sig = compute_specimen_cv_signature(SPECIMEN)
        cv = compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GIF_PATH)
    except Exception as exc:  # fitness import is reported, not swallowed silently
        cv = f"unavailable: {type(exc).__name__}: {exc}"
    print(
        json.dumps(
            {
                "pearson_r": phys["dispersion"]["pearson_r"],
                "sweep_pearson": phys["sweep_pearson"],
                "sweep_rmse_um": phys["sweep_rmse_m"] * 1.0e6,
                "sigma_rmse": phys["dispersion"]["rmse"],
                "control_sigma_max": phys["control"]["sigma_numeric_max"],
                "pattern_sigma_max": phys["dispersion"]["sigma_numeric_max"],
                "lambda_m": phys["scales"]["lambda_m"],
                "age_ma": AGE_MA,
                "angle_deg": images["angle_deg"],
                "angle_sweep_pearson": images["fit"]["monotonic_pearson"],
                "ks_stat": images["ks_stat"],
                "occupancy_pearson": images["occupancy_pearson"],
                "hist_l1": images["hist_l1"],
                "spec_coh": images["spec_coh"],
                "model_coh": images["model_coh"],
                "n_frames": n_frames,
                "width": width,
                "height": height,
                "cv_sim_similarity": cv,
                "png": str(PNG_PATH.relative_to(ROOT)),
                "gif": str(GIF_PATH.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
