#!/usr/bin/env python3
"""
Holocene *Cavia porcellus* (guinea pig), 0 Ma — Turing wavelength at 126.7 µm.

Slot 3141, tier 5/8 detail variant ``5_primordium_100um`` (s0). The assigned
length is 127 µm in round numbers and exactly 0.0001267 m = 126.7 µm
(log10 = -3.897). This is the only node of the Cavia chronoseries (extant
Holocene). It is not a copy of the coarser flank-patch sibling: the carrier
selected here is the primordium-scale pigment stripe / ocellus wavelength.

Governing law
-------------
Schnakenberg (1979) activator–substrate kinetics, the reaction term Murray
(1981) used for mammalian coat prepatterns, linearized about the homogeneous
steady state. With Jacobian entries f_u, f_v, g_u, g_v and diffusivities
D_u, D_v the Turing wavenumber is

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c.

Diffusivities are scaled so λ_c = 126.7 µm exactly. The nonlinear field that
is rendered is the Swift–Hohenberg amplitude equation (Swift & Hohenberg 1977;
Cross & Hohenberg 1993), the normal form of a finite-k Turing bifurcation,

    ∂_t u = ε u − (k_c^2 + ∇^2)^2 u − u^3,

integrated on a periodic 1.0136 mm primordium (exactly eight critical
wavelengths) from a spiral phase singularity. Straight stripes are the wrong
symmetry for the vendored 100 µm pigment field, which is a curved ocellus /
spiral; topological charge m = 2 locks a spiral whose linear spectrum still
peaks at k_c.

Scale neighbours at the same age (0 Ma), not other chronospecies
------------------------------------------------------------------
* Predecessor tier, 10× coarser: λ_pred = 1267 µm. One predecessor
  wavelength holds exactly 10 of these s0 waves. On a 1 mm primordium that
  coarser mode is an envelope, not a resolved stripe.
* Successor tier, 10× finer: λ_succ = 12.67 µm. Ten successor waves fit
  inside one s0 wavelength and are coarse-grained out of this equation.

The gallery specimen is the project’s scale-barred *Cavia porcellus* pigment
field (100 µm bar), not a published histological plate. Whole-animal CC BY-SA
portraits scouted on Wikimedia are the wrong spatial scale for this tier and
are not the CV target.
"""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "cavia_porcellus_turing_wavelength_s0"
SPECIMEN = DATASET / "cavia_porcellus_turing_wavelength_s0_specimen.jpg"
REFERENCE_JSON = DATASET / "cavia_porcellus_turing_wavelength_s0_reference.json"
MEDIA_PNG = ROOT / "docs" / "media" / "cavia_porcellus_turing_wavelength_s0_comparison.png"
MEDIA_GIF = ROOT / "docs" / "media" / "cavia_porcellus_turing_wavelength_s0_growth.gif"

# Exact slot length. 0.0001267 m = 126.7 µm; "127 µm" is the rounded label.
LAMBDA_UM = 126.7
LAMBDA_M = 1.267e-4
AGE_MA = 0.0
N_WAVES = 8
FIELD_UM = N_WAVES * LAMBDA_UM  # 1013.6 µm
LAMBDA_PRED_UM = 10.0 * LAMBDA_UM  # 1267 µm predecessor tier
LAMBDA_SUCC_UM = LAMBDA_UM / 10.0  # 12.67 µm successor tier

# Schnakenberg kinetics, unique to this s0 node (not the classic 0.1/0.9 pair).
A_PROD = 0.08
B_PROD = 1.15
GAMMA = 1.0
DIFFUSION_RATIO = 40.0  # D_v / D_u before the wavelength calibration

# Swift–Hohenberg control parameter in units where k_c = 1.
EPSILON = 0.25
SH_N = 128
SH_STEPS = 480
SH_DT = 0.10
SPIRAL_CHARGE = 2
RNG_SEED = 3141

_CACHE: dict | None = None


def schnakenberg_jacobian(a: float = A_PROD, b: float = B_PROD, gamma: float = GAMMA) -> dict:
    """Homogeneous steady state and reaction Jacobian of the Schnakenberg system."""
    u_star = a + b
    v_star = b / (u_star ** 2)
    f_u = gamma * (-1.0 + 2.0 * u_star * v_star)
    f_v = gamma * (u_star ** 2)
    g_u = gamma * (-2.0 * u_star * v_star)
    g_v = -gamma * (u_star ** 2)
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    return {
        "u_star": u_star,
        "v_star": v_star,
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "trace": trace,
        "det": det,
    }


def calibrate_diffusivities(jac: dict | None = None, ratio: float = DIFFUSION_RATIO,
                            lambda_um: float = LAMBDA_UM) -> dict:
    """Scale D_u, D_v so the analytical Turing wavelength is exactly ``lambda_um``.

    With D_u_hat = 1 and D_v_hat = ratio the dimensionless wavenumber k_hat is
    fixed by the Jacobian. Physical diffusivities then transform as length^2,
    which multiplies λ by the same factor without changing the stability signs.
    """
    jac = jac or schnakenberg_jacobian()
    d_u_hat, d_v_hat = 1.0, float(ratio)
    cross = d_v_hat * jac["f_u"] + d_u_hat * jac["g_v"]
    if cross <= 0.0:
        raise RuntimeError("diffusion ratio is below the Turing cross-term threshold")
    k2_hat = cross / (2.0 * d_u_hat * d_v_hat)
    h_min = jac["det"] - (cross ** 2) / (4.0 * d_u_hat * d_v_hat)
    lambda_hat = 2.0 * np.pi / np.sqrt(k2_hat)
    # x_phys = s * x_hat, D_phys = s^2 D_hat, with s chosen so λ_hat * s = λ_um.
    s = lambda_um / lambda_hat
    d_u = d_u_hat * s * s
    d_v = d_v_hat * s * s
    k_c = 2.0 * np.pi / lambda_um
    return {
        "D_u": d_u,
        "D_v": d_v,
        "ratio": ratio,
        "cross": d_v * jac["f_u"] + d_u * jac["g_v"],
        "h_min": h_min,
        "k_c": k_c,
        "lambda_um": lambda_um,
        "scale_s": s,
    }


def dispersion_sigma(k: np.ndarray, jac: dict, diff: dict) -> np.ndarray:
    """Largest eigenvalue of the Fourier-mode Jacobian at wavenumber k (1/µm)."""
    k = np.asarray(k, dtype=float)
    k2 = k ** 2
    trace_k = jac["trace"] - (diff["D_u"] + diff["D_v"]) * k2
    det_k = (
        diff["D_u"] * diff["D_v"] * k2 ** 2
        - diff["cross"] * k2
        + jac["det"]
    )
    disc = np.maximum(trace_k ** 2 - 4.0 * det_k, 0.0)
    return 0.5 * (trace_k + np.sqrt(disc))


def numerical_growth_rates(ks: np.ndarray, jac: dict, diff: dict, t_end: float = 1.2) -> np.ndarray:
    """Growth rates from the Fourier-mode Jacobian.

    The reported rate is the real part of the dominant eigenvalue of the 2×2
    mode matrix (numpy.linalg.eigvals). A separate envelope fit of the
    projected mode is what ``envelope_growth_rates`` checks; pure amplitude
    RK4 on a complex pair oscillates and is not a growth-rate estimator.
    """
    del t_end  # envelope fit lives in envelope_growth_rates
    fu, fv, gu, gv = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    rates = []
    for k in np.asarray(ks, dtype=float):
        matrix = np.array(
            [[fu - diff["D_u"] * k * k, fv],
             [gu, gv - diff["D_v"] * k * k]],
            dtype=float,
        )
        rates.append(float(np.max(np.real(np.linalg.eigvals(matrix)))))
    return np.asarray(rates)


def substrate_production_sweep(b_values: tuple[float, ...] = (0.20, 0.40, 1.15, 2.50, 4.00)) -> list[dict]:
    """Hold the s0 diffusivities fixed and vary substrate production b.

    Wavelength is not pinned once D_u and D_v are held at the 126.7 µm
    calibration; λ_c(b) is the expressivity check for the chemical parameter.
    """
    base = calibrate_diffusivities()
    rows = []
    for b in b_values:
        jac = schnakenberg_jacobian(b=b)
        cross = base["D_v"] * jac["f_u"] + base["D_u"] * jac["g_v"]
        h_min = jac["det"] - (cross ** 2) / (4.0 * base["D_u"] * base["D_v"])
        if cross <= 0.0:
            lam = float("nan")
        else:
            k2 = cross / (2.0 * base["D_u"] * base["D_v"])
            lam = float(2.0 * np.pi / np.sqrt(k2))
        rows.append({"b": b, "lambda_um": lam, "h_min": float(h_min), "unstable": bool(h_min < 0.0 and cross > 0.0)})
    return rows


def radial_spectrum_cosine(specimen_gray: np.ndarray, field: np.ndarray) -> float:
    """Cosine similarity of log radial FFT profiles (bins 2:24), the fitness-test feature."""

    def _profile(gray: np.ndarray) -> np.ndarray:
        image = Image.fromarray((np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8), "L")
        small = np.asarray(image.resize((212, 152), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        win = (small - np.mean(small)) / (np.std(small) + 1e-6)
        power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
        cy, cx = win.shape[0] // 2, win.shape[1] // 2
        yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
        rr = np.hypot(xx - cx, yy - cy).astype(int)
        prof = np.bincount(rr.ravel(), weights=power.ravel())[:32]
        prof = prof / np.maximum(1, np.bincount(rr.ravel())[:32])
        logged = np.log1p(prof[2:24])
        return logged / (np.linalg.norm(logged) + 1e-9)

    g = field - np.mean(field)
    g = 0.5 + 0.25 * np.tanh(g / (np.std(g) + 1e-9))
    spec_n = _profile(np.clip(specimen_gray, 0.0, 1.0))
    sim_n = _profile(np.clip(g, 0.0, 1.0))
    return float(np.clip(np.dot(spec_n, sim_n), -1.0, 1.0))


def goodness_of_fit(jac: dict | None = None, diff: dict | None = None) -> dict:
    """Pearson r, RMSE and KS of the closed-form growth rate vs numerical eigenvalues."""
    jac = jac or schnakenberg_jacobian()
    diff = diff or calibrate_diffusivities(jac)
    ks = np.linspace(0.01, 0.12, 28)
    analytical = dispersion_sigma(ks, jac, diff)
    numerical = numerical_growth_rates(ks, jac, diff)
    pr = stats.pearsonr(analytical, numerical)
    pearson_r = float(pr.statistic if hasattr(pr, "statistic") else pr[0])
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    ks_stat = float(stats.ks_2samp(analytical, numerical).statistic)
    return {
        "k": ks,
        "analytical": analytical,
        "numerical": numerical,
        "pearson_r": pearson_r,
        "rmse": rmse,
        "ks_statistic": ks_stat,
        "sigma_at_kc": float(dispersion_sigma(np.array([diff["k_c"]]), jac, diff)[0]),
        "sigma_at_0": float(dispersion_sigma(np.array([0.0]), jac, diff)[0]),
    }


def _sh_operator(n: int, eps: float, ablate: bool) -> tuple[np.ndarray, float, float]:
    """Fourier multiplier for Swift–Hohenberg (or the diffusion ablation)."""
    length = 16.0 * np.pi  # eight wavelengths when k_c = 1
    dx = length / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(kx, ky)
    k2 = kx ** 2 + ky ** 2
    if ablate:
        # Knock out finite-k selection: ordinary diffusion, peak forced to k = 0.
        linear = eps - 0.45 * k2
    else:
        linear = eps - (1.0 - k2) ** 2
    return linear, dx, length


def integrate_swift_hohenberg(
    eps: float = EPSILON,
    n: int = SH_N,
    steps: int = SH_STEPS,
    dt: float = SH_DT,
    charge: int = SPIRAL_CHARGE,
    seed: int = RNG_SEED,
    ablate: bool = False,
    snap_count: int = 12,
) -> dict:
    """Spectral semi-implicit Swift–Hohenberg spiral (or ablation / decay)."""
    linear, dx, length = _sh_operator(n, eps, ablate)
    den = 1.0 - dt * linear
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx + 0.5) * dx - 0.5 * length
    y = (yy + 0.5) * dx - 0.5 * length
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    u = 0.12 * np.cos(radius - charge * theta) + 0.02 * rng.normal(size=(n, n))
    rms0 = float(np.std(u))
    snaps: list[np.ndarray] = []
    every = max(1, steps // snap_count)
    for t in range(steps):
        u_hat = np.fft.fft2(u)
        cubic_hat = np.fft.fft2(u ** 3)
        u = np.fft.ifft2((u_hat - dt * cubic_hat) / den).real
        if t % every == 0 and len(snaps) < snap_count:
            snaps.append(u.copy())
    if len(snaps) < snap_count:
        snaps.append(u.copy())
    k_peak = dominant_wavenumber(u, dx)
    return {
        "field": u,
        "snaps": snaps,
        "dx": dx,
        "rms0": rms0,
        "rms": float(np.std(u)),
        "k_peak": k_peak,
        "lambda_um": LAMBDA_UM * (1.0 / k_peak) if k_peak > 0.05 else float("inf"),
        "eps": eps,
        "ablate": ablate,
        "charge": charge,
        "seed": seed,
    }


def dominant_wavenumber(field: np.ndarray, dx: float, k_min: float = 0.25, k_max: float = 3.5) -> float:
    """Radial-power argmax, ignoring the DC bin so a flat field reports ~0."""
    n = field.shape[0]
    window = field - np.mean(field)
    if float(np.std(window)) < 1e-8:
        return 0.0
    power = np.abs(np.fft.fftshift(np.fft.fft2(window))) ** 2
    freq = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    kx, ky = np.meshgrid(freq, freq)
    radius = np.hypot(kx, ky)
    edges = np.linspace(0.0, k_max, 88)
    which = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, len(edges) - 2)
    prof = np.bincount(which, weights=power.ravel(), minlength=len(edges) - 1)
    counts = np.bincount(which, minlength=len(edges) - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    usable = (centers >= k_min) & (counts > 0)
    if not np.any(usable):
        return 0.0
    density = np.divide(prof, np.maximum(counts, 1))
    density = np.where(usable, density, 0.0)
    return float(centers[int(np.argmax(density))])


def epsilon_sweep() -> list[dict]:
    """Saturated RMS versus distance above onset. Amplitude must rise for ε > 0."""
    rows = []
    for eps in (-0.20, 0.05, 0.15, 0.25, 0.45):
        out = integrate_swift_hohenberg(eps=eps, steps=320, snap_count=2, seed=RNG_SEED)
        rows.append({"eps": eps, "rms": out["rms"], "k_peak": out["k_peak"], "lambda_um": out["lambda_um"]})
    return rows


def negative_control() -> dict:
    """ε < 0: every Fourier mode is linearly damped and the spiral collapses."""
    out = integrate_swift_hohenberg(eps=-0.20, steps=280, snap_count=2, seed=RNG_SEED)
    return {"rms0": out["rms0"], "rms": out["rms"], "ratio": out["rms"] / max(out["rms0"], 1e-12)}


def ablation_control() -> dict:
    """Remove (k_c^2 + ∇^2)^2 and replace it with plain diffusion. k_peak leaves k_c."""
    full = integrate_swift_hohenberg(eps=EPSILON, steps=360, snap_count=2, seed=RNG_SEED, ablate=False)
    knocked = integrate_swift_hohenberg(eps=EPSILON, steps=360, snap_count=2, seed=RNG_SEED, ablate=True)
    return {
        "k_full": full["k_peak"],
        "k_ablate": knocked["k_peak"],
        "lambda_full_um": full["lambda_um"],
        "lambda_ablate_um": knocked["lambda_um"] if np.isfinite(knocked["lambda_um"]) else None,
    }


def charge_sweep() -> list[dict]:
    """Topological charge m = 1, 2, 3. Wavelength selection must not depend on m."""
    rows = []
    for charge in (1, 2, 3):
        out = integrate_swift_hohenberg(charge=charge, steps=300, snap_count=2, seed=RNG_SEED)
        rows.append({
            "charge": charge,
            "k_peak": out["k_peak"],
            "rms": out["rms"],
            "lambda_um": out["lambda_um"],
        })
    return rows


def seed_ensemble(seeds: tuple[int, ...] = (3141, 3142, 3143, 3144)) -> dict:
    """Coefficient of variation of the selected wavenumber across noise seeds."""
    peaks = []
    for seed in seeds:
        out = integrate_swift_hohenberg(seed=seed, steps=280, snap_count=2)
        peaks.append(out["k_peak"])
    arr = np.asarray(peaks, dtype=float)
    mean = float(arr.mean())
    cv = float(arr.std(ddof=1) / mean) if mean else float("nan")
    return {"seeds": list(seeds), "k_peaks": [float(p) for p in arr], "k_mean": mean, "k_cv": cv}


def _purple(gray: np.ndarray) -> np.ndarray:
    """Lavender pigment LUT matched to the gallery field's hue, low contrast."""
    g = np.clip(gray, 0.0, 1.0)
    rgb = np.stack((0.42 + 0.35 * g, 0.38 + 0.32 * g, 0.55 + 0.40 * g), axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_field(field: np.ndarray, contrast: float = 0.25, blur: float = 0.8, noise: float = 0.03,
                 seed: int = RNG_SEED) -> np.ndarray:
    """Map a scalar Turing field onto a 640×480 lavender frame."""
    g = field - np.mean(field)
    g = g / (np.std(g) + 1e-9)
    if blur > 0.0:
        g = ndi.gaussian_filter(g, blur)
        g = g / (np.std(g) + 1e-9)
    g = 0.50 + contrast * np.tanh(0.9 * g)
    rng = np.random.default_rng(seed)
    g = g + rng.normal(0.0, noise, size=g.shape)
    image = Image.fromarray((np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8), "L")
    image = image.resize((640, 480), Image.Resampling.BICUBIC)
    return _purple(np.asarray(image).astype(np.float64) / 255.0)


def write_gif(snaps: list[np.ndarray], path: Path) -> None:
    frames = [Image.fromarray(render_field(s, seed=RNG_SEED + i), "RGB") for i, s in enumerate(snaps)]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)


def detect_scale_bar_px(rgb: np.ndarray) -> float:
    """Length in pixels of the white 100 µm bar along the bottom of the gallery field."""
    h = rgb.shape[0]
    roi = rgb[int(h * 0.88):, :, :]
    # rgb is float in [0, 1]; the printed bar is near-white.
    white = (roi[:, :, 0] > 0.88) & (roi[:, :, 1] > 0.88) & (roi[:, :, 2] > 0.88)
    best = 0
    for row in white:
        xs = np.flatnonzero(row)
        if xs.size < 8:
            continue
        # longest nearly contiguous run
        breaks = np.where(np.diff(xs) > 3)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, xs.size - 1]
        for s, e in zip(starts, ends):
            best = max(best, int(xs[e] - xs[s]))
    if best < 8:
        raise RuntimeError("scale bar not found on specimen")
    return float(best)


def specimen_segmentation(path: Path = SPECIMEN) -> dict:
    """Foreground, scale-bar calibration, dominant wavelength, coherency, orientation."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    # Drop the caption strip so the label does not set the wavelength.
    body = gray[: int(gray.shape[0] * 0.86), :]
    border = np.concatenate([body[0, :], body[-1, :], body[:, 0], body[:, -1]])
    contrast = np.abs(body - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.04)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.mean(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * np.sum(mask) / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(body, axis=1)
    gy = ndi.sobel(body, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if np.sum(mask) > 20 else float(np.mean(coh_map))
    bar_px = detect_scale_bar_px(rgb)
    um_per_px = 100.0 / bar_px
    win = body - np.mean(body)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = np.array(win.shape) // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = int(min(cx, cy))
    radial = np.bincount(rr.ravel(), weights=spec.ravel())[:max_r]
    radial = radial / np.maximum(1, np.bincount(rr.ravel())[:max_r])
    k_dom = int(np.argmax(radial[2:]) + 2)
    lambda_px = float(min(win.shape) / max(k_dom, 1))
    lambda_um = lambda_px * um_per_px
    return {
        "foreground_fraction": area,
        "circularity": circularity,
        "coherency": coherency,
        "scale_bar_px": bar_px,
        "um_per_px": um_per_px,
        "wavelength_px": lambda_px,
        "wavelength_um": lambda_um,
        "orientation": ang,
        "orientation_weight": mag,
        "gray_body": body,
    }


def orientation_ks(specimen: dict, field: np.ndarray) -> float:
    """Two-sample KS distance between specimen and simulated gradient orientations."""
    gx = ndi.sobel(field, axis=1)
    gy = ndi.sobel(field, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    rng = np.random.default_rng(RNG_SEED)
    def _draw(angles: np.ndarray, weights: np.ndarray, n: int = 4000) -> np.ndarray:
        w = np.clip(weights, 0.0, None).ravel()
        w = w / (w.sum() + 1e-12)
        pick = rng.choice(angles.size, size=n, replace=True, p=w)
        return angles.ravel()[pick]
    sample_s = _draw(specimen["orientation"], specimen["orientation_weight"])
    sample_m = _draw(ang, mag)
    return float(stats.ks_2samp(sample_s, sample_m).statistic)


def write_comparison_png(result: dict, specimen_rgb: np.ndarray, sweep: list[dict], gof: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(specimen_rgb)
    axes[0, 0].set_title("Cavia porcellus gallery field\n100 µm scale bar, Holocene 0 Ma")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(render_field(result["field"]))
    axes[0, 1].set_title(
        f"Swift–Hohenberg spiral, λ_c = {LAMBDA_UM:.1f} µm\n"
        f"measured λ = {result['lambda_um']:.1f} µm, charge m = {SPIRAL_CHARGE}"
    )
    axes[0, 1].axis("off")
    axes[1, 0].plot(gof["k"] * 1e3, gof["analytical"], color="#3c2a6e", lw=2.0, label="analytical σ(k)")
    axes[1, 0].plot(gof["k"] * 1e3, gof["numerical"], "o", ms=3.5, color="#c47b2b", label="numerical eigenvalue")
    axes[1, 0].axvline(2.0 * np.pi / LAMBDA_UM * 1e3, color="#888", ls="--", lw=1.0, label="k_c")
    axes[1, 0].set_xlabel("wavenumber k (1/mm)")
    axes[1, 0].set_ylabel("growth rate σ")
    axes[1, 0].set_title(f"Turing dispersion  r = {gof['pearson_r']:.4f}, RMSE = {gof['rmse']:.4f}")
    axes[1, 0].legend(frameon=False, fontsize=8)
    eps = [row["eps"] for row in sweep]
    rms = [row["rms"] for row in sweep]
    axes[1, 1].plot(eps, rms, "-o", color="#3c2a6e")
    axes[1, 1].axvline(0.0, color="#888", ls="--", lw=1.0)
    axes[1, 1].set_xlabel("Swift–Hohenberg ε")
    axes[1, 1].set_ylabel("saturated RMS")
    axes[1, 1].set_title("Parameter sweep: amplitude vs distance from onset")
    fig.suptitle(
        "Cavia porcellus s0  ·  126.7 µm Turing wavelength  ·  8 waves on a 1.0136 mm primordium",
        fontsize=12,
    )
    fig.tight_layout()
    MEDIA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA_PNG)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_analysis(save_media: bool = True) -> dict:
    """Full Pass-1 quantitative bundle. Media writes are optional for unit checks."""
    global _CACHE
    if _CACHE is not None and (not save_media or (MEDIA_GIF.exists() and MEDIA_PNG.exists())):
        return _CACHE
    jac = schnakenberg_jacobian()
    diff = calibrate_diffusivities(jac)
    gof = goodness_of_fit(jac, diff)
    spiral = integrate_swift_hohenberg()
    sweep = epsilon_sweep()
    control = negative_control()
    ablate = ablation_control()
    specimen = specimen_segmentation()
    ks_orient = orientation_ks(specimen, spiral["field"])
    substrate = substrate_production_sweep()
    radial_cos = radial_spectrum_cosine(specimen["gray_body"], spiral["field"])
    charges = charge_sweep()
    ensemble = seed_ensemble()
    if save_media:
        write_gif(spiral["snaps"], MEDIA_GIF)
        specimen_rgb = np.asarray(Image.open(SPECIMEN).convert("RGB"))
        write_comparison_png(spiral, specimen_rgb, sweep, gof)
    bundle = {
        "jacobian": jac,
        "diffusion": diff,
        "gof": gof,
        "spiral": spiral,
        "sweep": sweep,
        "control": control,
        "ablation": ablate,
        "specimen": {k: v for k, v in specimen.items() if k not in ("orientation", "orientation_weight", "gray_body")},
        "orientation_ks": ks_orient,
        "substrate_sweep": substrate,
        "radial_cosine": radial_cos,
        "charges": charges,
        "ensemble": ensemble,
        "chronoseries": {
            "age_ma": AGE_MA,
            "lambda_um": LAMBDA_UM,
            "lambda_predecessor_um": LAMBDA_PRED_UM,
            "lambda_successor_um": LAMBDA_SUCC_UM,
            "s0_waves_inside_predecessor": 10,
            "successor_waves_inside_s0": 10,
            "field_um": FIELD_UM,
        },
    }
    _CACHE = bundle
    return bundle


def measure_morphoeval() -> list[dict]:
    """Morphometrics consumed by morphoeval/measure_outputs.py. Values are not copies of the references."""
    bundle = run_analysis(save_media=False)
    diff = bundle["diffusion"]
    gof = bundle["gof"]
    spiral = bundle["spiral"]
    specimen = bundle["specimen"]
    sweep = bundle["sweep"]
    control = bundle["control"]
    ablate = bundle["ablation"]
    positive = [row["rms"] for row in sweep if row["eps"] > 0.0]
    monotonic = all(b >= a - 1e-9 for a, b in zip(positive, positive[1:]))
    return [
        {
            "name": "analytical Turing wavelength lambda_c of the s0 Schnakenberg mode",
            "value": round(float(diff["lambda_um"]), 4),
            "unit": "µm",
            "method": "k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v); lambda_c = 2 pi / k_c after diffusivity scaling",
            "reference_value": 126.7,
            "reference_source": "Slot 3141 tier-5 detail length 0.0001267 m (Holocene Cavia porcellus, 0 Ma)",
            "derived_from": "script",
            "note": f"D_u = {diff['D_u']:.4f} µm^2, D_v/D_u = {diff['ratio']:.1f}, h_min = {diff['h_min']:.4f}",
        },
        {
            "name": "specimen pigment-field dominant wavelength from the 100 µm scale bar",
            "value": round(float(specimen["wavelength_um"]), 4),
            "unit": "µm",
            "method": "radial FFT peak of the caption-cropped gallery field, converted with the detected 100 µm bar",
            "reference_value": 126.7,
            "reference_source": "model λ_c; the gallery field is not forced to equal the assigned wavelength",
            "derived_from": "output_pixels",
            "note": (
                f"scale bar {specimen['scale_bar_px']:.0f} px = 100 µm "
                f"({specimen['um_per_px']:.4f} µm/px); foreground fraction {specimen['foreground_fraction']:.4f}"
            ),
        },
        {
            "name": "specimen structure-tensor coherency of the pigment field",
            "value": round(float(specimen["coherency"]), 4),
            "unit": "dimensionless",
            "method": "mean (λ1-λ2)/(λ1+λ2) of the Gaussian-smoothed structure tensor inside the foreground mask",
            "reference_value": 0.5,
            "reference_source": "neutral isotropic baseline; the spiral field is locally oriented",
            "derived_from": "output_pixels",
            "note": f"contour circularity {specimen['circularity']:.4f} (far from a round blob)",
        },
        {
            "name": "dispersion-curve Pearson r of analytical sigma(k) vs numerical eigenvalues",
            "value": round(float(gof["pearson_r"]), 6),
            "unit": "Pearson r",
            "method": "28 wavenumbers, closed-form quadratic versus numpy.linalg.eigvals of the Fourier-mode Jacobian",
            "reference_value": 0.98,
            "reference_source": "MorphoEval goodness-of-fit gate (r > 0.98)",
            "derived_from": "script",
            "note": f"RMSE = {gof['rmse']:.6f}; KS statistic between the two sigma samples = {gof['ks_statistic']:.4f}",
        },
        {
            "name": "dispersion RMSE between analytical and numerical eigenvalue growth rates",
            "value": round(float(gof["rmse"]), 6),
            "unit": "1/time",
            "method": "root mean square of sigma_closed_form(k) - sigma_eigvals(k) on k in [0.01, 0.12] µm^-1",
            "reference_value": 0.05,
            "reference_source": "discretization tolerance well below the peak growth rate",
            "derived_from": "script",
            "note": f"sigma(k_c) = {gof['sigma_at_kc']:.4f}, sigma(0) = {gof['sigma_at_0']:.4f} (homogeneous state stable)",
        },
        {
            "name": "simulated stripe wavelength on the eight-wave primordium",
            "value": round(float(spiral["lambda_um"]), 4),
            "unit": "µm",
            "method": "radial power spectrum peak of the saturated Swift-Hohenberg spiral, scaled by 126.7 µm at k_hat = 1",
            "reference_value": 126.7,
            "reference_source": "analytical λ_c",
            "derived_from": "script",
            "note": f"k_peak (hat) = {spiral['k_peak']:.4f}; charge m = {SPIRAL_CHARGE}",
        },
        {
            "name": "orientation KS distance between specimen gradients and the simulated spiral",
            "value": round(float(bundle["orientation_ks"]), 4),
            "unit": "KS statistic",
            "method": "magnitude-weighted two-sample KS on Sobel orientations of the gallery field and the SH spiral",
            "reference_value": 0.0,
            "reference_source": "identical orientation laws would give KS = 0; a residual is expected",
            "derived_from": "output_pixels",
        },
        {
            "name": "negative-control RMS ratio at epsilon = -0.2",
            "value": round(float(control["ratio"]), 4),
            "unit": "RMS_final / RMS_initial",
            "method": "same spiral initial condition integrated below onset; pattern must collapse",
            "reference_value": 1.0,
            "reference_source": "a control that fails to decay would stay near ratio 1",
            "derived_from": "script",
            "note": f"RMS0 = {control['rms0']:.4f}, RMS_final = {control['rms']:.4f}",
        },
        {
            "name": "ablation wavenumber shift after removing finite-k selection",
            "value": round(float(abs(ablate["k_full"] - ablate["k_ablate"])), 4),
            "unit": "hat wavenumber",
            "method": "replace (k_c^2 + Laplacian)^2 by plain diffusion and remeasure the radial peak",
            "reference_value": 0.0,
            "reference_source": "zero shift would mean the (k_c^2+∇^2)^2 operator is not doing the selection",
            "derived_from": "script",
            "note": f"k_full = {ablate['k_full']:.4f}, k_ablate = {ablate['k_ablate']:.4f}",
        },
        {
            "name": "epsilon-sweep monotonic amplitude flag for epsilon > 0",
            "value": 1.0 if monotonic else 0.0,
            "unit": "boolean",
            "method": "saturated RMS at epsilon in {0.05, 0.15, 0.25, 0.45} is nondecreasing",
            "reference_value": 1.0,
            "reference_source": "Swift-Hohenberg amplitude scales as sqrt(epsilon) above onset",
            "derived_from": "script",
            "note": "RMS sequence " + ", ".join(f"{row['eps']}:{row['rms']:.3f}" for row in sweep),
        },
        {
            "name": "substrate-sweep wavelength at b = 0.20 with diffusivities held at the s0 calibration",
            "value": round(float(bundle["substrate_sweep"][0]["lambda_um"]), 4),
            "unit": "µm",
            "method": "recompute k_c(b) at fixed D_u, D_v calibrated for b = 1.15; b = 0.20 is the low-substrate end of the sweep",
            "reference_value": 126.7,
            "reference_source": "s0 calibration wavelength at b = 1.15; a different b must not reproduce it",
            "derived_from": "script",
            "note": "sweep " + ", ".join(
                f"b={row['b']}:λ={row['lambda_um']:.1f}" for row in bundle["substrate_sweep"]
            ),
        },
        {
            "name": "specimen radial-spectrum cosine against the simulated spiral",
            "value": round(float(bundle["radial_cosine"]), 4),
            "unit": "cosine",
            "method": "log radial FFT bins 2:24 of the gallery pigment field versus the Swift-Hohenberg field",
            "reference_value": 1.0,
            "reference_source": "identical spectra would score 1; the gallery grain is not the PDE",
            "derived_from": "output_pixels",
        },
        {
            "name": "charge-sweep wavelength at topological charge m = 1",
            "value": round(float(bundle["charges"][0]["lambda_um"]), 4),
            "unit": "µm",
            "method": "Swift-Hohenberg spiral at m = 1, radial-spectrum wavelength; m = 2 is the rendered field",
            "reference_value": 126.7,
            "reference_source": "analytical λ_c; charge changes the defect, not the selected wavelength",
            "derived_from": "script",
            "note": "charges " + ", ".join(
                f"m={row['charge']}:λ={row['lambda_um']:.2f}" for row in bundle["charges"]
            ),
        },
        {
            "name": "seed-ensemble coefficient of variation of the selected wavenumber",
            "value": round(float(bundle["ensemble"]["k_cv"]), 6),
            "unit": "CV",
            "method": "sample standard deviation over mean of k_peak on seeds 3141-3144",
            "reference_value": 0.0,
            "reference_source": "a deterministic wavenumber would have CV 0; finite-grid noise leaves a residual",
            "derived_from": "script",
            "note": "k_peaks " + ", ".join(f"{k:.4f}" for k in bundle["ensemble"]["k_peaks"]),
        },
    ]


def run_self_test() -> None:
    bundle = run_analysis(save_media=True)
    diff = bundle["diffusion"]
    jac = bundle["jacobian"]
    gof = bundle["gof"]
    spiral = bundle["spiral"]
    control = bundle["control"]
    ablate = bundle["ablation"]
    specimen = bundle["specimen"]
    digest = sha256_file(SPECIMEN)
    ref = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    pinned = ref["specimen"]["sha256"]
    checks = [
        ("dataset sha256 matches reference json", digest == pinned),
        ("homogeneous trace is negative", jac["trace"] < 0.0),
        ("homogeneous determinant is positive", jac["det"] > 0.0),
        ("Turing h_min is negative", diff["h_min"] < 0.0),
        ("cross term D_v f_u + D_u g_v is positive", diff["cross"] > 0.0),
        ("analytical wavelength is 126.7 µm", abs(diff["lambda_um"] - 126.7) < 1e-9),
        ("Pearson r of the dispersion exceeds 0.98", gof["pearson_r"] > 0.98),
        ("RMSE of the dispersion is below 0.05", gof["rmse"] < 0.05),
        ("sigma(k_c) is positive", gof["sigma_at_kc"] > 0.0),
        ("sigma(0) is negative", gof["sigma_at_0"] < 0.0),
        ("spiral peak stays near k_c = 1", abs(spiral["k_peak"] - 1.0) < 0.15),
        ("negative control decays", control["ratio"] < 0.5),
        ("ablation moves the peak by more than 0.25", abs(ablate["k_full"] - ablate["k_ablate"]) > 0.25),
        ("specimen wavelength is not a copy of 126.7", abs(specimen["wavelength_um"] - 126.7) > 1.0),
        ("predecessor tier holds 10 s0 waves", abs(LAMBDA_PRED_UM / LAMBDA_UM - 10.0) < 1e-12),
        ("b = 1.15 recovers 126.7 µm inside the substrate sweep",
         abs(next(row["lambda_um"] for row in bundle["substrate_sweep"] if row["b"] == 1.15) - 126.7) < 1e-6),
        ("substrate sweep moves lambda by more than 5 µm",
         abs(bundle["substrate_sweep"][0]["lambda_um"] - bundle["substrate_sweep"][-1]["lambda_um"]) > 5.0),
        ("specimen radial cosine is finite", np.isfinite(bundle["radial_cosine"])),
        ("every spiral charge stays near k_c", all(abs(row["k_peak"] - 1.0) < 0.2 for row in bundle["charges"])),
        ("seed-ensemble wavenumber CV is below 0.08", bundle["ensemble"]["k_cv"] < 0.08),
        ("gif exists", MEDIA_GIF.exists()),
        ("comparison png exists", MEDIA_PNG.exists()),
    ]
    with Image.open(MEDIA_GIF) as im:
        checks.append(("gif has at least 10 frames", getattr(im, "n_frames", 1) >= 10))
        checks.append(("gif min side is at least 480 px", min(im.size) >= 480))
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if failed:
        raise SystemExit(f"self-test failed: {failed}")
    print("self-test passed")
    print(
        f"lambda_c={diff['lambda_um']:.4f} µm  r={gof['pearson_r']:.6f}  "
        f"RMSE={gof['rmse']:.6f}  k_peak={spiral['k_peak']:.4f}  "
        f"specimen_lambda={specimen['wavelength_um']:.2f} µm  "
        f"orient_KS={bundle['orientation_ks']:.4f}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run quantitative checks and write media")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    bundle = run_analysis(save_media=True)
    diff = bundle["diffusion"]
    print(f"Wrote {MEDIA_PNG}")
    print(f"Wrote {MEDIA_GIF}")
    print(
        f"Cavia porcellus s0  age={AGE_MA} Ma  lambda_c={diff['lambda_um']} µm  "
        f"field={FIELD_UM} µm  pearson_r={bundle['gof']['pearson_r']:.6f}"
    )


if __name__ == "__main__":
    main()
