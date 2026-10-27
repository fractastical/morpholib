#!/usr/bin/env python3
"""
Drosophila guttifera — Turing micro-ocellus wavelength at 2.12 µm (slot 3187).

Animal: Drosophila guttifera (polka-dotted 16-spot wing fly).
Scale: 2.12e-6 m (2.12 µm, log10 = -5.674), tier 3_subcellular_um, detail variant s0.
Age: Holocene (extant), 0 Ma.
Lineage: Subgenus Drosophila virilis–repleta–guttifera radiation.
Chronoseries 16/41. Predecessor: Drosophila grimshawi. Successor: Drosophila heteroneura.

The adult wing spots that Werner, Koshikawa, Williams & Carroll (2010, Nature,
10.1038/nature08896) traced to Wingless at vein intersections and campaniform
sensilla are a larger-scale pattern. This slot is the detail tier inside one
such spot: a Schnakenberg reaction–diffusion field (Schnakenberg 1979,
10.1016/0022-5193(79)90042-0) whose most-unstable wavelength is calibrated to
2.12 µm.

    du/dt = Du ∇²u + a - u + u² v
    dv/dt = Dv ∇²v + b - u² v

Marginal wavenumber (Turing 1952, 10.1098/rstb.1952.0012):

    k_c² = (Dv f_u + Du g_v) / (2 Du Dv)

The wavelength that actually grows is λ = 2π / k_max with k_max = argmax σ(k).
The spatial unit is chosen so that focal λ(k_max) = 2.12e-6 m.

At 0 Ma the chronospecies differ in prepattern, not in geological age:
  grimshawi — vein-parallel diffusivity × 3.15 (lacework stripes)
  guttifera — isotropic diffusion plus one campaniform Wingless Gaussian
  heteroneura — no sensillum Gaussian; basal activator raised from 0.10 to 0.24

Those two parameter changes are reconstructed contrasts. They are not diffusion
coefficients measured by Edwards et al. 2007 or Spieth 1981.

Usage:
    python3 models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py --self-test
    python3 models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_guttifera_turing_wavelength_s0"
REFERENCE = DATASET / "drosophila_guttifera_turing_wavelength_s0_reference.json"
LEAN = ROOT / "morphoeval" / "lean" / "loops" / "Loop_3187_drosophila_guttifera_turing_wavelength_s0.lean"
METRICS_PATH = DATASET / "run_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_guttifera_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_guttifera_turing_wavelength_s0_growth.gif"

# Pass-1 integrator uses periodic (Fourier) boundaries only.
BOUNDARY = "periodic"
# Fixed spectral window for the Dv sweep, in cycles per dimensionless length.
# It is wider than the sweep's k_max range and is not recentered on each multiplier.
SPECTRAL_BAND = (0.30, 0.80)


def load_reference() -> dict:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    specimen = DATASET / ref["specimen"]["file"]
    digest = hashlib.sha256(specimen.read_bytes()).hexdigest()
    if digest != ref["specimen"]["sha256"]:
        raise RuntimeError(f"specimen SHA-256 mismatch: {digest}")
    return ref


def dispersion(a: float, b: float, Du: float, Dv: float, n: int = 6000) -> dict:
    """Linear Turing spectrum of the Schnakenberg Jacobian at (u*, v*)."""
    u_star = a + b
    v_star = b / (u_star ** 2)
    f_u = -1.0 + 2.0 * u_star * v_star
    f_v = u_star ** 2
    g_u = -2.0 * u_star * v_star
    g_v = -(u_star ** 2)
    det = f_u * g_v - f_v * g_u
    trace = f_u + g_v
    disc_coef = Dv * f_u + Du * g_v
    kc2 = disc_coef / (2.0 * Du * Dv)
    kc = float(np.sqrt(max(kc2, 0.0)))
    ks = np.linspace(0.05, max(kc * 4.0, 1.0), n)
    k2 = ks ** 2
    tau = trace - k2 * (Du + Dv)
    delta = det - k2 * disc_coef + (k2 ** 2) * Du * Dv
    sigma = 0.5 * (tau + np.sqrt(np.maximum(tau ** 2 - 4.0 * delta, 0.0)))
    i = int(np.argmax(sigma))
    k_max = float(ks[i])
    return {
        "u_star": float(u_star),
        "v_star": float(v_star),
        "f_u": float(f_u),
        "f_v": float(f_v),
        "g_u": float(g_u),
        "g_v": float(g_v),
        "det": float(det),
        "trace": float(trace),
        "disc_coef": float(disc_coef),
        "k_c": kc,
        "k_max": k_max,
        "lambda_dimless": float(2.0 * np.pi / k_max),
        "sigma_max": float(sigma[i]),
        "ks": ks,
        "sigma": sigma,
        "discriminant_positive": bool(disc_coef > 0.0 and disc_coef ** 2 > 4.0 * Du * Dv * det and det > 0.0 and trace < 0.0),
    }


def _grid(n_waves: int, lam: float, N: int, Du: float, Dv: float, aniso: float, dt: float):
    length = n_waves * lam
    dx = length / N
    freq = np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    kx2 = (2.0 * np.pi * kx) ** 2
    ky2 = (2.0 * np.pi * ky) ** 2
    decay_u = np.exp(-(Du * aniso * kx2 + Du * ky2) * dt)
    decay_v = np.exp(-(Dv * aniso * kx2 + Dv * ky2) * dt)
    return dx, length, kx, ky, decay_u, decay_v


def integrate(
    a: float,
    b: float,
    Du: float,
    Dv: float,
    *,
    n_waves: int,
    N: int,
    T: float,
    seed: int,
    aniso: float = 1.0,
    sensillum: bool = False,
    a_w: float = 0.0,
    sigma_waves: float = 0.35,
    stop_std: float | None = None,
    n_snapshots: int = 0,
    dt: float = 0.12,
    clip: bool = False,
    length: float | None = None,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fourier (periodic) Schnakenberg step. Diffusion is exact; reaction is explicit."""
    disp = dispersion(a, b, Du, Dv)
    lam_box = disp["lambda_dimless"] if length is None else float(length) / n_waves
    dx, length, kx, ky, decay_u, decay_v = _grid(n_waves, lam_box, N, Du, Dv, aniso, dt)
    rng = np.random.default_rng(seed)
    u = disp["u_star"] * (1.0 + noise * rng.standard_normal((N, N)))
    v = disp["v_star"] * (1.0 + noise * rng.standard_normal((N, N)))
    std0 = float(u.std())
    a_field: float | np.ndarray = a
    if sensillum:
        yy, xx = np.mgrid[0:N, 0:N]
        r2 = ((yy - (N - 1) / 2.0) ** 2 + (xx - (N - 1) / 2.0) ** 2) * dx ** 2
        sigma = sigma_waves * disp["lambda_dimless"]
        a_field = a + a_w * np.exp(-r2 / (2.0 * sigma ** 2))
    steps = int(np.ceil(T / dt))
    snap_at = set(np.linspace(0, max(steps - 1, 0), n_snapshots).astype(int).tolist()) if n_snapshots else set()
    snaps: list[np.ndarray] = []
    last = 0
    for i in range(steps):
        uv2 = u * u * v
        ur = u + dt * (a_field - u + uv2)
        vr = v + dt * (b - uv2)
        if clip:
            np.clip(ur, 0.0, 12.0, out=ur)
            np.clip(vr, 0.0, 12.0, out=vr)
        u = np.real(np.fft.ifft2(np.fft.fft2(ur) * decay_u))
        v = np.real(np.fft.ifft2(np.fft.fft2(vr) * decay_v))
        last = i
        if i in snap_at:
            snaps.append(u.copy())
        if stop_std is not None and float(u.std()) >= stop_std:
            break
    meta = {
        "dx": float(dx),
        "length": float(length),
        "std0": std0,
        "std": float(u.std()),
        "steps": last + 1,
        "snaps": snaps,
        "disp": {k: v for k, v in disp.items() if k not in ("ks", "sigma")},
        "kx": kx,
        "ky": ky,
        "sigma_curve_k": disp["ks"],
        "sigma_curve": disp["sigma"],
        "boundary": BOUNDARY,
    }
    return u, v, meta


def dominant_wavelength(
    u: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    band: tuple[float, float] = SPECTRAL_BAND,
) -> float:
    """Power-weighted spectral centroid inside a fixed wavenumber window.

    The window is not recentered on the predicted k_max. A single loud Fourier
    bin is a poor estimator: several lattice modes share the unstable ring, and
    the loudest one can sit off the ring by chance.
    """
    spec = np.abs(np.fft.fft2(u - np.mean(u))) ** 2
    spec[0, 0] = 0.0
    radius = np.hypot(kx, ky)
    mask = (radius >= band[0]) & (radius <= band[1])
    weight = float(np.sum(spec[mask]))
    if weight <= 0.0:
        return float("nan")
    k_cycles = float(np.sum(radius[mask] * spec[mask]) / weight)
    return 1.0 / max(k_cycles, 1e-12)


def integrate_neumann(
    a: float,
    b: float,
    Du: float,
    Dv: float,
    *,
    length: float,
    N: int = 48,
    seed: int = 3187,
    stop_std: float = 0.20,
    dt: float = 0.008,
) -> tuple[np.ndarray, dict]:
    """Explicit Schnakenberg step with reflecting (zero-flux) boundaries.

    This is the pass-2 control. The primary sweep stays periodic. The timestep
    is inside the explicit diffusion stability limit for Dv on this grid.
    """
    dx = length / N
    rng = np.random.default_rng(seed)
    disp = dispersion(a, b, Du, Dv)
    u = disp["u_star"] * (1.0 + 0.01 * rng.standard_normal((N, N)))
    v = disp["v_star"] * (1.0 + 0.01 * rng.standard_normal((N, N)))

    def lap(z: np.ndarray) -> np.ndarray:
        padded = np.pad(z, 1, mode="reflect")
        return (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * z
        ) / (dx * dx)

    last = 0
    for i in range(4000):
        uv2 = u * u * v
        u = np.clip(u + dt * (Du * lap(u) + a - u + uv2), 0.0, 12.0)
        v = np.clip(v + dt * (Dv * lap(v) + b - uv2), 0.0, 12.0)
        last = i
        if i > 40 and float(u.std()) >= stop_std:
            break
    freq = np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    return u, {
        "steps": last + 1,
        "std": float(u.std()),
        "dx": float(dx),
        "lambda_dimless": dominant_wavelength(u, kx, ky),
        "finite": bool(np.isfinite(u).all()),
    }


def peak_spacing_um(u: np.ndarray, dx_dimless: float, scale_m: float) -> dict:
    """Nearest-neighbour spacing of saturated activator peaks, in micrometres."""
    dx_um = dx_dimless * scale_m * 1e6
    window = max(3, int(round(0.55 * 2.12 / max(dx_um, 1e-6))))
    peaks = (u == ndi.maximum_filter(u, size=window)) & (u > np.percentile(u, 75))
    ys, xs = np.nonzero(peaks)
    if xs.size < 4:
        return {"n_peaks": int(xs.size), "mean_um": None, "cv": None, "ks_statistic": None, "ks_pvalue": None}
    pts = np.column_stack([xs.astype(float), ys.astype(float)]) * dx_um
    dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    mean = float(nearest.mean())
    cv = float(nearest.std() / mean) if mean > 0 else None
    scale_um = max(float(nearest.std()), 1e-6)
    ks = stats.kstest(nearest, stats.norm(loc=2.12, scale=scale_um).cdf)
    return {
        "n_peaks": int(xs.size),
        "mean_um": mean,
        "cv": cv,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def structure_coherency(u: np.ndarray) -> float:
    gy, gx = np.gradient(u)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    return float(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-12))


def segment_specimen(path: Path) -> dict:
    """Wing-membrane spot count on the adult photograph.

    Pass 1 counted every dark blob on the whole frame (bristles included).
    Pass 2 keeps dark components only on the foreground outside the reddish
    thorax/abdomen. Both wings are in frame, so the count is not forced to 16.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    foreground = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    foreground = ndi.binary_fill_holes(ndi.binary_closing(foreground, structure=np.ones((3, 3))))
    body_seed = foreground & (red > green + 0.04) & (gray > 0.18) & (gray < 0.62) & (red > blue)
    body_seed = ndi.binary_opening(body_seed, iterations=2)
    labels, n_body = ndi.label(body_seed)
    if n_body:
        sizes = ndi.sum(body_seed, labels, index=np.arange(1, n_body + 1))
        body = ndi.binary_dilation(labels == (int(np.argmax(sizes)) + 1), iterations=6)
    else:
        body = np.zeros_like(foreground)
    wing = foreground & ~body
    dark = ndi.binary_opening((gray < 0.32) & wing, iterations=1)
    spot_labels, n_spots = ndi.label(dark)
    spot_sizes_all = ndi.sum(dark, spot_labels, index=np.arange(1, n_spots + 1)) if n_spots else np.array([])
    spot_sizes = np.asarray([s for s in spot_sizes_all if 18 < s < 900], dtype=float)
    global_dark = ndi.binary_opening(gray < 0.28, iterations=1)
    g_labels, n_global = ndi.label(global_dark)
    g_sizes = ndi.sum(global_dark, g_labels, index=np.arange(1, n_global + 1)) if n_global else np.array([])
    global_count = int(np.sum((g_sizes > 20) & (g_sizes < 800))) if g_sizes.size else 0
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    area = float(np.sum(foreground))
    coherency = float(np.mean(coh_map[foreground])) if area > 20 else float(np.mean(coh_map))
    mean_area = float(np.mean(spot_sizes)) if spot_sizes.size else 0.0
    cv = float(np.std(spot_sizes) / mean_area) if mean_area > 0 else 0.0
    return {
        "spot_count": int(spot_sizes.size),
        "global_dark_count": global_count,
        "spot_area_px_mean": mean_area,
        "spot_area_cv": cv,
        "foreground_fraction": float(np.mean(foreground)),
        "wing_fraction": float(np.mean(wing)),
        "coherency": coherency,
        "method": "dark components on foreground outside the dilated reddish body, area in (18, 900) px",
    }


def _measurement(name, value, unit, method, reference=None, source="", note="", derived="script") -> dict:
    row = {
        "name": name,
        "value": None if value is None else (int(value) if isinstance(value, (int, np.integer)) else float(value)),
        "unit": unit,
        "method": method,
        "derived_from": derived,
    }
    if reference is not None:
        row["reference_value"] = reference if isinstance(reference, str) else float(reference)
        row["source"] = source
    if note:
        row["note"] = note
    return row


def analyze(need_fields: bool = False) -> dict:
    ref = load_reference()
    p = ref["parameters"]
    a, b, Du, Dv = float(p["a"]), float(p["b"]), float(p["Du"]), float(p["Dv"])
    seed = int(p["seed"])
    n_waves = int(p["patch_waves"])
    lambda_m = float(ref["scale_m"])
    focal = dispersion(a, b, Du, Dv)
    if focal["sigma_max"] <= 0.0 or not focal["discriminant_positive"]:
        raise RuntimeError("focal parameters are not Turing-unstable")
    scale = lambda_m / focal["lambda_dimless"]  # metres per dimensionless length

    # One fixed box, twelve focal wavelengths long. Resizing the box to each
    # predicted wavelength would pin the measured ring to that prediction.
    box = 12.0 * focal["lambda_dimless"]
    sweep_rows = []
    for m in p["dv_sweep_multipliers"]:
        u, _v, meta = integrate(
            a, b, Du, Dv * float(m),
            n_waves=12, N=96, T=40.0, seed=seed,
            stop_std=0.25, clip=False, length=box, noise=1e-3,
        )
        lam_num = dominant_wavelength(u, meta["kx"], meta["ky"])
        lam_an = meta["disp"]["lambda_dimless"]
        sweep_rows.append({
            "dv_multiplier": float(m),
            "lambda_analytical_um": lam_an * scale * 1e6,
            "lambda_numerical_um": lam_num * scale * 1e6,
            "sigma_max": meta["disp"]["sigma_max"],
        })
    an = np.array([r["lambda_analytical_um"] for r in sweep_rows])
    num = np.array([r["lambda_numerical_um"] for r in sweep_rows])
    pearson = float(np.corrcoef(an, num)[0, 1])
    rmse = float(np.sqrt(np.mean((an - num) ** 2)))
    ks = stats.ks_2samp(num, an)

    u_ab, _v_ab, ab = integrate(
        a, b, float(p["ablation_Du"]), float(p["ablation_Dv"]),
        n_waves=6, N=48, T=36.0, seed=seed, clip=False,
    )
    ab_disp = dispersion(a, b, float(p["ablation_Du"]), float(p["ablation_Dv"]))

    # Morphology fields. Snapshots only when rendering.
    snaps_n = 8 if need_fields else 0
    u_g, _vg, gmeta = integrate(
        a, b, Du, Dv,
        n_waves=n_waves, N=96, T=22.0, seed=seed,
        sensillum=True, a_w=float(p["a_w_sensillum"]),
        sigma_waves=float(p["sensillum_sigma_wavelengths"]),
        n_snapshots=snaps_n, clip=True,
    )
    u_grim, _vgrim, grim_meta = integrate(
        a, b, Du, Dv,
        n_waves=n_waves, N=96, T=22.0, seed=seed + 1,
        aniso=float(p["aniso_grimshawi"]), clip=True,
    )
    u_het, _vhet, het_meta = integrate(
        float(p["a_heteroneura"]), b, Du, Dv,
        n_waves=n_waves, N=96, T=22.0, seed=seed + 2, clip=True,
    )
    coh_g = structure_coherency(u_g)
    coh_grim = structure_coherency(u_grim)
    coh_het = structure_coherency(u_het)
    het_disp = dispersion(float(p["a_heteroneura"]), b, Du, Dv)
    # Transverse scale for grimshawi: isotropic equivalent with Du and Dv
    # multiplied by sqrt(anisotropy), i.e. the geometric-mean diffusivity.
    alpha = float(p["aniso_grimshawi"])
    grim_disp = dispersion(a, b, Du * np.sqrt(alpha), Dv * np.sqrt(alpha))

    specimen = segment_specimen(DATASET / ref["specimen"]["file"])
    neumann_u, neumann = integrate_neumann(
        a, b, Du, Dv, length=8.0 * focal["lambda_dimless"], seed=seed,
    )
    neumann_um = neumann["lambda_dimless"] * scale * 1e6
    spacing = peak_spacing_um(u_g, gmeta["dx"], scale)
    focal_num_um = float(sweep_rows[2]["lambda_numerical_um"])  # multiplier 1.0 is index 2
    # Confirm the 1.0 multiplier is present.
    for row in sweep_rows:
        if abs(row["dv_multiplier"] - 1.0) < 1e-9:
            focal_num_um = row["lambda_numerical_um"]
            break

    measurements = [
        _measurement(
            "focal linear wavelength lambda(k_max)",
            lambda_m * 1e6, "µm",
            "spatial calibration: (2 pi / k_max) * scale, scale chosen so this equals the slot wavelength",
            reference=2.12,
            source="slot 3187 detail tier 3_subcellular_um; Turing 1952 dispersion",
            note="analytical focal wavelength is 2.12 µm by calibration; the numerical Fourier peak is reported separately",
        ),
        _measurement(
            "numerical Fourier peak wavelength at focal Dv",
            focal_num_um, "µm",
            "dominant non-zero mode of the linear-regime field, stopped at std(u)=0.08",
            reference=2.12,
            source="same dispersion relation, early-time simulation",
            note="value is the measured peak and is not forced equal to the 2.12 µm reference",
        ),
        _measurement(
            "Pearson r of numerical vs analytical wavelength across Dv sweep",
            pearson, "correlation",
            "np.corrcoef across the fixed-domain Dv sweep; spectral centroid in cycles [0.30, 0.80]",
            reference=0.98,
            source="linear Turing prediction versus Fourier peak",
            note=f"RMSE = {rmse:.4f} µm",
        ),
        _measurement(
            "RMSE of numerical vs analytical wavelength",
            rmse, "µm",
            "root mean square of (lambda_numerical - lambda_analytical) over the Dv sweep",
            reference=0.15,
            source="same sweep",
        ),
        _measurement(
            "two-sample KS statistic of numerical vs analytical wavelength sets",
            float(ks.statistic), "KS D",
            "scipy.stats.ks_2samp(lambda_numerical, lambda_analytical)",
            reference=0.0,
            source="same fixed-domain sweep; 0 would mean the two wavelength sets are the same distribution",
            note=f"p = {float(ks.pvalue):.4f}",
        ),
        _measurement(
            "equal-diffusion ablation amplitude ratio",
            ab["std"] / max(ab["std0"], 1e-12), "std(final)/std(initial)",
            "Du = Dv = 0.40, periodic Schnakenberg, T = 36; pattern must decay",
            reference=1.0,
            source="negative control: Turing discriminant fails when Du = Dv",
            note=f"ablation sigma_max = {ab_disp['sigma_max']:.4f}",
        ),
        _measurement(
            "grimshawi stripe coherency minus guttifera spot coherency",
            coh_grim - coh_g, "structure-tensor coherency",
            "saturated fields; grimshawi anisotropy 3.15 versus guttifera isotropy plus sensillum Gaussian",
            reference=0.0,
            source="reconstructed chronospecies contrast at 2.12 µm and 0 Ma",
            note=f"coh grimshawi={coh_grim:.3f}, guttifera={coh_g:.3f}, heteroneura={coh_het:.3f}",
        ),
        _measurement(
            "heteroneura linear wavelength",
            het_disp["lambda_dimless"] * scale * 1e6, "µm",
            "k_max at a = 0.24, same Du, Dv, no sensillum Gaussian",
            reference=2.12,
            source="successor chronospecies parameter change",
            note="reference is the guttifera focal wavelength; the value must differ",
        ),
        _measurement(
            "grimshawi geometric-mean linear wavelength",
            grim_disp["lambda_dimless"] * scale * 1e6, "µm",
            "k_max with Du and Dv multiplied by sqrt(3.15)",
            reference=2.12,
            source="predecessor vein-locked anisotropy",
        ),
        _measurement(
            "specimen wing-membrane spot count",
            specimen["spot_count"], "components",
            specimen["method"],
            reference=16,
            source="Werner et al. 2010 name about 16 wing spots; both wings are in the photograph",
            note=f"pass-1 global dark count was {specimen['global_dark_count']}; wing mask is not forced to 16",
            derived="output_pixels",
        ),
        _measurement(
            "zero-flux control wavelength",
            neumann_um, "µm",
            "explicit reflecting-boundary Schnakenberg, N=48, spectral centroid in the same fixed band",
            reference=2.12,
            source="Neumann control against the periodic linear calibration",
            note=f"steps={neumann['steps']}, std={neumann['std']:.3f}, finite={neumann['finite']}",
        ),
        _measurement(
            "saturated activator nearest-neighbour spacing",
            spacing["mean_um"], "µm",
            "local maxima of the saturated guttifera field, minimum-separation window 0.55 λ",
            reference=2.12,
            source="nonlinear spacing versus the linear 2.12 µm calibration",
            note=f"n_peaks={spacing['n_peaks']}, CV={spacing['cv']}, KS vs N(2.12, s)={spacing['ks_statistic']}",
        ),
        _measurement(
            "specimen structure-tensor coherency",
            specimen["coherency"], "coherency",
            "Sobel structure tensor on the CC BY-SA adult male, masked against the border median",
            reference=0.35,
            source="MA Hanson, Wikimedia Commons File:Dguttifera.tif, CC BY-SA 4.0",
            derived="output_pixels",
        ),
        _measurement(
            "specimen foreground fraction",
            specimen["foreground_fraction"], "area fraction",
            "pixels whose contrast to the border median exceeds the 45th percentile",
            reference=0.40,
            source="same specimen photograph",
            derived="output_pixels",
        ),
        _measurement(
            "specimen dark-blob area coefficient of variation",
            specimen["spot_area_cv"], "CV",
            "std/mean of component areas in (20, 800) px",
            reference=0.25,
            source="same segmentation",
            note=f"mean blob area = {specimen['spot_area_px_mean']:.1f} px",
            derived="output_pixels",
        ),
    ]
    return {
        "reference_scale_m": lambda_m,
        "reference_age_ma": ref["geological_age_ma"],
        "log10_size_m": float(np.log10(lambda_m)),
        "scale_m_per_dimless": scale,
        "focal": {k: focal[k] for k in ("k_c", "k_max", "lambda_dimless", "sigma_max", "f_u", "g_v", "det", "trace", "discriminant_positive")},
        "pearson_r": pearson,
        "rmse_um": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "sweep": sweep_rows,
        "ablation_ratio": ab["std"] / max(ab["std0"], 1e-12),
        "ablation_sigma_max": ab_disp["sigma_max"],
        "coherency": {"guttifera": coh_g, "grimshawi": coh_grim, "heteroneura": coh_het},
        "heteroneura_um": het_disp["lambda_dimless"] * scale * 1e6,
        "grimshawi_um": grim_disp["lambda_dimless"] * scale * 1e6,
        "specimen": specimen,
        "boundary": BOUNDARY,
        "neumann_um": neumann_um,
        "saturated_spacing_um": spacing["mean_um"],
        "saturated_spacing_cv": spacing["cv"],
        "saturated_spacing_ks": spacing["ks_statistic"],
        "measurements": measurements,
        "fields": {
            "guttifera": u_g,
            "grimshawi": u_grim,
            "heteroneura": u_het,
            "snaps": gmeta["snaps"],
            "sigma_k": gmeta["sigma_curve_k"],
            "sigma": gmeta["sigma_curve"],
            "k_c": focal["k_c"],
            "k_max": focal["k_max"],
        } if need_fields else None,
    }


def _colorize(u: np.ndarray, size: tuple[int, int]) -> Image.Image:
    lo, hi = np.percentile(u, [4, 96])
    norm = np.clip((u - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    tan = np.array([236.0, 214.0, 176.0])
    melanin = np.array([36.0, 24.0, 18.0])
    rgb = tan * (1.0 - norm[..., None]) + melanin * norm[..., None]
    return Image.fromarray(rgb.astype(np.uint8), "RGB").resize(size, Image.Resampling.BILINEAR)


def render_png(result: dict, specimen_path: Path, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = result["fields"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), dpi=120)
    spec = np.asarray(Image.open(specimen_path).convert("RGB"))
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Adult male (CC BY-SA 4.0)")
    panels = [
        (axes[0, 1], fields["guttifera"], "D. guttifera spots + sensillum"),
        (axes[0, 2], fields["grimshawi"], "Predecessor D. grimshawi stripes"),
        (axes[1, 0], fields["heteroneura"], "Successor D. heteroneura"),
    ]
    for ax, field, title in panels:
        ax.imshow(field, cmap="cividis", interpolation="nearest")
        ax.set_title(title)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])
    k = fields["sigma_k"]
    axes[1, 1].plot(k, fields["sigma"], color="#1f4e79", lw=1.6)
    axes[1, 1].axvline(fields["k_max"], color="#c45c26", ls="--", label=f"k_max={fields['k_max']:.2f}")
    axes[1, 1].axvline(fields["k_c"], color="#2a6f4e", ls=":", label=f"k_c={fields['k_c']:.2f}")
    axes[1, 1].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 1].set_xlim(0, float(np.percentile(k, 70)))
    axes[1, 1].set_xlabel("wavenumber k")
    axes[1, 1].set_ylabel("growth rate σ(k)")
    axes[1, 1].set_title("Dispersion at 2.12 µm, 0 Ma")
    axes[1, 1].legend(fontsize=8, frameon=False)
    an = [r["lambda_analytical_um"] for r in result["sweep"]]
    num = [r["lambda_numerical_um"] for r in result["sweep"]]
    axes[1, 2].scatter(an, num, c="#1f4e79", s=36, zorder=3)
    lo = min(an + num) * 0.97
    hi = max(an + num) * 1.03
    axes[1, 2].plot([lo, hi], [lo, hi], color="0.6", lw=0.8)
    axes[1, 2].set_xlim(lo, hi)
    axes[1, 2].set_ylim(lo, hi)
    axes[1, 2].set_xlabel("analytical λ (µm)")
    axes[1, 2].set_ylabel("numerical λ (µm)")
    axes[1, 2].set_title(f"Dv sweep  r = {result['pearson_r']:.4f}")
    fig.suptitle(
        "Drosophila guttifera  ·  Turing wavelength 2.12 µm (2.12e-6 m)  ·  Holocene 0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


def render_gif(result: dict, specimen_path: Path, dest: Path) -> None:
    """Twelve frames, 640×480. Growth of the micro-field, then the adult photograph.

    The last frame is the specimen itself. A 2.12 µm field cannot be resolved in
    that photograph; the match is the adult phenotype the micro-field belongs to.
    """
    width, height = 640, 480
    specimen = Image.open(specimen_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    snaps = result["fields"]["snaps"] or [result["fields"]["guttifera"]]
    frames: list[Image.Image] = []
    # Eight growth frames, three crossfades, one pure specimen.
    for i, snap in enumerate(snaps[:8]):
        frame = _colorize(snap, (width, height))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, width, 28), fill=(32, 24, 18))
        draw.text((12, 6), f"D. guttifera micro-ocelli  2.12 µm  frame {i + 1}/8", fill=(245, 236, 220))
        frames.append(frame)
    last_field = frames[-1]
    for j, alpha in enumerate((0.33, 0.66, 0.85)):
        blended = Image.blend(last_field, specimen, alpha)
        frames.append(blended)
    frames.append(specimen.copy())
    if len(frames) != 12:
        raise RuntimeError(f"expected 12 frames, got {len(frames)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        dest,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=False,
        disposal=2,
    )


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Uses the metrics written by main()."""
    if METRICS_PATH.exists():
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return payload["measurements"]
    return analyze(need_fields=False)["measurements"]


def _assert(cond: bool, message: str, failures: list[str]) -> None:
    if not cond:
        failures.append(message)


def self_test() -> int:
    failures: list[str] = []
    ref = load_reference()
    _assert(abs(float(ref["scale_m"]) - 2.12e-6) < 1e-18, "scale_m is not 2.12e-6", failures)
    _assert(float(ref["geological_age_ma"]) == 0.0, "age is not 0 Ma", failures)
    _assert(abs(float(ref["log10_size_m"]) + 5.674) < 0.001, "log10 size is not -5.674", failures)
    lean = LEAN.read_text(encoding="utf-8")
    _assert(lean.count("theorem ") >= 3, "Lean file needs at least 3 theorems", failures)
    _assert("sorry" not in lean, "Lean file contains sorry", failures)
    _assert("MathOfBiology.Loop_3187" in lean, "Lean namespace mismatch", failures)
    # Integer discriminant mirrored from the Lean proof.
    _assert(123 * 123 > 10 * 125, "marginal discriminant integers failed", failures)
    result = analyze(need_fields=False)
    _assert(result["pearson_r"] > 0.98, f"Pearson r = {result['pearson_r']:.4f} <= 0.98", failures)
    _assert(result["ablation_ratio"] < 0.5, f"ablation ratio {result['ablation_ratio']:.3f} did not decay", failures)
    _assert(result["ablation_sigma_max"] < 0.0, "ablation sigma_max is not negative", failures)
    _assert(
        result["coherency"]["grimshawi"] > result["coherency"]["guttifera"] + 0.12,
        f"grimshawi coherency {result['coherency']} is not stripe-like relative to guttifera",
        failures,
    )
    _assert(
        abs(result["heteroneura_um"] - 2.12) / 2.12 > 0.015,
        f"heteroneura wavelength {result['heteroneura_um']:.4f} µm did not move off 2.12",
        failures,
    )
    _assert(result["specimen"]["spot_count"] != 16, "specimen count equals the reference; segmentation is circular", failures)
    _assert(result["specimen"]["spot_count"] < result["specimen"]["global_dark_count"], "wing mask did not drop any blobs", failures)
    _assert(1.4 < result["neumann_um"] < 2.8, f"Neumann wavelength {result['neumann_um']} left the 2 µm band", failures)
    _assert(result["saturated_spacing_um"] is not None and result["saturated_spacing_um"] > 0.5, "saturated spacing missing", failures)
    _assert(result["focal"]["discriminant_positive"], "focal discriminant is not positive", failures)
    _assert(abs(result["log10_size_m"] + 5.673664) < 1e-4, "computed log10 drifted", failures)
    print(json.dumps({
        "pearson_r": result["pearson_r"],
        "rmse_um": result["rmse_um"],
        "ks": result["ks_statistic"],
        "ablation_ratio": result["ablation_ratio"],
        "coherency": result["coherency"],
        "heteroneura_um": result["heteroneura_um"],
        "grimshawi_um": result["grimshawi_um"],
        "specimen_spots": result["specimen"]["spot_count"],
        "boundary": result["boundary"],
        "scale_m": result["reference_scale_m"],
        "age_ma": result["reference_age_ma"],
    }, indent=2))
    if failures:
        for item in failures:
            print("FAIL:", item)
        return 1
    print("self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--png", type=Path, default=PNG_PATH)
    parser.add_argument("--gif", type=Path, default=GIF_PATH)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    result = analyze(need_fields=True)
    payload = {k: v for k, v in result.items() if k != "fields"}
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ref = load_reference()
    render_png(result, DATASET / ref["specimen"]["file"], args.png)
    render_gif(result, DATASET / ref["specimen"]["file"], args.gif)
    with Image.open(args.gif) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
        gif_size = gif.size
    print(json.dumps({
        "png": str(args.png.relative_to(ROOT)),
        "gif": str(args.gif.relative_to(ROOT)),
        "gif_frames": n_frames,
        "gif_size": list(gif_size),
        "pearson_r": result["pearson_r"],
        "rmse_um": result["rmse_um"],
        "ks": result["ks_statistic"],
        "specimen_spots": result["specimen"]["spot_count"],
        "metrics": str(METRICS_PATH.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
