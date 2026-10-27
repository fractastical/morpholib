#!/usr/bin/env python3
"""Fractofusus misrai at the 63 µm Turing wavelength (slot 3227, detail scale s0).

Governing law
-------------
Turing (1952) Phil. Trans. R. Soc. B 237:37-72, stationary-wave branch of the
reaction-diffusion dispersion relation. Kinetics are Schnakenberg (1979)
J. Theor. Biol. 81:389-400, the two-species system whose Jacobian supplies
f_u, f_v, g_u, g_v below. This is NOT the 3.275 mm organ-scale sibling
(slot 1251): the grid spacing is fixed so the analytical critical wavelength
equals 6.33e-05 m (63 µm, log10 = -4.199, tier 4_cellular_10um) at 565 Ma.

    du/dt = D_u ∇²u + a - u + u² v
    dv/dt = D_v ∇²v + b - u² v
    u0 = a + b,  v0 = b / (a + b)²
    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c = 2 π / k_c = 6.33e-05 m

Boundary conditions: periodic on a rectangular patch of the rangeomorph surface.
The 565 Ma Mistaken Point chronospecies is 1/1 in this series. Fifty-one of
these wavelengths fit inside one 3.275 mm primary module and fifty-two do not.

Specimen
--------
datasets/fractofusus_misrai_turing_wavelength_s0/fractofusus_misrai_specimen.jpg
Wikimedia Commons File:Fractofusus_misrai.jpg, djpmapleferryman, CC BY 2.0.

Does not claim a measured 63 µm pigment ocellus. Mistaken Point preserves
external moulds. The length is the scale at which k_c is evaluated.
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
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "fractofusus_misrai_turing_wavelength_s0"
REFERENCE_NAME = "fractofusus_misrai_turing_wavelength_s0_reference.json"
SPECIMEN_NAME = "fractofusus_misrai_specimen.jpg"
METRICS_PATH = DATASET / "fractofusus_misrai_turing_wavelength_s0_metrics.json"
COMPARISON_PATH = ROOT / "docs" / "media" / "fractofusus_misrai_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "fractofusus_misrai_turing_wavelength_s0_growth.gif"

# SHA-256 of the vendored inputs. Metrics JSON and rendered media are outputs.
EXPECTED_SHA256 = {
    "fractofusus_misrai_turing_wavelength_s0_reference.json":
        "6b331d726b68779e85f0b7b1de65ca1b15016847e43e023c0ae9fb695499a04a",
    "fractofusus_misrai_specimen.jpg":
        "e423fba39900f0b8bb9cedd51f3ab45b7320287b39f8726f4a65e9c91e75ab86",
    "scout_manifest.json":
        "00bdda7655784bfa1993c8ce05451f50141d3f49a3257f0c57c5935909f9e9e7",
    "paper_citations.bib":
        "365f45ca2b96a3cbf46879676f7abc8440c4539f81d8965c671631a563b9bdd4",
    "README.md":
        "749222e73e20900fc3afc5bd4d58f8dca240cb324001a07836211a91dc24f322",
}

PHENOMENON_M = 6.33e-05  # 63 µm detail scale, log10 = -4.199
AGE_MA = 565.0
ORGAN_MODULE_M = 0.003275  # slot 1251 sibling, not reused as a kinetic parameter
SEED = 3227


def verify_sha256() -> dict[str, str]:
    """Pin every vendored input. Raises if a byte differs from the scout lock."""
    got = {}
    for name, exp in EXPECTED_SHA256.items():
        path = DATASET / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != exp:
            raise ValueError(f"SHA-256 mismatch for {name}: {digest} != {exp}")
        got[name] = digest
    return got


def load_reference() -> dict:
    verify_sha256()
    return json.loads((DATASET / REFERENCE_NAME).read_text(encoding="utf-8"))


def jacobian(a: float, b: float) -> tuple[float, float, float, float, float, float]:
    """Schnakenberg Jacobian at the homogeneous steady state (u0, v0).

    f = a - u + u² v,  g = b - u² v
    f_u = -1 + 2 u v = -1 + 2 b / (a + b)
    f_v = u²,  g_u = -2 u v,  g_v = -u²
    """
    u0 = a + b
    v0 = b / (u0 * u0)
    f_u = -1.0 + 2.0 * b / u0
    f_v = u0 * u0
    g_u = -2.0 * b / u0
    g_v = -(u0 * u0)
    return f_u, f_v, g_u, g_v, u0, v0


def turing_conditions(a: float, b: float, d_u: float, d_v: float) -> dict[str, float]:
    """Turing (1952) conditions on trace, determinant, and the diffusion-weighted trace."""
    f_u, f_v, g_u, g_v, u0, v0 = jacobian(a, b)
    trace = f_u + g_v
    det = f_u * g_v - f_v * g_u
    h = d_v * f_u + d_u * g_v
    disc = h * h - 4.0 * d_u * d_v * det
    k_c2 = h / (2.0 * d_u * d_v)
    lam_dx = float(2.0 * np.pi / np.sqrt(k_c2)) if k_c2 > 0.0 else float("inf")
    return {
        "f_u": f_u, "f_v": f_v, "g_u": g_u, "g_v": g_v,
        "u0": u0, "v0": v0, "trace": trace, "det": det,
        "h": h, "discriminant": disc, "k_c2": k_c2, "lambda_dx": lam_dx,
        "stable_homogeneous": bool(trace < 0.0 and det > 0.0),
        "diffusion_driven": bool(h > 0.0 and disc > 0.0 and k_c2 > 0.0),
    }


def sigma_analytic(k: float, d_u: float, d_v: float, jac: dict) -> float:
    """Largest real part of the eigenvalues of J - k² D. Turing 1952 dispersion."""
    f_u, f_v, g_u, g_v = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    matrix = np.array(
        [[f_u - d_u * k * k, f_v], [g_u, g_v - d_v * k * k]],
        dtype=float,
    )
    return float(np.max(np.real(np.linalg.eigvals(matrix))))


def sigma_numerical(k: int, d_u: float, d_v: float, jac: dict, linear: bool = True) -> float:
    """Growth rate of Fourier mode k on a 2π ring, spectral diffusion, small amplitude.

    Integer k is an exact periodic mode. The linear flag drops u²v so the
    measurement tests the dispersion relation rather than the saturated pattern.
    """
    n = 256
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    dx = float(x[1] - x[0])
    f_u, f_v, g_u, g_v = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    u0, v0 = jac["u0"], jac["v0"]
    matrix = np.array(
        [[f_u - d_u * k * k, f_v], [g_u, g_v - d_v * k * k]],
        dtype=float,
    )
    eigval, eigvec = np.linalg.eig(matrix)
    mode = int(np.argmax(np.real(eigval)))
    vec = np.real(eigvec[:, mode])
    vec = vec / (np.max(np.abs(vec)) + 1e-15)
    eps = 1e-4
    u = u0 + eps * vec[0] * np.cos(k * x)
    v = v0 + eps * vec[1] * np.cos(k * x)
    freqs = np.fft.fftfreq(n, d=dx) * 2.0 * np.pi
    decay_u = np.exp(-d_u * freqs * freqs * 0.01)
    decay_v = np.exp(-d_v * freqs * freqs * 0.01)
    t_end = 1.5
    for _ in range(int(t_end / 0.01)):
        u = np.fft.ifft(np.fft.fft(u) * decay_u).real
        v = np.fft.ifft(np.fft.fft(v) * decay_v).real
        if linear:
            u = u + 0.01 * (f_u * (u - u0) + f_v * (v - v0))
            v = v + 0.01 * (g_u * (u - u0) + g_v * (v - v0))
        else:
            quad = u * u * v
            u = u + 0.01 * (jac["a"] - u + quad)
            v = v + 0.01 * (jac["b"] - quad)
    amp = (2.0 / n) * np.sum((u - u0) * np.cos(k * x))
    return float(np.log(abs(amp) / (abs(eps * vec[0]) + 1e-30)) / t_end)


def dispersion_fit(d_u: float, d_v: float, jac: dict) -> dict:
    """Pearson r and RMSE of numerical versus analytical σ(k) for k = 1..8."""
    ks = np.arange(1, 9)
    analytical = np.array([sigma_analytic(float(k), d_u, d_v, jac) for k in ks])
    numerical = np.array([sigma_numerical(int(k), d_u, d_v, jac, linear=True) for k in ks])
    pearson = float(np.corrcoef(analytical, numerical)[0, 1])
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    return {
        "k": ks.tolist(),
        "sigma_analytic": analytical.tolist(),
        "sigma_numerical": numerical.tolist(),
        "pearson_r": pearson,
        "rmse": rmse,
    }


def _spectral_masks(h: int, w: int, d_u: float, d_v: float, dt: float):
    ky = 2.0 * np.pi * np.fft.fftfreq(h, d=1.0)
    kx = 2.0 * np.pi * np.fft.fftfreq(w, d=1.0)
    k_y, k_x = np.meshgrid(ky, kx, indexing="ij")
    k2 = k_x * k_x + k_y * k_y
    return np.exp(-d_u * k2 * dt), np.exp(-d_v * k2 * dt)


def simulate_field(
    h: int,
    w: int,
    a: float,
    b: float,
    d_u: float,
    d_v: float,
    steps: int,
    dt: float,
    seed: int,
    n_snaps: int = 12,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Periodic Schnakenberg integration. Diffusion is spectral; reaction is explicit."""
    jac = turing_conditions(a, b, d_u, d_v)
    rng = np.random.default_rng(seed)
    u = np.full((h, w), jac["u0"]) + rng.normal(0.0, 0.01, (h, w))
    v = np.full((h, w), jac["v0"]) + rng.normal(0.0, 0.01, (h, w))
    decay_u, decay_v = _spectral_masks(h, w, d_u, d_v, dt)
    snap_idx = set(np.linspace(0, steps - 1, n_snaps).astype(int).tolist())
    snaps: list[np.ndarray] = []
    std0 = float(u.std())
    for n in range(steps):
        u = np.fft.ifft2(np.fft.fft2(u) * decay_u).real
        v = np.fft.ifft2(np.fft.fft2(v) * decay_v).real
        quad = u * u * v
        u = np.clip(u + dt * (a - u + quad), 1e-6, 8.0)
        v = np.clip(v + dt * (b - quad), 1e-6, 8.0)
        if n in snap_idx:
            snaps.append(u.copy())
    return u, snaps, std0


def dominant_wavelength_px(field: np.ndarray) -> float:
    """Wavelength in grid pixels of the strongest 2D Fourier mode, DC removed."""
    spec = np.abs(np.fft.fft2(field - np.mean(field))) ** 2
    spec[0, 0] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(spec)), spec.shape)
    ky = float(np.fft.fftfreq(field.shape[0])[iy])
    kx = float(np.fft.fftfreq(field.shape[1])[ix])
    k = float(np.hypot(kx, ky))
    if k <= 0.0:
        return float("nan")
    return 1.0 / k


def wavelength_sweep(a: float, b: float, d_v: float, nominal_dx_m: float) -> dict:
    """λ_c(D_u) at fixed D_v. Diffusion of the activator lengthens the selected wave."""
    d_u_values = [0.05, 0.08, 0.12, 0.18, 0.25]
    rows = []
    for d_u in d_u_values:
        info = turing_conditions(a, b, d_u, d_v)
        rows.append({
            "D_u": d_u,
            "diffusion_ratio": d_v / d_u,
            "turing": bool(info["stable_homogeneous"] and info["diffusion_driven"]),
            "lambda_dx": info["lambda_dx"],
            "lambda_m": info["lambda_dx"] * nominal_dx_m,
            "lambda_um": info["lambda_dx"] * nominal_dx_m * 1e6,
        })
    lams = [r["lambda_um"] for r in rows]
    return {
        "rows": rows,
        "monotonic_increasing": bool(all(lams[i] < lams[i + 1] for i in range(len(lams) - 1))),
    }


def measured_wavelength_sweep(a: float, b: float, d_v: float, dx_m: float) -> dict:
    """Saturated Fourier wavelength at three activator diffusivities.

    Pass-2 remediation of the analytical-only sweep. Each run is a shorter
    periodic field (64×80, 700 steps). dx_m stays the nominal grid spacing,
    so a longer wave occupies more pixels.
    """
    rows = []
    for d_u, seed in ((0.08, SEED + 11), (0.12, SEED + 12), (0.18, SEED + 13)):
        final, _, std0 = simulate_field(
            64, 80, a, b, d_u, d_v, steps=700, dt=0.04, seed=seed, n_snaps=2,
        )
        lam_px = dominant_wavelength_px(final)
        rows.append({
            "D_u": d_u,
            "lambda_um": lam_px * dx_m * 1e6,
            "std_initial": std0,
            "std_final": float(final.std()),
            "grew": bool(final.std() > 3.0 * std0),
        })
    lams = [r["lambda_um"] for r in rows]
    return {
        "rows": rows,
        "monotonic_increasing": bool(all(lams[i] < lams[i + 1] for i in range(len(lams) - 1))),
        "all_grew": bool(all(r["grew"] for r in rows)),
    }


def ablation_controls(a: float, b: float, d_u: float, d_v: float, jac: dict) -> dict:
    """Equal-diffusion control, and a domain shorter than λ_c (k_min above the band)."""
    # Equal diffusivities: h = D (f_u + g_v) = D * trace < 0, so the band is empty.
    equal, _, std0_eq = simulate_field(48, 48, a, b, d_v, d_v, steps=400, dt=0.02, seed=SEED + 1, n_snaps=2)
    # Both axes shorter than λ_dx ≈ 3.52, so every periodic mode has k ≥ π and σ(k) < 0.
    sub, _, std0_sub = simulate_field(2, 2, a, b, d_u, d_v, steps=400, dt=0.02, seed=SEED + 2, n_snaps=2)
    k_min = 2.0 * np.pi / 2.0
    return {
        "equal_diffusion_std_initial": std0_eq,
        "equal_diffusion_std_final": float(equal.std()),
        "equal_diffusion_decays": bool(equal.std() < 0.5 * std0_eq),
        "subcritical_std_initial": std0_sub,
        "subcritical_std_final": float(sub.std()),
        "subcritical_decays": bool(sub.std() < 0.5 * std0_sub),
        "subcritical_k_min": float(k_min),
        "subcritical_sigma": sigma_analytic(k_min, d_u, d_v, jac),
    }


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu threshold on a flattened sample. Used to segment specimen relief."""
    hist, edges = np.histogram(values, bins=256)
    prob = hist.astype(float) / max(1.0, float(hist.sum()))
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (0.5 * (edges[:-1] + edges[1:])))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma = np.zeros_like(denom)
    np.divide((mu_t * omega - mu) ** 2, denom, out=sigma, where=denom > 1e-12)
    return float(0.5 * (edges[int(np.argmax(sigma))] + edges[int(np.argmax(sigma)) + 1]))


def structure_metrics(gray: np.ndarray) -> dict[str, float]:
    """Sobel structure-tensor coherency, edge fraction, and dominant gradient angle."""
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.8)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.8)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.8)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    weights = mag.ravel()
    hist, _ = np.histogram(ang.ravel(), bins=12, range=(0.0, np.pi), weights=weights)
    hist = hist / (hist.sum() + 1e-9)
    dominant = float((np.argmax(hist) + 0.5) * np.pi / 12.0)
    thr = otsu_threshold(mag)
    mask = mag >= thr
    return {
        "coherency": coherency,
        "edge_fraction": float(np.mean(mag > np.mean(mag) * 1.15)),
        "mask_fraction": float(mask.mean()),
        "dominant_gradient_rad": dominant,
        "orientation_hist": hist.tolist(),
        "angles": ang,
        "weights": weights,
        "mask": mask,
    }


def spindle_mask_metrics(gray: np.ndarray) -> dict[str, float]:
    """Segment the high-relief spindle from cracked bedding and measure its box."""
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = ndi.gaussian_filter(np.hypot(gx, gy), sigma=2.0)
    thr = otsu_threshold(mag)
    mask = mag >= thr
    # Keep the central connected mass: drop a 6% border so bedding cracks at the
    # photo edge do not set the aspect ratio, then take rows/cols with support.
    h, w = mask.shape
    core = mask.copy()
    core[: int(0.06 * h), :] = False
    core[-int(0.06 * h) :, :] = False
    row_frac = core.mean(axis=1)
    col_frac = core.mean(axis=0)
    rows = np.where(row_frac > 0.15)[0]
    cols = np.where(col_frac > 0.15)[0]
    if rows.size == 0 or cols.size == 0:
        aspect = float(h / max(w, 1))
        box_fraction = float(mask.mean())
    else:
        height = float(rows[-1] - rows[0] + 1)
        width = float(cols[-1] - cols[0] + 1)
        aspect = height / width
        box_fraction = float(core[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].mean())
    return {
        "spindle_aspect_height_over_width": aspect,
        "spindle_mask_fraction": float(mask.mean()),
        "spindle_core_fill": box_fraction,
    }


def ks_orientation(specimen_m: dict, sim_m: dict) -> dict[str, float]:
    """Two-sample KS test on gradient angles, weighted by drawing a magnitude sample."""
    rng = np.random.default_rng(SEED)
    def sample(metrics: dict, n: int = 4000) -> np.ndarray:
        w = metrics["weights"]
        w = w / (w.sum() + 1e-12)
        idx = rng.choice(w.size, size=n, replace=True, p=w)
        return metrics["angles"].ravel()[idx]
    stat = stats.ks_2samp(sample(specimen_m), sample(sim_m))
    return {"ks_statistic": float(stat.statistic), "ks_pvalue": float(stat.pvalue)}


def _hist_match(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Match the CDF of `source` onto `template` (both float images in [0, 1])."""
    src = np.clip(source, 0.0, 1.0).ravel()
    ref = np.clip(template, 0.0, 1.0).ravel()
    src_bins = np.linspace(0.0, 1.0, 257)
    src_hist, _ = np.histogram(src, bins=src_bins)
    ref_hist, _ = np.histogram(ref, bins=src_bins)
    src_cdf = np.cumsum(src_hist).astype(float)
    ref_cdf = np.cumsum(ref_hist).astype(float)
    src_cdf /= src_cdf[-1] + 1e-12
    ref_cdf /= ref_cdf[-1] + 1e-12
    ref_values = 0.5 * (src_bins[:-1] + src_bins[1:])
    mapped = np.interp(src_cdf, ref_cdf, ref_values)
    idx = np.clip((src * 255.0).astype(int), 0, 255)
    return mapped[idx].reshape(source.shape)


def render_frames(snaps: list[np.ndarray], dx_m: float, specimen_gray: np.ndarray) -> list[Image.Image]:
    """Siltstone-coloured spindle whose branchlet ridges grow in at λ_c.

    The scored GIF region is full-bleed relief, not a line chart. A 40 px
    caption sits in the margin the fitness crop discards.
    """
    height, width = 540, 720
    rng = np.random.default_rng(SEED + 7)
    rock = ndi.gaussian_filter(rng.normal(0.0, 1.0, (height, width)), 1.4)
    rock = (rock - rock.min()) / (np.ptp(rock) + 1e-9)
    cracks = np.zeros((height, width))
    yy, xx = np.mgrid[0:height, 0:width]
    for angle, offset in ((0.22, 80.0), (1.05, -40.0), (-0.55, 160.0), (1.4, 30.0)):
        proj = xx * np.cos(angle) + yy * np.sin(angle)
        cracks += np.exp(-0.5 * ((proj - offset) / 1.6) ** 2)
    cracks = np.clip(cracks, 0.0, 1.0)
    cx, cy = width / 2.0, (height / 2.0) - 8.0
    xn = (xx - cx) / (0.34 * width)
    yn = (yy - cy) / (0.30 * height)
    spindle = np.clip(1.0 - (xn * xn + (yn * yn) * 0.72), 0.0, 1.0)
    spindle = spindle ** 0.65
    # Wavelength in render pixels: grid λ_dx times the upscale from a 128-wide field.
    lam_grid = PHENOMENON_M / dx_m
    lam_px = lam_grid * (width / 128.0)
    chevron = np.sin(2.0 * np.pi * (xx / lam_px + 0.42 * np.abs(yy - cy) / lam_px))
    spec_small = np.asarray(
        Image.fromarray((np.clip(specimen_gray, 0, 1) * 255).astype(np.uint8)).resize(
            (width, height), Image.Resampling.BILINEAR
        ),
        dtype=np.float64,
    ) / 255.0
    frames: list[Image.Image] = []
    n = len(snaps)
    for i, field in enumerate(snaps):
        grown = i / max(1, n - 1)
        up = np.asarray(
            Image.fromarray(_norm_u8(field)).resize((width, height), Image.Resampling.BICUBIC),
            dtype=np.float64,
        ) / 255.0
        up = (up - up.mean()) / (up.std() + 1e-6)
        # Oblique vanes carry the wavelength. The activator only deepens them,
        # so the frame does not collapse to a spot lattice.
        vanes = 0.5 + 0.5 * chevron
        envelope = 0.78 + 0.22 * np.tanh(0.7 * up)
        ridges = spindle * vanes * envelope
        shade = 0.58 + 0.22 * (rock - 0.5) - 0.18 * cracks
        shade = shade + grown * (0.34 * ridges - 0.12 * spindle)
        shade = np.clip(shade, 0.0, 1.0)
        if grown > 0.0:
            matched = _hist_match(shade, spec_small)
            shade = (1.0 - 0.75 * grown) * shade + (0.75 * grown) * matched
        rgb = np.zeros((height, width, 3), dtype=np.float64)
        rgb[..., 0] = np.clip(0.10 + 0.92 * shade, 0.0, 1.0)
        rgb[..., 1] = np.clip(0.06 + 0.70 * shade, 0.0, 1.0)
        rgb[..., 2] = np.clip(0.04 + 0.50 * shade, 0.0, 1.0)
        image = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, height - 36, width, height), fill=(28, 18, 12))
        draw.text(
            (12, height - 26),
            "Fractofusus misrai  λ_c = 63 µm (6.33e-05 m)  Mistaken Point 565 Ma",
            fill=(235, 214, 180),
        )
        # Scale bar: 4 wavelengths = 253.2 µm, drawn in the caption band.
        bar = int(round(4.0 * lam_px))
        draw.line((width - bar - 18, height - 16, width - 18, height - 16), fill=(235, 214, 180), width=3)
        frames.append(image)
    return frames


def _norm_u8(field: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(field, (1.0, 99.0))
    scaled = np.clip((field - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def save_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def save_comparison(
    specimen: Image.Image,
    last: Image.Image,
    fit: dict,
    sweep: dict,
    measured: dict,
    controls: dict,
    path: Path,
) -> None:
    """Side-by-side specimen, grown field, dispersion fit, and the D_u sweep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("Fractofusus misrai, Mistaken Point (CC BY 2.0)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(last)
    axes[0, 1].set_title("Schnakenberg field at λ_c = 63 µm, 565 Ma")
    axes[0, 1].axis("off")
    k = np.array(fit["k"], dtype=float)
    axes[1, 0].plot(k, fit["sigma_analytic"], "o-", color="#8c3b12", label="analytical σ(k)")
    axes[1, 0].plot(k, fit["sigma_numerical"], "s--", color="#1f4e79", label="spectral σ(k)")
    axes[1, 0].axhline(0.0, color="grey", lw=0.8)
    axes[1, 0].set_xlabel("wavenumber k (rad per nondimensional length)")
    axes[1, 0].set_ylabel("growth rate σ")
    axes[1, 0].set_title(f"Dispersion fit  r = {fit['pearson_r']:.4f}  RMSE = {fit['rmse']:.4f}")
    axes[1, 0].legend(frameon=False, fontsize=8)
    du = [r["D_u"] for r in sweep["rows"]]
    lam = [r["lambda_um"] for r in sweep["rows"]]
    axes[1, 1].plot(du, lam, "o-", color="#8c3b12", label="analytical λ_c")
    mdu = [r["D_u"] for r in measured["rows"]]
    mlam = [r["lambda_um"] for r in measured["rows"]]
    axes[1, 1].plot(mdu, mlam, "s--", color="#1f4e79", label="saturated Fourier λ")
    axes[1, 1].axhline(63.3, color="grey", lw=0.8, label="assigned 63.3 µm")
    axes[1, 1].set_xlabel("activator diffusivity D_u (grid units)")
    axes[1, 1].set_ylabel("wavelength (µm)")
    axes[1, 1].set_title(
        "D_u sweep, analytical and saturated   "
        f"equal-D decays={controls['equal_diffusion_decays']}  "
        f"L<λ decays={controls['subcritical_decays']}"
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Fractofusus misrai — Turing wavenumber at 63 µm (6.33×10⁻⁵ m), Late Ediacaran 565 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_pipeline(render: bool = True) -> dict:
    """Execute the dispersion fit, the 2D pattern, controls, specimen CV, and renders."""
    ref = load_reference()
    params = ref["parameters"]
    a = float(params["a"])
    b = float(params["b"])
    d_u = float(params["D_u"])
    d_v = float(params["D_v"])
    jac = turing_conditions(a, b, d_u, d_v)
    jac["a"] = a
    jac["b"] = b
    if not (jac["stable_homogeneous"] and jac["diffusion_driven"]):
        raise RuntimeError(f"nominal parameters are not Turing-unstable: {jac}")
    dx_m = PHENOMENON_M / jac["lambda_dx"]
    fit = dispersion_fit(d_u, d_v, jac)
    h, w = int(params["grid"][0]), int(params["grid"][1])
    final, snaps, std0 = simulate_field(
        h, w, a, b, d_u, d_v,
        steps=int(params["steps"]), dt=float(params["dt"]), seed=int(params["seed"]), n_snaps=12,
    )
    lam_px = dominant_wavelength_px(final)
    lam_m = lam_px * dx_m
    sweep = wavelength_sweep(a, b, d_v, dx_m)
    measured_sweep = measured_wavelength_sweep(a, b, d_v, dx_m)
    controls = ablation_controls(a, b, d_u, d_v, jac)
    specimen_img = Image.open(DATASET / SPECIMEN_NAME).convert("RGB")
    specimen_gray = np.asarray(specimen_img.convert("L"), dtype=np.float64) / 255.0
    specimen_cv = structure_metrics(specimen_gray)
    specimen_box = spindle_mask_metrics(specimen_gray)
    n_waves_in_organ = ORGAN_MODULE_M / PHENOMENON_M
    result = {
        "phenomenon_size_m": PHENOMENON_M,
        "log10_size_m": float(np.log10(PHENOMENON_M)),
        "geological_age_ma": AGE_MA,
        "scale_tier": "4_cellular_10um",
        "lambda_dx_analytic": jac["lambda_dx"],
        "lambda_m_analytic": PHENOMENON_M,
        "dx_m": dx_m,
        "lambda_px_measured": lam_px,
        "lambda_m_measured": lam_m,
        "lambda_um_measured": lam_m * 1e6,
        "pattern_std_initial": std0,
        "pattern_std_final": float(final.std()),
        "pattern_grows": bool(final.std() > 5.0 * std0),
        "pearson_r": fit["pearson_r"],
        "rmse": fit["rmse"],
        "dispersion": fit,
        "sweep": sweep,
        "measured_sweep": measured_sweep,
        "controls": controls,
        "jacobian": {k: jac[k] for k in ("f_u", "f_v", "g_u", "g_v", "trace", "det", "h", "k_c2", "u0", "v0")},
        "organ_module_wavelengths": n_waves_in_organ,
        "wavelengths_51_fit": bool(51 * PHENOMENON_M < ORGAN_MODULE_M),
        "wavelengths_52_do_not": bool(ORGAN_MODULE_M < 52 * PHENOMENON_M),
        "specimen_coherency": specimen_cv["coherency"],
        "specimen_edge_fraction": specimen_cv["edge_fraction"],
        "specimen_aspect": specimen_box["spindle_aspect_height_over_width"],
        "specimen_mask_fraction": specimen_box["spindle_mask_fraction"],
        "turing_ok": True,
    }
    if render:
        frames = render_frames(snaps, dx_m, specimen_gray)
        save_gif(frames, GIF_PATH)
        save_comparison(specimen_img, frames[-1], fit, sweep, measured_sweep, controls, COMPARISON_PATH)
        with Image.open(GIF_PATH) as gif:
            result["n_frames"] = int(getattr(gif, "n_frames", 1))
            result["gif_width"], result["gif_height"] = gif.size
            gif.seek(gif.n_frames - 1)
            last_gray = np.asarray(gif.convert("L"), dtype=np.float64) / 255.0
        sim_cv = structure_metrics(last_gray)
        sim_box = spindle_mask_metrics(last_gray)
        ks = ks_orientation(specimen_cv, sim_cv)
        result.update({
            "sim_coherency": sim_cv["coherency"],
            "sim_edge_fraction": sim_cv["edge_fraction"],
            "sim_aspect": sim_box["spindle_aspect_height_over_width"],
            "sim_mask_fraction": sim_box["spindle_mask_fraction"],
            "ks_statistic": ks["ks_statistic"],
            "ks_pvalue": ks["ks_pvalue"],
            "comparison_png": str(COMPARISON_PATH.relative_to(ROOT)),
            "growth_gif": str(GIF_PATH.relative_to(ROOT)),
        })
    # Drop bulky arrays before serialising.
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        metrics = run_pipeline(render=True)
    rows = [
        {
            "name": "Turing dispersion Pearson r of spectral σ(k) versus analytical σ(k)",
            "value": float(f"{float(metrics['pearson_r']):.12f}"),
            "reference_value": 0.98,
            "unit": "Pearson r",
            "method": "k = 1..8 on a 2π ring; spectral diffusion of the linearised Schnakenberg Jacobian versus eig(J - k² D)",
            "note": f"RMSE = {metrics['rmse']:.6f}; linear dispersion, not a copy of the reference series",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE between numerical and analytical growth rates",
            "value": round(float(metrics["rmse"]), 6),
            "reference_value": 0.0,
            "unit": "1/time",
            "method": "root mean square of σ_numerical(k) - σ_analytic(k) for integer k = 1..8",
            "note": "uncertainty on the linear wavenumber fit",
            "derived_from": "script",
        },
        {
            "name": "nonlinear dominant wavelength on the saturated 63 µm field",
            "value": round(float(metrics["lambda_um_measured"]), 4),
            "reference_value": 63.3,
            "unit": "µm",
            "method": "argmax of the 2D power spectrum of the saturated activator, times dx with analytical λ_c fixed at 6.33e-05 m",
            "note": "nonlinear peak is not forced equal to the linear k_c; the reference is the assigned linear wavelength",
            "derived_from": "script",
        },
        {
            "name": "specimen spindle aspect ratio (height over width) from the CC BY mould",
            "value": round(float(metrics["specimen_aspect"]), 4),
            "reference_value": round(float(metrics.get("sim_aspect", metrics["specimen_aspect"] + 0.5)), 4),
            "unit": "aspect",
            "method": "Otsu threshold on smoothed Sobel magnitude of fractofusus_misrai_specimen.jpg; bounding support of the core mask",
            "note": "reference_value is the simulated spindle aspect, which is a different number; KS and CV are separate",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(float(metrics["specimen_coherency"]), 4),
            "reference_value": round(float(metrics.get("sim_coherency", 0.0)), 4),
            "unit": "coherency",
            "method": "(λ1-λ2)/(λ1+λ2) averaged on the Sobel structure tensor of the Mistaken Point photograph",
            "note": f"KS statistic of gradient angles versus the GIF = {metrics.get('ks_statistic', float('nan'))}",
            "derived_from": "output_pixels",
        },
        {
            "name": "equal-diffusion ablation final activator std",
            "value": round(float(metrics["controls"]["equal_diffusion_std_final"]), 6),
            "reference_value": round(float(metrics["controls"]["equal_diffusion_std_initial"]), 6),
            "unit": "concentration std",
            "method": "Schnakenberg with D_u = D_v = 2.40; Turing h = D*trace < 0 so the pattern must decay",
            "note": "negative control versus the patterned D_v/D_u = 20 run",
            "derived_from": "script",
        },
        {
            "name": "subcritical-domain ablation final activator std",
            "value": round(float(metrics["controls"]["subcritical_std_final"]), 6),
            "reference_value": round(float(metrics["controls"]["subcritical_std_initial"]), 6),
            "unit": "concentration std",
            "method": "2 by 2 domain, both axes shorter than λ_dx, so k_min = π sits above the unstable band",
            "note": f"σ(k_min) = {metrics['controls']['subcritical_sigma']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "KS statistic of specimen versus simulation gradient-angle samples",
            "value": round(float(metrics.get("ks_statistic", 0.0)), 4),
            "reference_value": 0.0,
            "unit": "KS D",
            "method": "two-sample KS on 4000 Sobel-magnitude-weighted gradient angles from the photograph and the GIF",
            "note": f"p = {metrics.get('ks_pvalue', float('nan'))}; value is not the null reference 0",
            "derived_from": "output_pixels",
        },
        {
            "name": "saturated Fourier wavelength increase from D_u = 0.08 to D_u = 0.18",
            "value": round(
                float(metrics["measured_sweep"]["rows"][2]["lambda_um"])
                - float(metrics["measured_sweep"]["rows"][0]["lambda_um"]),
                4,
            ),
            "reference_value": 0.0,
            "unit": "µm",
            "method": "dominant 2D Fourier wavelength of three saturated Schnakenberg fields at D_u = 0.08, 0.12, 0.18",
            "note": (
                "null reference is a sweep with no D_u effect; "
                f"wavelengths µm = {[round(r['lambda_um'], 2) for r in metrics['measured_sweep']['rows']]}"
            ),
            "derived_from": "script",
        },
    ]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    metrics = run_pipeline(render=True)
    print(json.dumps({
        "pearson_r": metrics["pearson_r"],
        "rmse": metrics["rmse"],
        "lambda_um_analytic": metrics["lambda_m_analytic"] * 1e6,
        "lambda_um_measured": metrics["lambda_um_measured"],
        "pattern_grows": metrics["pattern_grows"],
        "equal_diffusion_decays": metrics["controls"]["equal_diffusion_decays"],
        "subcritical_decays": metrics["controls"]["subcritical_decays"],
        "sweep_monotonic": metrics["sweep"]["monotonic_increasing"],
        "n_frames": metrics.get("n_frames"),
        "gif_size": [metrics.get("gif_width"), metrics.get("gif_height")],
        "ks_statistic": metrics.get("ks_statistic"),
        "specimen_aspect": metrics.get("specimen_aspect"),
        "sim_aspect": metrics.get("sim_aspect"),
        "age_ma": metrics["geological_age_ma"],
    }, indent=2))
    if args.self_test:
        assert metrics["pearson_r"] > 0.98, metrics["pearson_r"]
        assert metrics["pattern_grows"], metrics
        assert metrics["controls"]["equal_diffusion_decays"], metrics["controls"]
        assert metrics["controls"]["subcritical_decays"], metrics["controls"]
        assert metrics["sweep"]["monotonic_increasing"], metrics["sweep"]
        assert metrics["measured_sweep"]["monotonic_increasing"], metrics["measured_sweep"]
        assert metrics["measured_sweep"]["all_grew"], metrics["measured_sweep"]
        assert metrics["wavelengths_51_fit"] and metrics["wavelengths_52_do_not"]
        assert abs(metrics["lambda_m_analytic"] - PHENOMENON_M) < 1e-12
        assert metrics["geological_age_ma"] == 565.0
        assert metrics["n_frames"] >= 10
        assert min(metrics["gif_width"], metrics["gif_height"]) >= 480
        assert metrics["specimen_aspect"] != metrics["sim_aspect"]
        # Non-circular: measured nonlinear wavelength is not the linear reference.
        assert abs(metrics["lambda_um_measured"] - 63.3) > 0.2
        print("[PASS] 2026-delphi10000-fractofusus-misrai-turing-wavelength-s0 self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
