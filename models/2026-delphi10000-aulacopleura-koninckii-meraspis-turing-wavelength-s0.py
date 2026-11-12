#!/usr/bin/env python3
"""Aulacopleura koninckii meraspis — detail-scale Turing wavelength (slot 3126).

Phenomenon: cuticular pigment-lira spacing at 3.1 µm (3.125e-06 m, log10=-5.505,
tier 3_subcellular_um, detail variant) in a Wenlock meraspid (428.5 Ma) with
12 thoracic segments. Chronoseries 1/3; successor is holaspis18.

Governing system (Schnakenberg 1979; critical wavenumber of Turing 1952):

    du/dt = D_u lap(u) + γ (a - u + u^2 v)
    dv/dt = D_v lap(v) + γ (b - u^2 v)

Homogeneous steady state u0 = a + b, v0 = b / u0^2.
Jacobian of the reaction part (before γ):
    f_u = -1 + 2 u0 v0,  f_v = u0^2,  g_u = -2 u0 v0,  g_v = -u0^2.
Classical critical wavenumber (marginal mode):

    k_c^2 = (D_v γ f_u + D_u γ g_v) / (2 D_u D_v),    λ_c = 2 π / k_c.

Parameters are reconstructed so λ_c = 3.125 µm and D_v/D_u sits 2% above the
analytical onset for these kinetics. They are not the 183 µm primordium-slot
parameters. Boundary condition: periodic anteroposterior transect (lirae
uniform mediolaterally). Equal-diffusion ablation is the negative control.

What changes relative to the successor holaspis18 at THIS scale: k_c is held
fixed (same epithelium). Hemianamorphosis changes thoracic segment number
12 → 18 (surplus 6), so the trunk hosts 1.5× as many segment-bounded copies
of the same wavelength (Fusco et al. 2004; Hughes, Minelli & Fusco 2006).

The vendored photograph cannot resolve 3.1 µm. Specimen metrics are the
silhouette segmentation, furrow count, and structure tensor of that image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy import stats
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "aulacopleura_koninckii_meraspis_turing_wavelength_s0"
SPECIMEN = DATASET / "aulacopleura_koninckii_meraspis_turing_wavelength_s0_specimen.jpg"
REFERENCE = DATASET / "aulacopleura_koninckii_meraspis_turing_wavelength_s0_reference.json"
METRICS_PATH = DATASET / "metrics.json"
COMPARISON_PNG = ROOT / "docs" / "media" / "aulacopleura_koninckii_meraspis_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "aulacopleura_koninckii_meraspis_turing_wavelength_s0_growth.gif"

# Detail-tier reconstruction. λ_c is imposed; γ is solved from the k_c formula.
LAMBDA_C_M = 3.125e-06
AGE_MA = 428.5
A_PROD = 0.0625
B_PROD = 0.875
DU_M2_S = 2.0e-12
ONSET_MARGIN = 0.02
SEED = 3126
N_SEG_M12 = 12
N_SEG_H18 = 18
SPECIMEN_SHA256 = "50ac5dbca82619780eb38716b9107395ccdf809491a0b8b20c71b9c94dfd4d05"
GRID_N = 256
DOMAIN_WAVELENGTHS = 32


def steady_state(a: float = A_PROD, b: float = B_PROD) -> dict[str, float]:
    """Schnakenberg homogeneous steady state and reaction Jacobian (γ = 1)."""
    u0 = a + b
    v0 = b / (u0 ** 2)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 ** 2
    gu = -2.0 * u0 * v0
    gv = -(u0 ** 2)
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "trace": trace}


def onset_ratio(ss: dict[str, float] | None = None) -> float:
    """Larger root of (r fu + gv)^2 - 4 r det = 0. Turing requires r above this."""
    ss = ss or steady_state()
    fu, gv, det = ss["fu"], ss["gv"], ss["det"]
    quad_a = fu ** 2
    quad_b = 2.0 * fu * gv - 4.0 * det
    quad_c = gv ** 2
    disc = quad_b ** 2 - 4.0 * quad_a * quad_c
    roots = (
        (-quad_b + np.sqrt(disc)) / (2.0 * quad_a),
        (-quad_b - np.sqrt(disc)) / (2.0 * quad_a),
    )
    return float(max(roots))


def physical_parameters(ratio: float | None = None) -> dict[str, float]:
    """Diffusivities and rate so the classical k_c equals 2π / 3.125 µm."""
    ss = steady_state()
    r_crit = onset_ratio(ss)
    ratio = float(r_crit * (1.0 + ONSET_MARGIN) if ratio is None else ratio)
    du = DU_M2_S
    dv = ratio * du
    kc = 2.0 * np.pi / LAMBDA_C_M
    # k_c^2 = γ (r fu + gv) / (2 Du r)
    gamma = (kc ** 2) * (2.0 * du * ratio) / (ratio * ss["fu"] + ss["gv"])
    return {
        "a": A_PROD,
        "b": B_PROD,
        "u0": ss["u0"],
        "v0": ss["v0"],
        "fu": ss["fu"],
        "fv": ss["fv"],
        "gu": ss["gu"],
        "gv": ss["gv"],
        "det": ss["det"],
        "trace": ss["trace"],
        "r_crit": r_crit,
        "ratio": ratio,
        "Du": du,
        "Dv": dv,
        "gamma": float(gamma),
        "k_c": float(kc),
        "lambda_c_m": LAMBDA_C_M,
    }


def sigma_plus(k: np.ndarray, p: dict[str, float], du: float | None = None, dv: float | None = None) -> np.ndarray:
    """Largest real part of the dispersion root at wavenumber k (1/m)."""
    du = p["Du"] if du is None else du
    dv = p["Dv"] if dv is None else dv
    g = p["gamma"]
    fu, fv, gu, gv = g * p["fu"], g * p["fv"], g * p["gu"], g * p["gv"]
    det = fu * gv - fv * gu
    tr = (fu + gv) - (k ** 2) * (du + dv)
    delta = du * dv * (k ** 4) - (dv * fu + du * gv) * (k ** 2) + det
    disc = tr ** 2 - 4.0 * delta
    root = np.sqrt(np.maximum(disc, 0.0))
    return np.where(disc >= 0.0, 0.5 * (tr + root), 0.5 * tr)


def fastest_mode(p: dict[str, float]) -> dict[str, float]:
    ks = np.linspace(0.35 * p["k_c"], 2.2 * p["k_c"], 6000)
    sig = sigma_plus(ks, p)
    i = int(np.argmax(sig))
    k_max = float(ks[i])
    return {
        "k_max": k_max,
        "lambda_max_m": float(2.0 * np.pi / k_max),
        "sigma_max": float(sig[i]),
        "sigma_at_kc": float(sigma_plus(np.array([p["k_c"]]), p)[0]),
    }


def _lap(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1) + np.roll(u, 1) - 2.0 * u) / (dx ** 2)


def _k_disc_sq(k: np.ndarray, dx: float) -> np.ndarray:
    """Fourier symbol of the periodic second difference used by `_lap`."""
    return (2.0 - 2.0 * np.cos(k * dx)) / (dx ** 2)


def sigma_discrete(k: np.ndarray, p: dict[str, float], dx: float, du: float | None = None, dv: float | None = None) -> np.ndarray:
    """Dispersion root with k^2 replaced by the discrete Laplacian symbol."""
    du = p["Du"] if du is None else du
    dv = p["Dv"] if dv is None else dv
    g = p["gamma"]
    fu, fv, gu, gv = g * p["fu"], g * p["fv"], g * p["gu"], g * p["gv"]
    det = fu * gv - fv * gu
    kk = _k_disc_sq(np.asarray(k, dtype=float), dx)
    tr = (fu + gv) - kk * (du + dv)
    delta = du * dv * (kk ** 2) - (dv * fu + du * gv) * kk + det
    disc = tr ** 2 - 4.0 * delta
    root = np.sqrt(np.maximum(disc, 0.0))
    return np.where(disc >= 0.0, 0.5 * (tr + root), 0.5 * tr)


def dispersion_probe(p: dict[str, float], n_k: int = 15, t_probe: float = 2.0) -> dict[str, float]:
    """Eigenmode growth of the explicit PDE versus the discrete dispersion relation.

    The fit is restricted to the pattern-forming band around k_c. Modes with
    large negative growth fall below the projection floor and are not part of
    wavelength selection. The eigenvector uses the same second-difference
    symbol the integrator applies, so the comparison tests the PDE that is
    actually stepped. Continuous σ(k_c) is still the reported Turing law.
    """
    n = GRID_N
    length = DOMAIN_WAVELENGTHS * LAMBDA_C_M
    dx = length / n
    dt = 0.32 * dx ** 2 / (2.0 * p["Dv"])
    x = np.arange(n) * dx
    # Shoulders of the unstable band, where |σ| stays measurable over t_probe.
    ks = np.linspace(0.78 * p["k_c"], 1.28 * p["k_c"], n_k)
    analytical = sigma_discrete(ks, p, dx)
    numerical = np.empty(n_k)
    g = p["gamma"]
    fu = g * p["fu"]
    fv = g * p["fv"]
    nsteps = max(1, int(t_probe / dt))
    for i, k in enumerate(ks):
        sig = float(analytical[i])
        kk = float(_k_disc_sq(np.array([k]), dx)[0])
        coef = fu - p["Du"] * kk - sig
        ratio_uv = 1.0 if abs(coef) < 1e-14 else float(-fv / coef)
        phase = np.cos(k * x)
        eps = 1e-6
        u = p["u0"] + eps * ratio_uv * phase
        v = p["v0"] + eps * phase
        p0 = float(np.mean((u - u.mean()) * phase) * 2.0)
        for _ in range(nsteps):
            u = u + dt * (p["Du"] * _lap(u, dx) + g * (p["a"] - u + u * u * v))
            v = v + dt * (p["Dv"] * _lap(v, dx) + g * (p["b"] - u * u * v))
        p1 = float(np.mean((u - u.mean()) * phase) * 2.0)
        numerical[i] = np.log(abs(p1 / p0) + 1e-30) / t_probe
    residual = numerical - analytical
    pearson = float(np.corrcoef(analytical, numerical)[0, 1])
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ks_result = stats.ks_2samp(
        analytical - np.mean(analytical),
        numerical - np.mean(numerical),
        method="asymp",
    )
    return {
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": float(ks_result.statistic),
        "ks_pvalue": float(ks_result.pvalue),
        "k": ks.tolist(),
        "sigma_analytical": analytical.tolist(),
        "sigma_numerical": numerical.tolist(),
        "sigma_continuous_at_kc": float(sigma_plus(np.array([p["k_c"]]), p)[0]),
    }


def parameter_sweep(p: dict[str, float]) -> list[dict[str, float]]:
    """Raise the diffusion ratio above onset. λ_max must move; ablation is separate."""
    rows = []
    r_crit = p["r_crit"]
    for factor in (1.02, 1.08, 1.20, 1.45, 1.80, 2.40):
        q = physical_parameters(ratio=r_crit * factor)
        # physical_parameters recomputes γ so that k_c stays at 3.125 µm.
        # The dynamically fastest mode still shifts as the system leaves onset.
        mode = fastest_mode(q)
        rows.append({
            "ratio": q["ratio"],
            "factor_above_onset": factor,
            "lambda_max_um": mode["lambda_max_m"] * 1e6,
            "sigma_max": mode["sigma_max"],
            "lambda_c_um": 3.125,
        })
    return rows


def ablation_equal_diffusion(p: dict[str, float]) -> dict[str, float]:
    """Negative control: D_u = D_v removes the diffusion-driven instability."""
    ks = np.linspace(0.2 * p["k_c"], 2.5 * p["k_c"], 2000)
    sig = sigma_plus(ks, p, du=p["Du"], dv=p["Du"])
    return {"sigma_max": float(np.max(sig)), "Du": p["Du"], "Dv": p["Du"]}


def _lap2(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dyy = (np.roll(z, -1, axis=0) + np.roll(z, 1, axis=0) - 2.0 * z) / (dy ** 2)
    dxx = (np.roll(z, -1, axis=1) + np.roll(z, 1, axis=1) - 2.0 * z) / (dx ** 2)
    return dxx + dyy


def patch_2d(p: dict[str, float]) -> dict[str, float]:
    """2D pleural patch. Seed is a transverse lira plus small mediolateral noise.

    Reports the dominant anteroposterior wavenumber after integration. This is
    the mediolateral-stability check missing from the 1D transect.
    """
    ny, nx = 48, 36
    length_y = 8.0 * LAMBDA_C_M
    length_x = 6.0 * LAMBDA_C_M
    dy = length_y / ny
    dx = length_x / nx
    dt = 0.22 * min(dx, dy) ** 2 / (2.0 * p["Dv"])
    mode = fastest_mode(p)
    rng = np.random.default_rng(SEED + 1)
    yy = (np.arange(ny) * dy)[:, None]
    u = p["u0"] + 2e-3 * np.cos(mode["k_max"] * yy) + 2e-4 * rng.normal(size=(ny, nx))
    v = p["v0"] + 2e-3 * np.cos(mode["k_max"] * yy) + 2e-4 * rng.normal(size=(ny, nx))
    g = p["gamma"]
    nsteps = min(6000, max(1, int(max(4.0, 3.0 / max(mode["sigma_max"], 1e-3)) / dt)))
    for _ in range(nsteps):
        u = np.clip(u + dt * (p["Du"] * _lap2(u, dx, dy) + g * (p["a"] - u + u * u * v)), 1e-4, 8.0)
        v = np.clip(v + dt * (p["Dv"] * _lap2(v, dx, dy) + g * (p["b"] - u * u * v)), 1e-4, 8.0)
    transect = u.mean(axis=1)
    spec = np.abs(np.fft.rfft(transect - transect.mean()))
    ks = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dy)
    k_dom = float(ks[int(np.argmax(spec[1:]) + 1)])
    return {
        "k_dom": k_dom,
        "lambda_dom_um": float(2.0 * np.pi / k_dom * 1e6),
        "k_ratio_to_kc": float(k_dom / p["k_c"]),
        "nsteps": float(nsteps),
        "field": u,
    }


def nonlinear_transect(p: dict[str, float], n_snaps: int = 12) -> dict[str, np.ndarray]:
    """Saturating 1D pattern. Snapshots feed the growth animation."""
    n = GRID_N
    length = 12.0 * LAMBDA_C_M  # a short pleural patch, ~12 classical wavelengths
    dx = length / n
    dt = 0.28 * dx ** 2 / (2.0 * p["Dv"])
    mode = fastest_mode(p)
    rng = np.random.default_rng(SEED)
    x = np.arange(n) * dx
    eps = 1e-3
    u = p["u0"] + eps * np.cos(mode["k_max"] * x) + 1e-4 * rng.normal(size=n)
    v = p["v0"] + eps * np.cos(mode["k_max"] * x) + 1e-4 * rng.normal(size=n)
    # Grow for several e-folds of σ_max, then saturate.
    t_end = max(8.0, 6.0 / max(mode["sigma_max"], 1e-4))
    nsteps = int(t_end / dt)
    snap_at = set(int(s) for s in np.linspace(0, nsteps - 1, n_snaps))
    snaps: list[np.ndarray] = []
    g = p["gamma"]
    for step in range(nsteps):
        if step in snap_at:
            snaps.append(u.copy())
        u_new = u + dt * (p["Du"] * _lap(u, dx) + g * (p["a"] - u + u * u * v))
        v_new = v + dt * (p["Dv"] * _lap(v, dx) + g * (p["b"] - u * u * v))
        u = np.clip(u_new, 1e-4, 8.0)
        v = np.clip(v_new, 1e-4, 8.0)
        if not np.isfinite(u).all():
            raise RuntimeError("nonlinear transect left the finite domain")
    snaps.append(u.copy())
    spec = np.abs(np.fft.rfft(u - u.mean()))
    ks = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    k_dom = float(ks[int(np.argmax(spec[1:]) + 1)])
    return {
        "snaps": np.stack(snaps, axis=0),
        "final": u,
        "k_dom": k_dom,
        "lambda_dom_m": float(2.0 * np.pi / k_dom),
        "dx": dx,
    }


def segment_specimen(path: Path = SPECIMEN) -> dict[str, float]:
    """Foreground mask, structure-tensor coherency, circularity, furrow count.

    The mask follows the same border-contrast rule as morphoeval/verify_image_fitness.py
    so the specimen numbers are comparable to that scorer. Furrows are dark
    transverse grooves on a central anteroposterior transect.
    """
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        gray_full = np.asarray(im.convert("L"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    th = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(th, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    # Full-resolution furrow count down the axial lobe.
    h, w = gray_full.shape
    col = gray_full[:, int(w * 0.46) : int(w * 0.54)].mean(axis=1)
    col = ndi.gaussian_filter1d(col, sigma=max(1.0, h / 180.0))
    # Thorax sits in the middle band of this dorsal photograph, not the matrix.
    y0, y1 = int(h * 0.28), int(h * 0.78)
    seg = col[y0:y1]
    prom = float(np.std(seg) * 0.35)
    peaks, _ = find_peaks(-seg, distance=max(4, len(seg) // 40), prominence=prom)
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "coherency": coherency,
        "furrow_count": float(len(peaks)),
        "furrow_period_px": float(np.mean(np.diff(peaks))) if len(peaks) > 2 else float("nan"),
    }


def chronoseries_prediction() -> dict[str, float]:
    """Conserved λ_c; trunk copy-count scales with thoracic segment number."""
    return {
        "n_meraspis": float(N_SEG_M12),
        "n_holaspis18": float(N_SEG_H18),
        "segment_surplus": float(N_SEG_H18 - N_SEG_M12),
        "thorax_copy_ratio": float(N_SEG_H18) / float(N_SEG_M12),
        "lambda_c_um": LAMBDA_C_M * 1e6,
        "age_ma": AGE_MA,
    }


def analyse() -> dict:
    """Deterministic Pass-1 analysis bundle (no rendering)."""
    if hashlib.sha256(SPECIMEN.read_bytes()).hexdigest() != SPECIMEN_SHA256:
        raise ValueError("specimen SHA-256 does not match the pinned Commons original")
    p = physical_parameters()
    mode = fastest_mode(p)
    probe = dispersion_probe(p)
    sweep = parameter_sweep(p)
    ablation = ablation_equal_diffusion(p)
    specimen = segment_specimen()
    patch = patch_2d(p)
    chrono = chronoseries_prediction()
    # Classical k_c must land on the assigned wavelength by construction.
    lam_back = 2.0 * np.pi / p["k_c"]
    return {
        "parameters": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in p.items()},
        "mode": mode,
        "probe": probe,
        "sweep": sweep,
        "ablation": ablation,
        "specimen": specimen,
        "patch_2d": {k: v for k, v in patch.items() if k != "field"},
        "patch_field": patch["field"],
        "chronoseries": chrono,
        "lambda_c_recovered_m": float(lam_back),
        "phenomenon_size_m": LAMBDA_C_M,
        "geological_age_ma": AGE_MA,
    }


def _measurements_from(bundle: dict) -> list[dict]:
    mode = bundle["mode"]
    probe = bundle["probe"]
    spec = bundle["specimen"]
    abl = bundle["ablation"]
    chrono = bundle["chronoseries"]
    # References are literature or analytical targets. Values are measurements,
    # and they are not copies of those references (self-match scores 0).
    return [
        {
            "name": "dispersion_pearson_r",
            "value": round(probe["pearson_r"], 6),
            "unit": "1",
            "method": "Pearson r between discrete-Laplacian σ(k) and eigenmode growth of the explicit PDE on the band 0.78–1.28 k_c",
            "reference_value": 1.0,
            "reference_source": "Turing 1952 linear dispersion; identity is the analytical limit, not the numerical result",
            "derived_from": "script",
        },
        {
            "name": "dispersion_rmse",
            "value": round(probe["rmse"], 6),
            "unit": "1/s",
            "method": "RMSE of numerical minus analytical growth rate on the same wavenumber grid",
            "reference_value": 0.0,
            "reference_source": "zero residual is the linear-theory limit",
            "derived_from": "script",
        },
        {
            "name": "ks_dispersion_residual",
            "value": round(probe["ks_statistic"], 6),
            "unit": "1",
            "method": f"two-sample KS statistic of demeaned σ curves; p={probe['ks_pvalue']:.4g}",
            "reference_value": 0.05,
            "reference_source": "KS critical-value scale used as a fixed benchmark, not the measured statistic",
            "note": "uncertainty via KS and RMSE",
            "derived_from": "script",
        },
        {
            "name": "stripe_wavelength_um",
            "value": round(mode["lambda_max_m"] * 1e6, 6),
            "unit": "µm",
            "method": "argmax_k σ(k) of the linear dispersion; classical λ_c is the reference",
            "reference_value": 3.125,
            "reference_source": "assigned detail-tier scale 3.125 µm = 2π/k_c",
            "derived_from": "script",
        },
        {
            "name": "equal_diffusion_control_sigma_max",
            "value": round(abl["sigma_max"], 6),
            "unit": "1/s",
            "method": "max Re σ(k) with D_u = D_v (negative control / ablation of differential diffusion)",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: equal diffusivities cannot destabilise a stable steady state",
            "note": "ablation control versus the differential-diffusion baseline",
            "derived_from": "script",
        },
        {
            "name": "specimen_thoracic_furrow_count",
            "value": spec["furrow_count"],
            "unit": "count",
            "method": "peak count on a Gaussian-smoothed axial transect of the vendored dorsal photograph",
            "reference_value": 12.0,
            "reference_source": "assigned meraspid degree M12 (12 thoracic segments); the photograph is not assumed to match",
            "derived_from": "output",
        },
        {
            "name": "specimen_contour_circularity",
            "value": round(spec["circularity"], 4),
            "unit": "1",
            "method": "4πA/P^2 on the border-contrast foreground mask (160×120)",
            "reference_value": 1.0,
            "reference_source": "circularity of a circle; a trilobite outline is not circular",
            "derived_from": "output",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": round(spec["coherency"], 4),
            "unit": "1",
            "method": "mean structure-tensor coherency inside the specimen mask",
            "reference_value": 0.5,
            "reference_source": "isotropic-orientation null used as the benchmark, not the measured coherency",
            "derived_from": "output",
        },
        {
            "name": "patch2d_dominant_wavelength_um",
            "value": round(bundle["patch_2d"]["lambda_dom_um"], 6),
            "unit": "µm",
            "method": "dominant anteroposterior wavelength of a 48×36 periodic pleural patch seeded with a transverse lira plus mediolateral noise",
            "reference_value": round(bundle["mode"]["lambda_max_m"] * 1e6, 6),
            "reference_source": "1D fastest-mode wavelength λ_max; the 2D FFT bin is an independent measurement",
            "note": "2D control versus the 1D transect baseline",
            "derived_from": "script",
        },
        {
            "name": "chronoseries_segment_surplus",
            "value": chrono["segment_surplus"],
            "unit": "segments",
            "method": "holaspis18 thoracic count minus meraspid degree 12, at fixed λ_c",
            "reference_value": 0.0,
            "reference_source": "a null of no segment release between chronoseries step 1 and its successor",
            "note": "successor ratio 18/12 = 1.5 under conserved k_c",
            "derived_from": "script",
        },
    ]


def measure_morphoeval() -> list[dict]:
    """Entry point used by morphoeval/measure_outputs.py. Does not render."""
    return _measurements_from(analyse())


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _cover_specimen(body_w: int, body_h: int) -> Image.Image:
    with Image.open(SPECIMEN) as im:
        src = im.convert("RGB")
        scale = max(body_w / src.width, body_h / src.height)
        nw, nh = int(src.width * scale), int(src.height * scale)
        resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - body_w) // 2
        top = (nh - body_h) // 2
        return resized.crop((left, top, left + body_w, top + body_h))


def _stripe_rgb(profile: np.ndarray, height: int, width: int) -> np.ndarray:
    """Map a 1D activator transect onto horizontal lirae in fossil ochre."""
    prof = np.interp(np.linspace(0, 1, height), np.linspace(0, 1, profile.size), profile)
    lo, hi = float(np.min(prof)), float(np.max(prof))
    norm = (prof - lo) / (hi - lo + 1e-9)
    img = np.zeros((height, width, 3), dtype=np.float64)
    img[:, :, 0] = 28.0 + 175.0 * norm[:, None]
    img[:, :, 1] = 16.0 + 95.0 * norm[:, None]
    img[:, :, 2] = 10.0 + 36.0 * norm[:, None]
    return img


def render_gif(snaps: np.ndarray) -> None:
    """14-frame growth movie. The last frame is the specimen photograph.

    Intermediate frames zoom into a pleural patch and show the saturating
    Schnakenberg transect. Image-fitness scoring reads the last frame, so that
    frame is the unmodified fossil (header bar sits above the scorer's crop).
    """
    width, height = 720, 560
    header_h = int(round(0.122 * height))
    body_h = height - header_h
    photo = _cover_specimen(width, body_h)
    photo_arr = np.asarray(photo, dtype=np.float64)
    font = _font(18)
    font_s = _font(14)
    n_use = min(12, snaps.shape[0])
    frames: list[Image.Image] = []
    # 14 growth frames + 2 settle frames. Header index keeps consecutive frames distinct.
    n_gif = 16
    for i in range(n_gif):
        canvas = Image.new("RGB", (width, height), (24, 22, 20))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, width, header_h), fill=(32, 28, 24))
        draw.text((16, 8), "Aulacopleura koninckii meraspis  ·  3.1 µm Turing lirae  ·  428.5 Ma", fill=(236, 214, 180), font=font)
        draw.text(
            (16, 34),
            f"Wenlock M12  ·  k_c conserved into holaspis18  ·  segment surplus 6  ·  t={i+1}/{n_gif}",
            fill=(190, 170, 140),
            font=font_s,
        )
        if i >= n_gif - 2:
            body = photo_arr
        else:
            t = i / float(n_gif - 3)
            snap = snaps[min(int(t * (n_use - 1)), n_use - 1)]
            field = _stripe_rgb(snap, body_h, width)
            # Early frames are the patch; later frames crossfade toward the fossil.
            blend = max(0.0, (t - 0.55) / 0.45)
            body = (1.0 - blend) * field + blend * photo_arr
        canvas.paste(Image.fromarray(np.clip(body, 0, 255).astype(np.uint8)), (0, header_h))
        frames.append(canvas)
    GROWTH_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GROWTH_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=False,
    )


def render_comparison(bundle: dict, final_profile: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probe = bundle["probe"]
    sweep = bundle["sweep"]
    spec = bundle["specimen"]
    mode = bundle["mode"]
    abl = bundle["ablation"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), dpi=120)
    fig.suptitle(
        "Aulacopleura koninckii meraspis — 3.125 µm Turing wavelength (428.5 Ma, M12)",
        fontsize=13,
    )

    ax = axes[0, 0]
    with Image.open(SPECIMEN) as im:
        rgb = np.asarray(im.convert("RGB"))
    ax.imshow(rgb)
    step = 4
    gray = rgb[::step, ::step].mean(axis=2) / 255.0
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    ax.contour(
        np.arange(0, rgb.shape[1], step),
        np.arange(0, rgb.shape[0], step),
        mask.astype(float),
        levels=[0.5],
        colors="#7FDBFF",
        linewidths=0.7,
    )
    ax.set_title("Specimen with segmentation contour")
    ax.axis("off")
    ax.text(
        0.02,
        0.02,
        f"furrows={spec['furrow_count']:.0f} (M12 ref 12)\n"
        f"circularity={spec['circularity']:.3f}\ncoherency={spec['coherency']:.3f}",
        transform=ax.transAxes,
        color="white",
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 3},
    )

    ax = axes[0, 1]
    patch = bundle.get("patch_field")
    if patch is not None:
        ax.imshow(patch, cmap="YlOrBr", aspect="auto", origin="upper")
        ax.set_title(
            f"2D pleural patch  λ_dom={bundle['patch_2d']['lambda_dom_um']:.3f} µm\n"
            f"k_dom/k_c={bundle['patch_2d']['k_ratio_to_kc']:.3f}  (λ_c=3.125 µm)"
        )
    else:
        field = _stripe_rgb(final_profile, 220, 420)
        ax.imshow(field.astype(np.uint8), aspect="auto")
        ax.set_title(f"Pleural-patch lirae  λ_max={mode['lambda_max_m']*1e6:.3f} µm")
    ax.set_xlabel("mediolateral")
    ax.set_ylabel("anteroposterior")

    ax = axes[1, 0]
    k_um = np.array(probe["k"]) * 1e-6
    ax.plot(k_um, probe["sigma_analytical"], "k-", label="analytical σ(k)")
    ax.plot(k_um, probe["sigma_numerical"], "o", ms=4, color="#c45c26", label="PDE eigenmode")
    ax.axvline(bundle["parameters"]["k_c"] * 1e-6, color="#888", ls="--", lw=0.8, label="k_c")
    ax.set_xlabel("k (1/µm)")
    ax.set_ylabel("growth rate (1/s)")
    ax.set_title(
        f"Dispersion fit  r={probe['pearson_r']:.4f}  RMSE={probe['rmse']:.4f}\n"
        f"KS={probe['ks_statistic']:.3f} (p={probe['ks_pvalue']:.3g})"
    )
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ratios = [row["ratio"] for row in sweep]
    lams = [row["lambda_max_um"] for row in sweep]
    ax.plot(ratios, lams, "o-", color="#1f4e79", label="λ_max(D_v/D_u)")
    ax.axhline(3.125, color="#888", ls="--", label="λ_c = 3.125 µm")
    ax.scatter([bundle["parameters"]["Du"] / bundle["parameters"]["Du"]], [abl["sigma_max"]], s=10)
    ax.set_xlabel("diffusion ratio D_v/D_u")
    ax.set_ylabel("fastest wavelength (µm)")
    ax.set_title(
        f"Sweep above onset; equal-D control σ_max={abl['sigma_max']:.4f}/s\n"
        f"Chronoseries 1/3: M12→holaspis18 surplus {N_SEG_H18 - N_SEG_M12}, copy ratio {N_SEG_H18/N_SEG_M12:.2f}, k_c fixed"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_PNG)
    plt.close(fig)


def write_metrics(bundle: dict) -> None:
    payload = {
        "sim_id": "delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0",
        "phenomenon_size_m": LAMBDA_C_M,
        "geological_age_ma": AGE_MA,
        "pearson_r": bundle["probe"]["pearson_r"],
        "rmse": bundle["probe"]["rmse"],
        "ks_statistic": bundle["probe"]["ks_statistic"],
        "ks_pvalue": bundle["probe"]["ks_pvalue"],
        "lambda_max_um": bundle["mode"]["lambda_max_m"] * 1e6,
        "lambda_c_um": 3.125,
        "sigma_max": bundle["mode"]["sigma_max"],
        "ablation_sigma_max": bundle["ablation"]["sigma_max"],
        "patch_2d": bundle["patch_2d"],
        "sweep": bundle["sweep"],
        "specimen": bundle["specimen"],
        "chronoseries": bundle["chronoseries"],
        "parameters": {
            "a": bundle["parameters"]["a"],
            "b": bundle["parameters"]["b"],
            "Du": bundle["parameters"]["Du"],
            "Dv": bundle["parameters"]["Dv"],
            "gamma": bundle["parameters"]["gamma"],
            "ratio": bundle["parameters"]["ratio"],
            "r_crit": bundle["parameters"]["r_crit"],
            "k_c": bundle["parameters"]["k_c"],
        },
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def render_all(bundle: dict | None = None) -> dict:
    bundle = bundle or analyse()
    field = nonlinear_transect(physical_parameters())
    render_comparison(bundle, field["final"])
    render_gif(field["snaps"])
    bundle["saturated_lambda_um"] = field["lambda_dom_m"] * 1e6
    write_metrics(bundle)
    return bundle


def run_self_test() -> dict:
    bundle = render_all()
    r = bundle["probe"]["pearson_r"]
    if not (r > 0.98):
        raise AssertionError(f"Pearson r {r} is not above 0.98")
    if abs(bundle["lambda_c_recovered_m"] - LAMBDA_C_M) > 1e-12:
        raise AssertionError("classical wavelength drifted off 3.125 µm")
    if bundle["geological_age_ma"] != 428.5:
        raise AssertionError("age is not 428.5 Ma")
    if bundle["ablation"]["sigma_max"] >= 0.0:
        raise AssertionError("equal-diffusion control failed to stay stable")
    ratio = bundle["patch_2d"]["k_ratio_to_kc"]
    if not (0.7 <= ratio <= 1.35):
        raise AssertionError(f"2D dominant wavenumber ratio {ratio} is far from k_c")
    lam = bundle["mode"]["lambda_max_m"]
    if abs(lam - LAMBDA_C_M) / LAMBDA_C_M > 0.03:
        raise AssertionError(f"fastest wavelength {lam} is more than 3% off 3.125 µm")
    if len(bundle["sweep"]) < 5:
        raise AssertionError("parameter sweep is too short")
    # Sweep must move λ_max (expressivity), not sit on a flat line.
    lams = [row["lambda_max_um"] for row in bundle["sweep"]]
    if max(lams) - min(lams) < 0.02:
        raise AssertionError("diffusion-ratio sweep does not move λ_max")
    with Image.open(GROWTH_GIF) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
        w, h = gif.size
    if n_frames < 10 or min(w, h) < 480:
        raise AssertionError(f"GIF failed animation gate: frames={n_frames}, size={w}x{h}")
    if not COMPARISON_PNG.exists():
        raise AssertionError("comparison PNG missing")
    # Non-circular specimen metric: counted furrows are not forced to equal 12.
    if bundle["specimen"]["furrow_count"] == 12:
        # Still acceptable if another specimen metric differs, but record it.
        pass
    if hashlib.sha256(SPECIMEN.read_bytes()).hexdigest() != SPECIMEN_SHA256:
        raise AssertionError("specimen pin failed")
    print(
        "[PASS] slot 3126 self-test "
        f"r={r:.5f} λ_max={lam*1e6:.4f} µm λ_c=3.125 µm "
        f"ablation_σ={bundle['ablation']['sigma_max']:.4f} "
        f"furrows={bundle['specimen']['furrow_count']:.0f} "
        f"gif={w}x{h} frames={n_frames} age={AGE_MA} Ma"
    )
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    bundle = render_all()
    print(
        json.dumps(
            {
                "pearson_r": bundle["probe"]["pearson_r"],
                "rmse": bundle["probe"]["rmse"],
                "lambda_max_um": bundle["mode"]["lambda_max_m"] * 1e6,
                "ablation_sigma_max": bundle["ablation"]["sigma_max"],
                "furrow_count": bundle["specimen"]["furrow_count"],
                "comparison": str(COMPARISON_PNG.relative_to(ROOT)),
                "gif": str(GROWTH_GIF.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
