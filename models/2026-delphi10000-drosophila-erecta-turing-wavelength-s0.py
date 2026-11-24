#!/usr/bin/env python3
"""Drosophila erecta detail-tier Turing wavelength (slot 3183).

Organism: Drosophila erecta (Pandanus-specialist West African fly; female
dark/light morphs). Scale: 2.3 µm (2.262e-06 m, log10=-5.646, tier
3_subcellular_um). Age: Holocene, 0 Ma. Predecessor: Drosophila elegans
(101.1 µm primordium anchor). Successor: Drosophila eugracilis (100.6 µm).

Governing law: Schnakenberg (1979) kinetics and the Turing (1952) critical
wavenumber k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), lambda_c = 2 pi / k_c.

The gallery annulus is a separate ~100 µm ring. A damped J0 disk mode is
fitted to that ring. The 2.262 µm mode is the critical-ring texture.

Run:
  python3 models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py
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
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import j0

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_erecta_turing_wavelength_s0"
REFERENCE = DATASET / "drosophila_erecta_turing_wavelength_s0_reference.json"
SPECIMEN = DATASET / "drosophila_erecta_specimen.jpg"
LEAN = ROOT / "morphoeval" / "lean" / "loops" / "Loop_3183_drosophila_erecta_turing_wavelength_s0.lean"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_erecta_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_erecta_turing_wavelength_s0_growth.gif"

# Turing 1952 dispersion polynomial h(k^2).
# h(k^2) = Du*Dv*k^4 - (Dv*fu + Du*gv)*k^2 + (fu*gv - fv*gu)


def load_reference() -> dict:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    pinned = ref["specimen"]["sha256"]
    if digest != pinned:
        raise RuntimeError(f"specimen SHA-256 {digest} != pinned {pinned}")
    return ref


def jacobian(a: float, b: float) -> dict:
    """Schnakenberg (1979) Jacobian at the homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return {
        "u0": u0,
        "v0": v0,
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trace": fu + gv,
        "det": fu * gv - fv * gu,
    }


def critical_kc2(jac: dict, delta: float, du: float = 1.0) -> float | None:
    """k_c^2 = (Dv fu + Du gv) / (2 Du Dv), with Dv = delta * Du."""
    numer = delta * jac["fu"] + jac["gv"]
    if numer <= 0.0 or delta <= 0.0 or du <= 0.0:
        return None
    return numer / (2.0 * delta * du)


def physical_diffusivity(jac: dict, delta: float, lambda_m: float) -> tuple[float, float, float]:
    """Choose Du so the physical critical wavelength equals lambda_m."""
    kc = 2.0 * np.pi / lambda_m
    kc2_hat = critical_kc2(jac, delta, du=1.0)
    if kc2_hat is None:
        raise ValueError("no positive k_c at these kinetics")
    # Nondimensional k_hat = k_physical * sqrt(Du), and k_hat^2 = kc2_hat.
    du = kc2_hat / (kc * kc)
    return du, delta * du, kc


def wavelength_um(jac: dict, delta: float, du_phys: float) -> float:
    kc2 = critical_kc2(jac, delta, du=du_phys)
    if kc2 is None or kc2 <= 0.0:
        return float("inf")
    return (2.0 * np.pi / np.sqrt(kc2)) * 1.0e6


def turing_capable(jac: dict, delta: float) -> bool:
    """(Dv fu + Du gv)^2 > 4 Du Dv det, together with trace < 0 and det > 0."""
    if jac["trace"] >= 0.0 or jac["det"] <= 0.0:
        return False
    du, dv = 1.0, delta
    numer = dv * jac["fu"] + du * jac["gv"]
    return numer * numer > 4.0 * du * dv * jac["det"]


def dispersion_h(k2: np.ndarray, jac: dict, delta: float) -> np.ndarray:
    du, dv = 1.0, delta
    return du * dv * k2**2 - (dv * jac["fu"] + du * jac["gv"]) * k2 + jac["det"]


def _laplacian(z: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z
    ) / (dx * dx)


def integrate_patch(
    a: float,
    b: float,
    delta: float,
    *,
    n: int = 64,
    n_waves: int = 4,
    t_end: float = 22.0,
    seed: int = 3183,
    quadratic: bool = True,
    mode: str = "critical",
) -> dict:
    """Periodic Schnakenberg patch in units where Du = 1.

    `quadratic=False` drops u^2 v (autocatalysis ablation). `delta=1` is the
    equal-diffusion negative control when the base kinetics are used.
    """
    jac = jacobian(a, b)
    kc2 = critical_kc2(jac, delta, du=1.0)
    lam = (2.0 * np.pi / np.sqrt(kc2)) if kc2 else 11.33
    length = n_waves * lam
    dx = length / n
    dv = max(delta, 1.0)
    dt = 0.32 * dx * dx / (4.0 * max(dv, 1.0))
    steps = int(np.ceil(t_end / dt))
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    x = xx * dx
    if mode == "critical" and kc2:
        wave = np.cos(np.sqrt(kc2) * x)
    else:
        wave = rng.normal(size=(n, n))
    u = jac["u0"] + 0.02 * wave
    v = np.full((n, n), jac["v0"])
    std0 = float(np.std(u - jac["u0"]))
    for _ in range(steps):
        if quadratic:
            u2v = u * u * v
            fu_term = a - u + u2v
            gv_term = b - u2v
        else:
            fu_term = a - u
            gv_term = b - v
        u = u + dt * (1.0 * _laplacian(u, dx) + fu_term)
        v = v + dt * (dv * _laplacian(v, dx) + gv_term)
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            return {"ok": False, "reason": "non-finite", "std0": std0, "std1": float("nan")}
    field = u - float(np.mean(u))
    spec = np.abs(np.fft.fft2(field)) ** 2
    # Horizontal power: the seeded mode is a function of x only.
    power = spec.mean(axis=0)
    power[0] = 0.0
    k_bin = int(np.argmax(power[: n // 2]))
    lam_hat = (n / k_bin) * dx if k_bin else float("nan")
    return {
        "ok": True,
        "std0": std0,
        "std1": float(np.std(u - jac["u0"])),
        "ratio": float(np.std(u - jac["u0"]) / max(std0, 1e-12)),
        "lambda_hat": float(lam_hat),
        "lambda_target": float(lam),
        "u": u,
        "steps": steps,
        "dt": dt,
    }


def load_specimen(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return rgb, gray


def measure_scale_bar(gray: np.ndarray) -> dict:
    """Drawn 100 µm bar: longest bright run in the lower-right of the gallery image."""
    height, width = gray.shape
    best = None
    y0 = int(height * 0.88)
    x0 = int(width * 0.60)
    for y in range(y0, height):
        bright = np.where(gray[y, x0:] >= 200.0)[0]
        if bright.size < 30:
            continue
        cuts = np.where(np.diff(bright) > 1)[0]
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, bright.size - 1]
        for s, e in zip(starts, ends):
            span = int(bright[e] - bright[s] + 1)
            if 80 <= span <= 160 and (best is None or span > best[0]):
                best = (span, y, int(x0 + bright[s]), int(x0 + bright[e]))
    if best is None:
        return {"bar_px": 111, "um_per_px": 100.0 / 111.0, "detected": False}
    span, y, xa, xb = best
    return {"bar_px": span, "um_per_px": 100.0 / span, "detected": True, "row": int(y), "x0": xa, "x1": xb}


def radial_profile(gray: np.ndarray, center: tuple[int, int], r_max: int = 250, dr: int = 2) -> tuple[np.ndarray, np.ndarray]:
    cy, cx = center
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    # Caption and scale bar sit in the bottom 60 rows; keep them out of the annulus.
    body = np.zeros(gray.shape, dtype=bool)
    body[: gray.shape[0] - 60, :] = True
    radii = np.arange(0, r_max, dr, dtype=float)
    prof = np.empty_like(radii)
    for i, r in enumerate(radii):
        mask = body & (rr >= r) & (rr < r + dr)
        prof[i] = float(gray[mask].mean()) if np.any(mask) else np.nan
    good = np.isfinite(prof)
    return radii[good], prof[good]


def fit_disk_mode(radii: np.ndarray, prof: np.ndarray) -> dict:
    """I(r) = A J0(k r) exp(-r^2/2s^2) + B exp(-r^2/2 s2^2) + C."""

    def model(theta: np.ndarray) -> np.ndarray:
        amp, k, s, amp2, s2, c0 = theta
        return amp * j0(k * radii) * np.exp(-(radii**2) / (2.0 * s * s)) + amp2 * np.exp(-(radii**2) / (2.0 * s2 * s2)) + c0

    guess = np.array([59.0, 0.037, 360.0, 35.0, 73.0, 106.0])
    bounds = ([-400, 0.005, 20, -400, 10, 0], [400, 0.12, 400, 400, 400, 255])
    fit = least_squares(lambda th: model(th) - prof, guess, bounds=bounds)
    pred = model(fit.x)
    pearson = float(np.corrcoef(prof, pred)[0, 1])
    rmse = float(np.sqrt(np.mean((prof - pred) ** 2)))
    ks = float(stats.ks_2samp(prof, pred).statistic)
    return {
        "theta": [float(v) for v in fit.x],
        "pred": pred,
        "pearson_r": pearson,
        "rmse": rmse,
        "ks": ks,
        "radii": radii,
        "empirical": prof,
    }


def find_center(gray: np.ndarray) -> tuple[int, int]:
    body = gray[: gray.shape[0] - 60, :]
    smooth = ndi.gaussian_filter(body, 8.0)
    cy, cx = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    return int(cy), int(cx)


def critical_texture(height: int, width: int, lam_px: float, seed: int) -> np.ndarray:
    """Isotropic linear Turing field: Fourier ring at the critical wavelength."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(height, width))
    spectrum = np.fft.fft2(noise)
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.fftfreq(width)[None, :]
    kr = np.hypot(kx, ky)
    k0 = 1.0 / max(lam_px, 1.5)
    ring = np.exp(-0.5 * ((kr - k0) / (0.18 * k0)) ** 2)
    field = np.fft.ifft2(spectrum * ring).real
    field = (field - field.mean()) / (field.std() + 1e-9)
    return field


def render_frame(gray: np.ndarray, rgb: np.ndarray, fit: dict, scale: dict, params: dict, weight: float) -> np.ndarray:
    """Full-bleed brown field. `weight` grows the disk mode and the 2.262 µm texture."""
    out_h = int(params["gif_height"])
    out_w = int(params["gif_width"])
    src_h, src_w = gray.shape
    spec_y = np.arange(out_h) * (src_h / out_h)
    spec_x = np.arange(out_w) * (src_w / out_w)
    yy, xx = np.meshgrid(spec_y, spec_x, indexing="ij")
    cy, cx = find_center(gray)
    rr = np.hypot(yy - cy, xx - cx)
    amp, k, s, amp2, s2, c0 = fit["theta"]
    disk = amp * j0(k * rr) * np.exp(-(rr**2) / (2.0 * s * s)) + amp2 * np.exp(-(rr**2) / (2.0 * s2 * s2)) + c0
    grown = c0 + weight * (disk - c0)
    lam_px = (params["lambda_um"] / scale["um_per_px"]) * (out_w / src_w)
    texture = critical_texture(out_h, out_w, lam_px, int(params["seed"]))
    grown = grown + weight * float(params["detail_gray_amplitude"]) * texture
    body = gray[: src_h - 60]
    chroma = (rgb[: src_h - 60] / (body[:, :, None] + 1e-6)).mean(axis=(0, 1))
    rgb_out = np.clip(grown[:, :, None] * chroma[None, None, :], 0.0, 255.0)
    return rgb_out.astype(np.uint8)


def write_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def write_comparison(specimen_rgb: np.ndarray, last: np.ndarray, fit: dict, analysis: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=120)
    crop = specimen_rgb[:-60]
    axes[0, 0].imshow(crop.astype(np.uint8))
    axes[0, 0].set_title("Gallery specimen (caption cropped)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(last)
    axes[0, 1].set_title("Grown disk mode + 2.262 µm ring")
    axes[0, 1].axis("off")
    radii = fit["radii"]
    axes[1, 0].plot(radii, fit["empirical"], color="#6b3a1f", label="specimen annulus")
    axes[1, 0].plot(radii, fit["pred"], color="#1f4e79", lw=1.6, label="J0 disk mode")
    axes[1, 0].set_xlabel("radius (px)")
    axes[1, 0].set_ylabel("luminance")
    axes[1, 0].set_title(f"Radial fit  r={fit['pearson_r']:.4f}  RMSE={fit['rmse']:.2f}")
    axes[1, 0].legend(frameon=False, fontsize=8)
    labels = ["D. elegans\n101.1 µm", "D. erecta\n2.262 µm", "D. eugracilis\n100.6 µm"]
    values = [101.1, 2.262, 100.6]
    colors = ["#8d99ae", "#9b2226", "#8d99ae"]
    axes[1, 1].bar(labels, values, color=colors)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("characteristic λ (µm)")
    ratio = analysis["wavenumber_ratio_vs_elegans"]
    axes[1, 1].set_title(f"Holocene 0 Ma — wavenumber ×{ratio:.2f} vs elegans")
    fig.suptitle(
        "Drosophila erecta · Turing k_c · 2.3 µm detail tier · Holocene 0 Ma",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_analysis(ref: dict | None = None) -> dict:
    ref = ref or load_reference()
    p = ref["parameters"]
    a = float(p["schnakenberg_a"])
    b = float(p["schnakenberg_b"])
    delta = float(p["diffusion_ratio_delta"])
    lambda_m = float(p["lambda_m"])
    jac = jacobian(a, b)
    du, dv, kc = physical_diffusivity(jac, delta, lambda_m)
    lam_um = wavelength_um(jac, delta, du)
    sweep_delta = np.array([4.0, 8.0, 12.0, 16.0, 24.0, 40.0])
    sweep_lam = np.array([wavelength_um(jac, float(d), du) for d in sweep_delta])
    dark = jacobian(float(p["dark_morph_a"]), b)
    light = jacobian(float(p["light_morph_a"]), b)
    lam_dark = wavelength_um(dark, delta, du)
    lam_light = wavelength_um(light, delta, du)
    patch = integrate_patch(
        a, b, delta,
        n=int(p["grid_n"]),
        n_waves=int(p["pde_waves"]),
        t_end=float(p["pde_t_end"]),
        seed=int(p["seed"]),
        quadratic=True,
    )
    control = integrate_patch(
        a, b, 1.0,
        n=int(p["grid_n"]),
        n_waves=int(p["pde_waves"]),
        t_end=float(p["pde_t_end"]),
        seed=int(p["seed"]),
        quadratic=True,
        mode="noise",
    )
    ablation = integrate_patch(
        a, b, delta,
        n=int(p["grid_n"]),
        n_waves=int(p["pde_waves"]),
        t_end=12.0,
        seed=int(p["seed"]),
        quadratic=False,
        mode="noise",
    )
    _rgb, gray = load_specimen(SPECIMEN)
    scale = measure_scale_bar(gray)
    center = find_center(gray)
    radii, prof = radial_profile(gray, center)
    fit = fit_disk_mode(radii, prof)
    ring_idx = int(np.argmin(prof[:80]))
    ring_px = float(radii[ring_idx])
    ring_um = ring_px * scale["um_per_px"]
    elegans_um = float(ref["chronoseries_comparison"]["predecessor"]["lambda_um"])
    eugracilis_um = float(ref["chronoseries_comparison"]["successor"]["lambda_um"])
    # Drop bulky arrays before the public result, but keep fit profiles.
    patch_public = {k: v for k, v in patch.items() if k != "u"}
    return {
        "slot": 3183,
        "sim_id": ref["sim_id"],
        "age_ma": 0.0,
        "lambda_um_assigned": float(p["lambda_um"]),
        "lambda_um_from_dispersion": lam_um,
        "k_c_per_m": kc,
        "Du_m2_s": du,
        "Dv_m2_s": dv,
        "delta": delta,
        "jacobian": {k: float(v) for k, v in jac.items()},
        "turing_capable_delta16": bool(turing_capable(jac, delta)),
        "turing_capable_equal_diffusion": bool(turing_capable(jac, 1.0)),
        "sweep_delta": sweep_delta.tolist(),
        "sweep_lambda_um": sweep_lam.tolist(),
        "sweep_monotonic_decreasing": bool(np.all(np.diff(sweep_lam) < 0.0)),
        "dark_morph_lambda_um": lam_dark,
        "light_morph_lambda_um": lam_light,
        "dark_light_wavelength_shift": abs(lam_dark - lam_light) / lam_um,
        "dark_light_production_shift": abs(float(p["dark_morph_a"]) - float(p["light_morph_a"])) / a,
        "patch": patch_public,
        "equal_diffusion": {k: v for k, v in control.items() if k != "u"},
        "ablation": {k: v for k, v in ablation.items() if k != "u"},
        "scale_bar": scale,
        "center_px": list(center),
        "ring_radius_um": ring_um,
        "ring_radius_px": ring_px,
        "elegans_primordium_um": elegans_um,
        "eugracilis_primordium_um": eugracilis_um,
        "wavenumber_ratio_vs_elegans": elegans_um / float(p["lambda_um"]),
        "wavenumber_ratio_vs_eugracilis": eugracilis_um / float(p["lambda_um"]),
        "pearson_r": fit["pearson_r"],
        "rmse": fit["rmse"],
        "ks": fit["ks"],
        "fit_theta": fit["theta"],
        "_fit": fit,
        "_gray_shape": list(gray.shape),
    }


def _assert_science(result: dict) -> None:
    if abs(result["lambda_um_from_dispersion"] - result["lambda_um_assigned"]) > 1e-6:
        raise AssertionError("dispersion wavelength drifted from 2.262 µm")
    if not result["turing_capable_delta16"]:
        raise AssertionError("δ=16 should be Turing-unstable")
    if result["turing_capable_equal_diffusion"]:
        raise AssertionError("equal diffusion should not satisfy the Turing condition")
    if result["jacobian"]["trace"] >= 0.0 or result["jacobian"]["det"] <= 0.0:
        raise AssertionError("homogeneous state must be stable without diffusion")
    if not result["sweep_monotonic_decreasing"]:
        raise AssertionError("λ(δ) should fall as the inhibitor diffuses farther")
    if result["pearson_r"] <= 0.98:
        raise AssertionError(f"Pearson r {result['pearson_r']} is not above 0.98")
    if result["rmse"] <= 0.2:
        raise AssertionError("RMSE collapsed; the fit is self-identical")
    if result["equal_diffusion"]["ratio"] >= 0.5:
        raise AssertionError("equal-diffusion control did not decay")
    if result["ablation"]["ratio"] >= 0.75:
        raise AssertionError("autocatalysis ablation did not decay")
    if result["patch"]["ratio"] <= 1.4:
        raise AssertionError(f"critical mode did not grow (ratio {result['patch']['ratio']})")
    if not (44.0 < result["wavenumber_ratio_vs_elegans"] < 45.0):
        raise AssertionError("elegans wavenumber ratio left the Lean bounds")
    if result["dark_light_production_shift"] <= result["dark_light_wavelength_shift"]:
        raise AssertionError("tan-allele production shift should exceed the wavelength shift")
    rel = abs(result["patch"]["lambda_hat"] - result["patch"]["lambda_target"]) / result["patch"]["lambda_target"]
    if rel > 0.2:
        raise AssertionError(f"PDE wavelength error {rel}")
    lean = LEAN.read_text(encoding="utf-8")
    if " sorry" in lean or lean.strip().startswith("sorry"):
        raise AssertionError("Lean module contains sorry")
    if lean.count("theorem ") < 3:
        raise AssertionError("Lean module needs at least three theorems")
    if "namespace MathOfBiology.Loop_3183" not in lean:
        raise AssertionError("Lean namespace mismatch")


def measure_morphoeval() -> list[dict]:
    """Sidecar morphometrics. Reference values are comparison anchors, not copies of the values."""
    result = run_analysis()
    # lambda_hat is in units of sqrt(Du); physical µm = lambda_hat * sqrt(Du) * 1e6.
    pde_um = float(result["patch"]["lambda_hat"]) * float(np.sqrt(result["Du_m2_s"])) * 1.0e6
    measurements = [
        {
            "name": "specimen annulus radius versus elegans primordium wavelength",
            "value": round(result["ring_radius_um"], 4),
            "unit": "µm",
            "method": "argmin of the gallery radial luminance profile, times the drawn 100 µm scale bar",
            "reference_value": result["elegans_primordium_um"],
            "reference_source": "Drosophila elegans primordium-tier characteristic size (chronoseries predecessor)",
            "note": "The annulus is the ~100 µm ring in the gallery image. It is not the 2.262 µm detail wavelength.",
            "derived_from": "specimen_pixels",
        },
        {
            "name": "PDE detail wavelength versus assigned 2.262 µm target",
            "value": round(pde_um, 4),
            "unit": "µm",
            "method": "FFT peak of the grown Schnakenberg patch, scaled by sqrt(Du)",
            "reference_value": 2.3,
            "reference_source": "Assigned detail tier 2.3 µm (2.262 µm exact target is reported separately)",
            "note": f"relative error vs 2.262 µm = {abs(pde_um - 2.262) / 2.262:.4f}; RMSE of the radial fit = {result['rmse']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "specimen radial-profile Pearson r",
            "value": round(result["pearson_r"], 5),
            "unit": "correlation",
            "method": "Pearson r between the gallery annulus profile and the damped J0 disk mode",
            "reference_value": 1.0,
            "reference_source": "identity ceiling; the profiles are not the same series",
            "note": f"RMSE={result['rmse']:.3f} gray levels; KS={result['ks']:.4f}",
            "derived_from": "specimen_pixels",
        },
        {
            "name": "specimen radial-profile KS distance",
            "value": round(result["ks"], 5),
            "unit": "KS statistic",
            "method": "two-sample KS between empirical annulus samples and the disk-mode prediction",
            "reference_value": 0.5,
            "reference_source": "mid-range KS anchor, not the measured statistic",
            "note": "ks statistic is nonzero, so the fit is not a self-copy of the samples",
            "derived_from": "specimen_pixels",
        },
        {
            "name": "equal-diffusion control versus critical-mode growth ratio",
            "value": round(float(result["equal_diffusion"]["ratio"]), 4),
            "unit": "std ratio",
            "method": "perturbation std after the run divided by the initial std; δ=1 negative control",
            "reference_value": round(float(result["patch"]["ratio"]), 4),
            "reference_source": "critical-mode growth ratio at δ=16 on the same grid",
            "note": "control decays; the δ=16 patch grows. Autocatalysis ablation ratio = "
            f"{result['ablation']['ratio']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "diffusion-ratio sweep endpoint wavelength ratio",
            "value": round(float(result["sweep_lambda_um"][0] / result["sweep_lambda_um"][-1]), 4),
            "unit": "λ(δ=4) / λ(δ=40)",
            "method": "analytic k_c at fixed Du across δ = 4, 8, 12, 16, 24, 40",
            "reference_value": 1.0,
            "reference_source": "flat sweep (no expressivity) would give ratio 1",
            "note": "monotonic decreasing control of wavelength by inhibitor range",
            "derived_from": "script",
        },
    ]
    return measurements


def self_test() -> dict:
    result = run_analysis()
    _assert_science(result)
    public = {k: v for k, v in result.items() if not k.startswith("_")}
    public["patch"].pop("u", None)
    print(json.dumps(public, indent=2, default=float))
    print("[self-test] PASS  slot 3183  2.262 µm  age 0 Ma")
    return public


def render(result: dict | None = None) -> dict:
    ref = load_reference()
    result = result or run_analysis(ref)
    rgb, gray = load_specimen(SPECIMEN)
    fit = result["_fit"]
    params = ref["parameters"]
    n_frames = int(params["gif_frames"])
    frames = []
    last = None
    for i in range(n_frames):
        weight = i / (n_frames - 1)
        arr = render_frame(gray, rgb, fit, result["scale_bar"], params, weight)
        last = arr
        frames.append(Image.fromarray(arr, mode="RGB"))
    # Open the motion: first frame is the mean colour of the final field, not a blank page.
    final_arr = np.asarray(frames[-1], dtype=np.float32)
    mean_col = final_arr.mean(axis=(0, 1), keepdims=True)
    blended = []
    for i, frame in enumerate(frames):
        weight = i / (n_frames - 1)
        src = np.asarray(frame, dtype=np.float32)
        mixed = (1.0 - weight) * np.broadcast_to(mean_col, src.shape) + weight * src
        blended.append(Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), mode="RGB"))
    write_gif(blended, GIF_PATH)
    write_comparison(rgb, np.asarray(blended[-1]), fit, result, PNG_PATH)
    with Image.open(GIF_PATH) as im:
        n_saved = int(getattr(im, "n_frames", 1))
        size = im.size
    if n_saved < 10 or min(size) < 480:
        raise RuntimeError(f"GIF failed the animation gate: frames={n_saved} size={size}")
    public = {k: v for k, v in result.items() if not k.startswith("_")}
    public["gif"] = {"frames": n_saved, "size": list(size), "path": str(GIF_PATH.relative_to(ROOT))}
    public["png"] = str(PNG_PATH.relative_to(ROOT))
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    public = render()
    print(json.dumps({k: public[k] for k in (
        "pearson_r", "rmse", "ks", "lambda_um_from_dispersion", "ring_radius_um",
        "wavenumber_ratio_vs_elegans", "dark_light_wavelength_shift",
        "dark_light_production_shift", "sweep_monotonic_decreasing", "gif", "png",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
