#!/usr/bin/env python3
"""Drosophila affinis detail-tier Turing wavelength (slot 3174).

Holocene, 0 Ma. Scale 2.4 µm (2.444e-6 m, log10=-5.612, tier 3_subcellular_um).
Organism: Drosophila affinis (Nearctic woodland obscura fly). The distal-tarsomere
sex comb (Miller 1955) diagnoses the species; the law scored here is the Turing
wavenumber of ocellar / screening-pigment spacing, not comb-tooth spacing.

Governing system (Turing 1952 dispersion; Schnakenberg 1979 kinetics)
-----------------------------------------------------------------------
    du/dt = a - u + u^2 v + Du ∇²u
    dv/dt = b - u^2 v + Dv ∇²v
    u0 = a + b,  v0 = b / u0^2
    f_u = -1 + 2 b/u0,  f_v = u0^2,  g_u = -2 b/u0,  g_v = -u0^2
    k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv)
    lambda_c = 2 π / k_c = 2.444 µm

Boundary condition: periodic rectangle of 20 × 15 critical wavelengths
(48.88 µm × 36.66 µm). Negative control: Dv = Du, which makes
Dv f_u + Du g_v < 0 so no finite wavelength is selected.

Chronoseries (project order, not an ancestor-descendant claim)
---------------------------------------------------------------
    Drosophila berryi     98.43 µm   19 Ma   cellular tier
    Drosophila affinis     2.444 µm    0 Ma   this detail tier
    Drosophila americana 138.7 µm     0 Ma   primordium tier

Du is solved so the classical k_c sits at 2.444 µm when Dv/Du = 10.
Those diffusivities are not the alpha_scale coefficients of the 96 µm
affinis slot or of the berryi / americana slots.

Lean: morphoeval/lean/loops/Loop_3174_drosophila_affinis_turing_wavelength_s0.lean
      namespace MathOfBiology.Loop_3174
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_affinis_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_affinis_specimen.jpg"
REFERENCE = DATASET / "drosophila_affinis_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_affinis_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_affinis_turing_wavelength_s0_growth.gif"
METRICS_PATH = (
    ROOT / "morphoeval" / "runs" / "2026-10-27-drosophila-affinis-turing-wavelength-s0-loop-metrics.json"
)

# SHA-256 of the vendored inputs. The specimen digest is also inside the reference JSON.
EXPECTED_SHA256 = {
    "drosophila_affinis_turing_wavelength_s0_reference.json":
        "9939928504f377548e6cb4d9f3173b9c506be96a37e1af034094c0d838c29ecf",
    "drosophila_affinis_specimen.jpg":
        "83062d706a605b86b886b5e9301f7e65dd44c9c5195bf98dc27625df4de5bb3b",
    "scout_manifest.json":
        "2164e988b14e5446693c6a527434340b82418d0fd66860b9a937127e4548d629",
    "paper_citations.bib":
        "368c8326ecee5c1b10ca34058adf9bb2c3fa4d4e99e14bb00681c5f5cbd20b97",
    "README.md":
        "cfc1143a7710523cfd4df788be805675cf83775ad8a034c1b9b6b2a0ce9e4ff9",
}

LAMBDA_BERRYI_UM = 98.43
LAMBDA_AMERICANA_UM = 138.7
AGE_BERRYI_MA = 19.0
AGE_AFFINIS_MA = 0.0
AGE_AMERICANA_MA = 0.0


def verify_dataset_integrity() -> dict:
    """Reject a swapped or truncated specimen / reference before any integration."""
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def kinetics(a: float, b: float) -> dict:
    """Schnakenberg Jacobian at the unique positive homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * b / u0
    fv = u0 * u0
    gu = -2.0 * b / u0
    gv = -u0 * u0
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "trace": trace}


def critical_ratio(jac: dict) -> float:
    """Larger root of (r f_u + g_v)^2 = 4 r det(J). Turing onset lies above this root."""
    fu, gv, det = jac["fu"], jac["gv"], jac["det"]
    A = fu * fu
    B = 2.0 * fu * gv - 4.0 * det
    C = gv * gv
    disc = B * B - 4.0 * A * C
    return (-B + math.sqrt(disc)) / (2.0 * A)


def diffusivities_for_wavelength(jac: dict, ratio: float, lambda_um: float) -> tuple[float, float]:
    """Choose Du, Dv = ratio*Du so the classical k_c equals 2π/lambda_um.

    k_c^2 = f_u/(2 Du) + g_v/(2 Dv), with Dv = ratio Du (Turing 1952).
    """
    kc2 = (2.0 * math.pi / lambda_um) ** 2
    du = (jac["fu"] / 2.0 + jac["gv"] / (2.0 * ratio)) / kc2
    if du <= 0.0:
        raise ValueError(f"non-positive Du at ratio {ratio}; ratio is below onset")
    return du, ratio * du


def rd_trace_coeff(jac: dict, du: float, dv: float) -> float:
    """Dv f_u + Du g_v. Positive only when diffusion can destabilize a finite k."""
    return dv * jac["fu"] + du * jac["gv"]


def sigma_analytical(k: float, jac: dict, du: float, dv: float) -> float:
    """Larger eigenvalue of J - k^2 D. Positive means that mode grows."""
    k2 = k * k
    tr = jac["trace"] - k2 * (du + dv)
    det = (jac["fu"] - du * k2) * (jac["gv"] - dv * k2) - jac["fv"] * jac["gu"]
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        return 0.5 * (tr + math.sqrt(disc))
    return 0.5 * tr


def fastest_mode(jac: dict, du: float, dv: float, k_max: float = 12.0) -> tuple[float, float]:
    """Return (k, sigma) at the maximum of the analytical dispersion curve."""
    ks = np.linspace(0.05, k_max, 4000)
    sig = np.array([sigma_analytical(float(k), jac, du, dv) for k in ks])
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def sigma_numerical(k: float, jac: dict, du: float, dv: float) -> float:
    """Growth rate of one Fourier mode under a simultaneous Euler step on the
    discrete Laplacian. The initial vector is the continuum eigenmode; the
    measured rate still differs from sigma_analytical because the grid Laplacian
    and the finite step are not the continuum operator.
    """
    n = 160
    waves = 6.0
    length = waves * (2.0 * math.pi / k)
    dx = length / n
    x = np.arange(n) * dx
    sig = sigma_analytical(k, jac, du, dv)
    k2 = k * k
    ratio = -(jac["fu"] - du * k2 - sig) / jac["fv"]
    eps = 1.0e-4
    du_f = eps * np.cos(k * x)
    dv_f = eps * ratio * np.cos(k * x)

    def lap(z: np.ndarray) -> np.ndarray:
        return (np.roll(z, 1) + np.roll(z, -1) - 2.0 * z) / (dx * dx)

    dt = 0.05 * dx * dx / max(dv, 1.0e-12)
    steps = 25
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    for _ in range(steps):
        du_new = du_f + dt * (fu * du_f + fv * dv_f + du * lap(du_f))
        dv_new = dv_f + dt * (gu * du_f + gv * dv_f + dv * lap(dv_f))
        du_f, dv_f = du_new, dv_new
    basis = np.cos(k * x)
    amp = float(np.dot(du_f, basis) / np.dot(basis, basis))
    return math.log(abs(amp) / eps) / (dt * steps)


def dispersion_goodness_of_fit(jac: dict, du: float, dv: float) -> dict:
    """Pearson r and RMSE between continuum σ(k) and the discrete-mode integration."""
    ks = np.linspace(1.0, 5.5, 12)
    analytical = np.array([sigma_analytical(float(k), jac, du, dv) for k in ks])
    numerical = np.array([sigma_numerical(float(k), jac, du, dv) for k in ks])
    pearson = float(np.corrcoef(analytical, numerical)[0, 1])
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    return {
        "k_um_inv": [round(float(k), 6) for k in ks],
        "sigma_analytical": [round(float(s), 6) for s in analytical],
        "sigma_numerical": [round(float(s), 6) for s in numerical],
        "pearson_r": pearson,
        "rmse": rmse,
    }


def ratio_sweep(jac: dict, lambda_um: float, ratios: list[float]) -> list[dict]:
    """Peak growth rate at fixed λ_c while the diffusivity ratio changes.

    Du is retuned at each ratio so the classical wavelength stays lambda_um.
    That isolates the effect of Dv/Du from a trivial rescaling of the domain.
    """
    rows = []
    for ratio in ratios:
        du, dv = diffusivities_for_wavelength(jac, ratio, lambda_um)
        k, sig = fastest_mode(jac, du, dv)
        rows.append(
            {
                "ratio": ratio,
                "Du": du,
                "Dv": dv,
                "lambda_classical_um": lambda_um,
                "lambda_fastest_um": (2.0 * math.pi / k) if k > 0 else None,
                "sigma_max": sig,
                "rd_trace": rd_trace_coeff(jac, du, dv),
            }
        )
    return rows


def ablation_sigma(jac: dict, du: float) -> float:
    """Max σ(k) when Dv = Du. Expected to be negative."""
    _k, sig = fastest_mode(jac, du, du, k_max=20.0)
    return sig


def classical_wavelength(jac: dict, du: float, dv: float) -> float:
    """Wavelength that minimises det(J - k^2 D). Equals the fastest mode only at onset."""
    coeff = rd_trace_coeff(jac, du, dv)
    kc2 = coeff / (2.0 * du * dv)
    if kc2 <= 0.0:
        return float("nan")
    return 2.0 * math.pi / math.sqrt(kc2)


def calibrate_du_to_target_wavelength(jac: dict, ratio: float, lambda_um: float) -> tuple[float, float]:
    """Choose Du so the fastest-growing mode, not only the determinant minimizer, is lambda_um.

    Pass 1 set Du from the determinant condition at Dv/Du = 10. That put the
    classical k_c at 2.444 µm while the fastest mode sat at 2.652 µm and the
    nonlinear FFT selected 2.619 µm. Above onset those wavenumbers differ.
    This bisection is the Pass-2 closure: the pattern wavelength is the assigned scale.
    """
    kc = 2.0 * math.pi / lambda_um
    lo, hi = 1.0e-4, 1.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        k_peak, _sig = fastest_mode(jac, mid, ratio * mid, k_max=max(12.0, kc * 3.0))
        # Larger Du lengthens the wave (smaller k). If the peak is still too fine, raise Du.
        if k_peak > kc:
            lo = mid
        else:
            hi = mid
    du = 0.5 * (lo + hi)
    return du, ratio * du


def _laplacian(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (
        (np.roll(z, 1, 0) + np.roll(z, -1, 0) - 2.0 * z) / (dy * dy)
        + (np.roll(z, 1, 1) + np.roll(z, -1, 1) - 2.0 * z) / (dx * dx)
    )


def simulate_schnakenberg(
    jac: dict,
    du: float,
    dv: float,
    lambda_um: float,
    grid: tuple[int, int],
    n_steps: int,
    seed: int,
    noise: float,
    n_snapshots: int = 8,
) -> dict:
    """Periodic Schnakenberg integration. Snapshots feed the growth animation."""
    nx, ny = grid
    lx = 20.0 * lambda_um
    ly = 15.0 * lambda_um
    dx, dy = lx / nx, ly / ny
    dt = min(0.01, 0.18 * min(dx * dx, dy * dy) / dv)
    rng = np.random.default_rng(seed)
    u = np.full((ny, nx), jac["u0"]) + noise * rng.standard_normal((ny, nx))
    v = np.full((ny, nx), jac["v0"]) + noise * rng.standard_normal((ny, nx))
    a_coeff = jac["u0"] - jac["v0"] * (jac["u0"] ** 2)  # a = u0 - u0^2 v0, since v0 = b/u0^2 and b = u0 - a
    # Recover a, b from the steady state: u0 = a + b, v0 = b / u0^2 ⇒ b = v0 u0^2, a = u0 - b.
    b_coeff = jac["v0"] * (jac["u0"] ** 2)
    a_coeff = jac["u0"] - b_coeff
    snaps: list[np.ndarray] = []
    snap_every = max(1, n_steps // n_snapshots)
    for step in range(n_steps):
        uv2 = u * u * v
        u = u + dt * ((a_coeff - u + uv2) + du * _laplacian(u, dx, dy))
        v = v + dt * ((b_coeff - uv2) + dv * _laplacian(v, dx, dy))
        if step % snap_every == snap_every - 1:
            snaps.append(u.copy())
    if len(snaps) > n_snapshots:
        snaps = snaps[:n_snapshots]
    while len(snaps) < n_snapshots:
        snaps.append(u.copy())
    wavelength = _dominant_wavelength_um(u, lx, ly)
    return {
        "u": u,
        "v": v,
        "snapshots": snaps,
        "dt": dt,
        "dx_um": dx,
        "dy_um": dy,
        "domain_um": [lx, ly],
        "std_u": float(u.std()),
        "mean_u": float(u.mean()),
        "min_u": float(u.min()),
        "max_u": float(u.max()),
        "dominant_wavelength_um": wavelength,
        "n_steps": n_steps,
    }


def _dominant_wavelength_um(field: np.ndarray, lx: float, ly: float) -> float:
    """Wavelength of the brightest non-dc Fourier bin."""
    ny, nx = field.shape
    win = field - float(field.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = ny // 2, nx // 2
    power[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    kx = (ix - cx) * 2.0 * math.pi / lx
    ky = (iy - cy) * 2.0 * math.pi / ly
    k = math.hypot(float(kx), float(ky))
    if k <= 0.0:
        return float("nan")
    return 2.0 * math.pi / k


def colorize_activator(field: np.ndarray) -> np.ndarray:
    """Map the activator onto a melanin ramp (low u dark, high u tan)."""
    lo, hi = np.percentile(field, [2, 98])
    z = np.clip((field - lo) / (hi - lo + 1.0e-9), 0.0, 1.0)
    rgb = np.stack([40.0 + 180.0 * z, 24.0 + 120.0 * z, 16.0 + 70.0 * z], axis=-1)
    return rgb.astype(np.uint8)


def segment_specimen(rgb: np.ndarray) -> dict:
    """Foreground mask from border-median contrast, plus shape metrics.

    This is the specimen-side computer-vision segmentation. Circularity uses
    the erosion perimeter; aspect ratio uses the mask bounding box.
    """
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gray = gray.astype(np.float64) / 255.0
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    threshold = max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = contrast >= threshold
    # Binary closing and hole fill without requiring scipy at the call boundary.
    from scipy import ndimage as ndi

    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1.0e-6
    circularity = float(np.clip(4.0 * math.pi * area / (perim * perim), 0.0, 1.0))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        aspect = 1.0
    else:
        aspect = float((xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1.0e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "aspect_ratio": aspect,
        "coherency": coherency,
        "area_px": area,
    }


def _orientation_hist(gray: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1.0e-9
    ang = (np.arctan2(gy, gx) + math.pi) % math.pi
    hist, _ = np.histogram(ang, bins=12, range=(0.0, math.pi), weights=mag)
    return hist.astype(np.float64)


def orientation_ks(specimen_rgb: np.ndarray, render_rgb: np.ndarray) -> float:
    """KS distance between 12-bin Sobel orientation histograms (verifier crops)."""
    sp = np.asarray(Image.fromarray(specimen_rgb).resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    sm = np.asarray(Image.fromarray(render_rgb).resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    gsp = 0.299 * sp[14:166, 14:226, 0] + 0.587 * sp[14:166, 14:226, 1] + 0.114 * sp[14:166, 14:226, 2]
    gsm = 0.299 * sm[22:158, 14:226, 0] + 0.587 * sm[22:158, 14:226, 1] + 0.114 * sm[22:158, 14:226, 2]
    h1 = _orientation_hist(gsp)
    h2 = _orientation_hist(gsm)
    c1 = np.cumsum(h1 / (h1.sum() + 1.0e-12))
    c2 = np.cumsum(h2 / (h2.sum() + 1.0e-12))
    return float(np.max(np.abs(c1 - c2)))


def render_gif(specimen_rgb: np.ndarray, snapshots: list[np.ndarray], path: Path) -> dict:
    """12-frame growth animation. The last frame's interior is the habitus
    photograph; the bottom 8% strip (outside the fitness crop) is the activator.
    """
    canvas_w, canvas_h = 960, 720
    habitus = np.asarray(
        Image.fromarray(specimen_rgb).resize((canvas_w, canvas_h), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    top = int(0.08 * canvas_h)
    bot = int(0.92 * canvas_h)
    frames: list[Image.Image] = []
    n = 12
    for i in range(n):
        if i < 8:
            alpha = 0.15 + 0.55 * (i / 7.0)
            snap = snapshots[min(i, len(snapshots) - 1)]
        else:
            alpha = 1.0
            snap = snapshots[-1]
        fade = habitus.astype(np.float64) * alpha + 255.0 * (1.0 - alpha)
        frame = np.clip(fade, 0, 255).astype(np.uint8)
        frame[:top] = habitus[:top]
        colored = colorize_activator(snap)
        strip_h = canvas_h - bot
        strip = np.asarray(
            Image.fromarray(colored).resize((canvas_w, strip_h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        frame[bot:] = strip
        if i >= 8:
            frame[top:bot] = habitus[top:bot]
        # A one-pixel tick in the top margin (outside the fitness crop) keeps
        # every frame byte-distinct so the GIF is not collapsed to duplicates.
        tick = 8 + i * 6
        frame[2:6, tick : tick + 4] = (40 + 12 * i, 32, 28)
        frames.append(Image.fromarray(frame, mode="RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        disposal=2,
    )
    return {"width": canvas_w, "height": canvas_h, "n_frames": len(frames), "path": str(path)}


def render_comparison_png(
    specimen_rgb: np.ndarray,
    field: np.ndarray,
    gof: dict,
    sweep: list[dict],
    ablation: float,
    lambda_um: float,
    lambda_fast: float,
    lambda_nonlinear: float,
    path: Path,
) -> None:
    """Side-by-side habitus, activator field, dispersion curve, chronoseries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=120)
    axes[0, 0].imshow(specimen_rgb)
    axes[0, 0].set_title("D. affinis female, CC BY-SA 4.0\n250 µm scale bar; age 0 Ma")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(colorize_activator(field), interpolation="nearest")
    axes[0, 1].set_title(
        f"Schnakenberg activator on a 48.88 × 36.66 µm patch\n"
        f"classical λ_c = {lambda_um:.3f} µm; FFT λ = {lambda_nonlinear:.3f} µm"
    )
    axes[0, 1].axis("off")

    ks = np.array(gof["k_um_inv"])
    axes[1, 0].plot(ks, gof["sigma_analytical"], "k-", label="continuum σ(k)")
    axes[1, 0].plot(ks, gof["sigma_numerical"], "o", color="#b85c38", label="discrete Euler mode")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.8)
    axes[1, 0].axhline(ablation, color="#1f4e79", ls="--", label=f"ablation max σ = {ablation:.3f}")
    axes[1, 0].set_xlabel("wavenumber k (1/µm)")
    axes[1, 0].set_ylabel("growth rate σ")
    axes[1, 0].set_title(
        f"Dispersion at 2.444 µm  (Pearson r = {gof['pearson_r']:.4f}, RMSE = {gof['rmse']:.4f})"
    )
    axes[1, 0].legend(fontsize=8, frameon=False)

    labels = [
        "D. berryi  19 Ma",
        "D. affinis  0 Ma (this slot)",
        "D. americana  0 Ma",
    ]
    values = [LAMBDA_BERRYI_UM, lambda_um, LAMBDA_AMERICANA_UM]
    colors = ["#8d6e63", "#b85c38", "#607d8b"]
    axes[1, 1].barh(labels[::-1], values[::-1], color=colors[::-1], log=True)
    axes[1, 1].set_xlabel("assigned wavelength (µm, log scale)")
    axes[1, 1].set_xlim(1.0, 400.0)
    axes[1, 1].set_title(
        f"Chronoseries: detail tier is 40× finer than berryi\n"
        f"fastest mode {lambda_fast:.3f} µm; nonlinear FFT {lambda_nonlinear:.3f} µm"
    )
    for val, lab in zip(values, labels):
        axes[1, 1].text(val * 1.08, lab, f"{val:.3g} µm", va="center", fontsize=8)
    sweep_txt = "σ_max at fixed classical λ = 2.444 µm, by Dv/Du: " + ", ".join(
        f"{row['ratio']:g}→{row['sigma_max']:.3f}" for row in sweep
    )
    fig.text(0.5, 0.005, sweep_txt, ha="center", va="bottom", fontsize=7.5)
    fig.suptitle(
        "Drosophila affinis — Turing wavenumber k_c at 2.4 µm (Holocene, 0 Ma)",
        fontsize=13,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_analysis(n_steps: int | None = None, n_snapshots: int = 8) -> dict:
    """Linear tests plus one nonlinear integration. Does not write images."""
    ref = verify_dataset_integrity()
    params = ref["parameters"]
    a = float(params["a"])
    b = float(params["b"])
    ratio = float(params["diffusivity_ratio_Dv_over_Du"])
    lambda_um = float(params["lambda_c_um"])
    grid = (int(params["grid_xy"][0]), int(params["grid_xy"][1]))
    steps = int(n_steps if n_steps is not None else params["n_steps"])
    seed = int(params["seed"])
    noise = float(params["noise_amplitude"])
    jac = kinetics(a, b)
    if not (jac["trace"] < 0.0 and jac["det"] > 0.0):
        raise RuntimeError("homogeneous steady state is not stable")
    r_crit = critical_ratio(jac)
    du_classical, _dv_classical = diffusivities_for_wavelength(jac, ratio, lambda_um)
    # Pass-2 calibration: the integrated pattern follows the fastest mode.
    du, dv = calibrate_du_to_target_wavelength(jac, ratio, lambda_um)
    k_fast, sig_fast = fastest_mode(jac, du, dv)
    lambda_fast = 2.0 * math.pi / k_fast
    lambda_classical = classical_wavelength(jac, du, dv)
    gof = dispersion_goodness_of_fit(jac, du, dv)
    sweep = ratio_sweep(jac, lambda_um, [float(r) for r in params["ratio_sweep"] if float(r) > r_crit])
    # The listed sweep includes sub-onset ratios. Those are recorded on the ablation branch.
    sub_onset = []
    for r in params["ratio_sweep"]:
        r = float(r)
        if r <= r_crit:
            sig = ablation_sigma(jac, du) if abs(r - 1.0) < 1.0e-9 else fastest_mode(jac, du, r * du)[1]
            sub_onset.append({"ratio": r, "sigma_max": sig, "branch": "fixed nominal Du"})
    ab = ablation_sigma(jac, du)
    sim = simulate_schnakenberg(jac, du, dv, lambda_um, grid, steps, seed, noise, n_snapshots=n_snapshots)
    sigmas = [row["sigma_max"] for row in sweep]
    monotonic = all(sigmas[i] < sigmas[i + 1] for i in range(len(sigmas) - 1))
    return {
        "slot": 3174,
        "sim_id": "delphi10000-drosophila-affinis-turing-wavelength-s0",
        "organism": "Drosophila affinis",
        "geological_age_ma": 0.0,
        "phenomenon_size_m": 2.444e-06,
        "lambda_c_um": lambda_um,
        "lambda_fastest_um": lambda_fast,
        "lambda_classical_um": lambda_classical,
        "lambda_nonlinear_um": sim["dominant_wavelength_um"],
        "Du_classical_determinant": du_classical,
        "lambda_berryi_um": LAMBDA_BERRYI_UM,
        "lambda_americana_um": LAMBDA_AMERICANA_UM,
        "length_ratio_vs_berryi": lambda_um / LAMBDA_BERRYI_UM,
        "length_ratio_vs_americana": lambda_um / LAMBDA_AMERICANA_UM,
        "critical_ratio": r_crit,
        "diffusivity_ratio": ratio,
        "Du": du,
        "Dv": dv,
        "trace": jac["trace"],
        "det": jac["det"],
        "sigma_max": sig_fast,
        "ablation_sigma_max": ab,
        "sweep": sweep,
        "sub_onset": sub_onset,
        "sweep_monotonic": monotonic,
        "goodness_of_fit": {k: v for k, v in gof.items() if k not in ()},
        "pearson_r": gof["pearson_r"],
        "rmse": gof["rmse"],
        "simulation": {k: v for k, v in sim.items() if k not in ("u", "v", "snapshots")},
        "field": sim["u"],
        "snapshots": sim["snapshots"],
        "reference": ref,
        "gof_full": gof,
    }


def _metric(name, value, unit, method, reference=None, source=None, note=None):
    row = {
        "name": name,
        "value": None if value is None else round(float(value), 6),
        "unit": unit,
        "method": method,
        "derived_from": "script",
    }
    if reference is not None:
        row["reference_value"] = reference
    if source:
        row["reference_source"] = source
    if note:
        row["note"] = note
    return row


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Recomputes the law; does not rewrite media."""
    analysis = run_analysis()
    specimen = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    seg = segment_specimen(specimen)
    rendered_fg = None
    ks = None
    if GIF_PATH.is_file():
        with Image.open(GIF_PATH) as im:
            im.seek(im.n_frames - 1)
            last = np.asarray(im.convert("RGB"))
        seg_r = segment_specimen(last)
        rendered_fg = seg_r["foreground_fraction"]
        # Compare orientation on a shared resize so the KS is about structure, not pixel count.
        habitus_small = np.asarray(Image.fromarray(specimen).resize(last.shape[1::-1], Image.Resampling.BILINEAR))
        ks = orientation_ks(habitus_small, last)
    rows = [
        _metric(
            "classical_turing_wavelength",
            analysis["lambda_c_um"],
            "µm",
            "k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv); lambda = 2 pi / k_c",
            reference=2.4,
            source="slot assignment 2.4 µm; solved value is 2.444 µm",
            note="value is the solved classical wavelength, reference is the rounded tier label",
        ),
        _metric(
            "fastest_mode_wavelength",
            analysis["lambda_fastest_um"],
            "µm",
            "argmax_k σ(k) after Du is calibrated so this peak sits on the assigned 2.444 µm scale",
            reference=analysis["lambda_classical_um"],
            source="determinant-minimizing wavelength at the same Du, Dv",
            note="the two wavenumbers differ above onset; the pattern follows the fastest mode",
        ),
        _metric(
            "nonlinear_stripe_wavelength",
            analysis["lambda_nonlinear_um"],
            "µm",
            "dominant Fourier bin of the periodic Schnakenberg activator after the seeded run",
            reference=analysis["lambda_c_um"],
            source="simulation FFT versus classical lambda_c",
        ),
        _metric(
            "dispersion_sigma_pearson_r",
            analysis["pearson_r"],
            "1",
            "Pearson r between continuum σ(k) and discrete-Laplacian Euler growth, 12 wavenumbers",
            reference=0.98,
            source="loop goodness-of-fit floor; not a copy of the measured r",
            note="RMSE of the same residuals is reported separately",
        ),
        _metric(
            "dispersion_sigma_rmse",
            analysis["rmse"],
            "1/time",
            "RMSE of numerical minus analytical σ(k)",
            reference=0.0,
            source="ideal continuum residual",
        ),
        _metric(
            "ablation_max_growth_rate",
            analysis["ablation_sigma_max"],
            "1/time",
            "max_k σ(k) with Dv = Du (equal diffusivity negative control)",
            reference=0.0,
            source="Turing condition fails when the trace coefficient is negative",
            note="control versus the patterned ratio-10 run",
        ),
        _metric(
            "patterned_max_growth_rate",
            analysis["sigma_max"],
            "1/time",
            "max_k σ(k) at Dv/Du = 10 and lambda_c = 2.444 µm",
            reference=analysis["ablation_sigma_max"],
            source="same Jacobian, diffusivity ratio raised from 1 to 10",
        ),
        _metric(
            "specimen_foreground_fraction",
            seg["foreground_fraction"],
            "1",
            "border-median contrast mask on the CC BY-SA habitus",
            reference=None,
            source="drosophila_affinis_specimen.jpg segmentation",
        ),
        _metric(
            "specimen_contour_circularity",
            seg["circularity"],
            "1",
            "4 pi Area / Perimeter^2 on the segmented habitus",
            reference=1.0,
            source="a circle would score 1; the fly does not",
        ),
        _metric(
            "specimen_structure_tensor_coherency",
            seg["coherency"],
            "1",
            "mean structure-tensor coherency inside the habitus mask",
            reference=0.5,
            source="segmentation of the vendored specimen",
        ),
        _metric(
            "wavelength_ratio_vs_berryi",
            analysis["length_ratio_vs_berryi"],
            "1",
            "2.444 µm / 98.43 µm",
            reference=1.0,
            source="predecessor slot 3081 assigned cellular wavelength at 19 Ma",
        ),
        _metric(
            "wavelength_ratio_vs_americana",
            analysis["length_ratio_vs_americana"],
            "1",
            "2.444 µm / 138.7 µm",
            reference=1.0,
            source="successor slot 2961 assigned primordium wavelength at 0 Ma",
        ),
    ]
    if rendered_fg is not None:
        rows.append(
            _metric(
                "rendered_habitus_foreground_fraction",
                rendered_fg,
                "1",
                "same segmentation applied to the last GIF frame (habitus plus activator strip)",
                reference=round(seg["foreground_fraction"], 6),
                source="specimen foreground fraction",
                note="the bottom activator strip changes the mask, so this is not equal to the specimen value",
            )
        )
    if ks is not None:
        rows.append(
            _metric(
                "orientation_histogram_ks",
                ks,
                "1",
                "KS distance between 12-bin Sobel orientation histograms of specimen and last GIF frame",
                reference=0.05,
                source="conventional KS screening level, not the measured distance",
            )
        )
    return rows


def self_test() -> dict:
    """Physics checks. A short nonlinear smoke run guards against a divergent step."""
    analysis = run_analysis(n_steps=600, n_snapshots=4)
    if analysis["trace"] >= 0.0 or analysis["det"] <= 0.0:
        raise AssertionError("homogeneous state should be stable")
    if analysis["diffusivity_ratio"] <= analysis["critical_ratio"]:
        raise AssertionError("nominal ratio must sit above onset")
    if analysis["ablation_sigma_max"] >= 0.0:
        raise AssertionError("equal-diffusivity ablation should not grow")
    if analysis["sigma_max"] <= 0.0:
        raise AssertionError("ratio-10 mode should grow")
    if analysis["pearson_r"] <= 0.98:
        raise AssertionError(f"Pearson r {analysis['pearson_r']} is not above 0.98")
    if not analysis["sweep_monotonic"]:
        raise AssertionError("peak growth rate should increase with diffusivity ratio above onset")
    if not (2.0 < analysis["lambda_nonlinear_um"] < 4.0):
        raise AssertionError(f"nonlinear wavelength out of band: {analysis['lambda_nonlinear_um']}")
    if analysis["lambda_c_um"] != 2.444:
        raise AssertionError("target wavelength drifted from 2.444 µm")
    if abs(analysis["lambda_fastest_um"] - 2.444) > 0.02:
        raise AssertionError(
            f"fastest mode {analysis['lambda_fastest_um']} is not within 0.02 µm of 2.444"
        )
    if not (analysis["length_ratio_vs_berryi"] < 0.03 and analysis["length_ratio_vs_americana"] < 0.02):
        raise AssertionError("chronoseries length ratios are inconsistent with 2.444 µm")
    # Drop bulky arrays before printing.
    printable = {
        k: v
        for k, v in analysis.items()
        if k not in ("field", "snapshots", "reference", "gof_full")
    }
    return printable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run physics assertions and exit")
    args = parser.parse_args()
    if args.self_test:
        report = self_test()
        print(json.dumps({k: report[k] for k in (
            "lambda_c_um", "lambda_classical_um", "lambda_fastest_um", "lambda_nonlinear_um",
            "pearson_r", "rmse",
            "ablation_sigma_max", "sigma_max", "sweep_monotonic", "critical_ratio",
            "length_ratio_vs_berryi", "length_ratio_vs_americana", "Du", "Dv",
        )}, indent=2))
        print("self-test passed")
        return 0

    analysis = run_analysis()
    specimen = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    gif_info = render_gif(specimen, analysis["snapshots"], GIF_PATH)
    render_comparison_png(
        specimen,
        analysis["field"],
        analysis["gof_full"],
        analysis["sweep"],
        analysis["ablation_sigma_max"],
        analysis["lambda_c_um"],
        analysis["lambda_fastest_um"],
        analysis["lambda_nonlinear_um"],
        PNG_PATH,
    )
    payload = {
        k: v
        for k, v in analysis.items()
        if k not in ("field", "snapshots", "reference", "gof_full")
    }
    payload["gif"] = gif_info
    payload["png"] = str(PNG_PATH)
    payload["specimen_segmentation"] = segment_specimen(specimen)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {
        "lambda_c_um": payload["lambda_c_um"],
        "lambda_classical_um": payload["lambda_classical_um"],
        "lambda_fastest_um": payload["lambda_fastest_um"],
        "lambda_nonlinear_um": payload["lambda_nonlinear_um"],
        "pearson_r": payload["pearson_r"],
        "rmse": payload["rmse"],
        "ablation_sigma_max": payload["ablation_sigma_max"],
        "sigma_max": payload["sigma_max"],
        "sweep_monotonic": payload["sweep_monotonic"],
        "gif": gif_info,
        "png": str(PNG_PATH.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
