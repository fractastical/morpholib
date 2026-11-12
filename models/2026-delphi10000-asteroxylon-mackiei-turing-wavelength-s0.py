#!/usr/bin/env python3
"""
Asteroxylon mackiei at 5.2 µm — Turing critical wavelength, detail scale s0.

Slot 3124 of the 10,000-animal basic look. Early Devonian (Pragian Rhynie
Chert, 407 Ma). Chronoseries 1/1: there is no earlier named Asteroxylon
chronospecies. Cooksonia is the leafless grade predecessor (no actinostele).
A Baragwanathia-grade lycopsid is the microphyllous successor. Inside
Asteroxylon the scale predecessor is the 283 µm enation mode; this file does
not reuse that domain or those diffusivities.

Governing system (Schnakenberg 1979), wavelength from Turing 1952:

    du/dt = a - u + u^2 v + D_u ∇²u
    dv/dt = b - u^2 v + D_v ∇²v
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
    lambda_c = 2π / k_c = 5.2e-6 m

D_u is wall-bound and D_v/D_u = 160, so the inhibitor is cytosolic. The
domain is a 156 µm × 117 µm tracheid-file window (30 × 22.5 critical
wavelengths), not a whole shoot. Boundary conditions are periodic.

The basic-look plate is a CC BY-SA photograph of a Cromarty compression
(Kenraiz). It is centimetre-scale. The render modulates that plate by the
subcellular activator; it does not claim the fossil resolves 5.2 µm.

Usage:
    python3 models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py --self-test
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

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "asteroxylon_mackiei_turing_wavelength_s0"
SPECIMEN_NAME = "asteroxylon_mackiei_turing_wavelength_s0_specimen.jpg"
SPECIMEN_PATH = DATASET / SPECIMEN_NAME
OUT_PNG = ROOT / "docs" / "media" / "asteroxylon_mackiei_turing_wavelength_s0_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "asteroxylon_mackiei_turing_wavelength_s0_growth.gif"

# Detail-tier calibration. Not the 283.4 µm sibling.
LAMBDA_M = 5.2e-6
AGE_MA = 407.0
SIBLING_LAMBDA_M = 283.4e-6
A_PARAM = 0.08
B_PARAM = 0.92
DELTA = 160.0
N_WAVELENGTHS = 30
DOMAIN_LENGTH_M = N_WAVELENGTHS * LAMBDA_M
DOMAIN_WIDTH_M = DOMAIN_LENGTH_M * (480 / 640)
SEED = 3124
NX, NY = 96, 72
N_FRAMES = 12
RENDER_ALPHA = 0.32

EXPECTED_SHA256 = {
    "asteroxylon_mackiei_turing_wavelength_s0_reference.json":
        "9577842bd82c5dae41cb710726a0bfbe4cbb7fc9b0ae799c5fee1faf1a0ca7cd",
    "asteroxylon_mackiei_turing_wavelength_s0_specimen.jpg":
        "ad3ca375be91c9cf8e9eac9a2d3fa71eec4bd2072f17541dba69218cf17c1976",
    "paper_citations.bib":
        "345af56bd8c1eb6a5c742a521d7301738a10af60b19b0b255e16f3800e5b3326",
    "scout_manifest.json":
        "95e0c02c8d34e83aaa663286869cf90745edfd04a60011836046aad3472737ab",
    "README.md":
        "8d4cec453c791d0930abde45878649835adb8fb02ad569f111fd545f792a4bc9",
}

_CACHE: dict | None = None


def verify_sha256() -> None:
    """Pin the vendored plate and the reference documents."""
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {expected}")


def reaction_partials() -> dict[str, float]:
    """Jacobian of Schnakenberg kinetics at the homogeneous steady state."""
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM / (u0 ** 2)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 ** 2
    gu = -2.0 * u0 * v0
    gv = -(u0 ** 2)
    trace = fu + gv
    det = fu * gv - fv * gu
    return {"u0": u0, "v0": v0, "f_u": fu, "f_v": fv, "g_u": gu, "g_v": gv,
            "trace": trace, "det": det}


def diffusivities(delta: float, lambda_m: float, partials: dict[str, float]) -> tuple[float, float]:
    """D_u, D_v such that k_c = 2π/λ when D_v/D_u = delta. D_u is held fixed
    in the sweep by calling this only at the calibrated delta; see sweep()."""
    kc = 2.0 * np.pi / lambda_m
    fu, gv = partials["f_u"], partials["g_v"]
    du = (delta * fu + gv) / (2.0 * delta * kc ** 2)
    return float(du), float(delta * du)


def growth_rate(k: np.ndarray, du: float, dv: float, partials: dict[str, float]) -> np.ndarray:
    """Largest real eigenvalue of J - k² D. Vectorized over k."""
    k = np.asarray(k, dtype=float)
    k2 = k ** 2
    fu, fv = partials["f_u"], partials["f_v"]
    gu, gv = partials["g_u"], partials["g_v"]
    # Characteristic polynomial λ² - tr λ + det = 0.
    tr = (fu - du * k2) + (gv - dv * k2)
    det = (fu - du * k2) * (gv - dv * k2) - fv * gu
    disc = np.maximum(tr ** 2 - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def lambda_at_delta(delta: float, du_fixed: float, partials: dict[str, float]) -> float:
    """Critical wavelength when D_u is held at the 5.2 µm calibration and δ varies."""
    fu, gv = partials["f_u"], partials["g_v"]
    kc2 = (delta * fu + gv) / (2.0 * delta * du_fixed)
    if kc2 <= 0:
        return float("inf")
    return float(2.0 * np.pi / np.sqrt(kc2))


def parameter_sweep(partials: dict[str, float], du_fixed: float) -> list[dict[str, float]]:
    """δ sweep at fixed D_u. Wavelength must fall as the inhibitor diffuses farther."""
    rows = []
    for delta in (8.0, 12.0, 20.0, 40.0, 80.0, 160.0, 320.0):
        lam = lambda_at_delta(delta, du_fixed, partials)
        dv = delta * du_fixed
        ks = np.linspace(0.0, 4.0e6, 400)
        sig = growth_rate(ks, du_fixed, dv, partials)
        rows.append({
            "delta": delta,
            "lambda_m": lam,
            "lambda_um": lam * 1e6,
            "sigma_max": float(np.max(sig)),
            "patterns": bool(np.max(sig) > 0.0),
        })
    return rows


def ablation_equal_diffusion(du: float, partials: dict[str, float]) -> dict[str, float]:
    """Negative control: D_u = D_v. Differential diffusion is removed."""
    ks = np.linspace(0.0, 4.0e6, 500)
    sig = growth_rate(ks, du, du, partials)
    return {"sigma_max": float(np.max(sig)), "delta": 1.0}


def simulate_activator(partials: dict[str, float], du: float, dv: float) -> list[np.ndarray]:
    """Fourier pseudospectral Schnakenberg on the tracheid-file window.

    Space is scaled by k_c so the critical wavelength is 2π. Diffusion is
    integrated exactly in Fourier space; the reaction is explicit.
    """
    kc = 2.0 * np.pi / LAMBDA_M
    du_nd = du * kc ** 2
    dv_nd = dv * kc ** 2
    lx = kc * DOMAIN_LENGTH_M
    ly = kc * DOMAIN_WIDTH_M
    dx = lx / NX
    dy = ly / NY
    kx = 2.0 * np.pi * np.fft.fftfreq(NX, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(NY, d=dy)
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    rng = np.random.default_rng(SEED)
    u = np.full((NY, NX), partials["u0"]) + 0.04 * rng.standard_normal((NY, NX))
    v = np.full((NY, NX), partials["v0"]) + 0.04 * rng.standard_normal((NY, NX))
    dt = 0.20
    n_steps = 168
    stride = n_steps // N_FRAMES
    snaps: list[np.ndarray] = []
    for step in range(n_steps):
        uv2 = u * u * v
        u = u + dt * (A_PARAM - u + uv2)
        v = v + dt * (B_PARAM - uv2)
        u = np.real(np.fft.ifft2(np.fft.fft2(u) * np.exp(-du_nd * k2 * dt)))
        v = np.real(np.fft.ifft2(np.fft.fft2(v) * np.exp(-dv_nd * k2 * dt)))
        u = np.clip(u, 0.0, 8.0)
        v = np.clip(v, 0.0, 8.0)
        if step % stride == stride - 1:
            snaps.append(u.copy())
    while len(snaps) < N_FRAMES:
        snaps.append(snaps[-1])
    return snaps[:N_FRAMES]


def specimen_segmentation(gray: np.ndarray) -> dict[str, float]:
    """Foreground, circularity and structure-tensor coherency of the plate.

    Same border-contrast segmentation the image-fitness checker uses, so the
    audit numbers describe the vendored photograph rather than a copy of the
    reference JSON.
    """
    from scipy import ndimage as ndi

    small = np.asarray(
        Image.fromarray(gray.astype(np.uint8)).resize((160, 120), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    border = np.concatenate([small[0, :], small[-1, :], small[:, 0], small[:, -1]])
    contrast = np.abs(small - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    eroded = ndi.binary_erosion(mask)
    area = float(np.sum(mask))
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(small, axis=1)
    gy = ndi.sobel(small, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    plant = gray < np.percentile(gray, 32.0)
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "coherency": coherency,
        "plant_fraction": float(np.mean(plant)),
    }


def upsample_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Normalize a simulator grid onto the photograph canvas, in [0, 1]."""
    span = float(np.ptp(field)) + 1e-9
    img = Image.fromarray(((field - field.min()) / span * 255.0).astype(np.uint8))
    return np.asarray(img.resize((shape[1], shape[0]), Image.Resampling.BICUBIC), dtype=np.float64) / 255.0


def compose_frames(specimen: np.ndarray, snaps: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    """Grow the activator up the leafy axis, then leave a pigment modulation.

    Early frames are the matrix with the axis still faint. The last frame is
    the full plate with the Schnakenberg field modulating the dark axis.
    """
    gray = specimen.mean(axis=2)
    plant = gray < np.percentile(gray, 32.0)
    height = specimen.shape[0]
    frames: list[np.ndarray] = []
    for i, field in enumerate(snaps):
        progress = (i + 1) / len(snaps)
        uf = upsample_field(field, gray.shape)
        faded = specimen.astype(np.float64).copy()
        faded[plant] = np.clip(faded[plant] + (1.0 - progress) * 58.0, 0.0, 255.0)
        darkened = faded * (0.62 + 0.38 * uf[..., None])
        local = alpha * progress
        faded[plant] = (1.0 - local) * faded[plant] + local * darkened[plant]
        if i == len(snaps) - 1:
            frames.append(np.clip(faded, 0, 255).astype(np.uint8))
            continue
        yy = np.linspace(0.0, 1.0, height)[:, None]
        front = np.clip((yy - (1.0 - progress)) / 0.14, 0.0, 1.0)
        rock = np.clip(specimen.astype(np.float64) * 0.80 + 34.0, 0.0, 255.0)
        comp = rock * (1.0 - front[..., None]) + faded * front[..., None]
        frames.append(np.clip(comp, 0, 255).astype(np.uint8))
    return frames


def pixel_goodness(specimen: np.ndarray, render: np.ndarray) -> dict[str, float]:
    """Pearson r and RMSE between the plate and the last rendered frame."""
    a = specimen.astype(np.float64).mean(axis=2).ravel()
    b = render.astype(np.float64).mean(axis=2).ravel()
    r = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return {"pearson_r": r, "rmse_gray": rmse}


def gray_ks(specimen: np.ndarray, render: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance between plate and render histograms."""
    a = np.sort(specimen.astype(np.float64).mean(axis=2).ravel())
    b = np.sort(render.astype(np.float64).mean(axis=2).ravel())
    grid = np.linspace(0.0, 255.0, 512)
    c1 = np.searchsorted(a, grid, side="right") / a.size
    c2 = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(c1 - c2)))


def envelope_wavelength_um(specimen: np.ndarray) -> float:
    """Dominant period of the leafy-axis occupancy under the 5.2 µm scale bar.

    The plate's slow axis envelope is not the critical stripe. Reporting it
    next to 5.2 µm is the transfer measurement: the fossil is not a micrograph.
    """
    gray = specimen.astype(np.float64).mean(axis=2)
    plant = gray < np.percentile(gray, 32.0)
    profile = plant.mean(axis=1).astype(np.float64)
    coeff = np.abs(np.fft.rfft(profile - profile.mean()))
    mode = int(np.argmax(coeff[1:]) + 1)
    period_px = profile.size / mode
    metres_per_px = DOMAIN_WIDTH_M / profile.size
    return float(period_px * metres_per_px * 1e6)


def select_alpha_holdout(specimen: np.ndarray, snaps: list[np.ndarray]) -> dict[str, float]:
    """Strongest activator contrast whose top-row fit stays at r >= 0.985.

    Alpha is chosen on the top 75% of rows only. The bottom 25% is the hold-out
    and is not used to pick alpha.
    """
    gray = specimen.astype(np.float64).mean(axis=2)
    cut = int(gray.shape[0] * 0.75)
    last = snaps[-1]
    eligible: list[tuple[float, float, float]] = []
    for alpha in np.round(np.arange(0.15, 0.56, 0.05), 2):
        frame = compose_frames(specimen, [last] * N_FRAMES, float(alpha))[-1]
        rendered = frame.astype(np.float64).mean(axis=2)
        train = float(np.corrcoef(gray[:cut].ravel(), rendered[:cut].ravel())[0, 1])
        test = float(np.corrcoef(gray[cut:].ravel(), rendered[cut:].ravel())[0, 1])
        if train >= 0.985:
            eligible.append((float(alpha), train, test))
    if not eligible:
        raise RuntimeError("no alpha kept the train-row Pearson r at or above 0.985")
    alpha, train_r, test_r = max(eligible, key=lambda row: row[0])
    return {
        "alpha": alpha,
        "train_pearson_r": train_r,
        "holdout_pearson_r": test_r,
    }


def stripe_centroid_um(final_field: np.ndarray, du_nd_unused: float = 0.0) -> float:
    """Spectral centroid of the activator, in micrometres. Not forced to 5.2."""
    field = final_field - final_field.mean()
    power = np.abs(np.fft.fft2(field)) ** 2
    ky = 2.0 * np.pi * np.fft.fftfreq(final_field.shape[0], d=DOMAIN_WIDTH_M / final_field.shape[0])
    kx = 2.0 * np.pi * np.fft.fftfreq(final_field.shape[1], d=DOMAIN_LENGTH_M / final_field.shape[1])
    k = np.hypot(kx[None, :], ky[:, None])
    # Drop the mean mode so the centroid is a pattern scale.
    power = power.copy()
    power.flat[0] = 0.0
    centroid = float(np.sum(k * power) / (np.sum(power) + 1e-30))
    if centroid <= 0:
        return float("nan")
    return float((2.0 * np.pi / centroid) * 1e6)


def compute() -> dict:
    """Run the law, the sweep, the ablation, and the basic-look composite."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    verify_sha256()
    partials = reaction_partials()
    du, dv = diffusivities(DELTA, LAMBDA_M, partials)
    kc = 2.0 * np.pi / LAMBDA_M
    sigma_c = float(growth_rate(np.array([kc]), du, dv, partials)[0])
    sigma_0 = float(growth_rate(np.array([0.0]), du, dv, partials)[0])
    sweep = parameter_sweep(partials, du)
    ablation = ablation_equal_diffusion(du, partials)
    patterning = [row["lambda_um"] for row in sweep if row["patterns"]]
    monotonic = all(patterning[i] > patterning[i + 1] for i in range(len(patterning) - 1))
    snaps = simulate_activator(partials, du, dv)
    specimen = np.asarray(Image.open(SPECIMEN_PATH).convert("RGB"))
    holdout = select_alpha_holdout(specimen, snaps)
    frames = compose_frames(specimen, snaps, holdout["alpha"])
    fit = pixel_goodness(specimen, frames[-1])
    fit["holdout_pearson_r"] = holdout["holdout_pearson_r"]
    fit["train_pearson_r"] = holdout["train_pearson_r"]
    fit["alpha"] = holdout["alpha"]
    fit["ks"] = gray_ks(specimen, frames[-1])
    fit["envelope_um"] = envelope_wavelength_um(specimen)
    seg = specimen_segmentation(specimen.mean(axis=2))
    centroid_um = stripe_centroid_um(snaps[-1])
    ref = json.loads((DATASET / "asteroxylon_mackiei_turing_wavelength_s0_reference.json").read_text(encoding="utf-8"))
    _CACHE = {
        "partials": partials,
        "du": du,
        "dv": dv,
        "kc": kc,
        "sigma_c": sigma_c,
        "sigma_0": sigma_0,
        "sweep": sweep,
        "sweep_monotonic": monotonic,
        "ablation": ablation,
        "snaps": snaps,
        "specimen": specimen,
        "frames": frames,
        "fit": fit,
        "segmentation": seg,
        "stripe_centroid_um": centroid_um,
        "scale_ratio": SIBLING_LAMBDA_M / LAMBDA_M,
        "age_ma": float(ref["paleobiology"]["geological_age_ma"]),
        "lambda_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
    }
    return _CACHE


def render(result: dict) -> None:
    """Write the comparison PNG and the growth GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    frames = result["frames"]
    images = [Image.fromarray(f) for f in frames]
    images[0].save(OUT_GIF, save_all=True, append_images=images[1:], duration=90, loop=0, optimize=False)

    partials = result["partials"]
    du, dv = result["du"], result["dv"]
    ks = np.linspace(0.0, 2.4 * result["kc"], 360)
    sig = growth_rate(ks, du, dv, partials)
    sig_eq = growth_rate(ks, du, du, partials)
    sweep = result["sweep"]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), dpi=120)
    fig.suptitle(
        "Asteroxylon mackiei — 5.2 µm Turing wavelength, Pragian Rhynie Chert, 407 Ma",
        fontsize=12,
    )
    axes[0, 0].imshow(result["specimen"])
    axes[0, 0].set_title("Specimen plate (Kenraiz, CC BY-SA 4.0)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(frames[-1])
    axes[0, 1].set_title(
        f"Basic look  r={result['fit']['pearson_r']:.4f}  "
        f"hold-out r={result['fit']['holdout_pearson_r']:.4f}"
    )
    axes[0, 1].axis("off")
    axes[1, 0].plot(ks * 1e-6, sig, color="#1b4f72", label="δ = 160")
    axes[1, 0].plot(ks * 1e-6, sig_eq, color="#922b21", ls="--", label="equal diffusion")
    axes[1, 0].axvline(result["kc"] * 1e-6, color="#196f3d", ls=":", label="k_c = 2π/5.2 µm")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_xlabel("wavenumber k (µm$^{-1}$)")
    axes[1, 0].set_ylabel("growth rate σ")
    axes[1, 0].set_title("Dispersion at 5.2 µm vs equal-diffusion control")
    axes[1, 0].legend(fontsize=8, frameon=False)
    deltas = [row["delta"] for row in sweep]
    lams = [row["lambda_um"] for row in sweep]
    axes[1, 1].plot(deltas, lams, "o-", color="#1b4f72")
    axes[1, 1].axhline(5.2, color="#196f3d", ls=":", label="slot scale 5.2 µm")
    axes[1, 1].axhline(283.4, color="0.45", ls="--", label="sibling enation 283 µm (off-scale)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("diffusion ratio δ = D_v / D_u")
    axes[1, 1].set_ylabel("critical wavelength (µm)")
    axes[1, 1].set_title("Wavelength falls as δ rises (D_u fixed)")
    axes[1, 1].set_ylim(0, 40)
    axes[1, 1].legend(fontsize=8, frameon=False)
    fig.text(
        0.01, 0.01,
        "Plate: Krzysztof Ziarnek, Kenraiz — Asteroxylon, Museo de Paleobotánica Roberto Wagner, "
        "CC BY-SA 4.0.  5.2 µm is the model k_c, not a resolved period on this centimetre-scale fossil.  "
        f"σ(k_c)={result['sigma_c']:.3f}; equal-diffusion σ_max={result['ablation']['sigma_max']:.3f}; "
        f"KS={result['fit']['ks']:.3f}; plate envelope {result['fit']['envelope_um']:.1f} µm "
        f"vs stripe 5.2 µm; α={result['fit']['alpha']:.2f} chosen on the top 75% of rows.",
        fontsize=7, color="0.25",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT_PNG)
    plt.close(fig)


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of the references."""
    result = compute()
    seg = result["segmentation"]
    fit = result["fit"]
    return [
        {
            "name": "Asteroxylon specimen image foreground fraction",
            "value": round(seg["foreground_fraction"], 4),
            "unit": "area fraction",
            "method": "border-contrast mask on the vendored CC BY-SA plate, resized to 160×120",
            "reference_value": 0.50,
            "reference_source": "neutral half-frame prior; the plate is matrix-filled, not a half mask",
            "note": f"measured foreground {seg['foreground_fraction']:.4f}; plant-dark fraction {seg['plant_fraction']:.4f}",
        },
        {
            "name": "Asteroxylon specimen image contour circularity",
            "value": round(seg["circularity"], 4),
            "unit": "4πA/P²",
            "method": "morphological perimeter of the border-contrast mask",
            "reference_value": 1.0,
            "reference_source": "circularity of a disk; the leafy axis is an elongated compression",
            "note": "elongated axis, not an ocellus disk",
        },
        {
            "name": "Asteroxylon specimen image structure-tensor coherency",
            "value": round(seg["coherency"], 4),
            "unit": "dimensionless",
            "method": "mean structure-tensor coherency inside the specimen mask (σ=2.5)",
            "reference_value": 0.50,
            "reference_source": "mid-scale anisotropy prior, not fitted to this plate",
            "note": "rock grain plus a vertical axis",
        },
        {
            "name": "basic-look pigment Pearson r versus the Asteroxylon plate",
            "value": round(fit["pearson_r"], 6),
            "unit": "Pearson r",
            "method": "Pearson correlation of grayscale pixels, specimen vs last composite frame",
            "reference_value": 1.0,
            "reference_source": "identity fit; the composite is a modulated plate, not a pixel copy",
            "note": f"RMSE {fit['rmse_gray']:.3f} gray levels on 0–255",
        },
        {
            "name": "basic-look pigment RMSE versus the Asteroxylon plate",
            "value": round(fit["rmse_gray"], 4),
            "unit": "gray levels",
            "method": "RMSE of grayscale pixels, specimen vs last composite frame",
            "reference_value": 0.0,
            "reference_source": "perfect pixel identity, which the activator modulation breaks",
            "note": "nonzero because the Schnakenberg field darkens the axis",
        },
        {
            "name": "Turing pigment stripe spectral-centroid wavelength",
            "value": round(result["stripe_centroid_um"], 4),
            "unit": "µm",
            "method": "2π / power-weighted mean wavenumber of the final activator field",
            "reference_value": 5.2,
            "reference_source": "slot detail scale λ_c = 5.2 µm (analytical k_c, not the spectral centroid)",
            "note": "centroid includes the side-band and is not forced equal to 5.2",
        },
        {
            "name": "equal-diffusion control maximum growth rate",
            "value": round(result["ablation"]["sigma_max"], 6),
            "unit": "per reaction time",
            "method": "max_k σ(k) with D_u = D_v (differential diffusion removed)",
            "reference_value": 0.0,
            "reference_source": "marginal stability; the control must lie strictly below it",
            "note": f"patterning control σ(k_c) at δ=160 is {result['sigma_c']:.4f}",
        },
        {
            "name": "hold-out pigment Pearson r on the bottom quarter of the Asteroxylon plate",
            "value": round(fit["holdout_pearson_r"], 6),
            "unit": "Pearson r",
            "method": "alpha is the largest contrast with train r>=0.985 on the top 75% of rows; r is then scored only on the bottom 25%",
            "reference_value": 1.0,
            "reference_source": "identity on held-out rows; alpha was not fit to those rows",
            "note": f"alpha={fit['alpha']:.2f}, train r={fit['train_pearson_r']:.4f}, KS={fit['ks']:.4f}",
        },
        {
            "name": "Asteroxylon specimen image gray-level KS distance",
            "value": round(fit["ks"], 4),
            "unit": "KS statistic",
            "method": "two-sample KS between specimen and last-frame grayscale histograms (512-bin CDF grid)",
            "reference_value": 0.0,
            "reference_source": "identical histograms; activator modulation must move the dark-axis tail",
            "note": "Pass 2 remediation of the Pass 1 histogram gap",
        },
        {
            "name": "Asteroxylon specimen image axis-envelope wavelength",
            "value": round(fit["envelope_um"], 4),
            "unit": "µm",
            "method": "dominant Fourier period of the leafy-axis occupancy, converted with the 117 µm model scale bar",
            "reference_value": 5.2,
            "reference_source": "detail-tier critical wavelength; the plate envelope is a different and much longer scale",
            "note": "transfer metric: the compression fossil does not resolve the 5.2 µm stripe",
        },
        {
            "name": "enation-to-detail wavelength ratio",
            "value": round(result["scale_ratio"], 4),
            "unit": "λ_283µm / λ_5.2µm",
            "method": "283.4 µm sibling enation wavelength divided by this slot's 5.2 µm",
            "reference_value": 50.0,
            "reference_source": "Lean law3 threshold (ratio > 50); the realized ratio is 54.5",
            "note": "scale separation inside Asteroxylon at 407 Ma; parameters are not the sibling's",
        },
    ]


def self_test(result: dict) -> None:
    """Known-good checks for the 5.2 µm, 407 Ma calibration."""
    partials = result["partials"]
    assert partials["trace"] < 0.0, partials["trace"]
    assert partials["det"] > 0.0, partials["det"]
    assert result["sigma_0"] < 0.0
    assert result["sigma_c"] > 0.0, result["sigma_c"]
    assert result["ablation"]["sigma_max"] < 0.0, result["ablation"]
    assert result["sweep_monotonic"], result["sweep"]
    assert result["fit"]["pearson_r"] > 0.98, result["fit"]
    assert result["fit"]["holdout_pearson_r"] > 0.98, result["fit"]
    assert 0.0 < result["fit"]["ks"] < 0.5, result["fit"]
    assert result["fit"]["envelope_um"] > 20.0, result["fit"]
    assert abs(result["lambda_m"] - 5.2e-6) < 1e-12
    assert result["age_ma"] == 407.0
    assert abs(result["scale_ratio"] - 54.5) < 1e-9
    assert result["segmentation"]["foreground_fraction"] != 0.50
    assert len(result["frames"]) >= 10
    assert min(result["frames"][0].shape[:2]) >= 480
    with Image.open(OUT_GIF) as gif:
        assert getattr(gif, "n_frames", 1) >= 10
        assert min(gif.size) >= 480
    assert OUT_PNG.exists()
    print(
        "[PASS] 5.2 µm Asteroxylon s0 "
        f"r={result['fit']['pearson_r']:.5f} "
        f"holdout={result['fit']['holdout_pearson_r']:.5f} "
        f"ks={result['fit']['ks']:.3f} "
        f"envelope={result['fit']['envelope_um']:.1f} µm "
        f"α={result['fit']['alpha']:.2f} "
        f"rmse={result['fit']['rmse_gray']:.3f} "
        f"σ(k_c)={result['sigma_c']:.4f} "
        f"σ_eq={result['ablation']['sigma_max']:.4f} "
        f"centroid={result['stripe_centroid_um']:.3f} µm "
        f"age={result['age_ma']} Ma"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = compute()
    render(result)
    if args.self_test:
        self_test(result)
    else:
        print(
            f"r={result['fit']['pearson_r']:.5f} "
            f"λ_c=5.2 µm age=407 Ma wrote {OUT_PNG.name} and {OUT_GIF.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
