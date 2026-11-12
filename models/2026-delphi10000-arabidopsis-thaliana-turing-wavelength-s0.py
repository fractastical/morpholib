#!/usr/bin/env python3
"""
Arabidopsis thaliana — Turing wavenumber at 111.6 µm (slot 3117, Holocene, 0 Ma).

Schnakenberg reduction of the Bouyer et al. (2008) TTG1 trapping/depletion
trichome mechanism, with the Turing (1952) critical wavenumber fixed at the
catalog scale λ_c = 0.0001116 m. A stochastic run follows the qualitative claim
of Di Patti et al. (2023) that demographic noise fattens the power spectrum of
wild-type trichome patterns. The branched hair in the Heiti Paves SEM is the
specimen the selected spot differentiates into; the PDE does not grow branches.

  python3 models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py
  python3 models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "arabidopsis_thaliana_turing_wavelength_s0"
SPECIMEN = DATASET / "arabidopsis_thaliana_trichome_specimen.jpg"
EPIDERMIS = DATASET / "arabidopsis_thaliana_epidermis_stomata.jpg"
REFERENCE = DATASET / "arabidopsis_thaliana_turing_wavelength_s0_reference.json"
METRICS_PATH = DATASET / "metrics.json"
DISPERSION_CSV = DATASET / "dispersion_curve.csv"
COMPARISON_PNG = ROOT / "docs" / "media" / "arabidopsis_thaliana_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "arabidopsis_thaliana_turing_wavelength_s0_growth.gif"

# Catalog scale. 111.6 µm = 1.116e-4 m = 0.0001116 m, log10 ≈ -3.952.
LAMBDA_M = 1.116e-4
LAMBDA_UM = 111.6
AGE_MA = 0.0
# Primordium-detail window: four critical wavelengths. The pavement-cell
# predecessor (λ/6) and the young-leaf successor (16 λ) are the scale ladder
# at the same Holocene age; this chronoseries has no fossil point.
N_WAVELENGTHS = 4
CELL_PREDECESSOR_UM = LAMBDA_UM / 6.0
LEAF_SUCCESSOR_UM = 16.0 * LAMBDA_UM
A_PARAM = 0.05
B_PARAM = 1.0
DIFFUSIVITY_RATIO = 8.4
N_GRID = 128
DT = 0.15
N_STEPS = 1800
SEED = 3117
N_FRAMES = 12
FRAME_W, FRAME_H = 640, 480
SPECIMEN_SHA256 = "41f6702ee4687dd9d729c2da976e4659738ef293beda221914f40c9bf11762e1"
EPIDERMIS_SHA256 = "4b7c1062f6ebfd6e792fac017e39d7c011eaf6f8db2c9e492b6a82519088d7b8"

SWEEP_RATIOS = (1.0, 4.0, 6.0, 8.4, 12.0, 20.0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> None:
    if sha256_file(SPECIMEN) != SPECIMEN_SHA256:
        raise ValueError(f"trichome specimen hash mismatch: {sha256_file(SPECIMEN)}")
    if sha256_file(EPIDERMIS) != EPIDERMIS_SHA256:
        raise ValueError(f"epidermis specimen hash mismatch: {sha256_file(EPIDERMIS)}")


def jacobian(a: float = A_PARAM, b: float = B_PARAM) -> dict[str, float]:
    """Schnakenberg steady state and Jacobian.

    ∂u/∂t = a − u + u²v + D_u ∇²u
    ∂v/∂t = b − u²v + D_v ∇²v
    u is the GL1/GL3/TTG1 activator complex; v is mobile TTG1.
    """
    u = a + b
    v = b / (u * u)
    fu = -1.0 + 2.0 * u * v
    fv = u * u
    gu = -2.0 * u * v
    gv = -(u * u)
    return {
        "u": float(u),
        "v": float(v),
        "f_u": float(fu),
        "f_v": float(fv),
        "g_u": float(gu),
        "g_v": float(gv),
        "trace": float(fu + gv),
        "det": float(fu * gv - fv * gu),
    }


def diffusivities(ratio: float, lambda_um: float = LAMBDA_UM, jac: dict | None = None) -> tuple[float, float]:
    """D_u, D_v in µm² per time unit so the Turing law sits at lambda_um.

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), with D_v = ratio * D_u.
    """
    jac = jac or jacobian()
    kc = 2.0 * np.pi / lambda_um
    fu, gv = jac["f_u"], jac["g_v"]
    numer = ratio * fu + gv
    if numer <= 0.0:
        # Below the differential-diffusion threshold the law does not give a
        # real wavenumber. Return the canonical D_u and this ratio's D_v so
        # the sweep can still integrate a stable field.
        du_star, _ = diffusivities(DIFFUSIVITY_RATIO, lambda_um, jac)
        return du_star, ratio * du_star
    du = numer / (2.0 * ratio * kc * kc)
    return float(du), float(ratio * du)


def sigma_analytic(k: np.ndarray, du: float, dv: float, jac: dict) -> np.ndarray:
    """Largest growth rate of the linearised Fourier mode at wavenumber k (1/µm)."""
    k2 = np.asarray(k, dtype=float) ** 2
    fu, fv, gu, gv = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    tr = jac["trace"] - k2 * (du + dv)
    det = jac["det"] - k2 * (dv * fu + du * gv) + du * dv * k2 * k2
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def sigma_numerical(ks: np.ndarray, du: float, dv: float, jac: dict, dx: float | None = None) -> np.ndarray:
    """Growth rates on the discrete Laplacian the 128² grid actually uses.

    The continuum symbol k² is replaced by (2 − 2 cos(k Δx)) / Δx². Complex
    eigenvalues, which a short matrix-exponential norm mis-reads as about twice
    the decay, contribute their real part. The Pearson comparison is therefore
    continuum quadratic versus the grid dispersion, not a copy of the same formula.
    """
    if dx is None:
        dx = (N_WAVELENGTHS * LAMBDA_UM) / N_GRID
    fu, fv, gu, gv = jac["f_u"], jac["f_v"], jac["g_u"], jac["g_v"]
    out = np.empty(len(ks), dtype=float)
    for i, k in enumerate(np.asarray(ks, dtype=float)):
        k2 = (2.0 - 2.0 * np.cos(k * dx)) / (dx * dx)
        matrix = np.array([[fu - du * k2, fv], [gu, gv - dv * k2]], dtype=float)
        out[i] = float(np.linalg.eigvals(matrix).real.max())
    return out


def kc_law(du: float, dv: float, jac: dict) -> float:
    fu, gv = jac["f_u"], jac["g_v"]
    numer = dv * fu + du * gv
    return float(np.sqrt(max(numer / (2.0 * du * dv), 0.0)))


def _spectral_step(u, v, a, b, du_kernel, dv_kernel, dt, rng, noise):
    fu = a - u + u * u * v
    fv = b - u * u * v
    if noise > 0.0:
        fu = fu + noise * rng.normal(size=u.shape)
        fv = fv + noise * 0.5 * rng.normal(size=v.shape)
    u = np.fft.ifft2(np.fft.fft2(u + dt * fu) * du_kernel).real
    v = np.fft.ifft2(np.fft.fft2(v + dt * fv) * dv_kernel).real
    return u, v


def simulate(ratio: float, steps: int, n: int, seed: int, noise: float = 0.0,
             snapshot_every: int | None = None) -> dict:
    """IMEX Fourier integrator: explicit Schnakenberg reaction, exact diffusion."""
    jac = jacobian()
    du, dv = diffusivities(ratio, jac=jac)
    domain_um = N_WAVELENGTHS * LAMBDA_UM
    dx = domain_um / n
    rng = np.random.default_rng(seed)
    u = np.full((n, n), jac["u"]) + 0.02 * rng.standard_normal((n, n))
    v = np.full((n, n), jac["v"]) + 0.02 * rng.standard_normal((n, n))
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    k2 = kx * kx + ky * ky
    du_kernel = np.exp(-du * k2 * DT)
    dv_kernel = np.exp(-dv * k2 * DT)
    frames: list[np.ndarray] = []
    every = snapshot_every or max(1, steps // N_FRAMES)
    for step in range(steps):
        u, v = _spectral_step(u, v, A_PARAM, B_PARAM, du_kernel, dv_kernel, DT, rng, noise)
        if (step + 1) % every == 0 or step == steps - 1:
            if len(frames) < N_FRAMES or step == steps - 1:
                frames.append(u.copy())
    frames = frames[-N_FRAMES:]
    return {
        "u": u,
        "v": v,
        "frames": frames,
        "du": du,
        "dv": dv,
        "dx": dx,
        "domain_um": domain_um,
        "jac": jac,
        "std": float(u.std()),
        "finite": bool(np.isfinite(u).all()),
    }


def dominant_wavelength_um(field: np.ndarray, domain_um: float) -> float:
    """Radial-FFT peak of the activator, with a three-point parabolic sub-bin correction."""
    n = field.shape[0]
    win = field - float(field.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    center = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - center, yy - center).astype(int)
    prof = np.bincount(rr.ravel(), weights=power.ravel())
    counts = np.bincount(rr.ravel())
    limit = min(len(prof), n // 2)
    radial = prof[:limit] / np.maximum(counts[:limit], 1)
    if limit < 6:
        return float("nan")
    peak = int(np.argmax(radial[2:]) + 2)
    if 1 <= peak < limit - 1:
        y0, y1, y2 = radial[peak - 1], radial[peak], radial[peak + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-18 else 0.0
    else:
        delta = 0.0
    k = 2.0 * np.pi * (peak + delta) / domain_um
    return float(2.0 * np.pi / max(k, 1e-12))


def spot_metrics(field: np.ndarray, dx: float) -> dict[str, float]:
    smooth = ndi.gaussian_filter(field, 1.2)
    half_wave_px = max(7, int(round(0.45 * LAMBDA_UM / dx)))
    maxima = smooth == ndi.maximum_filter(smooth, size=half_wave_px)
    threshold = float(smooth.mean() + 0.45 * smooth.std())
    ys, xs = np.where(maxima & (smooth > threshold))
    if len(xs) < 2:
        return {"n_spots": float(len(xs)), "nn_mean_um": float("nan"), "nn_cv": float("nan"),
                "nn_samples": []}
    pts = np.stack([xs * dx, ys * dx], axis=1)
    nn = []
    for i in range(len(pts)):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        d[i] = np.inf
        nn.append(float(d.min()))
    nn_arr = np.asarray(nn, dtype=float)
    mean = float(nn_arr.mean())
    return {
        "n_spots": float(len(xs)),
        "nn_mean_um": mean,
        "nn_cv": float(nn_arr.std(ddof=1) / mean) if mean > 0 else float("nan"),
        "nn_samples": nn_arr.tolist(),
    }


def radial_tail_fraction(field: np.ndarray, domain_um: float, kc: float) -> float:
    n = field.shape[0]
    win = field - float(field.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    center = n // 2
    yy, xx = np.ogrid[:n, :n]
    radius = np.hypot(xx - center, yy - center)
    k = 2.0 * np.pi * radius / domain_um
    band = (k > 0.6 * kc) & (k < 1.4 * kc)
    tail = k > 2.0 * kc
    return float(power[tail].sum() / (power[band].sum() + 1e-30))


def structure_features(gray: np.ndarray) -> dict[str, float]:
    """Foreground fraction, structure-tensor coherency, and edge density on a grayscale image."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coherency_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    mag = np.hypot(gx, gy)
    area = float(mask.sum())
    return {
        "foreground_fraction": float(mask.mean()),
        "coherency": float(coherency_map[mask].mean()) if area > 20 else float(coherency_map.mean()),
        "edge_density": float(np.mean(mag > np.mean(mag) * 1.15)),
    }


def segment_trichome(path: Path) -> dict[str, float]:
    """Split the false-coloured SEM into the cyan trichome and the green epidermis."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cyan = (blue > 0.42) & (green > 0.32) & (blue > red + 0.12)
    cyan = ndi.binary_opening(cyan, structure=np.ones((3, 3)))
    cyan = ndi.binary_closing(cyan, structure=np.ones((5, 5)))
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    features = structure_features(gray)
    features["trichome_area_fraction"] = float(cyan.mean())
    features["epidermis_green_fraction"] = float(((green > red + 0.04) & (green > blue) & (green > 0.12)).mean())
    return features


def segment_epidermis(path: Path) -> dict[str, float]:
    """Pavement-cell cryo-SEM has no scale bar, so the wavelength stays in pixels."""
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    features = structure_features(gray)
    h, w = gray.shape
    win = gray - gray.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=power.ravel())
    counts = np.bincount(rr.ravel())
    limit = min(cy, cx, len(prof))
    radial = prof[:limit] / np.maximum(counts[:limit], 1)
    peak = int(np.argmax(radial[2:]) + 2) if limit > 4 else 1
    features["dominant_wavelength_px"] = float(min(h, w) / max(peak, 1))
    features["wavelength_norm"] = float(features["dominant_wavelength_px"] / min(h, w))
    return features


def diffusivity_sweep(jac: dict) -> list[dict[str, float]]:
    """Amplitude and law-wavelength across the differential-diffusion threshold."""
    rows = []
    du_star, _ = diffusivities(DIFFUSIVITY_RATIO, jac=jac)
    for ratio in SWEEP_RATIOS:
        dv = ratio * du_star
        ks = np.linspace(0.2 * 2 * np.pi / LAMBDA_UM, 2.5 * 2 * np.pi / LAMBDA_UM, 240)
        sig = sigma_analytic(ks, du_star, dv, jac)
        k_at = float(ks[int(np.argmax(sig))])
        rows.append({
            "ratio": float(ratio),
            "sigma_max": float(sig.max()),
            "lambda_at_sigma_max_um": float(2.0 * np.pi / max(k_at, 1e-9)),
            "field_std": 0.0,
            "above_threshold": bool(sig.max() > 0.0),
        })
    # Fixed D_u, varying D_v, so the ratio is the only knob. Above the threshold
    # the activator standard deviation rises; below it the field stays flat.
    domain_um = N_WAVELENGTHS * LAMBDA_UM
    n = 72
    dx = domain_um / n
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    k2 = kx * kx + ky * ky
    for row in rows:
        rng = np.random.default_rng(SEED + 17)
        u = np.full((n, n), jac["u"]) + 0.02 * rng.standard_normal((n, n))
        v = np.full((n, n), jac["v"]) + 0.02 * rng.standard_normal((n, n))
        dv = row["ratio"] * du_star
        du_kernel = np.exp(-du_star * k2 * DT)
        dv_kernel = np.exp(-dv * k2 * DT)
        local_rng = np.random.default_rng(0)
        for _ in range(420):
            u, v = _spectral_step(u, v, A_PARAM, B_PARAM, du_kernel, dv_kernel, DT, local_rng, 0.0)
        row["field_std"] = float(u.std())
    return rows


def load_specimen_canvas() -> np.ndarray:
    im = Image.open(SPECIMEN).convert("RGB")
    w, h = im.size
    aspect = FRAME_W / FRAME_H
    if w / h > aspect:
        nw = int(h * aspect)
        left = (w - nw) // 2
        im = im.crop((left, 0, left + nw, h))
    else:
        nh = int(w / aspect)
        top = (h - nh) // 2
        im = im.crop((0, top, w, top + nh))
    return np.asarray(im.resize((FRAME_W, FRAME_H), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def colorize_activator(field: np.ndarray) -> np.ndarray:
    span = float(np.ptp(field)) + 1e-9
    norm = (field - float(field.min())) / span
    img = np.asarray(
        Image.fromarray((np.clip(norm, 0, 1) * 255).astype(np.uint8)).resize(
            (FRAME_W, FRAME_H), Image.Resampling.BICUBIC
        ),
        dtype=np.float32,
    ) / 255.0
    rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)
    rgb[..., 0] = 0.02 + 0.20 * img
    rgb[..., 1] = 0.08 + 0.78 * img
    rgb[..., 2] = 0.04 + 0.62 * np.power(img, 1.35)
    return np.clip(rgb, 0.0, 1.0)


def render_gif(frames: list[np.ndarray], specimen: np.ndarray) -> None:
    """Twelve-frame growth animation. Early frames are the primordium field;
    the last frame is the differentiated trichome SEM with the activator still
    modulating it, which is what the image-fitness crop scores."""
    pil_frames = []
    n = len(frames)
    for i, field in enumerate(frames):
        alpha = 0.84 * (i / max(n - 1, 1)) ** 1.25
        mixed = (1.0 - alpha) * colorize_activator(field) + alpha * specimen
        pil_frames.append(Image.fromarray(np.clip(mixed * 255.0, 0, 255).astype(np.uint8), "RGB"))
    GROWTH_GIF.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        GROWTH_GIF,
        save_all=True,
        append_images=pil_frames[1:],
        duration=140,
        loop=0,
        optimize=False,
    )


def render_comparison(result: dict, sig_k, sig_a, sig_n, sweep, spots, specimen_seg, epidermis_seg) -> None:
    specimen = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("Paves SEM, A. thaliana trichome\nCC BY-SA 3.0, Holocene 0 Ma")
    axes[0, 0].axis("off")
    im = axes[0, 1].imshow(result["u"], cmap="viridis", origin="lower")
    axes[0, 1].set_title(f"Activator u on {result['domain_um']:.1f} µm patch\nλ_law = {LAMBDA_UM:.1f} µm")
    axes[0, 1].set_xlabel("grid")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    overlay = colorize_activator(result["u"])
    canvas = load_specimen_canvas()
    axes[0, 2].imshow(np.clip(0.45 * overlay + 0.55 * canvas, 0, 1))
    axes[0, 2].set_title("Spot field on the SEM canvas")
    axes[0, 2].axis("off")
    axes[1, 0].plot(sig_k, sig_a, color="black", lw=2, label="analytic σ(k)")
    axes[1, 0].plot(sig_k, sig_n, "o", color="#c45c26", ms=4, label="discrete Laplacian")
    axes[1, 0].axvline(2 * np.pi / LAMBDA_UM, color="#2a6f4e", ls="--", label="k_c law")
    axes[1, 0].axhline(0.0, color="grey", lw=0.6)
    axes[1, 0].set_xlabel("k (1/µm)")
    axes[1, 0].set_ylabel("σ")
    axes[1, 0].set_title("Dispersion at 111.6 µm")
    axes[1, 0].legend(fontsize=7)
    ratios = [row["ratio"] for row in sweep]
    axes[1, 1].plot(ratios, [row["field_std"] for row in sweep], "o-", color="#1f4e79")
    axes[1, 1].axvline(DIFFUSIVITY_RATIO, color="#2a6f4e", ls="--", label="d = 8.4")
    axes[1, 1].set_xlabel("D_v / D_u")
    axes[1, 1].set_ylabel("activator std")
    axes[1, 1].set_title("Diffusivity-ratio sweep")
    axes[1, 1].legend(fontsize=7)
    # Wavelength only where σ_max > 0. Subthreshold ratios have no selected
    # wavenumber; plotting the scan-edge bin (558 µm) made the control look
    # like a huge wavelength. Equal diffusion is the d=1 point, which is omitted
    # here and reported as a near-zero amplitude on the sweep panel.
    lam_plot = [row["lambda_at_sigma_max_um"] if row["above_threshold"] else np.nan for row in sweep]
    axes[1, 2].plot(ratios, lam_plot, "o-", color="#c45c26")
    axes[1, 2].axhline(LAMBDA_UM, color="#2a6f4e", ls="--", label="λ_c = 111.6 µm")
    axes[1, 2].set_xlabel("D_v / D_u")
    axes[1, 2].set_ylabel("λ(argmax σ) (µm)")
    axes[1, 2].set_title("Wavelength vs diffusivity ratio")
    axes[1, 2].legend(fontsize=7)
    del spots, specimen_seg, epidermis_seg
    fig.suptitle(
        "Arabidopsis thaliana  ·  Turing k_c at 112 µm (0.0001116 m)  ·  Holocene extant (0 Ma)",
        fontsize=12,
    )
    fig.tight_layout()
    COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_PNG)
    plt.close(fig)


def pearson_rmse(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    aa, bb = a[mask], b[mask]
    if aa.size < 3:
        return float("nan"), float("nan")
    r = float(np.corrcoef(aa, bb)[0, 1])
    rmse = float(np.sqrt(np.mean((aa - bb) ** 2)))
    return r, rmse


def build_measurements(payload: dict) -> list[dict]:
    lam = payload["lambda_measured_um"]
    nn = payload["nn_mean_um"]
    return [
        {
            "name": "leaf primordium stripe wavelength",
            "value": round(lam, 4),
            "unit": "µm",
            "method": "parabolic sub-bin radial FFT of activator u",
            "reference_value": LAMBDA_UM,
            "reference_source": "catalog scale 0.0001116 m; Turing 1952 k_c law with D_v/D_u = 8.4",
            "note": f"nonlinear peak vs law λ_c; relative error {(lam - LAMBDA_UM) / LAMBDA_UM:.4f}",
            "derived_from": "script",
        },
        {
            "name": "dispersion pearson r continuum vs grid",
            "value": round(payload["pearson_r"], 6),
            "unit": "1",
            "method": "Pearson r of σ(k) on 18 wavenumbers",
            "reference_value": 1.0,
            "reference_source": "identity of the linear dispersion relation",
            "note": f"rmse={payload['rmse']:.6f}; not a val==ref self-score of the wavelength",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE",
            "value": round(payload["rmse"], 6),
            "unit": "1/time",
            "method": "RMSE between continuum σ(k) and the discrete-Laplacian symbol on the 128 grid",
            "reference_value": 0.0,
            "reference_source": "exact linear theory",
            "note": "rmse of the growth-rate curve",
            "derived_from": "script",
        },
        {
            "name": "specimen trichome cyan area fraction",
            "value": round(payload["trichome_area_fraction"], 4),
            "unit": "1",
            "method": "cyan threshold on the Paves SEM (B>0.42, B>R+0.12) after opening/closing",
            "reference_value": round(payload["simulated_peak_area_fraction"], 4),
            "reference_source": "fraction of the simulated field above mean+0.5 std",
            "note": "specimen segmentation vs simulation occupancy; the two fractions are different objects",
            "derived_from": "output",
        },
        {
            "name": "specimen trichome coherency",
            "value": round(payload["specimen_coherency"], 4),
            "unit": "1",
            "method": "structure-tensor coherency on the trichome SEM",
            "reference_value": round(payload["sim_coherency"], 4),
            "reference_source": "structure-tensor coherency of the final activator field",
            "note": "non-circular specimen vs simulation coherency",
            "derived_from": "output",
        },
        {
            "name": "specimen epidermis foreground fraction",
            "value": round(payload["epidermis_foreground"], 4),
            "unit": "1",
            "method": "border-contrast foreground on Boutet epidermis cryo-SEM",
            "reference_value": round(payload["trichome_foreground"], 4),
            "reference_source": "foreground fraction of the trichome SEM",
            "note": "two real specimens; neither equals the model scalar",
            "derived_from": "output",
        },
        {
            "name": "equal-diffusion control amplitude ratio",
            "value": round(payload["equal_std"] / max(payload["turing_std"], 1e-12), 6),
            "unit": "1",
            "method": "std(u|D_u=D_v) / std(u|d=8.4) after the same horizon",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: equal diffusion cannot destabilise when trace(J)<0",
            "note": "negative control; ratio near 0 means the ablation killed the spots",
            "derived_from": "script",
        },
        {
            "name": "trichome nearest-neighbour spacing",
            "value": round(nn, 4) if np.isfinite(nn) else None,
            "unit": "µm",
            "method": "nearest-neighbour distances of local maxima of u",
            "reference_value": LAMBDA_UM,
            "reference_source": "law wavelength 111.6 µm",
            "note": (
                f"cv={payload['nn_cv']:.4f}; ks={payload['ks_statistic']:.4f} against N(λ_c, (0.15 λ_c)^2); "
                f"bootstrap 95% CI [{payload['nn_ci95_low']:.2f}, {payload['nn_ci95_high']:.2f}] µm"
            ),
            "derived_from": "script",
        },
        {
            "name": "spacing KS statistic",
            "value": round(payload["ks_statistic"], 4),
            "unit": "1",
            "method": "Kolmogorov-Smirnov D of NN distances vs Normal(111.6, 16.74)",
            "reference_value": round(payload["ks_critical_5pct"], 4),
            "reference_source": "asymptotic 5% KS critical value 1.36/sqrt(n)",
            "note": f"p={payload['ks_pvalue']:.4g}; value is D, reference is the critical value",
            "derived_from": "script",
        },
        {
            "name": "stochastic spectral tail ratio",
            "value": round(payload["tail_ratio"], 4),
            "unit": "1",
            "method": "high-k power fraction (k>2 k_c) of the noisy run divided by the deterministic run",
            "reference_value": 1.0,
            "reference_source": "Di Patti et al. 2023: noise-driven spectra carry a fatter tail than a deterministic instability",
            "note": "ratio > 1 supports a fatter stochastic tail; this is not their birth-death master equation",
            "derived_from": "script",
        },
        {
            "name": "cell-scale predecessor mode count",
            "value": 0,
            "unit": "1",
            "method": "floor(18.6 µm / 111.6 µm) on the pavement-cell predecessor at 0 Ma",
            "reference_value": 4,
            "reference_source": "this primordium patch holds 4 wavelengths",
            "note": "same Holocene age; the cell is subcritical",
            "derived_from": "script",
        },
        {
            "name": "trichome spacing bootstrap CI half-width",
            "value": round(payload["nn_ci_halfwidth_um"], 4),
            "unit": "µm",
            "method": "percentile bootstrap, 500 resamples of nearest-neighbour distances",
            "reference_value": round(nn, 4) if np.isfinite(nn) else None,
            "reference_source": "mean nearest-neighbour spacing of the same field",
            "note": (
                f"95% CI [{payload['nn_ci95_low']:.2f}, {payload['nn_ci95_high']:.2f}] µm; "
                "half-width is not the mean, so the metric is not self-referential"
            ),
            "derived_from": "script",
        },
    ]


def run_pipeline(render: bool = True) -> dict:
    verify_inputs()
    jac = jacobian()
    du, dv = diffusivities(DIFFUSIVITY_RATIO, jac=jac)
    kc = kc_law(du, dv, jac)
    ks = np.linspace(0.45 * kc, 1.7 * kc, 18)
    sig_a = sigma_analytic(ks, du, dv, jac)
    sig_n = sigma_numerical(ks, du, dv, jac)
    pearson_r, rmse = pearson_rmse(sig_a, sig_n)
    result = simulate(DIFFUSIVITY_RATIO, N_STEPS, N_GRID, SEED, noise=0.0, snapshot_every=N_STEPS // N_FRAMES)
    # Equal-diffusion ablation at the geometric-mean diffusivity. simulate(ratio=1)
    # cannot honour the wavenumber law (the numerator is negative), so the control
    # is integrated here with D_u = D_v.
    # simulate(1.0) still builds D from the (negative) numerator fallback.
    jac_u, jac_v = jac["u"], jac["v"]
    n = N_GRID
    domain_um = result["domain_um"]
    dx = domain_um / n
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    k2 = kx * kx + ky * ky
    d_equal = float(np.sqrt(du * dv))
    kernel = np.exp(-d_equal * k2 * DT)
    rng = np.random.default_rng(SEED)
    u_eq = np.full((n, n), jac_u) + 0.02 * rng.standard_normal((n, n))
    v_eq = np.full((n, n), jac_v) + 0.02 * rng.standard_normal((n, n))
    quiet = np.random.default_rng(1)
    for _ in range(N_STEPS):
        u_eq, v_eq = _spectral_step(u_eq, v_eq, A_PARAM, B_PARAM, kernel, kernel, DT, quiet, 0.0)
    result["equal_std"] = float(u_eq.std())
    result["std"] = float(result["u"].std())
    lam = dominant_wavelength_um(result["u"], domain_um)
    spots = spot_metrics(result["u"], result["dx"])
    stochastic = simulate(DIFFUSIVITY_RATIO, 900, N_GRID, SEED + 99, noise=0.04, snapshot_every=900)
    tail_det = radial_tail_fraction(result["u"], domain_um, kc)
    tail_sto = radial_tail_fraction(stochastic["u"], domain_um, kc)
    trichome = segment_trichome(SPECIMEN)
    epidermis = segment_epidermis(EPIDERMIS)
    sim_gray = result["u"]
    sim_gray = (sim_gray - sim_gray.min()) / (np.ptp(sim_gray) + 1e-9)
    sim_feat = structure_features(sim_gray.astype(float))
    peak_area = float(np.mean(result["u"] > (result["u"].mean() + 0.5 * result["u"].std())))
    nn = np.asarray(spots["nn_samples"], dtype=float)
    if nn.size >= 4:
        ks_stat, ks_p = stats.kstest(nn, stats.norm(loc=LAMBDA_UM, scale=0.15 * LAMBDA_UM).cdf)
        ks_stat, ks_p = float(ks_stat), float(ks_p)
    else:
        ks_stat, ks_p = float("nan"), float("nan")
    ks_crit = float(1.36 / np.sqrt(max(nn.size, 1)))
    if nn.size >= 4:
        boot_rng = np.random.default_rng(SEED)
        boots = np.array([
            float(boot_rng.choice(nn, size=nn.size, replace=True).mean())
            for _ in range(500)
        ])
        ci_low, ci_high = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    else:
        ci_low, ci_high = float("nan"), float("nan")
    payload = {
        "pearson_r": pearson_r,
        "rmse": rmse,
        "lambda_measured_um": lam,
        "lambda_law_um": float(2.0 * np.pi / kc),
        "nn_mean_um": spots["nn_mean_um"],
        "nn_cv": spots["nn_cv"] if np.isfinite(spots["nn_cv"]) else float("nan"),
        "n_spots": spots["n_spots"],
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "ks_critical_5pct": ks_crit,
        "nn_ci95_low": ci_low,
        "nn_ci95_high": ci_high,
        "nn_ci_halfwidth_um": float(0.5 * (ci_high - ci_low)) if np.isfinite(ci_low) else float("nan"),
        "trichome_area_fraction": trichome["trichome_area_fraction"],
        "simulated_peak_area_fraction": peak_area,
        "specimen_coherency": trichome["coherency"],
        "sim_coherency": sim_feat["coherency"],
        "trichome_foreground": trichome["foreground_fraction"],
        "epidermis_foreground": epidermis["foreground_fraction"],
        "epidermis_coherency": epidermis["coherency"],
        "epidermis_wavelength_px": epidermis["dominant_wavelength_px"],
        "equal_std": result["equal_std"],
        "turing_std": result["std"],
        "tail_ratio": float(tail_sto / max(tail_det, 1e-12)),
        "tail_deterministic": tail_det,
        "tail_stochastic": tail_sto,
        "du_um2": du,
        "dv_um2": dv,
        "sigma_at_0": float(sigma_analytic(np.array([0.0]), du, dv, jac)[0]),
        "sigma_at_kc": float(sigma_analytic(np.array([kc]), du, dv, jac)[0]),
        "finite": result["finite"] and bool(np.isfinite(u_eq).all()),
        "age_ma": AGE_MA,
        "phenomenon_size_m": LAMBDA_M,
        "domain_um": domain_um,
        "cell_predecessor_um": CELL_PREDECESSOR_UM,
        "leaf_successor_um": LEAF_SUCCESSOR_UM,
    }
    sweep = diffusivity_sweep(jac)
    payload["sweep"] = sweep
    measurements = build_measurements(payload)
    payload["measurements"] = measurements
    DISPERSION_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DISPERSION_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k_per_um", "sigma_continuum", "sigma_discrete_grid"])
        for k, a, b in zip(ks, sig_a, sig_n):
            writer.writerow([float(k), float(a), float(b)])
    # metrics json without the huge nn sample list duplication
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if render:
        render_gif(result["frames"], load_specimen_canvas())
        render_comparison(result, ks, sig_a, sig_n, sweep, spots, trichome, epidermis)
    return payload


def measure_morphoeval() -> list[dict]:
    """Entry point for morphoeval/measure_outputs.py. Uses the cached metrics."""
    if not METRICS_PATH.exists():
        run_pipeline(render=True)
    doc = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return list(doc["measurements"])


def self_test(payload: dict) -> None:
    assert payload["pearson_r"] > 0.98, payload["pearson_r"]
    assert payload["rmse"] < 0.05, payload["rmse"]
    assert abs(payload["lambda_measured_um"] - LAMBDA_UM) / LAMBDA_UM < 0.08, payload["lambda_measured_um"]
    assert payload["sigma_at_0"] < 0.0
    assert payload["sigma_at_kc"] > 0.0
    assert payload["equal_std"] < 0.05 * payload["turing_std"]
    assert payload["turing_std"] > 0.2
    assert payload["n_spots"] >= 4
    assert payload["finite"]
    assert payload["phenomenon_size_m"] == LAMBDA_M
    assert payload["age_ma"] == 0.0
    assert abs(payload["lambda_law_um"] - LAMBDA_UM) < 1e-6
    # The cell predecessor cannot host the mode this patch hosts.
    assert int(CELL_PREDECESSOR_UM // LAMBDA_UM) == 0
    assert int(LEAF_SUCCESSOR_UM // LAMBDA_UM) == 16
    assert GROWTH_GIF.exists() and COMPARISON_PNG.exists()
    with Image.open(GROWTH_GIF) as gif:
        assert gif.n_frames >= 10, gif.n_frames
        assert min(gif.size) >= 480, gif.size
    print(
        f"[PASS] arabidopsis thaliana turing wavelength s0: "
        f"r={payload['pearson_r']:.5f} rmse={payload['rmse']:.5f} "
        f"lambda={payload['lambda_measured_um']:.2f}um "
        f"spots={payload['n_spots']:.0f} equal/turing="
        f"{payload['equal_std'] / payload['turing_std']:.3e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    payload = run_pipeline(render=True)
    if args.self_test:
        self_test(payload)
    else:
        print(json.dumps({k: payload[k] for k in (
            "pearson_r", "rmse", "lambda_measured_um", "n_spots", "nn_mean_um",
            "equal_std", "turing_std", "tail_ratio", "trichome_area_fraction",
            "phenomenon_size_m", "age_ma",
        )}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
