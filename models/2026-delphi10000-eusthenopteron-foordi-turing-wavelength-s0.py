#!/usr/bin/env python3
"""Eusthenopteron foordi at the 104 µm Turing wavelength (slot 3225, detail s0).

Late Devonian (Frasnian, 379 Ma). Chronoseries 1/1 of the tetrapod fin-to-limb series.
The phenomenon is the fundamental pigment / ocellus wavelength λ = 104.2 µm
(0.0001042 m, log10 = −3.982, tier 5_primordium_100um), not the 5.7 mm organ-scale
flank bar of models/1980-jarvik-eusthenopteron-foordi-turing-wavelength.py.

Governing system (Turing 1952 wavenumber; Schnakenberg 1979 kinetics), time already
rescaled so the reaction coefficients are O(1) and k_c is unchanged:

    ∂u/∂τ = D_u ∇²u + a − u + u² v
    ∂v/∂τ = D_v ∇²v + b − u² v
    k_c²  = (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c   = 2π / k_c = 104.2 µm

Steady state u0 = a + b, v0 = b/(a+b)².
Jacobian: f_u = −1 + 2 u0 v0, f_v = u0², g_u = −2 u0 v0, g_v = −u0².

Boundary: periodic Fourier patch. The fossil photograph is a display mask.
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

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "eusthenopteron_foordi_turing_wavelength_s0"
SPECIMEN = DATASET / "eusthenopteron_foordi_specimen.jpg"
REFERENCE = DATASET / "eusthenopteron_foordi_turing_wavelength_s0_reference.json"
METRICS = DATASET / "metrics.json"
PNG = ROOT / "docs" / "media" / "eusthenopteron_foordi_turing_wavelength_s0_comparison.png"
GIF = ROOT / "docs" / "media" / "eusthenopteron_foordi_turing_wavelength_s0_growth.gif"
GALLERY = ROOT / "datasets" / "species_gallery" / "I_Enaima_Vertebrata" / "Eusthenopteron_foordi.jpg"

# Length unit = one formula wavelength. Physical λ_c is assigned, not fitted to the photo.
LAMBDA_M = 1.042e-4
LAMBDA_UM = 104.2
AGE_MA = 379.0
A_PARAM = 0.1
B_PARAM = 0.9
DV_OVER_DU = 12.0
SEED = 3225

# Unscaled diffusivities whose onset k_c, after dividing out the reaction rate,
# sits at 2π per length unit (see _rescaled_diffusivities).
_DU_UNSCALED = 1.0
_DV_UNSCALED = 12.0


def _jacobian():
    u0 = A_PARAM + B_PARAM
    v0 = B_PARAM / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return u0, v0, fu, fv, gu, gv


def _rescaled_diffusivities(dv_over_du: float = DV_OVER_DU):
    """Return (Du, Dv, gamma) in reaction-time units with formula k_c = 2π."""
    u0, v0, fu, fv, gu, gv = _jacobian()
    du_u, dv_u = _DU_UNSCALED, _DU_UNSCALED * dv_over_du
    kc2 = (dv_u * fu + du_u * gv) / (2.0 * du_u * dv_u)
    gamma = (2.0 * np.pi) ** 2 / kc2
    return du_u / gamma, dv_u / gamma, float(gamma)


def verify_specimen() -> str:
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expect = ref["specimen"]["sha256"]
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    if digest != expect:
        raise ValueError(f"specimen SHA-256 {digest} != {expect}")
    if abs(float(ref["phenomenon_scale"]["phenomenon_size_m"]) - LAMBDA_M) > 1e-12:
        raise ValueError("reference phenomenon size is not 0.0001042 m")
    if float(ref["paleobiology"]["geological_age_ma"]) != AGE_MA:
        raise ValueError("reference age is not 379 Ma")
    return digest


def _wavenumbers(nx: int, ny: int, dx: float, dy: float):
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx, ky = np.meshgrid(kx, ky)
    return kx ** 2 + ky ** 2


def schnakenberg_step(u, v, k2, du, dv, dt):
    """One Fourier-diffusion + explicit Schnakenberg reaction step. Eq. in module doc."""
    u = np.fft.ifft2(np.fft.fft2(u) * np.exp(-du * k2 * dt)).real
    v = np.fft.ifft2(np.fft.fft2(v) * np.exp(-dv * k2 * dt)).real
    uu = u * u
    u = u + dt * (A_PARAM - u + uu * v)
    v = v + dt * (B_PARAM - uu * v)
    return u, v


def simulate_pattern(du, dv, nx=128, ny=80, lx=12.0, ly=7.5, steps=1400, dt=0.04, seed=SEED, snaps=0):
    u0, v0, *_ = _jacobian()
    dx, dy = lx / nx, ly / ny
    k2 = _wavenumbers(nx, ny, dx, dy)
    rng = np.random.default_rng(seed)
    u = np.full((ny, nx), u0) + rng.normal(0.0, 0.03, (ny, nx))
    v = np.full((ny, nx), v0) + rng.normal(0.0, 0.03, (ny, nx))
    frames = []
    # Integer indices are strictly increasing, so the GIF cannot collapse below `snaps`.
    snap_at = []
    if snaps:
        snap_at = [int(i * (steps - 1) / (snaps - 1)) for i in range(snaps)]
    snap_set = set(snap_at)
    for n in range(steps):
        if n in snap_set:
            frames.append(u.copy())
        u, v = schnakenberg_step(u, v, k2, du, dv, dt)
        if not np.isfinite(u).all():
            raise RuntimeError(f"Schnakenberg blew up at step {n}")
    if snaps and frames:
        frames[-1] = u.copy()
    return u, frames, dx, dy


def simulate_equal_diffusion(steps=1400, dt=0.04):
    """Negative control: Du = Dv removes the Turing instability."""
    du, dv, _ = _rescaled_diffusivities()
    dbar = 0.5 * (du + dv)
    u, _, _, _ = simulate_pattern(dbar, dbar, steps=steps, dt=dt, seed=SEED + 1, snaps=0)
    return u


def simulate_ablation(steps=1400, dt=0.04):
    """Ablation: drop u²v autocatalysis. Fields only relax toward (u0, v0)."""
    u0, v0, *_ = _jacobian()
    du, dv, _ = _rescaled_diffusivities()
    nx, ny, lx, ly = 128, 80, 12.0, 7.5
    dx, dy = lx / nx, ly / ny
    k2 = _wavenumbers(nx, ny, dx, dy)
    rng = np.random.default_rng(SEED + 2)
    u = np.full((ny, nx), u0) + rng.normal(0.0, 0.03, (ny, nx))
    v = np.full((ny, nx), v0) + rng.normal(0.0, 0.03, (ny, nx))
    for _ in range(steps):
        u = np.fft.ifft2(np.fft.fft2(u) * np.exp(-du * k2 * dt)).real
        v = np.fft.ifft2(np.fft.fft2(v) * np.exp(-dv * k2 * dt)).real
        u = u + dt * (-(u - u0))
        v = v + dt * (-(v - v0))
    return u


def dominant_frequency(field: np.ndarray, dx: float, dy: float) -> float:
    """Cycles per length-unit of the radial power peak, DC excluded."""
    power = np.abs(np.fft.fft2(field - np.mean(field))) ** 2
    fy = np.fft.fftfreq(field.shape[0], d=dy)
    fx = np.fft.fftfreq(field.shape[1], d=dx)
    radius = np.hypot(fx[None, :], fy[:, None])
    bins = np.linspace(0.25, 2.4, 86)
    idx = np.digitize(radius.ravel(), bins)
    flat = power.ravel()
    prof = np.zeros(len(bins) - 1)
    for i in range(1, len(bins)):
        sel = idx == i
        if np.any(sel):
            prof[i - 1] = flat[sel].mean()
    peak = int(np.argmax(prof))
    return float(0.5 * (bins[peak] + bins[peak + 1]))


def dispersion_curves(n_k: int = 12):
    """Analytical Re(σ) versus a linearized Fourier step. Real-eigenvalue band only.

    Below k ≈ 3.6 the Jacobian eigenvalues are a complex conjugate pair (oscillatory
    decay). The amplitude is not a pure exponential there, so those k are excluded
    from the Pearson comparison and reported separately.
    """
    du, dv, _ = _rescaled_diffusivities()
    _, _, fu, fv, gu, gv = _jacobian()
    ks = np.linspace(3.8, 11.2, n_k)
    analytical = []
    numerical = []
    for k in ks:
        matrix = np.array(
            [[fu - du * k * k, fv], [gu, gv - dv * k * k]],
            dtype=float,
        )
        eigvals, right, left = _eig(matrix)
        i_slow = int(np.argmax(np.real(eigvals)))
        analytical.append(float(np.real(eigvals[i_slow])))
        numerical.append(_sigma_numerical(k, du, dv, fu, fv, gu, gv, right[:, i_slow], left[i_slow]))
    analytical = np.asarray(analytical)
    numerical = np.asarray(numerical)
    pearson = float(np.corrcoef(analytical, numerical)[0, 1])
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    return {
        "k": ks.tolist(),
        "sigma_analytical": analytical.tolist(),
        "sigma_numerical": numerical.tolist(),
        "pearson_r": pearson,
        "rmse": rmse,
    }


def _eig(matrix: np.ndarray):
    eigvals, right = np.linalg.eig(matrix)
    left = np.linalg.inv(right)
    return eigvals, right, left


def _sigma_numerical(k, du, dv, fu, fv, gu, gv, right_vec, left_vec) -> float:
    nmode = 4
    length = 2.0 * np.pi * nmode / k
    nx = 160
    dx = length / nx
    x = np.arange(nx) * dx
    vec = np.real(right_vec)
    vec = vec / (np.linalg.norm(vec) + 1e-15)
    lvec = np.real(left_vec)
    lvec = lvec / (np.dot(lvec, vec) + 1e-15)
    u0, v0, *_ = _jacobian()
    eps = 1e-3
    u = u0 + eps * vec[0] * np.cos(k * x)
    v = v0 + eps * vec[1] * np.cos(k * x)
    k2 = (2.0 * np.pi * np.fft.fftfreq(nx, d=dx)) ** 2
    dt = 0.005

    def step(u, v, nsteps):
        for _ in range(nsteps):
            u = np.fft.ifft(np.fft.fft(u) * np.exp(-du * k2 * dt)).real
            v = np.fft.ifft(np.fft.fft(v) * np.exp(-dv * k2 * dt)).real
            du_ = u - u0
            dv_ = v - v0
            u = u + dt * (fu * du_ + fv * dv_)
            v = v + dt * (gu * du_ + gv * dv_)
        return u, v

    def proj(u, v):
        cu = np.fft.fft(u - u0) / nx
        cv = np.fft.fft(v - v0) / nx
        return lvec[0] * cu[nmode] + lvec[1] * cv[nmode]

    u, v = step(u, v, int(0.6 / dt))
    p1 = proj(u, v)
    u, v = step(u, v, int(0.8 / dt))
    p2 = proj(u, v)
    return float(np.log(abs(p2) / abs(p1)) / 0.8)


def band_centroid_frequency(field: np.ndarray, dx: float, dy: float) -> float:
    """Power-weighted centroid in the unstable band. Moves continuously with Dv, unlike a coarse FFT bin."""
    power = np.abs(np.fft.fft2(field - np.mean(field))) ** 2
    fy = np.fft.fftfreq(field.shape[0], d=dy)
    fx = np.fft.fftfreq(field.shape[1], d=dx)
    radius = np.hypot(fx[None, :], fy[:, None])
    band = (radius >= 0.55) & (radius <= 1.70)
    weight = power[band]
    if float(weight.sum()) <= 0.0:
        return 1.0
    return float(np.sum(radius[band] * weight) / np.sum(weight))


def parameter_sweep():
    """Dv/Du sweep at fixed Du on a 24-wavelength patch so the centroid can move."""
    du_ref, _, _ = _rescaled_diffusivities(DV_OVER_DU)
    rows = []
    for ratio in (10.0, 12.0, 16.0, 22.0, 30.0):
        dv = du_ref * ratio
        field, _, dx, dy = simulate_pattern(
            du_ref, dv, nx=144, ny=72, lx=24.0, ly=12.0, steps=1000, dt=0.035, seed=SEED
        )
        freq = band_centroid_frequency(field, dx, dy)
        rows.append(
            {
                "Dv_over_Du": ratio,
                "frequency_per_unit": freq,
                "wavelength_um": LAMBDA_UM / freq,
                "std_u": float(np.std(field)),
            }
        )
    wavelengths = np.array([r["wavelength_um"] for r in rows])
    ratios = np.array([r["Dv_over_Du"] for r in rows])
    monotonic = bool(np.all(np.diff(wavelengths) > 0.0))
    spearman = float(stats.spearmanr(ratios, wavelengths).statistic)
    return {"rows": rows, "monotonic_increasing": monotonic, "spearman_r": spearman}


def segment_body(rgb: np.ndarray) -> np.ndarray:
    """Fossil body = dark bone/scale mass, excluding the blue gallery wall and the pale slab."""
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    blue = (rgb[:, :, 2] > rgb[:, :, 0] + 0.08) & (rgb[:, :, 2] > 0.35)
    body = (gray < 0.42) & ~blue
    body = ndi.binary_opening(body, iterations=1)
    body = ndi.binary_closing(body, iterations=2)
    body = ndi.binary_fill_holes(body)
    labels, n = ndi.label(body)
    if n == 0:
        return body
    sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _mask_metrics(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return {"aspect_ratio": 0.0, "circularity": 0.0, "area_fraction": 0.0}
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    area = float(mask.sum())
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    return {
        "aspect_ratio": width / height,
        "circularity": circularity,
        "area_fraction": float(mask.mean()),
    }


def _orientation_samples(gray: np.ndarray, n: int = 4000, seed: int = SEED) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    thresh = np.quantile(mag, 0.70)
    sel = ang[mag >= thresh]
    rng = np.random.default_rng(seed)
    if sel.size == 0:
        return np.zeros(n)
    take = rng.choice(sel, size=n, replace=sel.size < n)
    return take


def load_photo_and_mask():
    im = Image.open(SPECIMEN).convert("RGB")
    # Height 480 px meets the fitness gate; width keeps the mount's aspect.
    out_h = 480
    out_w = int(round(out_h * im.size[0] / im.size[1]))
    im = im.resize((out_w, out_h), Image.Resampling.LANCZOS)
    rgb = np.asarray(im, dtype=np.float64) / 255.0
    mask = segment_body(rgb)
    return rgb, mask


def composite_frames(rgb, mask, fields, strength: float = 0.62):
    """Grow copper ocelli on the segmented fossil. Early frames stay close to the photograph.

    Dark bone barely changes under a multiply, so the pigment is mixed in as a tint.
    The slab, wall, and bone edges stay in place; only the mask interior takes the field.
    """
    tint = np.zeros_like(rgb)
    tint[:, :, 0] = 0.86
    tint[:, :, 1] = 0.38
    tint[:, :, 2] = 0.10
    images = []
    for i, field in enumerate(fields):
        alpha = strength * ((i / max(1, len(fields) - 1)) ** 1.15)
        # u0 = 1. Spots saturate near u ≈ 2; early noise stays near 1 and stays dark.
        pigment = 1.0 / (1.0 + np.exp(-8.0 * (field - 1.55)))
        tiled = np.tile(pigment, (3, 4))
        spot = np.asarray(
            Image.fromarray((np.clip(tiled, 0, 1) * 255).astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        ) / 255.0
        frame = rgb.copy()
        blend = np.clip(alpha * spot, 0.0, 0.78)
        frame[mask] = np.clip((1.0 - blend[mask, None]) * rgb[mask] + blend[mask, None] * tint[mask], 0.0, 1.0)
        images.append(Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8)))
    return images


def render_gif(images) -> None:
    GIF.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        GIF,
        save_all=True,
        append_images=images[1:],
        duration=110,
        loop=0,
        optimize=False,
    )


def render_png(rgb, mask, final_field, images, dispersion, sweep) -> None:
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), dpi=120)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Brussels mount (CC BY-SA 3.0)")
    overlay = rgb.copy()
    overlay[mask, 1] = np.clip(overlay[mask, 1] * 0.45 + 0.55, 0, 1)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("Segmented dermal/skeletal mask")
    axes[0, 2].imshow(np.asarray(images[-1]))
    axes[0, 2].set_title("Ocellus field on the body")
    im = axes[1, 0].imshow(final_field, cmap="magma", origin="lower")
    axes[1, 0].set_title("Schnakenberg u, 12 × 7.5 λ patch")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].set_xlabel("patch width = 12 × 104.2 µm")
    k = np.asarray(dispersion["k"])
    axes[1, 1].plot(k, dispersion["sigma_analytical"], "k-", label="analytical σ(k)")
    axes[1, 1].plot(k, dispersion["sigma_numerical"], "o", ms=4, label="linearized PDE")
    axes[1, 1].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 1].set_title(f"Dispersion  r={dispersion['pearson_r']:.4f}")
    axes[1, 1].set_xlabel("k (per 104.2 µm)")
    axes[1, 1].legend(fontsize=8)
    ratios = [r["Dv_over_Du"] for r in sweep["rows"]]
    lams = [r["wavelength_um"] for r in sweep["rows"]]
    axes[1, 2].plot(ratios, lams, "o-")
    axes[1, 2].axhline(LAMBDA_UM, color="0.4", ls="--", lw=0.8, label="formula λ = 104.2 µm")
    axes[1, 2].set_title("Dv/Du sweep")
    axes[1, 2].set_xlabel("Dv / Du")
    axes[1, 2].set_ylabel("realized λ (µm)")
    axes[1, 2].legend(fontsize=8)
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Eusthenopteron foordi — Turing k_c at 104 µm, Frasnian 379 Ma (chronoseries 1/1)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PNG)
    plt.close(fig)


def _measurement(name, value, unit, method, reference, source):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "method": method,
        "reference_value": reference,
        "reference_source": source,
    }


def build_measurements(bundle: dict) -> list:
    disp = bundle["dispersion"]
    sweep = bundle["sweep"]
    fossil = bundle["fossil_mask"]
    gallery = bundle["gallery_mask"]
    return [
        _measurement(
            "specimen_dispersion_pearson_r",
            disp["pearson_r"],
            "1",
            "Pearson r between analytical Turing σ(k) and a linearized Fourier step on the real-eigenvalue band",
            1.0,
            "Turing 1952 dispersion relation; value is the solver residual against that curve, not a copy of the reference",
        ),
        _measurement(
            "specimen_dispersion_rmse",
            disp["rmse"],
            "1/time",
            "RMSE of numerical versus analytical growth rates σ(k)",
            0.0,
            "Zero would be a perfect match; the solver leaves a small splitting residual",
        ),
        _measurement(
            "ocellus_wavelength_um",
            bundle["wavelength_um"],
            "µm",
            "Radial FFT peak of the saturated Schnakenberg field, converted by the 104.2 µm formula-wavelength unit",
            LAMBDA_UM,
            "Formula λ_c = 2π/k_c. The realized peak sits slightly off onset, so value ≠ reference",
        ),
        _measurement(
            "specimen_contour_aspect_ratio",
            fossil["aspect_ratio"],
            "1",
            "Bounding-box width/height of the dark fossil body segmented from the Brussels mount",
            gallery["aspect_ratio"],
            "Reference is the same measurement on the species-gallery life reconstruction, a different image",
        ),
        _measurement(
            "specimen_mask_circularity",
            fossil["circularity"],
            "1",
            "4πA/P² of the segmented fossil body",
            gallery["circularity"],
            "Gallery reconstruction circularity, measured independently",
        ),
        _measurement(
            "specimen_orientation_ks",
            bundle["ks_statistic"],
            "1",
            "Two-sample KS statistic of Sobel orientations, fossil crop versus last GIF frame",
            0.0,
            "KS = 0 would mean identical orientation samples; the overlay does not reproduce every bone edge",
        ),
        _measurement(
            "pattern_versus_equal_diffusion_std_ratio",
            bundle["std_ratio_control"],
            "1",
            "std(u) of the Turing run divided by std(u) of the equal-diffusion negative control",
            1.0,
            "Equal diffusion is the Turing-negative control; ratio near 1 would mean the control still patterned",
        ),
        _measurement(
            "pattern_versus_ablated_autocatalysis_std_ratio",
            bundle["std_ratio_ablation"],
            "1",
            "std(u) of the full model divided by std(u) after dropping u²v autocatalysis",
            1.0,
            "Ablation removes the Schnakenberg feedback; residual std is relaxation, not spots",
        ),
        _measurement(
            "wavelength_sweep_spearman_r",
            sweep["spearman_r"],
            "1",
            "Spearman rank correlation of realized λ against Dv/Du over the sweep",
            0.0,
            "Zero would be no rank association; the sweep is the expressivity check",
        ),
    ]


def run_all() -> dict:
    verify_specimen()
    du, dv, gamma = _rescaled_diffusivities()
    field, frames, dx, dy = simulate_pattern(du, dv, snaps=12)
    freq = dominant_frequency(field, dx, dy)
    wavelength_um = LAMBDA_UM / freq
    control = simulate_equal_diffusion()
    ablated = simulate_ablation()
    dispersion = dispersion_curves()
    sweep = parameter_sweep()
    rgb, mask = load_photo_and_mask()
    gallery_rgb = np.asarray(Image.open(GALLERY).convert("RGB").resize((800, 290)), dtype=np.float64) / 255.0
    gallery_gray = 0.299 * gallery_rgb[:, :, 0] + 0.587 * gallery_rgb[:, :, 1] + 0.114 * gallery_rgb[:, :, 2]
    gallery_mask = gallery_gray < 0.92
    gallery_mask = ndi.binary_opening(gallery_mask, iterations=1)
    gallery_mask = ndi.binary_fill_holes(gallery_mask)
    labels, n = ndi.label(gallery_mask)
    if n:
        sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        gallery_mask = labels == (int(np.argmax(sizes)) + 1)
    images = composite_frames(rgb, mask, frames)
    render_gif(images)
    render_png(rgb, mask, field, images, dispersion, sweep)
    with Image.open(GIF) as gim:
        n_frames = int(getattr(gim, "n_frames", 1))
        width, height = gim.size
    gray_photo = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gray_last = np.asarray(images[-1].convert("L"), dtype=np.float64) / 255.0
    ks = stats.ks_2samp(_orientation_samples(gray_photo), _orientation_samples(gray_last, seed=SEED + 7))
    fossil_m = _mask_metrics(mask)
    gallery_m = _mask_metrics(gallery_mask)
    std_pat = float(np.std(field))
    std_ctl = float(np.std(control))
    std_abl = float(np.std(ablated))
    bundle = {
        "dispersion": dispersion,
        "sweep": sweep,
        "fossil_mask": fossil_m,
        "gallery_mask": gallery_m,
        "wavelength_um": wavelength_um,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "std_ratio_control": std_pat / max(std_ctl, 1e-12),
        "std_ratio_ablation": std_pat / max(std_abl, 1e-12),
    }
    measurements = build_measurements(bundle)
    # Mode counts at this wavelength. Lengths are the species-registry values.
    n_this = 0.140 / LAMBDA_M
    n_successor = 0.165 / LAMBDA_M
    payload = {
        "sim_id": "delphi10000-eusthenopteron-foordi-turing-wavelength-s0",
        "phenomenon_size_m": LAMBDA_M,
        "size_human": "104 µm",
        "log10_size_m": float(np.log10(LAMBDA_M)),
        "geological_age_ma": AGE_MA,
        "chronoseries": "1/1",
        "parameters": {
            "a": A_PARAM,
            "b": B_PARAM,
            "Du": du,
            "Dv": dv,
            "Dv_over_Du": DV_OVER_DU,
            "gamma_removed_by_time_rescaling": gamma,
            "k_c_per_length_unit": float(2.0 * np.pi),
        },
        "pattern_std": std_pat,
        "control_std": std_ctl,
        "ablation_std": std_abl,
        "frequency_per_unit": freq,
        "wavelength_um": wavelength_um,
        "dispersion": dispersion,
        "sweep": sweep,
        "fossil_mask": fossil_m,
        "gallery_mask": gallery_m,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "chronoseries_delta": {
            "N_wavelengths_eusthenopteron_0.140m": n_this,
            "N_wavelengths_panderichthys_0.165m": n_successor,
            "delta_N_successor_minus_this": n_successor - n_this,
            "organ_tier_5.7mm_repeat_count": 0.0057 / LAMBDA_M,
        },
        "gif": {
            "path": str(GIF.relative_to(ROOT)),
            "n_frames": n_frames,
            "width": width,
            "height": height,
        },
        "png": str(PNG.relative_to(ROOT)),
        "measurements": measurements,
    }
    METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def measure_morphoeval() -> list:
    """Entry used by morphoeval/measure_outputs.py. Reads the last script run."""
    if not METRICS.exists():
        run_all()
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    return payload["measurements"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    payload = run_all()
    r = payload["dispersion"]["pearson_r"]
    print(
        f"[eusthenopteron-foordi-turing-wavelength-s0] "
        f"λ={payload['wavelength_um']:.2f} µm (formula 104.2), "
        f"r={r:.5f}, RMSE={payload['dispersion']['rmse']:.5f}, "
        f"KS={payload['ks_statistic']:.3f}, "
        f"control ratio={payload['pattern_std'] / max(payload['control_std'], 1e-12):.1f}, "
        f"gif={payload['gif']['n_frames']} frames {payload['gif']['width']}x{payload['gif']['height']}, "
        f"age={payload['geological_age_ma']} Ma"
    )
    if args.self_test:
        assert r > 0.98, payload["dispersion"]
        assert payload["pattern_std"] > 50.0 * payload["control_std"]
        assert payload["pattern_std"] > 20.0 * payload["ablation_std"]
        assert abs(payload["wavelength_um"] - LAMBDA_UM) / LAMBDA_UM < 0.15
        assert payload["gif"]["n_frames"] >= 10
        assert min(payload["gif"]["width"], payload["gif"]["height"]) >= 480
        assert payload["geological_age_ma"] == 379.0
        assert abs(payload["phenomenon_size_m"] - 0.0001042) < 1e-12
        assert payload["sweep"]["spearman_r"] > 0.8
        print("[PASS] self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
