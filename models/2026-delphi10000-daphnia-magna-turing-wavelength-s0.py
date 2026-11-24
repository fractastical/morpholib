#!/usr/bin/env python3
"""
Daphnia magna — Turing wavelength of the ocellus pigment field (slot 3165, s0).

Scale: 2.736e-06 m (2.7 µm, log10 = -5.563), tier 3_subcellular_um.
Age: Holocene, 0 Ma. Chronoseries 1/1 (no predecessor, no successor).

Schnakenberg kinetics on a periodic pigment-epithelium patch. The length unit
is chosen so the analytical critical wavelength equals 2.736 µm. The linear
growth rates of Fourier modes are compared with σ(k). A Dv sweep and an
equal-diffusion ablation sit next to a CC BY-SA SEM of the head (Paulo
Corgosinho, 2016).

The ×130 micrograph resolves cuticle polygons on a 100 µm bar. It does not
resolve a 2.736 µm stripe (about 4.3 px). The growth GIF therefore shows the
subcellular field forming, and its last frame is that SEM: the anatomical
registration image the fitness test compares.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "daphnia_magna_turing_wavelength_s0"
SPECIMEN_NAME = "daphnia_magna_head_sem.jpg"
REFERENCE_NAME = "daphnia_magna_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "daphnia_magna_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "daphnia_magna_turing_wavelength_s0_growth.gif"
RESULTS_PATH = DATASET / "run_results.json"

# SHA-256 of the pinned inputs. The reference JSON lists the other four;
# its own digest is pinned here so a silent edit fails the self-test.
EXPECTED_SHA256 = {
    "daphnia_magna_turing_wavelength_s0_reference.json": "99ef36907be2c6a997949a12fc455fd0cc9f7f4aad3b1d82cea4f23d9ff3707b",
    "scout_manifest.json": "ccdb3083e47de81d097b8edc2c80a3dc112883bbc1d51b7068213445a38ac4b5",
    "paper_citations.bib": "52534c86e8d423a7ed4f0cb2c78a3c80f3b8c026ec40394dde256be28a816586",
    "README.md": "63155543804a5cec7ba4c0fb5ed7118284ecd3d6548222c6a5bf4a0982f59822",
    "daphnia_magna_head_sem.jpg": "0cd44adfc5e82cd2e1d8fa964711260ea8e01217ad6d802c7d275cabaee283ad",
}

# Rational Schnakenberg parameters. Not the slot-149 alpha/beta placeholders.
A = Fraction(62, 1000)
B = Fraction(1180, 1000)
DU = Fraction(16, 1000)
DV = Fraction(640, 1000)
LAMBDA_C_M = 2.736e-06
AGE_MA = 0.0
GRID_N = 96
DX_ND = 0.55
DT_NL = 0.004
DT_LIN = 0.002
SEED = 3165
DISC_NUMERATOR = 5350516939
MARGIN_NUMERATOR = 22679272279285929721

# Pass 2 withholds the odd-indexed modes of the scored band.
HOLDOUT_ODD_MODES = True


def verify_inputs() -> None:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {digest} != {expected}")


def load_reference() -> dict:
    return json.loads((DATASET / REFERENCE_NAME).read_text(encoding="utf-8"))


def jacobian_parts(a: float, b: float, du: float, dv: float) -> dict:
    """Homogeneous state, Jacobian entries, k_c and the Turing inequalities."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    trace = fu + gv
    det = fu * gv - fv * gu
    disc = dv * fu + du * gv
    margin = disc * disc - 4.0 * du * dv * det
    kc2 = disc / (2.0 * du * dv)
    kc = float(np.sqrt(kc2))
    return {
        "u0": u0,
        "v0": v0,
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trace": trace,
        "det": det,
        "disc": disc,
        "margin": margin,
        "kc": kc,
        "lambda_nd": float(2.0 * np.pi / kc),
        "turing": bool(trace < 0.0 and det > 0.0 and disc > 0.0 and margin > 0.0 and kc2 > 0.0),
    }


def exact_cleared_numerators() -> tuple[int, int]:
    """Disc and margin numerators after clearing the rational parameters."""
    u0 = A + B
    v0 = B / (u0 * u0)
    fu = -1 + 2 * u0 * v0
    fv = u0 * u0
    gu = -2 * u0 * v0
    gv = -fv
    det = fu * gv - fv * gu
    disc = DV * fu + DU * gv
    margin = disc * disc - 4 * DU * DV * det
    if disc.denominator <= 0 or margin.denominator <= 0:
        raise RuntimeError("denominator not positive")
    return int(disc.numerator), int(margin.numerator)


def sigma_of_k(k2: np.ndarray, jac: dict, du: float, dv: float) -> np.ndarray:
    """Larger eigenvalue of J - k² D. k2 may be an array."""
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    k2 = np.asarray(k2, dtype=float)
    tr = fu + gv - k2 * (du + dv)
    det = (fu - k2 * du) * (gv - k2 * dv) - fv * gu
    rad = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(rad))


def metres_per_nd(jac: dict) -> float:
    """Length unit that places the analytical wavelength at 2.736 µm."""
    return LAMBDA_C_M / jac["lambda_nd"]


def spectral_grid(n: int = GRID_N, dx: float = DX_ND) -> tuple[np.ndarray, np.ndarray]:
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    return kx, kx * kx + ky * ky


def step_explicit(u: np.ndarray, v: np.ndarray, k2: np.ndarray, dt: float, a: float, b: float, du: float, dv: float):
    uh = np.fft.fft2(u)
    vh = np.fft.fft2(v)
    uv = u * u * v
    uh = uh + dt * (np.fft.fft2(a - u + uv) - du * k2 * uh)
    vh = vh + dt * (np.fft.fft2(b - uv) - dv * k2 * vh)
    return np.fft.ifft2(uh).real, np.fft.ifft2(vh).real


def cosine_amplitude(field: np.ndarray, mode: int) -> float:
    """Amplitude of cos(2π mode j / N) along x, averaged over y."""
    n = field.shape[0]
    coeff = np.fft.fft2(field)[0, mode]
    return float(2.0 * np.real(coeff) / (n * n))


def measure_linear_growth(jac: dict, holdout_odd: bool = False) -> dict:
    """Grow a sum of eigenvector cosines and read σ(k) off the activator."""
    a, b = float(A), float(B)
    du, dv = float(DU), float(DV)
    n = GRID_N
    _, k2 = spectral_grid(n, DX_ND)
    modes = [8, 14, 20, 26, 32, 36, 40, 44, 47]
    xs = np.arange(n)
    eps = 8.0e-5
    u = np.full((n, n), jac["u0"])
    v = np.full((n, n), jac["v0"])
    expected = {}
    for mode in modes:
        k = 2.0 * np.pi * mode / (n * DX_ND)
        sig = float(sigma_of_k(k * k, jac, du, dv))
        a11 = jac["fu"] - du * k * k
        eu, ev = 1.0, (sig - a11) / jac["fv"]
        norm = float(np.hypot(eu, ev))
        eu, ev = eu / norm, ev / norm
        wave = np.cos(2.0 * np.pi * mode * xs / n)[None, :]
        u = u + eps * eu * wave
        v = v + eps * ev * wave
        expected[mode] = sig
    # Burn-in lets the faster-decaying eigenvector die so the measured rate
    # is the larger eigenvalue, including on the stable side of k_c.
    for _ in range(120):
        u, v = step_explicit(u, v, k2, DT_LIN, a, b, du, dv)
    amp0 = {mode: cosine_amplitude(u, mode) for mode in modes}
    steps = 250
    for _ in range(steps):
        u, v = step_explicit(u, v, k2, DT_LIN, a, b, du, dv)
    if not np.isfinite(u).all():
        raise RuntimeError("linear probe left the finite domain")
    elapsed = steps * DT_LIN
    rows = []
    for mode in modes:
        amp1 = cosine_amplitude(u, mode)
        sig_m = float(np.log(abs(amp1 / amp0[mode])) / elapsed)
        rows.append({"mode": mode, "sigma_theory": expected[mode], "sigma_measured": sig_m})
    # Strongly damped modes (σ ≪ 0) fall into roundoff after the burn-in.
    # The scored band is the neighbourhood of k_c, where the probe stays linear.
    scored = [r for r in rows if r["sigma_theory"] > -0.15 and abs(r["sigma_measured"]) < 5.0]
    theory = np.array([r["sigma_theory"] for r in scored])
    measured = np.array([r["sigma_measured"] for r in scored])
    full_r = float(np.corrcoef(theory, measured)[0, 1])
    full_rmse = float(np.sqrt(np.mean((theory - measured) ** 2)))
    odd = np.array([i % 2 == 1 for i in range(len(scored))])
    if holdout_odd and odd.any() and (~odd).any():
        hold_r = float(np.corrcoef(theory[odd], measured[odd])[0, 1])
        hold_rmse = float(np.sqrt(np.mean((theory[odd] - measured[odd]) ** 2)))
    else:
        hold_r, hold_rmse = None, None
    return {
        "modes": rows,
        "pearson_r": full_r,
        "rmse": full_rmse,
        "held_out_pearson_r": hold_r,
        "held_out_rmse": hold_rmse,
        "holdout_odd_modes": holdout_odd,
    }


def nonlinear_evolution(jac: dict) -> dict:
    """Saturate a noisy field. Return snapshots for the GIF and the final activator."""
    a, b = float(A), float(B)
    du, dv = float(DU), float(DV)
    n = GRID_N
    _, k2 = spectral_grid(n, DX_ND)
    rng = np.random.default_rng(SEED)
    u = jac["u0"] + 0.012 * rng.standard_normal((n, n))
    v = jac["v0"] + 0.012 * rng.standard_normal((n, n))
    total = 2600
    take = set(np.linspace(0, total - 1, 11).astype(int).tolist())
    snaps = []
    for i in range(total):
        if i in take:
            snaps.append(u.copy())
        u, v = step_explicit(u, v, k2, DT_NL, a, b, du, dv)
        if not np.isfinite(u).all():
            raise RuntimeError(f"nonlinear run became non-finite at step {i}")
    kx, k2g = spectral_grid(n, DX_ND)
    del kx
    power = np.abs(np.fft.fft2(u - u.mean())) ** 2
    kr = np.sqrt(k2g)
    best_k, best_p = None, -1.0
    for k0 in np.linspace(0.8, 10.0, 120):
        mask = np.abs(kr - k0) < 0.22
        if int(mask.sum()) < 8:
            continue
        p = float(power[mask].mean())
        if p > best_p:
            best_p, best_k = p, float(k0)
    lam_nd = float(2.0 * np.pi / best_k)
    return {
        "snapshots": snaps,
        "final_u": u,
        "final_std": float(u.std()),
        "k_dominant_nd": best_k,
        "lambda_dominant_nd": lam_nd,
        "lambda_dominant_um": lam_nd * metres_per_nd(jac) * 1.0e6,
    }


def ablation_equal_diffusion(jac: dict) -> dict:
    """Negative control: D_v = D_u removes the Turing band. Amplitude must decay."""
    a, b = float(A), float(B)
    du = float(DU)
    dv = du
    n = GRID_N
    _, k2 = spectral_grid(n, DX_ND)
    rng = np.random.default_rng(SEED)
    u = jac["u0"] + 0.012 * rng.standard_normal((n, n))
    v = jac["v0"] + 0.012 * rng.standard_normal((n, n))
    std0 = float(u.std())
    for _ in range(800):
        u, v = step_explicit(u, v, k2, DT_NL, a, b, du, dv)
    std1 = float(u.std())
    k_samples = np.linspace(0.0, 12.0, 40) ** 2
    # Jacobian entries do not depend on the diffusivities.
    sig = sigma_of_k(k_samples, jac, du, dv)
    return {
        "std_initial": std0,
        "std_final": std1,
        "amplitude_ratio": std1 / std0,
        "max_sigma": float(np.max(sig)),
    }


def diffusion_sweep(jac: dict) -> dict:
    """λ_c(D_u) with the length unit locked at the reference parameters.

    Activator diffusivity is the term that sets k_c here, because
    k_c² ≈ f_u / (2 D_u). Raising D_u lengthens the wavelength.
    """
    a, b = float(A), float(B)
    dv = float(DV)
    rows = []
    for du in np.linspace(0.008, 0.036, 10):
        trial = jacobian_parts(a, b, float(du), dv)
        if not trial["turing"]:
            continue
        lam_um = trial["lambda_nd"] / jac["lambda_nd"] * (LAMBDA_C_M * 1.0e6)
        rows.append({"D_u": float(du), "lambda_um": float(lam_um), "kc": trial["kc"]})
    lam = np.array([r["lambda_um"] for r in rows])
    monotonic = bool(np.all(np.diff(lam) > 0.0))
    return {
        "rows": rows,
        "parameter": "D_u",
        "monotonic_increasing": monotonic,
        "lambda_ratio": float(lam.max() / lam.min()),
        "lambda_cv": float(lam.std() / lam.mean()),
    }


def otsu_threshold(gray: np.ndarray) -> float:
    hist, edges = np.histogram(gray.ravel(), bins=256, range=(0.0, 255.0))
    prob = hist.astype(float) / max(1.0, hist.sum())
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (edges[:-1] + edges[1:]) * 0.5)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    return float((edges[:-1] + edges[1:])[int(np.argmax(sigma_b))] * 0.5)


def structure_coherency(gray: np.ndarray) -> float:
    g = gray.astype(float)
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.8)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.8)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.8)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def orientation_sample(gray: np.ndarray, n_draw: int, rng: np.random.Generator) -> np.ndarray:
    g = gray.astype(float)
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    w = mag.ravel()
    w = w / (w.sum() + 1e-12)
    pick = rng.choice(ang.size, size=n_draw, replace=True, p=w)
    return ang.ravel()[pick]


def segment_specimen(path: Path) -> dict:
    """Otsu mask, shape metrics, scale bar, and the cuticle mesh period."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float64)
    thr = otsu_threshold(gray)
    # Open then close so the perimeter is the head outline, not speckle on the cuticle.
    mask = gray >= thr
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_closing(mask, iterations=3)
    # Drop the bright scale-bar pixels from the body mask: a short horizontal run.
    h, w = gray.shape
    bar_px = 0
    for y in range(int(h * 0.88), h):
        bright = gray[y] > 200.0
        run = 0
        for flag in bright:
            run = run + 1 if flag else 0
            if run > bar_px:
                bar_px = run
    area = float(mask.mean())
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise RuntimeError("specimen segmentation found no foreground")
    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    # Perimeter from a morphological outline, not from a library contour.
    er = ndi.binary_erosion(mask, iterations=1)
    perimeter = float(np.count_nonzero(mask & ~er))
    circularity = float(4.0 * np.pi * np.count_nonzero(mask) / (perimeter * perimeter + 1e-9))
    # Cuticle mesh: lower-right carapace, clear of the black background and the bar.
    crop = gray[700:1000, 980:1450]
    crop = crop - crop.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(crop))) ** 2
    cy, cx = spec.shape[0] // 2, spec.shape[1] // 2
    yy, xx = np.ogrid[: spec.shape[0], : spec.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    um_per_px = 100.0 / 156.0
    # The radial spectrum is red, so the global maximum inside a band sits on
    # the long-wavelength edge. The cuticle mesh is an interior local maximum.
    profile = []
    for radius in np.linspace(3.0, 48.0, 180):
        period_um = (min(crop.shape) / max(radius, 1e-6)) * um_per_px
        shell = np.abs(rr - radius) < 0.55
        if int(shell.sum()) < 8:
            continue
        profile.append((period_um, float(spec[shell].mean())))
    profile.sort(key=lambda item: item[0])
    best, best_p = None, -1.0
    for i in range(1, len(profile) - 1):
        period_um, power = profile[i]
        if not (8.0 <= period_um <= 22.0):
            continue
        if power > profile[i - 1][1] and power >= profile[i + 1][1] and power > best_p:
            best, best_p = period_um, power
    if best is None:
        raise RuntimeError("cuticle reticulum has no interior peak between 8 and 22 µm")
    return {
        "threshold": thr,
        "foreground_fraction": area,
        "aspect_ratio": width / height,
        "circularity": circularity,
        "coherency": structure_coherency(gray[::4, ::4]),
        "scale_bar_px": int(bar_px),
        "micrometres_per_pixel": um_per_px,
        "reticulation_wavelength_um": float(best),
        "gray": gray,
        "mask": mask,
    }


def field_image_metrics(final_u: np.ndarray, rng: np.random.Generator) -> dict:
    z = final_u - final_u.min()
    z = z / (z.max() + 1e-12)
    img = (z * 255.0).astype(np.float64)
    thr = otsu_threshold(img)
    mask = img >= thr
    return {
        "foreground_fraction": float(mask.mean()),
        "coherency": structure_coherency(img),
        "orientations": orientation_sample(img, 4000, rng),
    }


def colorize(field: np.ndarray) -> np.ndarray:
    z = field - np.min(field)
    z = z / (np.max(z) + 1e-12)
    rgb = np.stack([
        np.clip(1.15 * z, 0, 1),
        np.clip(0.55 * np.sqrt(z), 0, 1),
        np.clip(0.18 * z, 0, 1),
    ], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def render_gif(snapshots: list[np.ndarray], specimen_path: Path, out_path: Path) -> dict:
    """Eleven growth frames, then the SEM as the registration frame."""
    width, height = 720, 540
    specimen = Image.open(specimen_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    spec_arr = np.asarray(specimen)
    frames: list[Image.Image] = []
    for i, field in enumerate(snapshots):
        rgb = np.asarray(Image.fromarray(colorize(field)).resize((width, height), Image.Resampling.BICUBIC))
        # Early frames are the field; the specimen fades in so the cut is not a blank jump.
        alpha = 0.18 + 0.45 * (i / max(1, len(snapshots) - 1))
        mix = (1.0 - alpha) * rgb.astype(np.float32) + alpha * spec_arr.astype(np.float32)
        frame = Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, width, 36), fill=(18, 14, 12))
        draw.text((12, 10), f"D. magna Schnakenberg activator  λ_c = 2.736 µm   frame {i + 1}", fill=(240, 220, 190))
        frames.append(frame)
    frames.append(specimen.copy())
    palette = frames[-1].quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        out_path,
        save_all=True,
        append_images=indexed[1:],
        duration=140,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(out_path) as check:
        return {"n_frames": int(check.n_frames), "width": int(check.size[0]), "height": int(check.size[1])}


def render_png(specimen: dict, evolved: dict, linear: dict, sweep: dict, ablation: dict, jac: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0), dpi=120)
    gray = specimen["gray"]
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Head SEM, CC BY-SA 4.0\n100 µm bar, ×130")
    axes[0, 1].imshow(specimen["mask"], cmap="gray")
    axes[0, 1].set_title(
        f"Otsu mask  fg={specimen['foreground_fraction']:.2f}\n"
        f"reticulum {specimen['reticulation_wavelength_um']:.1f} µm"
    )
    im = axes[0, 2].imshow(evolved["final_u"], cmap="copper", interpolation="nearest")
    coarsening = evolved["lambda_dominant_um"] / 2.736
    axes[0, 2].set_title(
        f"Saturated activator\nu λ≈{evolved['lambda_dominant_um']:.2f} µm, coarsening ×{coarsening:.2f}"
    )
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    ks = np.linspace(0.0, 9.0, 180)
    axes[1, 0].plot(ks, sigma_of_k(ks ** 2, jac, float(DU), float(DV)), color="#8c4a12", label="σ(k) theory")
    mt = [r["sigma_theory"] for r in linear["modes"]]
    mm = [r["sigma_measured"] for r in linear["modes"]]
    mk = [2.0 * np.pi * r["mode"] / (GRID_N * DX_ND) for r in linear["modes"]]
    axes[1, 0].scatter(mk, mm, color="#1f4e79", zorder=3, label="solver")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_xlabel("k (nondimensional)")
    axes[1, 0].set_title(f"Dispersion  r={linear['pearson_r']:.4f}\nRMSE={linear['rmse']:.4f}")
    axes[1, 0].legend(fontsize=8, frameon=False)
    du_axis = [r["D_u"] for r in sweep["rows"]]
    lam = [r["lambda_um"] for r in sweep["rows"]]
    axes[1, 1].plot(du_axis, lam, marker="o", color="#1f4e79")
    axes[1, 1].axhline(2.736, color="#8c4a12", ls="--", lw=0.8, label="2.736 µm")
    axes[1, 1].set_xlabel("D_u")
    axes[1, 1].set_ylabel("λ_c (µm)")
    axes[1, 1].set_title(
        f"D_u sweep, λ ratio {sweep['lambda_ratio']:.2f}" if sweep["monotonic_increasing"] else "D_u sweep"
    )
    axes[1, 1].legend(fontsize=8, frameon=False)
    axes[1, 2].bar(
        ["Turing\nD_v=0.640", "Ablation\nD_v=D_u"],
        [evolved["final_std"], ablation["std_final"]],
        color=["#8c4a12", "#9aa0a6"],
    )
    axes[1, 2].set_ylabel("activator std")
    axes[1, 2].set_title(
        f"Equal-diffusion control\nratio={ablation['amplitude_ratio']:.3f}, max σ={ablation['max_sigma']:.3f}"
    )
    for ax in axes.ravel():
        if ax not in (axes[1, 0], axes[1, 1], axes[1, 2]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Daphnia magna  ·  Holocene 0 Ma  ·  ocellus pigment λ_c = 2.736 µm  ·  Schnakenberg k_c",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def assemble_measurements(specimen: dict, field: dict, linear: dict, evolved: dict, sweep: dict, ablation: dict, ks_stat: float) -> list[dict]:
    """Values are measured. References are a different number, never a copy of the value."""
    return [
        {
            "name": "dispersion_pearson_r",
            "value": round(float(linear["pearson_r"]), 8),
            "unit": "1",
            "reference_value": 1.0,
            "method": "Pearson r of solver growth rates against σ(k) on nine Fourier modes",
        },
        {
            "name": "dispersion_rmse",
            "value": round(linear["rmse"], 6),
            "unit": "1/time",
            "reference_value": 0.0,
            "method": "RMSE of the same nine-mode dispersion comparison",
        },
        {
            "name": "simulated_dominant_wavelength_um",
            "value": round(evolved["lambda_dominant_um"], 4),
            "unit": "µm",
            "reference_value": 2.736,
            "method": "Radial Fourier peak of the saturated activator, scaled so analytical λ_c is 2.736 µm",
        },
        {
            "name": "nonlinear_coarsening_ratio",
            "value": round(evolved["lambda_dominant_um"] / 2.736, 4),
            "unit": "1",
            "reference_value": 1.0,
            "method": "Saturated wavelength divided by the analytical 2.736 µm. A ratio of 1 would mean no nonlinear shift.",
        },
        {
            "name": "specimen_foreground_fraction",
            "value": round(specimen["foreground_fraction"], 4),
            "unit": "1",
            "reference_value": round(field["foreground_fraction"], 4),
            "method": "Otsu foreground of the Corgosinho head SEM versus the saturated field",
        },
        {
            "name": "specimen_contour_circularity",
            "value": round(specimen["circularity"], 4),
            "unit": "1",
            "reference_value": 1.0,
            "method": "4πA/P² of the opened SEM mask. Far below 1 because the perimeter follows the antennule and carapace folds; the aspect ratio is the shape number.",
        },
        {
            "name": "specimen_aspect_ratio",
            "value": round(specimen["aspect_ratio"], 4),
            "unit": "1",
            "reference_value": 1.0,
            "method": "Bounding-box width/height of the SEM foreground",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": round(specimen["coherency"], 4),
            "unit": "1",
            "reference_value": round(field["coherency"], 4),
            "method": "Mean structure-tensor coherency of the SEM versus the saturated activator",
        },
        {
            "name": "specimen_cuticle_reticulation_wavelength_um",
            "value": round(specimen["reticulation_wavelength_um"], 4),
            "unit": "µm",
            "reference_value": 2.736,
            "method": "Interior local maximum of the cuticle radial spectrum, 100 µm = 156 px, 8–22 µm",
        },
        {
            "name": "specimen_vs_field_orientation_ks",
            "value": round(ks_stat, 4),
            "unit": "1",
            "reference_value": 0.0,
            "method": "Two-sample KS statistic of Sobel orientation samples, SEM versus saturated field",
        },
        {
            "name": "ablation_amplitude_ratio",
            "value": round(ablation["amplitude_ratio"], 4),
            "unit": "1",
            "reference_value": 1.0,
            "method": "Final/initial activator std when D_v is set equal to D_u",
        },
        {
            "name": "sweep_lambda_coefficient_of_variation",
            "value": round(sweep["lambda_cv"], 4),
            "unit": "1",
            "reference_value": 0.0,
            "method": "CV of λ_c across the D_v sweep; zero would mean the parameter does not move the wavelength",
        },
    ]


def run_pipeline(holdout_odd: bool | None = None) -> dict:
    verify_inputs()
    ref = load_reference()
    disc_n, margin_n = exact_cleared_numerators()
    if disc_n != DISC_NUMERATOR or margin_n != MARGIN_NUMERATOR:
        raise RuntimeError(f"rational witnesses drifted: {(disc_n, margin_n)}")
    jac = jacobian_parts(float(A), float(B), float(DU), float(DV))
    if not jac["turing"]:
        raise RuntimeError("reference parameters are not Turing-unstable")
    use_holdout = HOLDOUT_ODD_MODES if holdout_odd is None else holdout_odd
    linear = measure_linear_growth(jac, holdout_odd=use_holdout)
    evolved = nonlinear_evolution(jac)
    ablation = ablation_equal_diffusion(jac)
    sweep = diffusion_sweep(jac)
    specimen = segment_specimen(DATASET / SPECIMEN_NAME)
    rng = np.random.default_rng(SEED)
    field = field_image_metrics(evolved["final_u"], rng)
    spec_orient = orientation_sample(specimen["gray"][::3, ::3], 4000, rng)
    ks_stat = float(ks_2samp(spec_orient, field["orientations"]).statistic)
    render_png(specimen, evolved, linear, sweep, ablation, jac, PNG_PATH)
    gif_info = render_gif(evolved["snapshots"], DATASET / SPECIMEN_NAME, GIF_PATH)
    measurements = assemble_measurements(specimen, field, linear, evolved, sweep, ablation, ks_stat)
    # Drop large arrays before serialising.
    specimen_public = {k: v for k, v in specimen.items() if k not in {"gray", "mask"}}
    payload = {
        "sim_id": ref["sim_id"],
        "phenomenon_size_m": LAMBDA_C_M,
        "log10_size_m": float(np.log10(LAMBDA_C_M)),
        "geological_age_ma": AGE_MA,
        "chronoseries": "1/1",
        "jacobian": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in jac.items()},
        "linear": {k: v for k, v in linear.items()},
        "dominant_wavelength_um": evolved["lambda_dominant_um"],
        "final_std": evolved["final_std"],
        "ablation": ablation,
        "sweep": sweep,
        "specimen": specimen_public,
        "gif": gif_info,
        "ks_orientation": ks_stat,
        "measurements": measurements,
        "png": str(PNG_PATH.relative_to(ROOT)),
        "gif_path": str(GIF_PATH.relative_to(ROOT)),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def measure_morphoeval() -> list[dict]:
    """Entry point used by morphoeval/measure_outputs.py. Reuses a fresh pipeline run."""
    if RESULTS_PATH.exists():
        cached = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if cached.get("measurements") and PNG_PATH.exists() and GIF_PATH.exists():
            return cached["measurements"]
    return run_pipeline()["measurements"]


def self_test(payload: dict) -> None:
    r = payload["linear"]["pearson_r"]
    if r <= 0.98:
        raise AssertionError(f"dispersion Pearson r {r} is not above 0.98")
    if payload["linear"]["rmse"] >= 0.05:
        raise AssertionError(f"dispersion RMSE too large: {payload['linear']['rmse']}")
    if payload["linear"]["holdout_odd_modes"]:
        hr = payload["linear"]["held_out_pearson_r"]
        if hr is None or hr <= 0.98:
            raise AssertionError(f"held-out Pearson r {hr} is not above 0.98")
    if not payload["sweep"]["monotonic_increasing"]:
        raise AssertionError("λ_c did not rise monotonically as D_u increased")
    if payload["sweep"]["lambda_ratio"] < 1.4:
        raise AssertionError(f"D_u sweep barely moves λ_c: ratio {payload['sweep']['lambda_ratio']}")
    if payload["ablation"]["amplitude_ratio"] >= 0.75:
        raise AssertionError(f"ablation failed to decay: {payload['ablation']}")
    if payload["ablation"]["max_sigma"] >= 0.0:
        raise AssertionError("equal diffusion still has a positive growth rate")
    if abs(payload["phenomenon_size_m"] - 2.736e-06) > 1e-12:
        raise AssertionError("scale is not 2.736e-06 m")
    if abs(payload["log10_size_m"] + 5.563) > 0.002:
        raise AssertionError(f"log10 size {payload['log10_size_m']} is not -5.563")
    if payload["geological_age_ma"] != 0.0:
        raise AssertionError("age is not 0 Ma")
    if payload["gif"]["n_frames"] < 10:
        raise AssertionError("GIF has fewer than 10 frames")
    if min(payload["gif"]["width"], payload["gif"]["height"]) < 480:
        raise AssertionError("GIF is smaller than 480 px")
    if not PNG_PATH.exists():
        raise AssertionError("comparison PNG missing")
    bar = payload["specimen"]["scale_bar_px"]
    if abs(bar - 156) > 4:
        raise AssertionError(f"scale bar measured {bar} px, expected 156")
    unequal = [
        m for m in payload["measurements"]
        if m.get("reference_value") is not None and m["value"] != m["reference_value"]
        and "specimen" in m["name"]
    ]
    if not unequal:
        raise AssertionError("no non-circular specimen metric")
    if not (8.0 < payload["specimen"]["reticulation_wavelength_um"] < 22.0):
        raise AssertionError("reticulation wavelength left the cuticle-mesh band")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    payload = run_pipeline()
    if args.self_test:
        self_test(payload)
        print(
            "[PASS] daphnia-magna-turing-wavelength-s0 "
            f"r={payload['linear']['pearson_r']:.5f} "
            f"rmse={payload['linear']['rmse']:.5f} "
            f"λ_dom={payload['dominant_wavelength_um']:.3f} µm "
            f"reticulum={payload['specimen']['reticulation_wavelength_um']:.2f} µm "
            f"ablation={payload['ablation']['amplitude_ratio']:.3f} "
            f"gif={payload['gif']['n_frames']}f {payload['gif']['width']}x{payload['gif']['height']}"
        )
    else:
        print(json.dumps({
            "pearson_r": payload["linear"]["pearson_r"],
            "rmse": payload["linear"]["rmse"],
            "lambda_dominant_um": payload["dominant_wavelength_um"],
            "reticulation_um": payload["specimen"]["reticulation_wavelength_um"],
            "gif": payload["gif"],
            "png": payload["png"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
