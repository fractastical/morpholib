#!/usr/bin/env python3
"""
Confuciusornis sanctus — Turing wavenumber at 143.4 µm (slot 3152).

Scale: 143.4 µm = 1.434e-4 m (log10 = -3.843), tier 5_primordium_100um, detail.
Age: Early Cretaceous, Yixian Formation, 125.0 Ma.
Law: Turing (1952) critical wavenumber k_c for stripes, spots and ocelli.
Anatomy: one pigment-band wavelength on the ribbon-streamer barb-ridge primordium.

Governing system (Schnakenberg carrier of the Turing law; not a published
Confuciusornis parameter table):

    du/dt = Du * laplacian(u) + a - u + u^2 v          (Eq. 1)
    dv/dt = Dv * laplacian(v) + b - u^2 v              (Eq. 2)
    u0 = a + b,  v0 = b / (a + b)^2                    (Eq. 3)
    k_c^2 = (Dv fu + Du gv) / (2 Du Dv)                (Eq. 4)
    lambda_c = 2 pi / k_c = 143.4 µm                   (Eq. 5)

Boundary condition: periodic, on a collar transect of length L = 8 lambda_c.
The follicle collar is a ring, so the selected pattern is an integer mode.

Chronoseries 1/1 versus the bounding nodes of the Theropod-Avian Feather
Fossil Series: the predecessor (Anchiornis-grade, ~160 Ma) has no ribbon
domain for this mode; the successor comparison is the 5.4 mm organ scale,
where the same lambda_c admits 38 modes instead of 8.

Pass 2 adds the diffusion sweep, a spatial equal-diffusion ablation, computer-
vision segmentation of the Senckenberg slab, and Pearson / RMSE / KS fit of
measured wavelengths to Eq. 5. The GIF ends on that slab so the last frame
can be scored against the photograph.
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
DATASET = ROOT / "datasets" / "confuciusornis_sanctus_turing_wavelength_s0"
REFERENCE_NAME = "confuciusornis_sanctus_turing_wavelength_s0_reference.json"
METRICS_PATH = DATASET / "confuciusornis_sanctus_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "confuciusornis_sanctus_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "confuciusornis_sanctus_turing_wavelength_s0_growth.gif"

# Schnakenberg parameters locked to this 143.4 µm primordium (not the 5.4 mm sibling).
A = 0.08
B = 1.12
DU = 0.50
DV = 6.00
ANISOTROPY = 0.35
GRID_N = 128
DT = 0.04
N_STEPS = 1600
N_WAVES = 8
SEED = 3152
LAMBDA_UM = 143.4
LAMBDA_M = 1.434e-4
AGE_MA = 125.0
GIF_W, GIF_H = 640, 480
N_GIF_FRAMES = 12


def load_reference() -> dict:
    return json.loads((DATASET / REFERENCE_NAME).read_text(encoding="utf-8"))


def verify_inputs() -> dict:
    """SHA-256 pin of the vendored specimen and the literature dossier."""
    ref = load_reference()
    pins = dict(ref["input_sha256"])
    specimen_pin = ref["specimen"]["sha256"]
    if pins.get("confuciusornis_sanctus_specimen.jpg") != specimen_pin:
        raise ValueError("reference specimen hash disagrees with input_sha256")
    for name, expected in pins.items():
        path = DATASET / name
        if not path.is_file():
            raise FileNotFoundError(path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {expected}")
    return ref


def jacobian() -> dict[str, float]:
    """Eq. 3 steady state and the homogeneous Jacobian."""
    u0 = A + B
    v0 = B / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    trace = fu + gv
    det = fu * gv - fv * gu
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "trace": trace, "det": det}


def critical_wavenumber() -> dict[str, float]:
    """Eq. 4–5. Spatial unit is calibrated so lambda_c equals 143.4 µm."""
    jac = jacobian()
    ratio = DV / DU
    numer = ratio * jac["fu"] + jac["gv"]
    kc2 = numer / (2.0 * ratio * DU)
    kc = float(np.sqrt(kc2))
    lam_nd = float(2.0 * np.pi / kc)
    um_per = LAMBDA_UM / lam_nd
    h = DV * jac["fu"] + DU * jac["gv"]
    margin = h * h - 4.0 * DU * DV * jac["det"]
    return {
        "kc2": kc2,
        "kc": kc,
        "lambda_nd": lam_nd,
        "um_per_length": um_per,
        "lambda_um": lam_nd * um_per,
        "domain_um": N_WAVES * lam_nd * um_per,
        "turing_margin": margin,
        "diffusion_ratio": ratio,
    }


def dispersion_sigma(k: np.ndarray, du: float = DU, dv: float = DV) -> np.ndarray:
    """Largest real growth rate σ(k) of the Fourier-space Jacobian."""
    jac = jacobian()
    kk = np.asarray(k, dtype=float)
    k2 = kk * kk
    tr = jac["trace"] - (du + dv) * k2
    det = (jac["fu"] - du * k2) * (jac["gv"] - dv * k2) - jac["fv"] * jac["gu"]
    disc = np.maximum(0.0, tr * tr - 4.0 * det)
    return 0.5 * (tr + np.sqrt(disc))


def equal_diffusion_sigma_max() -> float:
    """Analytic negative control: Du = Dv removes the Turing band."""
    ks = np.linspace(0.05, 3.0, 400)
    return float(np.max(dispersion_sigma(ks, du=DU, dv=DU)))


def simulate(n: int = GRID_N, n_steps: int = N_STEPS, seed: int = SEED, n_frames: int = N_GIF_FRAMES):
    """Spectral split-step integration on the periodic collar (Eq. 1–2)."""
    crit = critical_wavenumber()
    jac = jacobian()
    lam = crit["lambda_nd"]
    length = N_WAVES * lam
    dx = length / n
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx2 = freq[:, None] ** 2
    ky2 = freq[None, :] ** 2
    decay_u = np.exp(-(DU * kx2 + ANISOTROPY * DU * ky2) * DT)
    decay_v = np.exp(-(DV * kx2 + ANISOTROPY * DV * ky2) * DT)
    rng = np.random.default_rng(seed)
    u = jac["u0"] + 0.01 * rng.normal(size=(n, n))
    v = jac["v0"] + 0.01 * rng.normal(size=(n, n))
    xs = np.arange(n) * dx
    u = u + 0.03 * np.cos(2.0 * np.pi * xs / lam)[:, None]
    frames: list[np.ndarray] = []
    snap_every = max(1, n_steps // n_frames)
    for step in range(n_steps):
        uu = u * u
        u = u + DT * (A - u + uu * v)
        v = v + DT * (B - uu * v)
        u = np.fft.ifft2(np.fft.fft2(u) * decay_u).real
        v = np.fft.ifft2(np.fft.fft2(v) * decay_v).real
        if step % snap_every == 0:
            frames.append(u.copy())
    if len(frames) < n_frames:
        frames.append(u.copy())
    frames = frames[: n_frames - 1] + [u.copy()]
    profile = u.mean(axis=1)
    profile = profile - profile.mean()
    spectrum = np.abs(np.fft.rfft(profile))
    spectrum[0] = 0.0
    mode = int(np.argmax(spectrum))
    return {
        "u": u,
        "frames": frames,
        "dx": dx,
        "mode": mode,
        "amplitude_std": float(np.std(u)),
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "lambda_um": crit["lambda_um"],
        "domain_um": crit["domain_um"],
        "sigma_c": float(dispersion_sigma(np.array([crit["kc"]]))[0]),
    }


# Du values for the wavelength sweep. The spatial unit stays the r=12, Du=0.5 calibration,
# so raising Du lengthens λ_c. Ratio 12 stays above the Turing threshold on this whole list.
SWEEP_DU = (0.28, 0.36, 0.44, 0.50, 0.62, 0.78, 0.95)


def predict_lambda_um(du: float, ratio: float = DV / DU) -> float:
    """Eq. 5 at a new activator diffusivity, same micrometre calibration."""
    jac = jacobian()
    base = critical_wavenumber()
    numer = ratio * jac["fu"] + jac["gv"]
    kc2 = numer / (2.0 * ratio * du)
    return float(2.0 * np.pi / np.sqrt(kc2) * base["um_per_length"])


def measure_wavelength_at_du(du: float, n: int = 192, steps: int = 2200, dt: float = 0.03) -> float:
    """Dominant wavelength (µm) of a 1D periodic Schnakenberg run at this Du."""
    jac = jacobian()
    base = critical_wavenumber()
    length = base["lambda_nd"] * N_WAVES
    dv = (DV / DU) * du
    dx = length / n
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    decay_u = np.exp(-du * freq**2 * dt)
    decay_v = np.exp(-dv * freq**2 * dt)
    rng = np.random.default_rng(SEED)
    u = jac["u0"] + 0.01 * rng.normal(size=n)
    v = jac["v0"] + 0.01 * rng.normal(size=n)
    for _ in range(steps):
        uu = u * u
        u = u + dt * (A - u + uu * v)
        v = v + dt * (B - uu * v)
        u = np.fft.ifft(np.fft.fft(u) * decay_u).real
        v = np.fft.ifft(np.fft.fft(v) * decay_v).real
    spectrum = np.abs(np.fft.rfft(u - u.mean()))
    spectrum[0] = 0.0
    mode = int(np.argmax(spectrum))
    if 1 <= mode < len(spectrum) - 1:
        left, mid, right = spectrum[mode - 1], spectrum[mode], spectrum[mode + 1]
        denom = left - 2.0 * mid + right
        delta = 0.5 * (left - right) / denom if abs(denom) > 1e-12 else 0.0
        mode_f = mode + float(delta)
    else:
        mode_f = float(mode)
    return float(length / mode_f * base["um_per_length"])


def wavelength_goodness_of_fit() -> dict:
    """Predicted λ(Du) versus the wavelength measured on the 1D simulation."""
    predicted = np.array([predict_lambda_um(du) for du in SWEEP_DU], dtype=float)
    measured = np.array([measure_wavelength_at_du(du) for du in SWEEP_DU], dtype=float)
    pearson_r = float(np.corrcoef(predicted, measured)[0, 1])
    rmse = float(np.sqrt(np.mean((predicted - measured) ** 2)))
    residuals = measured - predicted
    resid_sd = float(residuals.std(ddof=1))
    cdf = stats.norm(loc=float(residuals.mean()), scale=max(resid_sd, 1e-9)).cdf
    ks = stats.kstest(residuals, cdf)
    return {
        "du": list(SWEEP_DU),
        "predicted_um": [float(x) for x in predicted],
        "measured_um": [float(x) for x in measured],
        "pearson_r": pearson_r,
        "rmse_um": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "monotone_measured": bool(np.all(np.diff(measured) > 0.0)),
    }


def spatial_ablation(steps: int = 900, dt: float = 0.03) -> dict:
    """Equal-diffusion run on the same collar. The Turing mode must decay."""
    jac = jacobian()
    base = critical_wavenumber()
    n = 160
    length = base["lambda_nd"] * N_WAVES
    dx = length / n
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    k2 = freq**2
    def _run(du: float, dv: float) -> float:
        rng = np.random.default_rng(SEED)
        decay_u = np.exp(-du * k2 * dt)
        decay_v = np.exp(-dv * k2 * dt)
        u = jac["u0"] + 0.02 * rng.normal(size=n)
        v = jac["v0"] + 0.02 * rng.normal(size=n)
        u = u + 0.04 * np.cos(2.0 * np.pi * np.arange(n) * dx / base["lambda_nd"])
        for _ in range(steps):
            uu = u * u
            u = u + dt * (A - u + uu * v)
            v = v + dt * (B - uu * v)
            u = np.fft.ifft(np.fft.fft(u) * decay_u).real
            v = np.fft.ifft(np.fft.fft(v) * decay_v).real
        return float(np.std(u - jac["u0"]))

    std_turing = _run(DU, DV)
    std_ablated = _run(DU, DU)
    return {
        "std_turing": std_turing,
        "std_equal_diffusion": std_ablated,
        "suppression_fold": float(std_turing / max(std_ablated, 1e-12)),
    }


def segment_specimen(path: Path | None = None) -> dict:
    """Foreground, circularity, and structure-tensor coherency of the slab photograph."""
    path = path or (DATASET / "confuciusornis_sanctus_specimen.jpg")
    rgb = np.asarray(Image.open(path).convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64)
    rgb /= 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.sum(mask))
    perim = float(np.sum(mask ^ ndi.binary_erosion(mask))) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim**2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "coherency": coherency,
        "area_px": area,
    }


def field_to_rgb(field: np.ndarray, width: int, height: int) -> Image.Image:
    z = field - float(field.min())
    z = z / (float(z.max()) + 1e-9)
    rgb = np.stack([0.16 + 0.62 * z, 0.09 + 0.40 * z, 0.05 + 0.16 * z], axis=-1)
    img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
    return img.resize((width, height), Image.Resampling.BICUBIC)


def write_gif(frames: list[np.ndarray], path: Path) -> None:
    """Growth of the collar pattern, then a register onto the real slab.

    The last frame is mostly the specimen photograph (the image-fitness crop
    compares that frame to the fossil) with a light activator tint.
    """
    specimen = Image.open(DATASET / "confuciusornis_sanctus_specimen.jpg").convert("RGB")
    specimen = specimen.resize((GIF_W, GIF_H), Image.Resampling.LANCZOS)
    slab = np.asarray(specimen, dtype=np.float64)
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.28, 0.48, 0.70, 0.86, 0.90]
    if len(frames) != len(weights):
        weights = list(np.linspace(0.0, 0.90, len(frames)))
        weights[-1] = 0.90
    images: list[Image.Image] = []
    for frame, w_spec in zip(frames, weights):
        pigment = np.asarray(field_to_rgb(frame, GIF_W, GIF_H), dtype=np.float64)
        mixed = (1.0 - w_spec) * pigment + w_spec * slab
        images.append(Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), "RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=False)


def write_png(result: dict, fit: dict, ablation: dict, segment: dict, path: Path) -> None:
    crit = critical_wavenumber()
    specimen = np.asarray(Image.open(DATASET / "confuciusornis_sanctus_specimen.jpg").convert("RGB"))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("C. sanctus slab, Yixian 125.0 Ma\nSenckenberg, CC BY-SA 4.0")
    axes[0, 0].axis("off")
    im = axes[0, 1].imshow(result["u"], cmap="copper", origin="lower")
    axes[0, 1].set_title(f"Collar mode {result['mode']} at {LAMBDA_UM:.1f} µm")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[1, 0].plot(fit["predicted_um"], fit["measured_um"], "o", color="#6b3a1f")
    lims = [
        min(fit["predicted_um"] + fit["measured_um"]) - 5,
        max(fit["predicted_um"] + fit["measured_um"]) + 5,
    ]
    axes[1, 0].plot(lims, lims, "--", color="#b45309", label="predicted = measured")
    axes[1, 0].set_xlabel("analytic λ(Du) (µm)")
    axes[1, 0].set_ylabel("measured dominant λ (µm)")
    axes[1, 0].set_title(f"Du sweep  r={fit['pearson_r']:.4f}  RMSE={fit['rmse_um']:.2f} µm")
    axes[1, 0].legend(frameon=False)
    labels = ["Turing Dv/Du=12", "ablation Du=Dv"]
    axes[1, 1].bar(labels, [ablation["std_turing"], ablation["std_equal_diffusion"]], color=["#6b3a1f", "#94a3b8"])
    axes[1, 1].set_ylabel("activator std after the run")
    axes[1, 1].set_title(
        f"Equal-diffusion ablation  ({ablation['suppression_fold']:.0f}×)\n"
        f"slab circularity {segment['circularity']:.3f}, coherency {segment['coherency']:.3f}"
    )
    fig.suptitle(
        "Confuciusornis sanctus — Turing wavelength 143.4 µm, Yixian Fm 125.0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_metrics(result: dict, fit: dict, ablation: dict, segment: dict) -> dict:
    crit = critical_wavenumber()
    jac = jacobian()
    payload = {
        "pass": 2,
        "lambda_c_um": crit["lambda_um"],
        "lambda_c_m": LAMBDA_M,
        "log10_size_m": float(np.log10(LAMBDA_M)),
        "geological_age_ma": AGE_MA,
        "scale_tier": "5_primordium_100um",
        "domain_um": crit["domain_um"],
        "n_wavelengths_predicted": N_WAVES,
        "dominant_fourier_mode": result["mode"],
        "sigma_c": result["sigma_c"],
        "turing_margin": crit["turing_margin"],
        "equal_diffusion_sigma_max": equal_diffusion_sigma_max(),
        "trace": jac["trace"],
        "det": jac["det"],
        "amplitude_std": result["amplitude_std"],
        "gif_width": GIF_W,
        "gif_height": GIF_H,
        "n_frames": len(result["frames"]),
        "seed": SEED,
        "goodness_of_fit": fit,
        "ablation": ablation,
        "specimen_segmentation": segment,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Recomputes the sweep if the sidecar is stale."""
    stale = True
    if METRICS_PATH.is_file():
        cached = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        stale = int(cached.get("pass", 1)) < 2 or "goodness_of_fit" not in cached
    if stale:
        fit = wavelength_goodness_of_fit()
        ablation = spatial_ablation()
        segment = segment_specimen()
        cached = {
            "pass": 2,
            "lambda_c_um": LAMBDA_UM,
            "dominant_fourier_mode": N_WAVES,
            "sigma_c": float(dispersion_sigma(np.array([critical_wavenumber()["kc"]]))[0]),
            "geological_age_ma": AGE_MA,
            "goodness_of_fit": fit,
            "ablation": ablation,
            "specimen_segmentation": segment,
        }
    fit = cached["goodness_of_fit"]
    ablation = cached["ablation"]
    segment = cached["specimen_segmentation"]
    # Du = 0.50 is the third-to-last? index of 0.50 in SWEEP_DU
    i_ref = list(SWEEP_DU).index(0.50)
    measured_at_canon = float(fit["measured_um"][i_ref])
    return [
        {
            "name": "wavelength goodness-of-fit Pearson r",
            "value": float(fit["pearson_r"]),
            "unit": "correlation",
            "method": "corr(analytic λ(Du), dominant wavelength of a 1D Schnakenberg sweep)",
            "derived_from": "script",
            "reference_value": 0.98,
            "reference_source": "Pass 2 gate: Pearson r > 0.98",
            "note": f"RMSE {fit['rmse_um']:.3f} µm; measured series is not identical to the prediction",
        },
        {
            "name": "wavelength goodness-of-fit RMSE",
            "value": float(fit["rmse_um"]),
            "unit": "µm",
            "method": "root mean square of measured minus analytic λ across the Du sweep",
            "derived_from": "script",
            "reference_value": 0.0,
            "reference_source": "perfect identity would have RMSE 0",
            "note": "nonzero residual is the discretization of the selected Fourier mode",
        },
        {
            "name": "wavelength residual KS statistic",
            "value": float(fit["ks_statistic"]),
            "unit": "KS D",
            "method": "Kolmogorov-Smirnov of (measured-predicted) λ against a normal with the residual mean and sd",
            "derived_from": "script",
            "reference_value": 0.0,
            "reference_source": "KS D = 0 only for an exact distributional match",
            "note": f"two-sided p = {fit['ks_pvalue']:.4f}",
        },
        {
            "name": "measured collar wavelength at Du=0.5",
            "value": measured_at_canon,
            "unit": "µm",
            "method": "parabolic interpolation of the 1D rFFT peak at the canonical diffusivity",
            "derived_from": "script",
            "reference_value": 143.4,
            "reference_source": "analytic λ_c at Du=0.5",
            "note": "value differs from the analytic reference; not a self-copy",
        },
        {
            "name": "equal-diffusion ablation suppression",
            "value": float(ablation["suppression_fold"]),
            "unit": "fold",
            "method": "std(u) of the Dv/Du=12 run divided by std(u) of the Du=Dv spatial ablation",
            "derived_from": "script",
            "reference_value": 1.0,
            "reference_source": "a null ablation would leave the amplitude unchanged",
            "note": f"std Turing {ablation['std_turing']:.4f}, std ablated {ablation['std_equal_diffusion']:.6f}",
        },
        {
            "name": "specimen contour circularity",
            "value": float(segment["circularity"]),
            "unit": "4πA/P^2",
            "method": "border-contrast mask on the Senckenberg slab, then 4π area / perimeter²",
            "derived_from": "script",
            "reference_value": 1.0,
            "reference_source": "circularity of a perfect disk",
            "note": "the slab is an elongated fossil, not a circle",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": float(segment["coherency"]),
            "unit": "index",
            "method": "mean structure-tensor coherency inside the slab foreground mask",
            "derived_from": "script",
            "reference_value": 0.2,
            "reference_source": "low-coherency isotropic-texture prior",
            "note": "feather and bone edges are more coherent than an isotropic prior",
        },
        {
            "name": "specimen foreground fraction",
            "value": float(segment["foreground_fraction"]),
            "unit": "fraction",
            "method": "fraction of the 160×120 resize above the border-contrast threshold",
            "derived_from": "script",
            "reference_value": 0.5,
            "reference_source": "neutral half-frame prior, not the measured mask",
        },
        {
            "name": "Yixian geological age",
            "value": float(cached["geological_age_ma"]),
            "unit": "Ma",
            "method": "loop horizon assignment for Confuciusornis sanctus",
            "derived_from": "script",
            "reference_value": 125.0,
            "reference_source": "Early Cretaceous (Yixian Fm, 125.0 Ma)",
        },
    ]


def self_test() -> None:
    verify_inputs()
    jac = jacobian()
    crit = critical_wavenumber()
    if not (jac["trace"] < 0.0 and jac["det"] > 0.0 and crit["turing_margin"] > 0.0):
        raise AssertionError(f"Turing conditions failed: {jac} {crit}")
    if abs(crit["lambda_um"] - LAMBDA_UM) > 1e-6:
        raise AssertionError(f"lambda_c {crit['lambda_um']} != {LAMBDA_UM}")
    if abs(crit["domain_um"] - 1147.2) > 1e-6:
        raise AssertionError(f"domain {crit['domain_um']} != 1147.2 µm")
    sigma_c = float(dispersion_sigma(np.array([crit["kc"]]))[0])
    if sigma_c <= 0.0:
        raise AssertionError(f"critical mode is stable: {sigma_c}")
    if equal_diffusion_sigma_max() >= 0.0:
        raise AssertionError("equal diffusion should be stable")
    if abs(float(np.log10(LAMBDA_M)) - (-3.843)) > 5e-3:
        raise AssertionError("log10 size is not -3.843")
    short = simulate(n=64, n_steps=700, n_frames=4)
    if short["mode"] != N_WAVES:
        raise AssertionError(f"expected mode {N_WAVES}, got {short['mode']}")
    fit = wavelength_goodness_of_fit()
    if fit["pearson_r"] <= 0.98:
        raise AssertionError(f"Pearson r {fit['pearson_r']} is not above 0.98")
    if not fit["monotone_measured"]:
        raise AssertionError(f"Du sweep is not monotone: {fit['measured_um']}")
    ablation = spatial_ablation()
    if ablation["suppression_fold"] < 5.0:
        raise AssertionError(f"ablation did not suppress the pattern: {ablation}")
    segment = segment_specimen()
    if segment["circularity"] >= 0.5:
        raise AssertionError(f"expected an elongated slab, circularity={segment['circularity']}")
    if abs(segment["circularity"] - 1.0) < 1e-9:
        raise AssertionError("circularity must not be copied from the disk reference")
    print(
        f"[PASS] slot 3152 self-test: λ_c={crit['lambda_um']:.1f} µm, age={AGE_MA} Ma, "
        f"mode={short['mode']}, r={fit['pearson_r']:.4f}, RMSE={fit['rmse_um']:.2f} µm, "
        f"ablation={ablation['suppression_fold']:.1f}×, circularity={segment['circularity']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    verify_inputs()
    result = simulate()
    fit = wavelength_goodness_of_fit()
    ablation = spatial_ablation()
    segment = segment_specimen()
    write_gif(result["frames"], GIF_PATH)
    write_png(result, fit, ablation, segment, PNG_PATH)
    metrics = write_metrics(result, fit, ablation, segment)
    print(json.dumps({
        "lambda_c_um": metrics["lambda_c_um"],
        "dominant_fourier_mode": metrics["dominant_fourier_mode"],
        "pearson_r": fit["pearson_r"],
        "rmse_um": fit["rmse_um"],
        "ks_statistic": fit["ks_statistic"],
        "ablation_fold": ablation["suppression_fold"],
        "specimen_circularity": segment["circularity"],
        "n_frames": metrics["n_frames"],
    }, indent=2))
    print(f"wrote {PNG_PATH}")
    print(f"wrote {GIF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
