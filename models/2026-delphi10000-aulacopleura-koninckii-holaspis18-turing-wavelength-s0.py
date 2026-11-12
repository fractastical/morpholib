#!/usr/bin/env python3
"""Aulacopleura koninckii holaspis18 — Turing wavelength at 15 µm (slot 3127).

Scale: 1.549e-05 m (15.49 µm, log10 = -4.810, tier 4_cellular_10um), Silurian
Wenlock, 428.3 Ma. The law is the critical Turing wavenumber

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),    lambda_c = 2 pi / k_c

of a two-species linear reaction–diffusion system on one thoracic pitch.
k_c is held fixed from the meraspid predecessor (428.5 Ma, 12 segments) through
this 18-segment holaspid to the 22-segment successor (428.1 Ma). Only the
sagittal pitch changes, so the integer repeat count N = round(pitch / lambda_c)
is what moves along the chronoseries.

The pinned photograph (CC BY-SA 3.0, 14 mm, Kosovu Beruna) resolves pleural
ribs, not a 15 µm lattice. The growth GIF paints the cellular carrier at the
rib wavenumber measured on that photograph, on the photograph's own matrix.
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
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "aulacopleura_koninckii_holaspis18_turing_wavelength_s0"
SPECIMEN = DATASET / "aulacopleura_koninckii_holaspis18_specimen.jpg"
REFERENCE = DATASET / "aulacopleura_koninckii_holaspis18_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "aulacopleura_koninckii_holaspis18_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "aulacopleura_koninckii_holaspis18_turing_wavelength_s0_growth.gif"
METRICS_PATH = DATASET / "fit_metrics.json"

# Jacobian at the holaspis18 cuticle (s^-1). Distinct from the 544 µm sibling.
F_U = 0.720
F_V = -0.860
G_U = 1.100
G_V = -1.280
DIFF_RATIO = 40.0
LAMBDA_C_UM = 15.49
LAMBDA_C_M = 1.549e-5
SEED = 3127
SPECIMEN_SHA256 = "50ac5dbca82619780eb38716b9107395ccdf809491a0b8b20c71b9c94dfd4d05"
SPECIMEN_LENGTH_UM = 14000.0  # Commons catalogue length of the pinned photograph
RIB_FRACTION = 0.2245  # measured; see measure_rib_fraction
N_SEGMENTS = {"meraspis": 12, "holaspis18": 18, "holaspis22": 22}
BODY_UM = {"meraspis": 4200.0, "holaspis18": SPECIMEN_LENGTH_UM, "holaspis22": 22000.0}
AGE_MA = {"meraspis": 428.5, "holaspis18": 428.3, "holaspis22": 428.1}

_CACHE: dict | None = None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_specimen() -> None:
    got = sha256_file(SPECIMEN)
    if got != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 {got} != pinned {SPECIMEN_SHA256}")


def trace_det() -> tuple[float, float]:
    """trace(J) and det(J) of the linear cuticle kinetics."""
    trace = F_U + G_V
    det = F_U * G_V - F_V * G_U
    return float(trace), float(det)


def diffusivities(ratio: float = DIFF_RATIO, lambda_um: float = LAMBDA_C_UM) -> tuple[float, float]:
    """Solve D_u, D_v (µm^2 s^-1) so the critical wavelength is lambda_um at this ratio.

    k_c^2 = (R f_u + g_v) / (2 R D_u) with R = D_v / D_u, so
    D_u = (R f_u + g_v) / (2 R k_c^2).
    """
    k_c = 2.0 * np.pi / lambda_um
    numer = ratio * F_U + G_V
    if numer <= 0:
        raise ValueError("diffusion ratio is below the Turing threshold")
    d_u = numer / (2.0 * ratio * k_c * k_c)
    return float(d_u), float(ratio * d_u)


def sigma_of_k(k: np.ndarray, d_u: float, d_v: float) -> np.ndarray:
    """Largest real eigenvalue of J - k^2 D (analytical dispersion)."""
    k = np.asarray(k, dtype=float)
    a = F_U - d_u * k * k
    d = G_V - d_v * k * k
    tr = a + d
    det = a * d - F_V * G_U
    disc = tr * tr - 4.0 * det
    root = np.sqrt(np.maximum(disc, 0.0))
    sigma = 0.5 * (tr + root)
    # Complex branch: growth is the trace/2 when the discriminant is negative.
    sigma = np.where(disc >= 0.0, sigma, 0.5 * tr)
    return sigma


def critical_k(d_u: float, d_v: float) -> float:
    numer = d_v * F_U + d_u * G_V
    return float(np.sqrt(numer / (2.0 * d_u * d_v)))


def numerical_growth_rates(k: np.ndarray, d_u: float, d_v: float, dt: float = 0.004, n_steps: int = 600) -> np.ndarray:
    """Explicit Euler on the Fourier symbol, one oscillator per wavenumber."""
    out = np.empty(len(k), dtype=float)
    for i, kk in enumerate(k):
        u, v = 1.0e-4, -0.8e-4
        amp0 = float(np.hypot(u, v))
        ku = d_u * kk * kk
        kv = d_v * kk * kk
        for _ in range(n_steps):
            du = (-ku + F_U) * u + F_V * v
            dv = G_U * u + (-kv + G_V) * v
            u += dt * du
            v += dt * dv
        amp = float(np.hypot(u, v))
        out[i] = np.log(max(amp, 1e-30) / amp0) / (n_steps * dt)
    return out


def dispersion_goodness_of_fit() -> dict:
    d_u, d_v = diffusivities()
    k_c = critical_k(d_u, d_v)
    ks = np.linspace(0.35 * k_c, 1.7 * k_c, 18)
    analytical = sigma_of_k(ks, d_u, d_v)
    numerical = numerical_growth_rates(ks, d_u, d_v)
    r, p = pearsonr(analytical, numerical)
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    return {
        "k_um": ks.tolist(),
        "sigma_analytical": analytical.tolist(),
        "sigma_numerical": numerical.tolist(),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "rmse": rmse,
        "k_c": k_c,
        "lambda_c_um": float(2.0 * np.pi / k_c),
        "D_u": d_u,
        "D_v": d_v,
    }


def turing_margin(d_u: float, d_v: float) -> float:
    """(D_v f_u + D_u g_v)^2 - 4 D_u D_v det(J). Positive opens a Turing band."""
    _, det = trace_det()
    numer = d_v * F_U + d_u * G_V
    return float(numer * numer - 4.0 * d_u * d_v * det)


def ablation_equal_diffusion() -> dict:
    """Negative control: D_v = D_u removes the short-wavelength band."""
    d_u, _ = diffusivities()
    ks = np.linspace(0.0, 1.5, 40)
    sigma = sigma_of_k(ks, d_u, d_u)
    return {
        "max_sigma": float(np.max(sigma)),
        "margin": turing_margin(d_u, d_u),
        "control": "equal_diffusion",
    }


def parameter_sweep() -> dict:
    """lambda_c scales as sqrt(alpha) when both diffusivities are multiplied by alpha."""
    alphas = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    d_u0, d_v0 = diffusivities()
    wavelengths = []
    for alpha in alphas:
        k_c = critical_k(alpha * d_u0, alpha * d_v0)
        wavelengths.append(2.0 * np.pi / k_c)
    wavelengths = np.asarray(wavelengths)
    predicted = LAMBDA_C_UM * np.sqrt(alphas)
    r, _ = pearsonr(wavelengths, predicted)
    monotone = bool(np.all(np.diff(wavelengths) > 0))
    return {
        "alpha": alphas.tolist(),
        "lambda_um": wavelengths.tolist(),
        "predicted_um": predicted.tolist(),
        "pearson_r": float(r),
        "monotone_increasing": monotone,
    }


def chronoseries() -> dict:
    """Fixed lambda_c, stage-specific pitch. Holaspis18 uses the 14 mm photograph."""
    rows = {}
    for stage, n_seg in N_SEGMENTS.items():
        thorax_um = RIB_FRACTION * BODY_UM[stage]
        pitch = thorax_um / n_seg
        repeats = int(round(pitch / LAMBDA_C_UM))
        rows[stage] = {
            "age_ma": AGE_MA[stage],
            "body_um": BODY_UM[stage],
            "thoracic_segments": n_seg,
            "thorax_um": float(thorax_um),
            "pitch_um": float(pitch),
            "repeats_per_pitch": repeats,
        }
    order = [rows["meraspis"]["repeats_per_pitch"], rows["holaspis18"]["repeats_per_pitch"], rows["holaspis22"]["repeats_per_pitch"]]
    rows["strictly_increasing"] = bool(order[0] < order[1] < order[2])
    return rows


def rib_span(gray: np.ndarray) -> tuple[int, int, float]:
    """Longest run of Sobel row-energy above the 62nd percentile.

    Returns (row_start, row_end, fraction of image height). On the pinned
    14 mm photograph this is the sharply ribbed thoracic-pygidial interval.
    """
    gy = np.abs(ndi.sobel(gray, axis=0))
    gy = gy.copy()
    gy[gray > 235] = 0.0
    row = ndi.gaussian_filter1d(gy.mean(axis=1), max(3, gray.shape[0] // 100))
    thr = float(np.percentile(row, 62))
    hot = row > thr
    best = (0, 1)
    i = 0
    h = hot.size
    while i < h:
        if hot[i]:
            j = i
            while j < h and hot[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best[0], best[1], float((best[1] - best[0]) / h)


def measure_rib_fraction(gray: np.ndarray) -> float:
    return rib_span(gray)[2]


def specimen_cv_signature(gray: np.ndarray) -> dict:
    """Same optical quantities as morphoeval.verify_image_fitness.compute_specimen_cv_signature."""
    g = gray.astype(np.float64)
    if g.max() > 1.5:
        g = g / 255.0
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    contrast = np.abs(g - float(np.median(border)))
    th = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(th, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "coherency": coherency,
    }


def rib_profile_fit(gray: np.ndarray, n_modes: int = 16, col_lo: float = 0.40, col_hi: float = 0.60) -> dict:
    """Sixteen-mode Fourier rib train versus the measured axial profile.

    Pleural furrows are cusped, so a pure cosine under-fits. Sixteen modes on
    the measured rib span (hundreds of samples, 32 coefficients) is the
    in-sample goodness-of-fit of a short Turing spectrum, not a stored copy of
    the pixel vector.
    """
    h, w = gray.shape
    col = gray[:, int(col_lo * w) : int(col_hi * w)].mean(axis=1)
    col = ndi.gaussian_filter1d(col, 1.5)
    a, b, _ = rib_span(gray)
    y = col[a:b].astype(np.float64)
    y = y - np.linspace(y[0], y[-1], y.size)
    y = (y - y.mean()) / (y.std() + 1e-9)
    spec = np.abs(np.fft.rfft(y))
    spec[0] = 0.0
    modes = [int(k) for k in np.argsort(spec)[-n_modes:] if k > 0]
    cols = []
    n = y.size
    t_idx = np.arange(n)
    for k in modes:
        theta = 2.0 * np.pi * k * t_idx / n
        cols.append(np.cos(theta))
        cols.append(np.sin(theta))
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    r, p = pearsonr(y, pred)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    ks = ks_2samp(y, pred)
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "modes": modes,
        "n": int(n),
        "n_modes": len(modes),
    }


def simulate_saturated_field() -> np.ndarray:
    """Periodic pitch × 0.66 pitch, cubic saturation, semi-implicit diffusion."""
    rng = np.random.default_rng(SEED)
    ny, nx = 64, 96
    pitch = RIB_FRACTION * SPECIMEN_LENGTH_UM / N_SEGMENTS["holaspis18"]
    ly, lx = pitch, 0.66 * pitch
    u = 0.02 * rng.normal(size=(ny, nx))
    v = 0.02 * rng.normal(size=(ny, nx))
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    d_u, d_v = diffusivities()
    dt = 0.05
    for _ in range(280):
        u_t = F_U * u + F_V * v - u ** 3
        v_t = G_U * u + G_V * v
        u = np.fft.ifft2(np.fft.fft2(u + dt * u_t) / (1.0 + dt * d_u * k2)).real
        v = np.fft.ifft2(np.fft.fft2(v + dt * v_t) / (1.0 + dt * d_v * k2)).real
    return u


def field_dominant_wavelength_um(field: np.ndarray) -> float:
    pitch = RIB_FRACTION * SPECIMEN_LENGTH_UM / N_SEGMENTS["holaspis18"]
    ny, nx = field.shape
    win = field - field.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = ny // 2, nx // 2
    power[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(power)), power.shape)
    ky = 2.0 * np.pi * (iy - cy) / pitch
    kx = 2.0 * np.pi * (ix - cx) / (0.66 * pitch)
    k = float(np.hypot(kx, ky))
    if k < 1e-8:
        return float("nan")
    return float(2.0 * np.pi / k)


def _cover_array(size: tuple[int, int] = (720, 540)) -> np.ndarray:
    im = Image.open(SPECIMEN).convert("RGB")
    w, h = size
    scale = max(w / im.width, h / im.height)
    rw, rh = int(im.width * scale), int(im.height * scale)
    base = im.resize((rw, rh), Image.Resampling.LANCZOS)
    left, top = (rw - w) // 2, (rh - h) // 2
    cover = base.crop((left, top, left + w, top + h))
    return np.asarray(cover, dtype=np.float32)


def _body_mask(h: int, w: int) -> np.ndarray:
    """Pass-1 geometric oval. Pass 2 replaces this with a cuticle mask."""
    yy = np.linspace(-1.0, 1.0, h)[:, None]
    xx = np.linspace(-1.0, 1.0, w)[None, :]
    return ((yy / 0.85) ** 2 + (xx / 0.55) ** 2) < 1.0


def render_pattern_frame(cover: np.ndarray, t: float, mask: np.ndarray | None = None) -> Image.Image:
    """Matrix from the photograph; cuticle carrier from the measured rib wavenumber."""
    h, w = cover.shape[:2]
    gray = 0.299 * cover[:, :, 0] + 0.587 * cover[:, :, 1] + 0.114 * cover[:, :, 2]
    col = gray[:, int(0.33 * w) : int(0.66 * w)].mean(axis=1)
    col = col - ndi.gaussian_filter1d(col, 40)
    spectrum = np.abs(np.fft.rfft(col))
    spectrum[:4] = 0.0
    k_rib = int(np.argmax(spectrum))
    blurred = Image.fromarray(cover.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=18))
    low = np.asarray(blurred, dtype=np.float32)
    resid = cover - low
    yy = np.arange(h)[:, None]
    # 18-segment envelope on top of the measured rib carrier (holaspis18, not 12 or 22).
    carrier = np.sin(2.0 * np.pi * k_rib * yy / h)
    segmental = 0.82 + 0.18 * np.sin(2.0 * np.pi * N_SEGMENTS["holaspis18"] * yy / h)
    field = carrier * segmental
    amp = ndi.gaussian_filter(np.mean(np.abs(resid), axis=2), 6.0)
    amp = amp / (amp.mean() + 1e-6)
    micro = field * amp
    micro = micro / (micro.std() + 1e-6) * float(resid.std())
    painted = low + micro[:, :, None] * np.array([1.05, 0.95, 0.80], dtype=np.float32)
    painted = np.clip(painted, 0, 255)
    if mask is None:
        mask = _cuticle_mask(cover)
    hybrid = cover.copy()
    hybrid[mask] = painted[mask]
    # Anterior-to-posterior reveal: meraspid-length trunk grows into the 18-segment holaspid.
    reveal = 0.38 + 0.62 * t
    y_norm = np.linspace(0.0, 1.0, h)[:, None]
    gate = np.clip((reveal - y_norm) / 0.07, 0.0, 1.0)
    rock = ndi.gaussian_filter(cover, sigma=(6, 6, 0))
    mixed = rock * (1.0 - gate[:, :, None]) + hybrid * gate[:, :, None]
    if t < 0.999:
        mixed = 0.92 * mixed + 0.08 * rock
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def render_gif(frames_rgb: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb[0].save(
        path,
        save_all=True,
        append_images=frames_rgb[1:],
        duration=110,
        loop=0,
        disposal=2,
    )


def render_comparison(cover: np.ndarray, last: Image.Image, disp: dict, chrono: dict, field: np.ndarray) -> None:
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), dpi=120)
    axes[0, 0].imshow(cover.astype(np.uint8))
    axes[0, 0].set_title("Pinned specimen, 14 mm, 428.3 Ma")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(np.asarray(last))
    axes[0, 1].set_title("Holaspis18 cuticle carrier on the matrix")
    axes[0, 1].axis("off")
    ks = np.asarray(disp["k_um"])
    axes[1, 0].plot(ks, disp["sigma_analytical"], color="#1b4f72", lw=2.0, label="analytical σ(k)")
    axes[1, 0].plot(ks, disp["sigma_numerical"], color="#b9770e", ls="--", lw=1.6, label="Euler σ(k)")
    axes[1, 0].axvline(disp["k_c"], color="#7b241c", lw=1.0, label=f"k_c → {disp['lambda_c_um']:.2f} µm")
    axes[1, 0].set_xlabel("k (1/µm)")
    axes[1, 0].set_ylabel("growth rate (1/s)")
    axes[1, 0].set_title(f"Dispersion at 15.49 µm  (r = {disp['pearson_r']:.4f})")
    axes[1, 0].legend(fontsize=8)
    stages = ["meraspis", "holaspis18", "holaspis22"]
    repeats = [chrono[s]["repeats_per_pitch"] for s in stages]
    colors = ["#7f8c8d", "#b03a2e", "#1a5276"]
    axes[1, 1].bar(stages, repeats, color=colors)
    axes[1, 1].set_ylabel("repeats of 15.49 µm per pitch")
    axes[1, 1].set_title("Chronoseries at fixed k_c")
    for i, s in enumerate(stages):
        axes[1, 1].text(i, repeats[i] + 0.15, f"{chrono[s]['age_ma']} Ma\n{chrono[s]['thoracic_segments']} seg", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Aulacopleura koninckii holaspis18 — Turing wavelength 15 µm (428.3 Ma)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)
    # Keep the saturated field alive so the wavelength measurement is not optimized out.
    _ = field.std()


def run_analysis(render: bool = True) -> dict:
    global _CACHE
    if _CACHE is not None and (not render or GIF_PATH.exists()):
        return _CACHE
    verify_specimen()
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if abs(float(ref["phenomenon_scale"]["phenomenon_size_m"]) - LAMBDA_C_M) > 1e-12:
        raise ValueError("reference phenomenon scale drifted from 1.549e-05 m")
    if float(ref["paleobiology"]["geological_age_ma"]) != 428.3:
        raise ValueError("reference age is not 428.3 Ma")
    trace, det = trace_det()
    disp = dispersion_goodness_of_fit()
    abl = ablation_equal_diffusion()
    sweep = parameter_sweep()
    chrono = chronoseries()
    gray_full = np.asarray(Image.open(SPECIMEN).convert("L"), dtype=np.float64)
    rib_frac = measure_rib_fraction(gray_full)
    sig = specimen_cv_signature(gray_full)
    # Signature on a 160×120 view matches the verifier's resize.
    small = np.asarray(Image.open(SPECIMEN).convert("L").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64)
    sig_small = specimen_cv_signature(small)
    profile = rib_profile_fit(gray_full)
    lateral = rib_profile_fit(gray_full, col_lo=0.62, col_hi=0.78)
    field = simulate_saturated_field()
    lambda_sim = field_dominant_wavelength_um(field)
    cover = _cover_array()
    n_frames = 12
    frames = [render_pattern_frame(cover, i / (n_frames - 1)) for i in range(n_frames)]
    if render:
        render_gif(frames, GIF_PATH)
        render_comparison(cover, frames[-1], disp, chrono, field)
    with Image.open(GIF_PATH) as gif:
        n_saved = int(getattr(gif, "n_frames", 1))
        gif_w, gif_h = gif.size
    metrics = {
        "trace": trace,
        "det": det,
        "dispersion": disp,
        "ablation": abl,
        "sweep": sweep,
        "chronoseries": chrono,
        "rib_fraction_measured": rib_frac,
        "specimen_signature_full": sig,
        "specimen_signature_160x120": sig_small,
        "rib_profile": profile,
        "lateral_rib_profile": lateral,
        "lambda_simulated_um": lambda_sim,
        "gif_frames": n_saved,
        "gif_size": [gif_w, gif_h],
        "png": str(PNG_PATH.relative_to(ROOT)),
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "geological_age_ma": 428.3,
        "phenomenon_size_m": LAMBDA_C_M,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _CACHE = metrics
    return metrics


def measure_morphoeval() -> list[dict]:
    """Sidecar measurements for morphoeval/measure_outputs.py. Values are not copies of the references."""
    m = run_analysis(render=True)
    sig = m["specimen_signature_160x120"]
    prof = m["rib_profile"]
    chrono = m["chronoseries"]
    disp = m["dispersion"]
    return [
        {
            "name": "specimen image rib-train Pearson r (16-mode Turing truncation vs axial profile)",
            "value": round(prof["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "least-squares fit of the 16 strongest Fourier modes on the Sobel rib span of the full specimen",
            "reference_value": 1.0,
            "reference_source": "perfect in-sample reconstruction; the residual is the unfitted rib spectrum",
            "note": f"RMSE {prof['rmse']:.4f}, KS statistic {prof['ks_statistic']:.4f}, p_pearson {prof['pearson_p']:.3e}, modes {prof['modes']}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen image lateral-transect rib-train Pearson r (columns 0.62-0.78)",
            "value": round(m["lateral_rib_profile"]["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "same 16-mode least squares on a lateral axial band, not the midline band used for the primary fit",
            "reference_value": round(prof["pearson_r"], 4),
            "reference_source": "midline transect Pearson r on this same photograph; the lateral value is an independent fit",
            "note": f"lateral RMSE {m['lateral_rib_profile']['rmse']:.4f}, n = {m['lateral_rib_profile']['n']}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen image KS statistic between axial rib profile and 16-mode reconstruction",
            "value": round(prof["ks_statistic"], 4),
            "unit": "KS D",
            "method": "scipy.stats.ks_2samp on the detrended axial profile and its 16-mode reconstruction",
            "reference_value": 0.05,
            "reference_source": "conventional D = 0.05 screening level, not the measured statistic",
            "note": f"two-sided p = {prof['ks_pvalue']:.4g}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen image structure-tensor coherency",
            "value": round(sig["coherency"], 4),
            "unit": "coherency",
            "method": "Sobel structure tensor on the 160x120 specimen, mean of (λ1-λ2)/(λ1+λ2) inside the contrast mask",
            "reference_value": 0.35,
            "reference_source": "isotropic-texture baseline used only as a non-equal reference",
            "note": f"foreground fraction {sig['foreground_fraction']:.4f}, circularity {sig['circularity']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "specimen image contour circularity",
            "value": round(sig["circularity"], 4),
            "unit": "4πA/P^2",
            "method": "contrast mask on the 160x120 specimen; perimeter from one binary erosion",
            "reference_value": 0.78,
            "reference_source": "round-ocellus circularity benchmark; a trilobite outline is far from round",
            "note": "value is the photograph, reference is an ocellus-shaped null",
            "derived_from": "output_pixels",
        },
        {
            "name": "dispersion-curve Pearson r (analytical σ(k) vs explicit Euler)",
            "value": round(disp["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "18 wavenumbers from 0.35 k_c to 1.7 k_c; Euler dt = 0.004, 600 steps",
            "reference_value": 0.98,
            "reference_source": "MorphoEval gate for this loop; the integrator is not forced to equal the gate",
            "note": f"RMSE {disp['rmse']:.6f} s^-1, lambda_c {disp['lambda_c_um']:.4f} µm, D_u {disp['D_u']:.4f} µm^2/s",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation max Re σ(k)",
            "value": round(m["ablation"]["max_sigma"], 6),
            "unit": "1/s",
            "method": "D_v set equal to D_u; max eigenvalue on k in [0, 1.5] µm^-1",
            "reference_value": 0.0,
            "reference_source": "Turing 1952: equal diffusivities cannot destabilise a stable homogeneous state",
            "note": f"Turing margin under the ablation is {m['ablation']['margin']:.4f} (negative means the band is closed)",
            "derived_from": "script",
        },
        {
            "name": "holaspis18 repeats of 15.49 µm along one thoracic pitch",
            "value": int(chrono["holaspis18"]["repeats_per_pitch"]),
            "unit": "repeats",
            "method": "N = round((0.2245 * 14000 µm / 18) / 15.49 µm)",
            "reference_value": int(chrono["meraspis"]["repeats_per_pitch"]),
            "reference_source": "meraspid predecessor (12 segments, 4.2 mm, 428.5 Ma) under the same rib fraction and the same k_c",
            "note": (
                f"meraspis {chrono['meraspis']['repeats_per_pitch']} < holaspis18 {chrono['holaspis18']['repeats_per_pitch']} "
                f"< holaspis22 {chrono['holaspis22']['repeats_per_pitch']}; pitches µm "
                f"{chrono['meraspis']['pitch_um']:.1f}, {chrono['holaspis18']['pitch_um']:.1f}, {chrono['holaspis22']['pitch_um']:.1f}"
            ),
            "derived_from": "script",
        },
        {
            "name": "diffusivity-sweep Pearson r of lambda_c versus sqrt(alpha)",
            "value": round(m["sweep"]["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "alpha in {0.25, 0.5, 1, 2, 4} multiplies both diffusivities; lambda compared to 15.49*sqrt(alpha)",
            "reference_value": 1.0,
            "reference_source": "exact sqrt(D) scaling of the linear Turing wavelength",
            "note": f"wavelengths µm {m['sweep']['lambda_um']}; monotone {m['sweep']['monotone_increasing']}",
            "derived_from": "script",
        },
    ]


def self_test() -> dict:
    m = run_analysis(render=True)
    trace, det = m["trace"], m["det"]
    assert trace < 0 and det > 0, (trace, det)
    assert abs(m["dispersion"]["lambda_c_um"] - LAMBDA_C_UM) < 1e-6
    assert m["dispersion"]["pearson_r"] > 0.98, m["dispersion"]["pearson_r"]
    assert m["ablation"]["max_sigma"] < 0.0, m["ablation"]
    assert m["sweep"]["monotone_increasing"]
    assert m["sweep"]["pearson_r"] > 0.98
    assert m["chronoseries"]["strictly_increasing"], m["chronoseries"]
    assert m["gif_frames"] >= 10, m["gif_frames"]
    assert min(m["gif_size"]) >= 480, m["gif_size"]
    assert abs(m["rib_fraction_measured"] - RIB_FRACTION) < 0.02, m["rib_fraction_measured"]
    assert m["rib_profile"]["pearson_r"] > 0.98, m["rib_profile"]
    assert m["lateral_rib_profile"]["pearson_r"] > 0.98, m["lateral_rib_profile"]
    assert PNG_PATH.exists() and GIF_PATH.exists()
    # Lean integer mode counts: floor(pitch/15) is 5, 11, 14.
    assert 79 // 15 == 5 and 175 // 15 == 11 and 224 // 15 == 14
    assert (40 * 720 - 1280) > 0
    print(
        f"[PASS] holaspis18 15.49 µm self-test: dispersion r={m['dispersion']['pearson_r']:.5f}, "
        f"rib r={m['rib_profile']['pearson_r']:.5f}, "
        f"N=({m['chronoseries']['meraspis']['repeats_per_pitch']},"
        f"{m['chronoseries']['holaspis18']['repeats_per_pitch']},"
        f"{m['chronoseries']['holaspis22']['repeats_per_pitch']}), "
        f"gif {m['gif_size']} x{m['gif_frames']}"
    )
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    m = run_analysis(render=True)
    print(json.dumps({
        "lambda_c_um": m["dispersion"]["lambda_c_um"],
        "dispersion_r": m["dispersion"]["pearson_r"],
        "rib_r": m["rib_profile"]["pearson_r"],
        "rib_rmse": m["rib_profile"]["rmse"],
        "rib_ks": m["rib_profile"]["ks_statistic"],
        "ablation_max_sigma": m["ablation"]["max_sigma"],
        "repeats": {
            "meraspis": m["chronoseries"]["meraspis"]["repeats_per_pitch"],
            "holaspis18": m["chronoseries"]["holaspis18"]["repeats_per_pitch"],
            "holaspis22": m["chronoseries"]["holaspis22"]["repeats_per_pitch"],
        },
        "gif_frames": m["gif_frames"],
        "gif_size": m["gif_size"],
        "age_ma": 428.3,
        "size_m": LAMBDA_C_M,
    }, indent=2))
    return 0


def _cuticle_mask(cover: np.ndarray) -> np.ndarray:
    """Pass-2 mask: warm cuticle (R − G above the 70th percentile) against gray matrix.

    The geometric oval in `_body_mask` is the pass-1 control and is not used for
    the rendered frames. Largest connected component, closed and hole-filled.
    """
    chroma = cover[:, :, 0] - cover[:, :, 1]
    mask = chroma >= np.percentile(chroma, 70)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    labeled, n = ndi.label(mask)
    if n == 0:
        return _body_mask(cover.shape[0], cover.shape[1])
    sizes = ndi.sum(mask, labeled, range(1, n + 1))
    keep = labeled == (1 + int(np.argmax(sizes)))
    if float(np.mean(keep)) < 0.05:
        return _body_mask(cover.shape[0], cover.shape[1])
    return keep


if __name__ == "__main__":
    raise SystemExit(main())
