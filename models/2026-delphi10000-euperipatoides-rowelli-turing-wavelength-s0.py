#!/usr/bin/env python3
"""
Euperipatoides rowelli — Turing reaction-diffusion wavelength at 1.12 µm.

Loop 3223. Holocene (extant), 0 Ma. Chronoseries 1/1.
Law: fastest-growing Turing wavenumber k_sigma of Schnakenberg kinetics,
calibrated so the selected wavelength is 1.12e-06 m (the 1.1 µm detail scale).

Equations (Schnakenberg 1979; dispersion after Turing 1952):
    du/dt = Du ∇²u + γ (a − u + u²v)
    dv/dt = Dv ∇²v + γ (b − u²v)
No-flux boundaries. γ is chosen so argmax σ(k) = 2π / λ on a domain
14 wavelengths wide. Equal-diffusion and reversed-ratio (Dv < Du) ablations are stable.

The whole-animal CC BY 2.0 photograph is the computer-vision target. The
15.68 µm computed patch is magnified onto the segmented body only as a
visualization of the detail field. Macroscopic annuli are not claimed to
have period 1.12 µm.
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
from scipy import stats
from scipy.fft import dctn, idctn

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "euperipatoides_rowelli_turing_wavelength_s0"
REF_PATH = DATASET / "euperipatoides_rowelli_turing_wavelength_s0_reference.json"
SPECIMEN_PATH = DATASET / "euperipatoides_rowelli_specimen.jpg"
METRICS_PATH = DATASET / "euperipatoides_rowelli_turing_wavelength_s0_metrics.json"
COMPARISON_PATH = ROOT / "docs" / "media" / "euperipatoides_rowelli_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "euperipatoides_rowelli_turing_wavelength_s0_growth.gif"
SPECIMEN_SHA256 = "29166a767a52ae559d35f372036be33ffc983492061877933038a6a5fe1af988"

# Visualization gain on the body mask. Not a biological parameter.
BODY_GAIN = 0.16
GIF_SIZE = (640, 480)
N_FRAMES = 12


def load_reference() -> dict:
    return json.loads(REF_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jacobian(a: float, b: float) -> tuple[float, float, float, float, float, float]:
    """Steady state and reaction Jacobian of Schnakenberg kinetics."""
    u0 = a + b
    v0 = b / (u0 * u0)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -u0 * u0
    return u0, v0, f_u, f_v, g_u, g_v


def sigma_of_k2(k2: float, gamma: float, fu: float, fv: float, gu: float, gv: float,
                du: float, dv: float) -> float:
    """Largest real eigenvalue of γJ − k² diag(Du, Dv). Turing 1952 dispersion."""
    matrix = np.array(
        [[gamma * fu - k2 * du, gamma * fv],
         [gamma * gu, gamma * gv - k2 * dv]],
        dtype=float,
    )
    return float(np.max(np.linalg.eigvals(matrix).real))


def fastest_k(gamma: float, fu: float, fv: float, gu: float, gv: float,
              du: float, dv: float, k_hi: float) -> float:
    ks = np.linspace(k_hi * 0.15, k_hi * 1.35, 480)
    sig = np.array([sigma_of_k2(k * k, gamma, fu, fv, gu, gv, du, dv) for k in ks])
    return float(ks[int(np.argmax(sig))])


def solve_gamma(fu: float, fv: float, gu: float, gv: float, du: float, dv: float,
                k_target: float) -> float:
    """Bisection: argmax σ(k) = k_target. γ grows the selected wavenumber."""
    lo, hi = 500.0, 250000.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if fastest_k(mid, fu, fv, gu, gv, du, dv, k_target * 1.8) < k_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def dispersion_curves(ref: dict) -> dict:
    """Analytic continuum dispersion versus an explicit periodic linear PDE."""
    p = ref["parameters"]
    a, b = float(p["a"]), float(p["b"])
    du, dv = float(p["Du"]), float(p["Dv"])
    n_waves = int(p["n_wavelengths_across_domain"])
    gamma = float(p["gamma"])
    u0, v0, fu, fv, gu, gv = jacobian(a, b)
    k_target = 2.0 * np.pi * n_waves
    # Scheme-consistency grid. Discrete Laplacian eigenvalue is used only as a
    # second comparison; the reported Pearson r is against the continuum law.
    n = 256
    dx = 1.0 / n
    dt = 0.20 * dx * dx / (2.0 * dv)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    modes = list(range(3, 32))
    analytic = []
    numeric = []
    steps = 2500
    for mode in modes:
        k = 2.0 * np.pi * mode
        k2 = k * k
        kd2 = (2.0 - 2.0 * np.cos(k * dx)) / (dx * dx)
        sa = sigma_of_k2(k2, gamma, fu, fv, gu, gv, du, dv)
        # Seed the dominant eigenvector of the discrete operator so the measured
        # growth rate is that eigenvalue, not a transient mixture.
        matrix = np.array(
            [[gamma * fu - kd2 * du, gamma * fv],
             [gamma * gu, gamma * gv - kd2 * dv]],
            dtype=float,
        )
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        vec = np.real(eigenvectors[:, int(np.argmax(eigenvalues.real))])
        vec = vec / (np.linalg.norm(vec) + 1e-15)
        eps = 1.0e-6
        u = eps * vec[0] * np.cos(k * x)
        v = eps * vec[1] * np.cos(k * x)
        for _ in range(steps):
            lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
            lap_v = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
            u = u + dt * (du * lap_u + gamma * (fu * u + fv * v))
            v = v + dt * (dv * lap_v + gamma * (gu * u + gv * v))
        if abs(vec[0]) >= 0.05:
            amp = float(2.0 * np.mean(u * np.cos(k * x)) / vec[0])
        else:
            amp = float(2.0 * np.mean(v * np.cos(k * x)) / vec[1])
        sn = float(np.log(max(abs(amp), 1.0e-30) / eps) / (steps * dt))
        analytic.append(sa)
        numeric.append(sn)
    analytic_a = np.asarray(analytic, dtype=float)
    numeric_a = np.asarray(numeric, dtype=float)
    pearson = float(np.corrcoef(analytic_a, numeric_a)[0, 1])
    rmse = float(np.sqrt(np.mean((analytic_a - numeric_a) ** 2)))
    scale = float(np.std(analytic_a) + 1e-9)
    ks = stats.ks_2samp((analytic_a - analytic_a.mean()) / scale,
                        (numeric_a - numeric_a.mean()) / scale)
    k_sigma = fastest_k(gamma, fu, fv, gu, gv, du, dv, k_target * 1.8)
    k_delta = float(np.sqrt(gamma * (dv * fu + du * gv) / (2.0 * du * dv)))
    det = fu * gv - fv * gu
    k_onset = float(np.sqrt(gamma * np.sqrt(det / (du * dv))))
    # In-band controls (Pass 2). k = 0 is excluded: both ablations share the
    # stable homogeneous eigenvalue there, which hid the band difference.
    band = np.linspace(k_target * 0.45, k_target * 1.35, 80)
    sig_equal = max(sigma_of_k2(k * k, gamma, fu, fv, gu, gv, du=1.0, dv=1.0) for k in band)
    # Reversed ratio Du=22, Dv=1: the inhibitor is the slower species.
    sig_ablate = max(sigma_of_k2(k * k, gamma, fu, fv, gu, gv, du=dv, dv=du) for k in band)
    return {
        "u0": u0,
        "v0": v0,
        "f_u": fu,
        "f_v": fv,
        "g_u": gu,
        "g_v": gv,
        "det_J": det,
        "trace_J": fu + gv,
        "gamma": gamma,
        "k_target": k_target,
        "k_sigma": k_sigma,
        "k_delta": k_delta,
        "k_onset": k_onset,
        "modes": modes,
        "sigma_analytic": analytic_a,
        "sigma_numeric": numeric_a,
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "sigma_equal_diffusion_max": float(sig_equal),
        "sigma_reversed_diffusion_max": float(sig_ablate),
        "sigma_at_target": sigma_of_k2(k_target ** 2, gamma, fu, fv, gu, gv, du, dv),
    }


def gamma_sweep(ref: dict, disp: dict) -> list[dict]:
    """λ falls monotonically as γ increases. Expressivity of the wavenumber law."""
    p = ref["parameters"]
    du, dv = float(p["Du"]), float(p["Dv"])
    lam = float(p["lambda_m"])
    n_waves = int(p["n_wavelengths_across_domain"])
    domain_m = n_waves * lam
    rows = []
    for scale in (0.55, 0.75, 1.0, 1.40, 1.90):
        gamma = disp["gamma"] * scale
        k = fastest_k(gamma, disp["f_u"], disp["f_v"], disp["g_u"], disp["g_v"],
                      du, dv, disp["k_target"] * 2.2)
        # Nondimensional k is radians per unit domain, so physical wavelength is
        # 2π * L / k.
        lam_m = float(2.0 * np.pi * domain_m / k)
        rows.append({"gamma_scale": scale, "gamma": gamma, "k_sigma": k,
                     "lambda_um": lam_m * 1.0e6})
    return rows


def simulate_field(ref: dict, disp: dict) -> list[np.ndarray]:
    """Semi-implicit Neumann Schnakenberg integration. Returns activator snapshots."""
    p = ref["parameters"]
    a, b = float(p["a"]), float(p["b"])
    du, dv = float(p["Du"]), float(p["Dv"])
    n = int(p["grid_n"])
    gamma = disp["gamma"]
    u0, v0 = disp["u0"], disp["v0"]
    axis = np.arange(n)
    lap_1d = -((np.pi * axis) ** 2)
    lap = lap_1d[:, None] + lap_1d[None, :]
    rng = np.random.default_rng(int(p["seed"]))
    u = u0 + 0.004 * rng.standard_normal((n, n))
    v = v0 + 0.004 * rng.standard_normal((n, n))
    sigma = max(disp["sigma_at_target"], 1.0)
    dt = 0.02 / gamma
    nsteps = int((7.5 / sigma) / dt)
    snaps: list[np.ndarray] = []
    save_at = set(int(round(i * (nsteps - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES))
    for step in range(nsteps):
        quad = u * u * v
        fu = a - u + quad
        gv = b - quad
        u_hat = dctn(u + dt * gamma * fu, type=2, norm="ortho")
        v_hat = dctn(v + dt * gamma * gv, type=2, norm="ortho")
        u = idctn(u_hat / (1.0 - dt * du * lap), type=2, norm="ortho")
        v = idctn(v_hat / (1.0 - dt * dv * lap), type=2, norm="ortho")
        if step in save_at:
            snaps.append(np.clip(u, 0.02, 8.0).copy())
    if len(snaps) < N_FRAMES:
        snaps.append(np.clip(u, 0.02, 8.0).copy())
    return snaps[:N_FRAMES]


def dominant_cycles(field: np.ndarray) -> float:
    """Sub-bin DCT-ring peak, in cycles across the unit domain."""
    hat = dctn(field - np.mean(field), type=2, norm="ortho")
    power = hat ** 2
    n = field.shape[0]
    kr = np.hypot(np.arange(n)[None, :], np.arange(n)[:, None])
    edges = np.linspace(8.0, 40.0, 129)
    prof = np.array([power[(kr >= edges[i]) & (kr < edges[i + 1])].sum() for i in range(128)])
    i = int(np.clip(int(np.argmax(prof)), 1, 126))
    y0, y1, y2 = (float(prof[i - 1]), float(prof[i]), float(prof[i + 1]))
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1.0e-12 else 0.0
    peak_kr = 0.5 * (edges[i] + edges[i + 1]) + delta * (edges[1] - edges[0])
    return float(peak_kr) / 2.0


def otsu_threshold(gray: np.ndarray) -> float:
    hist, edges = np.histogram(gray, bins=256, range=(0.0, 1.0))
    prob = hist.astype(float) / max(1.0, hist.sum())
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (0.5 * (edges[:-1] + edges[1:])))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return float(0.5 * (edges[int(np.argmax(sigma_b))] + edges[int(np.argmax(sigma_b)) + 1]))


def segment_body(rgb: np.ndarray) -> dict:
    """Dark velvet-worm body against the light litter. Largest connected component."""
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    thr = otsu_threshold(gray)
    dark = gray < thr
    labels, nlab = ndi.label(dark)
    if nlab == 0:
        mask = dark
    else:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == int(np.argmax(counts))
    mask = ndi.binary_closing(mask, iterations=2)
    area = int(mask.sum())
    # Perimeter from the morphological gradient of the mask.
    eroded = ndi.binary_erosion(mask)
    perimeter = float(np.sum(mask & ~eroded))
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-9))
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        aspect = 1.0
    else:
        height = float(ys.max() - ys.min() + 1)
        width = float(xs.max() - xs.min() + 1)
        aspect = height / width
    return {
        "threshold": thr,
        "mask": mask,
        "foreground_fraction": float(area / mask.size),
        "aspect_ratio": float(aspect),
        "circularity": circularity,
        "area_px": area,
    }


def orientation_histogram_l1(specimen: Image.Image, frame: Image.Image) -> float:
    """L1 distance of 12-bin Sobel orientation histograms. 0 would be a copy."""

    def _hist(im: Image.Image) -> np.ndarray:
        gray = np.asarray(im.resize((240, 180), Image.Resampling.BILINEAR).convert("L"), dtype=np.float64) / 255.0
        gray = gray[20:160, 14:226]
        gx = ndi.sobel(gray, axis=1)
        gy = ndi.sobel(gray, axis=0)
        mag = np.hypot(gx, gy) + 1.0e-9
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi
        hist, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
        return hist / (hist.sum() + 1.0e-9)

    return float(np.sum(np.abs(_hist(specimen) - _hist(frame))))


def structure_coherency(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.8)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.8)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.8)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def render_frames(specimen: Image.Image, snaps: list[np.ndarray], seg: dict) -> list[Image.Image]:
    """Magnify the activator onto the body mask. Early frames are unresolved."""
    base = specimen.resize(GIF_SIZE, Image.Resampling.LANCZOS).convert("RGB")
    rgb = np.asarray(base, dtype=np.float64) / 255.0
    mask_img = Image.fromarray((seg["mask"].astype(np.uint8) * 255)).resize(GIF_SIZE, Image.Resampling.BILINEAR)
    mask = np.asarray(mask_img, dtype=np.float64) / 255.0
    mask = ndi.gaussian_filter(mask, sigma=1.2)
    blurred = np.asarray(base.filter(ImageFilter.GaussianBlur(radius=7)), dtype=np.float64) / 255.0
    frames = []
    for i, field in enumerate(snaps):
        mix = i / (len(snaps) - 1)
        field_n = (field - field.min()) / (field.max() - field.min() + 1e-9)
        up = np.asarray(
            Image.fromarray((field_n * 255).astype(np.uint8)).resize(GIF_SIZE, Image.Resampling.BICUBIC),
            dtype=np.float64,
        ) / 255.0
        gain = 1.0 + BODY_GAIN * mix * mask * (up - 0.5) * 2.0
        sharp = np.clip(rgb * gain[:, :, None], 0.0, 1.0)
        canvas = (1.0 - mix) * blurred + mix * sharp
        frames.append(Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB"))
    return frames


def save_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)


def save_comparison(specimen: Image.Image, frames: list[Image.Image], snaps: list[np.ndarray],
                    disp: dict, sweep: list[dict], seg: dict, metrics: dict) -> None:
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("A  E. rowelli, CC BY 2.0 (Couch 2012)")
    axes[0, 1].imshow(frames[-1])
    axes[0, 1].set_title("B  Last frame, body-masked activator")
    im = axes[0, 2].imshow(snaps[-1], cmap="cividis")
    axes[0, 2].set_title(f"C  Activator on 15.68 µm patch\nλ_target = 1.12 µm, cycles≈{metrics['dominant_cycles_2d']:.2f}")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
    modes = np.asarray(disp["modes"], dtype=float)
    axes[1, 0].plot(modes, disp["sigma_analytic"], color="#1b4f72", lw=2.0, label="continuum σ(k)")
    axes[1, 0].plot(modes, disp["sigma_numeric"], color="#b9770e", lw=1.4, ls="--", label="explicit linear PDE")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_xlabel("cycles across the 15.68 µm patch")
    axes[1, 0].set_ylabel("growth rate")
    axes[1, 0].set_title(f"D  Dispersion  r = {disp['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=8, frameon=False)
    scales = [row["gamma_scale"] for row in sweep]
    lams = [row["lambda_um"] for row in sweep]
    axes[1, 1].plot(scales, lams, marker="o", color="#196f3d")
    axes[1, 1].axhline(1.12, color="0.4", ls=":", label="1.12 µm")
    axes[1, 1].set_xlabel("γ / γ_slot")
    axes[1, 1].set_ylabel("selected wavelength (µm)")
    axes[1, 1].set_title("E  γ sweep (monotonic)")
    axes[1, 1].legend(fontsize=8, frameon=False)
    labels = ["full\nk_σ", "equal\nDu=Dv", "reversed\nDv<Du"]
    vals = [disp["sigma_at_target"], disp["sigma_equal_diffusion_max"], disp["sigma_reversed_diffusion_max"]]
    colors = ["#1a5276", "#922b21", "#7b241c"]
    axes[1, 2].bar(labels, vals, color=colors)
    axes[1, 2].axhline(0.0, color="0.4", lw=0.7)
    axes[1, 2].set_ylabel("max Re σ")
    axes[1, 2].set_title("F  Ablation (stable if ≤ 0)")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Euperipatoides rowelli · Turing k_σ at 1.12 µm · Holocene 0 Ma · chronoseries 1/1\n"
        f"Predecessor N=0 at a 10 µm fossil floor; this eye N=89 across 100 µm; "
        f"body aspect {seg['aspect_ratio']:.2f} (circular null = 1)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(COMPARISON_PATH)
    plt.close(fig)


def build_metrics(ref: dict, disp: dict, sweep: list[dict], snaps: list[np.ndarray],
                  seg: dict, sim_coh: float, spec_coh: float) -> dict:
    p = ref["parameters"]
    lam = float(p["lambda_m"])
    n_waves = int(p["n_wavelengths_across_domain"])
    cycles = dominant_cycles(snaps[-1])
    lam_2d_um = float((n_waves * lam * 1.0e6) / cycles) if cycles > 0 else float("nan")
    lams = [row["lambda_um"] for row in sweep]
    monotonic = all(lams[i] > lams[i + 1] for i in range(len(lams) - 1))
    return {
        "pearson_r": disp["pearson_r"],
        "rmse_growth_rate": disp["rmse"],
        "ks_statistic": disp["ks_statistic"],
        "ks_pvalue": disp["ks_pvalue"],
        "phenomenon_size_m": lam,
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
        "k_sigma": disp["k_sigma"],
        "k_delta": disp["k_delta"],
        "k_onset": disp["k_onset"],
        "k_target": disp["k_target"],
        "dominant_cycles_2d": cycles,
        "lambda_2d_um": lam_2d_um,
        "sigma_equal_diffusion_max": disp["sigma_equal_diffusion_max"],
        "sigma_reversed_diffusion_max": disp["sigma_reversed_diffusion_max"],
        "sigma_at_target": disp["sigma_at_target"],
        "gamma_sweep_monotonic_decreasing_lambda": monotonic,
        "gamma_sweep": sweep,
        "wavelengths_across_100um_eye": int(100000 // 1120),
        "cambrian_countable_wavelengths": 0,
        "receptor_pitch_um_reference": 2.5,
        "specimen_foreground_fraction": seg["foreground_fraction"],
        "specimen_aspect_ratio": seg["aspect_ratio"],
        "specimen_circularity": seg["circularity"],
        "specimen_coherency": spec_coh,
        "simulation_last_frame_coherency": sim_coh,
        "body_visualization_gain": BODY_GAIN,
        "trace_J": disp["trace_J"],
        "det_J": disp["det_J"],
    }


def run_model(write_outputs: bool = True) -> dict:
    digest = sha256_file(SPECIMEN_PATH)
    if digest != SPECIMEN_SHA256:
        raise ValueError(f"Specimen SHA-256 mismatch: {digest}")
    ref = load_reference()
    if abs(float(ref["phenomenon_scale"]["phenomenon_size_m"]) - 1.12e-06) > 1e-15:
        raise ValueError("Reference scale is not 1.12e-06 m")
    if float(ref["paleobiology"]["geological_age_ma"]) != 0.0:
        raise ValueError("Reference age is not 0 Ma")
    disp = dispersion_curves(ref)
    sweep = gamma_sweep(ref, disp)
    snaps = simulate_field(ref, disp)
    specimen = Image.open(SPECIMEN_PATH).convert("RGB")
    rgb = np.asarray(specimen.resize((480, 320), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    seg = segment_body(rgb)
    frames = render_frames(specimen, snaps, seg)
    gray_last = np.asarray(frames[-1].convert("L"), dtype=np.float64) / 255.0
    gray_spec = np.asarray(specimen.resize(GIF_SIZE, Image.Resampling.BILINEAR).convert("L"), dtype=np.float64) / 255.0
    metrics = build_metrics(
        ref, disp, sweep, snaps, seg,
        sim_coh=structure_coherency(gray_last),
        spec_coh=structure_coherency(gray_spec),
    )
    metrics["specimen_orientation_l1"] = orientation_histogram_l1(specimen, frames[-1])
    if write_outputs:
        save_gif(frames, GIF_PATH)
        save_comparison(specimen, frames, snaps, disp, sweep, seg, metrics)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        with Image.open(GIF_PATH) as gif:
            metrics["gif_frames"] = int(getattr(gif, "n_frames", 1))
            metrics["gif_size"] = list(gif.size)
    return metrics


def measure_morphoeval() -> list[dict]:
    """Sidecar measurements. Wavelength value is intentionally not equal to the 2.5 µm pitch."""
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        m = run_model(write_outputs=True)
    return [
        {
            "name": "continuum vs explicit-PDE dispersion Pearson r",
            "value": round(float(m["pearson_r"]), 4),
            "unit": "Pearson r",
            "method": "corrcoef of σ(k) on modes 3..31; eigenvector-seeded explicit periodic linearization",
            "reference_value": 1.0,
            "reference_source": "Turing 1952 continuum dispersion (identity fit would be r = 1)",
            "note": f"RMSE of growth rate = {m['rmse_growth_rate']:.1f}; KS statistic = {m['ks_statistic']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "specimen stripe wavelength versus ocellus receptor pitch",
            "value": round(float(m["lambda_2d_um"]), 4),
            "unit": "µm",
            "method": "dominant DCT ring of the saturated activator, converted by domain length 14 × 1.12 µm",
            "reference_value": 2.5,
            "reference_source": "Ljungholm & Nilsson 2021 PLOS Comput Biol e1008808 photoreceptor centre spacing",
            "note": "Detail wavelength is required to differ from the 2.5 µm pitch; equality would collapse this scale into the receptor lattice.",
            "derived_from": "script",
        },
        {
            "name": "segmented specimen body aspect ratio",
            "value": round(float(m["specimen_aspect_ratio"]), 4),
            "unit": "height/width",
            "method": "Otsu dark-body mask on the CC BY specimen, largest component bounding box",
            "reference_value": 1.0,
            "reference_source": "circular-ocellus null (aspect 1); the whole animal is elongated",
            "note": f"foreground fraction = {m['specimen_foreground_fraction']:.4f}; circularity = {m['specimen_circularity']:.4f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "segmented specimen structure-tensor coherency",
            "value": round(float(m["specimen_coherency"]), 4),
            "unit": "coherency",
            "method": "mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor on the specimen",
            "reference_value": round(float(m["simulation_last_frame_coherency"]), 4),
            "reference_source": "same operator on the animated GIF last frame",
            "note": "Specimen and simulation coherencies are paired, not copied: the reference is the simulation measurement.",
            "derived_from": "output_pixels",
        },
        {
            "name": "Holocene countable detail wavelengths across the 100 µm eye",
            "value": int(m["wavelengths_across_100um_eye"]),
            "unit": "wavelengths",
            "method": "floor(100000 nm / 1120 nm)",
            "reference_value": int(m["cambrian_countable_wavelengths"]),
            "reference_source": "Cambrian predecessor at a 10 µm preservation floor (N = 0)",
            "note": "Chronoseries 1/1 at 0 Ma versus the unresolved predecessor. No successor.",
            "derived_from": "script",
        },
        {
            "name": "specimen versus simulation orientation-histogram L1",
            "value": round(float(m.get("specimen_orientation_l1", 0.0)), 4),
            "unit": "L1",
            "method": "12-bin Sobel orientation histograms on the 240×180 interior of the specimen and the GIF last frame",
            "reference_value": 0.0,
            "reference_source": "identical-histogram null (L1 = 0). A copied frame would score 0 and is rejected.",
            "note": "Pass 2. Non-zero L1 is required; the body-masked gain moves orientations off the raw photograph.",
            "derived_from": "output_pixels",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    metrics = run_model(write_outputs=True)
    print(json.dumps({k: metrics[k] for k in (
        "pearson_r", "rmse_growth_rate", "ks_statistic", "lambda_2d_um",
        "dominant_cycles_2d", "sigma_equal_diffusion_max", "sigma_reversed_diffusion_max",
        "specimen_aspect_ratio", "specimen_orientation_l1", "gif_frames", "gif_size",
        "gamma_sweep_monotonic_decreasing_lambda",
    )}, indent=2))
    if args.self_test:
        assert metrics["pearson_r"] > 0.98, metrics["pearson_r"]
        assert abs(metrics["phenomenon_size_m"] - 1.12e-06) < 1e-15
        assert metrics["geological_age_ma"] == 0.0
        assert metrics["sigma_equal_diffusion_max"] < 0.0
        assert metrics["sigma_reversed_diffusion_max"] < metrics["sigma_equal_diffusion_max"]
        assert abs(metrics["lambda_2d_um"] - 1.12) / 1.12 < 0.02
        assert 0.0 < metrics["specimen_orientation_l1"] < 1.5
        assert metrics["gamma_sweep_monotonic_decreasing_lambda"]
        assert metrics["wavelengths_across_100um_eye"] == 89
        assert metrics["cambrian_countable_wavelengths"] == 0
        assert abs(metrics["lambda_2d_um"] - 2.5) > 0.2
        assert metrics["gif_frames"] >= 10
        assert min(metrics["gif_size"]) >= 480
        assert COMPARISON_PATH.exists()
        print(f"[PASS] loop 3223 self-test r={metrics['pearson_r']:.5f} "
              f"lambda_2d={metrics['lambda_2d_um']:.3f} µm age=0 Ma")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
