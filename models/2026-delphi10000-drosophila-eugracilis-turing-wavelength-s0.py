#!/usr/bin/env python3
"""
Drosophila eugracilis — Turing wavenumber at the 1.5 µm detail scale (slot 3184).

Organism: Drosophila eugracilis (Southeast Asian Gracile Fruit Fly; reduced tarsal
sex-comb teeth). Scale: 1.485e-06 m (1.5 µm, log10=-5.828, tier 3_subcellular_um,
detail variant s0). Age: Holocene (Extant), 0 Ma. Chronoseries 13/41, predecessor
Drosophila erecta, successor Drosophila ficusphila.

Governing law (Turing 1952; Schnakenberg 1979 kinetics; Murray 1981 scale γ):

    ∂u/∂t = D_u ∇²u + γ (a − u + u² v)
    ∂v/∂t = D_v ∇²v + γ (b − u² v)

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c.

γ multiplies the Jacobian, so λ_c(γ) = λ_c(1) / sqrt(γ). The length unit is fixed
so λ_c(γ=1) = 1.485 µm for D. eugracilis. D. erecta uses γ=0.5 (longer wave);
D. ficusphila uses the γ that selects 1.180 µm (shorter wave). This is not the
101 µm primordium slot and does not reuse its parameters.

Formalization: morphoeval/lean/loops/Loop_3184_drosophila_eugracilis_turing_wavelength_s0.lean
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
DATASET = ROOT / "datasets" / "drosophila_eugracilis_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_eugracilis_turing_wavelength_s0_specimen.jpg"
REFERENCE = DATASET / "drosophila_eugracilis_turing_wavelength_s0_reference.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_eugracilis_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_eugracilis_turing_wavelength_s0_growth.gif"
METRICS_LATEST = DATASET / "metrics_latest.json"
PASS1_JSON = DATASET / "pass1_metrics.json"
PASS2_JSON = DATASET / "pass2_metrics.json"

# SHA-256 of the vendored dataset files. The reference JSON records the specimen
# digest; these locks cover every file in the dataset directory that is an input.
EXPECTED_SHA256 = {
    "README.md": "6814a062121581e4194427a0c2c7c2a56ee617afdbf198a5615858e513191839",
    "drosophila_eugracilis_turing_wavelength_s0_reference.json": "aeae6e2ee094bbd3ed190ab532cf3e4df5c1b267c6d34ebacb231137143ae640",
    "drosophila_eugracilis_turing_wavelength_s0_specimen.jpg": "f3705e558598c7f56969bf370f2658e280972cb8b54e67320dd511ab6d2dde33",
    "paper_citations.bib": "963fd37391fe7bc7b9987c5550ad5261d482e941cf41658cdde759ad0f09fc7f",
    "scout_manifest.json": "65cbd08f09bd7095ed61b8b3c964e42acfe99b0d2b2b9b51955b349e63faed98",
}

GIF_SIZE = (720, 540)
HERO_STEPS = (0, 40, 80, 140, 220, 320, 450, 600, 800, 1000, 1300, 1600)


def verify_dataset_integrity() -> dict:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    pinned = ref["specimen"]["sha256"]
    if pinned != EXPECTED_SHA256[SPECIMEN.name]:
        raise RuntimeError("reference JSON specimen sha256 does not match the locked digest")
    return ref


def jacobian(a: float, b: float, gamma: float) -> tuple[float, float, float, float, float, float]:
    """Schnakenberg Jacobian at the homogeneous steady state, scaled by Murray γ.

    f = a - u + u²v, g = b - u²v, steady state u0 = a+b, v0 = b/u0².
    f_u = -1 + 2 u0 v0, f_v = u0², g_u = -2 u0 v0, g_v = -u0², then × γ.
    """
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = gamma * (-1.0 + 2.0 * u0 * v0)
    fv = gamma * (u0 * u0)
    gu = gamma * (-2.0 * u0 * v0)
    gv = gamma * (-(u0 * u0))
    return fu, fv, gu, gv, u0, v0


def turing_wavenumber(a: float, b: float, du: float, dv: float, gamma: float) -> dict:
    """Critical Turing wavenumber. Returns NaNs if the Turing conditions fail."""
    fu, fv, gu, gv, u0, v0 = jacobian(a, b, gamma)
    trace = fu + gv
    det = fu * gv - fv * gu
    h = dv * fu + du * gv
    disc = h * h - 4.0 * du * dv * det
    ok = bool(trace < 0.0 and det > 0.0 and h > 0.0 and disc > 0.0)
    kc = float(np.sqrt(h / (2.0 * du * dv))) if ok else float("nan")
    return {
        "fu": fu, "fv": fv, "gu": gu, "gv": gv,
        "u0": u0, "v0": v0,
        "trace": trace, "det": det, "h": h, "discriminant": disc,
        "k_c": kc,
        "lambda_nd": float(2.0 * np.pi / kc) if ok else float("nan"),
        "turing": ok,
    }


def growth_rate(k: np.ndarray, spec: dict, du: float, dv: float) -> np.ndarray:
    """σ(k) = ½ (tr_k + sqrt(tr_k² − 4 det_k)), the unstable branch."""
    fu, gv, det = spec["fu"], spec["gv"], spec["det"]
    tr_k = (fu + gv) - (k ** 2) * (du + dv)
    det_k = du * dv * k ** 4 - (dv * fu + du * gv) * k ** 2 + det
    rad = np.maximum(0.0, tr_k ** 2 - 4.0 * det_k)
    return 0.5 * (tr_k + np.sqrt(rad))


def length_unit_m(lambda_nd_focal: float, lambda_focal_m: float) -> float:
    """Metres per nondimensional length so the focal species sits at lambda_focal_m."""
    return float(lambda_focal_m / lambda_nd_focal)


def lambda_m_of_gamma(gamma: float, lambda_focal_m: float, gamma_focal: float = 1.0) -> float:
    """Exact γ-scaling: λ(γ) = λ(γ_focal) * sqrt(γ_focal / γ)."""
    return float(lambda_focal_m * np.sqrt(gamma_focal / gamma))


def _fourier_grids(n: int, length: float) -> tuple[np.ndarray, float]:
    dx = length / n
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    return kx ** 2 + ky ** 2, dx


def simulate_schnakenberg(
    *,
    a: float,
    b: float,
    du: float,
    dv: float,
    gamma: float,
    length: float,
    n: int,
    dt: float,
    n_steps: int,
    seed: int,
    spiral: bool,
    k_c: float,
    snapshots: tuple[int, ...] | None = None,
) -> dict:
    """Pseudospectral Schnakenberg. Diffusion is integrated exactly in Fourier space."""
    k2, dx = _fourier_grids(n, length)
    spec = turing_wavenumber(a, b, du, dv, gamma)
    u0, v0 = spec["u0"], spec["v0"]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - n / 2) * dx
    y = (yy - n / 2) * dx
    if spiral and np.isfinite(k_c):
        radius = np.hypot(x, y) + 1e-8
        theta = np.arctan2(y, x)
        env = np.exp(-((radius / (0.46 * length / 2.0)) ** 2))
        phase = 2.0 * theta + k_c * radius
        u = u0 + 0.12 * np.cos(phase) * env + 0.004 * rng.normal(size=(n, n))
        v = v0 - 0.05 * np.cos(phase) * env
    else:
        u = u0 + 0.01 * rng.normal(size=(n, n))
        v = v0 + 0.01 * rng.normal(size=(n, n))
    frames: list[np.ndarray] = []
    std_trace: list[tuple[int, float]] = []
    snap_set = set(snapshots or ())
    step = 0
    if 0 in snap_set:
        frames.append(u.copy())
        std_trace.append((0, float(u.std())))
    while step < n_steps:
        burst = min(20, n_steps - step)
        for _ in range(burst):
            u2v = u * u * v
            u = np.clip(u + dt * gamma * (a - u + u2v), 0.0, 6.0)
            v = np.clip(v + dt * gamma * (b - u2v), 0.0, 6.0)
            u = np.fft.ifft2(np.fft.fft2(u) * np.exp(-du * k2 * dt)).real
            v = np.fft.ifft2(np.fft.fft2(v) * np.exp(-dv * k2 * dt)).real
            step += 1
            if step in snap_set:
                frames.append(u.copy())
                std_trace.append((step, float(u.std())))
        if step % 100 == 0 and step not in snap_set:
            std_trace.append((step, float(u.std())))
    return {"u": u, "v": v, "frames": frames, "std_trace": std_trace, "dx": dx, "spec": spec}


def dominant_wavelength(field: np.ndarray, dx: float) -> float:
    """Wavelength 2π/k of the dominant radial Fourier ring, with parabolic peak refine."""
    win = field - np.mean(field)
    std = float(np.std(win))
    if std < 1e-8:
        return float("nan")
    win = win / std
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    n = field.shape[0]
    cy = cx = n // 2
    yy, xx = np.ogrid[:n, :n]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    prof = np.bincount(rr.ravel(), weights=power.ravel())
    counts = np.maximum(1, np.bincount(rr.ravel()))
    prof = prof / counts
    hi = max(3, n // 2 - 1)
    i = int(np.argmax(prof[1:hi])) + 1
    y1, y2, y3 = float(prof[i - 1]), float(prof[i]), float(prof[i + 1])
    denom = y1 - 2.0 * y2 + y3
    delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
    delta = float(np.clip(delta, -0.6, 0.6))
    k_idx = i + delta
    k_phys = k_idx * (2.0 * np.pi) / (n * dx)
    if k_phys <= 0:
        return float("nan")
    return float(2.0 * np.pi / k_phys)


def specimen_lut(rgb: np.ndarray) -> np.ndarray:
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    bins = np.linspace(0.0, 1.0, 33)
    lut = np.zeros((32, 3), dtype=np.float64)
    for i in range(32):
        sel = (lum >= bins[i]) & (lum < bins[i + 1] if i < 31 else lum <= bins[i + 1])
        if int(sel.sum()) < 20:
            lut[i] = lut[i - 1] if i else rgb.mean(axis=(0, 1))
        else:
            lut[i] = rgb[sel].mean(axis=0)
    return lut


def field_to_rgb(field: np.ndarray, lut: np.ndarray) -> np.ndarray:
    gmin, gmax = float(field.min()), float(field.max())
    g = (field - gmin) / (gmax - gmin + 1e-9)
    g = 0.45 + 0.55 * (g - g.mean()) / (g.std() + 1e-9) * 0.22
    g = np.clip(g, 0.0, 1.0)
    idx = np.clip((g * 31.0).astype(int), 0, 31)
    return lut[idx]


def save_growth_gif(frames_u: list[np.ndarray], lut: np.ndarray, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for field in frames_u:
        rgb = np.clip(field_to_rgb(field, lut), 0.0, 1.0)
        im = Image.fromarray((rgb * 255.0).astype(np.uint8)).resize(GIF_SIZE, Image.Resampling.BILINEAR)
        images.append(im)
    images[0].save(path, save_all=True, append_images=images[1:], duration=90, loop=0, optimize=False)
    with Image.open(path) as im:
        return {"n_frames": int(getattr(im, "n_frames", 1)), "width": im.size[0], "height": im.size[1]}


def segment_specimen(rgb: np.ndarray) -> dict:
    """Otsu segmentation of the canonical plate plus a structure-tensor coherency."""
    from skimage.filters import threshold_otsu

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    thr = float(threshold_otsu(gray))
    dark = gray < thr
    gy, gx = np.gradient(gray)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    ys, xs = np.where(dark)
    if len(xs) < 10:
        aspect, circularity = float("nan"), float("nan")
    else:
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        aspect = width / max(height, 1.0)
        area = float(dark.sum())
        perimeter = float(np.sum(dark ^ ndi.binary_erosion(dark)))
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-9))
    return {
        "otsu_threshold": thr,
        "foreground_fraction": float(dark.mean()),
        "coherency": coherency,
        "aspect_ratio": aspect,
        "circularity": circularity,
        "gray": gray,
    }


def segment_specimen_smoothed(rgb: np.ndarray) -> dict:
    """Pass-2 arm mask: blur the plate before Otsu so grain does not set the perimeter."""
    from skimage.filters import threshold_otsu

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    blur = ndi.gaussian_filter(gray, 3.5)
    thr = float(threshold_otsu(blur))
    dark = ndi.binary_opening(blur < thr, iterations=1)
    ys, xs = np.where(dark)
    width = float(xs.max() - xs.min() + 1) if xs.size else float("nan")
    height = float(ys.max() - ys.min() + 1) if ys.size else float("nan")
    area = float(dark.sum())
    perimeter = float(np.sum(dark ^ ndi.binary_erosion(dark)))
    return {
        "smoothed_foreground_fraction": float(dark.mean()),
        "smoothed_aspect_ratio": width / max(height, 1.0),
        "smoothed_circularity": float(4.0 * np.pi * area / (perimeter * perimeter + 1e-9)),
        "smoothed_otsu_threshold": thr,
    }


def ocellus_spacing_cv(field: np.ndarray, dx: float) -> dict:
    """Nearest-neighbour spacing of high-activator foci (periodic metric)."""
    thr = float(np.quantile(field, 0.82))
    mask = field >= thr
    labels, nlab = ndi.label(mask)
    cents = []
    for i in range(1, nlab + 1):
        ys, xs = np.where(labels == i)
        if ys.size < 4:
            continue
        cents.append((float(ys.mean()), float(xs.mean())))
    if len(cents) < 3:
        return {"n_foci": len(cents), "spacing_px": float("nan"), "spacing_cv": float("nan"), "spacing_nd": float("nan")}
    arr = np.asarray(cents)
    n = field.shape[0]
    dists = []
    for i in range(len(arr)):
        dy = np.abs(arr[:, 0] - arr[i, 0])
        dx_ = np.abs(arr[:, 1] - arr[i, 1])
        dy = np.minimum(dy, n - dy)
        dx_ = np.minimum(dx_, n - dx_)
        dist = np.hypot(dy, dx_)
        dist[i] = np.inf
        dists.append(float(dist.min()))
    d = np.asarray(dists)
    mean = float(d.mean())
    return {
        "n_foci": int(len(arr)),
        "spacing_px": mean,
        "spacing_cv": float(d.std() / (mean + 1e-9)),
        "spacing_nd": float(mean * dx),
    }


def fit_series(analytic: np.ndarray, measured: np.ndarray) -> dict:
    mask = np.isfinite(analytic) & np.isfinite(measured)
    a = analytic[mask]
    m = measured[mask]
    if a.size < 3:
        return {"pearson_r": float("nan"), "rmse": float("nan"), "n": int(a.size)}
    r = float(np.corrcoef(a, m)[0, 1])
    rmse = float(np.sqrt(np.mean((m - a) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(a.size)}


def ks_luminance(specimen_gray: np.ndarray, rendered_rgb: np.ndarray) -> float:
    sim_gray = 0.299 * rendered_rgb[..., 0] + 0.587 * rendered_rgb[..., 1] + 0.114 * rendered_rgb[..., 2]
    sim_u8 = (np.clip(sim_gray, 0, 1) * 255).astype(np.uint8)
    sim_small = np.asarray(
        Image.fromarray(sim_u8).resize((specimen_gray.shape[1], specimen_gray.shape[0]), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0
    stat = stats.ks_2samp(specimen_gray.ravel()[::8], sim_small.ravel()[::8], method="asymp")
    return float(stat.statistic)


def load_specimen_rgb() -> np.ndarray:
    return np.asarray(Image.open(SPECIMEN).convert("RGB"), dtype=np.float64) / 255.0


def params_from_ref(ref: dict) -> dict:
    p = ref["parameters"]
    lam_e = float(p["lambda_eugracilis_m"])
    lam_f = float(p["lambda_ficusphila_m"])
    gamma_e = float(p["gamma_eugracilis"])
    gamma_r = float(p["gamma_erecta"])
    gamma_f = float((lam_e / lam_f) ** 2)
    return {
        "a": float(p["schnakenberg_a"]),
        "b": float(p["schnakenberg_b"]),
        "du": float(p["D_u"]),
        "dv": float(p["D_v"]),
        "gamma_e": gamma_e,
        "gamma_r": gamma_r,
        "gamma_f": gamma_f,
        "lam_e": lam_e,
        "lam_f": lam_f,
        "lam_r": lambda_m_of_gamma(gamma_r, lam_e, gamma_e),
        "n": int(p["grid_n"]),
        "dt": float(p["dt"]),
        "seed": int(p["seed"]),
        "n_waves": int(p["detail_window_wavelengths"]),
    }


def run_case(p: dict, gamma: float, *, spiral: bool, n_steps: int, snapshots=None, n: int | None = None) -> dict:
    focal = turing_wavenumber(p["a"], p["b"], p["du"], p["dv"], p["gamma_e"])
    ell = length_unit_m(focal["lambda_nd"], p["lam_e"])
    spec = turing_wavenumber(p["a"], p["b"], p["du"], p["dv"], gamma)
    length = p["n_waves"] * focal["lambda_nd"]  # same detail window for every chronospecies
    out = simulate_schnakenberg(
        a=p["a"], b=p["b"], du=p["du"], dv=p["dv"], gamma=gamma,
        length=length, n=p["n"] if n is None else n, dt=p["dt"], n_steps=n_steps,
        seed=p["seed"], spiral=spiral, k_c=spec["k_c"], snapshots=snapshots,
    )
    lam_meas_nd = dominant_wavelength(out["u"], out["dx"])
    return {
        "gamma": gamma,
        "spec": spec,
        "lambda_analytic_m": lambda_m_of_gamma(gamma, p["lam_e"], p["gamma_e"]),
        "lambda_measured_m": float(lam_meas_nd * ell) if np.isfinite(lam_meas_nd) else float("nan"),
        "lambda_measured_nd": lam_meas_nd,
        "final_std": float(out["u"].std()),
        "u": out["u"],
        "frames": out["frames"],
        "std_trace": out["std_trace"],
        "length_unit_m": ell,
        "window_m": float(length * ell),
    }


def write_comparison_png(specimen_rgb: np.ndarray, hero: dict, p: dict, extra: dict | None, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last = field_to_rgb(hero["u"], specimen_lut(specimen_rgb))
    if extra is None:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), dpi=140)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.2), dpi=140)
        axes = axes.ravel()
    axes[0].imshow(specimen_rgb, interpolation="nearest")
    axes[0].set_title("Canonical plate (gallery fallback)")
    axes[1].imshow(np.clip(last, 0, 1), interpolation="nearest")
    axes[1].set_title("Schnakenberg ocelli, γ=1")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    ks = np.linspace(0.0, hero["spec"]["k_c"] * 2.4, 240)
    for gamma, label, color in (
        (p["gamma_r"], "erecta γ=0.50", "#8c5a12"),
        (p["gamma_e"], "eugracilis γ=1", "#1f4e79"),
        (p["gamma_f"], "ficusphila", "#7a1f1f"),
    ):
        spec = turing_wavenumber(p["a"], p["b"], p["du"], p["dv"], gamma)
        sig = growth_rate(ks, spec, p["du"], p["dv"])
        axes[2].plot(ks, sig, color=color, lw=1.8, label=label)
        axes[2].axvline(spec["k_c"], color=color, ls="--", lw=0.8)
    axes[2].axhline(0.0, color="0.5", lw=0.6)
    axes[2].set_xlabel("wavenumber k")
    axes[2].set_ylabel("growth rate σ(k)")
    axes[2].set_title("Dispersion: k_c shifts with γ")
    axes[2].legend(fontsize=7, frameon=False)
    if extra is not None:
        sw = extra["sweep"]
        axes[3].scatter(sw["analytic_um"], sw["measured_um"], c="#1f4e79", s=36, zorder=3)
        lo = min(sw["analytic_um"] + sw["measured_um"])
        hi = max(sw["analytic_um"] + sw["measured_um"])
        axes[3].plot([lo, hi], [lo, hi], color="0.6", lw=0.8, label="identity")
        axes[3].set_xlabel("analytic λ_c (µm)")
        axes[3].set_ylabel("FFT λ (µm)")
        axes[3].set_title(f"GOF  r={extra['pearson_r']:.3f}  RMSE={extra['rmse_um']:.3f} µm")
        names = ["D. erecta\npredecessor", "D. eugracilis\nfocal 0 Ma", "D. ficusphila\nsuccessor"]
        vals = [p["lam_r"] * 1e6, p["lam_e"] * 1e6, p["lam_f"] * 1e6]
        axes[4].bar(names, vals, color=["#8c5a12", "#1f4e79", "#7a1f1f"])
        axes[4].set_ylabel("λ_c (µm)")
        axes[4].set_title("1.5 µm detail chronoseries")
        axes[4].axhline(1.485, color="0.4", ls=":", lw=0.8)
        axes[5].bar(
            ["Turing\nD_v/D_u=32", "Ablation\nD_u=D_v"],
            [extra["turing_std"], extra["ablation_std"]],
            color=["#1f4e79", "#b0b0b0"],
        )
        axes[5].set_ylabel("final activator std")
        axes[5].set_title("Equal-diffusion negative control")
    fig.suptitle(
        "Drosophila eugracilis  ·  Turing k_c  ·  1.5 µm (1.485e-06 m)  ·  Holocene 0 Ma",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_metrics(ref: dict, p: dict, hero: dict, seg: dict, gif_info: dict, extra: dict | None, pass_id: int) -> dict:
    spacing = ocellus_spacing_cv(hero["u"], hero["length_unit_m"] and 1.0)
    # spacing_nd was computed with dx inside ocellus_spacing_cv from the field's dx.
    # Recompute with the hero field's own dx via dominant measurement already stored.
    rendered = field_to_rgb(hero["u"], specimen_lut(load_specimen_rgb()))
    ks = ks_luminance(seg["gray"], rendered)
    metrics = {
        "pass": pass_id,
        "slot": 3184,
        "sim_id": ref["sim_id"],
        "organism": "Drosophila eugracilis",
        "phenomenon_size_m": 1.485e-06,
        "geological_age_ma": 0.0,
        "gamma_eugracilis": p["gamma_e"],
        "gamma_erecta": p["gamma_r"],
        "gamma_ficusphila": p["gamma_f"],
        "lambda_analytic_eugracilis_m": p["lam_e"],
        "lambda_analytic_erecta_m": p["lam_r"],
        "lambda_analytic_ficusphila_m": p["lam_f"],
        "lambda_measured_eugracilis_m": hero["lambda_measured_m"],
        "k_c_nondim": hero["spec"]["k_c"],
        "trace": hero["spec"]["trace"],
        "det": hero["spec"]["det"],
        "discriminant": hero["spec"]["discriminant"],
        "length_unit_m": hero["length_unit_m"],
        "detail_window_m": hero["window_m"],
        "final_std": hero["final_std"],
        "specimen_otsu_foreground_fraction": seg["foreground_fraction"],
        "specimen_structure_tensor_coherency": seg["coherency"],
        "specimen_dark_arm_aspect_ratio": seg["aspect_ratio"],
        "specimen_dark_arm_circularity": seg["circularity"],
        "ks_luminance": ks,
        "gif": gif_info,
        "n_foci": spacing["n_foci"],
        "ocellus_spacing_cv": spacing["spacing_cv"],
    }
    if extra is not None:
        metrics.update(extra)
    return metrics


def measurements_from_metrics(m: dict) -> list[dict]:
    """Non-circular morphometrics. reference_value is never copied from value."""
    rows = [
        {
            "name": "analytic_turing_wavelength_um",
            "value": round(m["lambda_analytic_eugracilis_m"] * 1e6, 6),
            "unit": "µm",
            "reference_value": round(m["lambda_measured_eugracilis_m"] * 1e6, 6),
            "method": "Schnakenberg dispersion k_c at gamma=1, length unit fixed to the assigned 1.485 µm; reference is the nonlinear FFT wavelength of the same run",
        },
        {
            "name": "simulated_stripe_fft_wavelength_um",
            "value": round(m["lambda_measured_eugracilis_m"] * 1e6, 6),
            "unit": "µm",
            "reference_value": 1.485,
            "method": "Radial Fourier ring of the saturated activator field, parabolic peak, converted by the focal length unit",
        },
        {
            "name": "predecessor_erecta_wavelength_um",
            "value": round(m["lambda_analytic_erecta_m"] * 1e6, 6),
            "unit": "µm",
            "reference_value": 1.485,
            "method": "lambda(gamma=0.5) = 1.485/sqrt(0.5); predecessor contrast, not a published erecta SEM",
        },
        {
            "name": "successor_ficusphila_wavelength_um",
            "value": round(m["lambda_analytic_ficusphila_m"] * 1e6, 6),
            "unit": "µm",
            "reference_value": 1.485,
            "method": "Assigned successor wavelength 1.180 µm used to set gamma_ficusphila",
        },
        {
            "name": "specimen_stripe_arm_foreground_fraction",
            "value": round(m["specimen_otsu_foreground_fraction"], 6),
            "unit": "1",
            "reference_value": 0.3543,
            "method": "Otsu dark-arm fraction of the vendored gallery plate; reference is verify_image_fitness foreground_fraction (different estimator)",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": round(m["specimen_structure_tensor_coherency"], 6),
            "unit": "1",
            "reference_value": 0.8014,
            "method": "Mean structure-tensor coherency of the full plate; reference is the fitness-script crop signature",
        },
        {
            "name": "specimen_vs_simulation_ks_luminance",
            "value": round(m["ks_luminance"], 6),
            "unit": "1",
            "reference_value": 0.0,
            "method": "Two-sample KS statistic between plate luminance and the rendered activator; 0 would be identical histograms",
        },
        {
            "name": "ocellus_nearest_neighbor_spacing_cv",
            "value": None if not np.isfinite(m["ocellus_spacing_cv"]) else round(m["ocellus_spacing_cv"], 6),
            "unit": "1",
            "reference_value": 0.0,
            "method": "CV of periodic nearest-neighbour distances among high-activator foci; 0 is a perfect lattice",
        },
    ]
    if isinstance(m.get("smoothed_segmentation"), dict):
        sm = m["smoothed_segmentation"]
        rows.append({
            "name": "specimen_segment_smoothed_foreground_fraction",
            "value": round(sm["smoothed_foreground_fraction"], 6),
            "unit": "1",
            "reference_value": round(m["specimen_otsu_foreground_fraction"], 6),
            "method": "Gaussian-blurred Otsu arm fraction; reference is the raw grain-sensitive Otsu fraction from the same plate",
        })
        rows.append({
            "name": "specimen_segment_smoothed_circularity",
            "value": round(sm["smoothed_circularity"], 6),
            "unit": "1",
            "reference_value": round(m["specimen_dark_arm_circularity"], 6),
            "method": "Circularity of the blurred arm mask; reference is the raw Otsu perimeter, which grain inflates",
        })
    if "pearson_r" in m:
        rows.append({
            "name": "goodness_of_fit_pearson_r",
            "value": round(m["pearson_r"], 6),
            "unit": "1",
            "reference_value": 1.0,
            "method": "Pearson r of analytic λ_c(γ) versus nonlinear FFT λ across the γ sweep, including erecta, eugracilis and ficusphila",
        })
        rows.append({
            "name": "goodness_of_fit_rmse_um",
            "value": round(m["rmse_um"], 6),
            "unit": "µm",
            "reference_value": 0.0,
            "method": "RMSE between analytic and FFT wavelengths on the same sweep",
        })
        rows.append({
            "name": "ablation_equal_diffusion_activator_std",
            "value": round(m["ablation_std"], 6),
            "unit": "1",
            "reference_value": round(m["turing_std"], 6),
            "method": "Final activator std with D_u=D_v (Turing h<0) versus the patterned D_v/D_u=32 control, same noise seed",
        })
    return rows


def measure_morphoeval() -> list[dict]:
    if not METRICS_LATEST.exists():
        return [{"name": "metrics missing", "value": None, "unit": None, "method": "run the script before measure_morphoeval"}]
    m = json.loads(METRICS_LATEST.read_text(encoding="utf-8"))
    return measurements_from_metrics(m)


def _public_case(case: dict) -> dict:
    return {
        "gamma": case["gamma"],
        "lambda_analytic_m": case["lambda_analytic_m"],
        "lambda_measured_m": case["lambda_measured_m"],
        "final_std": case["final_std"],
        "k_c": case["spec"]["k_c"],
        "turing": case["spec"]["turing"],
    }


def execute(pass_id: int) -> dict:
    ref = verify_dataset_integrity()
    p = params_from_ref(ref)
    specimen = load_specimen_rgb()
    lut = specimen_lut(specimen)
    seg = segment_specimen(specimen)
    hero = run_case(p, p["gamma_e"], spiral=True, n_steps=HERO_STEPS[-1], snapshots=HERO_STEPS)
    gif_info = save_growth_gif(hero["frames"], lut, GIF_PATH)
    extra = None
    if pass_id >= 2:
        gammas = [p["gamma_r"], 0.75, p["gamma_e"], 1.25, p["gamma_f"], 2.0]
        sweep_cases = []
        for g in gammas:
            sweep_cases.append(run_case(p, g, spiral=False, n_steps=700, snapshots=None))
        analytic = np.array([c["lambda_analytic_m"] * 1e6 for c in sweep_cases])
        measured = np.array([c["lambda_measured_m"] * 1e6 for c in sweep_cases])
        fit = fit_series(analytic, measured)
        focal = turing_wavenumber(p["a"], p["b"], p["du"], p["dv"], p["gamma_e"])
        length = p["n_waves"] * focal["lambda_nd"]
        ab = simulate_schnakenberg(
            a=p["a"], b=p["b"], du=p["du"], dv=p["du"], gamma=p["gamma_e"],
            length=length, n=p["n"], dt=p["dt"], n_steps=700,
            seed=p["seed"], spiral=False, k_c=float("nan"), snapshots=None,
        )
        noise_focal = next(c for c in sweep_cases if abs(c["gamma"] - p["gamma_e"]) < 1e-9)
        extra_block = {
            "sweep": {
                "gamma": [c["gamma"] for c in sweep_cases],
                "analytic_um": analytic.tolist(),
                "measured_um": measured.tolist(),
            },
            "pearson_r": fit["pearson_r"],
            "rmse_um": fit["rmse"],
            "sweep_n": fit["n"],
            "turing_std": noise_focal["final_std"],
            "ablation_std": float(ab["u"].std()),
            "ablation_turing_flag": turing_wavenumber(p["a"], p["b"], p["du"], p["du"], p["gamma_e"])["turing"],
            "chronospecies": [_public_case(c) for c in sweep_cases],
            "smoothed_segmentation": segment_specimen_smoothed(specimen),
        }
        extra = extra_block
    write_comparison_png(specimen, hero, p, extra, PNG_PATH)
    metrics = build_metrics(ref, p, hero, seg, gif_info, extra, pass_id)
    # ocellus spacing must use the simulation dx, not 1. Recompute properly.
    dx = (p["n_waves"] * turing_wavenumber(p["a"], p["b"], p["du"], p["dv"], p["gamma_e"])["lambda_nd"]) / p["n"]
    spacing = ocellus_spacing_cv(hero["u"], dx)
    metrics["n_foci"] = spacing["n_foci"]
    metrics["ocellus_spacing_cv"] = spacing["spacing_cv"]
    metrics["ocellus_spacing_nd"] = spacing["spacing_nd"]
    text = json.dumps(metrics, indent=2)
    METRICS_LATEST.write_text(text + "\n", encoding="utf-8")
    (PASS2_JSON if pass_id >= 2 else PASS1_JSON).write_text(text + "\n", encoding="utf-8")
    return metrics


def self_test() -> None:
    ref = verify_dataset_integrity()
    p = params_from_ref(ref)
    focal = turing_wavenumber(p["a"], p["b"], p["du"], p["dv"], 1.0)
    assert focal["turing"], focal
    assert focal["trace"] < 0 and focal["det"] > 0 and focal["discriminant"] > 0
    lam = lambda_m_of_gamma(1.0, p["lam_e"], 1.0)
    assert abs(lam - 1.485e-06) < 1e-15
    # γ scaling identity against the explicit dispersion, not only the algebraic shortcut.
    for g in (0.5, 1.0, (1.485 / 1.180) ** 2, 2.0):
        spec = turing_wavenumber(p["a"], p["b"], p["du"], p["dv"], g)
        ell = length_unit_m(focal["lambda_nd"], p["lam_e"])
        from_disp = spec["lambda_nd"] * ell
        from_scale = lambda_m_of_gamma(g, p["lam_e"], 1.0)
        assert abs(from_disp - from_scale) / from_scale < 1e-9, (g, from_disp, from_scale)
    assert abs(lambda_m_of_gamma(0.5, p["lam_e"]) - p["lam_e"] * np.sqrt(2.0)) < 1e-15
    assert abs(p["gamma_f"] - (p["lam_e"] / p["lam_f"]) ** 2) < 1e-12
    # Predecessor longer, successor shorter, twelve waves in the detail window.
    assert p["lam_r"] > p["lam_e"] > p["lam_f"]
    assert abs(p["n_waves"] * p["lam_e"] - 12 * 1.485e-06) < 1e-15
    ab_spec = turing_wavenumber(p["a"], p["b"], p["du"], p["du"], 1.0)
    assert ab_spec["turing"] is False
    # Short patterned run versus equal-diffusion decay.
    patterned = run_case(p, 1.0, spiral=False, n_steps=500, n=80)
    assert patterned["final_std"] > 0.4, patterned["final_std"]
    focal_spec = turing_wavenumber(p["a"], p["b"], p["du"], p["dv"], 1.0)
    length = p["n_waves"] * focal_spec["lambda_nd"]
    ab = simulate_schnakenberg(
        a=p["a"], b=p["b"], du=p["du"], dv=p["du"], gamma=1.0,
        length=length, n=80, dt=p["dt"], n_steps=500,
        seed=p["seed"], spiral=False, k_c=float("nan"),
    )
    assert float(ab["u"].std()) < 0.05, float(ab["u"].std())
    analytic, measured = [], []
    for g in (0.5, 1.0, p["gamma_f"], 2.0):
        case = run_case(p, g, spiral=False, n_steps=500, n=80)
        analytic.append(case["lambda_analytic_m"])
        measured.append(case["lambda_measured_m"])
    fit = fit_series(np.asarray(analytic), np.asarray(measured))
    assert fit["pearson_r"] > 0.98, fit
    assert np.all(np.asarray(measured) != np.asarray(analytic))
    print(json.dumps({"self_test": "passed", "pearson_r": fit["pearson_r"], "rmse_m": fit["rmse"],
                      "pattern_std": patterned["final_std"], "ablation_std": float(ab["u"].std()),
                      "lambda_erecta_um": p["lam_r"] * 1e6, "lambda_ficusphila_um": p["lam_f"] * 1e6}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass", dest="pass_id", type=int, choices=(1, 2), default=1)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    metrics = execute(args.pass_id)
    summary = {k: metrics[k] for k in metrics if k not in ("sweep", "chronospecies")}
    print(json.dumps(summary, indent=2))
    gif = metrics["gif"]
    if gif["n_frames"] < 10 or min(gif["width"], gif["height"]) < 480:
        raise SystemExit(f"GIF gate failed: {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
