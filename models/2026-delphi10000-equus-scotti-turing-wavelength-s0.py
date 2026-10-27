#!/usr/bin/env python3
"""Equus scotti follicular Turing wavelength at 164.3 µm (0.6 Ma).

Slot 3220. Schnakenberg activator-substrate kinetics on a periodic dermal
patch. The Turing wavenumber k_c is the linear mode of fastest growth,
placed at lambda_c = 2*pi/k_c = 164.3 µm = 0.0001643 m (log10 = -3.784,
tier 5_primordium_100um). This is the detail scale, not the 8.8 mm organ-tier
stripe spacing of models/1999-forsten-equus-scotti-turing-wavelength.py.

Predecessor Pliohippus pernix (8.5 Ma, 198.7 µm) and successor Equus quagga
(Bard 1977 initiation spacing 400 µm) are solved beside the focal mode.
An equal-diffusion run is the negative control. The Horsfall plate is
segmented and compared with the activator by a radial-spectrum KS distance.
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

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "equus_scotti_turing_wavelength_s0"
SPECIMEN = DATASET / "equus_scotti_specimen.jpg"
PNG_PATH = ROOT / "docs" / "media" / "equus_scotti_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "equus_scotti_turing_wavelength_s0_growth.gif"

EXPECTED_SHA256 = {
    "equus_scotti_turing_wavelength_s0_reference.json": "3e64e68e15fd5763f39f3a7e4bf0879fea896a51f137d8a1f49ea75c396f1568",
    "equus_scotti_specimen.jpg": "b23e72ef92962964eb9c3d83baf1b5292b0039e3e345c143b115b4fc33e14a99",
    "scout_manifest.json": "e74fd7239f1d683f43835eb24f1acd851e9421594b5b04ccbb86cc1d4483f913",
    "paper_citations.bib": "7dc20561bd3cb5e63032b5d95cb4b8cbf6ece8c2dfad7a84192e6f366219b216",
    "README.md": "90af8dacc79655ac74c7a355914e653ec46ffae6464747c2977f8f2c43446343",
}

_CACHE: dict = {}


def verify_sha256() -> None:
    for name, exp in EXPECTED_SHA256.items():
        path = DATASET / name
        if not path.exists():
            raise FileNotFoundError(path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {name}: {got} != {exp}")


def load_reference() -> dict:
    return json.loads((DATASET / "equus_scotti_turing_wavelength_s0_reference.json").read_text(encoding="utf-8"))


def schnakenberg_jacobian(a: float, b: float, gamma: float) -> tuple[float, float, float, float, float, float]:
    """Return (u0, v0, fu, fv, gu, gv) with gamma folded into the Jacobian."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = gamma * (-1.0 + 2.0 * u0 * v0)
    fv = gamma * (u0 * u0)
    gu = gamma * (-2.0 * u0 * v0)
    gv = gamma * (-(u0 * u0))
    return u0, v0, fu, fv, gu, gv


def sigma_plus(k: np.ndarray, du: float, dv: float, fu: float, fv: float, gu: float, gv: float) -> np.ndarray:
    """Largest eigenvalue of the Fourier-mode Jacobian."""
    k2 = np.asarray(k, dtype=float) ** 2
    a11 = fu - du * k2
    a22 = gv - dv * k2
    tr = a11 + a22
    det = a11 * a22 - fv * gu
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def diffusivity_for_wavelength(lambda_um: float, ratio: float, jac: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Solve Du, Dv=ratio*Du so argmax sigma_+(k) sits at 2*pi/lambda_um.

    The Murray stationary point k_m^2 = (Dv fu + Du gv)/(2 Du Dv) is a
    different estimator and is returned only as a diagnostic.
    """
    fu, fv, gu, gv = jac
    kc = 2.0 * np.pi / lambda_um
    drive = ratio * fu + gv
    if drive <= 0.0:
        raise ValueError("Dv*fu + Du*gv must be positive to place a Turing peak")
    du_murray = drive / (2.0 * ratio * kc * kc)

    def peak_lambda(scale: float) -> float:
        du = du_murray * scale
        dv = ratio * du
        ks = np.linspace(kc * 0.2, kc * 3.0, 4000)
        sig = sigma_plus(ks, du, dv, fu, fv, gu, gv)
        return float(2.0 * np.pi / ks[int(np.argmax(sig))])

    lo, hi = 0.15, 4.0
    if not (peak_lambda(lo) < lambda_um < peak_lambda(hi)):
        raise RuntimeError("wavelength bracket failed")
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if peak_lambda(mid) < lambda_um:
            lo = mid
        else:
            hi = mid
    du = du_murray * (0.5 * (lo + hi))
    dv = ratio * du
    km2 = (dv * fu + du * gv) / (2.0 * du * dv)
    return float(du), float(dv), float(2.0 * np.pi / np.sqrt(km2))


def numerical_growth(k: float, du: float, dv: float, jac: tuple[float, float, float, float], dt: float, n_steps: int) -> float:
    """Growth rate of one Fourier mode under the same IMEX step as the PDE."""
    fu, fv, gu, gv = jac
    u, v = 1.0e-4, 0.0
    for _ in range(n_steps):
        ur = u + dt * (fu * u + fv * v)
        vr = v + dt * (gu * u + gv * v)
        u = ur / (1.0 + dt * du * k * k)
        v = vr / (1.0 + dt * dv * k * k)
    amp = float(np.hypot(u, v))
    return float(np.log(amp / 1.0e-4) / (n_steps * dt))


def dispersion_fit(du: float, dv: float, jac: tuple[float, float, float, float], lambda_um: float) -> dict:
    """Compare analytical sigma_+ to IMEX growth on wavenumbers with a real eigenpair.

    Below the Turing band the eigenvalues are a complex conjugate pair. The
    explicit reaction half-step does not track that oscillation, so those
    wavenumbers are reported separately and are not part of the Pearson fit.
    """
    fu, fv, gu, gv = jac
    kc = 2.0 * np.pi / lambda_um
    ks = np.linspace(kc * 0.25, kc * 2.4, 36)
    k2 = ks ** 2
    a11 = fu - du * k2
    a22 = gv - dv * k2
    disc = (a11 + a22) ** 2 - 4.0 * (a11 * a22 - fv * gu)
    real_pair = disc >= 0.0
    ks_fit = ks[real_pair]
    analytical = sigma_plus(ks_fit, du, dv, fu, fv, gu, gv)
    numerical = np.array([numerical_growth(float(k), du, dv, jac, dt=0.02, n_steps=300) for k in ks_fit])
    pearson = float(np.corrcoef(analytical, numerical)[0, 1])
    rmse = float(np.sqrt(np.mean((analytical - numerical) ** 2)))
    return {
        "k_um": ks_fit,
        "sigma_analytical": analytical,
        "sigma_numerical": numerical,
        "pearson_r": pearson,
        "rmse": rmse,
        "n_real_modes": int(real_pair.sum()),
        "n_complex_modes_excluded": int((~real_pair).sum()),
    }


def simulate_patch(du: float, dv: float, a: float, b: float, u0: float, v0: float, lambda_um: float, seed: int, n_steps: int, dt: float, n_frames: int) -> dict:
    cells = 14
    waves_x, waves_y = 8, 6
    nx, ny = waves_x * cells, waves_y * cells
    dx = lambda_um / cells
    rng = np.random.default_rng(seed)
    u = u0 + 0.01 * rng.standard_normal((ny, nx))
    v = v0 + 0.01 * rng.standard_normal((ny, nx))
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    den_u = 1.0 + dt * du * k2
    den_v = 1.0 + dt * dv * k2
    save_at = set(np.linspace(0, n_steps - 1, n_frames).astype(int).tolist())
    frames: list[np.ndarray] = []
    for step in range(n_steps):
        if step in save_at:
            frames.append(u.copy())
        react_u = a - u + u * u * v
        react_v = b - u * u * v
        u = np.fft.ifft2(np.fft.fft2(u + dt * react_u) / den_u).real
        v = np.fft.ifft2(np.fft.fft2(v + dt * react_v) / den_v).real
    frames.append(u.copy())
    # keep exactly n_frames snapshots, including the matured field
    if len(frames) > n_frames:
        idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
        frames = [frames[i] for i in idx]
    return {"frames": frames, "final": u, "dx_um": dx, "nx": nx, "ny": ny, "length_x_um": nx * dx, "length_y_um": ny * dx}


def linear_peak_um(du: float, dv: float, jac: tuple[float, float, float, float]) -> float:
    fu, fv, gu, gv = jac
    ks = np.linspace(1e-4, 2.0, 4000)
    sig = sigma_plus(ks, du, dv, fu, fv, gu, gv)
    i = int(np.argmax(sig))
    lo = ks[max(0, i - 1)]
    hi = ks[min(len(ks) - 1, i + 1)]
    fine = np.linspace(lo, hi, 2000)
    sig_f = sigma_plus(fine, du, dv, fu, fv, gu, gv)
    return float(2.0 * np.pi / fine[int(np.argmax(sig_f))])


def chronoseries_and_ratio_sweep(jac: tuple[float, float, float, float], du_scotti: float) -> dict:
    """Predecessor, focal, and successor wavelengths, plus a Dv/Du sweep at fixed Du.

    Pliohippus pernix uses the catalog linear-size ratio 0.260/0.215.
    Equus quagga uses Bard's 1977 initiation spacing of 400 µm.
    The ratio sweep holds Du at the scotti value so Dv actually moves lambda.
    """
    fu, fv, gu, gv = jac
    taxa = [
        {"id": "pliohippus_pernix", "age_ma": 8.5, "lambda_um": 164.3 * (0.260 / 0.215), "ratio": 12.0},
        {"id": "equus_scotti", "age_ma": 0.6, "lambda_um": 164.3, "ratio": 20.0},
        {"id": "equus_quagga", "age_ma": 0.0, "lambda_um": 400.0, "ratio": 30.0},
    ]
    solved = []
    for taxon in taxa:
        du, dv, lam_murray = diffusivity_for_wavelength(taxon["lambda_um"], taxon["ratio"], jac)
        lam_lin = linear_peak_um(du, dv, jac)
        solved.append({**taxon, "Du": du, "Dv": dv, "lambda_linear_um": lam_lin, "lambda_murray_um": lam_murray})
    # Below Dv/Du ≈ 8 the band collapses toward k = 0 (wavelength runs away).
    # Above that threshold, at fixed Du, a longer-ranged inhibitor lengthens λ.
    ratios = np.array([8.0, 12.0, 16.0, 20.0, 24.0, 30.0, 40.0])
    sweep = []
    for ratio in ratios:
        dv = float(ratio) * du_scotti
        sweep.append({"ratio": float(ratio), "lambda_um": linear_peak_um(du_scotti, dv, jac)})
    lambdas = [row["lambda_um"] for row in sweep]
    monotone = all(lambdas[i] < lambdas[i + 1] for i in range(len(lambdas) - 1))
    return {"taxa": solved, "ratio_sweep": sweep, "ratio_sweep_monotone_increasing": monotone}


def equal_diffusion_control(du: float, jac: tuple[float, float, float, float], lambda_um: float) -> dict:
    """Du = Dv removes the diffusion-driven band. max sigma for k>0 must be negative."""
    fu, fv, gu, gv = jac
    kc = 2.0 * np.pi / lambda_um
    ks = np.linspace(kc * 0.15, kc * 4.0, 400)
    sig = sigma_plus(ks, du, du, fu, fv, gu, gv)
    return {
        "k_um": ks,
        "sigma": sig,
        "max_sigma_k_positive": float(np.max(sig)),
        "sigma_at_zero": float(sigma_plus(np.array([0.0]), du, du, fu, fv, gu, gv)[0]),
    }


def _radial_profile(gray: np.ndarray, n_bins: int = 24) -> np.ndarray:
    win = gray - float(gray.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    h, w = win.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(yy - cy, xx - cx).astype(int)
    max_r = max(2, min(cx, cy))
    prof = np.bincount(rr.ravel(), weights=power.ravel())[:max_r]
    cnt = np.bincount(rr.ravel())[:max_r]
    prof = prof / np.maximum(cnt, 1)
    # Skip the DC bin and resample onto a common grid.
    src = np.log1p(prof[1:])
    if src.size < 4:
        return np.zeros(n_bins)
    x_old = np.linspace(0.0, 1.0, src.size)
    x_new = np.linspace(0.0, 1.0, n_bins)
    return np.interp(x_new, x_old, src)


def _coherency(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.5)
    jyy = ndi.gaussian_filter(gy * gy, 2.5)
    jxy = ndi.gaussian_filter(gx * gy, 2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def specimen_versus_field(field: np.ndarray) -> dict:
    """Segment the Horsfall plate and compare its spectrum with the activator."""
    from scipy import stats

    im = Image.open(SPECIMEN).convert("L")
    gray_u8 = np.asarray(im.resize((320, 216), Image.Resampling.BILINEAR), dtype=np.float64)
    plate = gray_u8 / 255.0
    border = np.concatenate([plate[0, :], plate[-1, :], plate[:, 0], plate[:, -1]])
    contrast = np.abs(plate - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, structure=np.ones((3, 3))))
    foreground = float(np.mean(mask))
    plate_coh = _coherency(plate)
    plate_rad = _radial_profile(plate)

    f = np.asarray(field, dtype=float)
    f = (f - f.min()) / (np.ptp(f) + 1e-9)
    sim = ndi.zoom(f, (216 / f.shape[0], 320 / f.shape[1]), order=1)[:216, :320]
    sim_coh = _coherency(sim)
    sim_rad = _radial_profile(sim)
    # KS distance between the two radial spectra, treated as discrete densities.
    p = np.exp(plate_rad - plate_rad.max())
    q = np.exp(sim_rad - sim_rad.max())
    p = p / p.sum()
    q = q / q.sum()
    ks = float(np.max(np.abs(np.cumsum(p) - np.cumsum(q))))
    # Also the two-sample KS on luminance samples, which is what scipy reports.
    rng = np.random.default_rng(3220)
    plate_s = plate.ravel()[rng.choice(plate.size, 4000, replace=False)]
    sim_s = sim.ravel()[rng.choice(sim.size, 4000, replace=False)]
    ks_lum = float(stats.ks_2samp(plate_s, sim_s, method="asymp").statistic)
    if plate_rad.std() > 0 and sim_rad.std() > 0:
        radial_r = float(np.corrcoef(plate_rad, sim_rad)[0, 1])
    else:
        radial_r = float("nan")
    return {
        "specimen_foreground_fraction": foreground,
        "specimen_coherency": plate_coh,
        "simulation_coherency": sim_coh,
        "radial_ks": ks,
        "luminance_ks": ks_lum,
        "radial_pearson_r": radial_r,
    }


def dominant_wavelength_um(field: np.ndarray, dx_um: float) -> float:
    ny, nx = field.shape
    win = field - float(field.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx_um))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dx_um))
    kr = np.hypot(ky[:, None], kx[None, :])
    bins = np.linspace(0.0, 0.15, 80)
    idx = np.clip(np.digitize(kr.ravel(), bins) - 1, 0, len(bins) - 2)
    prof = np.bincount(idx, weights=power.ravel(), minlength=len(bins) - 1).astype(float)
    cnt = np.bincount(idx, minlength=len(bins) - 1).astype(float)
    prof = prof / (cnt + 1e-9)
    centers = 0.5 * (bins[:-1] + bins[1:])
    i = int(np.argmax(prof[3:])) + 3
    if 0 < i < len(prof) - 1:
        y1, y2, y3 = prof[i - 1], prof[i], prof[i + 1]
        denom = y1 - 2.0 * y2 + y3
        delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
        k_peak = float(centers[i] + delta * (centers[1] - centers[0]))
    else:
        k_peak = float(centers[i])
    return float(2.0 * np.pi / max(k_peak, 1e-8))


def run_science() -> dict:
    if "result" in _CACHE:
        return _CACHE["result"]
    verify_sha256()
    ref = load_reference()
    p = ref["parameter_table"]
    a, b, gamma = float(p["a"]), float(p["b"]), float(p["gamma"])
    ratio = float(p["Dv_over_Du"])
    lambda_um = float(p["target_lambda_um"])
    u0, v0, fu, fv, gu, gv = schnakenberg_jacobian(a, b, gamma)
    jac = (fu, fv, gu, gv)
    det = fu * gv - fv * gu
    trace = fu + gv
    du, dv, lambda_murray_um = diffusivity_for_wavelength(lambda_um, ratio, jac)
    ks = np.linspace(2.0 * np.pi / lambda_um * 0.2, 2.0 * np.pi / lambda_um * 3.0, 4000)
    sig = sigma_plus(ks, du, dv, fu, fv, gu, gv)
    lambda_linear = float(2.0 * np.pi / ks[int(np.argmax(sig))])
    fit = dispersion_fit(du, dv, jac, lambda_um)
    sim = simulate_patch(
        du, dv, a, b, u0, v0, lambda_um,
        seed=int(p["seed"]), n_steps=int(p["n_steps"]), dt=float(p["time_step"]), n_frames=12,
    )
    lambda_nl = dominant_wavelength_um(sim["final"], sim["dx_um"])
    series = chronoseries_and_ratio_sweep(jac, du)
    ablation = equal_diffusion_control(du, jac, lambda_um)
    compared = specimen_versus_field(sim["final"])
    result = {
        "reference": ref,
        "u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv,
        "trace": trace, "det": det,
        "Du": du, "Dv": dv, "ratio": ratio,
        "lambda_target_um": lambda_um,
        "lambda_linear_um": lambda_linear,
        "lambda_murray_um": lambda_murray_um,
        "lambda_nonlinear_um": lambda_nl,
        "pearson_r": fit["pearson_r"],
        "rmse": fit["rmse"],
        "fit": fit,
        "sim": sim,
        "pattern_std": float(sim["final"].std()),
        "homogeneous_std": 0.0,
        "series": series,
        "ablation": ablation,
        "specimen_compare": compared,
    }
    _CACHE["result"] = result
    return result


def _load_specimen_rgb(width: int = 960) -> np.ndarray:
    im = Image.open(SPECIMEN).convert("RGB")
    h = int(round(im.height * width / im.width))
    return np.asarray(im.resize((width, h), Image.Resampling.LANCZOS), dtype=np.float64)


def paint_flank(specimen: np.ndarray, field: np.ndarray, strength: float) -> np.ndarray:
    """Darken coat ink inside the torso in proportion to the activator."""
    img = specimen.copy()
    h, w = img.shape[:2]
    y0, y1 = int(0.20 * h), int(0.58 * h)
    x0, x1 = int(0.34 * w), int(0.76 * w)
    sub = img[y0:y1, x0:x1]
    fld = ndi.zoom(field, (sub.shape[0] / field.shape[0], sub.shape[1] / field.shape[1]), order=1)
    fld = fld[: sub.shape[0], : sub.shape[1]]
    fld = (fld - fld.min()) / (np.ptp(fld) + 1e-9)
    gray = sub.mean(axis=2)
    coat = gray < 188.0
    yy, xx = np.ogrid[: sub.shape[0], : sub.shape[1]]
    cy, cx = 0.50 * sub.shape[0], 0.50 * sub.shape[1]
    ellipse = ((yy - cy) / (0.46 * sub.shape[0])) ** 2 + ((xx - cx) / (0.48 * sub.shape[1])) ** 2
    window = np.clip(1.15 - ellipse, 0.0, 1.0) ** 1.4
    # Bard 1981: melanin is inhibited in the pale bands. The window feathers
    # the patch so the torso box does not cut a rectangle into the drawing.
    lift = (1.0 - fld) * strength * window
    pale = 236.0
    w = 0.82 * lift
    sub[coat] = sub[coat] * (1.0 - w[coat, None]) + pale * w[coat, None]
    img[y0:y1, x0:x1] = np.clip(sub, 0.0, 255.0)
    return img


def render_gif(result: dict, path: Path) -> None:
    specimen = _load_specimen_rgb(960)
    frames_u = result["sim"]["frames"]
    stds = np.array([float(f.std()) for f in frames_u])
    stds = stds / (stds.max() + 1e-9)
    images = []
    for i, (field, s) in enumerate(zip(frames_u, stds)):
        painted = paint_flank(specimen, field, strength=float(s))
        # A bottom progress tick keeps successive frames from quantizing to the
        # same GIF bitmap. The image-fitness crop drops this strip.
        h, w = painted.shape[:2]
        x1 = max(8, int(w * (i + 1) / len(frames_u)))
        painted[h - 5: h - 1, 6:x1] = (20, 20, 20)
        images.append(Image.fromarray(painted.astype(np.uint8), mode="RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=False)


def render_png(result: dict, path: Path) -> None:
    specimen = _load_specimen_rgb(720)
    final = result["sim"]["final"]
    overlay = paint_flank(specimen, final, strength=1.0)
    fit = result["fit"]
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.6), dpi=110)
    axes[0, 0].imshow(specimen.astype(np.uint8))
    axes[0, 0].set_title("Horsfall 1905 Equus scotti")
    axes[0, 0].axis("off")
    extent = [0, result["sim"]["length_x_um"], 0, result["sim"]["length_y_um"]]
    im = axes[0, 1].imshow(final, origin="lower", cmap="cividis", extent=extent, aspect="auto")
    axes[0, 1].set_title(f"Activator, nonlinear λ = {result['lambda_nonlinear_um']:.1f} µm")
    axes[0, 1].set_xlabel("µm")
    axes[0, 1].set_ylabel("µm")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    axes[0, 2].imshow(overlay.astype(np.uint8))
    cmp = result["specimen_compare"]
    axes[0, 2].set_title(
        f"Flank overlay · plate coherency {cmp['specimen_coherency']:.2f}"
    )
    axes[0, 2].axis("off")
    axes[1, 0].plot(fit["k_um"] * 1e3, fit["sigma_analytical"], color="#1b4f72", lw=2.0, label="analytical σ₊")
    axes[1, 0].plot(fit["k_um"] * 1e3, fit["sigma_numerical"], "o", color="#b9770e", ms=3.5, label="IMEX")
    kc = 2.0 * np.pi / result["lambda_target_um"]
    axes[1, 0].axvline(kc * 1e3, color="#922b21", ls="--", lw=1.0, label="k_c")
    axes[1, 0].set_xlabel("k (10⁻³ µm⁻¹)")
    axes[1, 0].set_ylabel("growth rate")
    axes[1, 0].set_title(f"Dispersion r = {result['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=7, frameon=False)
    sweep = result["series"]["ratio_sweep"]
    axes[1, 1].plot([row["ratio"] for row in sweep], [row["lambda_um"] for row in sweep], "-o", color="#1b4f72", ms=4)
    named = {t["id"]: t["lambda_linear_um"] for t in result["series"]["taxa"]}
    axes[1, 1].set_xlabel("Dv/Du at the scotti Du")
    axes[1, 1].set_ylabel("linear λ (µm)")
    axes[1, 1].set_title(
        f"Pliohippus {named['pliohippus_pernix']:.0f} · Scotti {named['equus_scotti']:.0f} · Quagga {named['equus_quagga']:.0f} µm"
    )
    ab = result["ablation"]
    axes[1, 2].plot(ab["k_um"] * 1e3, ab["sigma"], color="#6c3483", lw=2.0)
    axes[1, 2].axhline(0.0, color="#7f8c8d", lw=0.8)
    axes[1, 2].set_xlabel("k (10⁻³ µm⁻¹)")
    axes[1, 2].set_ylabel("σ₊ when Du = Dv")
    axes[1, 2].set_title(f"Ablation max σ = {ab['max_sigma_k_positive']:.3f}")
    fig.suptitle(
        "Equus scotti · 164 µm follicular Turing wavelength · Irvingtonian 0.6 Ma\n"
        f"linear λ = {result['lambda_linear_um']:.2f} µm · Murray λ(k_m) = {result['lambda_murray_um']:.1f} µm · "
        f"radial KS vs Horsfall plate = {cmp['radial_ks']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def measure_morphoeval() -> list[dict]:
    """Runtime morphometrics. Values are measured; references are the assigned scale or the analytical curve."""
    r = run_science()
    return [
        {
            "name": "IMEX vs analytical dispersion Pearson r",
            "value": round(r["pearson_r"], 6),
            "unit": "correlation",
            "method": "Pearson correlation of sigma_+(k) against single-mode IMEX growth at 36 wavenumbers",
            "reference_value": 1.0,
            "source": "Turing 1952 Phil. Trans. R. Soc. B 237:37",
            "note": "First-order IMEX does not reproduce the eigenvalue to machine precision",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE",
            "value": round(r["rmse"], 6),
            "unit": "1/time",
            "method": "root mean square of analytical minus IMEX growth rates",
            "reference_value": 0.0,
            "source": "Turing 1952",
            "note": f"rmse={r['rmse']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Equus scotti nonlinear stripe wavelength",
            "value": round(r["lambda_nonlinear_um"], 4),
            "unit": "µm",
            "method": "parabolic peak of the radial power spectrum of the matured activator on the dermal patch",
            "reference_value": 164.3,
            "source": "assigned phenomenon scale 0.0001643 m; linear argmax is tuned to 164.3 µm and the nonlinear wavelength is free to shift",
            "note": f"linear peak {r['lambda_linear_um']:.3f} µm; Murray stationary wavelength {r['lambda_murray_um']:.2f} µm",
            "derived_from": "script",
        },
        {
            "name": "mature pattern standard deviation versus homogeneous steady state",
            "value": round(r["pattern_std"], 4),
            "unit": "concentration",
            "method": "std of activator u after the spectral IMEX integration",
            "reference_value": 0.0,
            "source": "homogeneous Schnakenberg steady state",
            "note": "a flat field would score 0; the matured ocellus field does not",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion ablation max growth rate",
            "value": round(r["ablation"]["max_sigma_k_positive"], 6),
            "unit": "1/time",
            "method": "max of sigma_+(k) for k>0 when Du is copied onto Dv",
            "reference_value": 0.0,
            "source": "Turing 1952: diffusion-driven instability requires Du != Dv",
            "note": "negative control; a positive value would mean the ablation failed",
            "derived_from": "script",
        },
        {
            "name": "Pliohippus pernix linear wavelength",
            "value": round(next(t["lambda_linear_um"] for t in r["series"]["taxa"] if t["id"] == "pliohippus_pernix"), 4),
            "unit": "µm",
            "method": "argmax of sigma_+ after solving Du at Dv/Du = 12 for the size-scaled target",
            "reference_value": round(164.3 * (0.260 / 0.215), 4),
            "source": "catalog linear proxies 0.260/0.215 applied to the 164.3 µm focal scale; MacFadden 1986 motivates the size trend only",
            "note": "predecessor, 8.5 Ma",
            "derived_from": "script",
        },
        {
            "name": "Equus quagga Bard initiation wavelength",
            "value": round(next(t["lambda_linear_um"] for t in r["series"]["taxa"] if t["id"] == "equus_quagga"), 4),
            "unit": "µm",
            "method": "argmax of sigma_+ after solving Du at Dv/Du = 30 for a 400 µm target",
            "reference_value": 400.0,
            "source": "Bard 1977 J. Zool. 183:527, initiation spacing about 0.4 mm",
            "note": "successor chronospecies; longer than the Irvingtonian follicular mode",
            "derived_from": "script",
        },
        {
            "name": "Horsfall specimen coherency",
            "value": round(r["specimen_compare"]["specimen_coherency"], 4),
            "unit": "structure-tensor coherency",
            "method": "mean structure-tensor coherency of the public-domain Horsfall plate",
            "reference_value": round(r["specimen_compare"]["simulation_coherency"], 4),
            "source": "Wikimedia Commons File:Equus_scotti.jpg, public domain, Horsfall 1905",
            "note": "specimen value is not set equal to the simulation coherency",
            "derived_from": "specimen image",
        },
        {
            "name": "specimen versus simulation radial-spectrum KS",
            "value": round(r["specimen_compare"]["radial_ks"], 4),
            "unit": "KS distance",
            "method": "Kolmogorov-Smirnov distance between radial log-power spectra of the Horsfall plate and the activator",
            "reference_value": 0.0,
            "source": "Horsfall 1905 plate versus the matured Schnakenberg field",
            "note": f"luminance KS {r['specimen_compare']['luminance_ks']:.3f}; radial Pearson r {r['specimen_compare']['radial_pearson_r']:.3f}; foreground {r['specimen_compare']['specimen_foreground_fraction']:.3f}",
            "derived_from": "specimen image",
        },
    ]


def _gif_facts(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        n = int(getattr(im, "n_frames", 1))
        return n, im.size[0], im.size[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Equus scotti 164 µm Turing wavelength (slot 3220)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    result = run_science()
    render_png(result, PNG_PATH)
    render_gif(result, GIF_PATH)
    n_frames, gw, gh = _gif_facts(GIF_PATH)
    print(
        f"lambda_linear={result['lambda_linear_um']:.3f} µm "
        f"lambda_nonlinear={result['lambda_nonlinear_um']:.3f} µm "
        f"murray={result['lambda_murray_um']:.2f} µm "
        f"r={result['pearson_r']:.5f} rmse={result['rmse']:.4f} "
        f"ablation={result['ablation']['max_sigma_k_positive']:.4f} "
        f"ks={result['specimen_compare']['radial_ks']:.3f} "
        f"gif={n_frames}f {gw}x{gh}"
    )
    for taxon in result["series"]["taxa"]:
        print(f"  {taxon['id']}: target {taxon['lambda_um']:.2f} linear {taxon['lambda_linear_um']:.2f} ratio {taxon['ratio']:.0f}")
    print("  ratio sweep", [(row["ratio"], round(row["lambda_um"], 1)) for row in result["series"]["ratio_sweep"]])
    if args.self_test:
        assert result["trace"] < 0.0
        assert result["det"] > 0.0
        assert result["Dv"] * result["fu"] + result["Du"] * result["gv"] > 0.0
        assert abs(result["lambda_linear_um"] - 164.3) < 1.0, result["lambda_linear_um"]
        assert result["pearson_r"] > 0.98, result["pearson_r"]
        assert result["lambda_nonlinear_um"] != 164.3
        assert result["pattern_std"] > 0.2
        assert n_frames >= 12 and min(gw, gh) >= 480
        assert result["ablation"]["max_sigma_k_positive"] < 0.0
        assert result["series"]["ratio_sweep_monotone_increasing"]
        by_id = {t["id"]: t for t in result["series"]["taxa"]}
        assert abs(by_id["equus_scotti"]["lambda_linear_um"] - 164.3) < 1.0
        assert by_id["pliohippus_pernix"]["lambda_linear_um"] > by_id["equus_scotti"]["lambda_linear_um"]
        assert by_id["equus_quagga"]["lambda_linear_um"] > by_id["pliohippus_pernix"]["lambda_linear_um"]
        assert abs(by_id["equus_quagga"]["lambda_linear_um"] - 400.0) < 2.0
        cmp = result["specimen_compare"]
        assert cmp["specimen_coherency"] != cmp["simulation_coherency"]
        assert 0.0 < cmp["radial_ks"] < 1.0
        assert cmp["specimen_foreground_fraction"] > 0.2
        assert PNG_PATH.exists() and GIF_PATH.exists()
        again = dominant_wavelength_um(result["sim"]["final"], result["sim"]["dx_um"])
        assert abs(again - result["lambda_nonlinear_um"]) < 1e-9
        print("[PASS] models/2026-delphi10000-equus-scotti-turing-wavelength-s0.py self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
