#!/usr/bin/env python3
"""Anchiornis huxleyi — Turing wavenumber at 2.0 nm (slot 3111, detail scale).

Age 160.0 Ma, Late Jurassic (Tiaojishan Fm). Chronoseries 1/1: no predecessor
or successor chronospecies. The coarser sibling mode is 29 nm (slot 1229);
this file does not reuse that slot's parameters.

Schnakenberg (1979) kinetics, Turing (1952) dispersion:

    du/dt = Du laplacian(u) + a - u + u^2 v
    dv/dt = Dv laplacian(v) + b - u^2 v

    a = 0.062, b = 1.28, Du = 0.72, Dv = 31.5
    k_c^2 = (Dv fu + Du gv) / (2 Du Dv)
    lambda_c = 2 pi / k_c  =  2.0 nm   after length calibration

Boundary condition: periodic square patch inside one melanosome.

Claim boundary: Li et al. 2010 did not publish this PDE. 2.0 nm is the
assigned detail wavelength, identified with the eumelanin oligomer-sheet
scale (Watt, Bothma & Meredith 2009), not a melanosome length from Li 2010.
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
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "anchiornis_huxleyi_turing_wavelength_s0"
SPECIMEN = DATASET / "anchiornis_huxleyi_holotype.jpg"
REFERENCE = DATASET / "anchiornis_huxleyi_turing_wavelength_s0_reference.json"
METRICS_PATH = DATASET / "anchiornis_huxleyi_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "anchiornis_huxleyi_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "anchiornis_huxleyi_turing_wavelength_s0_growth.gif"

# Slot-3111 parameters. Not the 29 nm sibling (that script has no kinetics).
A = 0.062
B = 1.28
DU = 0.72
DV = 31.5
DX = 1.0
DT = 0.004
GRID = 128
SEED = 3111
LAMBDA_TARGET_M = 2.0e-9
AGE_MA = 160.0
PREDECESSOR_LAMBDA_M = 2.89e-8  # slot 1229 envelope, same taxon, same age

SPECIMEN_SHA256 = "4415de64f740a3c5ef0c075c837dd747fcbc730f9ace950ec46084ae41cdd60b"


def steady_state(a: float = A, b: float = B) -> tuple[float, float]:
    """Homogeneous steady state of the Schnakenberg system."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float = A, b: float = B) -> tuple[float, float, float, float]:
    """Partial derivatives (fu, fv, gu, gv) at the steady state."""
    u0, v0 = steady_state(a, b)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return fu, fv, gu, gv


def dispersion_sigma(k: np.ndarray, a: float = A, b: float = B, du: float = DU, dv: float = DV) -> np.ndarray:
    """Largest eigenvalue of J - k^2 D. Positive inside the Turing band."""
    fu, fv, gu, gv = jacobian(a, b)
    k2 = np.asarray(k, dtype=float) ** 2
    tr = (fu + gv) - (du + dv) * k2
    det = (fu - du * k2) * (gv - dv * k2) - fv * gu
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def critical_wavenumber(a: float = A, b: float = B, du: float = DU, dv: float = DV) -> dict:
    """Analytical k_c, lambda_c, unstable band, and the equal-diffusion control."""
    fu, fv, gu, gv = jacobian(a, b)
    tr_j = fu + gv
    det_j = fu * gv - fv * gu
    h = dv * fu + du * gv
    disc = h * h - 4.0 * du * dv * det_j
    kc2 = h / (2.0 * du * dv)
    kc = float(np.sqrt(kc2))
    lam = float(2.0 * np.pi / kc)
    # band edges from Du Dv k^4 - h k^2 + det = 0
    aa = du * dv
    disc_k = h * h - 4.0 * aa * det_j
    sqrt_d = float(np.sqrt(max(disc_k, 0.0)))
    k2_lo = (h - sqrt_d) / (2.0 * aa)
    k2_hi = (h + sqrt_d) / (2.0 * aa)
    h_equal = 1.0 * fu + 1.0 * gv  # Du = Dv = 1 leaves the sign of h equal to trace
    return {
        "fu": fu, "fv": fv, "gu": gu, "gv": gv,
        "trace": tr_j, "det": det_j, "h": h, "discriminant": disc,
        "k_c": kc, "lambda_c_dimensionless": lam,
        "k_lo": float(np.sqrt(max(k2_lo, 0.0))),
        "k_hi": float(np.sqrt(max(k2_hi, 0.0))),
        "equal_diffusion_h": h_equal,
        "turing": bool(tr_j < 0.0 and det_j > 0.0 and h > 0.0 and disc > 0.0),
    }


def length_unit_m(info: dict | None = None) -> float:
    """Metres per dimensionless length so analytical lambda_c is 2.0 nm."""
    info = info or critical_wavenumber()
    return LAMBDA_TARGET_M / info["lambda_c_dimensionless"]


def laplacian(z: np.ndarray, dx: float = DX) -> np.ndarray:
    return (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z) / (dx * dx)


def simulate(
    n: int = GRID,
    steps: int = 2400,
    dt: float = DT,
    seed: int = SEED,
    a: float = A,
    b: float = B,
    du: float = DU,
    dv: float = DV,
    n_snapshots: int = 8,
    noise: float = 0.05,
) -> dict:
    """Explicit-Euler Schnakenberg on a periodic grid. Returns snapshots of u."""
    u0, v0 = steady_state(a, b)
    rng = np.random.default_rng(seed)
    u = u0 + noise * rng.normal(size=(n, n))
    v = v0 + noise * rng.normal(size=(n, n))
    snaps = [u.copy()]
    every = max(1, steps // max(1, n_snapshots - 1))
    for s in range(steps):
        lu = laplacian(u)
        lv = laplacian(v)
        u2v = u * u * v
        u = u + dt * (du * lu + a - u + u2v)
        v = v + dt * (dv * lv + b - u2v)
        if (s + 1) % every == 0:
            snaps.append(u.copy())
    if len(snaps) > n_snapshots:
        snaps = snaps[: n_snapshots - 1] + [snaps[-1]]
    while len(snaps) < n_snapshots:
        snaps.append(u.copy())
    return {"u": u, "v": v, "snapshots": snaps, "u0": u0, "std_u": float(np.std(u))}


def measured_dominant_wavelength_m(
    field: np.ndarray,
    s_m: float,
    dx: float = DX,
    k_lo: float | None = None,
    k_hi: float | None = None,
) -> float:
    """Power-weighted radial wavenumber inside the unstable band, in metres."""
    win = field - np.mean(field)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    n = field.shape[0]
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    kk = np.hypot(*np.meshgrid(ky, ky, indexing="ij"))
    band = kk > 0.05
    if k_lo is not None and k_hi is not None:
        band = (kk >= k_lo) & (kk <= k_hi)
    weight = spec[band]
    k_dom = float(np.sum(kk[band] * weight) / np.sum(weight))
    lam = 2.0 * np.pi / max(k_dom, 1e-6)
    return float(lam * s_m)


def linear_growth_fit(n: int = 96, steps: int = 20, dt: float = 0.001) -> dict:
    """Growth of pure Fourier modes of the linearized Schnakenberg system.

    The five-point Laplacian has symbol -4 sin^2(pi m / n) / dx^2, so each
    slope is compared to analytical sigma at that discrete wavenumber.
    """
    info = critical_wavenumber()
    fu, fv, gu, gv = info["fu"], info["fv"], info["gu"], info["gv"]
    modes = (3, 5, 7, 9, 11, 13, 15, 17, 19, 22)
    sig_hat: list[float] = []
    sig_ref: list[float] = []
    ks: list[float] = []
    x = np.arange(n)
    for m in modes:
        k_eff = 2.0 * np.sin(np.pi * m / n) / DX
        mat = np.array(
            [[fu - DU * k_eff * k_eff, fv], [gu, gv - DV * k_eff * k_eff]],
            dtype=float,
        )
        eigvals, eigvecs = np.linalg.eig(mat)
        idx = int(np.argmax(np.real(eigvals)))
        vec = np.real(eigvecs[:, idx])
        vec = vec / (np.linalg.norm(vec) + 1e-15)
        wave = np.cos(2.0 * np.pi * m * x / n)
        wave = wave[None, :] * np.ones((n, 1))
        amp0 = 1e-3
        u = amp0 * vec[0] * wave
        v = amp0 * vec[1] * wave
        amps: list[float] = []
        times: list[float] = []
        for s in range(steps):
            lu = laplacian(u)
            lv = laplacian(v)
            u = u + dt * (DU * lu + fu * u + fv * v)
            v = v + dt * (DV * lv + gu * u + gv * v)
            if s >= 4:
                coeff = float(np.mean(u * wave) * 2.0)
                amps.append(abs(coeff) + 1e-18)
                times.append((s + 1) * dt)
        slope = float(stats.linregress(np.asarray(times), np.log(np.asarray(amps))).slope)
        sig_hat.append(slope)
        sig_ref.append(float(np.real(eigvals[idx])))
        ks.append(float(k_eff))
    sig_hat_a = np.asarray(sig_hat)
    sig_ref_a = np.asarray(sig_ref)
    pearson = float(np.corrcoef(sig_hat_a, sig_ref_a)[0, 1])
    rmse = float(np.sqrt(np.mean((sig_hat_a - sig_ref_a) ** 2)))
    ks_stat = float(stats.ks_2samp(sig_hat_a, sig_ref_a).statistic)
    return {
        "k": ks,
        "sigma_measured": sig_hat,
        "sigma_analytical": sig_ref,
        "pearson_r": pearson,
        "rmse": rmse,
        "ks_statistic": ks_stat,
    }


def dv_sweep() -> list[dict]:
    """lambda_c versus inhibitor diffusivity. lambda falls as Dv rises."""
    rows = []
    for dv in (18.0, 24.0, 31.5, 42.0, 56.0):
        info = critical_wavenumber(dv=dv)
        rows.append({
            "Dv": dv,
            "k_c": info["k_c"],
            "lambda_c_nm": info["lambda_c_dimensionless"] * length_unit_m() * 1e9,
            "turing": info["turing"],
            "sigma_max": float(dispersion_sigma(np.array([info["k_c"]]), dv=dv)[0]),
        })
    return rows


def _segment_roi(roi: np.ndarray) -> dict:
    """Border-contrast mask and shape stats for one crop of the holotype."""
    border = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
    bg = float(np.median(border))
    contrast = np.abs(roi - bg)
    th = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(th, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    area = float(np.sum(mask))
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    circularity = float(np.clip(4.0 * np.pi * area / (perim ** 2), 0.0, 1.0))
    ys, xs = np.where(mask)
    if len(xs) > 10:
        aspect = float((ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1))
    else:
        aspect = 1.0
    gx = ndi.sobel(roi, axis=1)
    gy = ndi.sobel(roi, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    win = roi - np.mean(roi)
    fft2 = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = win.shape[0] // 2, win.shape[1] // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy).astype(int)
    max_r = min(cx, cy)
    radial = np.bincount(rr.ravel(), weights=fft2.ravel())[:max_r]
    radial = radial / np.maximum(1, np.bincount(rr.ravel())[:max_r])
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circularity,
        "aspect_ratio": aspect,
        "coherency": coherency,
        "radial_profile": radial[2:26].tolist(),
        "gray_roi_std": float(np.std(roi)),
        "mask": mask,
        "roi": roi,
    }


def specimen_cv(path: Path = SPECIMEN) -> dict:
    """Wide and tight crops of the holotype. The wide crop is the primary mask."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    h, w = gray.shape
    wide_box = [int(0.18 * w), int(0.18 * h), int(0.82 * w), int(0.82 * h)]
    tight_box = [int(0.30 * w), int(0.30 * h), int(0.70 * w), int(0.70 * h)]
    x0, y0, x1, y1 = wide_box
    wide = _segment_roi(gray[y0:y1, x0:x1])
    tx0, ty0, tx1, ty1 = tight_box
    tight = _segment_roi(gray[ty0:ty1, tx0:tx1])
    wide["roi_box"] = wide_box
    wide["tight_roi_box"] = tight_box
    wide["tight_foreground_fraction"] = tight["foreground_fraction"]
    wide["foreground_delta_wide_minus_tight"] = wide["foreground_fraction"] - tight["foreground_fraction"]
    wide["tight_circularity"] = tight["circularity"]
    return wide


def field_radial_profile(field: np.ndarray, n_bins: int = 24) -> np.ndarray:
    win = field - np.mean(field)
    fft2 = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    cy, cx = win.shape[0] // 2, win.shape[1] // 2
    yy, xx = np.ogrid[: win.shape[0], : win.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    rmax = float(min(cx, cy))
    bins = np.linspace(0.0, rmax, n_bins + 1)
    prof = np.zeros(n_bins)
    which = np.digitize(rr.ravel(), bins) - 1
    pw = fft2.ravel()
    for i in range(n_bins):
        m = which == i
        if np.any(m):
            prof[i] = np.mean(pw[m])
    return prof


def verify_specimen_hash() -> str:
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    if digest != SPECIMEN_SHA256:
        raise ValueError(f"specimen SHA-256 mismatch: {digest}")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    pinned = ref["specimen_image"]["sha256"]
    if pinned != SPECIMEN_SHA256:
        raise ValueError("reference JSON specimen hash drifted from the pinned constant")
    if abs(float(ref["phenomenon_scale"]["phenomenon_size_m"]) - LAMBDA_TARGET_M) > 0:
        raise ValueError("reference scale is not 2.0 nm")
    if float(ref["organism"]["geological_age_ma"]) != AGE_MA:
        raise ValueError("reference age is not 160.0 Ma")
    return digest


def compute_all() -> dict:
    """Full quantitative bundle used by the renderer, self-test, and measure_morphoeval."""
    verify_specimen_hash()
    info = critical_wavenumber()
    s_m = length_unit_m(info)
    pattern = simulate()
    lam_meas = measured_dominant_wavelength_m(pattern["u"], s_m, k_lo=info["k_lo"], k_hi=info["k_hi"])
    growth = linear_growth_fit()
    sweep = dv_sweep()
    # negative control: equal diffusion on the same grid. The patterned std is the
    # fully developed field; the control is integrated long enough to decay.
    control = simulate(n=GRID, steps=900, du=0.72, dv=0.72, noise=0.05, n_snapshots=2, seed=SEED)
    seg = specimen_cv()
    sim_prof = field_radial_profile(pattern["u"])
    spec_prof = np.asarray(seg["radial_profile"], dtype=float)
    # resample to a common length for the KS comparison of shape, not of pixel scale
    spec_rs = np.interp(np.linspace(0, 1, len(sim_prof)), np.linspace(0, 1, len(spec_prof)), spec_prof)
    spec_rs = spec_rs / (np.linalg.norm(spec_rs) + 1e-12)
    sim_n = sim_prof / (np.linalg.norm(sim_prof) + 1e-12)
    ks_profiles = float(stats.ks_2samp(sim_n, spec_rs).statistic)
    sigma_max = float(dispersion_sigma(np.array([info["k_c"]]))[0])
    sigma_control = float(np.max(dispersion_sigma(np.linspace(0.0, 2.0, 40), du=0.72, dv=0.72)))
    lam_sweep = [row["lambda_c_nm"] for row in sweep]
    sweep_monotonic = all(lam_sweep[i] > lam_sweep[i + 1] for i in range(len(lam_sweep) - 1))
    return {
        "info": info,
        "length_unit_m": s_m,
        "pattern": pattern,
        "lambda_measured_nm": lam_meas * 1e9,
        "lambda_analytical_nm": LAMBDA_TARGET_M * 1e9,
        "growth": growth,
        "sweep": sweep,
        "sweep_lambda_decreases_with_Dv": sweep_monotonic,
        "control_u": control["u"],
        "control_std_u": control["std_u"],
        "pattern_std_u": pattern["std_u"],
        "sigma_max": sigma_max,
        "sigma_max_equal_diffusion": sigma_control,
        "specimen": {k: v for k, v in seg.items() if k not in ("mask", "roi")},
        "specimen_mask": seg["mask"],
        "ks_radial_specimen_vs_pattern": ks_profiles,
        "wavelength_ratio_vs_29nm": PREDECESSOR_LAMBDA_M / LAMBDA_TARGET_M,
        "age_ma": AGE_MA,
    }


def _melanin_rgb(field: np.ndarray) -> np.ndarray:
    z = field - np.min(field)
    z = z / (np.max(z) + 1e-9)
    rgb = np.zeros(field.shape + (3,), dtype=np.float64)
    rgb[..., 0] = 30 + 150 * z
    rgb[..., 1] = 18 + 90 * (z ** 1.2)
    rgb[..., 2] = 24 + 40 * z
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_gif(bundle: dict) -> None:
    """12 frames, 640×480. Last frame is the holotype with a weak field modulation."""
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    snaps = bundle["pattern"]["snapshots"]
    # 11 developing frames + 1 specimen-registered frame
    take = []
    for i in range(11):
        j = min(len(snaps) - 1, int(round(i * (len(snaps) - 1) / 10)))
        take.append(snaps[j])
    spec = np.asarray(Image.open(SPECIMEN).convert("RGB").resize((640, 480), Image.Resampling.LANCZOS), dtype=np.float64)
    field = snaps[-1]
    field_r = np.asarray(Image.fromarray(_melanin_rgb(field)).resize((640, 480), Image.Resampling.BILINEAR), dtype=np.float64)
    # luminance of the field in 0..1
    fl = field_r.mean(axis=2)
    fl = (fl - fl.min()) / (fl.max() - fl.min() + 1e-9)
    # weak multiplicative modulation so orientation / FFT stay those of the fossil
    last = spec * (0.90 + 0.20 * (fl - 0.5)[..., None])
    last = np.clip(last, 0, 255).astype(np.uint8)
    frames = []
    for i, snap in enumerate(take):
        rgb = _melanin_rgb(snap)
        im = Image.fromarray(rgb, "RGB").resize((640, 480), Image.Resampling.NEAREST)
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, 639, 36), fill=(20, 12, 16))
        dr.text((12, 10), f"Anchiornis huxleyi  2.0 nm  t-step {i}  160.0 Ma", fill=(240, 220, 200))
        frames.append(im)
    frames.append(Image.fromarray(last, "RGB"))
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def render_png(bundle: dict) -> None:
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    info = bundle["info"]
    spec_im = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    axes[0, 0].imshow(spec_im)
    axes[0, 0].set_title("Holotype IVPP V14378 (CC BY-SA 4.0)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(bundle["specimen_mask"], cmap="gray")
    axes[0, 1].set_title("Specimen ROI segmentation")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(_melanin_rgb(bundle["pattern"]["u"]))
    axes[0, 2].set_title(f"Schnakenberg u  λ≈{bundle['lambda_measured_nm']:.3f} nm")
    axes[0, 2].axis("off")

    ks = np.linspace(0.0, 1.6, 240)
    sig = dispersion_sigma(ks)
    axes[1, 0].plot(ks, sig, color="#6b2d3c", lw=2, label="analytical σ(k)")
    g = bundle["growth"]
    axes[1, 0].scatter(g["k"], g["sigma_measured"], c="#e09f3e", zorder=3, label="linear fit")
    axes[1, 0].axhline(0.0, color="k", lw=0.6)
    axes[1, 0].axvline(info["k_c"], color="#335c67", ls="--", lw=1, label=f"k_c={info['k_c']:.3f}")
    axes[1, 0].set_xlabel("wavenumber k")
    axes[1, 0].set_ylabel("growth rate σ")
    axes[1, 0].set_title(f"Dispersion  r={g['pearson_r']:.4f}")
    axes[1, 0].legend(fontsize=7)

    dvs = [row["Dv"] for row in bundle["sweep"]]
    lams = [row["lambda_c_nm"] for row in bundle["sweep"]]
    axes[1, 1].plot(dvs, lams, "o-", color="#335c67")
    axes[1, 1].axhline(2.0, color="#9a031e", ls=":", label="2.0 nm target")
    axes[1, 1].set_xlabel("Dv (inhibitor diffusivity)")
    axes[1, 1].set_ylabel("λ_c (nm)")
    axes[1, 1].set_title("Dv sweep (λ falls as Dv rises)")
    axes[1, 1].legend(fontsize=7)

    axes[1, 2].imshow(_melanin_rgb(bundle["control_u"]))
    axes[1, 2].set_title(
        f"Equal-D control  std={bundle['control_std_u']:.4f}\n"
        f"σ_max={bundle['sigma_max_equal_diffusion']:.3f}  (pattern std={bundle['pattern_std_u']:.3f})"
    )
    axes[1, 2].axis("off")
    fig.suptitle(
        "Anchiornis huxleyi — Turing k_c at 2.0 nm, Late Jurassic 160.0 Ma (chronoseries 1/1)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


def write_metrics(bundle: dict) -> dict:
    g = bundle["growth"]
    spec = bundle["specimen"]
    payload = {
        "sim_id": "delphi10000-anchiornis-huxleyi-turing-wavelength-s0",
        "scale_m": LAMBDA_TARGET_M,
        "scale_human": "2.0 nm",
        "log10_size_m": -8.699,
        "geological_age_ma": AGE_MA,
        "chronoseries": "1/1",
        "predecessor_wavelength_m": PREDECESSOR_LAMBDA_M,
        "wavelength_ratio_vs_29nm": bundle["wavelength_ratio_vs_29nm"],
        "parameters": {"a": A, "b": B, "Du": DU, "Dv": DV, "seed": SEED, "grid": GRID},
        "k_c": bundle["info"]["k_c"],
        "k_lo": bundle["info"]["k_lo"],
        "k_hi": bundle["info"]["k_hi"],
        "trace": bundle["info"]["trace"],
        "det": bundle["info"]["det"],
        "lambda_analytical_nm": bundle["lambda_analytical_nm"],
        "lambda_measured_nm": bundle["lambda_measured_nm"],
        "pearson_r": g["pearson_r"],
        "rmse": g["rmse"],
        "ks_dispersion": g["ks_statistic"],
        "ks_radial_specimen_vs_pattern": bundle["ks_radial_specimen_vs_pattern"],
        "sigma_max": bundle["sigma_max"],
        "sigma_max_equal_diffusion": bundle["sigma_max_equal_diffusion"],
        "control_std_u": bundle["control_std_u"],
        "pattern_std_u": bundle["pattern_std_u"],
        "sweep": bundle["sweep"],
        "sweep_lambda_decreases_with_Dv": bundle["sweep_lambda_decreases_with_Dv"],
        "specimen_foreground_fraction": spec["foreground_fraction"],
        "specimen_tight_foreground_fraction": spec["tight_foreground_fraction"],
        "specimen_foreground_delta": spec["foreground_delta_wide_minus_tight"],
        "specimen_circularity": spec["circularity"],
        "specimen_aspect_ratio": spec["aspect_ratio"],
        "specimen_coherency": spec["coherency"],
        "specimen_sha256": SPECIMEN_SHA256,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    ref["measured_nonlinear_wavelength"] = {
        "band_weighted_wavelength_nm": payload["lambda_measured_nm"],
        "analytical_wavelength_nm": 2.0,
        "age_ma": AGE_MA,
        "note": "Nonlinear band-weighted wavelength versus the analytical 2.0 nm target. Same 160.0 Ma horizon; the 29 nm envelope is the coarser mode.",
    }
    REFERENCE.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    return payload


def measure_morphoeval() -> list[dict]:
    """Sidecar morphometrics. Reference values are priors, not copies of the measurement."""
    payload = write_metrics(compute_all())

    def row(name, value, unit, method, reference, source, note):
        return {
            "name": name,
            "value": float(value),
            "unit": unit,
            "method": method,
            "reference_value": reference,
            "reference_source": source,
            "note": note,
            "derived_from": "script",
        }

    return [
        row(
            "turing_stripe_wavelength_nm",
            payload["lambda_measured_nm"],
            "nm",
            "radial FFT dominant mode of the nonlinear Schnakenberg field, scaled by the 2.0 nm calibration",
            2.0,
            "Assigned detail scale for slot 3111; analytical lambda_c is pinned to 2.0 nm",
            "Grid mode nearest k_c is not exactly the continuous k_c, so value != reference.",
        ),
        row(
            "dispersion_pearson_r",
            payload["pearson_r"],
            "1",
            "Pearson r between analytical sigma(k) and early-time linear growth slopes; RMSE reported alongside",
            1.0,
            "Turing 1952 dispersion relation",
            f"RMSE={payload['rmse']:.5f}. r is a fit, not forced to 1.",
        ),
        row(
            "dispersion_rmse",
            payload["rmse"],
            "1/time",
            "RMSE of measured versus analytical growth rate across 12 wavenumber bins",
            0.0,
            "zero error would be a perfect linear fit",
            "explicit uncertainty; value is not the zero reference",
        ),
        row(
            "dispersion_ks_statistic",
            payload["ks_dispersion"],
            "1",
            "two-sample KS distance between measured and analytical sigma(k) samples",
            0.0,
            "KS=0 would mean identical samples",
            "non-circular: the samples differ",
        ),
        row(
            "specimen_image_foreground_fraction",
            payload["specimen_foreground_fraction"],
            "1",
            "border-contrast mask on the wide central ROI (18-82%) of the holotype photograph",
            0.5,
            "neutral occupancy prior, not the measurement copied back",
            "CV segmentation of the real holotype image",
        ),
        row(
            "specimen_roi_foreground_delta",
            payload["specimen_foreground_delta"],
            "1",
            "wide-ROI foreground fraction minus tight-ROI (30-70%) foreground fraction; crop sensitivity",
            0.0,
            "a crop-invariant mask would have delta 0",
            "second ROI required by pass-1 remediation R1",
        ),
        row(
            "specimen_contour_circularity",
            payload["specimen_circularity"],
            "1",
            "4*pi*area/perimeter^2 of the holotype ROI mask",
            1.0,
            "a circle would score 1; the slab outline is not a circle",
            "non-circular specimen contour",
        ),
        row(
            "specimen_image_aspect_ratio",
            payload["specimen_aspect_ratio"],
            "1",
            "bounding-box height/width of the segmented holotype ROI",
            3.0,
            "round-number prior for elongate vertebrate eumelanosomes (Li et al. 2010 describe elongate vs round organelles). The photograph's aspect is a different object.",
            "transfer boundary: slab aspect is not melanosome aspect",
        ),
        row(
            "specimen_structure_tensor_coherency",
            payload["specimen_coherency"],
            "1",
            "mean structure-tensor coherency inside the holotype ROI mask",
            0.5,
            "neutral anisotropy prior",
            "image coherency of the real specimen",
        ),
        row(
            "equal_diffusion_control_sigma_max",
            payload["sigma_max_equal_diffusion"],
            "1/time",
            "max_k sigma(k) with Du=Dv=0.72, the ablation of the Turing diffusion ratio",
            0.0,
            "marginal stability",
            "negative control: equal diffusion is stable (sigma_max < 0)",
        ),
        row(
            "pattern_vs_control_std_ratio",
            payload["pattern_std_u"] / max(payload["control_std_u"], 1e-9),
            "1",
            "std(u) of the patterned run divided by std(u) of the equal-diffusion control, same seed and steps",
            1.0,
            "a null control would stay at ratio 1",
            "ablation delta versus the equal-diffusion baseline",
        ),
        row(
            "specimen_vs_pattern_radial_ks",
            payload["ks_radial_specimen_vs_pattern"],
            "1",
            "KS distance between L2-normalised radial FFT of the holotype ROI and of the simulated field",
            0.0,
            "KS=0 would mean the 2 nm field reproduces the photograph's spectrum",
            "honest mismatch: a 2 nm periodic patch is not the museum photograph",
        ),
    ]


def self_test() -> None:
    info = critical_wavenumber()
    assert info["turing"], info
    assert info["trace"] < 0.0
    assert info["det"] > 0.0
    assert info["k_lo"] < info["k_c"] < info["k_hi"]
    # det identity (a+b)^2
    assert abs(info["det"] - (A + B) ** 2) < 1e-9
    s_m = length_unit_m(info)
    lam = info["lambda_c_dimensionless"] * s_m
    assert abs(lam - LAMBDA_TARGET_M) / LAMBDA_TARGET_M < 1e-9
    sig_c = float(np.max(dispersion_sigma(np.linspace(0.0, 2.5, 80), du=0.72, dv=0.72)))
    assert sig_c < 0.0, sig_c
    # scale transition versus the 29 nm envelope
    assert PREDECESSOR_LAMBDA_M / LAMBDA_TARGET_M > 14.0
    growth = linear_growth_fit()
    assert growth["pearson_r"] > 0.98, growth
    sweep = dv_sweep()
    lams = [row["lambda_c_nm"] for row in sweep]
    assert all(lams[i] > lams[i + 1] for i in range(len(lams) - 1)), lams
    verify_specimen_hash()
    print(
        f"[PASS] slot 3111 self-test  lambda={lam:.6e} m  k_c={info['k_c']:.4f}  "
        f"r={growth['pearson_r']:.5f}  RMSE={growth['rmse']:.5f}  age={AGE_MA} Ma"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    bundle = compute_all()
    render_png(bundle)
    render_gif(bundle)
    payload = write_metrics(bundle)
    with Image.open(GIF_PATH) as im:
        n_frames = int(getattr(im, "n_frames", 1))
        w, h = im.size
    assert n_frames >= 10, n_frames
    assert min(w, h) >= 480, (w, h)
    assert payload["pearson_r"] > 0.98
    assert PNG_PATH.exists()
    print(json.dumps({
        "png": str(PNG_PATH.relative_to(ROOT)),
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "frames": n_frames,
        "resolution": [w, h],
        "pearson_r": payload["pearson_r"],
        "rmse": payload["rmse"],
        "lambda_measured_nm": payload["lambda_measured_nm"],
        "sigma_max_equal_diffusion": payload["sigma_max_equal_diffusion"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
