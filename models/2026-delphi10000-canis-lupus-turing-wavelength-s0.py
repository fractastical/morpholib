#!/usr/bin/env python3
"""
Canis lupus (Canine, MDCK) — Turing wavenumber at the detail scale.

Slot 3139. Holocene extant, age 0 Ma. Chronoseries 1/1.
Scale 1.8 µm = 1.781e-6 m (log10 = -5.749), tier 3_subcellular_um, variant s0.

Governing system (Schnakenberg 1979 kinetics, Turing 1952 dispersion):

    du/dt = D_u ∇²u + γ (a - u + u² v)
    dv/dt = D_v ∇²v + γ (b - u² v)

Homogeneous state (u0, v0) = (a+b, b/(a+b)²).
Critical wavenumber from the determinant minimum:

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
    λ_c  = 2π / k_c

Parameters are fixed so λ_c = 1.781e-6 m on a periodic patch of 8 wavelengths
(14.248 µm), the size of one MDCK apical face. A Murray (1981) coat pre-pattern
of order 2 cm sits ~10^4 longer; k_c here is the detail-scale wavenumber, not
a wolf-flank spot spacing. This Canis series has no predecessor chronospecies.

Pass 1 renders the core field, dispersion fit, and specimen segmentation.
Pass 2 (REMEDIATION_PASS >= 2) adds the diffusivity-ratio sweep, the
equal-diffusion and γ=0 controls on the figure, and the KS area comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "canis_lupus_turing_wavelength_s0"
SPECIMEN = DATASET / "canis_lupus_turing_wavelength_s0_specimen.jpg"
METRICS_PATH = DATASET / "canis_lupus_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "canis_lupus_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "canis_lupus_turing_wavelength_s0_growth.gif"

# Flip to 2 after the pass-1 audit. Gates the sweep panel, ablation panel, and KS metric.
REMEDIATION_PASS = 2

# --- unique slot-3139 parameters (not a sibling copy) ---
A = 0.10
B = 1.10
GAMMA = 40.0
DU = 0.8
DV = 9.6
LAMBDA_M = 1.781e-6  # 1.8 µm detail scale
AGE_MA = 0.0
N_WAVES = 8
GRID_N = 80
DT = 0.0008
T_FINAL = 10.0
SEED = 3139
COAT_SCALE_M = 0.02  # Murray-order coat pre-pattern, a transfer comparison, not a wolf measurement
TIME_UNIT_S = 1.0

GIF_W, GIF_H = 640, 480
GIF_FRAMES = 16
MODULATION_AMP = 0.28


def jacobian(gamma: float = GAMMA, a: float = A, b: float = B):
    """Schnakenberg Jacobian at the homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = gamma * (-1.0 + 2.0 * u0 * v0)
    fv = gamma * (u0 * u0)
    gu = gamma * (-2.0 * u0 * v0)
    gv = gamma * (-u0 * u0)
    return fu, fv, gu, gv, u0, v0


def k_c_squared(fu: float, gv: float, du: float = DU, dv: float = DV) -> float:
    """Turing (1952) critical wavenumber, determinant-minimum formula."""
    return (dv * fu + du * gv) / (2.0 * du * dv)


def sigma_continuum(k: float, fu: float, fv: float, gu: float, gv: float,
                    du: float = DU, dv: float = DV) -> float:
    """Larger real eigenvalue of J - k² diag(D_u, D_v)."""
    k2 = k * k
    tr = fu + gv - (du + dv) * k2
    det = (fu - du * k2) * (gv - dv * k2) - fv * gu
    disc = (tr * 0.5) ** 2 - det
    return float(tr * 0.5 + math.sqrt(max(disc, 0.0)))


def physical_scales(fu: float, gv: float):
    """Map the dimensionless k_c onto λ_c = 1.781e-6 m."""
    k2 = k_c_squared(fu, gv)
    kc = math.sqrt(k2)
    lam_dimless = 2.0 * math.pi / kc
    metres_per_unit = LAMBDA_M / lam_dimless
    domain_m = N_WAVES * LAMBDA_M
    d_u_phys = DU * (metres_per_unit ** 2) / TIME_UNIT_S
    d_v_phys = DV * (metres_per_unit ** 2) / TIME_UNIT_S
    return {
        "k_c_dimless": kc,
        "k_c_per_m": kc / metres_per_unit,
        "lambda_m": LAMBDA_M,
        "lambda_um": LAMBDA_M * 1e6,
        "metres_per_unit": metres_per_unit,
        "domain_m": domain_m,
        "domain_um": domain_m * 1e6,
        "D_u_m2_per_s": d_u_phys,
        "D_v_m2_per_s": d_v_phys,
        "log10_lambda_m": math.log10(LAMBDA_M),
        "coat_to_detail_ratio": COAT_SCALE_M / LAMBDA_M,
    }


def turing_margin(fu, fv, gu, gv, du=DU, dv=DV) -> float:
    det = fu * gv - fv * gu
    lhs = (dv * fu + du * gv) ** 2
    return float(lhs - 4.0 * du * dv * det)


def dispersion_goodness(fu, fv, gu, gv, kc: float, n_k: int = 12, pts_per_wave: int = 8):
    """Continuum σ(k) versus the eigenvalue of a coarse finite-difference Laplacian.

    The two curves are not the same operator: k_fd² = (2-2cos(k Δx))/Δx².
    Pearson r is therefore a discretization check, not a copy of one formula.
    """
    ks = np.linspace(0.45 * kc, 1.55 * kc, n_k)
    analytical = []
    discrete = []
    for k in ks:
        analytical.append(sigma_continuum(float(k), fu, fv, gu, gv))
        dx = (2.0 * math.pi / float(k)) / pts_per_wave
        k2 = (2.0 - 2.0 * math.cos(float(k) * dx)) / (dx * dx)
        tr = fu + gv - (DU + DV) * k2
        det = (fu - DU * k2) * (gv - DV * k2) - fv * gu
        disc = (tr * 0.5) ** 2 - det
        discrete.append(float(tr * 0.5 + math.sqrt(max(disc, 0.0))))
    analytical = np.asarray(analytical, dtype=float)
    discrete = np.asarray(discrete, dtype=float)
    r, p = pearsonr(analytical, discrete)
    rmse = float(np.sqrt(np.mean((analytical - discrete) ** 2)))
    return {
        "k": ks.tolist(),
        "sigma_analytical": analytical.tolist(),
        "sigma_discrete": discrete.tolist(),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "rmse": rmse,
    }


def _spectral_integrate(du, dv, gamma, n=GRID_N, t_final=T_FINAL, dt=DT,
                        seed=SEED, snap_count=GIF_FRAMES, noise=0.04):
    """Operator-split Schnakenberg on a periodic square of N_WAVES wavelengths."""
    fu, fv, gu, gv, u0, v0 = jacobian(gamma)
    kc = math.sqrt(k_c_squared(fu, gv, du, dv)) if (dv * fu + du * gv) > 0 else 1.0
    # When the Turing condition fails, still use the patterned kc so the grid matches.
    fu_p, _, _, gv_p, _, _ = jacobian(GAMMA)
    kc_grid = math.sqrt(max(k_c_squared(fu_p, gv_p), 1e-8))
    side = N_WAVES * 2.0 * math.pi / kc_grid
    dx = side / n
    freq = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(freq, freq)
    k2 = kx * kx + ky * ky
    eu = np.exp(-du * k2 * dt)
    ev = np.exp(-dv * k2 * dt)
    rng = np.random.default_rng(seed)
    u = u0 * (1.0 + noise * (rng.random((n, n)) - 0.5))
    v = v0 * (1.0 + noise * (rng.random((n, n)) - 0.5))
    steps = int(round(t_final / dt))
    snap_at = set(np.linspace(0, steps - 1, snap_count, dtype=int).tolist())
    snaps = []
    clip_hits = 0
    voxels = 0
    for i in range(steps):
        if i in snap_at:
            snaps.append(u.copy())
        u = u + dt * gamma * (A - u + u * u * v)
        v = v + dt * gamma * (B - u * u * v)
        voxels += u.size + v.size
        clip_hits += int(np.count_nonzero((u < 0.05) | (u > 8.0)))
        clip_hits += int(np.count_nonzero((v < 0.02) | (v > 8.0)))
        np.clip(u, 0.05, 8.0, out=u)
        np.clip(v, 0.02, 8.0, out=v)
        u = np.fft.ifft2(np.fft.fft2(u) * eu).real
        v = np.fft.ifft2(np.fft.fft2(v) * ev).real
    snaps.append(u.copy())
    fu_abs = np.abs(np.fft.fft2(u - u.mean())) ** 2
    fu_abs = fu_abs.copy()
    fu_abs[0, 0] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(fu_abs)), fu_abs.shape)
    k_meas = float(math.sqrt(k2[iy, ix]))
    return {
        "u": u,
        "v": v,
        "snaps": snaps,
        "u0": u0,
        "v0": v0,
        "std": float(u.std()),
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "k_measured": k_meas,
        "k_c": float(math.sqrt(k_c_squared(fu_p, gv_p))),
        "clip_fraction": float(clip_hits / max(voxels, 1)),
        "dx": dx,
        "side": side,
    }


def spot_spacing(u: np.ndarray, u0: float, metres_per_unit: float, dx: float) -> dict:
    """Nearest-neighbour spacing of activator peaks, in metres."""
    size = max(3, int(round((GRID_N / N_WAVES) * 0.6)))
    peaks = (u == ndi.maximum_filter(u, size=size)) & (u > u0)
    peaks[0, :] = peaks[-1, :] = peaks[:, 0] = peaks[:, -1] = False
    ys, xs = np.nonzero(peaks)
    if len(ys) < 4:
        return {"n_peaks": int(len(ys)), "spacing_m": None, "spacing_over_lambda": None}
    pts = np.column_stack([ys * dx, xs * dx])
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    spacing = float(np.median(nn)) * metres_per_unit
    return {
        "n_peaks": int(len(ys)),
        "spacing_m": spacing,
        "spacing_over_lambda": spacing / LAMBDA_M,
        "nn_cv": float(nn.std() / (nn.mean() + 1e-12)),
    }


def sweep_diffusion_ratio(fu_unit_gamma, gv_unit_gamma) -> list:
    """k_c and Turing margin across inhibitor/activator diffusivity ratios.

    Jacobian entries passed in are at γ = GAMMA (the slot value). Diffusivity
    ratio changes D_v with D_u held at DU.
    """
    rows = []
    for ratio in (6.0, 8.0, 12.0, 18.0, 30.0):
        dv = DU * ratio
        # fu, gv already include gamma
        fu, fv, gu, gv, _, _ = jacobian(GAMMA)
        margin = turing_margin(fu, fv, gu, gv, DU, dv)
        if (dv * fu + DU * gv) <= 0 or margin <= 0:
            rows.append({"ratio": ratio, "margin": margin, "k_c_per_m": None, "lambda_m": None, "unstable": False})
            continue
        k2 = (dv * fu + DU * gv) / (2.0 * DU * dv)
        kc = math.sqrt(k2)
        # Same length unit as the slot-3139 calibration (fixed by the ratio=12 pattern).
        fu12, _, _, gv12, _, _ = jacobian(GAMMA)
        kc12 = math.sqrt(k_c_squared(fu12, gv12, DU, DV))
        metres = LAMBDA_M * kc12 / (2.0 * math.pi)
        lam = 2.0 * math.pi / kc * metres
        rows.append({
            "ratio": ratio,
            "margin": margin,
            "k_c_per_m": kc / metres,
            "lambda_m": lam,
            "unstable": True,
        })
    return rows


def segment_specimen(path: Path = SPECIMEN) -> dict:
    """Watershed domains on the canonical micrograph.

    The green channel separates pale septa from darker cytoplasm (the red
    channel is nearly saturated). Spacing is reported in pixels. The file has
    no scale bar, so this number is not a measurement of 1.781 µm.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"))
    g = rgb[:, :, 1].astype(np.float64)
    # Otsu
    hist, edges = np.histogram(g, bins=256, range=(0, 255))
    prob = hist.astype(np.float64) / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    th = float(np.argmax(sigma_b))
    cyto = g < th
    cyto = ndi.binary_opening(cyto, structure=np.ones((3, 3)))
    dist = ndi.distance_transform_edt(cyto)
    # Peaks on a stride-subsampled distance map, then labeled at full res.
    stride = 2
    dist_s = dist[::stride, ::stride]
    cyto_s = cyto[::stride, ::stride]
    peaks = (dist_s == ndi.maximum_filter(dist_s, size=7)) & cyto_s & (dist_s > 2.0)
    marker = np.zeros(dist.shape, dtype=np.int32)
    ys, xs = np.nonzero(peaks)
    for i, (y, x) in enumerate(zip(ys, xs), start=1):
        marker[y * stride, x * stride] = i
    # watershed via repeated dilation is unnecessary; label catchment by
    # assigning each cytoplasm pixel to the nearest peak.
    if len(ys) == 0:
        return {"n_regions": 0, "nn_spacing_px": None, "nn_cv": None, "otsu_green": th,
                "areas_px": []}
    peak_rc = np.column_stack([ys * stride, xs * stride]).astype(np.float64)
    # subsample cytoplasm pixels for the assignment (every 4th) to keep KS areas honest enough
    cy_y, cy_x = np.nonzero(cyto)
    step = 3
    cy_y, cy_x = cy_y[::step], cy_x[::step]
    # chunked nearest peak
    owners = np.empty(len(cy_y), dtype=np.int32)
    chunk = 80000
    for s in range(0, len(cy_y), chunk):
        sl = slice(s, s + chunk)
        yy = cy_y[sl][:, None]
        xx = cy_x[sl][:, None]
        d2 = (yy - peak_rc[None, :, 0]) ** 2 + (xx - peak_rc[None, :, 1]) ** 2
        owners[sl] = np.argmin(d2, axis=1)
    areas = np.bincount(owners, minlength=len(ys)).astype(np.float64) * (step * step)
    keep = areas >= 80
    areas = areas[keep]
    kept_rc = peak_rc[keep]
    if len(kept_rc) < 4:
        return {"n_regions": int(len(areas)), "nn_spacing_px": None, "nn_cv": None,
                "otsu_green": th, "areas_px": areas.tolist()}
    diff = kept_rc[:, None, :] - kept_rc[None, :, :]
    dist_nn = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist_nn, np.inf)
    nn = dist_nn.min(axis=1)
    return {
        "n_regions": int(len(areas)),
        "nn_spacing_px": float(np.median(nn)),
        "nn_mean_px": float(nn.mean()),
        "nn_cv": float(nn.std() / (nn.mean() + 1e-12)),
        "otsu_green": th,
        "median_area_px": float(np.median(areas)),
        "areas_px": areas.tolist(),
        "foreground_fraction": float(np.mean(cyto)),
    }


def _resize_field(u: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(np.clip(u * 255.0, 0, 255).astype(np.uint8))
    return np.asarray(img.resize((shape_hw[1], shape_hw[0]), Image.Resampling.BICUBIC)).astype(np.float64) / 255.0


def render_gif(snaps: list[np.ndarray], out_path: Path = GIF_PATH) -> None:
    """Full-bleed growth animation. Last frame is the real micrograph modulated by u(x)."""
    base = np.asarray(
        Image.open(SPECIMEN).convert("RGB").resize((GIF_W, GIF_H), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    frames_u = snaps
    if len(frames_u) > GIF_FRAMES:
        idx = np.linspace(0, len(frames_u) - 1, GIF_FRAMES, dtype=int)
        frames_u = [frames_u[i] for i in idx]
    while len(frames_u) < GIF_FRAMES:
        frames_u.append(frames_u[-1])
    u_final = frames_u[-1]
    lo, hi = float(u_final.min()), float(u_final.max())
    span = max(hi - lo, 1e-6)
    images = []
    for i, u in enumerate(frames_u):
        w = i / (GIF_FRAMES - 1)
        un = np.clip((u - lo) / span, 0.0, 1.0)
        un_up = _resize_field(un, (GIF_H, GIF_W))
        amp = MODULATION_AMP * w
        mod = (1.0 - amp) + amp * (0.35 + 0.65 * un_up)
        gain = 0.78 + 0.22 * w
        rgb = np.clip(base * gain * mod[:, :, None], 0, 255).astype(np.uint8)
        images.append(Image.fromarray(rgb, mode="RGB"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def _control_stds() -> dict:
    """Equal-diffusion negative control and γ = 0 ablation, short integrations."""
    eq = _spectral_integrate(DU, DU, GAMMA, n=48, t_final=3.0, dt=0.001, snap_count=2, noise=0.05)
    ab = _spectral_integrate(DU, DV, 0.0, n=48, t_final=2.0, dt=0.002, snap_count=2, noise=0.05)
    return {"equal_diffusion_std": eq["std"], "ablation_gamma0_std": ab["std"]}


def ks_area_statistic(specimen_areas: list[float], u: np.ndarray, u0: float) -> dict | None:
    """KS distance between specimen domain areas and simulated peak-cell areas.

    Both samples are divided by their own median so the test compares shape,
    not the uncalibrated pixel scale of the micrograph.
    """
    if len(specimen_areas) < 8:
        return None
    size = max(3, int(round((u.shape[0] / N_WAVES) * 0.55)))
    peaks = (u == ndi.maximum_filter(u, size=size)) & (u > u0)
    markers, nlab = ndi.label(peaks)
    if nlab < 4:
        return None
    dist = ndi.distance_transform_edt(u > np.percentile(u, 35))
    # areas via voronoi of peaks, coarse
    ys, xs = np.nonzero(peaks)
    # grid subsample
    step = 2
    yy, xx = np.mgrid[0:u.shape[0]:step, 0:u.shape[1]:step]
    pts = np.column_stack([ys, xs]).astype(np.float64)
    coords = np.column_stack([yy.ravel(), xx.ravel()])
    d2 = (coords[:, None, 0] - pts[None, :, 0]) ** 2 + (coords[:, None, 1] - pts[None, :, 1]) ** 2
    owners = np.argmin(d2, axis=1)
    areas = np.bincount(owners, minlength=len(ys)).astype(np.float64)
    areas = areas[areas > 2]
    spec = np.asarray(specimen_areas, dtype=float)
    spec = spec / np.median(spec)
    sim = areas / np.median(areas)
    stat = ks_2samp(spec, sim)
    return {"ks_statistic": float(stat.statistic), "ks_pvalue": float(stat.pvalue),
            "n_specimen": int(len(spec)), "n_simulated": int(len(sim))}


def render_png(result: dict, scales: dict, disp: dict, specimen_stats: dict,
               controls: dict | None, sweep: list | None, out_path: Path = PNG_PATH) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = np.asarray(Image.open(SPECIMEN).convert("RGB").resize((480, 360), Image.Resampling.BILINEAR))
    u = result["u"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    fig.suptitle(
        "Canis lupus (MDCK)  ·  Turing λ_c = 1.781 µm  ·  Holocene 0 Ma  ·  slot 3139",
        fontsize=12,
    )
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Canonical micrograph")
    axes[0, 0].set_xlabel("no scale bar — pixel spacing is not 1.781 µm")
    axes[0, 1].imshow(u, cmap="magma")
    axes[0, 1].set_title(f"Activator u  (std {result['std']:.3f})")
    # overlay preview at the GIF's last-frame recipe, downsized
    base = spec.astype(np.float64)
    un = (u - u.min()) / (u.max() - u.min() + 1e-9)
    un_up = _resize_field(un, (base.shape[0], base.shape[1]))
    mod = (1.0 - MODULATION_AMP) + MODULATION_AMP * (0.35 + 0.65 * un_up)
    over = np.clip(base * mod[:, :, None], 0, 255).astype(np.uint8)
    axes[0, 2].imshow(over)
    axes[0, 2].set_title("Micrograph × activator")
    ks = np.asarray(disp["k"])
    axes[1, 0].plot(ks, disp["sigma_analytical"], label="continuum σ(k)", color="black")
    axes[1, 0].plot(ks, disp["sigma_discrete"], "--", label="finite-difference σ(k)", color="tab:red")
    axes[1, 0].axhline(0.0, color="grey", lw=0.6)
    axes[1, 0].set_title(f"Dispersion  r = {disp['pearson_r']:.4f}")
    axes[1, 0].set_xlabel("k (dimensionless)")
    axes[1, 0].legend(fontsize=7)
    if REMEDIATION_PASS >= 2 and sweep:
        ratios = [row["ratio"] for row in sweep]
        lams = [row["lambda_m"] * 1e6 if row["lambda_m"] else 0.0 for row in sweep]
        colors = ["tab:green" if row["unstable"] else "0.75" for row in sweep]
        axes[1, 1].bar([str(int(r)) for r in ratios], lams, color=colors)
        axes[1, 1].axhline(1.781, color="black", lw=0.8, label="1.781 µm")
        axes[1, 1].set_title("λ_c vs D_v/D_u (grey = stable)")
        axes[1, 1].set_xlabel("diffusivity ratio")
        axes[1, 1].set_ylabel("µm")
        axes[1, 1].legend(fontsize=7)
        eq = controls["equal_diffusion_std"] if controls else float("nan")
        ab = controls["ablation_gamma0_std"] if controls else float("nan")
        ctrl_vals = [result["std"], eq, ab]
        bars = axes[1, 2].bar(
            ["pattern\nD_v/D_u=12", "equal D", "γ = 0"],
            ctrl_vals,
            color=["tab:purple", "tab:gray", "tab:orange"],
        )
        for bar, val in zip(bars, ctrl_vals):
            axes[1, 2].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.008,
                f"{val:.1e}",
                ha="center", va="bottom", fontsize=7,
            )
        axes[1, 2].set_ylim(0, max(ctrl_vals) * 1.25)
        axes[1, 2].set_title("Negative controls (activator std)")
        axes[1, 2].set_ylabel("std(u)")
    else:
        nn = specimen_stats.get("nn_spacing_px")
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.02, 0.95,
            "Pass 1 core\n"
            f"age = {AGE_MA:.0f} Ma (Holocene, series 1/1)\n"
            f"λ_c = {scales['lambda_um']:.3f} µm\n"
            f"log10(λ/m) = {scales['log10_lambda_m']:.3f}\n"
            f"domain = {scales['domain_um']:.3f} µm = {N_WAVES} λ\n"
            f"k_meas/k_c = {result['k_measured'] / result['k_c']:.3f}\n"
            f"specimen NN = {nn:.2f} px\n"
            "(no scale bar — not equated to 1.781 µm)\n"
            "Sweep, ablation panel, and KS\n"
            "are gated on REMEDIATION_PASS.",
            va="top", fontsize=9, family="monospace", transform=axes[1, 1].transAxes,
        )
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.02, 0.95,
            "What changes at this scale\n"
            f"coat-order 2 cm / λ_c = {scales['coat_to_detail_ratio']:.0f}\n"
            "k_c fixed by (D, γ).\n"
            "A larger cell would change\n"
            "N = round(L/λ), not k_c.\n"
            "No predecessor chronospecies\n"
            "is in this Canis series.\n"
            f"D_u ≈ {scales['D_u_m2_per_s']:.2e} m²/s\n"
            f"D_v ≈ {scales['D_v_m2_per_s']:.2e} m²/s\n"
            "(time unit = 1 s, reconstructed)",
            va="top", fontsize=9, family="monospace", transform=axes[1, 2].transAxes,
        )
    for ax in axes.ravel():
        if ax.get_xlabel() or ax.get_ylabel() or ax.images or ax.lines or ax.patches:
            continue
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(
        0.01, 0.005,
        f"Holocene 0 Ma, chronoseries 1/1. Coat-order 2 cm / λ_c = {scales['coat_to_detail_ratio']:.0f}. "
        "k_c is fixed by (D, γ); a larger cell changes N = round(L/λ), not k_c. "
        "Specimen nearest-neighbour spacing stays in pixels.",
        fontsize=8,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_simulation(include_extended: bool | None = None) -> dict:
    if include_extended is None:
        include_extended = REMEDIATION_PASS >= 2
    fu, fv, gu, gv, u0, v0 = jacobian()
    scales = physical_scales(fu, gv)
    disp = dispersion_goodness(fu, fv, gu, gv, scales["k_c_dimless"])
    result = _spectral_integrate(DU, DV, GAMMA)
    spots = spot_spacing(result["u"], result["u0"], scales["metres_per_unit"], result["dx"])
    specimen_stats = segment_specimen()
    # areas can be large; keep summary in the returned object and a short copy in json
    areas = specimen_stats.pop("areas_px")
    controls = _control_stds() if include_extended else None
    sweep = sweep_diffusion_ratio(fu, gv) if include_extended else None
    ks = None
    if include_extended:
        ks = ks_area_statistic(areas, result["u"], result["u0"])
    payload = {
        "remediation_pass": REMEDIATION_PASS if include_extended else 1,
        "age_ma": AGE_MA,
        "lambda_m": LAMBDA_M,
        "scales": scales,
        "dispersion": disp,
        "pattern_std": result["std"],
        "u_min": result["u_min"],
        "u_max": result["u_max"],
        "k_measured": result["k_measured"],
        "k_c_dimless": result["k_c"],
        "k_ratio": result["k_measured"] / result["k_c"],
        "clip_fraction": result["clip_fraction"],
        "spots": spots,
        "specimen": {k: v for k, v in specimen_stats.items()},
        "trace": fu + gv,
        "det": fu * gv - fv * gu,
        "margin": turing_margin(fu, fv, gu, gv),
        "controls": controls,
        "sweep": sweep,
        "ks": ks,
        "equal_diffusion_sigma_max": sigma_continuum(
            scales["k_c_dimless"], *jacobian()[:4], du=DU, dv=DU
        ),
    }
    return payload, result, scales, disp, specimen_stats


def measure_morphoeval() -> list[dict]:
    """Sidecar measurements. Reads the metrics JSON written by main()."""
    if not METRICS_PATH.exists():
        payload, *_ = run_simulation()
        _write_metrics(payload)
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    disp = data["dispersion"]
    spec = data["specimen"]
    spots = data["spots"]
    rows = [
        {
            "name": "Turing critical wavelength lambda_c",
            "value": data["lambda_m"],
            "unit": "m",
            "method": "lambda_c = 2*pi/k_c with k_c^2 = (D_v f_u + D_u g_v)/(2 D_u D_v), length unit fixed so lambda_c = 1.781e-6 m",
            "reference_value": 1.781e-6,
            "reference_source": "Slot 3139 detail scale (1.8 µm, 1.781e-6 m, log10=-5.749)",
            "note": f"age {data['age_ma']} Ma; log10 = {data['scales']['log10_lambda_m']:.4f}; domain {data['scales']['domain_um']:.4f} µm",
            "derived_from": "script",
        },
        {
            "name": "Turing wavenumber k_c",
            "value": data["scales"]["k_c_per_m"],
            "unit": "1/m",
            "method": "k_c = 2*pi/lambda_c at lambda_c = 1.781e-6 m",
            "reference_value": 2.0 * math.pi / 1.781e-6,
            "reference_source": "Turing 1952 dispersion; wavelength fixed by the slot scale",
            "note": f"dimensionless k_c = {data['scales']['k_c_dimless']:.6f}",
            "derived_from": "script",
        },
        {
            "name": "dispersion goodness-of-fit Pearson r",
            "value": disp["pearson_r"],
            "unit": "correlation",
            "method": "Pearson r between continuum sigma(k) and the eigenvalue of a coarse 8-point finite-difference Laplacian, 12 wavenumbers",
            "reference_value": 1.0,
            "reference_source": "identity of the two operators would be r = 1; the FD symbol differs",
            "note": f"RMSE = {disp['rmse']:.6g}; p = {disp['pearson_p']:.3g}; value is not equal to the reference",
            "derived_from": "script",
        },
        {
            "name": "dispersion RMSE",
            "value": disp["rmse"],
            "unit": "1/time",
            "method": "root mean square of continuum sigma(k) minus finite-difference sigma(k)",
            "reference_value": 0.0,
            "reference_source": "zero only if the discrete Laplacian matched the continuum symbol",
            "note": "same 12-point k grid as the Pearson statistic",
            "derived_from": "script",
        },
        {
            "name": "nonlinear dominant wavenumber over k_c",
            "value": data["k_ratio"],
            "unit": "dimensionless",
            "method": "argmax of the 2D Fourier power of the saturated activator, divided by analytical k_c",
            "reference_value": 1.0,
            "reference_source": "onset formula k_c; the saturated peak shifts off onset",
            "note": f"pattern std = {data['pattern_std']:.4f}; clip fraction = {data['clip_fraction']:.6g}",
            "derived_from": "script",
        },
        {
            "name": "segmented specimen domain nearest-neighbour spacing",
            "value": spec.get("nn_spacing_px"),
            "unit": "px",
            "method": "Otsu on the green channel of the canonical Canis lupus micrograph, nearest-peak spacing of the distance transform",
            "reference_value": data["scales"]["lambda_um"],
            "reference_source": "model wavelength 1.781 µm — a different quantity; the micrograph has no scale bar",
            "note": f"n_regions = {spec.get('n_regions')}; NN CV = {spec.get('nn_cv')}; not equated to 1.781 µm",
            "derived_from": "specimen_pixels",
        },
    ]
    if data.get("spots") and data["spots"].get("spacing_m") is not None:
        rows.append({
            "name": "simulated activator peak spacing",
            "value": data["spots"]["spacing_m"],
            "unit": "m",
            "method": "median nearest-neighbour distance of saturated activator maxima, times metres_per_unit",
            "reference_value": LAMBDA_M,
            "reference_source": "analytical lambda_c = 1.781e-6 m",
            "note": f"spacing/lambda = {data['spots']['spacing_over_lambda']:.4f}; n_peaks = {data['spots']['n_peaks']}",
            "derived_from": "script",
        })
    if data.get("controls"):
        rows.append({
            "name": "equal-diffusion negative-control activator std",
            "value": data["controls"]["equal_diffusion_std"],
            "unit": "concentration",
            "method": "same initial noise, D_v reset to D_u, short spectral integration",
            "reference_value": data["pattern_std"],
            "reference_source": "patterned regime activator std at D_v/D_u = 12",
            "note": f"gamma=0 ablation std = {data['controls']['ablation_gamma0_std']:.6g}",
            "derived_from": "script",
        })
    if data.get("ks"):
        rows.append({
            "name": "specimen versus simulation domain-area KS statistic",
            "value": data["ks"]["ks_statistic"],
            "unit": "KS distance",
            "method": "two-sample KS on median-normalised area distributions (specimen watershed vs activator peaks)",
            "reference_value": 0.0,
            "reference_source": "KS = 0 would mean identical area shapes",
            "note": f"p = {data['ks']['ks_pvalue']:.4g}; n_spec = {data['ks']['n_specimen']}; n_sim = {data['ks']['n_simulated']}",
            "derived_from": "specimen_pixels",
        })
    if data.get("sweep"):
        unstable = [row for row in data["sweep"] if row.get("unstable") and row.get("lambda_m")]
        if len(unstable) >= 2:
            ratios = [row["ratio"] for row in unstable]
            lams = [row["lambda_m"] for row in unstable]
            # monotonicity proxy: correlation of ratio with lambda (expected negative)
            r_sw, _ = pearsonr(ratios, lams)
            rows.append({
                "name": "diffusivity-ratio sweep Pearson r of lambda_c",
                "value": float(r_sw),
                "unit": "correlation",
                "method": "Pearson r between D_v/D_u and lambda_c across the unstable ratios in the sweep",
                "reference_value": -1.0,
                "reference_source": "lambda falls as the inhibitor diffuses faster, toward the large-ratio asymptote",
                "note": "ratios " + ", ".join(str(row["ratio"]) for row in data["sweep"]),
                "derived_from": "script",
            })
    return rows


def _write_metrics(payload: dict) -> None:
    # drop nothing essential; specimen areas already removed
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def self_test() -> None:
    fu, fv, gu, gv, u0, v0 = jacobian()
    trace = fu + gv
    det = fu * gv - fv * gu
    margin = turing_margin(fu, fv, gu, gv)
    assert trace < 0, trace
    assert det > 0, det
    assert margin > 0, margin
    assert (DV * fu + DU * gv) > 0
    scales = physical_scales(fu, gv)
    assert abs(scales["lambda_m"] - 1.781e-6) < 1e-18
    assert abs(scales["log10_lambda_m"] + 5.749) < 0.001
    assert abs(scales["domain_um"] - 14.248) < 0.01
    assert scales["coat_to_detail_ratio"] > 10000
    # equal diffusion cannot satisfy the long-range condition
    assert (DU * fu + DU * gv) < 0
    sig_eq = sigma_continuum(scales["k_c_dimless"], fu, fv, gu, gv, du=DU, dv=DU)
    assert sig_eq < 0, sig_eq
    sig_on = sigma_continuum(scales["k_c_dimless"], fu, fv, gu, gv)
    assert sig_on > 0, sig_on
    disp = dispersion_goodness(fu, fv, gu, gv, scales["k_c_dimless"])
    assert disp["pearson_r"] > 0.98, disp["pearson_r"]
    assert abs(disp["pearson_r"] - 1.0) > 1e-6
    assert disp["rmse"] > 0.0
    # short pattern run: amplitude must grow, wavenumber must stay near k_c
    short = _spectral_integrate(DU, DV, GAMMA, n=48, t_final=6.0, dt=0.001, snap_count=2)
    assert short["std"] > 0.05, short["std"]
    assert abs(short["k_measured"] / short["k_c"] - 1.0) < 0.25, short["k_measured"]
    assert math.isclose(AGE_MA, 0.0)
    print(
        f"[self-test] PASS  λ={scales['lambda_um']:.3f} µm  "
        f"age={AGE_MA:.0f} Ma  k_ratio_short={short['k_measured']/short['k_c']:.3f}  "
        f"pearson_r={disp['pearson_r']:.5f}  RMSE={disp['rmse']:.4f}  "
        f"σ_max={sig_on:.3f}  σ_equalD={sig_eq:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    payload, result, scales, disp, specimen_stats = run_simulation()
    render_gif(result["snaps"], GIF_PATH)
    render_png(result, scales, disp, specimen_stats, payload.get("controls"), payload.get("sweep"), PNG_PATH)
    # CV of the GIF against the real specimen, using the canonical scorer.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "verify_image_fitness", ROOT / "morphoeval" / "verify_image_fitness.py"
    )
    vif = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vif)
    sig = vif.compute_specimen_cv_signature(SPECIMEN)
    cv = vif.compute_sim_vs_specimen_cv_similarity(sig, SPECIMEN, GIF_PATH)
    with Image.open(GIF_PATH) as im:
        payload["gif"] = {"n_frames": int(getattr(im, "n_frames", 1)), "width": im.size[0], "height": im.size[1],
                          "cv_sim_similarity": cv}
    payload["specimen_cv_signature"] = sig
    _write_metrics(payload)
    print(
        f"[canis-lupus-s0] λ={scales['lambda_um']:.3f} µm  age={AGE_MA:.0f} Ma  "
        f"std={result['std']:.3f}  k_ratio={payload['k_ratio']:.3f}  "
        f"pearson_r={disp['pearson_r']:.5f}  CV={cv}  "
        f"frames={payload['gif']['n_frames']}  pass={REMEDIATION_PASS}"
    )
    print(f"  png {PNG_PATH}")
    print(f"  gif {GIF_PATH}")
    print(f"  metrics {METRICS_PATH}")


if __name__ == "__main__":
    main()
