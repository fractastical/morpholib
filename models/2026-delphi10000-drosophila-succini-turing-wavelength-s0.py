#!/usr/bin/env python3
"""
Drosophila succini — detail-scale Turing wavelength (slot 3172, scale s0).

Organism: Drosophila succini Hennig, 1965 (Eocene Baltic amber stem drosophilid).
Scale: 2.8 µm (exactly 2.759e-06 m, log10 = -5.559, tier 3_subcellular_um).
Age: Middle Eocene, Lutetian Baltic amber, 44 Ma.
Chronoseries: Cenozoic amber drosophilids, index 1/41. No earlier Drosophila
chronospecies in the series. Successor: Drosophila berryi (Dominican amber, 19 Ma).

This is not the primordium-scale sibling at 114 µm
(models/1965-hennig-drosophila-succini-turing-wavelength.py). That slot fits a
power law on a 114 µm stripe. Here the selected quantity is the Schnakenberg
critical wavenumber of an ocellar pigment micro-pattern whose wavelength is
2.759 µm — between a modern rhabdomere diameter (~1–2 µm) and an ommatidial
facet (~15 µm), read as the subcellular detail of the same pigment system.

Governing system (Schnakenberg 1979 activator–substrate kinetics):

    du/dt = D_u ∇²u + a - u + u² v
    dv/dt = D_v ∇²v + b - u² v

Homogeneous steady state u0 = a + b, v0 = b / u0².
Jacobian entries f_u, f_v, g_u, g_v are in `jacobian`.
Linear growth rate σ(k) is the larger eigenvalue of J - k² diag(D_u, D_v).
The critical wavenumber k_c is argmax_k σ(k), and λ_c = 2π / k_c.
D_u is solved so that λ_c equals 2.759 µm at the Eocene diffusion ratio
r = D_v / D_u = 18.72, which sits above the Turing onset r_onset ≈ 10.77.

What changes at this scale and age
-----------------------------------
* Versus the pre-Baltic acalyptrate stem (series index 0; not a named
  Drosophila chronospecies): the inhibitor/activator diffusivity ratio is
  r_stem = 8.40 < r_onset, so σ(k) < 0 at every k and no 2.8 µm pigment
  lattice exists. At 44 Ma the ratio is raised to 18.72 and the mode turns on.
* Versus successor D. berryi at 19 Ma: r is held fixed (same kinetics class)
  and both diffusivities are scaled by (λ_berryi / λ_succini)². λ_berryi is
  the primordium-scale registry ratio 98.43 µm / 114 µm applied to 2.759 µm,
  i.e. 2.382 µm. That transfer is labelled as such; it is not a new fossil
  measurement. Absolute diffusion drops; the wavelength shortens 13.7%.

Law: Turing wavenumber k_c. Cohort: chemical Turing / reaction–diffusion.
Form: stripes, spots and ocelli (Schnakenberg spots on a small retinal patch).
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

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_succini_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_succini_turing_wavelength_s0_specimen.jpg"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_succini_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_succini_turing_wavelength_s0_growth.gif"
METRICS_PATH = DATASET / "drosophila_succini_turing_wavelength_s0_metrics.json"

# Frozen gallery-card digest (same bytes as the species-gallery fallback).
SPECIMEN_SHA256 = "1a0de2972916cbfa2e2b0e21ebeb0aed9e2f4f99f11b26993c29f2c9b36464cd"

# --- slot-3172 parameters (not the 114 µm sibling's alpha/beta power law) ---
LAMBDA_M = 2.759e-06  # 2.8 µm detail scale, metres
LOG10_SIZE = -5.559
AGE_MA = 44.0
SUCCESSOR_AGE_MA = 19.0
# Registry primordium wavelengths used only as a transfer ratio (µm).
PRIMORDIUM_SUCCINI_UM = 114.0
PRIMORDIUM_BERRYI_UM = 98.43
LAMBDA_BERRYI_M = LAMBDA_M * (PRIMORDIUM_BERRYI_UM / PRIMORDIUM_SUCCINI_UM)

A_PROD = 0.062
B_PROD = 1.180
R_SUCCINI = 18.72
R_STEM = 8.40          # predecessor state, below onset
R_ABLATION = 1.0       # equal diffusivities, negative control
SEED = 3172
LENGTH_UNIT_M = LAMBDA_M  # nondimensional x is in units of the target wavelength


def jacobian(a: float = A_PROD, b: float = B_PROD) -> dict[str, float]:
    """Schnakenberg Jacobian at the homogeneous steady state."""
    u0 = a + b
    v0 = b / (u0 * u0)
    f_u = -1.0 + 2.0 * u0 * v0
    f_v = u0 * u0
    g_u = -2.0 * u0 * v0
    g_v = -(u0 * u0)
    det = f_u * g_v - f_v * g_u
    return {
        "u0": u0,
        "v0": v0,
        "f_u": f_u,
        "f_v": f_v,
        "g_u": g_u,
        "g_v": g_v,
        "trace": f_u + g_v,
        "det": det,
    }


def turing_onset_ratio(jac: dict[str, float] | None = None) -> float:
    """Smallest r = D_v/D_u with (r f_u + g_v)² > 4 r det(J)."""
    jac = jac or jacobian()
    f_u, g_v, det = jac["f_u"], jac["g_v"], jac["det"]

    def margin(r: float) -> float:
        return (r * f_u + g_v) ** 2 - 4.0 * r * det

    lo, hi = 2.0, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if margin(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return float(hi)


def growth_rate(k: float, du: float, dv: float, jac: dict[str, float] | None = None) -> float:
    """Larger eigenvalue σ(k) of J - k² diag(D_u, D_v). k is nondimensional."""
    jac = jac or jacobian()
    k2 = float(k) * float(k)
    tr = jac["trace"] - k2 * (du + dv)
    det = (jac["f_u"] - k2 * du) * (jac["g_v"] - k2 * dv) - jac["f_v"] * jac["g_u"]
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        disc = 0.0
    return float(0.5 * (tr + np.sqrt(disc)))


def k_peak(du: float, r: float, jac: dict[str, float] | None = None, k_max: float = 24.0) -> tuple[float, float]:
    """Return (k_c, σ_max) with k_c = argmax σ(k)."""
    jac = jac or jacobian()
    ks = np.linspace(0.15, k_max, 4800)
    sig = np.array([growth_rate(k, du, r * du, jac) for k in ks])
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def solve_du_for_wavelength(r: float = R_SUCCINI, lambda_m: float = LAMBDA_M) -> float:
    """D_u (nondimensional, length unit = LAMBDA_M) so that λ(k_peak) = lambda_m.

    Because k scales as 1/sqrt(D_u) at fixed r, one evaluation at D_u = 1
    fixes the diffusivity. A short correction removes the grid bias of the
    wavenumber scan.
    """
    jac = jacobian()
    k_target = 2.0 * np.pi * (LAMBDA_M / lambda_m)
    k1, _ = k_peak(1.0, r, jac)
    du = (k1 / k_target) ** 2
    for _ in range(4):
        k_now, _ = k_peak(du, r, jac)
        du *= (k_now / k_target) ** 2
    return float(du)


def textbook_k_h(du: float, r: float, jac: dict[str, float] | None = None) -> float:
    """Turing's determinant-minimising wavenumber.

    k_h² = (D_v f_u + D_u g_v) / (2 D_u D_v).
    Near onset k_h ≈ k_c; above onset the fastest-growing mode k_c shifts.
    """
    jac = jac or jacobian()
    dv = r * du
    num = dv * jac["f_u"] + du * jac["g_v"]
    den = 2.0 * du * dv
    if num <= 0.0 or den <= 0.0:
        return float("nan")
    return float(np.sqrt(num / den))


def physical_diffusivity(du_nondim: float, r: float) -> dict[str, float]:
    """Map nondimensional D onto m²/s with reaction time unit τ = 1 s."""
    ell2 = LENGTH_UNIT_M ** 2
    d_u = du_nondim * ell2
    return {"D_u_m2_s": d_u, "D_v_m2_s": r * d_u, "tau_s": 1.0}


def dispersion_numeric(
    du: float,
    r: float,
    modes: range | None = None,
    length: float = 8.0,
    n: int = 256,
    t_horizon: float = 1.2,
) -> list[dict[str, float]]:
    """Early growth of Fourier modes of the 1D nonlinear PDE vs analytic σ(k).

    Spectral Laplacian, eigenvector initial data, seed-free (deterministic).
    Modes with analytic σ > -0.25 track the linear law; strongly damped modes
    mix the fast stable eigenvalue and are reported but not used for Pearson r.
    """
    jac = jacobian()
    dv = r * du
    dx = length / n
    x = np.arange(n) * dx
    kfreq = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    dt = 0.15 * dx * dx / (2.0 * dv)
    steps = int(t_horizon / dt)
    rows: list[dict[str, float]] = []
    mode_ids = modes if modes is not None else range(1, 18)
    for nmode in mode_ids:
        k = 2.0 * np.pi * nmode / length
        s = growth_rate(k, du, dv, jac)
        k2 = k * k
        ev_u = jac["f_v"]
        ev_v = -(jac["f_u"] - k2 * du - s)
        nrm = float(np.hypot(ev_u, ev_v)) + 1e-15
        eu, ev = ev_u / nrm, ev_v / nrm
        eps = 2e-4
        u = jac["u0"] + eps * eu * np.cos(k * x)
        v = jac["v0"] + eps * ev * np.cos(k * x)

        def proj(z: np.ndarray) -> float:
            return float((2.0 / n) * np.dot(z - np.mean(z), np.cos(k * x)))

        p0 = proj(u)
        for _ in range(steps):
            lu = np.fft.ifft(-kfreq**2 * np.fft.fft(u)).real
            lv = np.fft.ifft(-kfreq**2 * np.fft.fft(v)).real
            u = u + dt * (du * lu + A_PROD - u + u * u * v)
            v = v + dt * (dv * lv + B_PROD - u * u * v)
        p1 = proj(u)
        sn = float(np.log(abs(p1 / p0)) / (steps * dt))
        rows.append({"mode": float(nmode), "k": float(k), "sigma_numeric": sn, "sigma_analytic": float(s)})
    return rows


def pearson_rmse(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    if xs.size < 3:
        return float("nan"), float("nan")
    r = float(np.corrcoef(xs, ys)[0, 1])
    rmse = float(np.sqrt(np.mean((xs - ys) ** 2)))
    return r, rmse


def parameter_sweep(du_succini: float, jac: dict[str, float] | None = None) -> list[dict[str, float]]:
    """λ versus activator diffusivity at fixed r = 18.72.

    k_c scales as 1/sqrt(D_u), so λ_c scales as sqrt(D_u) and the sweep is monotone.
    The diffusion-ratio axis (onset, stem, ablation) is reported separately.
    """
    jac = jac or jacobian()
    rows = []
    for scale in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00):
        du = du_succini * scale
        k, smax = k_peak(du, R_SUCCINI, jac)
        lam = (2.0 * np.pi / k) * LENGTH_UNIT_M
        rows.append(
            {
                "D_u_scale": float(scale),
                "k_c": k,
                "sigma_max": smax,
                "lambda_m": lam,
                "unstable": bool(smax > 0.0),
            }
        )
    return rows


def ratio_regime(du_succini: float, jac: dict[str, float] | None = None) -> list[dict[str, float]]:
    """Stability versus r at the succini diffusivity. Pattern is off below onset."""
    jac = jac or jacobian()
    rows = []
    for r in (R_ABLATION, R_STEM, 12.5, 15.0, R_SUCCINI, 28.0, 40.0):
        k, smax = k_peak(du_succini, r, jac)
        lam = (2.0 * np.pi / k) * LENGTH_UNIT_M if k > 0 else float("nan")
        rows.append({"r": float(r), "k_c": k, "sigma_max": smax, "lambda_m": lam, "unstable": bool(smax > 0.0)})
    return rows


def simulate_spot_field(
    du: float,
    r: float,
    n: int = 128,
    length: float = 3.2,
    t_final: float = 16.0,
    n_frames: int = 12,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """2D Schnakenberg spots on a retinal patch a few wavelengths across.

    Diffusion is integrated with an exponential Fourier step; the local
    reaction is forward Euler. Frames are evenly spaced in time.
    """
    jac = jacobian()
    dv = r * du
    dx = length / n
    rng = np.random.default_rng(seed)
    u = jac["u0"] + 0.02 * rng.standard_normal((n, n))
    v = jac["v0"] + 0.02 * rng.standard_normal((n, n))
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    dt = 0.05
    steps = int(t_final / dt)
    snap_at = {int(round(i * (steps - 1) / (n_frames - 1))) for i in range(n_frames)}
    frames: list[np.ndarray] = []
    eu = np.exp(-du * k2 * dt)
    ev = np.exp(-dv * k2 * dt)
    for step in range(steps):
        if step in snap_at:
            frames.append(u.copy())
        u = np.fft.ifft2(np.fft.fft2(u) * eu).real
        v = np.fft.ifft2(np.fft.fft2(v) * ev).real
        u = u + dt * (A_PROD - u + u * u * v)
        v = v + dt * (B_PROD - u * u * v)
        np.clip(u, 0.0, 8.0, out=u)
        np.clip(v, 0.0, 8.0, out=v)
    if len(frames) < n_frames:
        frames.append(u.copy())
    field = np.stack(frames[:n_frames], axis=0)
    return {"frames": field, "final": field[-1], "length": np.array(length), "dt": np.array(dt)}


def dominant_k(field: np.ndarray, length: float) -> float:
    """Radial peak of the 2D power spectrum, in nondimensional wavenumber."""
    win = field - np.mean(field)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    n = field.shape[0]
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=length / n))
    ky, kx = np.meshgrid(freq, freq, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2)
    # bin
    k_max = float(kk.max())
    bins = np.linspace(0.0, k_max, 64)
    which = np.digitize(kk.ravel(), bins) - 1
    which = np.clip(which, 0, len(bins) - 2)
    power = np.bincount(which, weights=spec.ravel(), minlength=len(bins) - 1)
    counts = np.maximum(1, np.bincount(which, minlength=len(bins) - 1))
    prof = power / counts
    # ignore the DC bin
    i = int(np.argmax(prof[1:])) + 1
    return float(0.5 * (bins[i] + bins[i + 1]) * 2.0 * np.pi)  # cycles/length * 2π = wavenumber


def amber_rgb(field: np.ndarray) -> np.ndarray:
    """Map a scalar field onto the amber palette of the gallery card, without using its pixels."""
    z = field.astype(np.float64)
    lo, hi = np.percentile(z, [2.0, 98.0])
    t = np.clip((z - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    dark = np.array([0.27, 0.15, 0.05])
    mid = np.array([0.62, 0.42, 0.16])
    lite = np.array([0.90, 0.74, 0.42])
    t3 = t[..., None]
    rgb = np.where(t3 < 0.5, dark + (mid - dark) * (t3 / 0.5), mid + (lite - mid) * ((t3 - 0.5) / 0.5))
    return np.clip(rgb, 0.0, 1.0)


def render_gif(frames: np.ndarray, path: Path, size: tuple[int, int] = (960, 720)) -> None:
    images = []
    for fr in frames:
        rgb = amber_rgb(fr)
        im = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        im = im.resize(size, Image.Resampling.LANCZOS)
        images.append(im)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def render_png(bundle: dict, specimen_path: Path, path: Path) -> None:
    """Six-panel comparison: specimen, spots, dispersion, sweep, chronoseries, ablation."""
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), dpi=120)
    spec = np.asarray(Image.open(specimen_path).convert("RGB"))
    axes[0, 0].imshow(spec)
    axes[0, 0].set_title("Gallery card (fallback specimen)")
    axes[0, 0].axis("off")

    final = bundle["final_field"]
    axes[0, 1].imshow(amber_rgb(final), interpolation="bilinear")
    axes[0, 1].set_title(f"Schnakenberg u, λ target {LAMBDA_M*1e6:.3f} µm")
    axes[0, 1].axis("off")

    rows = bundle["dispersion"]
    k = np.array([r["k"] for r in rows])
    sa = np.array([r["sigma_analytic"] for r in rows])
    sn = np.array([r["sigma_numeric"] for r in rows])
    ks = np.linspace(0.2, float(k.max()), 400)
    du, r = bundle["du"], bundle["r"]
    curve = [growth_rate(kk, du, r * du) for kk in ks]
    axes[0, 2].plot(ks, curve, color="#8a4b08", label="analytic σ(k)")
    axes[0, 2].scatter(k, sn, s=22, color="#1f4e79", zorder=3, label="PDE modes")
    axes[0, 2].axvline(2 * np.pi, color="#555", ls="--", lw=0.8, label="k = 2π")
    axes[0, 2].axhline(0.0, color="#888", lw=0.6)
    axes[0, 2].set_xlabel("k (per target wavelength)")
    axes[0, 2].set_ylabel("σ")
    axes[0, 2].set_title(f"Dispersion, band r = {bundle['pearson_band']:.5f}")
    axes[0, 2].legend(fontsize=7, frameon=False)

    sweep = bundle["sweep"]
    scales = [s["D_u_scale"] for s in sweep]
    lam_um = [s["lambda_m"] * 1e6 for s in sweep]
    axes[1, 0].plot(scales, lam_um, color="#214e34", marker="o")
    axes[1, 0].axhline(LAMBDA_M * 1e6, color="#8a4b08", ls=":", lw=0.8)
    axes[1, 0].set_xlabel("D_u scale at r = 18.72")
    axes[1, 0].set_ylabel("λ(k_c) (µm)")
    axes[1, 0].set_title("λ increases with sqrt(D_u)")

    labels = ["stem\nr=8.40\n>44 Ma", "succini\nr=18.72\n44 Ma", "berryi\nD scaled\n19 Ma"]
    vals = [
        0.0,
        bundle["lambda_succini_m"] * 1e6,
        bundle["lambda_berryi_m"] * 1e6,
    ]
    axes[1, 1].bar([0, 1, 2], vals, color=["#b2182b", "#8a4b08", "#2166ac"])
    axes[1, 1].set_xticks([0, 1, 2], labels, fontsize=8)
    axes[1, 1].set_ylabel("detail wavelength (µm)")
    axes[1, 1].set_title("Chronoseries at 2.8 µm scale")

    k_ab = np.linspace(0.2, 14.0, 300)
    axes[1, 2].plot(k_ab, [growth_rate(kk, du, R_SUCCINI * du) for kk in k_ab], label="succini r=18.72")
    axes[1, 2].plot(k_ab, [growth_rate(kk, du, R_STEM * du) for kk in k_ab], label="stem r=8.40")
    axes[1, 2].plot(k_ab, [growth_rate(kk, du, R_ABLATION * du) for kk in k_ab], label="ablation r=1")
    axes[1, 2].axhline(0.0, color="#888", lw=0.6)
    axes[1, 2].set_xlabel("k")
    axes[1, 2].set_ylabel("σ")
    axes[1, 2].set_title("Negative controls")
    axes[1, 2].legend(fontsize=7, frameon=False)

    fig.suptitle(
        "Drosophila succini, 44 Ma — Turing k_c at 2.8 µm (2.759e-06 m), slot 3172",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def specimen_signature(path: Path) -> dict[str, float]:
    """Border-contrast segmentation plus structure-tensor coherency of the gallery card."""
    from scipy import ndimage as ndi

    rgb = np.asarray(Image.open(path).convert("RGB").resize((160, 120), Image.Resampling.BILINEAR), dtype=np.float64)
    rgb /= 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    th = float(np.percentile(contrast, 45.0))
    mask = contrast >= max(th, 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_fill_holes(mask)
    eroded = ndi.binary_erosion(mask)
    perim = float(np.sum(mask ^ eroded)) + 1e-6
    area = float(np.sum(mask))
    circ = float(np.clip(4.0 * np.pi * area / (perim**2), 0.0, 1.0))
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    coherency = float(np.mean(coh_map[mask])) if area > 20 else float(np.mean(coh_map))
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    return {
        "foreground_fraction": float(np.mean(mask)),
        "circularity": circ,
        "coherency": coherency,
        "orientation_samples": ang[mask].ravel()[::20],
    }


def field_orientation_samples(field: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    g = field.astype(np.float64)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    return ang.ravel()[::15]


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)


def run_model(render: bool = True) -> dict:
    jac = jacobian()
    r_onset = turing_onset_ratio(jac)
    du = solve_du_for_wavelength(R_SUCCINI, LAMBDA_M)
    k_c, sigma_max = k_peak(du, R_SUCCINI, jac)
    lam = (2.0 * np.pi / k_c) * LENGTH_UNIT_M
    k_h = textbook_k_h(du, R_SUCCINI, jac)
    # successor: same r, diffusivities scaled so λ matches the transferred target
    scale = (LAMBDA_BERRYI_M / LAMBDA_M) ** 2
    du_b = du * scale
    k_b, sig_b = k_peak(du_b, R_SUCCINI, jac)
    lam_b = (2.0 * np.pi / k_b) * LENGTH_UNIT_M
    k_stem, sig_stem = k_peak(du, R_STEM, jac)
    k_abl, sig_abl = k_peak(du, R_ABLATION, jac)
    disp = dispersion_numeric(du, R_SUCCINI)
    band = [row for row in disp if row["sigma_analytic"] > -0.25]
    r_band, rmse_band = pearson_rmse(
        np.array([row["sigma_numeric"] for row in band]),
        np.array([row["sigma_analytic"] for row in band]),
    )
    r_all, rmse_all = pearson_rmse(
        np.array([row["sigma_numeric"] for row in disp]),
        np.array([row["sigma_analytic"] for row in disp]),
    )
    sweep = parameter_sweep(du, jac)
    regime = ratio_regime(du, jac)
    lams = [row["lambda_m"] for row in sweep]
    sweep_monotone = all(lams[i] < lams[i + 1] for i in range(len(lams) - 1))
    field = simulate_spot_field(du, R_SUCCINI)
    k_nl = dominant_k(field["final"], float(field["length"]))
    lam_nl = (2.0 * np.pi / k_nl) * LENGTH_UNIT_M if k_nl > 0 else float("nan")
    spec_sig = specimen_signature(SPECIMEN)
    ks = ks_statistic(spec_sig["orientation_samples"], field_orientation_samples(field["final"]))
    # field coherency for a non-circular comparison
    from scipy import ndimage as ndi

    g = field["final"].astype(np.float64)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=1.6)
    jyy = ndi.gaussian_filter(gy * gy, sigma=1.6)
    jxy = ndi.gaussian_filter(gx * gy, sigma=1.6)
    coh_sim = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    diff = physical_diffusivity(du, R_SUCCINI)
    bundle = {
        "slot": 3172,
        "sim_id": "delphi10000-drosophila-succini-turing-wavelength-s0",
        "organism": "Drosophila succini",
        "age_ma": AGE_MA,
        "lambda_target_m": LAMBDA_M,
        "lambda_succini_m": lam,
        "lambda_nonlinear_m": lam_nl,
        "lambda_berryi_m": lam_b,
        "lambda_berryi_target_m": LAMBDA_BERRYI_M,
        "k_c": k_c,
        "k_h_textbook": k_h,
        "k_nonlinear": k_nl,
        "sigma_max": sigma_max,
        "du": du,
        "r": R_SUCCINI,
        "r_onset": r_onset,
        "r_stem": R_STEM,
        "sigma_stem": sig_stem,
        "sigma_ablation": sig_abl,
        "sigma_berryi": sig_b,
        "du_berryi": du_b,
        "D_u_m2_s": diff["D_u_m2_s"],
        "D_v_m2_s": diff["D_v_m2_s"],
        "pearson_band": r_band,
        "rmse_band": rmse_band,
        "pearson_all_modes": r_all,
        "rmse_all_modes": rmse_all,
        "n_band": len(band),
        "sweep_monotone_above_onset": sweep_monotone,
        "sweep": sweep,
        "regime": regime,
        "dispersion": disp,
        "specimen_foreground_fraction": spec_sig["foreground_fraction"],
        "specimen_circularity": spec_sig["circularity"],
        "specimen_coherency": spec_sig["coherency"],
        "simulation_coherency": coh_sim,
        "ks_orientation": ks,
        "jacobian": {k: float(v) for k, v in jac.items()},
        "final_field": field["final"],
        "frames": field["frames"],
    }
    if render:
        render_gif(field["frames"], GIF_PATH)
        render_png(bundle, SPECIMEN, PNG_PATH)
        with Image.open(GIF_PATH) as im:
            bundle["gif_n_frames"] = int(getattr(im, "n_frames", 1))
            bundle["gif_size"] = [int(im.size[0]), int(im.size[1])]
        serializable = {k: v for k, v in bundle.items() if k not in ("final_field", "frames", "dispersion")}
        serializable["dispersion"] = disp
        serializable["sweep"] = sweep
        serializable["regime"] = regime
        METRICS_PATH.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    return bundle


def measure_morphoeval() -> list[dict]:
    """Sidecar morphometrics. Reference values are comparators, not copies of the measurement."""
    bundle = run_model(render=False)
    # recompute is deterministic; avoid a second 2D solve when metrics json exists and PNG is fresh
    return [
        {
            "name": "succini critical wavelength from argmax sigma(k)",
            "value": round(bundle["lambda_succini_m"], 10),
            "unit": "m",
            "method": "k_c = argmax sigma(k); lambda = 2 pi / k_c * 2.759e-6 m",
            "reference_value": LAMBDA_M,
            "source": "slot scale 2.8 µm = 2.759e-06 m",
            "note": "linear prediction; reference is the assigned scale, value is the solved peak",
            "derived_from": "script",
        },
        {
            "name": "nonlinear spot-field dominant wavelength",
            "value": round(float(bundle["lambda_nonlinear_m"]), 10),
            "unit": "m",
            "method": "radial FFT peak of the saturated 2D Schnakenberg field",
            "reference_value": round(bundle["lambda_succini_m"], 10),
            "source": "linear k_c of the same parameters",
            "note": "finite-domain FFT binning shifts this off the linear peak",
            "derived_from": "script",
        },
        {
            "name": "Turing-band Pearson r of PDE growth vs analytic sigma",
            "value": round(bundle["pearson_band"], 6),
            "unit": "correlation",
            "method": "modes with analytic sigma > -0.25; spectral 1D PDE, T=1.2",
            "reference_value": 0.98,
            "source": "pre-registered band around the unstable wavenumber",
            "note": f"RMSE = {bundle['rmse_band']:.6g}; full-spectrum r = {bundle['pearson_all_modes']:.4f} (stable modes mix the fast eigenvalue)",
            "derived_from": "script",
        },
        {
            "name": "Turing-band RMSE of sigma",
            "value": round(bundle["rmse_band"], 6),
            "unit": "1/time",
            "method": "sqrt mean square of numeric minus analytic growth rates on the band",
            "reference_value": 0.05,
            "source": "dispersion validation tolerance",
            "note": "value is the residual, reference is the tolerance, they are not the same quantity",
            "derived_from": "script",
        },
        {
            "name": "stem predecessor sigma_max at r=8.40",
            "value": round(bundle["sigma_stem"], 6),
            "unit": "1/time",
            "method": "max_k sigma(k) at r_stem below onset, succini D_u",
            "reference_value": 0.0,
            "source": "negative control: pre-Baltic stem, series index 0",
            "note": "negative value means the 2.8 µm ocellar mode is absent before 44 Ma",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusivity ablation sigma_max",
            "value": round(bundle["sigma_ablation"], 6),
            "unit": "1/time",
            "method": "max_k sigma(k) at r=1",
            "reference_value": 0.0,
            "source": "Turing necessary condition D_v > D_u fails",
            "note": "ablation control",
            "derived_from": "script",
        },
        {
            "name": "berryi transferred detail wavelength",
            "value": round(bundle["lambda_berryi_m"], 10),
            "unit": "m",
            "method": "same r, D scaled by (98.43/114)^2; k_c remeasured",
            "reference_value": round(LAMBDA_BERRYI_M, 10),
            "source": "transfer of registry primordium ratio onto the 2.759 µm detail scale, not a new fossil measurement",
            "note": "successor Drosophila berryi, 19 Ma",
            "derived_from": "script",
        },
        {
            "name": "specimen foreground fraction",
            "value": round(bundle["specimen_foreground_fraction"], 4),
            "unit": "area fraction",
            "method": "border-median contrast mask on the gallery card, 160x120",
            "reference_value": 0.62,
            "source": "segmentation of datasets/.../drosophila_succini_turing_wavelength_s0_specimen.jpg",
            "note": "reference 0.62 is a generic filled-body prior, not this card",
            "derived_from": "output",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(bundle["specimen_coherency"], 4),
            "unit": "coherency",
            "method": "mean (λ1-λ2)/(λ1+λ2) on the contrast mask",
            "reference_value": round(bundle["simulation_coherency"], 4),
            "source": "gallery card vs Schnakenberg field",
            "note": "specimen value and simulation reference differ; KS orientation is separate",
            "derived_from": "output",
        },
        {
            "name": "KS distance of gradient orientations specimen vs simulation",
            "value": round(bundle["ks_orientation"], 4),
            "unit": "KS statistic",
            "method": "two-sample KS on Sobel orientation angles",
            "reference_value": 0.0,
            "source": "identical images would score 0",
            "note": "non-circular specimen metric",
            "derived_from": "output",
        },
        {
            "name": "specimen contour circularity",
            "value": round(bundle["specimen_circularity"], 4),
            "unit": "4πA/P²",
            "method": "mask perimeter from binary erosion",
            "reference_value": 0.80,
            "source": "a circular body prior; the amber-spiral card is not a body outline",
            "note": "low circularity is expected for a full-bleed spiral card",
            "derived_from": "output",
        },
    ]


def self_test() -> None:
    assert SPECIMEN.exists(), "missing specimen"
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    assert digest == SPECIMEN_SHA256, digest
    bundle = run_model(render=True)
    rel = abs(bundle["lambda_succini_m"] - LAMBDA_M) / LAMBDA_M
    assert rel < 0.01, rel
    assert bundle["pearson_band"] > 0.98, bundle["pearson_band"]
    assert bundle["rmse_band"] < 0.01, bundle["rmse_band"]
    assert bundle["sigma_stem"] < 0.0, bundle["sigma_stem"]
    assert bundle["sigma_ablation"] < 0.0, bundle["sigma_ablation"]
    assert bundle["r_onset"] < R_SUCCINI
    assert R_STEM < bundle["r_onset"]
    assert bundle["lambda_berryi_m"] < bundle["lambda_succini_m"]
    assert abs(bundle["lambda_berryi_m"] - LAMBDA_BERRYI_M) / LAMBDA_BERRYI_M < 0.03
    assert bundle["sweep_monotone_above_onset"]
    assert bundle["jacobian"]["trace"] < 0.0
    assert bundle["jacobian"]["det"] > 0.0
    assert PNG_PATH.exists() and GIF_PATH.exists()
    with Image.open(GIF_PATH) as im:
        assert im.n_frames >= 10, im.n_frames
        assert min(im.size) >= 480, im.size
        im.seek(0)
        first = np.asarray(im.convert("RGB"), dtype=np.float64)
        im.seek(im.n_frames - 1)
        last = np.asarray(im.convert("RGB"), dtype=np.float64)
    assert float(np.mean(np.abs(last - first))) > 0.003
    checks = {
        "lambda_rel_err": rel,
        "pearson_band": bundle["pearson_band"],
        "rmse_band": bundle["rmse_band"],
        "sigma_stem": bundle["sigma_stem"],
        "sigma_ablation": bundle["sigma_ablation"],
        "lambda_berryi_um": bundle["lambda_berryi_m"] * 1e6,
        "lambda_nonlinear_um": bundle["lambda_nonlinear_m"] * 1e6,
        "ks_orientation": bundle["ks_orientation"],
        "gif_frames": bundle["gif_n_frames"],
        "gif_size": bundle["gif_size"],
        "sweep_monotone": bundle["sweep_monotone_above_onset"],
    }
    print(json.dumps({"self_test": "PASS", **checks}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    bundle = run_model(render=True)
    summary = {
        "lambda_um": bundle["lambda_succini_m"] * 1e6,
        "pearson_band": bundle["pearson_band"],
        "rmse_band": bundle["rmse_band"],
        "sigma_stem": bundle["sigma_stem"],
        "sigma_ablation": bundle["sigma_ablation"],
        "ks_orientation": bundle["ks_orientation"],
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "png": str(PNG_PATH.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
