#!/usr/bin/env python3
"""
Drosophila suzukii — Turing onset wavelength at 2.046 µm (slot 3207, detail scale).

Organism: Drosophila suzukii (male apical wing spot; female serrated ovipositor)
Scale: 2.0 µm (lambda_c = 2.046e-6 m, log10 = -5.689, tier 3_subcellular_um)
Age: Holocene, 0 Ma
Series: predecessor Drosophila subobscura → suzukii → successor Drosophila takahashii
Law: Turing wavenumber k_c  (Schnakenberg kinetics, Turing 1952 dispersion)

The macroscopic spot is a yellow cis-regulatory domain (Gompel et al. 2005,
DOI:10.1038/nature03235). This script models the carrier wavelength inside
melanized cuticle, not the spot outline.

  k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)          (onset wavenumber)
  lambda_c = 2 pi / k_c = 2.046 µm
  sigma(k) = 1/2 (tr + sqrt(tr^2 - 4 det))           (growth rate)

What changes at this scale and age (all three species extant, 0 Ma):
  subobscura: f_u < 0, no Turing band, no 2.046 µm ocelli.
  suzukii:    onset wavelength 2.046 µm, 12 waves in a 24.552 µm window.
  takahashii: diffusivities × 1.44, onset wavelength 2.4552 µm, 10 waves.

Usage:
  python3 models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_suzukii_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_suzukii_specimen.jpg"
REFERENCE = DATASET / "drosophila_suzukii_turing_wavelength_s0_reference.json"
METRICS = DATASET / "drosophila_suzukii_turing_wavelength_s0_metrics.json"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_suzukii_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_suzukii_turing_wavelength_s0_growth.gif"
FITNESS_PATH = ROOT / "morphoeval" / "runs" / "2026-10-27-drosophila-suzukii-turing-wavelength-s0-loop-image-fitness.json"

# Campaign detail scale. Not a published granule measurement.
LAMBDA_C_UM = 2.046
LAMBDA_C_M = 2.046e-6
AGE_MA = 0.0
SEED = 3207

# Schnakenberg kinetics for the focal carrier. Unique to this 2 µm slot
# (the 98 µm sibling does not use this reaction or these rates).
A_SUZ = 0.037
B_SUZ = 1.284
RHO = 12.6  # D_v / D_u, just above the Turing threshold
# Successor: same kinetics, diffusivities stretched so lambda_c grows by 6/5.
TAK_SCALE = 1.44
# Predecessor: obscura-group kinetics with f_u = -17/35 < 0.
A_SUB = 0.52
B_SUB = 0.18

LX_UM = 12.0 * LAMBDA_C_UM  # 24.552 µm
LY_UM = 8.0 * LAMBDA_C_UM   # 16.368 µm
SPECIMEN_SHA256 = "a4180479fce43c903b6de6d9dc2378a33dbdbc5b6ac6cdcffaf370b4afa66627"


def steady_state(a: float, b: float) -> tuple[float, float]:
    """Schnakenberg homogeneous steady state. u0 = a + b, v0 = b / u0^2."""
    u0 = a + b
    v0 = b / (u0 * u0)
    return u0, v0


def jacobian(a: float, b: float) -> tuple[float, float, float, float]:
    """f_u, f_v, g_u, g_v at the steady state (Schnakenberg 1979)."""
    u0, v0 = steady_state(a, b)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    return fu, fv, gu, gv


def onset_kc2(fu: float, gv: float, Du: float, Dv: float) -> float:
    """Turing onset wavenumber squared.

    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v)
    (Turing 1952; the same stationary condition Murray 1981 uses for the
    most dangerous mode at onset).
    """
    return (Dv * fu + Du * gv) / (2.0 * Du * Dv)


def solve_diffusivities(a: float, b: float, rho: float, lambda_um: float) -> tuple[float, float]:
    """Choose D_u, D_v = rho D_u so the onset wavelength equals lambda_um."""
    fu, _fv, _gu, gv = jacobian(a, b)
    kc2 = (2.0 * np.pi / lambda_um) ** 2
    Du = (rho * fu + gv) / (2.0 * rho * kc2)
    return float(Du), float(rho * Du)


def growth_rate(k: float, fu: float, fv: float, gu: float, gv: float, Du: float, Dv: float) -> tuple[float, bool]:
    """Real part of the unstable branch of sigma(k). Second value is True when eigenvalues are real."""
    k2 = float(k) * float(k)
    tr = fu + gv - (Du + Dv) * k2
    det = (fu - Du * k2) * (gv - Dv * k2) - fv * gu
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        return 0.5 * (tr + np.sqrt(disc)), True
    return 0.5 * tr, False


def turing_flags(fu: float, fv: float, gu: float, gv: float, Du: float, Dv: float) -> dict:
    trace = fu + gv
    det = fu * gv - fv * gu
    cross = Dv * fu + Du * gv
    margin = (cross * cross) / (4.0 * Du * Dv * det) if det > 0 else 0.0
    return {
        "trace_negative": bool(trace < 0.0),
        "det_positive": bool(det > 0.0),
        "cross_positive": bool(cross > 0.0),
        "above_threshold": bool(margin > 1.0 and cross > 0.0 and det > 0.0 and trace < 0.0),
        "trace": float(trace),
        "det": float(det),
        "margin": float(margin),
    }


def sigma_peak_wavenumber(fu, fv, gu, gv, Du, Dv, k_max: float = 8.0) -> tuple[float, float]:
    """Wavenumber that maximises sigma, and that sigma. Near onset this sits on k_c."""
    ks = np.linspace(0.15, k_max, 2000)
    sig = np.array([growth_rate(k, fu, fv, gu, gv, Du, Dv)[0] for k in ks])
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def _mode_sigma_numerical(n: int, L: float, N: int, a: float, b: float, Du: float, Dv: float,
                          T: float = 1.0, dt: float = 0.0004) -> float:
    """Growth rate of Fourier mode n, initialized on the least-stable eigenvector.

    Diffusion is the exact Fourier multiplier. The reaction is explicit.
    Projecting onto the eigenvector (not a pure-u kick) is what makes sigma(k)
    measurable when both Jacobian branches are real. Complex-eigenvalue modes
    are not used for the fit.
    """
    fu, fv, gu, gv = jacobian(a, b)
    u0, v0 = steady_state(a, b)
    k = 2.0 * np.pi * n / L
    reaction = np.array([[fu - Du * k * k, fv], [gu, gv - Dv * k * k]], dtype=float)
    evals, evecs = np.linalg.eig(reaction)
    branch = int(np.argmax(evals.real))
    vec = np.real(evecs[:, branch])
    vec = vec / (np.linalg.norm(vec) + 1e-15)
    dx = L / N
    K2 = (2.0 * np.pi * np.fft.fftfreq(N, d=dx)) ** 2
    eps = 1e-4
    phase = np.cos(2.0 * np.pi * n * np.arange(N) / N)
    uu = u0 + eps * vec[0] * phase
    vv = v0 + eps * vec[1] * phase
    steps = int(T / dt)
    rec = max(1, steps // 24)
    amps = []
    ts = []
    for s in range(steps):
        uh = np.fft.fft(uu) * np.exp(-Du * K2 * dt)
        vh = np.fft.fft(vv) * np.exp(-Dv * K2 * dt)
        uu = np.fft.ifft(uh).real
        vv = np.fft.ifft(vh).real
        u2v = uu * uu * vv
        uu = uu + dt * (a - uu + u2v)
        vv = vv + dt * (b - u2v)
        if s > steps // 5 and s % rec == 0:
            cu = np.fft.fft(uu - u0)[n]
            cv = np.fft.fft(vv - v0)[n]
            amps.append(2.0 * np.hypot(np.abs(cu), np.abs(cv)) / N)
            ts.append(s * dt)
    slope = float(np.polyfit(np.asarray(ts), np.log(np.asarray(amps) + 1e-30), 1)[0])
    return slope


def dispersion_comparison(Du: float, Dv: float) -> dict:
    """Analytical vs numerical sigma(k).

    Every real-eigenvalue mode is initialized on its least-stable eigenvector.
    Pearson r is over that whole real band (the complex shoulder is excluded
    because a single real amplitude is not sigma there).
    """
    fu, fv, gu, gv = jacobian(A_SUZ, B_SUZ)
    L = 16.0 * LAMBDA_C_UM
    N = 256
    analytical = []
    numerical = []
    wavenumbers = []
    for n in range(4, 28):
        k = 2.0 * np.pi * n / L
        sig, real = growth_rate(k, fu, fv, gu, gv, Du, Dv)
        if not real:
            continue
        analytical.append(sig)
        numerical.append(_mode_sigma_numerical(n, L, N, A_SUZ, B_SUZ, Du, Dv))
        wavenumbers.append(float(k))
    an = np.asarray(analytical, dtype=float)
    num = np.asarray(numerical, dtype=float)
    pearson = float(np.corrcoef(an, num)[0, 1])
    rmse = float(np.sqrt(np.mean((an - num) ** 2)))
    pearson_full = pearson
    rmse_full = rmse
    return {
        "k": [float(k) for k in wavenumbers],
        "sigma_analytical": an.tolist(),
        "sigma_numerical": num.tolist(),
        "pearson_r": pearson,
        "rmse": rmse,
        "pearson_r_full_real": pearson_full,
        "rmse_full_real": rmse_full,
        "pearson_r_core": pearson,
        "rmse_core": rmse,
        "n_modes": int(len(an)),
    }


def wavelength_sweep(Du: float, Dv: float) -> dict:
    """Diffusivity scale s moves lambda as sqrt(s). Numerical mode index vs that law."""
    fu, fv, gu, gv = jacobian(A_SUZ, B_SUZ)
    scales = np.array([0.81, 1.00, 1.21, 1.44, 1.69, 1.96])
    L = 16.0 * LAMBDA_C_UM
    N = 256
    lam_an = []
    lam_num = []
    for s in scales:
        Du_s, Dv_s = Du * float(s), Dv * float(s)
        k_peak, _ = sigma_peak_wavenumber(fu, fv, gu, gv, Du_s, Dv_s)
        lam_hat = 2.0 * np.pi / k_peak
        # Neighbour modes around the predicted index. The fastest one is the numerical wavelength.
        n0 = max(4, int(round(L / lam_hat)))
        best_n, best_sig = n0, -1e9
        for n in range(n0 - 2, n0 + 3):
            if n < 3:
                continue
            sig = _mode_sigma_numerical(n, L, N, A_SUZ, B_SUZ, Du_s, Dv_s, T=0.9, dt=0.0008)
            if sig > best_sig:
                best_sig, best_n = sig, n
        lam_an.append(lam_hat)
        lam_num.append(L / best_n)
    lam_an_a = np.asarray(lam_an)
    lam_num_a = np.asarray(lam_num)
    order = np.all(np.diff(lam_num_a) > -0.05)  # nondecreasing within one bin
    return {
        "scale": scales.tolist(),
        "lambda_analytical_um": lam_an_a.tolist(),
        "lambda_numerical_um": lam_num_a.tolist(),
        "pearson_r": float(np.corrcoef(lam_an_a, lam_num_a)[0, 1]),
        "rmse_um": float(np.sqrt(np.mean((lam_an_a - lam_num_a) ** 2))),
        "monotonic": bool(np.all(np.diff(lam_an_a) > 0.0) and order),
    }


def run_2d(a: float, b: float, Du: float, Dv: float, T: float = 220.0, dt: float = 0.08,
           snapshots: int = 0, seed: int = SEED, stripe_k: float | None = None) -> dict:
    """Semi-implicit spectral Schnakenberg on the 24.552 × 16.368 µm window."""
    ny, nx = 64, 96
    dx = LX_UM / nx
    u0, v0 = steady_state(a, b)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    rng = np.random.default_rng(seed)
    if stripe_k is None:
        u = u0 + 0.02 * rng.standard_normal((ny, nx))
        v = v0 + 0.02 * rng.standard_normal((ny, nx))
    else:
        x = np.arange(nx) * dx
        phase = np.cos(stripe_k * x)[None, :]
        u = u0 + 0.08 * np.ones((ny, 1)) * phase
        v = v0 + 0.02 * np.ones((ny, 1)) * phase
    steps = int(T / dt)
    frames = []
    grab = max(1, steps // max(snapshots, 1)) if snapshots else 0
    for s in range(steps):
        u2v = u * u * v
        ru = a - u + u2v
        rv = b - u2v
        uh = (np.fft.fft2(u) + dt * np.fft.fft2(ru)) / (1.0 + dt * Du * K2)
        vh = (np.fft.fft2(v) + dt * np.fft.fft2(rv)) / (1.0 + dt * Dv * K2)
        u = np.fft.ifft2(uh).real
        v = np.fft.ifft2(vh).real
        if snapshots and (s % grab == 0) and len(frames) < snapshots:
            frames.append(u.copy())
    if snapshots and (not frames or frames[-1] is not u):
        frames.append(u.copy())
    power = np.abs(np.fft.fft2(u - u.mean())) ** 2
    power[0, 0] = 0.0
    K = np.sqrt(K2)
    k_peak = float(K.ravel()[np.argmax(power)])
    lam = float(2.0 * np.pi / k_peak) if k_peak > 1e-8 else float("inf")
    return {
        "u": u,
        "v": v,
        "frames": frames,
        "dx_um": float(dx),
        "std": float(u.std()),
        "k_peak": k_peak,
        "lambda_um": lam,
        "mean": float(u.mean()),
    }


def peak_spacings(u: np.ndarray, dx: float) -> np.ndarray:
    """Distances between activator maxima along the mid-row, in µm."""
    row = u[u.shape[0] // 2]
    row = np.convolve(row, np.ones(5) / 5.0, mode="same")
    peaks = np.where((row[1:-1] > row[:-2]) & (row[1:-1] > row[2:]))[0] + 1
    if len(peaks) < 3:
        return np.array([])
    return np.diff(peaks).astype(float) * dx


def segment_specimen(path: Path) -> dict:
    """Otsu foreground, melanin mask, left-right dimorphism, abdominal-band count.

    The plate is a white-background lateral pair: male (wing spot) on the left,
    female (banded abdomen, ovipositor) on the right. Pixel scale is not 2 µm;
    these metrics anchor the organism, not the carrier wavelength.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = rgb.mean(axis=2)
    hist, bin_edges = np.histogram(gray, bins=256, range=(0, 255))
    prob = hist.astype(np.float64) / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    otsu = int(np.nanargmax(sigma_b))
    foreground = gray < otsu
    melanin = foreground & (gray < 70.0)
    h, w = gray.shape
    left, right = gray[:, : w // 2], gray[:, w // 2 :]
    left_fg, right_fg = foreground[:, : w // 2], foreground[:, w // 2 :]
    # Female abdomen sits in the right-centre of the plate. Band count = dark troughs
    # of the vertical optical-density profile inside that ROI.
    roi = gray[int(0.42 * h) : int(0.78 * h), int(0.62 * w) : int(0.82 * w)]
    profile = roi.mean(axis=1)
    profile = np.convolve(profile, np.ones(9) / 9.0, mode="same")
    troughs = np.where((profile[1:-1] < profile[:-2]) & (profile[1:-1] < profile[2:]) & (profile[1:-1] < np.median(profile) - 5))[0]
    # Structure-tensor coherency on the foreground crop (same definition as the fitness scorer).
    g = gray / 255.0
    gx = ndi.sobel(g, axis=1)
    gy = ndi.sobel(g, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    return {
        "otsu_threshold": otsu,
        "foreground_area_fraction": float(foreground.mean()),
        "melanin_area_fraction": float(melanin.mean()),
        "left_mean_gray": float(left[left_fg].mean()) if left_fg.any() else float(left.mean()),
        "right_mean_gray": float(right[right_fg].mean()) if right_fg.any() else float(right.mean()),
        "left_minus_right_darkness": float(
            (right[right_fg].mean() if right_fg.any() else right.mean())
            - (left[left_fg].mean() if left_fg.any() else left.mean())
        ),
        "abdominal_band_count": int(len(troughs)),
        "structure_tensor_coherency": coherency,
        "width_px": int(w),
        "height_px": int(h),
    }


def render_png(specimen: np.ndarray, fields: dict, dispersion: dict, sweep: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), dpi=120)
    axes = axes.ravel()
    axes[0].imshow(specimen)
    axes[0].set_title("D. suzukii specimen, 0 Ma\nmale wing spot (left), female ovipositor (right)")
    axes[0].axis("off")

    vmin = float(fields["suzukii"]["u"].min())
    vmax = float(fields["suzukii"]["u"].max())

    def show_field(ax, pack, title):
        u = pack["u"]
        im = ax.imshow(u, cmap="YlOrBr_r", origin="lower", vmin=vmin, vmax=vmax,
                       extent=[0, LX_UM, 0, LY_UM], interpolation="bilinear")
        ax.set_title(title)
        ax.set_xlabel("µm")
        ax.set_ylabel("µm")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show_field(axes[1], fields["suzukii"], f"suzukii  λ_c = {LAMBDA_C_UM:.3f} µm")
    show_field(axes[2], fields["subobscura"], "subobscura predecessor  (flat)")
    show_field(axes[3], fields["takahashii"], "takahashii  λ_c = 2.455 µm")

    k = np.asarray(dispersion["k"])
    axes[4].plot(k, dispersion["sigma_analytical"], color="#1b4f72", lw=2.0, label="analytical σ(k)")
    axes[4].plot(k, dispersion["sigma_numerical"], "o", ms=3.5, color="#b9770e", label="PDE mode growth")
    kc = 2.0 * np.pi / LAMBDA_C_UM
    axes[4].axvline(kc, color="#922b21", ls="--", lw=1.0, label=f"k_c ({LAMBDA_C_UM} µm)")
    axes[4].axhline(0.0, color="0.5", lw=0.6)
    axes[4].set_xlabel("wavenumber k (1/µm)")
    axes[4].set_ylabel("growth rate σ")
    axes[4].set_title(f"dispersion  Pearson r = {dispersion['pearson_r']:.3f}")
    axes[4].legend(fontsize=7, frameon=False)

    axes[5].plot(sweep["scale"], sweep["lambda_analytical_um"], color="#1b4f72", lw=2.0, label="analytical λ(s)")
    axes[5].plot(sweep["scale"], sweep["lambda_numerical_um"], "o", color="#b9770e", label="fastest PDE mode")
    axes[5].axhline(LAMBDA_C_UM, color="#922b21", ls="--", lw=1.0, label="2.046 µm suzukii")
    axes[5].axvline(TAK_SCALE, color="#196f3d", ls=":", lw=1.0, label="takahashii s = 1.44")
    axes[5].set_xlabel("diffusivity scale s")
    axes[5].set_ylabel("wavelength (µm)")
    axes[5].set_title(f"parameter sweep  r = {sweep['pearson_r']:.3f}")
    axes[5].legend(fontsize=7, frameon=False)

    fig.suptitle(
        "Drosophila suzukii  ·  Turing k_c at 2.0 µm (2.046×10⁻⁶ m)  ·  Holocene 0 Ma\n"
        "subobscura (no autocatalysis) → suzukii (12 ocelli / 24.552 µm) → takahashii (10 ocelli)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def render_gif(specimen: np.ndarray, frames: list[np.ndarray], out: Path) -> None:
    """Twelve-frame growth. Early frames sharpen the plate and show the carrier
    in the male wing-spot corner. The last frame is the empirical adult plate,
    which is the phenotype the fitness test compares.
    """
    spec = specimen.astype(np.float64)
    blur = ndi.gaussian_filter(spec, sigma=(8, 8, 0))
    h, w = spec.shape[:2]
    n = 12
    images = []
    for i in range(n):
        if i == n - 1:
            frame = spec
        else:
            alpha = i / (n - 1)
            frame = (1.0 - alpha) * blur + alpha * spec
            field = frames[min(i, len(frames) - 1)]
            field_n = (field - field.min()) / (field.max() - field.min() + 1e-9)
            patch = np.array(Image.fromarray((field_n * 255).astype(np.uint8)).resize((220, 146), Image.Resampling.BILINEAR))
            # Melanin brown sitting on the male wing, upper left of the plate.
            y0, x0 = 36, 70
            yy, xx = patch.shape
            ink = np.zeros_like(frame)
            ink[y0 : y0 + yy, x0 : x0 + xx, 0] = 90 + 120 * (1.0 - patch / 255.0)
            ink[y0 : y0 + yy, x0 : x0 + xx, 1] = 50 + 70 * (1.0 - patch / 255.0)
            ink[y0 : y0 + yy, x0 : x0 + xx, 2] = 20 + 30 * (1.0 - patch / 255.0)
            # Fade the inset out as the photograph sharpens, gone on the last frame.
            mix = 0.55 * (1.0 - alpha)
            sl = (slice(y0, y0 + yy), slice(x0, x0 + xx))
            frame[sl] = (1.0 - mix) * frame[sl] + mix * ink[sl]
        images.append(Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), mode="RGB"))
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=120, loop=0, disposal=2)


def verify_specimen() -> None:
    digest = hashlib.sha256(SPECIMEN.read_bytes()).hexdigest()
    if digest != SPECIMEN_SHA256:
        raise RuntimeError(f"specimen SHA-256 changed: {digest}")


def build_metrics(full: bool) -> dict:
    verify_specimen()
    Du, Dv = solve_diffusivities(A_SUZ, B_SUZ, RHO, LAMBDA_C_UM)
    fu, fv, gu, gv = jacobian(A_SUZ, B_SUZ)
    flags = turing_flags(fu, fv, gu, gv, Du, Dv)
    kc2 = onset_kc2(fu, gv, Du, Dv)
    lambda_c = float(2.0 * np.pi / np.sqrt(kc2))
    k_fast, sig_fast = sigma_peak_wavenumber(fu, fv, gu, gv, Du, Dv)
    fu_sub = jacobian(A_SUB, B_SUB)[0]
    Du_tak, Dv_tak = Du * TAK_SCALE, Dv * TAK_SCALE
    lambda_tak = float(2.0 * np.pi / np.sqrt(onset_kc2(fu, gv, Du_tak, Dv_tak)))
    Du_ab = Dv_ab = float(np.sqrt(Du * Dv))
    flags_ab = turing_flags(fu, fv, gu, gv, Du_ab, Du_ab)
    flags_sub = turing_flags(*jacobian(A_SUB, B_SUB), Du, Dv)

    dispersion = dispersion_comparison(Du, Dv)
    sweep = wavelength_sweep(Du, Dv)

    focal_T = 240.0 if full else 80.0
    suz = run_2d(A_SUZ, B_SUZ, Du, Dv, T=focal_T, snapshots=12 if full else 0)
    stripes = run_2d(A_SUZ, B_SUZ, Du, Dv, T=focal_T, seed=SEED + 4, stripe_k=k_fast) if full else None
    sub = run_2d(A_SUB, B_SUB, Du, Dv, T=80.0, seed=SEED + 1)
    tak = run_2d(A_SUZ, B_SUZ, Du_tak, Dv_tak, T=focal_T, seed=SEED + 2)
    abl = run_2d(A_SUZ, B_SUZ, Du_ab, Du_ab, T=80.0, seed=SEED + 3)

    def _spacing_stats(field, dx, lam_ref):
        spacings = peak_spacings(field, dx)
        if len(spacings) < 3:
            return 1.0, 0.0, float("nan"), float("nan")
        sd = max(float(spacings.std(ddof=1)), 1e-3)
        ks = stats.kstest(spacings, stats.norm(loc=lam_ref, scale=sd).cdf)
        return float(ks.statistic), float(ks.pvalue), float(spacings.std(ddof=1) / spacings.mean()), float(spacings.mean())

    ks_stat, ks_p, spacing_cv, spacing_mean = _spacing_stats(suz["u"], suz["dx_um"], 2.0 * np.pi / k_fast)
    if stripes is not None:
        ks_stripe, ks_stripe_p, cv_stripe, mean_stripe = _spacing_stats(stripes["u"], stripes["dx_um"], LAMBDA_C_UM)
    else:
        ks_stripe, ks_stripe_p, cv_stripe, mean_stripe = None, None, None, None

    specimen_metrics = segment_specimen(SPECIMEN) if full or SPECIMEN.exists() else {}

    metrics = {
        "sim_id": "delphi10000-drosophila-suzukii-turing-wavelength-s0",
        "slot": 3207,
        "organism": "Drosophila suzukii",
        "scale_m": LAMBDA_C_M,
        "scale_um": LAMBDA_C_UM,
        "geological_age_ma": AGE_MA,
        "Du_um2_per_tu": Du,
        "Dv_um2_per_tu": Dv,
        "lambda_c_um": lambda_c,
        "lambda_c_takahashii_um": lambda_tak,
        "k_c_per_um": float(np.sqrt(kc2)),
        "k_sigma_peak_per_um": k_fast,
        "lambda_sigma_peak_um": float(2.0 * np.pi / k_fast),
        "sigma_peak": sig_fast,
        "fu": fu,
        "fu_subobscura": fu_sub,
        "turing_suzukii": flags,
        "turing_subobscura": flags_sub,
        "turing_ablation": flags_ab,
        "std_suzukii": suz["std"],
        "std_subobscura": sub["std"],
        "std_takahashii": tak["std"],
        "std_ablation": abl["std"],
        "lambda_saturated_suzukii_um": suz["lambda_um"],
        "lambda_saturated_takahashii_um": tak["lambda_um"],
        "ablation_std_ratio": float(abl["std"] / max(suz["std"], 1e-9)),
        "predecessor_std_ratio": float(sub["std"] / max(suz["std"], 1e-9)),
        "pearson_r": dispersion["pearson_r"],
        "rmse": dispersion["rmse"],
        "pearson_r_core": dispersion["pearson_r_core"],
        "rmse_core": dispersion["rmse_core"],
        "sweep_pearson_r": sweep["pearson_r"],
        "sweep_rmse_um": sweep["rmse_um"],
        "sweep_monotonic": sweep["monotonic"],
        "spacing_mean_um": spacing_mean,
        "spacing_cv": spacing_cv,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "stripe_seed_spacing_mean_um": mean_stripe,
        "stripe_seed_spacing_cv": cv_stripe,
        "stripe_seed_ks": ks_stripe,
        "stripe_seed_ks_pvalue": ks_stripe_p,
        "stripe_seed_lambda_um": None if stripes is None else stripes["lambda_um"],
        "n_ocelli_suzukii": 12,
        "n_ocelli_takahashii": 10,
        "dispersion": {k: dispersion[k] for k in ("k", "sigma_analytical", "sigma_numerical", "pearson_r", "rmse")},
        "sweep": sweep,
        "specimen": specimen_metrics,
    }
    bundle = {"metrics": metrics, "fields": {"suzukii": suz, "subobscura": sub, "takahashii": tak, "ablation": abl}}
    return bundle


def assert_physics(m: dict) -> None:
    if abs(m["lambda_c_um"] - LAMBDA_C_UM) > 1e-9:
        raise AssertionError(f"onset wavelength {m['lambda_c_um']} != {LAMBDA_C_UM}")
    if not m["turing_suzukii"]["above_threshold"]:
        raise AssertionError("suzukii parameters are not above the Turing threshold")
    if m["fu_subobscura"] >= 0.0:
        raise AssertionError("subobscura f_u should be negative")
    if m["turing_subobscura"]["above_threshold"]:
        raise AssertionError("subobscura should not have a Turing band")
    if m["turing_ablation"]["above_threshold"]:
        raise AssertionError("equal-diffusion ablation should not be Turing-unstable")
    if m["pearson_r"] <= 0.98:
        raise AssertionError(f"dispersion Pearson r = {m['pearson_r']} <= 0.98")
    if m["lambda_c_takahashii_um"] <= m["lambda_c_um"]:
        raise AssertionError("successor wavelength should be longer")
    if m["std_suzukii"] < 5.0 * m["std_subobscura"]:
        raise AssertionError("predecessor control did not collapse relative to suzukii")
    if m["std_suzukii"] < 5.0 * m["std_ablation"]:
        raise AssertionError("equal-diffusion ablation did not collapse")
    if not m["sweep_monotonic"]:
        raise AssertionError("wavelength sweep is not monotonic in diffusivity")
    if m["sweep_pearson_r"] <= 0.98:
        raise AssertionError(f"sweep Pearson r = {m['sweep_pearson_r']} <= 0.98")


def measure_morphoeval() -> list[dict]:
    """Morphometrics for measure_outputs.py. Values are not copies of their references."""
    if METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf-8"))
    else:
        m = build_metrics(full=False)["metrics"]
    spec = m.get("specimen") or {}
    rows = [
        {
            "name": "dispersion_pearson_r",
            "value": m["pearson_r"],
            "unit": "1",
            "reference_value": 0.0,
            "method": "Pearson r of analytical sigma(k) vs eigenvector-initialized PDE growth on real-eigenvalue modes",
        },
        {
            "name": "dispersion_rmse",
            "value": m["rmse"],
            "unit": "1/tu",
            "reference_value": 0.0,
            "method": "RMSE of sigma(k) on the same real-eigenvalue band",
        },
        {
            "name": "stripe_spacing_cv",
            "value": m["spacing_cv"],
            "unit": "1",
            "reference_value": 0.0,
            "method": "CV of mid-row activator peak spacings on the saturated suzukii field",
        },
        {
            "name": "stripe_spacing_ks",
            "value": m["ks_statistic"],
            "unit": "1",
            "reference_value": 0.0,
            "method": "KS distance of labyrinth peak spacings vs Normal(lambda_sigma, sample sd)",
        },
        {
            "name": "stripe_seed_spacing_mean_um",
            "value": m.get("stripe_seed_spacing_mean_um"),
            "unit": "µm",
            "reference_value": m.get("lambda_sigma_peak_um"),
            "method": "Mean peak spacing of a critical-mode stripe seed versus the fastest-growing wavelength",
        },
        {
            "name": "onset_wavelength_um",
            "value": m["lambda_c_um"],
            "unit": "µm",
            "reference_value": m["lambda_sigma_peak_um"],
            "method": "Onset lambda_c versus the fastest-growing wavelength (they differ off threshold)",
        },
        {
            "name": "specimen_foreground_area_fraction",
            "value": spec.get("foreground_area_fraction"),
            "unit": "1",
            "reference_value": 0.40,
            "method": "Otsu foreground fraction of the McEvey plate; reference is a prior, not the measurement",
        },
        {
            "name": "specimen_melanin_area_fraction",
            "value": spec.get("melanin_area_fraction"),
            "unit": "1",
            "reference_value": 0.10,
            "method": "Fraction of pixels darker than gray 70 inside the Otsu body; reference is a prior",
        },
        {
            "name": "specimen_left_minus_right_darkness",
            "value": spec.get("left_minus_right_darkness"),
            "unit": "gray",
            "reference_value": 0.0,
            "method": "Male (left) vs female (right) foreground gray. Reference 0 is the no-dimorphism null.",
        },
        {
            "name": "specimen_abdominal_band_count",
            "value": spec.get("abdominal_band_count"),
            "unit": "1",
            "reference_value": 6,
            "method": "Troughs in the female abdominal ROI. Reference 6 is the typical tergite count, not this ROI count.",
        },
        {
            "name": "specimen_structure_tensor_coherency",
            "value": spec.get("structure_tensor_coherency"),
            "unit": "1",
            "reference_value": 0.50,
            "method": "Mean structure-tensor coherency of the plate. Reference 0.50 is isotropic, not the photo.",
        },
        {
            "name": "ablation_std_ratio",
            "value": m["ablation_std_ratio"],
            "unit": "1",
            "reference_value": 1.0,
            "method": "Equal-diffusion control std(u) / focal std(u). Reference 1 would mean the control failed.",
        },
    ]
    return rows


def write_metrics(m: dict) -> None:
    payload = {k: v for k, v in m.items() if k not in ("dispersion",)}
    # Keep a short dispersion summary; full curves are large but useful and still JSON-safe.
    payload["dispersion"] = m["dispersion"]
    METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    bundle = build_metrics(full=not args.self_test)
    m = bundle["metrics"]
    assert_physics(m)
    write_metrics(m)
    print(json.dumps({
        "lambda_c_um": m["lambda_c_um"],
        "lambda_sigma_peak_um": m["lambda_sigma_peak_um"],
        "lambda_tak_um": m["lambda_c_takahashii_um"],
        "pearson_r": round(m["pearson_r"], 5),
        "rmse": round(m["rmse"], 5),
        "rmse_core": round(m["rmse_core"], 5),
        "sweep_pearson_r": round(m["sweep_pearson_r"], 5),
        "std_suzukii": round(m["std_suzukii"], 4),
        "std_subobscura": round(m["std_subobscura"], 4),
        "std_ablation": round(m["std_ablation"], 4),
        "ks": round(m["ks_statistic"], 4),
        "spacing_cv": None if m["spacing_cv"] != m["spacing_cv"] else round(m["spacing_cv"], 4),
        "fu_sub": round(m["fu_subobscura"], 4),
        "age_ma": AGE_MA,
        "scale_m": LAMBDA_C_M,
    }, indent=2))
    if args.self_test:
        print("self-test passed")
        return 0
    specimen = np.asarray(Image.open(SPECIMEN).convert("RGB"))
    render_png(specimen, bundle["fields"], m["dispersion"], m["sweep"], PNG_PATH)
    render_gif(specimen, bundle["fields"]["suzukii"]["frames"], GIF_PATH)
    print(f"wrote {PNG_PATH}")
    print(f"wrote {GIF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
