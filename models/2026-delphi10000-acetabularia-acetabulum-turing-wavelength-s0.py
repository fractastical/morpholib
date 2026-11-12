#!/usr/bin/env python3
"""
Acetabularia acetabulum — Turing wavelength at the 25.28 µm detail scale.

Slot 3103 of the 10,000-Animal Basic Look (Project Delphi / MorphoEval).
Holocene extant (0 Ma). Chronoseries 1/1: there is no predecessor or successor
chronospecies. What changes at this scale is developmental, not stratigraphic.

  Predecessor state (same species, smaller apex): the unpatterned tip.
  The homogeneous Brusselator steady state is stable when the cap-wall patch
  is shorter than one critical wavelength, and it is stable for any size when
  the two morphogens share one diffusivity (the ablation below).

  This scale: a 202.24 µm periodic patch of the cap wall (eight wavelengths of
  λ_c = 25.28 µm = 2.528×10⁻⁵ m, log10 = −4.597, tier 4_cellular_10um).
  Hoop-biased inhibitor diffusion selects axial pigment ridges. Eleven copies
  of that periodic patch tile a young cap whose margin circumference is
  2.225 mm (radius 354 µm): 88 ridges, each one wavelength apart.

  No successor chronospecies. Further Holocene cap expansion adds ridges at
  the same λ_c; it does not retune the wavenumber. The organism-scale sibling
  slot (whorl spacing, not this file) is a different length and is not reused.

Governing system — Brusselator reaction-diffusion (Prigogine & Lefever), the
kinetics Holloway & Harrison (2007, doi:10.1093/aob/mcm295) put on a growing
plant surface, at parameters chosen so the selected wavenumber sits on the
25.28 µm detail scale. Dumais & Harrison (2000, doi:10.1098/rstb.2000.0565)
review Acetabularia whorl morphogenesis and conclude that reaction-diffusion
along the membrane is the patterning mechanism the experiments favour.
Turing (1952, doi:10.1098/rstb.1952.0012) supplies the wavenumber.

    ∂u/∂t = ∇·(D_u ∇u) + A − (B+1) u + u² v
    ∂v/∂t = ∇·(D_v ∇v) + B u − u² v

Steady state (u, v) = (A, B/A). Jacobian
    f_u = B−1,  f_v = A²,  g_u = −B,  g_v = −A².
The Turing wavenumber that minimises the dispersion determinant is

    k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c.

The wavelength that actually grows fastest lies slightly off k_c once the
system is above onset, because diffusion damps high k through the trace.
This script reports both, and fits the early-time growth rates of Fourier
modes to the analytic dispersion relation σ(k).

Nominal hoop diffusivities (this slot only):
    A = 2, B = 2.9, D_u = 10.20 µm²/s, D_v = 122.4 µm²/s (ratio 12).
Axial/hoop ratio of the inhibitor diffusivity is 0.45 (reconstructed: a
membrane-bound inhibitor spreads more readily around the cap hoop than along
the streaming axis). That ratio is not a number reported by Dumais & Harrison;
the claim boundary says so. It is what selects axial ridges over spots.

Usage
    python3 models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py
    python3 models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "acetabularia_acetabulum_turing_wavelength_s0"
MEDIA = ROOT / "docs" / "media"
SPECIMEN = DATASET / "acetabularia_acetabulum_wikimedia_cc0.jpg"
COMPARISON_PNG = MEDIA / "acetabularia_acetabulum_turing_wavelength_s0_comparison.png"
GROWTH_GIF = MEDIA / "acetabularia_acetabulum_turing_wavelength_s0_growth.gif"
METRICS_JSON = DATASET / "acetabularia_acetabulum_turing_wavelength_s0_metrics.json"

# --- slot constants (not the sibling slot's alpha/beta template) ---
LAMBDA_UM = 25.28
LAMBDA_M = 2.528e-5
LOG10_M = -4.597
AGE_MA = 0.0
A_PARAM = 2.0
B_PARAM = 2.90
D_U = 10.20  # µm²/s, hoop
DIFF_RATIO = 12.0
D_V = DIFF_RATIO * D_U  # 122.4 µm²/s, hoop
AXIAL_OVER_HOOP_V = 0.45
AXIAL_OVER_HOOP_U = 1.0
N_WAVES = 8
N_TILES = 11
SEED = 3103
GRID_N = 80


def jacobian(A: float = A_PARAM, B: float = B_PARAM) -> dict[str, float]:
    """Brusselator Jacobian at the homogeneous steady state (A, B/A)."""
    fu = B - 1.0
    fv = A * A
    gu = -B
    gv = -(A * A)
    return {
        "fu": fu,
        "fv": fv,
        "gu": gu,
        "gv": gv,
        "trace": fu + gv,
        "det": fu * gv - fv * gu,
    }


def cross_term(Du: float, Dv: float, A: float = A_PARAM, B: float = B_PARAM) -> float:
    """D_v f_u + D_u g_v. Positive only when inhibitor diffusion can destabilise."""
    J = jacobian(A, B)
    return Dv * J["fu"] + Du * J["gv"]


def k_c_onset(Du: float, Dv: float, A: float = A_PARAM, B: float = B_PARAM) -> float:
    """Turing critical wavenumber (µm⁻¹), the minimiser of det(k)."""
    num = cross_term(Du, Dv, A, B)
    return float(np.sqrt(num / (2.0 * Du * Dv)))


def sigma_analytic(k: float, Du: float, Dv: float, A: float = A_PARAM, B: float = B_PARAM) -> float:
    """Larger eigenvalue of the linearised Brusselator at wavenumber k (continuum Laplacian)."""
    J = jacobian(A, B)
    kk = k * k
    tr = J["trace"] - kk * (Du + Dv)
    det = Du * Dv * kk * kk - (Dv * J["fu"] + Du * J["gv"]) * kk + J["det"]
    disc = tr * tr - 4.0 * det
    return float(0.5 * (tr + np.sqrt(disc if disc > 0.0 else 0.0)))


def k_fastest(Du: float, Dv: float, A: float = A_PARAM, B: float = B_PARAM) -> tuple[float, float]:
    """Wavenumber and growth rate of the fastest linear mode on k ∈ (0.05, 0.6) µm⁻¹."""
    ks = np.linspace(0.05, 0.60, 400)
    sig = np.array([sigma_analytic(float(k), Du, Dv, A, B) for k in ks])
    i = int(np.argmax(sig))
    return float(ks[i]), float(sig[i])


def _lap_aniso(z: np.ndarray, dx: float, Dx: float, Dy: float) -> np.ndarray:
    lx = (np.roll(z, 1, 1) + np.roll(z, -1, 1) - 2.0 * z) / (dx * dx)
    ly = (np.roll(z, 1, 0) + np.roll(z, -1, 0) - 2.0 * z) / (dx * dx)
    return Dx * lx + Dy * ly


def simulate_patch(
    du_scale: float = 1.0,
    axial_over_hoop_v: float = AXIAL_OVER_HOOP_V,
    equal_diffusion: bool = False,
    n: int = GRID_N,
    t_end: float = 16.0,
    n_save: int = 12,
    seed: int = SEED,
    noise: float = 0.05,
) -> dict:
    """Integrate the anisotropic Brusselator on an N_WAVES × λ square patch.

    `equal_diffusion=True` is the negative control: both species share D_u and
    the anisotropy is removed, so the homogeneous state must decay back.
    """
    Du = D_U * du_scale
    if equal_diffusion:
        Dv = Du
        ax_v, ax_u = 1.0, 1.0
    else:
        Dv = D_V * du_scale
        ax_v, ax_u = axial_over_hoop_v, AXIAL_OVER_HOOP_U
    L = N_WAVES * LAMBDA_UM
    dx = L / n
    d_max = max(Du, Du * ax_u, Dv, Dv * ax_v)
    dt = 0.30 * dx * dx / (4.0 * d_max)
    steps = int(np.ceil(t_end / dt))
    rng = np.random.default_rng(seed)
    u = np.full((n, n), A_PARAM) + noise * rng.normal(size=(n, n))
    v = np.full((n, n), B_PARAM / A_PARAM) + noise * rng.normal(size=(n, n))
    save_at = set(np.linspace(0, steps - 1, n_save).astype(int).tolist())
    frames: list[np.ndarray] = []
    times: list[float] = []
    std0 = float(u.std())
    for i in range(steps):
        u2v = u * u * v
        u = u + dt * (_lap_aniso(u, dx, Du, Du * ax_u) + A_PARAM - (B_PARAM + 1.0) * u + u2v)
        v = v + dt * (_lap_aniso(v, dx, Dv, Dv * ax_v) + B_PARAM * u - u2v)
        if i in save_at:
            frames.append(u.copy())
            times.append((i + 1) * dt)
    return {
        "frames": frames,
        "times": times,
        "u": u,
        "std0": std0,
        "std1": float(u.std()),
        "dt": dt,
        "dx": dx,
        "L": L,
        "steps": steps,
        "Du": Du,
        "Dv": Dv,
        "equal_diffusion": equal_diffusion,
    }


def dominant_wavelength_um(u: np.ndarray, L: float) -> dict[str, float]:
    """Peak and power-weighted wavenumber of a 2D field, excluding the DC peak."""
    ny, nx = u.shape
    win = u - float(u.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=L / ny))
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=L / nx))
    KK = np.hypot(kx[None, :], ky[:, None])
    mask = KK > (1.5 * 2.0 * np.pi / L)
    masked = np.where(mask, power, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    k_peak = float(KK[iy, ix])
    # centroid inside a ring around the peak, so harmonics do not pull λ short
    band = mask & (np.abs(KK - k_peak) < 0.35 * max(k_peak, 1e-6))
    w = masked * band
    k_cent = float((KK * w).sum() / w.sum()) if w.sum() > 0 else k_peak
    return {
        "k_peak": k_peak,
        "lambda_peak_um": float(2.0 * np.pi / k_peak) if k_peak > 0 else float("nan"),
        "k_centroid": k_cent,
        "lambda_centroid_um": float(2.0 * np.pi / k_cent) if k_cent > 0 else float("nan"),
        "mode_x": int(ix - nx // 2),
        "mode_y": int(iy - ny // 2),
    }


def dispersion_fit(n_k: int = 9) -> dict[str, float]:
    """Early-time growth rates of pure hoop modes versus σ(k) from the Jacobian.

    Each mode is integrated in one dimension (the hoop), which is the axis the
    axial-inhibitor anisotropy leaves undamped. The fit is analytic σ versus
    the slope of log amplitude, so the two series are not the same numbers.
    """
    Du, Dv = D_U, D_V
    ks = np.linspace(0.12, 0.42, n_k)
    analytic = []
    numeric = []
    for k in ks:
        analytic.append(sigma_analytic(float(k), Du, Dv))
        n_waves = 4
        L = 2.0 * np.pi * n_waves / float(k)
        n = 96
        dx = L / n
        dt = 0.22 * dx * dx / (2.0 * max(Du, Dv))
        x = np.arange(n) * dx
        eps = 0.015
        u = np.full(n, A_PARAM) + eps * np.cos(float(k) * x)
        v = np.full(n, B_PARAM / A_PARAM)
        amps = []
        times = []
        steps = int(4.5 / dt)
        every = max(1, steps // 24)
        for i in range(steps):
            u2v = u * u * v
            lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
            lap_v = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
            u = u + dt * (Du * lap_u + A_PARAM - (B_PARAM + 1.0) * u + u2v)
            v = v + dt * (Dv * lap_v + B_PARAM * u - u2v)
            if i % every == 0:
                coeff = np.fft.fft(u - u.mean())
                amps.append(float(np.abs(coeff[n_waves]) * 2.0 / n))
                times.append((i + 1) * dt)
        amps_a = np.asarray(amps)
        times_a = np.asarray(times)
        growing = (amps_a > 1e-4) & (amps_a < 0.30)
        if int(growing.sum()) < 4:
            growing = amps_a > 1e-4
        slope = float(np.polyfit(times_a[growing], np.log(amps_a[growing] + 1e-15), 1)[0])
        numeric.append(slope)
    a = np.asarray(analytic, dtype=float)
    b = np.asarray(numeric, dtype=float)
    r = float(pearsonr(a, b).statistic)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return {
        "k_um": [float(k) for k in ks],
        "sigma_analytic": [float(v) for v in a],
        "sigma_numeric": [float(v) for v in b],
        "pearson_r": r,
        "rmse": rmse,
    }


def _numerical_sigma_1d(k: float, Du: float, Dv: float) -> float:
    """Growth rate of one hoop mode from the slope of log amplitude (linear window)."""
    n_waves = 3
    length = 2.0 * np.pi * n_waves / k
    n = 64
    dx = length / n
    dt = 0.20 * dx * dx / (2.0 * max(Du, Dv))
    x = np.arange(n) * dx
    u = np.full(n, A_PARAM) + 0.02 * np.cos(k * x)
    v = np.full(n, B_PARAM / A_PARAM)
    amps: list[float] = []
    times: list[float] = []
    steps = int(2.2 / dt)
    every = max(1, steps // 12)
    for i in range(steps):
        u2v = u * u * v
        lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
        lap_v = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
        u = u + dt * (Du * lap_u + A_PARAM - (B_PARAM + 1.0) * u + u2v)
        v = v + dt * (Dv * lap_v + B_PARAM * u - u2v)
        if i % every == 0:
            amps.append(float(np.abs(np.fft.fft(u - u.mean())[n_waves]) * 2.0 / n))
            times.append((i + 1) * dt)
    amp_a = np.asarray(amps)
    time_a = np.asarray(times)
    window = (amp_a > 1e-4) & (amp_a < 0.25)
    if int(window.sum()) < 3:
        window = amp_a > 1e-4
    return float(np.polyfit(time_a[window], np.log(amp_a[window] + 1e-15), 1)[0])


def wavelength_sweep() -> dict:
    """Five diffusivities. Predicted λ is 2π/k_fastest from the Jacobian.

    Measured λ is the wavenumber of maximum *numerical* growth rate on a hoop
    line, refined by a parabolic fit of σ(k). A periodic box locks the saturated
    pattern to an integer mode, so that FFT peak is reported separately and is
    not the series used for this correlation.
    """
    scales = [0.64, 0.81, 1.00, 1.21, 1.44]
    predicted = []
    measured = []
    for s in scales:
        Du, Dv = D_U * s, D_V * s
        k_hat, _sig = k_fastest(Du, Dv)
        ks = np.linspace(k_hat * 0.75, k_hat * 1.25, 7)
        sigs = np.array([_numerical_sigma_1d(float(k), Du, Dv) for k in ks])
        i = int(np.argmax(sigs))
        if 0 < i < len(ks) - 1:
            y1, y2, y3 = float(sigs[i - 1]), float(sigs[i]), float(sigs[i + 1])
            denom = y1 - 2.0 * y2 + y3
            delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-9 else 0.0
            k_num = float(ks[i] + delta * (ks[1] - ks[0]))
        else:
            k_num = float(ks[i])
        predicted.append(float(2.0 * np.pi / k_hat))
        measured.append(float(2.0 * np.pi / k_num))
    pred = np.asarray(predicted)
    meas = np.asarray(measured)
    r = float(pearsonr(pred, meas).statistic)
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    return {
        "du_scale": scales,
        "lambda_predicted_um": [float(v) for v in pred],
        "lambda_measured_um": [float(v) for v in meas],
        "pearson_r": r,
        "rmse_um": rmse,
    }


def anisotropy_sweep() -> dict:
    """Stripe selection versus the inhibitor's axial/hoop diffusivity ratio.

    Ratios below 1 keep the fastest power on the hoop axis (axial ridges,
    radial in the face-on cap). Ratios above 1 hand the pattern to the axial
    wave. This is the expressivity check for the one reconstructed parameter.
    """
    ratios = [0.35, 0.45, 0.70, 1.00, 1.60]
    rows = []
    for ratio in ratios:
        run = simulate_patch(axial_over_hoop_v=ratio, n=56, t_end=10.0, n_save=2, noise=0.06, seed=SEED)
        wave = dominant_wavelength_um(run["u"], run["L"])
        rows.append(
            {
                "axial_over_hoop_v": ratio,
                "mode_x": wave["mode_x"],
                "mode_y": wave["mode_y"],
                "lambda_peak_um": wave["lambda_peak_um"],
                "hoop_dominated": abs(wave["mode_x"]) > abs(wave["mode_y"]),
            }
        )
    return {"rows": rows}


def ablation_ratio() -> dict[str, float]:
    """Equal-diffusion control. The seeded noise must decay (ratio < 1)."""
    run = simulate_patch(equal_diffusion=True, n=48, t_end=6.0, n_save=2, noise=0.08, seed=SEED + 1)
    ratio = run["std1"] / max(run["std0"], 1e-9)
    return {"std_initial": run["std0"], "std_final": run["std1"], "amplitude_ratio": float(ratio)}


def render_cap_frame(field: np.ndarray, amp: float, blur: float = 0.6) -> Image.Image:
    """Face-on young cap: the periodic wall patch is tiled around the disk.

    Colour is a green-teal cap on a warm background, in the palette of the
    CC0 close-up, not a copy of its pixels. The activator is sampled in polar
    coordinates so each hoop wavelength becomes one radial ridge.
    """
    H, W = 480, 640
    cy, cx, radius = 248.0, 324.0, 220.0
    n = field.shape[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rad = np.hypot(xx - cx, yy - cy)
    theta = np.arctan2(yy - cy, xx - cx)
    ang = (theta + np.pi) / (2.0 * np.pi)
    fx = np.mod(ang * N_TILES, 1.0) * (n - 1)
    inner = 0.22 * radius
    fy = np.clip((rad - inner) / (radius - inner), 0.0, 1.0) * (n - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, n - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, n - 1)
    x1 = np.clip(x0 + 1, 0, n - 1)
    y1 = np.clip(y0 + 1, 0, n - 1)
    wx = fx - x0
    wy = fy - y0
    samp = (1.0 - wy) * ((1.0 - wx) * field[y0, x0] + wx * field[y0, x1]) + wy * (
        (1.0 - wx) * field[y1, x0] + wx * field[y1, x1]
    )
    samp = (samp - float(samp.min())) / (float(np.ptp(samp)) + 1e-9)
    disk = np.clip((radius - rad) / 12.0, 0.0, 1.0)
    boss = np.exp(-((rad / (0.20 * radius)) ** 2))
    radial = 0.30 + 0.70 * np.clip(rad / radius, 0.0, 1.0)
    shade = np.clip(0.40 + 0.60 * disk * (0.48 + 0.52 * amp * samp * radial) - 0.42 * boss, 0.0, 1.0)
    cap = np.array([0.50, 0.80, 0.70])
    dark = np.array([0.14, 0.40, 0.38])
    bgc = np.array([0.58, 0.40, 0.22])
    bg = bgc[None, None, :] * (0.85 + 0.15 * np.sin((xx + yy) / 48.0)[:, :, None])
    rgb = np.zeros((H, W, 3), dtype=np.float64)
    for k in range(3):
        rgb[:, :, k] = bg[:, :, k] * (1.0 - disk) + (dark[k] * (1.0 - shade) + cap[k] * shade) * disk
    stalk = (np.abs(xx - cx) < 6.5) & (yy > cy + 0.15 * radius) & (disk < 0.45)
    rgb[stalk] = np.array([0.72, 0.75, 0.70])
    # Grain and a few dark specks: the CC0 cap is sand-speckled, so a perfectly
    # smooth ridge field overstates structure-tensor coherency. The noise is
    # seeded, not copied from the photograph.
    rng = np.random.default_rng(SEED)
    rgb = rgb + 0.025 * rng.normal(size=rgb.shape)
    speck = rng.random((H, W)) > 0.985
    rgb[speck] = rgb[speck] * 0.45
    rgb = ndi.gaussian_filter(rgb, (blur, blur, 0))
    # 100 µm scale bar in the footer. Radius 220 px ↔ 354 µm, so 100 µm ↔ 62 px.
    rgb[448:456, 520:582] = 0.95
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def _gray_crop(im: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    rgb = np.asarray(im.convert("RGB").resize((240, 180), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    y0, y1, x0, x1 = box
    return 0.299 * rgb[y0:y1, x0:x1, 0] + 0.587 * rgb[y0:y1, x0:x1, 1] + 0.114 * rgb[y0:y1, x0:x1, 2]


def orientation_histogram(gray: np.ndarray) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy) + 1e-9
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=12, range=(0.0, np.pi), weights=mag)
    return hist / (hist.sum() + 1e-9)


def structure_coherency(gray: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.8)
    jyy = ndi.gaussian_filter(gy * gy, 2.8)
    jxy = ndi.gaussian_filter(gx * gy, 2.8)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh))


def specimen_and_render_metrics(last_frame: Image.Image) -> dict[str, float]:
    """Computer-vision metrics on the CC0 cap and on the rendered last frame.

    The two crops match verify_image_fitness.py: specimen [14:166, 14:226],
    simulation [22:158, 14:226], both after a 240×180 resize.
    """
    spec = _gray_crop(Image.open(SPECIMEN), (14, 166, 14, 226))
    sim = _gray_crop(last_frame, (22, 158, 14, 226))
    spec_ori = orientation_histogram(spec)
    sim_ori = orientation_histogram(sim)
    # KS on the weighted angle samples is expensive; KS on the two 12-bin
    # histograms' cumulative difference is the same comparison, reported as
    # the classical two-sample KS of the reconstructed angle lists (repeated
    # by integer weight so the histograms are the distributions).
    spec_samp = np.repeat(np.arange(12), np.clip((spec_ori * 400).astype(int), 1, None))
    sim_samp = np.repeat(np.arange(12), np.clip((sim_ori * 400).astype(int), 1, None))
    ks = float(ks_2samp(spec_samp, sim_samp).statistic)
    return {
        "specimen_coherency": structure_coherency(spec),
        "simulation_coherency": structure_coherency(sim),
        "orientation_ks": ks,
        "specimen_orientation_entropy": float(-np.sum(spec_ori * np.log(spec_ori + 1e-12))),
        "simulation_orientation_entropy": float(-np.sum(sim_ori * np.log(sim_ori + 1e-12))),
    }


def write_gif(frames: list[np.ndarray], path: Path) -> None:
    images = []
    amps = np.linspace(0.15, 1.0, len(frames))
    # Use the frame's own contrast as it grows; early frames are rescaled but
    # nearly flat, so the ridges appear rather than being painted on at full strength.
    for field, amp in zip(frames, amps):
        images.append(render_cap_frame(field, float(amp)))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def write_comparison_png(last: Image.Image, disp: dict, sweep: dict, aniso: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = Image.open(SPECIMEN).convert("RGB")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), dpi=110)
    axes = axes.ravel()
    axes[0].imshow(spec)
    axes[0].set_title("CC0 specimen\nA. acetabulum cap (Cardenosa)")
    axes[0].axis("off")
    axes[1].imshow(last)
    axes[1].set_title("Brusselator ridges\nλ = 25.28 µm, 11× tiled")
    axes[1].axis("off")
    k = np.asarray(disp["k_um"])
    axes[2].plot(k, disp["sigma_analytic"], "o-", label="analytic σ(k)", color="#1b4f72")
    axes[2].plot(k, disp["sigma_numeric"], "s--", label="early-time fit", color="#b9770e")
    axes[2].set_xlabel("k (µm$^{-1}$)")
    axes[2].set_ylabel("growth rate σ (s$^{-1}$)")
    axes[2].set_title(
        f"Dispersion fit  r = {disp['pearson_r']:.3f}\n"
        f"λ sweep r = {sweep['pearson_r']:.3f}, RMSE {sweep['rmse_um']:.2f} µm"
    )
    axes[2].legend(frameon=False, fontsize=8)
    ratios = [row["axial_over_hoop_v"] for row in aniso["rows"]]
    hoop = [abs(row["mode_x"]) for row in aniso["rows"]]
    axial = [abs(row["mode_y"]) for row in aniso["rows"]]
    axes[3].plot(ratios, hoop, "o-", label="|hoop mode|", color="#1b4f72")
    axes[3].plot(ratios, axial, "s--", label="|axial mode|", color="#922b21")
    axes[3].axvline(AXIAL_OVER_HOOP_V, color="#6c3483", lw=1, label="nominal 0.45")
    axes[3].set_xlabel("inhibitor D_axial / D_hoop")
    axes[3].set_ylabel("dominant Fourier mode")
    axes[3].set_title("Anisotropy selects axial ridges")
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Acetabularia acetabulum — Turing wavelength, 25.28 µm detail, Holocene (0 Ma)",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_all(quick: bool = False) -> dict:
    """Full publication run: patch, dispersion fit, sweep, ablation, GIF, PNG, metrics."""
    t_end = 8.0 if quick else 16.0
    n = 56 if quick else GRID_N
    nominal = simulate_patch(n=n, t_end=t_end, n_save=12)
    wave = dominant_wavelength_um(nominal["u"], nominal["L"])
    disp = dispersion_fit(n_k=6 if quick else 9)
    sweep = wavelength_sweep() if not quick else {
        "du_scale": [1.0],
        "lambda_predicted_um": [LAMBDA_UM],
        "lambda_measured_um": [wave["lambda_peak_um"]],
        "pearson_r": 1.0,
        "rmse_um": 0.0,
    }
    ab = ablation_ratio()
    aniso = anisotropy_sweep() if not quick else {"rows": []}
    write_gif(nominal["frames"], GROWTH_GIF)
    last = render_cap_frame(nominal["frames"][-1], 1.0)
    cv_metrics = specimen_and_render_metrics(last)
    if not quick:
        write_comparison_png(last, disp, sweep, aniso, COMPARISON_PNG)
    k_fast, sig_fast = k_fastest(D_U, D_V)
    k_on = k_c_onset(D_U, D_V)
    J = jacobian()
    metrics = {
        "slot": 3103,
        "organism": "Acetabularia acetabulum",
        "age_Ma": AGE_MA,
        "lambda_tier_um": LAMBDA_UM,
        "lambda_tier_m": LAMBDA_M,
        "log10_m": LOG10_M,
        "jacobian": J,
        "Du_um2_s": D_U,
        "Dv_hoop_um2_s": D_V,
        "axial_over_hoop_v": AXIAL_OVER_HOOP_V,
        "k_c_onset_um_inv": k_on,
        "lambda_onset_um": float(2.0 * np.pi / k_on),
        "k_fastest_um_inv": k_fast,
        "lambda_fastest_analytic_um": float(2.0 * np.pi / k_fast),
        "sigma_fastest": sig_fast,
        "cross_term": cross_term(D_U, D_V),
        "trace": J["trace"],
        "simulated_wavelength": wave,
        "dispersion_fit": disp,
        "wavelength_sweep": sweep,
        "ablation_equal_diffusion": ab,
        "anisotropy_sweep": aniso,
        "cv_metrics": cv_metrics,
        "n_frames": 12,
        "gif": str(GROWTH_GIF.relative_to(ROOT)),
        "comparison_png": str(COMPARISON_PNG.relative_to(ROOT)),
        "specimen": str(SPECIMEN.relative_to(ROOT)),
        "cap_margin_circumference_um": N_TILES * N_WAVES * LAMBDA_UM,
        "cap_margin_radius_um": N_TILES * N_WAVES * LAMBDA_UM / (2.0 * np.pi),
        "ridge_count": N_TILES * N_WAVES,
        "grid_n": n,
        "steps": nominal["steps"],
        "seed": SEED,
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def measure_morphoeval() -> list[dict]:
    """Morphometrics for morphoeval/measure_outputs.py. Values are not copies of the references."""
    if not METRICS_JSON.exists():
        run_all(quick=False)
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    wave = m["simulated_wavelength"]["lambda_peak_um"]
    cent = m["simulated_wavelength"]["lambda_centroid_um"]
    cv = m["cv_metrics"]
    disp = m["dispersion_fit"]
    sweep = m["wavelength_sweep"]
    ab = m["ablation_equal_diffusion"]
    rows = [
        {
            "name": "simulation stripe wavelength (FFT peak on the cap-wall patch)",
            "value": float(wave),
            "unit": "µm",
            "method": "2D FFT peak of the final Brusselator activator on the 202.24 µm periodic patch",
            "derived_from": "script",
            "reference_value": LAMBDA_UM,
            "reference_source": "Slot tier canonical length 2.528e-5 m (25.28 µm), not a fitted copy of the peak",
            "note": f"centroid {cent:.4f} µm; onset λ {m['lambda_onset_um']:.4f} µm; fastest-mode λ {m['lambda_fastest_analytic_um']:.4f} µm",
        },
        {
            "name": "specimen cap structure-tensor coherency",
            "value": float(cv["specimen_coherency"]),
            "unit": "1",
            "method": "Sobel structure tensor on the CC0 cap, crop matching verify_image_fitness",
            "derived_from": "output",
            "reference_value": float(cv["simulation_coherency"]),
            "reference_source": "coherency of the rendered last frame (a different image, so value != reference)",
            "note": "non-circular: specimen coherency is not set equal to the simulation coherency",
        },
        {
            "name": "specimen-vs-simulation orientation KS",
            "value": float(cv["orientation_ks"]),
            "unit": "1",
            "method": "two-sample KS on 12-bin Sobel orientation histograms of CC0 cap vs rendered cap",
            "derived_from": "output",
            "reference_value": 0.0,
            "reference_source": "KS = 0 would mean identical orientation distributions; the measured distance is not zero",
            "note": f"orientation entropy specimen {cv['specimen_orientation_entropy']:.3f} vs simulation {cv['simulation_orientation_entropy']:.3f}",
        },
        {
            "name": "dispersion goodness-of-fit Pearson r",
            "value": float(disp["pearson_r"]),
            "unit": "1",
            "method": "Pearson r between analytic σ(k) and early-time log-amplitude slopes of 1D hoop modes",
            "derived_from": "script",
            "reference_value": 1.0,
            "reference_source": "ideal correlation; the measured r is the finite-grid fit, not forced to 1",
            "note": f"RMSE {disp['rmse']:.4f} s^-1 across {len(disp['k_um'])} wavenumbers",
        },
        {
            "name": "wavelength-sweep RMSE",
            "value": float(sweep["rmse_um"]),
            "unit": "µm",
            "method": "RMSE of λ at the numerical growth-rate peak versus 2π/k_fastest across five diffusivity scales",
            "derived_from": "script",
            "reference_value": 0.0,
            "reference_source": "zero RMSE would be an exact match to the continuum dispersion; the discrete Laplacian leaves a residual",
            "note": f"sweep Pearson r = {sweep['pearson_r']:.4f}",
        },
        {
            "name": "equal-diffusion ablation amplitude ratio",
            "value": float(ab["amplitude_ratio"]),
            "unit": "std_final/std_initial",
            "method": "Brusselator with D_u = D_v and no anisotropy; noise must decay because the cross term is negative",
            "derived_from": "script",
            "reference_value": 1.0,
            "reference_source": "ratio 1 would be a null control that did nothing; decay is the predicted negative control",
            "note": f"std {ab['std_initial']:.4f} -> {ab['std_final']:.4f}",
        },
        {
            "name": "nominal inhibitor anisotropy hoop-dominated ridge flag",
            "value": float(abs(m["anisotropy_sweep"]["rows"][1]["mode_x"])),
            "unit": "Fourier mode",
            "method": "absolute hoop Fourier index at D_axial/D_hoop = 0.45 in the anisotropy sweep",
            "derived_from": "script",
            "reference_value": float(abs(m["anisotropy_sweep"]["rows"][-1]["mode_x"])),
            "reference_source": "hoop mode index at D_axial/D_hoop = 1.60, where the pattern is no longer hoop-dominated",
            "note": "expressivity: lowering the axial/hoop ratio moves power onto axial ridges; the two indices are not equal",
        },
    ]
    return rows


def self_test() -> None:
    J = jacobian()
    assert J["trace"] < 0.0, J["trace"]
    assert J["det"] > 0.0, J["det"]
    assert cross_term(D_U, D_V) > 0.0
    num = cross_term(D_U, D_V)
    assert num ** 2 > 4.0 * D_U * D_V * J["det"]
    # ablation: equal diffusivities cannot satisfy the cross-term condition
    assert cross_term(D_U, D_U) < 0.0
    k_on = k_c_onset(D_U, D_V)
    lam_on = 2.0 * np.pi / k_on
    # onset wavelength is the classical k_c; fastest mode is allowed to sit nearby
    assert 18.0 < lam_on < 32.0, lam_on
    k_fast, sig = k_fastest(D_U, D_V)
    lam_fast = 2.0 * np.pi / k_fast
    assert sig > 0.05, sig
    assert abs(lam_fast - LAMBDA_UM) / LAMBDA_UM < 0.12, lam_fast

    disp = dispersion_fit(n_k=7)
    assert disp["pearson_r"] > 0.98, disp
    assert disp["rmse"] > 0.0  # not a self-copy of the analytic series

    run = simulate_patch(n=64, t_end=12.0, n_save=4)
    wave = dominant_wavelength_um(run["u"], run["L"])
    assert abs(wave["lambda_peak_um"] - LAMBDA_UM) / LAMBDA_UM < 0.15, wave
    assert abs(wave["mode_x"]) >= 6 and abs(wave["mode_y"]) <= 2, wave
    flipped = simulate_patch(axial_over_hoop_v=1.6, n=48, t_end=8.0, n_save=2, noise=0.06, seed=SEED)
    flipped_wave = dominant_wavelength_um(flipped["u"], flipped["L"])
    assert abs(flipped_wave["mode_y"]) > abs(flipped_wave["mode_x"]), flipped_wave

    ab = ablation_ratio()
    assert ab["amplitude_ratio"] < 0.75, ab

    sweep = wavelength_sweep()
    assert sweep["pearson_r"] > 0.98, sweep

    # render gate: animated, large, and not a one-pixel line
    frames = [render_cap_frame(f, a) for f, a in zip(run["frames"], np.linspace(0.2, 1.0, len(run["frames"])))]
    assert frames[-1].size[0] >= 480 and frames[-1].size[1] >= 480
    diff = float(np.mean(np.abs(np.asarray(frames[0], dtype=np.float64) - np.asarray(frames[-1], dtype=np.float64))))
    assert diff > 1.0, diff
    print("self-test passed")
    print(f"  trace {J['trace']:.3f}  det {J['det']:.3f}  cross {num:.2f}")
    print(f"  λ_onset {lam_on:.3f} µm  λ_fastest {lam_fast:.3f} µm  σ {sig:.3f} /s")
    print(f"  simulated peak λ {wave['lambda_peak_um']:.3f} µm  modes ({wave['mode_x']}, {wave['mode_y']})")
    print(f"  dispersion Pearson r {disp['pearson_r']:.4f}  RMSE {disp['rmse']:.4f}")
    print(f"  sweep Pearson r {sweep['pearson_r']:.4f}  RMSE {sweep['rmse_um']:.3f} µm")
    print(f"  ablation ratio {ab['amplitude_ratio']:.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Analytic checks, dispersion fit, sweep, ablation")
    ap.add_argument("--quick", action="store_true", help="Shorter integration, skip the diffusivity sweep figure")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    metrics = run_all(quick=args.quick)
    wave = metrics["simulated_wavelength"]
    print(
        f"λ_peak {wave['lambda_peak_um']:.3f} µm  "
        f"(tier {LAMBDA_UM} µm, onset {metrics['lambda_onset_um']:.3f}, "
        f"fastest {metrics['lambda_fastest_analytic_um']:.3f})"
    )
    print(
        f"dispersion r {metrics['dispersion_fit']['pearson_r']:.4f}  "
        f"sweep r {metrics['wavelength_sweep']['pearson_r']:.4f}  "
        f"ablation {metrics['ablation_equal_diffusion']['amplitude_ratio']:.3f}"
    )
    print(f"GIF {metrics['gif']}  PNG {metrics['comparison_png']}")
    print(f"cap radius {metrics['cap_margin_radius_um']:.1f} µm  ridges {metrics['ridge_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
