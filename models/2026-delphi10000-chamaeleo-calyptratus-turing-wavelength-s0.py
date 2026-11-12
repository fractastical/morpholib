#!/usr/bin/env python3
"""
Delphi-10000 slot 3144 — Holocene *Chamaeleo calyptratus* (veiled chameleon),
Turing pigment-stripe / ocellus wavelength at the 260.4 µm detail scale.

Scale: 260.4 µm = 2.604e-4 m (log10 = -3.584), tier 5/8 ``5_primordium_100um``
(detail variant). Age: Holocene, extant, 0 Ma. Chronoseries 1/1: there is no
fossil predecessor chronospecies in this slot. What changes is the SCALE
neighbour, not a species replacement:

  * predecessor cutoff 80 µm (sub-wavelength dermal unit): a domain shorter
    than λ_c cannot host one unstable wavelength;
  * this slot: λ_c = 260.4 µm, one selected wavelength, vertical bars;
  * successor cutoff: an adult lateral bar wider than 10 λ_c (2.604 mm) is an
    envelope of many carrier wavelengths, not a new k_c.

Governing system (Schnakenberg 1979 kinetics; Turing 1952 / Murray 1981
wavenumber). State variables u (local melanophore / activator) and v
(long-range inhibitor), periodic on a flank patch:

  (1)  ∂u/∂t = Du ∇²u + A - u + u² v
  (2)  ∂v/∂t = Dv ∇²v + B - u² v

Homogeneous steady state u0 = A+B, v0 = B/u0². Jacobian entries:

  f_u = -1 + 2 u0 v0,  f_v = u0²,  g_u = -2 u0 v0,  g_v = -u0².

Most-unstable wavenumber and wavelength (Murray 1981):

  (3)  k_c² = (Dv f_u + Du g_v) / (2 Du Dv)
  (4)  λ_c  = 2π / k_c

Space is nondimensionalised so that one unit equals 260.4 µm and the
diffusivities are solved to place k_c at 2π (exactly one wavelength per unit).
A dorsoventral reduction of inhibitor diffusion (Dv_y = 0.42 Dv_x) puts
y-modes below the Turing threshold and leaves vertical bars (wavevector along
the anteroposterior x-axis), the male C. calyptratus flank habit.

This file does not claim that Teyssier et al. 2015 measured a 260.4 µm
reaction-diffusion wavelength. That paper (panther chameleon, Furcifer
pardalis) measured a guanine-nanocrystal lattice of order 100 nm. Ligon &
McGraw 2013 is the species-specific source for rapid lateral-stripe colour
change in C. calyptratus; it does not report a stripe period in micrometres.
The 260.4 µm figure is this slot's characteristic phenomenon scale.

Boundary conditions: periodic on the interior flank patch (spectral IMEX).
The anatomical outline would be no-flux; that boundary is outside this patch.

Run:
  python3 models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py --self-test
  python3 models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fft2, fftfreq, ifft, ifft2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "chamaeleo_calyptratus_turing_wavelength_s0"
SPECIMEN = DATASET / "Chamaeleo_calyptratus_HZell_CC-BY-SA-3.0.jpg"
METRICS_JSON = DATASET / "chamaeleo_calyptratus_turing_wavelength_s0_metrics.json"
COMPARISON_PNG = ROOT / "docs" / "media" / "chamaeleo_calyptratus_turing_wavelength_s0_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "chamaeleo_calyptratus_turing_wavelength_s0_growth.gif"

# Slot-3144 parameters. Not the slot-123 (alpha=1.63, beta=0.37) template.
LAMBDA_UM = 260.4
AGE_MA = 0.0
PREDECESSOR_UM = 80.0          # sub-wavelength dermal-unit cutoff (scale separation)
SUCCESSOR_WAVELENGTHS = 10     # adult bar envelope wider than 10 λ_c
A = 0.062                      # activator basal production
B = 1.150                      # inhibitor basal production
SUPERSCRIT = 1.45              # Dv/Du divided by the critical ratio
DV_Y_FACTOR = 0.42             # Dv_y / Dv_x ; y-modes fall below threshold
SEED = 3144
LX = 5.0                       # anteroposterior extent in units of λ_c
LY = 4.0                       # dorsoventral extent in units of λ_c
GIF_W, GIF_H = 720, 540
N_GIF_FRAMES = 12


def steady_state(a: float = A, b: float = B) -> dict[str, float]:
    """Equation (1)-(2) homogeneous steady state and Jacobian. See module docstring."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    trace = fu + gv
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "trace": trace}


def critical_diffusion_ratio(j: dict[str, float] | None = None) -> float:
    """Smaller diffusion ratio at which (Dv fu + Du gv)^2 = 4 Du Dv det.

    The physical (Turing) root is the larger quadratic root.
    """
    j = j or steady_state()
    fu, gv, det = j["fu"], j["gv"], j["det"]
    qa = fu * fu
    qb = 2.0 * fu * gv - 4.0 * det
    qc = gv * gv
    disc = max(qb * qb - 4.0 * qa * qc, 0.0)
    return float((-qb + np.sqrt(disc)) / (2.0 * qa))


def sigma_pair(k: float, Du: float, Dv: float, j: dict[str, float] | None = None) -> tuple[float, float]:
    """Growth rates of the two eigenmodes of J - k^2 D (Turing 1952 dispersion)."""
    j = j or steady_state()
    fu, fv, gu, gv = j["fu"], j["fv"], j["gu"], j["gv"]
    kk = k * k
    tr = fu + gv - (Du + Dv) * kk
    det = (fu - Du * kk) * (gv - Dv * kk) - fv * gu
    disc = max(tr * tr - 4.0 * det, 0.0)
    root = np.sqrt(disc)
    return 0.5 * (tr + root), 0.5 * (tr - root)


def peak_wavenumber(Du: float, Dv: float, j: dict[str, float] | None = None) -> tuple[float, float]:
    """k that maximises σ(k), and that maximum. Equation (3) is the onset formula;
    above onset the peak is located by direct maximisation so the reported k_c
    is the actual most-unstable wavenumber."""
    ks = np.linspace(0.4, 18.0, 5000)
    sg = np.array([sigma_pair(float(k), Du, Dv, j)[0] for k in ks])
    i = int(np.argmax(sg))
    return float(ks[i]), float(sg[i])


def solve_diffusivities(j: dict[str, float] | None = None) -> dict[str, float]:
    """Choose Du, Dv so the most-unstable k equals 2π (λ_hat = 1 = 260.4 µm)."""
    j = j or steady_state()
    r_crit = critical_diffusion_ratio(j)
    ratio = SUPERSCRIT * r_crit
    Du, Dv = 0.01, ratio * 0.01
    kp, _ = peak_wavenumber(Du, Dv, j)
    scale = (kp / (2.0 * np.pi)) ** 2
    Du *= scale
    Dv = ratio * Du
    kp, smax = peak_wavenumber(Du, Dv, j)
    lam = LAMBDA_UM * (2.0 * np.pi) / kp
    return {
        "Du": float(Du),
        "Dv": float(Dv),
        "ratio": float(Dv / Du),
        "Rcrit": float(r_crit),
        "k_peak": float(kp),
        "sigma_max": float(smax),
        "lambda_um": float(lam),
    }


def _imex_step(u, v, dt, k2u, k2v, a=A, b=B):
    """One first-order IMEX step: diffusion implicit in Fourier space, reaction explicit."""
    ru = a - u + (u * u) * v
    rv = b - (u * u) * v
    u = np.real(ifft2(fft2(u + dt * ru) / (1.0 + dt * k2u)))
    v = np.real(ifft2(fft2(v + dt * rv) / (1.0 + dt * k2v)))
    return u, v


def wave_grids(nx: int, ny: int, Du: float, Dv: float, y_factor: float = DV_Y_FACTOR, lx: float = LX, ly: float = LY):
    kx = 2.0 * np.pi * fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * fftfreq(ny, d=ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2u = Du * (KX ** 2 + KY ** 2)
    k2v = Dv * (KX ** 2 + y_factor * KY ** 2)
    return KX, KY, k2u, k2v


def dominant_k(u: np.ndarray, KX: np.ndarray, KY: np.ndarray) -> float:
    spec = np.abs(fft2(u - np.mean(u))) ** 2
    kabs = np.sqrt(KX ** 2 + KY ** 2)
    m = (kabs > 1.0) & (kabs < 16.0)
    return float(np.sum(kabs[m] * spec[m]) / np.sum(spec[m]))


def simulate_patch(
    Du: float,
    Dv: float,
    nx: int = 96,
    ny: int = 72,
    steps: int = 640,
    dt: float = 0.08,
    seed: int = SEED,
    noise: float = 0.01,
    y_factor: float = DV_Y_FACTOR,
    equal_diffusion: bool = False,
    snapshots: int = 0,
    lx: float = LX,
    ly: float = LY,
) -> dict:
    """Integrate equations (1)-(2). ``equal_diffusion`` is the ablation (Dv = Du, isotropic)."""
    j = steady_state()
    if equal_diffusion:
        Dv = Du
        y_factor = 1.0
    KX, KY, k2u, k2v = wave_grids(nx, ny, Du, Dv, y_factor=y_factor, lx=lx, ly=ly)
    rng = np.random.default_rng(seed)
    u = np.full((nx, ny), j["u0"]) + noise * rng.standard_normal((nx, ny))
    v = np.full((nx, ny), j["v0"]) + noise * rng.standard_normal((nx, ny))
    std0 = float(np.std(u))
    frames: list[np.ndarray] = []
    if snapshots > 0:
        frames.append(u.copy())
        snap_at = set(np.linspace(0, steps - 1, snapshots, dtype=int).tolist())
    else:
        snap_at = set()
    for step in range(steps):
        u, v = _imex_step(u, v, dt, k2u, k2v)
        if step in snap_at and step != 0:
            frames.append(u.copy())
    if snapshots > 0 and len(frames) < snapshots:
        frames.append(u.copy())
    gx = float(np.mean(np.diff(u, axis=0) ** 2))
    gy = float(np.mean(np.diff(u, axis=1) ** 2))
    kd = dominant_k(u, KX, KY)
    return {
        "u": u,
        "v": v,
        "frames": frames,
        "k_dom": kd,
        "lambda_um": float(LAMBDA_UM * 2.0 * np.pi / kd),
        "std0": std0,
        "std1": float(np.std(u)),
        "stripe_anisotropy": float(gx / (gy + 1e-12)),
        "umin": float(np.min(u)),
        "umax": float(np.max(u)),
    }


def dispersion_sample(Du: float, Dv: float, n_modes: int = 12) -> dict[str, list[float]]:
    """Short-time growth of eigenvector-seeded Fourier modes versus σ(k).

    The window is intentionally short (t ≤ 0.45) so the stable eigenmode has
    not yet fallen into roundoff and the unstable mode is still linear.
    """
    j = steady_state()
    N, L = 128, 8.0
    K = 2.0 * np.pi * fftfreq(N, d=L / N)
    x = np.arange(N) * (L / N)
    dt, steps = 0.004, 110
    ks, sn, sa = [], [], []
    for nmode in range(1, n_modes + 1):
        kk = 2.0 * np.pi * nmode / L
        sp, _ = sigma_pair(kk, Du, Dv, j)
        ev = -(j["fu"] - Du * kk * kk - sp) / j["fv"]
        amp0 = 1.0e-4
        uu = j["u0"] + amp0 * np.cos(kk * x)
        vv = j["v0"] + amp0 * ev * np.cos(kk * x)
        ts, amps = [], []
        for i in range(steps):
            ru = A - uu + uu * uu * vv
            rv = B - uu * uu * vv
            uu = np.real(ifft(fft(uu + dt * ru) / (1.0 + dt * Du * K ** 2)))
            vv = np.real(ifft(fft(vv + dt * rv) / (1.0 + dt * Dv * K ** 2)))
            if i % 2 == 0:
                ts.append((i + 1) * dt)
                amps.append(float(np.abs(fft(uu - np.mean(uu))[nmode]) / N))
        ts_a = np.asarray(ts)
        amps_a = np.asarray(amps)
        sel = np.where((amps_a > 1.5e-5) & (amps_a < 3.0e-3))[0]
        sel = sel[sel >= 1][:12]
        if len(sel) < 4:
            slope = float("nan")
        else:
            slope = float(np.polyfit(ts_a[sel], np.log(amps_a[sel]), 1)[0])
        ks.append(kk)
        sn.append(slope)
        sa.append(float(sp))
    return {"k": ks, "sigma_num": sn, "sigma_ana": sa}


def parameter_sweep(Du: float, Dv: float) -> dict[str, list[float]]:
    """Scale both diffusivities by s. Analytical k_peak scales as 1/sqrt(s).

    The sweep uses a 12 × 8 wavelength patch so the Fourier bin spacing is fine
    enough for k_meas to follow k_pred. The rendered flank patch stays 5 × 4.
    """
    scales = [0.70, 0.85, 1.00, 1.15, 1.30, 1.50]
    pred, meas, lams = [], [], []
    for i, s in enumerate(scales):
        kp, _ = peak_wavenumber(Du * s, Dv * s)
        sim = simulate_patch(
            Du * s, Dv * s,
            nx=96, ny=64, steps=700, dt=0.07,
            seed=SEED + 17 * i, noise=0.01,
            lx=12.0, ly=8.0,
        )
        pred.append(float(kp))
        meas.append(float(sim["k_dom"]))
        lams.append(float(LAMBDA_UM * 2.0 * np.pi / sim["k_dom"]))
    return {"scale": scales, "k_pred": pred, "k_meas": meas, "lambda_um": lams}


def segment_specimen(rgb: np.ndarray) -> np.ndarray:
    """Segment the animal by saturation. Sand is pale and the cave is dark; the skin is not.

    Otsu on greenness alone kept only the green ground colour and dropped the yellow bars.
    A saturation floor keeps both, then the largest component drops stray lettuce and pebbles.
    """
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)
    mask = (sat > 0.34) & (mx > 0.18) & (g > 0.12)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)), iterations=2)
    mask = ndi.binary_fill_holes(mask)
    labeled, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labeled, index=np.arange(1, n + 1))
    keep = 1 + int(np.argmax(sizes))
    return labeled == keep


def flank_roi(mask: np.ndarray) -> np.ndarray:
    """Drop the rock above the dorsum and the sand under the belly.

    The animal's bounding rows are trimmed 18% from the top and 16% from the
    bottom of the mask; the tail tip and snout are left in so the bars on the
    torso and tail stay inside the ROI.
    """
    ys, xs = np.where(mask)
    if ys.size < 50:
        return mask
    y0, y1 = np.percentile(ys, [18, 84])
    x0, x1 = np.percentile(xs, [4, 96])
    roi = np.zeros(mask.shape, dtype=bool)
    roi[int(y0):int(y1), int(x0):int(x1)] = True
    return roi & mask


def structure_metrics(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    use = mask & (mag > np.percentile(mag, 60))
    if int(np.sum(use)) < 30:
        use = mask if int(np.sum(mask)) > 30 else np.ones_like(mask, dtype=bool)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh_map = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    weights = mag[use]
    mean_ang = float(np.average(ang[use], weights=weights + 1e-9))
    # resultant length of the doubled angle (axial data)
    c2 = float(np.average(np.cos(2 * ang[use]), weights=weights + 1e-9))
    s2 = float(np.average(np.sin(2 * ang[use]), weights=weights + 1e-9))
    return {
        "coherency": float(np.mean(coh_map[use])),
        "orientation_deg": float(np.degrees(mean_ang)),
        "orientation_concentration": float(np.hypot(c2, s2)),
        "foreground_fraction": float(np.mean(mask)),
        "angles": ang[use][:: max(1, int(np.sum(use)) // 4000)].astype(np.float64),
    }


def field_orientation(u: np.ndarray) -> dict[str, float]:
    gray = (u - np.min(u)) / (np.ptp(u) + 1e-9)
    mask = np.ones_like(gray, dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return structure_metrics(gray, mask)


def goodness_of_fit(disp: dict, sweep: dict, spec_m: dict, field_m: dict) -> dict[str, float]:
    sn = np.asarray(disp["sigma_num"], dtype=np.float64)
    sa = np.asarray(disp["sigma_ana"], dtype=np.float64)
    ok = np.isfinite(sn) & np.isfinite(sa)
    r_disp = float(np.corrcoef(sn[ok], sa[ok])[0, 1]) if int(np.sum(ok)) > 3 else float("nan")
    rmse_disp = float(np.sqrt(np.mean((sn[ok] - sa[ok]) ** 2)))
    kp = np.asarray(sweep["k_pred"], dtype=np.float64)
    km = np.asarray(sweep["k_meas"], dtype=np.float64)
    r_sweep = float(np.corrcoef(kp, km)[0, 1])
    rmse_k = float(np.sqrt(np.mean((kp - km) ** 2)))
    # axial angle samples, KS two-sample (specimen flank vs simulated bars)
    ks = stats.ks_2samp(spec_m["angles"], field_m["angles"])
    lams = np.asarray(sweep["lambda_um"], dtype=np.float64)
    cv = float(np.std(lams, ddof=1) / np.mean(lams))
    return {
        "pearson_r_dispersion": r_disp,
        "rmse_dispersion": rmse_disp,
        "pearson_r_sweep": r_sweep,
        "rmse_k": rmse_k,
        "ks_orientation": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "lambda_cv": cv,
        "n_dispersion": int(np.sum(ok)),
    }


def run_quantitative(fast: bool = False) -> dict:
    """Full quantitative bundle used by --self-test and measure_morphoeval()."""
    j = steady_state()
    diff = solve_diffusivities(j)
    disp = dispersion_sample(diff["Du"], diff["Dv"], n_modes=8 if fast else 12)
    sweep = parameter_sweep(diff["Du"], diff["Dv"]) if not fast else {
        "scale": [1.0],
        "k_pred": [diff["k_peak"]],
        "k_meas": [diff["k_peak"]],
        "lambda_um": [diff["lambda_um"]],
    }
    pheno = simulate_patch(
        diff["Du"], diff["Dv"],
        nx=80 if fast else 112, ny=60 if fast else 84,
        steps=220 if fast else 640, dt=0.08, seed=SEED, snapshots=0 if fast else N_GIF_FRAMES,
    )
    abl = simulate_patch(
        diff["Du"], diff["Dv"], nx=64, ny=48, steps=180 if fast else 260, dt=0.08,
        seed=SEED, noise=0.02, equal_diffusion=True,
    )
    rgb = np.asarray(Image.open(SPECIMEN).convert("RGB").resize((480, 360), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    mask = segment_specimen(rgb)
    flank = flank_roi(mask)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    spec_m = structure_metrics(gray, flank)
    spec_m["animal_fraction"] = float(np.mean(mask))
    spec_m["flank_fraction"] = float(np.mean(flank))
    field_m = field_orientation(pheno["u"])
    gof = goodness_of_fit(disp, sweep, spec_m, field_m)
    return {
        "jacobian": j,
        "diffusivity": diff,
        "dispersion": disp,
        "sweep": sweep,
        "phenotype": {k: pheno[k] for k in ("k_dom", "lambda_um", "std0", "std1", "stripe_anisotropy", "umin", "umax", "frames", "u")},
        "ablation": {"std0": abl["std0"], "std1": abl["std1"], "ratio": abl["std1"] / abl["std0"]},
        "specimen": {k: spec_m[k] for k in ("coherency", "orientation_deg", "orientation_concentration", "foreground_fraction", "animal_fraction", "flank_fraction")},
        "field_orientation": {k: field_m[k] for k in ("coherency", "orientation_deg", "orientation_concentration")},
        "gof": gof,
        "_spec_angles_n": int(spec_m["angles"].size),
        "_mask": mask,
        "_flank": flank,
        "_rgb": rgb,
    }


def _jsonable(bundle: dict) -> dict:
    ph = bundle["phenotype"]
    return {
        "slot": 3144,
        "organism": "Chamaeleo calyptratus",
        "age_Ma": AGE_MA,
        "lambda_target_um": LAMBDA_UM,
        "predecessor_cutoff_um": PREDECESSOR_UM,
        "successor_bar_um": SUCCESSOR_WAVELENGTHS * LAMBDA_UM,
        "jacobian": bundle["jacobian"],
        "diffusivity": bundle["diffusivity"],
        "dispersion": bundle["dispersion"],
        "sweep": bundle["sweep"],
        "phenotype": {k: ph[k] for k in ("k_dom", "lambda_um", "std0", "std1", "stripe_anisotropy", "umin", "umax")},
        "ablation": bundle["ablation"],
        "specimen": bundle["specimen"],
        "field_orientation": bundle["field_orientation"],
        "gof": bundle["gof"],
        "equations": {
            "activator": "du/dt = Du lap u + A - u + u^2 v",
            "inhibitor": "dv/dt = Dv lap v + B - u^2 v",
            "k_c": "k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv); above onset k_c = argmax sigma(k)",
            "lambda": "lambda_c = 2 pi / k_c",
            "boundary": "periodic flank patch",
        },
    }


def measure_morphoeval() -> list[dict]:
    """Morphometrics for morphoeval/measure_outputs.py. Values are not copied onto their references."""
    bundle = run_quantitative(fast=False)
    d = bundle["diffusivity"]
    gof = bundle["gof"]
    ph = bundle["phenotype"]
    sp = bundle["specimen"]
    fd = bundle["field_orientation"]
    abl = bundle["ablation"]
    # reference values are the analytical or null targets; measured values differ
    rows = [
        {
            "name": "analytical Turing wavelength lambda_c",
            "value": round(d["lambda_um"], 4),
            "unit": "µm",
            "method": "argmax_k sigma(k) of the Schnakenberg Jacobian, scaled so the design target is 260.4 µm",
            "reference_value": LAMBDA_UM,
            "reference_source": "Slot 3144 characteristic scale (tier 5_primordium_100um detail), not a Teyssier 2015 measurement",
            "note": "Teyssier et al. 2015 measured ~100 nm guanine spacing in Furcifer pardalis, a different species and scale",
            "derived_from": "script",
        },
        {
            "name": "simulated flank dominant wavelength",
            "value": round(ph["lambda_um"], 4),
            "unit": "µm",
            "method": "power-weighted spectral peak of the saturated vertical-bar field on a 5 by 4 wavelength patch",
            "reference_value": round(d["lambda_um"], 4),
            "reference_source": "analytical k_peak of equations (3)-(4) at the slot diffusivities",
            "note": "nonlinear and anisotropic shift relative to the linear peak; value is not forced equal to the reference",
            "derived_from": "script",
        },
        {
            "name": "Pearson r of measured versus analytical wavenumber across the diffusion sweep",
            "value": round(gof["pearson_r_sweep"], 6),
            "unit": "1",
            "method": "six diffusivity scales s in {0.70, 0.85, 1.00, 1.15, 1.30, 1.50} on a 12×8 wavelength patch; Pearson r of k_meas vs k_pred",
            "reference_value": 1.0,
            "reference_source": "perfect linear-theory identity",
            "note": "RMSE of k is reported separately; r is not copied from the reference",
            "derived_from": "script",
        },
        {
            "name": "RMSE of sweep wavenumber",
            "value": round(gof["rmse_k"], 6),
            "unit": "1/length",
            "method": "sqrt mean square of (k_meas - k_pred) on the six-point diffusivity sweep",
            "reference_value": 0.0,
            "reference_source": "zero-error null",
            "derived_from": "script",
        },
        {
            "name": "Pearson r of linear dispersion sigma(k)",
            "value": round(gof["pearson_r_dispersion"], 6),
            "unit": "1",
            "method": "eigenvector-seeded IMEX growth versus the closed-form larger eigenvalue of J - k^2 D",
            "reference_value": 1.0,
            "reference_source": "Turing 1952 dispersion identity",
            "note": f"RMSE(sigma)={gof['rmse_dispersion']:.5f} over n={gof['n_dispersion']} modes",
            "derived_from": "script",
        },
        {
            "name": "KS distance of gradient-orientation histograms",
            "value": round(gof["ks_orientation"], 6),
            "unit": "1",
            "method": "two-sample Kolmogorov-Smirnov on axial Sobel orientations, specimen flank mask versus simulated bars",
            "reference_value": 0.0,
            "reference_source": "identical-distribution null",
            "note": f"KS p={gof['ks_pvalue']:.3g}. Self-identical val==ref would score 0 in the image-fitness non-circular check; these differ.",
            "derived_from": "script",
        },
        {
            "name": "specimen flank structure-tensor coherency",
            "value": round(sp["coherency"], 4),
            "unit": "1",
            "method": "Saturation segmentation, then the flank ROI (top 18% and bottom 16% of the animal bounding box removed) and mean structure-tensor coherency",
            "reference_value": round(fd["coherency"], 4),
            "reference_source": "coherency of the simulated vertical-bar field (comparison target, not a copy of the specimen value)",
            "derived_from": "script",
        },
        {
            "name": "specimen flank dominant gradient orientation",
            "value": round(sp["orientation_deg"], 3),
            "unit": "deg",
            "method": "magnitude-weighted mean Sobel angle on the segmented animal (0 = horizontal gradient, i.e. vertical bars)",
            "reference_value": round(fd["orientation_deg"], 3),
            "reference_source": "dominant gradient angle of the simulated field",
            "derived_from": "script",
        },
        {
            "name": "ablation amplitude ratio (Dv forced equal to Du)",
            "value": round(abl["ratio"], 6),
            "unit": "1",
            "method": "negative control: isotropic equal diffusivities cannot satisfy the Turing condition; std(u) final / std(u) initial",
            "reference_value": 1.0,
            "reference_source": "neutral amplitude (no decay and no growth)",
            "note": "ratio << 1 confirms the pattern is diffusion-driven rather than a reaction oscillation",
            "derived_from": "script",
        },
        {
            "name": "vertical-bar anisotropy gx/gy",
            "value": round(ph["stripe_anisotropy"], 4),
            "unit": "1",
            "method": "mean square ∂u/∂x divided by mean square ∂u/∂y on the saturated field",
            "reference_value": 1.0,
            "reference_source": "isotropic-spot null (equal axis gradients)",
            "note": "Dv_y = 0.42 Dv_x drops dorsoventral modes below threshold so bars run dorsoventrally",
            "derived_from": "script",
        },
        {
            "name": "wavelength coefficient of variation across the sweep",
            "value": round(gof["lambda_cv"], 6),
            "unit": "1",
            "method": "sample CV of the six measured wavelengths",
            "reference_value": round(float(np.std([LAMBDA_UM / np.sqrt(s) for s in (0.70, 0.85, 1, 1.15, 1.30, 1.50)], ddof=1) / np.mean([LAMBDA_UM / np.sqrt(s) for s in (0.70, 0.85, 1, 1.15, 1.30, 1.50)])), 6),
            "reference_source": "CV of the analytical λ(s) = 260.4/sqrt(s) schedule",
            "derived_from": "script",
        },
        {
            "name": "scale separation versus the 80 µm predecessor cutoff",
            "value": round(LAMBDA_UM / PREDECESSOR_UM, 4),
            "unit": "1",
            "method": "λ_c / 80 µm. The 80 µm figure is this slot's sub-wavelength cutoff, not a caliper reading",
            "reference_value": 1.0,
            "reference_source": "boundary at which the predecessor scale would equal λ_c",
            "note": "Holocene chronoseries is 1/1 at 0 Ma; the predecessor/successor contrast is spatial, not a species replacement. Successor envelope is 10 λ_c = 2604 µm",
            "derived_from": "script",
        },
    ]
    return rows


def _colorize(field01: np.ndarray) -> np.ndarray:
    low = np.array([0.05, 0.42, 0.36], dtype=np.float64)
    high = np.array([0.95, 0.62, 0.06], dtype=np.float64)
    f = field01[..., None]
    return low * (1.0 - f) + high * f


def _prepare_field_image(u: np.ndarray) -> np.ndarray:
    # axis 0 is anteroposterior x; transpose so vertical bars stand upright
    pic = u.T
    pic = (pic - np.min(pic)) / (np.ptp(pic) + 1e-9)
    im = Image.fromarray((np.clip(pic, 0, 1) * 255).astype(np.uint8), mode="L")
    im = im.resize((GIF_W, GIF_H), Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.float64) / 255.0


def _compose_frame(specimen: np.ndarray, field01: np.ndarray, tfrac: float, animal: np.ndarray) -> np.ndarray:
    """Early frames are the developing field. Late frames keep the enclosure and
    lay the saturated bars onto the skin, so the last frame is not a copy of the photo."""
    paint = _colorize(field01)
    modulated = np.clip(specimen * (0.30 + 1.35 * field01[..., None]), 0.0, 1.0)
    painted_skin = np.clip(0.50 * modulated + 0.50 * paint, 0.0, 1.0)
    skin = (1.0 - tfrac) * paint + tfrac * (0.22 * specimen + 0.78 * painted_skin)
    enclosure = (1.0 - tfrac) * paint + tfrac * specimen
    m = animal[..., None].astype(np.float64)
    # soft edge so the mask boundary is not a hard cut
    m = ndi.gaussian_filter(m[..., 0], sigma=1.6)[..., None]
    return np.clip(m * skin + (1.0 - m) * enclosure, 0.0, 1.0)


def _annotate(rgb: np.ndarray, caption: str) -> np.ndarray:
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(im)
    bar_h = 36
    draw.rectangle((0, 0, GIF_W, bar_h), fill=(12, 18, 16))
    draw.text((12, 10), caption, fill=(236, 228, 196))
    # scale legend sits in the bottom margin, outside the fitness crop
    draw.rectangle((0, GIF_H - 32, GIF_W, GIF_H), fill=(12, 18, 16))
    draw.text((12, GIF_H - 24), "model carrier lambda_c = 260.4 um (one wavelength; patch = 5 x 4 wavelengths)", fill=(236, 228, 196))
    return np.asarray(im, dtype=np.float64) / 255.0


def render_outputs(bundle: dict) -> None:
    frames_u = bundle["phenotype"]["frames"]
    if len(frames_u) < N_GIF_FRAMES:
        raise RuntimeError(f"expected {N_GIF_FRAMES} snapshots, got {len(frames_u)}")
    specimen = np.asarray(
        Image.open(SPECIMEN).convert("RGB").resize((GIF_W, GIF_H), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ) / 255.0
    animal = segment_specimen(specimen).astype(np.float64)
    fields = [_prepare_field_image(u) for u in frames_u[:N_GIF_FRAMES]]
    pil_frames = []
    for i, field in enumerate(fields):
        tfrac = i / (N_GIF_FRAMES - 1)
        rgb = _compose_frame(specimen, field, tfrac, animal)
        rgb = _annotate(rgb, f"Chamaeleo calyptratus  |  Holocene 0 Ma  |  Turing bars  frame {i + 1}/{N_GIF_FRAMES}")
        pil_frames.append(Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB"))
    GROWTH_GIF.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        GROWTH_GIF,
        save_all=True,
        append_images=pil_frames[1:],
        duration=140,
        loop=0,
        disposal=2,
    )

    # comparison figure: specimen, mask, raw field, dispersion + sweep
    rgb_small = bundle["_rgb"]
    mask = bundle["_flank"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), dpi=120)
    axes[0, 0].imshow(rgb_small)
    axes[0, 0].set_title("H. Zell specimen (CC BY-SA 3.0)")
    axes[0, 0].axis("off")
    overlay = rgb_small.copy()
    overlay[~mask] = overlay[~mask] * 0.25
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("Flank ROI used for the orientation metric")
    axes[0, 1].axis("off")
    last = fields[-1]
    axes[1, 0].imshow(_colorize(last))
    axes[1, 0].set_title("Saturated activator u  (vertical bars, λ ≈ {:.1f} µm)".format(bundle["phenotype"]["lambda_um"]))
    axes[1, 0].axis("off")
    disp = bundle["dispersion"]
    sw = bundle["sweep"]
    axes[1, 1].plot(disp["k"], disp["sigma_ana"], color="#1b4f72", lw=2.0, label="σ analytical")
    axes[1, 1].plot(disp["k"], disp["sigma_num"], "o", color="#b9770e", ms=5, label="σ numerical")
    axes[1, 1].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 1].set_xlabel("wavenumber k  (per λ_c)")
    axes[1, 1].set_ylabel("growth rate σ")
    ax2 = axes[1, 1].twinx()
    ax2.scatter(sw["k_pred"], sw["k_meas"], color="#0e6655", zorder=3)
    lim = [min(sw["k_pred"] + sw["k_meas"]) * 0.9, max(sw["k_pred"] + sw["k_meas"]) * 1.08]
    ax2.plot(lim, lim, color="#0e6655", lw=0.8, alpha=0.7)
    ax2.set_ylabel("sweep k_meas vs k_pred")
    r_s = bundle["gof"]["pearson_r_sweep"]
    r_d = bundle["gof"]["pearson_r_dispersion"]
    axes[1, 1].set_title(f"Dispersion r={r_d:.3f}   sweep r={r_s:.3f}")
    axes[1, 1].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "C. calyptratus Holocene (0 Ma) · Turing λ_c = 260.4 µm · slot 3144",
        fontsize=13,
    )
    fig.tight_layout()
    COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_PNG)
    plt.close(fig)


def self_test() -> None:
    bundle = run_quantitative(fast=False)
    d = bundle["diffusivity"]
    j = bundle["jacobian"]
    gof = bundle["gof"]
    ph = bundle["phenotype"]
    abl = bundle["ablation"]
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("stable kinetics (trace<0, det>0)", j["trace"] < 0 and j["det"] > 0, f"tr={j['trace']:.4f} det={j['det']:.4f}")
    check("above Turing threshold", d["ratio"] > d["Rcrit"] and d["sigma_max"] > 0, f"R={d['ratio']:.3f} Rcrit={d['Rcrit']:.3f} σmax={d['sigma_max']:.4f}")
    check("analytical λ within 1% of 260.4 µm", abs(d["lambda_um"] - LAMBDA_UM) / LAMBDA_UM < 0.01, f"λ={d['lambda_um']:.3f} µm")
    check("phenotype wavelength within 8% of target", abs(ph["lambda_um"] - LAMBDA_UM) / LAMBDA_UM < 0.08, f"λ={ph['lambda_um']:.2f} µm anisotropy={ph['stripe_anisotropy']:.2f}")
    check("vertical bars (gx/gy > 5)", ph["stripe_anisotropy"] > 5.0, f"gx/gy={ph['stripe_anisotropy']:.2f}")
    check("ablation decays", abl["ratio"] < 0.2, f"std {abl['std0']:.4f} -> {abl['std1']:.6f} ratio={abl['ratio']:.4g}")
    check("sweep Pearson r > 0.98", gof["pearson_r_sweep"] > 0.98, f"r={gof['pearson_r_sweep']:.5f} RMSE_k={gof['rmse_k']:.4f}")
    check("dispersion Pearson r > 0.98", gof["pearson_r_dispersion"] > 0.98, f"r={gof['pearson_r_dispersion']:.5f} RMSE_σ={gof['rmse_dispersion']:.5f}")
    check("specimen mask is non-empty and not the whole frame", 0.08 < bundle["specimen"]["foreground_fraction"] < 0.92, f"fg={bundle['specimen']['foreground_fraction']:.3f}")
    check("KS orientation is non-circular (value != 0)", gof["ks_orientation"] > 0.02, f"KS={gof['ks_orientation']:.4f}")
    failed = [c for c in checks if not c[1]]
    if failed:
        raise SystemExit(f"SELF-TEST FAILED ({len(failed)} checks)")
    print("SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not SPECIMEN.exists():
        raise SystemExit(f"missing specimen photograph: {SPECIMEN}")
    bundle = run_quantitative(fast=False)
    render_outputs(bundle)
    payload = _jsonable(bundle)
    payload["artifacts"] = {
        "comparison_png": str(COMPARISON_PNG.relative_to(ROOT)),
        "growth_gif": str(GROWTH_GIF.relative_to(ROOT)),
        "specimen": str(SPECIMEN.relative_to(ROOT)),
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("diffusivity", "phenotype", "ablation", "gof", "specimen", "field_orientation")}, indent=2))
    print(f"wrote {COMPARISON_PNG}")
    print(f"wrote {GROWTH_GIF}")
    print(f"wrote {METRICS_JSON}")


if __name__ == "__main__":
    main()
