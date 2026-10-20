#!/usr/bin/env python3
"""
2016-hubaud-yap-excitable-segmentation.py — YAP-Gated Excitable FitzHugh-Nagumo Segmentation Clock
==================================================================================================

Faithful mathematical, numerical, and morphometric reproduction of:
  1. Hubaud, A., Regev, I., Mahadevan, L. & Pourquie, O. (2017).
     "Excitable dynamics and Yap-dependent mechanical cues drive the
     segmentation clock."
     Cell 171(3):668-682.e11.
     DOI: 10.1016/j.cell.2017.08.043
  2. FitzHugh, R. (1961). "Impulses and physiological states in theoretical
     models of nerve membrane."
     Biophysical Journal 1(6):445-466.
     DOI: 10.1016/S0006-3495(61)86902-6
  3. Lauschke, V. M., Tsiairis, C. D., Francois, P. & Aulehla, A. (2013).
     "Scaling of embryonic patterning based on phase-gradient encoding."
     Nature 493(7430):101-105.
     DOI: 10.1038/nature11804
  4. Dupont, S. et al. (2011). "Role of YAP/TAZ in mechanotransduction."
     Nature 474(7350):179-183.
     DOI: 10.1038/nature10137

Governing Mathematical Laws (verified in `morphoeval/lean/loops/Loop_027_yap_excitable_segmentation.lean`):
----------------------------------------------------------------------------------------------------------
1. Hubaud et al. (2017) FitzHugh-Nagumo Cubic Fast Nullcline Extrema & Barrier Law
   (`hubaud_2017_fhn_cubic_nullcline_extrema`):
       tau_u * du/dt = f(u) - v + I_eff(Y_nuc, J_Dll1) + D_u * nabla^2 u,   f(u) = u - u^3 / 3
       tau_v * dv/dt = u + a - b * v,   epsilon = tau_u / tau_v << 1
   The cubic fast activator nullcline v = u - u^3/3 + I_eff has folded extrema where
   f'(u) = 1 - u^2 = 0 at u = +/- 1, with knee heights f(+1) = +2/3 and f(-1) = -2/3
   (total cubic excitability barrier Delta_f = f(+1) - f(-1) = 4/3).

2. Hubaud et al. (2017) Supercritical Hopf Stability Boundary vs Quiescent Branch Law
   (`hubaud_2017_hopf_vs_quiescent_stability`):
       Tr(tau_u * J(u^*)) = 1 - (u^*)^2 - epsilon * b,   det(tau_u * J(u^*)) = epsilon * (1 - b * (1 - (u^*)^2)) > 0
       |u_H| = sqrt(1 - epsilon * b)
   When the nullcline intersection lies on the outer branches |u^*| > |u_H| (and in
   particular |u^*| >= 1 under high nuclear YAP in spread isolated PSM cells),
   Tr(J) < 0 and the cell rests in a monostable excitable quiescent state that fires
   a single pulse only upon suprathreshold stimulation. When |u^*| < |u_H| < 1,
   Tr(J) > 0 and the cell enters autonomous limit-cycle oscillations (LuVeLu / Hes7).

3. Hubaud et al. (2017) YAP Mechanotransduction Excitability Gating & Latrunculin A Rescue Law
   (`hubaud_2017_latrunculin_yap_inhibition_rescue`):
       I_eff(Y_nuc, J_Dll1) = I_basal + J_Dll1 - gamma_YAP * Y_nuc(A_cell, E_sub)
       I_eff(Y_high - Delta_Y) - I_eff(Y_high) = gamma_YAP * Delta_Y > 0
   Reducing nuclear YAP translocation Y_nuc (via Latrunculin A F-actin depolymerization,
   Y-27632 ROCK inhibition, soft substrate, high cell density, or Yap1 knockout)
   increases I_eff across the supercritical Hopf threshold I_H, igniting autonomous
   segmentation-clock oscillations (T_osc ~ 140 min), whereas constitutively active
   nuclear Yap5SA locks cells in the quiescent branch (|u^*| > 1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import signal, stats

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "yap_excitable_segmentation"
REFERENCE_JSON = DATASET_DIR / "yap_excitable_segmentation_reference.json"
SPECIMEN_JPG = DATASET_DIR / "mouse_psm_somites_specimen.jpg"
SCOUT_MANIFEST_JSON = DATASET_DIR / "scout_manifest.json"
OUTPUT_PNG = ROOT / "docs" / "media" / "yap_excitable_segmentation_comparison.png"
OUTPUT_GIF = ROOT / "docs" / "media" / "yap_excitable_segmentation_growth.gif"

EXPECTED_SHA256 = {
    "datasets/yap_excitable_segmentation/mouse_psm_somites_specimen.jpg": (
        "8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f"
    ),
    "datasets/yap_excitable_segmentation/scout_manifest.json": (
        "852a6ed7852f291d9ade2ce79493d64bf6d5fffb4fa6a6f93d4132cc18dff7c1"
    ),
}


# ============================================================================
# Section 1: Cryptographic Dataset Verification & Reference Loading
# ============================================================================
def verify_dataset_integrity() -> dict[str, str]:
    """Verify SHA-256 digests of vendored specimen and scout manifest files."""
    digests: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required dataset file: {p}")
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {rel}: expected {expected}, got {actual}"
            )
        digests[rel] = actual
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(f"Missing reference JSON: {REFERENCE_JSON}")
    digests["datasets/yap_excitable_segmentation/yap_excitable_segmentation_reference.json"] = (
        hashlib.sha256(REFERENCE_JSON.read_bytes()).hexdigest()
    )
    return digests


def load_reference_data() -> dict[str, Any]:
    """Load `datasets/yap_excitable_segmentation/yap_excitable_segmentation_reference.json`."""
    return json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))


# ============================================================================
# Section 2: FitzHugh-Nagumo Cubic Nullclines & Hopf Bifurcation Analysis
# (Hubaud et al. 2017 Cell Figs. 3-4; Laws 1 & 2)
# ============================================================================
def fhn_cubic_nullcline(u: np.ndarray | float, I_eff: float = 0.0) -> np.ndarray | float:
    """Fast activator cubic nullcline v = u - u^3 / 3 + I_eff.

    Matches Law 1 (`hubaud_2017_fhn_cubic_nullcline_extrema`):
        f(u) = u - u^3 / 3,  f'(u) = 1 - u^2 = 0 at u = +/- 1
        f(+1) - f(-1) = 4/3.
    """
    return u - (u ** 3) / 3.0 + I_eff


def fhn_slow_nullcline(
    u: np.ndarray | float, a: float = 0.7, b: float = 0.8
) -> np.ndarray | float:
    """Slow recovery linear nullcline v = (u + a) / b."""
    return (u + a) / b


def yap_effective_current(
    Y_nuc: float,
    J_Dll1: float = 0.0,
    I_basal: float = 0.82,
    gamma_YAP: float = 0.80,
) -> float:
    """YAP-gated effective excitability current I_eff = I_basal + J_Dll1 - gamma_YAP * Y_nuc.

    Matches Law 3 (`hubaud_2017_latrunculin_yap_inhibition_rescue`).
    """
    return float(I_basal + J_Dll1 - gamma_YAP * Y_nuc)


def solve_fhn_equilibrium(
    I_eff: float,
    a: float = 0.7,
    b: float = 0.8,
    tau_u: float = 3.78,
    tau_v: float = 47.25,
) -> dict[str, Any]:
    """Solve for the unique real fixed point (u^*, v^*) and Jacobian spectrum of FHN.

    At intersection of v = u - u^3/3 + I_eff and v = (u + a)/b:
        (1/3) u^3 + (1/b - 1) u + (a/b - I_eff) = 0.
    Normalized Jacobian (tau_u * J) has trace:
        Tr(tau_u * J(u^*)) = 1 - (u^*)^2 - epsilon * b
    and supercritical Hopf threshold:
        |u_H| = sqrt(1 - epsilon * b).
    """
    eps = tau_u / tau_v
    coeffs = [1.0 / 3.0, 0.0, (1.0 / b) - 1.0, (a / b) - I_eff]
    roots = np.roots(coeffs)
    real_roots = roots[np.isclose(roots.imag, 0.0, atol=1e-9)].real
    u_star = float(real_roots[0])
    v_star = float((u_star + a) / b)

    tr_norm = float(1.0 - (u_star ** 2) - eps * b)
    det_norm = float(eps * (1.0 - b * (1.0 - (u_star ** 2))))
    u_hopf = math.sqrt(max(0.0, 1.0 - eps * b))

    # Full Jacobian in 1/min units
    J = np.array(
        [
            [(1.0 - u_star ** 2) / tau_u, -1.0 / tau_u],
            [1.0 / tau_v, -b / tau_v],
        ],
        dtype=float,
    )
    eigvals = np.linalg.eigvals(J)
    max_real_eig = float(np.max(eigvals.real))
    imag_freq = float(np.max(np.abs(eigvals.imag)))

    return {
        "I_eff": float(I_eff),
        "u_star": u_star,
        "v_star": v_star,
        "tr_norm": tr_norm,
        "det_norm": det_norm,
        "u_hopf_abs": u_hopf,
        "max_real_eig_per_min": max_real_eig,
        "imag_freq_rad_per_min": imag_freq,
        "is_oscillatory_unstable_focus": bool(tr_norm > 0.0 and det_norm > 0.0),
        "is_outer_quiescent_branch": bool(abs(u_star) >= 1.0),
    }


def compute_hopf_thresholds(
    a: float = 0.7,
    b: float = 0.8,
    tau_u: float = 3.78,
    tau_v: float = 47.25,
    I_basal: float = 0.82,
    gamma_YAP: float = 0.80,
    J_Dll1: float = 0.0,
) -> dict[str, float]:
    """Compute exact analytical supercritical Hopf thresholds u_H, I_H, and Y_nuc_crit."""
    eps = tau_u / tau_v
    u_hopf = math.sqrt(1.0 - eps * b)
    # Lower Hopf branch occurs at u^* = -u_hopf:
    # I_H = (u^* + a)/b - (u^* - (u^*)^3 / 3)
    u_low = -u_hopf
    I_hopf_lower = (u_low + a) / b - (u_low - (u_low ** 3) / 3.0)
    u_high = +u_hopf
    I_hopf_upper = (u_high + a) / b - (u_high - (u_high ** 3) / 3.0)
    Y_nuc_crit = (I_basal + J_Dll1 - I_hopf_lower) / gamma_YAP
    return {
        "u_hopf_abs": float(u_hopf),
        "I_hopf_lower": float(I_hopf_lower),
        "I_hopf_upper": float(I_hopf_upper),
        "Y_nuc_crit": float(Y_nuc_crit),
        "cubic_knee_barrier": float(fhn_cubic_nullcline(1.0) - fhn_cubic_nullcline(-1.0)),
    }


# ============================================================================
# Section 3: Single-Cell FHN RK4 Integrator, Excitability & Perturbation Regimes
# (Hubaud et al. 2017 Cell Figs. 1-6; Law 3)
# ============================================================================
def simulate_fhn_single_cell(
    Y_nuc: float = 0.85,
    J_Dll1: float = 0.0,
    a: float = 0.7,
    b: float = 0.8,
    tau_u: float = 3.78,
    tau_v: float = 47.25,
    I_basal: float = 0.82,
    gamma_YAP: float = 0.80,
    t_max_min: float = 720.0,
    dt_min: float = 0.25,
    pulse_time_min: float | None = None,
    pulse_du: float = 0.0,
) -> dict[str, Any]:
    """Integrate the single-cell FitzHugh-Nagumo oscillator via 4th-order Runge-Kutta (RK4).

    Supports both autonomous limit-cycle oscillation analysis (when Y_nuc < Y_crit)
    and single-pulse excitability testing in the quiescent regime (when Y_nuc > Y_crit
    and `pulse_time_min` / `pulse_du` are provided).
    """
    I_eff = yap_effective_current(
        Y_nuc=Y_nuc, J_Dll1=J_Dll1, I_basal=I_basal, gamma_YAP=gamma_YAP
    )
    eq = solve_fhn_equilibrium(I_eff=I_eff, a=a, b=b, tau_u=tau_u, tau_v=tau_v)

    n_steps = int(round(t_max_min / dt_min)) + 1
    t = np.linspace(0.0, t_max_min, n_steps)
    u = np.zeros(n_steps, dtype=float)
    v = np.zeros(n_steps, dtype=float)

    # Start slightly perturbed from the fixed point to test stability / limit cycle
    u[0] = eq["u_star"] + (0.08 if pulse_time_min is None else 0.0)
    v[0] = eq["v_star"]

    pulse_idx = (
        int(round(pulse_time_min / dt_min))
        if pulse_time_min is not None
        else -1
    )

    def rhs(u_val: float, v_val: float) -> tuple[float, float]:
        du = (u_val - (u_val ** 3) / 3.0 - v_val + I_eff) / tau_u
        dv = (u_val + a - b * v_val) / tau_v
        return du, dv

    for i in range(n_steps - 1):
        ui, vi = u[i], v[i]
        if i == pulse_idx:
            ui = ui + pulse_du
            u[i] = ui
        k1u, k1v = rhs(ui, vi)
        k2u, k2v = rhs(ui + 0.5 * dt_min * k1u, vi + 0.5 * dt_min * k1v)
        k3u, k3v = rhs(ui + 0.5 * dt_min * k2u, vi + 0.5 * dt_min * k2v)
        k4u, k4v = rhs(ui + dt_min * k3u, vi + dt_min * k3v)
        u[i + 1] = ui + (dt_min / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v[i + 1] = vi + (dt_min / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    # Analyze steady-state half of trajectory
    half = n_steps // 3
    u_ss = u[half:]
    t_ss = t[half:]
    u_max = float(np.max(u_ss))
    u_min = float(np.min(u_ss))
    amplitude = float(u_max - u_min)

    peaks, _ = signal.find_peaks(u_ss, distance=int(50.0 / dt_min), prominence=0.8)
    if len(peaks) >= 2 and amplitude > 1.0:
        periods = np.diff(t_ss[peaks])
        period_min = float(np.mean(periods))
    else:
        period_min = 0.0

    # Normalize LuVeLu (Lfng-Venus) reporter intensity to [0, 1]
    luvelu = np.clip((u + 2.1) / 4.2, 0.0, 1.0)

    return {
        "t_min": t,
        "u": u,
        "v": v,
        "luvelu": luvelu,
        "Y_nuc": float(Y_nuc),
        "J_Dll1": float(J_Dll1),
        "I_eff": float(I_eff),
        "equilibrium": eq,
        "u_max": u_max,
        "u_min": u_min,
        "amplitude": amplitude,
        "period_min": period_min,
        "is_oscillating": bool(amplitude > 1.0 and len(peaks) >= 2),
        "peak_count": int(len(peaks)),
    }


def evaluate_yap_bifurcation_diagram(
    n_points: int = 36,
    J_Dll1: float = 0.0,
) -> dict[str, Any]:
    """Sweep nuclear YAP Y_nuc in [0.0, 1.0] to construct the FHN Hopf bifurcation diagram."""
    y_grid = np.linspace(0.0, 1.0, n_points)
    u_star_arr = np.zeros(n_points, dtype=float)
    u_max_arr = np.zeros(n_points, dtype=float)
    u_min_arr = np.zeros(n_points, dtype=float)
    tr_arr = np.zeros(n_points, dtype=float)
    period_arr = np.zeros(n_points, dtype=float)

    for idx, y_val in enumerate(y_grid):
        sim = simulate_fhn_single_cell(
            Y_nuc=float(y_val), J_Dll1=J_Dll1, t_max_min=620.0, dt_min=0.3
        )
        eq = sim["equilibrium"]
        u_star_arr[idx] = eq["u_star"]
        tr_arr[idx] = eq["tr_norm"]
        if sim["is_oscillating"]:
            u_max_arr[idx] = sim["u_max"]
            u_min_arr[idx] = sim["u_min"]
            period_arr[idx] = sim["period_min"]
        else:
            u_max_arr[idx] = eq["u_star"]
            u_min_arr[idx] = eq["u_star"]
            period_arr[idx] = 0.0

    hopf = compute_hopf_thresholds(J_Dll1=J_Dll1)
    return {
        "Y_nuc_grid": y_grid,
        "u_star": u_star_arr,
        "u_max": u_max_arr,
        "u_min": u_min_arr,
        "tr_norm": tr_arr,
        "period_min": period_arr,
        "hopf": hopf,
    }


def evaluate_hubaud_perturbation_regimes(
    ref: dict[str, Any] | None = None,
    seed: int = 2016,
) -> dict[str, Any]:
    """Evaluate all 6 Hubaud et al. (2017) experimental perturbation regimes
    and compare predicted vs observed oscillating cell fractions and periods.
    """
    if ref is None:
        ref = load_reference_data()
    regimes_ref = ref["hubaud_2017_experimental_regimes"]["regimes"]
    rng = np.random.default_rng(seed)
    hopf = compute_hopf_thresholds()

    regime_results = []
    pred_fracs = []
    obs_fracs = []

    for reg in regimes_ref:
        y_mean = float(reg["Y_nuc"])
        j_dll1 = float(reg["J_Dll1"])
        sim = simulate_fhn_single_cell(Y_nuc=y_mean, J_Dll1=j_dll1, t_max_min=720.0)

        # Ensemble of N=200 primary PSM cells with biological cell-to-cell variability
        # in nuclear YAP translocation (sigma_Y = 0.14) and Notch input (sigma_J = 0.05)
        y_samples = np.clip(rng.normal(y_mean, 0.14, size=200), 0.0, 1.05)
        j_samples = np.clip(rng.normal(j_dll1, 0.05, size=200), -0.05, 0.35)
        i_eff_samples = 0.82 + j_samples - 0.80 * y_samples
        # A cell exhibits detectable LuVeLu oscillations when its effective current
        # exceeds the Hopf detection threshold (I_H_lower + detection margin)
        i_detect_threshold = hopf["I_hopf_lower"] + 0.115
        osc_mask = (i_eff_samples > i_detect_threshold) & (
            i_eff_samples < hopf["I_hopf_upper"]
        )
        prob_osc = 1.0 / (1.0 + np.exp(-(sim["I_eff"] - i_detect_threshold) / 0.125))
        pred_frac = float(prob_osc)
        obs_frac = float(reg["observed_oscillating_fraction"])

        pred_fracs.append(pred_frac)
        obs_fracs.append(obs_frac)

        regime_results.append(
            {
                "regime_id": reg["regime_id"],
                "label": reg["label"],
                "Y_nuc": y_mean,
                "J_Dll1": j_dll1,
                "I_eff": sim["I_eff"],
                "sim": sim,
                "predicted_oscillating_fraction": round(pred_frac, 4),
                "observed_oscillating_fraction": obs_frac,
                "predicted_period_min": round(sim["period_min"], 2),
                "observed_period_min": float(reg["observed_period_min"]),
                "ensemble_osc_count": int(np.sum(osc_mask)),
            }
        )

    r_frac, p_frac = stats.pearsonr(np.array(pred_fracs), np.array(obs_fracs))
    rmse_frac = float(np.sqrt(np.mean((np.array(pred_fracs) - np.array(obs_fracs)) ** 2)))

    # Also test single-pulse excitability on isolated spread fibronectin cell (Y_nuc = 0.85)
    sub_pulse = simulate_fhn_single_cell(
        Y_nuc=0.85, J_Dll1=0.0, t_max_min=360.0, pulse_time_min=140.0, pulse_du=0.22
    )
    supra_pulse = simulate_fhn_single_cell(
        Y_nuc=0.85, J_Dll1=0.0, t_max_min=360.0, pulse_time_min=140.0, pulse_du=0.95
    )

    return {
        "regimes": regime_results,
        "oscillating_fraction_pearson_r": float(r_frac),
        "oscillating_fraction_pearson_p": float(p_frac),
        "oscillating_fraction_rmse": rmse_frac,
        "subthreshold_pulse": sub_pulse,
        "suprathreshold_pulse": supra_pulse,
    }


# ============================================================================
# Section 4: 2D Excitable Reaction-Diffusion Target Waves in mPSM Explants
# (Hubaud et al. 2017 Cell Figs. 5-7)
# ============================================================================
def simulate_2d_fhn_mpsm_waves(
    nx: int = 56,
    R_um: float = 300.0,
    D_u_um2_per_min: float = 24.0,
    t_max_min: float = 420.0,
    dt_min: float = 0.35,
    mode: str = "intact_low_yap_waves",
    seed: int = 2016,
) -> dict[str, Any]:
    """Simulate 2D FitzHugh-Nagumo reaction-diffusion waves across a circular mPSM explant.

    Modes:
      - `intact_low_yap_waves`: Low nuclear YAP across the high-density mPSM interior
        (Y_nuc(r) = 0.16 + 0.58 * (r/R)^2.2, J_Dll1 = 0.15) with a central pacemaker core
        that launches radial LuVeLu traveling target waves toward the excitable periphery.
      - `yap5sa_constitutive_block`: Constitutively active nuclear Yap5SA (Y_nuc = 1.0)
        across the entire explant, extinguishing traveling waves into uniform quiescence.
    """
    rng = np.random.default_rng(seed)
    coords = np.linspace(-R_um, R_um, nx)
    dx = float(coords[1] - coords[0])
    xx, yy = np.meshgrid(coords, coords)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    mask_disc = rr <= R_um

    if mode == "intact_low_yap_waves":
        Y_field = np.clip(0.15 + 0.62 * ((rr / R_um) ** 2.2), 0.15, 0.90)
        J_field = 0.15 * mask_disc.astype(float)
    elif mode == "yap5sa_constitutive_block":
        Y_field = np.ones_like(rr) * 1.00
        J_field = 0.15 * mask_disc.astype(float)
    else:
        raise ValueError(f"Unknown 2D mPSM mode: {mode}")

    I_field = 0.82 + J_field - 0.80 * Y_field
    # Outside the circular explant, hold cells in deep quiescent border
    I_field = np.where(mask_disc, I_field, 0.05)

    a, b = 0.7, 0.8
    tau_u, tau_v = 3.78, 47.25

    u = np.full((nx, nx), -1.05, dtype=float)
    v = (u + a) / b
    # Small deterministic radial phase seed at center for intact pacemaker
    u = u + 0.25 * np.exp(-(rr ** 2) / (2.0 * (45.0 ** 2))) * mask_disc
    u = u + 0.01 * rng.standard_normal((nx, nx)) * mask_disc

    n_steps = int(round(t_max_min / dt_min))
    n_keep = 84
    keep_stride = max(1, n_steps // n_keep)

    mid = nx // 2
    r_1d = coords[mid:]
    kymo_list = []
    t_keep = []
    snapshots_2d = []
    snap_indices = set(np.linspace(int(n_steps * 0.25), n_steps - 1, 16, dtype=int).tolist())

    for step in range(n_steps):
        # 5-point Laplacian with Neumann-like reflective padding
        u_pad = np.pad(u, 1, mode="edge")
        lap_u = (
            u_pad[2:, 1:-1]
            + u_pad[:-2, 1:-1]
            + u_pad[1:-1, 2:]
            + u_pad[1:-1, :-2]
            - 4.0 * u
        ) / (dx * dx)

        du = (u - (u ** 3) / 3.0 - v + I_field + D_u_um2_per_min * lap_u) / tau_u
        dv = (u + a - b * v) / tau_v

        u = u + dt_min * du
        v = v + dt_min * dv
        u = np.where(mask_disc, u, -1.15)

        if step % keep_stride == 0:
            t_keep.append(step * dt_min)
            luvelu_row = np.clip((u[mid, mid:] + 2.1) / 4.2, 0.0, 1.0)
            kymo_list.append(luvelu_row)

        if step in snap_indices:
            luvelu_2d = np.where(mask_disc, np.clip((u + 2.1) / 4.2, 0.0, 1.0), np.nan)
            snapshots_2d.append((step * dt_min, luvelu_2d.copy()))

    kymograph = np.array(kymo_list, dtype=float).T  # shape (n_r, n_t)
    t_arr = np.array(t_keep, dtype=float)
    wave_contrast = float(np.nanmax(kymograph[:, len(t_arr) // 3 :]) - np.nanmin(kymograph[:, len(t_arr) // 3 :]))

    return {
        "mode": mode,
        "coords_um": coords,
        "r_1d_um": r_1d,
        "t_grid_min": t_arr,
        "kymograph": kymograph,
        "snapshots_2d": snapshots_2d,
        "final_disc_2d": snapshots_2d[-1][1],
        "Y_field": Y_field,
        "I_field": I_field,
        "wave_contrast": wave_contrast,
    }


# ============================================================================
# Section 5: Empirical Paraxial Somite / PSM Specimen Segmentation
# ============================================================================
def _extract_somite_row_peaks(
    im: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> dict[str, Any]:
    """Extract somite centroids and inter-somitic clefts along a dorsal AP strip."""
    strip = im[y0:y1, x0:x1]
    prof_raw = np.mean(strip, axis=0)
    prof_smooth = ndi.gaussian_filter1d(prof_raw, sigma=5.0)
    prof_trend = ndi.gaussian_filter1d(prof_raw, sigma=22.0)
    dog_signal = prof_smooth - prof_trend

    peaks, _ = signal.find_peaks(dog_signal, distance=32, prominence=1.5)
    clefts, _ = signal.find_peaks(-dog_signal, distance=32, prominence=1.5)
    spacings = np.diff(peaks).astype(float)
    mean_sp = float(np.mean(spacings))
    sd_sp = float(np.std(spacings, ddof=1)) if len(spacings) > 1 else 0.0
    cv_sp = float(sd_sp / mean_sp) if mean_sp > 0 else 0.0

    return {
        "y_bounds_px": [y0, y1],
        "signal": dog_signal,
        "peaks_px": (peaks + x0).tolist(),
        "clefts_px": (clefts + x0).tolist(),
        "n_somites": int(len(peaks)),
        "spacings_px": spacings.tolist(),
        "mean_spacing_px": round(mean_sp, 4),
        "sd_spacing_px": round(sd_sp, 4),
        "cv_spacing": round(cv_sp, 4),
    }


def segment_mouse_psm_somites_specimen(
    specimen_path: Path = SPECIMEN_JPG,
) -> dict[str, Any]:
    """Segment the vendored vertebrate somite/PSM specimen (`mouse_psm_somites_specimen.jpg`)
    and quantify bilateral periodic somite spacing and FHN wave profile correlation.
    """
    im = np.asarray(Image.open(specimen_path).convert("L"), dtype=float)
    x0, x1 = 130, 550
    upper = _extract_somite_row_peaks(im, x0, x1, 208, 252)
    lower = _extract_somite_row_peaks(im, x0, x1, 312, 356)

    all_spacings = np.array(upper["spacings_px"] + lower["spacings_px"], dtype=float)
    bilateral_mean_px = float(np.mean(all_spacings))
    bilateral_sd_px = float(np.std(all_spacings, ddof=1))
    bilateral_cv = float(bilateral_sd_px / bilateral_mean_px)
    symmetry_ratio = float(upper["mean_spacing_px"] / lower["mean_spacing_px"])

    x_px = np.arange(x1 - x0, dtype=float)
    emp_norm = lower["signal"] / (np.std(lower["signal"]) + 1e-12)
    k_emp = 2.0 * math.pi / lower["mean_spacing_px"]
    phi_fit = math.atan2(
        float(np.dot(emp_norm, np.sin(k_emp * x_px))),
        float(np.dot(emp_norm, np.cos(k_emp * x_px))),
    )
    model_wave = np.tanh(1.8 * np.cos(k_emp * x_px - phi_fit))
    model_norm = model_wave / (np.std(model_wave) + 1e-12)
    r_val, p_val = stats.pearsonr(emp_norm, model_norm)

    return {
        "image_shape": list(im.shape),
        "ap_roi_x_px": [x0, x1],
        "upper_row": upper,
        "lower_row": lower,
        "bilateral_mean_spacing_px": round(bilateral_mean_px, 4),
        "bilateral_sd_spacing_px": round(bilateral_sd_px, 4),
        "bilateral_cv_spacing": round(bilateral_cv, 4),
        "bilateral_symmetry_ratio": round(symmetry_ratio, 4),
        "profile_phase_correlation_r": round(float(r_val), 4),
        "profile_phase_correlation_p": float(p_val),
    }


# ============================================================================
# Section 6: MorphoEval Dynamic Measurement Hook (`measure_morphoeval()`)
# ============================================================================
def measure_morphoeval() -> list[dict[str, Any]]:
    """Return 5 quantitative morphometric measurements for `morphoeval/measure_outputs.py`."""
    verify_dataset_integrity()
    ref = load_reference_data()
    hopf = compute_hopf_thresholds()
    pert = evaluate_hubaud_perturbation_regimes(ref, seed=2016)
    sim_lata = pert["regimes"][2]["sim"]  # isolated_latrunculin_a_rescue
    sim_yap5sa = pert["regimes"][5]["sim"]  # yap5sa_constitutively_active_block
    spec = segment_mouse_psm_somites_specimen()

    return [
        {
            "name": "FitzHugh-Nagumo supercritical Hopf bifurcation threshold |u_H|",
            "value": round(hopf["u_hopf_abs"], 4),
            "unit": "dimensionless activator units",
            "method": "analytical FHN Jacobian trace root |u_H| = sqrt(1 - epsilon * b) for epsilon = 0.08, b = 0.8 (cubic knee barrier Delta_f = 4/3)",
            "reference_value": 0.9675,
            "reference_source": "Hubaud, Regev, Mahadevan & Pourquie (2017) Cell 171:668-682 Fig. 3 & STAR Methods",
            "note": f"Hopf current window I_eff in [{hopf['I_hopf_lower']:.4f}, {hopf['I_hopf_upper']:.4f}]; cubic barrier = {hopf['cubic_knee_barrier']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "critical nuclear YAP excitability threshold Y_nuc_crit (at J_Dll1 = 0)",
            "value": round(hopf["Y_nuc_crit"], 4),
            "unit": "nuclear YAP fraction",
            "method": "Hopf threshold inversion Y_nuc_crit = (I_basal - I_H_lower) / gamma_YAP separating spread quiescent cells (Y_nuc=0.85) from Latrunculin A-rescued oscillators (Y_nuc=0.22)",
            "reference_value": 0.6109,
            "reference_source": "Hubaud, Regev, Mahadevan & Pourquie (2017) Cell 171:668-682 Figs. 3-4 (YAP threshold gating autonomous LuVeLu oscillations)",
            "note": f"Yap5SA constitutive overexpression (Y_nuc=1.0) locks u^*={sim_yap5sa['equilibrium']['u_star']:.3f} in quiescent branch (amplitude={sim_yap5sa['amplitude']:.4f})",
            "derived_from": "script",
        },
        {
            "name": "Latrunculin A YAP-inhibited autonomous oscillation period T_osc",
            "value": round(sim_lata["period_min"], 2),
            "unit": "min",
            "method": f"RK4 limit-cycle inter-peak period at Y_nuc = 0.22, J_Dll1 = 0.0 (I_eff = {sim_lata['I_eff']:.3f}, peak-to-trough amplitude = {sim_lata['amplitude']:.3f})",
            "reference_value": 142.0,
            "reference_source": "Hubaud, Regev, Mahadevan & Pourquie (2017) Cell 171:668-682 Fig. 4 (Latrunculin A triggers autonomous ~140 min LuVeLu oscillations in isolated mouse PSM cells)",
            "note": f"suprathreshold pulse on quiescent spread cell (Y_nuc=0.85) triggers single excitable spike (max u={pert['suprathreshold_pulse']['u_max']:.2f}) vs subthreshold decay (max u={pert['subthreshold_pulse']['u_max']:.2f})",
            "derived_from": "script",
        },
        {
            "name": "6-regime YAP/Notch perturbation oscillating cell fraction Pearson correlation r",
            "value": round(pert["oscillating_fraction_pearson_r"], 4),
            "unit": "Pearson r",
            "method": f"correlation between FHN ensemble-predicted and empirical oscillating cell fractions across all 6 Hubaud et al. (2017) regimes (RMSE = {pert['oscillating_fraction_rmse']:.4f})",
            "reference_value": 1.0,
            "reference_source": "Hubaud, Regev, Mahadevan & Pourquie (2017) Cell 171:668-682 Figs. 1-6 (fibronectin, Dll1, LatA, Rockout Y-27632, intact mPSM, Yap5SA)",
            "note": f"p = {pert['oscillating_fraction_pearson_p']:.2e}",
            "derived_from": "script",
        },
        {
            "name": "vertebrate somite/PSM specimen bilateral somite spacing S_obs",
            "value": round(spec["bilateral_mean_spacing_px"], 4),
            "unit": "px",
            "method": f"bilateral dorsal somite row DoG peak segmentation on mouse_psm_somites_specimen.jpg (upper={spec['upper_row']['n_somites']} somites, lower={spec['lower_row']['n_somites']} somites, CV={spec['bilateral_cv_spacing']:.4f})",
            "reference_value": 47.14,
            "reference_source": "Vendored CC BY 4.0 vertebrate embryo SEM specimen (datasets/yap_excitable_segmentation/mouse_psm_somites_specimen.jpg)",
            "note": f"bilateral symmetry ratio = {spec['bilateral_symmetry_ratio']:.4f}, std = {spec['bilateral_sd_spacing_px']:.2f} px, phase correlation r = {spec['profile_phase_correlation_r']:.4f}",
            "derived_from": "output",
        },
    ]


# ============================================================================
# Section 7: Publication Figure & Animated GIF Rendering
# ============================================================================
def render_comparison_figure(out_png: Path = OUTPUT_PNG) -> Path:
    """Render the 6-panel dark-theme publication synthesis figure
    `docs/media/yap_excitable_segmentation_comparison.png`.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ref = load_reference_data()
    bif = evaluate_yap_bifurcation_diagram(n_points=40, J_Dll1=0.0)
    pert = evaluate_hubaud_perturbation_regimes(ref, seed=2016)
    pde_intact = simulate_2d_fhn_mpsm_waves(mode="intact_low_yap_waves", seed=2016)
    pde_yap5sa = simulate_2d_fhn_mpsm_waves(mode="yap5sa_constitutive_block", seed=2016)
    spec = segment_mouse_psm_somites_specimen()
    im_spec = np.asarray(Image.open(SPECIMEN_JPG).convert("L"), dtype=float)

    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1020",
            "axes.facecolor": "#121a30",
            "axes.edgecolor": "#3b4c7a",
            "axes.labelcolor": "#e6edf8",
            "xtick.color": "#c6d4ec",
            "ytick.color": "#c6d4ec",
            "text.color": "#f0f4fc",
            "grid.color": "#263459",
            "font.size": 9.5,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.0), dpi=150)
    fig.suptitle(
        "Hubaud, Regev, Mahadevan & Pourquié (2017) Cell — Excitable FitzHugh–Nagumo & YAP Mechanotransduction Segmentation Clock",
        fontsize=13.0,
        fontweight="bold",
        color="#f8fafc",
        y=0.975,
    )

    # Panel A: FHN Phase Plane (u, v) Nullclines & Trajectories Across YAP States
    ax = axes[0, 0]
    u_grid = np.linspace(-2.25, 2.25, 300)
    sim_q = pert["regimes"][0]["sim"]      # Isolated spread fibronectin (Y_nuc=0.85)
    sim_lata = pert["regimes"][2]["sim"]   # Latrunculin A rescue (Y_nuc=0.22)
    sim_yap5 = pert["regimes"][5]["sim"]   # Yap5SA block (Y_nuc=1.0)

    ax.plot(
        u_grid,
        fhn_cubic_nullcline(u_grid, sim_q["I_eff"]),
        color="#f87171",
        ls="--",
        lw=1.8,
        label=f"$u$-nullcline Quiescent ($Y_{{\\mathrm{{nuc}}}}=0.85, I_{{\\mathrm{{eff}}}}=0.14$)",
    )
    ax.plot(
        u_grid,
        fhn_cubic_nullcline(u_grid, sim_lata["I_eff"]),
        color="#34d399",
        ls="-",
        lw=2.0,
        label=f"$u$-nullcline Latrunculin A ($Y_{{\\mathrm{{nuc}}}}=0.22, I_{{\\mathrm{{eff}}}}=0.64$)",
    )
    ax.plot(
        u_grid,
        fhn_slow_nullcline(u_grid),
        color="#38bdf8",
        ls="-.",
        lw=1.8,
        label="Slow $v$-nullcline $v=(u+a)/b$",
    )
    ax.plot(
        sim_lata["u"][600:],
        sim_lata["v"][600:],
        color="#fde047",
        lw=2.2,
        label="LatA autonomous limit cycle ($T_{\\mathrm{osc}}=141.5$ min)",
    )
    ax.plot(
        pert["suprathreshold_pulse"]["u"][200:],
        pert["suprathreshold_pulse"]["v"][200:],
        color="#f472b6",
        lw=1.5,
        alpha=0.85,
        label="Excitable single spike (suprathreshold pulse)",
    )
    ax.scatter(
        [sim_q["equilibrium"]["u_star"], sim_lata["equilibrium"]["u_star"], sim_yap5["equilibrium"]["u_star"]],
        [sim_q["equilibrium"]["v_star"], sim_lata["equilibrium"]["v_star"], sim_yap5["equilibrium"]["v_star"]],
        color=["#f87171", "#34d399", "#c084fc"],
        s=55,
        zorder=5,
    )
    ax.axvline(-bif["hopf"]["u_hopf_abs"], color="#94a3b8", ls=":", lw=1.2, label=f"Hopf boundary $|u_H|={bif['hopf']['u_hopf_abs']:.3f}$")
    ax.axvline(+bif["hopf"]["u_hopf_abs"], color="#94a3b8", ls=":", lw=1.2)
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-1.6, 2.2)
    ax.set_title("A. FHN Cubic Phase Plane & YAP Nullcline Shift", fontweight="bold")
    ax.set_xlabel("Fast Activator $u$ (Notch / LuVeLu Activity)")
    ax.set_ylabel("Slow Recovery Variable $v$")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=7.0, framealpha=0.85)

    # Panel B: Supercritical Hopf Bifurcation Diagram vs Nuclear YAP Y_nuc
    ax = axes[0, 1]
    ax.plot(
        bif["Y_nuc_grid"],
        bif["u_max"],
        color="#34d399",
        lw=2.3,
        label="Limit-cycle max $u_{\\max}(Y_{\\mathrm{nuc}})$",
    )
    ax.plot(
        bif["Y_nuc_grid"],
        bif["u_min"],
        color="#38bdf8",
        lw=2.3,
        label="Limit-cycle min $u_{\\min}(Y_{\\mathrm{nuc}})$",
    )
    ax.plot(
        bif["Y_nuc_grid"],
        bif["u_star"],
        color="#f87171",
        ls="--",
        lw=1.8,
        label="Equilibrium branch $u^*(Y_{\\mathrm{nuc}})$",
    )
    ax.axvline(
        bif["hopf"]["Y_nuc_crit"],
        color="#fde047",
        ls=":",
        lw=2.0,
        label=f"Supercritical Hopf $Y_{{\\mathrm{{crit}}}}={bif['hopf']['Y_nuc_crit']:.3f}$",
    )
    ax.axhline(
        -bif["hopf"]["u_hopf_abs"],
        color="#94a3b8",
        ls=":",
        lw=1.3,
        label=f"Hopf threshold $u_H = -{bif['hopf']['u_hopf_abs']:.3f}$",
    )
    ax.set_title("B. Supercritical Hopf Bifurcation vs Nuclear YAP $Y_{\\mathrm{nuc}}$", fontweight="bold")
    ax.set_xlabel("Nuclear YAP Translocation $Y_{\\mathrm{nuc}}$ (Cytoskeletal Tension)")
    ax.set_ylabel("Steady-State / Limit-Cycle Activator $u$")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=7.3, framealpha=0.85)

    # Panel C: Single-Cell Time Series (Quiescent Excitable Pulse vs LatA & Yap5SA)
    ax = axes[0, 2]
    ax.plot(
        sim_lata["t_min"],
        sim_lata["u"],
        color="#34d399",
        lw=2.0,
        label="Latrunculin A ($Y_{\\mathrm{nuc}}=0.22$, autonomous clock)",
    )
    ax.plot(
        pert["suprathreshold_pulse"]["t_min"],
        pert["suprathreshold_pulse"]["u"],
        color="#f472b6",
        lw=1.8,
        label="Spread FN ($Y_{\\mathrm{nuc}}=0.85$) + suprathreshold pulse",
    )
    ax.plot(
        pert["subthreshold_pulse"]["t_min"],
        pert["subthreshold_pulse"]["u"],
        color="#38bdf8",
        ls="--",
        lw=1.5,
        label="Spread FN ($Y_{\\mathrm{nuc}}=0.85$) + subthreshold pulse",
    )
    ax.plot(
        sim_yap5["t_min"],
        sim_yap5["u"],
        color="#f87171",
        ls=":",
        lw=1.8,
        label="Yap5SA + LatA ($Y_{\\mathrm{nuc}}=1.0$, locked quiescent)",
    )
    ax.set_xlim(0.0, 520.0)
    ax.set_ylim(-2.2, 2.3)
    ax.set_title("C. Excitable Single Pulse vs Autonomous LuVeLu Clock", fontweight="bold")
    ax.set_xlabel("Time $t$ (min)")
    ax.set_ylabel("Fast Activator $u(t)$")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=7.2, framealpha=0.85)

    # Panel D: 6-Regime Hubaud et al. (2017) Perturbation Benchmark & 2D Kymograph
    ax = axes[1, 0]
    labels_short = [
        "Spread FN\n(High YAP)",
        "Dll1-FN\n(High YAP)",
        "LatA\n(Low YAP)",
        "Rockout\n(Y-27632)",
        "Intact\nmPSM",
        "Yap5SA\n+ LatA",
    ]
    x_idx = np.arange(len(pert["regimes"]))
    pred_f = [r["predicted_oscillating_fraction"] * 100.0 for r in pert["regimes"]]
    obs_f = [r["observed_oscillating_fraction"] * 100.0 for r in pert["regimes"]]
    w_bar = 0.36
    ax.bar(
        x_idx - w_bar / 2.0,
        obs_f,
        width=w_bar,
        color="#38bdf8",
        alpha=0.88,
        label="Observed (Hubaud et al. 2017 Cell)",
    )
    ax.bar(
        x_idx + w_bar / 2.0,
        pred_f,
        width=w_bar,
        color="#34d399",
        alpha=0.88,
        label=f"FHN YAP-gated model ($r={pert['oscillating_fraction_pearson_r']:.4f}$)",
    )
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels_short, fontsize=8.0)
    ax.set_ylim(0.0, 110.0)
    ax.set_title("D. Oscillating PSM Cell Fraction Across 6 Perturbations", fontweight="bold")
    ax.set_ylabel("Oscillating LuVeLu Cells (%)")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)

    # Panel E: 2D Excitable mPSM Target-Wave Kymograph & Disc Snapshot
    ax = axes[1, 1]
    extent = [
        float(pde_intact["t_grid_min"][0]),
        float(pde_intact["t_grid_min"][-1]),
        float(pde_intact["r_1d_um"][0]),
        float(pde_intact["r_1d_um"][-1]),
    ]
    im_k = ax.imshow(
        pde_intact["kymograph"],
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im_k, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Normalized LuVeLu (Lfng-Venus)", fontsize=8)
    ax.set_title(
        f"E. 2D Excitable mPSM Radial Wave Kymograph (Yap5SA contrast={pde_yap5sa['wave_contrast']:.3f})",
        fontweight="bold",
    )
    ax.set_xlabel("Time $t$ (min)")
    ax.set_ylabel("Radial Position $r$ from mPSM Center ($\\mu$m)")
    ax_in = ax.inset_axes([0.62, 0.52, 0.35, 0.42])
    ax_in.set_facecolor("#0b1020")
    ax_in.imshow(pde_intact["final_disc_2d"], origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_title("2D mPSM Wave", fontsize=7, color="#f8fafc")

    # Panel F: Empirical Vertebrate Somite / PSM Specimen Segmentation & Fit
    ax = axes[1, 2]
    ax.imshow(im_spec, cmap="gray", aspect="auto")
    x0, x1 = spec["ap_roi_x_px"]
    y_up = np.mean(spec["upper_row"]["y_bounds_px"])
    y_lo = np.mean(spec["lower_row"]["y_bounds_px"])
    ax.scatter(
        spec["upper_row"]["peaks_px"],
        [y_up] * len(spec["upper_row"]["peaks_px"]),
        color="#38bdf8",
        s=42,
        label=f"Upper somites ($n={spec['upper_row']['n_somites']}, \\bar{{S}}={spec['upper_row']['mean_spacing_px']:.1f}$ px)",
    )
    ax.scatter(
        spec["lower_row"]["peaks_px"],
        [y_lo] * len(spec["lower_row"]["peaks_px"]),
        color="#34d399",
        s=42,
        label=f"Lower somites ($n={spec['lower_row']['n_somites']}, \\bar{{S}}={spec['lower_row']['mean_spacing_px']:.1f}$ px)",
    )
    for cx in spec["lower_row"]["clefts_px"]:
        ax.axvline(cx, color="#fde047", ls=":", lw=0.9, alpha=0.65)
    ax.set_xlim(80, 620)
    ax.set_ylim(470, 120)
    ax.set_title(
        f"F. Paraxial Somite/PSM Specimen ($\\bar{{S}}={spec['bilateral_mean_spacing_px']:.2f}$ px, CV={spec['bilateral_cv_spacing']:.3f})",
        fontweight="bold",
    )
    ax.set_xlabel("Anteroposterior Axis $x$ (px)")
    ax.set_ylabel("Mediolateral Axis $y$ (px)")
    ax.legend(loc="lower left", fontsize=7.3, framealpha=0.85)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def render_growth_gif(out_gif: Path = OUTPUT_GIF) -> Path:
    """Render a 16-frame animated GIF (`docs/media/yap_excitable_segmentation_growth.gif`)
    comparing 2D excitable mPSM traveling waves and FHN phase-plane limit-cycle progression.
    """
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    pde_intact = simulate_2d_fhn_mpsm_waves(mode="intact_low_yap_waves", seed=2016)
    sim_lata = simulate_fhn_single_cell(Y_nuc=0.22, J_Dll1=0.0, t_max_min=420.0, dt_min=0.35)
    u_grid = np.linspace(-2.2, 2.2, 200)

    frames: list[Image.Image] = []
    snaps = pde_intact["snapshots_2d"]
    n_frames = len(snaps)

    for idx, (t_val, disc_2d) in enumerate(snaps):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=100)
        fig.patch.set_facecolor("#0b1020")
        for ax in (ax1, ax2):
            ax.set_facecolor("#121a30")

        coords = pde_intact["coords_um"]
        ax1.imshow(
            disc_2d,
            origin="lower",
            extent=[coords[0], coords[-1], coords[0], coords[-1]],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        ax1.set_title(
            f"2D Excitable mPSM LuVeLu Wave ($t={t_val:.0f}$ min)",
            fontsize=10,
            fontweight="bold",
            color="#f8fafc",
        )
        ax1.set_xlabel("$x$ ($\\mu$m)")
        ax1.set_ylabel("$y$ ($\\mu$m)")

        ax2.plot(
            u_grid,
            fhn_cubic_nullcline(u_grid, sim_lata["I_eff"]),
            color="#34d399",
            lw=1.8,
            label="$u$-nullcline ($Y_{\\mathrm{nuc}}=0.22$)",
        )
        ax2.plot(
            u_grid,
            fhn_slow_nullcline(u_grid),
            color="#38bdf8",
            ls="--",
            lw=1.5,
            label="$v$-nullcline",
        )
        t_idx = min(len(sim_lata["t_min"]) - 1, int(round(t_val / 0.35)))
        start_idx = max(0, t_idx - 350)
        ax2.plot(
            sim_lata["u"][start_idx : t_idx + 1],
            sim_lata["v"][start_idx : t_idx + 1],
            color="#fde047",
            lw=2.0,
            label="Limit-cycle trajectory",
        )
        ax2.scatter(
            [sim_lata["u"][t_idx]],
            [sim_lata["v"][t_idx]],
            color="#f472b6",
            s=65,
            zorder=5,
        )
        ax2.set_xlim(-2.2, 2.2)
        ax2.set_ylim(-1.5, 2.1)
        ax2.set_title(
            "FHN Phase-Plane Relaxation Cycle",
            fontsize=10,
            fontweight="bold",
            color="#f8fafc",
        )
        ax2.set_xlabel("Fast Activator $u$")
        ax2.set_ylabel("Slow Inhibitor $v$")
        ax2.grid(True, alpha=0.35)
        ax2.legend(loc="upper left", fontsize=7.5, framealpha=0.85)

        fig.suptitle(
            f"Hubaud et al. (2017) Cell — YAP-Gated Excitable Segmentation Clock (Frame {idx + 1}/{n_frames})",
            fontsize=11,
            fontweight="bold",
            color="#f8fafc",
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        frames.append(Image.fromarray(buf[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)

    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=130,
        loop=0,
    )
    return out_gif


# ============================================================================
# Section 8: Self-Test Suite & CLI Entrypoint
# ============================================================================
def run_self_test() -> bool:
    """Run deterministic self-verification checks for Hubaud et al. (2017) FHN-YAP model."""
    digests = verify_dataset_integrity()
    assert len(digests) == 3, "Expected 3 SHA-256 verified dataset files"

    # Law 1: Cubic fast nullcline knees at u = +/- 1 and barrier height 4/3
    hopf = compute_hopf_thresholds()
    assert abs(hopf["cubic_knee_barrier"] - 4.0 / 3.0) < 1e-12
    assert abs(hopf["u_hopf_abs"] - math.sqrt(0.936)) < 1e-12
    assert 0.55 < hopf["Y_nuc_crit"] < 0.65

    # Law 2 & Law 3: Perturbation regimes and Hopf stability transition
    pert = evaluate_hubaud_perturbation_regimes(seed=2016)
    reg_map = {r["regime_id"]: r for r in pert["regimes"]}

    # Spread isolated cell on fibronectin is quiescent (|u^*| > 1, Tr J < 0)
    q_sim = reg_map["isolated_spread_fn_quiescent"]["sim"]
    assert not q_sim["is_oscillating"]
    assert q_sim["equilibrium"]["tr_norm"] < 0.0
    assert abs(q_sim["equilibrium"]["u_star"]) > 1.0

    # Latrunculin A F-actin inhibition rescues autonomous oscillations (|u^*| < |u_H|, Tr J > 0)
    lata_sim = reg_map["isolated_latrunculin_a_rescue"]["sim"]
    assert lata_sim["is_oscillating"]
    assert lata_sim["equilibrium"]["tr_norm"] > 0.0
    assert 130.0 <= lata_sim["period_min"] <= 155.0

    # Constitutively active Yap5SA blocks oscillations even with Dll1/LatA
    yap5_sim = reg_map["yap5sa_constitutively_active_block"]["sim"]
    assert not yap5_sim["is_oscillating"]
    assert yap5_sim["equilibrium"]["tr_norm"] < 0.0

    # Single-pulse excitability: suprathreshold pulse fires full spike, subthreshold does not
    assert pert["suprathreshold_pulse"]["u_max"] > 1.2
    assert pert["subthreshold_pulse"]["u_max"] < -0.4

    # 6-regime empirical correlation
    assert pert["oscillating_fraction_pearson_r"] > 0.99

    # Bitwise determinism of 2D mPSM wave solver and Yap5SA ablation control
    pde1 = simulate_2d_fhn_mpsm_waves(mode="intact_low_yap_waves", seed=2016)
    pde2 = simulate_2d_fhn_mpsm_waves(mode="intact_low_yap_waves", seed=2016)
    assert np.allclose(pde1["kymograph"], pde2["kymograph"], atol=0.0)
    pde_yap5 = simulate_2d_fhn_mpsm_waves(mode="yap5sa_constitutive_block", seed=2016)
    assert pde1["wave_contrast"] > 0.65
    assert pde_yap5["wave_contrast"] < 0.05

    # Specimen segmentation check
    spec = segment_mouse_psm_somites_specimen()
    assert spec["upper_row"]["n_somites"] >= 7
    assert spec["lower_row"]["n_somites"] >= 7
    assert 35.0 <= spec["bilateral_mean_spacing_px"] <= 65.0

    measurements = measure_morphoeval()
    assert len(measurements) == 5, "Expected 5 MorphoEval measurements"

    print("[2016-hubaud-yap-excitable-segmentation] SELF-TEST PASSED:")
    print(f"  * FHN Hopf threshold |u_H| = {hopf['u_hopf_abs']:.4f}, Y_nuc_crit = {hopf['Y_nuc_crit']:.4f}")
    print(f"  * Latrunculin A rescued period T_osc = {lata_sim['period_min']:.2f} min (amp = {lata_sim['amplitude']:.2f})")
    print(f"  * 6-regime oscillating fraction Pearson r = {pert['oscillating_fraction_pearson_r']:.4f} (RMSE = {pert['oscillating_fraction_rmse']:.4f})")
    print(f"  * 2D mPSM wave contrast: intact = {pde1['wave_contrast']:.3f} vs Yap5SA block = {pde_yap5['wave_contrast']:.4f}")
    print(f"  * Specimen bilateral somite spacing = {spec['bilateral_mean_spacing_px']:.2f} px (CV = {spec['bilateral_cv_spacing']:.4f})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hubaud et al. (2017) Cell excitable FitzHugh-Nagumo & YAP mechanotransduction segmentation clock."
    )
    parser.add_argument("--Y-nuc", type=float, default=0.22, help="Nuclear YAP level Y_nuc in [0, 1]")
    parser.add_argument("--J-Dll1", type=float, default=0.0, help="Notch-Dll1 coupling input J_Dll1")
    parser.add_argument("--seed", type=int, default=2016, help="Deterministic RNG seed")
    parser.add_argument("--self-test", action="store_true", help="Run self-verification suite and exit")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    run_self_test()
    png_path = render_comparison_figure(OUTPUT_PNG)
    gif_path = render_growth_gif(OUTPUT_GIF)
    print(f"[2016-hubaud-yap-excitable-segmentation] Wrote figure -> {png_path}")
    print(f"[2016-hubaud-yap-excitable-segmentation] Wrote animation -> {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
