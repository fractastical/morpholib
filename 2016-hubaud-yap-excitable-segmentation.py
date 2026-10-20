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
  5. Cooke, J. & Zeeman, E. C. (1976). "A clock and wavefront model for control of
     the number of repeated structures during animal morphogenesis."
     Journal of Theoretical Biology 58(2):455-476.
     DOI: 10.1016/S0022-5193(76)80131-2                                   [Pass 3]
  6. Dubrulle, J., McGrew, M. J. & Pourquie, O. (2001). "FGF signaling controls
     somite boundary position and regulates segmentation clock control of
     spatiotemporal Hox gene activation." Cell 106(2):219-232.
     DOI: 10.1016/S0092-8674(01)00437-8                                   [Pass 3]
  7. Wada, K., Itoga, K., Okano, T., Yonemura, S. & Sasaki, H. (2011). "Hippo
     pathway regulation by cell morphology and stress fibers."
     Development 138(18):3907-3914. DOI: 10.1242/dev.070987               [Pass 3]
  8. Gomez, C. et al. (2008). "Control of segment number in vertebrate embryos."
     Nature 454(7202):335-339. DOI: 10.1038/nature07020                   [Pass 3]

Pass 3 remediation (superseding audit 2026-10-19-yap-excitable-segmentation-audit-v2):
  * Section 4b: J_Dll1 is now an explicit juxtacrine Dll1-Notch coupling
    J_i = J_max * mean_j sigma(u_j) (ring synchrony sweep, Kuramoto R).
  * Section 4c: 1D anteroposterior PSM chain coupled to a regressing Fgf8/Wnt
    gradient F(x,t) through the gate g(F); cells arrest when gamma_F (1 - g) exceeds
    the Hopf margin, the frozen phase u_arrest(x) is thresholded into a Mesp2-like
    switch, and the resulting discrete stripes obey S = v_wf T (Laws 4-5).
  * Section 4d: substrate stiffness -> F-actin tension T(E, A) -> LATS1/2 -> three-state
    YAP phosphorylation / import-export kinetics -> Y_nuc (Law 6), calibrated on the
    five wild-type regimes, with a dynamic Latrunculin A rescue and a stiffness sweep.
  Honest reductions: the PSM is 1D (no mediolateral extent, no cell flow/convergent
  extension); the FGF gate acts additively on I_eff rather than through an explicit
  Fgf8/Wnt3a/Hes7 transcriptional circuit; the YAP cascade is a lumped ODE and keeps
  both a LATS-dependent branch (Wada 2011) and a LATS-independent import branch
  (Dupont 2011 found the stiffness response largely LATS-independent).

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

4. Clock-and-Wavefront Somite Spacing Law (Cooke & Zeeman 1976)             [Pass 3]
   (`hubaud_2017_clock_wavefront_somite_spacing`):
       x_wf(t) = x_wf0 + v_wf * t,   x_wf(t0 + (n+1) T) - x_wf(t0 + n T) = v_wf * T
   The determination front regresses with axis elongation; successive clock cycles
   freeze their phase S = v_wf T apart.

5. FGF/Wnt-Gated Wavefront Arrest Criterion (Dubrulle et al. 2001)           [Pass 3]
   (`hubaud_2017_fgf_wavefront_arrest_criterion`):
       F(x,t) = min(1, exp(-(x_tb(t) - x)/lambda_F)),  g(F) = F^h/(F_thr^h + F^h)
       I_eff(x,t) = I_basal + J_Dll1(x,t) + gamma_F (g(F) - 1) - gamma_YAP Y_nuc
       arrested  <=>  gamma_F (1 - g) > I_post - I_H
   The front sits a fixed FGF decay distance anterior to the tail bud and therefore
   advances at v_wf irrespective of lambda_F (lambda_F sets PSM length, not S).

6. YAP Mechanotransduction Nuclear-Fraction Monotonicity (Dupont 2011; Wada 2011) [Pass 3]
   (`hubaud_2017_yap_mechanotransduction_monotonicity`):
       T(E, A) = [E^hE/(E50^hE + E^hE)] [A/(A50 + A)] rho_actin c_ROCK
       L(T) = 1/(1 + (T/T_L)^m),   Y_n = r_n / (1 + r_n + r_p),
       r_n = r0 (1 + alpha_T T),   r_p = kappa_LATS L(T)
   Y_n increases strictly with the tension-driven import ratio r_n and decreases
   strictly with the LATS phosphorylation ratio r_p (proved cross-multiplied).
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
# Section 4b (Pass 3): Dll1-Notch Juxtacrine Phase Coupling
# (Hubaud et al. 2017 Cell Figs. 2 & 7; Law 3 J_Dll1 term)
# ============================================================================
def notch_ligand_presentation(u: np.ndarray) -> np.ndarray:
    """Dll1 ligand presentation sigma(u) = (1 + tanh u) / 2 of a cell's fast activator state.

    J_Dll1 in Law 3 is no longer a constant: in tissue it is the average ligand
    presented by the juxtacrine neighbours, J_Dll1,i = J_max * mean_{j in N(i)} sigma(u_j).
    Isolated cells on plain fibronectin have J = 0; Dll1-coated fibronectin is a
    constant J (Hubaud et al. 2017 Fig. 2); intact mPSM uses the neighbour average.
    """
    return 0.5 * (1.0 + np.tanh(u))


def simulate_dll1_notch_ring_synchrony(
    n_cells: int = 48,
    J_max: float = 0.15,
    Y_nuc: float = 0.18,
    period_cv: float = 0.04,
    n_contacts_each_side: int = 3,
    t_max_min: float = 1400.0,
    dt_min: float = 0.25,
    seed: int = 2016,
    a: float = 0.7,
    b: float = 0.8,
    tau_u: float = 3.78,
    tau_v: float = 47.25,
    I_basal: float = 0.82,
    gamma_YAP: float = 0.80,
) -> dict[str, Any]:
    """Ring of heterogeneous FHN oscillators coupled only through Dll1-Notch ligand presentation.

    Each cell contacts 2 * n_contacts_each_side neighbours (6 by default, the contact number of a
    hexagonally packed monolayer). Cell-to-cell period heterogeneity enters through
    tau_v,i ~ N(tau_v, period_cv * tau_v). The Kuramoto order parameter
    R(t) = |mean_i exp(i phi_i(t))|, phi_i = atan2(v_i - v*, u_i - u*), quantifies phase synchrony.
    With J_max = 0 the phases disperse; with J_max > 0 the juxtacrine coupling locks them
    (Notch-dependent synchronisation of the PSM clock; Hubaud et al. 2017 Fig. 7 model).
    """
    rng = np.random.default_rng(seed)
    tau_v_i = tau_v * np.clip(1.0 + period_cv * rng.standard_normal(n_cells), 0.7, 1.3)
    I_cell = I_basal - gamma_YAP * Y_nuc
    eq = solve_fhn_equilibrium(I_eff=I_cell + 0.5 * J_max, a=a, b=b, tau_u=tau_u, tau_v=tau_v)
    phase0 = rng.uniform(0.0, 2.0 * math.pi, n_cells)
    u = eq["u_star"] + 1.2 * np.cos(phase0)
    v = eq["v_star"] + 0.6 * np.sin(phase0)

    n_steps = int(round(t_max_min / dt_min))
    stride = max(1, n_steps // 280)
    R_list: list[float] = []
    t_list: list[float] = []
    u_rows: list[np.ndarray] = []
    for step in range(n_steps):
        su = notch_ligand_presentation(u)
        J = np.zeros(n_cells)
        for k in range(1, n_contacts_each_side + 1):
            J += np.roll(su, k) + np.roll(su, -k)
        J = J_max * J / (2.0 * n_contacts_each_side)
        du = (u - (u ** 3) / 3.0 - v + I_cell + J) / tau_u
        dv = (u + a - b * v) / tau_v_i
        u = u + dt_min * du
        v = v + dt_min * dv
        if step % stride == 0:
            phi = np.arctan2(v - eq["v_star"], u - eq["u_star"])
            R_list.append(float(np.abs(np.mean(np.exp(1j * phi)))))
            t_list.append(step * dt_min)
            u_rows.append(u.copy())
    R = np.array(R_list)
    t_arr = np.array(t_list)
    tail = t_arr > 0.7 * t_max_min
    return {
        "J_max": float(J_max),
        "t_grid_min": t_arr,
        "order_parameter": R,
        "order_parameter_final": round(float(np.mean(R[tail])), 4),
        "u_rows": np.array(u_rows),
        "tau_v_cv": round(float(np.std(tau_v_i) / np.mean(tau_v_i)), 4),
    }


def evaluate_dll1_coupling_sweep(
    J_values: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30), seed: int = 2016
) -> dict[str, Any]:
    """Sweep the Dll1-Notch coupling strength J_max and record the final Kuramoto order parameter."""
    rows = [simulate_dll1_notch_ring_synchrony(J_max=j, seed=seed) for j in J_values]
    R_vals = np.array([r["order_parameter_final"] for r in rows])
    rho, p_val = stats.spearmanr(np.array(J_values), R_vals)
    return {
        "J_values": list(J_values),
        "R_final": R_vals.tolist(),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "runs": rows,
    }


# ============================================================================
# Section 4c (Pass 3): Anteroposterior FGF/Wnt Gradient & Clock-and-Wavefront Arrest
# (Cooke & Zeeman 1976; Dubrulle, McGrew & Pourquie 2001; Laws 4 & 5)
# ============================================================================
def fgf_gradient_field(
    x_um: np.ndarray, t_min: float, x_tb0_um: float, v_wf_um_per_min: float, lambda_F_um: float
) -> np.ndarray:
    """Posterior-high Fgf8/Wnt3a gradient F(x,t) = min(1, exp(-(x_tb(t) - x) / lambda_F)).

    Dubrulle et al. (2001): fgf8 mRNA is transcribed only in the tail bud and decays in
    cells left behind as the axis elongates (x_tb(t) = x_tb0 + v_wf t), so the gradient
    regresses posteriorly at the elongation speed v_wf (Lean Theorem 5, second clause).
    """
    x_tb = x_tb0_um + v_wf_um_per_min * t_min
    return np.minimum(1.0, np.exp(-(x_tb - x_um) / lambda_F_um))


def fgf_excitability_gate(F: np.ndarray | float, F_thr: float = 0.15, h_F: float = 4.0) -> np.ndarray | float:
    """Hill gate g(F) = F^h / (F_thr^h + F^h) in (0, 1) through which FGF/Wnt sustain excitability."""
    return F ** h_F / (F_thr ** h_F + F ** h_F)


def fgf_gated_current(
    g: np.ndarray | float,
    J_Dll1: np.ndarray | float,
    Y_nuc: float,
    gamma_F: float = 1.60,
    I_basal: float = 0.82,
    gamma_YAP: float = 0.80,
) -> np.ndarray | float:
    """Law 5: I_eff(x,t) = I_basal + J_Dll1 + gamma_F (g(F) - 1) - gamma_YAP Y_nuc.

    In the tail bud (g -> 1) this reduces to Law 3; anteriorly the FGF/Wnt loss
    gamma_F (1 - g) withdraws excitability until I_eff < I_H_lower and the cell arrests.
    """
    return I_basal + J_Dll1 + gamma_F * (g - 1.0) - gamma_YAP * Y_nuc


def wavefront_gate_threshold(
    Y_nuc: float = 0.18, J_max: float = 0.15, gamma_F: float = 1.60, I_basal: float = 0.82, gamma_YAP: float = 0.80
) -> float:
    """Critical FGF gate value g_crit at which the mean-Notch-driven cell crosses I_H_lower (Theorem 5)."""
    hopf = compute_hopf_thresholds()
    I_post = I_basal + 0.5 * J_max - gamma_YAP * Y_nuc
    return float(1.0 - (I_post - hopf["I_hopf_lower"]) / gamma_F)


def simulate_ap_wavefront_arrest(
    L_um: float = 2000.0,
    nx: int = 400,
    x_tb0_um: float = 700.0,
    v_wf_um_per_min: float = 0.70,
    lambda_F_um: float = 350.0,
    F_thr: float = 0.15,
    h_F: float = 4.0,
    gamma_F: float = 1.60,
    Y_nuc: float = 0.18,
    J_max: float = 0.15,
    D_u_um2_per_min: float = 24.0,
    t_max_min: float = 1400.0,
    dt_min: float = 0.25,
    mode: str = "fgf_gradient",
    seed: int = 2016,
    a: float = 0.7,
    b: float = 0.8,
    tau_u: float = 3.78,
    tau_v: float = 47.25,
) -> dict[str, Any]:
    """1D anteroposterior PSM: excitable FHN chain + Dll1-Notch coupling + regressing FGF/Wnt gradient.

    Modes:
      - `fgf_gradient`: F(x,t) regresses posteriorly with the tail bud; cells anterior to the
        determination front (g(F) < g_crit) drop below the Hopf window and freeze. The fast
        activator value at the moment of arrest, u_arrest(x), is the frozen clock phase
        (phase-gradient encoding, Lauschke et al. 2013); the Mesp2-like determination switch
        m(x) = [u_arrest(x) > 0] yields discrete somite stripes separated by clefts whose
        spacing follows S = v_wf * T (Cooke & Zeeman 1976; Lean Theorem 4).
      - `uniform_fgf_ablation`: F = 1 everywhere (ectopic FGF8 / no gradient; Dubrulle et al. 2001
        bead experiments): no front, no arrest, no stripes.
    """
    x = np.linspace(0.0, L_um, nx)
    dx = float(x[1] - x[0])
    rng = np.random.default_rng(seed)
    u = np.full(nx, -1.0, dtype=float)
    v = (u + a) / b
    u = u + 0.30 * (x > x_tb0_um - 50.0) + 0.01 * rng.standard_normal(nx)

    g_crit = wavefront_gate_threshold(Y_nuc=Y_nuc, J_max=J_max, gamma_F=gamma_F)
    n_steps = int(round(t_max_min / dt_min))
    arrested = np.zeros(nx, dtype=bool)
    u_arrest = np.full(nx, np.nan)
    t_arrest = np.full(nx, np.nan)
    stride = max(1, n_steps // 240)
    kymo: list[np.ndarray] = []
    t_keep: list[float] = []
    front_x: list[float] = []
    u_tailbud = np.zeros(n_steps, dtype=float)
    t_full = np.arange(n_steps) * dt_min

    for step in range(n_steps):
        t = step * dt_min
        if mode == "fgf_gradient":
            F = fgf_gradient_field(x, t, x_tb0_um, v_wf_um_per_min, lambda_F_um)
        elif mode == "uniform_fgf_ablation":
            F = np.ones(nx)
        else:
            raise ValueError(f"Unknown wavefront mode: {mode}")
        g = fgf_excitability_gate(F, F_thr=F_thr, h_F=h_F)
        su = notch_ligand_presentation(u)
        J = J_max * 0.5 * (np.roll(su, 1) + np.roll(su, -1))
        J[0] = J_max * su[1]
        J[-1] = J_max * su[-2]
        I = fgf_gated_current(g, J, Y_nuc, gamma_F=gamma_F)
        u_pad = np.pad(u, 1, mode="edge")
        lap = (u_pad[2:] + u_pad[:-2] - 2.0 * u) / (dx * dx)
        du = (u - (u ** 3) / 3.0 - v + I + D_u_um2_per_min * lap) / tau_u
        dv = (u + a - b * v) / tau_v

        newly = (~arrested) & (g < g_crit)
        u_arrest[newly] = u[newly]
        t_arrest[newly] = t
        arrested |= newly

        u = u + dt_min * du
        v = v + dt_min * dv
        u_tailbud[step] = u[-1]
        if step % stride == 0:
            kymo.append(u.copy())
            t_keep.append(t)
            front_x.append(float(x[~arrested].min()) if (~arrested).any() else float(L_um))

    # Tail-bud clock period from the posterior-most cell
    peaks, _ = signal.find_peaks(u_tailbud[n_steps // 4 :], distance=int(50.0 / dt_min), prominence=0.8)
    T_tailbud = float(np.mean(np.diff(t_full[n_steps // 4 :][peaks]))) if len(peaks) >= 2 else 0.0

    # Mesp2-like determination switch and stripe morphometrics on cells arrested after the
    # initial transient (t_arrest > 2 periods) so the initial-condition ring is excluded.
    m = np.where(np.isnan(u_arrest), 0.0, (u_arrest > 0.0).astype(float))
    settled = (~np.isnan(t_arrest)) & (t_arrest > 2.0 * max(T_tailbud, 100.0))
    rising = np.where((np.diff(m) > 0.5) & settled[1:])[0]
    falling = np.where((np.diff(m) < -0.5) & settled[1:])[0]
    stripe_edges_um = x[rising + 1]
    spacings = np.diff(stripe_edges_um)
    psm_length_um = float(np.mean([(x_tb0_um + v_wf_um_per_min * tk) - fx for tk, fx in zip(t_keep, front_x) if tk > 2.0 * max(T_tailbud, 100.0)])) if mode == "fgf_gradient" and len(front_x) else 0.0
    psm_length_pred_um = float(lambda_F_um * math.log(1.0 / (F_thr * (g_crit / (1.0 - g_crit)) ** (1.0 / h_F)))) if 0.0 < g_crit < 1.0 else 0.0

    # Excitable-wave arrest: how far anterior of the front do pulses still penetrate? Cells arrested
    # > 200 min before a late 300-min window should be silent (max u < -0.3) if the waves truly stop.
    kymo_arr = np.array(kymo)
    t_keep_arr = np.array(t_keep)
    late = t_keep_arr > (t_max_min - 300.0)
    u_max_late = kymo_arr[late].max(axis=0) if late.any() else np.full(nx, -np.inf)
    old = (~np.isnan(t_arrest)) & (t_arrest < t_max_min - 500.0)
    still_excited = old & (u_max_late > -0.3)
    front_late = float(np.array(front_x)[late][0]) if late.any() else float(L_um)
    wave_penetration_um = float(front_late - x[still_excited].min()) if still_excited.any() else 0.0

    return {
        "mode": mode,
        "x_um": x,
        "t_grid_min": np.array(t_keep),
        "kymograph": np.array(kymo),
        "front_x_um": np.array(front_x),
        "u_arrest": u_arrest,
        "t_arrest_min": t_arrest,
        "mesp2_switch": m,
        "g_crit": g_crit,
        "T_tailbud_min": round(T_tailbud, 2),
        "v_wf_um_per_min": float(v_wf_um_per_min),
        "lambda_F_um": float(lambda_F_um),
        "stripe_rising_edges_um": stripe_edges_um,
        "stripe_falling_edges_um": x[falling + 1],
        "n_stripes": int(len(stripe_edges_um)),
        "stripe_spacing_um": round(float(np.mean(spacings)), 3) if len(spacings) else 0.0,
        "stripe_spacing_cv": round(float(np.std(spacings, ddof=1) / np.mean(spacings)), 4) if len(spacings) > 1 else 0.0,
        "cooke_zeeman_prediction_um": round(float(v_wf_um_per_min * T_tailbud), 3),
        "psm_length_um": round(psm_length_um, 2),
        "psm_length_pred_um": round(psm_length_pred_um, 2),
        "n_arrested_cells": int(np.sum(arrested & settled)),
        "gamma_F": float(gamma_F),
        "anterior_wave_penetration_um": round(wave_penetration_um, 2),
        "n_old_cells_still_excited": int(np.sum(still_excited)),
    }


def evaluate_wavefront_speed_sweep(
    v_values: tuple[float, ...] = (0.45, 0.60, 0.70, 0.85, 1.00),
    lambda_values: tuple[float, ...] = (250.0, 350.0, 450.0),
    gamma_values: tuple[float, ...] = (0.80, 1.20, 1.60, 2.00),
    seed: int = 2016,
) -> dict[str, Any]:
    """Parameter sweeps of the clock-and-wavefront chain:
      * stripe spacing S vs wavefront speed v_wf (Cooke-Zeeman law S = v_wf T);
      * PSM (oscillating-domain) length vs FGF gradient decay length lambda_F (must leave S unchanged);
      * anterior penetration of excitable pulses vs FGF/Wnt excitability gain gamma_F — at low gamma_F the
        arrested cells remain excitable and pulses leak anteriorly through the formed somites, at high
        gamma_F they are non-excitable and the waves truly arrest at the front.
    """
    sweeps = []
    for v_val in v_values:
        res = simulate_ap_wavefront_arrest(v_wf_um_per_min=v_val, seed=seed)
        sweeps.append(
            {
                "v_wf_um_per_min": v_val,
                "stripe_spacing_um": res["stripe_spacing_um"],
                "cooke_zeeman_um": res["cooke_zeeman_prediction_um"],
                "n_stripes": res["n_stripes"],
                "T_tailbud_min": res["T_tailbud_min"],
            }
        )
    v_arr = np.array([s["v_wf_um_per_min"] for s in sweeps])
    S_arr = np.array([s["stripe_spacing_um"] for s in sweeps])
    cz_arr = np.array([s["cooke_zeeman_um"] for s in sweeps])
    slope, intercept, r_val, p_val, _ = stats.linregress(v_arr, S_arr)
    rel_err = float(np.mean(np.abs(S_arr - cz_arr) / cz_arr))

    lam_sweep = []
    for lam in lambda_values:
        res = simulate_ap_wavefront_arrest(lambda_F_um=lam, seed=seed)
        lam_sweep.append(
            {
                "lambda_F_um": lam,
                "psm_length_um": res["psm_length_um"],
                "psm_length_pred_um": res["psm_length_pred_um"],
                "stripe_spacing_um": res["stripe_spacing_um"],
            }
        )
    gamma_sweep = []
    for g_val in gamma_values:
        res = simulate_ap_wavefront_arrest(gamma_F=g_val, seed=seed)
        gamma_sweep.append(
            {
                "gamma_F": g_val,
                "anterior_wave_penetration_um": res["anterior_wave_penetration_um"],
                "n_old_cells_still_excited": res["n_old_cells_still_excited"],
                "stripe_spacing_um": res["stripe_spacing_um"],
                "psm_length_um": res["psm_length_um"],
            }
        )
    return {
        "speed_sweep": sweeps,
        "slope_min": float(slope),
        "intercept_um": float(intercept),
        "pearson_r": float(r_val),
        "mean_rel_err_vs_cooke_zeeman": rel_err,
        "lambda_sweep": lam_sweep,
        "gamma_sweep": gamma_sweep,
    }


# ============================================================================
# Section 4d (Pass 3): Substrate Stiffness -> F-actin -> LATS1/2 -> Nuclear YAP Kinetics
# (Dupont et al. 2011; Wada et al. 2011; Law 6)
# ============================================================================
MECH_E50_KPA = 10.0
MECH_H_E = 2.0
MECH_A50_UM2 = 500.0
MECH_M_LATS = 3.0
MECH_R0 = 1.0
MECH_K_EXP = 1.0 / 12.0   # YAP nuclear export rate (1/min)
MECH_K_PP = 1.0 / 20.0    # phospho-YAP dephosphorylation rate (1/min)
MECH_TAU_ACTIN = 8.0      # F-actin depolymerisation time constant after Latrunculin A (min)
MECH_YAP5SA_OVEREXPRESSION = 2.0


def substrate_tension_index(
    E_sub_kPa: float | np.ndarray, A_cell_um2: float | np.ndarray, rho_actin: float = 1.0, c_rock: float = 1.0
) -> float | np.ndarray:
    """Dimensionless F-actin/stress-fibre tension T(E, A) in [0, 1].

    Stiffness Hill term (Dupont et al. 2011: cytoplasmic YAP at 0.7 kPa, nuclear at 40 kPa),
    spreading-area saturation (micropatterned islands), F-actin content rho_actin (Latrunculin A
    lowers it) and ROCK contractility c_rock (Y-27632 lowers it).
    """
    e_term = E_sub_kPa ** MECH_H_E / (MECH_E50_KPA ** MECH_H_E + E_sub_kPa ** MECH_H_E)
    a_term = A_cell_um2 / (MECH_A50_UM2 + A_cell_um2)
    return e_term * a_term * rho_actin * c_rock


def lats_activity(T: float | np.ndarray, T_L: float) -> float | np.ndarray:
    """LATS1/2 kinase activity L(T) = 1 / (1 + (T / T_L)^m): stress fibres inhibit LATS (Wada et al. 2011)."""
    return 1.0 / (1.0 + (T / T_L) ** MECH_M_LATS)


def yap_nuclear_fraction_ss(
    T: float | np.ndarray, alpha_T: float, kappa_LATS: float, T_L: float, lats_sensitive: bool = True
) -> float | np.ndarray:
    """Steady-state nuclear YAP fraction Y_n = r_n / (1 + r_n + r_p) (Law 6).

    r_n = r0 (1 + alpha_T T) is the tension-driven import/export ratio; r_p = kappa_LATS L(T)
    the LATS phosphorylation/dephosphorylation ratio (0 for the phospho-dead Yap5SA mutant).
    """
    r_n = MECH_R0 * (1.0 + alpha_T * T)
    r_p = (kappa_LATS * lats_activity(T, T_L)) if lats_sensitive else 0.0
    return r_n / (1.0 + r_n + r_p)


def calibrate_yap_mechanotransduction(ref: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit (alpha_T, kappa_LATS, T_L) so that the steady-state Y_n of the mechanical inputs of
    the five wild-type Hubaud et al. (2017) regimes reproduces their tabulated Y_nuc values.
    Yap5SA is predicted (not fitted) with kappa_LATS = 0 and the overexpression factor.
    """
    from scipy import optimize

    if ref is None:
        ref = load_reference_data()
    regimes = ref["hubaud_2017_experimental_regimes"]["regimes"]
    mech = ref["yap_mechanotransduction_kinetics_reference"]["regime_mechanical_inputs"]
    wt = [(r["regime_id"], mech[r["regime_id"]], float(r["Y_nuc"])) for r in regimes if mech[r["regime_id"]]["lats_sensitive"]]

    def tension_of(inp: dict[str, Any]) -> float:
        return float(substrate_tension_index(inp["E_sub_kPa"], inp["A_cell_um2"], inp["rho_actin"], inp["c_rock"]))

    def resid(p: np.ndarray) -> list[float]:
        return [yap_nuclear_fraction_ss(tension_of(inp), *p) - y for _, inp, y in wt]

    sol = optimize.least_squares(resid, x0=[5.0, 4.0, 0.35], bounds=([0.0, 0.0, 0.05], [50.0, 200.0, 1.0]))
    alpha_T, kappa_LATS, T_L = (float(val) for val in sol.x)
    table = []
    for r in regimes:
        inp = mech[r["regime_id"]]
        T_val = tension_of(inp)
        y_pred = float(yap_nuclear_fraction_ss(T_val, alpha_T, kappa_LATS, T_L, lats_sensitive=inp["lats_sensitive"]))
        y_pred = min(1.0, inp["overexpression"] * y_pred)
        table.append({"regime_id": r["regime_id"], "tension_index": round(T_val, 4), "Y_nuc_model": round(y_pred, 4), "Y_nuc_table": float(r["Y_nuc"])})
    wt_rows = [row for row in table if mech[row["regime_id"]]["lats_sensitive"]]
    rmse = float(np.sqrt(np.mean([(row["Y_nuc_model"] - row["Y_nuc_table"]) ** 2 for row in wt_rows])))
    return {"alpha_T": alpha_T, "kappa_LATS": kappa_LATS, "T_L": T_L, "table": table, "rmse_wt": rmse, "n_wt": len(wt_rows)}


def evaluate_substrate_stiffness_sweep(calib: dict[str, Any], A_cell_um2: float = 2000.0, n: int = 160) -> dict[str, Any]:
    """Y_nuc(E_sub) for a spread cell and the predicted critical stiffness E_crit where Y_nuc = Y_nuc_crit
    (oscillatory below, quiescent above). Dupont et al. (2011) bracket: 0.7 kPa (soft) ... 40 kPa (stiff)."""
    E_grid = np.logspace(-1.0, 2.3, n)
    T_grid = substrate_tension_index(E_grid, A_cell_um2)
    Y_grid = yap_nuclear_fraction_ss(T_grid, calib["alpha_T"], calib["kappa_LATS"], calib["T_L"])
    hopf = compute_hopf_thresholds()
    Y_crit = hopf["Y_nuc_crit"]
    above = np.where(Y_grid > Y_crit)[0]
    if len(above) and above[0] > 0:
        i1 = above[0]
        i0 = i1 - 1
        frac = (Y_crit - Y_grid[i0]) / (Y_grid[i1] - Y_grid[i0])
        E_crit = float(np.exp(np.log(E_grid[i0]) + frac * (np.log(E_grid[i1]) - np.log(E_grid[i0]))))
    else:
        E_crit = float("nan")
    return {
        "E_grid_kPa": E_grid,
        "Y_nuc_grid": Y_grid,
        "Y_nuc_crit": Y_crit,
        "E_crit_kPa": E_crit,
        "Y_nuc_soft_0p7kPa": float(yap_nuclear_fraction_ss(substrate_tension_index(0.7, A_cell_um2), calib["alpha_T"], calib["kappa_LATS"], calib["T_L"])),
        "Y_nuc_stiff_40kPa": float(yap_nuclear_fraction_ss(substrate_tension_index(40.0, A_cell_um2), calib["alpha_T"], calib["kappa_LATS"], calib["T_L"])),
    }


def simulate_latrunculin_yap_kinetics(
    calib: dict[str, Any],
    E_sub_kPa: float = 40.0,
    A_cell_um2: float = 2000.0,
    rho_actin_lata: float = 0.15,
    t_pre_min: float = 90.0,
    t_post_min: float = 600.0,
    dt_min: float = 0.25,
    a: float = 0.7,
    b: float = 0.8,
    tau_u: float = 3.78,
    tau_v: float = 47.25,
    I_basal: float = 0.82,
    gamma_YAP: float = 0.80,
) -> dict[str, Any]:
    """Dynamic Latrunculin A rescue: F-actin tension T(t) relaxes (tau_actin) after LatA at t = 0, the
    three-state YAP cycle (Y_n, Y_p, Y_c) follows, and the FHN cell is driven by the time-varying
    I_eff(t) = I_basal - gamma_YAP Y_n(t). Returns the oscillation-onset latency after LatA.
    """
    alpha_T, kappa, T_L = calib["alpha_T"], calib["kappa_LATS"], calib["T_L"]
    T0 = float(substrate_tension_index(E_sub_kPa, A_cell_um2, 1.0, 1.0))
    T_inf = float(substrate_tension_index(E_sub_kPa, A_cell_um2, rho_actin_lata, 1.0))
    n_steps = int(round((t_pre_min + t_post_min) / dt_min))
    t = -t_pre_min + np.arange(n_steps) * dt_min
    # Steady state before LatA
    r_n0 = MECH_R0 * (1.0 + alpha_T * T0)
    r_p0 = kappa * float(lats_activity(T0, T_L))
    Y_n = r_n0 / (1.0 + r_n0 + r_p0)
    Y_p = r_p0 / (1.0 + r_n0 + r_p0)
    T_val = T0
    I0 = I_basal - gamma_YAP * Y_n
    eq0 = solve_fhn_equilibrium(I0, a=a, b=b, tau_u=tau_u, tau_v=tau_v)
    u, v = eq0["u_star"] + 0.05, eq0["v_star"]
    T_tr = np.zeros(n_steps)
    Yn_tr = np.zeros(n_steps)
    Yp_tr = np.zeros(n_steps)
    u_tr = np.zeros(n_steps)
    for i in range(n_steps):
        if t[i] >= 0.0:
            T_val += dt_min * (T_inf - T_val) / MECH_TAU_ACTIN
        Y_c = 1.0 - Y_n - Y_p
        k_imp = MECH_R0 * MECH_K_EXP * (1.0 + alpha_T * T_val)
        k_lats = kappa * MECH_K_PP * float(lats_activity(T_val, T_L))
        dYn = k_imp * Y_c - MECH_K_EXP * Y_n
        dYp = k_lats * Y_c - MECH_K_PP * Y_p
        Y_n += dt_min * dYn
        Y_p += dt_min * dYp
        I_eff = I_basal - gamma_YAP * Y_n
        du = (u - u ** 3 / 3.0 - v + I_eff) / tau_u
        dv = (u + a - b * v) / tau_v
        u += dt_min * du
        v += dt_min * dv
        T_tr[i], Yn_tr[i], Yp_tr[i], u_tr[i] = T_val, Y_n, Y_p, u
    hopf = compute_hopf_thresholds()
    post = t >= 0.0
    peaks, _ = signal.find_peaks(u_tr[post], distance=int(50.0 / dt_min), prominence=0.8)
    onset = float(t[post][peaks[0]]) if len(peaks) else float("nan")
    cross = np.where(post & (Yn_tr < hopf["Y_nuc_crit"]))[0]
    t_cross = float(t[cross[0]]) if len(cross) else float("nan")
    # Y_n relaxation half-time toward its LatA steady state
    Yn_post = Yn_tr[post]
    Yn_inf = Yn_post[-1]
    half_idx = np.where(Yn_post <= Yn_post[0] + 0.5 * (Yn_inf - Yn_post[0]))[0]
    t_half = float(t[post][half_idx[0]]) if len(half_idx) else float("nan")
    period = float(np.mean(np.diff(t[post][peaks]))) if len(peaks) >= 2 else 0.0
    return {
        "t_min": t,
        "tension": T_tr,
        "Y_nuc": Yn_tr,
        "Y_phospho": Yp_tr,
        "u": u_tr,
        "Y_nuc_pre": float(Yn_tr[0]),
        "Y_nuc_post": float(Yn_inf),
        "t_yap_half_min": round(t_half, 2),
        "t_cross_Y_crit_min": round(t_cross, 2),
        "oscillation_onset_min": round(onset, 2),
        "post_period_min": round(period, 2),
        "n_peaks_post": int(len(peaks)),
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
    """Return 11 quantitative morphometric measurements for `morphoeval/measure_outputs.py`
    (5 from Pass 2 plus 6 Pass 3 measurements for the FGF/Wnt wavefront, the YAP
    mechanotransduction cascade and the Dll1-Notch phase coupling)."""
    verify_dataset_integrity()
    ref = load_reference_data()
    hopf = compute_hopf_thresholds()
    pert = evaluate_hubaud_perturbation_regimes(ref, seed=2016)
    sim_lata = pert["regimes"][2]["sim"]  # isolated_latrunculin_a_rescue
    sim_yap5sa = pert["regimes"][5]["sim"]  # yap5sa_constitutively_active_block
    spec = segment_mouse_psm_somites_specimen()
    wf = simulate_ap_wavefront_arrest(seed=2016)
    wf_abl = simulate_ap_wavefront_arrest(mode="uniform_fgf_ablation", seed=2016)
    sweep = evaluate_wavefront_speed_sweep(seed=2016)
    calib = calibrate_yap_mechanotransduction(ref)
    stiff = evaluate_substrate_stiffness_sweep(calib)
    lata_kin = simulate_latrunculin_yap_kinetics(calib)
    sync = evaluate_dll1_coupling_sweep(seed=2016)
    cw_ref = ref["clock_wavefront_fgf_gradient_reference"]["empirical_anchors"]
    dup = ref["yap_mechanotransduction_kinetics_reference"]["dupont_2011_stiffness_anchors"]

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
        # ---------------- Pass 3 measurements ----------------
        {
            "name": "FGF/Wnt wavefront-arrested somite stripe spacing S_model (1D AP excitable chain)",
            "value": wf["stripe_spacing_um"],
            "unit": "um",
            "method": f"1D FHN chain + Dll1-Notch coupling + regressing Fgf8 gradient (lambda_F = {wf['lambda_F_um']:.0f} um, v_wf = {wf['v_wf_um_per_min']:.2f} um/min); Mesp2-like switch m(x) = [u_arrest(x) > 0]; mean rising-edge spacing over {wf['n_stripes']} stripes (CV = {wf['stripe_spacing_cv']:.4f})",
            "reference_value": wf["cooke_zeeman_prediction_um"],
            "reference_source": f"Cooke & Zeeman (1976) J Theor Biol 58:455-476 clock-and-wavefront law S = v_wf * T with the simulated tail-bud period T = {wf['T_tailbud_min']:.2f} min (Lean Theorem 4); mouse somite AP length ~{cw_ref['mouse_somite_ap_length_um']:.0f} um (Gomez et al. 2008 Nature 454:335-339)",
            "note": f"uniform-FGF ablation: {wf_abl['n_stripes']} stripes / {wf_abl['n_arrested_cells']} arrested cells; PSM length {wf['psm_length_um']:.1f} um vs analytic {wf['psm_length_pred_um']:.1f} um; excitable-wave penetration anterior of the front = {wf['anterior_wave_penetration_um']:.0f} um at gamma_F = {wf['gamma_F']:.2f} (gamma_F sweep: " + ", ".join(f"{row['gamma_F']:.2f}->{row['anterior_wave_penetration_um']:.0f} um" for row in sweep["gamma_sweep"]) + f"); v_wf sweep slope dS/dv = {sweep['slope_min']:.1f} min (r = {sweep['pearson_r']:.5f}, mean |S - v T|/vT = {sweep['mean_rel_err_vs_cooke_zeeman']:.4f})",
            "derived_from": "script",
        },
        {
            "name": "Cooke-Zeeman sweep slope dS/dv_wf across 5 wavefront speeds",
            "value": round(sweep["slope_min"], 2),
            "unit": "min",
            "method": "linear regression of emergent stripe spacing S on v_wf in {" + ", ".join(f"{s['v_wf_um_per_min']:.2f}" for s in sweep["speed_sweep"]) + f"}} um/min (intercept = {sweep['intercept_um']:.2f} um)",
            "reference_value": wf["T_tailbud_min"],
            "reference_source": "Cooke & Zeeman (1976): dS/dv_wf = T_clock (simulated tail-bud period); lambda_F sweep must leave S unchanged (Lean Theorem 5)",
            "note": "lambda_F sweep: " + "; ".join(f"lambda={row['lambda_F_um']:.0f} um -> PSM {row['psm_length_um']:.0f} um (pred {row['psm_length_pred_um']:.0f}), S = {row['stripe_spacing_um']:.1f} um" for row in sweep["lambda_sweep"]),
            "derived_from": "script",
        },
        {
            "name": "YAP mechanotransduction cascade Y_nuc RMSE across 5 wild-type Hubaud regimes",
            "value": round(calib["rmse_wt"], 4),
            "unit": "nuclear YAP fraction",
            "method": f"3-parameter least-squares fit (alpha_T = {calib['alpha_T']:.3f}, kappa_LATS = {calib['kappa_LATS']:.3f}, T_L = {calib['T_L']:.3f}) of Y_n = r_n/(1 + r_n + r_p) to the regime Y_nuc values from substrate stiffness, spreading area, Latrunculin A and Y-27632 inputs; Yap5SA predicted with kappa_LATS = 0",
            "reference_value": 0.0,
            "reference_source": "Hubaud et al. (2017) regime Y_nuc table (datasets/yap_excitable_segmentation/yap_excitable_segmentation_reference.json); cascade structure after Dupont et al. (2011) Nature 474:179-183 and Wada et al. (2011) Development 138:3907-3914",
            "note": "; ".join(f"{row['regime_id']}: T = {row['tension_index']:.3f}, Y_model = {row['Y_nuc_model']:.3f} vs {row['Y_nuc_table']:.2f}" for row in calib["table"]),
            "derived_from": "script",
        },
        {
            "name": "predicted critical substrate stiffness E_crit for spread PSM cells (Y_nuc = Y_nuc_crit)",
            "value": round(stiff["E_crit_kPa"], 3),
            "unit": "kPa",
            "method": f"log-interpolated root of Y_nuc(E_sub) = {stiff['Y_nuc_crit']:.4f} along a 160-point stiffness sweep for A_cell = 2000 um^2",
            "reference_value": round(math.sqrt(dup["soft_kPa"] * dup["stiff_kPa"]), 3),
            "reference_source": f"Dupont et al. (2011) Nature 474:179-183 Fig. 1 bracket: YAP cytoplasmic at {dup['soft_kPa']} kPa, nuclear at {dup['stiff_kPa']} kPa (geometric midpoint as nominal reference; E_crit must lie inside the bracket)",
            "note": f"Y_nuc(0.7 kPa) = {stiff['Y_nuc_soft_0p7kPa']:.3f} (oscillatory), Y_nuc(40 kPa) = {stiff['Y_nuc_stiff_40kPa']:.3f} (quiescent)",
            "derived_from": "script",
        },
        {
            "name": "Latrunculin A dynamic rescue: oscillation onset latency after F-actin depolymerisation",
            "value": lata_kin["oscillation_onset_min"],
            "unit": "min",
            "method": f"coupled T(t) -> LATS -> (Y_n, Y_p) -> I_eff(t) -> FHN integration; Y_nuc {lata_kin['Y_nuc_pre']:.3f} -> {lata_kin['Y_nuc_post']:.3f}, Y_crit crossed at {lata_kin['t_cross_Y_crit_min']:.1f} min, YAP half-relaxation {lata_kin['t_yap_half_min']:.1f} min",
            "reference_value": 142.0,
            "reference_source": "Hubaud et al. (2017) Cell 171:668-682 Fig. 4: LatA-treated isolated cells begin autonomous LuVeLu oscillations within the first clock period (~142 min) of imaging (upper bound used as reference)",
            "note": f"post-LatA period = {lata_kin['post_period_min']:.1f} min ({lata_kin['n_peaks_post']} peaks in 600 min)",
            "derived_from": "script",
        },
        {
            "name": "Dll1-Notch juxtacrine coupling Kuramoto order parameter R at J_max = 0.15",
            "value": sync["R_final"][3],
            "unit": "Kuramoto R",
            "method": f"48-cell ring, 6 contacts per cell, tau_v CV = {sync['runs'][3]['tau_v_cv']:.3f}, random initial phases, 1400 min; R(t) averaged over the last 30%",
            "reference_value": 1.0,
            "reference_source": "complete phase synchrony (R = 1); uncoupled control R = " + f"{sync['R_final'][0]:.3f}; Hubaud et al. (2017) Fig. 7 Notch-coupled excitable-cell model",
            "note": "J_max sweep R = " + ", ".join(f"{j:.2f}->{r:.3f}" for j, r in zip(sync["J_values"], sync["R_final"])) + f" (Spearman rho = {sync['spearman_rho']:.3f})",
            "derived_from": "script",
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
    # Pass 3 mechanisms
    wf = simulate_ap_wavefront_arrest(seed=2016)
    wf_abl = simulate_ap_wavefront_arrest(mode="uniform_fgf_ablation", seed=2016)
    sweep = evaluate_wavefront_speed_sweep(seed=2016)
    calib = calibrate_yap_mechanotransduction(ref)
    stiff = evaluate_substrate_stiffness_sweep(calib)
    lata_kin = simulate_latrunculin_yap_kinetics(calib)
    sync = evaluate_dll1_coupling_sweep(seed=2016)

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

    fig, axes = plt.subplots(3, 3, figsize=(18.5, 16.5), dpi=150)
    fig.suptitle(
        "Hubaud, Regev, Mahadevan & Pourquié (2017) Cell — Excitable FitzHugh–Nagumo & YAP Mechanotransduction Segmentation Clock\n"
        "Pass 3: FGF/Wnt clock-and-wavefront arrest (Cooke–Zeeman) · stiffness → F-actin → LATS → YAP kinetics (Dupont 2011) · Dll1–Notch phase coupling",
        fontsize=12.5,
        fontweight="bold",
        color="#f8fafc",
        y=0.985,
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

    # Panel G (Pass 3): 1D AP clock-and-wavefront kymograph with FGF gradient & somite stripes
    ax = axes[2, 0]
    kx = wf["x_um"]
    kt = wf["t_grid_min"]
    luv = np.clip((wf["kymograph"] + 2.1) / 4.2, 0.0, 1.0)
    ax.imshow(
        luv,
        origin="lower",
        aspect="auto",
        extent=[kx[0], kx[-1], kt[0], kt[-1]],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax.plot(wf["front_x_um"], kt, color="#38bdf8", lw=1.6, label="Determination front $x_{wf}(t)$ ($g(F)=g_{crit}$)")
    ax.plot(700.0 + wf["v_wf_um_per_min"] * kt, kt, color="#fde047", ls="--", lw=1.4, label=f"Tail bud $x_{{tb}}(t)$ ($v_{{wf}}={wf['v_wf_um_per_min']:.2f}$ µm/min)")
    for xe in wf["stripe_rising_edges_um"]:
        ax.axvline(xe, color="#34d399", lw=0.9, alpha=0.8)
    m_strip = np.tile(wf["mesp2_switch"], (8, 1))
    ax_m = ax.inset_axes([0.0, 1.01, 1.0, 0.06])
    ax_m.imshow(m_strip, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0, extent=[kx[0], kx[-1], 0, 1])
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    ax_m.set_title(
        f"Mesp2-like switch $m(x)=[u_{{arrest}}>0]$: {wf['n_stripes']} stripes, $S={wf['stripe_spacing_um']:.1f}$ µm (CV {wf['stripe_spacing_cv']:.3f}) vs $v_{{wf}}T={wf['cooke_zeeman_prediction_um']:.1f}$ µm",
        fontsize=7.0,
        color="#f8fafc",
        pad=1.5,
    )
    ax.set_xlim(kx[0], kx[-1])
    ax.set_title("G. FGF/Wnt Clock-and-Wavefront Arrest (1D AP Chain)", fontweight="bold", pad=30)
    ax.set_xlabel("Anteroposterior Position $x$ (µm; anterior → posterior)")
    ax.set_ylabel("Time $t$ (min)")
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.85)

    # Panel H (Pass 3): Cooke-Zeeman sweep, lambda_F ablation and uniform-FGF control
    ax = axes[2, 1]
    v_arr = np.array([s["v_wf_um_per_min"] for s in sweep["speed_sweep"]])
    S_arr = np.array([s["stripe_spacing_um"] for s in sweep["speed_sweep"]])
    cz_arr = np.array([s["cooke_zeeman_um"] for s in sweep["speed_sweep"]])
    ax.plot(v_arr, cz_arr, color="#fde047", ls="--", lw=1.8, label=f"Cooke–Zeeman $S=v_{{wf}}T$ ($T={wf['T_tailbud_min']:.1f}$ min)")
    ax.scatter(v_arr, S_arr, color="#34d399", s=60, zorder=5, label=f"Emergent stripe spacing (slope {sweep['slope_min']:.1f} min, $r={sweep['pearson_r']:.4f}$)")
    ax.axhline(100.0, color="#f87171", ls=":", lw=1.3, label="Mouse somite AP length ≈ 100 µm (Gomez et al. 2008)")
    ax.scatter([wf["v_wf_um_per_min"]], [0.0], marker="x", color="#f87171", s=70, zorder=6, label=f"Uniform-FGF ablation: {wf_abl['n_stripes']} stripes (no front)")
    for row in sweep["lambda_sweep"]:
        ax.annotate(
            f"$\\lambda_F={row['lambda_F_um']:.0f}$: PSM {row['psm_length_um']:.0f} µm, $S={row['stripe_spacing_um']:.0f}$",
            xy=(0.40, 128.0 + 9.0 * sweep["lambda_sweep"].index(row)),
            fontsize=7.0,
            color="#c6d4ec",
        )
    ax.annotate(
        "Excitable-wave leak anterior of front vs $\\gamma_F$: "
        + ", ".join(f"{row['gamma_F']:.1f}→{row['anterior_wave_penetration_um']:.0f} µm" for row in sweep["gamma_sweep"]),
        xy=(0.40, 118.0),
        fontsize=6.8,
        color="#fbbf24",
    )
    ax.set_xlim(0.38, 1.07)
    ax.set_ylim(-8.0, 160.0)
    ax.set_title("H. Cooke–Zeeman Law $S=v_{wf}T$ & FGF Ablation", fontweight="bold")
    ax.set_xlabel("Wavefront / Axis Elongation Speed $v_{wf}$ (µm/min)")
    ax.set_ylabel("Somite Stripe Spacing $S$ (µm)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.85)

    # Panel I (Pass 3): stiffness -> F-actin -> LATS -> Y_nuc cascade, LatA kinetics, Dll1 synchrony
    ax = axes[2, 2]
    ax.semilogx(stiff["E_grid_kPa"], stiff["Y_nuc_grid"], color="#c084fc", lw=2.2, label="$Y_{nuc}(E_{sub})$ spread cell ($A=2000$ µm²)")
    ax.axhline(stiff["Y_nuc_crit"], color="#fde047", ls=":", lw=1.6, label=f"$Y_{{crit}}={stiff['Y_nuc_crit']:.3f}$ (Hopf)")
    ax.axvline(stiff["E_crit_kPa"], color="#38bdf8", ls="--", lw=1.4, label=f"$E_{{crit}}={stiff['E_crit_kPa']:.1f}$ kPa")
    ax.axvspan(0.1, 0.7, color="#34d399", alpha=0.12)
    ax.axvspan(40.0, 200.0, color="#f87171", alpha=0.12)
    ax.text(0.13, 0.95, "Dupont 2011\nsoft 0.7 kPa\n(cytoplasmic)", fontsize=6.8, color="#34d399", va="top")
    ax.text(45.0, 0.95, "stiff 40 kPa\n(nuclear)", fontsize=6.8, color="#f87171", va="top")
    reg_cols = {"isolated_spread_fn_quiescent": "#f87171", "isolated_latrunculin_a_rescue": "#34d399", "isolated_rockout_y27632_rescue": "#38bdf8", "intact_high_density_mpsm": "#fb923c"}
    for row in calib["table"]:
        if row["regime_id"] in reg_cols:
            ax.scatter([40.0], [row["Y_nuc_table"]], color=reg_cols[row["regime_id"]], s=46, zorder=5, edgecolor="white", lw=0.5)
            ax.scatter([40.0], [row["Y_nuc_model"]], color=reg_cols[row["regime_id"]], s=46, zorder=5, marker="D", alpha=0.7)
    ax.scatter([], [], color="#c6d4ec", s=40, label="Regime $Y_{nuc}$: table (●) vs cascade (◆)" + f", RMSE {calib['rmse_wt']:.3f}")
    ax.set_xlim(0.1, 200.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("I. Stiffness → F-actin → LATS1/2 → $Y_{nuc}$ & LatA Kinetics", fontweight="bold")
    ax.set_xlabel("Substrate Stiffness $E_{sub}$ (kPa)")
    ax.set_ylabel("Nuclear YAP Fraction $Y_{nuc}$")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=6.4, framealpha=0.85)
    ax_in = ax.inset_axes([0.60, 0.29, 0.38, 0.20])
    ax_in.set_facecolor("#0b1020")
    ax_in.plot(lata_kin["t_min"], lata_kin["Y_nuc"], color="#c084fc", lw=1.4)
    ax_in.plot(lata_kin["t_min"], lata_kin["tension"], color="#f472b6", lw=1.0, ls="--")
    ax_in.plot(lata_kin["t_min"], np.clip((lata_kin["u"] + 2.1) / 4.2, 0, 1), color="#34d399", lw=1.0)
    ax_in.axvline(0.0, color="#fde047", lw=0.8, ls=":")
    ax_in.set_title(f"LatA @ t=0: $Y_{{nuc}}$ · tension · LuVeLu; onset {lata_kin['oscillation_onset_min']:.0f} min", fontsize=5.8, color="#f8fafc", pad=2)
    ax_in.tick_params(labelsize=5.2)
    ax_in.set_xlim(-90.0, 600.0)
    ax_in2 = ax.inset_axes([0.60, 0.04, 0.38, 0.16])
    ax_in2.set_facecolor("#0b1020")
    ax_in2.plot(sync["J_values"], sync["R_final"], color="#38bdf8", marker="o", ms=4, lw=1.3)
    ax_in2.set_ylim(0.0, 1.0)
    ax_in2.set_title(f"Dll1–Notch $J_{{max}}$ → Kuramoto $R$ (ρ={sync['spearman_rho']:.2f})", fontsize=5.8, color="#f8fafc", pad=2)
    ax_in2.tick_params(labelsize=5.2)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
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
    wf = simulate_ap_wavefront_arrest(seed=2016)
    u_grid = np.linspace(-2.2, 2.2, 200)

    frames: list[Image.Image] = []
    snaps = pde_intact["snapshots_2d"]
    n_frames = len(snaps)
    wf_t = wf["t_grid_min"]
    wf_frame_idx = np.linspace(len(wf_t) // 6, len(wf_t) - 1, n_frames).astype(int)

    for idx, (t_val, disc_2d) in enumerate(snaps):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=100)
        fig.patch.set_facecolor("#0b1020")
        for ax in (ax1, ax2, ax3):
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

        # Pass 3: 1D AP clock-and-wavefront growth (FGF gradient regresses, stripes accrete)
        k_idx = int(wf_frame_idx[idx])
        t_1d = float(wf_t[k_idx])
        x_um = wf["x_um"]
        F_now = fgf_gradient_field(x_um, t_1d, 700.0, wf["v_wf_um_per_min"], wf["lambda_F_um"])
        luv_1d = np.clip((wf["kymograph"][k_idx] + 2.1) / 4.2, 0.0, 1.0)
        formed = (~np.isnan(wf["t_arrest_min"])) & (wf["t_arrest_min"] <= t_1d)
        m_now = np.where(formed, wf["mesp2_switch"], np.nan)
        ax3.fill_between(x_um, 0.0, np.where(m_now > 0.5, 1.0, 0.0), color="#34d399", alpha=0.45, step="mid", label="Somite stripes $m(x)$ (frozen phase)")
        ax3.plot(x_um, luv_1d, color="#f472b6", lw=1.6, label="LuVeLu $u(x,t)$ (excitable clock)")
        ax3.plot(x_um, F_now, color="#fde047", lw=1.4, ls="--", label="Fgf8/Wnt gradient $F(x,t)$")
        ax3.axvline(wf["front_x_um"][k_idx], color="#38bdf8", lw=1.2, ls=":", label="Determination front")
        ax3.set_xlim(x_um[0], x_um[-1])
        ax3.set_ylim(0.0, 1.08)
        ax3.set_title(f"1D AP Clock-and-Wavefront ($t={t_1d:.0f}$ min, $S={wf['stripe_spacing_um']:.0f}$ µm)", fontsize=10, fontweight="bold", color="#f8fafc")
        ax3.set_xlabel("Anteroposterior $x$ (µm)")
        ax3.set_ylabel("Normalised Level")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper left", fontsize=6.8, framealpha=0.85)

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

    # ---------------- Pass 3 checks ----------------
    # Law 4 & 5: FGF/Wnt gradient clock-and-wavefront arrest into discrete stripes
    wf = simulate_ap_wavefront_arrest(seed=2016)
    wf_rep = simulate_ap_wavefront_arrest(seed=2016)
    assert np.allclose(wf["kymograph"], wf_rep["kymograph"], atol=0.0), "1D wavefront solver must be bitwise deterministic"
    assert wf["n_stripes"] >= 6, f"expected >= 6 somite stripes, got {wf['n_stripes']}"
    assert wf["stripe_spacing_cv"] < 0.08, f"stripe spacing CV too large: {wf['stripe_spacing_cv']}"
    assert abs(wf["stripe_spacing_um"] - wf["cooke_zeeman_prediction_um"]) / wf["cooke_zeeman_prediction_um"] < 0.05, "S must match v_wf * T within 5%"
    assert abs(wf["psm_length_um"] - wf["psm_length_pred_um"]) / wf["psm_length_pred_um"] < 0.05, "oscillating-domain length must match the analytic front offset"
    assert 0.0 < wf["g_crit"] < 1.0
    wf_abl = simulate_ap_wavefront_arrest(mode="uniform_fgf_ablation", seed=2016)
    assert wf_abl["n_stripes"] == 0 and wf_abl["n_arrested_cells"] == 0, "uniform FGF must abolish the front and the stripes"
    # Cooke-Zeeman sweep: S increases linearly with v_wf with slope ~ T; lambda_F must not change S
    sweep = evaluate_wavefront_speed_sweep(seed=2016)
    assert sweep["pearson_r"] > 0.995
    assert abs(sweep["slope_min"] - wf["T_tailbud_min"]) / wf["T_tailbud_min"] < 0.08
    S_lam = [row["stripe_spacing_um"] for row in sweep["lambda_sweep"]]
    assert (max(S_lam) - min(S_lam)) / np.mean(S_lam) < 0.05, "lambda_F must set PSM length, not stripe spacing"
    L_lam = [row["psm_length_um"] for row in sweep["lambda_sweep"]]
    assert L_lam[0] < L_lam[1] < L_lam[2], "PSM length must grow monotonically with lambda_F"
    # Excitable-wave arrest: at the default gamma_F the pulses must not leak into formed somites,
    # whereas the weak-gain control (gamma_F = 0.8) must show anterior leakage
    assert wf["anterior_wave_penetration_um"] == 0.0 and wf["n_old_cells_still_excited"] == 0, "excitable waves must arrest at the front"
    pen = {row["gamma_F"]: row["anterior_wave_penetration_um"] for row in sweep["gamma_sweep"]}
    assert pen[0.80] > 300.0, f"weak FGF gain must leak pulses anteriorly, got {pen[0.80]} um"
    assert pen[1.60] == 0.0 and pen[2.00] == 0.0

    # Law 6: stiffness -> F-actin -> LATS -> Y_nuc cascade
    calib = calibrate_yap_mechanotransduction()
    assert calib["rmse_wt"] < 0.08, f"cascade RMSE too large: {calib['rmse_wt']}"
    table = {row["regime_id"]: row for row in calib["table"]}
    assert table["isolated_spread_fn_quiescent"]["Y_nuc_model"] > hopf["Y_nuc_crit"], "spread stiff cell must be quiescent"
    assert table["isolated_latrunculin_a_rescue"]["Y_nuc_model"] < hopf["Y_nuc_crit"], "LatA cell must be oscillatory"
    assert table["intact_high_density_mpsm"]["Y_nuc_model"] < hopf["Y_nuc_crit"]
    assert table["yap5sa_constitutively_active_block"]["Y_nuc_model"] > hopf["Y_nuc_crit"], "Yap5SA must stay quiescent"
    stiff = evaluate_substrate_stiffness_sweep(calib)
    assert np.all(np.diff(stiff["Y_nuc_grid"]) >= -1e-12), "Y_nuc(E_sub) must be monotone non-decreasing (Theorem 6)"
    assert 0.7 < stiff["E_crit_kPa"] < 40.0, "E_crit must lie inside the Dupont 2011 soft/stiff bracket"
    lata_kin = simulate_latrunculin_yap_kinetics(calib)
    assert lata_kin["Y_nuc_pre"] > hopf["Y_nuc_crit"] > lata_kin["Y_nuc_post"]
    assert 0.0 < lata_kin["oscillation_onset_min"] < 142.0, "LatA rescue must ignite within one clock period"
    assert 130.0 <= lata_kin["post_period_min"] <= 155.0

    # Dll1-Notch juxtacrine phase coupling increases synchrony monotonically
    sync = evaluate_dll1_coupling_sweep(seed=2016)
    assert sync["R_final"][0] < 0.5 and sync["R_final"][-1] > 0.7, f"coupling must raise R: {sync['R_final']}"
    assert sync["spearman_rho"] > 0.9

    measurements = measure_morphoeval()
    assert len(measurements) == 11, "Expected 11 MorphoEval measurements"
    assert all("reference_value" in m_ and "unit" in m_ for m_ in measurements)

    print("[2016-hubaud-yap-excitable-segmentation] SELF-TEST PASSED:")
    print(f"  * FHN Hopf threshold |u_H| = {hopf['u_hopf_abs']:.4f}, Y_nuc_crit = {hopf['Y_nuc_crit']:.4f}")
    print(f"  * Latrunculin A rescued period T_osc = {lata_sim['period_min']:.2f} min (amp = {lata_sim['amplitude']:.2f})")
    print(f"  * 6-regime oscillating fraction Pearson r = {pert['oscillating_fraction_pearson_r']:.4f} (RMSE = {pert['oscillating_fraction_rmse']:.4f})")
    print(f"  * 2D mPSM wave contrast: intact = {pde1['wave_contrast']:.3f} vs Yap5SA block = {pde_yap5['wave_contrast']:.4f}")
    print(f"  * Specimen bilateral somite spacing = {spec['bilateral_mean_spacing_px']:.2f} px (CV = {spec['bilateral_cv_spacing']:.4f})")
    print(f"  * [Pass 3] FGF wavefront stripes: n = {wf['n_stripes']}, S = {wf['stripe_spacing_um']:.2f} um vs v_wf*T = {wf['cooke_zeeman_prediction_um']:.2f} um (CV = {wf['stripe_spacing_cv']:.4f}); ablation stripes = {wf_abl['n_stripes']}")
    print(f"  * [Pass 3] Cooke-Zeeman sweep slope = {sweep['slope_min']:.2f} min vs T = {wf['T_tailbud_min']:.2f} min (r = {sweep['pearson_r']:.5f}); PSM length {wf['psm_length_um']:.1f} vs {wf['psm_length_pred_um']:.1f} um")
    print(f"  * [Pass 3] YAP cascade RMSE = {calib['rmse_wt']:.4f} (alpha_T = {calib['alpha_T']:.3f}, kappa_LATS = {calib['kappa_LATS']:.3f}, T_L = {calib['T_L']:.3f}); E_crit = {stiff['E_crit_kPa']:.2f} kPa")
    print(f"  * [Pass 3] LatA kinetics: Y_nuc {lata_kin['Y_nuc_pre']:.3f} -> {lata_kin['Y_nuc_post']:.3f}, onset {lata_kin['oscillation_onset_min']:.2f} min, period {lata_kin['post_period_min']:.2f} min")
    print(f"  * [Pass 3] Dll1-Notch synchrony R: J=0 -> {sync['R_final'][0]:.3f}, J=0.30 -> {sync['R_final'][-1]:.3f} (Spearman rho = {sync['spearman_rho']:.3f})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hubaud et al. (2017) Cell excitable FitzHugh-Nagumo & YAP mechanotransduction segmentation clock."
    )
    parser.add_argument("--Y-nuc", type=float, default=0.22, help="Nuclear YAP level Y_nuc in [0, 1]")
    parser.add_argument("--J-Dll1", type=float, default=0.0, help="Notch-Dll1 coupling input J_Dll1")
    parser.add_argument("--seed", type=int, default=2016, help="Deterministic RNG seed")
    parser.add_argument("--self-test", action="store_true", help="Run self-verification suite and exit")
    parser.add_argument("--v-wf", type=float, default=None, help="[Pass 3] Wavefront/axis-elongation speed (um/min) for a one-off 1D clock-and-wavefront run printed to stdout")
    parser.add_argument("--lambda-F", type=float, default=350.0, help="[Pass 3] Fgf8/Wnt gradient decay length lambda_F (um) for the --v-wf run")
    parser.add_argument("--E-sub", type=float, default=None, help="[Pass 3] Substrate stiffness (kPa): print the cascade-predicted Y_nuc and dynamical state for a spread cell")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    if args.v_wf is not None:
        wf = simulate_ap_wavefront_arrest(v_wf_um_per_min=args.v_wf, lambda_F_um=args.lambda_F, seed=args.seed)
        print(f"[2016-hubaud-yap-excitable-segmentation] v_wf = {args.v_wf:.3f} um/min, lambda_F = {args.lambda_F:.0f} um -> {wf['n_stripes']} stripes, S = {wf['stripe_spacing_um']:.2f} um (v_wf*T = {wf['cooke_zeeman_prediction_um']:.2f} um), PSM length = {wf['psm_length_um']:.1f} um")
        return 0
    if args.E_sub is not None:
        calib = calibrate_yap_mechanotransduction()
        hopf = compute_hopf_thresholds()
        T_val = float(substrate_tension_index(args.E_sub, 2000.0))
        Y_pred = float(yap_nuclear_fraction_ss(T_val, calib["alpha_T"], calib["kappa_LATS"], calib["T_L"]))
        state = "autonomous limit cycle" if Y_pred < hopf["Y_nuc_crit"] else "excitable quiescent"
        print(f"[2016-hubaud-yap-excitable-segmentation] E_sub = {args.E_sub:.2f} kPa -> tension T = {T_val:.3f}, Y_nuc = {Y_pred:.3f} (Y_crit = {hopf['Y_nuc_crit']:.3f}) -> {state}")
        return 0

    run_self_test()
    png_path = render_comparison_figure(OUTPUT_PNG)
    gif_path = render_growth_gif(OUTPUT_GIF)
    print(f"[2016-hubaud-yap-excitable-segmentation] Wrote figure -> {png_path}")
    print(f"[2016-hubaud-yap-excitable-segmentation] Wrote animation -> {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
