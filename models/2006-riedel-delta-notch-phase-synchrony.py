#!/usr/bin/env python3
"""
2006-riedel-delta-notch-phase-synchrony.py — Zebrafish Delta–Notch Delayed Phase Synchrony
===========================================================================================

Faithful mathematical, numerical, and morphometric reproduction of:
  1. Riedel-Kruse, I. H., Müller, C. & Oates, A. C. (2007).
     "Synchrony dynamics during initiation, failure, and rescue of the
     segmentation clock." Science 317(5846):1911-1915.
     DOI: 10.1126/science.1142538
  2. Morelli, L. G., Ares, S., Herrgen, L., Schröter, C., Jülicher, F. &
     Oates, A. C. (2009). "Delayed coupling theory of vertebrate segmentation."
     HFSP Journal 3(1):55-66.
     DOI: 10.2976/1.3027088
  3. Herrgen, L., Ares, S., Morelli, L. G., Schröter, C., Jülicher, F. &
     Oates, A. C. (2010). "Intercellular coupling regulates the period of the
     segmentation clock." Current Biology 20(14):1244-1253.
     DOI: 10.1016/j.cub.2010.06.034
  4. Horikawa, K., Ishimatsu, K., Yoshimoto, E., Kondo, S. & Takeda, H. (2006).
     "Noise-resistant and synchronized oscillation of the segmentation clock."
     Nature 441(7094):719-723.
     DOI: 10.1038/nature04861
  5. Itoh, M., Kim, C.-H., Palardy, G., Oda, T., Jiang, Y.-J., Maust, D., Yeo,
     S.-Y., Lorick, K., Wright, G. J., Ariza-McNaughton, L., Weissman, A. M.,
     Lewis, J., Chandrasekharappa, S. C. & Chitnis, A. B. (2003). "Mind bomb is
     a ubiquitin ligase that is essential for efficient activation of Notch
     signaling by Delta." Developmental Cell 4(1):67-82.
     DOI: 10.1016/S1534-5807(02)00409-4
  6. Dubrulle, J., McGrew, M. J. & Pourquié, O. (2001). "FGF signaling controls
     somite boundary position and regulates segmentation clock control of
     spatiotemporal Hox gene activation." Cell 106(2):219-232.
     DOI: 10.1016/S0092-8674(01)00437-8
  7. Soroldoni, D., Jörg, D. J., Morelli, L. G., Richmond, D. L., Schindelin, J.,
     Jülicher, F. & Oates, A. C. (2014). "A Doppler effect in embryonic pattern
     formation." Science 345(6193):222-225.
     DOI: 10.1126/science.1253089

Pass 3 remediation (superseding audit `2026-10-19-delta-notch-phase-synchrony-audit-v2`):
  * The uniform 8x8 periodic lattice of Pass 2 is replaced by an anteroposterior (AP) x lateral
    PSM lattice (24 x 12 cells, 4 cells per somite length, PSM length L = 6 somite lengths) whose
    cells are advected anteriorly by axis elongation while a tailbud-anchored FGF8/Wnt gradient
    imposes the Morelli (2009) frequency profile omega(x) (Law 4). This produces kinematic
    (phase) waves that travel anteriorly and slow down toward the determination front.
  * The Delta-Notch coupling strength is now a dynamical variable kappa(t) = kappa_max * u(t) * g(t)
    where u(t) is the Mind bomb (Mib1)-ubiquitinated, signalling-competent DeltaC fraction
    (Itoh et al. 2003; Law 5) and g(t) is the gamma-secretase (DAPT) cleavage gate with first-order
    washout clearance kinetics.
  * ALD and PRP are now EMERGENT: they are read from the first crossing of the lateral Kuramoto
    order parameter at the determination front below / above R_crit, shifted by the determination-to-
    boundary lag Delta_n_det (Law 6). Pass 2 hard-coded ALD = s_in + 5.5 and PRP = s_out + 5.5; those
    analytic values are still reported as the Riedel-Kruse (2007) prediction for comparison.
  * The ad hoc instantaneous mean-field restoring term and the 0.35/0.65 local/mean-field blend of
    Pass 2 (neither in the source model) are removed: the lattice is the pure nearest-neighbour
    delayed Kuramoto lattice of Riedel-Kruse (2007) Eq. 1 / Morelli (2009) Eq. 1.
  Declared reduced assumptions (carried as Pass 4 remediation items):
  * omega(x) = omega_0 * exp(-xi / lambda_F) is a single-exponential FGF8/Wnt-slaved frequency
    profile (xi = distance from the tailbud in somite lengths; f_A = omega(front)/omega_0 = 0.5),
    not a profile digitised from Morelli (2009) Fig. 3 or Soroldoni (2014) kymographs.
  * The coupling delay scales with the local autonomous period, tau_c(x) * omega(x) = tau_c * omega_0
    (the gradient slows the whole DeltaC synthesis/trafficking cycle); otherwise the fixed 20.5-min
    delay would turn repulsive (cos(omega(x) tau_c) < 0) in the slowed anterior PSM.
  * Mib1 kinetics (k_ub = 0.08 min^-1, k_endo = 0.02 min^-1) and DAPT clearance (tau_wash = 8 min,
    dose/IC50 = 20, Hill n = 2) are this script's reduced rate constants, not measured values.
  * The cell-to-cell autonomous frequency spread sigma_omega = 0.014 rad/min (5.6 % period CV) and
    phase diffusivity D_phi = 0.003 rad^2/min are calibrated so the emergent desynchronisation lag
    matches the Riedel-Kruse (2007) ALD data, exactly as the paper fitted its desynchronisation rate.

Governing Mathematical Laws (verified in `morphoeval/lean/loops/Loop_023_delta_notch_phase_synchrony.lean`):
-------------------------------------------------------------------------------------------------------------
1. Riedel-Kruse et al. (2007) & Morelli et al. (2009) Delayed Kuramoto Phase-Oscillator Lattice &
   Collective Frequency Shift (`riedel_2007_collective_frequency_shift_and_dapt_limit`):
       dtheta_i(t)/dt = omega_i + (kappa(t) / z_i) * sum_{j in N(i)} sin(theta_j(t - tau_c,j) - theta_i(t)) + xi_i(t)
   where theta_i(t) is the segmentation clock phase of presomitic mesoderm (PSM) cell i,
   omega_i = omega(x_i) + delta_omega_i is the autonomous single-cell frequency (omega_0 = 2*pi/T_0,
   T_0 = 25.0 min in the tailbud), kappa(t) is the intercellular Delta-Notch coupling strength,
   z_i = |N(i)| is the coordination number, tau_c is the intercellular DeltaC ligand synthesis,
   trafficking, and Notch receptor cleavage transduction delay, and xi_i(t) is Gaussian white
   phase noise (<xi_i(t) xi_j(t')> = 2 D_phi delta_ij delta(t - t')).
   In the spatially uniform synchronized state theta_i(t) = Omega * t, the collective
   oscillation frequency Omega satisfies the transcendental self-consistency law:
       Omega = omega_0 - kappa * sin(Omega * tau_c),    T_sync = 2 * pi / Omega.

2. Morelli et al. (2009) Linear Stability Stiffness Criterion
   (`morelli_2009_in_phase_stability_vs_antiphase_instability`):
       kappa_eff = kappa * cos(Omega * tau_c).
   In-phase synchronization is linearly stable if and only if kappa_eff > 0.

3. Riedel-Kruse et al. (2007) Kuramoto Order Parameter R(t), Anterior Limit of Defects (ALD),
   and Posterior Rescue Point (PRP) DAPT Pulse-Chase Law
   (`riedel_2007_ald_prp_pulse_chase_window_invariance`):
       R(t) = |(1 / N) * sum_{j=1}^{N} exp(i * theta_j(t))| in [0, 1].
       ALD(s_in) = s_in + Delta_n_ALD,  PRP(s_out) = s_out + Delta_n_PRP   (Delta_n ~ 5.5 somites).

4. Morelli et al. (2009) Kinematic Phase-Wave Law for an advected frequency profile
   (`morelli_2009_kinematic_wave_phase_lag_law`):
   In the frame co-moving with the tailbud, cells flow anteriorly at speed v = S / T_seg and a
   cell at position x oscillates at omega_eff(x) = Omega * c(x), c(x) = omega(x)/omega_0.
   The steady phase pattern phi(x, t) = Omega * t - Phi(x) carries the local wave number
       k(x) = dPhi/dx = (Omega - omega_eff(x)) / v = (Omega / v) * (1 - c(x)) >= 0,
   so the phase lags anteriorly (waves travel posterior -> anterior with speed Omega / k(x) that
   decreases toward the front), the number of waves across the PSM is
       N_w = (1 / 2 pi) * Int_0^L k(x) dx = (1 / S) * Int_0^L (1 - c(x)) dx,
   and a uniform profile (c = 1) carries no waves (k = 0).

5. Itoh et al. (2003) Mind bomb (Mib1) DeltaC Ubiquitination Gate
   (`itoh_2003_mib1_ubiquitination_gate_monotone`):
       du/dt = k_ub * M * (1 - u) - k_endo * u,     u_ss = k_ub M / (k_ub M + k_endo),
       kappa(t) = kappa_max * u(t) * g(t),          g(t) = 1 / (1 + (C_DAPT(t)/IC50)^n),
   where u is the Mib1-ubiquitinated, endocytosis-competent (signalling-active) DeltaC fraction,
   M is the Mib1 E3-ligase activity (M = 1 wild type, M = 0 mib^ta52b), and C_DAPT(t) decays with
   first-order clearance tau_wash after washout. u_ss = 0 at M = 0, u_ss <= 1, and u_ss is monotone
   increasing in M, so kappa never exceeds kappa_max and vanishes in mib mutants as under DAPT.

6. Emergent Defect-Span Law (`riedel_2007_emergent_defect_span_independent_of_determination_lag`):
       ALD = s_cross + Delta_n_det,  PRP = s_recross + Delta_n_det  =>  PRP - ALD = s_recross - s_cross,
   where s_cross / s_recross are the somite stages at which the determination-front lateral order
   parameter falls below / recovers above R_crit; the defective span is independent of the
   determination-to-boundary maturation lag Delta_n_det.
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
from scipy import optimize, signal, stats

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "delta_notch_phase_synchrony"
REFERENCE_JSON = DATASET_DIR / "delta_notch_phase_synchrony_reference.json"
SPECIMEN_JPG = DATASET_DIR / "zebrafish_psm_synchrony_specimen.jpg"
SCOUT_MANIFEST_JSON = DATASET_DIR / "scout_manifest.json"
OUTPUT_PNG = ROOT / "docs" / "media" / "delta_notch_phase_synchrony_comparison.png"
OUTPUT_GIF = ROOT / "docs" / "media" / "delta_notch_phase_synchrony_growth.gif"

EXPECTED_SHA256 = {
    "datasets/delta_notch_phase_synchrony/zebrafish_psm_synchrony_specimen.jpg": (
        "8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f"
    ),
    "datasets/delta_notch_phase_synchrony/scout_manifest.json": (
        "e3250a43b4bc0c22e8061af719f2506b5ca6f3208aa0f2cd6239a6a9b5e01718"
    ),
}

# ---------------------------------------------------------------------------
# Canonical zebrafish PSM parameters (Riedel-Kruse 2007 / Morelli 2009 / Herrgen 2010) and the
# Pass 3 reduced-mechanism rate constants (declared in the module docstring).
# ---------------------------------------------------------------------------
T0_MIN = 25.0                     # autonomous tailbud period (min)
KAPPA_WT_PER_MIN = 0.070          # wild-type Delta-Notch coupling (min^-1)
TAU_C_MIN = 20.5                  # intercellular coupling delay in the tailbud (min)
D_PHI_RAD2_PER_MIN = 0.003        # phase diffusivity (rad^2/min), calibrated to ALD data
SIGMA_OMEGA_RAD_PER_MIN = 0.014   # cell-to-cell autonomous frequency spread (rad/min)
F_ANTERIOR = 0.5                  # omega(front)/omega_0 of the FGF8/Wnt frequency profile
N_AP_CELLS = 24                   # AP lattice cells (PSM length L = 6 somite lengths)
N_LAT_CELLS = 12                  # lateral (mediolateral) lattice cells, periodic
CELLS_PER_SOMITE = 4              # AP cells per somite length S
R_CRIT = 0.50                     # boundary coherence threshold
DELTA_N_DET = 2.5                 # determination front -> morphological boundary lag (somites)
DELTA_N_ANALYTIC = 5.5            # Riedel-Kruse (2007) analytic ALD/PRP lag (somites)
NO_DEFECT_SENTINEL = 31.0         # total zebrafish somite count used when no defect forms
K_UB_PER_MIN = 0.08               # Mib1 ubiquitination rate of membrane DeltaC (min^-1, M = 1)
K_ENDO_PER_MIN = 0.02             # ubiquitinated-DeltaC endocytic turnover (min^-1)
TAU_WASH_MIN = 8.0                # DAPT first-order clearance time after washout (min)
DAPT_DOSE_RATIO = 20.0            # DAPT concentration / IC50 during treatment
DAPT_HILL_N = 2.0                 # Hill coefficient of the gamma-secretase gate
DT_MIN = 0.5                      # Euler-Maruyama time step (min)


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
    digests["datasets/delta_notch_phase_synchrony/delta_notch_phase_synchrony_reference.json"] = (
        hashlib.sha256(REFERENCE_JSON.read_bytes()).hexdigest()
    )
    return digests


def load_reference_data() -> dict[str, Any]:
    """Load `datasets/delta_notch_phase_synchrony/delta_notch_phase_synchrony_reference.json`."""
    return json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))


# ============================================================================
# Section 2: Delayed Kuramoto Collective Frequency & Linear Stability Analysis
#            (Riedel-Kruse et al. 2007 Science Eq. 1; Morelli et al. 2009 HFSP J. Eqs. 1-5)
# ============================================================================
def solve_collective_frequency_and_stability(
    T0_min: float = T0_MIN,
    kappa_per_min: float = KAPPA_WT_PER_MIN,
    tau_c_min: float = TAU_C_MIN,
) -> dict[str, Any]:
    """Law 1 & Law 2 (`riedel-delta-notch-phase-synchrony-1`, `-2`):
    Solve the delayed Kuramoto collective synchronization frequency equation
        Omega = omega_0 - kappa * sin(Omega * tau_c)
    around omega_0 = 2*pi / T_0 and compute the linearized restoring stiffness
        kappa_eff = kappa * cos(Omega * tau_c)   (in-phase stable iff kappa_eff > 0).
    Ref: Riedel-Kruse et al. (2007) Science 317:1911-1915 SOM; Morelli et al. (2009) HFSP J.
         3:55-66 Eqs. 2-5; Lean 4 theorems `riedel_2007_collective_frequency_shift_and_dapt_limit`
         and `morelli_2009_in_phase_stability_vs_antiphase_instability`.
    """
    omega_0 = 2.0 * math.pi / T0_min
    if kappa_per_min <= 0.0 or tau_c_min <= 0.0:
        Omega = omega_0
        kappa_eff = kappa_per_min
    else:
        candidates: list[float] = []
        grid = np.linspace(max(1e-4, omega_0 - 1.5 * kappa_per_min), omega_0 + 1.5 * kappa_per_min, 200)
        res = grid - omega_0 + kappa_per_min * np.sin(grid * tau_c_min)
        for idx in range(len(grid) - 1):
            if res[idx] * res[idx + 1] <= 0.0:
                root = float(
                    optimize.brentq(
                        lambda w: w - omega_0 + kappa_per_min * math.sin(w * tau_c_min),
                        float(grid[idx]),
                        float(grid[idx + 1]),
                    )
                )
                candidates.append(root)
        if not candidates:
            Omega = omega_0 - kappa_per_min * math.sin(omega_0 * tau_c_min)
        else:
            # Select the branch with maximal restoring stiffness (the physically attracting branch)
            candidates.sort(key=lambda w: kappa_per_min * math.cos(w * tau_c_min), reverse=True)
            Omega = candidates[0]
        kappa_eff = kappa_per_min * math.cos(Omega * tau_c_min)

    T_sync = 2.0 * math.pi / Omega
    return {
        "omega_0_rad_per_min": round(omega_0, 6),
        "T0_min": round(T0_min, 4),
        "kappa_per_min": round(kappa_per_min, 6),
        "tau_c_min": round(tau_c_min, 4),
        "Omega_rad_per_min": round(Omega, 6),
        "T_sync_min": round(T_sync, 4),
        "phase_delay_rad": round(Omega * tau_c_min, 4),
        "kappa_eff_per_min": round(kappa_eff, 6),
        "in_phase_stable": bool(kappa_eff > 0.0),
        "frequency_ratio_Omega_over_omega0": round(Omega / omega_0, 4),
    }


_OMEGA_CACHE: dict[tuple[float, float, float], float] = {}


def _collective_frequency_cached(T0_min: float, kappa_per_min: float, tau_c_min: float) -> float:
    """Memoised collective frequency Omega(kappa) used by the somite-stage clock (kappa quantised to 1e-4)."""
    key = (round(T0_min, 6), round(max(kappa_per_min, 0.0), 4), round(tau_c_min, 6))
    if key not in _OMEGA_CACHE:
        _OMEGA_CACHE[key] = solve_collective_frequency_and_stability(T0_min, key[1], tau_c_min)["Omega_rad_per_min"]
    return _OMEGA_CACHE[key]


# ============================================================================
# Section 3a: FGF8/Wnt Frequency Profile & Mib1 / DAPT Coupling Kinetics
#             (Morelli et al. 2009 HFSP J. frequency profile; Dubrulle et al. 2001 Cell FGF8 gradient;
#              Itoh et al. 2003 Dev Cell Mib1; Riedel-Kruse et al. 2007 DAPT washout)
# ============================================================================
def fgf_wnt_frequency_profile(
    n_ap_cells: int = N_AP_CELLS,
    cells_per_somite: int = CELLS_PER_SOMITE,
    f_anterior: float = F_ANTERIOR,
    T0_min: float = T0_MIN,
    gradient_on: bool = True,
) -> dict[str, Any]:
    """Law 4 (`riedel-delta-notch-phase-synchrony-4`): tailbud-anchored FGF8/Wnt gradient
    c(xi) = exp(-xi / lambda_F) (xi = distance from the tailbud in somite lengths S, Dubrulle et al.
    2001) slaving the autonomous frequency profile omega(xi) = omega_0 * c(xi) of Morelli et al. (2009),
    with lambda_F = L / ln(1 / f_A) so that omega(front) = f_A * omega_0.
    Returns per-column omega_x, the gradient c_x, the period-scaled delay factor omega_0/omega_x and the
    analytic waves-per-PSM N_w = (1/S) * Int (1 - c) dx (trapezoid over the lattice columns)."""
    omega_0 = 2.0 * math.pi / T0_min
    L_somites = n_ap_cells / float(cells_per_somite)
    xi = (n_ap_cells - 1 - np.arange(n_ap_cells)) / float(cells_per_somite)  # column 0 = anterior front
    if gradient_on and f_anterior < 1.0:
        lambda_F = L_somites / math.log(1.0 / f_anterior)
        c_x = np.exp(-xi / lambda_F)
    else:
        lambda_F = math.inf
        c_x = np.ones(n_ap_cells)
    omega_x = omega_0 * c_x
    dx_somite = 1.0 / float(cells_per_somite)
    n_waves_analytic = float(np.sum(1.0 - c_x) * dx_somite)  # (1/S) * Int (1 - c) dx, S = 1 somite length
    return {
        "xi_somites": xi,
        "c_x": c_x,
        "omega_x_rad_per_min": omega_x,
        "delay_scale_x": 1.0 / c_x,
        "lambda_F_somites": lambda_F,
        "L_psm_somites": L_somites,
        "f_anterior": float(f_anterior),
        "n_waves_analytic": round(n_waves_analytic, 4),
        "anterior_wavelength_over_S_analytic": round(1.0 / (1.0 - f_anterior), 4) if f_anterior < 1.0 else math.inf,
    }


def mib1_ubiquitination_steady_state(
    mib1_dose: float,
    k_ub_per_min: float = K_UB_PER_MIN,
    k_endo_per_min: float = K_ENDO_PER_MIN,
) -> float:
    """Law 5 (`riedel-delta-notch-phase-synchrony-5`): steady-state Mib1-ubiquitinated signalling-
    competent DeltaC fraction u_ss = k_ub M / (k_ub M + k_endo) (Itoh et al. 2003 Dev Cell 4:67-82;
    Lean theorem `itoh_2003_mib1_ubiquitination_gate_monotone`)."""
    num = k_ub_per_min * max(mib1_dose, 0.0)
    return float(num / (num + k_endo_per_min)) if (num + k_endo_per_min) > 0.0 else 0.0


def dapt_cleavage_gate(conc_over_ic50: float, hill_n: float = DAPT_HILL_N) -> float:
    """gamma-secretase cleavage gate g = 1 / (1 + (C/IC50)^n) (Riedel-Kruse et al. 2007 DAPT titration)."""
    return float(1.0 / (1.0 + max(conc_over_ic50, 0.0) ** hill_n))


def effective_coupling_for_mib1_dose(mib1_dose: float, kappa_wt_per_min: float = KAPPA_WT_PER_MIN) -> dict[str, float]:
    """kappa(M) = kappa_max * u_ss(M) with kappa_max normalised so that M = 1 gives kappa_WT, and the
    resulting Morelli (2009) stiffness kappa_eff(M) = kappa(M) * cos(Omega(M) tau_c)."""
    u_wt = mib1_ubiquitination_steady_state(1.0)
    kappa_max = kappa_wt_per_min / u_wt
    u = mib1_ubiquitination_steady_state(mib1_dose)
    kappa = kappa_max * u
    stab = solve_collective_frequency_and_stability(T0_MIN, kappa, TAU_C_MIN)
    return {
        "mib1_dose": float(mib1_dose),
        "u_ss": round(u, 4),
        "kappa_per_min": round(kappa, 5),
        "kappa_max_per_min": round(kappa_max, 5),
        "kappa_eff_per_min": stab["kappa_eff_per_min"],
        "T_sync_min": stab["T_sync_min"],
    }


# ============================================================================
# Section 3b: Advected AP x Lateral Delayed Kuramoto PSM Lattice with Kinematic Waves,
#             Mib1/DAPT Coupling Kinetics and Emergent ALD / PRP
#             (Riedel-Kruse et al. 2007 Science Figs. 1-4; Morelli et al. 2009 HFSP J.)
# ============================================================================
def simulate_delayed_kuramoto_psm_lattice(
    n_ap_cells: int = N_AP_CELLS,
    n_lat_cells: int = N_LAT_CELLS,
    cells_per_somite: int = CELLS_PER_SOMITE,
    n_somites: int = 22,
    T0_min: float = T0_MIN,
    kappa_per_min: float = KAPPA_WT_PER_MIN,
    tau_c_min: float = TAU_C_MIN,
    D_phi_rad2_per_min: float = D_PHI_RAD2_PER_MIN,
    sigma_omega_rad_per_min: float = SIGMA_OMEGA_RAD_PER_MIN,
    f_anterior: float = F_ANTERIOR,
    gradient_on: bool = True,
    s_in_somite: float | None = None,
    s_out_somite: float | None = None,
    mib1_dose: float = 1.0,
    mib1_restore_somite: float | None = None,
    tau_wash_min: float = TAU_WASH_MIN,
    dapt_dose_ratio: float = DAPT_DOSE_RATIO,
    k_ub_per_min: float = K_UB_PER_MIN,
    k_endo_per_min: float = K_ENDO_PER_MIN,
    R_init_target: float = 0.96,
    R_crit: float = R_CRIT,
    delta_n_det: float = DELTA_N_DET,
    seed: int = 2007,
) -> dict[str, Any]:
    """Laws 1-6: integrate the advected AP x lateral delayed Kuramoto PSM lattice
        dtheta_i/dt = omega_i + (kappa(t)/z_i) * sum_{j in N(i)} sin(theta_j(t - tau_c,j) - theta_i(t)) + xi_i(t)
    (Euler-Maruyama, dt = 0.5 min, nearest-neighbour coupling: open AP boundaries, periodic laterally).
      - Frequency profile omega_i = omega_0 c(x_i) + delta_omega_i (Law 4) is Eulerian (anchored to the
        tailbud); cells, their phase histories and their intrinsic delta_omega_i are advected one column
        anteriorly every 1/cells_per_somite of a segmentation period (Lagrangian), the front column exits
        into the somite and a new tailbud column is seeded from the current tailbud phase.
      - The per-column delay tau_c,j = tau_c * omega_0 / omega(x_j) is scaled with the local period.
      - kappa(t) = kappa_max * u(t) * g(t): u obeys the Mib1 ubiquitination ODE (Law 5) with Mib1 activity
        M = 0 before `mib1_restore_somite` and `mib1_dose` afterwards (M = mib1_dose throughout if None);
        g is the DAPT gamma-secretase gate: C/IC50 = dapt_dose_ratio for s in [s_in, s_out), then
        first-order clearance exp(-(t - t_out)/tau_wash).
      - Somite stage s(t) advances at the instantaneous collective frequency Omega(kappa(t)) / 2 pi, so
        the DAPT period lengthening of Herrgen et al. (2010) is emergent in the somite count.
      - R_front(t) = lateral Kuramoto order parameter of the anterior somite-length block (mean over its
        columns of |<exp(i theta)>_lateral|), smoothed over half a segmentation period. Emergent
        ALD = s_cross + Delta_n_det (first R_front < R_crit after s_in), emergent PRP = s_recross +
        Delta_n_det (first R_front > R_crit after s_out and after s_cross) (Law 6).
      - Per-somite boundary coherence somite_R[n] = R_front at the determination stage n - Delta_n_det
        drives the 1D boundary profile chi(s): coherent boundary iff somite_R[n] >= R_crit.
    """
    rng = np.random.default_rng(seed)
    omega_0 = 2.0 * math.pi / T0_min
    dt_min = DT_MIN
    profile = fgf_wnt_frequency_profile(n_ap_cells, cells_per_somite, f_anterior, T0_min, gradient_on)
    omega_x = profile["omega_x_rad_per_min"]
    tau_x = tau_c_min * profile["delay_scale_x"]
    d_steps = np.maximum(1, np.round(tau_x / dt_min).astype(int))
    H = int(d_steps.max()) + 1
    row_idx = (H - 1 - d_steps)[:, None]
    col_idx = np.arange(n_ap_cells)[:, None]
    lat_idx = np.arange(n_lat_cells)[None, :]

    # Mib1-gated coupling normalisation: kappa_max * u_ss(M = 1) = kappa_per_min (wild type)
    u_wt = mib1_ubiquitination_steady_state(1.0, k_ub_per_min, k_endo_per_min)
    kappa_max = kappa_per_min / u_wt if u_wt > 0.0 else 0.0
    M_initial = 0.0 if mib1_restore_somite is not None else mib1_dose
    u = mib1_ubiquitination_steady_state(M_initial, k_ub_per_min, k_endo_per_min)
    kappa_init = kappa_max * u
    stab_wt = solve_collective_frequency_and_stability(T0_min, kappa_per_min, tau_c_min)
    stab_init = solve_collective_frequency_and_stability(T0_min, kappa_init, tau_c_min)
    Omega_wt = stab_wt["Omega_rad_per_min"]
    Omega_init = stab_init["Omega_rad_per_min"]
    T_seg_wt = 2.0 * math.pi / Omega_wt

    # Analytic kinematic steady state (Law 4): dPhi/dx = (Omega/v)(1 - c) with v = cells_per_somite / T_seg
    v_cells_per_min = cells_per_somite / (2.0 * math.pi / Omega_init)
    k_x = Omega_init * (1.0 - profile["c_x"]) / v_cells_per_min  # rad per cell
    Phi_x = np.cumsum(k_x) - k_x[0]
    Phi_x = Phi_x - Phi_x[-1]  # tailbud column has zero lag

    if R_init_target >= 0.99:
        sigma_init = 0.05
    elif R_init_target <= 0.22:
        sigma_init = 2.20
    else:
        sigma_init = math.sqrt(max(0.01, -2.0 * math.log(R_init_target)))
    offsets = rng.normal(0.0, sigma_init, size=(n_ap_cells, n_lat_cells))
    delta_omega = rng.normal(0.0, sigma_omega_rad_per_min, size=(n_ap_cells, n_lat_cells))

    history = np.zeros((H, n_ap_cells, n_lat_cells), dtype=float)
    for kk in range(H):
        t_past = (kk - (H - 1)) * dt_min
        history[kk] = Omega_init * t_past + Phi_x[:, None] + offsets
    theta = history[-1].copy()

    front_cols = slice(0, cells_per_somite)
    smooth_steps = max(1, int(round(0.5 * T_seg_wt / dt_min)))

    def _lateral_R(ph: np.ndarray) -> float:
        return float(np.mean(np.abs(np.mean(np.exp(1j * ph[front_cols]), axis=1))))

    def _global_R(ph: np.ndarray) -> float:
        return float(np.abs(np.mean(np.exp(1j * ph))))

    t_hist: list[float] = [0.0]
    s_hist: list[float] = [0.0]
    R_inst_hist: list[float] = [_lateral_R(theta)]
    R_smooth_hist: list[float] = [R_inst_hist[0]]
    R_global_hist: list[float] = [_global_R(theta)]
    kappa_hist: list[float] = [kappa_init]
    u_hist: list[float] = [u]
    gate_hist: list[float] = [1.0]
    kymo_vec: list[np.ndarray] = [np.mean(np.exp(1j * theta), axis=1)]

    s_curr = 0.0
    t_curr = 0.0
    adv_acc = 0.0
    C_dapt = 0.0
    n_shifts = 0
    s_cross: float | None = None
    s_recross: float | None = None
    somite_formation_s: list[float] = []
    next_somite = 1
    max_steps = int(2.5 * n_somites * T0_min / dt_min) + 10

    for _step in range(max_steps):
        if len(somite_formation_s) >= n_somites:
            break
        # --- DAPT gamma-secretase gate with washout clearance ---
        in_dapt = s_in_somite is not None and s_out_somite is not None and (s_in_somite <= s_curr < s_out_somite)
        if in_dapt:
            C_dapt = dapt_dose_ratio
        elif C_dapt > 0.0:
            C_dapt *= math.exp(-dt_min / tau_wash_min)
            if C_dapt < 1e-6:
                C_dapt = 0.0
        gate = dapt_cleavage_gate(C_dapt)
        # --- Mib1 ubiquitination kinetics (Law 5) ---
        M_now = mib1_dose if (mib1_restore_somite is None or s_curr >= mib1_restore_somite) else 0.0
        u = u + dt_min * (k_ub_per_min * M_now * (1.0 - u) - k_endo_per_min * u)
        u = min(max(u, 0.0), 1.0)
        kappa_now = kappa_max * u * gate

        # --- delayed nearest-neighbour Kuramoto coupling (Law 1), open AP, periodic lateral ---
        theta_delayed = history[row_idx, col_idx, lat_idx]
        sin_sum = np.zeros_like(theta)
        z = np.zeros_like(theta)
        sin_sum[1:] += np.sin(theta_delayed[:-1] - theta[1:])
        z[1:] += 1.0
        sin_sum[:-1] += np.sin(theta_delayed[1:] - theta[:-1])
        z[:-1] += 1.0
        sin_sum += np.sin(np.roll(theta_delayed, 1, axis=1) - theta) + np.sin(np.roll(theta_delayed, -1, axis=1) - theta)
        z += 2.0
        coupling = kappa_now * sin_sum / z
        noise = rng.normal(0.0, math.sqrt(2.0 * D_phi_rad2_per_min * dt_min), size=theta.shape)
        theta = theta + dt_min * (omega_x[:, None] + delta_omega + coupling) + noise
        history[:-1] = history[1:]
        history[-1] = theta

        # --- somite-stage clock at the collective frequency Omega(kappa(t)) (Herrgen 2010) ---
        Omega_now = _collective_frequency_cached(T0_min, kappa_now, tau_c_min)
        ds = dt_min * Omega_now / (2.0 * math.pi)
        s_curr += ds
        t_curr += dt_min
        adv_acc += ds * cells_per_somite
        # --- Lagrangian advection of cells (phase histories + intrinsic frequencies) ---
        while adv_acc >= 1.0:
            adv_acc -= 1.0
            history[:, :-1] = history[:, 1:]
            history[:, -1] = history[:, -2] + rng.normal(0.0, 0.05, size=n_lat_cells)
            delta_omega[:-1] = delta_omega[1:]
            delta_omega[-1] = rng.normal(0.0, sigma_omega_rad_per_min, size=n_lat_cells)
            theta = history[-1].copy()
            n_shifts += 1

        R_inst = _lateral_R(theta)
        R_inst_hist.append(R_inst)
        R_smooth = float(np.mean(R_inst_hist[-smooth_steps:]))
        R_smooth_hist.append(R_smooth)
        R_global_hist.append(_global_R(theta))
        t_hist.append(t_curr)
        s_hist.append(s_curr)
        kappa_hist.append(kappa_now)
        u_hist.append(u)
        gate_hist.append(gate)
        kymo_vec.append(np.mean(np.exp(1j * theta), axis=1))

        perturbed = (s_in_somite is not None and s_curr >= s_in_somite) or (
            mib1_restore_somite is not None or mib1_dose < 1.0
        )
        if perturbed and s_cross is None and R_smooth < R_crit:
            s_cross = s_curr
        recovering = (s_out_somite is not None and s_curr >= s_out_somite) or (
            mib1_restore_somite is not None and s_curr >= mib1_restore_somite
        )
        if recovering and s_cross is not None and s_recross is None and R_smooth > R_crit:
            s_recross = s_curr
        while s_curr >= next_somite and len(somite_formation_s) < n_somites:
            somite_formation_s.append(t_curr)
            next_somite += 1

    t_arr = np.array(t_hist)
    s_arr = np.array(s_hist)
    R_smooth_arr = np.array(R_smooth_hist)

    # Emergent ALD / PRP (Law 6)
    if s_cross is None:
        ald_somite = NO_DEFECT_SENTINEL
    else:
        ald_somite = min(NO_DEFECT_SENTINEL, s_cross + delta_n_det)
    if s_cross is None:
        prp_somite = 0.0
    elif s_recross is None:
        prp_somite = NO_DEFECT_SENTINEL
    else:
        prp_somite = min(NO_DEFECT_SENTINEL, s_recross + delta_n_det)

    # Riedel-Kruse (2007) analytic prediction for comparison
    if s_in_somite is None or (s_out_somite is not None and s_out_somite <= s_in_somite):
        ald_analytic = NO_DEFECT_SENTINEL
        prp_analytic = 0.0
    else:
        ald_analytic = min(NO_DEFECT_SENTINEL, s_in_somite + DELTA_N_ANALYTIC)
        prp_analytic = min(NO_DEFECT_SENTINEL, s_out_somite + DELTA_N_ANALYTIC) if (s_out_somite is not None and s_out_somite < NO_DEFECT_SENTINEL) else NO_DEFECT_SENTINEL

    # Per-somite boundary coherence: R_front at the determination stage (n - Delta_n_det)
    somite_R = np.zeros(n_somites)
    for n in range(n_somites):
        s_det = max(0.0, (n + 1) - delta_n_det)
        idx = int(np.searchsorted(s_arr, s_det))
        somite_R[n] = R_smooth_arr[min(idx, len(R_smooth_arr) - 1)]

    # 1D AP boundary profile chi(s) from emergent per-somite coherence
    nx_ap = 880
    s_axis = np.linspace(0.0, float(n_somites), nx_ap)
    chi_ap = np.zeros_like(s_axis)
    defective_somites: list[int] = []
    for n in range(n_somites):
        mask = (s_axis >= n) & (s_axis <= n + 1)
        loc_phase = math.pi * (s_axis[mask] - n)
        r_eff = float(somite_R[n])
        if r_eff >= R_crit:
            chi_ap[mask] = max(0.75, r_eff) * (np.sin(loc_phase) ** 2) - 0.30
        else:
            defective_somites.append(n + 1)
            salt_pepper = rng.normal(0.0, 0.28 * (1.0 - r_eff), size=int(np.sum(mask)))
            chi_ap[mask] = 0.22 * (np.sin(loc_phase) ** 2) - 0.30 + salt_pepper

    # Kinematic wave morphometrics on the final lateral-mean phase profile (Law 4)
    kymo = np.array(kymo_vec)  # (steps, n_ap) complex lateral-mean phasors
    final_profile = np.unwrap(np.angle(kymo[-1]))
    n_waves_sim = float((final_profile[-1] - final_profile[0]) / (2.0 * math.pi))
    k_sim = np.gradient(final_profile)  # rad per cell
    k_ant = float(np.mean(k_sim[:cells_per_somite]))
    anterior_wavelength_over_S = float(2.0 * math.pi / (k_ant * cells_per_somite)) if k_ant > 1e-6 else math.inf
    total_time = float(t_arr[-1])

    return {
        "t_min": t_arr,
        "somite_time": s_arr,
        "R_history": R_smooth_arr,
        "R_inst_history": np.array(R_inst_hist),
        "R_global_history": np.array(R_global_hist),
        "kappa_history": np.array(kappa_hist),
        "u_history": np.array(u_hist),
        "gate_history": np.array(gate_hist),
        "kymograph": kymo,
        "somite_R": somite_R,
        "somite_formation_t_min": np.array(somite_formation_s),
        "mean_segmentation_period_min": round(total_time / max(1, len(somite_formation_s)), 4),
        "final_phases_rad": np.mod(theta, 2.0 * math.pi),
        "late_mean_R": round(float(np.mean(somite_R[-5:])), 4),
        "min_R": round(float(np.min(somite_R)), 4),
        "s_cross": None if s_cross is None else round(s_cross, 3),
        "s_recross": None if s_recross is None else round(s_recross, 3),
        "ald_somite": round(ald_somite, 2),
        "prp_somite": round(prp_somite, 2),
        "ald_analytic_somite": round(ald_analytic, 2),
        "prp_analytic_somite": round(prp_analytic, 2),
        "defective_somites": defective_somites,
        "s_axis": s_axis,
        "chi_ap": chi_ap,
        "stability": stab_wt,
        "profile": profile,
        "Phi_x_analytic_rad": Phi_x,
        "n_waves_sim": round(n_waves_sim, 4),
        "n_waves_analytic": profile["n_waves_analytic"],
        "anterior_wavelength_over_S_sim": round(anterior_wavelength_over_S, 4),
        "anterior_wavelength_over_S_analytic": profile["anterior_wavelength_over_S_analytic"],
        "kappa_max_per_min": round(kappa_max, 5),
        "u_ss_wt": round(u_wt, 4),
        "n_advection_shifts": n_shifts,
        "total_time_min": round(total_time, 2),
    }


# ============================================================================
# Section 4: 5-Regime DAPT Initiation, Failure (ALD) & Rescue (PRP) Evaluator + Sweeps
#            (Riedel-Kruse et al. 2007 Science Figs. 1-4)
# ============================================================================
REGIME_SEEDS = (2007, 2008, 2009, 2010, 2011)


def evaluate_dapt_regimes(ref: dict[str, Any] | None = None, seeds: tuple[int, ...] = REGIME_SEEDS) -> list[dict[str, Any]]:
    """Evaluate the 5 experimental DAPT regimes of `delta_notch_phase_synchrony_reference.json` with the
    emergent ALD/PRP read-out over `seeds` independent noise realisations (mean +/- sd), alongside the
    Riedel-Kruse (2007) analytic prediction and the observed values:
      1. Wild-type DMSO vehicle control            (obs ALD 31.0 = no defects)
      2. Continuous DAPT from gastrulation s_in = 0 (obs ALD 5.4 +/- 0.8)
      3. Continuous DAPT from s_in = 4              (obs ALD 9.6 +/- 0.9)
      4. Continuous DAPT from s_in = 9              (obs ALD 14.4 +/- 0.9)
      5. DAPT pulse s_in = 0 -> washout s_out = 9   (obs ALD 5.5 +/- 0.8, PRP 14.7 +/- 1.1)
    """
    if ref is None:
        ref = load_reference_data()
    regimes = ref["riedel_kruse_2007_dapt_perturbation_regimes"]["regimes"]
    results: list[dict[str, Any]] = []
    for r in regimes:
        s_in = float(r["s_in_somite"])
        s_out = float(r["s_out_somite"])
        is_wt = r["regime_id"] == "wildtype_vehicle_control"
        alds: list[float] = []
        prps: list[float] = []
        late_R: list[float] = []
        for sd in seeds:
            sim = simulate_delayed_kuramoto_psm_lattice(
                s_in_somite=None if is_wt else s_in,
                s_out_somite=None if is_wt else s_out,
                seed=sd,
            )
            alds.append(sim["ald_somite"])
            prps.append(sim["prp_somite"])
            late_R.append(sim["late_mean_R"])
        pred_ald = float(r["predicted_ald_somite"])
        obs_ald = float(r["observed_ald_somite"])
        results.append(
            {
                "regime_id": r["regime_id"],
                "label": r["label"],
                "s_in_somite": s_in,
                "s_out_somite": s_out,
                "sim_ald_somite": round(float(np.mean(alds)), 3),
                "sim_ald_sd_somite": round(float(np.std(alds)), 3),
                "sim_prp_somite": round(float(np.mean(prps)), 3),
                "sim_prp_sd_somite": round(float(np.std(prps)), 3),
                "pred_ald_somite": pred_ald,
                "obs_ald_somite": obs_ald,
                "obs_ald_sd_somite": float(r["observed_ald_sd_somite"]),
                "pred_prp_somite": float(r["predicted_prp_somite"]),
                "obs_prp_somite": float(r.get("observed_prp_somite", 0.0)),
                "obs_prp_sd_somite": float(r.get("observed_prp_sd_somite", 0.0)),
                "late_mean_R": round(float(np.mean(late_R)), 4),
                "abs_err_ald_analytic": round(abs(pred_ald - obs_ald), 4),
                "abs_err_ald_emergent": round(abs(float(np.mean(alds)) - obs_ald), 4),
                "n_seeds": len(seeds),
            }
        )
    return results


def regime_validation_statistics(regimes: list[dict[str, Any]]) -> dict[str, float]:
    """Pearson r / RMSE of emergent (and analytic) ALD-or-PRP predictions against the observed values across
    the 5 regimes (for the pulse-washout regime the compared quantity is the PRP)."""
    sim_vals, pred_vals, obs_vals = [], [], []
    for r in regimes:
        if r["regime_id"] == "dapt_pulse_washout_rescue":
            sim_vals.append(r["sim_prp_somite"])
            pred_vals.append(r["pred_prp_somite"])
            obs_vals.append(r["obs_prp_somite"])
        else:
            sim_vals.append(r["sim_ald_somite"])
            pred_vals.append(r["pred_ald_somite"])
            obs_vals.append(r["obs_ald_somite"])
    sim_v, pred_v, obs_v = np.array(sim_vals), np.array(pred_vals), np.array(obs_vals)
    r_sim, p_sim = stats.pearsonr(sim_v, obs_v)
    r_pred, p_pred = stats.pearsonr(pred_v, obs_v)
    # exclude the no-defect WT sentinel (31.0 by construction) from the RMSE of defect positions
    mask = np.array([r["regime_id"] != "wildtype_vehicle_control" for r in regimes])
    return {
        "pearson_r_emergent": round(float(r_sim), 4),
        "pearson_p_emergent": float(p_sim),
        "rmse_emergent_somites": round(float(np.sqrt(np.mean((sim_v[mask] - obs_v[mask]) ** 2))), 4),
        "pearson_r_analytic": round(float(r_pred), 4),
        "pearson_p_analytic": float(p_pred),
        "rmse_analytic_somites": round(float(np.sqrt(np.mean((pred_v[mask] - obs_v[mask]) ** 2))), 4),
        "max_abs_err_emergent_somites": round(float(np.max(np.abs(sim_v[mask] - obs_v[mask]))), 4),
    }


def sweep_frequency_profile(f_values: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), seed: int = 2007) -> list[dict[str, float]]:
    """Parameter sweep of the FGF8/Wnt profile steepness f_A -> waves per PSM (simulated vs analytic, Law 4)."""
    out = []
    for fA in f_values:
        sim = simulate_delayed_kuramoto_psm_lattice(f_anterior=fA, gradient_on=fA < 1.0, seed=seed)
        out.append(
            {
                "f_anterior": fA,
                "n_waves_sim": sim["n_waves_sim"],
                "n_waves_analytic": sim["n_waves_analytic"],
                "anterior_wavelength_over_S_sim": sim["anterior_wavelength_over_S_sim"],
                "late_mean_R": sim["late_mean_R"],
            }
        )
    return out


def sweep_mib1_dose(m_values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5), seed: int = 2007) -> list[dict[str, float]]:
    """Parameter sweep of Mib1 E3-ligase activity M -> u_ss, kappa_eff, late front coherence and emergent ALD (Law 5)."""
    out = []
    for M in m_values:
        eff = effective_coupling_for_mib1_dose(M)
        sim = simulate_delayed_kuramoto_psm_lattice(mib1_dose=M, seed=seed)
        out.append(
            {
                "mib1_dose": M,
                "u_ss": eff["u_ss"],
                "kappa_per_min": eff["kappa_per_min"],
                "kappa_eff_per_min": eff["kappa_eff_per_min"],
                "late_mean_R": sim["late_mean_R"],
                "ald_somite": sim["ald_somite"],
            }
        )
    return out


# ============================================================================
# Section 5: Biological Specimen Computer-Vision Segmentation
#            (`datasets/delta_notch_phase_synchrony/zebrafish_psm_synchrony_specimen.jpg`)
# ============================================================================
def segment_zebrafish_psm_specimen(
    img_path: Path = SPECIMEN_JPG,
) -> dict[str, Any]:
    """Segment the dorsal vertebrate embryo SEM specimen (`zebrafish_psm_synchrony_specimen.jpg`,
    CC BY 4.0) to extract:
      - Upper and lower bilateral paraxial somite rows flanking the central neural tube
      - Individual somite dome centroids and inter-somitic clefts along the AP axis
      - Empirical somite wavelength S_obs (px), spacing CV, and bilateral symmetry ratio
      - Goodness-of-fit (Pearson r) between the empirical AP somite intensity profile
        and the phase-synchronized Delta-Notch segmentation clock wave.
    """
    im = np.asarray(Image.open(img_path).convert("RGB"), dtype=float)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B

    x0, x1 = 130, 550
    upper_y0, upper_y1 = 208, 252
    lower_y0, lower_y1 = 312, 356

    def _extract_row(y0: int, y1: int) -> dict[str, Any]:
        strip_L = lum[y0:y1, x0:x1].mean(axis=0)
        sig = ndi.gaussian_filter1d(strip_L, sigma=3.5) - ndi.gaussian_filter1d(
            strip_L, sigma=22.0
        )
        peaks, _ = signal.find_peaks(sig, distance=32, prominence=1.8)
        clefts, _ = signal.find_peaks(-sig, distance=32, prominence=1.8)
        dx = np.diff(peaks).astype(float)
        mean_s = float(np.mean(dx))
        sd_s = float(np.std(dx))
        return {
            "signal": sig,
            "peaks_x": (peaks + x0).tolist(),
            "clefts_x": (clefts + x0).tolist(),
            "n_somites": int(len(peaks)),
            "spacings_px": dx.tolist(),
            "mean_spacing_px": round(mean_s, 4),
            "sd_spacing_px": round(sd_s, 4),
            "cv_spacing": round(sd_s / mean_s, 4),
        }

    upper = _extract_row(upper_y0, upper_y1)
    lower = _extract_row(lower_y0, lower_y1)
    all_spacings = np.array(upper["spacings_px"] + lower["spacings_px"], dtype=float)
    bilateral_mean_px = float(np.mean(all_spacings))
    bilateral_sd_px = float(np.std(all_spacings))
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
    """Return quantitative morphometric measurements for `morphoeval/measure_outputs.py`."""
    verify_dataset_integrity()
    ref = load_reference_data()
    stab_wt = solve_collective_frequency_and_stability(T0_MIN, KAPPA_WT_PER_MIN, TAU_C_MIN)
    stab_anti = solve_collective_frequency_and_stability(T0_MIN, KAPPA_WT_PER_MIN, 11.0)
    sim_wt = simulate_delayed_kuramoto_psm_lattice(seed=2007)
    sim_dapt_s0 = simulate_delayed_kuramoto_psm_lattice(s_in_somite=0.0, s_out_somite=22.0, seed=2007)
    sim_rescue = simulate_delayed_kuramoto_psm_lattice(s_in_somite=0.0, s_out_somite=9.0, seed=2007)
    sim_uniform = simulate_delayed_kuramoto_psm_lattice(gradient_on=False, seed=2007)
    sim_mib_null = simulate_delayed_kuramoto_psm_lattice(mib1_dose=0.0, seed=2007)
    sim_mib_restore = simulate_delayed_kuramoto_psm_lattice(mib1_dose=1.0, mib1_restore_somite=9.0, seed=2007)
    spec = segment_zebrafish_psm_specimen()
    regimes = evaluate_dapt_regimes(ref)
    val = regime_validation_statistics(regimes)
    rescue_reg = next(r for r in regimes if r["regime_id"] == "dapt_pulse_washout_rescue")
    s0_reg = next(r for r in regimes if r["regime_id"] == "dapt_early_gastrula_s0")
    mib_eff_null = effective_coupling_for_mib1_dose(0.0)
    mib_eff_half = effective_coupling_for_mib1_dose(0.5)

    return [
        {
            "name": "delayed Kuramoto collective synchronization period T_sync vs autonomous T_0",
            "value": round(stab_wt["T_sync_min"], 2),
            "unit": "min",
            "method": f"transcendental root of Omega = omega_0 - kappa*sin(Omega*tau_c) for T_0=25.0 min, kappa=0.070 min^-1, tau_c=20.5 min (Omega={stab_wt['Omega_rad_per_min']:.4f} rad/min, Omega/omega_0={stab_wt['frequency_ratio_Omega_over_omega0']:.4f}); emergent lattice segmentation period {sim_wt['mean_segmentation_period_min']:.2f} min",
            "reference_value": 22.18,
            "reference_source": "Riedel-Kruse et al. (2007) Science 317:1911-1915 SOM & Morelli et al. (2009) HFSP J. 3:55-66 Eqs. 2-3 (collective period 22.18 min vs uncoupled 25.0 min)",
            "note": f"DAPT Notch blockade (kappa=0) shifts period from {stab_wt['T_sync_min']:.2f} min to T_0 = 25.00 min (+12.7% slowing, Herrgen et al. 2010); continuous-DAPT lattice mean period {sim_dapt_s0['mean_segmentation_period_min']:.2f} min",
            "derived_from": "script",
        },
        {
            "name": "effective in-phase linear stability stiffness kappa_eff = kappa * cos(Omega * tau_c)",
            "value": round(stab_wt["kappa_eff_per_min"], 4),
            "unit": "min^-1",
            "method": f"linearized delayed Kuramoto stiffness kappa_eff = kappa*cos(Omega*tau_c) at tau_c=20.5 min (Omega*tau_c={stab_wt['phase_delay_rad']:.3f} rad) vs repulsive anti-phase delay tau_c=11.0 min (kappa_eff={stab_anti['kappa_eff_per_min']:.4f} min^-1)",
            "reference_value": 0.0623,
            "reference_source": "Morelli et al. (2009) HFSP J. 3:55-66 Eqs. 4-5 & Fig. 2 (in-phase synchronization stable iff kappa_eff = kappa*cos(Omega*tau_c) > 0)",
            "note": f"in_phase_stable = {stab_wt['in_phase_stable']} at tau_c=20.5 min vs {stab_anti['in_phase_stable']} at repulsive delay tau_c=11.0 min",
            "derived_from": "script",
        },
        {
            "name": "emergent DAPT pulse-chase Anterior Limit of Defects (ALD) at s_in = 0",
            "value": round(s0_reg["sim_ald_somite"], 2),
            "unit": "somites",
            "method": f"first crossing of the determination-front lateral Kuramoto R below R_crit=0.50 + Delta_n_det=2.5 in the advected 24x12 delayed Kuramoto PSM lattice, mean over {s0_reg['n_seeds']} seeds (sd {s0_reg['sim_ald_sd_somite']:.2f}); single-seed s_cross={sim_dapt_s0['s_cross']}",
            "reference_value": 5.4,
            "reference_source": "Riedel-Kruse et al. (2007) Science 317:1911-1915 Fig. 2 (continuous DAPT from gastrulation: ALD 5.4 +/- 0.8 somites)",
            "note": f"Riedel-Kruse analytic ALD = s_in + 5.5 = {sim_dapt_s0['ald_analytic_somite']:.1f}; continuous-DAPT late front R = {sim_dapt_s0['late_mean_R']:.3f} vs WT {sim_wt['late_mean_R']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "emergent DAPT washout Posterior Rescue Point (PRP) for pulse [s_in = 0, s_out = 9]",
            "value": round(rescue_reg["sim_prp_somite"], 2),
            "unit": "somites",
            "method": f"first recovery of the determination-front lateral Kuramoto R above R_crit after washout + Delta_n_det, mean over {rescue_reg['n_seeds']} seeds (sd {rescue_reg['sim_prp_sd_somite']:.2f}); DAPT clearance tau_wash={TAU_WASH_MIN:.0f} min, Mib1 u_ss={sim_rescue['u_ss_wt']:.2f}; single-seed ALD={sim_rescue['ald_somite']:.2f}, PRP={sim_rescue['prp_somite']:.2f}, min R={sim_rescue['min_R']:.3f} -> rescued R={sim_rescue['late_mean_R']:.3f}",
            "reference_value": 14.7,
            "reference_source": "Riedel-Kruse et al. (2007) Science 317:1911-1915 Fig. 4 (DAPT pulse 0 -> 9 somites: PRP 14.7 +/- 1.1 somites)",
            "note": f"emergent defect span PRP - ALD = {sim_rescue['prp_somite'] - sim_rescue['ald_somite']:.2f} somites = s_recross - s_cross (Delta_n_det cancels, Law 6); analytic PRP = {sim_rescue['prp_analytic_somite']:.1f}",
            "derived_from": "script",
        },
        {
            "name": "Riedel-Kruse (2007) 5-regime emergent ALD/PRP vs observed Pearson r",
            "value": val["pearson_r_emergent"],
            "unit": "Pearson r",
            "method": f"correlation between emergent lattice ALD (WT, s_in=0, 4, 9) / PRP (pulse 0->9) and the observed values across 5 regimes, {REGIME_SEEDS.__len__()} seeds each; RMSE over the 4 perturbed regimes = {val['rmse_emergent_somites']:.3f} somites (max |err| {val['max_abs_err_emergent_somites']:.2f})",
            "reference_value": 1.0,
            "reference_source": "Riedel-Kruse, Müller & Oates (2007) Science 317:1911-1915 Figs. 2B, 3D & 4 (unit-slope ALD(s_in) = s_in + Delta_n_ALD; PRP after washout)",
            "note": f"p = {val['pearson_p_emergent']:.2e}; analytic Riedel-Kruse prediction r = {val['pearson_r_analytic']:.4f}, RMSE = {val['rmse_analytic_somites']:.3f} somites",
            "derived_from": "script",
        },
        {
            "name": "kinematic phase waves across the PSM N_w (simulated vs Morelli 2009 analytic)",
            "value": sim_wt["n_waves_sim"],
            "unit": "waves per PSM",
            "method": f"unwrapped lateral-mean phase drop across the 6-somite-length PSM in the advected lattice with the FGF8/Wnt profile f_A={F_ANTERIOR} (analytic N_w = (1/S) Int (1 - c) dx = {sim_wt['n_waves_analytic']:.3f}); uniform-profile control N_w = {sim_uniform['n_waves_sim']:.3f}",
            "reference_value": sim_wt["n_waves_analytic"],
            "reference_source": "Morelli et al. (2009) HFSP J. 3:55-66 kinematic wave theory k = (Omega - omega(x))/v (Law 4); Soroldoni et al. (2014) Science 345:222-225 report ~1-2 waves across the zebrafish PSM",
            "note": f"anterior wavelength / somite length: simulated {sim_wt['anterior_wavelength_over_S_sim']:.2f} vs analytic 1/(1 - f_A) = {sim_wt['anterior_wavelength_over_S_analytic']:.2f}; waves slow toward the front (speed Omega/k(x))",
            "derived_from": "script",
        },
        {
            "name": "Mib1 ubiquitination gate: late front coherence R in mib1-null (M = 0) vs wild type",
            "value": sim_mib_null["late_mean_R"],
            "unit": "Kuramoto R",
            "method": f"kappa(t) = kappa_max * u(t) * g(t) with u_ss(M) = k_ub M/(k_ub M + k_endo) (k_ub={K_UB_PER_MIN}, k_endo={K_ENDO_PER_MIN} min^-1): M=0 gives u_ss={mib_eff_null['u_ss']:.2f}, kappa_eff={mib_eff_null['kappa_eff_per_min']:.4f}; M=0.5 gives u_ss={mib_eff_half['u_ss']:.2f}, kappa_eff={mib_eff_half['kappa_eff_per_min']:.4f}; WT late R={sim_wt['late_mean_R']:.3f}",
            "reference_value": 0.29,
            "reference_source": "Itoh et al. (2003) Dev Cell 4:67-82 (mib^ta52b abolishes Delta-Notch signalling); Riedel-Kruse et al. (2007) (mib/aei/des/bea mutants desynchronise like early DAPT); 12-cell lateral random-phase floor 1/sqrt(12) = 0.29",
            "note": f"mib1-null emergent ALD = {sim_mib_null['ald_somite']:.2f} somites; Mib1 restoration at somite 9 (tau_u = {1.0/(K_UB_PER_MIN + K_ENDO_PER_MIN):.0f} min) rescues to R={sim_mib_restore['late_mean_R']:.3f} at PRP={sim_mib_restore['prp_somite']:.2f}",
            "derived_from": "script",
        },
        {
            "name": "vertebrate embryo SEM specimen bilateral somite spacing S_obs",
            "value": round(spec["bilateral_mean_spacing_px"], 4),
            "unit": "px",
            "method": f"bilateral dorsal somite row DoG peak segmentation on zebrafish_psm_synchrony_specimen.jpg (upper={spec['upper_row']['n_somites']} somites, lower={spec['lower_row']['n_somites']} somites, CV={spec['bilateral_cv_spacing']:.4f})",
            "reference_value": 48.47,
            "reference_source": "Vendored CC BY 4.0 vertebrate embryo SEM specimen (datasets/delta_notch_phase_synchrony/zebrafish_psm_synchrony_specimen.jpg)",
            "note": f"bilateral symmetry ratio = {spec['bilateral_symmetry_ratio']:.4f}, std = {spec['bilateral_sd_spacing_px']:.2f} px, phase correlation r = {spec['profile_phase_correlation_r']:.4f}",
            "derived_from": "output",
        },
    ]


# ============================================================================
# Section 7: Publication Figure & Animated GIF Rendering
# ============================================================================
def _style_dark() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b0f19",
            "axes.facecolor": "#131b2e",
            "axes.edgecolor": "#3b4c6b",
            "axes.labelcolor": "#e2e8f0",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "font.size": 9.5,
        }
    )


def render_comparison_figure(out_png: Path = OUTPUT_PNG) -> Path:
    """Render the 9-panel dark-theme publication synthesis figure
    `docs/media/delta_notch_phase_synchrony_comparison.png`.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ref = load_reference_data()
    sim_wt = simulate_delayed_kuramoto_psm_lattice(seed=2007)
    sim_init = simulate_delayed_kuramoto_psm_lattice(R_init_target=0.18, seed=2007)
    sim_dapt = simulate_delayed_kuramoto_psm_lattice(s_in_somite=0.0, s_out_somite=22.0, seed=2007)
    sim_rescue = simulate_delayed_kuramoto_psm_lattice(s_in_somite=0.0, s_out_somite=9.0, seed=2007)
    sim_mib_restore = simulate_delayed_kuramoto_psm_lattice(mib1_dose=1.0, mib1_restore_somite=9.0, seed=2007)
    spec = segment_zebrafish_psm_specimen()
    regimes = evaluate_dapt_regimes(ref)
    val = regime_validation_statistics(regimes)
    f_sweep = sweep_frequency_profile()
    m_sweep = sweep_mib1_dose()

    _style_dark()
    fig, axes = plt.subplots(3, 3, figsize=(17.4, 14.6), dpi=140)
    fig.suptitle(
        "Riedel-Kruse et al. (2007) Science & Morelli et al. (2009) HFSP J.: Delayed Kuramoto Delta–Notch Synchrony\n"
        "FGF8/Wnt Kinematic Phase Waves, Mib1-Gated Coupling $\\kappa(t)$ & Emergent DAPT ALD / PRP (Pass 3)",
        fontsize=12.0,
        fontweight="bold",
        color="#38bdf8",
        y=0.992,
    )

    # Panel A: Initiation, Failure (DAPT), and Rescue (Washout) front order parameter R(s)
    ax = axes[0, 0]
    s_grid = np.arange(1, len(sim_wt["somite_R"]) + 1)
    ax.plot(sim_wt["somite_time"], sim_wt["R_history"], color="#38bdf8", lw=1.9, label=f"Wild-Type Notch+ ($\\bar{{R}}={sim_wt['late_mean_R']:.3f}$)")
    ax.plot(sim_init["somite_time"], sim_init["R_history"], color="#a78bfa", lw=1.6, ls="--", label=f"Gastrula initiation ($R_0=0.18 \\rightarrow {sim_init['late_mean_R']:.2f}$)")
    ax.plot(sim_dapt["somite_time"], sim_dapt["R_history"], color="#f43f5e", lw=1.9, label=f"Continuous DAPT $s_{{\\mathrm{{in}}}}=0$ (emergent ALD $={sim_dapt['ald_somite']:.1f}$)")
    ax.plot(sim_rescue["somite_time"], sim_rescue["R_history"], color="#34d399", lw=2.0, label=f"DAPT pulse $[0,9]$ + washout (PRP $={sim_rescue['prp_somite']:.1f}$)")
    ax.plot(sim_mib_restore["somite_time"], sim_mib_restore["R_history"], color="#fb923c", lw=1.5, ls="-.", label=f"$mib1^{{-}}$ + Mib1 restored at $s=9$ (PRP $={sim_mib_restore['prp_somite']:.1f}$)")
    ax.axhline(R_CRIT, color="#facc15", lw=1.4, ls=":", label="$R_{\\mathrm{crit}}=0.50$")
    ax.axvspan(sim_rescue["ald_somite"], sim_rescue["prp_somite"], color="#f43f5e", alpha=0.13, label="Emergent defective window $[\\mathrm{ALD}, \\mathrm{PRP}]$")
    ax.axvline(sim_rescue["ald_analytic_somite"], color="#f8fafc", lw=0.9, ls="--", alpha=0.6)
    ax.axvline(sim_rescue["prp_analytic_somite"], color="#f8fafc", lw=0.9, ls="--", alpha=0.6, label="Riedel-Kruse analytic $s+5.5$")
    ax.set_title("A. Front Synchrony $R(s)$: Initiation, DAPT Failure & Rescue (emergent ALD/PRP)", fontweight="bold", fontsize=9.6)
    ax.set_xlabel("Somite Stage $s$ (clock-counted, Herrgen 2010 DAPT slowing)")
    ax.set_ylabel("Lateral Kuramoto $R_{\\mathrm{front}}(s)$ at determination front")
    ax.set_xlim(0, 22)
    ax.set_ylim(0.0, 1.50)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper center", ncol=2, fontsize=6.5, framealpha=0.88, facecolor="#0f172a")

    # Panel B: Morelli et al. (2009) 2D Delayed Coupling Stability Morphospace
    ax = axes[0, 1]
    tau_grid = np.linspace(0.5, 32.0, 80)
    kappa_grid = np.linspace(0.01, 0.14, 70)
    TAU, KAP = np.meshgrid(tau_grid, kappa_grid)
    KEFF = np.zeros_like(TAU)
    for ii in range(TAU.shape[0]):
        for jj in range(TAU.shape[1]):
            st = solve_collective_frequency_and_stability(T0_MIN, float(KAP[ii, jj]), float(TAU[ii, jj]))
            KEFF[ii, jj] = st["kappa_eff_per_min"]
    cs = ax.contourf(TAU, KAP, KEFF, levels=18, cmap="coolwarm", vmin=-0.12, vmax=0.12)
    ax.contour(TAU, KAP, KEFF, levels=[0.0], colors=["#facc15"], linewidths=2.0, linestyles="--")
    ax.scatter([20.5], [0.070], s=90, c="#38bdf8", edgecolors="white", zorder=5, label="Zebrafish WT ($\\tau_c=20.5$ min, $\\kappa_{\\mathrm{eff}}>0$)")
    ax.scatter([11.0], [0.070], s=75, c="#f43f5e", marker="X", edgecolors="white", zorder=5, label="Anti-phase unstable ($\\tau_c=11.0$ min)")
    ax.scatter([20.5 / F_ANTERIOR], [0.070], s=60, c="#fb923c", marker="^", edgecolors="white", zorder=5, label=f"Front cell with un-scaled delay ($\\omega\\tau_c$ at $f_A={F_ANTERIOR}$)")
    ax.set_title("B. Delayed Coupling Stability $\\kappa_{\\mathrm{eff}} = \\kappa\\cos(\\Omega\\tau_c)$", fontweight="bold")
    ax.set_xlabel("Intercellular Delta–Notch Delay $\\tau_c$ (min)")
    ax.set_ylabel("Coupling Strength $\\kappa$ (min$^{-1}$)")
    ax.legend(loc="upper left", fontsize=7.0, framealpha=0.88, facecolor="#0f172a")
    fig.colorbar(cs, ax=ax, fraction=0.045, pad=0.03, label="$\\kappa_{\\mathrm{eff}}$ (min$^{-1}$)")

    # Panel C: Collective Period Branches
    ax = axes[0, 2]
    tau_scan = np.linspace(0.5, 32.0, 140)
    t_sync_scan, keff_scan = [], []
    for tc in tau_scan:
        st = solve_collective_frequency_and_stability(T0_MIN, KAPPA_WT_PER_MIN, float(tc))
        t_sync_scan.append(st["T_sync_min"])
        keff_scan.append(st["kappa_eff_per_min"])
    t_sync_arr = np.array(t_sync_scan)
    stable_mask = np.array(keff_scan) > 0.0
    ax.plot(tau_scan, t_sync_arr, color="#64748b", lw=1.4, ls="--", label="Unstable anti-phase branch ($\\kappa_{\\mathrm{eff}} < 0$)")
    ax.plot(np.where(stable_mask, tau_scan, np.nan), np.where(stable_mask, t_sync_arr, np.nan), color="#38bdf8", lw=2.5, label="Stable in-phase branch ($\\kappa_{\\mathrm{eff}} > 0$)")
    ax.axhline(25.0, color="#f43f5e", lw=1.6, ls=":", label="Uncoupled / DAPT period $T_0 = 25.0$ min")
    ax.scatter([20.5], [sim_wt["stability"]["T_sync_min"]], s=85, c="#34d399", edgecolors="white", zorder=5, label=f"WT operating point ($T_{{\\mathrm{{sync}}}}={sim_wt['stability']['T_sync_min']:.2f}$ min)")
    ax.axhline(sim_wt["mean_segmentation_period_min"], color="#34d399", lw=1.0, ls="-.", alpha=0.8, label=f"Lattice WT segmentation period {sim_wt['mean_segmentation_period_min']:.2f} min")
    ax.axhline(sim_dapt["mean_segmentation_period_min"], color="#fb923c", lw=1.0, ls="-.", alpha=0.8, label=f"Lattice continuous-DAPT period {sim_dapt['mean_segmentation_period_min']:.2f} min")
    ax.set_title("C. Collective Period $T_{\\mathrm{sync}}(\\tau_c) = 2\\pi / \\Omega$ & DAPT Slowing", fontweight="bold")
    ax.set_xlabel("Intercellular Delay $\\tau_c$ (min)")
    ax.set_ylabel("Collective Clock Period $T_{\\mathrm{sync}}$ (min)")
    ax.legend(loc="upper right", fontsize=6.8, framealpha=0.88, facecolor="#0f172a")

    # Panel D: 5-Regime DAPT ALD & PRP Validation (analytic vs emergent vs observed)
    ax = axes[1, 0]
    labels_short = ["WT DMSO\nControl", "DAPT at\n$s_{in}=0$", "DAPT at\n$s_{in}=4$", "DAPT at\n$s_{in}=9$", "Pulse $[0,9]$\nPRP"]
    x_pos = np.arange(len(regimes))
    pred_vals, sim_vals, sim_sds, obs_vals, obs_sds = [], [], [], [], []
    for r in regimes:
        if r["regime_id"] == "dapt_pulse_washout_rescue":
            pred_vals.append(r["pred_prp_somite"]); sim_vals.append(r["sim_prp_somite"]); sim_sds.append(r["sim_prp_sd_somite"])
            obs_vals.append(r["obs_prp_somite"]); obs_sds.append(r["obs_prp_sd_somite"])
        else:
            pred_vals.append(r["pred_ald_somite"]); sim_vals.append(r["sim_ald_somite"]); sim_sds.append(r["sim_ald_sd_somite"])
            obs_vals.append(r["obs_ald_somite"]); obs_sds.append(r["obs_ald_sd_somite"])
    ax.bar(x_pos - 0.27, pred_vals, width=0.26, color="#94a3b8", label="Riedel-Kruse analytic $s + 5.5$")
    ax.bar(x_pos, sim_vals, width=0.26, yerr=sim_sds, capsize=3, color="#38bdf8", ecolor="#e2e8f0", label=f"Emergent lattice ({len(REGIME_SEEDS)} seeds)")
    ax.bar(x_pos + 0.27, obs_vals, width=0.26, yerr=obs_sds, capsize=3, color="#34d399", ecolor="#e2e8f0", label="Riedel-Kruse (2007) observed")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short, fontsize=7.8)
    ax.set_title(f"D. ALD/PRP Validation (emergent $r={val['pearson_r_emergent']:.3f}$, RMSE $={val['rmse_emergent_somites']:.2f}$ som)", fontweight="bold", fontsize=9.6)
    ax.set_ylabel("Somite Boundary Index (ALD or PRP)")
    ax.set_ylim(0, 36)
    ax.legend(loc="upper right", fontsize=7.4, framealpha=0.88, facecolor="#0f172a")

    # Panel E: Specimen Segmentation
    ax = axes[1, 1]
    spec_im = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"))
    ax.imshow(spec_im)
    x0, x1 = spec["ap_roi_x_px"]
    ax.axhline(230, xmin=x0 / 736.0, xmax=x1 / 736.0, color="#38bdf8", lw=1.2, ls=":")
    ax.axhline(334, xmin=x0 / 736.0, xmax=x1 / 736.0, color="#34d399", lw=1.2, ls=":")
    ax.scatter(spec["upper_row"]["peaks_x"], [230] * len(spec["upper_row"]["peaks_x"]), c="#38bdf8", s=42, edgecolors="black", label=f"Upper row ({spec['upper_row']['n_somites']} somites, $\\bar{{S}}={spec['upper_row']['mean_spacing_px']:.1f}$ px)")
    ax.scatter(spec["lower_row"]["peaks_x"], [334] * len(spec["lower_row"]["peaks_x"]), c="#34d399", s=42, edgecolors="black", label=f"Lower row ({spec['lower_row']['n_somites']} somites, $\\bar{{S}}={spec['lower_row']['mean_spacing_px']:.1f}$ px)")
    ax.set_title("E. Vertebrate Embryo SEM Somite Row Segmentation (CC BY 4.0)", fontweight="bold")
    ax.set_xlabel("Anterior-Posterior Pixel Coordinate $x$ (px)")
    ax.set_ylabel("Dorsoventral Coordinate $y$ (px)")
    ax.legend(loc="lower left", fontsize=7.6, framealpha=0.88, facecolor="#0f172a")

    # Panel F: Empirical Somite Profile vs Synchronized Clock Wave
    ax = axes[1, 2]
    x_roi = np.arange(x0, x1, dtype=float)
    emp_sig = np.array(spec["lower_row"]["signal"], dtype=float)
    emp_norm = emp_sig / (np.std(emp_sig) + 1e-12)
    k_emp = 2.0 * math.pi / spec["lower_row"]["mean_spacing_px"]
    x_rel = x_roi - x0
    phi_fit = math.atan2(float(np.dot(emp_norm, np.sin(k_emp * x_rel))), float(np.dot(emp_norm, np.cos(k_emp * x_rel))))
    mod_wave = np.tanh(1.8 * np.cos(k_emp * x_rel - phi_fit))
    mod_norm = mod_wave / (np.std(mod_wave) + 1e-12)
    ax.plot(x_roi, emp_norm, color="#34d399", lw=1.8, label="Empirical SEM somite band-pass profile")
    ax.plot(x_roi, mod_norm, color="#38bdf8", lw=1.6, ls="--", label=f"Synchronized Notch wave ($r={spec['profile_phase_correlation_r']:.3f}$)")
    ax.set_title(f"F. Specimen vs Synchronized Wave ($\\bar{{S}}={spec['bilateral_mean_spacing_px']:.2f}$ px, CV={spec['bilateral_cv_spacing']:.3f})", fontweight="bold", fontsize=9.6)
    ax.set_xlabel("Anterior-Posterior Coordinate $x$ (px)")
    ax.set_ylabel("Normalized Somite Elevation $z$-score")
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.88, facecolor="#0f172a")

    # Panel G: Kinematic phase-wave kymograph of the DAPT pulse-washout regime
    ax = axes[2, 0]
    kymo = sim_rescue["kymograph"]  # (steps, n_ap)
    kymo_img = np.real(kymo).T  # |R_lat| * cos(phase): waves fade when laterally desynchronised
    s_t = sim_rescue["somite_time"]
    L_psm = sim_rescue["profile"]["L_psm_somites"]
    im_k = ax.imshow(kymo_img, aspect="auto", origin="lower", cmap="twilight", vmin=-1, vmax=1,
                     extent=[s_t[0], s_t[-1], 0.0, L_psm])
    ax.axvspan(0.0, 9.0, color="#f43f5e", alpha=0.10)
    ax.axvline(9.0, color="#34d399", lw=1.4, ls="--")
    ax.axvline(sim_rescue["s_cross"] if sim_rescue["s_cross"] is not None else 0.0, color="#facc15", lw=1.0, ls=":")
    ax.axvline(sim_rescue["s_recross"] if sim_rescue["s_recross"] is not None else 0.0, color="#facc15", lw=1.0, ls=":")
    ax.set_title(f"G. Kinematic Phase Waves (DAPT pulse $[0,9]$): $N_w={sim_wt['n_waves_sim']:.2f}$ (analytic {sim_wt['n_waves_analytic']:.2f})", fontweight="bold", fontsize=9.6)
    ax.set_xlabel("Somite Stage $s$  (red: DAPT on; green dash: washout; dotted: $s_{\\mathrm{cross}}, s_{\\mathrm{recross}}$)")
    ax.set_ylabel("PSM position (somite lengths from front)")
    fig.colorbar(im_k, ax=ax, fraction=0.045, pad=0.03, label="$R_{\\mathrm{lat}}(x)\\cos\\bar{\\theta}(x)$")
    # Zoomed inset: three rescued segmentation periods, where the posterior -> anterior tilt (wave travel) is visible
    zoom_mask = (s_t >= 18.0) & (s_t <= 21.0)
    ins_g = ax.inset_axes([0.60, 0.60, 0.38, 0.36])
    ins_g.imshow(kymo_img[:, zoom_mask], aspect="auto", origin="lower", cmap="twilight", vmin=-1, vmax=1,
                 extent=[18.0, 21.0, 0.0, L_psm])
    ins_g.set_xticks([18, 19, 20, 21])
    ins_g.set_yticks([0, 3, 6])
    ins_g.tick_params(labelsize=6.3, colors="#f8fafc")
    for sp in ins_g.spines.values():
        sp.set_color("#facc15")
    ins_g.set_title("zoom $s\\in[18,21]$: crests travel tail → front", fontsize=6.6, color="#f8fafc")

    # Panel H: FGF8/Wnt frequency profile, analytic vs simulated phase lag, f_A sweep
    ax = axes[2, 1]
    xi = sim_wt["profile"]["xi_somites"]
    x_front = L_psm - xi  # distance from front
    ax.plot(x_front, sim_wt["profile"]["c_x"], color="#fb923c", lw=2.0, label=f"FGF8/Wnt profile $\\omega(x)/\\omega_0 = e^{{-\\xi/\\lambda_F}}$ ($f_A={F_ANTERIOR}$)")
    phi_sim = np.unwrap(np.angle(sim_wt["kymograph"][-1]))
    phi_sim = (phi_sim - phi_sim[-1]) / (2.0 * math.pi)
    phi_an = sim_wt["Phi_x_analytic_rad"] / (2.0 * math.pi)
    ax.plot(x_front, -phi_an, color="#94a3b8", lw=1.6, ls="--", label="Analytic phase lag $\\Phi(x)/2\\pi$ (Law 4)")
    ax.plot(x_front, -phi_sim, color="#38bdf8", lw=2.0, label=f"Simulated phase lag ($N_w={sim_wt['n_waves_sim']:.2f}$)")
    ax.set_xlabel("PSM position $x$ (somite lengths from determination front)")
    ax.set_ylabel("$\\omega/\\omega_0$  and  phase lag / $2\\pi$ (waves)")
    ax.set_title("H. FGF8/Wnt Frequency Profile → Kinematic Phase Lag; $f_A$ Sweep (inset)", fontweight="bold", fontsize=9.6)
    ax.legend(loc="upper right", fontsize=7.0, framealpha=0.88, facecolor="#0f172a")
    ins = ax.inset_axes([0.58, 0.12, 0.40, 0.42])
    ins.set_facecolor("#0f172a")
    fa = [d["f_anterior"] for d in f_sweep]
    ins.plot(fa, [d["n_waves_analytic"] for d in f_sweep], "--", color="#94a3b8", lw=1.2, label="analytic")
    ins.plot(fa, [d["n_waves_sim"] for d in f_sweep], "o-", color="#38bdf8", lw=1.4, ms=3.5, label="lattice")
    ins.set_xlabel("$f_A$", fontsize=7, color="#e2e8f0")
    ins.set_ylabel("$N_w$", fontsize=7, color="#e2e8f0")
    ins.tick_params(labelsize=6.5, colors="#cbd5e1")
    ins.legend(fontsize=6.0, framealpha=0.7, facecolor="#0f172a", loc="upper right")

    # Panel I: Mib1 ubiquitination kinetics and dose sweep
    ax = axes[2, 2]
    ax.plot(sim_rescue["somite_time"], sim_rescue["kappa_history"] / sim_rescue["kappa_max_per_min"], color="#34d399", lw=1.8, label="DAPT pulse: $\\kappa(t)/\\kappa_{\\max} = u\\,g$ (washout $\\tau_{\\mathrm{wash}}$)")
    ax.plot(sim_mib_restore["somite_time"], sim_mib_restore["kappa_history"] / sim_mib_restore["kappa_max_per_min"], color="#fb923c", lw=1.8, ls="-.", label=f"$mib1^-$ → Mib1 on at $s=9$: $u(t)$ rise ($\\tau_u={1.0/(K_UB_PER_MIN+K_ENDO_PER_MIN):.0f}$ min)")
    ax.plot(sim_rescue["somite_time"], sim_rescue["gate_history"], color="#f43f5e", lw=1.0, ls=":", label="$\\gamma$-secretase gate $g(t)$")
    ax.set_xlabel("Somite Stage $s$")
    ax.set_ylabel("$\\kappa(t)/\\kappa_{\\max}$,  $u(t)$,  $g(t)$")
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.02, 1.08)
    ax.set_title("I. Mib1 DeltaC Ubiquitination Gate $\\kappa = \\kappa_{\\max} u\\, g$; Dose Sweep (inset)", fontweight="bold", fontsize=9.6)
    ax.legend(loc="upper left", fontsize=6.8, framealpha=0.88, facecolor="#0f172a")
    ins2 = ax.inset_axes([0.55, 0.10, 0.43, 0.44])
    ins2.set_facecolor("#0f172a")
    mm = [d["mib1_dose"] for d in m_sweep]
    ins2.plot(mm, [d["u_ss"] for d in m_sweep], "--", color="#94a3b8", lw=1.2, label="$u_{ss}(M)$")
    ins2.plot(mm, [d["late_mean_R"] for d in m_sweep], "o-", color="#fb923c", lw=1.4, ms=3.5, label="late $R_{\\mathrm{front}}$")
    ins2.axhline(R_CRIT, color="#facc15", lw=0.9, ls=":")
    ins2.set_xlabel("Mib1 activity $M$", fontsize=7, color="#e2e8f0")
    ins2.tick_params(labelsize=6.5, colors="#cbd5e1")
    ins2.legend(fontsize=6.0, framealpha=0.7, facecolor="#0f172a", loc="lower right")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def render_growth_gif(out_gif: Path = OUTPUT_GIF) -> Path:
    """Render an animated GIF (`docs/media/delta_notch_phase_synchrony_growth.gif`) comparing wild-type
    Delta-Notch synchronization against the Riedel-Kruse et al. (2007) DAPT pulse-chase rescue experiment:
    front order parameter R(s), the kinematic phase-wave ribbon across the PSM (lateral-mean phasor of the
    advected lattice) and the emergent somite boundary profile (defective somites in the emergent [ALD, PRP) window).
    """
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    sim_wt = simulate_delayed_kuramoto_psm_lattice(seed=2007)
    sim_rescue = simulate_delayed_kuramoto_psm_lattice(s_in_somite=0.0, s_out_somite=9.0, seed=2007)
    s_axis = sim_wt["s_axis"]
    L_psm = sim_wt["profile"]["L_psm_somites"]

    frames_pil: list[Image.Image] = []
    n_frames = 22
    for f_idx in range(n_frames):
        s_front = float(f_idx + 1)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.4, 6.6), dpi=100, gridspec_kw={"height_ratios": [2.2, 1.2, 1.6]})
        fig.patch.set_facecolor("#0b0f19")
        for a in (ax1, ax2, ax3):
            a.set_facecolor("#131b2e")
            for sp in a.spines.values():
                sp.set_color("#3b4c6b")
            a.tick_params(colors="#cbd5e1")

        for sim, col, lab in ((sim_wt, "#38bdf8", "Wild-Type Notch+ $R_{\\mathrm{front}}(s)$"), (sim_rescue, "#34d399", "DAPT Pulse $[0, 9]$ + Washout $R_{\\mathrm{front}}(s)$")):
            mask_t = sim["somite_time"] <= s_front
            ax1.plot(sim["somite_time"][mask_t], sim["R_history"][mask_t], color=col, lw=2.0, label=lab)
        ax1.axhline(R_CRIT, color="#facc15", lw=1.4, ls=":", label="$R_{\\mathrm{crit}}=0.50$")
        ax1.axvspan(sim_rescue["ald_somite"], sim_rescue["prp_somite"], color="#f43f5e", alpha=0.14, label=f"Emergent defective window [{sim_rescue['ald_somite']:.1f}, {sim_rescue['prp_somite']:.1f}]")
        ax1.set_xlim(0.0, 22.0)
        ax1.set_ylim(0.0, 1.08)
        ax1.set_title(f"Riedel-Kruse et al. (2007) Delta–Notch Synchrony, Kinematic Waves & DAPT Rescue — Somite {f_idx + 1}/22", color="#f8fafc", fontweight="bold", fontsize=10.0)
        ax1.set_ylabel("Kuramoto $R_{\\mathrm{front}}(s)$", color="#e2e8f0")
        ax1.legend(loc="lower right", fontsize=7.4, facecolor="#0f172a", labelcolor="#f8fafc")

        # Kinematic wave ribbon across the PSM at the current stage (WT above, DAPT pulse below)
        def _ribbon(sim: dict[str, Any]) -> np.ndarray:
            idx = int(np.searchsorted(sim["somite_time"], s_front))
            idx = min(idx, sim["kymograph"].shape[0] - 1)
            return np.real(sim["kymograph"][idx])[None, :]
        ribbon_w = np.vstack([np.tile(_ribbon(sim_wt), (10, 1)), np.tile(_ribbon(sim_rescue), (10, 1))])
        ax2.imshow(ribbon_w, aspect="auto", extent=[0.0, L_psm, 0.0, 2.0], cmap="twilight", vmin=-1, vmax=1)
        ax2.axhline(1.0, color="#f8fafc", lw=1.2)
        ax2.set_yticks([0.5, 1.5])
        ax2.set_yticklabels(["DAPT pulse", "Wild-Type"], color="#e2e8f0", fontsize=8.2)
        ax2.set_xlabel("PSM position (somite lengths from determination front → tailbud): $R_{\\mathrm{lat}}\\cos\\bar\\theta$ phase waves", color="#e2e8f0", fontsize=8.4)

        wt_strip = np.where(s_axis <= s_front, sim_wt["chi_ap"], -0.25)
        res_strip = np.where(s_axis <= s_front, sim_rescue["chi_ap"], -0.25)
        ribbon = np.vstack([np.tile(wt_strip, (14, 1)), np.tile(res_strip, (14, 1))])
        ax3.imshow(ribbon, aspect="auto", extent=[0.0, 22.0, 0.0, 2.0], cmap="coolwarm", vmin=-0.5, vmax=0.8)
        ax3.axhline(1.0, color="#f8fafc", lw=1.2)
        ax3.axvline(sim_rescue["ald_somite"], color="#f43f5e", lw=1.6, ls="--")
        ax3.axvline(sim_rescue["prp_somite"], color="#34d399", lw=1.6, ls="--")
        ax3.set_yticks([0.5, 1.5])
        ax3.set_yticklabels(["DAPT Pulse+Rescue", "Wild-Type (Notch+)"], color="#e2e8f0", fontsize=8.2)
        ax3.set_xlabel("Somite Position $s$ Along Zebrafish Axis [Anterior $\\rightarrow$ Posterior]", color="#e2e8f0")

        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames_pil.append(Image.fromarray(rgba[..., :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)

    frames_pil[0].save(out_gif, save_all=True, append_images=frames_pil[1:], duration=120, loop=0, optimize=True)
    return out_gif


# ============================================================================
# Section 8: Deterministic Self-Test Suite (`--self-test`)
# ============================================================================
def run_self_test() -> int:
    """Run deterministic mathematical, morphometric, and reproducibility assertions."""
    print("[2006-riedel-delta-notch-phase-synchrony] Running --self-test suite...")
    checks = 0
    digests = verify_dataset_integrity()
    assert len(digests) == 3, "Expected 3 SHA-256-verified dataset files"
    checks += 1

    # 1. Collective frequency shift & Morelli et al. (2009) linear stability stiffness criterion
    stab_wt = solve_collective_frequency_and_stability(T0_MIN, KAPPA_WT_PER_MIN, TAU_C_MIN)
    stab_dapt = solve_collective_frequency_and_stability(T0_MIN, 0.0, TAU_C_MIN)
    stab_anti = solve_collective_frequency_and_stability(T0_MIN, KAPPA_WT_PER_MIN, 11.0)
    assert stab_wt["in_phase_stable"] and stab_wt["kappa_eff_per_min"] > 0.04
    assert abs(stab_dapt["T_sync_min"] - 25.0) < 1e-6, "Uncoupled DAPT period must equal T_0 = 25.0 min"
    assert not stab_anti["in_phase_stable"] and stab_anti["kappa_eff_per_min"] < 0.0
    assert abs(stab_wt["T_sync_min"] - 22.18) < 0.02, f"T_sync expected 22.18, got {stab_wt['T_sync_min']}"
    checks += 1
    # Reference JSON parity with the solver (Pass 3 fixed the stale 0.2896 / 21.70 / 0.0566 entries)
    ref = load_reference_data()
    canon = ref["riedel_kruse_2007_morelli_2009_governing_equations"]["canonical_zebrafish_psm_parameters"]
    assert abs(canon["collective_frequency_Omega_rad_per_min"] - stab_wt["Omega_rad_per_min"]) < 5e-4
    assert abs(canon["collective_period_Tsync_min"] - stab_wt["T_sync_min"]) < 0.02
    assert abs(canon["effective_stiffness_kappa_eff_per_min"] - stab_wt["kappa_eff_per_min"]) < 5e-4
    checks += 1

    # 2. Bitwise determinism & wild-type front synchrony of the advected lattice
    sim_wt1 = simulate_delayed_kuramoto_psm_lattice(seed=2007)
    sim_wt2 = simulate_delayed_kuramoto_psm_lattice(seed=2007)
    assert np.allclose(sim_wt1["R_history"], sim_wt2["R_history"], atol=0.0), "Bitwise determinism failed"
    assert sim_wt1["late_mean_R"] > 0.85, f"Expected WT late front R > 0.85, got {sim_wt1['late_mean_R']}"
    assert sim_wt1["ald_somite"] == NO_DEFECT_SENTINEL and not sim_wt1["defective_somites"], "WT must form no defective somites"
    assert sim_wt1["n_advection_shifts"] == 22 * CELLS_PER_SOMITE, "Advection must deliver exactly cells_per_somite columns per somite"
    checks += 1

    # 3. Kinematic phase waves (Law 4): gradient -> waves, uniform -> none, analytic agreement
    sim_uniform = simulate_delayed_kuramoto_psm_lattice(gradient_on=False, seed=2007)
    assert sim_wt1["n_waves_sim"] > 1.0, f"Expected > 1 kinematic wave across the PSM, got {sim_wt1['n_waves_sim']}"
    assert abs(sim_wt1["n_waves_sim"] - sim_wt1["n_waves_analytic"]) / sim_wt1["n_waves_analytic"] < 0.30, (
        f"Simulated N_w {sim_wt1['n_waves_sim']} deviates > 30 % from analytic {sim_wt1['n_waves_analytic']}"
    )
    assert abs(sim_uniform["n_waves_sim"]) < 0.25, f"Uniform profile must carry ~no waves, got {sim_uniform['n_waves_sim']}"
    assert sim_wt1["anterior_wavelength_over_S_sim"] > 1.2, "Anterior wavelength must exceed one somite length (waves slow toward the front)"
    sweep = sweep_frequency_profile(f_values=(0.3, 0.5, 0.7, 0.9))
    nw = [d["n_waves_sim"] for d in sweep]
    assert all(nw[i] > nw[i + 1] for i in range(len(nw) - 1)), f"N_w must decrease monotonically with f_A: {nw}"
    checks += 1

    # 4. DAPT failure (emergent ALD), washout rescue (emergent PRP) and Law 6 span invariance
    sim_dapt = simulate_delayed_kuramoto_psm_lattice(s_in_somite=0.0, s_out_somite=22.0, seed=2007)
    assert sim_dapt["late_mean_R"] < 0.40, f"Expected continuous DAPT late R < 0.40, got {sim_dapt['late_mean_R']}"
    assert 4.0 <= sim_dapt["ald_somite"] <= 7.5, f"Expected emergent ALD in [4, 7.5], got {sim_dapt['ald_somite']}"
    assert sim_dapt["mean_segmentation_period_min"] > sim_wt1["mean_segmentation_period_min"] + 1.0, "DAPT must lengthen the segmentation period (Herrgen 2010)"
    sim_rescue = simulate_delayed_kuramoto_psm_lattice(s_in_somite=0.0, s_out_somite=9.0, seed=2007)
    assert sim_rescue["min_R"] < 0.45 and sim_rescue["late_mean_R"] > 0.80, (
        f"Expected DAPT pulse-washout to drop below 0.45 ({sim_rescue['min_R']}) then rescue above 0.80 ({sim_rescue['late_mean_R']})"
    )
    assert 12.0 <= sim_rescue["prp_somite"] <= 18.0, f"Expected emergent PRP in [12, 18], got {sim_rescue['prp_somite']}"
    assert abs((sim_rescue["prp_somite"] - sim_rescue["ald_somite"]) - (sim_rescue["s_recross"] - sim_rescue["s_cross"])) < 0.02, (
        "Law 6: PRP - ALD must equal s_recross - s_cross"
    )
    assert sim_rescue["defective_somites"] and min(sim_rescue["defective_somites"]) >= math.floor(sim_rescue["ald_somite"]), "Defective somites must start at the emergent ALD"
    checks += 1

    # 5. Mib1 ubiquitination gate (Law 5): monotone u_ss, kappa <= kappa_max, mib-null failure, restoration rescue
    u_vals = [mib1_ubiquitination_steady_state(M) for M in (0.0, 0.1, 0.5, 1.0, 2.0)]
    assert u_vals[0] == 0.0 and all(u_vals[i] < u_vals[i + 1] for i in range(len(u_vals) - 1)) and u_vals[-1] < 1.0
    assert np.all(sim_rescue["kappa_history"] <= sim_rescue["kappa_max_per_min"] + 1e-12)
    assert abs(float(sim_wt1["kappa_history"][-1]) - KAPPA_WT_PER_MIN) < 1e-9, "WT steady kappa must equal kappa_WT"
    sim_mib_null = simulate_delayed_kuramoto_psm_lattice(mib1_dose=0.0, seed=2007)
    assert sim_mib_null["late_mean_R"] < 0.40 and 3.0 <= sim_mib_null["ald_somite"] <= 8.0, f"mib1-null must desynchronise (R={sim_mib_null['late_mean_R']}, ALD={sim_mib_null['ald_somite']})"
    sim_mib_restore = simulate_delayed_kuramoto_psm_lattice(mib1_dose=1.0, mib1_restore_somite=9.0, seed=2007)
    assert sim_mib_restore["late_mean_R"] > 0.65 and sim_mib_restore["prp_somite"] >= sim_rescue["prp_somite"] - 0.5, (
        f"Mib1 restoration must rescue (R={sim_mib_restore['late_mean_R']}) no earlier than DAPT washout (PRP {sim_mib_restore['prp_somite']} vs {sim_rescue['prp_somite']})"
    )
    checks += 1

    # 6. 5-regime emergent validation against Riedel-Kruse (2007)
    regimes = evaluate_dapt_regimes(ref, seeds=REGIME_SEEDS[:3])
    val = regime_validation_statistics(regimes)
    assert val["pearson_r_emergent"] > 0.98, f"Emergent 5-regime Pearson r must exceed 0.98, got {val['pearson_r_emergent']}"
    assert val["rmse_emergent_somites"] < 1.5, f"Emergent RMSE must be < 1.5 somites, got {val['rmse_emergent_somites']}"
    for r in regimes:
        if r["regime_id"] == "wildtype_vehicle_control":
            assert r["sim_ald_somite"] == NO_DEFECT_SENTINEL
        elif r["regime_id"] == "dapt_pulse_washout_rescue":
            assert abs(r["sim_prp_somite"] - r["obs_prp_somite"]) <= 2.0 * r["obs_prp_sd_somite"] + 0.5
        else:
            assert abs(r["sim_ald_somite"] - r["obs_ald_somite"]) <= 2.0 * r["obs_ald_sd_somite"] + 0.5
    checks += 1

    # 7. Specimen segmentation & measure_morphoeval()
    spec = segment_zebrafish_psm_specimen()
    assert spec["upper_row"]["n_somites"] >= 7 and spec["lower_row"]["n_somites"] >= 8
    assert 40.0 <= spec["bilateral_mean_spacing_px"] <= 56.0
    meas = measure_morphoeval()
    assert len(meas) >= 8, f"Expected >= 8 MorphoEval measurements, got {len(meas)}"
    assert all("reference_value" in m and "unit" in m for m in meas)
    checks += 1

    print(
        f"[2006-riedel-delta-notch-phase-synchrony] SELF-TEST PASSED ({checks}/{checks} checks): "
        f"T_sync={stab_wt['T_sync_min']:.2f} min (T_0=25.00 min), kappa_eff={stab_wt['kappa_eff_per_min']:.4f} min^-1 "
        f"(anti={stab_anti['kappa_eff_per_min']:.4f}), WT R_front={sim_wt1['late_mean_R']:.3f}, N_w={sim_wt1['n_waves_sim']:.2f} "
        f"(analytic {sim_wt1['n_waves_analytic']:.2f}), emergent ALD={sim_dapt['ald_somite']:.2f}, "
        f"PRP={sim_rescue['prp_somite']:.2f} (R_min={sim_rescue['min_R']:.3f} -> R_rescue={sim_rescue['late_mean_R']:.3f}), "
        f"mib1-null R={sim_mib_null['late_mean_R']:.3f}, 5-regime r={val['pearson_r_emergent']:.4f} RMSE={val['rmse_emergent_somites']:.2f}, "
        f"specimen S_obs={spec['bilateral_mean_spacing_px']:.2f} px."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Riedel-Kruse et al. (2007) Science / Morelli et al. (2009) HFSP J. Delayed Kuramoto Delta-Notch Phase Synchrony"
    )
    ap.add_argument("--self-test", action="store_true", help="Run deterministic self-test suite and exit")
    ap.add_argument("--kappa", type=float, default=KAPPA_WT_PER_MIN, help="Wild-type Delta-Notch coupling strength kappa in min^-1")
    ap.add_argument("--tau-c", type=float, default=TAU_C_MIN, help="Intercellular coupling delay tau_c in min (tailbud)")
    ap.add_argument("--s-in", type=float, default=0.0, help="DAPT addition somite stage s_in")
    ap.add_argument("--s-out", type=float, default=9.0, help="DAPT washout somite stage s_out")
    ap.add_argument("--f-anterior", type=float, default=F_ANTERIOR, help="FGF8/Wnt frequency profile: omega(front)/omega_0")
    ap.add_argument("--mib1-dose", type=float, default=1.0, help="Mib1 E3-ligase activity M (1 = wild type, 0 = mib null)")
    ap.add_argument("--tau-wash", type=float, default=TAU_WASH_MIN, help="DAPT first-order clearance time after washout (min)")
    ap.add_argument("--no-gradient", action="store_true", help="Disable the FGF8/Wnt frequency gradient (uniform PSM)")
    ap.add_argument("--seed", type=int, default=2007, help="Deterministic RNG seed")
    ap.add_argument("--skip-gif", action="store_true", help="Skip rendering animated GIF")
    ap.add_argument("--summary", action="store_true", help="Print a single-run emergent summary for the CLI parameters and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    verify_dataset_integrity()
    if args.summary:
        sim = simulate_delayed_kuramoto_psm_lattice(
            kappa_per_min=args.kappa, tau_c_min=args.tau_c, s_in_somite=args.s_in, s_out_somite=args.s_out,
            f_anterior=args.f_anterior, gradient_on=not args.no_gradient, mib1_dose=args.mib1_dose,
            tau_wash_min=args.tau_wash, seed=args.seed,
        )
        print(json.dumps({k: v for k, v in sim.items() if not isinstance(v, (np.ndarray, dict))}, indent=2, default=str))
        return 0
    png_path = render_comparison_figure(OUTPUT_PNG)
    print(f"[2006-riedel-delta-notch-phase-synchrony] Rendered figure -> {png_path.relative_to(ROOT)}")
    if not args.skip_gif:
        gif_path = render_growth_gif(OUTPUT_GIF)
        print(f"[2006-riedel-delta-notch-phase-synchrony] Rendered animation -> {gif_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
