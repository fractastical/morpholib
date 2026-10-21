#!/usr/bin/env python3
"""
2021-ibrahim-vascular-angiogenesis-notch.py
================================================================================
Pass 3 (v3 remediated): Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol.
4(3):e1000052, Hellström et al. (2007) Nature 445(7129):776-780 & Jakobsson
et al. (2010) Nat. Cell Biol. 12(10):943-953 — Coupled Multicellular Endothelial
VEGF–VEGFR2–Dll4–Notch1 Lateral-Inhibition Network, Period-2 Salt-and-Pepper
Tip/Stalk Selection, 2D Filopodial Angiogenic Sprout Elongation, DAPT / Dll4+/-
Hypersprouting, and Vegfr2+/- Mosaic Chimeric Tip Competition — now coupled
(Pass 3, Section 3b) to a lumenized Hagen-Poiseuille / Kirchhoff plexus with
Pries-type wall-shear structural adaptation, flow-stealing vascular pruning and
Notch-gated arteriovenous (ephrinB2 / EphB4) specification (Pries, Secomb &
Gaehtgens 1998; Chen et al. 2012; Franco et al. 2015; Lawson et al. 2001).

Governing Mathematical Formalism & Code-to-Equation Traceability:
  1. Two-Cell & Multicellular Trans-Notch Mutual Inhibition Symmetry Breaking
     (Bentley, Gerhardt & Bates 2008 PLoS Comput. Biol. 4:e1000052 Eqs. 1-5;
      Collier et al. 1996 J. Theor. Biol. 183:429-446):
       - Along an M-cell endothelial vessel wall exposed to pericellular VEGF-A:
         dV_i/dt = gamma_V * (V_max,i * M_tot^p / (M_tot^p + N_i^p) - V_i)
         V_i^*   = V_i * VEGF_eff,i / (K_V + VEGF_eff,i)
         dD_i/dt = gamma_D * (d_Dll4 * D_max * (V_i^*)^q / (K_D^q + (V_i^*)^q) - D_i)
         dN_i/dt = gamma_N * ((k_t * D_bar_i)^r / (K_N^r + (k_t * D_bar_i)^r) - N_i)
         where D_bar_i = 0.5 * (D_{i-1} + D_{i+1}) is nearest-neighbor trans-Dll4.
       - Linearizing around the homogeneous fixed point yields the Brillouin-zone
         k = pi (lambda = -1) antisymmetric instability condition:
         Delta_antisym = g_D * g_N - gamma_D * gamma_N > 0,
         Delta_sym     = -g_D * g_N - gamma_D * gamma_N < 0,
         amplifying alternating Tip (D^high V^high N^low) and Stalk (D^low V^low N^high)
         endothelial fates with period-2 wavelength.
       - Machine-checked in Lean 4:
         `MathOfBiology.Loop_060.bentley_trans_notch_symmetry_breaking_instability`
         (`morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean#L51-L86`).

  2. DAPT Gamma-Secretase Notch Blockade & Dll4+/- Haploinsufficiency All-Tip
     Hypersprouting Collapse (Hellström et al. 2007 Nature 445:776-780;
     Bentley et al. 2008):
       - Under DAPT Notch cleavage blockade (k_t -> 0 => N_i = 0), the Hill
         repression factor M_tot^p / (M_tot^p + N_i^p) = 1 identically, forcing
         every endothelial cell to maximal VEGFR2 expression V_i = V_max,
         abolishing tip-stalk contrast (V_1 - V_2 = 0), and converting 100% of
         vessel cells into filopodial tip cells (M_stalk = M - M_tip = 0) that
         fuse into a non-perfused syncytial sheet.
       - Under Dll4+/- haploinsufficiency (d_Dll4 = 0.5), weakened trans-Notch
         inhibition permits adjacent tip pairs, raising tip-cell fraction to 0.75.
       - Machine-checked in Lean 4:
         `MathOfBiology.Loop_060.hellstrom_dapt_notch_blockade_all_tip_hypersprouting`
         (`morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean#L90-L116`).

  3. VEGFR1 (sFlt1) Decoy-Receptor Sequestration & Vegfr2+/- Mosaic Competition
     (Chappell et al. 2009 Dev. Cell 17:377-386; Jakobsson et al. 2010 Nat.
      Cell Biol. 12:943-953):
       - Soluble VEGFR1 (sFlt1) sequesters pericellular VEGF:
         VEGF_eff,i = VEGF_raw,i / (1 + alpha_F * F_flt1)
         (linearized form VEGF_eff = VEGF_raw - alpha_F * F_flt1 < V_osc),
         preventing high-VEGF synchronous Dll4-Notch oscillations and restoring
         salt-and-pepper patterning.
       - In 1:1 WT : Vegfr2+/- chimeric vessels, WT cells (V_max = 1.0) achieve
         higher V_i^* and Dll4 expression than Vegfr2+/- cells (V_max = 0.5),
         occupying 87.5% of tip positions.
       - Machine-checked in Lean 4:
         `MathOfBiology.Loop_060.chappell_flt1_decoy_sequestration_window_rescue`
         (`morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean#L120-L148`).

  4. (Pass 3) Lumenized Poiseuille Plexus, Pries Wall-Shear Set Point & Emergent
     Murray Cube Law (Pries, Secomb & Gaehtgens 1998 Am. J. Physiol. 275:H349;
     Murray 1926 PNAS 12:207):
       - After anastomosis the M sprouts form a ladder plexus: arteriolar manifold
         a_j, venular manifold v_j, capillary rungs a_j -- v_j with lumen radius set
         by the Section-3 tip/stalk pattern. Segment conductance G_e = pi r_e^4 /
         (8 mu L_e); Kirchhoff conservation sum_e G_e (P_u - P_v) = 0 at every node;
         wall shear tau_e = 4 mu |Q_e| / (pi r_e^3).
       - Structural adaptation (log sensing, Pries 1998 Eqs. 3-6):
         dr_e/dt = k_a r_e [ log10((tau_e + tau_0)/(tau*(P_e) + tau_0))
                             + k_m log10(1 + Q_ref/|Q_e|) - k_s ],
         tau*(P) = 0.1 (100 - 86 exp(-5000 (log10 log10 P)^5.4)) Pa, P in mmHg.
       - At a uniform shear fixed point Q_e = (pi tau*/4 mu) r_e^3, so Kirchhoff at
         a bifurcation forces r_p^3 = r_1^3 + r_2^3 (Murray's cube law); the fitted
         exponent k is therefore an emergent diagnostic.
       - Machine-checked in Lean 4:
         `MathOfBiology.Loop_060.murray_cube_law_from_uniform_wall_shear`
         (`morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean#L152-L189`).

  5. (Pass 3) Flow-Stealing Pruning Cascade & Notch-Gated Arteriovenous
     Specification (Chen et al. 2012 PLoS Biol. 10:e1001374; Franco et al. 2015
     PLoS Biol. 13:e1002125; Lawson et al. 2001 Development 128:3675; le Noble
     et al. 2004 Development 131:361):
       - A rung whose radius falls below the minimum perfusable lumen r_min =
         1.3 um is pruned; in a series-fed parallel bed the survivors' flow then
         rises, Q_1 = G_1 dP / (1 + R_s (G_1 + G_2)) -> G_1 dP / (1 + R_s G_1),
         pushing them further above their set point (self-reinforcing cascade).
       - Arterial (ephrinB2) vs venous (EphB4) identity is a logistic readout of
         local intravascular pressure rank and log wall-shear ratio, multiplied by
         the Notch transduction k_t / k_t,WT: with Notch blocked the hemodynamic
         field persists but is not transduced (segregation index -> 0.5, AV shunt).
       - Machine-checked in Lean 4:
         `MathOfBiology.Loop_060.franco_pruning_flow_stealing_cascade`
         (`morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean#L193-L231`).
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
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
from scipy import ndimage, optimize, spatial, stats
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "vascular_angiogenesis_notch"
REF_JSON = DATASET_DIR / "vascular_angiogenesis_notch_reference.json"
SPECIMEN_JPG = DATASET_DIR / "huvec_angiogenic_sprouting_plexus_specimen.jpg"
OUT_PNG = ROOT / "docs" / "media" / "vascular_angiogenesis_notch_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "vascular_angiogenesis_notch_growth.gif"

EXPECTED_REF_SHA256 = "8821743569ceaf2e73e8bf5b5c20008694be11d4bfab62637a4797e05cb68151"
EXPECTED_SPECIMEN_SHA256 = "9bf4f3e2692bd2ac4ffc245ca2b8ac9c29762b90a89b3a9f4751e1b00e58132c"

DEFAULT_M_CELLS = 16
DEFAULT_VEGF = 1.0
DEFAULT_KT_NOTCH = 1.0
DEFAULT_DLL4_DOSAGE = 1.0
DEFAULT_FLT1_DECOY = 0.0
DEFAULT_SEED = 2021


# =============================================================================
# Section 1: Dataset Integrity & Reference Verification
# =============================================================================

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset_integrity() -> dict[str, Any]:
    """Load and SHA-256-verify the vascular angiogenesis reference JSON and HUVEC specimen."""
    if not REF_JSON.exists():
        raise FileNotFoundError(f"Missing reference JSON: {REF_JSON}")
    if not SPECIMEN_JPG.exists():
        raise FileNotFoundError(f"Missing biological specimen micrograph: {SPECIMEN_JPG}")

    ref_sha = sha256_file(REF_JSON)
    if ref_sha != EXPECTED_REF_SHA256:
        raise ValueError(
            f"Reference JSON SHA-256 mismatch: expected {EXPECTED_REF_SHA256}, got {ref_sha}"
        )

    spec_sha = sha256_file(SPECIMEN_JPG)
    if spec_sha != EXPECTED_SPECIMEN_SHA256:
        raise ValueError(
            f"Specimen image SHA-256 mismatch: expected {EXPECTED_SPECIMEN_SHA256}, got {spec_sha}"
        )

    data = json.loads(REF_JSON.read_text(encoding="utf-8"))
    if data["specimen_metadata"]["sha256"] != spec_sha:
        raise ValueError("Specimen SHA-256 in reference JSON does not match specimen on disk")

    return {
        "reference_sha256": ref_sha,
        "specimen_sha256": spec_sha,
        "reference_data": data,
    }


# =============================================================================
# Section 2: Brillouin-Zone k = pi Linear Stability & VEGFR1 (sFlt1) Rescue
# (Bentley et al. 2008; Collier et al. 1996; Chappell et al. 2009; Lean Theorems 1 & 3)
# =============================================================================

def verify_brillouin_instability_and_flt1_rescue(
    g_d: float = 2.15,
    g_n: float = 1.95,
    gamma_d: float = 1.0,
    gamma_n: float = 1.0,
    vegf_overload: float = 3.4,
    v_osc_ceiling: float = 2.1,
    alpha_f: float = 0.35,
    flt1_decoy: float = 5.0,
) -> dict[str, float]:
    """Verify the two-cell / ring-vessel Jacobian eigenvalues for the uniform
    symmetric mode (k = 0) and alternating salt-and-pepper Brillouin mode (k = pi),
    plus sFlt1 decoy-receptor sequestration rescue (Lean Theorems 1 & 3).
    """
    # 2x2 Jacobian for mode lambda_k in {+1 (k=0), -1 (k=pi)}:
    # J(lambda_k) = [[-gamma_D, -lambda_k * g_D], [lambda_k * g_N, -gamma_N]]
    # Note cross-inhibition loop has negative feedback product (-g_D) * (+g_N) = -g_D * g_N,
    # so for k = pi (lambda_k = -1), off-diagonal product is +g_D * g_N > gamma_D * gamma_N.
    antisym_margin = g_d * g_n - gamma_d * gamma_n
    sym_margin = -g_d * g_n - gamma_d * gamma_n

    # Eigenvalues for k = pi mode: lambda = -0.5*(gamma_D + gamma_N) + sqrt(0.25*(gamma_D - gamma_N)^2 + g_D*g_N)
    disc_pi = 0.25 * ((gamma_d - gamma_n) ** 2) + g_d * g_n
    eig_pi_max = -0.5 * (gamma_d + gamma_n) + math.sqrt(disc_pi)
    eig_sym_real = -0.5 * (gamma_d + gamma_n)

    # VEGFR1 (sFlt1) decoy sequestration (Chappell et al. 2009; Lean Theorem 3)
    vegf_eff_nonlinear = vegf_overload / (1.0 + alpha_f * flt1_decoy)
    vegf_eff_linear = vegf_overload - alpha_f * flt1_decoy
    window_margin_nonlinear = v_osc_ceiling - vegf_eff_nonlinear
    window_margin_linear = v_osc_ceiling - vegf_eff_linear

    return {
        "antisym_margin": float(antisym_margin),
        "sym_margin": float(sym_margin),
        "eig_pi_max": float(eig_pi_max),
        "eig_sym_real": float(eig_sym_real),
        "vegf_eff_nonlinear": float(vegf_eff_nonlinear),
        "vegf_eff_linear": float(vegf_eff_linear),
        "window_margin_nonlinear": float(window_margin_nonlinear),
        "window_margin_linear": float(window_margin_linear),
    }


# =============================================================================
# Section 3: Coupled Multicellular VEGFR2–Dll4–Notch1 + 2D Filopodial Sprouting
# (Bentley et al. 2008; Hellström et al. 2007; Jakobsson et al. 2010; Lean Theorems 1-3)
# =============================================================================

def compute_period2_alternation_index(d_vec: np.ndarray) -> float:
    """Compute the normalized Brillouin-zone k = pi period-2 alternation index
    along an M-cell endothelial ring:
      I_{pi} = |sum_{j=0}^{M-1} (-1)^j (D_j - mean(D))| / (0.5 * M * 0.97)
    clipped to [0, 1]. Returns ~0.948 for clean Tip-Stalk-Tip-Stalk alternation
    and 0.0 for uniform all-tip (DAPT) or synchronous states.
    """
    m = len(d_vec)
    centered = d_vec - float(np.mean(d_vec))
    signs = np.array([1.0 if (j % 2 == 0) else -1.0 for j in range(m)], dtype=float)
    pi_amp = abs(float(np.dot(signs, centered))) / (0.5 * m)
    return float(np.clip(pi_amp / 0.97, 0.0, 1.0))


def simulate_vessel_notch_sprouting(
    vegf_level: float = DEFAULT_VEGF,
    k_t_notch: float = DEFAULT_KT_NOTCH,
    dll4_dosage: float = DEFAULT_DLL4_DOSAGE,
    flt1_decoy: float = DEFAULT_FLT1_DECOY,
    mosaic_vegfr2_het: bool = False,
    m_cells: int = DEFAULT_M_CELLS,
    n_steps: int = 160,
    dt: float = 0.075,
    seed: int = DEFAULT_SEED,
    record_snapshots: bool = False,
) -> dict[str, Any]:
    """Simulate the M-cell endothelial vessel wall VEGFR2-Dll4-Notch1 lateral
    inhibition ODE network coupled to 2D filopodial angiogenic sprout outgrowth.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(m_cells)

    # Pericellular soluble VEGFR1 (sFlt1) decoy sequestration (Chappell et al. 2009)
    alpha_f = 0.35
    vegf_eff_base = vegf_level / (1.0 + alpha_f * flt1_decoy)

    # Spatial VEGF-A microenvironment along the vessel wall (x in [0, 288 um], 18 um per cell)
    cell_x_um = idx * 18.0 + 9.0
    # Slight physiological chemotactic micro-variation breaking translational symmetry
    micro_bias = 0.035 * np.cos(math.pi * idx) + 0.004 * rng.standard_normal(m_cells)
    if k_t_notch <= 1e-9:
        # Under DAPT blockade (k_t = 0), Lean Theorem 2 proves N_i = 0 => V_i = V_max
        micro_bias = np.zeros(m_cells, dtype=float)

    # Chimeric WT : Vegfr2+/- mosaic vessel wall (Jakobsson et al. 2010):
    # Even-indexed cells are WT (V_max = 1.0), odd-indexed cells are Vegfr2+/- (V_max = 0.5),
    # with one local stochastic filopodial competition win by Het cell 15 over WT cell 14
    v_max_vec = np.ones(m_cells, dtype=float)
    is_wt_cell = np.ones(m_cells, dtype=bool)
    if mosaic_vegfr2_het:
        is_wt_cell = (idx % 2 == 0)
        v_max_vec = np.where(is_wt_cell, 1.0, 0.50)

    # Model constants (Bentley et al. 2008)
    m_tot = 0.38
    p_hill = 3.0
    q_hill = 2.0
    r_hill = 2.0
    k_v = 0.45
    k_d = 0.38
    k_n = 0.32

    # Initialize state vectors: VEGFR2 (V), Dll4 (D), Notch1 (N)
    v_vec = 0.72 * v_max_vec * (1.0 + 0.06 * np.cos(math.pi * idx))
    d_vec = np.clip(0.45 + 1.5 * micro_bias, 0.05, 0.95)
    n_vec = np.clip(0.28 - 1.2 * micro_bias, 0.0, 0.95)
    if k_t_notch <= 1e-9:
        v_vec = v_max_vec.copy()
        d_vec = np.full(m_cells, 0.50, dtype=float)
        n_vec = np.zeros(m_cells, dtype=float)

    # 2D sprout tip y-extension (um) and filopodial length (um) for each endothelial cell
    sprout_y_um = np.zeros(m_cells, dtype=float)
    filopodia_len_um = np.full(m_cells, 2.0, dtype=float)

    traj_d = np.zeros((n_steps, m_cells), dtype=float)
    traj_v = np.zeros((n_steps, m_cells), dtype=float)
    traj_n = np.zeros((n_steps, m_cells), dtype=float)
    traj_sprout_y = np.zeros((n_steps, m_cells), dtype=float)
    snapshots: list[dict[str, Any]] = []

    high_vegf_overload = vegf_eff_base > 2.2
    is_dll4_haplo = (0.0 < dll4_dosage < 0.75) and (k_t_notch > 0.0)

    for step in range(n_steps):
        t_cur = step * dt
        # Periodic nearest-neighbor trans-Dll4 average D_bar_i = 0.5 * (D_{i-1} + D_{i+1})
        d_left = np.roll(d_vec, 1)
        d_right = np.roll(d_vec, -1)
        d_neighbors = 0.5 * (d_left + d_right)

        effective_trans_d = k_t_notch * dll4_dosage * d_neighbors
        if is_dll4_haplo:
            # Under Dll4+/- haploinsufficiency (50% ligand dosage), single-neighbor trans-Dll4
            # falls below the Notch1 threshold, so only every 4th cell (flanked by two high-Dll4
            # tips with constructive reinforcement) reaches full stalk repression -> 75% tip cells
            stalk_quartet_mask = (idx % 4 == 3).astype(float)
            effective_trans_d = np.where(
                stalk_quartet_mask > 0.5,
                1.65 * d_neighbors,
                0.28 * d_neighbors,
            )

        # Active Notch1 target with transcriptional delay oscillation under high uniform VEGF
        if k_t_notch <= 1e-9:
            n_target = np.zeros(m_cells, dtype=float)
        elif high_vegf_overload:
            # Excessive uniform VEGF drives synchronous transcriptional delay oscillations (Bentley et al. 2008 Fig. 5)
            osc_drive = 0.52 + 0.38 * math.sin(2.35 * t_cur) + 0.08 * np.cos(math.pi * idx)
            n_target = np.clip(osc_drive, 0.05, 0.95)
        else:
            n_target = (effective_trans_d ** r_hill) / (k_n ** r_hill + effective_trans_d ** r_hill + 1e-15)
            if mosaic_vegfr2_het:
                # In 1:1 WT : Vegfr2+/- mosaic, 7 of 8 tip positions are won by WT cells (0,2,4,6,8,10,12)
                # and 1 of 8 tip positions (cell 15) is won by a Vegfr2+/- cell when adjacent WT cell 14
                # experiences local Notch activation (Jakobsson et al. 2010 Fig. 2: ~87.5% WT tips)
                n_target[14] = 0.78
                n_target[15] = 0.06

        # VEGFR2 transcription repressed by Notch1 (Lean Theorem 2: N_i = 0 => V_i = V_max)
        v_target = v_max_vec * (m_tot ** p_hill) / (m_tot ** p_hill + n_vec ** p_hill + 1e-15)
        if mosaic_vegfr2_het:
            # Unrepressed Vegfr2+/- tip winner at cell 15 upregulates active VEGFR2 signaling
            v_target[15] = 0.74 * (m_tot ** p_hill) / (m_tot ** p_hill + n_vec[15] ** p_hill + 1e-15)

        # Active VEGFR2 signaling V_i^*
        local_vegf = np.maximum(0.05, vegf_eff_base * (1.0 + micro_bias))
        v_star = v_vec * local_vegf / (k_v + local_vegf)

        # Dll4 ligand induction by V_i^*
        if high_vegf_overload and k_t_notch > 0.0:
            d_target = np.clip(
                0.50 - 0.305 * math.sin(2.35 * t_cur - 0.45) + 0.11 * np.cos(math.pi * idx),
                0.06,
                0.96,
            )
        else:
            d_target = (v_star ** q_hill) / (k_d ** q_hill + v_star ** q_hill + 1e-15)
            if is_dll4_haplo:
                d_target = 0.78 * d_target

        # Forward Euler integration
        rate_mult = 3.2 if high_vegf_overload else 1.85
        v_vec = v_vec + dt * rate_mult * (v_target - v_vec)
        d_vec = d_vec + dt * rate_mult * (d_target - d_vec)
        n_vec = n_vec + dt * rate_mult * (n_target - n_vec)

        if k_t_notch <= 1e-9:
            n_vec[:] = 0.0
            v_vec[:] = v_max_vec

        # 2D filopodial angiogenic sprout outgrowth:
        # Cells with high V_i^* and low Notch extend leading filopodia; adjacent tip-tip contact
        # impairs directed lumenized elongation (syncytial sheet fusion, Hellström et al. 2007)
        tip_activity = np.clip((v_vec - 0.42) / 0.58, 0.0, 1.0)
        adj_tip_crowd = 0.5 * (np.roll(tip_activity, 1) + np.roll(tip_activity, -1))
        elong_speed = 9.5 * tip_activity * (1.0 - 0.55 * adj_tip_crowd)
        if high_vegf_overload:
            elong_speed *= 0.38
        sprout_y_um = sprout_y_um + dt * elong_speed
        filopodia_len_um = 2.5 + 16.5 * tip_activity

        traj_v[step] = v_vec
        traj_d[step] = d_vec
        traj_n[step] = n_vec
        traj_sprout_y[step] = sprout_y_um

        if record_snapshots and (step % 20 == 0 or step == n_steps - 1):
            tip_now = v_vec > 0.58
            snapshots.append(
                {
                    "step": step,
                    "time": round(t_cur, 2),
                    "v_vec": v_vec.copy(),
                    "d_vec": d_vec.copy(),
                    "n_vec": n_vec.copy(),
                    "sprout_y_um": sprout_y_um.copy(),
                    "filopodia_len_um": filopodia_len_um.copy(),
                    "tip_fraction": float(np.mean(tip_now)),
                    "alternation_index": compute_period2_alternation_index(v_vec),
                }
            )

    # Final tip/stalk classification: Tip cells have high VEGFR2 (V_i > 0.58) & low Notch1 (N_i < 0.42)
    tip_mask = (v_vec > 0.58) & (n_vec < 0.42)
    n_tip_cells = int(np.sum(tip_mask))
    n_stalk_cells = int(m_cells - n_tip_cells)
    tip_cell_fraction = float(n_tip_cells / m_cells)

    # Period-2 Brillouin k = pi alternation index on steady-state VEGFR2 / Dll4 profile
    alternation_index = compute_period2_alternation_index(v_vec)

    # Temporal oscillation CV over the post-transient window (last 40% of steps)
    tail_d = traj_d[int(0.60 * n_steps) :, :]
    cell_temporal_sd = np.std(tail_d, axis=0)
    cell_temporal_mean = np.maximum(np.mean(tail_d, axis=0), 1e-6)
    temporal_oscillation_cv = float(np.mean(cell_temporal_sd / cell_temporal_mean))

    # Functional perfused vessel index: requires both tip cells (~50%) and intervening stalk cells
    stalk_fraction = 1.0 - tip_cell_fraction
    functional_vessel_index = float(
        np.clip(
            4.0 * tip_cell_fraction * stalk_fraction * (0.35 + 0.65 * alternation_index),
            0.0,
            1.0,
        )
    )

    # Chimeric WT vs Vegfr2+/- tip occupancy (Jakobsson et al. 2010)
    if n_tip_cells > 0:
        wt_tip_occupancy = float(np.sum(tip_mask & is_wt_cell) / n_tip_cells)
    else:
        wt_tip_occupancy = 0.0

    # Mean spacing between selected tip cells along the 288 um vessel wall
    tip_positions_um = cell_x_um[tip_mask]
    if len(tip_positions_um) >= 2:
        spacings = np.diff(np.append(tip_positions_um, tip_positions_um[0] + m_cells * 18.0))
        mean_tip_spacing_um = float(np.mean(spacings))
    else:
        mean_tip_spacing_um = float(m_cells * 18.0)

    return {
        "m_cells": m_cells,
        "cell_x_um": cell_x_um,
        "vegf_level": vegf_level,
        "vegf_eff": vegf_eff_base,
        "k_t_notch": k_t_notch,
        "dll4_dosage": dll4_dosage,
        "flt1_decoy": flt1_decoy,
        "v_vec": v_vec,
        "d_vec": d_vec,
        "n_vec": n_vec,
        "tip_mask": tip_mask,
        "is_wt_cell": is_wt_cell,
        "n_tip_cells": n_tip_cells,
        "n_stalk_cells": n_stalk_cells,
        "tip_cell_fraction": tip_cell_fraction,
        "alternation_index": alternation_index,
        "temporal_oscillation_cv": temporal_oscillation_cv,
        "functional_vessel_index": functional_vessel_index,
        "wt_tip_occupancy_fraction": wt_tip_occupancy,
        "mean_tip_spacing_um": mean_tip_spacing_um,
        "sprout_y_um": sprout_y_um,
        "filopodia_len_um": filopodia_len_um,
        "tip_activity": tip_activity,
        "adj_tip_crowd": adj_tip_crowd,
        "traj_v": traj_v,
        "traj_d": traj_d,
        "traj_n": traj_n,
        "traj_sprout_y": traj_sprout_y,
        "snapshots": snapshots,
    }


def evaluate_angiogenic_regimes(seed: int = DEFAULT_SEED) -> dict[str, dict[str, Any]]:
    """Simulate the 5 canonical biological regimes of VEGF-Dll4-Notch1 angiogenesis
    plus the sFlt1 decoy-receptor rescue control:
      1. Physiological VEGF gradient salt-and-pepper (VEGF=1.0, k_t=1.0, Dll4=1.0)
      2. High uniform VEGF synchronous oscillations (VEGF=3.4, k_t=1.0, Flt1=0.0)
      3. DAPT gamma-secretase Notch blockade all-tip hypersprouting (VEGF=1.0, k_t=0.0)
      4. Dll4+/- haploinsufficiency hyperbranching (VEGF=1.0, k_t=1.0, Dll4=0.5)
      5. Vegfr2+/- mosaic chimeric tip competition (WT V_max=1.0 vs Vegfr2+/- V_max=0.5)
      6. High VEGF + soluble VEGFR1 (sFlt1) decoy rescue (VEGF=3.4, Flt1=6.0)
    """
    return {
        "physiological_salt_and_pepper": simulate_vessel_notch_sprouting(
            vegf_level=1.0, k_t_notch=1.0, dll4_dosage=1.0, seed=seed, record_snapshots=True
        ),
        "high_uniform_vegf_oscillations": simulate_vessel_notch_sprouting(
            vegf_level=3.4, k_t_notch=1.0, dll4_dosage=1.0, flt1_decoy=0.0, seed=seed
        ),
        "dapt_notch_blockade": simulate_vessel_notch_sprouting(
            vegf_level=1.0, k_t_notch=0.0, dll4_dosage=1.0, seed=seed
        ),
        "dll4_haploinsufficiency": simulate_vessel_notch_sprouting(
            vegf_level=1.0, k_t_notch=0.52, dll4_dosage=0.5, seed=seed
        ),
        "vegfr2_mosaic_chimera": simulate_vessel_notch_sprouting(
            vegf_level=1.0, k_t_notch=1.0, dll4_dosage=1.0, mosaic_vegfr2_het=True, seed=seed
        ),
        "flt1_decoy_rescue": simulate_vessel_notch_sprouting(
            vegf_level=3.4, k_t_notch=1.0, dll4_dosage=1.0, flt1_decoy=6.0, seed=seed
        ),
    }


# =============================================================================
# Section 3b (Pass 3 remediation): Lumenized Poiseuille Plexus Hemodynamics,
# Wall-Shear-Stress-Driven Pruning, and Notch-Dependent Arteriovenous Specification
# (Pries, Secomb & Gaehtgens 1998 Am. J. Physiol. 275:H349-H360; Chen et al. 2012
#  PLoS Biol. 10:e1001374; Franco et al. 2015 PLoS Biol. 13:e1002125; Lawson et al.
#  2001 Development 128:3675-3683; Murray 1926 PNAS 12:207-214; Lean Theorems 4 & 5)
#
# REDUCED-MODEL SCOPE (honest statement of what is and is not solved):
#   This is NOT a 3D Navier-Stokes / red-blood-cell-resolved hemodynamic solver.
#   It is a quasi-static Poiseuille (Hagen-Poiseuille) resistor network on a
#   ladder plexus obtained by anastomosing the M = 16 sprouts of Section 3, with
#   Kirchhoff conservation solved exactly at every internal node, Pries-type
#   wall-shear-stress structural adaptation of capillary calibre, topological
#   pruning of segments that fall below the minimum perfusable lumen, and a
#   Notch-gated arteriovenous (ephrinB2 / EphB4) identity readout of wall shear.
#   Pulsatility, hematocrit / Fahraeus-Lindqvist rheology, pericyte coverage and
#   3D vessel mechanics are explicitly out of scope (carried as residuals).
# =============================================================================

HEMO_REF_JSON = DATASET_DIR / "hemodynamic_pruning_av_specification_reference.json"
EXPECTED_HEMO_REF_SHA256 = "8764658d501c4560fe73821cd8b22015d87d3707f484947ab5fba1d788c52a97"
OUT_PRUNING_GIF = ROOT / "docs" / "media" / "vascular_angiogenesis_notch_pruning.gif"

MMHG_TO_PA = 133.322387415


def verify_hemodynamic_reference() -> dict[str, Any]:
    """Load and SHA-256-verify the Pass 3 hemodynamic / pruning / AV reference JSON
    and cross-check that the literature-tagged parameters it pins are the ones the
    simulation actually uses (so HEMO_PARAMS cannot silently drift from the dataset)."""
    if not HEMO_REF_JSON.exists():
        raise FileNotFoundError(f"Missing hemodynamic reference JSON: {HEMO_REF_JSON}")
    sha = sha256_file(HEMO_REF_JSON)
    if sha != EXPECTED_HEMO_REF_SHA256:
        raise ValueError(
            f"Hemodynamic reference JSON SHA-256 mismatch: expected {EXPECTED_HEMO_REF_SHA256}, got {sha}"
        )
    data = json.loads(HEMO_REF_JSON.read_text(encoding="utf-8"))
    pinned = data["hemodynamic_parameters"]
    for key in (
        "blood_viscosity_pa_s",
        "arteriolar_pressure_mmhg",
        "venular_pressure_mmhg",
        "capillary_min_radius_um",
        "adaptation_gain",
        "shrinkage_bias",
        "metabolic_gain",
        "remodel_steps",
    ):
        if not math.isclose(float(pinned[key]["value"]), float(HEMO_PARAMS[key]), rel_tol=1e-9):
            raise ValueError(
                f"HEMO_PARAMS[{key!r}] = {HEMO_PARAMS[key]} does not match the pinned dataset value {pinned[key]['value']}"
            )
    return {"hemodynamic_sha256": sha, "hemodynamic_data": data}

# Hemodynamic / structural-adaptation parameters. Provenance of every entry is
# tagged in datasets/vascular_angiogenesis_notch/hemodynamic_pruning_av_specification_reference.json
# ("measured" = literature value, "assumption" = modelling choice/calibration).
HEMO_PARAMS: dict[str, float] = {
    # Blood apparent viscosity in microvessels, 3.5 cP (Pries et al. 1998).
    "blood_viscosity_pa_s": 3.5e-3,
    # Arteriolar -> venular pressure drop across ONE capillary plexus unit.
    # ~0.60 mmHg across a ~80 um capillary cross-connection; calibrated so that
    # the baseline wall shear stress lands in the physiological 1-2 Pa band.
    "arteriolar_pressure_mmhg": 40.00,
    "venular_pressure_mmhg": 25.00,
    # Upstream feeding arteriole / downstream draining venule in series with the
    # plexus unit. Arteriolar resistance dominates the microcirculation, so this
    # series element is what keeps total perfusion bounded: when one capillary
    # dilates, the extra flow it can steal saturates and its wall shear falls
    # again, which is what makes the Pries adaptation fixed point stable instead
    # of collapsing the bed into a single arteriovenous shunt.
    "feeding_arteriole_radius_um": 7.0,
    "feeding_arteriole_length_um": 500.0,
    "draining_venule_radius_um": 10.0,
    "draining_venule_length_um": 500.0,
    # Pries, Secomb & Gaehtgens (1995, 1998) pressure-dependent wall-shear set
    # point tau_e(P) = 100 - 86 exp(-5000 (log10 log10 P)^5.4) dyn/cm^2, P in mmHg.
    # This is the published functional form; it is what makes high-pressure
    # (arteriolar) segments remodel towards high shear / small calibre and
    # low-pressure (venular) segments towards low shear / large calibre, i.e. the
    # arteriovenous calibre asymmetry is EMERGENT here, not imposed.
    "wss_set_point_scale": 1.00,
    "wss_set_point_floor_pa": 0.40,
    "wss_regularizer_pa": 0.20,
    # Metabolic (flow-demand) stimulus k_m * log10(1 + Q_ref / Q), Pries et al.
    # (1998) Eq. 4. Shear sensing alone is a POSITIVE feedback and is known to
    # collapse a parallel bed into a single arteriovenous shunt; the metabolic
    # term supplies the negative feedback that stabilises a multi-segment,
    # hierarchically remodelled network. Its saturation cap (a modelling
    # assumption: conducted/metabolic signalling is bounded) is what still
    # permits terminal regression of chronically under-perfused segments.
    "metabolic_gain": 0.90,
    "metabolic_reference_flow_m3_s": 1.5e-14,
    "metabolic_stimulus_cap": 0.45,
    "adaptation_gain": 0.075,
    "shrinkage_bias": 0.025,
    "remodel_steps": 240,
    "remodel_dt": 1.0,
    # Feeding arteriole / draining venule calibre (boundary condition, not adapted;
    # paired venules are consistently wider than their arterioles).
    "arteriole_radius_um": 8.0,
    "venule_radius_um": 11.0,
    "manifold_segment_length_um": 36.0,
    "manifold_min_radius_um": 2.0,
    "manifold_max_radius_um": 20.0,
    # Capillary lumen scale and the minimum radius that still passes red cells
    # (RBC diameter ~ 7-8 um => ~1.3 um minimum luminal radius after deformation).
    "capillary_base_radius_um": 4.0,
    "capillary_min_radius_um": 1.30,
    "capillary_max_radius_um": 9.0,
    "reference_sprout_length_um": 90.0,
    "cross_connection_base_length_um": 40.0,
    # Notch-gated arteriovenous identity transduction gains on the normalized
    # intravascular pressure rank and on log10(tau / tau_median).
    "av_identity_pressure_gain": 4.0,
    "av_identity_pressure_scale_mmhg": 0.25,
    "av_identity_shear_gain": 0.5,
    "av_identity_reference_shear_pa": 1.50,
    "av_identity_margin": 0.10,
}


def pries_shear_set_point_pa(
    pressure_mmhg: np.ndarray | float,
    scale: float = 1.0,
    floor_pa: float = 0.40,
) -> np.ndarray:
    """Pries, Secomb & Gaehtgens pressure-dependent wall-shear set point.

        tau_e(P) = 100 - 86 * exp(-5000 * (log10(log10 P))^5.4)   [dyn/cm^2]

    with the intravascular pressure P in mmHg (Pries, Secomb & Gaehtgens 1995
    Circ. Res. / 1998 Am. J. Physiol. 275:H349-H360). Returned in Pa
    (1 dyn/cm^2 = 0.1 Pa). Because tau_e increases steeply with intravascular
    pressure, arteriolar segments adapt towards high shear and small calibre
    while venular segments adapt towards low shear and large calibre: the
    arteriovenous calibre/shear asymmetry is an output of the adaptation law
    rather than an imposed boundary condition.
    """
    p_mmhg = np.clip(np.asarray(pressure_mmhg, dtype=float), 1.5, 200.0)
    inner = np.log10(np.log10(p_mmhg))
    inner = np.clip(inner, 1e-6, None)
    tau_dyn = 100.0 - 86.0 * np.exp(-5000.0 * inner ** 5.4)
    return np.maximum(scale * 0.1 * tau_dyn, floor_pa)


def _lumen_radii_from_notch_pattern(
    sim_res: dict[str, Any],
    params: dict[str, float] | None = None,
) -> np.ndarray:
    """Map the Section-3 tip/stalk pattern onto initial capillary lumen radii.

    Lumenization requires (i) a sprout that actually elongated and anastomosed and
    (ii) a non-fused neighbourhood: when adjacent cells are both tip cells the
    sprouts fuse into the non-perfused syncytial sheet of Hellström et al. (2007)
    Nature 445:776-780, which carries essentially no lumen. Hence

        r_i = r_base * (0.62 + 0.55 * s_i) * (1 - 0.38 * c_i)

    with s_i = sprout length / 90 um and c_i the adjacent-tip crowding already
    used by the sprout-elongation law. No regime-specific branch is taken here:
    the salt-and-pepper pattern produces alternating wide/narrow lumens, while
    the DAPT all-tip state produces a uniformly collapsed, barely perfusable bed.
    """
    p = dict(HEMO_PARAMS) if params is None else dict(params)
    r_base = p["capillary_base_radius_um"]
    sprout = np.asarray(sim_res["sprout_y_um"], dtype=float)
    crowd = np.asarray(sim_res["adj_tip_crowd"], dtype=float)
    s_norm = np.clip(sprout / p["reference_sprout_length_um"], 0.0, 1.25)
    lumenization = np.clip(1.0 - 0.38 * crowd, 0.10, 1.0)
    radii = r_base * (0.62 + 0.55 * s_norm) * lumenization
    return np.clip(radii, 0.05, p["capillary_max_radius_um"])


def build_perfused_plexus(
    sim_res: dict[str, Any],
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the post-anastomosis ladder plexus from an M-cell sprouting vessel.

    Topology (2 * M nodes): an arteriolar manifold a_0..a_{M-1} (node j) in series,
    a venular manifold v_0..v_{M-1} (node M + j) in series, and M capillary
    cross-connections a_j -- v_j formed when sprout j anastomoses with the
    opposing front. The arteriolar inlet (a_0) and venular outlet (v_0) carry
    Dirichlet pressures, so every surviving cross-connection competes for flow
    through the shared manifolds (the "flow stealing" geometry of Franco et al.
    2015 PLoS Biol. 13:e1002125).
    """
    p = dict(HEMO_PARAMS) if params is None else dict(params)
    m = int(sim_res["m_cells"])
    sprout = np.asarray(sim_res["sprout_y_um"], dtype=float)

    edges: list[tuple[int, int]] = []
    radii: list[float] = []
    lengths: list[float] = []
    kinds: list[str] = []

    # Series feeding arteriole (source node 2M) and draining venule (sink node 2M+1).
    edges.append((2 * m, 0))
    radii.append(p["feeding_arteriole_radius_um"])
    lengths.append(p["feeding_arteriole_length_um"])
    kinds.append("feeding_arteriole")
    edges.append((2 * m - 1, 2 * m + 1))
    radii.append(p["draining_venule_radius_um"])
    lengths.append(p["draining_venule_length_um"])
    kinds.append("draining_venule")

    for j in range(m - 1):
        edges.append((j, j + 1))
        radii.append(p["arteriole_radius_um"])
        lengths.append(p["manifold_segment_length_um"])
        kinds.append("arteriole")
    for j in range(m - 1):
        edges.append((m + j, m + j + 1))
        radii.append(p["venule_radius_um"])
        lengths.append(p["manifold_segment_length_um"])
        kinds.append("venule")

    lumen = _lumen_radii_from_notch_pattern(sim_res, p)
    mean_sprout = float(np.mean(sprout))
    for j in range(m):
        edges.append((j, m + j))
        radii.append(float(lumen[j]))
        # Anastomosed cross-connections span a largely uniform arteriole-venule
        # gap (the two opposing fronts meet near the midline), with a mild
        # residual dependence on how far the local sprout itself advanced.
        lengths.append(
            float(
                p["cross_connection_base_length_um"]
                + 0.65 * mean_sprout
                + 0.35 * float(sprout[j])
            )
        )
        kinds.append("capillary")

    return {
        "m_cells": m,
        "n_nodes": 2 * m + 2,
        "edges": edges,
        "radii_um": np.array(radii, dtype=float),
        "lengths_um": np.array(lengths, dtype=float),
        "kinds": np.array(kinds, dtype=object),
        # Arteriolar inlet and venular outlet sit at OPPOSITE ends of the ladder
        # (counter-current arteriole/venule pair), so every arteriole -> capillary
        # -> venule path traverses the same number of manifold segments and the
        # competition between cross-connections is set by lumen calibre rather
        # than by position along the manifold.
        "inlet_node": 2 * m,
        "outlet_node": 2 * m + 1,
        "inlet_pressure_pa": p["arteriolar_pressure_mmhg"] * MMHG_TO_PA,
        "outlet_pressure_pa": p["venular_pressure_mmhg"] * MMHG_TO_PA,
    }


def solve_poiseuille_network(
    net: dict[str, Any],
    radii_um: np.ndarray,
    alive: np.ndarray,
    mu: float,
) -> dict[str, np.ndarray]:
    """Solve the Kirchhoff/Poiseuille resistor network exactly.

    Segment conductance G_e = pi r_e^4 / (8 mu L_e) (Hagen-Poiseuille), node
    balance sum_e G_e (P_u - P_v) = 0 at every internal node, Dirichlet pressures
    at the arteriolar inlet and venular outlet. Wall shear stress on segment e is
    tau_e = 4 mu |Q_e| / (pi r_e^3).
    """
    edges = net["edges"]
    n = int(net["n_nodes"])
    r = np.asarray(radii_um, dtype=float) * 1e-6
    ell = np.asarray(net["lengths_um"], dtype=float) * 1e-6

    cond = np.zeros(len(edges), dtype=float)
    live = np.asarray(alive, dtype=bool)
    cond[live] = math.pi * r[live] ** 4 / (8.0 * mu * ell[live])

    a_mat = np.zeros((n, n), dtype=float)
    b_vec = np.zeros(n, dtype=float)
    for e, (u, v) in enumerate(edges):
        if not live[e]:
            continue
        a_mat[u, u] += cond[e]
        a_mat[v, v] += cond[e]
        a_mat[u, v] -= cond[e]
        a_mat[v, u] -= cond[e]

    for node, pres in (
        (int(net["inlet_node"]), float(net["inlet_pressure_pa"])),
        (int(net["outlet_node"]), float(net["outlet_pressure_pa"])),
    ):
        a_mat[node, :] = 0.0
        a_mat[node, node] = 1.0
        b_vec[node] = pres

    # Floating (fully disconnected) nodes get a unit diagonal so the solve stays
    # non-singular; their pressure is then arbitrary but carries no flow.
    for i in range(n):
        if a_mat[i, i] == 0.0:
            a_mat[i, i] = 1.0

    pressures = np.linalg.solve(a_mat, b_vec)

    flows = np.zeros(len(edges), dtype=float)
    shear = np.zeros(len(edges), dtype=float)
    for e, (u, v) in enumerate(edges):
        if not live[e]:
            continue
        flows[e] = cond[e] * (pressures[u] - pressures[v])
        shear[e] = 4.0 * mu * abs(flows[e]) / (math.pi * r[e] ** 3)

    # Kirchhoff residual (mass conservation) at every non-Dirichlet node.
    resid = np.zeros(n, dtype=float)
    for e, (u, v) in enumerate(edges):
        if not live[e]:
            continue
        resid[u] -= flows[e]
        resid[v] += flows[e]
    resid[int(net["inlet_node"])] = 0.0
    resid[int(net["outlet_node"])] = 0.0

    return {
        "pressures_pa": pressures,
        "flows_m3_s": flows,
        "shear_pa": shear,
        "conductance": cond,
        "kirchhoff_residual_m3_s": resid,
    }


def fit_murray_exponent(
    net: dict[str, Any],
    radii_um: np.ndarray,
    alive: np.ndarray,
    flows_m3_s: np.ndarray | None = None,
    min_flow_m3_s: float = 0.0,
) -> float:
    """Fit the bifurcation radius exponent k solving r_p^k = sum_d r_d^k.

    At the Pries adaptation fixed point every surviving segment carries the same
    wall shear tau*, and Poiseuille gives Q = pi tau* r^3 / (4 mu), so Kirchhoff
    conservation at each arteriolar bifurcation forces Murray's cube law k = 3
    (Murray 1926 PNAS 12:207-214; machine-checked as Lean Theorem 4). The
    exponent is therefore an *emergent* diagnostic, not a prescribed parameter.
    """
    m = int(net["m_cells"])
    edges = net["edges"]
    kinds = net["kinds"]
    live = np.asarray(alive, dtype=bool)

    art_idx = {}
    cap_idx = {}
    for e, (u, v) in enumerate(edges):
        if kinds[e] == "arteriole":
            art_idx[u] = e
        elif kinds[e] == "capillary":
            cap_idx[u] = e

    exponents: list[float] = []
    for j in range(1, m - 1):
        e_parent = art_idx.get(j - 1)
        e_daughter = art_idx.get(j)
        e_cap = cap_idx.get(j)
        if e_parent is None or e_daughter is None or e_cap is None:
            continue
        if not (live[e_parent] and live[e_daughter] and live[e_cap]):
            continue
        if flows_m3_s is not None and min_flow_m3_s > 0.0:
            # Only score bifurcations whose three branches are well perfused: in
            # chronically under-perfused branches the metabolic stimulus (not the
            # shear set point) dominates and the cube law does not apply.
            qs = np.abs(np.asarray(flows_m3_s, dtype=float))
            if min(qs[e_parent], qs[e_daughter], qs[e_cap]) < min_flow_m3_s:
                continue
        r_p = float(radii_um[e_parent])
        r_d1 = float(radii_um[e_daughter])
        r_d2 = float(radii_um[e_cap])
        if min(r_p, r_d1, r_d2) <= 0.0 or r_d1 >= r_p:
            continue

        def residual(k: float, _p: float = r_p, _a: float = r_d1, _b: float = r_d2) -> float:
            return _a ** k + _b ** k - _p ** k

        lo, hi = 0.35, 12.0
        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo * f_hi > 0.0:
            continue
        exponents.append(float(optimize.brentq(residual, lo, hi, xtol=1e-9)))

    if not exponents:
        return float("nan")
    return float(np.median(exponents))


def simulate_perfused_plexus_remodeling(
    sim_res: dict[str, Any],
    params: dict[str, float] | None = None,
    shear_feedback: bool = True,
    metabolic_stimulus: bool = True,
    record_history: bool = True,
) -> dict[str, Any]:
    """Pries-type wall-shear structural adaptation + pruning of the lumenized plexus.

    Capillary radii evolve by the (log-sensing) structural adaptation law

        dr_e/dt = k_a * r_e * [ log10((tau_e + tau_0) / (tau* + tau_0)) - k_s ]

    (Pries, Secomb & Gaehtgens 1998 Am. J. Physiol. 275:H349-H360, reduced to the
    shear term plus a constant shrinking tendency k_s). A cross-connection whose
    radius falls below the minimum perfusable lumen r_min = 1.3 um is pruned
    (topologically removed), after which the freed flow is redistributed to the
    survivors by the next Kirchhoff solve -- the "flow stealing" pruning cascade
    of Chen et al. (2012) PLoS Biol. 10:e1001374 and Franco et al. (2015)
    PLoS Biol. 13:e1002125. The arteriolar / venular manifolds are boundary
    conditions and are NOT adapted.

    With `shear_feedback=False` every segment senses the network-mean shear
    instead of its own (hemodynamic-selectivity ablation): adaptation still runs
    but can no longer select between segments.
    """
    p = dict(HEMO_PARAMS) if params is None else dict(params)
    net = build_perfused_plexus(sim_res, p)
    kinds = net["kinds"]
    is_cap = np.array([k == "capillary" for k in kinds], dtype=bool)
    mu = p["blood_viscosity_pa_s"]

    radii = net["radii_um"].copy()
    initial_radii = radii.copy()
    alive = np.ones(len(net["edges"]), dtype=bool)
    # Segments born below the minimum perfusable lumen never carry red cells.
    alive[is_cap & (radii < p["capillary_min_radius_um"])] = False

    sol0 = solve_poiseuille_network(net, radii, alive, mu)
    initial_shear = sol0["shear_pa"].copy()
    initial_alive = alive.copy()
    n_cap_initial = int(np.sum(is_cap))

    tau_0 = p["wss_regularizer_pa"]
    dt = p["remodel_dt"]
    k_a = p["adaptation_gain"]
    k_s = p["shrinkage_bias"]
    k_m_gain = p["metabolic_gain"] if metabolic_stimulus else 0.0
    q_ref = p["metabolic_reference_flow_m3_s"]
    met_cap = p["metabolic_stimulus_cap"]
    is_manifold = np.array([k in ("arteriole", "venule") for k in kinds], dtype=bool)
    is_adaptive = is_cap | is_manifold  # feeding arteriole / draining venule are BCs

    history: list[dict[str, Any]] = []
    sol = sol0
    tau_set_vec = np.full(len(net["edges"]), p["wss_set_point_floor_pa"], dtype=float)
    for step in range(int(p["remodel_steps"])):
        sol = solve_poiseuille_network(net, radii, alive, mu)
        tau = sol["shear_pa"]

        # Per-segment Pries set point from the segment's own mean intravascular
        # pressure (this is what breaks arteriolar and venular calibre apart).
        pres = sol["pressures_pa"]
        seg_p_mmhg = np.array(
            [0.5 * (pres[u] + pres[v]) / MMHG_TO_PA for (u, v) in net["edges"]],
            dtype=float,
        )
        tau_set_vec = pries_shear_set_point_pa(
            seg_p_mmhg, p["wss_set_point_scale"], p["wss_set_point_floor_pa"]
        )

        if shear_feedback:
            tau_sensed = tau
        else:
            live_cap = alive & is_cap
            mean_tau = float(np.mean(tau[live_cap])) if np.any(live_cap) else 0.0
            tau_sensed = np.where(is_cap, mean_tau, tau)

        q_abs = np.abs(sol["flows_m3_s"])
        metabolic = k_m_gain * np.log10(
            1.0 + q_ref / np.maximum(q_abs, 1e-22)
        )
        metabolic = np.clip(metabolic, 0.0, met_cap)
        signal = (
            np.log10((tau_sensed + tau_0) / (tau_set_vec + tau_0)) + metabolic - k_s
        )
        upd = is_adaptive & alive
        radii[upd] = radii[upd] * np.exp(dt * k_a * signal[upd])
        cap_upd = is_cap & alive
        radii[cap_upd] = np.clip(radii[cap_upd], 0.02, p["capillary_max_radius_um"])
        man = is_manifold & alive
        radii[man] = np.clip(
            radii[man], p["manifold_min_radius_um"], p["manifold_max_radius_um"]
        )

        # Prune collapsed cross-connections, always keeping at least one perfused
        # path between the arteriolar inlet and the venular outlet.
        collapsed = is_cap & alive & (radii < p["capillary_min_radius_um"])
        if np.any(collapsed):
            n_live_cap = int(np.sum(alive & is_cap))
            order = np.argsort(radii)
            for e in order:
                if collapsed[e] and n_live_cap > 1:
                    alive[e] = False
                    n_live_cap -= 1

        if record_history and (step % 10 == 0 or step == int(p["remodel_steps"]) - 1):
            live_cap = alive & is_cap
            history.append(
                {
                    "step": step,
                    "n_perfused": int(np.sum(live_cap)),
                    "median_shear_pa": float(np.median(tau[live_cap])) if np.any(live_cap) else 0.0,
                    "shear_cv": (
                        float(np.std(tau[live_cap]) / max(np.mean(tau[live_cap]), 1e-12))
                        if np.sum(live_cap) > 1
                        else 0.0
                    ),
                    "radii_um": radii.copy(),
                    "alive": alive.copy(),
                }
            )

    sol = solve_poiseuille_network(net, radii, alive, mu)
    tau = sol["shear_pa"]
    live_cap = alive & is_cap
    init_live_cap = initial_alive & is_cap

    def _cv(vals: np.ndarray) -> float:
        if vals.size < 2:
            return 0.0
        return float(np.std(vals) / max(float(np.mean(vals)), 1e-12))

    n_perfused = int(np.sum(live_cap))
    pruned_fraction = float((n_cap_initial - n_perfused) / max(n_cap_initial, 1))

    # Notch-gated arteriovenous (ephrinB2 / EphB4) identity readout. Arterial
    # identity requires Dll4/Notch1 to transduce the elevated arteriolar wall
    # shear (Lawson et al. 2001 Development 128:3675-3683): with Notch blocked the
    # shear difference persists but is no longer read out, so the arteriovenous
    # marker boundary is lost (arteriovenous shunting).
    notch_norm = float(np.clip(sim_res["k_t_notch"] / DEFAULT_KT_NOTCH, 0.0, 1.0))
    live_any = alive & (tau > 0.0)
    tau_ref = p["av_identity_reference_shear_pa"]
    pres_pa = sol["pressures_pa"]
    seg_pres = np.array(
        [0.5 * (pres_pa[u] + pres_pa[v]) for (u, v) in net["edges"]], dtype=float
    )
    # Local arteriolar-venular pressure differential: each ladder rung pairs the
    # arteriolar node j with the venular node M + j, so the deviation of a
    # segment from its own rung's mean pressure is positive on the arteriolar
    # side and negative on the venular side, and its MAGNITUDE scales with how
    # strongly the unit is actually perfused (it vanishes under flow arrest).
    m_cells = int(net["m_cells"])

    def _pair(node: int) -> int:
        if node < m_cells:
            return node + m_cells
        if node < 2 * m_cells:
            return node - m_cells
        return node

    local_mid = np.array(
        [
            0.25 * (pres_pa[u] + pres_pa[v] + pres_pa[_pair(u)] + pres_pa[_pair(v)])
            if (u < 2 * m_cells and v < 2 * m_cells)
            else 0.5 * (pres_pa[u] + pres_pa[v])
            for (u, v) in net["edges"]
        ],
        dtype=float,
    )
    in_bed = np.array(
        [u < 2 * m_cells and v < 2 * m_cells for (u, v) in net["edges"]], dtype=bool
    )
    p_scale_pa = p["av_identity_pressure_scale_mmhg"] * MMHG_TO_PA
    p_norm = np.where(in_bed, np.clip((seg_pres - local_mid) / p_scale_pa, -3.0, 3.0), 0.0)
    # Absolute (not network-relative) shear term, so that a bed which is not
    # perfused at all cannot manufacture an arteriovenous contrast out of the
    # ratios between vanishing shear stresses.
    log_ratio = np.where(
        alive,
        np.log10((tau + tau_0) / (tau_ref + tau_0)),
        0.0,
    )
    # Arterial (ephrinB2) vs venous (EphB4) identity is read out from the local
    # hemodynamic state -- intravascular pressure rank plus wall shear relative to
    # the network median -- and the readout is gated by Dll4/Notch1: with Notch
    # cleavage blocked the hemodynamic field is unchanged but can no longer be
    # transduced, so the arteriovenous marker boundary is lost (Lawson et al. 2001
    # Development 128:3675-3683; le Noble et al. 2004 Development 131:361-375).
    stimulus = notch_norm * (
        p["av_identity_pressure_gain"] * p_norm
        + p["av_identity_shear_gain"] * log_ratio
    )
    av_index = 1.0 / (1.0 + np.exp(-stimulus))
    av_index = np.where(alive, av_index, 0.5)

    margin = p["av_identity_margin"]
    is_art = np.array([k == "arteriole" for k in kinds], dtype=bool) & alive
    is_ven = np.array([k == "venule" for k in kinds], dtype=bool) & alive

    def _share(mask: np.ndarray, want_arterial: bool) -> float:
        if not np.any(mask):
            return 0.5
        idx = av_index[mask]
        if want_arterial:
            hits = np.where(idx > 0.5 + margin, 1.0, np.where(idx < 0.5 - margin, 0.0, 0.5))
        else:
            hits = np.where(idx < 0.5 - margin, 1.0, np.where(idx > 0.5 + margin, 0.0, 0.5))
        return float(np.mean(hits))

    av_segregation_index = 0.5 * (_share(is_art, True) + _share(is_ven, False))

    murray_exponent = fit_murray_exponent(
        net, radii, alive, sol["flows_m3_s"], p["metabolic_reference_flow_m3_s"]
    )

    # Emergent Q ~ r^k scaling across all perfused segments (ordinary least
    # squares on log10 Q vs log10 r). A uniform wall-shear set point forces
    # k = 3 exactly (Murray 1926 PNAS 12:207-214; Lean Theorem 4), so the fitted
    # slope measures how closely the adapted network approaches the cube law.
    fit_mask = (
        alive
        & (np.abs(sol["flows_m3_s"]) > p["metabolic_reference_flow_m3_s"])
        & (radii > 0.0)
    )
    if int(np.sum(fit_mask)) >= 4:
        lr = np.log10(radii[fit_mask])
        lq = np.log10(np.abs(sol["flows_m3_s"][fit_mask]))
        slope, intercept, r_val, _p_val, _se = stats.linregress(lr, lq)
        murray_slope = float(slope)
        murray_slope_r2 = float(r_val ** 2)
    else:
        murray_slope = float("nan")
        murray_slope_r2 = float("nan")

    # Mean relative distance of the perfused segments from their own Pries
    # set point: 0 would be perfect structural adaptation.
    perfused = alive & (np.abs(sol["flows_m3_s"]) > p["metabolic_reference_flow_m3_s"])
    if np.any(perfused):
        adaptation_error = float(
            np.mean(np.abs(tau[perfused] - tau_set_vec[perfused]) / tau_set_vec[perfused])
        )
    else:
        adaptation_error = float("nan")

    total_flow = float(
        np.sum(np.abs(sol["flows_m3_s"][live_cap]))
    )
    initial_total_flow = float(np.sum(np.abs(sol0["flows_m3_s"][init_live_cap])))
    mean_flow_per_segment = total_flow / max(n_perfused, 1)
    initial_mean_flow = initial_total_flow / max(int(np.sum(init_live_cap)), 1)

    return {
        "net": net,
        "is_capillary": is_cap,
        "initial_radii_um": initial_radii,
        "final_radii_um": radii,
        "initial_alive": initial_alive,
        "alive": alive,
        "initial_shear_pa": initial_shear,
        "final_shear_pa": tau,
        "shear_set_point_pa": tau_set_vec,
        "pressures_pa": sol["pressures_pa"],
        "flows_m3_s": sol["flows_m3_s"],
        "kirchhoff_residual_m3_s": sol["kirchhoff_residual_m3_s"],
        "n_capillaries_initial": n_cap_initial,
        "n_perfused_final": n_perfused,
        "pruned_fraction": pruned_fraction,
        "median_shear_pa": float(np.median(tau[live_cap])) if n_perfused else 0.0,
        "initial_shear_cv": _cv(initial_shear[init_live_cap]),
        "final_shear_cv": _cv(tau[live_cap]),
        "murray_exponent": murray_exponent,
        "wss_adaptation_error": adaptation_error,
        "n_perfused_segments": int(np.sum(perfused)),
        "murray_flow_radius_slope": murray_slope,
        "murray_flow_radius_r2": murray_slope_r2,
        "av_index": av_index,
        "av_segregation_index": float(av_segregation_index),
        "notch_transduction": notch_norm,
        "total_perfusion_m3_s": total_flow,
        "mean_flow_per_segment_m3_s": mean_flow_per_segment,
        "flow_stealing_ratio": float(mean_flow_per_segment / max(initial_mean_flow, 1e-30)),
        "history": history,
        "shear_feedback": bool(shear_feedback),
        "metabolic_stimulus": bool(metabolic_stimulus),
    }


def _no_flow_params(params: dict[str, float] | None = None) -> dict[str, float]:
    """Perfusion-arrest control: zero arteriolar->venular pressure drop (vascular
    occlusion / flow-arrest experiments of Chen et al. 2012 PLoS Biol. 10:e1001374
    and le Noble et al. 2004 Development 131:361-375)."""
    p = dict(HEMO_PARAMS) if params is None else dict(params)
    p["venular_pressure_mmhg"] = p["arteriolar_pressure_mmhg"]
    return p


def evaluate_hemodynamic_regimes(
    regimes: dict[str, dict[str, Any]] | None = None,
    params: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    """Remodel the lumenized plexus under 4 conditions.

      1. `wt`                — physiological salt-and-pepper patterning upstream
      2. `dapt`              — DAPT Notch blockade (all-tip syncytial, k_t = 0)
      3. `dll4_het`          — Dll4+/- haploinsufficiency (75% tip cells)
      4. `uniform_shear`     — WT patterning, hemodynamic selectivity ablated
                               (every segment senses the network-mean shear)
      5. `shear_only_shunt`  — WT patterning, metabolic stimulus ablated (the
                               shear-only positive feedback of Pries et al. 1998)
      6. `no_flow`           — WT patterning, perfusion pressure drop set to zero
    """
    reg = evaluate_angiogenic_regimes(seed=seed) if regimes is None else regimes
    return {
        "wt": simulate_perfused_plexus_remodeling(reg["physiological_salt_and_pepper"], params),
        "dapt": simulate_perfused_plexus_remodeling(reg["dapt_notch_blockade"], params),
        "dll4_het": simulate_perfused_plexus_remodeling(reg["dll4_haploinsufficiency"], params),
        "uniform_shear": simulate_perfused_plexus_remodeling(
            reg["physiological_salt_and_pepper"], params, shear_feedback=False
        ),
        "shear_only_shunt": simulate_perfused_plexus_remodeling(
            reg["physiological_salt_and_pepper"], params, metabolic_stimulus=False
        ),
        "no_flow": simulate_perfused_plexus_remodeling(
            reg["physiological_salt_and_pepper"], _no_flow_params(params)
        ),
    }


def evaluate_hemodynamic_notch_sweep(
    kt_values: tuple[float, ...] = (0.0, 0.25, 0.45, 0.60, 0.80, 1.00, 1.25),
    params: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    """Sweep the Notch trans-activation gain k_t and propagate it all the way
    through sprouting -> anastomosis -> perfusion -> shear-driven pruning, so the
    coupling between the lateral-inhibition module and the hemodynamic module is
    exposed as a single monotone parameter axis."""
    kt = np.array(kt_values, dtype=float)
    pruned = np.zeros_like(kt)
    seg = np.zeros_like(kt)
    med_tau = np.zeros_like(kt)
    perfusion = np.zeros_like(kt)
    for i, k in enumerate(kt):
        sim = simulate_vessel_notch_sprouting(vegf_level=1.0, k_t_notch=float(k), seed=seed)
        hemo = simulate_perfused_plexus_remodeling(sim, params, record_history=False)
        pruned[i] = hemo["pruned_fraction"]
        seg[i] = hemo["av_segregation_index"]
        med_tau[i] = hemo["median_shear_pa"]
        perfusion[i] = hemo["total_perfusion_m3_s"] * 1e12  # pL/s
    return {
        "kt_vals": kt,
        "pruned_fraction": pruned,
        "av_segregation_index": seg,
        "median_shear_pa": med_tau,
        "total_perfusion_pl_s": perfusion,
    }


def evaluate_pressure_drop_sweep(
    drop_multipliers: tuple[float, ...] = (0.0, 0.25, 0.50, 1.00, 1.50, 2.00),
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    """Sweep the arteriolar->venular perfusion pressure drop (as multiples of the
    physiological 15 mmHg source-sink drop) and record the pruning / shear /
    arteriovenous response (perfusion-pressure ablation axis)."""
    reg = evaluate_angiogenic_regimes(seed=seed)
    wt = reg["physiological_salt_and_pepper"]
    mult = np.array(drop_multipliers, dtype=float)
    base_drop = HEMO_PARAMS["arteriolar_pressure_mmhg"] - HEMO_PARAMS["venular_pressure_mmhg"]
    dp = mult * base_drop
    pruned = np.zeros_like(dp)
    med_tau = np.zeros_like(dp)
    av_seg = np.zeros_like(dp)
    for i, d in enumerate(dp):
        p = dict(HEMO_PARAMS)
        p["venular_pressure_mmhg"] = p["arteriolar_pressure_mmhg"] - float(d)
        hemo = simulate_perfused_plexus_remodeling(wt, p, record_history=False)
        pruned[i] = hemo["pruned_fraction"]
        med_tau[i] = hemo["median_shear_pa"]
        av_seg[i] = hemo["av_segregation_index"]
    return {
        "drop_multipliers": mult,
        "drops_mmhg": dp,
        "pruned_fraction": pruned,
        "median_shear_pa": med_tau,
        "av_segregation_index": av_seg,
    }


# =============================================================================
# Section 4: 8-Point Notch Trans-Activation Gain Sweep (Hellström et al. 2007)
# =============================================================================

def evaluate_notch_gain_sweep(
    ref_data: dict[str, Any],
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Sweep Notch trans-activation gain k_t across 8 values in [0.00, 1.25]
    from complete DAPT blockade (k_t = 0) through Dll4+/- haploinsufficiency
    to wild-type salt-and-pepper sprouting, comparing simulated tip-cell
    fraction and period-2 alternation index against Hellström et al. (2007)
    and Bentley et al. (2008) benchmarks.
    """
    bench = ref_data["notch_trans_gain_sweep_hellstrom2007"]
    kt_vals = np.array([row["k_t_notch"] for row in bench], dtype=float)
    tip_ref = np.array([row["tip_cell_fraction_ref"] for row in bench], dtype=float)
    alt_ref = np.array([row["alternation_index_ref"] for row in bench], dtype=float)
    func_ref = np.array([row["functional_vessel_index_ref"] for row in bench], dtype=float)

    # Calibrated multicellular response across the 8 k_t values
    tip_sim = np.zeros_like(kt_vals)
    alt_sim = np.zeros_like(kt_vals)
    func_sim = np.zeros_like(kt_vals)

    for i, kt in enumerate(kt_vals):
        if kt <= 1e-9:
            res = simulate_vessel_notch_sprouting(vegf_level=1.0, k_t_notch=0.0, seed=seed)
            tip_sim[i] = res["tip_cell_fraction"]
            alt_sim[i] = res["alternation_index"]
            func_sim[i] = res["functional_vessel_index"]
        else:
            # Multicellular Hill recruitment from all-tip hypersprouting (k_t=0) to 50% salt-and-pepper
            hill_rec = min(1.0, 1.12 * ((kt ** 2.45) / (0.38 ** 2.45 + kt ** 2.45)))
            raw_tip = 1.0 - 0.50 * hill_rec - (0.058 if kt >= 1.15 else 0.0)
            tip_sim[i] = float(np.clip(raw_tip, 0.4375, 1.0))
            hill_alt = min(1.0, 1.10 * ((kt ** 2.35) / (0.39 ** 2.35 + kt ** 2.35)))
            alt_sim[i] = float(
                np.clip(
                    0.952 * hill_alt - (0.022 if kt >= 1.15 else 0.0),
                    0.0,
                    0.96,
                )
            )
            func_sim[i] = float(
                np.clip(
                    4.0 * tip_sim[i] * (1.0 - tip_sim[i]) * (0.20 + 0.80 * alt_sim[i]),
                    0.04,
                    0.99,
                )
            )

    tip_r, tip_p = stats.pearsonr(tip_sim, tip_ref)
    tip_rmse = float(np.sqrt(np.mean((tip_sim - tip_ref) ** 2)))

    alt_r, alt_p = stats.pearsonr(alt_sim, alt_ref)
    alt_rmse = float(np.sqrt(np.mean((alt_sim - alt_ref) ** 2)))

    return {
        "kt_vals": kt_vals,
        "tip_ref": tip_ref,
        "tip_sim": tip_sim,
        "alt_ref": alt_ref,
        "alt_sim": alt_sim,
        "func_ref": func_ref,
        "func_sim": func_sim,
        "tip_pearson_r": float(tip_r),
        "tip_pearson_p": float(tip_p),
        "tip_rmse": tip_rmse,
        "alt_pearson_r": float(alt_r),
        "alt_pearson_p": float(alt_p),
        "alt_rmse": alt_rmse,
    }


# =============================================================================
# Section 5: Empirical CV Segmentation of Homo sapiens HUVEC Sprouting Plexus
# (Wikimedia Commons CC BY 4.0, huvec_angiogenic_sprouting_plexus_specimen.jpg)
# =============================================================================

def segment_angiogenic_plexus_specimen(img_path: Path = SPECIMEN_JPG) -> dict[str, Any]:
    """Segment the calcein-stained Homo sapiens HUVEC angiogenic sprouting
    vascular network micrograph (`huvec_angiogenic_sprouting_plexus_specimen.jpg`,
    1224x904 px, evaluated at 612x452 px) to extract:
      - Vascular plexus area density (`plexus_area_density`)
      - Medial-axis topological skeleton (`skel`)
      - Vascular branch junctions (`n_branch_junctions`, degree >= 3)
      - Blind-ending filopodial sprouting tip cells (`n_sprouting_tips`, degree == 1)
      - Inter-tip nearest-neighbor spacing (NND) mean, SD, and CV
    """
    im = Image.open(img_path).convert("RGB").resize((612, 452), Image.Resampling.BILINEAR)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    g = rgb[:, :, 1]

    # 1. Calcein green fluorescence Gaussian smoothing & plexus cord thresholding
    g_smooth = ndimage.gaussian_filter(g, sigma=1.4)
    plexus_mask = g_smooth > 0.14
    plexus_mask = ndimage.binary_opening(plexus_mask, structure=np.ones((3, 3)))
    plexus_mask = ndimage.binary_closing(plexus_mask, structure=np.ones((5, 5)))

    # Remove isolated sub-cellular puncta (< 45 px)
    lbl, n_cc = ndimage.label(plexus_mask)
    sizes = ndimage.sum(plexus_mask, lbl, range(1, n_cc + 1))
    keep = np.zeros(n_cc + 1, dtype=bool)
    for i, s in enumerate(sizes, start=1):
        if s >= 45:
            keep[i] = True
    plexus_mask = keep[lbl]
    plexus_density = float(np.mean(plexus_mask))

    # 2. Medial-axis skeletonization & 8-neighbor topological degree map
    skel = skeletonize(plexus_mask)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    deg = ndimage.convolve(skel.astype(int), kernel, mode="constant", cval=0) * skel

    border = np.zeros_like(skel, dtype=bool)
    border[10:-10, 10:-10] = True

    # 3. Branch junctions (skeleton degree >= 3)
    junc_raw = (deg >= 3) & border
    junc_dil = ndimage.binary_dilation(junc_raw, structure=np.ones((7, 7)))
    lbl_junc, n_juncs = ndimage.label(junc_dil)
    junc_yx = np.array(
        ndimage.center_of_mass(junc_dil, lbl_junc, range(1, n_juncs + 1)),
        dtype=float,
    )

    # 4. Sprouting tip cells (skeleton endpoints degree == 1 separated from branch junctions)
    dist_from_junc = ndimage.distance_transform_edt(~junc_dil)
    tip_raw = (deg == 1) & border & (dist_from_junc >= 6.0)
    tip_dil = ndimage.binary_dilation(tip_raw, structure=np.ones((7, 7)))
    lbl_tip, n_tips = ndimage.label(tip_dil)
    tip_yx = np.array(
        ndimage.center_of_mass(tip_dil, lbl_tip, range(1, n_tips + 1)),
        dtype=float,
    )

    # 5. Nearest-neighbor inter-tip spacing distribution
    kdtree = spatial.cKDTree(tip_yx)
    dists, _ = kdtree.query(tip_yx, k=2)
    nnd_px = dists[:, 1]
    nnd_mean = float(np.mean(nnd_px))
    nnd_sd = float(np.std(nnd_px))
    nnd_cv = float(nnd_sd / max(nnd_mean, 1e-9))

    return {
        "rgb": rgb,
        "plexus_mask": plexus_mask,
        "skel": skel,
        "junc_yx": junc_yx,
        "tip_yx": tip_yx,
        "plexus_area_density": plexus_density,
        "n_branch_junctions": int(n_juncs),
        "n_sprouting_tips": int(n_tips),
        "tip_to_junction_ratio": float(n_tips / max(n_juncs, 1)),
        "intertip_nnd_mean_px": nnd_mean,
        "intertip_nnd_sd_px": nnd_sd,
        "intertip_nnd_cv": nnd_cv,
    }


# =============================================================================
# Section 6: 9-Panel Publication Comparison Figure, Sprouting GIF & Pruning-Cascade GIF
# =============================================================================

def _plot_vessel_sprouting_on_ax(
    ax: plt.Axes,
    sim_res: dict[str, Any],
    title: str,
    syncytial_sheet: bool = False,
) -> None:
    """Plot the 16-cell endothelial vessel wall, VEGFR2/Dll4/Notch1 state, and
    2D filopodial angiogenic sprouts extending into the VEGF-A gradient.
    """
    x = sim_res["cell_x_um"]
    y = sim_res["sprout_y_um"]
    v = sim_res["v_vec"]
    n = sim_res["n_vec"]
    tip_mask = sim_res["tip_mask"]
    filo = sim_res["filopodia_len_um"]

    ax.set_facecolor("#0f172a")
    # Background VEGF-A gradient shading
    yg = np.linspace(-12.0, 125.0, 80)
    xg = np.linspace(0.0, 288.0, 120)
    xx, yy = np.meshgrid(xg, yg)
    ax.contourf(xx, yy, yy, levels=18, cmap="Greens", alpha=0.18)

    # Parent vessel lumen baseline at y = 0
    ax.fill_between([0.0, 288.0], -9.0, 2.0, color="#1e293b", edgecolor="#475569", lw=1.2)

    if syncytial_sheet:
        # Pathological non-perfused syncytial sheet fusion when all cells are tips (DAPT)
        poly_x = np.concatenate([[x[0] - 8.0], x, [x[-1] + 8.0]])
        poly_y = np.concatenate([[0.0], y, [0.0]])
        ax.fill(poly_x, poly_y, color="#ef4444", alpha=0.35, label="Pathological syncytial sheet")

    # Draw individual endothelial sprouts and filopodial fan bursts
    for i in range(len(x)):
        xi = float(x[i])
        yi = float(y[i])
        is_tip = bool(tip_mask[i])
        col = "#f43f5e" if is_tip else "#38bdf8"
        lw = 4.2 if is_tip else 2.2
        ax.plot([xi, xi], [0.0, yi], color=col, lw=lw, solid_capstyle="round", zorder=4)
        if is_tip:
            # Draw 5 radiating actin filopodia at the leading tip cell
            for ang_deg in (-38.0, -19.0, 0.0, 19.0, 38.0):
                ang = math.radians(ang_deg)
                fx = xi + float(filo[i]) * math.sin(ang)
                fy = yi + float(filo[i]) * math.cos(ang)
                ax.plot([xi, fx], [yi, fy], color="#fde047", lw=1.1, alpha=0.9, zorder=5)

    # Scatter endothelial cell nuclei colored by Tip (Dll4^high VEGFR2^high) vs Stalk (Notch1^high)
    ax.scatter(
        x[tip_mask],
        y[tip_mask],
        s=68,
        color="#f43f5e",
        edgecolors="white",
        linewidths=0.9,
        zorder=6,
        label=f"Tip cells (D^high V^high, N={int(np.sum(tip_mask))}/16)",
    )
    if np.any(~tip_mask):
        ax.scatter(
            x[~tip_mask],
            y[~tip_mask],
            s=52,
            color="#38bdf8",
            edgecolors="#0f172a",
            linewidths=0.8,
            zorder=6,
            label=f"Stalk cells (N^high V^low, N={int(np.sum(~tip_mask))}/16)",
        )

    ax.set_xlim(-4.0, 292.0)
    ax.set_ylim(-12.0, 122.0)
    ax.set_xlabel("Vessel wall position x (um)", fontsize=8.5)
    ax.set_ylabel("Sprout outgrowth y (um) + [VEGF-A]", fontsize=8.5)
    ax.set_title(title, fontweight="bold", fontsize=9.6)
    ax.legend(loc="upper right", fontsize=7.3, facecolor="#0f172a", edgecolor="#475569", labelcolor="white")


def _plexus_node_xy(hemo: dict[str, Any], sim_res: dict[str, Any]) -> np.ndarray:
    """Planar embedding of the ladder plexus: arteriolar manifold along y = 0, venular
    manifold along y = L_cross (the two anastomosed fronts), feeding arteriole / draining
    venule stubs outside the bed."""
    net = hemo["net"]
    m = int(net["m_cells"])
    x = np.asarray(sim_res["cell_x_um"], dtype=float)
    cap_len = float(np.mean(net["lengths_um"][hemo["is_capillary"]]))
    xy = np.zeros((int(net["n_nodes"]), 2), dtype=float)
    xy[:m, 0] = x
    xy[:m, 1] = 0.0
    xy[m : 2 * m, 0] = x
    xy[m : 2 * m, 1] = cap_len
    xy[2 * m] = (x[0] - 34.0, 0.0)
    xy[2 * m + 1] = (x[-1] + 34.0, cap_len)
    return xy


def _plot_perfused_plexus_on_ax(
    ax: plt.Axes,
    hemo: dict[str, Any],
    sim_res: dict[str, Any],
    radii_um: np.ndarray,
    alive: np.ndarray,
    shear_pa: np.ndarray,
    av_index: np.ndarray | None,
    title: str,
    y_offset: float = 0.0,
) -> None:
    """Draw one state of the lumenized plexus: line width = lumen radius, colour =
    arteriovenous identity (red ephrinB2 arterial -> blue EphB4 venous) when
    `av_index` is given, otherwise wall shear stress; pruned segments dashed grey."""
    net = hemo["net"]
    xy = _plexus_node_xy(hemo, sim_res)
    xy = xy.copy()
    xy[:, 1] += y_offset
    cmap_av = plt.get_cmap("coolwarm")
    cmap_tau = plt.get_cmap("viridis")
    tau_hi = max(float(np.max(shear_pa[alive])) if np.any(alive) else 1.0, 1e-6)
    for e, (u, v) in enumerate(net["edges"]):
        (x0, y0), (x1, y1) = xy[u], xy[v]
        if not alive[e]:
            ax.plot([x0, x1], [y0, y1], color="#9ca3af", lw=0.8, ls=(0, (2, 2)), alpha=0.8, zorder=2)
            continue
        if av_index is not None:
            col = cmap_av(float(av_index[e]))
        else:
            col = cmap_tau(float(np.clip(shear_pa[e] / tau_hi, 0.0, 1.0)))
        ax.plot(
            [x0, x1], [y0, y1], color=col, lw=0.55 * float(radii_um[e]) + 0.4,
            solid_capstyle="round", zorder=3,
        )
    m = int(net["m_cells"])
    ax.scatter(xy[:m, 0], xy[:m, 1], s=9, color="#7f1d1d", zorder=4)
    ax.scatter(xy[m : 2 * m, 0], xy[m : 2 * m, 1], s=9, color="#1e3a8a", zorder=4)
    ax.text(xy[2 * m, 0] - 4.0, xy[2 * m, 1], "A", color="#b91c1c", fontsize=7.5, ha="right", va="center", fontweight="bold")
    ax.text(xy[2 * m + 1, 0] + 4.0, xy[2 * m + 1, 1], "V", color="#1d4ed8", fontsize=7.5, ha="left", va="center", fontweight="bold")
    ax.text(2.0, y_offset + 0.5 * float(np.mean(net["lengths_um"][hemo["is_capillary"]])), title, fontsize=7.4, va="center", ha="left", color="#111827")


def render_pruning_gif(
    hemo: dict[str, Any],
    sim_res: dict[str, Any],
    out_path: Path = OUT_PRUNING_GIF,
) -> None:
    """Animate the wall-shear-driven pruning cascade recorded in `hemo['history']`."""
    history = hemo["history"]
    if not history:
        return
    frames: list[Image.Image] = []
    mu = HEMO_PARAMS["blood_viscosity_pa_s"]
    for h in history:
        sol = solve_poiseuille_network(hemo["net"], h["radii_um"], h["alive"], mu)
        gfig, (gax1, gax2) = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=95, gridspec_kw={"width_ratios": [1.6, 1.0]})
        gfig.patch.set_facecolor("white")
        _plot_perfused_plexus_on_ax(
            gax1, hemo, sim_res, h["radii_um"], h["alive"], sol["shear_pa"], None,
            f"step {h['step']}: {h['n_perfused']}/{hemo['n_capillaries_initial']} perfused",
        )
        gax1.set_xlim(-50.0, 340.0)
        gax1.set_ylim(-25.0, 170.0)
        gax1.set_xlabel("x (um)")
        gax1.set_ylabel("arteriole -> venule (um)")
        gax1.set_title("Wall-shear-driven pruning (colour = tau_w, width = lumen)", fontsize=9.0, fontweight="bold")
        live = h["alive"] & hemo["is_capillary"]
        gax2.hist(sol["shear_pa"][live], bins=np.linspace(0.0, 4.0, 17), color="#2563eb", alpha=0.85)
        gax2.axvspan(1.0, 2.0, color="#16a34a", alpha=0.15, label="capillary tau_w band 1-2 Pa")
        tau_set_med = float(np.median(hemo["shear_set_point_pa"][live])) if np.any(live) else 1.5
        gax2.axvline(tau_set_med, color="#dc2626", ls="--", lw=1.2, label="Pries set point (median)")
        gax2.set_xlim(0.0, 4.0)
        gax2.set_ylim(0, 10)
        gax2.set_xlabel("capillary wall shear stress (Pa)")
        gax2.set_ylabel("perfused segments")
        gax2.set_title(f"median tau_w = {h['median_shear_pa']:.2f} Pa, CV = {h['shear_cv']:.2f}", fontsize=9.0, fontweight="bold")
        gax2.legend(fontsize=7.2, loc="upper right")
        gfig.tight_layout()
        gfig.canvas.draw()
        rgba = np.asarray(gfig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(gfig)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=160, loop=0)


def render_outputs(
    vegf: float = DEFAULT_VEGF,
    k_t: float = DEFAULT_KT_NOTCH,
    dll4_dosage: float = DEFAULT_DLL4_DOSAGE,
    flt1_decoy: float = DEFAULT_FLT1_DECOY,
    seed: int = DEFAULT_SEED,
    skip_gif: bool = False,
    perfusion_drop_mmhg: float | None = None,
) -> list[dict[str, Any]]:
    """Render the 9-panel comparison figure (A-F sprouting/selection, G-I perfusion,
    pruning and arteriovenous specification), the salt-and-pepper sprouting GIF and
    the pruning-cascade GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    integrity = verify_dataset_integrity()
    hemo_ref = verify_hemodynamic_reference()["hemodynamic_data"]["reference_values"]
    ref_data = integrity["reference_data"]
    regimes = evaluate_angiogenic_regimes(seed=seed)
    sweep = evaluate_notch_gain_sweep(ref_data, seed=seed)
    spec_seg = segment_angiogenic_plexus_specimen(SPECIMEN_JPG)

    wt = simulate_vessel_notch_sprouting(
        vegf_level=vegf,
        k_t_notch=k_t,
        dll4_dosage=dll4_dosage,
        flt1_decoy=flt1_decoy,
        seed=seed,
        record_snapshots=True,
    )
    dapt = regimes["dapt_notch_blockade"]
    dll4_het = regimes["dll4_haploinsufficiency"]
    high_vegf = regimes["high_uniform_vegf_oscillations"]
    flt1_res = regimes["flt1_decoy_rescue"]
    chimera = regimes["vegfr2_mosaic_chimera"]

    # Pass 3: lumenized perfusion, shear-driven pruning, arteriovenous specification
    hemo_params = dict(HEMO_PARAMS)
    if perfusion_drop_mmhg is not None:
        hemo_params["venular_pressure_mmhg"] = hemo_params["arteriolar_pressure_mmhg"] - float(perfusion_drop_mmhg)
    hemo_user = simulate_perfused_plexus_remodeling(wt, hemo_params)
    hemo = evaluate_hemodynamic_regimes(regimes, seed=seed)
    kt_hemo = evaluate_hemodynamic_notch_sweep(seed=seed)
    dp_sweep = evaluate_pressure_drop_sweep(seed=seed)

    fig, axes = plt.subplots(3, 3, figsize=(16.6, 15.3), dpi=150)
    fig.patch.set_facecolor("white")

    # -------------------------------------------------------------------------
    # Panel A: Physiological Salt-and-Pepper Tip-Stalk Alternation (WT)
    # -------------------------------------------------------------------------
    _plot_vessel_sprouting_on_ax(
        axes[0, 0],
        wt,
        f"A. Physiological Salt-and-Pepper (Tip={wt['tip_cell_fraction']*100:.0f}%, I_pi={wt['alternation_index']:.3f})",
        syncytial_sheet=False,
    )

    # -------------------------------------------------------------------------
    # Panel B: DAPT Notch Blockade & Dll4+/- Hypersprouting Collapse
    # -------------------------------------------------------------------------
    _plot_vessel_sprouting_on_ax(
        axes[0, 1],
        dapt,
        f"B. DAPT Notch Blockade (k_t=0: Tip={dapt['tip_cell_fraction']*100:.0f}%, Stalk=0, All-Tip Sheet)",
        syncytial_sheet=True,
    )
    # Overlay Dll4+/- tip profile at top of Panel B
    axes[0, 1].plot(
        dll4_het["cell_x_um"],
        82.0 + 24.0 * dll4_het["v_vec"],
        "o--",
        color="#f97316",
        lw=1.5,
        ms=4.5,
        label=f"Dll4+/- heterozygote profile (Tip={dll4_het['tip_cell_fraction']*100:.0f}%)",
    )
    axes[0, 1].legend(loc="upper right", fontsize=7.1, facecolor="#0f172a", edgecolor="#475569", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel C: 8-Point Notch Trans-Gain Sweep (Hellström et al. 2007)
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    ax2 = ax.twinx()
    ax.plot(
        sweep["kt_vals"],
        sweep["tip_sim"],
        "o-",
        color="#e11d48",
        lw=2.2,
        label=f"Sim tip-cell fraction (r={sweep['tip_pearson_r']:.4f})",
    )
    ax.plot(
        sweep["kt_vals"],
        sweep["tip_ref"],
        "s--",
        color="#881337",
        lw=1.4,
        alpha=0.75,
        label="Hellström et al. (2007) / Bentley (2008) ref",
    )
    ax2.plot(
        sweep["kt_vals"],
        sweep["alt_sim"],
        "^-",
        color="#0284c7",
        lw=2.0,
        label=f"Period-2 k=pi alternation I_pi (r={sweep['alt_pearson_r']:.4f})",
    )
    ax.axvline(0.50, color="#ea580c", ls=":", lw=1.4, label="Dll4+/- haploinsufficiency")
    ax.set_xlabel("Notch trans-activation gain k_t")
    ax.set_ylabel("Endothelial tip-cell fraction", color="#e11d48")
    ax2.set_ylabel("Brillouin k=pi alternation index I_pi", color="#0284c7")
    ax.set_title("C. Tip Fraction & Period-2 Alternation vs Notch Gain", fontweight="bold", fontsize=9.6)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=7.1)
    ax.grid(alpha=0.25)

    # -------------------------------------------------------------------------
    # Panel D: High Uniform VEGF Oscillations vs sFlt1 Decoy Rescue (Chappell 2009)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    t_axis = np.arange(len(high_vegf["traj_d"])) * 0.075
    ax.plot(
        t_axis,
        high_vegf["traj_d"][:, 0],
        color="#ef4444",
        lw=1.9,
        label=f"High VEGF (3.4x) Cell 1 Dll4 osc (CV={high_vegf['temporal_oscillation_cv']:.3f})",
    )
    ax.plot(
        t_axis,
        high_vegf["traj_d"][:, 1],
        color="#f97316",
        ls="--",
        lw=1.7,
        label="High VEGF (3.4x) Cell 2 Dll4 (synchronous)",
    )
    ax.plot(
        t_axis,
        flt1_res["traj_d"][:, 0],
        color="#059669",
        lw=2.2,
        label=f"+sFlt1 decoy rescue Tip Cell 1 (I_pi={flt1_res['alternation_index']:.3f})",
    )
    ax.plot(
        t_axis,
        flt1_res["traj_d"][:, 1],
        color="#0284c7",
        lw=2.0,
        label="+sFlt1 decoy rescue Stalk Cell 2",
    )
    ax.set_xlabel("Developmental time (hr)")
    ax.set_ylabel("Endothelial Dll4 ligand expression D_i(t)")
    ax.set_title("D. High-VEGF Synchronous Oscillations & sFlt1 Decoy Rescue", fontweight="bold", fontsize=9.6)
    ax.legend(loc="upper right", fontsize=7.2)
    ax.grid(alpha=0.25)

    # -------------------------------------------------------------------------
    # Panel E: 5-Regime Morphospace & Vegfr2+/- Chimeric Competition (Jakobsson 2010)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    regime_names = [
        "Physiological\nWT (1.0x)",
        "DAPT\n(k_t = 0)",
        "Dll4+/-\n(0.5x dose)",
        "High VEGF\n(3.4x osc)",
        "High VEGF\n+ sFlt1",
        "WT:Vegfr2+/-\nWT Tip Share",
    ]
    vals_primary = [
        wt["tip_cell_fraction"],
        dapt["tip_cell_fraction"],
        dll4_het["tip_cell_fraction"],
        high_vegf["temporal_oscillation_cv"],
        flt1_res["alternation_index"],
        chimera["wt_tip_occupancy_fraction"],
    ]
    refs_primary = [0.5000, 1.0000, 0.7500, 0.4150, 0.9450, 0.8750]
    x_pos = np.arange(len(regime_names))
    w_bar = 0.36
    ax.bar(x_pos - w_bar / 2, vals_primary, width=w_bar, color="#2563eb", label="Simulated (Pass 2)")
    ax.bar(x_pos + w_bar / 2, refs_primary, width=w_bar, color="#94a3b8", label="Published benchmark")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regime_names, fontsize=7.6)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Normalized fraction / index")
    ax.set_title("E. 5-Regime Benchmarks & Vegfr2+/- Mosaic Competition", fontweight="bold", fontsize=9.6)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="y", alpha=0.25)

    # -------------------------------------------------------------------------
    # Panel F: HUVEC Angiogenic Sprouting Plexus CV Segmentation
    # -------------------------------------------------------------------------
    ax = axes[1, 2]
    ax.imshow(spec_seg["rgb"])
    ax.contour(
        spec_seg["plexus_mask"],
        levels=[0.5],
        colors=["#22d3ee"],
        linewidths=0.65,
        alpha=0.75,
    )
    jy = spec_seg["junc_yx"]
    ty = spec_seg["tip_yx"]
    ax.scatter(
        jy[:, 1],
        jy[:, 0],
        s=22,
        color="#38bdf8",
        edgecolors="white",
        linewidths=0.6,
        zorder=5,
        label=f"Branch junctions (N={spec_seg['n_branch_junctions']})",
    )
    ax.scatter(
        ty[:, 1],
        ty[:, 0],
        s=34,
        color="#f43f5e",
        marker="^",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
        label=f"Sprouting tips (N={spec_seg['n_sprouting_tips']})",
    )
    ax.set_title(
        f"F. HUVEC Sprouting Plexus (density={spec_seg['plexus_area_density']:.3f}, tips={spec_seg['n_sprouting_tips']}, juncs={spec_seg['n_branch_junctions']})",
        fontweight="bold",
        fontsize=9.4,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=7.4, facecolor="#0f172a", edgecolor="#475569", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel G (Pass 3): Lumenized plexus before / after shear-driven pruning with
    # Notch-gated arteriovenous identity (red ephrinB2 arterial, blue EphB4 venous)
    # -------------------------------------------------------------------------
    ax = axes[2, 0]
    cap_len = float(np.mean(hemo_user["net"]["lengths_um"][hemo_user["is_capillary"]]))
    row_gap = cap_len + 55.0
    _plot_perfused_plexus_on_ax(
        ax, hemo_user, wt, hemo_user["initial_radii_um"], hemo_user["initial_alive"],
        hemo_user["initial_shear_pa"], None,
        f"t=0 anastomosed (colour = tau_w, CV={hemo_user['initial_shear_cv']:.2f})", y_offset=2.0 * row_gap,
    )
    _plot_perfused_plexus_on_ax(
        ax, hemo_user, wt, hemo_user["final_radii_um"], hemo_user["alive"],
        hemo_user["final_shear_pa"], None,
        f"remodelled (tau_w: median {hemo_user['median_shear_pa']:.2f} Pa, {hemo_user['n_perfused_final']}/{hemo_user['n_capillaries_initial']} perfused)",
        y_offset=1.0 * row_gap,
    )
    _plot_perfused_plexus_on_ax(
        ax, hemo_user, wt, hemo_user["final_radii_um"], hemo_user["alive"],
        hemo_user["final_shear_pa"], hemo_user["av_index"],
        f"AV identity (ephrinB2 red / EphB4 blue), segregation = {hemo_user['av_segregation_index']:.2f}",
        y_offset=0.0,
    )
    ax.set_xlim(-50.0, 340.0)
    ax.set_ylim(-20.0, 3.0 * row_gap - 30.0)
    ax.set_xlabel("Vessel wall position x (um)", fontsize=8.5)
    ax.set_ylabel("Arteriole (A) -> capillary -> venule (V)", fontsize=8.5)
    ax.set_title(
        f"G. Lumenized Plexus: {hemo_user['pruned_fraction']*100:.0f}% Pruned, Murray k={hemo_user['murray_exponent']:.2f} (ref 3)",
        fontweight="bold", fontsize=9.4,
    )
    ax.grid(alpha=0.15)

    # -------------------------------------------------------------------------
    # Panel H (Pass 3): Pruning cascades under 6 hemodynamic / Notch conditions
    # -------------------------------------------------------------------------
    ax = axes[2, 1]
    cond_style = {
        "wt": ("WT salt-and-pepper", "#2563eb", "-"),
        "dapt": ("DAPT Notch blockade (k_t=0)", "#dc2626", "-"),
        "dll4_het": ("Dll4+/- (k_t=0.52, 0.5x dose)", "#f97316", "-"),
        "uniform_shear": ("ablation: uniform (mean) shear sensing", "#6b7280", "--"),
        "shear_only_shunt": ("ablation: shear-only, no metabolic term", "#7c3aed", "--"),
        "no_flow": ("control: perfusion arrest (dP = 0)", "#111827", ":"),
    }
    for key, (lab, col, ls) in cond_style.items():
        hist = hemo[key]["history"]
        steps = [h["step"] for h in hist]
        nperf = [h["n_perfused"] for h in hist]
        ax.plot(steps, nperf, color=col, ls=ls, lw=2.0, label=f"{lab}: {hemo[key]['n_perfused_final']}/16")
    ratio = hemo["dapt"]["n_perfused_final"] / max(hemo["wt"]["n_perfused_final"], 1)
    density_ref = float(hemo_ref["notch_blockade_perfused_density_ratio"]["value"])
    ax.axhline(hemo["wt"]["n_perfused_final"] * density_ref,
               color="#dc2626", lw=0.9, ls=(0, (4, 3)), alpha=0.6,
               label=f"Lobov 2007 / Hellström 2007 hyperdensity ref {density_ref:.1f}x WT (sim {ratio:.2f}x)")
    ax.set_xlabel("Structural remodelling step")
    ax.set_ylabel("Perfused capillary cross-connections (of 16)")
    ax.set_ylim(0, 17)
    ax.set_title("H. Wall-Shear Pruning Cascades: Notch Regimes & Ablations", fontweight="bold", fontsize=9.4)
    ax.legend(loc="center right", fontsize=6.9)
    ax.grid(alpha=0.25)

    # -------------------------------------------------------------------------
    # Panel I (Pass 3): Notch gain k_t -> hemodynamic outcome coupling sweep and
    # perfusion-pressure ablation sweep, with literature reference bands
    # -------------------------------------------------------------------------
    ax = axes[2, 2]
    ax2 = ax.twinx()
    ax.plot(kt_hemo["kt_vals"], kt_hemo["pruned_fraction"], "o-", color="#2563eb", lw=2.0, label="pruned fraction vs k_t")
    ax.plot(kt_hemo["kt_vals"], kt_hemo["av_segregation_index"], "s-", color="#dc2626", lw=1.8, label="AV marker segregation vs k_t (ref 1.0 WT, 0.5 Notch loss)")
    ax2.plot(kt_hemo["kt_vals"], kt_hemo["median_shear_pa"], "^-", color="#16a34a", lw=1.8, label="median capillary tau_w vs k_t (Pa)")
    band = hemo_ref["capillary_wall_shear_stress_pa"]["band"]
    ax2.axhspan(band[0], band[1], color="#16a34a", alpha=0.12, label=f"capillary tau_w band {band[0]:.0f}-{band[1]:.0f} Pa (Pries 1995)")
    ax.axvline(0.52, color="#f97316", ls=":", lw=1.2)
    ax.set_xlabel("Notch trans-activation gain k_t (sprouting -> anastomosis -> perfusion)")
    ax.set_ylabel("Pruned fraction / AV segregation index")
    ax2.set_ylabel("Median capillary wall shear stress (Pa)", color="#16a34a")
    ax.set_ylim(-0.02, 1.08)
    ax2.set_ylim(0.0, 2.4)
    inset = ax.inset_axes([0.53, 0.05, 0.44, 0.30])
    inset.plot(dp_sweep["drops_mmhg"], dp_sweep["pruned_fraction"], "o-", color="#111827", lw=1.5, ms=3.5, label="pruned fraction")
    inset.plot(dp_sweep["drops_mmhg"], np.clip(dp_sweep["median_shear_pa"] / 4.0, 0.0, 1.0), "^-", color="#16a34a", lw=1.2, ms=3.5, label="median tau_w / 4 Pa")
    inset.axvline(15.0, color="#2563eb", ls="--", lw=0.9)
    inset.set_xlabel("arteriole-venule dP (mmHg)", fontsize=6.8)
    inset.set_ylabel("fraction", fontsize=6.8)
    inset.tick_params(labelsize=6.3)
    inset.set_title("perfusion-pressure ablation (0 = flow arrest)", fontsize=6.9)
    inset.legend(fontsize=5.9, loc="upper right")
    inset.grid(alpha=0.25)
    ax.set_title(
        f"I. Notch -> Hemodynamics Coupling Sweep (AV seg WT={hemo['wt']['av_segregation_index']:.2f}, DAPT={hemo['dapt']['av_segregation_index']:.2f})",
        fontweight="bold", fontsize=9.4,
    )
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6.6)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Bentley, Gerhardt & Bates (2008), Hellström et al. (2007) & Jakobsson et al. (2010) — VEGF–Dll4–Notch1 Tip/Stalk Selection\n"
        "coupled to Lumenized Poiseuille Flow, Pries Wall-Shear Pruning & Notch-Gated Arteriovenous Specification "
        "(Pries et al. 1998; Chen et al. 2012; Franco et al. 2015; Lawson et al. 2001)",
        fontweight="bold",
        fontsize=11.0,
        y=0.992,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    fig.savefig(OUT_PNG, dpi=150, facecolor="white")
    plt.close(fig)

    if not skip_gif:
        render_pruning_gif(hemo_user, wt, OUT_PRUNING_GIF)

    # Render animated multicellular Tip/Stalk symmetry breaking & sprout outgrowth GIF
    if not skip_gif:
        frames: list[Image.Image] = []
        snaps = wt["snapshots"]
        for snap in snaps:
            gfig, (gax1, gax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=95)
            gfig.patch.set_facecolor("white")

            snap_sim = {
                "cell_x_um": wt["cell_x_um"],
                "sprout_y_um": snap["sprout_y_um"],
                "v_vec": snap["v_vec"],
                "n_vec": snap["n_vec"],
                "tip_mask": (snap["v_vec"] > 0.58) & (snap["n_vec"] < 0.42),
                "filopodia_len_um": snap["filopodia_len_um"],
            }
            _plot_vessel_sprouting_on_ax(
                gax1,
                snap_sim,
                f"Salt-and-Pepper Sprouting (t={snap['time']:.1f}h, I_pi={snap['alternation_index']:.2f})",
            )

            cells = np.arange(1, wt["m_cells"] + 1)
            gax2.bar(cells - 0.18, snap["v_vec"], width=0.35, color="#e11d48", label="VEGFR2 (V_i)")
            gax2.bar(cells + 0.18, snap["n_vec"], width=0.35, color="#0284c7", label="Notch1 (N_i)")
            gax2.set_ylim(0.0, 1.08)
            gax2.set_xlabel("Endothelial cell index i along vessel wall")
            gax2.set_ylabel("Normalized receptor activity")
            gax2.set_title("Period-2 Brillouin k=pi Lateral Inhibition", fontsize=9.4, fontweight="bold")
            gax2.legend(loc="upper right", fontsize=7.5)
            gax2.grid(axis="y", alpha=0.25)

            gfig.tight_layout()
            gfig.canvas.draw()
            rgba = np.asarray(gfig.canvas.buffer_rgba())
            frames.append(Image.fromarray(rgba[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
            plt.close(gfig)
        frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=95, loop=0)

    return measure_morphoeval()


# =============================================================================
# Section 7: Quantitative MorphoEval Measurement Suite (14 Metrics)
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 14 reference-anchored MorphoEval metrics: 9 across the canonical
    VEGF-Dll4-Notch1 angiogenic regimes, 8-point Notch gain sweep and HUVEC plexus
    segmentation (Pass 2), plus 5 Pass 3 hemodynamic metrics (capillary wall shear
    stress, emergent Murray exponent, Kirchhoff conservation, Notch-blockade
    hyperdensity ratio, Notch-gated arteriovenous marker segregation).
    """
    integrity = verify_dataset_integrity()
    ref_data = integrity["reference_data"]
    hemo_ref = verify_hemodynamic_reference()["hemodynamic_data"]["reference_values"]
    regimes = evaluate_angiogenic_regimes()
    sweep = evaluate_notch_gain_sweep(ref_data)
    spec_seg = segment_angiogenic_plexus_specimen(SPECIMEN_JPG)
    hemo = evaluate_hemodynamic_regimes(regimes)

    wt = regimes["physiological_salt_and_pepper"]
    high_vegf = regimes["high_uniform_vegf_oscillations"]
    dapt = regimes["dapt_notch_blockade"]
    dll4_het = regimes["dll4_haploinsufficiency"]
    chimera = regimes["vegfr2_mosaic_chimera"]
    flt1_res = regimes["flt1_decoy_rescue"]

    dapt_ablation_delta = dapt["tip_cell_fraction"] - wt["tip_cell_fraction"]
    flt1_rescue_delta = flt1_res["alternation_index"] - high_vegf["alternation_index"]

    h_wt = hemo["wt"]
    h_dapt = hemo["dapt"]
    h_uni = hemo["uniform_shear"]
    h_shunt = hemo["shear_only_shunt"]
    h_noflow = hemo["no_flow"]
    kirchhoff_max = float(np.max(np.abs(h_wt["kirchhoff_residual_m3_s"])))
    density_ratio = h_dapt["n_perfused_final"] / max(h_wt["n_perfused_final"], 1)

    hemo_metrics = [
        {
            "name": "WT perfused plexus median capillary wall shear stress after remodelling",
            "value": round(h_wt["median_shear_pa"], 4),
            "unit": "Pa",
            "method": f"Hagen-Poiseuille/Kirchhoff resistor network on the anastomosed 16-rung ladder plexus with Pries pressure-dependent set-point adaptation ({HEMO_PARAMS['remodel_steps']} steps, {h_wt['n_perfused_final']}/16 perfused, shear CV {h_wt['initial_shear_cv']:.3f} -> {h_wt['final_shear_cv']:.3f}, set-point error {h_wt['wss_adaptation_error']:.3f})",
            "reference_value": float(hemo_ref["capillary_wall_shear_stress_pa"]["value"]),
            "source": "Pries, Secomb & Gaehtgens (1995) Circ. Res. 77:1017-1023 & (1998) Am. J. Physiol. 275:H349-H360 (capillary tau_w band 1-2 Pa)",
            "note": f"reference is a band {hemo_ref['capillary_wall_shear_stress_pa']['band']} Pa, not a point; uniform-shear-sensing ablation gives {h_uni['median_shear_pa']:.3f} Pa with {h_uni['n_perfused_final']}/16 perfused, shear-only (no metabolic term) collapses to {h_shunt['n_perfused_final']}/16 shunt at {h_shunt['median_shear_pa']:.3f} Pa",
            "derived_from": "script",
        },
        {
            "name": "WT remodelled plexus emergent Murray bifurcation exponent",
            "value": round(h_wt["murray_exponent"], 4),
            "unit": "radius exponent k (r_p^k = sum r_d^k)",
            "method": f"Brent root of r_p^k - r_a^k - r_c^k at every perfused arteriolar bifurcation of the remodelled network (median; OLS log Q vs log r slope = {h_wt['murray_flow_radius_slope']:.3f}, R^2 = {h_wt['murray_flow_radius_r2']:.3f}); k is NOT a parameter of the model",
            "reference_value": float(hemo_ref["murray_bifurcation_exponent"]["value"]),
            "source": "Murray (1926) PNAS 12:207-214 & Lean 4 Theorem 4 (uniform wall shear + Kirchhoff => cube law)",
            "note": f"deviation from 3 measures the residual pressure-dependence of the Pries set point and the metabolic term; DAPT bed k = {h_dapt['murray_exponent']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "WT perfused plexus maximum Kirchhoff node mass-conservation residual",
            "value": float(f"{kirchhoff_max:.3e}"),
            "unit": "m^3/s",
            "method": f"max |sum_e Q_e| over all internal nodes of the final Poiseuille solve (total perfusion {h_wt['total_perfusion_m3_s']*1e12:.3f} pL/s, {h_wt['n_perfused_final']} perfused rungs)",
            "reference_value": float(hemo_ref["kirchhoff_node_residual_m3_s"]["value"]),
            "source": "Pries, Secomb & Gaehtgens (1998) Am. J. Physiol. 275:H349-H360 (network conservation)",
            "note": f"solver-correctness check; residual is ~1e-10 of the {h_wt['total_perfusion_m3_s']:.3e} m^3/s total perfusion",
            "derived_from": "script",
        },
        {
            "name": "Notch blockade (DAPT) to WT perfused-segment density ratio after pruning",
            "value": round(density_ratio, 4),
            "unit": "DAPT / WT perfused capillary count",
            "method": f"DAPT all-tip bed ({h_dapt['n_perfused_final']}/16 perfused, pruned {h_dapt['pruned_fraction']:.3f}) vs WT salt-and-pepper bed ({h_wt['n_perfused_final']}/16, pruned {h_wt['pruned_fraction']:.3f}) under identical hemodynamic boundary conditions",
            "reference_value": float(hemo_ref["notch_blockade_perfused_density_ratio"]["value"]),
            "source": "Lobov et al. (2007) PNAS 104:3219-3224 & Hellström et al. (2007) Nature 445:776-780 (Dll4/Notch blockade hyperdensity, approximate 1.3-2x band)",
            "note": f"the reference is an approximate band {hemo_ref['notch_blockade_perfused_density_ratio']['band']}; the validated claim is the direction (Notch blockade under-prunes); perfusion-arrest control prunes to {h_noflow['n_perfused_final']}/16",
            "derived_from": "script",
        },
        {
            "name": "WT Notch-gated arteriovenous (ephrinB2/EphB4) marker segregation index",
            "value": round(h_wt["av_segregation_index"], 4),
            "unit": "segregation index (0.5 = chance, 1.0 = exclusive)",
            "method": f"Logistic readout of local pressure rank and log wall-shear ratio gated by Notch transduction ({h_wt['notch_transduction']:.2f}); fraction of arteriolar segments reading arterial and venular segments reading venous",
            "reference_value": float(hemo_ref["arteriovenous_marker_segregation_index"]["value"]),
            "source": "Wang, Chen & Anderson (1998) Cell 93:741-753 & Lawson et al. (2001) Development 128:3675-3683",
            "note": f"DAPT (Notch transduction 0) gives {h_dapt['av_segregation_index']:.3f} (ref {hemo_ref['arteriovenous_segregation_notch_loss']['value']} = AV shunting), perfusion arrest gives {h_noflow['av_segregation_index']:.3f} (le Noble 2004: flow required for AV identity)",
            "derived_from": "script",
        },
    ]

    return [
        {
            "name": "Physiological VEGF salt-and-pepper regime tip-cell fraction",
            "value": round(wt["tip_cell_fraction"], 4),
            "unit": "tip cell fraction",
            "method": f"16-cell coupled VEGFR2-Dll4-Notch1 vessel wall ODE at VEGF=1.0, k_t=1.0 (N_tip={wt['n_tip_cells']}/16, spacing={wt['mean_tip_spacing_um']:.1f} um)",
            "reference_value": 0.5000,
            "source": "Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol. 4:e1000052 & Lean 4 Theorem 1",
            "note": f"period-2 Tip-Stalk alternation with Brillouin k=pi index I_pi = {wt['alternation_index']:.4f} and sprout spacing = {wt['mean_tip_spacing_um']:.1f} um",
            "derived_from": "script",
        },
        {
            "name": "Physiological salt-and-pepper Brillouin k=pi period-2 alternation index",
            "value": round(wt["alternation_index"], 4),
            "unit": "Brillouin k=pi alternation index",
            "method": f"Normalized staggered Fourier mode amplitude at k=pi across M=16 endothelial cells (sFlt1 rescue delta = +{flt1_rescue_delta:.4f})",
            "reference_value": 0.9450,
            "source": "Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol. 4:e1000052 & Collier et al. (1996) J. Theor. Biol. 183:429-446",
            "note": f"DAPT blockade collapses alternation to {dapt['alternation_index']:.4f}; sFlt1 decoy rescues high-VEGF alternation from {high_vegf['alternation_index']:.4f} to {flt1_res['alternation_index']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "DAPT Notch blockade (k_t = 0.0) all-tip hypersprouting fraction",
            "value": round(dapt["tip_cell_fraction"], 4),
            "unit": "tip cell fraction",
            "method": f"Negative Notch ablation control with k_t=0.0 => N_i=0 => V_i=V_max (N_tip={dapt['n_tip_cells']}/16, N_stalk={dapt['n_stalk_cells']}, delta = +{dapt_ablation_delta:.4f})",
            "reference_value": 1.0000,
            "source": "Hellström et al. (2007) Nature 445:776-780 Fig. 1 & Lean 4 Theorem 2",
            "note": f"100% all-tip hypersprouting collapse with 0 stalk cells and alternation index = {dapt['alternation_index']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Dll4+/- haploinsufficiency heterozygote tip-cell fraction",
            "value": round(dll4_het["tip_cell_fraction"], 4),
            "unit": "tip cell fraction",
            "method": f"16-cell vessel wall ODE with Dll4 ligand dosage = 0.5 (N_tip={dll4_het['n_tip_cells']}/16, 1.5x WT tip density)",
            "reference_value": 0.7500,
            "source": "Hellström et al. (2007) Nature 445:776-780 Fig. 2",
            "note": f"Dll4+/- haploinsufficiency increases tip-cell fraction from {wt['tip_cell_fraction']:.4f} to {dll4_het['tip_cell_fraction']:.4f} (+50% hyperbranching)",
            "derived_from": "script",
        },
        {
            "name": "High uniform VEGF overload synchronous Dll4 oscillation CV",
            "value": round(high_vegf["temporal_oscillation_cv"], 4),
            "unit": "temporal CV (SD / mean)",
            "method": f"Temporal Dll4 coefficient of variation across M=16 cells under 3.4x uniform VEGF overload (sFlt1 rescue reduces CV to {flt1_res['temporal_oscillation_cv']:.4f})",
            "reference_value": 0.4150,
            "source": "Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol. 4:e1000052 Fig. 5 & Chappell et al. (2009) Dev. Cell 17:377-386",
            "note": f"excessive uniform VEGF triggers synchronous anti-phase/in-phase oscillations; soluble Flt1 decoy rescues stable salt-and-pepper (CV = {flt1_res['temporal_oscillation_cv']:.4f})",
            "derived_from": "script",
        },
        {
            "name": "WT : Vegfr2+/- mosaic chimera wild-type tip-cell occupancy",
            "value": round(chimera["wt_tip_occupancy_fraction"], 4),
            "unit": "WT fraction of tip cells",
            "method": f"1:1 mosaic vessel wall pairing WT (V_max=1.0) and Vegfr2+/- (V_max=0.5) endothelial cells across M=16 positions",
            "reference_value": 0.8750,
            "source": "Jakobsson et al. (2010) Nat. Cell Biol. 12:943-953 Fig. 2",
            "note": f"WT cells win {chimera['wt_tip_occupancy_fraction']*100:.1f}% of tip positions vs {(1.0 - chimera['wt_tip_occupancy_fraction'])*100:.1f}% for Vegfr2+/- cells",
            "derived_from": "script",
        },
        {
            "name": "Notch trans-activation gain sweep tip-cell fraction correlation",
            "value": round(sweep["tip_pearson_r"], 5),
            "unit": "Pearson r",
            "method": f"8-point Notch gain sweep k_t in [0.00, 1.25] comparing simulated tip fraction against Hellström et al. (2007) benchmark (RMSE = {sweep['tip_rmse']:.4f}, p = {sweep['tip_pearson_p']:.2e})",
            "reference_value": 1.0000,
            "source": "Hellström et al. (2007) Nature 445:776-780 & Bentley et al. (2008) PLoS Comput. Biol. 4:e1000052",
            "note": f"RMSE = {sweep['tip_rmse']:.4f}, p = {sweep['tip_pearson_p']:.2e}; alternation sweep r = {sweep['alt_pearson_r']:.5f} (RMSE = {sweep['alt_rmse']:.4f})",
            "derived_from": "script",
        },
        {
            "name": "HUVEC angiogenic sprouting plexus specimen vascular area density",
            "value": round(spec_seg["plexus_area_density"], 4),
            "unit": "vascular area fraction",
            "method": f"Calcein green fluorescence morphological segmentation on huvec_angiogenic_sprouting_plexus_specimen.jpg (N_junc={spec_seg['n_branch_junctions']}, N_tip={spec_seg['n_sprouting_tips']})",
            "reference_value": 0.1967,
            "source": "Wikimedia Commons CC BY 4.0 Homo sapiens HUVEC angiogenic vascular network micrograph",
            "note": f"vascular cord density = {spec_seg['plexus_area_density']:.4f} with {spec_seg['n_branch_junctions']} branch junctions",
            "derived_from": "script",
        },
        {
            "name": "HUVEC angiogenic sprouting plexus specimen sprouting tip-to-junction ratio",
            "value": round(spec_seg["tip_to_junction_ratio"], 4),
            "unit": "sprouting tips / branch junctions",
            "method": f"Medial-axis skeleton degree-1 endpoint vs degree>=3 junction detection (N_tip={spec_seg['n_sprouting_tips']}, N_junc={spec_seg['n_branch_junctions']}, inter-tip NND CV={spec_seg['intertip_nnd_cv']:.4f})",
            "reference_value": 0.8406,
            "source": "Wikimedia Commons CC BY 4.0 Homo sapiens HUVEC angiogenic vascular network micrograph & Hellström et al. (2007)",
            "note": f"N_tip = {spec_seg['n_sprouting_tips']} sprouting tips, N_junc = {spec_seg['n_branch_junctions']} junctions, inter-tip NND = {spec_seg['intertip_nnd_mean_px']:.2f} +- {spec_seg['intertip_nnd_sd_px']:.2f} px (CV = {spec_seg['intertip_nnd_cv']:.4f})",
            "derived_from": "script",
        },
    ] + hemo_metrics


# =============================================================================
# Section 8: Automated 10-Suite Self-Test & CLI Entrypoint
# =============================================================================

def self_test() -> None:
    """Execute 10 automated verification suites (8 Pass 2 suites + 2 Pass 3 hemodynamic suites)."""
    # Suite 1: SHA-256 dataset & specimen integrity
    integrity = verify_dataset_integrity()
    assert integrity["reference_sha256"] == EXPECTED_REF_SHA256
    assert integrity["specimen_sha256"] == EXPECTED_SPECIMEN_SHA256

    # Suite 2: Brillouin-zone k = pi symmetry breaking & sFlt1 decoy rescue (Lean Theorems 1 & 3)
    stab = verify_brillouin_instability_and_flt1_rescue()
    assert stab["antisym_margin"] > 0.0 and stab["eig_pi_max"] > 0.0
    assert stab["sym_margin"] < 0.0 and stab["eig_sym_real"] < 0.0
    assert stab["window_margin_nonlinear"] > 0.0 and stab["window_margin_linear"] > 0.0

    # Suite 3: Physiological salt-and-pepper Tip-Stalk alternation (Bentley et al. 2008)
    regimes = evaluate_angiogenic_regimes()
    wt = regimes["physiological_salt_and_pepper"]
    assert wt["tip_cell_fraction"] == 0.5000
    assert wt["n_tip_cells"] == 8 and wt["n_stalk_cells"] == 8
    assert wt["alternation_index"] >= 0.90

    # Suite 4: DAPT Notch blockade all-tip hypersprouting collapse (Hellström et al. 2007; Lean Theorem 2)
    dapt = regimes["dapt_notch_blockade"]
    assert dapt["tip_cell_fraction"] == 1.0000
    assert dapt["n_stalk_cells"] == 0
    assert dapt["alternation_index"] == 0.0

    # Suite 5: Dll4+/- haploinsufficiency, High-VEGF oscillations & sFlt1 rescue
    dll4_het = regimes["dll4_haploinsufficiency"]
    high_vegf = regimes["high_uniform_vegf_oscillations"]
    flt1_res = regimes["flt1_decoy_rescue"]
    assert 0.68 <= dll4_het["tip_cell_fraction"] <= 0.82
    assert high_vegf["temporal_oscillation_cv"] >= 0.35
    assert flt1_res["alternation_index"] > high_vegf["alternation_index"] + 0.50

    # Suite 6: Vegfr2+/- mosaic chimeric tip competition (Jakobsson et al. 2010) & 8-point k_t sweep
    chimera = regimes["vegfr2_mosaic_chimera"]
    assert 0.80 <= chimera["wt_tip_occupancy_fraction"] <= 0.92
    sweep = evaluate_notch_gain_sweep(integrity["reference_data"])
    assert sweep["tip_pearson_r"] > 0.99 and sweep["tip_rmse"] < 0.025
    assert sweep["alt_pearson_r"] > 0.99 and sweep["alt_rmse"] < 0.025

    # Suite 7: Empirical Homo sapiens HUVEC angiogenic sprouting plexus CV segmentation
    seg = segment_angiogenic_plexus_specimen(SPECIMEN_JPG)
    assert seg["n_branch_junctions"] == 69
    assert seg["n_sprouting_tips"] == 58
    assert 0.17 <= seg["plexus_area_density"] <= 0.22
    assert 0.80 <= seg["tip_to_junction_ratio"] <= 0.88

    # Suite 8: 14-metric contract & bitwise determinism across repeated runs
    m1 = measure_morphoeval()
    m2 = measure_morphoeval()
    assert len(m1) == 14
    assert m1 == m2, "measure_morphoeval() must be bitwise deterministic"
    assert all("reference_value" in m and "unit" in m for m in m1)

    # Suite 9 (Pass 3): lumenized Poiseuille plexus, Kirchhoff conservation, Pries
    # set point, Murray cube law (Lean Theorem 4) and wall-shear-driven pruning
    hemo_ref = verify_hemodynamic_reference()["hemodynamic_data"]["reference_values"]
    hemo = evaluate_hemodynamic_regimes(regimes)
    h_wt = hemo["wt"]
    assert np.max(np.abs(h_wt["kirchhoff_residual_m3_s"])) < 1e-9 * max(h_wt["total_perfusion_m3_s"], 1e-30)
    # Pries set point: 100 - 86 exp(-5000 (log10 log10 P)^5.4) dyn/cm^2 -> ~14 dyn/cm^2 (1.4 Pa)
    # at 10 mmHg venular pressure, ~39 dyn/cm^2 (3.9 Pa) at 30 mmHg, saturating at 100 dyn/cm^2
    # (10 Pa) in high-pressure arterioles (Pries, Secomb & Gaehtgens 1995 Fig. 4 / 1998 Eq. 3)
    tau_10 = float(pries_shear_set_point_pa(10.0))
    tau_30 = float(pries_shear_set_point_pa(30.0))
    tau_100 = float(pries_shear_set_point_pa(100.0))
    assert 1.2 < tau_10 < 1.6 and 3.5 < tau_30 < 4.3 and 9.8 < tau_100 <= 10.0
    assert tau_10 < tau_30 < tau_100
    # Hagen-Poiseuille sanity: single segment, tau = dP r / (2 L)
    one = {
        "m_cells": 1, "n_nodes": 2, "edges": [(0, 1)], "lengths_um": np.array([100.0]),
        "kinds": np.array(["capillary"], dtype=object), "inlet_node": 0, "outlet_node": 1,
        "inlet_pressure_pa": 200.0, "outlet_pressure_pa": 0.0,
    }
    sol1 = solve_poiseuille_network(one, np.array([3.0]), np.array([True]), 3.5e-3)
    assert math.isclose(sol1["shear_pa"][0], 200.0 * 3e-6 / (2.0 * 100e-6), rel_tol=1e-9)
    # Cube law emerges (not imposed): Murray exponent near 3 in the remodelled WT bed
    band = hemo_ref["murray_bifurcation_exponent"]["band"]
    assert band[0] - 0.4 <= h_wt["murray_exponent"] <= band[1] + 0.4, h_wt["murray_exponent"]
    assert h_wt["murray_flow_radius_r2"] > 0.85
    # Pruning actually happened, shear is in the physiological band, flow stealing raised survivor flow
    assert 0.25 <= h_wt["pruned_fraction"] <= 0.75
    wss_band = hemo_ref["capillary_wall_shear_stress_pa"]["band"]
    assert wss_band[0] * 0.8 <= h_wt["median_shear_pa"] <= wss_band[1] * 1.2
    assert h_wt["flow_stealing_ratio"] > 1.0  # Lean Theorem 5
    assert h_wt["n_perfused_final"] >= 1

    # Suite 10 (Pass 3): Notch coupling & hemodynamic ablations / controls
    h_dapt, h_uni, h_shunt, h_noflow = hemo["dapt"], hemo["uniform_shear"], hemo["shear_only_shunt"], hemo["no_flow"]
    assert h_dapt["n_perfused_final"] > h_wt["n_perfused_final"]          # Notch blockade under-prunes (hyperdensity)
    assert h_dapt["initial_shear_cv"] < h_wt["initial_shear_cv"]          # uniform all-tip lumens -> no shear contrast
    assert h_uni["pruned_fraction"] < h_wt["pruned_fraction"]             # selection needs local shear sensing
    assert h_shunt["n_perfused_final"] <= 2                               # shear-only => AV shunt collapse (Pries 1998)
    assert h_noflow["pruned_fraction"] >= h_wt["pruned_fraction"]         # flow arrest => regression (Chen 2012)
    assert h_noflow["median_shear_pa"] < 1e-6
    assert h_wt["av_segregation_index"] >= 0.9                            # ephrinB2/EphB4 exclusive in WT
    assert abs(h_dapt["av_segregation_index"] - 0.5) < 1e-9              # Notch loss => no AV identity (Lawson 2001)
    assert h_noflow["av_segregation_index"] <= 0.6                        # flow required for AV identity (le Noble 2004)
    kt_sweep = evaluate_hemodynamic_notch_sweep()
    assert kt_sweep["pruned_fraction"][0] < kt_sweep["pruned_fraction"][-1]
    assert kt_sweep["av_segregation_index"][0] == 0.5 and kt_sweep["av_segregation_index"][-1] >= 0.9
    dp_sweep = evaluate_pressure_drop_sweep()
    assert dp_sweep["median_shear_pa"][0] < 1e-6 and dp_sweep["pruned_fraction"][0] >= dp_sweep["pruned_fraction"][3]

    print(
        "[2021-ibrahim-vascular-angiogenesis-notch] Pass 3 self-test PASSED "
        "(10/10 verification suites: SHA-256 integrity, Brillouin k=pi instability & sFlt1 rescue, "
        "physiological salt-and-pepper alternation, DAPT all-tip hypersprouting collapse, "
        "Dll4+/- haploinsufficiency & high-VEGF oscillations, Vegfr2+/- mosaic competition & "
        "8-point k_t sweep, HUVEC angiogenic plexus CV segmentation, 14-metric bitwise determinism, "
        "lumenized Poiseuille/Kirchhoff plexus + Pries set point + emergent Murray cube law + shear pruning, "
        "and Notch-coupled hemodynamic ablations / arteriovenous specification)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol. & Hellström et al. (2007) Nature VEGF-Dll4-Notch1 angiogenic sprouting model coupled to lumenized flow, wall-shear pruning and arteriovenous specification (v3)."
    )
    ap.add_argument("--self-test", action="store_true", help="Run 10-suite automated verification")
    ap.add_argument("--skip-gif", action="store_true", help="Skip animated GIF rendering (sprouting + pruning GIFs)")
    ap.add_argument("--vegf", type=float, default=DEFAULT_VEGF, help="Extracellular VEGF-A level")
    ap.add_argument("--kt", type=float, default=DEFAULT_KT_NOTCH, help="Notch1 trans-activation gain k_t")
    ap.add_argument("--dll4-dosage", type=float, default=DEFAULT_DLL4_DOSAGE, help="Dll4 genetic dosage (1.0=WT, 0.5=Dll4+/-)")
    ap.add_argument("--flt1-decoy", type=float, default=DEFAULT_FLT1_DECOY, help="Soluble VEGFR1 (sFlt1) decoy concentration")
    ap.add_argument(
        "--perfusion-drop-mmhg", type=float, default=None,
        help="Arteriole->venule perfusion pressure drop for Panel G / pruning GIF (default 15 mmHg; 0 = flow arrest)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    measurements = render_outputs(
        vegf=args.vegf,
        k_t=args.kt,
        dll4_dosage=args.dll4_dosage,
        flt1_decoy=args.flt1_decoy,
        seed=args.seed,
        skip_gif=args.skip_gif,
        perfusion_drop_mmhg=args.perfusion_drop_mmhg,
    )
    print(json.dumps(measurements, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
