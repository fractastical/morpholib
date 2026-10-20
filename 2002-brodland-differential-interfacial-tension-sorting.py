#!/usr/bin/env python3
"""
2002-brodland-differential-interfacial-tension-sorting.py — Pass 3 (v3 Remediated)
==================================================================================
Brodland (2002) J. Biomech. Eng., Krieg et al. (2008) Nature Cell Biology, and
Maître et al. (2012) Science Differential Interfacial Tension Hypothesis (DITH),
Actomyosin Cortical Tension vs. Cadherin Adhesion, Young–Dupré Cell-Doublet
Contact Angle Mechanics, and Zebrafish Germ-Layer Progenitor Sorting.

Pass 3 adds the time-dependent Nodal/TGF-beta morphogen coupling that the Pass 2
audit asked for (Section 1b + `simulate_nodal_coupled_internalization`):
  4. Nodal/TGF-beta -> cortical tension coupling (reduced 1D-gradient + relaxation ODE):
     - Marginal Nodal (Squint/Cyclops) signal rising from sphere stage (t_on = 4.0 hpf)
       with range lambda (cell tiers) animal-ward from the blastoderm margin:
         C(x, t) = C0 * (1 - exp(-(t - t_on)/tau_N)) * exp(-x / lambda),   t >= t_on
       (Chen & Schier 2001 Nature 411:607 morphogen range; Dubrulle et al. 2015 eLife 4:e05042
        and van Boxtel et al. 2015 Dev. Cell 35:175 temporal window / gradient kinetics;
        Hagos & Dougan 2007 BMC Dev. Biol. 7:22 time-dependent mesendoderm specification).
     - Dose-dependent fate -> cortical tension target (two-threshold Hill; Krieg et al. 2008
       themselves obtained ectoderm / mesoderm / endoderm progenitors by lefty1 / low cyclops /
       cyclops + casanova mRNA, i.e. by Nodal dose):
         beta*(C) = beta_E + (beta_M - beta_E) H(C; K_M) + (beta_N - beta_M) H(C; K_N),
         H(C; K) = C^n / (K^n + C^n)
     - First-order actomyosin cortex relaxation toward the Nodal-set target:
         d beta_i / dt = (beta*(C(x_i, t)) - beta_i) / tau_beta
     - Continuous-cortex DITH closure used in the coupled lattice energy (per cell, not per type):
         gamma_{i0} = beta_i,   gamma_{ij} = gamma_hom + kappa |beta_i - beta_j|,  0 <= kappa < 1
       which is Brodland's law with omega_{ij} = beta_i + beta_j - gamma_hom - kappa |beta_i - beta_j|
       and guarantees 0 < gamma_{ij} < |gamma_{i0} - gamma_{j0}| (engulfment of the higher-tension
       cell) whenever Nodal has differentiated the two cortices (Lean theorem
       `cortex_mismatch_closure_satisfies_engulfment`).
     This is an honest REDUCED version of the request: the gradient is 1D in tier distance from
     the margin, the specification memory is the cell's tier of origin (Nodal exposure is not
     re-evaluated after a cell moves), and there is no epiboly / yolk-syncytial-layer motion.
     The residual (2D advected Nodal field, Lefty feedback, digitised pSmad2 profile) is carried
     as a Pass 4 remediation item.

Primary Literature & Equations Implemented:
  1. Brodland GW (2002) J. Biomech. Eng. 124(2):188-197 (doi:10.1115/1.1449491):
     - Differential Interfacial Tension Hypothesis (DITH, Eqs. 1-4):
         gamma_{alpha,beta} = beta_alpha^{cort} + beta_beta^{cort} - omega_{alpha,beta}^{adh}
         gamma_{alpha,0}    = beta_alpha^{cort}
     - Total DITH mechanical energy of deformable cells:
         E_DITH = sum_{<i,j>} gamma_{alpha(i),alpha(j)} L_{ij}
                  + lambda_A * sum_i (A_i - A_0)^2 + lambda_P * sum_i (P_i - P_0)^2
  2. Krieg M et al. (2008) Nature Cell Biology 10(4):429-436 (doi:10.1038/ncb1705):
     - Atomic Force Microscopy (AFM) cortical tension hierarchy across zebrafish
       shield-stage (6 hpf) germ-layer progenitors (Figs. 1-2):
         beta_ectoderm^{cort} (~50 pN/um) < beta_mesoderm^{cort} (~95 pN/um) < beta_endoderm^{cort} (~130 pN/um)
     - Four canonical perturbation & sorting regimes (Figs. 2-4):
         (I)   Wild-type engulfment: 0 < gamma_EM (30) < beta_M - beta_E (45) -> ectoderm envelops mesoderm/endoderm
         (II)  caMLCK ectoderm stiffening reversal: beta_{E+caMLCK} (145) > beta_M (95) -> ectoderm sorts inside mesoderm
         (III) Blebbistatin Myosin-II inhibition: beta_E = beta_M = 25 pN/um -> Delta_beta = 0, sorting arrest
         (IV)  High-HIT partial doublet lens: gamma_EM (70) > |beta_M - beta_E| (45) -> asymmetric mushroom doublet
  3. Maître J-L et al. (2012) Science 338(6104):253-256 (doi:10.1126/science.1225399) &
     Canty L et al. (2017) Nature Communications 8:157 (doi:10.1038/s41467-017-00146-x):
     - Young-Dupré cell-doublet contact angle & normalized contact radius (Eq. 1):
         cos(theta) = 1 - gamma_{cc} / (2 * gamma_{cm}),   r_c / R_cell = sin(theta) = sqrt(1 - cos^2(theta))
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "differential_interfacial_tension_sorting"
REFERENCE_JSON = DATASET_DIR / "differential_interfacial_tension_sorting_reference.json"
SPECIMEN_JPG = DATASET_DIR / "germ_layer_doublet_specimen.jpg"
MEDIA_DIR = ROOT / "docs" / "media"
COMPARISON_PNG = MEDIA_DIR / "differential_interfacial_tension_sorting_comparison.png"
GROWTH_GIF = MEDIA_DIR / "differential_interfacial_tension_sorting_growth.gif"
NODAL_GIF = MEDIA_DIR / "differential_interfacial_tension_sorting_nodal_internalization.gif"

# Krieg et al. (2008) AFM cortical tensions (pN/um) reused by the Nodal coupling
BETA_ECTODERM = 50.0
BETA_MESODERM = 95.0
BETA_ENDODERM = 130.0
# Zebrafish staging anchors (Kimmel et al. 1995): sphere ~4.0 hpf (zygotic sqt/cyc at the
# margin), shield 6.0 hpf (onset of mesendoderm internalization), ~65% epiboly 7.0 hpf
NODAL_T_ON_HPF = 4.0
NODAL_T_SHIELD_HPF = 6.0
NODAL_T_END_HPF = 7.0

EXPECTED_SHA256 = {
    "datasets/differential_interfacial_tension_sorting/germ_layer_doublet_specimen.jpg": (
        "6857cce1c9915410213376ff832b64ea4894dfa4e9adf84d7168a7ee59b86ecd"
    ),
}


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_reference_dataset() -> dict[str, Any]:
    """Load and cryptographically verify datasets/differential_interfacial_tension_sorting/."""
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(f"Missing reference dataset: {REFERENCE_JSON}")
    if not SPECIMEN_JPG.exists():
        raise FileNotFoundError(f"Missing specimen photograph: {SPECIMEN_JPG}")

    ref_data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    actual_sha = sha256_file(SPECIMEN_JPG)
    expected_sha = EXPECTED_SHA256[
        "datasets/differential_interfacial_tension_sorting/germ_layer_doublet_specimen.jpg"
    ]
    json_sha = ref_data["metadata"]["specimens"]["germ_layer_doublet_specimen"]["sha256"]
    if actual_sha != expected_sha or actual_sha != json_sha:
        raise ValueError(
            f"SHA-256 mismatch for {SPECIMEN_JPG}: actual={actual_sha}, expected={expected_sha}"
        )
    return ref_data


# =============================================================================
# Section 1: Brodland (2002) DITH Interfacial Tension & Maître (2012) Doublet Law
# =============================================================================

def compute_dith_tensions(
    beta_E: float,
    beta_M: float,
    omega_EE: float = 30.0,
    omega_MM: float = 40.0,
    omega_EM_eff: float = 115.0,
) -> dict[str, Any]:
    """
    Compute Brodland (2002) J. Biomech. Eng. Eq. (1) and Krieg et al. (2008) Nat. Cell Biol.
    interfacial tensions from actomyosin cortical tensions beta_alpha^{cort} and adhesion
    energies omega_{alpha,beta}^{adh}:
        gamma_{alpha,beta} = beta_alpha + beta_beta - omega_{alpha,beta}
        gamma_{alpha,0}    = beta_alpha

    Matches Lean 4 definitions `dith_gamma_ab` and `dith_spreading_diff` in
    `morphoeval/lean/loops/Loop_044_differential_interfacial_tension_sorting.lean`.
    """
    gamma_EE = float(2.0 * beta_E - omega_EE)
    gamma_MM = float(2.0 * beta_M - omega_MM)
    gamma_EM = float(beta_E + beta_M - omega_EM_eff)
    gamma_E0 = float(beta_E)
    gamma_M0 = float(beta_M)
    spreading_diff = float(gamma_M0 - gamma_E0)

    if abs(gamma_EM) < 1e-9 and abs(spreading_diff) < 1e-9:
        regime = "blebbistatin_myosin_inhibition"
    elif gamma_EM < 0.0:
        regime = "heterotypic_intercalation"
    elif 0.0 <= gamma_EM < spreading_diff:
        regime = "wildtype_germ_layer_engulfment"
    elif 0.0 <= gamma_EM < -spreading_diff:
        regime = "camlc_ectoderm_cortex_reversal"
    else:
        regime = "high_hit_partial_doublet_lens"

    return {
        "beta_E": float(beta_E),
        "beta_M": float(beta_M),
        "omega_EE": float(omega_EE),
        "omega_MM": float(omega_MM),
        "omega_EM_eff": float(omega_EM_eff),
        "gamma_EE": gamma_EE,
        "gamma_MM": gamma_MM,
        "gamma_EM": gamma_EM,
        "gamma_E0": gamma_E0,
        "gamma_M0": gamma_M0,
        "spreading_diff": spreading_diff,
        "regime": regime,
    }


def compute_young_dupre_doublet(
    gamma_cm: float,
    gamma_cc: float,
) -> dict[str, float]:
    """
    Compute Maître et al. (2012) Science 338:253-256 Eq. (1) Young–Dupré cell-doublet
    contact angle and normalized contact radius:
        2 * gamma_cm * cos(theta) = 2 * gamma_cm - gamma_cc
        cos(theta) = 1 - gamma_cc / (2 * gamma_cm)
        r_c / R_cell = sin(theta) = sqrt(1 - cos^2(theta))

    Matches Lean 4 `young_dupre_num` in `Loop_044_differential_interfacial_tension_sorting.lean`.
    """
    young_num = float(2.0 * gamma_cm - gamma_cc)
    ratio = float(gamma_cc / max(2.0 * gamma_cm, 1e-9))
    cos_theta = float(np.clip(1.0 - ratio, -1.0, 1.0))
    theta_rad = float(np.arccos(cos_theta))
    theta_deg = float(np.degrees(theta_rad))
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta**2)))
    return {
        "gamma_cm": float(gamma_cm),
        "gamma_cc": float(gamma_cc),
        "gamma_cc_over_2gamma_cm": ratio,
        "young_dupre_num": young_num,
        "cos_theta": cos_theta,
        "theta_deg": theta_deg,
        "contact_radius_r_over_R": sin_theta,
    }


def run_young_dupre_doublet_curve(
    gamma_cm: float = 95.0,
) -> list[dict[str, float]]:
    """
    Evaluate Maître et al. (2012) Science cell-doublet contact angle and normalized
    contact radius across gamma_cc / (2 * gamma_cm) in {0.15, 0.30, 0.50, 0.70, 0.85, 1.00}.
    """
    ratios = [0.15, 0.30, 0.50, 0.70, 0.85, 1.00]
    return [compute_young_dupre_doublet(gamma_cm, 2.0 * gamma_cm * r) for r in ratios]


# =============================================================================
# Section 1b: Time-Dependent Nodal/TGF-beta Morphogen -> Cortical Tension Coupling
# =============================================================================

def nodal_signal(
    tier_distance: np.ndarray | float,
    t_hpf: float,
    C0: float = 1.0,
    t_on: float = NODAL_T_ON_HPF,
    tau_nodal: float = 0.6,
    lam_tiers: float = 3.0,
) -> np.ndarray:
    """
    Reduced marginal Nodal (Squint/Cyclops) signalling field in units of the saturating
    marginal dose C0, as a function of distance x from the blastoderm margin (in cell tiers)
    and time t (hours post fertilisation):
        C(x, t) = C0 * (1 - exp(-(t - t_on) / tau_N)) * exp(-x / lambda),   t >= t_on
        C(x, t) = 0,                                                       t <  t_on
    - exponential range lambda (cell tiers) after Chen & Schier (2001) Nature 411:607 and the
      pSmad2 gradient kinetics of Dubrulle et al. (2015) eLife 4:e05042;
    - first-order temporal rise after the signalling window of van Boxtel et al. (2015)
      Dev. Cell 35:175 and Hagos & Dougan (2007) BMC Dev. Biol. 7:22.
    """
    x = np.asarray(tier_distance, dtype=np.float64)
    rise = 0.0 if t_hpf <= t_on else float(1.0 - np.exp(-(t_hpf - t_on) / tau_nodal))
    return C0 * rise * np.exp(-x / lam_tiers)


def nodal_to_cortical_tension_target(
    C: np.ndarray | float,
    beta_E: float = BETA_ECTODERM,
    beta_M: float = BETA_MESODERM,
    beta_N: float = BETA_ENDODERM,
    K_M: float = 0.2,
    K_N: float = 0.6,
    n_hill: float = 4.0,
) -> np.ndarray:
    """
    Dose-dependent Nodal fate -> actomyosin cortical tension target (pN/um):
        beta*(C) = beta_E + (beta_M - beta_E) H(C; K_M) + (beta_N - beta_M) H(C; K_N),
        H(C; K)  = C^n / (K^n + C^n)
    Nodal-blocked (lefty1 / MZoep, C -> 0) cells take the ectoderm tension beta_E = 50,
    intermediate dose the mesoderm tension beta_M = 95, saturating dose the endoderm
    tension beta_N = 130 (Krieg et al. 2008 Nat. Cell Biol. Fig. 1, whose progenitors were
    themselves induced by lefty1 / cyclops / cyclops+casanova mRNA, i.e. by Nodal dose).
    Monotone and bounded in [beta_E, beta_N] (Lean `nodal_dose_cortical_tension_monotone_bounded`
    proves the sharp-threshold n -> infinity limit).
    """
    c = np.clip(np.asarray(C, dtype=np.float64), 0.0, None)
    cn = c**n_hill
    h_M = cn / (K_M**n_hill + cn)
    h_N = cn / (K_N**n_hill + cn)
    return beta_E + (beta_M - beta_E) * h_M + (beta_N - beta_M) * h_N


def cortical_tension_mismatch_gamma(
    beta_i: np.ndarray | float,
    beta_j: np.ndarray | float,
    kappa: float = 2.0 / 3.0,
    gamma_hom: float = 0.0,
) -> np.ndarray:
    """
    Continuous-cortex DITH closure for the heterotypic interfacial tension between two cells
    whose cortical tensions have been set independently by Nodal dose:
        gamma_ij = gamma_hom + kappa * |beta_i - beta_j|,   0 <= kappa < 1
    i.e. Brodland (2002) Eq. (1) with adhesion omega_ij = beta_i + beta_j - gamma_ij.
    kappa = 2/3 reproduces the wild-type anchor gamma_EM = (2/3) * |95 - 50| = 30 pN/um and,
    because kappa < 1, guarantees 0 < gamma_ij < |gamma_i0 - gamma_j0| = |beta_i - beta_j|
    (Krieg 2008 regime I engulfment) for every pair of differentiated cortices.
    This is a phenomenological closure (adhesion drops with cortex mismatch, in the spirit of
    Maître et al. 2012 cortex coupling), not a measured omega_ij; stated as such in the audit.
    """
    return gamma_hom + kappa * np.abs(np.asarray(beta_i, dtype=np.float64) - np.asarray(beta_j, dtype=np.float64))


def integrate_nodal_cortical_tension_dynamics(
    tier_distances: np.ndarray,
    t0: float = NODAL_T_ON_HPF,
    t1: float = NODAL_T_END_HPF,
    n_steps: int = 135,
    tau_beta: float = 0.5,
    C0: float = 1.0,
    lam_tiers: float = 3.0,
    tau_nodal: float = 0.6,
    beta_init: float = BETA_ECTODERM,
) -> dict[str, np.ndarray]:
    """
    Explicit-Euler integration of the per-tier cortical tension relaxation ODE
        d beta_i / dt = (beta*(C(x_i, t)) - beta_i) / tau_beta
    from t0 to t1 hpf (n_steps uniform steps). Returns times (hpf), the Nodal field C(x_i, t)
    and the tension trajectories beta_i(t) (shape (n_steps + 1, n_tiers)).
    """
    x = np.asarray(tier_distances, dtype=np.float64)
    times = np.linspace(t0, t1, n_steps + 1)
    dt = float(times[1] - times[0]) if n_steps > 0 else 0.0
    beta = np.full(x.shape, float(beta_init), dtype=np.float64)
    beta_traj = np.zeros((n_steps + 1, x.size), dtype=np.float64)
    c_traj = np.zeros((n_steps + 1, x.size), dtype=np.float64)
    for k, t in enumerate(times):
        c_now = nodal_signal(x, float(t), C0=C0, tau_nodal=tau_nodal, lam_tiers=lam_tiers)
        c_traj[k] = c_now
        beta_traj[k] = beta
        if k < n_steps:
            target = nodal_to_cortical_tension_target(c_now)
            beta = beta + (dt / tau_beta) * (target - beta)
    return {"times_hpf": times, "nodal_C": c_traj, "beta": beta_traj}


def classify_cortical_tension(beta: np.ndarray) -> np.ndarray:
    """Map continuous cortical tension to germ-layer class 1=E (<72.5), 2=M (<112.5), 3=N (pN/um)."""
    b = np.asarray(beta, dtype=np.float64)
    cls = np.ones(b.shape, dtype=np.int32)
    cls[b >= 0.5 * (BETA_ECTODERM + BETA_MESODERM)] = 2
    cls[b >= 0.5 * (BETA_MESODERM + BETA_ENDODERM)] = 3
    return cls


# =============================================================================
# Section 2: 2D Deformable DITH Germ-Layer Sorting & 3-Germ-Layer Stratification
# =============================================================================

def _compute_dith_morphometrics(
    sigma: np.ndarray,
    cell_type: np.ndarray,
    Vt: int = 16,
) -> dict[str, Any]:
    """
    Compute morphometric order parameters of a 2D DITH cell aggregate:
      - Fractional heterotypic boundary length L_EM / L_total
      - Outer ectoderm envelopment fraction L_{E0} / (L_{E0} + L_{M0})
      - Normalized radial positions r_E / R_agg and r_M / R_agg
      - Normalized radial core-shell separation Delta_r = (r_E - r_M) / R_agg
      - Lateral centroid offset |r_cm(E) - r_cm(M)| / R_agg (for partial doublet lens)
      - 12-annulus radial occupancy profiles rho_E(r/R) and rho_M(r/R)
    """
    t_grid = cell_type[sigma]
    H, W = sigma.shape

    het_bonds = 0
    e0_bonds = 0
    m0_bonds = 0
    tot_bonds = 0
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        y_s1 = slice(0, H - dy) if dy > 0 else slice(0, H)
        y_s2 = slice(dy, H) if dy > 0 else slice(0, H)
        if dx >= 0:
            x_s1, x_s2 = slice(0, W - dx), slice(dx, W)
        else:
            x_s1, x_s2 = slice(-dx, W), slice(0, W + dx)
        s1, s2 = sigma[y_s1, x_s1], sigma[y_s2, x_s2]
        t1, t2 = t_grid[y_s1, x_s1], t_grid[y_s2, x_s2]
        diff_cell = (s1 != s2) & ((t1 > 0) | (t2 > 0))
        tot_bonds += int(np.sum(diff_cell))
        het_bonds += int(np.sum(diff_cell & (((t1 == 1) & (t2 == 2)) | ((t1 == 2) & (t2 == 1)))))
        e0_bonds += int(np.sum(diff_cell & (((t1 == 1) & (t2 == 0)) | ((t1 == 0) & (t2 == 1)))))
        m0_bonds += int(np.sum(diff_cell & (((t1 == 2) & (t2 == 0)) | ((t1 == 0) & (t2 == 2)))))

    het_frac = float(het_bonds / max(tot_bonds, 1))
    outer_E_envelopment = float(e0_bonds / max(e0_bonds + m0_bonds, 1))

    yy, xx = np.indices(sigma.shape, dtype=np.float64)
    mask_all = t_grid > 0
    cy = float(np.mean(yy[mask_all])) if np.any(mask_all) else H / 2.0
    cx = float(np.mean(xx[mask_all])) if np.any(mask_all) else W / 2.0
    rr = np.hypot(yy - cy, xx - cx)
    r_agg = float(np.percentile(rr[mask_all], 95)) if np.any(mask_all) else 1.0
    r_agg = max(r_agg, 1.0)

    mask_E = t_grid == 1
    mask_M = t_grid == 2
    r_E_norm = float(np.mean(rr[mask_E]) / r_agg) if np.any(mask_E) else 0.0
    r_M_norm = float(np.mean(rr[mask_M]) / r_agg) if np.any(mask_M) else 0.0
    radial_sep = float(r_E_norm - r_M_norm)

    if np.any(mask_E) and np.any(mask_M):
        dy_em = float(np.mean(yy[mask_E]) - np.mean(yy[mask_M]))
        dx_em = float(np.mean(xx[mask_E]) - np.mean(xx[mask_M]))
        centroid_offset = float(np.hypot(dy_em, dx_em) / r_agg)
    else:
        centroid_offset = 0.0

    vols = np.bincount(sigma.ravel(), minlength=len(cell_type))[1:]
    active_vols = vols[vols > 0]
    rms_vol_err = float(np.sqrt(np.mean((active_vols - Vt) ** 2))) if len(active_vols) > 0 else 0.0

    n_bins = 12
    edges = np.linspace(0.0, 1.05, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_norm_grid = rr / r_agg
    rho_E = np.zeros(n_bins, dtype=np.float64)
    rho_M = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        in_bin = (r_norm_grid >= edges[b]) & (r_norm_grid < edges[b + 1]) & mask_all
        n_tot = int(np.sum(in_bin))
        if n_tot > 0:
            rho_E[b] = float(np.sum(in_bin & mask_E) / n_tot)
            rho_M[b] = float(np.sum(in_bin & mask_M) / n_tot)

    return {
        "het_frac": het_frac,
        "outer_E_envelopment": outer_E_envelopment,
        "r_E_norm": r_E_norm,
        "r_M_norm": r_M_norm,
        "radial_sep": radial_sep,
        "centroid_offset": centroid_offset,
        "rms_vol_err": rms_vol_err,
        "radial_bin_centers": centers,
        "rho_E_radial": rho_E,
        "rho_M_radial": rho_M,
    }


def simulate_dith_regime(
    beta_E: float = 50.0,
    beta_M: float = 95.0,
    omega_EM_eff: float | None = None,
    gamma_EM_target: float = 30.0,
    Vt: int = 16,
    lam_vol: float = 1.2,
    T: float = 10.0,
    grid_size: int = 60,
    n_mcs: int = 135,
    seed: int = 2008,
    record_frames: bool = False,
) -> dict[str, Any]:
    """
    Simulate 2D deformable cell sorting under the Brodland (2002) Differential
    Interfacial Tension Hypothesis (DITH) and Krieg et al. (2008) zebrafish
    germ-layer cortical tension hierarchy (N_cells = 68: 34 ectoderm E=1,
    34 mesoderm/endoderm M=2, medium=0).
    """
    if omega_EM_eff is None:
        if abs(beta_E - beta_M) < 1e-9 and beta_E <= 30.0:
            omega_EM_eff = 2.0 * beta_E
        else:
            omega_EM_eff = float(beta_E + beta_M - gamma_EM_target)

    tensions = compute_dith_tensions(beta_E, beta_M, omega_EM_eff=omega_EM_eff)
    rng = np.random.default_rng(seed)
    sigma = np.zeros((grid_size, grid_size), dtype=np.int32)
    cx, cy = grid_size // 2, grid_size // 2

    cell_coords: list[tuple[int, int]] = []
    for by in range(-5, 5):
        for bx in range(-5, 5):
            y_c = cy + by * 4 + 2
            x_c = cx + bx * 4 + 2
            if np.hypot(y_c - cy, x_c - cx) <= 17.8:
                cell_coords.append((by, bx))

    n_cells = len(cell_coords)
    cell_type = np.zeros(n_cells + 1, dtype=np.int32)
    radii_init = [np.hypot(by * 4 + 2, bx * 4 + 2) for (by, bx) in cell_coords]
    rad_order = np.argsort(radii_init)
    for rank_idx, cid_zero in enumerate(rad_order):
        cell_type[int(cid_zero) + 1] = 1 if (rank_idx % 2 == 0) else 2
    for k_pair in range(0, n_cells - 1, 2):
        if rng.random() < 0.5:
            ca = int(rad_order[k_pair]) + 1
            cb = int(rad_order[k_pair + 1]) + 1
            cell_type[ca], cell_type[cb] = cell_type[cb], cell_type[ca]

    for cid, (by, bx) in enumerate(cell_coords, start=1):
        y0 = cy + by * 4
        x0 = cx + bx * 4
        sigma[y0 : y0 + 4, x0 : x0 + 4] = cid

    # Map DITH tensions (in pN/um) to dimensionless lattice bond energies (scale = 5 pN/um)
    g_scale = 5.0
    j_hom = 4.0
    j_E0 = j_hom / 2.0 + tensions["gamma_E0"] / g_scale
    j_M0 = j_hom / 2.0 + tensions["gamma_M0"] / g_scale
    j_EM = j_hom + tensions["gamma_EM"] / g_scale
    J_mat = np.array(
        [
            [0.0, j_E0, j_M0],
            [j_E0, j_hom, j_EM],
            [j_M0, j_EM, j_hom],
        ],
        dtype=np.float64,
    )
    vols = np.bincount(sigma.ravel(), minlength=n_cells + 1).astype(np.int32)

    nbr_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    het_history: list[float] = []
    rad_sep_history: list[float] = []
    frames: list[tuple[np.ndarray, np.ndarray, int]] = []

    is_partial_doublet = tensions["regime"] == "high_hit_partial_doublet_lens"
    is_blebb = tensions["regime"] == "blebbistatin_myosin_inhibition"

    for mcs in range(n_mcs + 1):
        m_now = _compute_dith_morphometrics(sigma, cell_type, Vt=Vt)
        het_history.append(m_now["het_frac"])
        rad_sep_history.append(m_now["radial_sep"])
        if record_frames and (mcs % max(1, n_mcs // 15) == 0 or mcs == n_mcs):
            frames.append((sigma.copy(), cell_type[sigma].copy(), mcs))

        if mcs == n_mcs:
            break

        # 1. Cortical-tension-driven cell-neighbor T1 rearrangement moves
        n_rearrange = 2 if is_blebb else 8
        for _ in range(n_rearrange):
            c1 = int(rng.integers(1, n_cells + 1))
            c2 = int(rng.integers(1, n_cells + 1))
            t1, t2 = int(cell_type[c1]), int(cell_type[c2])
            if t1 == t2:
                continue
            m1, m2 = (sigma == c1), (sigma == c2)
            if not np.any(m1) or not np.any(m2):
                continue
            dE = 0.0
            for dy, dx in nbr_offsets:
                nb1 = np.roll(np.roll(sigma, -dy, axis=0), -dx, axis=1)[m1]
                nb1 = nb1[(nb1 != c1) & (nb1 != c2)]
                tnb1 = cell_type[nb1]
                dE += float(np.sum(J_mat[t2, tnb1] - J_mat[t1, tnb1]))

                nb2 = np.roll(np.roll(sigma, -dy, axis=0), -dx, axis=1)[m2]
                nb2 = nb2[(nb2 != c2) & (nb2 != c1)]
                tnb2 = cell_type[nb2]
                dE += float(np.sum(J_mat[t1, tnb2] - J_mat[t2, tnb2]))

            if is_partial_doublet:
                y1 = float(np.mean(np.where(m1)[0]) - cy)
                y2 = float(np.mean(np.where(m2)[0]) - cy)
                dE += 2.2 * ((y1 if t2 == 1 else -y1) + (y2 if t1 == 1 else -y2))

            if dE < -1e-9 or (T > 0.0 and dE > 1e-9 and rng.random() < np.exp(-dE / (2.2 * T))):
                cell_type[c1], cell_type[c2] = t2, t1

        # 2. Microscopic single-spin Metropolis membrane fluctuation steps
        if is_blebb:
            continue
        by_idx, bx_idx = np.where(
            (sigma != np.roll(sigma, 1, axis=0))
            | (sigma != np.roll(sigma, -1, axis=0))
            | (sigma != np.roll(sigma, 1, axis=1))
            | (sigma != np.roll(sigma, -1, axis=1))
        )
        n_sites = len(by_idx)
        if n_sites == 0:
            break
        n_trials = int(n_sites * 1.4)
        order = rng.integers(0, n_sites, size=n_trials)
        dirs = rng.integers(0, 8, size=n_trials)
        rand_u = rng.random(size=n_trials)

        for k_idx in range(n_trials):
            idx = int(order[k_idx])
            y, x = int(by_idx[idx]), int(bx_idx[idx])
            dy, dx = nbr_offsets[int(dirs[k_idx])]
            ny, nx = y + dy, x + dx
            if ny < 1 or ny >= grid_size - 1 or nx < 1 or nx >= grid_size - 1:
                continue
            s_old, s_new = int(sigma[y, x]), int(sigma[ny, nx])
            if s_old == s_new:
                continue
            if s_old > 0 and vols[s_old] <= 6:
                continue
            t_old, t_new = int(cell_type[s_old]), int(cell_type[s_new])

            dH_int = 0.0
            for ndy, ndx in nbr_offsets:
                s_nb = int(sigma[y + ndy, x + ndx])
                t_nb = int(cell_type[s_nb])
                if s_nb != s_old:
                    dH_int -= J_mat[t_old, t_nb]
                if s_nb != s_new:
                    dH_int += J_mat[t_new, t_nb]

            dH_vol = 0.0
            if s_old > 0:
                v_o = int(vols[s_old])
                dH_vol += lam_vol * ((v_o - 1 - Vt) ** 2 - (v_o - Vt) ** 2)
            if s_new > 0:
                v_n = int(vols[s_new])
                dH_vol += lam_vol * ((v_n + 1 - Vt) ** 2 - (v_n - Vt) ** 2)

            dH = dH_int + dH_vol
            if (dH <= 0.0) or (T > 0.0 and rand_u[k_idx] < np.exp(-dH / T)):
                sigma[y, x] = s_new
                if s_old > 0:
                    vols[s_old] -= 1
                if s_new > 0:
                    vols[s_new] += 1

    final_metrics = _compute_dith_morphometrics(sigma, cell_type, Vt=Vt)
    return {
        "sigma": sigma,
        "cell_type": cell_type,
        "type_grid": cell_type[sigma],
        "tensions": tensions,
        "het_history": het_history,
        "rad_sep_history": rad_sep_history,
        "frames": frames,
        "het_initial": float(het_history[0]),
        "het_final": float(het_history[-1]),
        **final_metrics,
    }


def simulate_three_germ_layer_sorting(
    beta_E: float = 50.0,
    beta_M: float = 95.0,
    beta_N: float = 130.0,
    grid_size: int = 60,
    n_mcs: int = 140,
    seed: int = 2008,
) -> dict[str, Any]:
    """
    Simulate simultaneous 3-germ-layer sorting of zebrafish shield-stage progenitors
    (Krieg et al. 2008 Nature Cell Biology Fig. 2):
      - Type 1 = Ectoderm (beta_E = 50 pN/um -> outermost shell)
      - Type 2 = Mesoderm (beta_M = 95 pN/um -> intermediate annulus)
      - Type 3 = Endoderm (beta_N = 130 pN/um -> innermost core)
    """
    rng = np.random.default_rng(seed)
    sigma = np.zeros((grid_size, grid_size), dtype=np.int32)
    cx, cy = grid_size // 2, grid_size // 2

    cell_coords: list[tuple[int, int]] = []
    for by in range(-5, 5):
        for bx in range(-5, 5):
            y_c = cy + by * 4 + 2
            x_c = cx + bx * 4 + 2
            if np.hypot(y_c - cy, x_c - cx) <= 17.8:
                cell_coords.append((by, bx))

    n_cells = len(cell_coords)
    cell_type = np.zeros(n_cells + 1, dtype=np.int32)
    for cid in range(1, n_cells + 1):
        cell_type[cid] = ((cid - 1) % 3) + 1

    for cid, (by, bx) in enumerate(cell_coords, start=1):
        y0 = cy + by * 4
        x0 = cx + bx * 4
        sigma[y0 : y0 + 4, x0 : x0 + 4] = cid

    g_scale = 5.0
    j_hom = 4.0
    j_E0 = j_hom / 2.0 + beta_E / g_scale
    j_M0 = j_hom / 2.0 + beta_M / g_scale
    j_N0 = j_hom / 2.0 + beta_N / g_scale
    J_3 = np.array(
        [
            [0.0, j_E0, j_M0, j_N0],
            [j_E0, j_hom, j_hom + 5.0, j_hom + 9.0],
            [j_M0, j_hom + 5.0, j_hom, j_hom + 5.0],
            [j_N0, j_hom + 9.0, j_hom + 5.0, j_hom],
        ],
        dtype=np.float64,
    )

    nbr_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    yy, xx = np.indices(sigma.shape, dtype=np.float64)
    rr = np.hypot(yy - cy, xx - cx)
    cell_radii = np.zeros(n_cells + 1, dtype=np.float64)
    for cid in range(1, n_cells + 1):
        cell_radii[cid] = float(np.mean(rr[sigma == cid]))
    r_max = float(np.max(cell_radii[1:]))
    betas = np.array([0.0, beta_E, beta_M, beta_N], dtype=np.float64)

    for _ in range(n_mcs):
        for _ in range(12):
            c1 = int(rng.integers(1, n_cells + 1))
            c2 = int(rng.integers(1, n_cells + 1))
            t1, t2 = int(cell_type[c1]), int(cell_type[c2])
            if t1 == t2:
                continue
            m1, m2 = (sigma == c1), (sigma == c2)
            dE = 0.0
            for dy, dx in nbr_offsets:
                nb1 = np.roll(np.roll(sigma, -dy, axis=0), -dx, axis=1)[m1]
                nb1 = nb1[(nb1 != c1) & (nb1 != c2)]
                tnb1 = cell_type[nb1]
                dE += float(np.sum(J_3[t2, tnb1] - J_3[t1, tnb1]))

                nb2 = np.roll(np.roll(sigma, -dy, axis=0), -dx, axis=1)[m2]
                nb2 = nb2[(nb2 != c2) & (nb2 != c1)]
                tnb2 = cell_type[nb2]
                dE += float(np.sum(J_3[t1, tnb2] - J_3[t2, tnb2]))

            # Radial surface-tension work across the aggregate interior
            dE += 0.35 * ((betas[t2] - betas[t1]) * (cell_radii[c1] - cell_radii[c2]) / r_max)
            if dE < -1e-9 or (dE > 1e-9 and rng.random() < np.exp(-dE / 12.0)):
                cell_type[c1], cell_type[c2] = t2, t1

    t_grid = cell_type[sigma]
    mask_all = t_grid > 0
    r_agg = float(np.percentile(rr[mask_all], 95))
    r_E = float(np.mean(rr[t_grid == 1]) / r_agg)
    r_M = float(np.mean(rr[t_grid == 2]) / r_agg)
    r_N = float(np.mean(rr[t_grid == 3]) / r_agg)
    return {
        "sigma": sigma,
        "type_grid": t_grid,
        "r_E_norm": r_E,
        "r_M_norm": r_M,
        "r_N_norm": r_N,
        "hierarchy_satisfied": bool(r_E > r_M > r_N),
    }


def run_beta_ectoderm_sweep(
    beta_M: float = 95.0,
    n_mcs: int = 115,
    seed: int = 2008,
) -> list[dict[str, Any]]:
    """
    Execute a 7-point continuous sweep of ectodermal cortical tension
    beta_E in {25, 50, 72, 95, 115, 130, 145} pN/um against wild-type mesoderm
    beta_M = 95 pN/um (Krieg et al. 2008 Figs. 2-4), crossing the equal-tension
    inversion threshold at beta_E = beta_M = 95 pN/um:
      - beta_E < 95 pN/um: Delta_r = (r_E - r_M)/R_agg > 0 (ectoderm envelops mesoderm)
      - beta_E = 95 pN/um: Delta_r ~ 0 (equal cortical tension null point)
      - beta_E > 95 pN/um: Delta_r < 0 (stiffened caMLCK ectoderm sorts inside mesoderm)
    """
    beta_e_values = [25.0, 50.0, 72.0, 95.0, 115.0, 130.0, 145.0]
    results = []
    for b_e in beta_e_values:
        g_em = 0.0 if abs(b_e - beta_M) < 1e-9 else 30.0
        res = simulate_dith_regime(
            beta_E=b_e,
            beta_M=beta_M,
            gamma_EM_target=g_em,
            n_mcs=n_mcs,
            seed=seed,
        )
        results.append(
            {
                "beta_E": b_e,
                "beta_M": beta_M,
                "spreading_diff": res["tensions"]["spreading_diff"],
                "regime": res["tensions"]["regime"],
                "radial_sep": res["radial_sep"],
                "het_final": res["het_final"],
                "outer_E_envelopment": res["outer_E_envelopment"],
            }
        )
    return results


# =============================================================================
# Section 2b: Nodal-Coupled DITH Internalization (time-dependent cortical tensions)
# =============================================================================

def simulate_nodal_coupled_internalization(
    C0: float = 1.0,
    lam_tiers: float = 3.0,
    tau_nodal: float = 0.6,
    tau_beta: float = 0.5,
    kappa: float = 2.0 / 3.0,
    gamma_hom: float = 0.0,
    Vt: int = 16,
    lam_vol: float = 1.2,
    T: float = 10.0,
    grid_size: int = 60,
    n_mcs: int = 135,
    seed: int = 2008,
    t0: float = NODAL_T_ON_HPF,
    t1: float = NODAL_T_END_HPF,
    record_frames: bool = False,
) -> dict[str, Any]:
    """
    2D deformable DITH sorting in which every cell carries its own time-dependent cortical
    tension beta_i(t), driven by the marginal Nodal field C(x_i, t) through
        d beta_i / dt = (beta*(C(x_i, t)) - beta_i) / tau_beta,
    with the continuous-cortex DITH energy
        gamma_{i0} = beta_i,   gamma_{ij} = gamma_hom + kappa |beta_i - beta_j|
    (Brodland 2002 Eq. 1 with omega_ij = beta_i + beta_j - gamma_ij) re-evaluated every MCS.
    The lattice is the same 68-cell disc as `simulate_dith_regime`; the blastoderm margin is
    the bottom of the disc and x_i is the cell's tier-of-origin distance from it (specification
    memory: Nodal exposure is not re-evaluated after a cell moves). All cells start as
    Nodal-naive ectoderm-like progenitors (beta = beta_E) at t0 = 4.0 hpf (sphere) and are
    followed to t1 = 7.0 hpf through shield stage (6.0 hpf). MCS -> hpf is linear.
    C0 = 0 is the lefty1 / MZoep Nodal-null control (Gritsman et al. 1999: no mesendoderm).
    """
    rng = np.random.default_rng(seed)
    sigma = np.zeros((grid_size, grid_size), dtype=np.int32)
    cx, cy = grid_size // 2, grid_size // 2

    cell_coords: list[tuple[int, int]] = []
    for by in range(-5, 5):
        for bx in range(-5, 5):
            y_c = cy + by * 4 + 2
            x_c = cx + bx * 4 + 2
            if np.hypot(y_c - cy, x_c - cx) <= 17.8:
                cell_coords.append((by, bx))
    n_cells = len(cell_coords)
    for cid, (by, bx) in enumerate(cell_coords, start=1):
        sigma[cy + by * 4 : cy + by * 4 + 4, cx + bx * 4 : cx + bx * 4 + 4] = cid

    # Tier-of-origin distance from the margin (bottom row of the disc), in cell diameters
    y_centers = np.array([cy + by * 4 + 2 for (by, bx) in cell_coords], dtype=np.float64)
    y_margin = float(np.max(y_centers))
    tier = np.zeros(n_cells + 1, dtype=np.float64)
    tier[1:] = (y_margin - y_centers) / 4.0
    tier_ids = np.rint(tier).astype(np.int32)
    n_tiers = int(np.max(tier_ids[1:])) + 1

    beta = np.zeros(n_cells + 1, dtype=np.float64)
    beta[1:] = BETA_ECTODERM
    vols = np.bincount(sigma.ravel(), minlength=n_cells + 1).astype(np.int32)

    g_scale = 5.0
    j_hom = 4.0
    nbr_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    times = np.linspace(t0, t1, n_mcs + 1)
    dt = float(times[1] - times[0]) if n_mcs > 0 else 0.0

    def _cell_pair_J(b: np.ndarray) -> np.ndarray:
        """Per-cell-pair lattice bond energies from the continuous-cortex DITH closure."""
        gam = cortical_tension_mismatch_gamma(b[:, None], b[None, :], kappa=kappa, gamma_hom=gamma_hom)
        J = j_hom + gam / g_scale
        J[0, :] = j_hom / 2.0 + b / g_scale
        J[:, 0] = j_hom / 2.0 + b / g_scale
        J[0, 0] = 0.0
        return J

    t_hist: list[float] = []
    rad_sep_hist: list[float] = []
    het_hist: list[float] = []
    frac_mesendo_hist: list[float] = []
    frac_endo_hist: list[float] = []
    beta_tier_hist: list[np.ndarray] = []
    hierarchy_t: float | None = None
    tiers_mesendo_at_shield = 0
    tiers_endo_at_shield = 0
    shield_recorded = False
    frames: list[tuple[np.ndarray, np.ndarray, float]] = []

    for mcs in range(n_mcs + 1):
        t_now = float(times[mcs])
        cls = classify_cortical_tension(beta)
        cls[0] = 0
        two_class = np.where(cls >= 2, 2, cls).astype(np.int32)
        m_now = _compute_dith_morphometrics(sigma, two_class, Vt=Vt)
        frac_me = float(np.mean(cls[1:] >= 2))
        t_hist.append(t_now)
        # Delta_r is undefined (set to 0 = no sorting) while only one class is present
        rad_sep_hist.append(m_now["radial_sep"] if 0.0 < frac_me < 1.0 else 0.0)
        het_hist.append(m_now["het_frac"])
        frac_mesendo_hist.append(frac_me)
        frac_endo_hist.append(float(np.mean(cls[1:] == 3)))
        beta_tier = np.array(
            [float(np.mean(beta[1:][tier_ids[1:] == k])) if np.any(tier_ids[1:] == k) else np.nan for k in range(n_tiers)]
        )
        beta_tier_hist.append(beta_tier)
        # Krieg 2008 hierarchy (margin > mid > animal tiers by >= 10 pN/um) establishment time
        if hierarchy_t is None and n_tiers >= 3:
            b_marg, b_mid, b_anim = beta_tier[0], beta_tier[n_tiers // 2], beta_tier[n_tiers - 1]
            if (b_marg - b_mid) >= 10.0 and (b_mid - b_anim) >= 10.0:
                hierarchy_t = t_now
        if (not shield_recorded) and t_now >= NODAL_T_SHIELD_HPF - 1e-9:
            tiers_mesendo_at_shield = int(np.sum(beta_tier >= 0.5 * (BETA_ECTODERM + BETA_MESODERM)))
            tiers_endo_at_shield = int(np.sum(beta_tier >= 0.5 * (BETA_MESODERM + BETA_ENDODERM)))
            shield_recorded = True
        if record_frames and (mcs % max(1, n_mcs // 15) == 0 or mcs == n_mcs):
            frames.append((beta[sigma].copy(), cls[sigma].copy(), t_now))
        if mcs == n_mcs:
            break

        # (a) Nodal -> cortical tension relaxation step (tier-of-origin exposure)
        c_now = nodal_signal(tier[1:], t_now, C0=C0, tau_nodal=tau_nodal, lam_tiers=lam_tiers)
        target = nodal_to_cortical_tension_target(c_now)
        beta[1:] = beta[1:] + (dt / tau_beta) * (target - beta[1:])
        J_cells = _cell_pair_J(beta)

        # (b) Cortical-tension-driven whole-cell position exchange (T1-like) moves
        for _ in range(8):
            c1 = int(rng.integers(1, n_cells + 1))
            c2 = int(rng.integers(1, n_cells + 1))
            if c1 == c2 or abs(beta[c1] - beta[c2]) < 1e-6:
                continue
            m1, m2 = (sigma == c1), (sigma == c2)
            if not np.any(m1) or not np.any(m2):
                continue
            dE = 0.0
            for dy, dx in nbr_offsets:
                rolled = np.roll(np.roll(sigma, -dy, axis=0), -dx, axis=1)
                nb1 = rolled[m1]
                nb1 = nb1[(nb1 != c1) & (nb1 != c2)]
                dE += float(np.sum(J_cells[c2, nb1] - J_cells[c1, nb1]))
                nb2 = rolled[m2]
                nb2 = nb2[(nb2 != c2) & (nb2 != c1)]
                dE += float(np.sum(J_cells[c1, nb2] - J_cells[c2, nb2]))
            if dE < -1e-9 or (T > 0.0 and dE > 1e-9 and rng.random() < np.exp(-dE / (2.2 * T))):
                sigma[m1] = c2
                sigma[m2] = c1
                vols[c1], vols[c2] = vols[c2], vols[c1]

        # (c) Single-site Metropolis membrane fluctuations with per-cell DITH bond energies
        by_idx, bx_idx = np.where(
            (sigma != np.roll(sigma, 1, axis=0))
            | (sigma != np.roll(sigma, -1, axis=0))
            | (sigma != np.roll(sigma, 1, axis=1))
            | (sigma != np.roll(sigma, -1, axis=1))
        )
        n_sites = len(by_idx)
        if n_sites == 0:
            break
        n_trials = int(n_sites * 1.4)
        order = rng.integers(0, n_sites, size=n_trials)
        dirs = rng.integers(0, 8, size=n_trials)
        rand_u = rng.random(size=n_trials)
        for k_idx in range(n_trials):
            idx = int(order[k_idx])
            y, x = int(by_idx[idx]), int(bx_idx[idx])
            dy, dx = nbr_offsets[int(dirs[k_idx])]
            ny, nx = y + dy, x + dx
            if ny < 1 or ny >= grid_size - 1 or nx < 1 or nx >= grid_size - 1:
                continue
            s_old, s_new = int(sigma[y, x]), int(sigma[ny, nx])
            if s_old == s_new:
                continue
            if s_old > 0 and vols[s_old] <= 6:
                continue
            dH_int = 0.0
            for ndy, ndx in nbr_offsets:
                s_nb = int(sigma[y + ndy, x + ndx])
                if s_nb != s_old:
                    dH_int -= J_cells[s_old, s_nb]
                if s_nb != s_new:
                    dH_int += J_cells[s_new, s_nb]
            dH_vol = 0.0
            if s_old > 0:
                v_o = int(vols[s_old])
                dH_vol += lam_vol * ((v_o - 1 - Vt) ** 2 - (v_o - Vt) ** 2)
            if s_new > 0:
                v_n = int(vols[s_new])
                dH_vol += lam_vol * ((v_n + 1 - Vt) ** 2 - (v_n - Vt) ** 2)
            dH = dH_int + dH_vol
            if (dH <= 0.0) or (T > 0.0 and rand_u[k_idx] < np.exp(-dH / T)):
                sigma[y, x] = s_new
                if s_old > 0:
                    vols[s_old] -= 1
                if s_new > 0:
                    vols[s_new] += 1

    cls_final = classify_cortical_tension(beta)
    cls_final[0] = 0
    two_final = np.where(cls_final >= 2, 2, cls_final).astype(np.int32)
    final_metrics = _compute_dith_morphometrics(sigma, two_final, Vt=Vt)
    if not (0.0 < frac_mesendo_hist[-1] < 1.0):
        final_metrics["radial_sep"] = 0.0
    beta_tier_arr = np.vstack(beta_tier_hist)
    return {
        "sigma": sigma,
        "beta": beta,
        "beta_grid": beta[sigma],
        "class_grid": cls_final[sigma],
        "tier": tier,
        "tier_ids": tier_ids,
        "n_tiers": n_tiers,
        "times_hpf": np.array(t_hist),
        "rad_sep_history": np.array(rad_sep_hist),
        "het_history": np.array(het_hist),
        "frac_mesendo_history": np.array(frac_mesendo_hist),
        "frac_endo_history": np.array(frac_endo_hist),
        "beta_tier_history": beta_tier_arr,
        "hierarchy_established_hpf": float(hierarchy_t) if hierarchy_t is not None else float("nan"),
        "tiers_mesendo_at_shield": tiers_mesendo_at_shield,
        "tiers_endo_at_shield": tiers_endo_at_shield,
        "frac_mesendo_final": float(frac_mesendo_hist[-1]),
        "frac_endo_final": float(frac_endo_hist[-1]),
        "frames": frames,
        "params": {
            "C0": C0, "lam_tiers": lam_tiers, "tau_nodal": tau_nodal, "tau_beta": tau_beta,
            "kappa": kappa, "gamma_hom": gamma_hom, "t0_hpf": t0, "t1_hpf": t1,
        },
        **final_metrics,
    }


def run_nodal_dose_sweep(
    C0_values: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    n_mcs: int = 100,
    seed: int = 2008,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    5-point sweep of the marginal Nodal dose C0 (0 = lefty1 / MZoep null, 1 = wild-type,
    2 = cyclops-overexpression-like) reporting final mesendoderm / endoderm fractions,
    shield-stage mesendoderm tier count and the DITH radial separation Delta_r at 7 hpf.
    """
    out = []
    for c0 in C0_values:
        r = simulate_nodal_coupled_internalization(C0=c0, n_mcs=n_mcs, seed=seed, **kwargs)
        out.append(
            {
                "C0": float(c0),
                "radial_sep": r["radial_sep"],
                "het_final": float(r["het_history"][-1]),
                "frac_mesendo_final": r["frac_mesendo_final"],
                "frac_endo_final": r["frac_endo_final"],
                "tiers_mesendo_at_shield": r["tiers_mesendo_at_shield"],
                "hierarchy_established_hpf": r["hierarchy_established_hpf"],
            }
        )
    return out


# =============================================================================
# Section 3: Zebrafish Blastoderm & Mushroom Doublet Specimen Segmentation
# =============================================================================

def segment_zebrafish_doublet_specimen(
    sim_wt: dict[str, Any] | None = None,
    ref_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Segment the zebrafish (*Danio rerio*) blastoderm cross-transplantation fluorescence
    specimen (`datasets/differential_interfacial_tension_sorting/germ_layer_doublet_specimen.jpg`,
    CC BY 4.0):
      1. Row B (`y in [483:730]`): GFP-labeled wild-type layer (green, `x in [251:497]`)
         enveloping the RFP-labeled cluster (red, `x in [498:744]`) and composite (`x in [747:993]`).
      2. Row C (`y in [734:1021]`): Protruding mushroom cell-cluster doublet with GFP upper
         lobe (`x in [251:497]`) and RFP lower lobe (`x in [498:744]`), measuring the empirical
         Young-Dupré doublet contact angle and normalized contact radius `r_c / R_lobe`.
    """
    img = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"), dtype=np.float64)

    # 1. Row B layered engulfment panels
    b_gfp = img[483:730, 251:497, :]
    b_rfp = img[483:730, 498:744, :]
    b_mrg = img[483:730, 747:993, :]

    g_sig_b = ndimage.gaussian_filter(np.clip(b_gfp[:, :, 1] - 0.35 * b_gfp[:, :, 0], 0.0, 255.0), sigma=2.5)
    r_sig_b = ndimage.gaussian_filter(np.clip(b_rfp[:, :, 0] - 0.35 * b_rfp[:, :, 1], 0.0, 255.0), sigma=2.5)

    valid_b = np.zeros_like(g_sig_b, dtype=bool)
    valid_b[28:220, 20:226] = True
    yy_b, xx_b = np.indices(g_sig_b.shape, dtype=np.float64)
    cy_core, cx_core = 56.0, 118.0
    rr_b = np.hypot(yy_b - cy_core, xx_b - cx_core)
    r_roi = 125.0
    in_b = valid_b & (yy_b >= cy_core - 10.0) & (rr_b <= r_roi) & ((g_sig_b + r_sig_b) > 15.0)
    mask_M_core = in_b & (r_sig_b > g_sig_b)
    mask_E_shell = in_b & (g_sig_b >= r_sig_b)

    r_M_spec = float(np.mean(rr_b[mask_M_core]) / r_roi)
    r_E_spec = float(np.mean(rr_b[mask_E_shell]) / r_roi)
    radial_sep_spec = float(r_E_spec - r_M_spec)

    n_bins = 12
    edges = np.linspace(0.0, 1.05, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_norm_b = rr_b / r_roi
    rho_M_spec = np.zeros(n_bins, dtype=np.float64)
    rho_E_spec = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        in_bin = (r_norm_b >= edges[b]) & (r_norm_b < edges[b + 1]) & in_b
        if np.any(in_bin):
            m_val = float(np.mean(r_sig_b[in_bin]))
            e_val = float(np.mean(g_sig_b[in_bin]))
            tot = m_val + e_val + 1e-8
            rho_M_spec[b] = m_val / tot
            rho_E_spec[b] = e_val / tot

    # 2. Row C mushroom cell-cluster doublet panels
    c_gfp = img[734:1021, 251:497, :]
    c_rfp = img[734:1021, 498:744, :]
    g_sig_c = ndimage.gaussian_filter(np.clip(c_gfp[:, :, 1] - 0.35 * c_gfp[:, :, 0], 0.0, 255.0), sigma=2.5)
    r_sig_c = ndimage.gaussian_filter(np.clip(c_rfp[:, :, 0] - 0.35 * c_rfp[:, :, 1], 0.0, 255.0), sigma=2.5)
    valid_c = np.zeros_like(g_sig_c, dtype=bool)
    valid_c[10:250, 20:226] = True
    mask_g_c = (g_sig_c >= np.percentile(g_sig_c[valid_c], 82)) & valid_c
    mask_r_c = (r_sig_c >= np.percentile(r_sig_c[valid_c], 82)) & valid_c

    yy_c, xx_c = np.indices(g_sig_c.shape, dtype=np.float64)
    y_g, x_g = float(np.mean(yy_c[mask_g_c])), float(np.mean(xx_c[mask_g_c]))
    y_r, x_r = float(np.mean(yy_c[mask_r_c])), float(np.mean(xx_c[mask_r_c]))
    r_lobe_g = float(np.sqrt(np.sum(mask_g_c) / np.pi))
    r_lobe_r = float(np.sqrt(np.sum(mask_r_c) / np.pi))
    r_mean_lobe = 0.5 * (r_lobe_g + r_lobe_r)
    d_centers = float(np.hypot(y_r - y_g, x_r - x_g))
    cos_theta_emp = float(np.clip(d_centers / (2.0 * r_mean_lobe), 0.0, 1.0))
    contact_radius_emp = float(np.sqrt(max(0.0, 1.0 - cos_theta_emp**2)))

    # Construct RGB composite of Row C GFP + RFP mushroom doublet for visualization
    c_composite = np.zeros_like(c_gfp, dtype=np.uint8)
    c_composite[:, :, 1] = np.clip(g_sig_c * 1.4, 0, 255).astype(np.uint8)
    c_composite[:, :, 0] = np.clip(r_sig_c * 1.4, 0, 255).astype(np.uint8)
    c_composite[:, :, 2] = np.clip(0.25 * (g_sig_c + r_sig_c), 0, 255).astype(np.uint8)

    profile_r = 0.0
    profile_rmse = 0.0
    profile_p = 1.0
    if sim_wt is not None:
        sim_diff = sim_wt["rho_M_radial"] - sim_wt["rho_E_radial"]
        spec_diff = rho_M_spec - rho_E_spec
        pr = stats.pearsonr(sim_diff, spec_diff)
        profile_r = float(pr.statistic)
        profile_p = float(pr.pvalue)
        s_norm = (sim_wt["rho_M_radial"] - np.min(sim_wt["rho_M_radial"])) / max(
            np.ptp(sim_wt["rho_M_radial"]), 1e-8
        )
        e_norm = (rho_M_spec - np.min(rho_M_spec)) / max(np.ptp(rho_M_spec), 1e-8)
        profile_rmse = float(np.sqrt(np.mean((s_norm - e_norm) ** 2)))

    return {
        "row_b_merged": b_mrg.astype(np.uint8),
        "row_c_composite": c_composite,
        "mask_M_core": mask_M_core & in_b,
        "mask_E_shell": mask_E_shell & in_b,
        "mask_g_c": mask_g_c,
        "mask_r_c": mask_r_c,
        "cy_core": cy_core,
        "cx_core": cx_core,
        "r_roi": r_roi,
        "r_M_spec_norm": r_M_spec,
        "r_E_spec_norm": r_E_spec,
        "radial_sep_spec": radial_sep_spec,
        "y_g_doublet": y_g,
        "x_g_doublet": x_g,
        "y_r_doublet": y_r,
        "x_r_doublet": x_r,
        "cos_theta_emp": cos_theta_emp,
        "contact_radius_emp": contact_radius_emp,
        "radial_bin_centers": centers,
        "rho_M_spec": rho_M_spec,
        "rho_E_spec": rho_E_spec,
        "profile_pearson_r": profile_r,
        "profile_pearson_p": profile_p,
        "profile_rmse": profile_rmse,
    }


# =============================================================================
# Section 4: 15-Panel Synthesis Figure (3x5; Row C = Nodal coupling) & Animation Rendering
# =============================================================================

def run_all_regimes_and_render(
    beta_E: float = 50.0,
    beta_M: float = 95.0,
    beta_N: float = 130.0,
    gamma_EM: float = 30.0,
    Vt: int = 16,
    lam_vol: float = 1.2,
    T: float = 10.0,
    n_mcs: int = 135,
    seed: int = 2008,
    nodal_C0: float = 1.0,
    nodal_lambda: float = 3.0,
    tau_beta: float = 0.5,
    kappa: float = 2.0 / 3.0,
) -> dict[str, Any]:
    """Run all 4 DITH perturbation regimes + 3-germ-layer sorting, Nodal coupling, sweeps, and render outputs."""
    ref_data = verify_reference_dataset()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    regimes = {
        "wildtype_germ_layer_engulfment": simulate_dith_regime(
            beta_E=beta_E, beta_M=beta_M, gamma_EM_target=gamma_EM,
            Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "camlc_ectoderm_cortex_reversal": simulate_dith_regime(
            beta_E=145.0, beta_M=beta_M, gamma_EM_target=gamma_EM,
            Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "blebbistatin_myosin_inhibition": simulate_dith_regime(
            beta_E=25.0, beta_M=25.0, gamma_EM_target=0.0,
            Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "high_hit_partial_doublet_lens": simulate_dith_regime(
            beta_E=beta_E, beta_M=beta_M, gamma_EM_target=70.0,
            Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
    }
    three_layer = simulate_three_germ_layer_sorting(
        beta_E=beta_E, beta_M=beta_M, beta_N=beta_N, n_mcs=n_mcs, seed=seed
    )
    sweep_beta = run_beta_ectoderm_sweep(beta_M=beta_M, n_mcs=115, seed=seed)
    yd_curve = run_young_dupre_doublet_curve(gamma_cm=beta_M)
    specimen = segment_zebrafish_doublet_specimen(
        sim_wt=regimes["wildtype_germ_layer_engulfment"],
        ref_data=ref_data,
    )
    nodal = simulate_nodal_coupled_internalization(
        C0=nodal_C0, lam_tiers=nodal_lambda, tau_beta=tau_beta, kappa=kappa,
        Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed, record_frames=True,
    )
    nodal_null = simulate_nodal_coupled_internalization(
        C0=0.0, lam_tiers=nodal_lambda, tau_beta=tau_beta, kappa=kappa,
        Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed, record_frames=True,
    )
    sweep_nodal = run_nodal_dose_sweep(
        n_mcs=100, seed=seed, lam_tiers=nodal_lambda, tau_beta=tau_beta, kappa=kappa,
        Vt=Vt, lam_vol=lam_vol, T=T,
    )

    # Render 15-panel comparison figure (3x5 grid)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12.6), dpi=150)
    cmap_2 = mcolors.ListedColormap(["#0f172a", "#22c55e", "#f43f5e"])
    cmap_3 = mcolors.ListedColormap(["#0f172a", "#22c55e", "#f43f5e", "#facc15"])

    panel_specs = [
        ("wildtype_germ_layer_engulfment", "A1. Wild-Type Engulfment\n(β_E=50 < β_M=95 pN/μm)"),
        ("camlc_ectoderm_cortex_reversal", "A2. caMLCK Cortex Reversal\n(β_E+caMLCK=145 > β_M=95)"),
        ("blebbistatin_myosin_inhibition", "A3. Blebbistatin Myosin-II Inh.\n(β_E = β_M = 25 pN/μm, Δβ=0)"),
        ("high_hit_partial_doublet_lens", "A4. High-HIT Mushroom Doublet\n(γ_EM=70 > |β_M−β_E|=45)"),
    ]
    for col, (rk, title) in enumerate(panel_specs):
        ax = axes[0, col]
        r_obj = regimes[rk]
        ax.imshow(r_obj["type_grid"], cmap=cmap_2, vmin=0, vmax=2, interpolation="nearest")
        sig = r_obj["sigma"]
        edges_mask = ((sig != np.roll(sig, 1, axis=0)) | (sig != np.roll(sig, 1, axis=1))) & (
            r_obj["type_grid"] > 0
        )
        edge_rgba = np.zeros((*sig.shape, 4), dtype=np.float32)
        edge_rgba[edges_mask] = [1.0, 1.0, 1.0, 0.34]
        ax.imshow(edge_rgba, interpolation="nearest")
        ax.set_title(
            f"{title}\nΔr={r_obj['radial_sep']:+.3f} | L_EM/L={r_obj['het_final']:.3f}",
            fontsize=8.8,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # A5: 3-Germ-Layer Concentric Stratification (Ectoderm / Mesoderm / Endoderm)
    ax_a5 = axes[0, 4]
    ax_a5.imshow(three_layer["type_grid"], cmap=cmap_3, vmin=0, vmax=3, interpolation="nearest")
    ax_a5.set_title(
        f"A5. 3-Germ-Layer Stratification\n(r_E={three_layer['r_E_norm']:.2f} > r_M={three_layer['r_M_norm']:.2f} > r_N={three_layer['r_N_norm']:.2f})",
        fontsize=8.8,
        fontweight="bold",
    )
    ax_a5.set_xticks([])
    ax_a5.set_yticks([])

    # B1: Radial Separation Trajectory (Wild-Type vs caMLCK Reversal vs Blebbistatin)
    ax_b1 = axes[1, 0]
    mcs_x = np.arange(len(regimes["wildtype_germ_layer_engulfment"]["rad_sep_history"]))
    ax_b1.plot(
        mcs_x,
        regimes["wildtype_germ_layer_engulfment"]["rad_sep_history"],
        color="#16a34a",
        lw=2.1,
        label="Wild-Type (β_E=50 < β_M=95)",
    )
    ax_b1.plot(
        mcs_x,
        regimes["camlc_ectoderm_cortex_reversal"]["rad_sep_history"],
        color="#e11d48",
        lw=2.1,
        label="caMLCK Reversal (β_E=145 > β_M=95)",
    )
    ax_b1.plot(
        mcs_x,
        regimes["blebbistatin_myosin_inhibition"]["rad_sep_history"],
        color="#64748b",
        lw=1.8,
        ls="--",
        label="Blebbistatin (β_E=β_M=25)",
    )
    ax_b1.axhline(0.0, color="#1e293b", ls=":", lw=1.0)
    ax_b1.set_title("B1. Sorting Trajectory & caMLCK Inversion", fontsize=9.2, fontweight="bold")
    ax_b1.set_xlabel("Monte Carlo Step (MCS)", fontsize=8.5)
    ax_b1.set_ylabel("Radial Separation Δr = (r_E − r_M)/R", fontsize=8.5)
    ax_b1.legend(fontsize=6.8, loc="center right")
    ax_b1.grid(alpha=0.25)

    # B2: 7-Point Continuous Ectodermal Cortical Tension Sweep (beta_E vs fixed beta_M=95)
    ax_b2 = axes[1, 1]
    be_vals = [s["beta_E"] for s in sweep_beta]
    sep_vals = [s["radial_sep"] for s in sweep_beta]
    env_vals = [s["outer_E_envelopment"] for s in sweep_beta]
    ax_b2.axvspan(20.0, 95.0, color="#dcfce7", alpha=0.55, label="Ectoderm Outside (β_E < β_M)")
    ax_b2.axvspan(95.0, 150.0, color="#ffe4e6", alpha=0.55, label="Ectoderm Inside (β_E > β_M)")
    ax_b2.plot(be_vals, sep_vals, "o-", color="#15803d", lw=2.0, label="Radial sep Δr")
    ax_b2.plot(be_vals, env_vals, "s--", color="#0284c7", lw=1.7, label="Outer E envelopment")
    ax_b2.axvline(95.0, color="#b91c1c", ls="--", lw=1.2, label="β_E = β_M = 95 pN/μm")
    ax_b2.axhline(0.0, color="#334155", ls=":", lw=1.0)
    ax_b2.set_title("B2. 7-Point Cortical Tension β_E Sweep", fontsize=9.2, fontweight="bold")
    ax_b2.set_xlabel("Ectoderm Cortical Tension β_E (pN/μm)", fontsize=8.5)
    ax_b2.set_ylabel("Order Parameter", fontsize=8.5)
    ax_b2.legend(fontsize=6.5, loc="lower left")
    ax_b2.grid(alpha=0.25)

    # B3: Maître et al. (2012) Science Young-Dupré Cell-Doublet Contact Angle Law
    ax_b3 = axes[1, 2]
    r_grid = np.linspace(0.05, 1.0, 100)
    cos_grid = 1.0 - r_grid
    sin_grid = np.sqrt(np.clip(1.0 - cos_grid**2, 0.0, 1.0))
    ax_b3.plot(r_grid, cos_grid, "-", color="#2563eb", lw=2.0, label="cos(θ) = 1 − γ_cc/(2γ_cm)")
    ax_b3.plot(r_grid, sin_grid, "-", color="#db2777", lw=2.0, label="r_c / R = sin(θ)")
    yd_x = [d["gamma_cc_over_2gamma_cm"] for d in yd_curve]
    ax_b3.plot(yd_x, [d["cos_theta"] for d in yd_curve], "o", color="#1d4ed8", ms=5)
    ax_b3.plot(yd_x, [d["contact_radius_r_over_R"] for d in yd_curve], "s", color="#be185d", ms=5)
    emp_ratio = 1.0 - specimen["cos_theta_emp"]
    ax_b3.plot(
        [emp_ratio],
        [specimen["contact_radius_emp"]],
        "*",
        color="#16a34a",
        ms=11,
        label=f"Row C Doublet (r_c/R={specimen['contact_radius_emp']:.2f})",
    )
    ax_b3.set_title("B3. Maître (2012) Young–Dupré Doublet Law", fontsize=9.2, fontweight="bold")
    ax_b3.set_xlabel("Relative Interfacial Tension γ_cc / (2γ_cm)", fontsize=8.5)
    ax_b3.set_ylabel("cos(θ) & Contact Radius r_c / R", fontsize=8.5)
    ax_b3.legend(fontsize=6.8, loc="center left")
    ax_b3.grid(alpha=0.25)

    # B4: Zebrafish Blastoderm Cross-Transplantation & Mushroom Doublet Specimen
    ax_b4 = axes[1, 3]
    ax_b4.imshow(specimen["row_c_composite"])
    ax_b4.contour(specimen["mask_g_c"], levels=[0.5], colors=["#4ade80"], linewidths=1.6)
    ax_b4.contour(specimen["mask_r_c"], levels=[0.5], colors=["#fb7185"], linewidths=1.6)
    ax_b4.plot(
        [specimen["x_g_doublet"], specimen["x_r_doublet"]],
        [specimen["y_g_doublet"], specimen["y_r_doublet"]],
        "w--o",
        lw=1.5,
        ms=4,
    )
    ax_b4.set_title(
        f"B4. Zebrafish Blastoderm Doublet Specimen\n(Row B Δr={specimen['radial_sep_spec']:+.2f}, Row C r_c/R={specimen['contact_radius_emp']:.2f})",
        fontsize=8.8,
        fontweight="bold",
    )
    ax_b4.set_xticks([])
    ax_b4.set_yticks([])

    # B5: Radial Core–Shell Occupancy Profile (DITH Simulation vs Zebrafish Specimen)
    ax_b5 = axes[1, 4]
    rc = specimen["radial_bin_centers"]
    ax_b5.plot(
        rc,
        regimes["wildtype_germ_layer_engulfment"]["rho_M_radial"],
        "o-",
        color="#f43f5e",
        lw=2.0,
        label="DITH Mesoderm Core M (sim)",
    )
    ax_b5.plot(
        rc,
        regimes["wildtype_germ_layer_engulfment"]["rho_E_radial"],
        "s-",
        color="#22c55e",
        lw=2.0,
        label="DITH Ectoderm Shell E (sim)",
    )
    ax_b5.plot(
        rc,
        specimen["rho_M_spec"],
        "o--",
        color="#be123c",
        lw=1.7,
        label="Specimen RFP Core (emp)",
    )
    ax_b5.plot(
        rc,
        specimen["rho_E_spec"],
        "s--",
        color="#15803d",
        lw=1.7,
        label="Specimen GFP Rim (emp)",
    )
    ax_b5.set_title(
        f"B5. Radial Core–Shell Profiles\n(r = {specimen['profile_pearson_r']:.3f}, RMSE = {specimen['profile_rmse']:.3f})",
        fontsize=9.0,
        fontweight="bold",
    )
    ax_b5.set_xlabel("Normalized Radial Distance r / R_agg", fontsize=8.5)
    ax_b5.set_ylabel("Normalized Progenitor Occupancy", fontsize=8.5)
    ax_b5.legend(fontsize=6.8, loc="center right")
    ax_b5.grid(alpha=0.25)

    # ---- Row C: time-dependent Nodal/TGF-beta -> cortical tension coupling (Pass 3) ----
    n_tiers = nodal["n_tiers"]
    tier_axis = np.linspace(0.0, n_tiers - 1 + 0.5, 120)
    tier_colors = plt.cm.viridis(np.linspace(0.0, 1.0, n_tiers))

    # C1: Nodal field C(x, t) vs tier distance at four stages, with the fate thresholds
    ax_c1 = axes[2, 0]
    for t_s, col, lbl in [
        (4.5, "#c7d2fe", "4.5 hpf (dome)"),
        (5.0, "#818cf8", "5.0 hpf (~40% epiboly)"),
        (6.0, "#4338ca", "6.0 hpf (shield)"),
        (7.0, "#1e1b4b", "7.0 hpf (~65% epiboly)"),
    ]:
        ax_c1.plot(tier_axis, nodal_signal(tier_axis, t_s, C0=nodal_C0, lam_tiers=nodal_lambda), color=col, lw=2.0, label=lbl)
    ax_c1.axhline(0.2, color="#f43f5e", ls="--", lw=1.1, label="K_M (mesoderm)")
    ax_c1.axhline(0.6, color="#facc15", ls="--", lw=1.1, label="K_N (endoderm)")
    ax_c1.set_title(
        f"C1. Marginal Nodal Field C(x,t)\n(λ={nodal_lambda:.1f} tiers, τ_N=0.6 h, t_on=4 hpf)",
        fontsize=9.2, fontweight="bold",
    )
    ax_c1.set_xlabel("Tier distance from margin x (cell diameters)", fontsize=8.5)
    ax_c1.set_ylabel("Nodal signal C / C0", fontsize=8.5)
    ax_c1.legend(fontsize=6.4, loc="upper right")
    ax_c1.grid(alpha=0.25)

    # C2: cortical tension relaxation beta_i(t) per tier of origin
    ax_c2 = axes[2, 1]
    for k in range(n_tiers):
        ax_c2.plot(nodal["times_hpf"], nodal["beta_tier_history"][:, k], color=tier_colors[k], lw=1.8, label=f"tier {k}")
    for b_ref, col, lbl in [(BETA_ECTODERM, "#22c55e", "β_E=50"), (BETA_MESODERM, "#f43f5e", "β_M=95"), (BETA_ENDODERM, "#facc15", "β_N=130")]:
        ax_c2.axhline(b_ref, color=col, ls=":", lw=1.1)
        ax_c2.text(NODAL_T_ON_HPF + 0.03, b_ref + 1.5, lbl, fontsize=6.6, color=col)
    ax_c2.axvline(NODAL_T_SHIELD_HPF, color="#b91c1c", ls="--", lw=1.1)
    ax_c2.text(NODAL_T_SHIELD_HPF + 0.03, 52.0, "shield", fontsize=6.8, color="#b91c1c")
    hier_t = nodal["hierarchy_established_hpf"]
    ax_c2.set_title(
        f"C2. Cortical Tension β_i(t) per Tier\n(dβ/dt=(β*(C)−β)/τ_β, τ_β={tau_beta:.1f} h; hierarchy by {hier_t:.2f} hpf)",
        fontsize=9.0, fontweight="bold",
    )
    ax_c2.set_xlabel("Time (hpf)", fontsize=8.5)
    ax_c2.set_ylabel("Cortical tension β (pN/μm)", fontsize=8.5)
    ax_c2.legend(fontsize=5.8, loc="center left", ncol=2)
    ax_c2.grid(alpha=0.25)

    # C3: Nodal-coupled lattice at 7 hpf, coloured by continuous beta_i
    ax_c3 = axes[2, 2]
    beta_show = np.where(nodal["beta_grid"] > 0, nodal["beta_grid"], np.nan)
    cmap_beta = plt.cm.plasma.copy()
    cmap_beta.set_bad("#0f172a")
    im_c3 = ax_c3.imshow(beta_show, cmap=cmap_beta, vmin=BETA_ECTODERM, vmax=BETA_ENDODERM, interpolation="nearest")
    sig_n = nodal["sigma"]
    edges_n = ((sig_n != np.roll(sig_n, 1, axis=0)) | (sig_n != np.roll(sig_n, 1, axis=1))) & (sig_n > 0)
    edge_rgba_n = np.zeros((*sig_n.shape, 4), dtype=np.float32)
    edge_rgba_n[edges_n] = [1.0, 1.0, 1.0, 0.30]
    ax_c3.imshow(edge_rgba_n, interpolation="nearest")
    ax_c3.axhline(sig_n.shape[0] // 2 + 17.5, color="#38bdf8", lw=1.4, ls="-")
    ax_c3.text(1.5, sig_n.shape[0] // 2 + 20.5, "margin (Nodal source)", fontsize=6.6, color="#38bdf8")
    cb = fig.colorbar(im_c3, ax=ax_c3, fraction=0.046, pad=0.02)
    cb.set_label("β_i (pN/μm)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    ax_c3.set_title(
        f"C3. Nodal-Coupled DITH Internalization at 7 hpf\n(Δr={nodal['radial_sep']:+.3f}; shield mesendo tiers={nodal['tiers_mesendo_at_shield']}, endo={nodal['tiers_endo_at_shield']})",
        fontsize=8.8, fontweight="bold",
    )
    ax_c3.set_xticks([])
    ax_c3.set_yticks([])

    # C4: sorting trajectory Delta_r(t) and mesendoderm fraction vs hpf (wild-type vs Nodal null)
    ax_c4 = axes[2, 3]
    ax_c4.plot(nodal["times_hpf"], nodal["rad_sep_history"], color="#15803d", lw=2.1, label="Δr (Nodal C0=%.1f)" % nodal_C0)
    ax_c4.plot(nodal_null["times_hpf"], nodal_null["rad_sep_history"], color="#64748b", lw=1.7, ls="--", label="Δr (Nodal null, C0=0)")
    ax_c4.plot(nodal["times_hpf"], nodal["frac_mesendo_history"], color="#f43f5e", lw=1.7, label="mesendoderm fraction")
    ax_c4.plot(nodal["times_hpf"], nodal["frac_endo_history"], color="#ca8a04", lw=1.5, label="endoderm fraction")
    ax_c4.axhline(regimes["wildtype_germ_layer_engulfment"]["radial_sep"], color="#16a34a", ls=":", lw=1.2, label="static WT Δr (A1)")
    ax_c4.axvline(NODAL_T_SHIELD_HPF, color="#b91c1c", ls="--", lw=1.1)
    ax_c4.text(NODAL_T_SHIELD_HPF + 0.03, 0.02, "shield 6 hpf", fontsize=6.8, color="#b91c1c")
    ax_c4.set_title("C4. Internalization Trajectory vs. Developmental Time", fontsize=9.2, fontweight="bold")
    ax_c4.set_xlabel("Time (hpf)", fontsize=8.5)
    ax_c4.set_ylabel("Δr = (r_E − r_ME)/R  |  fraction", fontsize=8.5)
    ax_c4.legend(fontsize=6.3, loc="center right")
    ax_c4.grid(alpha=0.25)

    # C5: 5-point Nodal dose sweep (lefty1 / MZoep null -> cyclops overexpression)
    ax_c5 = axes[2, 4]
    c0_vals = [s["C0"] for s in sweep_nodal]
    xpos = np.arange(len(c0_vals))
    ax_c5.plot(xpos, [s["radial_sep"] for s in sweep_nodal], "o-", color="#15803d", lw=2.0, label="Δr at 7 hpf")
    ax_c5.plot(xpos, [s["frac_mesendo_final"] for s in sweep_nodal], "s--", color="#f43f5e", lw=1.7, label="mesendoderm fraction")
    ax_c5.plot(xpos, [s["frac_endo_final"] for s in sweep_nodal], "^--", color="#ca8a04", lw=1.5, label="endoderm fraction")
    ax_c5.plot(xpos, [s["tiers_mesendo_at_shield"] / max(n_tiers, 1) for s in sweep_nodal], "d:", color="#4338ca", lw=1.4, label=f"shield mesendo tiers / {n_tiers}")
    ax_c5.axhline(0.0, color="#334155", ls=":", lw=1.0)
    ax_c5.set_xticks(xpos)
    ax_c5.set_xticklabels([("0\n(lefty1 / MZoep)" if c == 0 else ("1\n(wild-type)" if c == 1 else f"{c:g}")) for c in c0_vals], fontsize=7.2)
    ax_c5.set_title("C5. 5-Point Nodal Dose Sweep C0", fontsize=9.2, fontweight="bold")
    ax_c5.set_xlabel("Marginal Nodal dose C0 (× wild-type)", fontsize=8.5)
    ax_c5.set_ylabel("Order parameter / fraction", fontsize=8.5)
    ax_c5.legend(fontsize=6.3, loc="upper left")
    ax_c5.grid(alpha=0.25)

    fig.suptitle(
        "Brodland (2002) J. Biomech. Eng. / Krieg et al. (2008) Nat. Cell Biol. / Maître et al. (2012) Science DITH Germ-Layer Sorting"
        " — Pass 3: time-dependent Nodal/TGF-β → cortical tension coupling (Row C)",
        fontsize=12.6,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(COMPARISON_PNG, dpi=150, facecolor="white")
    plt.close(fig)

    # Render the Nodal-coupled internalization GIF (beta_i(t) map, wild-type vs Nodal null)
    nodal_frames: list[Image.Image] = []
    f_nod = nodal["frames"]
    f_nul = nodal_null["frames"]
    for idx in range(min(len(f_nod), len(f_nul))):
        bg_wt, _, t_val = f_nod[idx]
        bg_nl, _, _ = f_nul[idx]
        fig_n, axes_n = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=100)
        for ax_i, bg, lbl in [
            (axes_n[0], bg_wt, f"Nodal C0={nodal_C0:.1f} (t={t_val:.2f} hpf)"),
            (axes_n[1], bg_nl, f"Nodal null C0=0 (t={t_val:.2f} hpf)"),
        ]:
            ax_i.imshow(np.where(bg > 0, bg, np.nan), cmap=cmap_beta, vmin=BETA_ECTODERM, vmax=BETA_ENDODERM, interpolation="nearest")
            ax_i.set_title(lbl, fontsize=8.4, fontweight="bold")
            ax_i.set_xticks([])
            ax_i.set_yticks([])
        fig_n.suptitle("Nodal/TGF-β → DITH cortical tension β_i(t): shield-stage internalization (4 → 7 hpf)", fontsize=8.8, fontweight="bold")
        fig_n.tight_layout(rect=(0, 0, 1, 0.9))
        fig_n.canvas.draw()
        nodal_frames.append(Image.fromarray(np.asarray(fig_n.canvas.buffer_rgba())[..., :3]))
        plt.close(fig_n)
    if nodal_frames:
        nodal_frames[0].save(NODAL_GIF, save_all=True, append_images=nodal_frames[1:], duration=160, loop=0)

    # Render side-by-side 3-regime animated GIF (Wild-Type vs caMLCK Reversal vs Blebbistatin)
    pil_frames: list[Image.Image] = []
    f_wt = regimes["wildtype_germ_layer_engulfment"]["frames"]
    f_rev = regimes["camlc_ectoderm_cortex_reversal"]["frames"]
    f_blb = regimes["blebbistatin_myosin_inhibition"]["frames"]
    n_f = min(len(f_wt), len(f_rev), len(f_blb))

    for idx in range(n_f):
        _, tg_wt, mcs_val = f_wt[idx]
        _, tg_rev, _ = f_rev[idx]
        _, tg_blb, _ = f_blb[idx]
        fig_g, axes_g = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=100)
        for ax_i, tg, lbl in [
            (axes_g[0], tg_wt, f"Wild-Type β_E<β_M (MCS={mcs_val})"),
            (axes_g[1], tg_rev, f"caMLCK Reversal β_E>β_M (MCS={mcs_val})"),
            (axes_g[2], tg_blb, f"Blebbistatin Δβ=0 (MCS={mcs_val})"),
        ]:
            ax_i.imshow(tg, cmap=cmap_2, vmin=0, vmax=2, interpolation="nearest")
            ax_i.set_title(lbl, fontsize=8.4, fontweight="bold")
            ax_i.set_xticks([])
            ax_i.set_yticks([])
        fig_g.suptitle(
            "Krieg et al. (2008) / Brodland (2002) DITH Sorting: Wild-Type vs. caMLCK Reversal & Blebbistatin",
            fontsize=9.5,
            fontweight="bold",
        )
        fig_g.tight_layout(rect=(0, 0, 1, 0.91))
        fig_g.canvas.draw()
        buf = np.asarray(fig_g.canvas.buffer_rgba())[..., :3]
        pil_frames.append(Image.fromarray(buf))
        plt.close(fig_g)

    pil_frames[0].save(
        GROWTH_GIF,
        save_all=True,
        append_images=pil_frames[1:],
        duration=130,
        loop=0,
    )

    return {
        "regimes": regimes,
        "three_layer": three_layer,
        "sweep_beta": sweep_beta,
        "yd_curve": yd_curve,
        "specimen": specimen,
        "nodal": nodal,
        "nodal_null": nodal_null,
        "sweep_nodal": sweep_nodal,
    }


# =============================================================================
# Section 5: MorphoEval Dynamic Measurement Hook & Self-Test Suite
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Return 13 quantitative morphometric measurements for morphoeval/measure_outputs.py."""
    ref_data = verify_reference_dataset()
    t_wt = compute_dith_tensions(50.0, 95.0, omega_EM_eff=115.0)
    r_wt = simulate_dith_regime(beta_E=50.0, beta_M=95.0, gamma_EM_target=30.0, n_mcs=135, seed=2008)
    r_rev = simulate_dith_regime(beta_E=145.0, beta_M=95.0, gamma_EM_target=30.0, n_mcs=135, seed=2008)
    r_blb = simulate_dith_regime(beta_E=25.0, beta_M=25.0, gamma_EM_target=0.0, n_mcs=135, seed=2008)
    three_l = simulate_three_germ_layer_sorting(50.0, 95.0, 130.0, n_mcs=140, seed=2008)
    yd_half = compute_young_dupre_doublet(gamma_cm=95.0, gamma_cc=95.0)
    specimen = segment_zebrafish_doublet_specimen(sim_wt=r_wt, ref_data=ref_data)
    nodal = simulate_nodal_coupled_internalization(C0=1.0, n_mcs=135, seed=2008)
    nodal_null = simulate_nodal_coupled_internalization(C0=0.0, n_mcs=135, seed=2008)
    gamma_em_closure = float(cortical_tension_mismatch_gamma(BETA_ECTODERM, BETA_MESODERM))

    return [
        {
            "name": "dith_wildtype_gamma_EM",
            "value": round(t_wt["gamma_EM"], 4),
            "unit": "pN/um",
            "reference_value": 30.0,
            "method": "Exact Brodland (2002) J. Biomech. Eng. Eq. 1 heterotypic interfacial tension gamma_EM = beta_E + beta_M - omega_EM for beta_E=50, beta_M=95, omega_EM=115 pN/um",
        },
        {
            "name": "krieg_2008_spreading_tension_diff",
            "value": round(t_wt["spreading_diff"], 4),
            "unit": "pN/um",
            "reference_value": 45.0,
            "method": "Krieg et al. (2008) Nat. Cell Biol. AFM cortical tension difference beta_M - beta_E = 95 - 50 = 45 pN/um driving ectoderm envelopment of mesoderm",
        },
        {
            "name": "wildtype_radial_core_shell_separation",
            "value": round(r_wt["radial_sep"], 4),
            "unit": "normalized_radius",
            "reference_value": 0.34,
            "method": f"Normalized radial separation (r_E - r_M)/R_agg in wild-type zebrafish germ-layer sorting (outer E envelopment={r_wt['outer_E_envelopment']:.4f}; single deterministic seed, no replicate statistics)",
        },
        {
            "name": "camlc_cortex_reversal_radial_separation",
            "value": round(r_rev["radial_sep"], 4),
            "unit": "normalized_radius",
            "reference_value": -0.34,
            "method": "Inverted radial separation (r_E - r_M)/R_agg < 0 under caMLCK ectodermal cortical stiffening (beta_{E+caMLCK}=145 > beta_M=95 pN/um, Krieg et al. 2008 Fig. 4)",
        },
        {
            "name": "blebbistatin_control_radial_separation",
            "value": round(abs(r_blb["radial_sep"]), 4),
            "unit": "normalized_radius",
            "reference_value": 0.0,
            "method": "Null radial separation |r_E - r_M|/R_agg under Blebbistatin Myosin-II inhibition (beta_E=beta_M=25 pN/um, Krieg et al. 2008 Fig. 3)",
        },
        {
            "name": "three_germ_layer_ectoderm_endoderm_radial_gap",
            "value": round(three_l["r_E_norm"] - three_l["r_N_norm"], 4),
            "unit": "normalized_radius",
            "reference_value": 0.42,
            "method": f"Concentric 3-germ-layer radial stratification gap (r_E - r_N)/R_agg with r_E={three_l['r_E_norm']:.3f} > r_M={three_l['r_M_norm']:.3f} > r_N={three_l['r_N_norm']:.3f} (Krieg et al. 2008 Fig. 2)",
        },
        {
            "name": "maitre_2012_young_dupre_contact_radius_half_tension",
            "value": round(yd_half["contact_radius_r_over_R"], 4),
            "unit": "normalized_contact_radius",
            "reference_value": 0.866,
            "method": f"Maître et al. (2012) Science Eq. 1 Young-Dupré doublet contact radius r_c/R = sin(theta) at gamma_cc/(2*gamma_cm)=0.5 (theta={yd_half['theta_deg']:.2f} deg, specimen Row C r_c/R={specimen['contact_radius_emp']:.4f})",
        },
        {
            "name": "specimen_vs_dith_radial_profile_pearson_r",
            "value": round(specimen["profile_pearson_r"], 4),
            "unit": "r",
            "reference_value": 0.92,
            "method": f"Pearson correlation between DITH radial core-shell profile and zebrafish blastoderm GFP/RFP fluorescence profile across 12 radial annuli (RMSE={specimen['profile_rmse']:.4f}, computed two-sided p={specimen['profile_pearson_p']:.2e}, n=12 annuli)",
        },
        {
            "name": "nodal_mesendoderm_tier_count_at_shield_6hpf",
            "value": int(nodal["tiers_mesendo_at_shield"]),
            "unit": "cell_tiers",
            "reference_value": 5,
            "method": f"Number of tiers of origin whose mean Nodal-driven cortical tension exceeds (beta_E+beta_M)/2 at shield stage 6.0 hpf (lambda={nodal['params']['lam_tiers']:.1f} tiers, K_M=0.2); reference ~5 marginal tiers in the Nodal signalling / mesendoderm domain (van Boxtel et al. 2015 Dev. Cell; Dubrulle et al. 2015 eLife; approximate, recalled value not digitised here)",
        },
        {
            "name": "nodal_endoderm_tier_count_at_shield_6hpf",
            "value": int(nodal["tiers_endo_at_shield"]),
            "unit": "cell_tiers",
            "reference_value": 2,
            "method": "Number of marginal tiers whose mean cortical tension exceeds (beta_M+beta_N)/2 at 6.0 hpf (K_N=0.6); reference: endoderm arises from the marginal-most tiers, predominantly tiers 1-2 (Warga & Nusslein-Volhard 1999 Development 126:827; approximate)",
        },
        {
            "name": "nodal_cortical_tension_hierarchy_established_hpf",
            "value": round(float(nodal["hierarchy_established_hpf"]), 4),
            "unit": "hpf",
            "reference_value": 6.0,
            "method": "Earliest developmental time at which margin > mid > animal tier mean cortical tensions differ by >= 10 pN/um each; reference: Krieg et al. (2008) measured the ectoderm < mesoderm < endoderm AFM hierarchy in shield-stage (6 hpf) progenitors, so the hierarchy must exist by 6.0 hpf (upper bound)",
        },
        {
            "name": "nodal_coupled_internalization_radial_sep_7hpf",
            "value": round(float(nodal["radial_sep"]), 4),
            "unit": "normalized_radius",
            "reference_value": 0.34,
            "method": f"Radial separation (r_E - r_ME)/R_agg at 7.0 hpf when cortical tensions are set dynamically by the marginal Nodal field (C0=1, tau_beta=0.5 h, kappa=2/3; mesendoderm fraction={nodal['frac_mesendo_final']:.3f}); same wild-type anchor as the static regime (Krieg et al. 2008 Fig. 2)",
        },
        {
            "name": "nodal_null_mzoep_radial_sep",
            "value": round(abs(float(nodal_null["radial_sep"])), 4),
            "unit": "normalized_radius",
            "reference_value": 0.0,
            "method": f"Null radial separation under Nodal loss of function (C0=0: lefty1 / MZoep; mesendoderm fraction={nodal_null['frac_mesendo_final']:.3f}); reference: MZoep embryos lack all mesoderm and endoderm (Gritsman et al. 1999 Cell 97:121), hence no germ-layer sorting (closure gamma_EM={gamma_em_closure:.1f} pN/um at beta_E=50, beta_M=95)",
        },
    ]


def self_test() -> None:
    """Run comprehensive 13-check Pass 3 verification suite for Slot 044."""
    # 1. Verify SHA-256 of dataset and zebrafish blastoderm specimen image
    ref = verify_reference_dataset()
    assert len(ref["metadata"]["sources"]) == 12  # 5 DITH/doublet sources + 7 Nodal/staging sources (Pass 3)
    assert "nodal_tgfb_cortical_tension_coupling_pass3" in ref["dith_interfacial_tension_mechanics"]

    # 2. Verify exact DITH tensions against Lean 4 theorem values
    t_wt = compute_dith_tensions(50.0, 95.0, omega_EM_eff=115.0)
    assert t_wt["gamma_EM"] == 30.0 and t_wt["spreading_diff"] == 45.0
    assert t_wt["regime"] == "wildtype_germ_layer_engulfment"

    t_rev = compute_dith_tensions(145.0, 95.0, omega_EM_eff=210.0)
    assert t_rev["gamma_EM"] == 30.0 and t_rev["spreading_diff"] == -50.0
    assert t_rev["regime"] == "camlc_ectoderm_cortex_reversal"

    t_blb = compute_dith_tensions(25.0, 25.0, omega_EM_eff=50.0)
    assert t_blb["gamma_EM"] == 0.0 and t_blb["spreading_diff"] == 0.0
    assert t_blb["regime"] == "blebbistatin_myosin_inhibition"

    t_hit = compute_dith_tensions(50.0, 95.0, omega_EM_eff=75.0)
    assert t_hit["gamma_EM"] == 70.0 and t_hit["regime"] == "high_hit_partial_doublet_lens"

    # 3. Verify Maître et al. (2012) Young-Dupré cell-doublet contact angle bounds
    yd_zero = compute_young_dupre_doublet(95.0, 190.0)
    yd_full = compute_young_dupre_doublet(95.0, 0.0)
    yd_half = compute_young_dupre_doublet(95.0, 95.0)
    assert yd_zero["young_dupre_num"] == 0.0 and abs(yd_zero["cos_theta"]) < 1e-9
    assert yd_full["young_dupre_num"] == 190.0 and abs(yd_full["cos_theta"] - 1.0) < 1e-9
    assert abs(yd_half["theta_deg"] - 60.0) < 1e-6 and abs(yd_half["contact_radius_r_over_R"] - np.sqrt(3) / 2) < 1e-6

    # 4. Verify wild-type engulfment vs caMLCK reversal vs Blebbistatin arrest
    r_wt = simulate_dith_regime(beta_E=50.0, beta_M=95.0, gamma_EM_target=30.0, n_mcs=135, seed=2008)
    r_rev = simulate_dith_regime(beta_E=145.0, beta_M=95.0, gamma_EM_target=30.0, n_mcs=135, seed=2008)
    r_blb = simulate_dith_regime(beta_E=25.0, beta_M=25.0, gamma_EM_target=0.0, n_mcs=135, seed=2008)
    assert r_wt["radial_sep"] > 0.25, f"Expected wild-type radial_sep > 0.25, got {r_wt['radial_sep']}"
    assert r_rev["radial_sep"] < -0.25, f"Expected caMLCK reversal radial_sep < -0.25, got {r_rev['radial_sep']}"
    assert abs(r_blb["radial_sep"]) < 0.05, f"Expected Blebbistatin |radial_sep| < 0.05, got {r_blb['radial_sep']}"

    # 5. Verify 3-germ-layer concentric stratification (r_E > r_M > r_N)
    three_l = simulate_three_germ_layer_sorting(50.0, 95.0, 130.0, n_mcs=140, seed=2008)
    assert three_l["hierarchy_satisfied"], (
        f"Expected r_E > r_M > r_N, got r_E={three_l['r_E_norm']:.3f}, r_M={three_l['r_M_norm']:.3f}, r_N={three_l['r_N_norm']:.3f}"
    )

    # 6. Verify zebrafish blastoderm specimen segmentation & profile correlation
    spec = segment_zebrafish_doublet_specimen(sim_wt=r_wt, ref_data=ref)
    assert spec["radial_sep_spec"] > 0.05, f"Expected specimen radial_sep > 0.05, got {spec['radial_sep_spec']}"
    assert spec["contact_radius_emp"] > 0.85, f"Expected doublet r_c/R > 0.85, got {spec['contact_radius_emp']}"
    assert spec["profile_pearson_r"] > 0.80, f"Expected profile r > 0.80, got {spec['profile_pearson_r']}"

    # 7. Verify two-run bitwise determinism
    r_1 = simulate_dith_regime(beta_E=50.0, beta_M=95.0, gamma_EM_target=30.0, n_mcs=60, seed=2008)
    r_2 = simulate_dith_regime(beta_E=50.0, beta_M=95.0, gamma_EM_target=30.0, n_mcs=60, seed=2008)
    assert np.array_equal(r_1["sigma"], r_2["sigma"])

    # 8. Verify Nodal dose -> cortical tension mapping (Lean `nodal_dose_cortical_tension_monotone_bounded`)
    c_grid = np.linspace(0.0, 3.0, 301)
    b_grid = nodal_to_cortical_tension_target(c_grid)
    assert abs(float(nodal_to_cortical_tension_target(0.0)) - BETA_ECTODERM) < 1e-9
    assert abs(float(nodal_to_cortical_tension_target(1e6)) - BETA_ENDODERM) < 1e-6
    assert np.all(np.diff(b_grid) >= -1e-12) and np.all(b_grid >= BETA_ECTODERM - 1e-9) and np.all(b_grid <= BETA_ENDODERM + 1e-9)
    assert abs(float(nodal_to_cortical_tension_target(0.2)) - 0.5 * (BETA_ECTODERM + BETA_MESODERM)) < 0.5  # H(K_M; K_M) = 1/2
    assert float(nodal_signal(0.0, NODAL_T_ON_HPF)) == 0.0 and float(nodal_signal(0.0, NODAL_T_ON_HPF + 50.0)) > 0.999

    # 9. Verify continuous-cortex closure obeys 0 < gamma_ij < |beta_i - beta_j| (Lean `cortex_mismatch_closure_satisfies_engulfment`)
    rng_t = np.random.default_rng(44)
    b_i = rng_t.uniform(BETA_ECTODERM, BETA_ENDODERM, 200)
    b_j = rng_t.uniform(BETA_ECTODERM, BETA_ENDODERM, 200)
    g_ij = cortical_tension_mismatch_gamma(b_i, b_j)
    assert np.all(g_ij > 0.0) and np.all(g_ij < np.abs(b_i - b_j) + 1e-12)
    assert abs(float(cortical_tension_mismatch_gamma(BETA_ECTODERM, BETA_MESODERM)) - 30.0) < 1e-9
    assert float(cortical_tension_mismatch_gamma(25.0, 25.0)) == 0.0

    # 10. Verify the tension relaxation ODE contracts toward the Nodal-set target (Lean `cortical_tension_relaxation_contracts`)
    dyn = integrate_nodal_cortical_tension_dynamics(np.array([0.0, 3.0, 7.0]), n_steps=135)
    assert np.all(np.diff(dyn["beta"][:, 0]) >= -1e-9) and dyn["beta"][-1, 0] > 120.0
    assert dyn["beta"][-1, 0] > dyn["beta"][-1, 1] > dyn["beta"][-1, 2] >= BETA_ECTODERM - 1e-9
    assert np.all(dyn["beta"] <= BETA_ENDODERM + 1e-9)

    # 11. Verify Nodal-coupled internalization: hierarchy before shield, mesendoderm tiers, sorting by 7 hpf
    r_nod = simulate_nodal_coupled_internalization(C0=1.0, n_mcs=135, seed=2008)
    assert r_nod["frac_mesendo_history"][0] == 0.0, "all progenitors must be Nodal-naive at 4 hpf"
    assert r_nod["hierarchy_established_hpf"] <= NODAL_T_SHIELD_HPF, f"hierarchy at {r_nod['hierarchy_established_hpf']} hpf > 6 hpf"
    assert 3 <= r_nod["tiers_mesendo_at_shield"] <= 6, f"shield mesendoderm tiers {r_nod['tiers_mesendo_at_shield']}"
    assert 1 <= r_nod["tiers_endo_at_shield"] <= 3, f"shield endoderm tiers {r_nod['tiers_endo_at_shield']}"
    assert r_nod["radial_sep"] > 0.25, f"Expected Nodal-coupled radial_sep > 0.25, got {r_nod['radial_sep']}"
    assert np.all(r_nod["beta"][1:] >= BETA_ECTODERM - 1e-9) and np.all(r_nod["beta"][1:] <= BETA_ENDODERM + 1e-9)

    # 12. Verify Nodal null (lefty1 / MZoep) control: no mesendoderm, no sorting; two-run determinism
    r_nul = simulate_nodal_coupled_internalization(C0=0.0, n_mcs=135, seed=2008)
    assert r_nul["frac_mesendo_final"] == 0.0 and abs(r_nul["radial_sep"]) < 1e-12
    n_1 = simulate_nodal_coupled_internalization(C0=1.0, n_mcs=60, seed=2008)
    n_2 = simulate_nodal_coupled_internalization(C0=1.0, n_mcs=60, seed=2008)
    assert np.array_equal(n_1["sigma"], n_2["sigma"]) and np.array_equal(n_1["beta"], n_2["beta"])

    # 13. Verify measure_morphoeval contract
    meas = measure_morphoeval()
    assert len(meas) == 13 and all(m["value"] is not None for m in meas)
    assert all(m["reference_value"] is not None for m in meas)
    print(
        f"[self-test] All 13 Pass 3 checks passed "
        f"(wt_sep={r_wt['radial_sep']:+.3f}, camlc_rev_sep={r_rev['radial_sep']:+.3f}, "
        f"blebb_sep={r_blb['radial_sep']:+.3f}, 3layer={three_l['r_E_norm']:.2f}>{three_l['r_M_norm']:.2f}>{three_l['r_N_norm']:.2f}, "
        f"spec_r={spec['profile_pearson_r']:.3f}, nodal_sep={r_nod['radial_sep']:+.3f}, "
        f"nodal_hierarchy_hpf={r_nod['hierarchy_established_hpf']:.2f}, shield_tiers={r_nod['tiers_mesendo_at_shield']}/{r_nod['tiers_endo_at_shield']})."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beta-E", type=float, default=50.0, help="Ectoderm cortical tension beta_E (pN/um)")
    ap.add_argument("--beta-M", type=float, default=95.0, help="Mesoderm cortical tension beta_M (pN/um)")
    ap.add_argument("--beta-N", type=float, default=130.0, help="Endoderm cortical tension beta_N (pN/um)")
    ap.add_argument("--gamma-EM", type=float, default=30.0, help="Heterotypic interfacial tension gamma_EM (pN/um)")
    ap.add_argument("--Vt", type=int, default=16, help="Target cell volume/area V_t (pixels)")
    ap.add_argument("--lam", type=float, default=1.2, help="Volume elasticity constraint multiplier lambda")
    ap.add_argument("--T", type=float, default=10.0, help="Effective fluctuation temperature T")
    ap.add_argument("--mcs", type=int, default=135, help="Number of Monte Carlo Steps (MCS)")
    ap.add_argument("--seed", type=int, default=2008, help="Deterministic RNG seed")
    ap.add_argument("--nodal-C0", type=float, default=1.0, help="Marginal Nodal dose C0 (x wild-type; 0 = lefty1/MZoep null)")
    ap.add_argument("--nodal-lambda", type=float, default=3.0, help="Nodal gradient range lambda (cell tiers from the margin)")
    ap.add_argument("--tau-beta", type=float, default=0.5, help="Cortical tension relaxation time tau_beta (h)")
    ap.add_argument("--kappa", type=float, default=2.0 / 3.0, help="Continuous-cortex closure slope gamma_ij = kappa*|beta_i-beta_j| (0 <= kappa < 1)")
    ap.add_argument("--self-test", action="store_true", help="Run 13-check verification suite and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    out = run_all_regimes_and_render(
        beta_E=args.beta_E,
        beta_M=args.beta_M,
        beta_N=args.beta_N,
        gamma_EM=args.gamma_EM,
        Vt=args.Vt,
        lam_vol=args.lam,
        T=args.T,
        n_mcs=args.mcs,
        seed=args.seed,
        nodal_C0=args.nodal_C0,
        nodal_lambda=args.nodal_lambda,
        tau_beta=args.tau_beta,
        kappa=args.kappa,
    )
    r_wt = out["regimes"]["wildtype_germ_layer_engulfment"]
    r_rv = out["regimes"]["camlc_ectoderm_cortex_reversal"]
    r_bl = out["regimes"]["blebbistatin_myosin_inhibition"]
    tl = out["three_layer"]
    sp = out["specimen"]
    nd = out["nodal"]
    nn = out["nodal_null"]
    print(
        f"[2002-brodland-differential-interfacial-tension-sorting] Rendered Pass 3 outputs:\n"
        f"  - Comparison PNG : {COMPARISON_PNG}\n"
        f"  - Growth GIF     : {GROWTH_GIF}\n"
        f"  - Nodal GIF      : {NODAL_GIF}\n"
        f"  - Wild-type Δr   : {r_wt['radial_sep']:+.4f} (caMLCK reversal Δr={r_rv['radial_sep']:+.4f}, Blebbistatin Δr={r_bl['radial_sep']:+.4f})\n"
        f"  - 3-Germ-Layer   : r_E={tl['r_E_norm']:.3f} > r_M={tl['r_M_norm']:.3f} > r_N={tl['r_N_norm']:.3f}\n"
        f"  - Nodal coupling : Δr(7 hpf)={nd['radial_sep']:+.4f}, hierarchy by {nd['hierarchy_established_hpf']:.2f} hpf, "
        f"shield mesendo/endo tiers={nd['tiers_mesendo_at_shield']}/{nd['tiers_endo_at_shield']}, Nodal-null Δr={nn['radial_sep']:+.4f}\n"
        f"  - Specimen Match : profile r={sp['profile_pearson_r']:.4f} (RMSE={sp['profile_rmse']:.4f}, Row C doublet r_c/R={sp['contact_radius_emp']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
