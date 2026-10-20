#!/usr/bin/env python3
"""
1959-murray-vascular-cube-law-branching.py — Pass 3 (Remediated Closed-Loop Reproduction)
=========================================================================================
Pass 3 additions (closing the two Pass 2 remediation items):
  * Section 4b — Fahraeus-Lindqvist apparent viscosity eta(r) (Pries, Neuhaus & Gaehtgens
    1992 in-vitro fit, H_D = 0.45) in the Poiseuille term; generalised stationarity
    2 P_met = (4 - epsilon) P_visc, local optimal exponent gamma_eff(r), symmetric junction
    exponent gamma_j(25 um) and a Newtonian-vs-F-L 7-generation exponent sweep quantifying the
    shift of the power-minimising exponent from 3.0 toward ~2.77.
  * Section 4c — 3D hemispherical retinal-cup geometry (globe radius 12 mm): arterial and
    venous Murray trees grown by geodesic steps with tangent-plane bifurcation rotations,
    interdigitated arcade roots, AVR-coupled venous root radius, a repulsive A/V growth
    coupling (with an uncoupled ablation control), fundus-projection foreshortening of the
    bifurcation angles, and projected A/V crossing counts.  This is a reduced geometric model
    (no haemodynamic network solve, no VEGF/oxygen-guided angiogenesis); the terminal-tip
    interdigitation index stays near random-mixing level and is carried as a residual.
  * Additional sources: Pries et al. (1992) Am. J. Physiol. 263:H1770 (10.1152/ajpheart.1992.263.6.H1770),
    Fahraeus & Lindqvist (1931) Am. J. Physiol. 96:562 (10.1152/ajplegacy.1931.96.3.562),
    Hubbard et al. (1999) Ophthalmology 106:2269 (10.1016/S0161-6420(99)90525-0) -- DOIs recalled,
    not verified online in this round.
  * Lean 4 Laws 4-5: `fahraeus_lindqvist_stationarity_shift`, `arteriovenous_coupled_cube_balance`.

Mathematical Morphogenesis & Optimality Reproduction for Slot 052:
  1. Murray CD (1926a). "The physiological principle of minimum work. I. The vascular
     system and the cost of blood volume." Proc. Natl. Acad. Sci. USA 12(3):207-214.
     doi:10.1073/pnas.12.3.207
  2. Murray CD (1926b). "The physiological principle of minimum work applied to the
     angle of branching of arteries." J. Gen. Physiol. 9(6):835-841.
     doi:10.1085/jgp.9.6.835
  3. Zamir M (1976). "Optimality principles in arterial branching." J. Theor. Biol.
     62(1):227-251. doi:10.1016/0022-5193(76)90058-8
  4. Sherman TF (1981). "On connecting large vessels to small. The meaning of Murray's
     law." J. Gen. Physiol. 78(4):431-453. doi:10.1085/jgp.78.4.431

Formal Lean 4 Machine-Checked Proofs (`morphoeval/lean/loops/Loop_052_vascular_cube_law_branching.lean`):
  - Law 1 (`MathOfBiology.Loop_052.murray_stationarity_viscous_twice_metabolic`):
    Stationarity dP/dr = 0 of P(r) = (8*eta*L*Q^2)/(pi*r^4) + b*pi*r^2*L occurs where
    metabolic volume power P_met(r*) = 2 * P_visc(r*), yielding Q = (pi/4)*sqrt(b/eta)*r^3
    and scale-invariant wall shear stress tau_w = sqrt(eta*b).
  - Law 2 (`MathOfBiology.Loop_052.murray_cube_law_generational_invariance`):
    Flow conservation Q_0 = Q_1 + Q_2 under Q(r) = k_M * r^3 implies Murray's cube law
    r_0^3 = r_1^3 + r_2^3 and multi-generation cubic sum invariance across arterial trees.
  - Law 3 (`MathOfBiology.Loop_052.murray_bifurcation_cosine_symmetry`):
    Optimal branching cosine numerators N_1 = r_0^4 + r_1^4 - r_2^4 and
    N_2 = r_0^4 + r_2^4 - r_1^4 cancel daughter asymmetry in sum (N_1 + N_2 = 2*r_0^4)
    and yield symmetric angles theta_1 = theta_2 = arccos(2^(-1/3)) = 37.4673 deg
    (total bifurcation angle theta_1 + theta_2 = 74.9346 deg) when r_1 = r_2.
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
from scipy import stats
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "vascular_cube_law_branching"
REF_JSON_PATH = DATASET_DIR / "vascular_cube_law_branching_reference.json"
SPECIMEN_IMG_PATH = DATASET_DIR / "retinal_vasculature_specimen.jpg"
MEDIA_DIR = ROOT / "docs" / "media"
OUTPUT_PNG = MEDIA_DIR / "vascular_cube_law_branching_comparison.png"
OUTPUT_GIF = MEDIA_DIR / "vascular_cube_law_branching_growth.gif"
OUTPUT_CUP_GIF = MEDIA_DIR / "vascular_cube_law_branching_retinal_cup_3d.gif"

EXPECTED_SHA256: dict[Path, str] = {
    SPECIMEN_IMG_PATH: "bc548d4150e070a509caa34cf5cc04d46650fa4fbe15f460f952ca54781dde76",
}

ETA_POISE_DEFAULT = 0.035        # Dynamic blood viscosity eta (dyn * s / cm^2 = Poise), Murray (1926a) p. 209
METABOLIC_B_DEFAULT = 1930.0     # Metabolic cost of blood volume b (erg / s / cm^3), Murray (1926a) p. 210
R0_ROOT_UM_DEFAULT = 82.0        # Human central retinal artery root radius (um), Sherman (1981) Table I
GAMMA_MURRAY = 3.0               # Murray's cubic exponent gamma = 3
GAMMA_DA_VINCI = 2.0             # Da Vinci area-preserving ablation exponent gamma = 2


# =============================================================================
# Section 1: Dataset Provenance & Cryptographic SHA-256 Verification
# =============================================================================

def verify_dataset_integrity() -> dict[str, Any]:
    """Load `vascular_cube_law_branching_reference.json` and verify SHA-256 digests
    of the vendored human retinal vasculature fundus specimen (`retinal_vasculature_specimen.jpg`)."""
    if not REF_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing reference dataset: {REF_JSON_PATH}")
    ref_data = json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))

    for path, expected_hex in EXPECTED_SHA256.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing vendored specimen file: {path}")
        actual_hex = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hex != expected_hex:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {expected_hex}, got {actual_hex}"
            )

    spec_meta = ref_data["metadata"]["specimens"]["retinal_vasculature_fundus"]
    if spec_meta["sha256"] != EXPECTED_SHA256[SPECIMEN_IMG_PATH]:
        raise ValueError("Reference JSON specimen SHA-256 does not match EXPECTED_SHA256.")

    return ref_data


# =============================================================================
# Section 2: Murray (1926a) Poiseuille + Metabolic Cost Functional & Stationarity
#            (Lean 4 Law 1: `murray_stationarity_viscous_twice_metabolic`)
# =============================================================================

def poiseuille_metabolic_power(
    r_cm: np.ndarray | float,
    q_cm3_s: float,
    length_cm: float = 1.0,
    eta: float = ETA_POISE_DEFAULT,
    b: float = METABOLIC_B_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Murray (1926a) PNAS 12:207-214 Eq. (1):
    Total power required to maintain laminar Poiseuille flow Q through a cylindrical
    vessel of radius r and length L plus the metabolic maintenance cost of blood volume:
        P(r) = P_visc(r) + P_met(r) = (8 * eta * L * Q^2) / (pi * r^4) + b * pi * r^2 * L
    """
    r = np.asarray(r_cm, dtype=float)
    p_visc = (8.0 * eta * length_cm * (q_cm3_s ** 2)) / (math.pi * (r ** 4))
    p_met = b * math.pi * (r ** 2) * length_cm
    return p_visc, p_met, p_visc + p_met


def murray_conductance_constant(eta: float = ETA_POISE_DEFAULT, b: float = METABOLIC_B_DEFAULT) -> float:
    """Murray (1926a) Eq. (4):
    Stationarity dP/dr = -32*eta*L*Q^2/(pi*r^5) + 2*b*pi*r*L = 0 implies:
        Q = k_M * r^3,   where k_M = (pi / 4) * sqrt(b / eta).
    """
    return float((math.pi / 4.0) * math.sqrt(b / eta))


def murray_optimal_radius(
    q_cm3_s: float,
    eta: float = ETA_POISE_DEFAULT,
    b: float = METABOLIC_B_DEFAULT,
) -> float:
    """Analytical minimum-work radius r* = (Q / k_M)^(1/3) = ((16 * eta * Q^2) / (pi^2 * b))^(1/6)."""
    return float(((16.0 * eta * (q_cm3_s ** 2)) / ((math.pi ** 2) * b)) ** (1.0 / 6.0))


def compute_wall_shear_stress(
    r_cm: np.ndarray | float,
    q_cm3_s: np.ndarray | float,
    eta: float = ETA_POISE_DEFAULT,
) -> np.ndarray | float:
    """Hagen-Poiseuille lumen wall shear stress (Zamir 1976 Eq. 9; Sherman 1981 p. 434):
        tau_w = (4 * eta * Q) / (pi * r^3).
    Under Murray's law Q = (pi/4)*sqrt(b/eta)*r^3, wall shear stress is strictly
    scale-invariant across all vessel radii: tau_w = sqrt(eta * b).
    """
    return (4.0 * eta * np.asarray(q_cm3_s, dtype=float)) / (math.pi * (np.asarray(r_cm, dtype=float) ** 3))


# =============================================================================
# Section 3: Murray (1926b) & Zamir (1976) Asymmetric Bifurcation Angle Laws
#            (Lean 4 Law 2 & Law 3: `murray_cube_law_generational_invariance`,
#             `murray_bifurcation_cosine_symmetry`)
# =============================================================================

def murray_asymmetric_bifurcation(
    r0: float,
    alpha: float = 1.0,
    gamma: float = GAMMA_MURRAY,
    optimality_mode: str = "murray_volume",
) -> dict[str, float]:
    """Murray (1926b) J. Gen. Physiol. 9:835-841 Eqs. (3)-(6) & Zamir (1976) Table 1:
    Given parent radius r0, daughter radius asymmetry ratio alpha = r2 / r1 in (0, 1],
    and junction exponent gamma (r0^gamma = r1^gamma + r2^gamma):
      - Daughter radii:
          r1 = r0 / (1 + alpha^gamma)^(1/gamma),   r2 = alpha * r1
      - Optimal deflection angles (theta_1, theta_2) from parent vessel axis:
        * 'murray_volume' (Murray 1926b minimum pumping + volume cost, cost_i ~ r_i^2):
            cos(theta_1) = (r0^4 + r1^4 - r2^4) / (2 * r0^2 * r1^2)
            cos(theta_2) = (r0^4 + r2^4 - r1^4) / (2 * r0^2 * r2^2)
            cos(theta_1 + theta_2) = (r0^4 - r1^4 - r2^4) / (2 * r1^2 * r2^2)
        * 'zamir_surface' (Zamir 1976 minimum lumen surface / endothelial drag, cost_i ~ r_i):
            cos(theta_1) = (r0^3 + r1^3 - r2^3) / (2 * r0^(3/2) * r1^(3/2))
            cos(theta_2) = (r0^3 + r2^3 - r1^3) / (2 * r0^(3/2) * r2^(3/2))
        * 'steiner_length' (Minimum total vessel length / Steiner tree, cost_i ~ 1):
            cos(theta_1) = 0.5,  cos(theta_2) = 0.5  (theta_1 + theta_2 = 120 deg)
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"Asymmetry ratio alpha = r2/r1 must be in (0, 1], got {alpha}")
    denom = (1.0 + (alpha ** gamma)) ** (1.0 / gamma)
    r1 = float(r0 / denom)
    r2 = float(alpha * r1)

    if optimality_mode == "murray_volume":
        cos1 = (r0 ** 4 + r1 ** 4 - r2 ** 4) / (2.0 * (r0 ** 2) * (r1 ** 2))
        cos2 = (r0 ** 4 + r2 ** 4 - r1 ** 4) / (2.0 * (r0 ** 2) * (r2 ** 2))
    elif optimality_mode == "zamir_surface":
        cos1 = (r0 ** 3 + r1 ** 3 - r2 ** 3) / (2.0 * (r0 ** 1.5) * (r1 ** 1.5))
        cos2 = (r0 ** 3 + r2 ** 3 - r1 ** 3) / (2.0 * (r0 ** 1.5) * (r2 ** 1.5))
    elif optimality_mode == "steiner_length":
        cos1 = 0.5
        cos2 = 0.5
    else:
        raise ValueError(f"Unknown optimality_mode: {optimality_mode}")

    theta1_deg = math.degrees(math.acos(float(np.clip(cos1, -1.0, 1.0))))
    theta2_deg = math.degrees(math.acos(float(np.clip(cos2, -1.0, 1.0))))
    area_beta = (r1 ** 2 + r2 ** 2) / (r0 ** 2)
    cubic_res = abs((r0 ** gamma) - (r1 ** gamma + r2 ** gamma))

    return {
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "alpha": alpha,
        "gamma": gamma,
        "r1_over_r0": r1 / r0,
        "r2_over_r0": r2 / r0,
        "daughter_ratio": r1 / r0,
        "area_expansion_beta": area_beta,
        "theta1_deg": theta1_deg,
        "theta2_deg": theta2_deg,
        "total_angle_deg": theta1_deg + theta2_deg,
        "cubic_residual": cubic_res,
    }


def murray_symmetric_bifurcation(r0: float) -> dict[str, float]:
    """Symmetric special case alpha = 1.0, gamma = 3.0 of `murray_asymmetric_bifurcation`."""
    return murray_asymmetric_bifurcation(r0=r0, alpha=1.0, gamma=GAMMA_MURRAY, optimality_mode="murray_volume")


def evaluate_empirical_retinal_junctions(ref_data: dict[str, Any]) -> dict[str, Any]:
    """Compare Murray's (1926b) theoretical bifurcation angle predictions against the
    empirical human retinal arterial junctions J01-J08 (Zamir et al. 1979 / Sherman 1981)."""
    junctions = ref_data["empirical_retinal_junctions_zamir_1979"]
    obs_total = np.array([j["total_angle_obs_deg"] for j in junctions], dtype=float)
    obs_theta1 = np.array([j["theta1_obs_deg"] for j in junctions], dtype=float)
    obs_theta2 = np.array([j["theta2_obs_deg"] for j in junctions], dtype=float)
    r0_arr = np.array([j["r0_um"] for j in junctions], dtype=float)
    r1_arr = np.array([j["r1_um"] for j in junctions], dtype=float)
    r2_arr = np.array([j["r2_um"] for j in junctions], dtype=float)

    pred_total = []
    pred_theta1 = []
    pred_theta2 = []
    for j in junctions:
        b = murray_asymmetric_bifurcation(j["r0_um"], alpha=j["alpha_obs"], gamma=GAMMA_MURRAY)
        pred_total.append(b["total_angle_deg"])
        pred_theta1.append(b["theta1_deg"])
        pred_theta2.append(b["theta2_deg"])

    pred_total_arr = np.array(pred_total, dtype=float)
    pred_theta1_arr = np.array(pred_theta1, dtype=float)
    pred_theta2_arr = np.array(pred_theta2, dtype=float)

    # Cube-law radius conservation ratio (r1^3 + r2^3)^(1/3) / r0
    cube_ratio = ((r1_arr ** 3 + r2_arr ** 3) ** (1.0 / 3.0)) / r0_arr
    # Da Vinci square-law ratio (r1^2 + r2^2)^(1/2) / r0 (ablation comparison)
    davinci_ratio = np.sqrt(r1_arr ** 2 + r2_arr ** 2) / r0_arr

    rmse_total = float(np.sqrt(np.mean((obs_total - pred_total_arr) ** 2)))
    rmse_branch = float(np.sqrt(np.mean(
        np.concatenate([(obs_theta1 - pred_theta1_arr) ** 2, (obs_theta2 - pred_theta2_arr) ** 2])
    )))
    r_val, p_val = stats.pearsonr(
        np.concatenate([obs_theta1, obs_theta2]),
        np.concatenate([pred_theta1_arr, pred_theta2_arr]),
    )
    r_tot, p_tot = stats.pearsonr(obs_total, pred_total_arr)

    return {
        "n_junctions": len(junctions),
        "rmse_total_angle_deg": rmse_total,
        "rmse_branch_angle_deg": rmse_branch,
        "pearson_r_branch_angles": float(r_val),
        "pearson_p_branch_angles": float(p_val),
        "pearson_r_total_angle": float(r_tot),
        "pearson_p_total_angle": float(p_tot),
        "cube_law_ratio_mean": float(np.mean(cube_ratio)),
        "cube_law_ratio_std": float(np.std(cube_ratio, ddof=1)),
        "cube_law_ratio_cv": float(np.std(cube_ratio, ddof=1) / np.mean(cube_ratio)),
        "davinci_ratio_mean": float(np.mean(davinci_ratio)),
        "obs_total_deg": obs_total,
        "pred_total_deg": pred_total_arr,
        "r0_um": r0_arr,
        "r1_um": r1_arr,
        "r2_um": r2_arr,
    }


# =============================================================================
# Section 4: Multi-Generation Vascular Tree Builder & Exponent Ablation Sweep
# =============================================================================

def evaluate_generational_hierarchy(
    generations: int = 7,
    r0_um: float = R0_ROOT_UM_DEFAULT,
    gamma: float = GAMMA_MURRAY,
    l0_cm: float = 0.25,
    eta: float = ETA_POISE_DEFAULT,
    b: float = METABOLIC_B_DEFAULT,
) -> dict[str, Any]:
    """Evaluate a multi-generation symmetric binary arterial tree with radius scaling
    r_g = r_0 * 2^(-g / gamma) and flow conservation Q_g = Q_0 * 2^(-g), where root
    flow Q_0 = k_M * r_0^3 is fixed by the physiological tissue perfusion demand."""
    k_m = murray_conductance_constant(eta=eta, b=b)
    r0_cm = r0_um * 1e-4
    q0_cm3_s = k_m * (r0_cm ** 3)

    rows: list[dict[str, float]] = []
    total_visc_power = 0.0
    total_met_power = 0.0

    for g in range(generations):
        n_vessels = 2 ** g
        r_g_cm = r0_cm * (2.0 ** (-float(g) / gamma))
        q_g_cm3_s = q0_cm3_s / float(n_vessels)
        l_g_cm = l0_cm * (2.0 ** (-float(g) / 3.0))
        p_v, p_m, _ = poiseuille_metabolic_power(r_g_cm, q_g_cm3_s, length_cm=l_g_cm, eta=eta, b=b)
        tau_g = float(compute_wall_shear_stress(r_g_cm, q_g_cm3_s, eta=eta))

        gen_visc = float(n_vessels * p_v)
        gen_met = float(n_vessels * p_m)
        total_visc_power += gen_visc
        total_met_power += gen_met

        rows.append({
            "generation": float(g),
            "n_vessels": float(n_vessels),
            "radius_um": float(r_g_cm * 1e4),
            "flow_nl_s": float(q_g_cm3_s * 1e6),
            "wall_shear_dyn_cm2": tau_g,
            "total_cross_area_um2": float(n_vessels * math.pi * ((r_g_cm * 1e4) ** 2)),
            "gen_power_erg_s": gen_visc + gen_met,
        })

    taus = np.array([r["wall_shear_dyn_cm2"] for r in rows], dtype=float)
    return {
        "gamma": gamma,
        "generations": generations,
        "rows": rows,
        "total_visc_power_erg_s": total_visc_power,
        "total_met_power_erg_s": total_met_power,
        "total_tree_power_erg_s": total_visc_power + total_met_power,
        "wall_shear_mean_dyn_cm2": float(np.mean(taus)),
        "wall_shear_cv": float(np.std(taus) / np.mean(taus)),
        "terminal_to_root_shear_ratio": float(taus[-1] / taus[0]),
    }


def run_gamma_power_sweep(
    gamma_grid: np.ndarray | None = None,
    generations: int = 7,
    r0_um: float = R0_ROOT_UM_DEFAULT,
) -> dict[str, Any]:
    """Sweep branching exponent gamma in [1.8, 4.2] across a 7-generation arterial tree
    to demonstrate that Murray's cube law (gamma = 3.0) uniquely minimizes total
    Poiseuille + metabolic vascular work."""
    if gamma_grid is None:
        gamma_grid = np.linspace(1.8, 4.2, 121)
    powers = []
    visc_powers = []
    met_powers = []
    for gm in gamma_grid:
        res = evaluate_generational_hierarchy(generations=generations, r0_um=r0_um, gamma=float(gm))
        powers.append(res["total_tree_power_erg_s"])
        visc_powers.append(res["total_visc_power_erg_s"])
        met_powers.append(res["total_met_power_erg_s"])

    powers_arr = np.array(powers, dtype=float)
    min_idx = int(np.argmin(powers_arr))
    gamma_opt = float(gamma_grid[min_idx])
    p_murray = evaluate_generational_hierarchy(generations=generations, r0_um=r0_um, gamma=GAMMA_MURRAY)["total_tree_power_erg_s"]
    p_davinci = evaluate_generational_hierarchy(generations=generations, r0_um=r0_um, gamma=GAMMA_DA_VINCI)["total_tree_power_erg_s"]
    p_quartic = evaluate_generational_hierarchy(generations=generations, r0_um=r0_um, gamma=4.0)["total_tree_power_erg_s"]

    return {
        "gamma_grid": gamma_grid,
        "total_power_erg_s": powers_arr,
        "visc_power_erg_s": np.array(visc_powers, dtype=float),
        "met_power_erg_s": np.array(met_powers, dtype=float),
        "gamma_opt": gamma_opt,
        "power_murray_erg_s": p_murray,
        "power_davinci_erg_s": p_davinci,
        "power_quartic_erg_s": p_quartic,
        "davinci_excess_power_pct": float(100.0 * (p_davinci - p_murray) / p_murray),
        "quartic_excess_power_pct": float(100.0 * (p_quartic - p_murray) / p_murray),
    }


def build_vascular_tree_2d(
    generations: int = 6,
    r0_um: float = R0_ROOT_UM_DEFAULT,
    l0_mm: float = 2.0,
    alpha: float = 1.0,
    gamma: float = GAMMA_MURRAY,
    optimality_mode: str = "murray_volume",
) -> list[dict[str, float]]:
    """Build a 2D vascular tree for any daughter asymmetry alpha = r2/r1, scaling
    exponent gamma, and angle optimality regime ('murray_volume', 'zamir_surface',
    or 'steiner_length')."""
    segments: list[dict[str, float]] = []

    def _grow(x0: float, y0: float, heading: float, r_um: float, length_mm: float, gen: int, parity: int) -> None:
        x1 = x0 + length_mm * math.cos(heading)
        y1 = y0 + length_mm * math.sin(heading)
        segments.append({
            "gen": float(gen),
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "r_um": r_um,
            "length_mm": length_mm,
        })
        if gen < generations:
            bif = murray_asymmetric_bifurcation(r_um, alpha=alpha, gamma=gamma, optimality_mode=optimality_mode)
            th1 = math.radians(bif["theta1_deg"])
            th2 = math.radians(bif["theta2_deg"])
            l1 = length_mm * (bif["r1_over_r0"] ** 0.85)
            l2 = length_mm * (bif["r2_over_r0"] ** 0.85)
            # Alternate side-branch handedness by parity so asymmetric trees fill space naturally
            if parity >= 0:
                _grow(x1, y1, heading + th1, bif["r1"], l1, gen + 1, -1)
                _grow(x1, y1, heading - th2, bif["r2"], l2, gen + 1, 1)
            else:
                _grow(x1, y1, heading - th1, bif["r1"], l1, gen + 1, 1)
                _grow(x1, y1, heading + th2, bif["r2"], l2, gen + 1, -1)

    _grow(0.0, 0.0, math.pi / 2.0, r0_um, l0_mm, 0, 1)
    return segments


# =============================================================================
# Section 4b (Pass 3): Non-Newtonian Fahraeus-Lindqvist Apparent Viscosity eta(r)
#            and the Resulting Shift of the Optimal Exponent Below gamma = 3
#            (Lean 4 Law 4: `fahraeus_lindqvist_stationarity_shift`)
# =============================================================================

FL_HEMATOCRIT_DEFAULT = 0.45          # Discharge hematocrit H_D of the Pries et al. (1992) in-vitro fit
FL_ETA_BULK_REL = 3.2                 # Large-tube asymptote of the relative apparent viscosity (Pries 1992 Eq. 1)
FL_PRECAPILLARY_RADIUS_UM = 25.0      # Pre-capillary arteriole radius threshold named in the Pass 2 remediation item


def fahraeus_lindqvist_relative_viscosity(diameter_um: np.ndarray | float) -> np.ndarray:
    """Pries, Neuhaus & Gaehtgens (1992) Am. J. Physiol. 263:H1770-H1778, Eq. (1)
    (in-vitro glass-tube fit at discharge hematocrit H_D = 0.45, D in micrometres):
        eta_rel(D) = 220 * exp(-1.3 * D) + 3.2 - 2.44 * exp(-0.06 * D^0.645)
    This is the empirical Fahraeus-Lindqvist (1931) effect: apparent blood viscosity falls
    from the bulk value (eta_rel -> 3.2 for D >> 100 um) to a minimum of ~1.3 near D ~ 7 um
    because of axial red-cell migration and the cell-free plasma layer.  The 220*exp(-1.3 D)
    term is the sub-capillary inversion (D < ~4 um) and is irrelevant for arterioles."""
    d = np.asarray(diameter_um, dtype=float)
    return 220.0 * np.exp(-1.3 * d) + 3.2 - 2.44 * np.exp(-0.06 * np.power(d, 0.645))


def apparent_viscosity_poise(
    r_cm: np.ndarray | float,
    eta_bulk: float = ETA_POISE_DEFAULT,
    viscosity_model: str = "fahraeus_lindqvist",
) -> np.ndarray:
    """Radius-dependent apparent dynamic viscosity eta(r) in Poise.
      - 'newtonian':            eta(r) = eta_bulk (Murray 1926a constant-viscosity assumption)
      - 'fahraeus_lindqvist':   eta(r) = eta_bulk * eta_rel(2 r) / eta_rel(inf), so that the
                                large-vessel limit reproduces Murray's eta_bulk = 0.035 P exactly
                                and only the *shape* of the Pries (1992) curve enters."""
    r = np.asarray(r_cm, dtype=float)
    if viscosity_model == "newtonian":
        return np.full_like(r, eta_bulk, dtype=float)
    if viscosity_model == "fahraeus_lindqvist":
        d_um = 2.0 * r * 1e4
        return eta_bulk * fahraeus_lindqvist_relative_viscosity(d_um) / FL_ETA_BULK_REL
    raise ValueError(f"Unknown viscosity_model: {viscosity_model}")


def viscosity_log_slope(r_cm: np.ndarray | float, eta_bulk: float = ETA_POISE_DEFAULT,
                        viscosity_model: str = "fahraeus_lindqvist", h: float = 1e-4) -> np.ndarray:
    """Logarithmic sensitivity epsilon(r) = d ln eta / d ln r (central difference in ln r).
    epsilon = 0 for Newtonian blood; epsilon > 0 across the arteriolar range because eta
    *decreases* toward smaller vessels under the Fahraeus-Lindqvist effect."""
    r = np.asarray(r_cm, dtype=float)
    lr = np.log(r)
    e_p = np.log(apparent_viscosity_poise(np.exp(lr + h), eta_bulk, viscosity_model))
    e_m = np.log(apparent_viscosity_poise(np.exp(lr - h), eta_bulk, viscosity_model))
    return (e_p - e_m) / (2.0 * h)


def fl_optimal_flow(r_cm: np.ndarray | float, eta_bulk: float = ETA_POISE_DEFAULT,
                    b: float = METABOLIC_B_DEFAULT, viscosity_model: str = "fahraeus_lindqvist") -> np.ndarray:
    """Generalised Murray stationarity with radius-dependent viscosity.
    P(r) = 8 L Q^2 eta(r) / (pi r^4) + b pi r^2 L.  Setting dP/dr = 0 with
    epsilon = d ln eta / d ln r gives (Lean 4 Law 4):
        2 P_met(r*) = (4 - epsilon) P_visc(r*)
        Q*(r)       = (pi / 2) r^3 sqrt( b / (eta(r) (4 - epsilon(r))) )
    For epsilon = 0 this collapses to Murray's Q = (pi/4) sqrt(b/eta) r^3 (P_met = 2 P_visc)."""
    r = np.asarray(r_cm, dtype=float)
    eta_r = apparent_viscosity_poise(r, eta_bulk, viscosity_model)
    eps = viscosity_log_slope(r, eta_bulk, viscosity_model)
    return (math.pi / 2.0) * (r ** 3) * np.sqrt(b / (eta_r * (4.0 - eps)))


def fl_local_optimal_exponent(r_cm: np.ndarray | float, eta_bulk: float = ETA_POISE_DEFAULT,
                              b: float = METABOLIC_B_DEFAULT, viscosity_model: str = "fahraeus_lindqvist",
                              h: float = 1e-3) -> np.ndarray:
    """Local optimal exponent gamma_eff(r) = d ln Q* / d ln r along the stationarity curve.
    gamma_eff = 3 identically for Newtonian blood; gamma_eff < 3 wherever epsilon(r) > 0."""
    r = np.asarray(r_cm, dtype=float)
    lr = np.log(r)
    q_p = np.log(fl_optimal_flow(np.exp(lr + h), eta_bulk, b, viscosity_model))
    q_m = np.log(fl_optimal_flow(np.exp(lr - h), eta_bulk, b, viscosity_model))
    return (q_p - q_m) / (2.0 * h)


def fl_symmetric_junction_exponent(r0_um: float, eta_bulk: float = ETA_POISE_DEFAULT,
                                   b: float = METABOLIC_B_DEFAULT,
                                   viscosity_model: str = "fahraeus_lindqvist") -> dict[str, float]:
    """Effective junction exponent of a *symmetric* bifurcation whose parent and daughters all sit
    on the generalised stationarity curve Q*(r): solve Q*(r1) = Q*(r0)/2 for r1 by bisection and
    report gamma_j = ln 2 / ln(r0 / r1)  (so that r0^gamma_j = 2 r1^gamma_j exactly)."""
    r0 = r0_um * 1e-4
    target = float(fl_optimal_flow(r0, eta_bulk, b, viscosity_model)) / 2.0
    lo, hi = 0.30 * r0, r0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(fl_optimal_flow(mid, eta_bulk, b, viscosity_model)) > target:
            hi = mid
        else:
            lo = mid
    r1 = 0.5 * (lo + hi)
    gamma_j = math.log(2.0) / math.log(r0 / r1)
    pv, _, _ = poiseuille_metabolic_power(r0, target * 2.0, eta=float(apparent_viscosity_poise(r0, eta_bulk, viscosity_model)), b=b)
    pm = b * math.pi * r0 ** 2
    return {
        "r0_um": r0_um,
        "r1_um": r1 * 1e4,
        "r1_over_r0": r1 / r0,
        "gamma_junction": float(gamma_j),
        "eta_rel_parent": float(fahraeus_lindqvist_relative_viscosity(2.0 * r0_um)) if viscosity_model == "fahraeus_lindqvist" else FL_ETA_BULK_REL,
        "epsilon_parent": float(viscosity_log_slope(r0, eta_bulk, viscosity_model)),
        "met_over_visc_parent": float(pm / pv),
    }


def evaluate_generational_hierarchy_fl(
    generations: int = 7,
    r0_um: float = R0_ROOT_UM_DEFAULT,
    gamma: float = GAMMA_MURRAY,
    l0_cm: float = 0.25,
    eta_bulk: float = ETA_POISE_DEFAULT,
    b: float = METABOLIC_B_DEFAULT,
    viscosity_model: str = "fahraeus_lindqvist",
) -> dict[str, Any]:
    """Same symmetric tree as `evaluate_generational_hierarchy` but the Poiseuille term of each
    generation uses the apparent viscosity eta(r_g) of that generation's radius.  Root flow is
    fixed at Q_0 = k_M r_0^3 (Newtonian Murray value) so trees of different gamma and viscosity
    model deliver identical perfusion and differ only in cost."""
    k_m = murray_conductance_constant(eta=eta_bulk, b=b)
    r0_cm = r0_um * 1e-4
    q0 = k_m * (r0_cm ** 3)
    tot_v = tot_m = 0.0
    radii_um, etas_rel, taus = [], [], []
    for g in range(generations):
        n = 2 ** g
        r_g = r0_cm * (2.0 ** (-float(g) / gamma))
        q_g = q0 / n
        l_g = l0_cm * (2.0 ** (-float(g) / 3.0))
        eta_g = float(apparent_viscosity_poise(r_g, eta_bulk, viscosity_model))
        p_v, p_m, _ = poiseuille_metabolic_power(r_g, q_g, length_cm=l_g, eta=eta_g, b=b)
        tot_v += n * float(p_v)
        tot_m += n * float(p_m)
        radii_um.append(r_g * 1e4)
        etas_rel.append(eta_g / eta_bulk * FL_ETA_BULK_REL)
        taus.append(float(compute_wall_shear_stress(r_g, q_g, eta=eta_g)))
    return {
        "gamma": gamma,
        "viscosity_model": viscosity_model,
        "radii_um": np.array(radii_um),
        "eta_rel": np.array(etas_rel),
        "wall_shear_dyn_cm2": np.array(taus),
        "total_visc_power_erg_s": tot_v,
        "total_met_power_erg_s": tot_m,
        "total_tree_power_erg_s": tot_v + tot_m,
    }


def run_fl_exponent_sweep(
    gamma_grid: np.ndarray | None = None,
    generations: int = 7,
    r0_um: float = R0_ROOT_UM_DEFAULT,
    eta_bulk: float = ETA_POISE_DEFAULT,
    b: float = METABOLIC_B_DEFAULT,
) -> dict[str, Any]:
    """Sweep the tree exponent gamma for Newtonian vs Fahraeus-Lindqvist blood and locate the
    power-minimising exponent of each (parabolic refinement around the grid minimum).  Also
    report the local stationarity exponent gamma_eff(r) and the symmetric junction exponent
    gamma_j at the pre-capillary radius r0 = 25 um named in the remediation item."""
    if gamma_grid is None:
        gamma_grid = np.linspace(2.2, 3.6, 141)
    out: dict[str, Any] = {"gamma_grid": gamma_grid}
    for model in ("newtonian", "fahraeus_lindqvist"):
        p = np.array([evaluate_generational_hierarchy_fl(generations, r0_um, float(g), eta_bulk=eta_bulk, b=b,
                                                         viscosity_model=model)["total_tree_power_erg_s"] for g in gamma_grid])
        i = int(np.argmin(p))
        g_opt = float(gamma_grid[i])
        if 0 < i < len(gamma_grid) - 1:  # parabolic refinement
            y0, y1, y2 = p[i - 1], p[i], p[i + 1]
            dg = gamma_grid[1] - gamma_grid[0]
            denom = (y0 - 2.0 * y1 + y2)
            if denom > 0:
                g_opt = float(gamma_grid[i] + 0.5 * dg * (y0 - y2) / denom)
        out[f"power_{model}"] = p
        out[f"gamma_opt_{model}"] = g_opt
    r_loc = np.logspace(math.log10(3.0e-4), math.log10(100.0e-4), 160)  # 3 um .. 100 um radius
    out["r_local_um"] = r_loc * 1e4
    out["gamma_eff_fl"] = fl_local_optimal_exponent(r_loc, eta_bulk, b, "fahraeus_lindqvist")
    out["gamma_eff_newtonian"] = fl_local_optimal_exponent(r_loc, eta_bulk, b, "newtonian")
    out["eta_rel_local"] = fahraeus_lindqvist_relative_viscosity(2.0 * r_loc * 1e4)
    out["junction_25um"] = fl_symmetric_junction_exponent(FL_PRECAPILLARY_RADIUS_UM, eta_bulk, b, "fahraeus_lindqvist")
    out["junction_12um"] = fl_symmetric_junction_exponent(12.0, eta_bulk, b, "fahraeus_lindqvist")
    out["junction_82um"] = fl_symmetric_junction_exponent(R0_ROOT_UM_DEFAULT, eta_bulk, b, "fahraeus_lindqvist")
    pre_cap = out["r_local_um"] < FL_PRECAPILLARY_RADIUS_UM
    out["gamma_eff_precapillary_mean"] = float(np.mean(out["gamma_eff_fl"][pre_cap]))
    out["gamma_eff_precapillary_min"] = float(np.min(out["gamma_eff_fl"][pre_cap]))
    out["gamma_shift"] = out["gamma_opt_fahraeus_lindqvist"] - out["gamma_opt_newtonian"]
    return out


# =============================================================================
# Section 4c (Pass 3): 3D Hemispherical Retinal-Cup Geometry with Coupled,
#            Interdigitated Arterial & Venous Murray Trees
#            (Lean 4 Law 5: `arteriovenous_coupled_cube_balance`)
# =============================================================================

EYE_RADIUS_MM_DEFAULT = 12.0      # Inner radius of the emmetropic human globe (axial length ~24 mm; schematic eye)
RETINA_POLAR_LIMIT_DEG = 95.0     # Growth limit measured from the posterior pole (ora serrata lies at ~100-110 deg)
AVR_DEFAULT = 0.67                # Clinical retinal arteriole-to-venule caliber ratio AVR ~ 2/3 (Hubbard et al. 1999)
VENOUS_ROOT_UM_DEFAULT = R0_ROOT_UM_DEFAULT / AVR_DEFAULT   # Central retinal vein root radius ~ 122 um


def _sphere_point(polar_deg: float, azimuth_deg: float, radius_mm: float) -> np.ndarray:
    th, ph = math.radians(polar_deg), math.radians(azimuth_deg)
    return radius_mm * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def build_vascular_tree_hemisphere(
    generations: int = 6,
    r0_um: float = R0_ROOT_UM_DEFAULT,
    eye_radius_mm: float = EYE_RADIUS_MM_DEFAULT,
    l0_mm: float = 3.2,
    alpha: float = 0.70,
    gamma: float = GAMMA_MURRAY,
    optimality_mode: str = "murray_volume",
    arcade_azimuths_deg: tuple[float, ...] = (45.0, 135.0, 225.0, 315.0),
    disc_polar_deg: float = 4.0,
    polar_limit_deg: float = RETINA_POLAR_LIMIT_DEG,
    vessel_type: str = "artery",
    seed: int | None = None,
    heading_jitter_deg: float = 0.0,
    rival_points: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Grow a Murray tree *on the inner surface of a sphere* (the retinal cup) by geodesic steps.
    State = (point p on sphere, unit tangent t).  A geodesic step of arc length s:
        p' = p cos(s/R) + R t sin(s/R),      t' = -(p/R) sin(s/R) + t cos(s/R)
    A bifurcation rotates the tangent inside the local tangent plane by +-theta about the
    surface normal n = p/R  (Rodrigues:  t_rot = t cos(theta) + (n x t) sin(theta)), so Murray's
    (1926b) optimal angles are realised as *true* angles on the curved retina.  Radii obey the
    cube law exactly (`murray_asymmetric_bifurcation`), lengths scale as l ~ r^0.85 as in the 2D
    builder.  Four arcades start at the optic disc (polar angle `disc_polar_deg`) at the given
    azimuths; the venous tree is built by calling this with `vessel_type='vein'`, a larger root
    radius (AVR coupling) and azimuths rotated by 45 deg so arteries and veins interdigitate.
    Growth stops at `polar_limit_deg` (ora serrata).  Optional deterministic heading jitter
    (seeded) breaks the perfect symmetry for display; the default 0.0 keeps the tree exact."""
    rng = np.random.default_rng(seed) if (seed is not None and heading_jitter_deg > 0.0) else None
    R = eye_radius_mm
    segs: list[dict[str, Any]] = []
    rival_tree = cKDTree(rival_points) if (rival_points is not None and len(rival_points) > 0) else None

    def _rot_tangent(n: np.ndarray, t: np.ndarray, ang: float) -> np.ndarray:
        return t * math.cos(ang) + np.cross(n, t) * math.sin(ang)

    def _grow(p: np.ndarray, t: np.ndarray, r_um: float, length_mm: float, gen: int, parity: int, arcade: int) -> None:
        s = length_mm / R
        p1 = p * math.cos(s) + R * t * math.sin(s)
        t1 = -(p / R) * math.sin(s) + t * math.cos(s)
        polar1 = math.degrees(math.acos(float(np.clip(p1[2] / R, -1.0, 1.0))))
        if polar1 > polar_limit_deg:
            return
        segs.append({
            "gen": gen, "arcade": arcade, "type": vessel_type,
            "p0": p.copy(), "p1": p1.copy(), "t0": t.copy(), "t1": t1.copy(),
            "r_um": r_um, "length_mm": length_mm, "polar_deg": polar1,
        })
        if gen < generations:
            bif = murray_asymmetric_bifurcation(r_um, alpha=alpha, gamma=gamma, optimality_mode=optimality_mode)
            th1, th2 = math.radians(bif["theta1_deg"]), math.radians(bif["theta2_deg"])
            if rng is not None:
                j = math.radians(heading_jitter_deg)
                th1 += float(rng.normal(0.0, j))
                th2 += float(rng.normal(0.0, j))
            l1 = length_mm * (bif["r1_over_r0"] ** 0.85)
            l2 = length_mm * (bif["r2_over_r0"] ** 0.85)
            n1 = p1 / R
            sgn = 1.0 if parity >= 0 else -1.0
            if rival_tree is not None:
                # Arteriovenous repulsive coupling (reduced model of the mutual A/V territory
                # exclusion of retinal development, veins growing into the gaps of the earlier
                # arterial plexus): of the two mirror-image side-branch placements, keep the one
                # whose geodesic end-point is farther from the nearest rival (other-type) vessel
                # node.  Murray's radii and angles are unchanged -- only the handedness is coupled.
                def _rival_gap(t_dir: np.ndarray, length: float) -> float:
                    ss = length / R
                    q = p1 * math.cos(ss) + R * t_dir * math.sin(ss)
                    return float(rival_tree.query(q)[0])
                gap_plus = _rival_gap(_rot_tangent(n1, t1, -sgn * th2), l2)
                gap_minus = _rival_gap(_rot_tangent(n1, t1, sgn * th2), l2)
                if gap_minus > gap_plus:
                    sgn = -sgn
            # Record the true tangent-plane deflections so projection bias can be measured later
            segs[-1]["bif_theta1_deg"] = math.degrees(th1)
            segs[-1]["bif_theta2_deg"] = math.degrees(th2)
            segs[-1]["bif_total_deg"] = math.degrees(th1 + th2)
            _grow(p1, _rot_tangent(n1, t1, sgn * th1), bif["r1"], l1, gen + 1, -parity, arcade)
            _grow(p1, _rot_tangent(n1, t1, -sgn * th2), bif["r2"], l2, gen + 1, parity, arcade)

    for k, az in enumerate(arcade_azimuths_deg):
        p0 = _sphere_point(disc_polar_deg, az, R)
        n0 = p0 / R
        # Initial tangent: direction of increasing polar angle at azimuth az (radially away from the disc)
        e_th = np.array([math.cos(math.radians(disc_polar_deg)) * math.cos(math.radians(az)),
                         math.cos(math.radians(disc_polar_deg)) * math.sin(math.radians(az)),
                         -math.sin(math.radians(disc_polar_deg))])
        e_th = e_th - np.dot(e_th, n0) * n0
        e_th /= np.linalg.norm(e_th)
        _grow(p0, e_th, r0_um, l0_mm, 0, 1 if k % 2 == 0 else -1, k)
    return segs


def _segments_intersect_2d(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Vectorised proper-intersection test for segment sets A (n,2,2) and B (m,2,2) -> (n, m) bool."""
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    A0, A1 = a0[:, None, :], a1[:, None, :]
    B0, B1 = b0[None, :, :], b1[None, :, :]
    d1 = orient(B0, B1, A0)
    d2 = orient(B0, B1, A1)
    d3 = orient(A0, A1, B0)
    d4 = orient(A0, A1, B1)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def evaluate_hemispherical_av_coupling(
    generations: int = 6,
    r0_art_um: float = R0_ROOT_UM_DEFAULT,
    avr: float = AVR_DEFAULT,
    eye_radius_mm: float = EYE_RADIUS_MM_DEFAULT,
    alpha: float = 0.70,
    gamma: float = GAMMA_MURRAY,
    eta_bulk: float = ETA_POISE_DEFAULT,
    b: float = METABOLIC_B_DEFAULT,
    coupling: str = "repulsive",
) -> dict[str, Any]:
    """Couple an arterial and a venous Murray tree on the hemispherical retinal cup and quantify:
      1. Sphericity residual max | |p| - R | of every node (geodesic construction check).
      2. Arteriovenous interdigitation index: order all terminal tips by azimuth about the optic
         axis and report the fraction of azimuthally adjacent tip pairs with opposite vessel type
         (1.0 = perfect A/V alternation, 0.5 = random mixing, 0.0 = fully segregated).
      3. Arteriovenous crossings: number of proper A/V segment crossings in the fundus
         (orthographic z) projection -- the model analogue of clinical AV-crossing sites.
      4. Coupled flow balance: with Q_a = k_a sum r_tip^3 (arterial) and Q_v = k_v sum R_tip^3
         (venous), closed circulation Q_a = Q_v fixes (k_a / k_v) = sum R^3 / sum r^3; combined
         with Murray's tau_w = sqrt(eta b) this gives the venous metabolic constant
         b_v = b_a (k_v / k_a)^2 and venous wall shear tau_v = tau_a (k_v / k_a) (Lean 4 Law 5).
      5. Fundus-projection foreshortening: true tangent-plane bifurcation angle versus its
         apparent angle in the orthographic projection, binned by retinal eccentricity.
    """
    R = eye_radius_mm
    az_a = (45.0, 135.0, 225.0, 315.0)
    az_v = (0.0, 90.0, 180.0, 270.0)
    if coupling not in ("none", "repulsive"):
        raise ValueError(f"Unknown coupling: {coupling}")
    # Arteries are laid down first (as in retinal development); veins then grow coupled to them.
    arteries = build_vascular_tree_hemisphere(generations, r0_art_um, R, alpha=alpha, gamma=gamma,
                                              arcade_azimuths_deg=az_a, vessel_type="artery")
    art_nodes = np.array([s["p1"] for s in arteries] + [s["p0"] for s in arteries])
    veins = build_vascular_tree_hemisphere(generations, r0_art_um / avr, R, alpha=alpha, gamma=gamma, l0_mm=3.2 * 1.06,
                                           arcade_azimuths_deg=az_v, vessel_type="vein",
                                           rival_points=art_nodes if coupling == "repulsive" else None)
    all_segs = arteries + veins

    pts = np.array([s["p1"] for s in all_segs] + [s["p0"] for s in all_segs])
    sphericity_residual_mm = float(np.max(np.abs(np.linalg.norm(pts, axis=1) - R)))

    # Terminal tips = segments at max generation reached (or whose growth was clipped)
    def _tips(segs):
        starts = {s["p0"].tobytes() for s in segs}
        return [s for s in segs if s["p1"].tobytes() not in starts]
    tips_a, tips_v = _tips(arteries), _tips(veins)
    az = np.array([math.atan2(s["p1"][1], s["p1"][0]) for s in tips_a + tips_v])
    typ = np.array([0] * len(tips_a) + [1] * len(tips_v))
    order = np.argsort(az)
    typ_sorted = typ[order]
    alternations = np.sum(typ_sorted != np.roll(typ_sorted, -1))
    interdigitation_index = float(alternations / len(typ_sorted))
    # Nearest-tip pairing: fraction of arterial tips whose nearest *other* tip is venous
    # (1.0 = every terminal arteriole is paired with a terminal venule, 0.5 = random mixing)
    tip_pts = np.array([s["p1"] for s in tips_a + tips_v])
    kd = cKDTree(tip_pts)
    dists, idx = kd.query(tip_pts, k=2)
    nn_type = typ[idx[:, 1]]
    av_pairing_fraction = float(np.mean(nn_type[: len(tips_a)] == 1))
    kd_v = cKDTree(np.array([s["p1"] for s in tips_v]))
    a_to_v_mm = kd_v.query(np.array([s["p1"] for s in tips_a]))[0]
    kd_a = cKDTree(np.array([s["p1"] for s in tips_a]))
    a_to_a_mm = kd_a.query(np.array([s["p1"] for s in tips_a]), k=2)[0][:, 1]

    # Fundus-projection A/V crossings
    a0 = np.array([s["p0"][:2] for s in arteries]); a1 = np.array([s["p1"][:2] for s in arteries])
    v0 = np.array([s["p0"][:2] for s in veins]); v1 = np.array([s["p1"][:2] for s in veins])
    crossings = int(np.sum(_segments_intersect_2d(a0, a1, v0, v1)))

    # Coupled flow balance
    sum_r3_a = float(sum(s["r_um"] ** 3 for s in tips_a))
    sum_r3_v = float(sum(s["r_um"] ** 3 for s in tips_v))
    k_ratio_v_over_a = sum_r3_a / sum_r3_v              # k_v / k_a
    tau_a = math.sqrt(eta_bulk * b)
    tau_v = tau_a * k_ratio_v_over_a
    b_v = b * k_ratio_v_over_a ** 2
    root_avr = (r0_art_um) / (r0_art_um / avr)

    # Projection foreshortening of bifurcation angles (arterial tree)
    true_tot, app_tot, ecc = [], [], []
    for s in arteries:
        if "bif_total_deg" not in s:
            continue
        kids = [c for c in arteries if np.allclose(c["p0"], s["p1"]) and c["gen"] == s["gen"] + 1]
        if len(kids) != 2:
            continue
        d1 = kids[0]["t0"][:2]; d2 = kids[1]["t0"][:2]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        app = math.degrees(math.acos(float(np.clip(np.dot(d1, d2) / (n1 * n2), -1.0, 1.0))))
        true_tot.append(s["bif_total_deg"]); app_tot.append(app); ecc.append(s["polar_deg"])
    true_tot, app_tot, ecc = np.array(true_tot), np.array(app_tot), np.array(ecc)
    bias = app_tot - true_tot
    central = ecc < 30.0
    periph = ecc >= 60.0

    total_len_a = float(sum(s["length_mm"] for s in arteries))
    chord_len_a = float(sum(np.linalg.norm(s["p1"] - s["p0"]) for s in arteries))
    return {
        "arteries": arteries,
        "veins": veins,
        "n_segments_artery": len(arteries),
        "n_segments_vein": len(veins),
        "n_tips_artery": len(tips_a),
        "n_tips_vein": len(tips_v),
        "sphericity_residual_mm": sphericity_residual_mm,
        "max_polar_deg": float(max(s["polar_deg"] for s in all_segs)),
        "interdigitation_index": interdigitation_index,
        "av_pairing_fraction": av_pairing_fraction,
        "tip_a_to_v_distance_mean_mm": float(np.mean(a_to_v_mm)),
        "tip_a_to_a_distance_mean_mm": float(np.mean(a_to_a_mm)),
        "av_crossings_projected": crossings,
        "root_avr": root_avr,
        "k_v_over_k_a": k_ratio_v_over_a,
        "tau_wall_artery_dyn_cm2": tau_a,
        "tau_wall_vein_dyn_cm2": tau_v,
        "b_vein_erg_s_cm3": b_v,
        "proj_angle_bias_mean_deg": float(np.mean(bias)),
        "proj_angle_bias_abs_mean_deg": float(np.mean(np.abs(bias))),
        "proj_angle_bias_central_deg": float(np.mean(np.abs(bias[central]))) if np.any(central) else float("nan"),
        "proj_angle_bias_peripheral_deg": float(np.mean(np.abs(bias[periph]))) if np.any(periph) else float("nan"),
        "proj_true_total_deg": true_tot,
        "proj_apparent_total_deg": app_tot,
        "proj_eccentricity_deg": ecc,
        "geodesic_over_chord_length": total_len_a / chord_len_a,
        "arterial_geodesic_length_mm": total_len_a,
    }


# =============================================================================
# Section 5: Real Human Retinal Vasculature Specimen Computer-Vision Pipeline
# =============================================================================

def _thin_binary_mask(mask: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Zhang-Suen 1-pixel morphological skeletonization in pure NumPy."""
    img = mask.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            b_cnt = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            a_cnt = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if step == 0:
                m1 = p2 * p4 * p6
                m2 = p4 * p6 * p8
            else:
                m1 = p2 * p4 * p8
                m2 = p2 * p6 * p8
            cond = (img[1:-1, 1:-1] == 1) & (b_cnt >= 2) & (b_cnt <= 6) & (a_cnt == 1) & (m1 == 0) & (m2 == 0)
            if np.any(cond):
                img[1:-1, 1:-1][cond] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def segment_retinal_vasculature_specimen(specimen_path: Path = SPECIMEN_IMG_PATH) -> dict[str, Any]:
    """Segment the human retinal fundus vasculature specimen (`retinal_vasculature_specimen.jpg`,
    Panels A and B) and extract:
      1. Green-channel hemoglobin absorption enhancement & vessel binary mask
      2. Euclidean distance transform (EDT) local lumen radius map r(x, y)
      3. Zhang-Suen 1-pixel centerline skeleton & degree>=3 arterial/venous Y-junction nodes
      4. Tortuosity index comparison between Panel A (normal radial) and Panel B (tortuous)
      5. Local parent/daughter radius ratio (r1^3 + r2^3)^(1/3) / r0 and branching angles
         across detected retinal Y-bifurcations.
    """
    with Image.open(specimen_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64)

    # Panel A (normal radial vessels) and Panel B (tortuous vessels) crops
    panel_a_rgb = rgb[15:375, 15:380]
    panel_b_rgb = rgb[15:375, 405:770]

    def _segment_panel(p_rgb: np.ndarray) -> dict[str, Any]:
        green = p_rgb[:, :, 1]
        red = p_rgb[:, :, 0]
        fov_mask = (red > 35.0) & (green > 12.0)
        fov_eroded = ndi.binary_erosion(fov_mask, iterations=10)

        # Blood vessels appear dark in the green channel relative to local retinal background
        bg_smooth = ndi.gaussian_filter(green, sigma=11.0)
        vessel_resp = np.clip(bg_smooth - green, 0.0, None)
        vessel_resp = ndi.gaussian_filter(vessel_resp, sigma=1.1) * fov_eroded

        valid_vals = vessel_resp[fov_eroded]
        thresh = float(np.percentile(valid_vals, 84.0))
        raw_mask = (vessel_resp >= thresh) & fov_eroded
        clean_mask = ndi.binary_opening(raw_mask, structure=np.ones((2, 2), dtype=bool))
        labeled, n_comp = ndi.label(clean_mask)
        if n_comp > 0:
            sizes = ndi.sum(clean_mask, labeled, range(1, n_comp + 1))
            keep = np.isin(labeled, np.where(sizes >= 35)[0] + 1)
            clean_mask = keep

        edt_radius = ndi.distance_transform_edt(clean_mask)
        skel = _thin_binary_mask(clean_mask)

        # Detect Y-junction branch candidates on the 1-pixel skeleton
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
        deg = ndi.convolve(skel.astype(int), kernel, mode="constant", cval=0) * skel.astype(int)
        junc_mask = (deg >= 3) & (edt_radius >= 1.1)
        j_lbl, n_junc = ndi.label(junc_mask, structure=np.ones((3, 3), dtype=int))
        centers = ndi.center_of_mass(junc_mask, j_lbl, range(1, n_junc + 1)) if n_junc > 0 else []

        # Measure branching angles and local vessel radii around well-isolated Y-junctions
        junction_angles: list[float] = []
        junction_cube_ratios: list[float] = []
        junction_coords: list[tuple[float, float]] = []

        h_p, w_p = skel.shape
        r_ring = 8
        for cy, cx in centers:
            iy, ix = int(round(cy)), int(round(cx))
            if iy < r_ring + 4 or iy >= h_p - r_ring - 4 or ix < r_ring + 4 or ix >= w_p - r_ring - 4:
                continue
            patch_skel = skel[iy - r_ring:iy + r_ring + 1, ix - r_ring:ix + r_ring + 1]
            patch_edt = edt_radius[iy - r_ring:iy + r_ring + 1, ix - r_ring:ix + r_ring + 1]
            yy, xx = np.ogrid[-r_ring:r_ring + 1, -r_ring:r_ring + 1]
            ring_band = (yy ** 2 + xx ** 2 >= (r_ring - 1.2) ** 2) & (yy ** 2 + xx ** 2 <= (r_ring + 1.2) ** 2)
            ring_hits = patch_skel & ring_band
            hit_lbl, n_arms = ndi.label(ring_hits, structure=np.ones((3, 3), dtype=int))
            if n_arms != 3:
                continue

            arm_angles = []
            arm_radii = []
            for a_idx in range(1, 4):
                ay, ax = ndi.center_of_mass(ring_hits, hit_lbl, a_idx)
                arm_angles.append(math.atan2(ay - r_ring, ax - r_ring))
                r_est = float(np.max(patch_edt[ndi.binary_dilation(hit_lbl == a_idx, iterations=2)]))
                arm_radii.append(max(r_est, 1.0))

            # Identify parent arm p_idx and two forward-splitting daughter arms d1, d2
            best_ang = None
            best_r = None
            for p_idx in range(3):
                d1 = (p_idx + 1) % 3
                d2 = (p_idx + 2) % 3
                diff = abs(math.degrees(arm_angles[d1] - arm_angles[d2]))
                if diff > 180.0:
                    diff = 360.0 - diff
                da1 = abs(math.degrees(arm_angles[p_idx] - arm_angles[d1]))
                if da1 > 180.0:
                    da1 = 360.0 - da1
                da2 = abs(math.degrees(arm_angles[p_idx] - arm_angles[d2]))
                if da2 > 180.0:
                    da2 = 360.0 - da2
                if 50.0 <= diff <= 105.0 and da1 > 105.0 and da2 > 105.0:
                    r0_px = max(arm_radii[p_idx], max(arm_radii[d1], arm_radii[d2]))
                    r1_px = arm_radii[d1]
                    r2_px = arm_radii[d2]
                    best_ang = diff
                    best_r = ((r1_px ** 3 + r2_px ** 3) ** (1.0 / 3.0)) / r0_px
                    break
            if best_ang is not None and best_r is not None:
                junction_angles.append(best_ang)
                junction_cube_ratios.append(best_r)
                junction_coords.append((float(cx), float(cy)))

        # Estimate individual branch-segment tortuosity (arc length / chord length)
        branches = skel & ~ndi.binary_dilation(deg >= 3, iterations=1)
        b_lbl, n_branches = ndi.label(branches, structure=np.ones((3, 3), dtype=int))
        tortuosities = []
        for b_idx in range(1, n_branches + 1):
            ys, xs = np.where(b_lbl == b_idx)
            if len(ys) < 18:
                continue
            coords = np.column_stack([xs, ys]).astype(float)
            dmat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
            chord = float(np.max(dmat))
            if chord >= 14.0:
                tortuosities.append(float(len(ys)) / chord)

        return {
            "rgb": p_rgb,
            "vessel_response": vessel_resp,
            "vessel_mask": clean_mask,
            "edt_radius": edt_radius,
            "skeleton": skel,
            "vessel_area_fraction": float(clean_mask.sum() / max(1, fov_eroded.sum())),
            "n_junctions_detected": len(junction_angles),
            "junction_angles_deg": np.array(junction_angles, dtype=float),
            "junction_cube_ratios": np.array(junction_cube_ratios, dtype=float),
            "junction_coords": junction_coords,
            "tortuosity_mean": float(np.mean(tortuosities)) if tortuosities else 1.01,
        }

    seg_a = _segment_panel(panel_a_rgb)
    seg_b = _segment_panel(panel_b_rgb)
    all_angles = np.concatenate([seg_a["junction_angles_deg"], seg_b["junction_angles_deg"]])
    all_ratios = np.concatenate([seg_a["junction_cube_ratios"], seg_b["junction_cube_ratios"]])

    return {
        "panel_a": seg_a,
        "panel_b": seg_b,
        "total_junctions_segmented": int(len(all_angles)),
        "segmented_angle_mean_deg": float(np.mean(all_angles)),
        "segmented_angle_std_deg": float(np.std(all_angles, ddof=1)),
        "segmented_angle_cv": float(np.std(all_angles, ddof=1) / np.mean(all_angles)),
        "segmented_cube_ratio_mean": float(np.mean(all_ratios)),
        "segmented_cube_ratio_std": float(np.std(all_ratios, ddof=1)),
        "tortuosity_ratio_b_over_a": float(seg_b["tortuosity_mean"] / seg_a["tortuosity_mean"]),
        "all_angles_deg": all_angles,
        "all_cube_ratios": all_ratios,
    }


# =============================================================================
# Section 6: 9-Panel Publication Synthesis Figure & Animated Growth GIFs
# =============================================================================

def render_outputs(
    output_png: Path = OUTPUT_PNG,
    output_gif: Path = OUTPUT_GIF,
    skip_gif: bool = False,
    alpha_cli: float = 1.0,
    gamma_cli: float = GAMMA_MURRAY,
    r0_um_cli: float = R0_ROOT_UM_DEFAULT,
    generations_cli: int = 6,
    eta_cli: float = ETA_POISE_DEFAULT,
    b_cli: float = METABOLIC_B_DEFAULT,
    eye_radius_mm_cli: float = EYE_RADIUS_MM_DEFAULT,
    avr_cli: float = AVR_DEFAULT,
    coupling_cli: str = "repulsive",
    seed_cli: int = 1926,
    output_cup_gif: Path | None = None,
) -> dict[str, Any]:
    """Render the 9-panel Pass 3 comparison figure (`vascular_cube_law_branching_comparison.png`),
    the 2D multi-regime growth animation (`vascular_cube_law_branching_growth.gif`) and the 3D
    hemispherical retinal-cup A/V growth animation (`vascular_cube_law_branching_retinal_cup_3d.gif`)."""
    ref_data = verify_dataset_integrity()
    emp = evaluate_empirical_retinal_junctions(ref_data)
    sweep = run_gamma_power_sweep(generations=7, r0_um=r0_um_cli)
    seg = segment_retinal_vasculature_specimen(SPECIMEN_IMG_PATH)
    fl = run_fl_exponent_sweep(generations=7, r0_um=r0_um_cli, eta_bulk=eta_cli, b=b_cli)
    av = evaluate_hemispherical_av_coupling(generations=generations_cli, r0_art_um=r0_um_cli, avr=avr_cli,
                                            eye_radius_mm=eye_radius_mm_cli, gamma=gamma_cli, eta_bulk=eta_cli,
                                            b=b_cli, coupling=coupling_cli)
    av_ctrl = evaluate_hemispherical_av_coupling(generations=generations_cli, r0_art_um=r0_um_cli, avr=avr_cli,
                                                 eye_radius_mm=eye_radius_mm_cli, gamma=gamma_cli, eta_bulk=eta_cli,
                                                 b=b_cli, coupling="none")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    if output_cup_gif is None:
        output_cup_gif = output_gif.with_name(output_gif.stem.replace("_growth", "") + "_retinal_cup_3d.gif")

    fig = plt.figure(figsize=(18.0, 15.6), dpi=140)
    fig.patch.set_facecolor("#0c1118")
    gs = fig.add_gridspec(3, 3, hspace=0.30, wspace=0.26, left=0.045, right=0.985, top=0.955, bottom=0.04)
    axes = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            if (i, j) == (2, 1):
                axes[i, j] = fig.add_subplot(gs[i, j], projection="3d")
            else:
                axes[i, j] = fig.add_subplot(gs[i, j])
    for ax in axes.flat:
        ax.set_facecolor("#131b26")
        ax.tick_params(colors="#d4dce8", labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#2e3f57")

    # -------------------------------------------------------------------------
    # Panel A: Murray (1926a) Power Functional P(r) & Tree Exponent Sweep P(gamma)
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    q0 = murray_conductance_constant(eta=eta_cli, b=b_cli) * ((r0_um_cli * 1e-4) ** 3)
    r_opt = murray_optimal_radius(q0, eta=eta_cli, b=b_cli)
    r_grid = np.linspace(0.60 * r_opt, 1.55 * r_opt, 220)
    p_visc, p_met, p_tot = poiseuille_metabolic_power(r_grid, q0, eta=eta_cli, b=b_cli)
    ax_a.plot(r_grid * 1e4, p_visc, "--", color="#4ea8de", lw=1.6, label="Viscous $P_{visc} \\propto r^{-4}$")
    ax_a.plot(r_grid * 1e4, p_met, ":", color="#f4a261", lw=1.8, label="Metabolic $P_{met} \\propto r^2$")
    ax_a.plot(r_grid * 1e4, p_tot, "-", color="#2ec4b6", lw=2.4, label="Total $P(r) = P_{visc} + P_{met}$")
    ax_a.axvline(r_opt * 1e4, color="#ff6b6b", ls="-.", lw=1.6, label=f"Murray $r_* = {r_opt*1e4:.1f}\\,\\mu$m ($P_{{met}}=2P_{{visc}}$)")
    ax_a.set_title("A. Murray (1926a) Minimum-Work Functional & $\\gamma$ Sweep", color="white", fontsize=10.5, fontweight="bold")
    ax_a.set_xlabel("Vessel lumen radius $r$ ($\\mu$m)", color="#d4dce8")
    ax_a.set_ylabel("Single-segment power (erg / s / cm)", color="#d4dce8")
    ax_a.set_ylim(0, np.percentile(p_tot, 92))
    ax_a.legend(loc="upper right", fontsize=7.3, facecolor="#131b26", edgecolor="#2e3f57", labelcolor="white")

    # Inset: 7-generation tree power vs exponent gamma
    ins_a = ax_a.inset_axes([0.46, 0.42, 0.50, 0.34])
    ins_a.set_facecolor("#0f151e")
    ins_a.plot(sweep["gamma_grid"], sweep["total_power_erg_s"] / sweep["power_murray_erg_s"], color="#ffd166", lw=1.8)
    ins_a.axvline(3.0, color="#2ec4b6", ls="--", lw=1.2, label="$\\gamma^*=3.0$")
    ins_a.axvline(2.0, color="#ff6b6b", ls=":", lw=1.2, label=f"Da Vinci $\\gamma=2$ (+{sweep['davinci_excess_power_pct']:.1f}%)")
    ins_a.set_xlabel("Exponent $\\gamma$", color="#d4dce8", fontsize=7)
    ins_a.set_ylabel("$P_{tree}(\\gamma)/P_{min}$", color="#d4dce8", fontsize=7)
    ins_a.tick_params(colors="#d4dce8", labelsize=6.5)
    ins_a.legend(fontsize=6.0, facecolor="#0f151e", edgecolor="#2e3f57", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel B: Murray (1926b) & Zamir (1976) Asymmetric Bifurcation Angle Law
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    alphas = np.linspace(0.15, 1.0, 180)
    th1_m = [murray_asymmetric_bifurcation(1.0, a, 3.0, "murray_volume")["theta1_deg"] for a in alphas]
    th2_m = [murray_asymmetric_bifurcation(1.0, a, 3.0, "murray_volume")["theta2_deg"] for a in alphas]
    tot_m = [murray_asymmetric_bifurcation(1.0, a, 3.0, "murray_volume")["total_angle_deg"] for a in alphas]
    tot_z = [murray_asymmetric_bifurcation(1.0, a, 3.0, "zamir_surface")["total_angle_deg"] for a in alphas]

    ax_b.plot(alphas, th1_m, "-", color="#4ea8de", lw=2.0, label="Main daughter $\\theta_1(\\alpha)$ (Murray 1926b)")
    ax_b.plot(alphas, th2_m, "-", color="#f4a261", lw=2.0, label="Minor branch $\\theta_2(\\alpha)$ (Murray 1926b)")
    ax_b.plot(alphas, tot_m, "-", color="#2ec4b6", lw=2.4, label="Total $\\theta_1+\\theta_2$ ($74.93^\\circ$ at $\\alpha=1$)")
    ax_b.plot(alphas, tot_z, "--", color="#a8dadc", lw=1.4, label="Zamir (1976) surface drag ($90.0^\\circ$ at $\\alpha=1$)")
    ax_b.axhline(120.0, color="#ff6b6b", ls=":", lw=1.3, label="Steiner length minimum ($120.0^\\circ$)")

    emp_alphas = [j["alpha_obs"] for j in ref_data["empirical_retinal_junctions_zamir_1979"]]
    emp_th1 = [j["theta1_obs_deg"] for j in ref_data["empirical_retinal_junctions_zamir_1979"]]
    emp_th2 = [j["theta2_obs_deg"] for j in ref_data["empirical_retinal_junctions_zamir_1979"]]
    emp_tot = [j["total_angle_obs_deg"] for j in ref_data["empirical_retinal_junctions_zamir_1979"]]
    ax_b.scatter(emp_alphas, emp_tot, color="#2ec4b6", edgecolors="white", s=42, zorder=5,
                 label=f"Retinal junctions $J_{{01..08}}$ (RMSE={emp['rmse_total_angle_deg']:.2f}$^\\circ$, $r={emp['pearson_r_branch_angles']:.4f}$)")
    ax_b.scatter(emp_alphas, emp_th1, color="#4ea8de", marker="^", s=28, zorder=5)
    ax_b.scatter(emp_alphas, emp_th2, color="#f4a261", marker="v", s=28, zorder=5)
    ax_b.set_title("B. Murray (1926b) Bifurcation Angles vs Retinal Data", color="white", fontsize=10.5, fontweight="bold")
    ax_b.set_xlabel("Daughter radius asymmetry $\\alpha = r_2 / r_1$", color="#d4dce8")
    ax_b.set_ylabel("Branching angle (degrees)", color="#d4dce8")
    ax_b.set_ylim(0, 132)
    ax_b.legend(loc="upper right", fontsize=6.9, facecolor="#131b26", edgecolor="#2e3f57", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel C: 2D Vascular Tree Comparison (Murray Cube vs Asymmetric vs Steiner/DaVinci)
    # -------------------------------------------------------------------------
    ax_c = axes[0, 2]
    tree_sym = build_vascular_tree_2d(generations=generations_cli, r0_um=r0_um_cli, l0_mm=1.85, alpha=alpha_cli, gamma=gamma_cli, optimality_mode="murray_volume")
    tree_asym = build_vascular_tree_2d(generations=generations_cli, r0_um=r0_um_cli, l0_mm=1.85, alpha=0.72, gamma=3.0, optimality_mode="murray_volume")
    tree_ctrl = build_vascular_tree_2d(generations=5, r0_um=r0_um_cli, l0_mm=1.55, alpha=1.0, gamma=2.0, optimality_mode="steiner_length")

    for i, s in enumerate(tree_sym):
        ax_c.plot([s["x0"] - 3.6, s["x1"] - 3.6], [s["y0"], s["y1"]], color="#e63946",
                  lw=max(0.65, s["r_um"] / 15.0), solid_capstyle="round",
                  label="Murray Cube ($\\gamma=3, \\alpha=1, 74.9^\\circ$)" if i == 0 else None)
    for i, s in enumerate(tree_asym):
        ax_c.plot([s["x0"] + 0.2, s["x1"] + 0.2], [s["y0"], s["y1"]], color="#2ec4b6",
                  lw=max(0.65, s["r_um"] / 15.0), solid_capstyle="round",
                  label="Asymmetric Arcade ($\\gamma=3, \\alpha=0.72$)" if i == 0 else None)
    for i, s in enumerate(tree_ctrl):
        ax_c.plot([s["x0"] + 4.0, s["x1"] + 4.0], [s["y0"], s["y1"]], color="#f4a261", alpha=0.88,
                  lw=max(0.65, s["r_um"] / 15.0), solid_capstyle="round",
                  label="Ablation Control ($\\gamma=2, \\Theta=120^\\circ$)" if i == 0 else None)

    ax_c.set_aspect("equal")
    ax_c.set_xlim(-6.5, 6.8)
    ax_c.set_ylim(-0.4, 7.6)
    ax_c.set_title("C. Recursive Arterial Tree Topologies (Gen 0..6)", color="white", fontsize=10.5, fontweight="bold")
    ax_c.set_xlabel("Lateral coordinate $x$ (mm)", color="#d4dce8")
    ax_c.set_ylabel("Axial coordinate $y$ (mm)", color="#d4dce8")
    ax_c.legend(loc="upper center", fontsize=7.2, facecolor="#131b26", edgecolor="#2e3f57", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel D: Sherman (1981) Generational Hierarchy & Wall Shear Invariance
    # -------------------------------------------------------------------------
    ax_d = axes[1, 0]
    h_murray = evaluate_generational_hierarchy(generations=7, r0_um=r0_um_cli, gamma=3.0)
    h_davinci = evaluate_generational_hierarchy(generations=7, r0_um=r0_um_cli, gamma=2.0)
    gens = np.array([r["generation"] for r in h_murray["rows"]])
    rad_m = np.array([r["radius_um"] for r in h_murray["rows"]])
    rad_d = np.array([r["radius_um"] for r in h_davinci["rows"]])
    tau_m = np.array([r["wall_shear_dyn_cm2"] for r in h_murray["rows"]])
    tau_d = np.array([r["wall_shear_dyn_cm2"] for r in h_davinci["rows"]])

    ax_d.plot(gens, rad_m, "o-", color="#2ec4b6", lw=2.2, label="Murray radius $r_g = r_0 2^{-g/3}$ ($\\mu$m)")
    ax_d.plot(gens, rad_d, "s--", color="#4ea8de", lw=1.6, label="Da Vinci radius $r_g = r_0 2^{-g/2}$ ($\\mu$m)")
    ax_d.set_xlabel("Arterial bifurcation generation $g$", color="#d4dce8")
    ax_d.set_ylabel("Vessel lumen radius $r_g$ ($\\mu$m)", color="#2ec4b6")

    ax_d2 = ax_d.twinx()
    ax_d2.plot(gens, tau_m, "^-", color="#ffd166", lw=2.0, label="Murray wall shear $\\tau_w = \\sqrt{\\eta b} = 8.22$ dyn/cm$^2$")
    ax_d2.plot(gens, tau_d, "v:", color="#ff6b6b", lw=1.8, label="Da Vinci shear $\\tau_w(g)$ (+100% drift)")
    ax_d2.set_ylabel("Wall shear stress $\\tau_w$ (dyn / cm$^2$)", color="#ffd166")
    ax_d2.tick_params(colors="#ffd166", labelsize=8.5)
    ax_d.set_title("D. Sherman (1981) Hierarchy & Wall Shear Invariance", color="white", fontsize=10.5, fontweight="bold")

    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=6.9,
                facecolor="#131b26", edgecolor="#2e3f57", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel E: Real Human Retinal Fundus Specimen & Segmented Y-Junctions
    # -------------------------------------------------------------------------
    ax_e = axes[1, 1]
    pa = seg["panel_a"]
    pb = seg["panel_b"]
    # Build composite side-by-side display: Panel A RGB + skeleton overlay | Panel B EDT radius + junctions
    comp_h, comp_w, _ = pa["rgb"].shape
    canvas = np.zeros((comp_h, comp_w * 2 + 12, 3), dtype=np.float64)
    left_vis = pa["rgb"].copy() / 255.0
    left_vis[pa["skeleton"]] = [0.18, 0.95, 0.85]
    right_vis = pb["rgb"].copy() / 255.0 * 0.45
    edt_norm = np.clip(pb["edt_radius"] / max(1.0, float(np.max(pb["edt_radius"]))), 0.0, 1.0)
    right_vis[:, :, 0] = np.clip(right_vis[:, :, 0] + 0.85 * edt_norm, 0.0, 1.0)
    right_vis[:, :, 1] = np.clip(right_vis[:, :, 1] + 0.55 * (pb["skeleton"].astype(float)), 0.0, 1.0)
    canvas[:, :comp_w, :] = left_vis
    canvas[:, comp_w + 12:, :] = right_vis

    ax_e.imshow(canvas)
    for cx, cy in pa["junction_coords"]:
        ax_e.plot(cx, cy, "o", color="#ffd166", markersize=5.0, markeredgecolor="black", markeredgewidth=0.6)
    for cx, cy in pb["junction_coords"]:
        ax_e.plot(cx + comp_w + 12, cy, "o", color="#2ec4b6", markersize=5.0, markeredgecolor="black", markeredgewidth=0.6)
    ax_e.set_title(
        f"E. Human Retinal Vasculature Specimen (N={seg['total_junctions_segmented']} Y-Junctions, Tortuosity B/A={seg['tortuosity_ratio_b_over_a']:.2f}x)",
        color="white",
        fontsize=9.8,
        fontweight="bold",
    )
    ax_e.set_xlabel("Left: Panel A Optic-Disc Skeleton | Right: Panel B Tortuous Caliber EDT & Y-Junctions", color="#d4dce8", fontsize=8)
    ax_e.set_xticks([])
    ax_e.set_yticks([])

    # -------------------------------------------------------------------------
    # Panel F: Cube-Law Radius Parity & Segmented Retinal Angle Distribution
    # -------------------------------------------------------------------------
    ax_f = axes[1, 2]
    r0_emp = emp["r0_um"]
    r_cube_emp = (emp["r1_um"] ** 3 + emp["r2_um"] ** 3) ** (1.0 / 3.0)
    r_dav_emp = np.sqrt(emp["r1_um"] ** 2 + emp["r2_um"] ** 2)

    r_line = np.linspace(30.0, 84.0, 100)
    ax_f.plot(r_line, r_line, "--", color="white", lw=1.4, label="Exact parity $(r_1^\\gamma+r_2^\\gamma)^{1/\\gamma} = r_0$")
    ax_f.scatter(r0_emp, r_cube_emp, color="#2ec4b6", s=48, edgecolors="white", zorder=5,
                 label=f"Murray $\\gamma=3$: $(r_1^3+r_2^3)^{{1/3}}/r_0 = {emp['cube_law_ratio_mean']:.3f}\\pm{emp['cube_law_ratio_std']:.3f}$")
    ax_f.scatter(r0_emp, r_dav_emp, color="#ff6b6b", marker="s", s=38, alpha=0.85,
                 label=f"Da Vinci $\\gamma=2$: $(r_1^2+r_2^2)^{{1/2}}/r_0 = {emp['davinci_ratio_mean']:.3f}$ (biased)")
    ax_f.set_title("F. Murray Cube-Law Parity & Segmented Angle Histogram", color="white", fontsize=10.5, fontweight="bold")
    ax_f.set_xlabel("Parent arteriole radius $r_0$ ($\\mu$m)", color="#d4dce8")
    ax_f.set_ylabel("Reconstructed $(r_1^\\gamma + r_2^\\gamma)^{1/\\gamma}$ ($\\mu$m)", color="#d4dce8")
    ax_f.legend(loc="upper left", fontsize=7.0, facecolor="#131b26", edgecolor="#2e3f57", labelcolor="white")

    # Inset: Histogram of segmented retinal Y-junction angles vs Murray 74.93 deg
    ins_f = ax_f.inset_axes([0.50, 0.13, 0.46, 0.36])
    ins_f.set_facecolor("#0f151e")
    ins_f.hist(seg["all_angles_deg"], bins=7, color="#4ea8de", edgecolor="white", alpha=0.85)
    ins_f.axvline(74.93, color="#2ec4b6", ls="--", lw=1.4, label="Murray $74.9^\\circ$")
    ins_f.axvline(seg["segmented_angle_mean_deg"], color="#ffd166", ls="-", lw=1.4,
                  label=f"Img $\\mu={seg['segmented_angle_mean_deg']:.1f}^\\circ$")
    ins_f.set_xlabel("Segmented $\\Theta$ (deg)", color="#d4dce8", fontsize=6.8)
    ins_f.set_ylabel("Count", color="#d4dce8", fontsize=6.8)
    ins_f.tick_params(colors="#d4dce8", labelsize=6.5)
    ins_f.legend(fontsize=5.8, facecolor="#0f151e", edgecolor="#2e3f57", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel G (Pass 3): Fahraeus-Lindqvist eta(r) and the local optimal exponent gamma_eff(r)
    # -------------------------------------------------------------------------
    ax_g = axes[2, 0]
    ax_g.semilogx(fl["r_local_um"], fl["eta_rel_local"], "-", color="#f4a261", lw=2.2,
                  label="Pries et al. (1992) $\\eta_{rel}(D=2r)$, $H_D=0.45$")
    ax_g.axhline(FL_ETA_BULK_REL, color="#f4a261", ls=":", lw=1.1, label="Newtonian bulk $\\eta_{rel}=3.2$ (Murray 1926a)")
    ax_g.axvline(FL_PRECAPILLARY_RADIUS_UM, color="#a8dadc", ls="--", lw=1.1, label="pre-capillary $r<25\\,\\mu$m")
    ax_g.set_xlabel("Vessel lumen radius $r$ ($\\mu$m, log)", color="#d4dce8")
    ax_g.set_ylabel("Relative apparent viscosity $\\eta_{rel}$", color="#f4a261")
    ax_g.set_ylim(1.0, 3.5)
    ax_g2 = ax_g.twinx()
    ax_g2.semilogx(fl["r_local_um"], fl["gamma_eff_fl"], "-", color="#2ec4b6", lw=2.2,
                   label="$\\gamma_{eff}(r)=d\\ln Q^*/d\\ln r$ (F-L blood)")
    ax_g2.semilogx(fl["r_local_um"], fl["gamma_eff_newtonian"], "--", color="#ff6b6b", lw=1.4, label="Newtonian $\\gamma_{eff}=3$")
    j25 = fl["junction_25um"]
    ax_g2.plot([25.0], [j25["gamma_junction"]], "o", color="#ffd166", markersize=7, markeredgecolor="black",
               label=f"symmetric junction $\\gamma_j$ at $r_0=25\\,\\mu$m = {j25['gamma_junction']:.3f}")
    ax_g2.set_ylabel("Local optimal exponent $\\gamma_{eff}$", color="#2ec4b6")
    ax_g2.set_ylim(2.6, 3.1)
    ax_g2.tick_params(colors="#2ec4b6", labelsize=8.5)
    ax_g.set_title("G. Fahraeus-Lindqvist $\\eta(r)$ & Exponent Shift Below $\\gamma=3$", color="white", fontsize=10.5, fontweight="bold")
    l1, lb1 = ax_g.get_legend_handles_labels()
    l2, lb2 = ax_g2.get_legend_handles_labels()
    ax_g.legend(l1 + l2, lb1 + lb2, loc="lower right", fontsize=6.6, facecolor="#131b26", edgecolor="#2e3f57", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel H (Pass 3): 3D hemispherical retinal cup with coupled arterial & venous Murray trees
    # -------------------------------------------------------------------------
    ax_h = axes[2, 1]
    ax_h.set_facecolor("#131b26")
    R_eye = eye_radius_mm_cli
    uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 48), np.linspace(0, math.radians(RETINA_POLAR_LIMIT_DEG), 24))
    ax_h.plot_surface(R_eye * np.sin(vv) * np.cos(uu), R_eye * np.sin(vv) * np.sin(uu), R_eye * np.cos(vv),
                      color="#1f2a3a", alpha=0.35, linewidth=0, antialiased=False, shade=True)
    for s in av["arteries"]:
        ax_h.plot([s["p0"][0], s["p1"][0]], [s["p0"][1], s["p1"][1]], [s["p0"][2], s["p1"][2]],
                  color="#e63946", lw=max(0.5, s["r_um"] / 22.0), solid_capstyle="round")
    for s in av["veins"]:
        ax_h.plot([s["p0"][0], s["p1"][0]], [s["p0"][1], s["p1"][1]], [s["p0"][2], s["p1"][2]],
                  color="#4ea8de", lw=max(0.5, s["r_um"] / 22.0), solid_capstyle="round")
    ax_h.plot([0.0], [0.0], [R_eye], "o", color="#ffd166", markersize=5)
    ax_h.set_xlim(-R_eye, R_eye); ax_h.set_ylim(-R_eye, R_eye); ax_h.set_zlim(-0.2 * R_eye, R_eye)
    ax_h.set_box_aspect((1, 1, 0.6), zoom=1.55)
    ax_h.view_init(elev=38, azim=-55)
    ax_h.set_axis_off()
    ax_h.set_title(
        f"H. 3D Retinal Cup ($R={R_eye:.0f}$ mm) — Arterial (red) & Venous (blue) Murray Trees\n"
        f"AVR={av['root_avr']:.2f}, A/V alternation={av['interdigitation_index']:.2f} (ctrl {av_ctrl['interdigitation_index']:.2f}), "
        f"tip $d_{{AV}}$={av['tip_a_to_v_distance_mean_mm']:.2f} mm (ctrl {av_ctrl['tip_a_to_v_distance_mean_mm']:.2f}), "
        f"crossings={av['av_crossings_projected']} (ctrl {av_ctrl['av_crossings_projected']})",
        color="white", fontsize=8.6, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel I (Pass 3): Newtonian vs F-L exponent sweep + fundus-projection foreshortening
    # -------------------------------------------------------------------------
    ax_i = axes[2, 2]
    g_grid = fl["gamma_grid"]
    pn = fl["power_newtonian"]; pf = fl["power_fahraeus_lindqvist"]
    ax_i.plot(g_grid, pn / pn.min(), "-", color="#ff6b6b", lw=2.0,
              label=f"Newtonian tree: $\\gamma^*={fl['gamma_opt_newtonian']:.3f}$")
    ax_i.plot(g_grid, pf / pf.min(), "-", color="#2ec4b6", lw=2.2,
              label=f"Fahraeus-Lindqvist tree: $\\gamma^*={fl['gamma_opt_fahraeus_lindqvist']:.3f}$")
    ax_i.axvline(3.0, color="#ff6b6b", ls=":", lw=1.1)
    ax_i.axvline(fl["gamma_opt_fahraeus_lindqvist"], color="#2ec4b6", ls=":", lw=1.1)
    ax_i.axvline(2.7, color="#ffd166", ls="--", lw=1.0, label="Pass 2 target $\\gamma\\approx2.7$")
    ax_i.set_xlabel("Tree exponent $\\gamma$ ($r_g = r_0 2^{-g/\\gamma}$, 7 generations)", color="#d4dce8")
    ax_i.set_ylabel("$P_{tree}(\\gamma)/P_{min}$", color="#d4dce8")
    ax_i.set_ylim(0.995, 1.10)
    ax_i.set_xlim(g_grid.min(), g_grid.max())
    ax_i.set_title(f"I. F-L Exponent Shift $\\Delta\\gamma^*={fl['gamma_shift']:+.3f}$ & Fundus Projection Bias", color="white", fontsize=10.5, fontweight="bold")
    ax_i.legend(loc="upper left", fontsize=6.9, facecolor="#131b26", edgecolor="#2e3f57", labelcolor="white")
    ins_i = ax_i.inset_axes([0.52, 0.50, 0.45, 0.42])
    ins_i.set_facecolor("#0f151e")
    ins_i.scatter(av["proj_eccentricity_deg"], av["proj_apparent_total_deg"] - av["proj_true_total_deg"],
                  s=6, color="#4ea8de", alpha=0.7)
    ins_i.axhline(0.0, color="white", lw=0.8, ls="--")
    ins_i.set_xlabel("Retinal eccentricity (deg from pole)", color="#d4dce8", fontsize=6.4)
    ins_i.set_ylabel("Projected $-$ true $\\Theta$ (deg)", color="#d4dce8", fontsize=6.4)
    ins_i.set_title(f"Fundus foreshortening: |bias| {av['proj_angle_bias_central_deg']:.1f}° (<30°) → {av['proj_angle_bias_peripheral_deg']:.1f}° (≥60°)",
                    color="#d4dce8", fontsize=6.4)
    ins_i.tick_params(colors="#d4dce8", labelsize=6.0)

    fig.suptitle(
        "Murray (1926a, 1926b) Minimum-Work Vascular Cube Law ($r_0^3 = r_1^3 + r_2^3$), Bifurcation Angles & Human Retinal Vasculature Validation\n"
        "Pass 3: Fahraeus-Lindqvist $\\eta(r)$ exponent shift (G, I) and 3D hemispherical retinal-cup arteriovenous trees (H)",
        color="white",
        fontsize=12.0,
        fontweight="bold",
        y=0.992,
    )
    fig.savefig(output_png, dpi=140)
    plt.close(fig)

    if not skip_gif:
        # 3D retinal-cup growth animation (arterial + venous trees by generation)
        cup_frames: list[Image.Image] = []
        for max_g in range(0, generations_cli + 1):
            cf = plt.figure(figsize=(6.0, 5.0), dpi=100)
            cf.patch.set_facecolor("#0c1118")
            cax = cf.add_subplot(111, projection="3d")
            cax.set_facecolor("#131b26")
            cax.plot_surface(R_eye * np.sin(vv) * np.cos(uu), R_eye * np.sin(vv) * np.sin(uu), R_eye * np.cos(vv),
                             color="#1f2a3a", alpha=0.35, linewidth=0, antialiased=False, shade=True)
            for s in av["arteries"]:
                if s["gen"] <= max_g:
                    cax.plot([s["p0"][0], s["p1"][0]], [s["p0"][1], s["p1"][1]], [s["p0"][2], s["p1"][2]],
                             color="#e63946", lw=max(0.6, s["r_um"] / 20.0))
            for s in av["veins"]:
                if s["gen"] <= max_g:
                    cax.plot([s["p0"][0], s["p1"][0]], [s["p0"][1], s["p1"][1]], [s["p0"][2], s["p1"][2]],
                             color="#4ea8de", lw=max(0.6, s["r_um"] / 20.0))
            cax.set_xlim(-R_eye, R_eye); cax.set_ylim(-R_eye, R_eye); cax.set_zlim(-0.2 * R_eye, R_eye)
            cax.set_box_aspect((1, 1, 0.6), zoom=1.5); cax.view_init(elev=38, azim=-55 + 6 * max_g); cax.set_axis_off()
            cax.set_title(f"3D retinal cup A/V Murray trees — Gen 0..{max_g}", color="white", fontsize=9.5)
            cf.tight_layout()
            cf.canvas.draw()
            rgba = np.asarray(cf.canvas.buffer_rgba())
            cup_frames.append(Image.fromarray(rgba[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
            plt.close(cf)
        cup_frames[0].save(output_cup_gif, save_all=True, append_images=cup_frames[1:], duration=420, loop=0)

    if not skip_gif:
        frames: list[Image.Image] = []
        for max_g in range(0, 7):
            f_fig, (f_ax1, f_ax2) = plt.subplots(1, 2, figsize=(8.8, 4.4), dpi=105)
            f_fig.patch.set_facecolor("#0c1118")
            for ax in (f_ax1, f_ax2):
                ax.set_facecolor("#131b26")
                ax.tick_params(colors="#d4dce8", labelsize=8)
                for sp in ax.spines.values():
                    sp.set_color("#2e3f57")

            for s in tree_sym:
                if s["gen"] <= max_g:
                    f_ax1.plot([s["x0"], s["x1"]], [s["y0"], s["y1"]], color="#e63946",
                               lw=max(0.75, s["r_um"] / 13.0), solid_capstyle="round")
            f_ax1.set_xlim(-4.8, 4.8)
            f_ax1.set_ylim(-0.3, 7.6)
            f_ax1.set_aspect("equal")
            f_ax1.set_title(f"Murray Cube Tree ($\\gamma=3, \\Theta=74.93^\\circ$) — Gen 0..{max_g}", color="white", fontsize=9.2)

            for s in tree_asym:
                if s["gen"] <= max_g:
                    f_ax2.plot([s["x0"], s["x1"]], [s["y0"], s["y1"]], color="#2ec4b6",
                               lw=max(0.75, s["r_um"] / 13.0), solid_capstyle="round")
            f_ax2.set_xlim(-4.8, 4.8)
            f_ax2.set_ylim(-0.3, 7.6)
            f_ax2.set_aspect("equal")
            f_ax2.set_title(f"Asymmetric Arcade ($\\alpha=0.72, \\Theta=78.9^\\circ$) — Gen 0..{max_g}", color="white", fontsize=9.2)

            f_fig.tight_layout()
            f_fig.canvas.draw()
            rgba = np.asarray(f_fig.canvas.buffer_rgba())
            frames.append(Image.fromarray(rgba[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
            plt.close(f_fig)

        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=340,
            loop=0,
        )

    return {"empirical": emp, "sweep": sweep, "segmentation": seg, "fahraeus_lindqvist": fl,
            "hemisphere_av": av, "hemisphere_av_control": av_ctrl}


# =============================================================================
# Section 7: MorphoEval Quantitative Measurement Suite & Self-Test
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute comprehensive quantitative morphometrics, uncertainty/dispersion stats,
    and negative-control ablation deltas for `morphoeval/measure_outputs.py`."""
    ref_data = verify_dataset_integrity()
    sym = murray_symmetric_bifurcation(R0_ROOT_UM_DEFAULT)
    emp = evaluate_empirical_retinal_junctions(ref_data)
    sweep = run_gamma_power_sweep(generations=7, r0_um=R0_ROOT_UM_DEFAULT)
    seg = segment_retinal_vasculature_specimen(SPECIMEN_IMG_PATH)
    h_murray = evaluate_generational_hierarchy(generations=7, r0_um=R0_ROOT_UM_DEFAULT, gamma=3.0)
    fl = run_fl_exponent_sweep(generations=7, r0_um=R0_ROOT_UM_DEFAULT)
    av = evaluate_hemispherical_av_coupling(coupling="repulsive")
    av_ctrl = evaluate_hemispherical_av_coupling(coupling="none")

    q0 = murray_conductance_constant() * ((R0_ROOT_UM_DEFAULT * 1e-4) ** 3)
    r_opt = murray_optimal_radius(q0)
    pv, pm, _ = poiseuille_metabolic_power(r_opt, q0)

    return [
        {
            "name": "symmetric daughter radius ratio r_1 / r_0",
            "value": round(sym["daughter_ratio"], 6),
            "unit": "ratio",
            "method": "Murray (1926a) cube law r_0^3 = 2 * r_1^3 => r_1/r_0 = 2^(-1/3)",
            "reference_value": 0.793701,
            "reference_source": "Murray (1926a) PNAS 12:207-214; Sherman (1981) J. Gen. Physiol. 78:431",
            "note": f"area expansion ratio beta = (r_1^2 + r_2^2)/r_0^2 = {sym['area_expansion_beta']:.6f} (ref 2^(1/3) = 1.259921)",
            "derived_from": "script",
        },
        {
            "name": "symmetric total bifurcation angle (theta_1 + theta_2)",
            "value": round(sym["total_angle_deg"], 4),
            "unit": "deg",
            "method": "2 * arccos(2^(-1/3)) from Murray (1926b) Eq. 5 vs Zamir 90 deg and Steiner 120 deg controls",
            "reference_value": 74.9346,
            "reference_source": "Murray (1926b) J. Gen. Physiol. 9:835-841 Eq. 5; Zamir (1976) J. Theor. Biol. 62:227",
            "note": "half-angle theta_1 = theta_2 = 37.4673 deg; Steiner length-minimizing control = 120.0 deg",
            "derived_from": "script",
        },
        {
            "name": "metabolic-to-viscous power ratio at optimal radius P_met(r*) / P_visc(r*)",
            "value": round(float(pm / pv), 6),
            "unit": "ratio",
            "method": "stationarity dP/dr = 0 evaluated at r* = murray_optimal_radius(Q) (Lean 4 theorem murray_stationarity_viscous_twice_metabolic)",
            "reference_value": 2.0,
            "reference_source": "Murray (1926a) PNAS 12:207-214 Eq. 4",
            "derived_from": "script",
        },
        {
            "name": "empirical human retinal bifurcation angle RMSE (Murray prediction vs J01-J08)",
            "value": round(emp["rmse_total_angle_deg"], 4),
            "unit": "deg",
            "method": f"RMSE and Pearson r across N={emp['n_junctions']} asymmetric human retinal arterial junctions (alpha in 0.56..0.99)",
            "reference_value": 0.0,
            "reference_source": "Zamir, Medeiros & Cunningham (1979) Invest. Ophthalmol. Vis. Sci. 18:659; Murray (1926b)",
            "note": f"branch angle Pearson r = {emp['pearson_r_branch_angles']:.4f} (p={emp['pearson_p_branch_angles']:.2e}), branch RMSE = {emp['rmse_branch_angle_deg']:.3f} deg",
            "derived_from": "script",
        },
        {
            "name": "empirical retinal cube-law conservation ratio (r_1^3 + r_2^3)^(1/3) / r_0",
            "value": round(emp["cube_law_ratio_mean"], 4),
            "unit": "ratio",
            "method": "mean ± SD (CV) of (r_1^3 + r_2^3)^(1/3) / r_0 vs Da Vinci square-law ablation control (r_1^2 + r_2^2)^(1/2) / r_0",
            "reference_value": 1.0,
            "reference_source": "Murray (1926a) PNAS 12:207-214; Sherman (1981) J. Gen. Physiol. 78:431",
            "note": f"std = {emp['cube_law_ratio_std']:.4f}, CV = {emp['cube_law_ratio_cv']:.4f}; Da Vinci gamma=2 control gives {emp['davinci_ratio_mean']:.4f} (+11.7% bias)",
            "derived_from": "script",
        },
        {
            "name": "Da Vinci area-preserving control (gamma=2) excess vascular tree power penalty",
            "value": round(sweep["davinci_excess_power_pct"], 3),
            "unit": "%",
            "method": "100 * (P_tree(gamma=2.0) - P_tree(gamma=3.0)) / P_tree(gamma=3.0) across 7-generation arterial tree",
            "reference_value": 0.0,
            "reference_source": "Murray (1926a) PNAS 12:207-214; Sherman (1981) J. Gen. Physiol. 78:431",
            "note": f"numerical minimum found at gamma* = {sweep['gamma_opt']:.3f}; wall shear CV at gamma=3 is {h_murray['wall_shear_cv']:.2e} vs +100% drift at gamma=2",
            "derived_from": "script",
        },
        {
            "name": "segmented retinal fundus specimen Y-junction bifurcation angle (mean)",
            "value": round(seg["segmented_angle_mean_deg"], 3),
            "unit": "deg",
            "method": f"green-channel top-hat vessel segmentation + medial-axis skeletonization across N={seg['total_junctions_segmented']} Y-junctions in retinal_vasculature_specimen.jpg",
            "reference_value": 74.9346,
            "reference_source": "Murray (1926b) J. Gen. Physiol. 9:835-841; Wikimedia Commons File:Retinal_vessel_tortuosity.jpg",
            "note": f"std = {seg['segmented_angle_std_deg']:.2f} deg, CV = {seg['segmented_angle_cv']:.3f}, segmented cube ratio = {seg['segmented_cube_ratio_mean']:.3f} ± {seg['segmented_cube_ratio_std']:.3f}, tortuosity B/A ratio = {seg['tortuosity_ratio_b_over_a']:.3f}",
            "derived_from": "output",
        },
        {
            "name": "generational wall shear stress tau_w across 7 Sherman (1981) arteriole levels",
            "value": round(h_murray["wall_shear_mean_dyn_cm2"], 4),
            "unit": "dyn/cm^2",
            "method": "tau_w = 4*eta*Q_g / (pi*r_g^3) = sqrt(eta*b) across generations g=0..6",
            "reference_value": 8.2189,
            "reference_source": "Murray (1926a) PNAS 12:207-214; Sherman (1981) J. Gen. Physiol. 78:431",
            "note": f"generational shear CV = {h_murray['wall_shear_cv']:.2e} (exact scale invariance)",
            "derived_from": "script",
        },
        # ---------------- Pass 3: Fahraeus-Lindqvist non-Newtonian viscosity ----------------
        {
            "name": "Fahraeus-Lindqvist relative apparent viscosity eta_rel at D = 10 um (H_D = 0.45)",
            "value": round(float(fahraeus_lindqvist_relative_viscosity(10.0)), 4),
            "unit": "ratio (eta_app / eta_plasma)",
            "method": "Pries, Neuhaus & Gaehtgens (1992) Eq. 1 in-vitro fit: 220 e^{-1.3 D} + 3.2 - 2.44 e^{-0.06 D^0.645}",
            "reference_value": 1.33,
            "reference_source": "Pries AR, Neuhaus D, Gaehtgens P (1992) Am. J. Physiol. 263:H1770-H1778 (Fig. 2/Eq. 1); Fahraeus & Lindqvist (1931) Am. J. Physiol. 96:562",
            "note": f"bulk asymptote eta_rel = {FL_ETA_BULK_REL}; at D=50 um eta_rel = {float(fahraeus_lindqvist_relative_viscosity(50.0)):.3f}, at D=164 um (root) {float(fahraeus_lindqvist_relative_viscosity(2*R0_ROOT_UM_DEFAULT)):.3f}; parity check of the implemented fit (not an independent datum)",
            "derived_from": "script",
        },
        {
            "name": "Fahraeus-Lindqvist power-minimising tree exponent gamma*_FL (7-generation sweep)",
            "value": round(fl["gamma_opt_fahraeus_lindqvist"], 4),
            "unit": "exponent",
            "method": "argmin_gamma of total Poiseuille(eta(r_g)) + metabolic tree power over gamma in [2.2, 3.6] (141 pts, parabolic refinement) vs Newtonian control",
            "reference_value": 2.7,
            "reference_source": "Pass 2 audit remediation target (gamma ~ 2.7 for pre-capillary arterioles under F-L rheology); Newtonian Murray (1926a) control = 3.0",
            "note": f"Newtonian control gamma* = {fl['gamma_opt_newtonian']:.4f}; shift Delta gamma* = {fl['gamma_shift']:+.4f}; stationarity partition 2 P_met = (4 - epsilon) P_visc (Lean 4 fahraeus_lindqvist_stationarity_shift)",
            "derived_from": "script",
        },
        {
            "name": "Fahraeus-Lindqvist symmetric junction exponent gamma_j at pre-capillary r_0 = 25 um",
            "value": round(fl["junction_25um"]["gamma_junction"], 4),
            "unit": "exponent",
            "method": "solve Q*(r_1) = Q*(r_0)/2 on the generalised stationarity curve Q*(r) = (pi/2) r^3 sqrt(b / (eta(r)(4 - epsilon))) ; gamma_j = ln2 / ln(r_0/r_1)",
            "reference_value": 3.0,
            "reference_source": "Murray (1926a) Newtonian cube law (control); Pries et al. (1992) eta(D)",
            "note": f"epsilon(25 um) = {fl['junction_25um']['epsilon_parent']:.4f}, P_met/P_visc = {fl['junction_25um']['met_over_visc_parent']:.4f} (<2); mean local gamma_eff over r<25 um = {fl['gamma_eff_precapillary_mean']:.4f}, min = {fl['gamma_eff_precapillary_min']:.4f}; at r_0 = 82 um gamma_j = {fl['junction_82um']['gamma_junction']:.4f}",
            "derived_from": "script",
        },
        # ---------------- Pass 3: 3D hemispherical retinal cup A/V coupling ----------------
        {
            "name": "3D retinal-cup geodesic tree sphericity residual max | |p| - R |",
            "value": float(f"{av['sphericity_residual_mm']:.3e}"),
            "unit": "mm",
            "method": f"all {av['n_segments_artery'] + av['n_segments_vein']} arterial + venous segment end-points of the hemispherical Murray trees checked against R = {EYE_RADIUS_MM_DEFAULT} mm",
            "reference_value": 0.0,
            "reference_source": "Geodesic construction on the schematic emmetropic globe (inner radius 12 mm, axial length ~24 mm)",
            "note": f"max polar reach {av['max_polar_deg']:.1f} deg (ora serrata limit {RETINA_POLAR_LIMIT_DEG} deg); geodesic/chord length ratio {av['geodesic_over_chord_length']:.5f}; arterial geodesic length {av['arterial_geodesic_length_mm']:.0f} mm",
            "derived_from": "script",
        },
        {
            "name": "coupled arteriovenous root caliber ratio AVR (artery/vein) on the retinal cup",
            "value": round(av["root_avr"], 4),
            "unit": "ratio",
            "method": "venous Murray tree root radius set by Q_a = Q_v coupling with AVR; venous constants k_v/k_a = sum r_tip^3 / sum R_tip^3 (Lean 4 arteriovenous_coupled_cube_balance)",
            "reference_value": 0.67,
            "reference_source": "Hubbard LD et al. (1999) Ophthalmology 106:2269-2280 (ARIC retinal AVR ~ 2/3); Murray (1926a)",
            "note": f"k_v/k_a = {av['k_v_over_k_a']:.4f} => venous wall shear tau_v = {av['tau_wall_vein_dyn_cm2']:.3f} dyn/cm^2 vs arterial {av['tau_wall_artery_dyn_cm2']:.3f}; venous metabolic constant b_v = {av['b_vein_erg_s_cm3']:.1f} erg/s/cm^3; AVR is an input (set by --avr), so this is a parity/propagation check",
            "derived_from": "script",
        },
        {
            "name": "arteriovenous interdigitation index of terminal tips on the retinal cup (repulsive coupling)",
            "value": round(av["interdigitation_index"], 4),
            "unit": "fraction",
            "method": "fraction of azimuthally adjacent terminal-tip pairs (about the optic axis) with opposite vessel type; 1.0 = perfect A/V alternation, 0.5 = random mixing",
            "reference_value": 1.0,
            "reference_source": "Idealised alternating arteriovenous territories (reduced model; no quantitative human datum vendored)",
            "note": f"uncoupled control = {av_ctrl['interdigitation_index']:.4f}; nearest-tip A/V pairing fraction {av['av_pairing_fraction']:.3f} (ctrl {av_ctrl['av_pairing_fraction']:.3f}); mean A->V tip distance {av['tip_a_to_v_distance_mean_mm']:.3f} mm (ctrl {av_ctrl['tip_a_to_v_distance_mean_mm']:.3f}); projected A/V crossings {av['av_crossings_projected']} (ctrl {av_ctrl['av_crossings_projected']}). Index stays near random-mixing level: residual for Pass 4.",
            "derived_from": "script",
        },
        {
            "name": "fundus-projection foreshortening bias of bifurcation angle |Theta_proj - Theta_true| (peripheral, >= 60 deg eccentricity)",
            "value": round(av["proj_angle_bias_peripheral_deg"], 3),
            "unit": "deg",
            "method": "orthographic (fundus-camera) projection of 3D tangent-plane Murray bifurcations on the retinal cup vs their true geodesic angles, binned by eccentricity",
            "reference_value": 0.0,
            "reference_source": "Murray (1926b) angles are tangent-plane quantities; the planar fundus specimen reports projected angles",
            "note": f"central (<30 deg) |bias| = {av['proj_angle_bias_central_deg']:.3f} deg, all-tree mean |bias| = {av['proj_angle_bias_abs_mean_deg']:.3f} deg, signed mean = {av['proj_angle_bias_mean_deg']:+.3f} deg; bounds how much of the segmented specimen's 79.3 deg vs 74.9 deg gap can be projection artefact",
            "derived_from": "script",
        },
    ]


def self_test() -> bool:
    """Comprehensive Pass 2 self-test suite verifying:
      1. Cryptographic SHA-256 integrity of `retinal_vasculature_specimen.jpg` and JSON parity
      2. Exact Murray (1926a) stationarity P_met(r*) / P_visc(r*) = 2.0 and tau_w = sqrt(eta*b)
      3. Exact Murray (1926b) symmetric & asymmetric bifurcation cosine laws against reference JSON
      4. Multi-generation exponent sweep global minimum at gamma* = 3.000 vs Da Vinci gamma = 2.0 control
      5. Empirical retinal junction validation (RMSE < 0.35 deg, Pearson r > 0.995)
      6. Real human retinal fundus specimen segmentation (Y-junction count, angle & cube-law ratio)
      7. Two-run bitwise determinism of vascular tree generation and morphometrics
    """
    ref_data = verify_dataset_integrity()
    phys = ref_data["murray_1926_minimum_work_model"]["physical_parameters"]

    # 1. Symmetric cube law & angle parity with reference JSON
    sym = murray_symmetric_bifurcation(R0_ROOT_UM_DEFAULT)
    assert abs(sym["daughter_ratio"] - phys["symmetric_daughter_radius_ratio"]) < 1e-12
    assert abs(sym["area_expansion_beta"] - phys["symmetric_area_expansion_ratio_beta"]) < 1e-12
    assert abs(sym["total_angle_deg"] - phys["symmetric_total_angle_deg"]) < 1e-9
    assert sym["cubic_residual"] < 1e-9

    # 2. Stationarity P_met(r*) = 2 * P_visc(r*) and scale-invariant wall shear stress
    q0 = murray_conductance_constant() * ((R0_ROOT_UM_DEFAULT * 1e-4) ** 3)
    r_opt = murray_optimal_radius(q0)
    assert abs(r_opt * 1e4 - R0_ROOT_UM_DEFAULT) < 1e-9
    pv, pm, _ = poiseuille_metabolic_power(r_opt, q0)
    assert abs((pm / pv) - 2.0) < 1e-11

    # 3. Asymmetric bifurcation table parity
    for row in ref_data["asymmetry_bifurcation_table"]:
        b = murray_asymmetric_bifurcation(1.0, alpha=row["asymmetry_alpha"], gamma=3.0)
        assert abs(b["r1_over_r0"] - row["r1_over_r0"]) < 1e-5
        assert abs(b["theta1_deg"] - row["theta1_deg"]) < 1e-2
        assert abs(b["theta2_deg"] - row["theta2_deg"]) < 1e-2

    # 4. Exponent sweep global minimum at gamma* = 3.0 and Da Vinci ablation penalty
    sweep = run_gamma_power_sweep(generations=7, r0_um=R0_ROOT_UM_DEFAULT)
    assert abs(sweep["gamma_opt"] - 3.0) < 1e-9
    assert sweep["davinci_excess_power_pct"] > 20.0
    assert sweep["quartic_excess_power_pct"] > 10.0

    # 5. Empirical retinal junction goodness-of-fit
    emp = evaluate_empirical_retinal_junctions(ref_data)
    assert emp["rmse_total_angle_deg"] < 0.35
    assert emp["pearson_r_branch_angles"] > 0.995
    assert abs(emp["cube_law_ratio_mean"] - 1.0) < 0.02

    # 6. Biological specimen segmentation sanity checks
    seg = segment_retinal_vasculature_specimen(SPECIMEN_IMG_PATH)
    assert seg["total_junctions_segmented"] >= 6
    assert 65.0 <= seg["segmented_angle_mean_deg"] <= 85.0
    assert 0.85 <= seg["segmented_cube_ratio_mean"] <= 1.15
    assert seg["tortuosity_ratio_b_over_a"] > 1.01

    # 7. Two-run bitwise determinism
    t1 = build_vascular_tree_2d(generations=6, r0_um=82.0, alpha=0.80, gamma=3.0)
    t2 = build_vascular_tree_2d(generations=6, r0_um=82.0, alpha=0.80, gamma=3.0)
    arr1 = np.array([[s["x0"], s["y0"], s["x1"], s["y1"], s["r_um"]] for s in t1])
    arr2 = np.array([[s["x0"], s["y0"], s["x1"], s["y1"], s["r_um"]] for s in t2])
    assert np.allclose(arr1, arr2, atol=0.0, rtol=0.0)

    # 8. (Pass 3) Fahraeus-Lindqvist viscosity: shape, Newtonian limit, and exponent shift
    eta_rel = fahraeus_lindqvist_relative_viscosity(np.array([10.0, 50.0, 164.0, 2000.0]))
    assert eta_rel[0] < eta_rel[1] < eta_rel[2] < eta_rel[3] < FL_ETA_BULK_REL + 1e-6   # monotone toward the bulk asymptote
    assert abs(eta_rel[0] - 1.33) < 0.02                                                # ~1.33 at D = 10 um
    r_test = np.array([5e-4, 25e-4, 82e-4])
    assert np.allclose(apparent_viscosity_poise(r_test, viscosity_model="newtonian"), ETA_POISE_DEFAULT)
    assert np.allclose(fl_local_optimal_exponent(r_test, viscosity_model="newtonian"), 3.0, atol=1e-6)
    assert np.allclose(fl_optimal_flow(r_test, viscosity_model="newtonian"),
                       murray_conductance_constant() * r_test ** 3, rtol=1e-9)        # Newtonian limit = Murray's Q = k_M r^3
    assert np.all(viscosity_log_slope(r_test) > 0.0)                                   # epsilon > 0 across arterioles
    fl = run_fl_exponent_sweep(generations=7, r0_um=R0_ROOT_UM_DEFAULT)
    assert abs(fl["gamma_opt_newtonian"] - 3.0) < 5e-3
    assert 2.6 < fl["gamma_opt_fahraeus_lindqvist"] < 2.95                              # shift toward ~2.7
    assert fl["gamma_shift"] < -0.1
    j25 = fl["junction_25um"]
    assert 2.7 < j25["gamma_junction"] < 3.0
    assert abs(j25["met_over_visc_parent"] - (4.0 - j25["epsilon_parent"]) / 2.0) < 1e-3   # 2 P_met = (4 - eps) P_visc

    # 9. (Pass 3) Hemispherical retinal cup A/V trees: sphericity, reach, coupling, determinism
    av = evaluate_hemispherical_av_coupling(coupling="repulsive")
    av0 = evaluate_hemispherical_av_coupling(coupling="none")
    assert av["sphericity_residual_mm"] < 1e-9
    assert av["max_polar_deg"] <= RETINA_POLAR_LIMIT_DEG
    assert av["n_tips_artery"] >= 64 and av["n_tips_vein"] >= 64
    assert abs(av["root_avr"] - AVR_DEFAULT) < 1e-12
    assert av["tau_wall_vein_dyn_cm2"] < av["tau_wall_artery_dyn_cm2"]                 # wider veins -> lower venous shear
    assert av["tip_a_to_v_distance_mean_mm"] > av0["tip_a_to_v_distance_mean_mm"]      # repulsive coupling separates A/V territories
    assert 0.0 <= av["interdigitation_index"] <= 1.0
    assert av["proj_angle_bias_peripheral_deg"] > av["proj_angle_bias_central_deg"]     # foreshortening grows with eccentricity
    for s in av["arteries"]:
        if "bif_total_deg" in s:
            assert abs(s["bif_total_deg"] - murray_asymmetric_bifurcation(s["r_um"], alpha=0.70)["total_angle_deg"]) < 1e-9
    h1 = build_vascular_tree_hemisphere(generations=5, seed=1926, heading_jitter_deg=3.0)
    h2 = build_vascular_tree_hemisphere(generations=5, seed=1926, heading_jitter_deg=3.0)
    assert np.allclose(np.array([s["p1"] for s in h1]), np.array([s["p1"] for s in h2]), atol=0.0, rtol=0.0)

    print(
        f"[self-test] 1959-murray-vascular-cube-law-branching.py (Pass 3 v3) OK — "
        f"r1/r0={sym['daughter_ratio']:.6f}, Theta_sym={sym['total_angle_deg']:.4f} deg, "
        f"gamma*={sweep['gamma_opt']:.2f}, retinal_RMSE={emp['rmse_total_angle_deg']:.3f} deg, "
        f"segmented_junctions={seg['total_junctions_segmented']} (mean={seg['segmented_angle_mean_deg']:.2f} deg), "
        f"FL gamma*={fl['gamma_opt_fahraeus_lindqvist']:.3f} (gamma_j(25um)={j25['gamma_junction']:.3f}), "
        f"3D cup: {av['n_segments_artery']}+{av['n_segments_vein']} segs, sphericity={av['sphericity_residual_mm']:.1e} mm, "
        f"II={av['interdigitation_index']:.3f}, AV crossings={av['av_crossings_projected']}"
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Run comprehensive Pass 2 self-test suite")
    ap.add_argument("--skip-gif", action="store_true", help="Skip animated GIF rendering")
    ap.add_argument("--alpha", type=float, default=1.0, help="Daughter radius asymmetry ratio alpha = r2/r1 in (0, 1]")
    ap.add_argument("--gamma", type=float, default=GAMMA_MURRAY, help="Vascular radius power-law exponent gamma")
    ap.add_argument("--eta", type=float, default=ETA_POISE_DEFAULT, help="Blood dynamic viscosity eta (Poise)")
    ap.add_argument("--b", type=float, default=METABOLIC_B_DEFAULT, help="Metabolic volume maintenance cost b (erg/s/cm^3)")
    ap.add_argument("--r0-um", type=float, default=R0_ROOT_UM_DEFAULT, help="Root arteriole lumen radius r0 (um)")
    ap.add_argument("--generations", type=int, default=6, help="Number of bifurcation generations to grow")
    ap.add_argument("--seed", type=int, default=1926, help="Deterministic RNG seed (reserved; all default models are closed-form/deterministic)")
    ap.add_argument("--eye-radius-mm", type=float, default=EYE_RADIUS_MM_DEFAULT, help="Inner globe radius of the hemispherical retinal cup (mm)")
    ap.add_argument("--avr", type=float, default=AVR_DEFAULT, help="Arteriole-to-venule root caliber ratio coupling the venous tree to the arterial tree")
    ap.add_argument("--coupling", choices=["none", "repulsive"], default="repulsive", help="Arteriovenous growth coupling rule on the retinal cup")
    ap.add_argument("--output-png", type=str, default=str(OUTPUT_PNG), help="Output comparison PNG path")
    ap.add_argument("--output-gif", type=str, default=str(OUTPUT_GIF), help="Output animated GIF path")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    render_outputs(
        output_png=Path(args.output_png),
        output_gif=Path(args.output_gif),
        skip_gif=args.skip_gif,
        alpha_cli=args.alpha,
        gamma_cli=args.gamma,
        r0_um_cli=args.r0_um,
        generations_cli=args.generations,
        eta_cli=args.eta,
        b_cli=args.b,
        eye_radius_mm_cli=args.eye_radius_mm,
        avr_cli=args.avr,
        coupling_cli=args.coupling,
        seed_cli=args.seed,
    )
    print(f"Rendered {args.output_png}, {args.output_gif} and {OUTPUT_CUP_GIF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
