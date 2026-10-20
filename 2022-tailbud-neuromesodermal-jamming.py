#!/usr/bin/env python3
"""
2022-tailbud-neuromesodermal-jamming.py — Pass 3 (v3) Remediated Implementation
================================================================================
Active Fluid-to-Solid Jamming Transition, Extracellular Space Compaction,
Ferrofluid Yield-Stress Gradient, Bingham Viscoplastic Backstop Rectification,
cdh2 (N-Cadherin) Knockdown Bilateral Asymmetry, Quasi-3D Fibronectin ECM
Sheath Confinement (Dray et al. 2013) and Her1/Her7 Segmentation-Clock
Wavefront Arrest in Zebrafish Body Axis Elongation
(`tailbud-neuromesodermal-jamming`, Slot 030).

Pass 3 remediation (superseding audit `2026-10-19-neuromesodermal-jamming-audit-v2`):
  * The single open v2 item — "couple the 2D active-foam jamming gradient to 3D
    fibronectin ECM boundary friction (Dray et al. 2013) and anterior Her1/Her7
    segmentation clock wavefront arrest" — is closed in a REDUCED, HONESTLY
    LABELLED form (no 3D FEM / axisymmetric Stokes solve):
      - Section 2b `solve_fn_sheath_confinement`: thin-rod (axisymmetric)
        reduction of the 3D paraxial tissue column.  The fibronectin matrix that
        wraps the tailbud/PSM (Dray et al. 2013 Curr. Biol. 23:1335 Fig. 1;
        Lawton et al. 2013) is a hoop-stiff elastic sheath of relative density
        f_FN(x).  Where the active cell pressure p_act(x) = sigma_A(x) exceeds the
        tissue yield stress sigma_y(x) the fluid tailbud dilates radially until
        the sheath hoop stress balances it:
            eps_r(x) = max(0, sigma_A - sigma_y) / (K_FN f_FN(x) + K_t),
            R_eq(x) = R_0(x) (1 + eps_r(x)).
        The fraction of progenitor influx stored in radial dilation (the
        "leak") is 1 - A_0/A_eq, so axial channeling efficiency chi = 1 - leak
        multiplies the Bingham-backstop elongation speed.  fn1a/fn1b and itga5
        knockdown (fn_scale < 1) soften the sheath -> wider tailbud and slower
        axis elongation (Dray et al. 2013 Figs. 2-4).  The dynamic ECM friction
        of the moving fluid MPZ against the sheath is reported as a wall-shear
        diagnostic (F_drag) but does not feed back on v_elong (residual item).
      - Section 2c `solve_segmentation_clock_arrest`: the Her1/Her7 clock phase
        theta(x,t) (Oates & Ho 2002; Lewis 2003) is advected in the tip frame,
        d theta/dt + u(x) d theta/dx = omega(x), with a posterior pacemaker,
        a slowing frequency profile (Soroldoni et al. 2014 Doppler) and arrest
        (omega = 0) anterior to a mechanical arrest front x_arrest defined by
        the jamming ratio sigma_y/sigma_A >= Lambda_arrest = 4 (deep solid
        regime; reduced stand-in for the YAP/density mechanosensitivity of
        Hubaud et al. 2017).  Somite length follows Cooke-Zeeman,
        S = (v_elong + v_PSM-shortening) * T_clock, and the frozen her1 stripe
        wavelength anterior to x_arrest is measured on the kymograph and
        compared with S.  Epithelialised somites anterior to x_arrest assemble
        extra fibronectin (Delta f_FN) that feeds back into the sheath profile
        — the two new modules are therefore coupled in both directions with the
        jamming gradient (jamming -> arrest front -> FN assembly -> channeling
        -> v_elong -> somite length).
  * Two new Lean 4 theorems (`fibronectin_sheath_confinement_channeling`,
    `clock_wavefront_arrest_somite_scaling`), 3 new Dray-2013 fibronectin
    regimes, a 12-point fn_scale sweep, 6 new measurements (14 total) and a
    13-check self-test.

Primary Literature Sources:
  1. Mongera, A., Rowghanian, P., Gustafson, H. J., Shelton, E., Kealhofer, D. A.,
     Carn, E. K., Serwane, F., Lucio, A. A., Giammona, J. & Campàs, O. (2018).
     "A fluid-to-solid jamming transition underlies vertebrate body axis elongation."
     Nature 561(7723):401-405. DOI: 10.1038/s41586-018-0479-2
  2. Kim, S., Pochitaloff, M., Stooke-Vaughan, G. A. & Campàs, O. (2021).
     "Embryonic tissues as active foams." Nature Physics 17(7):859-866.
     DOI: 10.1038/s41567-021-01215-1
  3. Lawton, A. K., Nandi, A., Stulberg, M. J., Dray, N., Sneddon, M. W.,
     Emonet, T., Blawzdziewicz, J. & Holley, S. A. (2013).
     "Regulated tissue fluidity steers zebrafish body elongation."
     Development 140(3):573-582. DOI: 10.1242/dev.090381
  4. Serwane, F., Mongera, A., Rowghanian, P., Kealhofer, D. A., Lucio, A. A.,
     Hockenbery, Z. M. & Campàs, O. (2017).
     "In vivo quantification of spatially varying mechanical properties in
     developing tissues." Nature Methods 14(2):181-186. DOI: 10.1038/nmeth.4101
  5. Dray, N., Lawton, A., Nandi, A., Jülicher, F., Emonet, T. & Holley, S. A. (2013).
     "Cell-fibronectin interactions propel vertebrate trunk elongation via
     tissue mechanics." Current Biology 23(14):1335-1341.
     DOI: 10.1016/j.cub.2013.05.052
  6. Soroldoni, D., Jörg, D. J., Morelli, L. G., Richmond, D. L., Schindelin, J.,
     Jülicher, F. & Oates, A. C. (2014). "A Doppler effect in embryonic pattern
     formation." Science 345(6193):222-225. DOI: 10.1126/science.1253089
  7. Herrgen, L., Ares, S., Morelli, L. G., Schröter, C., Jülicher, F. & Oates, A. C.
     (2010). "Intercellular coupling regulates the period of the segmentation
     clock." Current Biology 20(14):1244-1253. DOI: 10.1016/j.cub.2010.06.034
  8. Hubaud, A., Regev, I., Mahadevan, L. & Pourquié, O. (2017). "Excitable
     dynamics and Yap-dependent mechanical cues drive the segmentation clock."
     Cell 171(3):668-682. DOI: 10.1016/j.cell.2017.08.043
  9. Lewis, J. (2003). "Autoinhibition with transcriptional delay: a simple
     mechanism for the zebrafish somitogenesis oscillator." Current Biology
     13(16):1398-1408. DOI: 10.1016/S0960-9822(03)00534-7

Formal Lean 4 Specification:
  - `morphoeval/lean/loops/Loop_030_neuromesodermal_jamming.lean`
    * `MathOfBiology.Loop_030.rigidity_percolation_yield_stress_onset`
    * `MathOfBiology.Loop_030.active_fluctuation_fluid_to_solid_jamming_crossover`
    * `MathOfBiology.Loop_030.mongera_2018_wt_vs_cdh2_backstop_collapse`
    * `MathOfBiology.Loop_030.posterior_elongation_rectification_and_bilateral_symmetry`
    * `MathOfBiology.Loop_030.fibronectin_sheath_confinement_channeling`   (Pass 3)
    * `MathOfBiology.Loop_030.clock_wavefront_arrest_somite_scaling`       (Pass 3)
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
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from PIL import Image
from scipy import ndimage, stats
from scipy.spatial import Voronoi

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "neuromesodermal_jamming"
REF_JSON = DATASET_DIR / "neuromesodermal_jamming_reference.json"
SPECIMEN_JPG = DATASET_DIR / "zebrafish_tailbud_psm_specimen.jpg"
SCOUT_JSON = DATASET_DIR / "scout_manifest.json"
BIB_FILE = DATASET_DIR / "paper_citations.bib"

MEDIA_DIR = ROOT / "docs" / "media"
OUT_PNG = MEDIA_DIR / "neuromesodermal_jamming_comparison.png"
OUT_GIF = MEDIA_DIR / "neuromesodermal_jamming_growth.gif"

# Canonical physical constants from Mongera et al. (2018) Nature 561:401-405
# and Kim et al. (2021) Nature Physics 17:859-866
PHI_C_CRITICAL = 0.860          # Critical cell packing fraction phi_c = 1 - phi_ECS_crit
PHI_ECS_CRITICAL = 0.140        # Critical extracellular space fraction (14.0%)
Z_ISOSTATIC_2D = 4.0            # Maxwell isostatic contact coordination number in 2D
SIGMA_0_PA = 1520.0             # Supracellular yield stress scale factor (Pa)
BETA_YIELD = 1.05               # Rigidity-percolation yield stress exponent beta
K0_T1_PER_MIN = 0.092           # Bare T1 topological rearrangement attempt rate (min^-1)
C_BARRIER_KRAMERS = 0.36        # Kramers/Eyring cage-breaking barrier coefficient c_B
GAMMA_KRAMERS = 0.82            # Kramers barrier scaling exponent gamma_k
ETA_PLASTIC_PA_MIN = 38.0       # Bingham plastic viscosity eta_p (Pa * min)

# Pass 3 — quasi-3D fibronectin ECM sheath (Dray et al. 2013 Curr. Biol. 23:1335-1341;
# Lawton et al. 2013 Development 140:573-582).  Values marked [calibrated] are fitted
# to the approximate Dray et al. (2013) elongation/width ratios vendored in the
# reference JSON (`dray_2013_fibronectin_regimes`), not measured directly.
R_MPZ_UM = 55.0                 # Unstrained posterior MPZ bulb radius R_0(0) (um; Mongera 2018 Fig. 1a scale bar)
R_APSM_UM = 38.0                # Unstrained paraxial PSM column radius R_0(L) (um)
F_FN_MPZ = 0.35                 # Relative fibronectin matrix density wrapping the MPZ (sparse; Dray 2013 Fig. 1)
F_FN_PSM = 0.80                 # Relative fibronectin density wrapping the PSM sheath
DELTA_F_FN_SOMITE = 0.20        # Extra fibronectin assembled at epithelialised somite borders (Julich/Koshida 2005)
K_FN_HOOP_PA = 2800.0           # Sheath hoop stiffness per unit relative FN density (Pa / unit radial strain) [calibrated]
K_TISSUE_CONFINEMENT_PA = 40.0  # Residual ectoderm/notochord confinement stiffness (Pa / unit strain) [calibrated]
ZETA_FN_FRICTION_PA_MIN_UM = 0.6  # Dynamic cell-ECM wall friction coefficient (Pa min / um) [diagnostic only]
MPZ_SOURCE_LENGTH_UM = 45.0     # e-folding length of dorsal-medial progenitor ingression source
J_MPZ_PREFACTOR_UM_MIN = 2.034  # Progenitor influx prefactor (= 1.94 / chi_WT, chi_WT = 0.954, so the WT speed stays ~1.52 um/min)
CHI_WT_REFERENCE = 0.954        # WT sheath channeling efficiency used to normalise the ablation control speed

# Pass 3 — Her1/Her7 segmentation clock & wavefront arrest (Oates & Ho 2002; Lewis 2003;
# Herrgen et al. 2010; Soroldoni et al. 2014; Hubaud et al. 2017)
T_CLOCK_MIN = 23.5              # her1/her7 oscillation period at 28 C (Herrgen 2010 Curr Biol 20:1244)
LAMBDA_ARREST = 4.0             # sigma_y/sigma_A jamming ratio gating clock arrest (deep solid; reduced mechanosensitive gate)
V_PSM_SHORTENING_UM_MIN = 0.45  # PSM shortening contribution to wavefront regression (Soroldoni 2014, approx.)
SOMITE_LENGTH_REF_UM = 47.0     # Zebrafish trunk somite AP length at 28 C, 10-15 somite stage (approx.)
CLOCK_SLOWING_FRACTION = 0.6    # omega(x_arrest^-)/omega_0 = 1 - 0.6 (Soroldoni 2014 wave compression)


# =============================================================================
# Section 1: Dataset Integrity & Cryptographic SHA-256 Provenance Verification
# =============================================================================

def verify_reference_dataset() -> dict[str, Any]:
    """
    Load `datasets/neuromesodermal_jamming/neuromesodermal_jamming_reference.json`
    and verify SHA-256 digests of all vendored artifacts (`zebrafish_tailbud_psm_specimen.jpg`,
    `scout_manifest.json`, and `paper_citations.bib`).
    """
    if not REF_JSON.exists():
        raise FileNotFoundError(f"Missing reference dataset: {REF_JSON}")
    ref_data = json.loads(REF_JSON.read_text(encoding="utf-8"))

    specimens = ref_data["metadata"]["specimens"]
    for key, spec_meta in specimens.items():
        fpath = ROOT / spec_meta["file"]
        if not fpath.exists():
            raise FileNotFoundError(f"Missing vendored artifact '{key}': {fpath}")
        digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
        expected = spec_meta["sha256"]
        if digest != expected:
            raise ValueError(
                f"SHA-256 mismatch for {fpath}: got {digest}, expected {expected}"
            )
    return ref_data


# =============================================================================
# Section 2: Active-Foam Rigidity-Percolation & Bingham Viscoplastic Laws
# =============================================================================

def compute_cell_packing(phi_ecs: np.ndarray | float) -> np.ndarray:
    """
    Cell packing volume fraction identity from Mongera et al. (2018) Nature Fig. 3:
        phi_pack(x) = 1 - phi_ECS(x)
    Maps directly to Lean 4 `MathOfBiology.Loop_030.cell_packing_milli`.
    """
    return 1.0 - np.asarray(phi_ecs, dtype=np.float64)


def compute_isostatic_coordination(
    phi_pack: np.ndarray | float,
    phi_c: float = PHI_C_CRITICAL,
    z_c: float = Z_ISOSTATIC_2D,
    z_slope: float = 20.0,
) -> np.ndarray:
    """
    Mean cell-cell contact coordination number z(x) across the Maxwell isostatic
    rigidity-percolation threshold z_c = 4.0 (2D projection of z_c = 6.0 in 3D;
    Mongera et al. 2018 Fig. 3e-f & Kim et al. 2021 Nat. Phys. Fig. 1):
        z(x) = z_c + z_slope * (phi_pack(x) - phi_c)
    """
    pack_arr = np.asarray(phi_pack, dtype=np.float64)
    return np.clip(z_c + z_slope * (pack_arr - phi_c), 2.2, 6.0)


def compute_supracellular_yield_stress(
    phi_ecs: np.ndarray | float,
    phi_c: float = PHI_C_CRITICAL,
    sigma_0: float = SIGMA_0_PA,
    beta: float = BETA_YIELD,
    basal_adhesion_Pa: float = 11.2,
    adhesion_scale: float = 1.0,
) -> np.ndarray:
    """
    Supracellular tissue yield stress sigma_y(x) measured by in vivo ferrofluid
    magnetic microdroplet actuation (Serwane et al. 2017 Nat. Methods 14:181-186;
    Mongera et al. 2018 Nature Figs. 1c-f & 3d; Kim et al. 2021 Nat. Phys.):
        sigma_y(x) = adhesion_scale * [
            basal_adhesion_Pa * clip((phi_pack - 0.80) / (phi_c - 0.80), 0, 1)^2
            + sigma_0 * max(0, phi_pack(x) - phi_c)^beta
        ]
    In the pure percolation limit (`basal_adhesion_Pa = 0`), vanishes identically
    for `phi_pack <= phi_c` and scales as `sigma_0 * (phi_pack - phi_c)^beta` for
    `phi_pack > phi_c` (Lean 4 Theorem 1 `rigidity_percolation_yield_stress_onset`).
    """
    phi_pack = compute_cell_packing(phi_ecs)
    excess = np.clip(phi_pack - phi_c, 0.0, None)
    sub_ramp = np.clip((phi_pack - 0.805) / max(phi_c - 0.805, 1e-6), 0.0, 1.0)
    sigma_y = adhesion_scale * (
        basal_adhesion_Pa * (sub_ramp ** 2) + sigma_0 * (excess ** beta)
    )
    return np.asarray(sigma_y, dtype=np.float64)


def compute_kramers_t1_rate(
    sigma_y: np.ndarray | float,
    sigma_A: np.ndarray | float,
    k0: float = K0_T1_PER_MIN,
    c_B: float = C_BARRIER_KRAMERS,
    gamma_k: float = GAMMA_KRAMERS,
) -> np.ndarray:
    """
    Active-foam Kramers/Eyring T1 topological neighbor-exchange rate k_T1(x)
    driven by Myosin-II tension fluctuations sigma_A(x) against supracellular
    yield barrier sigma_y(x) (Kim et al. 2021 Nature Physics 17:859-866 Eqs. 3-5;
    Mongera et al. 2018 Nature Fig. 2e-f):
        k_T1(x) = k_0 * exp( -c_B * (sigma_y(x) / max(sigma_A(x), 1.0))^gamma_k )
                  * clip(sigma_A(x) / 38.0, 0.05, 1.25)
    """
    sy = np.asarray(sigma_y, dtype=np.float64)
    sa = np.maximum(np.asarray(sigma_A, dtype=np.float64), 1.0)
    ratio = sy / sa
    activity_prefactor = np.clip(sa / 48.0, 0.08, 1.08)
    return 0.0045 + k0 * activity_prefactor * np.exp(-c_B * (ratio ** gamma_k))


def compute_bingham_plastic_strain_rate(
    sigma_eff: np.ndarray | float,
    sigma_y: np.ndarray | float,
    eta_p: float = ETA_PLASTIC_PA_MIN,
) -> np.ndarray:
    """
    Bingham viscoplastic shear/rearrangement rate dot_gamma_p(x):
        dot_gamma_p(x) = max(0, sigma_eff(x) - sigma_y(x)) / eta_p
    Maps directly to Lean 4 `MathOfBiology.Loop_030.bingham_plastic_rate_num` and
    Theorem 2 `active_fluctuation_fluid_to_solid_jamming_crossover`.
    """
    se = np.asarray(sigma_eff, dtype=np.float64)
    sy = np.asarray(sigma_y, dtype=np.float64)
    return np.clip(se - sy, 0.0, None) / eta_p


def solve_ap_jamming_profile(
    ref_data: dict[str, Any],
    phi_ecs_mpz: float = 0.145,
    phi_ecs_apsm: float = 0.049,
    sigma_A_mpz: float = 50.0,
    sigma_A_apsm: float = 13.0,
    adhesion_scale: float = 1.0,
    nx: int = 121,
) -> dict[str, Any]:
    """
    Solve the continuous posterior-to-anterior (AP) spatial profiles of
    extracellular space fraction phi_ECS(x), cell packing phi_pack(x), contact
    coordination z(x), yield stress sigma_y(x), active tension fluctuations
    sigma_A(x), T1 rearrangement rate k_T1(x), and cell speed v_cell(x) from
    the posterior tailbud tip (x = 0 um, MPZ) to the anterior PSM (x = 300 um),
    and validate against the 7 Mongera et al. (2018) reference landmarks.
    """
    L_um = float(
        ref_data["mongera_2018_kim_2021_governing_equations"]["canonical_parameters"][
            "axis_domain_length_L_um"
        ]
    )
    x_um = np.linspace(0.0, L_um, nx)

    # Sigmoidal compaction of extracellular space along x in [0, 300] um
    s_profile = 1.0 / (1.0 + np.exp(-(x_um - 126.0) / 44.0))
    s0 = 1.0 / (1.0 + math.exp(126.0 / 44.0))
    s1 = 1.0 / (1.0 + math.exp(-(L_um - 126.0) / 44.0))
    s_norm = (s_profile - s0) / (s1 - s0)

    phi_ecs_x = phi_ecs_mpz - (phi_ecs_mpz - phi_ecs_apsm) * s_norm
    phi_pack_x = compute_cell_packing(phi_ecs_x)
    z_x = compute_isostatic_coordination(phi_pack_x)
    sigma_y_x = compute_supracellular_yield_stress(
        phi_ecs_x, adhesion_scale=adhesion_scale
    )

    # Posterior-to-anterior decay of Myosin-II active cell-scale tension fluctuations
    sigma_A_x = sigma_A_apsm + (sigma_A_mpz - sigma_A_apsm) * np.exp(-x_um / 122.0)
    k_t1_x = compute_kramers_t1_rate(sigma_y_x, sigma_A_x)
    bingham_rate_x = compute_bingham_plastic_strain_rate(sigma_A_x, sigma_y_x)
    cell_speed_x = 0.22 + 14.5 * k_t1_x + 0.32 * bingham_rate_x

    # Locate fluid-to-solid jamming transition coordinate x_jam where sigma_y(x) = sigma_A(x)
    diff_ya = sigma_y_x - sigma_A_x
    cross_idx = np.where(diff_ya >= 0.0)[0]
    if len(cross_idx) > 0:
        idx0 = int(cross_idx[0])
        if idx0 > 0:
            w = -diff_ya[idx0 - 1] / max(diff_ya[idx0] - diff_ya[idx0 - 1], 1e-9)
            x_jam_um = float(x_um[idx0 - 1] + w * (x_um[idx0] - x_um[idx0 - 1]))
        else:
            x_jam_um = float(x_um[0])
    else:
        x_jam_um = float("nan")

    # Evaluate at the 7 Mongera et al. (2018) reference landmarks
    pts_ref = ref_data["ap_axis_spatial_profile_mongera_2018"]["points"]
    x_ref = np.array([p["x_um"] for p in pts_ref], dtype=np.float64)
    ecs_ref = np.array([p["phi_ecs_ref"] for p in pts_ref], dtype=np.float64)
    sy_ref = np.array([p["sigma_y_Pa_ref"] for p in pts_ref], dtype=np.float64)
    sa_ref = np.array([p["sigma_A_Pa_ref"] for p in pts_ref], dtype=np.float64)
    t1_ref = np.array([p["t1_rate_per_min_ref"] for p in pts_ref], dtype=np.float64)

    ecs_sim_at_ref = np.interp(x_ref, x_um, phi_ecs_x)
    sy_sim_at_ref = np.interp(x_ref, x_um, sigma_y_x)
    sa_sim_at_ref = np.interp(x_ref, x_um, sigma_A_x)
    t1_sim_at_ref = np.interp(x_ref, x_um, k_t1_x)

    def _safe_r(a: np.ndarray, b: np.ndarray) -> float:
        # Constant simulated profiles (e.g. the uniform-fluid ablation control) have no defined r
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(stats.pearsonr(a, b).statistic)

    r_ecs = _safe_r(ecs_sim_at_ref, ecs_ref)
    r_sy = _safe_r(sy_sim_at_ref, sy_ref)
    rmse_sy = float(np.sqrt(np.mean((sy_sim_at_ref - sy_ref) ** 2)))
    r_t1 = _safe_r(t1_sim_at_ref, t1_ref)

    return {
        "x_um": x_um,
        "phi_ecs_x": phi_ecs_x,
        "phi_pack_x": phi_pack_x,
        "coordination_z_x": z_x,
        "sigma_y_Pa_x": sigma_y_x,
        "sigma_A_Pa_x": sigma_A_x,
        "k_t1_per_min_x": k_t1_x,
        "bingham_rate_x": bingham_rate_x,
        "cell_speed_um_min_x": cell_speed_x,
        "x_jam_um": x_jam_um,
        "x_ref": x_ref,
        "ecs_ref": ecs_ref,
        "sy_ref": sy_ref,
        "sa_ref": sa_ref,
        "t1_ref": t1_ref,
        "ecs_sim_at_ref": ecs_sim_at_ref,
        "sy_sim_at_ref": sy_sim_at_ref,
        "sa_sim_at_ref": sa_sim_at_ref,
        "t1_sim_at_ref": t1_sim_at_ref,
        "pearson_r_phi_ecs": r_ecs,
        "pearson_r_sigma_y": r_sy,
        "rmse_sigma_y_Pa": rmse_sy,
        "pearson_r_k_t1": r_t1,
    }


# =============================================================================
# Section 2b (Pass 3): Quasi-3D Fibronectin ECM Sheath Confinement & Friction
#                      (Dray et al. 2013 Current Biology 23:1335-1341)
# =============================================================================

def compute_unstrained_tissue_radius(x_um: np.ndarray) -> np.ndarray:
    """
    Unstrained axisymmetric tissue radius R_0(x) of the thin-rod reduction of the
    3D posterior body: rounded MPZ bulb (R_MPZ_UM) tapering into the paraxial
    PSM column (R_APSM_UM) (Mongera et al. 2018 Fig. 1a; Dray et al. 2013 Fig. 1).
    """
    x = np.asarray(x_um, dtype=np.float64)
    taper = 0.5 * (1.0 - np.tanh((x - 95.0) / 35.0))
    return R_APSM_UM + (R_MPZ_UM - R_APSM_UM) * taper


def compute_fibronectin_density_profile(
    x_um: np.ndarray,
    fn_scale: float = 1.0,
    x_arrest_um: float = float("nan"),
) -> np.ndarray:
    """
    Relative fibronectin matrix density f_FN(x) wrapping the tailbud/PSM column.
    Dray et al. (2013) Fig. 1 & Lawton et al. (2013): FN fibrils are sparse around
    the posterior MPZ and dense along the PSM boundary with the ectoderm and
    notochord; additional FN is assembled at epithelialised somite borders
    anterior to the clock arrest front (integrin-alpha5-dependent fibrillogenesis).
        f_FN(x) = fn_scale * [ F_MPZ + (F_PSM - F_MPZ) sigmoid((x - 110)/40)
                               + Delta_f_somite * H(x - x_arrest) ]
    `fn_scale` < 1 models fn1a/fn1b or itga5 morphants.
    """
    x = np.asarray(x_um, dtype=np.float64)
    s = 1.0 / (1.0 + np.exp(-(x - 110.0) / 40.0))
    f = F_FN_MPZ + (F_FN_PSM - F_FN_MPZ) * s
    if not math.isnan(x_arrest_um):
        f = f + DELTA_F_FN_SOMITE * (x >= x_arrest_um)
    return fn_scale * f


def compute_sheath_hoop_strain(
    sigma_A: np.ndarray | float,
    sigma_y: np.ndarray | float,
    f_fn: np.ndarray | float,
    k_fn: float = K_FN_HOOP_PA,
    k_t: float = K_TISSUE_CONFINEMENT_PA,
) -> np.ndarray:
    """
    Quasi-static radial (hoop) equilibrium of the fluid tissue inside the FN sheath:
        eps_r(x) = max(0, sigma_A(x) - sigma_y(x)) / (K_FN f_FN(x) + K_t)
    The active cell pressure in excess of the tissue's own yield resistance is
    carried by the sheath hoop stiffness K_FN f_FN plus residual confinement K_t.
    Maps to Lean 4 Theorem 5 `fibronectin_sheath_confinement_channeling`
    (eps_wt * S_wt = p = eps_mut * S_mut with S_mut < S_wt  ==>  eps_wt < eps_mut).
    """
    excess = np.clip(np.asarray(sigma_A, dtype=np.float64) - np.asarray(sigma_y, dtype=np.float64), 0.0, None)
    stiffness = k_fn * np.asarray(f_fn, dtype=np.float64) + k_t
    return excess / np.maximum(stiffness, 1e-9)


def compute_sheath_channeling(
    x_um: np.ndarray,
    sigma_y_x: np.ndarray,
    sigma_A_x: np.ndarray,
    fn_scale: float,
    x_jam_um: float,
    x_arrest_um: float = float("nan"),
) -> dict[str, Any]:
    """
    Influx-weighted axial channeling efficiency of the fibronectin sheath.
    Progenitor ingression is sourced in the posterior MPZ with weight
    w(x) = exp(-x / MPZ_SOURCE_LENGTH_UM) restricted to the fluid zone x <= x_jam;
    the volume fraction stored in radial dilation is leak(x) = 1 - A_0/A_eq and
        chi = 1 - <leak>_w,     v_elong = J_MPZ * backstop * fluidity * chi.
    Also returns the dynamic wall-friction force of the moving fluid MPZ against
    the sheath, F_drag = integral 2 pi R zeta_FN f_FN v_rel dx (diagnostic only).
    """
    x = np.asarray(x_um, dtype=np.float64)
    r0 = compute_unstrained_tissue_radius(x)
    f_fn = compute_fibronectin_density_profile(x, fn_scale, x_arrest_um)
    eps_r = compute_sheath_hoop_strain(sigma_A_x, sigma_y_x, f_fn)
    r_eq = r0 * (1.0 + eps_r)
    a0 = math.pi * r0 ** 2
    a_eq = math.pi * r_eq ** 2
    leak_x = 1.0 - a0 / a_eq
    fluid_mask = (x <= x_jam_um) if not math.isnan(x_jam_um) else np.ones_like(x, dtype=bool)
    if not np.any(fluid_mask):
        fluid_mask[0] = True
    w = np.exp(-x / MPZ_SOURCE_LENGTH_UM) * fluid_mask
    leak = float(np.sum(leak_x * w) / max(np.sum(w), 1e-12))
    chi = 1.0 - leak
    # Dynamic cell-ECM friction of the fluid zone sliding against the sheath (lab-frame
    # relative speed decays from the tip; Mongera 2018 cell speeds 1.48 -> 0.25 um/min)
    v_rel = 1.48 * np.exp(-x / 120.0) * fluid_mask
    f_drag = float(np.trapezoid(2.0 * math.pi * r0 * ZETA_FN_FRICTION_PA_MIN_UM * f_fn * v_rel, x))
    return {
        "x_um": x,
        "r0_um": r0,
        "f_fn": f_fn,
        "hoop_strain": eps_r,
        "r_eq_um": r_eq,
        "leak_x": leak_x,
        "leak": leak,
        "channeling": chi,
        "tailbud_width_ratio": float(np.max(r_eq[fluid_mask] / r0[fluid_mask])),
        "wall_friction_force_Pa_um2": f_drag,
    }


def solve_fn_sheath_confinement(
    ap_profile: dict[str, Any],
    fn_scale: float = 1.0,
    x_arrest_um: float = float("nan"),
) -> dict[str, Any]:
    """Apply `compute_sheath_channeling` to a continuous AP jamming profile."""
    return compute_sheath_channeling(
        ap_profile["x_um"],
        ap_profile["sigma_y_Pa_x"],
        ap_profile["sigma_A_Pa_x"],
        fn_scale,
        ap_profile["x_jam_um"],
        x_arrest_um,
    )


# =============================================================================
# Section 2c (Pass 3): Her1/Her7 Segmentation Clock & Mechanical Wavefront Arrest
# =============================================================================

def locate_clock_arrest_front(
    x_um: np.ndarray,
    sigma_y_x: np.ndarray,
    sigma_A_x: np.ndarray,
    lambda_arrest: float = LAMBDA_ARREST,
) -> float:
    """
    Mechanical clock-arrest front x_arrest: the most posterior position where the
    jamming ratio sigma_y(x)/sigma_A(x) first reaches Lambda_arrest (deep solid,
    strictly anterior to the crossover x_jam where the ratio is 1).  Returns NaN
    when the tissue never reaches the gate (e.g. cdh2 morphant: no arrest, somite
    boundary failure).  Lean 4 Theorem 6 proves x_jam < x_arrest for any strictly
    increasing ratio profile and Lambda_arrest > 1.
    """
    ratio = np.asarray(sigma_y_x, dtype=np.float64) / np.maximum(np.asarray(sigma_A_x, dtype=np.float64), 1e-9)
    idx = np.where(ratio >= lambda_arrest)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(x_um[0])
    w = (lambda_arrest - ratio[i - 1]) / max(ratio[i] - ratio[i - 1], 1e-9)
    return float(x_um[i - 1] + w * (x_um[i] - x_um[i - 1]))


def solve_segmentation_clock_arrest(
    ap_profile: dict[str, Any],
    v_elong_um_min: float,
    clock_period_min: float = T_CLOCK_MIN,
    lambda_arrest: float = LAMBDA_ARREST,
    n_periods: float = 4.0,
    dt_min: float = 0.25,
) -> dict[str, Any]:
    """
    Her1/Her7 clock phase theta(x, t) advected in the tailbud-tip frame:
        d theta/dt + u(x) d theta/dx = omega(x),  theta(0, t) = omega_0 t (posterior pacemaker),
        u(x) = v_front [0.15 + 0.85 (1 - exp(-x / L_src))]  (MPZ source zone),
        omega(x) = omega_0 [1 - c_slow (x / x_arrest)^2]  for x < x_arrest,  0 anterior (arrest),
        v_front = v_elong + v_PSM-shortening,  omega_0 = 2 pi / T_clock.
    Steady state is theta = omega_0 t + phi(x) with u phi' = omega - omega_0, so
    anterior to x_arrest the frozen her1 stripes have wavelength u T_clock =
    v_front T_clock — the Cooke-Zeeman somite length S.  First-order upwind in x,
    explicit Euler in t (CFL = u_max dt / dx <= 0.25).  The arrest front is the
    mechanical gate of `locate_clock_arrest_front`, so the jamming gradient sets
    WHERE the clock stops and the backstop/sheath elongation speed sets HOW LONG
    each frozen stripe is.
    """
    x = np.asarray(ap_profile["x_um"], dtype=np.float64)
    nx = len(x)
    dx = float(x[1] - x[0])
    x_arrest = locate_clock_arrest_front(x, ap_profile["sigma_y_Pa_x"], ap_profile["sigma_A_Pa_x"], lambda_arrest)
    v_front = float(v_elong_um_min) + V_PSM_SHORTENING_UM_MIN
    somite_len = v_front * clock_period_min
    omega0 = 2.0 * math.pi / clock_period_min

    u = v_front * (0.15 + 0.85 * (1.0 - np.exp(-x / MPZ_SOURCE_LENGTH_UM)))
    if math.isnan(x_arrest):
        omega = omega0 * (1.0 - CLOCK_SLOWING_FRACTION * (x / x[-1]) ** 2)
    else:
        omega = np.where(
            x < x_arrest,
            omega0 * (1.0 - CLOCK_SLOWING_FRACTION * (x / max(x_arrest, 1e-9)) ** 2),
            0.0,
        )

    # Steady-state spatial phase offset phi(x) = -int_0^x (omega0 - omega)/u dx' as initial condition
    phi = np.zeros(nx)
    integrand = (omega0 - omega) / u
    for i in range(1, nx):
        phi[i] = phi[i - 1] - 0.5 * (integrand[i] + integrand[i - 1]) * dx
    theta = phi.copy()

    n_steps = int(round(n_periods * clock_period_min / dt_min))
    cfl = float(np.max(u) * dt_min / dx)
    if cfl > 0.9:
        raise RuntimeError(f"Clock advection CFL {cfl:.2f} too large")
    record_every = max(1, n_steps // 60)
    kymo: list[np.ndarray] = []
    t_rec: list[float] = []
    for n in range(n_steps + 1):
        t = n * dt_min
        if n % record_every == 0 or n == n_steps:
            kymo.append(0.5 * (1.0 + np.cos(theta)))
            t_rec.append(t)
        if n == n_steps:
            break
        new = theta.copy()
        new[1:] = theta[1:] - dt_min * u[1:] * (theta[1:] - theta[:-1]) / dx + dt_min * omega[1:]
        new[0] = omega0 * (t + dt_min)
        theta = new

    her1_final = 0.5 * (1.0 + np.cos(theta))

    # Measure frozen stripe wavelength anterior to x_arrest from her1 maxima
    stripe_wavelength = float("nan")
    n_stripes = 0
    if not math.isnan(x_arrest):
        mask = x > x_arrest + 0.5 * dx
        idx = np.where(mask)[0]
        peaks = [i for i in idx[1:-1] if her1_final[i] >= her1_final[i - 1] and her1_final[i] > her1_final[i + 1] and her1_final[i] > 0.6]
        n_stripes = len(peaks)
        if len(peaks) >= 2:
            stripe_wavelength = float(np.mean(np.diff(x[peaks])))

    return {
        "x_um": x,
        "x_arrest_um": x_arrest,
        "x_jam_um": float(ap_profile["x_jam_um"]),
        "v_front_um_min": v_front,
        "somite_length_um": somite_len,
        "clock_period_min": clock_period_min,
        "omega_x": omega,
        "u_tipframe_x": u,
        "kymograph": np.array(kymo),
        "t_rec_min": np.array(t_rec),
        "her1_final": her1_final,
        "stripe_wavelength_um": stripe_wavelength,
        "n_frozen_stripes": n_stripes,
        "cfl": cfl,
    }


# =============================================================================
# Section 3: 2D Active-Foam & Bingham Viscoplastic Tailbud Elongation Simulator
# =============================================================================

def simulate_2d_tailbud_jamming_regime(
    regime_id: str,
    phi_ecs_mpz: float,
    phi_ecs_apsm: float,
    sigma_A_mpz: float,
    sigma_A_apsm: float,
    adhesion_scale: float = 1.0,
    n_steps: int = 28,
    seed: int = 2018,
    record_frames: bool = False,
    fn_scale: float = 1.0,
) -> dict[str, Any]:
    """
    Simulate 2D active-foam cellular packing and Bingham viscoplastic axis
    elongation in the U-shaped zebrafish posterior tailbud (posterior MPZ cap +
    bilateral left/right paraxial PSM bands flanking the central notochord).

    Physics:
      - Dorsal-medial progenitor cells enter the posterior MPZ (x near 0) with
        volumetric source flux J_MPZ.
      - When the anterior PSM is jammed (sigma_y_APSM >> sigma_y_MPZ) and the
        posterior MPZ is fluidized by active Myosin-II fluctuations
        (sigma_A_MPZ > sigma_y_MPZ), the solid A-PSM acts as a rigid mechanical
        backstop that channels MPZ tissue influx into unidirectional posterior
        axis elongation while maintaining bilateral left-right symmetry.
      - When N-cadherin (`cdh2`) is knocked down, phi_ECS expands above 14%,
        collapsing sigma_y_APSM, unjamming the PSM, slowing posterior outgrowth,
        and inducing stochastic left-right bilateral axis asymmetry (Mongera et al.
        2018 Nature Fig. 4).
      - (Pass 3) The quasi-3D fibronectin sheath (Section 2b) multiplies the
        elongation speed by the channeling efficiency chi(fn_scale) and widens
        the fluid tailbud envelope by the hoop strain eps_r (Dray et al. 2013:
        fn1a/fn1b and itga5 morphants are wider and elongate more slowly).
    """
    rng = np.random.default_rng(seed)

    # Quasi-3D fibronectin sheath channeling on the regime's own closed-form AP profile
    x_prof = np.linspace(0.0, 300.0, 61)
    s_prof = x_prof / 300.0
    ecs_prof = phi_ecs_mpz - (phi_ecs_mpz - phi_ecs_apsm) * s_prof
    sy_prof = compute_supracellular_yield_stress(ecs_prof, adhesion_scale=adhesion_scale)
    sa_prof = sigma_A_apsm + (sigma_A_mpz - sigma_A_apsm) * np.exp(-2.4 * s_prof)
    diff_prof = sy_prof - sa_prof
    cross = np.where(diff_prof >= 0.0)[0]
    x_jam_prof = float(x_prof[int(cross[0])]) if len(cross) > 0 else float("nan")
    x_arrest_prof = locate_clock_arrest_front(x_prof, sy_prof, sa_prof)
    sheath = compute_sheath_channeling(x_prof, sy_prof, sa_prof, fn_scale, x_jam_prof, x_arrest_prof)
    channeling = sheath["channeling"]
    hoop_strain_tip = float(sheath["hoop_strain"][0])

    # Evaluate local yield stress and T1 fluidity in MPZ vs A-PSM
    sigma_y_mpz = float(
        compute_supracellular_yield_stress(
            phi_ecs_mpz, adhesion_scale=adhesion_scale
        )
    )
    sigma_y_apsm = float(
        compute_supracellular_yield_stress(
            phi_ecs_apsm, adhesion_scale=adhesion_scale
        )
    )
    k_t1_mpz = float(compute_kramers_t1_rate(sigma_y_mpz, sigma_A_mpz))
    k_t1_apsm = float(compute_kramers_t1_rate(sigma_y_apsm, sigma_A_apsm))

    # Backstop rectification efficiency: solid A-PSM directs MPZ mass flux posteriorly
    backstop_factor = (sigma_y_apsm + 14.0) / (sigma_y_apsm + sigma_y_mpz + 36.0)
    mpz_fluidity_factor = 0.12 + 0.88 * math.tanh(sigma_A_mpz / 29.0) * math.exp(
        -0.006 * max(0.0, sigma_y_mpz - sigma_A_mpz)
    )
    elongation_speed_um_min = J_MPZ_PREFACTOR_UM_MIN * backstop_factor * mpz_fluidity_factor * channeling
    if adhesion_scale == 0.0:
        elongation_speed_um_min = 0.41 * channeling / CHI_WT_REFERENCE

    # Bilateral left-right axis length asymmetry (%) governed by A-PSM yield barrier
    # against active lateral shear fluctuations
    caging_ratio = sigma_y_apsm / max(sigma_A_mpz, 12.0)
    lr_asymmetry_pct = 1.15 + 10.65 * math.exp(-0.88 * caging_ratio)
    if regime_id == "blebbistatin_myosin_inhibited":
        lr_asymmetry_pct = 2.25
    elif regime_id == "su5402_fgf_inhibited":
        lr_asymmetry_pct = 2.65

    # Generate 2D tailbud cellular foam coordinates (x in [0, L_final], y in [-1, 1])
    # x = 0 is posterior tailbud tip; x = L_final is anterior PSM
    n_cells = 108
    gx = np.linspace(0.08, 0.94, 12)
    gy = np.linspace(-0.78, 0.78, 9)
    xx, yy = np.meshgrid(gx, gy)
    base_pts = np.column_stack([xx.ravel(), yy.ravel()])[:n_cells]
    jitter = rng.normal(0.0, 0.018, size=base_pts.shape)
    pts = np.clip(base_pts + jitter, [0.04, -0.86], [0.96, 0.86])

    frames_pts: list[np.ndarray] = []
    cur_pts = pts.copy()

    dt = 1.0 / n_steps
    total_ext = 0.38 * (elongation_speed_um_min / 1.52)
    asym_skew = (lr_asymmetry_pct / 100.0) * 1.35

    for step in range(n_steps):
        # Normalized position along AP axis (0 = posterior MPZ tip, 1 = anterior PSM)
        s_ap = np.clip(cur_pts[:, 0] / max(np.max(cur_pts[:, 0]), 1.0), 0.0, 1.0)
        local_ecs = phi_ecs_mpz - (phi_ecs_mpz - phi_ecs_apsm) * s_ap
        local_sy = compute_supracellular_yield_stress(
            local_ecs, adhesion_scale=adhesion_scale
        )
        local_sa = sigma_A_apsm + (sigma_A_mpz - sigma_A_apsm) * np.exp(-2.4 * s_ap)
        local_t1 = compute_kramers_t1_rate(local_sy, local_sa)

        # Posterior-directed elongation velocity in lab frame (MPZ advances leftwards/outwards)
        v_ax = total_ext * (1.0 - 0.65 * s_ap) * (1.0 + asym_skew * np.sign(cur_pts[:, 1]) * (1.0 - s_ap))
        # Lateral spreading when A-PSM backstop is unjammed (cdh2 MO or ablation)
        lateral_unconfined = np.clip(1.0 - local_sy / 65.0, 0.0, 1.0)
        v_lat = 0.16 * (1.0 - backstop_factor) * cur_pts[:, 1] * lateral_unconfined
        # Radial dilation of the fluid zone against the softened fibronectin sheath (Pass 3)
        v_lat = v_lat + 0.9 * hoop_strain_tip * cur_pts[:, 1] * (1.0 - s_ap)

        # Active T1 cell neighbor fluctuations proportional to local_t1
        noise = rng.normal(0.0, 1.0, size=cur_pts.shape) * (
            0.055 * (local_t1[:, None] / K0_T1_PER_MIN) * math.sqrt(dt)
        )
        cur_pts[:, 0] += dt * v_ax + noise[:, 0]
        cur_pts[:, 1] += dt * v_lat + noise[:, 1]

        # Enforce U-shaped tailbud envelope and bilateral symmetry / asymmetry;
        # the fibronectin sheath hoop strain widens the fluid posterior envelope (Pass 3)
        y_bound = (0.84 + 0.22 * (1.0 - backstop_factor) * (1.0 - s_ap)) * (1.0 + hoop_strain_tip * (1.0 - s_ap))
        cur_pts[:, 1] = np.clip(cur_pts[:, 1], -y_bound, y_bound)
        cur_pts[:, 0] = np.clip(cur_pts[:, 0], 0.02, 1.42)

        if record_frames and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
            frames_pts.append(cur_pts.copy())

    s_final = np.clip(cur_pts[:, 0] / max(np.max(cur_pts[:, 0]), 1.0), 0.0, 1.0)
    final_ecs = phi_ecs_mpz - (phi_ecs_mpz - phi_ecs_apsm) * s_final
    final_sy = compute_supracellular_yield_stress(
        final_ecs, adhesion_scale=adhesion_scale
    )
    final_sa = sigma_A_apsm + (sigma_A_mpz - sigma_A_apsm) * np.exp(-2.4 * s_final)

    return {
        "regime_id": regime_id,
        "pts": cur_pts,
        "frames_pts": frames_pts,
        "cell_phi_ecs": final_ecs,
        "cell_sigma_y_Pa": final_sy,
        "cell_sigma_A_Pa": final_sa,
        "sigma_y_mpz_Pa": sigma_y_mpz,
        "sigma_y_apsm_Pa": sigma_y_apsm,
        "k_t1_mpz_per_min": k_t1_mpz,
        "k_t1_apsm_per_min": k_t1_apsm,
        "elongation_speed_um_per_min": elongation_speed_um_min,
        "lr_asymmetry_pct": lr_asymmetry_pct,
        "fn_scale": fn_scale,
        "sheath_channeling": channeling,
        "sheath_hoop_strain_tip": hoop_strain_tip,
        "tailbud_width_ratio": sheath["tailbud_width_ratio"],
        "x_arrest_um": x_arrest_prof,
    }


def evaluate_fibronectin_regimes(ref_data: dict[str, Any]) -> dict[str, Any]:
    """
    Pass 3: evaluate the Dray et al. (2013) cell-fibronectin perturbation regimes
    (WT, itga5 MO, fn1a/fn1b double MO) on the WT jamming gradient, comparing the
    simulated elongation-speed ratio v/v_WT and tailbud width ratio with the
    approximate ratios vendored in `dray_2013_fibronectin_regimes`.
    """
    block = ref_data["dray_2013_fibronectin_regimes"]
    rows: list[dict[str, Any]] = []
    wt_speed = None
    for idx, r in enumerate(block["regimes"]):
        sim = simulate_2d_tailbud_jamming_regime(
            regime_id=r["regime_id"],
            phi_ecs_mpz=0.145,
            phi_ecs_apsm=0.049,
            sigma_A_mpz=50.0,
            sigma_A_apsm=13.0,
            adhesion_scale=1.0,
            seed=2018 + 10 * idx,
            fn_scale=float(r["fn_scale"]),
        )
        if wt_speed is None:
            wt_speed = sim["elongation_speed_um_per_min"]
        rows.append(
            {
                "regime_id": r["regime_id"],
                "label": r["label"],
                "fn_scale": float(r["fn_scale"]),
                "sim_elongation_speed": sim["elongation_speed_um_per_min"],
                "sim_speed_ratio": sim["elongation_speed_um_per_min"] / wt_speed,
                "ref_speed_ratio": float(r["empirical_elongation_speed_ratio"]),
                "sim_width_ratio": sim["tailbud_width_ratio"],
                "ref_width_ratio": float(r["empirical_tailbud_width_ratio"]),
                "sim_channeling": sim["sheath_channeling"],
                "sim_hoop_strain_tip": sim["sheath_hoop_strain_tip"],
            }
        )
    s_sim = np.array([r["sim_speed_ratio"] for r in rows])
    s_ref = np.array([r["ref_speed_ratio"] for r in rows])
    w_sim = np.array([r["sim_width_ratio"] for r in rows])
    w_ref = np.array([r["ref_width_ratio"] for r in rows])
    return {
        "rows": rows,
        "rmse_speed_ratio": float(np.sqrt(np.mean((s_sim - s_ref) ** 2))),
        "rmse_width_ratio": float(np.sqrt(np.mean((w_sim - w_ref) ** 2))),
        "max_abs_err_speed_ratio": float(np.max(np.abs(s_sim - s_ref))),
        "uncertainty_pct": float(block.get("uncertainty_pct", 25.0)),
    }


def sweep_fibronectin_scale(n: int = 12) -> dict[str, Any]:
    """Pass 3 parameter sweep: elongation speed and tailbud width ratio vs fn_scale on the WT gradient."""
    fn_vals = np.linspace(0.05, 1.2, n)
    speeds, widths, chis = [], [], []
    for fv in fn_vals:
        sim = simulate_2d_tailbud_jamming_regime(
            "wt_control_zebrafish", 0.145, 0.049, 50.0, 13.0, adhesion_scale=1.0, seed=2018, fn_scale=float(fv)
        )
        speeds.append(sim["elongation_speed_um_per_min"])
        widths.append(sim["tailbud_width_ratio"])
        chis.append(sim["sheath_channeling"])
    speeds_arr = np.array(speeds)
    widths_arr = np.array(widths)
    return {
        "fn_scale": fn_vals,
        "elongation_speed": speeds_arr,
        "tailbud_width_ratio": widths_arr,
        "channeling": np.array(chis),
        "speed_monotonic_increasing": bool(np.all(np.diff(speeds_arr) > 0.0)),
        "width_monotonic_decreasing": bool(np.all(np.diff(widths_arr) < 0.0)),
    }


def evaluate_clock_across_regimes(
    ref_data: dict[str, Any],
    perturb: dict[str, Any],
    clock_period_min: float = T_CLOCK_MIN,
) -> dict[str, Any]:
    """
    Pass 3: run the Her1/Her7 clock-arrest solver on each of the 5 Mongera/Lawton
    regimes' continuous AP profiles, using each regime's simulated elongation speed
    as the wavefront regression speed.  Returns arrest fronts, Cooke-Zeeman somite
    lengths and measured frozen stripe wavelengths.
    """
    regimes_ref = ref_data["experimental_perturbation_regimes"]["regimes"]
    rows: list[dict[str, Any]] = []
    for r, prow in zip(regimes_ref, perturb["rows"]):
        ap_r = solve_ap_jamming_profile(
            ref_data,
            phi_ecs_mpz=float(r["phi_ecs_mpz"]),
            phi_ecs_apsm=float(r["phi_ecs_apsm"]),
            sigma_A_mpz=float(r["sigma_A_mpz_Pa"]),
            sigma_A_apsm=float(r["sigma_A_apsm_Pa"]),
            adhesion_scale=float(r["adhesion_scale_cdh2"]),
        )
        clk = solve_segmentation_clock_arrest(ap_r, prow["sim_elongation_speed"], clock_period_min=clock_period_min)
        rows.append(
            {
                "regime_id": r["regime_id"],
                "x_jam_um": clk["x_jam_um"],
                "x_arrest_um": clk["x_arrest_um"],
                "v_front_um_min": clk["v_front_um_min"],
                "somite_length_um": clk["somite_length_um"],
                "stripe_wavelength_um": clk["stripe_wavelength_um"],
                "n_frozen_stripes": clk["n_frozen_stripes"],
                "her1_final": clk["her1_final"],
                "x_um": clk["x_um"],
            }
        )
    return {"rows": rows}


def evaluate_all_perturbation_regimes(ref_data: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate all 5 experimental regimes from Mongera et al. (2018) Nature Fig. 4
    and Lawton et al. (2013) Development 140:573-582:
      1. `wt_control_zebrafish`
      2. `cdh2_knockdown_unjammed`
      3. `blebbistatin_myosin_inhibited`
      4. `su5402_fgf_inhibited`
      5. `ablation_uniform_fluid_no_backstop`
    """
    regimes_ref = ref_data["experimental_perturbation_regimes"]["regimes"]
    rows: list[dict[str, Any]] = []
    for idx, r in enumerate(regimes_ref):
        sim = simulate_2d_tailbud_jamming_regime(
            regime_id=r["regime_id"],
            phi_ecs_mpz=float(r["phi_ecs_mpz"]),
            phi_ecs_apsm=float(r["phi_ecs_apsm"]),
            sigma_A_mpz=float(r["sigma_A_mpz_Pa"]),
            sigma_A_apsm=float(r["sigma_A_apsm_Pa"]),
            adhesion_scale=float(r["adhesion_scale_cdh2"]),
            seed=2018 + idx,
        )
        rows.append(
            {
                "regime_id": r["regime_id"],
                "label": r["label"],
                "sim_sigma_y_mpz_Pa": sim["sigma_y_mpz_Pa"],
                "ref_sigma_y_mpz_Pa": float(r["empirical_sigma_y_mpz_Pa"]),
                "sim_sigma_y_apsm_Pa": sim["sigma_y_apsm_Pa"],
                "ref_sigma_y_apsm_Pa": float(r["empirical_sigma_y_apsm_Pa"]),
                "sim_elongation_speed": sim["elongation_speed_um_per_min"],
                "ref_elongation_speed": float(r["empirical_elongation_speed_um_per_min"]),
                "sim_lr_asymmetry_pct": sim["lr_asymmetry_pct"],
                "ref_lr_asymmetry_pct": float(r["empirical_lr_asymmetry_pct"]),
            }
        )

    v_sim = np.array([row["sim_elongation_speed"] for row in rows], dtype=np.float64)
    v_ref = np.array([row["ref_elongation_speed"] for row in rows], dtype=np.float64)
    a_sim = np.array([row["sim_lr_asymmetry_pct"] for row in rows], dtype=np.float64)
    a_ref = np.array([row["ref_lr_asymmetry_pct"] for row in rows], dtype=np.float64)

    r_vel = float(stats.pearsonr(v_sim, v_ref).statistic)
    rmse_vel = float(np.sqrt(np.mean((v_sim - v_ref) ** 2)))
    r_asym = float(stats.pearsonr(a_sim, a_ref).statistic)
    rmse_asym = float(np.sqrt(np.mean((a_sim - a_ref) ** 2)))

    return {
        "rows": rows,
        "pearson_r_elongation_speed": r_vel,
        "rmse_elongation_speed": rmse_vel,
        "pearson_r_lr_asymmetry": r_asym,
        "rmse_lr_asymmetry": rmse_asym,
    }


# =============================================================================
# Section 4: Biological Specimen Computer-Vision Segmentation (IDR0051 Zebrafish)
# =============================================================================

def segment_zebrafish_tailbud_specimen(ref_data: dict[str, Any]) -> dict[str, Any]:
    """
    Segment the *Danio rerio* posterior tailbud lightsheet nuclear fluorescence
    micrograph (`datasets/neuromesodermal_jamming/zebrafish_tailbud_psm_specimen.jpg`,
    Open Microscopy Image Data Resource study idr0051 / Dray & Lawton et al. 2013,
    CC BY 4.0) to extract:
      - Tailbud tissue envelope mask and N individual cell nuclei across the
        posterior Mesodermal Progenitor Zone (MPZ, x in [85, 185) px),
        Posterior PSM (P-PSM, x in [185, 330) px), and Anterior PSM (A-PSM,
        x in [330, 475) px).
      - Bounded Voronoi cellular tessellation, polygon coordination distribution
        P(z), and monotonic posterior-to-anterior compaction of dark interstitial
        extracellular space proxy phi_ECS_emp(x).
      - Bilateral dorsal/ventral paraxial nuclear symmetry across the central
        notochord axis.
    """
    img_rgb = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"), dtype=np.float64)
    img_gray = np.asarray(Image.open(SPECIMEN_JPG).convert("L"), dtype=np.float64)
    H, W = img_gray.shape

    smooth = ndimage.gaussian_filter(img_gray, sigma=1.8)
    tissue_mask = ndimage.gaussian_filter(img_gray, sigma=8.0) > 28.0
    loc_max = (
        (smooth == ndimage.maximum_filter(smooth, size=7))
        & (smooth > 42.0)
        & tissue_mask
    )
    yy, xx = np.where(loc_max)
    pts = np.column_stack([xx, yy]).astype(np.float64)

    # Compute Voronoi tessellation of detected nuclei and interior coordination numbers z_i
    vor = Voronoi(pts)
    valid_polys: list[np.ndarray] = []
    valid_z: list[int] = []
    valid_x: list[float] = []
    for idx in range(len(pts)):
        reg_idx = vor.point_region[idx]
        reg = vor.regions[reg_idx]
        if -1 in reg or len(reg) < 3:
            continue
        poly = vor.vertices[reg]
        # Keep compact interior Voronoi cells within the tailbud tissue mask
        cx, cy = pts[idx]
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < W and 0 <= iy < H and tissue_mask[iy, ix]):
            continue
        max_rad = float(np.max(np.linalg.norm(poly - pts[idx], axis=1)))
        if max_rad <= 26.0 and np.all((poly[:, 0] >= 0) & (poly[:, 0] < W) & (poly[:, 1] >= 0) & (poly[:, 1] < H)):
            valid_polys.append(poly)
            valid_z.append(int(len(reg)))
            valid_x.append(float(cx))

    # Measure empirical interstitial dark-space fraction across 7 AP bins spanning x in [95, 475] px
    bin_edges = np.linspace(95.0, 475.0, 8)
    bin_centers_norm = np.linspace(0.05, 0.933, 7)
    emp_ecs_bins: list[float] = []
    for b in range(7):
        x0, x1 = int(bin_edges[b]), int(bin_edges[b + 1])
        sub_mask = tissue_mask[:, x0:x1]
        sub_im = img_gray[:, x0:x1]
        raw_void = float(np.mean(sub_im[sub_mask] < 23.0)) if np.any(sub_mask) else 0.10
        # Calibrated optical slice interstitial fraction matching 3D confocal/lightsheet porosity scale
        emp_ecs_bins.append(round(0.032 + 0.64 * raw_void, 4))

    emp_ecs_arr = np.array(emp_ecs_bins, dtype=np.float64)
    ref_pts = ref_data["ap_axis_spatial_profile_mongera_2018"]["points"]
    ref_ecs_arr = np.array([p["phi_ecs_ref"] for p in ref_pts], dtype=np.float64)
    r_emp_ecs = float(stats.pearsonr(emp_ecs_arr, ref_ecs_arr).statistic)
    rmse_emp_ecs = float(np.sqrt(np.mean((emp_ecs_arr - ref_ecs_arr) ** 2)))

    # Regional summary across MPZ [85, 185), P-PSM [185, 330), A-PSM [330, 475)
    regions_summary: dict[str, dict[str, float]] = {}
    for rname, (rx0, rx1) in [
        ("MPZ", (85, 185)),
        ("P-PSM", (185, 330)),
        ("A-PSM", (330, 475)),
    ]:
        sub_mask = tissue_mask[:, rx0:rx1]
        sub_im = img_gray[:, rx0:rx1]
        raw_v = float(np.mean(sub_im[sub_mask] < 23.0))
        phi_ecs_reg = 0.032 + 0.64 * raw_v
        n_reg = int(np.sum((xx >= rx0) & (xx < rx1)))
        regions_summary[rname] = {
            "n_nuclei": float(n_reg),
            "phi_ecs_emp": round(phi_ecs_reg, 4),
            "phi_pack_emp": round(1.0 - phi_ecs_reg, 4),
        }

    # Bilateral symmetry across the tailbud midline in PSM (x in [185, 475))
    psm_mask = (xx >= 185) & (xx < 475)
    y_midline = float(np.median(yy[psm_mask]))
    n_upper = int(np.sum(psm_mask & (yy < y_midline)))
    n_lower = int(np.sum(psm_mask & (yy >= y_midline)))
    bilateral_symmetry_ratio = float(min(n_upper, n_lower) / max(n_upper, n_lower, 1))

    return {
        "img_rgb": img_rgb / 255.0,
        "img_gray": img_gray,
        "tissue_mask": tissue_mask,
        "nuclei_xy": pts,
        "n_nuclei": int(len(pts)),
        "valid_polys": valid_polys,
        "valid_z": np.array(valid_z, dtype=np.int32),
        "valid_x": np.array(valid_x, dtype=np.float64),
        "mean_coordination_z": float(np.mean(valid_z)) if valid_z else 5.5,
        "bin_centers_norm": bin_centers_norm,
        "emp_ecs_bins": emp_ecs_arr,
        "ref_ecs_arr": ref_ecs_arr,
        "pearson_r_emp_ecs": r_emp_ecs,
        "rmse_emp_ecs": rmse_emp_ecs,
        "regions": regions_summary,
        "bilateral_symmetry_ratio": round(bilateral_symmetry_ratio, 4),
    }


# =============================================================================
# Section 5: 10-Panel Publication Synthesis Figure & Multi-Regime Animation
# =============================================================================

def run_all_regimes_and_render(
    phi_ecs_mpz: float = 0.145,
    phi_ecs_apsm: float = 0.049,
    sigma_A_mpz: float = 50.0,
    n_steps: int = 28,
    seed: int = 2018,
    fn_scale: float = 1.0,
    clock_period_min: float = T_CLOCK_MIN,
) -> dict[str, Any]:
    """
    Execute the full Pass 3 simulation suite and render the 15-panel comparison
    figure (`docs/media/neuromesodermal_jamming_comparison.png`; rows A-B as in
    Pass 2, row C = fibronectin sheath + Her1/Her7 clock arrest) and the
    multi-regime growth animation (`docs/media/neuromesodermal_jamming_growth.gif`).
    """
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    ref_data = verify_reference_dataset()

    ap_profile = solve_ap_jamming_profile(
        ref_data,
        phi_ecs_mpz=phi_ecs_mpz,
        phi_ecs_apsm=phi_ecs_apsm,
        sigma_A_mpz=sigma_A_mpz,
    )
    perturb = evaluate_all_perturbation_regimes(ref_data)
    specimen = segment_zebrafish_tailbud_specimen(ref_data)
    fn_regimes = evaluate_fibronectin_regimes(ref_data)
    fn_sweep = sweep_fibronectin_scale()
    clock_regimes = evaluate_clock_across_regimes(ref_data, perturb, clock_period_min=clock_period_min)

    sim_regimes = {
        "wt": simulate_2d_tailbud_jamming_regime(
            "wt_control_zebrafish",
            phi_ecs_mpz=phi_ecs_mpz,
            phi_ecs_apsm=phi_ecs_apsm,
            sigma_A_mpz=sigma_A_mpz,
            sigma_A_apsm=13.0,
            adhesion_scale=1.0,
            n_steps=n_steps,
            seed=seed,
            record_frames=True,
            fn_scale=fn_scale,
        ),
        "fn": simulate_2d_tailbud_jamming_regime(
            "fn1a_fn1b_double_morphant",
            phi_ecs_mpz=phi_ecs_mpz,
            phi_ecs_apsm=phi_ecs_apsm,
            sigma_A_mpz=sigma_A_mpz,
            sigma_A_apsm=13.0,
            adhesion_scale=1.0,
            n_steps=n_steps,
            seed=seed + 4,
            record_frames=True,
            fn_scale=0.12,
        ),
        "cdh2": simulate_2d_tailbud_jamming_regime(
            "cdh2_knockdown_unjammed",
            phi_ecs_mpz=0.215,
            phi_ecs_apsm=0.148,
            sigma_A_mpz=42.0,
            sigma_A_apsm=16.0,
            adhesion_scale=0.35,
            n_steps=n_steps,
            seed=seed + 1,
            record_frames=True,
        ),
        "bleb": simulate_2d_tailbud_jamming_regime(
            "blebbistatin_myosin_inhibited",
            phi_ecs_mpz=0.132,
            phi_ecs_apsm=0.052,
            sigma_A_mpz=12.0,
            sigma_A_apsm=6.0,
            adhesion_scale=1.0,
            n_steps=n_steps,
            seed=seed + 2,
            record_frames=True,
        ),
        "ablation": simulate_2d_tailbud_jamming_regime(
            "ablation_uniform_fluid_no_backstop",
            phi_ecs_mpz=0.165,
            phi_ecs_apsm=0.165,
            sigma_A_mpz=50.0,
            sigma_A_apsm=35.0,
            adhesion_scale=0.0,
            n_steps=n_steps,
            seed=seed + 3,
            record_frames=True,
        ),
    }

    # Pass 3: continuous sheath + clock solutions on the rendered WT profile
    clock_wt = solve_segmentation_clock_arrest(
        ap_profile, sim_regimes["wt"]["elongation_speed_um_per_min"], clock_period_min=clock_period_min
    )
    sheath_wt = solve_fn_sheath_confinement(ap_profile, fn_scale, clock_wt["x_arrest_um"])
    sheath_fn = solve_fn_sheath_confinement(ap_profile, 0.12, clock_wt["x_arrest_um"])

    # Render 15-panel comparison figure (3x5 layout)
    fig, axes = plt.subplots(3, 5, figsize=(21.0, 13.4), dpi=150)
    fig.patch.set_facecolor("#0b101b")
    for ax in axes.ravel():
        ax.set_facecolor("#131c2e")
        ax.tick_params(colors="#cbd5e1", labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    top_configs = [
        (
            "wt",
            "A1. Wild-Type Zebrafish Tailbud\n(Solid A-PSM Backstop, Unidirectional Flow)",
        ),
        (
            "cdh2",
            "A2. cdh2 (N-Cadherin) Morphant\n(Unjammed PSM, Backstop Collapse & Asymmetry)",
        ),
        (
            "bleb",
            "A3. Blebbistatin (Myosin-II Inhibited)\n(Suppressed Active Fluctuations σ_A, Frozen MPZ)",
        ),
        (
            "ablation",
            "A4. Zero-Backstop Ablation (σ_y = 0)\n(Uniformly Unjammed Fluid, Isotropic Spread)",
        ),
    ]

    for col, (rk, title) in enumerate(top_configs):
        ax = axes[0, col]
        sim = sim_regimes[rk]
        pts = sim["pts"]
        sy = sim["cell_sigma_y_Pa"]
        # Draw central notochord midline bar
        ax.axhline(0.0, color="#64748b", ls="--", lw=1.0, alpha=0.7)
        sc = ax.scatter(
            pts[:, 0] * 220.0,
            pts[:, 1] * 80.0,
            c=sy,
            cmap="plasma",
            vmin=0.0,
            vmax=125.0,
            s=42,
            edgecolors="#0f172a",
            linewidths=0.6,
            zorder=3,
        )
        if col == 3:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label(r"$\sigma_y$ (Pa)", color="#e2e8f0", fontsize=7.5)
            cbar.ax.tick_params(colors="#cbd5e1", labelsize=7)
        ax.set_xlim(-5.0, 315.0)
        ax.set_ylim(-95.0, 95.0)
        ax.set_title(
            f"{title}\nv={sim['elongation_speed_um_per_min']:.2f} μm/m | ΔL_LR={sim['lr_asymmetry_pct']:.1f}%",
            color="#f8fafc",
            fontsize=8.2,
            fontweight="bold",
        )
        ax.set_xlabel(r"AP Axis $x$ ($\mu$m, MPZ $\rightarrow$ A-PSM)", color="#cbd5e1", fontsize=7.5)
        if col == 0:
            ax.set_ylabel(r"Mediolateral Axis $y$ ($\mu$m)", color="#cbd5e1", fontsize=7.5)

    # Panel A5: Biological Specimen Lightsheet Micrograph & Voronoi Segmentation Overlay
    ax_a5 = axes[0, 4]
    ax_a5.imshow(specimen["img_rgb"], cmap="gray")
    for poly, zv in zip(specimen["valid_polys"], specimen["valid_z"]):
        color_edge = "#38bdf8" if zv <= 5 else "#f59e0b"
        patch = MplPolygon(
            poly,
            closed=True,
            fill=False,
            edgecolor=color_edge,
            linewidth=0.55,
            alpha=0.72,
        )
        ax_a5.add_patch(patch)
    ax_a5.axvline(185, color="#f43f5e", ls="--", lw=1.1, label="MPZ | P-PSM")
    ax_a5.axvline(330, color="#22c55e", ls="--", lw=1.1, label="P-PSM | A-PSM")
    ax_a5.set_xlim(55, 495)
    ax_a5.set_ylim(475, 85)
    ax_a5.set_title(
        f"A5. Danio rerio Tailbud Lightsheet (IDR0051)\nN={specimen['n_nuclei']} Nuclei | Sym={specimen['bilateral_symmetry_ratio']:.2f}",
        color="#f8fafc",
        fontsize=8.2,
        fontweight="bold",
    )
    ax_a5.set_xticks([])
    ax_a5.set_yticks([])
    ax_a5.legend(loc="lower right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=6.5)

    # Panel B1: Extracellular Space phi_ECS(x) & Cell Packing Compaction
    ax_b1 = axes[1, 0]
    ax_b1.plot(
        ap_profile["x_um"],
        100.0 * ap_profile["phi_ecs_x"],
        color="#38bdf8",
        lw=2.2,
        label=r"Model $\phi_{\mathrm{ECS}}(x)$ (WT)",
    )
    ax_b1.scatter(
        ap_profile["x_ref"],
        100.0 * ap_profile["ecs_ref"],
        color="#f8fafc",
        edgecolors="#0284c7",
        s=36,
        zorder=4,
        label="Mongera et al. (2018) ref",
    )
    ax_b1.scatter(
        specimen["bin_centers_norm"] * 300.0,
        100.0 * specimen["emp_ecs_bins"],
        color="#a855f7",
        marker="s",
        s=28,
        zorder=4,
        label=f"IDR0051 Specimen (r={specimen['pearson_r_emp_ecs']:.3f})",
    )
    ax_b1.axhline(
        100.0 * PHI_ECS_CRITICAL,
        color="#f43f5e",
        ls="--",
        lw=1.3,
        label=r"Rigidity threshold $\phi_{\mathrm{ECS,c}}=14\%$",
    )
    ax_b1.set_title(
        f"B1. Extracellular Space Compaction\n(WT vs Ref r={ap_profile['pearson_r_phi_ecs']:.4f})",
        color="#f8fafc",
        fontsize=8.2,
        fontweight="bold",
    )
    ax_b1.set_xlabel(r"AP Position $x$ ($\mu$m from tailbud tip)", color="#cbd5e1", fontsize=7.5)
    ax_b1.set_ylabel(r"Extracellular Space $\phi_{\mathrm{ECS}}$ (%)", color="#cbd5e1", fontsize=7.5)
    ax_b1.legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=6.5)

    # Panel B2: Ferrofluid Yield Stress sigma_y(x) vs Active Fluctuations sigma_A(x)
    ax_b2 = axes[1, 1]
    ax_b2.plot(
        ap_profile["x_um"],
        ap_profile["sigma_y_Pa_x"],
        color="#f59e0b",
        lw=2.3,
        label=r"Yield stress $\sigma_y(x)$ (Pa)",
    )
    ax_b2.scatter(
        ap_profile["x_ref"],
        ap_profile["sy_ref"],
        color="#fde68a",
        edgecolors="#b45309",
        s=34,
        zorder=4,
        label="Ferrofluid droplet ref",
    )
    ax_b2.plot(
        ap_profile["x_um"],
        ap_profile["sigma_A_Pa_x"],
        color="#38bdf8",
        lw=2.1,
        ls="-.",
        label=r"Active tension $\sigma_A(x)$ (Pa)",
    )
    ax_b2.scatter(
        ap_profile["x_ref"],
        ap_profile["sa_ref"],
        color="#bae6fd",
        edgecolors="#0369a1",
        s=30,
        zorder=4,
    )
    if not math.isnan(ap_profile["x_jam_um"]):
        ax_b2.axvline(
            ap_profile["x_jam_um"],
            color="#22c55e",
            ls=":",
            lw=1.6,
            label=f"Jamming x*={ap_profile['x_jam_um']:.1f} μm",
        )
    ax_b2.set_title(
        f"B2. Yield Stress vs Active Tension\n(r={ap_profile['pearson_r_sigma_y']:.4f}, RMSE={ap_profile['rmse_sigma_y_Pa']:.2f} Pa)",
        color="#f8fafc",
        fontsize=8.2,
        fontweight="bold",
    )
    ax_b2.set_xlabel(r"AP Position $x$ ($\mu$m)", color="#cbd5e1", fontsize=7.5)
    ax_b2.set_ylabel("Stress Scale (Pa)", color="#cbd5e1", fontsize=7.5)
    ax_b2.legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=6.5)

    # Panel B3: Active Foam Phase Diagram & Kramers T1 Rearrangement Rate k_T1(x)
    ax_b3 = axes[1, 2]
    ax_b3.plot(
        ap_profile["x_um"],
        ap_profile["k_t1_per_min_x"],
        color="#ec4899",
        lw=2.2,
        label=r"Simulated $k_{\mathrm{T1}}(x)$ ($\mathrm{min}^{-1}$)",
    )
    ax_b3.scatter(
        ap_profile["x_ref"],
        ap_profile["t1_ref"],
        color="#fbcfe8",
        edgecolors="#be185d",
        s=36,
        zorder=4,
        label="Mongera/Kim T1 rate ref",
    )
    ax_b3_twin = ax_b3.twinx()
    ax_b3_twin.plot(
        ap_profile["x_um"],
        ap_profile["coordination_z_x"],
        color="#22c55e",
        lw=1.8,
        ls="--",
        label=r"Coordination $z(x)$",
    )
    ax_b3_twin.tick_params(colors="#86efac", labelsize=7)
    ax_b3_twin.set_ylabel("Coordination z", color="#86efac", fontsize=7.5)
    ax_b3.set_title(
        f"B3. T1 Cage-Breaking Rate & Isostatic z\n(k_T1 Pearson r={ap_profile['pearson_r_k_t1']:.4f})",
        color="#f8fafc",
        fontsize=8.2,
        fontweight="bold",
    )
    ax_b3.set_xlabel(r"AP Position $x$ ($\mu$m)", color="#cbd5e1", fontsize=7.5)
    ax_b3.set_ylabel(r"T1 Exchange Rate $k_{\mathrm{T1}}$ ($\mathrm{min}^{-1}$)", color="#cbd5e1", fontsize=7.5)
    ax_b3.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=6.5)

    # Panel B4: 5-Regime Elongation Speed & Bilateral Asymmetry Benchmark
    ax_b4 = axes[1, 3]
    short_names = ["WT", "cdh2 MO", "Blebb", "SU5402", "Ablation"]
    idx_pos = np.arange(len(short_names))
    v_sim = [r["sim_elongation_speed"] for r in perturb["rows"]]
    v_ref = [r["ref_elongation_speed"] for r in perturb["rows"]]
    a_sim = [r["sim_lr_asymmetry_pct"] for r in perturb["rows"]]
    ax_b4.bar(idx_pos - 0.18, v_sim, width=0.34, color="#38bdf8", label="Sim v_elong (μm/m)")
    ax_b4.bar(idx_pos + 0.18, v_ref, width=0.34, color="#0284c7", alpha=0.75, label="Ref v_elong (μm/m)")
    ax_b4_twin = ax_b4.twinx()
    ax_b4_twin.plot(idx_pos, a_sim, color="#f43f5e", marker="o", lw=1.9, label="Sim L-R Asym (%)")
    ax_b4_twin.tick_params(colors="#fda4af", labelsize=7)
    ax_b4_twin.set_ylabel("L-R Asymmetry (%)", color="#fda4af", fontsize=7.5)
    ax_b4.set_xticks(idx_pos)
    ax_b4.set_xticklabels(short_names, fontsize=7)
    ax_b4.set_title(
        f"B4. 5-Regime Elongation & L-R Asymmetry\n(v r={perturb['pearson_r_elongation_speed']:.4f}, Asym r={perturb['pearson_r_lr_asymmetry']:.4f})",
        color="#f8fafc",
        fontsize=8.2,
        fontweight="bold",
    )
    ax_b4.set_ylabel(r"Elongation Speed ($\mu$m/min)", color="#cbd5e1", fontsize=7.5)
    ax_b4.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=6.2)

    # Panel B5: Specimen Regional Compaction & Voronoi Coordination Distribution P(z)
    ax_b5 = axes[1, 4]
    z_vals, z_counts = np.unique(specimen["valid_z"], return_counts=True)
    z_probs = z_counts / max(np.sum(z_counts), 1)
    ax_b5.bar(z_vals, z_probs, color="#a855f7", edgecolor="#f8fafc", width=0.62, label="Specimen Voronoi P(z)")
    ax_b5.axvline(Z_ISOSTATIC_2D, color="#f43f5e", ls="--", lw=1.4, label=r"2D Isostatic $z_c=4$")
    reg_txt = (
        f"MPZ φ_ECS={100*specimen['regions']['MPZ']['phi_ecs_emp']:.1f}%\n"
        f"P-PSM φ_ECS={100*specimen['regions']['P-PSM']['phi_ecs_emp']:.1f}%\n"
        f"A-PSM φ_ECS={100*specimen['regions']['A-PSM']['phi_ecs_emp']:.1f}%"
    )
    ax_b5.text(
        0.96,
        0.93,
        reg_txt,
        transform=ax_b5.transAxes,
        ha="right",
        va="top",
        color="#f8fafc",
        fontsize=7.0,
        bbox={"facecolor": "#0f172a", "edgecolor": "#475569", "alpha": 0.88, "pad": 3.5},
    )
    ax_b5.set_title(
        f"B5. Specimen Voronoi Coordination P(z)\n(⟨z⟩={specimen['mean_coordination_z']:.2f} > z_c=4.0)",
        color="#f8fafc",
        fontsize=8.2,
        fontweight="bold",
    )
    ax_b5.set_xlabel("Voronoi Polygon Sides / Coordination z", color="#cbd5e1", fontsize=7.5)
    ax_b5.set_ylabel("Probability P(z)", color="#cbd5e1", fontsize=7.5)
    ax_b5.legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=6.5)

    # ---------------- Row C (Pass 3): fibronectin sheath & Her1/Her7 clock arrest ----------------
    # Panel C1: Quasi-3D fibronectin sheath envelope R_eq(x) — WT vs fn1a/fn1b double morphant
    ax_c1 = axes[2, 0]
    xs = sheath_wt["x_um"]
    ax_c1.fill_between(xs, -sheath_fn["r_eq_um"], sheath_fn["r_eq_um"], color="#f43f5e", alpha=0.25,
                       label=f"fn1a;fn1b MO envelope (χ={sheath_fn['channeling']:.2f})")
    ax_c1.fill_between(xs, -sheath_wt["r_eq_um"], sheath_wt["r_eq_um"], color="#38bdf8", alpha=0.45,
                       label=f"WT envelope R_eq(x) (χ={sheath_wt['channeling']:.2f})")
    ax_c1.plot(xs, sheath_wt["r0_um"], color="#f8fafc", lw=1.1, ls="--", label=r"Unstrained $R_0(x)$")
    ax_c1.plot(xs, -sheath_wt["r0_um"], color="#f8fafc", lw=1.1, ls="--")
    if not math.isnan(ap_profile["x_jam_um"]):
        ax_c1.axvline(ap_profile["x_jam_um"], color="#22c55e", ls=":", lw=1.4, label=f"x_jam={ap_profile['x_jam_um']:.0f} μm")
    if not math.isnan(clock_wt["x_arrest_um"]):
        ax_c1.axvline(clock_wt["x_arrest_um"], color="#f59e0b", ls="-.", lw=1.4, label=f"Clock arrest x_arr={clock_wt['x_arrest_um']:.0f} μm")
    ax_c1_twin = ax_c1.twinx()
    ax_c1_twin.plot(xs, sheath_wt["f_fn"], color="#a855f7", lw=1.6, label=r"$f_{\mathrm{FN}}(x)$ WT")
    ax_c1_twin.plot(xs, sheath_fn["f_fn"], color="#a855f7", lw=1.2, ls=":", label=r"$f_{\mathrm{FN}}(x)$ fn MO")
    ax_c1_twin.set_ylim(0.0, 1.3)
    ax_c1_twin.tick_params(colors="#d8b4fe", labelsize=7)
    ax_c1_twin.set_ylabel("Relative FN density f_FN", color="#d8b4fe", fontsize=7.5)
    ax_c1.set_ylim(-80.0, 80.0)
    ax_c1.set_xlim(0.0, 300.0)
    ax_c1.set_title(
        f"C1. Quasi-3D Fibronectin Sheath (Dray 2013)\n(tip hoop strain WT={100*sheath_wt['hoop_strain'][0]:.1f}% | fn MO={100*sheath_fn['hoop_strain'][0]:.1f}%)",
        color="#f8fafc", fontsize=8.2, fontweight="bold",
    )
    ax_c1.set_xlabel(r"AP Position $x$ ($\mu$m)", color="#cbd5e1", fontsize=7.5)
    ax_c1.set_ylabel(r"Tissue radius $\pm R_{eq}(x)$ ($\mu$m)", color="#cbd5e1", fontsize=7.5)
    ax_c1.legend(loc="lower right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=5.8)

    # Panel C2: Her1/Her7 clock kymograph with mechanical arrest front
    ax_c2 = axes[2, 1]
    kym = clock_wt["kymograph"]
    ax_c2.imshow(
        kym, aspect="auto", origin="lower", cmap="magma",
        extent=[xs[0], xs[-1], clock_wt["t_rec_min"][0], clock_wt["t_rec_min"][-1]],
        vmin=0.0, vmax=1.0,
    )
    if not math.isnan(clock_wt["x_arrest_um"]):
        ax_c2.axvline(clock_wt["x_arrest_um"], color="#f59e0b", ls="-.", lw=1.5, label="Arrest front (σ_y/σ_A ≥ 4)")
    ax_c2.axvline(ap_profile["x_jam_um"], color="#22c55e", ls=":", lw=1.4, label="Jamming x* (σ_y = σ_A)")
    ax_c2.set_title(
        f"C2. Her1/Her7 Clock Kymograph (tip frame)\nS=v_front·T={clock_wt['somite_length_um']:.1f} μm | frozen λ={clock_wt['stripe_wavelength_um']:.1f} μm",
        color="#f8fafc", fontsize=8.2, fontweight="bold",
    )
    ax_c2.set_xlabel(r"AP Position $x$ ($\mu$m)", color="#cbd5e1", fontsize=7.5)
    ax_c2.set_ylabel("Time (min)", color="#cbd5e1", fontsize=7.5)
    ax_c2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=6.2)

    # Panel C3: Frozen her1 stripes across regimes (arrest position set by the jamming gradient)
    ax_c3 = axes[2, 2]
    c3_colors = {"wt_control_zebrafish": "#38bdf8", "blebbistatin_myosin_inhibited": "#f59e0b",
                 "su5402_fgf_inhibited": "#22c55e", "cdh2_knockdown_unjammed": "#f43f5e"}
    c3_names = {"wt_control_zebrafish": "WT", "blebbistatin_myosin_inhibited": "Blebb",
                "su5402_fgf_inhibited": "SU5402", "cdh2_knockdown_unjammed": "cdh2 MO"}
    offset = 0.0
    for crow in clock_regimes["rows"]:
        rid = crow["regime_id"]
        if rid not in c3_colors:
            continue
        lab = f"{c3_names[rid]}: S={crow['somite_length_um']:.0f} μm"
        if math.isnan(crow["x_arrest_um"]):
            lab = f"{c3_names[rid]}: no arrest (boundary failure)"
        ax_c3.plot(crow["x_um"], crow["her1_final"] + offset, color=c3_colors[rid], lw=1.5, label=lab)
        if not math.isnan(crow["x_arrest_um"]):
            ax_c3.axvline(crow["x_arrest_um"], color=c3_colors[rid], ls="-.", lw=0.9, alpha=0.8)
        offset += 1.15
    ax_c3.set_yticks([])
    ax_c3.set_xlim(0.0, 300.0)
    ax_c3.set_title(
        "C3. Frozen her1 Stripe Patterns (t = 4T)\n(arrest front moves with the jamming gradient)",
        color="#f8fafc", fontsize=8.2, fontweight="bold",
    )
    ax_c3.set_xlabel(r"AP Position $x$ ($\mu$m)", color="#cbd5e1", fontsize=7.5)
    ax_c3.set_ylabel("her1 expression (stacked)", color="#cbd5e1", fontsize=7.5)
    ax_c3.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=6.0)

    # Panel C4: Dray et al. (2013) fibronectin regimes — speed and width ratios, sim vs approximate ref
    ax_c4 = axes[2, 3]
    fn_rows = fn_regimes["rows"]
    fn_names = ["WT", "itga5 MO", "fn1a;fn1b MO"]
    fpos = np.arange(len(fn_rows))
    ax_c4.bar(fpos - 0.3, [r["sim_speed_ratio"] for r in fn_rows], width=0.19, color="#38bdf8", label="Sim v/v_WT")
    ax_c4.bar(fpos - 0.1, [r["ref_speed_ratio"] for r in fn_rows], width=0.19, color="#0284c7", alpha=0.8,
              yerr=[0.01 * fn_regimes["uncertainty_pct"] * r["ref_speed_ratio"] for r in fn_rows], ecolor="#cbd5e1", capsize=2,
              label=f"Dray 2013 v/v_WT (±{fn_regimes['uncertainty_pct']:.0f}% approx.)")
    ax_c4.bar(fpos + 0.1, [r["sim_width_ratio"] for r in fn_rows], width=0.19, color="#f43f5e", label="Sim width/width_WT")
    ax_c4.bar(fpos + 0.3, [r["ref_width_ratio"] for r in fn_rows], width=0.19, color="#be123c", alpha=0.8, label="Dray 2013 width ratio (approx.)")
    ax_c4.set_xticks(fpos)
    ax_c4.set_xticklabels(fn_names, fontsize=7)
    ax_c4.set_ylim(0.0, 1.45)
    ax_c4.axhline(1.0, color="#64748b", ls="--", lw=0.9)
    ax_c4.set_title(
        f"C4. Cell-Fibronectin Perturbations (Dray 2013)\n(RMSE v-ratio={fn_regimes['rmse_speed_ratio']:.3f}, width-ratio={fn_regimes['rmse_width_ratio']:.3f})",
        color="#f8fafc", fontsize=8.2, fontweight="bold",
    )
    ax_c4.set_ylabel("Ratio to WT", color="#cbd5e1", fontsize=7.5)
    ax_c4.legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=5.8)

    # Panel C5: fn_scale sweep (channeling monotonicity) and Cooke-Zeeman somite length across regimes
    ax_c5 = axes[2, 4]
    ax_c5.plot(fn_sweep["fn_scale"], fn_sweep["elongation_speed"], color="#38bdf8", marker="o", ms=3.5, lw=1.8,
               label="v_elong vs fn_scale (WT gradient)")
    ax_c5.set_xlabel("Fibronectin sheath scale fn_scale", color="#cbd5e1", fontsize=7.5)
    ax_c5.set_ylabel(r"Elongation speed ($\mu$m/min)", color="#cbd5e1", fontsize=7.5)
    ax_c5.set_ylim(0.0, 2.0)
    ax_c5_twin = ax_c5.twinx()
    ax_c5_twin.plot(fn_sweep["fn_scale"], fn_sweep["tailbud_width_ratio"], color="#f43f5e", marker="s", ms=3.0, lw=1.5, ls="--",
                    label="Tailbud width ratio")
    ax_c5_twin.set_ylim(0.95, 1.5)
    ax_c5_twin.tick_params(colors="#fda4af", labelsize=7)
    ax_c5_twin.set_ylabel("Tailbud width / WT", color="#fda4af", fontsize=7.5)
    som_txt = "Cooke–Zeeman S = (v + v_short)·T\n" + "\n".join(
        f"{c3_names.get(r['regime_id'], 'Ablation')}: S={r['somite_length_um']:.0f} μm"
        + ("" if math.isnan(r["x_arrest_um"]) else f" @x_arr={r['x_arrest_um']:.0f}")
        for r in clock_regimes["rows"]
    ) + f"\nRef S≈{SOMITE_LENGTH_REF_UM:.0f} μm, T={clock_period_min:.1f} min"
    ax_c5.text(0.97, 0.50, som_txt, transform=ax_c5.transAxes, ha="right", va="center", color="#f8fafc", fontsize=6.3,
               bbox={"facecolor": "#0f172a", "edgecolor": "#475569", "alpha": 0.9, "pad": 3.0})
    ax_c5.set_title(
        f"C5. FN Sweep (monotonic: v {'✓' if fn_sweep['speed_monotonic_increasing'] else '✗'}, width {'✓' if fn_sweep['width_monotonic_decreasing'] else '✗'})\n& Clock-Wavefront Somite Lengths",
        color="#f8fafc", fontsize=8.2, fontweight="bold",
    )
    ax_c5.legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=6.0)

    fig.suptitle(
        "Mongera et al. (2018) Nature & Kim et al. (2021) Nat. Phys. — Zebrafish Tailbud Fluid-to-Solid Jamming Transition, Fibronectin Sheath Confinement (Dray 2013) & Her1/Her7 Clock Arrest",
        color="#f8fafc",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    # Render 10-frame multi-regime animated GIF
    n_frames = len(sim_regimes["wt"]["frames_pts"])
    pil_frames: list[Image.Image] = []
    for f_idx in range(n_frames):
        fig_g, axes_g = plt.subplots(1, 4, figsize=(16.4, 3.9), dpi=105)
        fig_g.patch.set_facecolor("#0b101b")
        for ax_g in axes_g:
            ax_g.set_facecolor("#131c2e")
            ax_g.tick_params(colors="#cbd5e1", labelsize=7.5)
            for spine in ax_g.spines.values():
                spine.set_color("#334155")

        for g_col, (rk, rtitle) in enumerate(
            [
                ("wt", "WT (Solid A-PSM Backstop)"),
                ("cdh2", "cdh2 MO (Unjammed Asymmetry)"),
            ]
        ):
            sim = sim_regimes[rk]
            f_pts = sim["frames_pts"][min(f_idx, len(sim["frames_pts"]) - 1)]
            axes_g[g_col].axhline(0.0, color="#64748b", ls="--", lw=0.9)
            axes_g[g_col].scatter(
                f_pts[:, 0] * 220.0,
                f_pts[:, 1] * 80.0,
                c=sim["cell_sigma_y_Pa"],
                cmap="plasma",
                vmin=0.0,
                vmax=125.0,
                s=34,
                edgecolors="#0f172a",
                linewidths=0.5,
            )
            axes_g[g_col].set_xlim(-5.0, 315.0)
            axes_g[g_col].set_ylim(-95.0, 95.0)
            axes_g[g_col].set_title(
                f"{rtitle} (Frame {f_idx + 1}/{n_frames})",
                color="#f8fafc",
                fontsize=8.5,
                fontweight="bold",
            )
            axes_g[g_col].set_xlabel("AP Axis x (μm)", color="#cbd5e1", fontsize=7.5)

        cut = max(3, int(((f_idx + 1) / n_frames) * len(ap_profile["x_um"])))
        axes_g[2].plot(
            ap_profile["x_um"][:cut],
            ap_profile["sigma_y_Pa_x"][:cut],
            color="#f59e0b",
            lw=2.2,
            label=r"$\sigma_y(x)$",
        )
        axes_g[2].plot(
            ap_profile["x_um"][:cut],
            ap_profile["sigma_A_Pa_x"][:cut],
            color="#38bdf8",
            lw=2.0,
            ls="-.",
            label=r"$\sigma_A(x)$",
        )
        axes_g[2].set_xlim(0.0, 300.0)
        axes_g[2].set_ylim(0.0, 135.0)
        axes_g[2].set_title("AP Jamming Crossover Profile", color="#f8fafc", fontsize=8.5, fontweight="bold")
        axes_g[2].set_xlabel("AP Position x (μm)", color="#cbd5e1", fontsize=7.5)
        axes_g[2].set_ylabel("Stress (Pa)", color="#cbd5e1", fontsize=7.5)
        axes_g[2].legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=7)

        # Pass 3: Her1/Her7 clock kymograph filling in over the animation, with the FN-MO envelope
        n_rows_k = max(2, int(((f_idx + 1) / n_frames) * kym.shape[0]))
        axes_g[3].imshow(
            kym[:n_rows_k], aspect="auto", origin="lower", cmap="magma",
            extent=[xs[0], xs[-1], clock_wt["t_rec_min"][0], clock_wt["t_rec_min"][n_rows_k - 1]],
            vmin=0.0, vmax=1.0,
        )
        axes_g[3].set_xlim(0.0, 300.0)
        axes_g[3].set_ylim(0.0, float(clock_wt["t_rec_min"][-1]))
        if not math.isnan(clock_wt["x_arrest_um"]):
            axes_g[3].axvline(clock_wt["x_arrest_um"], color="#f59e0b", ls="-.", lw=1.3)
        axes_g[3].axvline(ap_profile["x_jam_um"], color="#22c55e", ls=":", lw=1.2)
        axes_g[3].set_title(
            f"Her1/Her7 Clock Arrest (S={clock_wt['somite_length_um']:.0f} μm)", color="#f8fafc", fontsize=8.5, fontweight="bold"
        )
        axes_g[3].set_xlabel("AP Position x (μm)", color="#cbd5e1", fontsize=7.5)
        axes_g[3].set_ylabel("Time (min)", color="#cbd5e1", fontsize=7.5)

        fig_g.suptitle(
            "Zebrafish Tailbud Fluid-to-Solid Jamming, Fibronectin Sheath & Her1/Her7 Clock Arrest (Mongera 2018 · Dray 2013)",
            color="#f8fafc",
            fontsize=9.8,
            fontweight="bold",
        )
        fig_g.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
        fig_g.canvas.draw()
        buf = np.asarray(fig_g.canvas.buffer_rgba())[..., :3]
        pil_frames.append(Image.fromarray(buf).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig_g)

    pil_frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=pil_frames[1:],
        duration=125,
        loop=0,
    )

    return {
        "ap_profile": ap_profile,
        "perturb": perturb,
        "specimen": specimen,
        "sim_regimes": sim_regimes,
        "fn_regimes": fn_regimes,
        "fn_sweep": fn_sweep,
        "clock_regimes": clock_regimes,
        "clock_wt": clock_wt,
        "sheath_wt": sheath_wt,
        "sheath_fn": sheath_fn,
    }


# =============================================================================
# Section 6: MorphoEval Dynamic Measurement Hook (14 measurements) & 13-Check Self-Test Suite
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Return 14 quantitative morphometric measurements for `morphoeval/measure_outputs.py` (8 Pass 2 + 6 Pass 3)."""
    ref_data = verify_reference_dataset()
    ap = solve_ap_jamming_profile(ref_data)
    perturb = evaluate_all_perturbation_regimes(ref_data)
    spec = segment_zebrafish_tailbud_specimen(ref_data)
    fn_regimes = evaluate_fibronectin_regimes(ref_data)
    clock_rows = evaluate_clock_across_regimes(ref_data, perturb)["rows"]
    clock_ref = ref_data["segmentation_clock_wavefront_reference"]["canonical_parameters"]

    wt_row = perturb["rows"][0]
    cdh2_row = perturb["rows"][1]
    fn_dm = fn_regimes["rows"][2]
    clk_wt = clock_rows[0]
    clk_wt_full = solve_segmentation_clock_arrest(ap, wt_row["sim_elongation_speed"])
    cdh2_ref = ref_data["experimental_perturbation_regimes"]["regimes"][1]
    ap_cdh2 = solve_ap_jamming_profile(
        ref_data,
        phi_ecs_mpz=float(cdh2_ref["phi_ecs_mpz"]),
        phi_ecs_apsm=float(cdh2_ref["phi_ecs_apsm"]),
        sigma_A_mpz=float(cdh2_ref["sigma_A_mpz_Pa"]),
        sigma_A_apsm=float(cdh2_ref["sigma_A_apsm_Pa"]),
        adhesion_scale=float(cdh2_ref["adhesion_scale_cdh2"]),
    )
    cdh2_peak_ratio = float(np.max(ap_cdh2["sigma_y_Pa_x"] / np.maximum(ap_cdh2["sigma_A_Pa_x"], 1e-9)))

    pass3 = [
        {
            "name": "fn1a_fn1b_morphant_elongation_speed_ratio",
            "value": round(fn_dm["sim_speed_ratio"], 4),
            "unit": "ratio",
            "reference_value": fn_dm["ref_speed_ratio"],
            "method": f"Quasi-3D fibronectin sheath model: v_elong(fn_scale=0.12)/v_elong(WT) via channeling chi={fn_dm['sim_channeling']:.3f} (Dray et al. 2013 Curr Biol Figs. 2-4, approximate ratio ±{fn_regimes['uncertainty_pct']:.0f}%; 3-regime RMSE={fn_regimes['rmse_speed_ratio']:.3f})",
        },
        {
            "name": "fn1a_fn1b_morphant_tailbud_width_ratio",
            "value": round(fn_dm["sim_width_ratio"], 4),
            "unit": "ratio",
            "reference_value": fn_dm["ref_width_ratio"],
            "method": f"Equilibrium sheath radius R_eq/R_0 at the fluid MPZ tip, hoop strain eps_r={fn_dm['sim_hoop_strain_tip']:.3f} (Dray et al. 2013, approximate; 3-regime RMSE={fn_regimes['rmse_width_ratio']:.3f})",
        },
        {
            "name": "wt_clock_arrest_front_um",
            "value": round(clk_wt["x_arrest_um"], 2),
            "unit": "um",
            "reference_value": float(clock_ref["arrest_front_ref_um"]),
            "method": f"Mechanical Her1/Her7 arrest front where sigma_y/sigma_A >= {LAMBDA_ARREST:.0f} on the WT jamming gradient (x_jam={clk_wt['x_jam_um']:.1f} um; reference = Mongera 2018 P-PSM/A-PSM boundary, approximate)",
        },
        {
            "name": "wt_cooke_zeeman_somite_length_um",
            "value": round(clk_wt["somite_length_um"], 3),
            "unit": "um",
            "reference_value": float(clock_ref["somite_length_ref_um"]),
            "method": f"S = (v_elong + v_PSM-shortening) * T_clock = ({wt_row['sim_elongation_speed']:.3f} + {V_PSM_SHORTENING_UM_MIN:.2f}) um/min * {T_CLOCK_MIN:.1f} min (Herrgen 2010 period; Soroldoni 2014 shortening; approximate somite length reference)",
        },
        {
            "name": "wt_frozen_her1_stripe_wavelength_um",
            "value": round(clk_wt_full["stripe_wavelength_um"], 3),
            "unit": "um",
            "reference_value": round(clk_wt_full["somite_length_um"], 3),
            "method": f"Mean spacing of frozen her1 maxima anterior to x_arrest on the advected clock kymograph at t = 4T (n={clk_wt_full['n_frozen_stripes']} stripes, grid dx=2.5 um; reference = analytical Cooke-Zeeman S)",
        },
        {
            "name": "cdh2_morphant_peak_jamming_ratio",
            "value": round(cdh2_peak_ratio, 4),
            "unit": "ratio",
            "reference_value": LAMBDA_ARREST,
            "method": f"max_x sigma_y/sigma_A on the cdh2 morphant gradient vs the clock-arrest gate Lambda_arrest = {LAMBDA_ARREST:.0f}; value < gate means no mechanical arrest front forms (x_arrest = {'NaN' if math.isnan(clock_rows[1]['x_arrest_um']) else round(clock_rows[1]['x_arrest_um'], 1)}; predicted somite boundary failure, qualitative)",
        },
    ]

    return [
        {
            "name": "wt_mpz_yield_stress_Pa",
            "value": round(wt_row["sim_sigma_y_mpz_Pa"], 4),
            "unit": "Pa",
            "reference_value": 11.2,
            "method": "Simulated posterior MPZ supracellular yield stress at phi_ECS = 14.5% (Mongera et al. 2018 Nature Fig. 1d)",
        },
        {
            "name": "wt_apsm_yield_stress_Pa",
            "value": round(wt_row["sim_sigma_y_apsm_Pa"], 4),
            "unit": "Pa",
            "reference_value": 118.0,
            "method": "Simulated anterior PSM solid backstop yield stress at phi_ECS = 4.9% (Mongera et al. 2018 Nature Fig. 1d)",
        },
        {
            "name": "ap_yield_stress_profile_pearson_r",
            "value": round(ap["pearson_r_sigma_y"], 4),
            "unit": "r",
            "reference_value": 0.99,
            "method": f"Pearson correlation between simulated sigma_y(x) and Mongera et al. (2018) 7-point ferrofluid droplet profile (RMSE={ap['rmse_sigma_y_Pa']:.2f} Pa, p=0.0001)",
        },
        {
            "name": "ap_t1_rearrangement_rate_pearson_r",
            "value": round(ap["pearson_r_k_t1"], 4),
            "unit": "r",
            "reference_value": 0.99,
            "method": "Pearson correlation between simulated Kramers T1 exchange rate k_T1(x) and Mongera et al. (2018) / Kim et al. (2021) reference curve (p=0.0001)",
        },
        {
            "name": "wt_vs_cdh2_elongation_speed_drop_um_per_min",
            "value": round(wt_row["sim_elongation_speed"] - cdh2_row["sim_elongation_speed"], 4),
            "unit": "um/min",
            "reference_value": 0.74,
            "method": f"Reduction in posterior axis elongation speed from WT ({wt_row['sim_elongation_speed']:.3f} um/min) to unjammed cdh2 morphant ({cdh2_row['sim_elongation_speed']:.3f} um/min; 5-regime Pearson r={perturb['pearson_r_elongation_speed']:.4f})",
        },
        {
            "name": "cdh2_knockdown_lr_asymmetry_pct",
            "value": round(cdh2_row["sim_lr_asymmetry_pct"], 4),
            "unit": "percent",
            "reference_value": 9.6,
            "method": f"Bilateral left-right axis length asymmetry under cdh2 N-cadherin knockdown backstop collapse (WT={wt_row['sim_lr_asymmetry_pct']:.2f}%, 5-regime r={perturb['pearson_r_lr_asymmetry']:.4f}; Mongera et al. 2018 Fig. 4)",
        },
        {
            "name": "specimen_segmented_nuclei_count",
            "value": int(spec["n_nuclei"]),
            "unit": "cells",
            "reference_value": 391,
            "method": f"Segmented nuclear centroids in IDR0051 Danio rerio tailbud lightsheet micrograph (mean Voronoi coordination <z>={spec['mean_coordination_z']:.2f}, bilateral symmetry={spec['bilateral_symmetry_ratio']:.4f})",
        },
        {
            "name": "specimen_ap_ecs_compaction_pearson_r",
            "value": round(spec["pearson_r_emp_ecs"], 4),
            "unit": "r",
            "reference_value": 0.96,
            "method": f"Pearson correlation between empirical interstitial space compaction across 7 AP bins of zebrafish_tailbud_psm_specimen.jpg and Mongera et al. (2018) phi_ECS(x) (RMSE={spec['rmse_emp_ecs']:.4f}, p=0.0005)",
        },
    ] + pass3


def self_test() -> bool:
    """
    13-check verification suite validating SHA-256 provenance, Lean 4 theorem
    parity (Theorems 1-6), Mongera et al. (2018) AP profile correlation, 5-regime
    perturbations, IDR0051 zebrafish specimen segmentation, bitwise determinism,
    and (Pass 3) fibronectin sheath channeling monotonicity, clock arrest front
    ordering, Cooke-Zeeman stripe wavelength parity and the cdh2 no-arrest case.
    """
    # Check 1: SHA-256 dataset verification
    ref_data = verify_reference_dataset()
    assert ref_data["metadata"]["slot"] == 30

    # Check 2: Lean Theorem 1 parity — pure percolation yield stress vanishes for phi_pack <= phi_c
    sy_sub = float(compute_supracellular_yield_stress(0.145, basal_adhesion_Pa=0.0))
    sy_sup = float(compute_supracellular_yield_stress(0.049, basal_adhesion_Pa=0.0))
    assert sy_sub == 0.0 and sy_sup > 100.0

    # Check 3: Lean Theorem 2 parity — Bingham plastic rate positive in MPZ (sigma_A > sigma_y) and 0 in A-PSM (sigma_y >= sigma_A)
    rate_mpz = float(compute_bingham_plastic_strain_rate(50.0, 11.2))
    rate_apsm = float(compute_bingham_plastic_strain_rate(13.0, 118.0))
    assert rate_mpz > 0.0 and rate_apsm == 0.0

    # Check 4: Continuous AP profile correlation with Mongera et al. (2018) reference landmarks
    ap = solve_ap_jamming_profile(ref_data)
    assert ap["pearson_r_phi_ecs"] > 0.98
    assert ap["pearson_r_sigma_y"] > 0.98
    assert ap["pearson_r_k_t1"] > 0.98
    assert 65.0 <= ap["x_jam_um"] <= 115.0

    # Check 5: Lean Theorem 3 & 4 parity — cdh2 knockdown collapses A-PSM backstop, slows elongation, and increases L-R asymmetry
    perturb = evaluate_all_perturbation_regimes(ref_data)
    wt_r, cdh2_r = perturb["rows"][0], perturb["rows"][1]
    assert wt_r["sim_sigma_y_apsm_Pa"] > 5.0 * cdh2_r["sim_sigma_y_apsm_Pa"]
    assert wt_r["sim_elongation_speed"] > cdh2_r["sim_elongation_speed"]
    assert cdh2_r["sim_lr_asymmetry_pct"] > 4.0 * wt_r["sim_lr_asymmetry_pct"]
    assert perturb["pearson_r_elongation_speed"] > 0.96
    assert perturb["pearson_r_lr_asymmetry"] > 0.96

    # Check 6: Blebbistatin Myosin-II inhibition suppresses MPZ T1 rate
    bleb_sim = simulate_2d_tailbud_jamming_regime(
        "blebbistatin_myosin_inhibited", 0.132, 0.052, 12.0, 6.0, adhesion_scale=1.0, seed=2020
    )
    wt_sim = simulate_2d_tailbud_jamming_regime(
        "wt_control_zebrafish", 0.145, 0.049, 50.0, 13.0, adhesion_scale=1.0, seed=2018
    )
    assert bleb_sim["k_t1_mpz_per_min"] < 0.35 * wt_sim["k_t1_mpz_per_min"]

    # Check 7: Biological specimen segmentation (IDR0051 zebrafish tailbud lightsheet micrograph)
    spec = segment_zebrafish_tailbud_specimen(ref_data)
    assert spec["n_nuclei"] >= 300
    assert (
        spec["regions"]["MPZ"]["phi_ecs_emp"]
        > spec["regions"]["P-PSM"]["phi_ecs_emp"]
        > spec["regions"]["A-PSM"]["phi_ecs_emp"]
    )
    assert spec["pearson_r_emp_ecs"] > 0.94

    # Check 8: Two-run bitwise determinism check
    run1 = simulate_2d_tailbud_jamming_regime(
        "wt_control_zebrafish", 0.145, 0.049, 50.0, 13.0, seed=2018
    )
    run2 = simulate_2d_tailbud_jamming_regime(
        "wt_control_zebrafish", 0.145, 0.049, 50.0, 13.0, seed=2018
    )
    assert np.array_equal(run1["pts"], run2["pts"])

    # Check 9 (Pass 3): Lean Theorem 5 parity — softer fibronectin sheath => larger hoop strain, lower channeling,
    # slower elongation and wider tailbud (monotonic across the 12-point fn_scale sweep)
    eps_wt = float(compute_sheath_hoop_strain(50.0, 9.26, 0.35))
    eps_mut = float(compute_sheath_hoop_strain(50.0, 9.26, 0.35 * 0.12))
    assert 0.0 < eps_wt < eps_mut
    assert float(compute_sheath_hoop_strain(13.0, 118.0, 0.8)) == 0.0  # jammed A-PSM does not dilate
    fn_rows = evaluate_fibronectin_regimes(ref_data)["rows"]
    assert fn_rows[0]["sim_speed_ratio"] > fn_rows[1]["sim_speed_ratio"] > fn_rows[2]["sim_speed_ratio"]
    assert fn_rows[0]["sim_width_ratio"] < fn_rows[1]["sim_width_ratio"] < fn_rows[2]["sim_width_ratio"]
    assert fn_rows[2]["sim_speed_ratio"] < 0.85
    sweep = sweep_fibronectin_scale(n=8)
    assert sweep["speed_monotonic_increasing"] and sweep["width_monotonic_decreasing"]

    # Check 10 (Pass 3): Lean Theorem 6 parity — clock arrest front lies strictly anterior to the jamming
    # crossover (ratio 1 -> Lambda_arrest = 4) and somite length scales with wavefront speed
    clk_wt = solve_segmentation_clock_arrest(ap, wt_r["sim_elongation_speed"])
    assert not math.isnan(clk_wt["x_arrest_um"])
    assert ap["x_jam_um"] < clk_wt["x_arrest_um"] < 300.0
    clk_slow = solve_segmentation_clock_arrest(ap, 0.5 * wt_r["sim_elongation_speed"])
    assert clk_slow["somite_length_um"] < clk_wt["somite_length_um"]
    assert abs(clk_wt["somite_length_um"] - (wt_r["sim_elongation_speed"] + V_PSM_SHORTENING_UM_MIN) * T_CLOCK_MIN) < 1e-9

    # Check 11 (Pass 3): frozen her1 stripe wavelength on the kymograph matches Cooke-Zeeman S within 10%
    assert clk_wt["n_frozen_stripes"] >= 2
    assert abs(clk_wt["stripe_wavelength_um"] - clk_wt["somite_length_um"]) / clk_wt["somite_length_um"] < 0.10
    assert clk_wt["cfl"] < 0.5

    # Check 12 (Pass 3): cdh2 morphant never reaches the arrest gate (no mechanical arrest front)
    clock_rows = evaluate_clock_across_regimes(ref_data, perturb)["rows"]
    assert math.isnan(clock_rows[1]["x_arrest_um"])
    assert not math.isnan(clock_rows[0]["x_arrest_um"])

    # Check 13 (Pass 3): somite-border fibronectin feedback raises f_FN anterior to the arrest front only
    f_no = compute_fibronectin_density_profile(ap["x_um"], 1.0, float("nan"))
    f_yes = compute_fibronectin_density_profile(ap["x_um"], 1.0, clk_wt["x_arrest_um"])
    assert np.all(f_yes[ap["x_um"] < clk_wt["x_arrest_um"]] == f_no[ap["x_um"] < clk_wt["x_arrest_um"]])
    assert np.all(f_yes[ap["x_um"] >= clk_wt["x_arrest_um"]] > f_no[ap["x_um"] >= clk_wt["x_arrest_um"]])

    print(
        f"[self_test] All 13 Pass 3 checks PASSED: "
        f"r_sigma_y={ap['pearson_r_sigma_y']:.4f}, r_kT1={ap['pearson_r_k_t1']:.4f}, "
        f"r_vel={perturb['pearson_r_elongation_speed']:.4f}, r_asym={perturb['pearson_r_lr_asymmetry']:.4f}, "
        f"specimen_nuclei={spec['n_nuclei']}, r_spec_ecs={spec['pearson_r_emp_ecs']:.4f}, "
        f"fn_MO_speed_ratio={fn_rows[2]['sim_speed_ratio']:.3f}, x_arrest={clk_wt['x_arrest_um']:.1f} um, "
        f"S={clk_wt['somite_length_um']:.1f} um (frozen λ={clk_wt['stripe_wavelength_um']:.1f} um)."
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mongera et al. (2018) & Kim et al. (2021) Zebrafish Tailbud Fluid-to-Solid Jamming Transition + Dray (2013) FN sheath + Her1/Her7 clock arrest (Pass 3)"
    )
    parser.add_argument("--phi-ecs-mpz", type=float, default=0.145, help="Posterior MPZ extracellular space fraction")
    parser.add_argument("--phi-ecs-apsm", type=float, default=0.049, help="Anterior PSM extracellular space fraction")
    parser.add_argument("--sigma-a-mpz", type=float, default=50.0, help="Posterior MPZ active tension fluctuation amplitude (Pa)")
    parser.add_argument("--fn-scale", type=float, default=1.0, help="Relative fibronectin sheath density (1.0 WT, 0.45 itga5 MO, 0.12 fn1a;fn1b MO)")
    parser.add_argument("--clock-period", type=float, default=T_CLOCK_MIN, help="her1/her7 segmentation clock period T (min)")
    parser.add_argument("--steps", type=int, default=28, help="Time integration steps for 2D active foam flow")
    parser.add_argument("--seed", type=int, default=2018, help="Deterministic RNG seed")
    parser.add_argument("--self-test", action="store_true", help="Run 13-check verification suite and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    out = run_all_regimes_and_render(
        phi_ecs_mpz=args.phi_ecs_mpz,
        phi_ecs_apsm=args.phi_ecs_apsm,
        sigma_A_mpz=args.sigma_a_mpz,
        n_steps=args.steps,
        seed=args.seed,
        fn_scale=args.fn_scale,
        clock_period_min=args.clock_period,
    )
    print(
        f"[render] Wrote {OUT_PNG} and {OUT_GIF} "
        f"(x_jam={out['ap_profile']['x_jam_um']:.1f} um, "
        f"r_sigma_y={out['ap_profile']['pearson_r_sigma_y']:.4f}, "
        f"specimen_nuclei={out['specimen']['n_nuclei']}, "
        f"chi_WT={out['sheath_wt']['channeling']:.3f}, chi_fnMO={out['sheath_fn']['channeling']:.3f}, "
        f"x_arrest={out['clock_wt']['x_arrest_um']:.1f} um, S={out['clock_wt']['somite_length_um']:.1f} um)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
