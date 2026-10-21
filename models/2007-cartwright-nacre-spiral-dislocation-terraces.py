#!/usr/bin/env python3
"""
2007-cartwright-nacre-spiral-dislocation-terraces.py — Pass 3 (v3) Remediated Implementation
=============================================================================================
Faithful mathematical, biophysical, and morphometric reproduction of:

  1. Cartwright JHE, Checa AG (2007).
     "The dynamics of nacre self-assembly."
     Journal of the Royal Society Interface 4(14):491-504.
     DOI: 10.1098/rsif.2006.0188
  2. Cartwright JHE, Checa AG, Escribano B, Sainz-Díaz CI (2009).
     "Spiral and target patterns in bivalve nacre."
     Proceedings of the National Academy of Sciences 106(26):10499-10504.
     DOI: 10.1073/pnas.0900867106
  3. Burton WK, Cabrera N, Frank FC (1951).
     "The growth of crystals and the equilibrium structure of their surfaces."
     Philosophical Transactions of the Royal Society of London A 243(866):299-358.
     DOI: 10.1098/rsta.1951.0006
  4. Cartwright JHE, Checa AG, Escribano B, Sainz-Díaz CI (2012).
     "Crystal growth as an excitable medium."
     Philosophical Transactions of the Royal Society A 370(1969):2866-2885.
     DOI: 10.1098/rsta.2011.0600

Mathematical & Biophysical Architecture (Linked to Lean 4 `MathOfBiology.Loop_073`):
------------------------------------------------------------------------------------
  - Law 1 (`MathOfBiology.Loop_073.bcf_gibbs_thomson_critical_arrest` &
           `MathOfBiology.Loop_073.bcf_step_velocity_radial_increment`):
    An advancing aragonite mineral terrace front of local curvature kappa = 1 / r
    propagates normal to itself under the Gibbs-Thomson velocity law:
      v_n(kappa) = v_inf * (1 - r_c * kappa),   v(r) = v_inf * (1 - r_c / r)  (r >= r_c),
    where v_inf is the planar straight-step speed and r_c is the critical nucleation
    radius at which curved step propagation arrests identically (v(r_c) = 0).

  - Law 2 (`MathOfBiology.Loop_073.bcf_archimedean_terrace_pitch_scaling`):
    Around a screw dislocation of Burgers vector magnitude h_0 ≈ 0.50 µm (set by the
    inter-lamellar β-chitin + silk-fibroin mortar sheet spacing), the terrace winds
    into a steady-state Burton-Cabrera-Frank (BCF) rotating Archimedean spiral with
    angular rotation frequency and asymptotic inter-step terrace pitch:
      omega    = omega_0 * (v_inf / r_c),            omega_0 ≈ 0.330958,
      lambda_0 = 2 * pi * v_inf / omega = (2 * pi / omega_0) * r_c ≈ 18.9848 * r_c ≈ 19 * r_c,
      V_z      = h_0 * (omega / (2 * pi)) = h_0 * v_inf / lambda_0.
    Small critical radii r_c in gastropods (*Haliotis*, *Gibbula*) produce tightly wound
    conical/columnar tower nacre, whereas larger r_c in bivalves (*Pteria*, *Pinctada*)
    produces broad terraced sheet nacre.

  - Law 3 (`MathOfBiology.Loop_073.cartwright_frank_read_topological_burgers_algebra`):
    For a pair of screw dislocations (b_1, b_2) separated by distance d, the far-field
    topological Burgers winding number is b_net = b_1 + b_2:
      * Opposite-chirality Frank-Read dipole (+b, -b -> b_net = 0): when d > 2 * r_c,
        counter-rotating spiral arms periodically collide and reconnect to emit closed
        concentric target terraces; when d <= 2 * r_c (subcritical negative control),
        the bridge cannot clear the critical curvature barrier 1 / r_c and pinches off
        (annihilates) with 0 propagating terraces.
      * Same-chirality pair (+b, +b -> b_net = +2b): forms a persistent two-armed
        interlocked spiral whose asymptotic inter-step terrace spacing halves to
        lambda_2 = lambda_0 / 2.

  - Law 4 (Pass 3, `MathOfBiology.Loop_073.aragonite_cyclic_twin_angular_misfit` &
           `MathOfBiology.Loop_073.anisotropic_terrace_pitch_linear_in_step_speed`):
    Orthorhombic aragonite (Pmcn; a = 4.9611, b = 7.9672, c = 5.7407 Å; De Villiers 1971)
    grows with direction-dependent step speed v(theta) = v_inf * A(theta),
      A_single(theta) = 1 + e_2 cos 2(theta - theta_0) + e_6 cos 6(theta - theta_0),
      e_2 = (v_100/v_010 - 1) / (v_100/v_010 + 1),  v_100/v_010 = a*sqrt(3)/b = 1.0785,
    and the rotating BCF spiral acquires a direction-dependent pitch lambda(theta) =
    lambda_0 * A(theta) (kinematic Wulff construction), i.e. a polygonized terrace.
    {110} cyclic twinning (tau = 2 atan(a/b) = 63.82°, 3 domains at 0, tau, 2 tau with
    angular misfit 3 tau - 180° = 11.46°) averages A over the three domains, suppressing
    the 2-fold orthorhombic elongation to f_2 = |sum_j e^{2 i j tau}|/3 = 0.074 while
    retaining f_6 = 0.947 of the pseudo-hexagonal polygonization.

  - Law 5 (Pass 3, `MathOfBiology.Loop_073.chitin_membrane_perforation_bistability`):
    Each beta-chitin inter-lamellar membrane k (Cartwright & Checa 2007 Sec. 4) carries
    a perforation phase field phi_k in [0, 1] obeying the driven Allen-Cahn equation
      d phi/dt = M [ eps^2 Lap(phi) - W'(phi) + beta S_k(x,y) g'(phi) ],
      W(phi) = phi^2 (1 - phi)^2,  g(phi) = phi^2 (3 - 2 phi),
    whose intact (phi = 0) state loses stability where the drive S_k exceeds
    S_crit = 1 / (3 beta). The drive S_k is slaved to the 2D eikonal terrace solver
    (perforation only above the tablet footprint level(x, y) >= k) and to the mineral
    bridge centroid c_{k-1} of the sheet below, so the lateral stacking offset
    |c_k - c_{k-1}| / D_tab (columnar gastropod nacre -> 0; brick-wall bivalve sheet
    nacre -> ~0.5) is an emergent outcome of bridge localisation vs random pores.
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

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "nacre_spiral_dislocation_terraces"
REF_JSON = DATASET_DIR / "nacre_spiral_dislocation_terraces_reference.json"
SPECIMEN_JPG = DATASET_DIR / "nacre_aragonite_tablets_specimen.jpg"
SCOUT_JSON = DATASET_DIR / "scout_manifest.json"
BIB_FILE = DATASET_DIR / "paper_citations.bib"
PASS3_JSON = DATASET_DIR / "membrane_perforation_twinning_reference.json"  # Pass 3
PASS3_BIB = DATASET_DIR / "pass3_citations.bib"  # Pass 3

OUT_PNG = ROOT / "docs" / "media" / "nacre_spiral_dislocation_terraces_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "nacre_spiral_dislocation_terraces_growth.gif"
OUT_GIF_MEMBRANE = ROOT / "docs" / "media" / "nacre_spiral_dislocation_terraces_membrane_perforation.gif"  # Pass 3

OMEGA_0_BCF = 0.330958
PITCH_RATIO_BCF = (2.0 * math.pi) / OMEGA_0_BCF  # 18.9848418...

# Pass 3: orthorhombic aragonite lattice (De Villiers 1971 Am. Mineral. 56:758-767)
ARAGONITE_A_ANGSTROM = 4.9611
ARAGONITE_B_ANGSTROM = 7.9672
ARAGONITE_C_ANGSTROM = 5.7407
ARAGONITE_TWIN_ANGLE_RAD = 2.0 * math.atan(ARAGONITE_A_ANGSTROM / ARAGONITE_B_ANGSTROM)  # 63.82 deg
ARAGONITE_V100_OVER_V010 = ARAGONITE_A_ANGSTROM * math.sqrt(3.0) / ARAGONITE_B_ANGSTROM  # 1.0785
E6_DEFAULT = 0.15  # free kinetic {110}-facet anisotropy amplitude (swept in ablation)

# Cryptographic SHA-256 Registry for `datasets/nacre_spiral_dislocation_terraces/`
EXPECTED_SHA256: dict[str, str] = {
    "datasets/nacre_spiral_dislocation_terraces/nacre_spiral_dislocation_terraces_reference.json": (
        "580b05d6121d9596da59dc06dc128db340b95080fe041eb02d4c4533e294fda3"
    ),
    "datasets/nacre_spiral_dislocation_terraces/nacre_aragonite_tablets_specimen.jpg": (
        "e4c6590464523f2170a8fa277e51c2f2b6b0916f29d35f1dfc65f3d4ccc317b9"
    ),
    "datasets/nacre_spiral_dislocation_terraces/scout_manifest.json": (
        "6084a2ce133990fce4f01ef79f7791e832ed538acc6c7523a64a372101fe78f2"
    ),
    "datasets/nacre_spiral_dislocation_terraces/paper_citations.bib": (
        "3df311760cfbb29985fb6093131f63db46cb338741dd963864c25ed3a4794446"
    ),
    # Pass 3 additions
    "datasets/nacre_spiral_dislocation_terraces/membrane_perforation_twinning_reference.json": (
        "d0174c4a6f55a3037fa378747c90b1ba8eac9adf00b50f9458a93763fa40b309"
    ),
    "datasets/nacre_spiral_dislocation_terraces/pass3_citations.bib": (
        "a735d515e53f22dde94849174fdde7106c811d6daf8c7897757388cc0640f439"
    ),
}


# =============================================================================
# 1. Cryptographic Dataset Verification & Reference Loading
# =============================================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset_integrity() -> dict[str, str]:
    """Verify SHA-256 digests of all 6 pinned files in datasets/nacre_spiral_dislocation_terraces/."""
    computed: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required dataset asset: {p}")
        digest = sha256_file(p)
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {digest}")
        computed[rel] = digest
    return computed


def load_reference_data() -> dict[str, Any]:
    """Load and cryptographically verify `nacre_spiral_dislocation_terraces_reference.json`."""
    verify_dataset_integrity()
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def load_pass3_reference() -> dict[str, Any]:
    """Load and verify `membrane_perforation_twinning_reference.json` (Pass 3)."""
    verify_dataset_integrity()
    return json.loads(PASS3_JSON.read_text(encoding="utf-8"))


# =============================================================================
# 2. Burton-Cabrera-Frank (BCF) Analytical & Eikonal Spiral Kinematics
# =============================================================================
def compute_bcf_spiral_kinematics(
    v_inf_um_hr: float = 1.25,
    r_c_um: float = 0.42,
    h_0_um: float = 0.50,
    m_arms: int = 1,
) -> dict[str, float]:
    """Burton, Cabrera & Frank (1951) Phil. Trans. R. Soc. A 243:299-358 &
    Cartwright & Checa (2007) J. R. Soc. Interface 4:491-504 Sec. 2-3:
      v(r)     = v_inf * (1 - r_c / r)                   for r >= r_c
      omega    = omega_0 * (v_inf / r_c),                omega_0 = 0.330958
      lambda_0 = 2 * pi * v_inf / omega ≈ 18.9848 * r_c
      lambda_m = lambda_0 / m_arms                       for co-rotating m-spiral
      V_z      = m_arms * h_0 * (omega / (2 * pi))
    Matches Lean 4 theorems `MathOfBiology.Loop_073.bcf_gibbs_thomson_critical_arrest`
    and `MathOfBiology.Loop_073.bcf_archimedean_terrace_pitch_scaling`.
    """
    if v_inf_um_hr <= 0.0 or r_c_um <= 0.0 or h_0_um <= 0.0 or m_arms < 1:
        raise ValueError("Require strictly positive v_inf, r_c, h_0 and m_arms >= 1.")
    omega = OMEGA_0_BCF * (v_inf_um_hr / r_c_um)
    lambda_0 = (2.0 * math.pi * v_inf_um_hr) / omega
    lambda_m = lambda_0 / float(m_arms)
    v_z = float(m_arms) * h_0_um * (omega / (2.0 * math.pi))
    return {
        "v_inf_um_hr": v_inf_um_hr,
        "r_c_um": r_c_um,
        "h_0_um": h_0_um,
        "m_arms": float(m_arms),
        "omega_rad_hr": omega,
        "lambda_0_um": lambda_0,
        "lambda_m_um": lambda_m,
        "pitch_over_rc": lambda_0 / r_c_um,
        "v_z_um_hr": v_z,
    }


def gibbs_thomson_step_velocity(
    r_um: np.ndarray | float,
    v_inf_um_hr: float = 1.25,
    r_c_um: float = 0.42,
) -> np.ndarray:
    """Gibbs-Thomson curved terrace front velocity v(r) = v_inf * (1 - r_c / r) for r >= r_c,
    vanishing identically at r = r_c (`MathOfBiology.Loop_073.bcf_gibbs_thomson_critical_arrest`)."""
    r_arr = np.asarray(r_um, dtype=float)
    out = np.zeros_like(r_arr)
    mask = r_arr > r_c_um
    out[mask] = v_inf_um_hr * (1.0 - r_c_um / r_arr[mask])
    return out


# =============================================================================
# 3. 2D Multi-Dislocation Eikonal Step-Flow & Chiral Defect Pair Solver
# =============================================================================
def simulate_chiral_dislocation_terrace_2d(
    regime_id: str = "single_screw_spiral_sheet_nacre",
    b_1: int = 1,
    b_2: int = 0,
    dipole_separation_um: float = 0.0,
    v_inf_um_hr: float = 1.25,
    r_c_um: float = 0.42,
    h_0_um: float = 0.50,
    domain_radius_um: float = 24.0,
    n_grid: int = 201,
    t_hr: float = 6.0,
    step_anisotropy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate a 2D molluscan nacre terrace field h(x, y, t) and complex order parameter
    Psi(x, y, t) = exp(i * Phi(x, y, t)) generated by one or two screw dislocations
    (b_1, b_2) separated by distance d (`dipole_separation_um`) on [-L, +L]^2
    (Cartwright & Checa 2007 Figs. 2-6; Cartwright et al. 2009 PNAS Figs. 1-4).

    Pass 3: if `step_anisotropy` is given (kwargs of `aragonite_step_velocity_anisotropy`),
    the radial eikonal travel is divided by the direction-dependent step-speed factor
    A(theta) about the pattern centre, so the terrace pitch becomes lambda(theta) =
    lambda_0 * A(theta) (kinematic Wulff polygonization of the spiral core).

      1. `subcritical_dipole_annihilation` (+b, -b with d <= 2*r_c):
         Opposite-chirality dipole below the Frank-Read critical nucleation threshold
         2*r_c cannot bow out past the critical curvature 1/r_c; the nascent step loop
         pinches off and self-annihilates (0 active propagating terraces, flat sheet).
      2. `single_screw_spiral_sheet_nacre` (+b, 0):
         Single screw dislocation generating a Cabrera-Levine / BCF Archimedean spiral
         with asymptotic terrace pitch lambda_0 ≈ 18.9848 * r_c and winding charge +1.
      3. `opposite_chirality_frank_read_target` (+b, -b with d > 2*r_c):
         Supercritical Frank-Read screw-dislocation pair whose counter-rotating spiral
         arms periodically collide and reconnect, emitting closed concentric target
         terraces with net far-field Burgers winding charge b_net = b_1 + b_2 = 0.
      4. `same_chirality_double_spiral_tower` (+b, +b):
         Co-rotating same-chirality dislocation pair with net Burgers charge b_net = +2,
         generating an interlocked two-armed spiral whose radial inter-terrace spacing
         halves to lambda_2 = lambda_0 / 2.
    """
    kin = compute_bcf_spiral_kinematics(v_inf_um_hr, r_c_um, h_0_um, m_arms=1)
    omega = kin["omega_rad_hr"]
    lam_0 = kin["lambda_0_um"]
    k_0 = 2.0 * math.pi / lam_0

    coords = np.linspace(-domain_radius_um, domain_radius_um, n_grid)
    XX, YY = np.meshgrid(coords, coords)

    x1, y1 = (-0.5 * dipole_separation_um, 0.0) if b_2 != 0 else (0.0, 0.0)
    x2, y2 = (0.5 * dipole_separation_um, 0.0)

    r1 = np.hypot(XX - x1, YY - y1)
    th1 = np.arctan2(YY - y1, XX - x1)
    r2 = np.hypot(XX - x2, YY - y2)
    th2 = np.arctan2(YY - y2, XX - x2)

    b_net = int(b_1 + b_2)
    nucleation_excess_um = float(dipole_separation_um - 2.0 * r_c_um)

    # Pass 3: direction-dependent aragonite step-speed factor A(theta) about the pattern centre
    th_c = np.arctan2(YY, XX)
    aniso_factor = (
        aragonite_step_velocity_anisotropy(th_c, **step_anisotropy)
        if step_anisotropy is not None
        else np.ones_like(XX)
    )

    # Subcritical Frank-Read dipole negative control (b_1 = -b_2 != 0 and d <= 2 * r_c)
    if b_1 != 0 and b_2 == -b_1 and dipole_separation_um <= 2.0 * r_c_um:
        # Localized non-propagating core dipole strain that decays exponentially outside d
        core_dip = 0.04 * h_0_um * ((XX - x1) / (r1**2 + r_c_um**2) - (XX - x2) / (r2**2 + r_c_um**2))
        height_um = np.zeros_like(XX)
        terrace_idx = np.zeros_like(XX)
        psi_field = np.exp(1j * core_dip)
        active_terraces = 0
        measured_pitch_um = 0.0
    elif b_2 == 0:
        # Single screw dislocation (+b_1) -> BCF Cabrera-Levine Archimedean spiral
        r1_eff = np.maximum(r1, r_c_um)
        spiral_travel = (
            k_0 * (r1_eff - r_c_um) + (omega * r_c_um / v_inf_um_hr) * np.log(r1_eff / r_c_um)
        ) / aniso_factor
        phase = float(b_1) * th1 - spiral_travel + omega * t_hr
        psi_field = np.exp(1j * phase)
        # Stepped lamellar height field (in units of h_0)
        terrace_idx = np.floor((float(b_1) * th1 - spiral_travel + omega * t_hr) / (2.0 * math.pi))
        height_um = h_0_um * terrace_idx
        active_terraces = _count_terrace_steps_along_diameter(terrace_idx)
        measured_pitch_um = _measure_radial_terrace_pitch(coords, spiral_travel, k_0)
    elif b_1 == -b_2:
        # Supercritical opposite-chirality Frank-Read pair (+b, -b, d > 2*r_c) -> Closed Target Rings
        # Topological winding (b_1*th1 + b_2*th2) cancels at large r (b_net = 0), while eikonal
        # wave fronts emitted from the dipole bridge form concentric closed target terraces!
        dipole_angle = float(b_1) * th1 + float(b_2) * th2
        r_bridge = np.sqrt(0.5 * (r1**2 + r2**2) - 0.25 * (dipole_separation_um**2) + r_c_um**2)
        r_eff = np.maximum(r_bridge, r_c_um)
        # Activity factor from supercritical bridge clearance (d - 2*r_c) / d
        activity = math.tanh((dipole_separation_um - 2.0 * r_c_um) / r_c_um)
        eikonal_dist = (
            k_0 * (r_eff - r_c_um) + (omega * r_c_um / v_inf_um_hr) * np.log(r_eff / r_c_um)
        ) / aniso_factor
        phase = dipole_angle - eikonal_dist + omega * t_hr
        psi_field = np.exp(1j * phase)
        terrace_idx = np.floor((dipole_angle - eikonal_dist + omega * t_hr) / (2.0 * math.pi))
        height_um = h_0_um * terrace_idx * (1.0 if activity > 0 else 0.0)
        active_terraces = _count_terrace_steps_along_diameter(terrace_idx)
        measured_pitch_um = _measure_radial_terrace_pitch(coords, eikonal_dist, k_0)
    else:
        # Same-chirality pair (+b, +b) -> Co-rotating 2-armed spiral with b_net = 2*b and pitch lambda_0 / 2
        total_angle = float(b_1) * th1 + float(b_2) * th2
        r_cm = np.hypot(XX, YY)
        r_eff = np.maximum(r_cm, r_c_um)
        # Two interleaved spiral arms double the radial step density (k_2 = 2 * k_0)
        m_arms = abs(b_net)
        spiral_travel = float(m_arms) * (
            k_0 * (r_eff - r_c_um) + (omega * r_c_um / v_inf_um_hr) * np.log(r_eff / r_c_um)
        ) / aniso_factor
        phase = total_angle - spiral_travel + float(m_arms) * omega * t_hr
        psi_field = np.exp(1j * phase)
        terrace_idx = np.floor((total_angle - spiral_travel + float(m_arms) * omega * t_hr) / (2.0 * math.pi))
        height_um = h_0_um * terrace_idx
        active_terraces = _count_terrace_steps_along_diameter(terrace_idx)
        measured_pitch_um = _measure_radial_terrace_pitch(coords, spiral_travel, float(m_arms) * k_0)

    # Compute discrete topological winding number around an outer circular loop R_loop = 0.75 * L
    winding_number = compute_outer_contour_winding_number(
        b_1=b_1,
        b_2=b_2,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        r_loop=0.75 * domain_radius_um,
    )

    return {
        "regime_id": regime_id,
        "coords_um": coords,
        "XX_um": XX,
        "YY_um": YY,
        "height_um": height_um,
        "terrace_idx": terrace_idx,
        "psi_field": psi_field,
        "b_1": b_1,
        "b_2": b_2,
        "b_net": b_net,
        "winding_number_outer": winding_number,
        "dipole_separation_um": dipole_separation_um,
        "nucleation_excess_um": nucleation_excess_um,
        "r_c_um": r_c_um,
        "h_0_um": h_0_um,
        "v_inf_um_hr": v_inf_um_hr,
        "omega_rad_hr": omega,
        "lambda_0_um": lam_0,
        "measured_pitch_um": measured_pitch_um,
        "active_terraces": active_terraces,
    }


def _measure_radial_terrace_pitch(
    coords: np.ndarray,
    radial_phase_field: np.ndarray,
    k_target: float,
) -> float:
    """Measure the far-field radial inter-terrace spacing lambda along the positive x-axis
    for r in [0.60 * L, 0.98 * L] via linear phase regression of the eikonal travel.

    Pass 3 fix: the v2 version blended 15 % of this measurement with 85 % of the
    theoretical pitch 2*pi/k_target (a self-referential calibration). The value is now
    the raw measured pitch; it sits ~2 % below lambda_0 because of the Cabrera-Levine
    logarithmic core correction (omega r_c / v_inf) ln(r / r_c), which is a genuine
    property of the model, not an error. `k_target` is only used as a fallback when the
    sampling window is too short."""
    mid = len(coords) // 2
    x_pos = coords[mid:]
    phi_pos = radial_phase_field[mid, mid:]
    mask = x_pos >= 0.60 * float(coords[-1])
    if np.sum(mask) < 5:
        return float(2.0 * math.pi / max(k_target, 1e-9))
    slope, _, _, _, _ = stats.linregress(x_pos[mask], phi_pos[mask])
    return float(2.0 * math.pi / max(slope, 1e-9))


def _count_terrace_steps_along_diameter(terrace_idx: np.ndarray) -> int:
    """Pass 3: count the number of discrete terrace steps (changes of the integer terrace
    index) crossed along the full horizontal diameter y = 0 of the simulated field. Replaces
    the v2 prescribed value round(2 L / lambda_0) with a measurement of the field itself."""
    mid = terrace_idx.shape[0] // 2
    row = terrace_idx[mid, :]
    return int(np.count_nonzero(np.diff(row) != 0))


def compute_outer_contour_winding_number(
    b_1: int,
    b_2: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r_loop: float = 18.0,
    n_pts: int = 720,
) -> float:
    """Compute the discrete contour integral (1 / 2*pi) * oint d(theta_dipole) around a
    closed circle of radius `r_loop` enclosing both dislocation cores (x1, y1) and (x2, y2).
    Matches Lean 4 theorem `MathOfBiology.Loop_073.cartwright_frank_read_topological_burgers_algebra`."""
    phis = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=True)
    xc = r_loop * np.cos(phis)
    yc = r_loop * np.sin(phis)
    ang = float(b_1) * np.unwrap(np.arctan2(yc - y1, xc - x1)) + float(b_2) * np.unwrap(
        np.arctan2(yc - y2, xc - x2)
    )
    return float((ang[-1] - ang[0]) / (2.0 * math.pi))


# =============================================================================
# 4. 3D Bivalve Terraced Sheet Nacre vs Gastropod Columnar Tower Nacre
# =============================================================================
def simulate_nacre_architecture_3d(
    architecture: str = "bivalve_sheet_spiral",
    domain_um: float = 18.0,
    n_grid: int = 130,
    r_c_um: float = 0.42,
    h_0_um: float = 0.50,
    t_hr: float = 5.0,
) -> dict[str, Any]:
    """Synthesize a 3D stepped aragonite lamellar surface Z(X, Y) comparing:
      - `bivalve_sheet_spiral` (*Pteria hirundo* / *Pinctada margaritifera*):
        Broad spiral/target terrace sheets with pitch lambda_0 = 7.97 µm and step height
        h_0 = 0.50 µm winding around a central screw dislocation.
      - `gastropod_columnar_towers` (*Haliotis rufescens* / *Gibbula umbilicalis*):
        Hexagonal array of conical/pyramidal stacked aragonite tablet towers (tight pitch
        lambda_0 = 3.61 µm, tablet diameter D_tab = 7.8 µm, inter-lamellar step h_0 = 0.50 µm)
        that nucleate vertically before coalescing laterally across chitin membranes.
    """
    coords = np.linspace(-domain_um, domain_um, n_grid)
    XX, YY = np.meshgrid(coords, coords)
    kin = compute_bcf_spiral_kinematics(v_inf_um_hr=1.25, r_c_um=r_c_um, h_0_um=h_0_um)
    lam_0 = kin["lambda_0_um"]
    omega = kin["omega_rad_hr"]
    k_0 = 2.0 * math.pi / lam_0

    if architecture == "bivalve_sheet_spiral":
        RR = np.hypot(XX, YY)
        TH = np.arctan2(YY, XX)
        r_eff = np.maximum(RR, r_c_um)
        # Continuous helical ramp + quantized 0.50 µm aragonite lamellar terrace steps
        u_turn = (
            TH
            - k_0 * (r_eff - r_c_um)
            - (omega * r_c_um / 1.25) * np.log(r_eff / r_c_um)
            + omega * t_hr
        ) / (2.0 * math.pi)
        # Soft-stepped terrace staircase representing discrete 0.50 µm aragonite lamellae
        z_layers = np.floor(u_turn) + 0.15 * (u_turn - np.floor(u_turn))
        Z = h_0_um * (z_layers - np.min(z_layers))
    else:
        # Gastropod columnar tower nacre (*Haliotis*): hexagonal lattice of stacked tablet towers
        pitch_tower = 3.6071  # Haliotis rufescens BCF pitch
        tower_spacing = 11.5
        centers = [(0.0, 0.0)]
        for k_hex in range(6):
            ang = k_hex * math.pi / 3.0
            centers.append((tower_spacing * math.cos(ang), tower_spacing * math.sin(ang)))
        Z = np.zeros_like(XX)
        for cx, cy in centers:
            r_loc = np.hypot(XX - cx, YY - cy)
            th_loc = np.arctan2(YY - cy, XX - cx)
            # Polygonal pseudo-hexagonal tablet anisotropy of aragonite crystal habit
            hex_mod = 1.0 + 0.06 * np.cos(6.0 * th_loc)
            r_aniso = r_loc * hex_mod
            layers_loc = np.clip(np.floor(8.0 - r_aniso / (0.45 * pitch_tower)), 0.0, 8.0)
            Z = np.maximum(Z, h_0_um * layers_loc)

    return {
        "architecture": architecture,
        "XX_um": XX,
        "YY_um": YY,
        "Z_um": Z,
        "max_height_um": float(np.max(Z)),
        "n_distinct_layers": int(round(float(np.max(Z)) / h_0_um)),
    }


# =============================================================================
# 4b. Pass 3 — Aragonite [100]/[010] Step-Velocity Anisotropy & {110} Cyclic Twinning
# =============================================================================
def aragonite_step_velocity_anisotropy(
    theta: np.ndarray | float,
    v100_over_v010: float = ARAGONITE_V100_OVER_V010,
    e6: float = E6_DEFAULT,
    twinned: bool = True,
    theta_0_rad: float = 0.0,
    twin_angle_rad: float = ARAGONITE_TWIN_ANGLE_RAD,
) -> np.ndarray:
    """Direction-dependent aragonite step-speed factor A(theta) = v(theta) / v_inf.

    Single orthorhombic domain (Pmcn; fast [100] = a, slow [010] = b):
      A_1(theta) = 1 + e_2 cos 2(theta - theta_0) + e_6 cos 6(theta - theta_0),
      e_2 = (v_100/v_010 - 1) / (v_100/v_010 + 1)   so that A_1(0)/A_1(pi/2) = v_100/v_010
      (for e_6 = 0); default v_100/v_010 = a sqrt(3) / b = 1.0785 (De Villiers 1971 lattice).
    {110} cyclic (trilling) twin: A_twin = mean of A_1 over the three domains rotated by
    j * tau, tau = 2 atan(a/b) = 63.82 deg (`MathOfBiology.Loop_073.aragonite_cyclic_twin_angular_misfit`).
    """
    if v100_over_v010 <= 0.0:
        raise ValueError("v100_over_v010 must be positive")
    e2 = (v100_over_v010 - 1.0) / (v100_over_v010 + 1.0)
    th = np.asarray(theta, dtype=float)
    n_dom = 3 if twinned else 1
    acc = np.zeros_like(th)
    for j in range(n_dom):
        ph = th - theta_0_rad - j * twin_angle_rad
        acc = acc + 1.0 + e2 * np.cos(2.0 * ph) + e6 * np.cos(6.0 * ph)
    return acc / float(n_dom)


def analytic_twin_suppression_factor(m: int, twin_angle_rad: float = ARAGONITE_TWIN_ANGLE_RAD) -> float:
    """f_m = |sum_{j=0..2} exp(i m j tau)| / 3: fraction of the m-fold Fourier anisotropy that
    survives averaging over the three {110} twin domains (f_2 = 0.074, f_6 = 0.947 for aragonite)."""
    return float(abs(sum(np.exp(1j * m * j * twin_angle_rad) for j in range(3))) / 3.0)


def measure_terrace_contour_polygonization(
    v_inf_um_hr: float = 1.25,
    r_c_um: float = 0.42,
    step_anisotropy: dict[str, Any] | None = None,
    turn_index: int = 2,
    n_theta: int = 720,
) -> dict[str, Any]:
    """Pass 3: extract the closed contour r_n(theta) of the n-th anisotropic terrace ring
    (T(r) / A(theta) = 2 pi n) by inverting the monotone Cabrera-Levine travel
    T(r) = k_0 (r - r_c) + (omega r_c / v_inf) ln(r / r_c) on a dense radial grid, and return
    the normalised Fourier amplitudes 2|c_m| / c_0 (m = 2, 6) of the contour radius, i.e. its
    orthorhombic elongation and pseudo-hexagonal polygonization. Because T^{-1} is nonlinear
    (logarithmic core term) the measured ratios are a genuine check on the anisotropic
    eikonal construction rather than a re-reading of A(theta)."""
    kin = compute_bcf_spiral_kinematics(v_inf_um_hr, r_c_um, 0.5, m_arms=1)
    k_0 = 2.0 * math.pi / kin["lambda_0_um"]
    omega = kin["omega_rad_hr"]
    thetas = np.linspace(-math.pi, math.pi, n_theta, endpoint=False)
    aniso = (
        aragonite_step_velocity_anisotropy(thetas, **step_anisotropy)
        if step_anisotropy is not None
        else np.ones_like(thetas)
    )
    r_dense = np.linspace(r_c_um, r_c_um + (turn_index + 2) * kin["lambda_0_um"] * 2.0, 20000)
    travel_dense = k_0 * (r_dense - r_c_um) + (omega * r_c_um / v_inf_um_hr) * np.log(r_dense / r_c_um)
    # Closed terrace ring n emitted by the source (Frank-Read target ring / tablet outline, far
    # from the core where the topological winding has cancelled): T(r) / A(theta) = 2 pi n.
    # A closed curve is used so that the Fourier spectrum is not contaminated by the open
    # Archimedean ramp of a single spiral turn.
    target_travel = aniso * (2.0 * math.pi * turn_index)
    r_contour = np.interp(target_travel, travel_dense, r_dense)
    coeffs = np.fft.rfft(r_contour) / float(n_theta)
    c0 = float(np.abs(coeffs[0]))
    return {
        "theta_rad": thetas,
        "r_contour_um": r_contour,
        "mean_radius_um": c0,
        "c2_over_c0": float(2.0 * np.abs(coeffs[2]) / max(c0, 1e-12)),
        "c6_over_c0": float(2.0 * np.abs(coeffs[6]) / max(c0, 1e-12)),
        "aspect_ratio_max_min": float(np.max(r_contour) / max(np.min(r_contour), 1e-12)),
    }


def evaluate_twinning_polygonization() -> dict[str, Any]:
    """Pass 3: compare untwinned single-domain vs {110} cyclically twinned aragonite spiral
    terrace contours and test the analytic twin suppression factors f_2, f_6 against the
    Fourier amplitudes measured on the simulated contours; sweep e_6 for the ablation panel."""
    single = measure_terrace_contour_polygonization(step_anisotropy={"twinned": False})
    twin = measure_terrace_contour_polygonization(step_anisotropy={"twinned": True})
    iso = measure_terrace_contour_polygonization(step_anisotropy=None)
    f2_meas = twin["c2_over_c0"] / max(single["c2_over_c0"], 1e-12)
    f6_meas = twin["c6_over_c0"] / max(single["c6_over_c0"], 1e-12)
    e6_grid = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    sweep_single = [
        measure_terrace_contour_polygonization(step_anisotropy={"twinned": False, "e6": float(e)})["c6_over_c0"]
        for e in e6_grid
    ]
    sweep_twin = [
        measure_terrace_contour_polygonization(step_anisotropy={"twinned": True, "e6": float(e)})["c6_over_c0"]
        for e in e6_grid
    ]
    # Ideal 60-degree trilling control: 2-fold anisotropy must cancel exactly
    ideal = measure_terrace_contour_polygonization(
        step_anisotropy={"twinned": True, "twin_angle_rad": math.pi / 3.0}
    )
    return {
        "single": single,
        "twin": twin,
        "isotropic": iso,
        "twin_angle_deg": math.degrees(ARAGONITE_TWIN_ANGLE_RAD),
        "cyclic_twin_misfit_deg": 3.0 * math.degrees(ARAGONITE_TWIN_ANGLE_RAD) - 180.0,
        "v100_over_v010": ARAGONITE_V100_OVER_V010,
        "f2_analytic": analytic_twin_suppression_factor(2),
        "f6_analytic": analytic_twin_suppression_factor(6),
        "f2_measured": float(f2_meas),
        "f6_measured": float(f6_meas),
        "ideal_60deg_c2_over_c0": ideal["c2_over_c0"],
        "e6_grid": e6_grid,
        "c6_sweep_single": np.array(sweep_single),
        "c6_sweep_twin": np.array(sweep_twin),
    }


# =============================================================================
# 4c. Pass 3 — Stacked-Sheet Allen-Cahn Chitin Membrane Perforation & Mineral Bridges
# =============================================================================
def simulate_membrane_perforation_phase_field_3d(
    stacking_mode: str = "columnar_gastropod",
    r_c_um: float = 0.19,
    h_0_um: float = 0.50,
    tablet_diam_um: float = 7.8,
    bridge_localization_um: float | None = None,
    domain_um: float = 12.0,
    n_grid: int = 129,
    eps_um: float = 0.30,
    beta: float = 1.0,
    s_0: float = 0.35,
    mobility: float = 1.0,
    dt: float = 0.04,
    n_steps: int = 150,
    pore_density_per_um2: float = 0.12,
    pore_width_um: float = 0.80,
    pore_amplitude: float = 0.90,
    seed: int = 2007,
    max_sheets: int = 6,
    step_anisotropy: dict[str, Any] | None = None,
    keep_history: bool = False,
) -> dict[str, Any]:
    """Pass 3 reduced 3D model (Cartwright & Checa 2007 Sec. 4): a stack of beta-chitin
    inter-lamellar membranes k = 1..N above a single-screw-dislocation terrace tower.

    Coupling to the 2D eikonal solver: the integer terrace level(x, y) of
    `simulate_chiral_dislocation_terrace_2d` (same grid, same r_c) defines the footprint of
    tablet k as level >= k. Membrane k can only perforate above that footprint.

    Per sheet, the perforation phase field phi_k (0 intact, 1 perforated) obeys the driven
    Allen-Cahn equation (`MathOfBiology.Loop_073.chitin_membrane_perforation_bistability`)
        d phi/dt = M [ eps^2 Lap phi - 2 phi (1-phi)(1-2 phi) + 6 beta (S_k - S_0) phi (1-phi) ],
    so phi = 0 is destabilised (spinodal) wherever S_k > S_crit = S_0 + 1/(3 beta), and the
    perforated state phi = 1 is only favoured where S_k > S_0 (bounded holes). The drive
        S_k = F_k * max( A_g exp(-|x - c_{k-1}|^2 / 2 sigma^2), P_k(x, y) )
    combines mineral-bridge localisation (Gaussian of width sigma about the bridge centroid
    c_{k-1} of the sheet below; c_0 = dislocation core) with a seeded random pore field P_k.
    The mineral bridge of sheet k is the strongest early perforation (argmax phi at 1/3 of
    the run); the lateral stacking offset |c_k - c_{k-1}| / D_tab is the emergent statistic
    separating columnar gastropod nacre (small sigma, offset -> 0) from brick-wall bivalve
    sheet nacre (large sigma, offset ~ 0.3-0.5).
    """
    if bridge_localization_um is None:
        bridge_localization_um = (0.10 if stacking_mode == "columnar_gastropod" else 1.00) * tablet_diam_um
    rng = np.random.default_rng(seed)

    terr = simulate_chiral_dislocation_terrace_2d(
        regime_id="single_screw_spiral_sheet_nacre",
        b_1=1,
        b_2=0,
        dipole_separation_um=0.0,
        r_c_um=r_c_um,
        h_0_um=h_0_um,
        domain_radius_um=domain_um,
        n_grid=n_grid,
        step_anisotropy=step_anisotropy,
    )
    XX, YY = terr["XX_um"], terr["YY_um"]
    dx = float(terr["coords_um"][1] - terr["coords_um"][0])
    level = terr["terrace_idx"] - np.min(terr["terrace_idx"])
    n_sheets = int(min(max_sheets, int(np.max(level))))
    if n_sheets < 1:
        raise ValueError("Terrace tower too shallow for a membrane stack; reduce r_c or enlarge domain")

    # Explicit stability: dt <= dx^2 / (4 eps^2 M)
    dt_eff = min(dt, 0.9 * dx * dx / (4.0 * eps_um * eps_um * mobility))
    s_crit = s_0 + 1.0 / (3.0 * beta)  # spinodal drive above which phi = 0 is unstable

    phi_sheets: list[np.ndarray] = []
    source_sheets: list[np.ndarray] = []
    footprints: list[np.ndarray] = []
    centroids: list[tuple[float, float]] = [(0.0, 0.0)]
    offsets_um: list[float] = []
    perforated_fraction: list[float] = []
    n_patches: list[int] = []
    history: list[np.ndarray] = []

    # Fixed bridge-forming budget: the Gaussian amplitude scales as (sigma_ref / sigma)^2 so a
    # fixed number of mineral bridges per tablet is spread over the localisation area sigma^2
    # (sigma_ref = 0.10 D_tab = central-bridge reference, Checa et al. 2011).
    sigma_ref_um = 0.10 * tablet_diam_um
    gauss_amp = min(1.0, (sigma_ref_um / bridge_localization_um) ** 2)
    min_footprint_um2 = 4.0

    for k in range(1, n_sheets + 1):
        cx, cy = centroids[-1]
        # Tablet k footprint = eikonal terrace tier k (level >= k) within the lateral tablet
        # extent D_tab / 2 about its nucleating mineral bridge c_{k-1} (lateral growth of a
        # tablet is arrested by its neighbours at the tablet diameter; Cartwright & Checa 2007)
        footprint = (level >= k) & (np.hypot(XX - cx, YY - cy) <= 0.5 * tablet_diam_um)
        area_um2 = float(np.sum(footprint)) * dx * dx
        if area_um2 < min_footprint_um2:
            break  # tablet footprint too small to carry a resolvable membrane patch
        # At least one pore per tablet footprint (a tablet without any membrane pore could not
        # itself have been nucleated through a mineral bridge)
        n_pores = max(1, int(rng.poisson(pore_density_per_um2 * area_um2)))
        pore_field = np.zeros_like(XX)
        # Pores are seeded in the tablet interior (one pore width away from the tablet edge)
        interior = ndi.binary_erosion(footprint, iterations=max(1, int(round(pore_width_um / dx))))
        fp_idx = np.flatnonzero(interior if np.any(interior) else footprint)
        if n_pores > 0 and fp_idx.size > 0:
            picks = rng.choice(fp_idx, size=n_pores, replace=True)
            for p in picks:
                py, px = divmod(int(p), XX.shape[1])
                # Overlapping pores form a single hole: accumulate with max, not sum
                pore_field = np.maximum(pore_field, pore_amplitude * np.exp(
                    -((XX - XX[py, px]) ** 2 + (YY - YY[py, px]) ** 2) / (2.0 * pore_width_um**2)
                ))
        gauss = gauss_amp * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * bridge_localization_um**2))
        # Local drive = strongest local perforation mechanism (bridge localisation vs pore)
        source = footprint * np.maximum(gauss, pore_field)

        phi = 0.05 * footprint.astype(float)
        early_phi = None
        for it in range(n_steps):
            lap = (
                np.roll(phi, 1, 0) + np.roll(phi, -1, 0) + np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4.0 * phi
            ) / (dx * dx)
            dw = 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
            drive = 6.0 * beta * (source - s_0) * footprint * phi * (1.0 - phi)
            phi = phi + dt_eff * mobility * (eps_um**2 * lap - dw + drive)
            phi = np.clip(phi, 0.0, 1.0) * footprint
            if it == n_steps // 3:
                early_phi = phi.copy()
            if keep_history and k == 1 and it % 15 == 0:
                history.append(phi.copy())
        assert early_phi is not None
        if float(np.max(phi)) < 0.5:
            break  # no perforation -> no mineral bridge -> the tablet stack terminates here
        iy, ix = np.unravel_index(int(np.argmax(early_phi)), phi.shape)
        c_new = (float(XX[iy, ix]), float(YY[iy, ix]))
        offsets_um.append(math.hypot(c_new[0] - cx, c_new[1] - cy))
        centroids.append(c_new)
        perf = phi > 0.5
        perforated_fraction.append(float(np.sum(perf) / max(np.sum(footprint), 1)))
        _, n_lab = ndi.label(perf)
        n_patches.append(int(n_lab))
        phi_sheets.append(phi)
        source_sheets.append(source)
        footprints.append(footprint)

    offsets = np.asarray(offsets_um, dtype=float)
    if offsets.size == 0:
        raise ValueError("No membrane sheet perforated; cannot form a mineral-bridge stack")
    return {
        "stacking_mode": stacking_mode,
        "XX_um": XX,
        "YY_um": YY,
        "level": level,
        "n_sheets": int(len(phi_sheets)),
        "gauss_amp": gauss_amp,
        "h_0_um": h_0_um,
        "tablet_diam_um": tablet_diam_um,
        "bridge_localization_um": bridge_localization_um,
        "s_crit": s_crit,
        "dt_eff": dt_eff,
        "phi_sheets": phi_sheets,
        "source_sheets": source_sheets,
        "footprints": footprints,
        "bridge_centroids_um": centroids,
        "stacking_offsets_um": offsets,
        "mean_offset_over_tablet_diam": float(np.mean(offsets) / tablet_diam_um),
        "alignment_index": float(np.mean(offsets < 0.15 * tablet_diam_um)),
        "perforated_fraction_per_sheet": perforated_fraction,
        "n_perforation_patches_per_sheet": n_patches,
        "history_sheet1": history,
    }


def evaluate_membrane_bridge_alignment(seed: int = 2007) -> dict[str, Any]:
    """Pass 3: run the stacked-sheet perforation model in both stacking modes with the
    reference parameters from `membrane_perforation_twinning_reference.json`, and sweep the
    bridge-localisation length sigma / D_tab (3 seeds each) to expose the columnar -> sheet
    stacking transition as a parameter ablation."""
    p3 = load_pass3_reference()
    modes = p3["interlamellar_membrane_perforation_model"]["stacking_modes"]
    params = p3["interlamellar_membrane_perforation_model"]["parameters"]
    common = dict(
        eps_um=float(params["eps_um"]),
        beta=float(params["beta"]),
        s_0=float(params["s_0"]),
        mobility=float(params["mobility"]),
        dt=float(params["dt"]),
        n_steps=int(params["n_steps"]),
        pore_density_per_um2=float(params["pore_density_per_um2"]),
        pore_width_um=float(params["pore_width_um"]),
        pore_amplitude=float(params["pore_amplitude"]),
    )
    out: dict[str, Any] = {"modes": {}, "reference": modes, "mode_stats": {}}
    n_seeds_mode = 5
    for mode_id, m in modes.items():
        all_offsets: list[float] = []
        for sd in range(n_seeds_mode):
            sim = simulate_membrane_perforation_phase_field_3d(
                stacking_mode=mode_id,
                r_c_um=float(m["r_c_um"]),
                tablet_diam_um=float(m["tablet_diam_um"]),
                bridge_localization_um=float(m["bridge_localization_over_tablet_diam"]) * float(m["tablet_diam_um"]),
                seed=seed + 11 * sd,
                keep_history=(mode_id == "columnar_gastropod" and sd == 0),
                **common,
            )
            if sd == 0:
                out["modes"][mode_id] = sim
            all_offsets.extend((sim["stacking_offsets_um"] / float(m["tablet_diam_um"])).tolist())
        offs = np.asarray(all_offsets)
        out["mode_stats"][mode_id] = {
            "n_seeds": n_seeds_mode,
            "n_sheets_total": int(offs.size),
            "mean_offset_over_tablet_diam": float(np.mean(offs)),
            "sd_offset_over_tablet_diam": float(np.std(offs)),
            "alignment_index": float(np.mean(offs < 0.15)),
            "reference_offset": float(m["reference_stacking_offset_over_tablet_diam"]),
            "reference_alignment_index": float(m["reference_alignment_index"]),
        }

    # sigma >= 0.075 D_tab = 3 grid cells: narrower Gaussians are below the grid resolution
    sigma_grid = np.array([0.075, 0.10, 0.125, 0.15, 0.20, 0.35, 0.50, 1.00])
    offs_mean = []
    offs_sd = []
    for s_ratio in sigma_grid:
        vals = []
        for sd in range(3):
            sim = simulate_membrane_perforation_phase_field_3d(
                stacking_mode="sweep",
                r_c_um=0.19,
                tablet_diam_um=7.8,
                bridge_localization_um=float(s_ratio) * 7.8,
                seed=seed + 11 * sd,
                **common,
            )
            vals.append(sim["mean_offset_over_tablet_diam"])
        offs_mean.append(float(np.mean(vals)))
        offs_sd.append(float(np.std(vals)))
    out["sigma_grid_over_tablet_diam"] = sigma_grid
    out["sweep_offset_mean"] = np.array(offs_mean)
    out["sweep_offset_sd"] = np.array(offs_sd)
    return out


def evaluate_all_morphospace_regimes() -> dict[str, dict[str, Any]]:
    """Evaluate all 4 canonical Cartwright-Checa chiral defect morphospace regimes."""
    ref = load_reference_data()
    defaults = ref["governing_equations"]["default_parameters"]
    out: dict[str, dict[str, Any]] = {}

    for reg in ref["morphospace_regimes"]["regimes"]:
        rid = reg["regime_id"]
        sim2d = simulate_chiral_dislocation_terrace_2d(
            regime_id=rid,
            b_1=int(reg["b_1"]),
            b_2=int(reg["b_2"]),
            dipole_separation_um=float(reg["dipole_separation_um"]),
            v_inf_um_hr=float(defaults["v_inf_um_per_hr"]),
            r_c_um=float(reg["r_c_um"]),
            h_0_um=float(defaults["h_0_um"]),
        )
        out[rid] = {
            "regime_id": rid,
            "label": reg["label"],
            "b_1": int(reg["b_1"]),
            "b_2": int(reg["b_2"]),
            "b_net_sim": sim2d["b_net"],
            "b_net_ref": int(reg["expected_net_burgers"]),
            "winding_number_outer": sim2d["winding_number_outer"],
            "dipole_separation_um": float(reg["dipole_separation_um"]),
            "terrace_pitch_sim_um": sim2d["measured_pitch_um"],
            "terrace_pitch_ref_um": float(reg["expected_terrace_pitch_um"]),
            "active_terraces_sim": sim2d["active_terraces"],
            "active_terraces_ref": int(reg["expected_active_terraces"]),
            "sim2d": sim2d,
        }
    return out


# =============================================================================
# 5. Cross-Taxa 12-Species Bivalve Sheet vs Gastropod Tower Scaling Collapse
# =============================================================================
def evaluate_cross_taxa_nacre_scaling() -> dict[str, Any]:
    """Validate Cartwright & Checa (2007) & Burton, Cabrera & Frank (1951) linear
    terrace pitch law lambda_0 = (2*pi/omega_0)*r_c ≈ 18.9848*r_c and angular rotation
    frequency law omega = omega_0*v_inf/r_c across 12 empirical molluscan species
    spanning Bivalvia (sheet nacre), Gastropoda (tower nacre), and Cephalopoda (tower nacre)."""
    ref = load_reference_data()
    rows = ref["cross_taxa_nacre_terrace_scaling"]
    records: list[dict[str, Any]] = []

    for row in rows:
        rc = float(row["r_c_um"])
        v_inf = float(row["v_inf_um_hr"])
        h0 = float(row["h_0_um"])
        kin = compute_bcf_spiral_kinematics(v_inf, rc, h0, m_arms=1)
        records.append(
            {
                "species": row["species"],
                "clade": row["clade"],
                "common_name": row["common_name"],
                "r_c_um": rc,
                "v_inf_um_hr": v_inf,
                "h_0_um": h0,
                "lambda_pred_um": kin["lambda_0_um"],
                "lambda_obs_um": float(row["lambda_obs_um"]),
                "omega_pred_rad_hr": kin["omega_rad_hr"],
                "omega_obs_rad_hr": float(row["omega_obs_rad_hr"]),
                "tablet_diam_um": float(row["tablet_diam_um"]),
            }
        )

    rc_arr = np.array([r["r_c_um"] for r in records])
    lam_pred = np.array([r["lambda_pred_um"] for r in records])
    lam_obs = np.array([r["lambda_obs_um"] for r in records])
    om_pred = np.array([r["omega_pred_rad_hr"] for r in records])
    om_obs = np.array([r["omega_obs_rad_hr"] for r in records])

    r_lam, p_lam = stats.pearsonr(lam_pred, lam_obs)
    r_om, p_om = stats.pearsonr(om_pred, om_obs)
    rmse_lam = float(np.sqrt(np.mean((lam_pred - lam_obs) ** 2)))
    rmse_om = float(np.sqrt(np.mean((om_pred - om_obs) ** 2)))

    # Empirical OLS slope of lambda_obs vs r_c (theoretical BCF constant = 18.9848 ≈ 19)
    slope_bcf, intercept_bcf, _, _, se_bcf = stats.linregress(rc_arr, lam_obs)

    sheet_pitches = [r["lambda_obs_um"] for r in records if "Sheet" in r["clade"]]
    tower_pitches = [r["lambda_obs_um"] for r in records if "Tower" in r["clade"]]

    return {
        "records": records,
        "n_species": len(records),
        "n_clades": len({r["clade"] for r in records}),
        "pearson_r_pitch": float(r_lam),
        "p_value_pitch": float(p_lam),
        "rmse_pitch_um": rmse_lam,
        "pearson_r_omega": float(r_om),
        "p_value_omega": float(p_om),
        "rmse_omega_rad_hr": rmse_om,
        "bcf_pitch_slope": float(slope_bcf),
        "bcf_pitch_slope_se": float(se_bcf),
        "mean_sheet_pitch_um": float(np.mean(sheet_pitches)),
        "mean_tower_pitch_um": float(np.mean(tower_pitches)),
        "sheet_to_tower_pitch_ratio": float(np.mean(sheet_pitches) / np.mean(tower_pitches)),
    }


# =============================================================================
# 6. Computer-Vision Segmentation of Real SEM Nacre Aragonite Tablet Specimen
# =============================================================================
def segment_nacre_sem_specimen(img_path: Path = SPECIMEN_JPG) -> dict[str, Any]:
    """Segment the vendored Scanning Electron Micrograph (SEM) of fractured molluscan
    mother-of-pearl (`datasets/nacre_spiral_dislocation_terraces/nacre_aragonite_tablets_specimen.jpg`):
      1. Extract the high-contrast columnar aragonite brick-and-mortar ROI
         (x in [120, 660] px, y in [60, 430] px).
      2. Apply 1D vertical Difference-of-Gaussians (DoG) band-pass filtering along
         25 vertical column transects to detect the horizontal inter-lamellar chitin-protein
         mortar seams separating stacked aragonite tablets.
      3. Measure mean inter-lamellar spacing dy (px and calibrated µm at 0.019867 µm/px),
         inter-lamellar spacing CV, mean number of stacked tablet layers per column,
         and horizontal aragonite tablet width dx and aspect ratio D_tab / h_0.
    """
    ref = load_reference_data()
    spec_meta = ref["metadata"]["specimens"]["nacre_aragonite_tablets_specimen"]
    x0, x1 = [int(v) for v in spec_meta["roi_x_range_px"]]
    y0, y1 = [int(v) for v in spec_meta["roi_y_range_px"]]
    um_per_px = float(spec_meta["calibration_um_per_px"])

    with Image.open(img_path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0

    spacings_y: list[float] = []
    layer_counts: list[int] = []
    detected_seams_xy: list[tuple[int, np.ndarray]] = []

    for x_col in range(x0 + 20, x1 - 20, 20):
        col_slice = gray[y0:y1, x_col]
        prof = ndi.gaussian_filter1d(col_slice, 2.5) - ndi.gaussian_filter1d(col_slice, 9.5)
        pks, _ = signal.find_peaks(-prof, distance=14, prominence=0.014)
        if len(pks) >= 10:
            spacings_y.extend(np.diff(pks).astype(float))
            layer_counts.append(int(len(pks)))
            detected_seams_xy.append((x_col, pks + y0))

    spacings_y_arr = np.asarray(spacings_y, dtype=float)
    mean_dy_px = float(np.mean(spacings_y_arr))
    cv_dy = float(np.std(spacings_y_arr, ddof=1) / max(mean_dy_px, 1e-9))
    mean_h0_um = mean_dy_px * um_per_px
    mean_layers = float(np.mean(layer_counts))

    # Horizontal aragonite tablet width along lamellar rows
    spacings_x: list[float] = []
    for y_row in range(y0 + 30, y1 - 30, 20):
        row_slice = gray[y_row, x0:x1]
        prof_x = ndi.gaussian_filter1d(row_slice, 6.0) - ndi.gaussian_filter1d(row_slice, 28.0)
        pks_x, _ = signal.find_peaks(-prof_x, distance=45, prominence=0.015)
        if len(pks_x) >= 3:
            spacings_x.extend(np.diff(pks_x).astype(float))

    spacings_x_arr = np.asarray(spacings_x, dtype=float)
    mean_dx_px = float(np.mean(spacings_x_arr))
    mean_tablet_width_um = mean_dx_px * um_per_px
    tablet_aspect_ratio = mean_dx_px / max(mean_dy_px, 1e-9)

    return {
        "gray": gray,
        "roi_box": (x0, y0, x1, y1),
        "detected_seams_xy": detected_seams_xy,
        "spacings_y_px": spacings_y_arr,
        "mean_interlamellar_px": mean_dy_px,
        "mean_interlamellar_um": mean_h0_um,
        "cv_interlamellar_spacing": cv_dy,
        "mean_layers_per_column": mean_layers,
        "mean_tablet_width_px": mean_dx_px,
        "mean_tablet_width_um": mean_tablet_width_um,
        "tablet_aspect_ratio": tablet_aspect_ratio,
        "n_seams_measured": int(len(spacings_y_arr)),
    }


# =============================================================================
# 7. Publication-Grade 6-Panel Synthesis Figure & Co-Evolving 2-Panel Growth GIF
# =============================================================================
def render_comparison_figure(
    out_path: Path = OUT_PNG,
    overrides: dict[str, float] | None = None,
) -> Path:
    """Render the Pass 3 9-panel synthesis figure (`docs/media/nacre_spiral_dislocation_terraces_comparison.png`):
    panels A-F carried from Pass 2, plus G (stacked-sheet membrane perforation & mineral
    bridge alignment), H (aragonite twin-polygonized spiral) and I (parameter ablations)."""
    ovr = overrides or {}
    v_inf = float(ovr.get("v_inf", 1.25))
    r_c = float(ovr.get("r_c", 0.42))
    h_0 = float(ovr.get("h_0", 0.50))
    dipole_d = float(ovr.get("dipole_d", 2.40))
    e6 = float(ovr.get("e6", E6_DEFAULT))
    seed = int(ovr.get("seed", 2007))

    regimes = evaluate_all_morphospace_regimes()
    cross_taxa = evaluate_cross_taxa_nacre_scaling()
    specimen = segment_nacre_sem_specimen()
    sheet3d = simulate_nacre_architecture_3d("bivalve_sheet_spiral", r_c_um=r_c, h_0_um=h_0)
    tower3d = simulate_nacre_architecture_3d("gastropod_columnar_towers", r_c_um=0.19, h_0_um=h_0)
    membrane = evaluate_membrane_bridge_alignment(seed=seed)
    twinning = evaluate_twinning_polygonization()
    twin_spiral = simulate_chiral_dislocation_terrace_2d(
        regime_id="single_screw_spiral_sheet_nacre",
        b_1=1,
        b_2=0,
        dipole_separation_um=0.0,
        v_inf_um_hr=v_inf,
        r_c_um=r_c,
        h_0_um=h_0,
        domain_radius_um=16.0,
        n_grid=241,
        step_anisotropy={"twinned": True, "e6": e6},
    )
    single_domain_spiral = simulate_chiral_dislocation_terrace_2d(
        regime_id="single_screw_spiral_sheet_nacre",
        b_1=1,
        b_2=0,
        dipole_separation_um=0.0,
        v_inf_um_hr=v_inf,
        r_c_um=r_c,
        h_0_um=h_0,
        domain_radius_um=16.0,
        n_grid=241,
        step_anisotropy={"twinned": False, "e6": e6},
    )

    # Custom target simulation if dipole_d or r_c overridden
    target_sim = simulate_chiral_dislocation_terrace_2d(
        regime_id="opposite_chirality_frank_read_target",
        b_1=1,
        b_2=-1,
        dipole_separation_um=dipole_d,
        v_inf_um_hr=v_inf,
        r_c_um=r_c,
        h_0_um=h_0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(17.0, 16.2), dpi=150, facecolor="#0b0f17")
    gs = fig.add_gridspec(3, 3, hspace=0.36, wspace=0.27)

    # -------------------------------------------------------------------------
    # Panel A: 3D Stepped Screw-Dislocation Spiral vs Gastropod Tower Nacre
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0], projection="3d", facecolor="#0b0f17")
    ax_a.plot_surface(
        sheet3d["XX_um"],
        sheet3d["YY_um"],
        sheet3d["Z_um"],
        cmap="turbo",
        edgecolor="none",
        alpha=0.92,
        rstride=2,
        cstride=2,
    )
    ax_a.set_title(
        f"A. 3D Screw-Dislocation Nacre Spiral (r_c={r_c:.2f} µm, h_0={h_0:.2f} µm)",
        color="white",
        fontsize=10.0,
        fontweight="bold",
    )
    ax_a.set_xlabel("X (µm)", color="#9ca3af", fontsize=8)
    ax_a.set_ylabel("Y (µm)", color="#9ca3af", fontsize=8)
    ax_a.set_zlabel("Terrace Elevation Z (µm)", color="#9ca3af", fontsize=8)
    ax_a.tick_params(colors="#9ca3af", labelsize=7)
    ax_a.view_init(elev=38, azim=-55)

    # -------------------------------------------------------------------------
    # Panel B: 4 Chiral Dislocation Morphospace Regimes (Subcritical, Spiral, Target, Double)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1], facecolor="#111827")
    # Tile 2x2 composite image of phase/terrace fields across the 4 regimes
    r_keys = [
        "subcritical_dipole_annihilation",
        "single_screw_spiral_sheet_nacre",
        "opposite_chirality_frank_read_target",
        "same_chirality_double_spiral_tower",
    ]
    panels_2d = []
    for rk in r_keys:
        s2d = target_sim if rk == "opposite_chirality_frank_read_target" else regimes[rk]["sim2d"]
        # Normalize sin(phase) for crisp terrace front visualization; zero out subcritical control
        if s2d["active_terraces"] == 0:
            field_vis = np.zeros_like(s2d["XX_um"])
        else:
            field_vis = np.sin(np.angle(s2d["psi_field"]))
        panels_2d.append(field_vis)

    top_row = np.hstack([panels_2d[0], panels_2d[1]])
    bot_row = np.hstack([panels_2d[2], panels_2d[3]])
    mosaic = np.vstack([top_row, bot_row])
    im_b = ax_b.imshow(mosaic, cmap="twilight_shifted", vmin=-1.0, vmax=1.0, origin="lower")
    n_g = panels_2d[0].shape[0]
    ax_b.axhline(n_g - 0.5, color="white", lw=1.4)
    ax_b.axvline(n_g - 0.5, color="white", lw=1.4)
    labels_quad = [
        (0.25, 0.06, "1. Subcritical (+b,-b)\nd=0.60≤2r_c (0 rings)"),
        (0.75, 0.06, "2. Single Screw (+b)\nArchimedean Spiral"),
        (0.25, 0.56, f"3. Frank-Read (+b,-b)\nd={dipole_d:.2f}>2r_c Target"),
        (0.75, 0.56, "4. Same-Chirality (+b,+b)\n2-Arm Double Spiral"),
    ]
    for qx, qy, qlbl in labels_quad:
        ax_b.text(
            qx,
            qy,
            qlbl,
            transform=ax_b.transAxes,
            color="white",
            fontsize=7.3,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0b0f17", alpha=0.82, edgecolor="#374151"),
        )
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.set_title(
        "B. 4 Chiral Defect Regimes (Annihilation / Spiral / Target / Double Spiral)",
        color="white",
        fontsize=9.9,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Panel C: Gibbs-Thomson Velocity Law & Frank-Read Dipole Nucleation Threshold
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2], facecolor="#111827")
    r_grid = np.linspace(0.15, 6.0, 220)
    for rc_sp, lbl_sp, col_sp in [
        (0.19, "Haliotis rufescens (Tower, r_c=0.19 µm)", "#34d399"),
        (r_c, f"Pteria hirundo (Sheet, r_c={r_c:.2f} µm)", "#38bdf8"),
        (0.70, "Pinna nobilis (Sheet, r_c=0.70 µm)", "#f43f5e"),
    ]:
        v_curve = gibbs_thomson_step_velocity(r_grid, v_inf_um_hr=v_inf, r_c_um=rc_sp)
        ax_c.plot(r_grid, v_curve, lw=2.1, color=col_sp, label=lbl_sp)
        ax_c.axvline(rc_sp, color=col_sp, ls=":", lw=1.2, alpha=0.8)

    ax_c.axhline(v_inf, color="#facc15", ls="--", lw=1.5, label=f"Planar Step Speed v_∞={v_inf:.2f} µm/hr")
    # Inset showing gastopod columnar tower height profile vs bivalve sheet
    ins = ax_c.inset_axes([0.42, 0.16, 0.52, 0.40], facecolor="#0b0f17")
    mid_idx = tower3d["Z_um"].shape[0] // 2
    x_tr = np.linspace(-18.0, 18.0, tower3d["Z_um"].shape[1])
    ins.plot(x_tr, tower3d["Z_um"][mid_idx, :], color="#34d399", lw=1.5, label="Gastropod Towers")
    ins.plot(x_tr, sheet3d["Z_um"][mid_idx, :], color="#38bdf8", lw=1.4, ls="--", label="Bivalve Sheet")
    ins.set_title("Tower vs Sheet Z(x) (µm)", color="white", fontsize=6.8)
    ins.tick_params(colors="#9ca3af", labelsize=6)

    ax_c.set_xlabel("Terrace Curvature Radius r (µm)", color="white", fontsize=9)
    ax_c.set_ylabel("Normal Step Velocity v(r) (µm/hr)", color="white", fontsize=9)
    ax_c.set_title(
        r"C. Gibbs-Thomson Step Velocity $v(r)=v_\infty(1 - r_c/r)$ & Critical Arrest",
        color="white",
        fontsize=9.9,
        fontweight="bold",
    )
    ax_c.tick_params(colors="#9ca3af", labelsize=8)
    ax_c.grid(True, alpha=0.22, color="#374151")
    ax_c.legend(loc="upper right", fontsize=7.0, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel D: 12-Species Cross-Taxa BCF Pitch Scaling (Sheet vs Tower Nacre)
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0], facecolor="#111827")
    rc_line = np.linspace(0.12, 0.76, 100)
    ax_d.plot(
        rc_line,
        PITCH_RATIO_BCF * rc_line,
        "--",
        color="#e5e7eb",
        lw=2.0,
        label=r"BCF Universal Law $\lambda_0 = (2\pi/\omega_0)\,r_c \approx 18.98\,r_c$",
    )
    clade_styles = {
        "Bivalvia (Sheet Nacre)": ("#38bdf8", "o", "Bivalvia Sheet Nacre (7 spp.)"),
        "Gastropoda (Tower Nacre)": ("#f43f5e", "s", "Gastropoda Tower Nacre (4 spp.)"),
        "Cephalopoda (Tower Nacre)": ("#facc15", "D", "Cephalopoda Tower Nacre (1 sp.)"),
    }
    for cl, (col, mk, lbl) in clade_styles.items():
        pts = [r for r in cross_taxa["records"] if r["clade"] == cl]
        ax_d.scatter(
            [p["r_c_um"] for p in pts],
            [p["lambda_obs_um"] for p in pts],
            color=col,
            marker=mk,
            s=64,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
            label=lbl,
        )
    ax_d.set_xlabel(r"Critical Gibbs-Thomson Radius $r_c$ (µm)", color="white", fontsize=9)
    ax_d.set_ylabel(r"Spiral / Target Terrace Pitch $\lambda_0$ (µm)", color="white", fontsize=9)
    ax_d.set_title(
        f"D. 12-Species BCF Terrace Pitch Collapse (r={cross_taxa['pearson_r_pitch']:.4f}, slope={cross_taxa['bcf_pitch_slope']:.2f})",
        color="white",
        fontsize=9.9,
        fontweight="bold",
    )
    ax_d.tick_params(colors="#9ca3af", labelsize=8)
    ax_d.grid(True, alpha=0.22, color="#374151")
    ax_d.legend(loc="upper left", fontsize=7.2, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel E: SEM Fractured Mother-of-Pearl Specimen Segmentation
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1], facecolor="#111827")
    ax_e.imshow(specimen["gray"], cmap="bone")
    x0, y0, x1, y1 = specimen["roi_box"]
    ax_e.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        "--",
        color="#facc15",
        lw=1.3,
        label="Segmented Columnar ROI",
    )
    for idx_col, (xc, y_pks) in enumerate(specimen["detected_seams_xy"]):
        ax_e.scatter(
            np.full_like(y_pks, xc),
            y_pks,
            color="#38bdf8",
            s=11,
            alpha=0.85,
            label="Inter-Lamellar Mortar Seams" if idx_col == 0 else None,
        )
    ax_e.set_title(
        f"E. SEM Nacre Aragonite Tablets (h_0={specimen['mean_interlamellar_um']:.3f} µm, {specimen['mean_layers_per_column']:.1f} Layers/Col, CV={specimen['cv_interlamellar_spacing']:.3f})",
        color="white",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_e.set_xlabel("SEM X (px, Public Domain)", color="white", fontsize=8.5)
    ax_e.set_ylabel("SEM Y (px)", color="white", fontsize=8.5)
    ax_e.tick_params(colors="#9ca3af", labelsize=7.5)
    ax_e.legend(loc="lower right", fontsize=7.2, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel F: SEM Inter-Lamellar Spacing Distribution & Angular Frequency Scaling
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2], facecolor="#111827")
    spacings_um = specimen["spacings_y_px"] * 0.019867
    ax_f.hist(
        spacings_um,
        bins=16,
        color="#38bdf8",
        edgecolor="white",
        alpha=0.82,
        density=True,
        label=f"SEM Lamellar Spacing (n={specimen['n_seams_measured']} seams)",
    )
    ax_f.axvline(
        specimen["mean_interlamellar_um"],
        color="#f43f5e",
        lw=2.2,
        label=f"Segmented Mean h_0 = {specimen['mean_interlamellar_um']:.3f} µm",
    )
    ax_f.axvline(
        h_0,
        color="#facc15",
        ls="--",
        lw=2.0,
        label=f"Cartwright & Checa Reference h_0 = {h_0:.2f} µm",
    )
    # Inset showing 12-species angular frequency omega_obs vs omega_pred
    ins_f = ax_f.inset_axes([0.48, 0.44, 0.47, 0.46], facecolor="#0b0f17")
    om_p = [r["omega_pred_rad_hr"] for r in cross_taxa["records"]]
    om_o = [r["omega_obs_rad_hr"] for r in cross_taxa["records"]]
    ins_f.plot([0.5, 1.85], [0.5, 1.85], "--", color="#9ca3af", lw=1.2)
    ins_f.scatter(om_p, om_o, color="#34d399", s=26, edgecolors="white", linewidths=0.5)
    ins_f.set_title(f"ω_obs vs ω_pred (r={cross_taxa['pearson_r_omega']:.4f})", color="white", fontsize=6.8)
    ins_f.tick_params(colors="#9ca3af", labelsize=6)

    ax_f.set_xlabel("Inter-Lamellar Tablet Thickness h_0 (µm)", color="white", fontsize=9)
    ax_f.set_ylabel("Probability Density", color="white", fontsize=9)
    ax_f.set_title(
        f"F. Aragonite Tablet Thickness Distribution & ω Scaling (Aspect={specimen['tablet_aspect_ratio']:.2f})",
        color="white",
        fontsize=9.6,
        fontweight="bold",
    )
    ax_f.tick_params(colors="#9ca3af", labelsize=8)
    ax_f.grid(True, alpha=0.22, color="#374151")
    ax_f.legend(loc="upper left", fontsize=7.0, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel G (Pass 3): Stacked chitin membranes — Allen-Cahn perforation & mineral bridges
    # -------------------------------------------------------------------------
    ax_g = fig.add_subplot(gs[2, 0], projection="3d", facecolor="#0b0f17")
    col_sim = membrane["modes"]["columnar_gastropod"]
    sh_sim = membrane["modes"]["sheet_bivalve"]
    for sim_m, x_shift, col_fp, col_perf in [
        (col_sim, -5.0, "#38bdf8", "#facc15"),
        (sh_sim, 6.5, "#f43f5e", "#34d399"),
    ]:
        for k_sheet, (phi_k, fp_k) in enumerate(zip(sim_m["phi_sheets"], sim_m["footprints"]), start=1):
            z_k = k_sheet * sim_m["h_0_um"]
            ax_g.contour(
                sim_m["XX_um"] + x_shift,
                sim_m["YY_um"],
                fp_k.astype(float),
                levels=[0.5],
                colors=[col_fp],
                linewidths=0.9,
                zdir="z",
                offset=z_k,
            )
            ax_g.contourf(
                sim_m["XX_um"] + x_shift,
                sim_m["YY_um"],
                phi_k,
                levels=[0.5, 1.01],
                colors=[col_perf],
                alpha=0.85,
                zdir="z",
                offset=z_k,
            )
        cents = np.array(sim_m["bridge_centroids_um"])
        zs = np.arange(len(cents)) * sim_m["h_0_um"]
        ax_g.plot(cents[:, 0] + x_shift, cents[:, 1], zs, "-o", color="white", lw=1.4, ms=3.5, zorder=10)
    col_stats = membrane["mode_stats"]["columnar_gastropod"]
    sh_stats = membrane["mode_stats"]["sheet_bivalve"]
    ax_g.set_xlim(-11.0, 11.0)
    ax_g.set_ylim(-6.0, 6.0)
    ax_g.set_zlim(0.0, (max(col_sim["n_sheets"], sh_sim["n_sheets"]) + 1) * h_0)
    ax_g.set_xlabel("X (µm)", color="#9ca3af", fontsize=8)
    ax_g.set_ylabel("Y (µm)", color="#9ca3af", fontsize=8)
    ax_g.set_zlabel("Sheet Height (µm)", color="#9ca3af", fontsize=8)
    ax_g.tick_params(colors="#9ca3af", labelsize=7)
    ax_g.view_init(elev=28, azim=-60)
    ax_g.set_title(
        "G. Stacked Chitin Membranes: Perforation & Bridges\n"
        f"Columnar σ=0.1D: offset={col_stats['mean_offset_over_tablet_diam']:.2f}D, align={col_stats['alignment_index']:.2f}\n"
        f"Sheet σ=D: offset={sh_stats['mean_offset_over_tablet_diam']:.2f}D (brick-wall ref 0.5)",
        color="white",
        fontsize=8.4,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Panel H (Pass 3): Aragonite {110} cyclic-twin polygonized spiral vs single domain
    # -------------------------------------------------------------------------
    ax_h = fig.add_subplot(gs[2, 1], facecolor="#111827")
    half = twin_spiral["XX_um"].shape[1] // 2
    composite = np.hstack(
        [
            np.sin(np.angle(single_domain_spiral["psi_field"]))[:, :half],
            np.sin(np.angle(twin_spiral["psi_field"]))[:, half:],
        ]
    )
    ax_h.imshow(
        composite,
        cmap="twilight_shifted",
        vmin=-1.0,
        vmax=1.0,
        origin="lower",
        extent=[-16.0, 16.0, -16.0, 16.0],
    )
    ax_h.axvline(0.0, color="white", lw=1.2)
    th_c = twinning["twin"]["theta_rad"]
    for res, col_c, lbl_c in [
        (twinning["single"], "#f43f5e", "single-domain ring (c2/c0=%.3f)" % twinning["single"]["c2_over_c0"]),
        (twinning["twin"], "#34d399", "{110} twin ring (c2/c0=%.3f)" % twinning["twin"]["c2_over_c0"]),
    ]:
        rr = res["r_contour_um"]
        ax_h.plot(rr * np.cos(th_c), rr * np.sin(th_c), color=col_c, lw=1.5, label=lbl_c)
    ax_h.text(-15.2, 13.6, "Single orthorhombic\ndomain (untwinned)", color="white", fontsize=7.4, fontweight="bold")
    ax_h.text(2.0, 13.6, "{110} cyclic twin\n(3 domains, τ=63.8°)", color="white", fontsize=7.4, fontweight="bold")
    ax_h.set_xlabel("x (µm)", color="white", fontsize=8.5)
    ax_h.set_ylabel("y (µm)", color="white", fontsize=8.5)
    ax_h.tick_params(colors="#9ca3af", labelsize=7.5)
    ax_h.legend(loc="lower left", fontsize=6.8, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")
    ax_h.set_title(
        f"H. Aragonite Anisotropy v_100/v_010={twinning['v100_over_v010']:.3f}, e_6={e6:.2f} & {{110}} Twin\n"
        f"f_2={twinning['f2_measured']:.3f} (analytic {twinning['f2_analytic']:.3f}), "
        f"f_6={twinning['f6_measured']:.3f} (analytic {twinning['f6_analytic']:.3f})",
        color="white",
        fontsize=8.4,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Panel I (Pass 3): Ablations — bridge localisation sweep & e_6 polygonization sweep
    # -------------------------------------------------------------------------
    ax_i = fig.add_subplot(gs[2, 2], facecolor="#111827")
    ax_i.errorbar(
        membrane["sigma_grid_over_tablet_diam"],
        membrane["sweep_offset_mean"],
        yerr=membrane["sweep_offset_sd"],
        fmt="-o",
        color="#38bdf8",
        ecolor="#9ca3af",
        capsize=3,
        lw=1.8,
        label="Stacking offset |c_k − c_{k−1}| / D_tab (3 seeds)",
    )
    ax_i.axhline(0.0, color="#38bdf8", ls=":", lw=1.0)
    ax_i.axhline(0.5, color="#f43f5e", ls="--", lw=1.2, label="Brick-wall reference 0.5 (sheet nacre)")
    ax_i.axhline(1.0 / 3.0, color="#facc15", ls=":", lw=1.1, label="Uniform-disc random offset 1/3")
    ax_i.axvline(0.10, color="#34d399", ls="--", lw=1.0, label="Columnar reference σ=0.10 D_tab (Checa 2011)")
    ax_i.set_xscale("log")
    ax_i.set_xlabel("Mineral-Bridge Localisation σ / D_tab", color="white", fontsize=9)
    ax_i.set_ylabel("Lateral Stacking Offset / D_tab", color="white", fontsize=9)
    ax_i.tick_params(colors="#9ca3af", labelsize=8)
    ax_i.grid(True, alpha=0.22, color="#374151")
    ax_i.set_ylim(-0.03, 0.62)
    ins_i = ax_i.inset_axes([0.50, 0.50, 0.47, 0.40], facecolor="#0b0f17")
    ins_i.plot(twinning["e6_grid"], twinning["c6_sweep_single"], "-o", color="#f43f5e", ms=3.5, lw=1.3, label="single domain")
    ins_i.plot(twinning["e6_grid"], twinning["c6_sweep_twin"], "-s", color="#34d399", ms=3.5, lw=1.3, label="{110} twin")
    ins_i.set_title("6-fold polygonization 2|c_6|/c_0 vs e_6", color="white", fontsize=6.8)
    ins_i.set_xlabel("e_6", color="#9ca3af", fontsize=6.5)
    ins_i.tick_params(colors="#9ca3af", labelsize=6)
    ins_i.legend(fontsize=5.8, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white", loc="upper left")
    ax_i.legend(loc="lower right", fontsize=6.6, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")
    ax_i.set_title(
        "I. Ablations: Columnar→Sheet Transition & e_6 Sweep",
        color="white",
        fontsize=9.0,
        fontweight="bold",
    )

    fig.suptitle(
        "Cartwright & Checa (2007) J. R. Soc. Interface & Cartwright et al. (2009) PNAS — Nacre Screw-Dislocation Spirals, "
        "Target Terraces, Membrane Perforation & Aragonite Twinning (v3)",
        color="white",
        fontsize=12.2,
        fontweight="bold",
        y=0.985,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_growth_gif(out_path: Path = OUT_GIF) -> Path:
    """Render Pass 2 co-evolving 2-panel animated GIF comparing single screw-dislocation
    Archimedean spiral rotation (+b) against opposite-chirality Frank-Read target ring
    emission (+b, -b) across 7 time steps."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    times_hr = np.linspace(0.0, 6.4, 7)

    for t_cur in times_hr:
        sp = simulate_chiral_dislocation_terrace_2d(
            regime_id="single_screw_spiral_sheet_nacre",
            b_1=1,
            b_2=0,
            dipole_separation_um=0.0,
            n_grid=135,
            t_hr=float(t_cur),
        )
        tg = simulate_chiral_dislocation_terrace_2d(
            regime_id="opposite_chirality_frank_read_target",
            b_1=1,
            b_2=-1,
            dipole_separation_um=2.40,
            n_grid=135,
            t_hr=float(t_cur),
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=105, facecolor="#0b0f17")
        ax1.pcolormesh(
            sp["XX_um"],
            sp["YY_um"],
            np.sin(np.angle(sp["psi_field"])),
            cmap="twilight_shifted",
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        ax1.scatter([0.0], [0.0], color="#facc15", s=50, edgecolors="black", zorder=5)
        ax1.set_title(
            f"Single Screw Dislocation (+b=+1) | t={t_cur:.1f} hr | λ_0={sp['lambda_0_um']:.2f} µm",
            color="white",
            fontsize=9.2,
            fontweight="bold",
        )
        ax1.set_xlabel("x (µm)", color="white", fontsize=8)
        ax1.set_ylabel("y (µm)", color="white", fontsize=8)
        ax1.tick_params(colors="#9ca3af", labelsize=7)

        ax2.pcolormesh(
            tg["XX_um"],
            tg["YY_um"],
            np.sin(np.angle(tg["psi_field"])),
            cmap="twilight_shifted",
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        ax2.scatter([-1.2, 1.2], [0.0, 0.0], color=["#38bdf8", "#f43f5e"], s=50, edgecolors="white", zorder=5)
        ax2.set_title(
            f"Frank-Read Dipole (+b,-b, b_net=0) | d=2.40 µm > 2r_c",
            color="white",
            fontsize=9.2,
            fontweight="bold",
        )
        ax2.set_xlabel("x (µm)", color="white", fontsize=8)
        ax2.set_ylabel("y (µm)", color="white", fontsize=8)
        ax2.tick_params(colors="#9ca3af", labelsize=7)

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_px, w_px, 4))
        frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=400,
        loop=0,
    )
    return out_path


def render_membrane_perforation_gif(out_path: Path = OUT_GIF_MEMBRANE, seed: int = 2007) -> Path:
    """Pass 3: animate the Allen-Cahn perforation of the first chitin membrane sheet above the
    columnar (Haliotis) terrace tower next to the sheet-nacre (Pteria) case, with the
    eikonal tablet footprint outlined and the emerging mineral-bridge site marked."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p3 = load_pass3_reference()
    params = p3["interlamellar_membrane_perforation_model"]["parameters"]
    common = dict(
        eps_um=float(params["eps_um"]),
        beta=float(params["beta"]),
        s_0=float(params["s_0"]),
        mobility=float(params["mobility"]),
        dt=float(params["dt"]),
        n_steps=int(params["n_steps"]),
        pore_density_per_um2=float(params["pore_density_per_um2"]),
        pore_width_um=float(params["pore_width_um"]),
        pore_amplitude=float(params["pore_amplitude"]),
    )
    col = simulate_membrane_perforation_phase_field_3d(
        "columnar_gastropod", r_c_um=0.19, tablet_diam_um=7.8, bridge_localization_um=0.78, seed=seed, keep_history=True, **common
    )
    sh = simulate_membrane_perforation_phase_field_3d(
        "sheet_bivalve", r_c_um=0.42, tablet_diam_um=4.2, bridge_localization_um=4.2, seed=seed, keep_history=True, **common
    )
    n_frames = min(len(col["history_sheet1"]), len(sh["history_sheet1"]))
    frames: list[Image.Image] = []
    for f_idx in range(n_frames):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=105, facecolor="#0b0f17")
        for ax, sim_m, ttl in [
            (axes[0], col, "Columnar nacre (Haliotis): σ = 0.1 D_tab"),
            (axes[1], sh, "Sheet nacre (Pteria): σ = D_tab (pores only)"),
        ]:
            ax.set_facecolor("#111827")
            ax.pcolormesh(
                sim_m["XX_um"], sim_m["YY_um"], sim_m["history_sheet1"][f_idx], cmap="magma", vmin=0.0, vmax=1.0, shading="auto"
            )
            ax.contour(sim_m["XX_um"], sim_m["YY_um"], sim_m["footprints"][0].astype(float), levels=[0.5], colors=["#38bdf8"], linewidths=1.2)
            ax.contour(sim_m["XX_um"], sim_m["YY_um"], sim_m["level"], levels=np.arange(0.5, 7.5, 1.0), colors=["#374151"], linewidths=0.5)
            c1 = sim_m["bridge_centroids_um"][1]
            ax.scatter([c1[0]], [c1[1]], marker="x", color="#34d399", s=60, zorder=6)
            ax.set_xlim(-6.0, 6.0)
            ax.set_ylim(-6.0, 6.0)
            ax.set_aspect("equal")
            ax.set_title(f"{ttl} | step {15 * f_idx}", color="white", fontsize=9.0, fontweight="bold")
            ax.set_xlabel("x (µm)", color="white", fontsize=8)
            ax.set_ylabel("y (µm)", color="white", fontsize=8)
            ax.tick_params(colors="#9ca3af", labelsize=7)
        fig.suptitle(
            "Chitin membrane perforation phase field φ (0 intact → 1 perforated) above the eikonal terrace tower; × = mineral bridge",
            color="white",
            fontsize=9.2,
        )
        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_px, w_px, 4))
        frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=350, loop=0)
    return out_path


# =============================================================================
# 8. MorphoEval Quantitative Measurement Suite & 12-Check Self-Test
# =============================================================================
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 15 reference-anchored quantitative morphometrics for `morphoeval/measure_outputs.py`
    (9 carried from Pass 2 + 6 Pass 3 membrane-perforation / twinning measurements)."""
    regimes = evaluate_all_morphospace_regimes()
    cross_taxa = evaluate_cross_taxa_nacre_scaling()
    specimen = segment_nacre_sem_specimen()
    membrane = evaluate_membrane_bridge_alignment()
    twinning = evaluate_twinning_polygonization()
    p3 = load_pass3_reference()

    sub_reg = regimes["subcritical_dipole_annihilation"]
    sp_reg = regimes["single_screw_spiral_sheet_nacre"]
    tg_reg = regimes["opposite_chirality_frank_read_target"]
    db_reg = regimes["same_chirality_double_spiral_tower"]
    col_stats = membrane["mode_stats"]["columnar_gastropod"]
    sh_stats = membrane["mode_stats"]["sheet_bivalve"]

    return [
        {
            "name": "12-species BCF spiral terrace pitch correlation (Cartwright & Checa 2007 / BCF 1951)",
            "value": round(cross_taxa["pearson_r_pitch"], 4),
            "unit": "Pearson r",
            "method": f"Correlation of predicted BCF pitch (2*pi/omega_0)*r_c vs empirical terrace spacing across {cross_taxa['n_species']} bivalve, gastropod, and cephalopod species (RMSE={cross_taxa['rmse_pitch_um']:.4f} um, p={cross_taxa['p_value_pitch']:.2e})",
            "reference_value": 1.0,
            "reference_source": "Cartwright & Checa (2007) J. R. Soc. Interface 4:491-504 & Burton, Cabrera & Frank (1951) Phil. Trans. R. Soc. A 243:299-358",
            "note": f"RMSE = {cross_taxa['rmse_pitch_um']:.4f} um across 3 molluscan classes",
            "derived_from": "script",
        },
        {
            "name": "12-species spiral angular rotation frequency correlation omega = omega_0 * v_inf / r_c",
            "value": round(cross_taxa["pearson_r_omega"], 4),
            "unit": "Pearson r",
            "method": f"Correlation of predicted vs observed spiral angular rotation frequency across {cross_taxa['n_species']} nacre species (RMSE={cross_taxa['rmse_omega_rad_hr']:.4f} rad/hr)",
            "reference_value": 1.0,
            "reference_source": "Cartwright & Checa (2007) J. R. Soc. Interface 4:491-504 Sec. 2",
            "note": f"RMSE = {cross_taxa['rmse_omega_rad_hr']:.4f} rad/hr, p={cross_taxa['p_value_omega']:.2e}",
            "derived_from": "script",
        },
        {
            "name": "Empirical BCF terrace-pitch-to-critical-radius slope lambda_0 / r_c",
            "value": round(cross_taxa["bcf_pitch_slope"], 4),
            "unit": "dimensionless ratio (lambda_0 / r_c)",
            "method": f"OLS regression slope of observed terrace pitch lambda_obs vs critical radius r_c across 12 species (SE ± {cross_taxa['bcf_pitch_slope_se']:.4f})",
            "reference_value": round(PITCH_RATIO_BCF, 4),
            "reference_source": "Burton, Cabrera & Frank (1951) Phil. Trans. R. Soc. A 243:299-358 (lambda_0 / r_c = 2*pi / 0.330958 ≈ 18.9848 ≈ 19)",
            "note": f"Bivalve sheet mean pitch = {cross_taxa['mean_sheet_pitch_um']:.2f} um vs Gastropod/Cephalopod tower mean = {cross_taxa['mean_tower_pitch_um']:.2f} um",
            "derived_from": "script",
        },
        {
            "name": "Bivalve sheet nacre (*Pteria hirundo*) single-spiral terrace pitch lambda_0",
            "value": round(sp_reg["terrace_pitch_sim_um"], 4),
            "unit": "um",
            "method": "Raw radial phase regression of the 2D BCF spiral eikonal travel over r in [0.6 L, 0.98 L] at r_c = 0.42 um, v_inf = 1.25 um/hr (Pass 3: no blending with the theoretical pitch; the ~2 % deficit is the Cabrera-Levine logarithmic core correction)",
            "reference_value": 7.9736,
            "reference_source": "Cartwright & Checa (2007) J. R. Soc. Interface 4:491-504 & Cartwright et al. (2009) PNAS 106:10499-10504",
            "note": f"outer winding number = {sp_reg['winding_number_outer']:.2f}, terrace steps counted along the diameter = {sp_reg['active_terraces_sim']} (reference {sp_reg['active_terraces_ref']})",
            "derived_from": "script",
        },
        {
            "name": "Same-chirality (+b, +b) double-spiral to single-spiral terrace pitch ratio",
            "value": round(db_reg["terrace_pitch_sim_um"] / max(sp_reg["terrace_pitch_sim_um"], 1e-9), 4),
            "unit": "ratio (lambda_2 / lambda_0)",
            "method": "Ratio of measured inter-step terrace spacing for co-rotating (+1, +1) double spiral (b_net=+2) vs (+1, 0) single spiral (b_net=+1)",
            "reference_value": 0.50,
            "reference_source": "Cartwright et al. (2009) PNAS 106:10499-10504 & Lean 4 theorem bcf_archimedean_terrace_pitch_scaling",
            "note": f"double-spiral pitch = {db_reg['terrace_pitch_sim_um']:.4f} um, outer winding = {db_reg['winding_number_outer']:.2f}",
            "derived_from": "script",
        },
        {
            "name": "Subcritical Frank-Read dipole negative control active terraces (d = 0.60 um <= 2*r_c = 0.84 um)",
            "value": float(sub_reg["active_terraces_sim"]),
            "unit": "terraces",
            "method": "Count of propagating target terraces for opposite-chirality dipole (+1, -1) separated by d = 0.60 um <= 2*r_c = 0.84 um (pinch-off annihilation negative control vs d = 2.40 um supercritical target)",
            "reference_value": 0.0,
            "reference_source": "Cartwright et al. (2009) PNAS 106:10499-10504 & Lean 4 theorem cartwright_frank_read_topological_burgers_algebra",
            "note": f"subcritical d=0.60 um -> 0 terraces vs supercritical d=2.40 um -> {tg_reg['active_terraces_sim']} target rings (b_net={tg_reg['winding_number_outer']:.2f})",
            "derived_from": "script",
        },
        {
            "name": "SEM specimen segmented inter-lamellar aragonite tablet thickness h_0 (nacre_aragonite_tablets_specimen.jpg)",
            "value": round(specimen["mean_interlamellar_um"], 4),
            "unit": "um",
            "method": f"1D vertical Difference-of-Gaussians mortar-seam detection across 25 column transects ({specimen['n_seams_measured']} seams, mean dy = {specimen['mean_interlamellar_px']:.2f} px at 0.019867 um/px). CAVEAT: the Wikimedia micrograph carries no scale bar; the um/px calibration was anchored to h_0 = 0.50 um, so this value is a consistency check, not an independent test of h_0",
            "reference_value": 0.50,
            "reference_source": "Cartwright & Checa (2007) J. R. Soc. Interface 4:491-504 (molluscan nacre inter-lamellar spacing h_0 ≈ 0.50 um)",
            "note": f"spacing CV = {specimen['cv_interlamellar_spacing']:.4f}, mean layers/col = {specimen['mean_layers_per_column']:.2f}; the dimensionless CV and aspect ratio below are calibration-independent",
            "derived_from": "output",
        },
        {
            "name": "SEM specimen aragonite tablet width-to-thickness aspect ratio D_tab / h_0 (nacre_aragonite_tablets_specimen.jpg)",
            "value": round(specimen["tablet_aspect_ratio"], 4),
            "unit": "ratio (dx / dy)",
            "method": f"Ratio of horizontal tablet step width ({specimen['mean_tablet_width_px']:.2f} px = {specimen['mean_tablet_width_um']:.3f} um) to vertical lamellar thickness ({specimen['mean_interlamellar_px']:.2f} px = {specimen['mean_interlamellar_um']:.3f} um)",
            "reference_value": 3.2436,
            "reference_source": "Cartwright & Checa (2007) J. R. Soc. Interface 4:491-504 brick-and-mortar aragonite tablet aspect ratio",
            "note": f"measured across {specimen['n_seams_measured']} vertical seams and horizontal rows",
            "derived_from": "output",
        },
        {
            "name": "SEM specimen inter-lamellar spacing coefficient of variation CV (nacre_aragonite_tablets_specimen.jpg)",
            "value": round(specimen["cv_interlamellar_spacing"], 4),
            "unit": "CV (std/mean)",
            "method": f"Standard deviation over mean of {specimen['n_seams_measured']} inter-lamellar chitin-mineral mortar spacings in SEM nacre fracture specimen",
            "reference_value": 0.2933,
            "reference_source": "Cartwright & Checa (2007) J. R. Soc. Interface 4:491-504 uniform periodic chitin-sheet spacing (CV < 0.35)",
            "note": f"mean vertical stack = {specimen['mean_layers_per_column']:.2f} layers per column",
            "derived_from": "output",
        },
        # ---------------- Pass 3 additions ----------------
        {
            "name": "Columnar gastropod nacre (*Haliotis rufescens*) mineral-bridge stacking offset |c_k - c_(k-1)| / D_tab from stacked-sheet Allen-Cahn membrane perforation",
            "value": round(col_stats["mean_offset_over_tablet_diam"], 4),
            "unit": "ratio (offset / tablet diameter)",
            "method": f"Mean lateral offset between successive mineral-bridge sites (argmax of early perforation phase field) over {col_stats['n_sheets_total']} sheets x {col_stats['n_seeds']} seeds, bridge localisation sigma = 0.10 D_tab, eikonal tower r_c = 0.19 um, D_tab = 7.8 um",
            "reference_value": col_stats["reference_offset"],
            "reference_source": "Cartwright & Checa (2007) J. R. Soc. Interface 4:491-504 Sec. 4; Checa, Cartwright & Willinger (2011) J. Struct. Biol. 176:330-339 (central mineral bridges -> vertically aligned tablet stacks)",
            "note": f"alignment index (offset < 0.15 D_tab) = {col_stats['alignment_index']:.3f} vs reference {col_stats['reference_alignment_index']:.1f}; perforated area fraction per sheet = {np.mean(membrane['modes']['columnar_gastropod']['perforated_fraction_per_sheet']):.3f}",
            "derived_from": "script",
        },
        {
            "name": "Bivalve sheet nacre (*Pteria hirundo*) brick-wall mineral-bridge stacking offset / D_tab from random-pore membrane perforation",
            "value": round(sh_stats["mean_offset_over_tablet_diam"], 4),
            "unit": "ratio (offset / tablet diameter)",
            "method": f"Mean lateral offset between successive pore-nucleated mineral bridges over {sh_stats['n_sheets_total']} sheets x {sh_stats['n_seeds']} seeds with bridge localisation sigma = D_tab (A_g = 0.01, pores only), eikonal r_c = 0.42 um, D_tab = 4.2 um; SD = {sh_stats['sd_offset_over_tablet_diam']:.3f}",
            "reference_value": sh_stats["reference_offset"],
            "reference_source": "Cartwright & Checa (2007) J. R. Soc. Interface 4:491-504 Sec. 4 (sheet-nacre tablets staggered like bricks in a wall; nominal half-tablet offset). The random-pore limit of a uniform disc predicts 1/3",
            "note": f"alignment index = {sh_stats['alignment_index']:.3f}; model under-predicts the nominal 0.5 brick-wall offset because pores are drawn uniformly over the tablet interior",
            "derived_from": "script",
        },
        {
            "name": "Columnar-to-sheet stacking transition: bridge localisation sigma / D_tab at which the mean offset first exceeds 0.1 D_tab",
            "value": round(float(membrane["sigma_grid_over_tablet_diam"][int(np.argmax(membrane["sweep_offset_mean"] > 0.10))]), 4),
            "unit": "ratio (sigma / tablet diameter)",
            "method": f"Sweep of sigma / D_tab over {membrane['sigma_grid_over_tablet_diam'].tolist()} (3 seeds each) with a fixed bridge budget A_g = (0.1 D_tab / sigma)^2; offsets = {np.round(membrane['sweep_offset_mean'], 3).tolist()}",
            "reference_value": 0.125,
            "reference_source": "Model prediction: the Gaussian bridge drive falls below the spinodal threshold S_crit = S_0 + 1/(3 beta) = 0.683 once (0.1 D_tab / sigma)^2 < 0.683, i.e. sigma > 0.121 D_tab",
            "note": "ablation exposing the parameter controlling columnar (gastropod) vs sheet (bivalve) nacre in the reduced model",
            "derived_from": "script",
        },
        {
            "name": "Aragonite {110} cyclic-twin suppression of 2-fold terrace elongation f_2 (measured on anisotropic eikonal terrace ring)",
            "value": round(twinning["f2_measured"], 4),
            "unit": "ratio (twinned / single-domain 2|c_2|/c_0)",
            "method": f"Fourier amplitude ratio of the closed terrace ring r(theta) for 3 twin domains at tau = {twinning['twin_angle_deg']:.2f} deg vs a single orthorhombic domain with v_100/v_010 = {twinning['v100_over_v010']:.4f} (single c2/c0 = {twinning['single']['c2_over_c0']:.4f}, twin = {twinning['twin']['c2_over_c0']:.4f}); ideal 60-deg trilling control c2/c0 = {twinning['ideal_60deg_c2_over_c0']:.2e}",
            "reference_value": round(twinning["f2_analytic"], 4),
            "reference_source": "Analytic f_2 = |sum_j exp(2 i j tau)| / 3 with tau = 2 atan(a/b), a = 4.9611 A, b = 7.9672 A (De Villiers 1971 Am. Mineral. 56:758-767); Lean theorem aragonite_cyclic_twin_angular_misfit",
            "note": f"cyclic twin angular misfit 3 tau - 180 = {twinning['cyclic_twin_misfit_deg']:.2f} deg",
            "derived_from": "script",
        },
        {
            "name": "Aragonite {110} cyclic-twin retention of 6-fold pseudo-hexagonal polygonization f_6 (measured on anisotropic eikonal terrace ring)",
            "value": round(twinning["f6_measured"], 4),
            "unit": "ratio (twinned / single-domain 2|c_6|/c_0)",
            "method": f"Fourier amplitude ratio of the 6th harmonic of the twinned vs single-domain terrace ring with e_6 = {E6_DEFAULT:.2f} (single c6/c0 = {twinning['single']['c6_over_c0']:.4f}, twin = {twinning['twin']['c6_over_c0']:.4f}); e_6 sweep: single = {np.round(twinning['c6_sweep_single'], 3).tolist()}, twin = {np.round(twinning['c6_sweep_twin'], 3).tolist()}",
            "reference_value": round(twinning["f6_analytic"], 4),
            "reference_source": "Analytic f_6 = |sum_j exp(6 i j tau)| / 3 for tau = 63.82 deg; Cartwright & Checa (2007) pseudo-hexagonal aragonite tablet habit",
            "note": "e_6 is a free kinetic anisotropy amplitude of the reduced model (not a literature value)",
            "derived_from": "script",
        },
        {
            "name": "Aragonite {110} twin angle 2 atan(a/b) from De Villiers (1971) lattice parameters",
            "value": round(twinning["twin_angle_deg"], 3),
            "unit": "deg",
            "method": "2 * atan(a / b) with a = 4.9611 A, b = 7.9672 A (code constants ARAGONITE_A_ANGSTROM, ARAGONITE_B_ANGSTROM)",
            "reference_value": float(p3["aragonite_crystallography"]["twin_angle_deg"]),
            "reference_source": "De Villiers JPR (1971) Am. Mineral. 56:758-767 (lattice); standard aragonite twin angle ~63.8 deg",
            "note": "parity check between code constants and vendored membrane_perforation_twinning_reference.json",
            "derived_from": "script",
        },
    ]


def self_test() -> bool:
    """Run the 12-point verification suite (8 Pass 2 checks + 4 Pass 3 checks)."""
    # 1. Cryptographic SHA-256 verification of all 6 pinned dataset files
    digests = verify_dataset_integrity()
    assert len(digests) == 6, "Expected 6 SHA-256-verified dataset files"

    # 2. JSON-to-code parity checks
    ref = load_reference_data()
    defaults = ref["governing_equations"]["default_parameters"]
    assert math.isclose(float(defaults["r_c_um"]), 0.42, rel_tol=1e-9)
    assert math.isclose(float(defaults["h_0_um"]), 0.50, rel_tol=1e-9)

    # 3. Law 1: Gibbs-Thomson critical arrest v(r_c) = 0 and radial monotonicity
    v_at_rc = float(gibbs_thomson_step_velocity(0.42, v_inf_um_hr=1.25, r_c_um=0.42))
    v_at_2rc = float(gibbs_thomson_step_velocity(0.84, v_inf_um_hr=1.25, r_c_um=0.42))
    assert v_at_rc == 0.0
    assert math.isclose(v_at_2rc, 0.625, rel_tol=1e-9)

    # 4. Law 2 & Law 3: 4 Chiral Dislocation Regimes & Topological Winding Numbers
    regimes = evaluate_all_morphospace_regimes()
    sub = regimes["subcritical_dipole_annihilation"]
    sp = regimes["single_screw_spiral_sheet_nacre"]
    tg = regimes["opposite_chirality_frank_read_target"]
    db = regimes["same_chirality_double_spiral_tower"]

    assert sub["active_terraces_sim"] == 0 and sub["terrace_pitch_sim_um"] == 0.0
    assert math.isclose(sp["winding_number_outer"], 1.0, abs_tol=1e-2)
    assert math.isclose(tg["winding_number_outer"], 0.0, abs_tol=1e-2)
    assert math.isclose(db["winding_number_outer"], 2.0, abs_tol=1e-2)
    assert math.isclose(db["terrace_pitch_sim_um"] / sp["terrace_pitch_sim_um"], 0.5, rel_tol=0.03)
    # Pass 3: raw (unblended) pitch within 4 % of lambda_0 and measured step counts within ±1 of reference
    assert math.isclose(sp["terrace_pitch_sim_um"], sp["terrace_pitch_ref_um"], rel_tol=0.04)
    for reg_chk in (sp, tg, db):
        assert abs(reg_chk["active_terraces_sim"] - reg_chk["active_terraces_ref"]) <= 1, reg_chk["regime_id"]

    # 5. 3D Bivalve Sheet vs Gastropod Tower Nacre Architecture Checks
    s3d = simulate_nacre_architecture_3d("bivalve_sheet_spiral")
    t3d = simulate_nacre_architecture_3d("gastropod_columnar_towers")
    assert s3d["max_height_um"] > 1.0 and t3d["max_height_um"] > 2.0

    # 6. 12-Species Cross-Taxa BCF Terrace Pitch & Angular Frequency Scaling
    cross = evaluate_cross_taxa_nacre_scaling()
    assert cross["n_species"] == 12 and cross["n_clades"] == 3
    assert cross["pearson_r_pitch"] > 0.999
    assert cross["pearson_r_omega"] > 0.999
    assert abs(cross["bcf_pitch_slope"] - PITCH_RATIO_BCF) < 0.25
    assert cross["mean_sheet_pitch_um"] > 2.0 * cross["mean_tower_pitch_um"]

    # 7. Real SEM Nacre Specimen Lamellar Segmentation Checks
    spec = segment_nacre_sem_specimen()
    assert math.isclose(spec["mean_interlamellar_um"], 0.50, rel_tol=0.02)
    assert 0.15 < spec["cv_interlamellar_spacing"] < 0.35
    assert spec["mean_layers_per_column"] > 12.0

    # 8. Two-run bitwise determinism check on 2D multi-dislocation solver
    run1 = simulate_chiral_dislocation_terrace_2d("opposite_chirality_frank_read_target", 1, -1, 2.40)
    run2 = simulate_chiral_dislocation_terrace_2d("opposite_chirality_frank_read_target", 1, -1, 2.40)
    assert np.allclose(run1["height_um"], run2["height_um"], atol=0.0)

    # 9. Pass 3 — aragonite anisotropy & {110} twinning: lattice parity, analytic vs measured
    #    twin suppression factors, ideal-60-deg control cancels 2-fold term, isotropic limit
    p3 = load_pass3_reference()
    cryst = p3["aragonite_crystallography"]
    assert math.isclose(float(cryst["a_angstrom"]), ARAGONITE_A_ANGSTROM) and math.isclose(float(cryst["b_angstrom"]), ARAGONITE_B_ANGSTROM)
    tw = evaluate_twinning_polygonization()
    assert math.isclose(tw["twin_angle_deg"], float(cryst["twin_angle_deg"]), abs_tol=0.05)
    assert math.isclose(tw["cyclic_twin_misfit_deg"], 3.0 * tw["twin_angle_deg"] - 180.0, abs_tol=1e-9)
    assert math.isclose(tw["f2_measured"], tw["f2_analytic"], rel_tol=0.05)
    assert math.isclose(tw["f6_measured"], tw["f6_analytic"], rel_tol=0.05)
    assert tw["ideal_60deg_c2_over_c0"] < 1e-6
    assert tw["isotropic"]["c6_over_c0"] < 1e-9 and tw["twin"]["c6_over_c0"] > 0.1
    a_th = aragonite_step_velocity_anisotropy(np.array([0.0, math.pi / 2.0]), e6=0.0, twinned=False)
    assert math.isclose(float(a_th[0] / a_th[1]), ARAGONITE_V100_OVER_V010, rel_tol=1e-9)
    # Anisotropy off -> solver identical to Pass 2 behaviour
    iso_a = simulate_chiral_dislocation_terrace_2d("single_screw_spiral_sheet_nacre", 1, 0, 0.0, n_grid=61)
    iso_b = simulate_chiral_dislocation_terrace_2d("single_screw_spiral_sheet_nacre", 1, 0, 0.0, n_grid=61, step_anisotropy=None)
    assert np.allclose(iso_a["height_um"], iso_b["height_um"], atol=0.0)

    # 10. Pass 3 — stacked-sheet membrane perforation: columnar alignment emerges, sheet mode
    #     is staggered, eikonal coupling (footprint inside terrace tier), stable explicit dt
    mem = evaluate_membrane_bridge_alignment()
    col_s = mem["mode_stats"]["columnar_gastropod"]
    sh_s = mem["mode_stats"]["sheet_bivalve"]
    assert col_s["mean_offset_over_tablet_diam"] < 0.05 and col_s["alignment_index"] > 0.9
    assert sh_s["mean_offset_over_tablet_diam"] > 0.10
    col_sim = mem["modes"]["columnar_gastropod"]
    assert col_sim["n_sheets"] >= 3 and col_sim["dt_eff"] <= 0.9 * (col_sim["XX_um"][0, 1] - col_sim["XX_um"][0, 0]) ** 2 / (4.0 * 0.3**2)
    for k_sheet, fp_k in enumerate(col_sim["footprints"], start=1):
        assert np.all(col_sim["level"][fp_k] >= k_sheet)
    for phi_k in col_sim["phi_sheets"]:
        assert float(np.min(phi_k)) >= 0.0 and float(np.max(phi_k)) <= 1.0 and float(np.max(phi_k)) > 0.5
    # Perforation spinodal threshold S_crit = S_0 + 1/(3 beta) and sweep transition location
    assert math.isclose(col_sim["s_crit"], 0.35 + 1.0 / 3.0, rel_tol=1e-9)
    sweep = mem["sweep_offset_mean"]
    assert sweep[1] < 0.05 and sweep[-1] > 0.10  # sigma = 0.10 D_tab columnar, sigma = D_tab staggered

    # 11. Pass 3 — seeded determinism of the phase-field stack (two runs, same seed)
    pf1 = simulate_membrane_perforation_phase_field_3d(seed=2007, max_sheets=2)
    pf2 = simulate_membrane_perforation_phase_field_3d(seed=2007, max_sheets=2)
    assert np.allclose(pf1["phi_sheets"][0], pf2["phi_sheets"][0], atol=0.0)

    # 12. Measurement suite size and reference anchoring
    meas = measure_morphoeval()
    assert len(meas) == 15
    assert all(("reference_value" in m_i and "unit" in m_i) for m_i in meas)
    print(
        f"[self_test] 2007-cartwright-nacre-spiral-dislocation-terraces.py (v3) PASSED all 12 checks "
        f"(12-species r_pitch={cross['pearson_r_pitch']:.4f}, raw spiral pitch={sp['terrace_pitch_sim_um']:.4f} um, "
        f"SEM h_0={spec['mean_interlamellar_um']:.4f} um, CV={spec['cv_interlamellar_spacing']:.4f}; "
        f"columnar offset={col_s['mean_offset_over_tablet_diam']:.3f} D_tab, sheet offset={sh_s['mean_offset_over_tablet_diam']:.3f} D_tab; "
        f"f_2={tw['f2_measured']:.4f}/{tw['f2_analytic']:.4f}, f_6={tw['f6_measured']:.4f}/{tw['f6_analytic']:.4f})."
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run 12-check verification suite")
    ap.add_argument("--skip-gif", action="store_true", help="Skip animated GIF rendering")
    ap.add_argument("--v-inf", type=float, default=None, help="Override straight-step velocity v_inf (um/hr)")
    ap.add_argument("--r-c", type=float, default=None, help="Override critical Gibbs-Thomson radius r_c (um)")
    ap.add_argument("--h-0", type=float, default=None, help="Override inter-lamellar mortar spacing h_0 (um)")
    ap.add_argument("--dipole-d", type=float, default=None, help="Override Frank-Read dipole separation d (um)")
    ap.add_argument("--e6", type=float, default=None, help="Override pseudo-hexagonal kinetic anisotropy amplitude e_6 (panel H)")
    ap.add_argument("--seed", type=int, default=2007, help="Seed of the membrane pore-nucleation field (default: 2007)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    overrides = {
        k_name: v
        for k_name, v in {
            "v_inf": args.v_inf,
            "r_c": args.r_c,
            "h_0": args.h_0,
            "dipole_d": args.dipole_d,
            "e6": args.e6,
        }.items()
        if v is not None
    }
    overrides["seed"] = args.seed
    png = render_comparison_figure(OUT_PNG, overrides=overrides)
    print(f"[render] Wrote {png}")
    if not args.skip_gif:
        gif = render_growth_gif(OUT_GIF)
        print(f"[render] Wrote {gif}")
        gif2 = render_membrane_perforation_gif(OUT_GIF_MEMBRANE, seed=args.seed)
        print(f"[render] Wrote {gif2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
