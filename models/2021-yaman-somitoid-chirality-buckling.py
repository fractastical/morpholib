#!/usr/bin/env python3
"""
2021-yaman-somitoid-chirality-buckling.py — Epithelial Somite Rosette Vertex Mechanics & Mesenchymal-to-Epithelial Transition (MET)
===================================================================================================================================

Faithful mathematical, numerical, and morphometric reproduction of:
  1. Yamanaka, Y., Hamidi, S., Yoshioka-Kobayashi, K., Munira, S., Sunadome, K.,
     Zhang, Y., Kurokawa, Y., Ericsson, R., Mieda, A., Thompson, J. L., Kerwin, J.,
     Lisgo, S., Yamamoto, T., Moris, N., Martinez-Arias, A., Tsujimura, T. & Alev, C. (2023).
     "Reconstituting human somitogenesis in vitro."
     Nature 614(7948):509-520. DOI: 10.1038/s41586-022-05649-2
  2. Miao, Y., Djeffal, Y., De Simone, A., Zhu, K., Lee, J. G., Lu, Z.,
     Silberfeld, A., Rao, J., Tarazona, O. A., Reiter, A. & Pourquie, O. (2023).
     "Reconstruction and deconstruction of human somitogenesis in vitro."
     Nature 614(7948):500-508. DOI: 10.1038/s41586-022-05655-4
  3. Adhyapok, P., Piatkowska, A. M., Norman, M. J., Clendenon, S. G.,
     Stern, C. D., Glazier, J. A. & Belmonte, J. M. (2021).
     "A mechanical model of early somite segmentation."
     Developmental Cell 56(9):1312-1324. DOI: 10.1016/j.devcel.2021.03.026
  4. Sanaki-Matsumiya, M., Matsuda, M., Gritti, N., Nakaki, F., Sharpe, J.,
     Trivedi, V. & Ebisuya, M. (2022).
     "Periodic formation of epithelial somites from human pluripotent stem cells."
     Nature Communications 13(1):2325. DOI: 10.1038/s41467-022-29967-1
  5. Lewis, J. (2003). "Autoinhibition with transcriptional delay: a simple mechanism
     for the zebrafish somitogenesis oscillator." Current Biology 13(16):1398-1408.
     DOI: 10.1016/S0960-9822(03)00534-7              (HES7 delayed-oscillator kernel, Pass 3)
  6. Diaz-Cuadros, M. et al. (2020). "In vitro characterization of the human segmentation
     clock." Nature 580(7801):113-118. DOI: 10.1038/s41586-019-1885-9   (human HES7 period ~5 h)
  7. Matsuda, M. et al. (2020). "Recapitulating the human segmentation clock with
     pluripotent stem cells." Nature 580(7801):124-129. DOI: 10.1038/s41586-020-2144-9
  8. Cooke, J. & Zeeman, E. C. (1976). "A clock and wavefront model for control of the
     number of repeated structures during animal morphogenesis." Journal of Theoretical
     Biology 58(2):455-476. DOI: 10.1016/S0022-5193(76)80131-2
  9. Gomez-Galvez, P. et al. (2018). "Scutoids are a geometrical solution to
     three-dimensional packing of epithelia." Nature Communications 9:2960.
     DOI: 10.1038/s41467-018-05376-1                 (3D polyhedral shell / scutoid audit, Pass 3)

Pass 3 remediation (superseding audit `2026-10-19-somitoid-chirality-buckling-audit-v2`):
------------------------------------------------------------------------------------------
  * v2 item 1 ("extend the 2D wedge rosette to a 3D apical-basal polyhedral scutoid shell to
    capture 3D somitecoel lumen pressurization") -> Section 2b. The rosette is now ALSO solved
    as a closed 3D polyhedral shell of N_3D = 96 radial frustum cells whose apical and basal
    faces are the spherical-Voronoi polygons of a Fibonacci-sphere seed set (48 hexagons,
    30 pentagons, 18 heptagons), with per-cell volume elasticity, apical/basal/lateral face
    tensions and an explicit somitecoel lumen pressure term -P * V_lumen.  REDUCED, STATED
    HONESTLY: the cells are radial prismatoids (frusta) whose apical and basal neighbour
    graphs are identical, so the shell contains 0 scutoids by construction; the
    Gomez-Galvez (2018) Voronoi argument predicts exactly that for a sphere (scutoids need
    anisotropic curvature), and `compute_scutoid_topology_audit` reports the apical/basal
    neighbour-graph difference (0) instead of asserting scutoids that the model cannot form.
  * v2 item 2 ("couple MESP2/RIPPLY2 wavefront arrest from a live HES7 delayed oscillator into
    the spatiotemporal activation of EphA4/ephrinB2 and RhoA/ROCK line tensions") ->
    Section 3b. A Lewis (2003) two-variable transcriptional-delay HES7 oscillator (time-scaled
    to the ~5 h human period of Diaz-Cuadros 2020 / Matsuda 2020) runs in every PSM cell; a
    Cooke-Zeeman determination wavefront arrests each cell's oscillator as it passes; the
    arrested HES7 phase gates MESP2, RIPPLY2 switches MESP2 off and fixes the boundary, and
    the boundary / arrest times drive exponential ramps of gamma_het (EphA4/ephrinB2) and
    gamma_ap (RhoA/ROCK) that are fed LIVE into the 2D vertex solver and the 3D shell law.

Governing Mathematical Laws (verified in `morphoeval/lean/loops/Loop_029_somitoid_chirality_buckling.lean`):
------------------------------------------------------------------------------------------------------------
1. Apico-Basal Differential Interfacial Tension Vertex Mechanics & Somitecoel Lumen Equilibrium Law
   (`yamanaka_adhyapok_wedge_rosette_tapering`):
       E = sum_{i=1}^N [ (K_A / 2) * (A_i - A_0)^2 + gamma_ap * L_{ap,i} + gamma_lat * L_{lat,i} + gamma_bas * L_{bas,i} ]
   For a radial somite rosette of N wedge-shaped epithelial cells around a central somitecoel lumen
   of radius R_in and outer basal lamina radius R_out = sqrt(R_in^2 + N * A_0 / pi):
       E(R_in, N) = 2 * pi * gamma_ap * R_in + 2 * pi * gamma_bas * R_out + N * gamma_lat * (R_out - R_in)
   Minimizing E(R_in, N) with respect to R_in yields the exact equilibrium apical-to-basal ratio:
       R_in^* / R_out^* = L_ap^* / L_bas^* = (N * gamma_lat - 2 * pi * gamma_ap) / (N * gamma_lat + 2 * pi * gamma_bas)
   with critical lumen-closure tension gamma_{ap,crit} = (N / (2 * pi)) * gamma_lat.

2. EphA4 / ephrinB2 Inter-Somite Cleft Boundary Sorting Energy Barrier Law
   (`adhyapok_2021_eph_ephrin_cleft_sorting_barrier`):
       Delta_E_cleft = 2 * L_cleft * (gamma_het - (gamma_ant + gamma_post) / 2) > 0
   Following MESP2/RIPPLY2 wavefront arrest, heterotypic EphA4/ephrinB2 contact repulsion elevates
   inter-somitic line tension gamma_het above homotypic N-cadherin (CDH2) adhesion, driving physical
   cleft separation and rounding of adjacent anterior and posterior somite rosettes.

3. Y-27632 ROCK Inhibition & Minus-Matrigel Apico-Basal Tension Collapse Law
   (`yamanaka_2023_rock_inhibition_curvature_collapse`):
       gamma_ap = gamma_bas  =>  Delta_gamma_{ap-bas} = gamma_ap - gamma_bas = 0
   Inhibiting Rho-kinase (Y-27632) or omitting Matrigel extracellular matrix abolishes the apico-basal
   tension differential, collapsing spontaneous wedge curvature and preventing polarized somite rosette
   and central somitecoel lumen formation.

4. 3D Polyhedral Shell Somitecoel Lumen Pressurization Law  (Pass 3, `yamanaka_2023_shell_lumen_pressurization`):
       E_3D = sum_i [ (K_V / 2) (V_i - V_0)^2 + gamma_ap A_{ap,i} + gamma_bas A_{bas,i} ]
              + gamma_lat * sum_{faces ij} A_{lat,ij}  -  P * V_lumen
   For a closed spherical shell of N_3D radial frustum cells with total angular edge length
   sum_ij l_ij on the unit sphere, Lambda = gamma_lat * sum_ij l_ij / 2 and
   R_out^3 = R_in^3 + 3 N_3D V_0 / (4 pi), the radial balance dE_3D/dR_in = 0 gives
       rho_3D = R_in / R_out = (Lambda - 4 pi gamma_ap + 2 pi P R_in) / (Lambda + 4 pi gamma_bas)
   so lumen pressure P > 0 strictly raises the equilibrium numerator (lumen dilation), and the
   shell stays closed (rho_3D < 1) iff 2 pi P R_in < 4 pi (gamma_ap + gamma_bas), i.e. the
   two-surface Laplace bound P < 2 (gamma_ap + gamma_bas) / R_in.

5. HES7 Clock-Wavefront Arrest & Tension Maturation Gradient Law  (Pass 3, `hes7_wavefront_arrest_tension_gradient`):
       dm/dt = k / (1 + (p(t - T_m) / p_0)^2) - c m,     dp/dt = a m(t - T_p) - b p      (Lewis 2003)
       x_wf(t) = v t,  cell j arrested at t_a,j = x_j / v,  segment length S = v T_HES7
       gamma_ap,j(t) = gamma_bas + (gamma_ap^WT - gamma_bas) (1 - exp(-(t - t_a,j)/tau_ROCK))
       gamma_het,k(t) = gamma_hom + (gamma_het^WT - gamma_hom) (1 - exp(-(t - t_det,k)/tau_Eph))
   Segment length grows linearly with the clock period at fixed wavefront speed, and for any
   two arrested cells the anterior (earlier-arrested) one carries the strictly larger
   RhoA/ROCK apical tension at every later time: an anterior-to-posterior MET maturation gradient.
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import optimize, signal, stats
from scipy.spatial import SphericalVoronoi

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "somitoid_chirality_buckling"
REFERENCE_JSON = DATASET_DIR / "somitoid_chirality_buckling_reference.json"
SPECIMEN_JPG = DATASET_DIR / "somitoid_rosette_specimen.jpg"
SCOUT_MANIFEST_JSON = DATASET_DIR / "scout_manifest.json"
OUTPUT_PNG = ROOT / "docs" / "media" / "somitoid_chirality_buckling_comparison.png"
OUTPUT_GIF = ROOT / "docs" / "media" / "somitoid_chirality_buckling_growth.gif"
OUTPUT_CLOCK_PNG = ROOT / "docs" / "media" / "somitoid_chirality_buckling_clock_coupling.png"

EXPECTED_SHA256 = {
    "datasets/somitoid_chirality_buckling/somitoid_rosette_specimen.jpg": (
        "8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f"
    ),
    "datasets/somitoid_chirality_buckling/scout_manifest.json": (
        "8326704c93aa8a75c90977f41fe3e00040bc848a2cceabe362af10fed750b806"
    ),
}
# Pass 3: the reference JSON is now SHA-256 pinned as well (v2 only hashed it without checking).
EXPECTED_REFERENCE_JSON_SHA256 = "72127bb5b83f59d0a76440ea7730e368fc6e27437f0dc5a71c89fcde68164c16"

# Pass 3 3D polyhedral shell defaults (Section 2b)
N3D_CELLS = 96               # frustum cells on the closed shell (Fibonacci-sphere Voronoi seeds)
V0_CELL_UM3 = 700.0          # target cell volume (um^3): ~ (8.9 um)^3, matches the 2D A0 = 115 um^2 x 6 um
K_V_VOLUME = 0.05            # volume elastic modulus (gamma_lat / um^5)

# Pass 3 HES7 clock-wavefront defaults (Section 3b); tensions are in units of gamma_lat
HES7_TARGET_PERIOD_H = 5.0   # Diaz-Cuadros et al. 2020; Matsuda et al. 2020: human HES7 period ~ 5 h
PSM_CELLS = 64               # 1D anteroposterior PSM cell column
PSM_CELL_DX_UM = 10.0        # cell diameter along the AP axis
WAVEFRONT_SPEED_UM_PER_H = 20.0   # determination-front regression speed -> S = v T = 100 um
TAU_ROCK_H = 3.0             # RhoA/ROCK apical-tension maturation time constant after arrest
TAU_EPH_H = 1.5              # EphA4/ephrinB2 boundary tension time constant after RIPPLY2 fixation
TAU_RIPPLY2_H = 1.0          # MESP2 -> RIPPLY2 induction delay (switches MESP2 off, fixes the boundary)


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
    ref_digest = hashlib.sha256(REFERENCE_JSON.read_bytes()).hexdigest()
    if ref_digest != EXPECTED_REFERENCE_JSON_SHA256:
        raise ValueError(
            "SHA-256 mismatch for somitoid_chirality_buckling_reference.json: "
            f"expected {EXPECTED_REFERENCE_JSON_SHA256}, got {ref_digest}"
        )
    digests["datasets/somitoid_chirality_buckling/somitoid_chirality_buckling_reference.json"] = ref_digest
    return digests


def load_reference_data() -> dict[str, Any]:
    """Load `datasets/somitoid_chirality_buckling/somitoid_chirality_buckling_reference.json`."""
    return json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))


# ============================================================================
# Section 2: 2D Apico-Basal Differential Interfacial Tension Vertex Solver
# (Yamanaka et al. 2023 Nature Figs. 1-3; Adhyapok et al. 2021 Dev. Cell Figs. 2-4;
#  Law 1 `yaman-somitoid-chirality-buckling-1`)
# ============================================================================
def analytical_rosette_equilibrium(
    N: int = 16,
    A0: float = 115.0,
    gamma_ap: float = 1.65,
    gamma_bas: float = 0.45,
    gamma_lat: float = 1.0,
) -> dict[str, float]:
    """Compute exact analytical continuum-ring equilibrium for an N-cell wedge
    rosette minimizing:
        E(R_in) = 2*pi*gamma_ap*R_in + 2*pi*gamma_bas*R_out + N*gamma_lat*(R_out - R_in)
    subject to total epithelial area pi*(R_out^2 - R_in^2) = N * A0.
    """
    theta_wedge = 2.0 * math.pi / N
    half_sin = math.sin(theta_wedge / 2.0)
    # Exact discrete N-gon chord factor: L_ap = 2*R_in*sin(pi/N), A_i = 0.5*sin(2*pi/N)*(R_out^2 - R_in^2)
    c_ap = 2.0 * N * half_sin * gamma_ap - N * gamma_lat
    c_bas = 2.0 * N * half_sin * gamma_bas + N * gamma_lat
    if c_ap >= 0.0:
        ratio_polygon = 0.0
    else:
        ratio_polygon = min(max(-c_ap / c_bas, 0.02), 0.98)

    # Continuum circular-arc formula:
    num_cont = max(N * gamma_lat - 2.0 * math.pi * gamma_ap, 0.0)
    den_cont = N * gamma_lat + 2.0 * math.pi * gamma_bas
    ratio_continuum = min(max(num_cont / den_cont, 0.02), 0.98)

    # Area of discrete N-wedge polygon ring: N * 0.5 * sin(2*pi/N) * (R_out^2 - R_in^2) = N * A0
    r_out_poly = math.sqrt(
        (2.0 * A0) / (math.sin(theta_wedge) * max(1.0 - ratio_polygon**2, 1e-6))
    )
    r_in_poly = ratio_polygon * r_out_poly

    r_out_cont = math.sqrt(
        (N * A0) / (math.pi * max(1.0 - ratio_continuum**2, 1e-6))
    )
    r_in_cont = ratio_continuum * r_out_cont

    gamma_ap_crit = (N / (2.0 * math.pi)) * gamma_lat
    return {
        "ratio_continuum": float(ratio_continuum),
        "ratio_polygon": float(ratio_polygon),
        "r_in_continuum": float(r_in_cont),
        "r_out_continuum": float(r_out_cont),
        "r_in_polygon": float(r_in_poly),
        "r_out_polygon": float(r_out_poly),
        "gamma_ap_crit": float(gamma_ap_crit),
        "theta_wedge_rad": float(theta_wedge),
    }


def solve_2d_vertex_rosette(
    N: int = 16,
    A0: float = 115.0,
    K_A: float = 0.08,
    gamma_ap: float = 1.65,
    gamma_bas: float = 0.45,
    gamma_lat: float = 1.0,
    radial_perturbation: float = 0.0,
    seed: int = 2021,
) -> dict[str, Any]:
    """Relax a full 2D 2N-vertex epithelial somite rosette minimizing the
    Apico-Basal Differential Interfacial Tension Vertex Energy Functional:
        E = sum_{i=0}^{N-1} [ 0.5 * K_A * (A_i - A0)^2
                              + gamma_ap * L_{ap,i}
                              + gamma_bas * L_{bas,i}
                              + gamma_lat * L_{lat,i} ]
    where each cell i is a 4-vertex wedge polygon with inner apical vertices
    v_{ap,i}, v_{ap,i+1} and outer basal vertices v_{bas,i+1}, v_{bas,i}.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
    r_in_init = 11.0 * (1.0 + radial_perturbation * rng.normal(0.0, 1.0, size=N))
    r_out_init = 26.0 * (1.0 + radial_perturbation * rng.normal(0.0, 1.0, size=N))
    x0 = np.concatenate([r_in_init, r_out_init])

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    def vertex_energy(radii: np.ndarray) -> float:
        r_in = np.maximum(radii[:N], 0.25)
        r_out = np.maximum(radii[N:], r_in + 0.5)
        x_ap = r_in * cos_a
        y_ap = r_in * sin_a
        x_bas = r_out * cos_a
        y_bas = r_out * sin_a

        x_ap_next = np.roll(x_ap, -1)
        y_ap_next = np.roll(y_ap, -1)
        x_bas_next = np.roll(x_bas, -1)
        y_bas_next = np.roll(y_bas, -1)

        l_ap = np.hypot(x_ap_next - x_ap, y_ap_next - y_ap)
        l_bas = np.hypot(x_bas_next - x_bas, y_bas_next - y_bas)
        l_lat = np.hypot(x_bas - x_ap, y_bas - y_ap)

        # Shoelace area of quadrilateral (ap_i -> bas_i -> bas_{i+1} -> ap_{i+1})
        area = 0.5 * np.abs(
            (x_ap * y_bas - y_ap * x_bas)
            + (x_bas * y_bas_next - y_bas * x_bas_next)
            + (x_bas_next * y_ap_next - y_bas_next * x_ap_next)
            + (x_ap_next * y_ap - y_ap_next * x_ap)
        )

        e_area = 0.5 * K_A * np.sum((area - A0) ** 2)
        e_ap = gamma_ap * np.sum(l_ap)
        e_bas = gamma_bas * np.sum(l_bas)
        e_lat = gamma_lat * np.sum(l_lat)
        return float(e_area + e_ap + e_bas + e_lat)

    bounds = [(0.5, 45.0)] * N + [(3.0, 65.0)] * N
    res = optimize.minimize(
        vertex_energy,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 600},
    )

    r_in_opt = res.x[:N]
    r_out_opt = res.x[N:]
    x_ap = r_in_opt * cos_a
    y_ap = r_in_opt * sin_a
    x_bas = r_out_opt * cos_a
    y_bas = r_out_opt * sin_a

    l_ap = np.hypot(np.roll(x_ap, -1) - x_ap, np.roll(y_ap, -1) - y_ap)
    l_bas = np.hypot(np.roll(x_bas, -1) - x_bas, np.roll(y_bas, -1) - y_bas)
    l_lat = np.hypot(x_bas - x_ap, y_bas - y_ap)

    mean_r_in = float(np.mean(r_in_opt))
    mean_r_out = float(np.mean(r_out_opt))
    ap_basal_ratio = float(np.mean(l_ap) / np.mean(l_bas))
    wedge_height = float(np.mean(l_lat))
    apico_basal_polarity = float(
        np.clip(
            (1.0 - ap_basal_ratio) * (1.0 + 0.45 * max(gamma_ap - gamma_bas, 0.0)),
            0.0,
            1.0,
        )
    )

    polygons: list[np.ndarray] = []
    for i in range(N):
        j = (i + 1) % N
        poly = np.array(
            [
                [x_ap[i], y_ap[i]],
                [x_bas[i], y_bas[i]],
                [x_bas[j], y_bas[j]],
                [x_ap[j], y_ap[j]],
            ],
            dtype=float,
        )
        polygons.append(poly)

    analytic = analytical_rosette_equilibrium(
        N=N, A0=A0, gamma_ap=gamma_ap, gamma_bas=gamma_bas, gamma_lat=gamma_lat
    )

    return {
        "N": N,
        "gamma_ap": gamma_ap,
        "gamma_bas": gamma_bas,
        "gamma_lat": gamma_lat,
        "energy_min": float(res.fun),
        "r_in_vertices": r_in_opt,
        "r_out_vertices": r_out_opt,
        "mean_r_in_um": mean_r_in,
        "mean_r_out_um": mean_r_out,
        "mean_l_ap_um": float(np.mean(l_ap)),
        "mean_l_bas_um": float(np.mean(l_bas)),
        "wedge_height_um": wedge_height,
        "vertex_ap_basal_ratio": ap_basal_ratio,
        "analytic_polygon_ratio": analytic["ratio_polygon"],
        "analytic_continuum_ratio": analytic["ratio_continuum"],
        "apico_basal_polarity": apico_basal_polarity,
        "polygons": polygons,
    }


def compute_rosette_energy_landscape(
    N: int = 16,
    A0: float = 115.0,
) -> dict[str, Any]:
    """Compute E(R_in, N) energy curves across WT (+Matrigel), Calpeptin
    hyper-constricted, Y-27632 ROCK-inhibited, and Minus-Matrigel regimes.
    """
    r_in = np.linspace(0.5, 28.0, 300)
    r_out = np.sqrt(r_in**2 + (N * A0) / math.pi)
    curves: dict[str, Any] = {"r_in": r_in, "r_out": r_out, "regimes": {}}
    configs = [
        ("wt_plus_matrigel", "WT (+10% Matrigel)", 1.65, 0.45, 1.0),
        ("high_myosin", "Calpeptin Hyper-Apical", 2.15, 0.45, 1.0),
        ("low_matrigel", "Low Matrigel (2.5%)", 1.35, 0.95, 1.0),
        ("rock_y27632", "Y-27632 ROCK Inh.", 0.45, 0.45, 1.0),
        ("minus_matrigel", "Minus-Matrigel (0%)", 0.72, 0.72, 1.0),
    ]
    for key, label, g_ap, g_bas, g_lat in configs:
        e = (
            2.0 * math.pi * g_ap * r_in
            + 2.0 * math.pi * g_bas * r_out
            + N * g_lat * (r_out - r_in)
        )
        idx = int(np.argmin(e))
        curves["regimes"][key] = {
            "label": label,
            "energy": e,
            "r_in_opt": float(r_in[idx]),
            "r_out_opt": float(r_out[idx]),
            "ratio_opt": float(r_in[idx] / r_out[idx]),
        }
    return curves


# ============================================================================
# Section 2b (Pass 3): 3D Apical-Basal Polyhedral Shell with Somitecoel Lumen
# Pressurization  (Law 4 `yaman-somitoid-chirality-buckling-4`;
# Yamanaka 2023 Figs. 2-3 somitecoel lumen; Gomez-Galvez 2018 scutoid packing)
# ============================================================================
def build_spherical_voronoi_shell(N3: int = N3D_CELLS) -> dict[str, Any]:
    """Build the angular skeleton of a closed polyhedral epithelial shell.

    N3 deterministic Fibonacci-sphere seed directions are tessellated with a
    spherical Voronoi diagram; each Voronoi polygon is the angular footprint of
    one radial frustum cell (apical face at R_in,i, basal face at R_out,i).
    Returns per-cell solid angles Omega_i (sum = 4 pi), the shared-face list
    (i, j, l_ij) with great-circle edge lengths l_ij on the unit sphere, and
    the polygon vertex lists used for rendering.
    """
    idx = np.arange(N3, dtype=float) + 0.5
    polar = np.arccos(1.0 - 2.0 * idx / N3)
    azim = math.pi * (1.0 + math.sqrt(5.0)) * idx
    pts = np.stack(
        [np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)],
        axis=1,
    )
    sv = SphericalVoronoi(pts, radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()
    omega = np.asarray(sv.calculate_areas(), dtype=float)

    edge_map: dict[tuple[int, int], list[int]] = {}
    for ci, region in enumerate(sv.regions):
        m = len(region)
        for k in range(m):
            a, b = region[k], region[(k + 1) % m]
            edge_map.setdefault((min(a, b), max(a, b)), []).append(ci)
    face_i: list[int] = []
    face_j: list[int] = []
    face_len: list[float] = []
    for (a, b), cells in sorted(edge_map.items()):
        if len(cells) != 2:
            continue
        va, vb = sv.vertices[a], sv.vertices[b]
        arc = math.acos(max(-1.0, min(1.0, float(np.dot(va, vb)))))
        face_i.append(cells[0])
        face_j.append(cells[1])
        face_len.append(arc)

    sides = np.array([len(r) for r in sv.regions], dtype=int)
    return {
        "N3": N3,
        "seed_dirs": pts,
        "omega": omega,
        "face_i": np.array(face_i, dtype=int),
        "face_j": np.array(face_j, dtype=int),
        "face_len": np.array(face_len, dtype=float),
        "regions": [list(r) for r in sv.regions],
        "unit_vertices": np.asarray(sv.vertices, dtype=float),
        "n_faces": int(len(face_len)),
        "total_angular_edge_length": float(np.sum(face_len)),
        "polygon_side_counts": sides,
        "n_pentagons": int(np.sum(sides == 5)),
        "n_hexagons": int(np.sum(sides == 6)),
        "n_heptagons": int(np.sum(sides == 7)),
    }


def analytical_shell_equilibrium(
    shell: dict[str, Any],
    V0: float = V0_CELL_UM3,
    gamma_ap: float = 1.65,
    gamma_bas: float = 0.45,
    gamma_lat: float = 1.0,
    P_lumen: float = 0.0,
) -> dict[str, float]:
    """Law 4 radial balance for the uniform closed shell:
        E_3D(R_in) = 4 pi gamma_ap R_in^2 + 4 pi gamma_bas R_out^2
                     + Lambda (R_out^2 - R_in^2) - P (4 pi / 3) R_in^3,
        R_out^3 = R_in^3 + 3 N_3D V_0 / (4 pi),   Lambda = gamma_lat * sum_ij l_ij / 2
    giving rho_3D = (Lambda - 4 pi gamma_ap + 2 pi P R_in) / (Lambda + 4 pi gamma_bas).
    P = 0 is closed-form; for P > 0 the (local) minimum is found numerically and
    `ruptured` is set when the pressurized lumen has no finite minimizer below
    the search cap (lumen blow-out past the saddle-node).
    """
    lam = gamma_lat * shell["total_angular_edge_length"] / 2.0
    v_tot = shell["N3"] * V0
    shell_const = 3.0 * v_tot / (4.0 * math.pi)

    def r_out_of(r_in: float) -> float:
        return (r_in**3 + shell_const) ** (1.0 / 3.0)

    def energy(r_in: float) -> float:
        r_out = r_out_of(r_in)
        return (
            4.0 * math.pi * gamma_ap * r_in**2
            + 4.0 * math.pi * gamma_bas * r_out**2
            + lam * (r_out**2 - r_in**2)
            - P_lumen * (4.0 * math.pi / 3.0) * r_in**3
        )

    rho_closed_form = min(max((lam - 4.0 * math.pi * gamma_ap) / (lam + 4.0 * math.pi * gamma_bas), 0.0), 0.98)
    r_cap = 45.0
    res = optimize.minimize_scalar(energy, bounds=(0.3, r_cap), method="bounded", options={"xatol": 1e-6})
    r_in = float(res.x)
    r_out = r_out_of(r_in)
    ruptured = bool(r_in > 0.98 * r_cap)
    rho = r_in / r_out
    # Pressure predicted by Law 4 at this geometry (= P_lumen at an interior minimum)
    p_balance = (4.0 * math.pi * gamma_ap - lam + rho * (4.0 * math.pi * gamma_bas + lam)) / (2.0 * math.pi * r_in)
    laplace_bound = 2.0 * (gamma_ap + gamma_bas) / r_in
    return {
        "lambda_lat": float(lam),
        "rho_closed_form_p0": float(rho_closed_form),
        "rho_3d": float(rho),
        "r_in_um": float(r_in),
        "r_out_um": float(r_out),
        "energy_min": float(res.fun),
        "ruptured": ruptured,
        "pressure_balance": float(p_balance),
        "laplace_open_bound": float(laplace_bound),
        "lumen_volume_um3": float(4.0 * math.pi / 3.0 * r_in**3),
    }


def solve_3d_polyhedral_shell(
    shell: dict[str, Any],
    V0: float = V0_CELL_UM3,
    K_V: float = K_V_VOLUME,
    gamma_ap: float = 1.65,
    gamma_bas: float = 0.45,
    gamma_lat: float = 1.0,
    P_lumen: float = 0.0,
    radial_perturbation: float = 0.0,
    seed: int = 2021,
) -> dict[str, Any]:
    """Relax the closed 3D polyhedral shell (2 N_3D radial degrees of freedom:
    R_in,i and R_out,i of every frustum cell) minimizing
        E_3D = sum_i [ 0.5 K_V (V_i - V_0)^2 + gamma_ap Omega_i R_in,i^2 + gamma_bas Omega_i R_out,i^2 ]
               + gamma_lat sum_faces l_ij * 0.5 * (Rbar_out,ij^2 - Rbar_in,ij^2)  -  P * sum_i Omega_i R_in,i^3 / 3
    with V_i = Omega_i (R_out,i^3 - R_in,i^3) / 3 and Rbar the face-averaged radii.
    This is the 3D analogue of `solve_2d_vertex_rosette` (same L-BFGS-B relaxation,
    same determinism contract).
    """
    N3 = shell["N3"]
    omega = shell["omega"]
    fi = shell["face_i"]
    fj = shell["face_j"]
    ell = shell["face_len"]
    rng = np.random.default_rng(seed)
    r_in0 = 10.0 * (1.0 + radial_perturbation * rng.normal(0.0, 1.0, size=N3))
    r_out0 = 26.0 * (1.0 + radial_perturbation * rng.normal(0.0, 1.0, size=N3))
    x0 = np.concatenate([r_in0, r_out0])

    def shell_energy(x: np.ndarray) -> float:
        r_in = x[:N3]
        r_out = x[N3:]
        vol = omega * (r_out**3 - r_in**3) / 3.0
        a_ap = omega * r_in**2
        a_bas = omega * r_out**2
        rin_f = 0.5 * (r_in[fi] + r_in[fj])
        rout_f = 0.5 * (r_out[fi] + r_out[fj])
        a_lat = ell * 0.5 * (rout_f**2 - rin_f**2)
        v_lumen = float(np.sum(omega * r_in**3) / 3.0)
        return float(
            0.5 * K_V * np.sum((vol - V0) ** 2)
            + gamma_ap * np.sum(a_ap)
            + gamma_bas * np.sum(a_bas)
            + gamma_lat * np.sum(a_lat)
            - P_lumen * v_lumen
        )

    bounds = [(0.5, 45.0)] * N3 + [(3.0, 65.0)] * N3
    res = optimize.minimize(
        shell_energy,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-11, "gtol": 1e-9, "maxiter": 1500, "maxfun": 200000},
    )
    r_in = res.x[:N3]
    r_out = res.x[N3:]
    vol = omega * (r_out**3 - r_in**3) / 3.0
    a_ap = omega * r_in**2
    a_bas = omega * r_out**2
    rin_f = 0.5 * (r_in[fi] + r_in[fj])
    rout_f = 0.5 * (r_out[fi] + r_out[fj])
    a_lat = ell * 0.5 * (rout_f**2 - rin_f**2)
    v_lumen = float(np.sum(omega * r_in**3) / 3.0)
    analytic = analytical_shell_equilibrium(
        shell, V0=V0, gamma_ap=gamma_ap, gamma_bas=gamma_bas, gamma_lat=gamma_lat, P_lumen=P_lumen
    )
    ratio = float(np.mean(r_in) / np.mean(r_out))
    return {
        "N3": N3,
        "gamma_ap": gamma_ap,
        "gamma_bas": gamma_bas,
        "gamma_lat": gamma_lat,
        "P_lumen": P_lumen,
        "converged": bool(res.success),
        "energy_min": float(res.fun),
        "r_in_cells": r_in,
        "r_out_cells": r_out,
        "mean_r_in_um": float(np.mean(r_in)),
        "mean_r_out_um": float(np.mean(r_out)),
        "shell_ratio_3d": ratio,
        "analytic_ratio_3d": analytic["rho_3d"],
        "analytic_ratio_3d_p0": analytic["rho_closed_form_p0"],
        "analytic_ruptured": analytic["ruptured"],
        "mean_cell_volume_um3": float(np.mean(vol)),
        "cell_volume_cv": float(np.std(vol) / np.mean(vol)),
        "apical_area_total_um2": float(np.sum(a_ap)),
        "basal_area_total_um2": float(np.sum(a_bas)),
        "lateral_area_total_um2": float(np.sum(a_lat)),
        "lumen_volume_um3": v_lumen,
        "apical_to_basal_area_ratio": float(np.sum(a_ap) / np.sum(a_bas)),
        "ruptured": bool(np.mean(r_in) > 0.98 * 45.0),
    }


def sweep_lumen_pressure(
    shell: dict[str, Any],
    gamma_ap: float = 1.65,
    gamma_bas: float = 0.45,
    gamma_lat: float = 1.0,
    n_p: int = 25,
    p_max: float = 0.16,
    discrete_at: tuple[float, ...] = (0.0, 0.04, 0.08),
) -> dict[str, Any]:
    """Sweep somitecoel lumen pressure P (units gamma_lat / um) through Law 4,
    locating the saddle-node rupture pressure P_crit beyond which the shell has
    no finite closed equilibrium, and cross-check selected points against the
    discrete 2 N_3D-DOF polyhedral relaxation.
    """
    p_grid = np.linspace(0.0, p_max, n_p)
    rho = np.zeros(n_p)
    r_in = np.zeros(n_p)
    ruptured = np.zeros(n_p, dtype=bool)
    for k, p in enumerate(p_grid):
        eq = analytical_shell_equilibrium(shell, gamma_ap=gamma_ap, gamma_bas=gamma_bas, gamma_lat=gamma_lat, P_lumen=float(p))
        rho[k] = eq["rho_3d"]
        r_in[k] = eq["r_in_um"]
        ruptured[k] = eq["ruptured"]

    # Bisection for the rupture (saddle-node) pressure
    lo, hi = 0.0, p_max
    if not ruptured.any():
        p_crit = float("nan")
    else:
        hi = float(p_grid[np.argmax(ruptured)])
        lo = float(p_grid[max(np.argmax(ruptured) - 1, 0)])
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if analytical_shell_equilibrium(shell, gamma_ap=gamma_ap, gamma_bas=gamma_bas, gamma_lat=gamma_lat, P_lumen=mid)["ruptured"]:
                hi = mid
            else:
                lo = mid
        p_crit = 0.5 * (lo + hi)

    discrete: list[dict[str, float]] = []
    for p in discrete_at:
        sol = solve_3d_polyhedral_shell(shell, gamma_ap=gamma_ap, gamma_bas=gamma_bas, gamma_lat=gamma_lat, P_lumen=float(p))
        discrete.append({"P": float(p), "ratio_discrete": sol["shell_ratio_3d"], "ratio_analytic": sol["analytic_ratio_3d"], "r_in_um": sol["mean_r_in_um"]})

    eq0 = analytical_shell_equilibrium(shell, gamma_ap=gamma_ap, gamma_bas=gamma_bas, gamma_lat=gamma_lat, P_lumen=0.0)
    return {
        "p_grid": p_grid,
        "rho": rho,
        "r_in_um": r_in,
        "ruptured": ruptured,
        "p_crit": float(p_crit),
        "laplace_open_bound_p0": eq0["laplace_open_bound"],
        "discrete": discrete,
        "monotone_increasing": bool(np.all(np.diff(rho[~ruptured]) > -1e-9)),
    }


def compute_scutoid_topology_audit(shell: dict[str, Any], sol: dict[str, Any]) -> dict[str, Any]:
    """Honest scutoid audit (Gomez-Galvez et al. 2018 Nat Commun 9:2960).

    A scutoid is a cell whose apical and basal neighbour sets differ. In this
    radial-frustum shell every cell's apical and basal faces are the same
    Voronoi polygon scaled by R_in,i and R_out,i, so the apical and basal
    neighbour graphs coincide and the scutoid fraction is exactly 0; the
    Gomez-Galvez Voronoi construction predicts the same for an isotropically
    curved (spherical) epithelium, where scutoids arise only under anisotropic
    curvature (tubes) or lateral cell rearrangement, which this model omits.
    """
    fi, fj = shell["face_i"], shell["face_j"]
    apical_graph = {tuple(sorted((int(a), int(b)))) for a, b in zip(fi, fj)}
    basal_graph = {tuple(sorted((int(a), int(b)))) for a, b in zip(fi, fj)}  # same polygon, scaled
    sym_diff = apical_graph ^ basal_graph
    # Curvature anisotropy of the relaxed shell (sphere -> 0): principal radii ratio
    r_out = sol["r_out_cells"]
    anisotropy = float((np.max(r_out) - np.min(r_out)) / np.mean(r_out))
    return {
        "n_cells": shell["N3"],
        "n_lateral_faces": shell["n_faces"],
        "apical_basal_neighbour_graph_difference": int(len(sym_diff)),
        "scutoid_fraction": float(len(sym_diff) / max(shell["N3"], 1)),
        "basal_radius_anisotropy": anisotropy,
        "mean_sides": float(np.mean(shell["polygon_side_counts"])),
        "statement": "0 scutoids by construction (radial frusta); consistent with Gomez-Galvez 2018 for spherical curvature; scutoid formation under anisotropic (tubular) curvature NOT modelled",
    }


# ============================================================================
# Section 3: EphA4/ephrinB2 Inter-Somite Cleft Boundary Separation &
#            6-Regime Experimental Perturbation Series
# (Adhyapok et al. 2021 Dev. Cell Figs. 1-3; Yamanaka et al. 2023 Nature Fig. 2;
#  Miao et al. 2023 Nature Fig. 4; Law 2 & Law 3)
# ============================================================================
def compute_eph_ephrin_cleft_mechanics(
    gamma_het: float = 2.35,
    gamma_ant: float = 1.0,
    gamma_post: float = 1.0,
    L_cleft_um: float = 28.0,
    rosette_radius_um: float = 25.4,
) -> dict[str, float]:
    """Compute the EphA4/ephrinB2 heterotypic boundary sorting energy barrier
    and Young-Dupre inter-somitic cleft indentation depth between anterior (S_I)
    and posterior (S_0) somite rosettes:
        Delta_E_cleft = L_cleft * (2 * gamma_het - (gamma_ant + gamma_post))
        cos(theta_cleft / 2) = clip((gamma_ant + gamma_post) / (2 * gamma_het), 0, 1)
    """
    gamma_hom_mean = 0.5 * (gamma_ant + gamma_post)
    delta_e_cleft = 2.0 * L_cleft_um * (gamma_het - gamma_hom_mean)
    cos_half = float(np.clip(gamma_hom_mean / max(gamma_het, 1e-6), 0.0, 1.0))
    cleft_separation_score = float(np.clip(1.0 - cos_half**2, 0.0, 1.0))
    cleft_depth_um = rosette_radius_um * (1.0 - cos_half)
    return {
        "gamma_het": float(gamma_het),
        "gamma_hom_mean": float(gamma_hom_mean),
        "delta_e_cleft": float(delta_e_cleft),
        "cos_half_angle": cos_half,
        "cleft_separation_score": cleft_separation_score,
        "cleft_depth_um": float(cleft_depth_um),
    }


def evaluate_perturbation_regimes(
    ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all 6 experimental regimes from Yamanaka et al. (2023),
    Miao et al. (2023), and Adhyapok et al. (2021) using the 2D vertex solver
    and EphA4/ephrinB2 cleft separation model.
    """
    if ref is None:
        ref = load_reference_data()
    regimes_spec = ref["somitoid_experimental_regimes"]

    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    pred_ratios: list[float] = []
    emp_ratios: list[float] = []

    for row in regimes_spec:
        rid = row["regime_id"]
        v_sol = solve_2d_vertex_rosette(
            N=16,
            A0=115.0,
            K_A=0.08,
            gamma_ap=float(row["gamma_ap"]),
            gamma_bas=float(row["gamma_bas"]),
            gamma_lat=float(row["gamma_lat"]),
        )
        cleft = compute_eph_ephrin_cleft_mechanics(
            gamma_het=float(row["gamma_het"]),
            gamma_ant=float(row["gamma_lat"]),
            gamma_post=float(row["gamma_lat"]),
            rosette_radius_um=v_sol["mean_r_out_um"],
        )
        rec = {
            "regime_id": rid,
            "label": row["label"],
            "matrigel_pct": float(row["matrigel_pct"]),
            "gamma_ap": float(row["gamma_ap"]),
            "gamma_bas": float(row["gamma_bas"]),
            "gamma_lat": float(row["gamma_lat"]),
            "gamma_het": float(row["gamma_het"]),
            "predicted_ap_basal_ratio": round(v_sol["vertex_ap_basal_ratio"], 4),
            "analytic_polygon_ratio": round(v_sol["analytic_polygon_ratio"], 4),
            "analytic_continuum_ratio": round(v_sol["analytic_continuum_ratio"], 4),
            "empirical_ap_basal_ratio": float(row["empirical_ap_basal_ratio"]),
            "mean_r_in_um": round(v_sol["mean_r_in_um"], 3),
            "mean_r_out_um": round(v_sol["mean_r_out_um"], 3),
            "wedge_height_um": round(v_sol["wedge_height_um"], 3),
            "predicted_polarity_score": round(v_sol["apico_basal_polarity"], 4),
            "empirical_polarity_score": float(row["empirical_rosette_polarity_score"]),
            "predicted_cleft_score": round(cleft["cleft_separation_score"], 4),
            "empirical_cleft_score": float(row["empirical_cleft_separation_score"]),
            "delta_e_cleft": round(cleft["delta_e_cleft"], 3),
            "cleft_depth_um": round(cleft["cleft_depth_um"], 3),
            "polygons": v_sol["polygons"],
        }
        records.append(rec)
        by_id[rid] = rec
        pred_ratios.append(v_sol["vertex_ap_basal_ratio"])
        emp_ratios.append(float(row["empirical_ap_basal_ratio"]))

    pred_arr = np.array(pred_ratios, dtype=float)
    emp_arr = np.array(emp_ratios, dtype=float)
    r_val, p_val = stats.pearsonr(pred_arr, emp_arr)
    rmse = float(np.sqrt(np.mean((pred_arr - emp_arr) ** 2)))

    return {
        "records": records,
        "by_id": by_id,
        "pearson_r": float(r_val),
        "pearson_p": float(p_val),
        "rmse_ratio": rmse,
    }


# ============================================================================
# Section 3b (Pass 3): Live HES7 Delayed Oscillator -> Cooke-Zeeman Wavefront
# Arrest -> MESP2 / RIPPLY2 Boundary Fixation -> EphA4/ephrinB2 & RhoA/ROCK
# Tension Activation  (Law 5 `yaman-somitoid-chirality-buckling-5`;
# Lewis 2003; Cooke & Zeeman 1976; Diaz-Cuadros 2020; Matsuda 2020;
# Yamanaka 2023 Fig. 4 MESP2/RIPPLY2 stripes; Adhyapok 2021 Eph/ephrin boundary)
# ============================================================================
def _lewis_hes7_single_cell(
    time_scale: float,
    dt_h: float = 0.005,
    t_end_h: float = 60.0,
    a: float = 4.5,
    b: float = 0.23,
    c: float = 0.23,
    k: float = 33.0,
    p0: float = 40.0,
    T_m_min: float = 12.0,
    T_p_min: float = 2.8,
) -> dict[str, Any]:
    """Lewis (2003) Current Biology 13:1398 two-variable transcriptional-delay
    oscillator (his her1 parameter set: a=4.5 protein/mRNA/min, b=c=0.23/min,
    k=33 mRNA/min, p0=40, T_m=12.0 min, T_p=2.8 min) with every rate divided
    and every delay multiplied by `time_scale`, so the zebrafish-minute kernel
    is stretched to the human HES7 period (hours):
        dm/dt = k / (1 + (p(t - T_m)/p0)^2) - c m,     dp/dt = a m(t - T_p) - b p
    """
    b_h = b * 60.0 / time_scale
    c_h = c * 60.0 / time_scale
    a_h = a * 60.0 / time_scale
    k_h = k * 60.0 / time_scale
    T_m_h = T_m_min * time_scale / 60.0
    T_p_h = T_p_min * time_scale / 60.0
    n = int(round(t_end_h / dt_h))
    d_m = int(round(T_m_h / dt_h))
    d_p = int(round(T_p_h / dt_h))
    m = np.zeros(n)
    p = np.zeros(n)
    m[0] = 1.0
    p[0] = 1.0
    for t in range(1, n):
        p_del = p[t - 1 - d_m] if t - 1 - d_m >= 0 else 0.0
        m_del = m[t - 1 - d_p] if t - 1 - d_p >= 0 else 0.0
        m[t] = m[t - 1] + dt_h * (k_h / (1.0 + (p_del / p0) ** 2) - c_h * m[t - 1])
        p[t] = p[t - 1] + dt_h * (a_h * m_del - b_h * p[t - 1])
    tail = p[n // 2 :]
    peaks, _ = signal.find_peaks(tail, distance=int(1.0 / dt_h))
    periods = np.diff(peaks) * dt_h
    return {
        "time_h": np.arange(n) * dt_h,
        "mrna": m,
        "protein": p,
        "period_h": float(np.mean(periods)) if len(periods) else float("nan"),
        "period_cv": float(np.std(periods) / np.mean(periods)) if len(periods) > 1 else 0.0,
        "T_m_h": T_m_h,
        "T_p_h": T_p_h,
        "protein_half_life_min": float(math.log(2.0) / b_h * 60.0),
        "amplitude_min": float(np.min(tail)),
        "amplitude_max": float(np.max(tail)),
    }


def calibrate_hes7_time_scale(target_period_h: float = HES7_TARGET_PERIOD_H) -> dict[str, Any]:
    """Deterministic two-step secant calibration of the Lewis kernel time
    stretch so the simulated HES7 period matches the human ~5 h period
    (Diaz-Cuadros et al. 2020; Matsuda et al. 2020). The Lewis period is
    linear in the stretch (every delay and lifetime scales together), so two
    evaluations suffice; a third run verifies.
    """
    s1 = 6.0
    r1 = _lewis_hes7_single_cell(s1)
    s2 = s1 * target_period_h / r1["period_h"]
    r2 = _lewis_hes7_single_cell(s2)
    slope = (r2["period_h"] - r1["period_h"]) / (s2 - s1)
    s3 = s2 + (target_period_h - r2["period_h"]) / slope
    r3 = _lewis_hes7_single_cell(s3)
    return {
        "time_scale": float(s3),
        "period_h": r3["period_h"],
        "period_cv": r3["period_cv"],
        "T_m_h": r3["T_m_h"],
        "T_p_h": r3["T_p_h"],
        "protein_half_life_min": r3["protein_half_life_min"],
        "amplitude_min": r3["amplitude_min"],
        "amplitude_max": r3["amplitude_max"],
        "single_cell": r3,
    }


def simulate_hes7_clock_wavefront(
    time_scale: float,
    n_cells: int = PSM_CELLS,
    dx_um: float = PSM_CELL_DX_UM,
    v_um_per_h: float = WAVEFRONT_SPEED_UM_PER_H,
    t_end_h: float = 28.0,
    dt_h: float = 0.005,
    gamma_ap_wt: float = 1.65,
    gamma_bas: float = 0.45,
    gamma_het_wt: float = 2.35,
    gamma_hom: float = 1.0,
    tau_rock_h: float = TAU_ROCK_H,
    tau_eph_h: float = TAU_EPH_H,
    tau_ripply2_h: float = TAU_RIPPLY2_H,
    rock_inhibited: bool = False,
    eph_knockdown: bool = False,
) -> dict[str, Any]:
    """Live coupling of the segmentation clock to rosette mechanics.

    * Every PSM cell j at x_j = j dx runs the Lewis HES7 delayed oscillator
      (synchronous Cooke-Zeeman clock; no phase gradient / kinematic waves).
    * The determination wavefront x_wf(t) = v t arrests cell j at t_a,j = x_j / v
      (oscillator state frozen).
    * MESP2 is induced in arrested cells whose frozen HES7 protein is below the
      limit-cycle midpoint (NICD high when HES7 low): the anterior half-segment.
    * RIPPLY2 follows MESP2 after tau_RIPPLY2 and switches MESP2 off; the time
      the anterior-most MESP2+ cell of a stripe passes RIPPLY2 = 0.5 is the
      boundary determination time t_det,k.  A boundary sits between an
      ephrinB2+ (MESP2-, posterior half of the previous segment) anterior
      neighbour and an EphA4+ (MESP2 lineage) posterior neighbour.
    * gamma_het,k(t) ramps from gamma_hom toward gamma_het^WT after t_det,k
      (abolished under Eph/ephrin knockdown); gamma_ap,j(t) ramps from gamma_bas
      toward gamma_ap^WT after t_a,j (abolished under Y-27632).
    """
    x_um = np.arange(n_cells, dtype=float) * dx_um
    t_arrest = x_um / v_um_per_h

    # Lewis kernel (human-scaled)
    b_h = 0.23 * 60.0 / time_scale
    c_h = 0.23 * 60.0 / time_scale
    a_h = 4.5 * 60.0 / time_scale
    k_h = 33.0 * 60.0 / time_scale
    p0 = 40.0
    d_m = int(round(12.0 * time_scale / 60.0 / dt_h))
    d_p = int(round(2.8 * time_scale / 60.0 / dt_h))
    n_t = int(round(t_end_h / dt_h))
    time_h = np.arange(n_t) * dt_h

    m = np.zeros((n_t, n_cells))
    p = np.zeros((n_t, n_cells))
    m[0, :] = 1.0
    p[0, :] = 1.0
    for t in range(1, n_t):
        p_del = p[t - 1 - d_m] if t - 1 - d_m >= 0 else np.zeros(n_cells)
        m_del = m[t - 1 - d_p] if t - 1 - d_p >= 0 else np.zeros(n_cells)
        m_new = m[t - 1] + dt_h * (k_h / (1.0 + (p_del / p0) ** 2) - c_h * m[t - 1])
        p_new = p[t - 1] + dt_h * (a_h * m_del - b_h * p[t - 1])
        arrested = time_h[t] >= t_arrest
        m[t] = np.where(arrested, m[t - 1], m_new)
        p[t] = np.where(arrested, p[t - 1], p_new)

    # Limit-cycle median (time spent below = 50%) from the posterior-most, never-arrested cell
    free_tail = p[n_t // 2 :, -1]
    p_mid = float(np.median(free_tail))
    arrested_final = time_h[-1] >= t_arrest
    p_at_arrest = np.array([p[min(int(round(ta / dt_h)), n_t - 1), j] for j, ta in enumerate(t_arrest)])
    mesp2_lineage = arrested_final & (p_at_arrest < p_mid)

    # Boundaries: anterior neighbour MESP2- (ephrinB2+), posterior neighbour MESP2+ (EphA4+), both arrested
    boundary_cells: list[int] = []
    for j in range(n_cells - 1):
        if arrested_final[j] and arrested_final[j + 1] and (not mesp2_lineage[j]) and mesp2_lineage[j + 1]:
            boundary_cells.append(j + 1)   # index of the posterior (EphA4+) cell; boundary at x_j - dx/2
    boundary_x = np.array([x_um[j] - 0.5 * dx_um for j in boundary_cells], dtype=float)
    # RIPPLY2 induction after MESP2 (delay tau_ripply2) -> boundary determination time
    t_det = np.array([t_arrest[j] + tau_ripply2_h + 0.5 * math.log(2.0) for j in boundary_cells], dtype=float)
    # (RIPPLY2 ramp 1 - exp(-(t - t_a - tau)/0.5 h) crosses 0.5 at tau + 0.5 ln 2)

    # Spatiotemporal tension fields
    age = np.clip(time_h[:, None] - t_arrest[None, :], 0.0, None)
    if rock_inhibited:
        gamma_ap_xt = np.full_like(age, gamma_bas)
    else:
        gamma_ap_xt = gamma_bas + (gamma_ap_wt - gamma_bas) * (1.0 - np.exp(-age / tau_rock_h)) * (time_h[:, None] >= t_arrest[None, :])
    gamma_het_t = np.zeros((n_t, len(boundary_cells)))
    for kb, td in enumerate(t_det):
        dtb = np.clip(time_h - td, 0.0, None)
        if eph_knockdown:
            gamma_het_t[:, kb] = gamma_hom + 0.05 * (1.0 - np.exp(-dtb / tau_eph_h)) * (time_h >= td)
        else:
            gamma_het_t[:, kb] = gamma_hom + (gamma_het_wt - gamma_hom) * (1.0 - np.exp(-dtb / tau_eph_h)) * (time_h >= td)

    # Segment bookkeeping (complete segments between consecutive boundaries)
    segments: list[dict[str, Any]] = []
    for kb in range(len(boundary_cells) - 1):
        j0, j1 = boundary_cells[kb], boundary_cells[kb + 1]
        cells = np.arange(j0, j1)
        segments.append(
            {
                "anterior_boundary_x_um": float(boundary_x[kb]),
                "posterior_boundary_x_um": float(boundary_x[kb + 1]),
                "length_um": float(boundary_x[kb + 1] - boundary_x[kb]),
                "cells": cells,
                "n_cells": int(len(cells)),
                "mesp2_fraction": float(np.mean(mesp2_lineage[cells])),
                "t_arrest_mean_h": float(np.mean(t_arrest[cells])),
                "t_det_h": float(t_det[kb]),
                "gamma_ap_final": float(np.mean(gamma_ap_xt[-1, cells])),
                "gamma_het_final": float(gamma_het_t[-1, kb]),
            }
        )
    seg_lengths = np.array([s["length_um"] for s in segments], dtype=float)
    stripe_period_um = float(np.mean(np.diff(boundary_x))) if len(boundary_x) > 1 else float("nan")

    # HES7 period of the free posterior cell
    peaks, _ = signal.find_peaks(free_tail, distance=int(1.0 / dt_h))
    periods = np.diff(peaks) * dt_h
    period_h = float(np.mean(periods)) if len(periods) else float("nan")

    return {
        "time_h": time_h,
        "x_um": x_um,
        "hes7_protein_xt": p,
        "hes7_mrna_xt": m,
        "t_arrest": t_arrest,
        "wavefront_x_t": v_um_per_h * time_h,
        "p_mid": p_mid,
        "mesp2_lineage": mesp2_lineage,
        "arrested_final": arrested_final,
        "boundary_cells": boundary_cells,
        "boundary_x_um": boundary_x,
        "t_det_h": t_det,
        "gamma_ap_xt": gamma_ap_xt,
        "gamma_het_t": gamma_het_t,
        "segments": segments,
        "segment_lengths_um": seg_lengths,
        "stripe_period_um": stripe_period_um,
        "predicted_segment_length_um": float(v_um_per_h * period_h),
        "period_h": period_h,
        "n_boundaries": int(len(boundary_cells)),
        "rock_inhibited": rock_inhibited,
        "eph_knockdown": eph_knockdown,
        "params": {
            "v_um_per_h": v_um_per_h,
            "dx_um": dx_um,
            "tau_rock_h": tau_rock_h,
            "tau_eph_h": tau_eph_h,
            "tau_ripply2_h": tau_ripply2_h,
            "time_scale": time_scale,
        },
    }


def couple_clock_to_rosette_mechanics(
    clock: dict[str, Any],
    shell: dict[str, Any] | None = None,
    gamma_bas: float = 0.45,
    gamma_lat: float = 1.0,
) -> dict[str, Any]:
    """Feed the LIVE gamma_ap,j(t_end) and gamma_het,k(t_end) of every complete
    segment into the 2D vertex rosette solver (Law 1), the 3D shell law (Law 4)
    and the Eph/ephrin cleft model (Law 2), yielding a stage-resolved
    (S0 = newest ... S-n = oldest) lumen-ratio and cleft-score maturation series.
    """
    if shell is None:
        shell = build_spherical_voronoi_shell()
    stages: list[dict[str, Any]] = []
    segs = clock["segments"]
    n = len(segs)
    for idx, seg in enumerate(segs):
        stage_index = n - 1 - idx  # 0 = newest (posterior-most complete segment)
        g_ap = seg["gamma_ap_final"]
        v2d = solve_2d_vertex_rosette(N=16, A0=115.0, K_A=0.08, gamma_ap=g_ap, gamma_bas=gamma_bas, gamma_lat=gamma_lat)
        a3d = analytical_shell_equilibrium(shell, gamma_ap=g_ap, gamma_bas=gamma_bas, gamma_lat=gamma_lat, P_lumen=0.0)
        cleft = compute_eph_ephrin_cleft_mechanics(gamma_het=seg["gamma_het_final"], gamma_ant=gamma_lat, gamma_post=gamma_lat, rosette_radius_um=v2d["mean_r_out_um"])
        stages.append(
            {
                "stage": f"S{'-' + _roman(stage_index) if stage_index > 0 else '0'}",
                "stage_index": stage_index,
                "age_h": float(clock["time_h"][-1] - seg["t_arrest_mean_h"]),
                "gamma_ap": g_ap,
                "gamma_het": seg["gamma_het_final"],
                "ratio_2d": v2d["vertex_ap_basal_ratio"],
                "ratio_3d": a3d["rho_3d"],
                "r_in_2d_um": v2d["mean_r_in_um"],
                "cleft_score": cleft["cleft_separation_score"],
                "length_um": seg["length_um"],
                "polygons": v2d["polygons"],
            }
        )
    stages_sorted = sorted(stages, key=lambda s: s["stage_index"])
    ratios = np.array([s["ratio_2d"] for s in stages_sorted])
    return {
        "stages": stages_sorted,
        "ratio_2d_by_stage": ratios,
        "ratio_3d_by_stage": np.array([s["ratio_3d"] for s in stages_sorted]),
        "cleft_by_stage": np.array([s["cleft_score"] for s in stages_sorted]),
        "gamma_ap_by_stage": np.array([s["gamma_ap"] for s in stages_sorted]),
        "monotone_ap_maturation": bool(np.all(np.diff(ratios) <= 1e-9)),
    }


def _roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


# ============================================================================
# Section 4: 2D Phase Diagram Across Matrigel ECM % & Apical Myosin Tension
# ============================================================================
def compute_met_phase_diagram(
    n_ap: int = 28,
    n_mat: int = 28,
    N: int = 16,
) -> dict[str, Any]:
    """Compute the 2D epithelial rosette polarization & lumen-to-basal aspect
    ratio phase diagram over apical myosin tension (gamma_ap / gamma_lat in [0.3, 2.5])
    and Matrigel ECM concentration (0% to 12%).
    """
    gamma_ap_grid = np.linspace(0.35, 2.45, n_ap)
    matrigel_grid = np.linspace(0.0, 12.0, n_mat)
    # Matrigel lowers basal tension from 1.15 (0% ECM) down to ~0.42 (10-12% ECM)
    gamma_bas_from_mat = 0.40 + 0.75 * np.exp(-matrigel_grid / 3.6)

    ratio_map = np.zeros((n_mat, n_ap), dtype=float)
    polarity_map = np.zeros((n_mat, n_ap), dtype=float)

    for i, g_bas in enumerate(gamma_bas_from_mat):
        for j, g_ap in enumerate(gamma_ap_grid):
            eq = analytical_rosette_equilibrium(
                N=N, A0=115.0, gamma_ap=float(g_ap), gamma_bas=float(g_bas), gamma_lat=1.0
            )
            ratio = eq["ratio_polygon"]
            ratio_map[i, j] = ratio
            # Polarized epithelial rosette requires both apical constriction (g_ap > g_bas)
            # and basal ECM engagement (matrigel_pct >= 2%)
            ecm_gate = 1.0 - math.exp(-float(matrigel_grid[i]) / 2.2)
            diff_drive = max(float(g_ap - g_bas), 0.0) / (float(g_ap) + 0.5)
            polarity_map[i, j] = float(np.clip((1.0 - ratio) * (0.35 + 0.65 * ecm_gate) * (0.4 + 0.9 * diff_drive), 0.0, 1.0))

    return {
        "gamma_ap_grid": gamma_ap_grid,
        "matrigel_grid": matrigel_grid,
        "gamma_bas_from_mat": gamma_bas_from_mat,
        "ratio_map": ratio_map,
        "polarity_map": polarity_map,
    }


# ============================================================================
# Section 5: Empirical Specimen Rosette & Lumen-Cortex Segmentation
# (`datasets/somitoid_chirality_buckling/somitoid_rosette_specimen.jpg`)
# ============================================================================
def _extract_rosette_row(
    im: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> dict[str, Any]:
    """Extract somite rosette centers, inter-somitic clefts, and radial
    cortex-to-somitecoel intensity contrast along one paraxial somite row.
    """
    strip = im[y0:y1, x0:x1]
    profile_raw = np.mean(strip, axis=0)
    dog_signal = ndi.gaussian_filter1d(profile_raw, sigma=3.5) - ndi.gaussian_filter1d(
        profile_raw, sigma=18.0
    )
    peaks, _ = signal.find_peaks(dog_signal, distance=32, prominence=1.5)
    clefts, _ = signal.find_peaks(-dog_signal, distance=32, prominence=1.5)
    spacings = np.diff(peaks).astype(float)
    mean_sp = float(np.mean(spacings))
    sd_sp = float(np.std(spacings, ddof=1)) if len(spacings) > 1 else 0.0
    cv_sp = float(sd_sp / mean_sp) if mean_sp > 0 else 0.0

    y_center = 0.5 * (y0 + y1)
    peaks_abs = (peaks + x0).astype(int)

    # Radial annular profiling around each detected somite rosette center
    r_bins = np.arange(1, 22, dtype=float)
    radial_profiles: list[np.ndarray] = []
    yy, xx = np.indices(im.shape)
    for px in peaks_abs:
        dist = np.hypot(xx - px, yy - y_center)
        prof = np.array(
            [
                float(np.mean(im[(dist >= r - 1.0) & (dist < r + 1.0)]))
                for r in r_bins
            ],
            dtype=float,
        )
        radial_profiles.append(prof)
    mean_radial = np.mean(radial_profiles, axis=0)

    return {
        "y_bounds_px": [y0, y1],
        "y_center_px": y_center,
        "signal": dog_signal,
        "peaks_px": peaks_abs.tolist(),
        "clefts_px": (clefts + x0).tolist(),
        "n_rosettes": int(len(peaks)),
        "spacings_px": spacings.tolist(),
        "mean_spacing_px": round(mean_sp, 4),
        "sd_spacing_px": round(sd_sp, 4),
        "cv_spacing": round(cv_sp, 4),
        "r_bins_px": r_bins,
        "mean_radial_profile": mean_radial,
    }


def segment_somite_rosette_specimen(
    specimen_path: Path = SPECIMEN_JPG,
) -> dict[str, Any]:
    """Segment the vendored vertebrate somite rosette SEM specimen
    (`somitoid_rosette_specimen.jpg`) to extract bilateral somite rosette
    centers, inter-somitic cleft boundaries, spacing regularity, and the AP
    metamere waveform correlation against the EphA4/ephrinB2 periodic cleft model.
    """
    im = np.asarray(Image.open(specimen_path).convert("L"), dtype=float)
    x0, x1 = 130, 550
    upper = _extract_rosette_row(im, x0, x1, 208, 252)
    lower = _extract_rosette_row(im, x0, x1, 312, 356)

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
    model_wave = np.tanh(1.85 * np.cos(k_emp * x_px - phi_fit))
    model_norm = model_wave / (np.std(model_wave) + 1e-12)
    r_val, p_val = stats.pearsonr(emp_norm, model_norm)

    mean_radial = 0.5 * (upper["mean_radial_profile"] + lower["mean_radial_profile"])
    core_intensity = float(np.mean(mean_radial[:6]))
    cleft_rim_intensity = float(np.mean(mean_radial[14:20]))
    cortex_cleft_contrast = float(core_intensity / max(cleft_rim_intensity, 1e-6))

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
        "cortex_cleft_contrast_ratio": round(cortex_cleft_contrast, 4),
        "mean_radial_profile": mean_radial,
        "r_bins_px": upper["r_bins_px"],
        "emp_norm": emp_norm,
        "model_norm": model_norm,
    }


# ============================================================================
# Section 6: MorphoEval Dynamic Measurement Hook (`measure_morphoeval()`)
# ============================================================================
def compute_pass3_models(ref: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the Pass 3 3D shell (Section 2b) and live clock coupling (Section 3b)
    once and return everything the measurements, figure and GIF need.
    """
    if ref is None:
        ref = load_reference_data()
    shell = build_spherical_voronoi_shell()
    shell_wt = solve_3d_polyhedral_shell(shell, gamma_ap=1.65, gamma_bas=0.45, gamma_lat=1.0, P_lumen=0.0)
    shell_rock = solve_3d_polyhedral_shell(shell, gamma_ap=0.45, gamma_bas=0.45, gamma_lat=1.0, P_lumen=0.0)
    sweep_wt = sweep_lumen_pressure(shell, gamma_ap=1.65, gamma_bas=0.45)
    sweep_rock = sweep_lumen_pressure(shell, gamma_ap=0.45, gamma_bas=0.45, discrete_at=(0.0,))
    scutoid = compute_scutoid_topology_audit(shell, shell_wt)
    cal = calibrate_hes7_time_scale(float(ref.get("hes7_clock_wavefront", {}).get("target_period_h", HES7_TARGET_PERIOD_H)))
    clock_wt = simulate_hes7_clock_wavefront(cal["time_scale"])
    clock_rock = simulate_hes7_clock_wavefront(cal["time_scale"], rock_inhibited=True)
    clock_ephkd = simulate_hes7_clock_wavefront(cal["time_scale"], eph_knockdown=True)
    couple_wt = couple_clock_to_rosette_mechanics(clock_wt, shell)
    couple_rock = couple_clock_to_rosette_mechanics(clock_rock, shell)
    couple_ephkd = couple_clock_to_rosette_mechanics(clock_ephkd, shell)
    return {
        "shell": shell,
        "shell_wt": shell_wt,
        "shell_rock": shell_rock,
        "sweep_wt": sweep_wt,
        "sweep_rock": sweep_rock,
        "scutoid": scutoid,
        "hes7_calibration": cal,
        "clock_wt": clock_wt,
        "clock_rock": clock_rock,
        "clock_ephkd": clock_ephkd,
        "couple_wt": couple_wt,
        "couple_rock": couple_rock,
        "couple_ephkd": couple_ephkd,
    }


def measure_morphoeval() -> list[dict[str, Any]]:
    """Return 10 quantitative morphometric measurements for `morphoeval/measure_outputs.py`
    (5 carried from Pass 2 + 5 Pass 3 measurements on the 3D shell and the live clock coupling).
    """
    verify_dataset_integrity()
    ref = load_reference_data()
    pert = evaluate_perturbation_regimes(ref)
    spec = segment_somite_rosette_specimen()
    p3 = compute_pass3_models(ref)

    wt = pert["by_id"]["wt_plus_matrigel_somitoid"]
    rock = pert["by_id"]["rock_inhibited_y27632"]
    nomat = pert["by_id"]["minus_matrigel_unpolarized"]
    ephkd = pert["by_id"]["epha4_ephrinb2_kd_fused"]

    sh_wt = p3["shell_wt"]
    sh_rock = p3["shell_rock"]
    sw = p3["sweep_wt"]
    cal = p3["hes7_calibration"]
    ck = p3["clock_wt"]
    cp = p3["couple_wt"]
    cp_rock = p3["couple_rock"]
    cp_kd = p3["couple_ephkd"]
    oldest = cp["stages"][-1]
    newest = cp["stages"][0]
    hes7_ref = ref.get("hes7_clock_wavefront", {})

    pass3_measurements = [
        {
            "name": "3D polyhedral shell wild-type somitecoel lumen ratio rho_3D = R_in / R_out",
            "value": round(sh_wt["shell_ratio_3d"], 4),
            "unit": "dimensionless ratio",
            "method": f"closed 3D shell of N_3D={sh_wt['N3']} radial frustum cells (spherical-Voronoi faces: {p3['shell']['n_hexagons']} hex / {p3['shell']['n_pentagons']} pent / {p3['shell']['n_heptagons']} hept, {p3['shell']['n_faces']} lateral faces), L-BFGS-B relaxation of 2*N_3D radii with K_V={K_V_VOLUME}, V_0={V0_CELL_UM3} um^3, gamma_ap=1.65, gamma_bas=0.45, P=0; R_in*={sh_wt['mean_r_in_um']:.2f} um, R_out*={sh_wt['mean_r_out_um']:.2f} um",
            "reference_value": 0.305,
            "reference_source": "Yamanaka et al. (2023) Nature 614:509-520 Figs. 2-3 cross-sectional somitecoel/somite radius ratio (empirical 0.305; the 2D polygon law gives 0.303, the 3D closed-form Law 4 gives {:.4f})".format(sh_wt["analytic_ratio_3d_p0"]),
            "note": f"3D analytic Law 4 rho = {sh_wt['analytic_ratio_3d']:.4f} (|discrete - analytic| = {abs(sh_wt['shell_ratio_3d'] - sh_wt['analytic_ratio_3d']):.1e}); Y-27632 3D shell rho = {sh_rock['shell_ratio_3d']:.4f}; apical/basal area ratio = {sh_wt['apical_to_basal_area_ratio']:.4f}; cell volume CV = {sh_wt['cell_volume_cv']:.1e}; scutoid fraction = {p3['scutoid']['scutoid_fraction']:.1f} (radial frusta, Gomez-Galvez 2018 spherical limit)",
            "derived_from": "script",
        },
        {
            "name": "3D shell somitecoel rupture (saddle-node) lumen pressure P_crit",
            "value": round(sw["p_crit"], 4),
            "unit": "gamma_lat / um",
            "method": f"bisection on Law 4 E_3D(R_in; P) for loss of the closed-shell local minimum (WT gamma_ap=1.65, gamma_bas=0.45); rho_3D rises monotonically from {sw['rho'][0]:.4f} at P=0 to {sw['rho'][~sw['ruptured']][-1]:.4f} just below P_crit",
            "reference_value": round(sw["laplace_open_bound_p0"], 4),
            "reference_source": "Law 4 two-surface Laplace open-shell bound 2 (gamma_ap + gamma_bas) / R_in* evaluated at the P=0 equilibrium (model-internal analytical reference; no measured somitocoel pressure exists in Yamanaka 2023 / Miao 2023)",
            "note": "discrete 2*N_3D-DOF shell vs analytic rho at P=0/0.04/0.08: " + "; ".join(f"{d['ratio_discrete']:.4f} vs {d['ratio_analytic']:.4f}" for d in sw["discrete"]) + f"; Y-27632 shell P_crit = {p3['sweep_rock']['p_crit']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "live HES7 delayed-oscillator period T_HES7 (human-scaled Lewis 2003 kernel)",
            "value": round(ck["period_h"], 3),
            "unit": "h",
            "method": f"Lewis (2003) two-variable transcriptional-delay oscillator, zebrafish her1 kernel stretched by time_scale={cal['time_scale']:.3f} (T_m={cal['T_m_h']:.2f} h, T_p={cal['T_p_h']:.2f} h, protein half-life={cal['protein_half_life_min']:.1f} min), explicit Euler dt=0.005 h, period from protein peaks of the never-arrested posterior PSM cell (CV={cal['period_cv']:.1e})",
            "reference_value": float(hes7_ref.get("target_period_h", HES7_TARGET_PERIOD_H)),
            "reference_source": "Diaz-Cuadros et al. (2020) Nature 580:113-118 and Matsuda et al. (2020) Nature 580:124-129: human iPSC-PSM HES7 reporter period ~5 h (the time_scale is CALIBRATED to this value; the measurement validates the calibration, not an independent prediction)",
            "note": f"single-cell calibration period = {cal['period_h']:.3f} h; amplitude {cal['amplitude_min']:.1f}-{cal['amplitude_max']:.1f} protein units",
            "derived_from": "script",
        },
        {
            "name": "clock-wavefront MESP2 boundary spacing S (segment length) from arrested HES7 phase",
            "value": round(ck["stripe_period_um"], 3),
            "unit": "um",
            "method": f"mean spacing of {ck['n_boundaries']} ephrinB2-|EphA4+ boundaries (MESP2 off->on transitions of arrested cells) after {ck['time_h'][-1]:.0f} h with wavefront speed v={ck['params']['v_um_per_h']} um/h over {PSM_CELLS} cells of {PSM_CELL_DX_UM} um; MESP2 stripe fraction = {np.mean([s_['mesp2_fraction'] for s_ in ck['segments']]):.2f} of each segment",
            "reference_value": round(ck["predicted_segment_length_um"], 3),
            "reference_source": "Cooke & Zeeman (1976) J. Theor. Biol. 58:455-476 clock-and-wavefront law S = v * T_HES7 (analytical reference; v is a model choice giving ~100 um somites, not a value digitised from Yamanaka 2023)",
            "note": f"segment lengths = {[round(float(l), 1) for l in ck['segment_lengths_um']]} um (CV = {float(np.std(ck['segment_lengths_um']) / np.mean(ck['segment_lengths_um'])):.1e}); boundary determination times t_det = {[round(float(t), 2) for t in ck['t_det_h']]} h (RIPPLY2 delay {TAU_RIPPLY2_H} h)",
            "derived_from": "script",
        },
        {
            "name": "stage-resolved live-coupled lumen ratio: oldest somite ({}) L_ap/L_bas under RhoA/ROCK ramp".format(oldest["stage"]),
            "value": round(oldest["ratio_2d"], 4),
            "unit": "dimensionless ratio",
            "method": f"gamma_ap,j(t) = gamma_bas + (1.65 - 0.45)(1 - exp(-(t - t_arrest)/{TAU_ROCK_H} h)) of every cell of the oldest complete segment (age {oldest['age_h']:.2f} h, gamma_ap = {oldest['gamma_ap']:.4f}) fed live into the 2D 32-vertex solver; 3D Law 4 gives rho_3D = {oldest['ratio_3d']:.4f}",
            "reference_value": 0.305,
            "reference_source": "Yamanaka et al. (2023) Nature 614:509-520 Figs. 2-3 mature epithelial somite (empirical 0.305); maturation series S0 -> {} = {}".format(oldest["stage"], [round(float(r), 4) for r in cp["ratio_2d_by_stage"]]),
            "note": f"newest somite {newest['stage']} (age {newest['age_h']:.2f} h) ratio = {newest['ratio_2d']:.4f}; Y-27632 coupled ratio stays {cp_rock['stages'][-1]['ratio_2d']:.4f} at every stage; cleft score {oldest['cleft_score']:.4f} WT vs {cp_kd['stages'][-1]['cleft_score']:.4f} Eph/ephrin KD (gamma_het ramp tau = {TAU_EPH_H} h); anterior-to-posterior monotone maturation = {cp['monotone_ap_maturation']}",
            "derived_from": "script",
        },
    ]

    return [
        {
            "name": "wild-type somite rosette equilibrium apical-to-basal ratio L_ap / L_bas",
            "value": round(wt["predicted_ap_basal_ratio"], 4),
            "unit": "dimensionless ratio",
            "method": f"2D 32-vertex apico-basal differential interfacial tension relaxation (N=16 wedge cells, gamma_ap={wt['gamma_ap']}, gamma_bas={wt['gamma_bas']}, R_in*={wt['mean_r_in_um']:.2f} um, R_out*={wt['mean_r_out_um']:.2f} um)",
            "reference_value": round(wt["analytic_polygon_ratio"], 4),
            "reference_source": "Yamanaka et al. (2023) Nature 614:509-520 Figs. 1-3; Adhyapok et al. (2021) Dev. Cell 56:1312-1324 Figs. 2-4 (analytical ratio = 0.3037, empirical = 0.305)",
            "note": f"apico-basal polarity score = {wt['predicted_polarity_score']:.4f}; wedge cell height = {wt['wedge_height_um']:.2f} um",
            "derived_from": "script",
        },
        {
            "name": "Y-27632 ROCK-inhibited apico-basal ratio dilation L_ap / L_bas",
            "value": round(rock["predicted_ap_basal_ratio"], 4),
            "unit": "dimensionless ratio",
            "method": f"2D vertex relaxation under Y-27632 ROCK / Blebbistatin myosin II inhibition (gamma_ap = gamma_bas = {rock['gamma_ap']}, Delta_gamma_ap_bas = 0)",
            "reference_value": round(rock["analytic_polygon_ratio"], 4),
            "reference_source": "Yamanaka et al. (2023) Nature 614:509-520 Fig. 3; Miao et al. (2023) Nature 614:500-508 Fig. 4 (empirical unpolarized ratio = 0.700)",
            "note": f"somitecoel lumen dilates from {wt['mean_r_in_um']:.2f} um (WT) to {rock['mean_r_in_um']:.2f} um (Y-27632); Minus-Matrigel ratio = {nomat['predicted_ap_basal_ratio']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "6-regime somite rosette perturbation series Pearson correlation r",
            "value": round(pert["pearson_r"], 4),
            "unit": "Pearson r",
            "method": f"correlation between 2D vertex-predicted and empirical apical-to-basal ratios across 6 Yamanaka (2023), Miao (2023) & Adhyapok (2021) regimes (RMSE = {pert['rmse_ratio']:.4f})",
            "reference_value": 1.0,
            "reference_source": "Yamanaka et al. (2023) Nature 614:509-520; Miao et al. (2023) Nature 614:500-508; Adhyapok et al. (2021) Dev. Cell 56:1312-1324",
            "note": f"p = {pert['pearson_p']:.2e}; spans Calpeptin hyper-constricted ({pert['by_id']['high_myosin_hyperconstricted']['predicted_ap_basal_ratio']:.4f}) to Y-27632 ({rock['predicted_ap_basal_ratio']:.4f})",
            "derived_from": "script",
        },
        {
            "name": "EphA4/ephrinB2 inter-somitic cleft separation index S_cleft",
            "value": round(wt["predicted_cleft_score"], 4),
            "unit": "dimensionless index",
            "method": f"heterotypic EphA4/ephrinB2 contact repulsion cleft separation score 1 - (gamma_hom / gamma_het)^2 at gamma_het = {wt['gamma_het']} (Delta_E_cleft = {wt['delta_e_cleft']:.2f} pN)",
            "reference_value": 0.94,
            "reference_source": "Adhyapok et al. (2021) Dev. Cell 56:1312-1324 Figs. 1-3; Yamanaka et al. (2023) Nature 614:509-520 Fig. 2",
            "note": f"cleft indentation depth = {wt['cleft_depth_um']:.2f} um in WT vs {ephkd['cleft_depth_um']:.2f} um (S_cleft = {ephkd['predicted_cleft_score']:.4f}) in EphA4/ephrinB2 knockdown",
            "derived_from": "script",
        },
        {
            "name": "vertebrate somite rosette specimen bilateral metamere spacing S_obs",
            "value": round(spec["bilateral_mean_spacing_px"], 4),
            "unit": "px",
            "method": f"bilateral dorsal somite rosette DoG peak & inter-somitic cleft segmentation on somitoid_rosette_specimen.jpg (upper={spec['upper_row']['n_rosettes']} rosettes, lower={spec['lower_row']['n_rosettes']} rosettes, CV={spec['bilateral_cv_spacing']:.4f})",
            "reference_value": 47.1429,
            "reference_source": "Vendored CC BY 4.0 vertebrate embryo somite rosette SEM specimen (datasets/somitoid_chirality_buckling/somitoid_rosette_specimen.jpg)",
            "note": f"bilateral symmetry ratio = {spec['bilateral_symmetry_ratio']:.4f}, AP cleft waveform r = {spec['profile_phase_correlation_r']:.4f}, dome/cleft contrast = {spec['cortex_cleft_contrast_ratio']:.4f}; NOTE the vendored reference_json rosette_metrics (lower_row_somite_count=8, CV=0.2010) disagree with this segmentation (lower row = {spec['lower_row']['n_rosettes']}, CV = {spec['bilateral_cv_spacing']:.4f}) — carried as an open Pass 4 item",
            "derived_from": "output",
        },
    ] + pass3_measurements


# ============================================================================
# Section 7: Publication Figure & Animated GIF Rendering
# ============================================================================
def _draw_rosette_polygons(
    ax: plt.Axes,
    polygons: list[np.ndarray],
    r_in: float,
    r_out: float,
    title: str,
    face_color: str,
    apical_color: str = "#ff4d6d",
    basal_color: str = "#38bdf8",
) -> None:
    """Helper to draw a 2D wedge-cell somite rosette on a matplotlib Axes."""
    for poly in polygons:
        patch = MplPolygon(
            poly,
            closed=True,
            facecolor=face_color,
            edgecolor="#e2e8f0",
            linewidth=0.9,
            alpha=0.82,
        )
        ax.add_patch(patch)
        # Highlight apical (inner) chord and basal (outer) chord
        ax.plot(
            [poly[0, 0], poly[3, 0]],
            [poly[0, 1], poly[3, 1]],
            color=apical_color,
            lw=2.2,
            zorder=4,
        )
        ax.plot(
            [poly[1, 0], poly[2, 0]],
            [poly[1, 1], poly[2, 1]],
            color=basal_color,
            lw=2.0,
            zorder=4,
        )
    theta = np.linspace(0, 2.0 * math.pi, 100)
    ax.fill(
        r_in * np.cos(theta),
        r_in * np.sin(theta),
        color="#090d16",
        alpha=0.95,
        zorder=2,
    )
    ax.set_xlim(-34, 34)
    ax.set_ylim(-34, 34)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5, fontweight="bold", color="#f8fafc")


def _draw_3d_shell_cutaway(ax: Any, shell: dict[str, Any], sol: dict[str, Any], z_cut: float = 0.30) -> None:
    """Draw the relaxed 3D polyhedral shell with the dorsal cap removed so the
    apical (lumen-facing) faces and the somitecoel are visible."""
    verts = shell["unit_vertices"]
    dirs = shell["seed_dirs"]
    r_in = sol["r_in_cells"]
    r_out = sol["r_out_cells"]
    basal, apical, lateral = [], [], []
    for i, region in enumerate(shell["regions"]):
        poly_unit = verts[region]
        if dirs[i, 2] < z_cut:
            basal.append(poly_unit * r_out[i])
        apical.append(poly_unit * r_in[i])
        if (z_cut - 0.30) < dirs[i, 2] < z_cut:
            m = len(region)
            for k in range(m):
                a, b = poly_unit[k], poly_unit[(k + 1) % m]
                lateral.append(np.array([a * r_in[i], b * r_in[i], b * r_out[i], a * r_out[i]]))
    # One collection so apical / lateral / basal faces are depth-sorted together (lumen visible through the cut)
    faces = apical + lateral + basal
    face_colors = ["#f43f5e"] * len(apical) + ["#64748b"] * len(lateral) + ["#1d4ed8"] * len(basal)
    edge_colors = ["#fecdd3"] * len(apical) + ["#cbd5e1"] * len(lateral) + ["#93c5fd"] * len(basal)
    coll = Poly3DCollection(faces, facecolors=face_colors, edgecolors=edge_colors, linewidths=0.4, alpha=0.92)
    coll.set_zsort("average")
    ax.add_collection3d(coll)
    lim = float(np.max(r_out)) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=52, azim=-55)
    ax.set_facecolor("#111827")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.09, 0.15, 1.0))
        axis.label.set_color("#e2e8f0")
    ax.tick_params(colors="#cbd5e1", labelsize=7)
    ax.set_xlabel("x (um)", fontsize=8)
    ax.set_ylabel("y (um)", fontsize=8)
    ax.set_zlabel("z (um)", fontsize=8)


def render_comparison_figure(out_png: Path = OUTPUT_PNG, p3: dict[str, Any] | None = None) -> Path:
    """Render the 9-panel dark-theme publication synthesis figure
    `docs/media/somitoid_chirality_buckling_comparison.png` (Panels A-F from
    Pass 2; Panels G-I added in Pass 3 for the 3D shell, lumen pressurization
    and the live HES7 clock-wavefront coupling).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ref = load_reference_data()
    curves = compute_rosette_energy_landscape()
    pert = evaluate_perturbation_regimes(ref)
    phase = compute_met_phase_diagram()
    spec = segment_somite_rosette_specimen()
    if p3 is None:
        p3 = compute_pass3_models(ref)
    im_spec = np.asarray(Image.open(SPECIMEN_JPG).convert("L"), dtype=float)

    plt.rcParams.update(
        {
            "figure.facecolor": "#0b0f19",
            "axes.facecolor": "#111827",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#e2e8f0",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "grid.color": "#1e293b",
        }
    )

    fig = plt.figure(figsize=(18.0, 16.8), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.36, wspace=0.28)

    # Panel A: 2D Vertex Somite Rosettes (WT +Matrigel vs Y-27632 ROCK-Inhibited)
    ax_a = fig.add_subplot(gs[0, 0])
    wt = pert["by_id"]["wt_plus_matrigel_somitoid"]
    rock = pert["by_id"]["rock_inhibited_y27632"]
    for poly in wt["polygons"]:
        shifted = poly.copy()
        shifted[:, 0] -= 29.0
        ax_a.add_patch(
            MplPolygon(
                shifted,
                closed=True,
                facecolor="#1d4ed8",
                edgecolor="#93c5fd",
                lw=0.85,
                alpha=0.85,
            )
        )
        ax_a.plot(
            [shifted[0, 0], shifted[3, 0]],
            [shifted[0, 1], shifted[3, 1]],
            color="#f43f5e",
            lw=2.2,
        )
        ax_a.plot(
            [shifted[1, 0], shifted[2, 0]],
            [shifted[1, 1], shifted[2, 1]],
            color="#38bdf8",
            lw=1.8,
        )
    for poly in rock["polygons"]:
        shifted = poly.copy()
        shifted[:, 0] += 29.0
        ax_a.add_patch(
            MplPolygon(
                shifted,
                closed=True,
                facecolor="#475569",
                edgecolor="#cbd5e1",
                lw=0.85,
                alpha=0.75,
            )
        )
        ax_a.plot(
            [shifted[0, 0], shifted[3, 0]],
            [shifted[0, 1], shifted[3, 1]],
            color="#f59e0b",
            lw=1.8,
        )
    ax_a.text(
        -29.0,
        -33.5,
        f"WT (+10% Matrigel)\nL_ap/L_bas = {wt['predicted_ap_basal_ratio']:.3f}\nR_in* = {wt['mean_r_in_um']:.1f} um",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#93c5fd",
        fontweight="bold",
    )
    ax_a.text(
        29.0,
        -33.5,
        f"+Y-27632 (ROCK Inh.)\nL_ap/L_bas = {rock['predicted_ap_basal_ratio']:.3f}\nR_in* = {rock['mean_r_in_um']:.1f} um",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#fcd34d",
        fontweight="bold",
    )
    ax_a.set_xlim(-60, 60)
    ax_a.set_ylim(-46, 34)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("Transverse Coordinate x (um)")
    ax_a.set_ylabel("Dorsoventral Coordinate y (um)")
    ax_a.set_title(
        "A. 2D Vertex Somite Rosettes: WT MET vs Y-27632 Collapse",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_a.grid(True, alpha=0.35)

    # Panel B: Radial Interfacial Energy Landscape E(R_in, N)
    ax_b = fig.add_subplot(gs[0, 1])
    r_in_g = curves["r_in"]
    palette = {
        "wt_plus_matrigel": "#38bdf8",
        "high_myosin": "#f43f5e",
        "low_matrigel": "#a855f7",
        "rock_y27632": "#f59e0b",
        "minus_matrigel": "#94a3b8",
    }
    for k, cdata in curves["regimes"].items():
        e_norm = cdata["energy"] - np.min(cdata["energy"])
        ax_b.plot(
            r_in_g,
            e_norm,
            lw=2.1,
            color=palette[k],
            label=f"{cdata['label']} (R_in*={cdata['r_in_opt']:.1f} um)",
        )
        ax_b.scatter(
            [cdata["r_in_opt"]],
            [0.0],
            color=palette[k],
            s=42,
            zorder=5,
        )
    ax_b.set_xlabel("Central Somitecoel Lumen Radius R_in (um)")
    ax_b.set_ylabel("Relative Interfacial Energy E(R_in) - E_min")
    ax_b.set_title(
        "B. Apico-Basal Vertex Energy Landscape E(R_in, N=16)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_b.legend(fontsize=7.6, loc="upper right", framealpha=0.85)
    ax_b.grid(True, alpha=0.35)

    # Panel C: 2D Phase Diagram Over Apical Myosin Tension & Matrigel ECM %
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.contourf(
        phase["gamma_ap_grid"],
        phase["matrigel_grid"],
        phase["ratio_map"],
        levels=18,
        cmap="viridis_r",
    )
    cbar = fig.colorbar(im_c, ax=ax_c, pad=0.02)
    cbar.set_label("Equilibrium Apical/Basal Ratio L_ap / L_bas", fontsize=8.5)
    for rec in pert["records"]:
        ax_c.scatter(
            [rec["gamma_ap"]],
            [rec["matrigel_pct"]],
            s=65,
            edgecolor="white",
            facecolor="#f43f5e" if rec["predicted_ap_basal_ratio"] < 0.45 else "#f59e0b",
            zorder=6,
        )
    ax_c.set_xlabel("Apical Myosin Tension Ratio gamma_ap / gamma_lat")
    ax_c.set_ylabel("Matrigel ECM Concentration (%)")
    ax_c.set_title(
        "C. Phase Diagram: Apical Constriction vs Matrigel ECM",
        fontsize=10.5,
        fontweight="bold",
    )

    # Panel D: 6-Regime Perturbation Benchmark (Model vs Empirical + Cleft Score)
    ax_d = fig.add_subplot(gs[1, 0])
    short_labels = [
        "WT\n+Matrigel",
        "Calpeptin\nHyper-Ap",
        "Low\nMatrigel",
        "Eph/ephrin\nKD",
        "Y-27632\nROCK Inh",
        "Minus\nMatrigel",
    ]
    x_idx = np.arange(len(pert["records"]))
    w = 0.27
    pred_vals = [r["predicted_ap_basal_ratio"] for r in pert["records"]]
    emp_vals = [r["empirical_ap_basal_ratio"] for r in pert["records"]]
    cleft_vals = [r["predicted_cleft_score"] for r in pert["records"]]
    ax_d.bar(x_idx - w, pred_vals, width=w, color="#38bdf8", label="2D Vertex L_ap/L_bas")
    ax_d.bar(x_idx, emp_vals, width=w, color="#818cf8", label="Empirical L_ap/L_bas")
    ax_d.bar(
        x_idx + w,
        cleft_vals,
        width=w,
        color="#34d399",
        label="Eph/ephrin Cleft Score",
    )
    ax_d.set_xticks(x_idx)
    ax_d.set_xticklabels(short_labels, fontsize=7.8)
    ax_d.set_ylabel("Dimensionless Ratio / Cleft Score")
    ax_d.set_title(
        f"D. 6-Regime MET & Cleft Benchmark (r = {pert['pearson_r']:.4f}, RMSE = {pert['rmse_ratio']:.3f})",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_d.legend(fontsize=7.6, loc="upper left", framealpha=0.85)
    ax_d.grid(True, axis="y", alpha=0.35)

    # Panel E: Empirical Specimen Bilateral Somite Rosette Segmentation
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.imshow(im_spec, cmap="gray")
    x0, x1 = spec["ap_roi_x_px"]
    for row_key, col_p, col_c in [
        ("upper_row", "#38bdf8", "#f43f5e"),
        ("lower_row", "#34d399", "#f59e0b"),
    ]:
        row = spec[row_key]
        yc = row["y_center_px"]
        ax_e.scatter(
            row["peaks_px"],
            [yc] * len(row["peaks_px"]),
            color=col_p,
            s=42,
            zorder=5,
            label=f"{row_key.replace('_', ' ').title()} Centers (n={row['n_rosettes']})",
        )
        for cx in row["clefts_px"]:
            ax_e.plot(
                [cx, cx],
                row["y_bounds_px"],
                color=col_c,
                ls="--",
                lw=1.3,
                alpha=0.9,
            )
    ax_e.set_xlim(90, 610)
    ax_e.set_ylim(430, 140)
    ax_e.set_title(
        f"E. Specimen Rosette & Cleft Segmentation (S = {spec['bilateral_mean_spacing_px']:.2f} px)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_e.legend(fontsize=7.5, loc="lower right", framealpha=0.85)
    ax_e.axis("off")

    # Panel F: Periodic Eph/ephrin Cleft Waveform Fit & Radial Rosette Profile
    ax_f = fig.add_subplot(gs[1, 2])
    x_rel = np.arange(len(spec["emp_norm"]), dtype=float)
    ax_f.plot(
        x_rel,
        spec["emp_norm"],
        color="#38bdf8",
        lw=1.9,
        label="Empirical Somite Row DoG Signal",
    )
    ax_f.plot(
        x_rel,
        spec["model_norm"],
        color="#f43f5e",
        lw=1.8,
        ls="--",
        label=f"Periodic Rosette/Cleft Fit (r = {spec['profile_phase_correlation_r']:.3f})",
    )
    ax_f.set_xlabel("Anteroposterior Position Along Paraxial Axis (px)")
    ax_f.set_ylabel("Normalized Metameric Intensity (z-score)")
    ax_f.set_title(
        f"F. AP Rosette-Cleft Periodicity & Contrast ({spec['cortex_cleft_contrast_ratio']:.3f}x)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_f.legend(fontsize=7.8, loc="upper right", framealpha=0.85)
    ax_f.grid(True, alpha=0.35)

    # ---- Pass 3 panels -------------------------------------------------------
    shell = p3["shell"]
    sh_wt = p3["shell_wt"]
    sh_rock = p3["shell_rock"]
    sw_wt = p3["sweep_wt"]
    sw_rock = p3["sweep_rock"]
    ck = p3["clock_wt"]
    cp = p3["couple_wt"]

    # Panel G: 3D polyhedral shell cut-away (WT)
    ax_g = fig.add_subplot(gs[2, 0], projection="3d")
    _draw_3d_shell_cutaway(ax_g, shell, sh_wt, z_cut=0.05)
    ax_g.set_title(
        f"G. 3D Polyhedral Shell Cut-away (N={shell['N3']} frusta, {shell['n_faces']} faces)\n"
        f"WT rho_3D = {sh_wt['shell_ratio_3d']:.3f} (Law 4 {sh_wt['analytic_ratio_3d']:.3f}; emp. 0.305); Y-27632 {sh_rock['shell_ratio_3d']:.3f}",
        fontsize=9.0,
        fontweight="bold",
    )
    ax_g.text2D(
        0.02,
        0.02,
        f"{shell['n_hexagons']} hex / {shell['n_pentagons']} pent / {shell['n_heptagons']} hept\nscutoids: {p3['scutoid']['scutoid_fraction']:.0%} (radial frusta; sphere limit of Gomez-Galvez 2018)",
        transform=ax_g.transAxes,
        fontsize=7.4,
        color="#cbd5e1",
    )

    # Panel H: lumen pressurization sweep rho_3D(P) with rupture pressure
    ax_h = fig.add_subplot(gs[2, 1])
    ok = ~sw_wt["ruptured"]
    ax_h.plot(sw_wt["p_grid"][ok], sw_wt["rho"][ok], color="#38bdf8", lw=2.3, label="WT shell (Law 4 analytic)")
    ok_r = ~sw_rock["ruptured"]
    ax_h.plot(sw_rock["p_grid"][ok_r], sw_rock["rho"][ok_r], color="#f59e0b", lw=2.0, ls="--", label="Y-27632 shell (gamma_ap = gamma_bas)")
    ax_h.scatter(
        [d["P"] for d in sw_wt["discrete"]],
        [d["ratio_discrete"] for d in sw_wt["discrete"]],
        s=60,
        facecolor="#f43f5e",
        edgecolor="white",
        zorder=6,
        label=f"discrete 2N_3D-DOF relaxation (N={shell['N3']})",
    )
    ax_h.axvline(sw_wt["p_crit"], color="#f43f5e", lw=1.4, ls=":", alpha=0.9)
    ax_h.text(sw_wt["p_crit"], 0.08, f" P_crit = {sw_wt['p_crit']:.3f}\n (lumen blow-out)", color="#fda4af", fontsize=7.6, va="bottom")
    ax_h.axhline(0.305, color="#a78bfa", lw=1.1, ls="--", alpha=0.8)
    ax_h.text(0.0025, 0.312, "empirical cross-section ratio 0.305 (Yamanaka 2023)", color="#c4b5fd", fontsize=7.2)
    ax_h.axhline(wt["predicted_ap_basal_ratio"], color="#34d399", lw=1.0, ls="-.", alpha=0.8)
    ax_h.text(0.085, wt["predicted_ap_basal_ratio"] - 0.03, f"2D vertex rosette {wt['predicted_ap_basal_ratio']:.3f}", color="#6ee7b7", fontsize=7.2)
    ax_h.set_xlim(0.0, float(sw_wt["p_grid"][-1]))
    ax_h.set_ylim(0.0, 1.0)
    ax_h.set_xlabel("Somitecoel Lumen Pressure P (gamma_lat / um)")
    ax_h.set_ylabel("3D Equilibrium Lumen Ratio rho_3D = R_in / R_out")
    ax_h.set_title("H. 3D Somitecoel Pressurization rho_3D(P)\n& Saddle-node Rupture Pressure P_crit", fontsize=9.6, fontweight="bold")
    ax_h.legend(fontsize=7.4, loc="upper left", framealpha=0.85)
    ax_h.grid(True, alpha=0.35)

    # Panel I: live HES7 clock-wavefront kymograph with MESP2 boundaries and tension activation
    ax_i = fig.add_subplot(gs[2, 2])
    t_h = ck["time_h"]
    x_um = ck["x_um"]
    stride = max(1, len(t_h) // 700)
    kym = ck["hes7_protein_xt"][::stride]
    ax_i.imshow(
        kym / max(float(np.max(kym)), 1e-9),
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[x_um[0] - 0.5 * PSM_CELL_DX_UM, x_um[-1] + 0.5 * PSM_CELL_DX_UM, t_h[0], t_h[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    ax_i.plot(ck["wavefront_x_t"], t_h, color="#38bdf8", lw=1.8, label="determination wavefront x_wf = v t")
    ax_i.contour(
        x_um,
        t_h[::stride],
        ck["gamma_ap_xt"][::stride],
        levels=[0.75, 1.05, 1.35, 1.6],
        colors=["#bbf7d0", "#86efac", "#4ade80", "#22c55e"],
        linewidths=0.9,
        alpha=0.9,
    )
    for kb, (bx, td) in enumerate(zip(ck["boundary_x_um"], ck["t_det_h"])):
        ax_i.plot([bx, bx], [td, t_h[-1]], color="#f43f5e", lw=1.6)
        ax_i.scatter([bx], [td], s=28, color="#fda4af", zorder=6, label="MESP2/RIPPLY2 boundary fixed (t_det)" if kb == 0 else None)
    mesp = ck["mesp2_lineage"]
    for j in range(len(x_um)):
        if mesp[j]:
            ax_i.plot([x_um[j] - 0.5 * PSM_CELL_DX_UM, x_um[j] + 0.5 * PSM_CELL_DX_UM], [t_h[-1] - 0.4, t_h[-1] - 0.4], color="#fbbf24", lw=3.2, solid_capstyle="butt")
    ax_i.plot([], [], color="#fbbf24", lw=3.2, label="MESP2-lineage (EphA4+) half-segments")
    ax_i.plot([], [], color="#4ade80", lw=0.9, label="gamma_ap(x,t) RhoA/ROCK contours 0.75-1.6")
    ax_i.set_xlim(x_um[0] - 0.5 * PSM_CELL_DX_UM, x_um[-1] + 0.5 * PSM_CELL_DX_UM)
    ax_i.set_ylim(t_h[0], t_h[-1])
    ax_i.set_xlabel("Anteroposterior PSM Position x (um; anterior left)")
    ax_i.set_ylabel("Time (h)")
    ax_i.set_title(
        f"I. Live HES7 Clock (T = {ck['period_h']:.2f} h) -> Arrest -> Tensions\n"
        f"S = {ck['stripe_period_um']:.0f} um = v T; {cp['stages'][-1]['stage']} L_ap/L_bas = {cp['stages'][-1]['ratio_2d']:.3f}, S0 = {cp['stages'][0]['ratio_2d']:.3f}",
        fontsize=9.0,
        fontweight="bold",
    )
    ax_i.legend(fontsize=6.9, loc="upper left", framealpha=0.8)

    fig.suptitle(
        "Yamanaka et al. (2023), Miao et al. (2023) & Adhyapok et al. (2021) — Epithelial Somite Rosette Vertex Mechanics, 3D Shell Pressurization & Live HES7 Clock Coupling",
        fontsize=12.6,
        fontweight="bold",
        color="#f8fafc",
        y=0.985,
    )
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def render_clock_coupling_figure(out_png: Path = OUTPUT_CLOCK_PNG, p3: dict[str, Any] | None = None) -> Path:
    """Render the Pass 3 detail figure `docs/media/somitoid_chirality_buckling_clock_coupling.png`:
    (a) calibrated single-cell HES7 oscillator, (b) gamma_ap(x,t) kymograph with MESP2/RIPPLY2
    boundaries and gamma_het ramps, (c) final MESP2 / EphA4 / ephrinB2 state map, (d) stage-resolved
    coupled lumen ratio & cleft score for WT, Y-27632 and Eph/ephrin knockdown.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if p3 is None:
        p3 = compute_pass3_models()
    cal = p3["hes7_calibration"]
    ck = p3["clock_wt"]
    cp = p3["couple_wt"]
    cp_rock = p3["couple_rock"]
    cp_kd = p3["couple_ephkd"]

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.4), dpi=140)
    fig.patch.set_facecolor("#0b0f19")
    for ax in axes.ravel():
        ax.set_facecolor("#111827")

    # (a) single-cell oscillator
    ax = axes[0, 0]
    sc = cal["single_cell"]
    tt = sc["time_h"]
    sel = tt >= 20.0
    ax.plot(tt[sel], sc["protein"][sel] / np.max(sc["protein"][sel]), color="#f43f5e", lw=2.0, label="HES7 protein p(t)")
    ax.plot(tt[sel], sc["mrna"][sel] / np.max(sc["mrna"][sel]), color="#38bdf8", lw=1.6, ls="--", label="HES7 mRNA m(t)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Normalized Level")
    ax.set_title(
        f"a. Lewis (2003) delayed oscillator scaled to human HES7: T = {cal['period_h']:.3f} h (target {HES7_TARGET_PERIOD_H} h)\n"
        f"T_m = {cal['T_m_h']:.2f} h, T_p = {cal['T_p_h']:.2f} h, protein t1/2 = {cal['protein_half_life_min']:.1f} min, time_scale = {cal['time_scale']:.3f}",
        fontsize=9.4,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.35)

    # (b) gamma_ap(x,t) kymograph + boundaries + gamma_het
    ax = axes[0, 1]
    t_h = ck["time_h"]
    x_um = ck["x_um"]
    stride = max(1, len(t_h) // 700)
    im = ax.imshow(
        ck["gamma_ap_xt"][::stride],
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[x_um[0] - 5, x_um[-1] + 5, t_h[0], t_h[-1]],
        vmin=0.45,
        vmax=1.65,
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("gamma_ap(x,t) (RhoA/ROCK apical tension)", fontsize=8)
    ax.plot(ck["wavefront_x_t"], t_h, color="white", lw=1.4, ls="--", label="wavefront")
    for kb, (bx, td) in enumerate(zip(ck["boundary_x_um"], ck["t_det_h"])):
        ax.plot([bx, bx], [td, t_h[-1]], color="#f43f5e", lw=1.8, label="EphA4/ephrinB2 boundary" if kb == 0 else None)
        ghet = ck["gamma_het_t"][:, kb]
        ax.plot(bx + 18.0 * (ghet - 1.0) / 1.35, t_h, color="#fbbf24", lw=1.1, alpha=0.9, label="gamma_het,k(t) ramp (offset)" if kb == 0 else None)
    ax.set_xlabel("AP position x (um)")
    ax.set_ylabel("Time (h)")
    ax.set_title(f"b. gamma_ap(x,t) after arrest (tau_ROCK = {TAU_ROCK_H} h) & gamma_het,k(t) (tau_Eph = {TAU_EPH_H} h)", fontsize=9.4, fontweight="bold")
    ax.legend(fontsize=7.4, loc="upper left", framealpha=0.8)

    # (c) final state map
    ax = axes[1, 0]
    mesp = ck["mesp2_lineage"].astype(float)
    arr = ck["arrested_final"].astype(float)
    ax.step(x_um, arr * 1.0, where="mid", color="#94a3b8", lw=1.6, label="arrested (post-wavefront)")
    ax.step(x_um, mesp * 0.8, where="mid", color="#fbbf24", lw=2.0, label="MESP2 lineage -> EphA4+")
    ax.step(x_um, (arr * (1 - mesp)) * 0.6, where="mid", color="#60a5fa", lw=2.0, label="posterior half -> ephrinB2+")
    g_final = ck["gamma_ap_xt"][-1]
    ax.plot(x_um, (g_final - 0.45) / 1.2 * 0.4, color="#4ade80", lw=2.0, label="gamma_ap(x, t_end) (scaled)")
    for bx in ck["boundary_x_um"]:
        ax.axvline(bx, color="#f43f5e", lw=1.3, alpha=0.9)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("AP position x (um)")
    ax.set_ylabel("State")
    ax.set_title(f"c. Final MESP2 / EphA4 / ephrinB2 state map ({ck['n_boundaries']} boundaries, S = {ck['stripe_period_um']:.0f} um)", fontsize=9.4, fontweight="bold")
    ax.legend(fontsize=7.4, loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # (d) stage-resolved coupled morphology
    ax = axes[1, 1]
    labels = [s["stage"] for s in cp["stages"]]
    xi = np.arange(len(labels))
    w = 0.2
    ax.bar(xi - 1.5 * w, cp["ratio_2d_by_stage"], width=w, color="#38bdf8", label="WT L_ap/L_bas (2D vertex, live gamma_ap)")
    ax.bar(xi - 0.5 * w, cp["ratio_3d_by_stage"], width=w, color="#818cf8", label="WT rho_3D (Law 4 shell)")
    ax.bar(xi + 0.5 * w, cp_rock["ratio_2d_by_stage"], width=w, color="#f59e0b", label="Y-27632 L_ap/L_bas")
    ax.bar(xi + 1.5 * w, cp["cleft_by_stage"], width=w, color="#34d399", label="WT Eph/ephrin cleft score")
    ax.plot(xi, cp_kd["cleft_by_stage"], color="#f43f5e", marker="o", lw=1.4, label="Eph/ephrin KD cleft score")
    ax.axhline(0.305, color="#c4b5fd", ls="--", lw=1.1, label="empirical mature ratio 0.305")
    ax.set_xticks(xi)
    ax.set_xticklabels([f"{l}\n{s['age_h']:.1f} h" for l, s in zip(labels, cp["stages"])], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Ratio / Score")
    ax.set_title("d. Stage-resolved live coupling (S0 newest -> oldest): MET maturation & clefting", fontsize=9.4, fontweight="bold")
    ax.legend(fontsize=7.2, loc="upper right", framealpha=0.8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Pass 3 — Live HES7 Delayed Oscillator -> Cooke-Zeeman Wavefront Arrest -> MESP2/RIPPLY2 -> EphA4/ephrinB2 & RhoA/ROCK Tensions -> Rosette Mechanics",
        fontsize=11.6,
        fontweight="bold",
        color="#f8fafc",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


def render_growth_gif(out_gif: Path = OUTPUT_GIF, n_frames: int = 20, p3: dict[str, Any] | None = None) -> Path:
    """Render a 20-frame animated GIF driven by the LIVE clock coupling: each
    frame is a time point t_k of the HES7 clock-wavefront simulation; the left
    panel shows the HES7 kymograph up to t_k with the boundaries fixed so far,
    the middle panel the WT rosette of the oldest complete somite relaxed with its
    current RhoA/ROCK apical tension gamma_ap(t_k), and the right panel the
    Y-27632 control (boundaries form, gamma_ap stays at gamma_bas, no MET).
    """
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    if p3 is None:
        p3 = compute_pass3_models()
    ck = p3["clock_wt"]
    t_h = ck["time_h"]
    x_um = ck["x_um"]
    frames: list[Image.Image] = []
    t_samples = np.linspace(6.0, float(t_h[-1]), n_frames)
    stride = max(1, len(t_h) // 500)

    for idx, t_k in enumerate(t_samples):
        it = int(min(np.searchsorted(t_h, t_k), len(t_h) - 1))
        fixed = [kb for kb, td in enumerate(ck["t_det_h"]) if td <= t_k]
        if len(fixed) >= 2:
            seg = ck["segments"][fixed[0]]
            g_ap = float(np.mean(ck["gamma_ap_xt"][it, seg["cells"]]))
            seg_label = f"oldest complete somite x={seg['anterior_boundary_x_um']:.0f}-{seg['posterior_boundary_x_um']:.0f} um"
        else:
            cells = np.where(ck["arrested_final"] & (ck["t_arrest"] <= t_k))[0]
            g_ap = float(np.mean(ck["gamma_ap_xt"][it, cells])) if len(cells) else 0.45
            seg_label = "anterior arrested cells (no complete somite yet)"
        wt_step = solve_2d_vertex_rosette(N=16, A0=115.0, K_A=0.08, gamma_ap=g_ap, gamma_bas=0.45, gamma_lat=1.0)
        rock_step = solve_2d_vertex_rosette(N=16, A0=115.0, K_A=0.08, gamma_ap=0.45, gamma_bas=0.45, gamma_lat=1.0)

        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), dpi=95)
        fig.patch.set_facecolor("#0b0f19")
        for ax in axes:
            ax.set_facecolor("#111827")

        kym = ck["hes7_protein_xt"][: it + 1 : stride]
        axes[0].imshow(
            kym / max(float(np.max(ck["hes7_protein_xt"])), 1e-9),
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[x_um[0] - 5, x_um[-1] + 5, 0.0, float(t_h[it])],
            vmin=0.0,
            vmax=1.0,
        )
        axes[0].plot(ck["wavefront_x_t"][: it + 1], t_h[: it + 1], color="#38bdf8", lw=1.6)
        for kb in fixed:
            bx, td = ck["boundary_x_um"][kb], ck["t_det_h"][kb]
            axes[0].plot([bx, bx], [td, t_h[it]], color="#f43f5e", lw=1.6)
        axes[0].set_xlim(x_um[0] - 5, x_um[-1] + 5)
        axes[0].set_ylim(0.0, float(t_h[-1]))
        axes[0].set_xlabel("AP position x (um)", color="#e2e8f0", fontsize=8)
        axes[0].set_ylabel("Time (h)", color="#e2e8f0", fontsize=8)
        axes[0].tick_params(colors="#cbd5e1", labelsize=7)
        axes[0].set_title(
            f"Live HES7 clock (T={ck['period_h']:.2f} h), t = {t_k:.1f} h\n{len(fixed)} MESP2/RIPPLY2 boundaries fixed",
            fontsize=9.0,
            fontweight="bold",
            color="#f8fafc",
        )

        _draw_rosette_polygons(
            axes[1],
            wt_step["polygons"],
            wt_step["mean_r_in_um"],
            wt_step["mean_r_out_um"],
            f"WT MET: live gamma_ap(t)={g_ap:.2f}\nL_ap/L_bas={wt_step['vertex_ap_basal_ratio']:.3f}, R_in={wt_step['mean_r_in_um']:.1f} um",
            face_color="#1d4ed8",
        )
        axes[1].text(0, -31.5, seg_label, ha="center", fontsize=6.8, color="#93c5fd")
        _draw_rosette_polygons(
            axes[2],
            rock_step["polygons"],
            rock_step["mean_r_in_um"],
            rock_step["mean_r_out_um"],
            f"+Y-27632 ROCK Inh.: gamma_ap=0.45\nL_ap/L_bas={rock_step['vertex_ap_basal_ratio']:.3f}, R_in={rock_step['mean_r_in_um']:.1f} um",
            face_color="#475569",
        )
        fig.suptitle(
            f"Somitoid Clock -> Wavefront -> MET Coupling (Frame {idx + 1}/{n_frames})",
            color="#f8fafc",
            fontsize=10.5,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(
            Image.fromarray(rgba[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE)
        )
        plt.close(fig)

    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
    )
    return out_gif


# ============================================================================
# Section 8: Self-Test Suite & CLI Entrypoint
# ============================================================================
def run_self_test() -> None:
    """Run deterministic verification of SHA-256 dataset integrity (3 pinned
    files), 2D vertex solver agreement with analytical wedge-rosette laws,
    EphA4/ephrinB2 cleft barrier positivity, Y-27632 / Minus-Matrigel negative
    controls, specimen segmentation, and (Pass 3) the 3D polyhedral shell
    (Law 4: determinism, analytic agreement, pressure monotonicity, rupture,
    scutoid audit) and the live HES7 clock-wavefront coupling (Law 5: period
    calibration, S = v T, MESP2 half-segments, AP maturation gradient,
    Y-27632 and Eph/ephrin knockdown controls), ending with 10 measurements.
    """
    digests = verify_dataset_integrity()
    assert len(digests) == 3, "Expected 3 verified dataset digests"

    # 1. Verify bitwise two-run determinism of the 2D vertex solver
    sol1 = solve_2d_vertex_rosette(gamma_ap=1.65, gamma_bas=0.45, seed=2021)
    sol2 = solve_2d_vertex_rosette(gamma_ap=1.65, gamma_bas=0.45, seed=2021)
    assert np.allclose(sol1["r_in_vertices"], sol2["r_in_vertices"], atol=0.0)
    assert abs(sol1["vertex_ap_basal_ratio"] - sol1["analytic_polygon_ratio"]) < 0.015

    # 2. Verify Law 1 & Law 3: WT apical constriction < Y-27632 ROCK-inhibited ratio
    pert = evaluate_perturbation_regimes()
    wt = pert["by_id"]["wt_plus_matrigel_somitoid"]
    hyper = pert["by_id"]["high_myosin_hyperconstricted"]
    rock = pert["by_id"]["rock_inhibited_y27632"]
    nomat = pert["by_id"]["minus_matrigel_unpolarized"]
    ephkd = pert["by_id"]["epha4_ephrinb2_kd_fused"]

    assert hyper["predicted_ap_basal_ratio"] < wt["predicted_ap_basal_ratio"] < rock["predicted_ap_basal_ratio"]
    assert nomat["predicted_ap_basal_ratio"] > 0.50
    assert pert["pearson_r"] > 0.99, f"Expected r > 0.99, got {pert['pearson_r']}"

    # 3. Verify Law 2: EphA4/ephrinB2 cleft energy barrier & knockdown collapse
    assert wt["delta_e_cleft"] > 50.0
    assert wt["predicted_cleft_score"] > 0.80
    assert ephkd["predicted_cleft_score"] < 0.15

    # 4. (Pass 3, Law 4) 3D polyhedral shell: determinism, analytic agreement, Y-27632 dilation,
    #    pressure monotonicity & rupture, honest scutoid audit
    shell = build_spherical_voronoi_shell()
    assert shell["N3"] == N3D_CELLS and shell["n_faces"] > 0
    assert abs(float(np.sum(shell["omega"])) - 4.0 * math.pi) < 1e-9, "Voronoi solid angles must tile 4 pi"
    s3a = solve_3d_polyhedral_shell(shell, gamma_ap=1.65, gamma_bas=0.45)
    s3b = solve_3d_polyhedral_shell(shell, gamma_ap=1.65, gamma_bas=0.45)
    assert np.allclose(s3a["r_in_cells"], s3b["r_in_cells"], atol=0.0), "3D shell relaxation must be bitwise deterministic"
    assert s3a["converged"], "3D shell L-BFGS-B must converge"
    assert abs(s3a["shell_ratio_3d"] - s3a["analytic_ratio_3d"]) < 0.005, f"3D discrete vs Law 4 analytic mismatch: {s3a['shell_ratio_3d']} vs {s3a['analytic_ratio_3d']}"
    assert abs(s3a["shell_ratio_3d"] - 0.305) < 0.03, f"3D WT lumen ratio {s3a['shell_ratio_3d']} far from empirical 0.305"
    assert s3a["cell_volume_cv"] < 1e-3, "cell volumes must be held near V_0 by K_V"
    s3r = solve_3d_polyhedral_shell(shell, gamma_ap=0.45, gamma_bas=0.45)
    assert s3r["shell_ratio_3d"] > s3a["shell_ratio_3d"] + 0.25, "Y-27632 must dilate the 3D lumen"
    sw = sweep_lumen_pressure(shell, discrete_at=(0.0, 0.06))
    assert sw["monotone_increasing"], "rho_3D(P) must increase monotonically with lumen pressure (Law 4)"
    assert 0.0 < sw["p_crit"] < sw["laplace_open_bound_p0"], "rupture pressure must be positive and below the Laplace open-shell bound"
    for d in sw["discrete"]:
        assert abs(d["ratio_discrete"] - d["ratio_analytic"]) < 0.005, f"pressurized discrete/analytic mismatch at P={d['P']}"
    sc = compute_scutoid_topology_audit(shell, s3a)
    assert sc["scutoid_fraction"] == 0.0 and sc["apical_basal_neighbour_graph_difference"] == 0

    # 5. (Pass 3, Law 5) live HES7 clock -> wavefront arrest -> MESP2/RIPPLY2 -> tension ramps
    cal = calibrate_hes7_time_scale()
    assert abs(cal["period_h"] - HES7_TARGET_PERIOD_H) < 0.05, f"HES7 period calibration off: {cal['period_h']}"
    assert cal["period_cv"] < 0.01, "HES7 limit cycle must be regular"
    ck1 = simulate_hes7_clock_wavefront(cal["time_scale"])
    ck2 = simulate_hes7_clock_wavefront(cal["time_scale"])
    assert np.array_equal(ck1["hes7_protein_xt"], ck2["hes7_protein_xt"]), "clock simulation must be bitwise deterministic"
    assert ck1["n_boundaries"] >= 4, f"expected >= 4 MESP2 boundaries, got {ck1['n_boundaries']}"
    assert abs(ck1["stripe_period_um"] - ck1["predicted_segment_length_um"]) < PSM_CELL_DX_UM, "segment length must match S = v T within one cell"
    assert all(0.4 <= s_["mesp2_fraction"] <= 0.6 for s_ in ck1["segments"]), "MESP2 lineage must occupy ~half of each segment"
    assert np.all(np.diff(ck1["t_det_h"]) > 0), "boundaries must be fixed in anterior-to-posterior order"
    # Theorem 5 numerically: anterior (earlier-arrested) cells carry larger gamma_ap at every later time
    g_final = ck1["gamma_ap_xt"][-1][ck1["arrested_final"]]
    assert np.all(np.diff(g_final) <= 1e-12), "gamma_ap must decrease monotonically from anterior to posterior"
    cp1 = couple_clock_to_rosette_mechanics(ck1, shell)
    assert cp1["monotone_ap_maturation"], "lumen ratio must tighten monotonically with somite age"
    assert cp1["stages"][-1]["ratio_2d"] < 0.32 and cp1["stages"][0]["ratio_2d"] > cp1["stages"][-1]["ratio_2d"]
    ck_rock = simulate_hes7_clock_wavefront(cal["time_scale"], rock_inhibited=True)
    assert ck_rock["n_boundaries"] == ck1["n_boundaries"], "Y-27632 must leave MESP2 boundary count unchanged"
    cp_rock = couple_clock_to_rosette_mechanics(ck_rock, shell)
    assert all(st["ratio_2d"] > 0.65 for st in cp_rock["stages"]), "Y-27632 coupled rosettes must stay unpolarized"
    ck_kd = simulate_hes7_clock_wavefront(cal["time_scale"], eph_knockdown=True)
    cp_kd = couple_clock_to_rosette_mechanics(ck_kd, shell)
    assert cp_kd["stages"][-1]["cleft_score"] < 0.15 < 0.80 < cp1["stages"][-1]["cleft_score"]

    # 6. Verify empirical specimen segmentation & 10 MorphoEval measurements
    measurements = measure_morphoeval()
    assert len(measurements) == 10, f"Expected 10 measurements, got {len(measurements)}"
    for m in measurements:
        assert math.isfinite(float(m["value"])), f"non-finite measurement {m['name']}"
    print("[self-test] All Pass 3 checks passed (10 morphometric measurements verified):")
    for m in measurements:
        print(f"  - {m['name']}: {m['value']} {m['unit']} (ref={m['reference_value']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Yamanaka (2023), Miao (2023) & Adhyapok (2021) somite rosette vertex mechanics reproduction."
    )
    ap.add_argument("--self-test", action="store_true", help="Run deterministic verification suite and exit")
    args = ap.parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    verify_dataset_integrity()
    p3 = compute_pass3_models()
    png = render_comparison_figure(OUTPUT_PNG, p3=p3)
    clock_png = render_clock_coupling_figure(OUTPUT_CLOCK_PNG, p3=p3)
    gif = render_growth_gif(OUTPUT_GIF, p3=p3)
    print(f"Rendered comparison figure -> {png}")
    print(f"Rendered clock-coupling figure -> {clock_png}")
    print(f"Rendered growth animation  -> {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
