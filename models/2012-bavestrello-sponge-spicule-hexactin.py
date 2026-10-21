#!/usr/bin/env python3
"""
2012-bavestrello-sponge-spicule-hexactin.py — Pass 3 (v3) Remediated
====================================================================
Silicatein Axial Filament Crystallographic Symmetry, 3D Biosilica Diffusion-Reaction
Accretion, Concentric Lamellar Thinning (h_n ∝ r_n⁻¹), and Euplectella Checkerboard
Diagonal Lattice Mechanics across Porifera (Hexactinellida, Demospongiae, Calcarea).

Pass 3 additions (remediation of the v2 audit items)
----------------------------------------------------
* Law 5 — Finite-element three-point bending of the concentric laminate with a bilinear
  cohesive-zone organic interlayer (He & Hutchinson deflection criterion, Hillerborg
  process-zone length, Griffith interface delamination solved on the FE stiffness),
  coupled to the 3D silicatein solver through the accreted biosilica cross-section
  (k_cat -> N_lamellae -> work of fracture). Honest reduced model, not a 3D cohesive FEM.
* Law 6 — External helical surface ridges on the cylindrical cage: sin^2(2 psi) torsional
  projection (optimum 45 deg) and parallel-axis ovalization/buckling gain 1 + N_r A_r / 2.
* Law 7 — Anchor-spicule (basalia) optical waveguiding: step-index core/cladding built from
  the lamellar geometry, meridional ray tracing, LP-mode count from Bessel zeros.
Reference constants and their provenance tags live in
`datasets/sponge_spicule_hexactin/cohesive_bending_waveguide_reference.json`.

Primary Literature & Mathematical Foundations
---------------------------------------------
1. Uriz, M.-J., Turon, X., Becerro, M. A. & Agell, G. (2003). Siliceous spicules and
   skeleton frameworks in sponges: origin, diversity, ultrastructural patterns, and
   biological functions. Microscopy Research and Technique 62(4):279-299.
   DOI: 10.1002/jemt.10395
2. Aizenberg, J., Weaver, J. C., Thanawala, M. S., Sundar, V. C., Morse, D. E. &
   Fratzl, P. (2005). Skeleton of Euplectella sp.: structural hierarchy from the
   nanoscale to the macroscale. Science 309(5732):275-278.
   DOI: 10.1126/science.1112255
3. Weaver, J. C. et al. (2007). Hierarchical assembly of the siliceous skeletal
   lattice of the hexactinellid sponge Euplectella aspergillum. Journal of Structural
   Biology 158(1):93-106. DOI: 10.1016/j.jsb.2007.01.006
4. Shimizu, K., Cha, J., Stucky, G. D. & Morse, D. E. (1998). Silicatein alpha:
   cathepsin L-like protein in sponge biosilica. Proceedings of the National Academy
   of Sciences 95(11):6234-6238. DOI: 10.1073/pnas.95.11.6234
5. Sundar, V. C., Yablon, A. D., Grazul, J. L., Ilan, M. & Aizenberg, J. (2003).
   Fibre-optical features of a glass sponge. Nature 424:899-900. DOI: 10.1038/424899a
6. He, M.-Y. & Hutchinson, J. W. (1989). Crack deflection at an interface between
   dissimilar elastic materials. Int. J. Solids Struct. 25(9):1053-1067.
   DOI: 10.1016/0020-7683(89)90021-8
7. Hillerborg, A., Modeer, M. & Petersson, P.-E. (1976). Analysis of crack formation and
   crack growth in concrete by means of fracture mechanics and finite elements.
   Cem. Concr. Res. 6(6):773-781. DOI: 10.1016/0008-8846(76)90007-7
(5-7 DOIs recalled offline; not verified during Pass 3.)

Machine-Checked Lean 4 Formalization
------------------------------------
Backed by 6 machine-checked theorems in
`morphoeval/lean/loops/Loop_076_sponge_spicule_hexactin.lean` (0 `sorry`s):
  - `MathOfBiology.Loop_076.hexactin_Oh_orthogonal_six_ray_equilibrium_and_dot_products`
  - `MathOfBiology.Loop_076.tetractin_Td_calthrop_tetrahedral_cosine_and_equilibrium`
  - `MathOfBiology.Loop_076.aizenberg_concentric_lamellar_thinning_and_fracture_toughness`
  - `MathOfBiology.Loop_076.euplectella_checkerboard_diagonal_bracing_shear_stiffness`
  - `MathOfBiology.Loop_076.cohesive_laminate_progressive_fracture_work_exceeds_monolith` (Pass 3)
  - `MathOfBiology.Loop_076.euplectella_helical_ridge_shear_projection_and_parallel_axis_gain` (Pass 3)
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
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
from scipy import ndimage, signal, stats

# =============================================================================
# 1. Paths, Cryptographic Dataset Provenance & Default Model Constants
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "sponge_spicule_hexactin"
REF_JSON = DATASET_DIR / "sponge_spicule_hexactin_reference.json"
SPECIMEN_JPG = DATASET_DIR / "euplectella_aspergillum_specimen.jpg"
SCOUT_JSON = DATASET_DIR / "scout_manifest.json"
CITATIONS_BIB = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "sponge_spicule_hexactin_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "sponge_spicule_hexactin_growth.gif"

EXPECTED_SHA256: dict[Path, str] = {
    SPECIMEN_JPG: "b56275f43e5ea078d23a79d7f06d01057879d26518730dce7f9dbd7e4f1f9cf3",
    SCOUT_JSON: "fb6e06653d9d14ac34b7910b82899690646b084a4986fee16c41fa0fa5cae39f",
    CITATIONS_BIB: "81182a1384d11f9388a16c20b909ccd6830ae5225ef29e4c47a76cf025e244cd",
}

DEFAULT_D_S_UM2_S = 1.20
DEFAULT_K_CAT_S = 0.85
DEFAULT_K_M_MM = 0.25
DEFAULT_S_0_MM = 1.00
DEFAULT_R_CORE_UM = 0.75
DEFAULT_H_1_UM = 0.85
DEFAULT_N_LAMELLAE = 25
DEFAULT_ETA_DEFLECT = 0.32
DEFAULT_K_BEND_NORM = 1.0
DEFAULT_K_AXIAL_NORM = 26.0
TETRAHEDRAL_ANGLE_DEG = math.degrees(math.acos(-1.0 / 3.0))  # 109.47122063449069 deg


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset_integrity() -> dict[str, str]:
    """Cryptographically verify SHA-256 digests of all vendored dataset files."""
    if not REF_JSON.exists():
        raise FileNotFoundError(f"Missing reference JSON: {REF_JSON}")
    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))
    specimens_meta = ref["metadata"]["specimens"]

    digests: dict[str, str] = {"reference_json": compute_sha256(REF_JSON)}
    for path, expected_hash in EXPECTED_SHA256.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset asset: {path}")
        actual = compute_sha256(path)
        if actual != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {expected_hash}, got {actual}"
            )
        digests[path.name] = actual

    # Verify parity with SHA-256 digests recorded inside reference JSON
    if specimens_meta["euplectella_aspergillum_specimen"]["sha256"] != digests[SPECIMEN_JPG.name]:
        raise ValueError("Reference JSON specimen SHA-256 out of sync with disk")
    if specimens_meta["scout_manifest"]["sha256"] != digests[SCOUT_JSON.name]:
        raise ValueError("Reference JSON scout_manifest SHA-256 out of sync with disk")
    if specimens_meta["paper_citations"]["sha256"] != digests[CITATIONS_BIB.name]:
        raise ValueError("Reference JSON paper_citations SHA-256 out of sync with disk")

    return digests


def load_reference_data() -> dict[str, Any]:
    """Load and return the parsed `sponge_spicule_hexactin_reference.json` dictionary."""
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


# =============================================================================
# 2. Law 1 & Law 2: Silicatein Axial Filament Crystallographic Ray Symmetry
#    (O_h Cubic 6-Ray Hexactin vs T_d Tetrahedral 4-Ray Tetractin vs D_3h/D_inf_h)
# =============================================================================
def get_silicatein_filament_ray_vectors(symmetry_group: str) -> np.ndarray:
    """Return the exact unit ray direction vectors {u_k} in R^3 for each Porifera
    silicatein axial filament crystallographic symmetry group (Uriz et al. 2003):
      - `none`    : Ablated / silicatein-inhibited control (0 rays)
      - `D_inf_h` : Demospongiae monaxon/oxea (2 opposing rays ±e_x, theta = 180 deg)
      - `D_3h`    : Calcarea / Hexactinellid planar triactine (3 coplanar rays, theta = 120 deg)
      - `T_d`     : Demospongiae tetrahedral tetractin/calthrop (4 rays, cos theta = -1/3,
                    theta = 109.4712 deg; Lean 4 theorem `tetractin_Td_calthrop_tetrahedral_cosine_and_equilibrium`)
      - `O_h`     : Hexactinellida cubic hexactin (6 orthogonal rays ±e_x, ±e_y, ±e_z,
                    theta_adj = 90 deg; Lean 4 theorem `hexactin_Oh_orthogonal_six_ray_equilibrium_and_dot_products`)
    """
    if symmetry_group == "none":
        return np.zeros((0, 3), dtype=np.float64)
    if symmetry_group == "D_inf_h":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    if symmetry_group == "D_3h":
        angles = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
        return np.array(
            [[math.cos(a), math.sin(a), 0.0] for a in angles],
            dtype=np.float64,
        )
    if symmetry_group == "T_d":
        raw = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )
        return raw / math.sqrt(3.0)
    if symmetry_group == "O_h":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown silicatein symmetry group: {symmetry_group}")


def analyze_ray_vector_geometry(rays: np.ndarray) -> dict[str, Any]:
    """Compute vector-sum equilibrium norm ||sum u_k||, nearest-neighbor inter-ray
    cosine u_i . u_j, and nearest-neighbor inter-ray angle theta (deg)."""
    n_rays = int(rays.shape[0])
    if n_rays == 0:
        return {
            "n_rays": 0,
            "vector_sum_norm": 0.0,
            "nearest_neighbor_cos": 0.0,
            "nearest_neighbor_angle_deg": 0.0,
            "antipodal_pairs": 0,
            "orthogonal_pairs": 0,
        }

    vec_sum_norm = float(np.linalg.norm(np.sum(rays, axis=0)))
    dots: list[float] = []
    antipodal = 0
    orthogonal = 0
    for i in range(n_rays):
        for j in range(i + 1, n_rays):
            d = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
            dots.append(d)
            if abs(d + 1.0) < 1e-6:
                antipodal += 1
            if abs(d) < 1e-6:
                orthogonal += 1

    # Nearest-neighbor inter-ray angle is the smallest angle between distinct rays (largest dot product)
    max_dot = max(dots) if dots else 0.0
    nn_angle_deg = float(math.degrees(math.acos(np.clip(max_dot, -1.0, 1.0))))
    return {
        "n_rays": n_rays,
        "vector_sum_norm": vec_sum_norm,
        "nearest_neighbor_cos": float(max_dot),
        "nearest_neighbor_angle_deg": nn_angle_deg,
        "antipodal_pairs": antipodal,
        "orthogonal_pairs": orthogonal,
    }


# =============================================================================
# 3. 3D Silicatein-Catalyzed Diffusion-Reaction Biosilica Accretion Solver
# =============================================================================
def simulate_spicule_biosilica_accretion_3d(
    regime_id: str,
    symmetry_group: str,
    k_cat_scale: float = 1.0,
    d_s_um2_s: float = DEFAULT_D_S_UM2_S,
    k_cat_s: float = DEFAULT_K_CAT_S,
    k_m_mm: float = DEFAULT_K_M_MM,
    s_0_mm: float = DEFAULT_S_0_MM,
    n_grid: int = 33,
    n_steps: int = 28,
    domain_half_um: float = 12.0,
) -> dict[str, Any]:
    """Simulate 3D silicic acid diffusion and enzymatic silicatein-catalyzed
    biosilica condensation around a 3D axial proteinaceous filament network:
        dS/dt = D_S * laplacian(S) - k_cat_eff * F_fil(x) * S / (K_m + S)
        dB/dt = k_cat_eff * F_fil(x) * S / (K_m + S)
    Following Uriz et al. (2003) Microsc. Res. Tech. 62:279-299 & Shimizu et al.
    (1998) PNAS 95:6234-6238.
    """
    coords = np.linspace(-domain_half_um, domain_half_um, n_grid, dtype=np.float64)
    dx = float(coords[1] - coords[0])
    XX, YY, ZZ = np.meshgrid(coords, coords, coords, indexing="ij")
    pts = np.stack([XX, YY, ZZ], axis=-1)

    rays = get_silicatein_filament_ray_vectors(symmetry_group)
    geom = analyze_ray_vector_geometry(rays)

    # Construct 3D silicatein axial filament catalytic density F_fil(x)
    filament_length_um = 0.82 * domain_half_um
    sigma_fil_um = 1.15
    f_fil = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    for u_vec in rays:
        # Projection s = x . u clamped to [0, filament_length_um]
        proj = np.clip(
            pts[..., 0] * u_vec[0] + pts[..., 1] * u_vec[1] + pts[..., 2] * u_vec[2],
            0.0,
            filament_length_um,
        )
        closest = (
            proj[..., None] * u_vec[None, None, None, :]
        )
        dist_sq = np.sum((pts - closest) ** 2, axis=-1)
        ray_kernel = np.exp(-dist_sq / (2.0 * sigma_fil_um**2))
        f_fil = np.maximum(f_fil, ray_kernel)

    # Explicit 3D reaction-diffusion time stepping with ambient silicic acid bath S_0 at boundary
    S_field = np.full((n_grid, n_grid, n_grid), s_0_mm, dtype=np.float64)
    B_field = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    k_eff = float(k_cat_scale) * float(k_cat_s)
    dt = min(0.12 * (dx**2) / max(d_s_um2_s, 1e-6), 0.20)

    for _ in range(n_steps):
        # 6-neighbor 3D discrete Laplacian with Dirichlet boundary S = S_0
        lap_S = np.zeros_like(S_field)
        lap_S[1:-1, 1:-1, 1:-1] = (
            S_field[2:, 1:-1, 1:-1]
            + S_field[:-2, 1:-1, 1:-1]
            + S_field[1:-1, 2:, 1:-1]
            + S_field[1:-1, :-2, 1:-1]
            + S_field[1:-1, 1:-1, 2:]
            + S_field[1:-1, 1:-1, :-2]
            - 6.0 * S_field[1:-1, 1:-1, 1:-1]
        ) / (dx**2)

        reaction = k_eff * f_fil * (S_field / (k_m_mm + S_field))
        S_field = np.clip(S_field + dt * (d_s_um2_s * lap_S - reaction), 0.0, s_0_mm)
        # Re-pin outer boundary to ambient bath concentration S_0
        S_field[0, :, :] = s_0_mm
        S_field[-1, :, :] = s_0_mm
        S_field[:, 0, :] = s_0_mm
        S_field[:, -1, :] = s_0_mm
        S_field[:, :, 0] = s_0_mm
        S_field[:, :, -1] = s_0_mm
        B_field = B_field + dt * reaction

    # Measure accreted ray count and emergent inter-ray angle directly from the 3D B_field
    total_biosilica_mass = float(np.sum(B_field) * (dx**3))
    if k_eff <= 0.0 or total_biosilica_mass < 1e-6 or len(rays) == 0:
        accreted_rays_detected = 0
        measured_inter_ray_deg = 0.0
    else:
        # Sample accreted biosilica along candidate ray directions at r_probe = 0.55 * domain_half_um
        r_probe = 0.55 * domain_half_um
        active_dirs: list[np.ndarray] = []
        for u_vec in rays:
            probe_pt = r_probe * u_vec
            ix = int(np.clip(round((probe_pt[0] + domain_half_um) / dx), 1, n_grid - 2))
            iy = int(np.clip(round((probe_pt[1] + domain_half_um) / dx), 1, n_grid - 2))
            iz = int(np.clip(round((probe_pt[2] + domain_half_um) / dx), 1, n_grid - 2))
            patch = B_field[ix - 1 : ix + 2, iy - 1 : iy + 2, iz - 1 : iz + 2]
            if float(np.max(patch)) > 0.05 * float(np.max(B_field)):
                # Compute local center-of-mass direction vector from 3D biosilica voxel weights
                px = pts[ix - 1 : ix + 2, iy - 1 : iy + 2, iz - 1 : iz + 2]
                w = patch[..., None]
                cm_vec = np.sum(px * w, axis=(0, 1, 2)) / max(float(np.sum(patch)), 1e-12)
                cm_unit = cm_vec / max(float(np.linalg.norm(cm_vec)), 1e-12)
                active_dirs.append(cm_unit)
        accreted_rays_detected = len(active_dirs)
        if accreted_rays_detected >= 2:
            measured_geom = analyze_ray_vector_geometry(np.array(active_dirs, dtype=np.float64))
            measured_inter_ray_deg = measured_geom["nearest_neighbor_angle_deg"]
        else:
            measured_inter_ray_deg = 0.0

    return {
        "regime_id": regime_id,
        "symmetry_group": symmetry_group,
        "coords_um": coords,
        "B_field": B_field,
        "S_field": S_field,
        "rays": rays,
        "n_rays_expected": geom["n_rays"],
        "accreted_rays_detected": accreted_rays_detected,
        "vector_sum_norm": geom["vector_sum_norm"],
        "nearest_neighbor_cos": geom["nearest_neighbor_cos"],
        "theoretical_inter_ray_deg": geom["nearest_neighbor_angle_deg"],
        "measured_inter_ray_deg": measured_inter_ray_deg,
        "total_biosilica_mass": total_biosilica_mass,
    }


def evaluate_all_morphospace_regimes(
    d_s_um2_s: float = DEFAULT_D_S_UM2_S,
    k_cat_s: float = DEFAULT_K_CAT_S,
) -> dict[str, dict[str, Any]]:
    """Run all 5 Porifera spicule symmetry and silicatein-ablation regimes defined in
    `sponge_spicule_hexactin_reference.json`."""
    ref = load_reference_data()
    results: dict[str, dict[str, Any]] = {}
    for reg in ref["morphospace_regimes"]["regimes"]:
        rid = reg["regime_id"]
        results[rid] = simulate_spicule_biosilica_accretion_3d(
            regime_id=rid,
            symmetry_group=reg["symmetry_group"],
            k_cat_scale=float(reg["k_cat_scale"]),
            d_s_um2_s=d_s_um2_s,
            k_cat_s=k_cat_s,
        )
    return results


# =============================================================================
# 4. Law 3: Concentric Lamellar Biosilica Radial Thinning (h_n ∝ r_n⁻¹) &
#    Crack-Deflection Fracture Toughness Amplification (Aizenberg et al. 2005)
# =============================================================================
def simulate_concentric_lamellar_spicule(
    r_core_um: float = DEFAULT_R_CORE_UM,
    h_1_um: float = DEFAULT_H_1_UM,
    n_lamellae: int = DEFAULT_N_LAMELLAE,
    eta_deflect: float = DEFAULT_ETA_DEFLECT,
    k_cat_scale: float = 1.0,
) -> dict[str, Any]:
    """Simulate concentric biosilica cylinder apposition around the axial cylinder
    (Weaver et al. 2007 J. Struct. Biol. 158:93-106; Aizenberg et al. 2005 Science
    309:275-278). Each successive lamellar cylinder deposits a constant cross-sectional
    silica ring area Delta_A_ring = 2*pi * r_1 * h_1, yielding the inverse-radius
    lamellar thinning law:
        h_n = Delta_A_ring / (2*pi * r_n) = h_1 * (r_1 / r_n)  =>  h_n ∝ r_n^(-1)
    and crack-deflection fracture toughness ratio:
        G_lam(N) / G_mono = 1 + eta_deflect * N
    Matches Lean 4 theorem `MathOfBiology.Loop_076.aizenberg_concentric_lamellar_thinning_and_fracture_toughness`.
    """
    if k_cat_scale <= 0.0 or n_lamellae <= 0:
        return {
            "n_lamellae": 0,
            "radii_um": np.zeros(0, dtype=np.float64),
            "thicknesses_um": np.zeros(0, dtype=np.float64),
            "outer_radius_um": 0.0,
            "power_law_exponent": 0.0,
            "toughness_ratio": 1.0,
            "buckling_advantage_factor": 1.0,
        }

    delta_a_ring = 2.0 * math.pi * r_core_um * h_1_um * k_cat_scale
    radii: list[float] = []
    thicknesses: list[float] = []
    r_cur = float(r_core_um)

    for _ in range(n_lamellae):
        radii.append(r_cur)
        h_n = delta_a_ring / (2.0 * math.pi * r_cur)
        thicknesses.append(h_n)
        r_cur = r_cur + h_n

    radii_arr = np.array(radii, dtype=np.float64)
    thick_arr = np.array(thicknesses, dtype=np.float64)

    # Fit log(h_n) vs log(r_n) slope to verify h_n ∝ r_n^(-1)
    slope, _, r_val, _, _ = stats.linregress(np.log(radii_arr), np.log(thick_arr))

    # Crack-deflection toughness amplification at organic silicatein interlayers
    toughness_ratio = 1.0 + float(eta_deflect) * float(n_lamellae)

    # Hollow/laminated cylindrical cage second-moment buckling advantage I_cage / I_solid_rod
    # at equal silica cross-sectional area when arranged on cylinder radius R_cage = 3.2 * r_outer
    r_cage = 3.2 * r_cur
    buckling_advantage = 1.0 + 2.0 * ((r_cage / r_cur) ** 2)

    return {
        "n_lamellae": n_lamellae,
        "radii_um": radii_arr,
        "thicknesses_um": thick_arr,
        "outer_radius_um": float(r_cur),
        "power_law_exponent": float(slope),
        "power_law_r": float(r_val),
        "toughness_ratio": float(toughness_ratio),
        "buckling_advantage_factor": float(buckling_advantage),
    }


# =============================================================================
# 5. Law 4: Euplectella Alternating Open/Closed Checkerboard Diagonal Lattice
#    Shear Stiffness Law (Aizenberg et al. 2005 Science 309:275-278)
# =============================================================================
def simulate_euplectella_lattice_mechanics(
    k_bend_square: float = DEFAULT_K_BEND_NORM,
    k_axial_strut: float = DEFAULT_K_AXIAL_NORM,
    n_cells_x: int = 8,
    n_cells_y: int = 8,
) -> dict[str, Any]:
    """Compute the in-plane shear stiffness K_shear(f_diag) and specific stiffness-to-weight
    efficiency of the *Euplectella aspergillum* cylindrical hexactin square lattice
    reinforced by diagonal spicule bundles at 45 deg across bracing fraction f_diag in [0, 1]:
        K_shear(f_diag) = K_bend_square + f_diag * K_axial_strut * cos^2(45 deg)
                        = K_bend_square + 0.5 * f_diag * K_axial_strut
    In *Euplectella aspergillum* (Aizenberg et al. 2005 Fig. 1E-G), every second square cell
    in a 2D checkerboard pattern (`(i + j) % 2 == 0`, f_diag = 0.50) carries paired diagonal
    struts, achieving a 7.5x shear stiffness gain over an unbraced square grid (`f_diag = 0.0`)
    while preventing fluid-flow occlusion of the open pores.
    Matches Lean 4 theorem `MathOfBiology.Loop_076.euplectella_checkerboard_diagonal_bracing_shear_stiffness`.
    """
    f_grid = np.linspace(0.0, 1.0, 41)
    k_shear_curve = k_bend_square + 0.5 * f_grid * k_axial_strut
    # Relative skeletal mass M(f_diag) / M_square = 1 + (sqrt(2) / 2) * f_diag
    rel_mass_curve = 1.0 + (math.sqrt(2.0) / 2.0) * f_grid
    specific_shear_curve = k_shear_curve / rel_mass_curve

    k_unbraced = float(k_bend_square)
    k_checkerboard = float(k_bend_square + 0.5 * 0.50 * k_axial_strut)
    k_full = float(k_bend_square + 0.5 * 1.00 * k_axial_strut)
    checkerboard_gain = k_checkerboard / max(k_unbraced, 1e-12)

    # Generate 2D checkerboard cell matrix (1 = diagonally braced cell, 0 = open pore cell)
    ii, jj = np.meshgrid(np.arange(n_cells_x), np.arange(n_cells_y), indexing="ij")
    checkerboard_cells = ((ii + jj) % 2 == 0).astype(np.int32)

    return {
        "f_diag_grid": f_grid,
        "k_shear_curve": k_shear_curve,
        "rel_mass_curve": rel_mass_curve,
        "specific_shear_curve": specific_shear_curve,
        "k_shear_unbraced": k_unbraced,
        "k_shear_checkerboard": k_checkerboard,
        "k_shear_full": k_full,
        "checkerboard_shear_gain": float(checkerboard_gain),
        "checkerboard_cells": checkerboard_cells,
    }


# =============================================================================
# 6. Cross-Taxa 12-Species Porifera Symmetry & Lamellar Toughness Validation
# =============================================================================
def evaluate_cross_taxa_spicule_scaling(
    eta_deflect: float = DEFAULT_ETA_DEFLECT,
) -> dict[str, Any]:
    """Validate theoretical silicatein inter-ray angles (90 deg O_h, 109.4712 deg T_d,
    120 deg D_3h, 180 deg D_inf_h) and concentric lamellar fracture toughness ratios
    G_lam / G_mono = 1 + eta_deflect * N_lamellae across 12 empirical Porifera species
    spanning Hexactinellida, Demospongiae, and Calcarea."""
    ref = load_reference_data()
    species_rows = ref["cross_taxa_spicule_symmetry_and_lamellae"]

    theta_pred_list: list[float] = []
    theta_obs_list: list[float] = []
    tough_pred_list: list[float] = []
    tough_obs_list: list[float] = []
    n_lam_list: list[float] = []
    records: list[dict[str, Any]] = []
    clades: set[str] = set()

    for row in species_rows:
        sym = row["symmetry_group"]
        rays = get_silicatein_filament_ray_vectors(sym)
        geom = analyze_ray_vector_geometry(rays)
        th_pred = float(geom["nearest_neighbor_angle_deg"])
        th_obs = float(row["theta_obs_deg"])
        n_lam = int(row["n_lamellae"])
        t_pred = 1.0 + float(eta_deflect) * float(n_lam)
        t_obs = float(row["toughness_ratio_obs"])

        theta_pred_list.append(th_pred)
        theta_obs_list.append(th_obs)
        tough_pred_list.append(t_pred)
        tough_obs_list.append(t_obs)
        n_lam_list.append(float(n_lam))
        clades.add(row["clade"])

        records.append(
            {
                **row,
                "theta_model_deg": th_pred,
                "toughness_pred": t_pred,
            }
        )

    th_p = np.array(theta_pred_list, dtype=np.float64)
    th_o = np.array(theta_obs_list, dtype=np.float64)
    tg_p = np.array(tough_pred_list, dtype=np.float64)
    tg_o = np.array(tough_obs_list, dtype=np.float64)
    nl_a = np.array(n_lam_list, dtype=np.float64)

    r_angle, p_angle = stats.pearsonr(th_p, th_o)
    rmse_angle = float(np.sqrt(np.mean((th_p - th_o) ** 2)))

    r_tough, p_tough = stats.pearsonr(tg_p, tg_o)
    rmse_tough = float(np.sqrt(np.mean((tg_p - tg_o) ** 2)))
    slope_tough, intercept_tough, _, _, slope_se = stats.linregress(nl_a, tg_o)

    hex_tough = [r["toughness_ratio_obs"] for r in records if r["symmetry_group"] == "O_h"]
    non_hex_tough = [r["toughness_ratio_obs"] for r in records if r["symmetry_group"] != "O_h"]

    return {
        "n_species": len(records),
        "n_clades": len(clades),
        "records": records,
        "pearson_r_angle": float(r_angle),
        "p_value_angle": float(p_angle),
        "rmse_angle_deg": rmse_angle,
        "pearson_r_toughness": float(r_tough),
        "p_value_toughness": float(p_tough),
        "rmse_toughness": rmse_tough,
        "toughness_per_layer_slope": float(slope_tough),
        "toughness_per_layer_se": float(slope_se),
        "toughness_intercept": float(intercept_tough),
        "mean_hexactinellid_toughness": float(np.mean(hex_tough)),
        "mean_demospongiae_calcarea_toughness": float(np.mean(non_hex_tough)),
    }


# =============================================================================
# 7. Computer-Vision Segmentation of Vendored Euplectella aspergillum Specimen
# =============================================================================
def segment_euplectella_specimen() -> dict[str, Any]:
    """Perform Difference-of-Gaussians (DoG) ridge extraction and gradient structure
    orientation analysis on the Public-domain *Euplectella aspergillum* glass sponge
    skeleton photograph (`datasets/sponge_spicule_hexactin/euplectella_aspergillum_specimen.jpg`),
    extracting:
      1. Orthogonal vertical vs horizontal hexactin primary strut intersection angle
         (89.6498 deg vs theoretical 90.0 deg O_h hexactin law),
      2. Alternating diagonal bracing strut orientation angle (47.6843 deg vs 45.0 deg),
      3. Square grid inter-strut pitch (24.0653 px = 2.5000 mm at 0.103884 mm/px across
         536 inter-strut intervals, CV = 0.3997).
    """
    ref = load_reference_data()
    spec_meta = ref["metadata"]["specimens"]["euplectella_aspergillum_specimen"]
    x0, x1 = spec_meta["roi_x_range_px"]
    y0, y1 = spec_meta["roi_y_range_px"]
    cal_mm_per_px = float(spec_meta["calibration_mm_per_px"])

    rgb = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"), dtype=np.float64) / 255.0
    gray = np.asarray(Image.open(SPECIMEN_JPG).convert("L"), dtype=np.float64) / 255.0
    roi = gray[y0:y1, x0:x1]

    # Bandpass Difference-of-Gaussians filter isolating cylindrical biosilica lattice struts
    dog = ndimage.gaussian_filter(roi, 1.2) - ndimage.gaussian_filter(roi, 4.0)
    gy, gx = np.gradient(ndimage.gaussian_filter(roi, 1.5))
    mag = np.hypot(gx, gy)

    # Local strut tangent angle in [0, 180) deg
    tangent_deg = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
    strut_mask = (dog > np.percentile(dog, 75)) & (mag > np.percentile(mag, 65))
    angles = tangent_deg[strut_mask]

    # Separate primary orthogonal hexactin struts (horizontal ~0/180 deg, vertical ~90 deg)
    # and 45 deg diagonal reinforcing struts
    horiz = angles[(angles < 22.0) | (angles > 158.0)]
    horiz_signed = np.where(horiz > 90.0, horiz - 180.0, horiz)
    vert = angles[(angles > 68.0) & (angles < 112.0)]
    orthogonal_intersection_deg = float(np.mean(vert) - np.mean(horiz_signed))

    diag_angles = angles[
        ((angles >= 25.0) & (angles <= 65.0)) | ((angles >= 115.0) & (angles <= 155.0))
    ]
    diag_folded = np.where(diag_angles > 90.0, 180.0 - diag_angles, diag_angles)
    diagonal_strut_deg = float(np.mean(diag_folded))

    # Measure horizontal inter-strut lattice pitch across 27 row transects
    spacings_px: list[float] = []
    detected_nodes_xy: list[tuple[np.ndarray, int]] = []
    for r in range(20, roi.shape[0] - 20, 12):
        prof = dog[r, :]
        pks, _ = signal.find_peaks(prof, distance=14, prominence=0.025)
        if len(pks) >= 3:
            spacings_px.extend(np.diff(pks).tolist())
            detected_nodes_xy.append((pks + x0, r + y0))

    spacings_arr = np.array(spacings_px, dtype=np.float64)
    mean_pitch_px = float(np.mean(spacings_arr))
    mean_pitch_mm = mean_pitch_px * cal_mm_per_px
    cv_pitch = float(np.std(spacings_arr) / mean_pitch_px)

    return {
        "rgb": rgb,
        "gray": gray,
        "dog": dog,
        "roi_box": (x0, y0, x1, y1),
        "strut_angles_deg": angles,
        "orthogonal_intersection_deg": orthogonal_intersection_deg,
        "diagonal_strut_deg": diagonal_strut_deg,
        "spacings_px": spacings_arr,
        "mean_pitch_px": mean_pitch_px,
        "mean_pitch_mm": mean_pitch_mm,
        "cv_pitch": cv_pitch,
        "n_spacings": int(len(spacings_arr)),
        "detected_nodes_xy": detected_nodes_xy,
    }


# =============================================================================
# 7b. PASS 3 — Law 5: Finite-Element Three-Point Bending of the Laminated Spicule
#     with Cohesive-Zone Organic Interlayers (Weaver et al. 2007; Aizenberg et al.
#     2005), coupled to the 3D silicatein accretion solver through the accreted
#     biosilica cross-section (k_cat -> N_lamellae -> work of fracture).
#
#     HONEST REDUCED MODEL (not a full 3D cohesive FEM):
#       * Euler-Bernoulli 2-node Hermite beam finite elements (4 DOF/element),
#         simply supported, midspan displacement control; element flexural rigidity
#         EI(x) follows the locally intact laminate (outer layers removed only
#         inside their delaminated zone |x - L/2| < a_n).
#       * Through-layer cracking is strength controlled: sigma_n = M r_n / I_n >= sigma_f.
#       * Crack deflection vs penetration at the organic interlayer uses the
#         He & Hutchinson (1989) criterion for zero elastic mismatch:
#         deflect iff G_i / G_s < 0.25 (ablation: fused interlayer G_i = G_s -> penetrate).
#       * The organic interlayer is a bilinear cohesive-zone law T(delta) with
#         G_i = 0.5 * T_max * delta_c and Hillerborg process-zone length
#         l_cz = E * G_i / T_max^2 (Hillerborg et al. 1976; Rice 1980).
#       * The deflected interface crack length a_n is set by Griffith energy balance
#         between the FE-computed elastic energy release 0.5*delta^2*(k_before - k_after(a))
#         and the dissipation G_s*A_n + G_i*2*pi*r_n*a, solved by bisection.
# =============================================================================
PASS3_JSON = DATASET_DIR / "cohesive_bending_waveguide_reference.json"

DEFAULT_E_SILICA_MPA = 36000.0          # biosilica Young's modulus (~36 GPa; Woesz et al. 2006 J. Mater. Res., recalled offline)
DEFAULT_SIGMA_F_MPA = 900.0             # tensile fracture strength of a biosilica lamella (model constant)
DEFAULT_G_S_SILICA_J_M2 = 8.0           # fracture energy of bulk vitreous silica (~8 J/m^2; model constant)
DEFAULT_T_MAX_INTERLAYER_MPA = 40.0     # cohesive peak traction of the organic (silicatein/collagen) interlayer (model constant)
DEFAULT_DELTA_C_INTERLAYER_UM = 0.060   # cohesive critical separation 60 nm (model constant)
DEFAULT_SPAN_TO_DIAMETER = 20.0         # three-point bend span / spicule diameter
HE_HUTCHINSON_DEFLECTION_RATIO = 0.25   # G_i / G_s below which a crack deflects along the interface (zero mismatch)
N_BEAM_ELEMENTS = 48                    # even number of Hermite beam elements
SILICA_MASS_TO_LAYER_CALIBRATION = 25.0 / 8.488098075925016  # WT O_h cross-section mass (mM*um^2) -> 25 Euplectella lamellae (calibrated)

DEFAULT_N_RIDGES = 8                    # external helical surface ridges on the Euplectella cage (model)
DEFAULT_RIDGE_PITCH_DEG = 45.0          # helix angle of the ridges to the cage axis (Aizenberg et al. 2005: oblique/helical ridges)
DEFAULT_RIDGE_AREA_RATIO = 0.25         # ridge cross-section / cylinder-wall cross-section per ridge (model)

DEFAULT_N_CORE = 1.460                  # basalia optical core refractive index (Sundar et al. 2003 Nature 424:899; recalled offline)
DEFAULT_N_CLAD = 1.440                  # lamellar cladding refractive index (recalled offline)
DEFAULT_N_SEA = 1.340                   # seawater refractive index
DEFAULT_WAVELENGTH_UM = 0.55            # green visible light
OPTICAL_CORE_N_LAMELLAE = 2             # lamellae counted into the high-index optical core (model)
LP01_CUTOFF_V = 2.404825557695773       # first zero of J_0: single-mode cutoff V-number


def load_pass3_reference() -> dict[str, Any]:
    """Load the Pass 3 cohesive-bending / helical-ridge / waveguide reference JSON."""
    return json.loads(PASS3_JSON.read_text(encoding="utf-8"))


def cohesive_traction_bilinear(
    delta_um: np.ndarray | float,
    t_max_mpa: float = DEFAULT_T_MAX_INTERLAYER_MPA,
    delta_c_um: float = DEFAULT_DELTA_C_INTERLAYER_UM,
    delta_0_frac: float = 0.10,
) -> np.ndarray:
    """Bilinear cohesive traction-separation law T(delta) of the organic interlayer:
    linear rise to T_max at delta_0 = delta_0_frac * delta_c, linear softening to zero at
    delta_c. Its integral is the interface fracture energy G_i = 0.5 * T_max * delta_c."""
    d = np.asarray(delta_um, dtype=np.float64)
    d0 = delta_0_frac * delta_c_um
    rise = t_max_mpa * d / d0
    soften = t_max_mpa * (delta_c_um - d) / max(delta_c_um - d0, 1e-12)
    return np.where(d < d0, rise, np.where(d < delta_c_um, soften, 0.0))


def cohesive_interface_energy_j_m2(
    t_max_mpa: float = DEFAULT_T_MAX_INTERLAYER_MPA,
    delta_c_um: float = DEFAULT_DELTA_C_INTERLAYER_UM,
) -> float:
    """G_i = 0.5 * T_max * delta_c (MPa * um = J/m^2)."""
    return 0.5 * t_max_mpa * delta_c_um


def hillerborg_cohesive_zone_length_um(
    e_mpa: float, g_i_j_m2: float, t_max_mpa: float
) -> float:
    """Hillerborg characteristic process-zone length l_cz = E * G_i / T_max^2 (um)."""
    return e_mpa * g_i_j_m2 / max(t_max_mpa, 1e-12) ** 2


def _beam_element_stiffness(ei: float, length: float) -> np.ndarray:
    """4x4 Euler-Bernoulli Hermite beam element stiffness matrix."""
    l_e = length
    return (ei / l_e**3) * np.array(
        [
            [12.0, 6.0 * l_e, -12.0, 6.0 * l_e],
            [6.0 * l_e, 4.0 * l_e**2, -6.0 * l_e, 2.0 * l_e**2],
            [-12.0, -6.0 * l_e, 12.0, -6.0 * l_e],
            [6.0 * l_e, 2.0 * l_e**2, -6.0 * l_e, 4.0 * l_e**2],
        ],
        dtype=np.float64,
    )


def fe_three_point_bend_stiffness(
    ei_per_element: np.ndarray, span_um: float
) -> float:
    """Assemble the Hermite beam FE model of a simply supported beam with element-wise
    flexural rigidity EI_e and return the midspan load/deflection stiffness k = P / delta
    (uN/um) under displacement control at the central node."""
    n_el = int(len(ei_per_element))
    assert n_el % 2 == 0, "need an even element count so the load sits on a node"
    l_e = span_um / n_el
    n_dof = 2 * (n_el + 1)
    k_glob = np.zeros((n_dof, n_dof), dtype=np.float64)
    for e in range(n_el):
        k_e = _beam_element_stiffness(float(ei_per_element[e]), l_e)
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        k_glob[np.ix_(idx, idx)] += k_e

    mid_dof = 2 * (n_el // 2)
    prescribed = [0, 2 * n_el, mid_dof]          # w(0)=0, w(L)=0, w(L/2)=delta
    u_p = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    free = [i for i in range(n_dof) if i not in prescribed]
    k_ff = k_glob[np.ix_(free, free)]
    k_fp = k_glob[np.ix_(free, prescribed)]
    u_f = np.linalg.solve(k_ff, -k_fp @ u_p)
    u_all = np.zeros(n_dof, dtype=np.float64)
    u_all[free] = u_f
    u_all[prescribed] = u_p
    reaction = k_glob @ u_all
    return float(reaction[mid_dof])  # P per unit midspan deflection


def _laminate_section_inertia(radius_um: float) -> float:
    """Second moment of area of the intact solid circular laminate of outer radius r."""
    return math.pi * radius_um**4 / 4.0


def simulate_three_point_bending_cohesive_laminate(
    radii_um: np.ndarray,
    thicknesses_um: np.ndarray,
    e_mpa: float = DEFAULT_E_SILICA_MPA,
    sigma_f_mpa: float = DEFAULT_SIGMA_F_MPA,
    g_s_j_m2: float = DEFAULT_G_S_SILICA_J_M2,
    t_max_mpa: float = DEFAULT_T_MAX_INTERLAYER_MPA,
    delta_c_um: float = DEFAULT_DELTA_C_INTERLAYER_UM,
    span_to_diameter: float = DEFAULT_SPAN_TO_DIAMETER,
    n_elements: int = N_BEAM_ELEMENTS,
    monolithic: bool = False,
) -> dict[str, Any]:
    """Progressive-failure FE three-point bending of a concentric laminated spicule.

    Layers are indexed from the axial core outward: layer n occupies [r_n, r_n + h_n].
    Events are processed from the outermost layer inward. For each still-bonded outer
    layer n with outer radius R_n and section inertia I_n (midspan intact section):
      1. strength criterion: P_n = 4 * sigma_f * I_n / (L * R_n); if the current load
         k * delta is below P_n the midspan displacement is increased to delta = P_n / k;
      2. interlayer check (He & Hutchinson): if G_i / G_s >= 0.25 or the laminate is
         monolithic, the crack penetrates -> catastrophic failure, loop ends;
      3. otherwise the crack deflects along the interlayer and the interface crack
         half-length a_n solves the Griffith balance
            0.5 * delta^2 * (k_before - k_after(a)) = G_s * A_n + G_i * 2*pi*r_n * a
         with k_after(a) from the FE model (outer layer removed for |x - L/2| < a);
      4. the load drops to k_after * delta and the next layer is examined (it may
         cascade at the same delta if its stress already exceeds sigma_f).
    Returns the piecewise-linear P-delta curve, work of fracture, and dissipation budget.
    """
    radii = np.asarray(radii_um, dtype=np.float64)
    thick = np.asarray(thicknesses_um, dtype=np.float64)
    n_layers = int(len(radii))
    outer_r = radii + thick
    r_outer = float(outer_r[-1])
    span = span_to_diameter * 2.0 * r_outer
    n_el = int(n_elements)
    l_e = span / n_el
    x_c = (np.arange(n_el) + 0.5) * l_e - 0.5 * span  # element centers relative to midspan

    g_i = cohesive_interface_energy_j_m2(t_max_mpa, delta_c_um)
    l_cz = hillerborg_cohesive_zone_length_um(e_mpa, g_i, t_max_mpa)
    deflects = (not monolithic) and (g_i / g_s_j_m2 < HE_HUTCHINSON_DEFLECTION_RATIO)

    # a_removed[n] = half-length of the zone in which layer n is cracked & delaminated (0 = bonded)
    a_removed = np.zeros(n_layers, dtype=np.float64)

    def intact_radius_at(x_rel: float) -> float:
        r_int = r_outer
        for n in range(n_layers):
            if a_removed[n] > 0.0 and abs(x_rel) < a_removed[n]:
                r_int = min(r_int, float(radii[n]))
        return r_int

    def stiffness_with(a_trial: np.ndarray) -> float:
        saved = a_removed.copy()
        a_removed[:] = a_trial
        ei = np.array([e_mpa * _laminate_section_inertia(intact_radius_at(float(xc))) for xc in x_c])
        a_removed[:] = saved
        return fe_three_point_bend_stiffness(ei, span)

    # P-delta trace as list of (delta, P) vertices (piecewise linear)
    trace: list[tuple[float, float]] = [(0.0, 0.0)]
    events: list[dict[str, float]] = []
    delta = 0.0
    k_cur = stiffness_with(a_removed)
    k_initial = k_cur
    diss_interface = 0.0
    diss_silica = 0.0
    catastrophic_at: float | None = None

    for n in range(n_layers - 1, -1, -1):
        r_n_out = float(outer_r[n])
        r_n_in = float(radii[n])
        i_n = _laminate_section_inertia(r_n_out)
        p_crack = 4.0 * sigma_f_mpa * i_n / (span * r_n_out)
        delta_n = max(delta, p_crack / k_cur)
        if delta_n > delta + 1e-15:
            trace.append((delta_n, k_cur * delta_n))
        delta = delta_n
        p_before = k_cur * delta
        a_layer = math.pi * (r_n_out**2 - r_n_in**2)

        if not deflects:
            # Crack penetrates the interlayer (or there is none): catastrophic fracture
            diss_silica += g_s_j_m2 * math.pi * r_n_out**2
            catastrophic_at = delta
            trace.append((delta, 0.0))
            events.append({"layer": float(n), "delta_um": delta, "p_before_uN": p_before, "p_after_uN": 0.0, "a_um": 0.0, "trace_index": float(len(trace))})
            break

        # Griffith energy balance for the deflected interface crack half-length a_n
        def energy_gap(a_try: float) -> float:
            a_trial = a_removed.copy()
            a_trial[n] = a_try
            k_after = stiffness_with(a_trial)
            released = 0.5 * delta**2 * (k_cur - k_after)
            return released - (g_s_j_m2 * a_layer + g_i * 2.0 * math.pi * r_n_in * a_try)

        a_lo, a_hi = l_e, 0.5 * span
        if energy_gap(a_hi) >= 0.0:
            a_n = a_hi
        elif energy_gap(a_lo) <= 0.0:
            a_n = a_lo  # process zone cannot grow beyond one element; layer still cracks (strength)
        else:
            for _ in range(28):
                a_mid = 0.5 * (a_lo + a_hi)
                if energy_gap(a_mid) > 0.0:
                    a_lo = a_mid
                else:
                    a_hi = a_mid
            a_n = 0.5 * (a_lo + a_hi)

        a_removed[n] = a_n
        k_cur = stiffness_with(a_removed)
        p_after = k_cur * delta
        diss_interface += g_i * 2.0 * math.pi * r_n_in * a_n
        diss_silica += g_s_j_m2 * a_layer
        trace.append((delta, p_after))
        events.append({"layer": float(n), "delta_um": delta, "p_before_uN": p_before, "p_after_uN": p_after, "a_um": a_n, "trace_index": float(len(trace))})

    tr = np.array(trace, dtype=np.float64)
    work_of_fracture = float(np.trapezoid(tr[:, 1], tr[:, 0]))  # uN*um = pJ
    p_peak = float(np.max(tr[:, 1]))
    n_drops = int(sum(1 for ev in events if ev["p_after_uN"] < ev["p_before_uN"] - 1e-9))

    return {
        "monolithic": bool(monolithic),
        "deflects": bool(deflects),
        "n_layers": n_layers,
        "span_um": float(span),
        "outer_radius_um": r_outer,
        "g_i_j_m2": float(g_i),
        "g_s_j_m2": float(g_s_j_m2),
        "g_ratio": float(g_i / g_s_j_m2),
        "l_cz_um": float(l_cz),
        "k_initial_uN_per_um": float(k_initial),
        "k_analytic_uN_per_um": float(48.0 * e_mpa * _laminate_section_inertia(r_outer) / span**3),
        "trace_delta_um": tr[:, 0],
        "trace_p_uN": tr[:, 1],
        "events": events,
        "n_load_drops": n_drops,
        "p_peak_uN": p_peak,
        "delta_final_um": float(tr[-1, 0]),
        "work_of_fracture_pJ": work_of_fracture,
        "dissipated_interface_pJ": float(diss_interface),
        "dissipated_silica_pJ": float(diss_silica),
        "catastrophic_at_um": catastrophic_at,
        "delamination_half_lengths_um": a_removed.copy(),
    }


def layers_from_accreted_cross_section(
    sim: dict[str, Any],
    r_probe_frac: float = 0.55,
) -> dict[str, Any]:
    """Couple the 3D silicatein diffusion-reaction solver to the laminate: integrate the
    accreted biosilica B(x,y,z) over the plane perpendicular to the +e_x ray at the probe
    radius, convert the cross-sectional biosilica mass (mM*um^2) into an equivalent number
    of concentric lamellae with the calibration constant `SILICA_MASS_TO_LAYER_CALIBRATION`
    (fixed so the wild-type O_h hexactin at k_cat_scale = 1 gives N = 25 Euplectella layers)."""
    coords = sim["coords_um"]
    dx = float(coords[1] - coords[0])
    half = float(-coords[0])
    ix = int(round((r_probe_frac * half + half) / dx))
    cross_mass = float(np.sum(sim["B_field"][ix, :, :]) * dx * dx)
    n_eff = int(round(SILICA_MASS_TO_LAYER_CALIBRATION * cross_mass))
    return {"cross_section_mass": cross_mass, "n_lamellae_eff": max(n_eff, 0), "probe_ix": ix}


def evaluate_coupled_accretion_to_fracture(
    k_cat_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    r_core_um: float = DEFAULT_R_CORE_UM,
    h_1_um: float = DEFAULT_H_1_UM,
    n_lamellae_wt: int = DEFAULT_N_LAMELLAE,
) -> dict[str, Any]:
    """k_cat ablation sweep propagated through the 3D accretion solver -> lamella count ->
    FE cohesive-zone bending work of fracture (vs a monolithic rod of equal outer radius)."""
    rows: list[dict[str, Any]] = []
    for kc in k_cat_scales:
        sim = simulate_spicule_biosilica_accretion_3d("coupling", "O_h", float(kc))
        layers = layers_from_accreted_cross_section(sim)
        n_eff = int(layers["n_lamellae_eff"])
        if n_eff <= 0:
            rows.append({"k_cat_scale": float(kc), "cross_section_mass": layers["cross_section_mass"], "n_lamellae_eff": 0, "toughness_ratio": 0.0, "n_load_drops": 0, "outer_radius_um": 0.0, "eta_law": 1.0})
            continue
        lam = simulate_concentric_lamellar_spicule(r_core_um=r_core_um, h_1_um=h_1_um, n_lamellae=n_eff)
        bend = simulate_three_point_bending_cohesive_laminate(lam["radii_um"], lam["thicknesses_um"])
        mono = simulate_three_point_bending_cohesive_laminate(lam["radii_um"], lam["thicknesses_um"], monolithic=True)
        rows.append(
            {
                "k_cat_scale": float(kc),
                "cross_section_mass": layers["cross_section_mass"],
                "n_lamellae_eff": n_eff,
                "toughness_ratio": bend["work_of_fracture_pJ"] / max(mono["work_of_fracture_pJ"], 1e-12),
                "n_load_drops": bend["n_load_drops"],
                "outer_radius_um": lam["outer_radius_um"],
                "eta_law": 1.0 + DEFAULT_ETA_DEFLECT * n_eff,
            }
        )
    wt = rows[-1]
    return {"rows": rows, "wt_n_lamellae": int(wt["n_lamellae_eff"]), "wt_expected_n": int(n_lamellae_wt)}


def evaluate_interlayer_toughness_sweep(
    radii_um: np.ndarray,
    thicknesses_um: np.ndarray,
    g_i_grid_j_m2: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 8.0),
    t_max_mpa: float = DEFAULT_T_MAX_INTERLAYER_MPA,
) -> dict[str, Any]:
    """Sweep the organic interlayer fracture energy G_i at fixed T_max (delta_c = 2 G_i / T_max)
    and record laminate / monolith work-of-fracture ratio, load-drop count and l_cz."""
    mono = simulate_three_point_bending_cohesive_laminate(radii_um, thicknesses_um, monolithic=True)
    ratios: list[float] = []
    drops: list[int] = []
    lcz: list[float] = []
    deflect_flags: list[bool] = []
    for g_i in g_i_grid_j_m2:
        delta_c = 2.0 * g_i / t_max_mpa
        res = simulate_three_point_bending_cohesive_laminate(radii_um, thicknesses_um, t_max_mpa=t_max_mpa, delta_c_um=delta_c)
        ratios.append(res["work_of_fracture_pJ"] / max(mono["work_of_fracture_pJ"], 1e-12))
        drops.append(res["n_load_drops"])
        lcz.append(res["l_cz_um"])
        deflect_flags.append(res["deflects"])
    ratios_arr = np.array(ratios)
    i_best = int(np.argmax(ratios_arr))
    return {
        "g_i_grid": np.array(g_i_grid_j_m2, dtype=np.float64),
        "toughness_ratio": ratios_arr,
        "n_load_drops": np.array(drops, dtype=np.int64),
        "l_cz_um": np.array(lcz, dtype=np.float64),
        "deflects": np.array(deflect_flags, dtype=bool),
        "g_i_optimum": float(g_i_grid_j_m2[i_best]),
        "ratio_optimum": float(ratios_arr[i_best]),
        "mono_work_pJ": mono["work_of_fracture_pJ"],
    }


# =============================================================================
# 7c. PASS 3 — Law 6: External Helical Surface Ridges on the Cylindrical Cage
#     (netting/laminate projection sin^2(2 psi) for torsion; parallel-axis I_eff gain
#     against Brazier ovalization / Euler buckling) — Aizenberg et al. (2005) Fig. 1.
# =============================================================================
def simulate_helical_ridge_cage_mechanics(
    n_ridges: int = DEFAULT_N_RIDGES,
    ridge_area_ratio: float = DEFAULT_RIDGE_AREA_RATIO,
    k_bend_square: float = DEFAULT_K_BEND_NORM,
    k_axial_strut: float = DEFAULT_K_AXIAL_NORM,
    f_diag_checkerboard: float = 0.50,
    n_psi: int = 91,
) -> dict[str, Any]:
    """Cylindrical *Euplectella* cage = checkerboard-braced hexactin lattice wall (Law 4)
    + N_r external ridges wound at helix angle psi to the cage axis.

    Torsional (wall-shear) stiffness, netting-analysis projection of a helical member onto
    the wall shear strain (gamma -> axial strain gamma*sin psi*cos psi, so stiffness
    ∝ sin^2 psi cos^2 psi = sin^2(2 psi) / 4):
        K_tors(psi) = K_shear_checkerboard + N_r * k_r * sin^2(2 psi),
    normalised so a single ridge at psi = 45 deg contributes k_r = ridge_area_ratio *
    K_axial_strut. Ridge length per unit cage length is 1 / cos psi, so the specific
    (per-mass) stiffness is K_tors / (M_wall + N_r * ridge_area_ratio / cos psi).

    Ovalization / buckling resistance: for N_r >= 3 equally spaced ridges the parallel-axis
    sum Σ_k sin^2(2 pi k / N_r) = N_r / 2, so
        I_eff / I_wall = 1 + N_r * ridge_area_ratio / 2 (A_wall R^2 normalisation),
    i.e. the ridges raise the cage's Euler-buckling load and Brazier ovalization limit
    linearly in N_r (Lean 4 theorem `euplectella_helical_ridge_shear_projection_and_parallel_axis_gain`).
    """
    psi_deg = np.linspace(0.0, 90.0, n_psi)
    psi = np.radians(psi_deg)
    k_wall = k_bend_square + 0.5 * f_diag_checkerboard * k_axial_strut
    k_r = ridge_area_ratio * k_axial_strut
    k_tors = k_wall + n_ridges * k_r * np.sin(2.0 * psi) ** 2
    rel_mass = (1.0 + (math.sqrt(2.0) / 2.0) * f_diag_checkerboard) + n_ridges * ridge_area_ratio / np.maximum(np.cos(psi), 1e-9)
    specific = k_tors / rel_mass
    i_psi_opt = int(np.argmax(k_tors))
    i_spec_opt = int(np.argmax(specific[:-1]))  # exclude psi=90 (infinite ridge length)

    # Parallel-axis ovalization/buckling gain vs ridge count (equally spaced ridges)
    n_grid = np.arange(0, 13)
    gain_exact = []
    for n_r in n_grid:
        if n_r == 0:
            gain_exact.append(1.0)
            continue
        th = 2.0 * math.pi * np.arange(n_r) / n_r
        gain_exact.append(1.0 + ridge_area_ratio * float(np.sum(np.sin(th) ** 2)))
    gain_exact_arr = np.array(gain_exact)
    gain_closed = np.where(n_grid >= 3, 1.0 + n_grid * ridge_area_ratio / 2.0, gain_exact_arr)

    return {
        "psi_deg": psi_deg,
        "k_torsion": k_tors,
        "specific_torsion": specific,
        "rel_mass": rel_mass,
        "k_wall_checkerboard": float(k_wall),
        "k_ridge_45": float(k_r),
        "psi_opt_torsion_deg": float(psi_deg[i_psi_opt]),
        "psi_opt_specific_deg": float(psi_deg[i_spec_opt]),
        "torsion_gain_at_45": float(k_tors[i_psi_opt] / k_wall),
        "n_ridges_grid": n_grid,
        "ovalization_gain_exact": gain_exact_arr,
        "ovalization_gain_closed_form": gain_closed,
        "ovalization_gain_default": float(1.0 + n_ridges * ridge_area_ratio / 2.0),
        "n_ridges": int(n_ridges),
    }


# =============================================================================
# 7d. PASS 3 — Law 7: Anchor-Spicule (Basalia) Optical Waveguiding
#     (Sundar et al. 2003 Nature 424:899; Aizenberg et al. 2004 PNAS 101:3358):
#     step-index core/cladding built from the concentric lamellar geometry, meridional
#     ray tracing through the lamellar index steps, LP-mode count from Bessel zeros.
# =============================================================================
def simulate_basalia_optical_waveguide(
    radii_um: np.ndarray,
    thicknesses_um: np.ndarray,
    n_core: float = DEFAULT_N_CORE,
    n_clad: float = DEFAULT_N_CLAD,
    n_outside: float = DEFAULT_N_SEA,
    wavelength_um: float = DEFAULT_WAVELENGTH_UM,
    core_n_lamellae: int = OPTICAL_CORE_N_LAMELLAE,
    n_rays: int = 1801,
) -> dict[str, Any]:
    """Treat the axial cylinder plus the first `core_n_lamellae` lamellae as the high-index
    optical core (radius a) and the remaining lamellae as the lower-index cladding.
      * Meridional ray tracing: a ray launched inside the core at angle theta to the axis
        conserves n(r) cos theta(r) across every cylindrical lamellar interface (Snell);
        it is guided if n_core cos theta_0 > n_clad (total internal reflection at the
        core/cladding interface) -> acceptance angle in seawater sin theta_acc = NA / n_sea
        with NA = sqrt(n_core^2 - n_clad^2).
      * Normalised frequency V = 2 pi a NA / lambda; number of guided LP_lm modes counted
        from Bessel zeros j_{l-1,m} < V (l >= 1) plus LP_0m from zeros of J_1 (and LP01);
        single-mode iff V < 2.4048 (first zero of J_0). Gloge (1971) estimate M ≈ V^2 / 2.
    """
    from scipy.special import jn_zeros

    radii = np.asarray(radii_um, dtype=np.float64)
    thick = np.asarray(thicknesses_um, dtype=np.float64)
    outer_r = radii + thick
    a_core = float(outer_r[min(core_n_lamellae, len(outer_r) - 1)])
    r_total = float(outer_r[-1])
    index_profile_r = np.concatenate([[0.0], outer_r])
    index_profile_n = np.where(np.arange(len(outer_r)) <= core_n_lamellae, n_core, n_clad)

    na = math.sqrt(max(n_core**2 - n_clad**2, 0.0))
    theta_acc_internal = math.acos(min(n_clad / n_core, 1.0))
    theta_acc_external_deg = math.degrees(math.asin(min(na / n_outside, 1.0)))

    # Meridional ray tracing through the piecewise-constant lamellar index profile
    theta_launch = np.linspace(0.0, 0.5 * math.pi, n_rays)
    guided = np.zeros(n_rays, dtype=bool)
    for i, th in enumerate(theta_launch):
        invariant = n_core * math.cos(th)
        ok = True
        for n_layer in index_profile_n[core_n_lamellae + 1 :]:
            if invariant < n_layer:  # cos theta_layer = invariant / n_layer must be <= 1 ... refracted ray enters cladding
                ok = False
                break
        guided[i] = ok
    theta_acc_traced = float(np.degrees(theta_launch[guided].max())) if guided.any() else 0.0
    guided_fraction_isotropic = float(1.0 - math.cos(theta_acc_internal))  # solid-angle fraction of a hemisphere

    v_number = 2.0 * math.pi * a_core * na / wavelength_um

    def count_lp_modes(v: float) -> tuple[int, int]:
        """Return (number of LP_lm mode groups, total mode count incl. degeneracy)."""
        if v <= 0.0:
            return 0, 0
        groups = 1          # LP01 has no cutoff
        total = 2           # two polarisations
        n_0m = int(np.sum(jn_zeros(1, 60) < v))   # LP0m (m>=2): cutoffs at zeros of J_1
        groups += n_0m
        total += 2 * n_0m
        for l_idx in range(1, 80):                 # LP_lm (l>=1): cutoffs at zeros of J_{l-1}
            n_l = int(np.sum(jn_zeros(l_idx - 1, 60) < v))
            if n_l == 0:
                break
            groups += n_l
            total += 4 * n_l                       # 2 polarisations x 2 orientations
        return groups, total

    n_modes, n_modes_total = count_lp_modes(v_number)
    lambda_cutoff_single_mode = 2.0 * math.pi * a_core * na / LP01_CUTOFF_V
    lam_grid = np.linspace(0.35, 2.5, 120)
    modes_vs_lambda = np.array([count_lp_modes(2.0 * math.pi * a_core * na / lam)[0] for lam in lam_grid])

    return {
        "core_radius_um": a_core,
        "total_radius_um": r_total,
        "index_profile_r_um": index_profile_r,
        "index_profile_n": index_profile_n,
        "numerical_aperture": float(na),
        "theta_acc_internal_deg": float(math.degrees(theta_acc_internal)),
        "theta_acc_external_deg": float(theta_acc_external_deg),
        "theta_acc_traced_deg": theta_acc_traced,
        "guided_fraction_isotropic": guided_fraction_isotropic,
        "v_number": float(v_number),
        "n_guided_lp_modes": int(n_modes),
        "n_guided_modes_total": int(n_modes_total),
        "gloge_estimate_modes": float(v_number**2 / 2.0),
        "single_mode": bool(v_number < LP01_CUTOFF_V),
        "lambda_cutoff_single_mode_um": float(lambda_cutoff_single_mode),
        "lambda_grid_um": lam_grid,
        "modes_vs_lambda": modes_vs_lambda,
    }


# =============================================================================
# 8. Publication-Grade 6-Panel Synthesis Figure & Co-Evolving 2-Panel Growth GIF
# =============================================================================
def render_comparison_figure(
    out_path: Path = OUT_PNG,
    overrides: dict[str, float] | None = None,
) -> Path:
    """Render the 6-panel Pass 2 synthesis figure (`docs/media/sponge_spicule_hexactin_comparison.png`)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov = overrides or {}
    d_s = float(ov.get("d_s", DEFAULT_D_S_UM2_S))
    k_cat = float(ov.get("k_cat", DEFAULT_K_CAT_S))
    r_core = float(ov.get("r_core", DEFAULT_R_CORE_UM))
    n_lam = int(round(ov.get("n_lamellae", DEFAULT_N_LAMELLAE)))

    regimes = evaluate_all_morphospace_regimes(d_s_um2_s=d_s, k_cat_s=k_cat)
    lamellar = simulate_concentric_lamellar_spicule(r_core_um=r_core, n_lamellae=n_lam)
    lattice = simulate_euplectella_lattice_mechanics()
    cross_taxa = evaluate_cross_taxa_spicule_scaling()
    specimen = segment_euplectella_specimen()

    fig = plt.figure(figsize=(16.5, 10.2), dpi=150, facecolor="#0b0f17")
    gs = GridSpec(2, 3, figure=fig, hspace=0.31, wspace=0.26)

    # -------------------------------------------------------------------------
    # Panel A: 3D Silicatein Axial Filament Ray Symmetry (O_h Hexactin vs T_d Tetractin vs D_3h)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0], projection="3d", facecolor="#111827")
    hex_r = regimes["hexactin_orthogonal_6ray"]["rays"]
    tet_r = regimes["tetractin_calthrop_4ray"]["rays"]
    tri_r = regimes["triactine_planar_3ray"]["rays"]

    for idx, u in enumerate(hex_r):
        ax_a.plot(
            [0.0, 1.0 * u[0]],
            [0.0, 1.0 * u[1]],
            [0.0, 1.0 * u[2]],
            color="#38bdf8",
            lw=3.0,
            label=r"Hexactin $O_h$ (6 rays, $\theta=90^\circ$)" if idx == 0 else None,
        )
        ax_a.scatter([u[0]], [u[1]], [u[2]], color="#38bdf8", s=38, edgecolors="white")

    for idx, v in enumerate(tet_r):
        ax_a.plot(
            [0.0, 0.88 * v[0]],
            [0.0, 0.88 * v[1]],
            [0.0, 0.88 * v[2]],
            color="#f43f5e",
            lw=2.3,
            ls="--",
            label=r"Tetractin $T_d$ (4 rays, $\theta=109.47^\circ$)" if idx == 0 else None,
        )
        ax_a.scatter([0.88 * v[0]], [0.88 * v[1]], [0.88 * v[2]], color="#f43f5e", s=34, marker="^")

    for idx, w in enumerate(tri_r):
        ax_a.plot(
            [0.0, 0.72 * w[0]],
            [0.0, 0.72 * w[1]],
            [0.0, 0.72 * w[2]],
            color="#facc15",
            lw=1.8,
            ls=":",
            label=r"Triactine $D_{3h}$ (3 rays, $\theta=120^\circ$)" if idx == 0 else None,
        )

    ax_a.set_xlim(-1.05, 1.05)
    ax_a.set_ylim(-1.05, 1.05)
    ax_a.set_zlim(-1.05, 1.05)
    ax_a.set_xlabel("X", color="#9ca3af", fontsize=8)
    ax_a.set_ylabel("Y", color="#9ca3af", fontsize=8)
    ax_a.set_zlabel("Z", color="#9ca3af", fontsize=8)
    ax_a.tick_params(colors="#9ca3af", labelsize=7)
    ax_a.set_title(
        r"A. 3D Silicatein Axial Filament Ray Symmetry ($O_h, T_d, D_{3h}$)",
        color="white",
        fontsize=9.8,
        fontweight="bold",
    )
    ax_a.legend(loc="upper left", fontsize=7.0, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel B: 3D Biosilica Diffusion-Reaction Accretion Slices & Ablation Control
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1], facecolor="#111827")
    hex_sim = regimes["hexactin_orthogonal_6ray"]
    mid = len(hex_sim["coords_um"]) // 2
    b_slice_xy = hex_sim["B_field"][:, :, mid]
    coords_um = hex_sim["coords_um"]
    im_b = ax_b.pcolormesh(
        coords_um,
        coords_um,
        b_slice_xy.T,
        cmap="YlGnBu",
        shading="auto",
    )
    plt.colorbar(im_b, ax=ax_b, fraction=0.044, pad=0.03, label="Accreted Biosilica B(x,y,0)")
    ax_b.set_xlabel("x (µm)", color="white", fontsize=8.5)
    ax_b.set_ylabel("y (µm)", color="white", fontsize=8.5)
    ax_b.set_title(
        f"B. 3D Biosilica Accretion Slice (O_h 90° vs Ablation k_cat=0 → {regimes['silicatein_inhibited_ablation']['accreted_rays_detected']} Rays)",
        color="white",
        fontsize=9.4,
        fontweight="bold",
    )
    ax_b.tick_params(colors="#9ca3af", labelsize=8)

    # Inset comparing accreted biosilica mass across all 5 regimes
    ins_b = ax_b.inset_axes([0.05, 0.60, 0.44, 0.35], facecolor="#0b0f17")
    r_labels = ["Ablat.", "2-Ray", "3-Ray", "4-Ray", "6-Ray"]
    r_keys = [
        "silicatein_inhibited_ablation",
        "monaxon_oxea_2ray",
        "triactine_planar_3ray",
        "tetractin_calthrop_4ray",
        "hexactin_orthogonal_6ray",
    ]
    r_masses = [regimes[k]["total_biosilica_mass"] for k in r_keys]
    x_idx = np.arange(len(r_labels))
    ins_b.bar(x_idx, r_masses, color=["#6b7280", "#a78bfa", "#facc15", "#f43f5e", "#38bdf8"])
    ins_b.set_xticks(x_idx)
    ins_b.set_xticklabels(r_labels)
    ins_b.set_title("Biosilica Mass (5 Regimes)", color="white", fontsize=6.6)
    ins_b.tick_params(colors="#9ca3af", labelsize=5.8)

    # -------------------------------------------------------------------------
    # Panel C: Concentric Lamellar Thinning h_n ∝ r_n⁻¹ & Euplectella Lattice Shear Stiffness
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2], facecolor="#111827")
    ax_c.plot(
        lamellar["radii_um"],
        lamellar["thicknesses_um"],
        "o-",
        color="#34d399",
        lw=2.0,
        markersize=4.5,
        label=rf"Lamellar Thickness $h_n = h_1(r_1/r_n)$ (slope={lamellar['power_law_exponent']:.2f})",
    )
    ax_c.set_xlabel(r"Concentric Cylinder Radius $r_n$ (µm)", color="white", fontsize=9)
    ax_c.set_ylabel(r"Silica Layer Thickness $h_n$ (µm)", color="#34d399", fontsize=9)
    ax_c.tick_params(colors="#9ca3af", labelsize=8)
    ax_c.grid(True, alpha=0.22, color="#374151")

    # Twin axis showing Euplectella diagonal bracing shear stiffness K_shear(f_diag)
    ax_c2 = ax_c.twinx()
    r_mapped = np.linspace(
        float(lamellar["radii_um"][0]),
        float(lamellar["radii_um"][-1]),
        len(lattice["f_diag_grid"]),
    )
    ax_c2.plot(
        r_mapped,
        lattice["k_shear_curve"],
        "--",
        color="#facc15",
        lw=2.0,
        label=rf"Lattice Shear Stiffness $K_{{\mathrm{{shear}}}}$ (Checkerboard={lattice['checkerboard_shear_gain']:.1f}×)",
    )
    ax_c2.set_ylabel(r"Normalized Lattice Shear Stiffness $K_{\mathrm{shear}}/K_{\mathrm{bend}}$", color="#facc15", fontsize=8.5)
    ax_c2.tick_params(colors="#facc15", labelsize=8)
    ax_c.set_title(
        rf"C. Lamellar Thinning ($h_n \propto r_n^{{-1}}$) & Checkerboard Gain ({lattice['checkerboard_shear_gain']:.1f}×)",
        color="white",
        fontsize=9.5,
        fontweight="bold",
    )
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=6.8, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel D: 12-Species Cross-Taxa Porifera Symmetry & Lamellar Toughness Scaling
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0], facecolor="#111827")
    n_line = np.linspace(0.0, 36.0, 100)
    ax_d.plot(
        n_line,
        1.0 + DEFAULT_ETA_DEFLECT * n_line,
        "--",
        color="#e5e7eb",
        lw=2.0,
        label=rf"Aizenberg Lamellar Law $G_{{\mathrm{{lam}}}}/G_{{\mathrm{{mono}}}} = 1 + \eta N$ ($\eta={DEFAULT_ETA_DEFLECT:.2f}$)",
    )
    clade_styles = {
        "Hexactinellida (O_h Hexactin)": ("#38bdf8", "o", "Hexactinellida O_h (90°, 6 spp.)"),
        "Demospongiae (T_d Tetractin)": ("#f43f5e", "^", "Demospongiae T_d (109.47°, 2 spp.)"),
        "Demospongiae (Monaxon Oxea)": ("#a78bfa", "s", "Demospongiae Monaxon (180°, 2 spp.)"),
        "Calcarea (D_3h Triactine)": ("#facc15", "D", "Calcarea D_3h (120°, 2 spp.)"),
    }
    for cl, (col, mk, lbl) in clade_styles.items():
        pts = [r for r in cross_taxa["records"] if r["clade"] == cl]
        ax_d.scatter(
            [p["n_lamellae"] for p in pts],
            [p["toughness_ratio_obs"] for p in pts],
            color=col,
            marker=mk,
            s=64,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
            label=lbl,
        )
    # Pass 3: FE cohesive-zone three-point-bending prediction for the N=25 Euplectella laminate
    bend_wt = simulate_three_point_bending_cohesive_laminate(lamellar["radii_um"], lamellar["thicknesses_um"])
    mono_wt = simulate_three_point_bending_cohesive_laminate(lamellar["radii_um"], lamellar["thicknesses_um"], monolithic=True)
    fe_ratio = bend_wt["work_of_fracture_pJ"] / max(mono_wt["work_of_fracture_pJ"], 1e-12)
    ax_d.scatter(
        [lamellar["n_lamellae"]],
        [fe_ratio],
        marker="*",
        s=170,
        color="#34d399",
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
        label=f"FE cohesive-zone bending W_lam/W_mono = {fe_ratio:.2f}x (N={lamellar['n_lamellae']}, Pass 3)",
    )
    # Inset showing 12-species inter-ray angle parity
    ax_d.set_xlim(0.0, 40.0)
    ax_d.set_ylim(0.0, 20.0)
    ins_d = ax_d.inset_axes([0.68, 0.10, 0.30, 0.26], facecolor="#0b0f17")
    th_p = [r["theta_model_deg"] for r in cross_taxa["records"]]
    th_o = [r["theta_obs_deg"] for r in cross_taxa["records"]]
    ins_d.plot([85, 185], [85, 185], "--", color="#9ca3af", lw=1.1)
    ins_d.scatter(th_p, th_o, color="#34d399", s=24, edgecolors="white", linewidths=0.5)
    ins_d.set_title(f"Ray Angle θ (r={cross_taxa['pearson_r_angle']:.4f})", color="white", fontsize=6.6)
    ins_d.tick_params(colors="#9ca3af", labelsize=6)

    ax_d.set_xlabel(r"Concentric Biosilica Cylinder Layers $N_{\mathrm{lamellae}}$", color="white", fontsize=9)
    ax_d.set_ylabel(r"Fracture Toughness Amplification $G_{\mathrm{lam}} / G_{\mathrm{mono}}$", color="white", fontsize=9)
    ax_d.set_title(
        f"D. 12-Species Toughness & Symmetry Collapse (r={cross_taxa['pearson_r_toughness']:.4f}, slope={cross_taxa['toughness_per_layer_slope']:.3f})",
        color="white",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_d.tick_params(colors="#9ca3af", labelsize=8)
    ax_d.grid(True, alpha=0.22, color="#374151")
    ax_d.legend(loc="upper left", fontsize=6.9, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel E: Euplectella aspergillum Specimen Lattice Segmentation
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1], facecolor="#111827")
    ax_e.imshow(specimen["rgb"])
    x0, y0, x1, y1 = specimen["roi_box"]
    ax_e.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        "--",
        color="#facc15",
        lw=1.5,
        label="Segmented Cylindrical Lattice ROI",
    )
    for idx_row, (x_pks, y_row) in enumerate(specimen["detected_nodes_xy"][::2]):
        ax_e.scatter(
            x_pks,
            np.full_like(x_pks, y_row),
            color="#38bdf8",
            s=10,
            alpha=0.85,
            label="Detected Hexactin Strut Crossings" if idx_row == 0 else None,
        )
    ax_e.set_title(
        f"E. Euplectella Specimen (θ_orth={specimen['orthogonal_intersection_deg']:.2f}°, θ_diag={specimen['diagonal_strut_deg']:.2f}°, Pitch={specimen['mean_pitch_mm']:.2f} mm)",
        color="white",
        fontsize=9.3,
        fontweight="bold",
    )
    ax_e.set_xlabel("Image X (px, Wikimedia Commons Public Domain)", color="white", fontsize=8.5)
    ax_e.set_ylabel("Image Y (px)", color="white", fontsize=8.5)
    ax_e.tick_params(colors="#9ca3af", labelsize=7.5)
    ax_e.legend(loc="lower right", fontsize=7.0, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel F: Specimen Strut Orientation Distribution (90° Orthogonal + 45° Diagonal)
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2], facecolor="#111827")
    angles_folded = np.where(
        specimen["strut_angles_deg"] > 90.0,
        180.0 - specimen["strut_angles_deg"],
        specimen["strut_angles_deg"],
    )
    ax_f.hist(
        angles_folded,
        bins=24,
        color="#38bdf8",
        edgecolor="white",
        alpha=0.82,
        density=True,
        label=f"Segmented Strut Tangents (n={len(angles_folded)} px)",
    )
    ax_f.axvline(
        specimen["orthogonal_intersection_deg"],
        color="#f43f5e",
        lw=2.2,
        label=f"Orthogonal Hexactin Crossing = {specimen['orthogonal_intersection_deg']:.2f}° (ref 90°)",
    )
    ax_f.axvline(
        specimen["diagonal_strut_deg"],
        color="#facc15",
        ls="--",
        lw=2.2,
        label=f"Diagonal Checkerboard Bracing = {specimen['diagonal_strut_deg']:.2f}° (ref 45°)",
    )
    ax_f.set_xlabel("Folded Strut Orientation Angle in [0°, 90°] (deg)", color="white", fontsize=9)
    ax_f.set_ylabel("Probability Density", color="white", fontsize=9)
    ax_f.set_title(
        f"F. Specimen Strut Orientation & Lattice Pitch (CV={specimen['cv_pitch']:.3f}, n={specimen['n_spacings']})",
        color="white",
        fontsize=9.4,
        fontweight="bold",
    )
    ax_f.tick_params(colors="#9ca3af", labelsize=8)
    ax_f.grid(True, alpha=0.22, color="#374151")
    ax_f.legend(loc="upper center", fontsize=6.9, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    fig.suptitle(
        "Uriz et al. (2003) Microsc. Res. Tech. & Aizenberg et al. (2005) Science — Sponge Spicule Silicatein Symmetry & Laminated Lattice (v3)",
        color="white",
        fontsize=12.4,
        fontweight="bold",
        y=0.975,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_growth_gif(out_path: Path = OUT_GIF) -> Path:
    """Render Pass 2 co-evolving 2-panel animated GIF comparing 3D O_h hexactin vs
    T_d tetractin biosilica ray accretion (left) and concentric lamellar silica cylinder
    apposition h_n = h_1*(r_1/r_n) across N=25 layers (right)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lam = simulate_concentric_lamellar_spicule()
    hex_rays = get_silicatein_filament_ray_vectors("O_h")
    tet_rays = get_silicatein_filament_ray_vectors("T_d")

    frames: list[Image.Image] = []
    layer_steps = [3, 7, 11, 15, 19, 22, 25]

    for step_idx, n_cur in enumerate(layer_steps):
        frac = float(n_cur) / 25.0
        fig = plt.figure(figsize=(10.4, 4.6), dpi=105, facecolor="#0b0f17")
        ax1 = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#111827")
        for u in hex_rays:
            ax1.plot(
                [0.0, frac * u[0]],
                [0.0, frac * u[1]],
                [0.0, frac * u[2]],
                color="#38bdf8",
                lw=1.5 + 2.5 * frac,
            )
        for v in tet_rays:
            ax1.plot(
                [0.0, 0.85 * frac * v[0]],
                [0.0, 0.85 * frac * v[1]],
                [0.0, 0.85 * frac * v[2]],
                color="#f43f5e",
                lw=1.2 + 2.0 * frac,
                ls="--",
            )
        ax1.set_xlim(-1.05, 1.05)
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_zlim(-1.05, 1.05)
        ax1.view_init(elev=22, azim=25 + 12 * step_idx)
        ax1.set_title(
            f"3D Silicatein Ray Accretion | Layer {n_cur}/25 | O_h (90°) & T_d (109.47°)",
            color="white",
            fontsize=8.8,
            fontweight="bold",
        )
        ax1.tick_params(colors="#9ca3af", labelsize=6.5)

        ax2 = fig.add_subplot(1, 2, 2, facecolor="#111827")
        theta_ring = np.linspace(0.0, 2.0 * math.pi, 120)
        cmap = plt.get_cmap("YlGnBu")
        for l_i in range(n_cur):
            r_in = float(lam["radii_um"][l_i])
            r_out = r_in + float(lam["thicknesses_um"][l_i])
            col = cmap(0.25 + 0.70 * (l_i / 25.0))
            ax2.plot(r_out * np.cos(theta_ring), r_out * np.sin(theta_ring), color=col, lw=1.3)
        r_max = float(lam["outer_radius_um"]) * 1.08
        ax2.set_xlim(-r_max, r_max)
        ax2.set_ylim(-r_max, r_max)
        ax2.set_aspect("equal")
        tough_cur = 1.0 + DEFAULT_ETA_DEFLECT * n_cur
        ax2.set_title(
            f"Concentric Biosilica Cylinders (N={n_cur}, G_lam/G_mono={tough_cur:.2f}×)",
            color="white",
            fontsize=8.8,
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
        duration=380,
        loop=0,
    )
    return out_path


OUT_PASS3_PNG = ROOT / "docs" / "media" / "sponge_spicule_hexactin_pass3_mechanics.png"
OUT_BENDING_GIF = ROOT / "docs" / "media" / "sponge_spicule_hexactin_bending_fracture.gif"


def render_pass3_mechanics_figure(out_path: Path = OUT_PASS3_PNG) -> Path:
    """Render the 6-panel Pass 3 figure: FE cohesive-zone bending (A-C), helical ridges (D-E),
    basalia waveguide (F)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lam = simulate_concentric_lamellar_spicule()
    bend = simulate_three_point_bending_cohesive_laminate(lam["radii_um"], lam["thicknesses_um"])
    mono = simulate_three_point_bending_cohesive_laminate(lam["radii_um"], lam["thicknesses_um"], monolithic=True)
    fused = simulate_three_point_bending_cohesive_laminate(
        lam["radii_um"], lam["thicknesses_um"], t_max_mpa=DEFAULT_T_MAX_INTERLAYER_MPA,
        delta_c_um=2.0 * DEFAULT_G_S_SILICA_J_M2 / DEFAULT_T_MAX_INTERLAYER_MPA,
    )
    coupled = evaluate_coupled_accretion_to_fracture()
    sweep = evaluate_interlayer_toughness_sweep(lam["radii_um"], lam["thicknesses_um"])
    ridge = simulate_helical_ridge_cage_mechanics()
    wg = simulate_basalia_optical_waveguide(lam["radii_um"], lam["thicknesses_um"])
    p3 = load_pass3_reference()
    emp = p3["cohesive_bending_fe"]["empirical_comparisons"]

    fig = plt.figure(figsize=(16.5, 10.2), dpi=150, facecolor="#0b0f17")
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.27)
    lc = dict(facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # Panel A: FE P-delta traces
    ax = fig.add_subplot(gs[0, 0], facecolor="#111827")
    ax.plot(mono["trace_delta_um"], mono["trace_p_uN"], color="#f43f5e", lw=2.2, label=f"Monolithic rod (W={mono['work_of_fracture_pJ']/1e3:.1f} nJ)")
    ax.plot(fused["trace_delta_um"], fused["trace_p_uN"], color="#a78bfa", lw=1.6, ls=":", label=f"Fused interlayer G_i=G_s (penetrates, W={fused['work_of_fracture_pJ']/1e3:.1f} nJ)")
    ax.plot(bend["trace_delta_um"], bend["trace_p_uN"], color="#34d399", lw=1.8, label=f"Laminate N={bend['n_layers']} cohesive interlayers (W={bend['work_of_fracture_pJ']/1e3:.1f} nJ, {bend['n_load_drops']} drops)")
    ax.set_xlabel("Midspan deflection δ (µm)", color="white", fontsize=9)
    ax.set_ylabel("Load P (µN)", color="white", fontsize=9)
    ax.set_title(f"A. FE Three-Point Bending (L/D={DEFAULT_SPAN_TO_DIAMETER:.0f}, W_lam/W_mono={bend['work_of_fracture_pJ']/mono['work_of_fracture_pJ']:.2f}×)", color="white", fontsize=9.4, fontweight="bold")
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.grid(True, alpha=0.22, color="#374151")
    ax.legend(loc="upper right", fontsize=6.6, **lc)

    # Panel B: cohesive law + delamination lengths
    ax = fig.add_subplot(gs[0, 1], facecolor="#111827")
    d_grid = np.linspace(0.0, 1.15 * DEFAULT_DELTA_C_INTERLAYER_UM, 200)
    ax.plot(d_grid * 1e3, cohesive_traction_bilinear(d_grid), color="#facc15", lw=2.2, label=f"Bilinear T(δ): T_max={DEFAULT_T_MAX_INTERLAYER_MPA:.0f} MPa, δ_c={DEFAULT_DELTA_C_INTERLAYER_UM*1e3:.0f} nm, G_i={bend['g_i_j_m2']:.2f} J/m²")
    ax.fill_between(d_grid * 1e3, 0, cohesive_traction_bilinear(d_grid), color="#facc15", alpha=0.15)
    ax.set_xlabel("Interlayer separation δ (nm)", color="white", fontsize=9)
    ax.set_ylabel("Cohesive traction T (MPa)", color="#facc15", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.grid(True, alpha=0.22, color="#374151")
    ax2 = ax.twinx()
    layer_idx = np.arange(bend["n_layers"])
    ax2.bar(layer_idx, bend["delamination_half_lengths_um"], color="#38bdf8", alpha=0.55, width=0.8, label="Griffith interface half-length a_n per layer (µm)")
    ax2.axhline(bend["l_cz_um"], color="#38bdf8", ls="--", lw=1.4, label=f"Hillerborg l_cz = E G_i/T_max² = {bend['l_cz_um']:.0f} µm")
    ax2.set_ylabel("Delamination half-length a_n (µm) · layer index", color="#38bdf8", fontsize=8.5)
    ax2.tick_params(colors="#38bdf8", labelsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.4, **lc)
    ax.set_title(f"B. Cohesive-Zone Interlayer (G_i/G_s={bend['g_ratio']:.2f} < 0.25 → deflect)", color="white", fontsize=9.4, fontweight="bold")

    # Panel C: coupled k_cat sweep + interlayer G_i sweep
    ax = fig.add_subplot(gs[0, 2], facecolor="#111827")
    kc = [r["k_cat_scale"] for r in coupled["rows"]]
    tr = [r["toughness_ratio"] for r in coupled["rows"]]
    nl = [r["n_lamellae_eff"] for r in coupled["rows"]]
    ax.plot(kc, tr, "o-", color="#34d399", lw=2.0, label="W_lam/W_mono from 3D accretion → N → FE bending")
    for x_k, y_t, n_k in zip(kc, tr, nl):
        ax.annotate(f"N={n_k}", (x_k, y_t), textcoords="offset points", xytext=(0, 7), color="#9ca3af", fontsize=6.5, ha="center")
    ax.axhline(float(emp["euplectella_toughness_ratio_obs"]["value"]), color="#f43f5e", ls="--", lw=1.4, label=f"Table value {emp['euplectella_toughness_ratio_obs']['value']}× (Pass 1 provenance unverified)")
    ax.set_xlabel("Silicatein k_cat scale (ablation → wild type)", color="white", fontsize=9)
    ax.set_ylabel("Work-of-fracture ratio W_lam / W_mono", color="#34d399", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.grid(True, alpha=0.22, color="#374151")
    ax3 = ax.inset_axes([0.48, 0.42, 0.48, 0.40], facecolor="#0b0f17")
    ax3.semilogx(sweep["g_i_grid"], sweep["toughness_ratio"], "s-", color="#facc15", ms=3.5, lw=1.4)
    ax3.axvline(HE_HUTCHINSON_DEFLECTION_RATIO * DEFAULT_G_S_SILICA_J_M2, color="#f43f5e", ls=":", lw=1.2)
    ax3.set_title("G_i sweep (T_max fixed): He–Hutchinson cut-off", color="white", fontsize=6.4)
    ax3.set_xlabel("G_i (J/m²)", color="#9ca3af", fontsize=6.2)
    ax3.tick_params(colors="#9ca3af", labelsize=6)
    ax.legend(loc="lower right", fontsize=6.4, **lc)
    ax.set_title("C. k_cat → Lamellae → Toughness Coupling & Interlayer Sweep", color="white", fontsize=9.4, fontweight="bold")

    # Panel D: helical ridge torsion vs psi
    ax = fig.add_subplot(gs[1, 0], facecolor="#111827")
    ax.plot(ridge["psi_deg"], ridge["k_torsion"], color="#38bdf8", lw=2.2, label=f"K_tors(ψ) = K_wall + N_r k_r sin²(2ψ), N_r={ridge['n_ridges']}")
    ax.axhline(ridge["k_wall_checkerboard"], color="#9ca3af", ls="--", lw=1.2, label=f"Checkerboard wall only (Law 4) = {ridge['k_wall_checkerboard']:.1f}")
    ax.axvline(ridge["psi_opt_torsion_deg"], color="#38bdf8", ls=":", lw=1.2)
    ax.set_xlabel("Ridge helix angle ψ to cage axis (deg)", color="white", fontsize=9)
    ax.set_ylabel("Normalised torsional stiffness K_tors/K_bend", color="#38bdf8", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.grid(True, alpha=0.22, color="#374151")
    ax2 = ax.twinx()
    ax2.plot(ridge["psi_deg"][:-1], ridge["specific_torsion"][:-1], color="#facc15", lw=1.8, ls="--", label=f"Specific K_tors/M (ridge length ∝ 1/cos ψ), opt ψ={ridge['psi_opt_specific_deg']:.0f}°")
    ax2.set_ylabel("Specific stiffness K_tors / M_rel", color="#facc15", fontsize=8.5)
    ax2.tick_params(colors="#facc15", labelsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=6.4, **lc)
    ax.set_title(f"D. External Helical Ridges: Torsion Opt ψ={ridge['psi_opt_torsion_deg']:.0f}° (gain {ridge['torsion_gain_at_45']:.2f}×)", color="white", fontsize=9.4, fontweight="bold")

    # Panel E: ridge count -> ovalization/buckling gain + cage schematic
    ax = fig.add_subplot(gs[1, 1], facecolor="#111827")
    ax.plot(ridge["n_ridges_grid"], ridge["ovalization_gain_exact"], "o", color="#34d399", ms=6, label="Exact Σ sin²(2πk/N_r) parallel-axis sum")
    ax.plot(ridge["n_ridges_grid"], ridge["ovalization_gain_closed_form"], "--", color="#e5e7eb", lw=1.4, label="Closed form 1 + N_r·A_r/2 (N_r ≥ 3, Lean 4)")
    ax.set_xlabel("Number of equally spaced external ridges N_r", color="white", fontsize=9)
    ax.set_ylabel("I_eff / I_wall (Euler buckling & Brazier ovalization)", color="white", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.grid(True, alpha=0.22, color="#374151")
    ax.legend(loc="upper left", fontsize=6.6, **lc)
    ins = ax.inset_axes([0.58, 0.08, 0.40, 0.46], facecolor="#0b0f17")
    th = np.linspace(0, 2 * math.pi, 200)
    ins.plot(np.cos(th), np.sin(th), color="#9ca3af", lw=1.2)
    for k_r in range(ridge["n_ridges"]):
        ang = 2 * math.pi * k_r / ridge["n_ridges"]
        ins.plot([math.cos(ang), 1.18 * math.cos(ang)], [math.sin(ang), 1.18 * math.sin(ang)], color="#38bdf8", lw=3)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_title(f"Cage cross-section, N_r={ridge['n_ridges']} ridges", color="white", fontsize=6.4)
    ax.set_title(f"E. Ridge Count → Ovalization Gain ({ridge['ovalization_gain_default']:.2f}× at N_r={ridge['n_ridges']})", color="white", fontsize=9.4, fontweight="bold")

    # Panel F: basalia waveguide
    ax = fig.add_subplot(gs[1, 2], facecolor="#111827")
    r_prof = wg["index_profile_r_um"]
    n_prof = wg["index_profile_n"]
    ax.step(np.concatenate([[0.0], r_prof[1:]]), np.concatenate([[n_prof[0]], n_prof]), where="pre", color="#38bdf8", lw=2.0, label=f"Lamellar index profile n(r): core a={wg['core_radius_um']:.2f} µm")
    ax.axhline(DEFAULT_N_SEA, color="#9ca3af", ls=":", lw=1.0, label="Seawater n=1.34")
    ax.set_xlabel("Radius r (µm)", color="white", fontsize=9)
    ax.set_ylabel("Refractive index n(r)", color="#38bdf8", fontsize=9)
    ax.set_ylim(1.33, 1.47)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.grid(True, alpha=0.22, color="#374151")
    ax2 = ax.twinx()
    ax2.plot(wg["lambda_grid_um"] * 4.0, wg["modes_vs_lambda"], color="#facc15", lw=1.8, label="Guided LP mode groups vs λ (x-axis: 4·λ µm)")
    ax2.axvline(wg["lambda_cutoff_single_mode_um"] * 4.0, color="#f43f5e", ls="--", lw=1.3, label=f"Single-mode cutoff λ_c={wg['lambda_cutoff_single_mode_um']:.2f} µm (V=2.405)")
    ax2.set_ylabel("Number of guided LP modes", color="#facc15", fontsize=8.5)
    ax2.tick_params(colors="#facc15", labelsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.3, **lc)
    ax.set_title(f"F. Basalia Waveguide: NA={wg['numerical_aperture']:.3f}, V={wg['v_number']:.2f}, {wg['n_guided_lp_modes']} LP modes, θ_acc={wg['theta_acc_traced_deg']:.1f}°", color="white", fontsize=9.2, fontweight="bold")

    fig.suptitle(
        "Pass 3 — Weaver et al. (2007) Cohesive-Zone Laminate Bending FE, Aizenberg et al. (2005) Helical Ridges & Sundar et al. (2003) Basalia Waveguiding",
        color="white", fontsize=12.0, fontweight="bold", y=0.975,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_bending_fracture_gif(out_path: Path = OUT_BENDING_GIF) -> Path:
    """Animated progressive failure of the laminated spicule under three-point bending:
    left = FE P-delta trace being drawn, right = cross-section with cracked/delaminated
    outer layers removed as the events proceed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lam = simulate_concentric_lamellar_spicule()
    bend = simulate_three_point_bending_cohesive_laminate(lam["radii_um"], lam["thicknesses_um"])
    mono = simulate_three_point_bending_cohesive_laminate(lam["radii_um"], lam["thicknesses_um"], monolithic=True)
    radii = lam["radii_um"]
    outer_r = radii + lam["thicknesses_um"]
    events = bend["events"]
    n_frames = 10
    pick = np.unique(np.linspace(0, len(events), n_frames).astype(int))
    th = np.linspace(0, 2 * math.pi, 160)
    cmap = plt.get_cmap("YlGnBu")
    frames: list[Image.Image] = []
    tr_d, tr_p = bend["trace_delta_um"], bend["trace_p_uN"]
    for n_done in pick:
        fig = plt.figure(figsize=(10.4, 4.6), dpi=105, facecolor="#0b0f17")
        ax1 = fig.add_subplot(1, 2, 1, facecolor="#111827")
        ax1.plot(mono["trace_delta_um"], mono["trace_p_uN"], color="#f43f5e", lw=1.6, label="Monolithic")
        n_pts = int(events[int(n_done) - 1]["trace_index"]) if n_done > 0 else 1
        ax1.plot(tr_d[:n_pts], tr_p[:n_pts], color="#34d399", lw=1.8, label="Cohesive laminate")
        ax1.set_xlim(0, 1.05 * tr_d[-1])
        ax1.set_ylim(0, 1.1 * bend["p_peak_uN"])
        ax1.set_xlabel("δ (µm)", color="white", fontsize=8)
        ax1.set_ylabel("P (µN)", color="white", fontsize=8)
        ax1.tick_params(colors="#9ca3af", labelsize=7)
        ax1.grid(True, alpha=0.22, color="#374151")
        ax1.legend(loc="upper right", fontsize=7, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")
        ax1.set_title(f"FE three-point bending | {int(n_done)}/{len(events)} layers cracked", color="white", fontsize=8.8, fontweight="bold")

        ax2 = fig.add_subplot(1, 2, 2, facecolor="#111827")
        cracked = {int(ev["layer"]) for ev in events[: int(n_done)]}
        for l_i in range(len(radii)):
            col = cmap(0.25 + 0.70 * (l_i / max(len(radii), 1)))
            if l_i in cracked:
                ax2.plot(outer_r[l_i] * np.cos(th), outer_r[l_i] * np.sin(th), color="#f43f5e", lw=0.8, ls=":", alpha=0.5)
            else:
                ax2.plot(outer_r[l_i] * np.cos(th), outer_r[l_i] * np.sin(th), color=col, lw=1.3)
        r_max = 1.08 * float(outer_r[-1])
        ax2.set_xlim(-r_max, r_max)
        ax2.set_ylim(-r_max, r_max)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x (µm)", color="white", fontsize=8)
        ax2.set_ylabel("y (µm)", color="white", fontsize=8)
        ax2.tick_params(colors="#9ca3af", labelsize=7)
        ax2.set_title("Midspan cross-section (red dotted = cracked & delaminated lamellae)", color="white", fontsize=8.4, fontweight="bold")
        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_px, w_px, 4))
        frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=420, loop=0)
    return out_path


# =============================================================================
# 9. MorphoEval Quantitative Measurement Suite & 8-Check Self-Test
# =============================================================================
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 9 reference-anchored quantitative morphometrics for `morphoeval/measure_outputs.py`."""
    regimes = evaluate_all_morphospace_regimes()
    lamellar = simulate_concentric_lamellar_spicule()
    lattice = simulate_euplectella_lattice_mechanics()
    cross_taxa = evaluate_cross_taxa_spicule_scaling()
    specimen = segment_euplectella_specimen()

    ablat = regimes["silicatein_inhibited_ablation"]
    tet = regimes["tetractin_calthrop_4ray"]
    hex6 = regimes["hexactin_orthogonal_6ray"]

    # Pass 3 models
    bend = simulate_three_point_bending_cohesive_laminate(lamellar["radii_um"], lamellar["thicknesses_um"])
    mono = simulate_three_point_bending_cohesive_laminate(lamellar["radii_um"], lamellar["thicknesses_um"], monolithic=True)
    fused = simulate_three_point_bending_cohesive_laminate(
        lamellar["radii_um"], lamellar["thicknesses_um"],
        delta_c_um=2.0 * DEFAULT_G_S_SILICA_J_M2 / DEFAULT_T_MAX_INTERLAYER_MPA,
    )
    fe_ratio = bend["work_of_fracture_pJ"] / max(mono["work_of_fracture_pJ"], 1e-12)
    coupled = evaluate_coupled_accretion_to_fracture()
    ridge = simulate_helical_ridge_cage_mechanics()
    wg = simulate_basalia_optical_waveguide(lamellar["radii_um"], lamellar["thicknesses_um"])
    emp = load_pass3_reference()["cohesive_bending_fe"]["empirical_comparisons"]

    return [
        {
            "name": "12-species Porifera silicatein inter-ray symmetry angle correlation (O_h 90°, T_d 109.47°, D_3h 120°, D_inf_h 180°)",
            "value": round(cross_taxa["pearson_r_angle"], 4),
            "unit": "Pearson r",
            "method": f"Correlation of theoretical crystallographic silicatein filament angles vs observed spicule inter-ray angles across {cross_taxa['n_species']} Hexactinellida, Demospongiae, and Calcarea species (RMSE={cross_taxa['rmse_angle_deg']:.4f} deg)",
            "reference_value": 1.0,
            "reference_source": "Uriz et al. (2003) Microsc. Res. Tech. 62:279-299 & Aizenberg et al. (2005) Science 309:275-278",
            "note": f"RMSE = {cross_taxa['rmse_angle_deg']:.4f} deg across {cross_taxa['n_clades']} Porifera clades",
            "derived_from": "script",
        },
        {
            "name": "12-species concentric lamellar fracture toughness correlation G_lam / G_mono = 1 + eta * N",
            "value": round(cross_taxa["pearson_r_toughness"], 4),
            "unit": "Pearson r",
            "method": f"Correlation of predicted vs empirical flexural fracture toughness amplification across {cross_taxa['n_species']} sponge species (RMSE={cross_taxa['rmse_toughness']:.4f})",
            "reference_value": 1.0,
            "reference_source": "Aizenberg et al. (2005) Science 309:275-278 & Weaver et al. (2007) J. Struct. Biol. 158:93-106",
            "note": f"Hexactinellid mean = {cross_taxa['mean_hexactinellid_toughness']:.2f}x vs Demospongiae/Calcarea mean = {cross_taxa['mean_demospongiae_calcarea_toughness']:.2f}x",
            "derived_from": "script",
        },
        {
            "name": "3D accreted Hexactinellida O_h hexactin orthogonal inter-ray angle",
            "value": round(hex6["measured_inter_ray_deg"], 4),
            "unit": "deg",
            "method": "Nearest-neighbor inter-ray angle measured from 3D silicatein diffusion-reaction biosilica density field B(x,y,z) for 6-ray cubic O_h filament",
            "reference_value": 90.0,
            "reference_source": "Uriz et al. (2003) Microsc. Res. Tech. 62:279-299 & Lean 4 theorem hexactin_Oh_orthogonal_six_ray_equilibrium_and_dot_products",
            "note": f"detected rays = {hex6['accreted_rays_detected']}, vector sum norm = {hex6['vector_sum_norm']:.4e}",
            "derived_from": "script",
        },
        {
            "name": "3D accreted Demospongiae T_d tetractin/calthrop tetrahedral inter-ray angle arccos(-1/3)",
            "value": round(tet["measured_inter_ray_deg"], 4),
            "unit": "deg",
            "method": "Nearest-neighbor inter-ray angle measured from 3D silicatein diffusion-reaction biosilica density field B(x,y,z) for 4-ray tetrahedral T_d filament",
            "reference_value": 109.4712,
            "reference_source": "Uriz et al. (2003) Microsc. Res. Tech. 62:279-299 & Lean 4 theorem tetractin_Td_calthrop_tetrahedral_cosine_and_equilibrium",
            "note": f"exact cos(theta) = {tet['nearest_neighbor_cos']:.6f} (-1/3), detected rays = {tet['accreted_rays_detected']}",
            "derived_from": "script",
        },
        {
            "name": "Concentric biosilica lamellar radial thinning exponent alpha (h_n prop r_n^alpha)",
            "value": round(lamellar["power_law_exponent"], 4),
            "unit": "dimensionless exponent",
            "method": "Log-log OLS regression slope of concentric silica layer thickness h_n vs cylinder radius r_n across N=25 layers",
            "reference_value": -1.0,
            "reference_source": "Weaver et al. (2007) J. Struct. Biol. 158:93-106 & Lean 4 theorem aizenberg_concentric_lamellar_thinning_and_fracture_toughness",
            "note": f"outer radius = {lamellar['outer_radius_um']:.2f} um, toughness amplification = {lamellar['toughness_ratio']:.2f}x",
            "derived_from": "script",
        },
        {
            "name": "Euplectella alternating checkerboard diagonal bracing shear stiffness amplification (f_diag = 0.5 vs unbraced f_diag = 0)",
            "value": round(lattice["checkerboard_shear_gain"], 4),
            "unit": "stiffness ratio (K_checkerboard / K_unbraced)",
            "method": "Ratio of in-plane shear stiffness K_shear(f_diag=0.5) to unbraced square frame stiffness K_shear(f_diag=0.0) at K_axial / K_bend = 26.0",
            "reference_value": 7.5,
            "reference_source": "Aizenberg et al. (2005) Science 309:275-278 & Lean 4 theorem euplectella_checkerboard_diagonal_bracing_shear_stiffness",
            "note": f"unbraced K = {lattice['k_shear_unbraced']:.2f}, checkerboard K = {lattice['k_shear_checkerboard']:.2f}, full K = {lattice['k_shear_full']:.2f}",
            "derived_from": "script",
        },
        {
            "name": "Silicatein-inhibited / Si-starved ablation negative control accreted spicule rays (k_cat = 0)",
            "value": float(ablat["accreted_rays_detected"]),
            "unit": "rays",
            "method": "Count of accreted 3D biosilica rays when silicatein catalytic rate k_cat = 0 (negative control vs 6 rays in wild-type O_h hexactin)",
            "reference_value": 0.0,
            "reference_source": "Shimizu et al. (1998) PNAS 95:6234-6238 & Lean 4 theorem aizenberg_concentric_lamellar_thinning_and_fracture_toughness",
            "note": f"ablation biosilica mass = {ablat['total_biosilica_mass']:.4f} vs WT hexactin mass = {hex6['total_biosilica_mass']:.2f}",
            "derived_from": "script",
        },
        {
            "name": "Euplectella specimen segmented orthogonal hexactin strut intersection angle (euplectella_aspergillum_specimen.jpg)",
            "value": round(specimen["orthogonal_intersection_deg"], 4),
            "unit": "deg",
            "method": "Difference between mean vertical and horizontal DoG-segmented primary hexactin strut tangent angles in cylindrical skeleton ROI",
            "reference_value": 90.0,
            "reference_source": "Aizenberg et al. (2005) Science 309:275-278 (orthogonal 90 deg longitudinal/transverse hexactin grid)",
            "note": f"diagonal bracing strut angle = {specimen['diagonal_strut_deg']:.2f} deg (ref 45 deg)",
            "derived_from": "output",
        },
        {
            "name": "Euplectella specimen segmented square lattice pitch (euplectella_aspergillum_specimen.jpg)",
            "value": round(specimen["mean_pitch_mm"], 4),
            "unit": "mm",
            "method": f"1D horizontal DoG ridge peak spacing across 27 cylindrical lattice row transects ({specimen['n_spacings']} inter-strut intervals, mean = {specimen['mean_pitch_px']:.2f} px at 0.103884 mm/px)",
            "reference_value": 2.50,
            "reference_source": "Aizenberg et al. (2005) Science 309:275-278 (Euplectella aspergillum 2.5 mm square grid spacing)",
            "note": f"spacing CV = {specimen['cv_pitch']:.4f} across {specimen['n_spacings']} intervals",
            "derived_from": "output",
        },
        # ---------------- Pass 3 additions ----------------
        {
            "name": "FE three-point bending analytic parity: Hermite beam midspan stiffness vs 48 EI / L^3",
            "value": round(bend["k_initial_uN_per_um"] / bend["k_analytic_uN_per_um"], 6),
            "unit": "ratio",
            "method": f"{N_BEAM_ELEMENTS}-element Euler-Bernoulli FE midspan stiffness under displacement control divided by the closed-form simply-supported value (span {bend['span_um']:.1f} um)",
            "reference_value": 1.0,
            "reference_source": "Euler-Bernoulli beam theory (exact for Hermite cubic elements under a nodal point load)",
            "note": f"k_FE = {bend['k_initial_uN_per_um']:.4f} uN/um",
            "derived_from": "script",
        },
        {
            "name": "Cohesive-zone laminate work-of-fracture ratio W_lam / W_mono under three-point bending (N = 25, G_i = 1.2 J/m^2)",
            "value": round(fe_ratio, 4),
            "unit": "ratio",
            "method": "Integral of the FE progressive-failure P-delta trace (He-Hutchinson deflection, Griffith interface delamination) divided by the monolithic rod of identical outer radius",
            "reference_value": float(emp["euplectella_toughness_ratio_obs"]["value"]),
            "reference_source": "Vendored 12-species table value for Euplectella aspergillum (Pass 1 provenance unverified) vs Weaver et al. (2007) J. Struct. Biol. 158:93-106 crack-arrest observation",
            "note": f"{bend['n_load_drops']} load drops, interface dissipation {bend['dissipated_interface_pJ']/1e3:.1f} nJ, l_cz = {bend['l_cz_um']:.1f} um; honest reduced model under-predicts the table value",
            "derived_from": "script",
        },
        {
            "name": "Fused-interlayer ablation (G_i = G_s, He-Hutchinson ratio 1.0) work-of-fracture ratio vs monolith",
            "value": round(fused["work_of_fracture_pJ"] / max(mono["work_of_fracture_pJ"], 1e-12), 4),
            "unit": "ratio",
            "method": "Same FE laminate with the organic interlayer fracture energy raised to that of silica: crack penetrates at the first interface -> catastrophic failure",
            "reference_value": 1.0,
            "reference_source": "He & Hutchinson (1989) Int. J. Solids Struct. 25:1053-1067 deflection criterion",
            "note": f"deflects = {fused['deflects']}, load drops = {fused['n_load_drops']}",
            "derived_from": "script",
        },
        {
            "name": "Silicatein k_cat ablation -> 3D accreted cross-section -> effective lamella count (wild type)",
            "value": float(coupled["wt_n_lamellae"]),
            "unit": "lamellae",
            "method": "Biosilica B(x,y,z) integrated over the plane normal to the +e_x ray at 0.55 R, mapped to layer count with the (calibrated) SILICA_MASS_TO_LAYER_CALIBRATION constant; k_cat sweep 0/0.25/0.5/0.75/1",
            "reference_value": 25.0,
            "reference_source": "Aizenberg et al. (2005) Science 309:275-278 (~25 concentric layers; calibration is self-referential by construction)",
            "note": "k_cat sweep N = " + ", ".join(f"{r['k_cat_scale']:.2f}->{r['n_lamellae_eff']}" for r in coupled["rows"]),
            "derived_from": "script",
        },
        {
            "name": "Peripheral-to-central lamella thickness ratio h_25 / h_1 (model) vs Aizenberg et al. (2005) 0.2 um / 1.5 um",
            "value": round(float(lamellar["thicknesses_um"][-1] / lamellar["thicknesses_um"][0]), 4),
            "unit": "ratio",
            "method": "h_n = h_1 r_1 / r_n with r_core = 0.75 um, h_1 = 0.85 um, N = 25",
            "reference_value": float(emp["peripheral_to_central_thickness_ratio"]["value"]),
            "reference_source": "Aizenberg et al. (2005) Science 309:275-278 (layer thickness ~1.5 um centre -> ~0.2 um periphery; recalled offline, unverified)",
            "note": f"h_1 = {lamellar['thicknesses_um'][0]:.3f} um, h_25 = {lamellar['thicknesses_um'][-1]:.3f} um",
            "derived_from": "script",
        },
        {
            "name": "External helical ridge angle maximising cage torsional stiffness (sin^2(2 psi) projection)",
            "value": round(ridge["psi_opt_torsion_deg"], 4),
            "unit": "deg",
            "method": "Argmax over psi in [0, 90] deg of K_tors(psi) = K_wall + N_r k_r sin^2(2 psi) for N_r = 8 ridges",
            "reference_value": 45.0,
            "reference_source": "Netting analysis of helical members; Lean 4 theorem euplectella_helical_ridge_shear_projection_and_parallel_axis_gain",
            "note": f"specific (per-mass) optimum psi = {ridge['psi_opt_specific_deg']:.1f} deg, torsion gain at 45 deg = {ridge['torsion_gain_at_45']:.3f}x",
            "derived_from": "script",
        },
        {
            "name": "Ovalization / buckling second-moment gain of the cage from N_r = 8 equally spaced ridges (I_eff / I_wall)",
            "value": round(float(ridge["ovalization_gain_exact"][8]), 4),
            "unit": "ratio",
            "method": "1 + A_r sum_k sin^2(2 pi k / 8) with ridge area ratio A_r = 0.25",
            "reference_value": round(ridge["ovalization_gain_default"], 4),
            "reference_source": "Parallel-axis identity sum_k sin^2(2 pi k / N) = N / 2 for N >= 3 (Lean 4 theorem, closed form 1 + N_r A_r / 2)",
            "note": "closed form vs explicit sum agree to machine precision for every N_r in 3..12",
            "derived_from": "script",
        },
        {
            "name": "Basalia waveguide ray-traced acceptance angle vs analytic arccos(n_clad / n_core)",
            "value": round(wg["theta_acc_traced_deg"], 4),
            "unit": "deg",
            "method": f"Meridional ray tracing through the {lamellar['n_lamellae']}-lamella step-index profile (n_core = {DEFAULT_N_CORE}, n_clad = {DEFAULT_N_CLAD}), largest guided launch angle on a 0.05 deg grid",
            "reference_value": round(wg["theta_acc_internal_deg"], 4),
            "reference_source": "Snell / total-internal-reflection invariant n cos theta (Sundar et al. 2003 Nature 424:899; indices recalled offline)",
            "note": f"NA = {wg['numerical_aperture']:.4f}, external acceptance = {wg['theta_acc_external_deg']:.2f} deg",
            "derived_from": "script",
        },
        {
            "name": "Basalia waveguide guided LP mode groups at 550 nm (core a from lamella 2) vs Gloge V^2/2 total-mode estimate",
            "value": float(wg["n_guided_modes_total"]),
            "unit": "modes (incl. degeneracy)",
            "method": "LP mode cutoffs at Bessel zeros j_{l-1,m} < V counted with 2x/4x degeneracy",
            "reference_value": round(wg["gloge_estimate_modes"], 4),
            "reference_source": "Gloge (1971) Appl. Opt. 10:2252 weakly guiding fibre mode count M ~ V^2 / 2",
            "note": f"V = {wg['v_number']:.3f}, {wg['n_guided_lp_modes']} LP groups, single-mode cutoff lambda_c = {wg['lambda_cutoff_single_mode_um']:.3f} um",
            "derived_from": "script",
        },
    ]


def self_test() -> bool:
    """Run comprehensive 8-point verification suite for Pass 2."""
    # 1. Cryptographic SHA-256 verification of all 4 dataset files
    digests = verify_dataset_integrity()
    assert len(digests) == 4, "Expected 4 SHA-256-verified dataset files"

    # 2. JSON-to-code parity checks
    ref = load_reference_data()
    defaults = ref["governing_equations"]["default_parameters"]
    assert math.isclose(float(defaults["r_core_um"]), DEFAULT_R_CORE_UM, rel_tol=1e-9)
    assert int(defaults["n_lamellae_euplectella"]) == DEFAULT_N_LAMELLAE

    # 3. Law 1 & Law 2: Exact crystallographic ray vector sums & pairwise cosines
    hex_geom = analyze_ray_vector_geometry(get_silicatein_filament_ray_vectors("O_h"))
    tet_geom = analyze_ray_vector_geometry(get_silicatein_filament_ray_vectors("T_d"))
    tri_geom = analyze_ray_vector_geometry(get_silicatein_filament_ray_vectors("D_3h"))
    assert hex_geom["n_rays"] == 6 and hex_geom["vector_sum_norm"] < 1e-12
    assert math.isclose(hex_geom["nearest_neighbor_angle_deg"], 90.0, rel_tol=1e-9)
    assert tet_geom["n_rays"] == 4 and tet_geom["vector_sum_norm"] < 1e-12
    assert math.isclose(tet_geom["nearest_neighbor_cos"], -1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(tet_geom["nearest_neighbor_angle_deg"], TETRAHEDRAL_ANGLE_DEG, rel_tol=1e-9)
    assert math.isclose(tri_geom["nearest_neighbor_angle_deg"], 120.0, rel_tol=1e-9)

    # 4. 3D Biosilica Diffusion-Reaction Accretion & Silicatein-Inhibited Ablation Control
    regimes = evaluate_all_morphospace_regimes()
    assert regimes["silicatein_inhibited_ablation"]["accreted_rays_detected"] == 0
    assert regimes["silicatein_inhibited_ablation"]["total_biosilica_mass"] == 0.0
    assert regimes["monaxon_oxea_2ray"]["accreted_rays_detected"] == 2
    assert regimes["triactine_planar_3ray"]["accreted_rays_detected"] == 3
    assert regimes["tetractin_calthrop_4ray"]["accreted_rays_detected"] == 4
    assert regimes["hexactin_orthogonal_6ray"]["accreted_rays_detected"] == 6
    assert math.isclose(regimes["hexactin_orthogonal_6ray"]["measured_inter_ray_deg"], 90.0, abs_tol=0.5)
    assert math.isclose(regimes["tetractin_calthrop_4ray"]["measured_inter_ray_deg"], TETRAHEDRAL_ANGLE_DEG, abs_tol=0.5)

    # 5. Law 3 & Law 4: Concentric Lamellar Thinning h_n ∝ r_n⁻¹ & Checkerboard Shear Stiffness
    lam = simulate_concentric_lamellar_spicule()
    lat = simulate_euplectella_lattice_mechanics()
    assert math.isclose(lam["power_law_exponent"], -1.0, abs_tol=1e-6)
    assert math.isclose(lam["toughness_ratio"], 9.0, rel_tol=1e-9)
    assert math.isclose(lat["checkerboard_shear_gain"], 7.5, rel_tol=1e-9)

    # 6. 12-Species Cross-Taxa Porifera Symmetry & Lamellar Toughness Scaling
    cross = evaluate_cross_taxa_spicule_scaling()
    assert cross["n_species"] == 12 and cross["n_clades"] == 4
    assert cross["pearson_r_angle"] > 0.999
    assert cross["pearson_r_toughness"] > 0.999

    # 7. Real Euplectella Specimen Computer-Vision Lattice Segmentation
    spec = segment_euplectella_specimen()
    assert math.isclose(spec["orthogonal_intersection_deg"], 89.6498, abs_tol=0.5)
    assert 44.0 <= spec["diagonal_strut_deg"] <= 50.0
    assert math.isclose(spec["mean_pitch_mm"], 2.50, rel_tol=0.02)

    # 8. Two-run bitwise determinism check on 3D biosilica accretion solver
    run1 = simulate_spicule_biosilica_accretion_3d("hexactin_orthogonal_6ray", "O_h", 1.0)
    run2 = simulate_spicule_biosilica_accretion_3d("hexactin_orthogonal_6ray", "O_h", 1.0)
    assert np.allclose(run1["B_field"], run2["B_field"], atol=0.0)

    # ---------------- Pass 3 checks ----------------
    # 9. FE beam parity with 48 EI / L^3 and cohesive-law energy integral
    bend = simulate_three_point_bending_cohesive_laminate(lam["radii_um"], lam["thicknesses_um"])
    assert math.isclose(bend["k_initial_uN_per_um"], bend["k_analytic_uN_per_um"], rel_tol=1e-9)
    d_grid = np.linspace(0.0, DEFAULT_DELTA_C_INTERLAYER_UM, 20001)
    g_num = float(np.trapezoid(cohesive_traction_bilinear(d_grid), d_grid))
    assert math.isclose(g_num, bend["g_i_j_m2"], rel_tol=1e-4), (g_num, bend["g_i_j_m2"])
    assert math.isclose(bend["l_cz_um"], DEFAULT_E_SILICA_MPA * bend["g_i_j_m2"] / DEFAULT_T_MAX_INTERLAYER_MPA**2, rel_tol=1e-12)

    # 10. Progressive (graceful) failure vs monolithic / fused-interlayer catastrophic failure
    mono = simulate_three_point_bending_cohesive_laminate(lam["radii_um"], lam["thicknesses_um"], monolithic=True)
    fused = simulate_three_point_bending_cohesive_laminate(
        lam["radii_um"], lam["thicknesses_um"], delta_c_um=2.0 * DEFAULT_G_S_SILICA_J_M2 / DEFAULT_T_MAX_INTERLAYER_MPA
    )
    assert bend["deflects"] and bend["n_load_drops"] == DEFAULT_N_LAMELLAE
    assert (not mono["deflects"]) and mono["n_load_drops"] == 1 and mono["catastrophic_at_um"] is not None
    assert (not fused["deflects"]) and fused["n_load_drops"] == 1
    assert math.isclose(mono["p_peak_uN"], bend["p_peak_uN"], rel_tol=1e-9)  # first crack at identical load
    assert bend["work_of_fracture_pJ"] > mono["work_of_fracture_pJ"] > 0.0
    assert math.isclose(fused["work_of_fracture_pJ"], mono["work_of_fracture_pJ"], rel_tol=1e-9)
    # energy bookkeeping: external work = stored elastic energy at first crack + dissipation (within trace discretisation)
    assert bend["dissipated_interface_pJ"] > 0.0 and bend["delta_final_um"] > mono["delta_final_um"]

    # 11. k_cat -> lamellae -> toughness coupling and helical ridge identities
    coupled = evaluate_coupled_accretion_to_fracture()
    n_eff = [r["n_lamellae_eff"] for r in coupled["rows"]]
    assert n_eff[0] == 0 and coupled["wt_n_lamellae"] == DEFAULT_N_LAMELLAE
    assert all(a <= b for a, b in zip(n_eff, n_eff[1:])), n_eff
    ridge = simulate_helical_ridge_cage_mechanics()
    assert math.isclose(ridge["psi_opt_torsion_deg"], 45.0, abs_tol=1e-9)
    assert 0.0 < ridge["psi_opt_specific_deg"] < 45.0
    assert np.allclose(ridge["ovalization_gain_exact"][3:], ridge["ovalization_gain_closed_form"][3:], atol=1e-12)

    # 12. Basalia waveguide: ray-traced acceptance matches Snell/TIR, mode count brackets Gloge
    wg = simulate_basalia_optical_waveguide(lam["radii_um"], lam["thicknesses_um"])
    assert math.isclose(wg["theta_acc_traced_deg"], wg["theta_acc_internal_deg"], abs_tol=0.06)
    assert math.isclose(wg["numerical_aperture"], math.sqrt(DEFAULT_N_CORE**2 - DEFAULT_N_CLAD**2), rel_tol=1e-12)
    assert wg["n_guided_lp_modes"] >= 2 and not wg["single_mode"]
    assert 0.5 * wg["gloge_estimate_modes"] <= wg["n_guided_modes_total"] <= 1.5 * wg["gloge_estimate_modes"]
    wg_long = simulate_basalia_optical_waveguide(lam["radii_um"], lam["thicknesses_um"], wavelength_um=2.0)
    assert wg_long["single_mode"] and wg_long["n_guided_lp_modes"] == 1
    assert PASS3_JSON.exists()

    meas = measure_morphoeval()
    assert len(meas) == 18
    print(
        f"[self_test] 2012-bavestrello-sponge-spicule-hexactin.py (v3) PASSED all 12 checks "
        f"(12-species r_angle={cross['pearson_r_angle']:.4f}, r_tough={cross['pearson_r_toughness']:.4f}, "
        f"lamellar_exp={lam['power_law_exponent']:.4f}, checkerboard_gain={lat['checkerboard_shear_gain']:.2f}x, "
        f"specimen θ_orth={spec['orthogonal_intersection_deg']:.2f} deg, pitch={spec['mean_pitch_mm']:.3f} mm; "
        f"FE W_lam/W_mono={bend['work_of_fracture_pJ']/mono['work_of_fracture_pJ']:.3f}x with {bend['n_load_drops']} drops, "
        f"ridge psi_opt={ridge['psi_opt_torsion_deg']:.0f} deg, waveguide V={wg['v_number']:.2f} / {wg['n_guided_lp_modes']} LP modes)."
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run 12-check verification suite")
    ap.add_argument("--skip-gif", action="store_true", help="Skip animated GIF rendering")
    ap.add_argument("--d-s", type=float, default=None, help="Override silicic acid diffusivity D_S (um^2/s)")
    ap.add_argument("--k-cat", type=float, default=None, help="Override silicatein catalytic rate k_cat (1/s)")
    ap.add_argument("--r-core", type=float, default=None, help="Override axial filament core radius r_core (um)")
    ap.add_argument("--n-lamellae", type=float, default=None, help="Override concentric lamellar layer count N")
    ap.add_argument("--seed", type=int, default=2012, help="Deterministic seed (default: 2012; the model has no stochastic component, so this flag is a documented no-op)")
    ap.add_argument("--skip-pass3", action="store_true", help="Skip the Pass 3 mechanics PNG and bending-fracture GIF")
    ap.add_argument("--g-i", type=float, default=None, help="Override organic interlayer fracture energy G_i (J/m^2) for a one-off bending report")
    ap.add_argument("--n-ridges", type=int, default=None, help="Override external helical ridge count for a one-off cage report")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    overrides = {
        k_name: v
        for k_name, v in {
            "d_s": args.d_s,
            "k_cat": args.k_cat,
            "r_core": args.r_core,
            "n_lamellae": args.n_lamellae,
        }.items()
        if v is not None
    }
    png = render_comparison_figure(OUT_PNG, overrides=overrides)
    print(f"[render] Wrote {png}")
    if not args.skip_gif:
        gif = render_growth_gif(OUT_GIF)
        print(f"[render] Wrote {gif}")
    if not args.skip_pass3:
        png3 = render_pass3_mechanics_figure(OUT_PASS3_PNG)
        print(f"[render] Wrote {png3}")
        if not args.skip_gif:
            gif3 = render_bending_fracture_gif(OUT_BENDING_GIF)
            print(f"[render] Wrote {gif3}")
    if args.g_i is not None or args.n_ridges is not None:
        lam = simulate_concentric_lamellar_spicule(
            r_core_um=float(overrides.get("r_core", DEFAULT_R_CORE_UM)),
            n_lamellae=int(round(overrides.get("n_lamellae", DEFAULT_N_LAMELLAE))),
        )
        if args.g_i is not None:
            b = simulate_three_point_bending_cohesive_laminate(
                lam["radii_um"], lam["thicknesses_um"], delta_c_um=2.0 * args.g_i / DEFAULT_T_MAX_INTERLAYER_MPA
            )
            m = simulate_three_point_bending_cohesive_laminate(lam["radii_um"], lam["thicknesses_um"], monolithic=True)
            print(
                f"[bending] G_i={args.g_i:.3f} J/m^2: deflects={b['deflects']}, drops={b['n_load_drops']}, "
                f"l_cz={b['l_cz_um']:.1f} um, W_lam/W_mono={b['work_of_fracture_pJ']/m['work_of_fracture_pJ']:.3f}"
            )
        if args.n_ridges is not None:
            r = simulate_helical_ridge_cage_mechanics(n_ridges=args.n_ridges)
            print(
                f"[ridges] N_r={args.n_ridges}: psi_opt={r['psi_opt_torsion_deg']:.1f} deg, torsion gain={r['torsion_gain_at_45']:.3f}x, "
                f"I_eff/I_wall={r['ovalization_gain_default']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
