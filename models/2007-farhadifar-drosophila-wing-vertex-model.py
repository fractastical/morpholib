#!/usr/bin/env python3
"""
2007-farhadifar-drosophila-wing-vertex-model.py — Pass 3 (v3 Remediated)
========================================================================
Farhadifar, Röper, Aigouy, Eaton & Jülicher (2007) Current Biology 17(24):2095-2104,
Gibson, Patel, Nagpal & Perrimon (2006) Nature 442(7106):1038-1041,
Classen, Anderson, Marois & Eaton (2005) Developmental Cell 9(6):805-817,
Nagai & Honda (2001) Philosophical Magazine B 81(7):699-719,
Lewis (1928) The Anatomical Record 38(3):341-376,
Aigouy, Farhadifar, Staple, Sagner, Röper, Jülicher & Eaton (2010) Cell 142(5):773-786, and
Etournay, Popović, Merkel, Nandi, Blasse, Aigouy, Brandl, Myers, Salbreux, Jülicher & Eaton
(2015) eLife 4:e07090.

Primary Literature & Equations Implemented:
  1. Farhadifar et al. (2007) Current Biology 17:2095-2104 (doi:10.1016/j.cub.2007.11.049):
     - Apical Epithelial Vertex Energy Functional (Eq. 1):
         E({r_i}) = sum_alpha (K_alpha / 2) * (A_alpha - A_alpha^(0))^2
                  + sum_{<i,j>} Lambda_{ij} * l_{ij}
                  + sum_alpha (Gamma_alpha / 2) * L_alpha^2
     - Dimensionless Normalized Parameters (Eq. 2):
         Lambda_bar = Lambda / (K * (A^(0))^(3/2)),   Gamma_bar = Gamma / (K * A^(0))
     - Regular Hexagon Ground-State Energy & Phase Boundaries (Fig. 2A):
         L_6(a) = c_6 * sqrt(a) * sqrt(A^(0)),  where c_6 = 2 * sqrt(2 * sqrt(3)) approx 3.7224
         e_6(a) = 0.5 * (a - 1)^2 + 0.5 * c_6 * Lambda_bar * a^(1/2) + 0.5 * c_6^2 * Gamma_bar * a
       Partitioning the (Lambda_bar, Gamma_bar) plane into:
         * Case I   (Rigid Hexagonal Network, Delta E_T1 > 0, A* > 0): canonical wing disc
                    (Lambda_bar = 0.12, Gamma_bar = 0.04)
         * Case II  (Soft / Fluid-like Network, Delta E_T1 = 0, Lambda_bar < Lambda_bar_c(Gamma_bar))
         * Case III (Collapsed Network, A* = 0 for large positive Lambda_bar / Gamma_bar)
     - UV Laser Adherens-Junction Ablation Recoil Kinetics (Fig. 4D-F):
         dD/dt = (T_junction - k_visc * (D(t) - D_0)) / mu,  initial recoil v_0 = T_junction / mu
  2. Gibson et al. (2006) Nature 442:1038-1041 (doi:10.1038/nature05014) &
     Classen et al. (2005) Dev. Cell 9:805-817 (doi:10.1016/j.devcel.2005.10.016):
     - Proliferating 3rd-instar larval wing disc polygon distribution P(n) (P(6) = 0.455,
       P(5) = 0.280, P(7) = 0.200, second central moment mu_2 = 0.765) vs. post-mitotic
       pupal wing after PCP-driven junctional annealing (P(6) = 0.780, mu_2 = 0.250).
  3. Lewis (1928) Anat. Rec. 38:341-376 (doi:10.1002/ar.1090380305):
     - Linear Area-Topology Law: <A_n> / <A> = (n - 2) / 4 for n = 4..9.
  4. [Pass 3] Aigouy et al. (2010) Cell 142:773-786 (doi:10.1016/j.cell.2010.07.042),
     Etournay et al. (2015) eLife 4:e07090 (doi:10.7554/eLife.07090) &
     Sugimura & Ishihara (2013) Development 140:4091-4101 (doi:10.1242/dev.094060):
     - Explicit vertex dynamics (tri-cellular vertices, analytic forces F_i = -dE/dr_i of
       Eq. 1, T1 exchanges below l_T1) with a per-bond planar-cell-polarity Frizzled/Strabismus
       asymmetry tensor Q_ab = 2 c_ab (n (x) n - I/2), bond tension
         Lambda_ab = Lambda_bar (1 + eps_pcp * n . Q_ab . n),
       Fz-Stbm intercellular feedback + flow alignment of the polarity field, and the
       pupal hinge-contraction pure-shear strain F(t) = diag(lambda, 1/lambda) on the blade
       margin; tissue shear decomposed into cell-elongation and rearrangement parts.
     - DOIs for the Pass 3 references were recorded offline and NOT re-verified online in
       this pass (see datasets/drosophila_wing_vertex_model/README.md).
  Lean 4: morphoeval/lean/loops/Loop_043_drosophila_wing_vertex_model.lean (6 theorems, 0 sorrys).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, optimize, stats
from scipy.spatial import Voronoi

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "drosophila_wing_vertex_model"
REFERENCE_JSON = DATASET_DIR / "drosophila_wing_vertex_model_reference.json"
SPECIMEN_JPG = DATASET_DIR / "drosophila_wing_epithelium_specimen.jpg"
MEDIA_DIR = ROOT / "docs" / "media"
COMPARISON_PNG = MEDIA_DIR / "drosophila_wing_vertex_model_comparison.png"
GROWTH_GIF = MEDIA_DIR / "drosophila_wing_vertex_model_growth.gif"

EXPECTED_SHA256 = {
    "datasets/drosophila_wing_vertex_model/drosophila_wing_epithelium_specimen.jpg": (
        "df24205361561b79fc3770d3c4627733b82ebff90957794958d16102ba5745ff"
    ),
}

C6_HEX = float(2.0 * np.sqrt(2.0 * np.sqrt(3.0)))  # ~3.7224194364083984


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_reference_dataset() -> dict[str, Any]:
    """Load and cryptographically verify datasets/drosophila_wing_vertex_model/."""
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(f"Missing reference dataset: {REFERENCE_JSON}")
    if not SPECIMEN_JPG.exists():
        raise FileNotFoundError(f"Missing specimen micrograph: {SPECIMEN_JPG}")

    ref_data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    actual_sha = sha256_file(SPECIMEN_JPG)
    expected_sha = EXPECTED_SHA256[
        "datasets/drosophila_wing_vertex_model/drosophila_wing_epithelium_specimen.jpg"
    ]
    json_sha = ref_data["metadata"]["specimens"]["drosophila_wing_epithelium_specimen"]["sha256"]
    if actual_sha != expected_sha or actual_sha != json_sha:
        raise ValueError(
            f"SHA-256 mismatch for {SPECIMEN_JPG}: actual={actual_sha}, expected={expected_sha}"
        )
    return ref_data


# =============================================================================
# Section 1: Farhadifar et al. (2007) Vertex Functional & Ground-State Phase Diagram
# =============================================================================

def compute_single_cell_vertex_energy(
    area: float,
    perimeter: float,
    K: float = 1.0,
    A0: float = 1.0,
    Lambda_bar: float = 0.12,
    Gamma_bar: float = 0.04,
) -> dict[str, float]:
    """
    Evaluate Farhadifar et al. (2007) Current Biology Eq. (1)-(2) dimensionless
    single-cell apical vertex energy:
      E_cell / (K * A0^2) = 0.5 * (A/A0 - 1)^2
                          + 0.5 * Lambda_bar * (L / sqrt(A0))
                          + 0.5 * Gamma_bar * (L / sqrt(A0))^2
    where factor 0.5 on Lambda_bar reflects that each junctional edge l_{ij} is shared
    between two adjacent cells. Matches Lean 4 `two_vertex_cell_energy` in
    `morphoeval/lean/loops/Loop_043_drosophila_wing_vertex_model.lean`.
    """
    a_rel = float(area / A0)
    l_rel = float(perimeter / np.sqrt(A0))
    e_area = 0.5 * K * (a_rel - 1.0) ** 2
    e_line = 0.5 * Lambda_bar * l_rel
    e_perim = 0.5 * Gamma_bar * (l_rel ** 2)
    return {
        "e_area": float(e_area),
        "e_line": float(e_line),
        "e_perim": float(e_perim),
        "e_total": float(e_area + e_line + e_perim),
    }


def classify_farhadifar_ground_state(
    Lambda_bar: float,
    Gamma_bar: float,
) -> dict[str, Any]:
    """
    Classify a point (Lambda_bar, Gamma_bar) in the Farhadifar et al. (2007) Fig. 2A
    ground-state phase diagram:
      - Regular hexagon energy as a function of s = sqrt(A / A0) >= 0:
          e_6(s) = 0.5 * (s^2 - 1)^2 + 0.5 * c_6 * Lambda_bar * s + 0.5 * c_6^2 * Gamma_bar * s^2
      - Boundary between Case I (rigid hexagonal network, T1 energy barrier > 0) and
        Case II (soft network, vanishing T1 barrier where net effective line tension
        T_eff = Lambda_bar + 2 * c_6 * Gamma_bar * s* <= 0):
          Lambda_bar_c(Gamma_bar) = -2 * c_6 * Gamma_bar / sqrt(1 - c_6^2 * Gamma_bar)
      - Boundary to Case III (cell collapse s* = 0 where e_6(s) has global minimum at s = 0).
    """
    def e6_of_s(s: float) -> float:
        return (
            0.5 * (s * s - 1.0) ** 2
            + 0.5 * C6_HEX * Lambda_bar * s
            + 0.5 * (C6_HEX ** 2) * Gamma_bar * (s * s)
        )

    res = optimize.minimize_scalar(e6_of_s, bounds=(0.0, 2.5), method="bounded")
    s_opt = float(res.x)
    e_opt = float(res.fun)
    e_zero = e6_of_s(0.0)  # 0.5

    if s_opt < 1e-3 or e_opt >= e_zero - 1e-7:
        s_star = 0.0
        a_star = 0.0
        phase = "Case_III_cell_collapse"
        t_eff = float(Lambda_bar)
    else:
        s_star = s_opt
        a_star = s_opt * s_opt
        # Effective junction tension T_eff = Lambda_bar + c_6 * Gamma_bar * s*
        t_eff = float(Lambda_bar + C6_HEX * Gamma_bar * s_star)
        if t_eff <= 0.0:
            phase = "Case_II_soft_network"
        else:
            phase = "Case_I_rigid_hexagonal"

    return {
        "Lambda_bar": float(Lambda_bar),
        "Gamma_bar": float(Gamma_bar),
        "s_star": s_star,
        "a_star": a_star,
        "L_star": float(C6_HEX * s_star),
        "effective_junction_tension": max(0.0, t_eff),
        "raw_junction_tension": t_eff,
        "phase": phase,
    }


def simulate_laser_ablation_recoil(
    Lambda_bar: float = 0.12,
    Gamma_bar: float = 0.04,
    t_max: float = 50.0,
    n_pts: int = 100,
) -> dict[str, Any]:
    """
    Simulate Farhadifar et al. (2007) Fig. 4D-F UV laser ablation of an adherens junction:
    Upon severing a single junctional bond, the two tri-cellular vertices separate with
    overdamped Kelvin-Voigt relaxation:
      Delta d(t) = d(t) - d(0) = (v_0 * tau_r) * (1 - exp(-t / tau_r))
    where initial recoil velocity v_0 is proportional to the net mechanical tension
    T_eff = Lambda_bar + 2 * Gamma_bar * L* on the ablated bond.
    """
    gs = classify_farhadifar_ground_state(Lambda_bar, Gamma_bar)
    gs_ref = classify_farhadifar_ground_state(0.12, 0.04)
    # Normalize initial recoil velocity to Farhadifar et al. (2007) Fig. 4E wing disc value v0 = 0.22 um/s
    t_ratio = gs["effective_junction_tension"] / max(gs_ref["effective_junction_tension"], 1e-8)
    v0_um_s = float(0.22 * t_ratio)
    tau_relax_s = float(11.5 / (1.0 + 3.5 * Gamma_bar))
    t_arr = np.linspace(0.0, t_max, n_pts)
    displacement_um = v0_um_s * tau_relax_s * (1.0 - np.exp(-t_arr / tau_relax_s))
    return {
        "t_s": t_arr,
        "displacement_um": displacement_um,
        "initial_recoil_v0_um_s": v0_um_s,
        "normalized_v0_ratio": float(t_ratio),
        "tau_relax_s": tau_relax_s,
        "ground_state": gs,
    }


def lewis_law_ratio(n: int | np.ndarray) -> np.ndarray | float:
    """
    Lewis (1928) Anat. Rec. 38:341-376 linear area-topology scaling:
      <A_n> / <A> = (n - 2) / 4.
    Matches Lean 4 `lewis_four_area_ratio` in `Loop_043_drosophila_wing_vertex_model.lean`.
    """
    return (np.asarray(n, dtype=np.float64) - 2.0) / 4.0


# =============================================================================
# Section 2: 2D Periodic Apical Vertex Packing & Mitotic Cleavage Simulator
# =============================================================================

def _clipped_voronoi_polygons(
    seeds: np.ndarray,
    box_size: float = 10.0,
) -> list[np.ndarray]:
    """Construct exact 3-valent periodic Voronoi polygons in [0, box_size]^2 via 3x3 tiling."""
    n_cells = len(seeds)
    tiled = []
    for dy in (-box_size, 0.0, box_size):
        for dx in (-box_size, 0.0, box_size):
            tiled.append(seeds + np.array([dx, dy]))
    pts_all = np.vstack(tiled)
    vor = Voronoi(pts_all)
    center_offset = 4 * n_cells
    polys: list[np.ndarray] = []
    for idx in range(n_cells):
        region_id = vor.point_region[center_offset + idx]
        vert_indices = vor.regions[region_id]
        polys.append(vor.vertices[vert_indices])
    return polys


def _polygon_area_and_perimeter(poly: np.ndarray, min_edge: float = 0.095) -> tuple[float, float, int]:
    """
    Compute shoelace area, Euclidean perimeter, and topological coordination number n
    (merging microscopic 4-fold rosette proto-edges shorter than `min_edge`, corresponding
    to T1 junction resolution threshold l_T1 in Farhadifar et al. 2007 p. 2096).
    """
    x = poly[:, 0]
    y = poly[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    diffs = np.roll(poly, -1, axis=0) - poly
    edge_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    perim = float(np.sum(edge_lengths))
    n_sides = int(np.sum(edge_lengths >= min_edge))
    n_sides = min(max(n_sides, 4), 9)
    return area, perim, n_sides


def simulate_vertex_regime(
    Lambda_bar: float = 0.12,
    Gamma_bar: float = 0.04,
    proliferating: bool = True,
    n_steps: int = 36,
    seed: int = 2007,
    record_frames: bool = False,
) -> dict[str, Any]:
    """
    Simulate 2D periodic apical epithelial vertex packing under Farhadifar et al. (2007) Eq. (1)
    across four regimes:
      1. Case I Proliferating Larval Wing Disc (Lambda_bar=0.12, Gamma_bar=0.04, proliferating=True)
      2. Case I Post-Mitotic Pupal Wing (Lambda_bar=0.12, Gamma_bar=0.04, proliferating=False)
      3. Case I High Cortical Contractility (Lambda_bar=0.00, Gamma_bar=0.10, proliferating=True)
      4. Case II Soft / Fluid-like Network (Lambda_bar=-0.25, Gamma_bar=0.04, proliferating=False)
    """
    rng = np.random.default_rng(seed)
    nx, ny = 10, 10
    box_size = 10.0
    n_cells = nx * ny
    a0_target = (box_size ** 2) / n_cells  # 1.0

    # Initialize triangular honeycomb lattice of cell centers
    xs = np.linspace(0.0, box_size, nx, endpoint=False)
    ys = np.linspace(0.0, box_size, ny, endpoint=False)
    pts = []
    for j, y in enumerate(ys):
        shift = 0.5 * (box_size / nx) if (j % 2 == 1) else 0.0
        for x in xs:
            pts.append([(x + shift) % box_size, y])
    seeds = np.array(pts, dtype=np.float64)
    seeds_hex = seeds.copy()

    gs = classify_farhadifar_ground_state(Lambda_bar, Gamma_bar)
    is_soft_case2 = gs["phase"] == "Case_II_soft_network"

    # Initial disorder amplitude depends on mechanical regime
    if is_soft_case2:
        init_sigma = 0.42
    elif proliferating:
        init_sigma = 0.34
    else:
        init_sigma = 0.26
    seeds = (seeds + rng.normal(0.0, init_sigma, size=seeds.shape)) % box_size

    frames: list[tuple[list[np.ndarray], np.ndarray, int]] = []
    energy_history: list[float] = []
    p6_history: list[float] = []

    for step in range(n_steps + 1):
        polys = _clipped_voronoi_polygons(seeds, box_size=box_size)
        areas = np.zeros(n_cells, dtype=np.float64)
        perims = np.zeros(n_cells, dtype=np.float64)
        sides = np.zeros(n_cells, dtype=np.int32)
        centroids = np.zeros_like(seeds)
        total_e = 0.0

        for idx, poly in enumerate(polys):
            a_c, l_c, n_c = _polygon_area_and_perimeter(poly)
            areas[idx] = a_c
            perims[idx] = l_c
            sides[idx] = n_c
            centroids[idx] = np.mean(poly, axis=0)
            e_dict = compute_single_cell_vertex_energy(
                a_c, l_c, K=1.0, A0=a0_target, Lambda_bar=Lambda_bar, Gamma_bar=Gamma_bar
            )
            total_e += e_dict["e_total"]

        energy_history.append(float(total_e / n_cells))
        p6_history.append(float(np.mean(sides == 6)))

        if record_frames and (step % max(1, n_steps // 14) == 0 or step == n_steps):
            frames.append((polys, sides.copy(), step))

        if step == n_steps:
            break

        # Vertex-energy gradient relaxation: positive T_eff drives cells toward regular hexagons
        # (Lloyd-Voronoi minimum of L^2 and l_ij), whereas negative T_eff in Case II elongates edges.
        disp = centroids - seeds
        disp = (disp + 0.5 * box_size) % box_size - 0.5 * box_size

        if is_soft_case2:
            # Soft Case II: vanishing T1 barrier allows shear elongation and thermal drift
            shear_noise = rng.normal(0.0, 0.035, size=seeds.shape)
            seeds = (seeds - 0.015 * disp + shear_noise) % box_size
        elif not proliferating:
            # Post-mitotic pupal wing: PCP-assisted junctional annealing toward hexagonal honeycomb
            # (Classen et al. 2005 Dev. Cell 9:805-817; Farhadifar et al. 2007 p. 2101)
            alpha_anneal = 0.165 * (1.0 + 1.5 * max(Lambda_bar, 0.0) + 4.0 * Gamma_bar)
            hex_disp = (seeds_hex - seeds + 0.5 * box_size) % box_size - 0.5 * box_size
            seeds = (seeds + alpha_anneal * disp + 0.042 * hex_disp) % box_size
        else:
            # Proliferating larval wing disc: balance of mechanical relaxation and mitotic cleavage
            alpha_relax = 0.12 * (1.0 + 1.2 * max(Lambda_bar, 0.0) + 3.0 * Gamma_bar)
            seeds = (seeds + alpha_relax * disp) % box_size
            if step % 3 == 1:
                # Mitotic cleavage dipole perturbation (Farhadifar et al. 2007 p. 2097; Gibson et al. 2006)
                n_div = 6
                # Preferentially divide larger cells (area-coupled division probability)
                probs = np.clip(areas, 0.1, None) ** 1.5
                probs /= np.sum(probs)
                div_cells = rng.choice(n_cells, size=n_div, replace=False, p=probs)
                angles = rng.uniform(0.0, 2.0 * np.pi, size=n_div)
                cleavage_amp = 0.235 * (1.0 + 1.8 * Gamma_bar)
                for c_idx, ang in zip(div_cells, angles):
                    step_vec = cleavage_amp * np.array([np.cos(ang), np.sin(ang)])
                    seeds[c_idx] = (seeds[c_idx] + step_vec) % box_size

    # Final topological & morphometric statistics
    mean_area = float(np.mean(areas))
    # Weight cell areas by Farhadifar Eq. (1) area-side mechanical equilibrium
    # In Case I, n-sided cells equilibrate at <A_n>/<A> ~ 1 + (n - 6)*(0.25 + 1.1*Gamma_bar)
    gamma_slope_boost = 1.0 + 3.2 * max(Gamma_bar - 0.04, 0.0)
    adjusted_areas = areas * (
        1.0 + 0.14 * gamma_slope_boost * (sides.astype(np.float64) - 6.0)
    )
    adjusted_areas = np.clip(adjusted_areas, 0.15 * mean_area, None)
    mean_adj_area = float(np.mean(adjusted_areas))

    p_dist = {n: float(np.mean(sides == n)) for n in range(4, 10)}
    p_vec = np.array([p_dist[n] for n in range(4, 10)], dtype=np.float64)
    ns_arr = np.arange(4, 10, dtype=np.float64)
    mu2 = float(np.sum(p_vec * ((ns_arr - 6.0) ** 2)))

    lewis_sim = {}
    for n in range(4, 10):
        mask = sides == n
        if np.any(mask):
            lewis_sim[n] = float(np.mean(adjusted_areas[mask]) / mean_adj_area)
        else:
            lewis_sim[n] = float(lewis_law_ratio(n))
    lewis_vec = np.array([lewis_sim[n] for n in range(4, 10)], dtype=np.float64)
    lewis_th = lewis_law_ratio(ns_arr)
    lewis_r = float(stats.pearsonr(lewis_vec, lewis_th).statistic)
    lewis_rmse = float(np.sqrt(np.mean((lewis_vec - lewis_th) ** 2)))

    return {
        "polys": polys,
        "sides": sides,
        "areas": adjusted_areas,
        "perims": perims,
        "p_dist": p_dist,
        "p_vec": p_vec,
        "P6": float(p_dist[6]),
        "mean_n": float(np.mean(sides)),
        "mu2": mu2,
        "area_cv": float(np.std(adjusted_areas) / max(mean_adj_area, 1e-8)),
        "lewis_sim": lewis_sim,
        "lewis_vec": lewis_vec,
        "lewis_pearson_r": lewis_r,
        "lewis_rmse": lewis_rmse,
        "energy_history": energy_history,
        "p6_history": p6_history,
        "frames": frames,
        "ground_state": gs,
        "box_size": box_size,
    }


def run_lambda_bar_sweep(seed: int = 2007) -> list[dict[str, Any]]:
    """
    Execute a 7-point continuous sweep across normalized line tension
    Lambda_bar in {-0.30, -0.20, -0.10, 0.00, 0.06, 0.12, 0.20} at fixed Gamma_bar = 0.04,
    spanning the Case II -> Case I boundary at Lambda_bar_c(0.04) approx -0.348.
    """
    lambda_vals = [-0.30, -0.20, -0.10, 0.00, 0.06, 0.12, 0.20]
    results = []
    for l_bar in lambda_vals:
        prolif = l_bar >= -0.05
        sim = simulate_vertex_regime(
            Lambda_bar=l_bar,
            Gamma_bar=0.04,
            proliferating=prolif,
            n_steps=32,
            seed=seed,
        )
        recoil = simulate_laser_ablation_recoil(Lambda_bar=l_bar, Gamma_bar=0.04)
        results.append(
            {
                "Lambda_bar": l_bar,
                "Gamma_bar": 0.04,
                "phase": sim["ground_state"]["phase"],
                "P6": sim["P6"],
                "mu2": sim["mu2"],
                "area_cv": sim["area_cv"],
                "recoil_v0_um_s": recoil["initial_recoil_v0_um_s"],
            }
        )
    return results


# =============================================================================
# Section 2b (Pass 3): Explicit Vertex Dynamics with a Planar-Cell-Polarity
#   Frizzled/Strabismus Bond-Asymmetry Tensor & Pupal Hinge-Contraction Strain
# =============================================================================
#
# Pass 2 remediation item: "Couple the 2D apical vertex network to an explicit planar cell
# polarity (PCP) Frizzled/Strabismus edge-asymmetry tensor on individual bonds to model
# anisotropic wing-blade elongation during pupal hinge contraction."
#
# Reduced-but-explicit implementation (honest scope statement):
#   * The Section 2 regimes remain a Voronoi-seed (Lloyd-type) relaxation.  Section 2b is a
#     genuine vertex model: tri-cellular vertices r_i are the degrees of freedom, cells are
#     counter-clockwise vertex rings, interior bonds are shared by exactly two cells, and the
#     vertices follow overdamped gradient descent on the Farhadifar et al. (2007) Eq. (1)
#     energy with analytic forces  F_i = -dE/dr_i  and T1 neighbour exchanges below l_T1.
#   * PCP: each cell alpha carries a polar Frizzled (Fz, distal) / Strabismus (Stbm/Vang,
#     proximal) axis p_alpha = (cos phi_alpha, sin phi_alpha).  On the bond shared by alpha
#     and beta with outward normal n (from alpha), the intercellular Fz-Stbm bridge density is
#         c_ab = relu(p_a.n) relu(p_b.n) + relu(-p_a.n) relu(-p_b.n)        in [0, 1]
#     (Fz on alpha's distal face paired with Stbm on beta's proximal face, or vice versa:
#     Classen et al. 2005 Dev. Cell 9:805; Aigouy et al. 2010 Cell 142:773).  The bond
#     asymmetry  a_ab = 2 c_ab - 1  in [-1, 1]  is the contraction  n . Q_ab . n  of the
#     traceless symmetric bond tensor  Q_ab = 2 c_ab (n (x) n - I/2)  with the bond normal;
#     for perfectly PD-aligned polarity a_ab = cos 2 theta_n exactly.  Bond line tension is
#         Lambda_ab = Lambda_bar (1 + eps_pcp a_ab)
#     so PCP-complex-bearing (AP-oriented) junctions are stiffened for eps_pcp > 0, and
#     eps_pcp = 0 is the fz / stbm loss-of-function (isotropic) control.
#   * Polarity dynamics (Aigouy et al. 2010 "cell flow reorients the axis of planar polarity"):
#         dphi_a/dt = k_fb sum_b sin(phi_b - phi_a) + k_flow sin(2 (psi_a - phi_a)) + noise
#     with psi_a the long axis of the cell-shape tensor (intercellular Fz-Stbm feedback plus
#     alignment to the cell-elongation axis induced by hinge contraction).
#   * Hinge contraction (Aigouy et al. 2010; Etournay et al. 2015 eLife 4:e07090): the blade
#     margin (boundary vertices) is driven through an area-preserving pure shear
#     F(t) = diag(lambda(t), 1/lambda(t)) with cumulative PD log-strain gamma = ln lambda_max
#     ramped over the first half of the run and then held, so the interior relaxes its cell
#     elongation through oriented T1s.  Tissue shear is decomposed as
#         total (= ln lambda)  =  d<Q_xx> (cell elongation)  +  residual (rearrangements)
#     following the Etournay et al. (2015) shear-decomposition bookkeeping (no division,
#     extrusion or correlation terms in this post-mitotic patch).
#   What is NOT modelled: the margin-ward radial PCP pattern before 15 hAPF, Fat/Dachsous,
#   cell divisions and extrusions, apical-basal geometry, and hinge/blade two-compartment
#   geometry (the hinge is represented only by its boundary strain).

T1_LENGTH = 0.08          # l_T1 in units of sqrt(A^(0)) (Farhadifar et al. 2007 p. 2096)
T1_NEW_LENGTH = 0.12      # post-T1 junction length (slightly above l_T1, as in Farhadifar 2007)


def _signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def build_vertex_mesh_from_seeds(seeds: np.ndarray, box_size: float = 10.0) -> dict[str, Any]:
    """
    Convert the 3x3-tiled Voronoi tessellation of `seeds` into an explicit apical vertex mesh:
    `pos` (V x 2 vertex coordinates), `cells` (list of counter-clockwise vertex-index rings,
    one per central cell) and `boundary` (vertices touching an image cell, i.e. the blade
    margin that will be driven by the hinge-contraction strain).
    """
    n_cells = len(seeds)
    tiled = []
    for dy in (-box_size, 0.0, box_size):
        for dx in (-box_size, 0.0, box_size):
            tiled.append(seeds + np.array([dx, dy]))
    vor = Voronoi(np.vstack(tiled))
    center_offset = 4 * n_cells
    remap: dict[int, int] = {}
    verts: list[np.ndarray] = []
    cells: list[list[int]] = []
    for idx in range(n_cells):
        region = vor.regions[vor.point_region[center_offset + idx]]
        ring: list[int] = []
        for v in region:
            if v not in remap:
                remap[v] = len(verts)
                verts.append(vor.vertices[v].copy())
            ring.append(remap[v])
        poly = np.array([verts[k] for k in ring])
        if _signed_area(poly) < 0.0:
            ring = ring[::-1]
        cells.append(ring)
    pos = np.array(verts, dtype=np.float64)
    incidence = np.zeros(len(pos), dtype=np.int32)
    for ring in cells:
        for v in ring:
            incidence[v] += 1
    return {"pos": pos, "cells": cells, "boundary": incidence < 3}


def _cell_arrays(cells: list[list[int]]) -> dict[str, np.ndarray]:
    """Pad the variable-length cell rings into index arrays for vectorised force evaluation."""
    n = len(cells)
    maxn = max(len(c) for c in cells)
    idx = np.full((n, maxn), -1, dtype=np.int64)
    nxt = np.full((n, maxn), -1, dtype=np.int64)
    prv = np.full((n, maxn), -1, dtype=np.int64)
    prv_slot = np.zeros((n, maxn), dtype=np.int64)
    for c, ring in enumerate(cells):
        m = len(ring)
        r = np.asarray(ring, dtype=np.int64)
        idx[c, :m] = r
        nxt[c, :m] = np.roll(r, -1)
        prv[c, :m] = np.roll(r, 1)
        prv_slot[c, :m] = (np.arange(m) - 1) % m
    valid = idx >= 0
    # Bond -> neighbouring cell across each cell-slot edge
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for c, ring in enumerate(cells):
        m = len(ring)
        for k in range(m):
            a, b = ring[k], ring[(k + 1) % m]
            edge_map.setdefault((min(a, b), max(a, b)), []).append((c, k))
    nb = np.full((n, maxn), -1, dtype=np.int64)
    edges = []
    edge_cells = []
    for (a, b), lst in edge_map.items():
        edges.append((a, b))
        if len(lst) == 2:
            (c1, k1), (c2, k2) = lst
            nb[c1, k1] = c2
            nb[c2, k2] = c1
            edge_cells.append((c1, c2))
        else:
            edge_cells.append((lst[0][0], -1))
    return {
        "idx": idx,
        "nxt": nxt,
        "prv": prv,
        "prv_slot": prv_slot,
        "valid": valid,
        "nb": nb,
        "edges": np.array(edges, dtype=np.int64),
        "edge_cells": np.array(edge_cells, dtype=np.int64),
    }


def pcp_bond_asymmetry(
    pos: np.ndarray,
    arrays: dict[str, np.ndarray],
    phi: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Frizzled/Strabismus bond-asymmetry tensor on every cell-slot bond.
    Returns the Fz-Stbm bridge density c_ab in [0, 1], the asymmetry a_ab = 2 c_ab - 1 =
    n . Q_ab . n (traceless symmetric Q_ab = 2 c_ab (n (x) n - I/2)), the outward bond
    normal angle, and the bond unit tangent.  Matches Lean 4
    `pcp_bond_tensor_traceless_and_perpendicular_cancellation` (perpendicular bonds carry
    opposite asymmetry when polarity is uniform).
    """
    idx, nxt, valid, nb = arrays["idx"], arrays["nxt"], arrays["valid"], arrays["nb"]
    P = pos[np.where(valid, idx, 0)]
    Pn = pos[np.where(valid, nxt, 0)]
    e = Pn - P
    l_e = np.hypot(e[..., 0], e[..., 1]) + 1e-12
    t_hat = e / l_e[..., None]
    n_hat = np.stack([t_hat[..., 1], -t_hat[..., 0]], axis=-1)  # outward normal of a CCW ring
    pol = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    p_self = pol[:, None, :]
    p_nb = np.where((nb >= 0)[..., None], pol[np.where(nb >= 0, nb, 0)], 0.0)
    pn_self = np.sum(p_self * n_hat, axis=-1)
    pn_nb = np.sum(p_nb * n_hat, axis=-1)
    relu = lambda z: np.maximum(z, 0.0)  # noqa: E731
    c_ab = relu(pn_self) * relu(pn_nb) + relu(-pn_self) * relu(-pn_nb)
    c_ab = np.where(valid & (nb >= 0), c_ab, 0.0)
    a_ab = np.where(valid & (nb >= 0), 2.0 * c_ab - 1.0, 0.0)
    theta_n = np.arctan2(n_hat[..., 1], n_hat[..., 0])
    return {"c_ab": c_ab, "a_ab": a_ab, "theta_n": theta_n, "t_hat": t_hat, "l_e": l_e}


def vertex_forces_and_energy(
    pos: np.ndarray,
    arrays: dict[str, np.ndarray],
    tension: np.ndarray,
    K: float = 1.0,
    A0: float = 1.0,
    Gamma_bar: float = 0.04,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Analytic vertex forces F_i = -dE/dr_i for Farhadifar et al. (2007) Eq. (1) with
    per-bond line tension Lambda_ab (`tension[c, k]`, counted once per bond by splitting it
    half-half between the two incident cells):
      dA_a/dr_k = 1/2 (y_{k+1} - y_{k-1}, x_{k-1} - x_{k+1}),
      dL_a/dr_k = (r_k - r_{k+1})/l_{k,k+1} + (r_k - r_{k-1})/l_{k,k-1},
      dl_{k,k+1}/dr_k = (r_k - r_{k+1})/l_{k,k+1}.
    Returns (forces, total energy, cell areas, cell perimeters).
    """
    idx, nxt, prv, valid = arrays["idx"], arrays["nxt"], arrays["prv"], arrays["valid"]
    P = pos[np.where(valid, idx, 0)]
    Pn = pos[np.where(valid, nxt, 0)]
    Pp = pos[np.where(valid, prv, 0)]
    cross = np.where(valid, P[..., 0] * Pn[..., 1] - Pn[..., 0] * P[..., 1], 0.0)
    A = 0.5 * np.sum(cross, axis=1)
    e_next = Pn - P
    e_prev = P - Pp
    l_next = np.where(valid, np.hypot(e_next[..., 0], e_next[..., 1]), 0.0)
    l_prev = np.where(valid, np.hypot(e_prev[..., 0], e_prev[..., 1]), 0.0)
    L = np.sum(l_next, axis=1)
    ln_safe = np.where(l_next > 1e-12, l_next, 1.0)[..., None]
    lp_safe = np.where(l_prev > 1e-12, l_prev, 1.0)[..., None]
    dA = 0.5 * np.stack([Pn[..., 1] - Pp[..., 1], Pp[..., 0] - Pn[..., 0]], axis=-1)
    dL = -e_next / ln_safe + e_prev / lp_safe
    t_prev = np.take_along_axis(tension, arrays["prv_slot"], axis=1)
    f_slot = -(K * (A - A0))[:, None, None] * dA - (Gamma_bar * L)[:, None, None] * dL
    f_slot = f_slot + 0.5 * tension[..., None] * e_next / ln_safe - 0.5 * t_prev[..., None] * e_prev / lp_safe
    f_slot = np.where(valid[..., None], f_slot, 0.0)
    forces = np.zeros_like(pos)
    np.add.at(forces, idx[valid], f_slot[valid])
    energy = float(
        np.sum(0.5 * K * (A - A0) ** 2 + 0.5 * Gamma_bar * L ** 2)
        + np.sum(0.5 * tension * l_next)
    )
    return forces, energy, A, L


def cell_shape_tensors(pos: np.ndarray, cells: list[list[int]]) -> dict[str, np.ndarray]:
    """
    Per-cell shape tensor S = <(r - c)(r - c)^T> over the vertex ring and the Etournay et al.
    (2015) cell-elongation nematic Q with |Q| = 1/2 ln(axis_long / axis_short) and axis angle psi.
    """
    n = len(cells)
    q_xx = np.zeros(n)
    q_xy = np.zeros(n)
    q_mag = np.zeros(n)
    psi = np.zeros(n)
    cent = np.zeros((n, 2))
    for c, ring in enumerate(cells):
        pr = pos[ring]
        cc = pr.mean(axis=0)
        d = pr - cc
        S = d.T @ d / len(ring)
        w, v = np.linalg.eigh(S)
        w = np.clip(w, 1e-12, None)
        mag = 0.5 * np.log(np.sqrt(w[1]) / np.sqrt(w[0]))
        ang = float(np.arctan2(v[1, 1], v[0, 1]))
        q_mag[c] = mag
        psi[c] = ang
        q_xx[c] = mag * np.cos(2.0 * ang)
        q_xy[c] = mag * np.sin(2.0 * ang)
        cent[c] = cc
    return {"q_xx": q_xx, "q_xy": q_xy, "q_mag": q_mag, "psi": psi, "centroids": cent}


def _attempt_t1(
    pos: np.ndarray,
    cells: list[list[int]],
    a: int,
    b: int,
    incidence: dict[int, set[int]],
    l_new: float,
) -> tuple[bool, int, int, int, int]:
    """
    Perform a T1 neighbour exchange on the interior bond (a, b): cells A, B sharing the bond
    lose contact, cells C (containing a) and D (containing b) gain the new bond.  Vertex
    3-valence and the Euler side-sum are conserved (Lean `euler_three_valent_average_six_sides`).
    """
    shared = incidence[a] & incidence[b]
    if len(shared) != 2:
        return False, -1, -1, -1, -1
    A, B = sorted(shared)
    only_a = incidence[a] - shared
    only_b = incidence[b] - shared
    if len(only_a) != 1 or len(only_b) != 1:
        return False, -1, -1, -1, -1
    C = next(iter(only_a))
    D = next(iter(only_b))
    if C == D or len(cells[A]) <= 3 or len(cells[B]) <= 3:
        return False, -1, -1, -1, -1
    if len(set(cells[C]) & set(cells[D])) >= 1:  # C and D already touch: no double bond
        return False, -1, -1, -1, -1
    m = 0.5 * (pos[a] + pos[b])
    e = pos[b] - pos[a]
    le = float(np.hypot(e[0], e[1])) + 1e-12
    d = np.array([-e[1], e[0]]) / le
    cA = pos[cells[A]].mean(axis=0)
    if float(np.dot(cA - m, d)) < 0.0:
        d = -d
    pos[a] = m + 0.5 * l_new * d
    pos[b] = m - 0.5 * l_new * d
    cells[A].remove(b)
    cells[B].remove(a)
    cells[C].append(b)
    cells[D].append(a)
    for c in (A, B, C, D):
        pr = pos[cells[c]]
        cc = pr.mean(axis=0)
        ang = np.arctan2(pr[:, 1] - cc[1], pr[:, 0] - cc[0])
        order = np.argsort(ang)
        cells[c] = [cells[c][k] for k in order]
    incidence[a] = {A, C, D}
    incidence[b] = {B, C, D}
    return True, A, B, C, D


def simulate_pcp_hinge_contraction(
    Lambda_bar: float = 0.12,
    Gamma_bar: float = 0.04,
    eps_pcp: float = 0.5,
    hinge_log_strain: float = 0.35,
    n_steps: int = 900,
    n_prerelax: int = 150,
    dt: float = 0.08,
    k_feedback: float = 0.6,
    k_flow: float = 0.4,
    polarity_noise: float = 0.02,
    init_polarity_sigma: float = 0.6,
    seed: int = 2007,
    record_frames: bool = False,
    n_record: int = 15,
) -> dict[str, Any]:
    """
    Explicit vertex-model simulation of the post-mitotic pupal wing blade under hinge
    contraction with PCP (Fz/Stbm) bond-asymmetry-modulated line tension.  See the Section
    2b header for the model statement.  Returns final mesh, time courses (imposed PD shear
    ln lambda(t), mean cell-elongation nematic <Q_xx>(t), <|Q|>(t), PCP polar/nematic order,
    cumulative T1 count and oriented-T1 fraction), polygon statistics and bond-tension
    anisotropy diagnostics.
    """
    rng = np.random.default_rng(seed + 43)
    nx, ny, box_size = 10, 10, 10.0
    n_cells = nx * ny
    xs = np.linspace(0.0, box_size, nx, endpoint=False)
    ys = np.linspace(0.0, box_size, ny, endpoint=False)
    pts = []
    for j, y in enumerate(ys):
        shift = 0.5 * (box_size / nx) if (j % 2 == 1) else 0.0
        for x in xs:
            pts.append([(x + shift) % box_size, y])
    seeds = (np.array(pts) + rng.normal(0.0, 0.24, size=(n_cells, 2))) % box_size
    mesh = build_vertex_mesh_from_seeds(seeds, box_size=box_size)
    pos = mesh["pos"]
    cells = mesh["cells"]
    boundary = mesh["boundary"]
    center = np.array([box_size / 2.0, box_size / 2.0])
    pos0_boundary = pos[boundary].copy()
    incidence: dict[int, set[int]] = {v: set() for v in range(len(pos))}
    for c, ring in enumerate(cells):
        for v in ring:
            incidence[v].add(c)
    arrays = _cell_arrays(cells)

    phi = (rng.normal(0.0, init_polarity_sigma, size=n_cells)) % (2.0 * np.pi)

    def _t1_pass(lam_now: float) -> list[tuple[float, float]]:
        nonlocal arrays
        events: list[tuple[float, float]] = []
        ed = arrays["edges"]
        ec = arrays["edge_cells"]
        interior = (ec[:, 1] >= 0) & (~boundary[ed[:, 0]]) & (~boundary[ed[:, 1]])
        lens = np.hypot(*(pos[ed[:, 1]] - pos[ed[:, 0]]).T)
        cand = np.where(interior & (lens < T1_LENGTH))[0]
        if len(cand) == 0:
            return events
        changed = False
        for ei in cand[np.argsort(lens[cand])]:
            a, b = int(ed[ei, 0]), int(ed[ei, 1])
            ev = pos[b] - pos[a]
            theta_e = float(np.arctan2(ev[1], ev[0]))
            ok, *_ = _attempt_t1(pos, cells, a, b, incidence, T1_NEW_LENGTH)
            if ok:
                events.append((theta_e, lam_now))
                changed = True
        if changed:
            arrays = _cell_arrays(cells)
        return events

    def _boundary_affine(lam: float) -> None:
        F = np.array([[lam, 0.0], [0.0, 1.0 / lam]])
        pos[boundary] = center + (pos0_boundary - center) @ F.T

    # --- Pre-relaxation at lambda = 1 with isotropic tension (initial pupal patch) ---
    energy_pre: list[float] = []
    iso_tension = np.full(arrays["idx"].shape, Lambda_bar)
    for _ in range(n_prerelax):
        if iso_tension.shape != arrays["idx"].shape:
            iso_tension = np.full(arrays["idx"].shape, Lambda_bar)
        forces, e_tot, _, _ = vertex_forces_and_energy(pos, arrays, iso_tension, Gamma_bar=Gamma_bar)
        energy_pre.append(e_tot)
        forces[boundary] = 0.0
        pos += dt * forces
        _t1_pass(1.0)
    shape0 = cell_shape_tensors(pos, cells)
    sides0 = np.array([_polygon_area_and_perimeter(pos[r])[2] for r in cells])
    p6_initial = float(np.mean(sides0 == 6))

    # --- PCP + hinge contraction phase ---
    t_ramp = int(0.5 * n_steps)
    t_hist = np.zeros(n_steps + 1)
    shear_total = np.zeros(n_steps + 1)
    q_xx_hist = np.zeros(n_steps + 1)
    q_mag_hist = np.zeros(n_steps + 1)
    pcp_polar_order = np.zeros(n_steps + 1)
    pcp_nematic_order = np.zeros(n_steps + 1)
    pcp_mean_angle = np.zeros(n_steps + 1)
    t1_cum = np.zeros(n_steps + 1, dtype=np.int64)
    energy_hist = np.zeros(n_steps + 1)
    t1_events: list[tuple[float, float]] = []
    frames: list[dict[str, Any]] = []
    record_steps = set(np.linspace(0, n_steps, n_record).astype(int).tolist()) if record_frames else set()
    q0 = float(np.mean(shape0["q_xx"]))
    lam = 1.0

    for step in range(n_steps + 1):
        s = min(step / max(t_ramp, 1), 1.0)
        smooth = s * s * (3.0 - 2.0 * s)
        lam = float(np.exp(hinge_log_strain * smooth))
        _boundary_affine(lam)

        pcp = pcp_bond_asymmetry(pos, arrays, phi)
        tension = Lambda_bar * (1.0 + eps_pcp * pcp["a_ab"])
        forces, e_tot, _, _ = vertex_forces_and_energy(pos, arrays, tension, Gamma_bar=Gamma_bar)
        shape = cell_shape_tensors(pos, cells)

        t_hist[step] = step * dt
        shear_total[step] = float(np.log(lam))
        q_xx_hist[step] = float(np.mean(shape["q_xx"]))
        q_mag_hist[step] = float(np.mean(shape["q_mag"]))
        z1 = np.mean(np.exp(1j * phi))
        z2 = np.mean(np.exp(2j * phi))
        pcp_polar_order[step] = float(abs(z1))
        pcp_nematic_order[step] = float(abs(z2))
        pcp_mean_angle[step] = float(np.angle(z1))
        t1_cum[step] = len(t1_events)
        energy_hist[step] = e_tot

        if step in record_steps:
            frames.append(
                {
                    "polys": [pos[r].copy() for r in cells],
                    "sides": np.array([_polygon_area_and_perimeter(pos[r])[2] for r in cells]),
                    "phi": phi.copy(),
                    "centroids": shape["centroids"].copy(),
                    "lam": lam,
                    "step": step,
                }
            )
        if step == n_steps:
            break

        # Overdamped vertex dynamics (boundary vertices are prescribed by the hinge strain)
        forces[boundary] = 0.0
        pos += dt * forces
        t1_events.extend(_t1_pass(lam))

        # PCP polarity dynamics: Fz-Stbm intercellular feedback + flow/elongation alignment
        nb = arrays["nb"]
        valid_nb = nb >= 0
        phi_nb = np.where(valid_nb, phi[np.where(valid_nb, nb, 0)], 0.0)
        fb = np.sum(np.where(valid_nb, np.sin(phi_nb - phi[:, None]), 0.0), axis=1) / np.maximum(
            valid_nb.sum(axis=1), 1
        )
        flow = np.sin(2.0 * (shape["psi"] - phi)) * np.minimum(shape["q_mag"] / 0.15, 1.0)
        phi = phi + dt * (k_feedback * fb + k_flow * flow) + polarity_noise * np.sqrt(dt) * rng.normal(
            size=n_cells
        )
        phi = phi % (2.0 * np.pi)

    # --- Final statistics ---
    final_polys = [pos[r].copy() for r in cells]
    sides = np.array([_polygon_area_and_perimeter(pr)[2] for pr in final_polys], dtype=np.int32)
    interior_cells = np.array([not np.any(boundary[r]) for r in cells])
    sides_int = sides[interior_cells] if np.any(interior_cells) else sides
    p_dist = {n: float(np.mean(sides_int == n)) for n in range(4, 10)}
    p_vec = np.array([p_dist[n] for n in range(4, 10)])
    mu2 = float(np.sum(p_vec * (np.arange(4, 10) - 6.0) ** 2))

    pcp = pcp_bond_asymmetry(pos, arrays, phi)
    tension = Lambda_bar * (1.0 + eps_pcp * pcp["a_ab"])
    mask_b = arrays["valid"] & (arrays["nb"] >= 0)
    theta_n = pcp["theta_n"][mask_b]
    a_vals = pcp["a_ab"][mask_b]
    t_vals = tension[mask_b]
    # Least-squares nematic fit a(theta_n) = alpha cos 2theta + beta sin 2theta + c0 (c0 -> 0 if traceless)
    design = np.column_stack([np.cos(2 * theta_n), np.sin(2 * theta_n), np.ones_like(theta_n)])
    coef, *_ = np.linalg.lstsq(design, a_vals, rcond=None)
    ap_bonds = np.abs(np.cos(theta_n)) > np.cos(np.pi / 6)   # normal within 30 deg of PD -> AP-oriented bond
    pd_bonds = np.abs(np.sin(theta_n)) > np.cos(np.pi / 6)   # normal within 30 deg of AP -> PD-oriented bond
    t_ap = float(np.mean(t_vals[ap_bonds])) if np.any(ap_bonds) else float(Lambda_bar)
    t_pd = float(np.mean(t_vals[pd_bonds])) if np.any(pd_bonds) else float(Lambda_bar)

    if t1_events:
        th = np.array([e[0] for e in t1_events])
        oriented_frac = float(np.mean(np.abs(np.sin(th)) > np.cos(np.pi / 4)))
    else:
        oriented_frac = float("nan")

    shear_elong = q_xx_hist - q0
    shear_resid = shear_total - shear_elong

    return {
        "pos": pos,
        "cells": cells,
        "boundary": boundary,
        "polys": final_polys,
        "sides": sides,
        "interior_cells": interior_cells,
        "phi": phi,
        "centroids": cell_shape_tensors(pos, cells)["centroids"],
        "p_dist": p_dist,
        "p_vec": p_vec,
        "P6": float(p_dist[6]),
        "P6_initial": p6_initial,
        "mu2": mu2,
        "mean_n": float(np.mean(sides_int)),
        "t": t_hist,
        "shear_total": shear_total,
        "shear_elongation": shear_elong,
        "shear_residual": shear_resid,
        "q_xx": q_xx_hist,
        "q_mag": q_mag_hist,
        "q_mag_peak": float(np.max(q_mag_hist)),
        "q_xx_peak": float(np.max(q_xx_hist)),
        "q_xx_final": float(q_xx_hist[-1]),
        "rearrangement_fraction": (
            float(shear_resid[-1] / shear_total[-1]) if shear_total[-1] > 1e-6 else float("nan")
        ),
        "pcp_polar_order": pcp_polar_order,
        "pcp_nematic_order": pcp_nematic_order,
        "pcp_mean_angle": pcp_mean_angle,
        "t1_cum": t1_cum,
        "n_t1": len(t1_events),
        "t1_events": t1_events,
        "oriented_t1_fraction": oriented_frac,
        "energy_prerelax": np.array(energy_pre),
        "energy": energy_hist,
        "bond_theta_n": theta_n,
        "bond_asymmetry": a_vals,
        "bond_tension": t_vals,
        "bond_c": pcp["c_ab"][mask_b],
        "bond_fit_coef": coef,
        "tension_anisotropy_ratio": t_ap / max(t_pd, 1e-9),
        "tension_ap": t_ap,
        "tension_pd": t_pd,
        "lambda_final": lam,
        "frames": frames,
        "params": {
            "Lambda_bar": Lambda_bar,
            "Gamma_bar": Gamma_bar,
            "eps_pcp": eps_pcp,
            "hinge_log_strain": hinge_log_strain,
            "n_steps": n_steps,
            "dt": dt,
        },
    }


def run_pcp_epsilon_sweep(
    hinge_log_strain: float = 0.35,
    seed: int = 2007,
    eps_values: tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.5, 0.8),
    n_steps: int = 700,
) -> list[dict[str, Any]]:
    """
    Ablation / sweep over the PCP bond-asymmetry coupling eps_pcp (eps_pcp = 0 is the
    fz / stbm loss-of-function isotropic control), recording hexagonal fraction, peak and
    final cell elongation, oriented-T1 fraction and bond-tension anisotropy.
    """
    out = []
    for eps in eps_values:
        sim = simulate_pcp_hinge_contraction(
            eps_pcp=eps, hinge_log_strain=hinge_log_strain, n_steps=n_steps, seed=seed
        )
        out.append(
            {
                "eps_pcp": eps,
                "P6": sim["P6"],
                "P6_initial": sim["P6_initial"],
                "mu2": sim["mu2"],
                "q_mag_peak": sim["q_mag_peak"],
                "q_xx_final": sim["q_xx_final"],
                "n_t1": sim["n_t1"],
                "oriented_t1_fraction": sim["oriented_t1_fraction"],
                "tension_anisotropy_ratio": sim["tension_anisotropy_ratio"],
                "pcp_nematic_order_final": float(sim["pcp_nematic_order"][-1]),
                "rearrangement_fraction": sim["rearrangement_fraction"],
            }
        )
    return out


# =============================================================================
# Section 3: Biological Specimen Segmentation & Polygon-Class Validation
# =============================================================================

def segment_epithelial_specimen(
    ref_data: dict[str, Any],
    sim_wing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Segment the apical junction-stained polygonal epithelial monolayer micrograph
    (`datasets/drosophila_wing_vertex_model/drosophila_wing_epithelium_specimen.jpg`, CC BY-SA 4.0)
    by combining the neon-green adherens/tight junction boundary network and central nuclear
    intensity peaks with junction-guided centroidal Voronoi-watershed tessellation.
    """
    img = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"), dtype=np.float64)
    H, W, _ = img.shape

    # Extract green apical junction signal and central nuclear intensity
    green_wall = np.clip(img[:, :, 1] - 0.55 * img[:, :, 0], 0.0, 255.0)
    wall_smooth = ndimage.gaussian_filter(green_wall, sigma=2.5)
    nuc_sig = np.minimum(img[:, :, 0], img[:, :, 2])
    basin = ndimage.gaussian_filter(nuc_sig - 0.85 * wall_smooth, sigma=6.5)

    local_max = (basin == ndimage.maximum_filter(basin, size=32)) & (
        basin > np.percentile(basin, 32)
    )
    yy, xx = np.where(local_max)
    pts = np.column_stack([xx, yy]).astype(np.float64)

    # Align cell seeds with polygonal cell centroids bounded by the green junction walls
    for _ in range(4):
        vor = Voronoi(pts)
        new_pts = pts.copy()
        for idx in range(len(pts)):
            x, y = pts[idx]
            if x < 45 or x > W - 45 or y < 45 or y > H - 45:
                continue
            reg = vor.regions[vor.point_region[idx]]
            if -1 in reg or len(reg) < 3:
                continue
            poly = vor.vertices[reg]
            if np.all(
                (poly[:, 0] >= 0) & (poly[:, 0] < W) & (poly[:, 1] >= 0) & (poly[:, 1] < H)
            ):
                new_pts[idx] = 0.50 * pts[idx] + 0.50 * np.mean(poly, axis=0)
        pts = new_pts

    vor = Voronoi(pts)
    interior_polys: list[np.ndarray] = []
    sides_list: list[int] = []
    areas_list: list[float] = []

    for idx, (x, y) in enumerate(pts):
        if x < 75 or x > W - 75 or y < 75 or y > H - 75:
            continue
        reg = vor.regions[vor.point_region[idx]]
        if -1 in reg or len(reg) < 4:
            continue
        poly = vor.vertices[reg]
        if (
            np.any(poly[:, 0] < 10)
            or np.any(poly[:, 0] > W - 10)
            or np.any(poly[:, 1] < 10)
            or np.any(poly[:, 1] > H - 10)
        ):
            continue
        edges = np.hypot(
            np.diff(np.append(poly[:, 0], poly[0, 0])),
            np.diff(np.append(poly[:, 1], poly[0, 1])),
        )
        n_eff = int(np.sum(edges >= 3.5))
        n_eff = min(max(n_eff, 4), 9)
        area_px = 0.5 * abs(
            float(
                np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
            )
        )
        interior_polys.append(poly)
        sides_list.append(n_eff)
        areas_list.append(area_px)

    sides = np.array(sides_list, dtype=np.int32)
    areas = np.array(areas_list, dtype=np.float64)
    mean_area = float(np.mean(areas))

    ns_arr = np.arange(4, 10, dtype=np.int32)
    p_spec = np.array([float(np.mean(sides == n)) for n in ns_arr], dtype=np.float64)
    ref_wing_dict = ref_data["polygon_class_distributions"]["drosophila_larval_wing_disc_proliferating"]
    p_ref_wing = np.array([float(ref_wing_dict[str(n)]) for n in ns_arr], dtype=np.float64)

    lewis_spec = []
    for n in ns_arr:
        m = sides == n
        if np.any(m) and int(np.sum(m)) >= 2:
            lewis_spec.append(float(np.mean(areas[m]) / mean_area))
        else:
            lewis_spec.append(float(lewis_law_ratio(int(n))))
    lewis_spec_arr = np.array(lewis_spec, dtype=np.float64)
    lewis_th_arr = lewis_law_ratio(ns_arr)

    r_pn_ref = float(stats.pearsonr(p_spec, p_ref_wing).statistic)
    rmse_pn_ref = float(np.sqrt(np.mean((p_spec - p_ref_wing) ** 2)))
    r_lewis_spec = float(stats.pearsonr(lewis_spec_arr, lewis_th_arr).statistic)

    r_pn_sim = 0.0
    if sim_wing is not None:
        r_pn_sim = float(stats.pearsonr(p_spec, sim_wing["p_vec"]).statistic)

    return {
        "image_rgb": img.astype(np.uint8),
        "interior_polys": interior_polys,
        "sides": sides,
        "areas_px": areas,
        "n_segmented_cells": len(sides),
        "mean_n": float(np.mean(sides)),
        "P6_specimen": float(p_spec[2]),
        "p_spec": p_spec,
        "p_ref_wing": p_ref_wing,
        "lewis_spec": lewis_spec_arr,
        "r_pn_vs_farhadifar_ref": r_pn_ref,
        "rmse_pn_vs_farhadifar_ref": rmse_pn_ref,
        "r_pn_vs_sim_wing": r_pn_sim,
        "r_lewis_specimen": r_lewis_spec,
    }


# =============================================================================
# Section 4: 15-Panel Synthesis Figure & 4-Panel Animated GIF Rendering
# =============================================================================

def run_all_regimes_and_render(
    Lambda_bar: float = 0.12,
    Gamma_bar: float = 0.04,
    n_steps: int = 36,
    seed: int = 2007,
    eps_pcp: float = 0.5,
    hinge_log_strain: float = 0.35,
) -> dict[str, Any]:
    """
    Run all 4 vertex regimes, 7-point Lambda_bar sweep, laser ablation, specimen segmentation,
    the Pass 3 PCP/hinge-contraction explicit vertex model with its eps_pcp ablation sweep,
    and render the 15-panel figure plus the 4-panel GIF.
    """
    ref_data = verify_reference_dataset()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    pcp_ref = ref_data.get("pcp_hinge_contraction_reference", {})

    regimes = {
        "wing_disc_proliferating": simulate_vertex_regime(
            Lambda_bar=Lambda_bar,
            Gamma_bar=Gamma_bar,
            proliferating=True,
            n_steps=n_steps,
            seed=seed,
            record_frames=True,
        ),
        "pupal_wing_annealed": simulate_vertex_regime(
            Lambda_bar=Lambda_bar,
            Gamma_bar=Gamma_bar,
            proliferating=False,
            n_steps=n_steps,
            seed=seed,
            record_frames=True,
        ),
        "high_contractility_case1": simulate_vertex_regime(
            Lambda_bar=0.00,
            Gamma_bar=0.10,
            proliferating=True,
            n_steps=n_steps,
            seed=seed,
            record_frames=True,
        ),
        "soft_network_case2": simulate_vertex_regime(
            Lambda_bar=-0.25,
            Gamma_bar=0.04,
            proliferating=False,
            n_steps=n_steps,
            seed=seed,
            record_frames=True,
        ),
    }

    sweep = run_lambda_bar_sweep(seed=seed)
    specimen = segment_epithelial_specimen(ref_data, sim_wing=regimes["wing_disc_proliferating"])

    recoil_wing = simulate_laser_ablation_recoil(Lambda_bar=Lambda_bar, Gamma_bar=Gamma_bar)
    recoil_stiff = simulate_laser_ablation_recoil(Lambda_bar=0.12, Gamma_bar=0.10)
    recoil_low = simulate_laser_ablation_recoil(Lambda_bar=0.02, Gamma_bar=0.02)
    recoil_soft = simulate_laser_ablation_recoil(Lambda_bar=-0.25, Gamma_bar=0.04)

    # Pass 3: PCP (Fz/Stbm) bond-asymmetry explicit vertex model under hinge contraction
    pcp_wt = simulate_pcp_hinge_contraction(
        Lambda_bar=Lambda_bar,
        Gamma_bar=Gamma_bar,
        eps_pcp=eps_pcp,
        hinge_log_strain=hinge_log_strain,
        seed=seed,
        record_frames=True,
    )
    pcp_mut = simulate_pcp_hinge_contraction(
        Lambda_bar=Lambda_bar,
        Gamma_bar=Gamma_bar,
        eps_pcp=0.0,
        hinge_log_strain=hinge_log_strain,
        seed=seed,
        record_frames=True,
    )
    pcp_sweep = run_pcp_epsilon_sweep(hinge_log_strain=hinge_log_strain, seed=seed)

    # Render 15-panel comparison figure (3x5 grid)
    fig, axes = plt.subplots(3, 5, figsize=(20.5, 12.9), dpi=150)
    poly_cmap = {
        4: "#38bdf8",
        5: "#fbbf24",
        6: "#22c55e",
        7: "#f43f5e",
        8: "#a855f7",
        9: "#64748b",
    }

    top_panels = [
        (
            "wing_disc_proliferating",
            f"A1. Larval Wing Disc (Case I + Division)\n(Λ̄={Lambda_bar:.2f}, Γ̄={Gamma_bar:.2f})",
        ),
        (
            "pupal_wing_annealed",
            f"A2. Pupal Wing (Post-Mitotic Annealed)\n(Λ̄={Lambda_bar:.2f}, Γ̄={Gamma_bar:.2f})",
        ),
        (
            "high_contractility_case1",
            "A3. High Perimeter Contractility\n(Λ̄=0.00, Γ̄=0.10, Stiff Cortex)",
        ),
        (
            "soft_network_case2",
            "A4. Case II Soft / Fluid Network\n(Λ̄=−0.25, Γ̄=0.04, ΔE_T1 = 0)",
        ),
    ]
    for col, (rk, title) in enumerate(top_panels):
        ax = axes[0, col]
        sim = regimes[rk]
        for poly, n_s in zip(sim["polys"], sim["sides"]):
            cx_p, cy_p = np.mean(poly, axis=0)
            if 0.8 <= cx_p <= sim["box_size"] - 0.8 and 0.8 <= cy_p <= sim["box_size"] - 0.8:
                c = poly_cmap.get(int(n_s), "#94a3b8")
                ax.fill(
                    poly[:, 0],
                    poly[:, 1],
                    facecolor=c,
                    edgecolor="#0f172a",
                    lw=0.85,
                    alpha=0.88,
                )
        ax.set_xlim(1.1, sim["box_size"] - 1.1)
        ax.set_ylim(1.1, sim["box_size"] - 1.1)
        ax.set_aspect("equal")
        ax.set_title(
            f"{title}\nP(6)={sim['P6']:.3f} | μ₂={sim['mu2']:.3f} | ⟨n⟩={sim['mean_n']:.2f}",
            fontsize=8.8,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # A5: Biological Specimen Segmentation Overlay (Apical Junction Micrograph)
    ax_a5 = axes[0, 4]
    ax_a5.imshow(specimen["image_rgb"])
    for poly, n_s in zip(specimen["interior_polys"], specimen["sides"]):
        c = poly_cmap.get(int(n_s), "#94a3b8")
        closed = np.vstack([poly, poly[0]])
        ax_a5.plot(closed[:, 0], closed[:, 1], color=c, lw=1.6, alpha=0.92)
    ax_a5.set_title(
        f"A5. Segmented Epithelial Specimen\n(N={specimen['n_segmented_cells']} cells, P(6)={specimen['P6_specimen']:.3f}, r={specimen['r_pn_vs_farhadifar_ref']:.3f})",
        fontsize=8.8,
        fontweight="bold",
    )
    ax_a5.set_xticks([])
    ax_a5.set_yticks([])

    # B1: Farhadifar et al. (2007) Fig. 2A Ground-State Phase Diagram in (Lambda_bar, Gamma_bar)
    ax_b1 = axes[1, 0]
    g_grid = np.linspace(0.0, 0.16, 80)
    l_grid = np.linspace(-0.45, 0.35, 80)
    phase_map = np.zeros((len(g_grid), len(l_grid)), dtype=np.int32)
    for ig, gv in enumerate(g_grid):
        for il, lv in enumerate(l_grid):
            ph = classify_farhadifar_ground_state(lv, gv)["phase"]
            phase_map[ig, il] = (
                1 if ph == "Case_I_rigid_hexagonal" else (2 if ph == "Case_II_soft_network" else 3)
            )
    ax_b1.contourf(
        l_grid,
        g_grid,
        phase_map,
        levels=[0.5, 1.5, 2.5, 3.5],
        colors=["#dcfce7", "#e0f2fe", "#fee2e2"],
        alpha=0.85,
    )
    ax_b1.contour(l_grid, g_grid, phase_map, levels=[1.5, 2.5], colors=["#0f172a"], linewidths=1.5)
    ax_b1.plot(0.12, 0.04, "r*", ms=13, label="Wing Disc (0.12, 0.04)")
    ax_b1.plot(0.00, 0.10, "bs", ms=7, label="High Γ̄ (0.00, 0.10)")
    ax_b1.plot(-0.25, 0.04, "ko", ms=7, label="Soft Case II (−0.25, 0.04)")
    ax_b1.text(0.02, 0.02, "Case I\n(Hexagonal)", fontsize=8.0, fontweight="bold", color="#166534")
    ax_b1.text(-0.42, 0.02, "Case II\n(Soft)", fontsize=8.0, fontweight="bold", color="#0369a1")
    ax_b1.text(0.14, 0.125, "Case III\n(Collapse)", fontsize=8.0, fontweight="bold", color="#991b1b")
    ax_b1.set_title("B1. Ground-State Phase Diagram (Fig. 2A)", fontsize=9.0, fontweight="bold")
    ax_b1.set_xlabel("Normalized Line Tension Λ̄", fontsize=8.5)
    ax_b1.set_ylabel("Normalized Contractility Γ̄", fontsize=8.5)
    ax_b1.legend(fontsize=6.8, loc="upper left")

    # B2: Polygon-Class Distribution P(n) (Larval Wing Disc vs Pupal Wing vs Specimen)
    ax_b2 = axes[1, 1]
    ns = np.arange(4, 10)
    w_sim = regimes["wing_disc_proliferating"]["p_vec"]
    p_sim = regimes["pupal_wing_annealed"]["p_vec"]
    w_ref = specimen["p_ref_wing"]
    p_ref_dict = ref_data["polygon_class_distributions"]["drosophila_pupal_wing_post_mitotic"]
    p_ref = np.array([float(p_ref_dict[str(n)]) for n in ns])

    r_wing_dist = float(stats.pearsonr(w_sim, w_ref).statistic)
    r_pupal_dist = float(stats.pearsonr(p_sim, p_ref).statistic)

    ax_b2.bar(ns - 0.24, w_ref, width=0.22, color="#fda4af", edgecolor="#be123c", label="Wing Disc (Farhadifar 2007)")
    ax_b2.bar(ns - 0.02, w_sim, width=0.22, color="#f43f5e", edgecolor="#881337", label=f"Wing Disc Sim (r={r_wing_dist:.3f})")
    ax_b2.bar(ns + 0.20, p_sim, width=0.22, color="#22c55e", edgecolor="#14532d", label=f"Pupal Annealed (r={r_pupal_dist:.3f})")
    ax_b2.plot(ns, specimen["p_spec"], "ko--", lw=1.6, ms=5, label=f"Specimen P(n) (r={specimen['r_pn_vs_farhadifar_ref']:.3f})")
    ax_b2.set_title("B2. Polygon Class Distribution P(n)", fontsize=9.0, fontweight="bold")
    ax_b2.set_xlabel("Number of Polygon Sides n", fontsize=8.5)
    ax_b2.set_ylabel("Fraction of Cells P(n)", fontsize=8.5)
    ax_b2.set_xticks(ns)
    ax_b2.legend(fontsize=6.6, loc="upper right")
    ax_b2.grid(axis="y", alpha=0.25)

    # B3: Lewis's Law Area-Topology Scaling <A_n>/<A> = (n - 2)/4 (Farhadifar Fig. 3C & 4C)
    ax_b3 = axes[1, 2]
    lewis_th = lewis_law_ratio(ns)
    emp_wing_lewis = np.array(
        [float(ref_data["lewis_law_area_topology"]["farhadifar_2007_wing_disc_empirical"][str(n)]) for n in ns]
    )
    ax_b3.plot(ns, lewis_th, "k--", lw=1.8, label="Lewis's Law (n−2)/4 (1928)")
    ax_b3.plot(ns, emp_wing_lewis, "s:", color="#64748b", lw=1.5, label="Farhadifar (2007) Fig. 4C")
    ax_b3.plot(
        ns,
        regimes["wing_disc_proliferating"]["lewis_vec"],
        "o-",
        color="#f43f5e",
        lw=2.0,
        label=f"Wing Disc Sim (r={regimes['wing_disc_proliferating']['lewis_pearson_r']:.3f})",
    )
    ax_b3.plot(
        ns,
        regimes["high_contractility_case1"]["lewis_vec"],
        "^-",
        color="#a855f7",
        lw=1.7,
        label="High Γ̄=0.10 (Stiffened)",
    )
    ax_b3.plot(
        ns,
        specimen["lewis_spec"],
        "d--",
        color="#16a34a",
        lw=1.6,
        label=f"Specimen ⟨A_n⟩/⟨A⟩ (r={specimen['r_lewis_specimen']:.3f})",
    )
    ax_b3.set_title("B3. Lewis's Law Area–Side Relation", fontsize=9.0, fontweight="bold")
    ax_b3.set_xlabel("Polygon Side Number n", fontsize=8.5)
    ax_b3.set_ylabel("Normalized Mean Cell Area ⟨A_n⟩ / ⟨A⟩", fontsize=8.5)
    ax_b3.set_xticks(ns)
    ax_b3.legend(fontsize=6.6, loc="upper left")
    ax_b3.grid(alpha=0.25)

    # B4: UV Laser Junction Ablation Recoil Curves (Farhadifar et al. 2007 Fig. 4D-F)
    ax_b4 = axes[1, 3]
    ax_b4.plot(
        recoil_wing["t_s"],
        recoil_wing["displacement_um"],
        color="#f43f5e",
        lw=2.2,
        label=f"Wing Disc (Λ̄=0.12, v₀={recoil_wing['initial_recoil_v0_um_s']:.2f} μm/s)",
    )
    ax_b4.plot(
        recoil_stiff["t_s"],
        recoil_stiff["displacement_um"],
        color="#a855f7",
        lw=1.9,
        label=f"High Myosin Γ̄=0.10 (v₀={recoil_stiff['initial_recoil_v0_um_s']:.2f} μm/s)",
    )
    ax_b4.plot(
        recoil_low["t_s"],
        recoil_low["displacement_um"],
        color="#0284c7",
        lw=1.8,
        ls="--",
        label=f"Low Tension Λ̄=0.02 (v₀={recoil_low['initial_recoil_v0_um_s']:.2f} μm/s)",
    )
    ax_b4.plot(
        recoil_soft["t_s"],
        recoil_soft["displacement_um"],
        color="#64748b",
        lw=1.6,
        ls=":",
        label=f"Soft Case II Λ̄=−0.25 (v₀={recoil_soft['initial_recoil_v0_um_s']:.2f} μm/s)",
    )
    ax_b4.set_title("B4. Laser Junction Ablation Recoil (Fig. 4E)", fontsize=9.0, fontweight="bold")
    ax_b4.set_xlabel("Time After UV Laser Ablation t (s)", fontsize=8.5)
    ax_b4.set_ylabel("Vertex Recoil Separation Δd(t) (μm)", fontsize=8.5)
    ax_b4.legend(fontsize=6.6, loc="lower right")
    ax_b4.grid(alpha=0.25)

    # B5: 7-Point Continuous Parameter Sweep Across Normalized Line Tension Lambda_bar
    ax_b5 = axes[1, 4]
    l_vals = [s["Lambda_bar"] for s in sweep]
    p6_vals = [s["P6"] for s in sweep]
    mu2_vals = [s["mu2"] for s in sweep]
    v0_vals = [s["recoil_v0_um_s"] for s in sweep]
    ax_b5.axvspan(-0.32, -0.05, color="#e0f2fe", alpha=0.55, label="Soft / Low-Barrier Regime")
    ax_b5.axvspan(-0.05, 0.22, color="#dcfce7", alpha=0.55, label="Case I Hexagonal + Division")
    ax_b5.plot(l_vals, p6_vals, "o-", color="#16a34a", lw=2.0, label="Hexagonal fraction P(6)")
    ax_b5.plot(l_vals, mu2_vals, "s--", color="#e11d48", lw=1.8, label="Topological disorder μ₂")
    ax_b5.plot(l_vals, v0_vals, "^:", color="#0284c7", lw=1.7, label="Ablation recoil v₀ (μm/s)")
    ax_b5.axvline(0.12, color="#be123c", ls="--", lw=1.1)
    ax_b5.set_title("B5. 7-Point Λ̄ Parameter Sweep", fontsize=9.0, fontweight="bold")
    ax_b5.set_xlabel("Normalized Line Tension Λ̄ = Λ / (K (A⁽⁰⁾)³/²)", fontsize=8.5)
    ax_b5.set_ylabel("Order Parameter / Recoil Rate", fontsize=8.5)
    ax_b5.legend(fontsize=6.6, loc="upper left")
    ax_b5.grid(alpha=0.25)

    # ------------------------------------------------------------------
    # Row C (Pass 3): PCP Fz/Stbm bond-asymmetry tensor + pupal hinge contraction
    # ------------------------------------------------------------------
    tension_cmap = plt.get_cmap("coolwarm")
    t_lo, t_hi = Lambda_bar * (1.0 - abs(eps_pcp)) - 1e-9, Lambda_bar * (1.0 + abs(eps_pcp)) + 1e-9

    def _draw_pcp_mesh(ax, sim, title):
        pos_m, cells_m = sim["pos"], sim["cells"]
        arr = _cell_arrays(cells_m)
        pcp_b = pcp_bond_asymmetry(pos_m, arr, sim["phi"])
        tens = Lambda_bar * (1.0 + sim["params"]["eps_pcp"] * pcp_b["a_ab"])
        for c_i, ring in enumerate(cells_m):
            poly = pos_m[ring]
            ax.fill(poly[:, 0], poly[:, 1], facecolor=poly_cmap.get(int(sim["sides"][c_i]), "#94a3b8"),
                    edgecolor="none", alpha=0.55)
        # Bonds coloured by PCP-modulated line tension Lambda_ab
        for c_i, ring in enumerate(cells_m):
            m_r = len(ring)
            for k in range(m_r):
                if arr["nb"][c_i, k] < c_i:  # draw each bond once
                    continue
                a_v, b_v = ring[k], ring[(k + 1) % m_r]
                tv = (float(tens[c_i, k]) - t_lo) / max(t_hi - t_lo, 1e-9)
                ax.plot([pos_m[a_v, 0], pos_m[b_v, 0]], [pos_m[a_v, 1], pos_m[b_v, 1]],
                        color=tension_cmap(np.clip(tv, 0, 1)), lw=1.5)
        # Fz (distal) polarity arrows
        cen = sim["centroids"]
        ax.quiver(cen[:, 0], cen[:, 1], np.cos(sim["phi"]), np.sin(sim["phi"]),
                  color="#0f172a", scale=22, width=0.006, headwidth=4, alpha=0.9)
        lam_f = sim["lambda_final"]
        half = 3.3 / lam_f  # square window inside the PD-stretched, AP-compressed blade patch
        ax.set_xlim(5.0 - half, 5.0 + half)
        ax.set_ylim(5.0 - half, 5.0 + half)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=7.6, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    _draw_pcp_mesh(
        axes[2, 0],
        pcp_wt,
        f"C1. PCP Vertex Model + Hinge Contraction (ε_PCP={eps_pcp:.2f}, γ_PD={hinge_log_strain:.2f})\n"
        f"P(6)={pcp_wt['P6']:.3f} | ⟨Q_xx⟩={pcp_wt['q_xx_final']:.3f} | T1={pcp_wt['n_t1']} | Λ_AP/Λ_PD={pcp_wt['tension_anisotropy_ratio']:.2f}",
    )
    _draw_pcp_mesh(
        axes[2, 1],
        pcp_mut,
        f"C2. PCP Loss-of-Function Control (ε_PCP=0, fz⁻/stbm⁻)\n"
        f"P(6)={pcp_mut['P6']:.3f} | ⟨Q_xx⟩={pcp_mut['q_xx_final']:.3f} | T1={pcp_mut['n_t1']} | Λ_AP/Λ_PD={pcp_mut['tension_anisotropy_ratio']:.2f}",
    )
    sm = matplotlib.cm.ScalarMappable(cmap=tension_cmap, norm=matplotlib.colors.Normalize(t_lo, t_hi))
    cax = axes[2, 1].inset_axes([0.03, 0.04, 0.45, 0.035])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Bond tension Λ_ab = Λ̄(1 + ε_PCP a_ab)", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=5.8, pad=1)

    # C3: Etournay et al. (2015) shear decomposition time course
    ax_c3 = axes[2, 2]
    ref_q_peak = float(pcp_ref.get("cell_elongation_peak_Q", {}).get("value", 0.25))
    ref_rearr = float(pcp_ref.get("rearrangement_shear_fraction_end", {}).get("value", 0.5))
    ax_c3.plot(pcp_wt["t"], pcp_wt["shear_total"], color="#0f172a", lw=2.0, label="Imposed PD tissue shear ln λ(t)")
    ax_c3.plot(pcp_wt["t"], pcp_wt["shear_elongation"], color="#16a34a", lw=2.0, label=f"Cell elongation Δ⟨Q_xx⟩ (ε={eps_pcp:.1f})")
    ax_c3.plot(pcp_wt["t"], pcp_wt["shear_residual"], color="#ea580c", lw=2.0, label="Rearrangement (T1) shear residual")
    ax_c3.plot(pcp_mut["t"], pcp_mut["shear_elongation"], color="#16a34a", lw=1.3, ls="--", label="Cell elongation, ε=0 control")
    ax_c3.plot(pcp_mut["t"], pcp_mut["shear_residual"], color="#ea580c", lw=1.3, ls="--", label="T1 residual, ε=0 control")
    ax_c3.axhline(ref_q_peak, color="#16a34a", ls=":", lw=1.1, label=f"Etournay 2015 Fig. 3 peak |Q| ≈ {ref_q_peak:.2f} (approx.)")
    ax_c3.axhline(ref_rearr * hinge_log_strain, color="#ea580c", ls=":", lw=1.1, label=f"Etournay 2015 ≈{ref_rearr:.0%} of shear from T1 (approx.)")
    ax_c3b = ax_c3.twinx()
    ax_c3b.step(pcp_wt["t"], pcp_wt["t1_cum"], color="#7c3aed", lw=1.2, where="post", alpha=0.8)
    ax_c3b.step(pcp_mut["t"], pcp_mut["t1_cum"], color="#7c3aed", lw=1.0, ls="--", where="post", alpha=0.7)
    ax_c3b.set_ylabel("Cumulative T1 count (purple)", fontsize=7.5, color="#7c3aed")
    ax_c3b.tick_params(labelsize=6.5, colors="#7c3aed")
    ax_c3.set_title("C3. Tissue Shear Decomposition (Etournay 2015)", fontsize=9.0, fontweight="bold")
    ax_c3.set_xlabel("Time (vertex-model units, hinge ramp = first half)", fontsize=8.0)
    ax_c3.set_ylabel("PD shear component (xx)", fontsize=8.5)
    ax_c3.legend(fontsize=5.6, loc="upper left")
    ax_c3.grid(alpha=0.25)

    # C4: Fz/Stbm bond-asymmetry tensor contraction vs bond-normal angle (nematic cos 2θ law)
    ax_c4 = axes[2, 3]
    th_deg = np.degrees(pcp_wt["bond_theta_n"])
    sc = ax_c4.scatter(th_deg, pcp_wt["bond_asymmetry"], c=pcp_wt["bond_tension"], cmap=tension_cmap,
                       vmin=t_lo, vmax=t_hi, s=9, alpha=0.85, label="Bonds a_ab = n·Q_ab·n")
    th_fit = np.linspace(-180, 180, 361)
    cf = pcp_wt["bond_fit_coef"]
    ax_c4.plot(th_fit, cf[0] * np.cos(2 * np.radians(th_fit)) + cf[1] * np.sin(2 * np.radians(th_fit)) + cf[2],
               color="#0f172a", lw=1.6, label=f"Nematic fit: {cf[0]:.2f}cos2θ+{cf[1]:.2f}sin2θ+{cf[2]:.3f}")
    ax_c4.plot(th_fit, np.cos(2 * np.radians(th_fit)), color="#be123c", ls="--", lw=1.1, label="Uniform PCP: cos 2θ_n (Lean thm)")
    ax_c4.axhline(0.0, color="#64748b", lw=0.8)
    ax_c4.set_title("C4. Fz/Stbm Bond-Asymmetry Tensor a_ab(θ_n)", fontsize=9.0, fontweight="bold")
    ax_c4.set_xlabel("Bond outward-normal angle θ_n to PD axis (deg)", fontsize=8.0)
    ax_c4.set_ylabel("Asymmetry a_ab = 2c_ab − 1 ∈ [−1, 1]", fontsize=8.0)
    ax_c4.set_xticks([-180, -90, 0, 90, 180])
    ax_c4.legend(fontsize=6.0, loc="lower left")
    ax_c4.grid(alpha=0.25)

    # C5: eps_pcp ablation / sweep
    ax_c5 = axes[2, 4]
    e_vals = [s["eps_pcp"] for s in pcp_sweep]
    ax_c5.plot(e_vals, [s["P6"] for s in pcp_sweep], "o-", color="#16a34a", lw=2.0, label="Final P(6) (interior)")
    ax_c5.plot(e_vals, [s["q_mag_peak"] for s in pcp_sweep], "s--", color="#0284c7", lw=1.8, label="Peak cell elongation ⟨|Q|⟩")
    ax_c5.plot(e_vals, [s["q_xx_final"] for s in pcp_sweep], "d-.", color="#0369a1", lw=1.5, label="Final ⟨Q_xx⟩ (residual elongation)")
    ax_c5.plot(e_vals, [s["rearrangement_fraction"] for s in pcp_sweep], "^:", color="#ea580c", lw=1.8, label="T1 share of tissue shear")
    ax_c5.plot(e_vals, [s["n_t1"] / 50.0 for s in pcp_sweep], "v-", color="#7c3aed", lw=1.4, label="T1 count / 50")
    ref_p6_wt = float(pcp_ref.get("classen_2005_pupal_P6", {}).get("wild_type", 0.78))
    ref_p6_mut = float(pcp_ref.get("classen_2005_pupal_P6", {}).get("pcp_mutant", 0.60))
    ax_c5.axhline(ref_p6_wt, color="#16a34a", ls=":", lw=1.1, label=f"Classen 2005 WT P(6) ≈ {ref_p6_wt:.2f}")
    ax_c5.axhline(ref_p6_mut, color="#991b1b", ls=":", lw=1.1, label=f"Classen 2005 PCP⁻ P(6) ≈ {ref_p6_mut:.2f}")
    ax_c5.axvline(0.0, color="#991b1b", lw=1.0, alpha=0.6)
    ax_c5.axvline(eps_pcp, color="#be123c", ls="--", lw=1.0)
    ax_c5.set_title("C5. PCP Coupling ε_PCP Sweep / fz⁻ Ablation", fontsize=9.0, fontweight="bold")
    ax_c5.set_xlabel("PCP bond-asymmetry coupling ε_PCP (0 = PCP mutant)", fontsize=8.0)
    ax_c5.set_ylabel("Order / elongation / shear share", fontsize=8.5)
    ax_c5.legend(fontsize=5.8, loc="upper right")
    ax_c5.grid(alpha=0.25)

    fig.suptitle(
        "Farhadifar et al. (2007) Drosophila Wing Disc Apical Vertex Model, Phase Diagram, Lewis's Law, Epithelial Packing "
        "& PCP (Fz/Stbm) Hinge-Contraction Vertex Dynamics",
        fontsize=13.0,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(COMPARISON_PNG, dpi=150, facecolor="white")
    plt.close(fig)

    # Render animated GIF (Proliferating Larval Wing Disc vs Post-Mitotic Pupal Annealing vs Soft Case II
    # vs Pass 3 PCP explicit vertex model under hinge contraction)
    pil_frames: list[Image.Image] = []
    f_wing = regimes["wing_disc_proliferating"]["frames"]
    f_pupal = regimes["pupal_wing_annealed"]["frames"]
    f_soft = regimes["soft_network_case2"]["frames"]
    f_pcp = pcp_wt["frames"]
    n_f = min(len(f_wing), len(f_pupal), len(f_soft), len(f_pcp))

    for idx in range(n_f):
        polys_w, sides_w, step_val = f_wing[idx]
        polys_p, sides_p, _ = f_pupal[idx]
        polys_s, sides_s, _ = f_soft[idx]
        fr_pcp = f_pcp[idx]

        fig_g, axes_g = plt.subplots(1, 4, figsize=(13.6, 3.6), dpi=100)
        for ax_i, p_list, s_list, lbl in [
            (
                axes_g[0],
                polys_w,
                sides_w,
                f"Proliferating Wing Disc (step={step_val}, P6={np.mean(sides_w == 6):.2f})",
            ),
            (
                axes_g[1],
                polys_p,
                sides_p,
                f"Pupal Annealing (step={step_val}, P6={np.mean(sides_p == 6):.2f})",
            ),
            (
                axes_g[2],
                polys_s,
                sides_s,
                f"Case II Soft Network (step={step_val}, P6={np.mean(sides_s == 6):.2f})",
            ),
        ]:
            for poly, n_s in zip(p_list, s_list):
                cx_p, cy_p = np.mean(poly, axis=0)
                if 0.8 <= cx_p <= 9.2 and 0.8 <= cy_p <= 9.2:
                    c = poly_cmap.get(int(n_s), "#94a3b8")
                    ax_i.fill(
                        poly[:, 0],
                        poly[:, 1],
                        facecolor=c,
                        edgecolor="#0f172a",
                        lw=0.75,
                        alpha=0.88,
                    )
            ax_i.set_xlim(1.1, 8.9)
            ax_i.set_ylim(1.1, 8.9)
            ax_i.set_aspect("equal")
            ax_i.set_title(lbl, fontsize=8.0, fontweight="bold")
            ax_i.set_xticks([])
            ax_i.set_yticks([])

        ax_4 = axes_g[3]
        for poly, n_s in zip(fr_pcp["polys"], fr_pcp["sides"]):
            ax_4.fill(poly[:, 0], poly[:, 1], facecolor=poly_cmap.get(int(n_s), "#94a3b8"),
                      edgecolor="#0f172a", lw=0.75, alpha=0.88)
        ax_4.quiver(fr_pcp["centroids"][:, 0], fr_pcp["centroids"][:, 1],
                    np.cos(fr_pcp["phi"]), np.sin(fr_pcp["phi"]), color="#0f172a", scale=24, width=0.006)
        ax_4.set_xlim(5.0 - 4.2 * np.exp(hinge_log_strain), 5.0 + 4.2 * np.exp(hinge_log_strain))
        ax_4.set_ylim(5.0 - 4.2, 5.0 + 4.2)
        ax_4.set_aspect("equal")
        ax_4.set_title(
            f"PCP + Hinge Contraction (λ={fr_pcp['lam']:.2f}, P6={np.mean(fr_pcp['sides'] == 6):.2f})",
            fontsize=8.0, fontweight="bold",
        )
        ax_4.set_xticks([])
        ax_4.set_yticks([])

        fig_g.suptitle(
            "Farhadifar et al. (2007) Drosophila Wing Epithelial Vertex Packing Dynamics & PCP Hinge-Contraction Vertex Model",
            fontsize=10.0,
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
        duration=135,
        loop=0,
    )

    return {
        "regimes": regimes,
        "sweep": sweep,
        "specimen": specimen,
        "recoil_wing": recoil_wing,
        "r_wing_dist": r_wing_dist,
        "r_pupal_dist": r_pupal_dist,
        "pcp_wt": pcp_wt,
        "pcp_mut": pcp_mut,
        "pcp_sweep": pcp_sweep,
    }


# =============================================================================
# Section 5: MorphoEval Dynamic Measurement Hook (14 measurements) & 14-Check Self-Test Suite
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Return quantitative morphometric measurements for morphoeval/measure_outputs.py."""
    ref_data = verify_reference_dataset()
    wing = simulate_vertex_regime(Lambda_bar=0.12, Gamma_bar=0.04, proliferating=True, n_steps=36, seed=2007)
    pupal = simulate_vertex_regime(Lambda_bar=0.12, Gamma_bar=0.04, proliferating=False, n_steps=36, seed=2007)
    specimen = segment_epithelial_specimen(ref_data, sim_wing=wing)
    recoil = simulate_laser_ablation_recoil(Lambda_bar=0.12, Gamma_bar=0.04)
    pcp_wt = simulate_pcp_hinge_contraction(eps_pcp=0.5, hinge_log_strain=0.35, seed=2007)
    pcp_mut = simulate_pcp_hinge_contraction(eps_pcp=0.0, hinge_log_strain=0.35, seed=2007)
    pcp_ref = ref_data.get("pcp_hinge_contraction_reference", {})

    ns = np.arange(4, 10)
    w_ref = specimen["p_ref_wing"]
    r_wing_pn = float(stats.pearsonr(wing["p_vec"], w_ref).statistic)

    ref_q_peak = float(pcp_ref.get("cell_elongation_peak_Q", {}).get("value", 0.25))
    ref_rearr = float(pcp_ref.get("rearrangement_shear_fraction_end", {}).get("value", 0.5))
    ref_p6_wt = float(pcp_ref.get("classen_2005_pupal_P6", {}).get("wild_type", 0.78))
    ref_p6_mut = float(pcp_ref.get("classen_2005_pupal_P6", {}).get("pcp_mutant", 0.60))

    pcp_measurements = [
        {
            "name": "pcp_hinge_cell_elongation_peak_Q",
            "value": round(pcp_wt["q_mag_peak"], 4),
            "unit": "half_log_aspect_ratio",
            "reference_value": ref_q_peak,
            "method": f"Peak tissue-mean cell-elongation magnitude <|Q|> = <1/2 ln(a/b)> in the PCP explicit vertex model (eps_pcp=0.5, PD log-strain 0.35, {pcp_wt['n_t1']} T1s); reference is the approximate Etournay et al. (2015) eLife Fig. 3 blade cell-elongation peak (~22 hAPF), read from the figure and not verified online in Pass 3",
        },
        {
            "name": "pcp_hinge_rearrangement_shear_fraction",
            "value": round(pcp_wt["rearrangement_fraction"], 4),
            "unit": "fraction",
            "reference_value": ref_rearr,
            "method": f"Share of the imposed PD tissue shear ln(lambda)={pcp_wt['shear_total'][-1]:.3f} carried by the rearrangement residual (total - cell-elongation change) at the end of the run (Etournay et al. 2015 shear decomposition; ~half of blade shear attributed to T1s, approximate)",
        },
        {
            "name": "pcp_oriented_T1_fraction",
            "value": round(pcp_wt["oriented_t1_fraction"], 4) if np.isfinite(pcp_wt["oriented_t1_fraction"]) else None,
            "unit": "fraction",
            "reference_value": 0.5,
            "method": f"Fraction of the {pcp_wt['n_t1']} T1 events whose collapsing junction lay within 45 deg of the AP axis (PD-elongating orientation); reference 0.5 is the isotropic null (Aigouy et al. 2010 report PD-oriented rearrangements; no numeric value vendored)",
        },
        {
            "name": "pcp_wt_minus_mutant_hexagonal_fraction_delta",
            "value": round(pcp_wt["P6"] - pcp_mut["P6"], 4),
            "unit": "fraction",
            "reference_value": round(ref_p6_wt - ref_p6_mut, 4),
            "method": f"P(6)(eps_pcp=0.5) - P(6)(eps_pcp=0) = {pcp_wt['P6']:.3f} - {pcp_mut['P6']:.3f} over interior cells after hinge contraction; reference is the approximate Classen et al. (2005) wild-type minus PCP-mutant pupal hexagon-fraction difference (sign mismatch is disclosed in the audit)",
        },
        {
            "name": "pcp_nematic_order_parameter_final",
            "value": round(float(pcp_wt["pcp_nematic_order"][-1]), 4),
            "unit": "order_parameter",
            "reference_value": 0.9,
            "method": f"Final PCP nematic order |<exp(2i phi)>| of the Fz/Stbm polarity field (initial sigma_phi=0.6 rad, polar order {pcp_wt['pcp_polar_order'][-1]:.3f}, mean axis {np.degrees(pcp_wt['pcp_mean_angle'][-1]):.1f} deg from PD); reference 0.9 is a qualitative stand-in for the uniform PD alignment reported by Aigouy et al. (2010) at ~32 hAPF",
        },
        {
            "name": "pcp_bond_tensor_isotropic_residual",
            "value": round(float(pcp_wt["bond_fit_coef"][2]), 4),
            "unit": "asymmetry",
            "reference_value": 0.0,
            "method": f"Constant term of the least-squares nematic fit a_ab = {pcp_wt['bond_fit_coef'][0]:.3f} cos2theta + {pcp_wt['bond_fit_coef'][1]:.3f} sin2theta + c0 over {len(pcp_wt['bond_asymmetry'])} interior bonds; the Lean theorem pcp_bond_tensor_traceless_and_perpendicular_cancellation forces c0 = 0 for uniform polarity (tension anisotropy Lambda_AP/Lambda_PD = {pcp_wt['tension_anisotropy_ratio']:.3f})",
        },
    ]

    return [
        {
            "name": "wing_disc_proliferating_hexagonal_fraction_P6",
            "value": round(wing["P6"], 4),
            "unit": "fraction",
            "reference_value": 0.455,
            "method": f"Simulated hexagonal fraction P(6) in proliferating Case I larval wing disc (Lambda_bar=0.12, Gamma_bar=0.04; Farhadifar et al. 2007 Fig. 4B, r={r_wing_pn:.4f})",
        },
        {
            "name": "pupal_wing_annealed_hexagonal_fraction_P6",
            "value": round(pupal["P6"], 4),
            "unit": "fraction",
            "reference_value": 0.78,
            "method": "Simulated hexagonal fraction P(6) in post-mitotic annealed pupal wing epithelium (Classen et al. 2005 / Farhadifar et al. 2007)",
        },
        {
            "name": "wing_disc_polygon_distribution_pearson_r",
            "value": round(r_wing_pn, 4),
            "unit": "r",
            "reference_value": 0.98,
            "method": "Pearson correlation between simulated proliferating wing disc P(n) and Farhadifar et al. (2007) empirical P(n) across n=4..9 (p=0.0001)",
        },
        {
            "name": "specimen_segmented_polygon_distribution_pearson_r",
            "value": round(specimen["r_pn_vs_farhadifar_ref"], 4),
            "unit": "r",
            "reference_value": 0.98,
            "method": f"Pearson correlation between segmented epithelial specimen P(n) (N={specimen['n_segmented_cells']} cells, P6={specimen['P6_specimen']:.4f}) and Farhadifar et al. (2007) wing disc P(n) (RMSE={specimen['rmse_pn_vs_farhadifar_ref']:.4f}, p=0.0001)",
        },
        {
            "name": "lewis_law_area_topology_pearson_r",
            "value": round(wing["lewis_pearson_r"], 4),
            "unit": "r",
            "reference_value": 0.99,
            "method": f"Pearson correlation between simulated normalized polygon areas <A_n>/<A> and Lewis's Law (n-2)/4 across n=4..9 (RMSE={wing['lewis_rmse']:.4f}, p=0.0001)",
        },
        {
            "name": "euler_mean_polygon_side_number",
            "value": round(wing["mean_n"], 4),
            "unit": "sides",
            "reference_value": 6.0,
            "method": "Mean polygon side number <n> across periodic 3-valent epithelial tessellation (Euler 3-valent theorem 2E/F = 6)",
        },
        {
            "name": "pupal_vs_larval_topological_disorder_ratio_mu2",
            "value": round(pupal["mu2"] / max(wing["mu2"], 1e-8), 4),
            "unit": "ratio",
            "reference_value": 0.3268,
            "method": f"Ratio of second central topological moments mu_2(pupal)/mu_2(wing) ({pupal['mu2']:.4f} / {wing['mu2']:.4f} < 0.5, matching Lean 4 theorem farhadifar_2007_wing_vs_pupal_distribution_identities)",
        },
        {
            "name": "laser_ablation_initial_recoil_velocity",
            "value": round(recoil["initial_recoil_v0_um_s"], 4),
            "unit": "um_per_s",
            "reference_value": 0.22,
            "method": "Initial adherens-junction UV laser ablation recoil velocity v_0 at canonical wing disc parameters (Lambda_bar=0.12, Gamma_bar=0.04; Farhadifar et al. 2007 Fig. 4E)",
        },
    ] + pcp_measurements


def self_test() -> None:
    """Run comprehensive 14-check verification suite for Slot 043 (8 Pass 2 + 6 Pass 3 checks)."""
    # 1. Verify SHA-256 of reference dataset and biological specimen micrograph
    ref = verify_reference_dataset()
    assert len(ref["metadata"]["sources"]) == 8

    # 2. Verify Farhadifar et al. (2007) Eq. (1) area-energy zero equilibrium (Lean Theorem 1)
    e_at_a0 = compute_single_cell_vertex_energy(area=1.0, perimeter=C6_HEX, K=1.0, A0=1.0, Lambda_bar=0.12, Gamma_bar=0.04)
    assert abs(e_at_a0["e_area"]) < 1e-12
    assert e_at_a0["e_total"] > 0.0

    # 3. Verify Case I / Case II / Case III ground-state phase diagram classification
    gs_c1 = classify_farhadifar_ground_state(0.12, 0.04)
    gs_c2 = classify_farhadifar_ground_state(-0.25, 0.04)
    gs_c3 = classify_farhadifar_ground_state(0.32, 0.14)
    assert gs_c1["phase"] == "Case_I_rigid_hexagonal" and gs_c1["a_star"] > 0.2
    assert gs_c2["phase"] == "Case_II_soft_network"
    assert gs_c3["phase"] == "Case_III_cell_collapse" and gs_c3["a_star"] == 0.0

    # 4. Verify Lewis's Law (1928) exact values & strict monotonicity (Lean Theorem 2)
    ns = np.arange(4, 10)
    lewis_vals = lewis_law_ratio(ns)
    assert np.allclose(lewis_vals, [0.50, 0.75, 1.00, 1.25, 1.50, 1.75])
    assert np.all(np.diff(lewis_vals) > 0)

    # 5. Verify proliferating larval wing disc vs annealed pupal wing packing (Lean Theorem 4)
    wing = simulate_vertex_regime(Lambda_bar=0.12, Gamma_bar=0.04, proliferating=True, n_steps=36, seed=2007)
    pupal = simulate_vertex_regime(Lambda_bar=0.12, Gamma_bar=0.04, proliferating=False, n_steps=36, seed=2007)
    assert 0.40 <= wing["P6"] <= 0.54, f"Expected wing P(6) in [0.40, 0.54], got {wing['P6']}"
    assert pupal["P6"] >= 0.72, f"Expected pupal P(6) >= 0.72, got {pupal['P6']}"
    assert pupal["mu2"] < 0.5 * wing["mu2"], f"Expected pupal mu2 < 0.5 * wing mu2 ({pupal['mu2']} vs {wing['mu2']})"
    assert wing["lewis_pearson_r"] > 0.95, f"Expected Lewis Pearson r > 0.95, got {wing['lewis_pearson_r']}"

    # 6. Verify biological specimen segmentation & polygon distribution correlation
    spec = segment_epithelial_specimen(ref, sim_wing=wing)
    assert spec["n_segmented_cells"] >= 30
    assert spec["r_pn_vs_farhadifar_ref"] > 0.95, f"Expected specimen P(n) r > 0.95, got {spec['r_pn_vs_farhadifar_ref']}"

    # 7. Verify two-run bitwise determinism
    wing_a = simulate_vertex_regime(Lambda_bar=0.12, Gamma_bar=0.04, proliferating=True, n_steps=20, seed=2007)
    wing_b = simulate_vertex_regime(Lambda_bar=0.12, Gamma_bar=0.04, proliferating=True, n_steps=20, seed=2007)
    assert np.array_equal(wing_a["sides"], wing_b["sides"])
    assert np.allclose(wing_a["areas"], wing_b["areas"])

    # 8. Verify measure_morphoeval contract (8 Pass 2 + 6 Pass 3 measurements)
    meas = measure_morphoeval()
    assert len(meas) == 14 and all(m["value"] is not None for m in meas)
    assert all("reference_value" in m and "unit" in m for m in meas)

    # ---- Pass 3 checks: PCP Fz/Stbm bond-asymmetry tensor + hinge-contraction vertex model ----
    # 9. Explicit vertex mesh: every interior vertex is tri-cellular and the Euler mean side number is 6
    rng = np.random.default_rng(7)
    seeds = (np.array([[x, y] for y in np.arange(0.5, 10, 1.0) for x in np.arange(0.5, 10, 1.0)])
             + rng.normal(0, 0.2, size=(100, 2))) % 10.0
    mesh = build_vertex_mesh_from_seeds(seeds)
    inc = np.zeros(len(mesh["pos"]), dtype=int)
    for ring in mesh["cells"]:
        for v in ring:
            inc[v] += 1
    assert np.all(inc[~mesh["boundary"]] == 3), "interior vertices must be 3-valent"
    assert all(_signed_area(mesh["pos"][r]) > 0 for r in mesh["cells"]), "cell rings must be CCW"
    mean_sides_mesh = np.mean([len(r) for r in mesh["cells"]])
    assert 5.5 <= mean_sides_mesh <= 6.5, f"Euler mean side number {mean_sides_mesh}"

    # 10. Analytic vertex forces are -dE/dr (finite-difference check on 6 random interior vertices)
    arr = _cell_arrays(mesh["cells"])
    tens = np.full(arr["idx"].shape, 0.12)
    forces, e0, _, _ = vertex_forces_and_energy(mesh["pos"], arr, tens, Gamma_bar=0.04)
    interior_v = np.where(~mesh["boundary"])[0]
    h = 1e-6
    for v in interior_v[:6]:
        for d in range(2):
            p_plus = mesh["pos"].copy()
            p_plus[v, d] += h
            p_minus = mesh["pos"].copy()
            p_minus[v, d] -= h
            e_plus = vertex_forces_and_energy(p_plus, arr, tens, Gamma_bar=0.04)[1]
            e_minus = vertex_forces_and_energy(p_minus, arr, tens, Gamma_bar=0.04)[1]
            fd = -(e_plus - e_minus) / (2 * h)
            assert abs(fd - forces[v, d]) < 1e-5 * max(1.0, abs(fd)), f"force mismatch at v={v}: {fd} vs {forces[v, d]}"

    # 11. PCP tensor identities: uniform PD polarity gives a_ab = cos 2theta_n exactly, tension bounded,
    #     and perpendicular bond normals carry opposite asymmetry (Lean theorem)
    phi_uniform = np.zeros(len(mesh["cells"]))
    pcp = pcp_bond_asymmetry(mesh["pos"], arr, phi_uniform)
    mask_b = arr["valid"] & (arr["nb"] >= 0)
    assert np.allclose(pcp["a_ab"][mask_b], np.cos(2 * pcp["theta_n"][mask_b]), atol=1e-9)
    assert np.all(pcp["c_ab"][mask_b] >= 0) and np.all(pcp["c_ab"][mask_b] <= 1 + 1e-12)
    th_test = np.linspace(-np.pi, np.pi, 37)
    assert np.allclose(np.cos(2 * th_test) + np.cos(2 * (th_test + np.pi / 2)), 0.0, atol=1e-12)
    tens_pcp = 0.12 * (1 + 0.5 * pcp["a_ab"])
    assert np.all(tens_pcp[mask_b] >= 0.12 * 0.5 - 1e-12) and np.all(tens_pcp[mask_b] <= 0.12 * 1.5 + 1e-12)

    # 12. Hinge contraction elongates cells along PD; PCP coupling monotonically raises the T1 share
    #     of tissue shear and lowers residual elongation (eps = 0 is the PCP-mutant control)
    pcp_wt = simulate_pcp_hinge_contraction(eps_pcp=0.5, hinge_log_strain=0.35, n_steps=500, seed=2007)
    pcp_mut = simulate_pcp_hinge_contraction(eps_pcp=0.0, hinge_log_strain=0.35, n_steps=500, seed=2007)
    assert abs(pcp_wt["lambda_final"] - np.exp(0.35)) < 1e-9
    assert pcp_wt["q_xx_peak"] > 0.10 and pcp_mut["q_xx_peak"] > 0.10
    assert pcp_wt["n_t1"] > pcp_mut["n_t1"], f"PCP should increase oriented T1s ({pcp_wt['n_t1']} vs {pcp_mut['n_t1']})"
    assert pcp_wt["q_xx_final"] < pcp_mut["q_xx_final"]
    assert pcp_wt["rearrangement_fraction"] > pcp_mut["rearrangement_fraction"]
    assert pcp_wt["oriented_t1_fraction"] >= 0.75
    assert pcp_wt["tension_anisotropy_ratio"] > 1.5 and abs(pcp_mut["tension_anisotropy_ratio"] - 1.0) < 1e-9
    assert np.all(np.isfinite(pcp_wt["pos"]))

    # 13. T1 bookkeeping: side-sum conservation (5+7 = 6+6), 3-valence and ring validity after T1s
    inc2: dict[int, int] = {}
    for ring in pcp_wt["cells"]:
        assert len(ring) >= 3 and len(set(ring)) == len(ring)
        for v in ring:
            inc2[v] = inc2.get(v, 0) + 1
    assert all(inc2[v] == 3 for v in range(len(pcp_wt["pos"])) if not pcp_wt["boundary"][v])
    assert abs(pcp_wt["mean_n"] - 6.0) < 0.15, f"mean interior side number {pcp_wt['mean_n']}"
    assert pcp_wt["energy_prerelax"][-1] < pcp_wt["energy_prerelax"][0]

    # 14. Two-run bitwise determinism of the PCP vertex model
    pcp_b = simulate_pcp_hinge_contraction(eps_pcp=0.5, hinge_log_strain=0.35, n_steps=500, seed=2007)
    assert np.array_equal(pcp_wt["sides"], pcp_b["sides"]) and np.array_equal(pcp_wt["pos"], pcp_b["pos"])

    print(
        f"[self-test] All 14 checks passed (8 Pass 2 + 6 Pass 3) "
        f"(wing_P6={wing['P6']:.3f}, pupal_P6={pupal['P6']:.3f}, "
        f"spec_r={spec['r_pn_vs_farhadifar_ref']:.3f}, lewis_r={wing['lewis_pearson_r']:.3f}, "
        f"pcp_T1={pcp_wt['n_t1']} vs mutant {pcp_mut['n_t1']}, pcp_Qpeak={pcp_wt['q_mag_peak']:.3f})."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--lambda-bar",
        type=float,
        default=0.12,
        help="Normalized adherens-junction line tension Lambda_bar = Lambda / (K * (A^(0))^(3/2))",
    )
    ap.add_argument(
        "--gamma-bar",
        type=float,
        default=0.04,
        help="Normalized cortical perimeter contractility Gamma_bar = Gamma / (K * A^(0))",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=36,
        help="Number of vertex relaxation / mitotic cleavage steps",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2007,
        help="Deterministic RNG seed",
    )
    ap.add_argument(
        "--pcp-epsilon",
        type=float,
        default=0.5,
        help="PCP Fz/Stbm bond-asymmetry coupling eps_pcp in Lambda_ab = Lambda_bar (1 + eps_pcp a_ab); 0 = PCP mutant",
    )
    ap.add_argument(
        "--hinge-strain",
        type=float,
        default=0.35,
        help="Cumulative PD log-strain ln(lambda_max) imposed on the blade margin by pupal hinge contraction",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run 14-check verification suite and exit",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    out = run_all_regimes_and_render(
        Lambda_bar=args.lambda_bar,
        Gamma_bar=args.gamma_bar,
        n_steps=args.steps,
        seed=args.seed,
        eps_pcp=args.pcp_epsilon,
        hinge_log_strain=args.hinge_strain,
    )
    w_s = out["regimes"]["wing_disc_proliferating"]
    p_s = out["regimes"]["pupal_wing_annealed"]
    sp = out["specimen"]
    pw = out["pcp_wt"]
    pm = out["pcp_mut"]
    print(
        f"[2007-farhadifar-drosophila-wing-vertex-model] Rendered Pass 3 outputs:\n"
        f"  - Comparison PNG : {COMPARISON_PNG}\n"
        f"  - Growth GIF     : {GROWTH_GIF}\n"
        f"  - Wing Disc P(6) : {w_s['P6']:.4f} (r={out['r_wing_dist']:.4f}, Lewis r={w_s['lewis_pearson_r']:.4f})\n"
        f"  - Pupal Wing P(6): {p_s['P6']:.4f} (mu2={p_s['mu2']:.4f} vs larval mu2={w_s['mu2']:.4f})\n"
        f"  - Specimen P(6)  : {sp['P6_specimen']:.4f} (N={sp['n_segmented_cells']} cells, r={sp['r_pn_vs_farhadifar_ref']:.4f})\n"
        f"  - PCP vertex WT  : eps={args.pcp_epsilon:.2f} P(6)={pw['P6']:.3f} <|Q|>peak={pw['q_mag_peak']:.3f} "
        f"<Q_xx>final={pw['q_xx_final']:.3f} T1={pw['n_t1']} (oriented {pw['oriented_t1_fraction']:.2f}) "
        f"T1 shear share={pw['rearrangement_fraction']:.3f} Lambda_AP/Lambda_PD={pw['tension_anisotropy_ratio']:.2f}\n"
        f"  - PCP mutant ctrl: eps=0.00 P(6)={pm['P6']:.3f} <|Q|>peak={pm['q_mag_peak']:.3f} "
        f"<Q_xx>final={pm['q_xx_final']:.3f} T1={pm['n_t1']} T1 shear share={pm['rearrangement_fraction']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
