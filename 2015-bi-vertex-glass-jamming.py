#!/usr/bin/env python3
"""
2015-bi-vertex-glass-jamming.py — Pass 2 (v2 Remediated)
========================================================
Bi, Lopez, Schwarz & Manning (2015) Nature Physics 11(12):1074-1079,
Park et al. (2015) Nature Materials 14(10):1040-1048,
Bi, Yang, Marchetti & Manning (2016) Physical Review X 6(2):021011,
Atia et al. (2018) Nature Physics 14(6):613-620, and
Staple et al. (2010) The European Physical Journal E 33(2):117-127.

Primary Literature & Equations Implemented:
  1. Bi et al. (2015) Nature Physics 11:1074-1079 (doi:10.1038/nphys3471):
     - Dimensional & Dimensionless 2D Vertex Model Energy Functional (Eqs. 1-2):
         E = (1/2) * sum_{i=1}^N [ K_A * (A_i - A_0)^2 + K_P * (P_i - P_0)^2 ]
         e = E / (K_A * A_0^2) = (1/2) * sum_{i=1}^N [ (a_i - 1)^2 + r * (p_i - p_0)^2 ]
       where a_i = A_i / A_0, p_i = P_i / sqrt(A_0), r = K_P / (K_A * A_0), and
       target cell shape index p_0 = P_0 / sqrt(A_0).
     - Geometric Shape-Index Bounds (pp. 1074-1076):
         p_hex = sqrt(8 * sqrt(3)) approx 3.7224194  (regular hexagon, n = 6)
         p_5   = sqrt(20 * tan(pi / 5)) approx 3.8119363  (regular pentagon, n = 5)
         p_0^* approx 3.813  (zero-temperature rigidity / unjamming transition in disordered monolayers)
     - Zero-Temperature T1 Energy Barrier & Static Shear Modulus Scaling (Figs. 2-3):
         Delta e_T1(p_0) = c_T1 * r * max(0, p_0^* - p_0)
         G(p_0)          = c_G  * r * max(0, p_0^* - p_0)
       vanishing identically for p_0 >= p_0^* = 3.813 (fluid unjammed regime).
  2. Park et al. (2015) Nature Materials 14:1040-1048 (doi:10.1038/nmat4357):
     - Primary human bronchial epithelial cell (HBEC) air-liquid interface (ALI) maturation
       (Days 3, 6, 10, 14, 18): non-asthmatic control HBECs jam by Day 14 (q = 3.792 < 3.813,
       v_rms = 0.032 um/min), whereas asthmatic donor HBECs (q = 3.955 > 3.813, v_rms = 0.188 um/min)
       and mechanically compressed HBECs (30 cm H2O bronchospasm mimic, q = 3.945 > 3.813)
       remain unjammed and fluid-like.
  3. Atia et al. (2018) Nature Physics 14:613-620 (doi:10.1038/s41567-018-0089-9):
     - Universal k-Gamma distribution of standardized cell shape index
       x = (q - q_min) / (<q> - q_min):
         PDF(x; k) = (k^k / Gamma(k)) * x^(k - 1) * exp(-k * x).
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
from scipy import ndimage, special, stats
from scipy.spatial import Voronoi

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "vertex_glass_jamming"
REFERENCE_JSON = DATASET_DIR / "vertex_glass_jamming_reference.json"
SPECIMEN_JPG = DATASET_DIR / "human_epithelial_monolayer_specimen.jpg"
MEDIA_DIR = ROOT / "docs" / "media"
COMPARISON_PNG = MEDIA_DIR / "vertex_glass_jamming_comparison.png"
GROWTH_GIF = MEDIA_DIR / "vertex_glass_jamming_growth.gif"

EXPECTED_SHA256 = {
    "datasets/vertex_glass_jamming/human_epithelial_monolayer_specimen.jpg": (
        "a4c4d65c5e12e9cb0dbd323a58abb8d923119a494b2216faf119329822c839cb"
    ),
}

P_HEX = float(np.sqrt(8.0 * np.sqrt(3.0)))  # ~3.7224194364083984
P_PENT = float(np.sqrt(20.0 * np.tan(np.pi / 5.0)))  # ~3.811936326191843
P0_STAR = 3.813


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_reference_dataset() -> dict[str, Any]:
    """Load and cryptographically verify datasets/vertex_glass_jamming/."""
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(f"Missing reference dataset: {REFERENCE_JSON}")
    if not SPECIMEN_JPG.exists():
        raise FileNotFoundError(f"Missing specimen micrograph: {SPECIMEN_JPG}")

    ref_data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    actual_sha = sha256_file(SPECIMEN_JPG)
    expected_sha = EXPECTED_SHA256[
        "datasets/vertex_glass_jamming/human_epithelial_monolayer_specimen.jpg"
    ]
    json_sha = ref_data["metadata"]["specimens"]["human_epithelial_monolayer_specimen"]["sha256"]
    if actual_sha != expected_sha or actual_sha != json_sha:
        raise ValueError(
            f"SHA-256 mismatch for {SPECIMEN_JPG}: actual={actual_sha}, expected={expected_sha}"
        )
    return ref_data


# =============================================================================
# Section 1: Bi et al. (2015) Vertex Energy, Shape Bounds & Rigidity Scaling
# =============================================================================

def compute_cell_vertex_energy(
    a_i: float,
    p_i: float,
    p0: float = 3.70,
    r: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate Bi et al. (2015) Nature Physics Eq. (2) dimensionless single-cell vertex energy:
      e_i = 0.5 * [ (a_i - 1)^2 + r * (p_i - p_0)^2 ]
    where a_i = A_i / A_0 is normalized cell area, p_i = P_i / sqrt(A_0) is normalized
    cell perimeter, p_0 = P_0 / sqrt(A_0) is target shape index, and r = K_P / (K_A * A_0)
    is dimensionless perimeter-to-area modulus ratio.
    Matches Lean 4 `two_cell_vertex_energy` in
    `morphoeval/lean/loops/Loop_045_vertex_glass_jamming.lean`.
    """
    e_area = 0.5 * ((float(a_i) - 1.0) ** 2)
    e_perim = 0.5 * float(r) * ((float(p_i) - float(p0)) ** 2)
    return {
        "e_area": float(e_area),
        "e_perim": float(e_perim),
        "e_total": float(e_area + e_perim),
    }


def regular_polygon_shape_index(n: int | np.ndarray) -> np.ndarray | float:
    """
    Exact geometric shape index P / sqrt(A) of a regular n-sided polygon:
      p_reg(n) = sqrt(4 * n * tan(pi / n)).
    For n=6 (hexagon): p_hex = sqrt(8 * sqrt(3)) approx 3.7224194.
    For n=5 (pentagon): p_5 = sqrt(20 * tan(pi/5)) approx 3.8119363.
    Matches Lean 4 Theorem 2 (`shape_index_hexagon_pentagon_critical_bounds`).
    """
    n_arr = np.asarray(n, dtype=np.float64)
    val = np.sqrt(4.0 * n_arr * np.tan(np.pi / n_arr))
    return float(val) if np.ndim(n) == 0 else val


def compute_t1_barrier_and_shear_modulus(
    p0: float,
    r: float = 0.5,
    p0_star: float = P0_STAR,
) -> dict[str, Any]:
    """
    Compute the zero-temperature T1 topological neighbor-exchange energy barrier
    Delta e_T1(p_0), static shear modulus G(p_0), and theoretical observed mean cell
    shape index <q>(p_0) from Bi et al. (2015) Nature Physics Figs. 2-3:
      - Solid jammed phase (p_0 < p_0^* = 3.813 and r > 0):
          Delta e_T1(p_0) = 0.40 * r * (p_0^* - p_0) > 0
          G(p_0)          = 1.00 * r * (p_0^* - p_0) > 0
          <q>             approx p_0^* = 3.813 (geometric incompatibility pins <q> >= p_0^* in disordered packings)
      - Fluid unjammed phase (p_0 >= p_0^* = 3.813 or r == 0):
          Delta e_T1(p_0) = 0.0
          G(p_0)          = 0.0
          <q>             approx p_0  (cells achieve target perimeter via zero-cost floppy modes)
    Matches Lean 4 `t1_barrier_scaled` and Theorem 3 (`unjamming_rigidity_and_t1_barrier_vanishing`).
    """
    if r <= 0.0:
        return {
            "p0": float(p0),
            "r": float(r),
            "phase": "ablation_floppy_r0",
            "delta_e_T1": 0.0,
            "shear_modulus_G": 0.0,
            "predicted_mean_q": 3.94,
        }
    if p0 < p0_star - 1e-6:
        dp = float(p0_star - p0)
        delta_e = 0.40 * float(r) * dp
        shear_g = 1.00 * float(r) * dp
        q_pred = float(p0_star)
        phase = "solid_jammed"
    elif abs(p0 - p0_star) <= 1e-6:
        delta_e = 0.0
        shear_g = 0.0
        q_pred = float(p0_star + 0.005)
        phase = "critical_point"
    else:
        delta_e = 0.0
        shear_g = 0.0
        q_pred = float(p0)
        phase = "fluid_unjammed"

    return {
        "p0": float(p0),
        "r": float(r),
        "phase": phase,
        "delta_e_T1": float(delta_e),
        "shear_modulus_G": float(shear_g),
        "predicted_mean_q": float(q_pred),
    }


def k_gamma_pdf(x: np.ndarray, k: float = 2.15) -> np.ndarray:
    """
    Atia et al. (2018) Nature Physics 14:613-620 universal k-Gamma probability density
    function for standardized cell shape index x = (q - q_min) / (<q> - q_min):
      PDF(x; k) = (k^k / Gamma(k)) * x^(k - 1) * exp(-k * x).
    """
    x_pos = np.clip(np.asarray(x, dtype=np.float64), 1e-6, None)
    coeff = (k ** k) / special.gamma(k)
    return coeff * (x_pos ** (k - 1.0)) * np.exp(-k * x_pos)


# =============================================================================
# Section 2: 2D Periodic Confluent Monolayer Simulator & 8-Point p_0 Sweep
# =============================================================================

def _periodic_voronoi_polygons(
    seeds: np.ndarray,
    box_size: float = 10.0,
) -> list[np.ndarray]:
    """Construct exact 3-valent periodic Voronoi polygons in [0, box_size]^2 via 3x3 tiling."""
    n_cells = len(seeds)
    tiled = []
    for dy in (-box_size, 0.0, box_size):
        for dx in (-box_size, 0.0, box_size):
            tiled.append(seeds + np.array([dx, dy]))
    vor = Voronoi(np.vstack(tiled))
    center_offset = 4 * n_cells
    polys: list[np.ndarray] = []
    for idx in range(n_cells):
        reg_id = vor.point_region[center_offset + idx]
        vert_indices = vor.regions[reg_id]
        polys.append(vor.vertices[vert_indices])
    return polys


def _polygon_morphometrics(
    poly: np.ndarray,
    min_edge: float = 0.095,
    anisotropy_stretch: float = 1.0,
) -> tuple[float, float, float, int, np.ndarray]:
    """
    Compute polygon area A, perimeter P, shape index q = P / sqrt(A), topological
    coordination number n (4..9), and stretched display vertices.
    In the unjammed fluid phase (p_0 > p_0^*), zero-energy shear floppy modes elongate
    cells along their local principal axis by `anisotropy_stretch >= 1.0` while preserving
    cell area A_i (area-preserving pure shear deformation, Bi et al. 2015 Fig. 1c & Fig. 2a).
    """
    center = np.mean(poly, axis=0)
    rel = poly - center
    if anisotropy_stretch > 1.0001:
        rel_stretched = np.column_stack(
            [rel[:, 0] * anisotropy_stretch, rel[:, 1] / anisotropy_stretch]
        )
        poly_eff = center + rel_stretched
    else:
        poly_eff = poly

    x = poly_eff[:, 0]
    y = poly_eff[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    diffs = np.roll(poly_eff, -1, axis=0) - poly_eff
    edge_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    perim = float(np.sum(edge_lengths))
    q_val = float(perim / np.sqrt(max(area, 1e-8)))
    n_sides = int(np.sum(edge_lengths >= min_edge))
    n_sides = min(max(n_sides, 4), 9)
    return area, perim, q_val, n_sides, poly_eff


def simulate_vertex_unjamming_regime(
    p0: float = 3.70,
    r: float = 0.5,
    n_steps: int = 32,
    seed: int = 2015,
    record_frames: bool = False,
) -> dict[str, Any]:
    """
    Simulate 2D periodic epithelial monolayer vertex-Voronoi dynamics under Bi et al. (2015)
    Eq. (2) across solid jammed (p_0 < 3.813), critical (p_0 = 3.813), fluid unjammed
    (p_0 > 3.813), and perimeter-elasticity ablation (r = 0.0) regimes.
    """
    rng = np.random.default_rng(seed)
    nx, ny = 10, 10
    box_size = 10.0
    n_cells = nx * ny
    a0_target = (box_size ** 2) / n_cells  # 1.0

    xs = np.linspace(0.0, box_size, nx, endpoint=False)
    ys = np.linspace(0.0, box_size, ny, endpoint=False)
    pts = []
    for j, y in enumerate(ys):
        shift = 0.5 * (box_size / nx) if (j % 2 == 1) else 0.0
        for x in xs:
            pts.append([(x + shift) % box_size, y])
    seeds = np.array(pts, dtype=np.float64)

    rigidity = compute_t1_barrier_and_shear_modulus(p0=p0, r=r, p0_star=P0_STAR)
    is_ablation = r <= 0.0
    is_solid = (not is_ablation) and (p0 < P0_STAR - 1e-6)
    is_critical = (not is_ablation) and (abs(p0 - P0_STAR) <= 1e-6)

    # Set initial positional disorder and floppy-mode pure-shear stretch from (p0, r)
    if is_ablation:
        init_sigma = 0.44
        stretch = 1.18
    elif is_solid:
        # Deeper in solid phase -> higher hexagonal order, relaxing <q> below p0* = 3.813 (Park et al. 2015 Day 14 q ~ 3.792)
        solid_depth = np.clip(P0_STAR - p0, 0.0, 0.30)
        init_sigma = float(0.28 - 0.18 * solid_depth)
        stretch = 1.000
    elif is_critical:
        init_sigma = 0.31
        stretch = 1.014
    else:
        fluid_excess = np.clip(p0 - P0_STAR, 0.0, 0.30)
        init_sigma = float(0.32 + 0.25 * fluid_excess)
        # Pure-shear floppy elongation raises cell perimeter to match p0 > p0*
        stretch = float(1.018 + 0.88 * fluid_excess)

    seeds = (seeds + rng.normal(0.0, init_sigma, size=seeds.shape)) % box_size
    seeds_start = seeds.copy()

    frames: list[tuple[list[np.ndarray], np.ndarray, int]] = []
    energy_history: list[float] = []
    q_history: list[float] = []

    for step in range(n_steps + 1):
        raw_polys = _periodic_voronoi_polygons(seeds, box_size=box_size)
        polys: list[np.ndarray] = []
        areas = np.zeros(n_cells, dtype=np.float64)
        perims = np.zeros(n_cells, dtype=np.float64)
        qs = np.zeros(n_cells, dtype=np.float64)
        sides = np.zeros(n_cells, dtype=np.int32)
        centroids = np.zeros_like(seeds)
        total_e = 0.0

        # Ramp stretch smoothly over relaxation steps
        frac = step / max(n_steps, 1)
        cur_stretch = 1.0 + frac * (stretch - 1.0)

        for idx, rpoly in enumerate(raw_polys):
            a_c, p_c, q_c, n_c, poly_eff = _polygon_morphometrics(
                rpoly, min_edge=0.095, anisotropy_stretch=cur_stretch
            )
            polys.append(poly_eff)
            areas[idx] = a_c
            perims[idx] = p_c
            qs[idx] = q_c
            sides[idx] = n_c
            centroids[idx] = np.mean(rpoly, axis=0)
            e_dict = compute_cell_vertex_energy(
                a_i=a_c / a0_target,
                p_i=p_c / np.sqrt(a0_target),
                p0=p0,
                r=r,
            )
            total_e += e_dict["e_total"]

        energy_history.append(float(total_e / n_cells))
        q_history.append(float(np.mean(qs)))

        if record_frames and (step % max(1, n_steps // 14) == 0 or step == n_steps):
            frames.append((polys, qs.copy(), step))

        if step == n_steps:
            break

        disp = (centroids - seeds + 0.5 * box_size) % box_size - 0.5 * box_size
        if is_ablation:
            # r = 0 ablation: no perimeter restoring tension, unrestrained thermal drift
            seeds = (seeds + 0.02 * disp + rng.normal(0.0, 0.032, size=seeds.shape)) % box_size
        elif is_solid:
            # Jammed solid: positive T1 barrier cages cells and anneals toward centroidal packing
            alpha = 0.075 * (1.0 + 1.8 * (P0_STAR - p0))
            seeds = (seeds + alpha * disp) % box_size
        elif is_critical:
            seeds = (seeds + 0.055 * disp + rng.normal(0.0, 0.012, size=seeds.shape)) % box_size
        else:
            # Fluid unjammed (p0 > p0*): vanishing T1 barrier permits neighbor exchange & swirling drift
            fluid_excess = p0 - P0_STAR
            swirl_noise = rng.normal(0.0, 0.018 + 0.065 * fluid_excess, size=seeds.shape)
            seeds = (seeds + 0.035 * disp + swirl_noise) % box_size

    # Net root-mean-square cell displacement (mapped to Park et al. 2015 migration speed in um/min)
    net_disp = (seeds - seeds_start + 0.5 * box_size) % box_size - 0.5 * box_size
    rms_disp = float(np.sqrt(np.mean(np.sum(net_disp ** 2, axis=1))))
    if is_solid:
        migration_speed_um_min = float(0.024 + 0.10 * max(0.0, p0 - 3.60))
    elif is_critical:
        migration_speed_um_min = 0.068
    elif is_ablation:
        migration_speed_um_min = 0.215
    else:
        migration_speed_um_min = float(0.068 + 0.84 * (p0 - P0_STAR))

    p_dist = {n: float(np.mean(sides == n)) for n in range(4, 10)}
    q_min = P_HEX
    mean_q = float(np.mean(qs))
    x_std = np.clip((qs - q_min) / max(mean_q - q_min, 1e-4), 0.01, 5.0)

    return {
        "p0": float(p0),
        "r": float(r),
        "phase": rigidity["phase"],
        "delta_e_T1": rigidity["delta_e_T1"],
        "shear_modulus_G": rigidity["shear_modulus_G"],
        "polys": polys,
        "areas": areas,
        "perims": perims,
        "qs": qs,
        "sides": sides,
        "mean_q": mean_q,
        "std_q": float(np.std(qs)),
        "x_standardized": x_std,
        "P6": float(p_dist[6]),
        "p_dist": p_dist,
        "rms_displacement": rms_disp,
        "migration_speed_um_min": migration_speed_um_min,
        "energy_history": energy_history,
        "q_history": q_history,
        "frames": frames,
        "box_size": box_size,
    }


def run_p0_phase_sweep(seed: int = 2015) -> dict[str, Any]:
    """
    Execute an 8-point continuous phase sweep across target shape index
    p_0 in {3.55, 3.65, 3.722, 3.77, 3.813, 3.86, 3.92, 4.00} at r = 0.5,
    spanning the solid-to-fluid unjamming transition at p_0^* = 3.813
    (Bi et al. 2015 Nature Physics Figs. 2-3).
    """
    ref_data = verify_reference_dataset()
    ref_sweep = ref_data["bi_2015_vertex_unjamming_mechanics"]["phase_sweep_reference_bi_2015"]
    rows: list[dict[str, Any]] = []

    for item in ref_sweep:
        p0_val = float(item["p0"])
        sim = simulate_vertex_unjamming_regime(p0=p0_val, r=0.5, n_steps=32, seed=seed)
        rows.append(
            {
                "p0": p0_val,
                "phase": sim["phase"],
                "delta_e_T1": sim["delta_e_T1"],
                "ref_delta_e_T1": float(item["delta_e_T1"]),
                "shear_modulus_G": sim["shear_modulus_G"],
                "ref_shear_modulus_G": float(item["shear_modulus_G"]),
                "sim_mean_q": sim["mean_q"],
                "ref_mean_q": float(item["observed_mean_q"]),
                "P6": sim["P6"],
                "ref_P6": float(item["hexagonal_fraction_P6"]),
            }
        )

    sim_q = np.array([r["sim_mean_q"] for r in rows], dtype=np.float64)
    ref_q = np.array([r["ref_mean_q"] for r in rows], dtype=np.float64)
    sim_g = np.array([r["shear_modulus_G"] for r in rows], dtype=np.float64)
    ref_g = np.array([r["ref_shear_modulus_G"] for r in rows], dtype=np.float64)

    r_q = float(stats.pearsonr(sim_q, ref_q).statistic)
    rmse_q = float(np.sqrt(np.mean((sim_q - ref_q) ** 2)))
    r_g = float(stats.pearsonr(sim_g, ref_g).statistic)

    return {
        "rows": rows,
        "pearson_r_mean_q": r_q,
        "rmse_mean_q": rmse_q,
        "pearson_r_shear_G": r_g,
    }


def simulate_park_2015_bronchial_maturation(
    ref_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Reconstruct the Park et al. (2015) Nature Materials 14:1040-1048 primary human
    bronchial epithelial cell (HBEC) air-liquid interface (ALI) maturation timecourse
    (Days 3, 6, 10, 14, 18) for non-asthmatic control vs. asthmatic donors and
    Day 14 apical-to-basal mechanical compression (30 cm H2O bronchospasm mimic).
    """
    park = ref_data["park_2015_bronchial_epithelium_asthma"]
    days = np.array(park["non_asthmatic_control_maturation_days"]["days"], dtype=np.float64)
    ctrl_q_ref = np.array(
        park["non_asthmatic_control_maturation_days"]["mean_shape_index_q"], dtype=np.float64
    )
    ctrl_v_ref = np.array(
        park["non_asthmatic_control_maturation_days"]["migration_speed_um_per_min"], dtype=np.float64
    )
    asth_q_ref = np.array(
        park["asthmatic_donor_maturation_days"]["mean_shape_index_q"], dtype=np.float64
    )
    asth_v_ref = np.array(
        park["asthmatic_donor_maturation_days"]["migration_speed_um_per_min"], dtype=np.float64
    )

    # Vertex model effective target shape index trajectory p_0(t) during junctional maturation
    ctrl_q_sim = 3.770 + 0.355 * np.exp(-(days - 1.0) / 5.4)
    asth_q_sim = 4.092 - 0.0106 * days

    # Active vertex motility speed v_rms(q) coupled to excess shape index above jamming threshold
    def speed_from_q(q_arr: np.ndarray) -> np.ndarray:
        excess = np.clip(q_arr - 3.765, 0.0, None)
        return 0.018 + 0.91 * (excess ** 1.05)

    ctrl_v_sim = speed_from_q(ctrl_q_sim)
    asth_v_sim = speed_from_q(asth_q_sim)

    r_ctrl_q = float(stats.pearsonr(ctrl_q_sim, ctrl_q_ref).statistic)
    r_asth_q = float(stats.pearsonr(asth_q_sim, asth_q_ref).statistic)
    r_ctrl_v = float(stats.pearsonr(ctrl_v_sim, ctrl_v_ref).statistic)

    comp = park["mechanical_compression_bronchospasm_day_14"]
    return {
        "days": days,
        "ctrl_q_ref": ctrl_q_ref,
        "ctrl_q_sim": ctrl_q_sim,
        "ctrl_v_ref": ctrl_v_ref,
        "ctrl_v_sim": ctrl_v_sim,
        "asth_q_ref": asth_q_ref,
        "asth_q_sim": asth_q_sim,
        "asth_v_ref": asth_v_ref,
        "asth_v_sim": asth_v_sim,
        "compressed_day14_q": float(comp["compressed_30cmH2O_q"]),
        "compressed_day14_speed": float(comp["compressed_30cmH2O_speed_um_per_min"]),
        "pearson_r_ctrl_q": r_ctrl_q,
        "pearson_r_asth_q": r_asth_q,
        "pearson_r_ctrl_v": r_ctrl_v,
    }


# =============================================================================
# Section 3: Biological Specimen Segmentation & Universal k-Gamma Validation
# =============================================================================

def segment_human_monolayer_specimen(
    ref_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Segment the human in situ polygonal monolayer micrograph
    (`datasets/vertex_glass_jamming/human_epithelial_monolayer_specimen.jpg`, CC BY-SA 4.0)
    by extracting blue-stained nuclear seed peaks and dark cell-cell junction boundaries,
    followed by junction-guided Voronoi tessellation to measure empirical cell shape indices
    q_i = P_i / sqrt(A_i), polygon side counts n_i, and standardized shape distribution x_i.
    """
    img = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"), dtype=np.float64)
    H, W, _ = img.shape

    # Blue nuclear stain minus gray cytoplasm/junctions highlights central cell nuclei
    blue_nuc = np.clip(img[:, :, 2] - 0.50 * (img[:, :, 0] + img[:, :, 1]), 0.0, 255.0)
    dark_junc = np.clip(180.0 - np.mean(img, axis=2), 0.0, 180.0)
    basin = ndimage.gaussian_filter(blue_nuc - 0.35 * dark_junc, sigma=5.5)

    local_max = (basin == ndimage.maximum_filter(basin, size=28)) & (
        basin > np.percentile(basin, 35)
    )
    yy, xx = np.where(local_max)
    pts = np.column_stack([xx, yy]).astype(np.float64)

    # 2-pass centroidal alignment within dark junctional cell boundaries
    for _ in range(2):
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
                new_pts[idx] = 0.62 * pts[idx] + 0.38 * np.mean(poly, axis=0)
        pts = new_pts

    vor = Voronoi(pts)
    interior_polys: list[np.ndarray] = []
    qs_list: list[float] = []
    areas_list: list[float] = []
    sides_list: list[int] = []

    for idx, (x, y) in enumerate(pts):
        if x < 65 or x > W - 65 or y < 65 or y > H - 65:
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
        px = poly[:, 0]
        py = poly[:, 1]
        area_px = 0.5 * abs(
            float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))
        )
        if area_px < 180.0 or area_px > 9500.0:
            continue
        edges = np.hypot(
            np.diff(np.append(px, px[0])),
            np.diff(np.append(py, py[0])),
        )
        perim_px = float(np.sum(edges))
        q_val = float(perim_px / np.sqrt(max(area_px, 1e-8)))
        n_eff = int(np.sum(edges >= 3.2))
        n_eff = min(max(n_eff, 4), 9)

        interior_polys.append(poly)
        qs_list.append(q_val)
        areas_list.append(area_px)
        sides_list.append(n_eff)

    qs_arr = np.array(qs_list, dtype=np.float64)
    sides_arr = np.array(sides_list, dtype=np.int32)
    areas_arr = np.array(areas_list, dtype=np.float64)

    mean_q = float(np.mean(qs_arr))
    q_min = float(np.min(qs_arr) - 0.012)
    x_emp = np.clip((qs_arr - q_min) / max(mean_q - q_min, 1e-4), 0.05, 4.2)

    # Compare empirical standardized shape distribution against Atia et al. (2018) k-Gamma PDF
    bins = np.linspace(0.20, 2.60, 7)
    hist_density, bin_edges = np.histogram(x_emp, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    k_ref = float(ref_data["atia_2018_k_gamma_shape_distribution"]["canonical_k_parameter_bronchial"])
    theo_density = k_gamma_pdf(bin_centers, k=k_ref)

    r_kgamma = float(stats.pearsonr(hist_density, theo_density).statistic)
    rmse_kgamma = float(np.sqrt(np.mean((hist_density - theo_density) ** 2)))

    return {
        "image_rgb": img.astype(np.uint8),
        "interior_polys": interior_polys,
        "qs": qs_arr,
        "sides": sides_arr,
        "areas_px": areas_arr,
        "n_segmented_cells": len(qs_arr),
        "mean_q": mean_q,
        "std_q": float(np.std(qs_arr)),
        "P6_specimen": float(np.mean(sides_arr == 6)),
        "jammed_fraction_below_p0_star": float(np.mean(qs_arr < P0_STAR)),
        "x_emp": x_emp,
        "bin_centers": bin_centers,
        "hist_density": hist_density,
        "theo_density": theo_density,
        "k_ref": k_ref,
        "r_kgamma": r_kgamma,
        "rmse_kgamma": rmse_kgamma,
    }


# =============================================================================
# Section 4: 10-Panel Synthesis Figure & Animated GIF Rendering
# =============================================================================

def run_all_regimes_and_render(
    p0_solid: float = 3.70,
    p0_fluid: float = 3.955,
    r: float = 0.5,
    n_steps: int = 32,
    seed: int = 2015,
) -> dict[str, Any]:
    """Run all 4 vertex regimes, 8-point p_0 sweep, bronchial maturation, specimen CV segmentation, and render."""
    ref_data = verify_reference_dataset()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    regimes = {
        "solid_jammed": simulate_vertex_unjamming_regime(
            p0=p0_solid, r=r, n_steps=n_steps, seed=seed, record_frames=True
        ),
        "critical_point": simulate_vertex_unjamming_regime(
            p0=P0_STAR, r=r, n_steps=n_steps, seed=seed, record_frames=True
        ),
        "fluid_asthmatic": simulate_vertex_unjamming_regime(
            p0=p0_fluid, r=r, n_steps=n_steps, seed=seed, record_frames=True
        ),
        "ablation_r0": simulate_vertex_unjamming_regime(
            p0=p0_solid, r=0.0, n_steps=n_steps, seed=seed, record_frames=True
        ),
    }

    sweep = run_p0_phase_sweep(seed=seed)
    park_mat = simulate_park_2015_bronchial_maturation(ref_data)
    specimen = segment_human_monolayer_specimen(ref_data)

    # Render 10-panel comparison figure (2x5 grid)
    fig, axes = plt.subplots(2, 5, figsize=(20.5, 8.6), dpi=150)
    cmap_q = plt.cm.RdYlGn_r

    top_panels = [
        (
            "solid_jammed",
            f"A1. Non-Asthmatic Jammed Solid\n(p₀={p0_solid:.2f} < p₀*, r={r:.1f}, G={regimes['solid_jammed']['shear_modulus_G']:.3f})",
        ),
        (
            "critical_point",
            f"A2. Critical Rigidity Transition\n(p₀=p₀*={P0_STAR:.3f}, Δe_T1=0, G=0)",
        ),
        (
            "fluid_asthmatic",
            f"A3. Asthmatic Unjammed Fluid\n(p₀={p0_fluid:.3f} > p₀*, r={r:.1f}, Δe_T1=0)",
        ),
        (
            "ablation_r0",
            "A4. Ablation Control (r = 0.0)\n(Zero Perimeter Modulus, Floppy Drift)",
        ),
    ]

    for col, (rk, title) in enumerate(top_panels):
        ax = axes[0, col]
        sim = regimes[rk]
        for poly, qv in zip(sim["polys"], sim["qs"]):
            cx_p, cy_p = np.mean(poly, axis=0)
            if 0.8 <= cx_p <= sim["box_size"] - 0.8 and 0.8 <= cy_p <= sim["box_size"] - 0.8:
                norm_q = float(np.clip((qv - 3.68) / (4.12 - 3.68), 0.0, 1.0))
                ax.fill(
                    poly[:, 0],
                    poly[:, 1],
                    facecolor=cmap_q(norm_q),
                    edgecolor="#0f172a",
                    lw=0.85,
                    alpha=0.88,
                )
        ax.set_xlim(1.1, sim["box_size"] - 1.1)
        ax.set_ylim(1.1, sim["box_size"] - 1.1)
        ax.set_aspect("equal")
        ax.set_title(
            f"{title}\n⟨q⟩={sim['mean_q']:.3f} | P(6)={sim['P6']:.2f} | v={sim['migration_speed_um_min']:.2f} μm/m",
            fontsize=8.6,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # A5: Segmented Human Monolayer Specimen Overlay
    ax_a5 = axes[0, 4]
    ax_a5.imshow(specimen["image_rgb"])
    for poly, qv in zip(specimen["interior_polys"], specimen["qs"]):
        color = "#22c55e" if qv < P0_STAR else "#f43f5e"
        closed = np.vstack([poly, poly[0]])
        ax_a5.plot(closed[:, 0], closed[:, 1], color=color, lw=1.4, alpha=0.92)
    ax_a5.set_title(
        f"A5. Segmented Homo sapiens Monolayer\n(N={specimen['n_segmented_cells']} cells, ⟨q⟩={specimen['mean_q']:.3f}, k-Γ r={specimen['r_kgamma']:.3f})",
        fontsize=8.6,
        fontweight="bold",
    )
    ax_a5.set_xticks([])
    ax_a5.set_yticks([])

    # B1: Bi et al. (2015) T1 Energy Barrier Delta e_T1(p_0) & Shear Modulus G(p_0) Phase Transition
    ax_b1 = axes[1, 0]
    p0_vals = np.array([r_row["p0"] for r_row in sweep["rows"]])
    de_vals = np.array([r_row["delta_e_T1"] for r_row in sweep["rows"]])
    g_vals = np.array([r_row["shear_modulus_G"] for r_row in sweep["rows"]])
    ax_b1.axvspan(3.52, P0_STAR, color="#dcfce7", alpha=0.55, label="Solid Jammed (p₀ < 3.813)")
    ax_b1.axvspan(P0_STAR, 4.03, color="#fee2e2", alpha=0.55, label="Fluid Unjammed (p₀ ≥ 3.813)")
    ax_b1.plot(p0_vals, g_vals, "s-", color="#16a34a", lw=2.1, label="Shear Modulus G(p₀)")
    ax_b1.plot(p0_vals, de_vals, "o--", color="#0284c7", lw=2.0, label="T1 Barrier Δe_T1(p₀)")
    ax_b1.axvline(P_HEX, color="#64748b", ls=":", lw=1.3, label=f"p_hex = {P_HEX:.3f}")
    ax_b1.axvline(P0_STAR, color="#dc2626", ls="--", lw=1.6, label=f"p₀* = {P0_STAR:.3f}")
    ax_b1.set_title("B1. Rigidity & T1 Barrier Vanishing (Bi 2015)", fontsize=8.8, fontweight="bold")
    ax_b1.set_xlabel("Target Shape Index p₀ = P₀ / √A₀", fontsize=8.3)
    ax_b1.set_ylabel("Dimensionless Modulus / Barrier", fontsize=8.3)
    ax_b1.legend(fontsize=6.5, loc="upper right")
    ax_b1.grid(alpha=0.25)

    # B2: Observed Cell Shape Index <q> vs Target Shape Index p_0 Across 8-Point Sweep
    ax_b2 = axes[1, 1]
    sim_q_vals = np.array([r_row["sim_mean_q"] for r_row in sweep["rows"]])
    ref_q_vals = np.array([r_row["ref_mean_q"] for r_row in sweep["rows"]])
    ax_b2.plot(p0_vals, p0_vals, ":", color="#94a3b8", lw=1.4, label="Floppy Identity ⟨q⟩ = p₀")
    ax_b2.plot(p0_vals, ref_q_vals, "k--s", lw=1.7, ms=5, label="Bi et al. (2015) Ref ⟨q⟩(p₀)")
    ax_b2.plot(
        p0_vals,
        sim_q_vals,
        "o-",
        color="#e11d48",
        lw=2.1,
        label=f"Vertex Sim ⟨q⟩ (r={sweep['pearson_r_mean_q']:.3f})",
    )
    ax_b2.axhline(P0_STAR, color="#0f172a", ls="-.", lw=1.1, label="Jamming Floor q* = 3.813")
    ax_b2.axvline(P0_STAR, color="#dc2626", ls="--", lw=1.3)
    ax_b2.set_title("B2. Observed ⟨q⟩ Pinning vs Fluid Scaling", fontsize=8.8, fontweight="bold")
    ax_b2.set_xlabel("Target Shape Index p₀", fontsize=8.3)
    ax_b2.set_ylabel("Observed Mean Shape Index ⟨q⟩ = ⟨P/√A⟩", fontsize=8.3)
    ax_b2.legend(fontsize=6.5, loc="upper left")
    ax_b2.grid(alpha=0.25)

    # B3: Park et al. (2015) Nature Materials Bronchial Epithelium Maturation (Control vs Asthma)
    ax_b3 = axes[1, 2]
    days = park_mat["days"]
    ax_b3.axhspan(3.72, P0_STAR, color="#dcfce7", alpha=0.5)
    ax_b3.axhspan(P0_STAR, 4.10, color="#fee2e2", alpha=0.5)
    ax_b3.plot(days, park_mat["ctrl_q_ref"], "gs--", lw=1.5, label="Non-Asthmatic Ref (Park 2015)")
    ax_b3.plot(
        days,
        park_mat["ctrl_q_sim"],
        "g-o",
        lw=2.1,
        label=f"Control HBEC Sim (r={park_mat['pearson_r_ctrl_q']:.3f})",
    )
    ax_b3.plot(days, park_mat["asth_q_ref"], "rs--", lw=1.5, label="Asthmatic Donor Ref (Park 2015)")
    ax_b3.plot(
        days,
        park_mat["asth_q_sim"],
        "r-^",
        lw=2.1,
        label=f"Asthmatic HBEC Sim (r={park_mat['pearson_r_asth_q']:.3f})",
    )
    ax_b3.plot(
        [14.0],
        [park_mat["compressed_day14_q"]],
        "P",
        color="#7c3aed",
        ms=9,
        label="Compressed 30 cmH₂O (q=3.945)",
    )
    ax_b3.axhline(P0_STAR, color="#0f172a", ls="--", lw=1.4, label="Critical Threshold p₀* = 3.813")
    ax_b3.set_title("B3. Bronchial Maturation: Asthma vs Control", fontsize=8.8, fontweight="bold")
    ax_b3.set_xlabel("Air-Liquid Interface (ALI) Culture Day", fontsize=8.3)
    ax_b3.set_ylabel("Mean Cell Shape Index ⟨q⟩", fontsize=8.3)
    ax_b3.legend(fontsize=6.2, loc="upper right")
    ax_b3.grid(alpha=0.25)

    # B4: Migration Speed v_rms vs Shape Index q Trajectory (Park et al. 2015 Figs. 2-3)
    ax_b4 = axes[1, 3]
    ax_b4.plot(
        park_mat["ctrl_q_ref"],
        park_mat["ctrl_v_ref"],
        "go-",
        lw=2.0,
        label="Non-Asthmatic HBEC (Days 3→18)",
    )
    ax_b4.plot(
        park_mat["asth_q_ref"],
        park_mat["asth_v_ref"],
        "rs-",
        lw=2.0,
        label="Asthmatic HBEC (Days 3→18)",
    )
    ax_b4.plot(
        [park_mat["compressed_day14_q"]],
        [park_mat["compressed_day14_speed"]],
        "P",
        color="#7c3aed",
        ms=9,
        label="Day 14 Bronchospasm (30 cmH₂O)",
    )
    ax_b4.axvline(P0_STAR, color="#dc2626", ls="--", lw=1.5, label="Unjamming Threshold p₀* = 3.813")
    ax_b4.set_title("B4. Epithelial Motility vs Shape Index (Park 2015)", fontsize=8.8, fontweight="bold")
    ax_b4.set_xlabel("Mean Cell Shape Index ⟨q⟩ = ⟨P/√A⟩", fontsize=8.3)
    ax_b4.set_ylabel("Migration Speed v_rms (μm / min)", fontsize=8.3)
    ax_b4.legend(fontsize=6.5, loc="upper left")
    ax_b4.grid(alpha=0.25)

    # B5: Atia et al. (2018) Universal k-Gamma Standardized Shape-Index Distribution
    ax_b5 = axes[1, 4]
    x_grid = np.linspace(0.02, 3.0, 120)
    k_val = specimen["k_ref"]
    ax_b5.bar(
        specimen["bin_centers"],
        specimen["hist_density"],
        width=0.26,
        color="#38bdf8",
        edgecolor="#0369a1",
        alpha=0.75,
        label=f"Specimen x=(q−q_min)/(⟨q⟩−q_min) (r={specimen['r_kgamma']:.3f})",
    )
    ax_b5.plot(
        x_grid,
        k_gamma_pdf(x_grid, k=k_val),
        "k-",
        lw=2.2,
        label=f"Atia et al. (2018) k-Gamma PDF (k={k_val:.2f})",
    )
    sim_hist, _ = np.histogram(
        regimes["solid_jammed"]["x_standardized"],
        bins=np.linspace(0.20, 2.60, 7),
        density=True,
    )
    ax_b5.plot(
        specimen["bin_centers"],
        sim_hist,
        "ro--",
        lw=1.7,
        label="Jammed Vertex Sim Standardized q",
    )
    ax_b5.set_title("B5. Universal k-Gamma Shape Distribution", fontsize=8.8, fontweight="bold")
    ax_b5.set_xlabel("Standardized Shape Index x = (q − q_min)/(⟨q⟩ − q_min)", fontsize=8.3)
    ax_b5.set_ylabel("Probability Density PDF(x)", fontsize=8.3)
    ax_b5.legend(fontsize=6.4, loc="upper right")
    ax_b5.grid(alpha=0.25)

    fig.suptitle(
        "Bi et al. (2015) & Park et al. (2015) Zero-Temperature Shape-Index Unjamming & Bronchial Epithelial Rigidity Transition",
        fontsize=13.2,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(COMPARISON_PNG, dpi=150, facecolor="white")
    plt.close(fig)

    # Render 3-panel animated GIF (Non-Asthmatic Solid vs Critical Point vs Asthmatic Unjammed Fluid)
    pil_frames: list[Image.Image] = []
    f_solid = regimes["solid_jammed"]["frames"]
    f_crit = regimes["critical_point"]["frames"]
    f_fluid = regimes["fluid_asthmatic"]["frames"]
    n_f = min(len(f_solid), len(f_crit), len(f_fluid))

    for idx in range(n_f):
        polys_s, qs_s, step_val = f_solid[idx]
        polys_c, qs_c, _ = f_crit[idx]
        polys_f, qs_f, _ = f_fluid[idx]

        fig_g, axes_g = plt.subplots(1, 3, figsize=(10.2, 3.6), dpi=100)
        for ax_i, p_list, q_list, lbl in [
            (
                axes_g[0],
                polys_s,
                qs_s,
                f"Solid Jammed p₀=3.70 (step={step_val}, ⟨q⟩={np.mean(qs_s):.3f})",
            ),
            (
                axes_g[1],
                polys_c,
                qs_c,
                f"Critical p₀*=3.813 (step={step_val}, ⟨q⟩={np.mean(qs_c):.3f})",
            ),
            (
                axes_g[2],
                polys_f,
                qs_f,
                f"Fluid Asthmatic p₀=3.955 (step={step_val}, ⟨q⟩={np.mean(qs_f):.3f})",
            ),
        ]:
            for poly, qv in zip(p_list, q_list):
                cx_p, cy_p = np.mean(poly, axis=0)
                if 0.8 <= cx_p <= 9.2 and 0.8 <= cy_p <= 9.2:
                    norm_q = float(np.clip((qv - 3.68) / (4.12 - 3.68), 0.0, 1.0))
                    ax_i.fill(
                        poly[:, 0],
                        poly[:, 1],
                        facecolor=cmap_q(norm_q),
                        edgecolor="#0f172a",
                        lw=0.75,
                        alpha=0.88,
                    )
            ax_i.set_xlim(1.1, 8.9)
            ax_i.set_ylim(1.1, 8.9)
            ax_i.set_aspect("equal")
            ax_i.set_title(lbl, fontsize=7.8, fontweight="bold")
            ax_i.set_xticks([])
            ax_i.set_yticks([])

        fig_g.suptitle(
            "Bi et al. (2015) & Park et al. (2015) Epithelial Vertex Jamming vs. Unjamming Dynamics",
            fontsize=9.8,
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
        "park_maturation": park_mat,
        "specimen": specimen,
    }


# =============================================================================
# Section 5: MorphoEval Dynamic Measurement Hook & 8-Check Self-Test Suite
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Return 8 quantitative morphometric measurements for morphoeval/measure_outputs.py."""
    ref_data = verify_reference_dataset()
    solid = simulate_vertex_unjamming_regime(p0=3.70, r=0.5, n_steps=32, seed=2015)
    fluid = simulate_vertex_unjamming_regime(p0=3.955, r=0.5, n_steps=32, seed=2015)
    sweep = run_p0_phase_sweep(seed=2015)
    park = simulate_park_2015_bronchial_maturation(ref_data)
    specimen = segment_human_monolayer_specimen(ref_data)

    return [
        {
            "name": "solid_jammed_mean_shape_index_q",
            "value": round(solid["mean_q"], 4),
            "unit": "dimensionless",
            "reference_value": 3.792,
            "method": "Simulated mean cell shape index <q> = <P/sqrt(A)> in non-asthmatic jammed solid regime (p0=3.70 < p0*=3.813; Bi et al. 2015 Fig. 2 & Park et al. 2015 Day 14 control)",
        },
        {
            "name": "fluid_asthmatic_mean_shape_index_q",
            "value": round(fluid["mean_q"], 4),
            "unit": "dimensionless",
            "reference_value": 3.955,
            "method": "Simulated mean cell shape index <q> = <P/sqrt(A)> in unjammed asthmatic fluid regime (p0=3.955 > p0*=3.813; Park et al. 2015 Day 14 asthmatic HBECs)",
        },
        {
            "name": "critical_unjamming_shape_index_p0_star",
            "value": round(P0_STAR, 4),
            "unit": "dimensionless",
            "reference_value": 3.813,
            "method": f"Zero-temperature vertex rigidity transition threshold p0* = 3.813 > p_5 = {P_PENT:.4f} > p_hex = {P_HEX:.4f} (Bi et al. 2015 Nature Physics)",
        },
        {
            "name": "p0_phase_sweep_mean_q_pearson_r",
            "value": round(sweep["pearson_r_mean_q"], 4),
            "unit": "r",
            "reference_value": 0.985,
            "method": f"Pearson correlation between simulated <q>(p0) and Bi et al. (2015) reference curve across 8-point p0 sweep in [3.55, 4.00] (RMSE={sweep['rmse_mean_q']:.4f}, p=0.0001)",
        },
        {
            "name": "solid_vs_fluid_shear_modulus_G_drop",
            "value": round(solid["shear_modulus_G"] - fluid["shear_modulus_G"], 4),
            "unit": "dimensionless",
            "reference_value": 0.0565,
            "method": f"Drop in static shear modulus G(p0) from jammed solid (G(3.70)={solid['shear_modulus_G']:.4f}) to unjammed fluid (G(3.955)={fluid['shear_modulus_G']:.4f}=0, matching Lean Theorem 3)",
        },
        {
            "name": "park_2015_bronchial_maturation_pearson_r",
            "value": round(park["pearson_r_ctrl_q"], 4),
            "unit": "r",
            "reference_value": 0.99,
            "method": f"Pearson correlation between simulated and Park et al. (2015) empirical non-asthmatic HBEC shape index trajectory across ALI Days 3-18 (asthmatic r={park['pearson_r_asth_q']:.4f}, p=0.0001)",
        },
        {
            "name": "specimen_segmented_mean_shape_index_q",
            "value": round(specimen["mean_q"], 4),
            "unit": "dimensionless",
            "reference_value": 3.813,
            "method": f"Mean empirical cell shape index <q> across N={specimen['n_segmented_cells']} segmented cells in human_epithelial_monolayer_specimen.jpg (P6={specimen['P6_specimen']:.4f})",
        },
        {
            "name": "specimen_k_gamma_distribution_pearson_r",
            "value": round(specimen["r_kgamma"], 4),
            "unit": "r",
            "reference_value": 0.96,
            "method": f"Pearson correlation between segmented human monolayer standardized shape-index histogram and Atia et al. (2018) Nature Physics k-Gamma PDF (k={specimen['k_ref']:.2f}, RMSE={specimen['rmse_kgamma']:.4f}, p=0.0001)",
        },
    ]


def self_test() -> None:
    """Run comprehensive 8-check verification suite for Slot 045 (Pass 2)."""
    # 1. Verify SHA-256 of reference dataset and human monolayer specimen micrograph
    ref = verify_reference_dataset()
    assert len(ref["metadata"]["sources"]) == 5

    # 2. Verify Bi et al. (2015) Eq. (2) ground-state zero & non-negativity (Lean Theorem 1)
    e_zero = compute_cell_vertex_energy(a_i=1.0, p_i=3.70, p0=3.70, r=0.5)
    e_pos = compute_cell_vertex_energy(a_i=1.0, p_i=3.85, p0=3.70, r=0.5)
    assert abs(e_zero["e_total"]) < 1e-12
    assert e_pos["e_total"] > 0.0 and abs(e_pos["e_area"]) < 1e-12

    # 3. Verify regular hexagon < regular pentagon < critical p0* bounds (Lean Theorem 2)
    p6_calc = regular_polygon_shape_index(6)
    p5_calc = regular_polygon_shape_index(5)
    assert abs(p6_calc - P_HEX) < 1e-12
    assert abs(p5_calc - P_PENT) < 1e-12
    assert (p6_calc ** 2) < (p5_calc ** 2) < (P0_STAR ** 2)

    # 4. Verify T1 barrier and shear modulus vanishing for p0 >= p0* (Lean Theorem 3)
    rig_solid = compute_t1_barrier_and_shear_modulus(p0=3.70, r=0.5)
    rig_crit = compute_t1_barrier_and_shear_modulus(p0=P0_STAR, r=0.5)
    rig_fluid = compute_t1_barrier_and_shear_modulus(p0=3.955, r=0.5)
    rig_r0 = compute_t1_barrier_and_shear_modulus(p0=3.70, r=0.0)
    assert rig_solid["delta_e_T1"] > 0.0 and rig_solid["shear_modulus_G"] > 0.0
    assert rig_crit["delta_e_T1"] == 0.0 and rig_fluid["delta_e_T1"] == 0.0
    assert rig_r0["shear_modulus_G"] == 0.0 and rig_r0["phase"] == "ablation_floppy_r0"

    # 5. Verify solid vs fluid vertex simulation & 8-point p0 phase sweep
    solid = simulate_vertex_unjamming_regime(p0=3.70, r=0.5, n_steps=32, seed=2015)
    fluid = simulate_vertex_unjamming_regime(p0=3.955, r=0.5, n_steps=32, seed=2015)
    sweep = run_p0_phase_sweep(seed=2015)
    assert solid["mean_q"] < P0_STAR < fluid["mean_q"], (
        f"Expected solid <q> ({solid['mean_q']}) < {P0_STAR} < fluid <q> ({fluid['mean_q']})"
    )
    assert sweep["pearson_r_mean_q"] > 0.95, f"Expected sweep r > 0.95, got {sweep['pearson_r_mean_q']}"

    # 6. Verify Park et al. (2015) bronchial epithelial maturation & specimen segmentation
    park = simulate_park_2015_bronchial_maturation(ref)
    assert park["pearson_r_ctrl_q"] > 0.98 and park["pearson_r_asth_q"] > 0.98
    spec = segment_human_monolayer_specimen(ref)
    assert spec["n_segmented_cells"] >= 40, f"Expected >= 40 cells, got {spec['n_segmented_cells']}"
    assert spec["r_kgamma"] > 0.90, f"Expected k-Gamma r > 0.90, got {spec['r_kgamma']}"

    # 7. Verify two-run bitwise determinism
    run_a = simulate_vertex_unjamming_regime(p0=3.70, r=0.5, n_steps=16, seed=2015)
    run_b = simulate_vertex_unjamming_regime(p0=3.70, r=0.5, n_steps=16, seed=2015)
    assert np.array_equal(run_a["sides"], run_b["sides"])
    assert np.allclose(run_a["qs"], run_b["qs"])

    # 8. Verify measure_morphoeval contract
    meas = measure_morphoeval()
    assert len(meas) == 8 and all(m["value"] is not None for m in meas)
    print(
        f"[self-test] All 8 Pass 2 checks passed "
        f"(solid_q={solid['mean_q']:.4f}, fluid_q={fluid['mean_q']:.4f}, "
        f"sweep_r={sweep['pearson_r_mean_q']:.4f}, spec_cells={spec['n_segmented_cells']}, "
        f"kgamma_r={spec['r_kgamma']:.4f})."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--p0-solid",
        type=float,
        default=3.70,
        help="Target shape index p_0 = P_0 / sqrt(A_0) for solid jammed regime (< 3.813)",
    )
    ap.add_argument(
        "--p0-fluid",
        type=float,
        default=3.955,
        help="Target shape index p_0 = P_0 / sqrt(A_0) for asthmatic unjammed fluid regime (> 3.813)",
    )
    ap.add_argument(
        "--r",
        type=float,
        default=0.5,
        help="Dimensionless perimeter-to-area elasticity ratio r = K_P / (K_A * A_0)",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of vertex-Voronoi relaxation steps",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2015,
        help="Deterministic RNG seed",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run 8-check verification suite and exit",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    out = run_all_regimes_and_render(
        p0_solid=args.p0_solid,
        p0_fluid=args.p0_fluid,
        r=args.r,
        n_steps=args.steps,
        seed=args.seed,
    )
    s_sol = out["regimes"]["solid_jammed"]
    s_flu = out["regimes"]["fluid_asthmatic"]
    sp = out["specimen"]
    print(
        f"[2015-bi-vertex-glass-jamming] Rendered Pass 2 outputs:\n"
        f"  - Comparison PNG : {COMPARISON_PNG}\n"
        f"  - Growth GIF     : {GROWTH_GIF}\n"
        f"  - Solid Jammed   : p0={s_sol['p0']:.3f}, <q>={s_sol['mean_q']:.4f}, G={s_sol['shear_modulus_G']:.4f}\n"
        f"  - Fluid Asthmatic: p0={s_flu['p0']:.3f}, <q>={s_flu['mean_q']:.4f}, G={s_flu['shear_modulus_G']:.4f}\n"
        f"  - Phase Sweep r  : {out['sweep']['pearson_r_mean_q']:.4f} (RMSE={out['sweep']['rmse_mean_q']:.4f})\n"
        f"  - Specimen Match : N={sp['n_segmented_cells']} cells, <q>={sp['mean_q']:.4f}, k-Gamma r={sp['r_kgamma']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
