#!/usr/bin/env python3
"""
2015-bi-vertex-glass-jamming.py — Pass 3 (v3 Remediated: SPV active motility coupling)
=======================================================================================
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
  4. Bi, Yang, Marchetti & Manning (2016) Physical Review X 6:021011
     (doi:10.1103/PhysRevX.6.021011) — Self-Propelled Voronoi (SPV) model (Pass 3):
     - Tissue energy (Bi 2016 Eq. 1, K_A = K_P = 1, A_0 = 1, i.e. r = 1, no 1/2 prefactor):
         E = sum_i [ K_A (A_i - A_0)^2 + K_P (P_i - P_0)^2 ]   (cells = Voronoi polygons of r_i)
     - Overdamped self-propelled dynamics (Bi 2016 Eqs. 2-3):
         dr_i/dt     = mu * F_i + v_0 * n_i,     F_i = -grad_i E,   n_i = (cos theta_i, sin theta_i)
         dtheta_i/dt = eta_i(t),                 <eta_i(t) eta_j(t')> = 2 D_r delta_ij delta(t - t')
       Forces are the EXACT analytic gradient of the Voronoi energy (circumcenter Jacobians,
       `spv_voronoi_forces`), verified against central finite differences in `--self-test`.
     - Dynamical order parameter (Bi 2016 Fig. 1): D_eff = D_s / D_0 with D_s = lim MSD / (4 t)
       and D_0 = v_0^2 / (2 D_r) the free active-Brownian diffusivity; Bi's long-time criterion
       D_eff < 1e-3 is a solid.  At the finite window used here (T = 150, N = 100) the solid
       noise floor is ~5e-3, so the phase boundary is extracted from the per-cell T1
       neighbour-exchange rate normalised by the free path, Gamma_T1 / v_0 (T1 events per
       cell per unit free-path length; threshold 0.02), and D_eff is reported alongside.
       Both are finite-time, finite-size estimates (see docstrings).
     - Structural order parameter (Bi 2016 Fig. 4): the observed shape index at the glass
       boundary is q_c ≈ 3.81 for all (v_0, D_r); the zero-motility limit p_0^c(0) = 3.81
       recovers the Bi (2015) rigidity point.  Both are tested in `measure_morphoeval()`.
     Lean 4: Theorems 5-6 (`spv_motility_lowers_critical_shape_index`,
     `spv_free_active_diffusivity_monotone_in_motility`) in
     `morphoeval/lean/loops/Loop_045_vertex_glass_jamming.lean`.

Caveat carried into the v3 audit: `simulate_vertex_unjamming_regime` (Section 2, Panels A1-A4
and the GIF) is a PHENOMENOLOGICAL vertex-Voronoi relaxation whose regime-dependent stretch and
noise amplitudes are set by phase, not derived from the energy; the SPV module (Section 2b) is
the force-based model and its zero-motility FIRE-minimised <q>(p_0) is overlaid in Panel B2.
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
from scipy.spatial import Delaunay, Voronoi

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "vertex_glass_jamming"
REFERENCE_JSON = DATASET_DIR / "vertex_glass_jamming_reference.json"
SPV_REFERENCE_JSON = DATASET_DIR / "bi_2016_spv_motility_reference.json"
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

# Bi et al. (2016) PRX SPV conventions and the finite-time criteria used in Section 2b
SPV_P0_STAR_2016 = 3.81  # zero-motility glass point & structural order parameter q_c (Bi 2016)
SPV_DEFF_THRESHOLD_BI2016 = 1e-3  # Bi 2016 long-time solid criterion D_eff < 1e-3 (not resolvable here)
SPV_T1_PER_PATH_THRESHOLD = 0.02  # finite-time fluid criterion: Gamma_T1 / v_0 > 0.02 T1 / cell / unit free path
SPV_P0_GRID = (3.50, 3.60, 3.70, 3.80, 3.85, 3.90)
SPV_V0_GRID = (0.05, 0.15, 0.30, 0.45)
SPV_DR_GRID = (0.25, 1.0, 4.0)
SPV_DR_SWEEP_V0 = 0.30


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
# Section 2b (Pass 3): Bi et al. (2016) PRX Self-Propelled Voronoi (SPV) Active Motility
# =============================================================================

def load_spv_reference() -> dict[str, Any]:
    """Load the vendored Bi et al. (2016) PRX 6:021011 SPV reference table (Pass 3)."""
    if not SPV_REFERENCE_JSON.exists():
        raise FileNotFoundError(f"Missing SPV reference dataset: {SPV_REFERENCE_JSON}")
    return json.loads(SPV_REFERENCE_JSON.read_text(encoding="utf-8"))


def _circumcenters(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Circumcenters of triangles (a, b, c); each argument is an (T, 2) array."""
    ax, ay = a[:, 0], a[:, 1]
    bx, by = b[:, 0], b[:, 1]
    cx, cy = c[:, 0], c[:, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.column_stack([ux, uy])


def _circumcenter_jacobian(
    h: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """
    Exact Jacobian dh/da of the circumcenter h of triangle (a, b, c) with respect to
    generator a.  Differentiating |h-a|^2 = |h-b|^2 = |h-c|^2 gives
      dh = perp(b - c) * [(h - a) . da] / [(b - a) . perp(b - c)],
    i.e. J[m, n] = perp(b-c)[m] * (h-a)[n] / ((b-a) . perp(b-c)), perp(v) = (-v_y, v_x).
    Returns an (T, 2, 2) array.  Used by `spv_voronoi_forces` (Bi et al. 2016 Appendix force
    on Voronoi centers); verified against finite differences in `self_test()`.
    """
    bc = b - c
    perp = np.column_stack([-bc[:, 1], bc[:, 0]])
    denom = np.sum((b - a) * perp, axis=1)
    ha = h - a
    return perp[:, :, None] * ha[:, None, :] / denom[:, None, None]


def _periodic_delaunay(
    seeds: np.ndarray, box_size: float, margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delaunay triangulation of the central cells plus periodic images within `margin`."""
    n = len(seeds)
    blocks = [seeds]
    origs = [np.arange(n)]
    for dy in (-box_size, 0.0, box_size):
        for dx in (-box_size, 0.0, box_size):
            if dx == 0.0 and dy == 0.0:
                continue
            img = seeds + np.array([dx, dy])
            m = (
                (img[:, 0] > -margin)
                & (img[:, 0] < box_size + margin)
                & (img[:, 1] > -margin)
                & (img[:, 1] < box_size + margin)
            )
            blocks.append(img[m])
            origs.append(np.arange(n)[m])
    tiled = np.vstack(blocks)
    orig = np.concatenate(origs)
    simplices = Delaunay(tiled).simplices
    simplices = simplices[(simplices < n).any(axis=1)]
    return tiled, orig, simplices


def spv_voronoi_forces(
    seeds: np.ndarray,
    box_size: float,
    p0: float,
    r: float = 1.0,
    a0: float = 1.0,
    margin: float | None = None,
    energy_prefactor: float = 2.0,
) -> dict[str, Any]:
    """
    Exact analytic forces F_i = -dE/dr_i on Voronoi cell centers for the Bi et al. (2016)
    PRX Eq. (1) tissue energy
      E = energy_prefactor * (1/2) * sum_i [ (a_i - 1)^2 + r (p_i - p_0)^2 ],
    where energy_prefactor = 2 reproduces Bi (2016) K_A = K_P = 1, A_0 = 1 (no 1/2) and
    energy_prefactor = 1 reproduces the Bi (2015) Eq. (2) dimensionless `e` of Section 1.
    Each Voronoi vertex h is the circumcenter of a Delaunay triangle, so
      F_g = - sum_{cells c} sum_{vertices h of c} (dE_c/dh) . (dh/dr_g)
    with dA_c/dh = (1/2) perp(h_next - h_prev) and dP_c/dh = u(h - h_prev) - u(h_next - h).
    Returns forces, energy, per-cell shape index q_i = P_i / sqrt(A_i), areas, perimeters,
    side counts and the Delaunay edge set (for T1 counting).
    """
    n = len(seeds)
    if margin is None:
        margin = 0.3 * box_size
    tiled, orig, S = _periodic_delaunay(seeds, box_size, margin)
    cmask = S < n
    A, B, C = tiled[S[:, 0]], tiled[S[:, 1]], tiled[S[:, 2]]
    H = _circumcenters(A, B, C)
    J = np.stack(
        [
            _circumcenter_jacobian(H, A, B, C),
            _circumcenter_jacobian(H, B, C, A),
            _circumcenter_jacobian(H, C, A, B),
        ],
        axis=1,
    )  # (T, 3, 2, 2)

    t_idx, slot = np.nonzero(cmask)
    cell = S[t_idx, slot]
    hv = H[t_idx]
    rel = hv - seeds[cell]
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.lexsort((ang, cell))
    cell_s, t_s, h_s = cell[order], t_idx[order], hv[order]
    _, start, count = np.unique(cell_s, return_index=True, return_counts=True)
    pos = np.arange(len(cell_s)) - start[cell_s]
    prev = start[cell_s] + (pos - 1) % count[cell_s]
    nxt = start[cell_s] + (pos + 1) % count[cell_s]
    hp, hn = h_s[prev], h_s[nxt]

    area = 0.5 * np.bincount(
        cell_s, weights=h_s[:, 0] * hn[:, 1] - hn[:, 0] * h_s[:, 1], minlength=n
    )
    e_next = hn - h_s
    len_next = np.hypot(e_next[:, 0], e_next[:, 1])
    perim = np.bincount(cell_s, weights=len_next, minlength=n)
    a_c = area / a0
    p_c = perim / np.sqrt(a0)

    dA = 0.5 * np.column_stack([hn[:, 1] - hp[:, 1], hp[:, 0] - hn[:, 0]])
    e_prev = h_s - hp
    len_prev = np.hypot(e_prev[:, 0], e_prev[:, 1])
    dP = e_prev / len_prev[:, None] - e_next / len_next[:, None]
    g = energy_prefactor * (
        (a_c[cell_s] - 1.0)[:, None] * dA / a0
        + r * (p_c[cell_s] - p0)[:, None] * dP / np.sqrt(a0)
    )

    forces = np.zeros((n, 2), dtype=np.float64)
    gens = S[t_s]
    for s in range(3):
        contrib = np.einsum("km,kmn->kn", g, J[t_s, s])
        np.add.at(forces, orig[gens[:, s]], -contrib)

    energy = 0.5 * energy_prefactor * float(np.sum((a_c - 1.0) ** 2 + r * (p_c - p0) ** 2))
    edges = np.vstack([S[:, [0, 1]], S[:, [1, 2]], S[:, [0, 2]]])
    edges = np.sort(orig[edges], axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edge_keys = np.unique(edges[:, 0].astype(np.int64) * n + edges[:, 1])

    return {
        "forces": forces,
        "energy": energy,
        "qs": perim / np.sqrt(np.maximum(area, 1e-12)),
        "areas": area,
        "perims": perim,
        "sides": count,
        "edge_keys": edge_keys,
    }


def spv_energy(seeds: np.ndarray, box_size: float, p0: float, r: float = 1.0) -> float:
    """Total Bi (2016) tissue energy for a Voronoi configuration (used for FD checks)."""
    return spv_voronoi_forces(seeds, box_size, p0, r=r)["energy"]


def spv_fire_minimize(
    seeds: np.ndarray,
    box_size: float,
    p0: float,
    r: float = 1.0,
    max_steps: int = 600,
    f_tol: float = 1e-7,
    dt0: float = 0.1,
    dt_max: float = 0.5,
) -> dict[str, Any]:
    """
    FIRE (Bitzek et al. 2006) zero-temperature energy minimisation of the SPV tissue energy
    (v_0 = 0 quench), used to prepare inherent-structure initial states and to measure the
    energy-minimised <q>(p_0) curve of Bi et al. (2015) Fig. 2 (<q> pinned near p_0^* for
    p_0 < p_0^*, <q> -> p_0 for p_0 > p_0^*).
    """
    x = seeds.copy()
    v = np.zeros_like(x)
    dt, alpha, n_pos = dt0, 0.1, 0
    out = spv_voronoi_forces(x, box_size, p0, r=r)
    steps = 0
    for steps in range(1, max_steps + 1):
        force = out["forces"]
        if np.abs(force).max() < f_tol:
            break
        power = float(np.sum(force * v))
        f_norm = float(np.linalg.norm(force))
        v_norm = float(np.linalg.norm(v))
        v = (1.0 - alpha) * v + alpha * force / max(f_norm, 1e-30) * v_norm
        if power > 0.0:
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            n_pos = 0
            dt *= 0.5
            alpha = 0.1
            v[:] = 0.0
        v = v + dt * force
        x = (x + dt * v) % box_size
        out = spv_voronoi_forces(x, box_size, p0, r=r)
    return {
        "seeds": x,
        "steps": steps,
        "max_force": float(np.abs(out["forces"]).max()),
        "energy_per_cell": out["energy"] / len(x),
        "mean_q": float(np.mean(out["qs"])),
        "qs": out["qs"],
        "converged": bool(np.abs(out["forces"]).max() < f_tol),
    }


def _time_averaged_msd(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Time-origin-averaged mean-squared displacement for lags 1..F//2 of unwrapped frames."""
    n_frames = frames.shape[0]
    lags = np.arange(1, n_frames // 2 + 1)
    msd = np.zeros(len(lags), dtype=np.float64)
    for i, lag in enumerate(lags):
        d = frames[lag:] - frames[:-lag]
        msd[i] = float(np.mean(np.sum(d ** 2, axis=2)))
    return lags, msd


def simulate_spv_active_monolayer(
    p0: float,
    v0: float,
    dr: float = 1.0,
    r: float = 1.0,
    n_cells: int = 100,
    t_total: float = 150.0,
    seed: int = 2016,
    sample_dt: float = 0.5,
    record_frames: bool = False,
    n_record: int = 15,
) -> dict[str, Any]:
    """
    Bi et al. (2016) PRX 6:021011 Self-Propelled Voronoi dynamics (Eqs. 1-3):
      r_i(t+dt)     = r_i + dt [ mu F_i + v_0 n_i ],   mu = 1,  F_i = -grad_i E (exact),
      theta_i(t+dt) = theta_i + sqrt(2 D_r dt) xi_i,   xi_i ~ N(0, 1).
    Initial state: uniform random seeds FIRE-quenched to the v_0 = 0 inherent structure.
    Time step dt = min(0.1, 0.2 / D_r) (overdamped Euler stable for dt <= 0.1 at K_A = 1;
    verified in prototyping — dt = 0.25 diverges).
    Observables (all finite-time, finite-size estimates over window T = t_total):
      - D_s: slope/4 of the time-origin-averaged MSD over lags tau in [T/8, T/2];
      - D_eff = D_s / D_0, D_0 = v_0^2 / (2 D_r) (Bi 2016 definition; threshold 1e-3 there);
      - Gamma_T1: Delaunay-edge losses per cell per unit time over the second half of the run;
        the fluid criterion uses the motility-normalised rate Gamma_T1 / v_0 (T1 per cell per
        unit free path v_0 t, so slow and fast fluids are judged on the same scale):
        fluid iff Gamma_T1 / v_0 > SPV_T1_PER_PATH_THRESHOLD = 0.02;
      - <q>: mean observed shape index P_i / sqrt(A_i) over the second half of the run.
    """
    dt = float(min(0.1, 0.2 / dr))
    rng = np.random.default_rng(seed)
    box_size = float(np.sqrt(n_cells))  # A_0 = 1
    a0 = 1.0
    seeds = rng.uniform(0.0, box_size, size=(n_cells, 2))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_cells)
    quench = spv_fire_minimize(seeds, box_size, p0, r=r)
    seeds = quench["seeds"]

    n_steps = int(round(t_total / dt))
    every = max(1, int(round(sample_dt / dt)))
    rec_every = max(1, n_steps // n_record)
    unwrapped = np.zeros_like(seeds)
    frames: list[np.ndarray] = []
    gif_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    q_second_half: list[float] = []
    t1_losses = 0
    edge_prev: np.ndarray | None = None
    half = n_steps // 2

    for k in range(n_steps):
        out = spv_voronoi_forces(seeds, box_size, p0, r=r, a0=a0)
        if k % every == 0:
            frames.append(unwrapped.copy())
            if edge_prev is not None and k >= half:
                t1_losses += int(len(np.setdiff1d(edge_prev, out["edge_keys"], assume_unique=True)))
                q_second_half.append(float(np.mean(out["qs"])))
            edge_prev = out["edge_keys"]
        if record_frames and (k % rec_every == 0):
            gif_frames.append((seeds.copy(), out["qs"].copy(), theta.copy(), k * dt))
        n_vec = np.column_stack([np.cos(theta), np.sin(theta)])
        disp = dt * (out["forces"] + v0 * n_vec)
        seeds = (seeds + disp) % box_size
        unwrapped += disp
        theta = theta + np.sqrt(2.0 * dr * dt) * rng.normal(size=n_cells)

    final = spv_voronoi_forces(seeds, box_size, p0, r=r, a0=a0)
    frames.append(unwrapped.copy())
    if edge_prev is not None:
        t1_losses += int(len(np.setdiff1d(edge_prev, final["edge_keys"], assume_unique=True)))
        q_second_half.append(float(np.mean(final["qs"])))
    frames_arr = np.array(frames)
    lags, msd = _time_averaged_msd(frames_arr)
    tau = lags * every * dt
    sel = tau >= tau[-1] / 4.0
    slope = float(np.polyfit(tau[sel], msd[sel], 1)[0])
    d_s = max(slope, 0.0) / 4.0
    d_0 = (v0 ** 2) / (2.0 * dr) if v0 > 0 else float("nan")
    d_eff = d_s / d_0 if v0 > 0 else 0.0
    t1_rate = t1_losses / (n_cells * (n_steps - half) * dt)
    t1_per_path = t1_rate / v0 if v0 > 0 else float("nan")
    mean_q = float(np.mean(q_second_half)) if q_second_half else float(np.mean(final["qs"]))

    return {
        "p0": float(p0),
        "v0": float(v0),
        "dr": float(dr),
        "r": float(r),
        "dt": dt,
        "t_total": float(n_steps * dt),
        "n_cells": int(n_cells),
        "box_size": box_size,
        "quench_steps": quench["steps"],
        "quench_max_force": quench["max_force"],
        "quench_mean_q": quench["mean_q"],
        "seeds": seeds,
        "theta": theta,
        "qs": final["qs"],
        "sides": final["sides"],
        "energy_per_cell": final["energy"] / n_cells,
        "tau": tau,
        "msd": msd,
        "D_s": d_s,
        "D_0": d_0,
        "D_eff": d_eff,
        "t1_rate": float(t1_rate),
        "t1_per_path": float(t1_per_path),
        "mean_q": mean_q,
        "is_fluid_t1": bool(v0 > 0 and t1_per_path > SPV_T1_PER_PATH_THRESHOLD),
        "net_displacement": unwrapped,
        "gif_frames": gif_frames,
    }


def _log_interp_crossing(
    xs: np.ndarray, ys: np.ndarray, threshold: float, qs: np.ndarray | None = None
) -> tuple[float, float]:
    """
    First crossing of `ys` (positive, increasing in x) through `threshold`, interpolated in
    log10(y); returns (x_c, q_c) with q interpolated linearly at x_c (nan if no crossing).
    """
    ly = np.log10(np.maximum(ys, 1e-9))
    lt = np.log10(threshold)
    for i in range(len(xs) - 1):
        if ly[i] <= lt < ly[i + 1]:
            frac = (lt - ly[i]) / max(ly[i + 1] - ly[i], 1e-12)
            x_c = float(xs[i] + frac * (xs[i + 1] - xs[i]))
            q_c = float(qs[i] + frac * (qs[i + 1] - qs[i])) if qs is not None else float("nan")
            return x_c, q_c
    if ly[0] > lt:
        return float("nan"), float("nan")  # fluid everywhere on the grid
    return float("nan"), float("nan")


def run_spv_motility_phase_map(
    v0_grid: tuple[float, ...] = SPV_V0_GRID,
    p0_grid: tuple[float, ...] = SPV_P0_GRID,
    dr: float = 1.0,
    seed: int = 2016,
    t_total: float = 150.0,
    n_cells: int = 100,
    record_showcase: bool = False,
) -> dict[str, Any]:
    """
    Map the Bi et al. (2016) Fig. 1 glass transition in the (p_0, v_0) plane at fixed D_r:
    for each v_0 column, the boundary p_0^c(v_0) is the log-interpolated crossing of the
    normalised T1 rate Gamma_T1 / v_0 through SPV_T1_PER_PATH_THRESHOLD, and q_c(v_0) is the
    observed shape index there
    (Bi 2016 Fig. 4 predicts q_c ≈ 3.81 independent of v_0).  The v_0 -> 0 intercept of a
    linear fit p_0^c(v_0) = p_0^c(0) - alpha v_0 is compared with the Bi (2015) point 3.81
    (Lean Theorem 5 `motility_shifted_p0c`).
    """
    runs: dict[tuple[float, float], dict[str, Any]] = {}
    for v0 in v0_grid:
        for p0 in p0_grid:
            showcase = record_showcase and abs(p0 - 3.70) < 1e-9 and (
                abs(v0 - v0_grid[1]) < 1e-9 or abs(v0 - v0_grid[-1]) < 1e-9
            )
            runs[(v0, p0)] = simulate_spv_active_monolayer(
                p0=p0, v0=v0, dr=dr, n_cells=n_cells, t_total=t_total, seed=seed,
                record_frames=showcase,
            )
    p0_arr = np.array(p0_grid, dtype=np.float64)
    t1 = np.array([[runs[(v0, p0)]["t1_rate"] for p0 in p0_grid] for v0 in v0_grid])
    t1_path = np.array([[runs[(v0, p0)]["t1_per_path"] for p0 in p0_grid] for v0 in v0_grid])
    deff = np.array([[runs[(v0, p0)]["D_eff"] for p0 in p0_grid] for v0 in v0_grid])
    mean_q = np.array([[runs[(v0, p0)]["mean_q"] for p0 in p0_grid] for v0 in v0_grid])
    p0c = np.full(len(v0_grid), np.nan)
    qc = np.full(len(v0_grid), np.nan)
    for i in range(len(v0_grid)):
        p0c[i], qc[i] = _log_interp_crossing(p0_arr, t1_path[i], SPV_T1_PER_PATH_THRESHOLD, mean_q[i])
    ok = np.isfinite(p0c)
    v0_arr = np.array(v0_grid, dtype=np.float64)
    if ok.sum() >= 2:
        slope, intercept = np.polyfit(v0_arr[ok], p0c[ok], 1)
    else:
        slope, intercept = float("nan"), float("nan")
    diffs = np.diff(p0c[ok])
    return {
        "dr": float(dr),
        "v0_grid": v0_arr,
        "p0_grid": p0_arr,
        "t1_rate": t1,
        "t1_per_path": t1_path,
        "D_eff": deff,
        "mean_q": mean_q,
        "p0c": p0c,
        "qc": qc,
        "boundary_slope_alpha": float(-slope),
        "boundary_intercept_p0c0": float(intercept),
        "boundary_monotone_decreasing": bool(np.all(diffs <= 1e-9)) if len(diffs) else False,
        "qc_mean": float(np.nanmean(qc)) if ok.any() else float("nan"),
        "qc_std": float(np.nanstd(qc)) if ok.any() else float("nan"),
        "runs": runs,
    }


def run_spv_rotational_diffusion_sweep(
    dr_grid: tuple[float, ...] = SPV_DR_GRID,
    p0_grid: tuple[float, ...] = SPV_P0_GRID,
    v0: float = SPV_DR_SWEEP_V0,
    seed: int = 2016,
    t_total: float = 150.0,
    n_cells: int = 100,
    reuse: dict[tuple[float, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Third axis of the Bi et al. (2016) (p_0, v_0, D_r) glass transition surface: at fixed
    v_0, sweep the rotational diffusion D_r (persistence time 1/D_r).  Bi (2016) Fig. 3:
    more persistent motion (smaller D_r) enlarges the fluid region, i.e. p_0^c decreases as
    D_r decreases; the absolute self-diffusivity D_s increases as D_r decreases.
    `reuse` lets the D_r = 1 column be shared with `run_spv_motility_phase_map`.
    """
    p0_arr = np.array(p0_grid, dtype=np.float64)
    rows: dict[float, dict[str, Any]] = {}
    for dr in dr_grid:
        runs_dr = []
        for p0 in p0_grid:
            if reuse is not None and abs(dr - 1.0) < 1e-12 and (v0, p0) in reuse:
                runs_dr.append(reuse[(v0, p0)])
            else:
                runs_dr.append(
                    simulate_spv_active_monolayer(
                        p0=p0, v0=v0, dr=dr, n_cells=n_cells, t_total=t_total, seed=seed
                    )
                )
        t1 = np.array([s["t1_rate"] for s in runs_dr])
        t1_path = np.array([s["t1_per_path"] for s in runs_dr])
        ds = np.array([s["D_s"] for s in runs_dr])
        deff = np.array([s["D_eff"] for s in runs_dr])
        mq = np.array([s["mean_q"] for s in runs_dr])
        p0c, qc = _log_interp_crossing(p0_arr, t1_path, SPV_T1_PER_PATH_THRESHOLD, mq)
        rows[dr] = {
            "t1_rate": t1, "t1_per_path": t1_path, "D_s": ds, "D_eff": deff, "mean_q": mq, "p0c": p0c, "qc": qc,
            "runs": runs_dr,
        }
    dr_arr = np.array(dr_grid, dtype=np.float64)
    p0c_arr = np.array([rows[d]["p0c"] for d in dr_grid])
    # D_s at the common p0 = 3.70 column across D_r (absolute diffusivity trend)
    i70 = int(np.argmin(np.abs(p0_arr - 3.70)))
    ds_70 = np.array([rows[d]["D_s"][i70] for d in dr_grid])
    t1_70 = np.array([rows[d]["t1_rate"][i70] for d in dr_grid])
    ok = np.isfinite(p0c_arr)
    return {
        "v0": float(v0),
        "dr_grid": dr_arr,
        "p0_grid": p0_arr,
        "rows": rows,
        "p0c": p0c_arr,
        "p0c_nondecreasing_in_dr": bool(np.all(np.diff(p0c_arr[ok]) >= -1e-9)) if ok.sum() >= 2 else False,
        "D_s_at_p0_3p70": ds_70,
        "t1_rate_at_p0_3p70": t1_70,
        "D_s_decreasing_in_dr": bool(np.all(np.diff(ds_70) <= 0.0)),
    }


def run_spv_zero_motility_minimized_sweep(seed: int = 2016, n_cells: int = 100) -> dict[str, Any]:
    """
    Energy-minimised (v_0 = 0, FIRE) <q>(p_0) over the Bi (2015) 8-point reference sweep —
    the force-based counterpart of the phenomenological `run_p0_phase_sweep`.  Bi (2015)
    Fig. 2: <q> is pinned near p_0^* in the solid (p_0 < 3.81) and follows <q> ≈ p_0 in the
    fluid.
    """
    ref_data = verify_reference_dataset()
    ref_sweep = ref_data["bi_2015_vertex_unjamming_mechanics"]["phase_sweep_reference_bi_2015"]
    box_size = float(np.sqrt(n_cells))
    rng = np.random.default_rng(seed)
    seeds0 = rng.uniform(0.0, box_size, size=(n_cells, 2))
    p0s, q_min, q_ref, e_cell, conv = [], [], [], [], []
    for item in ref_sweep:
        p0_val = float(item["p0"])
        res = spv_fire_minimize(seeds0, box_size, p0_val, r=1.0)
        p0s.append(p0_val)
        q_min.append(res["mean_q"])
        q_ref.append(float(item["observed_mean_q"]))
        e_cell.append(res["energy_per_cell"])
        conv.append(res["converged"])
    p0_arr = np.array(p0s)
    q_arr = np.array(q_min)
    ref_arr = np.array(q_ref)
    return {
        "p0": p0_arr,
        "minimized_mean_q": q_arr,
        "ref_mean_q": ref_arr,
        "energy_per_cell": np.array(e_cell),
        "converged": np.array(conv),
        "pearson_r": float(stats.pearsonr(q_arr, ref_arr).statistic),
        "rmse": float(np.sqrt(np.mean((q_arr - ref_arr) ** 2))),
        "solid_mean_q": float(np.mean(q_arr[p0_arr < P0_STAR - 1e-6])),
        "fluid_identity_rmse": float(
            np.sqrt(np.mean((q_arr[p0_arr > P0_STAR + 1e-6] - p0_arr[p0_arr > P0_STAR + 1e-6]) ** 2))
        ),
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
    spv_t_total: float = 150.0,
    spv_cells: int = 100,
    spv_seed: int = 2016,
) -> dict[str, Any]:
    """
    Run all 4 vertex regimes, 8-point p_0 sweep, bronchial maturation, specimen CV
    segmentation, the Pass 3 Bi (2016) SPV motility phase map / D_r sweep / zero-motility
    FIRE sweep, and render the 15-panel figure plus the 4-panel GIF.
    """
    ref_data = verify_reference_dataset()
    spv_ref = load_spv_reference()
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

    # Pass 3: Bi et al. (2016) SPV active motility — (p0, v0) map at D_r = 1, D_r sweep, FIRE sweep
    spv_map = run_spv_motility_phase_map(
        dr=1.0, seed=spv_seed, t_total=spv_t_total, n_cells=spv_cells, record_showcase=True
    )
    spv_dr = run_spv_rotational_diffusion_sweep(
        seed=spv_seed, t_total=spv_t_total, n_cells=spv_cells, reuse=spv_map["runs"]
    )
    spv_fire = run_spv_zero_motility_minimized_sweep(seed=spv_seed, n_cells=spv_cells)
    spv_solid_show = spv_map["runs"][(SPV_V0_GRID[1], 3.70)]
    spv_fluid_show = spv_map["runs"][(SPV_V0_GRID[-1], 3.70)]

    # Render 15-panel comparison figure (3x5 grid)
    fig, axes = plt.subplots(3, 5, figsize=(20.5, 13.0), dpi=150)
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
    ax_b2.plot(
        spv_fire["p0"],
        spv_fire["minimized_mean_q"],
        "D-",
        color="#7c3aed",
        lw=1.6,
        ms=4.5,
        label=f"SPV FIRE-minimised ⟨q⟩, v₀=0 (r={spv_fire['pearson_r']:.3f})",
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

    # ---------------- Row C (Pass 3): Bi et al. (2016) PRX Self-Propelled Voronoi motility ----------------
    def _draw_spv_snapshot(ax, sim: dict[str, Any], title: str) -> None:
        box = sim["box_size"]
        polys = _periodic_voronoi_polygons(sim["seeds"], box_size=box)
        for poly, qv in zip(polys, sim["qs"]):
            cx_p, cy_p = np.mean(poly, axis=0)
            if 0.6 <= cx_p <= box - 0.6 and 0.6 <= cy_p <= box - 0.6:
                norm_q = float(np.clip((qv - 3.68) / (4.12 - 3.68), 0.0, 1.0))
                ax.fill(poly[:, 0], poly[:, 1], facecolor=cmap_q(norm_q), edgecolor="#0f172a", lw=0.8, alpha=0.88)
        # Net displacement over the run (unwrapped) drawn from the final position backwards
        disp = sim["net_displacement"]
        for (x, y), (dx, dy) in zip(sim["seeds"], disp):
            if 0.9 <= x <= box - 0.9 and 0.9 <= y <= box - 0.9:
                ax.plot([x - dx, x], [y - dy, y], color="#1e3a8a", lw=0.9, alpha=0.75)
        # Current polarity vectors n_i
        ax.quiver(
            sim["seeds"][:, 0], sim["seeds"][:, 1],
            np.cos(sim["theta"]), np.sin(sim["theta"]),
            color="#111827", scale=28, width=0.004, alpha=0.65,
        )
        ax.set_xlim(0.9, box - 0.9)
        ax.set_ylim(0.9, box - 0.9)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.4, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    _draw_spv_snapshot(
        axes[2, 0],
        spv_solid_show,
        f"C1. SPV Glassy Solid (Bi 2016): p₀={spv_solid_show['p0']:.2f}, v₀={spv_solid_show['v0']:.2f}, D_r={spv_solid_show['dr']:.1f}\n"
        f"⟨q⟩={spv_solid_show['mean_q']:.3f} | Γ_T1/v₀={spv_solid_show['t1_per_path']:.4f} | D_eff={spv_solid_show['D_eff']:.3f}",
    )
    _draw_spv_snapshot(
        axes[2, 1],
        spv_fluid_show,
        f"C2. SPV Motility-Unjammed Fluid: p₀={spv_fluid_show['p0']:.2f}, v₀={spv_fluid_show['v0']:.2f}, D_r={spv_fluid_show['dr']:.1f}\n"
        f"⟨q⟩={spv_fluid_show['mean_q']:.3f} | Γ_T1/v₀={spv_fluid_show['t1_per_path']:.4f} | D_eff={spv_fluid_show['D_eff']:.3f}",
    )

    # C3: time-averaged MSD(τ) at p0 = 3.70 for all v0 (motility-driven cage escape)
    ax_c3 = axes[2, 2]
    v0_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(spv_map["v0_grid"])))
    for v0, col in zip(spv_map["v0_grid"], v0_colors):
        sim = spv_map["runs"][(float(v0), 3.70)]
        ax_c3.loglog(
            sim["tau"], sim["msd"], "-", color=col, lw=1.9,
            label=f"v₀={v0:.2f}: D_eff={sim['D_eff']:.3f}, Γ_T1/v₀={sim['t1_per_path']:.4f}",
        )
    tau_ref = np.array([1.0, 70.0])
    ax_c3.loglog(tau_ref, 0.02 * tau_ref, "k:", lw=1.1, label="∝ τ (diffusive)")
    ax_c3.set_title("C3. SPV MSD(τ) at p₀=3.70, D_r=1 (Bi 2016 Fig. 1b)", fontsize=8.8, fontweight="bold")
    ax_c3.set_xlabel("Lag τ (units of 1/(μ K_A A₀))", fontsize=8.3)
    ax_c3.set_ylabel("⟨Δr²(τ)⟩ (units of A₀)", fontsize=8.3)
    ax_c3.legend(fontsize=6.2, loc="upper left")
    ax_c3.grid(alpha=0.25, which="both")

    # C4: (p0, v0) glass transition map at D_r = 1 (Bi 2016 Fig. 1a) — log10 (Gamma_T1 / v0) heatmap
    ax_c4 = axes[2, 3]
    p0g, v0g = spv_map["p0_grid"], spv_map["v0_grid"]
    dp = p0g[1] - p0g[0]
    v0_edges = np.concatenate([[0.0], 0.5 * (v0g[1:] + v0g[:-1]), [v0g[-1] + 0.5 * (v0g[-1] - v0g[-2])]])
    p0_edges = np.concatenate([p0g - 0.5 * dp, [p0g[-1] + 0.5 * dp]])
    im = ax_c4.pcolormesh(
        p0_edges, v0_edges, np.log10(np.maximum(spv_map["t1_per_path"], 1e-4)),
        cmap="coolwarm", vmin=-4, vmax=0.3, shading="flat", edgecolors="white", lw=0.4,
    )
    for i, v0 in enumerate(v0g):
        for j, p0 in enumerate(p0g):
            ax_c4.text(p0, v0, f"{spv_map['mean_q'][i, j]:.3f}", ha="center", va="center", fontsize=5.6, color="#111827")
    ok = np.isfinite(spv_map["p0c"])
    ax_c4.plot(spv_map["p0c"][ok], v0g[ok], "k-o", lw=2.0, ms=5, label=f"p₀ᶜ(v₀): Γ_T1/v₀ = {SPV_T1_PER_PATH_THRESHOLD:g}")
    vfit = np.linspace(0.0, v0g[-1] * 1.05, 20)
    ax_c4.plot(
        spv_map["boundary_intercept_p0c0"] - spv_map["boundary_slope_alpha"] * vfit, vfit,
        "--", color="#111827", lw=1.2,
        label=f"fit p₀ᶜ = {spv_map['boundary_intercept_p0c0']:.3f} − {spv_map['boundary_slope_alpha']:.2f}·v₀",
    )
    ax_c4.plot([SPV_P0_STAR_2016], [0.0], "*", color="#facc15", mec="k", ms=13, label="Bi 2016 v₀→0: p₀ᶜ = 3.81")
    ax_c4.axvline(P0_STAR, color="#dc2626", ls="--", lw=1.2, label="Bi 2015 p₀* = 3.813")
    ax_c4.set_xlim(p0_edges[0], p0_edges[-1])
    ax_c4.set_ylim(0.0, v0_edges[-1])
    ax_c4.set_title(
        f"C4. SPV Glass Map (p₀, v₀) at D_r=1 — log₁₀(Γ_T1/v₀) (cells: ⟨q⟩)\nq_c = {spv_map['qc_mean']:.3f} ± {spv_map['qc_std']:.3f} (Bi 2016 Fig. 4: 3.81)",
        fontsize=8.6, fontweight="bold",
    )
    ax_c4.set_xlabel("Target Shape Index p₀", fontsize=8.3)
    ax_c4.set_ylabel("Self-Propulsion Speed v₀", fontsize=8.3)
    ax_c4.legend(fontsize=5.9, loc="lower left", framealpha=0.92)
    cb = fig.colorbar(im, ax=ax_c4, fraction=0.045, pad=0.02)
    cb.set_label("log₁₀(Γ_T1 / v₀) (T1 / cell / unit free path)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # C5: D_r axis — T1 rate vs p0 for three D_r at fixed v0 (Bi 2016 Fig. 3)
    ax_c5 = axes[2, 4]
    dr_colors = {0.25: "#dc2626", 1.0: "#2563eb", 4.0: "#16a34a"}
    for dr_val in spv_dr["dr_grid"]:
        row = spv_dr["rows"][float(dr_val)]
        col = dr_colors.get(float(dr_val), "#111827")
        lbl = f"D_r={dr_val:g} (τ_p={1.0 / dr_val:g}): p₀ᶜ={row['p0c']:.3f}, q_c={row['qc']:.3f}" if np.isfinite(row["p0c"]) else f"D_r={dr_val:g}: fluid on grid"
        ax_c5.semilogy(spv_dr["p0_grid"], np.maximum(row["t1_per_path"], 1e-4), "o-", color=col, lw=1.9, ms=5, label=lbl)
        if np.isfinite(row["p0c"]):
            ax_c5.plot([row["p0c"]], [SPV_T1_PER_PATH_THRESHOLD], "v", color=col, ms=9, mec="k")
    ax_c5.axhline(SPV_T1_PER_PATH_THRESHOLD, color="#0f172a", ls="--", lw=1.2, label=f"Finite-time fluid threshold Γ_T1/v₀ = {SPV_T1_PER_PATH_THRESHOLD:g}")
    ax_c5.axvline(P0_STAR, color="#dc2626", ls=":", lw=1.1)
    ds70 = spv_dr["D_s_at_p0_3p70"]
    ax_c5.text(
        0.03, 0.04,
        "D_s at p₀=3.70: " + ", ".join(f"{d:.1e}" for d in ds70) + f"\n(D_r = {', '.join(f'{d:g}' for d in spv_dr['dr_grid'])}; "
        + ("↓ with D_r ✓" if spv_dr["D_s_decreasing_in_dr"] else "not monotone"),
        transform=ax_c5.transAxes, fontsize=6.4, va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="#94a3b8", alpha=0.9),
    )
    ax_c5.set_title(f"C5. Persistence Axis D_r at v₀={spv_dr['v0']:.2f} (Bi 2016 Fig. 3)", fontsize=8.8, fontweight="bold")
    ax_c5.set_xlabel("Target Shape Index p₀", fontsize=8.3)
    ax_c5.set_ylabel("Normalised T1 Rate Γ_T1 / v₀ (per cell per unit free path)", fontsize=8.3)
    ax_c5.set_ylim(5e-5, 5.0)
    ax_c5.legend(fontsize=6.0, loc="upper left")
    ax_c5.grid(alpha=0.25, which="both")

    fig.suptitle(
        "Bi et al. (2015) & Park et al. (2015) Zero-Temperature Shape-Index Unjamming, Bronchial Epithelial Rigidity Transition "
        "& Bi et al. (2016) Self-Propelled Voronoi Motility-Driven Glass Transition",
        fontsize=12.6,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(COMPARISON_PNG, dpi=150, facecolor="white")
    plt.close(fig)

    # Render 4-panel animated GIF (Solid vs Critical vs Asthmatic Fluid vs SPV motile fluid)
    pil_frames: list[Image.Image] = []
    f_solid = regimes["solid_jammed"]["frames"]
    f_crit = regimes["critical_point"]["frames"]
    f_fluid = regimes["fluid_asthmatic"]["frames"]
    f_spv = spv_fluid_show["gif_frames"]
    n_f = min(len(f_solid), len(f_crit), len(f_fluid), len(f_spv))

    for idx in range(n_f):
        polys_s, qs_s, step_val = f_solid[idx]
        polys_c, qs_c, _ = f_crit[idx]
        polys_f, qs_f, _ = f_fluid[idx]
        spv_seeds, spv_qs, spv_theta, spv_t = f_spv[idx]
        spv_polys = _periodic_voronoi_polygons(spv_seeds, box_size=spv_fluid_show["box_size"])

        fig_g, axes_g = plt.subplots(1, 4, figsize=(13.6, 3.6), dpi=100)
        axes_g[3].quiver(
            spv_seeds[:, 0], spv_seeds[:, 1], np.cos(spv_theta), np.sin(spv_theta),
            color="#111827", scale=26, width=0.005, alpha=0.7, zorder=5,
        )
        for ax_i, p_list, q_list, lbl in [
            (
                axes_g[3],
                spv_polys,
                spv_qs,
                f"SPV Motile Fluid p₀=3.70, v₀={spv_fluid_show['v0']:.2f} (t={spv_t:.0f}, ⟨q⟩={np.mean(spv_qs):.3f})",
            ),
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
            "Bi et al. (2015) & Park et al. (2015) Epithelial Vertex Jamming vs. Unjamming Dynamics "
            "| Bi et al. (2016) Self-Propelled Voronoi Motility",
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
        "spv_map": spv_map,
        "spv_dr": spv_dr,
        "spv_fire": spv_fire,
        "spv_reference": spv_ref,
    }


# =============================================================================
# Section 5: MorphoEval Dynamic Measurement Hook & 8-Check Self-Test Suite
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Return 15 quantitative morphometric measurements for morphoeval/measure_outputs.py (8 Pass 2 + 7 Pass 3 SPV)."""
    ref_data = verify_reference_dataset()
    spv_ref = load_spv_reference()
    solid = simulate_vertex_unjamming_regime(p0=3.70, r=0.5, n_steps=32, seed=2015)
    fluid = simulate_vertex_unjamming_regime(p0=3.955, r=0.5, n_steps=32, seed=2015)
    sweep = run_p0_phase_sweep(seed=2015)
    park = simulate_park_2015_bronchial_maturation(ref_data)
    specimen = segment_human_monolayer_specimen(ref_data)
    spv_map = run_spv_motility_phase_map(dr=1.0, seed=2016)
    spv_dr = run_spv_rotational_diffusion_sweep(seed=2016, reuse=spv_map["runs"])
    spv_fire = run_spv_zero_motility_minimized_sweep(seed=2016)
    spv_claims = spv_ref["bi_2016_spv_glass_transition"]
    v0_lo, v0_hi = SPV_V0_GRID[1], SPV_V0_GRID[-1]
    t1_lo = spv_map["runs"][(v0_lo, 3.70)]["t1_rate"]
    t1_hi = spv_map["runs"][(v0_hi, 3.70)]["t1_rate"]
    p0c_str = ", ".join(
        f"v0={v:.2f}: {p:.3f}" for v, p in zip(spv_map["v0_grid"], spv_map["p0c"])
    )
    p0c_dr_str = ", ".join(
        f"D_r={d:g}: {p:.3f}" for d, p in zip(spv_dr["dr_grid"], spv_dr["p0c"])
    )

    spv_measurements = [
        {
            "name": "spv_zero_motility_boundary_intercept_p0c",
            "value": round(spv_map["boundary_intercept_p0c0"], 4),
            "unit": "dimensionless",
            "reference_value": float(spv_claims["zero_motility_glass_point_p0c"]),
            "method": f"v0 -> 0 intercept of the linear fit p0c(v0) = p0c(0) - alpha v0 to the SPV glass boundary at D_r=1 (normalised T1-rate criterion Gamma_T1/v0={SPV_T1_PER_PATH_THRESHOLD:g}; boundary points {p0c_str}; alpha={spv_map['boundary_slope_alpha']:.3f}); Bi et al. (2016) PRX Fig. 1: p0c(0)=3.81",
        },
        {
            "name": "spv_glass_boundary_structural_order_parameter_qc",
            "value": round(spv_map["qc_mean"], 4),
            "unit": "dimensionless",
            "reference_value": float(spv_claims["structural_order_parameter_q_c"]),
            "method": f"Mean observed shape index <q> interpolated at the SPV glass boundary across the v0 grid {tuple(float(v) for v in spv_map['v0_grid'])} at D_r=1 (std={spv_map['qc_std']:.4f}); Bi et al. (2016) PRX Fig. 4: q_c ~= 3.81 independent of (v0, D_r)",
        },
        {
            "name": "spv_motility_fluidization_slope_alpha",
            "value": round(spv_map["boundary_slope_alpha"], 4),
            "unit": "dimensionless (dp0c/dv0)",
            "reference_value": float(spv_claims["approximate_boundary_slope_alpha_dr1"]),
            "method": f"Slope -dp0c/dv0 of the SPV glass boundary at D_r=1 (monotone decreasing: {spv_map['boundary_monotone_decreasing']}); Bi et al. (2016) PRX Fig. 1a: the solid region shrinks with v0 (reference slope read approximately from the figure, uncertainty +/-0.3)",
        },
        {
            "name": "spv_t1_rate_ratio_high_vs_low_motility_at_p0_3p70",
            "value": round(float(t1_hi / max(t1_lo, 1e-9)), 2),
            "unit": "ratio",
            "reference_value": float(spv_claims["minimum_fluid_to_solid_t1_rate_contrast"]),
            "method": f"Ratio of per-cell T1 rates at fixed p0=3.70, D_r=1 between v0={v0_hi:.2f} (Gamma_T1={t1_hi:.4f}) and v0={v0_lo:.2f} (Gamma_T1={t1_lo:.4f}): motility alone unjams the tissue at fixed p0 (Bi 2016 Fig. 1); reference = minimum contrast of 10x expected across the boundary",
        },
        {
            "name": "spv_boundary_p0c_shift_dr4_minus_dr0p25",
            "value": round(float(spv_dr["p0c"][-1] - spv_dr["p0c"][0]), 4) if np.all(np.isfinite(spv_dr["p0c"][[0, -1]])) else None,
            "unit": "dimensionless",
            "reference_value": float(spv_claims["minimum_p0c_shift_dr_0p25_to_4"]),
            "method": f"p0c(D_r=4) - p0c(D_r=0.25) at v0={spv_dr['v0']:.2f} ({p0c_dr_str}); Bi et al. (2016) PRX Fig. 3: less persistent motion (larger D_r) enlarges the solid region (positive shift); D_s at p0=3.70 decreasing in D_r: {spv_dr['D_s_decreasing_in_dr']}",
        },
        {
            "name": "spv_fire_minimized_q_sweep_pearson_r",
            "value": round(spv_fire["pearson_r"], 4),
            "unit": "r",
            "reference_value": 0.985,
            "method": f"Pearson r between the FIRE energy-minimised (v0=0, exact SPV forces) <q>(p0) and the Bi et al. (2015) 8-point reference curve (RMSE={spv_fire['rmse']:.4f}; solid-branch <q>={spv_fire['solid_mean_q']:.4f} vs pinning 3.813; fluid-branch |<q>-p0| RMSE={spv_fire['fluid_identity_rmse']:.4f})",
        },
        {
            "name": "spv_fire_minimized_solid_branch_mean_q",
            "value": round(spv_fire["solid_mean_q"], 4),
            "unit": "dimensionless",
            "reference_value": 3.813,
            "method": "Mean energy-minimised <q> over the solid branch p0 in {3.55, 3.65, 3.722, 3.77} with exact Voronoi forces (Bi et al. 2015 Fig. 2: <q> pinned at p0* = 3.813 by geometric incompatibility)",
        },
    ]

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
    ] + spv_measurements


def self_test() -> None:
    """Run the 13-check verification suite for Slot 045 (8 Pass 2 checks + 5 Pass 3 SPV checks)."""
    # 1. Verify SHA-256 of reference dataset and human monolayer specimen micrograph
    ref = verify_reference_dataset()
    assert len(ref["metadata"]["sources"]) == 5
    spv_ref = load_spv_reference()
    assert spv_ref["metadata"]["doi"] == "10.1103/PhysRevX.6.021011"

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

    # 9. (Pass 3) Exact SPV Voronoi forces vs central finite differences of the energy
    rng = np.random.default_rng(7)
    box = 10.0
    x0 = rng.uniform(0.0, box, size=(100, 2))
    f_analytic = spv_voronoi_forces(x0, box, p0=3.70, r=1.0)["forces"]
    eps = 1e-6
    fd_err = 0.0
    fd_scale = 0.0
    for i in range(6):
        for d in range(2):
            xp, xm = x0.copy(), x0.copy()
            xp[i, d] += eps
            xm[i, d] -= eps
            f_num = -(spv_energy(xp, box, 3.70) - spv_energy(xm, box, 3.70)) / (2.0 * eps)
            fd_err = max(fd_err, abs(f_num - f_analytic[i, d]))
            fd_scale = max(fd_scale, abs(f_num))
    assert fd_err / fd_scale < 1e-5, f"SPV force FD mismatch: rel err {fd_err / fd_scale:.2e}"
    # Margin-restricted periodic images must equal the full 3x3 tiling
    f_full = spv_voronoi_forces(x0, box, p0=3.70, r=1.0, margin=box)["forces"]
    assert np.allclose(f_analytic, f_full, atol=1e-9), "margin tiling != full tiling"
    # Momentum conservation: internal forces sum to zero
    assert np.abs(f_analytic.sum(axis=0)).max() < 1e-8

    # 10. (Pass 3) FIRE quench reproduces Bi (2015) pinning (<q> ~ 3.81 for p0 < p0*) and
    #     fluid identity (<q> ~ p0 for p0 > p0*) with exact forces (not the heuristic)
    q_solid = spv_fire_minimize(x0, box, p0=3.60, r=1.0)
    q_fluid = spv_fire_minimize(x0, box, p0=3.95, r=1.0)
    assert q_solid["converged"], "FIRE did not converge in the solid"
    assert abs(q_solid["mean_q"] - P0_STAR) < 0.03, f"solid <q>={q_solid['mean_q']:.4f}"
    assert abs(q_fluid["mean_q"] - 3.95) < 0.03, f"fluid <q>={q_fluid['mean_q']:.4f}"
    assert q_fluid["energy_per_cell"] < 0.2 * q_solid["energy_per_cell"]

    # 11. (Pass 3) Motility unjams at fixed p0: T1 rate and D_eff rise with v0 (short window)
    spv_lo = simulate_spv_active_monolayer(p0=3.70, v0=0.05, dr=1.0, t_total=40.0, seed=2016)
    spv_hi = simulate_spv_active_monolayer(p0=3.70, v0=0.60, dr=1.0, t_total=40.0, seed=2016)
    assert spv_hi["t1_rate"] > 10.0 * max(spv_lo["t1_rate"], 1e-4), (
        f"T1 rates: lo={spv_lo['t1_rate']:.4f}, hi={spv_hi['t1_rate']:.4f}"
    )
    assert spv_hi["D_eff"] > spv_lo["D_eff"]
    assert spv_hi["mean_q"] > spv_lo["mean_q"]
    assert abs(spv_lo["dt"] - 0.1) < 1e-12 and abs(
        simulate_spv_active_monolayer(p0=3.70, v0=0.3, dr=4.0, t_total=2.0, seed=1)["dt"] - 0.05
    ) < 1e-12

    # 12. (Pass 3) Bitwise determinism of the stochastic SPV integrator
    spv_a = simulate_spv_active_monolayer(p0=3.70, v0=0.30, dr=1.0, t_total=10.0, seed=2016)
    spv_b = simulate_spv_active_monolayer(p0=3.70, v0=0.30, dr=1.0, t_total=10.0, seed=2016)
    assert np.array_equal(spv_a["seeds"], spv_b["seeds"]) and spv_a["t1_rate"] == spv_b["t1_rate"]

    # 13. (Pass 3) Boundary extractor and Lean Theorem 5 parity (motility lowers p0c)
    xs = np.array([3.5, 3.6, 3.7, 3.8])
    ys = np.array([1e-4, 1e-3, 1e-2, 1e-1])
    x_c, q_c = _log_interp_crossing(xs, ys, 5e-3, np.array([3.80, 3.80, 3.82, 3.86]))
    assert abs(x_c - (3.6 + 0.1 * np.log10(5.0))) < 1e-9 and 3.80 < q_c < 3.82
    p0_star_m, alpha_m = 3813, 400
    p0c = [p0_star_m - alpha_m * v for v in (0, 1, 2)]
    assert p0c[0] == p0_star_m and p0c[1] < p0_star_m and p0c[2] < p0c[1]

    # 8. Verify measure_morphoeval contract (15 measurements, each with a reference value)
    meas = measure_morphoeval()
    assert len(meas) == 15 and all(m["value"] is not None for m in meas)
    assert all("reference_value" in m and "unit" in m for m in meas)
    spv_meas = {m["name"]: m["value"] for m in meas if m["name"].startswith("spv_")}
    assert abs(spv_meas["spv_glass_boundary_structural_order_parameter_qc"] - SPV_P0_STAR_2016) < 0.05
    assert abs(spv_meas["spv_zero_motility_boundary_intercept_p0c"] - SPV_P0_STAR_2016) < 0.08
    assert spv_meas["spv_fire_minimized_q_sweep_pearson_r"] > 0.95
    print(
        f"[self-test] All 13 Pass 3 checks passed "
        f"(solid_q={solid['mean_q']:.4f}, fluid_q={fluid['mean_q']:.4f}, "
        f"sweep_r={sweep['pearson_r_mean_q']:.4f}, spec_cells={spec['n_segmented_cells']}, "
        f"kgamma_r={spec['r_kgamma']:.4f}, spv_force_fd_rel_err={fd_err / fd_scale:.1e}, "
        f"spv_qc={spv_meas['spv_glass_boundary_structural_order_parameter_qc']:.4f}, "
        f"spv_p0c0={spv_meas['spv_zero_motility_boundary_intercept_p0c']:.4f}, "
        f"fire_r={spv_meas['spv_fire_minimized_q_sweep_pearson_r']:.4f})."
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
        "--spv-time",
        type=float,
        default=150.0,
        help="Bi (2016) SPV production window T per (p0, v0, D_r) point, units 1/(mu K_A A_0) (Pass 3)",
    )
    ap.add_argument(
        "--spv-cells",
        type=int,
        default=100,
        help="Number of SPV Voronoi cells N (periodic box L = sqrt(N), A_0 = 1) (Pass 3)",
    )
    ap.add_argument(
        "--spv-seed",
        type=int,
        default=2016,
        help="Deterministic RNG seed for the SPV active-Brownian integrator (Pass 3)",
    )
    ap.add_argument(
        "--spv-point",
        type=float,
        nargs=3,
        metavar=("P0", "V0", "DR"),
        default=None,
        help="Run a single SPV point (p0, v0, D_r), print Gamma_T1 / D_eff / <q>, and exit (Pass 3)",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run 13-check verification suite and exit",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.spv_point is not None:
        p0_pt, v0_pt, dr_pt = args.spv_point
        pt = simulate_spv_active_monolayer(
            p0=p0_pt, v0=v0_pt, dr=dr_pt, n_cells=args.spv_cells, t_total=args.spv_time, seed=args.spv_seed
        )
        print(
            f"[spv-point] p0={pt['p0']:.3f} v0={pt['v0']:.3f} D_r={pt['dr']:.3g} dt={pt['dt']:.3g} T={pt['t_total']:.0f}: "
            f"<q>={pt['mean_q']:.4f}, Gamma_T1={pt['t1_rate']:.5f}, Gamma_T1/v0={pt['t1_per_path']:.4f} (fluid={pt['is_fluid_t1']}), "
            f"D_s={pt['D_s']:.3e}, D_0={pt['D_0']:.3e}, D_eff={pt['D_eff']:.4f}, "
            f"quench steps={pt['quench_steps']} max|F|={pt['quench_max_force']:.1e}"
        )
        return 0

    out = run_all_regimes_and_render(
        p0_solid=args.p0_solid,
        p0_fluid=args.p0_fluid,
        r=args.r,
        n_steps=args.steps,
        seed=args.seed,
        spv_t_total=args.spv_time,
        spv_cells=args.spv_cells,
        spv_seed=args.spv_seed,
    )
    s_sol = out["regimes"]["solid_jammed"]
    s_flu = out["regimes"]["fluid_asthmatic"]
    sp = out["specimen"]
    sm = out["spv_map"]
    sd = out["spv_dr"]
    sf = out["spv_fire"]
    p0c_str = ", ".join(f"v0={v:.2f}:{p:.3f}" for v, p in zip(sm["v0_grid"], sm["p0c"]))
    print(
        f"[2015-bi-vertex-glass-jamming] Rendered Pass 3 outputs:\n"
        f"  - Comparison PNG : {COMPARISON_PNG}\n"
        f"  - Growth GIF     : {GROWTH_GIF}\n"
        f"  - Solid Jammed   : p0={s_sol['p0']:.3f}, <q>={s_sol['mean_q']:.4f}, G={s_sol['shear_modulus_G']:.4f}\n"
        f"  - Fluid Asthmatic: p0={s_flu['p0']:.3f}, <q>={s_flu['mean_q']:.4f}, G={s_flu['shear_modulus_G']:.4f}\n"
        f"  - Phase Sweep r  : {out['sweep']['pearson_r_mean_q']:.4f} (RMSE={out['sweep']['rmse_mean_q']:.4f})\n"
        f"  - Specimen Match : N={sp['n_segmented_cells']} cells, <q>={sp['mean_q']:.4f}, k-Gamma r={sp['r_kgamma']:.4f}\n"
        f"  - SPV (Bi 2016)  : boundary {p0c_str}; p0c(0)={sm['boundary_intercept_p0c0']:.3f}, "
        f"alpha={sm['boundary_slope_alpha']:.3f}, q_c={sm['qc_mean']:.4f}+/-{sm['qc_std']:.4f}\n"
        f"  - SPV D_r sweep  : p0c(D_r) = {', '.join(f'{d:g}:{p:.3f}' for d, p in zip(sd['dr_grid'], sd['p0c']))}; "
        f"D_s(p0=3.70) decreasing in D_r: {sd['D_s_decreasing_in_dr']}\n"
        f"  - SPV FIRE sweep : <q>(p0) r={sf['pearson_r']:.4f} (RMSE={sf['rmse']:.4f}), solid-branch <q>={sf['solid_mean_q']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
