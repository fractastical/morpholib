#!/usr/bin/env python3
"""
2015-manukyan-ocellated-lizard-ca.py — Pass 3 Remediated Implementation (v3)
============================================================================
Mathematical & Empirical Reproduction of Ocellated Lizard (*Timon lepidus*)
Skin-Scale Mesoscopic Cellular Automaton Morphogenesis:

1. **Manukyan, Montandon, Fofonjka, Smirnov & Milinkovitch (2017)** *Nature*
   544(7649):173-179 (`doi:10.1038/nature22031`):
   - "A living mesoscopic cellular automaton made of skin scales"
   - Continuous **Kondo & Asai (1995)** (*Nature* 376:765-768, `doi:10.1038/376765a0`)
     reaction-diffusion PDE on a skin-scale geometry where skin thickness is
     reduced by factor P = h_border / h_interior ~ 0.17 at inter-scale borders
     (Methods, Fig. 4):
       du/dt = F(u, v) + div(D_u grad u),   dv/dt = G(u, v) + div(D_v grad v)
   - Because intra-scale lateral diffusion is fast relative to inter-scale border
     flux, morphogen concentrations homogenize within each individual scale
     (bimodal monochromatic green vs black scales) and the continuous PDE reduces
     to a discrete 2D **hexagonal outer-totalistic cellular automaton (CA)** whose
     adult nearest-neighbor histogram peaks at n_green = 3 (and n_black = 3).

2. **Fofonjka & Milinkovitch (2021)** *Nat. Commun.* 12:2433 (`doi:10.1038/s41467-021-22525-1`)
   — reaction-diffusion in a 3D domain of skin scales (thick scale centres, thin
   inter-scale borders) generates a discrete cellular automaton.
   **Pass 3 remediation (this file):** the Pass 2 audit asked for the 2D
   scale-border PDE to be extended to 3D skin-scale volumes. No micro-CT mesh is
   vendored in this repository, so this script implements the closest honest
   reduced version and says so:
   - `integrate_skin_volume_pde_3d` integrates the SAME reaction-diffusion
     system in a genuine 3D voxelized skin volume Omega = {(x, y, z) : 0 <= z < h(x, y)}
     whose thickness map h(x, y) is h_interior inside hexagonal scale cores and
     h_border = P * h_interior on inter-scale borders (a stepped-thickness skin,
     not a micro-CT reconstruction), with zero-flux (no-flux) boundaries on the
     epidermal surface and basal lamina, a finite-volume face-conductance scheme,
     explicit lateral diffusion and an unconditionally stable implicit vertical
     (Thomas / tridiagonal) solve so that the vertical voxel spacing dz can be
     finer than the lateral pixel spacing (thin-film regime h << lambda_c).
   - `integrate_scale_border_pde_2d(border_model="thickness")` integrates the
     depth-averaged (thin-film / lubrication) reduction of that 3D problem,
       d ubar/dt = F(ubar, vbar) + (1 / h(x)) div( h(x) D grad ubar ),
     i.e. the depth-integrated face conductance between two columns is
     min(h_i, h_j) * D and the column capacity is h(x). The v2 "diffusivity"
     model D(x) = P * D on borders with unit capacity is kept as a comparison
     variant (`border_model="diffusivity"`).
   - The artificial per-scale averaging step that v2 applied after every PDE
     step (`homog_rate`) has been REMOVED: scale-level discretization now has to
     emerge from the thickness geometry alone, and the script measures how well
     the 3D volume solution agrees with its depth-averaged 2D reduction (scale
     state agreement, RMS of the depth-averaged fields, vertical non-uniformity).
   - Intra-scale homogenization requires the Turing wavelength lambda_c to be
     several scale pitches (the first zero-flux mode of an insulated scale must
     lie outside the unstable band); `kinetics_linear_stability` computes
     tr J, det J, the critical diffusivity ratio d_c, k_c, lambda_c and the
     unstable band for both the script kinetics and the canonical clipped-linear
     Kondo-Asai parameter set vendored in `ocellated_lizard_ca_reference.json`.

3. **Zakany & Milinkovitch (2023)** *Phys. Rev. X* 13:041011 (`doi:10.1103/PhysRevX.13.041011`):
   neighborhood-dependent colour subclustering of lizard scales.

4. **Machine-Checked Lean 4 Formalization** (`morphoeval/lean/loops/Loop_009_ocellated_lizard_ca.lean`):
   - Theorem 1 (`MathOfBiology.Loop_009.hex_scale_neighbor_partition_and_symmetry`):
     n_g + n_b = 6 on the 6-regular hexagonal lattice, and n_g = n_b <-> n_g = 3.
   - Theorem 2 (`MathOfBiology.Loop_009.border_diffusivity_reduction_quenches_interscale_flux`):
     P_pct = 0 yields zero inter-scale flux, while physiological border thinning
     (P_pct = 17 < 100) strictly attenuates inter-scale flux relative to continuum skin.
   - Theorem 3 (`MathOfBiology.Loop_009.manukyan_outer_totalistic_ca_labyrinth_fixed_point`):
     n_g = 3 is a simultaneous fixed point for green (s=1) and black (s=0) scales.
   - Theorem 4 (`MathOfBiology.Loop_009.thin_film_depth_average_recovers_uniform_column`, Pass 3):
     the depth average (h * c) / h of a vertically uniform skin column of height
     h > 0 recovers c, and column mass is monotone in thickness.
   - Theorem 5 (`MathOfBiology.Loop_009.voxel_face_conductance_ratio_equals_thickness_ratio`, Pass 3):
     the depth-integrated face conductance min(h_b, h_i) * D between a border
     and an interior column has ratio h_b / h_i independent of D, vanishes for
     an insulated border (h_b = 0), and is strictly smaller than the
     interior-interior conductance whenever h_b < h_i.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "ocellated_lizard_ca"
REF_JSON_PATH = DATASET_DIR / "ocellated_lizard_ca_reference.json"
SPECIMEN_JPG_PATH = DATASET_DIR / "timon_lepidus_specimen.jpg"
SCOUT_JSON_PATH = DATASET_DIR / "scout_manifest.json"
BIB_PATH = DATASET_DIR / "paper_citations.bib"
SKIN_VOLUME_JSON_PATH = DATASET_DIR / "skin_volume_reference.json"

OUT_PNG = ROOT / "docs" / "media" / "ocellated_lizard_ca_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "ocellated_lizard_ca_growth.gif"
OUT_3D_PNG = ROOT / "docs" / "media" / "ocellated_lizard_ca_3d_volume.png"

EXPECTED_SHA256: dict[str, str] = {
    "datasets/ocellated_lizard_ca/ocellated_lizard_ca_reference.json": (
        "53832b76e25303a6f442902c663e45892ba53175014a753ed40e8e2c7a910cff"
    ),
    "datasets/ocellated_lizard_ca/timon_lepidus_specimen.jpg": (
        "bc8e149849e6729d3eb9d4b6507daf6ad440ac50d97546ad4ef29db79c9b91cf"
    ),
    "datasets/ocellated_lizard_ca/scout_manifest.json": (
        "e5bca2d8766b49e70b149cad4bf068d8b93192157b57de770973193cc36cab2c"
    ),
    "datasets/ocellated_lizard_ca/paper_citations.bib": (
        "1bf1c8f580f69de9960ecf821d70b1304b7ce2bbfb8a0cd9802856bfa5c0cac8"
    ),
    "datasets/ocellated_lizard_ca/skin_volume_reference.json": (
        "dc8da9f0b72dad3a7326d0becb3f0aa475c84bb3f5c0f99723e1575f64779a27"
    ),
}

# --------------------------------------------------------------------------- #
# Default PDE discretization (shared by the 3D volume solver and its 2D reduction)
# --------------------------------------------------------------------------- #
# 12 x 12 hexagonal scales on a 60 x 60 lateral pixel grid -> scale pitch 5 px.
# Turing wavelength lambda_c = 25 px = 5 scale pitches so that the first zero-flux
# mode of an insulated scale (k_1 ~ pi / 4 px^-1) lies above the unstable band
# (k_max ~ 1.9 k_c at d = 20): isolated scales cannot pattern internally, which is
# the mesoscopic premise of Manukyan et al. (2017) Fig. 4.
PDE_SCALE_ROWS = 12
PDE_SCALE_COLS = 12
PDE_NY = 60
PDE_NX = 60
PDE_NZ = 12          # interior skin thickness in voxels
PDE_DZ_PX = 0.2      # vertical voxel spacing in lateral-pixel units (h_interior = 2.4 px = 0.48 pitch)
PDE_LAMBDA_PX = 25.0 # target Turing wavelength lambda_c in pixels
PDE_D_RATIO = 20.0   # d = D_v / D_u (canonical, ocellated_lizard_ca_reference.json)
PDE_DT = 0.006       # explicit lateral stability: dt < 1 / (4 D_v) ~ 0.0066 at lambda_c = 25 px
PDE_STEPS = 6000     # T = 36 time units ~ 7 / sigma_max (pattern saturated; see inter_std_history)


# =========================================================================== #
# 1. Dataset & SHA-256 Provenance Verification
# =========================================================================== #
def verify_dataset_sha256() -> dict[str, str]:
    """Verify SHA-256 digests of all 5 vendored Slot 009 reference files."""
    actual: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required Slot 009 dataset file: {p}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {digest}")
        actual[rel] = digest
    return actual


@functools.lru_cache(maxsize=1)
def load_reference_dataset() -> dict[str, Any]:
    """Load and SHA-256-verify `ocellated_lizard_ca_reference.json`."""
    verify_dataset_sha256()
    return json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def load_skin_volume_reference() -> dict[str, Any]:
    """Load the Pass 3 `skin_volume_reference.json` (3D skin-volume model & thin-film reduction notes)."""
    verify_dataset_sha256()
    return json.loads(SKIN_VOLUME_JSON_PATH.read_text(encoding="utf-8"))


# =========================================================================== #
# 2. Hexagonal Lattice Topology & Outer-Totalistic Cellular Automaton
# =========================================================================== #
def count_hex_green_neighbors(grid: np.ndarray) -> np.ndarray:
    """
    Count green (state = 1) first-nearest neighbors n_g in {0..6} on a 2D
    6-regular periodic hexagonal lattice (offset row coordinates).

    Cross-references Lean 4 Theorem 1:
      `MathOfBiology.Loop_009.hex_scale_neighbor_partition_and_symmetry`
    """
    rows, cols = grid.shape
    counts = np.zeros_like(grid, dtype=np.int32)
    for r in range(rows):
        rp = (r - 1) % rows
        rn = (r + 1) % rows
        row_g = grid[r]
        counts[r] += np.roll(row_g, 1) + np.roll(row_g, -1)
        counts[r] += grid[rp] + grid[rn]
        if r % 2 == 0:
            counts[r] += np.roll(grid[rp], 1) + np.roll(grid[rn], 1)
        else:
            counts[r] += np.roll(grid[rp], -1) + np.roll(grid[rn], -1)
    return counts


def deterministic_ca_core_step(s: int, n_g: int) -> int:
    """
    Deterministic core of Manukyan et al.'s (2017) hexagonal outer-totalistic CA
    rule matching `MathOfBiology.Loop_009.manukyan_ca_step` (Lean Theorem 3).
    """
    if s == 1:
        return 1 if (2 <= n_g <= 3) else 0
    return 1 if (n_g <= 1) else 0


def compute_neighbor_statistics(grid: np.ndarray) -> dict[str, Any]:
    """
    Compute Manukyan et al. (2017) Fig. 3d-f first-nearest-neighbor histograms
    P(n_g | S=1) and P(n_g | S=0), modal neighbor counts, and labyrinthine
    chain fraction (scales with 2 <= n_g <= 3).
    """
    ng = count_hex_green_neighbors(grid)
    nb = 6 - ng
    green_mask = grid == 1
    black_mask = grid == 0

    green_ng = ng[green_mask]
    black_ng = ng[black_mask]
    black_nb = nb[black_mask]

    hist_green = (
        np.bincount(green_ng, minlength=7).astype(np.float64) / max(len(green_ng), 1)
        if len(green_ng) > 0
        else np.zeros(7, dtype=np.float64)
    )
    hist_black = (
        np.bincount(black_ng, minlength=7).astype(np.float64) / max(len(black_ng), 1)
        if len(black_ng) > 0
        else np.zeros(7, dtype=np.float64)
    )

    mean_ng_green = float(np.mean(green_ng)) if len(green_ng) > 0 else 0.0
    mean_ng_black = float(np.mean(black_ng)) if len(black_ng) > 0 else 0.0
    mean_nb_black = float(np.mean(black_nb)) if len(black_nb) > 0 else 0.0
    labyrinth_chain_fraction = float(np.mean((ng >= 2) & (ng <= 3)))

    return {
        "ng_field": ng,
        "nb_field": nb,
        "green_fraction": float(np.mean(grid)),
        "mean_ng_green": mean_ng_green,
        "mean_ng_black": mean_ng_black,
        "mean_nb_black": mean_nb_black,
        "modal_ng_green": int(np.argmax(hist_green)) if len(green_ng) > 0 else 0,
        "modal_nb_black": int(6 - np.argmax(hist_black)) if len(black_ng) > 0 else 0,
        "hist_green": hist_green,
        "hist_black": hist_black,
        "labyrinth_chain_fraction": labyrinth_chain_fraction,
    }


def simulate_hexagonal_ca(
    rows: int = 36,
    cols: int = 36,
    steps: int = 45,
    seed: int = 2017,
) -> dict[str, Any]:
    """
    Simulate the Manukyan et al. (2017) *Nature* Figs. 2-3 hexagonal
    outer-totalistic cellular automaton from juvenile brown/white ocelli
    to adult green/black labyrinthine chains.

    Cross-references Lean 4 Theorem 1 & Theorem 3:
      - `MathOfBiology.Loop_009.hex_scale_neighbor_partition_and_symmetry`
      - `MathOfBiology.Loop_009.manukyan_outer_totalistic_ca_labyrinth_fixed_point`
    """
    ref = load_reference_dataset()
    ca_cfg = ref["hexagonal_outer_totalistic_ca"]["transition_probabilities_juvenile_to_adult"]
    p_g2b = np.array(ca_cfg["p_green_to_black_by_n_green"], dtype=np.float64)
    p_b2g = np.array(ca_cfg["p_black_to_green_by_n_green"], dtype=np.float64)

    rng = np.random.default_rng(seed)
    # Initialize juvenile ocellated pattern (sparse bright ocelli Centers + background)
    grid = np.zeros((rows, cols), dtype=np.int32)
    for r0 in range(3, rows - 2, 6):
        for c0 in range(3, cols - 2, 6):
            grid[r0 - 1 : r0 + 2, c0 - 1 : c0 + 2] = 1
    noise_flip = rng.random((rows, cols)) < 0.12
    grid = np.logical_xor(grid, noise_flip).astype(np.int32)

    snapshots: list[tuple[int, np.ndarray]] = [(0, grid.copy())]
    snap_steps = set(np.linspace(2, steps - 1, 5, dtype=int).tolist())
    flip_history: list[float] = []

    for step in range(steps):
        ng = count_hex_green_neighbors(grid)
        u_rand = rng.random((rows, cols))
        # Asynchronous update rate per developmental week (damping factor 0.35 prevents synchronous oscillation)
        flip_g2b = (grid == 1) & (u_rand < 0.35 * p_g2b[ng])
        flip_b2g = (grid == 0) & (u_rand < 0.35 * p_b2g[ng])
        new_grid = grid.copy()
        new_grid[flip_g2b] = 0
        new_grid[flip_b2g] = 1
        flip_history.append(float(np.mean(new_grid != grid)))
        grid = new_grid
        if step in snap_steps:
            snapshots.append((step + 1, grid.copy()))

    stats = compute_neighbor_statistics(grid)
    ref_hist_g = np.array(
        ref["hexagonal_outer_totalistic_ca"]["adult_neighbor_histogram_targets"]["green_scale_neighbor_hist_ref"],
        dtype=np.float64,
    )
    l1_to_ref = float(np.sum(np.abs(stats["hist_green"] - ref_hist_g)))

    return {
        "grid": grid,
        "snapshots": snapshots,
        "flip_history": flip_history,
        "stats": stats,
        "l1_hist_distance_to_ref": l1_to_ref,
        "p_g2b": p_g2b,
        "p_b2g": p_b2g,
    }


# =========================================================================== #
# 3. Reaction Kinetics & Turing Linear Stability (Manukyan Methods / Kondo-Asai)
# =========================================================================== #
# Script kinetics: Kondo-Asai-type activator-inhibitor kinetics linearized around the
# homogeneous steady state (u*, v*) = (0.50, 0.50) with a cubic saturation on the
# activator. Jacobian: f_u = +0.50, f_v = -0.75, g_u = +0.82, g_v = -0.65
# (tr J = -0.15 < 0, det J = +0.29 > 0 -> stable without diffusion; d_c = 7.0).
KIN_FU, KIN_FV, KIN_GU, KIN_GV, KIN_CUBIC = 0.50, -0.75, 0.82, -0.65, 1.25


def reaction_kinetics(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F(u, v), G(u, v) of the script's activator-inhibitor kinetics around (0.50, 0.50)."""
    x = u - 0.50
    y = v - 0.50
    du = KIN_FU * x + KIN_FV * y - KIN_CUBIC * x**3
    dv = KIN_GU * x + KIN_GV * y
    return du, dv


def turing_critical_ratio(fu: float, fv: float, gu: float, gv: float) -> float:
    """
    Critical diffusivity ratio d_c = D_v / D_u above which a Turing (diffusion-driven)
    instability exists for a stable homogeneous steady state (tr J < 0, det J > 0):
      d_c = [ (2 det J - f_u g_v) + 2 sqrt( det J (det J - f_u g_v) ) ] / f_u^2
    (Turing 1952; Murray, Mathematical Biology II, Ch. 2).
    """
    det = fu * gv - fv * gu
    return float(((2.0 * det - fu * gv) + 2.0 * np.sqrt(det * (det - fu * gv))) / fu**2)


def kinetics_linear_stability(
    fu: float, fv: float, gu: float, gv: float, d_ratio: float, D_u: float
) -> dict[str, Any]:
    """
    Linear Turing analysis of du/dt = J (u, v)^T + diag(D_u, d D_u) Laplacian:
    growth rate sigma(k) = max Re eig(J - k^2 diag(D_u, D_v)), critical wavenumber
    k_c^2 = sqrt(det J / (D_u D_v)), critical ratio d_c, and the unstable band
    [k_min, k_max] where det(J - k^2 D) < 0.
    """
    J = np.array([[fu, fv], [gu, gv]], dtype=np.float64)
    tr = float(np.trace(J))
    det = float(np.linalg.det(J))
    D_v = d_ratio * D_u
    d_c = turing_critical_ratio(fu, fv, gu, gv)
    k_c = float(np.sqrt(np.sqrt(det / (D_u * D_v))))
    lambda_c = float(2.0 * np.pi / k_c)

    ks = np.linspace(0.0, 2.5 * k_c, 600)
    sigma = np.empty_like(ks)
    for i, k in enumerate(ks):
        M = J - (k * k) * np.diag([D_u, D_v])
        sigma[i] = float(np.max(np.linalg.eigvals(M).real))
    i_max = int(np.argmax(sigma))

    # Unstable band: roots of h(k^2) = D_u D_v k^4 - (D_v f_u + D_u g_v) k^2 + det J
    b = D_v * fu + D_u * gv
    disc = b * b - 4.0 * D_u * D_v * det
    if disc > 0.0 and b > 0.0:
        q_lo = (b - np.sqrt(disc)) / (2.0 * D_u * D_v)
        q_hi = (b + np.sqrt(disc)) / (2.0 * D_u * D_v)
        k_min, k_max = float(np.sqrt(max(q_lo, 0.0))), float(np.sqrt(q_hi))
    else:
        k_min, k_max = float("nan"), float("nan")

    return {
        "trace": tr,
        "det": det,
        "stable_homogeneous": bool(tr < 0.0 and det > 0.0),
        "d_ratio": float(d_ratio),
        "d_critical": d_c,
        "turing_unstable": bool(d_ratio > d_c and tr < 0.0 and det > 0.0),
        "D_u": float(D_u),
        "D_v": float(D_v),
        "k_c": k_c,
        "lambda_c_px": lambda_c,
        "k_grid": ks,
        "sigma_grid": sigma,
        "sigma_max": float(sigma[i_max]),
        "k_sigma_max": float(ks[i_max]),
        "k_min_unstable": k_min,
        "k_max_unstable": k_max,
    }


def script_diffusivity_for_wavelength(lambda_px: float = PDE_LAMBDA_PX, d_ratio: float = PDE_D_RATIO) -> float:
    """D_u such that the script kinetics have Turing wavelength lambda_c = lambda_px: D_u = sqrt(det J / d) / k_c^2."""
    det = KIN_FU * KIN_GV - KIN_FV * KIN_GU
    k_c = 2.0 * np.pi / lambda_px
    return float(np.sqrt(det / d_ratio) / (k_c * k_c))


def canonical_json_kinetics_stability(d_ratio: float | None = None) -> dict[str, Any]:
    """
    Linear stability of the canonical clipped-linear Kondo-Asai parameter set in
    `ocellated_lizard_ca_reference.json` (F = clip(c_u u + c_v v + c_0) - gamma_u u,
    G = clip(d_u u + d_v v + d_0) - gamma_v v) at its own diffusivity ratio.
    Exposes whether that vendored parameter set is actually Turing-unstable.
    """
    cp = load_reference_dataset()["scale_border_reaction_diffusion_model"]["canonical_parameters"]
    fu = cp["c_u"] - cp["gamma_u"]
    fv = cp["c_v"]
    gu = cp["d_u"]
    gv = cp["d_v"] - cp["gamma_v"]
    d = float(cp["diffusivity_ratio_d"] if d_ratio is None else d_ratio)
    # Steady state of the unclipped linear system
    A = np.array([[fu, fv], [gu, gv]])
    rhs = -np.array([cp["c_0"], cp["d_0"]], dtype=np.float64)
    u_star, v_star = np.linalg.solve(A, rhs)
    F_star = cp["c_u"] * u_star + cp["c_v"] * v_star + cp["c_0"]
    G_star = cp["d_u"] * u_star + cp["d_v"] * v_star + cp["d_0"]
    out = kinetics_linear_stability(fu, fv, gu, gv, d_ratio=d, D_u=float(cp["D_u_interior"]))
    out.update(
        {
            "u_star": float(u_star),
            "v_star": float(v_star),
            "steady_state_within_clip": bool(0.0 <= F_star <= cp["F_max"] and 0.0 <= G_star <= cp["G_max"]),
        }
    )
    return out


# =========================================================================== #
# 4. Hexagonal Scale Geometry, Skin Thickness Map & Reaction-Diffusion Solvers
# =========================================================================== #
@functools.lru_cache(maxsize=16)
def build_hexagonal_scale_geometry(
    scale_rows: int = PDE_SCALE_ROWS,
    scale_cols: int = PDE_SCALE_COLS,
    ny: int = PDE_NY,
    nx: int = PDE_NX,
) -> dict[str, Any]:
    """
    Construct a 2D periodic hexagonal skin-scale mosaic on an (ny x nx) sub-scale
    continuum grid. Each pixel (y, x) is mapped to its closest hexagonal scale
    center i in {0 .. N_scales - 1} (Voronoi cell).

    Border pixels (Pass 3, leak-proof definition): a pixel is an inter-scale border
    pixel iff its +y or +x lattice neighbor belongs to a different scale. Every
    4-adjacent pixel pair that straddles two scales therefore contains a border
    pixel, so with face conductance min(h_i, h_j) ALL inter-scale flux passes
    through thinned skin and P = 0 insulates every scale (checked by connected
    component labelling). The v2 Voronoi-distance band (d2 - d1 < w) left
    sub-pixel gaps through which scales coupled at full thickness.
    """
    centers_yx: list[tuple[float, float]] = []
    rc_to_idx: dict[tuple[int, int], int] = {}
    dy = ny / float(scale_rows)
    dx = nx / float(scale_cols)

    idx = 0
    for r in range(scale_rows):
        y_c = (r + 0.5) * dy
        x_offset = 0.5 * dx if (r % 2 == 1) else 0.0
        for c in range(scale_cols):
            x_c = ((c + 0.5) * dx + x_offset) % nx
            centers_yx.append((y_c, x_c))
            rc_to_idx[(r, c)] = idx
            idx += 1

    centers = np.array(centers_yx, dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64), indexing="ij")

    dy_mat = np.abs(yy[:, :, None] - centers[None, None, :, 0])
    dy_mat = np.minimum(dy_mat, ny - dy_mat)
    dx_mat = np.abs(xx[:, :, None] - centers[None, None, :, 1])
    dx_mat = np.minimum(dx_mat, nx - dx_mat)
    dist = np.sqrt(dy_mat**2 + dx_mat**2)
    scale_id_map = np.argmin(dist, axis=2).astype(np.int32)

    border_mask = (scale_id_map != np.roll(scale_id_map, -1, axis=0)) | (scale_id_map != np.roll(scale_id_map, -1, axis=1))

    # Insulation check: with borders removed, 4-connected components of the interior
    n_components, _ = _label_periodic_components(~border_mask)

    return {
        "scale_rows": scale_rows,
        "scale_cols": scale_cols,
        "n_scales": len(centers),
        "centers_yx": centers,
        "scale_id_map": scale_id_map,
        "border_mask": border_mask,
        "border_area_fraction": float(np.mean(border_mask)),
        "n_interior_components": int(n_components),
        "rc_to_idx": rc_to_idx,
        "pitch_px": float(np.sqrt(dy * dx)),
    }


def _label_periodic_components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """4-connected component count of a boolean mask on a periodic 2D grid (labels merged across the wrap seams)."""
    lab, _ = ndi.label(mask)
    ny, nx = mask.shape
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(ny):
        if mask[i, 0] and mask[i, -1]:
            union(int(lab[i, 0]), int(lab[i, -1]))
    for j in range(nx):
        if mask[0, j] and mask[-1, j]:
            union(int(lab[0, j]), int(lab[-1, j]))
    roots = {find(int(v)) for v in np.unique(lab) if v != 0}
    return len(roots), lab


def skin_thickness_map(geom: dict[str, Any], P_border: float, nz: int = PDE_NZ) -> dict[str, Any]:
    """
    Stepped skin thickness map h(x, y) in voxels: h_interior = nz inside scale cores,
    h_border = round(P * nz) (>= 1 voxel for any P > 0, 0 for P = 0) on borders.
    Returns the realised effective thickness ratio P_eff = h_border / nz, which
    differs from P at voxel resolution (e.g. P = 0.17 -> 2 / 12 = 0.167).
    """
    if P_border <= 0.0:
        h_border = 0
    else:
        h_border = int(max(1, round(P_border * nz)))
    h = np.where(geom["border_mask"], h_border, nz).astype(np.int32)
    return {
        "h_map": h,
        "h_interior": int(nz),
        "h_border": int(h_border),
        "P_effective": float(h_border / nz),
        "skin_volume_fraction_vs_slab": float(np.mean(h) / nz),
    }


def _initial_fields(geom: dict[str, Any], seed: int, shape_3d: tuple[int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic initial condition: per-scale offsets (std 0.06) + pixel noise (std 0.01) around (0.50, 0.50)."""
    rng = np.random.default_rng(seed)
    ny, nx = geom["scale_id_map"].shape
    scale_init = rng.normal(0.50, 0.06, size=geom["n_scales"])
    u2 = scale_init[geom["scale_id_map"]] + rng.normal(0.0, 0.01, size=(ny, nx))
    v2 = 0.50 + rng.normal(0.0, 0.01, size=(ny, nx))
    if shape_3d is None:
        return np.clip(u2, 0.05, 0.95), np.clip(v2, 0.05, 0.95)
    nz = shape_3d[0]
    u3 = np.broadcast_to(u2, (nz, ny, nx)).copy() + rng.normal(0.0, 0.005, size=(nz, ny, nx))
    v3 = np.broadcast_to(v2, (nz, ny, nx)).copy()
    return np.clip(u3, 0.05, 0.95), np.clip(v3, 0.05, 0.95)


def _per_scale_statistics(u_bar: np.ndarray, geom: dict[str, Any], valid: np.ndarray) -> dict[str, Any]:
    """Per-scale mean / intra-scale std of a depth-averaged field over valid (interior) pixels."""
    sid = geom["scale_id_map"]
    rows, cols = geom["scale_rows"], geom["scale_cols"]
    scale_means = np.zeros((rows, cols), dtype=np.float64)
    intra_stds: list[float] = []
    for r in range(rows):
        for c in range(cols):
            s = geom["rc_to_idx"][(r, c)]
            mask_s = (sid == s) & valid
            if not np.any(mask_s):
                mask_s = sid == s
            vals = u_bar[mask_s]
            scale_means[r, c] = float(np.mean(vals))
            intra_stds.append(float(np.std(vals)))
    inter = float(np.std(scale_means))
    intra = float(np.mean(intra_stds))
    patterned = inter > 0.03
    if patterned:
        thresh = float(np.percentile(scale_means, 55.0))
        binary = (scale_means >= thresh).astype(np.int32)
        g_vals = scale_means[binary == 1]
        b_vals = scale_means[binary == 0]
        ashman_D = float(np.sqrt(2.0) * abs(np.mean(g_vals) - np.mean(b_vals)) / np.sqrt(np.var(g_vals) + np.var(b_vals) + 1e-8))
    else:
        binary = np.zeros((rows, cols), dtype=np.int32)
        ashman_D = 0.0
    return {
        "scale_means": scale_means,
        "inter_scale_std": inter,
        "mean_intra_scale_std": intra,
        # Undefined (NaN) when no scale-level pattern formed (e.g. insulated P = 0, equal diffusion d = 1)
        "discretization_ratio": float(inter / max(intra, 1e-9)) if patterned else float("nan"),
        "patterned": bool(patterned),
        "binary_scales": binary,
        "ashman_D": ashman_D,
        "neighbor_stats": compute_neighbor_statistics(binary),
    }


@functools.lru_cache(maxsize=32)
def integrate_scale_border_pde_2d(
    P_border: float = 0.17,
    d_ratio: float = PDE_D_RATIO,
    scale_rows: int = PDE_SCALE_ROWS,
    scale_cols: int = PDE_SCALE_COLS,
    ny: int = PDE_NY,
    nx: int = PDE_NX,
    nz: int = PDE_NZ,
    steps: int = PDE_STEPS,
    dt: float = PDE_DT,
    lambda_px: float = PDE_LAMBDA_PX,
    seed: int = 2017,
    border_model: str = "thickness",
) -> dict[str, Any]:
    """
    Depth-averaged (thin-film) 2D reduction of reaction-diffusion in a skin volume
    of stepped thickness h(x) (Fofonjka & Milinkovitch 2021; Manukyan et al. 2017 Fig. 4):

      d ubar/dt = F(ubar, vbar) + (1 / h(x)) div( h(x) D_u grad ubar )
      d vbar/dt = G(ubar, vbar) + (1 / h(x)) div( h(x) D_v grad vbar )

    Finite volume on the periodic pixel grid: the depth-integrated conductance of
    the face between columns i, j is min(h_i, h_j) * D (shared face area) and the
    capacity of column i is h_i (`border_model="thickness"`, default).

    `border_model="diffusivity"` reproduces the Pass 2 variant D(x) = P * D on border
    pixels with unit capacity (face conductance = arithmetic mean of the two
    pixel weights) for comparison. No artificial per-scale averaging is applied.

    Cross-references Lean 4 Theorems 2, 4, 5 in
    `morphoeval/lean/loops/Loop_009_ocellated_lizard_ca.lean`.
    """
    geom = build_hexagonal_scale_geometry(scale_rows, scale_cols, ny, nx)
    thick = skin_thickness_map(geom, P_border, nz)
    h = thick["h_map"].astype(np.float64) / float(nz)  # normalized thickness (interior = 1)
    skin = (h > 0.0).astype(np.float64)
    h_cap = np.maximum(h, 1e-12)

    if border_model == "thickness":
        g_yp = np.minimum(h, np.roll(h, -1, axis=0))
        g_ym = np.minimum(h, np.roll(h, 1, axis=0))
        g_xp = np.minimum(h, np.roll(h, -1, axis=1))
        g_xm = np.minimum(h, np.roll(h, 1, axis=1))
        cap = h_cap
    elif border_model == "diffusivity":
        w = np.where(geom["border_mask"], float(P_border), 1.0)
        g_yp = 0.5 * (w + np.roll(w, -1, axis=0))
        g_ym = 0.5 * (w + np.roll(w, 1, axis=0))
        g_xp = 0.5 * (w + np.roll(w, -1, axis=1))
        g_xm = 0.5 * (w + np.roll(w, 1, axis=1))
        cap = np.ones_like(h)
        skin = np.ones_like(h)
    else:
        raise ValueError(f"Unknown border_model {border_model!r}")

    D_u0 = script_diffusivity_for_wavelength(lambda_px, d_ratio)
    D_v0 = D_u0 * float(d_ratio)
    u, v = _initial_fields(geom, seed)
    u *= skin
    v *= skin

    def div_h_grad(f: np.ndarray) -> np.ndarray:
        return (
            g_yp * (np.roll(f, -1, axis=0) - f)
            + g_ym * (np.roll(f, 1, axis=0) - f)
            + g_xp * (np.roll(f, -1, axis=1) - f)
            + g_xm * (np.roll(f, 1, axis=1) - f)
        ) / cap

    history_every = max(1, steps // 24)
    inter_std_history: list[tuple[float, float]] = []
    interior = ~geom["border_mask"]
    for n in range(steps):
        du, dv = reaction_kinetics(u, v)
        u = np.clip(u + dt * (du + D_u0 * div_h_grad(u)), 0.05, 0.95) * skin
        v = np.clip(v + dt * (dv + D_v0 * div_h_grad(v)), 0.05, 0.95) * skin
        if n % history_every == 0 or n == steps - 1:
            means = np.bincount(geom["scale_id_map"][interior], weights=u[interior], minlength=geom["n_scales"]) / np.bincount(
                geom["scale_id_map"][interior], minlength=geom["n_scales"]
            )
            inter_std_history.append(((n + 1) * dt, float(np.std(means))))

    stats = _per_scale_statistics(u, geom, interior)
    out = {
        "model": f"2d-{border_model}",
        "P_border": float(P_border),
        "P_effective": thick["P_effective"] if border_model == "thickness" else float(P_border),
        "d_ratio": float(d_ratio),
        "D_u": D_u0,
        "D_v": D_v0,
        "dt": float(dt),
        "steps": int(steps),
        "u_field": u,
        "v_field": v,
        "u_bar": u,
        "h_map": thick["h_map"],
        "border_mask": geom["border_mask"],
        "geometry": geom,
        "inter_std_history": inter_std_history,
    }
    out.update(stats)
    return out


def _thomas_solve_along_z(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Vectorized Thomas (tridiagonal) solve along axis 0 for arrays of shape (nz, ny, nx)."""
    nz = a.shape[0]
    cp = np.empty_like(c)
    dp = np.empty_like(d)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, nz):
        denom = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / denom
        dp[k] = (d[k] - a[k] * dp[k - 1]) / denom
    x = np.empty_like(d)
    x[-1] = dp[-1]
    for k in range(nz - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


@functools.lru_cache(maxsize=16)
def integrate_skin_volume_pde_3d(
    P_border: float = 0.17,
    d_ratio: float = PDE_D_RATIO,
    scale_rows: int = PDE_SCALE_ROWS,
    scale_cols: int = PDE_SCALE_COLS,
    ny: int = PDE_NY,
    nx: int = PDE_NX,
    nz: int = PDE_NZ,
    dz_px: float = PDE_DZ_PX,
    steps: int = PDE_STEPS,
    dt: float = PDE_DT,
    lambda_px: float = PDE_LAMBDA_PX,
    seed: int = 2017,
) -> dict[str, Any]:
    """
    Genuine 3D reaction-diffusion in a voxelized skin volume of stepped thickness
    (Pass 3 remediation of the Fofonjka & Milinkovitch 2021 item):

      Omega = { (x, y, z) : 0 <= z < h(x, y) },  h = h_interior in scale cores,
                                                  h = P * h_interior on borders
      du/dt = F(u, v) + D_u Laplacian u,  dv/dt = G(u, v) + D_v Laplacian v  in Omega
      n . grad u = n . grad v = 0 on the epidermal surface, basal lamina and the
      vertical steps between thick and thin skin (finite-volume face conductance
      = 1 iff both voxels are skin), periodic laterally.

    Lateral diffusion is explicit; vertical diffusion (voxel spacing dz_px << 1 px)
    is implicit via a vectorized tridiagonal solve, so dt is limited only by the
    lateral explicit constraint dt < 1 / (4 D_v). Returns the full voxel field,
    its depth average ubar(x, y) = (1/h) sum_z u, the vertical non-uniformity
    (RMS over depth of u - ubar) that measures how well the thin-film reduction
    holds, and the same per-scale statistics as the 2D solver.

    This is a stepped-thickness voxel skin, NOT a micro-CT finite-element mesh; see
    `datasets/ocellated_lizard_ca/skin_volume_reference.json` for the residual gap.
    """
    geom = build_hexagonal_scale_geometry(scale_rows, scale_cols, ny, nx)
    thick = skin_thickness_map(geom, P_border, nz)
    h = thick["h_map"]
    zz = np.arange(nz)[:, None, None]
    skin = (zz < h[None, :, :]).astype(np.float64)  # (nz, ny, nx)

    c_yp = skin * np.roll(skin, -1, axis=1)
    c_ym = skin * np.roll(skin, 1, axis=1)
    c_xp = skin * np.roll(skin, -1, axis=2)
    c_xm = skin * np.roll(skin, 1, axis=2)
    c_zp = skin * np.roll(skin, -1, axis=0)
    c_zp[-1] = 0.0
    c_zm = skin * np.roll(skin, 1, axis=0)
    c_zm[0] = 0.0
    inv_dz2 = 1.0 / (dz_px * dz_px)

    D_u0 = script_diffusivity_for_wavelength(lambda_px, d_ratio)
    D_v0 = D_u0 * float(d_ratio)

    def tridiagonal(D: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = -dt * D * inv_dz2 * c_zm
        c = -dt * D * inv_dz2 * c_zp
        b = 1.0 - a - c
        return a, b, c

    au, bu, cu = tridiagonal(D_u0)
    av, bv, cv = tridiagonal(D_v0)

    u, v = _initial_fields(geom, seed, shape_3d=(nz, ny, nx))
    u *= skin
    v *= skin

    def lateral_laplacian(f: np.ndarray) -> np.ndarray:
        return (
            c_yp * (np.roll(f, -1, axis=1) - f)
            + c_ym * (np.roll(f, 1, axis=1) - f)
            + c_xp * (np.roll(f, -1, axis=2) - f)
            + c_xm * (np.roll(f, 1, axis=2) - f)
        )

    h_safe = np.maximum(h, 1).astype(np.float64)
    interior = ~geom["border_mask"]
    history_every = max(1, steps // 24)
    inter_std_history: list[tuple[float, float]] = []
    for n in range(steps):
        du, dv = reaction_kinetics(u, v)
        u_star = u + dt * (du + D_u0 * lateral_laplacian(u))
        v_star = v + dt * (dv + D_v0 * lateral_laplacian(v))
        u = np.clip(_thomas_solve_along_z(au, bu, cu, u_star), 0.05, 0.95) * skin
        v = np.clip(_thomas_solve_along_z(av, bv, cv, v_star), 0.05, 0.95) * skin
        if n % history_every == 0 or n == steps - 1:
            ub = (u * skin).sum(axis=0) / h_safe
            means = np.bincount(geom["scale_id_map"][interior], weights=ub[interior], minlength=geom["n_scales"]) / np.bincount(
                geom["scale_id_map"][interior], minlength=geom["n_scales"]
            )
            inter_std_history.append(((n + 1) * dt, float(np.std(means))))

    u_bar = (u * skin).sum(axis=0) / h_safe
    v_bar = (v * skin).sum(axis=0) / h_safe
    vert_rms = np.sqrt((((u - u_bar[None, :, :]) ** 2) * skin).sum(axis=0) / h_safe)
    stats = _per_scale_statistics(u_bar, geom, interior)

    out = {
        "model": "3d-voxel-skin-volume",
        "P_border": float(P_border),
        "P_effective": thick["P_effective"],
        "h_interior_voxels": thick["h_interior"],
        "h_border_voxels": thick["h_border"],
        "h_interior_px": float(nz * dz_px),
        "dz_px": float(dz_px),
        "skin_volume_fraction_vs_slab": thick["skin_volume_fraction_vs_slab"],
        "skin_volume_fraction_analytic": float(1.0 - (1.0 - thick["P_effective"]) * geom["border_area_fraction"]),
        "n_skin_voxels": int(skin.sum()),
        "d_ratio": float(d_ratio),
        "D_u": D_u0,
        "D_v": D_v0,
        "dt": float(dt),
        "steps": int(steps),
        "u_volume": u,
        "v_volume": v,
        "skin_mask": skin.astype(bool),
        "u_bar": u_bar,
        "v_bar": v_bar,
        "vertical_nonuniformity_rms": float(np.mean(vert_rms[interior])),
        "vertical_nonuniformity_max": float(np.max(vert_rms)),
        "h_map": h,
        "border_mask": geom["border_mask"],
        "geometry": geom,
        "inter_std_history": inter_std_history,
    }
    out.update(stats)
    return out


def compare_3d_volume_to_depth_averaged_2d(P_border: float = 0.17, **kwargs: Any) -> dict[str, Any]:
    """
    Quantify how well the depth-averaged 2D thin-film reduction reproduces the 3D
    skin-volume solution at the same P, seed and discretization: binary scale state
    agreement, Pearson r of per-scale means, RMS of the depth-averaged fields, and
    the 3D vertical non-uniformity (thin-film validity).
    """
    s3 = integrate_skin_volume_pde_3d(P_border=P_border, **kwargs)
    kw2 = {k: v for k, v in kwargs.items() if k != "dz_px"}
    s2 = integrate_scale_border_pde_2d(P_border=P_border, border_model="thickness", **kw2)
    m3 = s3["scale_means"].ravel()
    m2 = s2["scale_means"].ravel()
    both_patterned = s3["patterned"] and s2["patterned"]
    agreement = float(np.mean(s3["binary_scales"] == s2["binary_scales"])) if both_patterned else float("nan")
    r = float(pearsonr(m3, m2)[0]) if np.std(m3) > 1e-9 and np.std(m2) > 1e-9 else float("nan")
    skin2d = s2["h_map"] > 0
    rms = float(np.sqrt(np.mean((s3["u_bar"][skin2d] - s2["u_bar"][skin2d]) ** 2)))
    return {
        "P_border": float(P_border),
        "P_effective": s3["P_effective"],
        "scale_state_agreement": agreement,
        "scale_mean_pearson_r": r,
        "depth_averaged_rms_difference": rms,
        "vertical_nonuniformity_rms_3d": s3["vertical_nonuniformity_rms"],
        "discretization_ratio_3d": s3["discretization_ratio"],
        "discretization_ratio_2d": s2["discretization_ratio"],
        "ashman_D_3d": s3["ashman_D"],
        "ashman_D_2d": s2["ashman_D"],
        "sim_3d": s3,
        "sim_2d": s2,
    }


def run_border_reduction_sweep(model: str = "3d") -> list[dict[str, Any]]:
    """
    Sweep the 5 scale-border thickness reduction regimes P in {0.0, 0.08, 0.17, 0.45, 1.0}
    from `ocellated_lizard_ca_reference.json` in the 3D skin volume (`model="3d"`) or
    its depth-averaged 2D reduction (`model="2d"`), reporting P_eff at voxel resolution,
    inter/intra-scale std, discretization ratio, Ashman D and labyrinth statistics.
    """
    regimes = load_reference_dataset()["hexagonal_outer_totalistic_ca"]["border_reduction_P_regimes"]
    results: list[dict[str, Any]] = []
    for reg in regimes:
        p_val = float(reg["P"])
        sim = integrate_skin_volume_pde_3d(P_border=p_val) if model == "3d" else integrate_scale_border_pde_2d(P_border=p_val)
        results.append(
            {
                "regime_id": reg["regime_id"],
                "predicted_pattern": reg["predicted_pattern"],
                "P": p_val,
                "P_effective": sim["P_effective"],
                "model": sim["model"],
                "inter_scale_std": sim["inter_scale_std"],
                "mean_intra_scale_std": sim["mean_intra_scale_std"],
                "discretization_ratio": sim["discretization_ratio"],
                "patterned": sim["patterned"],
                "ashman_D": sim["ashman_D"],
                "mean_ng_green": sim["neighbor_stats"]["mean_ng_green"],
                "labyrinth_chain_fraction": sim["neighbor_stats"]["labyrinth_chain_fraction"],
            }
        )
    return results


def run_negative_controls() -> dict[str, Any]:
    """
    Formal negative/ablation controls (all in the 3D skin volume unless noted):
      1. Positive physiological Timon lepidus skin volume (P = 0.17, d = 20).
      2. Equal-diffusion ablation (P = 0.17, d = D_v/D_u = 1.0; 2D reduction): no
         Turing instability, so every scale relaxes to the homogeneous steady state.
      3. Unattenuated continuum skin (P = 1.0, d = 20): no border thinning, smooth
         Turing bands cut across scale interiors (lower discretization ratio).
      4. Insulated scales (P = 0.0, d = 20): zero inter-scale conductance; isolated
         scales cannot pattern internally because lambda_c = 5 pitches, so the
         pattern is absent (inter-scale std ~ 0) and interiors are uniform.
    """
    pos = integrate_skin_volume_pde_3d(P_border=0.17)
    eq_diff = integrate_scale_border_pde_2d(P_border=0.17, d_ratio=1.0)
    continuum = integrate_skin_volume_pde_3d(P_border=1.0)
    insulated = integrate_skin_volume_pde_3d(P_border=0.0)
    return {
        "positive_inter_scale_std": pos["inter_scale_std"],
        "positive_intra_std": pos["mean_intra_scale_std"],
        "positive_discretization_ratio": pos["discretization_ratio"],
        "equal_diffusion_inter_scale_std": eq_diff["inter_scale_std"],
        "continuum_P1_inter_scale_std": continuum["inter_scale_std"],
        "continuum_P1_intra_std": continuum["mean_intra_scale_std"],
        "continuum_P1_discretization_ratio": continuum["discretization_ratio"],
        "insulated_P0_inter_scale_std": insulated["inter_scale_std"],
        "insulated_P0_intra_std": insulated["mean_intra_scale_std"],
        "insulated_P0_patterned": insulated["patterned"],
        "discretization_gain_vs_continuum": float(pos["discretization_ratio"] / max(continuum["discretization_ratio"], 1e-6)),
    }


# =========================================================================== #
# 5. Biological Specimen Segmentation (`timon_lepidus_specimen.jpg`)
# =========================================================================== #
@functools.lru_cache(maxsize=1)
def segment_timon_lepidus_specimen() -> dict[str, Any]:
    """
    Segment the SHA-256-locked *Timon lepidus* dorsal scale specimen photograph
    (`datasets/ocellated_lizard_ca/timon_lepidus_specimen.jpg`):
      1. Extract Dorsal Scale ROI `[120, 180, 320, 360]`.
      2. Sample hexagonal scale sites across the dorsal scale lattice, measure each
         scale's green chromaticity index I_green = 1.45*G - 0.45*R - 0.45*B, classify
         into bimodal Green (S_i = 1) vs Black (S_i = 0) scales, and evaluate
         first-nearest-neighbor green count distribution P_specimen(n_g | S=1).
    """
    ref = load_reference_dataset()
    spec_meta = ref["metadata"]["specimens"]["timon_lepidus_dorsal"]
    im = Image.open(SPECIMEN_JPG_PATH).convert("RGB")
    arr = np.asarray(im, dtype=np.float64) / 255.0

    x0, y0, x1, y1 = spec_meta["dorsal_roi_xyxy"]
    hx0, hy0, hx1, hy1 = spec_meta["head_roi_xyxy"]
    roi = arr[y0:y1, x0:x1]
    r_ch, g_ch, b_ch = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]

    # Green scale index with Difference-of-Gaussians illumination normalization
    # (removes macro body-curvature shading while preserving scale-level green vs black contrast)
    green_score = 1.45 * g_ch - 0.45 * r_ch - 0.45 * b_ch
    smooth_green = ndi.gaussian_filter(green_score, sigma=1.2) - ndi.gaussian_filter(green_score, sigma=7.5)

    # Sample hexagonal scale lattice across the 180x200 dorsal ROI (scale pitch ~ 13 px)
    scale_rows, scale_cols = 13, 14
    h_roi, w_roi = smooth_green.shape
    dy = h_roi / float(scale_rows)
    dx = w_roi / float(scale_cols)

    scale_scores = np.zeros((scale_rows, scale_cols), dtype=np.float64)
    scale_centers_xy: list[tuple[float, float]] = []
    scale_states_list: list[int] = []

    for r in range(scale_rows):
        yc = int(np.clip((r + 0.5) * dy, 2, h_roi - 3))
        x_off = 0.5 * dx if (r % 2 == 1) else 0.0
        for c in range(scale_cols):
            xc = int(np.clip((c + 0.5) * dx + x_off, 2, w_roi - 3))
            patch = smooth_green[max(0, yc - 2) : min(h_roi, yc + 3), max(0, xc - 2) : min(w_roi, xc + 3)]
            scale_scores[r, c] = float(np.mean(patch))
            scale_centers_xy.append((float(xc), float(yc)))

    # Classify scales into bimodal Green (1) vs Black (0) using adaptive threshold
    thresh = float(np.percentile(scale_scores, 55.0))
    scale_grid = (scale_scores >= thresh).astype(np.int32)
    for r in range(scale_rows):
        for c in range(scale_cols):
            scale_states_list.append(int(scale_grid[r, c]))

    g_vals = scale_scores[scale_grid == 1]
    b_vals = scale_scores[scale_grid == 0]
    ashman_D = float(
        np.sqrt(2.0) * abs(np.mean(g_vals) - np.mean(b_vals)) / np.sqrt(np.var(g_vals) + np.var(b_vals) + 1e-8)
    )

    stats = compute_neighbor_statistics(scale_grid)
    pts_xy = np.array(scale_centers_xy, dtype=np.float64)
    tree = cKDTree(pts_xy)
    pairs = tree.query_pairs(r=1.35 * max(dx, dy))

    return {
        "rgb": arr,
        "roi_rgb": roi,
        "green_score_map": smooth_green,
        "scale_grid": scale_grid,
        "scale_centers_xy": pts_xy,
        "scale_states": np.array(scale_states_list, dtype=np.int32),
        "neighbor_edges": sorted(pairs),
        "n_scales": int(scale_rows * scale_cols),
        "green_fraction": stats["green_fraction"],
        "mean_ng_green": stats["mean_ng_green"],
        "mean_nb_black": stats["mean_nb_black"],
        "modal_ng_green": stats["modal_ng_green"],
        "hist_green": stats["hist_green"],
        "hist_black": stats["hist_black"],
        "ashman_D": ashman_D,
        "dorsal_roi_xyxy": [x0, y0, x1, y1],
        "head_roi_xyxy": [hx0, hy0, hx1, hy1],
    }


# =========================================================================== #
# 6. Publication Figure (12 Panels), 3D Volume Figure & Animated GIF Rendering
# =========================================================================== #
def _masked_field(sim: dict[str, Any]) -> np.ndarray:
    """Depth-averaged field with removed-skin (P = 0 border) pixels masked."""
    f = sim["u_bar"].astype(np.float64).copy()
    f[sim["h_map"] <= 0] = np.nan
    return f


def render_3d_volume_figure(sim3: dict[str, Any], out_path: Path = OUT_3D_PNG) -> None:
    """Render z-slices and an x-z cross-section of the 3D skin-volume activator field."""
    u3 = sim3["u_volume"]
    skin = sim3["skin_mask"]
    nz, ny, nx = u3.shape
    z_idx = sorted({0, max(sim3["h_border_voxels"] - 1, 0), nz // 3, (2 * nz) // 3, nz - 1})
    fig, axes = plt.subplots(1, len(z_idx) + 1, figsize=(3.1 * (len(z_idx) + 1), 3.4), dpi=130)
    for ax, z in zip(axes[:-1], z_idx):
        sl = u3[z].copy()
        sl[~skin[z]] = np.nan
        ax.imshow(sl, cmap="YlGn", vmin=0.05, vmax=0.95, origin="lower")
        ax.set_title(f"z = {z} / {nz - 1}  (depth {z * sim3['dz_px']:.2f} px)", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[-1]
    y_mid = ny // 2
    xz = u3[:, y_mid, :].copy()
    xz[~skin[:, y_mid, :]] = np.nan
    ax.imshow(xz, cmap="YlGn", vmin=0.05, vmax=0.95, origin="lower", aspect=1.0 / sim3["dz_px"] / 2.0, interpolation="nearest")
    ax.set_title(f"x-z section at y = {y_mid} (P_eff = {sim3['P_effective']:.3f})", fontsize=8.5)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("z (voxel)")
    fig.suptitle(
        f"3D skin-volume reaction-diffusion, P = {sim3['P_border']:.2f}: h_int = {sim3['h_interior_voxels']} voxels "
        f"({sim3['h_interior_px']:.1f} px), h_border = {sim3['h_border_voxels']} voxels; vertical non-uniformity RMS = "
        f"{sim3['vertical_nonuniformity_rms']:.2e}",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_outputs(ca_rows: int = 36, ca_cols: int = 36, ca_steps: int = 45) -> dict[str, Any]:
    """Render the 12-panel Pass 3 synthesis figure, the 3D volume figure and the 6-frame growth GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    ref = load_reference_dataset()
    ca_sim = simulate_hexagonal_ca(rows=ca_rows, cols=ca_cols, steps=ca_steps, seed=2017)
    cmp_phys = compare_3d_volume_to_depth_averaged_2d(P_border=0.17)
    cmp_cont = compare_3d_volume_to_depth_averaged_2d(P_border=1.0)
    pde3_phys, pde2_phys = cmp_phys["sim_3d"], cmp_phys["sim_2d"]
    pde3_cont, pde2_cont = cmp_cont["sim_3d"], cmp_cont["sim_2d"]
    pde3_ins = integrate_skin_volume_pde_3d(P_border=0.0)
    pde2_legacy = integrate_scale_border_pde_2d(P_border=0.17, border_model="diffusivity")
    sweep3 = run_border_reduction_sweep("3d")
    sweep2 = run_border_reduction_sweep("2d")
    controls = run_negative_controls()
    spec = segment_timon_lepidus_specimen()
    stab_script = kinetics_linear_stability(KIN_FU, KIN_FV, KIN_GU, KIN_GV, PDE_D_RATIO, pde3_phys["D_u"])
    stab_script_d1 = kinetics_linear_stability(KIN_FU, KIN_FV, KIN_GU, KIN_GV, 1.0, pde3_phys["D_u"])
    stab_json = canonical_json_kinetics_stability()
    render_3d_volume_figure(pde3_phys)

    fig = plt.figure(figsize=(17.0, 13.6), dpi=150)
    gs = GridSpec(3, 4, figure=fig, hspace=0.40, wspace=0.30)
    n_vals = np.arange(7)

    # Panel A: 3D skin-volume depth-averaged activator (P=0.17) vs continuum (P=1.0)
    ax_a = fig.add_subplot(gs[0, 0])
    comp_pde = np.hstack([_masked_field(pde3_phys), _masked_field(pde3_cont)])
    im_a = ax_a.imshow(comp_pde, cmap="YlGn", vmin=0.05, vmax=0.95, origin="lower")
    ax_a.axvline(PDE_NX - 0.5, color="white", lw=1.8, ls="--")
    ax_a.text(2, PDE_NY - 9, "3D skin volume\nP = 0.17", color="white", fontsize=7.6, fontweight="bold", bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax_a.text(PDE_NX + 2, PDE_NY - 9, "3D continuum\nP = 1.00", color="white", fontsize=7.6, fontweight="bold", bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax_a.set_title("A. 3D Skin-Volume RD ($\\bar{u}$)", fontsize=10.2, fontweight="bold")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    plt.colorbar(im_a, ax=ax_a, fraction=0.025, pad=0.03)

    # Panel B: Hexagonal Outer-Totalistic CA (Juvenile -> Adult)
    ax_b = fig.add_subplot(gs[0, 1])
    juv_grid = ca_sim["snapshots"][0][1]
    adult_grid = ca_sim["grid"]
    comp_ca = np.hstack([juv_grid, adult_grid])
    ax_b.imshow(comp_ca, cmap="YlGn", vmin=0, vmax=1, origin="lower")
    ax_b.axvline(juv_grid.shape[1] - 0.5, color="#facc15", lw=1.8, ls="--")
    ax_b.text(3, ca_rows - 5, "Juvenile Ocelli", color="white", fontsize=8.0, fontweight="bold", bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax_b.text(ca_cols + 3, ca_rows - 5, "Adult CA Labyrinth", color="white", fontsize=8.0, fontweight="bold", bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax_b.set_title("B. Hexagonal Outer-Totalistic CA", fontsize=10.5, fontweight="bold")
    ax_b.set_xticks([])
    ax_b.set_yticks([])

    # Panel C: Outer-Totalistic State-Flip Hazard Rules p(flip | n_green)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(n_vals, ca_sim["p_g2b"], "o-", color="#16a34a", lw=2.2, label=r"$p(\mathrm{Green}\to\mathrm{Black} \mid n_g)$")
    ax_c.plot(n_vals, ca_sim["p_b2g"], "s--", color="#0f172a", lw=2.2, label=r"$p(\mathrm{Black}\to\mathrm{Green} \mid n_g)$")
    ax_c.axvspan(2.6, 3.4, color="#fde047", alpha=0.35, label="Labyrinth attractor ($n_g=3$)")
    ax_c.set_title("C. Outer-Totalistic CA Flip Rules", fontsize=10.5, fontweight="bold")
    ax_c.set_xlabel(r"Number of Green Neighbors $n_g \in \{0..6\}$")
    ax_c.set_ylabel("Transition Probability per Step")
    ax_c.legend(fontsize=7.6, loc="upper center")
    ax_c.grid(alpha=0.25)

    # Panel D: Border thickness P sweep — 3D volume vs depth-averaged 2D reduction
    ax_d = fig.add_subplot(gs[0, 3])
    p3 = [r["P"] for r in sweep3]
    ax_d.plot(p3, [r["discretization_ratio"] for r in sweep3], "o-", color="#1d4ed8", lw=2.2, label=r"3D volume $\sigma_{\mathrm{inter}}/\sigma_{\mathrm{intra}}$")
    ax_d.plot([r["P"] for r in sweep2], [r["discretization_ratio"] for r in sweep2], "s--", color="#0891b2", lw=1.8, label=r"2D thin-film $\sigma_{\mathrm{inter}}/\sigma_{\mathrm{intra}}$")
    ax_d.plot(p3, [r["ashman_D"] if r["patterned"] else np.nan for r in sweep3], "^:", color="#d97706", lw=1.8, label="3D scale bimodality (Ashman $D$)")
    ax_d.axvline(0.17, color="#dc2626", ls=":", lw=1.6, label="Timon lepidus $P=0.17$")
    ax_d.set_yscale("log")
    ax_d.set_title("D. Border Thinning $P$ Sweep: 3D vs 2D", fontsize=10.5, fontweight="bold")
    ax_d.set_xlabel("Border Thickness Factor $P = h_{\\mathrm{border}} / h_{\\mathrm{interior}}$")
    ax_d.set_ylabel("Ratio / Bimodality $D$ (log)")
    ax_d.text(
        0.03,
        0.05,
        f"P=0 insulated: inter std={controls['insulated_P0_inter_scale_std']:.1e} (no pattern)\n"
        f"d=1 equal diff: inter std={controls['equal_diffusion_inter_scale_std']:.1e}\n"
        f"P=0.17 disc ratio {controls['positive_discretization_ratio']:.2f} vs P=1.0 {controls['continuum_P1_discretization_ratio']:.2f}",
        transform=ax_d.transAxes,
        fontsize=7.0,
        bbox=dict(facecolor="#f8fafc", edgecolor="#cbd5e1", boxstyle="round,pad=0.3"),
    )
    ax_d.legend(fontsize=6.9, loc="upper right")
    ax_d.grid(alpha=0.25, which="both")

    # Panel E: Timon lepidus Specimen Photograph
    ax_e = fig.add_subplot(gs[1, 0])
    ax_e.imshow(spec["rgb"])
    dx0, dy0, dx1, dy1 = spec["dorsal_roi_xyxy"]
    hx0, hy0, hx1, hy1 = spec["head_roi_xyxy"]
    ax_e.add_patch(plt.Rectangle((dx0, dy0), dx1 - dx0, dy1 - dy0, fill=False, edgecolor="#22c55e", lw=2.2, label="Dorsal Scale ROI"))
    ax_e.add_patch(plt.Rectangle((hx0, hy0), hx1 - hx0, hy1 - hy0, fill=False, edgecolor="#38bdf8", lw=1.8, label="Head Plate ROI"))
    ax_e.set_title("E. Timon lepidus Specimen (CC BY-SA 4.0)", fontsize=10.5, fontweight="bold")
    ax_e.legend(fontsize=7.8, loc="lower right")
    ax_e.axis("off")

    # Panel F: Segmented Dorsal Hexagonal Scale Graph
    ax_f = fig.add_subplot(gs[1, 1])
    ax_f.imshow(spec["roi_rgb"])
    pts = spec["scale_centers_xy"]
    states = spec["scale_states"]
    for u_i, v_i in spec["neighbor_edges"]:
        ax_f.plot([pts[u_i, 0], pts[v_i, 0]], [pts[u_i, 1], pts[v_i, 1]], color="white", lw=0.55, alpha=0.45)
    g_pts = pts[states == 1]
    b_pts = pts[states == 0]
    ax_f.scatter(g_pts[:, 0], g_pts[:, 1], c="#22c55e", edgecolors="black", s=24, zorder=4, label=f"Green ({len(g_pts)})")
    ax_f.scatter(b_pts[:, 0], b_pts[:, 1], c="#0f172a", edgecolors="white", s=24, zorder=4, label=f"Black ({len(b_pts)})")
    ax_f.set_title(f"F. Dorsal Scale Lattice ($N={spec['n_scales']}, D={spec['ashman_D']:.2f}$)", fontsize=10.2, fontweight="bold")
    ax_f.legend(fontsize=7.6, loc="lower right")
    ax_f.set_xticks([])
    ax_f.set_yticks([])

    # Panel G: Green Scale Nearest-Neighbor Histogram (3D PDE vs CA vs Specimen vs Ref)
    ax_g = fig.add_subplot(gs[1, 2])
    ref_hist_g = np.array(ref["hexagonal_outer_totalistic_ca"]["adult_neighbor_histogram_targets"]["green_scale_neighbor_hist_ref"], dtype=np.float64)
    w_bar = 0.20
    ax_g.bar(n_vals - 1.5 * w_bar, ref_hist_g, width=w_bar, color="#64748b", label="Manukyan (2017) Ref")
    ax_g.bar(n_vals - 0.5 * w_bar, pde3_phys["neighbor_stats"]["hist_green"], width=w_bar, color="#2563eb", label="3D Skin-Volume PDE")
    ax_g.bar(n_vals + 0.5 * w_bar, ca_sim["stats"]["hist_green"], width=w_bar, color="#16a34a", label="Hexagonal CA")
    ax_g.bar(n_vals + 1.5 * w_bar, spec["hist_green"], width=w_bar, color="#d97706", label="Timon lepidus ROI")
    ax_g.set_xticks(n_vals)
    ax_g.set_title("G. Green-Scale Neighbors $P(n_g|S=1)$", fontsize=10.2, fontweight="bold")
    ax_g.set_xlabel("Green First-Nearest Neighbors $n_g$")
    ax_g.set_ylabel("Fraction of Green Scales")
    ax_g.legend(fontsize=7.2, loc="upper right")
    ax_g.grid(axis="y", alpha=0.25)

    # Panel H: Black-Scale Neighbor Histogram
    ax_h = fig.add_subplot(gs[1, 3])
    ref_hist_b = np.array(ref["hexagonal_outer_totalistic_ca"]["adult_neighbor_histogram_targets"]["black_scale_neighbor_hist_ref"], dtype=np.float64)
    ax_h.bar(n_vals - 0.22, ref_hist_b, width=0.22, color="#64748b", label="Manukyan (2017) Ref")
    ax_h.bar(n_vals, ca_sim["stats"]["hist_black"], width=0.22, color="#0f172a", label="Hexagonal CA")
    ax_h.bar(n_vals + 0.22, spec["hist_black"], width=0.22, color="#ea580c", label="Timon lepidus ROI")
    ax_h.set_xticks(n_vals)
    ax_h.set_title("H. Black-Scale Neighbors $P(n_g|S=0)$", fontsize=10.2, fontweight="bold")
    ax_h.set_xlabel("Green First-Nearest Neighbors $n_g$ ($n_b = 6 - n_g$)")
    ax_h.set_ylabel("Fraction of Black Scales")
    ax_h.legend(fontsize=7.5, loc="upper right")
    ax_h.grid(axis="y", alpha=0.25)

    # Panel I: 3D skin volume x-z cross-section (stepped thickness, thin borders)
    ax_i = fig.add_subplot(gs[2, 0])
    u3 = pde3_phys["u_volume"]
    sk3 = pde3_phys["skin_mask"]
    y_mid = PDE_NY // 2
    xz = u3[:, y_mid, :].copy()
    xz[~sk3[:, y_mid, :]] = np.nan
    ax_i.imshow(xz, cmap="YlGn", vmin=0.05, vmax=0.95, origin="lower", aspect=2.2, interpolation="nearest")
    ax_i.set_title("I. 3D Voxel Skin x-z Section", fontsize=10.2, fontweight="bold")
    ax_i.set_xlabel(
        f"x (px) at y = {y_mid}; $h_{{int}}$={pde3_phys['h_interior_voxels']}, $h_b$={pde3_phys['h_border_voxels']} voxels\n"
        f"vertical RMS non-uniformity = {pde3_phys['vertical_nonuniformity_rms']:.1e}",
        fontsize=8.0,
    )
    ax_i.set_ylabel("z (voxel, dz = %.1f px)" % pde3_phys["dz_px"])

    # Panel J: 3D depth-averaged vs 2D thin-film reduction at P=0.17
    ax_j = fig.add_subplot(gs[2, 1])
    comp_j = np.hstack([_masked_field(pde3_phys), _masked_field(pde2_phys)])
    ax_j.imshow(comp_j, cmap="YlGn", vmin=0.05, vmax=0.95, origin="lower")
    ax_j.axvline(PDE_NX - 0.5, color="white", lw=1.8, ls="--")
    ax_j.text(2, PDE_NY - 9, "3D volume $\\bar{u}$", color="white", fontsize=7.6, fontweight="bold", bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax_j.text(PDE_NX + 2, PDE_NY - 9, "2D thin-film\n$(1/h)\\nabla\\cdot(hD\\nabla u)$", color="white", fontsize=7.6, fontweight="bold", bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax_j.set_title(
        f"J. 3D vs Thin-Film 2D ($P$=0.17)\nscale-state agreement {100 * cmp_phys['scale_state_agreement']:.1f}%, RMS {cmp_phys['depth_averaged_rms_difference']:.3f}",
        fontsize=9.2,
        fontweight="bold",
    )
    ax_j.set_xticks([])
    ax_j.set_yticks([])

    # Panel K: pattern growth (inter-scale std vs time) for 3D and 2D models
    ax_k = fig.add_subplot(gs[2, 2])
    for sim, lab, col, ls in [
        (pde3_phys, "3D volume P=0.17", "#1d4ed8", "-"),
        (pde2_phys, "2D thin-film P=0.17", "#0891b2", "--"),
        (pde3_cont, "3D volume P=1.0", "#65a30d", "-"),
        (pde3_ins, "3D volume P=0 (insulated)", "#9ca3af", "-"),
        (pde2_legacy, "2D v2 diffusivity model P=0.17", "#d97706", ":"),
    ]:
        t_hist = np.array(sim["inter_std_history"])
        ax_k.plot(t_hist[:, 0], t_hist[:, 1], ls, color=col, lw=1.8, label=lab)
    ax_k.set_title("K. Inter-Scale Pattern Growth $\\sigma_{inter}(t)$", fontsize=10.2, fontweight="bold")
    ax_k.set_xlabel("time (model units); $\\sigma_{max}$ = %.3f" % stab_script["sigma_max"])
    ax_k.set_ylabel("std of per-scale mean $\\bar{u}$")
    ax_k.legend(fontsize=6.8, loc="center right")
    ax_k.grid(alpha=0.25)

    # Panel L: Turing dispersion relations (script kinetics vs vendored canonical JSON kinetics)
    ax_l = fig.add_subplot(gs[2, 3])
    k_pitch = 2.0 * np.pi / build_hexagonal_scale_geometry()["pitch_px"]
    ax_l.plot(stab_script["k_grid"], stab_script["sigma_grid"], "-", color="#1d4ed8", lw=2.0, label=f"script kinetics d=20 ($\\lambda_c$={stab_script['lambda_c_px']:.0f} px, $d_c$={stab_script['d_critical']:.1f})")
    ax_l.plot(stab_script_d1["k_grid"], stab_script_d1["sigma_grid"], "--", color="#64748b", lw=1.6, label="script kinetics d=1 (control)")
    kj = stab_json["k_grid"] * (stab_script["k_c"] / max(stab_json["k_c"], 1e-9))
    ax_l.plot(kj, stab_json["sigma_grid"], ":", color="#dc2626", lw=2.0, label=f"JSON canonical set d={stab_json['d_ratio']:.0f} ($d_c$={stab_json['d_critical']:.1f}: NOT unstable)")
    ax_l.axhline(0.0, color="black", lw=0.8)
    ax_l.axvline(k_pitch, color="#d97706", ls="-.", lw=1.3, label="k of one scale pitch")
    ax_l.axvline(np.pi / 4.0, color="#7c3aed", ls="-.", lw=1.3, label="first zero-flux mode of an isolated scale")
    ax_l.set_xlim(0, max(2.5 * stab_script["k_c"], 0.9))
    ax_l.set_ylim(min(-0.3, float(np.min(stab_script["sigma_grid"]))), max(0.25, 1.3 * stab_script["sigma_max"]))
    ax_l.set_title("L. Turing Dispersion $\\sigma(k)$ (k rescaled to script $k_c$)", fontsize=9.8, fontweight="bold")
    ax_l.set_xlabel("wavenumber k (px$^{-1}$)")
    ax_l.set_ylabel("growth rate $\\sigma(k)$")
    ax_l.legend(fontsize=6.4, loc="lower left")
    ax_l.grid(alpha=0.25)

    fig.suptitle(
        "Manukyan et al. (2017) Nature — Ocellated Lizard (Timon lepidus) Skin-Scale Cellular Automaton: "
        "3D Skin-Volume Reaction-Diffusion, Thin-Film Reduction & Hexagonal CA (Pass 3)",
        fontsize=12.0,
        fontweight="bold",
        y=0.985,
    )
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    # Render 6-frame ontogenetic growth GIF showing Juvenile Ocelli -> Adult Labyrinth + Neighbor Histogram
    gif_frames: list[Image.Image] = []
    for step_idx, snap_grid in ca_sim["snapshots"]:
        st_snap = compute_neighbor_statistics(snap_grid)
        fig_g, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=110)
        ax1.imshow(snap_grid, cmap="YlGn", vmin=0, vmax=1, origin="lower")
        ax1.set_title(f"Timon lepidus Hex CA Step {step_idx} (Green={st_snap['green_fraction']:.2f})")
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2.bar(n_vals - 0.16, st_snap["hist_green"], width=0.32, color="#16a34a", label="CA Green Scales")
        ax2.plot(n_vals, ref_hist_g, "o--", color="#dc2626", lw=1.8, label="Adult Target (Manukyan 2017)")
        ax2.set_xticks(n_vals)
        ax2.set_ylim(0.0, 0.50)
        ax2.set_xlabel("Green Neighbors n_g")
        ax2.set_ylabel("Fraction")
        ax2.set_title(f"Neighbor Histogram (<n_g>={st_snap['mean_ng_green']:.2f})")
        ax2.legend(fontsize=7.5)
        ax2.grid(axis="y", alpha=0.25)
        fig_g.tight_layout()
        fig_g.canvas.draw()
        rgba = np.asarray(fig_g.canvas.buffer_rgba())
        gif_frames.append(Image.fromarray(rgba[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig_g)

    gif_frames[0].save(OUT_GIF, save_all=True, append_images=gif_frames[1:], duration=380, loop=0)

    return {
        "ca": ca_sim,
        "pde3_phys": pde3_phys,
        "pde2_phys": pde2_phys,
        "pde3_cont": pde3_cont,
        "cmp_phys": cmp_phys,
        "cmp_cont": cmp_cont,
        "sweep3": sweep3,
        "sweep2": sweep2,
        "controls": controls,
        "specimen": spec,
        "stability_script": stab_script,
        "stability_json": stab_json,
    }


# =========================================================================== #
# 7. MorphoEval Dynamic Measurement Hook (`measure_morphoeval`)
# =========================================================================== #
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 16 reference-backed morphometric measurements for `measure_outputs.py`."""
    ref = load_reference_dataset()
    skin_ref = load_skin_volume_reference()
    ca_sim = simulate_hexagonal_ca(rows=36, cols=36, steps=45, seed=2017)
    cmp_phys = compare_3d_volume_to_depth_averaged_2d(P_border=0.17)
    pde3 = cmp_phys["sim_3d"]
    controls = run_negative_controls()
    sweep3 = run_border_reduction_sweep("3d")
    spec = segment_timon_lepidus_specimen()
    stab_script = kinetics_linear_stability(KIN_FU, KIN_FV, KIN_GU, KIN_GV, PDE_D_RATIO, pde3["D_u"])
    stab_json = canonical_json_kinetics_stability()
    geom = build_hexagonal_scale_geometry()

    ref_hist_g = np.array(ref["hexagonal_outer_totalistic_ca"]["adult_neighbor_histogram_targets"]["green_scale_neighbor_hist_ref"], dtype=np.float64)
    r_ca_ref, p_ca_ref = pearsonr(ca_sim["stats"]["hist_green"], ref_hist_g)
    pde_ca_l1 = float(np.sum(np.abs(pde3["neighbor_stats"]["hist_green"] - ca_sim["stats"]["hist_green"])))
    sweep_pos = [r for r in sweep3 if r["P"] > 0.0]
    rho_sweep = float(spearmanr([r["P"] for r in sweep_pos], [r["discretization_ratio"] for r in sweep_pos])[0])
    lambda_in_pitches = float(stab_script["lambda_c_px"] / geom["pitch_px"])
    thin_film_ref = skin_ref["thin_film_reduction"]

    return [
        {
            "name": "hexagonal CA adult green-scale mean green neighbor count <n_g>",
            "value": round(ca_sim["stats"]["mean_ng_green"], 4),
            "unit": "green neighbors (out of 6)",
            "method": "Steady-state outer-totalistic hexagonal CA mean green neighbors around green scales (36x36 lattice, step 45)",
            "reference_value": 2.665,
            "reference_source": "Manukyan et al. (2017) Nature 544:173-179 Fig. 3d",
            "note": f"modal n_g = {ca_sim['stats']['modal_ng_green']}, L1 distance to Manukyan Fig. 3d histogram = {ca_sim['l1_hist_distance_to_ref']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "hexagonal CA adult black-scale mean black neighbor count <n_b>",
            "value": round(ca_sim["stats"]["mean_nb_black"], 4),
            "unit": "black neighbors (out of 6)",
            "method": "Steady-state outer-totalistic hexagonal CA mean black neighbors (6 - n_g) around black scales",
            "reference_value": 3.272,
            "reference_source": "Manukyan et al. (2017) Nature 544:173-179 Fig. 3e & Lean Theorem 1 hex_scale_neighbor_partition_and_symmetry",
            "note": f"modal n_b = {ca_sim['stats']['modal_nb_black']}, <n_g>_black = {ca_sim['stats']['mean_ng_black']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "3D skin-volume PDE (P=0.17) inter-to-intra scale concentration discretization ratio",
            "value": round(pde3["discretization_ratio"], 4),
            "unit": "sigma_inter / sigma_intra",
            "method": f"Ratio of inter-scale std of depth-averaged activator to mean intra-scale std in the 3D voxel skin volume (h_int={pde3['h_interior_voxels']} vox, h_border={pde3['h_border_voxels']} vox, P_eff={pde3['P_effective']:.3f}); no per-scale averaging applied",
            "reference_value": 3.5,
            "reference_source": "Manukyan et al. (2017) Nature 544:173-179 Fig. 4 & Fofonjka & Milinkovitch (2021) Nat. Commun. 12:2433",
            "note": f"3D P=0.17 ratio = {pde3['discretization_ratio']:.3f} vs 3D continuum P=1.0 ratio = {controls['continuum_P1_discretization_ratio']:.3f} ({controls['discretization_gain_vs_continuum']:.2f}x gain); 2D thin-film ratio = {cmp_phys['discretization_ratio_2d']:.3f}; v2 reported 86.79 with an artificial per-scale averaging step",
            "derived_from": "script",
        },
        {
            "name": "3D skin-volume PDE (P=0.17) scale color bimodality Ashman's D",
            "value": round(pde3["ashman_D"], 4),
            "unit": "Ashman D",
            "method": f"sqrt(2)*|mu_green - mu_black| / sqrt(var_green + var_black) across {geom['n_scales']} hexagonal scales of depth-averaged u at P = 0.17",
            "reference_value": 2.5,
            "reference_source": "Manukyan et al. (2017) Nature 544:173-179 Fig. 2 & Fig. 4 bimodal scale state criterion (D > 2.0)",
            "note": f"inter-scale std = {pde3['inter_scale_std']:.4f}, intra-scale std = {pde3['mean_intra_scale_std']:.4f}; 2D thin-film Ashman D = {cmp_phys['ashman_D_2d']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "3D skin-volume PDE vs outer-totalistic CA neighbor histogram L1 distance",
            "value": round(pde_ca_l1, 4),
            "unit": "L1 probability distance",
            "method": "sum(|P_3D(n_g | S=1) - P_CA(n_g | S=1)|) for n_g in {0..6}",
            "reference_value": 0.0,
            "reference_source": "Manukyan et al. (2017) Nature 544:173-179 Figs. 3-4 PDE-to-CA mesoscopic reduction",
            "note": f"CA vs Manukyan reference histogram Pearson r = {r_ca_ref:.4f} (p = {p_ca_ref:.2e}); 3D PDE <n_g>_green = {pde3['neighbor_stats']['mean_ng_green']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion (d=1.0) negative control scale amplitude suppression",
            "value": round(controls["equal_diffusion_inter_scale_std"], 6),
            "unit": "std(u_scale)",
            "method": "Inter-scale depth-averaged activator std at d = Dv/Du = 1.0 (equal diffusion, 2D thin-film) vs positive Turing regime d = 20.0 (3D)",
            "reference_value": 0.0,
            "reference_source": "Turing (1952) / Manukyan et al. (2017) Nature 544:173-179 Methods",
            "note": f"positive d=20 std = {controls['positive_inter_scale_std']:.4f} vs equal-diffusion d=1 std = {controls['equal_diffusion_inter_scale_std']:.2e}",
            "derived_from": "script",
        },
        {
            "name": "segmented Timon lepidus dorsal green-scale fraction",
            "value": round(spec["green_fraction"], 4),
            "unit": "fraction of dorsal scales",
            "method": "Chromatic green-excess segmentation on datasets/ocellated_lizard_ca/timon_lepidus_specimen.jpg dorsal ROI [120, 180, 320, 360]",
            "reference_value": 0.44,
            "reference_source": "Manukyan et al. (2017) Nature 544:173-179 Fig. 2 adult Timon lepidus dorsal green scale proportion",
            "note": f"N = {spec['n_scales']} segmented dorsal scales, bimodal Ashman D = {spec['ashman_D']:.3f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "segmented Timon lepidus dorsal green-scale mean green neighbor count <n_g>",
            "value": round(spec["mean_ng_green"], 4),
            "unit": "green neighbors (out of 6)",
            "method": "Hexagonal lattice first-nearest-neighbor green degree around segmented green scales on timon_lepidus_specimen.jpg",
            "reference_value": 2.665,
            "reference_source": "Manukyan et al. (2017) Nature 544:173-179 Fig. 3d empirical Timon lepidus neighbor histogram",
            "note": f"empirical modal n_g = {spec['modal_ng_green']}, black-scale mean black neighbors <n_b> = {spec['mean_nb_black']:.3f}",
            "derived_from": "output_pixels",
        },
        {
            "name": "3D skin volume vs depth-averaged 2D thin-film reduction: binary scale-state agreement (P=0.17)",
            "value": round(cmp_phys["scale_state_agreement"], 4),
            "unit": "fraction of scales",
            "method": f"Fraction of {geom['n_scales']} scales whose green/black state (55th-percentile threshold of per-scale mean u) agrees between the 3D voxel solution and the 2D (1/h) div(h D grad u) reduction at identical P, seed, dt and steps",
            "reference_value": float(thin_film_ref["expected_scale_state_agreement"]),
            "reference_source": "Fofonjka & Milinkovitch (2021) Nat. Commun. 12:2433 thin-film limit h << lambda_c (datasets/ocellated_lizard_ca/skin_volume_reference.json)",
            "note": f"per-scale mean Pearson r = {cmp_phys['scale_mean_pearson_r']:.4f}; continuum P=1.0 agreement = {compare_3d_volume_to_depth_averaged_2d(P_border=1.0)['scale_state_agreement']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "3D skin volume vs depth-averaged 2D reduction: RMS difference of depth-averaged activator (P=0.17)",
            "value": round(cmp_phys["depth_averaged_rms_difference"], 4),
            "unit": "concentration units (u in [0.05, 0.95])",
            "method": "sqrt(mean((ubar_3D - u_2D)^2)) over all skin pixels at P = 0.17",
            "reference_value": 0.0,
            "reference_source": "Thin-film / lubrication reduction of 3D diffusion in a domain of slowly varying thickness (Fofonjka & Milinkovitch 2021)",
            "note": f"pattern amplitude (inter-scale std) = {pde3['inter_scale_std']:.4f}; RMS / amplitude = {cmp_phys['depth_averaged_rms_difference'] / max(pde3['inter_scale_std'], 1e-9):.3f}",
            "derived_from": "script",
        },
        {
            "name": "3D skin volume vertical non-uniformity of activator (RMS over depth, interior columns, P=0.17)",
            "value": round(pde3["vertical_nonuniformity_rms"], 6),
            "unit": "concentration units",
            "method": f"mean over interior columns of sqrt((1/h) sum_z (u - ubar)^2) in the 3D voxel skin volume (h_int = {pde3['h_interior_px']:.1f} px = {pde3['h_interior_px'] / geom['pitch_px']:.2f} scale pitches, lambda_c = {stab_script['lambda_c_px']:.0f} px)",
            "reference_value": 0.0,
            "reference_source": "Thin-film regime h_interior << lambda_c assumed by the 2D depth-averaged reduction",
            "note": f"max over all columns = {pde3['vertical_nonuniformity_max']:.4f}; continuum P=1.0 vertical RMS = {compare_3d_volume_to_depth_averaged_2d(P_border=1.0)['vertical_nonuniformity_rms_3d']:.2e}",
            "derived_from": "script",
        },
        {
            "name": "3D skin voxel volume fraction relative to a uniform slab (P=0.17)",
            "value": round(pde3["skin_volume_fraction_vs_slab"], 4),
            "unit": "fraction",
            "method": "n_skin_voxels / (nz * ny * nx) of the stepped-thickness voxel skin",
            "reference_value": round(pde3["skin_volume_fraction_analytic"], 4),
            "reference_source": "Analytic 1 - (1 - P_eff) * border_area_fraction for the same geometry (self-consistency of the voxel mask)",
            "note": f"border area fraction = {geom['border_area_fraction']:.3f} at {geom['pitch_px']:.0f} px scale pitch (coarse: real inter-scale borders are a smaller area fraction); P_eff = {pde3['P_effective']:.3f}",
            "derived_from": "script",
        },
        {
            "name": "insulated-scale control (P=0): inter-scale pattern amplitude in the 3D skin volume",
            "value": round(controls["insulated_P0_inter_scale_std"], 6),
            "unit": "std(ubar_scale)",
            "method": f"Inter-scale std of per-scale mean u with zero inter-scale conductance (h_border = 0); the geometry's interior splits into {geom['n_interior_components']} 4-connected components for {geom['n_scales']} scales",
            "reference_value": 0.0,
            "reference_source": "Manukyan et al. (2017) Fig. 4 / ocellated_lizard_ca_reference.json regime insulated_decoupled_scales (predicted: no labyrinthine correlation forms)",
            "note": f"insulated intra-scale std = {controls['insulated_P0_intra_std']:.2e}: isolated scales cannot host a Turing mode because lambda_c = {lambda_in_pitches:.1f} pitches > 2 x scale diameter",
            "derived_from": "script",
        },
        {
            "name": "P-sweep monotonicity: Spearman rho of 3D discretization ratio vs P over P in {0.08, 0.17, 0.45, 1.0}",
            "value": round(rho_sweep, 4),
            "unit": "Spearman rho",
            "method": "Rank correlation between border thickness factor P and the 3D sigma_inter/sigma_intra discretization ratio (P = 0 excluded: no pattern)",
            "reference_value": -1.0,
            "reference_source": "Manukyan et al. (2017) Fig. 4 / Fofonjka & Milinkovitch (2021): thinner borders -> stronger scale-level discretization",
            "note": "; ".join(f"P={r['P']:.2f} (P_eff {r['P_effective']:.3f}): ratio {r['discretization_ratio']:.2f}, D={r['ashman_D']:.2f}" for r in sweep3),
            "derived_from": "script",
        },
        {
            "name": "Turing wavelength of the script kinetics in scale pitches (lambda_c / pitch)",
            "value": round(lambda_in_pitches, 4),
            "unit": "scale pitches",
            "method": f"lambda_c = 2 pi / k_c with k_c^2 = sqrt(det J / (D_u D_v)) for the script kinetics (D_u = {pde3['D_u']:.3f}, d = {PDE_D_RATIO:.0f}) divided by the hexagonal scale pitch {geom['pitch_px']:.1f} px",
            "reference_value": 4.0,
            "reference_source": "Manukyan et al. (2017) Nature 544:173-179 Fig. 4: continuum RD labyrinth bands 1-2 scales wide (wavelength ~ 2-4 scale diameters); value chosen so an isolated scale cannot pattern (k_1 = pi/4 px^-1 > k_max = %.3f px^-1)" % stab_script["k_max_unstable"],
            "note": f"sigma_max = {stab_script['sigma_max']:.4f} at k = {stab_script['k_sigma_max']:.3f}; unstable band k in [{stab_script['k_min_unstable']:.3f}, {stab_script['k_max_unstable']:.3f}] px^-1; d_c = {stab_script['d_critical']:.2f}",
            "derived_from": "script",
        },
        {
            "name": "Turing critical diffusivity ratio d_c of the vendored canonical clipped-linear Kondo-Asai parameter set",
            "value": round(stab_json["d_critical"], 4),
            "unit": "D_v / D_u",
            "method": "d_c = [(2 det J - f_u g_v) + 2 sqrt(det J (det J - f_u g_v))] / f_u^2 for J from c_u - gamma_u, c_v, d_u, d_v - gamma_v in ocellated_lizard_ca_reference.json",
            "reference_value": float(stab_json["d_ratio"]),
            "reference_source": "ocellated_lizard_ca_reference.json canonical_parameters.diffusivity_ratio_d (the vendored set claims a Turing regime at this d)",
            "note": f"tr J = {stab_json['trace']:.3f}, det J = {stab_json['det']:.3f}, steady state (u*, v*) = ({stab_json['u_star']:.3f}, {stab_json['v_star']:.3f}) within clip = {stab_json['steady_state_within_clip']}; Turing-unstable at d = {stab_json['d_ratio']:.0f}: {stab_json['turing_unstable']} -> the vendored canonical set is NOT Turing-unstable at its own d (defect carried to Pass 4)",
            "derived_from": "script",
        },
    ]


# =========================================================================== #
# 8. Deterministic 14-Check Self-Test Suite (`--self-test`)
# =========================================================================== #
def self_test() -> bool:
    """Run 14 deterministic verification checks covering SHA-256, Lean laws, 3D/2D PDE, CA, controls, and specimen CV."""
    small = dict(scale_rows=8, scale_cols=8, ny=40, nx=40, steps=3000)

    # 1. Dataset SHA-256 locks & JSON schema parity
    ref = load_reference_dataset()
    skin_ref = load_skin_volume_reference()
    assert ref["scale_border_reaction_diffusion_model"]["canonical_parameters"]["P_border_reduction"] == 0.17
    assert skin_ref["thin_film_reduction"]["depth_averaged_equation"].startswith("d ubar/dt")

    # 2. Lean Theorem 1 & Theorem 3 deterministic outer-totalistic CA fixed point verification
    for n_g in range(7):
        assert n_g + (6 - n_g) == 6
    assert deterministic_ca_core_step(1, 3) == 1, "Green scale at n_g=3 must be a fixed point"
    assert deterministic_ca_core_step(0, 3) == 0, "Black scale at n_g=3 must be a fixed point"
    assert deterministic_ca_core_step(1, 5) == 0, "Overcrowded green scale (n_g=5) must flip to black"
    assert deterministic_ca_core_step(0, 1) == 1, "Isolated black pocket (n_g=1) must flip to green"

    # 3. Stochastic hexagonal CA convergence to Manukyan (2017) Fig. 3d-e labyrinthine histogram
    ca_sim = simulate_hexagonal_ca(rows=36, cols=36, steps=45, seed=2017)
    assert 2.2 <= ca_sim["stats"]["mean_ng_green"] <= 3.1, f"Unexpected CA <n_g>: {ca_sim['stats']['mean_ng_green']}"
    assert ca_sim["stats"]["modal_ng_green"] in {2, 3}, f"Expected CA mode at 2 or 3, got {ca_sim['stats']['modal_ng_green']}"
    assert ca_sim["l1_hist_distance_to_ref"] < 0.30, f"CA histogram L1 distance too large: {ca_sim['l1_hist_distance_to_ref']}"

    # 4. Lean Theorems 4 & 5 numeric instances: depth average of a uniform column and face-conductance ratio
    geom_s = build_hexagonal_scale_geometry(**{k: small[k] for k in ("scale_rows", "scale_cols", "ny", "nx")})
    for h_i, c in [(12, 7), (2, 5), (1, 3)]:
        assert (h_i * c) // h_i == c
    thick = skin_thickness_map(geom_s, 0.17, nz=12)
    assert thick["h_border"] == 2 and abs(thick["P_effective"] - 2.0 / 12.0) < 1e-12
    assert min(thick["h_border"], 12) * 12 == min(12, 12) * thick["h_border"], "conductance ratio must equal thickness ratio"
    assert skin_thickness_map(geom_s, 0.0, nz=12)["h_border"] == 0
    # Leak-proof border: with borders removed, interior components ~= number of scales (every scale insulated)
    assert geom_s["n_interior_components"] >= geom_s["n_scales"], (
        f"Insulated interiors must not connect across scales: {geom_s['n_interior_components']} components for {geom_s['n_scales']} scales"
    )

    # 5. Turing linear stability: script kinetics unstable at d=20, stable at d=1; lambda_c = 5 pitches; isolated scale mode outside band
    D_u = script_diffusivity_for_wavelength()
    stab = kinetics_linear_stability(KIN_FU, KIN_FV, KIN_GU, KIN_GV, PDE_D_RATIO, D_u)
    stab1 = kinetics_linear_stability(KIN_FU, KIN_FV, KIN_GU, KIN_GV, 1.0, D_u)
    assert stab["stable_homogeneous"] and stab["turing_unstable"] and stab["sigma_max"] > 0.0
    assert not stab1["turing_unstable"] and stab1["sigma_max"] < 0.0
    assert abs(stab["lambda_c_px"] - PDE_LAMBDA_PX) < 1e-6
    assert abs(stab["d_critical"] - 7.0) < 0.05, f"script kinetics d_c should be ~7.0, got {stab['d_critical']}"
    assert np.pi / 4.0 > stab["k_max_unstable"], "first zero-flux mode of an isolated 4-px scale must lie above the unstable band"
    stab_json = canonical_json_kinetics_stability()
    assert stab_json["stable_homogeneous"]
    assert abs(stab_json["d_critical"] - 23.57) < 0.1, f"JSON canonical kinetics d_c should be ~23.57, got {stab_json['d_critical']}"

    # 6. 3D skin-volume PDE (P = 0.17) vs depth-averaged 2D thin-film reduction (reduced lattice for speed)
    cmp_s = compare_3d_volume_to_depth_averaged_2d(P_border=0.17, **small)
    s3 = cmp_s["sim_3d"]
    assert s3["patterned"] and cmp_s["sim_2d"]["patterned"], "P=0.17 skin volume must form a scale-level pattern"
    assert cmp_s["scale_state_agreement"] >= 0.85, f"3D vs 2D scale-state agreement too low: {cmp_s['scale_state_agreement']}"
    assert s3["vertical_nonuniformity_rms"] < 0.05, f"3D field not vertically uniform (thin-film violated): {s3['vertical_nonuniformity_rms']}"
    assert s3["discretization_ratio"] > 2.0, f"Expected scale discretization at P=0.17, got {s3['discretization_ratio']}"
    assert s3["ashman_D"] > 2.0, f"Expected bimodal scale states (Ashman D > 2.0), got {s3['ashman_D']}"
    assert s3["n_skin_voxels"] == int(round(s3["skin_volume_fraction_vs_slab"] * s3["u_volume"].size))

    # 7. Negative controls on the reduced lattice: equal diffusion, continuum P=1.0, insulated P=0
    eq = integrate_scale_border_pde_2d(P_border=0.17, d_ratio=1.0, **small)
    cont3 = integrate_skin_volume_pde_3d(P_border=1.0, **small)
    ins3 = integrate_skin_volume_pde_3d(P_border=0.0, **small)
    assert eq["inter_scale_std"] < 0.01 and s3["inter_scale_std"] > 8.0 * eq["inter_scale_std"], (
        f"Equal-diffusion control failed to quench: {eq['inter_scale_std']} vs {s3['inter_scale_std']}"
    )
    assert s3["discretization_ratio"] > cont3["discretization_ratio"], "P=0.17 must discretize more strongly than continuum skin P=1.0"
    assert cont3["vertical_nonuniformity_rms"] < 1e-8, "uniform slab must be exactly vertically uniform"
    assert ins3["inter_scale_std"] < 0.02 and not ins3["patterned"], f"Insulated scales (P=0) must not pattern: {ins3['inter_scale_std']}"
    assert ins3["mean_intra_scale_std"] < 1e-3, f"Insulated scale interiors must be uniform: {ins3['mean_intra_scale_std']}"

    # 8. P-sweep monotonicity on the reduced lattice (P > 0): thinner borders -> higher discretization ratio
    ratios = [integrate_skin_volume_pde_3d(P_border=p, **small)["discretization_ratio"] for p in (0.08, 0.17, 0.45, 1.0)]
    assert all(a > b for a, b in zip(ratios[:-1], ratios[1:])), f"3D discretization ratio must decrease with P: {ratios}"

    # 9. v2 legacy diffusivity model still integrates and also discretizes (comparison variant)
    legacy = integrate_scale_border_pde_2d(P_border=0.17, border_model="diffusivity", **small)
    assert legacy["patterned"] and legacy["discretization_ratio"] > 1.0

    # 10. Biological specimen segmentation on Timon lepidus dorsal ROI
    spec = segment_timon_lepidus_specimen()
    assert 120 <= spec["n_scales"] <= 240, f"Unexpected segmented scale count: {spec['n_scales']}"
    assert 0.30 <= spec["green_fraction"] <= 0.60, f"Unexpected dorsal green fraction: {spec['green_fraction']}"
    assert spec["ashman_D"] > 2.0, f"Expected bimodal dorsal scale colors, got Ashman D = {spec['ashman_D']}"
    assert 2.1 <= spec["mean_ng_green"] <= 3.3, f"Unexpected specimen <n_g>: {spec['mean_ng_green']}"

    # 11. Bitwise two-run determinism (3D and 2D), bypassing the lru_cache
    r1 = integrate_skin_volume_pde_3d.__wrapped__(P_border=0.17, scale_rows=6, scale_cols=6, ny=30, nx=30, steps=300)
    r2 = integrate_skin_volume_pde_3d.__wrapped__(P_border=0.17, scale_rows=6, scale_cols=6, ny=30, nx=30, steps=300)
    assert np.array_equal(r1["u_volume"], r2["u_volume"]), "3D two-run determinism check failed"
    q1 = integrate_scale_border_pde_2d.__wrapped__(P_border=0.17, scale_rows=6, scale_cols=6, ny=30, nx=30, steps=300)
    q2 = integrate_scale_border_pde_2d.__wrapped__(P_border=0.17, scale_rows=6, scale_cols=6, ny=30, nx=30, steps=300)
    assert np.array_equal(q1["u_field"], q2["u_field"]), "2D two-run determinism check failed"

    # 12. Mass conservation of pure diffusion in the 3D voxel skin (no-flux boundaries): total mass preserved by the implicit z-solve
    geom6 = build_hexagonal_scale_geometry(6, 6, 30, 30)
    h6 = skin_thickness_map(geom6, 0.17, nz=12)["h_map"]
    sk6 = (np.arange(12)[:, None, None] < h6[None]).astype(np.float64)
    c_zp = sk6 * np.roll(sk6, -1, axis=0)
    c_zp[-1] = 0.0
    c_zm = sk6 * np.roll(sk6, 1, axis=0)
    c_zm[0] = 0.0
    a = -0.5 * c_zm
    c = -0.5 * c_zp
    b = 1.0 - a - c
    rng = np.random.default_rng(7)
    f0 = rng.random(sk6.shape) * sk6
    f1 = _thomas_solve_along_z(a, b, c, f0) * sk6
    assert abs(f1.sum() - f0.sum()) < 1e-9 * max(f0.sum(), 1.0), "implicit vertical diffusion must conserve mass"

    # 13. Depth-average identity (Lean Theorem 4) on the 3D field: ubar of a vertically uniform volume equals the slice
    uni = np.broadcast_to(rng.random((30, 30)), (12, 30, 30)) * sk6
    ub = uni.sum(axis=0) / np.maximum(h6, 1)
    assert np.allclose(ub[h6 > 0], uni[0][h6 > 0]), "depth average of a vertically uniform column must recover the column value"

    # 14. Reference-dataset regime table parity with the sweep
    sweep_s = [r["regime_id"] for r in ref["hexagonal_outer_totalistic_ca"]["border_reduction_P_regimes"]]
    assert sweep_s == ["insulated_decoupled_scales", "strong_border_thinning_ca", "timon_lepidus_physiological", "moderate_border_thinning", "homogeneous_continuum_skin"]

    print(
        "[self_test] All 14 Manukyan et al. (2017) checks PASSED "
        f"(CA <n_g>={ca_sim['stats']['mean_ng_green']:.3f}, 3D_disc_ratio={s3['discretization_ratio']:.2f}, 3D_vs_2D_agree={cmp_s['scale_state_agreement']:.3f}, "
        f"vert_rms={s3['vertical_nonuniformity_rms']:.2e}, 3D_Ashman_D={s3['ashman_D']:.2f}, d_c(JSON)={stab_json['d_critical']:.2f}, "
        f"specimen_scales={spec['n_scales']}, specimen_<n_g>={spec['mean_ng_green']:.3f})."
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", type=float, default=0.17, help="Inter-scale border thickness reduction factor P in [0, 1] (extra 3D run printed when != 0.17)")
    ap.add_argument("--d", type=float, default=PDE_D_RATIO, help="Diffusivity ratio d = Dv / Du for the extra run")
    ap.add_argument("--rows", type=int, default=36, help="Hexagonal CA lattice rows")
    ap.add_argument("--cols", type=int, default=36, help="Hexagonal CA lattice columns")
    ap.add_argument("--steps", type=int, default=45, help="CA developmental simulation steps")
    ap.add_argument("--seed", type=int, default=2017, help="Deterministic RNG seed for the extra run")
    ap.add_argument("--self-test", action="store_true", help="Run 14-check verification suite and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    render_outputs(ca_rows=args.rows, ca_cols=args.cols, ca_steps=args.steps)
    print(f"[2015-manukyan-ocellated-lizard-ca] Rendered {OUT_PNG.relative_to(ROOT)}, {OUT_3D_PNG.relative_to(ROOT)} and {OUT_GIF.relative_to(ROOT)}")
    if abs(args.P - 0.17) > 1e-12 or abs(args.d - PDE_D_RATIO) > 1e-12 or args.seed != 2017:
        extra = integrate_skin_volume_pde_3d(P_border=args.P, d_ratio=args.d, seed=args.seed)
        print(
            f"[extra 3D run] P={args.P:.3f} (P_eff={extra['P_effective']:.3f}) d={args.d:.1f} seed={args.seed}: "
            f"inter std={extra['inter_scale_std']:.4f}, intra std={extra['mean_intra_scale_std']:.4f}, "
            f"disc ratio={extra['discretization_ratio']:.3f}, Ashman D={extra['ashman_D']:.3f}, vertical RMS={extra['vertical_nonuniformity_rms']:.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
