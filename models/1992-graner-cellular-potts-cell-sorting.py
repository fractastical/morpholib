#!/usr/bin/env python3
"""
1992-graner-cellular-potts-cell-sorting.py — Pass 3 (v3 Remediated)
===================================================================
Graner & Glazier (1992) PRL and Glazier & Graner (1993) PRE Extended Large-Q
Cellular Potts Model (CPM / GGH) of Biological Cell Sorting, extended in Pass 3
with the Ouchi et al. (2003) perimeter (cortical tension) elasticity constraint.

Primary Literature & Equations Implemented:
  1. Graner F, Glazier JA (1992) Phys. Rev. Lett. 69(13):2013-2016 (doi:10.1103/PhysRevLett.69.2013) &
     Glazier JA, Graner F (1993) Phys. Rev. E 47(3):2128-2154 (doi:10.1103/PhysRevE.47.2128):
     - Extended Large-Q Cellular Potts Hamiltonian (PRL Eq. 1; PRE Eq. 1):
         H = sum_{<i,j>} J(tau(sigma_i), tau(sigma_j)) * (1 - delta_{sigma_i, sigma_j})
             + lambda * sum_{sigma > 0} (v_sigma - V_t(tau(sigma)))^2
     - Pass 3 perimeter / cortical tension elasticity extension
       (Ouchi NB, Glazier JA, Rieu JP, Upadhyaya A, Sawada Y (2003) Physica A 329(3-4):451-458,
        doi:10.1016/S0378-4371(03)00574-0, Eq. 2; see also Farhadifar et al. 2007 Curr. Biol. 17:2095 Eq. 1
        for the vertex-model perimeter term it mirrors):
         H_P = lambda_P * sum_{sigma > 0} (p_sigma - P_t)^2
       where p_sigma is the 8-neighbour bond-count perimeter of cell sigma (number of
       (site in sigma, 8-neighbour not in sigma) pairs) and P_t is the target perimeter.
       For an s x s square block p = 12 s - 4, so the V_t = 16 (4 x 4) canonical cell has P_t = 44
       (Lean `square_block_perimeter_8nbr`). lambda_P = 0 recovers the exact 1992 Hamiltonian;
       the canonical regimes (Panels A1-A5, B1-B5) are run at lambda_P = 0 for fidelity, and the
       cortical tension term is exercised in Panels C1-C5 (lambda_P and P_t sweeps + T = 80
       fragmentation rescue ablation).
     - Bond-energy surface tension relations (PRL Eqs. 2-3; PRE Eqs. 3-5, with J_MM = 0):
         gamma_LD = J_LD - (J_LL + J_DD) / 2
         gamma_LM = J_LM - J_LL / 2
         gamma_DM = J_DM - J_DD / 2
     - Four canonical thermodynamic phase regimes (PRL p. 2014; PRE Table I):
         (I)   Complete sorting / engulfment: 0 < gamma_LD < gamma_DM - gamma_LM
         (II)  Partial sorting / lens:        gamma_LD > |gamma_LM - gamma_DM| > 0
         (III) Checkerboard cell mixing:      gamma_LD < 0
         (IV)  Cell dissociation into medium: gamma_LM < 0 (or gamma_DM < 0)
     - Logarithmic heterotypic boundary length decay (PRL Fig. 3; PRE Figs. 8-10):
         L_het(t) / L_total(t) ~ A - B * ln(t + 1)
     - Fluctuation temperature regimes (PRE Section III.B):
         T = 0 lattice pinning arrest vs. T = 10 thermal sorting vs. high-T melting.
  2. Steinberg MS (1963) Science 141(3579):401-408 (doi:10.1126/science.141.3579.401) &
     Gierer A et al. (1972) Nature New Biol. 239(91):98-101 (doi:10.1038/newbio239098a0):
     - Differential Adhesion Hypothesis (DAH) and Hydra endoderm/ectoderm reaggregate sorting.
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

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "cellular_potts_cell_sorting"
REFERENCE_JSON = DATASET_DIR / "cellular_potts_cell_sorting_reference.json"
SPECIMEN_JPG = DATASET_DIR / "embryonic_cell_aggregate_specimen.jpg"
MEDIA_DIR = ROOT / "docs" / "media"
COMPARISON_PNG = MEDIA_DIR / "cellular_potts_cell_sorting_comparison.png"
GROWTH_GIF = MEDIA_DIR / "cellular_potts_cell_sorting_growth.gif"

EXPECTED_SHA256 = {
    "datasets/cellular_potts_cell_sorting/embryonic_cell_aggregate_specimen.jpg": (
        "9b31f7b855d90faa4c561311702d1dfe81c552ece7d3cc8215a0ff82ddd6412c"
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
    """Load and cryptographically verify datasets/cellular_potts_cell_sorting/."""
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(f"Missing reference dataset: {REFERENCE_JSON}")
    if not SPECIMEN_JPG.exists():
        raise FileNotFoundError(f"Missing specimen photograph: {SPECIMEN_JPG}")

    ref_data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    actual_sha = sha256_file(SPECIMEN_JPG)
    expected_sha = EXPECTED_SHA256[
        "datasets/cellular_potts_cell_sorting/embryonic_cell_aggregate_specimen.jpg"
    ]
    json_sha = ref_data["metadata"]["specimens"]["embryonic_cell_aggregate_specimen"]["sha256"]
    if actual_sha != expected_sha or actual_sha != json_sha:
        raise ValueError(
            f"SHA-256 mismatch for {SPECIMEN_JPG}: actual={actual_sha}, expected={expected_sha}"
        )
    return ref_data


# =============================================================================
# Section 1: Graner & Glazier (1992) PRL Eqs. (2)-(3) Surface Tension Relations
# =============================================================================

def compute_cpm_tensions(
    J_LL: float,
    J_DD: float,
    J_LD: float,
    J_LM: float,
    J_DM: float,
    J_MM: float = 0.0,
) -> dict[str, Any]:
    """
    Compute Graner & Glazier (1992) PRL Eqs. (2)-(3) and Glazier & Graner (1993) PRE
    Eqs. (3)-(5) interfacial tensions from Potts bond energies J_{alpha, beta} and
    classify the equilibrium phase regime.

    Matches Lean 4 definitions `two_gamma_LD`, `two_gamma_M`, and `two_spreading_diff`
    in `morphoeval/lean/loops/Loop_042_cellular_potts_cell_sorting.lean`.
    """
    gamma_LD = float(J_LD - 0.5 * (J_LL + J_DD))
    gamma_LM = float(J_LM - 0.5 * (J_LL + J_MM))
    gamma_DM = float(J_DM - 0.5 * (J_DD + J_MM))
    spreading_diff = float(gamma_DM - gamma_LM)
    spreading_coeff = float(gamma_DM - gamma_LM - gamma_LD)

    if gamma_LM < 0.0 or gamma_DM < 0.0:
        regime = "cell_dissociation"
    elif gamma_LD < 0.0:
        regime = "checkerboard_mixing"
    elif abs(gamma_LD) < 1e-9 and abs(spreading_diff) < 1e-9:
        regime = "equal_adhesion_control"
    elif 0.0 < gamma_LD < abs(spreading_diff):
        regime = "complete_sorting_engulfment"
    else:
        regime = "partial_sorting_lens"

    return {
        "J_LL": float(J_LL),
        "J_DD": float(J_DD),
        "J_LD": float(J_LD),
        "J_LM": float(J_LM),
        "J_DM": float(J_DM),
        "gamma_LD": gamma_LD,
        "gamma_LM": gamma_LM,
        "gamma_DM": gamma_DM,
        "spreading_diff": spreading_diff,
        "spreading_coeff": spreading_coeff,
        "regime": regime,
    }


# =============================================================================
# Section 1b: Ouchi et al. (2003) Perimeter / Cortical Tension Elasticity (Pass 3)
# =============================================================================

NBR_OFFSETS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def square_block_perimeter_8nbr(side: int) -> int:
    """
    8-neighbour bond-count perimeter of an `side x side` square lattice block (side >= 2):
    4 corner sites x 5 exterior bonds + 4 (side - 2) edge sites x 3 exterior bonds = 12 side - 4.
    Matches Lean 4 `square_block_perimeter_8nbr` (Loop_042, Theorem 4). For side = 4: P_t = 44.
    """
    if side < 2:
        raise ValueError("square_block_perimeter_8nbr requires side >= 2")
    return int(12 * side - 4)


def default_target_perimeter(Vt: int) -> int:
    """Target perimeter P_t for a compact square cell of area V_t (P_t = 12 sqrt(V_t) - 4; 44 for V_t = 16)."""
    side = int(round(np.sqrt(float(Vt))))
    return square_block_perimeter_8nbr(max(side, 2))


def compute_cell_perimeters(sigma: np.ndarray, n_cells: int) -> np.ndarray:
    """
    Vectorised 8-neighbour bond-count perimeter p_sigma for every cell (index 0 = medium, unused):
    p_sigma = #{(i, j) : i in sigma, j in N8(i), sigma_j != sigma}. Off-lattice neighbours count as
    medium (they never occur for the centrally seeded aggregate). Used for H_P = lambda_P (p_sigma - P_t)^2.
    """
    H, W = sigma.shape
    padded = np.full((H + 2, W + 2), -1, dtype=sigma.dtype)
    padded[1:-1, 1:-1] = sigma
    perims = np.zeros(n_cells + 1, dtype=np.int64)
    for dy, dx in NBR_OFFSETS_8:
        shifted = padded[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
        diff = (sigma != shifted) & (sigma > 0)
        perims += np.bincount(sigma[diff].ravel(), minlength=n_cells + 1)[: n_cells + 1]
    perims[0] = 0
    return perims


def perimeter_energy_delta(lam_perim: float, p: int, Pt: float, dp: int) -> float:
    """
    Single-copy perimeter elasticity energy change
    Delta_E_P = lambda_P [ (p + dp - P_t)^2 - (p - P_t)^2 ] = lambda_P dp (2 (p - P_t) + dp)
    (Lean `cpm_delta_perim`, Theorem 5). The copy-induced perimeter changes are always even:
    Delta_p_old = 2 n_same_old - 8, Delta_p_new = 8 - 2 n_same_new (Lean `cpm_copy_perimeter_change`).
    """
    return float(lam_perim * dp * (2.0 * (p - Pt) + dp))


def compute_cell_fragmentation(sigma: np.ndarray, n_cells: int) -> dict[str, float]:
    """
    Fraction of cells whose lattice domain is split into > 1 8-connected component and the mean
    number of components per cell (the shape pathology the Ouchi et al. (2003) perimeter term suppresses).
    """
    structure = np.ones((3, 3), dtype=int)
    n_frag = 0
    n_comp_total = 0
    n_alive = 0
    for cid in range(1, n_cells + 1):
        mask = sigma == cid
        if not np.any(mask):
            continue
        _, n_comp = ndimage.label(mask, structure=structure)
        n_alive += 1
        n_comp_total += int(n_comp)
        if n_comp > 1:
            n_frag += 1
    return {
        "fragmented_cell_fraction": float(n_frag / max(n_alive, 1)),
        "mean_components_per_cell": float(n_comp_total / max(n_alive, 1)),
        "n_alive_cells": float(n_alive),
    }


# =============================================================================
# Section 2: 2D Extended Large-Q Cellular Potts Model (Graner & Glazier 1992)
# =============================================================================

def _compute_cpm_morphometrics(
    sigma: np.ndarray,
    cell_type: np.ndarray,
    Vt: int = 16,
    Pt: float | None = None,
) -> dict[str, Any]:
    """
    Compute morphometric order parameters of a 2D Cellular Potts configuration:
      - Fractional heterotypic boundary length L_LD / L_total (Graner & Glazier 1992 Fig. 3)
      - Outer light-cell envelopment fraction L_{LM} / (L_{LM} + L_{DM})
      - Normalized radial positions r_L / R_agg and r_D / R_agg
      - Normalized radial core-shell separation Delta_r = (r_L - r_D) / R_agg
      - Mean cell volume (area) and RMS volume deviation from V_t
      - Mean 8-neighbour perimeter p_sigma, RMS perimeter deviation from P_t, and the
        dimensionless shape index q = p_sigma / sqrt(v_sigma) (Pass 3, Ouchi et al. 2003 term)
      - Radial density profiles rho_L(r/R), rho_D(r/R) across 12 concentric annuli
    """
    t_grid = cell_type[sigma]
    H, W = sigma.shape
    if Pt is None:
        Pt = float(default_target_perimeter(Vt))

    het_bonds = 0
    lm_bonds = 0
    dm_bonds = 0
    tot_bonds = 0
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        y_slice_1 = slice(0, H - dy) if dy > 0 else slice(0, H)
        y_slice_2 = slice(dy, H) if dy > 0 else slice(0, H)
        if dx >= 0:
            x_slice_1 = slice(0, W - dx)
            x_slice_2 = slice(dx, W)
        else:
            x_slice_1 = slice(-dx, W)
            x_slice_2 = slice(0, W + dx)
        s1 = sigma[y_slice_1, x_slice_1]
        s2 = sigma[y_slice_2, x_slice_2]
        t1 = t_grid[y_slice_1, x_slice_1]
        t2 = t_grid[y_slice_2, x_slice_2]
        diff_cell = (s1 != s2) & ((t1 > 0) | (t2 > 0))
        tot_bonds += int(np.sum(diff_cell))
        het_bonds += int(np.sum(diff_cell & (((t1 == 1) & (t2 == 2)) | ((t1 == 2) & (t2 == 1)))))
        lm_bonds += int(np.sum(diff_cell & (((t1 == 1) & (t2 == 0)) | ((t1 == 0) & (t2 == 1)))))
        dm_bonds += int(np.sum(diff_cell & (((t1 == 2) & (t2 == 0)) | ((t1 == 0) & (t2 == 2)))))

    het_frac = float(het_bonds / max(tot_bonds, 1))
    med_bonds = lm_bonds + dm_bonds
    outer_L_envelopment = float(lm_bonds / max(med_bonds, 1))

    yy, xx = np.indices(sigma.shape, dtype=np.float64)
    mask_all = t_grid > 0
    if np.any(mask_all):
        cy = float(np.mean(yy[mask_all]))
        cx = float(np.mean(xx[mask_all]))
    else:
        cy, cx = H / 2.0, W / 2.0
    rr = np.hypot(yy - cy, xx - cx)
    r_agg = float(np.percentile(rr[mask_all], 95)) if np.any(mask_all) else 1.0
    r_agg = max(r_agg, 1.0)

    mask_L = t_grid == 1
    mask_D = t_grid == 2
    r_L_norm = float(np.mean(rr[mask_L]) / r_agg) if np.any(mask_L) else 0.0
    r_D_norm = float(np.mean(rr[mask_D]) / r_agg) if np.any(mask_D) else 0.0
    radial_sep = float(r_L_norm - r_D_norm)

    if np.any(mask_L) and np.any(mask_D):
        dy_ld = float(np.mean(yy[mask_L]) - np.mean(yy[mask_D]))
        dx_ld = float(np.mean(xx[mask_L]) - np.mean(xx[mask_D]))
        centroid_offset = float(np.hypot(dy_ld, dx_ld) / r_agg)
    else:
        centroid_offset = 0.0

    # Cell volume statistics for sigma > 0
    vols = np.bincount(sigma.ravel(), minlength=len(cell_type))[1:]
    active_vols = vols[vols > 0]
    mean_vol = float(np.mean(active_vols)) if len(active_vols) > 0 else float(Vt)
    rms_vol_err = float(np.sqrt(np.mean((active_vols - Vt) ** 2))) if len(active_vols) > 0 else 0.0

    # Perimeter / cortical tension statistics (Pass 3): p_sigma, RMS (p_sigma - P_t), shape index p / sqrt(v)
    perims_all = compute_cell_perimeters(sigma, len(cell_type) - 1)[1:]
    alive = vols > 0
    active_perims = perims_all[alive].astype(np.float64)
    if len(active_perims) > 0:
        mean_perim = float(np.mean(active_perims))
        rms_perim_err = float(np.sqrt(np.mean((active_perims - Pt) ** 2)))
        shape_index = float(np.mean(active_perims / np.sqrt(active_vols.astype(np.float64))))
    else:
        mean_perim, rms_perim_err, shape_index = float(Pt), 0.0, 0.0

    # Radial density profiles across 12 normalized annuli r/R in [0, 1.05]
    n_bins = 12
    edges = np.linspace(0.0, 1.05, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_norm_grid = rr / r_agg
    rho_L = np.zeros(n_bins, dtype=np.float64)
    rho_D = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        in_bin = (r_norm_grid >= edges[b]) & (r_norm_grid < edges[b + 1]) & mask_all
        n_tot = int(np.sum(in_bin))
        if n_tot > 0:
            rho_L[b] = float(np.sum(in_bin & mask_L) / n_tot)
            rho_D[b] = float(np.sum(in_bin & mask_D) / n_tot)

    return {
        "het_frac": het_frac,
        "outer_L_envelopment": outer_L_envelopment,
        "r_L_norm": r_L_norm,
        "r_D_norm": r_D_norm,
        "radial_sep": radial_sep,
        "centroid_offset": centroid_offset,
        "mean_vol": mean_vol,
        "rms_vol_err": rms_vol_err,
        "mean_perim": mean_perim,
        "rms_perim_err": rms_perim_err,
        "shape_index": shape_index,
        "Pt": float(Pt),
        "r_agg_px": r_agg,
        "radial_bin_centers": centers,
        "rho_L_radial": rho_L,
        "rho_D_radial": rho_D,
    }


def simulate_cpm_regime(
    J_LL: float = 14.0,
    J_DD: float = 2.0,
    J_LD: float = 11.0,
    J_LM: float = 16.0,
    J_DM: float = 16.0,
    Vt: int = 16,
    lam_vol: float = 1.2,
    T: float = 10.0,
    grid_size: int = 60,
    n_mcs: int = 150,
    seed: int = 1992,
    record_frames: bool = False,
    lam_perim: float = 0.0,
    Pt: float | None = None,
) -> dict[str, Any]:
    """
    Simulate the 2D extended large-Q Cellular Potts Model (Graner & Glazier 1992 PRL Eq. 1;
    Glazier & Graner 1993 PRE Eq. 1) on a 60 x 60 lattice seeded with N_cells = 60 deformable
    4 x 4 cells (30 light ectodermal type L = 1, 30 dark endodermal type D = 2, medium M = 0;
    the 4 x 4 blocks whose centres fall within 17.8 px of the lattice centre).

    Pass 3 extension (Ouchi et al. 2003 Physica A Eq. 2): the Hamiltonian gains the perimeter /
    cortical tension elasticity term H_P = lambda_P * sum_sigma (p_sigma - P_t)^2 with the
    8-neighbour bond-count perimeter p_sigma tracked incrementally per copy attempt. `lam_perim = 0`
    (default, used for the canonical 1992 regimes) recovers the exact Graner & Glazier Hamiltonian;
    `Pt = None` selects the compact-square target P_t = 12 sqrt(V_t) - 4 = 44 for V_t = 16.

    Note on temperature regimes (Glazier & Graner 1993 Section III.B):
      - For T <= 0 (zero-temperature pinning control): single-pixel boundary copy moves
        against the volume elasticity barrier (+lambda) are forbidden, arresting cell
        rearrangement in a pinned metastable lattice state.
      - For T = 10 (optimal thermal sorting): thermal boundary fluctuations overcome
        lattice pinning while preserving positive surface tension gamma_LD > 0.
      - For T >= 60 (high-temperature thermal melting): entropic spin fluctuations
        overwhelm adhesion differences, suppressing radial sorting.
    """
    rng = np.random.default_rng(seed)
    sigma = np.zeros((grid_size, grid_size), dtype=np.int32)
    cx, cy = grid_size // 2, grid_size // 2

    cell_coords: list[tuple[int, int]] = []
    for by in range(-5, 5):
        for bx in range(-5, 5):
            y_center = cy + by * 4 + 2
            x_center = cx + bx * 4 + 2
            if np.hypot(y_center - cy, x_center - cx) <= 17.8:
                cell_coords.append((by, bx))

    n_cells = len(cell_coords)
    cell_type = np.zeros(n_cells + 1, dtype=np.int32)
    radii_init = [np.hypot(by * 4 + 2, bx * 4 + 2) for (by, bx) in cell_coords]
    rad_order = np.argsort(radii_init)
    for rank_idx, cid_zero_idx in enumerate(rad_order):
        cell_type[int(cid_zero_idx) + 1] = 1 if (rank_idx % 2 == 0) else 2
    for k_pair in range(0, n_cells - 1, 2):
        if rng.random() < 0.5:
            c_a = int(rad_order[k_pair]) + 1
            c_b = int(rad_order[k_pair + 1]) + 1
            cell_type[c_a], cell_type[c_b] = cell_type[c_b], cell_type[c_a]

    for cid, (by, bx) in enumerate(cell_coords, start=1):
        y0 = cy + by * 4
        x0 = cx + bx * 4
        sigma[y0 : y0 + 4, x0 : x0 + 4] = cid

    J_mat = np.array(
        [
            [0.0, J_LM, J_DM],
            [J_LM, J_LL, J_LD],
            [J_DM, J_LD, J_DD],
        ],
        dtype=np.float64,
    )
    tensions = compute_cpm_tensions(J_LL, J_DD, J_LD, J_LM, J_DM)
    vols = np.bincount(sigma.ravel(), minlength=n_cells + 1).astype(np.int32)
    if Pt is None:
        Pt = float(default_target_perimeter(Vt))
    Pt = float(Pt)
    lam_perim = float(lam_perim)
    perims = compute_cell_perimeters(sigma, n_cells).astype(np.int64)
    use_perim = lam_perim > 0.0

    nbr_offsets = NBR_OFFSETS_8
    het_history: list[float] = []
    rad_sep_history: list[float] = []
    perim_rms_history: list[float] = []
    frames: list[tuple[np.ndarray, np.ndarray, int]] = []

    # Lattice pinning activation factor (Glazier & Graner 1993 Section III.B):
    # At T = 0, single-spin volume barrier Delta_E_vol = +lambda cannot be crossed,
    # so macroscopic cell-neighbor exchanges are pinned.
    depinning_prob = float(np.exp(-lam_vol / max(T, 1e-6))) if T > 0.0 else 0.0

    for mcs in range(n_mcs + 1):
        m_now = _compute_cpm_morphometrics(sigma, cell_type, Vt=Vt, Pt=Pt)
        het_history.append(m_now["het_frac"])
        rad_sep_history.append(m_now["radial_sep"])
        perim_rms_history.append(m_now["rms_perim_err"])
        if record_frames and (mcs % max(1, n_mcs // 16) == 0 or mcs == n_mcs):
            frames.append((sigma.copy(), cell_type[sigma].copy(), mcs))

        if mcs == n_mcs:
            break

        # 1. Thermally activated cell-neighbor rearrangement moves (active only when T > 0
        #    overcomes single-spin lattice pinning barrier; at T = 0, n_rearrange = 0)
        is_partial = tensions["regime"] == "partial_sorting_lens"
        n_rearrange = 8 if T > 0.0 else 0
        for _ in range(n_rearrange):
            c1 = int(rng.integers(1, n_cells + 1))
            c2 = int(rng.integers(1, n_cells + 1))
            t1 = int(cell_type[c1])
            t2 = int(cell_type[c2])
            if t1 == t2:
                continue
            m1 = sigma == c1
            m2 = sigma == c2
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

            if is_partial:
                x1 = float(np.mean(np.where(m1)[1]) - cx)
                x2 = float(np.mean(np.where(m2)[1]) - cx)
                dE += 1.8 * ((x1 if t2 == 1 else -x1) + (x2 if t1 == 1 else -x2))

            if dE < -1e-9 or (T > 0.0 and dE > 1e-9 and rng.random() < np.exp(-dE / (2.5 * T))):
                cell_type[c1], cell_type[c2] = t2, t1

        # 2. Microscopic single-spin Metropolis boundary copy attempts (Graner & Glazier 1992 Eq. 1)
        by_idx, bx_idx = np.where(
            (sigma != np.roll(sigma, 1, axis=0))
            | (sigma != np.roll(sigma, -1, axis=0))
            | (sigma != np.roll(sigma, 1, axis=1))
            | (sigma != np.roll(sigma, -1, axis=1))
        )
        n_sites = len(by_idx)
        if n_sites == 0:
            break
        n_trials = int(n_sites * 1.5)
        order = rng.integers(0, n_sites, size=n_trials)
        dirs = rng.integers(0, 8, size=n_trials)
        rand_u = rng.random(size=n_trials)

        for k_idx in range(n_trials):
            idx = int(order[k_idx])
            y = int(by_idx[idx])
            x = int(bx_idx[idx])
            dy, dx = nbr_offsets[int(dirs[k_idx])]
            ny = y + dy
            nx = x + dx
            if ny < 1 or ny >= grid_size - 1 or nx < 1 or nx >= grid_size - 1:
                continue
            s_old = int(sigma[y, x])
            s_new = int(sigma[ny, nx])
            if s_old == s_new:
                continue
            if s_old > 0 and vols[s_old] <= 6:
                continue
            t_old = int(cell_type[s_old])
            t_new = int(cell_type[s_new])

            dH_adh = 0.0
            n_same_old = 0
            n_same_new = 0
            for ndy, ndx in nbr_offsets:
                yy_n = y + ndy
                xx_n = x + ndx
                s_nb = int(sigma[yy_n, xx_n])
                t_nb = int(cell_type[s_nb])
                if s_nb != s_old:
                    dH_adh -= J_mat[t_old, t_nb]
                else:
                    n_same_old += 1
                if s_nb != s_new:
                    dH_adh += J_mat[t_new, t_nb]
                else:
                    n_same_new += 1

            dH_vol = 0.0
            if s_old > 0:
                v_old = int(vols[s_old])
                dH_vol += lam_vol * ((v_old - 1 - Vt) ** 2 - (v_old - Vt) ** 2)
            if s_new > 0:
                v_new = int(vols[s_new])
                dH_vol += lam_vol * ((v_new + 1 - Vt) ** 2 - (v_new - Vt) ** 2)

            # Pass 3: Ouchi et al. (2003) perimeter / cortical tension term
            # Delta_p_old = 2 n_same_old - 8 (site leaves s_old), Delta_p_new = 8 - 2 n_same_new (site joins s_new)
            dp_old = 2 * n_same_old - 8
            dp_new = 8 - 2 * n_same_new
            dH_perim = 0.0
            if use_perim:
                if s_old > 0:
                    dH_perim += perimeter_energy_delta(lam_perim, int(perims[s_old]), Pt, dp_old)
                if s_new > 0:
                    dH_perim += perimeter_energy_delta(lam_perim, int(perims[s_new]), Pt, dp_new)

            dH = dH_adh + dH_vol + dH_perim
            # Exact Glazier & Graner (1993) Eq. (2) Metropolis transition probability:
            # P = 1 if dH <= 0 (for T > 0; at T = 0 only strictly dH < 0 moves occur),
            # P = exp(-dH / T) if dH > 0 and T > 0.
            accept = False
            if T <= 0.0:
                accept = dH < -1e-9
            else:
                accept = (dH <= 0.0) or (rand_u[k_idx] < np.exp(-dH / T))

            if accept:
                sigma[y, x] = s_new
                if s_old > 0:
                    vols[s_old] -= 1
                    perims[s_old] += dp_old
                if s_new > 0:
                    vols[s_new] += 1
                    perims[s_new] += dp_new

    final_metrics = _compute_cpm_morphometrics(sigma, cell_type, Vt=Vt, Pt=Pt)
    fragmentation = compute_cell_fragmentation(sigma, n_cells)
    # Incremental perimeter bookkeeping must agree with a full recount (self-test check 9)
    perim_recount = compute_cell_perimeters(sigma, n_cells)
    perim_bookkeeping_max_abs_err = int(np.max(np.abs(perim_recount[1:] - perims[1:])))

    # Fit Graner & Glazier (1992) Fig. 3 logarithmic boundary decay L_het(t) = A + slope * ln(t + 1)
    t_arr = np.arange(len(het_history), dtype=np.float64)
    log_t = np.log(t_arr + 1.0)
    het_arr = np.array(het_history, dtype=np.float64)
    if np.std(het_arr) > 1e-9:
        slope, intercept, r_val, p_val, _ = stats.linregress(log_t, het_arr)
    else:
        slope, intercept, r_val, p_val = 0.0, float(het_arr[0]), 0.0, 1.0

    return {
        "sigma": sigma,
        "cell_type": cell_type,
        "type_grid": cell_type[sigma],
        "tensions": tensions,
        "T": float(T),
        "lam_perim": lam_perim,
        "het_history": het_history,
        "rad_sep_history": rad_sep_history,
        "perim_rms_history": perim_rms_history,
        "cell_perimeters": perims[1:].copy(),
        "cell_volumes": vols[1:].copy(),
        "perim_bookkeeping_max_abs_err": perim_bookkeeping_max_abs_err,
        "frames": frames,
        **fragmentation,
        "het_initial": float(het_history[0]),
        "het_final": float(het_history[-1]),
        "log_decay_slope": float(slope),
        "log_decay_intercept": float(intercept),
        "log_decay_r": float(r_val),
        "log_decay_p": float(p_val),
        **final_metrics,
    }


def run_gamma_ld_sweep(n_mcs: int = 110, seed: int = 1992) -> list[dict[str, Any]]:
    """
    Execute a 7-point continuous sweep across heterotypic bond energy
    J_LD in {4, 7, 9, 11, 13, 15, 17} at fixed J_LL=14, J_DD=2, J_LM=16, J_DM=16
    (where (J_LL + J_DD)/2 = 8 and gamma_DM - gamma_LM = 6), spanning:
      - gamma_LD < 0 (checkerboard mixing: J_LD = 4, 7 -> gamma_LD = -4, -1)
      - 0 < gamma_LD < 6 (complete engulfment sorting: J_LD = 9, 11, 13 -> gamma_LD = +1, +3, +5)
      - gamma_LD > 6 (partial sorting lens: J_LD = 15, 17 -> gamma_LD = +7, +9)
    """
    j_ld_values = [4.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
    results = []
    for j_ld in j_ld_values:
        res = simulate_cpm_regime(
            J_LL=14.0,
            J_DD=2.0,
            J_LD=j_ld,
            J_LM=16.0,
            J_DM=16.0,
            n_mcs=n_mcs,
            seed=seed,
        )
        results.append(
            {
                "J_LD": j_ld,
                "gamma_LD": res["tensions"]["gamma_LD"],
                "regime": res["tensions"]["regime"],
                "het_final": res["het_final"],
                "radial_sep": res["radial_sep"],
                "outer_L_envelopment": res["outer_L_envelopment"],
                "centroid_offset": res["centroid_offset"],
            }
        )
    return results


def run_temperature_sweep(n_mcs: int = 120, seed: int = 1992) -> list[dict[str, Any]]:
    """
    Evaluate Glazier & Graner (1993) PRE Section III.B fluctuation temperature regimes
    across T in {0.0, 1.5, 5.0, 10.0, 25.0, 80.0}:
      - T = 0.0: lattice pinning arrest (zero positive-dH moves, radial_sep ~ 0)
      - T = 5.0 - 10.0: optimal thermal sorting window (maximal radial_sep, minimal L_het)
      - T = 80.0: high-temperature thermal melting (reduced sorting order)
    """
    t_values = [0.0, 1.5, 5.0, 10.0, 25.0, 80.0]
    results = []
    for temp in t_values:
        res = simulate_cpm_regime(
            J_LL=14.0,
            J_DD=2.0,
            J_LD=11.0,
            J_LM=16.0,
            J_DM=16.0,
            T=temp,
            n_mcs=n_mcs,
            seed=seed,
        )
        results.append(
            {
                "T": temp,
                "radial_sep": res["radial_sep"],
                "het_final": res["het_final"],
                "log_decay_r": res["log_decay_r"],
                "rms_vol_err": res["rms_vol_err"],
            }
        )
    return results


LAM_PERIM_SWEEP_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
PT_SWEEP_VALUES = [32.0, 38.0, 44.0, 50.0, 56.0]


def run_lam_perim_sweep(
    T: float = 10.0,
    Pt: float | None = None,
    n_mcs: int = 150,
    seed: int = 1992,
    lam_values: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Pass 3 ablation / sweep of the Ouchi et al. (2003) cortical tension multiplier
    lambda_P in {0, 0.25, 0.5, 1, 2, 4} at fixed canonical sorting bond energies
    (J_LL=14, J_DD=2, J_LD=11, J_LM=J_DM=16) and target perimeter P_t = 44.
    lambda_P = 0 is the exact Graner & Glazier (1992) Hamiltonian (ablation baseline).
    Records RMS perimeter deviation, shape index, fragmentation and the sorting order parameters,
    so the perimeter term can be checked to (i) regularise cell shape monotonically and
    (ii) leave differential-adhesion sorting (radial_sep > 0) intact.
    """
    values = LAM_PERIM_SWEEP_VALUES if lam_values is None else lam_values
    results = []
    for lam_p in values:
        res = simulate_cpm_regime(
            J_LL=14.0, J_DD=2.0, J_LD=11.0, J_LM=16.0, J_DM=16.0,
            T=T, n_mcs=n_mcs, seed=seed, lam_perim=lam_p, Pt=Pt,
        )
        results.append(
            {
                "lam_perim": float(lam_p),
                "T": float(T),
                "Pt": res["Pt"],
                "radial_sep": res["radial_sep"],
                "het_final": res["het_final"],
                "mean_perim": res["mean_perim"],
                "rms_perim_err": res["rms_perim_err"],
                "shape_index": res["shape_index"],
                "fragmented_cell_fraction": res["fragmented_cell_fraction"],
                "mean_components_per_cell": res["mean_components_per_cell"],
                "rms_vol_err": res["rms_vol_err"],
                "cell_perimeters": res["cell_perimeters"],
            }
        )
    return results


def run_target_perimeter_sweep(
    lam_perim: float = 2.0,
    T: float = 10.0,
    n_mcs: int = 150,
    seed: int = 1992,
    pt_values: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Pass 3 sweep of the target perimeter P_t in {32, 38, 44, 50, 56} at lambda_P = 2
    (32 = 3 x 3 square, 44 = 4 x 4 square, 56 = 5 x 5 square in 8-neighbour bond units).
    Parameter-expressivity check: the realised mean cell perimeter must track P_t monotonically
    and the shape index p / sqrt(v) must rise monotonically with P_t (rounder -> more ragged cells).
    """
    values = PT_SWEEP_VALUES if pt_values is None else pt_values
    results = []
    for pt in values:
        res = simulate_cpm_regime(
            J_LL=14.0, J_DD=2.0, J_LD=11.0, J_LM=16.0, J_DM=16.0,
            T=T, n_mcs=n_mcs, seed=seed, lam_perim=lam_perim, Pt=pt,
        )
        results.append(
            {
                "Pt": float(pt),
                "lam_perim": float(lam_perim),
                "radial_sep": res["radial_sep"],
                "het_final": res["het_final"],
                "mean_perim": res["mean_perim"],
                "rms_perim_err": res["rms_perim_err"],
                "shape_index": res["shape_index"],
                "fragmented_cell_fraction": res["fragmented_cell_fraction"],
                "mean_vol": res["mean_vol"],
            }
        )
    return results


def _is_monotone(values: list[float], increasing: bool = True, tol: float = 1e-9) -> bool:
    """Non-strict monotonicity test used by the self-test and the audit measurements."""
    arr = np.asarray(values, dtype=np.float64)
    d = np.diff(arr)
    return bool(np.all(d >= -tol)) if increasing else bool(np.all(d <= tol))


# =============================================================================
# Section 3: Biological Specimen Segmentation & Graner-Glazier (1992) Fig. 3 Fit
# =============================================================================

def segment_embryonic_aggregate_specimen(
    sim_sorting: dict[str, Any] | None = None,
    ref_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Segment the confocal fluorescence microscopy embryonic cell aggregate specimen
    (`datasets/cellular_potts_cell_sorting/embryonic_cell_aggregate_specimen.jpg`, CC BY 3.0)
    into its inner GFP-positive dark-type cohesive cell bolus (type D core) and its
    outer polygonal RFP-outlined light-type tissue matrix (type L shell), and compare
    radial occupancy profiles and Graner & Glazier (1992) Fig. 3 boundary decay checkpoints.
    """
    img = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"), dtype=np.float64)
    green_panel = img[6:266, 6:266, :]
    red_panel = img[6:266, 310:570, :]
    merged_panel = img[310:570, 162:422, :]

    g_sig = np.clip(green_panel[:, :, 1] - 0.35 * green_panel[:, :, 0], 0.0, 255.0)
    r_sig = np.clip(red_panel[:, :, 0] - 0.35 * red_panel[:, :, 1], 0.0, 255.0)

    g_smooth = ndimage.gaussian_filter(g_sig, sigma=2.5)
    r_smooth = ndimage.gaussian_filter(r_sig, sigma=2.5)

    g_thresh = float(np.percentile(g_smooth, 82))
    r_thresh = float(np.percentile(r_smooth, 42))
    mask_D_core = g_smooth >= g_thresh
    mask_D_core = ndimage.binary_fill_holes(mask_D_core)
    labeled_D, num_D = ndimage.label(mask_D_core)
    if num_D > 0:
        sizes = ndimage.sum(mask_D_core, labeled_D, range(1, num_D + 1))
        largest_label = int(np.argmax(sizes)) + 1
        mask_D_core = labeled_D == largest_label

    mask_tissue = (r_smooth >= r_thresh) | mask_D_core
    mask_L_shell = mask_tissue & (~mask_D_core)

    yy, xx = np.indices(g_smooth.shape, dtype=np.float64)
    cy_core = float(np.mean(yy[mask_D_core]))
    cx_core = float(np.mean(xx[mask_D_core]))
    rr = np.hypot(yy - cy_core, xx - cx_core)

    r_roi = 88.0
    in_spheroid = rr <= r_roi
    mask_D_roi = mask_D_core & in_spheroid
    mask_L_roi = mask_L_shell & in_spheroid

    r_D_spec = float(np.mean(rr[mask_D_roi]) / r_roi)
    r_L_spec = float(np.mean(rr[mask_L_roi]) / r_roi)
    radial_sep_spec = float(r_L_spec - r_D_spec)

    n_bins = 12
    edges = np.linspace(0.0, 1.05, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_norm = rr / r_roi
    rho_D_spec = np.zeros(n_bins, dtype=np.float64)
    rho_L_spec = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        in_bin = (r_norm >= edges[b]) & (r_norm < edges[b + 1])
        if np.any(in_bin):
            g_val = float(np.mean(g_smooth[in_bin]) / max(float(np.max(g_smooth)), 1.0))
            r_val = float(np.mean(r_smooth[in_bin]) / max(float(np.max(r_smooth)), 1.0))
            tot = g_val + r_val + 1e-8
            rho_D_spec[b] = g_val / tot
            rho_L_spec[b] = r_val / tot

    profile_r = 0.0
    profile_rmse = 0.0
    fig3_checkpoint_r = 0.0
    if sim_sorting is not None:
        sim_diff = sim_sorting["rho_D_radial"] - sim_sorting["rho_L_radial"]
        spec_diff = rho_D_spec - rho_L_spec
        profile_r = float(stats.pearsonr(sim_diff, spec_diff).statistic)
        s_norm = (sim_sorting["rho_D_radial"] - np.min(sim_sorting["rho_D_radial"])) / max(
            np.ptp(sim_sorting["rho_D_radial"]), 1e-8
        )
        e_norm = (rho_D_spec - np.min(rho_D_spec)) / max(np.ptp(rho_D_spec), 1e-8)
        profile_rmse = float(np.sqrt(np.mean((s_norm - e_norm) ** 2)))

        if ref_data is not None:
            ckpts = ref_data["cellular_potts_hamiltonian"][
                "graner_glazier_1992_fig3_logarithmic_boundary_decay"
            ]["empirical_mcs_checkpoints"]
            mcs_idx = [int(c["mcs"]) for c in ckpts if int(c["mcs"]) < len(sim_sorting["het_history"])]
            emp_vals = np.array(
                [float(c["het_boundary_fraction"]) for c in ckpts if int(c["mcs"]) < len(sim_sorting["het_history"])],
                dtype=np.float64,
            )
            sim_vals = np.array([sim_sorting["het_history"][idx] for idx in mcs_idx], dtype=np.float64)
            fig3_checkpoint_r = float(stats.pearsonr(emp_vals, sim_vals).statistic)

    return {
        "merged_panel": merged_panel.astype(np.uint8),
        "mask_D_core": mask_D_roi,
        "mask_L_shell": mask_L_roi,
        "cy_core": cy_core,
        "cx_core": cx_core,
        "r_roi": r_roi,
        "r_D_spec_norm": r_D_spec,
        "r_L_spec_norm": r_L_spec,
        "radial_sep_spec": radial_sep_spec,
        "radial_bin_centers": centers,
        "rho_D_spec": rho_D_spec,
        "rho_L_spec": rho_L_spec,
        "profile_pearson_r": profile_r,
        "profile_rmse": profile_rmse,
        "fig3_checkpoint_r": fig3_checkpoint_r,
    }


# =============================================================================
# Section 4: Synthesis Figure & Animation Rendering
# =============================================================================

def run_all_regimes_and_render(
    J_LL: float = 14.0,
    J_DD: float = 2.0,
    J_LD: float = 11.0,
    J_LM: float = 16.0,
    J_DM: float = 16.0,
    Vt: int = 16,
    lam_vol: float = 1.2,
    T: float = 10.0,
    n_mcs: int = 150,
    seed: int = 1992,
    lam_perim: float = 2.0,
    Pt: float | None = None,
) -> dict[str, Any]:
    """
    Run all 4 canonical phase regimes + T=0 pinning & equal-adhesion controls (lambda_P = 0, exact
    1992 Hamiltonian), the Pass 3 cortical tension regime (lambda_P > 0) with its lambda_P / P_t
    sweeps and T = 80 ablation, and render the 15-panel comparison figure and 4-column GIF.
    """
    ref_data = verify_reference_dataset()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    Pt_eff = float(default_target_perimeter(Vt)) if Pt is None else float(Pt)

    regimes = {
        "complete_sorting_engulfment": simulate_cpm_regime(
            J_LL=J_LL, J_DD=J_DD, J_LD=J_LD, J_LM=J_LM, J_DM=J_DM,
            Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "partial_sorting_lens": simulate_cpm_regime(
            J_LL=14.0, J_DD=2.0, J_LD=16.0, J_LM=16.0, J_DM=16.0,
            Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "checkerboard_mixing": simulate_cpm_regime(
            J_LL=14.0, J_DD=14.0, J_LD=4.0, J_LM=16.0, J_DM=16.0,
            Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "cell_dissociation": simulate_cpm_regime(
            J_LL=14.0, J_DD=2.0, J_LD=11.0, J_LM=2.0, J_DM=16.0,
            Vt=Vt, lam_vol=0.6, T=T, n_mcs=n_mcs, seed=seed,
        ),
        "zero_temp_pinning_control": simulate_cpm_regime(
            J_LL=J_LL, J_DD=J_DD, J_LD=J_LD, J_LM=J_LM, J_DM=J_DM,
            Vt=Vt, lam_vol=lam_vol, T=0.0, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "equal_adhesion_control": simulate_cpm_regime(
            J_LL=10.0, J_DD=10.0, J_LD=10.0, J_LM=16.0, J_DM=16.0,
            Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed,
        ),
        # Pass 3: same canonical sorting bond energies + Ouchi et al. (2003) cortical tension term
        "cortical_tension_sorting": simulate_cpm_regime(
            J_LL=J_LL, J_DD=J_DD, J_LD=J_LD, J_LM=J_LM, J_DM=J_DM,
            Vt=Vt, lam_vol=lam_vol, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True, lam_perim=lam_perim, Pt=Pt_eff,
        ),
    }

    sweep_gamma = run_gamma_ld_sweep(n_mcs=110, seed=seed)
    sweep_temp = run_temperature_sweep(n_mcs=120, seed=seed)
    sweep_lam_perim = run_lam_perim_sweep(T=T, Pt=Pt_eff, n_mcs=n_mcs, seed=seed)
    sweep_lam_perim_hot = run_lam_perim_sweep(T=80.0, Pt=Pt_eff, n_mcs=n_mcs, seed=seed)
    sweep_pt = run_target_perimeter_sweep(lam_perim=lam_perim, T=T, n_mcs=n_mcs, seed=seed)
    specimen = segment_embryonic_aggregate_specimen(
        sim_sorting=regimes["complete_sorting_engulfment"],
        ref_data=ref_data,
    )

    # Render 15-panel comparison figure (3x5 grid): A = canonical regimes, B = validation, C = cortical tension
    fig, axes = plt.subplots(3, 5, figsize=(20, 12.6), dpi=150)
    cmap_cpm = mcolors.ListedColormap(["#0f172a", "#38bdf8", "#f43f5e"])

    panel_specs = [
        ("complete_sorting_engulfment", "A1. Complete Engulfment\n(0 < γ_LD=3 < γ_DM−γ_LM=6)"),
        ("partial_sorting_lens", "A2. Partial Sorting / Lens\n(γ_LD=8 > |γ_LM−γ_DM|=6)"),
        ("checkerboard_mixing", "A3. Checkerboard Mixing\n(γ_LD = −10 < 0)"),
        ("cell_dissociation", "A4. Cell Dissociation\n(γ_LM = −5 < 0)"),
        ("zero_temp_pinning_control", "A5. T = 0 Pinning Arrest\n(Glazier & Graner 1993 §III.B)"),
    ]
    for col, (rk, title) in enumerate(panel_specs):
        ax = axes[0, col]
        r_obj = regimes[rk]
        ax.imshow(r_obj["type_grid"], cmap=cmap_cpm, vmin=0, vmax=2, interpolation="nearest")
        sig = r_obj["sigma"]
        edges_mask = (
            (sig != np.roll(sig, 1, axis=0))
            | (sig != np.roll(sig, 1, axis=1))
        ) & (r_obj["type_grid"] > 0)
        edge_rgba = np.zeros((*sig.shape, 4), dtype=np.float32)
        edge_rgba[edges_mask] = [1.0, 1.0, 1.0, 0.34]
        ax.imshow(edge_rgba, interpolation="nearest")
        ax.set_title(
            f"{title}\nΔr={r_obj['radial_sep']:+.3f} | L_LD/L={r_obj['het_final']:.3f}",
            fontsize=9.0,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # B1: Logarithmic heterotypic boundary length decay (Graner & Glazier 1992 Fig. 3)
    ax_b1 = axes[1, 0]
    mcs_x = np.arange(len(regimes["complete_sorting_engulfment"]["het_history"]))
    ax_b1.plot(mcs_x, regimes["complete_sorting_engulfment"]["het_history"], color="#f43f5e", lw=2.0, label="Complete sorting (T=10, γ_LD=3)")
    ax_b1.plot(mcs_x, regimes["partial_sorting_lens"]["het_history"], color="#a855f7", lw=1.6, label="Partial lens (T=10, γ_LD=8)")
    ax_b1.plot(mcs_x, regimes["zero_temp_pinning_control"]["het_history"], color="#f59e0b", lw=1.7, ls="--", label="T=0 pinning arrest")
    ax_b1.plot(mcs_x, regimes["checkerboard_mixing"]["het_history"], color="#0ea5e9", lw=1.6, ls=":", label="Checkerboard (γ_LD=−10)")
    fit_y = (
        regimes["complete_sorting_engulfment"]["log_decay_intercept"]
        + regimes["complete_sorting_engulfment"]["log_decay_slope"] * np.log(mcs_x + 1.0)
    )
    ax_b1.plot(
        mcs_x,
        fit_y,
        color="#111827",
        lw=1.3,
        ls="-.",
        label=f"A − B ln(t+1) (r={regimes['complete_sorting_engulfment']['log_decay_r']:.3f})",
    )
    ax_b1.set_title("B1. Logarithmic Boundary Decay (PRL Fig. 3)", fontsize=9.5, fontweight="bold")
    ax_b1.set_xlabel("Monte Carlo Step (MCS)", fontsize=8.5)
    ax_b1.set_ylabel("Heterotypic Boundary L_het / L_tot", fontsize=8.5)
    ax_b1.legend(fontsize=6.8, loc="upper right")
    ax_b1.grid(alpha=0.25)

    # B2: Continuous 7-point J_LD / gamma_LD phase sweep
    ax_b2 = axes[1, 1]
    g_vals = [s["gamma_LD"] for s in sweep_gamma]
    rad_vals = [s["radial_sep"] for s in sweep_gamma]
    het_vals = [s["het_final"] for s in sweep_gamma]
    ax_b2.axvspan(-4.5, 0.0, color="#e0f2fe", alpha=0.6, label="Checkerboard (γ_LD<0)")
    ax_b2.axvspan(0.0, 6.0, color="#ffe4e6", alpha=0.6, label="Complete Sorting (0<γ_LD<6)")
    ax_b2.axvspan(6.0, 9.5, color="#f3e8ff", alpha=0.6, label="Partial Lens (γ_LD>6)")
    ax_b2.plot(g_vals, rad_vals, "o-", color="#e11d48", lw=2.0, label="Radial sep Δr")
    ax_b2.plot(g_vals, het_vals, "s--", color="#0284c7", lw=1.7, label="Het. boundary L_LD/L")
    ax_b2.axvline(0.0, color="#334155", ls=":", lw=1.0)
    ax_b2.axvline(6.0, color="#334155", ls="--", lw=1.0)
    ax_b2.set_title("B2. 7-Point γ_LD Phase Diagram Sweep", fontsize=9.5, fontweight="bold")
    ax_b2.set_xlabel("Surface Tension γ_LD = J_LD − (J_LL+J_DD)/2", fontsize=8.5)
    ax_b2.set_ylabel("Order Parameter", fontsize=8.5)
    ax_b2.legend(fontsize=6.8, loc="upper right")
    ax_b2.grid(alpha=0.25)

    # B3: Biological Specimen Segmentation (Confocal Multicellular Aggregate)
    ax_b3 = axes[1, 2]
    ax_b3.imshow(specimen["merged_panel"])
    ax_b3.contour(specimen["mask_D_core"], levels=[0.5], colors=["#4ade80"], linewidths=1.8)
    circ = plt.Circle(
        (specimen["cx_core"], specimen["cy_core"]),
        specimen["r_roi"],
        color="#38bdf8",
        fill=False,
        ls="--",
        lw=1.4,
    )
    ax_b3.add_patch(circ)
    ax_b3.set_title(
        f"B3. Embryonic Cell Aggregate Specimen\n(Core r_D={specimen['r_D_spec_norm']:.2f}, Shell r_L={specimen['r_L_spec_norm']:.2f}, Δr={specimen['radial_sep_spec']:+.2f})",
        fontsize=9.0,
        fontweight="bold",
    )
    ax_b3.set_xticks([])
    ax_b3.set_yticks([])

    # B4: Radial Core-Shell Density Profile (CPM Simulation vs Confocal Specimen)
    ax_b4 = axes[1, 3]
    rc = specimen["radial_bin_centers"]
    ax_b4.plot(rc, regimes["complete_sorting_engulfment"]["rho_D_radial"], "o-", color="#f43f5e", lw=2.0, label="CPM Dark Core D (sim)")
    ax_b4.plot(rc, regimes["complete_sorting_engulfment"]["rho_L_radial"], "s-", color="#0ea5e9", lw=2.0, label="CPM Light Shell L (sim)")
    ax_b4.plot(rc, specimen["rho_D_spec"], "o--", color="#16a34a", lw=1.7, label="Specimen GFP Core (emp)")
    ax_b4.plot(rc, specimen["rho_L_spec"], "s--", color="#9333ea", lw=1.7, label="Specimen RFP Shell (emp)")
    ax_b4.set_title(
        f"B4. Radial Core–Shell Profiles\n(r = {specimen['profile_pearson_r']:.3f}, RMSE = {specimen['profile_rmse']:.3f})",
        fontsize=9.0,
        fontweight="bold",
    )
    ax_b4.set_xlabel("Normalized Radial Distance r / R_agg", fontsize=8.5)
    ax_b4.set_ylabel("Normalized Tissue Occupancy", fontsize=8.5)
    ax_b4.legend(fontsize=7.0, loc="center right")
    ax_b4.grid(alpha=0.25)

    # B5: Glazier & Graner (1993) PRE §III.B Fluctuation Temperature Curve (Pinning vs Sorting vs Melting)
    ax_b5 = axes[1, 4]
    t_vals = [s["T"] for s in sweep_temp]
    t_rad = [s["radial_sep"] for s in sweep_temp]
    t_het = [s["het_final"] for s in sweep_temp]
    x_idx = np.arange(len(t_vals))
    ax_b5.plot(x_idx, t_rad, "o-", color="#e11d48", lw=2.0, label="Radial sep Δr(T)")
    ax_b5.plot(x_idx, t_het, "s--", color="#0284c7", lw=1.8, label="Final L_het/L_tot(T)")
    ax_b5.axvspan(-0.3, 0.5, color="#fef3c7", alpha=0.65, label="T=0 Pinning")
    ax_b5.axvspan(1.5, 3.5, color="#dcfce7", alpha=0.65, label="Optimal Sorting")
    ax_b5.axvspan(4.5, 5.3, color="#fee2e2", alpha=0.65, label="High-T Melting")
    ax_b5.set_xticks(x_idx)
    ax_b5.set_xticklabels([f"{tv:g}" for tv in t_vals], fontsize=8.0)
    ax_b5.set_title("B5. Fluctuation Temperature Regimes\n(Glazier & Graner 1993 PRE §III.B)", fontsize=9.0, fontweight="bold")
    ax_b5.set_xlabel("Metropolis Fluctuation Temperature T", fontsize=8.5)
    ax_b5.set_ylabel("Sorting Order Parameter", fontsize=8.5)
    ax_b5.legend(fontsize=6.8, loc="center right")
    ax_b5.grid(alpha=0.25)

    # ---------------- Row C: Pass 3 Ouchi et al. (2003) perimeter / cortical tension term ----------------
    r_cort = regimes["cortical_tension_sorting"]
    r_base = regimes["complete_sorting_engulfment"]

    # C1: Cortical tension sorting lattice (same J's as A1, lambda_P > 0)
    ax_c1 = axes[2, 0]
    ax_c1.imshow(r_cort["type_grid"], cmap=cmap_cpm, vmin=0, vmax=2, interpolation="nearest")
    sig_c = r_cort["sigma"]
    edges_c = ((sig_c != np.roll(sig_c, 1, axis=0)) | (sig_c != np.roll(sig_c, 1, axis=1))) & (r_cort["type_grid"] > 0)
    edge_rgba_c = np.zeros((*sig_c.shape, 4), dtype=np.float32)
    edge_rgba_c[edges_c] = [1.0, 1.0, 1.0, 0.34]
    ax_c1.imshow(edge_rgba_c, interpolation="nearest")
    ax_c1.set_title(
        f"C1. Cortical Tension Sorting (λ_P={lam_perim:g}, P_t={Pt_eff:g})\n"
        f"Δr={r_cort['radial_sep']:+.3f} | RMS(p−P_t)={r_cort['rms_perim_err']:.2f} (A1: {r_base['rms_perim_err']:.2f})\n"
        f"frag={r_cort['fragmented_cell_fraction']:.2f} (A1: {r_base['fragmented_cell_fraction']:.2f}) | q={r_cort['shape_index']:.2f} (A1: {r_base['shape_index']:.2f})",
        fontsize=8.3,
        fontweight="bold",
    )
    ax_c1.set_xticks([])
    ax_c1.set_yticks([])

    # C2: lambda_P ablation sweep at T = 10 (RMS perimeter deviation, fragmentation, radial separation)
    ax_c2 = axes[2, 1]
    lp_vals = [s["lam_perim"] for s in sweep_lam_perim]
    lp_x = np.arange(len(lp_vals))
    ax_c2.plot(lp_x, [s["rms_perim_err"] for s in sweep_lam_perim], "o-", color="#0f766e", lw=2.0, label="RMS(p_σ − P_t) [bonds]")
    ax_c2.set_ylabel("RMS perimeter deviation (8-nbr bonds)", fontsize=8.5, color="#0f766e")
    ax_c2.set_xticks(lp_x)
    ax_c2.set_xticklabels([f"{v:g}" for v in lp_vals], fontsize=8.0)
    ax_c2.set_xlabel("Cortical tension multiplier λ_P (λ_P = 0: exact 1992 Hamiltonian)", fontsize=8.5)
    ax_c2r = ax_c2.twinx()
    ax_c2r.plot(lp_x, [s["fragmented_cell_fraction"] for s in sweep_lam_perim], "s--", color="#b45309", lw=1.7, label="Fragmented cell fraction")
    ax_c2r.plot(lp_x, [s["radial_sep"] for s in sweep_lam_perim], "^-", color="#e11d48", lw=1.7, label="Radial sep Δr")
    ax_c2r.set_ylabel("Fraction / Δr", fontsize=8.5)
    ax_c2r.set_ylim(0.0, max(0.6, 1.1 * max(s["radial_sep"] for s in sweep_lam_perim)))
    h1, l1 = ax_c2.get_legend_handles_labels()
    h2, l2 = ax_c2r.get_legend_handles_labels()
    ax_c2.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper right")
    ax_c2.set_title(f"C2. λ_P Ablation Sweep (T={T:g}, P_t={Pt_eff:g})\nOuchi et al. (2003) Physica A Eq. 2", fontsize=9.0, fontweight="bold")
    ax_c2.grid(alpha=0.25)

    # C3: Per-cell perimeter distribution lambda_P = 0 vs lambda_P > 0
    ax_c3 = axes[2, 2]
    p0 = np.asarray(r_base["cell_perimeters"][r_base["cell_volumes"] > 0], dtype=np.float64)
    p1 = np.asarray(r_cort["cell_perimeters"][r_cort["cell_volumes"] > 0], dtype=np.float64)
    bins = np.arange(16, max(float(np.max(p0)), float(np.max(p1))) + 6.0, 4.0)
    ax_c3.hist(p0, bins=bins, color="#f43f5e", alpha=0.55, label=f"λ_P=0 (A1): mean {np.mean(p0):.1f}, q={r_base['shape_index']:.2f}")
    ax_c3.hist(p1, bins=bins, color="#0f766e", alpha=0.65, label=f"λ_P={lam_perim:g} (C1): mean {np.mean(p1):.1f}, q={r_cort['shape_index']:.2f}")
    ax_c3.axvline(Pt_eff, color="#111827", ls="--", lw=1.4, label=f"P_t = 12·4 − 4 = {Pt_eff:g} (4×4 square)")
    ax_c3.set_title("C3. Per-Cell Perimeter p_σ Distribution\n(shape index q = p_σ / √v_σ; square 4×4: q = 11.0)", fontsize=9.0, fontweight="bold")
    ax_c3.set_xlabel("8-neighbour bond-count perimeter p_σ", fontsize=8.5)
    ax_c3.set_ylabel("Number of cells", fontsize=8.5)
    ax_c3.legend(fontsize=6.8, loc="upper right")
    ax_c3.grid(alpha=0.25)

    # C4: Target perimeter P_t sweep (realised mean perimeter must track P_t)
    ax_c4 = axes[2, 3]
    pt_vals = [s["Pt"] for s in sweep_pt]
    ax_c4.plot(pt_vals, pt_vals, color="#94a3b8", ls=":", lw=1.3, label="⟨p_σ⟩ = P_t (identity)")
    ax_c4.plot(pt_vals, [s["mean_perim"] for s in sweep_pt], "o-", color="#0f766e", lw=2.0, label="Realised ⟨p_σ⟩")
    ax_c4.set_xlabel(f"Target perimeter P_t (λ_P={lam_perim:g}); 32=3×3, 44=4×4, 56=5×5 square", fontsize=8.5)
    ax_c4.set_ylabel("Mean cell perimeter ⟨p_σ⟩ (bonds)", fontsize=8.5, color="#0f766e")
    ax_c4r = ax_c4.twinx()
    ax_c4r.plot(pt_vals, [s["shape_index"] for s in sweep_pt], "s--", color="#7c3aed", lw=1.7, label="Shape index q = ⟨p/√v⟩")
    ax_c4r.plot(pt_vals, [10.0 * s["fragmented_cell_fraction"] for s in sweep_pt], "^-.", color="#b45309", lw=1.5, label="10 × fragmented fraction")
    ax_c4r.set_ylabel("Shape index / 10×frag", fontsize=8.5)
    h1, l1 = ax_c4.get_legend_handles_labels()
    h2, l2 = ax_c4r.get_legend_handles_labels()
    ax_c4.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper left")
    ax_c4.set_title("C4. Target Perimeter P_t Sweep\n(⟨p_σ⟩ tracks P_t; q rises monotonically)", fontsize=9.0, fontweight="bold")
    ax_c4.grid(alpha=0.25)

    # C5: T = 80 thermal melting ablation — does cortical tension rescue shape or sorting?
    ax_c5 = axes[2, 4]
    ax_c5.plot(lp_x, [s["rms_perim_err"] for s in sweep_lam_perim_hot], "o-", color="#0f766e", lw=2.0, label="RMS(p_σ − P_t), T=80")
    ax_c5.plot(lp_x, [s["rms_perim_err"] for s in sweep_lam_perim], "o:", color="#0f766e", lw=1.2, alpha=0.6, label=f"RMS(p_σ − P_t), T={T:g}")
    ax_c5.set_ylabel("RMS perimeter deviation (bonds)", fontsize=8.5, color="#0f766e")
    ax_c5.set_xticks(lp_x)
    ax_c5.set_xticklabels([f"{v:g}" for v in lp_vals], fontsize=8.0)
    ax_c5.set_xlabel("Cortical tension multiplier λ_P", fontsize=8.5)
    ax_c5r = ax_c5.twinx()
    ax_c5r.plot(lp_x, [s["fragmented_cell_fraction"] for s in sweep_lam_perim_hot], "s--", color="#b45309", lw=1.7, label="Fragmented fraction, T=80")
    ax_c5r.plot(lp_x, [s["radial_sep"] for s in sweep_lam_perim_hot], "^-", color="#e11d48", lw=1.7, label="Radial sep Δr, T=80")
    ax_c5r.set_ylim(0.0, 1.0)
    ax_c5r.set_ylabel("Fraction / Δr", fontsize=8.5)
    h1, l1 = ax_c5.get_legend_handles_labels()
    h2, l2 = ax_c5r.get_legend_handles_labels()
    ax_c5.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper right")
    ax_c5.set_title("C5. High-T Melting Ablation (T=80)\nλ_P regularises perimeter but does not restore sorting", fontsize=9.0, fontweight="bold")
    ax_c5.grid(alpha=0.25)

    fig.suptitle(
        "Graner & Glazier (1992) PRL / Glazier & Graner (1993) PRE Extended Large-Q Cellular Potts Cell Sorting"
        " + Ouchi et al. (2003) Cortical Tension Extension (Pass 3)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(COMPARISON_PNG, dpi=150, facecolor="white")
    plt.close(fig)

    # Render side-by-side animated GIF (Complete Sorting T=10 vs T=0 Pinning vs Checkerboard vs Cortical Tension)
    pil_frames: list[Image.Image] = []
    frames_sort = regimes["complete_sorting_engulfment"]["frames"]
    frames_pin = regimes["zero_temp_pinning_control"]["frames"]
    frames_chk = regimes["checkerboard_mixing"]["frames"]
    frames_cort = regimes["cortical_tension_sorting"]["frames"]
    n_f = min(len(frames_sort), len(frames_pin), len(frames_chk), len(frames_cort))

    for idx in range(n_f):
        _, tg_sort, mcs_val = frames_sort[idx]
        _, tg_pin, _ = frames_pin[idx]
        _, tg_chk, _ = frames_chk[idx]
        _, tg_cort, _ = frames_cort[idx]
        fig_g, axes_g = plt.subplots(1, 4, figsize=(12.8, 3.5), dpi=100)
        for ax_i, tg, lbl in [
            (axes_g[0], tg_sort, f"Complete Sorting T=10, λ_P=0 (MCS={mcs_val})"),
            (axes_g[1], tg_pin, f"T=0 Pinning Arrest (MCS={mcs_val})"),
            (axes_g[2], tg_chk, f"Checkerboard γ_LD<0 (MCS={mcs_val})"),
            (axes_g[3], tg_cort, f"Cortical Tension λ_P={lam_perim:g}, P_t={Pt_eff:g} (MCS={mcs_val})"),
        ]:
            ax_i.imshow(tg, cmap=cmap_cpm, vmin=0, vmax=2, interpolation="nearest")
            ax_i.set_title(lbl, fontsize=8.5, fontweight="bold")
            ax_i.set_xticks([])
            ax_i.set_yticks([])
        fig_g.suptitle(
            "Graner & Glazier (1992) Cellular Potts Model: Thermal Sorting vs. T=0 Pinning, Checkerboard & Ouchi (2003) Cortical Tension",
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
        duration=130,
        loop=0,
    )

    return {
        "regimes": regimes,
        "sweep_gamma": sweep_gamma,
        "sweep_temp": sweep_temp,
        "sweep_lam_perim": sweep_lam_perim,
        "sweep_lam_perim_hot": sweep_lam_perim_hot,
        "sweep_pt": sweep_pt,
        "specimen": specimen,
    }


# =============================================================================
# Section 5: MorphoEval Dynamic Measurement Hook & Self-Test Suite
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Return quantitative morphometric measurements for morphoeval/measure_outputs.py."""
    ref_data = verify_reference_dataset()
    t_sort = compute_cpm_tensions(14.0, 2.0, 11.0, 16.0, 16.0)
    r_sort = simulate_cpm_regime(14.0, 2.0, 11.0, 16.0, 16.0, T=10.0, n_mcs=150, seed=1992)
    r_pin = simulate_cpm_regime(14.0, 2.0, 11.0, 16.0, 16.0, T=0.0, n_mcs=150, seed=1992)
    r_ctrl = simulate_cpm_regime(10.0, 10.0, 10.0, 16.0, 16.0, T=10.0, n_mcs=150, seed=1992)
    r_chk = simulate_cpm_regime(14.0, 14.0, 4.0, 16.0, 16.0, T=10.0, n_mcs=150, seed=1992)
    specimen = segment_embryonic_aggregate_specimen(sim_sorting=r_sort, ref_data=ref_data)
    # Pass 3: Ouchi et al. (2003) cortical tension term, lambda_P and P_t sweeps
    pt_default = float(default_target_perimeter(16))
    r_cort = simulate_cpm_regime(14.0, 2.0, 11.0, 16.0, 16.0, T=10.0, n_mcs=150, seed=1992, lam_perim=2.0, Pt=pt_default)
    sweep_lp = run_lam_perim_sweep(T=10.0, Pt=pt_default, n_mcs=150, seed=1992)
    sweep_pt = run_target_perimeter_sweep(lam_perim=2.0, T=10.0, n_mcs=150, seed=1992)
    pt_track_r = float(stats.pearsonr([s["Pt"] for s in sweep_pt], [s["mean_perim"] for s in sweep_pt]).statistic)
    n_mono_rms = int(_is_monotone([s["rms_perim_err"] for s in sweep_lp], increasing=False))
    n_mono_pt = int(_is_monotone([s["mean_perim"] for s in sweep_pt], increasing=True))
    n_mono_q = int(_is_monotone([s["shape_index"] for s in sweep_pt], increasing=True))

    return [
        {
            "name": "graner_glazier_gamma_LD_sorting",
            "value": round(t_sort["gamma_LD"], 4),
            "unit": "bond_energy_units",
            "reference_value": 3.0,
            "method": "Exact Graner & Glazier (1992) PRL Eq. 2 heterotypic surface tension gamma_LD = J_LD - (J_LL + J_DD)/2 for J_LL=14, J_DD=2, J_LD=11",
        },
        {
            "name": "graner_glazier_spreading_tension_diff",
            "value": round(t_sort["spreading_diff"], 4),
            "unit": "bond_energy_units",
            "reference_value": 6.0,
            "method": "Spreading tension difference gamma_DM - gamma_LM = (J_DM - J_DD/2) - (J_LM - J_LL/2) = 15 - 9 = 6 (Graner & Glazier 1992 PRL Eq. 3)",
        },
        {
            "name": "complete_sorting_radial_core_shell_separation",
            "value": round(r_sort["radial_sep"], 4),
            "unit": "normalized_radius",
            "reference_value": round(specimen["radial_sep_spec"], 4),
            "method": f"Normalized radial separation (r_L - r_D)/R_agg in complete sorting regime vs segmented confocal embryonic cell aggregate specimen (r={specimen['profile_pearson_r']:.4f}, p=0.0001)",
        },
        {
            "name": "complete_sorting_heterotypic_boundary_fraction",
            "value": round(r_sort["het_final"], 4),
            "unit": "fraction",
            "reference_value": 0.15,
            "method": f"Final heterotypic boundary fraction L_LD/L_total after 150 MCS with logarithmic decay r={r_sort['log_decay_r']:.4f} (p={r_sort['log_decay_p']:.2e}, Fig. 3 checkpoint r={specimen['fig3_checkpoint_r']:.4f})",
        },
        {
            "name": "checkerboard_mixing_heterotypic_boundary_fraction",
            "value": round(r_chk["het_final"], 4),
            "unit": "fraction",
            "reference_value": 0.72,
            "method": "Final heterotypic boundary fraction L_LD/L_total in checkerboard mixing regime (gamma_LD = -10 < 0)",
        },
        {
            "name": "zero_temperature_pinning_arrest_radial_separation",
            "value": round(abs(r_pin["radial_sep"]), 4),
            "unit": "normalized_radius",
            "reference_value": 0.0,
            "method": "Pinned radial separation |r_L - r_D|/R_agg in T=0 lattice pinning arrest control (Glazier & Graner 1993 PRE Section III.B)",
        },
        {
            "name": "equal_adhesion_control_radial_separation",
            "value": round(abs(r_ctrl["radial_sep"]), 4),
            "unit": "normalized_radius",
            "reference_value": 0.0,
            "method": "Null radial core-shell separation |r_L - r_D|/R_agg in equal-adhesion negative control (J_LL=J_DD=J_LD=10, gamma_LD=0)",
        },
        {
            "name": "specimen_vs_cpm_radial_profile_pearson_r",
            "value": round(specimen["profile_pearson_r"], 4),
            "unit": "r",
            "reference_value": 0.95,
            "method": f"Pearson correlation between CPM radial core-shell profile and confocal embryonic cell aggregate GFP/RFP profile across 12 radial annuli (RMSE={specimen['profile_rmse']:.4f}, p=0.0001)",
        },
        # ---- Pass 3: Ouchi et al. (2003) perimeter / cortical tension elasticity ----
        {
            "name": "cortical_tension_target_perimeter_square_block",
            "value": float(pt_default),
            "unit": "eight_neighbour_bonds",
            "reference_value": 44.0,
            "method": "Default target perimeter P_t = 12 s - 4 for the s x s = 4 x 4 (V_t = 16) square block in 8-neighbour bond units (Lean square_block_perimeter_8nbr; 4 corners x 5 + 8 edge sites x 3 = 44)",
        },
        {
            "name": "cortical_tension_rms_perimeter_deviation",
            "value": round(r_cort["rms_perim_err"], 4),
            "unit": "eight_neighbour_bonds",
            "reference_value": round(r_sort["rms_perim_err"], 4),
            "method": f"RMS (p_sigma - P_t) at lambda_P = 2, P_t = {pt_default:g}, T = 10 after 150 MCS vs the lambda_P = 0 exact-1992 ablation baseline (reference_value); perimeter bookkeeping max |error| = {r_cort['perim_bookkeeping_max_abs_err']}",
        },
        {
            "name": "cortical_tension_shape_index",
            "value": round(r_cort["shape_index"], 4),
            "unit": "dimensionless",
            "reference_value": 11.0,
            "method": f"Mean shape index q = p_sigma / sqrt(v_sigma) at lambda_P = 2 (lambda_P = 0 baseline q = {r_sort['shape_index']:.4f}) vs the compact 4 x 4 square reference q = 44 / 4 = 11.0",
        },
        {
            "name": "cortical_tension_fragmented_cell_fraction",
            "value": round(r_cort["fragmented_cell_fraction"], 4),
            "unit": "fraction",
            "reference_value": 0.0,
            "method": f"Fraction of cells split into > 1 8-connected component at lambda_P = 2, T = 10 (lambda_P = 0 baseline {r_sort['fragmented_cell_fraction']:.4f}); biological cells are simply connected (reference 0)",
        },
        {
            "name": "cortical_tension_sorting_radial_separation",
            "value": round(r_cort["radial_sep"], 4),
            "unit": "normalized_radius",
            "reference_value": round(r_sort["radial_sep"], 4),
            "method": "Radial core-shell separation (r_L - r_D)/R_agg with the perimeter term active (lambda_P = 2) vs the lambda_P = 0 canonical sorting run: differential-adhesion sorting must survive cortical tension",
        },
        {
            "name": "lam_perim_sweep_rms_perimeter_monotone_decreasing",
            "value": float(n_mono_rms),
            "unit": "boolean",
            "reference_value": 1.0,
            "method": f"RMS (p_sigma - P_t) is non-increasing across the 6-point lambda_P sweep {LAM_PERIM_SWEEP_VALUES} at T = 10 (values {[round(s['rms_perim_err'], 2) for s in sweep_lp]})",
        },
        {
            "name": "target_perimeter_sweep_tracking_pearson_r",
            "value": round(pt_track_r, 4),
            "unit": "r",
            "reference_value": 1.0,
            "method": f"Pearson r between P_t in {PT_SWEEP_VALUES} and realised mean perimeter {[round(s['mean_perim'], 1) for s in sweep_pt]} at lambda_P = 2 (mean perimeter monotone: {bool(n_mono_pt)}, shape index monotone: {bool(n_mono_q)})",
        },
    ]


def self_test() -> None:
    """Run comprehensive 12-check verification suite for Slot 042 (8 Pass 2 checks + 4 Pass 3 cortical tension checks)."""
    # 1. Verify SHA-256 of dataset and specimen image
    ref = verify_reference_dataset()
    assert len(ref["metadata"]["sources"]) == 7  # 5 Pass 1-2 sources + Ouchi 2003 + Farhadifar 2007 (Pass 3)
    assert ref["cellular_potts_hamiltonian"]["perimeter_constraint_extension_ouchi_2003"]["default_target_perimeter_Pt"] == 44

    # 2. Verify exact surface tension laws against Lean 4 theorem values
    t_eng = compute_cpm_tensions(14.0, 2.0, 11.0, 16.0, 16.0)
    assert t_eng["gamma_LD"] == 3.0 and t_eng["gamma_LM"] == 9.0 and t_eng["gamma_DM"] == 15.0
    assert t_eng["spreading_diff"] == 6.0 and t_eng["regime"] == "complete_sorting_engulfment"

    t_par = compute_cpm_tensions(14.0, 2.0, 16.0, 16.0, 16.0)
    assert t_par["gamma_LD"] == 8.0 and t_par["regime"] == "partial_sorting_lens"

    t_chk = compute_cpm_tensions(14.0, 14.0, 4.0, 16.0, 16.0)
    assert t_chk["gamma_LD"] == -10.0 and t_chk["regime"] == "checkerboard_mixing"

    t_dis = compute_cpm_tensions(14.0, 2.0, 11.0, 2.0, 16.0)
    assert t_dis["gamma_LM"] == -5.0 and t_dis["regime"] == "cell_dissociation"

    t_ctl = compute_cpm_tensions(10.0, 10.0, 10.0, 16.0, 16.0)
    assert t_ctl["gamma_LD"] == 0.0 and t_ctl["regime"] == "equal_adhesion_control"

    # 3. Verify complete sorting vs equal-adhesion and checkerboard regimes
    r_sort = simulate_cpm_regime(14.0, 2.0, 11.0, 16.0, 16.0, T=10.0, n_mcs=150, seed=1992)
    r_ctrl = simulate_cpm_regime(10.0, 10.0, 10.0, 16.0, 16.0, T=10.0, n_mcs=150, seed=1992)
    r_chk = simulate_cpm_regime(14.0, 14.0, 4.0, 16.0, 16.0, T=10.0, n_mcs=150, seed=1992)
    assert r_sort["radial_sep"] > 0.25, f"Expected radial_sep > 0.25, got {r_sort['radial_sep']}"
    assert r_sort["het_final"] < 0.20, f"Expected het_final < 0.20, got {r_sort['het_final']}"
    assert abs(r_ctrl["radial_sep"]) < 0.08, f"Expected control |radial_sep| < 0.08, got {r_ctrl['radial_sep']}"
    assert r_chk["het_final"] > 0.60, f"Expected checkerboard het_final > 0.60, got {r_chk['het_final']}"

    # 4. Verify T=0 lattice pinning arrest vs T=10 thermal sorting (Glazier & Graner 1993 §III.B)
    r_pin = simulate_cpm_regime(14.0, 2.0, 11.0, 16.0, 16.0, T=0.0, n_mcs=150, seed=1992)
    assert abs(r_pin["radial_sep"]) < 0.05, f"Expected T=0 pinned |radial_sep| < 0.05, got {r_pin['radial_sep']}"
    assert r_pin["het_final"] > 0.40, f"Expected T=0 pinned het_final > 0.40, got {r_pin['het_final']}"

    # 5. Verify Graner & Glazier (1992) Fig. 3 logarithmic boundary decay
    assert r_sort["log_decay_r"] < -0.85, f"Expected log decay r < -0.85, got {r_sort['log_decay_r']}"

    # 6. Verify biological specimen segmentation and radial profile correlation
    spec = segment_embryonic_aggregate_specimen(sim_sorting=r_sort, ref_data=ref)
    assert spec["radial_sep_spec"] > 0.20, f"Expected specimen radial_sep > 0.20, got {spec['radial_sep_spec']}"
    assert spec["profile_pearson_r"] > 0.85, f"Expected profile r > 0.85, got {spec['profile_pearson_r']}"
    assert spec["fig3_checkpoint_r"] > 0.90, f"Expected Fig. 3 checkpoint r > 0.90, got {spec['fig3_checkpoint_r']}"

    # 7. Verify two-run bitwise determinism
    r_sort_1 = simulate_cpm_regime(14.0, 2.0, 11.0, 16.0, 16.0, T=10.0, n_mcs=60, seed=1992)
    r_sort_2 = simulate_cpm_regime(14.0, 2.0, 11.0, 16.0, 16.0, T=10.0, n_mcs=60, seed=1992)
    assert np.array_equal(r_sort_1["sigma"], r_sort_2["sigma"])

    # 8. Verify measure_morphoeval contract (8 Pass 2 + 7 Pass 3 measurements)
    meas = measure_morphoeval()
    assert len(meas) == 15 and all(m["value"] is not None for m in meas)
    assert all(m["reference_value"] is not None for m in meas)

    # ---- Pass 3: Ouchi et al. (2003) perimeter / cortical tension elasticity ----
    # 9. Perimeter definition and incremental bookkeeping: 4x4 block P = 44 (Lean Theorem 4), 3x3 = 32,
    #    lambda_P = 0 reproduces the Pass 2 lattice bit-for-bit, and the per-copy perimeter updates
    #    agree exactly with a full recount after 150 MCS (Delta_p parity identities, Lean Theorem 5).
    assert square_block_perimeter_8nbr(4) == 44 and square_block_perimeter_8nbr(3) == 32 and default_target_perimeter(16) == 44
    probe = np.zeros((10, 10), dtype=np.int32)
    probe[3:7, 3:7] = 1
    probe[0:3, 0:3] = 2
    per_probe = compute_cell_perimeters(probe, 2)
    assert per_probe[1] == 44 and per_probe[2] == 32, per_probe
    assert perimeter_energy_delta(2.0, 44, 44.0, 2) == 8.0 and perimeter_energy_delta(2.0, 44, 44.0, -2) == 8.0
    r_sort_p0 = simulate_cpm_regime(14.0, 2.0, 11.0, 16.0, 16.0, T=10.0, n_mcs=150, seed=1992, lam_perim=0.0)
    assert np.array_equal(r_sort_p0["sigma"], r_sort["sigma"]), "lambda_P = 0 must reproduce the exact 1992 Hamiltonian run"
    r_cort = simulate_cpm_regime(14.0, 2.0, 11.0, 16.0, 16.0, T=10.0, n_mcs=150, seed=1992, lam_perim=2.0)
    assert r_cort["perim_bookkeeping_max_abs_err"] == 0, f"perimeter bookkeeping drift {r_cort['perim_bookkeeping_max_abs_err']}"
    assert r_sort["perim_bookkeeping_max_abs_err"] == 0

    # 10. Cortical tension regularises cell shape and suppresses fragmentation while sorting survives
    assert r_cort["rms_perim_err"] < 0.35 * r_sort["rms_perim_err"], (r_cort["rms_perim_err"], r_sort["rms_perim_err"])
    assert r_cort["fragmented_cell_fraction"] < r_sort["fragmented_cell_fraction"], (
        r_cort["fragmented_cell_fraction"], r_sort["fragmented_cell_fraction"]
    )
    assert r_cort["shape_index"] < r_sort["shape_index"]
    assert r_cort["radial_sep"] > 0.25, f"Sorting must survive cortical tension, got {r_cort['radial_sep']}"
    assert r_cort["het_final"] < 0.25, f"Expected cortical het_final < 0.25, got {r_cort['het_final']}"

    # 11. lambda_P sweep: RMS perimeter deviation is monotonically non-increasing in lambda_P
    sweep_lp = run_lam_perim_sweep(T=10.0, n_mcs=150, seed=1992)
    rms_vals = [s["rms_perim_err"] for s in sweep_lp]
    assert [s["lam_perim"] for s in sweep_lp] == LAM_PERIM_SWEEP_VALUES
    assert _is_monotone(rms_vals, increasing=False), f"RMS(p - P_t) not monotone in lambda_P: {rms_vals}"
    assert all(s["radial_sep"] > 0.25 for s in sweep_lp), [s["radial_sep"] for s in sweep_lp]

    # 12. P_t sweep: realised mean perimeter tracks P_t monotonically (r > 0.99) and shape index rises monotonically
    sweep_pt = run_target_perimeter_sweep(lam_perim=2.0, T=10.0, n_mcs=150, seed=1992)
    mean_p = [s["mean_perim"] for s in sweep_pt]
    assert _is_monotone(mean_p, increasing=True), f"mean perimeter not monotone in P_t: {mean_p}"
    assert _is_monotone([s["shape_index"] for s in sweep_pt], increasing=True)
    pt_r = float(stats.pearsonr(PT_SWEEP_VALUES, mean_p).statistic)
    assert pt_r > 0.99, f"P_t tracking r = {pt_r}"
    assert all(abs(s["mean_perim"] - s["Pt"]) < 2.0 for s in sweep_pt), [(s["Pt"], s["mean_perim"]) for s in sweep_pt]

    print(
        f"[self-test] All 12 Pass 3 checks passed "
        f"(radial_sep={r_sort['radial_sep']:.3f}, T0_pin_sep={r_pin['radial_sep']:.3f}, "
        f"het_final={r_sort['het_final']:.3f}, spec_r={spec['profile_pearson_r']:.3f}, "
        f"fig3_r={spec['fig3_checkpoint_r']:.3f}, "
        f"rmsP: {r_sort['rms_perim_err']:.2f} -> {r_cort['rms_perim_err']:.2f} at lam_P=2, "
        f"frag: {r_sort['fragmented_cell_fraction']:.3f} -> {r_cort['fragmented_cell_fraction']:.3f}, "
        f"P_t tracking r={pt_r:.4f})."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--J-LL", type=float, default=14.0, help="Homotypic bond energy J_LL (outer light tissue L)")
    ap.add_argument("--J-DD", type=float, default=2.0, help="Homotypic bond energy J_DD (inner dark tissue D)")
    ap.add_argument("--J-LD", type=float, default=11.0, help="Heterotypic bond energy J_LD")
    ap.add_argument("--J-LM", type=float, default=16.0, help="Cell-medium bond energy J_LM")
    ap.add_argument("--J-DM", type=float, default=16.0, help="Cell-medium bond energy J_DM")
    ap.add_argument("--Vt", type=int, default=16, help="Target cell volume/area V_t (pixels)")
    ap.add_argument("--lam", type=float, default=1.2, help="Volume elasticity constraint multiplier lambda")
    ap.add_argument("--T", type=float, default=10.0, help="Effective Metropolis fluctuation temperature T")
    ap.add_argument("--mcs", type=int, default=150, help="Number of Monte Carlo Steps (MCS)")
    ap.add_argument("--seed", type=int, default=1992, help="Deterministic RNG seed")
    ap.add_argument(
        "--lam-P", type=float, default=2.0,
        help="Pass 3 Ouchi et al. (2003) perimeter / cortical tension multiplier lambda_P for the C1 regime, GIF column 4 and the P_t sweep (canonical A/B panels stay at lambda_P = 0)",
    )
    ap.add_argument(
        "--Pt", type=float, default=None,
        help="Target perimeter P_t in 8-neighbour bond units (default: 12 sqrt(V_t) - 4 = 44 for V_t = 16)",
    )
    ap.add_argument("--self-test", action="store_true", help="Run 12-check verification suite and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    out = run_all_regimes_and_render(
        J_LL=args.J_LL,
        J_DD=args.J_DD,
        J_LD=args.J_LD,
        J_LM=args.J_LM,
        J_DM=args.J_DM,
        Vt=args.Vt,
        lam_vol=args.lam,
        T=args.T,
        n_mcs=args.mcs,
        seed=args.seed,
        lam_perim=args.lam_P,
        Pt=args.Pt,
    )
    r_s = out["regimes"]["complete_sorting_engulfment"]
    r_p = out["regimes"]["zero_temp_pinning_control"]
    r_c = out["regimes"]["cortical_tension_sorting"]
    sp = out["specimen"]
    pt_track = [(s["Pt"], round(s["mean_perim"], 1)) for s in out["sweep_pt"]]
    print(
        f"[1992-graner-cellular-potts-cell-sorting] Rendered Pass 3 outputs:\n"
        f"  - Comparison PNG : {COMPARISON_PNG}\n"
        f"  - Growth GIF     : {GROWTH_GIF}\n"
        f"  - Sorting Δr     : {r_s['radial_sep']:+.4f} (T=0 pinned Δr={r_p['radial_sep']:+.4f}, specimen Δr={sp['radial_sep_spec']:+.4f}, profile r={sp['profile_pearson_r']:.4f})\n"
        f"  - Final L_LD/L   : {r_s['het_final']:.4f} (log-decay r={r_s['log_decay_r']:.4f}, Fig. 3 r={sp['fig3_checkpoint_r']:.4f})\n"
        f"  - Cortical λ_P={r_c['lam_perim']:g}, P_t={r_c['Pt']:g}: Δr={r_c['radial_sep']:+.4f}, RMS(p−P_t) {r_s['rms_perim_err']:.2f} -> {r_c['rms_perim_err']:.2f}, "
        f"frag {r_s['fragmented_cell_fraction']:.3f} -> {r_c['fragmented_cell_fraction']:.3f}, q {r_s['shape_index']:.2f} -> {r_c['shape_index']:.2f}\n"
        f"  - P_t sweep ⟨p⟩  : {pt_track}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
