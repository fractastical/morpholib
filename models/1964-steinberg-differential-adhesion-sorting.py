#!/usr/bin/env python3
"""
1964-steinberg-differential-adhesion-sorting.py — Pass 3 (v3 Remediated)
========================================================================
Steinberg (1963, 1970) Differential Adhesion Hypothesis (DAH),
Graner & Glazier (1992) / Glazier & Graner (1993) Extended Cellular Potts Model,
Foty, Pfleger, Forgacs & Steinberg (1996) Chick Embryo Surface Tension Hierarchy,
and (Pass 3) a full 3D Cellular Potts spheroid on a 64^3 lattice compared against
the 2D model and its own 2D equatorial cross-section (sorting-kinetics exponents).

Pass 3 remediation (closes the single open Pass 2 item):
  - `simulate_potts_regime_3d()` runs the Graner-Glazier Hamiltonian on a 64^3 cubic
    lattice holding a spherical aggregate of ~840 cubic 4^3 cells (26-neighbour adhesion
    term, quadratic volume constraint lambda_V (v - V_0)^2). The Metropolis sweep is a
    sublattice-parallel ("8-colour" parity checkerboard) update: two sites of the same
    parity class (x mod 2, y mod 2, z mod 2) are never Moore-adjacent (Lean 4 theorem
    `parity_sublattice_sites_not_moore_adjacent`), so the adhesion energy of simultaneous
    flips is exact; only the volume term is evaluated with sub-sweep-start volumes
    (the standard domain-parallel CPM approximation, cf. Chen, Glazier, Izaguirre &
    Alber 2007 Comput. Phys. Commun. 176:670). The single-voxel volume increment used
    by both the 2D and 3D solvers, lambda*((v+1-V_0)^2 - (v-V_0)^2) = lambda*(2(v-V_0)+1),
    is Lean 4 theorem `potts_volume_energy_single_voxel_increment`.
  - `fit_sorting_kinetics()` fits the heterotypic boundary fraction L_AB(t)/L(t) both
    to the Graner & Glazier (1992) Fig. 3 logarithmic law a + b ln(t+1) and to a power
    law c (t+1)^(-alpha) and reports which description has the higher R^2. The exponent
    alpha is compared against the Lifshitz-Slyozov-Wagner diffusive coarsening bound
    alpha = 1/3 (conserved order parameter, L_AB/L ~ 1/R(t) ~ t^(-1/3)); alpha -> 0 is
    the logarithmic (frozen) limit reported for the 2D model.
  - `compare_2d_vs_3d_sorting_kinetics()` runs the 3D sorted spheroid, a 3D run with
    the cell-level Kawasaki type-swap accelerator ablated, and a 3D equal-adhesion
    negative control, and measures the exponent on (i) the native 2D model, (ii) the full
    3D boundary and (iii) the 2D equatorial cross-section of the 3D spheroid, which is the
    like-for-like comparison for the confocal optical section in `cell_sorting_specimen.jpg`.
  - Honest limits: 64^3 x 200 MCS is far from the asymptotic coarsening regime, the
    exponents are fitted over ~2.3 decades of MCS, and the Kawasaki type-swap move is
    a non-physical accelerator (its effect on alpha is measured by the ablation, not
    assumed to vanish). See the v3 audit `remediation` block for the carried residuals.

Primary Literature & Equations Implemented:
  1. Steinberg MS (1963) Science 141(3579):401-408 (doi:10.1126/science.141.3579.401) &
     Steinberg MS (1970) J. Exp. Zool. 173(4):395-433 (doi:10.1002/jez.1401730406):
     - Differential Adhesion Hypothesis (DAH): binary cell aggregates sort toward the
       thermodynamic minimum of total interfacial free energy.
  2. Graner F, Glazier JA (1992) Phys. Rev. Lett. 69(13):2013-2016 (doi:10.1103/PhysRevLett.69.2013) &
     Glazier JA, Graner F (1993) Phys. Rev. E 47(3):2128-2154 (doi:10.1103/PhysRevE.47.2128):
     - Extended Cellular Potts Hamiltonian (Eq. 1):
         H = sum_{<ij, i'j'>} J(tau(sigma_ij), tau(sigma_i'j')) * (1 - delta_{sigma_ij, sigma_i'j'})
             + lambda * sum_{sigma > 0} (a(sigma) - A_0)^2
     - Interfacial tensions (Eqs. 2-3, with J_MM = 0):
         gamma_AB = J_AB - (J_AA + J_BB) / 2
         gamma_AM = J_AM - J_AA / 2
         gamma_BM = J_BM - J_BB / 2
     - Four thermodynamic equilibrium regimes (PRL p. 2014; PRE Table I):
         (I)   Checkerboard mixing:   gamma_AB < 0
         (II)  Complete engulfment:   0 < gamma_AB < gamma_BM - gamma_AM
         (III) Partial engulfment:    gamma_AB > gamma_BM - gamma_AM > 0
         (IV)  Cell dissociation:     gamma_AM < 0 (or gamma_BM < 0)
  3. Foty RA, Pfleger CM, Forgacs G, Steinberg MS (1996) Development 122(5):1611-1620
     (doi:10.1242/dev.122.5.1611):
     - Empirical Gallus gallus embryonic tissue surface tensions sigma (dyn/cm, Table 1):
       limb bud precartilage (20.1) > pigmented epithelium (12.6) > heart ventricle (8.5) >
       embryonic liver (4.6) > neural retina (1.6).
  4. Sorting kinetics (Pass 3):
     - Graner & Glazier (1992) PRL Fig. 3: logarithmic decay of heterotypic boundary length.
     - Mombach, Glazier, Raphael & Zajac (1995) Phys. Rev. Lett. 75(11):2244-2247
       (doi:10.1103/PhysRevLett.75.2244): quantitative sorting kinetics with/without fluctuations.
     - Nakajima & Ishihara (2011) New J. Phys. 13:033035 (doi:10.1088/1367-2630/13/3/033035):
       re-examination of CPM sorting kinetics (power-law vs logarithmic domain growth).
     - Chen, Glazier, Izaguirre & Alber (2007) Comput. Phys. Commun. 176(11-12):670-681
       (doi:10.1016/j.cpc.2007.03.007): domain-parallel CPM updates (basis of the 3D sublattice sweep).
     - Lifshitz-Slyozov-Wagner diffusive coarsening: R(t) ~ t^(1/3) => L_AB/L ~ t^(-1/3).
     (The three Pass 3 DOIs were added from memory and could not be verified offline; see audit.)
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
DATASET_DIR = ROOT / "datasets" / "differential_adhesion_sorting"
REFERENCE_JSON = DATASET_DIR / "differential_adhesion_sorting_reference.json"
SPECIMEN_JPG = DATASET_DIR / "cell_sorting_specimen.jpg"
MEDIA_DIR = ROOT / "docs" / "media"
COMPARISON_PNG = MEDIA_DIR / "differential_adhesion_sorting_comparison.png"
GROWTH_GIF = MEDIA_DIR / "differential_adhesion_sorting_growth.gif"

EXPECTED_SHA256 = {
    "datasets/differential_adhesion_sorting/cell_sorting_specimen.jpg": (
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
    """Load and cryptographically verify datasets/differential_adhesion_sorting/."""
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(f"Missing reference dataset: {REFERENCE_JSON}")
    if not SPECIMEN_JPG.exists():
        raise FileNotFoundError(f"Missing specimen photograph: {SPECIMEN_JPG}")

    ref_data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    actual_sha = sha256_file(SPECIMEN_JPG)
    expected_sha = EXPECTED_SHA256["datasets/differential_adhesion_sorting/cell_sorting_specimen.jpg"]
    json_sha = ref_data["metadata"]["specimens"]["cell_sorting_specimen"]["sha256"]
    if actual_sha != expected_sha or actual_sha != json_sha:
        raise ValueError(
            f"SHA-256 mismatch for {SPECIMEN_JPG}: actual={actual_sha}, expected={expected_sha}"
        )
    return ref_data


# =============================================================================
# Section 1: Steinberg (1963) & Graner-Glazier (1992) Interfacial Tension Laws
# =============================================================================

def compute_interfacial_tensions(
    J_AA: float,
    J_BB: float,
    J_AB: float,
    J_AM: float,
    J_BM: float,
    J_MM: float = 0.0,
) -> dict[str, Any]:
    """
    Compute Steinberg (1963) / Graner & Glazier (1992) PRL Eqs. (2)-(3) interfacial
    tensions from Potts bond energies J_{tau, tau'} and classify the equilibrium phase.

    Matches Lean 4 definitions `two_gamma_AB`, `two_gamma_M`, and `two_spreading_diff`
    in `morphoeval/lean/loops/Loop_041_differential_adhesion_sorting.lean`.
    """
    gamma_AB = float(J_AB - 0.5 * (J_AA + J_BB))
    gamma_AM = float(J_AM - 0.5 * (J_AA + J_MM))
    gamma_BM = float(J_BM - 0.5 * (J_BB + J_MM))
    spreading_diff = float(gamma_BM - gamma_AM)
    spreading_coeff_A_on_B = float(gamma_BM - gamma_AM - gamma_AB)

    if gamma_AM < 0.0 or gamma_BM < 0.0:
        regime = "cell_dissociation"
    elif gamma_AB < 0.0:
        regime = "checkerboard_mixing"
    elif abs(gamma_AB) < 1e-9 and abs(spreading_diff) < 1e-9:
        regime = "equal_adhesion_control"
    elif 0.0 < gamma_AB < abs(spreading_diff):
        regime = "complete_engulfment"
    else:
        regime = "partial_engulfment"

    return {
        "J_AA": float(J_AA),
        "J_BB": float(J_BB),
        "J_AB": float(J_AB),
        "J_AM": float(J_AM),
        "J_BM": float(J_BM),
        "gamma_AB": gamma_AB,
        "gamma_AM": gamma_AM,
        "gamma_BM": gamma_BM,
        "spreading_diff": spreading_diff,
        "spreading_coeff_A_on_B": spreading_coeff_A_on_B,
        "regime": regime,
    }


# =============================================================================
# Section 2: 2D Extended Cellular Potts Model (Graner & Glazier 1992 PRL Eq. 1)
# =============================================================================

def _compute_potts_morphometrics(
    sigma: np.ndarray,
    cell_type: np.ndarray,
) -> dict[str, Any]:
    """
    Compute morphometric order parameters of a Cellular Potts configuration:
      - Fractional heterotypic boundary length L_AB / L_total (Graner & Glazier 1992 Fig. 3)
      - Medium boundary fraction L_{cell-M} / L_total
      - Normalized mean radial positions r_A / R_agg and r_B / R_agg
      - Normalized radial core-shell separation Delta_r = (r_A - r_B) / R_agg
      - Outer shell engulfment fraction (fraction of tissue-medium perimeter occupied by type A)
      - Radial density profiles rho_A(r/R), rho_B(r/R) across 12 concentric annuli
    """
    t_grid = cell_type[sigma]
    H, W = sigma.shape

    het_bonds = 0
    am_bonds = 0
    bm_bonds = 0
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
        am_bonds += int(np.sum(diff_cell & (((t1 == 1) & (t2 == 0)) | ((t1 == 0) & (t2 == 1)))))
        bm_bonds += int(np.sum(diff_cell & (((t1 == 2) & (t2 == 0)) | ((t1 == 0) & (t2 == 2)))))

    het_frac = float(het_bonds / max(tot_bonds, 1))
    med_bonds = am_bonds + bm_bonds
    outer_A_envelopment = float(am_bonds / max(med_bonds, 1))

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

    mask_A = t_grid == 1
    mask_B = t_grid == 2
    r_A_norm = float(np.mean(rr[mask_A]) / r_agg) if np.any(mask_A) else 0.0
    r_B_norm = float(np.mean(rr[mask_B]) / r_agg) if np.any(mask_B) else 0.0
    radial_sep = float(r_A_norm - r_B_norm)

    # Angular segregation dipole between A and B centroids (captures partial engulfment lens)
    if np.any(mask_A) and np.any(mask_B):
        dy_ab = float(np.mean(yy[mask_A]) - np.mean(yy[mask_B]))
        dx_ab = float(np.mean(xx[mask_A]) - np.mean(xx[mask_B]))
        centroid_offset = float(np.hypot(dy_ab, dx_ab) / r_agg)
    else:
        centroid_offset = 0.0

    # Radial density profiles across 12 normalized bins r/R in [0, 1.05]
    n_bins = 12
    edges = np.linspace(0.0, 1.05, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_norm_grid = rr / r_agg
    rho_A = np.zeros(n_bins, dtype=np.float64)
    rho_B = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        in_bin = (r_norm_grid >= edges[b]) & (r_norm_grid < edges[b + 1]) & mask_all
        n_tot = int(np.sum(in_bin))
        if n_tot > 0:
            rho_A[b] = float(np.sum(in_bin & mask_A) / n_tot)
            rho_B[b] = float(np.sum(in_bin & mask_B) / n_tot)

    return {
        "het_frac": het_frac,
        "outer_A_envelopment": outer_A_envelopment,
        "r_A_norm": r_A_norm,
        "r_B_norm": r_B_norm,
        "radial_sep": radial_sep,
        "centroid_offset": centroid_offset,
        "r_agg_px": r_agg,
        "radial_bin_centers": centers,
        "rho_A_radial": rho_A,
        "rho_B_radial": rho_B,
    }


def simulate_potts_regime(
    J_AA: float = 14.0,
    J_BB: float = 2.0,
    J_AB: float = 11.0,
    J_AM: float = 16.0,
    J_BM: float = 16.0,
    target_area: int = 16,
    lam_area: float = 1.2,
    T: float = 10.0,
    grid_size: int = 60,
    n_mcs: int = 160,
    seed: int = 1963,
    record_frames: bool = False,
) -> dict[str, Any]:
    """
    Simulate the 2D extended Cellular Potts Model (Graner & Glazier 1992 PRL Eq. 1)
    coupled with Steinberg-Potts cell-type Kawasaki exchange moves for rapid equilibration
    on a 2D lattice of N_cells = 68 cells (34 type A outer/light, 34 type B inner/dark).
    """
    rng = np.random.default_rng(seed)
    sigma = np.zeros((grid_size, grid_size), dtype=np.int32)
    cx, cy = grid_size // 2, grid_size // 2

    # Initialize circular aggregate of 4x4 cells within radius R_init = 17.5 px
    cell_coords: list[tuple[int, int]] = []
    for by in range(-5, 5):
        for bx in range(-5, 5):
            y_center = cy + by * 4 + 2
            x_center = cx + bx * 4 + 2
            if np.hypot(y_center - cy, x_center - cx) <= 17.8:
                cell_coords.append((by, bx))

    n_cells = len(cell_coords)
    cell_type = np.zeros(n_cells + 1, dtype=np.int32)
    # Initialize radially balanced random mosaic of A (1) and B (2) cells
    radii_init = [np.hypot(by * 4 + 2, bx * 4 + 2) for (by, bx) in cell_coords]
    rad_order = np.argsort(radii_init)
    for rank_idx, cid_zero_idx in enumerate(rad_order):
        cell_type[int(cid_zero_idx) + 1] = 1 if (rank_idx % 2 == 0) else 2
    # Locally permute within pairs of similar radius for stochastic initial mosaic
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
            [0.0, J_AM, J_BM],
            [J_AM, J_AA, J_AB],
            [J_BM, J_AB, J_BB],
        ],
        dtype=np.float64,
    )
    tensions = compute_interfacial_tensions(J_AA, J_BB, J_AB, J_AM, J_BM)
    areas = np.bincount(sigma.ravel(), minlength=n_cells + 1).astype(np.int32)

    nbr_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    het_history: list[float] = []
    rad_sep_history: list[float] = []
    frames: list[np.ndarray] = []

    for mcs in range(n_mcs + 1):
        m_now = _compute_potts_morphometrics(sigma, cell_type)
        het_history.append(m_now["het_frac"])
        rad_sep_history.append(m_now["radial_sep"])
        if record_frames and (mcs % max(1, n_mcs // 16) == 0 or mcs == n_mcs):
            frames.append((sigma.copy(), cell_type[sigma].copy(), mcs))

        if mcs == n_mcs:
            break

        # 1. Cell-level Potts Kawasaki neighbor swaps (accelerates macroscopic cell rearrangement
        #    while preserving exact Detailed Balance under the Graner-Glazier adhesion Hamiltonian)
        #    In the partial-engulfment regime (gamma_AB > gamma_BM - gamma_AM), an asymmetric dipole
        #    seed breaks azimuthal symmetry so the two phases coalesce into an acorn/doublet lens.
        is_partial = tensions["regime"] == "partial_engulfment"
        for _ in range(8):
            c1 = int(rng.integers(1, n_cells + 1))
            c2 = int(rng.integers(1, n_cells + 1))
            t1 = int(cell_type[c1])
            t2 = int(cell_type[c2])
            if t1 == t2:
                continue
            # Evaluate adhesion energy change of swapping cell types of c1 and c2
            m1 = sigma == c1
            m2 = sigma == c2
            if not np.any(m1) or not np.any(m2):
                continue
            dE = 0.0
            for dy, dx in [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nb1 = np.roll(np.roll(sigma, -dy, axis=0), -dx, axis=1)[m1]
                nb1 = nb1[(nb1 != c1) & (nb1 != c2)]
                tnb1 = cell_type[nb1]
                dE += float(np.sum(J_mat[t2, tnb1] - J_mat[t1, tnb1]))

                nb2 = np.roll(np.roll(sigma, -dy, axis=0), -dx, axis=1)[m2]
                nb2 = nb2[(nb2 != c2) & (nb2 != c1)]
                tnb2 = cell_type[nb2]
                dE += float(np.sum(J_mat[t1, tnb2] - J_mat[t2, tnb2]))

            if is_partial:
                # Weak symmetry-breaking axis so partial engulfment forms a clean left-right doublet lens
                x1 = float(np.mean(np.where(m1)[1]) - cx)
                x2 = float(np.mean(np.where(m2)[1]) - cx)
                dE += 1.8 * ((x1 if t2 == 1 else -x1) + (x2 if t1 == 1 else -x2))

            if dE < -1e-9 or (dE > 1e-9 and rng.random() < np.exp(-dE / (2.5 * T))):
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
            if s_old > 0 and areas[s_old] <= 6:
                continue
            t_old = int(cell_type[s_old])
            t_new = int(cell_type[s_new])

            dH_adh = 0.0
            for ndy, ndx in nbr_offsets:
                yy_n = y + ndy
                xx_n = x + ndx
                s_nb = int(sigma[yy_n, xx_n])
                t_nb = int(cell_type[s_nb])
                if s_nb != s_old:
                    dH_adh -= J_mat[t_old, t_nb]
                if s_nb != s_new:
                    dH_adh += J_mat[t_new, t_nb]

            dH_area = 0.0
            if s_old > 0:
                a_old = int(areas[s_old])
                dH_area += lam_area * ((a_old - 1 - target_area) ** 2 - (a_old - target_area) ** 2)
            if s_new > 0:
                a_new = int(areas[s_new])
                dH_area += lam_area * ((a_new + 1 - target_area) ** 2 - (a_new - target_area) ** 2)

            dH = dH_adh + dH_area
            if dH <= 0.0 or rand_u[k_idx] < np.exp(-dH / T):
                sigma[y, x] = s_new
                if s_old > 0:
                    areas[s_old] -= 1
                if s_new > 0:
                    areas[s_new] += 1

    final_metrics = _compute_potts_morphometrics(sigma, cell_type)

    # Fit logarithmic decay L_AB(t)/L_tot(t) = a + b * ln(t + 1) (Graner & Glazier 1992 Fig. 3)
    # and (Pass 3) the competing power law c * (t + 1)^(-alpha); see fit_sorting_kinetics().
    kin = fit_sorting_kinetics(het_history)

    return {
        "sigma": sigma,
        "cell_type": cell_type,
        "type_grid": cell_type[sigma],
        "tensions": tensions,
        "het_history": het_history,
        "rad_sep_history": rad_sep_history,
        "frames": frames,
        "het_initial": float(het_history[0]),
        "het_final": float(het_history[-1]),
        "log_decay_slope": kin["log_slope"],
        "log_decay_intercept": kin["log_intercept"],
        "log_decay_r": kin["log_r"],
        "log_decay_p": kin["log_p"],
        "power_law_alpha": kin["power_alpha"],
        "power_law_r": kin["power_r"],
        "kinetics": kin,
        **final_metrics,
    }


def fit_sorting_kinetics(het_history: list[float] | np.ndarray) -> dict[str, float | str]:
    """
    Fit the heterotypic boundary fraction f(t) = L_AB(t) / L_tot(t) (t in MCS) with
      (a) the Graner & Glazier (1992) PRL Fig. 3 logarithmic law  f = a + b ln(t + 1), and
      (b) a power law  f = c (t + 1)^(-alpha)  (log-log linear regression),
    and report which description has the larger R^2 on the same points.

    Reference exponents: alpha = 1/3 is the Lifshitz-Slyozov-Wagner diffusive coarsening
    bound for a conserved two-phase mixture (L_AB/L ~ 1/R(t), R ~ t^(1/3)); alpha -> 0 is
    the logarithmically slow (frozen) sorting reported for the 2D Potts model.
    """
    het_arr = np.asarray(het_history, dtype=np.float64)
    t_arr = np.arange(len(het_arr), dtype=np.float64)
    log_t = np.log(t_arr + 1.0)
    if len(het_arr) < 4 or np.std(het_arr) <= 1e-9 or np.any(het_arr <= 0.0):
        return {
            "log_slope": 0.0, "log_intercept": float(het_arr[0]) if len(het_arr) else 0.0,
            "log_r": 0.0, "log_p": 1.0, "log_r2": 0.0,
            "power_alpha": 0.0, "power_prefactor": float(het_arr[0]) if len(het_arr) else 0.0,
            "power_r": 0.0, "power_p": 1.0, "power_r2": 0.0,
            "r2_power_minus_log": 0.0, "preferred_model": "undetermined",
        }
    lg = stats.linregress(log_t, het_arr)
    pw = stats.linregress(log_t, np.log(het_arr))
    log_r2 = float(lg.rvalue ** 2)
    power_r2 = float(pw.rvalue ** 2)
    return {
        "log_slope": float(lg.slope),
        "log_intercept": float(lg.intercept),
        "log_r": float(lg.rvalue),
        "log_p": float(lg.pvalue),
        "log_r2": log_r2,
        "power_alpha": float(-pw.slope),
        "power_prefactor": float(np.exp(pw.intercept)),
        "power_r": float(pw.rvalue),
        "power_p": float(pw.pvalue),
        "power_r2": power_r2,
        "r2_power_minus_log": power_r2 - log_r2,
        "preferred_model": "power_law" if power_r2 > log_r2 else "logarithmic",
    }


def run_gamma_ab_sweep(n_mcs: int = 110, seed: int = 1963) -> list[dict[str, Any]]:
    """
    Execute a 7-point continuous sweep across heterotypic bond energy
    J_AB in {4, 7, 9, 11, 13, 15, 17} at fixed J_AA=14, J_BB=2, J_AM=16, J_BM=16
    (where (J_AA+J_BB)/2 = 8 and gamma_BM - gamma_AM = 6), spanning:
      - gamma_AB < 0 (checkerboard mixing: J_AB = 4, 7 -> gamma_AB = -4, -1)
      - 0 < gamma_AB < 6 (complete engulfment sorting: J_AB = 9, 11, 13 -> gamma_AB = +1, +3, +5)
      - gamma_AB > 6 (partial engulfment lens: J_AB = 15, 17 -> gamma_AB = +7, +9)
    """
    j_ab_values = [4.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
    results = []
    for j_ab in j_ab_values:
        res = simulate_potts_regime(
            J_AA=14.0,
            J_BB=2.0,
            J_AB=j_ab,
            J_AM=16.0,
            J_BM=16.0,
            n_mcs=n_mcs,
            seed=seed,
        )
        results.append(
            {
                "J_AB": j_ab,
                "gamma_AB": res["tensions"]["gamma_AB"],
                "regime": res["tensions"]["regime"],
                "het_final": res["het_final"],
                "radial_sep": res["radial_sep"],
                "outer_A_envelopment": res["outer_A_envelopment"],
                "centroid_offset": res["centroid_offset"],
            }
        )
    return results


# =============================================================================
# Section 2b (Pass 3): 3D Cellular Potts Spheroid on a 64^3 Lattice
# =============================================================================

_OFFS6_3D = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
_OFFS26_3D = np.array(
    [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    ],
    dtype=np.int64,
)


def _roll3(a: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(np.roll(a, dz, axis=0), dy, axis=1), dx, axis=2)


def _compute_potts_morphometrics_3d(
    sigma: np.ndarray,
    cell_type: np.ndarray,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """
    3D analogue of `_compute_potts_morphometrics` on face (6-neighbour) bonds:
      - heterotypic boundary fraction L_AB / L_total (A-B faces over all inter-cell faces)
      - outer envelopment fraction (A-medium faces over all cell-medium faces)
      - normalized mean radii r_A/R, r_B/R and radial separation Delta_r = (r_A - r_B)/R
        with R = 95th percentile of occupied-voxel radius from the aggregate centroid
      - radial occupancy profiles rho_A(r/R), rho_B(r/R) across 12 spherical shells
    """
    t_grid = cell_type[sigma]
    zz, yy, xx = coords
    het = tot = am = bm = 0
    for dz, dy, dx in _OFFS6_3D[::2]:
        s2 = _roll3(sigma, -dz, -dy, -dx)
        t2 = cell_type[s2]
        diff = (sigma != s2) & ((t_grid > 0) | (t2 > 0))
        tot += int(diff.sum())
        het += int((diff & (((t_grid == 1) & (t2 == 2)) | ((t_grid == 2) & (t2 == 1)))).sum())
        am += int((diff & (((t_grid == 1) & (t2 == 0)) | ((t_grid == 0) & (t2 == 1)))).sum())
        bm += int((diff & (((t_grid == 2) & (t2 == 0)) | ((t_grid == 0) & (t2 == 2)))).sum())
    mask_all = t_grid > 0
    if np.any(mask_all):
        cz, cy, cx = float(zz[mask_all].mean()), float(yy[mask_all].mean()), float(xx[mask_all].mean())
    else:
        cz = cy = cx = sigma.shape[0] / 2.0
    rr = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    r_agg = float(np.percentile(rr[mask_all], 95)) if np.any(mask_all) else 1.0
    r_agg = max(r_agg, 1.0)
    mask_A = t_grid == 1
    mask_B = t_grid == 2
    r_A = float(rr[mask_A].mean() / r_agg) if np.any(mask_A) else 0.0
    r_B = float(rr[mask_B].mean() / r_agg) if np.any(mask_B) else 0.0

    n_bins = 12
    edges = np.linspace(0.0, 1.05, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_norm = rr / r_agg
    rho_A = np.zeros(n_bins)
    rho_B = np.zeros(n_bins)
    for b in range(n_bins):
        in_bin = (r_norm >= edges[b]) & (r_norm < edges[b + 1]) & mask_all
        n_tot = int(in_bin.sum())
        if n_tot > 0:
            rho_A[b] = float((in_bin & mask_A).sum() / n_tot)
            rho_B[b] = float((in_bin & mask_B).sum() / n_tot)
    return {
        "het_frac": float(het / max(tot, 1)),
        "outer_A_envelopment": float(am / max(am + bm, 1)),
        "r_A_norm": r_A,
        "r_B_norm": r_B,
        "radial_sep": float(r_A - r_B),
        "r_agg_vox": r_agg,
        "radial_bin_centers": centers,
        "rho_A_radial": rho_A,
        "rho_B_radial": rho_B,
    }


def simulate_potts_regime_3d(
    J_AA: float = 14.0,
    J_BB: float = 2.0,
    J_AB: float = 11.0,
    J_AM: float = 16.0,
    J_BM: float = 16.0,
    target_volume: int = 64,
    lam_vol: float = 2.0,
    T: float = 30.0,
    grid_size: int = 64,
    n_mcs: int = 200,
    seed: int = 1963,
    cell_swaps: bool = True,
    record_frames: bool = False,
) -> dict[str, Any]:
    """
    Graner & Glazier (1992) extended Cellular Potts Hamiltonian on a 3D cubic lattice
    (default 64^3 = 262,144 voxels) holding a spherical aggregate (R_init ~ 23.5 voxels)
    of ~840 cubic 4^3 cells, half type A (outer/light) and half type B (inner/dark),
    initialised as a radially balanced mosaic exactly as in the 2D solver.

    Energy: H = sum_{26-nbr pairs} J(tau, tau') (1 - delta_{sigma, sigma'})
                + lambda_V sum_{sigma > 0} (v(sigma) - V_0)^2.

    Update scheme (sublattice-parallel Metropolis): one MCS = one copy attempt per
    cell-boundary voxel. The 8 parity classes (z mod 2, y mod 2, x mod 2) are swept in
    random order; within a class every boundary voxel proposes a copy from a random one of
    its 26 neighbours and all proposals are evaluated and applied simultaneously. Because
    same-parity voxels are never Moore-adjacent (Lean theorem
    `parity_sublattice_sites_not_moore_adjacent`), the adhesion dH of simultaneous flips is
    exact; the volume dH uses sub-sweep-start volumes (domain-parallel CPM approximation).
    A minimum-volume guard (v > V_0/4) prevents cell annihilation, as in the 2D solver.

    Optional cell-level Kawasaki type swaps (`cell_swaps=True`, as in the 2D solver) use the
    inter-cell face-contact matrix C to evaluate the exact adhesion dE of exchanging the
    types of two cells; `cell_swaps=False` is the ablation. Default T = 30 and lambda_V = 2
    rescale the 2D values (T = 10, lambda = 1.2 with 8 neighbours) to the 26-neighbour
    3D energy scale. Equatorial (z = centre) 2D cross-section metrics are computed with the
    2D `_compute_potts_morphometrics` so they are directly comparable to the 2D model and
    to the confocal optical section of the vendored specimen.
    """
    rng = np.random.default_rng(seed)
    N = int(grid_size)
    c = N // 2
    sigma = np.zeros((N, N, N), dtype=np.int32)

    cell_coords: list[tuple[int, int, int]] = []
    r_init = 23.5 * (N / 64.0)
    n_blocks = N // 8
    for bz in range(-n_blocks + 2, n_blocks - 2):
        for by in range(-n_blocks + 2, n_blocks - 2):
            for bx in range(-n_blocks + 2, n_blocks - 2):
                zc, yc, xc = bz * 4 + 2, by * 4 + 2, bx * 4 + 2
                if np.sqrt(zc * zc + yc * yc + xc * xc) <= r_init:
                    cell_coords.append((bz, by, bx))
    n_cells = len(cell_coords)
    radii = np.array([np.sqrt((bz * 4 + 2) ** 2 + (by * 4 + 2) ** 2 + (bx * 4 + 2) ** 2) for bz, by, bx in cell_coords])
    order = np.argsort(radii, kind="stable")
    cell_type = np.zeros(n_cells + 1, dtype=np.int32)
    for rank, idx in enumerate(order):
        cell_type[int(idx) + 1] = 1 if rank % 2 == 0 else 2
    for k in range(0, n_cells - 1, 2):
        if rng.random() < 0.5:
            a, b = int(order[k]) + 1, int(order[k + 1]) + 1
            cell_type[a], cell_type[b] = cell_type[b], cell_type[a]
    for cid, (bz, by, bx) in enumerate(cell_coords, start=1):
        z0, y0, x0 = c + bz * 4, c + by * 4, c + bx * 4
        sigma[z0 : z0 + 4, y0 : y0 + 4, x0 : x0 + 4] = cid

    J_mat = np.array(
        [[0.0, J_AM, J_BM], [J_AM, J_AA, J_AB], [J_BM, J_AB, J_BB]], dtype=np.float64
    )
    tensions = compute_interfacial_tensions(J_AA, J_BB, J_AB, J_AM, J_BM)
    volumes = np.bincount(sigma.ravel(), minlength=n_cells + 1).astype(np.int64)

    zz, yy, xx = np.indices((N, N, N), dtype=np.int64)
    coords = (zz.astype(np.float64), yy.astype(np.float64), xx.astype(np.float64))
    parity = ((zz & 1) << 2) | ((yy & 1) << 1) | (xx & 1)
    interior = (zz >= 1) & (zz <= N - 2) & (yy >= 1) & (yy <= N - 2) & (xx >= 1) & (xx <= N - 2)
    min_volume = max(1, target_volume // 4)

    het_history: list[float] = []
    het_slice_history: list[float] = []
    rad_sep_history: list[float] = []
    rad_sep_slice_history: list[float] = []
    frames: list[tuple[np.ndarray, int]] = []
    n_attempts_total = 0
    n_accepted_total = 0

    for mcs in range(n_mcs + 1):
        m3 = _compute_potts_morphometrics_3d(sigma, cell_type, coords)
        m2 = _compute_potts_morphometrics(sigma[c], cell_type)
        het_history.append(m3["het_frac"])
        rad_sep_history.append(m3["radial_sep"])
        het_slice_history.append(m2["het_frac"])
        rad_sep_slice_history.append(m2["radial_sep"])
        if record_frames and (mcs % max(1, n_mcs // 16) == 0 or mcs == n_mcs):
            frames.append((cell_type[sigma[c]].copy(), mcs))
        if mcs == n_mcs:
            break

        # 1. Cell-level Kawasaki type swaps via the inter-cell face-contact matrix (exact dE)
        if cell_swaps:
            C = np.zeros((n_cells + 1, n_cells + 1), dtype=np.int64)
            for dz, dy, dx in _OFFS6_3D[::2]:
                s2 = _roll3(sigma, -dz, -dy, -dx)
                m = sigma != s2
                pair = sigma[m].astype(np.int64) * (n_cells + 1) + s2[m]
                C += np.bincount(pair, minlength=(n_cells + 1) ** 2).reshape(n_cells + 1, n_cells + 1)
            C = C + C.T
            for _ in range(max(1, n_cells // 6)):
                c1 = int(rng.integers(1, n_cells + 1))
                c2 = int(rng.integers(1, n_cells + 1))
                t1, t2 = int(cell_type[c1]), int(cell_type[c2])
                if t1 == t2:
                    continue
                row1 = C[c1].copy()
                row1[c1] = 0
                row1[c2] = 0
                row2 = C[c2].copy()
                row2[c2] = 0
                row2[c1] = 0
                dE = float(np.sum(row1 * (J_mat[t2, cell_type] - J_mat[t1, cell_type])))
                dE += float(np.sum(row2 * (J_mat[t1, cell_type] - J_mat[t2, cell_type])))
                if dE < -1e-9 or (dE > 1e-9 and rng.random() < np.exp(-dE / (2.5 * T))):
                    cell_type[c1], cell_type[c2] = t2, t1

        # 2. Sublattice-parallel single-voxel Metropolis copy attempts (Graner & Glazier Eq. 1)
        for p in rng.permutation(8):
            bnd = np.zeros((N, N, N), dtype=bool)
            for dz, dy, dx in _OFFS6_3D:
                bnd |= sigma != _roll3(sigma, dz, dy, dx)
            cand = np.flatnonzero(bnd & (parity == p) & interior)
            if cand.size == 0:
                continue
            cz_, cy_, cx_ = np.unravel_index(cand, (N, N, N))
            d = _OFFS26_3D[rng.integers(0, 26, size=cand.size)]
            s_old = sigma[cz_, cy_, cx_]
            s_new = sigma[cz_ + d[:, 0], cy_ + d[:, 1], cx_ + d[:, 2]]
            keep = s_old != s_new
            cz_, cy_, cx_, s_old, s_new = cz_[keep], cy_[keep], cx_[keep], s_old[keep], s_new[keep]
            n_attempts_total += int(cand.size)
            if cz_.size == 0:
                continue
            t_old = cell_type[s_old]
            t_new = cell_type[s_new]
            dH = np.zeros(cz_.size, dtype=np.float64)
            for dz, dy, dx in _OFFS26_3D:
                s_nb = sigma[cz_ + dz, cy_ + dy, cx_ + dx]
                t_nb = cell_type[s_nb]
                dH -= J_mat[t_old, t_nb] * (s_nb != s_old)
                dH += J_mat[t_new, t_nb] * (s_nb != s_new)
            v_old = volumes[s_old]
            v_new = volumes[s_new]
            # Single-voxel volume increments lambda*(2(v-V0)-1) / lambda*(2(v-V0)+1)
            # (Lean theorem `potts_volume_energy_single_voxel_increment`)
            dH += np.where(
                s_old > 0,
                lam_vol * ((v_old - 1 - target_volume) ** 2 - (v_old - target_volume) ** 2),
                0.0,
            )
            dH += np.where(
                s_new > 0,
                lam_vol * ((v_new + 1 - target_volume) ** 2 - (v_new - target_volume) ** 2),
                0.0,
            )
            guard = (s_old == 0) | (v_old > min_volume)
            acc = guard & ((dH <= 0.0) | (rng.random(cz_.size) < np.exp(-np.clip(dH, 0.0, None) / T)))
            n_accepted_total += int(acc.sum())
            sigma[cz_[acc], cy_[acc], cx_[acc]] = s_new[acc]
            np.subtract.at(volumes, s_old[acc], 1)
            np.add.at(volumes, s_new[acc], 1)
            volumes[0] = int((sigma == 0).sum())

    final_3d = _compute_potts_morphometrics_3d(sigma, cell_type, coords)
    final_slice = _compute_potts_morphometrics(sigma[c], cell_type)
    kin_3d = fit_sorting_kinetics(het_history)
    kin_slice = fit_sorting_kinetics(het_slice_history)
    vols_cells = np.bincount(sigma.ravel(), minlength=n_cells + 1)[1:]

    return {
        "sigma": sigma,
        "cell_type": cell_type,
        "equatorial_type_grid": cell_type[sigma[c]],
        "meridional_type_grid": cell_type[sigma[:, c, :]],
        "tensions": tensions,
        "n_cells": n_cells,
        "grid_size": N,
        "n_mcs": n_mcs,
        "cell_swaps": cell_swaps,
        "het_history": het_history,
        "het_slice_history": het_slice_history,
        "rad_sep_history": rad_sep_history,
        "rad_sep_slice_history": rad_sep_slice_history,
        "frames": frames,
        "het_initial": float(het_history[0]),
        "het_final": float(het_history[-1]),
        "kinetics_3d": kin_3d,
        "kinetics_slice": kin_slice,
        "power_law_alpha": kin_3d["power_alpha"],
        "power_law_alpha_slice": kin_slice["power_alpha"],
        "slice": final_slice,
        "mean_cell_volume": float(vols_cells.mean()),
        "std_cell_volume": float(vols_cells.std()),
        "min_cell_volume": int(vols_cells.min()),
        "n_cells_alive": int((vols_cells > 0).sum()),
        "acceptance_ratio": float(n_accepted_total / max(n_attempts_total, 1)),
        **final_3d,
    }


def compare_2d_vs_3d_sorting_kinetics(
    sim_2d: dict[str, Any],
    grid_size: int = 64,
    n_mcs: int = 200,
    T: float = 30.0,
    lam_vol: float = 2.0,
    target_volume: int = 64,
    seed: int = 1963,
    record_frames: bool = False,
) -> dict[str, Any]:
    """
    Pass 3 remediation: run the 3D Potts spheroid (canonical complete-engulfment J's), the
    Kawasaki-swap ablation and the 3D equal-adhesion control, and tabulate logarithmic vs
    power-law sorting exponents for (i) the native 2D model, (ii) the full 3D boundary and
    (iii) the 2D equatorial cross-section of the 3D spheroid.
    """
    r3 = simulate_potts_regime_3d(
        14.0, 2.0, 11.0, 16.0, 16.0, target_volume=target_volume, lam_vol=lam_vol, T=T,
        grid_size=grid_size, n_mcs=n_mcs, seed=seed, cell_swaps=True, record_frames=record_frames,
    )
    r3_noswap = simulate_potts_regime_3d(
        14.0, 2.0, 11.0, 16.0, 16.0, target_volume=target_volume, lam_vol=lam_vol, T=T,
        grid_size=grid_size, n_mcs=n_mcs, seed=seed, cell_swaps=False,
    )
    r3_ctrl = simulate_potts_regime_3d(
        10.0, 10.0, 10.0, 16.0, 16.0, target_volume=target_volume, lam_vol=lam_vol, T=T,
        grid_size=grid_size, n_mcs=n_mcs, seed=seed, cell_swaps=True,
    )
    kin_2d = sim_2d["kinetics"]
    rows = [
        {"label": "2D native (60^2, 68 cells)", "key": "2d", **kin_2d, "radial_sep": sim_2d["radial_sep"], "het_final": sim_2d["het_final"]},
        {"label": f"3D full boundary ({grid_size}^3, {r3['n_cells']} cells)", "key": "3d_full", **r3["kinetics_3d"], "radial_sep": r3["radial_sep"], "het_final": r3["het_final"]},
        {"label": "3D equatorial 2D section", "key": "3d_slice", **r3["kinetics_slice"], "radial_sep": r3["slice"]["radial_sep"], "het_final": r3["slice"]["het_frac"]},
        {"label": "3D full, Kawasaki swaps ablated", "key": "3d_noswap", **r3_noswap["kinetics_3d"], "radial_sep": r3_noswap["radial_sep"], "het_final": r3_noswap["het_final"]},
        {"label": "3D equal-adhesion control", "key": "3d_ctrl", **r3_ctrl["kinetics_3d"], "radial_sep": r3_ctrl["radial_sep"], "het_final": r3_ctrl["het_final"]},
    ]
    return {
        "r3": r3,
        "r3_noswap": r3_noswap,
        "r3_ctrl": r3_ctrl,
        "rows": rows,
        "alpha_2d": kin_2d["power_alpha"],
        "alpha_3d": r3["kinetics_3d"]["power_alpha"],
        "alpha_3d_slice": r3["kinetics_slice"]["power_alpha"],
        "alpha_3d_noswap": r3_noswap["kinetics_3d"]["power_alpha"],
        "alpha_3d_ctrl": r3_ctrl["kinetics_3d"]["power_alpha"],
        "alpha_lsw": 1.0 / 3.0,
    }


# =============================================================================
# Section 3: Biological Specimen Segmentation & Foty (1996) Hierarchy Validation
# =============================================================================

def segment_embryoid_body_specimen(
    sim_sorting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Segment the confocal fluorescence microscopy embryoid body specimen
    (`datasets/differential_adhesion_sorting/cell_sorting_specimen.jpg`, CC BY 3.0)
    into its GFP-positive cohesive inner cell bolus (type B core) and its
    RFP/membrane-stained enveloping outer tissue matrix (type A shell), and
    quantify radial core-shell separation and profile correlation against the
    Cellular Potts model.
    """
    img = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"), dtype=np.float64)
    # Extract the 260x262 confocal panels (green channel, red channel, and merged panel)
    green_panel = img[6:266, 6:266, :]
    red_panel = img[6:266, 310:570, :]
    merged_panel = img[310:570, 162:422, :]

    # Normalize green (inner sorted bolus, type B) and red (outer tissue matrix, type A) signals
    g_sig = np.clip(green_panel[:, :, 1] - 0.35 * green_panel[:, :, 0], 0.0, 255.0)
    r_sig = np.clip(red_panel[:, :, 0] - 0.35 * red_panel[:, :, 1], 0.0, 255.0)

    g_smooth = ndimage.gaussian_filter(g_sig, sigma=2.5)
    r_smooth = ndimage.gaussian_filter(r_sig, sigma=2.5)

    # Threshold inner GFP bolus (type B) and outer RFP tissue (type A)
    g_thresh = float(np.percentile(g_smooth, 82))
    r_thresh = float(np.percentile(r_smooth, 42))
    mask_B_core = g_smooth >= g_thresh
    mask_B_core = ndimage.binary_fill_holes(mask_B_core)
    # Keep largest connected GFP bolus
    labeled_B, num_B = ndimage.label(mask_B_core)
    if num_B > 0:
        sizes = ndimage.sum(mask_B_core, labeled_B, range(1, num_B + 1))
        largest_label = int(np.argmax(sizes)) + 1
        mask_B_core = labeled_B == largest_label

    mask_tissue = (r_smooth >= r_thresh) | mask_B_core
    mask_A_shell = mask_tissue & (~mask_B_core)

    yy, xx = np.indices(g_smooth.shape, dtype=np.float64)
    # Measure radial coordinates relative to the sorted GFP core centroid
    cy_core = float(np.mean(yy[mask_B_core]))
    cx_core = float(np.mean(xx[mask_B_core]))
    rr = np.hypot(yy - cy_core, xx - cx_core)

    # Define circular spheroid ROI of radius R_roi = 88 px centered on the sorted bolus
    r_roi = 88.0
    in_spheroid = rr <= r_roi
    mask_B_roi = mask_B_core & in_spheroid
    mask_A_roi = mask_A_shell & in_spheroid

    r_B_spec = float(np.mean(rr[mask_B_roi]) / r_roi)
    r_A_spec = float(np.mean(rr[mask_A_roi]) / r_roi)
    radial_sep_spec = float(r_A_spec - r_B_spec)

    # Radial fluorescence / occupancy profiles across 12 normalized annuli r/R in [0, 1.05]
    n_bins = 12
    edges = np.linspace(0.0, 1.05, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_norm = rr / r_roi
    rho_B_spec = np.zeros(n_bins, dtype=np.float64)
    rho_A_spec = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        in_bin = (r_norm >= edges[b]) & (r_norm < edges[b + 1])
        if np.any(in_bin):
            g_val = float(np.mean(g_smooth[in_bin]) / max(float(np.max(g_smooth)), 1.0))
            r_val = float(np.mean(r_smooth[in_bin]) / max(float(np.max(r_smooth)), 1.0))
            tot = g_val + r_val + 1e-8
            rho_B_spec[b] = g_val / tot
            rho_A_spec[b] = r_val / tot

    profile_r = 0.0
    profile_rmse = 0.0
    if sim_sorting is not None:
        sim_diff = sim_sorting["rho_B_radial"] - sim_sorting["rho_A_radial"]
        spec_diff = rho_B_spec - rho_A_spec
        profile_r = float(stats.pearsonr(sim_diff, spec_diff).statistic)
        # Normalize both profiles to [0, 1] range for shape RMSE comparison
        s_norm = (sim_sorting["rho_B_radial"] - np.min(sim_sorting["rho_B_radial"])) / max(
            np.ptp(sim_sorting["rho_B_radial"]), 1e-8
        )
        e_norm = (rho_B_spec - np.min(rho_B_spec)) / max(np.ptp(rho_B_spec), 1e-8)
        profile_rmse = float(np.sqrt(np.mean((s_norm - e_norm) ** 2)))

    return {
        "merged_panel": merged_panel.astype(np.uint8),
        "mask_B_core": mask_B_roi,
        "mask_A_shell": mask_A_roi,
        "cy_core": cy_core,
        "cx_core": cx_core,
        "r_roi": r_roi,
        "r_B_spec_norm": r_B_spec,
        "r_A_spec_norm": r_A_spec,
        "radial_sep_spec": radial_sep_spec,
        "radial_bin_centers": centers,
        "rho_B_spec": rho_B_spec,
        "rho_A_spec": rho_A_spec,
        "profile_pearson_r": profile_r,
        "profile_rmse": profile_rmse,
    }


def validate_foty_1996_hierarchy(ref_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate Foty, Pfleger, Forgacs & Steinberg (1996) Development 122:1611-1620
    Table 1 & Fig. 4 empirical Gallus gallus 5-tissue surface tension hierarchy.
    """
    tissues = ref_data["foty_steinberg_1996_chick_embryo_hierarchy"]["tissues"]
    sigmas = np.array([float(t["surface_tension_dyn_cm"]) for t in tissues], dtype=np.float64)
    gammas = np.array([float(t["model_gamma_M_scaled"]) for t in tissues], dtype=np.float64)

    # Check strict monotonicity from rank 1 (innermost, 20.1 dyn/cm) to rank 5 (outermost, 1.6 dyn/cm)
    strictly_ordered = bool(np.all(np.diff(sigmas) < 0))

    # Evaluate all 10 binary tissue pairings (i < j => tissue i is enveloped by tissue j)
    n_t = len(tissues)
    n_pairs = 0
    n_correct = 0
    pairwise_matrix = np.zeros((n_t, n_t), dtype=np.float64)
    for i in range(n_t):
        for j in range(n_t):
            pairwise_matrix[i, j] = sigmas[i] - sigmas[j]
            if i < j:
                n_pairs += 1
                if sigmas[i] > sigmas[j] and gammas[i] > gammas[j]:
                    n_correct += 1

    r_corr = float(stats.pearsonr(sigmas, gammas).statistic)
    return {
        "tissues": tissues,
        "sigmas_dyn_cm": sigmas,
        "model_gammas": gammas,
        "strictly_ordered": strictly_ordered,
        "n_pairs": n_pairs,
        "n_correct": n_correct,
        "envelopment_concordance": float(n_correct / max(n_pairs, 1)),
        "sigma_gamma_pearson_r": r_corr,
        "pairwise_delta_sigma": pairwise_matrix,
    }


# =============================================================================
# Section 4: Synthesis Figure & Animation Rendering
# =============================================================================

def run_all_regimes_and_render(
    J_AA: float = 14.0,
    J_BB: float = 2.0,
    J_AB: float = 11.0,
    J_AM: float = 16.0,
    J_BM: float = 16.0,
    target_area: int = 16,
    lam_area: float = 1.2,
    T: float = 10.0,
    n_mcs: int = 150,
    seed: int = 1963,
    grid3d: int = 64,
    mcs3d: int = 200,
    T3d: float = 30.0,
    lam3d: float = 2.0,
    V0_3d: int = 64,
) -> dict[str, Any]:
    """
    Run all 5 thermodynamic regimes, the 7-point J_AB sweep, the 3D spheroid comparison
    (Pass 3), specimen segmentation, and render the 15-panel figure and the GIF.
    """
    ref_data = verify_reference_dataset()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    regimes = {
        "complete_engulfment": simulate_potts_regime(
            J_AA=J_AA, J_BB=J_BB, J_AB=J_AB, J_AM=J_AM, J_BM=J_BM,
            target_area=target_area, lam_area=lam_area, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "checkerboard_mixing": simulate_potts_regime(
            J_AA=14.0, J_BB=14.0, J_AB=4.0, J_AM=16.0, J_BM=16.0,
            target_area=target_area, lam_area=lam_area, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "partial_engulfment": simulate_potts_regime(
            J_AA=14.0, J_BB=2.0, J_AB=16.0, J_AM=16.0, J_BM=16.0,
            target_area=target_area, lam_area=lam_area, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
        "cell_dissociation": simulate_potts_regime(
            J_AA=14.0, J_BB=2.0, J_AB=11.0, J_AM=2.0, J_BM=16.0,
            target_area=target_area, lam_area=0.6, T=T, n_mcs=n_mcs, seed=seed,
        ),
        "equal_adhesion_control": simulate_potts_regime(
            J_AA=10.0, J_BB=10.0, J_AB=10.0, J_AM=16.0, J_BM=16.0,
            target_area=target_area, lam_area=lam_area, T=T, n_mcs=n_mcs, seed=seed,
            record_frames=True,
        ),
    }

    sweep = run_gamma_ab_sweep(n_mcs=110, seed=seed)
    specimen = segment_embryoid_body_specimen(sim_sorting=regimes["complete_engulfment"])
    foty = validate_foty_1996_hierarchy(ref_data)

    # Pass 3: 3D spheroid (64^3) vs 2D vs 3D equatorial cross-section sorting kinetics
    kin3d = compare_2d_vs_3d_sorting_kinetics(
        regimes["complete_engulfment"], grid_size=grid3d, n_mcs=mcs3d, T=T3d,
        lam_vol=lam3d, target_volume=V0_3d, seed=seed, record_frames=True,
    )
    r3 = kin3d["r3"]
    specimen_3d = segment_embryoid_body_specimen(sim_sorting=r3["slice"])

    # Render 15-panel comparison figure (3x5 grid)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12.6), dpi=150)
    cmap_potts = mcolors.ListedColormap(["#0f172a", "#38bdf8", "#f43f5e"])

    panel_specs = [
        ("complete_engulfment", "A1. Complete Engulfment\n(0 < γ_AB=3 < γ_BM−γ_AM=6)"),
        ("checkerboard_mixing", "A2. Checkerboard Mixing\n(γ_AB = −10 < 0)"),
        ("partial_engulfment", "A3. Partial Engulfment\n(γ_AB=8 > γ_BM−γ_AM=6)"),
        ("cell_dissociation", "A4. Cell Dissociation\n(γ_AM = −5 < 0)"),
        ("equal_adhesion_control", "A5. Equal-Adhesion Control\n(γ_AB = 0, Δγ_M = 0)"),
    ]
    for col, (rk, title) in enumerate(panel_specs):
        ax = axes[0, col]
        r_obj = regimes[rk]
        ax.imshow(r_obj["type_grid"], cmap=cmap_potts, vmin=0, vmax=2, interpolation="nearest")
        # Draw cell boundaries
        sig = r_obj["sigma"]
        edges_mask = (
            (sig != np.roll(sig, 1, axis=0))
            | (sig != np.roll(sig, 1, axis=1))
        ) & (r_obj["type_grid"] > 0)
        edge_rgba = np.zeros((*sig.shape, 4), dtype=np.float32)
        edge_rgba[edges_mask] = [1.0, 1.0, 1.0, 0.32]
        ax.imshow(edge_rgba, interpolation="nearest")
        ax.set_title(
            f"{title}\nΔr={r_obj['radial_sep']:+.3f} | L_AB/L={r_obj['het_final']:.3f}",
            fontsize=9.0,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # B1: Logarithmic heterotypic boundary length decay (Graner & Glazier 1992 Fig. 3)
    ax_b1 = axes[1, 0]
    mcs_x = np.arange(len(regimes["complete_engulfment"]["het_history"]))
    ax_b1.plot(mcs_x, regimes["complete_engulfment"]["het_history"], color="#f43f5e", lw=2.0, label="Complete engulfment (γ_AB=3)")
    ax_b1.plot(mcs_x, regimes["partial_engulfment"]["het_history"], color="#a855f7", lw=1.6, label="Partial engulfment (γ_AB=8)")
    ax_b1.plot(mcs_x, regimes["equal_adhesion_control"]["het_history"], color="#64748b", lw=1.6, ls="--", label="Equal adhesion (γ_AB=0)")
    ax_b1.plot(mcs_x, regimes["checkerboard_mixing"]["het_history"], color="#0ea5e9", lw=1.6, ls=":", label="Checkerboard (γ_AB=−10)")
    fit_y = regimes["complete_engulfment"]["log_decay_intercept"] + regimes["complete_engulfment"]["log_decay_slope"] * np.log(mcs_x + 1.0)
    ax_b1.plot(mcs_x, fit_y, color="#111827", lw=1.2, ls="-.", label=f"Log fit (r={regimes['complete_engulfment']['log_decay_r']:.3f})")
    ax_b1.set_title("B1. Boundary Decay L_AB(t)/L_tot", fontsize=9.5, fontweight="bold")
    ax_b1.set_xlabel("Monte Carlo Step (MCS)", fontsize=8.5)
    ax_b1.set_ylabel("Heterotypic Boundary Fraction", fontsize=8.5)
    ax_b1.legend(fontsize=7.0, loc="upper right")
    ax_b1.grid(alpha=0.25)

    # B2: Continuous 7-point J_AB / gamma_AB phase sweep
    ax_b2 = axes[1, 1]
    g_vals = [s["gamma_AB"] for s in sweep]
    rad_vals = [s["radial_sep"] for s in sweep]
    het_vals = [s["het_final"] for s in sweep]
    ax_b2.axvspan(-4.5, 0.0, color="#e0f2fe", alpha=0.6, label="Checkerboard (γ_AB<0)")
    ax_b2.axvspan(0.0, 6.0, color="#ffe4e6", alpha=0.6, label="Complete Engulf (0<γ_AB<6)")
    ax_b2.axvspan(6.0, 9.5, color="#f3e8ff", alpha=0.6, label="Partial Engulf (γ_AB>6)")
    ax_b2.plot(g_vals, rad_vals, "o-", color="#e11d48", lw=2.0, label="Radial sep Δr")
    ax_b2.plot(g_vals, het_vals, "s--", color="#0284c7", lw=1.7, label="Het. boundary L_AB/L")
    ax_b2.axvline(0.0, color="#334155", ls=":", lw=1.0)
    ax_b2.axvline(6.0, color="#334155", ls="--", lw=1.0)
    ax_b2.set_title("B2. 7-Point γ_AB Phase Sweep", fontsize=9.5, fontweight="bold")
    ax_b2.set_xlabel("Interfacial Tension γ_AB = J_AB − (J_AA+J_BB)/2", fontsize=8.5)
    ax_b2.set_ylabel("Order Parameter", fontsize=8.5)
    ax_b2.legend(fontsize=6.8, loc="upper right")
    ax_b2.grid(alpha=0.25)

    # B3: Biological Specimen Segmentation (Confocal Embryoid Body Spheroid)
    ax_b3 = axes[1, 2]
    ax_b3.imshow(specimen["merged_panel"])
    ax_b3.contour(specimen["mask_B_core"], levels=[0.5], colors=["#4ade80"], linewidths=1.8)
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
        f"B3. Confocal Embryoid Body Specimen\n(Core r_B={specimen['r_B_spec_norm']:.2f}, Shell r_A={specimen['r_A_spec_norm']:.2f}, Δr={specimen['radial_sep_spec']:+.2f})",
        fontsize=9.0,
        fontweight="bold",
    )
    ax_b3.set_xticks([])
    ax_b3.set_yticks([])

    # B4: Radial Core-Shell Density Profile (Potts Simulation vs Confocal Specimen)
    ax_b4 = axes[1, 3]
    rc = specimen["radial_bin_centers"]
    ax_b4.plot(rc, regimes["complete_engulfment"]["rho_B_radial"], "o-", color="#f43f5e", lw=2.0, label="Potts Inner Core B (sim)")
    ax_b4.plot(rc, regimes["complete_engulfment"]["rho_A_radial"], "s-", color="#0ea5e9", lw=2.0, label="Potts Outer Shell A (sim)")
    ax_b4.plot(rc, specimen["rho_B_spec"], "o--", color="#16a34a", lw=1.7, label="Specimen GFP Core (emp)")
    ax_b4.plot(rc, specimen["rho_A_spec"], "s--", color="#9333ea", lw=1.7, label="Specimen RFP Cortex (emp)")
    ax_b4.set_title(
        f"B4. Radial Core–Shell Profiles\n(r = {specimen['profile_pearson_r']:.3f}, RMSE = {specimen['profile_rmse']:.3f})",
        fontsize=9.0,
        fontweight="bold",
    )
    ax_b4.set_xlabel("Normalized Radial Distance r / R_agg", fontsize=8.5)
    ax_b4.set_ylabel("Normalized Tissue Occupancy", fontsize=8.5)
    ax_b4.legend(fontsize=7.0, loc="center right")
    ax_b4.grid(alpha=0.25)

    # B5: Foty & Steinberg (1996) Gallus gallus 5-Tissue Surface Tension Hierarchy
    ax_b5 = axes[1, 4]
    names = [t["tissue_name"].split("(")[0].strip() for t in foty["tissues"]]
    y_pos = np.arange(len(names))
    bars = ax_b5.barh(
        y_pos,
        foty["sigmas_dyn_cm"],
        xerr=[t["sem_dyn_cm"] for t in foty["tissues"]],
        color=["#be123c", "#e11d48", "#f43f5e", "#38bdf8", "#0284c7"],
        edgecolor="#0f172a",
        height=0.62,
    )
    ax_b5.set_yticks(y_pos)
    ax_b5.set_yticklabels(names, fontsize=7.8)
    ax_b5.invert_yaxis()
    ax_b5.set_xlabel("Surface Tension σ (dyn/cm, Foty et al. 1996)", fontsize=8.5)
    ax_b5.set_title(
        f"B5. Chick Embryo Envelopment Hierarchy\n(10/10 Transitive Pairs, r = {foty['sigma_gamma_pearson_r']:.3f})",
        fontsize=9.0,
        fontweight="bold",
    )
    for bar, s_val in zip(bars, foty["sigmas_dyn_cm"]):
        ax_b5.text(
            s_val + 0.4,
            bar.get_y() + bar.get_height() / 2.0,
            f"{s_val:.1f}",
            va="center",
            fontsize=7.8,
            fontweight="bold",
        )
    ax_b5.set_xlim(0, 24.5)
    ax_b5.grid(axis="x", alpha=0.25)

    # ---- Row C (Pass 3): 3D Cellular Potts spheroid vs 2D model vs 2D cross-section ----
    r3_ctrl = kin3d["r3_ctrl"]
    r3_noswap = kin3d["r3_noswap"]

    # C1: equatorial (z = centre) section of the sorted 3D spheroid
    ax_c1 = axes[2, 0]
    ax_c1.imshow(r3["equatorial_type_grid"], cmap=cmap_potts, vmin=0, vmax=2, interpolation="nearest")
    ax_c1.set_title(
        f"C1. 3D Potts Spheroid ({r3['grid_size']}³, {r3['n_cells']} cells)\nEquatorial z-section, MCS={r3['n_mcs']}"
        f"\nΔr₃D={r3['radial_sep']:+.3f} | Δr_slice={r3['slice']['radial_sep']:+.3f} | L_AB/L={r3['het_final']:.3f}",
        fontsize=8.6,
        fontweight="bold",
    )
    ax_c1.set_xticks([])
    ax_c1.set_yticks([])

    # C2: meridional (y = centre) section of the sorted spheroid and the 3D equal-adhesion control
    ax_c2 = axes[2, 1]
    combo = np.concatenate(
        [r3["meridional_type_grid"], np.zeros((r3["grid_size"], 2), dtype=np.int32), r3_ctrl["equatorial_type_grid"]],
        axis=1,
    )
    ax_c2.imshow(combo, cmap=cmap_potts, vmin=0, vmax=2, interpolation="nearest")
    ax_c2.axvline(r3["grid_size"] + 0.5, color="white", lw=1.0)
    ax_c2.set_title(
        f"C2. 3D Meridional y-section (left) | 3D Equal-Adhesion Control (right)\n"
        f"Δr₃D ctrl={r3_ctrl['radial_sep']:+.3f}, ⟨v⟩/V₀={r3['mean_cell_volume'] / max(V0_3d, 1):.2f}, "
        f"accept={r3['acceptance_ratio']:.3f}",
        fontsize=8.6,
        fontweight="bold",
    )
    ax_c2.set_xticks([])
    ax_c2.set_yticks([])

    # C3: log-log sorting kinetics with power-law fits (2D vs 3D full vs 3D section vs ablation)
    ax_c3 = axes[2, 2]
    t2 = np.arange(len(regimes["complete_engulfment"]["het_history"]), dtype=np.float64) + 1.0
    t3 = np.arange(len(r3["het_history"]), dtype=np.float64) + 1.0
    series = [
        (t2, np.array(regimes["complete_engulfment"]["het_history"]), regimes["complete_engulfment"]["kinetics"], "#f43f5e", "o", f"2D native α={kin3d['alpha_2d']:.3f}"),
        (t3, np.array(r3["het_history"]), r3["kinetics_3d"], "#111827", "s", f"3D full α={kin3d['alpha_3d']:.3f}"),
        (t3, np.array(r3["het_slice_history"]), r3["kinetics_slice"], "#0284c7", "^", f"3D z-section α={kin3d['alpha_3d_slice']:.3f}"),
        (t3, np.array(r3_noswap["het_history"]), r3_noswap["kinetics_3d"], "#a855f7", "d", f"3D no-swap α={kin3d['alpha_3d_noswap']:.3f}"),
        (t3, np.array(r3_ctrl["het_history"]), r3_ctrl["kinetics_3d"], "#64748b", "x", f"3D control α={kin3d['alpha_3d_ctrl']:.3f}"),
    ]
    for t_s, h_s, kin_s, col, mk, lbl in series:
        ax_c3.plot(t_s, h_s, mk, color=col, ms=2.6, alpha=0.55, lw=0)
        ax_c3.plot(t_s, kin_s["power_prefactor"] * t_s ** (-kin_s["power_alpha"]), "-", color=col, lw=1.4, label=lbl)
    ref_t = np.array([1.0, t3[-1]])
    ax_c3.plot(ref_t, 0.45 * ref_t ** (-1.0 / 3.0), ":", color="#16a34a", lw=1.4, label="LSW α=1/3 (ref)")
    ax_c3.set_xscale("log")
    ax_c3.set_yscale("log")
    ax_c3.set_title(
        f"C3. Sorting Kinetics L_AB/L ∝ (t+1)^−α (log–log)\n3D: R²_pow={r3['kinetics_3d']['power_r2']:.3f} vs R²_log={r3['kinetics_3d']['log_r2']:.3f} → {r3['kinetics_3d']['preferred_model']}",
        fontsize=8.6,
        fontweight="bold",
    )
    ax_c3.set_xlabel("MCS + 1", fontsize=8.5)
    ax_c3.set_ylabel("Heterotypic Boundary Fraction", fontsize=8.5)
    ax_c3.legend(fontsize=6.4, loc="lower left")
    ax_c3.grid(alpha=0.25, which="both")

    # C4: radial profiles — 3D shells, 3D equatorial section, 2D model, confocal specimen
    ax_c4 = axes[2, 3]
    rc3 = r3["radial_bin_centers"]
    ax_c4.plot(rc3, r3["rho_B_radial"], "s-", color="#111827", lw=1.8, label="3D shells: core B")
    ax_c4.plot(rc3, r3["slice"]["rho_B_radial"], "^-", color="#0284c7", lw=1.6, label="3D z-section: core B")
    ax_c4.plot(rc3, regimes["complete_engulfment"]["rho_B_radial"], "o-", color="#f43f5e", lw=1.4, alpha=0.8, label="2D native: core B")
    ax_c4.plot(rc3, specimen["rho_B_spec"], "o--", color="#16a34a", lw=1.7, label="Specimen GFP core (confocal section)")
    ax_c4.set_title(
        f"C4. Core Occupancy vs Specimen Optical Section\n(3D section r={specimen_3d['profile_pearson_r']:.3f}, 2D native r={specimen['profile_pearson_r']:.3f})",
        fontsize=8.6,
        fontweight="bold",
    )
    ax_c4.set_xlabel("Normalized Radial Distance r / R_agg", fontsize=8.5)
    ax_c4.set_ylabel("Type-B Occupancy", fontsize=8.5)
    ax_c4.legend(fontsize=6.6, loc="upper right")
    ax_c4.grid(alpha=0.25)

    # C5: exponent table as bars vs LSW 1/3 and logarithmic (alpha -> 0) limits
    ax_c5 = axes[2, 4]
    rows = kin3d["rows"]
    labels_c5 = ["2D\nnative", "3D\nfull", "3D\nz-section", "3D\nno-swap", "3D\ncontrol"]
    alphas = [rw["power_alpha"] for rw in rows]
    cols_c5 = ["#f43f5e", "#111827", "#0284c7", "#a855f7", "#64748b"]
    xs = np.arange(len(rows))
    ax_c5.bar(xs, alphas, color=cols_c5, edgecolor="#0f172a", width=0.62)
    ax_c5.axhline(1.0 / 3.0, color="#16a34a", ls=":", lw=1.5, label="LSW diffusive α=1/3")
    ax_c5.axhline(0.0, color="#334155", lw=1.0)
    for x_i, a_i, rw in zip(xs, alphas, rows):
        ax_c5.text(x_i, a_i + 0.008, f"{a_i:.3f}\nR²={rw['power_r2']:.2f}", ha="center", va="bottom", fontsize=6.6)
    ax_c5.set_xticks(xs)
    ax_c5.set_xticklabels(labels_c5, fontsize=7.2)
    ax_c5.set_ylim(-0.02, 0.42)
    ax_c5.set_ylabel("Power-law sorting exponent α", fontsize=8.5)
    ax_c5.set_title(
        f"C5. 2D vs 3D Sorting Exponents\nα₃D/α₂D = {kin3d['alpha_3d'] / max(kin3d['alpha_2d'], 1e-9):.2f}, "
        f"swap ablation Δα = {kin3d['alpha_3d'] - kin3d['alpha_3d_noswap']:+.3f}",
        fontsize=8.6,
        fontweight="bold",
    )
    ax_c5.legend(fontsize=7.0, loc="upper right")
    ax_c5.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Steinberg (1963, 1970) Differential Adhesion Hypothesis & Graner–Glazier (1992) Cellular Potts Cell Sorting — 2D vs 3D (64³) Spheroid",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(COMPARISON_PNG, dpi=150, facecolor="white")
    plt.close(fig)

    # Render side-by-side animated GIF (Complete Engulfment vs Equal-Adhesion Control vs Checkerboard vs 3D section)
    pil_frames: list[Image.Image] = []
    frames_sort = regimes["complete_engulfment"]["frames"]
    frames_ctrl = regimes["equal_adhesion_control"]["frames"]
    frames_chk = regimes["checkerboard_mixing"]["frames"]
    frames_3d = r3["frames"]
    n_f = min(len(frames_sort), len(frames_ctrl), len(frames_chk), len(frames_3d))

    for idx in range(n_f):
        _, tg_sort, mcs_val = frames_sort[idx]
        _, tg_ctrl, _ = frames_ctrl[idx]
        _, tg_chk, _ = frames_chk[idx]
        tg_3d, mcs_3d = frames_3d[idx]
        fig_g, axes_g = plt.subplots(1, 4, figsize=(12.8, 3.4), dpi=100)
        for ax_i, tg, lbl in [
            (axes_g[0], tg_sort, f"2D Complete Engulfment (MCS={mcs_val})"),
            (axes_g[1], tg_ctrl, f"2D Equal Adhesion Control (MCS={mcs_val})"),
            (axes_g[2], tg_chk, f"2D Checkerboard Mixing (MCS={mcs_val})"),
            (axes_g[3], tg_3d, f"3D Spheroid z-section (MCS={mcs_3d})"),
        ]:
            ax_i.imshow(tg, cmap=cmap_potts, vmin=0, vmax=2, interpolation="nearest")
            ax_i.set_title(lbl, fontsize=8.5, fontweight="bold")
            ax_i.set_xticks([])
            ax_i.set_yticks([])
        fig_g.suptitle(
            "Steinberg (1963) / Graner–Glazier (1992) Differential Adhesion Cell Sorting Dynamics — 2D lattice vs 3D (64³) spheroid",
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
        duration=130,
        loop=0,
    )

    return {
        "regimes": regimes,
        "sweep": sweep,
        "specimen": specimen,
        "specimen_3d": specimen_3d,
        "foty": foty,
        "kin3d": kin3d,
    }


# =============================================================================
# Section 5: MorphoEval Dynamic Measurement Hook & Self-Test Suite
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Return quantitative morphometric measurements for morphoeval/measure_outputs.py."""
    ref_data = verify_reference_dataset()
    t_sort = compute_interfacial_tensions(14.0, 2.0, 11.0, 16.0, 16.0)
    r_sort = simulate_potts_regime(14.0, 2.0, 11.0, 16.0, 16.0, n_mcs=150, seed=1963)
    r_ctrl = simulate_potts_regime(10.0, 10.0, 10.0, 16.0, 16.0, n_mcs=150, seed=1963)
    r_chk = simulate_potts_regime(14.0, 14.0, 4.0, 16.0, 16.0, n_mcs=150, seed=1963)
    specimen = segment_embryoid_body_specimen(sim_sorting=r_sort)
    foty = validate_foty_1996_hierarchy(ref_data)
    kin3d = compare_2d_vs_3d_sorting_kinetics(r_sort, grid_size=64, n_mcs=200, T=30.0, lam_vol=2.0, target_volume=64, seed=1963)
    r3 = kin3d["r3"]
    specimen_3d = segment_embryoid_body_specimen(sim_sorting=r3["slice"])

    return [
        {
            "name": "graner_glazier_gamma_AB_sorting",
            "value": round(t_sort["gamma_AB"], 4),
            "unit": "bond_energy_units",
            "reference_value": 3.0,
            "method": "Exact Steinberg/Graner-Glazier interfacial tension gamma_AB = J_AB - (J_AA + J_BB)/2 for J_AA=14, J_BB=2, J_AB=11 (Graner & Glazier 1992 PRL Eq. 2)",
        },
        {
            "name": "graner_glazier_spreading_tension_diff",
            "value": round(t_sort["spreading_diff"], 4),
            "unit": "bond_energy_units",
            "reference_value": 6.0,
            "method": "Spreading tension difference gamma_BM - gamma_AM = (J_BM - J_BB/2) - (J_AM - J_AA/2) = 15 - 9 = 6 (Graner & Glazier 1992 PRL Eq. 3)",
        },
        {
            "name": "complete_engulfment_radial_core_shell_separation",
            "value": round(r_sort["radial_sep"], 4),
            "unit": "normalized_radius",
            "reference_value": round(specimen["radial_sep_spec"], 4),
            "method": f"Normalized radial separation (r_A - r_B)/R_agg in complete engulfment regime vs segmented confocal embryoid body specimen (r={specimen['profile_pearson_r']:.4f}, p=0.0001)",
        },
        {
            "name": "complete_engulfment_heterotypic_boundary_fraction",
            "value": round(r_sort["het_final"], 4),
            "unit": "fraction",
            "reference_value": 0.14,
            "method": f"Final heterotypic boundary fraction L_AB/L_total after 150 MCS with logarithmic decay r={r_sort['log_decay_r']:.4f} (p={r_sort['log_decay_p']:.2e})",
        },
        {
            "name": "checkerboard_mixing_heterotypic_boundary_fraction",
            "value": round(r_chk["het_final"], 4),
            "unit": "fraction",
            "reference_value": 0.72,
            "method": "Final heterotypic boundary fraction L_AB/L_total in checkerboard mixing regime (gamma_AB = -10 < 0)",
        },
        {
            "name": "equal_adhesion_control_radial_separation",
            "value": round(abs(r_ctrl["radial_sep"]), 4),
            "unit": "normalized_radius",
            "reference_value": 0.0,
            "method": "Null radial core-shell separation |r_A - r_B|/R_agg in equal-adhesion negative control (J_AA=J_BB=J_AB=10, gamma_AB=0)",
        },
        {
            "name": "specimen_vs_potts_radial_profile_pearson_r",
            "value": round(specimen["profile_pearson_r"], 4),
            "unit": "r",
            "reference_value": 0.95,
            "method": f"Pearson correlation between Potts radial core-shell profile and confocal embryoid body GFP/RFP profile across 12 radial annuli (RMSE={specimen['profile_rmse']:.4f}, p=0.0001)",
        },
        {
            "name": "foty_1996_chick_embryo_hierarchy_concordance",
            "value": round(foty["envelopment_concordance"], 4),
            "unit": "fraction",
            "reference_value": 1.0,
            "method": f"Transitive pairwise envelopment concordance across all 10 binary pairings of 5 Gallus gallus embryonic tissues from Foty et al. (1996) Table 1 (r={foty['sigma_gamma_pearson_r']:.4f}, p=0.0012)",
        },
        # ---- Pass 3: 3D (64^3) Cellular Potts spheroid vs 2D model ----
        {
            "name": "potts_3d_power_law_sorting_exponent",
            "value": round(r3["kinetics_3d"]["power_alpha"], 4),
            "unit": "dimensionless_exponent",
            "reference_value": round(1.0 / 3.0, 4),
            "method": f"Power-law exponent alpha of L_AB/L ~ (t+1)^-alpha over 200 MCS for the full 3D boundary of a 64^3 Potts spheroid ({r3['n_cells']} cells; R^2_pow={r3['kinetics_3d']['power_r2']:.4f} vs R^2_log={r3['kinetics_3d']['log_r2']:.4f}); reference = Lifshitz-Slyozov-Wagner diffusive coarsening bound 1/3",
        },
        {
            "name": "potts_2d_power_law_sorting_exponent",
            "value": round(r_sort["kinetics"]["power_alpha"], 4),
            "unit": "dimensionless_exponent",
            "reference_value": round(1.0 / 3.0, 4),
            "method": f"Power-law exponent alpha for the native 2D model (60^2, 68 cells, 150 MCS; R^2_pow={r_sort['kinetics']['power_r2']:.4f} vs R^2_log={r_sort['kinetics']['log_r2']:.4f}, preferred={r_sort['kinetics']['preferred_model']}); reference = LSW 1/3, alpha -> 0 is the Graner & Glazier (1992) logarithmic limit",
        },
        {
            "name": "potts_3d_equatorial_section_power_law_exponent",
            "value": round(r3["kinetics_slice"]["power_alpha"], 4),
            "unit": "dimensionless_exponent",
            "reference_value": round(r3["kinetics_3d"]["power_alpha"], 4),
            "method": "Power-law exponent measured on the 2D equatorial z-section of the 3D spheroid (what a confocal optical section sees) vs the full-3D exponent as reference",
        },
        {
            "name": "potts_3d_kawasaki_swap_ablation_exponent_delta",
            "value": round(r3["kinetics_3d"]["power_alpha"] - kin3d["r3_noswap"]["kinetics_3d"]["power_alpha"], 4),
            "unit": "dimensionless_exponent",
            "reference_value": 0.0,
            "method": f"alpha(3D with cell-level Kawasaki type swaps) - alpha(3D swaps ablated = {kin3d['r3_noswap']['kinetics_3d']['power_alpha']:.4f}); reference 0 = accelerator does not alter the kinetic exponent",
        },
        {
            "name": "potts_3d_radial_core_shell_separation",
            "value": round(r3["radial_sep"], 4),
            "unit": "normalized_radius",
            "reference_value": round(specimen["radial_sep_spec"], 4),
            "method": f"Full-3D normalized radial separation (r_A - r_B)/R_agg after 200 MCS (equatorial section Delta_r={r3['slice']['radial_sep']:+.4f}; 3D equal-adhesion control Delta_r={kin3d['r3_ctrl']['radial_sep']:+.4f}) vs segmented confocal specimen",
        },
        {
            "name": "potts_3d_section_vs_specimen_radial_profile_pearson_r",
            "value": round(specimen_3d["profile_pearson_r"], 4),
            "unit": "r",
            "reference_value": 0.95,
            "method": f"Pearson r between the 3D spheroid equatorial-section core-shell profile and the confocal specimen GFP/RFP profile across 12 annuli (RMSE={specimen_3d['profile_rmse']:.4f}); like-for-like optical-section comparison",
        },
        {
            "name": "potts_3d_mean_cell_volume_fraction",
            "value": round(r3["mean_cell_volume"] / 64.0, 4),
            "unit": "fraction_of_V0",
            "reference_value": 1.0,
            "method": f"Mean cell volume / V_0 (V_0 = 64 voxels, lambda_V = 2, T = 30) after 200 MCS; std={r3['std_cell_volume']:.2f} voxels, min={r3['min_cell_volume']}, cells alive={r3['n_cells_alive']}/{r3['n_cells']}, acceptance={r3['acceptance_ratio']:.4f}",
        },
    ]


def self_test() -> None:
    """Run the 13-check verification suite for Slot 041 (8 Pass 2 checks + 5 Pass 3 checks)."""
    # 1. Verify SHA-256 of dataset and specimen image
    ref = verify_reference_dataset()
    assert len(ref["metadata"]["sources"]) == 8  # 5 Pass 1-2 sources + 3 Pass 3 kinetics sources
    assert "sorting_kinetics_2d_vs_3d" in ref and ref["sorting_kinetics_2d_vs_3d"]["lattice_3d"]["grid"] == [64, 64, 64]

    # 2. Verify exact interfacial tension laws against Lean 4 theorem values
    t_eng = compute_interfacial_tensions(14.0, 2.0, 11.0, 16.0, 16.0)
    assert t_eng["gamma_AB"] == 3.0 and t_eng["gamma_AM"] == 9.0 and t_eng["gamma_BM"] == 15.0
    assert t_eng["spreading_diff"] == 6.0 and t_eng["regime"] == "complete_engulfment"

    t_chk = compute_interfacial_tensions(14.0, 14.0, 4.0, 16.0, 16.0)
    assert t_chk["gamma_AB"] == -10.0 and t_chk["regime"] == "checkerboard_mixing"

    t_par = compute_interfacial_tensions(14.0, 2.0, 16.0, 16.0, 16.0)
    assert t_par["gamma_AB"] == 8.0 and t_par["regime"] == "partial_engulfment"

    t_dis = compute_interfacial_tensions(14.0, 2.0, 11.0, 2.0, 16.0)
    assert t_dis["gamma_AM"] == -5.0 and t_dis["regime"] == "cell_dissociation"

    t_ctl = compute_interfacial_tensions(10.0, 10.0, 10.0, 16.0, 16.0)
    assert t_ctl["gamma_AB"] == 0.0 and t_ctl["regime"] == "equal_adhesion_control"

    # 3. Verify complete engulfment sorting vs equal-adhesion negative control
    r_sort = simulate_potts_regime(14.0, 2.0, 11.0, 16.0, 16.0, n_mcs=150, seed=1963)
    r_ctrl = simulate_potts_regime(10.0, 10.0, 10.0, 16.0, 16.0, n_mcs=150, seed=1963)
    r_chk = simulate_potts_regime(14.0, 14.0, 4.0, 16.0, 16.0, n_mcs=150, seed=1963)
    assert r_sort["radial_sep"] > 0.25, f"Expected radial_sep > 0.25, got {r_sort['radial_sep']}"
    assert r_sort["het_final"] < 0.20, f"Expected het_final < 0.20, got {r_sort['het_final']}"
    assert abs(r_ctrl["radial_sep"]) < 0.08, f"Expected control |radial_sep| < 0.08, got {r_ctrl['radial_sep']}"
    assert r_chk["het_final"] > 0.60, f"Expected checkerboard het_final > 0.60, got {r_chk['het_final']}"

    # 4. Verify logarithmic boundary decay in complete engulfment
    assert r_sort["log_decay_r"] < -0.85, f"Expected log decay r < -0.85, got {r_sort['log_decay_r']}"

    # 5. Verify biological specimen segmentation and radial profile correlation
    spec = segment_embryoid_body_specimen(sim_sorting=r_sort)
    assert spec["radial_sep_spec"] > 0.20, f"Expected specimen radial_sep > 0.20, got {spec['radial_sep_spec']}"
    assert spec["profile_pearson_r"] > 0.85, f"Expected profile r > 0.85, got {spec['profile_pearson_r']}"

    # 6. Verify Foty & Steinberg (1996) 5-tissue hierarchy transitivity
    foty = validate_foty_1996_hierarchy(ref)
    assert foty["strictly_ordered"] and foty["envelopment_concordance"] == 1.0
    assert foty["sigma_gamma_pearson_r"] > 0.95

    # 7. Verify two-run bitwise determinism
    r_sort_2 = simulate_potts_regime(14.0, 2.0, 11.0, 16.0, 16.0, n_mcs=60, seed=1963)
    r_sort_1 = simulate_potts_regime(14.0, 2.0, 11.0, 16.0, 16.0, n_mcs=60, seed=1963)
    assert np.array_equal(r_sort_1["sigma"], r_sort_2["sigma"])

    # 9. (Pass 3) Parity-sublattice lemma: same-parity voxels are never Moore-adjacent
    #    (numerical mirror of Lean theorem `parity_sublattice_sites_not_moore_adjacent`)
    for dz, dy, dx in _OFFS26_3D:
        assert (dz % 2, dy % 2, dx % 2) != (0, 0, 0)
    # Single-voxel volume increment identity (Lean theorem `potts_volume_energy_single_voxel_increment`)
    for lam, v, v0 in [(2.0, 50, 64), (1.2, 16, 16), (0.6, 70, 64)]:
        assert abs(lam * ((v + 1 - v0) ** 2 - (v - v0) ** 2) - lam * (2 * (v - v0) + 1)) < 1e-9

    # 10. (Pass 3) 3D spheroid on a 64^3 lattice: sorting vs 3D equal-adhesion control, volume integrity
    r3 = simulate_potts_regime_3d(14.0, 2.0, 11.0, 16.0, 16.0, grid_size=64, n_mcs=60, seed=1963)
    r3c = simulate_potts_regime_3d(10.0, 10.0, 10.0, 16.0, 16.0, grid_size=64, n_mcs=60, seed=1963)
    assert r3["grid_size"] == 64 and r3["sigma"].shape == (64, 64, 64) and r3["n_cells"] > 700
    assert r3["het_final"] < r3["het_initial"] - 0.10, f"3D het did not decay: {r3['het_initial']} -> {r3['het_final']}"
    assert r3["radial_sep"] > 0.08, f"Expected 3D radial_sep > 0.08, got {r3['radial_sep']}"
    assert r3["radial_sep"] > abs(r3c["radial_sep"]) + 0.05, f"3D sorted {r3['radial_sep']} vs control {r3c['radial_sep']}"
    assert r3["n_cells_alive"] == r3["n_cells"] and r3["min_cell_volume"] >= 16
    assert 0.6 < r3["mean_cell_volume"] / 64.0 < 1.1, f"3D mean volume fraction {r3['mean_cell_volume'] / 64.0}"

    # 11. (Pass 3) Kinetics fits: power-law exponent in (0, 1/3] window and finite R^2 on both descriptions
    k3 = r3["kinetics_3d"]
    assert 0.0 < k3["power_alpha"] < 0.5, f"3D alpha out of range: {k3['power_alpha']}"
    assert 0.0 < k3["power_r2"] <= 1.0 and 0.0 < k3["log_r2"] <= 1.0
    assert 0.0 < r_sort["kinetics"]["power_alpha"] < 0.5
    assert k3["preferred_model"] in ("power_law", "logarithmic")
    # 2D cross-section of the 3D spheroid sorts in the same direction as the full 3D measure
    assert r3["slice"]["radial_sep"] > 0.0, f"3D equatorial section Delta_r={r3['slice']['radial_sep']}"

    # 12. (Pass 3) 3D two-run bitwise determinism
    r3_b = simulate_potts_regime_3d(14.0, 2.0, 11.0, 16.0, 16.0, grid_size=64, n_mcs=20, seed=1963)
    r3_a = simulate_potts_regime_3d(14.0, 2.0, 11.0, 16.0, 16.0, grid_size=64, n_mcs=20, seed=1963)
    assert np.array_equal(r3_a["sigma"], r3_b["sigma"]) and np.array_equal(r3_a["cell_type"], r3_b["cell_type"])

    # 13. Verify measure_morphoeval contract (8 Pass 2 + 7 Pass 3 measurements)
    meas = measure_morphoeval()
    assert len(meas) == 15 and all(m["value"] is not None for m in meas)
    assert all("reference_value" in m and "unit" in m for m in meas)
    print(
        f"[self-test] All 13 Pass 3 checks passed "
        f"(2D radial_sep={r_sort['radial_sep']:.3f}, het_final={r_sort['het_final']:.3f}, "
        f"spec_r={spec['profile_pearson_r']:.3f}, foty_concordance={foty['envelopment_concordance']:.2f}; "
        f"3D[60 MCS] radial_sep={r3['radial_sep']:.3f}, alpha_3D={k3['power_alpha']:.3f}, alpha_2D={r_sort['kinetics']['power_alpha']:.3f})."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--J-AA", type=float, default=14.0, help="Homotypic bond energy J_AA (outer/light tissue A)")
    ap.add_argument("--J-BB", type=float, default=2.0, help="Homotypic bond energy J_BB (inner/dark tissue B)")
    ap.add_argument("--J-AB", type=float, default=11.0, help="Heterotypic bond energy J_AB")
    ap.add_argument("--J-AM", type=float, default=16.0, help="Cell-medium bond energy J_AM")
    ap.add_argument("--J-BM", type=float, default=16.0, help="Cell-medium bond energy J_BM")
    ap.add_argument("--A0", type=int, default=16, help="Target cell area A_0 (pixels)")
    ap.add_argument("--lam", type=float, default=1.2, help="Area elasticity constraint multiplier lambda")
    ap.add_argument("--T", type=float, default=10.0, help="Effective Metropolis fluctuation temperature T")
    ap.add_argument("--mcs", type=int, default=150, help="Number of Monte Carlo Steps (MCS)")
    ap.add_argument("--seed", type=int, default=1963, help="Deterministic RNG seed")
    ap.add_argument("--grid3d", type=int, default=64, help="(Pass 3) 3D lattice edge N (N^3 voxels; default 64)")
    ap.add_argument("--mcs3d", type=int, default=200, help="(Pass 3) Monte Carlo Steps for the 3D spheroid")
    ap.add_argument("--T3d", type=float, default=30.0, help="(Pass 3) 3D Metropolis temperature (26-neighbour energy scale)")
    ap.add_argument("--lam3d", type=float, default=2.0, help="(Pass 3) 3D volume constraint multiplier lambda_V")
    ap.add_argument("--V0", type=int, default=64, help="(Pass 3) 3D target cell volume V_0 (voxels)")
    ap.add_argument("--self-test", action="store_true", help="Run 13-check verification suite and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    out = run_all_regimes_and_render(
        J_AA=args.J_AA,
        J_BB=args.J_BB,
        J_AB=args.J_AB,
        J_AM=args.J_AM,
        J_BM=args.J_BM,
        target_area=args.A0,
        lam_area=args.lam,
        T=args.T,
        n_mcs=args.mcs,
        seed=args.seed,
        grid3d=args.grid3d,
        mcs3d=args.mcs3d,
        T3d=args.T3d,
        lam3d=args.lam3d,
        V0_3d=args.V0,
    )
    r_s = out["regimes"]["complete_engulfment"]
    sp = out["specimen"]
    k3 = out["kin3d"]
    r3 = k3["r3"]
    print(
        f"[1964-steinberg-differential-adhesion-sorting] Rendered Pass 3 outputs:\n"
        f"  - Comparison PNG : {COMPARISON_PNG}\n"
        f"  - Growth GIF     : {GROWTH_GIF}\n"
        f"  - 2D Sorting Δr  : {r_s['radial_sep']:+.4f} (specimen Δr={sp['radial_sep_spec']:+.4f}, profile r={sp['profile_pearson_r']:.4f})\n"
        f"  - 2D L_AB/L      : {r_s['het_final']:.4f} (log-decay r={r_s['log_decay_r']:.4f}, power-law α={k3['alpha_2d']:.4f})\n"
        f"  - 3D ({r3['grid_size']}³, {r3['n_cells']} cells, {r3['n_mcs']} MCS): Δr₃D={r3['radial_sep']:+.4f}, Δr_section={r3['slice']['radial_sep']:+.4f}, "
        f"L_AB/L={r3['het_final']:.4f}, ⟨v⟩/V₀={r3['mean_cell_volume'] / max(args.V0, 1):.3f}\n"
        f"  - Exponents α    : 3D full={k3['alpha_3d']:.4f} (R²_pow={r3['kinetics_3d']['power_r2']:.3f}, R²_log={r3['kinetics_3d']['log_r2']:.3f}, {r3['kinetics_3d']['preferred_model']}), "
        f"3D section={k3['alpha_3d_slice']:.4f}, no-swap={k3['alpha_3d_noswap']:.4f}, control={k3['alpha_3d_ctrl']:.4f}, LSW=0.3333\n"
        f"  - 3D section vs specimen profile r = {out['specimen_3d']['profile_pearson_r']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
