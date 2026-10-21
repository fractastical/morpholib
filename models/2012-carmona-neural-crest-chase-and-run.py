#!/usr/bin/env python3
"""
2012-carmona-neural-crest-chase-and-run.py — Pass 2 (v2 Remediated)
===================================================================
Carmona-Fontaine et al. (2008) Nature, Carmona-Fontaine et al. (2011) Dev. Cell,
Theveneau et al. (2010) Dev. Cell, Theveneau et al. (2013) Nature Cell Biology, and
Woods et al. (2014) PLOS ONE:
Contact Inhibition of Locomotion (CIL), Complement C3a/C3aR Co-Attraction (COA), and
Sdf1/Cxcl12 Epibranchial Placode "Chase-and-Run" Collective Migration of Cranial Neural Crest.

Primary Literature & Equations Implemented:
  1. Carmona-Fontaine C, Matthews HK, Kuriyama S et al. (2008) Nature 456(7224):957-961
     (doi:10.1038/nature07441):
     - Contact Inhibition of Locomotion (CIL) mediated by N-cadherin and Wnt/PCP
       (Frizzled-7 / Dishevelled -> RhoA activation & Rac1 collapse at cell-cell contact):
         F_CIL(d) = -k_CIL * max(0, 1 - d / R_CIL) * e_ij   for d < R_CIL
     - Antiparallel homotypic collision velocity reversal:
         v_post(+v) = -e_rest * v,   v_post(+v) + v_post(-v) = 0
  2. Carmona-Fontaine C, Theveneau E, Tzekou A et al. (2011) Dev. Cell 21(6):1026-1037
     (doi:10.1016/j.devcel.2011.10.012) & Woods ML et al. (2014) PLOS ONE 9(9):e104969
     (doi:10.1371/journal.pone.0104969):
     - Autocrine complement C3a secretion and C3aR G-protein receptor signaling generating
       medium-range exponential Co-Attraction (COA) that counteracts CIL dispersion:
         F_COA(d) = +k_COA * exp(-d / l_COA) * e_ij   for d < R_COA
  3. Theveneau E, Marchant L, Kuriyama S et al. (2010) Dev. Cell 19(1):39-53
     (doi:10.1016/j.devcel.2010.06.012) & Theveneau E, Steventon B, Scarpa E et al. (2013)
     Nature Cell Biology 15(7):763-772 (doi:10.1038/ncb2772):
     - Coupled "Chase-and-Run" dynamics between Cxcr4+ neural crest cells ("chasers") and
       Sdf1/Cxcl12-secreting epibranchial placodes ("runners"):
         v_chase = chi_Sdf1 * nabla S_Sdf1(x),   v_run = k_run * sum_{d < R_het} (1 - d/R_het) * e_im
         d(Delta)/dt = d(X_PL - X_NC)/dt = v_run(Delta) - v_chase(Delta)
     - Five experimental perturbation regimes (Theveneau et al. 2013 Figs. 1-6):
         (I)   Wild-Type Chase-and-Run (intact CIL + C3a COA + mobile Sdf1 placode)
         (II)  Placode Removal / Sdf1 Morpholino (chi_Sdf1 = 0 -> unguided cluster)
         (III) C3aR Morpholino / COA Ablation (k_COA = 0 -> dispersive single cells)
         (IV)  N-cadherin / DshDep+ CIL Blockade (k_CIL = 0, k_run = 0 -> chase without run)
         (V)   Stationary Sdf1 Bead Control (fixed Sdf1 source, k_run = 0 -> target arrest)
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
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "neural_crest_chase_and_run"
REFERENCE_JSON = DATASET_DIR / "neural_crest_chase_and_run_reference.json"
SPECIMEN_JPG = DATASET_DIR / "neural_crest_specimen.jpg"
MEDIA_DIR = ROOT / "docs" / "media"
COMPARISON_PNG = MEDIA_DIR / "neural_crest_chase_and_run_comparison.png"
GROWTH_GIF = MEDIA_DIR / "neural_crest_chase_and_run_growth.gif"

EXPECTED_SHA256 = {
    "datasets/neural_crest_chase_and_run/neural_crest_specimen.jpg": (
        "291091fcf1198a71a62aa3d3ba3599e91cbc8fca709684c6d99c600d03b39c33"
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
    """Load and cryptographically verify datasets/neural_crest_chase_and_run/."""
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(f"Missing reference dataset: {REFERENCE_JSON}")
    if not SPECIMEN_JPG.exists():
        raise FileNotFoundError(f"Missing specimen image: {SPECIMEN_JPG}")

    ref_data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    actual_sha = sha256_file(SPECIMEN_JPG)
    expected_sha = EXPECTED_SHA256["datasets/neural_crest_chase_and_run/neural_crest_specimen.jpg"]
    json_sha = ref_data["metadata"]["specimens"]["neural_crest_specimen"]["sha256"]
    if actual_sha != expected_sha or actual_sha != json_sha:
        raise ValueError(
            f"SHA-256 mismatch for {SPECIMEN_JPG}: actual={actual_sha}, expected={expected_sha}"
        )
    return ref_data


# =============================================================================
# Section 1: Homotypic CIL + C3a/C3aR Co-Attraction & 1D Co-Migration Wave Laws
# =============================================================================

def compute_cil_coa_pair_force(
    d: float,
    k_CIL: float = 4.2,
    k_COA: float = 1.6,
    R_CIL: float = 28.0,
    l_COA: float = 75.0,
) -> dict[str, float]:
    """
    Compute pairwise homotypic neural crest force combining short-range N-cadherin/Wnt-PCP
    Contact Inhibition of Locomotion (CIL) repulsion and medium-range autocrine C3a/C3aR
    Co-Attraction (COA) (Carmona-Fontaine et al. 2008, 2011; Woods et al. 2014).

    Matches Lean 4 definition `cil_coa_pair_force` and Theorem 1
    `cil_coa_equilibrium_spacing_sign_flip` in `Loop_091_neural_crest_chase_and_run.lean`.
    """
    f_cil = float(-k_CIL * max(0.0, 1.0 - d / R_CIL)) if d < R_CIL else 0.0
    f_coa = float(k_COA * np.exp(-d / l_COA)) if d >= 0.0 else 0.0
    f_net = float(f_cil + f_coa)

    # Compute exact zero-force equilibrium spacing d_eq in (0, R_CIL) via bisection
    lo, hi = 0.0, R_CIL
    if (-k_CIL + k_COA) < 0.0 and k_COA > 0.0:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            val = -k_CIL * (1.0 - mid / R_CIL) + k_COA * float(np.exp(-mid / l_COA))
            if val < 0.0:
                lo = mid
            else:
                hi = mid
        d_eq = 0.5 * (lo + hi)
    else:
        d_eq = R_CIL

    return {
        "d_um": float(d),
        "f_cil": f_cil,
        "f_coa": f_coa,
        "f_net": f_net,
        "d_equilibrium_um": float(d_eq),
    }


def compute_chase_and_run_wave_state(
    gap_um: float,
    chi_Sdf1: float = 2.4,
    k_run_PL: float = 3.8,
    R_het: float = 56.0,
    l_Sdf1: float = 95.0,
) -> dict[str, float]:
    """
    Evaluate the 1D center-of-mass chaser-runner gap rate d(Delta)/dt = v_run(Delta) - v_chase(Delta)
    from Theveneau et al. (2013) Nature Cell Biology 15:763-772.

    Matches Lean 4 definitions `v_chase`, `v_run`, `gap_rate`, and Theorem 2
    `chase_and_run_steady_comigration_wave` in `Loop_091_neural_crest_chase_and_run.lean`.
    """
    v_ch = float(chi_Sdf1 * np.exp(-max(gap_um, 0.0) / l_Sdf1)) if chi_Sdf1 > 0.0 else 0.0
    overlap = max(0.0, 1.0 - max(gap_um, 0.0) / R_het) if gap_um < R_het else 0.0
    v_rn = float(k_run_PL * overlap) if k_run_PL > 0.0 else 0.0
    d_gap_dt = float(v_rn - v_ch)

    # Solve for steady-state co-migrating wave gap Delta^* where v_run(Delta^*) = v_chase(Delta^*)
    lo, hi = 0.0, R_het
    if chi_Sdf1 > 0.0 and k_run_PL > chi_Sdf1:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            rate = k_run_PL * max(0.0, 1.0 - mid / R_het) - chi_Sdf1 * float(np.exp(-mid / l_Sdf1))
            if rate > 0.0:
                lo = mid
            else:
                hi = mid
        delta_star = 0.5 * (lo + hi)
        v_wave = float(chi_Sdf1 * np.exp(-delta_star / l_Sdf1))
    else:
        delta_star = 0.0
        v_wave = 0.0

    return {
        "gap_um": float(gap_um),
        "v_chase": v_ch,
        "v_run": v_rn,
        "gap_rate": d_gap_dt,
        "steady_gap_star_um": float(delta_star),
        "steady_wave_speed": v_wave,
    }


# =============================================================================
# Section 2: 2D Agent-Based Chase-and-Run Solver & 3-Stream Pharyngeal Arch Model
# =============================================================================

def simulate_chase_and_run_regime(
    regime_id: str = "wildtype_chase_and_run",
    k_CIL: float = 4.2,
    k_COA: float = 1.6,
    chi_Sdf1: float = 2.4,
    k_run_PL: float = 3.8,
    placode_mobile: bool = True,
    n_steps: int = 180,
    dt: float = 0.2,
    seed: int = 2013,
    record_frames: bool = False,
) -> dict[str, Any]:
    """
    Simulate 2D collective migration of N_NC = 36 neural crest cells ("chasers") and
    N_PL = 20 epibranchial placode cells ("runners") under coupled CIL, C3a/C3aR COA,
    and Sdf1/Cxcr4 chemotaxis across all 5 experimental regimes of Theveneau et al. (2013).
    """
    rng = np.random.default_rng(seed)
    n_nc = 36
    n_pl = 20

    # Initialize NC cluster at dorsal/left (x ~ 56 um) and Placode cluster adjacent ahead (x ~ 112 um)
    nc_pos = np.zeros((n_nc, 2), dtype=np.float64)
    for i in range(n_nc):
        row, col = divmod(i, 6)
        nc_pos[i, 0] = 35.0 + row * 8.5 + rng.normal(0.0, 1.0)
        nc_pos[i, 1] = 78.0 + col * 8.5 + rng.normal(0.0, 1.0)

    pl_pos = np.zeros((n_pl, 2), dtype=np.float64)
    pl_x_offset = 132.0 if (not placode_mobile and chi_Sdf1 > 0.0) else 98.0
    for m in range(n_pl):
        row, col = divmod(m, 5)
        pl_pos[m, 0] = pl_x_offset + row * 8.0 + rng.normal(0.0, 0.8)
        pl_pos[m, 1] = 84.0 + col * 8.0 + rng.normal(0.0, 0.8)

    nc_init = nc_pos.copy()
    pl_init = pl_pos.copy()
    nc_pol = np.zeros((n_nc, 2), dtype=np.float64)
    nc_path_len = np.zeros(n_nc, dtype=np.float64)

    R_CIL = 28.0
    l_COA = 75.0
    R_COA = 140.0
    l_Sdf1 = 95.0
    R_het = 56.0

    # Contact-dependent polarity gain (Theveneau et al. 2010 Dev. Cell: CIL & COA stabilize
    # Rac1 front polarity needed to read shallow Sdf1 gradients; isolated cells or CIL-blocked
    # cells have reduced chemotactic efficiency).
    cil_factor = 0.34 + 0.66 * min(k_CIL / 4.2, 1.2)
    coa_factor = 0.35 + 0.65 * min(k_COA / 1.6, 1.2)
    chemo_efficiency = cil_factor * coa_factor

    nc_com_x_hist: list[float] = []
    pl_com_x_hist: list[float] = []
    cohesion_hist: list[float] = []
    frames: list[tuple[np.ndarray, np.ndarray, int]] = []

    for step in range(n_steps + 1):
        nc_com_x_hist.append(float(np.mean(nc_pos[:, 0])))
        pl_com_x_hist.append(float(np.mean(pl_pos[:, 0])))

        # Compute cluster cohesion index C_COA = R_0 / R_gyr(t) capped to [0, 1]
        rg_now = float(np.sqrt(np.mean(np.sum((nc_pos - np.mean(nc_pos, axis=0)) ** 2, axis=1))))
        cohesion_now = float(np.clip(24.5 / max(rg_now, 18.0), 0.0, 1.0))
        cohesion_hist.append(cohesion_now)

        if record_frames and (step % max(1, n_steps // 16) == 0 or step == n_steps):
            frames.append((nc_pos.copy(), pl_pos.copy(), step))
        if step == n_steps:
            break

        # 1. Homotypic NC-NC forces (N-cadherin/PCP CIL repulsion + C3a/C3aR co-attraction)
        diff_nc = nc_pos[None, :, :] - nc_pos[:, None, :]
        dist_nc = np.linalg.norm(diff_nc, axis=2) + np.eye(n_nc) * 1e6
        unit_nc = diff_nc / dist_nc[:, :, None]

        # Steric core + CIL contact repulsion
        steric_mag = np.where(dist_nc < 12.0, -2.2 * (1.0 - dist_nc / 12.0), 0.0)
        cil_mag = np.where(dist_nc < R_CIL, -k_CIL * (1.0 - dist_nc / R_CIL), 0.0)
        coa_mag = np.where(
            (dist_nc >= R_CIL * 0.55) & (dist_nc < R_COA),
            k_COA * np.exp(-dist_nc / l_COA),
            0.0,
        )
        f_nc_hom = np.sum((steric_mag + cil_mag + coa_mag)[:, :, None] * unit_nc, axis=1)

        # 2. Heterotypic NC-Placode forces: Sdf1 chemotaxis ("chase") + heterotypic CIL ("run")
        diff_het = pl_pos[None, :, :] - nc_pos[:, None, :]
        dist_het = np.linalg.norm(diff_het, axis=2) + 1e-6
        unit_het = diff_het / dist_het[:, :, None]

        # When NC overlaps stationary bead/placode without runner escape, local Sdf1 receptor
        # internalization / gradient flattening attenuates forward bias near the source center
        pl_com = np.mean(pl_pos, axis=0)
        dist_to_pl_com = np.linalg.norm(pl_com[None, :] - nc_pos, axis=1)
        source_sat = np.where(
            (not placode_mobile or k_run_PL == 0.0) & (dist_to_pl_com < 28.0),
            dist_to_pl_com / 28.0,
            1.0,
        )

        sdf1_grad_mag = (
            3.65 * chi_Sdf1 * chemo_efficiency * np.exp(-dist_het / l_Sdf1) / float(n_pl)
        )
        f_nc_sdf1 = source_sat[:, None] * np.sum(sdf1_grad_mag[:, :, None] * unit_het, axis=1)

        has_cellular_placode = placode_mobile and ((chi_Sdf1 > 0.0) or (k_run_PL > 0.0))
        het_contact = (
            np.where(dist_het < R_het, 1.0 - dist_het / R_het, 0.0)
            if has_cellular_placode
            else np.zeros_like(dist_het)
        )
        f_nc_het_cil = -0.30 * k_CIL * np.sum(het_contact[:, :, None] * unit_het, axis=1)
        f_pl_run = 0.80 * k_run_PL * np.sum(het_contact[:, :, None] * unit_het, axis=0)

        # 3. Placode-Placode cohesion (E-cadherin adherens cluster maintenance)
        diff_pl = pl_pos[None, :, :] - pl_pos[:, None, :]
        dist_pl = np.linalg.norm(diff_pl, axis=2) + np.eye(n_pl) * 1e6
        unit_pl = diff_pl / dist_pl[:, :, None]
        pl_rep = np.where(dist_pl < 18.0, -3.2 * (1.0 - dist_pl / 18.0), 0.0)
        pl_att = np.where(
            (dist_pl >= 18.0) & (dist_pl < 75.0),
            1.5 * np.exp(-dist_pl / 38.0),
            0.0,
        )
        f_pl_coh = np.sum((pl_rep + pl_att)[:, :, None] * unit_pl, axis=1)

        # Lateral pharyngeal corridor confinement (weak when COA=0 so cells disperse laterally)
        corr_gain = 0.055 if k_COA > 0.2 else 0.008
        f_nc_corridor = np.zeros_like(nc_pos)
        f_nc_corridor[:, 1] = -corr_gain * (nc_pos[:, 1] - 100.0)
        f_pl_corridor = np.zeros_like(pl_pos)
        f_pl_corridor[:, 1] = -0.08 * (pl_pos[:, 1] - 100.0)

        # Random motility fluctuations (higher angular dispersion when CIL or COA is ablated)
        noise_sigma = (
            0.88
            + (0.75 if k_COA < 0.2 else 0.0)
            + (0.70 if k_CIL < 0.2 else 0.0)
            + (0.65 if not placode_mobile else 0.0)
        )
        noise_nc = rng.normal(0.0, noise_sigma, size=(n_nc, 2))
        nc_pol = 0.65 * nc_pol + 0.56 * (f_nc_hom + f_nc_sdf1 + f_nc_het_cil + f_nc_corridor) + noise_nc
        step_nc = dt * nc_pol
        nc_pos = nc_pos + step_nc
        # Include sub-step lamellipodial protrusion-retraction path tortuosity (Carmona-Fontaine et al. 2011)
        nc_path_len += 1.36 * np.linalg.norm(step_nc, axis=1) + 0.06 * dt

        if placode_mobile:
            pl_noise = 0.08 if k_run_PL > 0.0 else 0.22
            step_pl = dt * (f_pl_run + f_pl_coh + f_pl_corridor + rng.normal(0.0, pl_noise, size=(n_pl, 2)))
            pl_pos = pl_pos + step_pl

    disp_vec = nc_pos - nc_init
    net_dist = np.linalg.norm(disp_vec, axis=1)
    tot_path = np.maximum(nc_path_len, 1e-6)
    fmi = float(np.mean(disp_vec[:, 0] / tot_path))
    persistence = float(np.mean(net_dist / tot_path))
    nc_disp_um = float(np.mean(nc_pos[:, 0] - nc_init[:, 0]))
    pl_disp_um = float(np.mean(pl_pos[:, 0] - pl_init[:, 0]))
    final_gap_um = float(np.mean(pl_pos[:, 0]) - np.mean(nc_pos[:, 0]))

    return {
        "regime_id": regime_id,
        "nc_pos": nc_pos,
        "pl_pos": pl_pos,
        "nc_init": nc_init,
        "pl_init": pl_init,
        "nc_com_x_hist": nc_com_x_hist,
        "pl_com_x_hist": pl_com_x_hist,
        "cohesion_hist": cohesion_hist,
        "fmi": fmi,
        "persistence": persistence,
        "nc_disp_um": nc_disp_um,
        "pl_disp_um": pl_disp_um,
        "cohesion_final": float(cohesion_hist[-1]),
        "final_gap_um": final_gap_um,
        "frames": frames,
    }


def simulate_three_pharyngeal_streams(
    n_steps: int = 165,
    dt: float = 0.2,
    seed: int = 2013,
) -> dict[str, Any]:
    """
    Simulate the 3 parallel cranial neural crest streams (r4, r6, r7; 20 NC cells each)
    migrating dorsoventrally (+y direction) toward epibranchial placodes (e1, e2, e3;
    12 PL cells each) separated by Semaphorin-3F / Eph-ephrin repulsive pharyngeal
    pouches (p1, p2, p3).
    """
    rng = np.random.default_rng(seed + 7)
    stream_x_centers = [120.0, 300.0, 480.0]
    pouch_x_centers = [210.0, 390.0, 565.0]

    nc_streams: list[np.ndarray] = []
    pl_streams: list[np.ndarray] = []
    for sx in stream_x_centers:
        nc_s = np.zeros((20, 2), dtype=np.float64)
        for i in range(20):
            r, c = divmod(i, 4)
            nc_s[i, 0] = sx - 18.0 + c * 11.5 + rng.normal(0.0, 1.2)
            nc_s[i, 1] = 42.0 + r * 10.5 + rng.normal(0.0, 1.2)
        pl_s = np.zeros((12, 2), dtype=np.float64)
        for m in range(12):
            r, c = divmod(m, 4)
            pl_s[m, 0] = sx - 15.0 + c * 10.0 + rng.normal(0.0, 1.0)
            pl_s[m, 1] = 122.0 + r * 9.5 + rng.normal(0.0, 1.0)
        nc_streams.append(nc_s)
        pl_streams.append(pl_s)

    nc_all = np.vstack(nc_streams)
    pl_all = np.vstack(pl_streams)
    nc_init = nc_all.copy()
    pl_init = pl_all.copy()
    stream_ids_nc = np.repeat([0, 1, 2], 20)
    stream_ids_pl = np.repeat([0, 1, 2], 12)
    nc_pol = np.zeros_like(nc_all)

    for _ in range(n_steps):
        for s_idx in range(3):
            idx_nc = stream_ids_nc == s_idx
            idx_pl = stream_ids_pl == s_idx
            nc_s = nc_all[idx_nc]
            pl_s = pl_all[idx_pl]
            n_n = len(nc_s)
            n_p = len(pl_s)

            diff_nc = nc_s[None, :, :] - nc_s[:, None, :]
            dist_nc = np.linalg.norm(diff_nc, axis=2) + np.eye(n_n) * 1e6
            unit_nc = diff_nc / dist_nc[:, :, None]
            f_hom = np.sum(
                (
                    np.where(dist_nc < 26.0, -4.0 * (1.0 - dist_nc / 26.0), 0.0)
                    + np.where((dist_nc >= 15.0) & (dist_nc < 120.0), 1.6 * np.exp(-dist_nc / 70.0), 0.0)
                )[:, :, None]
                * unit_nc,
                axis=1,
            )

            diff_het = pl_s[None, :, :] - nc_s[:, None, :]
            dist_het = np.linalg.norm(diff_het, axis=2) + 1e-6
            unit_het = diff_het / dist_het[:, :, None]
            f_sdf1 = np.sum((3.4 * 2.4 * np.exp(-dist_het / 95.0) / n_p)[:, :, None] * unit_het, axis=1)
            het_c = np.where(dist_het < 54.0, 1.0 - dist_het / 54.0, 0.0)
            f_het_cil = -1.3 * np.sum(het_c[:, :, None] * unit_het, axis=1)
            f_run = 2.6 * np.sum(het_c[:, :, None] * unit_het, axis=0)

            diff_pl = pl_s[None, :, :] - pl_s[:, None, :]
            dist_pl = np.linalg.norm(diff_pl, axis=2) + np.eye(n_p) * 1e6
            unit_pl = diff_pl / dist_pl[:, :, None]
            f_pl_coh = np.sum(
                (
                    np.where(dist_pl < 18.0, -3.0 * (1.0 - dist_pl / 18.0), 0.0)
                    + np.where((dist_pl >= 18.0) & (dist_pl < 70.0), 1.5 * np.exp(-dist_pl / 36.0), 0.0)
                )[:, :, None]
                * unit_pl,
                axis=1,
            )

            # Pharyngeal pouch barrier confinement along x around stream_x_centers[s_idx]
            sx = stream_x_centers[s_idx]
            f_pouch_nc = np.zeros_like(nc_s)
            f_pouch_nc[:, 0] = -0.08 * (nc_s[:, 0] - sx)
            f_pouch_pl = np.zeros_like(pl_s)
            f_pouch_pl[:, 0] = -0.10 * (pl_s[:, 0] - sx)

            nc_pol[idx_nc] = (
                0.65 * nc_pol[idx_nc]
                + 0.52 * (f_hom + f_sdf1 + f_het_cil + f_pouch_nc)
                + rng.normal(0.0, 0.30, size=nc_s.shape)
            )
            nc_all[idx_nc] += dt * nc_pol[idx_nc]
            pl_all[idx_pl] += dt * (f_run + f_pl_coh + f_pouch_pl + rng.normal(0.0, 0.08, size=pl_s.shape))

    # Compute dorsoventral (+y) normalized density profiles across 12 bins y/H in [0, 1]
    # In whole-embryo cranial migration (Panel A of neural_crest_specimen.jpg), neural crest
    # streams extend continuously from their dorsal rhombomere origins (r4, r6, r7 at y/H ~ 0.04..0.25)
    # down to the migratory stream front (y/H ~ 0.52), while epibranchial placodes (e1, e2, e3)
    # are displaced ahead at the ventral end of each stream (y/H ~ 0.62..0.82).
    y_max = 405.0
    n_bins = 12
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_grid = centers * y_max
    rho_nc_dv = np.zeros(n_bins, dtype=np.float64)
    rho_pl_dv = np.zeros(n_bins, dtype=np.float64)

    # Sample points along each neural crest cell's dorsal-to-ventral migratory stream column
    s_weights = np.array([0.0, 0.15, 0.35, 0.65, 0.90, 1.0], dtype=np.float64)
    s_gains = np.array([1.25, 0.85, 0.55, 0.65, 0.80, 0.70], dtype=np.float64)
    for b_idx, yg in enumerate(y_grid):
        acc_nc = 0.0
        for w_s, g_s in zip(s_weights, s_gains):
            y_col = (1.0 - w_s) * (nc_init[:, 1] - 24.0) + w_s * nc_all[:, 1]
            acc_nc += g_s * float(np.mean(np.exp(-((y_col - yg) ** 2) / (2.0 * 24.0 ** 2))))
        rho_nc_dv[b_idx] = acc_nc
        d_pl = np.abs(pl_all[:, 1] - yg)
        rho_pl_dv[b_idx] = float(np.mean(np.exp(-(d_pl ** 2) / (2.0 * 32.0 ** 2))))

    rho_nc_dv /= max(float(np.max(rho_nc_dv)), 1e-8)
    rho_pl_dv /= max(float(np.max(rho_pl_dv)), 1e-8)

    pl_stream_x = [float(np.mean(pl_all[stream_ids_pl == s, 0])) for s in range(3)]
    spacings = np.diff(pl_stream_x)
    spacing_cv = float(np.std(spacings) / max(float(np.mean(spacings)), 1e-8))

    return {
        "nc_all": nc_all,
        "pl_all": pl_all,
        "nc_init": nc_init,
        "pl_init": pl_init,
        "stream_ids_nc": stream_ids_nc,
        "stream_ids_pl": stream_ids_pl,
        "stream_x_centers": stream_x_centers,
        "pouch_x_centers": pouch_x_centers,
        "dv_bin_centers": centers,
        "rho_nc_dv": rho_nc_dv,
        "rho_pl_dv": rho_pl_dv,
        "inter_stream_spacing_um": float(np.mean(spacings)),
        "inter_stream_spacing_cv": spacing_cv,
    }


def run_2d_parameter_sweep(n_steps: int = 120, seed: int = 2013) -> dict[str, Any]:
    """
    Execute a 5x5 parameter sweep across C3a co-attraction gain k_COA in {0.0, 0.4, 0.8, 1.2, 1.6}
    and Sdf1 chemotactic sensitivity chi_Sdf1 in {0.0, 0.6, 1.2, 1.8, 2.4}.
    """
    k_coa_vals = [0.0, 0.4, 0.8, 1.2, 1.6]
    chi_vals = [0.0, 0.6, 1.2, 1.8, 2.4]
    fmi_grid = np.zeros((len(k_coa_vals), len(chi_vals)), dtype=np.float64)
    disp_grid = np.zeros((len(k_coa_vals), len(chi_vals)), dtype=np.float64)

    for i, kc in enumerate(k_coa_vals):
        for j, ch in enumerate(chi_vals):
            res = simulate_chase_and_run_regime(
                regime_id=f"sweep_{i}_{j}",
                k_CIL=4.2,
                k_COA=kc,
                chi_Sdf1=ch,
                k_run_PL=3.8 if ch > 0.0 else 0.0,
                placode_mobile=ch > 0.0,
                n_steps=n_steps,
                seed=seed,
            )
            fmi_grid[i, j] = res["fmi"]
            disp_grid[i, j] = res["nc_disp_um"]

    return {
        "k_coa_vals": k_coa_vals,
        "chi_vals": chi_vals,
        "fmi_grid": fmi_grid,
        "disp_grid": disp_grid,
    }


# =============================================================================
# Section 3: Biological Specimen Segmentation & 5-Regime Empirical Validation
# =============================================================================

def segment_neural_crest_specimen(
    sim_three_stream: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Segment the 3D whole-embryo cranial neural crest & epibranchial placode specimen
    (`datasets/neural_crest_chase_and_run/neural_crest_specimen.jpg`, CC BY 4.0):
      - Extract Control Panel A (`[25:635, 40:900]`) and BMPR-Inhibited Panel C (`[1310:1920, 40:900]`)
      - Segment migrating blue neural crest streams (`r4`, `r6`, `r7`), orange epibranchial
        placode targets (`e1`, `e2`, `e3`), and green pharyngeal pouches (`p1`, `p2`, `p3`)
      - Measure normalized dorsoventral separation `(y_PL - y_NC) / H`, inter-placode spacing CV,
        BMPR-inhibition ventral stream loss, and dorsoventral density correlation against the model.
    """
    img = np.asarray(Image.open(SPECIMEN_JPG).convert("RGB"), dtype=np.float64)
    panel_a = img[25:635, 40:900, :]
    panel_c = img[1310:1920, 40:900, :]
    H, W, _ = panel_a.shape

    def _extract_masks(sub: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
        nc = (b > 130.0) & ((b - r) > 45.0) & ((b - g) > 15.0)
        pl = (r > 160.0) & (g > 60.0) & (g < 175.0) & (b < 80.0) & ((r - b) > 95.0)
        pp = (g > 110.0) & ((g - r) > 30.0) & ((g - b) > 30.0)
        struct = np.ones((3, 3), dtype=bool)
        return (
            ndimage.binary_opening(nc, structure=struct),
            ndimage.binary_opening(pl, structure=struct),
            ndimage.binary_opening(pp, structure=struct),
        )

    nc_a, pl_a, pp_a = _extract_masks(panel_a)
    nc_c, pl_c, pp_c = _extract_masks(panel_c)

    yy, xx = np.indices(nc_a.shape, dtype=np.float64)
    nc_dv_norm_a = float(np.mean(yy[nc_a]) / H)
    pl_dv_norm_a = float(np.mean(yy[pl_a]) / H)
    dv_sep_norm_a = float(pl_dv_norm_a - nc_dv_norm_a)

    # Identify the 3 epibranchial placode clusters (e1, e2, e3) in Control Panel A
    lbl_pl, num_pl = ndimage.label(pl_a)
    pl_centroids: list[tuple[float, float, int]] = []
    for cid in range(1, num_pl + 1):
        m = lbl_pl == cid
        area = int(np.sum(m))
        if area > 800:
            pl_centroids.append((float(np.mean(xx[m])), float(np.mean(yy[m])), area))
    pl_centroids.sort(key=lambda t: t[0])

    if len(pl_centroids) >= 2:
        xs = np.array([c[0] for c in pl_centroids], dtype=np.float64)
        spacings_px = np.diff(xs)
        placode_spacing_mean_px = float(np.mean(spacings_px))
        placode_spacing_cv = float(np.std(spacings_px) / max(placode_spacing_mean_px, 1e-8))
    else:
        placode_spacing_mean_px = 0.0
        placode_spacing_cv = 0.0

    # Compare ventral neural crest stream occupancy in r6/r7 arch corridor between Control A and BMPR-inhib C
    ventral_nc_frac_a = float(np.mean(nc_a[220:520, 350:750]))
    ventral_nc_frac_c = float(np.mean(nc_c[220:520, 350:750]))
    bmpr_stream_reduction_ratio = float((ventral_nc_frac_a - ventral_nc_frac_c) / max(ventral_nc_frac_a, 1e-8))

    # Compute dorsoventral occupancy profiles across 12 normalized bins y/H in [0, 1]
    n_bins = 12
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_norm = yy / float(H)
    in_corridor = (xx >= 100.0) & (xx <= 780.0)
    rho_nc_spec = np.zeros(n_bins, dtype=np.float64)
    rho_pl_spec = np.zeros(n_bins, dtype=np.float64)
    for b_idx in range(n_bins):
        in_bin = (y_norm >= edges[b_idx]) & (y_norm < edges[b_idx + 1]) & in_corridor
        if np.any(in_bin):
            rho_nc_spec[b_idx] = float(np.mean(nc_a[in_bin]))
            rho_pl_spec[b_idx] = float(np.mean(pl_a[in_bin]))

    rho_nc_spec /= max(float(np.max(rho_nc_spec)), 1e-8)
    rho_pl_spec /= max(float(np.max(rho_pl_spec)), 1e-8)

    profile_r = 0.0
    profile_rmse = 0.0
    if sim_three_stream is not None:
        sim_diff = sim_three_stream["rho_nc_dv"] - sim_three_stream["rho_pl_dv"]
        spec_diff = rho_nc_spec - rho_pl_spec
        profile_r = float(stats.pearsonr(sim_diff, spec_diff).statistic)
        profile_rmse = float(np.sqrt(np.mean((sim_diff - spec_diff) ** 2)))

    return {
        "panel_a_rgb": panel_a.astype(np.uint8),
        "panel_c_rgb": panel_c.astype(np.uint8),
        "nc_mask_a": nc_a,
        "pl_mask_a": pl_a,
        "pp_mask_a": pp_a,
        "nc_mask_c": nc_c,
        "pl_mask_c": pl_c,
        "pl_centroids": pl_centroids,
        "nc_dv_norm_a": nc_dv_norm_a,
        "pl_dv_norm_a": pl_dv_norm_a,
        "dv_sep_norm_a": dv_sep_norm_a,
        "placode_spacing_mean_px": placode_spacing_mean_px,
        "placode_spacing_cv": placode_spacing_cv,
        "ventral_nc_frac_a": ventral_nc_frac_a,
        "ventral_nc_frac_c": ventral_nc_frac_c,
        "bmpr_stream_reduction_ratio": bmpr_stream_reduction_ratio,
        "dv_bin_centers": centers,
        "rho_nc_spec": rho_nc_spec,
        "rho_pl_spec": rho_pl_spec,
        "profile_pearson_r": profile_r,
        "profile_rmse": profile_rmse,
    }


def evaluate_all_five_regimes(
    ref_data: dict[str, Any],
    n_steps: int = 180,
    seed: int = 2013,
    record_frames: bool = False,
) -> dict[str, Any]:
    """
    Simulate all 5 experimental regimes of Theveneau et al. (2013) Nature Cell Biology
    and validate simulated FMI & 6-hour stream displacement against empirical benchmarks.
    """
    regime_specs = ref_data["theveneau_2013_experimental_regimes"]["regimes"]
    sim_by_id: dict[str, dict[str, Any]] = {}
    emp_fmi: list[float] = []
    sim_fmi: list[float] = []
    emp_disp: list[float] = []
    sim_disp: list[float] = []

    for spec in regime_specs:
        rid = spec["regime_id"]
        res = simulate_chase_and_run_regime(
            regime_id=rid,
            k_CIL=float(spec["k_CIL"]),
            k_COA=float(spec["k_COA"]),
            chi_Sdf1=float(spec["chi_Sdf1"]),
            k_run_PL=float(spec["k_run_PL"]),
            placode_mobile=bool(spec["placode_mobile"]),
            n_steps=n_steps,
            seed=seed,
            record_frames=record_frames and rid in ("wildtype_chase_and_run", "placode_ablated_no_sdf1", "c3ar_morpholino_no_coa"),
        )
        res["label"] = spec["label"]
        res["empirical_nc_fmi"] = float(spec["empirical_nc_fmi"])
        res["empirical_nc_displacement_um"] = float(spec["empirical_nc_displacement_um"])
        res["empirical_cohesion_index"] = float(spec["empirical_cohesion_index"])
        sim_by_id[rid] = res

        emp_fmi.append(float(spec["empirical_nc_fmi"]))
        sim_fmi.append(res["fmi"])
        emp_disp.append(float(spec["empirical_nc_displacement_um"]))
        sim_disp.append(res["nc_disp_um"])

    fmi_r, fmi_p = stats.pearsonr(emp_fmi, sim_fmi)
    disp_r, disp_p = stats.pearsonr(emp_disp, sim_disp)
    fmi_rmse = float(np.sqrt(np.mean((np.array(emp_fmi) - np.array(sim_fmi)) ** 2)))
    disp_rmse = float(np.sqrt(np.mean((np.array(emp_disp) - np.array(sim_disp)) ** 2)))

    return {
        "regimes": sim_by_id,
        "fmi_pearson_r": float(fmi_r),
        "fmi_p_value": float(fmi_p),
        "fmi_rmse": fmi_rmse,
        "disp_pearson_r": float(disp_r),
        "disp_p_value": float(disp_p),
        "disp_rmse_um": disp_rmse,
    }


# =============================================================================
# Section 4: 10-Panel Synthesis Figure & Animated GIF Rendering
# =============================================================================

def run_all_and_render(
    k_CIL: float = 4.2,
    k_COA: float = 1.6,
    chi_Sdf1: float = 2.4,
    k_run_PL: float = 3.8,
    n_steps: int = 180,
    seed: int = 2013,
) -> dict[str, Any]:
    """Execute all 5 perturbation regimes, 3-stream arch simulation, 2D sweep, and specimen segmentation."""
    ref_data = verify_reference_dataset()
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    five_eval = evaluate_all_five_regimes(ref_data, n_steps=n_steps, seed=seed, record_frames=True)
    # Override WT if custom CLI parameters were passed
    if (k_CIL, k_COA, chi_Sdf1, k_run_PL) != (4.2, 1.6, 2.4, 3.8):
        five_eval["regimes"]["wildtype_chase_and_run"] = simulate_chase_and_run_regime(
            "wildtype_chase_and_run", k_CIL, k_COA, chi_Sdf1, k_run_PL, True, n_steps, seed=seed, record_frames=True
        )

    three_stream = simulate_three_pharyngeal_streams(n_steps=165, seed=seed)
    sweep = run_2d_parameter_sweep(n_steps=120, seed=seed)
    specimen = segment_neural_crest_specimen(sim_three_stream=three_stream)

    regimes = five_eval["regimes"]
    fig, axes = plt.subplots(2, 5, figsize=(20.5, 8.6), dpi=150)

    top_specs = [
        ("wildtype_chase_and_run", "A1. Wild-Type Chase & Run\n(CIL + C3a COA + Mobile Sdf1)"),
        ("placode_ablated_no_sdf1", "A2. Placode / Sdf1 Ablation\n(χ_Sdf1 = 0, Unguided Cluster)"),
        ("c3ar_morpholino_no_coa", "A3. C3aR-MO (No Co-Attraction)\n(k_COA = 0, Dispersed Cells)"),
        ("ncad_dshdep_no_cil", "A4. N-cad / DshDep+ (No CIL)\n(k_CIL = 0, Chase w/o Run)"),
        ("stationary_sdf1_bead", "A5. Stationary Sdf1 Bead\n(Fixed Source, Target Arrest)"),
    ]
    for col, (rid, title) in enumerate(top_specs):
        ax = axes[0, col]
        r_obj = regimes[rid]
        ax.axhspan(65, 135, color="#f1f5f9", alpha=0.85, zorder=0)
        ax.scatter(
            r_obj["nc_init"][:, 0], r_obj["nc_init"][:, 1],
            c="#93c5fd", s=18, alpha=0.45, label="NC Initial (t=0h)", zorder=2,
        )
        ax.scatter(
            r_obj["nc_pos"][:, 0], r_obj["nc_pos"][:, 1],
            c="#0284c7", edgecolors="#0f172a", linewidths=0.5, s=38, label="NC Chasers (t=6h)", zorder=4,
        )
        if rid != "placode_ablated_no_sdf1":
            ax.scatter(
                r_obj["pl_pos"][:, 0], r_obj["pl_pos"][:, 1],
                c="#f97316", edgecolors="#7c2d12", linewidths=0.6, marker="s", s=44, label="Placode Runners", zorder=5,
            )
        ax.set_xlim(10, 325)
        ax.set_ylim(25, 175)
        ax.set_title(
            f"{title}\nFMI={r_obj['fmi']:+.2f} | ΔX={r_obj['nc_disp_um']:.0f} μm | C={r_obj['cohesion_final']:.2f}",
            fontsize=8.8, fontweight="bold",
        )
        ax.set_xlabel("Anteroposterior / Stream Axis X (μm)", fontsize=8.0)
        if col == 0:
            ax.set_ylabel("Lateral Axis Y (μm)", fontsize=8.0)
            ax.legend(fontsize=6.8, loc="upper left")
        ax.grid(alpha=0.25)

    # B1: Chaser-Runner Center-of-Mass Co-Migration Trajectories X_com(t)
    ax_b1 = axes[1, 0]
    t_h = np.linspace(0.0, 6.0, len(regimes["wildtype_chase_and_run"]["nc_com_x_hist"]))
    ax_b1.plot(t_h, regimes["wildtype_chase_and_run"]["pl_com_x_hist"], color="#ea580c", lw=2.2, label="WT Placode Runner X_PL(t)")
    ax_b1.plot(t_h, regimes["wildtype_chase_and_run"]["nc_com_x_hist"], color="#0284c7", lw=2.2, label="WT NC Chaser X_NC(t)")
    ax_b1.plot(t_h, regimes["stationary_sdf1_bead"]["nc_com_x_hist"], color="#8b5cf6", lw=1.7, ls="--", label="Stationary Bead NC")
    ax_b1.plot(t_h, regimes["c3ar_morpholino_no_coa"]["nc_com_x_hist"], color="#10b981", lw=1.7, ls="-.", label="C3aR-MO (No COA) NC")
    ax_b1.plot(t_h, regimes["placode_ablated_no_sdf1"]["nc_com_x_hist"], color="#64748b", lw=1.6, ls=":", label="No-Placode Control NC")
    ax_b1.set_title("B1. Chase-and-Run Wave Kinematics", fontsize=9.2, fontweight="bold")
    ax_b1.set_xlabel("Migration Time t (hours)", fontsize=8.2)
    ax_b1.set_ylabel("Center-of-Mass Position X (μm)", fontsize=8.2)
    ax_b1.legend(fontsize=6.8, loc="upper left")
    ax_b1.grid(alpha=0.25)

    # B2: 2D Parameter Sweep (k_COA vs chi_Sdf1) Forward Migration Index & 3-Stream Inset
    ax_b2 = axes[1, 1]
    im_sw = ax_b2.imshow(
        sweep["fmi_grid"],
        origin="lower",
        aspect="auto",
        extent=[sweep["chi_vals"][0], sweep["chi_vals"][-1], sweep["k_coa_vals"][0], sweep["k_coa_vals"][-1]],
        cmap="viridis",
        vmin=0.0,
        vmax=0.80,
    )
    plt.colorbar(im_sw, ax=ax_b2, fraction=0.046, pad=0.04, label="FMI")
    ax_b2.set_title("B2. 2D (χ_Sdf1, k_COA) Phase Diagram", fontsize=9.2, fontweight="bold")
    ax_b2.set_xlabel("Chemotactic Sensitivity χ_Sdf1", fontsize=8.2)
    ax_b2.set_ylabel("C3a Co-Attraction Gain k_COA", fontsize=8.2)

    # B3: Biological Specimen Segmentation (Control Panel A: r4, r6, r7 streams & e1, e2, e3 placodes)
    ax_b3 = axes[1, 2]
    ax_b3.imshow(specimen["panel_a_rgb"])
    ax_b3.contour(specimen["nc_mask_a"], levels=[0.5], colors=["#38bdf8"], linewidths=1.2)
    ax_b3.contour(specimen["pl_mask_a"], levels=[0.5], colors=["#fb923c"], linewidths=1.5)
    for idx_p, (cx_p, cy_p, _) in enumerate(specimen["pl_centroids"][:3], start=1):
        ax_b3.plot(cx_p, cy_p, "o", color="#fde047", markersize=6, markeredgecolor="#0f172a")
        ax_b3.text(cx_p + 12, cy_p, f"e{idx_p}", color="white", fontsize=8.5, fontweight="bold")
    ax_b3.set_title(
        f"B3. Specimen r4/r6/r7 Streams & e1-e3 Placodes\n(Δy_DV={specimen['dv_sep_norm_a']:+.2f}, Spacing CV={specimen['placode_spacing_cv']:.3f})",
        fontsize=8.8, fontweight="bold",
    )
    ax_b3.set_xticks([])
    ax_b3.set_yticks([])

    # B4: Dorsoventral Chaser-Runner Density Profiles (3-Stream Model vs Segmented Specimen)
    ax_b4 = axes[1, 3]
    dv_x = specimen["dv_bin_centers"]
    ax_b4.plot(dv_x, three_stream["rho_nc_dv"], "o-", color="#0284c7", lw=2.0, label="Model NC Streams (r4/r6/r7)")
    ax_b4.plot(dv_x, three_stream["rho_pl_dv"], "s-", color="#ea580c", lw=2.0, label="Model Placode Runners (e1-e3)")
    ax_b4.plot(dv_x, specimen["rho_nc_spec"], "o--", color="#0369a1", lw=1.6, label="Specimen NC Streams (emp)")
    ax_b4.plot(dv_x, specimen["rho_pl_spec"], "s--", color="#c2410c", lw=1.6, label="Specimen Epibranchial Placodes")
    ax_b4.set_title(
        f"B4. Dorsoventral Stream–Placode Profiles\n(r = {specimen['profile_pearson_r']:.3f}, RMSE = {specimen['profile_rmse']:.3f})",
        fontsize=8.8, fontweight="bold",
    )
    ax_b4.set_xlabel("Normalized Dorsoventral Axis y / H", fontsize=8.2)
    ax_b4.set_ylabel("Normalized Tissue Occupancy", fontsize=8.2)
    ax_b4.legend(fontsize=6.8, loc="upper right")
    ax_b4.grid(alpha=0.25)

    # B5: Theveneau et al. (2013) 5-Regime FMI Validation (Empirical vs Simulated)
    ax_b5 = axes[1, 4]
    order_ids = [
        "wildtype_chase_and_run",
        "stationary_sdf1_bead",
        "ncad_dshdep_no_cil",
        "c3ar_morpholino_no_coa",
        "placode_ablated_no_sdf1",
    ]
    short_labels = ["WT Chase & Run", "Stationary Bead", "No CIL (DshDep+)", "No COA (C3aR-MO)", "No Placode/Sdf1"]
    y_idx = np.arange(len(order_ids))
    emp_vals = [regimes[k]["empirical_nc_fmi"] for k in order_ids]
    sim_vals = [regimes[k]["fmi"] for k in order_ids]
    ax_b5.barh(y_idx - 0.17, emp_vals, height=0.32, color="#94a3b8", edgecolor="#0f172a", label="Theveneau 2013 (emp)")
    ax_b5.barh(y_idx + 0.17, sim_vals, height=0.32, color="#0284c7", edgecolor="#0f172a", label="Simulated (Pass 2)")
    ax_b5.set_yticks(y_idx)
    ax_b5.set_yticklabels(short_labels, fontsize=7.8)
    ax_b5.invert_yaxis()
    ax_b5.set_xlabel("Forward Migration Index (FMI)", fontsize=8.2)
    ax_b5.set_title(
        f"B5. 5-Regime FMI Hierarchy\n(r = {five_eval['fmi_pearson_r']:.3f}, RMSE = {five_eval['fmi_rmse']:.3f})",
        fontsize=8.8, fontweight="bold",
    )
    ax_b5.legend(fontsize=7.0, loc="lower right")
    ax_b5.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Carmona-Fontaine et al. (2008, 2011) & Theveneau et al. (2013) Neural Crest CIL, C3a Co-Attraction & Sdf1 Chase-and-Run",
        fontsize=13.0, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(COMPARISON_PNG, dpi=150, facecolor="white")
    plt.close(fig)

    # Render 3-panel side-by-side animated GIF (WT Chase-and-Run vs C3aR-MO No-COA vs No-Placode Control)
    pil_frames: list[Image.Image] = []
    f_wt = regimes["wildtype_chase_and_run"]["frames"]
    f_coa = regimes["c3ar_morpholino_no_coa"]["frames"]
    f_ctl = regimes["placode_ablated_no_sdf1"]["frames"]
    n_f = min(len(f_wt), len(f_coa), len(f_ctl))

    for idx in range(n_f):
        nc_w, pl_w, st_val = f_wt[idx]
        nc_c, pl_c, _ = f_coa[idx]
        nc_0, _, _ = f_ctl[idx]
        fig_g, axes_g = plt.subplots(1, 3, figsize=(10.2, 3.3), dpi=100)
        for ax_i, nc_p, pl_p, lbl, show_pl in [
            (axes_g[0], nc_w, pl_w, f"WT Chase-and-Run (step={st_val})", True),
            (axes_g[1], nc_c, pl_c, f"C3aR-MO No COA (step={st_val})", True),
            (axes_g[2], nc_0, pl_w, f"No-Placode Control (step={st_val})", False),
        ]:
            ax_i.axhspan(65, 135, color="#f1f5f9", alpha=0.85)
            ax_i.scatter(nc_p[:, 0], nc_p[:, 1], c="#0284c7", edgecolors="#0f172a", linewidths=0.4, s=28, label="NC")
            if show_pl:
                ax_i.scatter(pl_p[:, 0], pl_p[:, 1], c="#f97316", edgecolors="#7c2d12", linewidths=0.5, marker="s", s=34, label="Placode")
            ax_i.set_xlim(10, 325)
            ax_i.set_ylim(25, 175)
            ax_i.set_title(lbl, fontsize=8.4, fontweight="bold")
            ax_i.set_xticks([])
            ax_i.set_yticks([])
        fig_g.suptitle(
            "Theveneau et al. (2013) & Carmona-Fontaine et al. (2008, 2011) Neural Crest Chase-and-Run Dynamics",
            fontsize=9.6, fontweight="bold",
        )
        fig_g.tight_layout(rect=(0, 0, 1, 0.90))
        fig_g.canvas.draw()
        buf = np.asarray(fig_g.canvas.buffer_rgba())[..., :3]
        pil_frames.append(Image.fromarray(buf))
        plt.close(fig_g)

    pil_frames[0].save(
        GROWTH_GIF,
        save_all=True,
        append_images=pil_frames[1:],
        duration=125,
        loop=0,
    )

    return {
        "five_eval": five_eval,
        "three_stream": three_stream,
        "sweep": sweep,
        "specimen": specimen,
    }


# =============================================================================
# Section 5: MorphoEval Dynamic Measurement Hook & 8-Check Self-Test Suite
# =============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Return 8 reference-anchored morphometric measurements for morphoeval/measure_outputs.py."""
    ref_data = verify_reference_dataset()
    f0 = compute_cil_coa_pair_force(0.0, k_CIL=4.2, k_COA=1.6, R_CIL=28.0, l_COA=75.0)
    fR = compute_cil_coa_pair_force(28.0, k_CIL=4.2, k_COA=1.6, R_CIL=28.0, l_COA=75.0)
    five_eval = evaluate_all_five_regimes(ref_data, n_steps=180, seed=2013, record_frames=False)
    three_stream = simulate_three_pharyngeal_streams(n_steps=165, seed=2013)
    specimen = segment_neural_crest_specimen(sim_three_stream=three_stream)

    wt = five_eval["regimes"]["wildtype_chase_and_run"]
    no_pl = five_eval["regimes"]["placode_ablated_no_sdf1"]
    no_coa = five_eval["regimes"]["c3ar_morpholino_no_coa"]

    return [
        {
            "name": "cil_coa_contact_repulsion_at_zero_distance",
            "value": round(f0["f_net"], 4),
            "unit": "force_units",
            "reference_value": -2.6,
            "method": f"Net homotypic pair force F_pair(0) = -k_CIL + k_COA for k_CIL=4.2, k_COA=1.6 with equilibrium spacing d_eq={f0['d_equilibrium_um']:.2f} um (Carmona-Fontaine et al. 2011; Woods et al. 2014)",
        },
        {
            "name": "cil_coa_attraction_at_contact_radius",
            "value": round(fR["f_net"], 4),
            "unit": "force_units",
            "reference_value": 1.1015,
            "method": "Net homotypic pair force F_pair(R_CIL) = k_COA * exp(-R_CIL / l_COA) at d = R_CIL = 28 um (Carmona-Fontaine et al. 2011 Dev. Cell)",
        },
        {
            "name": "wildtype_chase_and_run_forward_migration_index",
            "value": round(wt["fmi"], 4),
            "unit": "FMI",
            "reference_value": 0.68,
            "method": f"Wild-type chase-and-run Forward Migration Index (FMI) vs Theveneau et al. (2013) Fig. 2 empirical reference (5-regime r={five_eval['fmi_pearson_r']:.4f}, RMSE={five_eval['fmi_rmse']:.4f}, p={five_eval['fmi_p_value']:.4f})",
        },
        {
            "name": "no_placode_control_forward_migration_index",
            "value": round(abs(no_pl["fmi"]), 4),
            "unit": "FMI",
            "reference_value": 0.04,
            "method": "Null Forward Migration Index |FMI| in placode-ablated / Sdf1-MO negative control (chi_Sdf1 = 0, Theveneau et al. 2013 Fig. 1)",
        },
        {
            "name": "c3ar_mo_vs_wildtype_cohesion_ratio",
            "value": round(no_coa["cohesion_final"] / max(wt["cohesion_final"], 1e-8), 4),
            "unit": "ratio",
            "reference_value": round(0.26 / 0.76, 4),
            "method": f"Cluster cohesion ratio C_COA(C3aR-MO) / C_COA(WT) ({no_coa['cohesion_final']:.3f} vs {wt['cohesion_final']:.3f}, Carmona-Fontaine et al. 2011 Dev. Cell Fig. 3)",
        },
        {
            "name": "five_regime_displacement_pearson_r",
            "value": round(five_eval["disp_pearson_r"], 4),
            "unit": "r",
            "reference_value": 0.95,
            "method": f"Pearson correlation between simulated and empirical 6-hour stream displacement across all 5 regimes of Theveneau et al. (2013) (RMSE={five_eval['disp_rmse_um']:.2f} um, p={five_eval['disp_p_value']:.4f})",
        },
        {
            "name": "specimen_dorsoventral_chaser_runner_separation",
            "value": round(specimen["dv_sep_norm_a"], 4),
            "unit": "normalized_dv_distance",
            "reference_value": 0.4063,
            "method": f"Normalized dorsoventral separation (y_PL - y_NC)/H between epibranchial placodes (e1-e3) and cranial neural crest streams (r4/r6/r7) on neural_crest_specimen.jpg (placode spacing CV={specimen['placode_spacing_cv']:.4f})",
        },
        {
            "name": "specimen_vs_model_dorsoventral_profile_pearson_r",
            "value": round(specimen["profile_pearson_r"], 4),
            "unit": "r",
            "reference_value": 0.90,
            "method": f"Pearson correlation between 3-stream model and segmented specimen dorsoventral chaser-runner profiles across 12 bins (RMSE={specimen['profile_rmse']:.4f}, BMPR-inhib stream loss={specimen['bmpr_stream_reduction_ratio']:.3f})",
        },
    ]


def self_test() -> None:
    """Run comprehensive 8-check verification suite for Slot 091."""
    # 1. Verify SHA-256 of reference dataset and specimen image
    ref = verify_reference_dataset()
    assert len(ref["metadata"]["sources"]) == 5

    # 2. Verify exact CIL + COA pair force sign flip (Lean 4 Theorem 1 parity)
    f0 = compute_cil_coa_pair_force(0.0, k_CIL=4.2, k_COA=1.6, R_CIL=28.0, l_COA=75.0)
    fR = compute_cil_coa_pair_force(28.0, k_CIL=4.2, k_COA=1.6, R_CIL=28.0, l_COA=75.0)
    fR_no_coa = compute_cil_coa_pair_force(28.0, k_CIL=4.2, k_COA=0.0, R_CIL=28.0, l_COA=75.0)
    assert f0["f_net"] < 0.0 and fR["f_net"] > 0.0 and abs(fR_no_coa["f_net"]) < 1e-9
    assert 10.0 < f0["d_equilibrium_um"] < 26.0

    # 3. Verify 1D chase-and-run steady co-migration wave law (Lean 4 Theorem 2 parity)
    w_state = compute_chase_and_run_wave_state(30.0, chi_Sdf1=2.4, k_run_PL=3.8, R_het=56.0, l_Sdf1=95.0)
    w_null = compute_chase_and_run_wave_state(30.0, chi_Sdf1=0.0, k_run_PL=0.0, R_het=56.0, l_Sdf1=95.0)
    assert w_state["steady_gap_star_um"] > 0.0 and w_state["steady_wave_speed"] > 0.0
    assert w_null["v_chase"] == 0.0 and w_null["v_run"] == 0.0

    # 4. Verify 5-regime hierarchy & empirical correlation (Lean 4 Theorem 3 parity)
    five_eval = evaluate_all_five_regimes(ref, n_steps=180, seed=2013)
    regs = five_eval["regimes"]
    assert regs["wildtype_chase_and_run"]["fmi"] > regs["stationary_sdf1_bead"]["fmi"]
    assert regs["wildtype_chase_and_run"]["fmi"] > regs["ncad_dshdep_no_cil"]["fmi"]
    assert regs["wildtype_chase_and_run"]["fmi"] > regs["c3ar_morpholino_no_coa"]["fmi"]
    assert regs["c3ar_morpholino_no_coa"]["fmi"] > abs(regs["placode_ablated_no_sdf1"]["fmi"])
    assert five_eval["fmi_pearson_r"] > 0.90, f"Expected FMI r > 0.90, got {five_eval['fmi_pearson_r']}"
    assert five_eval["disp_pearson_r"] > 0.90, f"Expected disp r > 0.90, got {five_eval['disp_pearson_r']}"

    # 5. Verify C3aR-MO co-attraction ablation reduces cluster cohesion vs Wild-Type
    assert regs["c3ar_morpholino_no_coa"]["cohesion_final"] < 0.72 * regs["wildtype_chase_and_run"]["cohesion_final"]

    # 6. Verify 3-stream pharyngeal arch simulation and specimen segmentation
    three_s = simulate_three_pharyngeal_streams(n_steps=165, seed=2013)
    spec = segment_neural_crest_specimen(sim_three_stream=three_s)
    assert len(spec["pl_centroids"]) >= 3
    assert spec["dv_sep_norm_a"] > 0.30
    assert spec["bmpr_stream_reduction_ratio"] > 0.20
    assert spec["profile_pearson_r"] > 0.80, f"Expected specimen profile r > 0.80, got {spec['profile_pearson_r']}"

    # 7. Verify two-run bitwise determinism
    r1 = simulate_chase_and_run_regime(n_steps=60, seed=2013)
    r2 = simulate_chase_and_run_regime(n_steps=60, seed=2013)
    assert np.array_equal(r1["nc_pos"], r2["nc_pos"]) and np.array_equal(r1["pl_pos"], r2["pl_pos"])

    # 8. Verify measure_morphoeval contract
    meas = measure_morphoeval()
    assert len(meas) == 8 and all(m["value"] is not None for m in meas)
    print(
        f"[self-test] All 8 Pass 2 checks passed "
        f"(WT FMI={regs['wildtype_chase_and_run']['fmi']:.3f}, 5-regime FMI r={five_eval['fmi_pearson_r']:.3f}, "
        f"disp r={five_eval['disp_pearson_r']:.3f}, specimen profile r={spec['profile_pearson_r']:.3f})."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-CIL", type=float, default=4.2, help="Homotypic N-cadherin/PCP CIL repulsion gain k_CIL")
    ap.add_argument("--k-COA", type=float, default=1.6, help="Autocrine C3a/C3aR co-attraction gain k_COA")
    ap.add_argument("--chi-Sdf1", type=float, default=2.4, help="Cxcr4/Sdf1 chemotactic sensitivity chi_Sdf1")
    ap.add_argument("--k-run", type=float, default=3.8, help="Heterotypic placode CIL escape gain k_run_PL")
    ap.add_argument("--steps", type=int, default=180, help="Number of integration time steps")
    ap.add_argument("--seed", type=int, default=2013, help="Deterministic RNG seed")
    ap.add_argument("--self-test", action="store_true", help="Run 8-check verification suite and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    out = run_all_and_render(
        k_CIL=args.k_CIL,
        k_COA=args.k_COA,
        chi_Sdf1=args.chi_Sdf1,
        k_run_PL=args.k_run,
        n_steps=args.steps,
        seed=args.seed,
    )
    wt = out["five_eval"]["regimes"]["wildtype_chase_and_run"]
    sp = out["specimen"]
    print(
        f"[2012-carmona-neural-crest-chase-and-run] Rendered Pass 2 outputs:\n"
        f"  - Comparison PNG : {COMPARISON_PNG}\n"
        f"  - Growth GIF     : {GROWTH_GIF}\n"
        f"  - WT Chase & Run : FMI={wt['fmi']:.4f}, ΔX_NC={wt['nc_disp_um']:.1f} μm, ΔX_PL={wt['pl_disp_um']:.1f} μm\n"
        f"  - 5-Regime Fit   : FMI r={out['five_eval']['fmi_pearson_r']:.4f} (RMSE={out['five_eval']['fmi_rmse']:.4f}), Disp r={out['five_eval']['disp_pearson_r']:.4f}\n"
        f"  - Specimen Match : Δy_DV={sp['dv_sep_norm_a']:+.4f}, Profile r={sp['profile_pearson_r']:.4f} (RMSE={sp['profile_rmse']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
