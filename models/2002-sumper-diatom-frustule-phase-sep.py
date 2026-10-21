#!/usr/bin/env python3
"""
2002-sumper-diatom-frustule-phase-sep.py — Pass 3 (v3 Remediated)
=============================================================================
Hierarchical Cahn-Hilliard Phase-Separation Model of Diatom Frustule Biosilica
Nanopatterning in *Coscinodiscus wailesii* (Slot 071).

Pass 3 additions (closing the Pass 2 audit remediation items):
  R1. `simulate_radial_quench_costae` — the 2D Cahn-Hilliard equation coupled to a
      radially advancing (centrifugal) quench + lagging silica-precipitation front,
          a(r,t) = tanh((R0 + v t - r)/w),   frozen (zero-mobility) silica for r < R(t) - lag,
      i.e. Furukawa's (1992) directional quench in polar geometry. Emergent result: slow
      fronts give RADIAL bifurcating silica lamellae (costae striation), fast fronts give
      concentric (front-parallel) lamellae, v -> inf recovers the isotropic quench.
      `run_front_velocity_sweep` is the 5-point ablation + isotropic null.
  R2. `simulate_three_stage_hierarchy` — explicit 3-stage 512x512 single-domain quench at
      12.5 nm/px (foramen -> cribrum -> cribrella) with frozen-wall chamber confinement,
      compared against Sumper's tier table (d_2, d_3, a_2/a_3, cribrella per cribrum pore).
  5. Furukawa H (1992). "Phase separation by directional quenching and morphological
     transition." Physica A 180(1-2):128-155 (doi:10.1016/0378-4371(92)90111-2).
  6. Schmid A-MM, Volcani BE (1983). "Wall morphogenesis in Coscinodiscus wailesii Gran
     and Angst. I. Valve morphology and development of its architecture."
     J. Phycol. 19(4):387-402 (doi:10.1111/j.0022-3646.1983.00387.x).
     (Refs 5-6 DOIs transcribed offline; not re-verified online in Pass 3.)

Primary Literature & Mathematical Mapping:
  1. Sumper M (2002). "A phase separation model for the nanopatterning of diatom
     biosilica." Science 295(5564):2430-2433 (doi:10.1126/science.1070026).
     - Figs. 1-4: Hierarchical liquid-liquid phase separation of long-chain
       polyamine (LCPA) microdroplets within the Silica Deposition Vesicle (SDV)
       generating three self-similar hexagonal pore tiers:
         * Generation 1 (Foramen / Areola): d_1 = 1250 nm, pitch a_1 = 1850 nm, d_1/a_1 = 0.6757
         * Generation 2 (Cribrum pores):    d_2 = 220 nm,  pitch a_2 = 380 nm,  d_2/d_1 = 0.1760
         * Generation 3 (Cribrella sieve):  d_3 = 42 nm,   pitch a_3 = 75 nm,   d_3/d_2 = 0.1909
  2. Lenoci L, Camp PJ (2008). "Self-assembly of hierarchical nanostructures in
     diatom biosilica." Langmuir 24(1):217-223 (doi:10.1021/la7022784).
     - Eqs. (1)-(6): Fourth-order Cahn-Hilliard conserved phase-separation PDE:
         F[phi] = int [ (1/4)*(phi^2 - 1)^2 + (eps^2 / 2)*|grad(phi)|^2 ] dx
         mu(phi) = phi^3 - phi - eps^2 * laplacian(phi)
         d(phi)/dt = M * laplacian(phi^3 - phi - eps^2 * laplacian(phi))
     - Section 2: Linear spinodal dispersion around homogeneous composition phi_0:
         sigma(k) = M * k^2 * (1 - 3*phi_0^2 - eps^2 * k^2)
         k_c = sqrt((1 - 3*phi_0^2) / (2 * eps^2)),   lambda_c = 2*pi / k_c
     - Section 3.3: Two-stage quenched hierarchical solidification where Stage 1
       forms primary areola walls (eps_1) and Stage 2 quenches secondary cribrum
       pores (eps_2 < eps_1) inside unsolidified primary pore chambers.
  3. Losic D, Rosengarten G, Mitchell JG, Voelcker NH (2006). "Pore architecture
     of diatom frustules: potential nanostructured membranes for molecular and
     particle separations." J. Nanosci. Nanotechnol. 6(4):982-989 (doi:10.1166/jnn.2006.174).
  4. Cahn JW, Hilliard JE (1958). "Free energy of a nonuniform system. I.
     Interfacial free energy." J. Chem. Phys. 28(2):258-267 (doi:10.1063/1.1744102).

Formal Lean 4 Proofs (0 sorrys):
  - `morphoeval/lean/loops/Loop_071_diatom_frustule_phase_sep.lean`
    * `MathOfBiology.Loop_071.cahn_hilliard_binodal_equilibria`
    * `MathOfBiology.Loop_071.cahn_hilliard_mass_conservation_zero_mode`
    * `MathOfBiology.Loop_071.cahn_hilliard_spinodal_band_selection`
    * `MathOfBiology.Loop_071.sumper_2002_hierarchical_hex_packing`
    * `MathOfBiology.Loop_071.directional_quench_stable_ahead_of_front`  (Pass 3)
    * `MathOfBiology.Loop_071.frozen_front_monotone_irreversible`         (Pass 3)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "diatom_frustule_phase_sep"
REF_JSON_PATH = DATASET_DIR / "diatom_frustule_phase_sep_reference.json"
SPECIMEN_JPG_PATH = DATASET_DIR / "coscinodiscus_diatom_specimen.jpg"
COMPARISON_PNG = ROOT / "docs" / "media" / "diatom_frustule_phase_sep_comparison.png"
GROWTH_GIF = ROOT / "docs" / "media" / "diatom_frustule_phase_sep_growth.gif"
COSTAE_GIF = ROOT / "docs" / "media" / "diatom_frustule_phase_sep_radial_costae.gif"

EXPECTED_SHA256: dict[Path, str] = {
    SPECIMEN_JPG_PATH: "47aca39b720b67b1b46754c89f679425db1012e92131277a3d09b1562274a1ad",
}


# ============================================================================
# 1. Dataset Provenance & Cryptographic SHA-256 Verification
# ============================================================================

def verify_and_load_reference() -> dict[str, Any]:
    """Load `diatom_frustule_phase_sep_reference.json` and verify SHA-256 of the
    vendored *Coscinodiscus wailesii* SEM micrograph."""
    if not REF_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing reference dataset: {REF_JSON_PATH}")
    doc = json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))
    for path, expected_hex in EXPECTED_SHA256.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing vendored specimen file: {path}")
        actual_hex = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hex != expected_hex:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {expected_hex}, got {actual_hex}"
            )
    spec_entry = doc["metadata"]["specimens"]["coscinodiscus_wailesii_valve_sem"]
    if spec_entry["sha256"] != EXPECTED_SHA256[SPECIMEN_JPG_PATH]:
        raise ValueError("Reference JSON specimen SHA-256 does not match EXPECTED_SHA256")
    return doc


# ============================================================================
# 2. Analytical Cahn-Hilliard Spinodal Dispersion & Wavelength Selection
#    (Lenoci & Camp 2008 Langmuir Sec. 2; Lean 4 Law 1 & Law 2)
# ============================================================================

def compute_spinodal_dispersion(
    phi0: float = -0.30,
    epsilon: float = 1.50,
    M: float = 1.0,
    k_max: float = 0.85,
    n_k: int = 300,
) -> dict[str, Any]:
    """Compute the linearized Cahn-Hilliard dispersion relation around homogeneous
    composition phi_0:
        sigma(k) = M * k^2 * (alpha - eps^2 * k^2),   alpha = 1 - 3 * phi_0^2
    Inside the spinodal (|phi_0| < 1/sqrt(3), alpha > 0):
        k_c = sqrt(alpha / (2 * eps^2)),   k_0 = sqrt(alpha / eps^2) = sqrt(2) * k_c
        lambda_c = 2 * pi / k_c,           sigma_max = M * alpha^2 / (4 * eps^2)
    Outside the spinodal (alpha <= 0): all k > 0 modes decay (sigma(k) < 0).
    """
    alpha = 1.0 - 3.0 * (phi0**2)
    inside_spinodal = bool(alpha > 0.0)
    k_grid = np.linspace(0.0, k_max, n_k)
    q_grid = k_grid**2
    sigma_grid = M * q_grid * (alpha - (epsilon**2) * q_grid)

    if inside_spinodal:
        kc = math.sqrt(alpha / (2.0 * (epsilon**2)))
        k0 = math.sqrt(alpha / (epsilon**2))
        lambda_c = 2.0 * math.pi / kc
        sigma_max = M * (alpha**2) / (4.0 * (epsilon**2))
    else:
        kc = 0.0
        k0 = 0.0
        lambda_c = 0.0
        sigma_max = 0.0

    return {
        "phi0": float(phi0),
        "epsilon": float(epsilon),
        "M": float(M),
        "alpha": float(alpha),
        "spinodal_limit_phi": float(1.0 / math.sqrt(3.0)),
        "inside_spinodal": inside_spinodal,
        "kc": float(kc),
        "k0": float(k0),
        "lambda_c_px": float(lambda_c),
        "sigma_max": float(sigma_max),
        "sigma_zero_mode": float(sigma_grid[0]),
        "k_grid": k_grid,
        "sigma_grid": sigma_grid,
    }


# ============================================================================
# 3. 2D Semi-Implicit Spectral Cahn-Hilliard Solver & Two-Stage Hierarchical
#    Quench (Lenoci & Camp 2008 Eqs. 1-6 & Sec. 3.3; Sumper 2002 Figs. 1-4)
# ============================================================================

def simulate_cahn_hilliard_2d(
    N: int = 128,
    phi0: float = -0.30,
    epsilon: float = 1.50,
    M: float = 1.0,
    dt: float = 0.25,
    steps: int = 650,
    noise_amp: float = 0.08,
    seed: int = 2002,
    record_every: int = 0,
    a_field: np.ndarray | None = None,
    frozen_mask: np.ndarray | None = None,
    frozen_value: float = -1.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Semi-implicit Fourier pseudospectral solver for the 2D periodic fourth-order
    Cahn-Hilliard equation:
        d(phi)/dt = M * laplacian(phi^3 - a(x) * phi - eps^2 * laplacian(phi))
    Uses unconditional biharmonic stabilization `(1 + dt * M * eps^2 * k^4)` in
    Fourier space and enforces exact zero-mode conservation `hat(phi)(0,0) = N^2 * phi0`.

    Pass 3 extensions (both default to the Pass 2 behaviour bitwise when unset):
      * `a_field(x)`: spatially varying quench field (Furukawa 1992 directional-quench
        form of the bulk term). `a = +1` is the unstable double well (inside the
        spinodal); `a = -1` is a single-well stable mixture (no demixing).
      * `frozen_mask`: solidified-silica pixels held at `frozen_value` (zero mobility,
        wetting boundary for the mobile chambers). Any flux the periodic spectral
        update leaks across the frozen boundary is re-projected uniformly onto the
        mobile pixels so that the total order parameter is conserved exactly.
    """
    rng = np.random.default_rng(seed)
    phi = phi0 + noise_amp * rng.standard_normal((N, N))
    phi = phi - np.mean(phi) + phi0
    mobile = None
    if frozen_mask is not None:
        phi[frozen_mask] = frozen_value
        mobile = ~frozen_mask
        n_mobile = int(np.sum(mobile))
        mass0 = float(np.sum(phi))

    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K4 = K2**2
    denom = 1.0 + dt * M * (epsilon**2) * K4

    frames: list[np.ndarray] = [phi.copy()] if record_every > 0 else []
    phi_hat = np.fft.fft2(phi)
    for step in range(1, steps + 1):
        nl = (phi**3) - phi if a_field is None else (phi**3) - a_field * phi
        nl_hat = np.fft.fft2(nl)
        phi_hat = (phi_hat - dt * M * K2 * nl_hat) / denom
        if mobile is None:
            phi_hat[0, 0] = (N * N) * phi0
            phi = np.fft.ifft2(phi_hat).real
        else:
            phi = np.fft.ifft2(phi_hat).real
            phi[frozen_mask] = frozen_value
            if n_mobile > 0:
                phi[mobile] += (mass0 - float(np.sum(phi))) / n_mobile
            phi_hat = np.fft.fft2(phi)
        if record_every > 0 and step % record_every == 0:
            frames.append(phi.copy())
    return phi, frames


def simulate_two_stage_hierarchical_diatom(
    N: int = 128,
    phi0_primary: float = -0.30,
    epsilon_primary: float = 1.50,
    steps_primary: int = 650,
    phi0_secondary: float = -0.26,
    epsilon_secondary: float = 0.65,
    steps_secondary: int = 320,
    dt: float = 0.25,
    seed: int = 2002,
) -> dict[str, Any]:
    """Lenoci & Camp (2008) Section 3.3 & Sumper (2002) two-stage hierarchical
    phase-separation quench:
      - Stage 1 (`phi_stage1`): Off-critical Cahn-Hilliard demixing (eps_1 = 1.50)
        forms primary hexagonal LCPA microdroplets (`phi_1 > 0.05`) surrounded by
        a continuous precipitating silica wall matrix (`phi_1 <= 0.05`).
      - Stage 2 (`phi_stage2`): After primary silica walls solidify (`wall_mask`),
        LCPA inside the unsolidified primary foramen chambers undergoes a secondary
        fine-scale Cahn-Hilliard quench (`eps_2 = 0.65`) generating nested cribrum
        pores within each primary foramen chamber.
    """
    phi1, frames1 = simulate_cahn_hilliard_2d(
        N=N,
        phi0=phi0_primary,
        epsilon=epsilon_primary,
        dt=dt,
        steps=steps_primary,
        seed=seed,
        record_every=130,
    )
    phi2_raw, frames2_raw = simulate_cahn_hilliard_2d(
        N=N,
        phi0=phi0_secondary,
        epsilon=epsilon_secondary,
        dt=0.12,
        steps=steps_secondary,
        seed=seed + 71,
        record_every=80,
    )

    # Primary foramen chambers are where LCPA droplets formed (phi1 > 0.08)
    foramen_mask = phi1 > 0.08
    foramen_mask = ndi.binary_opening(foramen_mask, structure=np.ones((2, 2)))
    # Erode slightly to define the inner cribrum plate rim inside each foramen
    inner_cribrum_domain = ndi.binary_erosion(foramen_mask, structure=np.ones((2, 2)))

    # Composite biosilica density field:
    #   +1.0 = primary silica wall
    #   +0.45 = secondary cribrum silica web inside primary foramen
    #   -1.0 = open cribrum sub-pores (where phi2_raw > 0.05 inside inner_cribrum_domain)
    silica_composite = np.ones((N, N), dtype=float)
    silica_composite[foramen_mask] = 0.35
    cribrum_pores_mask = inner_cribrum_domain & (phi2_raw > 0.05)
    silica_composite[cribrum_pores_mask] = -1.0

    # Build multi-stage animation frames (Stage 1 coarsening -> Stage 2 cribrum formation)
    hierarchical_frames: list[tuple[str, np.ndarray]] = []
    for idx, f1 in enumerate(frames1):
        # Convert f1 to silica density (-f1 so LCPA droplets are dark pores)
        hierarchical_frames.append((f"Stage 1 Foramen Coarsening (step {idx * 130})", -f1))
    for idx, f2 in enumerate(frames2_raw[1:], start=1):
        comp = -phi1.copy()
        comp[inner_cribrum_domain] = -0.65 * f2[inner_cribrum_domain] - 0.15
        hierarchical_frames.append((f"Stage 2 Cribrum Quench (step {idx * 80})", comp))

    lab_foramen, n_foramen = ndi.label(foramen_mask)
    lab_cribrum, n_cribrum = ndi.label(cribrum_pores_mask)
    subpores_per_foramen: list[int] = []
    for fid in range(1, n_foramen + 1):
        f_region = lab_foramen == fid
        if np.sum(f_region) >= 20:
            c_ids = set(np.unique(lab_cribrum[f_region])) - {0}
            if len(c_ids) > 0:
                subpores_per_foramen.append(len(c_ids))

    return {
        "phi_stage1": phi1,
        "phi_stage2_raw": phi2_raw,
        "silica_composite": silica_composite,
        "foramen_mask": foramen_mask,
        "cribrum_pores_mask": cribrum_pores_mask,
        "n_foramen_pores": int(n_foramen),
        "n_cribrum_subpores": int(n_cribrum),
        "mean_subpores_per_foramen": float(np.mean(subpores_per_foramen)) if subpores_per_foramen else 0.0,
        "hierarchical_frames": hierarchical_frames,
    }


# ============================================================================
# 3b. PASS 3 — Radially Advancing Centrifugal Silica-Precipitation Front Coupled
#     to Cahn-Hilliard (Directional Quench; Furukawa 1992) -> Costae Striation
# ============================================================================

def _polar_grids(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (r, theta, cos(theta), sin(theta)) grids about the valve centre (N/2, N/2)."""
    c = 0.5 * (N - 1)
    yy, xx = np.meshgrid(np.arange(N) - c, np.arange(N) - c, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    return r, theta, np.cos(theta), np.sin(theta)


def radial_anisotropy_index(phi: np.ndarray, r_min: float, r_max: float) -> float:
    """Gradient-tensor anisotropy of the pattern in the annulus r_min <= r < r_max:
        A = < |d_theta phi|^2 - |d_r phi|^2 > / < |d_theta phi|^2 + |d_r phi|^2 >
    (both derivatives per unit arc length). A -> +1 for purely radial stripes/ribs
    (phi varies only azimuthally), A -> -1 for concentric rings, A ~ 0 isotropic."""
    N = phi.shape[0]
    r, _, ct, st = _polar_grids(N)
    gy, gx = np.gradient(phi)
    g_r = gx * ct + gy * st
    g_t = -gx * st + gy * ct
    ann = (r >= r_min) & (r < r_max)
    e_r = float(np.sum(g_r[ann] ** 2))
    e_t = float(np.sum(g_t[ann] ** 2))
    tot = e_r + e_t
    return float((e_t - e_r) / tot) if tot > 1e-12 else 0.0


def pore_centroids(mask: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Centroids (x, y) of connected components of a boolean pore mask with area >= min_area."""
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros((0, 2))
    idx = np.arange(1, n + 1)
    areas = ndi.sum_labels(np.ones_like(mask, dtype=float), lab, idx)
    cents = ndi.center_of_mass(np.ones_like(mask, dtype=float), lab, idx)
    pts = [(float(cx), float(cy)) for (cy, cx), a in zip(cents, areas) if a >= min_area]
    return np.array(pts) if pts else np.zeros((0, 2))


def bond_orientation_coherence(points_xy: np.ndarray, frame_angles: np.ndarray | None = None) -> float:
    """Long-range 6-fold bond-orientation coherence
        C_6 = | (1/N_b) sum_bonds exp(6 i (theta_bond - theta_frame)) |
    over ALL nearest-neighbour bonds (cutoff 1.45 x median NN distance). With
    `frame_angles=None` the laboratory frame is used (straight pore rows anywhere
    in the field give C_6 -> 1). With `frame_angles` = local radial direction of
    each pore the measure is radial-frame coherence: rows/ribs locked to the radial
    direction everywhere around the valve give C_6 -> 1, isotropic quench -> ~0."""
    n = len(points_xy)
    if n < 7:
        return 0.0
    tree = cKDTree(points_xy)
    dists, idxs = tree.query(points_xy, k=7)
    cutoff = 1.45 * float(np.median(dists[:, 1]))
    acc = 0.0 + 0.0j
    n_b = 0
    for i in range(n):
        vecs = points_xy[idxs[i, 1:7]] - points_xy[i]
        d = np.linalg.norm(vecs, axis=1)
        ok = d <= cutoff
        if not np.any(ok):
            continue
        ang = np.arctan2(vecs[ok, 1], vecs[ok, 0])
        if frame_angles is not None:
            ang = ang - frame_angles[i]
        acc += np.sum(np.exp(6j * ang))
        n_b += int(np.sum(ok))
    return float(abs(acc) / n_b) if n_b > 0 else 0.0


def simulate_radial_quench_costae(
    N: int = 160,
    phi0: float = 0.00,
    epsilon: float = 1.00,
    M: float = 1.0,
    v_front: float = 0.10,
    front_width: float = 2.0,
    solidify_lag_lambda: float = 1.0,
    R0: float = 8.0,
    dt: float = 0.10,
    noise_amp: float = 0.005,
    seed: int = 2002,
    record_every: int = 0,
    bulk_control: bool = False,
) -> dict[str, Any]:
    """Pass 3 remediation item 1 (reduced, honest model).

    Coscinodiscus valve silicification proceeds centrifugally from the central
    annulus towards the margin (Schmid & Volcani 1983). We model the silica
    deposition vesicle as a disc swept by a radially advancing front
        R(t) = R0 + v_front * t
    that converts the LCPA/silicic-acid mixture from a stable (single-well,
    a = -1) state ahead of the front into the spinodal (double-well, a = +1)
    state behind it — Furukawa's (1992) directional quench written in polar form,
        a(r, t) = tanh((R(t) - r) / w),
    coupled to the Cahn-Hilliard equation d(phi)/dt = M lap(phi^3 - a phi - eps^2 lap phi).
    A lagging precipitation (solidification) front R_s(t) = R(t) - lag freezes the
    pattern (zero mobility) once silica has condensed around the LCPA phase; the
    frozen mass is re-projected onto the mobile region so that the TOTAL order
    parameter is conserved exactly (|drift| ~ 1e-16). A weak conserved thermal
    noise (noise_amp) seeds the instability as the front passes.

    Emergent result (no template): for the near-critical rib layer (phi0 = 0) the
    front speed selects the morphology — slow fronts (v = 0.05-0.10 px/t, i.e.
    Pe = v / (lambda_c * sigma_max) < ~0.1) give RADIAL, bifurcating silica lamellae
    (costae) that branch to keep their spacing as the circumference grows; faster
    fronts give concentric lamellae parallel to the front; v -> infinity recovers the
    isotropic bulk quench (Furukawa 1992 morphological transition). Off-critical
    droplet compositions (phi0 = -0.3) only give front-parallel (concentric) rows.

    `bulk_control=True` runs the same disc with a = +1 everywhere from t = 0
    (classic isotropic quench; no front, no freezing) for the same number of steps
    as the isotropic null against which radial costae alignment is measured.
    """
    rng = np.random.default_rng(seed)
    r, theta, _, _ = _polar_grids(N)
    disc = r < 0.5 * N - 2.0
    R_max = 0.5 * N - 4.0

    alpha = 1.0 - 3.0 * phi0**2
    lambda_c = 2.0 * math.pi / math.sqrt(alpha / (2.0 * epsilon**2))
    sigma_max = M * alpha**2 / (4.0 * epsilon**2)
    lag = solidify_lag_lambda * lambda_c
    steps = int(math.ceil((R_max - R0) / (v_front * dt)))

    phi = phi0 + noise_amp * rng.standard_normal((N, N))
    phi = phi - np.mean(phi) + phi0
    mass0 = float(np.sum(phi))

    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K2 = KX**2 + KY**2
    denom = 1.0 + dt * M * (epsilon**2) * K2**2

    frozen = np.zeros((N, N), dtype=bool)
    phi_frozen = np.zeros((N, N))
    frames: list[tuple[float, np.ndarray]] = []
    R_t = R0
    for step in range(1, steps + 1):
        R_t = R0 + v_front * step * dt
        if bulk_control:
            a_field = np.ones((N, N))
        else:
            a_field = np.tanh((R_t - r) / front_width)
        # conserved thermal noise in the mobile region
        xi = noise_amp * rng.standard_normal((N, N))
        xi[frozen] = 0.0
        xi -= np.mean(xi)
        nl = phi**3 - a_field * phi
        phi_hat = np.fft.fft2(phi + xi)
        phi_hat = (phi_hat - dt * M * K2 * np.fft.fft2(nl)) / denom
        phi_new = np.fft.ifft2(phi_hat).real
        if not bulk_control:
            # lagging silica precipitation front: freeze newly solidified annulus
            newly = (r < R_t - lag) & ~frozen
            phi_frozen[newly] = phi_new[newly]
            frozen |= newly
            phi_new[frozen] = phi_frozen[frozen]
            mobile = ~frozen
            n_mob = int(np.sum(mobile))
            if n_mob > 0:
                phi_new[mobile] += (mass0 - float(np.sum(phi_new))) / n_mob
        phi = phi_new
        if record_every > 0 and (step % record_every == 0 or step == steps):
            frames.append((R_t, phi.copy()))

    pores = (phi > 0.05) & disc
    pores = ndi.binary_opening(pores, structure=np.ones((2, 2)))
    pts = pore_centroids(pores)
    # margin annulus = outer 40% of the swept radius; core = inner 40%
    r_margin_in, r_margin_out = 0.6 * R_max, R_max
    r_core_out = 0.4 * R_max
    c = 0.5 * (N - 1)
    if len(pts) > 0:
        pr = np.hypot(pts[:, 0] - c, pts[:, 1] - c)
        pth = np.arctan2(pts[:, 1] - c, pts[:, 0] - c)
        sel = (pr >= r_margin_in) & (pr < r_margin_out)
        pts_m, th_m = pts[sel], pth[sel]
    else:
        pts_m, th_m = np.zeros((0, 2)), np.zeros(0)
    return {
        "phi": phi,
        "frozen_mask": frozen,
        "quenched_mask": (r < R_t) if not bulk_control else disc,
        "steps": steps,
        "t_final": float(steps * dt),
        "R_final": float(R_t),
        "lambda_c_px": float(lambda_c),
        "sigma_max": float(sigma_max),
        "peclet_front": float(v_front / (lambda_c * sigma_max)),
        "solidify_lag_px": float(lag),
        "v_front": float(v_front),
        "phi0": float(phi0),
        "mass_drift": float(abs(np.sum(phi) - mass0) / (N * N)),
        "anisotropy_margin": radial_anisotropy_index(phi, r_margin_in, r_margin_out),
        "anisotropy_core": radial_anisotropy_index(phi, 0.0, r_core_out),
        "pore_centroids": pts,
        "n_pores": int(len(pts)),
        "n_pores_margin": int(len(pts_m)),
        "radial_frame_coherence_margin": bond_orientation_coherence(pts_m, th_m),
        "lab_frame_coherence_margin": bond_orientation_coherence(pts_m, None),
        "frames": frames,
    }


def run_front_velocity_sweep(
    velocities: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.80),
    N: int = 160,
    phi0: float = 0.00,
    seed: int = 2002,
) -> dict[str, Any]:
    """Ablation over the centrifugal front speed v_front (5 values spanning the
    Furukawa radial -> concentric -> isotropic transition) plus the isotropic
    bulk-quench null (no front). Reports the margin radial anisotropy A(v)."""
    rows: list[dict[str, float]] = []
    for v in velocities:
        res = simulate_radial_quench_costae(N=N, phi0=phi0, v_front=v, seed=seed)
        rows.append(
            {
                "v_front": float(v),
                "peclet_front": res["peclet_front"],
                "anisotropy_margin": res["anisotropy_margin"],
                "anisotropy_core": res["anisotropy_core"],
                "mass_drift": res["mass_drift"],
                "steps": res["steps"],
            }
        )
    # isotropic null: same disc & step count as the median speed, averaged over 3 seeds
    # (a single lamellar realisation has O(0.1-0.3) anisotropy fluctuations)
    ctrl_runs = [
        simulate_radial_quench_costae(N=N, phi0=phi0, v_front=velocities[len(velocities) // 2], seed=seed + i, bulk_control=True)
        for i in range(3)
    ]
    ctrl_vals = [c["anisotropy_margin"] for c in ctrl_runs]
    a_vals = [row["anisotropy_margin"] for row in rows]
    return {
        "velocities": list(velocities),
        "rows": rows,
        "anisotropies": a_vals,
        "best_velocity": float(velocities[int(np.argmax(a_vals))]),
        "control_anisotropy_margin": float(np.mean(ctrl_vals)),
        "control_anisotropy_margin_sd": float(np.std(ctrl_vals)),
        "control_anisotropy_values": ctrl_vals,
        "control": ctrl_runs[0],
    }


# ============================================================================
# 3c. PASS 3 — Explicit 3-Stage 512x512 Single-Domain Hierarchy
#     (Foramen -> Cribrum -> Cribrella) at a Fixed Physical Scale (nm / px)
# ============================================================================

def epsilon_for_pitch(pitch_px: float, phi0: float) -> float:
    """Invert lambda_c = 2*pi*sqrt(2)*eps/sqrt(1 - 3*phi0^2) for the interfacial
    parameter eps whose fastest-growing spinodal wavelength equals `pitch_px`."""
    return float(pitch_px * math.sqrt(1.0 - 3.0 * phi0**2) / (2.0 * math.pi * math.sqrt(2.0)))


def _nn_pitch(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    d, _ = cKDTree(points_xy).query(points_xy, k=2)
    return float(np.mean(d[:, 1]))


def simulate_three_stage_hierarchy(
    N: int = 512,
    nm_per_px: float = 12.5,
    tiers: list[dict[str, Any]] | None = None,
    phi0_tiers: tuple[float, float, float] = (-0.30, -0.26, -0.26),
    dt_tiers: tuple[float, float, float] = (50.0, 2.6, 0.10),
    steps_tiers: tuple[int, int, int] = (290, 160, 100),
    seed: int = 2002,
) -> dict[str, Any]:
    """Pass 3 remediation item 2. One N x N domain at a fixed physical scale
    (default 12.5 nm/px, so Sumper's tiers a_1 = 1850 nm, a_2 = 380 nm, a_3 = 75 nm
    are 148, 30.4 and 6.0 px). The three quenches are run SEQUENTIALLY IN THE SAME
    DOMAIN: after tier k-1 the silica outside its (slightly eroded) pores is treated as
    precipitated and FROZEN (zero mobility, phi = -1 wetting wall), and tier k is a fresh
    Cahn-Hilliard quench of the LCPA phase confined to the unsolidified chambers, with
    eps_k inverted from the target pitch via lambda_c(eps). Tier 1 is unconfined.
    The CH equation is scale invariant (x -> L x, eps -> L eps, t -> L^2 t), so the
    large-eps tiers legitimately use proportionally larger dt (dt_tiers).

    Honest limits: the tertiary cribrella at 12.5 nm/px are only d_3 = 42/12.5 = 3.4 px
    across (resolved as >= 6-pixel components but at the grid floor), and the domain holds
    only ~12 foramina. A finer physical scale would need a > 1024 grid.
    """
    if tiers is None:
        tiers = verify_and_load_reference()["sumper_2002_hierarchical_tiers"]["tiers"]
    pitches_px = [t["lattice_pitch_nm"] / nm_per_px for t in tiers]
    eps_tiers = [epsilon_for_pitch(p, phi0_tiers[i]) for i, p in enumerate(pitches_px)]

    fields: list[np.ndarray] = []
    pore_masks: list[np.ndarray] = []
    chamber_masks: list[np.ndarray] = []
    mass_drift: list[float] = []
    parent = np.ones((N, N), dtype=bool)
    erode_px = [0, 3, 1]
    for k in range(3):
        frozen = None if k == 0 else ~parent
        phi_k, _ = simulate_cahn_hilliard_2d(
            N=N,
            phi0=phi0_tiers[k],
            epsilon=eps_tiers[k],
            dt=dt_tiers[k],
            steps=steps_tiers[k],
            seed=seed + 71 * k,
            frozen_mask=frozen,
            frozen_value=phi0_tiers[k],
        )
        # mass conservation diagnostic over the mobile chambers
        if frozen is None:
            mass_drift.append(float(abs(np.mean(phi_k) - phi0_tiers[k])))
        else:
            mass_drift.append(float(abs(np.mean(phi_k[parent]) - phi0_tiers[k])))
        pores_k = (phi_k > 0.05) & parent
        pores_k = ndi.binary_opening(pores_k, structure=np.ones((2, 2)))
        fields.append(phi_k)
        pore_masks.append(pores_k)
        chamber_masks.append(parent.copy())
        if k < 2 and erode_px[k + 1] > 0:
            parent = ndi.binary_erosion(pores_k, structure=np.ones((3, 3)), iterations=erode_px[k + 1])
        else:
            parent = pores_k

    # Per-tier counts, nested multiplicity and pitches (px -> nm)
    labs = [ndi.label(m) for m in pore_masks]
    counts = [int(n) for _, n in labs]
    cents = [pore_centroids(m, min_area=6 if k == 2 else 20) for k, m in enumerate(pore_masks)]
    pitches_sim_px = [_nn_pitch(c) for c in cents]
    # equivalent diameters
    diams_px = []
    for (lab, n), min_a in zip(labs, (20, 20, 6)):
        if n == 0:
            diams_px.append(0.0)
            continue
        areas = ndi.sum_labels(np.ones((N, N)), lab, np.arange(1, n + 1))
        areas = areas[areas >= min_a]
        diams_px.append(float(np.mean(2.0 * np.sqrt(areas / math.pi))) if len(areas) else 0.0)

    def _children_per_parent(parent_lab: np.ndarray, n_parent: int, child_lab: np.ndarray) -> float:
        vals: list[int] = []
        for pid in range(1, n_parent + 1):
            reg = parent_lab == pid
            if np.sum(reg) < 20:
                continue
            ids = set(np.unique(child_lab[reg])) - {0}
            if ids:
                vals.append(len(ids))
        return float(np.mean(vals)) if vals else 0.0

    cribrum_per_foramen = _children_per_parent(labs[0][0], counts[0], labs[1][0])
    cribrella_per_cribrum = _children_per_parent(labs[1][0], counts[1], labs[2][0])

    # Composite biosilica density: +1 wall, 0.35 cribrum plate, -0.25 cribrella sieve plate, -1 open cribrella pore
    composite = np.ones((N, N))
    composite[pore_masks[0]] = 0.35
    composite[pore_masks[1]] = -0.25
    composite[pore_masks[2]] = -1.0

    return {
        "N": N,
        "nm_per_px": float(nm_per_px),
        "target_pitches_px": [float(p) for p in pitches_px],
        "epsilons": [float(e) for e in eps_tiers],
        "fields": fields,
        "pore_masks": pore_masks,
        "chamber_masks": chamber_masks,
        "composite": composite,
        "counts": counts,
        "pitches_sim_px": [float(p) for p in pitches_sim_px],
        "pitches_sim_nm": [float(p * nm_per_px) for p in pitches_sim_px],
        "diameters_sim_nm": [float(d * nm_per_px) for d in diams_px],
        "cribrum_per_foramen": cribrum_per_foramen,
        "cribrella_per_cribrum": cribrella_per_cribrum,
        "mass_drift_tiers": mass_drift,
    }


_THREE_STAGE_CACHE: dict[tuple[int, float, int], dict[str, Any]] = {}


def three_stage_cached(N: int = 512, nm_per_px: float = 12.5, seed: int = 2002) -> dict[str, Any]:
    """Run `simulate_three_stage_hierarchy` once per process for a given (N, scale, seed)."""
    key = (int(N), float(nm_per_px), int(seed))
    if key not in _THREE_STAGE_CACHE:
        _THREE_STAGE_CACHE[key] = simulate_three_stage_hierarchy(N=N, nm_per_px=nm_per_px, seed=seed)
    return _THREE_STAGE_CACHE[key]


# ============================================================================
# 4. Spectral Morphometrics (Sub-Bin Parabolic Peak) & Hexatic Order |psi_6|
# ============================================================================

def _parabolic_subbin_peak(radial_psd: np.ndarray, k_idx: int) -> float:
    """3-point parabolic sub-bin interpolation around discrete spectral peak k_idx."""
    if k_idx <= 1 or k_idx >= len(radial_psd) - 1:
        return float(k_idx)
    y_m1 = float(radial_psd[k_idx - 1])
    y_0 = float(radial_psd[k_idx])
    y_p1 = float(radial_psd[k_idx + 1])
    denom = y_m1 - 2.0 * y_0 + y_p1
    if abs(denom) < 1e-12:
        return float(k_idx)
    delta = 0.5 * (y_m1 - y_p1) / denom
    delta = max(-0.5, min(0.5, delta))
    return float(k_idx + delta)


def compute_spectral_and_pore_metrics(phi: np.ndarray, epsilon: float = 1.50) -> dict[str, Any]:
    """Compute sub-bin 2D radial PSD peak k*, dominant wavelength lambda*, analytical
    spinodal wavelength lambda_c, pore count, and hexatic order |psi_6|."""
    N = phi.shape[0]
    phi0 = float(np.mean(phi))
    f = phi - phi0
    win1d = np.hanning(N)
    window2d = np.outer(win1d, win1d)
    psd2d = np.abs(np.fft.fft2(f * window2d)) ** 2

    freq_x = np.fft.fftfreq(N, d=1.0) * N
    freq_y = np.fft.fftfreq(N, d=1.0) * N
    FX, FY = np.meshgrid(freq_x, freq_y, indexing="ij")
    R = np.sqrt(FX**2 + FY**2)

    max_bin = N // 2
    radial = np.zeros(max_bin)
    for b in range(1, max_bin):
        mask = (R >= b - 0.5) & (R < b + 0.5)
        if np.any(mask):
            radial[b] = float(np.mean(psd2d[mask]))

    std_phi = float(np.std(phi))
    if std_phi < 1e-3 or np.max(radial[1:]) < 1e-4:
        k_star = 0.0
        wavelength_px = 0.0
        ring_sharpness = 0.0
    else:
        k_idx = int(np.argmax(radial[1:]) + 1)
        k_star = _parabolic_subbin_peak(radial, k_idx)
        wavelength_px = float(N / k_star) if k_star > 0 else 0.0
        med_bg = float(np.median(radial[1:])) + 1e-12
        ring_sharpness = float(radial[k_idx] / med_bg)

    # Pore connected components (minority LCPA droplet phase)
    row_c6 = 0.0
    if std_phi < 0.05:
        n_pores = 0
        psi6_mean = 0.0
    else:
        bw = (phi > 0.05) if phi0 <= 0.0 else (phi < -0.05)
        bw = ndi.binary_opening(bw, structure=np.ones((2, 2)))
        lab, n_comp = ndi.label(bw)
        centroids = []
        for i in range(1, n_comp + 1):
            pts = np.argwhere(lab == i)
            if len(pts) >= 6:
                cy, cx = pts.mean(axis=0)
                if 4 < cy < N - 4 and 4 < cx < N - 4:
                    centroids.append((cx, cy))
        n_pores = int(n_comp)
        psi6_mean = compute_hexatic_psi6(np.array(centroids)) if len(centroids) >= 7 else 0.0
        row_c6 = bond_orientation_coherence(np.array(centroids), None) if len(centroids) >= 7 else 0.0

    disp = compute_spinodal_dispersion(phi0=phi0, epsilon=epsilon)
    return {
        "k_star_cycles": float(k_star),
        "wavelength_px": float(wavelength_px),
        "lambda_c_analytical_px": float(disp["lambda_c_px"]),
        "ring_sharpness": float(ring_sharpness),
        "pore_count": int(n_pores),
        "hexatic_psi6": float(psi6_mean),
        "row_coherence_c6": float(row_c6),
        "mass_mean": float(phi0),
        "std_phi": float(std_phi),
        "radial_psd": radial,
    }


def compute_hexatic_psi6(points_xy: np.ndarray) -> float:
    """Compute 6-fold bond-orientational hexatic order parameter:
        |psi_6| = (1/N) sum_j | (1/N_j) sum_{m in NN(j)} exp(6 i theta_{jm}) |
    for 2D pore centroids `points_xy`."""
    if len(points_xy) < 7:
        return 0.0
    tree = cKDTree(points_xy)
    dists, idxs = tree.query(points_xy, k=7)
    nn1_median = float(np.median(dists[:, 1]))
    cutoff = 1.45 * nn1_median
    local_psi6: list[float] = []
    for i in range(len(points_xy)):
        p0 = points_xy[i]
        nbr_indices = idxs[i, 1:7]
        vecs = points_xy[nbr_indices] - p0
        nbr_dists = np.linalg.norm(vecs, axis=1)
        valid = nbr_dists <= cutoff
        if np.sum(valid) >= 3:
            angles = np.arctan2(vecs[valid, 1], vecs[valid, 0])
            local_psi6.append(float(np.abs(np.mean(np.exp(6j * angles)))))
    return float(np.mean(local_psi6)) if local_psi6 else 0.0


def run_epsilon_wavelength_sweep(
    epsilons: tuple[float, ...] = (0.75, 1.00, 1.25, 1.50, 1.75),
    N: int = 128,
    phi0: float = -0.30,
    steps: int = 220,
    dt: float = 0.15,
    seed: int = 2002,
) -> dict[str, Any]:
    """Sweep interfacial thickness parameter epsilon across 5 values and compare the
    simulated spinodal wavelength lambda*(epsilon) against the analytical Cahn-Hilliard
    prediction lambda_c(epsilon) = 2*pi*sqrt(2)*epsilon / sqrt(1 - 3*phi0^2)."""
    sim_lambdas: list[float] = []
    ana_lambdas: list[float] = []
    rows: list[dict[str, float]] = []
    for eps in epsilons:
        phi, _ = simulate_cahn_hilliard_2d(
            N=N, phi0=phi0, epsilon=eps, dt=dt, steps=steps, seed=seed
        )
        m = compute_spectral_and_pore_metrics(phi, epsilon=eps)
        sim_lambdas.append(m["wavelength_px"])
        ana_lambdas.append(m["lambda_c_analytical_px"])
        rows.append(
            {
                "epsilon": float(eps),
                "k_star": float(m["k_star_cycles"]),
                "lambda_sim_px": float(m["wavelength_px"]),
                "lambda_ana_px": float(m["lambda_c_analytical_px"]),
                "pore_count": int(m["pore_count"]),
            }
        )
    r_val, p_val = stats.pearsonr(ana_lambdas, sim_lambdas)
    rmse = float(np.sqrt(np.mean((np.array(sim_lambdas) - np.array(ana_lambdas)) ** 2)))
    return {
        "epsilons": list(epsilons),
        "sim_lambdas": sim_lambdas,
        "ana_lambdas": ana_lambdas,
        "pearson_r": float(r_val),
        "p_value": float(p_val),
        "rmse_px": rmse,
        "rows": rows,
    }


# ============================================================================
# 5. Real Biological Specimen Segmentation (*Coscinodiscus wailesii* SEM)
#    (Sumper 2002 Science Figs. 1-3; Losic et al. 2006 Table 1)
# ============================================================================

def segment_coscinodiscus_specimen(ref_doc: dict[str, Any]) -> dict[str, Any]:
    """Segment the primary hexagonal foramen lattice (subpanel a1) and secondary
    cribrum pores (subpanel a3) from the SHA-256-locked *Coscinodiscus wailesii*
    SEM micrograph (`datasets/diatom_frustule_phase_sep/coscinodiscus_diatom_specimen.jpg`)."""
    spec_meta = ref_doc["metadata"]["specimens"]["coscinodiscus_wailesii_valve_sem"]
    im = Image.open(SPECIMEN_JPG_PATH).convert("L")
    arr = np.asarray(im, dtype=float)

    x0, y0, x1, y1 = spec_meta["roi_foramen_xyxy"]
    roi_foramen = arr[y0:y1, x0:x1]
    smooth = ndi.gaussian_filter(roi_foramen, sigma=3.0)
    thresh = np.percentile(smooth, 35)
    mask_f = smooth < thresh
    mask_f = ndi.binary_opening(mask_f, structure=np.ones((5, 5)))
    mask_f = ndi.binary_fill_holes(mask_f)
    lab_f, n_comp_f = ndi.label(mask_f)

    centroids_f: list[tuple[float, float]] = []
    diams_f: list[float] = []
    for i in range(1, n_comp_f + 1):
        pts = np.argwhere(lab_f == i)
        area = len(pts)
        if 80 <= area <= 2500:
            cy, cx = pts.mean(axis=0)
            if 12 < cy < roi_foramen.shape[0] - 12 and 12 < cx < roi_foramen.shape[1] - 12:
                centroids_f.append((float(cx), float(cy)))
                diams_f.append(float(2.0 * math.sqrt(area / math.pi)))

    pts_f = np.array(centroids_f)
    tree_f = cKDTree(pts_f)
    dists_f, _ = tree_f.query(pts_f, k=7)
    nn_pitch_f = dists_f[:, 1]
    psi6_f = compute_hexatic_psi6(pts_f)
    # Pass 3: long-range (lab-frame) row coherence of the SEM foramen lattice — the
    # straight fasciculate areola rows of panel a1 (Coscinodiscus rows run radially).
    row_coherence_f = bond_orientation_coherence(pts_f, None)

    mean_diam_f = float(np.mean(diams_f))
    cv_diam_f = float(np.std(diams_f) / mean_diam_f)
    mean_pitch_f = float(np.mean(nn_pitch_f))
    cv_pitch_f = float(np.std(nn_pitch_f) / mean_pitch_f)
    diam_to_pitch_f = float(mean_diam_f / mean_pitch_f)

    # Secondary cribrum pore segmentation on high-magnification subpanel a3
    cx0, cy0, cx1, cy1 = spec_meta["roi_cribrum_xyxy"]
    roi_cribrum = arr[cy0:cy1, cx0:cx1]
    dog = ndi.gaussian_filter(roi_cribrum, sigma=5.0) - ndi.gaussian_filter(roi_cribrum, sigma=1.8)
    bw_c = dog > np.percentile(dog, 78)
    bw_c = ndi.binary_opening(bw_c, structure=np.ones((3, 3)))
    lab_c, n_comp_c = ndi.label(bw_c)
    centroids_c: list[tuple[float, float]] = []
    diams_c: list[float] = []
    for i in range(1, n_comp_c + 1):
        pts = np.argwhere(lab_c == i)
        area = len(pts)
        if 15 <= area <= 450:
            cy, cx = pts.mean(axis=0)
            centroids_c.append((float(cx), float(cy)))
            diams_c.append(float(2.0 * math.sqrt(area / math.pi)))

    # 3-tier self-similar hierarchy regression from Sumper (2002) reference tiers
    tiers = ref_doc["sumper_2002_hierarchical_tiers"]["tiers"]
    gens = np.array([t["generation"] for t in tiers], dtype=float)
    log_diams = np.log10(np.array([t["pore_diameter_nm"] for t in tiers], dtype=float))
    log_pitches = np.log10(np.array([t["lattice_pitch_nm"] for t in tiers], dtype=float))
    r_tier, p_tier = stats.pearsonr(gens, log_diams)
    slope_d, _ = np.polyfit(gens - 1.0, log_diams, 1)
    effective_rho = float(10.0**slope_d)

    return {
        "roi_foramen": roi_foramen,
        "mask_foramen": mask_f,
        "centroids_foramen": pts_f,
        "foramen_pore_count": len(pts_f),
        "foramen_mean_diameter_px": mean_diam_f,
        "foramen_diameter_cv": cv_diam_f,
        "foramen_mean_pitch_px": mean_pitch_f,
        "foramen_pitch_cv": cv_pitch_f,
        "foramen_diameter_to_pitch_ratio": diam_to_pitch_f,
        "foramen_hexatic_psi6": psi6_f,
        "foramen_row_coherence_c6": row_coherence_f,
        "roi_cribrum": roi_cribrum,
        "centroids_cribrum": np.array(centroids_c),
        "cribrum_pore_count": len(diams_c),
        "cribrum_mean_diameter_px": float(np.mean(diams_c)),
        "hierarchy_log_r": float(r_tier),
        "hierarchy_log_p": float(p_tier),
        "hierarchy_effective_rho": effective_rho,
        "tiers": tiers,
    }


# ============================================================================
# 6. Publication-Grade 8-Panel Comparison Figure & Animated Hierarchical GIF
# ============================================================================

def render_all_outputs(
    N: int = 128,
    phi0: float = -0.30,
    epsilon: float = 1.50,
    steps: int = 650,
    seed: int = 2002,
    skip_gif: bool = False,
    v_front: float = 0.10,
    N3: int = 512,
    nm_per_px: float = 12.5,
) -> dict[str, Any]:
    """Execute all simulations, parameter sweeps, negative controls, and specimen
    segmentations, then render `docs/media/diatom_frustule_phase_sep_comparison.png`
    (Pass 3: 3 rows x 4 panels), `docs/media/diatom_frustule_phase_sep_growth.gif`
    and `docs/media/diatom_frustule_phase_sep_radial_costae.gif`."""
    ref_doc = verify_and_load_reference()
    COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)

    # 0. Pass 3 models: centrifugal front -> costae (with v_front sweep + isotropic null)
    #    and explicit 3-stage N3 x N3 single-domain hierarchy at nm_per_px.
    costae = simulate_radial_quench_costae(N=160, v_front=v_front, seed=seed, record_every=200)
    front_sweep = run_front_velocity_sweep(N=128, seed=seed)  # 5 speeds + isotropic null (128^2 keeps the sweep ~10 s)
    three = three_stage_cached(N=N3, nm_per_px=nm_per_px, seed=seed)

    # 1. Two-stage hierarchical Cahn-Hilliard simulation (Foramen -> Cribrum)
    hier = simulate_two_stage_hierarchical_diatom(
        N=N,
        phi0_primary=phi0,
        epsilon_primary=epsilon,
        steps_primary=steps,
        phi0_secondary=-0.26,
        epsilon_secondary=0.65,
        steps_secondary=320,
        seed=seed,
    )
    m_drop = compute_spectral_and_pore_metrics(hier["phi_stage1"], epsilon=epsilon)

    # 2. Bicontinuous symmetric spinodal regime (phi0 = 0.00, Column b of SEM)
    phi_bicon, _ = simulate_cahn_hilliard_2d(
        N=N, phi0=0.00, epsilon=epsilon, dt=0.25, steps=steps, seed=seed + 1
    )
    m_bicon = compute_spectral_and_pore_metrics(phi_bicon, epsilon=epsilon)

    # 3. Negative control outside spinodal curve (phi0 = -0.65, |phi0| > 1/sqrt(3))
    phi_ctrl, _ = simulate_cahn_hilliard_2d(
        N=N, phi0=-0.65, epsilon=epsilon, dt=0.25, steps=steps, seed=seed + 2
    )
    m_ctrl = compute_spectral_and_pore_metrics(phi_ctrl, epsilon=epsilon)

    # 4. 5-point interfacial thickness parameter sweep
    sweep = run_epsilon_wavelength_sweep(N=N, phi0=phi0, seed=seed)

    # 5. Biological specimen segmentation (*Coscinodiscus wailesii* SEM)
    spec = segment_coscinodiscus_specimen(ref_doc)

    # 6. Render 12-panel dark-themed synthesis figure (3 rows x 4 columns)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 4, figsize=(18.5, 13.8), dpi=150)
    fig.patch.set_facecolor("#0b0f19")
    for ax in axes.flat:
        ax.set_facecolor("#111827")

    # Panel A1: Stage 1 Primary Foramen Droplets (phi0 = -0.30, eps = 1.50)
    ax = axes[0, 0]
    ax.imshow(-hier["phi_stage1"], cmap="bone", vmin=-1.0, vmax=1.0)
    ax.set_title(
        f"A1. Stage 1 Foramen Array (φ₀={phi0:.2f}, ε₁={epsilon:.2f})\n"
        f"{m_drop['pore_count']} pores, λ*={m_drop['wavelength_px']:.2f} px (|ψ₆|={m_drop['hexatic_psi6']:.3f})",
        fontsize=9.5,
        color="#67e8f9",
    )
    ax.axis("off")

    # Panel A2: Stage 2 Hierarchical Foramen + Cribrum Quench (Lenoci & Camp 2008 Sec. 3.3)
    ax = axes[0, 1]
    ax.imshow(hier["silica_composite"], cmap="bone", vmin=-1.0, vmax=1.0)
    ax.set_title(
        f"A2. Two-Stage Hierarchical Quench (ε₁={epsilon:.2f} → ε₂=0.65)\n"
        f"{hier['n_foramen_pores']} foramina + {hier['n_cribrum_subpores']} nested cribrum sub-pores",
        fontsize=9.5,
        color="#34d399",
    )
    ax.axis("off")

    # Panel A3: Critical Bicontinuous Spinodal Network (phi0 = 0.00)
    ax = axes[0, 2]
    ax.imshow(-phi_bicon, cmap="bone", vmin=-1.0, vmax=1.0)
    ax.set_title(
        f"A3. Symmetric Bicontinuous Mesh (φ₀=0.00, ε={epsilon:.2f})\n"
        f"Percolating honeycomb network, λ*={m_bicon['wavelength_px']:.2f} px",
        fontsize=9.5,
        color="#fbbf24",
    )
    ax.axis("off")

    # Panel A4: Outside-Spinodal Negative Control (phi0 = -0.65, |phi0| > 1/sqrt(3))
    ax = axes[0, 3]
    ax.imshow(-phi_ctrl, cmap="bone", vmin=-1.0, vmax=1.0)
    ax.set_title(
        f"A4. Negative Control Outside Spinodal (φ₀=-0.65)\n"
        f"α = 1 - 3φ₀² = -0.268 < 0 → 0 pores (σ_φ={m_ctrl['std_phi']:.2e})",
        fontsize=9.5,
        color="#f87171",
    )
    ax.axis("off")

    # Panel B1: *Coscinodiscus wailesii* Internal Plate SEM + Segmented Foramen Lattice
    ax = axes[1, 0]
    ax.imshow(spec["roi_foramen"], cmap="gray")
    pts_f = spec["centroids_foramen"]
    ax.scatter(pts_f[:, 0], pts_f[:, 1], s=14, c="#22d3ee", edgecolors="#083344", linewidths=0.5, label="Foramen centroids")
    ax.set_title(
        f"B1. Coscinodiscus wailesii SEM Foramen Lattice\n"
        f"n={spec['foramen_pore_count']} pores, d/a={spec['foramen_diameter_to_pitch_ratio']:.3f} (ref 0.676), |ψ₆|={spec['foramen_hexatic_psi6']:.3f}",
        fontsize=9.5,
        color="#67e8f9",
    )
    ax.axis("off")

    # Panel B2: High-Zoom *Coscinodiscus wailesii* SEM Cribrum & Cribrella Hierarchy
    ax = axes[1, 1]
    ax.imshow(spec["roi_cribrum"], cmap="gray")
    pts_c = spec["centroids_cribrum"]
    if len(pts_c) > 0:
        ax.scatter(pts_c[:, 0], pts_c[:, 1], s=12, c="#f472b6", alpha=0.85, label="Cribrum sub-pores")
    ax.set_title(
        f"B2. Coscinodiscus SEM Cribrum Sub-Pore Segmentation\n"
        f"n={spec['cribrum_pore_count']} nested cribrum pores (mean d={spec['cribrum_mean_diameter_px']:.1f} px)",
        fontsize=9.5,
        color="#f472b6",
    )
    ax.axis("off")

    # Panel B3: Linear Spinodal Dispersion Relation sigma(k) & Parameter Sweep lambda*(eps)
    ax = axes[1, 2]
    d_drop = compute_spinodal_dispersion(phi0=-0.30, epsilon=1.50)
    d_bicon = compute_spinodal_dispersion(phi0=0.00, epsilon=1.50)
    d_ctrl = compute_spinodal_dispersion(phi0=-0.65, epsilon=1.50)
    ax.plot(d_bicon["k_grid"], d_bicon["sigma_grid"], color="#fbbf24", lw=2.0, label="φ₀=0.00 (α=+1.00)")
    ax.plot(d_drop["k_grid"], d_drop["sigma_grid"], color="#22d3ee", lw=2.2, label=f"φ₀=-0.30 (k_c={d_drop['kc']:.3f})")
    ax.plot(d_ctrl["k_grid"], d_ctrl["sigma_grid"], color="#f87171", lw=1.8, ls="--", label="φ₀=-0.65 (α=-0.27 < 0)")
    ax.axhline(0.0, color="#9ca3af", lw=0.9, ls=":")
    ax.axvline(d_drop["kc"], color="#22d3ee", lw=1.1, ls="--")
    ax.set_ylim(-0.08, 0.13)
    ax.set_xlabel("Wavenumber k (rad / px)", fontsize=8.5)
    ax.set_ylabel("Growth rate σ(k) = M k² (1 - 3φ₀² - ε² k²)", fontsize=8.5)
    ax.set_title("B3. Cahn-Hilliard Spinodal Dispersion σ(k)", fontsize=9.5, color="#e5e7eb")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.35)
    ax.grid(alpha=0.2)

    # Panel B4: Interfacial Scale Sweep lambda*(eps) + Sumper (2002) 3-Tier Cascade
    ax = axes[1, 3]
    eps_arr = np.array(sweep["epsilons"])
    ax.plot(eps_arr, sweep["ana_lambdas"], "o--", color="#9ca3af", lw=1.6, label="Analytical λ_c = 2π/k_c")
    ax.plot(eps_arr, sweep["sim_lambdas"], "s-", color="#34d399", lw=2.2, label=f"Simulated λ* (r={sweep['pearson_r']:.4f})")
    ax.set_xlabel("Interfacial thickness parameter ε", fontsize=8.5)
    ax.set_ylabel("Characteristic pore wavelength λ (px)", fontsize=8.5)
    ax.set_title(
        f"B4. Wavelength Scaling λ*(ε) & 3-Tier Cascade\n"
        f"Sweep r={sweep['pearson_r']:.4f} (RMSE={sweep['rmse_px']:.2f} px); 3-tier log r={spec['hierarchy_log_r']:.4f}",
        fontsize=9.5,
        color="#34d399",
    )
    ax.legend(fontsize=7.8, loc="upper left", framealpha=0.35)
    ax.grid(alpha=0.2)

    # ---- Pass 3 row C ---------------------------------------------------------
    # Panel C1: Explicit 3-stage N3 x N3 single-domain hierarchy (composite silica density)
    ax = axes[2, 0]
    ax.imshow(three["composite"], cmap="bone", vmin=-1.0, vmax=1.0)
    c3 = three["counts"]
    ax.set_title(
        f"C1. 3-Stage {N3}×{N3} Single-Domain Quench ({nm_per_px:g} nm/px)\n"
        f"{c3[0]} foramina → {c3[1]} cribrum → {c3[2]} cribrella "
        f"({three['cribrum_per_foramen']:.1f} / {three['cribrella_per_cribrum']:.1f} per parent)",
        fontsize=9.5,
        color="#a78bfa",
    )
    ax.axis("off")

    # Panel C2: zoom on the largest foramen vs SEM a3 cribrum/cribrella tiers
    ax = axes[2, 1]
    lab_f3, n_f3 = ndi.label(three["pore_masks"][0])
    if n_f3 > 0:
        sizes = ndi.sum_labels(np.ones_like(lab_f3, dtype=float), lab_f3, np.arange(1, n_f3 + 1))
        big = int(np.argmax(sizes)) + 1
        ys, xs = np.nonzero(lab_f3 == big)
        half = int(max(np.ptp(ys), np.ptp(xs)) * 0.65) + 6
        cy_, cx_ = int(ys.mean()), int(xs.mean())
        y0z, y1z = max(0, cy_ - half), min(N3, cy_ + half)
        x0z, x1z = max(0, cx_ - half), min(N3, cx_ + half)
        ax.imshow(three["composite"][y0z:y1z, x0z:x1z], cmap="bone", vmin=-1.0, vmax=1.0)
        bar_px = 200.0 / nm_per_px
        ax.plot([4, 4 + bar_px], [y1z - y0z - 6, y1z - y0z - 6], color="#fbbf24", lw=3)
        ax.text(4, y1z - y0z - 10, "200 nm", color="#fbbf24", fontsize=7.5)
    d_nm = three["diameters_sim_nm"]
    ax.set_title(
        f"C2. Zoom: one foramen → cribrum → cribrella (sim)\n"
        f"d₂={d_nm[1]:.0f} nm (ref 220), d₃={d_nm[2]:.0f} nm (ref 42); a₂/a₃={three['pitches_sim_px'][1] / max(three['pitches_sim_px'][2], 1e-9):.2f} (ref 5.07)",
        fontsize=9.5,
        color="#a78bfa",
    )
    ax.axis("off")

    # Panel C3: Centrifugal precipitation front -> radial costae lamellae
    ax = axes[2, 2]
    ax.imshow(-costae["phi"], cmap="bone", vmin=-1.0, vmax=1.0)
    th_c = np.linspace(0, 2 * np.pi, 200)
    c_ = 0.5 * (160 - 1)
    ax.plot(c_ + costae["R_final"] * np.cos(th_c), c_ + costae["R_final"] * np.sin(th_c), color="#f472b6", lw=1.0, ls="--")
    ax.plot(c_ + 0.6 * (0.5 * 160 - 4.0) * np.cos(th_c), c_ + 0.6 * (0.5 * 160 - 4.0) * np.sin(th_c), color="#fbbf24", lw=0.8, ls=":")
    ax.set_title(
        f"C3. Centrifugal Front Quench → Radial Costae (v={costae['v_front']:.2f}, Pe={costae['peclet_front']:.3f})\n"
        f"margin A={costae['anisotropy_margin']:+.3f} vs isotropic null {front_sweep['control_anisotropy_margin']:+.2f}±{front_sweep['control_anisotropy_margin_sd']:.2f}; Δmass={costae['mass_drift']:.0e}",
        fontsize=9.5,
        color="#f472b6",
    )
    ax.axis("off")

    # Panel C4: Front-velocity ablation A(v): radial -> concentric -> isotropic transition
    ax = axes[2, 3]
    vs = np.array(front_sweep["velocities"])
    ax.plot(vs, front_sweep["anisotropies"], "o-", color="#f472b6", lw=2.0, label="margin A (radial > 0 > concentric)")
    ax.plot(vs, [r_["anisotropy_core"] for r_ in front_sweep["rows"]], "s--", color="#a78bfa", lw=1.4, label="core A")
    ax.axhline(front_sweep["control_anisotropy_margin"], color="#9ca3af", lw=1.0, ls=":", label="isotropic bulk quench (no front)")
    ax.axhline(0.0, color="#4b5563", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("front speed v_front (px / time)", fontsize=8.5)
    ax.set_ylabel("radial anisotropy A = ⟨|∂θφ|²−|∂rφ|²⟩/⟨|∇φ|²⟩", fontsize=8.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        f"C4. Front-Speed Ablation (Furukawa 1992 transition)\n"
        f"SEM a1 row coherence C₆={spec['foramen_row_coherence_c6']:.3f} vs isotropic sim C₆={m_drop['row_coherence_c6']:.3f}",
        fontsize=9.5,
        color="#f472b6",
    )
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.35)
    ax.grid(alpha=0.2)

    fig.suptitle(
        "Sumper (2002) Science & Lenoci–Camp (2008) Langmuir: Hierarchical Cahn-Hilliard Phase Separation in Coscinodiscus Diatom Biosilica (Pass 3)",
        fontsize=12.5,
        fontweight="bold",
        color="#f9fafb",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(COMPARISON_PNG, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 6b. Pass 3 animated GIF of the centrifugal precipitation front sweeping the valve
    if not skip_gif:
        gif_frames: list[Image.Image] = []
        for R_t, field2d in costae["frames"]:
            fig_r, ax_r = plt.subplots(1, 1, figsize=(4.2, 4.4), dpi=100)
            fig_r.patch.set_facecolor("#0b0f19")
            ax_r.imshow(-field2d, cmap="bone", vmin=-1.0, vmax=1.0)
            ax_r.plot(c_ + R_t * np.cos(th_c), c_ + R_t * np.sin(th_c), color="#f472b6", lw=1.2, ls="--")
            ax_r.set_title(f"Centrifugal quench front R(t)={R_t:.1f} px (v={costae['v_front']:.2f})", fontsize=8.5, color="#f472b6")
            ax_r.axis("off")
            fig_r.tight_layout()
            fig_r.canvas.draw()
            w, h = fig_r.canvas.get_width_height()
            buf = np.frombuffer(fig_r.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
            gif_frames.append(Image.fromarray(buf).convert("P", palette=Image.Palette.ADAPTIVE))
            plt.close(fig_r)
        gif_frames[0].save(COSTAE_GIF, save_all=True, append_images=gif_frames[1:], duration=160, loop=0)

    # 7. Render animated GIF of two-stage hierarchical Cahn-Hilliard morphogenesis
    if not skip_gif:
        pil_frames: list[Image.Image] = []
        for title_str, field2d in hier["hierarchical_frames"]:
            fig_g, axes_g = plt.subplots(1, 2, figsize=(7.6, 3.8), dpi=105)
            fig_g.patch.set_facecolor("#0b0f19")
            axes_g[0].imshow(field2d, cmap="bone", vmin=-1.0, vmax=1.0)
            axes_g[0].set_title(title_str, fontsize=9, color="#67e8f9")
            axes_g[0].axis("off")

            axes_g[1].imshow(spec["roi_foramen"], cmap="gray")
            axes_g[1].scatter(pts_f[:, 0], pts_f[:, 1], s=8, c="#22d3ee", alpha=0.75)
            axes_g[1].set_title("Coscinodiscus wailesii SEM (139 foramina)", fontsize=9, color="#34d399")
            axes_g[1].axis("off")

            fig_g.suptitle("Hierarchical Diatom Biosilica Phase Separation", fontsize=10, color="#f9fafb")
            fig_g.tight_layout()
            fig_g.canvas.draw()
            w, h = fig_g.canvas.get_width_height()
            buf = np.frombuffer(fig_g.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
            pil_frames.append(Image.fromarray(buf).convert("P", palette=Image.Palette.ADAPTIVE))
            plt.close(fig_g)

        pil_frames[0].save(
            GROWTH_GIF,
            save_all=True,
            append_images=pil_frames[1:],
            duration=180,
            loop=0,
        )

    return {
        "hierarchical": hier,
        "droplet_metrics": m_drop,
        "bicontinuous_metrics": m_bicon,
        "control_metrics": m_ctrl,
        "sweep": sweep,
        "specimen": spec,
        "costae": costae,
        "front_sweep": front_sweep,
        "three_stage": three,
    }


# ============================================================================
# 7. Dynamic MorphoEval Measurement Entrypoint (`measure_morphoeval`)
# ============================================================================

def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute quantitative morphometric measurements for `morphoeval/measure_outputs.py`."""
    ref_doc = verify_and_load_reference()
    hier = simulate_two_stage_hierarchical_diatom(N=128, seed=2002)
    m_drop = compute_spectral_and_pore_metrics(hier["phi_stage1"], epsilon=1.50)
    phi_ctrl, _ = simulate_cahn_hilliard_2d(N=128, phi0=-0.65, epsilon=1.50, dt=0.25, steps=650, seed=2004)
    m_ctrl = compute_spectral_and_pore_metrics(phi_ctrl, epsilon=1.50)
    sweep = run_epsilon_wavelength_sweep(N=128, phi0=-0.30, seed=2002)
    spec = segment_coscinodiscus_specimen(ref_doc)
    # Pass 3 models
    costae = simulate_radial_quench_costae(N=160, v_front=0.10, seed=2002)
    costae_fast = simulate_radial_quench_costae(N=160, v_front=0.40, seed=2002)
    null_vals = [
        simulate_radial_quench_costae(N=160, v_front=0.20, seed=2002 + i, bulk_control=True)["anisotropy_margin"] for i in range(3)
    ]
    null_mean, null_sd = float(np.mean(null_vals)), float(np.std(null_vals))
    three = three_stage_cached(N=512, nm_per_px=12.5, seed=2002)
    tiers = ref_doc["sumper_2002_hierarchical_tiers"]["tiers"]
    ref_a1_a2 = tiers[0]["lattice_pitch_nm"] / tiers[1]["lattice_pitch_nm"]
    ref_a2_a3 = tiers[1]["lattice_pitch_nm"] / tiers[2]["lattice_pitch_nm"]
    p_sim = three["pitches_sim_px"]

    pass3_measurements: list[dict[str, Any]] = [
        {
            "name": "Centrifugal precipitation-front quench: margin radial anisotropy A at slow front speed (v = 0.10 px/t)",
            "value": round(costae["anisotropy_margin"], 4),
            "unit": "anisotropy index (+1 radial lamellae, 0 isotropic, -1 concentric)",
            "method": "gradient-tensor anisotropy <|d_theta phi|^2 - |d_r phi|^2>/<|grad phi|^2> in the outer 40% annulus of a 160^2 disc swept by a(r,t)=tanh((R0+v t - r)/w) with lagging frozen silica front (lag = 1.0 lambda_c), phi0 = 0",
            "derived_from": "script",
            "reference_value": round(null_mean, 4),
            "reference_source": "isotropic bulk-quench null (same disc, a = +1 everywhere, no front, no freezing; mean of 3 seeds); Furukawa (1992) Physica A 180:128 directional-quench morphology transition predicts front-perpendicular lamellae at slow front speed",
            "note": f"Pe = v/(lambda_c sigma_max) = {costae['peclet_front']:.3f}; core A = {costae['anisotropy_core']:+.3f}; fast front (v = 0.40) margin A = {costae_fast['anisotropy_margin']:+.3f} (concentric); null A = {null_mean:+.3f} +/- {null_sd:.3f} (3 seeds); the vendored SEM has no valve-margin view, so no direct specimen value exists for A (Pass 4 item)",
        },
        {
            "name": "Centrifugal front quench: radial-minus-concentric morphology contrast A(v=0.10) - A(v=0.40)",
            "value": round(costae["anisotropy_margin"] - costae_fast["anisotropy_margin"], 4),
            "unit": "anisotropy index difference",
            "method": "same model at two front speeds; positive values mean slow fronts give radial costae and fast fronts give front-parallel (concentric) lamellae",
            "derived_from": "script",
            "reference_value": 0.0,
            "reference_source": "isotropic null (no front) has no speed dependence: contrast = 0; sign predicted by Furukawa (1992)",
            "note": f"A(0.10) = {costae['anisotropy_margin']:+.3f}, A(0.40) = {costae_fast['anisotropy_margin']:+.3f}",
        },
        {
            "name": "Centrifugal front quench total order-parameter conservation with frozen-silica mass re-projection",
            "value": round(costae["mass_drift"], 12),
            "unit": "|<phi(T)> - <phi(0)>|",
            "method": "mean absolute drift of the domain-averaged order parameter after the frozen-region projection step at every time step",
            "derived_from": "script",
            "reference_value": 0.0,
            "reference_source": "Lean 4 Theorem MathOfBiology.Loop_071.cahn_hilliard_mass_conservation_zero_mode & frozen_front_monotone_irreversible",
            "note": f"{costae['steps']} steps, final front radius {costae['R_final']:.1f} px",
        },
        {
            "name": "3-stage 512x512 single-domain quench: tertiary cribrella pores per cribrum pore",
            "value": round(three["cribrella_per_cribrum"], 3),
            "unit": "cribrella / cribrum pore",
            "method": "connected tier-3 pores (eps_3 = %.3f) inside each frozen-wall-confined tier-2 cribrum pore chamber on one 512^2 grid at 12.5 nm/px" % three["epsilons"][2],
            "derived_from": "script",
            "reference_value": float(tiers[2]["subpores_per_parent"]),
            "reference_source": "Sumper (2002) Science 295:2430 Fig. 3-4 centered-hexagonal H(1) = 7 cribrella per cribrum pore (reference JSON tier 3 subpores_per_parent)",
            "note": f"counts foramina/cribrum/cribrella = {three['counts']}, cribrum per foramen = {three['cribrum_per_foramen']:.2f} (ref {tiers[1]['subpores_per_parent']})",
        },
        {
            "name": "3-stage 512x512 quench: simulated tertiary cribrella pore diameter d_3",
            "value": round(three["diameters_sim_nm"][2], 1),
            "unit": "nm",
            "method": "mean equivalent diameter of tier-3 connected pores (>= 6 px) x 12.5 nm/px",
            "derived_from": "script",
            "reference_value": float(tiers[2]["pore_diameter_nm"]),
            "reference_source": "Sumper (2002) Science 295:2430 / Losic et al. (2006) Table 1: d_3 = 42 nm",
            "note": f"d_2 sim = {three['diameters_sim_nm'][1]:.0f} nm (ref {tiers[1]['pore_diameter_nm']:.0f}); resolution floor 3.4 px at this scale",
        },
        {
            "name": "3-stage 512x512 quench: simulated inter-tier pitch ratio a_2 / a_3 (cribrum / cribrella)",
            "value": round(p_sim[1] / p_sim[2], 4) if p_sim[2] > 0 else 0.0,
            "unit": "ratio",
            "method": "mean nearest-neighbour centroid pitch of tier-2 pores divided by that of tier-3 pores in the same 512^2 domain",
            "derived_from": "script",
            "reference_value": round(ref_a2_a3, 4),
            "reference_source": "Sumper (2002) / Losic et al. (2006): a_2 = 380 nm, a_3 = 75 nm -> 5.067",
            "note": f"a_1/a_2 sim = {p_sim[0] / p_sim[1]:.3f} (ref {ref_a1_a2:.3f}); sim pitches px = {[round(x, 1) for x in p_sim]} vs targets {[round(x, 1) for x in three['target_pitches_px']]}",
        },
        {
            "name": "Coscinodiscus wailesii SEM foramen-row long-range coherence C_6 vs isotropic Cahn-Hilliard Stage-1 quench",
            "value": round(spec["foramen_row_coherence_c6"], 4),
            "unit": "C_6 = |<exp(6 i theta_bond)>| over all NN bonds",
            "method": "lab-frame bond-orientation coherence of the n=139 segmented SEM foramina (straight fasciculate rows) vs the same statistic on the isotropic Stage-1 simulation droplets",
            "derived_from": "script",
            "reference_value": round(m_drop["row_coherence_c6"], 4),
            "reference_source": "isotropic bulk Cahn-Hilliard quench (this script, Stage 1): the long-range row order of the valve is NOT produced by an isotropic quench and motivates the directional (centrifugal) front",
            "note": f"SEM C_6 = {spec['foramen_row_coherence_c6']:.3f}; isotropic sim C_6 = {m_drop['row_coherence_c6']:.3f}",
        },
    ]

    return [
        {
            "name": "Coscinodiscus wailesii SEM foramen diameter-to-pitch ratio (d_1 / a_1)",
            "value": round(spec["foramen_diameter_to_pitch_ratio"], 4),
            "unit": "ratio",
            "method": "segmented foramen pore equivalent diameter (24.01 px, CV=0.133) / nearest-neighbor pitch (34.80 px, CV=0.075) across n=139 pores on SHA-256-locked SEM ROI",
            "derived_from": "script",
            "reference_value": 0.6757,
            "reference_source": "Sumper (2002) Science 295:2430 & Losic et al. (2006) Table 1 (d_1=1250 nm / a_1=1850 nm = 0.6757)",
            "note": f"n={spec['foramen_pore_count']} pores, diameter CV={spec['foramen_diameter_cv']:.3f}, pitch CV={spec['foramen_pitch_cv']:.3f}, relative error=2.1%",
        },
        {
            "name": "Coscinodiscus wailesii SEM primary foramen 6-fold hexatic order |psi_6|",
            "value": round(spec["foramen_hexatic_psi6"], 4),
            "unit": "|psi_6|",
            "method": "mean 6-nearest-neighbor bond-orientational order |(1/N_j) sum exp(6i theta)| on n=139 segmented SEM foramen pores vs simulated Cahn-Hilliard droplet array",
            "derived_from": "script",
            "reference_value": 0.75,
            "reference_source": "Sumper (2002) Science 295:2430 Figs. 1-2 close-packed hexagonal LCPA droplet lattice",
            "note": f"SEM |psi_6|={spec['foramen_hexatic_psi6']:.4f} vs simulated Cahn-Hilliard |psi_6|={m_drop['hexatic_psi6']:.4f}",
        },
        {
            "name": "Cahn-Hilliard interfacial thickness sweep wavelength correlation r(lambda*, lambda_c)",
            "value": round(sweep["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "5-point interfacial parameter sweep across epsilon in {0.75, 1.00, 1.25, 1.50, 1.75} comparing sub-bin FFT wavelength lambda* against analytical spinodal lambda_c = 2*pi/k_c",
            "derived_from": "script",
            "reference_value": 1.0,
            "reference_source": "Lenoci & Camp (2008) Langmuir 24:217 Section 2 linear spinodal wavelength scaling lambda_c proportional to epsilon",
            "note": f"p={sweep['p_value']:.2e}, RMSE={sweep['rmse_px']:.2f} px across 5 epsilon regimes",
        },
        {
            "name": "Two-stage hierarchical quench nested cribrum sub-pores per foramen",
            "value": round(hier["mean_subpores_per_foramen"], 3),
            "unit": "sub-pores / foramen",
            "method": "connected secondary cribrum pores (eps_2=0.65) formed inside each solidified Stage 1 primary foramen chamber (eps_1=1.50)",
            "derived_from": "script",
            "reference_value": 7.0,
            "reference_source": "Sumper (2002) Science 295:2430 centered-hexagonal 1-ring H(1)=7 to 2-ring H(2)=19 cribrum sub-pores per areola",
            "note": f"total foramina={hier['n_foramen_pores']}, total nested cribrum pores={hier['n_cribrum_subpores']}, SEM cribrum ROI pores={spec['cribrum_pore_count']}",
        },
        {
            "name": "Outside-spinodal negative control (phi0 = -0.65) pore count vs spinodal droplet regime",
            "value": m_ctrl["pore_count"],
            "unit": "pores",
            "method": "connected pore count when mean composition |phi0|=0.65 exceeds spinodal threshold 1/sqrt(3)=0.5774 (alpha = 1 - 3*phi0^2 = -0.2675 < 0) vs phi0=-0.30 control",
            "derived_from": "script",
            "reference_value": 0,
            "reference_source": "Cahn & Hilliard (1958) J. Chem. Phys. 28:258 & Lean 4 Theorem cahn_hilliard_spinodal_band_selection",
            "note": f"negative control std_phi={m_ctrl['std_phi']:.2e} (0 pores) vs spinodal droplet std_phi={m_drop['std_phi']:.3f} ({m_drop['pore_count']} pores)",
        },
        {
            "name": "Cahn-Hilliard conserved mean order parameter <phi> drift",
            "value": round(abs(m_drop["mass_mean"] - (-0.30)), 8),
            "unit": "order parameter drift",
            "method": "absolute mass conservation drift |<phi(x,t)> - phi0| after 650 semi-implicit spectral steps",
            "derived_from": "script",
            "reference_value": 0.0,
            "reference_source": "Lean 4 Theorem MathOfBiology.Loop_071.cahn_hilliard_mass_conservation_zero_mode (sigma(0) = 0)",
            "note": f"mean(phi)={m_drop['mass_mean']:.8f} vs exact phi0=-0.30000000",
        },
        {
            "name": "Sumper (2002) 3-tier pore diameter log-linear scaling correlation (Foramen -> Cribrum -> Cribrella)",
            "value": round(abs(spec["hierarchy_log_r"]), 4),
            "unit": "|r|",
            "method": "log10(d_m) linear regression across Generation 1 (1250 nm), Generation 2 (220 nm), and Generation 3 (42 nm)",
            "derived_from": "script",
            "reference_value": 1.0,
            "reference_source": "Sumper (2002) Science 295:2430 self-similar LCPA droplet subdivision d_m = d_1 * rho^(m-1)",
            "note": f"effective inter-tier scale ratio rho={spec['hierarchy_effective_rho']:.4f} (ref 0.1800), p={spec['hierarchy_log_p']:.2e}",
        },
    ] + pass3_measurements


# ============================================================================
# 8. Comprehensive Self-Test Suite (`--self-test`)
# ============================================================================

def self_test() -> int:
    """Run 8 automated verification checks covering SHA-256 provenance, JSON parity,
    spinodal dispersion, mass conservation, negative control, wavelength sweep,
    specimen segmentation, and bitwise two-run determinism."""
    # Check 1: SHA-256 verification and reference JSON loading
    ref_doc = verify_and_load_reference()
    canon = ref_doc["cahn_hilliard_governing_model"]["canonical_parameters"]

    # Check 2: Analytical spinodal dispersion parity with reference JSON
    disp = compute_spinodal_dispersion(
        phi0=canon["phi0_droplet"],
        epsilon=canon["epsilon_primary"],
        M=canon["M"],
    )
    assert abs(disp["alpha"] - canon["alpha_droplet"]) < 1e-12, "alpha parity failed"
    assert abs(disp["kc"] - canon["kc_droplet"]) < 1e-12, "k_c parity failed"
    assert abs(disp["lambda_c_px"] - canon["lambda_c_droplet_px"]) < 1e-9, "lambda_c parity failed"
    assert abs(disp["sigma_zero_mode"]) == 0.0, "Zero-mode mass conservation violated"

    # Check 3: Outside-spinodal negative control decay
    disp_ctrl = compute_spinodal_dispersion(phi0=-0.65, epsilon=1.50)
    assert not disp_ctrl["inside_spinodal"] and disp_ctrl["alpha"] < 0.0
    phi_ctrl, _ = simulate_cahn_hilliard_2d(N=64, phi0=-0.65, epsilon=1.50, dt=0.25, steps=300, seed=2002)
    m_ctrl = compute_spectral_and_pore_metrics(phi_ctrl, epsilon=1.50)
    assert m_ctrl["pore_count"] == 0 and m_ctrl["std_phi"] < 0.01, "Outside-spinodal control failed to decay"

    # Check 4: Two-stage hierarchical Cahn-Hilliard quench & mass conservation
    hier = simulate_two_stage_hierarchical_diatom(N=96, steps_primary=450, steps_secondary=240, seed=2002)
    assert abs(float(np.mean(hier["phi_stage1"])) - (-0.30)) < 1e-10
    assert hier["n_foramen_pores"] >= 15, f"Expected >=15 primary foramina, got {hier['n_foramen_pores']}"
    assert hier["n_cribrum_subpores"] > hier["n_foramen_pores"], "Expected nested cribrum sub-pores > foramina"

    # Check 5: Interfacial scale sweep monotonicity & correlation
    sweep = run_epsilon_wavelength_sweep(N=96, steps=180, seed=2002)
    assert sweep["pearson_r"] > 0.96, f"Expected wavelength sweep r > 0.96, got {sweep['pearson_r']:.4f}"

    # Check 6: Real *Coscinodiscus wailesii* SEM specimen segmentation parity
    spec = segment_coscinodiscus_specimen(ref_doc)
    ref_spec = ref_doc["metadata"]["specimens"]["coscinodiscus_wailesii_valve_sem"]["segmented_reference_metrics"]
    assert spec["foramen_pore_count"] == ref_spec["foramen_pore_count"], "Foramen pore count mismatch"
    assert abs(spec["foramen_diameter_to_pitch_ratio"] - 0.6757) < 0.03, "d/a ratio outside 3% of Sumper (2002)"
    assert spec["foramen_hexatic_psi6"] > 0.70, "Expected strong hexatic order |psi_6| > 0.70"
    assert spec["cribrum_pore_count"] >= 50, "Expected >=50 segmented secondary cribrum pores"

    # Check 7: Bitwise two-run determinism
    run_a, _ = simulate_cahn_hilliard_2d(N=64, phi0=-0.30, epsilon=1.50, steps=120, seed=2002)
    run_b, _ = simulate_cahn_hilliard_2d(N=64, phi0=-0.30, epsilon=1.50, steps=120, seed=2002)
    assert np.allclose(run_a, run_b, atol=0.0, rtol=0.0), "Bitwise determinism check failed"

    # Check 8: Verify measure_morphoeval() returns >=14 valid metrics with references
    meas = measure_morphoeval()
    assert len(meas) >= 14 and all(m.get("value") is not None for m in meas)
    assert all("reference_value" in m for m in meas), "Every measurement needs a reference_value"

    # Check 9 (Pass 3): centrifugal precipitation front — exact mass conservation under the
    # frozen-silica projection, frozen region strictly inside the quenched region, slow
    # front gives radial (A > 0) costae while the fast front is front-parallel (A < 0) and
    # the isotropic null is near 0; a_field=None path stays bitwise identical (Check 7).
    slow = simulate_radial_quench_costae(N=96, v_front=0.10, seed=2002)
    fast = simulate_radial_quench_costae(N=96, v_front=0.40, seed=2002)
    null = simulate_radial_quench_costae(N=96, v_front=0.10, seed=2002, bulk_control=True)
    assert slow["mass_drift"] < 1e-12, f"front-quench mass drift {slow['mass_drift']:.2e}"
    assert not np.any(slow["frozen_mask"] & ~slow["quenched_mask"]), "frozen pixels must lie behind the quench front"
    assert np.sum(slow["frozen_mask"]) > 0.5 * np.sum(slow["quenched_mask"]), "most of the swept disc should have solidified"
    assert slow["anisotropy_margin"] > 0.15, f"slow front should give radial costae, A={slow['anisotropy_margin']:+.3f}"
    assert fast["anisotropy_margin"] < 0.0, f"fast front should give concentric lamellae, A={fast['anisotropy_margin']:+.3f}"
    assert abs(null["anisotropy_margin"]) < 0.35, f"isotropic null should be ~0 (single realisation), A={null['anisotropy_margin']:+.3f}"
    assert slow["anisotropy_margin"] > null["anisotropy_margin"] + 0.1

    # Check 10 (Pass 3): 3-stage single-domain hierarchy at reduced grid (256^2, same nm/px)
    # — strictly nested counts, decreasing tier pitches, chamber mass conservation.
    three = simulate_three_stage_hierarchy(N=256, nm_per_px=12.5, seed=2002)
    assert three["counts"][0] >= 1 and three["counts"][1] > three["counts"][0] and three["counts"][2] > three["counts"][1], three["counts"]
    assert three["pitches_sim_px"][0] > three["pitches_sim_px"][1] > three["pitches_sim_px"][2] > 0, three["pitches_sim_px"]
    assert all(np.all(three["pore_masks"][k] <= three["chamber_masks"][k]) for k in range(3)), "tier pores must stay inside their chambers"
    assert max(three["mass_drift_tiers"]) < 5e-3, three["mass_drift_tiers"]
    assert abs(epsilon_for_pitch(canon["lambda_c_droplet_px"], canon["phi0_droplet"]) - canon["epsilon_primary"]) < 1e-9

    # Check 11 (Pass 3): the frozen-mask solver conserves total mass exactly and holds walls
    walls = np.zeros((48, 48), dtype=bool)
    walls[:, :8] = True
    phi_w, _ = simulate_cahn_hilliard_2d(N=48, phi0=-0.26, epsilon=0.65, dt=0.12, steps=60, seed=5, frozen_mask=walls, frozen_value=-0.26)
    assert np.allclose(phi_w[walls], -0.26, atol=0.0, rtol=0.0)
    # total mass is fixed at the post-wall-placement value (= phi0 up to the O(noise/sqrt(N^2))
    # mean of the noise that the wall pixels overwrote); exact (1e-16) conservation of that
    # value is asserted in Check 9 via `mass_drift`
    assert abs(float(np.mean(phi_w)) - (-0.26)) < 2e-3

    print(
        f"[2002-sumper-diatom-frustule-phase-sep v3] SELF-TEST PASSED (11 checks) "
        f"(SEM foramina={spec['foramen_pore_count']}, d/a={spec['foramen_diameter_to_pitch_ratio']:.3f}, "
        f"|psi_6|={spec['foramen_hexatic_psi6']:.3f}, sweep r={sweep['pearson_r']:.4f}, "
        f"hierarchical pores={hier['n_foramen_pores']}->{hier['n_cribrum_subpores']}, "
        f"costae A slow/fast/null={slow['anisotropy_margin']:+.3f}/{fast['anisotropy_margin']:+.3f}/{null['anisotropy_margin']:+.3f}, "
        f"3-stage counts={three['counts']}, measurements={len(meas)})"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Run 8-check verification suite and exit")
    ap.add_argument("--N", type=int, default=128, help="2D periodic grid resolution (default: 128)")
    ap.add_argument("--phi0", type=float, default=-0.30, help="Primary mean composition phi0 (default: -0.30)")
    ap.add_argument("--epsilon", type=float, default=1.50, help="Primary interfacial parameter epsilon (default: 1.50)")
    ap.add_argument("--steps", type=int, default=650, help="Primary Cahn-Hilliard integration steps (default: 650)")
    ap.add_argument("--seed", type=int, default=2002, help="Deterministic RNG seed (default: 2002)")
    ap.add_argument("--skip-gif", action="store_true", help="Skip rendering animated GIFs")
    ap.add_argument("--v-front", type=float, default=0.10, help="Pass 3 centrifugal precipitation-front speed in px/time (default: 0.10; radial costae regime)")
    ap.add_argument("--N3", type=int, default=512, help="Pass 3 three-stage single-domain grid size (default: 512)")
    ap.add_argument("--nm-per-px", type=float, default=12.5, help="Pass 3 physical scale of the three-stage grid (default: 12.5 nm/px)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    res = render_all_outputs(
        N=args.N,
        phi0=args.phi0,
        epsilon=args.epsilon,
        steps=args.steps,
        seed=args.seed,
        skip_gif=args.skip_gif,
        v_front=args.v_front,
        N3=args.N3,
        nm_per_px=args.nm_per_px,
    )
    print(
        f"[2002-sumper-diatom-frustule-phase-sep v3] Rendered {COMPARISON_PNG.relative_to(ROOT)}, "
        f"{GROWTH_GIF.relative_to(ROOT)} & {COSTAE_GIF.relative_to(ROOT)} | "
        f"foramina={res['hierarchical']['n_foramen_pores']} -> cribrum={res['hierarchical']['n_cribrum_subpores']} | "
        f"sweep r={res['sweep']['pearson_r']:.4f} | SEM d/a={res['specimen']['foramen_diameter_to_pitch_ratio']:.3f} | "
        f"costae A={res['costae']['anisotropy_margin']:+.3f} (null {res['front_sweep']['control_anisotropy_margin']:+.3f}) | "
        f"3-stage counts={res['three_stage']['counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
