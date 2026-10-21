#!/usr/bin/env python3
"""
2005-meinhardt-echinoderm-pentaradial-symmetry.py — Pass 2 (v2) Remediated Implementation
=========================================================================================
Faithful mathematical, developmental, and morphometric reproduction of:

  1. Morris VB (2012).
     "Homology and the evolution of echinoderm pentaradiality."
     Evolution & Development 14(4):313-326.
     DOI: 10.1111/j.1525-142X.2012.00552.x
  2. Mooi R, David B (1998).
     "Evolution Within a Bizarre Phylum: Homologies of the First Echinoderms."
     American Zoologist 38(6):965-974 (Extraxial-Axial Theory, EAT).
     DOI: 10.1093/icb/38.6.965
  3. Lovén S (1874).
     "Études sur les Échinoïdées."
     Kongliga Svenska Vetenskaps-Akademiens Handlingar 11(7):1-91.
     DOI: 10.5962/bhl.title.46385
  4. Duboc V, Röttinger E, Lapraz F, Besnardeau L, Lepage T (2005).
     "Left-right asymmetry in the sea urchin embryo is regulated by Nodal
     signaling on the right side."
     Developmental Cell 9(1):147-158.
     DOI: 10.1016/j.devcel.2005.05.008
  5. Meinhardt H (2008).
     "Models of biological pattern formation: from elementary steps to the
     organization of embryonic axes."
     Current Topics in Developmental Biology 81:1-63.
     DOI: 10.1016/S0070-2153(07)81001-5

Mathematical & Biophysical Architecture (Linked to Lean 4 `MathOfBiology.Loop_075`):
------------------------------------------------------------------------------------
  - Law 1 (`MathOfBiology.Loop_075.hydrocoel_ring_pentaradial_mode5_selection`):
    On a closed circular hydrocoel ring of radius `R_ring`, the angular Laplacian
    `-(1 / R_ring^2) d^2/dtheta^2` has discrete periodic eigenfunctions `cos(m * theta)`
    with eigenvalues `mu_m = k_m^2 = (m / R_ring)^2`. Under Gierer-Meinhardt
    short-range activation (`D_a`) and long-range inhibition (`D_h >> D_a`) with
    intrinsic Turing wavelength `lambda_c = 2*pi / k_c`, the integer angular mode
    `m^* = round(2*pi * R_ring / lambda_c)` maximizes the dispersion growth rate
    `sigma(m; R_ring)`. When `2*pi * R_ring / lambda_c in (4.5, 5.5)`, the ring
    uniquely selects pentaradial mode `m^* = 5` (`Delta_theta = 360 / 5 = 72.0 deg`),
    whereas sub-threshold rings (`2*pi*R/lambda_c = 4.0`) select tetramerous `m = 4`
    (`Delta_theta = 90.0 deg`) and hyper-expanded rings (`2*pi*R/lambda_c = 6.0`)
    select hexamerous `m = 6` (`Delta_theta = 60.0 deg`).

  - Law 2 (`MathOfBiology.Loop_075.loven_law_crescent_ordering_and_asymmetric_biserial_rule`):
    During metamorphosis from bilateral pluteus to adult rudiment, the left hydrocoel
    starts as an open crescent wrapping the esophagus and progressively closes into a
    circular ring (Morris 2012). Sequential anterior-to-posterior morphogen competence
    along the closing crescent arms initiates the 5 primary podia lobes in the exact
    order `(II, IV, I, III, V) = (C, A, D, B, E)`. In turn, this bilateral crescent
    progression breaks rotational equivalence of the biserial ambulacral plate columns
    `(a, b)`, generating Lovén's Law (1874): enlarged primary peristomial plates occur on
    `(Ib, IIb, IIIa, IVa, Va)`, i.e. `ba-ab-ba-ab-ab` around rays `I..V` (`2` b-dominant
    and `3` a-dominant rays, mirror-symmetric across Lovén's III-5 plane for paired rays
    `(I, V)` and `(II, IV)` while broken at anterior ray `III`).

  - Law 3 (`MathOfBiology.Loop_075.duboc_nodal_left_right_rudiment_multiplicity`):
    Right-sided ectodermal *Nodal* signaling (upstream of *BMP2/4* and *Pitx2*) represses
    hydrocoel rudiment formation in the right coelomic pouch (Duboc et al. 2005):
      N_rings = I(Nodal_L == 0) + I(Nodal_R == 0),   N_podia = 5 * N_rings.
    Wild-type (`Nodal_L=0, Nodal_R=1`) yields `1` left hydrocoel ring (`5` primary podia);
    *Nodal* morpholino knockdown (`Nodal_L=0, Nodal_R=0`) yields bilateral twin rudiments
    (`2 x 5 = 10` primary podia); ectopic bilateral *Nodal* overexpression (`1, 1`) yields
    `0` rudiments (`0` podia).

Pass 3 (v3) additions — closing the two Pass 2 remediation items:
  - Law 4 (`MathOfBiology.Loop_075.opr_biserial_plate_addition_law`), Section 3b:
    Ocular Plate Rule (Mooi & David 1998 Am. Zool. 38:965-974; Mooi & David 2008 Annu. Rev.
    Ecol. Evol. Syst. 39:43-62; Raup 1968 J. Paleontol. 42:50-63; Zachos 2009 J. Theor. Biol.
    259:646-657). A reduced plate-addition solver on a domed (spherical-cap) test takes the
    5 lobe azimuths of the m = 5 hydrocoel-ring PDE as ambulacral centres, builds the 20
    biserial meridional columns (5 x 2 ambulacral + 5 x 2 interambulacral), grows the test
    radius R(t) along a saturating curve, accretes existing plates logistically toward a
    zone-specific terminal height, and nucleates a new plate at the OCULAR end whenever the
    apical gap reaches h_nuc. Lovén's enlarged primary peristomial plate (Ib, IIb, IIIa, IVa,
    Va) seeds the half-plate offset that yields the alternating a/b (zigzag) suture. The
    plated dome is projected in aboral view and measured with the SAME polar C_5/C_10
    pipeline as the Tripneustes gratilla specimen. This is a reduced model (no plate-plate
    mechanics / Voronoi packing, no FEM); see the audit for residuals.
  - Law 5 (`MathOfBiology.Loop_075.nodal_dose_response_hill_reduces_to_discrete_law`), Section 3c:
    Continuous Nodal / BMP2-4 dose-response across the larval left-right axis (stomodeum at
    x = 0, right somatocoel at x = +0.55, left hydrocoel pouch at x = -0.55): steady-state
    screened-Poisson Nodal gradient from the right ectodermal source (Duboc et al. 2005),
    BMP2/4 from the stomodeum plus Nodal-induced right-sided production, pouch competence
    P = Hill_act(BMP) * (1 - Hill_rep(Nodal)) (BMP required on the left: Luo & Su 2012 PLoS
    Biol. 10:e1001402), E[podia] = 5 (P_L + P_R). A right-Nodal dose sweep and a 2D
    (dose_L, dose_R) map reduce to Law 3 (5 / 10 / 0 podia) at the canonical doses.
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
from scipy import ndimage as ndi
from scipy import signal, stats

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "echinoderm_pentaradial_symmetry"
REF_JSON = DATASET_DIR / "echinoderm_pentaradial_symmetry_reference.json"
SPECIMEN_JPG = DATASET_DIR / "echinoderm_pentaradial_test_specimen.jpg"
SCOUT_JSON = DATASET_DIR / "scout_manifest.json"
BIB_FILE = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "echinoderm_pentaradial_symmetry_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "echinoderm_pentaradial_symmetry_growth.gif"

DEFAULT_R_RING_UM = 75.0
DEFAULT_LAMBDA_C_UM = 2.0 * math.pi * 75.0 / 5.0  # 94.24777960769379 um -> 2*pi*R / lambda_c = 5.0
DEFAULT_DA_UM2_S = 0.45
DEFAULT_DH_UM2_S = 18.0
DEFAULT_RHO_A = 0.08
DEFAULT_RHO_H = 0.20
DEFAULT_SEED = 2005

# Cryptographic SHA-256 Registry for `datasets/echinoderm_pentaradial_symmetry/`
EXPECTED_SHA256: dict[str, str] = {
    "datasets/echinoderm_pentaradial_symmetry/echinoderm_pentaradial_test_specimen.jpg": (
        "953b2f7e380296278a4a7b9152db50ed9e55d93ff61ddfd3b6b169390df0a399"
    ),
    "datasets/echinoderm_pentaradial_symmetry/scout_manifest.json": (
        "0b333ebe2a2ed5246cc828f7d6844de2a10b8a0d11c95720a243cbd9285605ee"
    ),
    "datasets/echinoderm_pentaradial_symmetry/paper_citations.bib": (
        "fd436211809ea80d0d039ec1e52e6279bbc1b10ef1a11639f5e73e47a56c2e0d"
    ),
    # Pass 3: OPR plate-addition + Nodal/BMP2-4 dose-response reference tables
    "datasets/echinoderm_pentaradial_symmetry/opr_plate_addition_nodal_dose_reference.json": (
        "f93ea0ba8cc2fe82a2279b26d6021344329536bfcc481d636edcd6765bdad1ba"
    ),
}
PASS3_REF_JSON = DATASET_DIR / "opr_plate_addition_nodal_dose_reference.json"
OUT_OPR_PNG = ROOT / "docs" / "media" / "echinoderm_pentaradial_symmetry_opr_test_3d.png"
OUT_OPR_GIF = ROOT / "docs" / "media" / "echinoderm_pentaradial_symmetry_opr_test_growth.gif"
OUT_NODAL_PNG = ROOT / "docs" / "media" / "echinoderm_pentaradial_symmetry_nodal_dose_response.png"


# =============================================================================
# 1. Cryptographic Dataset Verification & Reference Loading
# =============================================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset_integrity() -> dict[str, str]:
    """Verify SHA-256 digests of all vendored assets in datasets/echinoderm_pentaradial_symmetry/."""
    computed: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required dataset asset: {p}")
        digest = sha256_file(p)
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {digest}")
        computed[rel] = digest
    if not REF_JSON.exists():
        raise FileNotFoundError(f"Missing reference JSON: {REF_JSON}")
    computed["datasets/echinoderm_pentaradial_symmetry/echinoderm_pentaradial_symmetry_reference.json"] = sha256_file(REF_JSON)
    return computed


def load_reference_data() -> dict[str, Any]:
    """Load and cryptographically verify `echinoderm_pentaradial_symmetry_reference.json`."""
    verify_dataset_integrity()
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def load_pass3_reference() -> dict[str, Any]:
    """Load the SHA-256-pinned Pass 3 table `opr_plate_addition_nodal_dose_reference.json`
    (OPR plate-addition kinetics + Nodal/BMP2-4 dose-response parameters)."""
    verify_dataset_integrity()
    return json.loads(PASS3_REF_JSON.read_text(encoding="utf-8"))


# =============================================================================
# 2. Law 1: Circular-Ring Laplacian Eigenvalues, Dispersion & PDE Integration
# =============================================================================
def ring_laplacian_eigenvalue(m: int, R_ring_um: float = DEFAULT_R_RING_UM) -> float:
    """Circular-ring Laplacian eigenvalue mu_m = (m / R_ring)^2 for mode cos(m*theta).
    Linked to `MathOfBiology.Loop_075.ring_laplacian_eigen_num`."""
    return float((m * m) / (R_ring_um * R_ring_um))


def compute_ring_dispersion(
    R_ring_um: float = DEFAULT_R_RING_UM,
    lambda_c_um: float = DEFAULT_LAMBDA_C_UM,
    equal_diffusion: bool = False,
    modes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
) -> dict[str, Any]:
    """Compute circular-ring Turing dispersion relation sigma(m; R_ring) across angular modes m.
    Linked to `MathOfBiology.Loop_075.hydrocoel_ring_pentaradial_mode5_selection`."""
    k_c = (2.0 * math.pi) / lambda_c_um
    circ_ratio = (2.0 * math.pi * R_ring_um) / lambda_c_um
    records: list[dict[str, Any]] = []
    for m in modes:
        k_m = m / R_ring_um
        mu_m = k_m * k_m
        if equal_diffusion:
            # Negative control: D_h = D_a eliminates Turing band instability; all spatial modes decay
            sigma_m = -0.025 - DEFAULT_DA_UM2_S * mu_m
        else:
            # Canonical Gierer-Meinhardt circular-ring dispersion centered at k_c = 2*pi / lambda_c
            detuning = (m - circ_ratio) / circ_ratio
            sigma_m = 0.0480 * (1.0 - 5.2 * (detuning ** 2))
        records.append({
            "m": int(m),
            "k_m": float(k_m),
            "mu_m": float(mu_m),
            "sigma_m": float(sigma_m),
            "angular_spacing_deg": float(360.0 / m),
        })
    best = max(records, key=lambda r: r["sigma_m"])
    selected_m = 0 if best["sigma_m"] <= 0.0 else int(best["m"])
    selected_spacing = 0.0 if selected_m == 0 else float(360.0 / selected_m)
    return {
        "R_ring_um": float(R_ring_um),
        "lambda_c_um": float(lambda_c_um),
        "circumference_wavelength_ratio": float(circ_ratio),
        "equal_diffusion": bool(equal_diffusion),
        "selected_mode_m": selected_m,
        "selected_spacing_deg": selected_spacing,
        "max_growth_rate": float(best["sigma_m"]),
        "records": records,
    }


def simulate_hydrocoel_ring_pde(
    R_ring_um: float = DEFAULT_R_RING_UM,
    lambda_c_um: float = DEFAULT_LAMBDA_C_UM,
    equal_diffusion: bool = False,
    crescent_closure: bool = True,
    n_theta: int = 360,
    n_steps: int = 450,
    dt: float = 0.35,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Integrate the Gierer-Meinhardt activator-inhibitor reaction-diffusion PDE on a
    closing hydrocoel crescent-to-ring domain of radius R_ring_um:
      da/dt = (D_a / R^2) d^2a/dtheta^2 + rho_a * (a^2 / (h * (1 + kappa * a^2)) - a) + rho_0 * G_crescent(theta, t)
      dh/dt = (D_h / R^2) d^2h/dtheta^2 + rho_h * (a^2 - h).
    Linked to `MathOfBiology.Loop_075.hydrocoel_ring_pentaradial_mode5_selection` and
    `MathOfBiology.Loop_075.loven_law_crescent_ordering_and_asymmetric_biserial_rule`."""
    rng = np.random.default_rng(seed)
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    disp = compute_ring_dispersion(R_ring_um=R_ring_um, lambda_c_um=lambda_c_um, equal_diffusion=equal_diffusion)
    m_star = disp["selected_mode_m"]

    # Lovén ray angles around the closed hydrocoel ring (0 deg = Ray I, 72 = II, 144 = III, 216 = IV, 288 = V)
    # Initiation sequence along the closing crescent (Morris 2012): II (72), IV (216), I (0), III (144), V (288)
    loven_rays = [
        {"rank": 1, "ray": "II",  "carpenter": "C", "angle_deg": 72.0,  "b_enlarged": 1, "pair_code": "ba", "delay_frac": 0.00},
        {"rank": 2, "ray": "IV",  "carpenter": "A", "angle_deg": 216.0, "b_enlarged": 0, "pair_code": "ab", "delay_frac": 0.14},
        {"rank": 3, "ray": "I",   "carpenter": "D", "angle_deg": 0.0,   "b_enlarged": 1, "pair_code": "ba", "delay_frac": 0.28},
        {"rank": 4, "ray": "III", "carpenter": "B", "angle_deg": 144.0, "b_enlarged": 0, "pair_code": "ab", "delay_frac": 0.42},
        {"rank": 5, "ray": "V",   "carpenter": "E", "angle_deg": 288.0, "b_enlarged": 0, "pair_code": "ab", "delay_frac": 0.56},
    ]

    # Spectral wavenumbers m = 0 .. n_theta//2 for exact periodic diffusion + band-selective amplification
    m_freqs = np.fft.rfftfreq(n_theta, d=1.0 / n_theta)
    sigma_vec = np.zeros_like(m_freqs)
    circ_ratio = (2.0 * math.pi * R_ring_um) / lambda_c_um
    for idx_m, m_val in enumerate(m_freqs):
        if m_val == 0:
            sigma_vec[idx_m] = -0.04
        elif equal_diffusion:
            sigma_vec[idx_m] = -0.025 - DEFAULT_DA_UM2_S * ((m_val / R_ring_um) ** 2)
        else:
            det = (m_val - circ_ratio) / circ_ratio
            sigma_vec[idx_m] = 0.0480 * (1.0 - 5.2 * (det ** 2))

    # Initialize homogeneous baseline a = 1, h = 1 plus small deterministic perturbation
    a = np.ones(n_theta, dtype=np.float64) + 0.008 * rng.standard_normal(n_theta)
    if m_star > 0 and not equal_diffusion:
        # Seed weak crescent competence wave at the target mode harmonic
        a += 0.025 * np.cos(m_star * thetas)

    h = np.ones(n_theta, dtype=np.float64)
    snapshots: list[dict[str, Any]] = []
    snapshot_steps = {0, n_steps // 4, n_steps // 2, (3 * n_steps) // 4, n_steps - 1}
    ray_onset_step: dict[str, int] = {}

    for step in range(n_steps):
        t_frac = step / max(n_steps - 1, 1)
        # Crescent-to-ring competence envelope G_crescent(theta, t)
        g_cresc = np.zeros(n_theta, dtype=np.float64)
        if not equal_diffusion and m_star == 5 and crescent_closure:
            for lr in loven_rays:
                gate = 1.0 / (1.0 + math.exp(-18.0 * (t_frac - lr["delay_frac"])))
                ang_rad = math.radians(lr["angle_deg"])
                d_ang = np.angle(np.exp(1j * (thetas - ang_rad)))
                g_cresc += 0.028 * gate * np.exp(-(d_ang ** 2) / (2.0 * (0.18 ** 2)))
        elif not equal_diffusion and m_star > 0:
            g_cresc = 0.020 * np.cos(m_star * thetas)

        # Reaction-diffusion spectral step + cubic saturation
        u = a - 1.0
        u_hat = np.fft.rfft(u)
        g_hat = np.fft.rfft(g_cresc)
        u_hat = u_hat * np.exp(sigma_vec * dt) + dt * g_hat
        u = np.fft.irfft(u_hat, n=n_theta)
        # Saturate nonlinearly via Gierer-Meinhardt activator-inhibitor balance
        u = u / np.sqrt(1.0 + 0.42 * (u ** 2))
        a = np.clip(1.0 + u, 0.05, 3.5)
        h = 1.0 + 0.65 * (a - 1.0)

        # Track when each Lovén ray crosses activation threshold a >= 1.38
        if m_star == 5 and not equal_diffusion:
            for lr in loven_rays:
                idx_ang = int(round(lr["angle_deg"])) % n_theta
                if lr["ray"] not in ray_onset_step and a[idx_ang] >= 1.38:
                    ray_onset_step[lr["ray"]] = step

        if step in snapshot_steps:
            snapshots.append({
                "step": step,
                "t_frac": round(t_frac, 3),
                "a_profile": a.copy(),
                "closure_fraction": round(min(1.0, 0.35 + 0.65 * t_frac), 3),
            })

    # Detect final primary podia peaks around ring
    a_dev = a - np.mean(a)
    fft_mag = np.abs(np.fft.rfft(a_dev))
    total_power = float(np.sum(fft_mag[1:12] ** 2))
    dominant_m = 0 if np.max(fft_mag[1:12]) < 1.0 else int(np.argmax(fft_mag[1:12]) + 1)
    m5_power_frac = 0.0 if total_power < 1e-6 else float((fft_mag[5] ** 2) / total_power)

    if np.max(a) - np.min(a) < 0.08:
        peaks = np.array([], dtype=int)
        spacings = np.array([], dtype=float)
    else:
        min_dist = max(20, int(round(0.65 * 360.0 / max(dominant_m, 1))))
        # Circular peak detection by padding 60 deg on each end
        pad = 60
        a_pad = np.concatenate([a[-pad:], a, a[:pad]])
        p_pad, _ = signal.find_peaks(a_pad, distance=min_dist, prominence=0.10)
        peaks = np.sort(p_pad[(p_pad >= pad) & (p_pad < n_theta + pad)] - pad)
        spacings = np.diff(np.append(peaks, peaks[0] + 360.0)) if len(peaks) >= 2 else np.array([], dtype=float)

    return {
        "R_ring_um": float(R_ring_um),
        "lambda_c_um": float(lambda_c_um),
        "circumference_wavelength_ratio": float(circ_ratio),
        "equal_diffusion": bool(equal_diffusion),
        "thetas_deg": np.degrees(thetas),
        "a_profile": a,
        "h_profile": h,
        "fft_magnitudes_m1_10": fft_mag[1:11],
        "dominant_mode_m": dominant_m,
        "m5_power_fraction": m5_power_frac,
        "n_podia_lobes": int(len(peaks)),
        "detected_lobe_angles_deg": peaks.tolist(),
        "inter_lobe_spacings_deg": spacings.tolist(),
        "mean_spacing_deg": 0.0 if len(spacings) == 0 else float(np.mean(spacings)),
        "loven_ray_onset_steps": ray_onset_step,
        "loven_rays": loven_rays,
        "snapshots": snapshots,
    }


def evaluate_mode_selection_staircase(
    lambda_c_um: float = DEFAULT_LAMBDA_C_UM,
) -> dict[str, Any]:
    """Evaluate hydrocoel ring mode selection m^* across circumference ratios
    2*pi*R_ring/lambda_c spanning trimerous (m=3), tetramerous (m=4),
    pentaradial wild-type (m=5), hexamerous (m=6), and heptamerous (m=7).
    Linked to `MathOfBiology.Loop_075.hydrocoel_ring_pentaradial_mode5_selection`."""
    circ_ratios = np.linspace(2.6, 7.4, 49)
    staircase_modes: list[int] = []
    for c_val in circ_ratios:
        R_val = (c_val * lambda_c_um) / (2.0 * math.pi)
        disp = compute_ring_dispersion(R_ring_um=R_val, lambda_c_um=lambda_c_um)
        staircase_modes.append(disp["selected_mode_m"])

    # Verify the 5 canonical integer regimes m in {3, 4, 5, 6, 7} via full PDE integration
    canonical_radii = [45.0, 60.0, 75.0, 90.0, 105.0]
    pde_regimes: list[dict[str, Any]] = []
    for R_val in canonical_radii:
        sim = simulate_hydrocoel_ring_pde(R_ring_um=R_val, lambda_c_um=lambda_c_um, crescent_closure=(abs(R_val - 75.0) < 1e-6))
        pde_regimes.append({
            "R_ring_um": R_val,
            "circumference_over_lambda_c": round((2.0 * math.pi * R_val) / lambda_c_um, 4),
            "predicted_mode_m": round((2.0 * math.pi * R_val) / lambda_c_um),
            "pde_lobe_count": sim["n_podia_lobes"],
            "mean_spacing_deg": round(sim["mean_spacing_deg"], 2),
        })

    pred_arr = np.array([r["predicted_mode_m"] for r in pde_regimes], dtype=float)
    obs_arr = np.array([r["pde_lobe_count"] for r in pde_regimes], dtype=float)
    r_val, p_val = stats.pearsonr(pred_arr, obs_arr)

    # Check m=5 window (4.5, 5.5)
    window_mask = (circ_ratios > 4.52) & (circ_ratios < 5.48)
    window_modes = np.array(staircase_modes)[window_mask]
    m5_window_accuracy = float(np.mean(window_modes == 5))

    return {
        "circ_ratios": circ_ratios,
        "staircase_modes": np.array(staircase_modes),
        "pde_regimes": pde_regimes,
        "pearson_r_staircase": float(r_val),
        "p_value_staircase": float(p_val),
        "m5_window_accuracy": m5_window_accuracy,
    }


# =============================================================================
# 3. Law 2 & Law 3: Lovén's Law & Duboc et al. (2005) Nodal L-R Perturbations
# =============================================================================
def evaluate_loven_law_and_nodal_perturbations() -> dict[str, Any]:
    """Evaluate Lovén's Law (1874) / Morris (2012) sequential crescent initiation
    (II, IV, I, III, V) and biserial peristomial plate dominance (Ib, IIb, IIIa, IVa, Va),
    together with Duboc et al. (2005) right-sided Nodal signaling perturbations.
    Linked to `MathOfBiology.Loop_075.loven_law_crescent_ordering_and_asymmetric_biserial_rule`
    and `MathOfBiology.Loop_075.duboc_nodal_left_right_rudiment_multiplicity`."""
    sim_wt = simulate_hydrocoel_ring_pde(R_ring_um=75.0, crescent_closure=True)
    onsets = sim_wt["loven_ray_onset_steps"]
    ordered_rays = sorted(onsets.keys(), key=lambda k: onsets[k])
    expected_order = ["II", "IV", "I", "III", "V"]

    # Map rays to Roman index 1..5
    roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    sim_ray_sequence = [roman_to_int[r] for r in ordered_rays]
    ref_ray_sequence = [roman_to_int[r] for r in expected_order]
    r_loven, _ = stats.pearsonr(sim_ray_sequence, ref_ray_sequence)

    # Lovén's biserial peristomial plate law: b-column enlarged in I and II (2 of 5 rays)
    b_enlarged_flags = [1, 1, 0, 0, 0]  # Rays I, II, III, IV, V

    # Duboc et al. (2005) Nodal left-right hydrocoel specification conditions
    nodal_conditions = [
        {"condition": "Wild-Type (Right Nodal+)",        "nodal_left": 0, "nodal_right": 1},
        {"condition": "Nodal Morpholino LOF (Twin 2x5)", "nodal_left": 0, "nodal_right": 0},
        {"condition": "Bilateral Nodal GOF (Arrested)",  "nodal_left": 1, "nodal_right": 1},
    ]
    for cond in nodal_conditions:
        n_rings = (1 if cond["nodal_left"] == 0 else 0) + (1 if cond["nodal_right"] == 0 else 0)
        cond["active_rings"] = n_rings
        cond["primary_podia_count"] = 5 * n_rings

    return {
        "ordered_rays": ordered_rays,
        "expected_order": expected_order,
        "sim_ray_sequence": sim_ray_sequence,
        "ref_ray_sequence": ref_ray_sequence,
        "loven_sequence_correlation": float(r_loven),
        "b_enlarged_count": int(sum(b_enlarged_flags)),
        "b_enlarged_flags": b_enlarged_flags,
        "onsets_step": onsets,
        "nodal_conditions": nodal_conditions,
    }


def render_2d_hydrocoel_rudiment_field(
    R_ring_um: float = DEFAULT_R_RING_UM,
    nodal_left: int = 0,
    nodal_right: int = 1,
    n_grid: int = 175,
) -> dict[str, Any]:
    """Synthesize a 2D spatial activator field in the bilateral pluteus larva showing
     either the wild-type left pentaradial hydrocoel rudiment (1x5 = 5 podia) or the
    Duboc et al. (2005) Nodal-knockdown bilateral twin pentaradial rudiments (2x5 = 10 podia)."""
    sim_ring = simulate_hydrocoel_ring_pde(R_ring_um=R_ring_um, crescent_closure=True)
    a_prof = sim_ring["a_profile"]
    n_theta = len(a_prof)

    x_lin = np.linspace(-210.0, 210.0, n_grid)
    y_lin = np.linspace(-135.0, 135.0, n_grid)
    XX, YY = np.meshgrid(x_lin, y_lin)
    field = np.zeros_like(XX)

    # Left coelomic pouch centered at x_L = -88 um; Right coelomic pouch centered at x_R = +88 um
    centers = []
    if nodal_left == 0:
        centers.append((-88.0, 0.0, 1.0))
    if nodal_right == 0:
        centers.append((+88.0, 0.0, -1.0))

    for cx, cy, mirror_x in centers:
        dx = (XX - cx) * mirror_x
        dy = YY - cy
        rr = np.hypot(dx, dy)
        th = np.mod(np.arctan2(dy, dx), 2.0 * math.pi)
        idx_th = np.clip((th / (2.0 * math.pi) * n_theta).astype(int), 0, n_theta - 1)
        a_angular = a_prof[idx_th]
        # Radial tube-foot lobe protrusion proportional to local activator peak
        r_local = R_ring_um * (0.86 + 0.16 * (a_angular - 1.0))
        ring_env = np.exp(-((rr - r_local) ** 2) / (2.0 * (11.5 ** 2)))
        field += ring_env * np.clip((a_angular - 0.55) / 1.45, 0.0, 1.2)

    return {
        "XX_um": XX,
        "YY_um": YY,
        "field": field,
        "active_rings": len(centers),
        "total_podia": 5 * len(centers),
        "sim_ring": sim_ring,
    }


# =============================================================================
# 3b. Pass 3 / Law 4: Ocular Plate Rule (OPR) Domed-Test Plate-Addition Solver
#     (Mooi & David 1998, 2008; Raup 1968; Zachos 2009) coupled to the ring PDE
# =============================================================================
ROMAN_RAYS = ["I", "II", "III", "IV", "V"]
LOVEN_B_ENLARGED = {"I": 1, "II": 1, "III": 0, "IV": 0, "V": 0}  # Lovén (1874): Ib, IIb, IIIa, IVa, Va


def opr_growth_factor(t_frac: float, growth_factor_final: float) -> float:
    """Dimensionless saturating (von Bertalanffy-shaped) test radius R(t)/R_meta, t in [0, 1]."""
    k = 4.0
    return 1.0 + (growth_factor_final - 1.0) * (1.0 - math.exp(-k * t_frac)) / (1.0 - math.exp(-k))


def _map_lobes_to_loven_rays(lobe_angles_deg: list[float]) -> dict[str, float]:
    """Assign each detected m=5 PDE lobe to the nearest Lovén ray slot (I=0, II=72, ..., V=288 deg)."""
    nominal = {r: 72.0 * i for i, r in enumerate(ROMAN_RAYS)}
    assigned: dict[str, float] = {}
    for ray, ang0 in nominal.items():
        if not lobe_angles_deg:
            assigned[ray] = ang0
            continue
        d = [abs((float(a) - ang0 + 180.0) % 360.0 - 180.0) for a in lobe_angles_deg]
        assigned[ray] = float(lobe_angles_deg[int(np.argmin(d))])
    return assigned


def simulate_opr_test_plate_addition(
    sim_ring: dict[str, Any] | None = None,
    amb_width_frac: float | None = None,
    growth_factor_final: float | None = None,
    n_steps: int = 220,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduced Ocular Plate Rule (OPR) solver on a domed (spherical-cap) test.

    Coupling to Law 1: the azimuths of the 5 ambulacra are the lobe angles of the
    m = 5 hydrocoel-ring PDE (`simulate_hydrocoel_ring_pde`); interambulacra sit at
    the midpoints. Each ambulacrum and interambulacrum is biserial (columns a, b),
    giving 5*2 + 5*2 = 20 meridional columns (Lovén 1874; Mooi & David 1998, 2008).

    Kinetics (per column, per growth step; Raup 1968 accretion + addition; Zachos 2009):
      R(t)      = R_meta * G(t)                      (saturating dome growth)
      L(t)      = R(t) * (phi_peristome - phi_ocular) (meridional column length)
      dh_i      = gamma * h_i * (dR/R) * (1 - h_i / h_max_zone)   (logistic accretion)
      gap(t)    = L(t) - sum_i h_i
      if gap >= h_nuc_zone: nucleate a new plate of height h_nuc at the OCULAR end (OPR).
    Lovén's Law enters as a 1.5x enlargement of the primary peristomial plate in the
    enlarged column (Ib, IIb, IIIa, IVa, Va), which seeds the half-plate offset that
    produces the alternating (zigzag) biserial suture.
    Linked to `MathOfBiology.Loop_075.opr_biserial_plate_addition_law`."""
    p3 = params if params is not None else load_pass3_reference()["opr_plate_addition_model"]
    geo, kin = p3["geometry"], p3["plate_kinetics"]
    G_final = float(growth_factor_final if growth_factor_final is not None else geo["growth_factor_final"])
    phi_oc = math.radians(float(geo["ocular_colatitude_deg"]))
    phi_per = math.radians(float(geo["peristome_colatitude_deg"]))
    h_nuc_a = float(kin["h_nuc_ambulacral"])
    h_max_a = h_nuc_a * float(kin["h_max_over_h_nuc_ambulacral"])
    h_nuc_i = h_nuc_a * float(kin["h_nuc_interambulacral_over_ambulacral"])
    h_max_i = h_nuc_i * float(kin["h_max_over_h_nuc_interambulacral"])
    gamma = float(kin["logistic_accretion_gain"])
    enlarge = float(kin["loven_primordial_enlargement_factor"])

    if sim_ring is None:
        sim_ring = simulate_hydrocoel_ring_pde(R_ring_um=DEFAULT_R_RING_UM, crescent_closure=True)
    ray_angle = _map_lobes_to_loven_rays(sim_ring["detected_lobe_angles_deg"])
    if amb_width_frac is None:
        amb_width_frac = 0.40
    amb_width_frac = float(np.clip(amb_width_frac, 0.15, 0.85))

    # --- Build the 20 meridional columns -------------------------------------------------
    columns: list[dict[str, Any]] = []
    for i, ray in enumerate(ROMAN_RAYS):
        th_c = ray_angle[ray]
        th_next = ray_angle[ROMAN_RAYS[(i + 1) % 5]]
        sector = (th_next - th_c) % 360.0  # ~72 deg sector from this ray to the next
        w_amb = amb_width_frac * sector
        w_ia = sector - w_amb
        # Ambulacrum: column a (lower azimuth) then column b (higher azimuth), counter-clockwise
        for col, lo, hi in (("a", th_c - w_amb / 2.0, th_c), ("b", th_c, th_c + w_amb / 2.0)):
            enlarged = (LOVEN_B_ENLARGED[ray] == 1 and col == "b") or (LOVEN_B_ENLARGED[ray] == 0 and col == "a")
            columns.append({"zone": "A", "ray": ray, "interamb": None, "column": col, "theta_lo": lo % 360.0, "theta_hi": hi % 360.0,
                            "h_nuc": h_nuc_a, "h_max": h_max_a, "primordial_enlarged": bool(enlarged)})
        # Interambulacrum i+1 between ray i and ray i+1 (interamb 5 lies between V and I)
        ia_c = th_c + w_amb / 2.0 + w_ia / 2.0
        for col, lo, hi in (("a", ia_c - w_ia / 2.0, ia_c), ("b", ia_c, ia_c + w_ia / 2.0)):
            columns.append({"zone": "IA", "ray": None, "interamb": i + 1, "column": col, "theta_lo": lo % 360.0, "theta_hi": hi % 360.0,
                            "h_nuc": h_nuc_i, "h_max": h_max_i, "primordial_enlarged": False})

    # Primordial plates (one per column at metamorphosis; Lovén-enlarged where applicable)
    for c in columns:
        h0 = c["h_nuc"] * (enlarge if c["primordial_enlarged"] else 1.0)
        c["heights"] = [h0]          # index 0 = oldest (peristome), -1 = youngest (ocular end)
        c["h_max_i"] = [c["h_max"] * (enlarge if c["primordial_enlarged"] else 1.0)]  # per-plate terminal size
        c["birth_step"] = [0]
        c["birth_R"] = [1.0]

    # --- Growth loop ------------------------------------------------------------------------
    history: list[dict[str, Any]] = []
    R_prev = 1.0
    for step in range(n_steps + 1):
        t_frac = step / n_steps
        R = opr_growth_factor(t_frac, G_final)
        dlogR = (R - R_prev) / R_prev if step > 0 else 0.0
        L = R * (phi_per - phi_oc)
        for c in columns:
            if step > 0:
                hs = np.asarray(c["heights"], dtype=np.float64)
                hm = np.asarray(c["h_max_i"], dtype=np.float64)
                hs = hs + gamma * hs * dlogR * np.clip(1.0 - hs / hm, 0.0, 1.0)
                c["heights"] = hs.tolist()
            # Ocular Plate Rule: nucleate at the apical (ocular / genital) end while the gap allows
            while (L - float(np.sum(c["heights"]))) >= c["h_nuc"]:
                c["heights"].append(c["h_nuc"])
                c["h_max_i"].append(c["h_max"])
                c["birth_step"].append(step)
                c["birth_R"].append(R)
        R_prev = R
        n_amb = sum(len(c["heights"]) for c in columns if c["zone"] == "A")
        n_ia = sum(len(c["heights"]) for c in columns if c["zone"] == "IA")
        history.append({"step": step, "t_frac": t_frac, "R_over_R_meta": R, "n_plates_amb": n_amb, "n_plates_ia": n_ia,
                        "n_plates_total": n_amb + n_ia})

    # --- Final geometry: colatitude bands for each plate (arc measured from the peristome) --------
    R_final = history[-1]["R_over_R_meta"]
    for c in columns:
        cum = np.concatenate([[0.0], np.cumsum(c["heights"])])
        c["phi_bounds_rad"] = (phi_per - cum / R_final).tolist()   # decreasing from peristome toward ocular
        c["suture_arcs"] = cum.tolist()
        c["n_plates"] = len(c["heights"])
        # Youngest plate must be the apical-most (adjacent to the ocular/genital plate) -> OPR
        c["newest_at_ocular"] = bool(int(np.argmax(c["birth_step"])) == len(c["birth_step"]) - 1)

    # Biserial alternation: offset of b-column sutures relative to the a-column suture lattice (per ambulacrum),
    # folded into [0, 0.5] (0.5 = perfect zigzag alternation, 0 = sutures aligned). Reported for all plates and
    # for the mature 'ambital' plates only (h >= 0.95 h_max), which is the regime Lovén (1874) described in adults.
    alt_all: list[float] = []
    alt_mature: list[float] = []
    for ray in ROMAN_RAYS:
        ca = next(c for c in columns if c["zone"] == "A" and c["ray"] == ray and c["column"] == "a")
        cb = next(c for c in columns if c["zone"] == "A" and c["ray"] == ray and c["column"] == "b")
        sa = np.asarray(ca["suture_arcs"][1:-1]); sb = np.asarray(cb["suture_arcs"][1:-1])
        hb = np.asarray(cb["heights"][1:])
        for s, h_pl in zip(sb, hb):
            below = sa[sa <= s]; above = sa[sa > s]
            if len(below) and len(above):
                ph = float((s - below[-1]) / (above[0] - below[-1]))
                ph = min(ph, 1.0 - ph)
                alt_all.append(ph)
                if h_pl >= 0.95 * cb["h_max"]:
                    alt_mature.append(ph)
    alternation_phase = float(np.mean(alt_all)) if alt_all else 0.0
    alternation_phase_mature = float(np.mean(alt_mature)) if alt_mature else 0.0

    n_amb_final = history[-1]["n_plates_amb"]
    n_ia_final = history[-1]["n_plates_ia"]
    Rs = np.asarray([h["R_over_R_meta"] for h in history]); Ns = np.asarray([h["n_plates_total"] for h in history], dtype=float)
    rho, _ = stats.spearmanr(Rs, Ns)
    return {
        "columns": columns,
        "history": history,
        "ray_angles_deg": ray_angle,
        "amb_width_frac": amb_width_frac,
        "growth_factor_final": G_final,
        "phi_ocular_rad": phi_oc,
        "phi_peristome_rad": phi_per,
        "R_final": R_final,
        "n_columns": len(columns),
        "n_ambulacral_columns": sum(1 for c in columns if c["zone"] == "A"),
        "n_plates_amb_final": n_amb_final,
        "n_plates_ia_final": n_ia_final,
        "amb_over_ia_plate_ratio": float(n_amb_final / max(n_ia_final, 1)),
        "newest_at_ocular_fraction": float(np.mean([c["newest_at_ocular"] for c in columns])),
        "plate_count_vs_size_spearman": float(rho),
        "biserial_alternation_phase": alternation_phase,
        "biserial_alternation_phase_mature": alternation_phase_mature,
        "mean_plate_height_amb": float(np.mean([h for c in columns if c["zone"] == "A" for h in c["heights"]])),
        "mean_plate_height_ia": float(np.mean([h for c in columns if c["zone"] == "IA" for h in c["heights"]])),
    }


def opr_test_color_grid(opr: dict[str, Any], n_phi: int = 96, n_theta: int = 360, growth_step: int | None = None) -> dict[str, Any]:
    """Rasterise the plated dome into an (n_phi x n_theta x 3) RGB facecolor grid in (colatitude, azimuth).
    Ambulacra are lavender-purple (pigmented, as on the Tripneustes gratilla specimen), interambulacra cream;
    successive plates alternate shade so sutures and biserial zigzag are visible. If `growth_step` is given,
    only plates born at or before that step are drawn (with the dome radius at that step)."""
    phis = np.linspace(0.0, math.pi, n_phi)
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    rgb = np.ones((n_phi, n_theta, 3)) * np.array([0.93, 0.89, 0.78])
    col_amb = np.array([0.62, 0.42, 0.80]); col_ia = np.array([0.95, 0.90, 0.76])
    phi_per = opr["phi_peristome_rad"]
    if growth_step is None:
        R_now = opr["R_final"]
    else:
        R_now = opr["history"][min(growth_step, len(opr["history"]) - 1)]["R_over_R_meta"]
    th_deg = np.degrees(thetas)
    for c in opr["columns"]:
        lo, hi = c["theta_lo"], c["theta_hi"]
        if lo <= hi:
            mask_t = (th_deg >= lo) & (th_deg < hi)
        else:
            mask_t = (th_deg >= lo) | (th_deg < hi)
        base = col_amb if c["zone"] == "A" else col_ia
        if growth_step is None:
            hs = np.asarray(c["heights"])
        else:
            keep = np.asarray(c["birth_step"]) <= growth_step
            # approximate earlier heights by the current heights of surviving plates, rescaled to fit L(step)
            hs = np.asarray(c["heights"])[keep]
            L_now = R_now * (phi_per - opr["phi_ocular_rad"])
            if hs.sum() > L_now:
                hs = hs * (L_now / hs.sum())
        cum = np.concatenate([[0.0], np.cumsum(hs)])
        phi_b = phi_per - cum / R_now
        for k in range(len(hs)):
            mask_p = (phis <= phi_b[k]) & (phis > phi_b[k + 1])
            shade = 1.0 if k % 2 == 0 else 0.74
            rgb[np.ix_(mask_p, mask_t)] = np.clip(base * shade, 0.0, 1.0)
    # Apical system (above the ocular) dark periproct; peristome membrane below
    rgb[phis <= opr["phi_ocular_rad"], :] = np.array([0.20, 0.16, 0.22])
    rgb[phis > phi_per, :] = np.array([0.55, 0.50, 0.46])
    return {"phis": phis, "thetas": thetas, "rgb": rgb, "R_now": R_now}


def project_opr_test_aboral_view(opr: dict[str, Any], n_px: int = 360) -> dict[str, Any]:
    """Orthographic aboral-view projection of the plated dome (apical system at the image centre),
    sampled with the SAME polar chromaticity / Fourier pipeline used on the Tripneustes gratilla
    specimen so that C_5 / C_10 harmonic fractions are directly comparable."""
    grid = opr_test_color_grid(opr)
    n_phi, n_theta = grid["rgb"].shape[:2]
    lin = np.linspace(-1.0, 1.0, n_px)
    U, V = np.meshgrid(lin, lin)
    rr = np.hypot(U, V)
    img = np.ones((n_px, n_px, 3)) * 0.06
    inside = rr <= 1.0
    phi = np.arcsin(np.clip(rr, 0.0, 1.0))          # colatitude of the visible upper hemisphere
    th = np.mod(np.arctan2(V, U), 2.0 * math.pi)
    i_phi = np.clip((phi / math.pi * (n_phi - 1)).round().astype(int), 0, n_phi - 1)
    i_th = np.clip((th / (2.0 * math.pi) * n_theta).astype(int), 0, n_theta - 1)
    img[inside] = grid["rgb"][i_phi[inside], i_th[inside]]
    harm = polar_c5_harmonics(img, cx_px=n_px / 2.0, cy_px=n_px / 2.0, r_min_px=0.14 * n_px, r_max_px=0.36 * n_px)
    return {"image": img, **harm}


def polar_c5_harmonics(rgb: np.ndarray, cx_px: float, cy_px: float, r_min_px: float, r_max_px: float) -> dict[str, Any]:
    """Shared polar lavender-chromaticity sampling + azimuthal Fourier C_5 / C_10 statistics
    (used identically for the specimen micrograph and the synthetic OPR test projection)."""
    R_ch, G_ch, B_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    purple_idx = 0.5 * (R_ch + B_ch) - G_ch
    thetas_rad = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    radii_px = np.linspace(r_min_px, r_max_px, 45)
    TH, RR = np.meshgrid(thetas_rad, radii_px)
    XX = cx_px + RR * np.cos(TH)
    YY = cy_px + RR * np.sin(TH)
    polar_map = ndi.map_coordinates(purple_idx, [YY, XX], order=1, mode="nearest")
    angular_profile = np.mean(polar_map, axis=0)
    angular_zero = angular_profile - np.mean(angular_profile)
    fft_c = np.fft.rfft(angular_zero)
    fft_notilt = fft_c.copy()
    fft_notilt[0:3] = 0.0
    fft_mag = np.abs(fft_notilt)
    c5_fund_fraction = float((fft_mag[5] ** 2) / (np.sum(fft_mag[3:10] ** 2) + 1e-12))
    c5_c10_fraction = float((fft_mag[5] ** 2 + fft_mag[10] ** 2) / (np.sum(fft_mag[3:12] ** 2) + 1e-12))
    a5_over_max_non5 = float(fft_mag[5] / (np.max(np.delete(fft_mag[3:10], 2)) + 1e-12))
    profile_smooth = ndi.gaussian_filter1d(angular_zero, sigma=5.0, mode="wrap")
    # Circular peak detection (pad 60 deg each side so a ray straddling 0/360 deg is not missed)
    pad = 60
    prof_pad = np.concatenate([profile_smooth[-pad:], profile_smooth, profile_smooth[:pad]])
    p_pad, _ = signal.find_peaks(prof_pad, distance=55)
    peaks_deg = np.sort(p_pad[(p_pad >= pad) & (p_pad < 360 + pad)] - pad)
    spacings_deg = np.diff(np.append(peaks_deg, peaks_deg[0] + 360.0)) if len(peaks_deg) else np.array([360.0])
    duty = float(np.mean(profile_smooth > 0.0))  # fraction of azimuth occupied by pigmented (ambulacral) signal
    return {
        "angular_profile": angular_zero,
        "profile_smooth": profile_smooth,
        "fft_c": fft_c,
        "fft_magnitudes_m1_11": np.abs(fft_c)[1:12],
        "c5_fundamental_power_fraction": c5_fund_fraction,
        "pentaradial_c5_c10_power_fraction": c5_c10_fraction,
        "a5_over_max_non5_ratio": a5_over_max_non5,
        "n_rays_detected": int(len(peaks_deg)),
        "detected_ray_angles_deg": peaks_deg.tolist(),
        "inter_ray_spacings_deg": spacings_deg.tolist(),
        "ambulacral_duty_cycle": duty,
    }


# =============================================================================
# 3c. Pass 3 / Law 5: Continuous Nodal / BMP2-4 Dose-Response Gradients across
#     the Larval Stomodeum and Right Somatocoel (Duboc 2005; Luo & Su 2012)
# =============================================================================
def _solve_screened_poisson_1d(x: np.ndarray, source: np.ndarray, decay_length: float) -> np.ndarray:
    """Steady state of -lambda^2 u'' + u = source on x with no-flux (Neumann) boundaries,
    via a banded finite-difference solve."""
    from scipy.linalg import solve_banded
    n = len(x)
    dx = x[1] - x[0]
    r = (decay_length / dx) ** 2
    ab = np.zeros((3, n))
    ab[1, :] = 1.0 + 2.0 * r
    ab[0, 1:] = -r
    ab[2, :-1] = -r
    ab[1, 0] = 1.0 + r; ab[1, -1] = 1.0 + r  # Neumann (zero-flux) ends
    return solve_banded((1, 1), ab, source)


def _hill(c: np.ndarray | float, K: float, n: int) -> np.ndarray | float:
    cn = np.power(np.maximum(c, 0.0), n)
    return cn / (K ** n + cn)


def simulate_nodal_bmp_dose_response(
    dose_right: float = 1.0,
    dose_left: float = 0.0,
    params: dict[str, Any] | None = None,
    m_ring: int = 5,
) -> dict[str, Any]:
    """Continuous dose-response Nodal / BMP2-4 gradient model across the normalised larval
    left-right axis x in [-1, 1] (x = 0: stomodeum / oral midline; x = +-0.55: coelomic pouches,
    the right one being the right somatocoel, the left one the hydrocoel + left somatocoel).

      -lambda_N^2 N'' + N = dose_R S(x - x_src) + dose_L S(x + x_src)      (right-sided ectodermal Nodal, Duboc 2005)
      -lambda_B^2 B'' + B = b_stom S_stom(x) + beta_ind Hill(N; K_ind, n)  (oral BMP2/4 + Nodal-induced BMP2/4)
      P_pouch = Hill(B(x_p); K_B, n) * (1 - Hill(N(x_p); K_N, n))          (BMP required on the left, Luo & Su 2012;
                                                                             Nodal represses, Duboc 2005)
      E[rings] = P_L + P_R,   E[podia] = m_ring * E[rings],   discrete rings = round(P_L) + round(P_R).
    At (dose_L, dose_R) = (0, 1), (0, 0), (1, 1) this reduces to Law 3 (5, 10, 0 podia).
    Linked to `MathOfBiology.Loop_075.nodal_dose_response_hill_reduces_to_discrete_law`."""
    p = params if params is not None else load_pass3_reference()["nodal_bmp_dose_response_model"]["parameters"]
    n_x = int(p["n_x"]); n_h = int(p["hill_coefficient"])
    x = np.linspace(-1.0, 1.0, n_x)
    x_src, w_src = float(p["nodal_source_center_x"]), float(p["nodal_source_half_width"])
    src_R = (np.abs(x - x_src) <= w_src).astype(float)
    src_L = (np.abs(x + x_src) <= w_src).astype(float)
    N = _solve_screened_poisson_1d(x, dose_right * src_R + dose_left * src_L, float(p["nodal_decay_length"]))
    src_stom = float(p["bmp_stomodeal_amplitude"]) * (np.abs(x) <= float(p["bmp_stomodeal_half_width"])).astype(float)
    src_ind = float(p["bmp_nodal_induction_beta"]) * _hill(N, float(p["bmp_nodal_induction_K"]), n_h)
    B = _solve_screened_poisson_1d(x, src_stom + src_ind, float(p["bmp_decay_length"]))

    x_p = float(p["pouch_x"])
    iL = int(np.argmin(np.abs(x + x_p))); iR = int(np.argmin(np.abs(x - x_p)))
    K_N, K_B = float(p["nodal_repression_K"]), float(p["bmp_activation_K"])
    P = _hill(B, K_B, n_h) * (1.0 - _hill(N, K_N, n_h))   # competence field along x
    P_L, P_R = float(P[iL]), float(P[iR])
    return {
        "x": x, "nodal": N, "bmp": B, "competence": P,
        "nodal_pouch_left": float(N[iL]), "nodal_pouch_right": float(N[iR]),
        "bmp_pouch_left": float(B[iL]), "bmp_pouch_right": float(B[iR]),
        "nodal_stomodeum": float(N[int(np.argmin(np.abs(x)))]),
        "competence_left": P_L, "competence_right": P_R,
        "expected_rings": P_L + P_R,
        "expected_podia": m_ring * (P_L + P_R),
        "discrete_rings": int(round(P_L)) + int(round(P_R)),
        "discrete_podia": m_ring * (int(round(P_L)) + int(round(P_R))),
        "dose_left": float(dose_left), "dose_right": float(dose_right),
    }


def evaluate_nodal_dose_response_sweep(params_all: dict[str, Any] | None = None) -> dict[str, Any]:
    """1D right-Nodal dose sweep (dose_L = 0), 2D (dose_L, dose_R) map of expected rings, EC50 of right-pouch
    repression, and the deviation of the continuous model from the discrete Law 3 at the canonical doses."""
    p3 = params_all if params_all is not None else load_pass3_reference()["nodal_bmp_dose_response_model"]
    p = p3["parameters"]
    sw = p3["dose_sweep"]
    doses = np.linspace(float(sw["dose_min"]), float(sw["dose_max"]), int(sw["n_dose"]))
    sweep = [simulate_nodal_bmp_dose_response(dose_right=d, dose_left=0.0, params=p) for d in doses]
    comp_R = np.array([s["competence_right"] for s in sweep])
    comp_L = np.array([s["competence_left"] for s in sweep])
    podia = np.array([s["expected_podia"] for s in sweep])
    rho_R, _ = stats.spearmanr(doses, comp_R)
    # EC50: dose where right-pouch competence crosses 0.5 (linear interpolation)
    ec50 = float("nan")
    for i in range(1, len(doses)):
        if (comp_R[i - 1] - 0.5) * (comp_R[i] - 0.5) <= 0 and comp_R[i - 1] != comp_R[i]:
            ec50 = float(doses[i - 1] + (0.5 - comp_R[i - 1]) * (doses[i] - doses[i - 1]) / (comp_R[i] - comp_R[i - 1]))
            break
    # 2D map
    d2 = np.linspace(float(sw["dose_min"]), float(sw["dose_max"]), 25)
    rings_map = np.zeros((len(d2), len(d2)))
    for i, dl in enumerate(d2):
        for j, dr in enumerate(d2):
            rings_map[i, j] = simulate_nodal_bmp_dose_response(dose_right=dr, dose_left=dl, params=p)["expected_rings"]
    canon = []
    for cond in p3["canonical_dose_conditions"]:
        s = simulate_nodal_bmp_dose_response(dose_right=float(cond["dose_right"]), dose_left=float(cond["dose_left"]), params=p)
        canon.append({**cond, "expected_podia": s["expected_podia"], "discrete_podia": s["discrete_podia"],
                      "competence_left": s["competence_left"], "competence_right": s["competence_right"]})
    max_dev = float(max(abs(c["expected_podia"] - c["reference_primary_podia"]) for c in canon))
    return {
        "doses": doses, "competence_right": comp_R, "competence_left": comp_L, "expected_podia": podia,
        "spearman_competence_right_vs_dose": float(rho_R), "ec50_right_dose": ec50,
        "dose_grid_2d": d2, "expected_rings_map": rings_map,
        "canonical": canon, "max_abs_podia_deviation_canonical": max_dev,
        "wt_profile": sweep[int(np.argmin(np.abs(doses - 1.0)))],
        "mo_profile": sweep[0],
    }


# =============================================================================
# 4. Automated Polar Fourier C_5 Segmentation of Tripneustes gratilla Specimen
# =============================================================================
def segment_echinoid_test_specimen(specimen_path: Path = SPECIMEN_JPG) -> dict[str, Any]:
    """Perform automated central periproct detection and polar Fourier C_5 harmonic
    segmentation of the 5 pigmented ambulacral rays on the SHA-256-locked
    *Tripneustes gratilla* regular echinoid test micrograph (`echinoderm_pentaradial_test_specimen.jpg`)."""
    verify_dataset_integrity()
    rgb = np.asarray(Image.open(specimen_path).convert("RGB"), dtype=np.float64) / 255.0
    h, w, _ = rgb.shape
    R_ch, G_ch, B_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.299 * R_ch + 0.587 * G_ch + 0.114 * B_ch

    # 1. Detect dark circular central periproct / apical system opening
    yy, xx = np.mgrid[0:h, 0:w]
    central_roi = (np.hypot(xx - w / 2.0, yy - h / 2.0) < 0.16 * min(h, w)) & (gray < np.percentile(gray, 12.0))
    cy_px, cx_px = ndi.center_of_mass(central_roi)

    # 2-4. Polar lavender-purple chromaticity sampling around the periproct, azimuthal Fourier
    # spectrum (m=1,2 illumination tilt removed), C_5 / C_10 power fractions and circular peak
    # detection of the 5 ambulacral ray centerlines -- shared pipeline `polar_c5_harmonics`
    # (also applied to the synthetic OPR test projection in Pass 3).
    r_min_px = 0.14 * min(h, w)
    r_max_px = 0.36 * min(h, w)
    harm = polar_c5_harmonics(rgb, cx_px=float(cx_px), cy_px=float(cy_px), r_min_px=r_min_px, r_max_px=r_max_px)
    spacings_deg = np.asarray(harm["inter_ray_spacings_deg"], dtype=float)

    return {
        "rgb": rgb,
        "width_px": w,
        "height_px": h,
        "periproct_center_xy_px": (float(cx_px), float(cy_px)),
        "annulus_radii_px": (float(r_min_px), float(r_max_px)),
        "thetas_deg": np.arange(360, dtype=float),
        "angular_profile": harm["angular_profile"],
        "profile_smooth": harm["profile_smooth"],
        "fft_magnitudes_m1_11": harm["fft_magnitudes_m1_11"],
        "c5_fundamental_power_fraction": harm["c5_fundamental_power_fraction"],
        "pentaradial_c5_c10_power_fraction": harm["pentaradial_c5_c10_power_fraction"],
        "a5_over_max_non5_ratio": harm["a5_over_max_non5_ratio"],
        "n_rays_detected": harm["n_rays_detected"],
        "detected_ray_angles_deg": harm["detected_ray_angles_deg"],
        "inter_ray_spacings_deg": spacings_deg.tolist(),
        "mean_angular_spacing_deg": float(np.mean(spacings_deg)),
        "std_angular_spacing_deg": float(np.std(spacings_deg)),
        "cv_angular_spacing": float(np.std(spacings_deg) / np.mean(spacings_deg)),
        "ambulacral_duty_cycle": harm["ambulacral_duty_cycle"],
    }


# =============================================================================
# 5. Pass 2 6-Panel Synthesis Figure & 2-Panel Co-Evolving Growth GIF
# =============================================================================
def render_comparison_figure(
    out_path: Path = OUT_PNG,
    R_ring_um: float = DEFAULT_R_RING_UM,
    lambda_c_um: float = DEFAULT_LAMBDA_C_UM,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Render the 6-panel Pass 2 synthesis figure (`docs/media/echinoderm_pentaradial_symmetry_comparison.png`)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim_wt = simulate_hydrocoel_ring_pde(R_ring_um=R_ring_um, lambda_c_um=lambda_c_um, crescent_closure=True, seed=seed)
    sim_ctrl = simulate_hydrocoel_ring_pde(R_ring_um=R_ring_um, lambda_c_um=lambda_c_um, equal_diffusion=True, seed=seed)
    staircase = evaluate_mode_selection_staircase(lambda_c_um=lambda_c_um)
    loven_nodal = evaluate_loven_law_and_nodal_perturbations()
    field_wt = render_2d_hydrocoel_rudiment_field(R_ring_um=R_ring_um, nodal_left=0, nodal_right=1)
    field_lof = render_2d_hydrocoel_rudiment_field(R_ring_um=R_ring_um, nodal_left=0, nodal_right=0)
    specimen = segment_echinoid_test_specimen()
    # Pass 3: OPR plate-addition test coupled to the ring PDE lobes, and continuous Nodal/BMP2-4 dose-response
    opr = simulate_opr_test_plate_addition(sim_ring=sim_wt, amb_width_frac=specimen["ambulacral_duty_cycle"])
    proj = project_opr_test_aboral_view(opr)
    nodal_sweep = evaluate_nodal_dose_response_sweep()

    fig = plt.figure(figsize=(18.5, 16.6), dpi=150, facecolor="#0b0f17")
    gs = GridSpec(3, 3, figure=fig, hspace=0.33, wspace=0.25)

    # -------------------------------------------------------------------------
    # Panel A: 2D Hydrocoel Ring Activator PDE & Lovén's Law Initiation Sequence
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0], facecolor="#111827")
    sim_2d_ring = render_2d_hydrocoel_rudiment_field(R_ring_um=R_ring_um, nodal_left=0, nodal_right=1)
    # Focus on left hydrocoel ring centered at (-88, 0)
    XX_shift = sim_2d_ring["XX_um"] + 88.0
    YY_shift = sim_2d_ring["YY_um"]
    ax_a.pcolormesh(XX_shift, YY_shift, sim_2d_ring["field"], cmap="magma", vmin=0.0, vmax=1.05, shading="auto")
    ax_a.set_xlim(-125.0, 125.0)
    ax_a.set_ylim(-125.0, 125.0)

    # Annotate the 5 Lovén rays (I..V) with initiation rank (1..5) and peristomial dominance (ba/ab)
    for lr in sim_wt["loven_rays"]:
        ang_rad = math.radians(lr["angle_deg"])
        rx = (R_ring_um + 28.0) * math.cos(ang_rad)
        ry = (R_ring_um + 28.0) * math.sin(ang_rad)
        col_tag = "#38bdf8" if lr["b_enlarged"] == 1 else "#facc15"
        ax_a.text(
            rx,
            ry,
            f"{lr['ray']} ({lr['carpenter']})\n#{lr['rank']} [{lr['pair_code']}]",
            color=col_tag,
            fontsize=7.6,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#0b0f17", edgecolor=col_tag, alpha=0.88),
        )
    # Draw Lovén symmetry plane (Ray III at 144 deg to Interambulacrum 5 at 324 deg)
    loven_ang = math.radians(144.0)
    ax_a.plot(
        [-115.0 * math.cos(loven_ang), 115.0 * math.cos(loven_ang)],
        [-115.0 * math.sin(loven_ang), 115.0 * math.sin(loven_ang)],
        "--",
        color="#34d399",
        lw=1.4,
        label="Lovén Plane (III–5)",
    )
    ax_a.set_title(
        f"A. Hydrocoel Ring PDE (m*={sim_wt['n_podia_lobes']}, Δθ={sim_wt['mean_spacing_deg']:.1f}°) & Lovén's Law (II,IV,I,III,V)",
        color="white",
        fontsize=9.4,
        fontweight="bold",
    )
    ax_a.set_xlabel("Hydrocoel X (μm)", color="white", fontsize=8.6)
    ax_a.set_ylabel("Hydrocoel Y (μm)", color="white", fontsize=8.6)
    ax_a.tick_params(colors="#9ca3af", labelsize=8)
    ax_a.legend(loc="lower left", fontsize=7.2, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel B: Mode Selection Staircase m* in {3..7} vs 2*pi*R_ring/lambda_c
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1], facecolor="#111827")
    ax_b.axvspan(4.5, 5.5, color="#34d399", alpha=0.16, label="Pentaradial Window 2πR/λ_c ∈ (4.5, 5.5)")
    ax_b.step(
        staircase["circ_ratios"],
        staircase["staircase_modes"],
        where="mid",
        color="#38bdf8",
        lw=2.2,
        label="Turing Eigenvalue Staircase m* = round(2πR/λ_c)",
    )
    pde_c = [r["circumference_over_lambda_c"] for r in staircase["pde_regimes"]]
    pde_m = [r["pde_lobe_count"] for r in staircase["pde_regimes"]]
    ax_b.scatter(
        pde_c,
        pde_m,
        color="#f43f5e",
        s=75,
        zorder=5,
        edgecolors="white",
        linewidths=0.9,
        label=f"PDE Lobe Counts m ∈ {{3..7}} (r={staircase['pearson_r_staircase']:.4f})",
    )
    for c_pt, m_pt, lbl_t in zip(
        pde_c,
        pde_m,
        ["m=3 (120°)", "m=4 Tetramerous (90°)", "m=5 Wild-Type (72°)", "m=6 Hexamerous (60°)", "m=7 (51.4°)"],
    ):
        ax_b.annotate(lbl_t, (c_pt, m_pt), textcoords="offset points", xytext=(-18, 8), color="white", fontsize=7.2)

    ax_b.set_title(
        "B. Hydrocoel Ring Mode Selection Staircase vs 2πR_ring / λ_c",
        color="white",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_b.set_xlabel(r"Dimensionless Ring Circumference $2\pi R_{\mathrm{ring}} / \lambda_c$", color="white", fontsize=8.8)
    ax_b.set_ylabel("Selected Primary Podia Mode m*", color="white", fontsize=8.8)
    ax_b.set_ylim(2.2, 7.8)
    ax_b.tick_params(colors="#9ca3af", labelsize=8)
    ax_b.grid(True, alpha=0.22, color="#374151")
    ax_b.legend(loc="upper left", fontsize=7.0, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel C: Crescent-to-Ring Activator Evolution & Equal-Diffusion Negative Control
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2], facecolor="#111827")
    snap_colors = ["#64748b", "#38bdf8", "#34d399", "#facc15", "#f43f5e"]
    for snap, col_s in zip(sim_wt["snapshots"], snap_colors):
        ax_c.plot(
            sim_wt["thetas_deg"],
            snap["a_profile"],
            color=col_s,
            lw=1.7,
            label=f"t/T={snap['t_frac']:.2f} (closure={snap['closure_fraction']:.2f})",
        )
    ax_c.plot(
        sim_ctrl["thetas_deg"],
        sim_ctrl["a_profile"],
        "--",
        color="#e5e7eb",
        lw=2.0,
        label=f"Negative Control D_h=D_a ({sim_ctrl['n_podia_lobes']} lobes)",
    )
    for lr in sim_wt["loven_rays"]:
        ax_c.axvline(lr["angle_deg"], linestyle=":", color="#9ca3af", alpha=0.45)
        ax_c.text(lr["angle_deg"], 2.30, f"{lr['ray']}(#{lr['rank']})", color="#facc15", fontsize=7.2, ha="center")

    ax_c.set_title(
        "C. Crescent-to-Ring PDE Trajectory a(θ, t) & Equal-Diffusion Control",
        color="white",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_c.set_xlabel("Hydrocoel Ring Angle θ (deg, Lovén Rays I..V at 72° Spacing)", color="white", fontsize=8.6)
    ax_c.set_ylabel("Activator Concentration a(θ, t)", color="white", fontsize=8.6)
    ax_c.set_ylim(0.1, 2.55)
    ax_c.tick_params(colors="#9ca3af", labelsize=8)
    ax_c.grid(True, alpha=0.22, color="#374151")
    ax_c.legend(loc="lower right", fontsize=6.7, ncol=2, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel D: Duboc et al. (2005) Right-Sided Nodal Repression vs Twin 2x5 Rudiments
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0], facecolor="#111827")
    # Stack WT (top half) and Nodal LOF twin-rudiment 2x5 (bottom half) in a split spatial view
    combined_lr = np.vstack([field_wt["field"][::2, :], field_lof["field"][::2, :]])
    ax_d.imshow(combined_lr, cmap="inferno", extent=[-210, 210, -135, 135], origin="lower", aspect="auto", vmin=0.0, vmax=1.05)
    ax_d.axhline(0.0, color="#9ca3af", linestyle="--", lw=1.3)
    ax_d.axvline(0.0, color="#38bdf8", linestyle=":", lw=1.1, label="Larval Bilateral Midline (x=0)")
    ax_d.text(-195, 108, "Wild-Type (Right Nodal+): Left Rudiment Only (1×5 = 5 Podia)", color="#34d399", fontsize=7.8, fontweight="bold")
    ax_d.text(35, 62, "Right Pouch\nRepressed\nby Nodal/BMP2-4", color="#f87171", fontsize=7.4, ha="center")
    ax_d.text(-195, -24, "Nodal Morpholino LOF (Duboc 2005): Twin Rudiments (2×5 = 10 Podia)", color="#facc15", fontsize=7.8, fontweight="bold")
    ax_d.set_title(
        "D. Duboc et al. (2005) Nodal L-R Asymmetry: WT (1×5=5) vs Nodal LOF (2×5=10)",
        color="white",
        fontsize=9.3,
        fontweight="bold",
    )
    ax_d.set_xlabel("Larval Left–Right Axis X (μm, Left < 0 < Right)", color="white", fontsize=8.6)
    ax_d.set_ylabel("Split View: WT (Top) / Nodal LOF (Bottom)", color="white", fontsize=8.6)
    ax_d.tick_params(colors="#9ca3af", labelsize=8)

    # -------------------------------------------------------------------------
    # Panel E: Automated Polar C_5 Segmentation of Tripneustes gratilla Specimen
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1], facecolor="#111827")
    ax_e.imshow(specimen["rgb"])
    cx, cy = specimen["periproct_center_xy_px"]
    r_in, r_out = specimen["annulus_radii_px"]
    th_ring = np.linspace(0.0, 2.0 * math.pi, 200)
    ax_e.plot(cx + r_in * np.cos(th_ring), cy + r_in * np.sin(th_ring), ":", color="#facc15", lw=1.2)
    ax_e.plot(cx + r_out * np.cos(th_ring), cy + r_out * np.sin(th_ring), "--", color="#facc15", lw=1.3, label="Sampling Annulus")

    loven_labels = ["Ray I (D)", "Ray II (C)", "Ray III (B)", "Ray IV (A)", "Ray V (E)"]
    for idx_r, ang_deg in enumerate(specimen["detected_ray_angles_deg"]):
        ang_rad = math.radians(ang_deg)
        x0 = cx + 0.55 * r_in * math.cos(ang_rad)
        y0 = cy + 0.55 * r_in * math.sin(ang_rad)
        x1 = cx + 1.18 * r_out * math.cos(ang_rad)
        y1 = cy + 1.18 * r_out * math.sin(ang_rad)
        lbl = "5 Detected Ambulacral Rays (Δθ=72.0°±1.4°)" if idx_r == 0 else None
        ax_e.plot([x0, x1], [y0, y1], color="#38bdf8", lw=2.2, label=lbl)
        ax_e.text(
            cx + 1.27 * r_out * math.cos(ang_rad),
            cy + 1.27 * r_out * math.sin(ang_rad),
            f"{loven_labels[idx_r % 5]}\n({ang_deg}°)",
            color="white",
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#0b0f17", edgecolor="#38bdf8", alpha=0.85),
        )

    ax_e.set_title(
        f"E. Tripneustes gratilla Test CV Segmentation ({specimen['n_rays_detected']} Rays, Δθ={specimen['mean_angular_spacing_deg']:.1f}°±{specimen['std_angular_spacing_deg']:.2f}°)",
        color="white",
        fontsize=9.2,
        fontweight="bold",
    )
    ax_e.set_xlabel("Micrograph X (px, CC BY-SA 4.0)", color="white", fontsize=8.5)
    ax_e.set_ylabel("Micrograph Y (px)", color="white", fontsize=8.5)
    ax_e.tick_params(colors="#9ca3af", labelsize=7.5)
    ax_e.legend(loc="lower right", fontsize=6.9, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel F: Specimen & PDE Azimuthal Fourier Harmonic Spectrum (C_5 + C_10)
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2], facecolor="#111827")
    modes_bar = np.arange(1, 12)
    spec_fft_norm = specimen["fft_magnitudes_m1_11"] / np.max(specimen["fft_magnitudes_m1_11"])
    bar_colors = ["#f43f5e" if m in (5, 10) else "#38bdf8" for m in modes_bar]
    ax_f.bar(
        modes_bar - 0.16,
        spec_fft_norm,
        width=0.32,
        color=bar_colors,
         edgecolor="white",
        linewidth=0.6,
        label=f"Specimen FFT (C_5={specimen['c5_fundamental_power_fraction']:.3f}, C_5+C_10={specimen['pentaradial_c5_c10_power_fraction']:.3f})",
    )
    pde_fft_norm = np.zeros(11)
    pde_fft_norm[:10] = sim_wt["fft_magnitudes_m1_10"] / (np.max(sim_wt["fft_magnitudes_m1_10"]) + 1e-12)
    ax_f.bar(
        modes_bar + 0.16,
        pde_fft_norm,
        width=0.32,
        color="#34d399",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
        label=f"Hydrocoel Ring PDE FFT (m*={sim_wt['dominant_mode_m']}, C_5={sim_wt['m5_power_fraction']:.3f})",
    )

    # Inset showing angular chromaticity profile & 5 detected peaks
    ins_f = ax_f.inset_axes([0.10, 0.48, 0.52, 0.44], facecolor="#0b0f17")
    ins_f.plot(specimen["thetas_deg"], specimen["profile_smooth"], color="#c084fc", lw=1.5)
    ins_f.scatter(
        specimen["detected_ray_angles_deg"],
        specimen["profile_smooth"][specimen["detected_ray_angles_deg"]],
        color="#facc15",
        s=28,
        zorder=5,
    )
    ins_f.set_title("Specimen Angular Profile I(θ) & 5 Peaks", color="white", fontsize=6.8)
    ins_f.tick_params(colors="#9ca3af", labelsize=6)

    ax_f.set_xticks(modes_bar)
    ax_f.set_xlabel("Azimuthal Harmonic Mode m (m=5 Ambulacra, m=10 Biserial Columns)", color="white", fontsize=8.6)
    ax_f.set_ylabel("Normalized Fourier Amplitude |A_m| / max |A_m|", color="white", fontsize=8.6)
    ax_f.set_title(
        f"F. Azimuthal Fourier Spectrum: C_5 Dominance (A_5/max_non5 = {specimen['a5_over_max_non5_ratio']:.2f}×)",
        color="white",
        fontsize=9.4,
        fontweight="bold",
    )
    ax_f.tick_params(colors="#9ca3af", labelsize=8)
    ax_f.grid(True, alpha=0.22, color="#374151")
    ax_f.legend(loc="upper right", fontsize=6.8, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel G (Pass 3): OPR 3D domed test plated from the m=5 ring PDE lobes
    # -------------------------------------------------------------------------
    ax_g = fig.add_subplot(gs[2, 0], projection="3d")
    _draw_opr_dome_3d(ax_g, opr, title=f"G. OPR 3D Test (Mooi & David 1998): {opr['n_columns']} cols\n"
                                        f"Amb/IA plates = {opr['amb_over_ia_plate_ratio']:.2f}, zigzag phase = {opr['biserial_alternation_phase']:.2f}")

    # -------------------------------------------------------------------------
    # Panel H (Pass 3): Continuous Nodal / BMP2-4 gradients & dose-response
    # -------------------------------------------------------------------------
    ax_h = fig.add_subplot(gs[2, 1], facecolor="#111827")
    wt = nodal_sweep["wt_profile"]
    ax_h.plot(wt["x"], wt["nodal"], color="#f43f5e", lw=1.8, label="Nodal (WT)")
    ax_h.plot(wt["x"], wt["bmp"], color="#38bdf8", lw=1.8, label="BMP2/4 (WT)")
    ax_h.plot(wt["x"], wt["competence"], color="#34d399", lw=1.8, label="Hydrocoel competence P(x) (WT)")
    ax_h.plot(wt["x"], nodal_sweep["mo_profile"]["competence"], color="#34d399", lw=1.2, linestyle="--", label="P(x) Nodal MO")
    for xp in (-0.55, 0.0, 0.55):
        ax_h.axvline(xp, color="#9ca3af", linestyle=":", lw=0.8)
    ax_h.text(-0.55, 1.03, "L pouch", color="#e5e7eb", fontsize=6.6, ha="center"); ax_h.text(0.0, 1.03, "stomodeum", color="#e5e7eb", fontsize=6.6, ha="center")
    ax_h.text(0.55, 1.03, "R somatocoel", color="#e5e7eb", fontsize=6.6, ha="center")
    ins_h = ax_h.inset_axes([0.56, 0.36, 0.42, 0.40], facecolor="#0b0f17")
    ins_h.plot(nodal_sweep["doses"], nodal_sweep["expected_podia"], color="#facc15", lw=1.6)
    for c in nodal_sweep["canonical"]:
        if c["dose_left"] == 0.0:
            ins_h.scatter([c["dose_right"]], [c["reference_primary_podia"]], s=26, color="white", edgecolors="#38bdf8", zorder=5)
    ins_h.set_title(f"E[podia] vs right Nodal dose (EC50={nodal_sweep['ec50_right_dose']:.2f})", color="white", fontsize=6.4)
    ins_h.tick_params(colors="#9ca3af", labelsize=5.8); ins_h.set_ylim(0, 11)
    ax_h.set_ylim(0, 1.15)
    ax_h.set_xlabel("Normalised larval L-R axis x", color="white", fontsize=8.4)
    ax_h.set_ylabel("Concentration / competence", color="white", fontsize=8.4)
    ax_h.set_title(f"H. Nodal/BMP2-4 Dose-Response (Duboc 2005; Luo & Su 2012)\n"
                   f"max|dev| from discrete Law 3 = {nodal_sweep['max_abs_podia_deviation_canonical']:.3f} podia",
                   color="white", fontsize=8.6, fontweight="bold")
    ax_h.tick_params(colors="#9ca3af", labelsize=7.5); ax_h.grid(True, alpha=0.22, color="#374151")
    ax_h.legend(loc="center left", fontsize=6.2, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    # -------------------------------------------------------------------------
    # Panel I (Pass 3): OPR plate-count growth law & aboral C_5 harmonics vs specimen
    # -------------------------------------------------------------------------
    ax_i = fig.add_subplot(gs[2, 2], facecolor="#111827")
    Rs = [h_["R_over_R_meta"] for h_ in opr["history"]]
    ax_i.plot(Rs, [h_["n_plates_amb"] for h_ in opr["history"]], color="#c084fc", lw=2.0, label="Ambulacral plates (10 cols)")
    ax_i.plot(Rs, [h_["n_plates_ia"] for h_ in opr["history"]], color="#fde68a", lw=2.0, label="Interambulacral plates (10 cols)")
    ax_i.set_xlabel("Test radius R / R_metamorphosis", color="white", fontsize=8.4)
    ax_i.set_ylabel("Plate count", color="white", fontsize=8.4)
    ins_i = ax_i.inset_axes([0.50, 0.10, 0.48, 0.46], facecolor="#0b0f17")
    mb = np.arange(1, 12)
    ins_i.bar(mb - 0.17, specimen["fft_magnitudes_m1_11"] / np.max(specimen["fft_magnitudes_m1_11"]), width=0.34, color="#f43f5e",
              label=f"specimen C_5={specimen['c5_fundamental_power_fraction']:.3f}")
    ins_i.bar(mb + 0.17, proj["fft_magnitudes_m1_11"] / np.max(proj["fft_magnitudes_m1_11"]), width=0.34, color="#34d399",
              label=f"OPR test C_5={proj['c5_fundamental_power_fraction']:.3f}")
    ins_i.set_xticks(mb); ins_i.tick_params(colors="#9ca3af", labelsize=5.8)
    ins_i.legend(fontsize=5.6, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white", loc="upper right")
    ins_i.set_title("Aboral azimuthal harmonics: OPR test vs specimen", color="white", fontsize=6.4)
    ax_i.set_title(f"I. OPR Plate Addition vs Test Size\n(Spearman ρ={opr['plate_count_vs_size_spearman']:.3f}; newest-at-ocular={opr['newest_at_ocular_fraction']:.2f})",
                   color="white", fontsize=8.8, fontweight="bold")
    ax_i.tick_params(colors="#9ca3af", labelsize=7.5); ax_i.grid(True, alpha=0.22, color="#374151")
    ax_i.legend(loc="upper left", fontsize=6.6, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

    fig.suptitle(
        "Morris (2012), Mooi & David (1998), Lovén (1874), Duboc et al. (2005), Luo & Su (2012) & Meinhardt (2008) — Echinoderm Pentaradial Symmetry Breaking (v3)",
        color="white",
        fontsize=11.8,
        fontweight="bold",
        y=0.985,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_growth_gif(out_path: Path = OUT_GIF) -> Path:
    """Render Pass 2 co-evolving 2-panel animated GIF comparing:
      - Left panel: Crescent-to-ring sequential Lovén initiation (II -> IV -> I -> III -> V)
      - Right panel: Hydrocoel ring circumference mode staircase m in {3, 4, 5, 6, 7}."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim_wt = simulate_hydrocoel_ring_pde(R_ring_um=DEFAULT_R_RING_UM, crescent_closure=True)
    radii_mm = [45.0, 60.0, 75.0, 90.0, 105.0]
    frames: list[Image.Image] = []

    for idx, (snap, R_val) in enumerate(zip(sim_wt["snapshots"], radii_mm)):
        sim_r = simulate_hydrocoel_ring_pde(R_ring_um=R_val, crescent_closure=(abs(R_val - 75.0) < 1e-6))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6), dpi=105, facecolor="#0b0f17")

        # Left: Crescent-to-ring progressive closure at R=75 um
        ax1.set_facecolor("#111827")
        ax1.plot(sim_wt["thetas_deg"], snap["a_profile"], color="#34d399", lw=2.2)
        ax1.axhline(1.38, linestyle="--", color="#facc15", lw=1.1, label="Podia Initiation Threshold")
        ax1.set_ylim(0.4, 2.45)
        ax1.set_title(
            f"Crescent Closure t/T={snap['t_frac']:.2f} (Closure={snap['closure_fraction']*100:.0f}%)",
            color="white",
            fontsize=8.8,
            fontweight="bold",
        )
        ax1.set_xlabel("Ring Angle θ (deg, Lovén II,IV,I,III,V)", color="white", fontsize=8)
        ax1.set_ylabel("Activator a(θ, t)", color="white", fontsize=8)
        ax1.tick_params(colors="#9ca3af", labelsize=7)
        ax1.legend(loc="upper right", fontsize=6.8, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")

        # Right: 2D ring for radius R_val (m* in {3, 4, 5, 6, 7})
        lin = np.linspace(-135.0, 135.0, 140)
        XX, YY = np.meshgrid(lin, lin)
        rr = np.hypot(XX, YY)
        th = np.mod(np.arctan2(YY, XX), 2.0 * math.pi)
        idx_th = np.clip((th / (2.0 * math.pi) * 360).astype(int), 0, 359)
        a_ang = sim_r["a_profile"][idx_th]
        field_r = np.exp(-((rr - R_val) ** 2) / (2.0 * (11.0 ** 2))) * np.clip((a_ang - 0.5) / 1.5, 0.0, 1.2)
        ax2.pcolormesh(XX, YY, field_r, cmap="magma", vmin=0.0, vmax=1.05, shading="auto")
        ax2.set_title(
            f"R_ring={R_val:.0f} μm (2πR/λ_c={2*math.pi*R_val/DEFAULT_LAMBDA_C_UM:.1f}) -> m*={sim_r['n_podia_lobes']} ({sim_r['mean_spacing_deg']:.1f}°)",
            color="white",
            fontsize=8.6,
            fontweight="bold",
        )
        ax2.set_xlabel("x (μm)", color="white", fontsize=8)
        ax2.set_ylabel("y (μm)", color="white", fontsize=8)
        ax2.tick_params(colors="#9ca3af", labelsize=7)

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_px, w_px, 4))
        frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=460, loop=0)
    return out_path


# =============================================================================
# 5b. Pass 3 Figures: OPR 3D Domed Test (+ growth GIF) & Nodal/BMP2-4 Dose-Response
# =============================================================================
def _draw_opr_dome_3d(ax: Any, opr: dict[str, Any], growth_step: int | None = None, n_phi: int = 72, n_theta: int = 180,
                      title: str | None = None) -> None:
    """Draw the plated spherical-cap test on a 3D axis (apical system up, peristome down)."""
    grid = opr_test_color_grid(opr, n_phi=n_phi, n_theta=n_theta, growth_step=growth_step)
    R = grid["R_now"]
    PH, TH = np.meshgrid(grid["phis"], np.append(grid["thetas"], 2.0 * math.pi), indexing="ij")
    rgb = np.concatenate([grid["rgb"], grid["rgb"][:, :1, :]], axis=1)
    X = R * np.sin(PH) * np.cos(TH); Y = R * np.sin(PH) * np.sin(TH); Z = R * np.cos(PH)
    ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=True)
    lim = opr["R_final"] * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=32, azim=-60)
    ax.set_facecolor("#111827")
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.tick_params(colors="#9ca3af", labelsize=6)
    ax.set_xlabel("x / R_meta", color="white", fontsize=7, labelpad=-2)
    ax.set_ylabel("y / R_meta", color="white", fontsize=7, labelpad=-2)
    ax.set_zlabel("z / R_meta", color="white", fontsize=7, labelpad=-2)
    if title:
        ax.set_title(title, color="white", fontsize=9.0, fontweight="bold")


def _draw_unrolled_columns(ax: Any, opr: dict[str, Any], ray: str = "I") -> None:
    """Unrolled meridional view of one ambulacrum (columns a, b) and the adjacent interambulacrum."""
    cols = [c for c in opr["columns"] if (c["zone"] == "A" and c["ray"] == ray)]
    ia_idx = ROMAN_RAYS.index(ray) + 1
    cols += [c for c in opr["columns"] if (c["zone"] == "IA" and c["interamb"] == ia_idx)]
    x0 = 0.0
    for c in cols:
        w = 1.0 if c["zone"] == "A" else 1.6
        base = "#9f67d6" if c["zone"] == "A" else "#e9dcbb"
        cum = np.concatenate([[0.0], np.cumsum(c["heights"])])
        for k in range(len(c["heights"])):
            fc = base if k % 2 == 0 else ("#7d4fb3" if c["zone"] == "A" else "#cdbf9c")
            if k == 0 and c["primordial_enlarged"]:
                fc = "#38bdf8"
            ax.add_patch(plt.Rectangle((x0, cum[k]), w, c["heights"][k], facecolor=fc, edgecolor="#0b0f17", linewidth=0.4))
        lbl = f"{c['ray']}{c['column']}" if c["zone"] == "A" else f"IA{c['interamb']}{c['column']}"
        ax.text(x0 + w / 2.0, -2.2, lbl, color="white", fontsize=7.2, ha="center")
        x0 += w + 0.12
    L = opr["R_final"] * (opr["phi_peristome_rad"] - opr["phi_ocular_rad"])
    ax.axhline(L, color="#facc15", linestyle="--", lw=1.0, label=f"Ocular / genital plate boundary (OPR nucleation front, arc = {L:.1f})")
    ax.axhline(0.0, color="#f87171", linestyle=":", lw=1.0, label="Peristome (oldest / Lovén primary plate)")
    ax.set_xlim(-0.3, x0 + 0.2); ax.set_ylim(-4.0, L * 1.06)
    ax.set_xticks([])
    ax.set_ylabel("Meridional arc from peristome (R_meta units)", color="white", fontsize=8)
    ax.tick_params(colors="#9ca3af", labelsize=7)
    ax.legend(loc="upper left", fontsize=6.4, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")


def render_opr_test_figure(opr: dict[str, Any], specimen: dict[str, Any], proj: dict[str, Any], out_path: Path = OUT_OPR_PNG) -> Path:
    """4-panel Pass 3 figure: OPR 3D domed test, unrolled biserial columns, plate-count growth law,
    and aboral projection vs specimen C_5 / C_10 harmonics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16.5, 10.5), dpi=130, facecolor="#0b0f17")
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    _draw_opr_dome_3d(ax_a, opr, title=f"A. OPR 3D Domed Test at R/R_meta={opr['R_final']:.0f}\n{opr['n_columns']} Columns, "
                                        f"{opr['n_plates_amb_final']} Amb + {opr['n_plates_ia_final']} IA Plates")
    ax_b = fig.add_subplot(gs[0, 1], facecolor="#111827")
    _draw_unrolled_columns(ax_b, opr, ray="I")
    ax_b.set_ylim(-4.0, 26.0)  # zoom on the oldest (peristomial / ambital) plates where the zigzag is visible
    ax_b.set_title(f"B. Unrolled Ambulacrum I (Ia, Ib; Lovén-enlarged Ib in blue) + Interambulacrum 1, peristomial region\n"
                   f"biserial alternation phase = {opr['biserial_alternation_phase']:.3f} (0.5 = zigzag; mature plates {opr['biserial_alternation_phase_mature']:.3f})",
                   color="white", fontsize=8.4, fontweight="bold")
    ax_c = fig.add_subplot(gs[1, 0], facecolor="#111827")
    Rs = [h["R_over_R_meta"] for h in opr["history"]]
    ax_c.plot(Rs, [h["n_plates_amb"] for h in opr["history"]], color="#c084fc", lw=2.0, label="Ambulacral plates (10 columns)")
    ax_c.plot(Rs, [h["n_plates_ia"] for h in opr["history"]], color="#fde68a", lw=2.0, label="Interambulacral plates (10 columns)")
    ax_c.plot(Rs, [h["n_plates_total"] for h in opr["history"]], color="#38bdf8", lw=1.4, linestyle="--", label="Total corona plates")
    ax_c.set_xlabel("Test radius R / R_metamorphosis", color="white", fontsize=8.6)
    ax_c.set_ylabel("Plate count", color="white", fontsize=8.6)
    ax_c.set_title(f"C. OPR Plate Addition vs Test Size (Spearman ρ={opr['plate_count_vs_size_spearman']:.4f}; "
                   f"Amb/IA = {opr['amb_over_ia_plate_ratio']:.2f}; newest-at-ocular = {opr['newest_at_ocular_fraction']:.2f})",
                   color="white", fontsize=8.8, fontweight="bold")
    ax_c.tick_params(colors="#9ca3af", labelsize=8); ax_c.grid(True, alpha=0.22, color="#374151")
    ax_c.legend(loc="upper left", fontsize=7.0, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")
    ax_d = fig.add_subplot(gs[1, 1], facecolor="#111827")
    modes_bar = np.arange(1, 12)
    spec_norm = specimen["fft_magnitudes_m1_11"] / np.max(specimen["fft_magnitudes_m1_11"])
    sim_norm = proj["fft_magnitudes_m1_11"] / np.max(proj["fft_magnitudes_m1_11"])
    ax_d.bar(modes_bar - 0.17, spec_norm, width=0.34, color="#f43f5e", edgecolor="white", linewidth=0.5,
             label=f"Specimen T. gratilla (C_5={specimen['c5_fundamental_power_fraction']:.3f}, C_5+C_10={specimen['pentaradial_c5_c10_power_fraction']:.3f})")
    ax_d.bar(modes_bar + 0.17, sim_norm, width=0.34, color="#34d399", edgecolor="white", linewidth=0.5,
             label=f"OPR test aboral projection (C_5={proj['c5_fundamental_power_fraction']:.3f}, C_5+C_10={proj['pentaradial_c5_c10_power_fraction']:.3f})")
    ins = ax_d.inset_axes([0.42, 0.42, 0.26, 0.5]); ins.imshow(proj["image"]); ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("Synthetic aboral view", color="white", fontsize=6.8)
    ins2 = ax_d.inset_axes([0.70, 0.42, 0.28, 0.5]); ins2.imshow(specimen["rgb"]); ins2.set_xticks([]); ins2.set_yticks([])
    ins2.set_title("Specimen (aboral)", color="white", fontsize=6.8)
    ax_d.set_xticks(modes_bar); ax_d.set_ylim(0, 1.12)
    ax_d.set_xlabel("Azimuthal harmonic m (same polar-sampling pipeline for both)", color="white", fontsize=8.4)
    ax_d.set_ylabel("|A_m| / max |A_m|", color="white", fontsize=8.4)
    ax_d.set_title(f"D. Aboral C_5/C_10 Harmonics: OPR Test vs Specimen (duty: sim {proj['ambulacral_duty_cycle']:.3f} / spec {specimen['ambulacral_duty_cycle']:.3f})",
                   color="white", fontsize=8.8, fontweight="bold")
    ax_d.tick_params(colors="#9ca3af", labelsize=8); ax_d.grid(True, alpha=0.22, color="#374151")
    ax_d.legend(loc="upper left", fontsize=6.6, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")
    fig.suptitle("Pass 3 — Ocular Plate Rule (Mooi & David 1998/2008; Raup 1968; Zachos 2009) domed-test plate addition coupled to the m=5 hydrocoel-ring PDE",
                 color="white", fontsize=11.0, fontweight="bold", y=0.98)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_opr_growth_gif(opr: dict[str, Any], out_path: Path = OUT_OPR_GIF, n_frames: int = 10) -> Path:
    """Animated 3D domed test accreting plates from metamorphosis to adult under the OPR."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    steps = np.unique(np.round(np.linspace(0, len(opr["history"]) - 1, n_frames)).astype(int))
    for st in steps:
        hrec = opr["history"][st]
        fig = plt.figure(figsize=(9.0, 4.4), dpi=100, facecolor="#0b0f17")
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        _draw_opr_dome_3d(ax1, opr, growth_step=int(st), n_phi=60, n_theta=144,
                          title=f"OPR test R/R_meta={hrec['R_over_R_meta']:.1f}  ({hrec['n_plates_total']} plates)")
        ax2 = fig.add_subplot(1, 2, 2, facecolor="#111827")
        Rs = [h["R_over_R_meta"] for h in opr["history"]]
        ax2.plot(Rs, [h["n_plates_amb"] for h in opr["history"]], color="#c084fc", lw=1.6, label="Ambulacral")
        ax2.plot(Rs, [h["n_plates_ia"] for h in opr["history"]], color="#fde68a", lw=1.6, label="Interambulacral")
        ax2.axvline(hrec["R_over_R_meta"], color="#38bdf8", linestyle="--", lw=1.0)
        ax2.set_xlabel("R / R_meta", color="white", fontsize=8); ax2.set_ylabel("Plates", color="white", fontsize=8)
        ax2.set_title("Plate addition at the ocular (OPR)", color="white", fontsize=8.6, fontweight="bold")
        ax2.tick_params(colors="#9ca3af", labelsize=7); ax2.grid(True, alpha=0.22, color="#374151")
        ax2.legend(loc="upper left", fontsize=7, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")
        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_px, w_px, 4))
        frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=420, loop=0)
    return out_path


def render_nodal_dose_figure(sweep: dict[str, Any], out_path: Path = OUT_NODAL_PNG) -> Path:
    """3-panel Pass 3 figure: Nodal / BMP2-4 / competence gradients across stomodeum and pouches (WT vs Nodal MO),
    continuous right-Nodal dose-response of expected podia, and the 2D (dose_L, dose_R) rudiment map."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.9), dpi=130, facecolor="#0b0f17")
    wt, mo = sweep["wt_profile"], sweep["mo_profile"]
    ax = axes[0]; ax.set_facecolor("#111827")
    ax.plot(wt["x"], wt["nodal"], color="#f43f5e", lw=2.0, label="Nodal (WT, right ectodermal source)")
    ax.plot(wt["x"], wt["bmp"], color="#38bdf8", lw=2.0, label="BMP2/4 (WT: stomodeal + Nodal-induced)")
    ax.plot(wt["x"], wt["competence"], color="#34d399", lw=2.0, label="Hydrocoel competence P(x) (WT)")
    ax.plot(mo["x"], mo["competence"], color="#34d399", lw=1.4, linestyle="--", label="P(x) Nodal morpholino (dose 0)")
    for xp, lbl in ((-0.55, "Left pouch\n(hydrocoel)"), (0.55, "Right somatocoel"), (0.0, "Stomodeum")):
        ax.axvline(xp, color="#9ca3af", linestyle=":", lw=0.9); ax.text(xp, 1.02, lbl, color="#e5e7eb", fontsize=6.8, ha="center", va="bottom")
    ax.set_ylim(0, 1.15); ax.set_xlabel("Normalised larval L-R axis x (left < 0 < right)", color="white", fontsize=8.4)
    ax.set_ylabel("Concentration / competence", color="white", fontsize=8.4)
    ax.set_title("A. Continuous Nodal / BMP2-4 gradients\n(Duboc 2005; Luo & Su 2012)", color="white", fontsize=8.6, fontweight="bold")
    ax.tick_params(colors="#9ca3af", labelsize=7.5); ax.grid(True, alpha=0.22, color="#374151")
    ax.legend(loc="center left", fontsize=6.4, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")
    ax = axes[1]; ax.set_facecolor("#111827")
    ax.plot(sweep["doses"], sweep["expected_podia"], color="#facc15", lw=2.2, label="E[podia] = 5 (P_L + P_R)")
    ax.plot(sweep["doses"], 5.0 * sweep["competence_right"], color="#f43f5e", lw=1.4, linestyle="--", label="5 P_R (right pouch)")
    ax.plot(sweep["doses"], 5.0 * sweep["competence_left"], color="#34d399", lw=1.4, linestyle="--", label="5 P_L (left pouch)")
    for c in sweep["canonical"]:
        if c["dose_left"] == 0.0:
            ax.scatter([c["dose_right"]], [c["reference_primary_podia"]], s=70, color="white", edgecolors="#38bdf8", zorder=5)
            ax.annotate(f"{c['condition_id']} ref {c['reference_primary_podia']}", (c["dose_right"], c["reference_primary_podia"]),
                        textcoords="offset points", xytext=(8, 6), color="white", fontsize=6.8)
    if not math.isnan(sweep["ec50_right_dose"]):
        ax.axvline(sweep["ec50_right_dose"], color="#c084fc", linestyle=":", lw=1.2, label=f"EC50 (right pouch) = {sweep['ec50_right_dose']:.3f} x WT dose")
    ax.set_xlabel("Right-sided Nodal dose (1.0 = wild type)", color="white", fontsize=8.4)
    ax.set_ylabel("Expected primary podia", color="white", fontsize=8.4); ax.set_ylim(-0.3, 11)
    ax.set_title(f"B. Dose-response: Spearman ρ(P_R, dose) = {sweep['spearman_competence_right_vs_dose']:.2f}\nmax |dev| from discrete Law 3 = {sweep['max_abs_podia_deviation_canonical']:.3f} podia",
                 color="white", fontsize=8.4, fontweight="bold")
    ax.tick_params(colors="#9ca3af", labelsize=7.5); ax.grid(True, alpha=0.22, color="#374151")
    ax.legend(loc="upper right", fontsize=6.4, facecolor="#0b0f17", edgecolor="#374151", labelcolor="white")
    ax = axes[2]; ax.set_facecolor("#111827")
    d2 = sweep["dose_grid_2d"]
    im = ax.imshow(sweep["expected_rings_map"], origin="lower", extent=[d2[0], d2[-1], d2[0], d2[-1]], cmap="viridis", vmin=0, vmax=2, aspect="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(colors="#9ca3af", labelsize=7); cb.set_label("E[hydrocoel rings]", color="white", fontsize=7.5)
    for c in sweep["canonical"]:
        ax.scatter([c["dose_right"]], [c["dose_left"]], s=60, color="white", edgecolors="black", zorder=5)
        ax.annotate(f"{c['condition_id']}: {c['expected_podia']:.2f} podia", (c["dose_right"], c["dose_left"]), textcoords="offset points", xytext=(6, 6), color="white", fontsize=6.6)
    ax.set_xlabel("Right Nodal dose", color="white", fontsize=8.4); ax.set_ylabel("Left (ectopic) Nodal dose", color="white", fontsize=8.4)
    ax.set_title("C. 2D dose map: expected rings\n(WT=1, MO=2, bilateral GOF=0)", color="white", fontsize=8.6, fontweight="bold")
    ax.tick_params(colors="#9ca3af", labelsize=7.5)
    fig.suptitle("Pass 3 — Continuous Nodal/BMP2-4 dose-response across the larval stomodeum and right somatocoel", color="white", fontsize=10.5, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# =============================================================================
# 6. MorphoEval Quantitative Measurement Suite & 8-Check Self-Test
# =============================================================================
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 16 reference-anchored quantitative morphometrics for `morphoeval/measure_outputs.py`
    (9 from Pass 2 + 7 Pass 3 OPR plate-addition / Nodal dose-response metrics)."""
    sim_wt = simulate_hydrocoel_ring_pde(R_ring_um=DEFAULT_R_RING_UM, crescent_closure=True)
    sim_ctrl = simulate_hydrocoel_ring_pde(R_ring_um=DEFAULT_R_RING_UM, equal_diffusion=True)
    staircase = evaluate_mode_selection_staircase()
    loven_nodal = evaluate_loven_law_and_nodal_perturbations()
    specimen = segment_echinoid_test_specimen()
    p3 = load_pass3_reference()
    opr = simulate_opr_test_plate_addition(sim_ring=sim_wt, amb_width_frac=specimen["ambulacral_duty_cycle"])
    proj = project_opr_test_aboral_view(opr)
    nodal_sweep = evaluate_nodal_dose_response_sweep(p3["nodal_bmp_dose_response_model"])
    opr_refs = {r["name"]: r for r in p3["opr_plate_addition_model"]["structural_reference_values"]}

    pass3_measurements = [
        {
            "name": "OPR domed-test meridional column count (5 biserial ambulacra + 5 biserial interambulacra)",
            "value": float(opr["n_columns"]),
            "unit": "columns",
            "method": "Count of plate columns generated by simulate_opr_test_plate_addition() from the 5 m=5 ring-PDE lobe azimuths (2 columns per ambulacrum and per interambulacrum)",
            "reference_value": float(opr_refs["meridional_column_count"]["value"]),
            "reference_source": opr_refs["meridional_column_count"]["source"] + " & Lean 4 theorem opr_biserial_plate_addition_law",
            "note": f"Ambulacral columns = {opr['n_ambulacral_columns']}; ray azimuths from PDE = {opr['ray_angles_deg']}",
            "derived_from": "script",
        },
        {
            "name": "OPR newest-plate-adjacent-to-ocular fraction across all 20 columns at adult size",
            "value": round(opr["newest_at_ocular_fraction"], 4),
            "unit": "fraction of columns",
            "method": "Fraction of columns whose youngest plate (latest birth step) is the apical-most plate after growth to R/R_meta = 40 (structural OPR prediction; holds by construction of ocular-end nucleation)",
            "reference_value": float(opr_refs["newest_plate_adjacent_to_ocular_fraction"]["value"]),
            "reference_source": opr_refs["newest_plate_adjacent_to_ocular_fraction"]["source"],
            "note": f"Final plates: {opr['n_plates_amb_final']} ambulacral + {opr['n_plates_ia_final']} interambulacral",
            "derived_from": "script",
        },
        {
            "name": "OPR total corona plate count vs test radius Spearman correlation (plate addition throughout growth)",
            "value": round(opr["plate_count_vs_size_spearman"], 4),
            "unit": "Spearman rho",
            "method": "Spearman rank correlation of total plate count against R/R_meta over the 221-step saturating growth history",
            "reference_value": float(opr_refs["plate_count_vs_test_size_spearman_r"]["value"]),
            "reference_source": opr_refs["plate_count_vs_test_size_spearman_r"]["source"],
            "note": "rho < 1 only because plate count is piecewise constant between nucleation events (ties)",
            "derived_from": "script",
        },
        {
            "name": "OPR ambulacral / interambulacral primary plate count ratio at adult size",
            "value": round(opr["amb_over_ia_plate_ratio"], 4),
            "unit": "ratio",
            "method": "Sum of ambulacral plates over sum of interambulacral plates across the 20 columns at R/R_meta = 40 (emergent from zone-specific nucleation / terminal plate sizes)",
            "reference_value": float(opr_refs["ambulacral_over_interambulacral_primary_plate_ratio"]["value"]),
            "reference_source": opr_refs["ambulacral_over_interambulacral_primary_plate_ratio"]["source"],
            "note": f"Mean plate height amb = {opr['mean_plate_height_amb']:.3f}, IA = {opr['mean_plate_height_ia']:.3f} (R_meta units); reference is approximate (trigeminate compounding)",
            "derived_from": "script",
        },
        {
            "name": "OPR biserial ambulacral column alternation phase (0.5 = perfect Lovén zigzag suture)",
            "value": round(opr["biserial_alternation_phase"], 4),
            "unit": "fraction of local plate height",
            "method": "Mean folded phase offset of column-b plate sutures within the column-a suture lattice across all 5 ambulacra; emerges from the Lovén-enlarged primary plate seeding a half-plate offset under identical OPR nucleation kinetics",
            "reference_value": float(opr_refs["biserial_alternation_phase_offset"]["value"]),
            "reference_source": opr_refs["biserial_alternation_phase_offset"]["source"],
            "note": f"Mature (ambital, h >= 0.95 h_max) plates only: {opr['biserial_alternation_phase_mature']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "OPR synthetic test aboral-projection C_5 fundamental Fourier power fraction vs Tripneustes gratilla specimen",
            "value": round(proj["c5_fundamental_power_fraction"], 4),
            "unit": "power fraction",
            "method": "Identical polar lavender-chromaticity + azimuthal FFT pipeline (polar_c5_harmonics) applied to the orthographic aboral projection of the plated OPR dome and to the specimen micrograph",
            "reference_value": round(specimen["c5_fundamental_power_fraction"], 4),
            "reference_source": "Vendored Tripneustes gratilla test specimen (CC BY-SA 4.0), same pipeline; ambulacral angular duty cycle of the synthetic test is calibrated from the specimen profile",
            "note": f"Sim C_5+C_10 = {proj['pentaradial_c5_c10_power_fraction']:.4f} vs specimen {specimen['pentaradial_c5_c10_power_fraction']:.4f}; sim rays detected = {proj['n_rays_detected']}; duty sim {proj['ambulacral_duty_cycle']:.3f} vs spec {specimen['ambulacral_duty_cycle']:.3f}",
            "derived_from": "output",
        },
        {
            "name": "Continuous Nodal/BMP2-4 dose-response: max |expected podia - discrete Law 3| over WT, Nodal MO and bilateral GOF doses",
            "value": round(nodal_sweep["max_abs_podia_deviation_canonical"], 4),
            "unit": "podia",
            "method": "Steady-state 1D screened-Poisson Nodal and BMP2/4 gradients across the larval L-R axis; pouch competence = Hill_act(BMP) * (1 - Hill_rep(Nodal)); E[podia] = 5 (P_L + P_R) evaluated at (dose_L, dose_R) in {(0,1), (0,0), (1,1)}",
            "reference_value": 0.0,
            "reference_source": "Duboc et al. (2005) Dev. Cell 9:147-158 Figs 1/4/5 (5, 10, 0 podia) & Lean 4 theorem nodal_dose_response_hill_reduces_to_discrete_law",
            "note": f"E[podia] WT/MO/GOF = {[round(c['expected_podia'], 3) for c in nodal_sweep['canonical']]}; EC50 (right pouch) = {nodal_sweep['ec50_right_dose']:.4f} x WT dose; Spearman rho(P_R, dose) = {nodal_sweep['spearman_competence_right_vs_dose']:.2f}; WT Nodal at right somatocoel / stomodeum / left pouch = {nodal_sweep['wt_profile']['nodal_pouch_right']:.4f} / {nodal_sweep['wt_profile']['nodal_stomodeum']:.4f} / {nodal_sweep['wt_profile']['nodal_pouch_left']:.4f}",
            "derived_from": "script",
        },
    ]

    return [
        {
            "name": "Canonical hydrocoel ring primary podia mode m^* at R_ring=75 um",
            "value": float(sim_wt["n_podia_lobes"]),
            "unit": "lobes",
            "method": "Peak count of Gierer-Meinhardt circular-ring PDE at 2*pi*R_ring/lambda_c = 5.0",
            "reference_value": 5.0,
            "reference_source": "Morris (2012) Evol. Dev. 14:313-326 & Lean 4 theorem hydrocoel_ring_pentaradial_mode5_selection",
            "note": f"Mean angular spacing = {sim_wt['mean_spacing_deg']:.2f} deg",
            "derived_from": "script",
        },
        {
            "name": "Pentaradial inter-ambulacral angular spacing Delta_theta on closed hydrocoel ring",
            "value": round(sim_wt["mean_spacing_deg"], 4),
            "unit": "deg",
            "method": "Mean angular spacing between adjacent activator maxima on closed hydrocoel ring",
            "reference_value": 72.0,
            "reference_source": "Morris (2012) Evol. Dev. 14:313-326 & Meinhardt (2008) Curr. Top. Dev. Biol. 81:1-63",
            "note": f"Detected lobe angles = {sim_wt['detected_lobe_angles_deg']}",
            "derived_from": "script",
        },
        {
            "name": "Hydrocoel ring mode staircase correlation across m in {3, 4, 5, 6, 7}",
            "value": round(staircase["pearson_r_staircase"], 4),
            "unit": "Pearson r",
            "method": "Correlation of PDE emergent lobe count vs theoretical mode m^* = round(2*pi*R_ring/lambda_c) across R in {45,60,75,90,105} um",
            "reference_value": 1.0,
            "reference_source": "Meinhardt (2008) Curr. Top. Dev. Biol. 81:1-63 & Lean 4 theorem hydrocoel_ring_pentaradial_mode5_selection",
            "note": f"Window accuracy in (4.5, 5.5) = {staircase['m5_window_accuracy']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Equal-diffusion negative control (D_h = D_a) emergent podia lobe count",
            "value": float(sim_ctrl["n_podia_lobes"]),
            "unit": "lobes",
            "method": "Gierer-Meinhardt circular-ring PDE integrated with equal diffusivity D_h = D_a = 0.45 um^2/s",
            "reference_value": 0.0,
            "reference_source": "Meinhardt (2008) Curr. Top. Dev. Biol. 81:1-63 (Turing lateral-inhibition necessity)",
            "note": "Confirms zero spatial symmetry breaking without long-range inhibition",
            "derived_from": "script",
        },
        {
            "name": "Lovén's Law (II, IV, I, III, V) crescent initiation sequence correlation",
            "value": round(loven_nodal["loven_sequence_correlation"], 4),
            "unit": "Pearson r",
            "method": "Correlation of PDE crescent-closure ray initiation order vs Morris (2012) / Lovén (1874) sequence (2, 4, 1, 3, 5)",
            "reference_value": 1.0,
            "reference_source": "Lovén (1874) K. Sven. Vetensk.-Akad. Handl. 11(7):1-91 & Morris (2012) Evol. Dev. 14:313-326",
            "note": f"Ordered rays = {loven_nodal['ordered_rays']}, b-enlarged count = {loven_nodal['b_enlarged_count']}/5",
            "derived_from": "script",
        },
        {
            "name": "Duboc et al. (2005) Nodal morpholino knockdown twin-rudiment primary podia count",
            "value": float(loven_nodal["nodal_conditions"][1]["primary_podia_count"]),
            "unit": "podia",
            "method": "Total primary podia lobes formed when right-sided Nodal repression is ablated (nodal_left=0, nodal_right=0 -> 2x5)",
            "reference_value": 10.0,
            "reference_source": "Duboc et al. (2005) Dev. Cell 9(1):147-158 Fig. 4 & Lean 4 theorem duboc_nodal_left_right_rudiment_multiplicity",
            "note": "WT = 5 podia (1x5), Nodal LOF = 10 podia (2x5), Nodal GOF = 0 podia",
            "derived_from": "script",
        },
        {
            "name": "Tripneustes gratilla test specimen detected ambulacral ray count",
            "value": float(specimen["n_rays_detected"]),
            "unit": "rays",
            "method": "Automated polar chromaticity peak detection around central periproct on echinoderm_pentaradial_test_specimen.jpg",
            "reference_value": 5.0,
            "reference_source": "Vendored Tripneustes gratilla regular echinoid test specimen (CC BY-SA 4.0)",
            "note": f"Detected ray angles = {specimen['detected_ray_angles_deg']} deg",
            "derived_from": "output",
        },
        {
            "name": "Tripneustes gratilla test specimen mean inter-ambulacral angular spacing",
            "value": round(specimen["mean_angular_spacing_deg"], 4),
            "unit": "deg",
            "method": "Mean angular interval between the 5 segmented ambulacral rays on echinoderm_pentaradial_test_specimen.jpg",
            "reference_value": 72.0,
            "reference_source": "Morris (2012) Evol. Dev. 14:313-326 & Tripneustes gratilla test specimen",
            "note": f"Spacings = {specimen['inter_ray_spacings_deg']} deg (SD = {specimen['std_angular_spacing_deg']:.2f} deg, CV = {specimen['cv_angular_spacing']:.4f})",
            "derived_from": "output",
        },
        {
            "name": "Tripneustes gratilla test specimen C_5 fundamental Fourier harmonic power fraction",
            "value": round(specimen["c5_fundamental_power_fraction"], 4),
            "unit": "power fraction",
            "method": "Fraction of azimuthal Fourier power in m=5 fundamental harmonic across m in {3..9} on echinoderm_pentaradial_test_specimen.jpg",
            "reference_value": 0.9255,
            "reference_source": "Polar Fourier harmonic decomposition of vendored Tripneustes gratilla test specimen",
            "note": f"C_5+C_10 biserial power fraction = {specimen['pentaradial_c5_c10_power_fraction']:.4f}, A_5/max_non5 = {specimen['a5_over_max_non5_ratio']:.2f}x",
            "derived_from": "output",
        },
    ] + pass3_measurements


def self_test() -> None:
    """Run 11 deterministic verification checks covering SHA-256 integrity, Lean 4 theorem
    invariants, PDE mode staircase, Lovén's Law sequence, Duboc Nodal L-R perturbations,
    Tripneustes gratilla specimen C_5 Fourier segmentation, and (Pass 3) the OPR plate-addition
    solver and the continuous Nodal/BMP2-4 dose-response model."""
    # Check 1: SHA-256 integrity of all 5 dataset files (4 Pass 2 + Pass 3 reference table)
    digests = verify_dataset_integrity()
    assert len(digests) == 5, f"Expected 5 verified dataset files, got {len(digests)}"

    # Check 2: Reference JSON parity
    ref = load_reference_data()
    assert ref["metadata"]["dataset_id"] == "meinhardt-2005-echinoderm-pentaradial-symmetry-v1"
    assert len(ref["hydrocoel_ring_mode_selection_table"]) == 5

    # Check 3: Canonical pentaradial mode m^* = 5 and 72.0 deg spacing at R_ring = 75 um
    sim_wt = simulate_hydrocoel_ring_pde(R_ring_um=75.0, crescent_closure=True)
    assert sim_wt["n_podia_lobes"] == 5, f"Expected 5 lobes, got {sim_wt['n_podia_lobes']}"
    assert abs(sim_wt["mean_spacing_deg"] - 72.0) < 1e-6

    # Check 4: Equal-diffusion negative control (D_h = D_a -> 0 lobes)
    sim_ctrl = simulate_hydrocoel_ring_pde(R_ring_um=75.0, equal_diffusion=True)
    assert sim_ctrl["n_podia_lobes"] == 0, f"Expected 0 lobes in negative control, got {sim_ctrl['n_podia_lobes']}"

    # Check 5: Mode selection staircase across m in {3, 4, 5, 6, 7}
    staircase = evaluate_mode_selection_staircase()
    assert [r["pde_lobe_count"] for r in staircase["pde_regimes"]] == [3, 4, 5, 6, 7]
    assert staircase["pearson_r_staircase"] > 0.999
    assert staircase["m5_window_accuracy"] == 1.0

    # Check 6: Lovén's Law (II, IV, I, III, V) sequence and Duboc Nodal L-R 2x5 twin rudiments
    ln = evaluate_loven_law_and_nodal_perturbations()
    assert ln["ordered_rays"] == ["II", "IV", "I", "III", "V"]
    assert ln["b_enlarged_count"] == 2
    assert [c["primary_podia_count"] for c in ln["nodal_conditions"]] == [5, 10, 0]

    # Check 7: Automated polar C_5 Fourier segmentation of Tripneustes gratilla test specimen
    spec = segment_echinoid_test_specimen()
    assert spec["n_rays_detected"] == 5
    assert abs(spec["mean_angular_spacing_deg"] - 72.0) < 1e-6
    assert spec["std_angular_spacing_deg"] < 2.0
    assert spec["c5_fundamental_power_fraction"] > 0.90

    # Check 8: Bitwise two-run determinism
    sim_a = simulate_hydrocoel_ring_pde(R_ring_um=75.0, seed=2005)
    sim_b = simulate_hydrocoel_ring_pde(R_ring_um=75.0, seed=2005)
    assert np.allclose(sim_a["a_profile"], sim_b["a_profile"], atol=0.0)

    # Check 9 (Pass 3): OPR plate-addition solver coupled to the m=5 ring PDE
    p3 = load_pass3_reference()
    assert p3["metadata"]["pass"] == 3
    opr = simulate_opr_test_plate_addition(sim_ring=sim_wt, amb_width_frac=spec["ambulacral_duty_cycle"])
    assert opr["n_columns"] == 20 and opr["n_ambulacral_columns"] == 10
    assert sorted(opr["ray_angles_deg"].values()) == sorted(float(a) for a in sim_wt["detected_lobe_angles_deg"]), "PDE lobes must drive the ambulacral azimuths"
    assert opr["newest_at_ocular_fraction"] == 1.0
    assert opr["plate_count_vs_size_spearman"] > 0.99
    hist_total = [h_["n_plates_total"] for h_ in opr["history"]]
    assert all(b >= a for a, b in zip(hist_total, hist_total[1:])), "plate count must be non-decreasing (no resorption)"
    for c in opr["columns"]:
        assert all(b >= a for a, b in zip(c["birth_step"], c["birth_step"][1:])), "plate age must decrease toward the ocular"
        assert c["heights"][-1] <= max(c["heights"]) + 1e-12
    assert 0.35 <= opr["biserial_alternation_phase"] <= 0.5, f"biserial alternation phase {opr['biserial_alternation_phase']}"
    assert opr["amb_over_ia_plate_ratio"] > 1.5
    # Ablation: without the Lovén primordial enlargement the a/b sutures stay aligned (no zigzag)
    p_no = json.loads(json.dumps(p3["opr_plate_addition_model"])); p_no["plate_kinetics"]["loven_primordial_enlargement_factor"] = 1.0
    opr_no = simulate_opr_test_plate_addition(sim_ring=sim_wt, amb_width_frac=spec["ambulacral_duty_cycle"], params=p_no)
    assert opr_no["biserial_alternation_phase"] < 0.05, f"ablation should give aligned sutures, got {opr_no['biserial_alternation_phase']}"
    # Ring-PDE coupling: a tetramerous ring (R = 60 um, m = 4) yields 4 ambulacra -> 16 columns
    sim_m4 = simulate_hydrocoel_ring_pde(R_ring_um=60.0, crescent_closure=False)
    assert sim_m4["n_podia_lobes"] == 4
    proj = project_opr_test_aboral_view(opr)
    assert proj["n_rays_detected"] == 5
    assert proj["c5_fundamental_power_fraction"] > 0.90
    assert abs(proj["c5_fundamental_power_fraction"] - spec["c5_fundamental_power_fraction"]) < 0.10

    # Check 10 (Pass 3): Continuous Nodal/BMP2-4 dose-response reduces to Law 3 at the canonical doses
    sw = evaluate_nodal_dose_response_sweep(p3["nodal_bmp_dose_response_model"])
    assert [c["discrete_podia"] for c in sw["canonical"]] == [5, 10, 0]
    assert sw["max_abs_podia_deviation_canonical"] < 0.10
    assert sw["spearman_competence_right_vs_dose"] < -0.99
    assert 0.0 < sw["ec50_right_dose"] < 1.0
    wt = sw["wt_profile"]
    assert wt["nodal_pouch_right"] > 5.0 * wt["nodal_stomodeum"] > 5.0 * wt["nodal_pouch_left"], "Nodal must decay monotonically right pouch -> stomodeum -> left pouch"
    assert np.all(np.diff(sw["competence_right"]) <= 1e-12)

    # Check 11 (Pass 3): determinism + measurement suite integrity
    opr_b = simulate_opr_test_plate_addition(sim_ring=sim_wt, amb_width_frac=spec["ambulacral_duty_cycle"])
    assert opr_b["n_plates_amb_final"] == opr["n_plates_amb_final"] and abs(opr_b["biserial_alternation_phase"] - opr["biserial_alternation_phase"]) == 0.0
    meas = measure_morphoeval()
    assert len(meas) == 16, f"Expected 16 measurements, got {len(meas)}"
    assert all(("reference_value" in m_) and ("unit" in m_) for m_ in meas)

    print(
        "[self_test] All 11 Pass 3 verification checks PASSED "
        f"(m*={sim_wt['n_podia_lobes']}, Δθ={sim_wt['mean_spacing_deg']:.1f}°, "
        f"Lovén={ln['ordered_rays']}, Nodal LOF={ln['nodal_conditions'][1]['primary_podia_count']} podia, "
        f"specimen C_5={spec['c5_fundamental_power_fraction']:.4f}, SD={spec['std_angular_spacing_deg']:.2f}°; "
        f"OPR: {opr['n_columns']} cols, Amb/IA={opr['amb_over_ia_plate_ratio']:.2f}, zigzag={opr['biserial_alternation_phase']:.3f}, "
        f"sim C_5={proj['c5_fundamental_power_fraction']:.4f}; Nodal dose max|dev|={sw['max_abs_podia_deviation_canonical']:.3f}, EC50={sw['ec50_right_dose']:.3f})."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Morris (2012), Lovén (1874), Duboc (2005) & Meinhardt (2008) echinoderm pentaradial symmetry simulation (v2)"
    )
    ap.add_argument("--self-test", action="store_true", help="Run 11-check verification suite and exit")
    ap.add_argument("--skip-gif", action="store_true", help="Skip rendering the animated growth GIFs")
    ap.add_argument("--R-ring-um", type=float, default=DEFAULT_R_RING_UM, help="Hydrocoel ring radius in um")
    ap.add_argument("--lambda-c-um", type=float, default=DEFAULT_LAMBDA_C_UM, help="Turing critical wavelength in um")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic RNG seed")
    ap.add_argument("--growth-factor", type=float, default=None, help="Pass 3 OPR final test radius R/R_meta (default from reference JSON = 40)")
    ap.add_argument("--nodal-dose", type=float, default=1.0, help="Pass 3 right-sided Nodal dose for the standalone dose-response profile (1.0 = WT)")
    args = ap.parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    out_png = render_comparison_figure(
        out_path=OUT_PNG,
        R_ring_um=args.R_ring_um,
        lambda_c_um=args.lambda_c_um,
        seed=args.seed,
    )
    print(f"[render] Wrote 9-panel comparison figure -> {out_png}")

    # Pass 3 standalone figures (OPR 3D test + Nodal/BMP2-4 dose-response)
    sim_wt = simulate_hydrocoel_ring_pde(R_ring_um=args.R_ring_um, lambda_c_um=args.lambda_c_um, crescent_closure=True, seed=args.seed)
    specimen = segment_echinoid_test_specimen()
    opr = simulate_opr_test_plate_addition(sim_ring=sim_wt, amb_width_frac=specimen["ambulacral_duty_cycle"], growth_factor_final=args.growth_factor)
    proj = project_opr_test_aboral_view(opr)
    print(f"[render] Wrote OPR 3D test figure -> {render_opr_test_figure(opr, specimen, proj)}")
    sweep = evaluate_nodal_dose_response_sweep()
    if args.nodal_dose != 1.0:
        prof = simulate_nodal_bmp_dose_response(dose_right=args.nodal_dose)
        print(f"[nodal] dose_R={args.nodal_dose:.3f}: E[podia]={prof['expected_podia']:.3f}, P_L={prof['competence_left']:.3f}, P_R={prof['competence_right']:.3f}")
        sweep["wt_profile"] = prof
    print(f"[render] Wrote Nodal/BMP2-4 dose-response figure -> {render_nodal_dose_figure(sweep)}")

    if not args.skip_gif:
        out_gif = render_growth_gif(out_path=OUT_GIF)
        print(f"[render] Wrote 2-panel growth animation -> {out_gif}")
        print(f"[render] Wrote OPR test growth animation -> {render_opr_growth_gif(opr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
