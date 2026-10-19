#!/usr/bin/env python3
"""Bressloff et al. (2001/2002) V1 Euclidean symmetry-breaking & retino-cortical form constants.

Implements the orientation-hypercolumn model of geometric visual hallucinations in primary
visual cortex (V1), its log-polar retino-cortical map into Klüver's four form-constant
families (tunnels/funnels, spirals, cobwebs, honeycombs), cubic amplitude-equation branch
selection, and Rule, Stoffregen & Ermentrout (2011) flicker-phosphene Floquet resonance
tongues, bridged against the SHA-256-pinned reference reports from
`MesmerPrism/bressloff-v1-form-constants`.

Primary Sources
---------------
1. Bressloff PC, Cowan JD, Golubitsky M, Thomas PJ, Wiener MC (2001).
   Geometric visual hallucinations, Euclidean symmetry and the functional architecture
   of striate cortex. Phil. Trans. R. Soc. Lond. B 356(1407):299-330.
   https://doi.org/10.1098/rstb.2000.0769 (PMID: 11316482)
2. Bressloff PC, Cowan JD, Golubitsky M, Thomas PJ, Wiener MC (2002).
   What geometric visual hallucinations tell us about the visual cortex.
   Neural Computation 14(3):473-491.
   https://doi.org/10.1162/089976602317250861 (PMID: 11860679)
3. Ermentrout GB, Cowan JD (1979).
   A mathematical theory of visual hallucination patterns.
   Biological Cybernetics 34(3):137-150.
   https://doi.org/10.1007/BF00336965 (PMID: 486593)
4. Rule M, Stoffregen M, Ermentrout B (2011).
   A model for the origin and properties of flicker-induced geometric phosphenes.
   PLoS Computational Biology 7(9):e1002158.
   https://doi.org/10.1371/journal.pcbi.1002158 (PMID: 21980269)
5. Horton JC, Hoyt WF (1991).
   The representation of the visual field in human striate cortex: a revision of the
   classic Holmes map. Archives of Ophthalmology 109(6):816-824.
   https://doi.org/10.1001/archopht.1991.01080060080030 (PMID: 2043069)
   Empirical anchor: linear cortical magnification M(E) = 17.3 / (E + 0.75) mm/deg.
6. Upstream reference implementation & calibration fixtures:
   MesmerPrism/bressloff-v1-form-constants (commit 344863a1bd9e8a8e8c9cb20aced2897f3ffb2c4c, MIT License;
   derived from karacsm/V1-sim by Márton Karácsony, extended by Till Holzapfel).

Scope & Claim Boundary
----------------------
* The perturbative marginal-stability dispersion curves lambda_even(q), lambda_odd(q),
  Fourier-Bessel lateral coupling harmonics, orientation eigenfunctions u_even(phi),
  u_odd(phi), cubic amplitude-equation quartic coefficients gamma(theta), and the
  contoured/non-contoured planform generators match the analytical formulation in
  `MesmerPrism/bressloff-v1-form-constants/rust-v1-sim/src/models/bressloff/planform.rs`
  and reproduce all 5 `stability_reports` in `datasets/bressloff/paper-calibration.json`
  to machine precision (~1e-15).
* The retino-cortical map is the exact two-regime form of Bressloff et al. (2001,
  Section 2; 2002, Eq. 2.1): x = (alpha/eps) ln(1 + eps r/w0), y = beta r theta/(w0 + eps r),
  inverted analytically for the retinal pullback (no foveal hole; only r = 0 is undefined).
  Its linear magnification dx/dr = alpha/(w0 + eps r) is compared against the human
  Horton & Hoyt (1991) law M(E) = 17.3/(E + 0.75) mm/deg over 1-40 deg eccentricity.
* Rendered planforms use the model's critical wavenumber q* (quantised to integer cycles
  on the periodic cortical torus) and are labelled with both the rendered lattice and the
  branch the cubic amplitude equations select globally; parity with the upstream fixture is
  reported separately from the fixture's own paper-expectation checks (fig40's odd-branch
  hexagon/triangle choice is degenerate at cubic order and is flagged, not hidden).
* The Rule et al. (2011) Fig. 8 overlay is drawn from an upstream fixture whose
  `calibration_claim_allowed` flag is false; the panel states this and the overlap RMS.
* The Gaussian-only negative control removes Mexican-hat lateral inhibition
  (lateral_inhibition = 0), verifying that the Turing/Euclidean band-pass wavenumber
  peak q* collapses to q -> 0 (homogeneous mode).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

if any(flag in sys.argv for flag in ("--no-show", "--self-test")) or not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
BRESSLOFF_DATASET_DIR = REPO_ROOT / "datasets" / "bressloff"

FIXTURE_CHECKSUMS: Dict[str, str] = {
    "paper-calibration.json": "af12b2c1aa3e38faf984166b3aa227a2aa064480a98279f14c004342ce2e7dd4",
    "rule-2011-floquet.json": "5e349c7b80185de463e1ea8656a2fe91a83523c212875af4ce46d09e9dde2a8e",
    "driven-neural-fields-registry.json": "4f07b5175a48bcb50bad4998953505a3f92faf4c6f5514d2b4643c2a67c36b10",
    "horton_hoyt_1991_magnification.json": "90d639ebb4601be59308699e128ac9b59574a9da6dbc47f0c7ad00be8d9e4a8f",
}

# Retino-cortical log-polar mapping constants (Bressloff et al. 2001 Section 2; 2002 Eq. 2.1)
RETINO_EPS: float = 0.051
RETINO_W0: float = 0.087
RETINO_ALPHA: float = 3.0 / math.pi
RETINO_BETA: float = 1.589 / 2.0


# =============================================================================
# 0. Exact retino-cortical map (Bressloff et al. 2001 §2; 2002 Eq. 2.1) and the
#    Horton & Hoyt (1991) cortical-magnification anchor
# =============================================================================
def retino_cortical_forward(r_deg: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Retina (r in deg, theta in rad) -> cortex (x_c, y_c in mm): x = (α/ε) ln(1 + ε r/w₀), y = β r θ/(w₀ + ε r)."""
    r = np.asarray(r_deg, dtype=float)
    x_c = (RETINO_ALPHA / RETINO_EPS) * np.log1p(RETINO_EPS * r / RETINO_W0)
    y_c = RETINO_BETA * r * np.asarray(theta, dtype=float) / (RETINO_W0 + RETINO_EPS * r)
    return x_c, y_c


def retino_cortical_inverse(x_c: np.ndarray, y_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cortex (x_c, y_c in mm) -> retina (r in deg, theta in rad); exact inverse of `retino_cortical_forward`."""
    x = np.asarray(x_c, dtype=float)
    r = (RETINO_W0 / RETINO_EPS) * np.expm1(RETINO_EPS * x / RETINO_ALPHA)
    r_safe = np.maximum(r, 1.0e-12)
    theta = np.asarray(y_c, dtype=float) * (RETINO_W0 + RETINO_EPS * r_safe) / (RETINO_BETA * r_safe)
    return r, theta


def cortical_magnification_model(ecc_deg: np.ndarray) -> np.ndarray:
    """Linear magnification of the Bressloff map, dx/dr = α/(w₀ + ε r) in mm/deg."""
    return RETINO_ALPHA / (RETINO_W0 + RETINO_EPS * np.asarray(ecc_deg, dtype=float))


def cortical_magnification_horton_hoyt(ecc_deg: np.ndarray, a_mm_per_deg: float, e2_deg: float) -> np.ndarray:
    """Horton & Hoyt (1991) human V1 linear magnification M(E) = A/(E + E₂) mm/deg (A = 17.3, E₂ = 0.75)."""
    return a_mm_per_deg / (np.asarray(ecc_deg, dtype=float) + e2_deg)


def compare_cortical_magnification() -> Dict[str, Any]:
    """Compare the map's magnification with Horton & Hoyt (1991) over 1-40 deg and return the residuals."""
    with open(BRESSLOFF_DATASET_DIR / "horton_hoyt_1991_magnification.json", "r", encoding="utf-8") as fh:
        hh = json.load(fh)
    a_hh = float(hh["A_mm_per_deg"])
    e2_hh = float(hh["E2_deg"])
    lo, hi = (float(v) for v in hh.get("comparison_range_deg", [1.0, 40.0]))
    ecc = np.linspace(lo, hi, 157)
    m_model = cortical_magnification_model(ecc)
    m_hh = cortical_magnification_horton_hoyt(ecc, a_hh, e2_hh)
    rel = (m_model - m_hh) / m_hh
    # The model is algebraically of Horton-Hoyt form with A = α/ε and E₂ = w₀/ε
    a_model = RETINO_ALPHA / RETINO_EPS
    e2_model = RETINO_W0 / RETINO_EPS
    # Total cortical extent fovea -> 90 deg along the horizontal meridian (mm)
    extent_model = float(retino_cortical_forward(np.array([90.0]), np.array([0.0]))[0][0])
    extent_hh = a_hh * math.log((90.0 + e2_hh) / e2_hh)
    return {
        "eccentricity_deg": ecc,
        "model_mm_per_deg": m_model,
        "horton_hoyt_mm_per_deg": m_hh,
        "rel_error": rel,
        "rms_rel_error": float(np.sqrt(np.mean(rel * rel))),
        "max_abs_rel_error": float(np.max(np.abs(rel))),
        "model_A_mm_per_deg": a_model,
        "model_E2_deg": e2_model,
        "hh_A_mm_per_deg": a_hh,
        "hh_E2_deg": e2_hh,
        "model_extent_0_90_mm": extent_model,
        "hh_extent_0_90_mm": extent_hh,
        "source": hh.get("source", "Horton & Hoyt (1991) Arch. Ophthalmol. 109:816"),
    }


# =============================================================================
# 1. Fixture Checksum & Provenance Verification
# =============================================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_fixture_checksums() -> Dict[str, Tuple[bool, str]]:
    results: Dict[str, Tuple[bool, str]] = {}
    for fname, expected in FIXTURE_CHECKSUMS.items():
        fpath = BRESSLOFF_DATASET_DIR / fname
        if not fpath.is_file():
            results[fname] = (False, "MISSING")
            continue
        actual = sha256_file(fpath)
        results[fname] = (actual == expected, actual)
    return results


# =============================================================================
# 2. Bressloff (2001/2002) Orientation Hypercolumn & Planform Formalism
# =============================================================================
@dataclass
class BressloffParams:
    """Parameters for the Bressloff et al. (2001/2002) V1 orientation hypercolumn model."""

    preset_id: str = "manual"
    pattern: str = "cobweb"          # auto | rings | rays | spiral | cobweb | honeycomb | rhombic | hex_pi | triangle
    contour_mode: str = "contoured"  # contoured | noncontoured
    parity: str = "even"             # even | odd
    n: int = 96
    m: int = 24
    wave_count: float = 12.0
    drift: float = 0.35
    pattern_angle_deg: float = 45.0
    sharpness: float = 2.1
    eigen_beta: float = 0.35
    hypercolumn_mm: float = 2.0
    local_sigma_deg: float = 20.0
    local_wide_sigma_deg: float = 60.0
    local_inhibition: float = 1.0
    lateral_sigma: float = 1.0
    lateral_wide_sigma: float = 1.5
    lateral_inhibition: float = 1.0
    lateral_spread_deg: float = 0.0
    stability_q_min: float = 0.05
    stability_q_max: float = 3.5
    stability_samples: int = 80


PAPER_PRESETS: Dict[str, Dict[str, Any]] = {
    "fig16_odd": {
        "pattern": "auto",
        "contour_mode": "contoured",
        "parity": "even",
        "eigen_beta": 0.4,
        "lateral_sigma": 1.0,
        "lateral_wide_sigma": 3.0,
        "lateral_inhibition": 1.0,
        "lateral_spread_deg": 0.0,
        "stability_q_min": 0.05,
        "stability_q_max": 3.5,
        "stability_samples": 128,
        "wave_count": 12.0,
        "sharpness": 1.8,
    },
    "fig17_even": {
        "pattern": "auto",
        "contour_mode": "contoured",
        "parity": "even",
        "eigen_beta": 0.4,
        "lateral_sigma": 1.0,
        "lateral_wide_sigma": 3.0,
        "lateral_inhibition": 1.0,
        "lateral_spread_deg": 60.0,
        "stability_q_min": 0.05,
        "stability_q_max": 3.5,
        "stability_samples": 128,
        "wave_count": 12.0,
        "sharpness": 1.8,
    },
    "fig29_square_noncontoured": {
        "pattern": "cobweb",
        "contour_mode": "noncontoured",
        "parity": "even",
        "lateral_spread_deg": 0.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 90.0,
        "sharpness": 2.1,
    },
    "fig29_roll_noncontoured": {
        "pattern": "rings",
        "contour_mode": "noncontoured",
        "parity": "even",
        "lateral_spread_deg": 0.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 0.0,
        "sharpness": 2.1,
    },
    "fig30_rhombic_noncontoured": {
        "pattern": "rhombic",
        "contour_mode": "noncontoured",
        "parity": "even",
        "lateral_spread_deg": 0.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 45.0,
        "sharpness": 2.1,
    },
    "fig30_hex_noncontoured": {
        "pattern": "honeycomb",
        "contour_mode": "noncontoured",
        "parity": "even",
        "lateral_spread_deg": 0.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 60.0,
        "sharpness": 2.1,
    },
    "fig31_square_even": {
        "pattern": "cobweb",
        "contour_mode": "contoured",
        "parity": "even",
        "lateral_spread_deg": 60.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 90.0,
        "sharpness": 2.1,
    },
    "fig32_square_odd": {
        "pattern": "cobweb",
        "contour_mode": "contoured",
        "parity": "odd",
        "lateral_spread_deg": 0.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 90.0,
        "sharpness": 2.1,
    },
    "fig33_rhombic_even": {
        "pattern": "rhombic",
        "contour_mode": "contoured",
        "parity": "even",
        "lateral_spread_deg": 60.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 45.0,
        "sharpness": 2.1,
    },
    "fig33_rhombic_even_roll": {
        "pattern": "spiral",
        "contour_mode": "contoured",
        "parity": "even",
        "lateral_spread_deg": 60.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 45.0,
        "sharpness": 2.1,
    },
    "fig34_rhombic_odd": {
        "pattern": "rhombic",
        "contour_mode": "contoured",
        "parity": "odd",
        "lateral_spread_deg": 0.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 45.0,
        "sharpness": 2.1,
    },
    "fig35_hex_even": {
        "pattern": "hex_pi",
        "contour_mode": "contoured",
        "parity": "even",
        "lateral_spread_deg": 60.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 60.0,
        "sharpness": 2.1,
    },
    "fig35_hex_zero_even": {
        "pattern": "honeycomb",
        "contour_mode": "contoured",
        "parity": "even",
        "lateral_spread_deg": 60.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 60.0,
        "sharpness": 2.1,
    },
    "fig36_triangle_odd": {
        "pattern": "triangle",
        "contour_mode": "contoured",
        "parity": "odd",
        "lateral_spread_deg": 0.0,
        "wave_count": 12.0,
        "pattern_angle_deg": 60.0,
        "sharpness": 2.1,
    },
    "fig7_lattice_tunnel": {
        "pattern": "rings",
        "contour_mode": "contoured",
        "parity": "even",
        "lateral_spread_deg": 60.0,
        "wave_count": 14.0,
        "pattern_angle_deg": 0.0,
        "sharpness": 2.3,
    },
}


def make_preset_params(preset_id: str, n: int = 96, m: int = 24) -> BressloffParams:
    if preset_id not in PAPER_PRESETS:
        raise ValueError(f"Unknown paper preset: {preset_id}")
    kw = dict(PAPER_PRESETS[preset_id])
    return BressloffParams(preset_id=preset_id, n=n, m=m, **kw)


def cell_mm_for(params: BressloffParams) -> float:
    return (2.0 * math.pi * RETINO_BETA / RETINO_EPS) / float(params.n)


def safe_denominator(value: float) -> float:
    if abs(value) < 1.0e-6:
        return -1.0e-6 if value < 0.0 else 1.0e-6
    return value


def local_weight_coeff(params: BressloffParams, n_harm: int) -> float:
    """Fourier harmonic W_n of local intra-hypercolumn orientation coupling (Bressloff 2001 Eq. 3.12)."""
    xi = math.radians(params.local_sigma_deg)
    xi_hat = math.radians(params.local_wide_sigma_deg)
    n2 = float(n_harm * n_harm)
    return math.exp(-2.0 * n2 * xi * xi) - params.local_inhibition * math.exp(-2.0 * n2 * xi_hat * xi_hat)


def lateral_spread_factor(params: BressloffParams, n_harm: int) -> float:
    theta0 = math.radians(params.lateral_spread_deg)
    if n_harm == 0 or abs(theta0) < 1.0e-9:
        return 1.0
    x = 2.0 * float(n_harm) * theta0
    return math.sin(x) / x


def modified_bessel_i(n_harm: int, x: float) -> float:
    """Power-series modified Bessel function I_n(x) matching rust-v1-sim."""
    if abs(x) < 1.0e-12:
        return 1.0 if n_harm == 0 else 0.0
    half_x = 0.5 * x
    factorial = 1.0
    for v in range(1, n_harm + 1):
        factorial *= float(v)
    term = (half_x ** n_harm) / factorial
    total = term
    for k in range(1, 80):
        term *= (half_x * half_x) / (float(k) * float(k + n_harm))
        total += term
        if abs(term) < max(abs(total), 1.0) * 1.0e-13:
            break
    return total


def lateral_weight_coeff(params: BressloffParams, n_harm: int, q: float) -> float:
    """Fourier-Bessel harmonic W_n^lat(q) of anisotropic lateral coupling (Bressloff 2001 Eq. 4.9)."""
    xi = params.lateral_sigma
    xi_hat = params.lateral_wide_sigma
    narrow = 0.25 * xi * xi * q * q
    broad = 0.25 * xi_hat * xi_hat * q * q
    sign = 1.0 if (n_harm % 2 == 0) else -1.0
    return (
        lateral_spread_factor(params, n_harm)
        * 0.5
        * sign
        * (
            math.exp(-narrow) * modified_bessel_i(n_harm, narrow)
            - params.lateral_inhibition * math.exp(-broad) * modified_bessel_i(n_harm, broad)
        )
    )


def signed_lateral_pair(
    params: BressloffParams, parity: str, left_harmonic: int, right_harmonic: int, q: float
) -> float:
    left = lateral_weight_coeff(params, left_harmonic, q)
    right = lateral_weight_coeff(params, right_harmonic, q)
    return (left + right) if parity == "even" else (left - right)


def branch_coupling_sum(params: BressloffParams, parity: str, q: float, harmonics: int = 10) -> float:
    w1 = local_weight_coeff(params, 1)
    total = 0.0
    for m in range(harmonics + 1):
        if m == 1:
            continue
        left = lateral_weight_coeff(params, 1, q) if m == 0 else lateral_weight_coeff(params, m - 1, q)
        right = lateral_weight_coeff(params, m + 1, q)
        num = (left + right) if parity == "even" else (left - right)
        total += (num * num) / safe_denominator(w1 - local_weight_coeff(params, m))
    return total


def branch_growth(params: BressloffParams, parity: str, q: float) -> float:
    """Perturbative dispersion relation lambda_even/odd(q) (Bressloff 2001 Section 4, Figs. 16-17)."""
    beta = params.eigen_beta
    return (
        local_weight_coeff(params, 1)
        + beta * signed_lateral_pair(params, parity, 0, 2, q)
        + (beta * beta) * branch_coupling_sum(params, parity, q, 10)
    )


@dataclass
class OrientationEigenDetails:
    parity: str
    beta: float
    cos_coefficients: List[Tuple[float, float]] = field(default_factory=list)
    sin_coefficients: List[Tuple[float, float]] = field(default_factory=list)


def orientation_eigen_details(params: BressloffParams, q: float) -> OrientationEigenDetails:
    max_harmonic = 4
    cos_coeffs: List[Tuple[float, float]] = []
    sin_coeffs: List[Tuple[float, float]] = []
    if params.parity == "even":
        cos_coeffs.append((1.0, 1.0))
        u0 = lateral_weight_coeff(params, 1, q) / safe_denominator(
            local_weight_coeff(params, 1) - local_weight_coeff(params, 0)
        )
        cos_coeffs.append((0.0, float(np.clip(params.eigen_beta * u0, -1.5, 1.5))))
        for m in range(2, max_harmonic + 1):
            coeff = (
                lateral_weight_coeff(params, m - 1, q) + lateral_weight_coeff(params, m + 1, q)
            ) / safe_denominator(local_weight_coeff(params, 1) - local_weight_coeff(params, m))
            cos_coeffs.append((float(m), float(np.clip(params.eigen_beta * coeff, -1.5, 1.5))))
    else:
        sin_coeffs.append((1.0, 1.0))
        for m in range(2, max_harmonic + 1):
            coeff = (
                lateral_weight_coeff(params, m - 1, q) - lateral_weight_coeff(params, m + 1, q)
            ) / safe_denominator(local_weight_coeff(params, 1) - local_weight_coeff(params, m))
            sin_coeffs.append((float(m), float(np.clip(params.eigen_beta * coeff, -1.5, 1.5))))
    return OrientationEigenDetails(
        parity=params.parity,
        beta=params.eigen_beta,
        cos_coefficients=cos_coeffs,
        sin_coefficients=sin_coeffs,
    )


def orientation_eigen_value(delta: float, eigen: OrientationEigenDetails) -> float:
    cos_part = sum(c * math.cos(2.0 * h * delta) for h, c in eigen.cos_coefficients)
    sin_part = sum(c * math.sin(2.0 * h * delta) for h, c in eigen.sin_coefficients)
    return cos_part + sin_part


def amplitude_gamma2(eigen: OrientationEigenDetails) -> float:
    """Quadratic hexagonal amplitude coefficient eta_hex = gamma_2 (Bressloff 2001 Eq. 5.14)."""
    samples = 720
    total = 0.0
    for i in range(samples):
        phi = math.pi * (float(i) + 0.5) / float(samples)
        total += (
            orientation_eigen_value(phi, eigen)
            * orientation_eigen_value(phi - 2.0 * math.pi / 3.0, eigen)
            * orientation_eigen_value(phi + 2.0 * math.pi / 3.0, eigen)
        )
    return total / float(samples)


def amplitude_gamma3(theta: float, eigen: OrientationEigenDetails) -> float:
    """Cubic cross-mode coupling coefficient gamma_3(theta) (Bressloff 2001 Eq. 5.15, Figs. 37-40)."""
    samples = 720
    total = 0.0
    for i in range(samples):
        phi = math.pi * (float(i) + 0.5) / float(samples)
        shifted = orientation_eigen_value(phi - theta, eigen)
        base = orientation_eigen_value(phi, eigen)
        total += (shifted * shifted) * (base * base)
    return total / float(samples)


def _branch_target_lattice(pattern: str) -> str:
    if pattern == "auto":
        return "global"
    if pattern == "cobweb":
        return "square"
    if pattern == "rhombic":
        return "rhombic"
    if pattern in ("honeycomb", "hex_pi", "triangle"):
        return "hexagonal"
    return "roll"


def _make_branch_candidate(
    family: str,
    pattern: str,
    mode_count: int,
    theta_rad: float,
    gamma0: float,
    gamma_cross: float,
    lam: float,
    denominator: float,
    eta: float,
    stable: bool,
) -> Dict[str, Any]:
    denom = max(denominator, 1.0e-9)
    lam_pos = max(lam, 0.0)
    if lam_pos <= 0.0:
        amp = 0.0
    elif abs(eta) > 1.0e-9:
        amp = max(0.0, (abs(eta) + math.sqrt(eta * eta + 4.0 * denom * lam_pos)) / (2.0 * denom))
    else:
        amp = math.sqrt(lam_pos / denom)
    if lam_pos <= 0.0:
        score = float("-inf")
    elif mode_count == 1:
        score = (lam_pos * lam_pos) / (4.0 * max(gamma0, 1.0e-9))
    else:
        score = float(mode_count) * (
            0.5 * lam_pos * amp * amp + abs(eta) * (amp ** 3) / 3.0 - 0.25 * denom * (amp ** 4)
        )
    return {
        "family": family,
        "pattern": pattern,
        "mode_count": mode_count,
        "theta_rad": theta_rad,
        "gamma_cross": gamma_cross,
        "eta": eta,
        "amplitude": amp,
        "score": score,
        "stable": bool(stable and lam_pos > 0.0 and math.isfinite(amp)),
    }


def branch_selection_for(params: BressloffParams, q: float, growth: float) -> Dict[str, Any]:
    lam = max(growth, 0.0)
    eigen = orientation_eigen_details(params, q)
    square_theta = math.pi / 2.0
    rhombic_theta = float(np.clip(math.radians(params.pattern_angle_deg), math.pi / 12.0, 5.0 * math.pi / 12.0))
    hex_theta = 2.0 * math.pi / 3.0

    gamma0 = amplitude_gamma3(0.0, eigen)
    gamma_square = amplitude_gamma3(square_theta, eigen)
    gamma_rhombic = amplitude_gamma3(rhombic_theta, eigen)
    gamma_hex = amplitude_gamma3(hex_theta, eigen)
    eta_hex = amplitude_gamma2(eigen) if params.parity == "even" else 0.0

    roll_stable = (
        gamma0 > 0.0
        and 2.0 * gamma_square > gamma0
        and 2.0 * gamma_rhombic > gamma0
        and 2.0 * gamma_hex > gamma0
    )
    roll = _make_branch_candidate("roll", "spiral", 1, 0.0, gamma0, 0.0, lam, gamma0, 0.0, roll_stable)
    square = _make_branch_candidate(
        "square", "cobweb", 2, square_theta, gamma0, gamma_square, lam,
        gamma0 + 2.0 * gamma_square, 0.0, gamma_square > 0.0 and 2.0 * gamma_square < gamma0,
    )
    rhombic = _make_branch_candidate(
        "rhombic", "rhombic", 2, rhombic_theta, gamma0, gamma_rhombic, lam,
        gamma0 + 2.0 * gamma_rhombic, 0.0, gamma_rhombic > 0.0 and 2.0 * gamma_rhombic < gamma0,
    )
    hex_pattern = "hex_pi" if eta_hex < 0.0 else "honeycomb"
    # At cubic order the odd branch has eta_hex = 0, so H0 (honeycomb) and H_pi (triangle) are
    # degenerate; the fixture's "honeycomb" label is kept for parity but flagged as undetermined.
    hex_pattern_degenerate = params.parity != "even" or abs(eta_hex) < 1.0e-12
    hex_stable = (
        gamma_hex > 0.0
        and (params.parity == "even" or 2.0 * gamma_hex < gamma0)
        and (gamma0 + 4.0 * gamma_hex > 0.0)
    )
    hex_cand = _make_branch_candidate(
        "hexagonal", hex_pattern, 3, hex_theta, gamma0, gamma_hex, lam,
        gamma0 + 4.0 * gamma_hex, eta_hex, hex_stable,
    )

    candidates = [roll, square, rhombic, hex_cand]
    candidates.sort(key=lambda c: (c["stable"], c["score"]), reverse=True)
    global_selected = candidates[0]
    target_lattice = _branch_target_lattice(params.pattern)

    if target_lattice == "global":
        selected = global_selected
        selected_scope = "global"
        selected_lattice_stable = global_selected["stable"]
    else:
        def _in_lattice(c: Dict[str, Any]) -> bool:
            if target_lattice == "roll":
                return c["family"] == "roll"
            return c["family"] in (target_lattice, "roll")

        def _loc_stable(c: Dict[str, Any]) -> bool:
            if c["family"] == "roll":
                if target_lattice == "square":
                    return gamma0 > 0.0 and 2.0 * gamma_square > gamma0
                if target_lattice == "rhombic":
                    return gamma0 > 0.0 and 2.0 * gamma_rhombic > gamma0
                if target_lattice == "hexagonal":
                    return gamma0 > 0.0 and 2.0 * gamma_hex > gamma0
                return gamma0 > 0.0
            if c["family"] == "square":
                return gamma_square > 0.0 and 2.0 * gamma_square < gamma0
            if c["family"] == "rhombic":
                return gamma_rhombic > 0.0 and 2.0 * gamma_rhombic < gamma0
            return c["stable"]

        scoped = [(c, _loc_stable(c)) for c in candidates if _in_lattice(c)]
        scoped.sort(key=lambda pair: (pair[1], pair[0]["score"]), reverse=True)
        selected, selected_lattice_stable = scoped[0] if scoped else (global_selected, global_selected["stable"])
        selected_scope = "lattice-local" if scoped else "global-fallback"

    return {
        "lambda": lam,
        "gamma0": gamma0,
        "gamma_square": gamma_square,
        "gamma_rhombic": gamma_rhombic,
        "gamma_hex": gamma_hex,
        "eta_hex": eta_hex,
        "target_lattice": target_lattice,
        "selected_scope": selected_scope,
        "selected_family": selected["family"],
        "selected_pattern": selected["pattern"],
        "selected_lattice_stable": selected_lattice_stable,
        "global_selected_family": global_selected["family"],
        "global_selected_pattern": global_selected["pattern"],
        "global_selected_stable": global_selected["stable"],
        "roll_stable": roll_stable,
        "square_stable": square["stable"],
        "rhombic_stable": rhombic["stable"],
        "hex_stable": hex_stable,
        "hex_pattern_degenerate": hex_pattern_degenerate,
        "candidates": candidates,
    }


def stability_scan(params: BressloffParams) -> Dict[str, Any]:
    """Scan q in [q_min, q_max] for maximal even/odd branch growth rate."""
    samples = max(params.stability_samples, 2)
    q_min = min(params.stability_q_min, params.stability_q_max)
    q_max = max(params.stability_q_max, q_min + 1.0e-6)
    qs = np.linspace(q_min, q_max, samples)
    even_curve: List[float] = []
    odd_curve: List[float] = []
    critical_q = q_min
    critical_branch = "even"
    critical_growth = float("-inf")

    for q_val in qs:
        q = float(q_val)
        eg = branch_growth(params, "even", q)
        og = branch_growth(params, "odd", q)
        even_curve.append(eg)
        odd_curve.append(og)
        if eg >= critical_growth:
            critical_growth = eg
            critical_q = q
            critical_branch = "even"
        if og >= critical_growth:
            critical_growth = og
            critical_q = q
            critical_branch = "odd"

    eff_params = BressloffParams(**{**params.__dict__})
    if eff_params.pattern == "auto":
        eff_params.parity = critical_branch
    # Rendered wavenumber is always derived from the model's critical q* (|k| = q*/d_hyp), then
    # quantised to an integer number of cycles on the periodic cortical torus of circumference
    # L = 2*pi*beta/eps so that the FFT ring sits exactly on k*. The preset `wave_count` is kept
    # only as a legacy reference value and no longer drives the figure.
    torus_mm = float(eff_params.n) * cell_mm_for(eff_params)
    wave_number_qstar = critical_q / max(eff_params.hypercolumn_mm, 1.0e-9)
    cycles_unquantised = wave_number_qstar * torus_mm / (2.0 * math.pi)
    rendered_cycles = max(1, int(round(cycles_unquantised)))
    wave_number = 2.0 * math.pi * float(rendered_cycles) / torus_mm
    q_eval = wave_number * eff_params.hypercolumn_mm
    bsel = branch_selection_for(eff_params, critical_q, critical_growth)

    return {
        "qs": [float(v) for v in qs],
        "even_growth": even_curve,
        "odd_growth": odd_curve,
        "critical_q": critical_q,
        "critical_branch": critical_branch,
        "critical_growth": critical_growth,
        "rendered_parity": eff_params.parity,
        "wave_number": wave_number,
        "wave_number_source": "critical_q / hypercolumn_mm, quantised to integer torus cycles",
        "wave_number_qstar_unquantised": wave_number_qstar,
        "rendered_cycles": rendered_cycles,
        "cycles_unquantised": cycles_unquantised,
        "critical_wavelength_mm": 2.0 * math.pi / max(wave_number_qstar, 1.0e-12),
        "torus_mm": torus_mm,
        "preset_wave_count_legacy": float(eff_params.wave_count),
        "q_eval": q_eval,
        "branch_selection": bsel,
    }


def planform_modes(params: BressloffParams, pattern: str) -> List[Tuple[float, float, float, float]]:
    """Return list of (normal_angle, phase_scale, phase_offset, amplitude) for a planform family."""
    ang = math.radians(params.pattern_angle_deg)
    if pattern in ("auto", "cobweb"):
        return [(0.0, 1.0, 0.0, 1.0), (math.pi / 2.0, 0.35, 0.0, 1.0)]
    if pattern == "rings":
        return [(0.0, 1.0, 0.0, 1.0)]
    if pattern == "rays":
        return [(math.pi / 2.0, 1.0, 0.0, 1.0)]
    if pattern == "spiral":
        return [(ang, 1.0, 0.0, 1.0)]
    if pattern == "honeycomb":
        return [
            (0.0, 1.0, 0.0, 1.0),
            (2.0 * math.pi / 3.0, 1.0, 0.0, 1.0),
            (-2.0 * math.pi / 3.0, 1.0, 0.0, 1.0),
        ]
    if pattern == "rhombic":
        return [(0.0, 1.0, 0.0, 1.0), (ang, -0.25, 0.0, 1.0)]
    if pattern == "hex_pi":
        return [
            (0.0, 1.0, 0.0, 1.0),
            (2.0 * math.pi / 3.0, -1.0, 0.0, -1.0),
            (-2.0 * math.pi / 3.0, 0.5, 0.0, 1.0),
        ]
    if pattern == "triangle":
        return [
            (0.0, 1.0, -math.pi / 2.0, 1.0),
            (2.0 * math.pi / 3.0, 1.0, -math.pi / 2.0, 1.0),
            (-2.0 * math.pi / 3.0, 1.0, -math.pi / 2.0, 1.0),
        ]
    return [(0.0, 1.0, 0.0, 1.0)]


def render_cortical_and_retinal_planform(
    params: BressloffParams,
    phase: float = 0.0,
    retinal_res: int = 160,
    retinal_max_deg: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Evaluate a V1 cortical planform on the torus [0, L] x [-L/2, L/2] and its exact retinal pullback.

    L = 2*pi*beta/eps is the far-field circumference of the retino-cortical map; x_c = 0 is the fovea.
    The retinal panel covers the central disc |r| <= `retinal_max_deg` using the exact inverse of
    x = (alpha/eps) ln(1 + eps r/w0), y = beta r theta/(w0 + eps r); each hemifield (theta in
    [-pi/2, pi/2] from the horizontal meridian) is mapped onto the same planform, i.e. both
    hemispheres are assumed to carry the same bilateral eigenmode, so the vertical meridian is
    continuous by construction and there is no foveal hole (only r = 0 itself is singular).
    """
    stab = stability_scan(params)
    eff_params = BressloffParams(**{**params.__dict__})
    if eff_params.pattern == "auto":
        eff_params.parity = stab["critical_branch"]
        eff_pattern = stab["branch_selection"]["selected_pattern"]
    else:
        eff_pattern = eff_params.pattern

    cell_mm = cell_mm_for(eff_params)
    extent = float(eff_params.n) * cell_mm
    half = 0.5 * extent
    wave_number = stab["wave_number"]
    q_eval = stab["q_eval"]
    eigen = orientation_eigen_details(eff_params, q_eval)
    modes = planform_modes(eff_params, eff_pattern)

    # 1) Cortical torus grid (x_c in [0, L] with the fovea at x_c = 0, y_c in [-L/2, L/2]) in mm
    xs = (np.arange(eff_params.n, dtype=float) + 0.5) * cell_mm
    ys = (np.arange(eff_params.n, dtype=float) + 0.5) * cell_mm - half
    xg, yg = np.meshgrid(xs, ys)

    def _eval_xy(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        if eff_params.contour_mode == "noncontoured":
            val = np.zeros_like(x_arr, dtype=float)
            for norm_ang, p_scale, p_off, amp in modes:
                proj = x_arr * math.cos(norm_ang) + y_arr * math.sin(norm_ang)
                val += amp * np.cos(wave_number * proj + phase * p_scale + p_off)
            return np.tanh(val * eff_params.sharpness)

        samples = max(eff_params.m, 8)
        best = np.zeros_like(x_arr, dtype=float)
        for k in range(samples):
            phi = math.pi * float(k) / float(samples)
            val = np.zeros_like(x_arr, dtype=float)
            for norm_ang, p_scale, p_off, amp in modes:
                proj = x_arr * math.cos(norm_ang) + y_arr * math.sin(norm_ang)
                spatial = amp * np.cos(wave_number * proj + phase * p_scale + p_off)
                tangent_center = norm_ang + math.pi / 2.0
                u_phi = orientation_eigen_value(phi - tangent_center, eigen)
                val += spatial * u_phi
            mask_better = np.abs(val) > np.abs(best)
            best = np.where(mask_better, val, best)
        return np.tanh(best * eff_params.sharpness)

    cortical_field = _eval_xy(xg, yg)

    # 2) Retinal visual-field pullback through the exact retino-cortical map.
    #    theta is measured from the horizontal meridian within each hemifield so that the
    #    left/right hemifields hit the same cortical planform mirror-symmetrically.
    r_max = float(retinal_max_deg)
    u_lin = np.linspace(-r_max, r_max, retinal_res)
    ug, vg = np.meshgrid(u_lin, u_lin)
    r_ret = np.hypot(ug, vg)
    theta_hemi = np.arctan2(vg, np.abs(ug))
    x_cort, y_cort = retino_cortical_forward(r_ret, theta_hemi)
    retinal_field = _eval_xy(x_cort, y_cort)
    retinal_field = np.where(r_ret <= r_max, retinal_field, np.nan)

    x_cort_max = float(retino_cortical_forward(np.array([r_max]), np.array([0.0]))[0][0])
    stab = dict(stab)
    stab["rendered_pattern"] = eff_pattern
    stab["retinal_max_deg"] = r_max
    stab["cortical_x_for_retinal_max_mm"] = x_cort_max
    stab["cortical_torus_mm"] = extent
    return cortical_field, retinal_field, stab


# =============================================================================
# 3. 2D Radial & Azimuthal Fourier Power Spectrum Metrics
# =============================================================================
def spectral_ring_metrics(field: np.ndarray) -> Dict[str, float]:
    """Compute dominant radial wavenumber k*, wavelength lambda*, ring sharpness, and azimuthal anisotropy."""
    clean = np.nan_to_num(field, nan=0.0).astype(float)
    clean = clean - clean.mean()
    h, w = clean.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    power = np.abs(np.fft.fftshift(np.fft.fft2(clean * window))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(power.shape)
    rr = np.hypot(yy - cy, xx - cx)
    r_int = rr.astype(int)
    max_r = min(h, w) // 2

    radial_sum = np.bincount(r_int.ravel(), weights=power.ravel())[:max_r]
    radial_cnt = np.maximum(np.bincount(r_int.ravel())[:max_r], 1)
    radial_psd = radial_sum / radial_cnt
    radial_psd[:2] = 0.0  # exclude DC / box drift

    k_star_idx = int(np.argmax(radial_psd))
    peak_power = float(radial_psd[k_star_idx])
    mean_power = float(np.mean(radial_psd[2:])) if len(radial_psd) > 2 else 1.0e-12
    ring_sharpness = peak_power / max(mean_power, 1.0e-12)
    wavelength_px = float(min(h, w)) / float(k_star_idx) if k_star_idx > 0 else float("inf")

    # Azimuthal harmonic decomposition along the k* ring (m = 2 roll, m = 4 square, m = 6 hex)
    ring_mask = (rr >= max(1.5, k_star_idx - 1.5)) & (rr <= k_star_idx + 1.5)
    if np.any(ring_mask):
        angles = np.arctan2(yy[ring_mask] - cy, xx[ring_mask] - cx)
        weights = power[ring_mask]
        w_sum = max(float(weights.sum()), 1.0e-12)
        c2 = float(np.abs(np.sum(weights * np.exp(1j * 2.0 * angles))) / w_sum)
        c4 = float(np.abs(np.sum(weights * np.exp(1j * 4.0 * angles))) / w_sum)
        c6 = float(np.abs(np.sum(weights * np.exp(1j * 6.0 * angles))) / w_sum)
    else:
        c2 = c4 = c6 = 0.0

    # Normalised discriminator: a single wavevector pair gives c2 = c4 = c6 = 1 (all harmonics of a
    # two-point ring), so m4/m6 alone cannot separate rolls from lattices. Use the excess over m2.
    m4_excess = max(c4 - c2, 0.0)
    m6_excess = max(c6 - c2, 0.0)
    if c2 > 0.8:
        lattice_class = "roll"
    elif m4_excess > 0.4 and m4_excess >= m6_excess:
        lattice_class = "square"
    elif m6_excess > 0.4:
        lattice_class = "hexagonal"
    else:
        lattice_class = "mixed/rhombic"

    return {
        "k_star_cycles": float(k_star_idx),
        "wavelength_grid_cells": round(wavelength_px, 3),
        "ring_sharpness": round(ring_sharpness, 3),
        "azimuthal_m2_roll": round(c2, 4),
        "azimuthal_m4_square": round(c4, 4),
        "azimuthal_m6_hex": round(c6, 4),
        "m4_excess_over_m2": round(m4_excess, 4),
        "m6_excess_over_m2": round(m6_excess, 4),
        "lattice_class": lattice_class,
    }


# =============================================================================
# 4. Upstream Parity Checks against `MesmerPrism/bressloff-v1-form-constants`
# =============================================================================
def verify_paper_calibration_parity(tol: float = 1.0e-9) -> Tuple[bool, float, List[Dict[str, Any]]]:
    """Verify exact numerical parity with `datasets/bressloff/paper-calibration.json` stability_reports.

    Parity means this script reproduces the upstream fixture's *actual* values. The fixture also
    carries its own paper-expectation `checks` (expected vs actual); those are surfaced per row as
    `fixture_checks` / `fixture_checks_passed` and are NOT folded into the parity verdict, so a
    fixture-level failure (fig40: expected 'triangle', actual 'honeycomb') is reported, not hidden.
    """
    cal_path = BRESSLOFF_DATASET_DIR / "paper-calibration.json"
    with open(cal_path, "r", encoding="utf-8") as fh:
        cal_data = json.load(fh)

    spec_to_preset = {
        "fig37_even_coefficients": "fig17_even",
        "fig38_even_hex_bifurcation": "fig35_hex_zero_even",
        "fig39_odd_coefficients": "fig16_odd",
        "fig40_odd_hex_bifurcation": "fig36_triangle_odd",
        "rhombic_stability_angle": "fig33_rhombic_even",
    }

    max_err = 0.0
    rows: List[Dict[str, Any]] = []
    all_ok = True
    for rep in cal_data["stability_reports"]:
        rep_id = rep["id"]
        preset_id = spec_to_preset[rep_id]
        params = make_preset_params(preset_id, n=128, m=12)
        stab = stability_scan(params)
        bsel = stab["branch_selection"]

        errs = [
            abs(stab["critical_q"] - float(rep["critical_q"])),
            abs(bsel["gamma0"] - float(rep["gamma0"])),
            abs(bsel["gamma_square"] - float(rep["gamma_square"])),
            abs(bsel["gamma_rhombic"] - float(rep["gamma_rhombic"])),
            abs(bsel["gamma_hex"] - float(rep["gamma_hex"])),
            abs(bsel["eta_hex"] - float(rep["eta_hex"])),
        ]
        row_err = max(errs)
        max_err = max(max_err, row_err)
        branch_match = (
            stab["critical_branch"] == rep["critical_branch"]
            and bsel["selected_family"] == rep["selected_family"]
            and bsel["selected_pattern"] == rep["selected_pattern"]
        )
        all_ok = all_ok and branch_match and (row_err <= tol)
        fixture_checks = [
            {
                "name": str(c.get("name")),
                "expected": c.get("expected"),
                "actual": c.get("actual"),
                "passed": bool(c.get("passed")),
            }
            for c in rep.get("checks", [])
        ]
        rows.append(
            {
                "id": rep_id,
                "paper_figure": rep["paper_figure"],
                "critical_branch": stab["critical_branch"],
                "expected_branch": rep["critical_branch"],
                "critical_q": stab["critical_q"],
                "ref_critical_q": float(rep["critical_q"]),
                "selected_family": bsel["selected_family"],
                "selected_pattern": bsel["selected_pattern"],
                "selected_scope": bsel["selected_scope"],
                "global_selected_family": bsel["global_selected_family"],
                "global_selected_pattern": bsel["global_selected_pattern"],
                "global_selected_stable": bsel["global_selected_stable"],
                "roll_stable": bsel["roll_stable"],
                "hex_pattern_degenerate": bsel["hex_pattern_degenerate"],
                "gamma0": bsel["gamma0"],
                "gamma_square": bsel["gamma_square"],
                "gamma_rhombic": bsel["gamma_rhombic"],
                "gamma_hex": bsel["gamma_hex"],
                "eta_hex": bsel["eta_hex"],
                "max_abs_err": row_err,
                "branch_match": branch_match,
                "fixture_checks": fixture_checks,
                "fixture_checks_passed": all(c["passed"] for c in fixture_checks),
                "fixture_failed_checks": [c for c in fixture_checks if not c["passed"]],
            }
        )
    return all_ok, max_err, rows


def summarize_fixture_checks(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the upstream fixture's own paper-expectation checks across all stability reports."""
    total = sum(len(r["fixture_checks"]) for r in rows)
    failed = [(r["id"], c) for r in rows for c in r["fixture_failed_checks"]]
    return {
        "total": total,
        "passed": total - len(failed),
        "failed": [
            {"report": rid, "name": c["name"], "expected": c["expected"], "actual": c["actual"]}
            for rid, c in failed
        ],
        "all_passed": not failed,
    }


def evaluate_gaussian_negative_control() -> Dict[str, float]:
    """Compare Mexican-hat lateral inhibition (lateral_inhibition=1.0) vs Gaussian-only control (0.0)."""
    p_mex = make_preset_params("fig16_odd")
    p_gauss = make_preset_params("fig16_odd")
    p_gauss.lateral_inhibition = 0.0

    stab_mex = stability_scan(p_mex)
    stab_gauss = stability_scan(p_gauss)

    # Interior peak prominence: max(lambda(q)) - lambda(q_min)
    prom_mex = float(max(stab_mex["odd_growth"]) - stab_mex["odd_growth"][0])
    prom_gauss = float(max(stab_gauss["odd_growth"]) - stab_gauss["odd_growth"][0])

    return {
        "mexican_hat_critical_q": float(stab_mex["critical_q"]),
        "mexican_hat_peak_prominence": prom_mex,
        "gaussian_control_critical_q": float(stab_gauss["critical_q"]),
        "gaussian_control_peak_prominence": prom_gauss,
    }


# =============================================================================
# 5. Publication Figure & CSV Export
# =============================================================================
def run_bressloff_analysis(
    output_png: Path = REPO_ROOT / "bressloff_v1_form_constants_morphospace.png",
    output_csv: Path = REPO_ROOT / "bressloff_v1_metrics.csv",
    show: bool = False,
) -> Dict[str, Any]:
    checksums = verify_fixture_checksums()
    for fname, (ok, digest) in checksums.items():
        if not ok:
            raise RuntimeError(f"Fixture SHA-256 mismatch for {fname}: got {digest}")

    parity_ok, max_cal_err, cal_rows = verify_paper_calibration_parity()
    if not parity_ok:
        raise RuntimeError(f"Bressloff calibration parity failed: max_err={max_cal_err:.3e}")
    fixture_summary = summarize_fixture_checks(cal_rows)
    for f in fixture_summary["failed"]:
        print(
            f"[WARN] upstream fixture paper-expectation check failed: {f['report']}/{f['name']} "
            f"expected={f['expected']!r} actual={f['actual']!r} (reported, not hidden)"
        )

    neg_ctrl = evaluate_gaussian_negative_control()
    magnification = compare_cortical_magnification()

    # Representative Klüver form-constant presets across Bressloff (2001) Figs. 31-36 & 2002 Fig. 7
    kluver_presets = [
        ("fig7_lattice_tunnel", "Tunnel / Rings\n(Fig. 7 / Fig. 31b Roll)"),
        ("fig33_rhombic_even_roll", "Spiral / Funnel\n(Fig. 33b Oblique Roll)"),
        ("fig31_square_even", "Cobweb / Square\n(Fig. 31 Even Square)"),
        ("fig35_hex_zero_even", "Honeycomb / Hex\n(Fig. 35 Even Hex)"),
    ]
    rendered_gallery: List[Dict[str, Any]] = []
    for pid, title in kluver_presets:
        p = make_preset_params(pid, n=96, m=16)
        cort, ret, stab = render_cortical_and_retinal_planform(p, phase=0.0, retinal_res=160)
        spec = spectral_ring_metrics(cort)
        # The contoured rendering (max over orientation φ, then tanh) is strongly nonlinear and, for
        # multi-mode lattices, moves most FFT power to the difference-wavevector ring (|k_i - k_j| =
        # √3 k for hexagons). The spectral check against q* is therefore made on the linear
        # eigenmode superposition (non-contoured, weak tanh); both spectra are reported.
        p_lin = BressloffParams(**{**p.__dict__})
        p_lin.contour_mode = "noncontoured"
        p_lin.sharpness = 0.5
        cort_lin, _ret_lin, _stab_lin = render_cortical_and_retinal_planform(p_lin, phase=0.0, retinal_res=16)
        spec_lin = spectral_ring_metrics(cort_lin)
        rendered_gallery.append(
            {
                "preset_id": pid,
                "title": title,
                "cortical": cort,
                "retinal": ret,
                "stability": stab,
                "spectral": spec,
                "spectral_linear": spec_lin,
            }
        )

    # Marginal stability curves (Fig. 16 narrow spread vs Fig. 17 wide spread + Gaussian negative control)
    p16 = make_preset_params("fig16_odd")
    p17 = make_preset_params("fig17_even")
    p_gauss = make_preset_params("fig16_odd")
    p_gauss.lateral_inhibition = 0.0
    stab16 = stability_scan(p16)
    stab17 = stability_scan(p17)
    stab_gauss = stability_scan(p_gauss)

    # Load Rule et al. (2011) Floquet resonance tongue report
    with open(BRESSLOFF_DATASET_DIR / "rule-2011-floquet.json", "r", encoding="utf-8") as fh:
        floquet_data = json.load(fh)
    floquet_fit = floquet_data["source_curve_comparison"].get("fit_objective", {})
    floquet_status = {
        "acceptance_status": str(floquet_fit.get("acceptance_status", "unknown")),
        "calibration_claim_allowed": bool(floquet_fit.get("calibration_claim_allowed", False)),
        "domain_rms_beta_error": float(
            floquet_data["source_curve_comparison"]["domain_beta_mapping"]["rms_error"]
        ),
        "domain_rms_beta_error_max": float(floquet_fit.get("domain_normalized_rms_beta_error_max", float("nan"))),
        "failed_acceptance_checks": list(floquet_fit.get("failed_acceptance_checks", [])),
    }

    # Write CSV summary. Lattice selection is reported twice: `selected_*` is the branch chosen within
    # the preset's target lattice (lattice-local scope) and `global_selected_*` is the branch the
    # cubic amplitude equations select with no lattice restriction.
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "section",
                "id",
                "paper_figure",
                "parity",
                "selected_family",
                "selected_pattern",
                "selected_scope",
                "global_selected_family",
                "global_selected_stable",
                "roll_stable",
                "rendered_pattern",
                "critical_q",
                "gamma0_or_k_star",
                "gamma_cross_or_sharpness",
                "eta_hex_or_m4_m6",
                "lattice_class_or_flag",
                "fixture_paper_checks",
                "max_abs_err_vs_rust",
            ]
        )
        for r in cal_rows:
            n_checks = len(r["fixture_checks"])
            n_pass = sum(1 for c in r["fixture_checks"] if c["passed"])
            fail_txt = "; ".join(
                f"{c['name']}: expected {c['expected']} got {c['actual']}" for c in r["fixture_failed_checks"]
            )
            writer.writerow(
                [
                    "stability_calibration",
                    r["id"],
                    r["paper_figure"],
                    r["critical_branch"],
                    r["selected_family"],
                    r["selected_pattern"],
                    r["selected_scope"],
                    r["global_selected_family"],
                    str(bool(r["global_selected_stable"])),
                    str(bool(r["roll_stable"])),
                    "",
                    f"{r['critical_q']:.6f}",
                    f"{r['gamma0']:.6f}",
                    f"{r['gamma_hex']:.6f}",
                    f"{r['eta_hex']:.6f}",
                    "hex_H0/Hpi_degenerate_at_cubic_order" if r["hex_pattern_degenerate"] else "",
                    f"{n_pass}/{n_checks}" + (f" FAILED[{fail_txt}]" if fail_txt else ""),
                    f"{r['max_abs_err']:.3e}",
                ]
            )
        for g in rendered_gallery:
            sp = g["spectral"]
            st = g["stability"]
            bs = st["branch_selection"]
            writer.writerow(
                [
                    "kluver_form_constant",
                    g["preset_id"],
                    g["title"].replace("\n", " "),
                    st["rendered_parity"],
                    bs["selected_family"],
                    bs["selected_pattern"],
                    bs["selected_scope"],
                    bs["global_selected_family"],
                    str(bool(bs["global_selected_stable"])),
                    str(bool(bs["roll_stable"])),
                    st["rendered_pattern"],
                    f"{st['critical_q']:.6f}",
                    f"{sp['k_star_cycles']:.2f}",
                    f"{sp['ring_sharpness']:.3f}",
                    f"{max(sp['m4_excess_over_m2'], sp['m6_excess_over_m2']):.4f}",
                    sp["lattice_class"],
                    f"rendered_cycles={st['rendered_cycles']} (q*/d_hyp -> {st['cycles_unquantised']:.2f} cyc); "
                    f"linear_k_star={g['spectral_linear']['k_star_cycles']:.0f}; contoured_k_star={sp['k_star_cycles']:.0f}",
                    "n/a",
                ]
            )
        writer.writerow(
            [
                "negative_control",
                "gaussian_only_no_lateral_inhibition",
                "Bressloff 2001 Fig. 16 control",
                "odd",
                "homogeneous",
                "none",
                "global", "homogeneous", "False", "False", "",
                f"{neg_ctrl['gaussian_control_critical_q']:.6f}",
                f"{neg_ctrl['mexican_hat_peak_prominence']:.6f}",
                f"{neg_ctrl['gaussian_control_peak_prominence']:.6f}",
                "0.0000",
                "",
                "",
                "n/a",
            ]
        )
        writer.writerow(
            [
                "magnification_anchor",
                "horton_hoyt_1991",
                "Horton & Hoyt 1991 M(E)=17.3/(E+0.75)",
                "", "", "", "", "", "", "", "",
                f"{magnification['rms_rel_error']:.6f}",
                f"{magnification['model_A_mm_per_deg']:.6f}",
                f"{magnification['model_E2_deg']:.6f}",
                f"{magnification['max_abs_rel_error']:.6f}",
                f"model_extent_0_90={magnification['model_extent_0_90_mm']:.2f}mm vs HH {magnification['hh_extent_0_90_mm']:.2f}mm",
                "rms_rel_err / A_model / E2_model / max_rel_err over 1-40 deg",
                "n/a",
            ]
        )
        writer.writerow(
            [
                "floquet_diagnostic",
                "rule_2011_fig8_overlay",
                "Rule et al. 2011 Fig. 8",
                "", "", "", "", "", "", "", "",
                f"{floquet_status['domain_rms_beta_error']:.6f}",
                f"{floquet_status['domain_rms_beta_error_max']:.6f}",
                "", "",
                floquet_status["acceptance_status"],
                f"calibration_claim_allowed={floquet_status['calibration_claim_allowed']}; "
                f"failed={','.join(floquet_status['failed_acceptance_checks'])}",
                "n/a",
            ]
        )

    # Render 4x4 synthesis figure
    fig = plt.figure(figsize=(15.5, 15.0), dpi=150)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.08, 0.95], hspace=0.5, wspace=0.3)

    cmap_planform = "twilight_shifted"
    for col, g in enumerate(rendered_gallery):
        ax_c = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])
        st = g["stability"]
        bs = st["branch_selection"]
        torus = st["cortical_torus_mm"]

        ax_c.imshow(
            g["cortical"], cmap=cmap_planform, vmin=-1.0, vmax=1.0, origin="lower",
            extent=(0.0, torus, -0.5 * torus, 0.5 * torus),
        )
        ax_c.axvline(st["cortical_x_for_retinal_max_mm"], color="white", ls="--", lw=0.9, alpha=0.85)
        ax_c.text(
            st["cortical_x_for_retinal_max_mm"] + 1.5, -0.47 * torus, f"{st['retinal_max_deg']:.0f}°",
            color="white", fontsize=7.0, va="bottom",
        )
        ax_c.set_title(
            f"V1 Cortex: {g['title']}\n"
            f"$k^*$ lin {g['spectral_linear']['k_star_cycles']:.0f} / cont {g['spectral']['k_star_cycles']:.0f} cyc "
            f"($q^*/d_{{hyp}}$={st['cycles_unquantised']:.1f})\n"
            f"$\\lambda_c$={st['critical_wavelength_mm']:.1f} mm, FFT class: {g['spectral']['lattice_class']}",
            fontsize=7.4,
            fontweight="bold",
        )
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        if col == 0:
            ax_c.set_ylabel("V1 Cortical Sheet $(x_c, y_c)$, fovea at left", fontsize=9.0, fontweight="bold")

        rmax = st["retinal_max_deg"]
        ax_r.imshow(
            g["retinal"], cmap=cmap_planform, vmin=-1.0, vmax=1.0, origin="lower",
            extent=(-rmax, rmax, -rmax, rmax),
        )
        ax_r.set_title(
            f"Retinal Pullback (exact map, $|r|\\leq{rmax:.0f}°$)\n"
            f"rendered={st['rendered_pattern']} | {bs['selected_scope']}: {bs['selected_family']}\n"
            f"global cubic selection: {bs['global_selected_family']}{'' if bs['global_selected_stable'] else ' (unstable)'}",
            fontsize=7.4,
        )
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        if col == 0:
            ax_r.set_ylabel("Retinal Visual Field (exact inverse map)", fontsize=9.0, fontweight="bold")

    # Bottom Row Panel A (cols 0..1): Bressloff (2001) Figs. 16 & 17 marginal stability + Gaussian negative control
    ax_stab = fig.add_subplot(gs[2, 0:2])
    ax_stab.plot(
        stab16["qs"],
        stab16["odd_growth"],
        color="#c0392b",
        lw=2.2,
        label=f"Fig. 16 Odd branch ($\\theta_0=0^\\circ$, $q^*={stab16['critical_q']:.2f}$)",
    )
    ax_stab.plot(
        stab16["qs"],
        stab16["even_growth"],
        color="#e67e22",
        ls="--",
        lw=1.6,
        label="Fig. 16 Even branch ($\\theta_0=0^\\circ$)",
    )
    ax_stab.plot(
        stab17["qs"],
        stab17["even_growth"],
        color="#2980b9",
        lw=2.2,
        label=f"Fig. 17 Even branch ($\\theta_0=60^\\circ$, $q^*={stab17['critical_q']:.2f}$)",
    )
    ax_stab.plot(
        stab_gauss["qs"],
        stab_gauss["odd_growth"],
        color="#7f8c8d",
        ls=":",
        lw=2.0,
        label=f"Gaussian-only negative control ($I_{{\\mathrm{{lat}}}}=0$, $q^*\\to {stab_gauss['critical_q']:.2f}$)",
    )
    ax_stab.axvline(stab16["critical_q"], color="#c0392b", ls=":", alpha=0.6)
    ax_stab.axvline(stab17["critical_q"], color="#2980b9", ls=":", alpha=0.6)
    ax_stab.set_xlabel("Dimensionless Cortical Wavenumber $q = |\\mathbf{k}| d_{\\mathrm{hyp}}$", fontsize=9.0)
    ax_stab.set_ylabel("Perturbative Eigenvalue $\\lambda(q)$", fontsize=9.0)
    ax_stab.set_title(
        "A. Bressloff (2001) Figs. 16–17 Marginal Stability & Mexican-Hat Negative Control",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_stab.legend(fontsize=7.8, loc="lower right")
    ax_stab.grid(alpha=0.25)

    # Bottom Row Panel B (col 2): Cubic amplitude-equation coefficients (Figs. 37-40)
    ax_amp = fig.add_subplot(gs[2, 2])
    thetas_deg = np.linspace(0.0, 180.0, 181)
    eigen_even = orientation_eigen_details(p17, stab17["critical_q"])
    eigen_odd = orientation_eigen_details(p16, stab16["critical_q"])
    g3_even = [amplitude_gamma3(math.radians(t), eigen_even) for t in thetas_deg]
    g3_odd = [amplitude_gamma3(math.radians(t), eigen_odd) for t in thetas_deg]
    ax_amp.plot(thetas_deg, g3_even, color="#2980b9", lw=2.1, label="Even $\\gamma_3(\\theta)$ (Fig. 37)")
    ax_amp.plot(thetas_deg, g3_odd, color="#c0392b", lw=2.1, label="Odd $\\gamma_3(\\theta)$ (Fig. 39)")
    ax_amp.axhline(0.5 * g3_even[0], color="#2980b9", ls="--", lw=1.1, label="$\\frac{1}{2}\\gamma_3(0)$ roll/lattice boundary")
    for ang_mark, lbl in [(60.0, "Hex"), (90.0, "Sq")]:
        ax_amp.axvline(ang_mark, color="#555555", ls=":", lw=1.0)
        ax_amp.text(ang_mark + 2.0, max(g3_even) * 0.92, lbl, fontsize=7.5, color="#333333")
    ax_amp.set_xlabel("Wavevector Intersection Angle $\\theta$ (deg)", fontsize=9.0)
    ax_amp.set_ylabel("Cubic Coupling $\\gamma_3(\\theta)$", fontsize=9.0)
    ax_amp.set_title(
        f"B. Amplitude Selection (Figs. 37–40)\nRust parity max $|\\Delta| = {max_cal_err:.1e}$\n"
        f"fixture paper checks {fixture_summary['passed']}/{fixture_summary['total']}"
        + (" (fig40 H0/Hπ degenerate)" if not fixture_summary["all_passed"] else ""),
        fontsize=8.0,
        fontweight="bold",
    )
    ax_amp.legend(fontsize=7.5, loc="upper right")
    ax_amp.grid(alpha=0.25)

    # Bottom Row Panel C (col 3): Rule et al. (2011) Fig. 8 Floquet 1:1 vs 2:1 Resonance Tongues
    ax_rule = fig.add_subplot(gs[2, 3])
    plotted_plus = False
    plotted_minus = False
    beta_scale = float(
        floquet_data.get("wave_number_normalization", {}).get("source_beta_per_model_cycle", 0.42868451880191133)
    )
    for curve in floquet_data.get("boundary_curves", []):
        pts = curve.get("points", [])
        if not pts:
            continue
        periods = [float(p["period_ms"]) for p in pts]
        betas = [
            float(p.get("source_beta", float(p["beta_cycles"]) * beta_scale))
            for p in pts
        ]
        cid = curve.get("curve_id", "")
        if "minus_period_doubling" in cid:
            ax_rule.plot(
                periods,
                betas,
                color="#8e44ad",
                lw=1.8,
                marker="o",
                ms=3.0,
                label="2:1 Subharmonic Hex ($-1$)" if not plotted_minus else None,
            )
            plotted_minus = True
        else:
            ax_rule.plot(
                periods,
                betas,
                color="#16a085",
                lw=1.8,
                marker="s",
                ms=3.0,
                label="1:1 Harmonic Stripes ($+1$)" if not plotted_plus else None,
            )
            plotted_plus = True

    ax_rule.set_xlabel("Flicker Forcing Period $T$ (ms)", fontsize=9.0)
    ax_rule.set_ylabel("Normalized Wavenumber $\\beta_{\\mathrm{source}}$", fontsize=9.0)
    ax_rule.set_title(
        "C. Rule et al. (2011) Fig. 8 Floquet Tongues\nDIAGNOSTIC OVERLAY — "
        f"{floquet_status['acceptance_status']},\ncalibration_claim_allowed={floquet_status['calibration_claim_allowed']}",
        fontsize=8.0,
        fontweight="bold",
        color="#7b241c",
    )
    ax_rule.text(
        0.02, 0.03,
        f"domain-normalised RMS β error = {floquet_status['domain_rms_beta_error']:.3f} "
        f"(threshold {floquet_status['domain_rms_beta_error_max']:.2f})\n"
        f"failed: {len(floquet_status['failed_acceptance_checks'])} acceptance checks\n"
        "numerical calibration targets, not the published figure",
        transform=ax_rule.transAxes, fontsize=6.4, va="bottom", ha="left", color="#7b241c",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fdf2f0", ec="#e6b0aa", lw=0.6),
    )
    ax_rule.legend(fontsize=7.4, loc="upper right")
    ax_rule.grid(alpha=0.25)

    # Bottom Row Panel D (cols 0..1): empirical anchor — cortical magnification vs Horton & Hoyt (1991)
    ax_mag = fig.add_subplot(gs[3, 0:2])
    ax_mag.plot(
        magnification["eccentricity_deg"], magnification["horton_hoyt_mm_per_deg"],
        color="#1b2631", lw=2.4,
        label=f"Horton & Hoyt (1991) human V1: $M(E)=A/(E+E_2)$, A={magnification['hh_A_mm_per_deg']:.1f}, $E_2$={magnification['hh_E2_deg']:.2f}°",
    )
    ax_mag.plot(
        magnification["eccentricity_deg"], magnification["model_mm_per_deg"],
        color="#2980b9", lw=2.0, ls="--",
        label=f"Bressloff map $dx/dr=\\alpha/(w_0+\\epsilon r)$: A={magnification['model_A_mm_per_deg']:.2f}, $E_2$={magnification['model_E2_deg']:.2f}°",
    )
    ax_mag.set_yscale("log")
    ax_mag.set_xlabel("Eccentricity $E$ (deg)", fontsize=9.0)
    ax_mag.set_ylabel("Linear magnification (mm / deg)", fontsize=9.0)
    ax_mag.set_title(
        "D. Empirical anchor: retino-cortical magnification vs. Horton & Hoyt (1991)\n"
        f"RMS relative error {100.0 * magnification['rms_rel_error']:.1f}% (max {100.0 * magnification['max_abs_rel_error']:.1f}%) over 1–40°; "
        f"fovea→90° extent {magnification['model_extent_0_90_mm']:.1f} mm (model) vs {magnification['hh_extent_0_90_mm']:.1f} mm (H&H)",
        fontsize=9.0, fontweight="bold",
    )
    ax_mag.legend(fontsize=7.6, loc="upper right")
    ax_mag.grid(alpha=0.25, which="both")

    # Bottom Row Panel E (cols 2..3): calibration & provenance ledger (what is and is not claimed)
    ax_led = fig.add_subplot(gs[3, 2:4])
    ax_led.axis("off")
    ledger_lines = [
        f"Fixture SHA-256: {sum(1 for ok, _ in checksums.values() if ok)}/{len(checksums)} verified",
        f"Rust stability parity (q*, γ0, γ(θ), η): max |Δ| = {max_cal_err:.1e}, {len(cal_rows)} reports",
        f"Upstream fixture paper-expectation checks: {fixture_summary['passed']}/{fixture_summary['total']} passed",
    ]
    for f in fixture_summary["failed"]:
        ledger_lines.append(f"   ✗ {f['report']}/{f['name']}:")
        ledger_lines.append(f"     expected '{f['expected']}', got '{f['actual']}'")
    ledger_lines += [
        "   odd branch has η_hex = 0 ⇒ H0 (honeycomb) vs Hπ (triangle) undetermined",
        "   at cubic order; quintic selection (Bressloff 2001 §5) NOT implemented",
        "Lattice scope: gallery presets select within their target lattice",
        "   (lattice-local); the unrestricted global branch is printed under",
        "   each retinal panel and in the CSV",
        "Rendered |k| = q*/d_hyp, quantised to integer cycles on the L = 2πβ/ε",
        "   torus (preset wave_count no longer used)",
        "Retinal pullback: exact x=(α/ε)ln(1+εr/w₀), y=βrθ/(w₀+εr);",
        "   both hemispheres carry the same mode (mirror-symmetric)",
        f"Rule (2011) overlay: {floquet_status['acceptance_status']},",
        f"   RMS β err {floquet_status['domain_rms_beta_error']:.3f} — no calibration claim",
        f"Negative control: Gaussian-only (I_lat=0) q* → {neg_ctrl['gaussian_control_critical_q']:.2f},",
        f"   interior-peak prominence {neg_ctrl['gaussian_control_peak_prominence']:.1e}",
    ]
    ax_led.text(
        0.0, 1.0, "\n".join(ledger_lines), transform=ax_led.transAxes, fontsize=7.0, va="top", ha="left",
        family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9f9", ec="#b3b6b7", lw=0.7),
    )
    ax_led.set_title("E. Calibration & provenance ledger (claims vs. diagnostics)", fontsize=9.0, fontweight="bold")

    fig.suptitle(
        "Bressloff–Cowan–Golubitsky–Thomas–Wiener (2001/2002) V1 Euclidean Symmetry-Breaking & Form Constants\n"
        "Cortical Orientation Planforms, Exact Retinal Pullbacks, Amplitude Selection, Horton–Hoyt Magnification Anchor & Rule (2011) Floquet Diagnostics",
        fontsize=11.0,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved Bressloff V1 form-constants figure to {output_png}")
    print(f"Saved quantitative metrics table to {output_csv}")

    return {
        "checksums": checksums,
        "parity_ok": parity_ok,
        "max_cal_err": max_cal_err,
        "calibration_rows": cal_rows,
        "fixture_checks": fixture_summary,
        "negative_control": neg_ctrl,
        "gallery": rendered_gallery,
        "magnification": magnification,
        "floquet_status": floquet_status,
        "floquet_overlap_rms": floquet_status["domain_rms_beta_error"],
        "output_png": str(output_png),
        "output_csv": str(output_csv),
    }


DEFAULT_GIF_PATH = REPO_ROOT / "docs" / "media" / "bressloff_v1_form_constants.gif"


def make_gif(
    output_gif: Path = DEFAULT_GIF_PATH,
    n_frames: int = 36,
    fps: int = 8,
) -> None:
    """Animate the same four Klüver presets as the PNG gallery (1x2: V1 cortical torus -> exact retinal pullback).

    The only difference from the static figure is `contour_mode="noncontoured"` so the phase sweep is
    smooth; wavenumber, parity, lattice and map are identical to the PNG (no wave-count override).
    """
    from PIL import Image

    presets = [
        ("fig7_lattice_tunnel", "1/4  Tunnel / Rings", "vertical V1 stripes"),
        ("fig33_rhombic_even_roll", "2/4  Spiral / Funnel", "oblique V1 stripes (60°)"),
        ("fig31_square_even", "3/4  Cobweb / Square", "orthogonal V1 square lattice"),
        ("fig35_hex_zero_even", "4/4  Honeycomb / Hex", "three-wave V1 hex lattice"),
    ]
    params_list = [(pid, lbl, sub, make_preset_params(pid, n=72, m=8)) for pid, lbl, sub in presets]
    frames_rgb: List[Image.Image] = []
    frames_per_preset = max(1, n_frames // len(params_list))

    for fi in range(n_frames):
        idx = min(fi // frames_per_preset, len(params_list) - 1)
        local_fi = fi - idx * frames_per_preset
        phase = 2.0 * math.pi * float(local_fi) / float(frames_per_preset)
        _pid, lbl, sub, p = params_list[idx]

        # Non-contoured spatial envelope + continuous phase sweep for smooth motion; same q*-derived
        # wavenumber, parity and lattice as the PNG gallery.
        p_anim = BressloffParams(**{**p.__dict__})
        p_anim.contour_mode = "noncontoured"

        cort, ret, stab_anim = render_cortical_and_retinal_planform(p_anim, phase=phase, retinal_res=140)

        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), dpi=100, facecolor="#fafaf8")
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.80, wspace=0.10)

        axes[0].imshow(cort, cmap="twilight_shifted", vmin=-1.0, vmax=1.0, origin="lower", interpolation="bilinear")
        axes[0].set_title(f"V1 Cortex ({sub})", fontsize=10.0, fontweight="bold", color="#1e293b", pad=6)
        axes[0].axis("off")

        axes[1].set_facecolor("#fafaf8")
        axes[1].imshow(ret, cmap="twilight_shifted", vmin=-1.0, vmax=1.0, origin="lower", interpolation="bilinear")
        axes[1].set_title(
            f"Retina (exact map, |r| ≤ {stab_anim['retinal_max_deg']:.0f}°)",
            fontsize=10.0, fontweight="bold", color="#1e293b", pad=6,
        )
        axes[1].axis("off")

        fig.suptitle(
            f"Bressloff et al. (2001) — {lbl}  (k*={stab_anim['rendered_cycles']} cyc, non-contoured)",
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
            y=0.95,
        )
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames_rgb.append(Image.fromarray(buf[:, :, :3]).convert("RGB"))
        plt.close(fig)

    w, h = frames_rgb[0].size
    strip = Image.new("RGB", (w, h * 4))
    for i, fr in enumerate(frames_rgb[:: max(1, len(frames_rgb) // 4)][:4]):
        strip.paste(fr, (0, i * h))
    pal = strip.quantize(colors=96, method=Image.MEDIANCUT)
    q_frames = [fr.quantize(palette=pal, dither=Image.NONE) for fr in frames_rgb]

    q_frames[0].save(
        output_gif,
        save_all=True,
        append_images=q_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"Saved Bressloff form-constants animation to {output_gif} ({w}x{h}, {len(q_frames)} frames, {output_gif.stat().st_size // 1024} KB)")


def self_test() -> bool:
    """Run deterministic self-tests for Bressloff (2001/2002) and Rule (2011) fixtures."""
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
        ok = ok and bool(cond)

    for fname, (passed, digest) in verify_fixture_checksums().items():
        check(passed, f"SHA-256 verified for {fname}: {digest[:16]}...")

    parity_ok, max_err, rows = verify_paper_calibration_parity()
    check(
        parity_ok,
        f"Exact parity with MesmerPrism/bressloff-v1-form-constants stability_reports (max |diff| = {max_err:.2e})",
    )
    for r in rows:
        check(
            r["branch_match"],
            f"  {r['id']} ({r['paper_figure']}): branch={r['critical_branch']}, "
            f"q*={r['critical_q']:.4f}, selected={r['selected_family']}/{r['selected_pattern']} "
            f"[{r['selected_scope']}; global={r['global_selected_family']}]",
        )
    fx = summarize_fixture_checks(rows)
    print(
        f"[INFO] upstream fixture paper-expectation checks: {fx['passed']}/{fx['total']} passed"
        + "".join(
            f"\n       ✗ {f['report']}/{f['name']}: expected {f['expected']!r}, actual {f['actual']!r}"
            for f in fx["failed"]
        )
    )
    check(
        any(f["report"] == "fig40_odd_hex_bifurcation" and f["name"] == "selected-pattern" for f in fx["failed"]),
        "Known fig40 odd-hex H0/Hπ degeneracy is surfaced (not hidden) by the parity report",
    )

    # Exact retino-cortical map round-trip and Horton & Hoyt anchor
    r_test = np.array([0.5, 2.0, 10.0, 40.0, 90.0])
    th_test = np.array([-1.2, -0.3, 0.0, 0.7, 1.5])
    xc, yc = retino_cortical_forward(r_test, th_test)
    r_back, th_back = retino_cortical_inverse(xc, yc)
    check(
        bool(np.allclose(r_back, r_test, rtol=1e-10, atol=1e-10) and np.allclose(th_back, th_test, rtol=1e-10, atol=1e-10)),
        f"Exact retino-cortical map inverts analytically (max |Δr| = {np.max(np.abs(r_back - r_test)):.1e} deg)",
    )
    mag = compare_cortical_magnification()
    check(
        mag["rms_rel_error"] < 0.25,
        f"Map magnification within 25% RMS of Horton & Hoyt (1991) over 1-40 deg "
        f"(RMS rel. err = {100.0 * mag['rms_rel_error']:.1f}%, A={mag['model_A_mm_per_deg']:.2f} vs 17.3, "
        f"E2={mag['model_E2_deg']:.2f} vs 0.75 deg)",
    )

    neg = evaluate_gaussian_negative_control()
    check(
        neg["mexican_hat_peak_prominence"] > 0.05 and neg["gaussian_control_peak_prominence"] <= 1.0e-9,
        f"Mexican-hat lateral inhibition produces interior wavenumber peak "
        f"(q*={neg['mexican_hat_critical_q']:.3f}, prominence={neg['mexican_hat_peak_prominence']:.4f}) "
        f"while Gaussian-only control collapses to q_min={neg['gaussian_control_critical_q']:.3f} "
        f"(prominence={neg['gaussian_control_peak_prominence']:.2e})",
    )

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_bressloff_analysis(
            output_png=Path(td) / "fig.png",
            output_csv=Path(td) / "metrics.csv",
            show=False,
        )
        ks = [g["spectral_linear"]["k_star_cycles"] for g in res["gallery"]]
        ks_c = [g["spectral"]["k_star_cycles"] for g in res["gallery"]]
        cyc = [g["stability"]["rendered_cycles"] for g in res["gallery"]]
        check(
            all(k >= 6.0 for k in ks) and all(abs(k - c) <= 1.0 for k, c in zip(ks, cyc)),
            f"All 4 Klüver linear planforms have FFT rings at the q*-derived cycle count: k* = {ks}, rendered = {cyc} "
            f"(contoured-rendering k* = {ks_c}; hex moves power to the √3·k difference ring)",
        )
        classes = [g["spectral"]["lattice_class"] for g in res["gallery"]]
        check(
            classes == ["roll", "roll", "square", "hexagonal"],
            f"Normalised azimuthal discriminator classifies rings/spiral as roll, cobweb as square, honeycomb as hexagonal: {classes}",
        )
        check(
            not res["floquet_status"]["calibration_claim_allowed"] and res["floquet_overlap_rms"] > 0.1,
            f"Rule (2011) overlay is reported as diagnostic only (status={res['floquet_status']['acceptance_status']}, "
            f"RMS β err={res['floquet_overlap_rms']:.3f})",
        )
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / "bressloff_v1_form_constants_morphospace.png"))
    parser.add_argument("--csv", default=str(REPO_ROOT / "bressloff_v1_metrics.csv"))
    parser.add_argument("--gif", nargs="?", const=str(DEFAULT_GIF_PATH), help=f"Write the animation (default path: {DEFAULT_GIF_PATH.relative_to(REPO_ROOT)})")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive matplotlib window")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic self-test and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    run_bressloff_analysis(
        output_png=Path(args.out),
        output_csv=Path(args.csv),
        show=not args.no_show,
    )
    if args.gif:
        make_gif(output_gif=Path(args.gif))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
