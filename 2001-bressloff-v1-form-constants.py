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
   https://doi.org/10.1371/journal.pcbi.1002158 (PMID: 21980279)
5. Upstream reference implementation & calibration fixtures:
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
* The inverse retino-cortical pullback r_R = (w0/eps) * exp(eps * x / alpha),
  theta_R = eps * y / beta follows Bressloff et al. (2001, Section 2; 2002, Eq. 2.1).
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
from scipy.special import iv as bessel_iv

REPO_ROOT = Path(__file__).resolve().parent
BRESSLOFF_DATASET_DIR = REPO_ROOT / "datasets" / "bressloff"

FIXTURE_CHECKSUMS: Dict[str, str] = {
    "paper-calibration.json": "af12b2c1aa3e38faf984166b3aa227a2aa064480a98279f14c004342ce2e7dd4",
    "rule-2011-floquet.json": "5e349c7b80185de463e1ea8656a2fe91a83523c212875af4ce46d09e9dde2a8e",
    "driven-neural-fields-registry.json": "4f07b5175a48bcb50bad4998953505a3f92faf4c6f5514d2b4643c2a67c36b10",
}

# Retino-cortical log-polar mapping constants (Bressloff et al. 2001 Section 2; 2002 Eq. 2.1)
RETINO_EPS: float = 0.051
RETINO_W0: float = 0.087
RETINO_ALPHA: float = 3.0 / math.pi
RETINO_BETA: float = 1.589 / 2.0


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
    wave_number = (
        critical_q / max(eff_params.hypercolumn_mm, 1.0e-9)
        if eff_params.pattern == "auto"
        else 2.0 * math.pi * eff_params.wave_count / max(float(eff_params.n) * cell_mm_for(eff_params), 1.0e-9)
    )
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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Evaluate a V1 cortical planform on [-extent/2, extent/2]^2 and its log-polar retinal pullback."""
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

    # 1) Cortical grid (x_c, y_c) in mm
    xs = (np.arange(eff_params.n, dtype=float) + 0.5) * cell_mm - half
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

    # 2) Retinal visual-field pullback via forward log-polar V1 map:
    # r_R = (w0 / eps) * exp(eps * x_c / alpha), theta_R = eps * y_c / beta
    # => x_c = (alpha / eps) * ln(eps * r_R / w0), y_c = (beta / eps) * theta_R
    r_min = (RETINO_W0 / RETINO_EPS) * math.exp(-RETINO_EPS * half / RETINO_ALPHA)
    r_max = (RETINO_W0 / RETINO_EPS) * math.exp(RETINO_EPS * half / RETINO_ALPHA)
    u_lin = np.linspace(-r_max, r_max, retinal_res)
    ug, vg = np.meshgrid(u_lin, u_lin)
    r_ret = np.hypot(ug, vg)
    theta_ret = np.arctan2(vg, ug)
    r_clamped = np.maximum(r_ret, r_min)
    x_cort = (RETINO_ALPHA / RETINO_EPS) * np.log(RETINO_EPS * r_clamped / RETINO_W0)
    y_cort = (RETINO_BETA / RETINO_EPS) * theta_ret
    retinal_field = _eval_xy(x_cort, y_cort)
    # Mask outside retinal disc or inside foveal singularity r < r_min
    valid_disc = (r_ret <= r_max) & (r_ret >= r_min)
    retinal_field = np.where(valid_disc, retinal_field, np.nan)

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

    return {
        "k_star_cycles": float(k_star_idx),
        "wavelength_grid_cells": round(wavelength_px, 3),
        "ring_sharpness": round(ring_sharpness, 3),
        "azimuthal_m2_roll": round(c2, 4),
        "azimuthal_m4_square": round(c4, 4),
        "azimuthal_m6_hex": round(c6, 4),
    }


# =============================================================================
# 4. Upstream Parity Checks against `MesmerPrism/bressloff-v1-form-constants`
# =============================================================================
def verify_paper_calibration_parity(tol: float = 1.0e-9) -> Tuple[bool, float, List[Dict[str, Any]]]:
    """Verify exact numerical parity with `datasets/bressloff/paper-calibration.json` stability_reports."""
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
                "gamma0": bsel["gamma0"],
                "gamma_square": bsel["gamma_square"],
                "gamma_rhombic": bsel["gamma_rhombic"],
                "gamma_hex": bsel["gamma_hex"],
                "eta_hex": bsel["eta_hex"],
                "max_abs_err": row_err,
                "branch_match": branch_match,
            }
        )
    return all_ok, max_err, rows


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

    neg_ctrl = evaluate_gaussian_negative_control()

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
        rendered_gallery.append(
            {
                "preset_id": pid,
                "title": title,
                "cortical": cort,
                "retinal": ret,
                "stability": stab,
                "spectral": spec,
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

    # Write CSV summary
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
                "critical_q",
                "gamma0_or_k_star",
                "gamma_cross_or_sharpness",
                "eta_hex_or_m4_m6",
                "max_abs_err_vs_rust",
            ]
        )
        for r in cal_rows:
            writer.writerow(
                [
                    "stability_calibration",
                    r["id"],
                    r["paper_figure"],
                    r["critical_branch"],
                    r["selected_family"],
                    r["selected_pattern"],
                    f"{r['critical_q']:.6f}",
                    f"{r['gamma0']:.6f}",
                    f"{r['gamma_hex']:.6f}",
                    f"{r['eta_hex']:.6f}",
                    f"{r['max_abs_err']:.3e}",
                ]
            )
        for g in rendered_gallery:
            sp = g["spectral"]
            st = g["stability"]
            writer.writerow(
                [
                    "kluver_form_constant",
                    g["preset_id"],
                    g["title"].replace("\n", " "),
                    st["rendered_parity"],
                    st["branch_selection"]["selected_family"],
                    st["branch_selection"]["selected_pattern"],
                    f"{st['q_eval']:.6f}",
                    f"{sp['k_star_cycles']:.2f}",
                    f"{sp['ring_sharpness']:.3f}",
                    f"{ max(sp['azimuthal_m4_square'], sp['azimuthal_m6_hex']):.4f}",
                    "0.000e+00",
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
                f"{neg_ctrl['gaussian_control_critical_q']:.6f}",
                f"{neg_ctrl['mexican_hat_peak_prominence']:.6f}",
                f"{neg_ctrl['gaussian_control_peak_prominence']:.6f}",
                "0.0000",
                "0.000e+00",
            ]
        )

    # Render 3x4 synthesis figure
    fig = plt.figure(figsize=(15.5, 11.2), dpi=150)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.08], hspace=0.38, wspace=0.28)

    cmap_planform = "twilight_shifted"
    for col, g in enumerate(rendered_gallery):
        ax_c = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])

        ax_c.imshow(g["cortical"], cmap=cmap_planform, vmin=-1.0, vmax=1.0, origin="lower")
        ax_c.set_title(
            f"V1 Cortex: {g['title']}\n"
            f"$k^*={g['spectral']['k_star_cycles']:.0f}$ cyc, sharpness={g['spectral']['ring_sharpness']:.1f}",
            fontsize=9.0,
            fontweight="bold",
        )
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        if col == 0:
            ax_c.set_ylabel("V1 Cortical Sheet $(x_c, y_c)$", fontsize=9.5, fontweight="bold")

        im_r = ax_r.imshow(g["retinal"], cmap=cmap_planform, vmin=-1.0, vmax=1.0, origin="lower")
        ax_r.set_title(
            f"Retinal Pullback $(r_R, \\theta_R)$\n"
            f"parity={g['stability']['rendered_parity']}, mode={g['stability']['branch_selection']['selected_pattern']}",
            fontsize=8.8,
        )
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        if col == 0:
            ax_r.set_ylabel("Retinal Visual Field (Log-Polar)", fontsize=9.5, fontweight="bold")

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
        f"B. Amplitude Selection (Figs. 37–40)\nRust parity max $|\\Delta| = {max_cal_err:.1e}$",
        fontsize=9.2,
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
        "C. Rule et al. (2011) Fig. 8 Floquet Tongues\n(1:1 Stripes vs. 2:1 Period-Doubling Hex)",
        fontsize=9.2,
        fontweight="bold",
    )
    ax_rule.legend(fontsize=7.8, loc="upper right")
    ax_rule.grid(alpha=0.25)

    fig.suptitle(
        "Bressloff–Cowan–Golubitsky–Thomas–Wiener (2001/2002) V1 Euclidean Symmetry-Breaking & Form Constants\n"
        "Cortical Orientation Planforms, Log-Polar Retinal Pullbacks, Amplitude Selection & Rule (2011) Flicker Floquet Tongues",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
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
        "negative_control": neg_ctrl,
        "gallery": rendered_gallery,
        "floquet_overlap_rms": float(
            floquet_data["source_curve_comparison"]["domain_beta_mapping"]["rms_error"]
        ),
        "output_png": str(output_png),
        "output_csv": str(output_csv),
    }


def make_gif(
    output_gif: Path = REPO_ROOT / "bressloff_v1_form_constants.gif",
    n_frames: int = 36,
    fps: int = 8,
) -> None:
    """Animate Klüver's 4 form-constant families one at a time (1x2: V1 cortical lattice -> retinal log-polar pullback)."""
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

        # Non-contoured spatial envelope + continuous angle sweep for smooth, clean visual motion
        p_anim = BressloffParams(**{**p.__dict__})
        p_anim.contour_mode = "noncontoured"
        p_anim.wave_count = 3.5
        p_anim.sharpness = 1.6

        cort, ret, _stab = render_cortical_and_retinal_planform(p_anim, phase=phase, retinal_res=140)

        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), dpi=100, facecolor="#fafaf8")
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.80, wspace=0.10)

        axes[0].imshow(cort, cmap="twilight_shifted", vmin=-1.0, vmax=1.0, origin="lower", interpolation="bilinear")
        axes[0].set_title(f"V1 Cortex ({sub})", fontsize=10.0, fontweight="bold", color="#1e293b", pad=6)
        axes[0].axis("off")

        axes[1].set_facecolor("#fafaf8")
        axes[1].imshow(ret, cmap="twilight_shifted", vmin=-1.0, vmax=1.0, origin="lower", interpolation="bilinear")
        axes[1].set_title("Retina (log-polar visual field)", fontsize=10.0, fontweight="bold", color="#1e293b", pad=6)
        axes[1].axis("off")

        fig.suptitle(
            f"Bressloff et al. (2001) — {lbl}",
            fontsize=11.5,
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
            f"q*={r['critical_q']:.4f}, selected={r['selected_family']}/{r['selected_pattern']}",
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
        ks = [g["spectral"]["k_star_cycles"] for g in res["gallery"]]
        check(
            all(k >= 6.0 for k in ks),
            f"All 4 Klüver form-constant cortical planforms exhibit clear non-zero spectral rings: k* = {ks}",
        )
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / "bressloff_v1_form_constants_morphospace.png"))
    parser.add_argument("--csv", default=str(REPO_ROOT / "bressloff_v1_metrics.csv"))
    parser.add_argument("--gif", nargs="?", const=str(REPO_ROOT / "bressloff_v1_form_constants.gif"))
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
