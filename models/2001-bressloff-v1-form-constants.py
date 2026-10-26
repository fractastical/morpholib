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
6. Tamekue C, Prandi D, Chitour Y (2024).
   A mathematical model of the visual MacKay effect.
   SIAM Journal on Applied Dynamical Systems 23(3):2138-2178.
   https://doi.org/10.1137/23M1616686
   Source of the driven neural-field examples (incl. the Gaussian-only negative control
   `mackay_gaussian_kernel_negative_check`) catalogued in `driven-neural-fields-registry.json`.
7. Klüver H (1966). Mescal and Mechanisms of Hallucinations. University of Chicago Press.
   Origin of the four form-constant classes (lattice/honeycomb, cobweb, tunnel/funnel, spiral).
8. Upstream reference implementation & calibration fixtures:
   MesmerPrism/bressloff-v1-form-constants (commit 344863a1bd9e8a8e8c9cb20aced2897f3ffb2c4c, MIT License;
   derived from karacsm/V1-sim by Márton Karácsony, extended by Till Holzapfel).
9. Matthews PC, Cox SM (2000). Pattern formation with a conservation law.
   Nonlinearity 13(4):1293-1320. https://doi.org/10.1088/0951-7715/13/4/317
   Cited only for the mean-mode-coupled amplitude equations that the `near_resonant` flag
   says would be needed; not implemented here.

Scope & Claim Boundary
----------------------
* The perturbative marginal-stability dispersion curves lambda_even(q), lambda_odd(q),
  Fourier-Bessel lateral coupling harmonics, orientation eigenfunctions u_even(phi),
  u_odd(phi), cubic amplitude-equation quartic coefficients gamma(theta), and the
  contoured/non-contoured planform generators match the analytical formulation in
  `MesmerPrism/bressloff-v1-form-constants/rust-v1-sim/src/models/bressloff/planform.rs`
  and reproduce all 5 `stability_reports` and all 24 preset `runs` in
  `datasets/bressloff/paper-calibration.json` (q*, branch, selected family/pattern/scope)
  to machine precision (~1e-15). The fixture's own paper-expectation `checks` are counted
  in full (stability_reports: 11; runs: 122) and every failure is listed, not hidden.
* The retino-cortical map is the exact two-regime form of Bressloff et al. (2001,
  Section 2; 2002, Eq. 1.1): x = (alpha/eps) ln(1 + eps r/w0), y = beta r theta/(w0 + eps r),
  inverted analytically for the retinal pullback. w0 and eps are the paper's values; alpha
  and beta are the upstream implementation's choices (params.rs) and are checked against the
  paper's own stated scale (11.5 mm/deg foveal magnification, 5.75 mm/deg at 1.7 deg) and
  against the human Horton & Hoyt (1991) law M(E) = 17.3/(E + 0.75) mm/deg over 1-40 deg.
* Beyond fixture parity, the EXACT orientation-space operator W_k (Hermitian matrix on the
  harmonics e^{2 i m phi}, |m| <= 8) is diagonalised: its top eigenvalue agrees with the
  perturbative n=1 scan to |dlambda| ~ 1e-4, and its top eigenvector's dominant harmonic
  DERIVES the contoured (m = 1) / non-contoured (m = 0, Bressloff 2001 Section 4a; Ermentrout
  & Cowan 1979) regime instead of imposing it: m = 0 is critical for local inhibition
  A_loc < ~0.25 (W_0^loc = W_1^loc). Fixture presets that impose a harmonic or parity that is
  not critical are rendered at their own q* and flagged (imposed_harmonic,
  parity_imposed_subcritical).
* Pattern selection uses the FULL cubic amplitude equations (Bressloff 2001 Eqs. 5.9-5.15):
  gamma(theta) = [3 s_3 Gamma_3(theta) - 2 s_2^2 Q(theta)]/N with the firing-rate derivative
  weights s_2 = sigma_2/sigma_1, s_3 = -sigma_3/sigma_1 of a sigmoid at baseline sigma_0
  (default 1/2, the paper's inflection-point case where the quadratic terms vanish) and the
  quadratic feedback Q through the linear resolvent (lambda_c - W_K)^{-1} W_K at K = 0 and
  K = k_i +- k_j; eta = 2 s_2 Gamma_2/N gives the eta-aware hexagon/roll thresholds
  Lambda_r < Lambda_h. Roll stability is tested against perturbations in EVERY direction
  theta (5-degree grid), squares/rhombi against all external directions. At sigma_0 = 1/2
  the coefficients reduce to the bare overlap integrals of the upstream implementation,
  which is kept as `branch_selection_upstream` for parity. Findings: in the contoured
  (m = 1) regime rhombic (~45-55 deg) planforms are selected for every sigma_0 and the
  quadratic feedback does not rescue rolls or squares; in the derived non-contoured (m = 0)
  regime rolls ARE linearly stable against all theta and SELECTED (tunnel and spiral, gallery
  I-II), with squares taking over near the regime boundary. For sigma_0 != 1/2 the resolvent
  is dominated by the nearly marginal uniform (K = 0) mode of this flat-dispersion operator
  (`feedback_fraction` > 1); such cases are flagged `near_resonant` and would need a mean-mode-
  coupled amplitude equation (Matthews & Cox 2000). The cubic truncation is also subcritical
  (gamma(0) <= 0) for sigma_0 <~ 0.25; quintic terms (2001 Section 5) are not implemented.
* Rendered planforms use the branch's q* with k_x AND k_y quantised to integer cycles on the
  periodic cortical torus. Each gallery panel is labelled SELECTED / STABLE-coexisting /
  IMPOSED-unstable against the full cubic analysis with the rhombic angle free. The paper's
  perceptual scale (30-36 stripes, 2.4-3.2 mm) is compared with lambda_c = 2 pi d_hyp / q*
  for the parity presets (8.7 mm, 3.1x too long) and TESTED with a paper-scale preset
  (lateral widths / 3.3, q scan to 6): lambda_c = 2.6 mm is in range, and the fovea-to-90-deg
  ring count is 29 on the upstream alpha = 3/pi map versus 32 on the Horton & Hoyt map, so the
  residual shortfall is the map's alpha, not the wavenumber.
* Contoured planforms are rendered as the scalar envelope max_phi a(r, phi) (signed argmax
  with the exact eigenfunction u(phi); adjacent half-cycles select orthogonal orientations, so
  the envelope has period lambda/2 while the perceived contour spacing lambda is carried by the
  orientation field) and as an explicit orientation field drawn as line elements in the cortex
  and, through the tangent map phi_R = phi + theta_R, in the visual field (2001 Figs. 31-36).
* The retinal pullback uses the conformal complex-logarithm angular coordinate y = beta theta/eps
  of Bressloff 2001 Section 2 by default, under which the cortical torus closes exactly at every
  eccentricity and a periodic planform pulls back without any seam while keeping chirality. The
  exact 2002 angular coordinate y = beta r theta/(w0 + eps r) (used by earlier versions, and the
  source of their left-meridian seam) is available as `retinal_angular_map = "exact"`; its seam is
  quantified on all four half-meridians with a finite, normalised excess-discontinuity metric for
  the full-field, chiral-mirror (left hemisphere = reflected planform) and bilateral-mirror
  conventions. Chirality is measured as 1 - max(left-right, up-down reflection correlation).
  Retinal morphometrics (ring counts, spoke count, cortical roll angle and visual log-spiral
  angle) are reported per panel.
* The Rule et al. (2011) Fig. 8 overlay is drawn from an upstream fixture whose
  `calibration_claim_allowed` flag is false; this script does not solve the parametrically
  forced E-I field or its Floquet monodromy. The panel states this, separates the five
  forcing amplitudes, and reports the overlap RMS.
* The Gaussian-only negative control removes Mexican-hat lateral inhibition
  (lateral_inhibition = 0), verifying that the Turing/Euclidean band-pass wavenumber
  peak q* collapses to q -> 0 (homogeneous mode); the analogous registry control from
  Tamekue et al. (2024) is cross-referenced, not re-implemented.
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
from matplotlib.colors import to_rgb as matplotlib_colors_to_rgb
from matplotlib.patches import Patch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BRESSLOFF_DATASET_DIR = REPO_ROOT / "datasets" / "bressloff"

FIXTURE_CHECKSUMS: Dict[str, str] = {
    "paper-calibration.json": "af12b2c1aa3e38faf984166b3aa227a2aa064480a98279f14c004342ce2e7dd4",
    "rule-2011-floquet.json": "5e349c7b80185de463e1ea8656a2fe91a83523c212875af4ce46d09e9dde2a8e",
    "driven-neural-fields-registry.json": "4f07b5175a48bcb50bad4998953505a3f92faf4c6f5514d2b4643c2a67c36b10",
    "horton_hoyt_1991_magnification.json": "90d639ebb4601be59308699e128ac9b59574a9da6dbc47f0c7ad00be8d9e4a8f",
}

# Retino-cortical log-polar mapping constants.
#   w0, eps : Bressloff et al. (2001 §2; 2002 §1.1, Eq. 1.1) — "estimates of w0 = 0.087 and
#             eps = 0.051 in appropriate units can be obtained from Drasdo's data".
#   alpha, beta : NOT given numerically in either paper ("alpha and beta are constants").
#             The values below are the upstream implementation's choices,
#             MesmerPrism/bressloff-v1-form-constants rust-v1-sim/src/params.rs:5-8
#             (RETINO_ALPHA = 3/PI, RETINO_BETA = 1.589/2), kept for fixture parity.
#             Consequences, all reported by the script:
#               foveal magnification  alpha/w0            = 10.98 mm/deg (paper states 11.5 => alpha ~ 1.00)
#               far-field law         dx/dr = alpha/(w0+eps r) ~ A/(E+E2), A = alpha/eps = 18.72, E2 = w0/eps = 1.71 deg
#               cortical torus        L = 2 pi beta/eps   = 97.9 mm (circumference of the y-periodic sheet)
RETINO_EPS: float = 0.051
RETINO_W0: float = 0.087
RETINO_ALPHA: float = 3.0 / math.pi
RETINO_BETA: float = 1.589 / 2.0
RETINO_CONSTANT_PROVENANCE: Dict[str, str] = {
    "w0": "Bressloff et al. 2002 §1.1 (Drasdo 1977 ganglion-cell density)",
    "eps": "Bressloff et al. 2002 §1.1 (Drasdo 1977 ganglion-cell density)",
    "alpha": "upstream rust-v1-sim/src/params.rs:7 (3/pi); paper gives only the implied 11.5 mm/deg foveal magnification",
    "beta": "upstream rust-v1-sim/src/params.rs:8 (1.589/2); paper gives no value",
}

# Perceptual / anatomical scale statements made in Bressloff et al. (2002 §1.1, §4) that the
# rendered planforms can be checked against. These are the paper's own numbers, not fits.
PAPER_SCALE_ANCHORS: Dict[str, Any] = {
    "foveal_magnification_mm_per_deg": 11.5,          # "11.5 mm/degrees of visual angle at the fovea"
    "magnification_at_w0_over_eps_mm_per_deg": 5.75,  # "5.75 mm/deg at r_R = w0/eps = 1.7 degrees"
    "perceived_stripe_count": (30, 36),               # "about 30 to 36 stripes ... are perceived"
    "perceived_contour_count": (60, 72),              # "about 60 to 72 contours"
    "cortical_wavelength_mm": (2.4, 3.2),             # "a wavelength of about 2.4 to 3.2 mm in human V1"
    "hypercolumn_spacing_mm": (1.33, 2.0),            # 2001 abstract: "ca. 1.33-2 mm"
    "hubel_wiesel_mode_critical_wavelength_mm": 2.67,  # 2002 §4: "critical wavelength of about 2.67 mm"
    "source": "Bressloff et al. 2002 Neural Comput. 14:473, §1.1 and §4",
}


# =============================================================================
# 0. Exact retino-cortical map (Bressloff et al. 2001 §2; 2002 Eq. 1.1) and the
#    Horton & Hoyt (1991) cortical-magnification anchor
# =============================================================================
def retino_cortical_forward(
    r_deg: np.ndarray, theta: np.ndarray, angular_map: str = "exact"
) -> Tuple[np.ndarray, np.ndarray]:
    """Retina (r in deg, theta in rad) -> cortex (x_c, y_c in mm): x = (α/ε) ln(1 + ε r/w₀) and
    y = β r θ/(w₀ + ε r) (`angular_map="exact"`, Bressloff et al. 2002 §1.1) or y = β θ/ε
    (`"conformal"`: the far-field complex-logarithm limit r >> w₀/ε used for the planform pullbacks of
    Bressloff et al. 2001 §2, under which the cortical torus of circumference L = 2πβ/ε closes exactly
    at every eccentricity). The radial coordinate is the same in both."""
    r = np.asarray(r_deg, dtype=float)
    x_c = (RETINO_ALPHA / RETINO_EPS) * np.log1p(RETINO_EPS * r / RETINO_W0)
    if angular_map == "conformal":
        y_c = (RETINO_BETA / RETINO_EPS) * np.asarray(theta, dtype=float) * np.ones_like(r)
    else:
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


def compare_paper_scale_anchors(stab: Dict[str, Any]) -> Dict[str, Any]:
    """Check the map constants and a rendered planform against the scale statements in
    Bressloff et al. (2002 §1.1): foveal magnification, magnification at w0/eps, perceived stripe
    count and cortical wavelength. Returns the model values, the paper's values and the ratios;
    disagreements are reported (ledger, CSV, self-test), not corrected."""
    anc = PAPER_SCALE_ANCHORS
    r_knee = RETINO_W0 / RETINO_EPS
    fovea_model = float(cortical_magnification_model(np.array([0.0]))[0])
    knee_model = float(cortical_magnification_model(np.array([r_knee]))[0])
    alpha_implied = anc["foveal_magnification_mm_per_deg"] * RETINO_W0
    lam_c = float(stab["critical_wavelength_mm"])
    lo_w, hi_w = anc["cortical_wavelength_mm"]
    lo_s, hi_s = anc["perceived_stripe_count"]
    cycles = int(stab["rendered_cycles"])
    # Torus circumference implied by the paper's own stripe count x wavelength
    torus_paper_lo = lo_s * lo_w
    torus_paper_hi = hi_s * hi_w
    return {
        "foveal_magnification_model": fovea_model,
        "foveal_magnification_paper": float(anc["foveal_magnification_mm_per_deg"]),
        "foveal_magnification_rel_error": fovea_model / anc["foveal_magnification_mm_per_deg"] - 1.0,
        "alpha_implied_by_paper": alpha_implied,
        "alpha_used": RETINO_ALPHA,
        "knee_eccentricity_deg": r_knee,
        "magnification_at_knee_model": knee_model,
        "magnification_at_knee_paper": float(anc["magnification_at_w0_over_eps_mm_per_deg"]),
        "rendered_cycles": cycles,
        "perceived_stripe_count_paper": (lo_s, hi_s),
        "stripe_count_ratio_paper_over_model": 0.5 * (lo_s + hi_s) / max(cycles, 1),
        "critical_wavelength_model_mm": lam_c,
        "cortical_wavelength_paper_mm": (lo_w, hi_w),
        "wavelength_ratio_model_over_paper": lam_c / (0.5 * (lo_w + hi_w)),
        "q_star_needed_for_paper_wavelength": 2.0 * math.pi * 2.0 / (0.5 * (lo_w + hi_w)),
        "torus_model_mm": float(stab["torus_mm"]),
        "torus_implied_by_paper_mm": (torus_paper_lo, torus_paper_hi),
        "torus_within_paper_range": torus_paper_lo <= float(stab["torus_mm"]) <= torus_paper_hi,
        "source": anc["source"],
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
    contour_mode: str = "contoured"  # contoured | noncontoured | auto (derived from the critical local harmonic)
    parity: str = "even"             # even | odd
    n: int = 96
    m: int = 24
    wave_count: float = 12.0         # legacy upstream preset value; NOT used for rendering (see stability_scan)
    hemifield_mode: str = "full-field"  # full-field (paper Figs. 31-36 convention) | chiral-mirror | bilateral-mirror
    retinal_angular_map: str = "conformal"  # conformal (2001 complex-log pullback, torus closes) | exact (2002 map; seams)
    pattern_angle_deg: float = 45.0
    # --- Pass 4: parameters of the full cubic amplitude equations (Bressloff 2001 §5) ---------------
    # Firing-rate nonlinearity sigma(a) = 1/(1 + exp(-g (a - kappa))) expanded about the fixed point
    # a_0 with sigma(a_0) = sigma0: sigma_1 = g s(1-s), sigma_2/sigma_1 = g(1-2s)/2,
    # sigma_3/sigma_1 = g^2 (6 s^2 - 6 s + 1)/6.  Every ratio that enters the stability analysis
    # (gamma(theta)/gamma(0), the hexagon thresholds Lambda_r, Lambda_h) is independent of the gain g,
    # so the baseline sigma0 is the only new physical parameter. sigma0 = 1/2 (inflection point,
    # sigma_2 = 0) is the paper's simplifying case: the quadratic term and the quadratic feedback vanish
    # and the full coefficients reduce to the bare orientation-overlap integrals used upstream.
    sigmoid_baseline: float = 0.5
    bifurcation_distance: float = 0.1   # Lambda = (mu - mu_c)/mu_c used for hexagon/roll thresholds
    contour_envelope: str = "signed"    # signed (argmax_phi a) | abs (argmax_phi |a|, upstream)
    quantise_ky: bool = True            # quantise k_y to integer cycles on the y-periodic torus
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
    # Remaining upstream presets (rust-v1-sim/src/models/bressloff/presets.rs:642-806), included so
    # that every one of the 24 `runs[]` in paper-calibration.json can be checked for parity.
    "fig31_square_even_roll": {
        "pattern": "rings", "contour_mode": "contoured", "parity": "even",
        "lateral_spread_deg": 60.0, "wave_count": 12.0, "pattern_angle_deg": 0.0, "sharpness": 2.1,
    },
    "fig32_square_odd_roll": {
        "pattern": "rings", "contour_mode": "contoured", "parity": "odd",
        "lateral_spread_deg": 0.0, "wave_count": 12.0, "pattern_angle_deg": 0.0, "sharpness": 2.1,
    },
    "fig34_rhombic_odd_roll": {
        "pattern": "spiral", "contour_mode": "contoured", "parity": "odd",
        "lateral_spread_deg": 0.0, "wave_count": 12.0, "pattern_angle_deg": 45.0, "sharpness": 2.1,
    },
    "fig36_hex_zero_odd": {
        "pattern": "honeycomb", "contour_mode": "contoured", "parity": "odd",
        "lateral_spread_deg": 0.0, "wave_count": 12.0, "pattern_angle_deg": 60.0, "sharpness": 2.1,
    },
    "fig5_roll_cortical": {
        "pattern": "rings", "contour_mode": "contoured", "parity": "even",
        "lateral_spread_deg": 60.0, "wave_count": 12.0, "pattern_angle_deg": 0.0, "sharpness": 2.1,
    },
    "fig5_hex_cortical": {
        "pattern": "hex_pi", "contour_mode": "contoured", "parity": "even",
        "lateral_spread_deg": 60.0, "wave_count": 12.0, "pattern_angle_deg": 60.0, "sharpness": 2.1,
    },
    "fig5_honeycomb_cortical": {
        "pattern": "honeycomb", "contour_mode": "contoured", "parity": "even",
        "lateral_spread_deg": 60.0, "wave_count": 12.0, "pattern_angle_deg": 60.0, "sharpness": 2.1,
    },
    "fig5_square_cortical": {
        "pattern": "cobweb", "contour_mode": "contoured", "parity": "even",
        "lateral_spread_deg": 60.0, "wave_count": 12.0, "pattern_angle_deg": 90.0, "sharpness": 2.1,
    },
    "fig6_visual_field_planforms": {
        "pattern": "cobweb", "contour_mode": "contoured", "parity": "even",
        "lateral_spread_deg": 60.0, "wave_count": 12.0, "pattern_angle_deg": 90.0, "sharpness": 2.1,
    },
    # ---- Presets added in this repository (Pass 4); not in the upstream fixture ----------------------
    # Non-contoured regime DERIVED from the operator, not imposed: with weak local inhibition the n = 0
    # (orientation-independent, Hubel-Wiesel-free) local harmonic W_0 = 1 - A_loc exceeds W_1, so the
    # critical eigenfunction is u(phi) ~ const (Bressloff 2001 §4(a): "non-contoured" planforms, Figs.
    # 29-30; Ermentrout & Cowan 1979 regime). Here W_0 > W_1 for A_loc < 0.243 (analytic, local terms;
    # the exact operator's dominant harmonic, measured by conjugate-pair power, switches at
    # A_loc ~ 0.22 because the lateral terms shift the crossing — Pass N+1).
    # In this regime gamma(theta) is nearly flat, so rolls are linearly stable (2 gamma(theta) > gamma(0)
    # for all theta) and are SELECTED, every roll direction being degenerate: the tunnel (k_y = 0) and the
    # 45-degree spiral are two members of the selected family, not two presets fighting the equations.
    "roll_noncontoured_derived": {
        "pattern": "rings", "contour_mode": "auto", "parity": "even",
        "local_inhibition": 0.15, "lateral_spread_deg": 60.0, "wave_count": 12.0,
        "pattern_angle_deg": 0.0, "sharpness": 2.1,
    },
    "spiral_noncontoured_derived": {
        "pattern": "spiral", "contour_mode": "auto", "parity": "even",
        "local_inhibition": 0.15, "lateral_spread_deg": 60.0, "wave_count": 12.0,
        "pattern_angle_deg": 45.0, "sharpness": 2.1,
    },
    # Paper-scale preset: lateral coupling widths scaled by 1/3.3 so that q* ~ 4.8 and the cortical
    # wavelength lambda_c = 2 pi d_hyp / q* falls in Bressloff 2002 §1.1's 2.4-3.2 mm, with the q scan
    # extended to 6. This TESTS the scale discrepancy of the parity presets (lambda_c = 8.7 mm) instead
    # of only reporting it. Result: lambda_c is in range, but the ring count fovea -> 90 deg is ~29 on the
    # upstream alpha = 3/pi map (74.6 mm) versus ~32 on the Horton & Hoyt (1991) map (83 mm): the
    # remaining shortfall against the paper's 30-36 stripes is the map's alpha, not q*. Not a fixture
    # preset; no parity claim.
    "paper_scale_tunnel": {
        "pattern": "rings", "contour_mode": "contoured", "parity": "even",
        "lateral_sigma": 1.0 / 3.3, "lateral_wide_sigma": 1.5 / 3.3, "lateral_spread_deg": 60.0,
        "stability_q_min": 0.05, "stability_q_max": 6.0, "stability_samples": 160,
        "wave_count": 12.0, "pattern_angle_deg": 0.0, "sharpness": 2.3,
    },
}

REPO_ADDED_PRESETS = ("roll_noncontoured_derived", "spiral_noncontoured_derived", "paper_scale_tunnel")


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
    # Diagnostics of the two upstream truncation choices (planform.rs:745-770) reproduced here:
    max_harmonic: int = 4
    clamp_limit: float = 1.5
    max_abs_prefactor_unclamped: float = 0.0
    clamp_active: bool = False


EIGEN_MAX_HARMONIC: int = 4   # upstream truncates the O(beta) eigenfunction at m = 4 (planform.rs:745);
EIGEN_CLAMP: float = 1.5      # ... and clamps each beta*coefficient to [-1.5, 1.5] (planform.rs:750-768).
# branch_coupling_sum uses 10 harmonics for lambda(q) (planform.rs:296); the mismatch (4 vs 10) is
# upstream's and is kept for parity. Whether the clamp is ever active is reported in the ledger.


def orientation_eigen_details(params: BressloffParams, q: float) -> OrientationEigenDetails:
    """O(beta) Rayleigh-Schrödinger eigenfunction u(phi) = cos(2 phi) + beta * sum_m c_m cos(2 m phi)
    (even) or sin(2 phi) + beta * sum_m s_m sin(2 m phi) (odd), Bressloff 2001 §4.2 Eqs. 4.14-4.16.
    Truncated at m = EIGEN_MAX_HARMONIC and clamped to ±EIGEN_CLAMP exactly as upstream; the
    unclamped maximum |beta c_m| is recorded so the clamp's (in)activity can be disclosed."""
    max_harmonic = EIGEN_MAX_HARMONIC
    cos_coeffs: List[Tuple[float, float]] = []
    sin_coeffs: List[Tuple[float, float]] = []
    raw_prefactors: List[float] = []

    def _clamped(value: float) -> float:
        raw_prefactors.append(abs(value))
        return float(np.clip(value, -EIGEN_CLAMP, EIGEN_CLAMP))

    if params.parity == "even":
        cos_coeffs.append((1.0, 1.0))
        u0 = lateral_weight_coeff(params, 1, q) / safe_denominator(
            local_weight_coeff(params, 1) - local_weight_coeff(params, 0)
        )
        cos_coeffs.append((0.0, _clamped(params.eigen_beta * u0)))
        for m in range(2, max_harmonic + 1):
            coeff = (
                lateral_weight_coeff(params, m - 1, q) + lateral_weight_coeff(params, m + 1, q)
            ) / safe_denominator(local_weight_coeff(params, 1) - local_weight_coeff(params, m))
            cos_coeffs.append((float(m), _clamped(params.eigen_beta * coeff)))
    else:
        sin_coeffs.append((1.0, 1.0))
        for m in range(2, max_harmonic + 1):
            coeff = (
                lateral_weight_coeff(params, m - 1, q) - lateral_weight_coeff(params, m + 1, q)
            ) / safe_denominator(local_weight_coeff(params, 1) - local_weight_coeff(params, m))
            sin_coeffs.append((float(m), _clamped(params.eigen_beta * coeff)))
    max_raw = max(raw_prefactors) if raw_prefactors else 0.0
    return OrientationEigenDetails(
        parity=params.parity,
        beta=params.eigen_beta,
        cos_coefficients=cos_coeffs,
        sin_coefficients=sin_coeffs,
        max_harmonic=max_harmonic,
        clamp_limit=EIGEN_CLAMP,
        max_abs_prefactor_unclamped=float(max_raw),
        clamp_active=bool(max_raw > EIGEN_CLAMP),
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


# =============================================================================
# 2b. Exact orientation-space operator, its resolvent and the FULL cubic coefficients (Pass 4)
# =============================================================================
# The linearised model at wavevector k acting on an orientation profile v(phi) is
#     (W_k v)(phi) = (w_loc * v)(phi) + beta * w_lat(|k|, phi - theta_k) v(phi),
# w_lat(q, psi) = sum_n W_n^lat(q) e^{2 i n psi}. In the harmonic basis e^{2 i m phi}, m = -M..M, this
# is the Hermitian matrix  [W_k]_{m m'} = W_{|m|}^loc delta_{m m'} + beta W_{|m-m'|}^lat(q) e^{-2i(m-m')theta_k}.
# Its top eigenvalue is the growth rate lambda(q) (the perturbative `branch_growth` is its O(beta^2)
# expansion anchored on the n = 1 harmonic), its top eigenvector the critical orientation eigenfunction
# u(phi), and (lambda_c - W_K)^{-1} W_K the resolvent that carries the quadratic feedback of the
# weakly nonlinear expansion. With mu = mu_c(1 + eps^2 Lambda), a = eps a1 + eps^2 a2 + ..., and
# sigma(a) = sigma_1 a + sigma_2 a^2 + sigma_3 a^3 about the fixed point, solvability at O(eps^3) gives
# (Bressloff 2001 §5, Eqs. 5.9-5.15; derived here for the operator as implemented)
#
#     dc_i/dtau = Lambda c_i + eta conj(c_j) conj(c_k) - gamma(0)|c_i|^2 c_i - 2 sum_{j != i} gamma(theta_ij)|c_j|^2 c_i
#     gamma(theta) = [ 3 s_3 Gamma_3(theta) - 2 s_2^2 Q(theta) ] / N,      eta = 2 s_2 Gamma_2 / N,
#     Gamma_3(theta) = <u_0^2 u_theta^2>,  Gamma_2 = <u_0 u_120 u_240>,  N = <u^2>,
#     Q(theta) = <u_0^2 R_0[u_theta^2]> + <u_0 u_theta (R_{k_0+k_theta} + R_{k_0-k_theta})[u_0 u_theta]>,
#     R_K = (lambda_c - W_K)^{-1} W_K  (pseudo-inverse on the critical direction),
#     s_2 = sigma_2/sigma_1 = (1 - 2 sigma0)/2,   s_3 = -sigma_3/sigma_1 = (6 sigma0 (1 - sigma0) - 1)/6,
#
# where <.> is the mean over phi in [0, pi) and u_theta(phi) = u(phi - theta). At sigma0 = 1/2 the
# s_2 terms vanish and gamma(theta) is proportional to the bare overlap Gamma_3(theta) used by the
# upstream implementation (`amplitude_gamma3`), which is therefore the sigma_2 = 0 special case.
ORIENT_HARMONICS: int = 8
ORIENT_SAMPLES: int = 720
_PHI_GRID = (np.arange(ORIENT_SAMPLES) + 0.5) * math.pi / ORIENT_SAMPLES
_HARM_M = np.arange(-ORIENT_HARMONICS, ORIENT_HARMONICS + 1)
_HARM_E = np.exp(2j * np.outer(_HARM_M, _PHI_GRID))   # e^{2 i m phi}


def sigmoid_expansion_ratios(sigma0: float) -> Tuple[float, float]:
    """(s_2, s_3) = (sigma_2/sigma_1, -sigma_3/sigma_1) for sigma(a) = 1/(1+e^{-g(a-kappa)}) at sigma(a_0) = sigma0,
    gain g = 1 (g cancels from every stability ratio). s_3 > 0 (supercritical cubic) iff 0.211 < sigma0 < 0.789."""
    s2 = (1.0 - 2.0 * sigma0) / 2.0
    s3 = (6.0 * sigma0 * (1.0 - sigma0) - 1.0) / 6.0
    return s2, s3


def orientation_operator_matrix(params: BressloffParams, q: float, theta_k: float) -> np.ndarray:
    """Hermitian matrix of W_k on e^{2 i m phi}, m = -M..M (see the section comment)."""
    loc = np.array([local_weight_coeff(params, abs(int(m))) for m in _HARM_M])
    lat = np.array([lateral_weight_coeff(params, d, q) for d in range(0, 2 * ORIENT_HARMONICS + 1)])
    diff = _HARM_M[:, None] - _HARM_M[None, :]
    return np.diag(loc).astype(complex) + params.eigen_beta * lat[np.abs(diff)] * np.exp(-2j * diff * theta_k)


def exact_orientation_modes(params: BressloffParams, q: float) -> List[Dict[str, Any]]:
    """All eigenmodes of W_k at |k| = q (theta_k = 0), sorted by decreasing eigenvalue, each with its
    parity (even = cos-type, odd = sin-type) and dominant harmonic |m| (0 = non-contoured, 1 = contoured)."""
    vals, vecs = np.linalg.eigh(orientation_operator_matrix(params, q, 0.0))
    modes: List[Dict[str, Any]] = []
    for i in np.argsort(vals)[::-1]:
        v = vecs[:, i]
        vr = v[::-1]
        if np.allclose(v, vr, atol=1e-7):
            parity = "even"
        elif np.allclose(v, -vr, atol=1e-7):
            parity = "odd"
        else:
            parity = "mixed"
        # Dominant harmonic by conjugate-pair power (Pass N+1): a real mode
        # e^{2imphi} splits its energy across the +/-m coefficients, so a
        # single-component argmax undercounts every m >= 1 by sqrt(2) and
        # biases the m=0 vs m=1 (contoured) decision.
        _pair_power = [float(abs(v[ORIENT_HARMONICS]) ** 2)] + [
            float(abs(v[ORIENT_HARMONICS - _m]) ** 2 + abs(v[ORIENT_HARMONICS + _m]) ** 2)
            for _m in range(1, ORIENT_HARMONICS + 1)
        ]
        dom = int(np.argmax(_pair_power))
        w = v @ _HARM_E                      # real function up to a global complex phase
        j = int(np.argmax(np.abs(w)))
        u = np.real(w * np.conj(w[j]) / max(abs(w[j]), 1e-300))
        if dom == 0:
            norm = float(np.mean(u))
        elif parity == "odd":
            norm = 2.0 * float(np.mean(u * np.sin(2.0 * dom * _PHI_GRID)))
        else:
            norm = 2.0 * float(np.mean(u * np.cos(2.0 * dom * _PHI_GRID)))
        if abs(norm) > 1e-12:
            u = u / norm
        modes.append({"lambda": float(vals[i]), "parity": parity, "harmonic": dom, "u": u})
    return modes


def exact_critical_mode(params: BressloffParams, q: float, parity: Optional[str] = None,
                        harmonic: Optional[int] = None) -> Dict[str, Any]:
    """Top eigenmode of W_k at |k| = q, optionally restricted to a parity and/or dominant harmonic."""
    for mode in exact_orientation_modes(params, q):
        if parity is not None and mode["parity"] != parity:
            continue
        if harmonic is not None and mode["harmonic"] != harmonic:
            continue
        return mode
    raise RuntimeError(f"no eigenmode with parity={parity} harmonic={harmonic} at q={q}")


def exact_dispersion_scan(params: BressloffParams) -> Dict[str, Any]:
    """Exact (matrix) dispersion curves lambda(q) for the even/odd n=1 branches and the n=0 branch, and the
    overall critical mode. Returns the derived contour mode ('noncontoured' iff the n = 0 harmonic is
    critical, Bressloff 2001 §4(a)) and the mismatch against the perturbative, n=1-anchored scan."""
    samples = max(params.stability_samples, 2)
    q_min = min(params.stability_q_min, params.stability_q_max)
    q_max = max(params.stability_q_max, q_min + 1.0e-6)
    qs = np.linspace(q_min, q_max, samples)
    curves: Dict[str, List[float]] = {"even_1": [], "odd_1": [], "even_0": []}
    top: List[Tuple[float, str, int]] = []
    for q in qs:
        modes = exact_orientation_modes(params, float(q))
        best: Dict[str, float] = {}
        for mo in modes:
            key = f"{mo['parity']}_{mo['harmonic']}"
            if key in curves and key not in best:
                best[key] = mo["lambda"]
        for key in curves:
            curves[key].append(best.get(key, float("nan")))
        top.append((modes[0]["lambda"], modes[0]["parity"], modes[0]["harmonic"]))
    lam_top = np.array([t[0] for t in top])
    i_star = int(np.argmax(lam_top))
    lam_c, parity_c, harm_c = top[i_star]
    per_branch = {}
    for key, cur in curves.items():
        arr = np.array(cur)
        if np.all(np.isnan(arr)):
            continue
        j = int(np.nanargmax(arr))
        per_branch[key] = {"q_star": float(qs[j]), "lambda": float(arr[j])}
    return {
        "qs": [float(v) for v in qs],
        "curves": {k: [float(v) for v in cur] for k, cur in curves.items()},
        "critical_q": float(qs[i_star]),
        "critical_growth": float(lam_c),
        "critical_parity": parity_c,
        "critical_harmonic": int(harm_c),
        "derived_contour_mode": "noncontoured" if harm_c == 0 else "contoured",
        "per_branch": per_branch,
    }


def resolvent_apply(params: BressloffParams, lam_c: float, q_K: float, theta_K: float, g: np.ndarray) -> np.ndarray:
    """R_K[g] = (lambda_c - W_K)^{-1} W_K g on the phi grid, with the critical direction (|lambda_c - lambda_n| <
    1e-6 lambda_c) projected out — the quadratic resonance it carries is the eta term, not a feedback term."""
    vals, vecs = np.linalg.eigh(orientation_operator_matrix(params, q_K, theta_K))
    gm = (np.conj(_HARM_E) @ g) / ORIENT_SAMPLES
    coeffs = vecs.conj().T @ gm
    out = np.zeros_like(gm)
    for i in range(len(vals)):
        d = lam_c - vals[i]
        if abs(d) < 1.0e-6 * max(abs(lam_c), 1e-12):
            continue
        out += (vals[i] / d) * coeffs[i] * vecs[:, i]
    return np.real(out @ _HARM_E)


def _shift_profile(u: np.ndarray, theta: float) -> np.ndarray:
    return np.interp(np.mod(_PHI_GRID - theta, math.pi), _PHI_GRID, u, period=math.pi)


FULL_CUBIC_ANGLES_DEG = tuple(float(a) for a in range(0, 91, 5))   # theta grid for gamma(theta); 60 and 90 included


def full_cubic_coefficients(
    params: BressloffParams, q: float, parity: str, harmonic: int, sigma0: Optional[float] = None
) -> Dict[str, Any]:
    """Full cubic coefficients gamma(theta) on FULL_CUBIC_ANGLES_DEG, eta, and their ingredients (see the
    section comment) for the critical mode of the given parity/harmonic at |k| = q."""
    s0 = params.sigmoid_baseline if sigma0 is None else float(sigma0)
    s2, s3 = sigmoid_expansion_ratios(s0)
    mode = exact_critical_mode(params, q, parity=parity, harmonic=harmonic)
    lam_c, u0 = mode["lambda"], mode["u"]
    N = float(np.mean(u0 * u0))
    # Validity of the cubic truncation: the resolvent (lambda_c - W_K)^{-1} W_K is large when a mode of
    # W_K that the quadratic forcing actually excites lies close to lambda_c. With beta = 0.35 the
    # dispersion lambda(q) is flat, so the uniform K = 0 mode is nearly marginal in BOTH regimes (gap
    # 3-5 %, realised gain ||R_K g||/||g|| of 50-2000). Whether that matters depends on s_2^2: the
    # ratio of the quadratic-feedback term to the direct cubic term, max_theta |2 s_2^2 Q| / |3 s_3
    # Gamma_3|, is reported as `feedback_fraction`; when it exceeds 1 the coefficient is dominated by
    # the near-marginal mode, a mean-mode-coupled amplitude equation (Matthews & Cox 2000) would be
    # needed, and the selection is flagged `near_resonant` (unreliable). At s_2 = 0 the issue is absent.
    gaps = []
    for qK, thK in ((0.0, 0.0), (2.0 * q, 0.0)):
        vals_K = np.linalg.eigvalsh(orientation_operator_matrix(params, qK, thK))
        rel = np.abs(lam_c - vals_K) / max(abs(lam_c), 1e-12)
        rel = rel[rel >= 1.0e-6]
        gaps.append(float(np.min(rel)) if rel.size else float("inf"))
    resonance_gap = float(min(gaps))
    max_gain = 0.0

    def _gain(out: np.ndarray, g: np.ndarray) -> float:
        ng = float(np.linalg.norm(g))
        return float(np.linalg.norm(out)) / ng if ng > 1e-12 else 0.0
    thetas = np.radians(FULL_CUBIC_ANGLES_DEG)
    g3_list, q_list, gam_list = [], [], []
    for th in thetas:
        uj = _shift_profile(u0, float(th))
        g3 = float(np.mean(u0 * u0 * uj * uj))
        kp = q * math.sqrt(max(2.0 + 2.0 * math.cos(th), 0.0))
        km = q * math.sqrt(max(2.0 - 2.0 * math.cos(th), 0.0))
        r0 = resolvent_apply(params, lam_c, 0.0, 0.0, uj * uj)
        rp = resolvent_apply(params, lam_c, kp, th / 2.0, u0 * uj)
        rm = resolvent_apply(params, lam_c, km, (th - math.pi) / 2.0, u0 * uj)
        max_gain = max(max_gain, _gain(r0, uj * uj), _gain(rp, u0 * uj), _gain(rm, u0 * uj))
        qf = float(np.mean(u0 * u0 * r0))
        qf += float(np.mean(u0 * uj * (rp + rm)))
        g3_list.append(g3)
        q_list.append(qf)
        gam_list.append((3.0 * s3 * g3 - 2.0 * s2 * s2 * qf) / N)
    g2 = float(np.mean(u0 * _shift_profile(u0, 2.0 * math.pi / 3.0) * _shift_profile(u0, 4.0 * math.pi / 3.0)))
    return {
        "sigma0": s0, "s2": s2, "s3": s3, "lambda_c": lam_c, "parity": parity, "harmonic": harmonic,
        "u": u0, "N": N,
        "theta_deg": list(FULL_CUBIC_ANGLES_DEG),
        "gamma3_bare": g3_list, "quadratic_feedback": q_list, "gamma": gam_list,
        "gamma2_bare": g2, "eta": 2.0 * s2 * g2 / N,
        "resonance_gap": resonance_gap, "resonance_gap_k0": gaps[0], "resonance_gap_2k": gaps[1],
        "feedback_gain": max_gain,
        "feedback_fraction": float(max(
            (abs(2.0 * s2 * s2 * qf) / max(abs(3.0 * s3 * g3), 1e-12) for g3, qf in zip(g3_list, q_list)), default=0.0
        )),
        "near_resonant": bool(abs(s2) > 1e-12 and max(
            (abs(2.0 * s2 * s2 * qf) / max(abs(3.0 * s3 * g3), 1e-12) for g3, qf in zip(g3_list, q_list)), default=0.0
        ) > 1.0),
    }


def _gamma_at(cubic: Dict[str, Any], theta_deg: float) -> float:
    return float(np.interp(theta_deg, cubic["theta_deg"], cubic["gamma"]))


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
    """Stationary branch of the cubic amplitude equations and its ranking score.

    For N equal-amplitude modes |c_i| = A the cubic amplitude equations (Bressloff 2001 Eqs.
    5.9-5.13; the rhombic/square form dc_i/dt = c_i[lambda - gamma_3(0)|c_i|^2 - 2 gamma_3(theta)|c_j|^2]
    and the hexagonal form with the quadratic eta c_j* c_k* term) are the gradient flow of

        F = -sum_i [ lambda |c_i|^2 / 2 - D |c_i|^4 / 4 ] - eta * (c_1 c_2 c_3 + c.c.) / 3,
        D = gamma_3(0) + 2 (N - 1) gamma_3(theta)   (N = 1 roll, 2 square/rhombic, 3 hexagonal)

    so the branch amplitude is the positive root of lambda - D A^2 + |eta| A = 0 and `score` is
    -F evaluated on that branch (= lambda^2 / (4 gamma_3(0)) for rolls). Branches are ranked first
    by linear stability (the exact Bressloff 2001 §5 conditions, computed in `branch_selection_for`)
    and only then by this Lyapunov value, which is the standard but heuristic tie-break between
    coexisting stable branches; it is not a claim about basins of attraction."""
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


def branch_selection_upstream(params: BressloffParams, q: float, growth: float) -> Dict[str, Any]:
    """Upstream (rust-v1-sim) cubic-order selection, kept byte-for-byte for fixture parity: bare overlap
    integrals of the O(beta) perturbative eigenfunction (sigma_2 feedback omitted), `lambda` set to the
    eigenvalue itself, roll stability tested only at theta in {preset angle, 60, 90}, and the imposed
    parity's eigenfunction evaluated at the critical branch's q. `branch_selection_full` is the
    analysis used for every badge, CSV row and sweep; this one is reported as `branch_selection_upstream`."""
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

    # Unrestricted selection: the rhombic candidate above uses the preset's pattern_angle_deg, so
    # "global" selection still depends on a rendering parameter. Here the rhombic angle is chosen
    # where gamma_3(theta) is minimal (the most stable rhombus, Bressloff 2001 §5.2) and the
    # winner is recomputed; this is the parameter-free answer to "what does the cubic truncation
    # select for this coupling?". For near-cos(2 phi) eigenfunctions gamma_3 ~ 1/4 + cos(4 theta)/8,
    # so theta_opt ~ 45 deg, rolls (need 2 gamma_3(theta) > gamma_3(0) for all theta) are never
    # stable and squares (need 2 gamma_3(90) < gamma_3(0)) are never stable.
    theta_grid = np.linspace(math.pi / 12.0, 5.0 * math.pi / 12.0, 61)
    g3_grid = [amplitude_gamma3(float(t), eigen) for t in theta_grid]
    i_opt = int(np.argmin(g3_grid))
    theta_opt = float(theta_grid[i_opt])
    gamma_rhombic_opt = float(g3_grid[i_opt])
    rhombic_opt = _make_branch_candidate(
        "rhombic", "rhombic", 2, theta_opt, gamma0, gamma_rhombic_opt, lam,
        gamma0 + 2.0 * gamma_rhombic_opt, 0.0, gamma_rhombic_opt > 0.0 and 2.0 * gamma_rhombic_opt < gamma0,
    )
    unrestricted = sorted([roll, square, rhombic_opt, hex_cand], key=lambda c: (c["stable"], c["score"]), reverse=True)[0]

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
        "rhombic_theta_optimal_deg": math.degrees(theta_opt),
        "gamma_rhombic_optimal": gamma_rhombic_opt,
        "unrestricted_selected_family": unrestricted["family"],
        "unrestricted_selected_pattern": unrestricted["pattern"],
        "unrestricted_selected_stable": unrestricted["stable"],
        "eigen_clamp_active": eigen.clamp_active,
        "eigen_max_abs_prefactor": eigen.max_abs_prefactor_unclamped,
        "candidates": candidates,
    }


def _gamma_folded(cubic: Dict[str, Any], theta_deg: float) -> float:
    """gamma(theta) for any angle, using gamma(theta) = gamma(-theta) = gamma(180 - theta)."""
    t = abs(theta_deg) % 180.0
    return _gamma_at(cubic, 180.0 - t if t > 90.0 else t)


def branch_selection_full(params: BressloffParams, exact: Dict[str, Any]) -> Dict[str, Any]:
    """Pattern selection from the FULL cubic amplitude equations (section 2b) for the branch the figure
    renders: parity from the preset (or the critical parity for `pattern = auto`), orientation harmonic
    derived from the operator (`contour_mode = auto`) or imposed by the preset and flagged when it is not
    the critical one. Every branch is evaluated at ITS OWN q* (not the global critical q), and roll
    stability is tested against perturbations in every direction theta in (0, 90] (5-degree grid), not
    only the three lattice angles. Hexagons and rolls use the eta-aware thresholds Lambda_r, Lambda_h at
    the configured bifurcation distance Lambda. Returns the same key set as `branch_selection_upstream`
    plus the full-model diagnostics (cubic_mode, sigma0, eta, lambda_r, lambda_h, roll_min_margin,
    gamma_table, critical_harmonic, derived_contour_mode, imposed_harmonic, parity_imposed_subcritical)."""
    Lam = float(params.bifurcation_distance)
    parity = exact["critical_parity"] if params.pattern == "auto" else params.parity
    derived_harm = int(exact["critical_harmonic"])
    if params.contour_mode == "auto":
        harmonic = derived_harm
    else:
        harmonic = 0 if params.contour_mode == "noncontoured" else 1
    if harmonic == 0:
        parity = "even"   # the m = 0 harmonic has no odd (sin) partner
    key = f"{parity}_{harmonic}"
    branch = exact["per_branch"].get(key)
    if branch is None:
        q_mode, lam_mode = exact["critical_q"], exact["critical_growth"]
    else:
        q_mode, lam_mode = branch["q_star"], branch["lambda"]
    parity_imposed_subcritical = bool(lam_mode < exact["critical_growth"] - 1.0e-9)
    cubic = full_cubic_coefficients(params, q_mode, parity, harmonic)
    eta = float(cubic["eta"]) if parity == "even" else 0.0
    thetas = cubic["theta_deg"]
    gam = cubic["gamma"]
    gamma0 = float(gam[0])
    gamma_square = _gamma_at(cubic, 90.0)
    gamma_hex = _gamma_at(cubic, 60.0)
    rhombic_deg = float(np.clip(params.pattern_angle_deg, 15.0, 75.0))
    gamma_rhombic = _gamma_folded(cubic, rhombic_deg)
    interior = [(t, g) for t, g in zip(thetas, gam) if t > 0.0]
    roll_margins = [(2.0 * g - gamma0) / abs(gamma0) if abs(gamma0) > 1e-12 else float("nan") for _, g in interior]
    roll_min_margin = float(min(roll_margins)) if roll_margins else float("nan")
    d_h = 2.0 * gamma_hex - gamma0
    lambda_r = (eta * eta * gamma0 / (d_h * d_h)) if abs(d_h) > 1e-12 else float("inf")
    lambda_h = (2.0 * eta * eta * (gamma0 + gamma_hex) / (d_h * d_h)) if abs(d_h) > 1e-12 else float("inf")
    eta_active = abs(eta) > 1.0e-9

    roll_stable = gamma0 > 0.0 and roll_min_margin > 0.0 and (not eta_active or Lam > lambda_r)

    def _lattice_external_ok(angles: List[float], denom: float) -> bool:
        # Perturbation of an N-mode lattice by a mode at direction theta grows at rate
        # Lambda - 2 A^2 sum_i gamma(theta - alpha_i); with A^2 = Lambda/denom this is stable iff
        # 2 sum_i gamma(theta - alpha_i) > denom for every theta (eta = 0 case).
        if denom <= 0.0:
            return False
        for t in np.arange(0.0, 180.0, 5.0):
            if 2.0 * sum(_gamma_folded(cubic, t - a) for a in angles) <= denom:
                return False
        return True

    sq_denom = gamma0 + 2.0 * gamma_square
    square_stable = gamma_square > 0.0 and 2.0 * gamma_square < gamma0 and _lattice_external_ok([0.0, 90.0], sq_denom)
    rh_denom = gamma0 + 2.0 * gamma_rhombic
    rhombic_stable = gamma_rhombic > 0.0 and 2.0 * gamma_rhombic < gamma0 and _lattice_external_ok([0.0, rhombic_deg], rh_denom)
    hex_denom = gamma0 + 4.0 * gamma_hex
    if eta_active:
        hex_stable = hex_denom > 0.0 and (d_h <= 0.0 or Lam < lambda_h)
    else:
        hex_stable = hex_denom > 0.0 and d_h < 0.0 and _lattice_external_ok([0.0, 60.0, 120.0], hex_denom)
    hex_pattern = "hex_pi" if eta < 0.0 else "honeycomb"
    hex_pattern_degenerate = not eta_active

    roll = _make_branch_candidate("roll", "spiral", 1, 0.0, gamma0, 0.0, Lam, gamma0, 0.0, roll_stable)
    square = _make_branch_candidate("square", "cobweb", 2, math.pi / 2.0, gamma0, gamma_square, Lam, sq_denom, 0.0, square_stable)
    rhombic = _make_branch_candidate("rhombic", "rhombic", 2, math.radians(rhombic_deg), gamma0, gamma_rhombic, Lam, rh_denom, 0.0, rhombic_stable)
    hex_cand = _make_branch_candidate("hexagonal", hex_pattern, 3, math.radians(60.0), gamma0, gamma_hex, Lam, hex_denom, eta, hex_stable)

    # Optimal rhombus: the angle in [15, 75] deg minimising gamma(theta).
    rh_grid = [(t, g) for t, g in zip(thetas, gam) if 15.0 <= t <= 75.0]
    theta_opt_deg, gamma_rhombic_opt = min(rh_grid, key=lambda tg: tg[1])
    rh_opt_denom = gamma0 + 2.0 * gamma_rhombic_opt
    rhombic_opt = _make_branch_candidate(
        "rhombic", "rhombic", 2, math.radians(theta_opt_deg), gamma0, gamma_rhombic_opt, Lam, rh_opt_denom, 0.0,
        gamma_rhombic_opt > 0.0 and 2.0 * gamma_rhombic_opt < gamma0 and _lattice_external_ok([0.0, theta_opt_deg], rh_opt_denom),
    )

    candidates = [roll, square, rhombic, hex_cand]
    candidates.sort(key=lambda c: (c["stable"], c["score"]), reverse=True)
    global_selected = candidates[0]
    unrestricted = sorted([roll, square, rhombic_opt, hex_cand], key=lambda c: (c["stable"], c["score"]), reverse=True)[0]
    target_lattice = _branch_target_lattice(params.pattern)
    if target_lattice == "global":
        selected, selected_scope, selected_lattice_stable = global_selected, "global", global_selected["stable"]
    else:
        scoped = [c for c in candidates if (c["family"] == target_lattice or c["family"] == "roll")]
        scoped.sort(key=lambda c: (c["stable"], c["score"]), reverse=True)
        selected = scoped[0] if scoped else global_selected
        selected_scope = "lattice-local" if scoped else "global-fallback"
        selected_lattice_stable = selected["stable"]
    any_stable = any(c["stable"] for c in [roll, square, rhombic_opt, hex_cand])

    return {
        "cubic_mode": "full",
        "lambda": Lam,
        "lambda_eigen": float(lam_mode),
        "q_mode": float(q_mode),
        "sigma0": cubic["sigma0"], "s2": cubic["s2"], "s3": cubic["s3"],
        "N": float(cubic["N"]),
        "gamma0": gamma0,
        "gamma_square": gamma_square,
        "gamma_rhombic": gamma_rhombic,
        "gamma_hex": gamma_hex,
        "eta_hex": eta,
        "eta": eta,
        "lambda_r": float(lambda_r),
        "lambda_h": float(lambda_h),
        "roll_min_margin": roll_min_margin,
        "gamma_table": {"theta_deg": list(thetas), "gamma": [float(g) for g in gam],
                        "gamma3_bare": [float(g) for g in cubic["gamma3_bare"]],
                        "quadratic_feedback": [float(g) for g in cubic["quadratic_feedback"]]},
        "critical_harmonic": derived_harm,
        "derived_contour_mode": exact["derived_contour_mode"],
        "rendered_harmonic": harmonic,
        "rendered_contour_mode": "noncontoured" if harmonic == 0 else "contoured",
        "imposed_harmonic": bool(harmonic != derived_harm),
        "rendered_parity": parity,
        "parity_imposed_subcritical": parity_imposed_subcritical,
        "subcritical_cubic": bool(gamma0 <= 0.0),
        "resonance_gap": float(cubic["resonance_gap"]),
        "feedback_gain": float(cubic["feedback_gain"]),
        "feedback_fraction": float(cubic["feedback_fraction"]),
        "near_resonant": bool(cubic["near_resonant"]),
        "cubic_truncation_reliable": bool(gamma0 > 0.0 and not cubic["near_resonant"]),
        "any_stable_branch": bool(any_stable),
        "target_lattice": target_lattice,
        "selected_scope": selected_scope,
        "selected_family": selected["family"],
        "selected_pattern": selected["pattern"],
        "selected_lattice_stable": selected_lattice_stable,
        "global_selected_family": global_selected["family"],
        "global_selected_pattern": global_selected["pattern"],
        "global_selected_stable": global_selected["stable"],
        "roll_stable": bool(roll_stable),
        "square_stable": bool(square["stable"]),
        "rhombic_stable": bool(rhombic["stable"]),
        "hex_stable": bool(hex_cand["stable"]),
        "hex_pattern_degenerate": hex_pattern_degenerate,
        "rhombic_theta_optimal_deg": float(theta_opt_deg),
        "gamma_rhombic_optimal": float(gamma_rhombic_opt),
        "unrestricted_selected_family": unrestricted["family"],
        "unrestricted_selected_pattern": unrestricted["pattern"],
        "unrestricted_selected_stable": unrestricted["stable"],
        "eigen_clamp_active": False,
        "eigen_max_abs_prefactor": float("nan"),
        "u_profile": [float(v) for v in cubic["u"]],
        "candidates": candidates,
    }


LATTICE_CONSISTENCY_LABELS: Dict[str, Tuple[str, str]] = {
    # state: (badge text template, colour)
    "selected": ("SELECTED by cubic eqs.", "#1e8449"),
    "stable-coexisting": ("STABLE, coexists (winner: {winner})", "#b9770e"),
    "imposed-unstable": ("IMPOSED: unstable (cubic → {winner})", "#b03a2e"),
}


def lattice_consistency_for(rendered_family: str, bsel: Dict[str, Any]) -> str:
    """Classify a rendered lattice against the unrestricted cubic amplitude-equation analysis:
    'selected' (it is the winning branch), 'stable-coexisting' (a linearly stable branch that loses the
    Lyapunov tie-break to the winner) or 'imposed-unstable' (not a stable branch at cubic order)."""
    if rendered_family == bsel["unrestricted_selected_family"]:
        return "selected"
    stable = {
        "roll": bool(bsel["roll_stable"]),
        "square": bool(bsel["square_stable"]),
        "rhombic": bool(bsel["rhombic_stable"]),
        "hexagonal": bool(bsel["hex_stable"]),
    }.get(rendered_family, False)
    return "stable-coexisting" if stable else "imposed-unstable"


def lattice_badge(consistency: str, bsel: Dict[str, Any]) -> Tuple[str, str]:
    template, colour = LATTICE_CONSISTENCY_LABELS[consistency]
    return template.format(winner=bsel["unrestricted_selected_family"]), colour


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
    bsel_upstream = branch_selection_upstream(eff_params, critical_q, critical_growth)

    # Exact (matrix) dispersion and the full cubic selection for the rendered branch. The rendered
    # branch's own q* is used for the wavenumber (an imposed parity/harmonic is evaluated where IT is
    # marginal, and flagged as subcritical relative to the true critical mode).
    exact = exact_dispersion_scan(eff_params)
    bsel = branch_selection_full(eff_params, exact)
    eff_params.parity = bsel["rendered_parity"]
    eff_params.contour_mode = bsel["rendered_contour_mode"]
    q_render = float(bsel["q_mode"])
    # Rendered wavenumber is derived from the branch's q* (|k| = q*/d_hyp), then quantised to an
    # integer number of cycles on the periodic cortical torus of circumference L = n * cell so that
    # the FFT ring sits exactly on k*. The preset `wave_count` is kept only as a legacy reference
    # value and no longer drives the figure.
    torus_mm = float(eff_params.n) * cell_mm_for(eff_params)
    wave_number_qstar = q_render / max(eff_params.hypercolumn_mm, 1.0e-9)
    cycles_unquantised = wave_number_qstar * torus_mm / (2.0 * math.pi)
    rendered_cycles = max(1, int(round(cycles_unquantised)))
    wave_number = 2.0 * math.pi * float(rendered_cycles) / torus_mm
    q_eval = wave_number * eff_params.hypercolumn_mm
    pert_key = f"{critical_branch}_1"
    pert_vs_exact = exact["per_branch"].get(pert_key, {})
    return {
        "qs": [float(v) for v in qs],
        "even_growth": even_curve,
        "odd_growth": odd_curve,
        "critical_q": critical_q,                      # perturbative n=1-anchored scan (upstream parity)
        "critical_branch": critical_branch,
        "critical_growth": critical_growth,
        "critical_q_exact": exact["critical_q"],       # exact matrix operator, all harmonics
        "critical_growth_exact": exact["critical_growth"],
        "critical_parity_exact": exact["critical_parity"],
        "critical_harmonic_exact": exact["critical_harmonic"],
        "derived_contour_mode": exact["derived_contour_mode"],
        "perturbative_vs_exact_dq": float(critical_q - pert_vs_exact.get("q_star", float("nan"))),
        "perturbative_vs_exact_dlambda": float(critical_growth - pert_vs_exact.get("lambda", float("nan"))),
        "exact_curves": exact["curves"],
        "exact_per_branch": exact["per_branch"],
        "rendered_parity": eff_params.parity,
        "rendered_contour_mode": eff_params.contour_mode,
        "rendered_q": q_render,
        "wave_number": wave_number,
        "wave_number_source": "rendered branch q* / hypercolumn_mm (exact operator), quantised to integer torus cycles",
        "wave_number_qstar_unquantised": wave_number_qstar,
        "rendered_cycles": rendered_cycles,
        "cycles_unquantised": cycles_unquantised,
        "critical_wavelength_mm": 2.0 * math.pi / max(wave_number_qstar, 1.0e-12),
        "torus_mm": torus_mm,
        "preset_wave_count_legacy": float(eff_params.wave_count),
        "q_eval": q_eval,
        "branch_selection": bsel,
        "branch_selection_upstream": bsel_upstream,
    }


def planform_modes(params: BressloffParams, pattern: str) -> List[Tuple[float, float, float, float]]:
    """Return list of (normal_angle, phase_scale, phase_offset, amplitude) for a planform family.

    Stationary planforms of the amplitude equations have time-independent phases; `phase_scale`
    multiplies the GIF's animation phase only and is 1.0 for every mode (the upstream per-mode
    multipliers 0.35 / -0.25 made the square and rhombic lattices drift as two independent rolls,
    which is not a solution of the equations). The H_pi hexagon (Bressloff 2001 §5(c)) is the
    equal-amplitude solution with phases summing to pi, taken here as (0, pi, 0): the sign of the
    second mode is flipped; its amplitudes and phase scales are otherwise identical to H_0."""
    ang = math.radians(params.pattern_angle_deg)
    if pattern in ("auto", "cobweb"):
        return [(0.0, 1.0, 0.0, 1.0), (math.pi / 2.0, 1.0, 0.0, 1.0)]
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
        return [(0.0, 1.0, 0.0, 1.0), (ang, 1.0, 0.0, 1.0)]
    if pattern == "hex_pi":
        return [
            (0.0, 1.0, 0.0, 1.0),
            (2.0 * math.pi / 3.0, 1.0, math.pi, 1.0),
            (-2.0 * math.pi / 3.0, 1.0, 0.0, 1.0),
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
    uniform_phase: bool = False,
    envelope_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Evaluate a V1 cortical planform on the torus [0, L] x [-L/2, L/2] and its exact retinal pullback.

    `envelope_only=True` renders the scalar mode superposition without the orientation argmax (used by
    the GIF and by the linear FFT check) while leaving the regime, parity and wavenumber exactly as the
    stability analysis derived them.

    L = 2*pi*beta/eps is the far-field circumference of the retino-cortical map; x_c = 0 is the fovea.
    The retinal panel covers the central disc |r| <= `retinal_max_deg` using the exact inverse of
    x = (alpha/eps) ln(1 + eps r/w0), y = beta r theta/(w0 + eps r).

    `params.retinal_angular_map` selects the angular coordinate of the map:
      * "conformal" (default): y = beta theta / eps, the complex-logarithm limit used for the planform
        pullbacks of Bressloff et al. 2001 §2. The cortical torus (circumference L = 2 pi beta/eps)
        then closes exactly at every eccentricity, so a torus-periodic planform pulls back WITHOUT any
        seam and the tangent map phi_R = phi + theta_R is exact. This is the self-consistent choice
        for a planform defined on the periodic torus.
      * "exact": y = beta r theta/(w0 + eps r) (Bressloff et al. 2002 §1.1). Angle is compressed
        near the fovea, so y(r, +-pi) < +-L/2 and a periodic planform cannot close: a seam with phase
        jump k_y (L - 2 y(r, pi)) appears wherever the planform is not even in y. The seam of every
        gallery preset under the exact map is quantified and reported; it was the source of the
        "left-meridian seam" of earlier versions, which used the exact map for the pullback.
    `params.hemifield_mode` selects the hemifield convention (matters for the exact map):
      * "full-field" (default; Bressloff 2001 Figs. 31-36): theta in (-pi, pi] on one periodic sheet.
      * "chiral-mirror": each hemisphere is a cortex with theta in [-pi/2, pi/2] from the horizontal
        meridian; the left hemisphere carries the REFLECTED planform (k_y -> -k_y, phi -> -phi, phase
        offset n_y pi). Reflection of the pattern composed with reflection of the hemifield continues
        the right-hemifield pattern, so chirality is preserved, the left horizontal meridian is exactly
        continuous, and under the exact map the residual mismatch moves to the two vertical
        half-meridians with half the phase slip (it vanishes in the far field and under the conformal map).
      * "bilateral-mirror": both hemispheres carry the SAME mode. Seamless everywhere but reflects
        the left hemifield, so spirals become mirror-symmetric chevrons.
    Every seam is quantified (normalised, per half-meridian) by `retinal_pullback_diagnostics`.

    Wavevectors: |k| = q*/d_hyp quantised to integer x-cycles on the torus (stability_scan); with
    `params.quantise_ky` the y-component of every oblique mode is also quantised to an integer
    number n_y of cycles on the y-periodic torus at fixed |k| (the normal angle moves by up to
    ~5.4 degrees for 60-degree hex modes at 11 cycles; < 1 degree only for near-axis modes),
    so the cortical pattern is exactly periodic in y and the hemifield phase offsets
    n_y pi are well defined.

    For contoured planforms the orientation field phi*(x) = argmax_phi a(x, phi) (`contour_envelope
    = "signed"`; "abs" reproduces the upstream argmax of |a|) uses the EXACT critical eigenfunction
    u(phi) of the orientation operator and is returned in `stab["orientation_cortical"]` and,
    through the tangent map phi_R = phi + theta_R (Bressloff 2002 §1.1), in
    `stab["orientation_retinal"]`. Note that the envelope of a contoured roll max_phi cos(k.x) u(phi)
    has period lambda/2 whenever u takes both signs (both half-cycles select some orientation): the
    PERCEIVED contour spacing is lambda, carried by the orientation field, while the scalar envelope
    ring spacing is lambda/2. `retinal_morphometrics` reports both.
    `uniform_phase=True` makes every mode advance with the same phase (GIF).
    """
    stab = stability_scan(params)
    eff_params = BressloffParams(**{**params.__dict__})
    eff_params.parity = stab["rendered_parity"]
    eff_params.contour_mode = stab["rendered_contour_mode"]
    if eff_params.pattern == "auto":
        eff_pattern = stab["branch_selection"]["selected_pattern"]
    else:
        eff_pattern = eff_params.pattern

    cell_mm = cell_mm_for(eff_params)
    extent = float(eff_params.n) * cell_mm
    half = 0.5 * extent
    wave_number = stab["wave_number"]
    u_profile = np.asarray(stab["branch_selection"]["u_profile"], dtype=float)
    base_modes = planform_modes(eff_params, eff_pattern)
    if uniform_phase:
        base_modes = [(a, 1.0, off, amp) for a, _ps, off, amp in base_modes]
    contoured = eff_params.contour_mode != "noncontoured" and not envelope_only

    # Wavevectors, with k_y quantised to integer cycles on the y-periodic torus.
    modes: List[Dict[str, float]] = []
    for norm_ang, p_scale, p_off, amp in base_modes:
        kx = wave_number * math.cos(norm_ang)
        ky = wave_number * math.sin(norm_ang)
        n_y = ky * extent / (2.0 * math.pi)
        if eff_params.quantise_ky:
            n_y = float(round(n_y))
            ky_q = 2.0 * math.pi * n_y / extent
            if abs(ky_q) > wave_number:
                ky_q = math.copysign(wave_number, ky_q)
            kx = math.copysign(math.sqrt(max(wave_number * wave_number - ky_q * ky_q, 0.0)), kx if abs(kx) > 1e-12 else 1.0)
            ky = ky_q
        modes.append({
            "kx": kx, "ky": ky, "n_y": n_y, "angle_rad": math.atan2(ky, kx), "angle_requested_rad": norm_ang,
            "p_scale": p_scale, "p_off": p_off, "amp": amp,
        })

    # 1) Cortical torus grid (x_c in [0, L] with the fovea at x_c = 0, y_c in [-L/2, L/2]) in mm
    xs = (np.arange(eff_params.n, dtype=float) + 0.5) * cell_mm
    ys = (np.arange(eff_params.n, dtype=float) + 0.5) * cell_mm - half
    xg, yg = np.meshgrid(xs, ys)
    signed_env = eff_params.contour_envelope != "abs"

    def _u_of(delta: float) -> float:
        return float(np.interp(math.fmod(delta, math.pi) % math.pi, _PHI_GRID, u_profile, period=math.pi))

    def _eval_xy(x_arr: np.ndarray, y_arr: np.ndarray, reflected: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        sgn = -1.0 if reflected else 1.0
        if not contoured:
            val = np.zeros_like(x_arr, dtype=float)
            for mo in modes:
                arg = mo["kx"] * x_arr + sgn * mo["ky"] * y_arr + phase * mo["p_scale"] + mo["p_off"]
                if reflected:
                    arg = arg + mo["n_y"] * math.pi
                val += mo["amp"] * np.cos(arg)
            return np.tanh(val * eff_params.sharpness), None

        samples = max(eff_params.m, 8)
        best = np.full_like(x_arr, -np.inf, dtype=float)
        best_phi = np.zeros_like(x_arr, dtype=float)
        spatials = []
        for mo in modes:
            arg = mo["kx"] * x_arr + sgn * mo["ky"] * y_arr + phase * mo["p_scale"] + mo["p_off"]
            if reflected:
                arg = arg + mo["n_y"] * math.pi
            spatials.append(mo["amp"] * np.cos(arg))
        for k in range(samples):
            phi = math.pi * float(k) / float(samples)
            val = np.zeros_like(x_arr, dtype=float)
            for mo, spatial in zip(modes, spatials):
                tangent_center = sgn * mo["angle_rad"] + math.pi / 2.0
                val += spatial * _u_of(phi - tangent_center)
            mask_better = (val > best) if signed_env else (np.abs(val) > np.abs(np.where(np.isfinite(best), best, 0.0)))
            best = np.where(mask_better, val, best)
            best_phi = np.where(mask_better, phi, best_phi)
        return np.tanh(best * eff_params.sharpness), best_phi

    cortical_field, orientation_cortical = _eval_xy(xg, yg)

    # 2) Retinal visual-field pullback through the exact retino-cortical map.
    r_max = float(retinal_max_deg)
    u_lin = np.linspace(-r_max, r_max, retinal_res)
    ug, vg = np.meshgrid(u_lin, u_lin)
    r_ret = np.hypot(ug, vg)
    theta_full = np.arctan2(vg, ug)
    left = ug < 0.0
    if eff_params.hemifield_mode in ("bilateral-mirror", "chiral-mirror"):
        theta_map = np.arctan2(vg, np.abs(ug))
    else:
        theta_map = theta_full
    x_cort, y_cort = retino_cortical_forward(r_ret, theta_map, eff_params.retinal_angular_map)
    retinal_field, orientation_retinal_cort_frame = _eval_xy(x_cort, y_cort)
    if eff_params.hemifield_mode == "chiral-mirror":
        refl_field, refl_orient = _eval_xy(x_cort, y_cort, reflected=True)
        retinal_field = np.where(left, refl_field, retinal_field)
        if orientation_retinal_cort_frame is not None and refl_orient is not None:
            orientation_retinal_cort_frame = np.where(left, refl_orient, orientation_retinal_cort_frame)
    disc = r_ret <= r_max
    retinal_field = np.where(disc, retinal_field, np.nan)
    orientation_retinal: Optional[np.ndarray] = None
    if orientation_retinal_cort_frame is not None:
        # Tangent map of the complex logarithm: phi_R = phi + theta_R (Bressloff 2002 §1.1). In both
        # mirror conventions the left hemifield is a reflection, so its orientation is reflected too.
        if eff_params.hemifield_mode in ("bilateral-mirror", "chiral-mirror"):
            phi_c = np.where(left, math.pi - orientation_retinal_cort_frame, orientation_retinal_cort_frame)
        else:
            phi_c = orientation_retinal_cort_frame
        orientation_retinal = np.where(disc, np.mod(phi_c + theta_full, math.pi), np.nan)

    x_cort_max = float(retino_cortical_forward(np.array([r_max]), np.array([0.0]))[0][0])
    stab = dict(stab)
    stab["rendered_pattern"] = eff_pattern
    stab["rendered_modes"] = modes
    stab["retinal_max_deg"] = r_max
    stab["cortical_x_for_retinal_max_mm"] = x_cort_max
    stab["cortical_torus_mm"] = extent
    stab["hemifield_mode"] = eff_params.hemifield_mode
    stab["retinal_angular_map"] = eff_params.retinal_angular_map
    stab["contour_envelope"] = "signed" if signed_env else "abs"
    stab["orientation_cortical"] = orientation_cortical
    stab["orientation_retinal"] = orientation_retinal
    stab["retinal_grid_deg"] = u_lin
    return cortical_field, retinal_field, stab


def retinal_pullback_diagnostics(retinal_field: np.ndarray, r_min_frac: float = 0.25) -> Dict[str, float]:
    """Quantify a retinal pullback on the square grid returned by `render_cortical_and_retinal_planform`.

      mirror_symmetry_lr / _ud : Pearson correlation between the field and its left-right / up-down
                        reflection (1.0 = exactly symmetric).
      mirror_symmetry : max of the two.
      chirality       : 1 - max(|lr|, |ud|) in [0, 1]; a handed pattern (log-spiral) fails both
                        reflections and scores well above 0; rings, cobwebs and honeycombs (achiral,
                        possibly shifted by one reflection) score ~0; bilateral-mirror forces lr = 1.
      seam_h_left, seam_h_right, seam_v_top, seam_v_bottom :
                        EXCESS discontinuity across each half-meridian: mean |Δf| between the two
                        pixel rows (columns) straddling the meridian minus the mean |Δf| between
                        neighbouring row (column) pairs on either side of it, divided by the mean
                        nearest-neighbour |Δf| of the whole annulus. 0 means the meridian is no more
                        discontinuous than the field's own texture; values of order 1 mean a jump
                        of the size of the field's typical pixel-to-pixel contrast. Always finite
                        for a non-constant field (unlike the previous ratio-type metric, which was
                        undefined for y-independent planforms and ignored the vertical meridians).
      seam_contrast   : max of the four (the scalar used by the audit measurements).
    The annulus r >= r_min_frac * r_max is used so the singular foveal cells do not dominate.
    """
    f = np.asarray(retinal_field, dtype=float)
    h, w = f.shape
    mask = np.isfinite(f)

    def _corr(g: np.ndarray) -> float:
        both = mask & np.isfinite(g)
        a = f[both] - np.mean(f[both])
        b = g[both] - np.mean(g[both])
        denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
        return float(np.sum(a * b) / denom) if denom > 0 else float("nan")

    # Left-right AND up-down reflections: a pattern that is merely shifted by one reflection (e.g. a
    # square lattice with an odd number of y-cycles, whose y-mode changes sign under theta -> pi - theta)
    # is still achiral if it is symmetric under the other; only a handed pattern (log-spiral) fails both.
    mirror_lr = _corr(f[:, ::-1])
    mirror_ud = _corr(f[::-1, :])
    mirror = max(mirror_lr, mirror_ud) if (math.isfinite(mirror_lr) and math.isfinite(mirror_ud)) else float("nan")

    cy, cx = h // 2, w // 2
    r_min = int(round(r_min_frac * cx))
    yy, xx = np.indices(f.shape)
    rr = np.hypot(yy - cy + 0.5, xx - cx + 0.5)
    annulus = mask & (rr >= r_min)
    dy = np.abs(np.diff(f, axis=0))
    dx = np.abs(np.diff(f, axis=1))
    ann_y = annulus[1:, :] & annulus[:-1, :]
    ann_x = annulus[:, 1:] & annulus[:, :-1]
    grad_ref = float(np.mean(np.concatenate([dy[ann_y], dx[ann_x]]))) if (np.any(ann_y) or np.any(ann_x)) else float("nan")
    grad_ref = max(grad_ref, 1.0e-9) if math.isfinite(grad_ref) else float("nan")

    def _excess(diffs: np.ndarray, idx_cross: int, span: slice, axis: int) -> float:
        # diffs[idx, span] along `axis`: idx_cross straddles the meridian; the 3 pairs on each side are the texture.
        take = (lambda i: diffs[i, span]) if axis == 0 else (lambda i: diffs[span, i])
        cross = take(idx_cross)
        cross = cross[np.isfinite(cross)]
        neigh = np.concatenate([take(idx_cross + d) for d in (-3, -2, -1, 1, 2, 3)])
        neigh = neigh[np.isfinite(neigh)]
        if cross.size == 0 or neigh.size == 0 or not math.isfinite(grad_ref):
            return float("nan")
        return max(float(np.mean(cross)) - float(np.mean(neigh)), 0.0) / grad_ref

    # The visual-field grid is symmetric about the centre between pixels cy-1 | cy (and cx-1 | cx).
    seam_h_left = _excess(dy, cy - 1, slice(2, cx - r_min), 0)
    seam_h_right = _excess(dy, cy - 1, slice(cx + r_min, w - 2), 0)
    seam_v_top = _excess(dx, cx - 1, slice(2, cy - r_min), 1)
    seam_v_bottom = _excess(dx, cx - 1, slice(cy + r_min, h - 2), 1)
    seams = [s for s in (seam_h_left, seam_h_right, seam_v_top, seam_v_bottom) if math.isfinite(s)]
    return {
        "mirror_symmetry": mirror,
        "mirror_symmetry_lr": mirror_lr,
        "mirror_symmetry_ud": mirror_ud,
        "chirality": (1.0 - max(abs(mirror_lr), abs(mirror_ud))) if math.isfinite(mirror) else float("nan"),
        "seam_h_left": seam_h_left,
        "seam_h_right": seam_h_right,
        "seam_v_top": seam_v_top,
        "seam_v_bottom": seam_v_bottom,
        "seam_contrast": max(seams) if seams else float("nan"),
        "texture_gradient_ref": grad_ref,
    }


def retinal_morphometrics(
    retinal_field: np.ndarray, stab: Dict[str, Any], r_min_deg: float = 1.0
) -> Dict[str, Any]:
    """Morphometrics of the retinal pullback that can be compared with Bressloff et al. 2002 §1.1
    ("about 30 to 36 stripes ... are perceived") and with the preset's requested spiral angle:

      rings_analytic      : number of cortical wavelengths between r_min_deg and the disc edge along
                            the horizontal meridian, Δx_c k / 2 pi, i.e. the number of perceived
                            contours of a tunnel (k_y = 0) planform in the rendered disc.
      rings_full_field    : the same count extrapolated to the paper's field, r from w0/eps (1.7 deg,
                            the end of the foveal plateau) to 90 deg, for comparison with 30-36.
      rings_fovea_to_90   : count from the fovea (0 deg) to 90 deg on the model map (extent x(90)).
      rings_fovea_to_90_hh: count the same wavelength would give on the Horton & Hoyt (1991) map
                            (extent A ln((90 + E2)/E2) = 83 mm): isolates how much of any shortfall
                            is the map's alpha (3/pi, fixed by upstream) rather than the wavelength.
      rings_counted       : sign changes of the rendered scalar field along the right horizontal
                            meridian / 2 (envelope periods). For contoured planforms whose u(phi)
                            takes both signs this is 2x rings_analytic (see the renderer docstring).
      spokes_at_20deg     : sign changes around the circle r = 20 deg / 2, i.e. the azimuthal
                            spoke (ray) count; k_y n_y cycles give n_y spokes for a pure ray planform.
      spiral_pitch_deg    : angle of the rendered oblique roll in the cortex, atan(k_y / k_x), from the
                            quantised wavevector of the first mode (compare `pattern_angle_deg`).
      visual_spiral_angle_deg : angle between the pulled-back log-spiral and the circles of constant
                            eccentricity in the far field, atan(beta k_y / (alpha k_x)): under the
                            complex-log map x = (alpha/eps) ln r, y = (beta/eps) theta a cortical roll
                            at angle a maps to a log-spiral whose angle to the circle is atan((beta/
                            alpha) tan a); with the upstream alpha = 3/pi, beta = 1.589/2 a 45-degree
                            roll gives 39.8 degrees (Bressloff 2001 §2 with alpha = beta would give 45).
    """
    f = np.asarray(retinal_field, dtype=float)
    h, w = f.shape
    grid = np.asarray(stab["retinal_grid_deg"], dtype=float)
    r_max = float(stab["retinal_max_deg"])
    k = float(stab["wave_number"])
    x_min = float(retino_cortical_forward(np.array([r_min_deg]), np.array([0.0]))[0][0])
    x_max = float(retino_cortical_forward(np.array([r_max]), np.array([0.0]))[0][0])
    x_plateau = float(retino_cortical_forward(np.array([RETINO_W0 / RETINO_EPS]), np.array([0.0]))[0][0])
    x_90 = float(retino_cortical_forward(np.array([90.0]), np.array([0.0]))[0][0])
    rings_analytic = (x_max - x_min) * k / (2.0 * math.pi)
    rings_full_field = (x_90 - x_plateau) * k / (2.0 * math.pi)
    rings_fovea_90 = x_90 * k / (2.0 * math.pi)
    with open(BRESSLOFF_DATASET_DIR / "horton_hoyt_1991_magnification.json", "r", encoding="utf-8") as fh:
        hh = json.load(fh)
    extent_hh = float(hh["A_mm_per_deg"]) * math.log((90.0 + float(hh["E2_deg"])) / float(hh["E2_deg"]))
    rings_hh = extent_hh * k / (2.0 * math.pi)

    cy = h // 2
    row = f[cy, :]
    right = row[(grid >= r_min_deg) & np.isfinite(row)]
    rings_counted = float(np.count_nonzero(np.diff(np.sign(right)) != 0)) / 2.0 if right.size > 2 else float("nan")

    r_probe = min(20.0, 0.75 * r_max)
    ang = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    px = r_probe * np.cos(ang)
    py = r_probe * np.sin(ang)
    step = grid[1] - grid[0]
    ix = np.clip(np.round((px - grid[0]) / step).astype(int), 0, w - 1)
    iy = np.clip(np.round((py - grid[0]) / step).astype(int), 0, h - 1)
    circ = f[iy, ix]
    circ = circ[np.isfinite(circ)]
    spokes = float(np.count_nonzero(np.diff(np.sign(np.append(circ, circ[0]))) != 0)) / 2.0 if circ.size > 2 else float("nan")

    modes = stab.get("rendered_modes", [])
    if modes:
        m0 = modes[0]
        pitch = math.degrees(math.atan2(abs(m0["ky"]), abs(m0["kx"]))) if abs(m0["kx"]) > 1e-12 else 90.0
        visual = math.degrees(math.atan2(RETINO_BETA * abs(m0["ky"]), RETINO_ALPHA * abs(m0["kx"])))
        n_y = float(m0["n_y"])
    else:
        pitch, visual, n_y = float("nan"), float("nan"), float("nan")
    lo, hi = PAPER_SCALE_ANCHORS["perceived_stripe_count"]
    return {
        "rings_analytic": float(rings_analytic),
        "rings_full_field": float(rings_full_field),
        "rings_full_field_in_paper_range": bool(lo <= rings_full_field <= hi),
        "rings_fovea_to_90": float(rings_fovea_90),
        "rings_fovea_to_90_hh": float(rings_hh),
        "rings_hh_in_paper_range": bool(lo <= rings_hh <= hi),
        "model_extent_0_90_mm": float(x_90),
        "hh_extent_0_90_mm": float(extent_hh),
        "rings_counted": rings_counted,
        "spokes_at_probe": spokes,
        "probe_radius_deg": float(r_probe),
        "spiral_pitch_deg": float(pitch),
        "visual_spiral_angle_deg": float(visual),
        "first_mode_ny": n_y,
        "paper_stripe_range": [lo, hi],
    }


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

    psd_norm = radial_psd / max(peak_power, 1.0e-12)
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
        "radial_psd_normalised": psd_norm,  # peak = 1; used to test harmonic rings (2k vs sqrt(3)k)
    }


def harmonic_ring_powers(spec: Dict[str, Any], k_fundamental: float) -> Dict[str, float]:
    """Normalised radial power at the fundamental k, at sqrt(3) k (hexagonal difference ring) and at
    2k (second harmonic of the tanh / max-over-phi nonlinearity), each taken as the max over ±1 bin."""
    psd = np.asarray(spec["radial_psd_normalised"], dtype=float)

    def _at(k: float) -> float:
        i = int(round(k))
        lo, hi = max(0, i - 1), min(len(psd), i + 2)
        return float(np.max(psd[lo:hi])) if hi > lo else 0.0

    return {
        "power_at_k": _at(k_fundamental),
        "power_at_sqrt3_k": _at(math.sqrt(3.0) * k_fundamental),
        "power_at_2k": _at(2.0 * k_fundamental),
        "sqrt3_k_bin": round(math.sqrt(3.0) * k_fundamental, 2),
        "two_k_bin": round(2.0 * k_fundamental, 2),
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
        bsel = stab["branch_selection_upstream"]   # fixture parity is against the upstream cubic path

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


def verify_paper_calibration_runs(tol: float = 1.0e-9) -> Tuple[bool, float, List[Dict[str, Any]]]:
    """Verify parity with every preset run in `paper-calibration.json` `runs[]` (24 runs) and surface
    each run's own paper-expectation `checks` (122 in total) and `status`.

    Parity compares this script's stability scan and branch selection for the run's preset with the
    fixture's recorded critical_q / critical_branch / selected_family / selected_pattern /
    selected_scope / rendered_pattern. Runs whose preset is not in PAPER_PRESETS are reported as
    `preset_available = False` and excluded from the parity verdict (none at this fixture version).
    """
    cal_path = BRESSLOFF_DATASET_DIR / "paper-calibration.json"
    with open(cal_path, "r", encoding="utf-8") as fh:
        cal_data = json.load(fh)
    rows: List[Dict[str, Any]] = []
    all_ok = True
    max_err = 0.0
    for run in cal_data.get("runs", []):
        preset = run.get("preset", {})
        pid = str(preset.get("id", ""))
        checks = [
            {
                "name": str(c.get("name")),
                "expected": c.get("expected"),
                "actual": c.get("actual"),
                "passed": bool(c.get("passed")),
            }
            for c in run.get("checks", [])
        ]
        row: Dict[str, Any] = {
            "id": pid,
            "paper_figure": str(preset.get("paper_figure", "")),
            "source_key": str(preset.get("source_key", "")),
            "status": str(run.get("status", "")),
            "calibration_status": str(preset.get("calibration_status", "")),
            "expected_family": preset.get("expected_family"),
            "expected_pattern": preset.get("expected_pattern"),
            "fixture_checks": checks,
            "fixture_checks_passed": all(c["passed"] for c in checks),
            "fixture_failed_checks": [c for c in checks if not c["passed"]],
            "preset_available": pid in PAPER_PRESETS,
            "parity_ok": None,
            "max_abs_err": float("nan"),
        }
        if pid in PAPER_PRESETS:
            params = make_preset_params(pid, n=128, m=12)
            stab = stability_scan(params)
            bsel = stab["branch_selection_upstream"]   # fixture parity is against the upstream cubic path
            rendered = bsel["selected_pattern"] if params.pattern == "auto" else params.pattern
            q_err = abs(stab["critical_q"] - float(run["critical_q"]))
            parity = (
                q_err <= tol
                and stab["critical_branch"] == run.get("critical_branch")
                and bsel["selected_family"] == run.get("selected_family")
                and bsel["selected_pattern"] == run.get("selected_pattern")
                and bsel["selected_scope"] == run.get("selected_scope")
                and rendered == run.get("rendered_pattern")
            )
            row.update(
                {
                    "critical_q": stab["critical_q"],
                    "ref_critical_q": float(run["critical_q"]),
                    "critical_branch": stab["critical_branch"],
                    "selected_family": bsel["selected_family"],
                    "selected_pattern": bsel["selected_pattern"],
                    "selected_scope": bsel["selected_scope"],
                    "global_selected_family": bsel["global_selected_family"],
                    "global_selected_stable": bsel["global_selected_stable"],
                    "unrestricted_selected_family": bsel["unrestricted_selected_family"],
                    "roll_stable": bsel["roll_stable"],
                    "rendered_pattern": rendered,
                    "rendered_parity": stab["critical_branch"] if params.pattern == "auto" else params.parity,
                    "parity_ok": bool(parity),
                    "max_abs_err": q_err,
                }
            )
            all_ok = all_ok and parity
            max_err = max(max_err, q_err)
        rows.append(row)
    return all_ok, max_err, rows


def summarize_fixture_checks(
    rows: Sequence[Dict[str, Any]], run_rows: Sequence[Dict[str, Any]] = ()
) -> Dict[str, Any]:
    """Aggregate the upstream fixture's own paper-expectation checks across the 5 stability reports
    AND the 24 preset runs (previously only the former were counted, under-disclosing 11 failures)."""
    total_sr = sum(len(r["fixture_checks"]) for r in rows)
    failed_sr = [(r["id"], c) for r in rows for c in r["fixture_failed_checks"]]
    total_runs = sum(len(r["fixture_checks"]) for r in run_rows)
    failed_runs = [(r["id"], c) for r in run_rows for c in r["fixture_failed_checks"]]
    status_counts: Dict[str, int] = {}
    for r in run_rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    fmt = lambda pairs: [  # noqa: E731
        {"report": rid, "name": c["name"], "expected": c["expected"], "actual": c["actual"]} for rid, c in pairs
    ]
    return {
        "total": total_sr + total_runs,
        "passed": (total_sr - len(failed_sr)) + (total_runs - len(failed_runs)),
        "failed": fmt(failed_sr) + fmt(failed_runs),
        "all_passed": not failed_sr and not failed_runs,
        "stability_reports": {"total": total_sr, "passed": total_sr - len(failed_sr), "failed": fmt(failed_sr)},
        "runs": {
            "total": total_runs,
            "passed": total_runs - len(failed_runs),
            "failed": fmt(failed_runs),
            "run_count": len(run_rows),
            "status_counts": status_counts,
            "review_runs": [r["id"] for r in run_rows if r["status"] != "pass"],
        },
    }


def load_registry_negative_control() -> Dict[str, Any]:
    """Read the Gaussian-only control entry of `driven-neural-fields-registry.json` so the registry is
    used (not merely checksummed) and the analogous control in Tamekue et al. (2024) is cross-referenced."""
    with open(BRESSLOFF_DATASET_DIR / "driven-neural-fields-registry.json", "r", encoding="utf-8") as fh:
        reg = json.load(fh)
    entries = {e.get("id"): e for e in reg.get("examples", [])}
    ctrl = entries.get("mackay_gaussian_kernel_negative_check", {})
    return {
        "registry_format": reg.get("format"),
        "implemented_count": reg.get("implemented_count"),
        "future_count": reg.get("future_count"),
        "example_count": len(entries),
        "control_id": ctrl.get("id", "mackay_gaussian_kernel_negative_check"),
        "control_label": ctrl.get("label", ""),
        "control_source_key": ctrl.get("source_key", ""),
        "control_kernel": ctrl.get("kernel_family", ""),
        "control_expected": ctrl.get("expected_behavior", ""),
        "control_claim_level": ctrl.get("public_claim_level", ""),
        "control_status": ctrl.get("implementation_status", ""),
        "control_found": bool(ctrl),
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


SWEEPABLE_PARAMS: Dict[str, Tuple[float, float, str]] = {
    # name: (default lo, default hi, axis label)
    "lateral_wide_sigma": (1.5, 6.0, "lateral surround width $\\hat{\\xi}_{lat}$ (hypercolumns)"),
    "lateral_inhibition": (0.0, 1.0, "lateral inhibition weight $I_{lat}$"),
    "lateral_spread_deg": (0.0, 90.0, "lateral spread $\\theta_0$ (deg)"),
    "lateral_sigma": (0.5, 2.0, "lateral centre width $\\xi_{lat}$ (hypercolumns)"),
    "eigen_beta": (0.1, 0.6, "lateral/local coupling ratio $\\beta$"),
    "local_sigma_deg": (10.0, 40.0, "local centre width $\\xi_{loc}$ (deg)"),
    "local_inhibition": (0.05, 0.6, "local inhibition $A_{loc}$ (contoured $\\leftrightarrow$ non-contoured)"),
    "sigmoid_baseline": (0.2, 0.8, "firing-rate baseline $\\sigma_0$ (quadratic feedback)"),
}


def regime_map(
    a_values: Sequence[float], sigma_values: Sequence[float], base_preset: str = "fig17_even"
) -> Dict[str, Any]:
    """2-D map over local inhibition A_loc x firing-rate baseline sigma0 of the critical harmonic
    (contoured / non-contoured regime), the full-cubic unrestricted selection and its stability."""
    fam_code = {"roll": 0, "rhombic": 1, "square": 2, "hexagonal": 3}
    fam = np.zeros((len(sigma_values), len(a_values)), dtype=int)
    harm = np.zeros_like(fam)
    stable = np.zeros(fam.shape, dtype=bool)
    subcrit = np.zeros(fam.shape, dtype=bool)
    resonant = np.zeros(fam.shape, dtype=bool)
    for i, s0 in enumerate(sigma_values):
        for j, a in enumerate(a_values):
            p = make_preset_params(base_preset)
            p.local_inhibition = float(a)
            p.sigmoid_baseline = float(s0)
            p.contour_mode = "auto"
            p.pattern = "auto"
            st = stability_scan(p)
            b = st["branch_selection"]
            fam[i, j] = fam_code[b["unrestricted_selected_family"]]
            harm[i, j] = int(st["critical_harmonic_exact"])
            stable[i, j] = bool(b["unrestricted_selected_stable"])
            subcrit[i, j] = bool(b["subcritical_cubic"])
            resonant[i, j] = bool(b["near_resonant"])
    return {
        "a_values": [float(v) for v in a_values], "sigma_values": [float(v) for v in sigma_values],
        "family": fam, "harmonic": harm, "stable": stable, "subcritical": subcrit, "near_resonant": resonant,
        "family_code": fam_code,
    }


def parameter_sweep(
    param: str,
    values: Optional[Sequence[float]] = None,
    base_preset: str = "fig17_even",
    n_points: int = 25,
) -> List[Dict[str, Any]]:
    """Sweep one coupling parameter of a paper preset and record q*, the critical parity, the
    cubic-order lattice selection (lattice-restricted, global and unrestricted) and lambda_c.
    This replaces the two-point contrasts (theta_0 = 0/60 deg, I_lat = 1/0) of earlier versions."""
    if param not in SWEEPABLE_PARAMS:
        raise ValueError(f"Unknown sweep parameter {param!r}; choose from {sorted(SWEEPABLE_PARAMS)}")
    lo, hi, _label = SWEEPABLE_PARAMS[param]
    if values is None:
        values = np.linspace(lo, hi, n_points)
    out: List[Dict[str, Any]] = []
    for v in values:
        p = make_preset_params(base_preset)
        p.contour_mode = "auto"   # sweeps follow the operator's own regime (critical harmonic), not the preset's
        setattr(p, param, float(v))
        stab = stability_scan(p)
        b = stab["branch_selection"]
        out.append(
            {
                "param": param,
                "value": float(v),
                "critical_q": float(stab["critical_q"]),
                "critical_branch": stab["critical_branch"],
                "critical_growth": float(stab["critical_growth"]),
                "critical_wavelength_mm": float(stab["critical_wavelength_mm"]),
                "rendered_cycles": int(stab["rendered_cycles"]),
                "selected_family": b["selected_family"],
                "global_selected_family": b["global_selected_family"],
                "unrestricted_selected_family": b["unrestricted_selected_family"],
                "rhombic_theta_optimal_deg": float(b["rhombic_theta_optimal_deg"]),
                "roll_stable": bool(b["roll_stable"]),
                "square_stable": bool(b["square_stable"]),
                "rhombic_stable": bool(b["rhombic_stable"]),
                "hex_stable": bool(b["hex_stable"]),
                "gamma0": float(b["gamma0"]),
                "gamma_hex": float(b["gamma_hex"]),
                "gamma_rhombic_optimal": float(b["gamma_rhombic_optimal"]),
                "eta_hex": float(b["eta_hex"]),
                # full-model fields (Pass 4)
                "critical_q_exact": float(stab["critical_q_exact"]),
                "critical_harmonic": int(stab["critical_harmonic_exact"]),
                "derived_contour_mode": stab["derived_contour_mode"],
                "roll_min_margin": float(b["roll_min_margin"]),
                "subcritical_cubic": bool(b["subcritical_cubic"]),
                "any_stable_branch": bool(b["any_stable_branch"]),
                "unrestricted_selected_stable": bool(b["unrestricted_selected_stable"]),
                "near_resonant": bool(b["near_resonant"]),
                "resonance_gap": float(b["resonance_gap"]),
                "sigma0": float(b["sigma0"]),
            }
        )
    return out


def write_sweep_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


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
    runs_ok, max_run_err, run_rows = verify_paper_calibration_runs()
    if not runs_ok:
        bad = [r["id"] for r in run_rows if r.get("parity_ok") is False]
        raise RuntimeError(f"Bressloff preset-run parity failed for {bad}: max q* err={max_run_err:.3e}")
    fixture_summary = summarize_fixture_checks(cal_rows, run_rows)
    for f in fixture_summary["failed"]:
        print(
            f"[WARN] upstream fixture paper-expectation check failed: {f['report']}/{f['name']} "
            f"expected={f['expected']!r} actual={f['actual']!r} (reported, not hidden)"
        )
    print(
        f"[INFO] fixture paper-expectation checks: {fixture_summary['passed']}/{fixture_summary['total']} passed "
        f"(stability_reports {fixture_summary['stability_reports']['passed']}/{fixture_summary['stability_reports']['total']}, "
        f"runs {fixture_summary['runs']['passed']}/{fixture_summary['runs']['total']}); "
        f"run status counts: {fixture_summary['runs']['status_counts']}"
    )

    neg_ctrl = evaluate_gaussian_negative_control()
    registry_ctrl = load_registry_negative_control()
    magnification = compare_cortical_magnification()

    # Gallery: Klüver's four form constants plus the rhombic lattice. Tunnel and spiral are rendered in
    # the DERIVED non-contoured regime (weak local inhibition -> critical m = 0 harmonic -> rolls stable
    # and selected by the full cubic equations, Bressloff 2001 §4(a)/Figs. 29-30 regime); cobweb,
    # honeycomb and rhombic use the upstream contoured presets (Figs. 31, 35, 33a). Every column is
    # labelled SELECTED / STABLE-coexisting / IMPOSED-unstable against the full cubic analysis.
    kluver_presets = [
        ("roll_noncontoured_derived", "I. Tunnel / Rings\n(n=0 roll, regime DERIVED)"),
        ("spiral_noncontoured_derived", "II. Spiral / Funnel\n(n=0 oblique roll, DERIVED)"),
        ("fig31_square_even", "III. Cobweb / Square\n(Fig. 31 even square)"),
        ("fig35_hex_zero_even", "IV. Honeycomb / Hex\n(Fig. 35 even $H_0$)"),
        ("fig33_rhombic_even", "Rhombic lattice\n(Fig. 33a even rhombic)"),
    ]
    rendered_gallery: List[Dict[str, Any]] = []
    for pid, title in kluver_presets:
        p = make_preset_params(pid, n=96, m=16)
        cort, ret, stab = render_cortical_and_retinal_planform(p, phase=0.0, retinal_res=160)
        spec = spectral_ring_metrics(cort)
        # The contoured rendering (max over orientation φ, then tanh) is strongly nonlinear; for the
        # hexagonal lattice most FFT power moves to the 2k harmonic ring (measured below, together with
        # the √3·k difference ring), so the spectral check against q* is made on the linear eigenmode
        # superposition (non-contoured, weak tanh). Both spectra are reported.
        p_lin = BressloffParams(**{**p.__dict__})
        p_lin.sharpness = 0.5
        cort_lin, _ret_lin, _stab_lin = render_cortical_and_retinal_planform(p_lin, phase=0.0, retinal_res=16, envelope_only=True)
        spec_lin = spectral_ring_metrics(cort_lin)
        harmonics = harmonic_ring_powers(spec, spec_lin["k_star_cycles"])
        # Same preset under the EXACT (2002) angular map in the three hemifield conventions, for the
        # chirality / seam comparison against the rendered conformal pullback.
        diag_rendered = retinal_pullback_diagnostics(ret)
        diag_by_mode: Dict[str, Dict[str, float]] = {}
        ret_by_mode: Dict[str, np.ndarray] = {}
        for hm_mode in ("full-field", "chiral-mirror", "bilateral-mirror"):
            p_alt = BressloffParams(**{**p.__dict__})
            p_alt.hemifield_mode = hm_mode
            p_alt.retinal_angular_map = "exact"
            _c_a, ret_alt, _s_a = render_cortical_and_retinal_planform(p_alt, phase=0.0, retinal_res=160)
            diag_by_mode[hm_mode] = retinal_pullback_diagnostics(ret_alt)
            ret_by_mode[hm_mode] = ret_alt
        bs = stab["branch_selection"]
        rendered_family = _branch_target_lattice(stab["rendered_pattern"])
        rendered_family = "roll" if rendered_family == "global" else rendered_family
        lattice_consistency = lattice_consistency_for(rendered_family, bs)
        rendered_gallery.append(
            {
                "preset_id": pid,
                "title": title,
                "cortical": cort,
                "retinal": ret,
                "retinal_mirror": ret_by_mode["bilateral-mirror"],
                "retinal_exact_fullfield": ret_by_mode["full-field"],
                "stability": stab,
                "spectral": spec,
                "spectral_linear": spec_lin,
                "harmonics": harmonics,
                "rendered_family": rendered_family,
                "lattice_consistency": lattice_consistency,
                "diag_full": diag_rendered,                       # rendered: conformal map, full-field
                "diag_fullfield": diag_by_mode["full-field"],      # exact map, full-field (earlier versions)
                "diag_mirror": diag_by_mode["bilateral-mirror"],   # exact map, bilateral-mirror
                "diag_chiral": diag_by_mode["chiral-mirror"],      # exact map, chiral-mirror
                "morphometrics": retinal_morphometrics(ret, stab),
            }
        )
    n_selected = sum(1 for g in rendered_gallery if g["lattice_consistency"] == "selected")
    n_stable = sum(1 for g in rendered_gallery if g["lattice_consistency"] == "stable-coexisting")
    n_imposed = sum(1 for g in rendered_gallery if g["lattice_consistency"] == "imposed-unstable")
    scale_anchor = compare_paper_scale_anchors(rendered_gallery[0]["stability"])
    # Paper-scale preset: lateral widths /3.3 so q* ~ 4.84 and lambda_c ~ 2.60 mm; rendered for Panel D.
    p_scale = make_preset_params("paper_scale_tunnel", n=192, m=16)
    cort_scale, ret_scale, stab_scale = render_cortical_and_retinal_planform(p_scale, phase=0.0, retinal_res=200)
    scale_anchor_paper = compare_paper_scale_anchors(stab_scale)
    morph_scale = retinal_morphometrics(ret_scale, stab_scale)
    paper_scale = {
        "preset_id": "paper_scale_tunnel",
        "stability": stab_scale,
        "retinal": ret_scale,
        "cortical": cort_scale,
        "anchor": scale_anchor_paper,
        "morphometrics": morph_scale,
        "diag": retinal_pullback_diagnostics(ret_scale),
    }
    print(
        f"[INFO] gallery lattice consistency: {n_selected} selected / {n_stable} stable-coexisting / {n_imposed} imposed-unstable "
        f"of {len(rendered_gallery)} panels (full cubic amplitude equations, rhombic angle free; reported per panel)"
    )
    print(
        f"[INFO] paper-scale anchor (gallery presets): rendered {scale_anchor['rendered_cycles']} cycles / λ_c = "
        f"{scale_anchor['critical_wavelength_model_mm']:.1f} mm vs paper 30-36 stripes / 2.4-3.2 mm "
        f"(ratio {scale_anchor['wavelength_ratio_model_over_paper']:.1f}x); paper_scale_tunnel preset: q* = "
        f"{stab_scale['rendered_q']:.2f}, λ_c = {scale_anchor_paper['critical_wavelength_model_mm']:.2f} mm, "
        f"{morph_scale['rings_fovea_to_90']:.1f} rings 0-90° on the model map / {morph_scale['rings_fovea_to_90_hh']:.1f} on H&H vs 30-36; foveal magnification "
        f"{scale_anchor['foveal_magnification_model']:.2f} vs 11.5 mm/deg"
    )
    print(
        f"[INFO] derived regime: gallery tunnel critical harmonic m={rendered_gallery[0]['stability']['critical_harmonic_exact']} "
        f"({rendered_gallery[0]['stability']['derived_contour_mode']}), roll_stable={rendered_gallery[0]['stability']['branch_selection']['roll_stable']}, "
        f"roll min margin={rendered_gallery[0]['stability']['branch_selection']['roll_min_margin']:.2f}; "
        f"n=1 presets: exact-vs-perturbative |Δλ| = {abs(rendered_gallery[2]['stability']['perturbative_vs_exact_dlambda']):.1e}"
    )

    # Marginal stability curves (Fig. 16 narrow spread vs Fig. 17 wide spread + Gaussian negative control)
    p16 = make_preset_params("fig16_odd")
    p17 = make_preset_params("fig17_even")
    p_gauss = make_preset_params("fig16_odd")
    p_gauss.lateral_inhibition = 0.0
    stab16 = stability_scan(p16)
    stab17 = stability_scan(p17)
    stab_gauss = stability_scan(p_gauss)
    gallery_params = make_preset_params(kluver_presets[0][0])

    # Parameter sweeps (replace the two-point contrasts; also available from the CLI via --sweep)
    sweeps = {
        "lateral_wide_sigma": parameter_sweep("lateral_wide_sigma", n_points=19),
        "lateral_inhibition": parameter_sweep("lateral_inhibition", n_points=21),
        "lateral_spread_deg": parameter_sweep("lateral_spread_deg", n_points=31),
        "local_inhibition": parameter_sweep("local_inhibition", n_points=23),
        "sigmoid_baseline": parameter_sweep("sigmoid_baseline", n_points=25, base_preset="roll_noncontoured_derived"),
    }
    sweep_summary = {
        name: {
            "value_min": rows[0]["value"],
            "value_max": rows[-1]["value"],
            "q_star_min": min(r["critical_q"] for r in rows),
            "q_star_max": max(r["critical_q"] for r in rows),
            "parity_flips": sum(1 for a, b in zip(rows, rows[1:]) if a["critical_branch"] != b["critical_branch"]),
            "parity_flip_values": [b["value"] for a, b in zip(rows, rows[1:]) if a["critical_branch"] != b["critical_branch"]],
            "families": sorted({r["unrestricted_selected_family"] for r in rows}),
            "roll_ever_stable": any(r["roll_stable"] for r in rows),
            "square_ever_stable": any(r["square_stable"] for r in rows),
            "harmonic_flips": sum(1 for a, b in zip(rows, rows[1:]) if a["critical_harmonic"] != b["critical_harmonic"]),
            "harmonic_flip_values": [b["value"] for a, b in zip(rows, rows[1:]) if a["critical_harmonic"] != b["critical_harmonic"]],
            "subcritical_values": [r["value"] for r in rows if r["subcritical_cubic"]],
            "near_resonant_values": [r["value"] for r in rows if r["near_resonant"]],
        }
        for name, rows in sweeps.items()
    }
    # Regime map A_loc x sigma0 (Panel G)
    regime = regime_map(np.linspace(0.05, 1.0, 20), np.linspace(0.2, 0.8, 13))
    # Analytic contoured/non-contoured threshold: W_0^loc = W_1^loc, i.e. the local harmonics alone
    # (the lateral terms shift it slightly; the map shows the operator's actual answer).
    p_thr = make_preset_params("fig17_even")
    a_grid = np.linspace(0.01, 0.8, 400)
    w_diff = []
    for a in a_grid:
        p_thr.local_inhibition = float(a)
        w_diff.append(local_weight_coeff(p_thr, 0) - local_weight_coeff(p_thr, 1))
    w_diff = np.array(w_diff)
    sign_change = np.where(np.diff(np.sign(w_diff)) != 0)[0]
    a_threshold_local = float(a_grid[sign_change[0]]) if sign_change.size else float("nan")

    # Load Rule et al. (2011) Floquet resonance tongue report
    with open(BRESSLOFF_DATASET_DIR / "rule-2011-floquet.json", "r", encoding="utf-8") as fh:
        floquet_data = json.load(fh)
    floquet_fit = floquet_data["source_curve_comparison"].get("fit_objective", {})
    floquet_amplitudes = sorted({float(c.get("amplitude", float("nan"))) for c in floquet_data.get("boundary_curves", [])})
    floquet_status = {
        "acceptance_status": str(floquet_fit.get("acceptance_status", "unknown")),
        "calibration_claim_allowed": bool(floquet_fit.get("calibration_claim_allowed", False)),
        "domain_rms_beta_error": float(
            floquet_data["source_curve_comparison"]["domain_beta_mapping"]["rms_error"]
        ),
        "domain_rms_beta_error_max": float(floquet_fit.get("domain_normalized_rms_beta_error_max", float("nan"))),
        "failed_acceptance_checks": list(floquet_fit.get("failed_acceptance_checks", [])),
        "boundary_curve_count": len(floquet_data.get("boundary_curves", [])),
        "forcing_amplitudes": floquet_amplitudes,
        "solver": "none in this script — curves are read from the upstream fixture (homogeneous-orbit monodromy grid)",
    }

    # Write CSV summary. Lattice selection is reported three times: `selected_*` is the branch chosen
    # within the preset's target lattice (lattice-local scope), `global_selected_*` the branch the
    # cubic amplitude equations select with the preset's rhombic angle, and `unrestricted_family`
    # the branch selected with the rhombic angle free (argmin gamma_3).
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
                "unrestricted_family",
                "lattice_consistency",
                "critical_q",
                "gamma0_or_k_star",
                "gamma_cross_or_sharpness",
                "eta_hex_or_m4_m6",
                "lattice_class_or_flag",
                "fixture_paper_checks",
                "retinal_diagnostics",
                "max_abs_err_vs_rust",
            ]
        )

        def _fail_txt(r: Dict[str, Any]) -> str:
            return "; ".join(
                f"{c['name']}: expected {c['expected']} got {c['actual']}" for c in r["fixture_failed_checks"]
            )

        for r in cal_rows:
            n_checks = len(r["fixture_checks"])
            n_pass = sum(1 for c in r["fixture_checks"] if c["passed"])
            fail_txt = _fail_txt(r)
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
                    "",
                    "",
                    f"{r['critical_q']:.6f}",
                    f"{r['gamma0']:.6f}",
                    f"{r['gamma_hex']:.6f}",
                    f"{r['eta_hex']:.6f}",
                    "hex_H0/Hpi_degenerate_at_cubic_order" if r["hex_pattern_degenerate"] else "",
                    f"{n_pass}/{n_checks}" + (f" FAILED[{fail_txt}]" if fail_txt else ""),
                    "",
                    f"{r['max_abs_err']:.3e}",
                ]
            )
        for r in run_rows:
            n_checks = len(r["fixture_checks"])
            n_pass = sum(1 for c in r["fixture_checks"] if c["passed"])
            fail_txt = _fail_txt(r)
            writer.writerow(
                [
                    "fixture_run_parity",
                    r["id"],
                    r["paper_figure"],
                    r.get("critical_branch", ""),
                    r.get("selected_family", ""),
                    r.get("selected_pattern", ""),
                    r.get("selected_scope", ""),
                    r.get("global_selected_family", ""),
                    str(bool(r.get("global_selected_stable", False))),
                    str(bool(r.get("roll_stable", False))),
                    r.get("rendered_pattern", ""),
                    r.get("unrestricted_selected_family", ""),
                    "",
                    f"{r['critical_q']:.6f}" if "critical_q" in r else "",
                    "", "", "",
                    f"status={r['status']}; calibration={r['calibration_status']}",
                    f"{n_pass}/{n_checks}" + (f" FAILED[{fail_txt}]" if fail_txt else ""),
                    "",
                    f"{r['max_abs_err']:.3e}" if r.get("parity_ok") is not None else "preset_unavailable",
                ]
            )
        for g in rendered_gallery:
            sp = g["spectral"]
            st = g["stability"]
            bs = st["branch_selection"]
            hm = g["harmonics"]
            writer.writerow(
                [
                    "kluver_form_constant",
                    g["preset_id"],
                    g["title"].replace("\n", " ").replace("$", ""),
                    st["rendered_parity"],
                    bs["selected_family"],
                    bs["selected_pattern"],
                    bs["selected_scope"],
                    bs["global_selected_family"],
                    str(bool(bs["global_selected_stable"])),
                    str(bool(bs["roll_stable"])),
                    st["rendered_pattern"],
                    f"{bs['unrestricted_selected_family']}@{bs['rhombic_theta_optimal_deg']:.0f}deg",
                    g["lattice_consistency"].upper(),
                    # Exact-operator q_mode, not the perturbative n=1 scan
                    # critical_q (Pass N+1: they differ for the m=0 presets).
                    f"{bs['q_mode']:.6f}",
                    f"{sp['k_star_cycles']:.2f}",
                    f"{sp['ring_sharpness']:.3f}",
                    f"{max(sp['m4_excess_over_m2'], sp['m6_excess_over_m2']):.4f}",
                    sp["lattice_class"],
                    f"rendered_cycles={st['rendered_cycles']} (q*/d_hyp -> {st['cycles_unquantised']:.2f} cyc); "
                    f"linear_k_star={g['spectral_linear']['k_star_cycles']:.0f}; contoured_k_star={sp['k_star_cycles']:.0f}; "
                    f"contoured power at k/sqrt3k/2k = {hm['power_at_k']:.2f}/{hm['power_at_sqrt3_k']:.2f}/{hm['power_at_2k']:.2f}",
                    f"rendered {st['retinal_angular_map']}/{st['hemifield_mode']}: chirality={g['diag_full']['chirality']:.3f}, "
                    f"seam_max={g['diag_full']['seam_contrast']:.3f}; exact map: full-field seam_max={g['diag_fullfield']['seam_contrast']:.3f} "
                    f"(h_left {g['diag_fullfield']['seam_h_left']:.3f}), chiral-mirror seam_max={g['diag_chiral']['seam_contrast']:.3f} "
                    f"(h_left {g['diag_chiral']['seam_h_left']:.3f}), bilateral-mirror chirality={g['diag_mirror']['chirality']:.3f}",
                    "n/a",
                ]
            )
        # Full cubic coefficients and regime diagnostics per gallery preset (Pass 4)
        for g in rendered_gallery:
            st = g["stability"]
            bs = st["branch_selection"]
            gt = bs["gamma_table"]
            g0 = bs["gamma0"] if abs(bs["gamma0"]) > 1e-12 else float("nan")
            writer.writerow(
                [
                    "cubic_coefficients_full",
                    g["preset_id"],
                    f"sigma0={bs['sigma0']:.2f}; s2={bs['s2']:.3f}; s3={bs['s3']:.4f}; Lambda={bs['lambda']:.2f}",
                    st["rendered_parity"],
                    bs["unrestricted_selected_family"],
                    bs["unrestricted_selected_pattern"],
                    "unrestricted",
                    bs["global_selected_family"],
                    str(bool(bs["unrestricted_selected_stable"])),
                    str(bool(bs["roll_stable"])),
                    f"m={bs['rendered_harmonic']} ({bs['rendered_contour_mode']}; derived m={bs['critical_harmonic']}"
                    f"{'; IMPOSED' if bs['imposed_harmonic'] else ''}{'; parity subcritical' if bs['parity_imposed_subcritical'] else ''})",
                    f"roll_min_margin={bs['roll_min_margin']:.3f}",
                    f"any_stable={bs['any_stable_branch']}; subcritical={bs['subcritical_cubic']}",
                    f"{bs['q_mode']:.6f}",
                    f"{bs['gamma0']:.6f}",
                    f"g(60)/g0={bs['gamma_hex'] / g0:.4f}; g(90)/g0={bs['gamma_square'] / g0:.4f}; g(th*)/g0={bs['gamma_rhombic_optimal'] / g0:.4f}@{bs['rhombic_theta_optimal_deg']:.0f}",
                    f"eta={bs['eta']:.6f}; Lambda_r={bs['lambda_r']:.4g}; Lambda_h={bs['lambda_h']:.4g}",
                    "gamma(theta)/gamma(0) @ " + ",".join(f"{t:.0f}" for t in gt["theta_deg"][::3]) + " = "
                    + ",".join(f"{v / g0:.3f}" for v in gt["gamma"][::3]),
                    "bare gamma3/N @ same = " + ",".join(f"{(3.0 * bs['s3'] * v / bs['N']) / bs['gamma0'] if bs['gamma0'] else float('nan'):.3f}" for v in gt["gamma3_bare"][::3]),
                    f"exact_vs_pert: dq={st['perturbative_vs_exact_dq']:.2e}, dlambda={st['perturbative_vs_exact_dlambda']:.2e}",
                    "n/a",
                ]
            )
        # Retinal morphometrics and hemifield seams per gallery preset + paper-scale preset
        for entry in [*rendered_gallery, paper_scale]:
            st = entry["stability"]
            mo = entry["morphometrics"]
            dg = entry.get("diag_full", entry.get("diag"))
            writer.writerow(
                [
                    "retinal_morphometrics",
                    entry["preset_id"],
                    f"rings 0-90deg {mo['rings_fovea_to_90']:.1f} (model map) / {mo['rings_fovea_to_90_hh']:.1f} (H&H map) vs paper 30-36 "
                    f"({'in' if mo['rings_hh_in_paper_range'] else 'out of'} range on H&H map)",
                    st["rendered_parity"],
                    "", "", "", "", "", "",
                    st["rendered_pattern"],
                    f"roll={mo['spiral_pitch_deg']:.2f}deg (requested {st['rendered_modes'][0]['angle_requested_rad'] * 180.0 / math.pi:.1f}); "
                    f"visual spiral angle={mo['visual_spiral_angle_deg']:.2f}deg; n_y={mo['first_mode_ny']:.0f}",
                    st["hemifield_mode"],
                    f"{st['rendered_q']:.6f}",
                    f"{mo['rings_analytic']:.4f}",
                    f"{mo['rings_counted']:.1f}",
                    f"{mo['spokes_at_probe']:.1f}",
                    f"rings_analytic(1-{st['retinal_max_deg']:.0f}deg) / rings_counted(envelope) / spokes@{mo['probe_radius_deg']:.0f}deg",
                    f"lambda_c={st['critical_wavelength_mm']:.3f}mm; cycles={st['rendered_cycles']}",
                    f"seams h_left={dg['seam_h_left']:.3f} h_right={dg['seam_h_right']:.3f} v_top={dg['seam_v_top']:.3f} "
                    f"v_bottom={dg['seam_v_bottom']:.3f} (excess / texture gradient); chirality={dg['chirality']:.3f}",
                    "n/a",
                ]
            )
        writer.writerow(
            [
                "regime_threshold",
                "local_inhibition_contoured_noncontoured",
                "W_0^loc = W_1^loc threshold (Bressloff 2001 §4a)",
                "", "", "", "", "", "", "", "", "",
                f"A_loc* (local terms only) = {a_threshold_local:.4f}",
                f"{a_threshold_local:.6f}",
                f"{sweep_summary['local_inhibition']['harmonic_flip_values'][0] if sweep_summary['local_inhibition']['harmonic_flip_values'] else float('nan'):.6f}",
                "", "",
                "A_loc* local-only / A_loc* from full-operator sweep (first harmonic flip)",
                f"sweep families: {'/'.join(sweep_summary['local_inhibition']['families'])}; roll_ever_stable={sweep_summary['local_inhibition']['roll_ever_stable']}",
                "",
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
                "global", "homogeneous", "False", "False", "", "", "",
                f"{neg_ctrl['gaussian_control_critical_q']:.6f}",
                f"{neg_ctrl['mexican_hat_peak_prominence']:.6f}",
                f"{neg_ctrl['gaussian_control_peak_prominence']:.6f}",
                "0.0000",
                f"registry analogue: {registry_ctrl['control_id']} ({registry_ctrl['control_source_key']}, "
                f"{registry_ctrl['control_kernel']}; {registry_ctrl['control_claim_level']})",
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
                "", "", "", "", "", "", "", "", "", "",
                f"{magnification['rms_rel_error']:.6f}",
                f"{magnification['model_A_mm_per_deg']:.6f}",
                f"{magnification['model_E2_deg']:.6f}",
                f"{magnification['max_abs_rel_error']:.6f}",
                f"model_extent_0_90={magnification['model_extent_0_90_mm']:.2f}mm vs HH {magnification['hh_extent_0_90_mm']:.2f}mm",
                "rms_rel_err / A_model / E2_model / max_rel_err over 1-40 deg",
                "",
                "n/a",
            ]
        )
        writer.writerow(
            [
                "paper_scale_anchor",
                "bressloff_2002_section_1_1",
                "Bressloff 2002 §1.1 scale statements",
                "", "", "", "", "", "", "", "", "", "",
                f"{scale_anchor['foveal_magnification_model']:.6f}",
                f"{scale_anchor['critical_wavelength_model_mm']:.6f}",
                f"{scale_anchor['wavelength_ratio_model_over_paper']:.6f}",
                f"{scale_anchor['stripe_count_ratio_paper_over_model']:.6f}",
                f"foveal_mag {scale_anchor['foveal_magnification_model']:.2f} vs 11.5 mm/deg (alpha {RETINO_ALPHA:.4f} vs implied "
                f"{scale_anchor['alpha_implied_by_paper']:.4f}); rendered {scale_anchor['rendered_cycles']} cyc vs 30-36 stripes; "
                f"lambda_c {scale_anchor['critical_wavelength_model_mm']:.2f} vs 2.4-3.2 mm; torus {scale_anchor['torus_model_mm']:.1f} mm "
                f"vs paper-implied {scale_anchor['torus_implied_by_paper_mm'][0]:.0f}-{scale_anchor['torus_implied_by_paper_mm'][1]:.0f} mm",
                "foveal_mag / lambda_c_mm / lambda ratio model:paper / stripe ratio paper:model",
                "",
                "n/a",
            ]
        )
        writer.writerow(
            [
                "floquet_diagnostic",
                "rule_2011_fig8_overlay",
                "Rule et al. 2011 Fig. 8",
                "", "", "", "", "", "", "", "", "", "",
                f"{floquet_status['domain_rms_beta_error']:.6f}",
                f"{floquet_status['domain_rms_beta_error_max']:.6f}",
                "", "",
                floquet_status["acceptance_status"],
                f"calibration_claim_allowed={floquet_status['calibration_claim_allowed']}; "
                f"failed={','.join(floquet_status['failed_acceptance_checks'])}; "
                f"{floquet_status['boundary_curve_count']} curves over amplitudes {floquet_status['forcing_amplitudes']}; no Floquet solver here",
                "",
                "n/a",
            ]
        )
        for name, rows in sweeps.items():
            s = sweep_summary[name]
            writer.writerow(
                [
                    "parameter_sweep",
                    name,
                    f"{SWEEPABLE_PARAMS[name][2].replace('$', '')} in [{s['value_min']:g}, {s['value_max']:g}]",
                    "/".join(sorted({r['critical_branch'] for r in rows})),
                    "/".join(s["families"]),
                    "", "", "", "",
                    str(s["roll_ever_stable"]),
                    "", "", "",
                    f"{s['q_star_min']:.6f}",
                    f"{s['q_star_max']:.6f}",
                    f"{len(rows)}",
                    f"{s['parity_flips']}",
                    f"parity flips at {', '.join(f'{v:g}' for v in s['parity_flip_values'])}" if s["parity_flips"] else "no parity flip",
                    f"q*_min / q*_max / n_points / n_parity_flips; square_ever_stable={s['square_ever_stable']}",
                    "",
                    "n/a",
                ]
            )

    # Render the synthesis figure: 2 gallery rows x 5 columns + two rows of diagnostics
    n_cols = len(rendered_gallery)
    fig = plt.figure(figsize=(19.0, 17.5), dpi=150)
    gs = fig.add_gridspec(4, n_cols, height_ratios=[1.0, 1.0, 1.1, 1.0], hspace=0.68, wspace=0.28)

    cmap_planform = "twilight_shifted"
    seg_step = 8  # orientation line elements every 8 grid cells

    def _fmt_seam(v: float) -> str:
        return "n/a" if not math.isfinite(v) else f"{v:.2f}"

    def _draw_orientation(ax: Any, orient: Optional[np.ndarray], xs_: np.ndarray, ys_: np.ndarray, length: float) -> None:
        if orient is None:
            return
        sub = orient[seg_step // 2 :: seg_step, seg_step // 2 :: seg_step]
        xg_, yg_ = np.meshgrid(xs_[seg_step // 2 :: seg_step], ys_[seg_step // 2 :: seg_step])
        ok = np.isfinite(sub)
        dx = 0.5 * length * np.cos(sub[ok])
        dy = 0.5 * length * np.sin(sub[ok])
        x0, y0 = xg_[ok], yg_[ok]
        ax.plot(
            np.vstack([x0 - dx, x0 + dx]), np.vstack([y0 - dy, y0 + dy]),
            color="white", lw=0.55, alpha=0.75, solid_capstyle="round",
        )

    for col, g in enumerate(rendered_gallery):
        ax_c = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])
        st = g["stability"]
        bs = st["branch_selection"]
        torus = st["cortical_torus_mm"]
        badge, badge_color = lattice_badge(g["lattice_consistency"], bs)

        ax_c.imshow(
            g["cortical"], cmap=cmap_planform, vmin=-1.0, vmax=1.0, origin="lower",
            extent=(0.0, torus, -0.5 * torus, 0.5 * torus),
        )
        n_c = g["cortical"].shape[0]
        cell = torus / n_c
        xs_c = (np.arange(n_c) + 0.5) * cell
        ys_c = (np.arange(n_c) + 0.5) * cell - 0.5 * torus
        _draw_orientation(ax_c, st["orientation_cortical"], xs_c, ys_c, length=0.75 * seg_step * cell)
        ax_c.axvline(st["cortical_x_for_retinal_max_mm"], color="white", ls="--", lw=0.9, alpha=0.85)
        ax_c.text(
            st["cortical_x_for_retinal_max_mm"] + 1.5, -0.47 * torus, f"{st['retinal_max_deg']:.0f}°",
            color="white", fontsize=7.0, va="bottom",
        )
        regime_txt = (
            f"m={bs['rendered_harmonic']} {bs['rendered_contour_mode']}"
            + (" (DERIVED)" if (not bs["imposed_harmonic"] and make_preset_params(g["preset_id"]).contour_mode == "auto")
               else (" (imposed)" if bs["imposed_harmonic"] else " (= critical)"))
        )
        ax_c.set_title(
            f"V1 Cortex: {g['title']}\n"
            f"$k^*$ lin {g['spectral_linear']['k_star_cycles']:.0f} / cont {g['spectral']['k_star_cycles']:.0f} cyc "
            f"($q^*/d_{{hyp}}$={st['cycles_unquantised']:.1f}), $\\lambda_c$={st['critical_wavelength_mm']:.1f} mm\n"
            f"{regime_txt}; FFT {g['spectral']['lattice_class']}",
            fontsize=6.8,
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
        _draw_orientation(ax_r, st["orientation_retinal"], st["retinal_grid_deg"], st["retinal_grid_deg"],
                          length=0.75 * seg_step * (2.0 * rmax / len(st["retinal_grid_deg"])))
        mo = g["morphometrics"]
        ax_r.set_title(
            f"Retinal pullback ({st['retinal_angular_map']} map, {st['hemifield_mode']}, $|r|\\leq{rmax:.0f}°$)\n"
            f"{st['rendered_pattern']} ({g['rendered_family']}) | roll-stable {bs['roll_stable']} "
            f"(min 2γ/γ₀−1 = {bs['roll_min_margin']:+.2f})\n"
            f"chirality {g['diag_full']['chirality']:.2f}, seam {_fmt_seam(g['diag_full']['seam_contrast'])} "
            f"(exact map: {_fmt_seam(g['diag_fullfield']['seam_contrast'])}); roll {mo['spiral_pitch_deg']:.0f}°, "
            f"{mo['rings_analytic']:.1f} rings",
            fontsize=6.6,
        )
        ax_r.text(
            0.5, -0.045, badge, transform=ax_r.transAxes, ha="center", va="top", fontsize=7.4,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=badge_color, ec="none"),
        )
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        if col == 0:
            ax_r.set_ylabel("Retinal Visual Field (exact inverse map)", fontsize=9.0, fontweight="bold")

    # Row 2, Panel A (cols 0..1): Bressloff (2001) Figs. 16 & 17 marginal stability + Gaussian negative control
    ax_stab = fig.add_subplot(gs[2, 0:2])
    ax_stab.plot(
        stab16["qs"], stab16["odd_growth"], color="#c0392b", lw=2.2,
        label=f"Fig. 16 odd ($\\theta_0=0^\\circ$, $\\hat{{\\xi}}_{{lat}}$={p16.lateral_wide_sigma:g}, $q^*={stab16['critical_q']:.2f}$)",
    )
    ax_stab.plot(stab16["qs"], stab16["even_growth"], color="#e67e22", ls="--", lw=1.6, label="Fig. 16 even ($\\theta_0=0^\\circ$)")
    ax_stab.plot(
        stab17["qs"], stab17["even_growth"], color="#2980b9", lw=2.2,
        label=f"Fig. 17 even ($\\theta_0=60^\\circ$, $\\hat{{\\xi}}_{{lat}}$={p17.lateral_wide_sigma:g}, $q^*={stab17['critical_q']:.2f}$)",
    )
    # Pass N+1: gallery[0] is the derived m=0 preset (A_loc=0.15), whose n=1
    # perturbative curve does not describe the contoured gallery III-V.
    # Use the contoured square preset (A_loc=1.0) and label it as such.
    stab_gal = rendered_gallery[2]["stability"]
    ax_stab.plot(
        stab_gal["qs"], stab_gal["even_growth"], color="#117a65", ls="-.", lw=1.6,
        label=f"Gallery III–V contoured even ($\\theta_0$={gallery_params.lateral_spread_deg:g}°, $\\hat{{\\xi}}_{{lat}}$={gallery_params.lateral_wide_sigma:g}, $q^*={stab_gal['critical_q']:.2f}$)",
    )
    ax_stab.plot(
        stab_gauss["qs"], stab_gauss["odd_growth"], color="#7f8c8d", ls=":", lw=2.0,
        label=f"Gaussian-only negative control ($I_{{\\mathrm{{lat}}}}=0$, $q^*\\to {stab_gauss['critical_q']:.2f}$)",
    )
    # Exact operator (all harmonics) for the derived non-contoured preset: the m = 0 branch overtakes m = 1
    stab_der = rendered_gallery[0]["stability"]
    ax_stab.plot(
        stab_der["qs"], stab_der["exact_curves"]["even_0"], color="#6c3483", lw=2.0,
        label=f"Exact $m=0$ branch, $A_{{loc}}$={make_preset_params(kluver_presets[0][0]).local_inhibition:g} "
              f"(gallery I–II; $q^*={stab_der['critical_q_exact']:.2f}$, non-contoured)",
    )
    ax_stab.plot(
        stab_der["qs"], stab_der["exact_curves"]["even_1"], color="#6c3483", ls="--", lw=1.3,
        label=f"Exact $m=1$ even branch, same $A_{{loc}}$ (perturbative scan gives $q^*={stab_der['critical_q']:.2f}$)",
    )
    ax_stab.axvline(stab16["critical_q"], color="#c0392b", ls=":", alpha=0.6)
    ax_stab.axvline(stab17["critical_q"], color="#2980b9", ls=":", alpha=0.6)
    ax_stab.axvline(stab_gal["critical_q"], color="#117a65", ls=":", alpha=0.6)
    ax_stab.axvline(stab_der["critical_q_exact"], color="#6c3483", ls=":", alpha=0.6)
    ax_stab.set_xlabel("Dimensionless Cortical Wavenumber $q = |\\mathbf{k}| d_{\\mathrm{hyp}}$", fontsize=9.0)
    ax_stab.set_ylabel("Eigenvalue $\\lambda(q)$", fontsize=9.0)
    ax_stab.set_title(
        "A. Marginal stability: Figs. 16–17 (perturbative, Rust parity), exact-operator branches, negative control\n"
        f"exact vs perturbative $n=1$: $|\\Delta\\lambda|$={abs(stab17['perturbative_vs_exact_dlambda']):.0e}, "
        f"$|\\Delta q^*|$={abs(stab17['perturbative_vs_exact_dq']):.0e}; gallery I–II: critical harmonic "
        f"$m$={stab_der['critical_harmonic_exact']} ({stab_der['derived_contour_mode']})",
        fontsize=7.6,
        fontweight="bold",
    )
    ax_stab.legend(fontsize=5.9, loc="lower right", ncol=2, handlelength=1.8, columnspacing=0.9)
    ax_stab.grid(alpha=0.25)

    # Row 2, Panel B (col 2): FULL cubic coefficients gamma(theta)/gamma(0) (Eq. 5.15 with sigma_2 feedback)
    ax_amp = fig.add_subplot(gs[2, 2])
    bs_gal = rendered_gallery[0]["stability"]["branch_selection"]
    bs_n1 = stab17["branch_selection"]
    curves_b = []
    for s0, col_b, ls_b, lbl in [
        (0.5, "#2980b9", "-", "$m$=1 even (Fig. 17), $\\sigma_0$=½ (bare $\\Gamma_3$, Fig. 37)"),
        (0.35, "#1a5276", "--", "$m$=1 even, $\\sigma_0$=0.35 (quadratic feedback)"),
        (0.25, "#85929e", ":", "$m$=1 even, $\\sigma_0$=0.25"),
    ]:
        cub = full_cubic_coefficients(p17, bs_n1["q_mode"], "even", 1, sigma0=s0)
        g0b = cub["gamma"][0]
        ratio = [v / g0b for v in cub["gamma"]] if abs(g0b) > 1e-12 else [float("nan")] * len(cub["gamma"])
        curves_b.append((s0, g0b, ratio))
        ax_amp.plot(cub["theta_deg"], ratio, color=col_b, ls=ls_b, lw=1.9,
                    label=lbl + (f"; $\\gamma(0)$={g0b:+.3f}" + (" SUBCRITICAL" if g0b <= 0 else "")))
    gt0 = bs_gal["gamma_table"]
    g0_der = bs_gal["gamma0"]
    ax_amp.plot(gt0["theta_deg"], [v / g0_der for v in gt0["gamma"]], color="#6c3483", lw=2.2,
                label=f"$m$=0 non-contoured (gallery I–II), $\\sigma_0$={bs_gal['sigma0']:g}; rolls STABLE")
    ax_amp.axhline(0.5, color="#333333", ls="--", lw=1.1, label="$\\gamma/\\gamma(0)=\\frac{1}{2}$: rolls stable above")
    for ang_mark, lbl in [(bs_n1["rhombic_theta_optimal_deg"], "Rh*"), (60.0, "Hex"), (90.0, "Sq")]:
        ax_amp.axvline(ang_mark, color="#555555", ls=":", lw=1.0)
        ax_amp.text(ang_mark + 1.0, 1.02, lbl, fontsize=7.0, color="#333333")
    ax_amp.set_ylim(-0.2, 1.3)
    ax_amp.set_xlim(0.0, 90.0)
    ax_amp.set_xlabel("Wavevector Intersection Angle $\\theta$ (deg)", fontsize=9.0)
    ax_amp.set_ylabel("$\\gamma(\\theta)/\\gamma(0)$ (full cubic, Eq. 5.15)", fontsize=9.0)
    ax_amp.set_title(
        f"B. Full cubic $\\gamma(\\theta)=[3s_3\\Gamma_3-2s_2^2Q]/N$ (Eq. 5.15)\n"
        f"$m$=1: rolls unstable; $m$=0: margin {bs_gal['roll_min_margin']:+.2f} ⇒ stable",
        fontsize=7.0,
        fontweight="bold",
    )
    ax_amp.legend(fontsize=5.6, loc="lower left")
    ax_amp.grid(alpha=0.25)

    # Row 2, Panel C (col 3): Rule et al. (2011) Fig. 8 Floquet tongues, one shade per forcing amplitude
    ax_rule = fig.add_subplot(gs[2, 3])
    beta_scale = float(
        floquet_data.get("wave_number_normalization", {}).get("source_beta_per_model_cycle", 0.42868451880191133)
    )
    amp_list = floquet_status["forcing_amplitudes"]
    cm_minus = plt.get_cmap("Purples")
    cm_plus = plt.get_cmap("GnBu")
    labelled: set = set()
    for curve in floquet_data.get("boundary_curves", []):
        pts = curve.get("points", [])
        if not pts:
            continue
        periods = [float(p["period_ms"]) for p in pts]
        betas = [float(p.get("source_beta", float(p["beta_cycles"]) * beta_scale)) for p in pts]
        cid = curve.get("curve_id", "")
        amp = float(curve.get("amplitude", float("nan")))
        shade = 0.35 + 0.6 * (amp_list.index(amp) / max(len(amp_list) - 1, 1)) if amp in amp_list else 0.7
        is_minus = "minus_period_doubling" in cid
        key = ("-" if is_minus else "+", amp)
        ax_rule.plot(
            periods, betas,
            color=(cm_minus if is_minus else cm_plus)(shade), lw=1.6,
            marker="o" if is_minus else "s", ms=2.8,
            label=(f"{'2:1 hex (−1)' if is_minus else '1:1 stripes (+1)'}, amp {amp:g}" if key not in labelled else None),
        )
        labelled.add(key)
    ax_rule.set_xlabel("Flicker Forcing Period $T$ (ms)", fontsize=9.0)
    ax_rule.set_ylabel("Normalized Wavenumber $\\beta_{\\mathrm{source}}$", fontsize=9.0)
    ax_rule.set_title(
        "C. Rule (2011) Fig. 8 tongues — DIAGNOSTIC OVERLAY\n"
        f"{floquet_status['boundary_curve_count']} fixture curves, {len(amp_list)} amplitudes (shade); no solver",
        fontsize=7.0,
        fontweight="bold",
        color="#7b241c",
    )
    ax_rule.text(
        0.02, 0.03,
        f"domain-normalised RMS β error = {floquet_status['domain_rms_beta_error']:.3f} "
        f"(threshold {floquet_status['domain_rms_beta_error_max']:.2f})\n"
        f"failed: {len(floquet_status['failed_acceptance_checks'])} acceptance checks\n"
        "numerical calibration targets, not the published figure",
        transform=ax_rule.transAxes, fontsize=6.2, va="bottom", ha="left", color="#7b241c",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fdf2f0", ec="#e6b0aa", lw=0.6),
    )
    ax_rule.legend(fontsize=5.6, loc="upper right", ncol=2, handlelength=1.4, columnspacing=0.8)
    ax_rule.grid(alpha=0.25)

    # Row 2, Panel F (col 4, three stacked axes): q* vs lateral surround width, lateral inhibition, lateral spread
    gs_f = gs[2, 4].subgridspec(3, 1, hspace=1.25)
    ax_sw1 = fig.add_subplot(gs_f[0])
    ax_sw1b = fig.add_subplot(gs_f[1])
    ax_sw1c = fig.add_subplot(gs_f[2])
    sw_ws = sweeps["lateral_wide_sigma"]
    sw_in = sweeps["lateral_inhibition"]
    s_ws, s_in = sweep_summary["lateral_wide_sigma"], sweep_summary["lateral_inhibition"]
    ax_sw1.plot([r["value"] for r in sw_ws], [r["critical_q"] for r in sw_ws], color="#1f618d", lw=1.6, marker="o", ms=2.2)
    for v_mark, c_mark, lbl in [(gallery_params.lateral_wide_sigma, "#117a65", "gallery"), (p17.lateral_wide_sigma, "#2980b9", "Fig. 17")]:
        ax_sw1.axvline(v_mark, color=c_mark, ls=":", lw=1.0)
        ax_sw1.text(v_mark + 0.08, s_ws["q_star_min"] + 0.05, lbl, fontsize=5.8, color=c_mark)
    ax_sw1.set_xlabel("$\\hat{\\xi}_{lat}$ (hypercolumns)", fontsize=6.4, labelpad=0.5)
    ax_sw1.set_ylabel("$q^*$", fontsize=7.5)
    ax_sw1.tick_params(labelsize=6.0)
    ax_sw1.set_title(
        f"F. $q^*$ sweeps (fig17_even base, full cubic)\n"
        f"$\\hat{{\\xi}}_{{lat}}$ {s_ws['value_min']:g}→{s_ws['value_max']:g}: $q^*$ {s_ws['q_star_max']:.2f}→{s_ws['q_star_min']:.2f}; "
        f"rolls/squares ever stable: {s_ws['roll_ever_stable'] or s_in['roll_ever_stable'] or s_ws['square_ever_stable'] or s_in['square_ever_stable']}",
        fontsize=6.6, fontweight="bold",
    )
    ax_sw1.grid(alpha=0.25)
    ax_sw1b.plot([r["value"] for r in sw_in], [r["critical_q"] for r in sw_in], color="#b9770e", lw=1.6, marker="s", ms=2.2)
    ax_sw1b.set_xlabel("$I_{lat}$", fontsize=6.4, labelpad=0.5)
    ax_sw1b.set_ylabel("$q^*$", fontsize=7.5)
    ax_sw1b.tick_params(labelsize=6.0)
    ax_sw1b.set_title(
        f"$I_{{lat}}$ 0→1: $q^*$ {s_in['q_star_min']:.2f}→{s_in['q_star_max']:.2f} "
        f"(selected: {'/'.join(sorted(set(s_ws['families']) | set(s_in['families'])))})",
        fontsize=6.6, fontweight="bold",
    )
    ax_sw1b.grid(alpha=0.25)
    sw_sp = sweeps["lateral_spread_deg"]
    s_sp = sweep_summary["lateral_spread_deg"]
    vals = np.array([r["value"] for r in sw_sp])
    qs_sp = np.array([r["critical_q"] for r in sw_sp])
    par = np.array([r["critical_branch"] for r in sw_sp])
    for parity_name, color in [("even", "#2980b9"), ("odd", "#c0392b")]:
        m = par == parity_name
        ax_sw1c.scatter(vals[m], qs_sp[m], color=color, s=9, label=parity_name, zorder=3)
    ax_sw1c.plot(vals, qs_sp, color="#7f8c8d", lw=0.9, zorder=2)
    for v_flip in s_sp["parity_flip_values"]:
        ax_sw1c.axvline(v_flip, color="#555555", ls="--", lw=0.8)
    ax_sw1c.set_xlabel("$\\theta_0$ (deg)", fontsize=6.4, labelpad=0.5)
    ax_sw1c.set_ylabel("$q^*$", fontsize=7.5)
    ax_sw1c.tick_params(labelsize=6.0)
    ax_sw1c.set_title(
        f"$\\theta_0$ 0→90°: parity flips {s_sp['parity_flips']} at "
        f"{', '.join(f'{v:.0f}°' for v in s_sp['parity_flip_values']) or '—'}; selected {'/'.join(s_sp['families'])}",
        fontsize=6.6, fontweight="bold",
    )
    ax_sw1c.legend(fontsize=5.6, loc="best")
    ax_sw1c.grid(alpha=0.25)

    # Row 3, Panel D (cols 0..1): empirical anchor — cortical magnification vs Horton & Hoyt (1991) + paper scale
    ax_mag = fig.add_subplot(gs[3, 0:2])
    ax_mag.plot(
        magnification["eccentricity_deg"], magnification["horton_hoyt_mm_per_deg"],
        color="#1b2631", lw=2.4,
        label=f"Horton & Hoyt (1991) human V1: $M(E)=A/(E+E_2)$, A={magnification['hh_A_mm_per_deg']:.1f}, $E_2$={magnification['hh_E2_deg']:.2f}°",
    )
    ax_mag.plot(
        magnification["eccentricity_deg"], magnification["model_mm_per_deg"],
        color="#2980b9", lw=2.0, ls="--",
        label=f"Bressloff map $dx/dr=\\alpha/(w_0+\\epsilon r)$: A={magnification['model_A_mm_per_deg']:.2f}, $E_2$={magnification['model_E2_deg']:.2f}° (α, β from upstream params.rs)",
    )
    ax_mag.scatter(
        [0.0, scale_anchor["knee_eccentricity_deg"]],
        [scale_anchor["foveal_magnification_paper"], scale_anchor["magnification_at_knee_paper"]],
        color="#c0392b", marker="*", s=70, zorder=5,
        label=f"Bressloff 2002 §1.1 stated: 11.5 mm/deg at fovea, 5.75 at {scale_anchor['knee_eccentricity_deg']:.1f}° (model: "
              f"{scale_anchor['foveal_magnification_model']:.2f}, {scale_anchor['magnification_at_knee_model']:.2f})",
    )
    ax_mag.set_yscale("log")
    ax_mag.set_xlim(-0.5, 40.5)
    ax_mag.set_xlabel("Eccentricity $E$ (deg)", fontsize=9.0)
    ax_mag.set_ylabel("Linear magnification (mm / deg)", fontsize=9.0)
    ax_mag.set_title(
        "D. Empirical anchors: magnification vs. Horton & Hoyt (1991) and the paper's own scale statements\n"
        f"H&H RMS rel. error {100.0 * magnification['rms_rel_error']:.1f}% (max {100.0 * magnification['max_abs_rel_error']:.1f}%) over 1–40°; "
        f"fovea→90° extent {magnification['model_extent_0_90_mm']:.1f} mm vs {magnification['hh_extent_0_90_mm']:.1f} mm (H&H)",
        fontsize=8.4, fontweight="bold",
    )
    ax_mag.text(
        0.02, 0.04,
        f"Perceptual scale (2002 §1.1): gallery presets λ_c = {scale_anchor['critical_wavelength_model_mm']:.1f} mm, "
        f"{rendered_gallery[0]['morphometrics']['rings_fovea_to_90']:.0f} rings 0→90°\n"
        f"vs paper 30–36 stripes, 2.4–3.2 mm ⇒ {scale_anchor['wavelength_ratio_model_over_paper']:.1f}× too long (needs q* ≈ "
        f"{scale_anchor['q_star_needed_for_paper_wavelength']:.1f}).\n"
        f"paper_scale_tunnel (ξ_lat, ξ̂_lat ÷ 3.3; q scan ≤ 6): q* = {stab_scale['rendered_q']:.2f}, λ_c = "
        f"{scale_anchor_paper['critical_wavelength_model_mm']:.2f} mm,\n"
        f"rings 0→90°: {morph_scale['rings_fovea_to_90']:.0f} on this map ({morph_scale['model_extent_0_90_mm']:.0f} mm) / "
        f"{morph_scale['rings_fovea_to_90_hh']:.0f} on H&H's ({morph_scale['hh_extent_0_90_mm']:.0f} mm; paper 30–36) → inset.\n"
        f"Residual shortfall = upstream α (map too short), not q*. Torus L = {scale_anchor['torus_model_mm']:.0f} mm.",
        transform=ax_mag.transAxes, fontsize=6.4, va="bottom", ha="left", color="#7b241c",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f0", ec="#e6b0aa", lw=0.6),
    )
    ax_ins = ax_mag.inset_axes([0.60, 0.30, 0.19, 0.46])
    rmax_s = stab_scale["retinal_max_deg"]
    ax_ins.imshow(ret_scale, cmap=cmap_planform, vmin=-1.0, vmax=1.0, origin="lower",
                  extent=(-rmax_s, rmax_s, -rmax_s, rmax_s))
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.set_title(f"paper-scale tunnel, |r|≤{rmax_s:.0f}°\n{morph_scale['rings_analytic']:.0f} rings 1°→{rmax_s:.0f}°",
                     fontsize=5.8)
    ax_mag.legend(fontsize=6.6, loc="upper right")
    ax_mag.grid(alpha=0.25, which="both")

    # Row 3, Panel E (cols 2..3): calibration & provenance ledger (what is and is not claimed)
    ax_led = fig.add_subplot(gs[3, 2:4])
    ax_led.axis("off")
    fx_sr, fx_runs = fixture_summary["stability_reports"], fixture_summary["runs"]
    hex_g = next(g for g in rendered_gallery if g["rendered_family"] == "hexagonal")
    s_li = sweep_summary["local_inhibition"]
    s_sb = sweep_summary["sigmoid_baseline"]
    seam_chiral_max = float(np.nanmax([g["diag_chiral"]["seam_contrast"] for g in rendered_gallery]))
    seam_full_max = float(np.nanmax([g["diag_fullfield"]["seam_contrast"] for g in rendered_gallery]))
    seam_conformal_max = float(np.nanmax([g["diag_full"]["seam_contrast"] for g in rendered_gallery]))
    ledger_lines = [
        f"Fixture SHA-256: {sum(1 for ok, _ in checksums.values() if ok)}/{len(checksums)} verified; registry read: "
        f"{registry_ctrl['example_count']} examples",
        f"Rust parity (upstream cubic path: q*, γ0, γ(θ), η, branch, scope): max |Δ| = {max(max_cal_err, max_run_err):.1e}",
        f"   {len(cal_rows)} stability_reports + {len(run_rows)} preset runs, all match",
        f"Fixture paper-expectation checks: {fixture_summary['passed']}/{fixture_summary['total']} passed",
        f"   stability_reports {fx_sr['passed']}/{fx_sr['total']}  |  runs {fx_runs['passed']}/{fx_runs['total']}; "
        f"run status {fx_runs['status_counts']}",
    ]
    fail_names: Dict[str, int] = {}
    for f in fixture_summary["failed"]:
        fail_names[f["name"]] = fail_names.get(f["name"], 0) + 1
    ledger_lines.append("   failed by check: " + ", ".join(f"{k}×{v}" for k, v in sorted(fail_names.items())))
    ledger_lines += [
        f"FULL cubic eqs. (Eq. 5.15: σ-derivative weights + quadratic feedback via resolvent), σ0 = {bs_gal['sigma0']:g}:",
        f"   σ0 = ½ ⇒ s2 = 0 ⇒ γ ∝ bare Γ3 (upstream); m=1 regime: rhombic@{bs_n1['rhombic_theta_optimal_deg']:.0f}° selected for all σ0,",
        f"   γ(0) < 0 (subcritical) for σ0 ≤ {max([c[0] for c in curves_b if c[1] <= 0], default=float('nan')):g}; "
        f"quintic selection (2001 §5) NOT implemented",
        f"Contour mode DERIVED: critical harmonic m=0 (non-contoured) iff A_loc < {a_threshold_local:.3f} (W0=W1; "
        f"sweep flip at {s_li['harmonic_flip_values'][0] if s_li['harmonic_flip_values'] else float('nan'):.3f});",
        f"   there rolls are STABLE vs all θ (min 2γ/γ0−1 = {bs_gal['roll_min_margin']:+.2f}) and SELECTED ⇒ tunnel & spiral (I–II)",
        f"   σ0 sweep in m=0 regime: subcritical for σ0 ∈ [{min(s_sb['subcritical_values'], default=float('nan')):.2f}, "
        f"{max(s_sb['subcritical_values'], default=float('nan')):.2f}] (K=0 resonance) — reported, not hidden",
        f"Gallery: {n_selected} SELECTED / {n_stable} STABLE-coexisting / {n_imposed} IMPOSED-unstable of "
        f"{len(rendered_gallery)} panels (full cubic, rhombic angle free)",
        f"Exact operator (m ≤ {ORIENT_HARMONICS}) vs perturbative n=1 scan: |Δλ| = {abs(stab17['perturbative_vs_exact_dlambda']):.0e}, "
        f"|Δq*| = {abs(stab17['perturbative_vs_exact_dq']):.0e}; rendering uses exact u(φ)",
        f"Rendered |k| = q*/d_hyp, k_x and k_y quantised to integer torus cycles (L = {rendered_gallery[0]['stability']['torus_mm']:.1f} mm; "
        f"spiral pitch {rendered_gallery[1]['morphometrics']['spiral_pitch_deg']:.1f}° for 45° requested)",
        f"Hex contoured FFT: power at k/√3k/2k = {hex_g['harmonics']['power_at_k']:.2f}/"
        f"{hex_g['harmonics']['power_at_sqrt3_k']:.2f}/{hex_g['harmonics']['power_at_2k']:.2f} ⇒ "
        f"{'√3k difference ring' if hex_g['harmonics']['power_at_sqrt3_k'] >= hex_g['harmonics']['power_at_2k'] else '2k harmonic'} of tanh∘max",
        f"Retinal pullback: conformal (2001 complex-log) map ⇒ torus closes, seam max {seam_conformal_max:.2f} over gallery; spiral chirality "
        f"{rendered_gallery[1]['diag_full']['chirality']:.2f}",
        f"   exact (2002) map seams: full-field {seam_full_max:.2f} (left meridian; earlier versions), chiral-mirror {seam_chiral_max:.2f} (vertical)",
        f"Paper scale: gallery λ_c {scale_anchor['critical_wavelength_model_mm']:.1f} mm ({scale_anchor['wavelength_ratio_model_over_paper']:.1f}× paper); "
        f"paper_scale_tunnel λ_c {scale_anchor_paper['critical_wavelength_model_mm']:.2f} mm, rings 0→90° {morph_scale['rings_fovea_to_90']:.0f} (map) / "
        f"{morph_scale['rings_fovea_to_90_hh']:.0f} (H&H; paper 30–36)",
        f"Rule (2011) overlay: {floquet_status['acceptance_status']}, RMS β err {floquet_status['domain_rms_beta_error']:.3f}; "
        f"no Floquet solver — no calibration claim",
        f"Negative control: Gaussian-only (I_lat=0) q* → {neg_ctrl['gaussian_control_critical_q']:.2f}, prominence "
        f"{neg_ctrl['gaussian_control_peak_prominence']:.1e}",
        f"  registry analogue: {registry_ctrl['control_id']} ({registry_ctrl['control_source_key']})",
    ]
    ax_led.text(
        0.0, 1.0, "\n".join(ledger_lines), transform=ax_led.transAxes, fontsize=5.4, va="top", ha="left",
        family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9f9", ec="#b3b6b7", lw=0.7),
    )
    ax_led.set_title("E. Calibration & provenance ledger (claims vs. diagnostics)", fontsize=9.0, fontweight="bold")

    # Row 3, Panel G (col 4): regime map A_loc x sigma0 — critical harmonic and full-cubic selection
    ax_sw2 = fig.add_subplot(gs[3, 4])
    fam_colors = {0: "#6c3483", 1: "#b9770e", 2: "#1e8449", 3: "#2980b9"}   # roll, rhombic, square, hex
    fam_names = {v: k for k, v in regime["family_code"].items()}
    a_vals = np.array(regime["a_values"])
    s_vals = np.array(regime["sigma_values"])
    rgb = np.zeros(regime["family"].shape + (3,))
    for code, col_hex in fam_colors.items():
        rgb[regime["family"] == code] = matplotlib_colors_to_rgb(col_hex)
    unreliable = regime["subcritical"] | regime["near_resonant"]
    rgb[unreliable] = matplotlib_colors_to_rgb("#d5d8dc")   # selection not shown where the cubic truncation is unreliable
    da = (a_vals[1] - a_vals[0]) / 2.0
    ds = (s_vals[1] - s_vals[0]) / 2.0
    ax_sw2.imshow(rgb, origin="lower", aspect="auto", extent=(a_vals[0] - da, a_vals[-1] + da, s_vals[0] - ds, s_vals[-1] + ds),
                  interpolation="nearest", alpha=0.85)
    # grey + hatch: subcritical (gamma0 <= 0); grey + dot: feedback-dominated (cubic truncation unreliable)
    for i, s0 in enumerate(s_vals):
        for j, a in enumerate(a_vals):
            if regime["subcritical"][i, j]:
                ax_sw2.add_patch(plt.Rectangle((a - da, s0 - ds), 2 * da, 2 * ds, fill=False, hatch="////", ec="#7f8c8d", lw=0))
            elif regime["near_resonant"][i, j]:
                ax_sw2.plot(a, s0, ".", color="#7f8c8d", ms=3.0)
    ax_sw2.contour(a_vals, s_vals, regime["harmonic"], levels=[0.5], colors="black", linewidths=1.6)
    ax_sw2.axvline(a_threshold_local, color="black", ls=":", lw=1.0)
    ax_sw2.axhline(0.5, color="#555555", ls=":", lw=0.9)
    ax_sw2.plot([make_preset_params(kluver_presets[0][0]).local_inhibition], [0.5], marker="*", color="white", ms=10, mec="black")
    ax_sw2.plot([p17.local_inhibition], [0.5], marker="o", color="white", ms=7, mec="black")
    present = sorted({int(v) for v in regime["family"][~unreliable]})
    handles = [Patch(fc=fam_colors[c], label=f"{fam_names[c]} selected") for c in present]
    handles += [Patch(fc="#d5d8dc", ec="#7f8c8d", hatch="////", label="subcritical γ(0)≤0 (grey: not shown)"),
                plt.Line2D([], [], marker=".", color="#d5d8dc", mec="#7f8c8d", mfc="#7f8c8d", ls="", ms=6, label="feedback-dominated |2s₂²Q|>|3s₃Γ₃|"),
                plt.Line2D([], [], color="black", lw=1.6, label="m=0 | m=1 (exact operator)"),
                plt.Line2D([], [], marker="*", color="white", mec="black", ls="", ms=9, label="gallery I–II"),
                plt.Line2D([], [], marker="o", color="white", mec="black", ls="", ms=6, label="Fig. 17 / III–V")]
    ax_sw2.legend(handles=handles, fontsize=5.3, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, framealpha=0.95)
    ax_sw2.set_xlabel("local inhibition $A_{loc}$", fontsize=8.0)
    ax_sw2.set_ylabel("firing-rate baseline $\\sigma_0$", fontsize=8.0)
    ax_sw2.tick_params(labelsize=6.8)
    row_half = int(np.argmin(np.abs(s_vals - 0.5)))
    fam_m0_half = sorted({fam_names[int(c)] for c, h in zip(regime["family"][row_half], regime["harmonic"][row_half]) if h == 0})
    ax_sw2.set_title(
        f"G. Regime map (fig17 base): full-cubic selection\n"
        f"m=0 for $A_{{loc}}$ < {a_threshold_local:.2f}: {'/'.join(fam_m0_half)} at $\\sigma_0$=½; m=1: rhombic\n"
        f"near-marginal K=0 mode: {int(np.sum(regime['near_resonant'] | regime['subcritical']))}/{regime['family'].size} cells feedback-dominated or subcritical",
        fontsize=6.8, fontweight="bold",
    )

    fig.suptitle(
        "Bressloff–Cowan–Golubitsky–Thomas–Wiener (2001/2002) V1 Euclidean Symmetry-Breaking & Form Constants\n"
        "Exact orientation operator & full cubic amplitude equations (derived contoured/non-contoured regime, roll stability vs all θ), "
        "seamless complex-log retinal pullbacks (exact-map seams quantified), paper-scale preset, sweeps & regime map, Horton–Hoyt anchors, Rule (2011) diagnostics",
        fontsize=10.5,
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
        "runs_parity_ok": runs_ok,
        "max_run_err": max_run_err,
        "calibration_rows": cal_rows,
        "run_rows": run_rows,
        "fixture_checks": fixture_summary,
        "negative_control": neg_ctrl,
        "registry_control": registry_ctrl,
        "gallery": rendered_gallery,
        "gallery_selected_count": n_selected,
        "gallery_stable_coexisting_count": n_stable,
        "gallery_imposed_count": n_imposed,
        "magnification": magnification,
        "scale_anchor": scale_anchor,
        "paper_scale": paper_scale,
        "scale_anchor_paper_scale": scale_anchor_paper,
        "regime_map": regime,
        "local_inhibition_threshold": a_threshold_local,
        "exact_vs_perturbative": {
            "dq": float(stab17["perturbative_vs_exact_dq"]),
            "dlambda": float(stab17["perturbative_vs_exact_dlambda"]),
        },
        "full_cubic_curves_fig17": [
            {"sigma0": s0, "gamma0": g0b, "ratio_min": float(np.nanmin(ratio))} for s0, g0b, ratio in curves_b
        ],
        "sweeps": sweeps,
        "sweep_summary": sweep_summary,
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
    """Animate the same five presets as the PNG gallery (1x2: V1 cortical torus -> exact retinal pullback).

    Differences from the static figure: the scalar field is the non-contoured envelope (the orientation
    line elements are not animated) and `uniform_phase=True` so every mode advances with the same phase
    and each preset's loop closes exactly. Wavenumber, parity, lattice, hemifield convention and map
    (conformal, full-field) are identical to the PNG; the regime (m = 0 derived for I-II) is the same
    because `stability_scan` derives it before the envelope flag is applied. Each frame carries the
    SELECTED / STABLE / IMPOSED badge of the full cubic amplitude-equation selection.
    """
    from PIL import Image

    presets = [
        ("roll_noncontoured_derived", "1/5  Tunnel / Rings", "vertical V1 stripes, m=0 derived"),
        ("spiral_noncontoured_derived", "2/5  Spiral / Funnel", "oblique V1 stripes, m=0 derived"),
        ("fig31_square_even", "3/5  Cobweb / Square", "orthogonal V1 square lattice"),
        ("fig35_hex_zero_even", "4/5  Honeycomb / Hex", "three-wave V1 hex lattice"),
        ("fig33_rhombic_even", "5/5  Rhombic lattice", "two-wave V1 rhombic lattice"),
    ]
    params_list = [(pid, lbl, sub, make_preset_params(pid, n=72, m=8)) for pid, lbl, sub in presets]
    frames_rgb: List[Image.Image] = []
    frames_per_preset = max(1, n_frames // len(params_list))

    for fi in range(n_frames):
        idx = min(fi // frames_per_preset, len(params_list) - 1)
        local_fi = fi - idx * frames_per_preset
        phase = 2.0 * math.pi * float(local_fi) / float(frames_per_preset)
        _pid, lbl, sub, p = params_list[idx]
        if p.pattern == "spiral":
            sub = f"{sub} ({p.pattern_angle_deg:.0f}°)"

        # Non-contoured spatial envelope + uniform phase sweep for smooth, seamless motion; same
        # q*-derived wavenumber, parity and lattice as the PNG gallery.
        cort, ret, stab_anim = render_cortical_and_retinal_planform(
            p, phase=phase, retinal_res=140, uniform_phase=True, envelope_only=True
        )
        bsel = stab_anim["branch_selection"]
        fam = _branch_target_lattice(stab_anim["rendered_pattern"])
        fam = "roll" if fam == "global" else fam
        badge, badge_color = lattice_badge(lattice_consistency_for(fam, bsel), bsel)

        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.5), dpi=100, facecolor="#fafaf8")
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.09, top=0.80, wspace=0.10)

        axes[0].imshow(cort, cmap="twilight_shifted", vmin=-1.0, vmax=1.0, origin="lower", interpolation="bilinear")
        axes[0].set_title(f"V1 Cortex: {sub}", fontsize=9.5, fontweight="bold", color="#1e293b", pad=6)
        axes[0].axis("off")

        axes[1].set_facecolor("#fafaf8")
        axes[1].imshow(ret, cmap="twilight_shifted", vmin=-1.0, vmax=1.0, origin="lower", interpolation="bilinear")
        axes[1].set_title(
            f"Retina ({stab_anim['hemifield_mode']}, |r| ≤ {stab_anim['retinal_max_deg']:.0f}°)",
            fontsize=9.5, fontweight="bold", color="#1e293b", pad=6,
        )
        axes[1].axis("off")
        fig.text(
            0.5, 0.02, badge, ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=badge_color, ec="none"),
        )

        _cmode = stab_anim.get("rendered_contour_mode", "noncontoured")
        fig.suptitle(
            f"Bressloff et al. (2001) — {lbl}  (k*={stab_anim['rendered_cycles']} cyc, "
            f"{'non-contoured' if _cmode == 'noncontoured' else 'contoured'})",
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
    n_strip = len(params_list)
    strip = Image.new("RGB", (w, h * n_strip))
    for i, fr in enumerate(frames_rgb[:: max(1, len(frames_rgb) // n_strip)][:n_strip]):
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
    runs_ok, max_run_err, run_rows = verify_paper_calibration_runs()
    n_avail = sum(1 for r in run_rows if r["preset_available"])
    check(
        runs_ok and n_avail == len(run_rows),
        f"Exact parity with all {len(run_rows)} preset runs in paper-calibration.json runs[] "
        f"(q*, branch, selected family/pattern/scope, rendered pattern; max |Δq*| = {max_run_err:.2e})",
    )
    fx = summarize_fixture_checks(rows, run_rows)
    print(
        f"[INFO] upstream fixture paper-expectation checks: {fx['passed']}/{fx['total']} passed "
        f"(stability_reports {fx['stability_reports']['passed']}/{fx['stability_reports']['total']}, "
        f"runs {fx['runs']['passed']}/{fx['runs']['total']}; run status {fx['runs']['status_counts']})"
        + "".join(
            f"\n       ✗ {f['report']}/{f['name']}: expected {f['expected']!r}, actual {f['actual']!r}"
            for f in fx["failed"]
        )
    )
    check(
        any(f["report"] == "fig40_odd_hex_bifurcation" and f["name"] == "selected-pattern" for f in fx["failed"]),
        "Known fig40 odd-hex H0/Hπ degeneracy is surfaced (not hidden) by the parity report",
    )
    check(
        fx["runs"]["total"] >= 100 and len(fx["runs"]["failed"]) >= 10,
        f"runs[].checks are counted in full ({fx['runs']['total']} checks, {len(fx['runs']['failed'])} failures, "
        f"{len(fx['runs']['review_runs'])} runs not 'pass') — previously under-disclosed",
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
    reg = load_registry_negative_control()
    check(
        reg["control_found"] and "tamekue" in reg["control_source_key"],
        f"driven-neural-fields-registry.json is read, not just checksummed: {reg['example_count']} examples, "
        f"control '{reg['control_id']}' from {reg['control_source_key']} ({reg['control_kernel']})",
    )

    # Orientation field + tangent map: a cortical roll with normal along x (rings) has orientation
    # preference pi/2 everywhere, so in the visual field phi_R = pi/2 + theta_R is tangent to circles.
    p_rings = make_preset_params("fig7_lattice_tunnel", n=48, m=12)
    _c, _r, st_rings = render_cortical_and_retinal_planform(p_rings, retinal_res=96)
    orient_r = st_rings["orientation_retinal"]
    u = st_rings["retinal_grid_deg"]
    ug, vg = np.meshgrid(u, u)
    theta_r = np.arctan2(vg, ug)
    radial_component = np.abs(np.cos(orient_r - theta_r))  # 0 = tangential, 1 = radial
    ring_mask = np.isfinite(orient_r) & (np.hypot(ug, vg) > 0.3 * u.max())
    bimodal = float(np.mean((radial_component[ring_mask] < 0.3) | (radial_component[ring_mask] > 0.95)))
    frac_tangential = float(np.mean(radial_component[ring_mask] < 0.3))
    check(
        st_rings["orientation_cortical"] is not None and bimodal > 0.9 and 0.3 < frac_tangential < 0.7,
        f"Orientation field for the contoured even roll: line elements alternate tangential / radial between "
        f"adjacent rings (argmax_φ a with the exact u(φ): {100 * bimodal:.0f}% within 0.3 of one of the two, "
        f"{100 * frac_tangential:.0f}% tangential) under the tangent map φ_R = φ + θ_R",
    )

    # Retinal pullback of the derived spiral: the conformal (2001 complex-log) map closes the torus, so the
    # full-field pullback is seamless AND chiral; the exact (2002) map leaves a left-meridian seam in
    # full-field, which chiral-mirror moves (halved) to the vertical meridians; bilateral-mirror is
    # seamless but destroys chirality. All quantified, not assumed.
    p_sp = make_preset_params("spiral_noncontoured_derived", n=48, m=12)
    _c, ret_conf, st_conf = render_cortical_and_retinal_planform(p_sp, retinal_res=160)
    d_conf = retinal_pullback_diagnostics(ret_conf)
    d_by_mode: Dict[str, Dict[str, float]] = {}
    for hm_mode in ("full-field", "chiral-mirror", "bilateral-mirror"):
        p_h = BressloffParams(**{**p_sp.__dict__})
        p_h.hemifield_mode = hm_mode
        p_h.retinal_angular_map = "exact"
        _c, ret_h, st_h = render_cortical_and_retinal_planform(p_h, retinal_res=160)
        d_by_mode[hm_mode] = retinal_pullback_diagnostics(ret_h)
    d_ch, d_ff, d_bm = d_by_mode["chiral-mirror"], d_by_mode["full-field"], d_by_mode["bilateral-mirror"]
    check(
        st_conf["retinal_angular_map"] == "conformal" and d_conf["chirality"] > 0.3 and d_conf["seam_contrast"] < 0.05
        and d_ff["seam_h_left"] > 1.0 and d_ch["seam_h_left"] < 0.05 and d_ch["seam_contrast"] > 1.0
        and d_bm["mirror_symmetry_lr"] > 0.99
        and all(math.isfinite(d[k]) for d in [d_conf, *d_by_mode.values()] for k in ("seam_h_left", "seam_h_right", "seam_v_top", "seam_v_bottom")),
        f"Spiral pullback: conformal map seam {d_conf['seam_contrast']:.3f} with chirality {d_conf['chirality']:.2f} (rendered); exact map: "
        f"full-field left-meridian seam {d_ff['seam_h_left']:.2f} (earlier versions), chiral-mirror moves it to the vertical meridians "
        f"(left {d_ch['seam_h_left']:.3f}, max {d_ch['seam_contrast']:.2f}), bilateral-mirror chirality {d_bm['chirality']:.3f}; all four seams finite",
    )
    # Seam metric is finite for y-independent planforms (rings) too, and ~0
    p_rg = make_preset_params("roll_noncontoured_derived", n=48, m=12)
    _c, ret_rg, st_rg = render_cortical_and_retinal_planform(p_rg, retinal_res=120)
    d_rg = retinal_pullback_diagnostics(ret_rg)
    check(
        math.isfinite(d_rg["seam_contrast"]) and d_rg["seam_contrast"] < 0.05
        and all(abs(mo["n_y"] - round(mo["n_y"])) < 1e-9 for mo in st_rg["rendered_modes"]),
        f"Rings: seam metric finite and ~0 ({d_rg['seam_contrast']:.3f}; previously NaN); k_y quantised to integer torus cycles "
        f"(n_y = {[mo['n_y'] for mo in st_rg['rendered_modes']]})",
    )

    # Uniform-phase GIF loop closes exactly; stationary planforms have no per-mode phase multipliers
    p_cw = make_preset_params("fig31_square_even", n=32, m=8)
    f0, _r0, _ = render_cortical_and_retinal_planform(p_cw, phase=0.0, retinal_res=8, uniform_phase=True, envelope_only=True)
    f1, _r1, _ = render_cortical_and_retinal_planform(p_cw, phase=2.0 * math.pi, retinal_res=8, uniform_phase=True, envelope_only=True)
    g1, _, _ = render_cortical_and_retinal_planform(p_cw, phase=2.0 * math.pi, retinal_res=8, uniform_phase=False, envelope_only=True)
    check(
        float(np.max(np.abs(f1 - f0))) < 1.0e-9 and float(np.max(np.abs(g1 - f0))) < 1.0e-9
        and all(ps == 1.0 for _a, ps, _o, _amp in planform_modes(p_cw, "cobweb") + planform_modes(p_cw, "rhombic")),
        f"GIF phase loop closes (max |Δ| = {float(np.max(np.abs(f1 - f0))):.1e}); square/rhombic lattices are stationary "
        f"(no per-mode phase multipliers 0.35 / -0.25)",
    )

    # ---- Full cubic amplitude equations (Pass 4) ----
    p17s = make_preset_params("fig17_even")
    ex17 = exact_dispersion_scan(p17s)
    b17 = branch_selection_full(p17s, ex17)
    bu17 = branch_selection_upstream(p17s, b17["q_mode"], ex17["critical_growth"])
    r_full = [b17["gamma_hex"] / b17["gamma0"], b17["gamma_square"] / b17["gamma0"]]
    r_up = [bu17["gamma_hex"] / bu17["gamma0"], bu17["gamma_square"] / bu17["gamma0"]]
    check(
        abs(b17["s2"]) < 1e-12 and max(abs(a - b) for a, b in zip(r_full, r_up)) < 2.0e-3,
        f"At σ0 = ½ (s2 = 0) the full cubic ratios reduce to the bare overlap ratios: γ(60)/γ(0), γ(90)/γ(0) = "
        f"{r_full[0]:.4f}, {r_full[1]:.4f} (full) vs {r_up[0]:.4f}, {r_up[1]:.4f} (upstream, truncated m≤4 eigenfunction)",
    )
    st17 = stability_scan(p17s)
    check(
        abs(st17["perturbative_vs_exact_dlambda"]) < 1.0e-3 and abs(st17["perturbative_vs_exact_dq"]) < 0.05
        and st17["critical_harmonic_exact"] == 1,
        f"Exact orientation operator (m ≤ {ORIENT_HARMONICS}) agrees with the perturbative n=1 scan at fig17: |Δλ| = "
        f"{abs(st17['perturbative_vs_exact_dlambda']):.1e}, |Δq*| = {abs(st17['perturbative_vs_exact_dq']):.1e}; critical harmonic m=1 (contoured)",
    )
    subcrit_sigmas = []
    for s0 in (0.5, 0.4, 0.3, 0.25, 0.2):
        cub = full_cubic_coefficients(p17s, b17["q_mode"], "even", 1, sigma0=s0)
        if cub["gamma"][0] <= 0.0:
            subcrit_sigmas.append(s0)
        pfull = BressloffParams(**{**p17s.__dict__})
        pfull.sigmoid_baseline = s0
        bf = branch_selection_full(pfull, ex17)
        if s0 >= 0.4:
            check(
                not bf["roll_stable"] and not bf["square_stable"] and bf["unrestricted_selected_family"] == "rhombic",
                f"  m=1 regime, σ0={s0:g}: quadratic feedback does not rescue rolls/squares; unrestricted selection = "
                f"{bf['unrestricted_selected_family']}@{bf['rhombic_theta_optimal_deg']:.0f}° (γ(0)={bf['gamma0']:+.4f}, "
                f"roll min margin {bf['roll_min_margin']:+.2f})",
            )
    check(
        subcrit_sigmas and max(subcrit_sigmas) <= 0.3 and 0.2 in subcrit_sigmas,
        f"Cubic truncation becomes subcritical (γ(0) ≤ 0) for σ0 ≤ {max(subcrit_sigmas) if subcrit_sigmas else float('nan'):g} "
        f"(s3 < 0 below σ0 = 0.211, plus feedback) — disclosed, quintic terms not implemented",
    )
    # Derived non-contoured regime
    p_lo = make_preset_params("fig17_even")
    thr_hits = []
    for a_loc in (0.15, 0.2, 0.24, 0.26, 0.3):
        p_lo.local_inhibition = a_loc
        thr_hits.append((a_loc, exact_dispersion_scan(p_lo)["critical_harmonic"]))
    check(
        # Boundary 0.22, not the local-only analytic W0 = W1 crossing 0.243:
        # with the conjugate-pair harmonic norm (Pass N+1) the exact operator
        # switches between A_loc = 0.20 and 0.24.
        all(h == 0 for a, h in thr_hits if a < 0.22) and all(h == 1 for a, h in thr_hits if a > 0.22),
        f"Critical harmonic switches m=0 → m=1 at A_loc ≈ 0.22 (exact operator, pair norm; "
        f"local-only W0 = W1 at 0.243): {thr_hits}",
    )
    st_der = stability_scan(make_preset_params("roll_noncontoured_derived"))
    b_der = st_der["branch_selection"]
    check(
        st_der["derived_contour_mode"] == "noncontoured" and st_der["rendered_contour_mode"] == "noncontoured"
        and not b_der["imposed_harmonic"] and b_der["roll_stable"] and b_der["roll_min_margin"] > 0.5
        and not b_der["square_stable"] and not b_der["hex_stable"] and b_der["unrestricted_selected_family"] == "roll",
        f"Derived non-contoured regime (A_loc=0.15): rolls stable against ALL θ (min 2γ/γ0−1 = {b_der['roll_min_margin']:+.2f}), "
        f"squares/hex unstable, unrestricted selection = {b_der['unrestricted_selected_family']} (tunnel & spiral SELECTED)",
    )
    st_imp = stability_scan(make_preset_params("fig29_roll_noncontoured"))
    check(
        st_imp["branch_selection"]["imposed_harmonic"] and st_imp["branch_selection"]["parity_imposed_subcritical"]
        and st_imp["derived_contour_mode"] == "contoured",
        f"Fixture preset fig29_roll_noncontoured: imposed m=0 is flagged (critical harmonic m="
        f"{st_imp['critical_harmonic_exact']}, imposed branch subcritical by Δλ = "
        f"{st_imp['critical_growth_exact'] - st_imp['branch_selection']['lambda_eigen']:.3f})",
    )
    # Hexagon thresholds with eta != 0 (sigma0 != 1/2, even m=1 branch): Lambda_r < Lambda_h
    p_eta = make_preset_params("fig17_even")
    p_eta.sigmoid_baseline = 0.4
    b_eta = branch_selection_full(p_eta, ex17)
    check(
        abs(b_eta["eta"]) > 1e-6 and math.isfinite(b_eta["lambda_r"]) and math.isfinite(b_eta["lambda_h"])
        and 0.0 < b_eta["lambda_r"] < b_eta["lambda_h"],
        f"η ≠ 0 at σ0=0.4 (η={b_eta['eta']:+.4f}): roll threshold Λ_r={b_eta['lambda_r']:.3g} < hexagon upper threshold "
        f"Λ_h={b_eta['lambda_h']:.3g} (bistability window), hex pattern {b_eta['selected_pattern'] if b_eta['selected_family']=='hexagonal' else ('hex_pi' if b_eta['eta']<0 else 'honeycomb')}",
    )
    # Paper-scale preset reaches the paper's stripe count
    p_ps = make_preset_params("paper_scale_tunnel", n=192, m=8)
    _c, ret_ps, st_ps = render_cortical_and_retinal_planform(p_ps, retinal_res=120)
    mo_ps = retinal_morphometrics(ret_ps, st_ps)
    check(
        4.0 <= st_ps["rendered_q"] <= 5.0 and 2.4 <= st_ps["critical_wavelength_mm"] <= 3.2
        and mo_ps["rings_hh_in_paper_range"] and mo_ps["rings_fovea_to_90"] < 30.0,
        f"paper_scale_tunnel: q* = {st_ps['rendered_q']:.2f}, λ_c = {st_ps['critical_wavelength_mm']:.2f} mm (paper 2.4–3.2 ✓); "
        f"ring count 0→90°: {mo_ps['rings_fovea_to_90']:.1f} on the α=3/π map ({mo_ps['model_extent_0_90_mm']:.0f} mm) vs "
        f"{mo_ps['rings_fovea_to_90_hh']:.1f} on the Horton–Hoyt map ({mo_ps['hh_extent_0_90_mm']:.0f} mm; paper 30–36) — "
        f"the residual shortfall is the upstream α, not q*",
    )
    # Morphometrics: spiral pitch ~ requested angle, ring count ~ analytic
    p_sp2 = make_preset_params("spiral_noncontoured_derived", n=96, m=8)
    _c, ret_sp2, st_sp2 = render_cortical_and_retinal_planform(p_sp2, retinal_res=160)
    mo_sp2 = retinal_morphometrics(ret_sp2, st_sp2)
    mo_rg = retinal_morphometrics(ret_rg, st_rg)
    check(
        abs(mo_sp2["spiral_pitch_deg"] - 45.0) < 3.0 and abs(mo_rg["rings_counted"] - mo_rg["rings_analytic"]) <= 1.5,
        f"Morphometrics: spiral pitch {mo_sp2['spiral_pitch_deg']:.2f}° (requested 45°, k_y quantised, n_y={mo_sp2['first_mode_ny']:.0f}); "
        f"rings counted {mo_rg['rings_counted']:.1f} vs analytic {mo_rg['rings_analytic']:.1f}",
    )

    # Paper-scale anchors are evaluated (and the disagreement disclosed, not hidden)
    sc = compare_paper_scale_anchors(st_rings)
    check(
        math.isfinite(sc["wavelength_ratio_model_over_paper"]) and sc["wavelength_ratio_model_over_paper"] > 1.0
        and abs(sc["foveal_magnification_rel_error"]) < 0.1,
        f"Paper-scale anchors evaluated: foveal magnification {sc['foveal_magnification_model']:.2f} vs 11.5 mm/deg "
        f"({100.0 * sc['foveal_magnification_rel_error']:+.1f}%); λ_c {sc['critical_wavelength_model_mm']:.1f} mm vs paper 2.4–3.2 mm "
        f"({sc['wavelength_ratio_model_over_paper']:.1f}× too long, reported as a discrepancy); rendered {sc['rendered_cycles']} cyc vs 30–36 stripes",
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
            f"All {len(ks)} linear planforms have FFT rings at the q*-derived cycle count: k* = {ks}, rendered = {cyc} "
            f"(contoured-rendering k* = {ks_c})",
        )
        hex_g = next(g for g in res["gallery"] if g["rendered_family"] == "hexagonal")
        hm = hex_g["harmonics"]
        k_lin_h = hex_g["spectral_linear"]["k_star_cycles"]
        k_con_h = hex_g["spectral"]["k_star_cycles"]
        check(
            (abs(k_con_h - 2.0 * k_lin_h) <= 1.5 or abs(k_con_h - math.sqrt(3.0) * k_lin_h) <= 1.5)
            and max(hm["power_at_2k"], hm["power_at_sqrt3_k"]) >= 0.9 * hm["power_at_k"],
            f"Hex contoured FFT peak k*={k_con_h:.0f} is a harmonic ring (√3k = {math.sqrt(3.0) * k_lin_h:.1f} or 2k = {2 * k_lin_h:.0f}) of "
            f"tanh∘argmax_φ, not the fundamental (power at k/√3k/2k = {hm['power_at_k']:.2f}/{hm['power_at_sqrt3_k']:.2f}/{hm['power_at_2k']:.2f})",
        )
        classes = [g["spectral"]["lattice_class"] for g in res["gallery"]]
        check(
            classes[:4] == ["roll", "roll", "square", "hexagonal"],
            f"Normalised azimuthal discriminator classifies rings/spiral as roll, cobweb as square, honeycomb as hexagonal: {classes}",
        )
        cons = dict((g["preset_id"], g["lattice_consistency"]) for g in res["gallery"])
        check(
            cons["roll_noncontoured_derived"] == "selected" and cons["spiral_noncontoured_derived"] == "selected"
            and cons["fig33_rhombic_even"] == "selected"
            and cons["fig35_hex_zero_even"] == "stable-coexisting"
            and cons["fig31_square_even"] == "imposed-unstable",
            f"Gallery lattice consistency labelled per panel: {cons} "
            f"(derived-regime tunnel/spiral SELECTED, rhombic selected, hex stable-coexisting, square imposed-unstable)",
        )
        sw = res["sweep_summary"]
        lat_sweeps = ("lateral_wide_sigma", "lateral_inhibition", "lateral_spread_deg")
        check(
            not any(sw[k]["roll_ever_stable"] or sw[k]["square_ever_stable"] for k in lat_sweeps)
            and sw["lateral_spread_deg"]["parity_flips"] >= 1
            and sw["lateral_wide_sigma"]["q_star_max"] > sw["lateral_wide_sigma"]["q_star_min"]
            and sw["local_inhibition"]["roll_ever_stable"] and sw["local_inhibition"]["harmonic_flips"] >= 1
            and set(sw["local_inhibition"]["families"]) >= {"roll", "rhombic"}
            and len(sw["sigmoid_baseline"]["near_resonant_values"]) >= 0.8 * len(res["sweeps"]["sigmoid_baseline"]) - 1,
            f"Parameter sweeps: q* falls with ξ̂_lat ({sw['lateral_wide_sigma']['q_star_max']:.2f}→"
            f"{sw['lateral_wide_sigma']['q_star_min']:.2f}), parity flips at θ₀ ≈ "
            f"{[round(v) for v in sw['lateral_spread_deg']['parity_flip_values']]}°, rolls/squares never stable in the lateral sweeps; "
            f"A_loc sweep flips the critical harmonic at {[round(v, 3) for v in sw['local_inhibition']['harmonic_flip_values']]} "
            f"and changes the selected family ({'/'.join(sw['local_inhibition']['families'])}); σ0 sweep in the m=0 regime is "
            f"near-resonant at {len(sw['sigmoid_baseline']['near_resonant_values'])}/{len(res['sweeps']['sigmoid_baseline'])} points (disclosed)",
        )
        rm = res["regime_map"]
        i_half = int(np.argmin(np.abs(np.array(rm["sigma_values"]) - 0.5)))
        a_arr = np.array(rm["a_values"])
        m0_half = rm["harmonic"][i_half] == 0
        reliable_half = ~(rm["subcritical"][i_half] | rm["near_resonant"][i_half])
        inner = m0_half & (a_arr <= 0.2)          # m=0 cells away from the W0 = W1 threshold
        fam_inner = set(int(v) for v in rm["family"][i_half][inner])
        m1_mid = (rm["harmonic"] == 1) & (np.abs(np.array(rm["sigma_values"])[:, None] - 0.5) <= 0.1 + 1e-9) & (a_arr[None, :] >= 0.5)
        check(
            bool(np.any(rm["harmonic"] == 0)) and bool(np.any(rm["harmonic"] == 1)) and bool(np.all(reliable_half))
            and fam_inner <= {rm["family_code"]["roll"], rm["family_code"]["square"]} and rm["family_code"]["roll"] in fam_inner
            and bool(np.all(rm["family"][m1_mid] == rm["family_code"]["rhombic"]))
            and bool(np.all(~(rm["subcritical"] | rm["near_resonant"])[m1_mid])),
            f"Regime map A_loc × σ0: both harmonics present; at σ0 = ½ every cell is reliable and m=0 cells with A_loc ≤ 0.2 select rolls or "
            f"squares (non-contoured Fig. 29 planforms); m=1 cells with A_loc ≥ 0.5, |σ0 − ½| ≤ 0.1 are reliable and select rhombic; "
            f"{int(np.sum(rm['subcritical'] | rm['near_resonant']))}/{rm['subcritical'].size} cells subcritical or feedback-dominated (hatched/dotted)",
        )
        ps = res["paper_scale"]
        check(
            2.4 <= ps["stability"]["critical_wavelength_mm"] <= 3.2 and ps["morphometrics"]["rings_hh_in_paper_range"]
            and math.isfinite(ps["diag"]["seam_contrast"]),
            f"Paper-scale preset in the figure: λ_c {ps['stability']['critical_wavelength_mm']:.2f} mm, "
            f"{ps['morphometrics']['rings_fovea_to_90']:.1f} rings 0→90° (model map) / {ps['morphometrics']['rings_fovea_to_90_hh']:.1f} (H&H map; 30–36), "
            f"seam {ps['diag']['seam_contrast']:.2f}",
        )
        check(
            not res["floquet_status"]["calibration_claim_allowed"] and res["floquet_overlap_rms"] > 0.1
            and len(res["floquet_status"]["forcing_amplitudes"]) == 5,
            f"Rule (2011) overlay is reported as diagnostic only (status={res['floquet_status']['acceptance_status']}, "
            f"RMS β err={res['floquet_overlap_rms']:.3f}, {res['floquet_status']['boundary_curve_count']} curves over "
            f"{len(res['floquet_status']['forcing_amplitudes'])} amplitudes, no solver)",
        )
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / "bressloff_v1_form_constants_morphospace.png"))
    parser.add_argument("--csv", default=str(REPO_ROOT / "bressloff_v1_metrics.csv"))
    parser.add_argument("--gif", nargs="?", const=str(DEFAULT_GIF_PATH), help=f"Write the animation (default path: {DEFAULT_GIF_PATH.relative_to(REPO_ROOT)})")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive matplotlib window")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic self-test and exit")
    parser.add_argument(
        "--sweep", metavar="PARAM[:LO:HI[:N]]", default=None,
        help="Sweep one coupling parameter of --preset, print the table and write it to --csv if given explicitly, "
             f"else <csv stem>_sweep_<PARAM>.csv; PARAM in {sorted(SWEEPABLE_PARAMS)}. Skips the figure unless --out is also given explicitly.",
    )
    parser.add_argument("--preset", default="fig17_even", help=f"Base paper preset for --sweep ({', '.join(PAPER_PRESETS)})")
    parser.add_argument(
        "--hemifield-mode", choices=("full-field", "chiral-mirror", "bilateral-mirror"), default=None,
        help="Override the hemifield convention of the retinal pullback for the gallery/GIF (default: full-field)",
    )
    parser.add_argument(
        "--retinal-map", choices=("conformal", "exact"), default=None,
        help="Angular coordinate of the retino-cortical map for the pullback: conformal (2001 complex-log, seamless; default) or exact (2002; seams)",
    )
    parser.add_argument(
        "--sigmoid-baseline", type=float, default=None,
        help="Override the firing-rate baseline sigma0 for every preset (0.5 = inflection point, no quadratic feedback)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    if args.hemifield_mode:
        for preset in PAPER_PRESETS.values():
            preset["hemifield_mode"] = args.hemifield_mode
    if args.sigmoid_baseline is not None:
        for preset in PAPER_PRESETS.values():
            preset["sigmoid_baseline"] = args.sigmoid_baseline
    if args.retinal_map:
        for preset in PAPER_PRESETS.values():
            preset["retinal_angular_map"] = args.retinal_map

    if args.sweep:
        parts = args.sweep.split(":")
        name = parts[0]
        if name not in SWEEPABLE_PARAMS:
            parser.error(f"unknown sweep parameter {name!r}; choose from {sorted(SWEEPABLE_PARAMS)}")
        lo, hi, _ = SWEEPABLE_PARAMS[name]
        n_pts = 25
        if len(parts) >= 3:
            lo, hi = float(parts[1]), float(parts[2])
        if len(parts) >= 4:
            n_pts = int(parts[3])
        rows = parameter_sweep(name, np.linspace(lo, hi, n_pts), base_preset=args.preset)
        explicit_csv = any(a.startswith("--csv") for a in (argv if argv is not None else sys.argv[1:]))
        sweep_csv = Path(args.csv) if explicit_csv else Path(args.csv).with_name(f"{Path(args.csv).stem}_sweep_{name}.csv")
        write_sweep_csv(rows, sweep_csv)
        print(f"{'value':>10} {'q*':>8} {'branch':>6} {'m':>2} {'λ_c mm':>8} {'cycles':>6} {'unrestricted':>12} {'θ_rh*':>6} {'rollmin':>8} roll sq rh hex")
        for r in rows:
            print(
                f"{r['value']:10.4f} {r['critical_q_exact']:8.4f} {r['critical_branch']:>6} {r['critical_harmonic']:2d} {r['critical_wavelength_mm']:8.2f} "
                f"{r['rendered_cycles']:6d} {r['unrestricted_selected_family']:>12} {r['rhombic_theta_optimal_deg']:6.1f} {r['roll_min_margin']:+8.3f} "
                f"{int(r['roll_stable'])}    {int(r['square_stable'])}  {int(r['rhombic_stable'])}  {int(r['hex_stable'])}"
                + ("  SUBCRITICAL" if r["subcritical_cubic"] else "")
            )
        print(f"Saved sweep table to {sweep_csv}")
        explicit_out = any(a.startswith("--out") for a in (argv if argv is not None else sys.argv[1:]))
        if not explicit_out:
            return 0

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
