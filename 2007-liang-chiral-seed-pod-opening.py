#!/usr/bin/env python3
"""
2007-liang-chiral-seed-pod-opening.py — Pass 3 (v3) Remediated Implementation
=============================================================================
Faithful mathematical, biomechanical, and morphometric reproduction of:

  1. Armon S, Efrati E, Kupferman R, Sharon E (2011).
     "Geometry and Mechanics in the Opening of Chiral Seed Pods."
     Science 333(6050):1726-1730.
     DOI: 10.1126/science.1203874
  2. Efrati E, Sharon E, Kupferman R (2009).
     "Elastic Theory of Unconstrained Non-Euclidean Plates."
     Journal of the Mechanics and Physics of Solids 57(4):762-775.
     DOI: 10.1016/j.jmps.2008.12.004
  3. Armon S, Arbel H, Moshe M, Sharon E (2014).
     "Shape Selection in Chiral Self-Shaping Latex Strips."
     Soft Matter 10(16):2733-2740.
     DOI: 10.1039/c3sm52313f
  4. Forterre Y, Skotheim JM, Dumais J, Mahadevan L (2005).
     "How the Venus Flytrap Snaps."
     Nature 433(7024):421-425.
     DOI: 10.1038/nature03185

Mathematical & Biomechanical Architecture (Linked to Lean 4 `MathOfBiology.Loop_065`):
-------------------------------------------------------------------------------------
  - Law 1 (`MathOfBiology.Loop_065.armon_gauss_saddle_incompatibility`):
    In *Bauhinia variegata* (Orchid tree) seed pod valves, two fibrous sclerenchyma
    layers with orthogonal +-45 deg cellulose microfibril orientation shrink
    perpendicularly to their fibers upon desiccation, prescribing a saddle target
    second fundamental form b_0 = diag(kappa_0, -kappa_0) in the principal frame.
    Its trace vanishes (Tr(b_0) = 0) and its determinant is strictly negative:
      K_target = det(b_0) = -kappa_0^2 < 0 != K(a_0) = 0,
    proving by Gauss's Theorema Egregium that no zero-energy isometric immersion
    can simultaneously realize the flat metric a_0 = I and saddle curvature b_0.

  - Law 2 (`MathOfBiology.Loop_065.armon_narrow_helicoid_chirality`):
    Rotating b_0 = diag(kappa_0, -kappa_0) into strip coordinates (s, t) cut at
    angle theta relative to the principal +kappa_0 axis yields:
      b_0(theta) = [[kappa_0 cos(2*theta), kappa_0 sin(2*theta)],
                    [kappa_0 sin(2*theta), -kappa_0 cos(2*theta)]].
    For narrow strips (dimensionless Föppl-von Kármán width w_tilde = W*sqrt(kappa_0/h)
    <= w_tilde_c), bending dominates stretching, mean curvature vanishes (H* = 0),
    and at the natural *Bauhinia* angle theta = 45 deg the strip forms a Pure-Twist
    Helicoid (helix radius R = 0, geodesic torsion tau = D*sin(2*theta) maximized).
    At theta = 0 deg or theta = 90 deg, tau = 0 vanishes identically, forming an
    achiral circular ring with zero helical pitch (P = 0).

  - Law 3 (`MathOfBiology.Loop_065.armon_wide_cylindrical_ribbon_asymptote`):
    Above the critical dimensionless width:
      w_tilde_c(nu) = (16 * (1 + nu) / sqrt(1 - nu^2))^(1/4),
    Föppl-von Kármán stretching energy proportional to w_tilde^4 * (H^2 - D^2)^2
    destabilizes the saddle helicoid (H = 0) via a supercritical pitchfork
    bifurcation into two mirror-symmetric Cylindrical Helical Ribbon branches:
      H*(w_tilde) = +- ((1 - nu)/2) * kappa_0 * sqrt(1 - (w_tilde_c / w_tilde)^4),
      D*(w_tilde) = ((1 - nu)/2) * kappa_0,
    approaching an isometric cylinder (K = H^2 - D^2 -> 0, 2*H_inf = (1 - nu)*kappa_0)
    as w_tilde -> infinity.

  - Law 4, Pass 3 (`MathOfBiology.Loop_065.freund_crack_speed_arrest_and_monotonicity`):
    The quasi-static minimizer supplies the suture energy release rate
      G = B_eff (kappa_0^2 - e_final) W / h,  B_eff = Y h^3 / (12 (1 + nu)),
    and Freund's (1990) mode-I crack equation of motion gives the transient crack speed
      da/dt = c_R max(0, 1 - Gamma_c(a) / G(t))          (arrest when Gamma_c >= G, v < c_R)
    along the suture, driven by desiccation kappa_0(t) = kappa_0,f (1 - exp(-t/tau_dry))
    -> Griffith initiation time, stick-slip jumps, opening fraction.
    (Reduced 1-D model; see Section 4b and datasets/.../suture_fracture_ballistic_reference.json.)

  - Law 5, Pass 3 (`MathOfBiology.Loop_065.elastic_catapult_energy_partition`):
    Released energy Delta_U_seg = B_eff (kappa_0^2 - e_final) W l_seg >= 0 is converted with
    efficiency eta into kinetic energy of the valve segment + seed:
      v_seed = sqrt(2 eta Delta_U_seg / (m_seg/3 + m_seed)),
    then flown with quadratic drag to obtain the ballistic range (Vogel 2005; benchmarks
    Hofhuis et al. 2016 Cardamine ~10 m/s, Swaine & Beer 1977 Hura ~70 m/s, transfer boundary).
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
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import optimize, signal, stats

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "chiral_seed_pod_opening"
REF_JSON = DATASET_DIR / "chiral_seed_pod_opening_reference.json"
SPECIMEN_JPG = DATASET_DIR / "bauhinia_chiral_seed_pod_specimen.jpg"
SCOUT_JSON = DATASET_DIR / "scout_manifest.json"
BIB_FILE = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "chiral_seed_pod_opening_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "chiral_seed_pod_opening_growth.gif"
OUT_FRACTURE_PNG = ROOT / "docs" / "media" / "chiral_seed_pod_opening_suture_fracture.png"
OUT_FRACTURE_GIF = ROOT / "docs" / "media" / "chiral_seed_pod_opening_suture_fracture.gif"

# Cryptographic SHA-256 Registry for `datasets/chiral_seed_pod_opening/`
EXPECTED_SHA256: dict[str, str] = {
    "datasets/chiral_seed_pod_opening/chiral_seed_pod_opening_reference.json": (
        "4b1fa5df06327c65e841f6333d69f071478068db01d4a0c000c6162891042057"
    ),
    "datasets/chiral_seed_pod_opening/bauhinia_chiral_seed_pod_specimen.jpg": (
        "047f357961bc58c5612bcfc3b769db407e99bcda7106765a9d8c24b659a64ec4"
    ),
    "datasets/chiral_seed_pod_opening/scout_manifest.json": (
        "05711e5e7df6cb173f02c9193d1e18339d4993c8e446e13a331fbf875a649262"
    ),
    "datasets/chiral_seed_pod_opening/paper_citations.bib": (
        "41f9ca2ead3c8d74f3120a52a8d785677df58380a757703d9ba0812fb681aa4c"
    ),
    # Pass 3: suture fracture / ballistic ejection parameter table (provenance-flagged)
    "datasets/chiral_seed_pod_opening/suture_fracture_ballistic_reference.json": (
        "fc8fab08281dd6e00aea8ec5b40244d4d6580c252cb541a182aa06bfce702639"
    ),
}


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
    """Verify SHA-256 digests of all 5 files in datasets/chiral_seed_pod_opening/."""
    computed: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required dataset asset: {p}")
        digest = sha256_file(p)
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {digest}")
        computed[rel] = digest
    return computed


def load_reference_data() -> dict[str, Any]:
    """Load and cryptographically verify `chiral_seed_pod_opening_reference.json`."""
    verify_dataset_integrity()
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


# =============================================================================
# 2. Incompatible Saddle Curvature Tensor & Sclerenchyma Bilayer Mechanics
# =============================================================================
def compute_sclerenchyma_bilayer_kappa0(
    delta_eps_hygro: float = 0.0266,
    h_valve_cm: float = 0.095,
) -> float:
    """Armon et al. (2011) Science p. 1726-1727: Symmetric orthogonal +-45 deg
    sclerenchyma bilayer differential hygroscopic shrinkage strain delta_eps_hygro
    across total valve thickness h_valve_cm prescribes principal intrinsic curvature:
      kappa_0 = (3 / 2) * delta_eps_hygro / h_valve_cm (cm^-1).
    Linked to Lean 4 `MathOfBiology.Loop_065.armon_gauss_saddle_incompatibility`."""
    if delta_eps_hygro <= 0.0 or h_valve_cm <= 0.0:
        return 0.0
    return float(1.5 * delta_eps_hygro / h_valve_cm)


def compute_target_curvature_tensor(
    kappa0: float = 0.42,
    theta_deg: float = 45.0,
) -> dict[str, Any]:
    """Compute the rotated 2x2 target second fundamental form b_0(theta) in strip
    coordinates (s, t) at cutting angle theta relative to the +kappa_0 principal axis:
      b_0(theta) = [[kappa_0 cos(2*theta), kappa_0 sin(2*theta)],
                    [kappa_0 sin(2*theta), -kappa_0 cos(2*theta)]].
    Verifies Gauss's Theorema Egregium incompatibility:
      Tr(b_0) = 0,  det(b_0) = -kappa_0^2 < 0 != K(a_0) = 0.
    Linked to Lean 4 `MathOfBiology.Loop_065.armon_gauss_saddle_incompatibility`."""
    two_theta = math.radians(2.0 * theta_deg)
    c2 = math.cos(two_theta)
    s2 = math.sin(two_theta)
    b0 = np.array(
        [
            [kappa0 * c2, kappa0 * s2],
            [kappa0 * s2, -kappa0 * c2],
        ],
        dtype=float,
    )
    trace_b0 = float(np.trace(b0))
    det_b0 = float(np.linalg.det(b0))
    return {
        "theta_deg": float(theta_deg),
        "kappa0": float(kappa0),
        "b0_matrix": b0,
        "kappa_s0": float(b0[0, 0]),
        "kappa_t0": float(b0[1, 1]),
        "tau_0": float(b0[0, 1]),
        "trace_b0": trace_b0,
        "det_b0_gaussian_K": det_b0,
        "gauss_incompatible": bool(det_b0 < -1e-12),
    }


def critical_fvk_width(nu: float = 0.30) -> float:
    """Armon et al. (2011) Science Eq. 3 & SOM: Exact dimensionless Föppl-von Kármán
    critical width w_tilde_c for the supercritical pitchfork bifurcation from a
    Pure-Twist Helicoid (H = 0) to a Cylindrical Helical Ribbon (H != 0):
      w_tilde_c(nu) = (16 * (1 + nu) / sqrt(1 - nu^2))^(1/4)."""
    return float(((16.0 * (1.0 + nu)) / math.sqrt(max(1.0 - nu * nu, 1e-12))) ** 0.25)


def dimensionless_fvk_width(W_cm: float, kappa0_cm_inv: float, h_cm: float) -> float:
    """Dimensionless ribbon width w_tilde = W * sqrt(kappa_0 / h)."""
    if W_cm <= 0.0 or kappa0_cm_inv <= 0.0 or h_cm <= 0.0:
        return 0.0
    return float(W_cm * math.sqrt(kappa0_cm_inv / h_cm))


def analytical_pitchfork_curvatures(
    w_tilde: float,
    kappa0: float = 0.42,
    nu: float = 0.30,
) -> dict[str, float]:
    """Exact analytical equilibrium mean curvature H*(w_tilde) and deviatoric
    curvature D*(w_tilde) = (kappa_1 - kappa_2)/2 of the reduced non-Euclidean
    Föppl-von Kármán ribbon energy functional:
      U(H, D; w_tilde) = ((1+nu)/(1-nu))*H^2 + (D - kappa_0)^2 + Lambda(w_tilde)*(D^2 - H^2)^2,
    where Lambda(w_tilde) = (2 (1 + nu) / ((1 - nu)^3 * kappa_0^2)) * (w_tilde / w_tilde_c)^4
    (the factor 2 is what places the pitchfork exactly at w_tilde_c; it matches the code
    below — the Pass 2 docstring omitted it; note the SHA-pinned reference JSON
    `reduced_non_euclidean_energy_density` entry still shows the un-normalised form).
    - For w_tilde <= w_tilde_c: H* = 0 and D* solves (D - kappa_0) + 2*Lambda*D^3 = 0.
    - For w_tilde > w_tilde_c: D* = ((1 - nu)/2)*kappa_0 and
      H* = +- ((1 - nu)/2)*kappa_0 * sqrt(1 - (w_tilde_c / w_tilde)^4).
    Linked to Lean 4 `MathOfBiology.Loop_065.armon_wide_cylindrical_ribbon_asymptote`."""
    if kappa0 <= 0.0:
        return {
            "w_tilde": 0.0,
            "w_tilde_c": critical_fvk_width(nu),
            "H": 0.0,
            "D": 0.0,
            "K_gaussian": 0.0,
        }
    wc = critical_fvk_width(nu)
    ratio4 = (w_tilde / wc) ** 4
    lam = (2.0 * (1.0 + nu) / (((1.0 - nu) ** 3) * (kappa0**2))) * ratio4

    if w_tilde <= wc:
        H_star = 0.0
        if lam <= 1e-15:
            D_star = float(kappa0)
        else:
            # Solve 2*lam*D^3 + D - kappa0 = 0 on [0, kappa0]
            sol = optimize.brentq(lambda d: 2.0 * lam * (d**3) + d - kappa0, 0.0, kappa0)
            D_star = float(sol)
    else:
        D_star = 0.5 * (1.0 - nu) * kappa0
        H_star = D_star * math.sqrt(max(1.0 - (wc / w_tilde) ** 4, 0.0))

    return {
        "w_tilde": float(w_tilde),
        "w_tilde_c": wc,
        "H": float(H_star),
        "D": float(D_star),
        "K_gaussian": float(H_star**2 - D_star**2),
    }


# =============================================================================
# 3. Variational Non-Euclidean Incompatible Plate Energy Minimizer (L-BFGS-B)
# =============================================================================
def relax_incompatible_ribbon_equilibrium(
    w_tilde: float = 3.781,
    kappa0: float = 0.42,
    nu: float = 0.30,
    theta_deg: float = 45.0,
    branch: int = 1,
    seed: int = 2011,
) -> dict[str, Any]:
    """Minimize the 2D non-Euclidean Föppl-von Kármán incompatible shell energy
    density E[H, D, phi] = E_bend[H, D, phi] + E_stretch[H, D; w_tilde] via L-BFGS-B:
      E_bend(H, D, phi) = ((1 + nu)/(1 - nu)) * H^2 + D^2 - 2*D*kappa_0*cos(2*phi) + kappa_0^2,
      E_stretch(H, D)   = Lambda(w_tilde) * (D^2 - H^2)^2,
    where H = (kappa_1 + kappa_2)/2 is mean curvature, D = (kappa_1 - kappa_2)/2 is
    deviatoric curvature, K = H^2 - D^2 is actual Gaussian curvature, and phi is the
    principal curvature misalignment angle relative to b_0.
    Also reconstructs the transverse Föppl-von Kármán axial membrane strain profile
      eps_s(t/W) = (1/2) * (D^2 - H^2) * W^2 * ((t/W)^2 - 1/12)
    resulting from compatibility d^2(eps_s)/dt^2 = -K = D^2 - H^2."""
    wc = critical_fvk_width(nu)
    if kappa0 <= 0.0 or w_tilde <= 0.0:
        return {
            "w_tilde": float(w_tilde),
            "w_tilde_c": wc,
            "kappa0": float(kappa0),
            "nu": float(nu),
            "theta_deg": float(theta_deg),
            "H": 0.0,
            "D": 0.0,
            "phi_rad": 0.0,
            "kappa_1": 0.0,
            "kappa_2": 0.0,
            "K_gaussian": 0.0,
            "kappa_s": 0.0,
            "kappa_t": 0.0,
            "tau": 0.0,
            "R_cm": 0.0,
            "P_cm": 0.0,
            "R_kappa0": 0.0,
            "P_kappa0_over_2pi": 0.0,
            "energy_initial": 0.0,
            "energy_final": 0.0,
            "energy_drop_ratio": 0.0,
            "regime": "flat_zero_curvature_control",
        }

    rng = np.random.default_rng(seed)
    ratio4 = (w_tilde / wc) ** 4
    lam = (2.0 * (1.0 + nu) / (((1.0 - nu) ** 3) * (kappa0**2))) * ratio4
    alpha_nu = (1.0 + nu) / (1.0 - nu)

    def energy_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        H_val, D_val, phi_val = float(x[0]), float(x[1]), float(x[2])
        c2p = math.cos(2.0 * phi_val)
        s2p = math.sin(2.0 * phi_val)
        diff_sq = D_val * D_val - H_val * H_val
        e_bend = alpha_nu * (H_val**2) + (D_val**2) - 2.0 * D_val * kappa0 * c2p + (kappa0**2)
        e_stretch = lam * (diff_sq**2)
        total_e = e_bend + e_stretch

        dE_dH = 2.0 * alpha_nu * H_val - 4.0 * lam * H_val * diff_sq
        dE_dD = 2.0 * D_val - 2.0 * kappa0 * c2p + 4.0 * lam * D_val * diff_sq
        dE_dphi = 4.0 * D_val * kappa0 * s2p
        return float(total_e), np.array([dE_dH, dE_dD, dE_dphi], dtype=float)

    # Initial state: flat unbent reference plate (H=0, D=0) + small deterministic symmetry-breaking seed
    sgn = 1.0 if branch >= 0 else -1.0
    x_flat = np.array([0.0, 0.0, 0.0], dtype=float)
    e_init, _ = energy_and_grad(x_flat)

    h_guess = sgn * 0.35 * kappa0 * (1.0 + 0.01 * float(rng.standard_normal()))
    d_guess = 0.65 * kappa0 * (1.0 + 0.01 * float(rng.standard_normal()))
    phi_guess = 0.02 * float(rng.standard_normal())
    x0 = np.array([h_guess, d_guess, phi_guess], dtype=float)

    res = optimize.minimize(
        energy_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-2.0 * kappa0, 2.0 * kappa0), (0.0, 2.0 * kappa0), (-0.5 * math.pi, 0.5 * math.pi)],
        options={"ftol": 1e-15, "gtol": 1e-13, "maxiter": 400},
    )
    H_opt, D_opt, phi_opt = float(res.x[0]), float(res.x[1]), float(res.x[2])
    if abs(phi_opt) < 1e-5:
        phi_opt = 0.0
    if w_tilde <= wc and abs(H_opt) < 1e-5 * kappa0:
        H_opt = 0.0
    elif w_tilde > wc:
        H_opt = sgn * abs(H_opt)
    e_final, _ = energy_and_grad(np.array([H_opt, D_opt, phi_opt], dtype=float))

    # Strip curvature components at cutting angle theta relative to principal +kappa_0 axis
    two_theta = math.radians(2.0 * theta_deg) - 2.0 * phi_opt
    kappa_s = H_opt + D_opt * math.cos(two_theta)
    kappa_t = H_opt - D_opt * math.cos(two_theta)
    tau = D_opt * math.sin(two_theta)
    if abs(tau) < 1e-12:
        tau = 0.0

    denom = kappa_s * kappa_s + tau * tau
    if denom < 1e-15:
        R_cm = 0.0
        P_cm = 0.0
    else:
        R_cm = abs(kappa_s) / denom
        P_cm = (2.0 * math.pi * abs(tau)) / denom

    K_gauss = H_opt * H_opt - D_opt * D_opt
    regime = "pure_twist_helicoid" if w_tilde <= wc + 1e-6 else "cylindrical_helical_ribbon"

    return {
        "w_tilde": float(w_tilde),
        "w_tilde_c": wc,
        "kappa0": float(kappa0),
        "nu": float(nu),
        "theta_deg": float(theta_deg),
        "H": H_opt,
        "D": D_opt,
        "H_over_kappa0": H_opt / kappa0,
        "D_over_kappa0": D_opt / kappa0,
        "phi_rad": phi_opt,
        "kappa_1": H_opt + D_opt,
        "kappa_2": H_opt - D_opt,
        "K_gaussian": K_gauss,
        "K_over_kappa0_sq": K_gauss / (kappa0 * kappa0),
        "kappa_s": kappa_s,
        "kappa_t": kappa_t,
        "tau": tau,
        "R_cm": R_cm,
        "P_cm": P_cm,
        "R_kappa0": R_cm * kappa0,
        "P_kappa0_over_2pi": (P_cm * kappa0) / (2.0 * math.pi),
        "energy_initial": e_init,
        "energy_final": e_final,
        "energy_drop_ratio": float((e_init - e_final) / max(e_init, 1e-12)),
        "regime": regime,
    }


def generate_3d_ribbon_mesh(
    L_cm: float = 18.0,
    W_cm: float = 1.80,
    h_cm: float = 0.095,
    kappa0: float = 0.42,
    nu: float = 0.30,
    theta_deg: float = 45.0,
    branch: int = 1,
    n_s: int = 160,
    n_t: int = 21,
    seed: int = 2011,
) -> dict[str, Any]:
    """Reconstruct the 3D embedded ribbon surface r(s, t) from the L-BFGS-B relaxed
    curvatures (kappa_s, tau, kappa_t) for visualization and geometric verification."""
    w_tilde = dimensionless_fvk_width(W_cm, kappa0, h_cm)
    eq = relax_incompatible_ribbon_equilibrium(
        w_tilde=w_tilde,
        kappa0=kappa0,
        nu=nu,
        theta_deg=theta_deg,
        branch=branch,
        seed=seed,
    )
    s = np.linspace(-0.5 * L_cm, 0.5 * L_cm, n_s)
    t = np.linspace(-0.5 * W_cm, 0.5 * W_cm, n_t)
    S, T = np.meshgrid(s, t, indexing="ij")

    kappa_s = eq["kappa_s"]
    tau = eq["tau"]
    omega = math.sqrt(max(kappa_s * kappa_s + tau * tau, 1e-12))

    if omega < 1e-9:
        # Flat uncoiled strip control (kappa0 = 0)
        X = S
        Y = T
        Z = np.zeros_like(S)
    elif abs(kappa_s) < 1e-7:
        # Pure-Twist Helicoid (R = 0, H = 0, tau != 0)
        phi = tau * S
        X = T * np.cos(phi)
        Y = T * np.sin(phi)
        Z = S
    elif abs(tau) < 1e-7:
        # Achiral Circular Ring Control (theta = 0 deg or 90 deg, tau = 0, P = 0)
        R = 1.0 / abs(kappa_s)
        phi = kappa_s * S
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        Z = T
    else:
        # Cylindrical Helical Ribbon (R > 0, P > 0)
        R = eq["R_cm"]
        sin_alpha = abs(kappa_s) / omega
        cos_alpha = tau / omega
        phi = omega * S
        # Binormal + normal cross-section along helix
        e_r_x = np.cos(phi)
        e_r_y = np.sin(phi)
        b_x = cos_alpha * np.sin(phi)
        b_y = -cos_alpha * np.cos(phi)
        b_z = np.full_like(S, sin_alpha)
        # Cupping across transverse coordinate t from kappa_t
        cup = 0.5 * eq["kappa_t"] * (T**2)
        X = (R - cup) * e_r_x + T * b_x
        Y = (R - cup) * e_r_y + T * b_y
        Z = cos_alpha * S + T * b_z

    return {
        "eq": eq,
        "S": S,
        "T": T,
        "X": X,
        "Y": Y,
        "Z": Z,
    }


# =============================================================================
# 4. Pitchfork Width Sweep, Cutting-Angle Chirality Sweep & Multi-Specimen Eval
# =============================================================================
def evaluate_width_pitchfork_bifurcation(
    kappa0: float = 0.42,
    nu: float = 0.30,
    seed: int = 2011,
) -> dict[str, Any]:
    """Sweep dimensionless width w_tilde in [0.5, 10.0] across the supercritical
    pitchfork bifurcation w_tilde_c = 2.1609 and compare L-BFGS-B relaxed H/kappa_0,
    D/kappa_0, R*kappa_0, and P*kappa_0/(2*pi) against the reference benchmark table."""
    ref = load_reference_data()
    table = ref["width_bifurcation_sweep_theta_45"]

    rows: list[dict[str, Any]] = []
    H_sim, H_obs = [], []
    D_sim, D_obs = [], []
    R_sim, R_obs = [], []
    P_sim, P_obs = [], []

    for idx, item in enumerate(table):
        wt = float(item["w_tilde"])
        eq = relax_incompatible_ribbon_equilibrium(
            w_tilde=wt,
            kappa0=kappa0,
            nu=nu,
            theta_deg=45.0,
            branch=1,
            seed=seed + idx,
        )
        ana = analytical_pitchfork_curvatures(wt, kappa0=kappa0, nu=nu)
        H_sim.append(eq["H_over_kappa0"])
        H_obs.append(float(item["H_over_kappa0_obs"]))
        D_sim.append(eq["D_over_kappa0"])
        D_obs.append(float(item["D_over_kappa0_obs"]))
        R_sim.append(eq["R_kappa0"])
        R_obs.append(float(item["R_kappa0_obs"]))
        P_sim.append(eq["P_kappa0_over_2pi"])
        P_obs.append(float(item["P_kappa0_over_2pi_obs"]))
        rows.append(
            {
                "w_tilde": wt,
                "H_over_kappa0_sim": eq["H_over_kappa0"],
                "H_over_kappa0_ana": ana["H"] / kappa0,
                "H_over_kappa0_obs": float(item["H_over_kappa0_obs"]),
                "D_over_kappa0_sim": eq["D_over_kappa0"],
                "D_over_kappa0_obs": float(item["D_over_kappa0_obs"]),
                "R_kappa0_sim": eq["R_kappa0"],
                "R_kappa0_obs": float(item["R_kappa0_obs"]),
                "P_kappa0_over_2pi_sim": eq["P_kappa0_over_2pi"],
                "P_kappa0_over_2pi_obs": float(item["P_kappa0_over_2pi_obs"]),
                "K_over_kappa0_sq_sim": eq["K_over_kappa0_sq"],
                "regime": eq["regime"],
            }
        )

    H_sim_a, H_obs_a = np.asarray(H_sim), np.asarray(H_obs)
    R_sim_a, R_obs_a = np.asarray(R_sim), np.asarray(R_obs)
    P_sim_a, P_obs_a = np.asarray(P_sim), np.asarray(P_obs)

    r_H, p_H = stats.pearsonr(H_sim_a, H_obs_a)
    r_R, p_R = stats.pearsonr(R_sim_a, R_obs_a)
    r_P, p_P = stats.pearsonr(P_sim_a, P_obs_a)

    rmse_H = float(np.sqrt(np.mean((H_sim_a - H_obs_a) ** 2)))
    rmse_R = float(np.sqrt(np.mean((R_sim_a - R_obs_a) ** 2)))
    rmse_P = float(np.sqrt(np.mean((P_sim_a - P_obs_a) ** 2)))

    return {
        "w_tilde_c": critical_fvk_width(nu),
        "rows": rows,
        "pearson_r_H": float(r_H),
        "p_value_H": float(p_H),
        "rmse_H": rmse_H,
        "pearson_r_R": float(r_R),
        "p_value_R": float(p_R),
        "rmse_R": rmse_R,
        "pearson_r_P": float(r_P),
        "p_value_P": float(p_P),
        "rmse_P": rmse_P,
    }


def evaluate_cutting_angle_chirality_sweep(
    w_tilde: float = 4.0,
    kappa0: float = 0.42,
    nu: float = 0.30,
    seed: int = 2011,
) -> dict[str, Any]:
    """Sweep cutting angle theta in [0 deg, 90 deg] at w_tilde = 4.0 > w_tilde_c,
    verifying the theta = 0 deg and theta = 90 deg achiral ring controls (tau = 0,
    pitch P = 0) and maximal helical twist at the natural *Bauhinia* angle theta = 45 deg.
    Linked to Lean 4 `MathOfBiology.Loop_065.armon_narrow_helicoid_chirality`."""
    ref = load_reference_data()
    table = ref["cutting_angle_chirality_sweep_w_tilde_4"]

    rows: list[dict[str, Any]] = []
    tau_sim, tau_obs = [], []
    P_sim, P_obs = [], []

    for idx, item in enumerate(table):
        th = float(item["theta_deg"])
        eq = relax_incompatible_ribbon_equilibrium(
            w_tilde=w_tilde,
            kappa0=kappa0,
            nu=nu,
            theta_deg=th,
            branch=1,
            seed=seed + 100 + idx,
        )
        tau_norm = eq["tau"] / kappa0
        tau_sim.append(tau_norm)
        tau_obs.append(float(item["tau_over_kappa0_obs"]))
        P_sim.append(eq["P_cm"])
        P_obs.append(float(item["P_cm_obs"]))
        rows.append(
            {
                "theta_deg": th,
                "kappa_s_over_kappa0_sim": eq["kappa_s"] / kappa0,
                "kappa_s_over_kappa0_obs": float(item["kappa_s_over_kappa0_obs"]),
                "tau_over_kappa0_sim": tau_norm,
                "tau_over_kappa0_obs": float(item["tau_over_kappa0_obs"]),
                "R_cm_sim": eq["R_cm"],
                "R_cm_obs": float(item["R_cm_obs"]),
                "P_cm_sim": eq["P_cm"],
                "P_cm_obs": float(item["P_cm_obs"]),
                "morphology": item["morphology"],
            }
        )

    tau_sim_a, tau_obs_a = np.asarray(tau_sim), np.asarray(tau_obs)
    P_sim_a, P_obs_a = np.asarray(P_sim), np.asarray(P_obs)
    r_tau, p_tau = stats.pearsonr(tau_sim_a, tau_obs_a)
    r_P, p_P = stats.pearsonr(P_sim_a, P_obs_a)
    rmse_tau = float(np.sqrt(np.mean((tau_sim_a - tau_obs_a) ** 2)))
    rmse_P = float(np.sqrt(np.mean((P_sim_a - P_obs_a) ** 2)))

    return {
        "rows": rows,
        "pearson_r_tau": float(r_tau),
        "p_value_tau": float(p_tau),
        "rmse_tau": rmse_tau,
        "pearson_r_P": float(r_P),
        "p_value_P": float(p_P),
        "rmse_P_cm": rmse_P,
        "achiral_theta0_pitch_cm": rows[0]["P_cm_sim"],
        "achiral_theta90_pitch_cm": rows[-1]["P_cm_sim"],
        "max_twist_theta45_tau_over_kappa0": rows[3]["tau_over_kappa0_sim"],
    }


def evaluate_pod_and_mimetic_specimens(seed: int = 2011) -> dict[str, Any]:
    """Compare simulated vs empirical helix radius R (cm) and pitch P (cm) across
    the 4 benchmark systems in Armon et al. (2011) Science:
      1. *Bauhinia variegata* wide mature pod valve (w_tilde = 3.781 > w_tilde_c)
      2. *Bauhinia variegata* surgically narrowed strip (w_tilde = 1.365 < w_tilde_c, R = 0)
      3. *Erythrina corallodendron* coral tree pod valve (w_tilde = 3.687 > w_tilde_c)
      4. Biomimetic pre-stretched orthogonal latex bilayer (w_tilde = 5.933, nu = 0.48)"""
    ref = load_reference_data()
    table = ref["pod_and_mimetic_specimens_table"]

    rows: list[dict[str, Any]] = []
    R_sim, R_obs = [], []
    P_sim, P_obs = [], []

    for idx, item in enumerate(table):
        k0 = float(item["kappa0_cm_inv"])
        h_cm = float(item["h_cm"])
        W_cm = float(item["W_cm"])
        nu = float(item["nu"])
        wt = dimensionless_fvk_width(W_cm, k0, h_cm)
        eq = relax_incompatible_ribbon_equilibrium(
            w_tilde=wt,
            kappa0=k0,
            nu=nu,
            theta_deg=45.0,
            branch=1,
            seed=seed + 200 + idx,
        )
        R_sim.append(eq["R_cm"])
        R_obs.append(float(item["helix_radius_cm_obs"]))
        P_sim.append(eq["P_cm"])
        P_obs.append(float(item["helix_pitch_cm_obs"]))
        rows.append(
            {
                "specimen_id": item["specimen_id"],
                "species": item["species"],
                "w_tilde": wt,
                "w_tilde_c": eq["w_tilde_c"],
                "R_cm_sim": eq["R_cm"],
                "R_cm_obs": float(item["helix_radius_cm_obs"]),
                "P_cm_sim": eq["P_cm"],
                "P_cm_obs": float(item["helix_pitch_cm_obs"]),
                "regime": eq["regime"],
            }
        )

    R_sim_a, R_obs_a = np.asarray(R_sim), np.asarray(R_obs)
    P_sim_a, P_obs_a = np.asarray(P_sim), np.asarray(P_obs)
    r_R, p_R = stats.pearsonr(R_sim_a, R_obs_a)
    r_P, p_P = stats.pearsonr(P_sim_a, P_obs_a)
    rmse_R = float(np.sqrt(np.mean((R_sim_a - R_obs_a) ** 2)))
    rmse_P = float(np.sqrt(np.mean((P_sim_a - P_obs_a) ** 2)))

    return {
        "rows": rows,
        "pearson_r_radius": float(r_R),
        "p_value_radius": float(p_R),
        "rmse_radius_cm": rmse_R,
        "pearson_r_pitch": float(r_P),
        "p_value_pitch": float(p_P),
        "rmse_pitch_cm": rmse_P,
    }


# =============================================================================
# 4b. Pass 3: Quasi-Static Energy -> Suture Fracture -> Ballistic Seed Ejection
# =============================================================================
# Reduced transient coupling of the L-BFGS-B non-Euclidean minimizer (Section 3)
# to 1-D dynamic fracture along the pod suture (Freund 1990) and an elastic-
# catapult seed launch (Vogel 2005).  This is the honest reduced version of the
# Pass 2 remediation item "couple the quasi-static shell energy minimizer to
# transient elastodynamic fracture propagation along the suture line to model
# ballistic seed ejection velocities"; it is NOT a 3-D elastodynamic FEM.
# Constants and their provenance flags live in
# `datasets/chiral_seed_pod_opening/suture_fracture_ballistic_reference.json`.
FRACTURE_JSON = DATASET_DIR / "suture_fracture_ballistic_reference.json"


def load_fracture_reference() -> dict[str, Any]:
    """Load (SHA-256 verified) the Pass 3 suture-fracture / ballistic parameter table."""
    verify_dataset_integrity()
    return json.loads(FRACTURE_JSON.read_text(encoding="utf-8"))


def _fp(ref: dict[str, Any], block: str, key: str) -> float:
    return float(ref[block][key]["value"])


def bending_energy_prefactor(Y_Pa: float, h_m: float, nu: float) -> float:
    """B_eff = Y h^3 / (12 (1 + nu)) [J]; the reduced energy density in
    `relax_incompatible_ribbon_equilibrium` is E_bend [1/m^2] times B_eff.
    Derived from the Efrati et al. (2009) incompatible plate bending content
    (Y/(1-nu^2)) [nu Tr(M)^2 + (1-nu) Tr(M^2)] with M = b - b_0, Tr(M) = 2H and
    Tr(M^2) = 2 (H^2 + D^2 - 2 D kappa_0 cos 2phi + kappa_0^2), which collapses to
    (Y h^3 (1-nu) / (12 (1-nu^2))) * [((1+nu)/(1-nu)) H^2 + D^2 - 2 D kappa_0 cos 2phi + kappa_0^2]."""
    return float(Y_Pa * h_m**3 / (12.0 * (1.0 + nu)))


def released_valve_energy_density(
    kappa0_cm_inv: float,
    W_m: float,
    h_m: float,
    Y_Pa: float,
    nu: float,
    theta_deg: float = 45.0,
    seed: int = 2011,
) -> dict[str, float]:
    """Energy released per unit valve area when a valve held flat in the closed pod
    (H = D = 0, E_bend = kappa_0^2) relaxes to the quasi-static minimizer state
    (E = e_final):  Delta_e = B_eff (kappa_0^2 - e_final) [J/m^2].
    Suture energy release rate per valve: G = Delta_e * W / h_suture [J/m^2]."""
    if kappa0_cm_inv <= 0.0:
        return {"kappa0_cm_inv": 0.0, "w_tilde": 0.0, "delta_e_J_m2": 0.0, "G_J_m2": 0.0, "e_init": 0.0, "e_final": 0.0, "R_cm": 0.0, "P_cm": 0.0}
    wt = dimensionless_fvk_width(W_m * 100.0, kappa0_cm_inv, h_m * 100.0)
    eq = relax_incompatible_ribbon_equilibrium(w_tilde=wt, kappa0=kappa0_cm_inv, nu=nu, theta_deg=theta_deg, seed=seed)
    b_eff = bending_energy_prefactor(Y_Pa, h_m, nu)
    # energies in the minimizer are in cm^-2 -> convert to m^-2
    e_init_m2 = eq["energy_initial"] * 1.0e4
    e_final_m2 = eq["energy_final"] * 1.0e4
    delta_e = b_eff * max(e_init_m2 - e_final_m2, 0.0)
    return {
        "kappa0_cm_inv": float(kappa0_cm_inv),
        "w_tilde": float(wt),
        "e_init": float(eq["energy_initial"]),
        "e_final": float(eq["energy_final"]),
        "delta_e_J_m2": float(delta_e),
        "G_J_m2": float(delta_e * W_m / h_m),
        "R_cm": float(eq["R_cm"]),
        "P_cm": float(eq["P_cm"]),
        "regime": eq["regime"],
    }


def rayleigh_wave_speed(Y_Pa: float, rho: float, nu: float, factor: float = 0.93) -> float:
    """c_R ~= factor * c_s,  c_s = sqrt(Y / (2 (1 + nu) rho))."""
    return float(factor * math.sqrt(Y_Pa / (2.0 * (1.0 + nu) * rho)))


def freund_crack_speed(G_J_m2: float, Gamma_c_J_m2: float, c_R: float) -> float:
    """Freund (1990) mode-I equation of motion G_dyn(v) ~= (1 - v/c_R) G = Gamma_c
    =>  v = c_R max(0, 1 - Gamma_c / G).  v = 0 (arrest) when Gamma_c >= G.
    Linked to Lean 4 `MathOfBiology.Loop_065.freund_crack_speed_arrest_and_monotonicity`."""
    if G_J_m2 <= 0.0:
        return 0.0
    return float(c_R * max(0.0, 1.0 - Gamma_c_J_m2 / G_J_m2))


def heterogeneous_suture_toughness(
    n: int,
    Gamma_mean: float,
    cv: float,
    seed: int = 2011,
    smooth_sigma: float = 6.0,
) -> np.ndarray:
    """Gamma_c(a) = Gamma_mean * max(0.05, 1 + cv * xi(a)) with xi a unit-variance
    Gaussian-smoothed noise field (seeded, deterministic)."""
    rng = np.random.default_rng(seed)
    xi = ndi.gaussian_filter1d(rng.standard_normal(n), sigma=smooth_sigma, mode="reflect")
    xi = xi / max(float(np.std(xi)), 1e-12) if cv > 0.0 else np.zeros(n)
    return Gamma_mean * np.maximum(0.05, 1.0 + cv * xi)


def integrate_suture_crack(
    G_J_m2: float,
    Gamma_field: np.ndarray,
    L_m: float,
    c_R: float,
    a0_m: float,
    dt_s: float | None = None,
    max_steps: int = 200_000,
    v_arrest_frac: float = 0.01,
) -> dict[str, Any]:
    """Explicit integration of da/dt = c_R max(0, 1 - Gamma_c(a)/G) from the stylar
    flaw a_0 along the suture of length L. Returns the trajectory, arrest position a*
    (where Gamma_c(a*) >= G, operationally v < v_arrest_frac * c_R so that the
    asymptotic creep toward the arrest point is not counted as crack run time),
    opening fraction a*/L and mean crack speed."""
    n = len(Gamma_field)
    xs = np.linspace(0.0, L_m, n)
    if dt_s is None:
        dt_s = (L_m / n) / max(c_R, 1e-9) * 0.5
    a = float(a0_m)
    t = 0.0
    traj_t = [0.0]
    traj_a = [a]
    arrested = False
    v_min = v_arrest_frac * c_R
    for _ in range(max_steps):
        gam = float(np.interp(a, xs, Gamma_field))
        v = freund_crack_speed(G_J_m2, gam, c_R)
        if v < v_min:
            arrested = True
            break
        a = min(a + v * dt_s, L_m)
        t += dt_s
        traj_t.append(t)
        traj_a.append(a)
        if a >= L_m:
            break
    traj_t_a = np.asarray(traj_t)
    traj_a_a = np.asarray(traj_a)
    run_len = traj_a_a[-1] - a0_m
    v_mean = float(run_len / traj_t_a[-1]) if traj_t_a[-1] > 0 else 0.0
    return {
        "t_s": traj_t_a,
        "a_m": traj_a_a,
        "a_final_m": float(traj_a_a[-1]),
        "opening_fraction": float(traj_a_a[-1] / L_m),
        "arrested": bool(arrested),
        "run_time_s": float(traj_t_a[-1]),
        "v_mean_m_s": v_mean,
        "v_over_cR_mean": float(v_mean / c_R) if c_R > 0 else 0.0,
        "v_homogeneous_m_s": freund_crack_speed(G_J_m2, float(np.mean(Gamma_field)), c_R),
    }


def elastic_catapult_seed_speed(
    delta_e_J_m2: float,
    W_m: float,
    l_seg_m: float,
    h_m: float,
    rho: float,
    m_seed_kg: float,
    eta: float,
) -> dict[str, float]:
    """v_seed = sqrt(2 eta Delta_U_seg / m_eff), Delta_U_seg = Delta_e W l_seg,
    m_eff = m_seg/3 + m_seed (cantilever-tip effective mass), m_seg = rho h W l_seg.
    Linked to Lean 4 `MathOfBiology.Loop_065.elastic_catapult_energy_partition`."""
    dU = delta_e_J_m2 * W_m * l_seg_m
    m_seg = rho * h_m * W_m * l_seg_m
    m_eff = m_seg / 3.0 + m_seed_kg
    ke = eta * dU
    v = math.sqrt(2.0 * ke / m_eff) if m_eff > 0 else 0.0
    return {"delta_U_seg_J": float(dU), "m_seg_kg": float(m_seg), "m_eff_kg": float(m_eff), "kinetic_J": float(ke), "v_seed_m_s": float(v)}


def valve_segment_coil_time(Y_Pa: float, h_m: float, nu: float, rho: float, l_seg_m: float) -> float:
    """Quarter period of the first cantilever flexural mode of a valve segment:
    omega_1 = 3.516 sqrt(B/(rho h)) / l^2 with B = Y h^3 / (12 (1 - nu^2))."""
    B = Y_Pa * h_m**3 / (12.0 * (1.0 - nu * nu))
    omega1 = 3.516 * math.sqrt(B / (rho * h_m)) / (l_seg_m**2)
    return float(0.5 * math.pi / omega1)


def ballistic_range_with_drag(
    v0: float,
    angle_deg: float,
    z0: float,
    m_kg: float,
    r_m: float,
    C_d: float,
    rho_air: float,
    g: float,
    dt: float = 5e-4,
) -> dict[str, float]:
    """RK4 integration of m dv/dt = m g - (1/2) rho_air C_d A |v| v until the seed
    falls back to the ground (z = 0). Returns the horizontal range and flight time,
    plus the drag-free analytical bound for launch height z0:
      x_free = v0 cos(a) * (v0 sin(a) + sqrt(v0^2 sin^2(a) + 2 g z0)) / g."""
    A = math.pi * r_m * r_m
    k = 0.5 * rho_air * C_d * A / m_kg
    th = math.radians(angle_deg)
    vz0 = v0 * math.sin(th)
    x_free = v0 * math.cos(th) * (vz0 + math.sqrt(vz0 * vz0 + 2.0 * g * z0)) / g
    state = np.array([0.0, z0, v0 * math.cos(th), vz0], dtype=float)

    def f(s: np.ndarray) -> np.ndarray:
        vx, vz = s[2], s[3]
        sp = math.hypot(vx, vz)
        return np.array([vx, vz, -k * sp * vx, -g - k * sp * vz])

    t = 0.0
    for _ in range(200_000):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        new = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if new[1] <= 0.0 and t > dt:
            frac = state[1] / max(state[1] - new[1], 1e-12)
            x_land = state[0] + frac * (new[0] - state[0])
            return {"range_m": float(x_land), "flight_time_s": float(t - dt + frac * dt), "drag_free_bound_m": float(x_free)}
        state = new
    return {"range_m": float(state[0]), "flight_time_s": float(t), "drag_free_bound_m": float(x_free)}


def simulate_suture_fracture_ballistic_ejection(
    overrides: dict[str, float] | None = None,
    seed: int = 2011,
    n_drying: int = 96,
) -> dict[str, Any]:
    """Transient coupling of the quasi-static shell minimizer to suture fracture and
    seed ejection for one Bauhinia variegata pod:
      1. Desiccation kappa_0(t) = kappa_0,f (1 - exp(-t/tau_dry)); for each t the
         Section 3 minimizer gives the relaxed valve energy and G(t) = Delta_e(t) W / h.
      2. Griffith initiation: t_c solves G(t_c) = Gamma_c(a_0) (brentq on the drying curve).
      3. Freund crack run da/dt = c_R max(0, 1 - Gamma_c(a)/G) over a heterogeneous
         toughness field -> arrest position a*, opening fraction, mean crack speed.
      4. Each seed i at suture position a_i (<= a*) is launched at
         t_eject,i = t_c + (a_i - a_0)/v + t_coil with elastic-catapult speed v_seed.
      5. Ballistic range with quadratic drag (RK4) from launch height z_0."""
    ref = load_fracture_reference()
    ov = overrides or {}
    Y = _fp(ref, "material_and_geometry", "Y_modulus_Pa")
    nu = float(ov.get("nu", _fp(ref, "material_and_geometry", "nu_poisson")))
    rho = _fp(ref, "material_and_geometry", "rho_valve_kg_m3")
    h_m = float(ov.get("h", _fp(ref, "material_and_geometry", "h_valve_m") * 100.0)) / 100.0
    W_m = float(ov.get("W", _fp(ref, "material_and_geometry", "W_valve_m") * 100.0)) / 100.0
    L_m = float(ov.get("L", _fp(ref, "material_and_geometry", "L_valve_m") * 100.0)) / 100.0
    k0_final = float(ov.get("kappa0", _fp(ref, "material_and_geometry", "kappa0_final_cm_inv")))
    h_sut = _fp(ref, "material_and_geometry", "h_suture_m")
    Gamma_mean = float(ov.get("gamma_c", _fp(ref, "fracture_parameters", "Gamma_c_mean_J_m2")))
    cv_gamma = _fp(ref, "fracture_parameters", "cv_Gamma")
    a0 = _fp(ref, "fracture_parameters", "a0_flaw_m")
    tau_dry_s = _fp(ref, "fracture_parameters", "tau_dry_h") * 3600.0
    c_R = rayleigh_wave_speed(Y, rho, nu, _fp(ref, "fracture_parameters", "rayleigh_factor"))
    n_seeds = int(_fp(ref, "seed_parameters", "n_seeds"))
    m_seed = float(ov.get("m_seed", _fp(ref, "seed_parameters", "m_seed_kg")))
    r_seed = _fp(ref, "seed_parameters", "r_seed_m")
    C_d = _fp(ref, "seed_parameters", "C_d")
    eta = float(ov.get("eta", _fp(ref, "seed_parameters", "eta_catapult")))
    ang = _fp(ref, "seed_parameters", "launch_angle_deg")
    z0 = _fp(ref, "seed_parameters", "launch_height_m")
    rho_air = _fp(ref, "seed_parameters", "rho_air_kg_m3")
    g = _fp(ref, "seed_parameters", "g_m_s2")

    # 1. Drying trajectory -> quasi-static valve energy release rate G(t)
    t_grid = np.linspace(0.0, 6.0 * tau_dry_s, n_drying)
    k0_grid = k0_final * (1.0 - np.exp(-t_grid / tau_dry_s))
    G_grid = np.zeros_like(t_grid)
    dE_grid = np.zeros_like(t_grid)
    wt_grid = np.zeros_like(t_grid)
    for i, k0 in enumerate(k0_grid):
        rel = released_valve_energy_density(float(k0), W_m, h_m, Y, nu, seed=seed)
        dE_grid[i] = rel["delta_e_J_m2"]
        G_grid[i] = rel["delta_e_J_m2"] * W_m / h_sut
        wt_grid[i] = rel["w_tilde"]
    G_final = float(G_grid[-1])
    dE_final = float(dE_grid[-1])

    # 3-prep. Heterogeneous suture toughness along the pod
    n_field = 361
    Gamma_field = heterogeneous_suture_toughness(n_field, Gamma_mean, cv_gamma, seed=seed + 300)
    xs_field = np.linspace(0.0, L_m, n_field)
    Gamma_a0 = float(np.interp(a0, xs_field, Gamma_field))

    # 2.+3. Two-timescale stick-slip dehiscence. Drying (hours) raises G(t); whenever
    # G(t) exceeds the local toughness Gamma_c(a) at the crack tip the crack runs
    # dynamically (ms) with the Freund law until it arrests where Gamma_c(a) >= G(t),
    # or reaches the pod base a = L. Griffith initiation time t_c = time of the first jump.
    l_seg = L_m / n_seeds
    t_coil = valve_segment_coil_time(Y, h_m, nu, rho, l_seg)
    seed_pos = (np.arange(n_seeds) + 0.5) * l_seg
    a = float(a0)
    jumps: list[dict[str, float]] = []
    seeds: list[dict[str, Any]] = [
        {"index": i, "a_m": float(a_i), "released": False, "t_eject_s": float("nan"), "v_seed_m_s": 0.0, "kappa0_at_eject_cm_inv": float("nan")}
        for i, a_i in enumerate(seed_pos)
    ]
    t_c = float("inf")
    k0_c = float("nan")
    G_c = float("nan")
    for i_t in range(len(t_grid)):
        G_now = float(G_grid[i_t])
        if a >= L_m:
            break
        if G_now <= 0.0:
            continue
        gam_tip = float(np.interp(a, xs_field, Gamma_field))
        if gam_tip >= G_now:
            continue
        crack = integrate_suture_crack(G_now, Gamma_field, L_m, c_R, a)
        if crack["a_final_m"] <= a + 1e-12:
            continue
        if not jumps:
            t_c = float(t_grid[i_t])
            k0_c = float(k0_grid[i_t])
            G_c = G_now
        cat = elastic_catapult_seed_speed(float(dE_grid[i_t]), W_m, l_seg, h_m, rho, m_seed, eta)
        for s in seeds:
            if not s["released"] and a < s["a_m"] <= crack["a_final_m"]:
                t_arr = float(np.interp(s["a_m"], crack["a_m"], crack["t_s"]))
                s["released"] = True
                s["t_eject_s"] = float(t_grid[i_t]) + t_arr + t_coil
                s["v_seed_m_s"] = cat["v_seed_m_s"]
                s["kappa0_at_eject_cm_inv"] = float(k0_grid[i_t])
        jumps.append(
            {
                "t_start_s": float(t_grid[i_t]),
                "a_from_m": a,
                "a_to_m": crack["a_final_m"],
                "run_time_s": crack["run_time_s"],
                "v_mean_m_s": crack["v_mean_m_s"],
                "v_over_cR_mean": crack["v_over_cR_mean"],
                "G_J_m2": G_now,
                "kappa0_cm_inv": float(k0_grid[i_t]),
            }
        )
        a = crack["a_final_m"]
    initiated = bool(jumps)
    a_final = a
    opening_fraction = float(a_final / L_m)
    n_ejected = int(sum(1 for s in seeds if s["released"]))
    released_v = [s["v_seed_m_s"] for s in seeds if s["released"]]
    v_seed_mean = float(np.mean(released_v)) if released_v else 0.0
    v_seed_max = float(np.max(released_v)) if released_v else 0.0
    total_run_time = float(sum(j["run_time_s"] for j in jumps))
    v_crack_mean = float((a_final - a0) / total_run_time) if total_run_time > 0 else 0.0

    # 4. Reference catapult speed for the fully dried pod (upper bound for this pod)
    cat_final = elastic_catapult_seed_speed(dE_final, W_m, l_seg, h_m, rho, m_seed, eta)

    # 5. Ballistics of the mean-speed ejected seed
    ball = (
        ballistic_range_with_drag(v_seed_mean, ang, z0, m_seed, r_seed, C_d, rho_air, g)
        if v_seed_mean > 0
        else {"range_m": 0.0, "flight_time_s": 0.0, "drag_free_bound_m": 0.0}
    )

    return {
        "params": {"Y_Pa": Y, "nu": nu, "rho": rho, "h_m": h_m, "W_m": W_m, "L_m": L_m, "kappa0_final_cm_inv": k0_final, "Gamma_c_mean_J_m2": Gamma_mean, "cv_Gamma": cv_gamma, "a0_m": a0, "tau_dry_s": tau_dry_s, "c_R_m_s": c_R, "n_seeds": n_seeds, "m_seed_kg": m_seed, "eta": eta, "l_seg_m": l_seg},
        "B_eff_J": bending_energy_prefactor(Y, h_m, nu),
        "drying": {"t_s": t_grid, "kappa0_cm_inv": k0_grid, "w_tilde": wt_grid, "delta_e_J_m2": dE_grid, "G_J_m2": G_grid},
        "G_final_J_m2": G_final,
        "delta_e_final_J_m2": dE_final,
        "stored_energy_per_valve_J": dE_final * W_m * L_m,
        "Gamma_at_flaw_J_m2": Gamma_a0,
        "Gamma_field": Gamma_field,
        "xs_field_m": xs_field,
        "initiated": initiated,
        "t_c_s": t_c,
        "t_c_h": t_c / 3600.0 if initiated else float("inf"),
        "kappa0_at_initiation_cm_inv": k0_c,
        "G_at_initiation_J_m2": G_c,
        "jumps": jumps,
        "n_jumps": len(jumps),
        "a_final_m": a_final,
        "opening_fraction": opening_fraction,
        "total_crack_run_time_s": total_run_time,
        "v_crack_mean_m_s": v_crack_mean,
        "v_crack_over_cR": float(v_crack_mean / c_R),
        "catapult_final": cat_final,
        "t_coil_s": t_coil,
        "seeds": seeds,
        "n_ejected": n_ejected,
        "v_seed_m_s": v_seed_mean,
        "v_seed_max_m_s": v_seed_max,
        "v_seed_fully_dried_m_s": cat_final["v_seed_m_s"],
        "ballistics": ball,
    }


def evaluate_suture_toughness_sweep(seed: int = 2011) -> dict[str, Any]:
    """Ablation / parameter sweep over the suture toughness ratio Gamma_c/G_final:
    crack speed v/c_R (= 1 - Gamma/G for a homogeneous suture), opening fraction,
    number of ejected seeds. Includes the Gamma_c >= G no-fracture negative control."""
    ref = load_fracture_reference()
    base = simulate_suture_fracture_ballistic_ejection(seed=seed)
    G_final = base["G_final_J_m2"]
    ratios = [float(r) for r in ref["toughness_sweep_design"]["Gamma_over_G_grid"]]
    rows: list[dict[str, float]] = []
    for r in ratios:
        res = simulate_suture_fracture_ballistic_ejection(overrides={"gamma_c": r * G_final}, seed=seed)
        rows.append(
            {
                "Gamma_over_G": r,
                "initiated": float(res["initiated"]),
                "t_c_h": res["t_c_h"],
                "n_jumps": float(res["n_jumps"]),
                "v_over_cR_mean": res["v_crack_over_cR"],
                "opening_fraction": res["opening_fraction"],
                "n_ejected": float(res["n_ejected"]),
                "v_seed_m_s": res["v_seed_m_s"],
            }
        )
    # In the drying-coupled run the crack moves in stick-slip jumps each starting with
    # G(t) barely above Gamma_c (so v/c_R is small).  To test the Freund law itself we
    # also run the fully dried pod (G = G_final frozen) for each toughness ratio:
    # (homogeneous suture cv = 0 -> exact Freund law v/c_R = 1 - Gamma/G; heterogeneous
    # suture cv = cv_Gamma -> arrest / partial opening as Gamma/G -> 1).
    dried_rows: list[dict[str, float]] = []
    c_R = base["params"]["c_R_m_s"]
    L_m = base["params"]["L_m"]
    a0 = base["params"]["a0_m"]
    for r in ratios:
        gam_h = heterogeneous_suture_toughness(361, r * G_final, 0.0, seed=seed + 300)
        gam_x = heterogeneous_suture_toughness(361, r * G_final, base["params"]["cv_Gamma"], seed=seed + 300)
        crack_h = integrate_suture_crack(G_final, gam_h, L_m, c_R, a0)
        crack_x = integrate_suture_crack(G_final, gam_x, L_m, c_R, a0)
        dried_rows.append(
            {
                "Gamma_over_G": r,
                "v_over_cR_homogeneous": crack_h["v_over_cR_mean"],
                "v_over_cR_theory": max(0.0, 1.0 - r),
                "opening_fraction_homogeneous": crack_h["opening_fraction"],
                "v_over_cR_heterogeneous": crack_x["v_over_cR_mean"],
                "opening_fraction_heterogeneous": crack_x["opening_fraction"],
            }
        )
    v_sim = np.array([d["v_over_cR_homogeneous"] for d in dried_rows])
    v_th = np.array([d["v_over_cR_theory"] for d in dried_rows])
    r_v, p_v = stats.pearsonr(v_sim, v_th)
    return {
        "rows": rows,
        "dried_rows": dried_rows,
        "pearson_r_v_vs_freund": float(r_v),
        "p_value_v": float(p_v),
        "rmse_v_over_cR": float(np.sqrt(np.mean((v_sim - v_th) ** 2))),
        "no_fracture_control_opening_fraction": dried_rows[-1]["opening_fraction_homogeneous"],
        "heterogeneous_opening_fraction_at_Gamma_eq_G": dried_rows[ratios.index(1.0)]["opening_fraction_heterogeneous"] if 1.0 in ratios else float("nan"),
        "G_final_J_m2": G_final,
    }


# =============================================================================
# 5. Computer-Vision Segmentation of Bauhinia variegata Specimen Image
# =============================================================================
def segment_bauhinia_pod_specimen(specimen_path: Path = SPECIMEN_JPG) -> dict[str, Any]:
    """Segment the SHA-256-locked biological specimen photograph
    `datasets/chiral_seed_pod_opening/bauhinia_chiral_seed_pod_specimen.jpg`
    (*Bauhinia variegata* Muséum de Toulouse MHNT.BOT.2011.3.22):
      1. Separates the top dehisced chiral helical valves from the bottom intact
         flat pre-dehiscence pod and released seeds against the white background.
      2. Extracts the upper and lower dehisced valve ribbon contours and centerlines.
      3. Tracks the periodic inner-tan vs outer-dark-brown sclerenchyma bilayer
         reflectance oscillation along both twisted valves to localize the helical
         half-pitch lobe apices and compute inter-lobe spacing CV."""
    verify_dataset_integrity()
    rgb = np.asarray(Image.open(specimen_path).convert("RGB"), dtype=float) / 255.0
    x0, y0, x1, y1 = 150, 60, 928, 452
    roi = rgb[y0:y1, x0:x1]

    dark = 1.0 - roi.mean(axis=2)
    mask = ndi.binary_closing(dark > 0.14, structure=np.ones((5, 5)))
    lbl, n_comp = ndi.label(mask)
    sizes = ndi.sum(mask, lbl, range(1, n_comp + 1))
    valve_mask = lbl == int(np.argmax(sizes) + 1)

    cols = np.where(valve_mask.sum(axis=0) > 6)[0]
    upper_cy = np.zeros(len(cols))
    lower_cy = np.zeros(len(cols))
    upper_tan = np.zeros(len(cols))
    lower_tan = np.zeros(len(cols))
    upper_width = np.zeros(len(cols))

    for idx, c in enumerate(cols):
        col_m = valve_mask[:, c]
        runs = np.where(np.diff(np.concatenate(([0], col_m.view(np.int8), [0]))))[0].reshape(-1, 2)
        r0_u, r1_u = runs[0]
        ys_u = np.arange(r0_u, min(r1_u, r0_u + 55))
        upper_cy[idx] = float(np.mean(ys_u))
        upper_width[idx] = float(len(ys_u))
        upper_tan[idx] = float(np.mean(roi[ys_u, c, 0]))

        r0_l, r1_l = runs[-1]
        ys_l = np.arange(max(r0_l, r1_l - 55), r1_l)
        lower_cy[idx] = float(np.mean(ys_l))
        lower_tan[idx] = float(np.mean(roi[ys_l, c, 0]))

    u_sig = ndi.gaussian_filter1d(upper_tan, sigma=9.0)
    l_sig = ndi.gaussian_filter1d(lower_tan, sigma=9.0)
    u_cy_smooth = ndi.gaussian_filter1d(upper_cy, sigma=6.0)
    l_cy_smooth = ndi.gaussian_filter1d(lower_cy, sigma=6.0)

    peaks_u, _ = signal.find_peaks(u_sig, distance=85, prominence=0.015)
    peaks_l, _ = signal.find_peaks(l_sig, distance=85, prominence=0.015)

    sp_u = np.diff(cols[peaks_u]).astype(float)
    sp_l = np.diff(cols[peaks_l]).astype(float)
    all_sp = np.concatenate([sp_u, sp_l])

    mean_half_pitch_px = float(np.mean(all_sp))
    std_half_pitch_px = float(np.std(all_sp))
    cv_half_pitch = float(std_half_pitch_px / max(mean_half_pitch_px, 1e-9))

    apices_xy: list[tuple[float, float]] = []
    for p in peaks_u:
        apices_xy.append((float(cols[p]), float(u_cy_smooth[p])))
    for p in peaks_l:
        apices_xy.append((float(cols[p]), float(l_cy_smooth[p])))

    return {
        "roi_rgb": roi,
        "valve_mask": valve_mask,
        "valve_area_fraction": float(np.mean(valve_mask)),
        "cols_x": cols,
        "upper_cy": u_cy_smooth,
        "lower_cy": l_cy_smooth,
        "upper_tan_signal": u_sig,
        "lower_tan_signal": l_sig,
        "peaks_u_idx": peaks_u,
        "peaks_l_idx": peaks_l,
        "apices_xy": np.asarray(apices_xy, dtype=float),
        "n_helical_lobes": int(len(apices_xy)),
        "mean_half_pitch_px": mean_half_pitch_px,
        "std_half_pitch_px": std_half_pitch_px,
        "cv_half_pitch": cv_half_pitch,
    }


# =============================================================================
# 6. 9-Panel Publication Synthesis Figure & Animated Desiccation GIF
# =============================================================================
def _draw_fracture_panels(ax_g, ax_h, ax_i, frac: dict[str, Any], sweep: dict[str, Any], ref: dict[str, Any]) -> None:
    """Panels G-I (Pass 3): desiccation-driven suture energy release & stick-slip
    crack position; Freund crack-speed law vs toughness sweep; seed ejection speeds
    & ballistic range against published explosive-dispersal benchmarks."""
    dry = frac["drying"]
    t_h = dry["t_s"] / 3600.0
    gam_mean = frac["params"]["Gamma_c_mean_J_m2"]
    cv = frac["params"]["cv_Gamma"]

    # Panel G
    ax_g.plot(t_h, dry["G_J_m2"], color="#38bdf8", lw=2.3, label="Suture energy release rate G(t) = Δe(κ₀(t))·W/h")
    ax_g.axhspan(gam_mean * (1 - cv), gam_mean * (1 + cv), color="#f43f5e", alpha=0.18, label=f"Suture toughness Γ_c = {gam_mean:.0f}±{cv*gam_mean:.0f} J/m² (model)")
    ax_g.axhline(gam_mean, color="#f43f5e", ls="--", lw=1.3)
    if frac["initiated"]:
        ax_g.axvline(frac["t_c_h"], color="#facc15", ls=":", lw=1.8, label=f"Griffith initiation t_c = {frac['t_c_h']:.1f} h (κ₀ = {frac['kappa0_at_initiation_cm_inv']:.3f} cm⁻¹)")
    ax_g2 = ax_g.twinx()
    a_steps_t = [0.0]
    a_steps_a = [frac["params"]["a0_m"] / frac["params"]["L_m"]]
    for j in frac["jumps"]:
        a_steps_t += [j["t_start_s"] / 3600.0, j["t_start_s"] / 3600.0]
        a_steps_a += [j["a_from_m"] / frac["params"]["L_m"], j["a_to_m"] / frac["params"]["L_m"]]
    a_steps_t.append(float(t_h[-1]))
    a_steps_a.append(a_steps_a[-1])
    ax_g2.plot(a_steps_t, a_steps_a, color="#34d399", lw=2.0, label=f"Stick-slip crack position a(t)/L ({frac['n_jumps']} jumps)")
    for s in frac["seeds"]:
        if s["released"]:
            ax_g2.scatter([s["t_eject_s"] / 3600.0], [s["a_m"] / frac["params"]["L_m"]], marker="*", s=110, color="#fb923c", edgecolors="#0b0f17", zorder=6)
    ax_g2.scatter([], [], marker="*", s=110, color="#fb923c", label=f"Seed ejection events (n = {frac['n_ejected']}/{frac['params']['n_seeds']})")
    ax_g2.set_ylim(-0.02, 1.08)
    ax_g2.set_ylabel("Crack position a/L", color="#34d399", fontsize=9)
    ax_g2.tick_params(colors="#8b949e", labelsize=8)
    ax_g.set_title(
        f"(G) Quasi-Static Energy → Suture Fracture Coupling\nG_final = {frac['G_final_J_m2']:.0f} J/m², stored {1e3*frac['stored_energy_per_valve_J']:.1f} mJ/valve, κ₀(t)=κ₀,f(1-e^(-t/τ)), τ = {frac['params']['tau_dry_s']/3600:.0f} h",
        color="#e6edf3",
        fontsize=10,
    )
    ax_g.set_xlabel("Desiccation time t (h)", color="#c9d1d9", fontsize=9)
    ax_g.set_ylabel("G, Γ_c (J/m²)", color="#c9d1d9", fontsize=9)
    ax_g.tick_params(colors="#8b949e", labelsize=8)
    ax_g.grid(color="#1f2937", alpha=0.6, lw=0.7)
    h1, l1 = ax_g.get_legend_handles_labels()
    h2, l2 = ax_g2.get_legend_handles_labels()
    ax_g.legend(h1 + h2, l1 + l2, facecolor="#0b0f17", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.2, loc="lower right")

    # Panel H
    ratios = np.array([d["Gamma_over_G"] for d in sweep["dried_rows"]])
    v_hom = np.array([d["v_over_cR_homogeneous"] for d in sweep["dried_rows"]])
    v_het = np.array([d["v_over_cR_heterogeneous"] for d in sweep["dried_rows"]])
    of_het = np.array([d["opening_fraction_heterogeneous"] for d in sweep["dried_rows"]])
    of_dry = np.array([d["opening_fraction"] for d in sweep["rows"]])
    n_ej = np.array([d["n_ejected"] for d in sweep["rows"]])
    rr = np.linspace(0.0, 1.3, 200)
    ax_h.plot(rr, np.maximum(0.0, 1.0 - rr), color="#38bdf8", lw=2.2, label="Freund (1990): v/c_R = 1 − Γ_c/G")
    ax_h.scatter(ratios, v_hom, color="#22d3ee", s=55, zorder=5, label=f"Sim homogeneous suture (r = {sweep['pearson_r_v_vs_freund']:.4f})")
    ax_h.scatter(ratios, v_het, marker="s", color="#a78bfa", s=45, zorder=5, label=f"Sim heterogeneous suture (CV_Γ = {cv:.2f})")
    ax_h.plot(ratios, of_het, "o--", color="#34d399", lw=1.8, label="Opening fraction a*/L (fully dried, heterogeneous)")
    ax_h.plot(ratios, of_dry, "^-", color="#fb923c", lw=1.8, label="Opening fraction a*/L (drying-coupled stick-slip)")
    ax_h.plot(ratios, n_ej / frac["params"]["n_seeds"], "*:", color="#facc15", lw=1.4, ms=10, label="Fraction of seeds ejected (drying-coupled)")
    ax_h.axvline(1.0, color="#f43f5e", ls=":", lw=1.6, label="Γ_c = G: no-fracture negative control")
    ax_h.set_title(
        f"(H) Dynamic Crack Speed & Arrest vs Suture Toughness\nc_R = {frac['params']['c_R_m_s']:.0f} m/s, pod opens in {1e3*frac['total_crack_run_time_s']:.2f} ms of cumulative crack run (⟨v⟩ = {frac['v_crack_mean_m_s']:.0f} m/s)",
        color="#e6edf3",
        fontsize=10,
    )
    ax_h.set_xlabel("Suture toughness ratio Γ_c / G_final", color="#c9d1d9", fontsize=9)
    ax_h.set_ylabel("v/c_R, opening fraction, ejected fraction", color="#c9d1d9", fontsize=9)
    ax_h.set_ylim(-0.05, 1.1)
    ax_h.tick_params(colors="#8b949e", labelsize=8)
    ax_h.grid(color="#1f2937", alpha=0.6, lw=0.7)
    ax_h.legend(facecolor="#0b0f17", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.2, loc="upper right")

    # Panel I
    bench = ref["benchmarks"]
    v_cardamine = float(next(b["value_m_s"] for b in bench if b["id"] == "cardamine_hofhuis2016_launch_speed"))
    v_hura = float(next(b["value_m_s"] for b in bench if b["id"] == "hura_swaine1977_launch_speed"))
    seeds = frac["seeds"]
    pos_cm = [100.0 * s["a_m"] for s in seeds]
    v_s = [s["v_seed_m_s"] for s in seeds]
    ax_i.bar(pos_cm, v_s, width=2.2, color="#fb923c", edgecolor="#fed7aa", label=f"Bauhinia seed launch speed (drying-coupled, mean {frac['v_seed_m_s']:.2f} m/s)")
    ax_i.axhline(frac["v_seed_fully_dried_m_s"], color="#fde047", ls="--", lw=1.6, label=f"Fully dried catapult bound {frac['v_seed_fully_dried_m_s']:.2f} m/s (η = {frac['params']['eta']:.1f})")
    ax_i.axhline(v_cardamine, color="#38bdf8", ls=":", lw=1.8, label=f"Cardamine hirsuta ≈ {v_cardamine:.0f} m/s (Hofhuis et al. 2016, transfer)")
    ax_i.set_yscale("log")
    ax_i.set_ylim(0.3, 150.0)
    ax_i.axhline(v_hura, color="#f43f5e", ls="-.", lw=1.4, label=f"Hura crepitans ≈ {v_hura:.0f} m/s (Swaine & Beer 1977, upper bound)")
    ball = frac["ballistics"]
    ax_i.text(
        0.02,
        0.50,
        f"Ballistics (45°, z₀ = 3 m, m_seed = {1e3*frac['params']['m_seed_kg']:.2f} g, C_d = 1):\n"
        f"range with drag = {ball['range_m']:.2f} m  (drag-free bound {ball['drag_free_bound_m']:.2f} m)\n"
        f"flight time = {ball['flight_time_s']:.2f} s; valve coil time t_coil = {1e3*frac['t_coil_s']:.2f} ms",
        transform=ax_i.transAxes,
        color="#e6edf3",
        fontsize=7.8,
        va="bottom",
        zorder=7,
        bbox=dict(facecolor="#0b0f17", edgecolor="#30363d", alpha=0.9),
    )
    ax_i.set_title(
        "(I) Elastic-Catapult Seed Ejection vs Explosive-Dispersal Benchmarks\nv_seed = √(2η ΔU_seg / (m_seg/3 + m_seed)); Bauhinia predicted far below Cardamine / Hura",
        color="#e6edf3",
        fontsize=10,
    )
    ax_i.set_xlabel("Seed position along suture from stylar tip (cm)", color="#c9d1d9", fontsize=9)
    ax_i.set_ylabel("Launch speed (m/s, log)", color="#c9d1d9", fontsize=9)
    ax_i.tick_params(colors="#8b949e", labelsize=8)
    ax_i.grid(color="#1f2937", alpha=0.6, lw=0.7, which="both")
    ax_i.legend(facecolor="#0b0f17", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.2, loc="upper right")


def render_fracture_ballistic_figure(out_png: Path | None = None, overrides: dict[str, float] | None = None, seed: int = 2011) -> Path:
    """Standalone 3-panel Pass 3 figure (G-I) -> docs/media/chiral_seed_pod_opening_suture_fracture.png."""
    out_png = out_png or OUT_FRACTURE_PNG
    out_png.parent.mkdir(parents=True, exist_ok=True)
    frac = simulate_suture_fracture_ballistic_ejection(overrides=overrides, seed=seed)
    sweep = evaluate_suture_toughness_sweep(seed=seed)
    ref = load_fracture_reference()
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8), dpi=140, facecolor="#0b0f17")
    for ax in axes:
        ax.set_facecolor("#111827")
    _draw_fracture_panels(axes[0], axes[1], axes[2], frac, sweep, ref)
    fig.suptitle("Pass 3 — Quasi-static non-Euclidean valve energy → Freund suture fracture → ballistic seed ejection (Bauhinia variegata)", color="#f0f6fc", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


def render_comparison_figure(
    out_png: Path = OUT_PNG,
    overrides: dict[str, float] | None = None,
    seed: int = 2011,
) -> Path:
    """Render the 9-panel dark-theme publication synthesis figure comparing:
      (A) 3D Non-Euclidean Ribbon Morphologies: Narrow Pure-Twist Helicoid (w_tilde=1.36)
          vs Opposing Wide Bauhinia Cylindrical Helical Valves (w_tilde=3.78, +H & -H)
      (B) Supercritical Pitchfork Bifurcation Diagram H*(w_tilde)/kappa_0 & D*(w_tilde)/kappa_0
      (C) Helix Radius R*kappa_0, Pitch P*kappa_0/(2*pi), and Gaussian |K|/kappa_0^2 vs w_tilde
      (D) Cutting-Angle Chirality Sweep theta in [0 deg, 90 deg] (including theta=0 deg ring control)
      (E) Computer-Vision Segmentation of Bauhinia variegata Specimen (MHNT.BOT.2011.3.22)
      (F) 4-System Radius/Pitch Benchmark & Sclerenchyma Bilayer Helical Lobe Oscillation
      (G) Pass 3: desiccation-driven suture energy release rate G(t), Griffith initiation & stick-slip crack a(t)
      (H) Pass 3: Freund crack speed law v/c_R = 1 - Gamma_c/G, arrest & opening fraction vs toughness
      (I) Pass 3: elastic-catapult seed launch speeds, ballistic range, explosive-dispersal benchmarks"""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ov = overrides or {}
    kappa0 = float(ov.get("kappa0", 0.42))
    nu = float(ov.get("nu", 0.30))
    W_cm = float(ov.get("W", 1.80))
    h_cm = float(ov.get("h", 0.095))
    L_cm = float(ov.get("L", 18.0))

    bif = evaluate_width_pitchfork_bifurcation(kappa0=kappa0, nu=nu, seed=seed)
    chir = evaluate_cutting_angle_chirality_sweep(w_tilde=4.0, kappa0=kappa0, nu=nu, seed=seed)
    spec_table = evaluate_pod_and_mimetic_specimens(seed=seed)
    cv_seg = segment_bauhinia_pod_specimen()
    frac = simulate_suture_fracture_ballistic_ejection(overrides=overrides, seed=seed)
    sweep = evaluate_suture_toughness_sweep(seed=seed)
    frac_ref = load_fracture_reference()

    # 3D meshes for Panel A: narrow helicoid vs wide opposing valve pair (+H and -H)
    mesh_narrow = generate_3d_ribbon_mesh(
        L_cm=L_cm, W_cm=0.65, h_cm=h_cm, kappa0=kappa0, nu=nu, theta_deg=45.0, branch=1, seed=seed
    )
    mesh_wide_pos = generate_3d_ribbon_mesh(
        L_cm=L_cm, W_cm=W_cm, h_cm=h_cm, kappa0=kappa0, nu=nu, theta_deg=45.0, branch=1, seed=seed
    )
    mesh_wide_neg = generate_3d_ribbon_mesh(
        L_cm=L_cm, W_cm=W_cm, h_cm=h_cm, kappa0=kappa0, nu=nu, theta_deg=45.0, branch=-1, seed=seed
    )

    fig = plt.figure(figsize=(18.5, 16.4), dpi=150, facecolor="#0b0f17")
    fig.suptitle(
        "Armon, Efrati, Kupferman & Sharon (2011) Science — Non-Euclidean Incompatible Shell Elasticity & Chiral Seed Pod Opening in Bauhinia variegata",
        color="#f0f6fc",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    # Panel A: 3D Ribbon Morphologies (Pure-Twist Helicoid vs Opposing Cylindrical Helical Valves)
    ax_a = fig.add_subplot(3, 3, 1, projection="3d", facecolor="#0b0f17")
    ax_a.plot_surface(
        mesh_narrow["X"] - 8.5,
        mesh_narrow["Y"],
        mesh_narrow["Z"],
        cmap="plasma",
        alpha=0.90,
        linewidth=0.1,
        edgecolor="#1f2937",
    )
    ax_a.plot_surface(
        mesh_wide_pos["X"] + 3.2,
        mesh_wide_pos["Y"],
        mesh_wide_pos["Z"],
        cmap="viridis",
        alpha=0.88,
        linewidth=0.1,
        edgecolor="#1f2937",
    )
    ax_a.plot_surface(
        mesh_wide_neg["X"] + 3.2,
        -mesh_wide_neg["Y"] - 2.5,
        mesh_wide_neg["Z"],
        cmap="YlOrBr",
        alpha=0.85,
        linewidth=0.1,
        edgecolor="#1f2937",
    )
    ax_a.set_title(
        f"(A) 3D Non-Euclidean Ribbon Equilibrium\nNarrow Helicoid (w̃=1.36, R=0) vs Wide Valve Pair (w̃={mesh_wide_pos['eq']['w_tilde']:.2f})",
        color="#e6edf3",
        fontsize=10,
        pad=8,
    )
    ax_a.set_xlabel("X (cm)", color="#8b949e", fontsize=8)
    ax_a.set_ylabel("Y (cm)", color="#8b949e", fontsize=8)
    ax_a.set_zlabel("Z (cm)", color="#8b949e", fontsize=8)
    ax_a.tick_params(colors="#8b949e", labelsize=7)
    ax_a.view_init(elev=20, azim=-58)

    # Panel B: Supercritical Pitchfork Bifurcation H*(w_tilde) & D*(w_tilde)
    ax_b = fig.add_subplot(3, 3, 2, facecolor="#111827")
    w_dense = np.linspace(0.45, 10.0, 220)
    H_dense = np.array([analytical_pitchfork_curvatures(float(w), kappa0, nu)["H"] / kappa0 for w in w_dense])
    D_dense = np.array([analytical_pitchfork_curvatures(float(w), kappa0, nu)["D"] / kappa0 for w in w_dense])
    wt_pts = [r["w_tilde"] for r in bif["rows"]]
    H_sim_pts = [r["H_over_kappa0_sim"] for r in bif["rows"]]
    H_obs_pts = [r["H_over_kappa0_obs"] for r in bif["rows"]]
    D_sim_pts = [r["D_over_kappa0_sim"] for r in bif["rows"]]

    ax_b.plot(w_dense, H_dense, color="#38bdf8", lw=2.4, label="Analytical +H*(w̃) / κ₀")
    ax_b.plot(w_dense, -H_dense, color="#38bdf8", lw=1.8, ls="--", label="Mirror Valve -H*(w̃) / κ₀")
    ax_b.plot(w_dense, D_dense, color="#a78bfa", lw=2.0, ls="-.", label="Deviatoric D*(w̃) / κ₀")
    ax_b.scatter(wt_pts, H_sim_pts, color="#22d3ee", s=46, zorder=5, label=f"L-BFGS-B H/κ₀ (r={bif['pearson_r_H']:.4f})")
    ax_b.scatter(wt_pts, H_obs_pts, facecolors="none", edgecolors="#facc15", s=64, lw=1.5, zorder=6, label="Armon et al. (2011) ref")
    ax_b.scatter(wt_pts, D_sim_pts, color="#c084fc", marker="s", s=34, zorder=5)
    ax_b.axvline(bif["w_tilde_c"], color="#f43f5e", ls=":", lw=1.8, label=f"Critical w̃_c = {bif['w_tilde_c']:.4f}")
    ax_b.axhline(0.5 * (1.0 - nu), color="#34d399", ls=":", lw=1.2, label=f"Asymptote (1-ν)/2 = {0.5*(1-nu):.2f}")
    ax_b.set_title(
        "(B) Supercritical Pitchfork Bifurcation\nHelicoid (H=0) → Cylindrical Helical Ribbon (H≠0)",
        color="#e6edf3",
        fontsize=10,
    )
    ax_b.set_xlabel("Dimensionless Föppl–von Kármán width w̃ = W√(κ₀/h)", color="#c9d1d9", fontsize=9)
    ax_b.set_ylabel("Normalized curvature (H/κ₀, D/κ₀)", color="#c9d1d9", fontsize=9)
    ax_b.tick_params(colors="#8b949e", labelsize=8)
    ax_b.grid(color="#1f2937", alpha=0.6, lw=0.7)
    ax_b.legend(facecolor="#0b0f17", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5, loc="lower right")

    # Panel C: Dimensionless Helix Radius R*kappa_0, Pitch P*kappa_0/(2*pi), and Gaussian |K|/kappa_0^2
    ax_c = fig.add_subplot(3, 3, 3, facecolor="#111827")
    R_sim_pts = [r["R_kappa0_sim"] for r in bif["rows"]]
    R_obs_pts = [r["R_kappa0_obs"] for r in bif["rows"]]
    P_sim_pts = [r["P_kappa0_over_2pi_sim"] for r in bif["rows"]]
    P_obs_pts = [r["P_kappa0_over_2pi_obs"] for r in bif["rows"]]
    K_sim_pts = [abs(r["K_over_kappa0_sq_sim"]) for r in bif["rows"]]

    ax_c.plot(wt_pts, R_sim_pts, "o-", color="#34d399", lw=2.2, label=f"Helix radius R·κ₀ (r={bif['pearson_r_R']:.4f})")
    ax_c.scatter(wt_pts, R_obs_pts, facecolors="none", edgecolors="#facc15", s=65, lw=1.5, label="Ref R·κ₀")
    ax_c.plot(wt_pts, P_sim_pts, "s-", color="#fb923c", lw=2.2, label=f"Helix pitch P·κ₀/(2π) (r={bif['pearson_r_P']:.4f})")
    ax_c.scatter(wt_pts, P_obs_pts, marker="D", facecolors="none", edgecolors="#fde047", s=55, lw=1.4, label="Ref P·κ₀/(2π)")
    ax_c.plot(wt_pts, K_sim_pts, "^--", color="#f472b6", lw=1.8, label="Gaussian |K| / κ₀² (→ 0 for w̃ > w̃_c)")
    ax_c.axvline(bif["w_tilde_c"], color="#f43f5e", ls=":", lw=1.8, label=f"Pitchfork w̃_c = {bif['w_tilde_c']:.2f}")
    ax_c.set_title(
        "(C) Helix Radius R(w̃), Pitch P(w̃) & Isometric Limit\nGauss Incompatibility Relief |K|/κ₀² → 0",
        color="#e6edf3",
        fontsize=10,
    )
    ax_c.set_xlabel("Dimensionless ribbon width w̃ = W√(κ₀/h)", color="#c9d1d9", fontsize=9)
    ax_c.set_ylabel("Dimensionless radius, pitch & |K|/κ₀²", color="#c9d1d9", fontsize=9)
    ax_c.tick_params(colors="#8b949e", labelsize=8)
    ax_c.grid(color="#1f2937", alpha=0.6, lw=0.7)
    ax_c.legend(facecolor="#0b0f17", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5, loc="upper right")

    # Panel D: Cutting-Angle Chirality Sweep theta in [0 deg, 90 deg]
    ax_d = fig.add_subplot(3, 3, 4, facecolor="#111827")
    th_grid = np.linspace(0.0, 90.0, 181)
    eq4 = relax_incompatible_ribbon_equilibrium(w_tilde=4.0, kappa0=kappa0, nu=nu, theta_deg=45.0, seed=seed)
    ks_curve = (eq4["H"] + eq4["D"] * np.cos(np.radians(2.0 * th_grid))) / kappa0
    tau_curve = (eq4["D"] * np.sin(np.radians(2.0 * th_grid))) / kappa0
    th_pts = [r["theta_deg"] for r in chir["rows"]]
    tau_pts = [r["tau_over_kappa0_sim"] for r in chir["rows"]]
    tau_obs = [r["tau_over_kappa0_obs"] for r in chir["rows"]]
    ks_pts = [r["kappa_s_over_kappa0_sim"] for r in chir["rows"]]

    ax_d.plot(th_grid, tau_curve, color="#f97316", lw=2.3, label=f"Geodesic torsion τ(θ)/κ₀ = (D/κ₀)sin(2θ) (r={chir['pearson_r_tau']:.4f})")
    ax_d.plot(th_grid, ks_curve, color="#38bdf8", lw=2.1, ls="--", label="Normal curvature κ_s(θ)/κ₀ = (H + D cos 2θ)/κ₀")
    ax_d.scatter(th_pts, tau_pts, color="#fb923c", s=50, zorder=5)
    ax_d.scatter(th_pts, tau_obs, facecolors="none", edgecolors="#facc15", s=70, lw=1.5, zorder=6, label="Armon et al. (2011) ref τ/κ₀")
    ax_d.scatter(th_pts, ks_pts, color="#38bdf8", marker="s", s=40, zorder=5)
    ax_d.axvline(45.0, color="#34d399", ls=":", lw=1.7, label="Natural Bauhinia valve θ = 45° (maximal twist)")
    ax_d.scatter([0.0, 90.0], [0.0, 0.0], color="#f43f5e", s=85, zorder=7, label="Achiral ring controls θ = 0°, 90° (τ = 0, P = 0)")
    ax_d.set_title(
        "(D) Cutting-Angle Chirality Sweep θ ∈ [0°, 90°] (w̃ = 4.0)\nMaximal Chiral Twist at θ = 45° vs Achiral Rings at θ = 0°, 90°",
        color="#e6edf3",
        fontsize=10,
    )
    ax_d.set_xlabel("Strip cutting angle θ relative to principal +κ₀ axis (deg)", color="#c9d1d9", fontsize=9)
    ax_d.set_ylabel("Normalized strip curvature κ_s/κ₀ & torsion τ/κ₀", color="#c9d1d9", fontsize=9)
    ax_d.tick_params(colors="#8b949e", labelsize=8)
    ax_d.grid(color="#1f2937", alpha=0.6, lw=0.7)
    ax_d.legend(facecolor="#0b0f17", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.3, loc="upper right")

    # Panel E: Computer-Vision Segmentation of Bauhinia variegata Specimen (MHNT.BOT.2011.3.22)
    ax_e = fig.add_subplot(3, 3, 5, facecolor="#111827")
    ax_e.imshow(cv_seg["roi_rgb"])
    ax_e.contour(cv_seg["valve_mask"], levels=[0.5], colors=["#22d3ee"], linewidths=1.1)
    ax_e.plot(cv_seg["cols_x"], cv_seg["upper_cy"], color="#f43f5e", lw=2.0, label="Upper twisted valve centerline")
    ax_e.plot(cv_seg["cols_x"], cv_seg["lower_cy"], color="#a855f7", lw=2.0, ls="--", label="Lower twisted valve centerline")
    ap = cv_seg["apices_xy"]
    ax_e.scatter(
        ap[:, 0],
        ap[:, 1],
        color="#facc15",
        edgecolors="#0b0f17",
        s=75,
        lw=1.3,
        zorder=6,
        label=f"Helical lobe apices (n={cv_seg['n_helical_lobes']}, CV={cv_seg['cv_half_pitch']:.3f})",
    )
    ax_e.set_title(
        f"(E) Bauhinia variegata Specimen Segmentation (MHNT.BOT.2011.3.22)\nDehisced Chiral Valves: Half-Pitch = {cv_seg['mean_half_pitch_px']:.1f}±{cv_seg['std_half_pitch_px']:.1f} px",
        color="#e6edf3",
        fontsize=10,
    )
    ax_e.set_xlabel("ROI X coordinate (px)", color="#c9d1d9", fontsize=9)
    ax_e.set_ylabel("ROI Y coordinate (px)", color="#c9d1d9", fontsize=9)
    ax_e.tick_params(colors="#8b949e", labelsize=8)
    ax_e.legend(facecolor="#0b0f17", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5, loc="lower left")

    # Panel F: 4-System Empirical vs Simulated Radius & Pitch Benchmark
    ax_f = fig.add_subplot(3, 3, 6, facecolor="#111827")
    labels = [
        "Bauhinia\nwide valve\n(w̃=3.78)",
        "Bauhinia\nnarrow strip\n(w̃=1.36)",
        "Erythrina\nvalve\n(w̃=3.69)",
        "Latex\nbilayer\n(w̃=5.93)",
    ]
    x_idx = np.arange(len(labels))
    w_bar = 0.19
    r_sim_b = [r["R_cm_sim"] for r in spec_table["rows"]]
    r_obs_b = [r["R_cm_obs"] for r in spec_table["rows"]]
    p_sim_b = [r["P_cm_sim"] / 5.0 for r in spec_table["rows"]]
    p_obs_b = [r["P_cm_obs"] / 5.0 for r in spec_table["rows"]]

    ax_f.bar(x_idx - 1.5 * w_bar, r_sim_b, width=w_bar, color="#34d399", label=f"Sim R (cm) [r={spec_table['pearson_r_radius']:.4f}]")
    ax_f.bar(x_idx - 0.5 * w_bar, r_obs_b, width=w_bar, color="#059669", hatch="//", edgecolor="#a7f3d0", label="Obs R (cm)")
    ax_f.bar(x_idx + 0.5 * w_bar, p_sim_b, width=w_bar, color="#fb923c", label=f"Sim P/5 (cm) [r={spec_table['pearson_r_pitch']:.4f}]")
    ax_f.bar(x_idx + 1.5 * w_bar, p_obs_b, width=w_bar, color="#ea580c", hatch="//", edgecolor="#fed7aa", label="Obs P/5 (cm)")
    ax_f.set_xticks(x_idx)
    ax_f.set_xticklabels(labels, color="#e6edf3", fontsize=8)
    ax_f.set_ylabel("Helix Radius R (cm) & Scaled Pitch P/5 (cm)", color="#c9d1d9", fontsize=9)
    ax_f.set_title(
        f"(F) Multi-System Radius & Pitch Validation\nRMSE_R = {spec_table['rmse_radius_cm']:.4f} cm, RMSE_P = {spec_table['rmse_pitch_cm']:.4f} cm",
        color="#e6edf3",
        fontsize=10,
    )
    ax_f.tick_params(colors="#8b949e", labelsize=8)
    ax_f.grid(color="#1f2937", axis="y", alpha=0.6, lw=0.7)
    ax_f.legend(facecolor="#0b0f17", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5, loc="upper right")

    # Panels G-I (Pass 3): suture fracture & ballistic seed ejection
    ax_g = fig.add_subplot(3, 3, 7, facecolor="#111827")
    ax_h = fig.add_subplot(3, 3, 8, facecolor="#111827")
    ax_i = fig.add_subplot(3, 3, 9, facecolor="#111827")
    _draw_fracture_panels(ax_g, ax_h, ax_i, frac, sweep, frac_ref)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


def render_growth_gif(out_gif: Path = OUT_GIF, seed: int = 2011) -> Path:
    """Render an animated GIF of progressive hygroscopic desiccation shrinkage
    (kappa_0 increasing from 0.02 to 0.42 cm^-1) driving spontaneous chiral
    opening of the two opposed *Bauhinia variegata* seed pod valves."""
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    k0_steps = np.linspace(0.02, 0.42, 12)

    for k0 in k0_steps:
        m_pos = generate_3d_ribbon_mesh(L_cm=18.0, W_cm=1.80, h_cm=0.095, kappa0=float(k0), nu=0.30, branch=1, seed=seed)
        m_neg = generate_3d_ribbon_mesh(L_cm=18.0, W_cm=1.80, h_cm=0.095, kappa0=float(k0), nu=0.30, branch=-1, seed=seed)
        wt = m_pos["eq"]["w_tilde"]
        wc = m_pos["eq"]["w_tilde_c"]
        reg_str = "Helicoid (w̃ ≤ w̃_c)" if wt <= wc else "Cylindrical Helical Ribbon (w̃ > w̃_c)"

        fig = plt.figure(figsize=(6.4, 4.8), dpi=95, facecolor="#0b0f17")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0b0f17")
        ax.plot_surface(m_pos["X"], m_pos["Y"] + 1.2, m_pos["Z"], cmap="viridis", alpha=0.88, linewidth=0.1)
        ax.plot_surface(m_neg["X"], -m_neg["Y"] - 1.2, m_neg["Z"], cmap="YlOrBr", alpha=0.86, linewidth=0.1)
        ax.set_title(
            f"Bauhinia Pod Dehiscence: κ₀={k0:.2f} cm⁻¹, w̃={wt:.2f} (w̃_c={wc:.2f})\n{reg_str}",
            color="#f0f6fc",
            fontsize=9.5,
        )
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-6.5, 6.5)
        ax.set_zlim(-9.0, 9.0)
        ax.set_xlabel("X (cm)", color="#8b949e", fontsize=8)
        ax.set_ylabel("Y (cm)", color="#8b949e", fontsize=8)
        ax.set_zlabel("Z (cm)", color="#8b949e", fontsize=8)
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.view_init(elev=20, azim=-55)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(Image.fromarray(buf[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)

    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return out_gif


def render_fracture_gif(out_gif: Path = OUT_FRACTURE_GIF, seed: int = 2011, n_frames: int = 36) -> Path:
    """Pass 3 animated GIF: 2-D side view of the pod suture. Frames run over the
    desiccation timeline (hours) with a magnified millisecond inset during each
    stick-slip crack jump; released seeds follow their drag-ballistic arcs."""
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frac = simulate_suture_fracture_ballistic_ejection(seed=seed)
    ref = load_fracture_reference()
    P = frac["params"]
    L_cm = 100.0 * P["L_m"]
    xs_cm = 100.0 * frac["xs_field_m"]
    gam = frac["Gamma_field"]
    dry = frac["drying"]
    t_end = float(dry["t_s"][-1])
    jumps = frac["jumps"]
    # time samples: dense around the jumps, sparse elsewhere
    t_samples = list(np.linspace(0.0, t_end, n_frames // 2))
    for j in jumps:
        t_samples += [j["t_start_s"] - 1.0, j["t_start_s"] + 0.5 * j["run_time_s"], j["t_start_s"] + 1.0]
    t_samples = sorted(set(float(t) for t in t_samples))

    r_seed = _fp(ref, "seed_parameters", "r_seed_m")
    C_d = _fp(ref, "seed_parameters", "C_d")
    rho_air = _fp(ref, "seed_parameters", "rho_air_kg_m3")
    g = _fp(ref, "seed_parameters", "g_m_s2")
    ang = _fp(ref, "seed_parameters", "launch_angle_deg")
    z0 = _fp(ref, "seed_parameters", "launch_height_m")

    def crack_pos_at(t: float) -> float:
        a = P["a0_m"]
        for j in jumps:
            if t >= j["t_start_s"] + j["run_time_s"]:
                a = j["a_to_m"]
            elif t >= j["t_start_s"]:
                a = j["a_from_m"] + (j["a_to_m"] - j["a_from_m"]) * (t - j["t_start_s"]) / max(j["run_time_s"], 1e-12)
                break
        return a

    def seed_xy(s: dict[str, Any], t: float) -> tuple[float, float] | None:
        if not s["released"] or t < s["t_eject_s"]:
            return None
        dt_f = min(t - s["t_eject_s"], 3.0)
        # drag-ballistic position by explicit integration (short, coarse)
        A = math.pi * r_seed**2
        k = 0.5 * rho_air * C_d * A / P["m_seed_kg"]
        th = math.radians(ang)
        x, z, vx, vz = 0.0, z0, s["v_seed_m_s"] * math.cos(th), s["v_seed_m_s"] * math.sin(th)
        n = 200
        h_ = dt_f / n
        for _ in range(n):
            sp = math.hypot(vx, vz)
            vx += -k * sp * vx * h_
            vz += (-g - k * sp * vz) * h_
            x += vx * h_
            z += vz * h_
            if z <= 0.0:
                z = 0.0
                break
        return x, z

    frames: list[Image.Image] = []
    for t in t_samples:
        a_now = crack_pos_at(t)
        G_now = float(np.interp(t, dry["t_s"], dry["G_J_m2"]))
        k0_now = float(np.interp(t, dry["t_s"], dry["kappa0_cm_inv"]))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=90, facecolor="#0b0f17", gridspec_kw={"width_ratios": [1.15, 1.0]})
        for ax in (ax1, ax2):
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#8b949e", labelsize=7)
        # left: suture + toughness profile + crack tip
        ax1.fill_between(xs_cm, 0, gam, color="#f43f5e", alpha=0.25, label="Γ_c(a) suture toughness")
        ax1.axhline(G_now, color="#38bdf8", lw=2.0, label=f"G(t) = {G_now:.0f} J/m²")
        ax1.axvspan(0.0, 100.0 * a_now, color="#34d399", alpha=0.25, label=f"opened suture a = {100*a_now:.1f} cm")
        ax1.axvline(100.0 * a_now, color="#34d399", lw=2.2)
        for s in frac["seeds"]:
            col = "#fb923c" if (s["released"] and t >= s["t_eject_s"]) else "#facc15"
            ax1.scatter([100.0 * s["a_m"]], [0.08 * max(gam.max(), G_now)], marker="o", s=70, color=col, edgecolors="#0b0f17", zorder=5)
        ax1.set_xlim(0.0, L_cm)
        ax1.set_ylim(0.0, 1.15 * max(float(gam.max()), float(dry["G_J_m2"].max())))
        ax1.set_xlabel("Suture coordinate a (cm, stylar tip → base)", color="#c9d1d9", fontsize=8)
        ax1.set_ylabel("J/m²", color="#c9d1d9", fontsize=8)
        in_jump = any(j["t_start_s"] <= t <= j["t_start_s"] + j["run_time_s"] for j in jumps)
        ax1.set_title(f"t = {t/3600:.1f} h, κ₀ = {k0_now:.3f} cm⁻¹" + ("  — CRACK RUNNING (ms)" if in_jump else ""), color="#f0f6fc", fontsize=9)
        ax1.legend(facecolor="#0b0f17", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=6.8, loc="upper right")
        # right: ballistic arcs
        ax2.axhline(0.0, color="#8b949e", lw=1.0)
        ax2.plot([0.0], [z0], marker="s", color="#a78bfa", ms=8)
        for s in frac["seeds"]:
            if s["released"]:
                tr = [seed_xy(s, s["t_eject_s"] + u) for u in np.linspace(0.0, 3.0, 60) if s["t_eject_s"] + u <= t]
                tr = [p for p in tr if p is not None]
                if tr:
                    xs_t = [p[0] for p in tr]
                    zs_t = [p[1] for p in tr]
                    ax2.plot(xs_t, zs_t, color="#fb923c", lw=1.2, alpha=0.8)
                    ax2.scatter([xs_t[-1]], [zs_t[-1]], color="#fb923c", s=40, zorder=5)
        ax2.set_xlim(-0.2, max(2.5, 1.3 * frac["ballistics"]["range_m"]))
        ax2.set_ylim(-0.2, z0 + 1.0)
        ax2.set_xlabel("Horizontal distance (m)", color="#c9d1d9", fontsize=8)
        ax2.set_ylabel("Height (m)", color="#c9d1d9", fontsize=8)
        ax2.set_title(f"Seed ballistics: v₀ ≈ {frac['v_seed_m_s']:.2f} m/s, range {frac['ballistics']['range_m']:.2f} m ({frac['n_ejected']} ejected)", color="#f0f6fc", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(Image.fromarray(buf[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=160, loop=0)
    return out_gif


# =============================================================================
# 7. Quantitative MorphoEval Measurements & 12-Point Self-Test Suite
# =============================================================================
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 15 reference-anchored quantitative morphometric measurements for
    `morphoeval/measure_outputs.py` (9 Pass 2 + 6 Pass 3 fracture/ballistic)."""
    bif = evaluate_width_pitchfork_bifurcation(seed=2011)
    chir = evaluate_cutting_angle_chirality_sweep(seed=2011)
    spec_table = evaluate_pod_and_mimetic_specimens(seed=2011)
    cv_seg = segment_bauhinia_pod_specimen()
    frac = simulate_suture_fracture_ballistic_ejection(seed=2011)
    sweep = evaluate_suture_toughness_sweep(seed=2011)
    b0_info = compute_target_curvature_tensor(kappa0=0.42, theta_deg=45.0)
    k0_bilayer = compute_sclerenchyma_bilayer_kappa0(delta_eps_hygro=0.0266, h_valve_cm=0.095)

    return [
        {
            "name": "Armon et al. (2011) critical Föppl-von Kármán pitchfork width w_tilde_c (nu=0.30)",
            "value": round(bif["w_tilde_c"], 4),
            "unit": "dimensionless",
            "method": "Exact non-Euclidean shell pitchfork threshold w_tilde_c = (16*(1+nu)/sqrt(1-nu^2))^(1/4)",
            "reference_value": 2.1609,
            "reference_source": "Armon, Efrati, Kupferman & Sharon (2011) Science 333:1726-1730 Eq. 3 & SOM",
            "note": f"narrow helicoid w_tilde=1.0 gives H/kappa_0 = {bif['rows'][1]['H_over_kappa0_sim']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Incompatible saddle Gaussian target curvature det(b_0) = -kappa_0^2 (kappa_0=0.42 cm^-1)",
            "value": round(b0_info["det_b0_gaussian_K"], 4),
            "unit": "cm^-2",
            "method": "Determinant of rotated +-45 deg sclerenchyma target curvature tensor b_0(theta) vs flat reference metric K(a_0)=0",
            "reference_value": round(-(0.42**2), 4),
            "reference_source": "Efrati, Sharon & Kupferman (2009) JMPS 57:762-775 & Armon et al. (2011) Science",
            "note": f"Tr(b_0) = {b0_info['trace_b0']:.4f} cm^-1, bilayer kappa_0 = {k0_bilayer:.4f} cm^-1",
            "derived_from": "script",
        },
        {
            "name": "Supercritical pitchfork mean curvature H*(w_tilde)/kappa_0 correlation across w_tilde in [0.5, 10]",
            "value": round(bif["pearson_r_H"], 4),
            "unit": "Pearson r",
            "method": f"L-BFGS-B non-Euclidean plate relaxation vs benchmark across 11 widths (p={bif['p_value_H']:.2e}, RMSE={bif['rmse_H']:.4f})",
            "reference_value": 1.0,
            "reference_source": "Armon et al. (2011) Science 333:1726-1730 Fig. 3B-C",
            "note": f"wide asymptote H(w_tilde=10)/kappa_0 = {bif['rows'][-1]['H_over_kappa0_sim']:.4f} vs (1-nu)/2 = 0.3500",
            "derived_from": "script",
        },
        {
            "name": "Dimensionless helix radius R(w_tilde)*kappa_0 transition correlation across w_tilde_c",
            "value": round(bif["pearson_r_R"], 4),
            "unit": "Pearson r",
            "method": f"L-BFGS-B helix radius R(w_tilde)*kappa_0 across helicoid-to-ribbon transition (p={bif['p_value_R']:.2e}, RMSE={bif['rmse_R']:.4f})",
            "reference_value": 1.0,
            "reference_source": "Armon et al. (2011) Science 333:1726-1730 Fig. 3D & Fig. 4",
            "note": f"narrow helicoid (w_tilde <= w_tilde_c) R*kappa_0 = {bif['rows'][2]['R_kappa0_sim']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Dimensionless helical pitch P(w_tilde)*kappa_0/(2*pi) transition correlation across w_tilde",
            "value": round(bif["pearson_r_P"], 4),
            "unit": "Pearson r",
            "method": f"L-BFGS-B helical pitch P(w_tilde)*kappa_0/(2*pi) across 11 dimensionless widths (p={bif['p_value_P']:.2e}, RMSE={bif['rmse_P']:.4f})",
            "reference_value": 1.0,
            "reference_source": "Armon et al. (2011) Science 333:1726-1730 Fig. 3D",
            "note": f"RMSE_P = {bif['rmse_P']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Cutting-angle chirality torsion tau(theta)/kappa_0 correlation across theta in [0 deg, 90 deg]",
            "value": round(chir["pearson_r_tau"], 4),
            "unit": "Pearson r",
            "method": f"Geodesic torsion tau(theta)/kappa_0 = (D/kappa_0)*sin(2*theta) across 7 cutting angles (p={chir['p_value_tau']:.2e}, RMSE={chir['rmse_tau']:.4f})",
            "reference_value": 1.0,
            "reference_source": "Armon et al. (2011) Science 333:1726-1730 Fig. 2 & SOM",
            "note": f"maximal twist at theta=45 deg: tau/kappa_0 = {chir['max_twist_theta45_tau_over_kappa0']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Achiral ring negative control helical pitch P at cutting angle theta = 0 deg",
            "value": round(chir["achiral_theta0_pitch_cm"], 6),
            "unit": "cm",
            "method": "Helical pitch P(theta=0 deg) when principal intrinsic curvature aligns with strip axis (tau = D*sin(0) = 0)",
            "reference_value": 0.0,
            "reference_source": "Armon et al. (2011) Science 333:1726-1730 Fig. 2A (theta=0 deg achiral circular ring)",
            "note": f"theta=90 deg pitch = {chir['achiral_theta90_pitch_cm']:.6f} cm, ring radius R(0 deg) = {chir['rows'][0]['R_cm_sim']:.4f} cm",
            "derived_from": "script",
        },
        {
            "name": "4-system legume & biomimetic latex helical pitch P correlation (Bauhinia, Erythrina, Latex)",
            "value": round(spec_table["pearson_r_pitch"], 4),
            "unit": "Pearson r",
            "method": f"Predicted vs empirical helical pitch P across 4 experimental systems (p={spec_table['p_value_pitch']:.2e}, RMSE={spec_table['rmse_pitch_cm']:.4f} cm)",
            "reference_value": 1.0,
            "reference_source": "Armon et al. (2011) Science 333:1726-1730 Figs. 1-4",
            "note": f"4-system helix radius Pearson r = {spec_table['pearson_r_radius']:.4f}, RMSE = {spec_table['rmse_radius_cm']:.4f} cm",
            "derived_from": "script",
        },
        {
            "name": "Bauhinia variegata specimen segmented helical lobe half-pitch CV (bauhinia_chiral_seed_pod_specimen.jpg)",
            "value": round(cv_seg["cv_half_pitch"], 4),
            "unit": "CV (std/mean)",
            "method": f"Inter-lobe half-pitch spacing dispersion across n={cv_seg['n_helical_lobes']} segmented helical twist lobes on both dehisced valves",
            "reference_value": 0.24,
            "reference_source": "Armon et al. (2011) Science Fig. 1A & Muséum de Toulouse MHNT.BOT.2011.3.22",
            "note": f"mean half-pitch = {cv_seg['mean_half_pitch_px']:.2f} ± {cv_seg['std_half_pitch_px']:.2f} px",
            "derived_from": "output",
        },
        # ---- Pass 3: quasi-static energy -> suture fracture -> ballistic ejection ----
        {
            "name": "Suture energy release rate G_final = Delta_e(kappa_0=0.42) * W / h from the quasi-static minimizer (fully dried pod)",
            "value": round(frac["G_final_J_m2"], 2),
            "unit": "J/m^2",
            "method": f"B_eff (kappa_0^2 - e_final) W / h with B_eff = Y h^3/(12(1+nu)) = {frac['B_eff_J']:.4e} J, e_final from L-BFGS-B relaxation (stored {1e3*frac['stored_energy_per_valve_J']:.2f} mJ per valve)",
            "reference_value": round(frac["params"]["Gamma_c_mean_J_m2"], 2),
            "reference_source": "Griffith criterion vs model suture toughness Gamma_c (suture_fracture_ballistic_reference.json, provenance=model; Freund 1990); G_final > Gamma_c is required for dehiscence",
            "note": f"Delta_e_final = {frac['delta_e_final_J_m2']:.3f} J/m^2; Gamma_c is a free model parameter, not a Bauhinia measurement",
            "derived_from": "script",
        },
        {
            "name": "Griffith initiation time t_c of suture dehiscence on the desiccation curve kappa_0(t) = kappa_0,f (1 - exp(-t/tau_dry))",
            "value": round(frac["t_c_h"], 2),
            "unit": "h",
            "method": f"First drying time at which G(t) exceeds the local suture toughness at the stylar flaw (kappa_0(t_c) = {frac['kappa0_at_initiation_cm_inv']:.4f} cm^-1, G(t_c) = {frac['G_at_initiation_J_m2']:.1f} J/m^2)",
            "reference_value": round(-frac["params"]["tau_dry_s"] / 3600.0 * math.log(1.0 - math.sqrt(frac["Gamma_at_flaw_J_m2"] / frac["G_final_J_m2"])), 2),
            "reference_source": "Analytical estimate from G ~ kappa_0^2 scaling: t_c = -tau_dry ln(1 - sqrt(Gamma_c(a_0)/G_final)); tau_dry = 72 h is a model constant",
            "note": "tau_dry and Gamma_c are provenance=model; the measurement tests the coupling, not a Bauhinia field observation",
            "derived_from": "script",
        },
        {
            "name": "Freund dynamic crack speed law v/c_R = 1 - Gamma_c/G across the 7-point suture toughness sweep (homogeneous suture, fully dried)",
            "value": round(sweep["pearson_r_v_vs_freund"], 4),
            "unit": "Pearson r",
            "method": f"Explicit integration of da/dt = c_R max(0, 1 - Gamma_c(a)/G) vs analytical 1 - Gamma/G at Gamma/G in {{0.2..1.2}} (p={sweep['p_value_v']:.2e}, RMSE={sweep['rmse_v_over_cR']:.4f}; c_R = {frac['params']['c_R_m_s']:.1f} m/s)",
            "reference_value": 1.0,
            "reference_source": "Freund LB (1990) Dynamic Fracture Mechanics, Cambridge Univ. Press, Ch. 7 mode-I equation of motion G_dyn(v) = (1 - v/c_R) G",
            "note": f"no-fracture negative control Gamma_c >= G: opening fraction = {sweep['no_fracture_control_opening_fraction']:.4f} (flaw only); heterogeneous suture at Gamma = G opens {sweep['heterogeneous_opening_fraction_at_Gamma_eq_G']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Drying-coupled stick-slip pod opening fraction a*/L and cumulative dynamic crack run time (Gamma_c = 120 J/m^2)",
            "value": round(frac["opening_fraction"], 4),
            "unit": "fraction of suture length",
            "method": f"{frac['n_jumps']} stick-slip crack jumps over {frac['t_c_h']:.1f}-{(frac['jumps'][-1]['t_start_s']/3600.0 if frac['jumps'] else float('nan')):.1f} h; cumulative run time {1e3*frac['total_crack_run_time_s']:.3f} ms; mean dynamic speed {frac['v_crack_mean_m_s']:.1f} m/s = {frac['v_crack_over_cR']:.4f} c_R",
            "reference_value": 1.0,
            "reference_source": "Complete dehiscence of both Bauhinia variegata valves (Armon et al. 2011 Fig. 1; vendored specimen MHNT.BOT.2011.3.22 shows fully separated valves)",
            "note": f"Gamma_c/G_final = {frac['params']['Gamma_c_mean_J_m2']/frac['G_final_J_m2']:.3f}; opening fraction is emergent from the toughness field, not prescribed",
            "derived_from": "script",
        },
        {
            "name": "Elastic-catapult seed launch speed v_seed (drying-coupled mean over ejected seeds)",
            "value": round(frac["v_seed_m_s"], 3),
            "unit": "m/s",
            "method": f"v = sqrt(2 eta Delta_U_seg / (m_seg/3 + m_seed)), eta = {frac['params']['eta']:.2f}, m_seed = {1e3*frac['params']['m_seed_kg']:.2f} g, l_seg = {100*frac['params']['l_seg_m']:.1f} cm; {frac['n_ejected']}/{frac['params']['n_seeds']} seeds ejected; fully dried bound {frac['v_seed_fully_dried_m_s']:.3f} m/s",
            "reference_value": 10.0,
            "reference_source": "Hofhuis et al. (2016) Cell 166:222-233 Cardamine hirsuta launch speed ~10 m/s (transfer_boundary: different taxon, order-of-magnitude benchmark; no Bauhinia launch-speed measurement available offline)",
            "note": f"Bauhinia predicted ~{10.0/max(frac['v_seed_m_s'],1e-9):.1f}x slower than Cardamine and ~{70.0/max(frac['v_seed_m_s'],1e-9):.0f}x slower than Hura crepitans (70 m/s); eta, m_seed, rho are provenance=model",
            "derived_from": "script",
        },
        {
            "name": "Seed ballistic range with quadratic drag from a 3 m launch height (45 deg)",
            "value": round(frac["ballistics"]["range_m"], 3),
            "unit": "m",
            "method": f"RK4 of m dv/dt = m g - (1/2) rho_air C_d A |v| v (C_d = 1.0, r = 6 mm); flight time {frac['ballistics']['flight_time_s']:.2f} s",
            "reference_value": round(frac["ballistics"]["drag_free_bound_m"], 3),
            "reference_source": "Drag-free analytical upper bound x = v cos(a)(v sin(a) + sqrt(v^2 sin^2(a) + 2 g z_0))/g (theory bound, must exceed the drag range)",
            "note": "no published Bauhinia variegata dispersal-distance measurement was verifiable offline; the reference is a consistency bound, not an observation",
            "derived_from": "script",
        },
    ]


def self_test() -> bool:
    """Run comprehensive 8-point verification suite for Pass 2."""
    # 1. Cryptographic SHA-256 verification of all 5 dataset files
    digests = verify_dataset_integrity()
    assert len(digests) == 5, "Expected 5 SHA-256-verified dataset files"

    # 2. Law 1: Incompatible saddle target curvature det(b_0) = -kappa_0^2 < 0 & Tr(b_0) = 0 across all theta
    for th in (0.0, 30.0, 45.0, 60.0, 90.0):
        b0 = compute_target_curvature_tensor(kappa0=0.42, theta_deg=th)
        assert abs(b0["trace_b0"]) < 1e-12
        assert math.isclose(b0["det_b0_gaussian_K"], -(0.42**2), rel_tol=1e-12)
        assert b0["gauss_incompatible"] is True

    # 3. Law 2 & Negative Controls: Narrow Pure-Twist Helicoid (w_tilde < w_tilde_c -> H=0, R=0),
    #    achiral circular ring controls (theta=0 deg & 90 deg -> tau=0, P=0), and kappa_0=0 flat control
    eq_narrow = relax_incompatible_ribbon_equilibrium(w_tilde=1.20, kappa0=0.42, nu=0.30, theta_deg=45.0)
    assert abs(eq_narrow["H"]) < 1e-7 and abs(eq_narrow["R_cm"]) < 1e-7
    assert eq_narrow["P_cm"] > 10.0 and eq_narrow["energy_drop_ratio"] > 0.70

    eq_ring0 = relax_incompatible_ribbon_equilibrium(w_tilde=4.0, kappa0=0.42, nu=0.30, theta_deg=0.0)
    eq_ring90 = relax_incompatible_ribbon_equilibrium(w_tilde=4.0, kappa0=0.42, nu=0.30, theta_deg=90.0)
    assert abs(eq_ring0["tau"]) < 1e-12 and abs(eq_ring0["P_cm"]) < 1e-12
    assert abs(eq_ring90["tau"]) < 1e-12 and abs(eq_ring90["P_cm"]) < 1e-12

    eq_flat = relax_incompatible_ribbon_equilibrium(w_tilde=4.0, kappa0=0.0, nu=0.30, theta_deg=45.0)
    assert eq_flat["H"] == 0.0 and eq_flat["D"] == 0.0 and eq_flat["R_cm"] == 0.0

    # 4. Law 3: Supercritical pitchfork bifurcation across w_tilde in [0.5, 10.0]
    bif = evaluate_width_pitchfork_bifurcation(seed=2011)
    assert math.isclose(bif["w_tilde_c"], 2.1609, rel_tol=1e-4)
    assert bif["pearson_r_H"] > 0.999 and bif["pearson_r_R"] > 0.999 and bif["pearson_r_P"] > 0.999

    # 5. Cutting-angle chirality sweep & opposing valve handedness inversion
    chir = evaluate_cutting_angle_chirality_sweep(seed=2011)
    assert chir["pearson_r_tau"] > 0.999 and chir["pearson_r_P"] > 0.999
    eq_pos = relax_incompatible_ribbon_equilibrium(w_tilde=3.781, kappa0=0.42, nu=0.30, branch=1)
    eq_neg = relax_incompatible_ribbon_equilibrium(w_tilde=3.781, kappa0=0.42, nu=0.30, branch=-1)
    assert math.isclose(eq_pos["H"], -eq_neg["H"], rel_tol=1e-9)

    # 6. Multi-specimen legume & biomimetic latex validation
    sp = evaluate_pod_and_mimetic_specimens(seed=2011)
    assert sp["pearson_r_radius"] > 0.999 and sp["pearson_r_pitch"] > 0.995

    # 7. Biological specimen image segmentation checks
    cv_seg = segment_bauhinia_pod_specimen()
    assert cv_seg["n_helical_lobes"] >= 6
    assert 0.05 < cv_seg["cv_half_pitch"] < 0.35
    assert cv_seg["valve_area_fraction"] > 0.15

    # 8. Two-run bitwise determinism check on L-BFGS-B non-Euclidean shell solver
    m1 = generate_3d_ribbon_mesh(seed=2011)
    m2 = generate_3d_ribbon_mesh(seed=2011)
    assert np.allclose(m1["X"], m2["X"], atol=0.0)
    assert np.allclose(m1["Y"], m2["Y"], atol=0.0)
    assert np.allclose(m1["Z"], m2["Z"], atol=0.0)

    # ---- Pass 3 checks: quasi-static energy -> suture fracture -> ballistic ejection ----
    # 9. Energy bookkeeping: B_eff prefactor, released energy >= 0, monotone in kappa_0,
    #    zero for the flat kappa_0 = 0 control; G scales with W/h.
    assert math.isclose(bending_energy_prefactor(4.2e8, 9.5e-4, 0.30), 4.2e8 * 9.5e-4**3 / (12 * 1.3), rel_tol=1e-12)
    rel0 = released_valve_energy_density(0.0, 0.018, 9.5e-4, 4.2e8, 0.30)
    assert rel0["delta_e_J_m2"] == 0.0 and rel0["G_J_m2"] == 0.0
    dE_prev = -1.0
    for k0 in (0.1, 0.2, 0.3, 0.42):
        rel = released_valve_energy_density(k0, 0.018, 9.5e-4, 4.2e8, 0.30)
        assert rel["delta_e_J_m2"] > dE_prev >= -1.0 and rel["e_final"] <= rel["e_init"]
        dE_prev = rel["delta_e_J_m2"]
    assert math.isclose(rel["G_J_m2"], rel["delta_e_J_m2"] * 0.018 / 9.5e-4, rel_tol=1e-12)

    # 10. Freund law: v = c_R (1 - Gamma/G), arrest for Gamma >= G, bounded by c_R, monotone in G
    c_R = rayleigh_wave_speed(4.2e8, 800.0, 0.30)
    assert 300.0 < c_R < 600.0
    assert freund_crack_speed(200.0, 200.0, c_R) == 0.0 and freund_crack_speed(200.0, 250.0, c_R) == 0.0
    assert math.isclose(freund_crack_speed(200.0, 50.0, c_R), 0.75 * c_R, rel_tol=1e-12)
    assert freund_crack_speed(100.0, 50.0, c_R) < freund_crack_speed(300.0, 50.0, c_R) < c_R
    hom = heterogeneous_suture_toughness(361, 100.0, 0.0)
    assert np.allclose(hom, 100.0)
    crack_run = integrate_suture_crack(250.0, hom, 0.18, c_R, 0.002)
    assert crack_run["opening_fraction"] == 1.0 and not crack_run["arrested"]
    assert math.isclose(crack_run["v_over_cR_mean"], 0.6, rel_tol=2e-3)
    crack_stop = integrate_suture_crack(100.0, hom, 0.18, c_R, 0.002)
    assert crack_stop["arrested"] and crack_stop["a_final_m"] == 0.002

    # 11. Drying-coupled stick-slip dehiscence + catapult + ballistics (default parameters)
    frac = simulate_suture_fracture_ballistic_ejection(seed=2011)
    assert frac["initiated"] and 0.0 < frac["t_c_h"] < 6.0 * frac["params"]["tau_dry_s"] / 3600.0
    assert frac["G_at_initiation_J_m2"] >= frac["Gamma_at_flaw_J_m2"] * 0.99
    assert frac["opening_fraction"] > 0.95 and frac["n_ejected"] == frac["params"]["n_seeds"]
    assert 0.0 < frac["v_crack_mean_m_s"] < frac["params"]["c_R_m_s"]
    assert 0.0 < frac["v_seed_m_s"] <= frac["v_seed_fully_dried_m_s"] < 10.0
    assert frac["ballistics"]["range_m"] < frac["ballistics"]["drag_free_bound_m"]
    t_ej = [s["t_eject_s"] for s in frac["seeds"] if s["released"]]
    assert all(t_ej[i] <= t_ej[i + 1] for i in range(len(t_ej) - 1))  # seeds eject tip -> base
    # no-fracture negative control: Gamma_c above the fully-dried G never initiates
    frac_none = simulate_suture_fracture_ballistic_ejection(overrides={"gamma_c": 2.0 * frac["G_final_J_m2"]}, seed=2011)
    assert (not frac_none["initiated"]) and frac_none["n_ejected"] == 0 and frac_none["v_seed_m_s"] == 0.0
    # energy partition: doubling eta multiplies v by sqrt(2)
    frac_eta = simulate_suture_fracture_ballistic_ejection(overrides={"eta": 2.0 * frac["params"]["eta"]}, seed=2011)
    assert math.isclose(frac_eta["v_seed_fully_dried_m_s"], math.sqrt(2.0) * frac["v_seed_fully_dried_m_s"], rel_tol=1e-9)

    # 12. Toughness sweep reproduces the Freund law and the arrest transition; deterministic
    sweep = evaluate_suture_toughness_sweep(seed=2011)
    assert sweep["pearson_r_v_vs_freund"] > 0.999 and sweep["rmse_v_over_cR"] < 0.01
    assert sweep["no_fracture_control_opening_fraction"] < 0.02
    ofs = [r["opening_fraction"] for r in sweep["rows"]]
    assert ofs[0] == 1.0 and ofs[-1] < 0.1
    frac2 = simulate_suture_fracture_ballistic_ejection(seed=2011)
    assert frac2["t_c_s"] == frac["t_c_s"] and frac2["v_seed_m_s"] == frac["v_seed_m_s"] and frac2["opening_fraction"] == frac["opening_fraction"]

    meas = measure_morphoeval()
    assert len(meas) == 15
    print(
        f"[self_test] 2007-liang-chiral-seed-pod-opening.py (v3) PASSED all 12 checks "
        f"(w_tilde_c={bif['w_tilde_c']:.4f}, pitchfork r_H={bif['pearson_r_H']:.4f}, "
        f"chirality r_tau={chir['pearson_r_tau']:.4f}, 4-system r_P={sp['pearson_r_pitch']:.4f}, "
        f"specimen lobes={cv_seg['n_helical_lobes']}, CV={cv_seg['cv_half_pitch']:.4f}; "
        f"Pass 3: G_final={frac['G_final_J_m2']:.1f} J/m^2, t_c={frac['t_c_h']:.1f} h, "
        f"{frac['n_jumps']} stick-slip jumps, <v_crack>={frac['v_crack_mean_m_s']:.1f} m/s, "
        f"v_seed={frac['v_seed_m_s']:.2f} m/s, range={frac['ballistics']['range_m']:.2f} m, Freund r={sweep['pearson_r_v_vs_freund']:.4f})."
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run 12-check verification suite")
    ap.add_argument("--skip-gif", action="store_true", help="Skip animated GIF rendering")
    ap.add_argument("--kappa0", type=float, default=None, help="Override intrinsic shrinkage curvature kappa_0 (cm^-1)")
    ap.add_argument("--nu", type=float, default=None, help="Override Poisson ratio nu")
    ap.add_argument("--W", type=float, default=None, help="Override pod valve width W (cm)")
    ap.add_argument("--h", type=float, default=None, help="Override pod valve thickness h (cm)")
    ap.add_argument("--L", type=float, default=None, help="Override pod valve length L (cm)")
    ap.add_argument("--seed", type=int, default=2011, help="Deterministic RNG seed (default: 2011)")
    ap.add_argument("--gamma-c", type=float, default=None, help="Pass 3: override mean suture fracture toughness Gamma_c (J/m^2; default 120 from suture_fracture_ballistic_reference.json)")
    ap.add_argument("--eta", type=float, default=None, help="Pass 3: override elastic-catapult energy conversion efficiency eta (default 0.5)")
    ap.add_argument("--m-seed", type=float, default=None, help="Pass 3: override seed mass (kg; default 4e-4)")
    ap.add_argument("--fracture-json", action="store_true", help="Pass 3: run only the suture-fracture/ballistic coupling, print a JSON summary and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    overrides = {
        k: v
        for k, v in {
            "kappa0": args.kappa0,
            "nu": args.nu,
            "W": args.W,
            "h": args.h,
            "L": args.L,
            "gamma_c": args.gamma_c,
            "eta": args.eta,
            "m_seed": args.m_seed,
        }.items()
        if v is not None
    }
    if args.fracture_json:
        frac = simulate_suture_fracture_ballistic_ejection(overrides=overrides, seed=args.seed)
        summary = {k: frac[k] for k in ("G_final_J_m2", "delta_e_final_J_m2", "stored_energy_per_valve_J", "Gamma_at_flaw_J_m2", "initiated", "t_c_h", "kappa0_at_initiation_cm_inv", "n_jumps", "opening_fraction", "total_crack_run_time_s", "v_crack_mean_m_s", "v_crack_over_cR", "t_coil_s", "n_ejected", "v_seed_m_s", "v_seed_fully_dried_m_s", "ballistics")}
        summary["params"] = frac["params"]
        summary["seeds"] = frac["seeds"]
        print(json.dumps(summary, indent=2, default=float))
        return 0
    png = render_comparison_figure(OUT_PNG, overrides=overrides, seed=args.seed)
    print(f"[render] Wrote {png}")
    png2 = render_fracture_ballistic_figure(OUT_FRACTURE_PNG, overrides=overrides, seed=args.seed)
    print(f"[render] Wrote {png2}")
    if not args.skip_gif:
        gif = render_growth_gif(OUT_GIF, seed=args.seed)
        print(f"[render] Wrote {gif}")
        gif2 = render_fracture_gif(OUT_FRACTURE_GIF, seed=args.seed)
        print(f"[render] Wrote {gif2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
