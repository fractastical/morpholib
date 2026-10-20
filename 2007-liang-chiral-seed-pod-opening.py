#!/usr/bin/env python3
"""
2007-liang-chiral-seed-pod-opening.py — Pass 2 (v2) Remediated Implementation
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
        "b3a693ac6a538b04f0ed0d7953907b2b48ab303a2b97341bb2884b8749c4fecf"
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
    """Verify SHA-256 digests of all 4 files in datasets/chiral_seed_pod_opening/."""
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
    where Lambda(w_tilde) = ((1 + nu) / ((1 - nu)^3 * kappa_0^2)) * (w_tilde / w_tilde_c)^4.
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
# 6. 6-Panel Publication Synthesis Figure & Animated Desiccation GIF
# =============================================================================
def render_comparison_figure(
    out_png: Path = OUT_PNG,
    overrides: dict[str, float] | None = None,
    seed: int = 2011,
) -> Path:
    """Render the 6-panel dark-theme publication synthesis figure comparing:
      (A) 3D Non-Euclidean Ribbon Morphologies: Narrow Pure-Twist Helicoid (w_tilde=1.36)
          vs Opposing Wide Bauhinia Cylindrical Helical Valves (w_tilde=3.78, +H & -H)
      (B) Supercritical Pitchfork Bifurcation Diagram H*(w_tilde)/kappa_0 & D*(w_tilde)/kappa_0
      (C) Helix Radius R*kappa_0, Pitch P*kappa_0/(2*pi), and Gaussian |K|/kappa_0^2 vs w_tilde
      (D) Cutting-Angle Chirality Sweep theta in [0 deg, 90 deg] (including theta=0 deg ring control)
      (E) Computer-Vision Segmentation of Bauhinia variegata Specimen (MHNT.BOT.2011.3.22)
      (F) 4-System Radius/Pitch Benchmark & Sclerenchyma Bilayer Helical Lobe Oscillation"""
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

    fig = plt.figure(figsize=(18.5, 11.2), dpi=150, facecolor="#0b0f17")
    fig.suptitle(
        "Armon, Efrati, Kupferman & Sharon (2011) Science — Non-Euclidean Incompatible Shell Elasticity & Chiral Seed Pod Opening in Bauhinia variegata",
        color="#f0f6fc",
        fontsize=13.5,
        fontweight="bold",
        y=0.978,
    )

    # Panel A: 3D Ribbon Morphologies (Pure-Twist Helicoid vs Opposing Cylindrical Helical Valves)
    ax_a = fig.add_subplot(2, 3, 1, projection="3d", facecolor="#0b0f17")
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
    ax_b = fig.add_subplot(2, 3, 2, facecolor="#111827")
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
    ax_c = fig.add_subplot(2, 3, 3, facecolor="#111827")
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
    ax_d = fig.add_subplot(2, 3, 4, facecolor="#111827")
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
    ax_e = fig.add_subplot(2, 3, 5, facecolor="#111827")
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
    ax_f = fig.add_subplot(2, 3, 6, facecolor="#111827")
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

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
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


# =============================================================================
# 7. Quantitative MorphoEval Measurements & 8-Point Self-Test Suite
# =============================================================================
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute 9 reference-anchored quantitative morphometric measurements for
    `morphoeval/measure_outputs.py`."""
    bif = evaluate_width_pitchfork_bifurcation(seed=2011)
    chir = evaluate_cutting_angle_chirality_sweep(seed=2011)
    spec_table = evaluate_pod_and_mimetic_specimens(seed=2011)
    cv_seg = segment_bauhinia_pod_specimen()
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
    ]


def self_test() -> bool:
    """Run comprehensive 8-point verification suite for Pass 2."""
    # 1. Cryptographic SHA-256 verification of all 4 dataset files
    digests = verify_dataset_integrity()
    assert len(digests) == 4, "Expected 4 SHA-256-verified dataset files"

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

    meas = measure_morphoeval()
    assert len(meas) == 9
    print(
        f"[self_test] 2007-liang-chiral-seed-pod-opening.py (v2) PASSED all 8 checks "
        f"(w_tilde_c={bif['w_tilde_c']:.4f}, pitchfork r_H={bif['pearson_r_H']:.4f}, "
        f"chirality r_tau={chir['pearson_r_tau']:.4f}, 4-system r_P={sp['pearson_r_pitch']:.4f}, "
        f"specimen lobes={cv_seg['n_helical_lobes']}, CV={cv_seg['cv_half_pitch']:.4f})."
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run 8-check verification suite")
    ap.add_argument("--skip-gif", action="store_true", help="Skip animated GIF rendering")
    ap.add_argument("--kappa0", type=float, default=None, help="Override intrinsic shrinkage curvature kappa_0 (cm^-1)")
    ap.add_argument("--nu", type=float, default=None, help="Override Poisson ratio nu")
    ap.add_argument("--W", type=float, default=None, help="Override pod valve width W (cm)")
    ap.add_argument("--h", type=float, default=None, help="Override pod valve thickness h (cm)")
    ap.add_argument("--L", type=float, default=None, help="Override pod valve length L (cm)")
    ap.add_argument("--seed", type=int, default=2011, help="Deterministic RNG seed (default: 2011)")
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
        }.items()
        if v is not None
    }
    png = render_comparison_figure(OUT_PNG, overrides=overrides, seed=args.seed)
    print(f"[render] Wrote {png}")
    if not args.skip_gif:
        gif = render_growth_gif(OUT_GIF, seed=args.seed)
        print(f"[render] Wrote {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
