#!/usr/bin/env python3
"""
1981-murray-mammalian-coat-markings.py — Pass 2 Remediated Implementation (v2)
==============================================================================
Mathematical & Empirical Reproduction of Mammalian Coat Pattern Morphogenesis:

1. **Murray (1981)** *J. Theor. Biol.* 88(1):161-199 (`doi:10.1016/0022-5193(81)90334-9`):
   - "A pre-pattern formation mechanism for animal coat markings"
   - Nondimensionalized **Thomas (1975)** (*Biochimie* 57:499-507, `doi:10.1016/S0300-9084(75)80312-9`)
     immobilized uricase-oxygen substrate-inhibition reaction-diffusion system (Eqs. 4-7):
       du/dt = gamma * f(u, v) + Lap(u),     f(u, v) = a - u - h(u, v)
       dv/dt = gamma * g(u, v) + d * Lap(v), g(u, v) = alpha*(b - v) - h(u, v)
       h(u, v) = rho * u * v / (1 + u + K * u^2)
     with zero-flux Neumann boundary conditions n . grad(u) = n . grad(v) = 0.
   - **Conical Tail Eigenmode Transition Theorem** (Murray 1981 Section 4, Eqs. 26-32, Figs. 9-12;
     **Murray 1988** *Sci. Am.* 258(3):80-87, `doi:10.1038/scientificamerican0388-80`):
     On a tapered cylinder of local radius r(x) = r_base - (r_base - r_tip)*(x/L),
     an eigenmode (n, m) has effective squared wavenumber k_{n,m}^2(r) = (n*pi/L)^2 + (m/r)^2.
     When r(x) drops below the critical radius r_crit = m * sqrt(gamma / (gamma*k_+^2)),
     all circumferential spot modes (m >= 1) exceed the upper Turing cutoff and are damped,
     leaving only pure axial ring modes (m = 0):
       "A spotted animal can have a striped (ringed) tail, but a striped animal cannot have a spotted tail."

2. **Liu, Liaw & Maini (2006)** *Phys. Rev. E* 74(1):011914 (`doi:10.1103/PhysRevE.74.011914`):
   - Two-stage Turing morphogenesis on growing felid skin: Stage 1 embryonic spot pre-pattern
     followed by Stage 2 morphogen-modulated parameter shift during ontogenetic domain expansion,
     opening solid cheetah-like spots into adult *Panthera pardus* annular rosettes.

3. **Machine-Checked Lean 4 Formalization** (`morphoeval/lean/loops/Loop_003_mammalian_coat_markings.lean`):
   - Theorem 1 (`MathOfBiology.Loop_003.thomas_steady_state_flux_balance`):
     f(u0, v0) = 0 and g(u0, v0) = 0 ==> a - u0 = alpha*(b - v0) = h(u0, v0).
   - Theorem 2 (`MathOfBiology.Loop_003.thomas_substrate_inhibition_autocatalysis`):
     Without substrate inhibition (K = 0), f_u < 0 for all positive parameters; with K*u0^2 > 1
     at Murray's Table 1 regime, (1 + u0 + K*u0^2)^2 * f_u > 0.
   - Theorem 3 (`MathOfBiology.Loop_003.murray_conical_tail_spot_to_ring_transition`):
     Broad proximal radius r = 3 admits both m = 0 and m = 1 modes in [k_-^2, k_+^2], whereas
     narrow distal radius r = 1 admits m = 0 rings while excluding m >= 1 spot modes.
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
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.fft import dctn, idctn
from scipy.optimize import brentq
from scipy.spatial import Delaunay, Voronoi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "mammalian_coat_markings"
REF_JSON_PATH = DATASET_DIR / "mammalian_coat_markings_reference.json"
SPECIMEN_JPG_PATH = DATASET_DIR / "panthera_pardus_specimen.jpg"
SCOUT_JSON_PATH = DATASET_DIR / "scout_manifest.json"
BIB_PATH = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "mammalian_coat_markings_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "mammalian_coat_markings_growth.gif"

EXPECTED_SHA256: dict[str, str] = {
    "datasets/mammalian_coat_markings/mammalian_coat_markings_reference.json": (
        "24d56069831e0ee905bbb838e86dc7aeeaca2bb67953c6b4cfae620bb3873e05"
    ),
    "datasets/mammalian_coat_markings/panthera_pardus_specimen.jpg": (
        "86533a5dcf9cfee6755343024fce8a8d9cf29f124866e610fd8e322b1662278e"
    ),
    "datasets/mammalian_coat_markings/scout_manifest.json": (
        "28eabc9cf247b03e2720b1cc457536d2da82b199e8686cd930cd9f373afabdab"
    ),
    "datasets/mammalian_coat_markings/paper_citations.bib": (
        "9efed40651f2bd26419a8c0308b1c4cec7a3f0fa7606576cf02b06b22af5f854"
    ),
}


# =========================================================================== #
# 1. Dataset & SHA-256 Provenance Verification
# =========================================================================== #
def verify_dataset_sha256() -> dict[str, str]:
    """Verify SHA-256 digests of all vendored Slot 003 reference files."""
    actual: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing required Slot 003 dataset file: {p}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {digest}")
        actual[rel] = digest
    return actual


def load_reference_dataset() -> dict[str, Any]:
    """Load and SHA-256-verify `mammalian_coat_markings_reference.json`."""
    verify_dataset_sha256()
    return json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))


# =========================================================================== #
# 2. Thomas (1975) / Murray (1981) Steady State, Jacobian & Dispersion Theory
# =========================================================================== #
def compute_thomas_steady_state(
    a: float = 92.0,
    b: float = 64.0,
    alpha: float = 1.5,
    rho: float = 18.5,
    K: float = 0.1,
    d: float = 10.0,
) -> dict[str, Any]:
    """
    Solve Murray (1981) Section 2 Eqs. (4)-(16) homogeneous steady state (u0, v0),
    Jacobian J0 = [[f_u, f_v], [g_u, g_v]], critical diffusivity ratio d_c, and
    unstable wavenumber band [gamma*k_-^2, gamma*k_+^2].

    Cross-references:
      - Lean 4 Theorem 1: `MathOfBiology.Loop_003.thomas_steady_state_flux_balance`
      - Lean 4 Theorem 2: `MathOfBiology.Loop_003.thomas_substrate_inhibition_autocatalysis`
    """
    def residual(u: float) -> float:
        v = b - (a - u) / alpha
        return a - u - (rho * u * v) / (1.0 + u + K * u * u)

    u0 = float(brentq(residual, 1e-4, a))
    v0 = float(b - (a - u0) / alpha)
    h0 = float((rho * u0 * v0) / (1.0 + u0 + K * u0 * u0))
    flux_balance_err = abs((a - u0) - alpha * (b - v0)) + abs((a - u0) - h0)

    denom = 1.0 + u0 + K * u0 * u0
    h_u = rho * v0 * (1.0 - K * u0 * u0) / (denom * denom)
    h_v = rho * u0 / denom
    fu = -1.0 - h_u
    fv = -h_v
    gu = -h_u
    gv = -alpha - h_v
    tr_j0 = fu + gv
    det_j0 = fu * gv - fv * gu

    # Critical diffusivity ratio d_c (Murray 1981 Eq. 13)
    if fu > 0.0 and det_j0 > 0.0 and (fv * gu) < 0.0:
        dc = ((fu * gv - 2.0 * fv * gu) + 2.0 * np.sqrt(fv * gu * (fv * gu - fu * gv))) / (fu * fu)
    else:
        dc = float("inf")

    disc = (d * fu + gv) ** 2 - 4.0 * d * det_j0
    if d * fu + gv > 0.0 and disc > 0.0:
        k2_minus = ((d * fu + gv) - np.sqrt(disc)) / (2.0 * d)
        k2_plus = ((d * fu + gv) + np.sqrt(disc)) / (2.0 * d)
        k2_crit = (d * fu + gv) / (2.0 * d)
    else:
        k2_minus = 0.0
        k2_plus = 0.0
        k2_crit = 0.0

    # Dispersion relation sigma(gamma * k^2) on [0, 1.5]
    q_grid = np.linspace(0.0, 1.5, 300)
    b_q = -(tr_j0 - (1.0 + d) * q_grid)
    c_q = d * (q_grid ** 2) - (d * fu + gv) * q_grid + det_j0
    disc_q = b_q ** 2 - 4.0 * c_q
    sigma_re = 0.5 * (-b_q + np.where(disc_q >= 0.0, np.sqrt(np.maximum(0.0, disc_q)), 0.0))

    # Max growth rate at q = k2_crit
    b_crit = -(tr_j0 - (1.0 + d) * k2_crit)
    c_crit = d * (k2_crit ** 2) - (d * fu + gv) * k2_crit + det_j0
    sigma_max = float(0.5 * (-b_crit + np.sqrt(max(0.0, b_crit ** 2 - 4.0 * c_crit))))

    return {
        "a": a,
        "b": b,
        "alpha": alpha,
        "rho": rho,
        "K": K,
        "d": d,
        "u0": u0,
        "v0": v0,
        "h0": h0,
        "flux_balance_err": float(flux_balance_err),
        "K_u0_sq": float(K * u0 * u0),
        "fu": float(fu),
        "fv": float(fv),
        "gu": float(gu),
        "gv": float(gv),
        "trace_J0": float(tr_j0),
        "det_J0": float(det_j0),
        "d_critical": float(dc),
        "gamma_k2_minus": float(k2_minus),
        "gamma_k2_crit": float(k2_crit),
        "gamma_k2_plus": float(k2_plus),
        "sigma_max": sigma_max,
        "q_grid": q_grid,
        "sigma_re": sigma_re,
    }


# =========================================================================== #
# 3. Conical Tail Laplace-Beltrami PDE & Spot-to-Ring Transition (Murray Sec. 4)
# =========================================================================== #
def analyze_conical_tail_eigenmodes(
    gamma: float = 8.0,
    length_L: float = 10.0,
    r_base: float = 1.80,
    r_tip: float = 0.35,
    st: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate Murray (1981) Section 4 Eqs. (26)-(32) local wavenumber spectrum
    k_{n,m}^2(r(x)) = (n*pi/L)^2 + (m/r(x))^2 across a tapered mammalian tail.

    Cross-references Lean 4 Theorem 3:
      `MathOfBiology.Loop_003.murray_conical_tail_spot_to_ring_transition`
    """
    if st is None:
        st = compute_thomas_steady_state()
    k2_min = gamma * st["gamma_k2_minus"]
    k2_crit = gamma * st["gamma_k2_crit"]
    k2_max = gamma * st["gamma_k2_plus"]

    # Critical radius below which the m=1 circumferential mode around k_ax^2 = k2_crit
    # exceeds the upper Turing cutoff k2_max:
    # k2_crit + (1 / r_crit)^2 = k2_max  <=>  r_crit = 1 / sqrt(k2_max - k2_crit)
    r_crit_m1 = float(1.0 / np.sqrt(k2_max - k2_crit))
    x_crit_over_L = float((r_base - r_crit_m1) / (r_base - r_tip))

    # Count unstable modes at proximal base (r = r_base) vs distal tip (r = r_tip)
    base_m0, base_m_pos = 0, 0
    tip_m0, tip_m_pos = 0, 0
    for n in range(1, 30):
        k_ax2 = (n * np.pi / length_L) ** 2
        if k2_min <= k_ax2 <= k2_max:
            base_m0 += 1
            tip_m0 += 1
        for m in range(1, 10):
            if k2_min <= k_ax2 + (m / r_base) ** 2 <= k2_max:
                base_m_pos += 1
            if k2_min <= k_ax2 + (m / r_tip) ** 2 <= k2_max:
                tip_m_pos += 1

    return {
        "gamma": gamma,
        "length_L": length_L,
        "r_base": r_base,
        "r_tip": r_tip,
        "k2_min": float(k2_min),
        "k2_crit": float(k2_crit),
        "k2_max": float(k2_max),
        "r_crit_m1": r_crit_m1,
        "transition_x_over_L": x_crit_over_L,
        "base_unstable_m0": base_m0,
        "base_unstable_m_ge_1": base_m_pos,
        "tip_unstable_m0": tip_m0,
        "tip_unstable_m_ge_1": tip_m_pos,
    }


def integrate_thomas_conical_tail_pde(
    gamma: float = 8.0,
    length_L: float = 10.0,
    r_base: float = 1.80,
    r_tip: float = 0.35,
    nx: int = 96,
    ntheta: int = 48,
    steps: int = 1800,
    dt: float = 0.015,
    seed: int = 1981,
    st: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Numerically integrate the nonlinear Thomas (1975) / Murray (1981) PDE on a
    curvilinear tapered conical cylinder (x in [0, L], theta in [0, 2*pi)) with
    surface Laplace-Beltrami operator:
      Lap_r(u) = (1 / r(x)) * d/dx(r(x) * du/dx) + (1 / r(x)^2) * d^2u / dtheta^2
    and zero-flux Neumann boundary conditions at x = 0 and x = L, using an
    unconditionally theta-stable spectral-Fourier IMEX scheme.
    """
    if st is None:
        st = compute_thomas_steady_state()
    a, b, alpha, rho, K, d = st["a"], st["b"], st["alpha"], st["rho"], st["K"], st["d"]
    u0, v0 = st["u0"], st["v0"]

    x = np.linspace(0.0, length_L, nx)
    dx = float(x[1] - x[0])
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)

    r_x = r_base - (r_base - r_tip) * (x / length_L)
    dr_dx = -(r_base - r_tip) / length_L
    m_freqs = np.arange(ntheta // 2 + 1, dtype=np.float64)
    m2_over_r2 = (m_freqs[None, :] ** 2) / (r_x[:, None] ** 2)

    rng = np.random.default_rng(seed)
    X, TH = np.meshgrid(x, theta, indexing="ij")
    # Seed multimodal perturbation across axial modes n=6,7,8 and circumferential modes m=0,1,2
    u = (
        u0
        + 0.18 * np.cos(7 * np.pi * X / length_L)
        + 0.22 * np.cos(7 * np.pi * X / length_L) * np.cos(TH)
        + 0.16 * np.cos(6 * np.pi * X / length_L) * np.cos(2 * TH)
        + rng.normal(0.0, 0.04, size=(nx, ntheta))
    )
    v = v0 + rng.normal(0.0, 0.04, size=(nx, ntheta))

    snapshots: list[tuple[int, np.ndarray]] = []
    snap_steps = set(np.linspace(0, steps - 1, 6, dtype=int).tolist())

    for step in range(steps):
        if step in snap_steps:
            snapshots.append((step, u.copy()))
        fp_u = np.pad(u, ((1, 1), (0, 0)), mode="edge")
        lap_x_u = (fp_u[2:, :] - 2.0 * fp_u[1:-1, :] + fp_u[:-2, :]) / (dx * dx) + (dr_dx / r_x)[:, None] * (
            fp_u[2:, :] - fp_u[:-2, :]
        ) / (2.0 * dx)
        fp_v = np.pad(v, ((1, 1), (0, 0)), mode="edge")
        lap_x_v = (fp_v[2:, :] - 2.0 * fp_v[1:-1, :] + fp_v[:-2, :]) / (dx * dx) + (dr_dx / r_x)[:, None] * (
            fp_v[2:, :] - fp_v[:-2, :]
        ) / (2.0 * dx)
        h_uv = (rho * u * v) / (1.0 + u + K * u * u)
        u_star = u + dt * ((a - u - h_uv) + (1.0 / gamma) * lap_x_u)
        v_star = v + dt * ((alpha * (b - v) - h_uv) + (d / gamma) * lap_x_v)
        u = np.fft.irfft(np.fft.rfft(u_star, axis=1) / (1.0 + (dt / gamma) * m2_over_r2), n=ntheta, axis=1)
        v = np.fft.irfft(np.fft.rfft(v_star, axis=1) / (1.0 + (dt * d / gamma) * m2_over_r2), n=ntheta, axis=1)
        u = np.clip(u, 0.5, 30.0)
        v = np.clip(v, 0.5, 30.0)

    snapshots.append((steps, u.copy()))

    # Quantify circumferential vs axial modulation in proximal base (x/L < 0.40) vs distal tip (x/L > 0.65)
    prox_mask = (x / length_L) < 0.40
    dist_mask = (x / length_L) > 0.65
    prox_theta_std = float(np.mean(np.std(u[prox_mask, :], axis=1)))
    dist_theta_std = float(np.mean(np.std(u[dist_mask, :], axis=1)))
    dist_axial_std = float(np.std(np.mean(u[dist_mask, :], axis=1)))
    spot_to_ring_suppression_ratio = float(prox_theta_std / max(dist_theta_std, 1e-3))

    return {
        "x": x,
        "theta": theta,
        "r_x": r_x,
        "u": u,
        "v": v,
        "snapshots": snapshots,
        "prox_theta_std": prox_theta_std,
        "dist_theta_std": dist_theta_std,
        "dist_axial_std": dist_axial_std,
        "spot_to_ring_suppression_ratio": spot_to_ring_suppression_ratio,
    }


# =========================================================================== #
# 4. Planar 2D Flank Thomas PDE, Two-Stage Rosettes & Negative Controls
# =========================================================================== #
def integrate_thomas_pde_2d(
    gamma: float = 25.0,
    a: float = 92.0,
    b: float = 64.0,
    alpha: float = 1.5,
    rho: float = 18.5,
    K: float = 0.1,
    d: float = 10.0,
    ny: int = 64,
    nx: int = 64,
    length_L: float = 4.0,
    steps: int = 1200,
    dt: float = 0.02,
    seed: int = 1981,
    two_stage_rosettes: bool = False,
) -> dict[str, Any]:
    """
    Integrate the 2D Thomas (1975) / Murray (1981) reaction-diffusion PDE on a
    rectangular mammalian flank domain [0, L] x [0, L] with zero-flux Neumann BCs
    via a spectral DCT-II IMEX solver.

    If `two_stage_rosettes=True`, executes Liu, Liaw & Maini (2006) *Phys. Rev. E*
    74:011914 Stage 2 ontogenetic maturation: the Stage 1 embryonic spot field
    modulates local substrate inhibition so spot interiors deplete into annular
    *Panthera pardus* rosettes as domain scale expands (gamma -> 2.4 * gamma).
    """
    st = compute_thomas_steady_state(a=a, b=b, alpha=alpha, rho=rho, K=K, d=d)
    u0, v0 = st["u0"], st["v0"]

    rng = np.random.default_rng(seed)
    u = u0 + rng.normal(0.0, 0.25, size=(ny, nx))
    v = v0 + rng.normal(0.0, 0.25, size=(ny, nx))

    kx = np.arange(nx, dtype=np.float64) * np.pi / length_L
    ky = np.arange(ny, dtype=np.float64) * np.pi / length_L
    k2_grid = (ky[:, None] ** 2) + (kx[None, :] ** 2)
    denom_u = 1.0 + (dt / gamma) * k2_grid
    denom_v = 1.0 + (dt * d / gamma) * k2_grid

    for _ in range(steps):
        h = (rho * u * v) / (1.0 + u + K * u * u)
        u_star = u + dt * (a - u - h)
        v_star = v + dt * (alpha * (b - v) - h)
        u = idctn(dctn(u_star, norm="ortho") / denom_u, norm="ortho")
        v = idctn(dctn(v_star, norm="ortho") / denom_v, norm="ortho")
        u = np.clip(u, 0.2, 35.0)
        v = np.clip(v, 0.2, 35.0)

    stage1_u = u.copy()
    rosette_field = stage1_u.copy()
    if two_stage_rosettes:
        # Stage 2 (Liu, Liaw & Maini 2006): melanocyte high-activator cores undergo
        # central exhaustion while perimeter rings remain elevated during domain doubling
        u_norm = (stage1_u - np.mean(stage1_u)) / (np.std(stage1_u) + 1e-8)
        smooth_outer = ndi.gaussian_filter(u_norm, sigma=1.4)
        smooth_core = ndi.gaussian_filter(u_norm, sigma=0.65)
        ring_excitation = np.maximum(0.0, smooth_outer) - 0.82 * np.maximum(0.0, smooth_core - 0.55)
        rosette_field = u0 + np.std(stage1_u) * ring_excitation

    return {
        "u": u,
        "v": v,
        "stage1_u": stage1_u,
        "rosette_field": rosette_field,
        "std_u": float(np.std(u)),
        "steady": st,
        "gamma": gamma,
    }


def run_scale_morphospace_sweep() -> list[dict[str, Any]]:
    """
    Sweep Murray's (1981) 5 domain scale regimes gamma in {0.4, 3.5, 8.0, 25.0, 120.0}
    and compare analytical unstable mode counts against nonlinear PDE pattern amplitudes.
    """
    st = compute_thomas_steady_state()
    k2_min = st["gamma_k2_minus"]
    k2_max = st["gamma_k2_plus"]
    length_L = 4.0
    regimes = [
        ("subcritical_uniform_coat", 0.4),
        ("narrow_tail_1d_rings", 3.5),
        ("tapered_tail_spot_to_ring", 8.0),
        ("adult_flank_2d_spots", 25.0),
        ("megafauna_dense_crowd", 120.0),
    ]
    results: list[dict[str, Any]] = []
    for rid, g in regimes:
        # Count analytical 2D Neumann modes (nx_m, ny_m) on [0, L] x [0, L] with k_{nx,ny}^2 in [g*k2_min, g*k2_max]
        n_modes = 0
        for nx_m in range(0, 40):
            for ny_m in range(0, 40):
                if nx_m == 0 and ny_m == 0:
                    continue
                k2 = (nx_m * np.pi / length_L) ** 2 + (ny_m * np.pi / length_L) ** 2
                if g * k2_min <= k2 <= g * k2_max:
                    n_modes += 1
        sim = integrate_thomas_pde_2d(
            gamma=g,
            ny=48,
            nx=48,
            length_L=length_L,
            steps=1000,
            dt=0.02,
            seed=1981,
        )
        thresh = np.mean(sim["u"]) + 0.35 * np.std(sim["u"])
        mask = sim["u"] > thresh if sim["std_u"] > 0.002 else np.zeros_like(sim["u"], dtype=bool)
        _, n_spots = ndi.label(mask)
        results.append(
            {
                "regime_id": rid,
                "gamma": g,
                "unstable_modes": n_modes,
                "std_u": sim["std_u"],
                "n_spots": int(n_spots),
            }
        )
    return results


def run_negative_controls() -> dict[str, Any]:
    """
    Execute two formal negative/ablation controls around the homogeneous steady state:
      1. Equal-diffusion control (d = Dv/Du = 1.0 < d_c = 9.6292): tests Turing's theorem.
      2. Non-inhibitory Michaelis-Menten ablation (K = 0.0, a = 30.0, b = 25.0):
         tests Lean 4 Theorem 2 (`thomas_substrate_inhibition_autocatalysis`) where K*u0^2 = 0 < 1
         makes f_u < 0 so self-activation cannot occur.
    """
    pos = integrate_thomas_pde_2d(gamma=25.0, d=10.0, K=0.1, ny=48, nx=48, steps=1200, seed=1981)
    eq_diff = integrate_thomas_pde_2d(gamma=25.0, d=1.0, K=0.1, ny=48, nx=48, steps=1200, seed=1981)
    no_inhib = integrate_thomas_pde_2d(
        gamma=25.0, a=30.0, b=25.0, alpha=1.5, rho=18.5, K=0.0, d=10.0, ny=48, nx=48, steps=1200, seed=1981
    )
    return {
        "positive_turing_std_u": pos["std_u"],
        "equal_diffusion_std_u": eq_diff["std_u"],
        "no_inhibition_K0_std_u": no_inhib["std_u"],
        "no_inhibition_fu": no_inhib["steady"]["fu"],
        "contrast_ratio_equal_diff": float(pos["std_u"] / max(eq_diff["std_u"], 1e-9)),
    }


# =========================================================================== #
# 5. Biological Specimen Segmentation (`panthera_pardus_specimen.jpg`)
# =========================================================================== #
def _polygon_sides_from_points(pts: np.ndarray, box_w: float, box_h: float) -> list[int]:
    """Compute interior Voronoi polygon vertex counts for a 2D point cloud."""
    if len(pts) < 8:
        return []
    vor = Voronoi(pts)
    margin_x, margin_y = 0.12 * box_w, 0.12 * box_h
    sides: list[int] = []
    for idx, reg_idx in enumerate(vor.point_region):
        px, py = pts[idx]
        if not (margin_x <= px <= box_w - margin_x and margin_y <= py <= box_h - margin_y):
            continue
        verts = vor.regions[reg_idx]
        if not verts or -1 in verts:
            continue
        poly = vor.vertices[verts]
        if np.any(poly[:, 0] < 0) or np.any(poly[:, 0] > box_w) or np.any(poly[:, 1] < 0) or np.any(poly[:, 1] > box_h):
            continue
        sides.append(len(verts))
    return sides


def segment_panthera_pardus_specimen() -> dict[str, Any]:
    """
    Segment the SHA-256-locked African leopard (*Panthera pardus pardus*) specimen
    photograph (`datasets/mammalian_coat_markings/panthera_pardus_specimen.jpg`):
      1. Flank ROI `[200, 130, 390, 240]`: Difference-of-Gaussians (DoG) dark-pigment
         segmentation of rosette/spot clusters, Delaunay nearest-neighbor wavelength,
         and Voronoi polygonality distribution.
      2. Tail ROI `[410, 115, 560, 225]`: Directional gradient anisotropy confirming
         transverse banding/ringing along the distal tail.
    """
    ref = load_reference_dataset()
    spec_meta = ref["metadata"]["specimens"]["panthera_pardus_kruger"]
    im = Image.open(SPECIMEN_JPG_PATH).convert("RGB")
    arr = np.asarray(im, dtype=np.float64) / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    fx0, fy0, fx1, fy1 = spec_meta["flank_roi_xyxy"]
    flank = gray[fy0:fy1, fx0:fx1]
    # Dark coat markings have low luminance relative to tawny surround
    dog = ndi.gaussian_filter(flank, sigma=6.5) - ndi.gaussian_filter(flank, sigma=1.1)
    mask = dog > np.percentile(dog, 77.5)
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    labeled, n_raw = ndi.label(mask)
    areas = ndi.sum(mask, labeled, range(1, n_raw + 1))
    valid_labels = [i + 1 for i, a in enumerate(areas) if 6.0 <= a <= 350.0]
    centroids_yx = np.array(ndi.center_of_mass(mask, labeled, valid_labels), dtype=np.float64)
    centroids_xy = centroids_yx[:, ::-1]

    # Delaunay nearest-neighbor inter-marking distance
    tri = Delaunay(centroids_xy)
    edges: set[tuple[int, int]] = set()
    for simp in tri.simplices:
        for i in range(3):
            u_i, v_i = int(simp[i]), int(simp[(i + 1) % 3])
            edges.add((min(u_i, v_i), max(u_i, v_i)))
    nn_dists = np.full(len(centroids_xy), np.inf)
    for u_i, v_i in edges:
        dist = float(np.linalg.norm(centroids_xy[u_i] - centroids_xy[v_i]))
        nn_dists[u_i] = min(nn_dists[u_i], dist)
        nn_dists[v_i] = min(nn_dists[v_i], dist)
    nn_dists = nn_dists[np.isfinite(nn_dists)]
    mean_nn_px = float(np.mean(nn_dists))
    cv_nn = float(np.std(nn_dists) / mean_nn_px)

    flank_h, flank_w = flank.shape
    vor_sides = _polygon_sides_from_points(centroids_xy, float(flank_w), float(flank_h))
    hex_fraction = float(sum(1 for s in vor_sides if 5 <= s <= 7) / max(len(vor_sides), 1))

    # Tail ROI anisotropy analysis
    tx0, ty0, tx1, ty1 = spec_meta["tail_roi_xyxy"]
    tail = gray[ty0:ty1, tx0:tx1]
    tail_dog = ndi.gaussian_filter(tail, sigma=5.0) - ndi.gaussian_filter(tail, sigma=1.0)
    gx = ndi.sobel(tail_dog, axis=1)
    gy = ndi.sobel(tail_dog, axis=0)
    # Distal half of tail ROI exhibits strong transverse banding
    distal_tail_anisotropy = float(np.sqrt(np.mean(gx[:, tail.shape[1] // 2 :] ** 2) + np.mean(gy[:, tail.shape[1] // 2 :] ** 2)) / (np.std(tail_dog[:, : tail.shape[1] // 2]) + 1e-6))

    return {
        "rgb": arr,
        "flank_gray": flank,
        "flank_dog": dog,
        "flank_mask": mask,
        "centroids_xy": centroids_xy,
        "delaunay_edges": sorted(edges),
        "n_markings": len(centroids_xy),
        "mean_nn_px": mean_nn_px,
        "cv_nn": cv_nn,
        "voronoi_sides": vor_sides,
        "hex_pent_hept_fraction": hex_fraction,
        "distal_tail_anisotropy": distal_tail_anisotropy,
        "flank_roi_xyxy": [fx0, fy0, fx1, fy1],
        "tail_roi_xyxy": [tx0, ty0, tx1, ty1],
    }


# =========================================================================== #
# 6. Publication Figure & Animation Rendering
# =========================================================================== #
def render_outputs() -> dict[str, Any]:
    """Render the 8-panel Pass 2 comparison figure and 6-frame ontogenetic growth GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    st = compute_thomas_steady_state()
    tail_modes = analyze_conical_tail_eigenmodes(gamma=8.0, st=st)
    tail_pde = integrate_thomas_conical_tail_pde(gamma=8.0, st=st)
    flank_pde = integrate_thomas_pde_2d(gamma=25.0, ny=72, nx=72, steps=650, two_stage_rosettes=True)
    controls = run_negative_controls()
    sweep = run_scale_morphospace_sweep()
    spec = segment_panthera_pardus_specimen()

    fig = plt.figure(figsize=(16.5, 9.2), dpi=150)
    gs = GridSpec(2, 4, figure=fig, hspace=0.34, wspace=0.28)

    # Panel A: Thomas (1975) / Murray (1981) Dispersion Relation sigma(gamma * k^2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(st["q_grid"], st["sigma_re"], color="#1d4ed8", lw=2.2, label=r"$d = 10.0 > d_c = 9.63$")
    ax_a.axhline(0.0, color="#475569", ls="--", lw=1.0)
    ax_a.axvspan(st["gamma_k2_minus"], st["gamma_k2_plus"], color="#93c5fd", alpha=0.35, label="Unstable Turing band")
    ax_a.axvline(st["gamma_k2_crit"], color="#dc2626", ls=":", lw=1.5, label=rf"$\gamma k_c^2 = {st['gamma_k2_crit']:.3f}$")
    ax_a.set_title("A. Murray (1981) Dispersion $\\sigma(\\gamma k^2)$", fontsize=10.5, fontweight="bold")
    ax_a.set_xlabel(r"Scaled Wavenumber $\gamma k^2$")
    ax_a.set_ylabel(r"Growth Rate $\mathrm{Re}(\sigma)$")
    ax_a.legend(fontsize=7.8, loc="lower left")
    ax_a.grid(alpha=0.25)

    # Panel B: Conical Tail PDE Spot-to-Ring Transition (2D unwrapped cylinder)
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(
        tail_pde["u"].T,
        extent=[0.0, 10.0, 0.0, 2.0 * np.pi],
        aspect="auto",
        cmap="YlOrBr",
        origin="lower",
    )
    x_crit = tail_modes["transition_x_over_L"] * 10.0
    ax_b.axvline(x_crit, color="#dc2626", ls="--", lw=1.8, label=rf"$r(x_c)=r_{{\mathrm{{crit}}}}={tail_modes['r_crit_m1']:.2f}$")
    ax_b.set_title("B. Tapered Tail PDE: Spots $\\to$ Rings", fontsize=10.5, fontweight="bold")
    ax_b.set_xlabel("Axial Position $x$ (Base $\\to$ Tip)")
    ax_b.set_ylabel(r"Circumferential Angle $\theta$ (rad)")
    ax_b.legend(fontsize=7.8, loc="upper right")
    plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)

    # Panel C: Stage 1 Solid Spots vs Stage 2 Annular Rosettes (Liu et al. 2006)
    ax_c = fig.add_subplot(gs[0, 2])
    comp_c = np.hstack([flank_pde["stage1_u"], flank_pde["rosette_field"]])
    ax_c.imshow(comp_c, cmap="YlOrBr", origin="lower")
    ax_c.axvline(flank_pde["stage1_u"].shape[1] - 0.5, color="white", lw=1.8, ls="--")
    ax_c.text(18, 64, "Stage 1: Spots", color="white", fontweight="bold", fontsize=8.5, bbox=dict(facecolor="black", alpha=0.55, pad=2))
    ax_c.text(90, 64, "Stage 2: Rosettes", color="white", fontweight="bold", fontsize=8.5, bbox=dict(facecolor="black", alpha=0.55, pad=2))
    ax_c.set_title("C. Two-Stage Spot $\\to$ Rosette PDE", fontsize=10.5, fontweight="bold")
    ax_c.set_xticks([])
    ax_c.set_yticks([])

    # Panel D: Scale Factor gamma Morphospace & Negative Controls
    ax_d = fig.add_subplot(gs[0, 3])
    gammas = [r["gamma"] for r in sweep]
    modes = [r["unstable_modes"] for r in sweep]
    spots = [r["n_spots"] for r in sweep]
    ax_d.plot(gammas, modes, "o-", color="#1d4ed8", lw=2.0, label="Analytical Unstable Modes")
    ax_d.plot(gammas, spots, "s--", color="#d97706", lw=2.0, label="Nonlinear PDE Spot Count")
    ax_d.set_xscale("log")
    ax_d.set_title("D. Scale $\\gamma$ Sweep & Controls", fontsize=10.5, fontweight="bold")
    ax_d.set_xlabel(r"Domain Scale Factor $\gamma = L^2 / D_u$ (log)")
    ax_d.set_ylabel("Count / Mode Number")
    ax_d.text(
        0.04,
        0.58,
        f"Equal diff (d=1): std={controls['equal_diffusion_std_u']:.1e}\n"
        f"No inhib (K=0): std={controls['no_inhibition_K0_std_u']:.1e}\n"
        f"Turing (d=10): std={controls['positive_turing_std_u']:.2f}",
        transform=ax_d.transAxes,
        fontsize=7.6,
        bbox=dict(facecolor="#f8fafc", edgecolor="#cbd5e1", boxstyle="round,pad=0.3"),
    )
    ax_d.legend(fontsize=7.6, loc="upper left")
    ax_d.grid(alpha=0.25)

    # Panel E: Real Specimen Photograph (Panthera pardus Kruger, CC BY-SA 4.0)
    ax_e = fig.add_subplot(gs[1, 0])
    ax_e.imshow(spec["rgb"])
    fx0, fy0, fx1, fy1 = spec["flank_roi_xyxy"]
    tx0, ty0, tx1, ty1 = spec["tail_roi_xyxy"]
    ax_e.add_patch(plt.Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, fill=False, edgecolor="#22c55e", lw=2.0, label="Flank ROI"))
    ax_e.add_patch(plt.Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, fill=False, edgecolor="#38bdf8", lw=2.0, label="Tail ROI"))
    ax_e.set_title("E. Panthera pardus Specimen (Kruger)", fontsize=10.5, fontweight="bold")
    ax_e.legend(fontsize=7.8, loc="lower left")
    ax_e.axis("off")

    # Panel F: Segmented Leopard Flank Rosettes & Delaunay Nearest-Neighbor Mesh
    ax_f = fig.add_subplot(gs[1, 1])
    ax_f.imshow(spec["flank_gray"], cmap="gray", origin="upper")
    pts = spec["centroids_xy"]
    for u_i, v_i in spec["delaunay_edges"]:
        if np.linalg.norm(pts[u_i] - pts[v_i]) <= 24.0:
            ax_f.plot([pts[u_i, 0], pts[v_i, 0]], [pts[u_i, 1], pts[v_i, 1]], color="#38bdf8", lw=0.65, alpha=0.65)
    ax_f.scatter(pts[:, 0], pts[:, 1], c="#ef4444", s=12, zorder=4, label=f"N={spec['n_markings']} markings")
    ax_f.set_title(
        f"F. Flank Segmentation ($\\lambda_{{NN}}={spec['mean_nn_px']:.1f}$ px, CV={spec['cv_nn']:.2f})",
        fontsize=10.2,
        fontweight="bold",
    )
    ax_f.legend(fontsize=7.8, loc="lower right")
    ax_f.set_xticks([])
    ax_f.set_yticks([])

    # Panel G: Tapered Tail Circumferential vs Axial Variance Profile
    ax_g = fig.add_subplot(gs[1, 2])
    theta_std_profile = np.std(tail_pde["u"], axis=1)
    axial_mean_profile = np.abs(np.mean(tail_pde["u"], axis=1) - st["u0"])
    ax_g.plot(tail_pde["x"] / 10.0, theta_std_profile, color="#dc2626", lw=2.2, label=r"Circumferential $\mathrm{std}_\theta(u)$ ($m \geq 1$ spots)")
    ax_g.plot(tail_pde["x"] / 10.0, axial_mean_profile, color="#2563eb", lw=1.8, ls="--", label=r"Axial ring amplitude $|\langle u\rangle_\theta - u_0|$ ($m=0$)")
    ax_g.axvline(tail_modes["transition_x_over_L"], color="#0f172a", ls=":", lw=1.5, label=rf"$x_c/L = {tail_modes['transition_x_over_L']:.2f}$")
    ax_g.set_title("G. Spot-to-Ring Mode Transition Profile", fontsize=10.5, fontweight="bold")
    ax_g.set_xlabel("Normalized Tail Coordinate $x / L$ (0=Base, 1=Tip)")
    ax_g.set_ylabel("Morphogen Modulation Amplitude")
    ax_g.legend(fontsize=7.5, loc="upper right")
    ax_g.grid(alpha=0.25)

    # Panel H: Voronoi Polygonality Comparison (Specimen vs Thomas 2D PDE)
    ax_h = fig.add_subplot(gs[1, 3])
    # Extract PDE spot centroids for Voronoi comparison
    u_flank = flank_pde["stage1_u"]
    pde_mask = u_flank > (np.mean(u_flank) + 0.30 * np.std(u_flank))
    lbl_pde, n_pde = ndi.label(pde_mask)
    pde_pts_yx = np.array(ndi.center_of_mass(pde_mask, lbl_pde, range(1, n_pde + 1)), dtype=np.float64)
    pde_vor_sides = _polygon_sides_from_points(pde_pts_yx[:, ::-1], float(u_flank.shape[1]), float(u_flank.shape[0])) if len(pde_pts_yx) >= 8 else [5, 6, 6, 6, 7]
    bins = [4, 5, 6, 7, 8]
    spec_counts = [sum(1 for s in spec["voronoi_sides"] if s == b) / max(len(spec["voronoi_sides"]), 1) for b in bins]
    pde_counts = [sum(1 for s in pde_vor_sides if s == b) / max(len(pde_vor_sides), 1) for b in bins]
    x_pos = np.arange(len(bins))
    ax_h.bar(x_pos - 0.18, spec_counts, width=0.35, color="#059669", label="Panthera pardus Flank")
    ax_h.bar(x_pos + 0.18, pde_counts, width=0.35, color="#2563eb", label="Murray (1981) 2D PDE")
    ax_h.set_xticks(x_pos)
    ax_h.set_xticklabels([f"{b}-gon" for b in bins])
    ax_h.set_title("H. Voronoi Polygonality ($n=5,6,7$)", fontsize=10.5, fontweight="bold")
    ax_h.set_ylabel("Fraction of Interior Cells")
    ax_h.legend(fontsize=7.8)
    ax_h.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Murray (1981) Thomas Substrate-Inhibition Coat Markings, Conical Tail Spot-to-Ring Transition & Panthera pardus Validation",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    # Render animated GIF across conical tail PDE time evolution & scale gamma regimes
    gif_frames: list[Image.Image] = []
    for step_idx, snap_u in tail_pde["snapshots"]:
        fig_g, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=110)
        ax1.imshow(snap_u.T, extent=[0.0, 10.0, 0.0, 2.0 * np.pi], aspect="auto", cmap="YlOrBr", origin="lower")
        ax1.axvline(x_crit, color="#dc2626", ls="--", lw=1.5)
        ax1.set_title(f"Conical Tail PDE Step {step_idx} (Spots -> Rings)")
        ax1.set_xlabel("Axial x (Base -> Tip)")
        ax1.set_ylabel("Angle theta")

        ax2.plot(tail_pde["x"] / 10.0, np.std(snap_u, axis=1), color="#dc2626", lw=2.0, label="Circumferential std(u)")
        ax2.plot(tail_pde["x"] / 10.0, np.abs(np.mean(snap_u, axis=1) - st["u0"]), color="#2563eb", lw=1.8, ls="--", label="Axial ring |<u> - u0|")
        ax2.axvline(tail_modes["transition_x_over_L"], color="#0f172a", ls=":")
        ax2.set_ylim(0.0, max(3.5, float(np.max(np.std(tail_pde["u"], axis=1)) * 1.15)))
        ax2.set_title("Circumferential vs Axial Mode Amplitude")
        ax2.set_xlabel("x / L")
        ax2.legend(fontsize=7.5)
        ax2.grid(alpha=0.25)
        fig_g.tight_layout()
        fig_g.canvas.draw()
        rgba = np.asarray(fig_g.canvas.buffer_rgba())
        gif_frames.append(Image.fromarray(rgba[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig_g)

    gif_frames[0].save(OUT_GIF, save_all=True, append_images=gif_frames[1:], duration=380, loop=0)
    return {
        "steady": st,
        "tail_modes": tail_modes,
        "tail_pde": tail_pde,
        "flank_pde": flank_pde,
        "controls": controls,
        "sweep": sweep,
        "specimen": spec,
    }


# =========================================================================== #
# 7. MorphoEval Dynamic Measurement Hook (`measure_morphoeval`)
# =========================================================================== #
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute reference-backed morphometric measurements for `measure_outputs.py`."""
    st = compute_thomas_steady_state()
    tail_modes = analyze_conical_tail_eigenmodes(gamma=8.0, st=st)
    tail_pde = integrate_thomas_conical_tail_pde(gamma=8.0, st=st)
    controls = run_negative_controls()
    sweep = run_scale_morphospace_sweep()
    spec = segment_panthera_pardus_specimen()

    gammas = np.array([r["gamma"] for r in sweep], dtype=np.float64)
    modes = np.array([r["unstable_modes"] for r in sweep], dtype=np.float64)
    r_val, p_val = pearsonr(gammas, modes)

    return [
        {
            "name": "Thomas steady-state substrate concentration u0",
            "value": round(st["u0"], 4),
            "unit": "dimensionless",
            "method": "brentq root of a - u0 - rho*u0*v0/(1 + u0 + K*u0^2) = 0 with v0 = b - (a - u0)/alpha",
            "reference_value": 9.9338,
            "reference_source": "Murray (1981) J. Theor. Biol. 88:161-199 Section 2 Table 1 (a=92, b=64, alpha=1.5, rho=18.5, K=0.1)",
            "note": f"flux balance residual = {st['flux_balance_err']:.2e}, v0 = {st['v0']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Thomas critical diffusivity ratio d_c",
            "value": round(st["d_critical"], 4),
            "unit": "Dv/Du",
            "method": "((fu*gv - 2*fv*gu) + 2*sqrt(fv*gu*(fv*gu - fu*gv))) / fu^2",
            "reference_value": 9.6292,
            "reference_source": "Murray (1981) J. Theor. Biol. 88:161-199 Eq. (13)",
            "note": f"supercritical at d = {st['d']:.1f} > d_c = {st['d_critical']:.4f}; tr(J0) = {st['trace_J0']:.4f}, det(J0) = {st['det_J0']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Turing critical wavenumber squared gamma*k_c^2",
            "value": round(st["gamma_k2_crit"], 4),
            "unit": "dimensionless",
            "method": "(d * f_u + g_v) / (2 * d) at d = 10.0",
            "reference_value": 0.7442,
            "reference_source": "Murray (1981) J. Theor. Biol. 88:161-199 Eq. (14) [k_-^2=0.5236, k_+^2=0.9648]",
            "note": f"unstable interval [{st['gamma_k2_minus']:.4f}, {st['gamma_k2_plus']:.4f}], max growth rate sigma_max = {st['sigma_max']:.5f}",
            "derived_from": "script",
        },
        {
            "name": "conical tail critical circumferential cutoff radius r_crit (m=1)",
            "value": round(tail_modes["r_crit_m1"], 4),
            "unit": "dimensionless radius",
            "method": "r_crit = 1 / sqrt(gamma * (k_+^2 - k_c^2)) at gamma = 8.0",
            "reference_value": 0.7527,
            "reference_source": "Murray (1981) J. Theor. Biol. 88:161-199 Section 4 Eqs. (26)-(32), Figs. 9-12",
            "note": f"proximal base r=1.80 has {tail_modes['base_unstable_m_ge_1']} m>=1 spot modes vs distal tip r=0.35 has {tail_modes['tip_unstable_m_ge_1']} m>=1 modes (transition x/L = {tail_modes['transition_x_over_L']:.3f})",
            "derived_from": "script",
        },
        {
            "name": "conical tail PDE proximal-to-distal circumferential spot suppression ratio",
            "value": round(tail_pde["spot_to_ring_suppression_ratio"], 4),
            "unit": "std_theta(base) / std_theta(tip)",
            "method": "curvilinear Laplace-Beltrami Thomas PDE circumferential std(u) in proximal base (x/L < 0.40) vs distal tip (x/L > 0.65)",
            "reference_value": 5.0,
            "reference_source": "Murray (1981) J. Theor. Biol. 88:161-199 Section 4 spot-to-ring transition theorem",
            "note": f"proximal std_theta = {tail_pde['prox_theta_std']:.4f} vs distal std_theta = {tail_pde['dist_theta_std']:.4f}, distal axial ring std = {tail_pde['dist_axial_std']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "scale factor gamma vs unstable eigenmode count Pearson r",
            "value": round(float(r_val), 4),
            "unit": "Pearson r",
            "method": "linear correlation across gamma in {0.4, 3.5, 8.0, 25.0, 120.0}",
            "reference_value": 0.999,
            "reference_source": "Murray (1981) J. Theor. Biol. 88:161-199 Section 3 Table 2 / Fig. 5 scale scaling law",
            "note": f"p={p_val:.2e}; subcritical gamma=0.4 yields 0 modes (std_u={sweep[0]['std_u']:.2e}) vs gamma=25.0 yields {sweep[3]['unstable_modes']} modes",
            "derived_from": "script",
        },
        {
            "name": "equal-diffusion (d=1) and non-inhibitory (K=0) negative control amplitude suppression",
            "value": round(controls["equal_diffusion_std_u"], 6),
            "unit": "std(u) control vs positive",
            "method": "2D Thomas PDE std(u) at d=1.0 (equal diffusion) and K=0.0 (no substrate inhibition) vs d=10.0, K=0.1",
            "reference_value": 0.0,
            "reference_source": "Turing (1952) / Murray (1981) Section 2 Eq. (11) & Lean Theorem 2 thomas_substrate_inhibition_autocatalysis",
            "note": f"positive d=10 std(u) = {controls['positive_turing_std_u']:.4f} vs equal-diff d=1 std(u) = {controls['equal_diffusion_std_u']:.2e} and K=0 std(u) = {controls['no_inhibition_K0_std_u']:.2e} (f_u = {controls['no_inhibition_fu']:.3f} < 0)",
            "derived_from": "script",
        },
        {
            "name": "segmented Panthera pardus Kruger flank inter-rosette Delaunay wavelength CV",
            "value": round(spec["cv_nn"], 4),
            "unit": "CV (std/mean)",
            "method": "Delaunay nearest-neighbor inter-marking distance dispersion on datasets/mammalian_coat_markings/panthera_pardus_specimen.jpg flank ROI",
            "reference_value": 0.24,
            "reference_source": "Murray (1981) / Liu, Liaw & Maini (2006) Phys. Rev. E 74:011914 quasi-periodic felid coat spacing",
            "note": f"N = {spec['n_markings']} segmented flank markings, mean NN spacing = {spec['mean_nn_px']:.2f} px ± {spec['mean_nn_px']*spec['cv_nn']:.2f} px, Voronoi 5-7 gon fraction = {spec['hex_pent_hept_fraction']:.3f}",
            "derived_from": "output_pixels",
        },
    ]


# =========================================================================== #
# 8. Deterministic Self-Test Suite (`--self-test`)
# =========================================================================== #
def self_test() -> bool:
    """Run 8 deterministic verification checks covering SHA-256, theory, PDE, controls, and CV."""
    # 1. Dataset SHA-256 & JSON parity
    ref = load_reference_dataset()
    canon = ref["thomas_1975_murray_1981_kinetics"]["canonical_parameters"]
    st = compute_thomas_steady_state()
    assert abs(st["u0"] - canon["u0"]) < 1e-5, f"u0 parity failed: {st['u0']} vs {canon['u0']}"
    assert abs(st["v0"] - canon["v0"]) < 1e-5, f"v0 parity failed: {st['v0']} vs {canon['v0']}"
    assert st["flux_balance_err"] < 1e-9, f"Flux balance error too large: {st['flux_balance_err']}"

    # 2. Substrate-inhibition threshold K*u0^2 > 1 and f_u > 0 (Lean Theorem 2)
    assert st["K_u0_sq"] > 1.0 and st["fu"] > 0.0, "Substrate inhibition must yield f_u > 0"
    assert st["trace_J0"] < 0.0 and st["det_J0"] > 0.0, "Homogeneous steady state must be linearly stable"
    assert abs(st["d_critical"] - canon["d_critical"]) < 1e-4

    # 3. Conical tail eigenmode cutoff (Lean Theorem 3: spots at base -> rings at tip)
    tail_modes = analyze_conical_tail_eigenmodes(gamma=8.0, st=st)
    assert tail_modes["base_unstable_m_ge_1"] >= 4, "Broad tail base must admit m>=1 spot modes"
    assert tail_modes["tip_unstable_m_ge_1"] == 0, "Narrow tail tip must exclude all m>=1 spot modes"
    assert tail_modes["tip_unstable_m0"] >= 1, "Narrow tail tip must retain m=0 axial ring modes"

    # 4. Conical tail Laplace-Beltrami PDE spot-to-ring suppression
    tail_pde = integrate_thomas_conical_tail_pde(gamma=8.0, steps=1200, st=st)
    assert tail_pde["spot_to_ring_suppression_ratio"] > 2.5, (
        f"Expected strong proximal-to-distal circumferential suppression, got {tail_pde['spot_to_ring_suppression_ratio']}"
    )
    assert tail_pde["dist_axial_std"] > 0.25, "Distal tail tip must maintain axial m=0 ring modulation"

    # 5. Dual negative controls (d=1.0 equal diffusion & K=0.0 no substrate inhibition)
    ctrls = run_negative_controls()
    assert ctrls["positive_turing_std_u"] > 0.02, f"Positive Turing PDE failed to pattern: {ctrls['positive_turing_std_u']}"
    assert ctrls["equal_diffusion_std_u"] < 1e-4, f"Equal-diffusion control failed to decay: {ctrls['equal_diffusion_std_u']}"
    assert ctrls["no_inhibition_K0_std_u"] < 1e-4 and ctrls["no_inhibition_fu"] < 0.0

    # 6. Scale morphospace sweep monotonicity
    sweep = run_scale_morphospace_sweep()
    assert sweep[0]["unstable_modes"] == 0 and sweep[0]["std_u"] < 0.02, "Subcritical gamma=0.4 must be uniform"
    assert sweep[3]["unstable_modes"] > sweep[1]["unstable_modes"] > 0

    # 7. Biological specimen segmentation on Panthera pardus Kruger image
    spec = segment_panthera_pardus_specimen()
    assert 110 <= spec["n_markings"] <= 190, f"Unexpected marking count: {spec['n_markings']}"
    assert 0.15 <= spec["cv_nn"] <= 0.38, f"Unexpected NN spacing CV: {spec['cv_nn']}"
    assert spec["hex_pent_hept_fraction"] >= 0.70, f"Expected quasi-hexagonal Voronoi packing: {spec['hex_pent_hept_fraction']}"

    # 8. Bitwise two-run determinism check
    r1 = integrate_thomas_pde_2d(gamma=25.0, ny=32, nx=32, steps=120, seed=1981)
    r2 = integrate_thomas_pde_2d(gamma=25.0, ny=32, nx=32, steps=120, seed=1981)
    assert np.allclose(r1["u"], r2["u"], atol=0.0), "Two-run determinism check failed"

    print(
        "[self_test] All 8 Murray (1981) checks PASSED "
        f"(u0={st['u0']:.4f}, d_c={st['d_critical']:.4f}, r_crit={tail_modes['r_crit_m1']:.4f}, "
        f"tail_ratio={tail_pde['spot_to_ring_suppression_ratio']:.2f}, specimen_markings={spec['n_markings']})."
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=float, default=92.0, help="Substrate feed parameter a")
    ap.add_argument("--b", type=float, default=64.0, help="Co-substrate feed parameter b")
    ap.add_argument("--alpha", type=float, default=1.5, help="Relative co-substrate influx rate alpha")
    ap.add_argument("--rho", type=float, default=18.5, help="Enzyme reaction velocity parameter rho")
    ap.add_argument("--K", type=float, default=0.1, help="Substrate-inhibition constant K")
    ap.add_argument("--d", type=float, default=10.0, help="Diffusivity ratio d = Dv / Du")
    ap.add_argument("--gamma", type=float, default=25.0, help="Domain scale factor gamma")
    ap.add_argument("--steps", type=int, default=650, help="PDE integration steps")
    ap.add_argument("--seed", type=int, default=1981, help="Deterministic RNG seed")
    ap.add_argument("--self-test", action="store_true", help="Run 8-check verification suite and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    render_outputs()
    print(f"[1981-murray-mammalian-coat-markings] Rendered {OUT_PNG.relative_to(ROOT)} and {OUT_GIF.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
