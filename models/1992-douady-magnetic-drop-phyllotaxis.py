#!/usr/bin/env python3
r"""
1992-douady-magnetic-drop-phyllotaxis.py — Self-Organized Magnetic-Drop Phyllotaxis & Imperfect Pitchfork Cascade
=================================================================================================================

Reproduces and quantitatively validates the experimental and theoretical model of
self-organized botanical phyllotaxis by Stéphane Douady and Yves Couder:

1. **Douady & Couder (1992) Ferrofluid Magnetic-Drop Experiment & Iterative Dipole Model**
   (*Physical Review Letters* 68(13): 2098–2101, DOI: 10.1103/PhysRevLett.68.2098;
    *Journal of Theoretical Biology* 178(3): 255–294, 1996, Parts I & II,
    DOI: 10.1006/jtbi.1996.0024, 10.1006/jtbi.1996.0025; Levitov 1991 *Europhys. Lett.*
    14(6): 533–539, DOI: 10.1209/0295-5075/14/6/006).
   - Polarized ferrofluid droplets of diameter $d \approx 2\text{ mm}$ are injected with period $T$
     onto the periphery $r = R_0$ of a central truncated cone in a horizontal dish of silicone oil
     under a vertical magnetic field gradient that advects drops radially at velocity $V$.
   - The dimensionless **plastochron ratio** is $G = V T / R_0$:
       * Linear silicone-oil dish advection (PRL 1992 Eq. 1):
             r_k / R_0 = 1 + k G
       * Exponential shoot apical meristem (SAM) advection (PRL 1992 & JTB 1996 Eq. 9):
             r_k / R_0 = \exp(k G)
   - Each older drop $k \ge 1$ at $(r_k, \phi_{n-k})$ acts as a vertical magnetic dipole exerting
     a repulsive interaction potential proportional to $d_k^{-\gamma}$ ($\gamma = 3$ for dipole-dipole
     repulsion) on a candidate drop at $(R_0, \phi)$ on the generative circle:
         d_k^2(\phi, \phi_{n-k}) = 1 + (r_k/R_0)^2 - 2 (r_k/R_0) \cos(\phi - \phi_{n-k})
         E(\phi) = \sum_{k=1}^{N_{\text{mem}}} \left[ d_k^2(\phi, \phi_{n-k}) \right]^{-\gamma / 2}
   - Each new primordium / drop forms at the global energy minimum on the rim:
         \phi_n = \arg\min_{\phi \in [0, 2\pi)} E(\phi)

2. **Analytical Opposition-to-Spiral Pitchfork Bifurcation & Lean 4 Theorem Parity**
   - In the alternating distichous $(1,1)$ state ($\Delta\phi = \pi$), the static 2-drop curvature
     at $\phi = 0$ is:
         \left.\frac{\partial^2 E_2}{\partial \phi^2}\right|_{\phi=0}
           = \frac{3 r_1}{(r_1 + 1)^5} - \frac{3 r_2}{(r_2 - 1)^5}
     whose cross-multiplied integer discriminant $\mathcal{C}(r_1, r_2) = 3 r_1 (r_2 - 1)^5 - 3 r_2 (r_1 + 1)^5$
     is machine-checked in `morphoeval/lean/loops/Loop_031_magnetic_drop_phyllotaxis.lean`
     ($\mathcal{C}(4, 7) = +27687 > 0$ at $G=3$ vs $\mathcal{C}(3, 5) = -6144 < 0$ at $G=2$, with root
     $G_{\text{fold}}^{(2,\text{lin})} = 2.51456033$ and $G_{\text{fold}}^{(2,\text{exp})} = 0.77348399$).
   - Including the dynamical shift $\phi_n - \phi_{n-2} = 2\pi - 2\epsilon$ along a steady spiral
     $\Delta\phi = \pi - \epsilon$, the linear stability multiplier $\lambda = \epsilon_n / \epsilon_{n-1}$
     crosses $+1$ at the supercritical pitchfork threshold:
         \sum_{k=1}^{M} (-1)^{k-1} k \frac{r_k}{(r_k - (-1)^k)^5} = 0
     giving $G_{c,1}^{(2,\text{exp})} = 0.86053703$ ($G_{c,1}^{(\infty,\text{exp})} = 0.85901889$)
     and $G_{c,1}^{(2,\text{lin})} = 3.63143185$ ($G_{c,1}^{(\infty,\text{lin})} = 3.61755512$).

3. **Imperfect Pitchfork Cascade: Fibonacci $(1,2,3,5,8,13)$ vs. Disconnected Lucas $(1,3,4,7,11)$**
   - Solves the exact regular-spiral zero-force equilibrium equation $F(G, \Delta\phi) = \left.\frac{\partial E}{\partial \phi}\right|_{\phi=0} = 0$
     and Hessian curvature $S(G, \Delta\phi) = \left.\frac{\partial^2 E}{\partial \phi^2}\right|_{\phi=0}$ across $(G, \Delta\phi)$,
     tracing both the continuous main Fibonacci cascade converging to the Golden Angle
     $\alpha_{\text{Fib}} = 360^\circ / \tau^2 \approx 137.507764^\circ$ and the disconnected saddle-node
     Lucas branch converging to $\alpha_{\text{Lucas}} = 360^\circ / (3 + 1/\tau) \approx 99.501553^\circ$.

4. **Biological Specimen Segmentation & Log-Polar 2D FFT Validation (`conifer_cone_specimen.jpg`)**
   - Detects DoG (Difference-of-Gaussians) blob responses on an apical photograph of a *Pinus*
     conifer seed cone (`File:Top of a pine cone.jpg`, CC BY-SA 4.0), unwraps the cone into
     log-polar coordinates $(u = \ln r, \theta)$ around apex $(316, 307)$, and extracts the
     opposing 2D Fourier spiral modes $(m_{\text{CW}}, m_{\text{CCW}}) = (5, 8)$ matching the
     Douady & Couder $(5, 8)$ pine-cone regime at $G = 0.06$.

5. **Pass 3 Remediation (v3): Emergent Bifurcation, Lattice Inversion & Honest Dish Geometry**
   - *Emergent pitchfork:* the iterated placement map is run through the $(1,1) \to (1,2)$
     transition with **no imposed 180° clamp**; a bisection on the steady divergence locates the
     emergent threshold $G_{c,1}^{\text{emp}}$ and compares it with the analytical full-chain root
     $G_{c,1}^{(\infty,\text{exp})} = 0.859019$ (v2 hard-coded this constant at three places).
   - *Dynamical discriminant (Lean 4 Theorem 4):* for two drops under linear advection the
     dynamical pitchfork condition cross-multiplies to the integer discriminant
     $\mathcal{D}(r_1, r_2) = r_1 (r_2 - 1)^5 - 2 r_2 (r_1 + 1)^5$, with $\mathcal{D}(4,7) = -12646 < 0$
     while $\mathcal{C}(4,7) = +27687 > 0$: at $G = 3$ opposition is a static energy minimum but is
     dynamically unstable, proving $G_{\text{fold}} < 3 < G_{c,1}$ (static/dynamic gap).
   - *Reciprocal-lattice inversion of the specimen:* the sub-bin-refined log-polar FFT peaks
     $(m, \omega_m)$ of the two dominant parastichy families obey $m\,\alpha + \omega_m G = 2\pi j_m$
     for a regular spiral lattice with divergence $\alpha$ and plastochron ratio $G$; solving the
     $2 \times 2$ system for $(5, 8)$ yields the specimen's own divergence angle and $G$, validated
     on a synthetic log-polar image of the simulated $G = 0.06$ lattice. Lean 4 Theorem 5 proves that
     a $(5, 8)$ lattice confines $\alpha$ to the Fibonacci convergent window
     $360^\circ \cdot 3/8 = 135^\circ < \alpha < 360^\circ \cdot 5/13 = 138.46^\circ$.
   - *Honest dish geometry:* v2 displayed and triangulated the exponential lattice with a clipped
     visual growth rate `g_vis = clip(G, 0.012, 0.12)`, which mislabelled the Delaunay pair at
     $G = 0.30$ as $(3,5)$ and at $G = 0.5$ as $(5,8)$. v3 triangulates the true radii
     $r_k = R_0 e^{kG}$ and displays the lattice in the log-radial projection
     $\rho_k = 1 + \ln(r_k / R_0) = 1 + kG$, which is *exactly* the silicone-oil dish geometry of
     PRL 1992 Eq. (1), so Row B shows what the ferrofluid dish would show.
   - *Parameter expressivity:* memory-depth sweep $N_{\text{mem}} \in \{1, 2, 3, 5, 8, 13, 21, 45\}$
     and comparison of the simulated divergence / parastichy pair against the six experimental
     regimes tabulated from Douady & Couder (1992) Figs. 1-2 in the vendored reference JSON.
   - *Measurement hygiene:* the v2 measurement names `pitchfork_threshold_G_c1_*` reported the
     static fold $G_{\text{fold}}$, not $G_{c,1}$, and `lucas_branch_divergence_angle_G006` was
     evaluated at $G = 0.11$ against the asymptote; both renamed and re-referenced to the
     benchmark table. The DoG detection count ($N = 123$) is now reported as *detections*, with the
     FFT-derived scale count $\Delta u / G_{\text{emp}}$ as the lattice-consistent estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import Delaunay, Voronoi
from scipy.stats import ks_2samp, pearsonr

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "magnetic_drop_phyllotaxis"
REF_JSON = DATASET_DIR / "magnetic_drop_phyllotaxis_reference.json"
SPECIMEN_JPG = DATASET_DIR / "conifer_cone_specimen.jpg"
BIB_FILE = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "magnetic_drop_phyllotaxis_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "magnetic_drop_phyllotaxis_growth.gif"

EXPECTED_SHA256: Dict[str, str] = {
    "datasets/magnetic_drop_phyllotaxis/magnetic_drop_phyllotaxis_reference.json": (
        "9ae33106a0fea23f08c11853acb17f63a7b3770b738e5caef534dbc2107f25c8"
    ),
    "datasets/magnetic_drop_phyllotaxis/conifer_cone_specimen.jpg": (
        "7bcb68e856d4db535e622eb9674c6d7f6699047aa008ef429cbeefbfa0bd1fcb"
    ),
    "datasets/magnetic_drop_phyllotaxis/paper_citations.bib": (
        "2c3aa1ec0e4c53de2b0da472b31fc1c136af044e4acc2aedea2e2316751541c0"
    ),
}

GOLDEN_RATIO: float = 0.5 * (1.0 + math.sqrt(5.0))
GOLDEN_ANGLE_DEG: float = 360.0 / (GOLDEN_RATIO ** 2)
LUCAS_ANGLE_DEG: float = 360.0 / (3.0 + 1.0 / GOLDEN_RATIO)
F4_ANGLE_DEG: float = 360.0 / (4.0 + 1.0 / GOLDEN_RATIO)


# ============================================================================
# Section 1: Dataset SHA-256 Provenance Verification
# ============================================================================

def verify_dataset_integrity() -> Dict[str, str]:
    """Verify SHA-256 digests of all vendored reference files in `datasets/magnetic_drop_phyllotaxis/`."""
    verified: Dict[str, str] = {}
    for rel_path, expected in EXPECTED_SHA256.items():
        fpath = ROOT / rel_path
        if not fpath.exists():
            raise FileNotFoundError(f"Missing vendored dataset file: {fpath}")
        actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {rel_path}: expected {expected}, got {actual}"
            )
        verified[rel_path] = actual
    return verified


def load_reference_dataset() -> Dict[str, Any]:
    """Load `datasets/magnetic_drop_phyllotaxis/magnetic_drop_phyllotaxis_reference.json`."""
    verify_dataset_integrity()
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


# ============================================================================
# Section 2: Douady & Couder (1992, 1996) Radial Advection & Dipole Energy Law
# (Maps to PRL 1992 Eqs. 1-3 & Loop_031_magnetic_drop_phyllotaxis.lean Law 1)
# ============================================================================

def advected_radii(
    ks: np.ndarray,
    G: float,
    advection: str = "exponential",
) -> np.ndarray:
    """Compute dimensionless radial coordinates r_k / R_0 for preceding drops k >= 1.

    - `advection == 'linear'`: r_k / R_0 = 1 + k * G (PRL 1992 Eq. 1 horizontal silicone-oil dish)
    - `advection == 'exponential'`: r_k / R_0 = exp(k * G) (PRL 1992 & JTB 1996 Eq. 9 conical SAM)
    """
    if advection == "linear":
        return 1.0 + ks * G
    if advection == "exponential":
        return np.exp(np.clip(ks * G, 0.0, 30.0))
    raise ValueError(f"Unknown advection mode: {advection}")


def dipole_energy_grid(
    phi_grid: np.ndarray,
    prev_angles: np.ndarray,
    G: float,
    gamma: float = 3.0,
    advection: str = "exponential",
) -> np.ndarray:
    """Evaluate repulsive energy E(phi) = sum_{k=1}^{M} d_k(phi, phi_{n-k})^{-gamma} on rim r = R_0.

    Maps to Douady & Couder (1992) Phys. Rev. Lett. 68:2098 Eqs. (1)-(3) and
    `MathOfBiology.Loop_031.chord_distance_sq`.
    """
    if gamma <= 0.0 or len(prev_angles) == 0:
        return np.zeros_like(phi_grid, dtype=float)
    M = len(prev_angles)
    ks = np.arange(1, M + 1, dtype=float)
    r_k = advected_radii(ks, G, advection=advection)
    dphi = phi_grid[:, None] - prev_angles[None, :]
    dist2 = 1.0 + r_k[None, :] ** 2 - 2.0 * r_k[None, :] * np.cos(dphi)
    return np.sum(dist2 ** (-0.5 * gamma), axis=1)


def step_magnetic_drop(
    prev_angles: np.ndarray,
    G: float,
    gamma: float = 3.0,
    advection: str = "exponential",
    theta_guess: float | None = None,
    n_grid: int = 720,
    rng: np.random.Generator | None = None,
) -> float:
    """Place the n-th drop at phi_n = argmin_{phi in [0, 2*pi)} E(phi) on the rim r = R_0.

    When `theta_guess` is provided (adiabatic branch continuation in G), refines the local
    energy minimum closest to `theta_guess` to machine precision via bounded Brent search.
    When `gamma <= 0.0` (zero-repulsion ablation control), samples uniformly on [0, 2*pi).
    """
    if gamma <= 0.0:
        gen = rng if rng is not None else np.random.default_rng(1992)
        return float(gen.uniform(0.0, 2.0 * np.pi))

    grid = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    E_grid = dipole_energy_grid(grid, prev_angles, G, gamma=gamma, advection=advection)

    if theta_guess is not None:
        left = np.roll(E_grid, 1)
        right = np.roll(E_grid, -1)
        min_idx = np.where((E_grid <= left) & (E_grid <= right))[0]
        if len(min_idx) > 0:
            ang_diff = np.abs(np.angle(np.exp(1j * (grid[min_idx] - theta_guess))))
            best_i = int(min_idx[int(np.argmin(ang_diff))])
        else:
            best_i = int(np.argmin(E_grid))
    else:
        best_i = int(np.argmin(E_grid))

    phi0 = float(grid[best_i])
    d_step = 2.0 * np.pi / n_grid
    ks = np.arange(1, len(prev_angles) + 1, dtype=float)
    r_k = advected_radii(ks, G, advection=advection)

    def obj(phi: float) -> float:
        d2 = 1.0 + r_k ** 2 - 2.0 * r_k * np.cos(phi - prev_angles)
        return float(np.sum(d2 ** (-0.5 * gamma)))

    res = minimize_scalar(obj, bounds=(phi0 - d_step, phi0 + d_step), method="bounded")
    return float(res.x % (2.0 * np.pi))


# ============================================================================
# Section 3: Pitchfork Bifurcation Thresholds & Regular-Spiral Equilibria
# (Maps to PRL 1992 Fig. 3, JTB 1996 Sec. 3.1 & Lean 4 Theorem 2)
# ============================================================================

def integer_pitchfork_curvature_discriminant(r1: int, r2: int) -> int:
    """Exact integer cross-multiplied curvature discriminant at opposition (1,1):
        C(r_1, r_2) = 3 * r_1 * (r_2 - 1)^5 - 3 * r_2 * (r_1 + 1)^5
    matching `MathOfBiology.Loop_031.pitchfork_curvature_discriminant` in Lean 4 (Theorem 2).
    """
    return int(3 * r1 * ((r2 - 1) ** 5) - 3 * r2 * ((r1 + 1) ** 5))


def integer_dynamical_pitchfork_discriminant(r1: int, r2: int) -> int:
    """Exact integer cross-multiplied *dynamical* pitchfork discriminant for two drops:
        D(r_1, r_2) = r_1 * (r_2 - 1)^5 - 2 * r_2 * (r_1 + 1)^5
    i.e. the numerator of sum_{k=1}^{2} (-1)^{k-1} k r_k / (r_k - (-1)^k)^5 after clearing the
    positive denominators (r_1 + 1)^5 (r_2 - 1)^5. Matches
    `MathOfBiology.Loop_031.dynamical_pitchfork_discriminant` (Lean 4 Theorem 4):
    D(4, 7) = -12646 < 0 while C(4, 7) = +27687 > 0, so at linear G = 3 the distichous state is a
    static energy minimum yet dynamically unstable (G_fold < 3 < G_c1).
    """
    return int(r1 * ((r2 - 1) ** 5) - 2 * r2 * ((r1 + 1) ** 5))


def dynamical_pitchfork_discriminant_sum(G: float, advection: str = "exponential", M: int = 60) -> float:
    """Full-chain dynamical pitchfork discriminant for the steady spiral dphi = pi - eps:
        S_dyn(G) = sum_{k=1}^{M} (-1)^{k-1} k r_k / (r_k - (-1)^k)^5
    (Douady & Couder 1996 Part I Sec. 3.1 linearisation of the iterated placement map about
    opposition). S_dyn > 0 <=> the distichous (1,1) state is a stable fixed point of the map;
    the root S_dyn(G_c1) = 0 is the supercritical pitchfork threshold (0.859019 exp / 3.617555 lin).
    """
    s = 0.0
    for k in range(1, M + 1):
        rk = math.exp(min(k * G, 30.0)) if advection == "exponential" else (1.0 + k * G)
        if k % 2 == 1:
            s += k * rk / ((rk + 1.0) ** 5)
        else:
            s -= k * rk / ((rk - 1.0) ** 5)
    return s


def compute_pitchfork_bifurcation_thresholds(M: int = 60) -> Dict[str, Any]:
    """Compute exact static Hessian fold and dynamical pitchfork bifurcation thresholds G_{c,1}
    for the (1,1) -> (1,2) opposition-to-spiral transition under both exponential and linear advection.
    """
    # 1. Static Hessian fold: d^2 E / d phi^2 |_{phi=0, fixed history} = 0
    f_fold_exp_2 = lambda G: math.exp(G) / ((math.exp(G) + 1.0) ** 5) - math.exp(2.0 * G) / ((math.exp(2.0 * G) - 1.0) ** 5)
    f_fold_lin_2 = lambda G: (1.0 + G) / ((2.0 + G) ** 5) - (1.0 + 2.0 * G) / ((2.0 * G) ** 5)

    def f_fold_full(G: float, advection: str) -> float:
        s = 0.0
        for k in range(1, M + 1):
            rk = math.exp(min(k * G, 30.0)) if advection == "exponential" else (1.0 + k * G)
            if k % 2 == 1:
                s += 3.0 * rk / ((rk + 1.0) ** 5)
            else:
                s -= 3.0 * rk / ((rk - 1.0) ** 5)
        return s

    # 2. Dynamical pitchfork threshold where steady spiral dphi = pi - eps bifurcates from pi:
    # d/deps F(G, pi - eps)|_{eps=0} = 0  <=>  sum_{k>=1} (-1)^{k-1} * k * r_k / (r_k - (-1)^k)^5 = 0
    f_dyn_exp_2 = lambda G: math.exp(G) / ((math.exp(G) + 1.0) ** 5) - 2.0 * math.exp(2.0 * G) / ((math.exp(2.0 * G) - 1.0) ** 5)
    f_dyn_lin_2 = lambda G: (1.0 + G) / ((2.0 + G) ** 5) - 2.0 * (1.0 + 2.0 * G) / ((2.0 * G) ** 5)

    def f_dyn_full(G: float, advection: str) -> float:
        return dynamical_pitchfork_discriminant_sum(G, advection=advection, M=M)

    c_g3 = integer_pitchfork_curvature_discriminant(4, 7)
    c_g2 = integer_pitchfork_curvature_discriminant(3, 5)
    d_g3 = integer_dynamical_pitchfork_discriminant(4, 7)
    d_g4 = integer_dynamical_pitchfork_discriminant(5, 9)

    return {
        "static_fold_exp_2drop": float(brentq(f_fold_exp_2, 0.5, 2.0)),
        "static_fold_exp_full": float(brentq(lambda g: f_fold_full(g, "exponential"), 0.5, 2.0)),
        "static_fold_lin_2drop": float(brentq(f_fold_lin_2, 0.8, 3.5)),
        "static_fold_lin_full": float(brentq(lambda g: f_fold_full(g, "linear"), 0.8, 3.5)),
        "dynamical_pitchfork_exp_2drop": float(brentq(f_dyn_exp_2, 0.5, 2.0)),
        "dynamical_pitchfork_exp_full": float(brentq(lambda g: f_dyn_full(g, "exponential"), 0.5, 2.0)),
        "dynamical_pitchfork_lin_2drop": float(brentq(f_dyn_lin_2, 1.0, 5.0)),
        "dynamical_pitchfork_lin_full": float(brentq(lambda g: f_dyn_full(g, "linear"), 1.0, 5.0)),
        "integer_discriminant_G3_r4_r7": c_g3,
        "integer_discriminant_G2_r3_r5": c_g2,
        "integer_dynamical_discriminant_G3_r4_r7": d_g3,
        "integer_dynamical_discriminant_G4_r5_r9": d_g4,
    }


def measure_emergent_pitchfork_threshold(
    G_lo: float = 0.80,
    G_hi: float = 0.92,
    n_bisect: int = 10,
    n_iter: int = 160,
    tol_deg: float = 0.5,
    n_mem: int = 45,
) -> Dict[str, Any]:
    """Locate the *emergent* (1,1) -> (1,2) pitchfork threshold of the iterated placement map
    by bisection on G, with NO imposed 180-degree clamp (Pass 3 remediation of the v2 hard-coded
    constant 0.85901889). At each trial G the map is iterated `n_iter` times from a 0.05 rad
    perturbation of opposition; the state is classified as distichous when the steady divergence
    deviates from 180 deg by less than `tol_deg`. Returns the bracket midpoint, the bracket, and
    the Delta_phi(G) scan used for Panel D1, for comparison with the analytical full-chain root
    of `dynamical_pitchfork_discriminant_sum` (0.859019).
    """

    def steady_deviation(G: float) -> float:
        phi_hist: List[float] = [0.0, math.pi - 0.05]
        last = math.pi - 0.05
        ds: List[float] = []
        for _ in range(n_iter):
            prev = np.array(phi_hist[-n_mem:][::-1], dtype=float)
            guess = (phi_hist[-1] + last) % (2.0 * math.pi)
            phi_new = step_magnetic_drop(prev, G, gamma=3.0, theta_guess=guess)
            d = (phi_new - phi_hist[-1]) % (2.0 * math.pi)
            if d > math.pi:
                d = 2.0 * math.pi - d
            last = d
            phi_hist.append(phi_new)
            ds.append(math.degrees(d))
        return 180.0 - float(np.mean(ds[-20:]))

    lo, hi = float(G_lo), float(G_hi)
    assert steady_deviation(lo) > tol_deg and steady_deviation(hi) <= tol_deg, "bisection bracket invalid"
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if steady_deviation(mid) > tol_deg:
            lo = mid
        else:
            hi = mid
    scan_G = np.linspace(0.70, 0.95, 26)
    scan_dphi = np.array([180.0 - steady_deviation(float(g)) for g in scan_G], dtype=float)
    analytic = float(brentq(lambda g: dynamical_pitchfork_discriminant_sum(g, "exponential", 60), 0.5, 2.0))
    return {
        "G_c1_emergent": 0.5 * (lo + hi),
        "bracket": (lo, hi),
        "G_c1_analytic_full": analytic,
        "abs_error": abs(0.5 * (lo + hi) - analytic),
        "scan_G": scan_G,
        "scan_dphi_deg": scan_dphi,
    }


def spiral_equilibrium_force_and_hessian(
    G: float,
    dphi_deg: float,
    gamma: float = 3.0,
    advection: str = "exponential",
    M: int = 55,
) -> Tuple[float, float]:
    """Evaluate exact regular-spiral azimuthal gradient F(G, dphi) = dE/dphi|_{phi=0}
    and Hessian curvature S(G, dphi) = d^2E/dphi^2|_{phi=0} when preceding drops lie at
    angles phi_{n-k} = -k * dphi.

    Maps to Douady & Couder (1992) Fig. 3 and (1996) Part I Sec. 3 Eq. (11).
    """
    dphi = math.radians(dphi_deg)
    ks = np.arange(1, M + 1, dtype=float)
    r_k = advected_radii(ks, G, advection=advection)
    ang = ks * dphi
    u = 1.0 + r_k ** 2 - 2.0 * r_k * np.cos(ang)
    dE = -gamma * float(np.sum(r_k * np.sin(ang) * (u ** (-1.0 - 0.5 * gamma))))
    d2E = gamma * float(
        np.sum(
            r_k
            * ((gamma + 2.0) * r_k * (np.sin(ang) ** 2) - np.cos(ang) * u)
            * (u ** (-2.0 - 0.5 * gamma))
        )
    )
    return dE, d2E


def compute_equilibrium_bifurcation_diagram(
    G_min: float = 0.025,
    G_max: float = 0.95,
    n_G: int = 75,
    gamma: float = 3.0,
) -> Dict[str, Any]:
    """Trace all stable (d^2E/dphi^2 > 0) and unstable saddle (d^2E/dphi^2 < 0) regular-spiral
    equilibrium branches F(G, dphi) = 0 across G in [G_min, G_max].
    """
    G_grid = np.geomspace(G_min, G_max, n_G)
    dphi_scan = np.linspace(72.0, 179.8, 320)
    stable_G: List[float] = []
    stable_dphi: List[float] = []
    unstable_G: List[float] = []
    unstable_dphi: List[float] = []

    for G in G_grid:
        # Distichous state dphi = 180 deg is always a critical point by symmetry. Its stability
        # as a fixed point of the iterated map is decided by the sign of the dynamical
        # discriminant S_dyn(G) (Pass 3: computed, not hard-coded).
        if dynamical_pitchfork_discriminant_sum(float(G), "exponential", 60) > 0.0:
            stable_G.append(float(G))
            stable_dphi.append(180.0)
        else:
            unstable_G.append(float(G))
            unstable_dphi.append(180.0)

        forces = [spiral_equilibrium_force_and_hessian(float(G), float(a), gamma=gamma)[0] for a in dphi_scan]
        for i in range(len(dphi_scan) - 1):
            if forces[i] * forces[i + 1] <= 0.0:
                try:
                    root = float(
                        brentq(
                            lambda a: spiral_equilibrium_force_and_hessian(float(G), a, gamma=gamma)[0],
                            float(dphi_scan[i]),
                            float(dphi_scan[i + 1]),
                        )
                    )
                    _, curv = spiral_equilibrium_force_and_hessian(float(G), root, gamma=gamma)
                    if curv > 0.0:
                        stable_G.append(float(G))
                        stable_dphi.append(root)
                    else:
                        unstable_G.append(float(G))
                        unstable_dphi.append(root)
                except ValueError:
                    continue

    return {
        "stable_G": np.array(stable_G, dtype=float),
        "stable_dphi": np.array(stable_dphi, dtype=float),
        "unstable_G": np.array(unstable_G, dtype=float),
        "unstable_dphi": np.array(unstable_dphi, dtype=float),
    }


# ============================================================================
# Section 4: Contact Parastichy Extraction (nearest-neighbour & Delaunay) & Iterative Dish Simulations
# ============================================================================

def extract_delaunay_parastichy_pair(
    x: np.ndarray,
    y: np.ndarray,
    radii: np.ndarray,
) -> Tuple[int, int]:
    """Extract the dominant opposing contact parastichy pair (m, n) (with m <= n) from
    Delaunay triangulation index differences |i - j| in the middle annulus of the dish.
    """
    n_pts = len(x)
    if n_pts < 16:
        return (1, 1)
    pts = np.column_stack([x, y])
    tri = Delaunay(pts)
    r_lo = float(np.percentile(radii, 18))
    r_hi = float(np.percentile(radii, 86))

    edge_diffs: List[int] = []
    edge_dists: Dict[int, List[float]] = {}
    for simp in tri.simplices:
        for a, b in ((simp[0], simp[1]), (simp[1], simp[2]), (simp[0], simp[2])):
            i, j = (a, b) if a < b else (b, a)
            if radii[i] < r_lo or radii[j] > r_hi:
                continue
            dk = int(j - i)
            if 1 <= dk <= 25:
                dist = float(math.hypot(x[i] - x[j], y[i] - y[j]))
                edge_diffs.append(dk)
                edge_dists.setdefault(dk, []).append(dist)

    if not edge_diffs:
        return (1, 1)

    vals, counts = np.unique(edge_diffs, return_counts=True)
    # Score each candidate parastichy number by edge count / median chord length
    scored: List[Tuple[float, int]] = []
    for v, c in zip(vals, counts):
        if c < 4:
            continue
        med_d = float(np.median(edge_dists[int(v)])) + 1e-6
        scored.append((float(c) / med_d, int(v)))
    scored.sort(reverse=True)
    if len(scored) == 0:
        return (1, 1)
    if len(scored) == 1:
        return (scored[0][1], scored[0][1])
    p1, p2 = scored[0][1], scored[1][1]
    return (min(p1, p2), max(p1, p2))


def extract_contact_parastichy_pair(
    phis: np.ndarray,
    radii: np.ndarray,
    k_max: int = 34,
) -> Tuple[Tuple[int, int], float]:
    """Contact parastichy pair by Douady & Couder's own definition (PRL 1992 Fig. 2 caption;
    JTB 1996 Part I Sec. 4): each element is joined to its two nearest neighbours of opposite
    winding sense; the index differences (k_1, k_2) of those neighbours are the parastichy
    numbers. Evaluated on the TRUE simulated positions for every drop in the middle annulus
    (self-similarity is not assumed), returning the modal pair and its frequency.
    """
    n = len(phis)
    if n < 12:
        return (1, 1), 1.0
    x = radii * np.cos(phis)
    y = radii * np.sin(phis)
    lo, hi = int(0.15 * n), int(0.85 * n)
    pairs: List[Tuple[int, int]] = []
    for i in range(lo, hi):
        ks = np.arange(1, min(k_max, i) + 1)
        j = i - ks
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        wrapped = (phis[j] - phis[i] + math.pi) % (2.0 * math.pi) - math.pi
        sgn = np.sign(wrapped)
        order = np.argsort(d)
        k1 = int(ks[order[0]])
        k2 = None
        for o in order[1:]:
            if sgn[o] != sgn[order[0]] and sgn[o] != 0.0:
                k2 = int(ks[o])
                break
        if k2 is None:
            continue
        pairs.append((min(k1, k2), max(k1, k2)))
    if not pairs:
        return (1, 1), 1.0
    vals, counts = np.unique(np.array(pairs), axis=0, return_counts=True)
    best = int(np.argmax(counts))
    return (int(vals[best][0]), int(vals[best][1])), float(counts[best] / len(pairs))


def compute_voronoi_area_cv(x: np.ndarray, y: np.ndarray, radii: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute radial-normalized Voronoi cell areas and their coefficient of variation (CV)
    for interior drops to quantify packing regularity against random placement controls.
    """
    pts = np.column_stack([x, y])
    vor = Voronoi(pts)
    r_lo = float(np.percentile(radii, 20))
    r_hi = float(np.percentile(radii, 78))
    norm_areas: List[float] = []
    for idx, reg_idx in enumerate(vor.point_region):
        if not (r_lo <= radii[idx] <= r_hi):
            continue
        reg = vor.regions[reg_idx]
        if not reg or -1 in reg or len(reg) < 3:
            continue
        poly = vor.vertices[reg]
        # Shoelace area normalized by local r^2
        x_p, y_p = poly[:, 0], poly[:, 1]
        area = 0.5 * float(np.abs(np.dot(x_p, np.roll(y_p, 1)) - np.dot(y_p, np.roll(x_p, 1))))
        norm_a = area / (float(radii[idx]) ** 2 + 1e-9)
        if 0.0005 < norm_a < 2.5:
            norm_areas.append(norm_a)
    arr = np.array(norm_areas, dtype=float)
    if len(arr) < 5:
        return 1.0, np.ones(5, dtype=float)
    arr_norm = arr / float(np.mean(arr))
    cv = float(np.std(arr_norm) / (np.mean(arr_norm) + 1e-12))
    return cv, arr_norm


def simulate_fixed_G_dish(
    G: float,
    n_drops: int = 96,
    n_mem: int = 45,
    gamma: float = 3.0,
    advection: str = "exponential",
    branch: str = "fibonacci",
    seed: int = 1992,
) -> Dict[str, Any]:
    """Simulate a ferrofluid magnetic-drop dish at plastochron ratio G.

    - `branch == 'fibonacci'`: warms up adiabatically from distichous opposition (G=0.82 -> G)
      along the main Fibonacci branch.
    - `branch == 'lucas'`: seeds initial history in the Lucas (1,3) basin (~99.50 deg) and
      evolves under the iterative dipole minimization rule at G.
    - `n_mem == 1`: single-drop memory ablation control (eliminates k >= 2 destabilizing repulsion).
    - `gamma == 0.0`: zero-repulsion random placement control.
    """
    rng = np.random.default_rng(seed)

    if branch == "lucas":
        lucas_rad = math.radians(LUCAS_ANGLE_DEG)
        phi_hist: List[float] = [float((k * lucas_rad) % (2.0 * math.pi)) for k in range(32)]
        last_dphi = lucas_rad
    else:
        phi_hist = [0.0, math.pi - 0.012]
        last_dphi = math.pi - 0.012
        if G < 0.78 and n_mem > 1 and gamma > 0.0:
            for G_warm in np.linspace(0.82, G, 30):
                for _ in range(10):
                    prev = np.array(phi_hist[-n_mem:][::-1], dtype=float)
                    guess = (phi_hist[-1] + last_dphi) % (2.0 * math.pi)
                    phi_new = step_magnetic_drop(
                        prev, float(G_warm), gamma=gamma, advection=advection, theta_guess=guess, rng=rng
                    )
                    dphi = (phi_new - phi_hist[-1]) % (2.0 * math.pi)
                    if dphi > math.pi:
                        dphi = 2.0 * math.pi - dphi
                    last_dphi = dphi
                    phi_hist.append(phi_new)

    dphi_deg_series: List[float] = []
    for _ in range(n_drops):
        prev = np.array(phi_hist[-n_mem:][::-1], dtype=float)
        guess = (phi_hist[-1] + last_dphi) % (2.0 * math.pi) if (n_mem > 1 and gamma > 0.0 and G < 0.85) else None
        phi_new = step_magnetic_drop(
            prev, G, gamma=gamma, advection=advection, theta_guess=guess, rng=rng
        )
        dphi = (phi_new - phi_hist[-1]) % (2.0 * math.pi)
        if dphi > math.pi:
            dphi = 2.0 * math.pi - dphi
        last_dphi = dphi
        phi_hist.append(phi_new)
        dphi_deg_series.append(float(math.degrees(dphi)))

    phis = np.array(phi_hist[-n_drops:], dtype=float)
    ages = np.arange(len(phis) - 1, -1, -1, dtype=float)
    # Pass 3: triangulate the TRUE advected radii of the model (v2 clipped the exponential
    # growth rate to g_vis in [0.012, 0.12] for display and triangulated that fudge, which
    # mislabelled the Delaunay pair at G = 0.30 as (3,5) and at G = 0.5 as (5,8)).
    radii = advected_radii(ages, G, advection=advection)
    x = radii * np.cos(phis)
    y = radii * np.sin(phis)
    # Log-radial projection rho = 1 + ln(r/R_0) = 1 + k G for exponential advection: identical
    # to the silicone-oil dish geometry r/R_0 = 1 + k G of PRL 1992 Eq. (1) (display only).
    rho = 1.0 + ages * G if advection == "exponential" else radii
    mean_div = float(np.mean(dphi_deg_series[-24:]))

    if n_mem == 1 or abs(mean_div - 180.0) < 0.5:
        # Distichous opposition detected from the emergent divergence (no G threshold imposed)
        parastichy_pair = (1, 1)
        delaunay_pair = (1, 1)
        contact_frac = 1.0
    else:
        # Primary: Douady & Couder nearest-neighbour contact definition on the true positions.
        # Secondary: Delaunay edge-census extractor (v2 primary; biased to the higher Fibonacci
        # pair near the van Iterson transitions, see Pass 3 audit).
        parastichy_pair, contact_frac = extract_contact_parastichy_pair(phis, radii)
        delaunay_pair = extract_delaunay_parastichy_pair(x, y, radii)
    vor_cv, vor_areas = compute_voronoi_area_cv(x, y, radii)

    return {
        "G": float(G),
        "gamma": float(gamma),
        "n_mem": int(n_mem),
        "advection": advection,
        "branch": branch,
        "phis": phis,
        "radii": radii,
        "x": x,
        "y": y,
        "rho_display": rho,
        "x_display": rho * np.cos(phis),
        "y_display": rho * np.sin(phis),
        "dphi_deg_series": np.array(dphi_deg_series, dtype=float),
        "mean_divergence_deg": mean_div,
        "std_divergence_deg": float(np.std(dphi_deg_series[-24:])),
        "parastichy_pair": parastichy_pair,
        "contact_pair_fraction": float(contact_frac),
        "delaunay_pair": delaunay_pair,
        "voronoi_area_cv": vor_cv,
        "voronoi_norm_areas": vor_areas,
    }


def simulate_adiabatic_branch_trajectories() -> Dict[str, Any]:
    """Simulate adiabatic G-ramps along both the Fibonacci branch and the Lucas branch,
    and compare against the benchmark points in `magnetic_drop_phyllotaxis_reference.json`.
    """
    ref = load_reference_dataset()
    fib_bench = ref["bifurcation_branches_exponential_gamma3"]["fibonacci_main_branch"]
    lucas_bench = ref["bifurcation_branches_exponential_gamma3"]["lucas_secondary_branch"]

    # Dense Fibonacci ramp from G=0.92 down to G=0.025. Pass 3: the ramp starts above the
    # pitchfork and the iterated map is left to find opposition by itself (v2 short-circuited
    # every G >= 0.85901889 to exactly 180 deg, so the bifurcation was imposed, not emergent).
    G_fib_dense = np.linspace(0.92, 0.025, 95)
    phi_hist: List[float] = [0.0, math.pi - 0.01]
    last_dphi = math.pi - 0.01
    fib_dphi_dense: List[float] = []
    for G in G_fib_dense:
        local_dphis: List[float] = []
        for _ in range(22):
            prev = np.array(phi_hist[-45:][::-1], dtype=float)
            guess = (phi_hist[-1] + last_dphi) % (2.0 * math.pi)
            phi_new = step_magnetic_drop(prev, float(G), gamma=3.0, theta_guess=guess)
            dphi = (phi_new - phi_hist[-1]) % (2.0 * math.pi)
            if dphi > math.pi:
                dphi = 2.0 * math.pi - dphi
            last_dphi = dphi
            phi_hist.append(phi_new)
            local_dphis.append(float(math.degrees(dphi)))
        fib_dphi_dense.append(float(np.mean(local_dphis[-6:])))

    # Evaluate at the exact 10 benchmark G points on the Fibonacci branch
    fib_sim_at_bench: List[float] = []
    fib_ref_at_bench: List[float] = []
    fib_sim_pairs: List[Tuple[int, int]] = []
    fib_ref_pairs: List[Tuple[int, int]] = []
    for row in fib_bench:
        g_val = float(row["G"])
        fib_ref_at_bench.append(float(row["divergence_deg"]))
        fib_ref_pairs.append((int(row["parastichy_pair"][0]), int(row["parastichy_pair"][1])))
        sim_g = simulate_fixed_G_dish(g_val, n_drops=70, branch="fibonacci")
        fib_sim_at_bench.append(float(sim_g["mean_divergence_deg"]))
        fib_sim_pairs.append(sim_g["parastichy_pair"])
    n_pair_match = int(sum(1 for a, b in zip(fib_sim_pairs, fib_ref_pairs) if a == b))

    # Evaluate at the exact 6 benchmark G points on the Lucas branch
    lucas_sim_at_bench: List[float] = []
    lucas_ref_at_bench: List[float] = []
    lucas_G_list: List[float] = []
    for row in lucas_bench:
        g_val = float(row["G"])
        lucas_G_list.append(g_val)
        lucas_ref_at_bench.append(float(row["divergence_deg"]))
        sim_l = simulate_fixed_G_dish(g_val, n_drops=70, branch="lucas")
        lucas_sim_at_bench.append(float(sim_l["mean_divergence_deg"]))

    fib_sim_arr = np.array(fib_sim_at_bench, dtype=float)
    fib_ref_arr = np.array(fib_ref_at_bench, dtype=float)
    r_fib, p_fib = pearsonr(fib_sim_arr, fib_ref_arr)
    rmse_fib = float(np.sqrt(np.mean((fib_sim_arr - fib_ref_arr) ** 2)))

    lucas_sim_arr = np.array(lucas_sim_at_bench, dtype=float)
    lucas_ref_arr = np.array(lucas_ref_at_bench, dtype=float)
    rmse_lucas = float(np.sqrt(np.mean((lucas_sim_arr - lucas_ref_arr) ** 2)))

    return {
        "G_fib_dense": G_fib_dense,
        "fib_dphi_dense": np.array(fib_dphi_dense, dtype=float),
        "fib_bench_G": np.array([float(r["G"]) for r in fib_bench], dtype=float),
        "fib_sim_deg": fib_sim_arr,
        "fib_ref_deg": fib_ref_arr,
        "fib_sim_pairs": fib_sim_pairs,
        "fib_ref_pairs": fib_ref_pairs,
        "fib_pair_matches": n_pair_match,
        "pearson_r_fib": float(r_fib),
        "pearson_p_fib": float(p_fib),
        "rmse_fib_deg": rmse_fib,
        "lucas_bench_G": np.array(lucas_G_list, dtype=float),
        "lucas_sim_deg": lucas_sim_arr,
        "lucas_ref_deg": lucas_ref_arr,
        "rmse_lucas_deg": rmse_lucas,
    }


def run_gamma_universality_sweep(G: float = 0.06) -> List[Dict[str, Any]]:
    """Sweep repulsive power-law exponent gamma in {1.0, 3.0, 5.0} at G=0.06 to test
    Douady & Couder's (1992, 1996) universality claim that the golden-angle attractor
    is insensitive to the steepness of the repulsive interaction law.
    """
    results: List[Dict[str, Any]] = []
    for gamma in (1.0, 3.0, 5.0):
        sim = simulate_fixed_G_dish(G, n_drops=85, gamma=gamma, branch="fibonacci")
        results.append(
            {
                "gamma": float(gamma),
                "mean_divergence_deg": float(sim["mean_divergence_deg"]),
                "abs_err_from_golden_deg": float(abs(sim["mean_divergence_deg"] - GOLDEN_ANGLE_DEG)),
                "parastichy_pair": sim["parastichy_pair"],
            }
        )
    return results


def run_memory_depth_sweep(
    G: float = 0.06,
    n_mems: Tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 45),
    tol_deg: float = 0.5,
) -> Dict[str, Any]:
    """Pass 3 parameter-expressivity sweep of the interaction memory depth N_mem at fixed G.

    Douady & Couder (1996, Part I Sec. 2) truncate the repulsion sum to the drops still inside
    the dish; here N_mem controls how many older drops repel the new one. N_mem = 1 locks
    opposition (180 deg, Lean 4 Theorem 1), N_mem = 2 gives the two-drop (1,2) spiral, and the
    divergence converges to the full-history value once the k-th drop's repulsion
    r_k^{-3} ~ exp(-3 k G) is negligible. Returns the sweep and the smallest N_mem whose
    divergence is within `tol_deg` of the deepest-memory reference.
    """
    rows: List[Dict[str, Any]] = []
    for nm in n_mems:
        sim = simulate_fixed_G_dish(G, n_drops=80, n_mem=int(nm), branch="fibonacci")
        rows.append(
            {
                "n_mem": int(nm),
                "mean_divergence_deg": float(sim["mean_divergence_deg"]),
                "std_divergence_deg": float(sim["std_divergence_deg"]),
                "parastichy_pair": sim["parastichy_pair"],
            }
        )
    ref_div = rows[-1]["mean_divergence_deg"]
    abs_err = [abs(r["mean_divergence_deg"] - ref_div) for r in rows]
    n_conv = None
    for r, e in zip(rows, abs_err):
        if e < tol_deg:
            n_conv = r["n_mem"]
            break
    return {
        "G": float(G),
        "rows": rows,
        "abs_err_vs_deepest_deg": abs_err,
        "reference_divergence_deg": ref_div,
        "n_mem_converged": int(n_conv) if n_conv is not None else -1,
    }


def compare_ferrofluid_experimental_regimes() -> Dict[str, Any]:
    """Pass 3 comparison against the six experimental regimes of the ferrofluid dish tabulated
    from Douady & Couder (1992) Figs. 1-2 in `magnetic_drop_phyllotaxis_reference.json`
    (`ferrofluid_experiment_parameters.experimental_regimes`): simulated steady divergence and
    Delaunay parastichy pair at each experimental G versus the observed values.
    """
    ref = load_reference_dataset()
    regimes = ref["ferrofluid_experiment_parameters"]["experimental_regimes"]
    rows: List[Dict[str, Any]] = []
    for row in regimes:
        branch = "lucas" if "Lucas" in str(row["label"]) else "fibonacci"
        sim = simulate_fixed_G_dish(float(row["G_exp"]), n_drops=96, branch=branch)
        obs_pair = (int(row["observed_parastichies"][0]), int(row["observed_parastichies"][1]))
        rows.append(
            {
                "label": str(row["label"]),
                "G_exp": float(row["G_exp"]),
                "observed_divergence_deg": float(row["mean_divergence_deg"]),
                "simulated_divergence_deg": float(sim["mean_divergence_deg"]),
                "observed_pair": obs_pair,
                "simulated_pair": sim["parastichy_pair"],
                "pair_match": bool(sim["parastichy_pair"] == obs_pair),
                "branch": branch,
            }
        )
    obs = np.array([r["observed_divergence_deg"] for r in rows], dtype=float)
    sim_arr = np.array([r["simulated_divergence_deg"] for r in rows], dtype=float)
    return {
        "rows": rows,
        "rmse_divergence_deg": float(np.sqrt(np.mean((sim_arr - obs) ** 2))),
        "max_abs_divergence_err_deg": float(np.max(np.abs(sim_arr - obs))),
        "n_pair_match": int(sum(1 for r in rows if r["pair_match"])),
        "n_regimes": len(rows),
    }


# ============================================================================
# Section 5: Biological Specimen Segmentation (`conifer_cone_specimen.jpg`)
# ============================================================================

LOG_POLAR_U_RANGE_PX: Tuple[float, float] = (55.0, 240.0)
LOG_POLAR_SHAPE: Tuple[int, int] = (128, 256)
FFT_U_ZERO_PAD: int = 8


def log_polar_spiral_spectrum(
    log_polar_img: np.ndarray,
    u_grid: np.ndarray,
    m_modes: np.ndarray,
    pad: int = FFT_U_ZERO_PAD,
    omega_min: float = 3.0,
) -> Dict[str, Any]:
    """Hann-windowed, u-zero-padded 2D FFT of a log-polar (u = ln r, theta) image.

    For every azimuthal harmonic m the dominant radial frequency omega_m (rad per unit u,
    signed per `np.fft.fftfreq`) is located with `pad`-fold zero padding in u (sub-bin
    refinement of the coarse 2*pi/Delta_u bin spacing). A regular spiral lattice with
    divergence alpha and plastochron ratio G satisfies the reciprocal-lattice condition
        m * alpha + omega_m * G = 2 * pi * j_m
    for every peak (Douady & Couder 1996 Part I Sec. 4 van Iterson lattice; this is the
    Fourier-space form of the parastichy winding relation).
    """
    nu, nth = log_polar_img.shape
    L_u = float(u_grid[-1] - u_grid[0]) * (nu / (nu - 1))
    win = np.hanning(nu)[:, None]
    F = np.fft.fft2(log_polar_img * win, s=(nu * pad, nth))
    P = np.abs(F) ** 2
    omega = np.fft.fftfreq(nu * pad, d=L_u / nu) * 2.0 * np.pi
    valid = np.abs(omega) >= omega_min
    peak_omega: Dict[int, float] = {}
    peak_power: Dict[int, float] = {}
    pos_power: List[float] = []
    neg_power: List[float] = []
    for mm in m_modes:
        col = P[:, int(mm)].copy()
        col[~valid] = 0.0
        i = int(np.argmax(col))
        peak_omega[int(mm)] = float(omega[i])
        peak_power[int(mm)] = float(col[i])
        pos_power.append(float(np.max(col[(omega > 0) & valid])))
        neg_power.append(float(np.max(col[(omega < 0) & valid])))
    return {
        "omega_axis": omega,
        "peak_omega": peak_omega,
        "peak_power": peak_power,
        "pos_power": np.array(pos_power, dtype=float),
        "neg_power": np.array(neg_power, dtype=float),
        "L_u": L_u,
    }


def invert_reciprocal_lattice_peaks(
    m1: int,
    omega1: float,
    m2: int,
    omega2: float,
    alpha_guess_deg: float = GOLDEN_ANGLE_DEG,
) -> Dict[str, float]:
    """Solve the 2x2 reciprocal-lattice system
        m1 * alpha + omega1 * G = 2 pi j1,   m2 * alpha + omega2 * G = 2 pi j2
    for (alpha, G) given the two dominant parastichy harmonics (m1, m2) and their sub-bin
    radial frequencies. The winding integers j_m = round(m * alpha_guess / 2 pi) are fixed by
    the Fibonacci guess (j_5 = 2, j_8 = 3 for alpha ~ 137.5 deg). A negative G solution is the
    mirror-chirality lattice (alpha -> 2 pi - alpha, G -> -G leaves the equations invariant),
    so the reported divergence is folded to [0, 180] deg and G to |G|.
    """
    a_g = math.radians(alpha_guess_deg)
    j1 = round(m1 * a_g / (2.0 * math.pi))
    j2 = round(m2 * a_g / (2.0 * math.pi))
    A = np.array([[float(m1), omega1], [float(m2), omega2]], dtype=float)
    b = np.array([2.0 * math.pi * j1, 2.0 * math.pi * j2], dtype=float)
    alpha, G = np.linalg.solve(A, b)
    chirality = 1.0 if G >= 0.0 else -1.0
    alpha_deg = math.degrees(alpha) % 360.0
    alpha_fold = min(alpha_deg, 360.0 - alpha_deg)
    return {
        "divergence_deg": float(alpha_fold),
        "divergence_raw_deg": float(alpha_deg),
        "G": float(abs(G)),
        "chirality": chirality,
        "j1": float(j1),
        "j2": float(j2),
    }


def synthetic_log_polar_lattice_image(
    phis: np.ndarray,
    G: float,
    u_span: Tuple[float, float],
    shape: Tuple[int, int] = LOG_POLAR_SHAPE,
    sigma_px: float = 1.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a simulated drop sequence as a Gaussian-blob log-polar image (u = k G, theta =
    phi_k) on the same grid as the specimen pipeline, to validate `invert_reciprocal_lattice_peaks`
    against a lattice whose divergence and G are known exactly.
    """
    nu, nth = shape
    u_grid = np.linspace(u_span[0], u_span[1], nu)
    img = np.zeros((nu, nth), dtype=float)
    ages = np.arange(len(phis) - 1, -1, -1, dtype=float)
    u_k = ages * G
    du = (u_span[1] - u_span[0]) / (nu - 1)
    for uk, pk in zip(u_k, phis):
        if uk < u_span[0] or uk > u_span[1]:
            continue
        iu = (uk - u_span[0]) / du
        it = (pk % (2.0 * math.pi)) / (2.0 * math.pi) * nth
        img[int(round(iu)) % nu, int(round(it)) % nth] += 1.0
    img = ndi.gaussian_filter(img, sigma_px, mode=("reflect", "wrap"))
    img -= img.mean(axis=1, keepdims=True)
    img /= img.std(axis=1, keepdims=True) + 1e-6
    return img, u_grid


def validate_lattice_inversion_on_simulation(G: float = 0.06) -> Dict[str, Any]:
    """Pass 3 self-consistency check: run the specimen FFT pipeline on the simulated G lattice
    and recover its divergence angle and plastochron ratio from the (5, 8) reciprocal peaks.
    """
    sim = simulate_fixed_G_dish(G, n_drops=110, branch="fibonacci")
    ages = np.arange(len(sim["phis"]) - 1, -1, -1, dtype=float)
    u_hi = float(ages.max() * G)
    span = (0.35 * u_hi, 0.95 * u_hi)
    img, u_grid = synthetic_log_polar_lattice_image(sim["phis"], G, span)
    m_modes = np.arange(3, 19, dtype=int)
    spec = log_polar_spiral_spectrum(img, u_grid, m_modes)
    inv = invert_reciprocal_lattice_peaks(5, spec["peak_omega"][5], 8, spec["peak_omega"][8])
    return {
        "true_divergence_deg": float(sim["mean_divergence_deg"]),
        "true_G": float(G),
        "recovered_divergence_deg": inv["divergence_deg"],
        "recovered_G": inv["G"],
        "divergence_abs_err_deg": abs(inv["divergence_deg"] - float(sim["mean_divergence_deg"])),
        "G_rel_err": abs(inv["G"] - G) / G,
        "peak_omega_5": spec["peak_omega"][5],
        "peak_omega_8": spec["peak_omega"][8],
        "img": img,
        "u_grid": u_grid,
    }


def segment_conifer_cone_specimen() -> Dict[str, Any]:
    """Detect DoG blob responses on the woody scales and extract the opposing Fibonacci (5, 8)
    spiral parastichy families from `datasets/magnetic_drop_phyllotaxis/conifer_cone_specimen.jpg`
    via log-polar (u = ln r, theta) 2D FFT; then invert the (5, 8) reciprocal-lattice peaks for
    the specimen's own divergence angle and plastochron ratio (Pass 3).

    Pass 3 disclosure: the DoG local-maximum count (N = 123 in the full disc) is a count of blob
    *detections*, not of scales. Rank-ordering those detections by radius does not recover a
    137.5 deg divergence sequence (circular mean resultant R < 0.2), because adjacent scales on a
    mature cone differ in apparent radius by less than the detection jitter and several scales
    yield two DoG maxima (umbo + scale tip). The lattice-level analysis therefore rests on the
    FFT, whose scale count Delta_u / G_fft is reported alongside the detection count.
    """
    verify_dataset_integrity()
    im = Image.open(SPECIMEN_JPG).convert("RGB")
    rgb = np.asarray(im, dtype=float) / 255.0
    H, W, _ = rgb.shape
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Apex center (cx, cy) = (316, 307) determined from polar symmetry of apical depression
    cx, cy = 316.0, 307.0
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(xx - cx, yy - cy)

    # 1. Scale blob detection via Difference-of-Gaussians (DoG)
    dog_scales = ndi.gaussian_filter(gray, 5.5) - ndi.gaussian_filter(gray, 14.0)
    local_max = (
        (dog_scales == ndi.maximum_filter(dog_scales, size=23))
        & (dog_scales > 0.012)
        & (rr >= 26.0)
        & (rr <= 255.0)
    )
    umask_y, umask_x = np.where(local_max)
    u_radii = np.hypot(umask_x - cx, umask_y - cy)
    order = np.argsort(u_radii)
    umask_x = umask_x[order]
    umask_y = umask_y[order]
    u_radii = u_radii[order]

    # 2. Log-polar unwrapping (u = ln r, theta) on annulus r in [55, 240] px
    r_in, r_out = LOG_POLAR_U_RANGE_PX
    dog_fine = ndi.gaussian_filter(gray, 3.5) - ndi.gaussian_filter(gray, 11.0)
    u_grid = np.linspace(math.log(r_in), math.log(r_out), LOG_POLAR_SHAPE[0])
    th_grid = np.linspace(0.0, 2.0 * math.pi, LOG_POLAR_SHAPE[1], endpoint=False)
    UU, TT = np.meshgrid(u_grid, th_grid, indexing="ij")
    RR = np.exp(UU)
    X_pol = cx + RR * np.cos(TT)
    Y_pol = cy + RR * np.sin(TT)
    log_polar_img = ndi.map_coordinates(dog_fine, [Y_pol, X_pol], order=1)
    log_polar_img -= log_polar_img.mean(axis=1, keepdims=True)
    log_polar_img /= log_polar_img.std(axis=1, keepdims=True) + 1e-6

    # 3. Chirality-separated spiral spectrum (v2 unpadded spectrum kept for the C3 bars) and
    #    Pass 3 zero-padded sub-bin peak refinement for the lattice inversion
    m_modes = np.arange(3, 19, dtype=int)
    spec = log_polar_spiral_spectrum(log_polar_img, u_grid, m_modes)
    ccw_power = spec["pos_power"]
    cw_power = spec["neg_power"]
    m_ccw_dom = int(m_modes[int(np.argmax(ccw_power))])
    m_cw_dom = int(m_modes[int(np.argmax(cw_power))])
    parastichy_pair = (min(m_cw_dom, m_ccw_dom), max(m_cw_dom, m_ccw_dom))

    # 4. Reciprocal-lattice inversion on the two dominant harmonics -> (alpha_emp, G_emp)
    m_lo, m_hi = parastichy_pair
    inv = invert_reciprocal_lattice_peaks(m_lo, spec["peak_omega"][m_lo], m_hi, spec["peak_omega"][m_hi])
    delta_u = math.log(r_out / r_in)
    n_scales_fft = delta_u / max(inv["G"], 1e-9)
    # Reciprocal-lattice closure residual of the secondary harmonic m3 = m_hi - m_lo: for a
    # perfect lattice omega_{m_hi - m_lo} = omega_{m_hi} - omega_{m_lo}; the residual (in bins of
    # 2 pi / Delta_u) measures how far the specimen is from a log-periodic lattice.
    m_diff = int(m_hi - m_lo)
    bin_width = 2.0 * math.pi / spec["L_u"]
    closure_residual_bins = float(
        abs(spec["peak_omega"][m_diff] - (spec["peak_omega"][m_hi] - spec["peak_omega"][m_lo])) / bin_width
    ) if m_diff in spec["peak_omega"] else float("nan")

    # 5. Detection-based diagnostics (disclosed as unreliable at lattice level; see docstring)
    sel = (u_radii >= r_in) & (u_radii <= r_out)
    n_annulus_detections = int(np.sum(sel))
    G_rank_count = float(delta_u / max(n_annulus_detections, 1))
    th_det = np.arctan2(umask_y[sel] - cy, umask_x[sel] - cx)
    succ = np.diff(th_det) % (2.0 * math.pi)
    rank_resultant_R = float(abs(np.mean(np.exp(1j * succ)))) if len(succ) > 2 else 0.0
    ranks = np.arange(n_annulus_detections, dtype=float)
    r_sel = u_radii[sel].astype(float)
    A = np.vstack([ranks, np.ones_like(ranks)]).T

    def _r2(y: np.ndarray) -> float:
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        res = y - A @ coef
        return float(1.0 - np.sum(res ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))

    r2_linear = _r2(r_sel)
    r2_exponential = _r2(np.log(r_sel))

    return {
        "rgb": rgb,
        "cx": cx,
        "cy": cy,
        "scale_x": umask_x,
        "scale_y": umask_y,
        "scale_radii": u_radii,
        "n_scales_total": int(len(umask_x)),
        "n_scales_annulus": n_annulus_detections,
        "G_empirical_annulus": G_rank_count,
        "rank_sequence_resultant_R": rank_resultant_R,
        "radial_law_R2_linear": r2_linear,
        "radial_law_R2_exponential": r2_exponential,
        "log_polar_img": log_polar_img,
        "u_grid": u_grid,
        "th_grid": th_grid,
        "m_modes": m_modes,
        "ccw_power": ccw_power / float(np.max(ccw_power)),
        "cw_power": cw_power / float(np.max(cw_power)),
        "dominant_ccw_m": m_ccw_dom,
        "dominant_cw_m": m_cw_dom,
        "observed_parastichy_pair": parastichy_pair,
        "peak_omega": spec["peak_omega"],
        "lattice_divergence_deg": inv["divergence_deg"],
        "lattice_G": inv["G"],
        "lattice_chirality": inv["chirality"],
        "lattice_n_scales_fft": n_scales_fft,
        "lattice_closure_residual_bins": closure_residual_bins,
        "delta_u": delta_u,
    }


# ============================================================================
# Section 6: 16-Panel Comparison Figure & Animated Growth GIF Rendering
# ============================================================================

def _draw_delaunay_parastichy_edges(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    radii: np.ndarray,
    m1: int,
    m2: int,
    x_disp: np.ndarray | None = None,
    y_disp: np.ndarray | None = None,
) -> None:
    """Overlay Delaunay contact edges colored by parastichy family index difference |i - j|.

    The triangulation is computed on the TRUE model geometry (x, y, radii); when display
    coordinates (x_disp, y_disp) are given (log-radial projection rho = 1 + ln r), the same
    index pairs are drawn there, so the parastichy families shown are those of the model.
    """
    if len(x) < 12:
        return
    pts = np.column_stack([x, y])
    tri = Delaunay(pts)
    disp = np.column_stack([x_disp, y_disp]) if x_disp is not None else pts
    r_hi = float(np.percentile(radii, 92))
    segs_m1: List[np.ndarray] = []
    segs_m2: List[np.ndarray] = []
    for simp in tri.simplices:
        for a, b in ((simp[0], simp[1]), (simp[1], simp[2]), (simp[0], simp[2])):
            i, j = (a, b) if a < b else (b, a)
            if radii[j] > r_hi:
                continue
            dk = int(j - i)
            if dk == m1:
                segs_m1.append(disp[[i, j]])
            elif dk == m2:
                segs_m2.append(disp[[i, j]])
    if segs_m1:
        ax.add_collection(LineCollection(segs_m1, colors="#38bdf8", linewidths=1.1, alpha=0.8, label=f"Δk={m1}"))
    if segs_m2:
        ax.add_collection(LineCollection(segs_m2, colors="#f472b6", linewidths=1.1, alpha=0.8, label=f"Δk={m2}"))


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#131b29")
    for sp in ax.spines.values():
        sp.set_color("#334155")
    ax.tick_params(colors="#cbd5e1", labelsize=8)


def render_figures(skip_gif: bool = False) -> Dict[str, Any]:
    """Render the 16-panel Pass 3 synthesis figure (`magnetic_drop_phyllotaxis_comparison.png`)
    and animated multi-panel growth GIF (`magnetic_drop_phyllotaxis_growth.gif`).
    """
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    pf = compute_pitchfork_bifurcation_thresholds()
    eq_diag = compute_equilibrium_bifurcation_diagram()
    trajs = simulate_adiabatic_branch_trajectories()
    gamma_sweep = run_gamma_universality_sweep(0.06)
    mem_sweep = run_memory_depth_sweep(0.06)
    exp_cmp = compare_ferrofluid_experimental_regimes()
    emergent = measure_emergent_pitchfork_threshold()
    cone = segment_conifer_cone_specimen()
    lattice_val = validate_lattice_inversion_on_simulation(0.06)

    sim_dist = simulate_fixed_G_dish(0.92, n_drops=45, branch="fibonacci")
    sim_23 = simulate_fixed_G_dish(0.30, n_drops=75, branch="fibonacci")
    sim_58 = simulate_fixed_G_dish(0.06, n_drops=96, branch="fibonacci")
    sim_lucas = simulate_fixed_G_dish(0.06, n_drops=96, branch="lucas")
    sim_ablation_mem1 = simulate_fixed_G_dish(0.06, n_drops=60, n_mem=1, branch="fibonacci")
    sim_random = simulate_fixed_G_dish(0.06, n_drops=96, gamma=0.0, branch="fibonacci")

    fig, axes = plt.subplots(4, 4, figsize=(18.5, 17.2), dpi=140)
    fig.patch.set_facecolor("#0b0f17")
    for ax in axes.flat:
        _style_axes(ax)
    LEG = dict(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    # Panel A1 (0,0): Imperfect Pitchfork Bifurcation Diagram (Fibonacci + Lucas + Saddles)
    ax = axes[0, 0]
    ax.scatter(eq_diag["unstable_G"], eq_diag["unstable_dphi"], s=5, color="#64748b", alpha=0.45, label="Unstable saddle F=0")
    ax.scatter(eq_diag["stable_G"], eq_diag["stable_dphi"], s=8, color="#38bdf8", alpha=0.65, label="Stable branch F=0")
    ax.plot(trajs["G_fib_dense"], trajs["fib_dphi_dense"], color="#22d3ee", lw=2.0, label="Adiabatic Fib (1,1)→(8,13), unclamped")
    ax.plot(trajs["lucas_bench_G"], trajs["lucas_sim_deg"], "o-", color="#f472b6", lw=1.8, ms=4.5, label="Lucas branch (1,3)→(7,11)")
    ax.scatter(trajs["fib_bench_G"], trajs["fib_ref_deg"], marker="s", s=28, color="#facc15", edgecolors="#f8fafc", zorder=6, label="Benchmark table (10 pts)")
    ax.axhline(GOLDEN_ANGLE_DEG, color="#facc15", ls="--", lw=1.1, label=f"Golden {GOLDEN_ANGLE_DEG:.2f}°")
    ax.axhline(LUCAS_ANGLE_DEG, color="#fb7185", ls=":", lw=1.1, label=f"Lucas {LUCAS_ANGLE_DEG:.2f}°")
    ax.axvline(pf["dynamical_pitchfork_exp_full"], color="#a855f7", ls="-.", lw=1.1, label=f"G_c1={pf['dynamical_pitchfork_exp_full']:.3f}")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 0.024)
    ax.set_ylim(75, 185)
    ax.set_xlabel("Plastochron ratio G = VT/R₀ (log reversed)", color="#e2e8f0", fontsize=8.5)
    ax.set_ylabel("Divergence angle Δφ (deg)", color="#e2e8f0", fontsize=8.5)
    ax.set_title(f"A1. Imperfect Pitchfork Cascade (r={trajs['pearson_r_fib']:.4f}, pairs {trajs['fib_pair_matches']}/10)", color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=6.3, loc="lower left", **LEG)

    # Panel A2 (0,1): Opposition (1,1) -> (1,2) Pitchfork Curvature & Lean 4 Verification
    ax = axes[0, 1]
    G_scan_pf = np.linspace(0.45, 1.25, 160)
    dyn_curve_exp = [
        math.exp(g) / ((math.exp(g) + 1.0) ** 5) - 2.0 * math.exp(2.0 * g) / ((math.exp(2.0 * g) - 1.0) ** 5)
        for g in G_scan_pf
    ]
    fold_curve_exp = [
        math.exp(g) / ((math.exp(g) + 1.0) ** 5) - math.exp(2.0 * g) / ((math.exp(2.0 * g) - 1.0) ** 5)
        for g in G_scan_pf
    ]
    ax.plot(G_scan_pf, np.array(dyn_curve_exp) * 1e3, color="#38bdf8", lw=2.0, label="Dynamical pitchfork (λ=1), 2-drop")
    ax.plot(G_scan_pf, np.array(fold_curve_exp) * 1e3, color="#f472b6", lw=1.7, ls="--", label="Static Hessian fold, 2-drop")
    ax.axhline(0.0, color="#94a3b8", lw=1.0)
    ax.axvline(pf["dynamical_pitchfork_exp_2drop"], color="#38bdf8", ls=":", lw=1.2)
    ax.axvline(pf["static_fold_exp_2drop"], color="#f472b6", ls=":", lw=1.2)
    ax.axvline(emergent["G_c1_emergent"], color="#22c55e", ls="-", lw=1.4, alpha=0.9, label=f"Emergent G_c1={emergent['G_c1_emergent']:.4f}")
    ax.text(
        0.36,
        0.05,
        f"G_c1(exp, full) = {pf['dynamical_pitchfork_exp_full']:.4f}\n"
        f"G_fold(exp, full) = {pf['static_fold_exp_full']:.4f}\n"
        f"Lean 4 Thm 2: C(4,7)=+{pf['integer_discriminant_G3_r4_r7']}, C(3,5)={pf['integer_discriminant_G2_r3_r5']}\n"
        f"Lean 4 Thm 4: D(4,7)={pf['integer_dynamical_discriminant_G3_r4_r7']}, D(5,9)=+{pf['integer_dynamical_discriminant_G4_r5_r9']}",
        transform=ax.transAxes,
        color="#e2e8f0",
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", edgecolor="#475569"),
    )
    ax.set_ylim(-15, 5)
    ax.set_xlabel("Plastochron ratio G = VT/R₀", color="#e2e8f0", fontsize=8.5)
    ax.set_ylabel("Scaled discriminant (×10³)", color="#e2e8f0", fontsize=8.5)
    ax.set_title("A2. Static Fold vs Dynamical Pitchfork (Lean 4 Thms 2, 4)", color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=6.8, loc="upper left", **LEG)

    # Panel A3 (0,2): Rim Repulsive Energy Landscape E(phi) across Regimes
    ax = axes[0, 2]
    phi_deg_grid = np.linspace(0.0, 360.0, 360)
    phi_rad_grid = np.radians(phi_deg_grid)
    for sim_r, col, lbl in [
        (sim_dist, "#a855f7", "G=0.92 (1,1) Distichous"),
        (sim_58, "#38bdf8", "G=0.06 (5,8) Fibonacci"),
        (sim_lucas, "#f472b6", "G=0.06 (4,7) Lucas"),
    ]:
        prev_r = sim_r["phis"][-40:][::-1]
        E_r = dipole_energy_grid(phi_rad_grid, prev_r, sim_r["G"], gamma=3.0)
        E_norm = (E_r - np.min(E_r)) / (np.ptp(E_r) + 1e-12)
        ax.plot(phi_deg_grid, E_norm, color=col, lw=1.7, label=lbl)
        imin = int(np.argmin(E_norm))
        ax.scatter([phi_deg_grid[imin]], [E_norm[imin]], color=col, edgecolors="#f8fafc", s=45, zorder=5)
    ax.set_xlabel("Candidate rim azimuth φ (deg)", color="#e2e8f0", fontsize=8.5)
    ax.set_ylabel("Normalized dipole energy E(φ)", color="#e2e8f0", fontsize=8.5)
    ax.set_title("A3. Rim Dipole Energy Landscape E(φ)", color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7, **LEG)

    # Panel A4 (0,3): Ablation & Exponent Universality Controls
    ax = axes[0, 3]
    ctrl_labels = ["γ=1.0\n(Coulomb)", "γ=3.0\n(Dipole)", "γ=5.0\n(Steep)", "Lucas\n(1,3) Basin", "Ablation\nN_mem=1"]
    ctrl_vals = [
        gamma_sweep[0]["mean_divergence_deg"],
        gamma_sweep[1]["mean_divergence_deg"],
        gamma_sweep[2]["mean_divergence_deg"],
        sim_lucas["mean_divergence_deg"],
        sim_ablation_mem1["mean_divergence_deg"],
    ]
    ctrl_colors = ["#22d3ee", "#38bdf8", "#60a5fa", "#f472b6", "#ef4444"]
    bars = ax.bar(ctrl_labels, ctrl_vals, color=ctrl_colors, edgecolor="#f8fafc", width=0.58)
    ax.axhline(GOLDEN_ANGLE_DEG, color="#facc15", ls="--", lw=1.2, label="Golden 137.51°")
    ax.axhline(LUCAS_ANGLE_DEG, color="#fb7185", ls=":", lw=1.2, label="Lucas 99.50°")
    for b, v in zip(bars, ctrl_vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 3.0, f"{v:.1f}°", ha="center", color="#f8fafc", fontsize=7.5)
    ax.set_ylim(0, 205)
    ax.set_ylabel("Steady divergence angle Δφ (deg)", color="#e2e8f0", fontsize=8.5)
    ax.set_title("A4. Universality (γ∈{1,3,5}) & N_mem=1 Ablation", color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", **LEG)

    # Row B: 4 Dish Morphologies in the log-radial / silicone-oil dish projection rho = 1 + kG
    dish_panels = [
        (axes[1, 0], sim_dist, "B1. Distichous (1,1) at G=0.92"),
        (axes[1, 1], sim_23, "B2. Fibonacci (2,3) at G=0.30"),
        (axes[1, 2], sim_58, "B3. Fibonacci (5,8) at G=0.06"),
        (axes[1, 3], sim_lucas, "B4. Lucas (4,7) at G=0.06"),
    ]
    for ax, sim_d, title_d in dish_panels:
        m1, m2 = sim_d["parastichy_pair"]
        if m1 > 1 or m2 > 1:
            _draw_delaunay_parastichy_edges(
                ax, sim_d["x"], sim_d["y"], sim_d["radii"], m1, m2,
                x_disp=sim_d["x_display"], y_disp=sim_d["y_display"],
            )
        ax.scatter(
            sim_d["x_display"], sim_d["y_display"],
            c=np.arange(len(sim_d["x"])), cmap="viridis", s=26, edgecolors="#f8fafc", linewidths=0.4, zorder=4,
        )
        ax.add_patch(plt.Circle((0, 0), 1.0, color="#facc15", fill=False, ls="--", lw=1.0, label="Rim R₀"))
        lim = float(np.max(sim_d["rho_display"])) * 1.08
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(
            f"{title_d}\n⟨Δφ⟩={sim_d['mean_divergence_deg']:.2f}°, pair={sim_d['parastichy_pair']} ({100 * sim_d['contact_pair_fraction']:.0f}%), Delaunay={sim_d['delaunay_pair']}",
            color="#f8fafc", fontsize=8.4, fontweight="bold",
        )
        ax.set_xlabel("ρ cos φ,  ρ = 1 + ln(r/R₀) = 1 + kG (dish geometry)", color="#cbd5e1", fontsize=7.5)
        ax.set_ylabel("ρ sin φ", color="#cbd5e1", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right", **LEG)

    # Panel C1 (2,0): Conifer Cone Specimen (`conifer_cone_specimen.jpg`) + DoG detections
    ax = axes[2, 0]
    ax.imshow(cone["rgb"])
    ax.scatter(cone["scale_x"], cone["scale_y"], s=22, facecolors="none", edgecolors="#22d3ee", linewidths=1.2,
               label=f"DoG detections (N={cone['n_scales_total']})")
    ax.scatter([cone["cx"]], [cone["cy"]], color="#facc15", marker="+", s=90, linewidths=2.0, label="Apex (316,307)")
    ax.add_patch(plt.Circle((cone["cx"], cone["cy"]), LOG_POLAR_U_RANGE_PX[0], color="#f472b6", fill=False, ls="--", lw=1.0))
    ax.add_patch(plt.Circle((cone["cx"], cone["cy"]), LOG_POLAR_U_RANGE_PX[1], color="#f472b6", fill=False, ls="--", lw=1.0))
    ax.set_xlim(0, cone["rgb"].shape[1])
    ax.set_ylim(cone["rgb"].shape[0], 0)
    ax.set_title("C1. Pinus sp. Cone Specimen (CC BY-SA 4.0)", color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right", **LEG)

    # Panel C2 (2,1): Unwrapped Log-Polar lattice with the FITTED (alpha_emp, G_emp) lattice overlay
    ax = axes[2, 1]
    ax.imshow(
        cone["log_polar_img"], aspect="auto", cmap="magma",
        extent=[0.0, 360.0, float(cone["u_grid"][-1]), float(cone["u_grid"][0])],
    )
    # Overlay the parastichy lines of the inverted lattice: along the m-parastichy, consecutive
    # scales differ by m plastochrons (Delta u = m G) and by m*alpha - 2 pi j (Delta theta)
    a_emp = math.radians(cone["lattice_divergence_deg"]) * (-cone["lattice_chirality"])
    g_emp = cone["lattice_G"]
    u0, u1 = float(cone["u_grid"][0]), float(cone["u_grid"][-1])
    for m_par, col in ((cone["observed_parastichy_pair"][0], "#f472b6"), (cone["observed_parastichy_pair"][1], "#38bdf8")):
        dth = (m_par * a_emp + math.pi) % (2.0 * math.pi) - math.pi  # wrapped azimuth step per parastichy step
        du = m_par * g_emp
        slope = du / dth if abs(dth) > 1e-9 else 0.0  # u per radian along the parastichy
        for k_off in range(m_par):
            th0 = 2.0 * math.pi * k_off / m_par
            th_line = np.linspace(-2.0 * math.pi, 4.0 * math.pi, 400)
            u_line = u0 + slope * (th_line - th0)
            ok = (u_line >= u0) & (u_line <= u1)
            ax.plot(np.degrees(th_line[ok]) % 360.0, u_line[ok], ".", color=col, ms=1.2, alpha=0.55)
    ax.set_ylim(u1, u0)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Azimuth θ (deg)", color="#e2e8f0", fontsize=8.5)
    ax.set_ylabel("Log-radius u = ln(r / px)", color="#e2e8f0", fontsize=8.5)
    ax.set_title(
        f"C2. Log-Polar Lattice Inversion: α={cone['lattice_divergence_deg']:.1f}°, G={g_emp:.3f}, N≈{cone['lattice_n_scales_fft']:.0f}",
        color="#f8fafc", fontsize=9.0, fontweight="bold",
    )

    # Panel C3 (2,2): Log-Polar 2D FFT Opposing Spiral Spectrum (m_CW=5, m_CCW=8)
    ax = axes[2, 2]
    m_arr = cone["m_modes"]
    ax.bar(m_arr - 0.18, cone["ccw_power"], width=0.35, color="#38bdf8", label=f"ω>0 spirals (peak m={cone['dominant_ccw_m']})")
    ax.bar(m_arr + 0.18, cone["cw_power"], width=0.35, color="#f472b6", label=f"ω<0 spirals (peak m={cone['dominant_cw_m']})")
    ax.set_xticks(m_arr)
    ax.set_xlabel("Azimuthal parastichy harmonic m", color="#e2e8f0", fontsize=8.5)
    ax.set_ylabel("Normalized 2D FFT spiral power", color="#e2e8f0", fontsize=8.5)
    ax.set_title(
        f"C3. Cone 2D FFT: Pair {cone['observed_parastichy_pair']}, closure residual {cone['lattice_closure_residual_bins']:.1f} bins",
        color="#f8fafc", fontsize=9.0, fontweight="bold",
    )
    ax.legend(fontsize=7, **LEG)

    # Panel C4 (2,3): Voronoi Cell Area Regularity (Dipole vs Random Control KS Test)
    ax = axes[2, 3]
    ks_stat, ks_p = ks_2samp(sim_58["voronoi_norm_areas"], sim_random["voronoi_norm_areas"])
    bins = np.linspace(0.1, 2.2, 18)
    ax.hist(sim_58["voronoi_norm_areas"], bins=bins, density=True, alpha=0.75, color="#38bdf8", edgecolor="#f8fafc",
            label=f"Dipole γ=3 (CV={sim_58['voronoi_area_cv']:.3f})")
    ax.hist(sim_random["voronoi_norm_areas"], bins=bins, density=True, alpha=0.55, color="#ef4444", edgecolor="#f8fafc",
            label=f"Random γ=0 (CV={sim_random['voronoi_area_cv']:.3f})")
    ax.set_xlabel("Normalized Voronoi cell area A / ⟨A⟩", color="#e2e8f0", fontsize=8.5)
    ax.set_ylabel("Probability density", color="#e2e8f0", fontsize=8.5)
    ax.set_title(f"C4. Packing Regularity (KS D={ks_stat:.2f}, p={ks_p:.1e})", color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7, **LEG)

    # Panel D1 (3,0): Emergent pitchfork of the iterated map (no clamp) vs analytical thresholds
    ax = axes[3, 0]
    ax.plot(emergent["scan_G"], emergent["scan_dphi_deg"], "o-", color="#22d3ee", ms=4, lw=1.5, label="Iterated map, steady Δφ (unclamped)")
    ax.axvline(emergent["G_c1_analytic_full"], color="#a855f7", ls="-.", lw=1.3, label=f"Analytic G_c1 (full chain) = {emergent['G_c1_analytic_full']:.4f}")
    ax.axvline(emergent["G_c1_emergent"], color="#22c55e", ls="-", lw=1.3, label=f"Emergent G_c1 (bisection) = {emergent['G_c1_emergent']:.4f}")
    ax.axvline(pf["static_fold_exp_full"], color="#f472b6", ls=":", lw=1.3, label=f"Static fold G_fold = {pf['static_fold_exp_full']:.4f}")
    ax.axhline(180.0, color="#94a3b8", lw=0.9)
    ax.set_xlabel("Plastochron ratio G", color="#e2e8f0", fontsize=8.5)
    ax.set_ylabel("Steady divergence Δφ (deg)", color="#e2e8f0", fontsize=8.5)
    ax.set_title(f"D1. Emergent (1,1)→(1,2) Pitchfork: |ΔG_c1| = {emergent['abs_error']:.1e}", color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=6.8, loc="lower right", **LEG)

    # Panel D2 (3,1): Memory-depth sweep N_mem
    ax = axes[3, 1]
    nm = [r["n_mem"] for r in mem_sweep["rows"]]
    dv = [r["mean_divergence_deg"] for r in mem_sweep["rows"]]
    ax.plot(nm, dv, "o-", color="#38bdf8", ms=5, lw=1.6, label="⟨Δφ⟩(N_mem) at G=0.06")
    ax.axhline(GOLDEN_ANGLE_DEG, color="#facc15", ls="--", lw=1.1, label="Golden 137.51°")
    ax.axhline(mem_sweep["reference_divergence_deg"], color="#22c55e", ls=":", lw=1.1, label=f"N_mem=45 ref {mem_sweep['reference_divergence_deg']:.2f}°")
    ax.axvline(mem_sweep["n_mem_converged"], color="#f472b6", ls="-.", lw=1.1, label=f"Converged (<0.5°) at N_mem={mem_sweep['n_mem_converged']}")
    for r in mem_sweep["rows"]:
        ax.annotate(f"{r['parastichy_pair']}", (r["n_mem"], r["mean_divergence_deg"]), textcoords="offset points", xytext=(4, 6), color="#cbd5e1", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xticks(nm)
    ax.set_xticklabels([str(v) for v in nm])
    ax.set_xlabel("Interaction memory depth N_mem (drops)", color="#e2e8f0", fontsize=8.5)
    ax.set_ylabel("Steady divergence Δφ (deg)", color="#e2e8f0", fontsize=8.5)
    ax.set_title("D2. Memory-Depth Sweep: 180° → (1,2) → Golden Angle", color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=6.8, loc="upper right", **LEG)

    # Panel D3 (3,2): Ferrofluid experimental regimes (PRL 1992 Figs. 1-2 table) vs simulation
    ax = axes[3, 2]
    rows = exp_cmp["rows"]
    xs_ = np.arange(len(rows))
    obs = [r["observed_divergence_deg"] for r in rows]
    simv = [r["simulated_divergence_deg"] for r in rows]
    ax.bar(xs_ - 0.18, obs, width=0.36, color="#facc15", edgecolor="#f8fafc", label="Observed (dish, PRL 1992)")
    ax.bar(xs_ + 0.18, simv, width=0.36, color="#38bdf8", edgecolor="#f8fafc", label="Simulated (same G)")
    for xi, r in zip(xs_, rows):
        tag = "✓" if r["pair_match"] else "✗"
        ax.text(xi, max(r["observed_divergence_deg"], r["simulated_divergence_deg"]) + 4,
                f"{r['observed_pair']}→{r['simulated_pair']} {tag}", ha="center", color="#f8fafc", fontsize=6.3)
    ax.set_xticks(xs_)
    ax.set_xticklabels([f"G={r['G_exp']:.2f}\n{r['label'].split(' (')[0]}" for r in rows], fontsize=6.3)
    ax.set_ylim(0, 215)
    ax.set_ylabel("Mean divergence Δφ (deg)", color="#e2e8f0", fontsize=8.5)
    ax.set_title(
        f"D3. Ferrofluid Regimes: RMSE={exp_cmp['rmse_divergence_deg']:.2f}°, pairs {exp_cmp['n_pair_match']}/{exp_cmp['n_regimes']}",
        color="#f8fafc", fontsize=9.5, fontweight="bold",
    )
    ax.legend(fontsize=6.8, loc="upper right", **LEG)

    # Panel D4 (3,3): Specimen lattice vs model lattice in reciprocal/parameter space
    ax = axes[3, 3]
    fib_G = trajs["fib_bench_G"]
    fib_sim = trajs["fib_sim_deg"]
    ax.plot(fib_G, fib_sim, "o-", color="#22d3ee", ms=4, lw=1.4, label="Model Fibonacci branch (sim)")
    ax.scatter(trajs["fib_bench_G"], trajs["fib_ref_deg"], marker="s", s=30, color="#facc15", edgecolors="#f8fafc", zorder=5, label="Benchmark table")
    ax.errorbar([cone["lattice_G"]], [cone["lattice_divergence_deg"]], xerr=[[0.12 * cone["lattice_G"]], [0.12 * cone["lattice_G"]]],
                yerr=[[0.7], [0.7]], fmt="*", ms=14, color="#f472b6", ecolor="#f472b6", capsize=3, zorder=7,
                label=f"Cone specimen (FFT inversion): α={cone['lattice_divergence_deg']:.1f}°, G={cone['lattice_G']:.3f}")
    ax.scatter([lattice_val["recovered_G"]], [lattice_val["recovered_divergence_deg"]], marker="D", s=50, color="#22c55e", edgecolors="#f8fafc", zorder=7,
               label=f"Inversion check on sim G=0.06: {lattice_val['recovered_divergence_deg']:.2f}°, G={lattice_val['recovered_G']:.4f}")
    ax.axhline(GOLDEN_ANGLE_DEG, color="#facc15", ls="--", lw=1.0)
    ax.axhspan(135.0, 360.0 * 5.0 / 13.0, color="#38bdf8", alpha=0.12, label="(5,8) window 135°–138.46° (Lean 4 Thm 5)")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 0.02)
    ax.set_ylim(132, 150)
    ax.set_xlabel("Plastochron ratio G (log reversed)", color="#e2e8f0", fontsize=8.5)
    ax.set_ylabel("Divergence angle α (deg)", color="#e2e8f0", fontsize=8.5)
    ax.set_title("D4. Specimen vs Model in (G, α) Space", color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=6.3, loc="upper right", **LEG)

    fig.suptitle(
        "Douady & Couder (1992, 1996) Self-Organized Magnetic-Drop Phyllotaxis — Emergent Bifurcation, Lean 4 Proofs, Dish Geometry & Conifer Cone Lattice Inversion (Pass 3)",
        color="#f8fafc",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)

    if not skip_gif:
        fig_g, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.2, 4.8), dpi=100)
        fig_g.patch.set_facecolor("#0b0f17")
        n_total = len(sim_58["phis"])

        def update(frame_idx: int) -> list:
            ax_l.clear()
            ax_r.clear()
            ax_l.set_facecolor("#131b29")
            ax_r.set_facecolor("#131b29")
            n_cur = min(10 + frame_idx * 5, n_total)
            phis_c = sim_58["phis"][:n_cur]
            ages_c = np.arange(len(phis_c) - 1, -1, -1, dtype=float)
            rho_c = 1.0 + ages_c * sim_58["G"]  # dish geometry rho = 1 + kG (= 1 + ln r/R0)
            ax_l.scatter(rho_c * np.cos(phis_c), rho_c * np.sin(phis_c), c=ages_c, cmap="viridis", s=36, edgecolors="#f8fafc", linewidths=0.4)
            ax_l.add_patch(plt.Circle((0, 0), 1.0, color="#facc15", fill=False, ls="--", lw=1.0))
            lim = 1.0 + n_total * sim_58["G"] * 1.05
            ax_l.set_xlim(-lim, lim)
            ax_l.set_ylim(-lim, lim)
            ax_l.set_aspect("equal")
            ax_l.set_title(f"Magnetic-Drop Dish ρ=1+kG (G=0.06, N={n_cur}, Pair={sim_58['parastichy_pair']})", color="#f8fafc", fontsize=9.5)
            ax_l.tick_params(colors="#cbd5e1")

            dphis_c = sim_58["dphi_deg_series"][: max(2, n_cur - 1)]
            ax_r.plot(np.arange(1, len(dphis_c) + 1), dphis_c, "o-", color="#38bdf8", ms=3.5, lw=1.4, label="Iterative Δφ_n")
            ax_r.axhline(GOLDEN_ANGLE_DEG, color="#facc15", ls="--", lw=1.3, label=f"Golden angle {GOLDEN_ANGLE_DEG:.2f}°")
            ax_r.set_xlim(1, len(sim_58["dphi_deg_series"]))
            ax_r.set_ylim(130, 145)
            ax_r.set_xlabel("Drop index n", color="#e2e8f0")
            ax_r.set_ylabel("Divergence angle Δφ_n (deg)", color="#e2e8f0")
            ax_r.set_title("Self-Organized Convergence to 137.51°", color="#f8fafc", fontsize=10)
            ax_r.tick_params(colors="#cbd5e1")
            ax_r.legend(fontsize=8, **LEG)
            return []

        anim = FuncAnimation(fig_g, update, frames=18, interval=110, blit=False)
        anim.save(OUT_GIF, writer=PillowWriter(fps=8))
        plt.close(fig_g)

    return {
        "pitchfork": pf,
        "trajectories": trajs,
        "gamma_sweep": gamma_sweep,
        "memory_sweep": mem_sweep,
        "experimental_regimes": exp_cmp,
        "emergent_pitchfork": emergent,
        "lattice_validation": lattice_val,
        "cone": cone,
        "sim_58": sim_58,
        "sim_lucas": sim_lucas,
        "sim_ablation_mem1": sim_ablation_mem1,
        "sim_random": sim_random,
    }


# ============================================================================
# Section 7: MorphoEval Quantitative Measurement Suite (`measure_morphoeval`)
# ============================================================================

def measure_morphoeval() -> List[Dict[str, Any]]:
    """Compute the Pass 3 morphometric measurement suite for `measure_outputs.py`.

    Every entry carries a `reference_value` with its provenance in `method`. Pass 3 renames the
    two v2 entries whose names misdescribed what they measured (`pitchfork_threshold_G_c1_*`
    reported the static fold G_fold; `lucas_branch_divergence_angle_G006` was evaluated at
    G = 0.11 against the asymptote) and adds the emergent-threshold, dish-experiment,
    memory-depth and specimen-lattice measurements.
    """
    pf = compute_pitchfork_bifurcation_thresholds()
    trajs = simulate_adiabatic_branch_trajectories()
    emergent = measure_emergent_pitchfork_threshold()
    mem_sweep = run_memory_depth_sweep(0.06)
    exp_cmp = compare_ferrofluid_experimental_regimes()
    lattice_val = validate_lattice_inversion_on_simulation(0.06)
    sim_58 = simulate_fixed_G_dish(0.06, n_drops=96, branch="fibonacci")
    sim_58_lin = simulate_fixed_G_dish(0.06, n_drops=96, advection="linear", branch="fibonacci")
    sim_lucas_010 = simulate_fixed_G_dish(0.10, n_drops=96, branch="lucas")
    sim_ablation = simulate_fixed_G_dish(0.06, n_drops=60, n_mem=1, branch="fibonacci")
    sim_random = simulate_fixed_G_dish(0.06, n_drops=96, gamma=0.0, branch="fibonacci")
    cone = segment_conifer_cone_specimen()
    ks_stat, ks_p = ks_2samp(sim_58["voronoi_norm_areas"], sim_random["voronoi_norm_areas"])
    ref = load_reference_dataset()
    lucas_ref_010 = next(
        float(r["divergence_deg"])
        for r in ref["bifurcation_branches_exponential_gamma3"]["lucas_secondary_branch"]
        if abs(float(r["G"]) - 0.10) < 1e-9
    )

    return [
        {
            "name": "fibonacci_divergence_angle_G006_exp",
            "value": float(round(sim_58["mean_divergence_deg"], 4)),
            "unit": "deg",
            "reference_value": float(round(GOLDEN_ANGLE_DEG, 4)),
            "method": f"Iterative dipole minimization (gamma=3, exponential advection, G=0.06, SD={sim_58['std_divergence_deg']:.4f} deg); reference = golden angle 360/tau^2",
        },
        {
            "name": "fibonacci_divergence_angle_G006_linear_dish",
            "value": float(round(sim_58_lin["mean_divergence_deg"], 4)),
            "unit": "deg",
            "reference_value": float(round(GOLDEN_ANGLE_DEG, 4)),
            "method": f"Linear silicone-oil dish advection r_k = R_0(1+k*G) at G=0.06 (SD={sim_58_lin['std_divergence_deg']:.4f} deg)",
        },
        {
            "name": "lucas_branch_divergence_angle_G010",
            "value": float(round(sim_lucas_010["mean_divergence_deg"], 4)),
            "unit": "deg",
            "reference_value": float(round(lucas_ref_010, 4)),
            "method": f"Disconnected Lucas branch at benchmark G=0.10 (SD={sim_lucas_010['std_divergence_deg']:.4f} deg, Delaunay={sim_lucas_010['parastichy_pair']}); reference = lucas_secondary_branch row G=0.10 of magnetic_drop_phyllotaxis_reference.json (asymptote 99.5016 deg)",
        },
        {
            "name": "static_fold_threshold_G_fold_exponential_2drop",
            "value": float(round(pf["static_fold_exp_2drop"], 6)),
            "unit": "dimensionless",
            "reference_value": 0.773484,
            "method": "Exact 2-drop static Hessian fold root r_1/(r_1+1)^5 = r_2/(r_2-1)^5 (v2 misnamed this G_c1); reference = analytical_benchmarks.two_drop_exponential_G_c1 in the reference JSON",
        },
        {
            "name": "static_fold_threshold_G_fold_linear_2drop",
            "value": float(round(pf["static_fold_lin_2drop"], 6)),
            "unit": "dimensionless",
            "reference_value": 2.51456,
            "method": "Exact linear-advection 2-drop fold root (1+G)/(2+G)^5 = (1+2G)/(2G)^5 bracketed by Lean 4 Theorem 2 (C(3,5) < 0 < C(4,7))",
        },
        {
            "name": "dynamical_pitchfork_G_c1_exponential_full_chain",
            "value": float(round(pf["dynamical_pitchfork_exp_full"], 6)),
            "unit": "dimensionless",
            "reference_value": float(round(emergent["G_c1_emergent"], 6)),
            "method": f"Root of S_dyn(G) = sum_k (-1)^(k-1) k r_k/(r_k-(-1)^k)^5 (M=60) vs the EMERGENT threshold of the unclamped iterated map (bisection bracket [{emergent['bracket'][0]:.5f}, {emergent['bracket'][1]:.5f}], |error|={emergent['abs_error']:.2e}); linear-advection 2-drop analogue bracketed by Lean 4 Theorem 4 (D(4,7) < 0 < D(5,9))",
        },
        {
            "name": "fibonacci_bifurcation_branch_pearson_r",
            "value": float(round(trajs["pearson_r_fib"], 5)),
            "unit": "r",
            "reference_value": 1.0,
            "method": f"Pearson correlation across 10 benchmark G points with no 180-deg clamp (RMSE={trajs['rmse_fib_deg']:.4f} deg, p={trajs['pearson_p_fib']:.2e}); Delaunay pairs on true radii match the table at {trajs['fib_pair_matches']}/10 points",
        },
        {
            "name": "fibonacci_branch_parastichy_pair_matches_of_10",
            "value": int(trajs["fib_pair_matches"]),
            "unit": "count",
            "reference_value": 10,
            "method": f"Delaunay contact pair on the true exponential radii r_k = exp(kG) vs the benchmark table pairs; sim={trajs['fib_sim_pairs']} ref={trajs['fib_ref_pairs']} (mismatches sit at the tabulated transition rows)",
        },
        {
            "name": "ferrofluid_experiment_divergence_rmse_deg",
            "value": float(round(exp_cmp["rmse_divergence_deg"], 4)),
            "unit": "deg",
            "reference_value": 0.0,
            "method": f"RMSE of simulated vs observed mean divergence over the {exp_cmp['n_regimes']} experimental dish regimes tabulated from Douady & Couder (1992) Figs. 1-2 (max |err|={exp_cmp['max_abs_divergence_err_deg']:.3f} deg; pairs matched {exp_cmp['n_pair_match']}/{exp_cmp['n_regimes']})",
        },
        {
            "name": "memory_ablation_control_Nmem1_divergence_deg",
            "value": float(round(sim_ablation["mean_divergence_deg"], 4)),
            "unit": "deg",
            "reference_value": 180.0,
            "method": f"Single-drop memory ablation control (N_mem=1 at G=0.06) vs N_mem=45 baseline ({sim_58['mean_divergence_deg']:.2f} deg); Lean 4 Theorem 1",
        },
        {
            "name": "memory_depth_Nmem_for_convergence_within_0p5deg",
            "value": int(mem_sweep["n_mem_converged"]),
            "unit": "drops",
            "reference_value": 8,
            "method": "Smallest N_mem in {1,2,3,5,8,13,21,45} whose steady divergence at G=0.06 is within 0.5 deg of the N_mem=45 value; reference = e^{-3 k G} < 0.25 for k >= 8 (dipole tail negligible); sweep: "
            + ", ".join(f"N{r['n_mem']}={r['mean_divergence_deg']:.1f}" for r in mem_sweep["rows"]),
        },
        {
            "name": "voronoi_area_cv_dipole_vs_random_control",
            "value": float(round(sim_58["voronoi_area_cv"], 4)),
            "unit": "CV",
            "reference_value": 0.0,
            "method": f"Radially normalised Voronoi cell-area CV of the G=0.06 lattice; reference 0 = perfectly self-similar regular lattice (random gamma=0 control CV={sim_random['voronoi_area_cv']:.4f}, two-sample KS D={ks_stat:.3f}, p={ks_p:.2e})",
        },
        {
            "name": "conifer_cone_specimen_ccw_parastichy_count",
            "value": int(cone["dominant_ccw_m"]),
            "unit": "spirals",
            "reference_value": 8,
            "method": f"Log-polar 2D FFT dominant omega>0 spiral mode on Pinus sp. specimen (N={cone['n_scales_total']} DoG detections; FFT-consistent scale count in annulus = {cone['lattice_n_scales_fft']:.0f})",
        },
        {
            "name": "conifer_cone_specimen_cw_parastichy_count",
            "value": int(cone["dominant_cw_m"]),
            "unit": "spirals",
            "reference_value": 5,
            "method": "Log-polar 2D FFT dominant omega<0 spiral mode on Pinus sp. specimen confirming Fibonacci (5, 8) pair",
        },
        {
            "name": "conifer_cone_lattice_divergence_angle_deg",
            "value": float(round(cone["lattice_divergence_deg"], 3)),
            "unit": "deg",
            "reference_value": float(round(GOLDEN_ANGLE_DEG, 4)),
            "method": f"Reciprocal-lattice inversion of the sub-bin (5,8) log-polar FFT peaks omega_5={cone['peak_omega'][5]:.2f}, omega_8={cone['peak_omega'][8]:.2f} rad/u (5a+w5 G=4pi, 8a+w8 G=6pi); inversion recovers the simulated G=0.06 lattice to {lattice_val['divergence_abs_err_deg']:.3f} deg; Lean 4 Theorem 5 window 135-138.46 deg",
        },
        {
            "name": "conifer_cone_lattice_plastochron_ratio_G",
            "value": float(round(cone["lattice_G"], 4)),
            "unit": "dimensionless",
            "reference_value": 0.06,
            "method": f"Plastochron ratio of the specimen from the same (5,8) inversion (chirality {int(cone['lattice_chirality']):+d}); reference = model (5,8) regime G=0.06 (table spans 0.04-0.10 for (5,8)); rank-count estimate Delta_u/N_det = {cone['G_empirical_annulus']:.4f} is biased low by DoG over-detection; inversion recovers sim G=0.06 to {100 * lattice_val['G_rel_err']:.2f}%",
        },
        {
            "name": "conifer_cone_radial_law_R2_linear_minus_exponential",
            "value": float(round(cone["radial_law_R2_linear"] - cone["radial_law_R2_exponential"], 4)),
            "unit": "delta_R2",
            "reference_value": 0.0,
            "method": f"R^2(linear r vs rank) - R^2(ln r vs rank) over {cone['n_scales_annulus']} annulus detections = {cone['radial_law_R2_linear']:.3f} - {cone['radial_law_R2_exponential']:.3f}; > 0 means the mature cone's apparent radial law is closer to the dish's linear advection than to exponential meristem growth (transfer-boundary diagnostic; reference 0 = indifferent)",
        },
        {
            "name": "conifer_cone_rank_sequence_resultant_R",
            "value": float(round(cone["rank_sequence_resultant_R"], 4)),
            "unit": "R",
            "reference_value": 1.0,
            "method": "Circular mean resultant length of successive azimuth differences of radius-ranked DoG detections; 1 = clean 137.5 deg generative sequence. Disclosed negative result: rank ordering of a mature cone photo does not recover the divergence sequence, so the lattice claim rests on the FFT inversion",
        },
    ]


# ============================================================================
# Section 8: Comprehensive Self-Test Suite & CLI Entrypoint
# ============================================================================

def run_self_test() -> None:
    """Execute 14 quantitative verification checks covering dataset SHA-256 locks,
    Lean 4 theorem parity (Theorems 2 & 4), pitchfork roots, the emergent (unclamped)
    bifurcation, Fibonacci/Lucas cascades on true radii, ablation and memory-depth sweeps,
    the ferrofluid regime table, specimen segmentation, lattice inversion, and repeatability.
    """
    # 1. Dataset SHA-256 verification
    digests = verify_dataset_integrity()
    assert len(digests) == 3, "Expected 3 SHA-256-locked dataset files"

    # 2. Lean 4 Theorem 2 & 4 integer discriminant parity
    pf = compute_pitchfork_bifurcation_thresholds()
    assert pf["integer_discriminant_G3_r4_r7"] == 27687, "Lean 4 Thm 2 G=3 discriminant mismatch"
    assert pf["integer_discriminant_G2_r3_r5"] == -6144, "Lean 4 Thm 2 G=2 discriminant mismatch"
    assert pf["integer_dynamical_discriminant_G3_r4_r7"] == -12646, "Lean 4 Thm 4 D(4,7) mismatch"
    assert pf["integer_dynamical_discriminant_G4_r5_r9"] == 23872, "Lean 4 Thm 4 D(5,9) mismatch"
    assert abs(pf["static_fold_exp_2drop"] - 0.77348399) < 1e-6
    assert abs(pf["static_fold_lin_2drop"] - 2.51456033) < 1e-6
    # Static/dynamic gap: fold < 3 < dynamical threshold under linear advection (Thm 4)
    assert pf["static_fold_lin_2drop"] < 3.0 < pf["dynamical_pitchfork_lin_2drop"] < 4.0

    # 3. Emergent pitchfork of the unclamped iterated map matches the analytic full-chain root
    emergent = measure_emergent_pitchfork_threshold()
    assert emergent["abs_error"] < 2e-3, f"Emergent G_c1 {emergent['G_c1_emergent']} vs analytic {emergent['G_c1_analytic_full']}"
    assert emergent["scan_dphi_deg"][-1] > 179.9 and emergent["scan_dphi_deg"][0] < 160.0

    # 4. Fibonacci & Lucas branch accuracy (no clamps) and true-radius Delaunay pairs
    trajs = simulate_adiabatic_branch_trajectories()
    assert trajs["pearson_r_fib"] > 0.999, f"Expected r > 0.999, got {trajs['pearson_r_fib']}"
    assert trajs["rmse_fib_deg"] < 0.25, f"Expected RMSE < 0.25 deg, got {trajs['rmse_fib_deg']}"
    assert trajs["rmse_lucas_deg"] < 0.25, f"Expected Lucas RMSE < 0.25 deg, got {trajs['rmse_lucas_deg']}"
    assert trajs["fib_dphi_dense"][0] > 179.9, "Unclamped ramp must find opposition at G=0.92 by itself"
    assert trajs["fib_pair_matches"] >= 9, f"Expected >= 9/10 pair matches, got {trajs['fib_pair_matches']}"

    # 5. Nearest-neighbour contact parastichy pairs on TRUE radii: (2,3) at G=0.30, (5,8) at G=0.06,
    #    (3,4) Lucas at G=0.11 and (4,7) Lucas at G=0.06 (table rows); Delaunay census agrees at G=0.06
    sim_23 = simulate_fixed_G_dish(0.30, n_drops=75, branch="fibonacci")
    sim_58 = simulate_fixed_G_dish(0.06, n_drops=96, branch="fibonacci")
    sim_lucas_011 = simulate_fixed_G_dish(0.11, n_drops=96, branch="lucas")
    sim_lucas = simulate_fixed_G_dish(0.06, n_drops=96, branch="lucas")
    assert sim_23["parastichy_pair"] == (2, 3), f"Expected (2, 3) at G=0.30 (v2 clipped geometry gave (3,5)), got {sim_23['parastichy_pair']}"
    assert sim_58["parastichy_pair"] == (5, 8), f"Expected (5, 8), got {sim_58['parastichy_pair']}"
    assert sim_58["delaunay_pair"] == (5, 8), f"Delaunay census must agree at G=0.06, got {sim_58['delaunay_pair']}"
    assert sim_lucas_011["parastichy_pair"] == (3, 4), f"Expected Lucas (3, 4) at G=0.11, got {sim_lucas_011['parastichy_pair']}"
    assert sim_lucas["parastichy_pair"] == (4, 7), f"Expected Lucas (4, 7) at G=0.06, got {sim_lucas['parastichy_pair']}"
    assert sim_58["contact_pair_fraction"] > 0.9
    assert simulate_fixed_G_dish(0.92, n_drops=45)["parastichy_pair"] == (1, 1)

    # 6. Single-drop memory ablation control (N_mem = 1) locks at 180.0 deg
    sim_ablation = simulate_fixed_G_dish(0.06, n_drops=50, n_mem=1, branch="fibonacci")
    assert abs(sim_ablation["mean_divergence_deg"] - 180.0) < 1e-3, "N_mem=1 ablation must lock at 180 deg"

    # 7. Power-law universality sweep gamma in {1, 3, 5}
    gsweep = run_gamma_universality_sweep(0.06)
    for row in gsweep:
        assert row["abs_err_from_golden_deg"] < 1.2, f"Universality failed for gamma={row['gamma']}"

    # 8. Memory-depth sweep converges and N_mem=2 is the two-drop spiral (not opposition)
    mem = run_memory_depth_sweep(0.06)
    assert mem["rows"][0]["mean_divergence_deg"] > 179.99
    assert 100.0 < mem["rows"][1]["mean_divergence_deg"] < 160.0, "N_mem=2 must leave opposition"
    assert 1 < mem["n_mem_converged"] <= 13, f"Expected convergence by N_mem=13, got {mem['n_mem_converged']}"
    assert mem["abs_err_vs_deepest_deg"][-2] < 0.1

    # 9. Ferrofluid experimental regime table (PRL 1992 Figs. 1-2)
    exp_cmp = compare_ferrofluid_experimental_regimes()
    assert exp_cmp["rmse_divergence_deg"] < 0.5, f"Dish regime RMSE too large: {exp_cmp['rmse_divergence_deg']}"
    assert exp_cmp["n_pair_match"] >= 5, f"Expected >= 5/6 pair matches, got {exp_cmp['n_pair_match']}"

    # 10. Conifer cone specimen (5, 8) log-polar 2D FFT parastichy pair
    cone = segment_conifer_cone_specimen()
    assert cone["n_scales_total"] >= 100, f"Expected >=100 detections, got {cone['n_scales_total']}"
    assert cone["observed_parastichy_pair"] == (5, 8), f"Expected (5, 8), got {cone['observed_parastichy_pair']}"

    # 11. Specimen lattice inversion lands in the Lean 4 Theorem 5 (5,8) window and near the golden angle
    assert 135.0 < cone["lattice_divergence_deg"] < 360.0 * 5.0 / 13.0, f"Lattice divergence {cone['lattice_divergence_deg']} outside (5,8) window"
    assert abs(cone["lattice_divergence_deg"] - GOLDEN_ANGLE_DEG) < 1.5
    assert 0.02 < cone["lattice_G"] < 0.10, f"Specimen G {cone['lattice_G']} outside the (5,8)-(8,13) range"

    # 12. Lattice inversion validated on the simulated lattice with known (alpha, G)
    lv = validate_lattice_inversion_on_simulation(0.06)
    assert lv["divergence_abs_err_deg"] < 0.5, f"Inversion divergence error {lv['divergence_abs_err_deg']}"
    assert lv["G_rel_err"] < 0.10, f"Inversion G error {lv['G_rel_err']}"

    # 13. Disclosed negative result: radius-ranked detections do not form a clean divergence sequence
    assert cone["rank_sequence_resultant_R"] < 0.5
    assert cone["radial_law_R2_linear"] > cone["radial_law_R2_exponential"]

    # 14. Bitwise two-run determinism check
    sim_a = simulate_fixed_G_dish(0.14, n_drops=50, seed=1992)
    sim_b = simulate_fixed_G_dish(0.14, n_drops=50, seed=1992)
    assert np.allclose(sim_a["phis"], sim_b["phis"], atol=0.0), "Two-run determinism check failed"

    print(
        "[1992-douady-magnetic-drop-phyllotaxis v3] All 14 self-tests PASSED "
        f"(Fib r={trajs['pearson_r_fib']:.5f}, RMSE={trajs['rmse_fib_deg']:.3f}°, pairs {trajs['fib_pair_matches']}/10, "
        f"emergent G_c1={emergent['G_c1_emergent']:.4f} vs {emergent['G_c1_analytic_full']:.4f}, "
        f"Delaunay={sim_58['parastichy_pair']}, Cone FFT={cone['observed_parastichy_pair']}, "
        f"cone α={cone['lattice_divergence_deg']:.2f}°, G={cone['lattice_G']:.4f}, dish RMSE={exp_cmp['rmse_divergence_deg']:.3f}°)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Douady & Couder (1992, 1996) magnetic-drop phyllotaxis & bifurcation cascade (v3, Pass 3)"
    )
    ap.add_argument("--self-test", action="store_true", help="Run 14-check verification suite and exit")
    ap.add_argument("--skip-gif", action="store_true", help="Skip animated GIF generation")
    ap.add_argument("--G", type=float, default=0.06, help="Plastochron ratio G = V*T/R_0 for custom dish run")
    ap.add_argument("--gamma", type=float, default=3.0, help="Repulsive power-law exponent gamma (default: 3.0)")
    ap.add_argument("--n-mem", type=int, default=45, help="Interaction memory depth N_mem (default: 45)")
    ap.add_argument(
        "--advection",
        choices=["exponential", "linear"],
        default="exponential",
        help="Radial advection law (exponential SAM vs linear silicone-oil dish)",
    )
    ap.add_argument(
        "--branch",
        choices=["fibonacci", "lucas"],
        default="fibonacci",
        help="Phyllotactic branch basin (fibonacci vs lucas)",
    )
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    run_self_test()
    render_figures(skip_gif=args.skip_gif)
    if args.G != 0.06 or args.gamma != 3.0 or args.advection != "exponential" or args.branch != "fibonacci" or args.n_mem != 45:
        custom = simulate_fixed_G_dish(
            args.G, gamma=args.gamma, n_mem=args.n_mem, advection=args.advection, branch=args.branch
        )
        print(
            f"[Custom Run G={args.G}, gamma={args.gamma}, N_mem={args.n_mem}, {args.advection}, {args.branch}] "
            f"⟨Δφ⟩={custom['mean_divergence_deg']:.4f}°, parastichy_pair={custom['parastichy_pair']}"
        )
    print(f"[1992-douady-magnetic-drop-phyllotaxis v3] Rendered {OUT_PNG} and {OUT_GIF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
