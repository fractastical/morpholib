#!/usr/bin/env python3
r"""
1972-gierer-hydra-activator-inhibitor.py — Gierer & Meinhardt (1972) / Meinhardt (1993, 2012)
Short-Range Autocatalytic Activator / Long-Range Lateral Inhibitor Morphogenesis in Hydra vulgaris.

Mathematical & Biological Specification
=======================================
1. **Gierer & Meinhardt (1972) Kybernetik 12(1):30-39, Eqs. (15a)-(15b):**
   Short-range autocatalytic activator $a(x,t)$ and long-range lateral inhibitor $h(x,t)$
   along the apico-basal body column or sub-hypostomal tentacle ring of *Hydra vulgaris*:

       da/dt = rho_0 * rho(x) + c * rho(x) * a^2 / h - mu_a * a + D_a * d^2a/dx^2
       dh/dt = c' * rho(x) * a^2 - mu_h * h + D_h * d^2h/dx^2

   - At $\rho_0 = 0$, the exact homogeneous steady state is:
         a_* = (c / c') * (mu_h / mu_a),    h_* = (c / c') * rho * (mu_h / mu_a)^2
   - The homogeneous reaction Jacobian $J_0$ has diagonal entries $J_{11} = +\mu_a$
     (self-enhancing autocatalysis) and $J_{22} = -\mu_h$ (inhibitor decay), with:
         tr(J_0) = mu_a - mu_h < 0   (iff mu_h > mu_a, preventing homogeneous oscillations)
         det(J_0) = mu_a * mu_h > 0
   - At spatial wavenumber squared $q = k^2$, the Fourier-mode determinant is:
         det_J(q) = (D_a * q - mu_a) * (D_h * q + mu_h) + 2 * mu_a * mu_h
     which becomes negative (triggering symmetry-breaking Turing-Gierer-Meinhardt instability)
     if and only if the diffusivity ratio $d = D_h / D_a$ exceeds the exact threshold:
         d_c = (sqrt(2) + 1)^2 = 3 + 2 * sqrt(2) = 5.8284271247461903

2. **Lean 4 Formal Verification (`morphoeval/lean/loops/Loop_001_hydra_activator_inhibitor.lean`):**
   - `MathOfBiology.Loop_001.gierer_meinhardt_homogeneous_steady_state`
   - `MathOfBiology.Loop_001.gierer_meinhardt_equal_diffusion_stable`
   - `MathOfBiology.Loop_001.gierer_meinhardt_lateral_inhibition_instability`
   - `MathOfBiology.Loop_001.meinhardt_three_system_off_state_discriminant_antitone` (Pass 3)
   - `MathOfBiology.Loop_001.meinhardt_tentacle_drive_monotone` (Pass 3)
   - `MathOfBiology.Loop_001.meinhardt_head_foot_source_density_exclusion` (Pass 3)
   - `MathOfBiology.Loop_001.cylindrical_shell_laplacian_annihilates_constants` (Pass 3)

2b. **Pass 3: Coupled Three-System Model on a 3D Cylindrical Shell (Meinhardt 1993, 2012) —
   reduced, honestly labelled form.** The Pass 2 audit asked for the 1D column and 1D ring to be
   extended to a coupled 3D shell with separate head, tentacle and foot activator-inhibitor loops.
   This script implements the closest reduced version that stays within the runtime budget:

   - **Geometry:** a surface of revolution (2-manifold embedded in 3D) whose radius profile
     $r(s)$ along the apico-basal arclength $s \in [0, L]$ is *measured on the vendored specimen*
     (Euclidean distance transform sampled along the PCA body axis from the hypostome tip to the
     basal disc, `measure_specimen_body_profile`). The default body length is $L = 90$ cells,
     chosen so that the model's intrinsic tentacle spacing ($\approx 8.1$ cells, set by the
     tentacle-inhibitor range $\ell_c = \sqrt{D_c/\mu_c}$) equals the specimen's inter-tentacle
     spacing / body-length ratio ($0.092$) — this is a *scale calibration*, declared as such.
   - **Operator:** the Laplace–Beltrami operator of the surface of revolution,
         Δu = (1/r) ∂_s ( r ∂_s u ) + (1/r^2) ∂_θθ u,
     is discretised by finite differences on an $N_s \times N_\theta$ structured grid (this is a
     finite-difference shell, *not* a finite-element mesh — the FEM part of the Pass 2 request is
     carried forward as an open item). Diffusion is integrated with an unconditionally stable
     operator split: the circumferential part exactly in Fourier space (row-dependent symbol
     $-D k_n^2 / r_i^2$), the axial part by backward Euler on the tridiagonal $(1/r)\partial_s(r\partial_s)$
     operator with zero-flux ends; the reaction terms are explicit.
   - **Three coupled activator-inhibitor loops + source density** (Meinhardt 1993 Dev. Biol.
     157:321-333; Meinhardt 2012 Int. J. Dev. Biol. 56:447-462), written with the basal
     inhibitor production $\rho_h, \rho_c, \rho_g$ that gives every subsystem a stable "off"
     state and therefore a *firing threshold* on its drive:
         head      ∂a/∂t = σ [ c a^2 / (h (1 + κ_h a^2)) + ρ_a ] − μ_a a + D_a Δa
                   ∂h/∂t = σ c' a^2 − μ_h h + ρ_h + D_h Δh
         tentacle  ∂b/∂t = φ_t [ c b^2 / c_i + ρ_b ] − μ_a b + D_b Δb,     φ_t = H_t(σ) / (1 + κ_a a)
                   ∂c_i/∂t = c' b^2 − μ_h c_i + ρ_c + D_c Δc_i,            H_t(σ) = σ^p / (σ_t^p + σ^p)
         foot      ∂f/∂t = φ_f [ c f^2 / (g (1 + κ_f f^2)) + ρ_f ] − μ_a f + D_f Δf,  φ_f = 1 / (1 + (σ/σ_f)^2)
                   ∂g/∂t = c' f^2 − μ_h g + ρ_g + D_g Δg
         source    ∂σ/∂t = c_σ a − μ_σ (σ − σ_0(s)) + D_σ Δσ,              σ_0(s) = exp(−4 s / L)
     Tentacle activation needs a high source density (so it sits as close to the hypostome as
     possible) and is suppressed locally by the head activator (so it is excluded from the
     hypostome itself); the foot fires where the source density is lowest; the head activator
     slowly raises the source density (Meinhardt 1993 §"how to form structures close to each
     other / at a distance"). The functional forms $H_t$, $1/(1+\kappa_a a)$ and $\phi_f$ and all
     coefficients are this script's reduced choices, not digitised from the paper.
   - **Analytic firing threshold.** For a thresholded subsystem with drive $w$ the homogeneous
     "off" state exists iff $\mu^2 \rho_{inh} \ge 4 w^2 c\, \mu_{inh}\, \rho_{act}$, i.e. iff
         w ≤ w_c = (μ / 2) · sqrt( ρ_inh / (c μ_inh ρ_act) )   (= 0.1633 for the defaults),
     so the tentacle ring must lie in the band where $\phi_t(s) > w_c$ (checked in `--self-test`
     against the simulated band; monotonicity of the discriminant in $w$ is the Lean theorem
     `meinhardt_three_system_off_state_discriminant_antitone`).
   - **What the shell reproduces (default parameters):** an azimuthally uniform hypostome cap at
     the apex (CV_θ ≈ 0), a single sub-hypostomal tentacle ring of 7 primordia at axial offset
     $s_{ring}/L = 0.106$ (specimen: 6 tentacles attached at $0.070 \pm 0.087$), a uniform basal
     foot at the opposite pole (head–foot separation $= 1.0\,L$), monotone shift of the ring away
     from the apex with the head→tentacle suppression $\kappa_a$ (ablation $\kappa_a = 0$ lets
     tentacle activator leak into the hypostome ~20×), monotone increase of the tentacle number
     with body size $L$, and — after amputation at $L/2$ — regeneration of a new hypostome at the
     fragment tip ($t_{head} < t_{ring}$) followed by a new tentacle ring directly below it.

3. **Hydra Biological Morphogenesis & Experimental Benchmarks:**
   - **Apical Hypostome Dominance (Gierer & Meinhardt 1972 Fig. 2):** A shallow apico-basal
     source density gradient $\rho(x) = 1 - 0.20 (x/L)$ on a 40-cell column ($D_a = 0.03$,
     $D_h = 6.0$) channels lateral inhibition into a single dominant apical head peak at
     $x = 0$ ($a_{\max} = 9.53$) while suppressing secondary peaks across the entire column
     ($a(x \ge 20) \le 0.052$).
   - **Surgical Bisection & Autonomous Head Regeneration (Gierer & Meinhardt 1972 Fig. 3):**
     Cutting the 40-cell Hydra at mid-body ($x = 20$) removes the apical head and its
     long-range inhibitor supply. As residual inhibitor $h(x,t)$ decays in the basal half
     ($x \in [20, 39]$), the suppressed basal tissue ($a_{\text{cut}} = 0.052$) autonomously
     fires autocatalysis at its new apical wound edge ($x_{\text{frag}} = 0$), regenerating
     a full hypostome peak ($a_{\text{regen}} = 8.34$, $87.5\%$ of intact amplitude).
   - **Sub-Hypostomal Periodic Tentacle Ring (Meinhardt 1993 Dev. Biol. 157:321-333):**
     Around a circular sub-hypostomal ring of circumference $C = 64$ cells ($D_a = 0.015$,
     $D_h = 0.45$, $d = 30 > d_c$), Turing-Gierer-Meinhardt instability spontaneously
     generates $N_{\text{tent}} = 6$ quasi-equidistant tentacle primordia (mean spacing
     $10.67 \pm 0.75$ cells, $\text{CV} = 0.070$), matching the 6 tentacles segmented from
     the vendored *Hydra vulgaris* specimen photograph.
   - **Equal-Diffusion Negative Control ($D_a = D_h = 0.015$, $d = 1.0 < d_c$):**
     Perturbations decay exponentially to the spatially uniform equilibrium ($\sigma(k_{\min}) < 0$,
     $\text{std}(a) = 1.8 \times 10^{-15}$, $0$ peaks).

4. **Pass 3 provenance fixes:** the vendored `hydra_activator_inhibitor_reference.json` block
   `hydra_regimes_and_experiments` is now in parity with the code (N = 40, slope −0.20, ring 64,
   domain lengths {22..88}) and carries the new `three_system_cylindrical_shell` block whose
   parameters are asserted against `THREE_SYSTEM_PARAMS` in `--self-test`.
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
from scipy.fft import fft, ifft
from scipy.linalg import solve_banded
from scipy.signal import find_peaks
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "hydra_activator_inhibitor"
REF_JSON_PATH = DATASET_DIR / "hydra_activator_inhibitor_reference.json"
SPECIMEN_PATH = DATASET_DIR / "hydra_vulgaris_specimen.jpg"
SCOUT_PATH = DATASET_DIR / "scout_manifest.json"
BIB_PATH = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "hydra_activator_inhibitor_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "hydra_activator_inhibitor_growth.gif"

EXPECTED_SHA256: dict[str, str] = {
    "datasets/hydra_activator_inhibitor/hydra_vulgaris_specimen.jpg": (
        "007d127bcfa8501d3b9737ffaf7a512efec37db8a0c83a5334568b513da214f9"
    ),
    "datasets/hydra_activator_inhibitor/scout_manifest.json": (
        "b57d0d923283daaf442c82adfb0b709dc507e23ec0d05ae8ea4e9926782e6478"
    ),
    "datasets/hydra_activator_inhibitor/paper_citations.bib": (
        "49b5c0a5531a00fe76fed6d969e9e0af0c9c4ec23499de11893523405f080da6"
    ),
}


def verify_dataset_sha256() -> dict[str, str]:
    """Verify cryptographic SHA-256 digests of all vendored files in datasets/hydra_activator_inhibitor/."""
    computed: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing vendored dataset file: {rel}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {digest}")
        computed[rel] = digest
    if not REF_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing reference JSON: {REF_JSON_PATH}")
    computed["datasets/hydra_activator_inhibitor/hydra_activator_inhibitor_reference.json"] = (
        hashlib.sha256(REF_JSON_PATH.read_bytes()).hexdigest()
    )
    return computed


def load_reference() -> dict[str, Any]:
    """Load and return the vendored Gierer & Meinhardt (1972) reference dataset JSON."""
    return json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Exact Algebraic Steady State, Jacobian & Linear Dispersion (Eqs. 15a-15b)
# --------------------------------------------------------------------------- #
def compute_linear_stability(
    mu_a: float = 0.02,
    mu_h: float = 0.03,
    D_a: float = 0.015,
    D_h: float = 0.45,
    c: float = 0.05,
    c_prime: float = 0.05,
    rho: float = 1.0,
) -> dict[str, Any]:
    """
    Compute the exact Gierer & Meinhardt (1972) Eqs. (15a)-(15b) homogeneous steady state
    (for basal production rho_0 -> 0), reaction Jacobian J_0, critical diffusivity ratio
    d_c = (sqrt(2) + 1)^2 = 3 + 2*sqrt(2), and dispersion relation Re sigma(k) across
    spatial wavenumbers k.
    """
    a_star = (c / c_prime) * (mu_h / mu_a)
    h_star = (c / c_prime) * rho * (mu_h / mu_a) ** 2

    # Reaction Jacobian J_0 = [[mu_a, -b_j], [c_j, -mu_h]]
    j11 = mu_a
    j12 = -(c_prime / c) * (mu_a ** 2 / mu_h) / rho
    j21 = 2.0 * c_prime * rho * a_star
    j22 = -mu_h
    trace_J0 = j11 + j22
    det_J0 = j11 * j22 - j12 * j21  # equals +mu_a * mu_h

    d_c = (math.sqrt(2.0) + 1.0) ** 2
    d_ratio = D_h / D_a
    l_a = math.sqrt(D_a / mu_a)
    l_h = math.sqrt(D_h / mu_h)

    q_c = (math.sqrt(2.0) - 1.0) * (mu_a / D_a)
    k_c = math.sqrt(q_c)
    lam_c = 2.0 * math.pi / k_c

    q_vals = np.linspace(0.0, 2.5, 500)
    k_vals = np.sqrt(q_vals)

    def dispersion_for_dh(dh_val: float) -> tuple[np.ndarray, np.ndarray]:
        tr_q = (mu_a - mu_h) - (D_a + dh_val) * q_vals
        det_q = (D_a * q_vals - mu_a) * (dh_val * q_vals + mu_h) + 2.0 * mu_a * mu_h
        disc = tr_q ** 2 - 4.0 * det_q
        re_part = np.where(
            disc >= 0.0,
            0.5 * (tr_q + np.sqrt(np.maximum(0.0, disc))),
            0.5 * tr_q,
        )
        return re_part, det_q

    re_sigma_super, det_q_super = dispersion_for_dh(D_h)
    re_sigma_crit, _ = dispersion_for_dh(D_a * d_c)
    re_sigma_equal, det_q_equal = dispersion_for_dh(D_a)

    idx_max = int(np.argmax(re_sigma_super))
    k_dom = float(k_vals[idx_max])
    lam_dom = float(2.0 * math.pi / k_dom) if k_dom > 0 else float("inf")

    return {
        "a_star": a_star,
        "h_star": h_star,
        "J0": [[j11, j12], [j21, j22]],
        "trace_J0": trace_J0,
        "det_J0": det_J0,
        "d_c": d_c,
        "d_ratio": d_ratio,
        "l_a": l_a,
        "l_h": l_h,
        "range_ratio": l_h / l_a,
        "k_c": k_c,
        "lambda_c": lam_c,
        "k_dom": k_dom,
        "lambda_dom": lam_dom,
        "max_sigma_super": float(re_sigma_super[idx_max]),
        "max_sigma_equal": float(np.max(re_sigma_equal)),
        "min_det_super": float(np.min(det_q_super)),
        "min_det_equal": float(np.min(det_q_equal)),
        "k_vals": k_vals,
        "re_sigma_super": re_sigma_super,
        "re_sigma_crit": re_sigma_crit,
        "re_sigma_equal": re_sigma_equal,
    }


# --------------------------------------------------------------------------- #
# 2. 1D Reaction-Diffusion Solver (Polar Column, Bisection Regeneration, Ring)
# --------------------------------------------------------------------------- #
def laplacian_1d(u: np.ndarray, bc: str = "neumann") -> np.ndarray:
    """Discrete 1D second-difference Laplacian with zero-flux Neumann or periodic BCs."""
    if bc == "periodic":
        return np.roll(u, -1) + np.roll(u, 1) - 2.0 * u
    out = np.empty_like(u)
    out[1:-1] = u[:-2] + u[2:] - 2.0 * u[1:-1]
    out[0] = 2.0 * (u[1] - u[0])
    out[-1] = 2.0 * (u[-2] - u[-1])
    return out


def detect_1d_peaks(a: np.ndarray, bc: str = "neumann", min_height: float = 2.0, min_dist: int = 4) -> np.ndarray:
    """Detect local activator maxima including boundary peaks (for Neumann) or wrap-around peaks (for periodic)."""
    if bc == "periodic":
        a_pad = np.concatenate([[a[-1]], a, [a[0]]])
        pks, _ = find_peaks(a_pad, height=min_height, distance=min_dist)
        pks = pks - 1
        pks = pks[(pks >= 0) & (pks < len(a))]
        return np.unique(pks)
    # For zero-flux Neumann BCs, reflect-pad by 1 cell so boundary maxima at x=0 or x=N-1 are detected:
    a_pad = np.concatenate([[a[1]], a, [a[-2]]])
    pks, _ = find_peaks(a_pad, height=min_height, distance=min_dist)
    return pks - 1


def simulate_gierer_meinhardt_1d(
    N: int = 40,
    steps: int = 18000,
    dt: float = 0.05,
    c: float = 0.05,
    c_prime: float = 0.05,
    mu_a: float = 0.02,
    mu_h: float = 0.03,
    rho_0: float = 0.001,
    D_a: float = 0.03,
    D_h: float = 6.0,
    gradient_slope: float = -0.20,
    bc: str = "neumann",
    seed: int = 1972,
    noise_amp: float = 0.002,
    a0: np.ndarray | None = None,
    h0: np.ndarray | None = None,
    record_every: int = 450,
) -> dict[str, Any]:
    """Integrate the 1D Gierer-Meinhardt PDE with Neumann or periodic boundary conditions."""
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    if gradient_slope != 0.0:
        rho = 1.0 * (1.0 + gradient_slope * (x / max(1.0, float(N - 1))))
    else:
        rho = np.ones(N, dtype=float)

    a_eq = (c / c_prime) * (mu_h / mu_a)
    h_eq = (c / c_prime) * (mu_h / mu_a) ** 2
    if a0 is None:
        a = np.full(N, a_eq, dtype=float) + rng.normal(0.0, noise_amp, size=N)
        h = np.full(N, h_eq, dtype=float)
    else:
        a = a0.astype(float).copy()
        h = h0.astype(float).copy()
    a = np.clip(a, 1e-4, None)

    history_a = [a.copy()]
    history_h = [h.copy()]
    history_t = [0.0]

    for step in range(1, steps + 1):
        h_safe = np.maximum(h, 1e-6)
        da = rho_0 * rho + c * rho * (a * a) / h_safe - mu_a * a + D_a * laplacian_1d(a, bc=bc)
        dh = c_prime * rho * (a * a) - mu_h * h + D_h * laplacian_1d(h, bc=bc)
        a = np.clip(a + dt * da, 1e-5, 250.0)
        h = np.clip(h + dt * dh, 1e-5, 500.0)
        if step % record_every == 0 or step == steps:
            history_a.append(a.copy())
            history_h.append(h.copy())
            history_t.append(step * dt)

    peaks = detect_1d_peaks(a, bc=bc, min_height=2.0, min_dist=4)
    return {
        "x": x,
        "rho": rho,
        "a": a,
        "h": h,
        "peaks": peaks,
        "n_peaks": int(len(peaks)),
        "apical_max_idx": int(np.argmax(a)),
        "a_max": float(np.max(a)),
        "a_min": float(np.min(a)),
        "a_std": float(np.std(a)),
        "history_a": np.array(history_a),
        "history_h": np.array(history_h),
        "history_t": np.array(history_t),
    }


def simulate_bisection_regeneration(
    N_intact: int = 40,
    cut_index: int = 20,
    steps_intact: int = 18000,
    steps_regen: int = 18000,
    dt: float = 0.05,
    D_a: float = 0.03,
    D_h: float = 6.0,
    seed: int = 1972,
) -> dict[str, Any]:
    """
    Gierer & Meinhardt (1972) Fig. 3 Surgical Bisection & Head Regeneration Experiment:
      Phase 1 (Intact Hydra): A 40-cell body column with shallow source gradient
        rho(x) = 1.0 * (1 - 0.20 * x/39) forms a single tall apical hypostome peak
        at x = 0 (a_max = 9.53), while long-range inhibitor h(x) diffuses into the
        basal half (x >= 20) and suppresses activator to a_cut <= 0.052.
      Phase 2 (Surgical Amputation of Apical Half at x = 20): The basal fragment
        (cells 20..39, length N_frag = 20) is isolated with zero-flux Neumann BCs at
        its new wound edge (x_frag = 0). Deprived of the apical inhibitor source,
        h(x,t) decays below the autocatalytic threshold and a new apical hypostome
        activator peak autonomously regenerates at x_frag = 0 (a_regen = 8.34,
        recovery ratio = 87.5%).
    """
    intact = simulate_gierer_meinhardt_1d(
        N=N_intact,
        steps=steps_intact,
        dt=dt,
        D_a=D_a,
        D_h=D_h,
        gradient_slope=-0.20,
        bc="neumann",
        seed=seed,
        record_every=600,
    )
    a_cut_init = intact["a"][cut_index:].copy()
    h_cut_init = intact["h"][cut_index:].copy()
    pre_cut_basal_max = float(np.max(a_cut_init))

    regen = simulate_gierer_meinhardt_1d(
        N=N_intact - cut_index,
        steps=steps_regen,
        dt=dt,
        D_a=D_a,
        D_h=D_h,
        gradient_slope=-0.10,
        bc="neumann",
        seed=seed,
        a0=a_cut_init,
        h0=h_cut_init,
        record_every=600,
    )
    recovery_ratio = float(regen["a_max"] / intact["a_max"])
    suppression_amplification = float(regen["a_max"] / max(pre_cut_basal_max, 1e-6))
    return {
        "intact": intact,
        "cut_index": cut_index,
        "a_cut_init": a_cut_init,
        "h_cut_init": h_cut_init,
        "pre_cut_basal_max": pre_cut_basal_max,
        "regen": regen,
        "recovery_ratio": recovery_ratio,
        "suppression_amplification": suppression_amplification,
    }


def simulate_tentacle_ring_and_controls(
    N_ring: int = 64,
    steps: int = 14000,
    dt: float = 0.35,
    D_a: float = 0.015,
    D_h: float = 0.45,
    seed: int = 1972,
) -> dict[str, Any]:
    """
    Meinhardt (1993) Dev. Biol. 157:321-333 Periodic Sub-Hypostomal Tentacle Ring:
      1. Supercritical tentacle ring (D_a = 0.015, D_h = 0.45, d = 30.0 > d_c = 5.8284)
         on a periodic ring of N_ring = 64 cells forms N_tent = 6 evenly spaced
         tentacle activator peaks around the circumference.
      2. Equal-diffusion negative control (D_a = D_h = 0.015, d = 1.0 < d_c = 5.8284)
         with identical initial random perturbation decays to the flat homogeneous
         equilibrium (0 peaks, std(a) < 1e-4).
      3. 6-point diffusivity-ratio sweep across d = D_h / D_a in {1.0, 3.0, 5.0, 8.0, 15.0, 30.0}
         confirming zero pattern below d_c = 5.8284 and spontaneous symmetry breaking above d_c.
    """
    ring_super = simulate_gierer_meinhardt_1d(
        N=N_ring,
        steps=steps,
        dt=dt,
        D_a=D_a,
        D_h=D_h,
        gradient_slope=0.0,
        bc="periodic",
        seed=seed,
        noise_amp=0.01,
        record_every=500,
    )
    pks = ring_super["peaks"]
    if len(pks) >= 2:
        spacings = np.diff(np.append(pks, pks[0] + N_ring)).astype(float)
        mean_spacing_cells = float(np.mean(spacings))
        std_spacing_cells = float(np.std(spacings))
        cv_spacing = float(std_spacing_cells / mean_spacing_cells)
        angles_deg = (pks.astype(float) / N_ring) * 360.0
        angular_spacings_deg = (spacings / N_ring) * 360.0
    else:
        spacings = np.array([], dtype=float)
        mean_spacing_cells = 0.0
        std_spacing_cells = 0.0
        cv_spacing = 0.0
        angles_deg = np.array([], dtype=float)
        angular_spacings_deg = np.array([], dtype=float)

    ring_control = simulate_gierer_meinhardt_1d(
        N=N_ring,
        steps=steps,
        dt=dt,
        D_a=D_a,
        D_h=D_a,  # equal diffusion D_h = D_a
        gradient_slope=0.0,
        bc="periodic",
        seed=seed,
        noise_amp=0.01,
        record_every=500,
    )

    sweep_ratios = [1.0, 3.0, 5.0, 12.0, 20.0, 30.0]
    sweep_results = []
    for d_val in sweep_ratios:
        dh_val = D_a * d_val
        dt_val = min(0.65, 0.35 * (0.45 / max(dh_val, 0.015)))
        sim_d = simulate_gierer_meinhardt_1d(
            N=N_ring,
            steps=14000,
            dt=dt_val,
            D_a=D_a,
            D_h=dh_val,
            gradient_slope=0.0,
            bc="periodic",
            seed=seed,
            noise_amp=0.02,
            record_every=3500,
        )
        sweep_results.append(
            {
                "d_ratio": d_val,
                "D_h": dh_val,
                "a_std": sim_d["a_std"],
                "a_amplitude": sim_d["a_max"] - sim_d["a_min"],
                "n_peaks": sim_d["n_peaks"],
            }
        )

    return {
        "N_ring": N_ring,
        "ring_super": ring_super,
        "n_tentacle_peaks": int(len(pks)),
        "peak_indices": pks.tolist(),
        "peak_angles_deg": angles_deg.tolist(),
        "spacings_cells": spacings.tolist(),
        "angular_spacings_deg": angular_spacings_deg.tolist(),
        "mean_spacing_cells": mean_spacing_cells,
        "std_spacing_cells": std_spacing_cells,
        "cv_spacing": cv_spacing,
        "ring_control": ring_control,
        "sweep_results": sweep_results,
    }


def simulate_domain_scaling_sweep(
    domain_lengths: list[int] | None = None,
    D_a: float = 0.015,
    D_h: float = 0.45,
    seed: int = 1972,
) -> dict[str, Any]:
    """
    Allometric Domain-Length Scaling Sweep:
    Simulate periodic Gierer-Meinhardt rings across domain circumferences
    L in {22, 32, 42, 53, 64, 75, 86} cells and verify linear peak-count scaling
    N_peaks(L) ~ L / lambda_nonlinear (Pearson r, RMSE).
    """
    if domain_lengths is None:
        domain_lengths = [22, 33, 44, 54, 64, 76, 88]
    counts: list[int] = []
    mean_wavelengths: list[float] = []
    for L in domain_lengths:
        sim_L = simulate_gierer_meinhardt_1d(
            N=L,
            steps=12000,
            dt=0.35,
            D_a=D_a,
            D_h=D_h,
            gradient_slope=0.0,
            bc="periodic",
            seed=seed + L,
            noise_amp=0.01,
            record_every=4000,
        )
        n_p = sim_L["n_peaks"]
        counts.append(n_p)
        mean_wavelengths.append(float(L) / max(1, n_p))

    L_arr = np.array(domain_lengths, dtype=float)
    n_arr = np.array(counts, dtype=float)
    r_corr = float(np.corrcoef(L_arr, n_arr)[0, 1])
    mean_lam = float(np.mean(mean_wavelengths))
    predicted = L_arr / mean_lam
    rmse_peaks = float(np.sqrt(np.mean((n_arr - predicted) ** 2)))
    return {
        "domain_lengths": domain_lengths,
        "peak_counts": counts,
        "mean_wavelengths": mean_wavelengths,
        "mean_nonlinear_wavelength": mean_lam,
        "pearson_r": r_corr,
        "rmse_peaks": rmse_peaks,
    }


# --------------------------------------------------------------------------- #
# 3. Biological Specimen Computer-Vision Segmentation (Hydra vulgaris)
# --------------------------------------------------------------------------- #
def segment_hydra_specimen(specimen_path: Path = SPECIMEN_PATH) -> dict[str, Any]:
    """
    Segment the dark-field whole-mount photograph of Hydra vulgaris
    (`datasets/hydra_activator_inhibitor/hydra_vulgaris_specimen.jpg`, CC BY-SA 3.0):
      1. Extract the translucent polyp silhouette via Gaussian smoothing and intensity
         thresholding + morphological opening/closing.
      2. Compute the Euclidean distance transform on the polyp silhouette to isolate
         the thick tubular body column (stalk core where radius > 22 px).
      3. Perform Principal Component Analysis (PCA) on the body column core to identify
         the apico-basal polarity axis, basal disc (foot) coordinate, and apical
         hypostome head center `(x_hypo, y_hypo)`.
      4. Sample the radial intensity profile in the sub-hypostomal tentacle crown
         annulus (`r in [60, 130]` px around `(x_hypo, y_hypo)`, masking out the basal
         body-stalk sector) to detect the 6 radiating tentacles and compute their
         angular positions and inter-tentacle spacing statistics.
    """
    img = np.asarray(Image.open(specimen_path).convert("RGB"))
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    smooth = ndi.gaussian_filter(gray, sigma=1.8)

    mask = smooth > np.percentile(smooth, 76)
    mask = ndi.binary_opening(mask, iterations=1)
    lab, n_comp = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n_comp + 1))
    main_mask = lab == (1 + int(np.argmax(sizes)))

    dist = ndi.distance_transform_edt(main_mask)
    stalk_core = dist > 22.0
    cy, cx = np.where(stalk_core)
    pts = np.column_stack([cx.astype(float), cy.astype(float)])
    s_mean = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - s_mean).T))
    axis_vec = evecs[:, np.argmax(evals)]
    if axis_vec[1] > 0:  # orient axis from basal foot (bottom-left) to apical hypostome (top-right)
        axis_vec = -axis_vec
    proj = (pts - s_mean) @ axis_vec
    hypo_xy = pts[int(np.argmax(proj))]
    foot_xy = pts[int(np.argmin(proj))]
    stalk_length_px = float(np.linalg.norm(hypo_xy - foot_xy))

    # Sample axial luminance profile along the body stalk from hypostome (s=0) to foot (s=1)
    n_axial = 40
    s_grid = np.linspace(0.0, 1.0, n_axial)
    axial_x = np.clip(np.round(hypo_xy[0] * (1.0 - s_grid) + foot_xy[0] * s_grid).astype(int), 0, img.shape[1] - 1)
    axial_y = np.clip(np.round(hypo_xy[1] * (1.0 - s_grid) + foot_xy[1] * s_grid).astype(int), 0, img.shape[0] - 1)
    axial_intensity = smooth[axial_y, axial_x]

    # Sample circular intensity profile around hypostome across annulus r in [60, 130] px
    angles_deg = np.linspace(-180.0, 180.0, 360, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)
    radii = range(60, 130, 3)
    prof = np.zeros_like(angles_rad)
    for r in radii:
        px = np.clip(np.round(hypo_xy[0] + r * np.cos(angles_rad)).astype(int), 0, img.shape[1] - 1)
        py = np.clip(np.round(hypo_xy[1] + r * np.sin(angles_rad)).astype(int), 0, img.shape[0] - 1)
        prof += smooth[py, px]
    prof /= float(len(radii))

    foot_angle_deg = float(np.rad2deg(np.arctan2(foot_xy[1] - hypo_xy[1], foot_xy[0] - hypo_xy[0])))
    ang_diff = np.abs((angles_deg - foot_angle_deg + 180.0) % 360.0 - 180.0)
    prof_tent = prof.copy()
    prof_tent[ang_diff < 28.0] = np.percentile(prof, 15)
    prof_smooth = ndi.gaussian_filter1d(prof_tent, sigma=3.0, mode="wrap")
    peaks, _ = find_peaks(prof_smooth, prominence=4.0, distance=18)

    tent_angles_deg = np.sort(angles_deg[peaks])
    if len(tent_angles_deg) >= 2:
        tent_gaps_deg = np.diff(np.append(tent_angles_deg, tent_angles_deg[0] + 360.0))
        mean_gap_deg = float(np.mean(tent_gaps_deg))
        cv_gap_deg = float(np.std(tent_gaps_deg) / mean_gap_deg)
    else:
        tent_gaps_deg = np.array([], dtype=float)
        mean_gap_deg = 0.0
        cv_gap_deg = 0.0

    r_marker = 95.0
    tent_coords_xy = np.column_stack(
        [
            hypo_xy[0] + r_marker * np.cos(np.deg2rad(tent_angles_deg)),
            hypo_xy[1] + r_marker * np.sin(np.deg2rad(tent_angles_deg)),
        ]
    )

    return {
        "image_rgb": img,
        "main_mask": main_mask,
        "stalk_core": stalk_core,
        "hypostome_xy": hypo_xy.tolist(),
        "foot_xy": foot_xy.tolist(),
        "stalk_length_px": stalk_length_px,
        "axial_s": s_grid,
        "axial_intensity": axial_intensity,
        "angles_deg": angles_deg,
        "angular_profile": prof_smooth,
        "foot_angle_deg": foot_angle_deg,
        "n_tentacles_segmented": int(len(tent_angles_deg)),
        "tentacle_angles_deg": tent_angles_deg.tolist(),
        "tentacle_gaps_deg": tent_gaps_deg.tolist(),
        "mean_tentacle_gap_deg": mean_gap_deg,
        "cv_tentacle_gap": cv_gap_deg,
        "tentacle_coords_xy": tent_coords_xy,
    }


# --------------------------------------------------------------------------- #
# 3b. Specimen Body-Column Radius Profile & Tentacle-Ring Attachment (Pass 3)
# --------------------------------------------------------------------------- #
def measure_specimen_body_profile(spec_seg: dict[str, Any], n_samples: int = 200) -> dict[str, Any]:
    """
    Measure the apico-basal body-column radius profile r(s) of the segmented Hydra vulgaris
    specimen and the axial position of its tentacle-ring attachment, both used by the 3D
    cylindrical-shell model:
      1. Walk from the PCA hypostome point along the body axis while the Euclidean distance
         transform keeps decreasing (the tapering hypostome dome) -> apical tip; walk beyond the
         basal point until the silhouette ends -> basal disc end. Their separation is the full
         body length L_px.
      2. Sample the distance transform (inscribed radius) at n_samples points from tip to basal
         disc -> r_px(s), lightly smoothed (sigma = 1.5 samples), floored at 3 px.
      3. The tentacle ring attaches where the continuous hypostome disc fragments into discrete
         arms: the first radius around the hypostome centre at which the tissue fraction of the
         circle (outside the +-28 deg basal-stalk sector) drops below 0.98 -> r_attach_px.
      4. Specimen references for the shell: ring axial offset tip->attachment / L_px, ring
         half-thickness r_attach / L_px, and inter-tentacle spacing 2*pi*r_attach / N_tent / L_px.
    All quantities are measured on a 2D projection of a tilted animal and are therefore
    approximate; they are used as reference bands, not as exact targets.
    """
    mask = spec_seg["main_mask"]
    dist = ndi.distance_transform_edt(mask)
    hx, hy = spec_seg["hypostome_xy"]
    fx, fy = spec_seg["foot_xy"]
    axis_vec = np.array([hx - fx, hy - fy], dtype=float)
    axis_vec /= np.linalg.norm(axis_vec)

    def dist_at(p: np.ndarray) -> float:
        xi, yi = int(round(p[0])), int(round(p[1]))
        if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]:
            return float(dist[yi, xi])
        return 0.0

    hypo = np.array([hx, hy], dtype=float)
    foot = np.array([fx, fy], dtype=float)
    k = 0
    prev = dist_at(hypo)
    while k < 60:
        d_next = dist_at(hypo + (k + 1) * axis_vec)
        if d_next <= 0.0 or d_next > prev + 0.5:
            break
        prev = d_next
        k += 1
    tip_offset_px = float(k)
    k = 0
    while k < 60 and dist_at(foot - (k + 1) * axis_vec) > 0.0:
        k += 1
    foot_offset_px = float(k)
    tip_xy = hypo + tip_offset_px * axis_vec
    base_xy = foot - foot_offset_px * axis_vec
    body_length_px = float(np.linalg.norm(tip_xy - base_xy))

    s_grid = (np.arange(n_samples) + 0.5) / n_samples
    pts = tip_xy[None, :] * (1.0 - s_grid[:, None]) + base_xy[None, :] * s_grid[:, None]
    r_px = np.array([dist_at(p) for p in pts], dtype=float)
    r_px = ndi.gaussian_filter1d(r_px, 1.5, mode="nearest")
    r_px = np.maximum(r_px, 3.0)

    angles = np.deg2rad(np.linspace(-180.0, 180.0, 360, endpoint=False))
    ang_diff = np.abs((np.rad2deg(angles) - spec_seg["foot_angle_deg"] + 180.0) % 360.0 - 180.0)
    sel = ang_diff >= 28.0
    r_attach_px = float("nan")
    for rr in range(5, 150):
        px = np.clip(np.round(hx + rr * np.cos(angles)).astype(int), 0, mask.shape[1] - 1)
        py = np.clip(np.round(hy + rr * np.sin(angles)).astype(int), 0, mask.shape[0] - 1)
        tissue_frac = float(mask[py, px][sel].mean())
        if tissue_frac < 0.98:
            r_attach_px = float(rr)
            break
    n_tent = max(1, int(spec_seg["n_tentacles_segmented"]))
    ring_offset_frac = tip_offset_px / body_length_px
    ring_half_thickness_frac = r_attach_px / body_length_px
    spacing_frac = 2.0 * math.pi * r_attach_px / n_tent / body_length_px
    return {
        "tip_xy": tip_xy.tolist(),
        "base_xy": base_xy.tolist(),
        "tip_offset_px": tip_offset_px,
        "foot_offset_px": foot_offset_px,
        "body_length_px": body_length_px,
        "s_grid": s_grid,
        "r_px": r_px,
        "r_frac": r_px / body_length_px,
        "max_radius_frac": float(np.max(r_px) / body_length_px),
        "r_attach_px": r_attach_px,
        "ring_offset_frac": float(ring_offset_frac),
        "ring_half_thickness_frac": float(ring_half_thickness_frac),
        "tentacle_spacing_frac": float(spacing_frac),
        "n_tentacles": n_tent,
    }


# --------------------------------------------------------------------------- #
# 3c. Coupled Three-System Model on a Specimen-Shaped 3D Cylindrical Shell (Pass 3)
# --------------------------------------------------------------------------- #
THREE_SYSTEM_PARAMS: dict[str, float] = {
    # shared Gierer-Meinhardt kinetics (Pass 2 canonical values)
    "c": 0.05, "c_prime": 0.05, "mu_a": 0.02, "mu_h": 0.03,
    # head (hypostome) loop: short-range activator, body-size-range inhibitor, saturating autocatalysis
    "D_a": 0.03, "D_h": 60.0, "rho_a": 0.001, "rho_h": 0.0001, "kappa_head": 0.02,
    # tentacle loop: Pass 2 ring diffusivities, source-density gating, head-activator suppression
    "D_b": 0.015, "D_c": 0.45, "rho_b": 0.005, "rho_c": 0.002, "kappa_a": 1.0, "sigma_t": 0.6, "p_t": 8.0,
    # foot loop: fires where source density is lowest
    "D_f": 0.03, "D_g": 60.0, "rho_f": 0.005, "rho_g": 0.002, "kappa_f": 0.08, "sigma_f": 0.03, "p_f": 2.0,
    # source density: graded initial profile exp(-s / (L * l_sigma_frac)), raised slowly by the head activator
    "D_sigma": 0.05, "c_sigma": 0.0012, "mu_sigma": 0.004, "l_sigma_frac": 0.25,
}
SHELL_DEFAULT_LENGTH_CELLS = 90
SHELL_DEFAULT_NTHETA = 64
SHELL_DEFAULT_STEPS = 3000
SHELL_DEFAULT_DT = 0.5
SHELL_KAPPA_A_SWEEP = [0.0, 0.3, 1.0, 3.0, 10.0]
SHELL_LENGTH_SWEEP = [64, 72, 90, 110, 131, 150]


def three_system_firing_threshold(
    mu_act: float, c: float, mu_inh: float, rho_act: float, rho_inh: float
) -> float:
    """
    Drive threshold w_c of a thresholded activator-inhibitor subsystem
        b' = w (c b^2 / c_i + rho_act) - mu_act b,   c_i' = c' b^2 - mu_inh c_i + rho_inh
    (small-b limit c' b^2 << rho_inh): the homogeneous "off" state exists iff the quadratic
    (w c mu_inh / rho_inh) b^2 - mu_act b + w rho_act = 0 has real roots, i.e. iff
        mu_act^2 rho_inh >= 4 w^2 c mu_inh rho_act   <=>   w <= w_c = (mu_act/2) sqrt(rho_inh / (c mu_inh rho_act)).
    Above w_c the subsystem cannot stay off and must fire (Lean: meinhardt_three_system_off_state_discriminant_antitone).
    """
    return 0.5 * mu_act * math.sqrt(rho_inh / (c * mu_inh * rho_act))


def three_system_off_state_discriminant(w: float, mu_act: float, c: float, mu_inh: float, rho_act: float, rho_inh: float) -> float:
    """Discriminant mu_act^2 rho_inh - 4 w^2 c mu_inh rho_act of the off-state quadratic (>= 0 iff the off state exists)."""
    return mu_act ** 2 * rho_inh - 4.0 * w ** 2 * c * mu_inh * rho_act


class CylindricalShellGrid:
    """
    Finite-difference discretisation of the Laplace-Beltrami operator on a surface of revolution
    with radius profile r(s) (cells), N_s axial rows (unit spacing, zero-flux ends) and N_theta
    periodic circumferential columns:
        Delta u = (1/r) d/ds ( r du/ds ) + (1/r^2) d^2u/dtheta^2.
    `diffuse(u, D, dt)` applies exp(dt * D * Delta_theta) exactly in Fourier space (row-dependent
    symbol) followed by a backward-Euler step of the tridiagonal axial operator — unconditionally
    stable for any D, which is what makes the body-size-range head/foot inhibitors (D = 60) affordable.
    """

    def __init__(self, r_cells: np.ndarray, n_theta: int) -> None:
        self.r = np.asarray(r_cells, dtype=float)
        self.n_s = int(self.r.size)
        self.n_theta = int(n_theta)
        d_theta = 2.0 * math.pi / self.n_theta
        n = np.arange(self.n_theta)
        k_theta_sq = (2.0 - 2.0 * np.cos(2.0 * math.pi * n / self.n_theta)) / d_theta ** 2
        self.theta_symbol = k_theta_sq[None, :] / (self.r[:, None] ** 2)
        self.r_half = 0.5 * (self.r[1:] + self.r[:-1])
        self._banded_cache: dict[tuple[float, float], np.ndarray] = {}

    def banded_axial_operator(self, D: float, dt: float) -> np.ndarray:
        key = (float(D), float(dt))
        ab = self._banded_cache.get(key)
        if ab is not None:
            return ab
        n = self.n_s
        right = np.zeros(n)
        left = np.zeros(n)
        right[:-1] = self.r_half / self.r[:-1]
        left[1:] = self.r_half / self.r[1:]
        coef = D * dt
        diag = 1.0 + coef * (right + left)
        upper = np.zeros(n)
        lower = np.zeros(n)
        upper[1:] = -coef * right[:-1]
        lower[:-1] = -coef * left[1:]
        ab = np.vstack([upper, diag, lower])
        self._banded_cache[key] = ab
        return ab

    def diffuse(self, u: np.ndarray, D: float, dt: float) -> np.ndarray:
        if D <= 0.0:
            return u
        U = fft(u, axis=1)
        U *= np.exp(-D * dt * self.theta_symbol)
        u_theta = ifft(U, axis=1).real
        return solve_banded((1, 1), self.banded_axial_operator(D, dt), u_theta)

    def surface_xyz(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3D coordinates (apex at the top, z = L - s) with the periodic seam closed for rendering."""
        theta = np.linspace(0.0, 2.0 * math.pi, self.n_theta + 1)
        s = np.arange(self.n_s, dtype=float) + 0.5
        R, TH = np.meshgrid(self.r, theta, indexing="ij")
        X = R * np.cos(TH)
        Y = R * np.sin(TH)
        Z = (float(self.n_s) - s)[:, None] * np.ones_like(TH)
        return X, Y, Z


def specimen_shell_radius_profile(body_profile: dict[str, Any], L_cells: int) -> np.ndarray:
    """Resample the specimen radius profile r(s)/L onto an L_cells-row shell (cells)."""
    s_frac = (np.arange(L_cells) + 0.5) / L_cells
    return np.interp(s_frac, body_profile["s_grid"], body_profile["r_frac"]) * float(L_cells)


def initial_shell_state(n_s: int, n_theta: int, seed: int, params: dict[str, float]) -> dict[str, Any]:
    """Homogeneous Gierer-Meinhardt head state + small 'off' tentacle/foot levels + graded source density."""
    rng = np.random.default_rng(seed)
    s_frac = (np.arange(n_s) + 0.5) / n_s
    sigma0 = np.exp(-s_frac / params["l_sigma_frac"])[:, None] * np.ones((1, n_theta))
    a_eq = (params["c"] / params["c_prime"]) * (params["mu_h"] / params["mu_a"])
    shape = (n_s, n_theta)
    return {
        "a": np.clip(a_eq + rng.normal(0.0, 0.002, shape), 1e-4, None),
        "h": np.full(shape, a_eq ** 2),
        "b": np.clip(0.05 + rng.normal(0.0, 0.005, shape), 1e-4, None),
        "c": np.full(shape, 0.1),
        "f": np.clip(0.05 + rng.normal(0.0, 0.005, shape), 1e-4, None),
        "g": np.full(shape, 0.1),
        "sigma": sigma0.copy(),
        "sigma0": sigma0,
    }


def tentacle_drive(sigma: np.ndarray, a: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """phi_t = H_t(sigma) / (1 + kappa_a a): source-density gating x head-activator suppression."""
    p = params["p_t"]
    H_t = sigma ** p / (params["sigma_t"] ** p + sigma ** p)
    return H_t / (1.0 + params["kappa_a"] * a)


def foot_drive(sigma: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """phi_f = 1 / (1 + (sigma / sigma_f)^p_f): the foot fires where the source density is lowest."""
    return 1.0 / (1.0 + (sigma / params["sigma_f"]) ** params["p_f"])


def simulate_three_system_shell(
    r_cells: np.ndarray,
    n_theta: int = SHELL_DEFAULT_NTHETA,
    steps: int = SHELL_DEFAULT_STEPS,
    dt: float = SHELL_DEFAULT_DT,
    seed: int = 1972,
    params: dict[str, float] | None = None,
    state: dict[str, Any] | None = None,
    record_every: int = 250,
) -> dict[str, Any]:
    """
    Integrate the coupled head / tentacle / foot activator-inhibitor loops and the source density
    (module docstring §2b) on the specimen-shaped shell. Explicit Euler for the reactions, exact
    Fourier (theta) + backward-Euler (s) operator splitting for diffusion. `state` lets a run
    continue from an amputated fragment of a previous run (shell regeneration experiment).
    """
    p = dict(THREE_SYSTEM_PARAMS if params is None else params)
    grid = CylindricalShellGrid(r_cells, n_theta)
    st = initial_shell_state(grid.n_s, n_theta, seed, p) if state is None else {k: v.copy() for k, v in state.items()}
    diffusivities = {"a": p["D_a"], "h": p["D_h"], "b": p["D_b"], "c": p["D_c"], "f": p["D_f"], "g": p["D_g"], "sigma": p["D_sigma"]}
    history: list[dict[str, np.ndarray]] = []
    history_t: list[float] = []
    c, cp, mu_a, mu_h = p["c"], p["c_prime"], p["mu_a"], p["mu_h"]
    for step in range(1, steps + 1):
        a, h, b, ci, f, g, sigma = (st[k] for k in ("a", "h", "b", "c", "f", "g", "sigma"))
        hs, cs, gs = np.maximum(h, 1e-6), np.maximum(ci, 1e-6), np.maximum(g, 1e-6)
        da = sigma * (c * a * a / (hs * (1.0 + p["kappa_head"] * a * a)) + p["rho_a"]) - mu_a * a
        dh = sigma * cp * a * a - mu_h * h + p["rho_h"]
        phi_t = tentacle_drive(sigma, a, p)
        db = phi_t * (c * b * b / cs + p["rho_b"]) - mu_a * b
        dc = cp * b * b - mu_h * ci + p["rho_c"]
        phi_f = foot_drive(sigma, p)
        df = phi_f * (c * f * f / (gs * (1.0 + p["kappa_f"] * f * f)) + p["rho_f"]) - mu_a * f
        dg = cp * f * f - mu_h * g + p["rho_g"]
        dsigma = p["c_sigma"] * a - p["mu_sigma"] * (sigma - st["sigma0"])
        st["a"] = np.clip(a + dt * da, 1e-5, 250.0)
        st["h"] = np.clip(h + dt * dh, 1e-5, 1e4)
        st["b"] = np.clip(b + dt * db, 1e-5, 250.0)
        st["c"] = np.clip(ci + dt * dc, 1e-5, 1e4)
        st["f"] = np.clip(f + dt * df, 1e-5, 250.0)
        st["g"] = np.clip(g + dt * dg, 1e-5, 1e4)
        st["sigma"] = np.clip(sigma + dt * dsigma, 1e-3, 10.0)
        for key, D in diffusivities.items():
            st[key] = grid.diffuse(st[key], D, dt)
        if step % record_every == 0 or step == steps:
            history.append({k: st[k].copy() for k in ("a", "b", "f", "sigma")})
            history_t.append(step * dt)
    st["grid"] = grid
    st["r_cells"] = np.asarray(r_cells, dtype=float)
    st["history"] = history
    st["history_t"] = history_t
    st["params"] = p
    st["dt"] = dt
    return st


def analyze_shell_pattern(st: dict[str, Any]) -> dict[str, Any]:
    """
    Morphometrics of a shell state: apical head cap (row of max azimuthal-mean a, its azimuthal CV,
    number of axial head peaks), tentacle ring (rows where max_theta b > 0.5 max b, ring-averaged
    theta profile, primordia count via periodic peak detection, spacing CV, axial offset and
    circumference), foot (row of max f, azimuthal CV), apex tentacle leakage b(s=0)/b_ring and the
    band predicted by the analytic firing threshold phi_t(s) > w_c.
    """
    a, b, f, sigma = st["a"], st["b"], st["f"], st["sigma"]
    p = st["params"]
    n_s, n_theta = a.shape
    r = st["r_cells"]
    a_axial = a.mean(axis=1)
    f_axial = f.mean(axis=1)
    b_axial_max = b.max(axis=1)
    i_head = int(np.argmax(a_axial))
    i_foot = int(np.argmax(f_axial))
    head_cv = float(a[i_head].std() / max(a[i_head].mean(), 1e-9))
    foot_cv = float(f[i_foot].std() / max(f[i_foot].mean(), 1e-9))
    head_peaks = detect_1d_peaks(a_axial, bc="neumann", min_height=0.5 * float(a_axial.max()), min_dist=3)
    band = np.where(b_axial_max > 0.5 * float(b_axial_max.max()))[0]
    ring_profile = b[band].mean(axis=0)
    ring_peaks = detect_1d_peaks(ring_profile, bc="periodic", min_height=0.5 * float(ring_profile.max()), min_dist=3)
    if len(ring_peaks) >= 2:
        spacings = np.diff(np.append(ring_peaks, ring_peaks[0] + n_theta)).astype(float)
        spacing_cv = float(spacings.std() / spacings.mean())
    else:
        spacings = np.array([], dtype=float)
        spacing_cv = 0.0
    ring_row = float(np.median(band))
    ring_circ_cells = float(2.0 * math.pi * r[int(round(ring_row))])
    n_ring = int(len(ring_peaks))
    spacing_cells = ring_circ_cells / max(n_ring, 1)
    b_centroid = float((b.mean(axis=1) * np.arange(n_s)).sum() / max(b.mean(axis=1).sum(), 1e-12))
    w_c = three_system_firing_threshold(p["mu_a"], p["c"], p["mu_h"], p["rho_b"], p["rho_c"])
    phi_t_axial = tentacle_drive(sigma.mean(axis=1), a_axial, p)
    predicted_band = np.where(phi_t_axial > w_c)[0]
    return {
        "i_head": i_head,
        "head_cv_theta": head_cv,
        "n_head_peaks": int(len(head_peaks)),
        "a_max": float(a.max()),
        "sigma_apex": float(sigma[0].mean()),
        "ring_rows": band.tolist(),
        "ring_row": ring_row,
        "ring_offset_frac": ring_row / n_s,
        "ring_centroid_frac": b_centroid / n_s,
        "ring_width_rows": int(band.max() - band.min() + 1),
        "ring_profile": ring_profile,
        "ring_peak_indices": ring_peaks.tolist(),
        "n_tentacles": n_ring,
        "ring_spacing_cv": spacing_cv,
        "ring_circumference_cells": ring_circ_cells,
        "tentacle_spacing_cells": spacing_cells,
        "tentacle_spacing_frac": spacing_cells / n_s,
        "b_max": float(b.max()),
        "apex_tentacle_leak": float(b[0].max() / max(float(b.max()), 1e-12)),
        "i_foot": i_foot,
        "foot_offset_frac": i_foot / n_s,
        "foot_cv_theta": foot_cv,
        "f_max": float(f.max()),
        "head_foot_separation_frac": (i_foot - i_head) / n_s,
        "firing_threshold_w_c": w_c,
        "predicted_ring_band": predicted_band.tolist(),
        "phi_t_axial": phi_t_axial,
    }


def simulate_shell_regeneration(
    intact: dict[str, Any],
    cut_frac: float = 0.5,
    steps: int = 4000,
    record_every: int = 100,
) -> dict[str, Any]:
    """
    Amputate the apical half of the intact shell (rows s < cut) and let the basal fragment evolve
    with its own radius profile and inherited source density. Records the first time the new head
    cap reaches half its final amplitude at the fragment tip (t_head) and the first time a tentacle
    ring (rows >= 1) reaches half its final amplitude (t_ring); Meinhardt (1993, 2012) fragments
    regenerate a hypostome and a new tentacle ring directly below it.
    """
    n_s = intact["a"].shape[0]
    cut = int(round(cut_frac * n_s))
    frag_state = {k: intact[k][cut:].copy() for k in ("a", "h", "b", "c", "f", "g", "sigma", "sigma0")}
    regen = simulate_three_system_shell(
        intact["r_cells"][cut:],
        n_theta=intact["a"].shape[1],
        steps=steps,
        dt=intact["dt"],
        seed=1972,
        params=intact["params"],
        state=frag_state,
        record_every=record_every,
    )
    an = analyze_shell_pattern(regen)
    a_tip_final = float(regen["a"][0].mean())
    b_final = float(regen["b"].max())
    t_head = None
    t_ring = None
    for snap, t in zip(regen["history"], regen["history_t"]):
        if t_head is None and float(snap["a"][0].mean()) > 0.5 * a_tip_final:
            t_head = float(t)
        if t_ring is None and float(snap["b"][1:].max()) > 0.5 * b_final:
            t_ring = float(t)
    return {
        "cut_row": cut,
        "regen": regen,
        "analysis": an,
        "head_recovery_ratio": float(regen["a"].max() / max(float(intact["a"].max()), 1e-12)),
        "a_tip_final": a_tip_final,
        "t_head": t_head,
        "t_ring": t_ring,
        "pre_cut_tip_a": float(intact["a"][cut].mean()),
        "pre_cut_tip_b": float(intact["b"][cut].max()),
    }


def simulate_shell_experiments(
    body_profile: dict[str, Any],
    L_cells: int = SHELL_DEFAULT_LENGTH_CELLS,
    n_theta: int = SHELL_DEFAULT_NTHETA,
    steps: int = SHELL_DEFAULT_STEPS,
    seed: int = 1972,
    kappa_a: float = THREE_SYSTEM_PARAMS["kappa_a"],
    kappa_sweep: list[float] | None = None,
    length_sweep: list[int] | None = None,
    regen_steps: int = 4000,
) -> dict[str, Any]:
    """
    Full Pass 3 shell experiment set: intact three-system shell, kappa_a (head->tentacle
    suppression) sweep incl. the kappa_a = 0 ablation, body-length sweep (tentacle number vs size),
    and amputation/regeneration of the basal half. Returns all states plus the analyses and the
    Spearman monotonicity statistics used by measure_morphoeval() and self_test().
    """
    kappa_sweep = SHELL_KAPPA_A_SWEEP if kappa_sweep is None else kappa_sweep
    length_sweep = SHELL_LENGTH_SWEEP if length_sweep is None else length_sweep
    params = dict(THREE_SYSTEM_PARAMS)
    params["kappa_a"] = kappa_a
    r_cells = specimen_shell_radius_profile(body_profile, L_cells)
    intact = simulate_three_system_shell(r_cells, n_theta=n_theta, steps=steps, seed=seed, params=params)
    intact_an = analyze_shell_pattern(intact)

    kappa_results = []
    for ka in kappa_sweep:
        pk = dict(params)
        pk["kappa_a"] = ka
        if abs(ka - kappa_a) < 1e-12:
            st_k, an_k = intact, intact_an
        else:
            st_k = simulate_three_system_shell(r_cells, n_theta=n_theta, steps=steps, seed=seed, params=pk, record_every=steps)
            an_k = analyze_shell_pattern(st_k)
        kappa_results.append(
            {
                "kappa_a": ka,
                "ring_offset_frac": an_k["ring_offset_frac"],
                "ring_centroid_frac": an_k["ring_centroid_frac"],
                "apex_tentacle_leak": an_k["apex_tentacle_leak"],
                "n_tentacles": an_k["n_tentacles"],
                "state": st_k,
            }
        )
    ka_arr = np.array([row["kappa_a"] for row in kappa_results])
    offs = np.array([row["ring_centroid_frac"] for row in kappa_results])
    kappa_spearman = float(spearmanr(ka_arr, offs).statistic) if len(ka_arr) >= 3 else float("nan")

    length_results = []
    for L in length_sweep:
        if L == L_cells:
            an_L = intact_an
        else:
            r_L = specimen_shell_radius_profile(body_profile, L)
            st_L = simulate_three_system_shell(r_L, n_theta=n_theta, steps=steps, seed=seed, params=params, record_every=steps)
            an_L = analyze_shell_pattern(st_L)
        length_results.append(
            {
                "L_cells": L,
                "n_tentacles": an_L["n_tentacles"],
                "ring_offset_frac": an_L["ring_offset_frac"],
                "ring_circumference_cells": an_L["ring_circumference_cells"],
                "tentacle_spacing_cells": an_L["tentacle_spacing_cells"],
                "tentacle_spacing_frac": an_L["tentacle_spacing_frac"],
            }
        )
    L_arr = np.array([row["L_cells"] for row in length_results], dtype=float)
    n_arr = np.array([row["n_tentacles"] for row in length_results], dtype=float)
    length_spearman = float(spearmanr(L_arr, n_arr).statistic) if len(L_arr) >= 3 else float("nan")

    regen = simulate_shell_regeneration(intact, cut_frac=0.5, steps=regen_steps)
    ablation = kappa_results[0] if kappa_results and kappa_results[0]["kappa_a"] == 0.0 else None
    return {
        "L_cells": L_cells,
        "n_theta": n_theta,
        "r_cells": r_cells,
        "params": params,
        "intact": intact,
        "intact_analysis": intact_an,
        "kappa_results": kappa_results,
        "kappa_spearman": kappa_spearman,
        "length_results": length_results,
        "length_spearman": length_spearman,
        "regen": regen,
        "ablation": ablation,
    }


# --------------------------------------------------------------------------- #
# 4. 15-Panel Publication Figure & Animated Growth/Regeneration GIF
# --------------------------------------------------------------------------- #
def render_outputs(
    c: float = 0.05,
    c_prime: float = 0.05,
    mu_a: float = 0.02,
    mu_h: float = 0.03,
    D_a: float = 0.015,
    D_h: float = 0.45,
    seed: int = 1972,
    skip_gif: bool = False,
    shell_length: int = SHELL_DEFAULT_LENGTH_CELLS,
    shell_steps: int = SHELL_DEFAULT_STEPS,
    kappa_a: float = THREE_SYSTEM_PARAMS["kappa_a"],
) -> dict[str, Any]:
    """Render the 15-panel comparison synthesis figure and animated growth/regeneration GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    verify_dataset_sha256()

    stab = compute_linear_stability(mu_a=mu_a, mu_h=mu_h, D_a=D_a, D_h=D_h, c=c, c_prime=c_prime)
    bisect = simulate_bisection_regeneration(seed=seed)
    ring_data = simulate_tentacle_ring_and_controls(D_a=D_a, D_h=D_h, seed=seed)
    dom_sweep = simulate_domain_scaling_sweep(D_a=D_a, D_h=D_h, seed=seed)
    spec_seg = segment_hydra_specimen(SPECIMEN_PATH)
    body_prof = measure_specimen_body_profile(spec_seg)
    shell = simulate_shell_experiments(body_prof, L_cells=shell_length, steps=shell_steps, seed=seed, kappa_a=kappa_a)

    fig = plt.figure(figsize=(16.5, 22.5), dpi=135)
    gs = fig.add_gridspec(5, 3)
    axes = np.empty((5, 3), dtype=object)
    for i in range(5):
        for j in range(3):
            if (i, j) == (3, 0):
                axes[i, j] = fig.add_subplot(gs[i, j], projection="3d")
            else:
                axes[i, j] = fig.add_subplot(gs[i, j])
    fig.patch.set_facecolor("#0b0f17")
    for ax in axes.flat:
        ax.set_facecolor("#131b29")
        ax.tick_params(colors="#cbd5e1", labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#334155")

    # Panel A1: Linear Dispersion Relation sigma(k) across d = D_h/D_a
    ax = axes[0, 0]
    ax.plot(stab["k_vals"], stab["re_sigma_super"], color="#38bdf8", lw=2.3, label=r"$d = 30.0 > d_c$ (supercritical)")
    ax.plot(stab["k_vals"], stab["re_sigma_crit"], color="#f59e0b", lw=1.8, ls="--", label=rf"$d = d_c = (\sqrt{{2}}+1)^2 \approx {stab['d_c']:.2f}$")
    ax.plot(stab["k_vals"], stab["re_sigma_equal"], color="#94a3b8", lw=1.8, ls=":", label=r"$d = 1.0$ ($D_a = D_h$ control)")
    ax.axhline(0.0, color="#64748b", lw=1.0, ls="--")
    ax.axvline(stab["k_dom"], color="#38bdf8", lw=1.2, ls=":", label=rf"$k_{{\mathrm{{dom}}}} = {stab['k_dom']:.2f}$ ($\lambda = {stab['lambda_dom']:.1f}$)")
    ax.set_title("A1. Gierer-Meinhardt Linear Dispersion $\\mathrm{Re}\\,\\sigma(k)$", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Spatial wavenumber $k$ (rad/cell)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Growth rate $\\mathrm{Re}\\,\\sigma(k)$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="lower left")

    # Panel A2: Intact Hydra Polar Body Column (Apical Hypostome Dominance, Fig. 2)
    ax = axes[0, 1]
    intact = bisect["intact"]
    ax.plot(intact["x"], intact["a"], color="#f43f5e", lw=2.5, label=f"Activator $a(x)$ (head peak={intact['a_max']:.2f})")
    ax.plot(intact["x"], intact["h"], color="#a855f7", lw=2.0, ls="--", label="Long-range inhibitor $h(x)$")
    ax.plot(intact["x"], intact["rho"] * 2.5, color="#22c55e", lw=1.4, ls=":", label=r"Source gradient $2.5\rho(x)$")
    ax.axvline(bisect["cut_index"], color="#eab308", lw=1.5, ls="-.", label=f"Surgical bisection plane ($x={bisect['cut_index']}$)")
    ax.set_title("A2. Intact Hydra Polar Column (Apical Dominance, N=40)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Body column cell $x$ (0 = apical hypostome, 39 = basal foot)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Morphogen concentration", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5)

    # Panel A3: Surgical Bisection & Autonomous Head Regeneration (Gierer & Meinhardt 1972 Fig. 3)
    ax = axes[0, 2]
    regen = bisect["regen"]
    x_frag = regen["x"] + bisect["cut_index"]
    ax.plot(x_frag, bisect["a_cut_init"], color="#94a3b8", lw=2.0, ls=":", label=f"Post-cut basal $a(x,0)$ (max={bisect['pre_cut_basal_max']:.3f})")
    ax.plot(x_frag, bisect["h_cut_init"], color="#64748b", lw=1.5, ls="--", label="Post-cut inhibitor $h(x,0)$")
    ax.plot(x_frag, regen["a"], color="#f43f5e", lw=2.5, label=f"Regenerated head $a(x,T)$ (max={regen['a_max']:.2f})")
    ax.plot(x_frag, regen["h"], color="#a855f7", lw=2.0, ls="-.", label="Regenerated inhibitor $h(x,T)$")
    ax.set_title(f"A3. Basal Fragment Head Regeneration ({bisect['recovery_ratio']*100:.1f}% Recovery)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Original body column coordinate $x \\in [20, 39]$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Morphogen concentration", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5)

    # Panel B1: Space-Time Kymograph of Bisection & Autonomous Head Regeneration
    ax = axes[1, 0]
    hist_int = intact["history_a"]
    hist_reg = regen["history_a"]
    kymo = np.zeros((len(hist_int) + len(hist_reg), intact["a"].size), dtype=float)
    kymo[: len(hist_int), :] = hist_int
    kymo[len(hist_int) :, bisect["cut_index"] :] = hist_reg
    im_k = ax.imshow(
        kymo,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, 39, 0, len(kymo) - 1],
    )
    ax.axhline(len(hist_int) - 0.5, color="#38bdf8", lw=1.6, ls="--", label="Surgical cut at $t_{\\mathrm{cut}}$")
    ax.axvline(bisect["cut_index"], color="#38bdf8", lw=1.2, ls=":")
    ax.set_title("B1. Space-Time Kymograph: Intact -> Bisection -> Regeneration", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Body column position $x$ (0..39)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Simulation snapshot index (Intact $\\to$ Regenerating)", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper right")
    cb = fig.colorbar(im_k, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#cbd5e1", labelsize=7.5)

    # Panel B2: Sub-Hypostomal Periodic Tentacle Ring vs Equal-Diffusion Control
    ax = axes[1, 1]
    r_sup = ring_data["ring_super"]
    r_ctl = ring_data["ring_control"]
    theta_deg = (r_sup["x"] / ring_data["N_ring"]) * 360.0
    ax.plot(theta_deg, r_sup["a"], color="#38bdf8", lw=2.3, label=f"Supercritical $d=30$ ({ring_data['n_tentacle_peaks']} tentacle peaks)")
    ax.plot(theta_deg, r_sup["h"], color="#a855f7", lw=1.7, ls="--", label="Ring inhibitor $h(\\theta)$")
    ax.plot(theta_deg, r_ctl["a"], color="#f59e0b", lw=2.0, ls=":", label=f"Equal-diffusion control $D_a=D_h$ (std={r_ctl['a_std']:.1e})")
    if ring_data["peak_indices"]:
        pk_idx = np.array(ring_data["peak_indices"], dtype=int)
        ax.scatter(theta_deg[pk_idx], r_sup["a"][pk_idx], color="#f43f5e", s=45, zorder=5, label="Tentacle primordia")
    ax.set_title(f"B2. Sub-Hypostomal Tentacle Ring ($N={ring_data['n_tentacle_peaks']}$ Peaks, CV={ring_data['cv_spacing']:.3f})", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Circumferential angle $\\theta$ around sub-hypostomal ring (deg)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Morphogen concentration", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="upper right")

    # Panel B3: Diffusivity-Ratio Bifurcation Sweep across d_c = (sqrt(2)+1)^2
    ax = axes[1, 2]
    d_vals = [row["d_ratio"] for row in ring_data["sweep_results"]]
    amps = [row["a_amplitude"] for row in ring_data["sweep_results"]]
    pks_d = [row["n_peaks"] for row in ring_data["sweep_results"]]
    ax.plot(d_vals, amps, "o-", color="#22c55e", lw=2.2, markersize=6, label="Activator amplitude $a_{\\max} - a_{\\min}$")
    ax.axvline(stab["d_c"], color="#f43f5e", lw=1.8, ls="--", label=rf"Analytical threshold $d_c = {stab['d_c']:.2f}$")
    for dv, amp, npk in zip(d_vals, amps, pks_d):
        ax.annotate(f"{npk}p", (dv, amp), textcoords="offset points", xytext=(0, 7), ha="center", color="#f8fafc", fontsize=8)
    ax.set_title("B3. Diffusivity-Ratio Sweep Across $d_c = (\\sqrt{2}+1)^2$", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Inhibitor / activator diffusivity ratio $d = D_h / D_a$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Steady-state activator amplitude", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8, loc="lower right")

    # Panel C1: Hydra vulgaris Specimen Segmentation (Body Axis, Hypostome & 6 Tentacles)
    ax = axes[2, 0]
    ax.imshow(spec_seg["image_rgb"])
    ax.contour(spec_seg["main_mask"], levels=[0.5], colors=["#38bdf8"], linewidths=1.1)
    hx, hy = spec_seg["hypostome_xy"]
    fx, fy = spec_seg["foot_xy"]
    ax.plot([fx, hx], [fy, hy], color="#22c55e", lw=2.4, ls="--", label="Body column axis (foot $\\to$ hypostome)")
    ax.scatter([fx], [fy], color="#eab308", s=65, marker="s", zorder=6, label="Basal disc (foot)")
    ax.scatter([hx], [hy], color="#f43f5e", s=85, marker="*", zorder=7, label="Apical hypostome center")
    t_xy = spec_seg["tentacle_coords_xy"]
    if len(t_xy) > 0:
        ax.scatter(t_xy[:, 0], t_xy[:, 1], color="#a855f7", edgecolor="#f8fafc", s=55, zorder=7, label=f"Segmented tentacles ($N={spec_seg['n_tentacles_segmented']}$)")
    ax.set_title(f"C1. Hydra vulgaris Specimen Segmentation ($N={spec_seg['n_tentacles_segmented']}$ Tentacles)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="lower right")
    ax.axis("off")

    # Panel C2: Specimen Sub-Hypostomal Angular Luminance Profile vs PDE Tentacle Ring
    ax = axes[2, 1]
    prof_norm = (spec_seg["angular_profile"] - np.min(spec_seg["angular_profile"])) / max(
        1e-6, float(np.ptp(spec_seg["angular_profile"]))
    )
    ax.plot(spec_seg["angles_deg"], prof_norm, color="#a855f7", lw=2.0, label="Specimen annulus luminance ($r \\in [60,130]$ px)")
    for tang in spec_seg["tentacle_angles_deg"]:
        ax.axvline(tang, color="#f43f5e", lw=1.2, ls="--", alpha=0.8)
    pde_shifted = theta_deg - 180.0
    pde_norm = (r_sup["a"] - np.min(r_sup["a"])) / max(1e-6, float(np.ptp(r_sup["a"])))
    ax.plot(pde_shifted, pde_norm, color="#38bdf8", lw=1.8, ls=":", label=f"PDE periodic ring $a(\\theta)$ ($N={ring_data['n_tentacle_peaks']}$ peaks)")
    ax.set_title("C2. Specimen Tentacle Crown Profile vs PDE Ring ($N=6$)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Azimuthal angle around apical hypostome (deg)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Normalized intensity / activator $a$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper right")

    # Panel C3: Allometric Domain-Length Scaling Sweep N_peaks(L)
    ax = axes[2, 2]
    L_arr = np.array(dom_sweep["domain_lengths"], dtype=float)
    n_arr = np.array(dom_sweep["peak_counts"], dtype=float)
    ax.plot(L_arr, n_arr, "o-", color="#38bdf8", lw=2.2, markersize=6.5, label="PDE activator peaks $N_{\\mathrm{peaks}}(L)$")
    ax.plot(
        L_arr,
        L_arr / dom_sweep["mean_nonlinear_wavelength"],
        "--",
        color="#f59e0b",
        lw=1.8,
        label=rf"Linear scaling $L / \bar{{\lambda}}$ ($\bar{{\lambda}}={dom_sweep['mean_nonlinear_wavelength']:.2f}$ cells)",
    )
    ax.axhline(
        spec_seg["n_tentacles_segmented"],
        color="#f43f5e",
        lw=1.3,
        ls=":",
        label=f"Hydra vulgaris specimen ($N={spec_seg['n_tentacles_segmented']}$ tentacles)",
    )
    ax.set_title(
        f"C3. Allometric Domain Scaling ($r = {dom_sweep['pearson_r']:.4f}$, RMSE = {dom_sweep['rmse_peaks']:.3f})",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlabel("Ring circumference $L$ (cells)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Number of activator peaks $N_{\\mathrm{peaks}}$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper left")

    # ------------------------------------------------------------------ #
    # Row D/E (Pass 3): specimen-shaped 3D cylindrical shell, three coupled systems
    # ------------------------------------------------------------------ #
    sh_int = shell["intact"]
    sh_an = shell["intact_analysis"]
    grid = sh_int["grid"]
    n_s_sh, n_t_sh = sh_int["a"].shape
    s_frac_sh = (np.arange(n_s_sh) + 0.5) / n_s_sh
    theta_sh = np.arange(n_t_sh) / n_t_sh * 360.0

    def composite_rgb(st: dict[str, Any]) -> np.ndarray:
        """Head activator -> red, tentacle activator -> cyan, foot activator -> green (each normalised)."""
        A = st["a"] / max(float(st["a"].max()), 1e-9)
        B = st["b"] / max(float(st["b"].max()), 1e-9)
        F = st["f"] / max(float(st["f"].max()), 1e-9)
        rgb = np.zeros(st["a"].shape + (3,), dtype=float)
        rgb[..., 0] = 0.24 + 0.76 * A
        rgb[..., 1] = 0.26 + 0.70 * B + 0.74 * F
        rgb[..., 2] = 0.36 + 0.64 * B
        return np.clip(rgb, 0.0, 1.0)

    # Panel D1: 3D surface of revolution (specimen radius profile) coloured by the three activators
    ax = axes[3, 0]
    X, Y, Z = grid.surface_xyz()
    rgb_int = composite_rgb(sh_int)
    rgb_closed = np.concatenate([rgb_int, rgb_int[:, :1, :]], axis=1)
    # body axis drawn horizontally (apex on the left) so the true 1:6 aspect fits the panel
    ax.plot_surface(float(n_s_sh) - Z, Y, X, facecolors=rgb_closed, rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)
    r_max = float(np.max(grid.r))
    ax.set_box_aspect((1.0, 2.0 * r_max / float(n_s_sh), 2.0 * r_max / float(n_s_sh)), zoom=1.55)
    ax.view_init(elev=18.0, azim=-72.0)
    ax.set_facecolor("#131b29")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#131b29")
        pane.set_edgecolor("#334155")
    ax.tick_params(colors="#cbd5e1", labelsize=6.5)
    ax.set_xlabel("apico-basal $s$ (cells, 0 = hypostome)", color="#cbd5e1", fontsize=7.5, labelpad=2)
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(
        f"D1. 3D Shell, specimen $r(s)$, $L={n_s_sh}$ (red head, cyan {sh_an['n_tentacles']} tentacles, green foot)",
        color="#f8fafc", fontsize=9.4, fontweight="bold",
    )

    # Panel D2: unrolled (s, theta) composite map with ring position markers
    ax = axes[3, 1]
    ax.imshow(np.transpose(rgb_int, (1, 0, 2)), aspect="auto", origin="lower", extent=[0.0, 1.0, 0.0, 360.0], interpolation="nearest")
    ax.axvline(sh_an["ring_offset_frac"], color="#38bdf8", lw=1.4, ls="--", label=f"PDE tentacle ring $s/L={sh_an['ring_offset_frac']:.3f}$")
    ax.axvspan(
        max(0.0, body_prof["ring_offset_frac"] - body_prof["ring_half_thickness_frac"]),
        body_prof["ring_offset_frac"] + body_prof["ring_half_thickness_frac"],
        color="#f8fafc", alpha=0.12, label=f"Specimen attachment band $0.070\\pm{body_prof['ring_half_thickness_frac']:.3f}$",
    )
    ax.axvline(body_prof["ring_offset_frac"], color="#f8fafc", lw=1.0, ls=":")
    for pk_i in sh_an["ring_peak_indices"]:
        ax.plot([sh_an["ring_offset_frac"]], [theta_sh[pk_i]], marker="o", color="#f43f5e", ms=4.5, zorder=5)
    ax.axvline(sh_an["foot_offset_frac"], color="#22c55e", lw=1.2, ls="-.", label=f"Foot $s/L={sh_an['foot_offset_frac']:.2f}$")
    ax.set_title(
        f"D2. Unrolled Shell: head CV$_\\theta$={sh_an['head_cv_theta']:.3f}, ring {sh_an['n_tentacles']} peaks (CV={sh_an['ring_spacing_cv']:.3f})",
        color="#f8fafc", fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Apico-basal position $s/L$ (0 = hypostome, 1 = basal disc)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Circumferential angle $\\theta$ (deg)", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="upper right")

    # Panel D3: axial profiles of the three activators + source density + analytic firing band
    ax = axes[3, 2]
    a_ax_n = sh_int["a"].mean(axis=1) / max(float(sh_int["a"].max()), 1e-9)
    b_ax_n = sh_int["b"].max(axis=1) / max(float(sh_int["b"].max()), 1e-9)
    f_ax_n = sh_int["f"].mean(axis=1) / max(float(sh_int["f"].max()), 1e-9)
    sig_ax = sh_int["sigma"].mean(axis=1)
    ax.plot(s_frac_sh, a_ax_n, color="#f43f5e", lw=2.2, label="Head activator $\\bar a(s)/a_{\\max}$")
    ax.plot(s_frac_sh, b_ax_n, color="#38bdf8", lw=2.2, label="Tentacle activator $\\max_\\theta b(s)/b_{\\max}$")
    ax.plot(s_frac_sh, f_ax_n, color="#22c55e", lw=2.2, label="Foot activator $\\bar f(s)/f_{\\max}$")
    ax.plot(s_frac_sh, sig_ax / max(float(sig_ax.max()), 1e-9), color="#eab308", lw=1.5, ls="--", label="Source density $\\sigma(s)/\\sigma_{\\max}$")
    ax.plot(s_frac_sh, sh_an["phi_t_axial"], color="#a855f7", lw=1.5, ls=":", label="Tentacle drive $\\phi_t(s)$")
    ax.axhline(sh_an["firing_threshold_w_c"], color="#a855f7", lw=1.0, ls="-.", label=f"Firing threshold $w_c={sh_an['firing_threshold_w_c']:.3f}$")
    pb = sh_an["predicted_ring_band"]
    if pb:
        ax.axvspan((min(pb)) / n_s_sh, (max(pb) + 1) / n_s_sh, color="#a855f7", alpha=0.10)
    ax.set_xlim(0.0, 1.0)
    ax.set_title("D3. Axial Profiles: head -> tentacle band ($\\phi_t > w_c$) -> foot", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Apico-basal position $s/L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Normalised level", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="upper right")

    # Panel E1: kappa_a sweep (head -> tentacle suppression) with the kappa_a = 0 ablation
    ax = axes[4, 0]
    ka_vals = [row["kappa_a"] for row in shell["kappa_results"]]
    ka_off = [row["ring_centroid_frac"] for row in shell["kappa_results"]]
    ka_leak = [row["apex_tentacle_leak"] for row in shell["kappa_results"]]
    ka_x = [kv if kv > 0 else 0.03 for kv in ka_vals]
    ax.plot(ka_x, ka_off, "o-", color="#38bdf8", lw=2.2, ms=6, label="Tentacle ring centroid $s/L$")
    ax.axhspan(
        max(0.0, body_prof["ring_offset_frac"] - body_prof["ring_half_thickness_frac"]),
        body_prof["ring_offset_frac"] + body_prof["ring_half_thickness_frac"],
        color="#f8fafc", alpha=0.10, label="Specimen attachment band",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Head->tentacle suppression $\\kappa_a$ (0 plotted at 0.03 = ablation)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Ring axial centroid $s/L$", color="#cbd5e1", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(ka_x, ka_leak, "s--", color="#f59e0b", lw=1.6, ms=5, label="Apex leak $b(s{=}0)/b_{\\max}$")
    ax2.set_ylabel("Tentacle activator leak into hypostome", color="#f59e0b", fontsize=9)
    ax2.tick_params(colors="#f59e0b", labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color("#334155")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="center right")
    ax.set_title(f"E1. $\\kappa_a$ Sweep: Ring Shifts Basally (Spearman $\\rho={shell['kappa_spearman']:.2f}$)", color="#f8fafc", fontsize=10.5, fontweight="bold")

    # Panel E2: body-length sweep -> tentacle number vs specimen
    ax = axes[4, 1]
    L_vals = [row["L_cells"] for row in shell["length_results"]]
    n_vals = [row["n_tentacles"] for row in shell["length_results"]]
    ax.plot(L_vals, n_vals, "o-", color="#38bdf8", lw=2.2, ms=6.5, label="Shell tentacle primordia $N(L)$")
    ax.axhline(body_prof["n_tentacles"], color="#f43f5e", lw=1.3, ls=":", label=f"Specimen ($N={body_prof['n_tentacles']}$)")
    ax.axvline(shell["L_cells"], color="#eab308", lw=1.2, ls="--", label=f"Default $L={shell['L_cells']}$ (spacing/L calibrated)")
    for Lv, nv, row in zip(L_vals, n_vals, shell["length_results"]):
        ax.annotate(f"{row['tentacle_spacing_cells']:.1f}c", (Lv, nv), textcoords="offset points", xytext=(0, 7), ha="center", color="#f8fafc", fontsize=7.5)
    ax.set_title(f"E2. Tentacle Number vs Body Size (Spearman $\\rho={shell['length_spearman']:.2f}$)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Shell body length $L$ (cells); labels = tentacle spacing", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Tentacle primordia on the ring", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper left")

    # Panel E3: shell amputation & regeneration (unrolled fragment + timeline inset)
    ax = axes[4, 2]
    rg = shell["regen"]
    rg_st = rg["regen"]
    rgb_rg = composite_rgb(rg_st)
    n_frag = rg_st["a"].shape[0]
    ax.imshow(np.transpose(rgb_rg, (1, 0, 2)), aspect="auto", origin="lower", extent=[0.0, 1.0, 0.0, 360.0], interpolation="nearest")
    rg_an = rg["analysis"]
    ax.axvline(rg_an["ring_offset_frac"], color="#38bdf8", lw=1.4, ls="--", label=f"Regenerated ring $s/L_{{frag}}={rg_an['ring_offset_frac']:.3f}$")
    for pk_i in rg_an["ring_peak_indices"]:
        ax.plot([rg_an["ring_offset_frac"]], [theta_sh[pk_i]], marker="o", color="#f43f5e", ms=4.5, zorder=5)
    t_head_txt = f"{rg['t_head']:.0f}" if rg["t_head"] is not None else "n/a"
    t_ring_txt = f"{rg['t_ring']:.0f}" if rg["t_ring"] is not None else "n/a"
    ax.set_title(
        f"E3. Shell Regeneration: head {rg['head_recovery_ratio']*100:.0f}% at tip, $t_{{head}}={t_head_txt} < t_{{ring}}={t_ring_txt}$",
        color="#f8fafc", fontsize=9.6, fontweight="bold",
    )
    ax.set_xlabel(f"Fragment position $s/L_{{frag}}$ (cut at $s = L/2$, $L_{{frag}}={n_frag}$)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Circumferential angle $\\theta$ (deg)", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="upper right")
    ins = ax.inset_axes([0.42, 0.08, 0.55, 0.36])
    ins.set_facecolor("#0f172a")
    t_hist = np.array(rg_st["history_t"])
    a_tip_hist = np.array([float(s["a"][0].mean()) for s in rg_st["history"]])
    b_ring_hist = np.array([float(s["b"][1:].max()) for s in rg_st["history"]])
    ins.plot(t_hist, a_tip_hist / max(a_tip_hist.max(), 1e-9), color="#f43f5e", lw=1.6, label="$a$ at tip")
    ins.plot(t_hist, b_ring_hist / max(b_ring_hist.max(), 1e-9), color="#38bdf8", lw=1.6, label="$b$ ring")
    ins.tick_params(colors="#cbd5e1", labelsize=6.5)
    for sp in ins.spines.values():
        sp.set_color("#334155")
    ins.set_xlabel("time after cut", color="#cbd5e1", fontsize=7)
    ins.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=6.5, loc="lower right")

    fig.suptitle(
        "Gierer & Meinhardt (1972) / Meinhardt (1993, 2012) Hydra vulgaris Activator-Inhibitor Morphogenesis\n"
        "Regeneration, Tentacle Ring & Coupled Three-System Model on a Specimen-Shaped 3D Shell (Pass 3)",
        color="#f8fafc",
        fontsize=12.5,
        fontweight="bold",
        y=0.996,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.978])
    fig.savefig(OUT_PNG, dpi=135)
    plt.close(fig)

    # Render animated GIF showing Phase 1 (Intact Polar Formation) -> Phase 2 (Bisection & Head Regeneration)
    if not skip_gif:
        frames_pil: list[Image.Image] = []
        hist_int_a = intact["history_a"]
        hist_int_h = intact["history_h"]
        hist_reg_a = regen["history_a"]
        hist_reg_h = regen["history_h"]
        y_max = max(float(np.max(hist_int_a)), float(np.max(hist_int_h))) * 1.08

        # Sample 18 frames from intact formation and 18 frames from post-bisection regeneration
        int_indices = np.linspace(0, len(hist_int_a) - 1, 18, dtype=int)
        reg_indices = np.linspace(0, len(hist_reg_a) - 1, 18, dtype=int)

        for idx in int_indices:
            f_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=90)
            f_fig.patch.set_facecolor("#0b0f17")
            for a_sub in (ax1, ax2):
                a_sub.set_facecolor("#131b29")
                a_sub.tick_params(colors="#cbd5e1", labelsize=8)
                for sp in a_sub.spines.values():
                    sp.set_color("#334155")
            ax1.plot(intact["x"], hist_int_a[idx], color="#f43f5e", lw=2.4, label="Activator $a(x,t)$")
            ax1.plot(intact["x"], hist_int_h[idx], color="#a855f7", lw=1.9, ls="--", label="Inhibitor $h(x,t)$")
            ax1.axvline(bisect["cut_index"], color="#eab308", lw=1.2, ls=":")
            ax1.set_ylim(0, y_max)
            ax1.set_xlim(0, 39)
            ax1.set_title(f"Phase 1: Intact Apical Polarity (t={intact['history_t'][idx]:.0f})", color="#f8fafc", fontsize=9.5)
            ax1.set_xlabel("Body column cell $x$ (0=head, 39=foot)", color="#cbd5e1", fontsize=8.5)
            ax1.set_ylabel("Concentration", color="#cbd5e1", fontsize=8.5)
            ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5)

            r_idx = min(idx, len(ring_data["ring_super"]["history_a"]) - 1)
            ax2.plot(theta_deg, ring_data["ring_super"]["history_a"][r_idx], color="#38bdf8", lw=2.2, label="Ring activator $a(\\theta,t)$")
            ax2.plot(theta_deg, ring_data["ring_control"]["history_a"][r_idx], color="#f59e0b", lw=1.6, ls=":", label="Control $D_a=D_h$")
            ax2.set_ylim(0, 4.5)
            ax2.set_xlim(0, 360)
            ax2.set_title("Sub-Hypostomal 6-Tentacle Ring Formation", color="#f8fafc", fontsize=9.5)
            ax2.set_xlabel("Ring angle $\\theta$ (deg)", color="#cbd5e1", fontsize=8.5)
            ax2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5)
            f_fig.tight_layout()
            f_fig.canvas.draw()
            rgba = np.asarray(f_fig.canvas.buffer_rgba())
            frames_pil.append(Image.fromarray(rgba[:, :, :3]))
            plt.close(f_fig)

        for idx in reg_indices:
            f_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=90)
            f_fig.patch.set_facecolor("#0b0f17")
            for a_sub in (ax1, ax2):
                a_sub.set_facecolor("#131b29")
                a_sub.tick_params(colors="#cbd5e1", labelsize=8)
                for sp in a_sub.spines.values():
                    sp.set_color("#334155")
            ax1.plot(x_frag, hist_reg_a[idx], color="#f43f5e", lw=2.4, label="Regenerating $a(x,t)$")
            ax1.plot(x_frag, hist_reg_h[idx], color="#a855f7", lw=1.9, ls="--", label="Inhibitor $h(x,t)$")
            ax1.plot(x_frag, bisect["a_cut_init"], color="#94a3b8", lw=1.4, ls=":", label="Pre-regeneration $a(x,0)$")
            ax1.set_ylim(0, y_max)
            ax1.set_xlim(0, 39)
            ax1.set_title(f"Phase 2: Basal Fragment Regeneration (t=+{regen['history_t'][idx]:.0f})", color="#f8fafc", fontsize=9.5)
            ax1.set_xlabel("Body column cell $x \\in [20, 39]$ (amputated at $x=20$)", color="#cbd5e1", fontsize=8.5)
            ax1.set_ylabel("Concentration", color="#cbd5e1", fontsize=8.5)
            ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5)

            ax2.plot(theta_deg, ring_data["ring_super"]["a"], color="#38bdf8", lw=2.2, label=f"6 Tentacle peaks (CV={ring_data['cv_spacing']:.3f})")
            ax2.plot(theta_deg, ring_data["ring_control"]["a"], color="#f59e0b", lw=1.6, ls=":", label="Control $D_a=D_h$ (0 peaks)")
            ax2.set_ylim(0, 4.5)
            ax2.set_xlim(0, 360)
            ax2.set_title("Steady-State 6-Tentacle Whorl vs Control", color="#f8fafc", fontsize=9.5)
            ax2.set_xlabel("Ring angle $\\theta$ (deg)", color="#cbd5e1", fontsize=8.5)
            ax2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5)
            f_fig.tight_layout()
            f_fig.canvas.draw()
            rgba = np.asarray(f_fig.canvas.buffer_rgba())
            frames_pil.append(Image.fromarray(rgba[:, :, :3]))
            plt.close(f_fig)

        # Phase 3 (Pass 3): unrolled specimen-shaped shell developing head cap, tentacle ring and foot,
        # then the amputated basal fragment regenerating a hypostome and a new ring below it.
        sh_hist = sh_int["history"]
        rg_hist = rg_st["history"]
        sh_idx = np.linspace(0, len(sh_hist) - 1, min(12, len(sh_hist)), dtype=int)
        rg_idx = np.linspace(0, len(rg_hist) - 1, min(12, len(rg_hist)), dtype=int)
        for phase, hist, idxs, t_arr, title in (
            ("intact", sh_hist, sh_idx, sh_int["history_t"], "Phase 3: 3D Shell — Head / Tentacle Ring / Foot"),
            ("regen", rg_hist, rg_idx, rg_st["history_t"], "Phase 3b: Fragment Regeneration on the Shell"),
        ):
            for idx in idxs:
                f_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=90)
                f_fig.patch.set_facecolor("#0b0f17")
                for a_sub in (ax1, ax2):
                    a_sub.set_facecolor("#131b29")
                    a_sub.tick_params(colors="#cbd5e1", labelsize=8)
                    for sp in a_sub.spines.values():
                        sp.set_color("#334155")
                snap = hist[idx]
                rgb_s = composite_rgb(snap)
                ax1.imshow(np.transpose(rgb_s, (1, 0, 2)), aspect="auto", origin="lower", extent=[0.0, 1.0, 0.0, 360.0], interpolation="nearest")
                ax1.set_title(f"{title} (t={t_arr[idx]:.0f})", color="#f8fafc", fontsize=9.0)
                ax1.set_xlabel("$s/L$ (0 = apex)", color="#cbd5e1", fontsize=8.5)
                ax1.set_ylabel("$\\theta$ (deg)", color="#cbd5e1", fontsize=8.5)
                n_rows = snap["a"].shape[0]
                s_fr = (np.arange(n_rows) + 0.5) / n_rows
                ax2.plot(s_fr, snap["a"].mean(axis=1) / max(float(sh_int["a"].max()), 1e-9), color="#f43f5e", lw=2.0, label="head $\\bar a$")
                ax2.plot(s_fr, snap["b"].max(axis=1) / max(float(sh_int["b"].max()), 1e-9), color="#38bdf8", lw=2.0, label="tentacle $\\max_\\theta b$")
                ax2.plot(s_fr, snap["f"].mean(axis=1) / max(float(sh_int["f"].max()), 1e-9), color="#22c55e", lw=2.0, label="foot $\\bar f$")
                ax2.set_ylim(0, 1.15)
                ax2.set_xlim(0, 1)
                ax2.set_title("Axial activator profiles (normalised to intact maxima)", color="#f8fafc", fontsize=9.0)
                ax2.set_xlabel("$s/L$", color="#cbd5e1", fontsize=8.5)
                ax2.legend(loc="upper center", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5)
                f_fig.tight_layout()
                f_fig.canvas.draw()
                rgba = np.asarray(f_fig.canvas.buffer_rgba())
                frames_pil.append(Image.fromarray(rgba[:, :, :3]))
                plt.close(f_fig)

        frames_pil[0].save(
            OUT_GIF,
            save_all=True,
            append_images=frames_pil[1:],
            duration=85,
            loop=0,
        )

    return {
        "stability": stab,
        "bisection": bisect,
        "ring": ring_data,
        "domain_sweep": dom_sweep,
        "specimen": spec_seg,
        "body_profile": body_prof,
        "shell": shell,
    }


# --------------------------------------------------------------------------- #
# 5. MorphoEval Sidecar Quantitative Measurements (`measure_morphoeval()`)
# --------------------------------------------------------------------------- #
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute quantitative morphometric measurements for `morphoeval/measure_outputs.py`."""
    stab = compute_linear_stability()
    bisect = simulate_bisection_regeneration(seed=1972)
    ring_data = simulate_tentacle_ring_and_controls(seed=1972)
    dom_sweep = simulate_domain_scaling_sweep(seed=1972)
    spec_seg = segment_hydra_specimen(SPECIMEN_PATH)
    body_prof = measure_specimen_body_profile(spec_seg)
    shell = simulate_shell_experiments(body_prof, seed=1972)
    sh_an = shell["intact_analysis"]
    rg = shell["regen"]
    abl = shell["ablation"]
    abl_leak = abl["apex_tentacle_leak"] if abl is not None else float("nan")
    abl_off = abl["ring_centroid_frac"] if abl is not None else float("nan")
    L_counts = [row["n_tentacles"] for row in shell["length_results"]]
    L_vals = [row["L_cells"] for row in shell["length_results"]]
    pb = sh_an["predicted_ring_band"]
    band_pred_lo = (min(pb) if pb else -1)
    band_pred_hi = (max(pb) if pb else -1)
    ring_rows = sh_an["ring_rows"]
    band_inside = bool(pb) and min(ring_rows) >= band_pred_lo and max(ring_rows) <= band_pred_hi
    shell_measurements = [
        {
            "name": "3D shell hypostome cap azimuthal coefficient of variation CV_theta(a) at the apex",
            "value": round(sh_an["head_cv_theta"], 4),
            "unit": "CV (std/mean over theta)",
            "method": f"Coupled head/tentacle/foot loops on the specimen-shaped surface of revolution (L={shell['L_cells']} cells, N_theta={shell['n_theta']}); CV of a over theta at the row of maximal azimuthal-mean a",
            "reference_value": 0.0,
            "reference_source": "Meinhardt (1993) Dev. Biol. 157:321-333 & Meinhardt (2012) Int. J. Dev. Biol. 56:447-462: single azimuthally symmetric hypostome at the apex",
            "note": f"head row s={sh_an['i_head']} (apex), a_max={sh_an['a_max']:.2f}, {sh_an['n_head_peaks']} axial head peak(s), sigma_apex={sh_an['sigma_apex']:.2f}; saturating autocatalysis kappa_h={THREE_SYSTEM_PARAMS['kappa_head']} gives the cap instead of a spot",
            "derived_from": "script",
        },
        {
            "name": "3D shell sub-hypostomal tentacle ring axial offset s_ring / L vs specimen attachment offset",
            "value": round(sh_an["ring_offset_frac"], 4),
            "unit": "fraction of body length (0 = hypostome tip)",
            "method": "Median row of the band where max_theta b > 0.5 b_max on the intact shell; specimen: hypostome-tip -> tentacle-attachment distance / body length measured on the vendored photograph (measure_specimen_body_profile)",
            "reference_value": round(body_prof["ring_offset_frac"], 4),
            "reference_source": "Hydra vulgaris specimen (CC BY-SA 3.0) body profile: attachment band 0.070 +- 0.081 (half-thickness = attachment radius / body length); Meinhardt (1993) 'tentacles form at the border of the hypostome'",
            "note": f"prediction (not calibrated): ring rows {min(ring_rows)}-{max(ring_rows)} of {shell['L_cells']} (width {sh_an['ring_width_rows']} rows), centroid {sh_an['ring_centroid_frac']:.3f}; specimen band [{max(0.0, body_prof['ring_offset_frac'] - body_prof['ring_half_thickness_frac']):.3f}, {body_prof['ring_offset_frac'] + body_prof['ring_half_thickness_frac']:.3f}]; analytic firing band phi_t > w_c = {sh_an['firing_threshold_w_c']:.4f} is rows {band_pred_lo}-{band_pred_hi} (simulated band inside: {band_inside})",
            "derived_from": "script",
        },
        {
            "name": "3D shell tentacle primordia count on the ring vs Hydra vulgaris specimen",
            "value": int(sh_an["n_tentacles"]),
            "unit": "tentacle peaks",
            "method": f"Periodic peak detection on the ring-averaged theta profile of b (circumference {sh_an['ring_circumference_cells']:.1f} cells at the ring row)",
            "reference_value": int(body_prof["n_tentacles"]),
            "reference_source": "Hydra vulgaris specimen (6 segmented tentacles); Meinhardt (1993) Dev. Biol. 157:321-333",
            "note": f"spacing {sh_an['tentacle_spacing_cells']:.2f} cells (CV {sh_an['ring_spacing_cv']:.3f}) = {sh_an['tentacle_spacing_frac']:.4f} L vs specimen spacing/L {body_prof['tentacle_spacing_frac']:.4f} (the body length L={shell['L_cells']} was CALIBRATED so these two ratios agree; the count itself is then a geometric consequence, 7 vs 6 = +1); body-length sweep L={L_vals} gives N={L_counts}",
            "derived_from": "script",
        },
        {
            "name": "3D shell foot activator position and head-foot axial separation",
            "value": round(sh_an["head_foot_separation_frac"], 4),
            "unit": "fraction of body length",
            "method": "Row of maximal azimuthal-mean foot activator f minus head row, divided by L; foot drive phi_f = 1/(1+(sigma/sigma_f)^2) fires where the source density is lowest",
            "reference_value": round((shell["L_cells"] - 1) / shell["L_cells"], 4),
            "reference_source": "Meinhardt (2012) Int. J. Dev. Biol. 56:447-462: foot forms at the position of lowest source density opposite the head",
            "note": f"foot row s={sh_an['i_foot']} of {shell['L_cells']} (basal disc), f_max={sh_an['f_max']:.2f}, foot CV_theta={sh_an['foot_cv_theta']:.4f} (uniform basal disc)",
            "derived_from": "script",
        },
        {
            "name": "Ablation kappa_a = 0 (no head->tentacle suppression): tentacle-activator leak into the hypostome",
            "value": round(abl_leak, 4),
            "unit": "b(s=0) / b_max",
            "method": "Same shell, same seed, kappa_a = 0 vs default kappa_a = 1.0; apex-row tentacle activator relative to the ring maximum",
            "reference_value": round(sh_an["apex_tentacle_leak"], 4),
            "reference_source": "Meinhardt (1993) Dev. Biol. 157:321-333: local inhibition of tentacle activation by the head activator excludes tentacles from the hypostome",
            "note": f"leak {abl_leak:.4f} without suppression vs {sh_an['apex_tentacle_leak']:.4f} with (x{abl_leak / max(sh_an['apex_tentacle_leak'], 1e-9):.1f}); ring centroid shifts {abl_off:.3f} -> {sh_an['ring_centroid_frac']:.3f} L; kappa_a sweep {SHELL_KAPPA_A_SWEEP} ring centroids {[round(r['ring_centroid_frac'], 3) for r in shell['kappa_results']]} (Spearman rho = {shell['kappa_spearman']:.2f})",
            "derived_from": "script",
        },
        {
            "name": "Body-length sweep: Spearman monotonicity of tentacle number with shell size",
            "value": round(shell["length_spearman"], 4),
            "unit": "Spearman rho",
            "method": f"Intact shells at L in {L_vals} cells (specimen radius profile rescaled), tentacle primordia counted on the ring",
            "reference_value": 1.0,
            "reference_source": "Model prediction consistent with size-dependent tentacle number in Hydra (Meinhardt 1993 discussion); not quantitatively validated against a size series",
            "note": f"N(L) = {L_counts}; intrinsic spacing ~{np.mean([row['tentacle_spacing_cells'] for row in shell['length_results'][1:]]):.1f} cells set by the tentacle-inhibitor range l_c = sqrt(D_c/mu_h) = {math.sqrt(THREE_SYSTEM_PARAMS['D_c'] / THREE_SYSTEM_PARAMS['mu_h']):.2f} cells",
            "derived_from": "script",
        },
        {
            "name": "3D shell amputation (s = L/2): hypostome recovery at the fragment tip and head-before-ring ordering",
            "value": round(rg["head_recovery_ratio"], 4),
            "unit": "a_regen_max / a_intact_max",
            "method": "Basal half of the intact shell continued with inherited source density; t_head = first time a(tip) > 0.5 final, t_ring = first time max b (rows >= 1) > 0.5 final",
            "reference_value": 1.0,
            "reference_source": "Meinhardt (1993) Dev. Biol. 157:321-333 & (2012): fragments regenerate a hypostome at the apical cut and a new tentacle ring directly below it",
            "note": f"pre-cut tip a={rg['pre_cut_tip_a']:.3f} -> regenerated a_max={rg['regen']['a'].max():.2f} at the tip row; t_head={rg['t_head']} < t_ring={rg['t_ring']} (time units); regenerated ring at s/L_frag={rg['analysis']['ring_offset_frac']:.3f} with {rg['analysis']['n_tentacles']} peaks (b_max {rg['analysis']['b_max']:.2f}; fragment ring circumference {rg['analysis']['ring_circumference_cells']:.0f} cells — the fragment keeps the mid-body radius, so the count is not comparable to the intact ring). LIMITATIONS: on the wide un-remodelled cut surface the new head is a localised spot (CV_theta = {rg['analysis']['head_cv_theta']:.2f}, not a uniform cap: no wound closure / tissue remodelling in the fixed geometry), and Meinhardt's tentacle-before-hypostome prediction is NOT reproduced (see remediation)",
            "derived_from": "script",
        },
    ]

    return [
        {
            "name": "Gierer-Meinhardt critical diffusivity ratio d_c = (sqrt(2)+1)^2",
            "value": round(stab["d_c"], 4),
            "unit": "D_h / D_a",
            "method": "Exact analytical Turing-Gierer-Meinhardt threshold (3 + 2*sqrt(2)) at mu_h > mu_a",
            "reference_value": 5.8284,
            "reference_source": "Gierer & Meinhardt (1972) Kybernetik 12(1):30-39 Eqs. (15a)-(15b)",
            "note": f"canonical d = {stab['d_ratio']:.1f} > d_c (max Re sigma = +{stab['max_sigma_super']:.5f}); equal-diffusion control d = 1.0 has max Re sigma = {stab['max_sigma_equal']:.5f}",
            "derived_from": "script",
        },
        {
            "name": "Homogeneous steady-state activator concentration a_* (rho_0 = 0)",
            "value": round(stab["a_star"], 4),
            "unit": "concentration units",
            "method": "Exact steady-state formula a_* = (c / c_prime) * (mu_h / mu_a) with h_* = (mu_h / mu_a)^2 = 2.25",
            "reference_value": 1.5,
            "reference_source": "Gierer & Meinhardt (1972) Kybernetik 12(1):30-39 Eq. (15)",
            "note": f"homogeneous Jacobian tr(J_0) = {stab['trace_J0']:.4f} < 0, det(J_0) = {stab['det_J0']:.5f} > 0",
            "derived_from": "script",
        },
        {
            "name": "Intact Hydra polar column apical hypostome peak cell index (N=40)",
            "value": int(bisect["intact"]["apical_max_idx"]),
            "unit": "cell index (0=apical head)",
            "method": "argmax_x a(x, T) on 40-cell polar column (D_a=0.03, D_h=6.0, source slope=-0.20)",
            "reference_value": 0,
            "reference_source": "Gierer & Meinhardt (1972) Kybernetik 12(1):30-39 Fig. 2 (apical dominance)",
            "note": f"intact apical peak a_max = {bisect['intact']['a_max']:.3f} vs suppressed basal half max = {bisect['pre_cut_basal_max']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Surgical bisection basal-fragment head regeneration recovery ratio",
            "value": round(bisect["recovery_ratio"], 4),
            "unit": "a_regen_max / a_intact_max",
            "method": "Amputate apical half at x=20 and evolve basal fragment x in [20,39] under zero-flux Neumann BCs",
            "reference_value": 0.875,
            "reference_source": "Gierer & Meinhardt (1972) Kybernetik 12(1):30-39 Fig. 3 (autonomous head regeneration)",
            "note": f"basal wound edge amplifies from pre-cut a={bisect['pre_cut_basal_max']:.4f} to regenerated head peak a={bisect['regen']['a_max']:.3f} ({bisect['suppression_amplification']:.1f}x amplification at new apical index {bisect['regen']['apical_max_idx']})",
            "derived_from": "script",
        },
        {
            "name": "Sub-hypostomal periodic ring tentacle primordia count vs Hydra vulgaris specimen",
            "value": int(ring_data["n_tentacle_peaks"]),
            "unit": "tentacle peaks",
            "method": "Circular periodic Gierer-Meinhardt ring PDE (N_ring=64, D_a=0.015, D_h=0.45) vs specimen annulus segmentation",
            "reference_value": int(spec_seg["n_tentacles_segmented"]),
            "reference_source": "Meinhardt (1993) Dev. Biol. 157:321-333 & Hydra vulgaris specimen (CC BY-SA 3.0, 6 tentacles)",
            "note": f"PDE inter-peak spacing = {ring_data['mean_spacing_cells']:.2f} ± {ring_data['std_spacing_cells']:.2f} cells (CV = {ring_data['cv_spacing']:.4f}); specimen segmented {spec_seg['n_tentacles_segmented']} tentacles with mean angular gap {spec_seg['mean_tentacle_gap_deg']:.1f} deg (CV = {spec_seg['cv_tentacle_gap']:.4f})",
            "derived_from": "script",
        },
        {
            "name": "Equal-diffusion negative control (D_a = D_h = 0.015) activator spatial std",
            "value": round(ring_data["ring_control"]["a_std"], 6),
            "unit": "concentration std",
            "method": "Ablation control with D_a = D_h = 0.015 (d = 1.0 < d_c = 5.8284) from identical initial perturbation",
            "reference_value": 0.0,
            "reference_source": "Gierer & Meinhardt (1972) lateral inhibition necessity theorem (Lean 4 gierer_meinhardt_equal_diffusion_stable)",
            "note": f"0 peaks formed in equal-diffusion control (std = {ring_data['ring_control']['a_std']:.2e}) vs {ring_data['n_tentacle_peaks']} peaks in supercritical ring (std = {ring_data['ring_super']['a_std']:.3f})",
            "derived_from": "script",
        },
        {
            "name": "Allometric domain-length scaling correlation r(L, N_peaks)",
            "value": round(dom_sweep["pearson_r"], 4),
            "unit": "Pearson r",
            "method": "Linear correlation between ring circumference L in [22..86] cells and steady-state peak count N_peaks(L)",
            "reference_value": 1.0,
            "reference_source": "Gierer & Meinhardt (1972) / Meinhardt (2012) Int. J. Dev. Biol. 56:447-462 wavelength invariance",
            "note": f"peak counts {dom_sweep['peak_counts']} across L={dom_sweep['domain_lengths']}; mean nonlinear wavelength = {dom_sweep['mean_nonlinear_wavelength']:.2f} cells, RMSE = {dom_sweep['rmse_peaks']:.4f} peaks",
            "derived_from": "script",
        },
    ] + shell_measurements


# --------------------------------------------------------------------------- #
# 6. Comprehensive Deterministic Self-Test Suite (`--self-test`)
# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Run 14 deterministic verification checks covering SHA-256 locks, theory, PDE regimes,
    specimen segmentation, the Pass 3 three-system cylindrical shell, and bitwise reproducibility."""
    # 1. Verify SHA-256 digests of all vendored dataset files
    digests = verify_dataset_sha256()
    ref = load_reference()
    assert len(digests) == 4, "Expected 4 verified dataset files"

    # 2. Verify JSON-to-code parameter and algebraic parity
    canon = ref["canonical_parameters"]
    stab = compute_linear_stability(
        mu_a=canon["mu_a"],
        mu_h=canon["mu_h"],
        D_a=canon["D_a"],
        D_h=canon["D_h"],
        c=canon["c"],
        c_prime=canon["c_prime"],
    )
    assert abs(stab["d_c"] - canon["critical_diffusivity_ratio_dc"]) < 1e-12, "d_c parity failure"
    assert abs(stab["a_star"] - canon["homogeneous_a_star_rho0_zero"]) < 1e-12, "a_* parity failure"
    assert abs(stab["h_star"] - canon["homogeneous_h_star_rho0_zero"]) < 1e-12, "h_* parity failure"
    assert stab["trace_J0"] < 0.0 and stab["det_J0"] > 0.0, "Homogeneous Jacobian must be stable"
    assert stab["max_sigma_super"] > 0.0 and stab["max_sigma_equal"] < 0.0, "Dispersion sign check failed"

    # 3. Verify intact polar column apical dominance & surgical bisection head regeneration
    bisect = simulate_bisection_regeneration(seed=1972)
    assert bisect["intact"]["apical_max_idx"] == 0, "Intact Hydra must form apical peak at x=0"
    assert bisect["pre_cut_basal_max"] < 0.15, "Basal half must be laterally inhibited prior to cut"
    assert bisect["regen"]["apical_max_idx"] == 0, "Amputated basal fragment must regenerate head at new apical tip x=0"
    assert bisect["recovery_ratio"] >= 0.85, f"Expected >=85% head peak recovery, got {bisect['recovery_ratio']:.3f}"

    # 4. Verify periodic tentacle ring (6 peaks) & equal-diffusion negative control (0 peaks)
    ring_data = simulate_tentacle_ring_and_controls(seed=1972)
    assert ring_data["n_tentacle_peaks"] == 6, f"Expected 6 tentacle ring peaks, got {ring_data['n_tentacle_peaks']}"
    assert ring_data["cv_spacing"] < 0.10, f"Expected regular tentacle spacing CV < 0.10, got {ring_data['cv_spacing']:.4f}"
    assert ring_data["ring_control"]["n_peaks"] == 0, "Equal-diffusion control must form 0 peaks"
    assert ring_data["ring_control"]["a_std"] < 1e-4, "Equal-diffusion control must decay to uniform state"

    # 5. Verify diffusivity-ratio sweep across d_c = 5.8284
    subcrit = [r for r in ring_data["sweep_results"] if r["d_ratio"] < stab["d_c"]]
    supercrit = [r for r in ring_data["sweep_results"] if r["d_ratio"] > stab["d_c"]]
    assert all(r["n_peaks"] == 0 and r["a_std"] < 1e-3 for r in subcrit), "Subcritical d < d_c must not pattern"
    assert all(r["n_peaks"] >= 4 and r["a_amplitude"] > 1.0 for r in supercrit), "Supercritical d > d_c must pattern"

    # 6. Verify allometric domain-length scaling
    dom_sweep = simulate_domain_scaling_sweep(seed=1972)
    assert dom_sweep["pearson_r"] >= 0.97, f"Expected domain scaling r >= 0.97, got {dom_sweep['pearson_r']:.4f}"

    # 7. Verify Hydra vulgaris specimen segmentation (6 tentacles)
    spec_seg = segment_hydra_specimen(SPECIMEN_PATH)
    assert spec_seg["n_tentacles_segmented"] == 6, (
        f"Expected 6 segmented tentacles on Hydra vulgaris specimen, got {spec_seg['n_tentacles_segmented']}"
    )
    assert spec_seg["stalk_length_px"] > 150.0, "Invalid segmented Hydra stalk length"

    # 8. Verify bitwise two-run determinism
    run1 = simulate_gierer_meinhardt_1d(N=32, steps=1000, dt=0.2, D_a=0.015, D_h=0.45, bc="periodic", seed=1972)
    run2 = simulate_gierer_meinhardt_1d(N=32, steps=1000, dt=0.2, D_a=0.015, D_h=0.45, bc="periodic", seed=1972)
    assert np.allclose(run1["a"], run2["a"], atol=0.0, rtol=0.0), "Bitwise determinism check failed"

    # 9. Reference-JSON parity for the 1D regimes and the Pass 3 three-system block
    regimes = ref["hydra_regimes_and_experiments"]
    assert regimes["intact_polar_column"]["N_cells"] == 40 and abs(regimes["intact_polar_column"]["source_gradient_slope"] + 0.20) < 1e-12, "intact column JSON parity"
    assert regimes["bisection_regeneration"]["N_cells_fragment"] == 20 and bisect["recovery_ratio"] >= regimes["bisection_regeneration"]["peak_recovery_ratio_min"], "bisection JSON parity"
    assert regimes["tentacle_ring_periodic"]["ring_cells"] == 64 and regimes["tentacle_ring_periodic"]["expected_tentacle_peaks"] == ring_data["n_tentacle_peaks"], "ring JSON parity"
    assert regimes["allometric_domain_scaling"]["domain_lengths_cells"] == dom_sweep["domain_lengths"], "domain-scaling JSON parity"
    assert regimes["allometric_domain_scaling"]["expected_peak_counts"] == dom_sweep["peak_counts"], "domain-scaling peak-count parity"
    shell_ref = ref["three_system_cylindrical_shell"]
    for key, val in shell_ref["parameters"].items():
        assert key in THREE_SYSTEM_PARAMS and abs(THREE_SYSTEM_PARAMS[key] - val) < 1e-12, f"three-system parameter parity failure for {key}"
    assert shell_ref["body_length_cells"] == SHELL_DEFAULT_LENGTH_CELLS and shell_ref["n_theta"] == SHELL_DEFAULT_NTHETA, "shell geometry JSON parity"
    w_c = three_system_firing_threshold(
        THREE_SYSTEM_PARAMS["mu_a"], THREE_SYSTEM_PARAMS["c"], THREE_SYSTEM_PARAMS["mu_h"], THREE_SYSTEM_PARAMS["rho_b"], THREE_SYSTEM_PARAMS["rho_c"]
    )
    assert abs(w_c - shell_ref["tentacle_firing_threshold_w_c"]) < 1e-9, "firing threshold JSON parity"
    # analytic off-state discriminant: non-negative at w_c, negative just above, antitone in w (Lean theorem mirror)
    disc_args = (THREE_SYSTEM_PARAMS["mu_a"], THREE_SYSTEM_PARAMS["c"], THREE_SYSTEM_PARAMS["mu_h"], THREE_SYSTEM_PARAMS["rho_b"], THREE_SYSTEM_PARAMS["rho_c"])
    assert abs(three_system_off_state_discriminant(w_c, *disc_args)) < 1e-12, "discriminant must vanish at w_c"
    assert three_system_off_state_discriminant(1.05 * w_c, *disc_args) < 0.0 < three_system_off_state_discriminant(0.95 * w_c, *disc_args), "discriminant sign across w_c"
    w_grid = np.linspace(0.0, 1.0, 11)
    disc_grid = [three_system_off_state_discriminant(w, *disc_args) for w in w_grid]
    assert all(d2 <= d1 for d1, d2 in zip(disc_grid[:-1], disc_grid[1:])), "discriminant must be antitone in the drive"

    # 10. Specimen body profile used by the shell (tip, basal disc, radius profile, ring attachment)
    body_prof = measure_specimen_body_profile(spec_seg)
    assert body_prof["body_length_px"] > spec_seg["stalk_length_px"] > 150.0, "full body length must exceed the stalk-core length"
    assert 0.0 < body_prof["ring_offset_frac"] < 0.2, f"specimen ring attachment must be apical, got {body_prof['ring_offset_frac']:.3f}"
    assert np.isfinite(body_prof["r_attach_px"]) and 10.0 < body_prof["r_attach_px"] < 60.0, "tentacle attachment radius out of range"
    assert 0.05 < body_prof["max_radius_frac"] < 0.3, "specimen radius/length profile out of range"

    # 11. Intact three-system shell: uniform apical hypostome, one sub-hypostomal tentacle ring, basal foot
    shell = simulate_shell_experiments(body_prof, seed=1972)
    an = shell["intact_analysis"]
    assert an["i_head"] == 0 and an["n_head_peaks"] == 1, "shell must form a single head at the apex"
    assert an["head_cv_theta"] < 0.05, f"hypostome cap must be azimuthally uniform, CV={an['head_cv_theta']:.4f}"
    lo = max(0.0, body_prof["ring_offset_frac"] - body_prof["ring_half_thickness_frac"])
    hi = body_prof["ring_offset_frac"] + body_prof["ring_half_thickness_frac"]
    assert lo <= an["ring_offset_frac"] <= hi, f"tentacle ring offset {an['ring_offset_frac']:.3f} outside specimen band [{lo:.3f}, {hi:.3f}]"
    assert an["ring_width_rows"] <= 6, f"tentacle ring must be a single row band, width={an['ring_width_rows']}"
    assert 5 <= an["n_tentacles"] <= 8 and an["ring_spacing_cv"] < 0.25, f"expected 5-8 regular tentacle primordia, got {an['n_tentacles']} (CV {an['ring_spacing_cv']:.3f})"
    assert an["i_foot"] >= 0.9 * shell["L_cells"] and an["foot_cv_theta"] < 0.05, "foot must be a uniform basal disc"
    assert an["apex_tentacle_leak"] < 0.02, "tentacle activator must be excluded from the hypostome"
    pb = an["predicted_ring_band"]
    assert pb and min(an["ring_rows"]) >= min(pb) and max(an["ring_rows"]) <= max(pb), "simulated ring must lie inside the analytic firing band phi_t > w_c"

    # 12. kappa_a ablation + sweep and body-length sweep monotonicity
    abl = shell["ablation"]
    assert abl is not None and abl["apex_tentacle_leak"] > 5.0 * an["apex_tentacle_leak"], "kappa_a = 0 ablation must let tentacle activator leak into the hypostome"
    assert abl["ring_centroid_frac"] < an["ring_centroid_frac"], "ablation must pull the ring toward the apex"
    assert shell["kappa_spearman"] >= 0.9, f"ring position must increase monotonically with kappa_a, rho={shell['kappa_spearman']:.2f}"
    assert shell["length_spearman"] >= 0.9, f"tentacle number must increase with body size, rho={shell['length_spearman']:.2f}"
    counts = [row["n_tentacles"] for row in shell["length_results"]]
    assert counts == sorted(counts), f"tentacle counts must be non-decreasing in L, got {counts}"
    assert body_prof["n_tentacles"] in range(min(counts), max(counts) + 1), "specimen tentacle count must fall inside the swept range"

    # 13. Shell amputation/regeneration: new hypostome at the fragment tip, then a new ring below it
    rg = shell["regen"]
    assert rg["analysis"]["i_head"] == 0 and rg["head_recovery_ratio"] >= 0.6, f"fragment must regenerate a hypostome at its tip, recovery={rg['head_recovery_ratio']:.3f}"
    # NOTE: on the wide, un-remodelled cut surface the regenerated head is a localised spot (CV_theta ~ 1), not a
    # uniform cap — a known limitation of the fixed geometry, reported (not hidden) in measure_morphoeval().
    assert rg["t_head"] is not None and rg["t_ring"] is not None and rg["t_head"] < rg["t_ring"], f"head must regenerate before the tentacle ring (t_head={rg['t_head']}, t_ring={rg['t_ring']})"
    assert rg["analysis"]["b_max"] > 0.5 and rg["analysis"]["n_tentacles"] >= 3 and rg["analysis"]["ring_offset_frac"] < 0.2, "fragment must regenerate a sub-hypostomal tentacle ring"
    assert rg["analysis"]["i_foot"] >= 0.9 * rg["regen"]["a"].shape[0], "fragment must keep its basal foot"

    # 14. Bitwise determinism of the shell solver (FFT + banded solve are deterministic)
    r_small = specimen_shell_radius_profile(body_prof, 40)
    s1 = simulate_three_system_shell(r_small, n_theta=32, steps=200, seed=1972)
    s2 = simulate_three_system_shell(r_small, n_theta=32, steps=200, seed=1972)
    assert all(np.array_equal(s1[k], s2[k]) for k in ("a", "b", "f", "sigma")), "shell solver determinism check failed"

    print(
        "[1972-gierer-hydra-activator-inhibitor] SELF-TEST PASSED (14/14 checks: SHA-256 dataset locks, "
        f"d_c={stab['d_c']:.4f}, bisection recovery={bisect['recovery_ratio']*100:.1f}%, "
        f"tentacle ring={ring_data['n_tentacle_peaks']} peaks vs specimen={spec_seg['n_tentacles_segmented']} tentacles, "
        f"equal-diffusion std={ring_data['ring_control']['a_std']:.2e}, domain scaling r={dom_sweep['pearson_r']:.4f}; "
        f"3D shell: head CV={an['head_cv_theta']:.3f}, ring s/L={an['ring_offset_frac']:.3f} (specimen {body_prof['ring_offset_frac']:.3f}), "
        f"{an['n_tentacles']} tentacles, foot s/L={an['foot_offset_frac']:.2f}, kappa rho={shell['kappa_spearman']:.2f}, "
        f"L rho={shell['length_spearman']:.2f}, regen recovery={rg['head_recovery_ratio']*100:.0f}%)."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Run 14-check deterministic verification suite and exit")
    ap.add_argument("--skip-gif", action="store_true", help="Skip rendering animated GIF")
    ap.add_argument("--shell-length", type=int, default=SHELL_DEFAULT_LENGTH_CELLS, help="3D shell body length L in cells (default 90, spacing/L calibrated to the specimen)")
    ap.add_argument("--shell-steps", type=int, default=SHELL_DEFAULT_STEPS, help="3D shell time steps (dt = 0.5)")
    ap.add_argument("--kappa-a", type=float, default=THREE_SYSTEM_PARAMS["kappa_a"], help="Head-activator suppression of tentacle activation kappa_a (0 = ablation)")
    ap.add_argument("--c", type=float, default=0.05, help="Autocatalytic activator production rate c")
    ap.add_argument("--c-prime", type=float, default=0.05, help="Inhibitor production rate c'")
    ap.add_argument("--mu-a", type=float, default=0.02, help="Activator degradation rate mu_a")
    ap.add_argument("--mu-h", type=float, default=0.03, help="Inhibitor degradation rate mu_h")
    ap.add_argument("--Da", type=float, default=0.015, help="Activator diffusivity D_a")
    ap.add_argument("--Dh", type=float, default=0.45, help="Inhibitor diffusivity D_h")
    ap.add_argument("--seed", type=int, default=1972, help="Deterministic RNG seed")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    res = render_outputs(
        c=args.c,
        c_prime=args.c_prime,
        mu_a=args.mu_a,
        mu_h=args.mu_h,
        D_a=args.Da,
        D_h=args.Dh,
        seed=args.seed,
        skip_gif=args.skip_gif,
        shell_length=args.shell_length,
        shell_steps=args.shell_steps,
        kappa_a=args.kappa_a,
    )
    sh_an = res["shell"]["intact_analysis"]
    print(
        f"[1972-gierer-hydra-activator-inhibitor] Rendered 15-panel comparison -> {OUT_PNG.relative_to(ROOT)} "
        f"and animation -> {OUT_GIF.relative_to(ROOT)} "
        f"(regen recovery={res['bisection']['recovery_ratio']*100:.1f}%, "
        f"tentacle peaks={res['ring']['n_tentacle_peaks']}, "
        f"specimen tentacles={res['specimen']['n_tentacles_segmented']}; "
        f"3D shell: ring s/L={sh_an['ring_offset_frac']:.3f} vs specimen {res['body_profile']['ring_offset_frac']:.3f}, "
        f"{sh_an['n_tentacles']} tentacles, head CV={sh_an['head_cv_theta']:.3f}, foot s/L={sh_an['foot_offset_frac']:.2f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
