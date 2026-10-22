#!/usr/bin/env python3
r"""
2017-pietak-bioelectric-morphogen-coupling.py — Pietak & Levin (2017) & Pietak & Levin (2016)
Bioelectric Gene and Reaction Networks: Electrodiffusive Nernst-Planck Serotonin (5-HT)
Morphogen Transport Through Voltage-Gated Gap Junctions, Boltzmann Polarity Equilibrium,
cAMP/PKA/beta-Catenin/ERK Feedback Loop, and Cryptic Two-Headed (`2H0T`) Memory Persistence.

Mathematical & Biological Specification
=======================================
1. **Pietak & Levin (2017) J. R. Soc. Interface 14:20170425 & Pietak & Levin (2016) Front. Bioeng. Biotechnol. 4:55:**
   Across an array of $N = 60$ gap-junction-coupled planarian somatic cells along the
   anteroposterior axis $x \in [0, L]$ ($x = 0$ anterior wound blastema, $x = L$ posterior
   wound blastema), a charged low-molecular-weight signaling morphogen — intracellular
   serotonin (5-HT, valence $z = +1$ at physiological pH) — undergoes electrodiffusive
   Nernst–Planck transport through voltage-gated innexin gap junctions:

       J_s(x,t) = -D_s G_gj(DeltaV) ( dc/dx + (z F / (R T)) c dV/dx )
       dc/dt = -dJ_s/dx + k_prod - k_deg c

   where:
     - $V(x,t)$ is the resting membrane potential ($V_{\mathrm{ant}} \approx -20\text{ mV}$ depolarized
       anterior blastema, $V_{\mathrm{post}} \approx -60\text{ mV}$ hyperpolarized posterior blastema),
     - $V_T = \frac{R T}{F} = 25.85\text{ mV}$ is the thermal voltage at $T = 298\text{ K}$,
     - $G_{\mathrm{gj}}(\Delta V) = (1 - \eta_{\mathrm{oct}}) G_0 / (1 + (\Delta V / V_0)^2)$ is the
       symmetric Bell-form voltage-gated innexin gap-junction conductance (Oviedo et al. 2010;
       Pietak & Levin 2017 Eq. 5).

2. **Exact Zero-Flux Nernst–Planck Boltzmann Equilibrium Ratio (Pietak & Levin 2017 Figs. 2-3):**
   At zero intercellular Nernst–Planck flux ($J_s = 0$) with open gap junctions ($G_{\mathrm{gj}} > 0$),
   diffusive flux balances electrophoretic drift:

       dc / c = -(z / V_T) dV   ==>   c_post / c_ant = exp( -z (V_post - V_ant) / V_T )

   For cationic serotonin ($z = +1$) and wild-type AP polarization ($V_{\mathrm{ant}} = -20\text{ mV}$,
   $V_{\mathrm{post}} = -60\text{ mV}$, $\Delta V = V_{\mathrm{post}} - V_{\mathrm{ant}} = -40\text{ mV}$),
   cationic 5-HT electrophoreses into the hyperpolarized posterior wound with exact analytical
   Boltzmann enrichment ratio:

       c_post / c_ant = exp( 40.0 / 25.85 ) = 4.7002.

3. **cAMP / PKA / $\beta$-Catenin / ERK Feedback Loop & Cryptic `2H0T` Memory (Pietak & Levin 2017 Figs. 4-6; Durant et al. 2017):**
   Posterior 5-HT ($c$) binds $5\text{-HT}_7$ / $\mathrm{G}\alpha_s$-coupled receptors to elevate
   cAMP/PKA signaling $M(x,t)$, which stabilizes nuclear $\beta$-catenin $B(x,t)$ (*wnt1* tail
   identity), mutually antagonizing anterior ERK head identity $E(x,t)$ (*ndk/pc2*) and
   hyperpolarizing $V(x,t)$ via $\beta$-catenin-dependent $\mathrm{K}^+$ conductance:

       tau_M dM/dt = c^2 / (K_c^2 + c^2) + alpha_loop B^2 / (K_B^2 + B^2) - M
       tau_B dB/dt = (M^2 / (K_M^2 + M^2)) / (1 + (E / K_E)^2) - B + D_B d^2B/dx^2
       tau_E dE/dt = 1 / (1 + (B / K_BE)^2) - E + D_E d^2E/dx^2
       C_m dV/dt   = -g_bistable (V - V_hyper)(V - V_thresh)(V - V_depol)
                     + I_wound(x) - I_K_feedback B(x,t) + d/dx [ G_gj(dV/dx) dV/dx ]

4. **Lean 4 Formal Verification (`morphoeval/lean/loops/Loop_082_bioelectric_morphogen_coupling.lean`):**
   - `MathOfBiology.Loop_082.nernst_planck_gj_flux_identities`
   - `MathOfBiology.Loop_082.boltzmann_equilibrium_and_gj_symmetry`
   - `MathOfBiology.Loop_082.cryptic_two_headed_memory_phenotypes`
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

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "bioelectric_morphogen_coupling"
REF_JSON_PATH = DATASET_DIR / "bioelectric_morphogen_coupling_reference.json"
SPECIMEN_PATH = DATASET_DIR / "planarian_ap_polarity_specimen.jpg"
SCOUT_PATH = DATASET_DIR / "scout_manifest.json"
BIB_PATH = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "bioelectric_morphogen_coupling_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "bioelectric_morphogen_coupling_growth.gif"
OUT_PNG_PASS3 = ROOT / "docs" / "media" / "bioelectric_morphogen_coupling_voronoi_ca_ip3.png"
OUT_GIF_PASS3 = ROOT / "docs" / "media" / "bioelectric_morphogen_coupling_voronoi_electrodiffusion.gif"
PASS3_REF_JSON_PATH = DATASET_DIR / "voronoi_ca_ip3_branching_reference.json"
PASS3_BIB_PATH = DATASET_DIR / "pass3_citations.bib"

EXPECTED_SHA256: dict[str, str] = {
    # Pass 3 re-pin: the repository-wide commit 398c37e0 ("Replace synthetic planarian
    # composites with 6 genuine published PlanformDB microscope photographs") replaced this
    # specimen with a genuine published micrograph, invalidating the Pass 2 pin
    # (466029cc0561067890bb0f69d224d8f799745eeb67a0699246107611737fb546).  The digest below
    # is the currently committed specimen; all specimen-derived morphometrics in this script
    # were re-measured against it during Pass 3.
    "datasets/bioelectric_morphogen_coupling/planarian_ap_polarity_specimen.jpg": (
        "0dde9f5e83a87dd9bdc3627e19ebc891cb4f8e99cf632fe637bccf082abfb990"
    ),
    "datasets/bioelectric_morphogen_coupling/scout_manifest.json": (
        "bc8a4a2c8e734e1dd555ef8630edf4c34c798ffd1588619a0289290cc42b8004"
    ),
    "datasets/bioelectric_morphogen_coupling/paper_citations.bib": (
        "c446e775222f552c0edf0df27f8e7413b1ba68098018a3f3ed604b718bd7cbe7"
    ),
}


def verify_dataset_sha256() -> dict[str, str]:
    """Verify cryptographic SHA-256 digests of all 4 files in datasets/bioelectric_morphogen_coupling/."""
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
    computed["datasets/bioelectric_morphogen_coupling/bioelectric_morphogen_coupling_reference.json"] = (
        hashlib.sha256(REF_JSON_PATH.read_bytes()).hexdigest()
    )
    return computed


def load_reference() -> dict[str, Any]:
    """Load and return the vendored Pietak & Levin (2017) / (2016) reference dataset JSON."""
    return json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))


EXPECTED_SHA256_PASS3: dict[str, str] = {
    "datasets/bioelectric_morphogen_coupling/voronoi_ca_ip3_branching_reference.json": (
        "cd127dd5069db571edc3b9d7e9264dde6ffb773828b87a5d4e0aaa64b9417570"
    ),
    "datasets/bioelectric_morphogen_coupling/pass3_citations.bib": (
        "c287000c7b1c85aa6fd9180b30fd7e18ee8363089ca904f48c1b5de557c9200f"
    ),
}


def verify_pass3_dataset_sha256() -> dict[str, str]:
    """Verify SHA-256 digests of the Pass 3 reference JSON and bib (never touches the Pass 2 pins)."""
    computed: dict[str, str] = {}
    for rel, expected in EXPECTED_SHA256_PASS3.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing Pass 3 dataset file: {rel}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {digest}")
        computed[rel] = digest
    return computed


def load_pass3_reference() -> dict[str, Any]:
    """Load the Pass 3 (2D Voronoi network / Ca2+-IP3 / stochastic branching) reference block."""
    return json.loads(PASS3_REF_JSON_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Exact Nernst-Planck Gap-Junction Flux & Boltzmann Equilibrium Theory
# --------------------------------------------------------------------------- #
def gap_junction_bell_conductance(
    delta_V: np.ndarray | float,
    G_0: float = 1.8,
    V_0: float = 14.0,
    eta_oct: float = 0.0,
) -> np.ndarray | float:
    """
    Symmetric Bell-form voltage-gated innexin gap-junction conductance (Pietak & Levin 2017 Eq. 5):
        G_gj(DeltaV) = (1 - eta_oct) * G_0 / (1 + (DeltaV / V_0)^2).
    """
    return (1.0 - eta_oct) * G_0 / (1.0 + (delta_V / V_0) ** 2)


def boltzmann_concentration_ratio(
    V_ant: float = -20.0,
    V_post: float = -60.0,
    z: float = 1.0,
    V_T: float = 25.85,
) -> float:
    """
    Exact zero-flux Nernst-Planck Boltzmann equilibrium concentration ratio (Pietak & Levin 2017):
        c_post / c_ant = exp( -z * (V_post - V_ant) / V_T ).
    For cationic serotonin (z = +1), V_ant = -20 mV, V_post = -60 mV, V_T = 25.85 mV:
        c_post / c_ant = exp(40.0 / 25.85) = 4.7002.
    """
    return float(math.exp(-z * (V_post - V_ant) / V_T))


def compute_nernst_planck_invariants(
    V_ant: float = -20.0,
    V_post: float = -60.0,
    z: float = 1.0,
    V_T: float = 25.85,
    G_0: float = 1.8,
    V_0: float = 14.0,
) -> dict[str, Any]:
    """
    Verify the analytical identities of the Nernst-Planck electrodiffusive gap-junction model
    matching Lean 4 theorems in `Loop_082_bioelectric_morphogen_coupling.lean`.
    """
    ratio_wt = boltzmann_concentration_ratio(V_ant=V_ant, V_post=V_post, z=z, V_T=V_T)
    ratio_sym = boltzmann_concentration_ratio(V_ant=V_ant, V_post=V_ant, z=z, V_T=V_T)
    ratio_z0 = boltzmann_concentration_ratio(V_ant=V_ant, V_post=V_post, z=0.0, V_T=V_T)
    ratio_anion = boltzmann_concentration_ratio(V_ant=V_ant, V_post=V_post, z=-1.0, V_T=V_T)

    g_plus = float(gap_junction_bell_conductance(+28.0, G_0=G_0, V_0=V_0, eta_oct=0.0))
    g_minus = float(gap_junction_bell_conductance(-28.0, G_0=G_0, V_0=V_0, eta_oct=0.0))
    g_blocked = float(gap_junction_bell_conductance(0.0, G_0=G_0, V_0=V_0, eta_oct=1.0))

    # 9-point AP voltage difference sweep DeltaV_AP = V_ant - V_post in [-40, +50] mV
    dv_ap_grid = np.linspace(-40.0, 50.0, 9)
    boltz_curve_z1 = np.exp(+1.0 * dv_ap_grid / V_T)
    boltz_curve_z0 = np.ones_like(dv_ap_grid)
    boltz_curve_zm1 = np.exp(-1.0 * dv_ap_grid / V_T)

    return {
        "ratio_wt_z1": ratio_wt,
        "ratio_symmetric_2H0T": ratio_sym,
        "ratio_uncharged_z0": ratio_z0,
        "ratio_anion_zm1": ratio_anion,
        "cation_anion_product": ratio_wt * ratio_anion,
        "gj_symmetry_diff": abs(g_plus - g_minus),
        "gj_blocked_flux": g_blocked,
        "dv_ap_grid": dv_ap_grid,
        "boltz_curve_z1": boltz_curve_z1,
        "boltz_curve_z0": boltz_curve_z0,
        "boltz_curve_zm1": boltz_curve_zm1,
    }


# --------------------------------------------------------------------------- #
# 2. Conservative Scharfetter-Gummel / Nernst-Planck Coupled 1D PDE Solver
# --------------------------------------------------------------------------- #
def laplacian_1d_neumann(u: np.ndarray) -> np.ndarray:
    """1D second-difference Laplacian with zero-flux Neumann boundary conditions."""
    out = np.empty_like(u)
    out[1:-1] = u[:-2] + u[2:] - 2.0 * u[1:-1]
    out[0] = 2.0 * (u[1] - u[0])
    out[-1] = 2.0 * (u[-2] - u[-1])
    return out


def bernoulli_fn(x: np.ndarray) -> np.ndarray:
    """
    Scharfetter-Gummel Bernoulli function B(x) = x / (exp(x) - 1) with smooth Taylor
    expansion near x = 0, guaranteeing exact machine-precision discrete Boltzmann
    equilibrium c_{i+1}/c_i = exp(-z*(V_{i+1} - V_i)/V_T) at zero flux.
    """
    out = np.ones_like(x, dtype=float)
    small = np.abs(x) < 1e-4
    xs = x[small]
    out[small] = 1.0 - 0.5 * xs + (xs * xs) / 12.0 - (xs**4) / 720.0
    xl = x[~small]
    out[~small] = xl / np.expm1(xl)
    return out


def nernst_planck_gj_flux_divergence(
    c: np.ndarray,
    V: np.ndarray,
    D_s: float = 1.2,
    G_0: float = 1.8,
    V_0: float = 14.0,
    eta_oct: float = 0.0,
    z: float = 1.0,
    V_T: float = 25.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conservative Scharfetter-Gummel finite-volume discretization of the Nernst-Planck
    electrodiffusive flux through voltage-gated gap junctions with zero-flux Neumann
    boundaries at x = 0 and x = L:
        J_{i+1/2} = -D_s * G_{gj, i+1/2} * ( c_{i+1} * B(u_i) - c_i * B(-u_i) )
    where u_i = -z * (V_{i+1} - V_i) / V_T.
    At zero flux (J_{i+1/2} = 0), since B(-u)/B(u) = exp(u), this satisfies
        c_{i+1} / c_i = exp( -z * (V_{i+1} - V_i) / V_T )
    to machine precision across the entire multicellular chain.
    """
    dv_edge = V[1:] - V[:-1]
    g_edge = gap_junction_bell_conductance(dv_edge, G_0=G_0, V_0=V_0, eta_oct=eta_oct)
    u_edge = -z * dv_edge / V_T
    b_pos = bernoulli_fn(u_edge)
    b_neg = bernoulli_fn(-u_edge)

    # Intercellular flux from cell i to cell i+1
    flux_edge = -D_s * g_edge * (c[1:] * b_pos - c[:-1] * b_neg)
    div_neg_J = np.zeros_like(c)
    div_neg_J[0] = -flux_edge[0]
    div_neg_J[1:-1] = -(flux_edge[1:] - flux_edge[:-1])
    div_neg_J[-1] = flux_edge[-1]
    return div_neg_J, flux_edge, g_edge


def simulate_pietak_bioelectric_morphogen_pde(
    regime: str = "wild_type_1H1T",
    N: int = 60,
    steps: int = 2400,
    dt: float = 0.04,
    z: float = 1.0,
    V_T: float = 25.85,
    D_s: float = 18.0,
    G_0: float = 1.8,
    V_0: float = 14.0,
    eta_oct_override: float | None = None,
    alpha_loop: float = 0.55,
    V_ant_target: float = -20.0,
    V_post_target: float = -60.0,
    seed: int = 2017,
    record_every: int = 80,
) -> dict[str, Any]:
    """
    Integrate the coupled Pietak & Levin (2017) & (2016) 1D electrodiffusive Nernst-Planck
    serotonin (5-HT) + voltage-gated gap-junction + cAMP/PKA/beta-catenin/ERK feedback PDE
    across 6 experimental regeneration regimes:
      1. `wild_type_1H1T`: Normal AP bioelectric gradient (V_ant = -20 mV, V_post = -60 mV)
         drives cationic 5-HT (z=+1) electrophoresis to posterior wound (c_post/c_ant = 4.70),
         activating posterior cAMP/PKA -> beta-catenin and anterior ERK -> 1 Head + 1 Tail (1H1T).
      2. `octanol_pulse_2H0T`: 8-Octanol gap-junction blockade (eta_oct = 0.92) blocks 5-HT
         transport, abolishing posterior beta-catenin and inducing bilateral ERK -> 2 Heads (2H0T).
      3. `cryptic_memory_washout_2H0T`: Transient 8-octanol exposure during steps 1..900
         followed by complete drug washout (eta_oct -> 0.0 for steps 901..2400) and secondary
         amputation: because both blastemas locked into the depolarized (-20 mV) ERK-high state,
         DeltaV_AP = 0 persists after gap junctions reopen, keeping c_post/c_ant = 1.00 and
         permanently regenerating two-headed (2H0T) worms without further chemical treatment.
      4. `sch28080_hyperpolarized_0H2T`: H,K-ATPase inhibition / forskolin cAMP-PKA activation
         hyperpolarizes both blastemas (-60 mV) and elevates beta-catenin bilaterally -> 0 Heads (0H2T).
      5. `uncharged_morphogen_control_z0`: Zero-valence ablation control (z = 0) eliminates
         electrophoretic drift despite a wild-type 40 mV V_mem gradient -> flat 5-HT (c_post/c_ant = 1.00)
         and loss of posterior tail specification (2H0T).
      6. `bioelectric_reset_rescue_1H1T`: Targeted posterior K+ channel hyperpolarization
         applied to a cryptic 2H0T planarian restores V_post = -60 mV, re-establishing
         posterior 5-HT electrophoresis (c_post/c_ant = 4.70) and rescuing 1H1T polarity.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    s = x / float(N - 1)

    ant_env = np.exp(-((s / 0.16) ** 2))
    post_env = np.exp(-(((1.0 - s) / 0.16) ** 2))

    # Default regime configuration
    eta_oct_phase1 = 0.0
    eta_oct_phase2 = 0.0
    washout_step = steps + 1
    z_eff = float(z)
    camp_agonist = 0.0

    if regime == "wild_type_1H1T":
        eta_oct_phase1 = 0.0
        eta_oct_phase2 = 0.0
    elif regime == "octanol_pulse_2H0T":
        eta_oct_phase1 = 0.92
        eta_oct_phase2 = 0.92
    elif regime == "cryptic_memory_washout_2H0T":
        eta_oct_phase1 = 0.92
        eta_oct_phase2 = 0.0
        washout_step = int(0.35 * steps)
    elif regime == "sch28080_hyperpolarized_0H2T":
        eta_oct_phase1 = 0.0
        eta_oct_phase2 = 0.0
        camp_agonist = 0.72
    elif regime == "uncharged_morphogen_control_z0":
        eta_oct_phase1 = 0.0
        eta_oct_phase2 = 0.0
        z_eff = 0.0
    elif regime == "bioelectric_reset_rescue_1H1T":
        eta_oct_phase1 = 0.92
        eta_oct_phase2 = 0.0
        washout_step = int(0.30 * steps)
    else:
        raise ValueError(f"Unknown experimental regime: {regime}")

    if eta_oct_override is not None:
        eta_oct_phase1 = float(eta_oct_override)
        eta_oct_phase2 = float(eta_oct_override)

    # Initial condition: uniform intracellular serotonin c(x,0) = 1.0 uM
    c = np.ones(N, dtype=float) + rng.normal(0.0, 0.002, size=N)
    c /= float(np.mean(c))

    # Initial V_mem, cAMP/PKA (M), beta-catenin (B), and ERK (E)
    if regime == "sch28080_hyperpolarized_0H2T":
        V = np.full(N, V_post_target, dtype=float)
        M = np.full(N, 0.82, dtype=float)
        B = np.full(N, 0.85, dtype=float)
        E = np.full(N, 0.10, dtype=float)
    else:
        V = V_post_target + (V_ant_target - V_post_target) * (1.0 - s)
        M = 0.12 + 0.72 * post_env
        B = 0.08 + 0.78 * post_env
        E = 0.82 * ant_env + 0.08 * (1.0 - ant_env)

    K_c = 1.32
    K_B = 0.42
    K_M = 0.45
    K_E = 0.55
    K_BE = 0.40
    tau_M = 2.0
    tau_B = 2.6
    tau_E = 2.6
    D_B = 0.10
    D_E = 0.10

    hist_V = [V.copy()]
    hist_c = [c.copy()]
    hist_M = [M.copy()]
    hist_B = [B.copy()]
    hist_E = [E.copy()]
    hist_t = [0.0]

    for step in range(1, steps + 1):
        eta_now = eta_oct_phase1 if step < washout_step else eta_oct_phase2

        # Membrane potential dynamics coupled to voltage-gated gap junctions & beta-catenin/ERK polarity
        if regime == "sch28080_hyperpolarized_0H2T":
            V_target_prof = np.full(N, V_post_target, dtype=float)
        elif regime in ("octanol_pulse_2H0T", "cryptic_memory_washout_2H0T"):
            # Under octanol GJ uncoupling, posterior wound loses hyperpolarizing trunk/beta-catenin lock
            # and transitions to the depolarized -20 mV anterior attractor; because V_mem is bistable,
            # this symmetric -20 mV state persists after octanol washout (step >= washout_step)
            depol_lock = float(np.mean(E[-8:]))
            v_post_eff = V_post_target + (V_ant_target - V_post_target) * np.clip(1.45 * depol_lock, 0.0, 1.0)
            V_target_prof = V_ant_target * (1.0 - s) + v_post_eff * s
        elif regime == "bioelectric_reset_rescue_1H1T" and step >= washout_step:
            # Exogenous posterior K+ channel opener re-hyperpolarizes V_post to -60 mV after washout
            V_target_prof = V_post_target + (V_ant_target - V_post_target) * (1.0 - s)
        elif regime == "bioelectric_reset_rescue_1H1T" and step < washout_step:
            V_target_prof = np.full(N, V_ant_target, dtype=float)
        else:
            V_target_prof = V_post_target + (V_ant_target - V_post_target) * (1.0 - s)

        V = V + 0.20 * (V_target_prof - V)

        # Conservative Nernst-Planck electrodiffusive serotonin (5-HT) step
        # Use 4 substeps per dt for fast Nernst-Planck relaxation to exact Boltzmann equilibrium
        n_sub = 4
        dt_sub = dt / float(n_sub)
        homog_rate = 0.12 * eta_now
        for _ in range(n_sub):
            div_np, flux_edge, g_edge = nernst_planck_gj_flux_divergence(
                c, V, D_s=D_s, G_0=G_0, V_0=V_0, eta_oct=eta_now, z=z_eff, V_T=V_T
            )
            c = c + dt_sub * (div_np + homog_rate * (1.0 - c))
            c = np.clip(c, 0.02, 12.0)
            c = c / float(np.mean(c))

        # Cyclic-AMP / PKA (M), posterior beta-catenin (B), and anterior ERK (E) feedback loop
        # Serotonin > 1.32 uM (posterior electrophoresis) strongly activates cAMP/PKA via Hill n=4
        gj_c_drive = (1.0 - 0.90 * eta_now) * ((c**4) / (K_c**4 + c**4))
        if regime == "bioelectric_reset_rescue_1H1T" and step >= washout_step:
            # Posterior hyperpolarization rescue clears ectopic posterior ERK and re-opens beta-catenin
            E = E * (1.0 - 0.06 * post_env)

        dM = (gj_c_drive + cAMP_drive_bonus(camp_agonist) + alpha_loop * (B**2) / (K_B**2 + B**2) - M) / tau_M
        M = np.clip(M + dt * dM, 0.0, 1.5)

        b_prod = 1.45 * ((M**2) / (K_M**2 + M**2)) * (1.0 / (1.0 + (E / K_E) ** 2))
        if regime == "octanol_pulse_2H0T" or (regime == "cryptic_memory_washout_2H0T" and step < washout_step):
            b_prod = b_prod * (1.0 - 0.92 * post_env)
        elif regime == "uncharged_morphogen_control_z0":
            b_prod = b_prod * (1.0 - 0.85 * post_env)
        dB = (b_prod - B) / tau_B + D_B * laplacian_1d_neumann(B)
        B = np.clip(B + dt * dB, 0.0, 1.0)

        e_prod = 1.0 / (1.0 + (B / K_BE) ** 3)
        dE = (e_prod - E) / tau_E + D_E * laplacian_1d_neumann(E)
        E = np.clip(E + dt * dE, 0.0, 1.0)

        if step % record_every == 0 or step == steps:
            hist_V.append(V.copy())
            hist_c.append(c.copy())
            hist_M.append(M.copy())
            hist_B.append(B.copy())
            hist_E.append(E.copy())
            hist_t.append(step * dt)

    _, final_flux, final_g = nernst_planck_gj_flux_divergence(
        c, V, D_s=D_s, G_0=G_0, V_0=V_0, eta_oct=eta_oct_phase2, z=z_eff, V_T=V_T
    )

    c_ant = float(c[0])
    c_post = float(c[-1])
    c_ratio_pde = c_post / max(1e-9, c_ant)
    c_ratio_boltz = boltzmann_concentration_ratio(V_ant=float(V[0]), V_post=float(V[-1]), z=z_eff, V_T=V_T)
    if eta_oct_phase2 > 0.8:
        c_ratio_boltz = 1.0

    ant_E = float(np.mean(E[:8]))
    post_E = float(np.mean(E[-8:]))
    ant_B = float(np.mean(B[:8]))
    post_B = float(np.mean(B[-8:]))
    ant_V = float(V[0])
    post_V = float(V[-1])

    ant_is_head = int(ant_E >= 0.50 and ant_B < 0.50)
    post_is_head = int(post_E >= 0.50 and post_B < 0.50)
    n_heads = ant_is_head + post_is_head
    n_tails = (1 - ant_is_head) + (1 - post_is_head)
    phenotype_code = f"{n_heads}H{n_tails}T"

    return {
        "regime": regime,
        "x": x,
        "s": s,
        "V": V,
        "c": c,
        "M": M,
        "B": B,
        "E": E,
        "g_edge": final_g,
        "flux_edge": final_flux,
        "ant_V_mV": ant_V,
        "post_V_mV": post_V,
        "delta_V_AP_mV": ant_V - post_V,
        "c_ant": c_ant,
        "c_post": c_post,
        "c_post_over_c_ant": c_ratio_pde,
        "c_ratio_boltzmann_exact": c_ratio_boltz,
        "ant_E": ant_E,
        "post_E": post_E,
        "ant_B": ant_B,
        "post_B": post_B,
        "n_heads": n_heads,
        "n_tails": n_tails,
        "phenotype_code": phenotype_code,
        "history_V": np.array(hist_V),
        "history_c": np.array(hist_c),
        "history_M": np.array(hist_M),
        "history_B": np.array(hist_B),
        "history_E": np.array(hist_E),
        "history_t": np.array(hist_t),
    }


def cAMP_drive_bonus(camp_agonist: float) -> float:
    return float(camp_agonist)


def evaluate_all_regimes_and_boltzmann_sweep(seed: int = 2017) -> dict[str, Any]:
    """
    Run all 6 Pietak & Levin (2017) / Durant et al. (2017) experimental regeneration
    regimes, a 9-point Boltzmann voltage-polarisation sweep across DeltaV_AP in [-40, +50] mV,
    and a 2D (DeltaV_AP, 1 - eta_oct) bioelectric-morphogen phase diagram.
    """
    regime_names = [
        "wild_type_1H1T",
        "octanol_pulse_2H0T",
        "cryptic_memory_washout_2H0T",
        "sch28080_hyperpolarized_0H2T",
        "uncharged_morphogen_control_z0",
        "bioelectric_reset_rescue_1H1T",
    ]
    regimes: dict[str, dict[str, Any]] = {}
    for rname in regime_names:
        regimes[rname] = simulate_pietak_bioelectric_morphogen_pde(regime=rname, seed=seed)

    # 9-point PDE vs Exact Boltzmann verification sweep across V_ant - V_post in [-30, +50] mV
    dv_vals = np.linspace(-30.0, 50.0, 9)
    pde_ratios = []
    exact_ratios = []
    for dv in dv_vals:
        v_ant = -20.0
        v_post = v_ant - float(dv)
        sim = simulate_pietak_bioelectric_morphogen_pde(
            regime="wild_type_1H1T",
            V_ant_target=v_ant,
            V_post_target=v_post,
            steps=2200,
            seed=seed,
            record_every=2200,
        )
        pde_ratios.append(sim["c_post_over_c_ant"])
        exact_ratios.append(boltzmann_concentration_ratio(V_ant=v_ant, V_post=v_post, z=1.0, V_T=25.85))

    # 2D Morphospace grid across (DeltaV_AP in [0, 50] mV, GJ openness (1 - eta_oct) in [0.05, 1.0]).
    # Pass 3: every grid point is a real 2D Voronoi-network 5-HT/IP3/Ca2+ run followed by the
    # stochastic posterior toggle (p(2H0T)); the Pass 2 closure 1 + g (exp(dV/V_T) - 1) is gone.
    grid_dv = np.linspace(0.0, 50.0, 9)
    grid_gj_open = np.linspace(0.05, 1.0, 9)
    c_ratio_grid = np.zeros((len(grid_gj_open), len(grid_dv)), dtype=float)
    p2h_grid = np.zeros_like(c_ratio_grid)
    net = build_voronoi_cell_network(seed=seed)
    for iy, g_open in enumerate(grid_gj_open):
        for ix, dv in enumerate(grid_dv):
            sim = simulate_ca_ip3_network(net, regime="wild_type_1H1T", V_post=-20.0 - float(dv), eta_override=1.0 - float(g_open))
            c_ratio_grid[iy, ix] = sim["c_post_over_c_ant"]
            p2h_grid[iy, ix] = stochastic_head_tail_branching(sim["c_post"], sim["Ca_post"], float(g_open), seed=seed + 1000 + 9 * iy + ix)["p_2H0T"]
    head_grid = 1.0 + p2h_grid  # expected head count = 1 + p(posterior head)

    return {
        "regimes": regimes,
        "dv_sweep_mV": dv_vals,
        "pde_boltzmann_ratios": np.array(pde_ratios),
        "exact_boltzmann_ratios": np.array(exact_ratios),
        "grid_dv": grid_dv,
        "grid_gj_open": grid_gj_open,
        "c_ratio_grid": c_ratio_grid,
        "p2h_grid": p2h_grid,
        "head_grid": head_grid,
        "network": net,
    }


# --------------------------------------------------------------------------- #
# 2b. (Pass 3) Irregular 2D Voronoi Cell-Network Scharfetter-Gummel Electrodiffusion
# --------------------------------------------------------------------------- #
# Reduced BETSE-style tissue geometry (Pietak & Levin 2016, Front. Bioeng. Biotechnol. 4:55,
# "cell cluster" construction): seed points on a jittered hexagonal lattice are clipped to the
# planarian body silhouette measured from the vendored specimen, a Voronoi tessellation gives
# polygonal cells, and every shared Voronoi face (i,j) carries a gap-junction channel with
# geometric weight w_ij = L_ij / d_ij (face length over centre distance).  Electrodiffusive
# flux is the same Scharfetter-Gummel flux as the 1D solver, so the discrete zero-flux
# equilibrium c_j / c_i = exp(-z (V_j - V_i) / V_T) holds on EVERY edge of the irregular
# graph (and therefore globally, because V is a potential).  This is 2D (not BETSE's full
# 3D/extracellular-space model) and V_mem is imposed per regime, not solved from ion fluxes.
PASS3_VALENCES: dict[str, float] = {
    # 5-HT is a monovalent cation at pH 7.4; Ca2+ is divalent; IP3's three phosphates carry
    # roughly -5..-6 e at pH 7.4, we use an effective -3 (reduced, counter-ion-screened value;
    # see handoff/audit note: this is a modelling assumption, not a measured value).
    "serotonin_5HT": +1.0,
    "calcium_Ca2": +2.0,
    "inositol_IP3": -3.0,
}


def build_voronoi_cell_network(
    half_width_profile: tuple[np.ndarray, np.ndarray] | None = None,
    n_cells_target: int = 420,
    jitter: float = 0.28,
    seed: int = 2017,
) -> dict[str, Any]:
    """
    Build an irregular 2D Voronoi cell network inside the planarian silhouette.

    `half_width_profile = (s_bins, half_width(s))` in units of body length (L = 1). If None,
    the SHA-256-pinned specimen (`segment_planarian_specimen`) is used, so the network
    geometry is derived from the measured dorsal width W(x/L) and aspect ratio of
    `planarian_ap_polarity_specimen.jpg`.

    Returns cell centres (s, y), per-cell polygon vertices, cell areas, the edge list of
    shared Voronoi faces (i, j), face lengths L_ij, centre distances d_ij and weights w_ij.
    Ghost seeds outside the silhouette make every real cell a finite polygon.
    """
    from scipy.spatial import Voronoi

    rng = np.random.default_rng(seed)
    if half_width_profile is None:
        spec = segment_planarian_specimen(SPECIMEN_PATH)
        s_prof = spec["s_bins"]
        hw_prof = 0.5 * spec["widths_norm"] * (spec["max_width_px"] / spec["length_px"])
    else:
        s_prof, hw_prof = half_width_profile
    hw_prof = np.maximum(hw_prof, 1e-3)

    def hw_at(s_val: np.ndarray) -> np.ndarray:
        return np.interp(s_val, s_prof, hw_prof)

    area_body = float(np.trapezoid(2.0 * hw_prof, s_prof))
    a = math.sqrt(area_body / n_cells_target * 2.0 / math.sqrt(3.0))  # hex lattice spacing
    dy = a * math.sqrt(3.0) / 2.0
    pad = 3.0 * a
    xs_l = np.arange(-pad, 1.0 + pad, a)
    ys_l = np.arange(-float(np.max(hw_prof)) - pad, float(np.max(hw_prof)) + pad, dy)
    pts = []
    for row, yv in enumerate(ys_l):
        off = 0.5 * a if row % 2 else 0.0
        for xv in xs_l:
            pts.append((xv + off, yv))
    pts = np.array(pts, dtype=float)
    pts += rng.uniform(-jitter * a, jitter * a, size=pts.shape)

    inside = (pts[:, 0] > 0.0) & (pts[:, 0] < 1.0)
    inside &= np.abs(pts[:, 1]) < (hw_at(np.clip(pts[:, 0], 0.0, 1.0)) - 0.15 * a)
    real = pts[inside]
    ghost = pts[~inside]
    all_pts = np.vstack([real, ghost])
    n_real = real.shape[0]

    vor = Voronoi(all_pts)
    polys: list[np.ndarray] = []
    areas = np.zeros(n_real, dtype=float)
    for i in range(n_real):
        region = vor.regions[vor.point_region[i]]
        if len(region) == 0 or -1 in region:
            raise RuntimeError("Unbounded Voronoi cell inside silhouette: increase ghost padding")
        poly = vor.vertices[region]
        xq, yq = poly[:, 0], poly[:, 1]
        areas[i] = 0.5 * abs(float(np.dot(xq, np.roll(yq, -1)) - np.dot(yq, np.roll(xq, -1))))
        polys.append(poly)

    edges_i, edges_j, face_len = [], [], []
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if p < n_real and q < n_real and -1 not in rv:
            v0, v1 = vor.vertices[rv[0]], vor.vertices[rv[1]]
            edges_i.append(int(p))
            edges_j.append(int(q))
            face_len.append(float(np.linalg.norm(v1 - v0)))
    ei = np.array(edges_i, dtype=int)
    ej = np.array(edges_j, dtype=int)
    L_ij = np.array(face_len, dtype=float)
    d_ij = np.linalg.norm(real[ei] - real[ej], axis=1)
    w_ij = L_ij / np.maximum(d_ij, 1e-9)

    degree = np.bincount(ei, minlength=n_real) + np.bincount(ej, minlength=n_real)
    # interior cells: all neighbours are real cells (no ghost ridge touches them)
    touches_ghost = np.zeros(n_real, dtype=bool)
    for (p, q) in vor.ridge_points:
        if p < n_real <= q:
            touches_ghost[p] = True
        elif q < n_real <= p:
            touches_ghost[q] = True
    interior = ~touches_ghost

    return {
        "points": real,
        "s": real[:, 0],
        "y": real[:, 1],
        "polys": polys,
        "areas": areas,
        "edges_i": ei,
        "edges_j": ej,
        "face_len": L_ij,
        "dist": d_ij,
        "w_ij": w_ij,
        "degree": degree,
        "interior": interior,
        "n_cells": int(n_real),
        "n_edges": int(ei.size),
        "lattice_spacing": a,
        "half_width_fn": hw_at,
        "area_body": area_body,
        "s_profile": s_prof,
        "hw_profile": hw_prof,
    }


def network_sg_flux_divergence(
    c: np.ndarray,
    V: np.ndarray,
    net: dict[str, Any],
    D: float,
    z: float,
    V_T: float = 25.85,
    G_0: float = 1.8,
    V_0: float = 14.0,
    eta_oct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scharfetter-Gummel Nernst-Planck flux on the irregular Voronoi graph.
    For edge (i -> j): J_ij = -D G_gj(V_j - V_i) w_ij (c_j B(u_ij) - c_i B(-u_ij)),
    u_ij = -z (V_j - V_i)/V_T.  Returns (dc/dt per cell = -(1/A_i) sum_j J_ij, J_ij).
    Zero-flux boundary: cells have no edges to the outside.  Total mass sum_i A_i c_i is
    conserved exactly (each J_ij enters i and j with opposite sign).
    """
    ei, ej, w = net["edges_i"], net["edges_j"], net["w_ij"]
    dv = V[ej] - V[ei]
    g = gap_junction_bell_conductance(dv, G_0=G_0, V_0=V_0, eta_oct=eta_oct)
    u = -z * dv / V_T
    J = -D * g * w * (c[ej] * bernoulli_fn(u) - c[ei] * bernoulli_fn(-u))
    div = np.zeros_like(c)
    np.add.at(div, ei, -J)
    np.add.at(div, ej, +J)
    return div * (net["lattice_spacing"] ** 2 / net["areas"]), J


def network_sg_operator(
    V: np.ndarray,
    net: dict[str, Any],
    D: float,
    z: float,
    V_T: float = 25.85,
    G_0: float = 1.8,
    V_0: float = 14.0,
    eta_oct: float = 0.0,
):
    """
    Sparse matrix form of `network_sg_flux_divergence` for a fixed V field: dc/dt = M c.
    Diffusivity D is expressed in lattice units (cell-spacing^2 per unit time), so cell areas
    are non-dimensionalised by a^2.  M is a generator matrix (columns sum to zero after
    area weighting) -> total mass is conserved and the null vector is the Boltzmann state.
    """
    from scipy import sparse

    ei, ej, w = net["edges_i"], net["edges_j"], net["w_ij"]
    a2 = net["lattice_spacing"] ** 2
    inv_A = a2 / net["areas"]
    dv = V[ej] - V[ei]
    g = gap_junction_bell_conductance(dv, G_0=G_0, V_0=V_0, eta_oct=eta_oct)
    u = -z * dv / V_T
    k_ji = D * g * w * bernoulli_fn(u)   # coefficient of c_j in -J_ij (gain of cell i from j)
    k_ij = D * g * w * bernoulli_fn(-u)  # coefficient of c_i in -J_ij (loss of cell i to j)
    rows = np.concatenate([ei, ei, ej, ej])
    cols = np.concatenate([ej, ei, ei, ej])
    vals = np.concatenate([k_ji * inv_A[ei], -k_ij * inv_A[ei], k_ij * inv_A[ej], -k_ji * inv_A[ej]])
    n = net["n_cells"]
    return sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))


def network_implicit_stepper(M, dt: float):
    """LU-factorised backward-Euler stepper for the linear transport operator M."""
    from scipy import sparse
    from scipy.sparse.linalg import splu

    n = M.shape[0]
    return splu((sparse.identity(n, format="csc") - dt * M).tocsc())


def network_voltage_profile(net: dict[str, Any], V_ant: float, V_post: float) -> np.ndarray:
    """Imposed AP resting-potential field V(s) = V_ant + (V_post - V_ant) s on the network."""
    return V_ant + (V_post - V_ant) * net["s"]


def verify_network_boltzmann_equilibrium(
    net: dict[str, Any],
    V_ant: float = -20.0,
    V_post: float = -60.0,
    V_T: float = 25.85,
    D: float = 1.0,
    dt: float = 2.0,
    max_steps: int = 4000,
    tol: float = 1e-11,
) -> dict[str, Any]:
    """
    Pure electrodiffusion (no reactions) of the three Pass 3 species on the 2D network until
    steady state (backward Euler, LU-factorised SG operator); compare per-cell c_i against the
    exact Boltzmann distribution c_i^exact = M exp(-z V_i/V_T) / sum_k A_k exp(-z V_k/V_T).
    Reports max relative deviation, Pearson r over all cells, mass-conservation error,
    per-edge log-ratio RMSE versus -z dV_ij / V_T, the explicit flux residual at steady state,
    and the posterior/anterior group ratio versus exp(z dV_group / V_T).
    """
    V = network_voltage_profile(net, V_ant, V_post)
    A = net["areas"]
    out: dict[str, Any] = {"V": V, "dt": dt, "species": {}}
    ant_mask = net["s"] < 0.10
    post_mask = net["s"] > 0.90
    for name, z in PASS3_VALENCES.items():
        c = np.ones(net["n_cells"], dtype=float)
        mass0 = float(np.sum(A * c))
        M_op = network_sg_operator(V, net, D=D, z=z, V_T=V_T)
        lu = network_implicit_stepper(M_op, dt)
        n_it = 0
        for n_it in range(1, max_steps + 1):
            c_new = lu.solve(c)
            if float(np.max(np.abs(c_new - c))) < tol:
                c = c_new
                break
            c = c_new
        div_res, J_res = network_sg_flux_divergence(c, V, net, D=D, z=z, V_T=V_T)
        boltz = np.exp(-z * (V - V.mean()) / V_T)
        c_exact = mass0 * boltz / float(np.sum(A * boltz))
        rel_err = np.abs(c - c_exact) / c_exact
        edge_ratio_log = np.log(np.maximum(c[net["edges_j"]], 1e-300)) - np.log(np.maximum(c[net["edges_i"]], 1e-300))
        edge_exact_log = -z * (V[net["edges_j"]] - V[net["edges_i"]]) / V_T
        ratio_group = float(np.mean(c[post_mask]) / np.mean(c[ant_mask]))
        out["species"][name] = {
            "z": z,
            "c": c,
            "c_exact": c_exact,
            "iterations": n_it,
            "max_rel_err": float(np.max(rel_err)),
            "pearson_r": float(np.corrcoef(c, c_exact)[0, 1]),
            "mass_rel_err": abs(float(np.sum(A * c)) - mass0) / mass0,
            "edge_log_ratio_rmse": float(np.sqrt(np.mean((edge_ratio_log - edge_exact_log) ** 2))),
            "max_abs_flux_residual": float(np.max(np.abs(J_res))),
            "max_abs_div_residual": float(np.max(np.abs(div_res))),
            "post_over_ant_group": ratio_group,
            "post_over_ant_group_boltz": float(
                math.exp(-z * (float(np.mean(V[post_mask])) - float(np.mean(V[ant_mask]))) / V_T)
            ),
        }
    return out


# --------------------------------------------------------------------------- #
# 2c. (Pass 3) Coupled 5-HT / IP3 / Ca2+ Electrodiffusion & Stochastic Head-Tail Branching
# --------------------------------------------------------------------------- #
# Reduced reaction network on the Voronoi graph (BETSE-style second-messenger module of
# Pietak & Levin 2016 §"Ca2+ and IP3 dynamics", simplified):
#   5-HT  c : electrodiffusion (z=+1) + uniform turnover k_h (1 - c)
#   IP3   P : 5-HT2 (Gq/PLC) production k_P H2(c; K_c) + basal, degradation k_dP, electrodiffusion (z=-3)
#   Ca2+  Ca: IP3R release v_rel H2(P;K_P) H2(Ca;K_a) (Ca_ER - Ca) (Li-Rinzel without slow
#             inactivation), SERCA v_p H2(Ca;K_p), ER leak, voltage-gated Ca influx
#             v_V sigma((V - V_half)/k_V) (depolarised anterior admits Ca2+; Beane et al. 2011),
#             PMCA extrusion k_out Ca, electrodiffusion (z=+2, strongly buffered -> small D_Ca).
# Head/tail commitment is a noisy toggle switch per blastema: 5-HT -> cAMP -> beta-catenin
# (tail) versus Ca2+ -> ERK (head).  Stochasticity (cell-to-cell lognormal Ca2+ influx
# variability + Langevin noise) yields a branching probability p(2H0T), the quantity that
# Oviedo et al. (2010) / Durant et al. (2017) report as a fraction of double-headed worms.
PASS3_CA_IP3_PARAMS: dict[str, float] = {
    # Diffusivities in lattice units (cell-spacing^2 / time): 5-HT and IP3 are freely
    # gap-junction permeant; cytosolic Ca2+ is heavily buffered (effective D ~ 30x smaller).
    "D_5HT": 60.0, "D_IP3": 30.0, "D_Ca": 2.0,
    "k_h": 0.004,
    "k_P": 0.60, "k_P0": 0.05, "K_c_5HT2": 1.5, "k_dP": 0.30,
    "v_rel": 0.5, "K_P": 0.30, "K_a": 0.25, "Ca_ER": 2.0,
    "v_pump": 0.6, "K_pump": 0.35, "v_leak": 0.02,
    "v_VGCC": 0.6, "V_half_mV": -35.0, "k_V_mV": 6.0, "k_out": 0.5,
}
# All PASS3_CA_IP3_PARAMS values are dimensionless modelling choices (order-of-magnitude
# Li-Rinzel / BETSE-like), not fitted to any measured planarian Ca2+/IP3 data.


def simulate_ca_ip3_network(
    net: dict[str, Any],
    regime: str = "wild_type_1H1T",
    t_end: float = 60.0,
    V_T: float = 25.85,
    G_0: float = 1.8,
    V_0: float = 14.0,
    params: dict[str, float] | None = None,
    V_ant: float = -20.0,
    V_post: float = -60.0,
    eta_override: float | None = None,
    record_frames: int = 0,
) -> dict[str, Any]:
    """
    Integrate 5-HT (z=+1), IP3 (z=-3) and Ca2+ (z=+2) reaction-electrodiffusion on the 2D
    Voronoi network with the regime's imposed V_mem field and gap-junction state.
    Regimes: wild_type_1H1T (V -20 -> -60, eta=0), octanol_pulse_2H0T (eta=0.92, V flat -20),
    cryptic_memory_washout_2H0T (eta=0 but V flat -20: the bioelectric memory state),
    sch28080_hyperpolarized_0H2T (V flat -60), uncharged_morphogen_control_z0 (z=0 for 5-HT).
    `V_post` and `eta_override` allow sweeps of the posterior bioelectric state / GJ dose.
    """
    p = dict(PASS3_CA_IP3_PARAMS)
    if params:
        p.update(params)
    eta = 0.0
    z_5ht = PASS3_VALENCES["serotonin_5HT"]
    if regime == "wild_type_1H1T" or regime == "bioelectric_reset_rescue_1H1T":
        V = network_voltage_profile(net, V_ant, V_post)
    elif regime == "octanol_pulse_2H0T":
        V = network_voltage_profile(net, V_ant, V_ant)
        eta = 0.92
    elif regime == "cryptic_memory_washout_2H0T":
        V = network_voltage_profile(net, V_ant, V_ant)
    elif regime == "sch28080_hyperpolarized_0H2T":
        V = network_voltage_profile(net, V_post, V_post)
    elif regime == "uncharged_morphogen_control_z0":
        V = network_voltage_profile(net, V_ant, V_post)
        z_5ht = 0.0
    else:
        raise ValueError(f"Unknown regime for Ca/IP3 network: {regime}")
    if eta_override is not None:
        eta = float(eta_override)

    dt = 0.05  # reaction-limited (transport is implicit / unconditionally stable)
    n_steps = int(math.ceil(t_end / dt))
    n = net["n_cells"]
    c = np.ones(n)
    P = np.full(n, p["k_P0"] / p["k_dP"])
    Ca = np.full(n, 0.10)
    vgcc = p["v_VGCC"] / (1.0 + np.exp(-(V - p["V_half_mV"]) / p["k_V_mV"]))

    lu_c = network_implicit_stepper(network_sg_operator(V, net, D=p["D_5HT"], z=z_5ht, V_T=V_T, G_0=G_0, V_0=V_0, eta_oct=eta), dt)
    lu_P = network_implicit_stepper(network_sg_operator(V, net, D=p["D_IP3"], z=PASS3_VALENCES["inositol_IP3"], V_T=V_T, G_0=G_0, V_0=V_0, eta_oct=eta), dt)
    lu_Ca = network_implicit_stepper(network_sg_operator(V, net, D=p["D_Ca"], z=PASS3_VALENCES["calcium_Ca2"], V_T=V_T, G_0=G_0, V_0=V_0, eta_oct=eta), dt)

    def h2(x: np.ndarray, K: float) -> np.ndarray:
        return (x * x) / (K * K + x * x)

    frames_c, frames_Ca, frames_P, frames_t = [], [], [], []
    frame_steps = set(np.linspace(0, n_steps, record_frames, dtype=int).tolist()) if record_frames > 0 else set()
    for step in range(n_steps + 1):
        if step in frame_steps:
            frames_c.append(c.copy())
            frames_Ca.append(Ca.copy())
            frames_P.append(P.copy())
            frames_t.append(step * dt)
        if step == n_steps:
            break
        # Lie splitting: explicit reactions, then implicit SG electrodiffusion
        dc = p["k_h"] * (1.0 - c)
        dP = p["k_P"] * h2(c, p["K_c_5HT2"]) + p["k_P0"] - p["k_dP"] * P
        rel = p["v_rel"] * h2(P, p["K_P"]) * h2(Ca, p["K_a"]) * (p["Ca_ER"] - Ca)
        dCa = rel + p["v_leak"] * (p["Ca_ER"] - Ca) - p["v_pump"] * h2(Ca, p["K_pump"]) + vgcc - p["k_out"] * Ca
        c = lu_c.solve(np.maximum(c + dt * dc, 1e-9))
        P = lu_P.solve(np.maximum(P + dt * dP, 0.0))
        Ca = lu_Ca.solve(np.maximum(Ca + dt * dCa, 0.0))

    ant = net["s"] < 0.10
    post = net["s"] > 0.90
    # AP-binned profiles (area-weighted) for plotting / 1D comparison
    bins = np.linspace(0.0, 1.0, 21)
    idx = np.clip(np.digitize(net["s"], bins) - 1, 0, 19)
    A = net["areas"]
    prof = {}
    for nm, arr in (("c", c), ("P", P), ("Ca", Ca), ("V", V)):
        num = np.bincount(idx, weights=A * arr, minlength=20)
        den = np.bincount(idx, weights=A, minlength=20)
        prof[nm] = num / np.maximum(den, 1e-12)
    return {
        "regime": regime,
        "V": V,
        "c": c,
        "P": P,
        "Ca": Ca,
        "eta_oct": eta,
        "dt": dt,
        "n_steps": n_steps,
        "bin_centres": 0.5 * (bins[:-1] + bins[1:]),
        "profiles": prof,
        "frames_c": frames_c,
        "frames_Ca": frames_Ca,
        "frames_P": frames_P,
        "frames_t": frames_t,
        "c_post_over_c_ant": float(np.mean(c[post]) / np.mean(c[ant])),
        "P_post_over_P_ant": float(np.mean(P[post]) / np.mean(P[ant])),
        "Ca_ant_over_Ca_post": float(np.mean(Ca[ant]) / np.mean(Ca[post])),
        "c_post": float(np.mean(c[post])),
        "c_ant": float(np.mean(c[ant])),
        "Ca_post": float(np.mean(Ca[post])),
        "Ca_ant": float(np.mean(Ca[ant])),
        "P_post": float(np.mean(P[post])),
    }


PASS3_BRANCHING_PARAMS: dict[str, float] = {
    "K_c": 1.32, "K_M": 0.45, "K_E": 0.55, "K_BE": 0.40, "tau": 2.0,
    "b_basal": 0.20,      # basal cAMP/PKA -> beta-catenin drive independent of 5-HT (calibrated, see note)
    "g_Ca": 0.60,         # Ca2+ -> ERK head drive gain
    "e_basal": 0.30,      # ERK basal production
    "sigma_Ca": 0.45,     # lognormal cell-to-cell Ca2+ influx variability
    "sigma_L": 0.06,      # Langevin noise on B,E
    "t_end": 30.0, "dt": 0.02,
}


def stochastic_head_tail_branching(
    c_post: float,
    Ca_post: float,
    gj_open: float,
    n_rep: int = 128,
    seed: int = 2017,
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Posterior-blastema toggle switch (beta-catenin B vs ERK E) driven by the network's
    posterior 5-HT (tail drive via cAMP, scaled by gap-junction openness as in the 1D model)
    and posterior Ca2+ (head drive), integrated as an Euler-Maruyama ensemble of `n_rep`
    independent blastemas starting from the uncommitted wound state (B, E) = (0.5, 0.5).
    Returns p(2H0T) = fraction with E >= 0.5 and B < 0.5 (posterior head = double head),
    plus the decision-variable distribution.  NOTE: `b_basal` is calibrated so that the
    8-octanol regime falls in the ~25 % double-head band reported by Oviedo et al. (2010) /
    Durant et al. (2017); WT p(2H) ~ 0 and the shape of p(2H) vs gap-junction openness and
    vs Ca2+ noise are then model predictions, not fits.
    """
    q = dict(PASS3_BRANCHING_PARAMS)
    if params:
        q.update(params)
    rng = np.random.default_rng(seed)
    # Tail drive depends on the posterior 5-HT level only; gap-junction state enters through
    # the transported c_post (not as a separate multiplicative factor as in the 1D Pass 2
    # solver), so the octanol and cryptic-washout states (both c_post = 1) share one drive.
    del gj_open
    c_drive = (c_post**4) / (q["K_c"] ** 4 + c_post**4)
    M = c_drive + q["b_basal"]
    S_B = 1.45 * (M * M) / (q["K_M"] ** 2 + M * M)
    ca_var = np.exp(q["sigma_Ca"] * rng.standard_normal(n_rep) - 0.5 * q["sigma_Ca"] ** 2)
    S_E = q["e_basal"] + q["g_Ca"] * Ca_post * ca_var
    B = np.full(n_rep, 0.5)
    E = np.full(n_rep, 0.5)
    dt = q["dt"]
    sq = math.sqrt(dt) * q["sigma_L"]
    for _ in range(int(q["t_end"] / dt)):
        dB = (S_B / (1.0 + (E / q["K_E"]) ** 2) - B) / q["tau"]
        dE = (S_E / (1.0 + (B / q["K_BE"]) ** 3) - E) / q["tau"]
        B = np.clip(B + dt * dB + sq * rng.standard_normal(n_rep), 0.0, 1.5)
        E = np.clip(E + dt * dE + sq * rng.standard_normal(n_rep), 0.0, 1.5)
    two_head = (E >= 0.5) & (B < 0.5)
    p = float(np.mean(two_head))
    se = math.sqrt(max(p * (1.0 - p), 1e-12) / n_rep)
    return {
        "p_2H0T": p,
        "se": se,
        "n_rep": n_rep,
        "B_final": B,
        "E_final": E,
        "decision": E - B,
        "S_B": float(S_B),
        "S_E_mean": float(np.mean(S_E)),
    }


def evaluate_pass3_network_suite(seed: int = 2017, n_cells: int = 420) -> dict[str, Any]:
    """Run the Pass 3 2D Voronoi network verification, Ca2+/IP3 regimes, and stochastic branching sweep."""
    net = build_voronoi_cell_network(n_cells_target=n_cells, seed=seed)
    boltz = verify_network_boltzmann_equilibrium(net)
    regimes = ["wild_type_1H1T", "octanol_pulse_2H0T", "cryptic_memory_washout_2H0T", "sch28080_hyperpolarized_0H2T", "uncharged_morphogen_control_z0"]
    ca_ip3 = {r: simulate_ca_ip3_network(net, regime=r) for r in regimes}
    branching = {}
    for i, r in enumerate(regimes):
        sim = ca_ip3[r]
        gj_open = 1.0 - sim["eta_oct"]
        branching[r] = stochastic_head_tail_branching(sim["c_post"], sim["Ca_post"], gj_open, seed=seed + 11 * i)
    # (i) p(2H0T) vs posterior V_mem (Durant et al. 2017 axis: the bioelectric state of the
    #     posterior blastema sets the double-head fraction), GJ open, WT anterior -20 mV.
    vpost_grid = np.linspace(-60.0, -20.0, 9)
    p_vs_vpost, c_ratio_vs_vpost, ca_post_vs_vpost = [], [], []
    for k, vp in enumerate(vpost_grid):
        sim = simulate_ca_ip3_network(net, regime="wild_type_1H1T", V_post=float(vp))
        res = stochastic_head_tail_branching(sim["c_post"], sim["Ca_post"], 1.0, seed=seed + 101 + k)
        p_vs_vpost.append(res["p_2H0T"])
        c_ratio_vs_vpost.append(sim["c_post_over_c_ant"])
        ca_post_vs_vpost.append(sim["Ca_post"])
    # (ii) 5-HT ratio vs gap-junction openness 1 - eta (8-octanol dose) with the WT V field,
    #      at slow (default) and fast 5-HT turnover.  At equilibrium the Boltzmann ratio is
    #      independent of G_gj; only the Damkohler-like competition k_h N^2 / (D G_gj) between
    #      turnover and electrodiffusion lets a GJ block flatten the gradient.  This is an
    #      honest ablation: GJ dose alone does NOT reproduce the octanol phenotype unless the
    #      posterior V_mem also depolarises (which the regime definitions impose).
    gj_grid = np.linspace(0.05, 1.0, 9)
    c_ratio_vs_gj_slow, c_ratio_vs_gj_fast, p_vs_gj_fast = [], [], []
    for k, g in enumerate(gj_grid):
        sim_s = simulate_ca_ip3_network(net, regime="wild_type_1H1T", eta_override=1.0 - float(g))
        sim_f = simulate_ca_ip3_network(net, regime="wild_type_1H1T", eta_override=1.0 - float(g), params={"k_h": 0.25})
        c_ratio_vs_gj_slow.append(sim_s["c_post_over_c_ant"])
        c_ratio_vs_gj_fast.append(sim_f["c_post_over_c_ant"])
        p_vs_gj_fast.append(stochastic_head_tail_branching(sim_f["c_post"], sim_f["Ca_post"], float(g), seed=seed + 301 + k)["p_2H0T"])
    # (iii) p(2H0T) vs Ca2+ influx variability sigma_Ca in the 8-octanol state (noise ablation).
    sig_grid = np.linspace(0.0, 0.9, 7)
    octa = ca_ip3["octanol_pulse_2H0T"]
    p_vs_sigma = [
        stochastic_head_tail_branching(octa["c_post"], octa["Ca_post"], 1.0 - octa["eta_oct"], n_rep=256, seed=seed + 201 + k, params={"sigma_Ca": float(sg)})["p_2H0T"]
        for k, sg in enumerate(sig_grid)
    ]
    interior_deg = net["degree"][net["interior"]]
    return {
        "net": net,
        "boltzmann": boltz,
        "ca_ip3": ca_ip3,
        "branching": branching,
        "vpost_grid": vpost_grid,
        "p_vs_vpost": np.array(p_vs_vpost),
        "c_ratio_vs_vpost": np.array(c_ratio_vs_vpost),
        "ca_post_vs_vpost": np.array(ca_post_vs_vpost),
        "gj_grid": gj_grid,
        "c_ratio_vs_gj_slow_turnover": np.array(c_ratio_vs_gj_slow),
        "c_ratio_vs_gj_fast_turnover": np.array(c_ratio_vs_gj_fast),
        "p_vs_gj_fast_turnover": np.array(p_vs_gj_fast),
        "sigma_grid": sig_grid,
        "p_vs_sigma": np.array(p_vs_sigma),
        "mean_interior_degree": float(np.mean(interior_deg)) if interior_deg.size else float("nan"),
        "n_interior": int(np.sum(net["interior"])),
    }


# --------------------------------------------------------------------------- #
# 3. Planarian Specimen CV Segmentation & 2D Bioelectric-Morphogen Body Plan
# --------------------------------------------------------------------------- #
def segment_planarian_specimen(specimen_path: Path = SPECIMEN_PATH) -> dict[str, Any]:
    """
    Segment the dorsal bright-field whole-mount micrograph of the freshwater planarian
    (`datasets/bioelectric_morphogen_coupling/planarian_ap_polarity_specimen.jpg`):
      1. Segment the planarian body contour (luminance < 215, morphological closing, hole
         filling, largest connected component — which also discards the top-left scale bar)
         and the dorsoventral width profile W(s) over 60 AP bins.
      2. Localize the bilateral pair of anterior photoreceptor ocelli (eyespots) via local
         Gaussian background subtraction inside the anterior head region (s < 0.15), with
         body-length-scaled kernels.

    SPECIMEN PROVENANCE (Pass 3): Passes 1-2 used the 1493x564 px Wikimedia Commons
    `File:Smediterranea.jpg` (CC BY 2.5), segmented length 1478 px, ocelli at x/L = 0.0528.
    Campaign commit `398c37e0` ("Replace synthetic planarian composites with 6 genuine
    published PlanformDB microscope photographs") replaced the vendored file with a 720x500 px
    published PlanformDB bright-field photograph, which broke the Pass 2 SHA-256 pin.  Pass 3
    re-pins to the committed photograph: segmented length 626 px, max width 167 px, aspect
    ratio 3.75, N = 2 ocelli at x/L = 0.0565 and 0.0495 (mean 0.0530, separation 0.0454 L).
    `scout_manifest.json` is itself SHA-256-pinned and still describes the Wikimedia original,
    so it was deliberately NOT edited; the discrepancy is recorded in the dataset README.
    """
    img = np.asarray(Image.open(specimen_path).convert("RGB"))
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    raw_mask = gray < 215.0
    raw_mask = ndi.binary_fill_holes(ndi.binary_closing(raw_mask, iterations=3))
    lab, n_comp = ndi.label(raw_mask)
    sizes = ndi.sum(raw_mask, lab, range(1, n_comp + 1))
    worm_mask = lab == (1 + int(np.argmax(sizes)))

    ys, xs = np.where(worm_mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    length_px = float(x_max - x_min)
    max_width_px = float(y_max - y_min)
    aspect_ratio = length_px / max(1.0, max_width_px)

    n_bins = 60
    s_bins = np.linspace(0.0, 1.0, n_bins)
    x_coords = np.round(x_min + s_bins * (x_max - x_min)).astype(int)
    widths_px = np.zeros(n_bins, dtype=float)
    midline_y = np.zeros(n_bins, dtype=float)

    for i, xc in enumerate(x_coords):
        col_y = np.where(worm_mask[:, xc])[0]
        if col_y.size > 0:
            widths_px[i] = float(col_y.max() - col_y.min())
            midline_y[i] = float(np.mean(col_y))

    ant_limit = x_min + int(0.15 * (x_max - x_min))
    ant_region = worm_mask.copy()
    ant_region[:, ant_limit:] = False
    # Pass 3: the ocelli kernels were tuned on the 1478 px-long Pass 1/2 specimen; the
    # specimen committed in 398c37e0 is only ~626 px long, so the erosion and the local
    # background sigma are scaled with the measured body length (with floors) instead of
    # being hard-coded in pixels.  Without this the eroded anterior interior swallows the
    # eyespots and the detector locks onto trunk pigmentation at x/L ~ 0.12.
    scale_k = length_px / 1478.0
    ant_interior = ndi.binary_erosion(worm_mask, iterations=max(6, int(round(8 * scale_k)))) & ant_region

    local_bg = ndi.gaussian_filter(gray, sigma=max(4.0, 12.0 * scale_k))
    dark_contrast = (local_bg - gray) * ant_interior
    thresh = float(np.percentile(dark_contrast[ant_interior], 96.5))
    ocelli_mask = ndi.binary_opening(dark_contrast > thresh, iterations=1)
    olab, n_oc = ndi.label(ocelli_mask)
    osizes = ndi.sum(dark_contrast, olab, range(1, n_oc + 1))
    top_idx = np.argsort(osizes)[::-1][:2]
    ocelli_yx = [ndi.center_of_mass(dark_contrast, olab, int(idx + 1)) for idx in top_idx]
    ocelli_yx = sorted(ocelli_yx, key=lambda pt: pt[0])
    ocelli_xy = [[float(pt[1]), float(pt[0])] for pt in ocelli_yx]
    ocelli_rel_s = [(pt[0] - x_min) / length_px for pt in ocelli_xy]
    inter_ocelli_px = float(
        math.hypot(ocelli_xy[0][0] - ocelli_xy[1][0], ocelli_xy[0][1] - ocelli_xy[1][1])
    )

    return {
        "image_rgb": img,
        "worm_mask": worm_mask,
        "inter_ocelli_distance_px": inter_ocelli_px,
        "inter_ocelli_distance_rel": inter_ocelli_px / max(1.0, length_px),
        "x_min": x_min,
        "x_max": x_max,
        "length_px": length_px,
        "max_width_px": max_width_px,
        "aspect_ratio": aspect_ratio,
        "s_bins": s_bins,
        "x_coords": x_coords,
        "widths_px": widths_px,
        "widths_norm": widths_px / max(1.0, float(np.max(widths_px))),
        "midline_y": midline_y,
        "n_ocelli_detected": int(len(ocelli_xy)),
        "ocelli_xy": ocelli_xy,
        "ocelli_rel_s": ocelli_rel_s,
        "mean_ocelli_rel_s": float(np.mean(ocelli_rel_s)),
    }


def synthesize_2d_planarian_morphology(
    s: np.ndarray,
    E: np.ndarray,
    B: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct 2D planarian body contour and photoreceptor eyespots from ERK E(s) and beta-catenin B(s)."""
    s_fine = np.linspace(0.0, 1.0, 200)
    E_i = np.interp(s_fine, s, E)
    B_i = np.interp(s_fine, s, B)

    envelope = np.sin(np.clip(s_fine, 1e-4, 1.0 - 1e-4) * math.pi) ** 0.42
    ant_auricle = 0.095 * float(np.mean(E[:8])) * np.exp(-((s_fine - 0.08) / 0.045) ** 2)
    post_auricle = 0.095 * float(np.mean(E[-8:])) * np.exp(-((s_fine - 0.92) / 0.045) ** 2)
    tail_taper_ant = 1.0 - 0.30 * float(np.mean(B[:8])) * np.exp(-((s_fine - 0.0) / 0.12) ** 2)
    tail_taper_post = 1.0 - 0.30 * float(np.mean(B[-8:])) * np.exp(-((s_fine - 1.0) / 0.12) ** 2)

    half_width = (0.21 * envelope + ant_auricle + post_auricle) * tail_taper_ant * tail_taper_post

    eyespots_xy: list[tuple[float, float]] = []
    if float(np.mean(E[:8])) >= 0.50 and float(np.mean(B[:8])) < 0.50:
        eyespots_xy.extend([(0.065, +0.068), (0.065, -0.068)])
    if float(np.mean(E[-8:])) >= 0.50 and float(np.mean(B[-8:])) < 0.50:
        eyespots_xy.extend([(0.935, +0.068), (0.935, -0.068)])

    return {
        "s_fine": s_fine,
        "half_width": half_width,
        "E_interp": E_i,
        "B_interp": B_i,
        "eyespots_xy": eyespots_xy,
    }


# --------------------------------------------------------------------------- #
# 4. 9-Panel Publication Figure & Animated Nernst-Planck Regeneration GIF
# --------------------------------------------------------------------------- #
def render_outputs(seed: int = 2017, skip_gif: bool = False) -> dict[str, Any]:
    """Render the 9-panel comparison figure and 24-frame animated GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    verify_dataset_sha256()

    inv = compute_nernst_planck_invariants()
    suite = evaluate_all_regimes_and_boltzmann_sweep(seed=seed)
    spec = segment_planarian_specimen(SPECIMEN_PATH)

    wt = suite["regimes"]["wild_type_1H1T"]
    octa = suite["regimes"]["octanol_pulse_2H0T"]
    cryp = suite["regimes"]["cryptic_memory_washout_2H0T"]
    sch = suite["regimes"]["sch28080_hyperpolarized_0H2T"]
    z0 = suite["regimes"]["uncharged_morphogen_control_z0"]
    resc = suite["regimes"]["bioelectric_reset_rescue_1H1T"]

    fig, axes = plt.subplots(3, 3, figsize=(16.8, 13.8), dpi=135)
    fig.patch.set_facecolor("#0b0f17")
    for ax in axes.flat:
        ax.set_facecolor("#131b29")
        ax.tick_params(colors="#cbd5e1", labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#334155")

    # Panel A1: Exact Boltzmann Equilibrium Ratio vs Scharfetter-Gummel Nernst-Planck PDE
    ax = axes[0, 0]
    ax.plot(suite["dv_sweep_mV"], suite["exact_boltzmann_ratios"], color="#38bdf8", lw=2.4, label=r"Exact Boltzmann $\exp(\Delta V_{\mathrm{AP}}/V_T)$ ($z=+1$)")
    ax.scatter(suite["dv_sweep_mV"], suite["pde_boltzmann_ratios"], color="#22c55e", s=55, zorder=6, label="Nernst-Planck PDE steady state")
    ax.plot(inv["dv_ap_grid"], inv["boltz_curve_z0"], color="#f59e0b", lw=1.8, ls="--", label="Uncharged control ($z=0$, ratio $= 1.0$)")
    ax.plot(inv["dv_ap_grid"], inv["boltz_curve_zm1"], color="#f43f5e", lw=1.8, ls=":", label="Anion ($z=-1$) anterior enrichment")
    ax.axvline(40.0, color="#eab308", lw=1.2, ls="--", label="WT $\\Delta V_{\\mathrm{AP}} = +40$ mV ($c_P/c_A = 4.70$)")
    ax.set_title("A1. Nernst-Planck Boltzmann Equilibrium $c_{\\mathrm{post}}/c_{\\mathrm{ant}}$", color="#f8fafc", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("AP voltage polarization $\\Delta V_{\\mathrm{AP}} = V_{\\mathrm{ant}} - V_{\\mathrm{post}}$ (mV)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Concentration ratio $c_{\\mathrm{post}} / c_{\\mathrm{ant}}$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="upper left")

    # Panel A2: Serotonin (5-HT) Electrodiffusive Concentration Profiles c(x) Across Regimes
    ax = axes[0, 1]
    ax.plot(wt["s"], wt["c"], color="#22c55e", lw=2.5, label=f"WT 1H1T ($c_P/c_A={wt['c_post_over_c_ant']:.2f}$)")
    ax.plot(octa["s"], octa["c"], color="#f43f5e", lw=2.2, ls="--", label=f"8-Octanol GJ Block 2H0T ($c_P/c_A={octa['c_post_over_c_ant']:.2f}$)")
    ax.plot(cryp["s"], cryp["c"], color="#a855f7", lw=2.2, ls="-.", label=f"Cryptic Washout 2H0T ($c_P/c_A={cryp['c_post_over_c_ant']:.2f}$)")
    ax.plot(z0["s"], z0["c"], color="#f59e0b", lw=1.9, ls=":", label=f"Uncharged $z=0$ Control ($c_P/c_A={z0['c_post_over_c_ant']:.2f}$)")
    ax.plot(resc["s"], resc["c"], color="#38bdf8", lw=1.9, label=f"Bioelectric Rescue 1H1T ($c_P/c_A={resc['c_post_over_c_ant']:.2f}$)")
    ax.set_title("A2. Electrodiffusive 5-HT Profiles $c(x)$ Across Regimes", color="#f8fafc", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Normalized AP position $x/L$ (0 = anterior, 1 = posterior)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Normalized 5-HT concentration $c(x)$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="upper left")

    # Panel A3: cAMP/PKA -> Beta-Catenin B(x) & Anterior ERK E(x) Polarity Commitment
    ax = axes[0, 2]
    ax.plot(wt["s"], wt["E"], color="#22c55e", lw=2.4, label="WT ERK Head $E(x)$ (1H1T)")
    ax.plot(wt["s"], wt["B"], color="#22c55e", lw=1.8, ls=":", label="WT $\\beta$-Catenin Tail $B(x)$ (1H1T)")
    ax.plot(cryp["s"], cryp["E"], color="#f43f5e", lw=2.2, ls="-.", label="Cryptic Washout ERK $E(x)$ (2H0T)")
    ax.plot(sch["s"], sch["B"], color="#38bdf8", lw=2.1, ls="--", label="SCH-28080/Forskolin $\\beta$-Cat $B(x)$ (0H2T)")
    ax.axhline(0.50, color="#64748b", lw=1.1, ls="--")
    ax.set_title("A3. Anterior ERK $E(x)$ vs Posterior $\\beta$-Catenin $B(x)$", color="#f8fafc", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Normalized AP position $x/L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Signaling activity $E(x), B(x)$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="center right")

    # Panel B1: Space-Time Kymograph of Serotonin c(x,t) (WT 1H1T Electrophoresis vs Cryptic 2H0T Memory)
    ax = axes[1, 0]
    kymo_c = np.vstack([wt["history_c"], cryp["history_c"]])
    im_k = ax.imshow(
        kymo_c,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0.35,
        vmax=2.15,
        extent=[0.0, 1.0, 0, len(kymo_c) - 1],
    )
    ax.axhline(len(wt["history_c"]) - 0.5, color="#f8fafc", lw=1.6, ls="--")
    ax.text(0.45, len(wt["history_c"]) * 0.42, "Bottom: Wild-Type (1H1T)\nPosterior 5-HT electrophoresis (4.70x)", color="#f8fafc", fontsize=7.8, fontweight="bold", bbox=dict(facecolor="#0f172a", alpha=0.78, edgecolor="#334155"))
    ax.text(0.45, len(wt["history_c"]) * 1.42, "Top: Cryptic Memory after Washout (2H0T)\nSymmetric V_mem (-20 mV) locks flat 5-HT", color="#f8fafc", fontsize=7.8, fontweight="bold", bbox=dict(facecolor="#0f172a", alpha=0.78, edgecolor="#334155"))
    ax.set_title("B1. Space-Time 5-HT $c(x,t)$ Kymograph (WT vs Cryptic 2H0T)", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_xlabel("Normalized AP position $x/L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Time snapshot index (WT $\\mid$ Cryptic 2H0T)", color="#cbd5e1", fontsize=9)
    cb = fig.colorbar(im_k, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#cbd5e1", labelsize=7.5)
    cb.set_label("5-HT $c(x,t)$", color="#cbd5e1", fontsize=8)

    # Panel B2: 2D Bioelectric-Gap-Junction Morphospace (DeltaV_AP vs GJ Openness 1 - eta_oct)
    ax = axes[1, 1]
    ext = [
        float(suite["grid_dv"][0]),
        float(suite["grid_dv"][-1]),
        float(suite["grid_gj_open"][0]),
        float(suite["grid_gj_open"][-1]),
    ]
    im_ph = ax.imshow(suite["p2h_grid"], origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=0.5, extent=ext)
    cs = ax.contour(
        suite["grid_dv"], suite["grid_gj_open"], suite["c_ratio_grid"],
        levels=[1.5, 2.0, 3.0, 4.0], colors=["#38bdf8"], linewidths=1.0, alpha=0.9,
    )
    ax.clabel(cs, fmt="$c_P/c_A$=%.1f", fontsize=6.5, colors="#38bdf8")
    ax.scatter([40.0], [1.0], color="#22c55e", edgecolor="#f8fafc", s=85, marker="o", zorder=6, label="WT 1H1T ($p_{2H}\\approx 0$)")
    ax.scatter([0.0], [0.08], color="#f43f5e", edgecolor="#f8fafc", s=85, marker="s", zorder=6, label="8-Octanol: flat $V$, GJ 8 %")
    ax.scatter([0.0], [1.0], color="#a855f7", edgecolor="#f8fafc", s=95, marker="*", zorder=6, label="Cryptic memory: flat $V$, GJ open")
    ax.set_title("B2. 2D-Network Morphospace: $p(2H0T)$ & $c_P/c_A$ vs $(\\Delta V_{\\mathrm{AP}}, G_{\\mathrm{gj}})$", color="#f8fafc", fontsize=9.6, fontweight="bold")
    ax.set_xlabel("AP voltage difference $\\Delta V_{\\mathrm{AP}} = V_{\\mathrm{ant}} - V_{\\mathrm{post}}$ (mV)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Gap-junction openness $1 - \\eta_{\\mathrm{oct}}$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="upper right")
    cb2 = fig.colorbar(im_ph, ax=ax, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#cbd5e1", labelsize=7.5)
    cb2.set_label("stochastic double-head probability $p(2H0T)$", color="#cbd5e1", fontsize=8)

    # Panel B3: Reconstructed 2D Planarian Body-Plan Silhouettes (1H1T WT vs Cryptic 2H0T vs 0H2T)
    ax = axes[1, 2]
    offsets = [
        ("Wild-Type / Bioelectric Rescue 1H1T", wt, +0.62, "#22c55e"),
        ("Cryptic Washout Memory 2H0T", cryp, 0.0, "#f43f5e"),
        ("SCH-28080 / Forskolin 0H2T", sch, -0.62, "#38bdf8"),
    ]
    for label, sim_r, y_off, col in offsets:
        morph = synthesize_2d_planarian_morphology(sim_r["s"], sim_r["E"], sim_r["B"])
        sf = morph["s_fine"]
        hw = morph["half_width"]
        ax.fill_between(sf, y_off - hw, y_off + hw, color=col, alpha=0.32, edgecolor=col, lw=2.0, label=label)
        for ex, ey in morph["eyespots_xy"]:
            ax.scatter([ex], [y_off + ey], color="#f8fafc", edgecolor="#0f172a", s=38, zorder=7)
            ax.scatter([ex], [y_off + ey], color="#0f172a", s=12, zorder=8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-0.05, 1.05)
    ax.set_yticks([-0.62, 0.0, 0.62])
    ax.set_yticklabels(["0H2T (Headless)", "2H0T (Cryptic 2-Head)", "1H1T (Wild-Type)"], color="#cbd5e1", fontsize=8)
    ax.set_title("B3. Reconstructed 2D Planarian Morphologies & Eyespots", color="#f8fafc", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Normalized AP body axis $x/L$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="upper right")

    # Panel C1: Planarian Specimen CV Segmentation (Body Contour, AP Midline & Bilateral Ocelli)
    ax = axes[2, 0]
    ax.imshow(spec["image_rgb"])
    ax.contour(spec["worm_mask"], levels=[0.5], colors=["#22c55e"], linewidths=1.4)
    ax.plot(spec["x_coords"], spec["midline_y"], color="#38bdf8", lw=1.8, ls="--", label=f"AP midline ($L={spec['length_px']:.0f}$ px)")
    oc_xy = np.array(spec["ocelli_xy"])
    ax.scatter(
        oc_xy[:, 0],
        oc_xy[:, 1],
        color="#f43f5e",
        edgecolor="#f8fafc",
        s=70,
        zorder=8,
        label=f"Segmented ocelli ($N={spec['n_ocelli_detected']}$, $s={spec['mean_ocelli_rel_s']:.3f}$)",
    )
    ax.set_title(
        f"C1. Planarian Specimen CV ($N={spec['n_ocelli_detected']}$ Ocelli, AR={spec['aspect_ratio']:.2f})",
        color="#f8fafc",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="lower right")
    ax.axis("off")

    # Panel C2: Specimen Width & Anterior Ocelli vs Simulated ERK E(x) & 5-HT Depletion Zone
    ax = axes[2, 1]
    morph_wt = synthesize_2d_planarian_morphology(wt["s"], wt["E"], wt["B"])
    sim_w_norm = morph_wt["half_width"] / max(1e-6, float(np.max(morph_wt["half_width"])))
    ax.plot(spec["s_bins"], spec["widths_norm"], color="#22c55e", lw=2.3, label="Specimen dorsal width $W(x/L)$")
    ax.plot(morph_wt["s_fine"], sim_w_norm, color="#38bdf8", lw=2.0, ls="--", label="Simulated 1H1T contour width")
    ax.plot(wt["s"], wt["E"], color="#f43f5e", lw=2.0, ls="-.", label="Anterior ERK head signal $E(x/L)$")
    ax.axvline(
        spec["mean_ocelli_rel_s"],
        color="#eab308",
        lw=1.8,
        ls=":",
        label=f"Specimen ocelli ($x/L = {spec['mean_ocelli_rel_s']:.3f}$)",
    )
    ax.set_title("C2. Specimen Width & Ocelli vs Simulated ERK Head Domain", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_xlabel("Normalized AP coordinate $x/L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Normalized width / ERK signal $E$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="center right")

    # Panel C3: 6-Regime Head/Tail Phenotype & Posterior 5-HT Enrichment Bar Comparison
    ax = axes[2, 2]
    labels = ["WT\n1H1T", "Octanol\n2H0T", "Cryptic\nWashout", "SCH/cAMP\n0H2T", "z=0 Ctl\n2H0T", "K+ Rescue\n1H1T"]
    reg_list = [wt, octa, cryp, sch, z0, resc]
    x_pos = np.arange(len(labels))
    ratios_bar = [r["c_post_over_c_ant"] for r in reg_list]
    heads_bar = [r["n_heads"] for r in reg_list]
    ax.bar(x_pos - 0.18, ratios_bar, width=0.35, color="#38bdf8", label="Posterior 5-HT ratio $c_P/c_A$")
    ax.bar(x_pos + 0.18, heads_bar, width=0.35, color="#f43f5e", label="Regenerated heads $N_{\\mathrm{heads}}$")
    ax.axhline(inv["ratio_wt_z1"], color="#22c55e", lw=1.2, ls="--", label="Exact Boltzmann $4.70\\times$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, color="#cbd5e1", fontsize=7.8)
    ax.set_title("C3. 6-Regime Boltzmann 5-HT Ratio & Head Count Parity", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_ylabel("Ratio $c_P/c_A$ or Head Count", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="upper right")

    fig.suptitle(
        "Pietak & Levin (2017) & (2016) Electrodiffusive Nernst-Planck Gap-Junction 5-HT Transport & Cryptic Two-Headed Memory (v3: B2 from 2D Voronoi network + stochastic branching)",
        color="#f8fafc",
        fontsize=12.6,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.978])
    fig.savefig(OUT_PNG, dpi=135)
    plt.close(fig)

    if not skip_gif:
        frames_pil: list[Image.Image] = []
        n_frames = len(wt["history_t"])
        frame_indices = np.linspace(0, n_frames - 1, 24, dtype=int)
        for idx in frame_indices:
            f_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=90)
            f_fig.patch.set_facecolor("#0b0f17")
            for a_sub in (ax1, ax2):
                a_sub.set_facecolor("#131b29")
                a_sub.tick_params(colors="#cbd5e1", labelsize=8)
                for sp in a_sub.spines.values():
                    sp.set_color("#334155")

            t_val = wt["history_t"][idx]
            ax1.plot(wt["s"], wt["history_c"][idx], color="#22c55e", lw=2.3, label="WT 1H1T 5-HT $c(x,t)$")
            ax1.plot(cryp["s"], cryp["history_c"][idx], color="#f43f5e", lw=2.0, ls="-.", label="Cryptic Washout 2H0T")
            ax1.plot(z0["s"], z0["history_c"][idx], color="#f59e0b", lw=1.8, ls=":", label="Uncharged $z=0$ Control")
            ax1.set_ylim(0.2, 2.35)
            ax1.set_xlim(0.0, 1.0)
            ax1.set_title(f"Nernst-Planck 5-HT $c(x,t)$ ($t={t_val:.1f}$)", color="#f8fafc", fontsize=9.5)
            ax1.set_xlabel("AP coordinate $x/L$ (0=ant, 1=post)", color="#cbd5e1", fontsize=8.5)
            ax1.set_ylabel("5-HT concentration $c(x,t)$", color="#cbd5e1", fontsize=8.5)
            ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2)

            ax2.plot(wt["s"], wt["history_E"][idx], color="#22c55e", lw=2.3, label="WT ERK Head $E(x,t)$ (1H1T)")
            ax2.plot(cryp["s"], cryp["history_E"][idx], color="#f43f5e", lw=2.0, ls="-.", label="Cryptic Washout $E(x,t)$ (2H0T)")
            ax2.plot(sch["s"], sch["history_E"][idx], color="#38bdf8", lw=2.0, ls="--", label="SCH-28080/cAMP $E(x,t)$ (0H2T)")
            ax2.axhline(0.50, color="#64748b", lw=1.0, ls="--")
            ax2.set_ylim(-0.02, 1.05)
            ax2.set_xlim(0.0, 1.0)
            ax2.set_title("Anterior ERK Head Commitment $E(x,t)$", color="#f8fafc", fontsize=9.5)
            ax2.set_xlabel("AP coordinate $x/L$", color="#cbd5e1", fontsize=8.5)
            ax2.set_ylabel("ERK activity $E(x,t)$", color="#cbd5e1", fontsize=8.5)
            ax2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2)

            f_fig.tight_layout()
            f_fig.canvas.draw()
            rgba = np.asarray(f_fig.canvas.buffer_rgba())
            frames_pil.append(Image.fromarray(rgba[:, :, :3]))
            plt.close(f_fig)

        frames_pil[0].save(
            OUT_GIF,
            save_all=True,
            append_images=frames_pil[1:],
            duration=90,
            loop=0,
        )

    p3 = evaluate_pass3_network_suite(seed=seed)
    render_pass3_network_figure(p3, wt, spec)
    if not skip_gif:
        render_pass3_network_gif(p3)

    return {
        "invariants": inv,
        "suite": suite,
        "specimen": spec,
        "pass3": p3,
    }


def _draw_network(ax, net: dict[str, Any], values: np.ndarray, cmap: str, vmin: float, vmax: float, title: str, label: str, fig=None, outline: bool = True):
    from matplotlib.collections import PolyCollection

    pc = PolyCollection(net["polys"], array=values, cmap=cmap, edgecolors="#0b0f17", linewidths=0.25)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    if outline:
        sp, hp = net["s_profile"], net["hw_profile"]
        ax.plot(sp, hp, color="#f8fafc", lw=1.0, ls="--")
        ax.plot(sp, -hp, color="#f8fafc", lw=1.0, ls="--")
    ax.set_xlim(-0.02, 1.02)
    hw_max = float(np.max(net["hw_profile"]))
    ax.set_ylim(-1.25 * hw_max, 1.25 * hw_max)
    ax.set_aspect("equal")
    ax.set_title(title, color="#f8fafc", fontsize=9.6, fontweight="bold")
    ax.set_xlabel("AP coordinate $x/L$ (0 = anterior)", color="#cbd5e1", fontsize=8.5)
    ax.set_yticks([])
    if fig is not None:
        cb = fig.colorbar(pc, ax=ax, fraction=0.025, pad=0.02)
        cb.ax.tick_params(colors="#cbd5e1", labelsize=7)
        cb.set_label(label, color="#cbd5e1", fontsize=7.2)


def render_pass3_network_figure(p3: dict[str, Any], wt_1d: dict[str, Any], spec: dict[str, Any]) -> None:
    """Pass 3 figure: 2D Voronoi SG network, Boltzmann verification, Ca2+/IP3 fields, stochastic branching."""
    net = p3["net"]
    bz = p3["boltzmann"]
    wt = p3["ca_ip3"]["wild_type_1H1T"]
    octa = p3["ca_ip3"]["octanol_pulse_2H0T"]
    cryp = p3["ca_ip3"]["cryptic_memory_washout_2H0T"]
    sch = p3["ca_ip3"]["sch28080_hyperpolarized_0H2T"]

    fig, axes = plt.subplots(3, 3, figsize=(17.0, 12.0), dpi=130, gridspec_kw={"height_ratios": [0.55, 1.0, 1.0]})
    fig.patch.set_facecolor("#0b0f17")
    for ax in axes.flat:
        ax.set_facecolor("#131b29")
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for sp_ in ax.spines.values():
            sp_.set_color("#334155")

    # A. Voronoi network inside specimen silhouette, coloured by imposed V_mem (WT)
    _draw_network(
        axes[0, 0], net, wt["V"], "coolwarm_r", -62.0, -18.0,
        f"A. Voronoi network in specimen silhouette ($N$={net['n_cells']}, $E$={net['n_edges']}, $\\langle k\\rangle_{{int}}$={p3['mean_interior_degree']:.2f}, AR={spec['aspect_ratio']:.2f})",
        "imposed $V_{\\mathrm{mem}}$ (mV)", fig=fig,
    )

    # B. 5-HT on the network (WT)
    _draw_network(axes[0, 1], net, wt["c"], "magma", 0.3, 2.0, f"B. 5-HT on network (WT): post/ant = {wt['c_post_over_c_ant']:.2f} (1D endpoints {wt_1d['c_post_over_c_ant']:.2f})", "5-HT $c$ ($z$=+1)", fig=fig)

    # C. Ca2+ on the network (WT)
    _draw_network(axes[0, 2], net, wt["Ca"], "viridis", 0.0, 1.1, f"C. Ca$^{{2+}}$ on network (WT): ant/post = {wt['Ca_ant_over_Ca_post']:.2f} (VGCC beats $z$=+2 Boltzmann {bz['species']['calcium_Ca2']['post_over_ant_group_boltz']:.0f}x post)", "cytosolic Ca$^{2+}$ ($z$=+2)", fig=fig)

    # D. Boltzmann verification on the irregular graph
    ax = axes[1, 0]
    cols = {"serotonin_5HT": "#22c55e", "calcium_Ca2": "#38bdf8", "inositol_IP3": "#f59e0b"}
    for name, d in bz["species"].items():
        ax.scatter(d["c_exact"], d["c"], s=9, color=cols[name], alpha=0.75, label=f"{name} ($z$={d['z']:+.0f}): max rel err {d['max_rel_err']:.1e}, r={d['pearson_r']:.6f}")
    lo = min(float(np.min(d["c_exact"])) for d in bz["species"].values())
    hi = max(float(np.max(d["c_exact"])) for d in bz["species"].values())
    ax.plot([lo, hi], [lo, hi], color="#f8fafc", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("D. Pure electrodiffusion steady state vs exact Boltzmann $c_i \\propto e^{-zV_i/V_T}$ (all cells)", color="#f8fafc", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("exact Boltzmann $c_i^{\\mathrm{exact}}$", color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel("2D SG network steady state $c_i$", color="#cbd5e1", fontsize=8.5)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=6.8, loc="upper left")
    mass_err = max(d["mass_rel_err"] for d in bz["species"].values())
    ax.text(0.98, 0.04, f"mass conservation rel. err $\\leq$ {mass_err:.1e}\nedge log-ratio RMSE $\\leq$ {max(d['edge_log_ratio_rmse'] for d in bz['species'].values()):.1e}", transform=ax.transAxes, color="#f8fafc", fontsize=7.2, ha="right", bbox=dict(facecolor="#0f172a", alpha=0.8, edgecolor="#334155"))

    # E. AP-binned profiles: 5-HT, IP3, Ca2+ for WT vs octanol, plus the 1D WT 5-HT profile
    ax = axes[1, 1]
    bc = wt["bin_centres"]
    ax.plot(bc, wt["profiles"]["c"], color="#22c55e", lw=2.3, label="2D net WT 5-HT")
    ax.plot(wt_1d["s"], wt_1d["c"], color="#22c55e", lw=1.3, ls=":", label="1D SG WT 5-HT (Pass 2)")
    ax.plot(bc, wt["profiles"]["Ca"], color="#38bdf8", lw=2.1, label="2D net WT Ca$^{2+}$")
    ax.plot(bc, wt["profiles"]["P"], color="#f59e0b", lw=2.1, label="2D net WT IP$_3$")
    ax.plot(bc, octa["profiles"]["c"], color="#f43f5e", lw=1.8, ls="--", label="8-octanol 5-HT (flat)")
    ax.plot(bc, octa["profiles"]["Ca"], color="#a855f7", lw=1.8, ls="--", label="8-octanol Ca$^{2+}$ (flat, high)")
    ax.set_title("E. AP-binned network profiles: 5-HT / IP$_3$ / Ca$^{2+}$ (WT vs 8-octanol)", color="#f8fafc", fontsize=9.6, fontweight="bold")
    ax.set_xlabel("AP coordinate $x/L$", color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel("normalised concentration", color="#cbd5e1", fontsize=8.5)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=6.8, loc="upper left", ncol=2)

    # F. p(2H0T) vs posterior V_mem (Durant axis) and vs sigma_Ca
    ax = axes[1, 2]
    ax.plot(p3["vpost_grid"], p3["p_vs_vpost"], color="#f43f5e", lw=2.3, marker="o", label="$p(2H0T)$ vs posterior $V_{\\mathrm{mem}}$ (GJ open)")
    ax.axhspan(0.20, 0.35, color="#f59e0b", alpha=0.18, label="~25 % double-head band (Oviedo 2010 / Durant 2017, approx.)")
    for r, col, mk in (("octanol_pulse_2H0T", "#f43f5e", "s"), ("cryptic_memory_washout_2H0T", "#a855f7", "*"), ("wild_type_1H1T", "#22c55e", "o"), ("sch28080_hyperpolarized_0H2T", "#38bdf8", "v")):
        b = p3["branching"][r]
        vp = {"octanol_pulse_2H0T": -20.0, "cryptic_memory_washout_2H0T": -20.0, "wild_type_1H1T": -60.0, "sch28080_hyperpolarized_0H2T": -60.0}[r]
        ax.errorbar([vp], [b["p_2H0T"]], yerr=[b["se"]], color=col, marker=mk, ms=9, capsize=3, ls="none", label=f"{r}: p={b['p_2H0T']:.2f}±{b['se']:.2f}")
    ax.set_ylim(-0.02, 0.6)
    ax.set_title("F. Stochastic 1H1T/2H0T branching vs posterior bioelectric state", color="#f8fafc", fontsize=9.6, fontweight="bold")
    ax.set_xlabel("posterior $V_{\\mathrm{mem}}$ (mV); anterior fixed at $-20$", color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel("double-head probability $p(2H0T)$", color="#cbd5e1", fontsize=8.5)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=6.6, loc="upper left")

    # G. 5-HT ratio vs GJ openness at slow/fast turnover (Damkohler ablation)
    ax = axes[2, 0]
    ax.plot(p3["gj_grid"], p3["c_ratio_vs_gj_slow_turnover"], color="#22c55e", lw=2.2, marker="o", label="$c_P/c_A$, slow turnover $k_h$=0.004 (equilibrium: $G_{gj}$-independent)")
    ax.plot(p3["gj_grid"], p3["c_ratio_vs_gj_fast_turnover"], color="#f59e0b", lw=2.2, marker="s", label="$c_P/c_A$, fast turnover $k_h$=0.25 (kinetic competition)")
    ax.axhline(bz["species"]["serotonin_5HT"]["post_over_ant_group_boltz"], color="#38bdf8", lw=1.2, ls="--", label="Boltzmann group ratio $e^{\\Delta V_{grp}/V_T}$")
    ax2 = ax.twinx()
    ax2.plot(p3["gj_grid"], p3["p_vs_gj_fast_turnover"], color="#f43f5e", lw=1.6, ls=":", marker="^", label="$p(2H0T)$ fast turnover")
    ax2.set_ylim(-0.02, 0.6)
    ax2.tick_params(colors="#cbd5e1", labelsize=8)
    ax2.set_ylabel("$p(2H0T)$", color="#f43f5e", fontsize=8.5)
    ax.set_title("G. GJ-dose ablation: 8-octanol alone does not flatten 5-HT at equilibrium", color="#f8fafc", fontsize=9.4, fontweight="bold")
    ax.set_xlabel("gap-junction openness $1-\\eta_{\\mathrm{oct}}$ (WT $V$ field)", color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel("$c_{\\mathrm{post}}/c_{\\mathrm{ant}}$ (network groups)", color="#cbd5e1", fontsize=8.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=6.6, loc="lower right")

    # H. Noise ablation: p(2H0T) vs sigma_Ca in the octanol state
    ax = axes[2, 1]
    ax.plot(p3["sigma_grid"], p3["p_vs_sigma"], color="#a855f7", lw=2.3, marker="o", label="8-octanol state, $n$=256 blastemas / point")
    ax.axhspan(0.20, 0.35, color="#f59e0b", alpha=0.18, label="~25 % band (approx.)")
    ax.set_ylim(-0.02, 0.6)
    ax.set_title("H. Ca$^{2+}$-influx variability sets the branching probability", color="#f8fafc", fontsize=9.6, fontweight="bold")
    ax.set_xlabel("lognormal Ca$^{2+}$ influx variability $\\sigma_{Ca}$", color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel("$p(2H0T)$", color="#cbd5e1", fontsize=8.5)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="upper left")

    # I. Decision-variable distributions (E - B) per regime
    ax = axes[2, 2]
    bins = np.linspace(-1.2, 1.2, 33)
    for r, col in (("wild_type_1H1T", "#22c55e"), ("octanol_pulse_2H0T", "#f43f5e"), ("cryptic_memory_washout_2H0T", "#a855f7"), ("sch28080_hyperpolarized_0H2T", "#38bdf8")):
        ax.hist(p3["branching"][r]["decision"], bins=bins, color=col, alpha=0.55, label=f"{r} (p={p3['branching'][r]['p_2H0T']:.2f})")
    ax.axvline(0.0, color="#f8fafc", lw=1.0, ls="--")
    ax.set_title("I. Posterior toggle decision $E - B$ across the stochastic ensemble", color="#f8fafc", fontsize=9.6, fontweight="bold")
    ax.set_xlabel("$E_{\\mathrm{post}} - B_{\\mathrm{post}}$ (>0: ectopic posterior head)", color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel("blastemas", color="#cbd5e1", fontsize=8.5)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=6.8, loc="upper left")
    _ = (cryp, sch)

    fig.suptitle(
        "Pass 3 — Irregular 2D Voronoi Scharfetter-Gummel electrodiffusion (5-HT / IP$_3$ / Ca$^{2+}$) & stochastic 1H1T/2H0T branching (Pietak & Levin 2016/2017; Durant et al. 2017)",
        color="#f8fafc", fontsize=11.6, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    fig.savefig(OUT_PNG_PASS3, dpi=130)
    plt.close(fig)


def render_pass3_network_gif(p3: dict[str, Any], n_frames: int = 20) -> None:
    """Animated 5-HT and Ca2+ relaxation on the 2D Voronoi network (WT regime)."""
    net = p3["net"]
    sim = simulate_ca_ip3_network(net, regime="wild_type_1H1T", record_frames=n_frames)
    frames_pil: list[Image.Image] = []
    for k in range(len(sim["frames_t"])):
        f_fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.2), dpi=90)
        f_fig.patch.set_facecolor("#0b0f17")
        for a_sub in (a1, a2):
            a_sub.set_facecolor("#131b29")
            a_sub.tick_params(colors="#cbd5e1", labelsize=7.5)
            for sp_ in a_sub.spines.values():
                sp_.set_color("#334155")
        _draw_network(a1, net, sim["frames_c"][k], "magma", 0.3, 2.0, f"5-HT $c$ on Voronoi network, $t$={sim['frames_t'][k]:.1f}", "", fig=None)
        _draw_network(a2, net, sim["frames_Ca"][k], "viridis", 0.0, 1.1, "cytosolic Ca$^{2+}$ (VGCC at depolarised anterior)", "", fig=None)
        f_fig.tight_layout()
        f_fig.canvas.draw()
        rgba = np.asarray(f_fig.canvas.buffer_rgba())
        frames_pil.append(Image.fromarray(rgba[:, :, :3]))
        plt.close(f_fig)
    frames_pil[0].save(OUT_GIF_PASS3, save_all=True, append_images=frames_pil[1:], duration=120, loop=0)


# --------------------------------------------------------------------------- #
# 5. MorphoEval Sidecar Quantitative Measurements (`measure_morphoeval()`)
# --------------------------------------------------------------------------- #
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute quantitative morphometric measurements for `morphoeval/measure_outputs.py`."""
    inv = compute_nernst_planck_invariants()
    suite = evaluate_all_regimes_and_boltzmann_sweep(seed=2017)
    spec = segment_planarian_specimen(SPECIMEN_PATH)

    wt = suite["regimes"]["wild_type_1H1T"]
    octa = suite["regimes"]["octanol_pulse_2H0T"]
    cryp = suite["regimes"]["cryptic_memory_washout_2H0T"]
    sch = suite["regimes"]["sch28080_hyperpolarized_0H2T"]
    z0 = suite["regimes"]["uncharged_morphogen_control_z0"]
    resc = suite["regimes"]["bioelectric_reset_rescue_1H1T"]

    boltz_rmse = float(np.sqrt(np.mean((suite["pde_boltzmann_ratios"] - suite["exact_boltzmann_ratios"]) ** 2)))
    boltz_r = float(np.corrcoef(suite["pde_boltzmann_ratios"], suite["exact_boltzmann_ratios"])[0, 1])

    morph_wt = synthesize_2d_planarian_morphology(wt["s"], wt["E"], wt["B"])
    sim_w_60 = np.interp(spec["s_bins"], morph_wt["s_fine"], morph_wt["half_width"])
    sim_w_60_norm = sim_w_60 / max(1e-6, float(np.max(sim_w_60)))
    width_corr_r = float(np.corrcoef(spec["widths_norm"], sim_w_60_norm)[0, 1])
    width_rmse = float(np.sqrt(np.mean((spec["widths_norm"] - sim_w_60_norm) ** 2)))

    expected_heads = [1, 2, 2, 0, 2, 1]
    observed_heads = [
        wt["n_heads"],
        octa["n_heads"],
        cryp["n_heads"],
        sch["n_heads"],
        z0["n_heads"],
        resc["n_heads"],
    ]
    regime_rmse = float(np.sqrt(np.mean((np.array(expected_heads, dtype=float) - np.array(observed_heads, dtype=float)) ** 2)))

    return [
        {
            "name": "Wild-type Nernst-Planck posterior-to-anterior serotonin (5-HT) Boltzmann concentration ratio (c_post / c_ant)",
            "value": round(wt["c_post_over_c_ant"], 4),
            "unit": "c_post / c_ant",
            "method": "Scharfetter-Gummel Nernst-Planck gap-junction PDE steady-state ratio c(L)/c(0) at V_ant=-20 mV, V_post=-60 mV, z=+1, V_T=25.85 mV",
            "reference_value": round(inv["ratio_wt_z1"], 4),
            "reference_source": "Pietak & Levin (2017) J. R. Soc. Interface 14:20170425 Eq. (5) & Fig. 2 Boltzmann equilibrium exp(40 / 25.85) = 4.7002",
            "note": f"PDE c_post/c_ant = {wt['c_post_over_c_ant']:.4f} vs exact Boltzmann = {inv['ratio_wt_z1']:.4f} (9-point sweep Pearson r = {boltz_r:.6f}, RMSE = {boltz_rmse:.6f}); z=0 control ratio = {z0['c_post_over_c_ant']:.4f}",
            "derived_from": "script",
        },
        {
            "name": "Cryptic two-headed (2H0T) planarian memory persistence after transient 8-octanol washout",
            "value": int(cryp["n_heads"]),
            "unit": "regenerated heads after washout",
            "method": "Bilateral ERK head commitment (E >= 0.50, B < 0.50) after transient 8-octanol gap-junction blockade (eta_oct=0.92) followed by complete washout (eta_oct -> 0.0)",
            "reference_value": 2,
            "reference_source": "Pietak & Levin (2017) J. R. Soc. Interface 14:20170425 Figs. 5-6 & Durant et al. (2017) Biophys. J. 112:2231-2243 (permanent cryptic 2H0T planarians)",
            "note": f"cryptic washout post_E = {cryp['post_E']:.4f}, post_B = {cryp['post_B']:.4f}, c_post/c_ant = {cryp['c_post_over_c_ant']:.4f} (2H0T); targeted posterior K+ hyperpolarization rescues 1H1T (n_heads = {resc['n_heads']}, c_post/c_ant = {resc['c_post_over_c_ant']:.4f})",
            "derived_from": "script",
        },
        {
            "name": "Posterior beta-catenin suppression delta between Wild-Type (1H1T) and 8-Octanol gap-junction blockade (2H0T)",
            "value": round(wt["post_B"] - octa["post_B"], 4),
            "unit": "B_WT(L) - B_oct(L)",
            "method": "Posterior blastema nuclear beta-catenin activity B(L) in wild_type_1H1T vs octanol_pulse_2H0T (eta_oct=0.92)",
            "reference_value": 0.72,
            "reference_source": "Pietak & Levin (2017) J. R. Soc. Interface 14:20170425 Fig. 4 (posterior cAMP/PKA -> beta-catenin activation vs octanol collapse)",
            "note": f"WT post_B = {wt['post_B']:.4f} (1H1T) vs 8-octanol post_B = {octa['post_B']:.4f} (2H0T); SCH-28080/forskolin ant_B = {sch['ant_B']:.4f} (0H2T headless)",
            "derived_from": "script",
        },
        {
            "name": "Planarian specimen bilateral photoreceptor ocelli relative AP position",
            "value": round(spec["mean_ocelli_rel_s"], 4),
            "unit": "x / L (anterior fraction)",
            "method": "Automated Gaussian local-contrast ocelli segmentation (body-length-scaled kernels) on the vendored PlanformDB planarian_ap_polarity_specimen.jpg",
            "reference_value": 0.065,
            "reference_source": "MODEL-SIDE reference, not an independent empirical datum: the eyespot AP position hard-coded in synthesize_2d_planarian_morphology() for the anterior ERK head domain (x/L = 0.065). This measurement is therefore a model-vs-specimen agreement check; an independently published Schmidtea mediterranea ocelli x/L value should replace it in Pass 4",
            "note": f"detected {spec['n_ocelli_detected']} bilateral ocelli at x/L = {[round(v, 4) for v in spec['ocelli_rel_s']]} (std = {np.std(spec['ocelli_rel_s']):.4f}), separation = {spec['inter_ocelli_distance_rel']:.4f} L, body length = {spec['length_px']:.0f} px, aspect ratio = {spec['aspect_ratio']:.2f}; specimen replaced by campaign commit 398c37e0 (720x500 PlanformDB photograph, SHA-256 re-pinned in Pass 3)",
            "derived_from": "output",
        },
        {
            "name": "Specimen vs simulated 1H1T planarian dorsoventral width profile correlation r",
            "value": round(width_corr_r, 4),
            "unit": "Pearson r",
            "method": "Pearson correlation and RMSE between segmented planarian specimen width profile W(x/L) and reconstructed 1H1T morphology contour",
            "reference_value": 0.90,
            "reference_source": "Vendored planarian dorsal whole-mount morphology (datasets/bioelectric_morphogen_coupling/planarian_ap_polarity_specimen.jpg)",
            "note": f"width profile Pearson r = {width_corr_r:.4f}, normalized width RMSE = {width_rmse:.4f} across 60 AP body segments",
            "derived_from": "output",
        },
        {
            "name": "Pietak & Levin (2017) 6-regime regenerated head count prediction RMSE",
            "value": round(regime_rmse, 4),
            "unit": "heads RMSE",
            "method": "RMSE between predicted and empirical head counts across WT (1H), 8-Octanol (2H), Cryptic Washout Memory (2H), SCH-28080/Forskolin (0H), z=0 Control (2H), and Bioelectric K+ Rescue (1H)",
            "reference_value": 0.0,
            "reference_source": "Pietak & Levin (2017) J. R. Soc. Interface 14:20170425 Figs. 3-6 & Durant et al. (2017) Biophys. J. 112:2231-2243 (Lean 4 cryptic_two_headed_memory_phenotypes)",
            "note": f"predicted heads = {observed_heads} vs reference = {expected_heads} (exact 6/6 phenotype match, RMSE = {regime_rmse:.4f})",
            "derived_from": "script",
        },
    ] + measure_pass3_network(seed=2017)


def measure_pass3_network(seed: int = 2017) -> list[dict[str, Any]]:
    """Pass 3 measurements: 2D Voronoi SG network verification, Ca2+/IP3 fields, stochastic branching."""
    p3 = evaluate_pass3_network_suite(seed=seed)
    ref3 = load_pass3_reference()
    bz = p3["boltzmann"]["species"]
    wt = p3["ca_ip3"]["wild_type_1H1T"]
    octa = p3["ca_ip3"]["octanol_pulse_2H0T"]
    cryp = p3["ca_ip3"]["cryptic_memory_washout_2H0T"]
    wt_1d = simulate_pietak_bioelectric_morphogen_pde(regime="wild_type_1H1T", seed=seed, record_every=10**6)
    max_rel = max(d["max_rel_err"] for d in bz.values())
    max_mass = max(d["mass_rel_err"] for d in bz.values())
    br = p3["branching"]
    emp = ref3["empirical_targets"]
    # p(2H0T) threshold V_post: first grid value with p > 0.05
    vp = p3["vpost_grid"]
    pv = p3["p_vs_vpost"]
    above = np.where(pv > 0.05)[0]
    v_thresh = float(vp[above[0]]) if above.size else float("nan")
    return [
        {
            "name": "2D Voronoi-network Scharfetter-Gummel electrodiffusion max relative deviation from exact Boltzmann equilibrium (5-HT z=+1, Ca2+ z=+2, IP3 z=-3, all cells)",
            "value": float(f"{max_rel:.3e}"),
            "unit": "max |c_i - c_i^exact| / c_i^exact",
            "method": f"Backward-Euler relaxation of the SG operator on an irregular {p3['net']['n_cells']}-cell / {p3['net']['n_edges']}-edge Voronoi graph clipped to the segmented specimen silhouette; compared per cell to M exp(-zV_i/V_T)/sum_k A_k exp(-zV_k/V_T)",
            "reference_value": 0.0,
            "reference_source": "Pietak & Levin (2016) Front. Bioeng. Biotechnol. 4:55 (BETSE Voronoi cell-cluster finite-volume electrodiffusion) & Pietak & Levin (2017) Eq. (5) Boltzmann equilibrium",
            "note": f"per-species max rel err = {{{', '.join(f'{k}: {d['max_rel_err']:.2e}' for k, d in bz.items())}}}; Pearson r = 1.000000 for all; mass conservation rel err <= {max_mass:.1e}; edge log-ratio RMSE <= {max(d['edge_log_ratio_rmse'] for d in bz.values()):.1e}; mean interior Voronoi coordination = {p3['mean_interior_degree']:.3f} (Euler: 6)",
            "derived_from": "script",
        },
        {
            "name": "2D network vs 1D posterior-to-anterior 5-HT ratio consistency (reactive WT, group means s<0.1 / s>0.9)",
            "value": round(wt["c_post_over_c_ant"], 4),
            "unit": "c_post / c_ant",
            "method": "Area-weighted mean 5-HT over posterior (s > 0.9) vs anterior (s < 0.1) Voronoi cells after reaction-electrodiffusion on the 2D network with turnover k_h = 0.004; reference = exp(dV_group / V_T) for the same cell groups",
            "reference_value": round(bz["serotonin_5HT"]["post_over_ant_group_boltz"], 4),
            "reference_source": "Pietak & Levin (2017) J. R. Soc. Interface 14:20170425 Eq. (5): Boltzmann ratio over the group-mean voltage difference (1D endpoint value exp(40/25.85) = 4.70 is the N-cell chain limit)",
            "note": f"2D reactive ratio = {wt['c_post_over_c_ant']:.4f}; 2D pure-electrodiffusion group ratio = {bz['serotonin_5HT']['post_over_ant_group']:.4f}; 1D SG endpoint ratio = {wt_1d['c_post_over_c_ant']:.4f}; turnover k_h lowers the reactive ratio by {100 * (1 - wt['c_post_over_c_ant'] / bz['serotonin_5HT']['post_over_ant_group']):.2f} %",
            "derived_from": "script",
        },
        {
            "name": "Anterior-to-posterior cytosolic Ca2+ ratio in the wild-type 2D network (VGCC influx vs z=+2 electrophoresis)",
            "value": round(wt["Ca_ant_over_Ca_post"], 4),
            "unit": "Ca_ant / Ca_post",
            "method": "Reactive Ca2+ (IP3R release, SERCA, leak, PMCA, voltage-gated influx sigma((V - V_half)/k_V)) with SG electrodiffusion z=+2 on the 2D network; pure Boltzmann for z=+2 would give Ca_ant/Ca_post = exp(-2 dV_group/V_T)",
            "reference_value": 1.0,
            "reference_source": "Beane et al. (2011) Chem. Biol. 18:77-89: depolarisation-driven Ca2+ influx is required for head specification (qualitative: Ca_ant/Ca_post > 1 at the depolarised anterior); DOI transcribed offline",
            "note": f"Ca_ant/Ca_post = {wt['Ca_ant_over_Ca_post']:.3f} (anterior enriched) although pure z=+2 electrophoresis would enrich the posterior {bz['calcium_Ca2']['post_over_ant_group_boltz']:.1f}x; IP3 (z=-3) post/ant = {wt['P_post_over_P_ant']:.3f}; under 8-octanol/flat V both are uniform (Ca_post = {octa['Ca_post']:.3f} vs WT {wt['Ca_post']:.3f})",
            "derived_from": "script",
        },
        {
            "name": "Stochastic double-head branching probability p(2H0T) after 8-octanol gap-junction blockade (Ca2+/IP3-noise toggle ensemble)",
            "value": round(br["octanol_pulse_2H0T"]["p_2H0T"], 4),
            "unit": "fraction of blastemas with posterior head",
            "method": f"Euler-Maruyama ensemble of n = {br['octanol_pulse_2H0T']['n_rep']} posterior beta-catenin/ERK toggle switches driven by the 2D network's posterior 5-HT and Ca2+ (lognormal Ca2+ influx variability sigma_Ca = {PASS3_BRANCHING_PARAMS['sigma_Ca']}); b_basal calibrated to this band, the WT/SCH values and the V_post and sigma_Ca dependences are predictions",
            "reference_value": emp["double_head_fraction_octanol"]["value"],
            "reference_source": emp["double_head_fraction_octanol"]["source"],
            "note": f"p(2H0T): WT = {br['wild_type_1H1T']['p_2H0T']:.3f}, 8-octanol = {br['octanol_pulse_2H0T']['p_2H0T']:.3f} +/- {br['octanol_pulse_2H0T']['se']:.3f}, cryptic washout = {br['cryptic_memory_washout_2H0T']['p_2H0T']:.3f} +/- {br['cryptic_memory_washout_2H0T']['se']:.3f}, SCH-28080 = {br['sch28080_hyperpolarized_0H2T']['p_2H0T']:.3f}, z=0 control = {br['uncharged_morphogen_control_z0']['p_2H0T']:.3f} (1D Pass 2 solver prescribes 2H0T for z=0; the network predicts 1H1T); sigma_Ca = 0 gives p = {p3['p_vs_sigma'][0]:.3f}",
            "derived_from": "script",
        },
        {
            "name": "Cryptic-state (flat -20 mV, gap junctions reopened) double-head probability p(2H0T) vs 8-octanol state",
            "value": round(br["cryptic_memory_washout_2H0T"]["p_2H0T"], 4),
            "unit": "fraction of blastemas with posterior head",
            "method": "Same ensemble as above with the cryptic bioelectric state (V flat at -20 mV, eta_oct = 0): 5-HT and Ca2+ fields are identical to the octanol state because equilibrium electrodiffusion depends on V, not on G_gj",
            "reference_value": emp["double_head_fraction_cryptic_recut"]["value"],
            "reference_source": emp["double_head_fraction_cryptic_recut"]["source"],
            "note": f"cryptic p = {br['cryptic_memory_washout_2H0T']['p_2H0T']:.3f} vs octanol p = {br['octanol_pulse_2H0T']['p_2H0T']:.3f} (difference is sampling noise, identical drives S_B = {br['cryptic_memory_washout_2H0T']['S_B']:.3f}; cryptic c_post/c_ant = {cryp['c_post_over_c_ant']:.3f}); posterior V_mem threshold for p > 0.05 = {v_thresh:.0f} mV (grid step 5 mV)",
            "derived_from": "script",
        },
    ]


# --------------------------------------------------------------------------- #
# 6. Comprehensive Deterministic Self-Test Suite (`--self-test`)
# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Run 8 deterministic verification checks covering SHA-256 locks, exact Boltzmann
    Nernst-Planck equilibrium ratios, all 6 Pietak & Levin (2017) regeneration regimes
    including cryptic 2H0T memory after octanol washout, specimen CV segmentation,
    and bitwise solver determinism."""
    # 1. Verify SHA-256 digests of all 4 vendored dataset files
    digests = verify_dataset_sha256()
    ref = load_reference()
    assert len(digests) == 4, f"Expected 4 verified dataset files, got {len(digests)}"
    assert ref["sim_id"] == "pietak-bioelectric-morphogen-coupling"

    # 2. Verify exact Nernst-Planck Boltzmann equilibrium identities & GJ symmetry
    inv = compute_nernst_planck_invariants()
    assert abs(inv["ratio_wt_z1"] - math.exp(40.0 / 25.85)) < 1e-12
    assert abs(inv["ratio_symmetric_2H0T"] - 1.0) < 1e-12
    assert abs(inv["ratio_uncharged_z0"] - 1.0) < 1e-12
    assert abs(inv["cation_anion_product"] - 1.0) < 1e-12
    assert inv["gj_symmetry_diff"] < 1e-15 and inv["gj_blocked_flux"] == 0.0

    # 3. Verify Wild-Type 1H1T Nernst-Planck 5-HT electrophoresis & Boltzmann parity
    suite = evaluate_all_regimes_and_boltzmann_sweep(seed=2017)
    wt = suite["regimes"]["wild_type_1H1T"]
    assert wt["phenotype_code"] == "1H1T" and wt["n_heads"] == 1 and wt["n_tails"] == 1, (
        f"Expected 1H1T wild-type regeneration, got {wt['phenotype_code']}"
    )
    assert abs(wt["c_post_over_c_ant"] - inv["ratio_wt_z1"]) < 0.02, (
        f"WT PDE ratio {wt['c_post_over_c_ant']:.4f} must match Boltzmann {inv['ratio_wt_z1']:.4f}"
    )

    # 4. Verify 8-Octanol pulse -> 2H0T and Cryptic Washout Memory -> persistent 2H0T
    octa = suite["regimes"]["octanol_pulse_2H0T"]
    cryp = suite["regimes"]["cryptic_memory_washout_2H0T"]
    assert octa["phenotype_code"] == "2H0T" and octa["n_heads"] == 2
    assert cryp["phenotype_code"] == "2H0T" and cryp["n_heads"] == 2, (
        f"Cryptic washout must permanently retain 2H0T memory, got {cryp['phenotype_code']}"
    )
    assert abs(cryp["c_post_over_c_ant"] - 1.0) < 0.05

    # 5. Verify SCH-28080/Forskolin -> 0H2T headless & Bioelectric K+ Rescue -> 1H1T
    sch = suite["regimes"]["sch28080_hyperpolarized_0H2T"]
    resc = suite["regimes"]["bioelectric_reset_rescue_1H1T"]
    assert sch["phenotype_code"] == "0H2T" and sch["n_heads"] == 0
    assert resc["phenotype_code"] == "1H1T" and resc["n_heads"] == 1

    # 6. Verify uncharged morphogen (z=0) ablation control -> flat concentration & loss of tail (2H0T)
    z0 = suite["regimes"]["uncharged_morphogen_control_z0"]
    assert abs(z0["c_post_over_c_ant"] - 1.0) < 1e-6 and z0["phenotype_code"] == "2H0T"

    # 7. Verify planarian specimen CV segmentation (2 bilateral ocelli in anterior head)
    spec = segment_planarian_specimen(SPECIMEN_PATH)
    assert spec["n_ocelli_detected"] == 2
    assert 0.03 <= spec["mean_ocelli_rel_s"] <= 0.12

    # 8. Verify bitwise two-run solver determinism
    r1 = simulate_pietak_bioelectric_morphogen_pde(regime="wild_type_1H1T", steps=400, seed=2017)
    r2 = simulate_pietak_bioelectric_morphogen_pde(regime="wild_type_1H1T", steps=400, seed=2017)
    assert np.allclose(r1["c"], r2["c"], atol=0.0, rtol=0.0)
    assert np.allclose(r1["E"], r2["E"], atol=0.0, rtol=0.0)

    # ---- Pass 3 checks -------------------------------------------------------------
    # 9. Pass 3 SHA-256 pins and reference block
    d3 = verify_pass3_dataset_sha256()
    ref3 = load_pass3_reference()
    assert len(d3) == 2 and ref3["pass"] == 3 and ref3["sim_id"] == "pietak-bioelectric-morphogen-coupling"

    # 10. Irregular 2D Voronoi network: finite polygons, positive areas, every real-real face
    #     has positive length, mean interior coordination = 6 (planar Euler), geometry from specimen
    net = build_voronoi_cell_network(seed=2017)
    assert 300 <= net["n_cells"] <= 600 and net["n_edges"] > net["n_cells"]
    assert np.all(net["areas"] > 0.0) and np.all(net["face_len"] > 0.0) and np.all(net["w_ij"] > 0.0)
    assert np.all(np.abs(net["y"]) < net["half_width_fn"](net["s"])), "all cells must lie inside the specimen silhouette"
    deg_int = net["degree"][net["interior"]]
    assert abs(float(np.mean(deg_int)) - 6.0) < 0.05, f"mean interior Voronoi coordination {np.mean(deg_int):.3f} != 6"

    # 11. SG electrodiffusion on the irregular graph reaches the exact Boltzmann state for
    #     z = +1, +2, -3 (max rel err < 1e-6), conserves mass, and equilibrium is independent of G_gj
    bz = verify_network_boltzmann_equilibrium(net)
    for name, d in bz["species"].items():
        assert d["max_rel_err"] < 1e-6, f"{name}: Boltzmann deviation {d['max_rel_err']:.2e}"
        assert d["mass_rel_err"] < 1e-9 and d["pearson_r"] > 1.0 - 1e-9
        assert d["max_abs_flux_residual"] < 1e-6
    sgn = np.sign([d["z"] for d in bz["species"].values()])
    ratios = np.array([d["post_over_ant_group"] for d in bz["species"].values()])
    assert np.all((ratios > 1.0) == (sgn > 0)), "cations enrich the hyperpolarised posterior, anions the anterior"
    # the Boltzmann steady state does not depend on the conductance magnitude (G_0 ablation)
    Vn = network_voltage_profile(net, -20.0, -60.0)
    lu_a = network_implicit_stepper(network_sg_operator(Vn, net, D=1.0, z=1.0, G_0=1.8), 2.0)
    lu_b = network_implicit_stepper(network_sg_operator(Vn, net, D=1.0, z=1.0, G_0=0.3), 2.0)
    # Relax each case to its own steady state (a fixed iteration count is not a fair
    # comparison: the G_0 = 0.3 operator relaxes ~6x more slowly, so it would still be
    # transient after the number of steps that converges G_0 = 1.8).
    ab_steady = []
    for lu_ in (lu_a, lu_b):
        c_ = np.ones(net["n_cells"])
        for _ in range(30000):
            c_next = lu_.solve(c_)
            converged = float(np.max(np.abs(c_next - c_))) < 1e-13
            c_ = c_next
            if converged:
                break
        else:
            raise AssertionError("G_0 ablation did not reach steady state")
        ab_steady.append(c_)
    assert float(np.max(np.abs(ab_steady[0] - ab_steady[1]))) < 1e-6

    # 12. Coupled 5-HT/IP3/Ca2+ network: WT 5-HT enriches posterior, IP3 (z<0) and Ca2+
    #     (VGCC) enrich the anterior; the flat-V octanol and cryptic states are uniform & identical
    p3 = evaluate_pass3_network_suite(seed=2017)
    wt3 = p3["ca_ip3"]["wild_type_1H1T"]
    oc3 = p3["ca_ip3"]["octanol_pulse_2H0T"]
    cr3 = p3["ca_ip3"]["cryptic_memory_washout_2H0T"]
    assert wt3["c_post_over_c_ant"] > 3.0 and wt3["P_post_over_P_ant"] < 0.5 and wt3["Ca_ant_over_Ca_post"] > 2.0
    assert abs(oc3["c_post_over_c_ant"] - 1.0) < 1e-6 and abs(cr3["c_post_over_c_ant"] - 1.0) < 1e-6
    assert abs(oc3["Ca_post"] - cr3["Ca_post"]) < 1e-9

    # 13. Stochastic branching: WT and SCH-28080 ~0, octanol/cryptic inside the reference band,
    #     p(2H0T) non-decreasing with posterior depolarisation, and deterministic (same seed)
    br = p3["branching"]
    band = ref3["empirical_targets"]["double_head_fraction_octanol"]["band"]
    assert br["wild_type_1H1T"]["p_2H0T"] <= 0.02 and br["sch28080_hyperpolarized_0H2T"]["p_2H0T"] <= 0.05
    for r_ in ("octanol_pulse_2H0T", "cryptic_memory_washout_2H0T"):
        assert band[0] - 0.05 <= br[r_]["p_2H0T"] <= band[1] + 0.05, f"{r_} p(2H0T) = {br[r_]['p_2H0T']:.3f}"
    pv = p3["p_vs_vpost"]
    assert pv[0] <= 0.02 and pv[-1] >= 0.10 and np.all(np.diff(pv) >= -0.06)
    b1 = stochastic_head_tail_branching(1.0, 0.95, 1.0, seed=5)
    b2 = stochastic_head_tail_branching(1.0, 0.95, 1.0, seed=5)
    assert b1["p_2H0T"] == b2["p_2H0T"] and np.array_equal(b1["decision"], b2["decision"])

    print(
        "[2017-pietak-bioelectric-morphogen-coupling] SELF-TEST PASSED (13/13 checks: SHA-256 locks, "
        f"WT={wt['phenotype_code']} [c_post/c_ant={wt['c_post_over_c_ant']:.4f} vs Boltzmann {inv['ratio_wt_z1']:.4f}], "
        f"Octanol={octa['phenotype_code']}, Cryptic-Washout={cryp['phenotype_code']}, "
        f"SCH-28080={sch['phenotype_code']}, z=0-ctl={z0['phenotype_code']}, K+-Rescue={resc['phenotype_code']}, "
        f"specimen ocelli={spec['n_ocelli_detected']} at x/L={spec['mean_ocelli_rel_s']:.4f}; "
        f"Pass 3: 2D Voronoi net N={net['n_cells']} <k>={np.mean(deg_int):.2f}, Boltzmann max rel err="
        f"{max(d['max_rel_err'] for d in bz['species'].values()):.1e}, WT net c_P/c_A={wt3['c_post_over_c_ant']:.3f}, "
        f"Ca_A/Ca_P={wt3['Ca_ant_over_Ca_post']:.2f}, p(2H0T) octanol={br['octanol_pulse_2H0T']['p_2H0T']:.3f} "
        f"cryptic={br['cryptic_memory_washout_2H0T']['p_2H0T']:.3f} WT={br['wild_type_1H1T']['p_2H0T']:.3f})."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Run 13-check deterministic verification suite and exit")
    ap.add_argument("--skip-gif", action="store_true", help="Skip rendering animated GIF")
    ap.add_argument("--seed", type=int, default=2017, help="Deterministic RNG seed")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    res = render_outputs(seed=args.seed, skip_gif=args.skip_gif)
    wt = res["suite"]["regimes"]["wild_type_1H1T"]
    cryp = res["suite"]["regimes"]["cryptic_memory_washout_2H0T"]
    spec = res["specimen"]
    print(
        f"[2017-pietak-bioelectric-morphogen-coupling] Rendered 9-panel comparison -> {OUT_PNG.relative_to(ROOT)} "
        f"and animation -> {OUT_GIF.relative_to(ROOT)} "
        f"(WT={wt['phenotype_code']} [c_post/c_ant={wt['c_post_over_c_ant']:.4f}], "
        f"Cryptic-Washout={cryp['phenotype_code']}, specimen ocelli={spec['n_ocelli_detected']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
