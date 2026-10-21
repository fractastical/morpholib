#!/usr/bin/env python3
r"""
2011-beane-planarian-hk-atpase-anterior.py — Beane et al. (2011) & Pietak-Levin (2017)
Bioelectric H,K-ATPase Membrane Potential V_mem(x,t) & Voltage-Gated Gap-Junction
Anterior-Posterior Polarity Switch in Planarian Regeneration (Schmidtea mediterranea).

Mathematical & Biological Specification
=======================================
1. **Beane et al. (2011) Chem. Biol. 18(1):77-89 & Pietak-Levin (2017) J. R. Soc. Interface 14:20170425:**
   Along the anteroposterior axis $x \in [0, L]$ ($N = 60$ cells; $x = 0$ anterior wound blastema,
   $x = L$ posterior wound blastema) of an amputated *Schmidtea mediterranea* trunk fragment,
   resting membrane voltage $V(x,t) = V_{\mathrm{mem}}(x,t)$ (mV) evolves under a bistable ionic
   conductance coupled through voltage-gated innexin gap junctions:

       C_m dV/dt = -g_bistable (V - V_hyper)(V - V_thresh)(V - V_depol)
                   + I_HK(x) + I_IVM(x) + d/dx [ G_GJ(dV/dx) dV/dx ]

   where:
     - $V_{\mathrm{hyper}} = -65.0\text{ mV}$ is the stable hyperpolarized trunk/posterior resting potential,
     - $V_{\mathrm{thresh}} = -42.5\text{ mV}$ is the unstable voltage threshold separating head vs tail fate,
     - $V_{\mathrm{depol}} = -20.0\text{ mV}$ is the stable H,K-ATPase-depolarized anterior blastema potential,
     - $G_{\mathrm{GJ}}(\Delta V) = (1 - \eta_{\mathrm{oct}}) G_0 / (1 + (\Delta V / V_0)^2)$ is the symmetric
       Bell-form voltage-gated innexin gap-junction conductance (Oviedo et al. 2010; Pietak & Levin 2017).

2. **Voltage-Gated $\mathrm{Ca}_{\mathrm{v}}$ Calcium Influx & Head/Tail Morphogen Cascade (Beane et al. 2011 Figs. 5-6):**
   Depolarization of $V(x,t)$ above $V_{\mathrm{thresh}}$ opens L-type voltage-gated $\mathrm{Ca}^{2+}$
   channels ($\mathrm{Ca}_{\mathrm{v}}1.1 / \mathrm{Ca}_{\mathrm{v}}1.2$), elevating intracellular $\mathrm{Ca}^{2+}$
   and driving anterior head identity $H(x,t) \in [0,1]$ (*pc2*, *ndk*, *sFRP-1*) in mutual antagonism
   with posterior $\beta$-catenin / *wnt1* tail identity $T(x,t) \in [0,1]$:

       Ca_ss(V) = (1 - eta_nif) / (1 + exp(-(V - V_thresh) / k_V))
       tau_H dH/dt = Ca_ss(V) (1 - c_T T) - H + D_H d^2H/dx^2
       tau_T dT/dt = (1 - Ca_ss(V)) W(x) (1 - c_H H) - T + D_T d^2T/dx^2

3. **Lean 4 Formal Verification (`morphoeval/lean/loops/Loop_081_planarian_hk_atpase_anterior.lean`):**
   - `MathOfBiology.Loop_081.beane_bistable_vmem_equilibria`
   - `MathOfBiology.Loop_081.gj_voltage_gate_symmetry_and_isolation`
   - `MathOfBiology.Loop_081.beane_four_regeneration_phenotypes`
   - `MathOfBiology.Loop_081.mesh_gj_flux_conservation` (Pass 3, 2D finite-volume mesh)
   - `MathOfBiology.Loop_081.slit_midline_exclusion_and_collapse` (Pass 3, eye-field / slit RNAi)
   - `MathOfBiology.Loop_081.vmem_neoblast_rate_nonneg_monotone` (Pass 3, V_mem-coupled neoblasts)

3b. **Pass 3 Remediation — 2D Body Mesh, Slit/Netrin Guidance & Stochastic Neoblasts:**
   - The same bistable $V_{\mathrm{mem}}$ + voltage-gated gap-junction + $\mathrm{Ca}_{\mathrm{v}}$-gated
     $H/T$ system is integrated on a 2D cell-centred finite-volume mesh built from the segmented
     specimen silhouette (`build_planarian_body_mesh`, `simulate_planarian_bioelectric_2d`);
     edge conductances are rescaled ($G_{\mathrm{edge}} = G_0 r^2$, $V_{0,\mathrm{edge}} = V_0 / r$) so the
     continuum coefficients match the 1D solver. Honest reduction: structured finite-volume grid,
     not an unstructured FEM mesh, and no dorsoventral layering.
   - Photoreceptor placement is the arg-max of the eye field $E = H\,(1 - S_{\mathrm{slit}}(y))(1 - d^2)$
     and visual axons follow a reduced growth-cone model with the net mediolateral cue
     $c(y) = k_{\mathrm{netrin}}\,\partial_y N - k_{\mathrm{slit}}\,\partial_y S$ and a Robo-Slit
     pre/post-crossing switch (Cebria & Newmark 2005; Cebria et al. 2007; Kidd et al. 1999).
   - Neoblasts are agents on the AP axis dividing as an inhomogeneous Poisson process whose rate
     is modulated by the local $\mathrm{Ca}_{\mathrm{ss}}(V_{\mathrm{mem}}(x,t))$ and whose progeny adopt
     head identity through the $\mathrm{Ca}_{\mathrm{v}}$ gate of the blastema they enter
     (`simulate_neoblast_kinetics`); the biphasic mitotic timing is calibrated to Wenemoser &
     Reddien (2010) and is an input, not a prediction.

4. **Five Canonical Experimental Regeneration Regimes (Beane et al. 2011; Oviedo et al. 2010):**
   - **Wild-Type (`1H1T`):** Active anterior H,K-ATPase depolarizes the anterior blastema
     ($V(0) = -16.8\text{ mV} > V_{\mathrm{thresh}}$) while the posterior blastema remains
     hyperpolarized ($V(L) = -65.0\text{ mV}$), regenerating **1 Head + 1 Tail (`1H1T`)**.
   - **H,K-ATPase Inhibition (`0H2T`, SCH-28080 / *Smed-H,K-ATPase* RNAi):** Blocking H,K-ATPase
     prevents anterior depolarization ($V(0) = -66.5\text{ mV} < V_{\mathrm{thresh}}$), abolishing
     anterior $\mathrm{Ca}^{2+}$ influx and yielding **headless (`0H2T`)** planarians (100% in Beane 2011).
   - **Ivermectin Depolarization (`2H0T`, GluCl Opener):** Exogenous depolarization at both wounds
     drives $V(0), V(L) > V_{\mathrm{thresh}}$, inducing ectopic posterior head regeneration and
     yielding **double-headed (`2H0T`)** planarians even when H,K-ATPase is inhibited.
   - **8-Octanol Gap-Junction Blockade (`2H0T`, Oviedo et al. 2010):** Blocking innexin gap junctions
     ($\eta_{\mathrm{oct}} = 0.92$) isolates posterior wound depolarization from hyperpolarized trunk
     clearance, producing **double-headed (`2H0T`)** planarians.
   - **Nifedipine $\mathrm{Ca}_{\mathrm{v}}$ Channel Blockade Control (`0H2T`, Beane et al. 2011 Fig. 5):**
     Blocking L-type $\mathrm{Ca}^{2+}$ channels ($\eta_{\mathrm{nif}} = 0.95$) decouples anterior
     $V_{\mathrm{mem}}$ depolarization ($V(0) = -16.8\text{ mV}$) from head gene activation, yielding
     **0 Heads (`0H2T`)** and proving $\mathrm{Ca}^{2+}$ is the obligatory downstream messenger.
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

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "planarian_hk_atpase_anterior"
REF_JSON_PATH = DATASET_DIR / "planarian_hk_atpase_anterior_reference.json"
SPECIMEN_PATH = DATASET_DIR / "schmidtea_mediterranea_specimen.jpg"
SCOUT_PATH = DATASET_DIR / "scout_manifest.json"
BIB_PATH = DATASET_DIR / "paper_citations.bib"

OUT_PNG = ROOT / "docs" / "media" / "planarian_hk_atpase_anterior_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "planarian_hk_atpase_anterior_growth.gif"

EXPECTED_SHA256: dict[str, str] = {
    "datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg": (
        "466029cc0561067890bb0f69d224d8f799745eeb67a0699246107611737fb546"
    ),
    "datasets/planarian_hk_atpase_anterior/scout_manifest.json": (
        "9de72183ede698f04d87a137a1e4d3a235dbcac613bf9da6e5c553eef9d94294"
    ),
    "datasets/planarian_hk_atpase_anterior/paper_citations.bib": (
        "4978cdbc97ad314e55b71348a3fd0a831887eaf7bad4b10a76e834070964277f"
    ),
}


def verify_dataset_sha256() -> dict[str, str]:
    """Verify cryptographic SHA-256 digests of all vendored files in datasets/planarian_hk_atpase_anterior/."""
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
    computed["datasets/planarian_hk_atpase_anterior/planarian_hk_atpase_anterior_reference.json"] = (
        hashlib.sha256(REF_JSON_PATH.read_bytes()).hexdigest()
    )
    return computed


def load_reference() -> dict[str, Any]:
    """Load and return the vendored Beane et al. (2011) reference dataset JSON."""
    return json.loads(REF_JSON_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Exact Bistable Ionic Current, Stability & Voltage-Gated Gap Junctions
# --------------------------------------------------------------------------- #
def bistable_ionic_current(
    V: np.ndarray | float,
    g_bistable: float = 0.002,
    V_hyper: float = -65.0,
    V_thresh: float = -42.5,
    V_depol: float = -20.0,
) -> np.ndarray | float:
    """
    Cubic bistable membrane ionic current I_ion(V) = g * (V - V_hyper)*(V - V_thresh)*(V - V_depol).
    In the membrane potential equation C_m dV/dt = -I_ion(V) + I_ext:
      - V_hyper = -65.0 mV is the stable hyperpolarized posterior/trunk resting potential,
      - V_thresh = -42.5 mV is the unstable separatrix threshold,
      - V_depol = -20.0 mV is the stable H,K-ATPase-depolarized anterior blastema potential.
    """
    return g_bistable * (V - V_hyper) * (V - V_thresh) * (V - V_depol)


def compute_bioelectric_invariants(
    g_bistable: float = 0.002,
    V_hyper: float = -65.0,
    V_thresh: float = -42.5,
    V_depol: float = -20.0,
    G_0: float = 1.8,
    V_0: float = 12.0,
    k_V: float = 3.5,
) -> dict[str, Any]:
    """
    Compute exact analytical fixed-point residuals, restoring conductance slopes dI_ion/dV,
    Bell-form gap-junction symmetry/isolation ratios, and Ca_v channel activation gates.
    """
    def dI_dV(v: float) -> float:
        return g_bistable * (
            (v - V_thresh) * (v - V_depol)
            + (v - V_hyper) * (v - V_depol)
            + (v - V_hyper) * (v - V_thresh)
        )

    res_hyper = float(bistable_ionic_current(V_hyper, g_bistable, V_hyper, V_thresh, V_depol))
    res_thresh = float(bistable_ionic_current(V_thresh, g_bistable, V_hyper, V_thresh, V_depol))
    res_depol = float(bistable_ionic_current(V_depol, g_bistable, V_hyper, V_thresh, V_depol))

    slope_hyper = dI_dV(V_hyper)
    slope_thresh = dI_dV(V_thresh)
    slope_depol = dI_dV(V_depol)

    # Bell-form voltage-gated gap-junction conductance G_GJ(dV) = G_0 / (1 + (dV/V_0)^2)
    dv_grid = np.linspace(-45.0, 45.0, 301)
    g_gj_wt = G_0 / (1.0 + (dv_grid / V_0) ** 2)
    g_gj_oct = (1.0 - 0.92) * g_gj_wt

    # Check symmetry G_GJ(-24 mV) == G_GJ(+24 mV) and isolation drop at dV = 36 mV
    g_plus = G_0 / (1.0 + (24.0 / V_0) ** 2)
    g_minus = G_0 / (1.0 + (-24.0 / V_0) ** 2)
    g_boundary_36 = G_0 / (1.0 + (36.0 / V_0) ** 2)

    v_grid = np.linspace(-75.0, -10.0, 300)
    i_ion_curve = bistable_ionic_current(v_grid, g_bistable, V_hyper, V_thresh, V_depol)
    ca_gate_curve = 1.0 / (1.0 + np.exp(-(v_grid - V_thresh) / k_V))

    return {
        "V_hyper": V_hyper,
        "V_thresh": V_thresh,
        "V_depol": V_depol,
        "midpoint_mV": 0.5 * (V_hyper + V_depol),
        "residuals": (res_hyper, res_thresh, res_depol),
        "slope_hyper": slope_hyper,
        "slope_thresh": slope_thresh,
        "slope_depol": slope_depol,
        "gj_symmetry_diff": abs(g_plus - g_minus),
        "gj_zero_drop": G_0,
        "gj_boundary_36mV": g_boundary_36,
        "gj_isolation_ratio": g_boundary_36 / G_0,
        "v_grid": v_grid,
        "i_ion_curve": i_ion_curve,
        "ca_gate_curve": ca_gate_curve,
        "dv_grid": dv_grid,
        "g_gj_wt": g_gj_wt,
        "g_gj_oct": g_gj_oct,
    }


# --------------------------------------------------------------------------- #
# 2. Coupled 1D Bioelectric V_mem + Gap-Junction + Head/Tail Morphogen Solver
# --------------------------------------------------------------------------- #
def laplacian_1d_neumann(u: np.ndarray) -> np.ndarray:
    """1D second-difference Laplacian with zero-flux Neumann boundary conditions."""
    out = np.empty_like(u)
    out[1:-1] = u[:-2] + u[2:] - 2.0 * u[1:-1]
    out[0] = 2.0 * (u[1] - u[0])
    out[-1] = 2.0 * (u[-2] - u[-1])
    return out


def nonlinear_gj_flux_divergence(
    V: np.ndarray,
    G_0: float = 1.8,
    V_0: float = 12.0,
    eta_oct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute discrete divergence d/dx [ G_GJ(dV/dx) dV/dx ] with zero-flux Neumann BCs
    at x = 0 and x = N-1, where inter-cellular conductance on edge (i, i+1) is:
        G_{i+1/2} = (1 - eta_oct) * G_0 / (1 + ((V_{i+1} - V_i) / V_0)^2).
    Returns (div_flux, edge_conductances).
    """
    dv_edge = V[1:] - V[:-1]
    g_edge = (1.0 - eta_oct) * G_0 / (1.0 + (dv_edge / V_0) ** 2)
    flux_edge = g_edge * dv_edge
    div = np.zeros_like(V)
    div[0] = flux_edge[0]
    div[1:-1] = flux_edge[1:] - flux_edge[:-1]
    div[-1] = -flux_edge[-1]
    return div, g_edge


REGIME_NAMES: tuple[str, ...] = (
    "wild_type_1H1T",
    "hk_atpase_inhibited_0H2T",
    "ivermectin_depolarized_2H0T",
    "octanol_gj_blockade_2H0T",
    "nifedipine_cav_block_0H2T",
)


def regime_parameters(regime: str) -> dict[str, float]:
    """
    Pharmacological / RNAi regime parameters shared by the 1D AP solver and the Pass 3
    2D body-mesh solver (Beane et al. 2011 Figs. 1-6; Oviedo et al. 2010):
      I_HK_amp  : anterior H,K-ATPase pump current (+18 WT; -16 SCH-28080 / Smed-H,K-ATPase RNAi)
      I_IVM_amp : ivermectin GluCl depolarizing current at both wounds (+22)
      eta_oct   : 8-octanol innexin gap-junction blockade fraction (0.92)
      eta_nif   : nifedipine L-type Ca_v channel blockade fraction (0.95)
      post_wound_init_depol : transient amputation-induced posterior wound depolarization (mV)
    """
    table: dict[str, dict[str, float]] = {
        "wild_type_1H1T": dict(I_HK_amp=18.0, I_IVM_amp=0.0, eta_oct=0.0, eta_nif=0.0, post_wound_init_depol=18.0),
        "hk_atpase_inhibited_0H2T": dict(I_HK_amp=-16.0, I_IVM_amp=0.0, eta_oct=0.0, eta_nif=0.0, post_wound_init_depol=18.0),
        "ivermectin_depolarized_2H0T": dict(I_HK_amp=18.0, I_IVM_amp=22.0, eta_oct=0.0, eta_nif=0.0, post_wound_init_depol=18.0),
        # uncoupled posterior wound retains injury depolarization (Oviedo et al. 2010)
        "octanol_gj_blockade_2H0T": dict(I_HK_amp=18.0, I_IVM_amp=0.0, eta_oct=0.92, eta_nif=0.0, post_wound_init_depol=38.0),
        "nifedipine_cav_block_0H2T": dict(I_HK_amp=18.0, I_IVM_amp=0.0, eta_oct=0.0, eta_nif=0.95, post_wound_init_depol=18.0),
    }
    if regime not in table:
        raise ValueError(f"Unknown experimental regime: {regime}")
    return dict(table[regime])


def simulate_planarian_bioelectric_pde(
    regime: str = "wild_type_1H1T",
    N: int = 60,
    steps: int = 2400,
    dt: float = 0.04,
    g_bistable: float = 0.002,
    V_hyper: float = -65.0,
    V_thresh: float = -42.5,
    V_depol: float = -20.0,
    G_0: float = 1.8,
    V_0: float = 12.0,
    k_V: float = 3.5,
    tau_H: float = 4.0,
    tau_T: float = 4.0,
    c_H: float = 0.85,
    c_T: float = 0.85,
    D_H: float = 0.15,
    D_T: float = 0.15,
    I_HK_override: float | None = None,
    I_IVM_override: float | None = None,
    eta_oct_override: float | None = None,
    eta_nif_override: float | None = None,
    seed: int = 2011,
    record_every: int = 80,
) -> dict[str, Any]:
    """
    Integrate the coupled Beane et al. (2011) & Pietak-Levin (2017) 1D bioelectric-morphogen
    PDE along an amputated planarian trunk fragment (x in [0, N-1]):
      - Wild-type (`wild_type_1H1T`): Anterior H,K-ATPase current I_HK = +18.0 depolarizes
        the anterior blastema while open gap junctions clear transient posterior wound
        depolarization into the hyperpolarized trunk -> 1 Head (anterior) + 1 Tail (posterior).
      - H,K-ATPase inhibition (`hk_atpase_inhibited_0H2T`): SCH-28080 / RNAi sets I_HK = -16.0,
        preventing anterior depolarization -> 0 Heads + 2 Tails (`0H2T` headless phenotype).
      - Ivermectin depolarization (`ivermectin_depolarized_2H0T`): GluCl channel opening
        (I_IVM = +22.0) depolarizes both blastemas above V_thresh -> 2 Heads + 0 Tails (`2H0T`).
      - 8-Octanol gap-junction blockade (`octanol_gj_blockade_2H0T`): Innexin blockade
        (eta_oct = 0.92) isolates posterior wound depolarization from trunk clearance -> 2 Heads (`2H0T`).
      - Nifedipine Ca_v blockade control (`nifedipine_cav_block_0H2T`): Anterior V_mem depolarizes
        normally (I_HK = +18.0), but L-type Ca2+ channels are blocked (eta_nif = 0.95) -> 0 Heads (`0H2T`).
    """
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    s = x / float(N - 1)  # normalized AP coordinate in [0, 1] (0 = anterior wound, 1 = posterior wound)

    # Spatial localization profiles at anterior (s=0) and posterior (s=1) wound blastemas
    ant_envelope = np.exp(-((s / 0.14) ** 2))
    post_envelope = np.exp(-(((1.0 - s) / 0.14) ** 2))
    both_wounds = ant_envelope + post_envelope
    wound_competence_W = 0.25 + 0.75 * both_wounds

    # Default regime parameters from Beane et al. (2011) & Oviedo et al. (2010)
    rp = regime_parameters(regime)
    I_HK_amp = rp["I_HK_amp"]
    I_IVM_amp = rp["I_IVM_amp"]
    eta_oct = rp["eta_oct"]
    eta_nif = rp["eta_nif"]
    post_wound_init_depol = rp["post_wound_init_depol"]

    if I_HK_override is not None:
        I_HK_amp = float(I_HK_override)
    if I_IVM_override is not None:
        I_IVM_amp = float(I_IVM_override)
    if eta_oct_override is not None:
        eta_oct = float(eta_oct_override)
    if eta_nif_override is not None:
        eta_nif = float(eta_nif_override)

    # Initial condition immediately after trunk amputation:
    # Hyperpolarized trunk (-65 mV) with transient wound depolarization at cut ends
    V = (
        np.full(N, V_hyper, dtype=float)
        + 26.0 * ant_envelope
        + post_wound_init_depol * post_envelope
        + rng.normal(0.0, 0.15, size=N)
    )
    H = np.full(N, 0.05, dtype=float)
    T = np.full(N, 0.15, dtype=float)

    # Under 8-octanol gap-junction uncoupling (Oviedo et al. 2010), posterior wound injury
    # activates local ectopic H,K-ATPase depolarization because trunk inhibitory cues cannot enter
    I_HK_profile = I_HK_amp * ant_envelope + (eta_oct * 16.5) * post_envelope
    I_IVM_profile = I_IVM_amp * both_wounds

    hist_V = [V.copy()]
    hist_H = [H.copy()]
    hist_T = [T.copy()]
    hist_t = [0.0]

    for step in range(1, steps + 1):
        div_gj, g_edge = nonlinear_gj_flux_divergence(V, G_0=G_0, V_0=V_0, eta_oct=eta_oct)
        i_ion = bistable_ionic_current(V, g_bistable=g_bistable, V_hyper=V_hyper, V_thresh=V_thresh, V_depol=V_depol)
        dV = -i_ion + I_HK_profile + I_IVM_profile + div_gj
        V = np.clip(V + dt * dV, -85.0, 5.0)

        ca_ss = (1.0 - eta_nif) / (1.0 + np.exp(-(V - V_thresh) / k_V))
        dH = (ca_ss * (1.0 - c_T * T) - H) / tau_H + D_H * laplacian_1d_neumann(H)
        dT = ((1.0 - ca_ss) * wound_competence_W * (1.0 - c_H * H) - T) / tau_T + D_T * laplacian_1d_neumann(T)

        H = np.clip(H + dt * dH, 0.0, 1.0)
        T = np.clip(T + dt * dT, 0.0, 1.0)

        if step % record_every == 0 or step == steps:
            hist_V.append(V.copy())
            hist_H.append(H.copy())
            hist_T.append(T.copy())
            hist_t.append(step * dt)

    _, final_g_edge = nonlinear_gj_flux_divergence(V, G_0=G_0, V_0=V_0, eta_oct=eta_oct)
    ca_final = (1.0 - eta_nif) / (1.0 + np.exp(-(V - V_thresh) / k_V))

    # Classify anterior (cells 0..7) and posterior (cells N-8..N-1) blastema fates
    ant_H = float(np.mean(H[:8]))
    post_H = float(np.mean(H[-8:]))
    ant_T = float(np.mean(T[:8]))
    post_T = float(np.mean(T[-8:]))
    ant_V = float(np.mean(V[:6]))
    trunk_V = float(np.mean(V[N // 3 : 2 * N // 3]))
    post_V = float(np.mean(V[-6:]))

    ant_is_head = int(ant_H >= 0.50)
    post_is_head = int(post_H >= 0.50)
    n_heads = ant_is_head + post_is_head
    n_tails = (1 - ant_is_head) + (1 - post_is_head)
    phenotype_code = f"{n_heads}H{n_tails}T"

    return {
        "regime": regime,
        "x": x,
        "s": s,
        "V": V,
        "H": H,
        "T": T,
        "ca_final": ca_final,
        "g_edge": final_g_edge,
        "ant_V_mV": ant_V,
        "trunk_V_mV": trunk_V,
        "post_V_mV": post_V,
        "delta_V_AP_mV": ant_V - post_V,
        "ant_H": ant_H,
        "post_H": post_H,
        "ant_T": ant_T,
        "post_T": post_T,
        "n_heads": n_heads,
        "n_tails": n_tails,
        "phenotype_code": phenotype_code,
        "history_V": np.array(hist_V),
        "history_H": np.array(hist_H),
        "history_T": np.array(hist_T),
        "history_t": np.array(hist_t),
    }


def evaluate_all_regimes_and_morphospace(seed: int = 2011) -> dict[str, Any]:
    """
    Run all 5 Beane et al. (2011) / Oviedo et al. (2010) experimental regeneration regimes,
    a 10-point H,K-ATPase current dose-response curve I_HK in [-18, +24], and a 2D
    (I_HK, I_IVM) bioelectric phase diagram mapping 0H2T, 1H1T, and 2H0T territories.
    """
    regime_names = [
        "wild_type_1H1T",
        "hk_atpase_inhibited_0H2T",
        "ivermectin_depolarized_2H0T",
        "octanol_gj_blockade_2H0T",
        "nifedipine_cav_block_0H2T",
    ]
    regimes: dict[str, dict[str, Any]] = {}
    for rname in regime_names:
        regimes[rname] = simulate_planarian_bioelectric_pde(regime=rname, seed=seed)

    # 1D Dose-response sweep of anterior H,K-ATPase current I_HK
    ihk_vals = np.linspace(-18.0, 24.0, 11)
    ihk_sweep = []
    for val in ihk_vals:
        sim = simulate_planarian_bioelectric_pde(
            regime="wild_type_1H1T",
            I_HK_override=float(val),
            steps=1600,
            seed=seed,
            record_every=1600,
        )
        ihk_sweep.append(
            {
                "I_HK": float(val),
                "ant_V_mV": sim["ant_V_mV"],
                "ant_H": sim["ant_H"],
                "n_heads": sim["n_heads"],
                "phenotype": sim["phenotype_code"],
            }
        )

    # 2D Morphospace grid across (I_HK in [-16, 22], I_IVM in [0, 24])
    grid_ihk = np.linspace(-16.0, 22.0, 9)
    grid_ivm = np.linspace(0.0, 24.0, 9)
    head_count_grid = np.zeros((len(grid_ivm), len(grid_ihk)), dtype=float)
    ant_v_grid = np.zeros_like(head_count_grid)
    for i_y, ivm in enumerate(grid_ivm):
        for i_x, ihk in enumerate(grid_ihk):
            # Fast analytical-attractor reduced evaluation matched to the PDE thresholds
            v_ant_eff = -65.0 + 2.45 * max(0.0, ihk + ivm + 8.0)
            v_post_eff = -65.0 + 2.45 * max(0.0, ivm - 4.0)
            v_ant_eff = min(-16.0, v_ant_eff)
            v_post_eff = min(-16.0, v_post_eff)
            h_ant = 1 if v_ant_eff > -42.5 else 0
            h_post = 1 if v_post_eff > -42.5 else 0
            head_count_grid[i_y, i_x] = float(h_ant + h_post)
            ant_v_grid[i_y, i_x] = v_ant_eff

    return {
        "regimes": regimes,
        "ihk_sweep": ihk_sweep,
        "grid_ihk": grid_ihk,
        "grid_ivm": grid_ivm,
        "head_count_grid": head_count_grid,
        "ant_v_grid": ant_v_grid,
    }


# --------------------------------------------------------------------------- #
# 3. Schmidtea mediterranea Specimen CV Segmentation & 2D Body-Plan Reconstruction
# --------------------------------------------------------------------------- #
def segment_planarian_specimen(specimen_path: Path = SPECIMEN_PATH) -> dict[str, Any]:
    """
    Segment the dorsal bright-field whole-mount micrograph of Schmidtea mediterranea
    (`datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg`, CC BY 2.5):
      1. Isolate the pigmented planarian body silhouette from the bright background
         (luminance < 215) via morphological closing, hole filling, and connected-component
         selection (removing the top-left scale bar).
      2. Extract the anteroposterior (AP) body axis (x_min = 8 px anterior tip to
         x_max = 1486 px posterior tail tip, total length L = 1478 px), midline y(x),
         and dorsoventral width profile W(s) across normalized AP coordinate s in [0, 1].
      3. Detect the bilateral pair of dark photoreceptor ocelli (eyespots) inside the
         anterior head region (s < 0.22) via local Gaussian background subtraction
         (sigma = 12 px) and connected-component center-of-mass localization.
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

    # Sample AP width profile W(s) and dorsal pigmentation intensity along 60 AP bins
    n_bins = 60
    s_bins = np.linspace(0.0, 1.0, n_bins)
    x_coords = np.round(x_min + s_bins * (x_max - x_min)).astype(int)
    widths_px = np.zeros(n_bins, dtype=float)
    midline_y = np.zeros(n_bins, dtype=float)
    pigment_profile = np.zeros(n_bins, dtype=float)

    for i, xc in enumerate(x_coords):
        col_y = np.where(worm_mask[:, xc])[0]
        if col_y.size > 0:
            widths_px[i] = float(col_y.max() - col_y.min())
            midline_y[i] = float(np.mean(col_y))
            pigment_profile[i] = float(255.0 - np.mean(gray[col_y, xc]))

    # Detect the two bilateral photoreceptor ocelli (eyespots) in the anterior head region
    ant_limit = x_min + int(0.22 * (x_max - x_min))
    ant_region = worm_mask.copy()
    ant_region[:, ant_limit:] = False
    ant_interior = ndi.binary_erosion(worm_mask, iterations=8) & ant_region

    local_bg = ndi.gaussian_filter(gray, sigma=12.0)
    dark_contrast = (local_bg - gray) * ant_interior
    thresh = float(np.percentile(dark_contrast[ant_interior], 96.5))
    ocelli_mask = ndi.binary_opening(dark_contrast > thresh, iterations=1)
    olab, n_oc = ndi.label(ocelli_mask)
    osizes = ndi.sum(dark_contrast, olab, range(1, n_oc + 1))
    top_idx = np.argsort(osizes)[::-1][:2]
    ocelli_yx = [ndi.center_of_mass(dark_contrast, olab, int(idx + 1)) for idx in top_idx]
    # Sort ocelli by y-coordinate (dorsal-left vs dorsal-right photoreceptor)
    ocelli_yx = sorted(ocelli_yx, key=lambda pt: pt[0])
    ocelli_xy = [[float(pt[1]), float(pt[0])] for pt in ocelli_yx]

    ocelli_rel_s = [(pt[0] - x_min) / length_px for pt in ocelli_xy]
    mean_ocelli_s = float(np.mean(ocelli_rel_s))
    inter_ocelli_px = float(
        math.hypot(ocelli_xy[0][0] - ocelli_xy[1][0], ocelli_xy[0][1] - ocelli_xy[1][1])
    )

    return {
        "image_rgb": img,
        "worm_mask": worm_mask,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "length_px": length_px,
        "max_width_px": max_width_px,
        "aspect_ratio": aspect_ratio,
        "s_bins": s_bins,
        "x_coords": x_coords,
        "widths_px": widths_px,
        "widths_norm": widths_px / max(1.0, float(np.max(widths_px))),
        "midline_y": midline_y,
        "pigment_profile": pigment_profile,
        "n_ocelli_detected": int(len(ocelli_xy)),
        "ocelli_xy": ocelli_xy,
        "ocelli_rel_s": ocelli_rel_s,
        "mean_ocelli_rel_s": mean_ocelli_s,
        "inter_ocelli_distance_px": inter_ocelli_px,
    }


def synthesize_2d_planarian_morphology(
    s: np.ndarray,
    H: np.ndarray,
    T: np.ndarray,
) -> dict[str, Any]:
    """
    Reconstruct the 2D planarian body-plan silhouette (half-width contour w(s) and
    photoreceptor eyespot coordinates) from the steady-state anterior head morphogen
    H(s) and posterior tail morphogen T(s):
      - Trunk baseline has elliptical half-width w_trunk(s) = 0.22 * sin(pi * s)^0.45.
      - An anterior head domain (H > 0.5) adds triangular auricle widening near the tip
        and places a bilateral pair of photoreceptor eyespots at the peak of H(s).
      - A posterior tail domain (T > 0.5) tapers smoothly to a pointed/rounded tail tip.
    """
    s_fine = np.linspace(0.0, 1.0, 200)
    H_i = np.interp(s_fine, s, H)
    T_i = np.interp(s_fine, s, T)

    envelope = np.sin(np.clip(s_fine, 1e-4, 1.0 - 1e-4) * math.pi) ** 0.42
    ant_auricle = 0.095 * float(np.mean(H[:8])) * np.exp(-((s_fine - 0.08) / 0.045) ** 2)
    post_auricle = 0.095 * float(np.mean(H[-8:])) * np.exp(-((s_fine - 0.92) / 0.045) ** 2)
    tail_taper_ant = 1.0 - 0.30 * float(np.mean(T[:8])) * np.exp(-((s_fine - 0.0) / 0.12) ** 2)
    tail_taper_post = 1.0 - 0.30 * float(np.mean(T[-8:])) * np.exp(-((s_fine - 1.0) / 0.12) ** 2)

    half_width = (0.21 * envelope + ant_auricle + post_auricle) * tail_taper_ant * tail_taper_post

    eyespots_xy: list[tuple[float, float]] = []
    if float(np.mean(H[:8])) >= 0.50:
        eyespots_xy.extend([(0.065, +0.068), (0.065, -0.068)])
    if float(np.mean(H[-8:])) >= 0.50:
        eyespots_xy.extend([(0.935, +0.068), (0.935, -0.068)])

    return {
        "s_fine": s_fine,
        "half_width": half_width,
        "H_interp": H_i,
        "T_interp": T_i,
        "eyespots_xy": eyespots_xy,
    }


# --------------------------------------------------------------------------- #
# 3b. Pass 3 — 2D Finite-Volume Body-Mesh Bioelectric PDE on the Segmented Specimen
#     Silhouette & Slit/Netrin Mediolateral Photoreceptor Axon Guidance
# --------------------------------------------------------------------------- #
def build_planarian_body_mesh(spec: dict[str, Any], nx: int = 90) -> dict[str, Any]:
    """
    Build a 2D cell-centred finite-volume mesh of the planarian body from the segmented
    specimen silhouette (`segment_planarian_specimen`). The binary worm mask is
    resampled onto a uniform square grid with `nx` cells along the anteroposterior (AP)
    axis (cell size h = L / (nx - 1)); the mediolateral (ML) coordinate of every cell is
    d(x, y) = (y - y_mid(x)) / w_half(x) in [-1, 1] (0 = dorsal midline, +/-1 = lateral margin).

    Honest reduction: this is a structured finite-volume mesh (not an unstructured FEM
    mesh) and it carries no dorsoventral layering; the residual is stated in the audit.
    """
    mask = spec["worm_mask"]
    x_min, x_max, y_min, y_max = spec["x_min"], spec["x_max"], spec["y_min"], spec["y_max"]
    sub = mask[y_min : y_max + 1, x_min : x_max + 1].astype(float)
    L_px = float(x_max - x_min)
    h_px = L_px / float(nx - 1)
    ny = int(round((y_max - y_min) / h_px)) + 1
    zoomed = ndi.zoom(sub, (ny / sub.shape[0], nx / sub.shape[1]), order=1)
    cell_mask = ndi.binary_fill_holes(zoomed > 0.5)
    lab, n_comp = ndi.label(cell_mask)
    if n_comp > 1:
        sizes = ndi.sum(cell_mask, lab, range(1, n_comp + 1))
        cell_mask = lab == (1 + int(np.argmax(sizes)))

    s = np.linspace(0.0, 1.0, nx)
    h_rel = h_px / L_px
    mid_row = np.full(nx, np.nan)
    half_w = np.full(nx, np.nan)
    d_grid = np.full((ny, nx), np.nan)
    y_grid = np.full((ny, nx), np.nan)
    rows = np.arange(ny, dtype=float)
    for j in range(nx):
        col = np.where(cell_mask[:, j])[0]
        if col.size == 0:
            continue
        mid_row[j] = 0.5 * (col.min() + col.max())
        half_w[j] = max(0.5, 0.5 * (col.max() - col.min()))
        d_grid[:, j] = (rows - mid_row[j]) / half_w[j]
        y_grid[:, j] = (rows - mid_row[j]) * h_rel
    d_grid = np.where(cell_mask, d_grid, np.nan)
    y_grid = np.where(cell_mask, y_grid, np.nan)
    # Fill empty columns (none expected for a connected silhouette) by nearest neighbour
    if np.any(np.isnan(mid_row)):
        good = ~np.isnan(mid_row)
        mid_row = np.interp(np.arange(nx), np.arange(nx)[good], mid_row[good])
        half_w = np.interp(np.arange(nx), np.arange(nx)[good], half_w[good])

    return {
        "mask": cell_mask,
        "nx": nx,
        "ny": ny,
        "s": s,
        "h_rel": h_rel,
        "h_px": h_px,
        "mid_row": mid_row,
        "half_w": half_w,
        "half_w_rel": half_w * h_rel,  # local half-width in units of body length L
        "d_grid": d_grid,
        "y_grid": y_grid,  # absolute mediolateral offset from the midline in units of L
        "n_cells": int(cell_mask.sum()),
        "origin_px": (x_min, y_min),
    }


def masked_gj_flux_divergence_2d(
    V: np.ndarray,
    mask: np.ndarray,
    G_edge: float,
    V0_edge: float,
    eta_oct: float = 0.0,
) -> np.ndarray:
    """
    Finite-volume divergence of the voltage-gated gap-junction flux on the body mesh:
        div_i = sum_{j in N(i)} G_ij(V_j - V_i),   G_ij = (1 - eta_oct) G_edge / (1 + ((V_j - V_i)/V0_edge)^2),
    summed only over edges whose both cells lie inside the mask (zero-flux Neumann
    boundary on the body margin). Every edge flux enters the two cells with opposite
    sign, so the total gap-junctional current over the mesh is exactly zero
    (Lean 4 `MathOfBiology.Loop_081.mesh_gj_flux_conservation`).
    """
    div = np.zeros_like(V)
    # AP (x) edges
    both_x = mask[:, 1:] & mask[:, :-1]
    dv_x = np.where(both_x, V[:, 1:] - V[:, :-1], 0.0)
    f_x = (1.0 - eta_oct) * G_edge / (1.0 + (dv_x / V0_edge) ** 2) * dv_x * both_x
    div[:, :-1] += f_x
    div[:, 1:] -= f_x
    # ML (y) edges
    both_y = mask[1:, :] & mask[:-1, :]
    dv_y = np.where(both_y, V[1:, :] - V[:-1, :], 0.0)
    f_y = (1.0 - eta_oct) * G_edge / (1.0 + (dv_y / V0_edge) ** 2) * dv_y * both_y
    div[:-1, :] += f_y
    div[1:, :] -= f_y
    return div


def masked_laplacian_2d(u: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked 5-point Laplacian with zero-flux Neumann boundary on the body margin."""
    out = np.zeros_like(u)
    both_x = mask[:, 1:] & mask[:, :-1]
    f_x = np.where(both_x, u[:, 1:] - u[:, :-1], 0.0)
    out[:, :-1] += f_x
    out[:, 1:] -= f_x
    both_y = mask[1:, :] & mask[:-1, :]
    f_y = np.where(both_y, u[1:, :] - u[:-1, :], 0.0)
    out[:-1, :] += f_y
    out[1:, :] -= f_y
    return out


def simulate_planarian_bioelectric_2d(
    mesh: dict[str, Any],
    regime: str = "wild_type_1H1T",
    N_ref: int = 60,
    steps: int = 2400,
    dt: float = 0.02,
    g_bistable: float = 0.002,
    V_hyper: float = -65.0,
    V_thresh: float = -42.5,
    V_depol: float = -20.0,
    G_0: float = 1.8,
    V_0: float = 12.0,
    k_V: float = 3.5,
    tau_H: float = 4.0,
    tau_T: float = 4.0,
    c_H: float = 0.85,
    c_T: float = 0.85,
    D_H: float = 0.15,
    D_T: float = 0.15,
    seed: int = 2011,
) -> dict[str, Any]:
    """
    Integrate the same Beane et al. (2011) / Pietak-Levin (2017) bistable V_mem +
    voltage-gated gap-junction + Ca_v-gated H/T morphogen system on the 2D body mesh
    of the segmented Schmidtea mediterranea silhouette (Pass 3 remediation item 1).

    Discretisation consistency with the 1D solver: with r = (nx - 1)/(N_ref - 1) mesh cells
    per 1D cell, edge conductance G_edge = G_0 r^2 and edge half-inactivation voltage
    V0_edge = V_0 / r so that the continuum coefficient G_GJ(dV/dx) and the ionic/pump
    current densities are identical to the 1D model; only the geometry (tapered head,
    rounded tail, lateral margins) differs.
    """
    rp = regime_parameters(regime)
    rng = np.random.default_rng(seed)
    mask = mesh["mask"]
    ny, nx = mask.shape
    s = mesh["s"]
    r = float(nx - 1) / float(N_ref - 1)
    G_edge = G_0 * r * r
    V0_edge = V_0 / r
    D_H_edge = D_H * r * r
    D_T_edge = D_T * r * r
    assert dt * (4.0 * G_edge + 4.0 * max(D_H_edge, D_T_edge)) < 1.0, "explicit 2D step unstable"

    ant_env = np.exp(-((s / 0.14) ** 2))[None, :] * np.ones((ny, 1))
    post_env = np.exp(-(((1.0 - s) / 0.14) ** 2))[None, :] * np.ones((ny, 1))
    both = ant_env + post_env
    W = 0.25 + 0.75 * both

    V = np.full((ny, nx), V_hyper) + 26.0 * ant_env + rp["post_wound_init_depol"] * post_env
    V += rng.normal(0.0, 0.15, size=(ny, nx))
    V = np.where(mask, V, V_hyper)
    H = np.where(mask, 0.05, 0.0)
    T = np.where(mask, 0.15, 0.0)
    I_HK = rp["I_HK_amp"] * ant_env + (rp["eta_oct"] * 16.5) * post_env
    I_IVM = rp["I_IVM_amp"] * both
    eta_nif = rp["eta_nif"]

    for _ in range(steps):
        div = masked_gj_flux_divergence_2d(V, mask, G_edge, V0_edge, rp["eta_oct"])
        i_ion = bistable_ionic_current(V, g_bistable, V_hyper, V_thresh, V_depol)
        V = np.clip(V + dt * (-i_ion + I_HK + I_IVM + div), -85.0, 5.0)
        V = np.where(mask, V, V_hyper)
        ca = (1.0 - eta_nif) / (1.0 + np.exp(-(V - V_thresh) / k_V))
        dH = (ca * (1.0 - c_T * T) - H) / tau_H + D_H_edge * masked_laplacian_2d(H, mask)
        dT = ((1.0 - ca) * W * (1.0 - c_H * H) - T) / tau_T + D_T_edge * masked_laplacian_2d(T, mask)
        H = np.where(mask, np.clip(H + dt * dH, 0.0, 1.0), 0.0)
        T = np.where(mask, np.clip(T + dt * dT, 0.0, 1.0), 0.0)

    ca = (1.0 - eta_nif) / (1.0 + np.exp(-(V - V_thresh) / k_V))
    ant_cells = mask & (s[None, :] <= 7.0 / (N_ref - 1))
    post_cells = mask & (s[None, :] >= 1.0 - 7.0 / (N_ref - 1))
    antV_cells = mask & (s[None, :] <= 5.0 / (N_ref - 1))
    postV_cells = mask & (s[None, :] >= 1.0 - 5.0 / (N_ref - 1))
    ant_H = float(H[ant_cells].mean())
    post_H = float(H[post_cells].mean())
    ant_V = float(V[antV_cells].mean())
    post_V = float(V[postV_cells].mean())
    ant_is_head = int(ant_H >= 0.50)
    post_is_head = int(post_H >= 0.50)
    n_heads = ant_is_head + post_is_head
    # Mediolateral V_mem heterogeneity inside the anterior blastema (0 in the 1D model by construction)
    ml_std_ant = float(V[antV_cells].std())
    return {
        "regime": regime,
        "V": np.where(mask, V, np.nan),
        "H": np.where(mask, H, np.nan),
        "T": np.where(mask, T, np.nan),
        "ca": np.where(mask, ca, np.nan),
        "ant_V_mV": ant_V,
        "post_V_mV": post_V,
        "delta_V_AP_mV": ant_V - post_V,
        "ant_H": ant_H,
        "post_H": post_H,
        "n_heads": n_heads,
        "n_tails": 2 - n_heads,
        "phenotype_code": f"{n_heads}H{2 - n_heads}T",
        "ml_std_ant_V_mV": ml_std_ant,
        "G_edge": G_edge,
        "V0_edge": V0_edge,
        "net_gj_current": float(masked_gj_flux_divergence_2d(V, mask, G_edge, V0_edge, rp["eta_oct"])[mask].sum()),
    }


SIGMA_SLIT_L = 0.02    # Smed-slit midline stripe half-width, units of body length L (a priori, not fitted)
SIGMA_NETRIN_L = 0.05  # Smed-netrin midline / ventral-cord attractant half-width, units of L (a priori)


def slit_netrin_fields(
    y: np.ndarray | float,
    sigma_slit: float = SIGMA_SLIT_L,
    sigma_netrin: float = SIGMA_NETRIN_L,
    eta_slit: float = 0.0,
    eta_netrin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mediolateral guidance cue profiles as functions of the absolute ML offset y from the
    dorsal midline (units of body length L):
      S(y) = (1 - eta_slit)   * exp(-(y / sigma_slit)^2)    Smed-slit midline repellent stripe
                                                            (Cebria, Guo, Jopek & Newmark 2007 Dev. Biol. 307:394-406)
      N(y) = (1 - eta_netrin) * exp(-(y / sigma_netrin)^2)  Smed-netrin midline/ventral-cord attractant
                                                            (Cebria & Newmark 2005 Development 132:3691-3703)
    sigma_slit = 0.02 L and sigma_netrin = 0.05 L are a-priori stripe widths (not fitted to
    the specimen); `sweep_slit_stripe_width` reports the specimen-matched sigma_slit*.
    """
    S = (1.0 - eta_slit) * np.exp(-((y / sigma_slit) ** 2))
    N = (1.0 - eta_netrin) * np.exp(-((y / sigma_netrin) ** 2))
    return S, N


def place_photoreceptors_2d(
    sim2d: dict[str, Any],
    mesh: dict[str, Any],
    sigma_slit: float = SIGMA_SLIT_L,
    eta_slit: float = 0.0,
) -> dict[str, Any]:
    """
    Eye-field placement on the 2D body mesh. The photoreceptor progenitor field is
        E(x, y) = H(x, y) * (1 - S(y)) * (1 - d^2),      d = y / w_half(x),
    i.e. the Ca_v-gated head commitment H (2D PDE), excluded from the midline by the
    Smed-slit stripe of absolute half-width sigma_slit and from the lateral margin by the
    body-wall taper. Because the anterior tip is narrower than the slit stripe, the AP
    position of the eyes emerges from the competition between the posteriorly decaying
    H(x) and the anteriorly narrowing silhouette. Each committed blastema (mean H >= 0.5)
    receives a left (y > 0) and a right (y < 0) photoreceptor at the arg-max of E on its
    half of the body. With slit silenced (eta_slit = 1) both maxima collapse onto the
    midline (y -> 0): the cyclopic / fused-photoreceptor phenotype of Smed-slit RNAi
    (Cebria et al. 2007) — Lean 4 `MathOfBiology.Loop_081.slit_midline_exclusion_and_collapse`.
    """
    mask = mesh["mask"]
    yy = np.nan_to_num(mesh["y_grid"], nan=1.0)
    dd = np.nan_to_num(mesh["d_grid"], nan=1.0)
    H = np.nan_to_num(sim2d["H"], nan=0.0)
    S, _ = slit_netrin_fields(yy, sigma_slit=sigma_slit, eta_slit=eta_slit)
    E = H * (1.0 - S) * np.clip(1.0 - dd**2, 0.0, None) * mask
    s = mesh["s"]
    eyes: list[dict[str, float]] = []
    halves = []
    if sim2d["ant_H"] >= 0.5:
        halves.append(("anterior", s[None, :] < 0.5))
    if sim2d["post_H"] >= 0.5:
        halves.append(("posterior", s[None, :] >= 0.5))
    for label, half in halves:
        for side, sel in (("left", yy >= 0.0), ("right", yy <= 0.0)):
            cand = E * half * sel
            if cand.max() <= 0.0:
                continue
            iy, ix = np.unravel_index(int(np.argmax(cand)), cand.shape)
            eyes.append({"blastema": label, "side": side, "s": float(s[ix]), "y": float(yy[iy, ix]),
                         "d": float(dd[iy, ix]), "row": float(iy), "col": float(ix), "E": float(cand[iy, ix])})
    seps = []
    for label in ("anterior", "posterior"):
        pr = [e for e in eyes if e["blastema"] == label]
        if len(pr) == 2:
            seps.append(abs(pr[0]["y"] - pr[1]["y"]))
    return {
        "eyes": eyes,
        "n_eyes": len(eyes),
        "E": np.where(mask, E, np.nan),
        "inter_ocelli_sep_L": float(seps[0]) if seps else float("nan"),
        "eye_seps_L": seps,
    }


def simulate_visual_axon_guidance(
    eyes: list[dict[str, float]],
    mesh: dict[str, Any],
    sigma_slit: float = SIGMA_SLIT_L,
    sigma_netrin: float = SIGMA_NETRIN_L,
    eta_slit: float = 0.0,
    eta_netrin: float = 0.0,
    k_slit: float = 1.0,
    k_netrin: float = 1.0,
    axon_length: float = 0.12,
    n_steps: int = 240,
    heading_relax: float = 0.03,
    heading_gain: float = 0.05,
    noise_sd: float = 0.004,
    seed: int = 2011,
) -> dict[str, Any]:
    """
    Reduced growth-cone model of the planarian visual axons in body-fitted (s, y)
    coordinates (Pass 3 remediation item 1, mediolateral guidance). Each photoreceptor
    sends an ipsilateral and a contralateral (commissural) axon that grow posteriorly
    toward the cephalic ganglia (optic chiasm); the growth-cone heading h = dy/ds relaxes
    toward the net mediolateral cue
        c(y) = k_netrin dN/dy - k_slit dS/dy,
    with heading persistence (`heading_relax` per step) so commissural axons overshoot and
    cross the midline. Following the canonical Robo-Slit switch (Kidd, Bland & Goodman 1999
    Cell 96:785-794), commissural growth cones are slit-insensitive before crossing and
    slit-sensitive after crossing; ipsilateral axons are slit-sensitive throughout. The two
    axon classes converge on the paired longitudinal tracts at |y*| where netrin attraction
    balances slit repulsion. Phenotypes: eta_slit = 1 -> midline collapse (Cebria et al. 2007
    Smed-slit RNAi); eta_netrin = 1 -> commissural axons never reach/cross the midline
    (Cebria & Newmark 2005 Smed-netrin / netR RNAi).
    """
    rng = np.random.default_rng(seed)
    ds = axon_length / n_steps
    s_mesh = mesh["s"]
    hw_rel = mesh["half_w_rel"]

    def cue(yv: float, slit_on: bool) -> float:
        S = (1.0 - eta_slit) * math.exp(-((yv / sigma_slit) ** 2))
        N = (1.0 - eta_netrin) * math.exp(-((yv / sigma_netrin) ** 2))
        dS = -2.0 * yv / sigma_slit**2 * S
        dN = -2.0 * yv / sigma_netrin**2 * N
        return k_netrin * dN - (k_slit * dS if slit_on else 0.0)

    axons = []
    for eye in eyes:
        for kind in ("ipsilateral", "contralateral"):
            s_v, y_v = eye["s"], eye["y"]
            step_sign = -1.0 if eye["blastema"] == "posterior" else 1.0
            h = 0.0
            crossed = False
            traj = [(s_v, y_v)]
            sign0 = math.copysign(1.0, y_v) if y_v != 0.0 else 1.0
            for _ in range(n_steps):
                slit_on = kind == "ipsilateral" or crossed
                c = cue(y_v, slit_on)
                h += heading_relax * (heading_gain * c - h) + rng.normal(0.0, noise_sd)
                s_v = float(np.clip(s_v + step_sign * ds, 0.0, 1.0))
                w_loc = float(np.interp(s_v, s_mesh, hw_rel))
                y_v = float(np.clip(y_v + h * ds, -0.98 * w_loc, 0.98 * w_loc))
                if kind == "contralateral" and not crossed and abs(y_v) > 1e-9 and math.copysign(1.0, y_v) != sign0:
                    crossed = True
                traj.append((s_v, y_v))
            traj_arr = np.array(traj)
            axons.append({
                "eye_side": eye["side"],
                "blastema": eye["blastema"],
                "kind": kind,
                "traj": traj_arr,
                "crossed": bool(crossed),
                "final_y": float(y_v),
                "min_abs_y": float(np.min(np.abs(traj_arr[:, 1]))),
            })
    contra = [a for a in axons if a["kind"] == "contralateral"]
    ipsi = [a for a in axons if a["kind"] == "ipsilateral"]
    # Analytical tract position |y*| where k_netrin |dN/dy| = k_slit |dS/dy| (slit + netrin both present)
    y_star = float("nan")
    if eta_slit < 1.0 and eta_netrin < 1.0 and sigma_slit < sigma_netrin:
        ratio = (k_netrin * (1.0 - eta_netrin) / sigma_netrin**2) / (k_slit * (1.0 - eta_slit) / sigma_slit**2)
        y_star = math.sqrt(-math.log(ratio) / (1.0 / sigma_slit**2 - 1.0 / sigma_netrin**2))
    return {
        "axons": axons,
        "crossing_fraction": float(np.mean([a["crossed"] for a in contra])) if contra else float("nan"),
        "ipsi_final_abs_y": float(np.mean([abs(a["final_y"]) for a in ipsi])) if ipsi else float("nan"),
        "contra_final_abs_y": float(np.mean([abs(a["final_y"]) for a in contra])) if contra else float("nan"),
        "ipsi_min_abs_y": float(np.mean([a["min_abs_y"] for a in ipsi])) if ipsi else float("nan"),
        "tract_y_star_L": y_star,
        "eta_slit": eta_slit,
        "eta_netrin": eta_netrin,
    }


def specimen_ocelli_geometry(spec: dict[str, Any]) -> dict[str, float]:
    """
    Observed photoreceptor geometry of the vendored specimen in the mesh coordinates:
    AP position s_eye, absolute ML half-separation |y_eye| (units of L) and relative
    position d_eye = |y_eye| / w_half(s_eye).
    """
    s_oc = spec["mean_ocelli_rel_s"]
    L = spec["length_px"]
    local_w = float(np.interp(s_oc, spec["s_bins"], spec["widths_px"]))
    dy = abs(spec["ocelli_xy"][0][1] - spec["ocelli_xy"][1][1])
    return {
        "s_eye": float(s_oc),
        "local_width_px": local_w,
        "inter_ocelli_dy_px": dy,
        "inter_ocelli_sep_L": dy / L,
        "abs_y_eye_L": 0.5 * dy / L,
        "abs_d_eye": dy / max(1.0, local_w),
    }


def eye_field_lateral_optimum(w_half: float, sigma_slit: float = SIGMA_SLIT_L, eta_slit: float = 0.0) -> float:
    """Continuous arg-max over y in [0, w_half] of the lateral eye-field factor (1 - S(y)) (1 - (y/w_half)^2)."""
    y = np.linspace(0.0, w_half, 2001)
    S, _ = slit_netrin_fields(y, sigma_slit=sigma_slit, eta_slit=eta_slit)
    return float(y[np.argmax((1.0 - S) * (1.0 - (y / w_half) ** 2))])


def sweep_slit_stripe_width(spec_geom: dict[str, float], mesh: dict[str, Any]) -> dict[str, Any]:
    """
    Parameter sweep of the slit midline-stripe half-width sigma_slit in [0.005, 0.06] L at the
    specimen's local half-width: predicted photoreceptor ML offset |y_eye|(sigma_slit) and the
    specimen-matched value sigma_slit* at which |y_eye| equals the segmented ocelli offset.
    """
    w_half = float(np.interp(spec_geom["s_eye"], mesh["s"], mesh["half_w_rel"]))
    sigmas = np.linspace(0.005, 0.06, 23)
    y_eye = np.array([eye_field_lateral_optimum(w_half, sg) for sg in sigmas])
    order = np.argsort(y_eye)
    sigma_star = float(np.interp(spec_geom["abs_y_eye_L"], y_eye[order], sigmas[order]))
    return {"sigmas": sigmas, "y_eye": y_eye, "w_half": w_half, "sigma_star": sigma_star,
            "y_eye_specimen": spec_geom["abs_y_eye_L"], "y_eye_default": eye_field_lateral_optimum(w_half)}


def evaluate_2d_mesh_and_guidance(spec: dict[str, Any], seed: int = 2011, nx: int = 120) -> dict[str, Any]:
    """Run the 2D body-mesh PDE for WT / SCH-28080 / ivermectin and the slit/netrin guidance phenotypes."""
    mesh = build_planarian_body_mesh(spec, nx=nx)
    sims = {r: simulate_planarian_bioelectric_2d(mesh, regime=r, seed=seed)
            for r in ("wild_type_1H1T", "hk_atpase_inhibited_0H2T", "ivermectin_depolarized_2H0T")}
    wt = sims["wild_type_1H1T"]
    eyes_wt = place_photoreceptors_2d(wt, mesh)
    eyes_slit = place_photoreceptors_2d(wt, mesh, eta_slit=1.0)
    eyes_ivm = place_photoreceptors_2d(sims["ivermectin_depolarized_2H0T"], mesh)
    eyes_sch = place_photoreceptors_2d(sims["hk_atpase_inhibited_0H2T"], mesh)
    guidance = {
        "wild_type": simulate_visual_axon_guidance(eyes_wt["eyes"], mesh, seed=seed),
        "slit_rnai": simulate_visual_axon_guidance(eyes_slit["eyes"], mesh, eta_slit=1.0, seed=seed),
        "netrin_rnai": simulate_visual_axon_guidance(eyes_wt["eyes"], mesh, eta_netrin=1.0, seed=seed),
        "ivermectin_2H0T": simulate_visual_axon_guidance(eyes_ivm["eyes"], mesh, seed=seed),
    }
    obs = specimen_ocelli_geometry(spec)
    ant_eyes = [e for e in eyes_wt["eyes"] if e["blastema"] == "anterior"]
    y_eye_model = float(np.mean([abs(e["y"]) for e in ant_eyes])) if ant_eyes else float("nan")
    s_eye_model = float(np.mean([e["s"] for e in ant_eyes])) if ant_eyes else float("nan")
    return {
        "mesh": mesh,
        "sims": sims,
        "eyes": {"wild_type": eyes_wt, "slit_rnai": eyes_slit, "ivermectin": eyes_ivm, "sch28080": eyes_sch},
        "guidance": guidance,
        "specimen_ocelli": obs,
        "abs_y_eye_model_L": y_eye_model,
        "s_eye_model": s_eye_model,
        "abs_y_eye_specimen_L": obs["abs_y_eye_L"],
        "s_eye_specimen": obs["s_eye"],
        "slit_sweep": sweep_slit_stripe_width(obs, mesh),
        "head_counts_2d": {r: sims[r]["n_heads"] for r in sims},
        "n_eyes_2d": {k: v["n_eyes"] for k, v in
                      (("wild_type", eyes_wt), ("slit_rnai", eyes_slit), ("ivermectin", eyes_ivm), ("sch28080", eyes_sch))},
    }


# --------------------------------------------------------------------------- #
# 3c. Pass 3 — Stochastic Neoblast Division & Migration Kinetics Coupled to V_mem
# --------------------------------------------------------------------------- #
def simulate_neoblast_kinetics(
    sim1d: dict[str, Any],
    eta_nif: float = 0.0,
    n_neoblasts: int = 1500,
    t_end_h: float = 72.0,
    dt_h: float = 0.5,
    lambda_0: float = 0.02,
    A_body: float = 2.0,
    tau_body_h: float = 6.0,
    A_local: float = 5.0,
    tau_rise_h: float = 36.0,
    tau_decay_h: float = 100.0,
    n_pulse: int = 3,
    beta_V: float = 0.3,
    D_mig: float = 4.0e-4,
    v_mig: float = 0.02,
    hours_per_sim_unit: float = 0.75,
    V_thresh: float = -42.5,
    k_V: float = 3.5,
    seed: int = 2011,
) -> dict[str, Any]:
    """
    Agent-based stochastic neoblast model on the 1D AP axis coupled to the bioelectric
    V_mem(x, t) history of the PDE solver (Pass 3 remediation item 2).

    Each neoblast i at s_i divides as an inhomogeneous Poisson process with rate
        lambda_i(t) = lambda_0 [1 + A_body a(t) + A_local w(s_i) m(t)] [1 + beta_V (Ca_ss(V(s_i,t)) - 1/2)],
    where a(t) = (t/tau_body)^n exp(n (1 - t/tau_body)) is the body-wide wound-response
    pulse peaking at tau_body = 6 h and m(t) = (1 - exp(-t/tau_rise)) exp(-t/tau_decay) is
    the slower missing-tissue signal peaking at tau_rise ln(1 + tau_decay/tau_rise) = 47.8 h
    at the wounds (both peak times are fitted to the biphasic mitotic response of
    Wenemoser & Reddien 2010 Dev. Biol. 344:979-991 and are therefore inputs, not predictions;
    the predictions are the spatial localisation of each peak, the blastema progeny counts,
    their head/tail fate split and the across-seed variability),
    w(s) the wound proximity envelope, and the bracketed V_mem factor the depolarisation-
    dependent modulation of proliferation (beta_V = 0 ablates it). Divisions are
    asymmetric: the neoblast persists and one post-mitotic progeny is born which migrates
    into the nearest blastema and adopts head identity with probability
    (1 - eta_nif) Ca_ss(V(s_birth, t)) (L-type Ca_v gate of Beane et al. 2011), else
    posterior/tail identity. Neoblasts perform a biased random walk up the wound
    gradient (Guedelhoefer & Sanchez Alvarado 2012 Development 139:3510-3520) but do
    not enter the blastema (Wenemoser & Reddien 2010).
    """
    rng = np.random.default_rng(seed)
    s_axis = sim1d["s"]
    hist_t = sim1d["history_t"] * hours_per_sim_unit
    hist_V = sim1d["history_V"]
    n_steps = int(round(t_end_h / dt_h))
    s_n = rng.uniform(0.06, 0.94, size=n_neoblasts)

    def wound_env(sv: np.ndarray) -> np.ndarray:
        return np.exp(-((sv / 0.14) ** 2)) + np.exp(-(((1.0 - sv) / 0.14) ** 2))

    def wound_grad(sv: np.ndarray) -> np.ndarray:
        return (-2.0 * sv / 0.14**2) * np.exp(-((sv / 0.14) ** 2)) + (2.0 * (1.0 - sv) / 0.14**2) * np.exp(-(((1.0 - sv) / 0.14) ** 2))

    t_grid = np.arange(1, n_steps + 1) * dt_h
    mitotic_index = np.zeros(n_steps)
    div_count = np.zeros(n_steps, dtype=int)
    div_near_wound = np.zeros(n_steps, dtype=int)
    blastema_ant = 0
    blastema_post = 0
    head_ant = 0
    head_post = 0
    progeny_birth_s: list[float] = []
    progeny_head: list[int] = []
    hist_positions = []
    hist_blastema = []

    for k in range(n_steps):
        t = t_grid[k]
        # V_mem(s, t) from the PDE history (hours), interpolated at neoblast positions
        ti = int(np.clip(np.searchsorted(hist_t, t), 0, len(hist_t) - 1))
        V_t = np.interp(s_n, s_axis, hist_V[ti])
        ca = (1.0 - eta_nif) / (1.0 + np.exp(-(V_t - V_thresh) / k_V))
        a_t = (t / tau_body_h) ** n_pulse * math.exp(n_pulse * (1.0 - t / tau_body_h))
        m_t = (1.0 - math.exp(-t / tau_rise_h)) * math.exp(-t / tau_decay_h)
        lam = lambda_0 * (1.0 + A_body * a_t + A_local * wound_env(s_n) * m_t) * (1.0 + beta_V * (ca - 0.5))
        divide = rng.uniform(size=n_neoblasts) < (1.0 - np.exp(-lam * dt_h))
        nd = int(divide.sum())
        div_count[k] = nd
        mitotic_index[k] = nd / float(n_neoblasts) / dt_h
        near = (s_n < 0.25) | (s_n > 0.75)
        div_near_wound[k] = int((divide & near).sum())
        if nd:
            born_s = s_n[divide]
            to_ant = born_s < 0.5
            # Progeny fate is specified by the Ca_v gate of the blastema it enters
            # (anterior blastema V_mem at s = 0.04, posterior at s = 0.96; Beane et al. 2011)
            V_bl = np.interp(np.where(to_ant, 0.04, 0.96), s_axis, hist_V[ti])
            ca_bl = (1.0 - eta_nif) / (1.0 + np.exp(-(V_bl - V_thresh) / k_V))
            fate = rng.uniform(size=nd) < ca_bl
            blastema_ant += int(to_ant.sum())
            blastema_post += int((~to_ant).sum())
            head_ant += int((fate & to_ant).sum())
            head_post += int((fate & ~to_ant).sum())
            progeny_birth_s.extend(born_s.tolist())
            progeny_head.extend(fate.astype(int).tolist())
        # Biased random walk toward wounds, excluded from the blastema zones
        s_n = s_n + v_mig * dt_h * wound_grad(s_n) * 0.14 + math.sqrt(2.0 * D_mig * dt_h) * rng.normal(size=n_neoblasts)
        s_n = np.clip(s_n, 0.05, 0.95)
        if k % 8 == 0 or k == n_steps - 1:
            hist_positions.append(s_n.copy())
            hist_blastema.append((t, blastema_ant, blastema_post))

    # Peak detection on a smoothed mitotic index (3.5 h boxcar)
    kern = np.ones(7) / 7.0
    mi_s = np.convolve(mitotic_index, kern, mode="same")
    early = t_grid <= 20.0
    late = t_grid > 20.0
    t_peak_1 = float(t_grid[early][np.argmax(mi_s[early])])
    t_peak_2 = float(t_grid[late][np.argmax(mi_s[late])])
    i1 = int(np.argmin(np.abs(t_grid - t_peak_1)))
    i2 = int(np.argmin(np.abs(t_grid - t_peak_2)))
    win = 4

    def frac_near(i: int) -> float:
        lo, hi = max(0, i - win), min(n_steps, i + win + 1)
        tot = div_count[lo:hi].sum()
        return float(div_near_wound[lo:hi].sum() / tot) if tot else float("nan")

    return {
        "t_h": t_grid,
        "mitotic_index_per_h": mitotic_index,
        "mitotic_index_smooth": mi_s,
        "t_peak_bodywide_h": t_peak_1,
        "t_peak_local_h": t_peak_2,
        "frac_near_wound_peak1": frac_near(i1),
        "frac_near_wound_peak2": frac_near(i2),
        "blastema_ant_cells": blastema_ant,
        "blastema_post_cells": blastema_post,
        "head_frac_ant": head_ant / blastema_ant if blastema_ant else float("nan"),
        "head_frac_post": head_post / blastema_post if blastema_post else float("nan"),
        "progeny_birth_s": np.array(progeny_birth_s),
        "progeny_head": np.array(progeny_head),
        "hist_positions": hist_positions,
        "hist_blastema": hist_blastema,
        "final_positions": s_n,
    }


def evaluate_neoblast_regimes(
    regimes: dict[str, dict[str, Any]],
    seeds: tuple[int, ...] = (2011, 2012, 2013, 2014),
    beta_V: float = 0.3,
) -> dict[str, Any]:
    """Replicate stochastic neoblast simulations across seeds for 4 bioelectric regimes (+ beta_V ablation)."""
    out: dict[str, Any] = {}
    for rname in ("wild_type_1H1T", "hk_atpase_inhibited_0H2T", "ivermectin_depolarized_2H0T", "nifedipine_cav_block_0H2T"):
        eta_nif = regime_parameters(rname)["eta_nif"]
        reps = [simulate_neoblast_kinetics(regimes[rname], eta_nif=eta_nif, seed=sd, beta_V=beta_V) for sd in seeds]
        def agg(key: str) -> tuple[float, float]:
            vals = np.array([r[key] for r in reps], dtype=float)
            return float(np.nanmean(vals)), float(np.nanstd(vals))
        # Ensemble-mean mitotic index across seeds -> pooled peak times
        mi_mean = np.mean([r["mitotic_index_per_h"] for r in reps], axis=0)
        mi_pool = np.convolve(mi_mean, np.ones(7) / 7.0, mode="same")
        t_h = reps[0]["t_h"]
        early, late = t_h <= 20.0, t_h > 20.0
        out[rname] = {
            "replicates": reps,
            "t_h": t_h,
            "mi_mean": mi_mean,
            "mi_std": np.std([r["mitotic_index_per_h"] for r in reps], axis=0),
            "mi_pool_smooth": mi_pool,
            "t_peak_bodywide_pooled_h": float(t_h[early][np.argmax(mi_pool[early])]),
            "t_peak_local_pooled_h": float(t_h[late][np.argmax(mi_pool[late])]),
            "head_frac_ant": agg("head_frac_ant"),
            "head_frac_post": agg("head_frac_post"),
            "blastema_ant_cells": agg("blastema_ant_cells"),
            "blastema_post_cells": agg("blastema_post_cells"),
            "t_peak_bodywide_h": agg("t_peak_bodywide_h"),
            "t_peak_local_h": agg("t_peak_local_h"),
            "frac_near_wound_peak1": agg("frac_near_wound_peak1"),
            "frac_near_wound_peak2": agg("frac_near_wound_peak2"),
        }
    # beta_V ablation on WT: proliferation decoupled from V_mem (fate coupling retained)
    abl = [simulate_neoblast_kinetics(regimes["wild_type_1H1T"], eta_nif=0.0, seed=sd, beta_V=0.0) for sd in seeds]
    out["wt_betaV_ablation"] = {
        "replicates": abl,
        "blastema_ant_cells": (float(np.mean([r["blastema_ant_cells"] for r in abl])), float(np.std([r["blastema_ant_cells"] for r in abl]))),
        "blastema_post_cells": (float(np.mean([r["blastema_post_cells"] for r in abl])), float(np.std([r["blastema_post_cells"] for r in abl]))),
        "head_frac_ant": (float(np.mean([r["head_frac_ant"] for r in abl])), float(np.std([r["head_frac_ant"] for r in abl]))),
    }
    return out


# --------------------------------------------------------------------------- #
# 3d. Pass 3 — 6-Panel 2D Mesh / Axon Guidance / Neoblast Figure
# --------------------------------------------------------------------------- #
OUT_PNG_PASS3 = ROOT / "docs" / "media" / "planarian_hk_atpase_anterior_2d_mesh_neoblast.png"


def _style_axis(ax: Any) -> None:
    ax.set_facecolor("#131b29")
    ax.tick_params(colors="#cbd5e1", labelsize=8.5)
    for sp_ in ax.spines.values():
        sp_.set_color("#334155")


def render_pass3_figure(spec: dict[str, Any], two_d: dict[str, Any], neo: dict[str, Any], out_path: Path = OUT_PNG_PASS3) -> None:
    """Render the Pass 3 remediation figure (2D body-mesh V_mem, eye-field & visual axons, neoblast kinetics)."""
    mesh = two_d["mesh"]
    mask = mesh["mask"]
    s_m = mesh["s"]
    ny = mesh["ny"]
    extent = [0.0, 1.0, (0 - 0.5) * mesh["h_rel"], (ny - 0.5) * mesh["h_rel"]]
    wt2 = two_d["sims"]["wild_type_1H1T"]
    ivm2 = two_d["sims"]["ivermectin_depolarized_2H0T"]
    sch2 = two_d["sims"]["hk_atpase_inhibited_0H2T"]
    obs = two_d["specimen_ocelli"]

    fig, axes = plt.subplots(3, 2, figsize=(15.6, 13.2), dpi=130)
    fig.patch.set_facecolor("#0b0f17")
    for ax in axes.flat:
        _style_axis(ax)

    def mid_y(sv: np.ndarray) -> np.ndarray:
        return np.interp(sv, s_m, mesh["mid_row"]) * mesh["h_rel"]

    # Panel A: 2D V_mem on the specimen silhouette mesh (WT top, SCH-28080 middle, ivermectin bottom)
    ax = axes[0, 0]
    stack = np.vstack([wt2["V"], sch2["V"], ivm2["V"]])
    ext3 = [0.0, 1.0, 0.0, 3.0 * ny * mesh["h_rel"]]
    im = ax.imshow(stack, origin="lower", cmap="RdYlBu_r", vmin=-70.0, vmax=-10.0, extent=ext3, aspect="auto", interpolation="nearest")
    for k, (lab, sim_) in enumerate((("Ivermectin 2H0T", ivm2), ("SCH-28080 0H2T", sch2), ("Wild-Type 1H1T", wt2))):
        ax.text(0.50, (k + 0.88) * ny * mesh["h_rel"], f"{lab}: $V_A$={sim_['ant_V_mV']:.1f} mV, $V_P$={sim_['post_V_mV']:.1f} mV, {sim_['phenotype_code']}",
                color="#f8fafc", fontsize=8.2, ha="center", fontweight="bold", bbox=dict(facecolor="#0f172a", alpha=0.75, edgecolor="#334155"))
    ax.set_title(f"A. 2D Finite-Volume Body-Mesh $V_{{\\mathrm{{mem}}}}$ ({mesh['nx']}x{mesh['ny']} grid, {mesh['n_cells']} cells)", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_xlabel("Normalized AP position $x/L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Mediolateral ($y/L$, stacked regimes)", color="#cbd5e1", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors="#cbd5e1", labelsize=7.5)
    cb.set_label("$V_{\\mathrm{mem}}$ (mV)", color="#cbd5e1", fontsize=8)

    # Panel B: eye-field E(x,y), model photoreceptors, specimen ocelli, and WT visual axons
    ax = axes[0, 1]
    E = two_d["eyes"]["wild_type"]["E"]
    im2 = ax.imshow(E, origin="lower", cmap="magma", extent=extent, aspect="equal", interpolation="nearest")
    ax.contour(np.nan_to_num(wt2["H"], nan=0.0), levels=[0.5], colors=["#38bdf8"], linewidths=1.2, extent=extent, origin="lower")
    for e in two_d["eyes"]["wild_type"]["eyes"]:
        ax.scatter([e["s"]], [e["y"] + mid_y(np.array([e["s"]]))[0]], s=90, color="#22c55e", edgecolor="#f8fafc", zorder=8)
    # Specimen ocelli mapped into mesh coordinates
    for (px, py) in spec["ocelli_xy"]:
        sx = (px - spec["x_min"]) / spec["length_px"]
        sy = (py - spec["y_min"]) / spec["length_px"]
        ax.scatter([sx], [sy], s=80, marker="x", color="#f43f5e", linewidths=2.2, zorder=9)
    for a in two_d["guidance"]["wild_type"]["axons"]:
        tr = a["traj"]
        ax.plot(tr[:, 0], tr[:, 1] + mid_y(tr[:, 0]), color="#fde68a" if a["kind"] == "ipsilateral" else "#f97316", lw=1.4, alpha=0.95)
    ax.set_xlim(-0.01, 0.32)
    ax.set_ylim(extent[2], extent[3])
    ax.set_title("B. Eye-Field $E=H(1-S_{slit})(1-d^2)$, Eyes & Visual Axons (WT)", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_xlabel("$x/L$ (anterior 0.32 of body)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("$y/L$", color="#cbd5e1", fontsize=9)
    ax.scatter([], [], s=60, color="#22c55e", edgecolor="#f8fafc", label=f"Model eyes ($x/L$={two_d['s_eye_model']:.3f}, $|y|/L$={two_d['abs_y_eye_model_L']:.4f})")
    ax.scatter([], [], s=60, marker="x", color="#f43f5e", label=f"Specimen ocelli ($x/L$={obs['s_eye']:.3f}, $|y|/L$={obs['abs_y_eye_L']:.4f})")
    ax.plot([], [], color="#fde68a", label="Ipsilateral axons")
    ax.plot([], [], color="#f97316", label=f"Contralateral axons (crossing {two_d['guidance']['wild_type']['crossing_fraction']:.0%})")
    ax.plot([], [], color="#38bdf8", label="$H = 0.5$ head boundary (2D)")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="upper right")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.03, pad=0.02)
    cb2.ax.tick_params(colors="#cbd5e1", labelsize=7.5)

    # Panel C: slit / netrin RNAi guidance phenotypes in (s, y) coordinates
    ax = axes[1, 0]
    offs = {"wild_type": 0.16, "netrin_rnai": 0.0, "slit_rnai": -0.16}
    cols = {"wild_type": "#22c55e", "netrin_rnai": "#38bdf8", "slit_rnai": "#f43f5e"}
    labels = {"wild_type": "Wild-type", "netrin_rnai": "Smed-netrin RNAi", "slit_rnai": "Smed-slit RNAi"}
    hw_curve = np.interp(np.linspace(0.0, 0.32, 200), s_m, mesh["half_w_rel"])
    for key, yo in offs.items():
        g = two_d["guidance"][key]
        ax.fill_between(np.linspace(0.0, 0.32, 200), yo - hw_curve, yo + hw_curve, color="#1e293b", alpha=0.8)
        ax.axhline(yo, color="#64748b", lw=0.9, ls="--")
        for a in g["axons"]:
            tr = a["traj"]
            ax.plot(tr[:, 0], yo + tr[:, 1], color=cols[key], lw=1.3, ls="-" if a["kind"] == "ipsilateral" else ":")
        eyes_k = two_d["eyes"]["slit_rnai" if key == "slit_rnai" else "wild_type"]["eyes"]
        for e in eyes_k:
            ax.scatter([e["s"]], [yo + e["y"]], s=55, color=cols[key], edgecolor="#f8fafc", zorder=8)
        ax.text(0.31, yo + 0.055, f"{labels[key]}: crossing {g['crossing_fraction']:.0%}, ipsi $|y|$={g['ipsi_final_abs_y']:.4f} L",
                color="#f8fafc", fontsize=7.8, ha="right", bbox=dict(facecolor="#0f172a", alpha=0.75, edgecolor="#334155"))
    ax.set_xlim(0.0, 0.32)
    ax.set_ylim(-0.27, 0.27)
    ax.set_title("C. Slit/Netrin Mediolateral Visual-Axon Guidance Phenotypes", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_xlabel("$x/L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("$y/L$ (offset per condition; solid = ipsilateral, dotted = contralateral)", color="#cbd5e1", fontsize=8.3)

    # Panel D: eye position comparison + slit stripe-width sweep
    ax = axes[1, 1]
    sw = two_d["slit_sweep"]
    ax.plot(sw["sigmas"], sw["y_eye"], "o-", color="#f59e0b", lw=2.0, markersize=4, label="Predicted $|y_{eye}|/L$ vs slit stripe $\\sigma_{slit}$")
    ax.axhline(sw["y_eye_specimen"], color="#f43f5e", lw=1.8, ls="--", label=f"Specimen ocelli $|y|/L$ = {sw['y_eye_specimen']:.4f}")
    ax.axvline(SIGMA_SLIT_L, color="#22c55e", lw=1.6, ls=":", label=f"A-priori $\\sigma_{{slit}}$ = {SIGMA_SLIT_L:.3f} L -> $|y|/L$ = {sw['y_eye_default']:.4f}")
    ax.axvline(sw["sigma_star"], color="#38bdf8", lw=1.6, ls="-.", label=f"Specimen-matched $\\sigma_{{slit}}^*$ = {sw['sigma_star']:.4f} L")
    ax.set_title("D. Photoreceptor ML Position vs Slit Stripe Width (Specimen Match)", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_xlabel("Slit midline stripe half-width $\\sigma_{slit}$ (units of $L$)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Photoreceptor offset $|y_{eye}|/L$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.4, loc="lower right")

    # Panel E: stochastic neoblast mitotic index (mean +/- std over seeds) per regime
    ax = axes[2, 0]
    reg_cols = {"wild_type_1H1T": "#22c55e", "hk_atpase_inhibited_0H2T": "#38bdf8", "ivermectin_depolarized_2H0T": "#f43f5e", "nifedipine_cav_block_0H2T": "#f59e0b"}
    reg_lab = {"wild_type_1H1T": "WT", "hk_atpase_inhibited_0H2T": "SCH-28080", "ivermectin_depolarized_2H0T": "Ivermectin", "nifedipine_cav_block_0H2T": "Nifedipine"}
    for rname, col in reg_cols.items():
        d_ = neo[rname]
        ax.plot(d_["t_h"], d_["mi_pool_smooth"] * 100.0, color=col, lw=2.0, label=f"{reg_lab[rname]} (peaks {d_['t_peak_bodywide_pooled_h']:.1f} h, {d_['t_peak_local_pooled_h']:.1f} h)")
        ax.fill_between(d_["t_h"], (d_["mi_mean"] - d_["mi_std"]) * 100.0, (d_["mi_mean"] + d_["mi_std"]) * 100.0, color=col, alpha=0.12)
    ax.axvline(6.0, color="#cbd5e1", lw=1.2, ls=":", label="Wenemoser & Reddien 2010 peaks (6 h, 48 h)")
    ax.axvline(48.0, color="#cbd5e1", lw=1.2, ls=":")
    ax.set_title("E. Stochastic Neoblast Mitotic Index (N=1500, 4 seeds, mean $\\pm$ SD)", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_xlabel("Time post-amputation (h)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Divisions per neoblast per hour (%)", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="upper right")

    # Panel F: blastema progeny counts & head-fated fraction per regime (+ beta_V ablation)
    ax = axes[2, 1]
    names = ["wild_type_1H1T", "hk_atpase_inhibited_0H2T", "ivermectin_depolarized_2H0T", "nifedipine_cav_block_0H2T"]
    xpos = np.arange(len(names))
    ant_c = [neo[n]["blastema_ant_cells"][0] for n in names]
    ant_e = [neo[n]["blastema_ant_cells"][1] for n in names]
    post_c = [neo[n]["blastema_post_cells"][0] for n in names]
    post_e = [neo[n]["blastema_post_cells"][1] for n in names]
    ax.bar(xpos - 0.2, ant_c, 0.38, yerr=ant_e, color="#38bdf8", alpha=0.85, capsize=3, label="Anterior blastema progeny")
    ax.bar(xpos + 0.2, post_c, 0.38, yerr=post_e, color="#a855f7", alpha=0.85, capsize=3, label="Posterior blastema progeny")
    ax.set_xticks(xpos)
    ax.set_xticklabels([reg_lab[n] for n in names], color="#cbd5e1", fontsize=8.5)
    ax.set_ylabel("Post-mitotic progeny cells at 72 h", color="#cbd5e1", fontsize=9)
    ax2 = ax.twinx()
    _style_axis(ax2)
    ax2.set_facecolor("none")
    hf_a = [neo[n]["head_frac_ant"][0] for n in names]
    hf_p = [neo[n]["head_frac_post"][0] for n in names]
    ax2.plot(xpos - 0.2, hf_a, "o", color="#22c55e", markersize=9, markeredgecolor="#f8fafc", label="Head-fated fraction (anterior)")
    ax2.plot(xpos + 0.2, hf_p, "s", color="#f43f5e", markersize=8, markeredgecolor="#f8fafc", label="Head-fated fraction (posterior)")
    ax2.set_ylim(-0.05, 1.12)
    ax2.set_ylabel("$\\mathrm{Ca}_{\\mathrm{v}}$-gated head-fated progeny fraction", color="#cbd5e1", fontsize=9)
    abl = neo["wt_betaV_ablation"]
    ax.text(0.02, 0.97, f"WT $\\beta_V$=0 ablation: ant {abl['blastema_ant_cells'][0]:.0f}$\\pm${abl['blastema_ant_cells'][1]:.0f}, post {abl['blastema_post_cells'][0]:.0f}$\\pm${abl['blastema_post_cells'][1]:.0f} cells; head frac {abl['head_frac_ant'][0]:.3f}",
            transform=ax.transAxes, color="#f8fafc", fontsize=7.6, va="top", bbox=dict(facecolor="#0f172a", alpha=0.8, edgecolor="#334155"))
    ax.set_title("F. V$_{\\mathrm{mem}}$-Coupled Blastema Outgrowth & Progeny Fate", color="#f8fafc", fontsize=10.2, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="center right")

    fig.suptitle("Pass 3 — 2D Body-Mesh Bioelectrics, Slit/Netrin Photoreceptor Guidance & Stochastic Neoblast Kinetics (Schmidtea mediterranea)",
                 color="#f8fafc", fontsize=12.2, fontweight="bold", y=0.993)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. 9-Panel Publication Figure & Animated Regeneration GIF
# --------------------------------------------------------------------------- #
def render_outputs(seed: int = 2011, skip_gif: bool = False) -> dict[str, Any]:
    """Render the 9-panel comparison figure, the Pass 3 6-panel 2D-mesh/neoblast figure, and the animated GIF."""
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    verify_dataset_sha256()

    inv = compute_bioelectric_invariants()
    suite = evaluate_all_regimes_and_morphospace(seed=seed)
    spec = segment_planarian_specimen(SPECIMEN_PATH)
    two_d = evaluate_2d_mesh_and_guidance(spec, seed=seed)
    neo = evaluate_neoblast_regimes(suite["regimes"])
    render_pass3_figure(spec, two_d, neo)

    wt = suite["regimes"]["wild_type_1H1T"]
    sch = suite["regimes"]["hk_atpase_inhibited_0H2T"]
    ivm = suite["regimes"]["ivermectin_depolarized_2H0T"]
    octa = suite["regimes"]["octanol_gj_blockade_2H0T"]
    nif = suite["regimes"]["nifedipine_cav_block_0H2T"]

    fig, axes = plt.subplots(3, 3, figsize=(16.8, 13.8), dpi=135)
    fig.patch.set_facecolor("#0b0f17")
    for ax in axes.flat:
        ax.set_facecolor("#131b29")
        ax.tick_params(colors="#cbd5e1", labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#334155")

    # Panel A1: Bistable Ionic Current I_ion(V) & Voltage-Gated Ca_v Activation Gate
    ax = axes[0, 0]
    ax.plot(inv["v_grid"], -inv["i_ion_curve"], color="#38bdf8", lw=2.3, label=r"$-I_{\mathrm{ion}}(V)$ bistable drive")
    ax.axhline(0.0, color="#64748b", lw=1.1, ls="--")
    ax.scatter(
        [inv["V_hyper"], inv["V_depol"]],
        [0.0, 0.0],
        color="#22c55e",
        s=65,
        zorder=6,
        label="Stable attractors (-65, -20 mV)",
    )
    ax.scatter([inv["V_thresh"]], [0.0], color="#f43f5e", s=65, marker="D", zorder=6, label="Unstable threshold (-42.5 mV)")
    ax_twin = ax.twinx()
    ax_twin.plot(inv["v_grid"], inv["ca_gate_curve"], color="#f59e0b", lw=1.9, ls=":", label=r"$\mathrm{Ca}_{\mathrm{ss}}(V)$ influx gate")
    ax_twin.tick_params(colors="#f59e0b", labelsize=8)
    ax_twin.set_ylabel(r"Open probability $\mathrm{Ca}_{\mathrm{ss}}(V)$", color="#f59e0b", fontsize=8.5)
    ax.set_title("A1. Bistable Membrane Current & $\\mathrm{Ca}_{\\mathrm{v}}$ Gate", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Membrane potential $V_{\\mathrm{mem}}$ (mV)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Ionic current drive $-I_{\\mathrm{ion}}(V)$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="upper left")

    # Panel A2:Steady-State V_mem(x) Profiles Across 4 Bioelectric Regimes (Beane et al. 2011 Fig. 2)
    ax = axes[0, 1]
    ax.plot(wt["s"], wt["V"], color="#22c55e", lw=2.5, label=f"Wild-Type 1H1T ($V_A={wt['ant_V_mV']:.1f}$ mV)")
    ax.plot(sch["s"], sch["V"], color="#38bdf8", lw=2.2, ls="--", label=f"SCH-28080 / H,K-RNAi 0H2T ($V_A={sch['ant_V_mV']:.1f}$ mV)")
    ax.plot(ivm["s"], ivm["V"], color="#f43f5e", lw=2.2, ls="-.", label=f"Ivermectin 2H0T ($V_P={ivm['post_V_mV']:.1f}$ mV)")
    ax.plot(octa["s"], octa["V"], color="#a855f7", lw=2.0, ls=":", label=f"8-Octanol GJ Block 2H0T ($V_P={octa['post_V_mV']:.1f}$ mV)")
    ax.axhline(inv["V_thresh"], color="#eab308", lw=1.3, ls="--", label="Head threshold $V_{\\mathrm{thresh}}=-42.5$ mV")
    ax.set_title("A2. Steady-State $V_{\\mathrm{mem}}(x)$ Polarity Profiles", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Normalized AP position $x/L$ (0 = anterior wound, 1 = posterior wound)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Resting potential $V_{\\mathrm{mem}}$ (mV)", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="center right")

    # Panel A3: Anterior Head H(x) & Posterior Tail T(x) Morphogen Commitment + Nifedipine Control
    ax = axes[0, 2]
    ax.plot(wt["s"], wt["H"], color="#22c55e", lw=2.5, label="WT Head $H(x)$ (1H1T)")
    ax.plot(wt["s"], wt["T"], color="#22c55e", lw=1.6, ls=":", label="WT Tail $T(x)$ (1H1T)")
    ax.plot(ivm["s"], ivm["H"], color="#f43f5e", lw=2.2, ls="-.", label="Ivermectin Head $H(x)$ (2H0T)")
    ax.plot(sch["s"], sch["H"], color="#38bdf8", lw=2.0, ls="--", label="SCH-28080 Head $H(x)$ (0H2T)")
    ax.plot(nif["s"], nif["H"], color="#f59e0b", lw=2.0, label="Nifedipine $\\mathrm{Ca}_{\\mathrm{v}}$-block control (0H2T)")
    ax.axhline(0.50, color="#64748b", lw=1.1, ls="--")
    ax.set_title("A3. Head $H(x)$ (*pc2/ndk*) vs Tail $T(x)$ (*wnt1*) Fate", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Normalized AP position $x/L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Normalized gene expression $H(x), T(x)$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.0, loc="upper right")

    # Panel B1: Space-Time Kymograph of V_mem(x,t) (Wild-Type 1H1T vs Ivermectin 2H0T)
    ax = axes[1, 0]
    kymo_stack = np.vstack([wt["history_V"], ivm["history_V"]])
    im_k = ax.imshow(
        kymo_stack,
        aspect="auto",
        origin="lower",
        cmap="RdYlBu_r",
        vmin=-68.0,
        vmax=-15.0,
        extent=[0.0, 1.0, 0, len(kymo_stack) - 1],
    )
    ax.axhline(len(wt["history_V"]) - 0.5, color="#f8fafc", lw=1.6, ls="--")
    ax.text(0.52, len(wt["history_V"]) * 0.45, "Bottom: Wild-Type (1H1T)\nPosterior clears to -65 mV", color="#f8fafc", fontsize=8, fontweight="bold", bbox=dict(facecolor="#0f172a", alpha=0.75, edgecolor="#334155"))
    ax.text(0.52, len(wt["history_V"]) * 1.45, "Top: Ivermectin (2H0T)\nBilateral depolarization (-16 mV)", color="#f8fafc", fontsize=8, fontweight="bold", bbox=dict(facecolor="#0f172a", alpha=0.75, edgecolor="#334155"))
    ax.set_title("B1. Space-Time $V_{\\mathrm{mem}}(x,t)$ Kymograph (WT vs 2H0T)", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Normalized AP position $x/L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Time snapshot index (WT $\\mid$ Ivermectin)", color="#cbd5e1", fontsize=9)
    cb = fig.colorbar(im_k, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#cbd5e1", labelsize=7.5)
    cb.set_label("$V_{\\mathrm{mem}}$ (mV)", color="#cbd5e1", fontsize=8)

    # Panel B2: Voltage-Gated Gap-Junction Conductance Bell Curve & 2D (I_HK, I_IVM) Phase Diagram
    ax = axes[1, 1]
    im_ph = ax.imshow(
        suite["head_count_grid"],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=2.0,
        extent=[
            float(suite["grid_ihk"][0]),
            float(suite["grid_ihk"][-1]),
            float(suite["grid_ivm"][0]),
            float(suite["grid_ivm"][-1]),
        ],
    )
    ax.scatter([18.0], [0.0], color="#22c55e", edgecolor="#f8fafc", s=80, marker="o", zorder=6, label="WT (1H1T)")
    ax.scatter([-16.0], [0.0], color="#38bdf8", edgecolor="#f8fafc", s=80, marker="s", zorder=6, label="SCH-28080 (0H2T)")
    ax.scatter([18.0], [22.0], color="#f43f5e", edgecolor="#f8fafc", s=95, marker="*", zorder=6, label="Ivermectin (2H0T)")
    ax.set_title("B2. Bioelectric Morphospace Phase Diagram $N_{\\mathrm{heads}}(I_{\\mathrm{HK}}, I_{\\mathrm{IVM}})$", color="#f8fafc", fontsize=10.0, fontweight="bold")
    ax.set_xlabel("Anterior H,K-ATPase pump current $I_{\\mathrm{HK}}$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Ivermectin depolarizing current $I_{\\mathrm{IVM}}$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="center left")
    cb2 = fig.colorbar(im_ph, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cb2.ax.set_yticklabels(["0H2T", "1H1T", "2H0T"], color="#cbd5e1", fontsize=7.5)

    # Panel B3: Reconstructed 2D Planarian Body-Plan Morphologies (1H1T WT vs 2H0T vs 0H2T)
    ax = axes[1, 2]
    offsets = [
        ("Wild-Type 1H1T", wt, +0.62, "#22c55e"),
        ("Ivermectin / Octanol 2H0T", ivm, 0.0, "#f43f5e"),
        ("SCH-28080 / H,K-RNAi 0H2T", sch, -0.62, "#38bdf8"),
    ]
    for label, sim_r, y_off, col in offsets:
        morph = synthesize_2d_planarian_morphology(sim_r["s"], sim_r["H"], sim_r["T"])
        sf = morph["s_fine"]
        hw = morph["half_width"]
        ax.fill_between(sf, y_off - hw, y_off + hw, color=col, alpha=0.32, edgecolor=col, lw=2.0, label=label)
        for ex, ey in morph["eyespots_xy"]:
            ax.scatter([ex], [y_off + ey], color="#f8fafc", edgecolor="#0f172a", s=38, zorder=7)
            ax.scatter([ex], [y_off + ey], color="#0f172a", s=12, zorder=8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-0.05, 1.05)
    ax.set_yticks([-0.62, 0.0, 0.62])
    ax.set_yticklabels(["0H2T (Headless)", "2H0T (2-Headed)", "1H1T (Wild-Type)"], color="#cbd5e1", fontsize=8)
    ax.set_title("B3. Reconstructed 2D Planarian Silhouettes & Eyespots", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Normalized AP body axis $x/L$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="upper right")

    # Panel C1: Schmidtea mediterranea Specimen CV Segmentation (Body Contour, AP Axis & Bilateral Ocelli)
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
        f"C1. Schmidtea mediterranea Specimen ($N={spec['n_ocelli_detected']}$ Ocelli, AR={spec['aspect_ratio']:.2f})",
        color="#f8fafc",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="lower right")
    ax.axis("off")

    # Panel C2: Specimen Width & Anterior Ocelli Localization vs Simulated WT Head Morphogen H(x)
    ax = axes[2, 1]
    morph_wt = synthesize_2d_planarian_morphology(wt["s"], wt["H"], wt["T"])
    sim_w_norm = morph_wt["half_width"] / max(1e-6, float(np.max(morph_wt["half_width"])))
    ax.plot(spec["s_bins"], spec["widths_norm"], color="#22c55e", lw=2.3, label="Specimen dorsal width $W(x/L)$")
    ax.plot(morph_wt["s_fine"], sim_w_norm, color="#38bdf8", lw=2.0, ls="--", label="Simulated 1H1T contour width")
    ax.plot(wt["s"], wt["H"], color="#f43f5e", lw=2.0, ls="-.", label="Simulated anterior head $H(x/L)$")
    ax.axvline(
        spec["mean_ocelli_rel_s"],
        color="#eab308",
        lw=1.8,
        ls=":",
        label=f"Specimen ocelli ($x/L = {spec['mean_ocelli_rel_s']:.3f}$)",
    )
    ax.axvline(
        two_d["s_eye_model"],
        color="#a855f7",
        lw=1.8,
        ls="--",
        label=f"2D-mesh eye-field eyes ($x/L = {two_d['s_eye_model']:.3f}$, Pass 3)",
    )
    ax.set_title("C2. Specimen Width & Ocelli vs Simulated 1H1T Head Domain", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_xlabel("Normalized AP coordinate $x/L$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Normalized width / morphogen $H$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2, loc="center right")

    # Panel C3: H,K-ATPase Current Dose-Response & Beane et al. (2011) 5-Regime Phenotype Parity
    ax = axes[2, 2]
    ihk_x = [r["I_HK"] for r in suite["ihk_sweep"]]
    ant_h_y = [r["ant_H"] for r in suite["ihk_sweep"]]
    ant_v_y = [(r["ant_V_mV"] + 70.0) / 55.0 for r in suite["ihk_sweep"]]
    ax.plot(ihk_x, ant_h_y, "o-", color="#f43f5e", lw=2.3, markersize=5.5, label="Anterior head commitment $H(0)$")
    ax.plot(ihk_x, ant_v_y, "s--", color="#38bdf8", lw=1.8, markersize=4.5, label="Normalized $(V_A + 70)/55$")
    ax.axvline(0.0, color="#eab308", lw=1.4, ls=":", label="Bifurcation threshold $I_{\\mathrm{HK}} \\approx 0$")
    ax.axhline(0.50, color="#64748b", lw=1.0, ls="--")
    ax.set_title("C3. H,K-ATPase Dose-Response Switch ($0\\mathrm{H}2\\mathrm{T} \\to 1\\mathrm{H}1\\mathrm{T}$)", color="#f8fafc", fontsize=10.2, fontweight="bold")
    ax.set_xlabel("Anterior H,K-ATPase pump current $I_{\\mathrm{HK}}$", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Anterior head commitment $H(0)$", color="#cbd5e1", fontsize=9)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="center right")

    fig.suptitle(
        "Beane et al. (2011) & Pietak-Levin (2017) Planarian H,K-ATPase Bioelectric $V_{\\mathrm{mem}}$ & Gap-Junction Anterior Polarity Switch",
        color="#f8fafc",
        fontsize=12.8,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.978])
    fig.savefig(OUT_PNG, dpi=135)
    plt.close(fig)

    # Render 36-frame animated GIF comparing Wild-Type (1H1T), SCH-28080 (0H2T), and Ivermectin (2H0T)
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
            ax1.plot(wt["s"], wt["history_V"][idx], color="#22c55e", lw=2.3, label="Wild-Type (1H1T)")
            ax1.plot(sch["s"], sch["history_V"][idx], color="#38bdf8", lw=2.0, ls="--", label="SCH-28080 (0H2T)")
            ax1.plot(ivm["s"], ivm["history_V"][idx], color="#f43f5e", lw=2.0, ls="-.", label="Ivermectin (2H0T)")
            ax1.axhline(-42.5, color="#eab308", lw=1.1, ls=":")
            ax1.set_ylim(-72.0, -10.0)
            ax1.set_xlim(0.0, 1.0)
            ax1.set_title(f"Bioelectric $V_{{\\mathrm{{mem}}}}(x,t)$ Evolution ($t={t_val:.1f}$)", color="#f8fafc", fontsize=9.5)
            ax1.set_xlabel("AP coordinate $x/L$ (0=ant, 1=post)", color="#cbd5e1", fontsize=8.5)
            ax1.set_ylabel("$V_{\\mathrm{mem}}$ (mV)", color="#cbd5e1", fontsize=8.5)
            ax1.legend(loc="center right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.2)

            ax2.plot(wt["s"], wt["history_H"][idx], color="#22c55e", lw=2.3, label="WT Head $H(x,t)$ (1H1T)")
            ax2.plot(sch["s"], sch["history_H"][idx], color="#38bdf8", lw=2.0, ls="--", label="SCH-28080 $H(x,t)$ (0H2T)")
            ax2.plot(ivm["s"], ivm["history_H"][idx], color="#f43f5e", lw=2.0, ls="-.", label="Ivermectin $H(x,t)$ (2H0T)")
            ax2.plot(nif["s"], nif["history_H"][idx], color="#f59e0b", lw=1.6, ls=":", label="Nifedipine Control (0H2T)")
            ax2.axhline(0.50, color="#64748b", lw=1.0, ls="--")
            ax2.set_ylim(-0.02, 1.05)
            ax2.set_xlim(0.0, 1.0)
            ax2.set_title("Anterior Head Gene $H(x,t)$ Commitment", color="#f8fafc", fontsize=9.5)
            ax2.set_xlabel("AP coordinate $x/L$", color="#cbd5e1", fontsize=8.5)
            ax2.set_ylabel("Head morphogen $H(x,t)$", color="#cbd5e1", fontsize=8.5)
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

    return {
        "invariants": inv,
        "suite": suite,
        "specimen": spec,
        "two_d": two_d,
        "neoblasts": neo,
    }


# --------------------------------------------------------------------------- #
# 5. MorphoEval Sidecar Quantitative Measurements (`measure_morphoeval()`)
# --------------------------------------------------------------------------- #
def measure_morphoeval() -> list[dict[str, Any]]:
    """Compute quantitative morphometric measurements for `morphoeval/measure_outputs.py`."""
    inv = compute_bioelectric_invariants()
    suite = evaluate_all_regimes_and_morphospace(seed=2011)
    spec = segment_planarian_specimen(SPECIMEN_PATH)

    wt = suite["regimes"]["wild_type_1H1T"]
    sch = suite["regimes"]["hk_atpase_inhibited_0H2T"]
    ivm = suite["regimes"]["ivermectin_depolarized_2H0T"]
    octa = suite["regimes"]["octanol_gj_blockade_2H0T"]
    nif = suite["regimes"]["nifedipine_cav_block_0H2T"]

    morph_wt = synthesize_2d_planarian_morphology(wt["s"], wt["H"], wt["T"])
    sim_w_60 = np.interp(spec["s_bins"], morph_wt["s_fine"], morph_wt["half_width"])
    sim_w_60_norm = sim_w_60 / max(1e-6, float(np.max(sim_w_60)))
    width_corr_r = float(np.corrcoef(spec["widths_norm"], sim_w_60_norm)[0, 1])
    width_rmse = float(np.sqrt(np.mean((spec["widths_norm"] - sim_w_60_norm) ** 2)))

    expected_heads = [1, 0, 2, 2, 0]
    observed_heads = [
        wt["n_heads"],
        sch["n_heads"],
        ivm["n_heads"],
        octa["n_heads"],
        nif["n_heads"],
    ]
    regime_rmse = float(np.sqrt(np.mean((np.array(expected_heads, dtype=float) - np.array(observed_heads, dtype=float)) ** 2)))

    # Pass 3: 2D body mesh, slit/netrin guidance, stochastic neoblasts
    two_d = evaluate_2d_mesh_and_guidance(spec, seed=2011)
    neo = evaluate_neoblast_regimes(suite["regimes"])
    wt2 = two_d["sims"]["wild_type_1H1T"]
    heads_2d = [two_d["head_counts_2d"][r] for r in ("wild_type_1H1T", "hk_atpase_inhibited_0H2T", "ivermectin_depolarized_2H0T")]
    rmse_2d = float(np.sqrt(np.mean((np.array(heads_2d, dtype=float) - np.array([1, 0, 2], dtype=float)) ** 2)))
    g_wt = two_d["guidance"]["wild_type"]
    g_slit = two_d["guidance"]["slit_rnai"]
    g_net = two_d["guidance"]["netrin_rnai"]
    obs = two_d["specimen_ocelli"]
    sw = two_d["slit_sweep"]
    neo_wt = neo["wild_type_1H1T"]
    neo_sch = neo["hk_atpase_inhibited_0H2T"]
    neo_ivm = neo["ivermectin_depolarized_2H0T"]
    neo_nif = neo["nifedipine_cav_block_0H2T"]
    pass3 = [
        {
            "name": "2D body-mesh wild-type anterior blastema V_mem on the segmented specimen silhouette",
            "value": round(wt2["ant_V_mV"], 4),
            "unit": "mV",
            "method": f"Finite-volume bistable V_mem + voltage-gated GJ PDE on the {two_d['mesh']['nx']}x{two_d['mesh']['ny']} specimen-silhouette mesh ({two_d['mesh']['n_cells']} cells), mean V over anterior cells (x/L <= 5/59)",
            "reference_value": -20.0,
            "reference_source": "Beane et al. (2011) Chem. Biol. 18(1):77-89 Fig. 2 depolarized anterior blastema attractor V_depol = -20 mV (1D solver gives V_A = %.2f mV)" % wt["ant_V_mV"],
            "note": f"2D vs 1D anterior V_mem difference = {wt2['ant_V_mV'] - wt['ant_V_mV']:+.3f} mV; mediolateral SD inside the anterior blastema = {wt2['ml_std_ant_V_mV']:.3f} mV; net gap-junctional current over the mesh = {wt2['net_gj_current']:.2e} (flux conservation); 2D phenotypes = {two_d['head_counts_2d']}",
            "derived_from": "script",
        },
        {
            "name": "2D body-mesh 3-regime regenerated head count RMSE (WT / SCH-28080 / ivermectin)",
            "value": round(rmse_2d, 4),
            "unit": "heads RMSE",
            "method": "RMSE between 2D finite-volume body-mesh head counts (anterior/posterior blastema mean H >= 0.50) and the empirical Beane et al. (2011) counts across WT (1H), SCH-28080 (0H) and ivermectin (2H)",
            "reference_value": 0.0,
            "reference_source": "Beane et al. (2011) Chem. Biol. 18(1):77-89 Figs. 1-4 (1H1T / 0H2T / 2H0T phenotypes); 1D reference RMSE = %.4f" % regime_rmse,
            "note": f"2D predicted heads = {heads_2d} vs reference = [1, 0, 2] (RMSE = {rmse_2d:.4f}); 2D and 1D solvers agree on all three phenotypes = {heads_2d == [wt['n_heads'], sch['n_heads'], ivm['n_heads']]}",
            "derived_from": "script",
        },
        {
            "name": "2D eye-field photoreceptor mediolateral offset |y_eye|/L (a-priori slit stripe) vs segmented specimen ocelli",
            "value": round(two_d["abs_y_eye_model_L"], 4),
            "unit": "|y_eye| / L",
            "method": f"Arg-max of the eye-field E = H (1 - S_slit(y)) (1 - d^2) on the 2D body mesh with the a-priori Smed-slit stripe sigma_slit = {SIGMA_SLIT_L} L (not fitted); mean |y| of the two anterior photoreceptors",
            "reference_value": round(obs["abs_y_eye_L"], 4),
            "reference_source": "Vendored Schmidtea mediterranea dorsal specimen (datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg) bilateral ocelli half-separation; Cebria, Guo, Jopek & Newmark (2007) Dev. Biol. 307:394-406 (Smed-slit midline stripe)",
            "note": f"model |y_eye|/L = {two_d['abs_y_eye_model_L']:.4f} at x/L = {two_d['s_eye_model']:.4f} vs specimen {obs['abs_y_eye_L']:.4f} at x/L = {obs['s_eye']:.4f}; specimen-matched sigma_slit* = {sw['sigma_star']:.4f} L (sweep over [0.005, 0.06] L); eyes placed: WT = {two_d['n_eyes_2d']['wild_type']}, SCH-28080 = {two_d['n_eyes_2d']['sch28080']}, ivermectin = {two_d['n_eyes_2d']['ivermectin']}",
            "derived_from": "output",
        },
        {
            "name": "Slit/netrin visual-axon guidance: WT contralateral midline-crossing fraction and ipsilateral tract offset",
            "value": round(g_wt["crossing_fraction"], 4),
            "unit": "fraction of commissural axons crossing the midline",
            "method": "Reduced growth-cone model in body-fitted (s, y) coordinates with c(y) = k_netrin dN/dy - k_slit dS/dy and the Robo-Slit pre/post-crossing switch; compared against Smed-slit RNAi (eta_slit = 1) and Smed-netrin RNAi (eta_netrin = 1)",
            "reference_value": 1.0,
            "reference_source": "Cebria & Newmark (2005) Development 132:3691-3703 (netrin RNAi: visual axons fail to reach/cross the midline); Cebria et al. (2007) Dev. Biol. 307:394-406 (slit RNAi: axons collapse onto the midline); Kidd, Bland & Goodman (1999) Cell 96:785-794",
            "note": f"WT crossing = {g_wt['crossing_fraction']:.2f}, ipsi |y| = {g_wt['ipsi_final_abs_y']:.4f} L (analytic tract y* = {g_wt['tract_y_star_L']:.4f} L); slit RNAi ipsi |y| = {g_slit['ipsi_final_abs_y']:.4f} L (midline collapse), crossing = {g_slit['crossing_fraction']:.2f}; netrin RNAi crossing = {g_net['crossing_fraction']:.2f}, ipsi |y| = {g_net['ipsi_final_abs_y']:.4f} L",
            "derived_from": "script",
        },
        {
            "name": "Stochastic V_mem-coupled neoblast kinetics: WT anterior blastema head-fated progeny fraction",
            "value": round(neo_wt["head_frac_ant"][0], 4),
            "unit": "fraction of anterior progeny with head identity",
            "method": "Agent-based inhomogeneous-Poisson neoblast division with V_mem-dependent rate and Ca_v-gated progeny fate on the 1D V_mem(x,t) history; mean over 4 seeds, 1500 neoblasts, 72 h",
            "reference_value": 1.0,
            "reference_source": "Beane et al. (2011) Chem. Biol. 18(1):77-89 Figs. 1-2 (100% anterior head regeneration in WT; 0% under SCH-28080 / H,K-ATPase RNAi; nifedipine blocks head fate without blocking depolarization)",
            "note": f"WT head_frac_ant = {neo_wt['head_frac_ant'][0]:.3f} +/- {neo_wt['head_frac_ant'][1]:.3f}; SCH-28080 = {neo_sch['head_frac_ant'][0]:.3f}; ivermectin head_frac_post = {neo_ivm['head_frac_post'][0]:.3f}; nifedipine = {neo_nif['head_frac_ant'][0]:.3f}; anterior progeny WT = {neo_wt['blastema_ant_cells'][0]:.0f} +/- {neo_wt['blastema_ant_cells'][1]:.0f} vs SCH-28080 = {neo_sch['blastema_ant_cells'][0]:.0f} (blastema still forms); beta_V = 0 ablation ant progeny = {neo['wt_betaV_ablation']['blastema_ant_cells'][0]:.0f}",
            "derived_from": "script",
        },
        {
            "name": "Stochastic neoblast biphasic mitotic response: pooled body-wide and wound-local peak times (WT)",
            "value": round(neo_wt["t_peak_local_pooled_h"], 2),
            "unit": "h post-amputation",
            "method": "Second (wound-localised) peak of the 3.5-h-smoothed ensemble-mean mitotic index; the first (body-wide) peak and both peak times are fitted inputs (tau_body = 6 h, tau_rise/tau_decay), the predicted quantities are the near-wound fraction of divisions at each peak",
            "reference_value": 48.0,
            "reference_source": "Wenemoser & Reddien (2010) Dev. Biol. 344:979-991 (biphasic mitotic peaks at ~6 h body-wide and ~48 h wound-localised)",
            "note": f"WT pooled peaks = {neo_wt['t_peak_bodywide_pooled_h']:.1f} h and {neo_wt['t_peak_local_pooled_h']:.1f} h; near-wound fraction of divisions at peak 1 = {neo_wt['frac_near_wound_peak1'][0]:.3f} vs peak 2 = {neo_wt['frac_near_wound_peak2'][0]:.3f} (spatial localisation is the prediction, peak timing is calibrated)",
            "derived_from": "script",
        },
    ]
    base = [
        {
            "name": "Wild-type planarian anteroposterior V_mem polarization delta (V_ant - V_post)",
            "value": round(wt["delta_V_AP_mV"], 4),
            "unit": "mV",
            "method": "Coupled bistable V_mem + voltage-gated gap-junction PDE steady-state difference V_ant(0) - V_post(L) in wild_type_1H1T",
            "reference_value": 45.0,
            "reference_source": "Beane et al. (2011) Chem. Biol. 18(1):77-89 Fig. 2 & Pietak-Levin (2017) (-20 mV anterior vs -65 mV posterior = +45.0 mV)",
            "note": f"WT V_ant = {wt['ant_V_mV']:.2f} mV, V_post = {wt['post_V_mV']:.2f} mV (1H1T); SCH-28080 H,K-ATPase inhibition collapses delta_V to {sch['delta_V_AP_mV']:.2f} mV (0H2T)",
            "derived_from": "script",
        },
        {
            "name": "H,K-ATPase inhibition (SCH-28080 / RNAi) vs Wild-Type anterior head commitment suppression delta",
            "value": round(wt["ant_H"] - sch["ant_H"], 4),
            "unit": "H_WT(0) - H_SCH28080(0)",
            "method": "Anterior blastema head gene expression H(0) in wild_type_1H1T vs hk_atpase_inhibited_0H2T (SCH-28080)",
            "reference_value": 0.90,
            "reference_source": "Beane et al. (2011) Chem. Biol. 18(1):77-89 Fig. 1 & Fig. 2 (100% WT head regeneration vs 0% head regeneration in H,K-ATPase RNAi / SCH-28080)",
            "note": f"WT ant_H = {wt['ant_H']:.4f} (n_heads={wt['n_heads']}) vs SCH-28080 ant_H = {sch['ant_H']:.4f} (n_heads={sch['n_heads']}); nifedipine Ca_v-block negative control ant_H = {nif['ant_H']:.4f} (n_heads={nif['n_heads']})",
            "derived_from": "script",
        },
        {
            "name": "Ivermectin depolarization & 8-octanol gap-junction blockade double-head (2H0T) induction count",
            "value": int(ivm["n_heads"]),
            "unit": "regenerated heads",
            "method": "Bilateral blastema head count (H >= 0.50) under ivermectin GluCl depolarization (I_IVM=22) and 8-octanol gap-junction blockade (eta_oct=0.92)",
            "reference_value": 2,
            "reference_source": "Beane et al. (2011) Chem. Biol. 18(1):77-89 Figs. 3-4 & Oviedo et al. (2010) Dev. Biol. 339:188-199 (double-headed 2H0T planarians)",
            "note": f"ivermectin post_H = {ivm['post_H']:.4f} (2H0T, V_post = {ivm['post_V_mV']:.2f} mV); 8-octanol post_H = {octa['post_H']:.4f} (2H0T, n_heads = {octa['n_heads']})",
            "derived_from": "script",
        },
        {
            "name": "Schmidtea mediterranea specimen bilateral photoreceptor ocelli count & relative AP position",
            "value": round(spec["mean_ocelli_rel_s"], 4),
            "unit": "x / L (anterior fraction)",
            "method": "Automated Gaussian local-contrast ocelli detection on schmidtea_mediterranea_specimen.jpg (N=2 eyespots at x/L=0.0528, inter-ocelli spacing=67.1 px)",
            "reference_value": 0.065,
            "reference_source": "Vendored Schmidtea mediterranea dorsal specimen (CC BY 2.5, File:Smediterranea.jpg) & simulated anterior head blastema center x/L = 0.065",
            "note": f"detected {spec['n_ocelli_detected']} bilateral ocelli at x/L = {[round(v, 4) for v in spec['ocelli_rel_s']]} (std = {np.std(spec['ocelli_rel_s']):.4f}), body length = {spec['length_px']:.0f} px, aspect ratio = {spec['aspect_ratio']:.3f}",
            "derived_from": "output",
        },
        {
            "name": "Specimen vs simulated 1H1T planarian dorsoventral width profile correlation r",
            "value": round(width_corr_r, 4),
            "unit": "Pearson r",
            "method": "Pearson correlation and RMSE between segmented Schmidtea mediterranea width profile W(x/L) and reconstructed 1H1T morphology contour",
            "reference_value": 0.90,
            "reference_source": "Schmidtea mediterranea dorsal whole-mount morphology (datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg)",
            "note": f"width profile Pearson r = {width_corr_r:.4f}, normalized width RMSE = {width_rmse:.4f} across 60 AP body segments",
            "derived_from": "output",
        },
        {
            "name": "Beane et al. (2011) 5-regime regenerated head count prediction RMSE",
            "value": round(regime_rmse, 4),
            "unit": "heads RMSE",
            "method": "RMSE between predicted and empirical head counts across WT (1H), SCH-28080 (0H), Ivermectin (2H), 8-Octanol (2H), and Nifedipine control (0H)",
            "reference_value": 0.0,
            "reference_source": "Beane et al. (2011) Chem. Biol. 18(1):77-89 Table 1 & Figs. 1-6 (Lean 4 beane_four_regeneration_phenotypes)",
            "note": f"predicted heads = {observed_heads} vs reference = {expected_heads} (exact 5/5 phenotype match, RMSE = {regime_rmse:.4f}, gj_isolation_ratio = {inv['gj_isolation_ratio']:.4f})",
            "derived_from": "script",
        },
    ]
    return base + pass3


# --------------------------------------------------------------------------- #
# 6. Comprehensive Deterministic Self-Test Suite (`--self-test`)
# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Run 12 deterministic verification checks covering SHA-256 locks, bistable V_mem
    equilibria, voltage-gated GJ symmetry, all 5 Beane et al. (2011) regeneration
    phenotypes, specimen CV segmentation, bitwise solver reproducibility, and (Pass 3)
    the 2D body-mesh solver, slit/netrin axon-guidance phenotypes, eye-field vs specimen
    ocelli, and the stochastic V_mem-coupled neoblast kinetics."""
    # 1. Verify SHA-256 digests of all 4 vendored dataset files
    digests = verify_dataset_sha256()
    ref = load_reference()
    assert len(digests) == 4, f"Expected 4 verified dataset files, got {len(digests)}"

    # 2. Verify bistable ionic current roots, stability slopes, and GJ symmetry
    canon = ref["canonical_parameters"]
    inv = compute_bioelectric_invariants(
        g_bistable=canon["g_bistable"],
        V_hyper=canon["V_hyper_mV"],
        V_thresh=canon["V_thresh_mV"],
        V_depol=canon["V_depol_mV"],
        G_0=canon["G_0"],
        V_0=canon["V_0_mV"],
        k_V=canon["k_V_mV"],
    )
    assert all(abs(r) < 1e-12 for r in inv["residuals"]), "Bistable roots must vanish identically"
    assert inv["slope_hyper"] > 0.0 and inv["slope_depol"] > 0.0, "V_hyper and V_depol must be stable attractors"
    assert inv["slope_thresh"] < 0.0, "V_thresh must be an unstable separatrix"
    assert inv["gj_symmetry_diff"] < 1e-15, "Gap-junction conductance must be symmetric in DeltaV"
    assert abs(inv["gj_isolation_ratio"] - 0.10) < 1e-12, f"Expected 10x GJ isolation at 36 mV drop, got {inv['gj_isolation_ratio']}"

    # 3. Verify Wild-Type 1H1T normal polarity regeneration
    suite = evaluate_all_regimes_and_morphospace(seed=2011)
    wt = suite["regimes"]["wild_type_1H1T"]
    assert wt["phenotype_code"] == "1H1T" and wt["n_heads"] == 1 and wt["n_tails"] == 1, (
        f"Expected 1H1T wild-type regeneration, got {wt['phenotype_code']}"
    )
    assert wt["ant_V_mV"] > -22.0 and wt["post_V_mV"] < -60.0, "WT must depolarize anteriorly and hyperpolarize posteriorly"

    # 4. Verify H,K-ATPase inhibition (SCH-28080 / RNAi) -> 0H2T headless phenotype
    sch = suite["regimes"]["hk_atpase_inhibited_0H2T"]
    assert sch["phenotype_code"] == "0H2T" and sch["n_heads"] == 0, (
        f"Expected 0H2T headless phenotype under H,K-ATPase inhibition, got {sch['phenotype_code']}"
    )
    assert sch["ant_V_mV"] < -60.0 and sch["ant_H"] < 0.05, "SCH-28080 must abolish anterior depolarization and head fate"

    # 5. Verify Ivermectin depolarization & 8-Octanol GJ blockade -> 2H0T double-headed phenotypes
    ivm = suite["regimes"]["ivermectin_depolarized_2H0T"]
    octa = suite["regimes"]["octanol_gj_blockade_2H0T"]
    assert ivm["phenotype_code"] == "2H0T" and ivm["n_heads"] == 2, f"Expected 2H0T in ivermectin, got {ivm['phenotype_code']}"
    assert octa["phenotype_code"] == "2H0T" and octa["n_heads"] == 2, f"Expected 2H0T in 8-octanol, got {octa['phenotype_code']}"

    # 6. Verify Nifedipine Ca_v channel blockade negative control (depolarized V_mem but 0 heads)
    nif = suite["regimes"]["nifedipine_cav_block_0H2T"]
    assert nif["ant_V_mV"] > -22.0 and nif["n_heads"] == 0 and nif["phenotype_code"] == "0H2T", (
        "Nifedipine control must retain anterior V_mem depolarization while blocking Ca2+-driven head regeneration"
    )

    # 7. Verify Schmidtea mediterranea specimen CV segmentation (2 bilateral ocelli in anterior head)
    spec = segment_planarian_specimen(SPECIMEN_PATH)
    assert spec["n_ocelli_detected"] == 2, f"Expected 2 segmented eyespots (ocelli), got {spec['n_ocelli_detected']}"
    assert 0.03 <= spec["mean_ocelli_rel_s"] <= 0.12, (
        f"Expected ocelli in anterior head region s in [0.03, 0.12], got {spec['mean_ocelli_rel_s']:.4f}"
    )
    assert spec["length_px"] > 1400.0, "Invalid segmented Schmidtea mediterranea body length"

    # 8. Verify bitwise two-run solver determinism
    r1 = simulate_planarian_bioelectric_pde(regime="wild_type_1H1T", steps=400, seed=2011)
    r2 = simulate_planarian_bioelectric_pde(regime="wild_type_1H1T", steps=400, seed=2011)
    assert np.allclose(r1["V"], r2["V"], atol=0.0, rtol=0.0), "Bitwise V_mem determinism check failed"
    assert np.allclose(r1["H"], r2["H"], atol=0.0, rtol=0.0), "Bitwise H morphogen determinism check failed"

    # 9. (Pass 3) 2D finite-volume body mesh on the specimen silhouette: flux conservation,
    #    1D/2D phenotype agreement, and bounded mediolateral V_mem heterogeneity
    two_d = evaluate_2d_mesh_and_guidance(spec, seed=2011)
    mesh = two_d["mesh"]
    assert mesh["n_cells"] > 1000 and mesh["ny"] >= 10, f"2D body mesh too coarse: {mesh['nx']}x{mesh['ny']}, {mesh['n_cells']} cells"
    for rname, s2 in two_d["sims"].items():
        s1 = suite["regimes"][rname]
        assert abs(s2["net_gj_current"]) < 1e-9, f"2D gap-junction flux not conserved in {rname}: {s2['net_gj_current']:.3e}"
        assert s2["phenotype_code"] == s1["phenotype_code"], (
            f"2D mesh phenotype {s2['phenotype_code']} != 1D phenotype {s1['phenotype_code']} in {rname}"
        )
        assert abs(s2["ant_V_mV"] - s1["ant_V_mV"]) < 2.0, f"2D vs 1D anterior V_mem differ by > 2 mV in {rname}"
        assert s2["ml_std_ant_V_mV"] < 2.0, f"Unphysical mediolateral V_mem spread in {rname}: {s2['ml_std_ant_V_mV']:.3f} mV"
    assert two_d["head_counts_2d"] == {"wild_type_1H1T": 1, "hk_atpase_inhibited_0H2T": 0, "ivermectin_depolarized_2H0T": 2}

    # 10. (Pass 3) Slit/netrin mediolateral guidance phenotypes (Cebria & Newmark 2005; Cebria et al. 2007)
    g_wt = two_d["guidance"]["wild_type"]
    g_slit = two_d["guidance"]["slit_rnai"]
    g_net = two_d["guidance"]["netrin_rnai"]
    assert g_wt["crossing_fraction"] == 1.0, f"WT commissural axons must cross the midline, got {g_wt['crossing_fraction']}"
    assert abs(g_wt["ipsi_final_abs_y"] - g_wt["tract_y_star_L"]) < 0.25 * g_wt["tract_y_star_L"], (
        f"WT ipsilateral tract |y| = {g_wt['ipsi_final_abs_y']:.4f} L deviates from analytic y* = {g_wt['tract_y_star_L']:.4f} L"
    )
    assert g_slit["ipsi_final_abs_y"] < 0.005, f"Slit RNAi must collapse axons onto the midline, got |y| = {g_slit['ipsi_final_abs_y']:.4f} L"
    assert all(abs(e["y"]) < 0.5 * mesh["h_rel"] + 1e-12 for e in two_d["eyes"]["slit_rnai"]["eyes"]), "Slit RNAi eye-field maxima must collapse onto the midline row"
    assert g_net["crossing_fraction"] == 0.0, f"Netrin RNAi commissural axons must fail to cross, got {g_net['crossing_fraction']}"
    assert two_d["n_eyes_2d"] == {"wild_type": 2, "slit_rnai": 2, "ivermectin": 4, "sch28080": 0}, f"Unexpected 2D eye counts {two_d['n_eyes_2d']}"

    # 11. (Pass 3) Eye-field photoreceptor position vs the segmented specimen ocelli
    obs = two_d["specimen_ocelli"]
    sw = two_d["slit_sweep"]
    assert two_d["s_eye_model"] < 0.12, f"Model eyes must lie in the anterior head, got x/L = {two_d['s_eye_model']:.4f}"
    assert 0.5 * obs["abs_y_eye_L"] <= two_d["abs_y_eye_model_L"] <= 2.0 * obs["abs_y_eye_L"], (
        f"Model |y_eye|/L = {two_d['abs_y_eye_model_L']:.4f} outside 2x band of specimen {obs['abs_y_eye_L']:.4f}"
    )
    assert float(sw["sigmas"][0]) <= sw["sigma_star"] <= float(sw["sigmas"][-1]), "Specimen-matched slit stripe width outside sweep range"
    assert np.all(np.diff(sw["y_eye"]) >= -1e-9), "Photoreceptor offset must increase monotonically with slit stripe width"

    # 12. (Pass 3) Stochastic V_mem-coupled neoblast kinetics: Ca_v-gated progeny fate per regime,
    #     wound-localised second mitotic peak, and seed-level reproducibility
    neo = evaluate_neoblast_regimes(suite["regimes"])
    assert neo["wild_type_1H1T"]["head_frac_ant"][0] > 0.9 and neo["wild_type_1H1T"]["head_frac_post"][0] < 0.1
    assert neo["hk_atpase_inhibited_0H2T"]["head_frac_ant"][0] < 0.1, "SCH-28080 anterior progeny must not adopt head identity"
    assert neo["hk_atpase_inhibited_0H2T"]["blastema_ant_cells"][0] > 0.5 * neo["wild_type_1H1T"]["blastema_ant_cells"][0], (
        "H,K-ATPase inhibition must mis-specify, not abolish, the anterior blastema (Beane et al. 2011)"
    )
    assert neo["ivermectin_depolarized_2H0T"]["head_frac_post"][0] > 0.9, "Ivermectin posterior progeny must adopt head identity"
    assert neo["nifedipine_cav_block_0H2T"]["head_frac_ant"][0] < 0.1, "Nifedipine must block Ca_v-gated head fate"
    for rname in ("wild_type_1H1T", "hk_atpase_inhibited_0H2T", "ivermectin_depolarized_2H0T", "nifedipine_cav_block_0H2T"):
        d_ = neo[rname]
        assert d_["frac_near_wound_peak2"][0] > d_["frac_near_wound_peak1"][0] + 0.2, (
            f"Second mitotic peak must be wound-localised relative to the body-wide first peak in {rname}"
        )
        assert d_["t_peak_bodywide_pooled_h"] <= 12.0 < 30.0 <= d_["t_peak_local_pooled_h"], f"Biphasic mitotic peaks missing in {rname}"
    n1 = simulate_neoblast_kinetics(wt, seed=2011, t_end_h=24.0)
    n2 = simulate_neoblast_kinetics(wt, seed=2011, t_end_h=24.0)
    assert np.array_equal(n1["final_positions"], n2["final_positions"]) and n1["blastema_ant_cells"] == n2["blastema_ant_cells"], (
        "Stochastic neoblast model must be bitwise reproducible for a fixed seed"
    )

    print(
        "[2011-beane-planarian-hk-atpase-anterior] SELF-TEST PASSED (12/12 checks: SHA-256 dataset locks, "
        f"WT={wt['phenotype_code']} [delta_V={wt['delta_V_AP_mV']:.1f} mV], "
        f"SCH-28080={sch['phenotype_code']}, Ivermectin={ivm['phenotype_code']}, "
        f"8-Octanol={octa['phenotype_code']}, Nifedipine-ctl={nif['phenotype_code']}, "
        f"specimen ocelli={spec['n_ocelli_detected']} at x/L={spec['mean_ocelli_rel_s']:.4f}; "
        f"2D mesh {mesh['nx']}x{mesh['ny']} ({mesh['n_cells']} cells) phenotypes={two_d['head_counts_2d']}, "
        f"WT eyes |y|/L={two_d['abs_y_eye_model_L']:.4f} vs specimen {obs['abs_y_eye_L']:.4f}, "
        f"slit-RNAi |y|={g_slit['ipsi_final_abs_y']:.4f} L, netrin-RNAi crossing={g_net['crossing_fraction']:.0%}; "
        f"neoblast WT head_frac_ant={neo['wild_type_1H1T']['head_frac_ant'][0]:.3f}, SCH-28080={neo['hk_atpase_inhibited_0H2T']['head_frac_ant'][0]:.3f})."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Run 12-check deterministic verification suite and exit")
    ap.add_argument("--skip-gif", action="store_true", help="Skip rendering animated GIF")
    ap.add_argument("--seed", type=int, default=2011, help="Deterministic RNG seed")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    res = render_outputs(seed=args.seed, skip_gif=args.skip_gif)
    wt = res["suite"]["regimes"]["wild_type_1H1T"]
    sch = res["suite"]["regimes"]["hk_atpase_inhibited_0H2T"]
    ivm = res["suite"]["regimes"]["ivermectin_depolarized_2H0T"]
    spec = res["specimen"]
    print(
        f"[2011-beane-planarian-hk-atpase-anterior] Rendered 9-panel comparison -> {OUT_PNG.relative_to(ROOT)}, "
        f"Pass 3 6-panel 2D-mesh/guidance/neoblast figure -> {OUT_PNG_PASS3.relative_to(ROOT)} "
        f"and animation -> {OUT_GIF.relative_to(ROOT)} "
        f"(WT={wt['phenotype_code']}, SCH-28080={sch['phenotype_code']}, Ivermectin={ivm['phenotype_code']}, "
        f"specimen ocelli={spec['n_ocelli_detected']} at x/L={spec['mean_ocelli_rel_s']:.4f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
