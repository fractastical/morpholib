#!/usr/bin/env python3
"""
morphoeval/math_kg.py — Validated "Math of Biology" Knowledge Graph Builder,
Lean 4 Kernel Verifier & SPARQL/SQL Query Engine.

Bridges the gap between prose-embedded morphometrics in `rdfs:comment`, human-readable
formulas (`skos:notation`), and machine-checked formal specifications & theorems in
**Lean 4** (`morphoeval/lean/MathOfBiology.lean`, typed as `skos:definition "..."^^pk:lean4`).

What this script produces & queries:
  1. `morphoeval/lean/MathOfBiology.lean`
       Formal Lean 4 module defining all 19 governing biological laws across all 12
       evaluated morphogenesis systems, with machine-checked theorems (zero `sorry`s)
       verified directly by the Lean 4 kernel (`lean morphoeval/lean/MathOfBiology.lean`).
  2. `morphoeval/kg/math_of_biology.ttl` & `morphoeval/kg/math_of_biology.trig`
       Unified Turtle & TriG RDF knowledge graphs across all 12 evaluated simulations:
         - `pk:Claim` (`pkr:claim/morphoeval-<sim>`) for each organism/model reproduction
         - `pk:Concept` (`pkr:math-law/<sim>-<k>`) for each governing mathematical equation,
           carrying BOTH:
             * `skos:notation` — human-readable mathematical formula
             * `skos:definition "..."^^pk:lean4` — formal Lean 4 `def` + `theorem` code
             * `pk:leanDeclaration` & `pk:leanTheorem` — fully qualified Lean 4 identifiers
             * `pk:leanVerified` (`xsd:boolean`) — whether the Lean 4 kernel verified the proof
             * `rdfs:seeAlso` — commit-pinned GitHub code loci for both Python & Lean 4
         - `pk:EvidenceItem` (`pkr:evidence/<audit>-<sim>-<k>`) for each of the 129
           validated morphometric measurements from `*.measurements.yaml`, with typed
           numeric literals (`schema:value`, `schema:targetValue`, `schema:unitText`,
           `schema:error` for relative error %, `pk:pValue`, `pk:statisticalTest`)
  3. SQLite tables inside `morphoeval/runs/telemetry.sqlite`:
       - `math_models`       (12 evaluated biological models, scores, fingerprints, DOIs)
       - `math_equations`    (19 governing equations with both `formula` and `lean_code` / `lean_theorem` / `lean_verified`)
       - `math_measurements` (129 quantitative measurements, reference values, relative error %)

Usage examples:
  # Verify Lean 4 proofs and rebuild the RDF knowledge graph + SQLite tables
  ./venv/bin/python3 morphoeval/math_kg.py --build

  # Run the Lean 4 kernel check and list all 19 formal Lean 4 declarations & theorems
  ./venv/bin/python3 morphoeval/math_kg.py --lean

  # Run a built-in SPARQL preset (equations | lean-proofs | exact-matches | by-family | p-values)
  ./venv/bin/python3 morphoeval/math_kg.py --preset lean-proofs

  # Inspect one biological system in detail (human formula + Lean 4 theorem + measurements)
  ./venv/bin/python3 morphoeval/math_kg.py --sim vogel-phyllotaxis
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml
from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, SKOS, XSD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDITS_DIR = HERE / "audits"
KG_DIR = HERE / "kg"
LEAN_PATH = HERE / "lean" / "MathOfBiology.lean"
RUBRIC_PATH = HERE / "rubric.yaml"
SQLITE_PATH = HERE / "runs" / "telemetry.sqlite"
MATH_TTL_PATH = KG_DIR / "math_of_biology.ttl"
MATH_TRIG_PATH = KG_DIR / "math_of_biology.trig"

PK = Namespace("https://w3id.org/probknow/ontology/")
PKR = Namespace("https://w3id.org/probknow/resource/")
SCHEMA = Namespace("https://schema.org/")
NP = Namespace("http://www.nanopub.org/nschema#")
NPX = Namespace("http://purl.org/nanopub/x/")
NT = Namespace("https://w3id.org/np/o/ntemplate/")

P_VALUE_RE = re.compile(r"\bp\s*[=<≈]\s*(0?\.[0-9]+(?:e[+-]?[0-9]+)?|[0-9]+(?:\.[0-9]+)?e-[0-9]+)")


# --------------------------------------------------------------------------- #
# Canonical Mathematical Formalisms & Lean 4 Specifications (12 Systems, 19 Laws)
# --------------------------------------------------------------------------- #
MATH_LAWS_REGISTRY: dict[str, dict[str, Any]] = {
    "thompson-scarus-pomacanthus": {
        "math_family": "Biharmonic Thin-Plate Spline & Affine Coordinate Transformation",
        "equations": [
            {
                "law_name": "2D Biharmonic Thin-Plate Spline (TPS) Deformation Field",
                "formula": "Φ(x, y) = a_0 + a_x x + a_y y + ∑_{i=1}^K w_i r_i^2 ln(r_i^2),  Δ^2 Φ = 0",
                "lean_decl": "MathOfBiology.tps_zero_displacement_kernel",
                "lean_theorem": "MathOfBiology.tps_kernel_zero_at_origin",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L181-L189",
                "lean_code": (
                    "def tps_zero_displacement_kernel (r_sq log_r_sq : Int) : Int := r_sq * log_r_sq\n"
                    "theorem tps_kernel_zero_at_origin (log_val : Int) :\n"
                    "    tps_zero_displacement_kernel 0 log_val = 0 := by\n"
                    "  simp [tps_zero_displacement_kernel]"
                ),
                "variables": "(x, y): source Scarus landmark coordinates; r_i = ||(x,y) - p_i||; w_i: biharmonic kernel weights",
                "role": "Maps homologous anatomical landmarks of Scarus (parrotfish) onto Pomacanthus (angelfish) with minimum bending energy",
                "source_citation": "Thompson (1917) On Growth and Form Ch. XVII Fig. 517-518; Bookstein (1989) IEEE TPAMI 11:567",
                "code_locus": "thompson/1917-thompson.py#L95-L185",
            },
            {
                "law_name": "Affine Oblique Shear & Principal Strain Tensor",
                "formula": "x' = α_x x + γ_xy y + t_x,  y' = γ_yx x + α_y y + t_y;  J = det(∇Φ)",
                "lean_decl": "MathOfBiology.affine_jacobian_det",
                "lean_theorem": "MathOfBiology.thompson_unit_shear_area_preserving",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L191-L200",
                "lean_code": (
                    "def affine_jacobian_det (a_x a_y g_xy g_yx : Int) : Int := a_x * a_y - g_xy * g_yx\n"
                    "theorem thompson_unit_shear_area_preserving (g_xy : Int) :\n"
                    "    affine_jacobian_det 1 1 g_xy 0 = 1 := by\n"
                    "  simp [affine_jacobian_det]"
                ),
                "variables": "α_x, α_y: axial stretch ratios; γ_xy, γ_yx: oblique grid shear angles; J: local areal dilation",
                "role": "Linear global coordinate shear baseline compared against non-affine TPS warp across 14 anatomical landmarks",
                "source_citation": "Thompson (1917) On Growth and Form Ch. XVII",
                "code_locus": "thompson/1917-thompson.py#L186-L260",
            },
        ],
    },
    "turing-gray-scott": {
        "math_family": "Reaction-Diffusion Partial Differential Equations (PDE)",
        "equations": [
            {
                "law_name": "Gray-Scott Autocatalytic Reaction-Diffusion System",
                "formula": "∂u/∂t = D_u ∇^2 u - u v^2 + F(1 - u),   ∂v/∂t = D_v ∇^2 v + u v^2 - (F + k)v",
                "lean_decl": "MathOfBiology.gray_scott_react_u",
                "lean_theorem": "MathOfBiology.gray_scott_homogeneous_equilibrium",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L208-L221",
                "lean_code": (
                    "def gray_scott_react_u (F u v : Int) : Int := -(u * v * v) + F * (1 - u)\n"
                    "def gray_scott_react_v (F k u v : Int) : Int := u * v * v - (F + k) * v\n"
                    "theorem gray_scott_homogeneous_equilibrium (F k : Int) :\n"
                    "    gray_scott_react_u F 1 0 = 0 ∧ gray_scott_react_v F k 1 0 = 0 := by\n"
                    "  simp [gray_scott_react_u, gray_scott_react_v]"
                ),
                "variables": "u(x,y,t): substrate concentration; v(x,y,t): autocatalyst concentration; D_u, D_v: diffusivities; F: feed rate; k: kill rate",
                "role": "Generates symmetry-breaking Turing spots (mitosis), labyrinthine stripes, and coral/soliton regimes across (F, k) morphospace",
                "source_citation": "Turing (1952) Phil. Trans. R. Soc. B 237:37; Pearson (1993) Science 261:189",
                "code_locus": "1952-turing-morpho.py#L45-L140",
            },
        ],
    },
    "raup-shell-coiling": {
        "math_family": "Logarithmic Helicospiral Parametric Differential Geometry",
        "equations": [
            {
                "law_name": "Raup (1966) 4-Parameter Logarithmic Helicospiral Surface",
                "formula": "r(θ, φ) = r_0 W^{θ/2π} [1 + ((1 - D)/(1 + D)) cos φ],   z(θ, φ) = r_0 T (W^{θ/2π} - 1) + r_0 W^{θ/2π} S ((1 - D)/(1 + D)) sin φ",
                "lean_decl": "MathOfBiology.raup_whorl_radius",
                "lean_theorem": "MathOfBiology.raup_whorl_expansion_recovery",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L147-L169",
                "lean_code": (
                    "def raup_whorl_radius (r0 W : Nat) : Nat → Nat\n"
                    "  | 0 => r0\n"
                    "  | k + 1 => W * raup_whorl_radius r0 W k\n"
                    "theorem raup_whorl_expansion_recovery (r0 W k : Nat) :\n"
                    "    raup_whorl_radius r0 W (k + 1) = W * raup_whorl_radius r0 W k := by rfl"
                ),
                "variables": "W: whorl expansion rate; D: relative distance of generating curve from coiling axis; T: whorl translation rate (dz_c/dr_c); S: aperture aspect ratio",
                "role": "Spans gastropod, cephalopod, bivalve, and brachiopod shell morphospace in 3D cylindrical coordinates",
                "source_citation": "Raup (1966) J. Paleontol. 40(5):1178-1190 Eq. (1)-(6); Noshita (2014)",
                "code_locus": "create_raup_animations.py#L72-L148",
            },
            {
                "law_name": "Generating-Curve Center Parameter Recovery Identities",
                "formula": "W_rec = r_c(θ) / r_c(θ - 2π),   D_rec = r_min(θ) / r_max(θ),   T_rec = Δz_c / Δr_c",
                "lean_decl": "MathOfBiology.raup_axial_step",
                "lean_theorem": "MathOfBiology.raup_translation_rate_recovery",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L161-L173",
                "lean_code": (
                    "def raup_axial_step (T delta_r : Nat) : Nat := T * delta_r\n"
                    "theorem raup_translation_rate_recovery (T delta_r : Nat) (h_pos : delta_r > 0) :\n"
                    "    (raup_axial_step T delta_r) / delta_r = T := by\n"
                    "  simp [raup_axial_step, Nat.mul_div_cancel _ h_pos]"
                ),
                "variables": "r_c, z_c: generating-curve centroid coordinates; r_min, r_max: inner and outer aperture margins",
                "role": "Exact inverse morphometric recovery verifying 3D mesh fidelity against input (W, D, T) across all 6 canonical shell regimes",
                "source_citation": "Raup (1966) J. Paleontol. 40(5):1178-1190; Okabe & Yoshimura (2017)",
                "code_locus": "morphoeval/measure_outputs.py#L220-L310",
            },
        ],
    },
    "vogel-phyllotaxis": {
        "math_family": "Kinematic Spiral Geometry, Repulsive Dynamical Systems & Delaunay Topology",
        "equations": [
            {
                "law_name": "Vogel (1979) Planar Capitulum Golden-Angle Spiral Law",
                "formula": "r_n = c √n,   θ_n = n α_0 (mod 2π),   α_0 = 360° / τ^2 = 360° (2 - τ) ≈ 137.507764°",
                "lean_decl": "MathOfBiology.vogel_radius_sq",
                "lean_theorem": "MathOfBiology.vogel_constant_annulus_area",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L97-L116",
                "lean_code": (
                    "def vogel_radius_sq (c_sq n : Nat) : Nat := c_sq * n\n"
                    "theorem vogel_constant_annulus_area (c_sq n : Nat) :\n"
                    "    vogel_radius_sq c_sq (n + 1) - vogel_radius_sq c_sq n = c_sq := by\n"
                    "  simp [vogel_radius_sq, Nat.mul_add, Nat.add_sub_cancel_left]\n"
                    "theorem vogel_fibonacci_parastichy_cascade :\n"
                    "    (fib 6, fib 7) = (8, 13) ∧ (fib 7, fib 8) = (13, 21) ∧ (fib 8, fib 9) = (21, 34) ∧\n"
                    "    (fib 9, fib 10) = (34, 55) ∧ (fib 10, fib 11) = (55, 89) := by decide"
                ),
                "variables": "n: floret plastochron index; r_n: radial distance; θ_n: divergence angle; τ = (1+√5)/2: golden ratio; J: jugacy",
                "role": "Constant-area centripetal floret packing producing the rising Fibonacci parastichy cascade (8,13)->(13,21)->(21,34)->(34,55) and bounded Voronoi optimality",
                "source_citation": "Vogel (1979) Mathematical Biosciences 44:179-189 Eq. (1)-(4)",
                "code_locus": "1979-vogel-phyllotaxis.py#L110-L154",
            },
            {
                "law_name": "Douady & Couder (1992, 1996) Hofmeister Inhibitory Energy Minimization",
                "formula": "E(θ) = ∑_{j < n} exp(-d(r_0, θ; r_j(t), θ_j) / λ),   θ_n^* = argmin_θ E(θ),   r_j(t) = r_0 (1 + G)^m",
                "lean_decl": "MathOfBiology.hofmeister_pair_kernel",
                "lean_theorem": "MathOfBiology.douady_couder_kernel_symmetric",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L118-L128",
                "lean_code": (
                    "def hofmeister_pair_kernel (kernel_fn : Nat → Nat) (dist_sq : Nat) : Nat := kernel_fn dist_sq\n"
                    "theorem douady_couder_kernel_symmetric (kernel_fn : Nat → Nat) (d_ij d_ji : Nat)\n"
                    "    (h_sym : d_ij = d_ji) :\n"
                    "    hofmeister_pair_kernel kernel_fn d_ij = hofmeister_pair_kernel kernel_fn d_ji := by\n"
                    "  rw [h_sym]"
                ),
                "variables": "G = v Δt / r_0: dimensionless plastochron ratio; λ: inhibitory interaction decay length; d(·): Euclidean distance on meristem",
                "role": "Dynamical self-organization of sequentially initiated primordia onto the main Fibonacci (137.51°) and Lucas (99.50°) noble-angle attractors",
                "source_citation": "Douady & Couder (1992) Phys. Rev. Lett. 68:2098; (1996) J. Theor. Biol. 178:255",
                "code_locus": "1979-vogel-phyllotaxis.py#L383-L441",
            },
            {
                "law_name": "Delaunay Contact Parastichy & Outer-Rim Dislocation Asymmetry",
                "formula": "m_obs = m_Fib - Δm_rim (Δm_rim ∈ {0, 1}),   P(Fib - 1) > P(Fib + 1)  (binomial p = 5.07×10^{-5})",
                "lean_decl": "MathOfBiology.rim_dislocation_count",
                "lean_theorem": "MathOfBiology.swinton_rim_dislocation_nonincreasing",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L130-L139",
                "lean_code": (
                    "def rim_dislocation_count (m_fib : Nat) (lost_chain : Bool) : Nat :=\n"
                    "  if lost_chain then m_fib - 1 else m_fib\n"
                    "theorem swinton_rim_dislocation_nonincreasing (m_fib : Nat) (lost_chain : Bool) :\n"
                    "    rim_dislocation_count m_fib lost_chain ≤ m_fib := by\n"
                    "  cases lost_chain <;> simp [rim_dislocation_count, Nat.sub_le]"
                ),
                "variables": "(m, n): opposed Delaunay contact-spiral counts in annulus [r_1, r_2); σ_r, σ_θ: biological placement noise",
                "role": "Explains why incomplete outer-rim contact chains under biological noise cause a 3:1 excess of (Fibonacci - 1) over (Fibonacci + 1) counts in Swinton et al. (2016)",
                "source_citation": "Swinton, Ochu & MSI Turing's Sunflower Consortium (2016) R. Soc. Open Sci. 3:160091 Table 1",
                "code_locus": "1979-vogel-phyllotaxis.py#L470-L562",
            },
        ],
    },
    "wolfram-conus-textile-rule30": {
        "math_family": "Elementary Cellular Automata (GF(2)) & Reaction-Diffusion PDE",
        "equations": [
            {
                "law_name": "Wolfram (1983, 1984) Rule 30 Boolean & GF(2) Algebraic Normal Form",
                "formula": "a_{t+1, i} = a_{t, i-1} ⊕ (a_{t, i} ∨ a_{t, i+1}) ≡ a_{t, i-1} + a_{t, i} + a_{t, i+1} + a_{t, i} a_{t, i+1} (mod 2)",
                "lean_decl": "MathOfBiology.rule30",
                "lean_theorem": "MathOfBiology.rule30_anf_equiv",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L21-L37",
                "lean_code": (
                    "def rule30 (p q r : Bool) : Bool := bne p (q || r)\n"
                    "def rule30_anf (p q r : Bool) : Bool := bne (bne p q) (bne r (q && r))\n"
                    "theorem rule30_anf_equiv (p q r : Bool) : rule30 p q r = rule30_anf p q r := by\n"
                    "  cases p <;> cases q <;> cases r <;> rfl"
                ),
                "variables": "a_{t,i} ∈ {0, 1}: pigment state at mantle cell i and step t; ⊕: XOR; ∨: OR",
                "role": "Discrete chaotic generator of self-similar white triangular tents on Conus textile shells and OEIS A051023 center column",
                "source_citation": "Wolfram (1983) Rev. Mod. Phys. 55:601; (1984) Nature 311:419",
                "code_locus": "1984-wolfram-conus-textile-rule30.py#L96-L152",
            },
            {
                "law_name": "Erica Jen (1990) Left-Bijective Spatial Inverse & Diagonal Period-Doubling Recursion",
                "formula": "a_{t, i-1} = a_{t+1, i} ⊕ (a_{t, i} ∨ a_{t, i+1}),   d_k(t+1) = d_{k-2}(t) ⊕ (d_{k-1}(t) ∨ d_k(t))",
                "lean_decl": "MathOfBiology.rule30_left_inverse",
                "lean_theorem": "MathOfBiology.jen_1990_left_bijective",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L39-L61",
                "lean_code": (
                    "def rule30_left_inverse (a_next q r : Bool) : Bool := bne a_next (q || r)\n"
                    "theorem jen_1990_left_bijective (p q r : Bool) :\n"
                    "    rule30_left_inverse (rule30 p q r) q r = p := by\n"
                    "  cases p <;> cases q <;> cases r <;> rfl\n"
                    "theorem jen_1990_barrier_shielding (d_km2 d_k : Bool) :\n"
                    "    jen_diagonal_step d_km2 true d_k = !d_km2 := by\n"
                    "  cases d_km2 <;> cases d_k <;> rfl"
                ),
                "variables": "d_k(t) = a_{t, k-t}: k-th left-cone diagonal; P_k = 2^m: exact temporal period doubling at k ∈ {0, 3, 8, 29, 400, ...}",
                "role": "Proves exact left-bijectivity (0 bit errors) and explains why left-cone diagonals lock into period-doubling crystals separated by d_{k-1}=0 barriers",
                "source_citation": "Jen (1990) Physica D 45(1-3):3-18 Theorem 1 & Section 2",
                "code_locus": "1984-wolfram-conus-textile-rule30.py#L172-L256",
            },
            {
                "law_name": "Meinhardt & Klingler (1987) 3-Species Activator-Substrate-Hormone Mantle PDE",
                "formula": "∂a/∂t = ρ s (a^2 / (1 + κ a^2) + ρ_0) - μ_a a + D_a ∂^2a/∂x^2,   ∂s/∂t = σ (h / (1 + γ h)) - ρ s a^2 - μ_s s + D_s ∂^2s/∂x^2",
                "lean_decl": "MathOfBiology.meinhardt_activator_reaction",
                "lean_theorem": "MathOfBiology.meinhardt_basal_quiescence",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L63-L82",
                "lean_code": (
                    "def meinhardt_activator_reaction (rho kappa rho0 mu_a : Nat) (a s : Nat) : Int :=\n"
                    "  Int.ofNat (rho * s * (a * a + rho0 * (1 + kappa * a * a))) -\n"
                    "    Int.ofNat (mu_a * a * (1 + kappa * a * a))\n"
                    "theorem meinhardt_basal_quiescence (rho kappa mu_a s : Nat) :\n"
                    "    meinhardt_activator_reaction rho kappa 0 mu_a 0 s = 0 := by\n"
                    "  simp [meinhardt_activator_reaction]"
                ),
                "variables": "a(x,t): pigment activator; s(x,t): depleted substrate; h(x,t): fast-diffusing systemic hormone",
                "role": "Continuous biochemical mechanism producing self-extinguishing pigment triangles and V-shaped travelling waves after mantle lesion",
                "source_citation": "Meinhardt & Klingler (1987) J. Theor. Biol. 126(1):63-89 Eq. (1)-(3)",
                "code_locus": "1984-wolfram-conus-textile-rule30.py#L315-L412",
            },
        ],
    },
    "cervera-levin-mafe": {
        "math_family": "Bioelectric Gap-Junction Coupled Nonlinear Dynamical Network",
        "equations": [
            {
                "law_name": "Cervera-Levin-Mafe (2021) Gap-Junction Coupled Membrane Potential ODE",
                "formula": "C_m dV_i/dt = -I_dep(V_i) - I_hyp(V_i) + ∑_{j ∈ N(i)} G_{ij}(V_i, V_j) (V_j - V_i)",
                "lean_decl": "MathOfBiology.gap_junction_current",
                "lean_theorem": "MathOfBiology.cervera_equipotential_zero_current",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L229-L237",
                "lean_code": (
                    "def gap_junction_current (G_ij V_i V_j : Int) : Int := G_ij * (V_j - V_i)\n"
                    "theorem cervera_equipotential_zero_current (G_ij V : Int) :\n"
                    "    gap_junction_current G_ij V V = 0 := by\n"
                    "  simp [gap_junction_current]"
                ),
                "variables": "V_i: transmembrane voltage of cell i; C_m: membrane capacitance; I_dep, I_hyp: inward/outward ion currents; G_{ij}: gap-junction conductance",
                "role": "Models bioelectric pattern memory, bistable polarized vs. depolarized domains, and double-head (2H) vs. wild-type (1H) planarian regeneration attractors",
                "source_citation": "Cervera, Levin & Mafe (2021) J. Phys. Chem. B / Bioelectrochemistry",
                "code_locus": "2021-Cervera–Levin–Mafe.py#L65-L185",
            },
            {
                "law_name": "Voltage-Gated Bell-Boltzmann Gap-Junction Conductance",
                "formula": "G_{ij}(V_i, V_j) = G_0 / [ (1 + exp((V_i - V_th)/V_0)) (1 + exp((V_j - V_th)/V_0)) ]",
                "lean_decl": "MathOfBiology.bell_boltzmann_denom",
                "lean_theorem": "MathOfBiology.cervera_bell_boltzmann_symmetric",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L239-L248",
                "lean_code": (
                    "def bell_boltzmann_denom (b_i b_j : Nat) : Nat := (1 + b_i) * (1 + b_j)\n"
                    "theorem cervera_bell_boltzmann_symmetric (b_i b_j : Nat) :\n"
                    "    bell_boltzmann_denom b_i b_j = bell_boltzmann_denom b_j b_i := by\n"
                    "  simp [bell_boltzmann_denom, Nat.mul_comm]"
                ),
                "variables": "G_0: maximum open gap-junction conductance; V_th: gating threshold voltage; V_0: thermal/voltage slope factor",
                "role": "Nonlinear intercellular coupling that isolates depolarized wounds or unlocks regional tissue-level bioelectric synchronization",
                "source_citation": "Cervera, Levin & Mafe (2021)",
                "code_locus": "2021-Cervera–Levin–Mafe.py#L110-L160",
            },
        ],
    },
    "planform-morphology-animation": {
        "math_family": "Formal Anatomical Region Graph Grammar & Polar Geometry",
        "equations": [
            {
                "law_name": "Lobo et al. (2013) PlanformDB Region-Link Polar Graph Encoding",
                "formula": "p_B = p_A + r_A(θ_{AB}) u(θ_{AB}) + d_{AB} u(θ_{AB}),   R_i(φ) = piecewise-ellipse(r_{ant}, r_{post}, r_{lat})",
                "lean_decl": "MathOfBiology.planform_wildtype_ap_length",
                "lean_theorem": "MathOfBiology.lobo_2013_wildtype_dimensions",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L256-L266",
                "lean_code": (
                    "def planform_wildtype_ap_length (r_head r_ant_trunk r_post_trunk r_tail : Nat) : Nat :=\n"
                    "  r_head + r_ant_trunk + r_post_trunk + r_tail\n"
                    "theorem lobo_2013_wildtype_dimensions :\n"
                    "    planform_wildtype_ap_length 80 200 200 80 = 560 ∧ (60 + 60 : Nat) = 120 := by decide"
                ),
                "variables": "p_A, p_B: connected body-region centers; θ_{AB}: polar attachment angle; r_{ant}, r_{post}, r_{lat}: directional semi-axes",
                "role": "Decodes all 369 relational planarian body morphologies and internal organ placements (eyes, brain, pharynx) from PlanformDB",
                "source_citation": "Lobo, Malone & Levin (2013) PLoS Comput. Biol. 9(10):e1003287",
                "code_locus": "2013-lobo-planform-morphology.py#L80-L240",
            },
        ],
    },
    "lobo-limbform-morphology": {
        "math_family": "Recursive Tree Graph Grammar & Segmental Kinematics",
        "equations": [
            {
                "law_name": "Lobo et al. (2014) Limbform Recursive Segmental Tree Vector Law",
                "formula": "x_{k+1} = x_k + L_k [cos(∑_{j≤k} α_j), sin(∑_{j≤k} α_j)]^T,   T = (s, α, r, L, children)",
                "lean_decl": "MathOfBiology.limb_segment_count",
                "lean_theorem": "MathOfBiology.lobo_2014_three_segment_axis",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L274-L287",
                "lean_code": (
                    "inductive LimbTree where\n"
                    "  | leaf (length : Nat) : LimbTree\n"
                    "  | branch (length : Nat) (child : LimbTree) : LimbTree\n"
                    "theorem lobo_2014_three_segment_axis (L1 L2 L3 : Nat) :\n"
                    "    limb_segment_count (LimbTree.branch L1 (LimbTree.branch L2 (LimbTree.leaf L3))) = 3 := by rfl"
                ),
                "variables": "L_k: limb podial/zeugopod/stylopod segment length; α_j: relative joint branching angle; r: segment radius",
                "role": "Decodes normal, regenerating, and ectopic/supernumerary amphibian and vertebrate limb morphologies from LimbformDB",
                "source_citation": "Lobo, Feldman & Levin (2014) Dev. Biol. / Bioinformatics",
                "code_locus": "2014-lobo-limbform-morphology.py#L75-L210",
            },
        ],
    },
    "santella-wormguides-celegans": {
        "math_family": "Sulston Binary Lineage Tree & 4D Spatiotemporal Trajectory Kinematics",
        "equations": [
            {
                "law_name": "Sulston Embryonic Binary Lineage Division & 4D Nuclear Interpolation",
                "formula": "x_i(t) = (1 - λ(t)) x_i(t_k) + λ(t) x_i(t_{k+1}),   v_i(t) = ||x_i(t + Δt) - x_i(t)|| / Δt",
                "lean_decl": "MathOfBiology.nuclear_lerp",
                "lean_theorem": "MathOfBiology.sulston_nuclear_lerp_endpoints",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L295-L303",
                "lean_code": (
                    "def nuclear_lerp (x0 x1 lam : Int) : Int := (1 - lam) * x0 + lam * x1\n"
                    "theorem sulston_nuclear_lerp_endpoints (x0 x1 : Int) :\n"
                    "    nuclear_lerp x0 x1 0 = x0 ∧ nuclear_lerp x0 x1 1 = x1 := by\n"
                    "  simp [nuclear_lerp]"
                ),
                "variables": "x_i(t) = (X_i, Y_i, Z_i): 3D nuclear centroid of blastomere i at embryonic minute t; founder lineages ∈ {AB, MS, E, C, D, P4}",
                "role": "Reconstructs 3D embryonic morphogenesis, gastrulation, and ventral enclosure of Caenorhabditis elegans from WormGUIDES nuclear tracks",
                "source_citation": "Santella et al. (2015) BMC Bioinformatics 16:189; Sulston et al. (1983)",
                "code_locus": "2015-santella-wormguides-celegans.py#L90-L230",
            },
        ],
    },
    "hoffmann-wing-venation": {
        "math_family": "Planar Straight-Line Graph (PSLG) Polygonal Tessellation Geometry",
        "equations": [
            {
                "law_name": "Wing Vein Polygonal Cell Area & Isoperimetric Circularity Law",
                "formula": "A_i = (1/2) |∑_{k=1}^m (x_k y_{k+1} - x_{k+1} y_k)|,   C_i = 4π A_i / P_i^2 ∈ (0, 1]",
                "lean_decl": "MathOfBiology.shoelace_edge_term",
                "lean_theorem": "MathOfBiology.hoffmann_shoelace_antisymmetric",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L311-L319",
                "lean_code": (
                    "def shoelace_edge_term (xk yk xk1 yk1 : Int) : Int := xk * yk1 - xk1 * yk\n"
                    "theorem hoffmann_shoelace_antisymmetric (xk yk xk1 yk1 : Int) :\n"
                    "    shoelace_edge_term xk1 yk1 xk yk = - shoelace_edge_term xk yk xk1 yk1 := by\n"
                    "  simp [shoelace_edge_term, Int.neg_sub]"
                ),
                "variables": "(x_k, y_k): ordered polygon vertices of wing cell i; A_i: polygon area; P_i: perimeter; C_i: isoperimetric circularity",
                "role": "Quantifies primary vs. secondary insect wing venation domain geometry and cell shape distributions",
                "source_citation": "Hoffmann, Donoughe et al. (2018) PNAS 115(40):9905-9910",
                "code_locus": "2018-hoffmann-wing-venation.py#L60-L180",
            },
        ],
    },
    "eshghi-odonata-wing-cells": {
        "math_family": "Spatial Polygonal Morphometrics & Covariance Eigenvalue Elongation",
        "equations": [
            {
                "law_name": "Odonata Wing Cell Circularity & Second-Moment Eigenvalue Elongation",
                "formula": "C_i = 4π A_i / P_i^2,   E_i = 1 - √(λ_min(Σ_i) / λ_max(Σ_i))",
                "lean_decl": "MathOfBiology.elongation_anisotropy_diff",
                "lean_theorem": "MathOfBiology.eshghi_isotropic_zero_elongation",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L327-L336",
                "lean_code": (
                    "def elongation_anisotropy_diff (lam_max lam_min : Nat) : Nat := lam_max - lam_min\n"
                    "theorem eshghi_isotropic_zero_elongation (lam : Nat) :\n"
                    "    elongation_anisotropy_diff lam lam = 0 := by\n"
                    "  simp [elongation_anisotropy_diff]"
                ),
                "variables": "A_i, P_i: wing cell polygon area and perimeter; Σ_i: 2×2 boundary/interior covariance matrix; λ_min, λ_max: principal eigenvalues",
                "role": "Measures proximo-distal and anterior-posterior cell size, circularity, and elongation gradients across dragonfly (Anisoptera) and damselfly (Zygoptera) wings",
                "source_citation": "Eshghi et al. (2026) Odonata Wing Venation Morphometrics",
                "code_locus": "2026-eshghi-odonata-wing-cells.py#L70-L215",
            },
        ],
    },
    "chindapol-coral-flow": {
        "math_family": "Advection-Diffusion Nutrient Transport & Péclet-Number Accretion",
        "equations": [
            {
                "law_name": "Advection-Diffusion Nutrient Boundary Layer & Péclet Number",
                "formula": "u · ∇c = D ∇^2 c,   Pe = U L / D,   v_n = k_g (∇c · n)|_Γ",
                "lean_decl": "MathOfBiology.peclet_numerator",
                "lean_theorem": "MathOfBiology.chindapol_zero_flow_diffusive",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L344-L352",
                "lean_code": (
                    "def peclet_numerator (U L : Nat) : Nat := U * L\n"
                    "theorem chindapol_zero_flow_diffusive (L : Nat) :\n"
                    "    peclet_numerator 0 L = 0 := by\n"
                    "  simp [peclet_numerator]"
                ),
                "variables": "c(x): dissolved inorganic carbon/nutrient concentration; u(x): flow velocity field; Pe: Péclet number; v_n: normal skeletal accretion rate",
                "role": "Couples unidirectional vs. high-flow hydrodynamic boundary layer thinning to symmetric bushy vs. upstream-curved Pocillopora coral branching",
                "source_citation": "Chindapol et al. (2013) PLoS Comput. Biol. 9(1):e1002849",
                "code_locus": "2013-chindapol-coral-flow.py#L60-L190",
            },
        ],
    },
    "runions-leaf-venation": {
        "math_family": "Auxin Space-Colonization, Murray's Minimum-Work Vascular Law & Planar Dual Loop Nesting",
        "equations": [
            {
                "law_name": "Murray (1926) / Price et al. (2013) Vascular Pipe-Model Conservation Law",
                "formula": "r_parent^γ = ∑_{i ∈ children} r_i^γ,   r_root^γ = ∑_{leaf ∈ tips} r_leaf^γ",
                "lean_decl": "MathOfBiology.ObeysMurrayLaw",
                "lean_theorem": "MathOfBiology.murray_1926_tree_conservation",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L367-L403",
                "lean_code": (
                    "def ObeysMurrayLaw (γ : Nat) : VeinTree → Prop\n"
                    "  | VeinTree.tip _ => True\n"
                    "  | VeinTree.fork r l right =>\n"
                    "      r ^ γ = (vein_root_radius l) ^ γ + (vein_root_radius right) ^ γ ∧\n"
                    "      ObeysMurrayLaw γ l ∧ ObeysMurrayLaw γ right\n"
                    "theorem murray_1926_tree_conservation (γ : Nat) (t : VeinTree) (h : ObeysMurrayLaw γ t) :\n"
                    "    (vein_root_radius t) ^ γ = sum_gamma_powers γ (vein_tip_radii t)"
                ),
                "variables": "r_parent, r_i: parent and daughter vein radii; γ ∈ [2, 3]: vascular scaling exponent (γ = 3 Murray cube law in minor veins -> γ = 2 Da Vinci area-preserving in major veins)",
                "role": "Governs basipetal vascular radius accumulation and daughter/parent cross-sectional area ratios across Horton-Strahler orders 1-8",
                "source_citation": "Murray (1926) PNAS 12:207; Runions et al. (2005) Sec. 3.5; Price, Knox & Brodribb (2013) PLoS ONE 8(12):e85420",
                "code_locus": "2005-runions-leaf-venation.py#L463-L518",
            },
            {
                "law_name": "Runions et al. (2005) Eq. (2)-(3) Toussaint Relative Neighborhood Graph (RNG) Lune Law",
                "formula": "V(s) = {v ∈ V : ∀ u ∈ V, ||v - s|| ≤ max(||u - s||, ||v - u||)},   v' = v + D (∑_{s ∈ S(v)} (s - v)/||s - v||) / ||∑_{s ∈ S(v)} (s - v)/||s - v||||",
                "lean_decl": "MathOfBiology.rng_witness_excludes",
                "lean_theorem": "MathOfBiology.runions_2005_rng_lune_iff",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L410-L422",
                "lean_code": (
                    "def in_rng_lune (d_vs d_us d_vu : Nat) : Prop := d_us < d_vs ∧ d_vu < d_vs\n"
                    "def rng_witness_excludes (d_vs d_us d_vu : Nat) : Bool := Nat.max d_us d_vu < d_vs\n"
                    "theorem runions_2005_rng_lune_iff (d_vs d_us d_vu : Nat) :\n"
                    "    rng_witness_excludes d_vs d_us d_vu = true ↔ in_rng_lune d_vs d_us d_vu := by\n"
                    "  simp [rng_witness_excludes, in_rng_lune, Nat.max_lt]"
                ),
                "variables": "s ∈ S: discrete auxin source; v ∈ V: vein node; V(s): Relative Neighborhood vein set; D: vein advance step; d_k: kill distance",
                "role": "Attracts multiple surrounding vein tips toward a shared auxin source via empty-lune RNG neighborhoods to form closed anastomosing areole loops",
                "source_citation": "Runions, Fuhrer, Lane, Federl, Rolland-Lagan & Prusinkiewicz (2005) ACM Trans. Graph. 24(3):702-711 Eq. (1)-(3)",
                "code_locus": "2005-runions-leaf-venation.py#L183-L456",
            },
            {
                "law_name": "Euler-Poincaré Planar Areole Cycle Rank & Katifori (2012) Hierarchical Edge-Deletion Law",
                "formula": "L = E - V + 1,   q_j = 1 - |r_j - s_j| / (r_j + s_j - 1),   i_u = (1/(N-1)) ∑_j q_j",
                "lean_decl": "MathOfBiology.planar_cycle_count",
                "lean_theorem": "MathOfBiology.katifori_2012_edge_deletion_merges_areoles",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L430-L444",
                "lean_code": (
                    "def planar_cycle_count (edges vertices : Nat) : Nat := edges + 1 - vertices\n"
                    "theorem katifori_2012_tree_zero_cycles (n_edges : Nat) :\n"
                    "    planar_cycle_count n_edges (n_edges + 1) = 0 := by simp [planar_cycle_count]\n"
                    "theorem katifori_2012_edge_deletion_merges_areoles (n_edges k_bridges : Nat) (h : 1 ≤ k_bridges) :\n"
                    "    planar_cycle_count (n_edges + k_bridges - 1) (n_edges + 1) =\n"
                    "    planar_cycle_count (n_edges + k_bridges) (n_edges + 1) - 1"
                ),
                "variables": "V, E: planar vascular vertices and edges; L: enclosed areole count; r_j, s_j: merged subtree sizes in hierarchical loop decomposition; i_u: unweighted nesting number",
                "role": "Separates open dichotomous venation (L = 0, e.g. Ginkgo biloba) from closed reticulate venation (L > 0) and quantifies hierarchical loop nesting",
                "source_citation": "Katifori & Magnasco (2012) PLoS ONE 7(6):e37994; Ronellenfitsch et al. (2015) PLoS Comput. Biol. 11(12):e1004680",
                "code_locus": "2005-runions-leaf-venation.py#L634-L712",
            },
        ],
    },
    "bressloff-v1-form-constants": {
        "math_family": "Euclidean Symmetry-Breaking Neural-Field Bifurcation & Log-Polar Retino-Cortical Geometry",
        "equations": [
            {
                "law_name": "Ermentrout & Cowan (1979) / Bressloff et al. (2001) Retino-Cortical Pullback Classification",
                "formula": "x = (α/ε) ln(1 + ε r/w₀),  y = β r θ/(w₀ + ε r)  (far field ε r ≫ w₀: x ≈ (α/ε) ln(ε r/w₀), y ≈ (β/ε) θ);  phase(k_x, k_y) = k_x x + k_y y:  k_y = 0 → rings (tunnel),  k_x = 0 → rays (funnel),  else log-spiral θ = c − (k_x/k_y) ln r",
                "lean_decl": "MathOfBiology.retino_cortical_phase",
                "lean_theorem": "MathOfBiology.ermentrout_cowan_log_spiral",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L507-L531",
                "lean_code": (
                    "def retino_cortical_phase (kx ky x y : Int) : Int := kx * x + ky * y\n"
                    "theorem ermentrout_cowan_tunnel_rings (kx x y y' : Int) :\n"
                    "    retino_cortical_phase kx 0 x y = retino_cortical_phase kx 0 x y' := by simp [retino_cortical_phase]\n"
                    "theorem ermentrout_cowan_funnel_rays (ky x x' y : Int) :\n"
                    "    retino_cortical_phase 0 ky x y = retino_cortical_phase 0 ky x' y := by simp [retino_cortical_phase]\n"
                    "theorem ermentrout_cowan_log_spiral (kx ky x x' y y' : Int) :\n"
                    "    retino_cortical_phase kx ky x y - retino_cortical_phase kx ky x' y' =\n"
                    "    kx * (x - x') + ky * (y - y') := by\n"
                    "  simp [retino_cortical_phase, Int.mul_sub]\n"
                    "  omega"
                ),
                "variables": "r, θ: retinal polar coordinates; x, y: cortical coordinates (mm); α, β: cortical magnification constants; ε: foveal scale; w₀: foveal radius; (k_x, k_y): cortical planform wavevector",
                "role": "Maps V1 roll planforms onto Klüver's tunnel, funnel and spiral form constants; the cortical lattice orientation decides which retinal form constant is seen",
                "source_citation": "Ermentrout & Cowan (1979) Biol. Cybern. 34:137; Bressloff et al. (2001) Phil. Trans. R. Soc. B 356:299 §2; Bressloff et al. (2002) Neural Comput. 14:473 Eq. 1.1 (α, β values from upstream params.rs, not the papers)",
                "code_locus": "2001-bressloff-v1-form-constants.py#L129-L173,L982-L1013,L1097-L1118",
            },
            {
                "law_name": "Bressloff et al. (2001) Cubic Amplitude-Equation Roll/Lattice Stability Boundary",
                "formula": "dA₁/dt = A₁ (λ − γ₀ A₁ − 2 γ₃(θ) A₂);  roll fixed point λ = γ₀ A₁;  transverse growth (γ₀ − 2γ₃(θ)) A₁;  roll stable ⇔ γ₀ < 2 γ₃(θ)",
                "lean_decl": "MathOfBiology.amplitude_rhs",
                "lean_theorem": "MathOfBiology.bressloff_roll_stable_iff",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L533-L561",
                "lean_code": (
                    "def amplitude_rhs (lam gamma0 gamma_theta a_self a_other : Int) : Int :=\n"
                    "  lam - gamma0 * a_self - 2 * gamma_theta * a_other\n"
                    "theorem bressloff_roll_fixed_point (gamma0 gamma_theta a : Int) :\n"
                    "    amplitude_rhs (gamma0 * a) gamma0 gamma_theta a 0 = 0 := by simp [amplitude_rhs]\n"
                    "theorem bressloff_roll_transverse_growth (gamma0 gamma_theta a : Int) :\n"
                    "    amplitude_rhs (gamma0 * a) gamma0 gamma_theta 0 a = (gamma0 - 2 * gamma_theta) * a := by\n"
                    "  simp [amplitude_rhs, Int.sub_mul]\n"
                    "theorem bressloff_roll_stable_iff (gamma0 gamma_theta : Int) :\n"
                    "    amplitude_rhs (gamma0 * 1) gamma0 gamma_theta 0 1 < 0 ↔ gamma0 < 2 * gamma_theta := by\n"
                    "  simp [amplitude_rhs]\n"
                    "  omega\n"
                    "theorem bressloff_fig38_roll_unstable :\n"
                    "    amplitude_rhs (3768 * 1) 3768 1869 0 1 = 30 := by decide"
                ),
                "variables": "A_i: squared mode amplitudes; λ: bifurcation parameter (distance past marginal stability); γ₀ = γ₃(0): self-coupling; γ₃(θ): cross-mode cubic coupling at wavevector angle θ (Eq. 5.15)",
                "role": "Selects rolls vs rhombic, square or hexagonal lattices (Figs. 37–40); decides whether a hallucination is a single-family tunnel/spiral or a cobweb/honeycomb",
                "source_citation": "Bressloff et al. (2001) Phil. Trans. R. Soc. B 356:299 §5 Eqs. 5.12–5.15, Figs. 37–40",
                "code_locus": "2001-bressloff-v1-form-constants.py#L672-L697,L765-L916",
            },
            {
                "law_name": "Rule, Stoffregen & Ermentrout (2011) Floquet Resonance Tongues (1:1 Harmonic vs 2:1 Period-Doubling)",
                "formula": "mode after n forcing periods = μⁿ;  μ = +1 (1:1 stripes) → μⁿ = 1;  μ = −1 (2:1 hexagons) → μ^{2n} = 1, μ^{2n+1} = −1",
                "lean_decl": "MathOfBiology.floquet_iterate",
                "lean_theorem": "MathOfBiology.rule_2011_period_doubling_even_odd",
                "lean_locus": "morphoeval/lean/MathOfBiology.lean#L563-L588",
                "lean_code": (
                    "def floquet_iterate (mu : Int) : Nat → Int\n"
                    "  | 0 => 1\n"
                    "  | n + 1 => mu * floquet_iterate mu n\n"
                    "theorem rule_2011_harmonic_returns_each_period (n : Nat) : floquet_iterate 1 n = 1\n"
                    "theorem rule_2011_period_doubling_even_odd (n : Nat) :\n"
                    "    floquet_iterate (-1) (2 * n) = 1 ∧ floquet_iterate (-1) (2 * n + 1) = -1"
                ),
                "variables": "μ: Floquet multiplier of the homogeneous flicker orbit; T: forcing period (ms); β: normalised wave number; n: number of elapsed forcing periods",
                "role": "Separates flicker-induced stripe phosphenes (harmonic tongue) from period-doubled hexagonal phosphenes (subharmonic tongue) across (T, β)",
                "source_citation": "Rule, Stoffregen & Ermentrout (2011) PLoS Comput. Biol. 7(9):e1002158 Fig. 8",
                "code_locus": "2001-bressloff-v1-form-constants.py#L1708-L1724,L2091-L2137",
            },
        ],
    },
}


def verify_lean_module() -> dict[str, Any]:
    """Check `morphoeval/lean/MathOfBiology.lean` with the Lean 4 kernel if `lean` is available."""
    if not LEAN_PATH.exists():
        return {"verified": False, "version": "missing", "elapsed_ms": 0.0, "error": "MathOfBiology.lean not found"}

    lean_text = LEAN_PATH.read_text()
    # Strip block comments `/- ... -/` and line comments `-- ...` before checking for `sorry`
    no_blocks = re.sub(r"/-.*?-/", "", lean_text, flags=re.DOTALL)
    code_lines = [ln.split("--")[0] for ln in no_blocks.splitlines()]
    has_sorry = any(re.search(r"\bsorry\b", ln) for ln in code_lines)
    if has_sorry:
        return {"verified": False, "version": "contains-sorry", "elapsed_ms": 0.0, "error": "Found `sorry` in Lean module"}

    lean_bin = shutil.which("lean") or str(Path.home() / ".elan" / "bin" / "lean")
    if not Path(lean_bin).exists():
        return {"verified": False, "version": "lean-not-installed", "elapsed_ms": 0.0, "error": "lean binary not found"}

    try:
        ver_proc = subprocess.run([lean_bin, "--version"], capture_output=True, text=True, timeout=8)
        if ver_proc.returncode != 0:
            return {"verified": False, "version": "toolchain-pending", "elapsed_ms": 0.0, "error": ver_proc.stderr.strip()}
        version_str = ver_proc.stdout.strip().splitlines()[0]
        t0 = time.perf_counter()
        chk_proc = subprocess.run([lean_bin, str(LEAN_PATH)], capture_output=True, text=True, timeout=30)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if chk_proc.returncode == 0:
            return {"verified": True, "version": version_str, "elapsed_ms": elapsed_ms, "error": ""}
        return {"verified": False, "version": version_str, "elapsed_ms": elapsed_ms, "error": (chk_proc.stdout + chk_proc.stderr).strip()}
    except Exception as exc:
        return {"verified": False, "version": "error", "elapsed_ms": 0.0, "error": str(exc)}


def _to_float(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    s = str(val).strip()
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except ValueError:
        return None


def _extract_p_value(text: str | None) -> float | None:
    if not text:
        return None
    m = P_VALUE_RE.search(text)
    if not m:
        return None
    return _to_float(m.group(1))


def _compute_errors(val_num: float | None, ref_num: float | None) -> tuple[float | None, float | None]:
    if val_num is None or ref_num is None:
        return None, None
    abs_err = abs(val_num - ref_num)
    if abs(ref_num) > 1e-12:
        rel_pct = (abs_err / abs(ref_num)) * 100.0
    else:
        rel_pct = 0.0 if abs_err < 1e-9 else abs_err * 100.0
    return round(abs_err, 6), round(rel_pct, 4)


def load_head_audits() -> dict[str, dict[str, Any]]:
    """Resolve the supersession chain across all primary audits in `morphoeval/audits/`
    and return the latest (head) audit + measurements for each of the 12 simulations."""
    rubric = yaml.safe_load(RUBRIC_PATH.read_text())
    dims = rubric["dimensions"]

    audits: dict[str, tuple[Path, dict[str, Any]]] = {}
    superseded_ids: set[str] = set()
    for p in sorted(AUDITS_DIR.glob("*.yaml")):
        if p.name.endswith(".measurements.yaml"):
            continue
        d = yaml.safe_load(p.read_text())
        if d["audit"].get("role") == "second_assessor":
            continue
        aid = d["audit"]["id"]
        sup = d["audit"].get("supersedes") or d["audit"].get("supersedes_audit_id")
        if sup:
            superseded_ids.add(sup)
        audits[aid] = (p, d)

    ordered_ids = sorted(audits.keys(), key=lambda k: (k not in superseded_ids, k))
    head_by_sim: dict[str, dict[str, Any]] = {}

    for aid in ordered_ids:
        p, d = audits[aid]
        mpath = p.with_name(p.stem + ".measurements.yaml")
        meas_doc = yaml.safe_load(mpath.read_text()) if mpath.exists() else {}
        refs = d.get("references", {})
        cp = d["audit"].get("checkpoint", {})
        repo = cp.get("repository", "https://github.com/astronomeer/infinitemorphospace")
        commit = cp.get("commit", "HEAD")

        for sim in d.get("simulations", []):
            if sim.get("status") != "evaluated":
                continue
            sid = sim["id"]
            scores = sim["scores"]
            overall = round(sum(d_item["weight"] * scores[d_item["id"]]["value"] for d_item in dims) / 100.0, 1)
            sim_meas = (meas_doc.get("measurements") or {}).get(sid, [])
            sim_prov = (meas_doc.get("provenance") or {}).get(sid, {})
            sim_refs = [refs[r] for r in sim.get("source_refs", []) if r in refs]
            dois = [f"https://doi.org/{r['doi']}" for r in sim_refs if r.get("doi")]

            head_by_sim[sid] = {
                "sim_id": sid,
                "audit_id": aid,
                "sim": sim,
                "overall": overall,
                "model_fidelity": scores["model-fidelity"]["value"],
                "specimen_match": scores["specimen-match"]["value"],
                "morphometric_validation": scores["morphometric-validation"]["value"],
                "measurements": sim_meas,
                "provenance": sim_prov,
                "references": sim_refs,
                "dois": dois,
                "repo": repo,
                "commit": commit,
            }

    return head_by_sim


def add_math_triples_for_simulation(
    g: Graph,
    sim_id: str,
    audit_id: str,
    sim: dict[str, Any],
    measurements: list[dict[str, Any]],
    references: list[dict[str, Any]],
    repo: str,
    commit: str,
    overall_score: float,
    lean_verified: bool = True,
) -> tuple[int, int]:
    """Emit first-class RDF triples for a simulation's mathematical laws (`pk:Concept`,
    including both `skos:notation` human formulas and `skos:definition "..."^^pk:lean4`
    machine-checked Lean 4 definitions/theorems) and quantitative morphometric
    measurements (`pk:EvidenceItem` / `schema:QuantitativeValue`) into `g`."""
    claim_uri = PKR[f"claim/morphoeval-{sim_id}"]
    morph_au_uri = PKR[f"assessment/{audit_id}-{sim_id}-morphometric-validation"]
    model_au_uri = PKR[f"assessment/{audit_id}-{sim_id}-model-fidelity"]
    math_root = PKR["concept/math-of-biology"]

    g.add((math_root, RDF.type, PK.Concept))
    g.add((math_root, RDF.type, SKOS.Concept))
    g.add((math_root, RDFS.label, Literal("Validated Mathematics of Biological Form")))

    organism = sim.get("organism", "unspecified organism")
    g.add((claim_uri, RDF.type, PK.Claim))
    g.add((claim_uri, RDFS.label, Literal(f"{sim['name']} ({organism})")))
    g.add((claim_uri, SCHEMA.about, Literal(organism)))
    g.add((claim_uri, PK.weightOfEvidence, Literal(round(overall_score / 100.0, 4), datatype=XSD.double)))

    paper_uris = [URIRef(f"https://doi.org/{r['doi']}") for r in references if r.get("doi")]
    for pu in paper_uris:
        g.add((claim_uri, PK.derivedFromPaper, pu))

    reg = MATH_LAWS_REGISTRY.get(sim_id, {})
    math_family = reg.get("math_family", "Mathematical Morphogenesis Model")
    g.add((claim_uri, PK.hypothesisType, Literal(math_family)))

    # 1. Governing Mathematical Equations / Laws (Human Notation + Lean 4 Formalization)
    eq_list = list(reg.get("equations", []))
    cb = sim.get("claim_boundary") or {}
    n_eq = 0
    for idx, eq in enumerate(eq_list, start=1):
        eq_uri = PKR[f"math-law/{sim_id}-{idx}"]
        g.add((eq_uri, RDF.type, PK.Concept))
        g.add((eq_uri, RDF.type, SKOS.Concept))
        g.add((eq_uri, SKOS.broader, math_root))
        g.add((math_root, SKOS.narrower, eq_uri))
        g.add((claim_uri, DCTERMS.conformsTo, eq_uri))
        g.add((model_au_uri, DCTERMS.references, eq_uri))

        g.add((eq_uri, RDFS.label, Literal(eq["law_name"])))
        g.add((eq_uri, SKOS.notation, Literal(eq["formula"])))
        g.add((eq_uri, PK.hypothesisType, Literal(math_family)))
        g.add((eq_uri, DCTERMS.description, Literal(f"{eq['role']} | Variables: {eq['variables']}")))
        g.add((eq_uri, DCTERMS.source, Literal(eq["source_citation"])))

        # Formal Lean 4 layer
        if eq.get("lean_code"):
            g.add((eq_uri, SKOS.definition, Literal(eq["lean_code"], datatype=PK.lean4)))
            g.add((eq_uri, SCHEMA.programmingLanguage, Literal("Lean 4")))
        if eq.get("lean_decl"):
            g.add((eq_uri, PK.leanDeclaration, Literal(eq["lean_decl"])))
        if eq.get("lean_theorem"):
            g.add((eq_uri, PK.leanTheorem, Literal(eq["lean_theorem"])))
        g.add((eq_uri, PK.leanVerified, Literal(bool(lean_verified), datatype=XSD.boolean)))

        if eq.get("code_locus"):
            locus_url = f"{repo}/blob/{commit}/{eq['code_locus']}"
            g.add((eq_uri, RDFS.seeAlso, URIRef(locus_url)))
        if eq.get("lean_locus"):
            lean_url = f"{repo}/blob/{commit}/{eq['lean_locus']}"
            g.add((eq_uri, RDFS.seeAlso, URIRef(lean_url)))
        for pu in paper_uris:
            g.add((eq_uri, PROV.wasDerivedFrom, pu))
        n_eq += 1

    # Also attach structured claim_boundary statements if present
    for key in ("reported_in_source", "reconstructed_in_script", "transfer_boundary", "does_not_claim"):
        for item in cb.get(key) or []:
            g.add((claim_uri, SCHEMA.additionalProperty, Literal(f"{key}: {' '.join(str(item).split())}")))

    # 2. Validated Quantitative Morphometric Measurements (`pk:EvidenceItem`)
    n_meas = 0
    for idx, m in enumerate(measurements, start=1):
        if m.get("value") is None:
            continue
        ev_uri = PKR[f"evidence/{audit_id}-{sim_id}-{idx}"]
        g.add((ev_uri, RDF.type, PK.EvidenceItem))
        g.add((ev_uri, RDF.type, SCHEMA.QuantitativeValue))
        g.add((claim_uri, PK.hasEvidence, ev_uri))
        g.add((morph_au_uri, PK.hasEvidence, ev_uri))

        m_name = str(m["name"])
        g.add((ev_uri, RDFS.label, Literal(m_name)))
        g.add((ev_uri, SCHEMA.about, Literal(organism)))

        val_raw = m["value"]
        val_num = _to_float(val_raw)
        if val_num is not None:
            g.add((ev_uri, SCHEMA.value, Literal(val_num, datatype=XSD.double)))
        else:
            g.add((ev_uri, SCHEMA.value, Literal(str(val_raw))))

        if m.get("unit"):
            g.add((ev_uri, SCHEMA.unitText, Literal(str(m["unit"]))))

        ref_raw = m.get("reference_value")
        ref_num = _to_float(ref_raw)
        if ref_num is not None:
            g.add((ev_uri, SCHEMA.targetValue, Literal(ref_num, datatype=XSD.double)))
        elif ref_raw is not None:
            g.add((ev_uri, SCHEMA.targetValue, Literal(str(ref_raw))))

        abs_err, rel_pct = _compute_errors(val_num, ref_num)
        if rel_pct is not None:
            g.add((ev_uri, SCHEMA.error, Literal(rel_pct, datatype=XSD.double)))

        if m.get("method"):
            g.add((ev_uri, DCTERMS.description, Literal(" ".join(str(m["method"]).split()))))
        if m.get("reference_source"):
            g.add((ev_uri, DCTERMS.source, Literal(" ".join(str(m["reference_source"]).split()))))
        if m.get("derived_from"):
            g.add((ev_uri, PROV.wasGeneratedBy, Literal(str(m["derived_from"]))))
        if m.get("note"):
            g.add((ev_uri, RDFS.comment, Literal(" ".join(str(m["note"]).split()))))

        pval = _extract_p_value(str(m.get("note") or "")) or _extract_p_value(str(m.get("method") or ""))
        if pval is not None:
            g.add((ev_uri, PK.pValue, Literal(pval, datatype=XSD.double)))
            g.add((ev_uri, PK.statisticalTest, Literal(str(m.get("note") or m.get("method")))))

        n_meas += 1

    return n_eq, n_meas


def build_math_knowledge_graph() -> dict[str, Any]:
    """Verify `morphoeval/lean/MathOfBiology.lean` with the Lean 4 kernel, build
    `morphoeval/kg/math_of_biology.ttl` and `.trig`, and populate `math_models`,
    `math_equations`, and `math_measurements` in `morphoeval/runs/telemetry.sqlite`."""
    lean_info = verify_lean_module()
    lean_ok = bool(lean_info["verified"])
    heads = load_head_audits()

    g = Graph()
    for pfx, ns in (
        ("pk", PK),
        ("pkr", PKR),
        ("schema", SCHEMA),
        ("dct", DCTERMS),
        ("prov", PROV),
        ("rdfs", RDFS),
        ("skos", SKOS),
        ("xsd", XSD),
    ):
        g.bind(pfx, ns)

    for sid, info in sorted(heads.items()):
        aid = info["audit_id"]
        for trig_file in sorted((KG_DIR / aid).glob(f"{sid}-*.trig")):
            ds_tmp = Dataset()
            ds_tmp.parse(trig_file, format="trig")
            for ctx in ds_tmp.graphs():
                if str(ctx.identifier).endswith("assertion"):
                    for t in ctx:
                        g.add(t)

    model_rows: list[tuple[Any, ...]] = []
    eq_rows: list[tuple[Any, ...]] = []
    meas_rows: list[tuple[Any, ...]] = []

    total_eq = 0
    total_meas = 0

    for sid, info in sorted(heads.items()):
        aid = info["audit_id"]
        sim = info["sim"]
        meas = info["measurements"]
        refs = info["references"]
        repo = info["repo"]
        commit = info["commit"]
        overall = info["overall"]
        reg = MATH_LAWS_REGISTRY.get(sid, {})
        math_family = reg.get("math_family", "Mathematical Morphogenesis Model")

        neq, nmeas = add_math_triples_for_simulation(
            g=g,
            sim_id=sid,
            audit_id=aid,
            sim=sim,
            measurements=meas,
            references=refs,
            repo=repo,
            commit=commit,
            overall_score=overall,
            lean_verified=lean_ok,
        )
        total_eq += neq
        total_meas += nmeas

        per_audit_ttl = KG_DIR / aid / f"morphoeval_{aid}.ttl"
        if per_audit_ttl.exists():
            ag = Graph()
            for pfx, ns in (("pk", PK), ("pkr", PKR), ("schema", SCHEMA), ("dct", DCTERMS), ("prov", PROV), ("rdfs", RDFS), ("skos", SKOS), ("xsd", XSD)):
                ag.bind(pfx, ns)
            ag.parse(per_audit_ttl, format="turtle")
            add_math_triples_for_simulation(ag, sid, aid, sim, meas, refs, repo, commit, overall, lean_verified=lean_ok)
            per_audit_ttl.write_text(ag.serialize(format="turtle"))

        model_rows.append(
            (
                sid,
                sim["name"],
                sim.get("organism", ""),
                sim["script"],
                aid,
                math_family,
                overall,
                info["model_fidelity"],
                info["morphometric_validation"],
                info["specimen_match"],
                str(PKR[f"claim/morphoeval-{sid}"]),
                ", ".join(info["dois"]),
                (info["provenance"] or {}).get("spec_fingerprint", ""),
            )
        )

        for idx, eq in enumerate(reg.get("equations", []), start=1):
            eq_rows.append(
                (
                    f"{sid}-eq-{idx}",
                    sid,
                    sim.get("organism", ""),
                    eq["law_name"],
                    eq["formula"],
                    eq.get("lean_decl", ""),
                    eq.get("lean_theorem", ""),
                    eq.get("lean_code", ""),
                    eq.get("lean_locus", ""),
                    1 if lean_ok else 0,
                    math_family,
                    eq["variables"],
                    eq["role"],
                    eq["source_citation"],
                    eq.get("code_locus", ""),
                )
            )

        for idx, m in enumerate(meas, start=1):
            if m.get("value") is None:
                continue
            val_num = _to_float(m["value"])
            ref_num = _to_float(m.get("reference_value"))
            abs_err, rel_pct = _compute_errors(val_num, ref_num)
            pval = _extract_p_value(str(m.get("note") or "")) or _extract_p_value(str(m.get("method") or ""))
            within_5 = 1 if (rel_pct is not None and rel_pct <= 5.0) else (0 if rel_pct is not None else None)
            meas_rows.append(
                (
                    f"{aid}-{sid}-m{idx}",
                    sid,
                    aid,
                    sim.get("organism", ""),
                    str(m["name"]),
                    val_num,
                    str(m["value"]),
                    str(m.get("unit") or ""),
                    ref_num,
                    str(m.get("reference_value")) if m.get("reference_value") is not None else None,
                    abs_err,
                    rel_pct,
                    within_5,
                    " ".join(str(m.get("method") or "").split()),
                    str(m.get("derived_from") or ""),
                    " ".join(str(m.get("reference_source") or "").split()),
                    pval,
                    " ".join(str(m.get("note") or "").split()),
                )
            )

    MATH_TTL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATH_TTL_PATH.write_text(g.serialize(format="turtle"))

    ds = Dataset()
    for pfx, ns in (("pk", PK), ("pkr", PKR), ("schema", SCHEMA), ("dct", DCTERMS), ("prov", PROV), ("rdfs", RDFS), ("skos", SKOS), ("xsd", XSD), ("np", NP)):
        ds.bind(pfx, ns)
    asrt_g = ds.graph(URIRef("https://w3id.org/probknow/resource/graph/math-of-biology-assertion"))
    for t in g:
        asrt_g.add(t)
    MATH_TRIG_PATH.write_text(ds.serialize(format="trig"))

    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    with conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS math_models;
            DROP TABLE IF EXISTS math_equations;
            DROP TABLE IF EXISTS math_measurements;

            CREATE TABLE math_models (
                sim_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                organism TEXT NOT NULL,
                script TEXT NOT NULL,
                head_audit_id TEXT NOT NULL,
                math_family TEXT NOT NULL,
                overall_score REAL NOT NULL,
                model_fidelity_score INTEGER NOT NULL,
                morphometric_score INTEGER NOT NULL,
                specimen_match_score INTEGER NOT NULL,
                claim_uri TEXT NOT NULL,
                dois TEXT,
                spec_fingerprint TEXT
            );

            CREATE TABLE math_equations (
                eq_id TEXT PRIMARY KEY,
                sim_id TEXT NOT NULL,
                organism TEXT NOT NULL,
                law_name TEXT NOT NULL,
                formula TEXT NOT NULL,
                lean_decl TEXT NOT NULL,
                lean_theorem TEXT NOT NULL,
                lean_code TEXT NOT NULL,
                lean_locus TEXT NOT NULL,
                lean_verified INTEGER NOT NULL,
                math_family TEXT NOT NULL,
                variables TEXT NOT NULL,
                role TEXT NOT NULL,
                source_citation TEXT NOT NULL,
                code_locus TEXT
            );

            CREATE TABLE math_measurements (
                meas_id TEXT PRIMARY KEY,
                sim_id TEXT NOT NULL,
                audit_id TEXT NOT NULL,
                organism TEXT NOT NULL,
                name TEXT NOT NULL,
                value_num REAL,
                value_raw TEXT NOT NULL,
                unit TEXT,
                ref_num REAL,
                ref_raw TEXT,
                abs_error REAL,
                rel_error_pct REAL,
                within_5pct INTEGER,
                method TEXT,
                derived_from TEXT,
                reference_source TEXT,
                p_value REAL,
                note TEXT
            );
            """
        )
        conn.executemany(
            "INSERT INTO math_models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            model_rows,
        )
        conn.executemany(
            "INSERT INTO math_equations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            eq_rows,
        )
        conn.executemany(
            "INSERT INTO math_measurements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            meas_rows,
        )
    conn.close()

    return {
        "num_models": len(model_rows),
        "num_equations": total_eq,
        "num_measurements": total_meas,
        "num_triples": len(g),
        "lean_info": lean_info,
        "ttl_path": str(MATH_TTL_PATH),
        "trig_path": str(MATH_TRIG_PATH),
        "sqlite_path": str(SQLITE_PATH),
    }


# --------------------------------------------------------------------------- #
# Built-in SPARQL Presets
# --------------------------------------------------------------------------- #
SPARQL_PRESETS: dict[str, tuple[str, str]] = {
    "equations": (
        "All governing mathematical equations of biology, their formulas, Lean 4 theorems, and organisms",
        """
PREFIX pk:     <https://w3id.org/probknow/ontology/>
PREFIX schema: <https://schema.org/>
PREFIX skos:   <http://www.w3.org/2004/02/skos/core#>
PREFIX dct:    <http://purl.org/dc/terms/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?organism ?law ?formula ?leanTheorem ?verified
WHERE {
  ?claim a pk:Claim ;
         schema:about ?organism ;
         dct:conformsTo ?eq .
  ?eq a pk:Concept ;
      rdfs:label ?law ;
      skos:notation ?formula ;
      pk:leanTheorem ?leanTheorem ;
      pk:leanVerified ?verified .
}
ORDER BY ?organism ?law
""",
    ),
    "lean-proofs": (
        "All 19 formal Lean 4 declarations & machine-checked theorems (`skos:definition` typed as `pk:lean4`)",
        """
PREFIX pk:     <https://w3id.org/probknow/ontology/>
PREFIX schema: <https://schema.org/>
PREFIX skos:   <http://www.w3.org/2004/02/skos/core#>
PREFIX dct:    <http://purl.org/dc/terms/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?organism ?leanDecl ?leanTheorem ?verified ?leanCode
WHERE {
  ?claim a pk:Claim ;
         schema:about ?organism ;
         dct:conformsTo ?eq .
  ?eq a pk:Concept ;
      pk:leanDeclaration ?leanDecl ;
      pk:leanTheorem ?leanTheorem ;
      pk:leanVerified ?verified ;
      skos:definition ?leanCode .
  FILTER(DATATYPE(?leanCode) = pk:lean4)
}
ORDER BY ?organism ?leanDecl
""",
    ),
    "exact-matches": (
        "Validated morphometric measurements within 1.5% relative error of published empirical/analytical references",
        """
PREFIX pk:     <https://w3id.org/probknow/ontology/>
PREFIX schema: <https://schema.org/>
PREFIX dct:    <http://purl.org/dc/terms/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?organism ?measurement ?val ?ref ?unit ?errPct ?source
WHERE {
  ?ev a pk:EvidenceItem ;
      schema:about ?organism ;
      rdfs:label ?measurement ;
      schema:value ?val ;
      schema:targetValue ?ref ;
      schema:error ?errPct ;
      dct:source ?source .
  OPTIONAL { ?ev schema:unitText ?unit }
  FILTER(?errPct <= 1.5)
}
ORDER BY ?errPct ?organism
""",
    ),
    "by-family": (
        "Biological systems grouped by mathematical family, with overall Weight-of-Evidence and equation count",
        """
PREFIX pk:     <https://w3id.org/probknow/ontology/>
PREFIX schema: <https://schema.org/>
PREFIX dct:    <http://purl.org/dc/terms/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?family ?organism ?woe (COUNT(DISTINCT ?eq) AS ?equations) (COUNT(DISTINCT ?ev) AS ?measurements)
WHERE {
  ?claim a pk:Claim ;
         schema:about ?organism ;
         pk:hypothesisType ?family ;
         pk:weightOfEvidence ?woe ;
         dct:conformsTo ?eq .
  OPTIONAL { ?claim pk:hasEvidence ?ev }
}
GROUP BY ?family ?organism ?woe
ORDER BY DESC(?woe)
""",
    ),
    "p-values": (
        "Empirical statistical hypothesis tests (p-values) attached to morphometric evidence items",
        """
PREFIX pk:     <https://w3id.org/probknow/ontology/>
PREFIX schema: <https://schema.org/>
PREFIX dct:    <http://purl.org/dc/terms/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?organism ?measurement ?val ?ref ?pValue ?source
WHERE {
  ?ev a pk:EvidenceItem ;
      schema:about ?organism ;
      rdfs:label ?measurement ;
      schema:value ?val ;
      pk:pValue ?pValue ;
      dct:source ?source .
  OPTIONAL { ?ev schema:targetValue ?ref }
}
ORDER BY ?pValue
""",
    ),
}


def run_sparql(query: str) -> None:
    if not MATH_TTL_PATH.exists():
        build_math_knowledge_graph()
    g = Graph()
    g.parse(MATH_TTL_PATH, format="turtle")
    res = g.query(query)
    vars_list = [str(v) for v in res.vars]
    rows = [[(" ".join(str(row[i]).split())) if row[i] is not None else "—" for i in range(len(vars_list))] for row in res]
    if not rows:
        print("(0 rows returned)")
        return
    widths = [min(56, max(len(vars_list[i]), *(len(r[i]) for r in rows))) for i in range(len(vars_list))]
    hdr = " | ".join(vars_list[i].ljust(widths[i])[: widths[i]] for i in range(len(vars_list)))
    sep = "-+-".join("-" * widths[i] for i in range(len(vars_list)))
    print(hdr)
    print(sep)
    for r in rows:
        print(" | ".join(r[i].ljust(widths[i])[: widths[i]] for i in range(len(vars_list))))
    print(f"\n({len(rows)} row(s) returned from {MATH_TTL_PATH.relative_to(ROOT)})")


def run_sql(query: str) -> None:
    if not SQLITE_PATH.exists():
        build_math_knowledge_graph()
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.execute(query)
    cols = [d[0] for d in (cur.description or [])]
    rows = [[(" ".join(str(v).split())) if v is not None else "—" for v in r] for r in cur.fetchall()]
    conn.close()
    if not rows:
        print("(0 rows returned)")
        return
    widths = [min(56, max(len(cols[i]), *(len(r[i]) for r in rows))) for i in range(len(cols))]
    print(" | ".join(cols[i].ljust(widths[i])[: widths[i]] for i in range(len(cols))))
    print("-+-".join("-" * widths[i] for i in range(len(cols))))
    for r in rows:
        print(" | ".join(r[i].ljust(widths[i])[: widths[i]] for i in range(len(cols))))
    print(f"\n({len(rows)} row(s) returned from {SQLITE_PATH.relative_to(ROOT)})")


def print_sim_card(sim_id: str) -> None:
    if not SQLITE_PATH.exists():
        build_math_knowledge_graph()
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    m = conn.execute("SELECT * FROM math_models WHERE sim_id = ?", (sim_id,)).fetchone()
    if not m:
        print(f"Unknown sim_id '{sim_id}'. Available: {[r[0] for r in conn.execute('SELECT sim_id FROM math_models').fetchall()]}")
        conn.close()
        return
    eqs = conn.execute("SELECT * FROM math_equations WHERE sim_id = ?", (sim_id,)).fetchall()
    meas = conn.execute("SELECT * FROM math_measurements WHERE sim_id = ?", (sim_id,)).fetchall()
    conn.close()

    print("=" * 104)
    print(f"MODEL: {m['name']}  ({m['sim_id']})")
    print(f"Organism:     {m['organism']}")
    print(f"Math Family:  {m['math_family']}")
    print(f"Script:       {m['script']}  (Audit: {m['head_audit_id']})")
    print(f"Scores:       Overall {m['overall_score']}/100 | Model Fidelity {m['model_fidelity_score']} | Morphometrics {m['morphometric_score']} | Specimen Match {m['specimen_match_score']}")
    print(f"Claim URI:    {m['claim_uri']}")
    print(f"Paper DOIs:   {m['dois'] or '—'}")
    print("-" * 104)
    print(f"GOVERNING MATHEMATICAL EQUATIONS & LEAN 4 PROOFS ({len(eqs)}):")
    for idx, e in enumerate(eqs, start=1):
        status_badge = "✓ LEAN 4 KERNEL VERIFIED" if e["lean_verified"] else "PENDING LEAN KERNEL"
        print(f"  [{idx}] {e['law_name']}  [{status_badge}]")
        print(f"      Notation (skos:notation):   {e['formula']}")
        print(f"      Lean 4 Decl / Theorem:      {e['lean_decl']}  ->  {e['lean_theorem']}  ({e['lean_locus']})")
        for ln in e["lean_code"].splitlines():
            print(f"        | {ln}")
        print(f"      Variables:                  {e['variables']}")
        print(f"      Role:                       {e['role']}")
        print(f"      Source:                     {e['source_citation']}  (Python: {e['code_locus']})")
    print("-" * 104)
    print(f"VALIDATED QUANTITATIVE MEASUREMENTS ({len(meas)}):")
    for idx, ms in enumerate(meas, start=1):
        err_str = f" [rel err: {ms['rel_error_pct']:.3f}%]" if ms["rel_error_pct"] is not None else ""
        ref_str = f" (ref: {ms['ref_raw']})" if ms["ref_raw"] is not None else ""
        p_str = f" [p = {ms['p_value']:.2e}]" if ms["p_value"] is not None else ""
        print(f"  ({idx:2d}) {ms['name']}: {ms['value_raw']} {ms['unit']}{ref_str}{err_str}{p_str}")
        if ms["reference_source"]:
            print(f"       Source: {ms['reference_source']} | Method: {ms['method']}")
    print("=" * 104)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="Verify Lean 4 module and rebuild morphoeval/kg/math_of_biology.{ttl,trig} + SQLite tables")
    ap.add_argument("--lean", action="store_true", help="Run Lean 4 kernel check on morphoeval/lean/MathOfBiology.lean and list all 19 formal theorems")
    ap.add_argument("--equations", action="store_true", help="List all governing mathematical equations across all 12 organisms")
    ap.add_argument("--measurements", action="store_true", help="List validated morphometric measurements")
    ap.add_argument("--max-error-pct", type=float, default=None, help="Filter measurements by max relative error %%")
    ap.add_argument("--sim", type=str, default=None, help="Inspect full math, Lean 4 theorem & measurement card for one simulation ID")
    ap.add_argument("--preset", choices=sorted(SPARQL_PRESETS.keys()), default=None, help="Run a built-in SPARQL query preset")
    ap.add_argument("--sparql", type=str, default=None, help="Execute a custom SPARQL query on morphoeval/kg/math_of_biology.ttl")
    ap.add_argument("--sql", type=str, default=None, help="Execute a custom SQL query on morphoeval/runs/telemetry.sqlite")
    args = ap.parse_args()

    if args.build or not MATH_TTL_PATH.exists():
        stats = build_math_knowledge_graph()
        li = stats["lean_info"]
        lean_msg = (
            f"Lean 4 VERIFIED ({li['version']} in {li['elapsed_ms']}ms, 0 sorry)"
            if li["verified"]
            else f"Lean 4 status: {li['version']} ({li['error']})"
        )
        print(
            f"[math_kg] Built Math of Biology KG: {stats['num_models']} biological models, "
            f"{stats['num_equations']} governing equations ({lean_msg}), "
            f"{stats['num_measurements']} quantitative measurements, "
            f"{stats['num_triples']} RDF triples -> {stats['ttl_path']} & {stats['sqlite_path']}"
        )

    if args.lean:
        li = verify_lean_module()
        print(f"=== Lean 4 Kernel Verification ({LEAN_PATH.relative_to(ROOT)}) ===")
        print(f"Status:  {'VERIFIED (exit 0, 0 sorry)' if li['verified'] else 'FAILED/PENDING'}")
        print(f"Kernel:  {li['version']}  ({li['elapsed_ms']} ms)")
        if li["error"]:
            print(f"Detail:  {li['error']}")
        print("-" * 100)
        run_sql(
            "SELECT sim_id, law_name, lean_decl, lean_theorem, lean_locus, lean_verified FROM math_equations ORDER BY sim_id, eq_id"
        )
        return

    if args.sim:
        print_sim_card(args.sim)
        return

    if args.preset:
        desc, q = SPARQL_PRESETS[args.preset]
        print(f"=== SPARQL Preset '{args.preset}': {desc} ===")
        print(q.strip())
        print("-" * 90)
        run_sparql(q)
        return

    if args.sparql:
        run_sparql(args.sparql)
        return

    if args.sql:
        run_sql(args.sql)
        return

    if args.equations:
        run_sql(
            "SELECT sim_id, law_name, formula, lean_theorem, lean_verified, code_locus FROM math_equations ORDER BY sim_id, eq_id"
        )
        return

    if args.measurements:
        if args.max_error_pct is not None:
            run_sql(
                f"SELECT sim_id, name, value_raw, unit, ref_raw, rel_error_pct, reference_source "
                f"FROM math_measurements WHERE rel_error_pct IS NOT NULL AND rel_error_pct <= {float(args.max_error_pct)} "
                f"ORDER BY rel_error_pct, sim_id"
            )
        else:
            run_sql(
                "SELECT sim_id, name, value_raw, unit, ref_raw, rel_error_pct "
                "FROM math_measurements ORDER BY sim_id, meas_id"
            )
        return

    # Default summary view
    run_sql(
        """
        SELECT m.sim_id,
               substr(m.organism, 1, 28) AS organism,
               m.overall_score AS score,
               (SELECT COUNT(*) FROM math_equations e WHERE e.sim_id = m.sim_id) AS eqs,
               (SELECT SUM(e.lean_verified) FROM math_equations e WHERE e.sim_id = m.sim_id) AS lean_ok,
               (SELECT COUNT(*) FROM math_measurements ms WHERE ms.sim_id = m.sim_id) AS meas,
               (SELECT COUNT(*) FROM math_measurements ms WHERE ms.sim_id = m.sim_id AND ms.within_5pct = 1) AS within_5pct
        FROM math_models m
        ORDER BY m.overall_score DESC
        """
    )


if __name__ == "__main__":
    main()
