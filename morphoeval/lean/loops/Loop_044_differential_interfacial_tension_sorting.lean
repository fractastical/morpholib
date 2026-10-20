/-
  morphoeval/lean/loops/Loop_044_differential_interfacial_tension_sorting.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 044:
  Brodland (2002) J. Biomech. Eng. 124:188-197, Krieg et al. (2008) Nature Cell Biology
  10:429-436, and Maître et al. (2012) Science 338:253-256 Differential Interfacial
  Tension Hypothesis (DITH) & Zebrafish Germ-Layer Cortical Tension Sorting
  (`brodland-differential-interfacial-tension-sorting`).

  Pure core Lean 4 (zero external imports, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_044_differential_interfacial_tension_sorting.lean
-/

namespace MathOfBiology.Loop_044

/-- Brodland (2002) J. Biomech. Eng. Eq. (1) Differential Interfacial Tension Hypothesis (DITH):
    effective interfacial tension `γ_{αβ} = β_α^{cort} + β_β^{cort} - ω_{αβ}^{adh}`
    along a contact interface between cell types `α` and `β`. -/
def dith_gamma_ab (beta_a beta_b omega_ab : Int) : Int :=
  beta_a + beta_b - omega_ab

/-- Cell-medium spreading tension difference `γ_{M0} - γ_{E0} = β_M^{cort} - β_E^{cort}`
    governing whether low-tension ectoderm (`E`) envelops high-tension mesoderm/endoderm (`M`)
    in medium (`0`) (Brodland 2002; Krieg et al. 2008 Nat. Cell Biol. 10:429). -/
def dith_spreading_diff (beta_E beta_M : Int) : Int :=
  beta_M - beta_E

/-- Theorem 1 (`dith_blebbistatin_equal_cortex_null_sorting`):
    Under Blebbistatin Myosin-II inhibition where cortical tensions equalize (`β_E = β_M = β`)
    and contact tension reduction balances `2 * β`, both the heterotypic interfacial tension
    `γ_{EM}` and the cell-medium spreading tension difference `γ_{M0} - γ_{E0}` vanish
    identically for all integer cortical tensions `β`. -/
theorem dith_blebbistatin_equal_cortex_null_sorting (beta : Int) :
    dith_gamma_ab beta beta (2 * beta) = 0 ∧ dith_spreading_diff beta beta = 0 := by
  constructor
  · dsimp [dith_gamma_ab]
    omega
  · dsimp [dith_spreading_diff]
    omega

/-- Theorem 2 (`krieg_2008_germ_layer_hierarchy_and_camlc_reversal`):
    Machine-checked verification of Krieg et al. (2008) Nature Cell Biology 10:429-436
    zebrafish germ-layer AFM cortical tension hierarchy and `caMLCK` sorting reversal:
    (1) Wild-type cortical tension hierarchy: `β_E = 50 < β_M = 95 < β_N = 130` (pN/μm),
        with `γ_{EM} = 30` and `γ_{M0} - γ_{E0} = 45`, satisfying `0 < γ_{EM} < γ_{M0} - γ_{E0}`
        (ectoderm envelops mesoderm/endoderm core);
    (2) `caMLCK` ectoderm cortical stiffening (`β_{E+caMLCK} = 145 > β_M = 95`):
        `γ_{M0} - γ_{E0} = -50 < 0` and `0 < γ_{EM} = 30 < γ_{E0} - γ_{M0} = 50`,
        reversing the engulfment hierarchy so stiffened ectoderm sorts inside mesoderm;
    (3) High-HIT partial doublet lens (`γ_{EM} = 70 > γ_{M0} - γ_{E0} = 45 > 0`). -/
theorem krieg_2008_germ_layer_hierarchy_and_camlc_reversal :
    (50 < 95 ∧ 95 < 130) ∧
    dith_gamma_ab 50 95 115 = 30 ∧
    dith_spreading_diff 50 95 = 45 ∧
    (0 < dith_gamma_ab 50 95 115 ∧ dith_gamma_ab 50 95 115 < dith_spreading_diff 50 95) ∧
    dith_spreading_diff 145 95 = -50 ∧
    (0 < dith_gamma_ab 145 95 210 ∧ dith_gamma_ab 145 95 210 < dith_spreading_diff 95 145) ∧
    dith_gamma_ab 50 95 75 > dith_spreading_diff 50 95 := by
  decide

/-- Scaled Young–Dupré cell-doublet contact angle numerator
    `2 * γ_{cm} * cos(θ) = 2 * γ_{cm} - γ_{cc}`
    from Maître et al. (2012) Science 338:253-256 Eq. (1). -/
def young_dupre_num (gamma_cm gamma_cc : Int) : Int :=
  2 * gamma_cm - gamma_cc

/-- Theorem 3 (`maitre_2012_young_dupre_contact_bounds`):
    Proves that:
    (a) in the non-adhering detachment limit `γ_{cc} = 2 * γ_{cm}`, the Young–Dupré
        contact numerator vanishes (`2 * γ_{cm} * cos(θ) = 0`),
    (b) in the complete cortical tension reduction limit `γ_{cc} = 0`, it attains its
        maximum `2 * γ_{cm}` (`cos(θ) = 1`), and
    (c) increasing cell-cell interfacial tension by `dγ` reduces the contact numerator
        by exactly `dγ`. -/
theorem maitre_2012_young_dupre_contact_bounds (gamma_cm gamma_cc dgamma : Int) :
    young_dupre_num gamma_cm (2 * gamma_cm) = 0 ∧
    young_dupre_num gamma_cm 0 = 2 * gamma_cm ∧
    young_dupre_num gamma_cm (gamma_cc + dgamma) = young_dupre_num gamma_cm gamma_cc - dgamma := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [young_dupre_num]
    omega
  · dsimp [young_dupre_num]
    omega
  · dsimp [young_dupre_num]
    omega

/-! ### Pass 3 — time-dependent Nodal/TGF-β → cortical tension coupling -/

/-- Sharp-threshold (`n → ∞`) limit of the Nodal dose → cortical tension target used in
    `nodal_to_cortical_tension_target`: Nodal-blocked cells (`C < K_M`, lefty1 / MZoep) take the
    ectoderm tension 50, intermediate dose (`K_M ≤ C < K_N`, low cyclops) the mesoderm tension 95,
    saturating dose (`C ≥ K_N`, cyclops + casanova) the endoderm tension 130 pN/μm
    (Krieg et al. 2008 Nat. Cell Biol. Fig. 1; Gritsman et al. 1999 Cell 97:121). -/
def nodal_tension_target (C K_M K_N : Int) : Int :=
  if C < K_M then 50 else if C < K_N then 95 else 130

/-- Theorem 4 (`nodal_dose_cortical_tension_monotone_bounded`):
    for any thresholds `K_M, K_N` with `0 < K_M`, the Nodal dose → cortical tension map is
    (a) monotone in the dose (`C₁ ≤ C₂ → β*(C₁) ≤ β*(C₂)`),
    (b) bounded by the ectoderm and endoderm AFM tensions `50 ≤ β* ≤ 130`, and
    (c) returns the ectoderm value in the Nodal-null limit (`C = 0 < K_M`), so two Nodal-null
        cells have zero DITH spreading difference (no germ-layer sorting, MZoep phenotype). -/
theorem nodal_dose_cortical_tension_monotone_bounded (C₁ C₂ K_M K_N : Int)
    (hC : C₁ ≤ C₂) (h0 : 0 < K_M) :
    nodal_tension_target C₁ K_M K_N ≤ nodal_tension_target C₂ K_M K_N ∧
    (50 ≤ nodal_tension_target C₁ K_M K_N ∧ nodal_tension_target C₁ K_M K_N ≤ 130) ∧
    dith_spreading_diff (nodal_tension_target 0 K_M K_N) (nodal_tension_target 0 K_M K_N) = 0 ∧
    nodal_tension_target 0 K_M K_N = 50 := by
  unfold nodal_tension_target dith_spreading_diff
  refine ⟨?_, ?_, ?_, ?_⟩
  · split <;> (try split) <;> (try split) <;> (try split) <;> omega
  · split <;> (try split) <;> omega
  · split <;> (try split) <;> omega
  · split <;> (try split) <;> omega

/-- Continuous-cortex DITH closure used by `simulate_nodal_coupled_internalization`, scaled by 3
    so that `κ = 2/3` is exact over the integers: `3 γ_{ij} = 2 |β_i − β_j|`
    (i.e. Brodland's `γ_{ij} = β_i + β_j − ω_{ij}` with `ω_{ij} = β_i + β_j − γ_{ij}`). -/
def cortex_mismatch_gamma3 (beta_i beta_j : Int) : Int :=
  2 * (if beta_i ≥ beta_j then beta_i - beta_j else beta_j - beta_i)

/-- Theorem 5 (`cortex_mismatch_closure_satisfies_engulfment`):
    whenever Nodal has differentiated two cortices (`β_i ≠ β_j`), the closure gives a strictly
    positive heterotypic tension that is strictly smaller than the cell–medium spreading
    difference `|γ_{i0} − γ_{j0}| = |β_i − β_j|` — Krieg et al. (2008) regime I
    (`0 < γ_{EM} < γ_{M0} − γ_{E0}`): the higher-tension (mesendoderm) cell is engulfed by the
    lower-tension (ectoderm) cell. At equal tensions the heterotypic tension vanishes
    (Blebbistatin-like null), and the wild-type anchor `β_E = 50, β_M = 95` returns
    `3 γ_{EM} = 90`, i.e. `γ_{EM} = 30` pN/μm. -/
theorem cortex_mismatch_closure_satisfies_engulfment (beta_i beta_j : Int) :
    (beta_i ≠ beta_j →
      0 < cortex_mismatch_gamma3 beta_i beta_j ∧
      cortex_mismatch_gamma3 beta_i beta_j < 3 * (if beta_i ≥ beta_j then beta_i - beta_j else beta_j - beta_i)) ∧
    cortex_mismatch_gamma3 beta_i beta_i = 0 ∧
    cortex_mismatch_gamma3 50 95 = 90 := by
  unfold cortex_mismatch_gamma3
  refine ⟨?_, ?_, ?_⟩
  · intro hne
    constructor
    · split <;> omega
    · split <;> omega
  · split <;> omega
  · decide

/-- One explicit-Euler step of the cortical tension relaxation ODE
    `dβ/dt = (β* − β)/τ_β` with `dt/τ_β = 1/4` (integer-scaled). -/
def cortex_relax_step (beta target : Int) : Int :=
  beta + (target - beta) / 4

/-- Theorem 6 (`cortical_tension_relaxation_contracts`):
    the relaxation step never overshoots: the updated tension lies between the current tension
    and the Nodal-set target (both orderings), so tensions stay inside `[β_E, β_N]` whenever the
    target does — the invariant checked numerically in `self_test` check 10. -/
theorem cortical_tension_relaxation_contracts (beta target : Int) :
    (beta ≤ target → beta ≤ cortex_relax_step beta target ∧ cortex_relax_step beta target ≤ target) ∧
    (target ≤ beta → target ≤ cortex_relax_step beta target ∧ cortex_relax_step beta target ≤ beta) := by
  unfold cortex_relax_step
  constructor
  · intro h
    omega
  · intro h
    omega

end MathOfBiology.Loop_044
