/-
  morphoeval/lean/loops/Loop_041_differential_adhesion_sorting.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 041:
  Steinberg (1963, 1970) Differential Adhesion Hypothesis (DAH),
  Graner & Glazier (1992) Extended Cellular Potts Model, and
  Foty et al. (1996) Chick Embryo Tissue Surface Tension Hierarchy
  (`steinberg-differential-adhesion-sorting`).

  Pure core Lean 4 (zero external imports, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_041_differential_adhesion_sorting.lean
-/

namespace MathOfBiology.Loop_041

/-- Twice the heterotypic interfacial tension `2 * γ_{AB} = 2 * J_{AB} - (J_{AA} + J_{BB})`
    from Steinberg (1963) Science 141:401-408 and Graner & Glazier (1992) PRL 69:2013 Eq. (2).
    Working with `2 * γ` keeps all bond energies in exact integer arithmetic. -/
def two_gamma_AB (J_AA J_BB J_AB : Int) : Int :=
  2 * J_AB - (J_AA + J_BB)

/-- Twice the cell-medium interfacial tension `2 * γ_{τM} = 2 * J_{τM} - J_{ττ}` (with `J_{MM} = 0`)
    from Graner & Glazier (1992) PRL 69:2013 Eq. (3). -/
def two_gamma_M (J_tt J_tM : Int) : Int :=
  2 * J_tM - J_tt

/-- Twice the spreading tension difference `2 * (γ_{BM} - γ_{AM})` governing complete engulfment
    of higher-tension inner tissue `B` by lower-tension outer tissue `A`. -/
def two_spreading_diff (J_AA J_BB J_AM J_BM : Int) : Int :=
  two_gamma_M J_BB J_BM - two_gamma_M J_AA J_AM

/-- Theorem 1 (`steinberg_equal_adhesion_zero_tension`):
    In the equal-adhesion negative control (`J_{AA} = J_{BB} = J_{AB} = J` and `J_{AM} = J_{BM} = J_M`),
    both the heterotypic interfacial tension `γ_{AB}` and the tissue-medium surface tension difference
    `γ_{BM} - γ_{AM}` vanish identically for all integers `J, J_M`. -/
theorem steinberg_equal_adhesion_zero_tension (J J_M : Int) :
    two_gamma_AB J J J = 0 ∧ two_spreading_diff J J J_M J_M = 0 := by
  constructor
  · dsimp [two_gamma_AB]
    omega
  · dsimp [two_spreading_diff, two_gamma_M]
    omega

/-- Theorem 2 (`graner_glazier_1992_complete_engulfment_inequalities`):
    Exact verification of the four thermodynamic phase regimes of Steinberg (1963) and
    Graner & Glazier (1992) PRL 69:2013:
    (1) Canonical sorting / complete engulfment (`J_{AA}=14, J_{BB}=2, J_{AB}=11, J_{AM}=16, J_{BM}=16`):
        `2*γ_{AB} = 6`, `2*(γ_{BM} - γ_{AM}) = 12`, satisfying `0 < γ_{AB} < γ_{BM} - γ_{AM}`;
    (2) Checkerboard cell mixing (`J_{AA}=14, J_{BB}=14, J_{AB}=4`):
        `2*γ_{AB} = -20 < 0`;
    (3) Partial engulfment / doublet lens (`J_{AA}=14, J_{BB}=2, J_{AB}=16, J_{AM}=16, J_{BM}=16`):
        `2*γ_{AB} = 16 > 2*(γ_{BM} - γ_{AM}) = 12 > 0`;
    (4) Cell dissociation into medium (`J_{AA}=14, J_{AM}=2`):
        `2*γ_{AM} = -10 < 0`. -/
theorem graner_glazier_1992_complete_engulfment_inequalities :
    two_gamma_AB 14 2 11 = 6 ∧
    two_gamma_M 14 16 = 18 ∧
    two_gamma_M 2 16 = 30 ∧
    two_spreading_diff 14 2 16 16 = 12 ∧
    (0 < two_gamma_AB 14 2 11 ∧ two_gamma_AB 14 2 11 < two_spreading_diff 14 2 16 16) ∧
    two_gamma_AB 14 14 4 < 0 ∧
    two_gamma_AB 14 2 16 > two_spreading_diff 14 2 16 16 ∧
    two_gamma_M 14 2 < 0 := by
  decide

/-- Foty, Pfleger, Forgacs & Steinberg (1996) Development 122:1611-1620 Table 1 empirical
    surface tensions `10 * σ` (in `0.1 dyn/cm` integer units) for the 5 embryonic
    *Gallus gallus* tissues:
    limb bud precartilage (201) > pigmented epithelium (126) > heart ventricle (85) >
    embryonic liver (46) > neural retina (16). -/
def sigma10_limb_bud : Int := 201
def sigma10_pigmented_epithelium : Int := 126
def sigma10_heart_ventricle : Int := 85
def sigma10_embryonic_liver : Int := 46
def sigma10_neural_retina : Int := 16

/-- Theorem 3 (`foty_1996_transitive_envelopment_hierarchy`):
    Proves that the Foty & Steinberg (1996) *Gallus gallus* embryonic tissue surface tensions
    form a strict total order `σ_limb > σ_pigment > σ_heart > σ_liver > σ_retina > 0`, and that
    for any three tissues with `σ_C > σ_B > σ_A`, the envelopment tension driving force is
    strictly additive and transitive (`σ_C - σ_A = (σ_C - σ_B) + (σ_B - σ_A)` and `σ_C > σ_A`). -/
theorem foty_1996_transitive_envelopment_hierarchy (sA sB sC : Int)
    (hAB : sB > sA) (hBC : sC > sB) :
    (sC > sA) ∧
    ((sC - sA) = (sC - sB) + (sB - sA)) ∧
    (sigma10_limb_bud > sigma10_pigmented_epithelium ∧
     sigma10_pigmented_epithelium > sigma10_heart_ventricle ∧
     sigma10_heart_ventricle > sigma10_embryonic_liver ∧
     sigma10_embryonic_liver > sigma10_neural_retina ∧
     sigma10_neural_retina > 0) := by
  refine ⟨by omega, by omega, ?_⟩
  decide

/-- Glazier & Graner (1992) Cellular Potts quadratic area elasticity energy term
    `E_area(a, A_0) = λ * (a - A_0)^2`. -/
def potts_area_energy (lam a A0 : Int) : Int :=
  lam * (a - A0) * (a - A0)

/-- Theorem 4 (`potts_area_energy_equilibrium_zero`):
    At the target cell area `a = A_0`, the Potts area constraint penalty vanishes identically
    for any elasticity modulus `lam`. -/
theorem potts_area_energy_equilibrium_zero (lam A0 : Int) :
    potts_area_energy lam A0 A0 = 0 := by
  dsimp [potts_area_energy]
  rw [Int.sub_self, Int.mul_zero]

end MathOfBiology.Loop_041
