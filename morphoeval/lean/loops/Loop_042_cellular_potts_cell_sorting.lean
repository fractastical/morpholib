/-
  morphoeval/lean/loops/Loop_042_cellular_potts_cell_sorting.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 042:
  Graner & Glazier (1992) Phys. Rev. Lett. 69:2013-2016 and
  Glazier & Graner (1993) Phys. Rev. E 47:2128-2154 Extended Large-Q
  Cellular Potts Model (CPM / GGH) Cell Sorting (`graner-cellular-potts-cell-sorting`).

  Pure core Lean 4 (zero external imports, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_042_cellular_potts_cell_sorting.lean
-/

namespace MathOfBiology.Loop_042

/-- Twice the heterotypic cell-cell interfacial tension
    `2 * γ_{LD} = 2 * J_{LD} - (J_{LL} + J_{DD})`
    from Graner & Glazier (1992) PRL 69:2013 Eq. (2) and Glazier & Graner (1993) PRE 47:2128 Eq. (3). -/
def two_gamma_LD (J_LL J_DD J_LD : Int) : Int :=
  2 * J_LD - (J_LL + J_DD)

/-- Twice the cell-medium interfacial tension
    `2 * γ_{αM} = 2 * J_{αM} - J_{αα}` (with `J_{MM} = 0`)
    from Graner & Glazier (1992) PRL 69:2013 Eq. (3). -/
def two_gamma_M (J_aa J_aM : Int) : Int :=
  2 * J_aM - J_aa

/-- Twice the cell-medium spreading tension difference
    `2 * (γ_{DM} - γ_{LM}) = (2 * J_{DM} - J_{DD}) - (2 * J_{LM} - J_{LL})`
    governing engulfment of dark endodermal core cells (`D`) by light ectodermal shell cells (`L`). -/
def two_spreading_diff (J_LL J_DD J_LM J_DM : Int) : Int :=
  two_gamma_M J_DD J_DM - two_gamma_M J_LL J_LM

/-- Theorem 1 (`cpm_equal_adhesion_null_tension`):
    In the equal-adhesion negative control (`J_{LL} = J_{DD} = J_{LD} = J` and `J_{LM} = J_{DM} = J_M`),
    both the heterotypic interfacial tension `γ_{LD}` and the cell-medium spreading difference
    `γ_{DM} - γ_{LM}` vanish identically for all integer bond energies `J, J_M`. -/
theorem cpm_equal_adhesion_null_tension (J J_M : Int) :
    two_gamma_LD J J J = 0 ∧ two_spreading_diff J J J_M J_M = 0 := by
  constructor
  · dsimp [two_gamma_LD]
    omega
  · dsimp [two_spreading_diff, two_gamma_M]
    omega

/-- Theorem 2 (`graner_glazier_four_phase_regimes`):
    Machine-checked verification of the four canonical thermodynamic phase regimes of the
    Cellular Potts Model (Graner & Glazier 1992 PRL p. 2014; Glazier & Graner 1993 PRE Table I):
    (1) Complete sorting / engulfment (`J_{LL}=14, J_{DD}=2, J_{LD}=11, J_{LM}=16, J_{DM}=16`):
        `2*γ_{LD} = 6`, `2*γ_{LM} = 18`, `2*γ_{DM} = 30`, `2*(γ_{DM} - γ_{LM}) = 12`,
        satisfying `0 < γ_{LD} < γ_{DM} - γ_{LM}`;
    (2) Partial sorting / lens formation (`J_{LL}=14, J_{DD}=2, J_{LD}=16, J_{LM}=16, J_{DM}=16`):
        `2*γ_{LD} = 16 > 2*(γ_{DM} - γ_{LM}) = 12 > 0`;
    (3) Checkerboard mixing (`J_{LL}=14, J_{DD}=14, J_{LD}=4`):
        `2*γ_{LD} = -20 < 0`;
    (4) Cell dissociation into medium (`J_{LL}=14, J_{LM}=2`):
        `2*γ_{LM} = -10 < 0`. -/
theorem graner_glazier_four_phase_regimes :
    two_gamma_LD 14 2 11 = 6 ∧
    two_gamma_M 14 16 = 18 ∧
    two_gamma_M 2 16 = 30 ∧
    two_spreading_diff 14 2 16 16 = 12 ∧
    (0 < two_gamma_LD 14 2 11 ∧ two_gamma_LD 14 2 11 < two_spreading_diff 14 2 16 16) ∧
    two_gamma_LD 14 2 16 > two_spreading_diff 14 2 16 16 ∧
    two_gamma_LD 14 14 4 < 0 ∧
    two_gamma_M 14 2 < 0 := by
  decide

/-- Cellular Potts quadratic volume/area elasticity energy penalty
    `E_{vol}(v, V_t) = λ * (v - V_t)^2` from Graner & Glazier (1992) PRL Eq. (1). -/
def cpm_volume_energy (lam v Vt : Int) : Int :=
  lam * (v - Vt) * (v - Vt)

/-- Single-site spin copy volume energy change `ΔE_{vol}^+(v) = E_{vol}(v + 1, V_t) - E_{vol}(v, V_t)`
    when a cell of volume `v` gains one lattice pixel. -/
def cpm_delta_vol_gain (lam v Vt : Int) : Int :=
  cpm_volume_energy lam (v + 1) Vt - cpm_volume_energy lam v Vt

/-- Single-site spin copy volume energy change `ΔE_{vol}^-(v) = E_{vol}(v - 1, V_t) - E_{vol}(v, V_t)`
    when a cell of volume `v` loses one lattice pixel. -/
def cpm_delta_vol_loss (lam v Vt : Int) : Int :=
  cpm_volume_energy lam (v - 1) Vt - cpm_volume_energy lam v Vt

/-- Theorem 3 (`cpm_volume_elasticity_restoring_identity`):
    Proves that:
    (a) at target volume `v = V_t`, the volume elasticity energy vanishes (`E_{vol}(λ, V_t, V_t) = 0`), and
    (b) for any volume `v = V_t`, gaining or losing a single lattice pixel costs exactly `+λ`
        (`ΔE_{vol}^+(V_t) = λ` and `ΔE_{vol}^-(V_t) = λ`), creating a symmetric restoring barrier around `V_t`. -/
theorem cpm_volume_elasticity_restoring_identity (lam Vt : Int) :
    cpm_volume_energy lam Vt Vt = 0 ∧
    cpm_delta_vol_gain lam Vt Vt = lam ∧
    cpm_delta_vol_loss lam Vt Vt = lam := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [cpm_volume_energy]
    rw [Int.sub_self, Int.mul_zero]
  · dsimp [cpm_delta_vol_gain, cpm_volume_energy]
    omega
  · dsimp [cpm_delta_vol_loss, cpm_volume_energy]
    omega

end MathOfBiology.Loop_042
