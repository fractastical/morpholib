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

/-! ### Pass 3 — Ouchi et al. (2003) perimeter / cortical tension elasticity extension

  Ouchi NB, Glazier JA, Rieu JP, Upadhyaya A, Sawada Y (2003) Physica A 329(3-4):451-458, Eq. (2):
  `H_P = λ_P * ∑_{σ > 0} (p_σ - P_t)^2`, where `p_σ` is the 8-neighbour bond-count perimeter of
  cell `σ` (number of (site ∈ σ, 8-neighbour ∉ σ) ordered pairs). -/

/-- 8-neighbour bond-count perimeter of an `s × s` square lattice block (`s ≥ 2`):
    4 corner sites × 5 exterior bonds + 4 (s − 2) edge sites × 3 exterior bonds = `12 s − 4`. -/
def square_block_perimeter_8nbr (s : Int) : Int :=
  12 * s - 4

/-- Theorem 4 (`cpm_square_block_target_perimeter`):
    The compact-square target perimeters used by the script's `default_target_perimeter`:
    `P_t(3 × 3) = 32`, `P_t(4 × 4) = 44` (the canonical `V_t = 16` cell), `P_t(5 × 5) = 56`;
    and the per-side increment is exactly 12 bonds. -/
theorem cpm_square_block_target_perimeter :
    square_block_perimeter_8nbr 3 = 32 ∧
    square_block_perimeter_8nbr 4 = 44 ∧
    square_block_perimeter_8nbr 5 = 56 ∧
    (∀ s : Int, square_block_perimeter_8nbr (s + 1) - square_block_perimeter_8nbr s = 12) := by
  refine ⟨by decide, by decide, by decide, ?_⟩
  intro s
  dsimp [square_block_perimeter_8nbr]
  omega

/-- Perimeter / cortical tension elasticity energy `E_P(p, P_t) = λ_P * (p − P_t)^2`
    (Ouchi et al. 2003 Eq. 2; script `perimeter_energy_delta`). -/
def cpm_perimeter_energy (lamP p Pt : Int) : Int :=
  lamP * ((p - Pt) * (p - Pt))

/-- Perimeter energy change when a cell's perimeter changes by `k` bonds:
    `ΔE_P(p, k) = E_P(p + k, P_t) − E_P(p, P_t)`. -/
def cpm_delta_perim (lamP p Pt k : Int) : Int :=
  cpm_perimeter_energy lamP (p + k) Pt - cpm_perimeter_energy lamP p Pt

/-- Perimeter change of the cell a site *leaves* when `n` of the site's 8 neighbours belong to it:
    the `8 − n` exterior bonds of the site vanish and its `n` same-cell neighbours each gain one
    exterior bond, so `Δp_old = n − (8 − n) = 2 n − 8` (script `dp_old`). -/
def cpm_copy_perimeter_change_old (n : Int) : Int :=
  2 * n - 8

/-- Perimeter change of the cell a site *joins* when `n` of the site's 8 neighbours already belong
    to it: `Δp_new = (8 − n) − n = 8 − 2 n` (script `dp_new`). -/
def cpm_copy_perimeter_change_new (n : Int) : Int :=
  8 - 2 * n

/-- Theorem 5 (`cpm_perimeter_elasticity_restoring_identity`):
    (a) at the target perimeter the cortical tension energy vanishes, `E_P(P_t, P_t) = 0`;
    (b) a single-site copy changes a cell's perimeter by an even number of bonds, bounded by 8,
        for every neighbour count `0 ≤ n ≤ 8`, and the two change laws are exact negatives;
    (c) the minimal single-copy perimeter change `±2` from the target costs exactly `4 λ_P`
        (the perimeter analogue of the volume barrier `+λ` in Theorem 3), and the maximal
        change `±8` (isolated-pixel detachment / attachment) costs `64 λ_P`. -/
theorem cpm_perimeter_elasticity_restoring_identity (lamP Pt : Int) :
    cpm_perimeter_energy lamP Pt Pt = 0 ∧
    (∀ n : Int, 0 ≤ n → n ≤ 8 →
      (∃ m : Int, cpm_copy_perimeter_change_old n = 2 * m) ∧
      cpm_copy_perimeter_change_old n = - cpm_copy_perimeter_change_new n ∧
      -8 ≤ cpm_copy_perimeter_change_old n ∧ cpm_copy_perimeter_change_old n ≤ 8) ∧
    cpm_delta_perim lamP Pt Pt 2 = 4 * lamP ∧
    cpm_delta_perim lamP Pt Pt (-2) = 4 * lamP ∧
    cpm_delta_perim lamP Pt Pt 8 = 64 * lamP := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · dsimp [cpm_perimeter_energy]
    rw [Int.sub_self, Int.mul_zero, Int.mul_zero]
  · intro n hn0 hn8
    refine ⟨⟨n - 4, ?_⟩, ?_, ?_, ?_⟩
    · dsimp [cpm_copy_perimeter_change_old]; omega
    · dsimp [cpm_copy_perimeter_change_old, cpm_copy_perimeter_change_new]; omega
    · dsimp [cpm_copy_perimeter_change_old]; omega
    · dsimp [cpm_copy_perimeter_change_old]; omega
  · dsimp [cpm_delta_perim, cpm_perimeter_energy]
    omega
  · dsimp [cpm_delta_perim, cpm_perimeter_energy]
    omega
  · dsimp [cpm_delta_perim, cpm_perimeter_energy]
    omega

/-- Theorem 6 (`cpm_cortical_tension_monotone_above_target`):
    For a non-negative cortical tension multiplier `λ_P ≥ 0`, once a cell's perimeter is at or above
    its target (`p ≥ P_t`) every further perimeter increase `k ≥ 0` is energetically non-favourable:
    `E_P(p, P_t) ≤ E_P(p + k, P_t)`, i.e. `ΔE_P ≥ 0`. This is the sense in which the perimeter term
    acts as an effective cortical line tension that opposes ragged, fragmented cell boundaries. -/
theorem cpm_cortical_tension_monotone_above_target
    (lamP p Pt k : Int) (hlam : 0 ≤ lamP) (hp : Pt ≤ p) (hk : 0 ≤ k) :
    cpm_perimeter_energy lamP p Pt ≤ cpm_perimeter_energy lamP (p + k) Pt := by
  dsimp [cpm_perimeter_energy]
  have h1 : p - Pt ≤ p + k - Pt := by omega
  have h2 : 0 ≤ p - Pt := by omega
  have h3 : 0 ≤ p + k - Pt := by omega
  have hsq : (p - Pt) * (p - Pt) ≤ (p + k - Pt) * (p + k - Pt) :=
    Int.mul_le_mul h1 h1 h2 h3
  exact Int.mul_le_mul_of_nonneg_left hsq hlam

end MathOfBiology.Loop_042
