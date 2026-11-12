/-
  Loop 3211 — Drosophila willistoni, subcellular Turing wavelength.
  Scale: 1.5 µm (1466 nm = 1.466e-6 m), tier 3_subcellular_um, Holocene, 0 Ma.
  Predecessor Drosophila virilis (1676 nm). Successor Drosophila yakuba (1509 nm).
  Kinetics are conserved. Wavelength and granules-per-cell are not.
  No `sorry`. Core Lean 4 only (Nat / Int / decide).
-/
namespace MathOfBiology.Loop_3211

/-- Wavelength in nanometres. 0 virilis (predecessor), 1 willistoni (focal), 2 yakuba (successor). -/
def lambdaNm (species : Nat) : Nat :=
  match species with
  | 0 => 1676
  | 1 => 1466
  | 2 => 1509
  | _ => 0

/-- Epidermal cell diameter in nanometres, linear in catalog body length. -/
def cellNm (species : Nat) : Nat :=
  match species with
  | 0 => 12398
  | 1 => 8460
  | 2 => 9189
  | _ => 0

theorem focal_wavelength_is_1_5_um_scale :
    lambdaNm 1 = 1466 := by decide

theorem willistoni_shorter_than_predecessor_virilis :
    lambdaNm 1 < lambdaNm 0 := by decide

theorem willistoni_shorter_than_successor_yakuba :
    lambdaNm 1 < lambdaNm 2 := by decide

theorem successor_yakuba_shorter_than_predecessor_virilis :
    lambdaNm 2 < lambdaNm 0 := by decide

/-- Granules across a cell compare as d_a / λ_a > d_b / λ_b iff d_a * λ_b > d_b * λ_a. -/
theorem virilis_has_more_granules_per_cell_than_willistoni :
    cellNm 0 * lambdaNm 1 > cellNm 1 * lambdaNm 0 := by decide

theorem yakuba_granules_per_cell_lie_between :
    cellNm 0 * lambdaNm 2 > cellNm 2 * lambdaNm 0
      ∧ cellNm 2 * lambdaNm 1 > cellNm 1 * lambdaNm 2 := by decide

/-- Positive margin means D_v / D_u > critNum / critDen.
    Focal diffusivities are stored in units of 10⁻⁶ µm²: D_u = 15919, D_v = 214909,
    so D_v / D_u = 13.500, above the Schnakenberg onset ratio 11.37. -/
def turingMargin (dv du critNum critDen : Nat) : Int :=
  (dv * critDen : Int) - (du * critNum : Int)

theorem willistoni_above_turing_onset :
    turingMargin 214909 15919 1137 100 > 0 := by decide

/-- Ablation: equal diffusivities sit below the same onset ratio. -/
theorem equal_diffusivity_ablation_is_below_onset :
    turingMargin 15919 15919 1137 100 < 0 := by decide

/-- Five full 1.466 µm wavelengths fit inside the 8.46 µm willistoni cell, and six do not. -/
theorem willistoni_cell_holds_five_granule_wavelengths :
    5 * lambdaNm 1 < cellNm 1 ∧ cellNm 1 < 6 * lambdaNm 1 := by decide

end MathOfBiology.Loop_3211
