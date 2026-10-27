/-
  Loop 3217 — Emiliania huxleyi, Holocene (extant, 0 Ma).
  Scale: 5.808 nm critical Turing wavelength (5808 pm).
  The vesicle patch is 8 Holocene wavelengths wide (46464 pm).
  The reconstructed predecessor puts 6 wavelengths on that same patch
  (7744 pm). There is no successor chronospecies.
  No Mathlib import: the identities are exact integer arithmetic.
-/
namespace MathOfBiology.Loop_3217

/-- Holocene critical wavelength, picometres. -/
def lambdaHolocenePm : Nat := 5808

/-- Reconstructed predecessor spacing: exactly 6 wavelengths on the Holocene patch. -/
def lambdaPredecessorPm : Nat := 7744

/-- Extant terminal node. The successor wavelength is the Holocene value. -/
def lambdaSuccessorPm : Nat := lambdaHolocenePm

def domainPm : Nat := 46464

def holoceneModes : Nat := 8

def predecessorModes : Nat := 6

/-- Calcite a-axis used only as a lattice yardstick, picometres. -/
def calciteAPm : Nat := 499

theorem holocene_patch_is_eight_wavelengths :
    domainPm = holoceneModes * lambdaHolocenePm := by decide

theorem predecessor_patch_is_six_wavelengths :
    domainPm = predecessorModes * lambdaPredecessorPm := by decide

theorem holocene_repeat_is_shorter_than_predecessor :
    lambdaHolocenePm < lambdaPredecessorPm := by decide

theorem successor_is_the_extant_fixed_point :
    lambdaSuccessorPm = lambdaHolocenePm := by decide

/-- Entering the Holocene node adds two critical modes on a fixed patch. -/
theorem holocene_gains_two_modes :
    holoceneModes = predecessorModes + 2 := by decide

/-- 5.808 nm is not an integer number of calcite a-axis cells. -/
theorem wavelength_not_integer_calcite_cells :
    11 * calciteAPm < lambdaHolocenePm ∧ lambdaHolocenePm < 12 * calciteAPm := by decide

/-- Diffusion product for the slot ratio (D_u, D_v scaled by 1e4 nm²):
    D_u ≈ 0.3181 → 3181, D_v = 18 D_u → 57258, and the product exceeds the
    stability gap used by the script's h_min < 0 test (det scaled). -/
def duScaled : Nat := 3181
def dvScaled : Nat := 57258
def detGapScaled : Nat := 100000000

theorem diffusion_product_clears_the_stability_gap :
    duScaled * dvScaled > detGapScaled := by decide

end MathOfBiology.Loop_3217
