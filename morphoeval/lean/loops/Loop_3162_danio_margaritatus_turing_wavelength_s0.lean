/-
  morphoeval/lean/loops/Loop_3162_danio_margaritatus_turing_wavelength_s0.lean
  Slot 3162. Danio margaritatus, Holocene 0 Ma, cellular detail rung.
  lambda_c = 13390 nm = 1.339e-05 m (tier 4_cellular_10um).
  Predecessor Danio albolineatus is below the Turing threshold.
  Successor Danio nigrofasciatus selects a longer cellular wave (17800 nm).
  No `sorry`.
-/
namespace MathOfBiology.Loop_3162

/-- Elementary margaritatus wavelength in nanometres (1.339e-05 m). -/
def lambdaMargNm : Nat := 13390

/-- Successor nigrofasciatus cellular wavelength in nanometres (1.780e-05 m). -/
def lambdaNigroNm : Nat := 17800

/-- Inhibitor/activator diffusion ratio scaled by 100.
    margaritatus 40.00, critical threshold 8.50, albolineatus 1.35. -/
def deltaMargX100 : Nat := 4000
def deltaCritX100 : Nat := 850
def deltaAlboX100 : Nat := 135

/-- Detail-window length used for the mode count: 12 wavelengths. -/
def windowNm : Nat := 12 * lambdaMargNm

/-- Law 1. At 13.39 µm the margaritatus diffusion ratio is above the Turing
    threshold and the albolineatus ratio is below it. -/
theorem margaritatus_unstable_albolineatus_stable :
    deltaAlboX100 < deltaCritX100 ∧ deltaCritX100 < deltaMargX100 := by
  decide

/-- Law 2. The successor's cellular wavelength is strictly longer than the
    margaritatus pearl wavelength. Geological age is not the variable:
    both species are Holocene (0 Ma). -/
theorem successor_wavelength_longer :
    lambdaMargNm < lambdaNigroNm := by
  decide

/-- Law 3. A 12-wavelength Neumann patch holds exactly twelve elementary
    margaritatus modes, and that count is not the successor count on the
    same physical window (integer division). -/
theorem mode_count_on_detail_window :
    windowNm / lambdaMargNm = 12 ∧ windowNm / lambdaNigroNm = 9 := by
  decide

/-- Law 4. The adult pearl gap on the pinned plate is 510101 nm.
    That gap holds 38 elementary 13390 nm cells with a remainder, so the
    photograph does not resolve the cellular wavelength. -/
theorem elementary_cell_is_not_the_adult_pearl :
    38 * lambdaMargNm < 510101 ∧ 510101 < 39 * lambdaMargNm ∧
      510101 / lambdaMargNm = 38 ∧ lambdaMargNm < 510101 := by
  decide

end MathOfBiology.Loop_3162
