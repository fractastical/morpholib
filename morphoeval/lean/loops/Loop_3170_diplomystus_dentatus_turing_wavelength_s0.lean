/-
  Loop 3170 — Diplomystus dentatus, detail-tier Turing wavelength.
  Scale: 184.3 µm (0.0001843 m), tier 5_primordium_100um.
  Horizon: Early Eocene (Ypresian), 51.5 Ma. Chronoseries 1/1.
  Successor compared here is the organ-tier wavelength 7.56 mm of the same
  species (not a second chronospecies). Predecessor is a 20 µm chromatophore,
  shorter than half of λ_c, so the unstable wave does not fit.
  Lengths below are stored in tenths of a micrometre (184.3 µm = 1843).
-/

namespace MathOfBiology.Loop_3170

def lambdaDetailUmX10 : Nat := 1843
def lambdaOrganUmX10 : Nat := 75600
def chromatophoreUmX10 : Nat := 200
def diffusivityRatioX100 : Nat := 917
def fuX10000 : Nat := 7130
def gvAbsX10000 : Nat := 7957
def flankWindowUmX10 : Nat := 29488

/-- The 184.3 µm detail wave is shorter than the 7.56 mm organ-tier successor. -/
theorem detail_wavelength_shorter_than_organ_successor :
    lambdaDetailUmX10 < lambdaOrganUmX10 := by
  decide

/-- k_detail / k_organ = λ_organ / λ_detail > 41 at this scale versus the 7.56 mm successor. -/
theorem detail_wavenumber_exceeds_organ_successor_fortyfold :
    lambdaOrganUmX10 > 41 * lambdaDetailUmX10 := by
  decide

/-- Inhibitor/activator ratio 9.17 clears the trace-diffusion threshold fu and |gv| of this Jacobian. -/
theorem diffusivity_ratio_clears_turing_threshold :
    diffusivityRatioX100 * fuX10000 > gvAbsX10000 * 100 := by
  decide

/-- A 20 µm chromatophore is shorter than λ_c / 2, so the predecessor scale holds no full half-wave. -/
theorem chromatophore_predecessor_is_shorter_than_half_wavelength :
    2 * chromatophoreUmX10 < lambdaDetailUmX10 := by
  decide

/-- Sixteen detail waves fill a 2.9488 mm flank window, which is still shorter than one organ wave. -/
theorem flank_window_holds_sixteen_detail_waves_and_no_organ_wave :
    lambdaDetailUmX10 * 16 = flankWindowUmX10 ∧ flankWindowUmX10 < lambdaOrganUmX10 := by
  decide

end MathOfBiology.Loop_3170
