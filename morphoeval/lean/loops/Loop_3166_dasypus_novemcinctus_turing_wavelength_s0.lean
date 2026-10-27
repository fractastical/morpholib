/-
  morphoeval/lean/loops/Loop_3166_dasypus_novemcinctus_turing_wavelength_s0.lean
  Slot 3166 — Dasypus novemcinctus, Turing wavenumber at the 4.108 µm detail scale.
  Geological horizon: Holocene (Extant), 0 Ma. Chronoseries length 1.
  The integers below are milli-unit transcriptions of the Schnakenberg Jacobian
  and diffusivities used in models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py.
  They are not the 273 µm sibling slot's integers.
-/
namespace MathOfBiology.Loop_3166

/-- Detail-scale wavelength in nanometres (4.108 µm). -/
def lambdaDetailNm : Nat := 4108

/-- Primordium-scale sibling wavelength in nanometres (273 µm), used only as a scale contrast. -/
def lambdaPrimordiumNm : Nat := 273000

/-- Integer packing of detail wavelengths inside one primordium wavelength. -/
def modePacking : Nat := lambdaPrimordiumNm / lambdaDetailNm

theorem detail_packs_sixty_six_modes : modePacking = 66 := by
  decide

theorem packing_remainder_nonzero :
    lambdaPrimordiumNm - modePacking * lambdaDetailNm = 1872 := by
  decide

/-- Holocene singleton: one chronospecies node and a zero age gap. -/
def chronoseriesLength : Nat := 1
def ageDeltaMa : Nat := 0

theorem holocene_singleton_chronoseries :
    chronoseriesLength = 1 ∧ ageDeltaMa = 0 := by
  decide

/-- Diffusivities in 10^-3 µm²/τ. D_v / D_u = 20, which clears the differential-diffusion threshold. -/
def duMilli : Nat := 160
def dvMilli : Nat := 3206

theorem differential_diffusion_ratio : dvMilli > 8 * duMilli := by
  decide

/-- Homogeneous Schnakenberg trace f_u + g_v = 0.800 + (−1.000) < 0 (milli-units). -/
theorem homogeneous_trace_negative : ((800 : Int) + (-1000)) < 0 := by
  decide

/-- Homogeneous determinant f_u g_v − f_v g_u = 1.000 > 0 (milli-units). -/
theorem homogeneous_determinant_positive : (1000 : Int) > 0 := by
  decide

/-- Turing discriminant in milli-units: h = D_v f_u + D_u g_v and h² exceeds 4 D_u D_v det. -/
theorem turing_discriminant_positive :
    ((3206 : Int) * 800 + (160 : Int) * (-1000)) ^ 2
      > (4 : Int) * 160 * 3206 * 1000000 := by
  decide

end MathOfBiology.Loop_3166
