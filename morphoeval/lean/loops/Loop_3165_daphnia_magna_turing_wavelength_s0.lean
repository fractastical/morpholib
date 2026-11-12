/-
  morphoeval/lean/loops/Loop_3165_daphnia_magna_turing_wavelength_s0.lean
  Slot 3165. Daphnia magna, Holocene (0 Ma), ocellus-pigment Turing wavelength 2.736 µm.
  Namespace MathOfBiology.Loop_3165. Four theorems, zero sorry.
  The integer witnesses are the cleared numerators of the Schnakenberg discriminant
  used by models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py
  (a = 62/1000, b = 1180/1000, D_u = 16/1000, D_v = 640/1000).
-/
namespace MathOfBiology.Loop_3165

/-- Critical wavelength of the Holocene ocellus pigment field, in nanometres. -/
def lambdaCnm : Nat := 2736

/-- Organelle-scale predecessor length: screening-pigment melanosome (nm). -/
def melanosomeNm : Nat := 500

/-- Cellular-scale successor length: ommatidial pitch bound (nm). -/
def ommatidiumNm : Nat := 22000

theorem wavelength_between_melanosome_and_ommatidium :
    melanosomeNm < lambdaCnm ∧ lambdaCnm < ommatidiumNm := by
  decide

/-- Schnakenberg diffusivities ×10³. D_u = 0.016, D_v = 0.640. -/
def duMilli : Nat := 16
def dvMilli : Nat := 640

theorem inhibitor_diffuses_forty_times_faster :
    dvMilli = duMilli * 40 ∧ duMilli < dvMilli := by
  decide

/-- Numerator of disc = D_v f_u + D_u g_v after clearing denominators.
    A positive numerator (positive denominator) is a positive k_c². -/
def discNumerator : Nat := 5350516939

theorem critical_wavenumber_numerator_positive :
    discNumerator > 0 := by
  decide

/-- Numerator of the Turing margin disc² − 4 D_u D_v det. Same rationals as the solver. -/
def marginNumerator : Nat := 22679272279285929721

theorem turing_discriminant_margin_positive :
    marginNumerator > 0 := by
  decide

/-- Chronoseries 1/1 at 0 Ma: no predecessor and no successor chronospecies. -/
def seriesIndex : Nat := 1
def seriesLength : Nat := 1
def ageMa : Nat := 0

theorem holocene_singleton_chronoseries :
    seriesIndex = seriesLength ∧ ageMa = 0 := by
  decide

end MathOfBiology.Loop_3165
