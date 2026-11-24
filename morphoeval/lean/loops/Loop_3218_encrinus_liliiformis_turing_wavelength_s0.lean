/-
  Loop 3218 — Encrinus liliiformis
  Scale: 12 nm (12 380 pm = 1.238e-8 m, log10 = -7.907), tier 1_molecular_nm
  Age: Middle Triassic (Muschelkalk), 240 Ma
  Law: Turing wavenumber k_c for Schnakenberg kinetics inside one calcite nanograin
  Neighbors: Holocrinus dubius (248 Ma, 18.72 nm) and Chelocrinus cassianus (232 Ma, 15.64 nm)
-/
namespace MathOfBiology.Loop_3218

def lambdaPm : Nat := 12380
def lambdaHolocrinusPm : Nat := 18720
def lambdaChelocrinusPm : Nat := 15640
def ageEncrinusMa : Nat := 240
def ageHolocrinusMa : Nat := 248
def ageChelocrinusMa : Nat := 232
def nanograinPm : Nat := 100000

/-- The Muschelkalk species is the short-wavelength end-member of this three-point series. -/
theorem muschelkalk_wavelength_is_chronoseries_minimum :
    lambdaPm < lambdaChelocrinusPm ∧ lambdaChelocrinusPm < lambdaHolocrinusPm := by
  decide

/-- Reciprocal wavenumber, in units of 10^9 / pm, is highest at 12 nm. -/
theorem wavenumber_peaks_in_the_muschelkalk :
    (1000000000 / lambdaPm) > (1000000000 / lambdaChelocrinusPm)
      ∧ (1000000000 / lambdaChelocrinusPm) > (1000000000 / lambdaHolocrinusPm) := by
  decide

/-- 240 Ma sits strictly between the Early Triassic predecessor and the Carnian successor. -/
theorem muschelkalk_age_lies_between_neighbors :
    ageHolocrinusMa > ageEncrinusMa ∧ ageEncrinusMa > ageChelocrinusMa := by
  decide

/-- Trace f_u + g_v = 553/600 − 36/25 < 0 at a = 47/1000, b = 1153/1000. -/
theorem schnakenberg_trace_negative :
    553 * 25 < 36 * 600 := by
  decide

/-- δ = 112/5 makes δ f_u + g_v positive; equal diffusion (δ = 1) does not. -/
theorem inhibitor_drive_positive_and_equal_diffusion_fails :
    112 * 553 * 25 > 36 * 5 * 600 ∧ 553 * 25 < 36 * 600 := by
  decide

/-- A 100 nm nanograin holds more 12 nm wavelengths than either neighbor mode. -/
theorem nanograin_packs_more_encrinus_wavelengths :
    nanograinPm / lambdaPm > nanograinPm / lambdaChelocrinusPm
      ∧ nanograinPm / lambdaChelocrinusPm > nanograinPm / lambdaHolocrinusPm := by
  decide

end MathOfBiology.Loop_3218
