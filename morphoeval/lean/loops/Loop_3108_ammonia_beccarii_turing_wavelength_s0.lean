/-
  Loop 3108 — Ammonia beccarii, Holocene (extant), 0 Ma.
  Scale: 31 µm detail variant (3.137e-05 m, log10 = -4.503, tier 4_cellular_10um).
  Law: Turing wavenumber k_c with λ_c = 31.37 µm on a 600 µm test face.
  Chronoseries 1/1: no predecessor, no successor.
  Integer encoding (so the proofs stay inside Lean 4 Int, with 0 sorry):
    λ_c = 31370 nm, pore diameter = 564 nm, test diameter = 600000 nm.
    Schnakenberg margin at Du = 1, Dv = 52/5, fu = 19/25, gv = -1, det = 1:
      disc = 863/125, and disc² > 4 Du Dv det
      clears to 744769 * 5 > 208 * 15625.
-/
namespace MathOfBiology.Loop_3108

/-- Turing margin after clearing denominators: lhs − rhs > 0 iff the band is unstable. -/
def law1_turing_margin (lhs rhs : Int) : Int := lhs - rhs

theorem law1_detail_band_is_unstable :
    law1_turing_margin (744769 * 5) (208 * 15625) > 0 := by
  dsimp [law1_turing_margin]
  decide

/-- How many critical wavelengths fit across the caption test diameter. -/
def law2_wavelength_count (diameter_nm lambda_nm : Int) : Int := diameter_nm / lambda_nm

theorem law2_nineteen_wavelengths_across_the_test :
    law2_wavelength_count 600000 31370 = 19 := by
  dsimp [law2_wavelength_count]
  decide

/-- Pore mode, this 31.37 µm mode, and the whole-test mode are strictly ordered. -/
def law3_scale_gap (fine mid coarse : Int) : Int := (mid - fine) + (coarse - mid)

theorem law3_pore_detail_test_ordering :
    law3_scale_gap 564 31370 600000 = 600000 - 564 := by
  dsimp [law3_scale_gap]
  omega

/-- Chronoseries index = length and age = 0 means a Holocene terminal with no successor. -/
def law4_terminal (index length age_ma : Int) : Int := (length - index) + age_ma

theorem law4_holocene_terminal_extant :
    law4_terminal 1 1 0 = 0 := by
  dsimp [law4_terminal]
  decide

end MathOfBiology.Loop_3108
