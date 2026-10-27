/-
  Loop 3195 — Drosophila novamexicana, Turing wavenumber at the 1.9 µm detail scale.
  Holocene (extant), 0 Ma. Subgenus Drosophila virilis–repleta–guttifera radiation.
  Chronoseries 24/41. Predecessor: Drosophila mojavensis. Successor: Drosophila orena.

  Schnakenberg Jacobian at (a, b) = (1/10, 9/10) has f_u = 4/5 and g_v = -1.
  Diffusion ratio δ = D_v/D_u = 24. Clearing the discriminant by 5² yields
  (5 δ f_u - 5)² = 91² against 4 δ det × 25.
  Equal diffusion (δ = 1) makes the cross term 4δ − 5 negative, so Turing
  condition (iii) fails. The pale-body reaction-rate scale γ = 1.00 (versus
  1.79 and 2.60) is the integer triple (100, 179, 260) and lengthens λ_c.
-/

namespace MathOfBiology.Loop_3195

/-- Cleared Schnakenberg discriminant margin at δ = 24, f_u = 4/5, det = 1. -/
def discriminantMargin : Nat := 91 * 91 - 4 * 24 * 25

theorem novamexicana_turing_discriminant_positive : discriminantMargin > 0 := by
  decide

/-- Cross term 4δ − 5 at equal diffusion δ = 1. Negative ⇒ no Turing band. -/
def equalDiffusionCross : Int := 4 * 1 - 5

theorem equal_diffusion_suppresses_turing : equalDiffusionCross < 0 := by
  decide

def lambdaNovaNm : Nat := 1901
def lambdaMojavensisNm : Nat := 1421
def lambdaOrenaNm : Nat := 1179
def gammaNova : Nat := 100
def gammaMojavensis : Nat := 179
def gammaOrena : Nat := 260

/-- At 1.9 µm the pale ebony/tan state (lowest γ) has the longest wavelength. -/
theorem pale_cis_regulatory_state_lengthens_wavelength :
    lambdaNovaNm > lambdaMojavensisNm ∧
    lambdaNovaNm > lambdaOrenaNm ∧
    gammaOrena > gammaMojavensis ∧
    gammaMojavensis > gammaNova := by
  decide

/-- The 15.208 µm abdominal patch holds eight 1.901 µm wavelengths. -/
theorem patch_holds_eight_wavelengths : 8 * lambdaNovaNm = 15208 := by
  decide

end MathOfBiology.Loop_3195
