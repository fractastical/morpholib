/-
  Loop 3224 — Euplectella aspergillum, Turing wavenumber at the 34 µm detail scale.
  λ_c = 3.43e-5 m = 343 × 10^-7 m (34.3 µm). Holocene, 0 Ma. Chronoseries 1/1.
  k_c = round(2π / λ_c) = 183183 m^-1.
  Schnakenberg (a, b, D_u, D_v, γ) = (0.1, 0.9, 1, 30, 1):
    trace(J) = -1/5 < 0, det(J) = 1 > 0, h = D_v f_u + D_u g_v = 23,
    h² = 529 > 4 D_u D_v det = 120.
  Equal diffusion D_u = D_v = 1 yields h = -1/5 < 0 (no Turing band).
  Scale predecessor 51 µm = 510 × 10^-7 m is a comparison constant, not a copied parameter set.
-/
namespace MathOfBiology.Loop_3224

/-- Detail-scale wavelength in units of 10^-7 m. 343 × 10^-7 m = 3.43 × 10^-5 m. -/
def lambdaDetail : Nat := 343

/-- Coarser cellular-tier comparison wavelength: 510 × 10^-7 m = 5.10 × 10^-5 m. -/
def lambdaCoarse : Nat := 510

/-- Rounded critical wavenumber k_c in m^-1 for the detail scale. -/
def kDetail : Nat := 183183

theorem detail_wavelength_positive : 0 < lambdaDetail := by
  decide

theorem detail_scale_shorter_than_coarse_tier : lambdaDetail < lambdaCoarse := by
  decide

/-- |k_c · λ_c − 2π| < one step in k_c, with 2π scaled by 10^7 and λ by 10^-7. -/
theorem kc_rounds_two_pi_over_lambda :
    Int.natAbs ((183183 : Int) * 343 - 62831853) < 343 := by
  decide

/-- Schnakenberg Turing inequality at D_v/D_u = 30: h² = 23² > 4 D_u D_v det(J) = 120. -/
theorem schnakenberg_turing_inequality : 23 * 23 > 4 * 30 := by
  decide

/-- Reaction Jacobian trace is negative: 10³(f_u + g_v) = 800 + (−1000) < 0. -/
theorem reaction_trace_negative : (800 : Int) + (-1000) < 0 := by
  decide

/-- Equal-diffusion control: 10 · h = 10·(D f_u + D g_v) = 8 + (−10) < 0, so condition (iii) fails. -/
theorem equal_diffusion_has_no_turing_band : (8 : Int) + (-10) < 0 := by
  decide

/-- Same window, more wavelengths at 34.3 µm than at 51 µm: 1.48 < λ_coarse/λ_detail < 1.49. -/
theorem detail_packs_more_wavelengths :
    lambdaCoarse * 100 > lambdaDetail * 148 ∧ lambdaDetail * 149 > lambdaCoarse * 100 := by
  decide

end MathOfBiology.Loop_3224
