/-
  Loop 3118 — Arachnoidiscus ehrenbergii, Turing wavelength, detail scale.
  Scale: 242 nm = 2.42e-7 m (tier 2_macromolecular_100nm, log10 = -6.616).
  Age: Holocene extant, 0 Ma. Chronoseries 1/1 (no fossil predecessor, no successor).
  Law: selected wavenumber k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π / k_c.
  Integer witnesses use milli-units of the slot Jacobian (a = 0.062, b = 1.35):
    f_u = +0.912, g_v = -1.994, D_v/D_u = 31.18.
  The h > 0 onset is |g_v|/f_u ≈ 2.188; the discriminant onset is 13.605.
  The slot ratio sits above the discriminant onset. These are not the slot-0199 witnesses (49, 14).
-/
namespace MathOfBiology.Loop_3118

/-- λ_c in nanometres for this detail-scale slot. -/
def wavelengthNm : Nat := 242

/-- Macromolecular window: 100 nm < λ_c < 1000 nm (tier 2, not the cellular tiers). -/
theorem wavelength_in_macromolecular_window :
    100 < wavelengthNm ∧ wavelengthNm < 1000 := by
  decide

/-- Diffusion ratio and discriminant onset, both in milli-units (31.180 and 13.605). -/
def diffusionRatioMilli : Nat := 31180
def turingOnsetRatioMilli : Nat := 13605

/-- D_v/D_u = 31.18 lies strictly above the Schnakenberg discriminant onset 13.605. -/
theorem diffusion_ratio_exceeds_turing_onset :
    diffusionRatioMilli > turingOnsetRatioMilli := by
  decide

/-- Detail patch holds six critical wavelengths: 6 × 242 nm = 1452 nm = 1.452 μm. -/
def domainNm : Nat := 6 * wavelengthNm

theorem detail_domain_holds_six_wavelengths :
    domainNm = 1452 ∧ domainNm / wavelengthNm = 6 := by
  decide

/-- Equal diffusion cannot satisfy D_v f_u + D_u g_v > 0 when f_u = 912 and |g_v| = 1994 (milli). -/
theorem equal_diffusion_trace_term_is_negative :
    (912 : Int) < 1994 := by
  decide

/-- Holocene chronoseries 1/1: age offset from the extant datum is zero, and there is
    no predecessor horizon to subtract. -/
theorem holocene_chronoseries_age_offset_is_zero :
    (0 : Int) - 0 = 0 := by
  decide

end MathOfBiology.Loop_3118
