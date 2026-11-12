/-
  Loop 3130 — Bacillus subtilis, Turing wavenumber at the 16 µm detail scale.
  Holocene extant, chronoseries 1/1, age 0 Ma.
  Namespace MathOfBiology.Loop_3130. No imports. No sorry.
  Integer statements are the slot's scaled Schnakenberg threshold
  (f_u = 832/1000, g_v = -1, det J = 1, operating ratio 9200/1000 = 9.2)
  and the 16 µm versus 10 µm tier-center identities.
-/
namespace MathOfBiology.Loop_3130

/-- Trace f_u + g_v = 0.832 - 1 < 0, so the homogeneous state is stable without diffusion. -/
theorem trace_negative : (832 : Int) + (-1000) < 0 := by
  decide

/-- Twelve wavelengths of 16 µm fill the 192 µm periodic domain. -/
theorem domain_holds_twelve_wavelengths : (16 : Int) * 12 = 192 := by
  decide

/-- k(16 µm) / k(10 µm) = 10/16 = 625/1000. -/
theorem wavenumber_ratio_versus_tier_center : (16 : Int) * 625 = 10 * 1000 := by
  decide

/-- At fixed kinetics λ ∝ √D, so D(16)/D(10) = (16/10)² = 64/25 = 2.56. -/
theorem diffusion_rescales_with_wavelength_squared : (16 : Int) * 16 * 25 = 10 * 10 * 64 := by
  decide

/-- Operating ratio 9.2 lies above the Turing threshold: (r f_u + g_v)² > 4 r det. -/
theorem ratio_9200_exceeds_turing_threshold :
    ((832 : Int) * 9200 - 1000000) ^ 2 > 4 * 9200 * 1000000000 := by
  decide

/-- Equal diffusion (ratio 1) fails the same inequality, so the band is stable. -/
theorem ratio_1000_fails_turing_threshold :
    ((832 : Int) * 1000 - 1000000) ^ 2 < 4 * 1000 * 1000000000 := by
  decide

/-- Chronoseries length 1: the age step to a predecessor node is zero. -/
theorem chronoseries_age_step_is_zero : (0 : Int) = (1 - 1) * 16 := by
  decide

end MathOfBiology.Loop_3130
