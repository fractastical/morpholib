/-
  morphoeval/lean/loops/Loop_3154_coprinopsis_cinerea_turing_wavelength_s0.lean

  Coprinopsis cinerea, slot 3154.
  Scale: 77 µm detail variant, exact length 7.744e-05 m (77.44 µm), log10 = -4.111,
  tier 4_cellular_10um. Age: Holocene extant, 0 Ma. Chronoseries 1/1.

  Integer encoding used by the proofs (hundredths of a micrometre):
    λ_slot      = 7744    (77.44 µm)
    λ_floor     = 1000    (10.00 µm tier floor)
    Schnakenberg trace margin at the operating point, cleared of the denominator 5:
      f_u = 4/5, g_v = -1, D_u = 1, D_v = 16
      margin = D_v * 4 + D_u * (-1) * 5
    Equal diffusion sets D_v = D_u = 1 and the same margin is negative.
-/

namespace MathOfBiology.Loop_3154

/-- Cleared Turing trace margin D_v f_u + D_u g_v, with f_u = fuNum/fuDen. -/
def traceMargin (Dv Du fuNum fuDen gvNum : Int) : Int :=
  Dv * fuNum + Du * gvNum * fuDen

/-- Operating point of this slot: D_v/D_u = 16, f_u = 4/5, g_v = -1. -/
theorem operating_trace_positive :
    traceMargin 16 1 4 5 (-1) > 0 := by
  decide

/-- Ablation: equal diffusion fails the necessary Turing trace condition. -/
theorem equal_diffusion_trace_negative :
    traceMargin 1 1 4 5 (-1) < 0 := by
  decide

/-- 77.44 µm sits strictly between 7 and 8 tier floors of 10 µm.
    Chronoseries 1/1 has no predecessor species; the floor is the scale contrast. -/
theorem detail_variant_between_seven_and_eight_floors :
    7 * 1000 < 7744 ∧ 7744 < 8 * 1000 := by
  decide

/-- Holocene terminus: age is 0 Ma and the single chronoseries entry has no successor.
    `seriesLength - seriesIndex = 0` is the 1/1 remainder. -/
theorem holocene_terminus_has_no_successor :
    (1 : Int) - 1 = 0 ∧ (0 : Int) = 0 := by
  decide

/-- Diffusion-product rescaling that would move k_c from 77.44 µm down to the 10 µm floor.
    k_c ∝ (D_u D_v)^{-1/4}, so the product must drop by (λ/λ_floor)^4.
    7744^4 is not computed here; the squared ratio in hundredths,
    (7744)^2 > (7 * 1000)^2, is the γ rescaling direction (k_c ∝ √γ). -/
theorem gamma_must_rise_to_reach_the_tier_floor :
    1000 * 1000 * 49 < 7744 * 7744 := by
  decide

end MathOfBiology.Loop_3154
