/-
  Loop 3110 — Anas platyrhynchos within-feather Turing wavelength.
  Scale 378.4 µm (0.0003784 m) at 0 Ma (Holocene). Schnakenberg Jacobian
  entries are stored as integers ×1000. Lengths that must stay exact are
  stored as tenths of a micrometre.
-/
namespace MathOfBiology.Loop_3110

/-- Homogeneous Schnakenberg steady state u* = a + b, with a = 0.10 and b = 0.90 scaled by 100. -/
def uStar : Int := 10 + 90

theorem steady_state_is_sum : uStar = 100 := by
  decide

/-- Trace f_u + g_v = 0.800 + (−1.000) = −0.200, scaled by 1000. -/
def traceScaled : Int := 800 + (-1000)

theorem trace_is_negative : traceScaled = -200 ∧ traceScaled < 0 := by
  decide

/-- Determinant f_u g_v − f_v g_u scaled by 1000².
    800*(−1000) − 1000*(−1800) = 1_000_000, i.e. det = +1. -/
def detScaled : Int := 800 * (-1000) - 1000 * (-1800)

theorem determinant_is_positive : detScaled = 1000000 ∧ 0 < detScaled := by
  decide

/-- Classical Turing discriminant at r = 16, Du = 1, scaled so the comparison is integral.
    (Dv f_u + Du g_v) = 16*0.8 − 1 = 11.8 → 118 (×10).
    4 Du Dv det = 64. 118² = 13924 > 6400 = 64 × 100. -/
theorem discriminant_above_onset : 118 * 118 > 64 * 100 := by
  decide

/-- Equal diffusion (r = 1) is below the integer onset gate r > 9
    (analytic onset for this Jacobian is r_c ≈ 8.57). Nominal r = 16 is above it. -/
theorem equal_diffusion_below_onset : 1 < 9 ∧ 9 < 16 := by
  decide

/-- Organism-scale sibling wavelength 23540.0 µm versus this detail wavelength 378.4 µm,
    both in tenths of a micrometre: 62 detail wavelengths plus a 79.2 µm remainder. -/
theorem detail_wavelengths_inside_organism_sibling : 62 * 3784 + 792 = 235400 := by
  decide

end MathOfBiology.Loop_3110
