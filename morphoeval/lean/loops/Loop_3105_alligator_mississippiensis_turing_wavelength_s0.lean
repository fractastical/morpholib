/-
  Loop 3105 — Alligator mississippiensis, Holocene (0 Ma), chronoseries 1/1.
  Detail scale λ_c = 1498 µm (1.498 mm) on a 17976 µm × 11984 µm flank patch.
  Schnakenberg Jacobian scaled by 100 at the homogeneous state (u, v) = (1, 87/100):
    f_u = 74, f_v = 100, g_u = -174, g_v = -100 (units of γ/100).
  Diffusion integers D_u = 5, D_v = 82 stand for 0.05 and 0.82 mm²/time (ratio 16.4).
  No in-series predecessor or successor: series_index = series_length and age_ma = 0.
-/
namespace MathOfBiology.Loop_3105

def lambdaCu : Nat := 1498
def flankLengthUm : Nat := 17976
def flankHeightUm : Nat := 11984

theorem twelve_critical_wavelengths_span_the_flank :
    lambdaCu * 12 = flankLengthUm := by decide

theorem eight_critical_wavelengths_span_the_height :
    lambdaCu * 8 = flankHeightUm := by decide

def traceJacobian : Int := 74 + (-100)

theorem diffusion_free_state_is_asymptotically_stable :
    traceJacobian < 0 := by decide

def detJacobian : Int := 74 * (-100) - (100 * (-174))

theorem activator_inhibitor_determinant_is_positive :
    detJacobian = 10000 := by decide

/-- D_v f_u + D_u g_v. Positive only when inhibitor diffusion outruns activator diffusion. -/
def diffusionMargin (Du Dv : Int) : Int := Dv * 74 + Du * (-100)

theorem slot_3105_ratio_opens_a_turing_band :
    diffusionMargin 5 82 = 5568 ∧ diffusionMargin 5 82 > 0 := by decide

theorem equal_diffusion_closes_the_band :
    diffusionMargin 5 5 = -130 ∧ diffusionMargin 5 5 < 0 := by decide

theorem turing_inequality_holds_for_the_1498_um_mode :
    diffusionMargin 5 82 * diffusionMargin 5 82 >
      4 * 5 * 82 * detJacobian := by decide

def seriesIndex : Nat := 1
def seriesLength : Nat := 1
def ageMa : Nat := 0

theorem holocene_chronoseries_is_a_single_extant_point :
    seriesIndex = seriesLength ∧ ageMa = 0 := by decide

end MathOfBiology.Loop_3105
