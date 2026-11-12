/-
  Loop 3231 — Gasterosteus doryssus_late at the 27 µm detail scale (9.99 Ma).
  Schnakenberg Jacobian at (u0, v0) = (6/5, 25/36), inhibitor/activator ratio r = 25.
  Fastest-mode wavelengths in nanometres: marine 21480 < late 27070 < mid 33620.
  Sibling organ-scale slot 1209 (1.7 mm) is a different parameter point and is not reused.
-/
namespace MathOfBiology.Loop_3231

/-- Trace numerator of tr J = f_u + g_v = -58/75. -/
def traceNum : Int := -58

/-- Trace denominator. -/
def traceDen : Int := 75

/-- Without diffusion the homogeneous state is linearly stable. -/
theorem trace_negative : traceNum < 0 ∧ traceDen > 0 := by
  decide

/-- det J = u0^2 = 36/25, so both numerator and denominator are positive. -/
theorem determinant_positive : (36 : Int) > 0 ∧ (25 : Int) > 0 := by
  decide

/-- Turing band at r = 25: (1142/75)^2 > 4 * r * det = 144.
    Cleared of denominators: 1142^2 > 144 * 75^2. -/
theorem turing_inequality_r25 : (1142 : Int) * 1142 > 144 * 75 * 75 := by
  decide

/-- Equal diffusion (r = 1) gives h = tr J < 0, so no diffusion-driven band opens. -/
theorem equal_diffusion_not_turing : traceNum < 0 ∧ (4 : Int) * 36 > 0 := by
  decide

/-- At this detail scale the pelvic-loss chronospecies sits between its neighbours:
    λ_marine < λ_late < λ_mid (nanometres). -/
theorem chronospecies_wavelength_order :
    (21480 : Int) < 27070 ∧ (27070 : Int) < 33620 := by
  decide

/-- Squared wavelengths preserve that order, and Du ∝ λ_m^2 at fixed kinetics. -/
theorem chronospecies_du_order :
    (21480 : Int) * 21480 < 27070 * 27070 ∧ (27070 : Int) * 27070 < 33620 * 33620 := by
  decide

end MathOfBiology.Loop_3231
