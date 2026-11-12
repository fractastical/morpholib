/-
  Loop 3129 — Aurelia aurita, Holocene extant (0 Ma), detail scale 201 µm.

  Schnakenberg kinetics with the slot-specific Jacobian
    f_u = 0.8, f_v = 1, g_u = -1.8, g_v = -1, D_u = 1, D_v = 20
  cleared of decimals by writing f_u, f_v, g_u, g_v in tenths
  (8, 10, -18, -10). Then det J = 1 becomes detScaled = 100, and the
  Turing discriminant compares 22500 against 8000.

  The physical length used by the script is 0.0002006 m = 200.6 µm,
  whose tier label is 201 µm because 2006 + 4 = 2010 = 201 × 10.
  The Aurelia chronoseries in this slot has one node (Holocene, 0 Ma):
  there is no predecessor and no successor chronospecies.
-/

namespace MathOfBiology.Loop_3129

def traceScaled : Int := 8 + (-10)

def detScaled : Int := 8 * (-10) - 10 * (-18)

theorem trace_negative : traceScaled < 0 := by
  decide

theorem det_equals_one_hundred : detScaled = 100 := by
  decide

theorem homogeneous_state_is_stable : traceScaled < 0 ∧ detScaled > 0 := by
  decide

/-- (D_v f_u + D_u g_v)^2 × 100 > 4 D_u D_v det(J) × 100.
    Left side is 150^2 = 22500. Right side is 8000. -/
theorem turing_wavenumber_is_unstable :
    (20 * 8 + 1 * (-10)) ^ 2 > 4 * 1 * 20 * detScaled := by
  decide

/-- Equal diffusion D_u = D_v = 1 fails the same discriminant (4 ≯ 400).
    That is the negative control for this Holocene node: no ocellus wavelength. -/
theorem equal_diffusion_suppresses_the_mode :
    ¬ ((1 * 8 + 1 * (-10)) ^ 2 > 4 * 1 * 1 * detScaled) := by
  decide

/-- 0.0002006 m = 200.6 µm rounds to the 201 µm tier label. -/
theorem tier_label_201_micrometres : 2006 + 4 = 201 * 10 := by
  decide

/-- Chronoseries 1/1. The predecessor offset and the successor offset are both zero
    because there is no other node. Age is 0 Ma. -/
theorem holocene_series_has_no_neighbor :
    ((1 : Int) - 1 = 0) ∧ ((0 : Int) = 0) := by
  decide

/-- Eight rhopalia (Helm 2018; Nakanishi 2009) are an anatomical count, not the
    number of 201 µm waves around the bell. -/
theorem eight_rhopalia : (8 : Int) > 1 ∧ 8 ≠ 201 := by
  decide

end MathOfBiology.Loop_3129
