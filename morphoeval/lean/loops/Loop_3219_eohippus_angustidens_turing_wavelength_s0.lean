/-
  Loop 3219 — Eohippus angustidens, cellular-detail Turing wavelength.
  Scale: 78 µm = 7.752e-05 m (tier 4_cellular_10um, variant s0), Wasatchian 52.0 Ma.
  Integer shadow of the Schnakenberg Jacobian at (a, b) = (0.062, 0.940),
  scaled by 10^6. Diffusion ratio 8.22 = 822/100 is this slot's ratio, not the
  3.3 mm organ-tier sibling's.
-/
namespace MathOfBiology.Loop_3219

def fu : Int := 876248
def fv : Int := 1004004
def gu : Int := -1876248
def gv : Int := -1004004

/-- det(J) × 10^12. -/
def detJ : Int := fu * gv - fv * gu

/-- Trace of the kinetic Jacobian is negative, so the homogeneous state is stable without diffusion. -/
theorem trace_negative : fu + gv < 0 := by
  decide

/-- Determinant of the kinetic Jacobian is positive. -/
theorem determinant_positive : detJ > 0 := by
  decide

/-- Equal diffusion (ratio 1) fails the Turing band inequality, so the pattern decays. -/
theorem equal_diffusion_is_stable :
    (fu + gv) * (fu + gv) < 4 * detJ := by
  decide

/-- Ratio 822/100 = 8.22 lies above onset: (r f_u + g_v)^2 > 4 r det(J). -/
theorem differential_diffusion_selects_a_band :
    (822 * fu + 100 * gv) * (822 * fu + 100 * gv) > 4 * 822 * 100 * detJ := by
  decide

/-- 77.520 µm sits strictly between 6 and 7 keratinocyte diameters of 12.000 µm. -/
theorem detail_wavelength_is_six_to_seven_cells :
    6 * 12000 < 77520 ∧ 77520 < 7 * 12000 := by
  decide

/-- Detail wavenumber over the 3.298 mm organ-scale sibling is strictly between 42 and 43. -/
theorem detail_wavenumber_exceeds_organ_scale :
    42 * 7752 < 329800 ∧ 329800 < 43 * 7752 := by
  decide

/-- Wasatchian 52.0 Ma precedes the Bridgerian Orohippus anchor at 48.0 Ma. -/
theorem wasatchian_precedes_bridgerian_successor :
    520 - 480 = 40 ∧ 40 > 0 := by
  decide

end MathOfBiology.Loop_3219
