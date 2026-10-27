/-
  Loop 3213 — Dugesia japonica, Holocene (0 Ma), Turing wavelength 6.912 µm.

  Saturated Schnakenberg gap at diffusion ratio R, in milli-rate units rounded
  from the Jacobian at κ = 0.15: f_u ≈ 0.468 → 468, −g_v ≈ 1.598 → 1598.
  The gap r * 468 − 1598 is the sign of D_v f_u + D_u g_v. Positive gap is the
  necessary diffusion-driven condition; R = 1 (equal diffusion) fails it.

  The detail patch is exactly seven critical wavelengths:
  6912 nm × 7 = 48384 nm. Chronoseries length is 1, so the predecessor count is 0.
-/
namespace MathOfBiology.Loop_3213

def turing_gap (r fu_milli gv_removal_milli : Int) : Int :=
  r * fu_milli - gv_removal_milli

theorem slot3213_diffusion_gap_positive :
    turing_gap 48 468 1598 > 0 := by
  decide

theorem homogeneous_trace_negative :
    (468 : Int) - 1598 < 0 := by
  decide

theorem seven_wavelengths_fill_detail_field :
    (6912 : Int) * 7 = 48384 := by
  decide

theorem equal_diffusion_ablation_is_not_turing :
    turing_gap 1 468 1598 < 0 := by
  decide

theorem higher_diffusion_ratio_widens_the_gap :
    turing_gap 48 468 1598 > turing_gap 36 468 1598 := by
  decide

theorem holocene_chronoseries_has_no_predecessor :
    (1 : Int) - 1 = 0 := by
  decide

end MathOfBiology.Loop_3213
