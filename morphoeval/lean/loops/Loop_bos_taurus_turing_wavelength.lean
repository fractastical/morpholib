/-
  morphoeval/lean/loops/Loop_bos_taurus_turing_wavelength.lean
  Formal check for the Bos taurus Turing-wavelength loop (2026-10-26).

  The Schnakenberg Jacobian at the homogeneous steady state (u0, v0) = (1, 0.9),
  scaled by 5 to integers, is J5 = [[4, 5], [-9, -5]]  (i.e. J = J5 / 5).
  The three theorems below verify, for D_u = 1, D_v = 20:
    1. stability without diffusion:  tr(J) < 0 and det(J) > 0
       (checked on the scaled integers: tr(J5) = -1, det(J5) = 25);
    2. the Turing inequality (D_v f_u + D_u g_v)^2 > 4 D_u D_v det(J),
       in scaled form (20*4 + 1*(-5))^2 = 75^2 = 5625 > 2000 = 4*1*20*25;
    3. the equal-diffusion control cannot destabilise: with D_v = D_u the
       cross term is 1*4 + 1*(-5) = -1 <= 0.
  These mirror `turing_analysis` in
  models/1981-murray-bos-taurus-turing-wavelength.py (lines 107-137).
-/
namespace MathOfBiology.BosTaurusTuring

/-- Scaled trace: 5 * tr(J) = 4 + (-5) = -1 < 0 (stable at k = 0). -/
theorem trace_negative : (4 : Int) + (-5) < 0 := by decide

/-- Scaled determinant: 25 * det(J) = 4*(-5) - 5*(-9) = 25 > 0. -/
theorem det_positive : (0 : Int) < 4 * (-5) - 5 * (-9) := by decide

/-- Turing inequality (scaled): (D_v f_u + D_u g_v)^2 > 4 D_u D_v det(J). -/
theorem turing_inequality : 4 * 1 * 20 * (4 * (-5) - 5 * (-9)) < (20 * 4 + 1 * (-5)) ^ 2 := by
  decide

/-- Equal-diffusion control: D_v = D_u gives cross term -1, so no Turing band. -/
theorem equal_diffusion_no_turing : (1 : Int) * 4 + 1 * (-5) ≤ 0 := by decide

end MathOfBiology.BosTaurusTuring
