/-
  Loop 3101 — Acanthometra pellucida, Holocene (0 Ma), Turing wavelength 206 nm.

  Schnakenberg Jacobian at (a, b) = (0.14, 0.76), cleared of denominators:
    f_u / gamma = 31/45,  g_v / gamma = -81/100.
  The diffusion-weighted trace, multiplied by 45*100 = 4500, is
    Dv * 3100 + Du * (-3645).
  Positive trace is required for a real critical wavenumber
    k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv).
  Reference inhibitor ratio Dv/Du = 18. Equal diffusion is the ablation.
  A 1339 nm cortical patch holds more 206 nm waves than 412 nm waves.
-/
namespace MathOfBiology.Loop_3101

def weightedTrace (Du Dv : Int) : Int := Dv * 3100 + Du * (-3645)

theorem reference_inhibitor_diffusion_is_unstable :
    weightedTrace 1 18 > 0 := by
  decide

theorem equal_diffusion_ablation_is_stable :
    weightedTrace 1 1 < 0 := by
  decide

theorem wavenumber_numerator_increases_with_inhibitor_diffusion :
    weightedTrace 1 8 < weightedTrace 1 18 := by
  decide

def stripeCount (patchNm wavelengthNm : Nat) : Nat := patchNm / wavelengthNm

theorem detail_scale_packs_more_stripes_than_coarser_octave :
    stripeCount 1339 206 > stripeCount 1339 412 := by
  decide

end MathOfBiology.Loop_3101
