/-
  morphoeval/lean/loops/Loop_3124_asteroxylon_mackiei_turing_wavelength_s0.lean
  Slot 3124 — Asteroxylon mackiei, detail-tier Turing wavelength.
  Scale: 5.2 µm (5.2e-6 m, tier 3_subcellular_um, variant s0).
  Age: Early Devonian, Pragian Rhynie Chert, 407 Ma.
  Law: Schnakenberg critical wavenumber k_c = 2π / λ_c.
  Integer encoding (nanometres, hundredths of Jacobian entries) so the file
  stands alone in Lean 4 with no mathlib import and no sorry.
-/
namespace MathOfBiology.Loop_3124

/-- Law 1. Diffusion-ratio threshold at 5.2 µm.
    δ = 160, f_u = 84/100, and δ f_u > 1 is required before any finite k_c exists.
    `law1_threshold` is δ f_u_num − f_u_den, positive when the inequality holds. -/
def law1_threshold (delta fu_num fu_den : Int) : Int :=
  delta * fu_num - fu_den

theorem law1_diffusion_ratio_exceeds_unity :
    law1_threshold 160 84 100 > 0 := by
  decide

/-- Law 2. Thirty critical wavelengths tile the 156 µm tracheid-file window.
    156000 nm / 5200 nm = 30. This count is the detail-scale domain, not the
    single-wavelength 283 µm enation domain. -/
def law2_mode_count (domain_nm lambda_nm : Int) : Int :=
  domain_nm / lambda_nm

theorem law2_thirty_critical_modes :
    law2_mode_count 156000 5200 = 30 := by
  decide

/-- Law 3. The 5.2 µm wavenumber dominates the 283.4 µm enation sibling.
    Integer division 283400 / 5200 = 54, which is already greater than 50.
    Cooksonia has no actinostele to carry this mode; the ratio is the change
    inside Asteroxylon between organ spacing and the subcellular stripe. -/
def law3_scale_ratio (sibling_nm detail_nm : Int) : Int :=
  sibling_nm / detail_nm

theorem law3_subcellular_wavenumber_dominates_enation :
    law3_scale_ratio 283400 5200 > 50 := by
  decide

/-- Law 4. The homogeneous steady state is stable without diffusion.
    tr J in hundredths is f_u + g_v = 84 + (−100) < 0, so patterning requires
    the differential diffusion of Law 1 and is not a well-mixed oscillation. -/
def law4_trace (fu_hundredths gv_hundredths : Int) : Int :=
  fu_hundredths + gv_hundredths

theorem law4_homogeneous_trace_negative :
    law4_trace 84 (-100) < 0 := by
  decide

end MathOfBiology.Loop_3124
