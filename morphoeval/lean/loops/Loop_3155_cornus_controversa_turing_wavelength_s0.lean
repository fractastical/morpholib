-- Loop 3155 — Cornus controversa, Turing wavenumber at 1.939 mm.
-- Scale: 0.001939 m (log10 = -2.712), tier 6_organ_mm, Holocene extant, 0 Ma.
-- Chronoseries 1/1: no temporal predecessor. Integers are thousandths of the
-- Schnakenberg Jacobian (f_u = 0.880, g_v = -1, g_u = -1.880, f_v = 1)
-- and ten-thousandths of the diffusivities that pin λ_c = 1.939 mm
-- (D_u = 0.0378 mm²/time, D_v = 0.4434 mm²/time).
namespace MathOfBiology.Loop_3155

/-- Trace of the Schnakenberg Jacobian. Stable kinetics require tr J < 0. -/
def traceJacobian (fu gv : Int) : Int := fu + gv

theorem trace_stable_without_diffusion :
    traceJacobian 880 (-1000) < 0 := by
  decide

/-- det J = f_u g_v - f_v g_u, scaled so the real determinant 1.000 is 1_000_000. -/
def determinant (fu fv gu gv : Int) : Int := fu * gv - fv * gu

theorem determinant_positive :
    determinant 880 1000 (-1880) (-1000) > 0 := by
  decide

/-- Critical wavelength identity: k_c ≈ 3.240 mm⁻¹, λ_c = 1.939 mm, so
    k_c λ_c is within 2×10⁻³ of 2π (integers: k×10³ and λ×10³, 2π×10⁶). -/
theorem wavelength_closes_on_two_pi :
    6283185 - 1939 * 3240 < 2000 ∧ 6283185 - 1939 * 3240 > 0 := by
  decide

/-- Inhibitor-weighted contrast h = f_u D_v + g_v D_u > 0
    (D_u, D_v stored in ten-thousandths). Equal diffusion flips the sign
    because f_u + g_v = tr J < 0, which is the ablation control. -/
theorem inhibitor_contrast_opens_a_band :
    880 * 4434 + (-1000) * 378 > 0 := by
  decide

theorem equal_diffusion_contrast_is_negative :
    378 * (880 + (-1000)) < 0 := by
  decide

/-- Chronoseries 1/1 at 0 Ma: the gap to a successor and the age are both zero.
    There is no predecessor chronospecies whose parameters this slot could copy. -/
def chronoseriesGap (index length ageMa : Int) : Int := (length - index) + ageMa

theorem holocene_singleton_has_no_temporal_successor :
    chronoseriesGap 1 1 0 = 0 := by
  decide

end MathOfBiology.Loop_3155
