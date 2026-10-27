/-
  Loop 3149 — Cloudina hartmannae at the 2.397 µm detail scale.
  Age: terminal Ediacaran, Nama Group, 549 Ma. Chronoseries 1/1.
  Law: Turing wavenumber k_c for an organic-lamella stack.
  Predecessor rung (not a second age): 104.2 µm funnel scale, slot 1252.
  Standalone Lean 4. No imports. No sorry.
-/
namespace MathOfBiology.Loop_3149

/-- Detail-scale wavelength in nanometres (2.397 µm = 2.397e-6 m). -/
def lambdaS0Nm : Nat := 2397

/-- Predecessor funnel rung in nanometres (104.2 µm). -/
def lambdaFunnelNm : Nat := 104200

/-- Yang et al. 2020 Namibian organic-lamella band, in nanometres. -/
def lamellaBandLowNm : Nat := 500
def lamellaBandHighNm : Nat := 5000

/-- Grant 1990 shell-layer thickness bounds, in nanometres. -/
def grantLayerLowNm : Nat := 8000
def grantLayerHighNm : Nat := 50000

/-- Maximum lamella count reported in a single Namibian wall layer. -/
def nLamellae : Nat := 8

def stackThicknessNm : Nat := nLamellae * lambdaS0Nm

/-- 2.397 µm lies strictly inside the 0.5–5 µm Namibian lamella band. -/
theorem lamella_wavelength_inside_yang_band :
    lamellaBandLowNm < lambdaS0Nm ∧ lambdaS0Nm < lamellaBandHighNm := by
  decide

/-- Eight lamellae at 2.397 µm span 19.176 µm, inside the 8–50 µm Grant layer. -/
theorem eight_lamella_stack_inside_grant_layer :
    grantLayerLowNm < stackThicknessNm ∧ stackThicknessNm < grantLayerHighNm := by
  decide

/-- This rung is shorter than the 104.2 µm funnel rung by a factor between 43 and 44. -/
theorem detail_scale_packs_between_43_and_44_funnel_wavelengths :
    lambdaS0Nm * 43 < lambdaFunnelNm ∧ lambdaFunnelNm < lambdaS0Nm * 44 := by
  decide

/-- Turing discriminant after both diffusivities are rescaled by a positive integer. -/
def discScaled (disc s : Int) : Int := s * s * disc

/-- λ ∝ √D, so a common positive rescaling of Du and Dv multiplies the discriminant
    by s² and cannot change its sign. The 2.397 µm rung is that rescaling of a
    longer-wavelength pair: (2.397/104.2)² ≈ 5.29e-4. -/
theorem diffusion_rescaling_preserves_turing_discriminant
    (disc s : Int) (hs : 0 < s) (hd : 0 < disc) :
    0 < discScaled disc s := by
  dsimp [discScaled]
  exact Int.mul_pos (Int.mul_pos hs hs) hd

/-- Critical-wavenumber balance on positive integer diffusivities and Jacobian entries:
    2 Du Dv kc2 = Dv fu + Du gv, rearranged so the cross-difference vanishes.
    This is the integer shadow of k_c² = (Dv fu + Du gv) / (2 Du Dv). -/
def wavenumberBalance (Du Dv fu gv kc2 : Int) : Int :=
  2 * Du * Dv * kc2 - (Dv * fu + Du * gv)

theorem critical_wavenumber_balance_is_exact
    (Du Dv fu gv kc2 : Int)
    (h : 2 * Du * Dv * kc2 = Dv * fu + Du * gv) :
    wavenumberBalance Du Dv fu gv kc2 = 0 := by
  dsimp [wavenumberBalance]
  omega

end MathOfBiology.Loop_3149
