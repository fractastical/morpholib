/-
  Loop 3150 — Clytia hemisphaerica, Turing wavelength at the 8.64 µm detail scale.
  Holocene chronoseries 1/1 (age 0 Ma). No fossil predecessor, so the temporal
  increment is zero. The scale contrast is the mode count: a 250 µm planula
  holds fewer repeats of λ = 8640 nm than a 2 mm medusa bell.
  Integer encoding of the Schnakenberg margin at Dv/Du = 20, f_u = 4/5, |g_v| = 1:
  multiply (4/5 r − 1) by 5 to clear the denominator.
-/
namespace MathOfBiology.Loop_3150

/-- Cleared Turing margin 5·((4/5)·ratio − 1) = 4·ratio − 5. -/
def turingMargin (ratio : Nat) : Int := (4 : Int) * ratio - 5

theorem slot_ratio_is_turing_unstable : turingMargin 20 > 0 := by
  decide

/-- Selected detail-scale wavelength, in nanometres (8.64 µm). -/
def selectedWavelengthNm : Nat := 8640

theorem selected_wavelength_is_detail_scale :
    selectedWavelengthNm = 8640 ∧ 0 < selectedWavelengthNm := by
  decide

/-- Repeats of the 8640 nm mode that fit in a body of the given length. -/
def modeCount (lengthNm : Nat) : Nat := lengthNm / 8640

/-- 250 µm planula versus 2 mm bell. The detail slot resolves fewer stripes. -/
theorem detail_scale_has_fewer_modes_than_bell :
    modeCount 250000 < modeCount 2000000 := by
  decide

/-- λ ∝ √D_u at fixed rate and diffusion ratio, so quadrupling D_u doubles λ. -/
theorem wavelength_scales_with_sqrt_diffusivity :
    (17280 : Nat) * 17280 = 4 * (8640 * 8640) := by
  decide

/-- Discriminant clearance (4r − 5)² − 100 r > 0 encodes (0.8 r − 1)² > 4 r. -/
def discriminantClearance (ratio : Nat) : Int :=
  let m : Int := 4 * (ratio : Int) - 5
  m * m - 100 * (ratio : Int)

theorem supercritical_discriminant_at_slot_ratio :
    discriminantClearance 20 > 0 := by
  decide

/-- Sole chronospecies in this 1/1 series. Age delta to a predecessor is undefined
    because there is none; the terminus itself is 0 Ma. -/
def holoceneAgeMa : Nat := 0

theorem no_temporal_predecessor : holoceneAgeMa = 0 := by
  decide

end MathOfBiology.Loop_3150
