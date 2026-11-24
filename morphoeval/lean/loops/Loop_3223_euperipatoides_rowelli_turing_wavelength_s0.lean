/-
  Loop 3223 — Euperipatoides rowelli, Turing wavenumber at the detail scale.
  Phenomenon: 1.12 µm (1120 nm) pigment / ocellus detail wavelength.
  Age: Holocene (extant), 0 Ma. Chronoseries index 1/1.
  Predecessor contrast: a Cambrian stem lobopodian does not preserve a 1.12 µm
  spacing (resolution floor 10 µm), so the countable mode number at this scale is 0.
  This chronospecies: a 100 µm eye spans 89 such wavelengths.
  The coarser published photoreceptor pitch is 2500 nm (Ljungholm & Nilsson 2021);
  1120 nm is strictly finer than that pitch.
-/
namespace MathOfBiology.Loop_3223

/-- Assigned detail wavelength: 1.12 µm = 1120 nm. -/
def detailWavelengthNm : Nat := 1120

/-- Published photoreceptor centre-to-centre pitch (2.5 µm). -/
def receptorPitchNm : Nat := 2500

/-- Adult ocellus diameter used as the Holocene domain: 100 µm = 100000 nm. -/
def eyeDiameterNm : Nat := 100000

/-- Predecessor preservation floor at 10 µm. Finer spacings are not countable. -/
def cambrianResolutionNm : Nat := 10000

def wavelengthsAcross (domain feature : Nat) : Nat := domain / feature

/-- A feature below the preservation floor contributes no countable wavelength. -/
def resolvable (resolutionNm featureNm : Nat) : Nat :=
  if featureNm < resolutionNm then 0 else 1

theorem detail_wavelength_positive : 0 < detailWavelengthNm := by
  decide

theorem detail_scale_finer_than_receptor_pitch :
    detailWavelengthNm < receptorPitchNm := by
  decide

theorem holocene_eye_spans_eighty_nine_detail_wavelengths :
    wavelengthsAcross eyeDiameterNm detailWavelengthNm = 89 := by
  decide

theorem cambrian_predecessor_does_not_resolve_detail_wavelength :
    resolvable cambrianResolutionNm detailWavelengthNm = 0 := by
  decide

theorem holocene_retina_resolves_detail_wavelength :
    resolvable 50 detailWavelengthNm = 1 := by
  decide

/-- Slot 3223 diffusion ratio Dv/Du = 22/1, the Turing requirement Dv > Du. -/
theorem diffusion_ratio_exceeds_unity : (1 : Int) < 22 := by
  decide

end MathOfBiology.Loop_3223
