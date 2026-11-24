/-
  Loop 3104 — *Acropora millepora* (staghorn coral), Holocene extant (0 Ma).
  Scale: Turing pigment-stripe / ocellus wavelength λ_c = 385.2 µm
  (3.852×10⁻⁴ m, log10 = −3.414), tier 5/8 primordium, detail variant.
  Chronoseries 1/1: the extant crown is the only node, so there is no
  in-series predecessor or successor. The inhibitor/activator ratio 15.75
  lies above the Schnakenberg onset threshold (~8.08); equal diffusion does not.
-/
namespace MathOfBiology.Loop_3104

/-- Selected wavelength in tenths of a micrometre (385.2 µm). -/
def lambdaCDeciUm : Nat := 3852

/-- Coenenchyme patch used for this detail scale: eight selected wavelengths. -/
def nWavelengths : Nat := 8

/-- Inhibitor:activator diffusivity ratio 15.75, in hundredths. -/
def diffRatioHundredths : Nat := 1575

/-- Equal-diffusion ablation, in hundredths. -/
def ablationRatioHundredths : Nat := 100

/-- Onset threshold for the slot's Schnakenberg Jacobian, in hundredths (r_c ≈ 8.08). -/
def onsetRatioHundredths : Nat := 808

/-- Holocene age of the sole chronoseries node, in Ma. -/
def ageMa : Nat := 0

/-- Acropora millepora chronoseries length at this slot (extant crown only). -/
def seriesLength : Nat := 1

theorem domain_fits_fundamental_mode :
    nWavelengths * lambdaCDeciUm > lambdaCDeciUm := by
  decide

theorem inhibitor_exceeds_onset_and_ablation_does_not :
    diffRatioHundredths > onsetRatioHundredths ∧
      ablationRatioHundredths < onsetRatioHundredths := by
  decide

theorem holocene_is_terminal_singleton :
    seriesLength = 1 ∧ ageMa = 0 := by
  decide

/-- 385.2 µm sits inside the primordium tier, 100 µm ≤ λ ≤ 1000 µm, in deci-µm. -/
theorem primordium_tier_contains_wavelength :
    1000 ≤ lambdaCDeciUm ∧ lambdaCDeciUm ≤ 10000 := by
  decide

end MathOfBiology.Loop_3104
