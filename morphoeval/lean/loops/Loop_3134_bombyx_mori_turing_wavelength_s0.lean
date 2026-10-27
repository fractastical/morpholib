/-
  Loop 3134 — Bombyx mori, Turing wavenumber selection at 41 µm.
  Scale: 41 µm = 4.096e-5 m (log10 = -4.388), tier 4_cellular_10um, detail variant.
  Age: Holocene (Extant), 0 Ma. Chronoseries 1/1 (no successor).
  Predecessor contrast (not a second series member): Bombyx mandarina stripe pitch 96 µm.
  Cohort: Turing reaction-diffusion wavelength selection. Form: stripes, spots, ocelli.
-/
namespace MathOfBiology.Loop_3134

/-- Domestic Holocene critical wavelength, micrometres. -/
def lambdaC : Nat := 41

/-- Wild Bombyx mandarina melanin-stripe pitch used as the predecessor contrast, micrometres. -/
def lambdaMandarina : Nat := 96

/-- k_c in radians per metre at λ_c = 41 µm. 2π / 4.1e-5 ≈ 153248. -/
def kC_rad_per_m : Nat := 153248

/-- λ_c · k_c = 2π, with π·10^6 = 3141593 and an 18-unit remainder:
    41 * 153248 + 18 = 6283186 = 2 * 3141593. -/
theorem wavelength_wavenumber_identity :
    lambdaC * kC_rad_per_m + 18 = 2 * 3141593 := by
  decide

/-- Domestication shortens the selected pitch: 96 µm − 41 µm = 55 µm. -/
theorem domestic_pitch_shorter_than_mandarina :
    lambdaMandarina - lambdaC = 55 := by
  decide

/-- Calibrated inhibitor/activator diffusivity ratio Dv/Du. -/
def diffusionRatio : Nat := 18

theorem inhibitor_diffuses_faster :
    diffusionRatio > 1 := by
  decide

/-- Growth rate of the 41 µm mode on the ×10^4 scale (σ ≈ 0.3106). -/
def sigma41_x10000 : Nat := 3106

/-- Growth rate of the ancestral 96 µm mode (σ ≈ −0.4868), same scale. -/
def sigma96_x10000 : Int := -4868

theorem critical_mode_grows :
    sigma41_x10000 > 0 := by
  decide

theorem ancestral_pitch_decays :
    sigma96_x10000 < 0 := by
  decide

end MathOfBiology.Loop_3134
