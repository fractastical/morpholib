/-
  Loop 3234 — Gasterosteus aculeatus benthic, cellular Turing wavelength.
  Scale: 43 µm (tier 4_cellular_10um), Holocene post-glacial lake, 0 Ma.
  Predecessor organ-scale marine wavelength: 2400 µm at 0.015 Ma.
  Same-chronospecies organ-scale bar (not this slot): 2200 µm.
  Kinetics integerised from Schnakenberg (a=1/10, b=9/10, γ=3/10, Du=11/20, Dv=12):
    fu, fv, gu, gv = 240, 300, -540, -300   (reaction entries × 1000)
    Du, Dv         = 55, 1200                (diffusivities × 100)
  No `sorry`. Core Lean 4 only.
-/
namespace MathOfBiology.Loop_3234

/-- Critical wavelength in micrometres at the cellular tier (rounded assignment). -/
def lambdaDetailUm : Nat := 43

/-- Benthic organ-scale flank-bar wavelength in micrometres (sibling tier, 2.2 mm). -/
def lambdaBenthicOrganUm : Nat := 2200

/-- Marine predecessor organ-scale wavelength in micrometres (2.4 mm, 0.015 Ma). -/
def lambdaMarineOrganUm : Nat := 2400

/-- Law 1. At 0 Ma the cellular mode is strictly shorter than both organ-scale bars. -/
theorem cellular_wavelength_shorter_than_organ_bars :
    lambdaDetailUm < lambdaBenthicOrganUm ∧ lambdaDetailUm < lambdaMarineOrganUm := by
  decide

/-- Wavenumbers in reciprocal units normalised so k · λ equals 2200 · 43. -/
def kDetail : Nat := 2200
def kOrganBenthic : Nat := 43

/-- Law 2. k_c · λ_c is the same product at both tiers, and k_c is larger at 43 µm. -/
theorem wavenumber_inverts_when_wavelength_shrinks :
    kDetail * lambdaDetailUm = kOrganBenthic * lambdaBenthicOrganUm ∧
    kDetail > kOrganBenthic := by
  decide

/-- A 258 µm continuous dermal window (six cellular wavelengths, plates absent). -/
theorem six_cellular_modes_on_low_plate_dermis :
    (258 : Nat) / lambdaDetailUm = 6 := by
  decide

/-- Trace of the integerised Jacobian is negative (stable without diffusion). -/
theorem homogeneous_trace_negative :
    (240 : Int) + (-300 : Int) < 0 := by
  decide

/-- Determinant fu·gv − fv·gu stays positive, so the homogeneous state is a sink. -/
theorem homogeneous_determinant_positive :
    (240 : Int) * (-300) - (300 : Int) * (-540) > 0 := by
  decide

/-- Diffusion-driven coefficient h = Dv·fu + Du·gv is positive (long-range inhibition). -/
theorem diffusion_drive_positive :
    (1200 : Int) * 240 + (55 : Int) * (-300) > 0 := by
  decide

/-- Law 3. Turing discriminant h² − 4·Du·Dv·det(J) is positive at these rates,
    so a band of wavenumbers around k_c is unstable. -/
theorem turing_discriminant_positive :
    (271500 : Int) * 271500 - 4 * 55 * 1200 * 90000 > 0 := by
  decide

/-- Holocene benthic horizon (0 ka) is later than the marine predecessor (15 ka). -/
theorem holocene_benthic_is_after_marine_predecessor :
    (0 : Nat) < 15 := by
  decide

/-- Low-plate benthic armor (at most 7 plates) is below the marine complete morph (32). -/
theorem low_plate_below_marine_complete_morph :
    (7 : Nat) < 32 := by
  decide

end MathOfBiology.Loop_3234
