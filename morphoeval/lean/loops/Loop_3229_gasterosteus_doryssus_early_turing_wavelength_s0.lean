/-
  Loop 3229 — Gasterosteus doryssus_early at the 33 µm cellular-detail rung.
  Age 10.02 Ma (Truckee Formation), full pelvic girdle, T = 0 kyr.
  λ_c = 32.86 µm = 3.286e-5 m. Successor doryssus_mid lengthens to 41.50 µm.
  Integer quantities are hundredths of a micrometre, or Jacobian entries × 10^4.
-/
namespace MathOfBiology.Loop_3229

/-- Critical wavelength of the early chronospecies, in hundredths of a micrometre. -/
def lambdaEarly : Nat := 3286

/-- Analytic successor wavelength (doryssus_mid, 10.01 Ma, vestigial spine). -/
def lambdaMid : Nat := 4150

/-- Marine complete-morph outgroup. Not a chronoseries predecessor. -/
def lambdaMarine : Nat := 2840

/-- Visible high-pass band on the gallery ocellus, hundredths of a micrometre.
    The script measures 81.08 µm (8108); that band is not λ_c. -/
def specimenBand : Nat := 8108

/-- Dermal patch side, hundredths of a micrometre (280.00 µm). -/
def patchSide : Nat := 28000

/-- Jacobian trace and determinant scaled by 10^4 for the locked (a, b). -/
def traceJac : Int := 8772 + (-10201)
def detJac : Int := 10201

/-- Activator and inhibitor diffusivities in thousandths of µm² per time. -/
def duMilli : Nat := 8204
def dvMilli : Nat := 100721

theorem successor_lengthens_cellular_wavelength :
    lambdaEarly < lambdaMid := by
  decide

theorem early_is_longer_than_marine_outgroup :
    lambdaMarine < lambdaEarly := by
  decide

theorem patch_holds_eight_to_nine_critical_wavelengths :
    8 * lambdaEarly < patchSide ∧ patchSide < 9 * lambdaEarly := by
  decide

theorem specimen_band_is_not_the_critical_wavelength :
    lambdaEarly < specimenBand ∧ specimenBand < 3 * lambdaEarly := by
  decide

theorem jacobian_is_stable_without_diffusion :
    traceJac < 0 ∧ detJac > 0 := by
  decide

theorem inhibitor_diffuses_farther_than_activator :
    duMilli < dvMilli := by
  decide

end MathOfBiology.Loop_3229
