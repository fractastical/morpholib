/-
  Loop 3143 — Ceratopteris richardii, Holocene extant (0 Ma).
  Scale: 256 µm primordium-detail Turing wavelength (tier 5/8, detail variant).
  Law: critical wavenumber k_c of a Schnakenberg activator–inhibitor system.
  These integers are the slot-3143 parameter set (D_v/D_u = 12, window = 3 λ_c).
  They are not the dimensionless thresholds used by any coarser sibling slot.
-/
namespace MathOfBiology.Loop_3143

/-- Observation scale, micrometres. -/
def observationScaleUm : Nat := 256

/-- Chronoseries age. Holocene extant, million years. -/
def holoceneAgeMa : Nat := 0

/-- Activator diffusivity, µm² s⁻¹, on the 256 µm window. -/
def diffusivityActivator : Nat := 8

/-- Inhibitor diffusivity, µm² s⁻¹. Twelve times the activator. -/
def diffusivityInhibitor : Nat := 96

/-- Three critical wavelengths across a primordium window. -/
def primordiumWindowUm : Nat := 768

/-- Counterfactual doubled-meristem predecessor. Not a scored fossil node. -/
def predecessorCounterfactualUm : Nat := 512

theorem observation_scale_is_256 :
    observationScaleUm = 256 := by
  rfl

theorem holocene_extant_is_present :
    holoceneAgeMa = 0 := by
  rfl

theorem inhibitor_diffuses_twelvefold :
    diffusivityActivator * 12 = diffusivityInhibitor := by
  decide

theorem window_holds_three_wavelengths :
    primordiumWindowUm = observationScaleUm * 3 := by
  decide

theorem predecessor_doubles_this_wavelength :
    predecessorCounterfactualUm = observationScaleUm * 2 := by
  decide

end MathOfBiology.Loop_3143
