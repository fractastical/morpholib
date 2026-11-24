/-
  Loop 3144 — Holocene Chamaeleo calyptratus, Turing wavelength at 260.4 µm.
  Chronoseries 1/1 (extant, 0 Ma). The predecessor/successor contrast is the
  neighbouring SCALE, not another chronospecies:
    predecessor cutoff 80.0 µm = 800 tenths of a micrometre (sub-wavelength);
    this carrier λ_c = 260.4 µm = 2604 tenths;
    successor envelope = 10 λ_c = 26040 tenths = 2.604 mm (an adult bar made
    of many carrier wavelengths).
  All proofs are `decide` / `rfl` on integers. No `sorry`.
-/

namespace MathOfBiology.Loop_3144

/-- Carrier wavelength of this slot, in tenths of a micrometre (2604 = 260.4 µm). -/
def lambdaTenths : Nat := 2604

/-- Sub-wavelength dermal-unit cutoff (80.0 µm). A domain this short holds no full mode. -/
def predecessorTenths : Nat := 800

/-- Adult lateral-bar envelope used as the successor scale: ten carrier wavelengths. -/
def successorTenths : Nat := 26040

/-- Holocene age of the extant chronospecies, in Ma. The series has a single point. -/
def ageMa : Nat := 0

/-- Positive integer stand-in for the Turing numerator Dv·f_u + Du·g_v
    at the slot's supercritical ratio (both diffusivities and both Jacobian
    weights taken positive after the sign of g_v is absorbed into the
    inhibitor-diffusion excess). The value is specific to slot 3144. -/
def kcNumerator : Nat := 145 * 62 + 42 * 15

theorem holocene_extant_age : ageMa = 0 := by
  rfl

theorem predecessor_is_subwavelength : predecessorTenths < lambdaTenths := by
  decide

theorem successor_is_ten_wavelengths : successorTenths = 10 * lambdaTenths := by
  decide

theorem successor_exceeds_carrier : lambdaTenths < successorTenths := by
  decide

/-- Eight carrier wavelengths fit in a 2083.2 µm flank patch (20832 tenths). -/
theorem eight_modes_on_flank_patch : 20832 / lambdaTenths = 8 := by
  decide

theorem kc_numerator_positive : 0 < kcNumerator := by
  decide

end MathOfBiology.Loop_3144
