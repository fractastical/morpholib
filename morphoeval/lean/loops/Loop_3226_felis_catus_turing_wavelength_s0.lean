/-
  Loop 3226 — Felis catus, Turing wavenumber at 436 µm, Holocene (0 Ma).

  The Felis chronoseries in this slot has a single chronospecies, so there is no
  earlier species to compare. The theorems pin what changes across scale instead:
  the cellular predecessor (80 µm) is shorter than one wavelength, this primordium
  wavelength is 436 µm, and the successor flank (4360 µm) holds ten such waves.
  Inhibitor diffusion exceeds activator diffusion, and the integer form of
  D_v f_u + D_u g_v is positive only when that separation is present.
-/

namespace MathOfBiology.Loop_3226

def wavelengthUm : Int := 436

def ageMa : Int := 0

/-- Turing wavelength sits inside the primordium tier (100 µm–1 mm), detail 436 µm. -/
theorem wavelength_in_primordium_tier :
    100 < wavelengthUm ∧ wavelengthUm < 1000 := by
  decide

/-- Holocene chronospecies, series 1/1. Age is 0 Ma. -/
theorem felis_chronoseries_is_extant_holocene :
    ageMa = 0 := by
  rfl

/-- Cellular-tier predecessor (80 µm) cannot fit one 436 µm wave, so its fundamental mode is unpatterned. -/
theorem cellular_predecessor_shorter_than_wavelength :
    (80 : Int) < wavelengthUm := by
  decide

/-- Successor flank is ten wavelengths of the same k*, not a new wavenumber. -/
theorem successor_flank_holds_ten_wavelengths :
    (10 : Int) * wavelengthUm = 4360 := by
  decide

/-- Inhibitor diffuses farther than the activator (D_v = 1.742, D_u = 0.031, in thousandths). -/
theorem inhibitor_diffuses_farther :
    (1742 : Int) > 31 := by
  decide

/-- D_v f_u + D_u g_v > 0 at the Holocene Jacobian (thousandths). Equal diffusion fails this test. -/
def diffusionDrive (dv du fu gv : Int) : Int := dv * fu + du * gv

theorem diffusion_driven_instability :
    diffusionDrive 1742 31 5304 (-9510) > 0 := by
  decide

theorem equal_diffusion_drive_is_negative :
    diffusionDrive 31 31 5304 (-9510) < 0 := by
  decide

end MathOfBiology.Loop_3226
