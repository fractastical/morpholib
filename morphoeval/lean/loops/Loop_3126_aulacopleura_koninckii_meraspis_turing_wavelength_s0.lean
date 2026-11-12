/-
  Loop 3126 — Aulacopleura koninckii meraspis, detail-scale Turing wavelength.
  Characteristic scale: 3.125 µm = 3125 nm (tier 3_subcellular_um, detail variant).
  Age: Silurian Wenlock, 428.5 Ma. Chronoseries 1/3.
  Successor holaspis18 has 18 thoracic segments; this meraspis has 12.
  The subcellular wavenumber is not the 183 µm primordium mode (183100 nm).
-/
namespace MathOfBiology.Loop_3126

/-- Critical Turing wavelength in nanometres (3.125 µm). -/
def lambdaNm : Nat := 3125

/-- Thoracic segments released between meraspid degree 12 and holaspis 18. -/
def segmentSurplus : Nat := 18 - 12

/-- Diffusion ratio and onset threshold in parts per thousand (r = 6.683, r_crit = 6.552). -/
def diffusionRatioPpt : Nat := 6683
def onsetRatioPpt : Nat := 6552

theorem segment_surplus_is_six : segmentSurplus = 6 := by
  decide

theorem detail_wavelength_below_ten_micrometres : lambdaNm < 10000 := by
  decide

/-- Sixteen critical wavelengths tile a 50 µm (50 000 nm) pleural patch. -/
theorem pleural_patch_tiles_sixteen : 16 * lambdaNm = 50000 := by
  decide

/-- This detail-tier wavelength is finer than the 183 µm primordium sibling. -/
theorem detail_finer_than_primordium_sibling : lambdaNm < 183100 := by
  decide

/-- The operating diffusion ratio sits strictly above the integer onset threshold. -/
theorem above_turing_onset : diffusionRatioPpt > onsetRatioPpt := by
  decide

end MathOfBiology.Loop_3126
