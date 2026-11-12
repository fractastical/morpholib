/-
  Loop 3120 — Archaeopteryx lithographica, Turing wavenumber at the 75 µm cellular scale.
  Horizon: Late Jurassic (Tithonian), 150.0 Ma. Chronoseries point 1/1.
  Lineage neighbors (not additional Archaeopteryx horizons): Anchiornis huxleyi 160.8 Ma,
  Microraptor gui 120 Ma. Wavelengths are micrometres × 100 so the orderings are Nat facts.
  Organ-scale sibling (Loop 1230) is 2.919 mm and is not reused.
-/
namespace MathOfBiology.Loop_3120

/-- Archaeopteryx lithographica cellular wavelength, 74.78 µm, stored as µm × 100. -/
def lambdaArchUmX100 : Nat := 7478

/-- Anchiornis huxleyi isotropic-vane prediction at the same dx, µm × 100 (95.33 µm). -/
def lambdaAnchUmX100 : Nat := 9533

/-- Microraptor gui iridescent-stack prediction at the same dx, µm × 100 (57.05 µm). -/
def lambdaMicroUmX100 : Nat := 5705

/-- Loop-1230 organ-scale vane wavelength 2.919 mm, stored as µm × 100. -/
def lambdaOrganUmX100 : Nat := 291900

/-- Activator diffusivity 0.052, in milli-units. -/
def duMilli : Nat := 52

/-- Inhibitor diffusivity 1.48, in milli-units. -/
def dvMilli : Nat := 1480

/-- Successor cellular wavelength is shorter than the Tithonian Archaeopteryx wavelength. -/
theorem successor_shorter_than_archaeopteryx :
    lambdaMicroUmX100 < lambdaArchUmX100 := by
  decide

/-- Archaeopteryx cellular wavelength is shorter than the Oxfordian Anchiornis prediction. -/
theorem archaeopteryx_shorter_than_anchiornis :
    lambdaArchUmX100 < lambdaAnchUmX100 := by
  decide

/-- 75 µm cellular scale is shorter than the 2.919 mm organ-scale vane by a factor between 39 and 40. -/
theorem cellular_scale_between_39_and_40_organ_wavelengths :
    lambdaArchUmX100 * 39 < lambdaOrganUmX100 ∧ lambdaOrganUmX100 < lambdaArchUmX100 * 40 := by
  decide

/-- Inhibitor diffuses at least 28× faster than the activator (Dv/Du ∈ (28, 29)). -/
theorem differential_diffusivity_ratio :
    duMilli * 28 < dvMilli ∧ dvMilli < duMilli * 29 := by
  decide

/-- Eight critical wavelengths fit in a 600 µm vane patch: 8 × 74.78 µm < 600 µm. -/
theorem eight_wavelengths_fit_in_vane_patch :
    8 * lambdaArchUmX100 < 60000 := by
  decide

/-- Jiufotang 120.0 Ma < Solnhofen 150.0 Ma < Tiaojishan 160.8 Ma (ages in 0.1 Ma). -/
theorem feather_series_age_order :
    1200 < 1500 ∧ 1500 < 1608 := by
  decide

end MathOfBiology.Loop_3120
