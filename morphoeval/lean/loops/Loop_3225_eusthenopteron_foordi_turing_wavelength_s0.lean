/-
  Loop 3225 — Eusthenopteron foordi, Turing wavenumber at the 104 µm detail scale.
  Age: Late Devonian (Frasnian), 379 Ma. Chronoseries 1/1.
  Namespace MathOfBiology.Loop_3225. No sorry.
  Integer statements are the exact arithmetic shadows of the quantities used in
  models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py:
    λ = 104 µm, Schnakenberg margin at Dv/Du = 12, registry body lengths.
-/
namespace MathOfBiology.Loop_3225

/-- Diffusion-weighted trace margin Dv·f_u + Du·g_v.
    Coefficients are the reference Jacobian scaled by 10:
    f_u = 0.8 → 8, g_v = −1 → −10, D_u = 1 → 10, D_v = 12 → 120. -/
def turingMargin (dv fu du gv : Int) : Int := dv * fu + du * gv

theorem detail_scale_diffusion_margin :
    turingMargin 120 8 10 (-10) > 0 := by
  decide

/-- Number of 104 µm wavelengths in a length given in micrometres. -/
def wavelengths (lengthUm wavelengthUm : Nat) : Nat := lengthUm / wavelengthUm

/-- Registry characteristic length 0.140 m = 140000 µm holds 1346 detail waves. -/
theorem eusthenopteron_body_holds_detail_waves :
    wavelengths 140000 104 = 1346 := by
  decide

/-- Morphological successor Panderichthys rhombolepis, registry length 0.165 m,
    packs strictly more 104 µm waves than Eusthenopteron at fixed k_c.
    Eusthenopteron itself is chronoseries 1/1 (no second chronospecies). -/
theorem successor_packs_more_detail_wavelengths :
    wavelengths 165000 104 > wavelengths 140000 104 := by
  decide

/-- The 5.7 mm organ-tier flank bar is 54 repeats of this 104 µm fundamental,
    so the two slots are different wavenumbers, not a copied parameter set. -/
theorem organ_tier_counts_many_detail_waves :
    wavelengths 5700 104 = 54 := by
  decide

end MathOfBiology.Loop_3225
