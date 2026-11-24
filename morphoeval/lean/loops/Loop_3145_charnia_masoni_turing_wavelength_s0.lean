/-
  Loop 3145 — Charnia masoni, apical Turing wavelength at 140.4 µm, 565 Ma.
  Integer theorems only (core Lean 4, no mathlib, no sorry).
  Scales are exact tenths: 140.4 µm = 1404, organ-sibling 6.322 mm = 63220,
  both in units of 0.1 µm.  Kinetics integers match models/2026-delphi10000-
  charnia-masoni-turing-wavelength-s0.py: f_u = 7383/10000, g_v = -11449/10000,
  diffusion ratio r = 1350/62.  The Turing numerator below is
  10000 * (r * f_u + g_v) cleared over the positive denominator 62.
-/
namespace MathOfBiology.Loop_3145

/-- r * f_u + g_v, cleared over denominator 62 * 10000.
    Positive value is the diffusion-driven Turing condition at this scale. -/
def turingNumerator (rNum fuNum gvDen : Int) : Int :=
  rNum * fuNum + gvDen

theorem apical_turing_condition_140um :
    turingNumerator 1350 7383 (-11449 * 62) > 0 := by
  decide

/-- The detail wavelength is strictly finer than the 6.322 mm organ-scale sibling. -/
theorem detail_wavelength_finer_than_organ_sibling :
    (63220 : Int) > 1404 := by
  decide

/-- Inherited glide reflection is a half-period offset: 2 * 702 = 1404. -/
def glideClosure (delta wavelength : Int) : Int :=
  2 * delta - wavelength

theorem glide_offset_is_half_period :
    glideClosure 702 1404 = 0 := by
  decide

/-- Wavenumber ratio k_detail/k_organ = 63220/1404 lies strictly between 45 and 46.
    That ratio is what changes at 140.4 µm relative to the organ-scale sibling. -/
theorem wavenumber_ratio_between_45_and_46 :
    ((45 * 1404 < 63220) ∧ (63220 < 46 * 1404)) := by
  decide

/-- Late Ediacaran 565.0 Ma is before the Ediacaran–Cambrian boundary near 538.8 Ma.
    This chronoseries has no Cambrian successor chronospecies. -/
theorem ediacaran_565_before_cambrian_boundary :
    (5650 : Int) > 5388 := by
  decide

end MathOfBiology.Loop_3145
