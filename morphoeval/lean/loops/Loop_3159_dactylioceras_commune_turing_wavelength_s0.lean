/-
  Loop 3159 — Dactylioceras commune at the 24 µm detail scale (Toarcian, 181 Ma).

  Catalogue wavelength label: 24 µm. SI calibration used by the script: 2.419e-05 m
  (24.19 µm). The integer theorems below use the catalogue label 24 µm so the
  mode-count comparison is decidable in Lean 4 Init (no mathlib).

  Predecessor: Ceratites nodosus, 240 Ma, organ-scale Turing wavelength 1088 µm.
  This species: measured outer-rib spacing 3484 µm on the 57 mm Whitby specimen.
  Successor: Placenticeras meeki, 75 Ma, organ-scale Turing wavelength 2530 µm.
  Mode count is the number of 24 µm critical waves across that interval.
-/
namespace MathOfBiology.Loop_3159

/-- Denominator of the critical-wavenumber square, 2 D_u D_v, in positive integer units. -/
def kcDen (Du Dv : Int) : Int := 2 * Du * Dv

theorem kc_den_positive (Du Dv : Int) (hDu : 0 < Du) (hDv : 0 < Dv) :
    0 < kcDen Du Dv := by
  dsimp [kcDen]
  exact Int.mul_pos (by decide) (Int.mul_pos hDu hDv)

/-- Diffusion ratio 7.2 = 72/10 lies strictly above the Schnakenberg onset ratio
    6.724 = 6724/1000 for (a, b) = (0.04, 0.96). Cross-multiplication avoids rationals. -/
theorem diffusion_ratio_above_onset :
    (72 : Nat) * 1000 > 6724 * 10 := by
  decide

/-- How many catalogue 24 µm waves fit across an interval measured in micrometres. -/
def modesPerRib (intervalUm lamUm : Nat) : Nat := intervalUm / lamUm

/-- At 24 µm the Whitby rib (3484 µm) holds more critical waves than the
    Ceratites organ-scale wavelength (1088 µm) at 240 Ma. -/
theorem dactylioceras_exceeds_ceratites_at_24um :
    modesPerRib 3484 24 > modesPerRib 1088 24 := by
  decide

/-- At 24 µm the Whitby rib also holds more critical waves than the
    Placenticeras organ-scale wavelength (2530 µm) at 75 Ma. -/
theorem dactylioceras_exceeds_placenticeras_at_24um :
    modesPerRib 3484 24 > modesPerRib 2530 24 := by
  decide

/-- Assigned suture dimension D_f = 1.55 is strictly above a smooth curve (D_f = 1). -/
theorem fractal_suture_above_smooth_chord :
    (155 : Nat) > 100 := by
  decide

end MathOfBiology.Loop_3159
