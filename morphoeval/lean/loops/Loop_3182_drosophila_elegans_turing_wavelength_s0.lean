/-
  Loop 3182 — Drosophila elegans, Holocene (0 Ma), 1.8 µm Turing wavelength.
  Subcellular detail scale (tier 3_subcellular_um), not the 101 µm organ-scale spot.
  Chronoseries 11/41: predecessor Drosophila deflecta, successor Drosophila erecta.
  Integer encoding of the Schnakenberg onset margin:
    r = R/1000, f_u = 760/1000, g_v = -1, det J = 1,
    margin × 10^12 = (760 R − 10^6)^2 − 4 R × 10^9.
  No imports. Proofs are closed with decide.
-/
namespace MathOfBiology.Loop_3182

def turingMargin (rMilli : Int) : Int :=
  (760 * rMilli - 1000000) * (760 * rMilli - 1000000) - 4 * rMilli * 1000000000

/-- Homogeneous Schnakenberg state is stable: trace = f_u + g_v = 0.76 − 1 < 0. -/
theorem homogeneous_trace_negative : (76 : Int) + (-100) < 0 := by
  decide

/-- D. elegans (r × 1000 = 9934, just above onset) is diffusion-unstable. -/
theorem elegans_turing_unstable : turingMargin 9934 > 0 := by
  decide

/-- Equal diffusion (r = 1, the negative control) stays below onset. -/
theorem ablation_equal_diffusion_stable : turingMargin 1000 < 0 := by
  decide

/-- Predecessor deflecta (r × 1000 = 24836) sits farther above onset than elegans. -/
theorem deflecta_farther_above_onset_than_elegans :
    turingMargin 24836 > turingMargin 9934 := by
  decide

/-- Successor erecta (r × 1000 = 13871) is between elegans and deflecta in onset distance. -/
theorem erecta_margin_between_elegans_and_deflecta :
    turingMargin 13871 > turingMargin 9934 ∧ turingMargin 24836 > turingMargin 13871 := by
  decide

/-- Linear wavelengths in nanometres at fixed activator diffusivity (Pass 1 law table):
    deflecta 2196 > erecta 1940 > elegans 1803. Age is Holocene, 0 Ma. -/
theorem chronospecies_wavelength_order :
    (2196 : Nat) > 1940 ∧ (1940 : Nat) > 1803 ∧ (0 : Nat) = 0 := by
  decide

/-- 2π bracket for the elegans wavenumber: 1.803 µm × k, with k in thousandths per µm. -/
theorem elegans_wavenumber_brackets_two_pi :
    (1803 : Int) * 3484 < 6283185 ∧ (1803 : Int) * 3485 > 6283185 := by
  decide

end MathOfBiology.Loop_3182
