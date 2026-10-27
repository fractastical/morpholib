/-
  Loop 3115 — Apis mellifera, Turing wavenumber at the 11.2 µm detail scale, Holocene 0 Ma.
  Namespace MathOfBiology.Loop_3115. No sorry. Integer encodings of the
  Schnakenberg (a, b) = (0.1, 0.9) diffusion threshold, the 12-wavelength
  epidermal window, the comb-scale gap, and the monotone γ sweep.
-/
namespace MathOfBiology.Loop_3115

/-- (4d − 5)² stands in for (0.8 d − 1)², the squared Schnakenberg Turing inequality
    0.8 d − 1 > 2 √d after dividing out γ > 0. Nat subtraction saturates, so the
    d = 1 ablation is the saturated (non-positive) margin. -/
def turingMarginSq (d : Nat) : Nat :=
  (4 * d - 5) * (4 * d - 5)

/-- At the slot ratio d = D_v / D_u = 20 the squared margin clears 100 d. -/
theorem honeybee_detail_scale_is_turing_unstable (d : Nat) (hd : d = 20) :
    turingMarginSq d > 100 * d := by
  subst hd
  decide

/-- Equal diffusivities (d = 1) fail the same strict inequality: the ablation. -/
theorem equal_diffusion_ablation_is_stable :
    turingMarginSq 1 < 100 * 1 := by
  decide

/-- Twelve critical wavelengths fill the 134.4 µm window when λ_c = 11.2 µm
    (both lengths in tenths of a micrometre: 1344 / 112 = 12). -/
theorem twelve_wavelengths_on_epidermal_window :
    1344 / 112 = 12 := by
  decide

/-- Comb-cell wavelength 5200 µm (Camazine sibling, organ scale) over this 11 µm
    detail wavelength. Nat division: 5200 / 11 = 472, so k_c is hundreds of times
    the comb wavenumber. This is the scale gap, not a shared parameter set. -/
theorem detail_wavenumber_gap_versus_comb :
    5200 / 11 = 472 := by
  decide

/-- λ in tenths of a micrometre along the γ sweep. Index 2 is the 11.2 µm slot. -/
def lambdaTenthsUm : Nat → Nat
  | 0 => 144
  | 1 => 129
  | 2 => 112
  | _ => 102

theorem wavelength_decreases_as_gamma_rises :
    lambdaTenthsUm 0 > lambdaTenthsUm 1 ∧
    lambdaTenthsUm 1 > lambdaTenthsUm 2 ∧
    lambdaTenthsUm 2 > lambdaTenthsUm 3 := by
  decide

end MathOfBiology.Loop_3115
