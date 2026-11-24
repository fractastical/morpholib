/-
  Loop 3113 — Anomalocaris canadensis, Turing wavenumber at the 24 µm detail scale.
  Exact slot length: 2.397e-05 m = 23970 nm (23.97 µm), logged as the 24 µm tier.
  Horizon: Burgess Shale, 508 Ma. Predecessor eye horizon: Emu Bay Shale, 515 Ma.
  Reconstructed Emu Bay facet spacing at a 2.5 cm hemisphere with 16000 lenses: 266 µm.
  Kinetics at the homogeneous state (a, b) = (1/10, 9/10): trace(J) < 0.
  Diffusion-ratio gate for these partial derivatives: Dv/Du > 25/4.
-/
namespace MathOfBiology.Loop_3113

/-- Exact slot length in nanometres (2.397e-05 m). -/
def lambda_nm : Nat := 23970

/-- Tier label in micrometres. -/
def lambda_label_um : Nat := 24

/-- Rounded reconstructed facet spacing (µm) for a 2.5 cm hemispherical eye with 16000 lenses. -/
def emu_bay_facet_spacing_um : Nat := 266

/-- This chronospecies node, Ma. -/
def burgess_age_ma : Nat := 508

/-- Predecessor eye horizon (Emu Bay Shale), Ma. -/
def emu_bay_age_ma : Nat := 515

/-- Young end of the Fezouata successor interval quoted by Van Roy & Briggs (2011), Ma. -/
def fezouata_young_ma : Nat := 472

/-- 5 * trace(J) at (u0, v0) = (1, 9/10): 5*f_u + 5*g_v = 4 + (-5). -/
def trace_scaled : Int := 4 + (-5)

theorem label_bounds_the_exact_length :
    lambda_nm < lambda_label_um * 1000 ∧ lambda_label_um * 1000 - lambda_nm < 1000 := by
  decide

theorem detail_wavelength_is_subfacet :
    lambda_nm / 1000 < emu_bay_facet_spacing_um := by
  decide

theorem burgess_node_is_younger_than_emu_bay :
    burgess_age_ma < emu_bay_age_ma := by
  decide

theorem fezouata_successor_is_younger_than_burgess :
    fezouata_young_ma < burgess_age_ma := by
  decide

theorem homogeneous_state_is_stable :
    trace_scaled < 0 := by
  decide

/-- Milli-unit diffusivities used by the script (Du ≈ 0.608, Dv ≈ 12.159). -/
theorem inhibitor_diffuses_fast_enough :
    (4 : Int) * 12159 > 25 * 608 := by
  decide

theorem equal_diffusion_fails_ratio_gate :
    ¬ ((4 : Int) * 608 > 25 * 608) := by
  decide

end MathOfBiology.Loop_3113
