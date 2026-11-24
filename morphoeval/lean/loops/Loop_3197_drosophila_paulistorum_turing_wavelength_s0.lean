/-
  Loop 3197 — Drosophila paulistorum, detail rung s0.
  Scale: 2.417 µm (2.417e-6 m, log10 = -5.617), tier 3_subcellular_um.
  Age: Holocene extant, 0 Ma.
  Predecessor catalog chronospecies: Drosophila orena, cellular wavelength 89440 nm.
  Successor catalog chronospecies: Drosophila persimilis, primordium wavelength 116700 nm.
  Schnakenberg steady state at a = 0.10, b = 0.90:
    f_u = 0.800, f_v = 1.000, g_u = -1.800, g_v = -1.000 (thousandths below).
  No imports. No unfinished proofs.
-/
namespace MathOfBiology.Loop_3197

/-- Trace of the reaction Jacobian in thousandths. tr = f_u + g_v = -0.200. -/
def schnakenbergTraceMilli : Int := (800 : Int) + (-1000)

/-- Homogeneous steady state is not a Hopf point: trace is strictly negative. -/
theorem homogeneous_steady_state_is_stable :
    schnakenbergTraceMilli < 0 := by
  unfold schnakenbergTraceMilli
  decide

/-- det(J) in millionths. det = f_u g_v - f_v g_u = 1.000. -/
def schnakenbergDetMillion : Int :=
  (800 : Int) * (-1000) - (1000 : Int) * (-1800)

/-- The reaction Jacobian at equilibrium has positive determinant. -/
theorem reaction_jacobian_determinant_positive :
    schnakenbergDetMillion > 0 := by
  unfold schnakenbergDetMillion
  decide

/-- Numerator d·f_u + g_v at diffusivity ratio d = 36, in thousandths.
    Positivity is the finite-k_c condition for this rung. -/
def diffusionNumeratorMilli : Int := (36 : Int) * 800 + (-1000)

theorem critical_wavenumber_numerator_positive :
    diffusionNumeratorMilli > 0 := by
  unfold diffusionNumeratorMilli
  decide

/-- 2.417 µm packs 37 times into the Drosophila orena cellular wavelength
    (89.440 µm), with remainder 11 nm. This rung is strictly shorter. -/
theorem detail_scale_shorter_than_orena_cellular :
    (2417 : Int) * 37 < 89440 ∧ 89440 - (2417 : Int) * 37 = 11 := by
  decide

/-- 2.417 µm packs 48 times into the Drosophila persimilis primordium
    wavelength (116.700 µm), with remainder 684 nm. -/
theorem detail_scale_shorter_than_persimilis_primordium :
    (2417 : Int) * 48 < 116700 ∧ 116700 - (2417 : Int) * 48 = 684 := by
  decide

/-- Six critical wavelengths tile the 14.502 µm ocellar cortex patch exactly. -/
theorem six_wavelengths_tile_ocellar_patch :
    (2417 : Int) * 6 = 14502 := by
  decide

end MathOfBiology.Loop_3197
