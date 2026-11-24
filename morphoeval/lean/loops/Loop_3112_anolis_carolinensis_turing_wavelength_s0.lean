/-
  Loop 3112 — Anolis carolinensis, Turing wavelength at 101.1 µm, Holocene 0 Ma.
  Schnakenberg steady state (a, b) = (55/1000, 945/1000), so u0 = 1 and v0 = 945/1000.
  f_u = 890/1000, g_v = -1, det J = 1, trace J = -110/1000.
  Diffusivity ratio D_v/D_u = 82/10, just above the Turing onset near 7.1.
  No fossil chronospecies: the only comparison is the shorter cellular domain, which
  cannot fit a 101 µm critical mode.
-/
namespace MathOfBiology.Loop_3112

/-- Trace(J) × 1000 at the Schnakenberg steady state. -/
def traceMilli : Int := 890 - 1000

/-- Determinant of J × 1000. The unscaled determinant equals 1. -/
def detMilli : Int := 1000

/-- Inhibitor/activator diffusivity ratio in tenths (8.2). -/
def ratioTenths : Int := 82

/-- Assigned critical wavelength in tenths of a micrometre (101.1 µm). -/
def lambdaTenthsUm : Int := 1011

/-- How many critical wavelengths fit in an integer-micrometre domain. -/
def modeCount (domainUm wavelengthUm : Int) : Int := domainUm / wavelengthUm

theorem homogeneous_stable_without_diffusion : traceMilli < 0 := by
  decide

theorem homogeneous_determinant_positive : detMilli > 0 := by
  decide

/-- Onset for these kinetics is near ratio 7.12; 8.2 is above it and above 1. -/
theorem inhibitor_diffuses_faster_than_onset : ratioTenths > 71 := by
  decide

/-- (r f_u + g_v)^2 > 4 r det, cleared to integers.
    r f_u + g_v = 62980/10000 at r = 8.2 and f_u = 0.89. -/
theorem diffusion_driven_instability :
    (82 * 890 + 10 * (-1000)) * (82 * 890 + 10 * (-1000)) * 10
      - 4 * 82 * 10000 * 10000 > 0 := by
  decide

theorem scale_sits_in_primordium_tier :
    1000 ≤ lambdaTenthsUm ∧ lambdaTenthsUm < 10000 := by
  decide

/-- 808 µm dermal patch (eight nominal 101 µm modes). -/
theorem primordium_patch_fits_eight_wavelengths :
    modeCount 808 101 = 8 := by
  decide

/-- Cellular predecessor, 40 µm: the critical mode does not fit. -/
theorem cellular_predecessor_fits_no_critical_mode :
    modeCount 40 101 = 0 := by
  decide

/-- Holocene age of this chronoseries point, in integer Ma. -/
def ageMa : Int := 0

theorem holocene_extant_endpoint : ageMa = 0 ∧ ageMa ≤ lambdaTenthsUm := by
  decide

end MathOfBiology.Loop_3112
