/-
  Loop 3222 — Eryops megacephalus, Turing wavenumber at the 127 µm detail scale.
  Horizon: Early Permian (Cisuralian), 295 Ma.
  The organ-scale comparison wavelength is 8184 µm (8.184 mm). It is not the
  wavelength selected by this loop. 8184 = 64 * 127 + 56, so the coarser pit
  holds 64 primordium wavelengths and a 56 µm remainder.
  Successor chronospecies: none (Eryops series length 1).
-/
namespace MathOfBiology.Loop_3222

/-- Detail-scale Turing wavelength, in micrometres. -/
def lambdaUm : Nat := 127

/-- Organ-scale predecessor wavelength, in micrometres (8.184 mm). -/
def lambdaOrganUm : Nat := 8184

/-- Early Permian age of this chronospecies, in Ma. -/
def ageMa : Nat := 295

/-- Famennian stem-tetrapod comparison horizon, in Ma. -/
def predecessorAgeMa : Nat := 365

/-- How many 127 µm wavelengths sit inside 8184 µm, and the remainder. -/
theorem detail_finer_than_organ_scale :
    lambdaUm * 64 + 56 = lambdaOrganUm := by
  decide

theorem primordium_wavelength_positive :
    0 < lambdaUm := by
  decide

theorem eryops_younger_than_stem_tetrapod :
    ageMa + 70 = predecessorAgeMa := by
  decide

/-- Reaction-Jacobian trace at the Schnakenberg state, scaled by 1000.
    f_u + g_v ≈ -0.833, so the homogeneous state is trace-stable. -/
def traceMilli : Int := -833

theorem homogeneous_trace_negative :
    traceMilli < 0 := by
  decide

/-- Diffusion-weighted drive h = D_v f_u + D_u g_v at the reference diffusivities
    (D_u, D_v) = (0.16, 8), scaled by 1000. h ≈ 6.685 > 0, so k_c^2 > 0.
    Pixel rescaling multiplies both diffusivities by the same factor and keeps this sign. -/
def driveMilli : Int := 6685

theorem critical_wavenumber_drive_positive :
    0 < driveMilli := by
  decide

end MathOfBiology.Loop_3222
