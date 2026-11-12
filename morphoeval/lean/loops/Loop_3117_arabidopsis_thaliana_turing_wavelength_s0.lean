/-
  Loop 3117 — Arabidopsis thaliana, Holocene extant (0 Ma), Turing wavenumber at 111.6 µm.
  Chronoseries 1/1: there is no fossil predecessor. The comparisons below are the
  scale ladder at the same age. Integer lengths are micrometres × 10
  (1116 = 111.6 µm, 4464 = 446.4 µm primordium patch, 186 = 18.6 µm cell,
  17856 = 1.7856 mm young leaf).
  Diffusivity witness uses D_v/D_u = 84/10 = 8.4, f_u = 9, g_v = −11, the sign
  pattern of the Schnakenberg Jacobian at (a, b) = (0.05, 1): f_u > 0, g_v < 0,
  and f_u + g_v < 0 so equal diffusion cannot satisfy the Turing numerator.
-/
namespace MathOfBiology.Loop_3117

def kcNumerator (Du Dv fu gv : Int) : Int := Dv * fu + Du * gv

def modeCount (domain_um10 wavelength_um10 : Int) : Int :=
  domain_um10 / wavelength_um10

/-- Law k_c² numerator D_v f_u + D_u g_v is positive at the 8.4 diffusivity ratio. -/
theorem arabidopsis_112um_turing_numerator_positive :
    kcNumerator 10 84 9 (-11) = 646 := by
  decide

/-- Equal diffusion on the same Jacobian makes the numerator negative, so no k_c. -/
theorem arabidopsis_112um_equal_diffusion_numerator_negative :
    kcNumerator 10 10 9 (-11) = -20 := by
  decide

/-- This primordium-detail patch (446.4 µm) holds four critical wavelengths. -/
theorem arabidopsis_primordium_patch_holds_four_wavelengths :
    modeCount 4464 1116 = 4 := by
  decide

/-- The 18.6 µm pavement-cell predecessor holds no full wavelength. -/
theorem arabidopsis_cell_predecessor_holds_no_wavelength :
    modeCount 186 1116 = 0 := by
  decide

/-- The 1.7856 mm young-leaf successor holds sixteen wavelengths at the same λ_c and 0 Ma. -/
theorem arabidopsis_leaf_successor_holds_sixteen_wavelengths :
    modeCount 17856 1116 = 16 := by
  decide

end MathOfBiology.Loop_3117
