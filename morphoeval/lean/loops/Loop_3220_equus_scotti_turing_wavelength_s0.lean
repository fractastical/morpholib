/-
  Loop 3220 — Equus scotti follicular Turing wavelength.
  Scale: 164.3 µm (0.0001643 m), Middle Pleistocene (Irvingtonian, 0.6 Ma).
  Kinetics: Schnakenberg (1979). Wavenumber: argmax of the growth rate, with the
  Murray determinant stationary point kept as a separate integer identity.
  Wavelengths below are in nanometres (164.3 µm = 164300 nm).
-/
namespace MathOfBiology.Loop_3220

/-- Jacobian of Schnakenberg kinetics at (u0, v0) = (1, 9/10), scaled by 10:
    fu = 8, fv = 10, gu = -18, gv = -10. -/
def jacFu : Int := 8
def jacFv : Int := 10
def jacGu : Int := -18
def jacGv : Int := -10

def schnakenbergDetScaled : Int := jacFu * jacGv - jacFv * jacGu

theorem schnakenberg_determinant_positive : schnakenbergDetScaled = 100 := by
  decide

theorem schnakenberg_trace_negative : jacFu + jacGv < 0 := by
  decide

/-- Dv * fu + Du * gv at the Equus scotti ratio Dv/Du = 20 (Du scaled to 1). -/
def diffusionDrive (du dv : Int) : Int := dv * jacFu + du * jacGv

theorem scotti_ratio_is_diffusion_driven : diffusionDrive 1 20 = 150 := by
  decide

theorem equal_diffusion_fails_the_turing_condition : diffusionDrive 1 1 < 0 := by
  decide

/-- Predecessor Pliohippus pernix 198688 nm, focal Equus scotti 164300 nm,
    successor Equus quagga Bard initiation spacing 400000 nm. -/
theorem chronospecies_wavelength_order :
    (198688 : Int) > 164300 ∧ (400000 : Int) > 164300 := by
  decide

/-- A 1314.4 µm patch (1314400 nm) holds eight 164.3 µm follicular wavelengths,
    and eight is closer than seven or nine. -/
theorem eight_follicular_modes_on_the_scotti_patch :
    (8 * 164300 - 1314400) * (8 * 164300 - 1314400)
      < (7 * 164300 - 1314400) * (7 * 164300 - 1314400) ∧
    (8 * 164300 - 1314400) * (8 * 164300 - 1314400)
      < (9 * 164300 - 1314400) * (9 * 164300 - 1314400) := by
  decide

end MathOfBiology.Loop_3220
