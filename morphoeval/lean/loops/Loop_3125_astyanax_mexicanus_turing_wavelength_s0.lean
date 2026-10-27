/-
  Loop 3125 — Astyanax mexicanus, Turing wavenumber at the 666 nm detail scale.
  Phenomenon size: 6.656e-7 m (Holocene cave morph, 0 Ma).
  Sibling cellular wavelength used only as a comparison: 2.418e-5 m.
  Integer Jacobian is 5× the Schnakenberg steady-state derivatives
  (fu, fv, gu, gv) = (0.8, 1, -1.8, -1), with Du:Dv = 2:20.
-/
namespace MathOfBiology.Loop_3125

def fu : Int := 4
def fv : Int := 5
def gu : Int := -9
def gv : Int := -5
def Du : Int := 2
def Dv : Int := 20

def trace : Int := fu + gv
def detJ : Int := fu * gv - fv * gu
def diffusionDrive : Int := Dv * fu + Du * gv

/-- Without diffusion the steady state is linearly stable: trace(J) < 0. -/
theorem trace_stable_without_diffusion : trace < 0 := by
  decide

/-- The reaction Jacobian has positive determinant, so k = 0 is not a saddle. -/
theorem determinant_positive : detJ > 0 := by
  decide

/-- Inhibitor diffusion is fast enough that Dv fu + Du gv > 0. -/
theorem diffusion_drive_positive : diffusionDrive > 0 := by
  decide

/-- Turing discriminant: (Dv fu + Du gv)^2 > 4 Du Dv det(J). -/
theorem turing_discriminant :
    diffusionDrive * diffusionDrive > 4 * Du * Dv * detJ := by
  decide

/-- Equal diffusivities violate the drive condition, so the mode is not Turing. -/
theorem equal_diffusivity_not_turing :
    (Du * fu + Du * gv) < 0 := by
  decide

/-- Card wavelength 6.656e-7 m = 6656 × 10^-10 m.
    floor(2π × 10^7) = 62831853 brackets k_c = 2π / λ. -/
def lambda_tenth_nm : Nat := 6656
def two_pi_times_1e7 : Nat := 62831853

theorem wavenumber_brackets_two_pi :
    lambda_tenth_nm * 9439 < two_pi_times_1e7 ∧
    two_pi_times_1e7 < lambda_tenth_nm * 9440 := by
  decide

/-- The 24 µm cellular sibling (24180 nm) is 36–37 times this wavelength. -/
def lambda_cell_nm : Nat := 24180
def lambda_nm : Nat := 666

theorem cellular_sibling_is_longer_wavelength :
    lambda_nm * 36 < lambda_cell_nm ∧ lambda_cell_nm < lambda_nm * 37 := by
  decide

end MathOfBiology.Loop_3125
