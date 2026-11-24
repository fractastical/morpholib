/-
  Loop 3156 — Coscinodiscus wailesii, Turing wavelength at the Holocene organ-mm scale.

  Scale: 4.3 mm = 4316 µm (0.004316 m, log10 = -2.365), tier 6_organ_mm, detail.
  Age: Holocene (extant), 0 Ma. Chronoseries 1/1: no predecessor, no successor.

  Kinetics are Schnakenberg (1979) placed on a spatial domain. Integer parameters
  are the slot values ×10 (f_u = 0.8 → 8, D_u = 1.6 → 16, D_v = 32 → 320) and the
  Jacobian determinant is ×100, so the Turing discriminant inequality is exact.
  Wavelength tags are D_v ×10. The milli-length values are round(1000 × λ_σ) for
  λ_σ = 2π / sqrt(argmax σ₊) on the 4000-point q grid used by the Python script.
-/
namespace MathOfBiology.Loop_3156

def reaction_trace (fu gv : Int) : Int := fu + gv

def turing_drive (Du Dv fu gv : Int) : Int := Dv * fu + Du * gv

def reaction_det (fu fv gu gv : Int) : Int := fu * gv - fv * gu

def turing_discriminant (Du Dv fu gv detJ : Int) : Int :=
  let drive := turing_drive Du Dv fu gv
  drive * drive - 4 * Du * Dv * detJ

/-- The homogeneous mode is linearly stable: f_u + g_v = -0.2 < 0. -/
theorem homogeneous_mode_stable :
    reaction_trace 8 (-10) = -2 ∧ reaction_trace 8 (-10) < 0 := by
  decide

/-- det J = 1 > 0 at the Schnakenberg steady state (entries ×10). -/
theorem reaction_determinant_positive :
    reaction_det 8 10 (-18) (-10) = 100 ∧
    reaction_det 8 10 (-18) (-10) > 0 := by
  decide

/-- Nominal Holocene diffusivities D_u = 1.6, D_v = 32 lie inside the Turing band. -/
theorem nominal_holocene_turing_band :
    turing_drive 16 320 8 (-10) = 2400 ∧
    turing_discriminant 16 320 8 (-10) 100 > 0 := by
  decide

/-- Equal diffusivities are the ablation: the Turing drive is negative and the
    discriminant condition fails, so no finite wavelength is selected. -/
theorem equal_diffusivity_ablation_suppresses_pattern :
    turing_drive 16 16 8 (-10) = -32 ∧
    turing_drive 16 16 8 (-10) < 0 ∧
    turing_discriminant 16 16 8 (-10) 100 < 0 := by
  decide

/-- round(1000 × λ_σ) at D_v = 1.6, 3.2, 6.4. Inhibitor diffusion lengthens the wave. -/
def sigma_wavelength_milli (tag : Nat) : Nat :=
  match tag with
  | 160 => 14086
  | 320 => 16328
  | 640 => 18895
  | _ => 0

theorem inhibitor_diffusion_lengthens_selected_wavelength :
    sigma_wavelength_milli 160 = 14086 ∧
    sigma_wavelength_milli 320 = 16328 ∧
    sigma_wavelength_milli 640 = 18895 ∧
    sigma_wavelength_milli 160 < sigma_wavelength_milli 320 ∧
    sigma_wavelength_milli 320 < sigma_wavelength_milli 640 := by
  decide

/-- Slot scale 4316 µm sits in the organ-millimetre band; the chronoseries is the
    single Holocene point at 0 Ma. -/
def wavelength_micrometres : Nat := 4316
def chronoseries_length : Nat := 1
def age_ma : Nat := 0

theorem holocene_organ_mm_scale :
    wavelength_micrometres = 4316 ∧
    wavelength_micrometres > 1000 ∧
    wavelength_micrometres < 10000 ∧
    age_ma = 0 ∧
    chronoseries_length = 1 := by
  decide

end MathOfBiology.Loop_3156
