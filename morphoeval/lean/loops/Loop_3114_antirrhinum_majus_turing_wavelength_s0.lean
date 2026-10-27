/-
  Loop 3114 — Antirrhinum majus, Turing wavenumber at 259 µm, Holocene (0 Ma).
  Jacobian entries are scaled by 1000 (f_u = 1250 means 1.25 h⁻¹).
  Diffusivities are the integers used in the script: D_u = 640, D_v = 4341 µm² h⁻¹.
  No `sorry`. Core Lean only (`decide` on Int).
-/
namespace MathOfBiology.Loop_3114

/-- Reaction trace tr J = f_u + g_v. -/
def trace (fu gv : Int) : Int := fu + gv

/-- det J = f_u g_v − f_v g_u. With entries ×1000 this value is det × 10⁶. -/
def determinant (fu fv gu gv : Int) : Int := fu * gv - fv * gu

/-- h = D_v f_u + D_u g_v, rates still ×1000. -/
def diffusionTrace (du dv fu gv : Int) : Int := dv * fu + du * gv

/-- Turing margin h² − 4 D_u D_v det. Positive means a band of unstable k. -/
def turingDiscriminant (du dv fu fv gu gv : Int) : Int :=
  let h := diffusionTrace du dv fu gv
  let det := determinant fu fv gu gv
  h * h - 4 * du * dv * det

theorem trace_value : trace 1250 (-1750) = -500 := by decide

theorem trace_negative : trace 1250 (-1750) < 0 := by decide

theorem determinant_value : determinant 1250 (-1500) 2000 (-1750) = 812500 := by decide

theorem determinant_positive : determinant 1250 (-1500) 2000 (-1750) > 0 := by decide

theorem turing_margin_value :
    turingDiscriminant 640 4341 1250 (-1500) 2000 (-1750) = 9514509062500 := by decide

theorem turing_band_open :
    turingDiscriminant 640 4341 1250 (-1500) 2000 (-1750) > 0 := by decide

theorem equal_diffusion_trace_value :
    diffusionTrace 640 640 1250 (-1750) = -320000 := by decide

theorem equal_diffusion_closes_band :
    diffusionTrace 640 640 1250 (-1750) < 0 := by decide

/-- Domain of one assigned wavelength (2592 tenths of a micrometre) holds one mode. -/
theorem one_wavelength_at_this_scale : (2592 : Int) / 2592 = 1 := by decide

/-- A 40 µm conical cell (400 tenths) is shorter than λ_c. -/
theorem cell_predecessor_subwavelength : (400 : Int) < 2592 := by decide

/-- 10 mm / 259 µm = 38 critical wavelengths on the petal-lobe successor. -/
theorem petal_successor_mode_count : (10000 : Int) / 259 = 38 := by decide

end MathOfBiology.Loop_3114
