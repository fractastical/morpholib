/-
  Loop 3232 — Gasterosteus aculeatus_marine at the 52 µm cellular Turing wavelength.
  Age 0.015 Ma (Late Pleistocene marine ancestor, chronoseries node 4/6).
  Predecessor G. doryssus_late is assigned 104 µm; successor G. aculeatus is assigned 78 µm.
  Integers are the script's Schnakenberg coefficients scaled by 1000
  (D_u = 0.500, D_v = 20.000, f_u = 0.800, g_v = -1.000).
-/
namespace MathOfBiology.Loop_3232

/-- Trace-diffusion coefficient h = D_v f_u + D_u g_v. Positive h is required for a finite Turing band. -/
def traceDiffusion (Du Dv fu gv : Int) : Int := Dv * fu + Du * gv

theorem marine_trace_diffusion_value :
    traceDiffusion 500 20000 800 (-1000) = 15500000 := by
  decide

theorem marine_wavenumber_coefficient_positive :
    0 < traceDiffusion 500 20000 800 (-1000) := by
  decide

/-- k_c^2 = h / (2 D_u D_v) = 0.775, checked as h * 1000 = 775 * (2 D_u D_v). -/
theorem marine_kc_squared_is_775_thousandths :
    traceDiffusion 500 20000 800 (-1000) * 1000 = 775 * (2 * 500 * 20000) := by
  decide

/-- Equal diffusivities fall outside the Turing band (h < 0), so the ablation has no critical wavenumber. -/
theorem equal_diffusion_ablation_has_no_turing_band :
    traceDiffusion 500 500 800 (-1000) < 0 := by
  decide

/-- At 0.015 Ma the marine complete morph is shorter-wavelength than both neighbors (µm). -/
theorem marine_shorter_than_predecessor_and_successor :
    (52 : Int) < 104 ∧ (52 : Int) < 78 := by
  decide

/-- Seventeen 52 µm wavelengths fit in an 887 µm gap between plates of a 32-plate flank. -/
theorem interplate_mode_count : (887 : Nat) / 52 = 17 := by
  decide

end MathOfBiology.Loop_3232
