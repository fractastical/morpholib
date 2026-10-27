/-
  Loop 3102 — Acanthostega gunnari, Turing wavenumber k_c.
  Scale: 105 µm (1.049e-4 m, tier 5_primordium_100um). Age: Famennian, 365 Ma.
  Predecessor: Tiktaalik roseae, Frasnian 375 Ma (fin-ray anisotropy).
  Successor: Ichthyostega stensioei, Famennian 363 Ma (seven digits, slower inhibitor).
  No `sorry`. Integer arithmetic only, so the file is standalone Lean 4.
-/
namespace MathOfBiology.Loop_3102

/-- Law 1. Inhibitor/activator diffusivity ratio r = Dv / Du. This slot uses r = 10. -/
def law1_diffusion_ratio (dv du : Int) : Int := dv / du

theorem law1_inhibitor_faster :
    law1_diffusion_ratio 10 1 = 10 := by
  decide

/-- Law 2. Turing numerator Dv·fu + Du·gv, with fu and gv in thousandths
    (fu = 905/1000, gv = −1103/1000 at the Schnakenberg steady state a = 0.05, b = 1). -/
def law2_turing_numerator (dv du fu gv : Int) : Int := dv * fu + du * gv

theorem law2_instability_numerator_positive :
    law2_turing_numerator 10 1 905 (-1103) > 0 := by
  decide

/-- Law 3. Equal-diffusion ablation (r = 1) makes that numerator negative, so there is no real k_c. -/
theorem law3_equal_diffusion_ablation :
    law2_turing_numerator 1 1 905 (-1103) < 0 := by
  decide

/-- Law 4. One reconstructed Acanthostega digit corridor (4200 µm) holds 40 wavelengths of 105 µm. -/
theorem law4_digit_corridor_mode_count :
    (4200 : Int) / 105 = 40 := by
  decide

/-- Law 5. Tiktaalik fin-ray anisotropy (34 tenths) exceeds isotropic Acanthostega (10 tenths). -/
def law5_anisotropy_excess (alpha_tenth : Int) : Int := alpha_tenth - 10

theorem law5_fin_ray_anisotropy_drops_at_acanthostega :
    law5_anisotropy_excess 34 > law5_anisotropy_excess 10 := by
  decide

end MathOfBiology.Loop_3102
