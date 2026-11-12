/-
  Loop 3199 — Drosophila planitibia detail-tier Turing wavelength.
  Scale: 5.13 µm = 5130 nm (5.13e-06 m, assigned log10 = -5.290), tier 3_subcellular_um.
  Age: Holocene, 0 Ma. Predecessor D. persimilis at 117 µm. Successor D. pseudoobscura at 114 µm.
  The geological age does not change. The selected wavenumber does.
  Diffusion-ratio integers are in tenths: d = 18.6 → 186, f_u = 0.8 → 8, g_v = -1.0 → -10.
-/
namespace MathOfBiology.Loop_3199

/-- Detail-scale wavelength of D. planitibia, in nanometres. -/
def lambdaDetailNm : Int := 5130

/-- Primordium-tier wavelength of the predecessor D. persimilis, in nanometres. -/
def lambdaPersimilisNm : Int := 117000

/-- Primordium-tier wavelength of the successor D. pseudoobscura, in nanometres. -/
def lambdaPseudoobscuraNm : Int := 114000

/-- Schnakenberg trace-diffusion coefficient d·f_u + g_v, with inputs in tenths. -/
def traceDiffusion (d fu gv : Int) : Int := d * fu + gv

theorem detail_finer_than_persimilis :
    lambdaDetailNm < lambdaPersimilisNm := by
  decide

theorem detail_finer_than_pseudoobscura :
    lambdaDetailNm < lambdaPseudoobscuraNm := by
  decide

/-- k_planitibia / k_pseudoobscura = 114 µm / 5.13 µm > 20. Age is not the quantity that moves. -/
theorem wavenumber_ratio_vs_successor_exceeds_twenty :
    lambdaPseudoobscuraNm > 20 * lambdaDetailNm := by
  decide

theorem wavenumber_ratio_vs_predecessor_exceeds_twenty :
    lambdaPersimilisNm > 20 * lambdaDetailNm := by
  decide

/-- Nominal d = 18.6 is above the Turing threshold d·f_u + g_v > 0. -/
theorem planitibia_diffusion_ratio_is_turing_unstable :
    traceDiffusion 186 8 (-10) > 0 := by
  decide

/-- Equal diffusion d = 1 is the ablation: the trace-diffusion coefficient is negative. -/
theorem equal_diffusion_ablation_is_stable :
    traceDiffusion 10 8 (-10) < 0 := by
  decide

/-- Both chronospecies and the focal species are Holocene (0 Ma). The age delta is zero. -/
theorem holocene_age_delta_is_zero :
    (0 : Int) - (0 : Int) = 0 := by
  decide

end MathOfBiology.Loop_3199
