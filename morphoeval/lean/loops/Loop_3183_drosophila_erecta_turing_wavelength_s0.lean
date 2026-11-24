/-
  Loop 3183 — Drosophila erecta, Turing wavenumber at the 2.262 µm detail tier.
  Age 0 Ma (Holocene). Predecessor Drosophila elegans (101.1 µm primordium).
  Successor Drosophila eugracilis (100.6 µm primordium).
  Integer wavelengths are nanometres. fuMilli / gvAbsMilli are 1000× roundings of
  the Schnakenberg Jacobian entries f_u = 0.691 and |g_v| = 1.210 at a = 0.17, b = 0.93.
  Proofs are closed by `decide`.
-/
namespace MathOfBiology.Loop_3183

def lambdaDetailNm : Nat := 2262
def lambdaElegansPrimordiumNm : Nat := 101100
def lambdaEugracilisPrimordiumNm : Nat := 100600
def diffusionRatio : Nat := 16
def fuMilli : Nat := 691
def gvAbsMilli : Nat := 1210

/-- The detail-tier wavelength is shorter than the elegans primordium anchor. -/
theorem detail_wavelength_shorter_than_elegans_primordium :
    lambdaDetailNm < lambdaElegansPrimordiumNm := by
  decide

/-- The detail-tier wavelength is shorter than the eugracilis primordium anchor. -/
theorem detail_wavelength_shorter_than_eugracilis_primordium :
    lambdaDetailNm < lambdaEugracilisPrimordiumNm := by
  decide

/-- Eight detail wavelengths span the 18.096 µm epidermal patch used in the mode count. -/
theorem eight_detail_wavelengths_span_the_patch :
    8 * lambdaDetailNm = 18096 := by
  decide

/-- δ f_u + g_v > 0 at δ = 16, so a positive critical wavenumber exists. -/
theorem activator_inhibitor_ratio_above_turing_threshold :
    diffusionRatio * fuMilli > gvAbsMilli := by
  decide

/-- Equal diffusion (δ = 1) falls below the same threshold, so k_c is not real. -/
theorem equal_diffusion_below_turing_threshold :
    1 * fuMilli < gvAbsMilli := by
  decide

/-- Relative to elegans, the detail wavenumber jumps by a factor strictly between 44 and 45. -/
theorem wavenumber_jumps_about_fortyfive_fold :
    44 * lambdaDetailNm < lambdaElegansPrimordiumNm ∧
      lambdaElegansPrimordiumNm < 45 * lambdaDetailNm := by
  decide

end MathOfBiology.Loop_3183
