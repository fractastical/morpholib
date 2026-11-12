/-
  Loop 3193 — Drosophila miranda, detail-scale Turing wavelength.
  Scale: 1.9 µm (1.864e-6 m, log10 = -5.730, tier 3_subcellular_um).
  Age: Holocene (Extant), 0 Ma. Lineage: Sophophora obscura & willistoni clade.
  Chronoseries 22/41. Predecessor: Drosophila melanogaster (λ = 1.412 µm).
  Successor: Drosophila mojavensis (λ = 2.286 µm).
  Law: Schnakenberg Turing wavenumber k_c on a 22.368 µm ocellar pigment field.
  No `sorry`.
-/
namespace MathOfBiology.Loop_3193

/-- Jacobian trace at the Schnakenberg steady state, in hundredths.
    f_u = 0.84, g_v = −1, so tr J = −0.16 < 0 (stable without diffusion). -/
def trace_hundredths : Int := 84 + (-100)

theorem trace_negative : trace_hundredths < 0 := by
  decide

/-- det J = f_u g_v − f_v g_u in thousandths.
    840·(−1000) − 1000·(−1840) = 1_000_000 > 0. -/
def determinant_thousandths : Int := 840 * (-1000) - 1000 * (-1840)

theorem determinant_positive : determinant_thousandths > 0 := by
  decide

/-- Equal diffusion cannot Turing-destabilize a trace-negative steady state.
    With D_u = D_v the diffusion weight has the sign of tr J. -/
theorem equal_diffusion_weighted_trace_negative :
    28630 * (84 + (-100)) < 0 := by
  decide

/-- Inhibitor diffuses faster than the activator, and the fixed-point Turing
    margin (D_v f_u + D_u g_v)² > 4 D_u D_v det is positive.
    Diffusivities are in 10⁻⁶ µm² per time; kinetic entries are in hundredths. -/
theorem inhibitor_faster_and_turing_margin :
    274882 > 28630 ∧
    (20227088 : Int) * 20227088 > 4 * 28630 * 274882 * 10000 := by
  decide

/-- Detail-tier chronoseries at 0 Ma, wavelengths in nanometres:
    D. melanogaster 1412 < D. miranda 1864 < D. mojavensis 2286. -/
theorem chronoseries_wavelength_order :
    1412 < 1864 ∧ 1864 < 2286 := by
  decide

/-- The same chronoseries in wavenumber (milliradians per µm) and in the
    number of wavelengths that fit in the 22.368 µm miranda field:
    k_mel > k_mir > k_moj and N = 16 > 12 > 10. -/
theorem chronoseries_wavenumber_and_mode_count :
    4450 > 3371 ∧ 3371 > 2749 ∧ 16 > 12 ∧ 12 > 10 := by
  decide

end MathOfBiology.Loop_3193
