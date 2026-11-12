/-
  Loop 3200 — Drosophila pseudoobscura, detail-scale Turing wavelength.
  Scale: 2.0 µm = 1961 nm (1.961e-6 m, log10 = -5.708), tier 3_subcellular_um.
  Age: Holocene (Extant), 0 Ma. Lineage: Sophophora obscura & willistoni clade.
  Chronoseries 29/41. Predecessor: Drosophila planitibia, 197200 nm primordium rung.
  Successor: Drosophila santomea, 102000 nm primordium rung.
  Age does not change (all three nodes are 0 Ma). The wavelength rung does.

  Jacobian fixed point used by the script (rates × 100):
    f_u = 84, f_v = -90, g_u = 155, g_v = -120
    corresponding to 0.84, -0.90, 1.55, -1.20 per second.
  Diffusivities × 10^6 µm²/s: D_u = 28046, D_v = 131825.
-/

namespace MathOfBiology.Loop_3200

def fu : Int := 84
def fv : Int := -90
def gu : Int := 155
def gv : Int := -120

def traceJ : Int := fu + gv
def detJ : Int := fu * gv - fv * gu

/-- The homogeneous steady state is asymptotically stable without diffusion. -/
theorem homogeneous_steady_state_stable : traceJ < 0 ∧ detJ > 0 := by
  decide

/-- The 2.0 µm detail rung is finer than the successor primordium rung,
    which is itself finer than the predecessor picture-wing rung. -/
theorem detail_rung_finer_than_chronospecies :
    1961 < 102000 ∧ 102000 < 197200 := by
  decide

/-- On a 24.000 µm epithelial patch the 1961 nm mode has stripe count 12. -/
theorem mode_count_bracket_on_24um_patch :
    12 * 1961 < 24000 ∧ 24000 < 13 * 1961 := by
  decide

/-- The inhibitor diffuses at least 4.70 times faster than the activator. -/
theorem inhibitor_diffuses_faster :
    131825 * 100 > 470 * 28046 ∧ 131825 > 28046 := by
  decide

/-- Equal diffusivities cannot open a Turing band: D·trace(J) is negative,
    so the numerator D_v f_u + D_u g_v collapses to D·trace(J) < 0. -/
theorem equal_diffusion_numerator_negative :
    (28046 : Int) * traceJ < 0 := by
  decide

end MathOfBiology.Loop_3200
