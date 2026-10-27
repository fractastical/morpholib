/-
  Loop 3188 — Drosophila heteroneura, detail-scale Turing wavelength.
  Scale: 5.222 µm (5222 nm), tier 3_subcellular_um. Age: Holocene, 0 Ma.
  Chronoseries neighbors at the same detail tier (not a stratigraphic lineage):
  predecessor Drosophila guttifera (8400 nm), successor Drosophila hydei (6350 nm).
  Integer shadows of the Schnakenberg dispersion relation. The sign of h(k_c)
  is checked by the Python self-test; these theorems pin the ordering that
  relation induces. Zero `sorry`.
-/
namespace MathOfBiology.Loop_3188

/-- Critical wavelength of the hypercephalic ocellar pigment mode, in nanometres. -/
def lambdaHetNm : Nat := 5222
/-- Predecessor D. guttifera detail-tier granule wavelength, in nanometres. -/
def lambdaGutNm : Nat := 8400
/-- Successor D. hydei detail-tier thoracic wavelength, in nanometres. -/
def lambdaHydNm : Nat := 6350
/-- Reconstructed male head-capsule width used as the mode-count domain, in nanometres. -/
def headWidthNm : Nat := 1720000

/-- Reaction rate γ in milli-units. At fixed diffusivities, λ ∝ γ^{-1/2}. -/
def gammaHetMilli : Nat := 18600
def gammaHydMilli : Nat := 12579
def gammaGutMilli : Nat := 7188

theorem wavelength_orders_chronospecies :
    lambdaHetNm < lambdaHydNm ∧ lambdaHydNm < lambdaGutNm := by
  decide

theorem reaction_rate_orders_inversely_with_wavelength :
    gammaHetMilli > gammaHydMilli ∧ gammaHydMilli > gammaGutMilli := by
  decide

theorem hypercephalic_mode_count_exceeds_predecessor :
    headWidthNm / lambdaHetNm > headWidthNm / lambdaGutNm := by
  decide

theorem mode_count_gap_at_least_one_hundred :
    headWidthNm / lambdaHetNm > headWidthNm / lambdaGutNm + 100 := by
  decide

end MathOfBiology.Loop_3188
