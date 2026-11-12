/-
  Loop 3198 — Drosophila persimilis, Turing wavenumber at the 2.1 µm detail tier.
  Scale: 2.098e-6 m (log10 = -5.678), tier 3_subcellular_um. Age: Holocene, 0 Ma.
  Predecessor Drosophila paulistorum (1947 nm). Focal species 2098 nm.
  Successor Drosophila planitibia (2735 nm). Diffusion ratio r = 20, onset r = 9.
  Namespace MathOfBiology.Loop_3198. No sorry.
-/
namespace MathOfBiology.Loop_3198

/-- Schnakenberg diffusion ratio used for D. persimilis at 2.098 µm. -/
def diffusionRatioPersimilis : Nat := 20

/-- Smallest integer ratio at which (0.8 r - 1)^2 > 4 r for these kinetics. -/
def diffusionRatioOnset : Nat := 9

/-- Equal diffusion, the negative control, sits strictly below onset. -/
theorem equalDiffusionIsNotTuring :
    1 < diffusionRatioOnset := by
  decide

theorem persimilisAboveTuringOnset :
    diffusionRatioOnset < diffusionRatioPersimilis := by
  decide

/-- Wavelengths in nanometres. 2098 nm = 2.098 µm, the assigned detail scale. -/
def lambdaNmPaulistorum : Nat := 1947
def lambdaNmPersimilis : Nat := 2098
def lambdaNmPlanitibia : Nat := 2735

theorem chronoseriesWavelengthOrder :
    lambdaNmPaulistorum < lambdaNmPersimilis ∧
    lambdaNmPersimilis < lambdaNmPlanitibia := by
  decide

/-- The Hawaiian picture-wing step at this scale exceeds the willistoni-to-obscura step. -/
theorem hawaiianStepExceedsWillistoniStep :
    lambdaNmPlanitibia - lambdaNmPersimilis >
    lambdaNmPersimilis - lambdaNmPaulistorum := by
  decide

/-- Integer pin of k_c × λ ≈ 2π, with k in mrad/µm and λ in nm.
    2994 * 2098 + 1588 = 6283 * 1000. -/
def kMilli : Nat := 2994
def twoPiMilli : Nat := 6283
def roundingResidual : Nat := 1588

theorem wavenumberWavelengthProduct :
    kMilli * lambdaNmPersimilis + roundingResidual = twoPiMilli * 1000 := by
  decide

end MathOfBiology.Loop_3198
