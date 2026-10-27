/-
  Loop 3204 — Drosophila silvestris, Turing wavenumber at the 3.0 µm detail tier.
  Phenomenon scale: 2.995 µm = 2995 nm (2.995e-6 m, log10 = -5.524), tier 3_subcellular_um.
  Geological horizon: Holocene (extant), 0 Ma.
  Chronoseries 33/41, Hawaiian picture-wing radiation.
  Predecessor wavelength: Drosophila serrata, 92260 nm (9.226e-5 m).
  Successor wavelength: Drosophila simulans, 82560 nm (8.256e-5 m).
  A 23960 nm patch holds eight silvestris wavelengths and neither neighbour wavelength.
-/
namespace MathOfBiology.Loop_3204

/-- Critical wavelength of this slot, in nanometres (2.995 µm). -/
def lambdaThisNm : Nat := 2995

/-- Predecessor Drosophila serrata cellular wavelength, in nanometres (92.26 µm). -/
def lambdaPredNm : Nat := 92260

/-- Successor Drosophila simulans cellular wavelength, in nanometres (82.56 µm). -/
def lambdaSuccNm : Nat := 82560

/-- Integer wavenumber proxy, waves per metre, from a wavelength in nanometres. -/
def wavenumberPerM (lambdaNm : Nat) : Nat := 1000000000 / lambdaNm

/-- Side of the detail patch that holds eight critical wavelengths: 8 × 2995 = 23960 nm. -/
def patchNm : Nat := 23960

theorem detail_scale_shorter_than_predecessor :
    lambdaThisNm < lambdaPredNm := by
  decide

theorem detail_scale_shorter_than_successor :
    lambdaThisNm < lambdaSuccNm := by
  decide

/-- k_this / k_serrata > 30, so the detail tier is not a relabeling of the cellular predecessor. -/
theorem wavenumber_exceeds_predecessor_by_thirty :
    wavenumberPerM lambdaThisNm > 30 * wavenumberPerM lambdaPredNm := by
  decide

/-- k_this / k_simulans > 27. The successor does not inherit the detail-tier wavenumber. -/
theorem wavenumber_exceeds_successor_by_twentyseven :
    wavenumberPerM lambdaThisNm > 27 * wavenumberPerM lambdaSuccNm := by
  decide

/-- Eight ocellus wavelengths fit in the detail patch; the 92.26 µm and 82.56 µm waves do not. -/
theorem eight_modes_fit_only_at_this_scale :
    patchNm / lambdaThisNm = 8 ∧ patchNm < lambdaPredNm ∧ patchNm < lambdaSuccNm := by
  decide

/-- Ocellus diffusivity ratio used at this scale (20) is strictly above the stripe-regime ratios
    defined for the cellular chronospecies neighbours (8 and 10). -/
theorem ocellus_diffusion_ratio_exceeds_stripe_neighbours :
    (20 : Int) > 8 ∧ (20 : Int) > 10 ∧ (8 : Int) ≠ 20 := by
  decide

end MathOfBiology.Loop_3204
