/-
  morphoeval/lean/loops/Loop_3228_gallus_gallus_turing_wavelength_s0.lean
  Slot 3228 — Gallus gallus (chick), Holocene extant, chronoseries 1/1.
  Scale: Turing pigment-stripe / ocellus wavelength λ_c = 319.2 µm
  (3.192 × 10⁻⁴ m, log10 = −3.496, tier 5_primordium_100um, detail s0).
  Cohort: Turing reaction-diffusion wavelength selection, wavenumber k_c.
  No predecessor and no successor chronospecies exist in this singleton series,
  so the Holocene wavelength is not a copy of any sibling fossil parameter set.
-/
namespace MathOfBiology.Loop_3228

/-- Critical wavelength in tenths of a micrometre (319.2 µm). -/
def lambdaCTenthUm : Nat := 3192

/-- Geological age of the sole Gallus chronospecies, in Ma. -/
def ageMa : Nat := 0

/-- Position and length of the Gallus chronoseries (1/1). -/
def chronoseriesIndex : Nat := 1
def chronoseriesLength : Nat := 1

/-- Schnakenberg inhibitor/activator diffusivities used at this scale (Dv, Du). -/
def dv : Nat := 16
def du : Nat := 1

/-- Cleared-denominator activator slope 5·(2b − 1) with b = 0.7995 → 599/1000,
    stored as the integer 599 so the onset gap below is denominator-free. -/
def activatorSlopeMilli : Nat := 599

theorem wavelength_strictly_positive : 0 < lambdaCTenthUm := by
  decide

theorem holocene_extant_age : ageMa = 0 := by
  decide

theorem singleton_chronoseries : chronoseriesIndex = 1 ∧ chronoseriesLength = 1 := by
  decide

/-- Index 1 has no earlier Gallus chronospecies (Nat subtraction). -/
theorem no_predecessor_chronospecies : chronoseriesIndex - 1 = 0 := by
  decide

/-- No successor: the series length equals the index. -/
theorem no_successor_chronospecies : chronoseriesLength - chronoseriesIndex = 0 := by
  decide

/-- Onset wavenumber gap for (Du, Dv) = (1, 16) at b = 0.7995.
    h/γ = Dv·(2b−1) − Du = 8.584, and h² > 4·Du·Dv·det with det/γ² = 1.
    Integers: (8584)² > 4·1·16·1000², i.e. the same comparison after ×1000. -/
theorem turing_onset_gap : 8584 * 8584 > 4 * du * dv * 1000 * 1000 := by
  decide

/-- Equal diffusion (Dv = Du = 1) gives h/γ = (2b−1) − 1 < 0, so the
    homogeneous state is not Turing-unstable. 599 < 1000 encodes
    0.599 < 1 after clearing the 0.7995 slope. -/
theorem equal_diffusion_ablation_is_stable : activatorSlopeMilli < 1000 := by
  decide

/-- Six critical wavelengths span the dorsal-tract patch used in the script
    (6 × 319.2 µm = 1915.2 µm). -/
theorem six_wavelength_domain : 6 * lambdaCTenthUm = 19152 := by
  decide

end MathOfBiology.Loop_3228
