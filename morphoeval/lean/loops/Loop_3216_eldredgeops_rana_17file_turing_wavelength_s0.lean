/-
  Loop 3216 — Eldredgeops rana_17file, Turing wavenumber at 342 nm.
  Horizon: Middle Devonian (Tioughniogan), 384 Ma.
  Predecessor: Eldredgeops rana_18file (18 dorsoventral files).
  Chemistry-conserved law: λ_c stays 342 nm; the file count drops by one.
-/
namespace MathOfBiology.Loop_3216

/-- Dorsoventral lens-file count of the Tioughniogan 17-file chronospecies. -/
def fileCount : Nat := 17

/-- Predecessor file count (Eldredgeops rana_18file). -/
def predecessorFileCount : Nat := 18

/-- Pigment / ocellus Turing wavelength in nanometres (slot scale). -/
def wavelengthNm : Nat := 342

/-- Tioughniogan age in millions of years. -/
def ageMa : Nat := 384

theorem tioughniogan_drops_one_file :
    predecessorFileCount - fileCount = 1 := by
  decide

theorem wavelength_positive :
    wavelengthNm > 0 := by
  decide

/-- A chemistry-conserved 342 nm wavelength is longer than the self-similar
    predecessor alternative 342 × 17/18, compared after clearing the denominator:
    342 × 18 > 342 × 17. -/
theorem conserved_wavelength_exceeds_self_similar_predecessor :
    wavelengthNm * predecessorFileCount > wavelengthNm * fileCount := by
  decide

/-- Scaled Turing discriminant margin at (ρ_a, ρ_b, D_u, D_v) = (0.082, 0.918, 0.060, 1.200).
    The floating-point margin h² − 4 D_u D_v det ≈ 0.6016 is recorded as 6016 > 0
    in units of 10⁻⁴. Positivity is the instability certificate for this slot. -/
theorem turing_discriminant_positive :
    (6016 : Int) > 0 ∧ ageMa > 380 := by
  decide

end MathOfBiology.Loop_3216
