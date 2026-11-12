/-
  Loop 3175 — Drosophila americana detail-scale Turing wavelength.
  Expressed scale: 2.117 µm = 2117 nm at 0 Ma (Holocene).
  Predecessor Drosophila affinis: 1958 nm. Successor Drosophila ananassae: 1900 nm.
  These detail-scale lengths are buffered (body-length exponent 0.35).
  They are not the primordium-scale sibling wavelengths 95940 nm, 138700 nm, 84620 nm.
-/
namespace MathOfBiology.Loop_3175

/-- Detail-scale wavelength in nanometres. -/
def detailWavelengthNm (lambdaNm : Nat) : Nat := lambdaNm

/-- On a fixed 80 µm (80000 nm) ocellar window, stripe count is integer L / λ. -/
def stripeCount (lambdaNm : Nat) : Nat := 80000 / lambdaNm

theorem americana_detail_wavelength_positive :
    detailWavelengthNm 2117 > 0 := by
  decide

/-- Virilis-group americana keeps a longer detail-scale wavelength than both neighbors. -/
theorem detail_wavelength_chronoseries_order :
    detailWavelengthNm 2117 > detailWavelengthNm 1958
      ∧ detailWavelengthNm 1958 > detailWavelengthNm 1900 := by
  decide

/-- Shorter successor wavelength packs more stripes into the same 80 µm window. -/
theorem successor_packs_more_detail_stripes :
    stripeCount 1900 > stripeCount 1958 ∧ stripeCount 1958 > stripeCount 2117 := by
  decide

/-- Detail-scale americana/affinis ratio is strictly smaller than the primordium-scale ratio
    138700/95940. Cross-multiplied: 2117 * 95940 < 138700 * 1958. -/
theorem detail_ratio_buffered_against_affinis :
    2117 * 95940 < 138700 * 1958 := by
  decide

/-- Same buffer against ananassae: primordium ratio 138700/84620 exceeds 2117/1900. -/
theorem detail_ratio_buffered_against_ananassae :
    2117 * 84620 < 138700 * 1900 := by
  decide

/-- Discriminant of the fu = 4/5 Jacobian, cleared by 25.
    At r = 10 the Turing bracket is positive; at equal diffusion r = 1 it is negative. -/
theorem turing_discriminant_survives_at_r10_and_fails_at_equal_diffusion :
    ((4 * 10 - 5) ^ 2 - 100 * 10 > 0)
      ∧ ((4 * 1 - 5) ^ 2 - 100 * 1 < 0) := by
  decide

end MathOfBiology.Loop_3175
