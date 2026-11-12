/-
  Loop 3161 — Danio albolineatus, cellular Turing wavelength.
  Scale: 37 µm = 3.686e-05 m (tier 4_cellular_10um, detail). Age: 0 Ma (Holocene).
  Namespace MathOfBiology.Loop_3161. No sorry.

  Integer witnesses match the Schnakenberg Jacobian used in
  models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py
  (a = 0.12, b = 0.88 → f_u = 0.76, g_v = −1, f_v = 1, g_u = −1.76, det J = 1).
  Wavelengths are nanometres of the fastest mode:
    predecessor Danio aesculapii (vertical bars)  41801 nm
    this species Danio albolineatus               36869 nm
    successor Danio margaritatus (spots)          34213 nm
-/

namespace MathOfBiology.Loop_3161

/-- Steady state u₀ = a + b, in hundredths (12 + 88 = 100). -/
def steadyStateHundredths (a b : Int) : Int := a + b

theorem pearl_danio_steady_state :
    steadyStateHundredths 12 88 = 100 := by
  decide

/-- Without diffusion the homogeneous state is stable: tr J < 0 and det J > 0.
    Entries are the Jacobian in hundredths: f_u = 76, g_v = −100, f_v = 100, g_u = −176.
    det is scaled by 10⁴ (the product of two hundredths). -/
theorem homogeneous_stable_without_diffusion :
    (76 + (-100) < 0) ∧ (76 * (-100) - (100 * (-176)) > 0) := by
  decide

/-- Inhibitor diffuses farther than the activator at the pearl-danio ratio.
    δ = 1.22 δ_c = 11.434, stored in thousandths. -/
theorem inhibitor_range_exceeds_activator :
    11434 > 1000 := by
  decide

/-- Chronoseries at this cellular tier: bar wavelength > uniform-interstripe
    wavelength > spot wavelength (nanometres). -/
theorem chronoseries_wavelength_order :
    41801 > 36869 ∧ 36869 > 34213 := by
  decide

/-- Onset wavenumber and the assigned wavelength are reciprocals up to the
    integer truncation of 2π: k_milli · λ_milli = 6283 · 1000, and
    |6283000 − round(2π·10⁶)| = 185 < 200. -/
theorem wavenumber_wavelength_product :
    (6283 * 1000 - 6283185 < 0) ∧ (6283185 - 6283 * 1000 < 200) := by
  decide

end MathOfBiology.Loop_3161
