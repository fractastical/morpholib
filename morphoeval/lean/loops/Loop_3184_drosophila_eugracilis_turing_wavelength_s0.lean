/-
  Loop 3184 — Drosophila eugracilis, Turing wavenumber at the 1.5 µm detail scale.
  Holocene (extant), 0 Ma. Chronoseries 13/41.
  Predecessor Drosophila erecta (longer wavelength). Successor Drosophila ficusphila (shorter).
  Integer nanometres: erecta 2100, eugracilis 1485, ficusphila 1180.
  Detail window = 12 focal wavelengths = 17820 nm.
  No sorry.
-/
namespace MathOfBiology.Loop_3184

def lamErectaNm : Nat := 2100
def lamEugracilisNm : Nat := 1485
def lamFicusphilaNm : Nat := 1180
def gammaErectaMilli : Nat := 500
def gammaEugracilisMilli : Nat := 1000
def gammaFicusphilaMilli : Nat := 1584
def windowNm : Nat := 17820

/-- Focal detail length sits strictly between the predecessor and the successor. -/
theorem predecessor_wavelength_longer :
    lamErectaNm > lamEugracilisNm := by decide

theorem successor_wavelength_shorter :
    lamEugracilisNm > lamFicusphilaNm := by decide

/-- Wavenumber k ∝ 1/λ reverses the wavelength order (integer reciprocal, 10^6 / nm). -/
theorem wavenumber_reverses_wavelength :
    (1000000 / lamFicusphilaNm) > (1000000 / lamEugracilisNm) ∧
    (1000000 / lamEugracilisNm) > (1000000 / lamErectaNm) := by decide

/-- On a 17.820 µm window the shorter wave fits strictly more stripes. -/
theorem detail_window_mode_count :
    (windowNm / lamFicusphilaNm) > (windowNm / lamEugracilisNm) ∧
    (windowNm / lamEugracilisNm) > (windowNm / lamErectaNm) := by decide

/-- The focal window is exactly twelve eugracilis wavelengths: 12 × 1485 = 17820. -/
theorem eugracilis_window_is_twelve_wavelengths :
    windowNm = 12 * lamEugracilisNm := by decide

/-- Murray γ increases from erecta through eugracilis to ficusphila, shortening λ. -/
theorem murray_gamma_orders_with_shorter_wavelength :
    gammaFicusphilaMilli > gammaEugracilisMilli ∧
    gammaEugracilisMilli > gammaErectaMilli := by decide

end MathOfBiology.Loop_3184
