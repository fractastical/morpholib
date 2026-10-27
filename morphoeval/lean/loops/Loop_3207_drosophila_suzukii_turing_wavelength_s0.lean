/-
  Loop 3207 — Drosophila suzukii, Turing onset wavelength at the 2.046 µm detail scale.
  Age 0 Ma (Holocene, extant). Chronoseries predecessor: Drosophila subobscura.
  Successor: Drosophila takahashii. All three extant; the order is comparative.

  Integer encodings of the sign facts used by the Python model:
    trace(J) < 0 and det(J) > 0 at the suzukii homogeneous state,
    subobscura activator slope f_u < 0 (no Turing band),
    and a 20% longer successor wavelength fitting fewer carrier ocelli
    in the same 24.552 µm cuticle window.
-/
namespace MathOfBiology.Loop_3207

/-- Onset wavelength of the focal carrier, in picometres (2.046 µm). -/
def lambdaCpm : Nat := 2046000

/-- Successor (D. takahashii) onset wavelength, 6/5 of the focal value. -/
def lambdaTakPm : Nat := 2455200

/-- Cuticle window used for the ocellus count: exactly 12 focal wavelengths. -/
def windowPm : Nat := 24552000

def nOcelli (window lam : Nat) : Nat := window / lam

/-- Jacobian trace at the suzukii steady state, scaled by 10^6.
    f_u + g_v = 0.943982 + (−1.745041) < 0, so diffusionless kinetics are stable. -/
def traceMicro : Int := 943982 + (-1745041)

/-- Determinant f_u g_v − f_v g_u with each entry scaled by 10^6.
    The product of two scaled entries carries 10^12. -/
def detMicro : Int :=
  943982 * (-1745041) - 1745041 * (-1943982)

/-- Subobscura kinetics a = 0.52, b = 0.18 give f_u = −17/35. -/
def fuSubNumerator : Int := -17

theorem suzukii_onset_wavelength_positive : 0 < lambdaCpm := by
  decide

theorem homogeneous_trace_negative : traceMicro < 0 := by
  decide

theorem homogeneous_determinant_positive : detMicro > 0 := by
  decide

theorem window_holds_twelve_suzukii_wavelengths :
    nOcelli windowPm lambdaCpm = 12 := by
  decide

theorem takahashii_wavelength_is_longer : lambdaCpm < lambdaTakPm := by
  decide

theorem takahashii_fits_fewer_ocelli :
    nOcelli windowPm lambdaTakPm < nOcelli windowPm lambdaCpm := by
  decide

theorem successor_drops_two_ocelli :
    nOcelli windowPm lambdaCpm - nOcelli windowPm lambdaTakPm = 2 := by
  decide

theorem subobscura_activator_autocatalysis_absent : fuSubNumerator < 0 := by
  decide

end MathOfBiology.Loop_3207
