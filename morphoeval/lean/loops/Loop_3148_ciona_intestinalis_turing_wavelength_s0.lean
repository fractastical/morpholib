/-
  Loop 3148 — Ciona intestinalis, Turing wavenumber at the detail tier.
  Scale pin: 5.056e-5 m (tier label 51 µm, tier 4_cellular_10um).
  Age: Holocene extant, 0 Ma. Chronoseries length 1, so there is no temporal
  predecessor and no temporal successor. The comparison below is across scale:
  a 1 µm melanosome, a 4 × 51 µm detail patch, and a 10 mm adult body.
  Kinetics: Schnakenberg parenthesis at (u*, v*) = (1, 9/10), Jacobian scaled by 10.
-/
namespace MathOfBiology.Loop_3148

def fu : Int := 8
def gv : Int := -10
def fv : Int := 10
def gu : Int := -18
def Du : Int := 1
def Dv : Int := 10

def traceJ : Int := fu + gv
def detJ : Int := fu * gv - fv * gu
def crossDiff : Int := Dv * fu + Du * gv

/-- tr J = f_u + g_v < 0, required for a stable homogeneous state. -/
theorem trace_negative : traceJ < 0 := by
  decide

/-- det J > 0, so the homogeneous state is not a saddle of the kinetics alone. -/
theorem det_positive : detJ > 0 := by
  decide

/-- The inhibitor diffuses faster than the activator (D_v > D_u). -/
theorem inhibitor_diffuses_faster : Du < Dv := by
  decide

/-- D_v f_u + D_u g_v > 0, Turing's differential-diffusion condition. -/
theorem cross_diffusion_positive : crossDiff > 0 := by
  decide

/-- Discriminant of the critical-wavenumber quadratic is positive, so a band of
    unstable wavenumbers exists. Integer form of
    (D_v f_u + D_u g_v)^2 − 4 D_u D_v det J. -/
theorem discriminant_positive : crossDiff * crossDiff - 4 * Du * Dv * detJ > 0 := by
  decide

/-- Ablation: forcing D_v = D_u makes the cross term negative. -/
theorem equal_diffusion_ablation : (1 : Int) * fu + (1 : Int) * gv < 0 := by
  decide

/-- This scale sits strictly between a 1 µm melanosome and the 10 mm adult body.
    The detail patch used by the script is four tier-label wavelengths (4 × 51 µm). -/
theorem scale_sandwich : 1 < 4 * 51 ∧ 4 * 51 < 10000 := by
  decide

/-- Chronoseries 1/1. The number of earlier chronospecies is zero. -/
theorem holocene_has_no_temporal_predecessor : (1 : Nat) - 1 = 0 := by
  decide

/-- SI pin 5.056e-5 m, stored as an integer count of 10⁻⁸ m. -/
def lambda_1e8_m : Nat := 5056

theorem wavelength_pin_positive : 0 < lambda_1e8_m := by
  decide

end MathOfBiology.Loop_3148
