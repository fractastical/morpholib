/-
  Loop 3171 — Dorudon atrox (Late Eocene basilosaurid), slot scale 253 µm.
  Horizon: Priabonian, 37.0 Ma. Cohort: Turing reaction-diffusion wavelength selection.
  Integer image of the Schnakenberg Jacobian used by
  models/2026-delphi10000-dorudon-atrox-turing-wavelength-s0.py
  (fu, fv, gu, gv scaled by 1e4; Du, Dv rounded in µm² per reaction-time).
  No imports. No sorry.
-/
namespace MathOfBiology.Loop_3171

/-- Assigned critical wavelength, micrometres (tier 5 primordium, detail). -/
def lambdaUm : Nat := 253

/-- Protocetid hindlimb bud, in units of λ_c. Predecessor on the limb-to-flipper series. -/
def predecessorModes : Nat := 8

/-- Dorudon atrox vestigial hindlimb bud at 37.0 Ma, in units of λ_c. -/
def dorudonModes : Nat := 3

/-- Crown-cetacean external hindlimb: the domain is gone. -/
def successorHindlimbModes : Nat := 0

def fu : Int := 6723
def fv : Int := 10020
def gu : Int := -16723
def gv : Int := -10020
def Du : Int := 406
def Dv : Int := 6486

def detJ : Int := fu * gv - fv * gu
def traceJ : Int := fu + gv

/-- (D_v f_u + D_u g_v)² − 4 D_u D_v det(J). Positive ⇒ diffusion-driven instability. -/
def turingMargin : Int :=
  (Dv * fu + Du * gv) * (Dv * fu + Du * gv) - 4 * Du * Dv * detJ

/-- Same margin with D_u = D_v. Negative ⇒ equal diffusion does not destabilise. -/
def equalDiffusionMargin : Int :=
  (Du * fu + Du * gv) * (Du * fu + Du * gv) - 4 * Du * Du * detJ

theorem trace_negative_without_diffusion : traceJ < 0 := by
  decide

theorem determinant_positive : detJ > 0 := by
  decide

theorem turing_margin_positive : turingMargin > 0 := by
  decide

theorem equal_diffusion_margin_negative : equalDiffusionMargin < 0 := by
  decide

theorem dorudon_domain_three_wavelengths : dorudonModes * lambdaUm = 759 := by
  decide

theorem predecessor_domain_eight_wavelengths : predecessorModes * lambdaUm = 2024 := by
  decide

theorem hindlimb_domain_shrinks :
    predecessorModes > dorudonModes ∧ dorudonModes > successorHindlimbModes := by
  decide

end MathOfBiology.Loop_3171
