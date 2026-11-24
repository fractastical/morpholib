/-
  Loop 3230 — Gasterosteus doryssus_mid, Turing wavenumber at the 42 µm detail scale.
  Age: Late Miocene (Truckee Fm), 10.01 Ma, vestigial pelvic spine, T = +9 kyr.
  Predecessor: G. doryssus_early (10.02 Ma, T = 0, full girdle).
  Successor: G. doryssus_late (9.99 Ma, T = +22 kyr, pelvic loss).
  Wavelengths are the Schnakenberg fastest-growing modes in nanometres, pinned by
  models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py.
  No sorry.
-/
namespace MathOfBiology.Loop_3230

/-- Predicted critical wavelengths in nanometres: early 32.139 µm, mid 41.940 µm, late 56.898 µm. -/
def lambdaEarlyNm : Nat := 32139
def lambdaMidNm : Nat := 41940
def lambdaLateNm : Nat := 56898

/-- At this 42 µm scale the selected wavelength lengthens as the pelvis reduces. -/
theorem wavelength_lengthens_along_pelvic_reduction :
    lambdaEarlyNm < lambdaMidNm ∧ lambdaMidNm < lambdaLateNm := by
  decide

/-- Diffusivity ratios ×10. Mid (vestigial spine) sits strictly between early and late. -/
def ratioEarlyTimes10 : Nat := 142
def ratioMidTimes10 : Nat := 180
def ratioLateTimes10 : Nat := 248

theorem diffusivity_ratio_increases_as_armor_is_lost :
    ratioEarlyTimes10 < ratioMidTimes10 ∧ ratioMidTimes10 < ratioLateTimes10 := by
  decide

/-- Schnakenberg reaction trace in milli-units at (a, b) = (0.062, 1.18):
    fu = 900, gv = −1543. Equal diffusivities cannot be Turing-unstable when the trace is negative. -/
def reactionTraceMilli : Int := 900 + (-1543)

theorem equal_diffusion_reaction_trace_is_negative :
    reactionTraceMilli < 0 := by
  decide

/-- Mid-stage Turing margin h = Dv·fu + Du·gv with Du = 100, Dv = 1800, fu = 900, gv = −1543. -/
def midTuringMarginH : Int := 1800 * 900 + 100 * (-1543)

theorem mid_stage_turing_margin_is_positive :
    midTuringMarginH > 0 := by
  decide

end MathOfBiology.Loop_3230
