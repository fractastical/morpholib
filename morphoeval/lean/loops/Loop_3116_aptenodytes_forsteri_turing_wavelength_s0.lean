/-
  Loop 3116 — Aptenodytes forsteri, Holocene (0 Ma), Turing wavelength at the
  organ-millimetre detail tier. 10^(-2.929) m = 1178 µm sits inside the
  published eggshell-thickness window 940–1320 µm. The chronoseries has one
  node, so there is no predecessor chronospecies; the successor length is the
  egg long axis (121300 µm), which holds more than 100 critical wavelengths.
-/
namespace MathOfBiology.Loop_3116

def tierWavelengthUm : Nat := 1178
def shellLoUm : Nat := 940
def shellHiUm : Nat := 1320
def eggLengthUm : Nat := 121300
def diffusionRatio : Nat := 42
def turingRatioThreshold : Nat := 28
def chronoseriesLength : Nat := 1
def ageMa : Nat := 0

/-- The 1.178 mm tier lies strictly inside the measured shell-thickness range. -/
theorem tier_inside_shell_thickness :
    shellLoUm < tierWavelengthUm ∧ tierWavelengthUm < shellHiUm := by
  decide

/-- δ = 42 exceeds the integer Turing threshold δ > 27 for these kinetics. -/
theorem diffusion_ratio_exceeds_turing_threshold :
    turingRatioThreshold < diffusionRatio := by
  decide

/-- Sufficient integer margin at δ = 42: 4δ − 5 = 163 > 130 ≥ 20√42. -/
theorem schnakenberg_margin_at_delta_42 :
    4 * diffusionRatio - 5 > 130 := by
  decide

/-- Equal diffusion (δ = 1) fails the sign condition 4δ − 5 > 0. -/
theorem equal_diffusion_margin_fails :
    4 * 1 - 5 < 0 := by
  decide

/-- Egg length is the successor scale: more than 100 wavelengths fit along it. -/
theorem egg_is_successor_scale :
    tierWavelengthUm * 100 < eggLengthUm := by
  decide

/-- Chronoseries 1/1, Holocene extant: age 0 Ma and a single node. -/
theorem holocene_terminal_node :
    ageMa = 0 ∧ chronoseriesLength = 1 := by
  decide

end MathOfBiology.Loop_3116
