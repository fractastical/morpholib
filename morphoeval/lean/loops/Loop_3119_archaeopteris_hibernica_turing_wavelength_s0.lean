/-
  Loop 3119 — Archaeopteris hibernica, Famennian (370 Ma), 31 µm epidermal Turing wavelength.
  Namespace MathOfBiology.Loop_3119. No sorry. Core Lean 4 only.
  The integers are the slot's own chronoseries, not a sibling loop's parameters:
    Tetraxylopteris schmidtii (Givetian, 385 Ma, unplanated cortex) λ = 66 µm
    Archaeopteris hibernica (Famennian, 370 Ma, planated megaphyll) λ = 31 µm
    Elkinsia polymorpha (late Famennian, 365 Ma, cupulate frond) λ = 21 µm
-/
namespace MathOfBiology.Loop_3119

/-- Focal critical wavelength, micrometres. -/
def lambdaHibernicaUm : Nat := 31

/-- Periodic analysis domain holding six critical waves, micrometres. -/
def domainUm : Nat := 186

def modeCount : Nat := 6

theorem domainHoldsSixWavelengths : modeCount * lambdaHibernicaUm = domainUm := by
  decide

/-- Predecessor wavelength on the unplanated aneurophyte cortex. -/
def lambdaTetraxylopterisUm : Nat := 66

/-- Successor wavelength on the thinner cupulate seed-plant frond. -/
def lambdaElkinsiaUm : Nat := 21

theorem planationThenMiniaturization :
    lambdaElkinsiaUm < lambdaHibernicaUm ∧ lambdaHibernicaUm < lambdaTetraxylopterisUm := by
  decide

/-- Diffusivities times 100, so the ratio is exact in Nat: Dv/Du = 24. -/
def duCenti : Nat := 310

def dvCenti : Nat := 7440

theorem inhibitorOutrangesActivator : duCenti < dvCenti := by
  decide

theorem diffusionRatioIsTwentyFour : dvCenti = 24 * duCenti := by
  decide

/-- Positive diffusion-driven cross Dv·f_u − Du·|g_v| in milli-Jacobian units.
    f_u ≈ 0.732 and |g_v| ≈ 0.778 at the Famennian steady state. -/
def turingCrossProxy : Nat := 7440 * 732 - 310 * 778

theorem diffusionDrivenCrossPositive : turingCrossProxy > 0 := by
  decide

end MathOfBiology.Loop_3119
