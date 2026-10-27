/-
  Loop 3121 — Argonauta argo, Turing wavenumber at the 95 µm detail scale.
  Holocene (extant), geological age 0 Ma. Chronoseries node 1/1 (no predecessor).
  Schnakenberg Jacobian entries are scaled by 100:
    f_u = 80, f_v = 100, g_u = -180, g_v = -100, D_u = 1, D_v = 18.
  These integers belong to this slot only (a = 0.10, b = 0.90, D_v/D_u = 18).
-/
namespace MathOfBiology.Loop_3121

/-- Trace of the reaction Jacobian at the ocellus steady state, scaled by 100. -/
def traceScaled : Int := 80 + (-100)

theorem steadyStateTraceNegative : traceScaled < 0 := by
  decide

/-- det J = f_u g_v - f_v g_u, scaled so both entries carry the factor 100. -/
def detScaled : Int := 80 * (-100) - 100 * (-180)

theorem steadyStateDetPositive : detScaled = 10000 := by
  decide

/-- Squared Turing margin (D_v f_u + D_u g_v)^2 > 4 D_u D_v det.
    Left side uses h = 18·80 + 1·(-100) = 1340. Right side is 4·1·18·10000. -/
def turingLhs : Int := (18 * 80 + 1 * (-100)) * (18 * 80 + 1 * (-100))
def turingRhs : Int := 4 * 1 * 18 * 10000

theorem turingWavenumberUnstable : turingLhs > turingRhs := by
  decide

/-- Equal-diffusion ablation: D_u = D_v = 1 drives the cross-diffusion margin negative,
    so no positive critical wavenumber exists. -/
def equalDiffusionMargin : Int := 1 * 80 + 1 * (-100)

theorem equalDiffusionSuppressesPattern : equalDiffusionMargin < 0 := by
  decide

/-- Campaign phenomenon size 9.48e-05 m is 948 tenths of a micrometre (94.8 µm,
    reported as the 95 µm tier). Holocene age is 0 Ma, the only chronoseries node. -/
def tierScaleTenthUm : Int := 948
def holoceneAgeMa : Int := 0

theorem holoceneDetailScaleExtant : tierScaleTenthUm > 0 ∧ holoceneAgeMa = 0 := by
  decide

end MathOfBiology.Loop_3121
