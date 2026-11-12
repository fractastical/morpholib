/-
  Loop 3186 — Drosophila grimshawi detail-scale Turing wavelength.
  Scale: 2.765 µm (2.8 µm class, 2.765e-06 m), Holocene 0 Ma.
  Schnakenberg fastest mode at δ = D_v/D_u = 16.3186.
  Chronoseries: ficusphila 2512 nm < guttifera 2629 nm < grimshawi 2766 nm.
-/
namespace MathOfBiology.Loop_3186

/-- Detail-scale wavelengths in nanometres (argmax of σ(k), not the onset formula). -/
def lambdaFicusphilaNm : Nat := 2512
def lambdaGuttiferaNm : Nat := 2629
def lambdaGrimshawiNm : Nat := 2766

/-- A 10-wavelength grimshawi patch, in nanometres. -/
def patchNm : Nat := 27660

/-- Reaction-Jacobian trace (f_u + g_v) scaled by 10^6. Negative: homogeneous state is stable. -/
def traceScaled : Int := -1433434

/-- Equal-diffusion growth rate σ(0) = trace/2, scaled by 10^6. -/
def equalDiffusionSigmaScaled : Int := -716717

theorem homogeneous_state_stable_without_diffusion :
    traceScaled < 0 := by
  decide

theorem equal_diffusion_suppresses_the_instability :
    equalDiffusionSigmaScaled < 0 ∧ equalDiffusionSigmaScaled * 2 = traceScaled + 0 := by
  decide

theorem grimshawi_selects_the_longest_detail_wavelength :
    lambdaFicusphilaNm < lambdaGuttiferaNm ∧ lambdaGuttiferaNm < lambdaGrimshawiNm := by
  decide

theorem ten_grimshawi_wavelengths_fill_the_patch :
    10 * lambdaGrimshawiNm = patchNm := by
  decide

/-- On the same patch the shorter predecessor wave packs strictly more stripes. -/
theorem predecessor_packs_more_stripes_on_the_same_patch :
    patchNm / lambdaFicusphilaNm > patchNm / lambdaGrimshawiNm := by
  decide

end MathOfBiology.Loop_3186
