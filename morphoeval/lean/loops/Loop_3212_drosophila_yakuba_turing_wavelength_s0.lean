/-
  morphoeval/lean/loops/Loop_3212_drosophila_yakuba_turing_wavelength_s0.lean
  Slot 3212 — Drosophila yakuba, Holocene (0 Ma), subcellular Turing wavelength.
  Phenomenon: λ* = 2.157 µm = 2157 nm (tier 3_subcellular_um, detail s0).
  Predecessor chronospecies: Drosophila willistoni at cellular λ = 86300 nm.
  Kinetics: Schnakenberg (1979). Jacobian entries below are the steady-state
  values f_u = 0.8, g_v = −1 scaled by 10 so the inequalities are integral.
  No sorry.
-/
namespace MathOfBiology.Loop_3212

/-- Trace of the Schnakenberg Jacobian at (u0, v0), ×10. f_u = 8, g_v = −10. -/
def traceScaled : Int := 8 + (-10)

/-- Diffusion-free steady state is attracting: tr J < 0. -/
theorem homogeneous_state_stable_trace : traceScaled < 0 := by
  decide

/-- Cross coefficient D_v f_u + D_u g_v with D_u = 1, D_v = 20, ×10. -/
def crossScaled : Int := 20 * 8 + 1 * (-10)

/-- Inhibitor diffuses far enough for a positive cross term (necessary for Turing). -/
theorem inhibitor_long_range_cross_positive : crossScaled > 0 := by
  decide

/-- (cross)² > 4 D_u D_v det, with det(J) = 1 and cross scaled by 10
    so the determinant term carries a factor 100. -/
theorem diffusion_driven_instability_discriminant :
    crossScaled * crossScaled > 4 * 1 * 20 * 100 := by
  decide

/-- Equal-diffusion ablation D_u = D_v = 1 makes the cross term negative,
    so diffusion cannot destabilise the steady state. -/
theorem equal_diffusion_control_is_stable : (1 * 8 + 1 * (-10)) < 0 := by
  decide

/-- At 0 Ma the yakuba subcellular wavelength (2157 nm) is more than 30×
    finer than the willistoni cellular predecessor band (86300 nm).
    86300 / 2157 = 40 in integer division. -/
theorem yakuba_subcellular_wavenumber_exceeds_willistoni :
    (86300 : Int) / 2157 > 30 := by
  decide

end MathOfBiology.Loop_3212
