/-
  Loop 3164 — Danio rerio, Holocene (0 Ma), detail cellular scale.
  Phenomenon wavelength λ = 24.08 µm = 2.408e-05 m (tier 4_cellular_10um).
  Turing margin at scale S = 100:
    h = Dv * fu - Du * |gv|
    margin = h^2 - 4 * Du * Dv * det * S
  Positive margin: diffusion-driven instability. Negative: that axis is stable.
  Integers are the slot-3164 Schnakenberg Jacobian and diffusivities rounded to
  0.01 (fu = 0.84, |gv| = det = 1.82, Du = 0.80). Inhibitor diffusivities, with
  Dv_y fixed at 20.80 so the DV peak stays on k_c: D. rerio Dv_x = 5.40
  (anisotropy 3.85); predecessor D. nigrofasciatus Dv_x = 18.57 (anisotropy 1.12).
  Predecessor is unstable on both axes (spots). Extant D. rerio is unstable only
  on the dorsoventral axis (horizontal stripes).
-/
namespace MathOfBiology.Loop_3164

/-- Signed Turing discriminant. `gv` is |g_v|. -/
def turingMargin (dv fu du gv det scale : Nat) : Int :=
  let h : Int := dv * fu - du * gv
  h * h - (4 * du * dv * det * scale)

/-- Holocene D. rerio, DV axis, λ-selecting inhibitor diffusivity: unstable. -/
theorem rerio_dv_axis_turing_unstable :
    turingMargin 2080 84 80 182 182 100 > 0 := by
  decide

/-- Holocene D. rerio, AP axis, reduced inhibitor diffusivity: stable (no spots). -/
theorem rerio_ap_axis_stable :
    turingMargin 540 84 80 182 182 100 < 0 := by
  decide

/-- Predecessor D. nigrofasciatus at the same 24 µm scale: both axes unstable. -/
theorem nigrofasciatus_both_axes_unstable :
    turingMargin 2080 84 80 182 182 100 > 0 ∧
    turingMargin 1857 84 80 182 182 100 > 0 := by
  decide

/-- Six critical wavelengths fill the 144.48 µm DV window (24.08 µm = 2408 hundredths). -/
theorem six_wavelengths_span_the_cellular_window :
    6 * 2408 = 14448 := by
  decide

/-- The rerio DV/AP inhibitor ratio (385/100) exceeds the predecessor ratio (112/100). -/
theorem rerio_anisotropy_exceeds_predecessor :
    385 * 100 > 112 * 100 := by
  decide

end MathOfBiology.Loop_3164
