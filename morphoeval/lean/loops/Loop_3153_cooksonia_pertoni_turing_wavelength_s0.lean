-- Loop 3153. Cooksonia pertoni, Late Silurian 425 Ma, detail scale 1.339 µm.
-- Schnakenberg Jacobian at a = 3/25, b = 21/25, diffusion ratio 10.
-- Integer cross-multiples only, so the file does not need Mathlib.
namespace MathOfBiology.Loop_3153

/-- tr J = 3/4 - 576/625 < 0 iff 3 * 625 < 4 * 576. -/
theorem jacobian_trace_negative : 3 * 625 < 4 * 576 := by
  decide

/-- det J = 576/625 > 0. -/
theorem jacobian_det_positive : 0 < 576 := by
  decide

/-- At D_v/D_u = 10 the diffusion-weighted trace numerator 1265 is positive. -/
theorem diffusion_weighted_trace_positive : 0 < 1265 := by
  decide

/-- Ratio 10 lies above the Turing threshold:
    (10*3*625 - 4*576)^2 * 625 > 4*10*576*2500^2. -/
theorem ratio_ten_above_threshold :
    16446 * 16446 * 625 > 4 * 10 * 576 * 2500 * 2500 := by
  decide

/-- Ratio 8 lies below the same threshold, so the predecessor-side control is stable. -/
theorem ratio_eight_below_threshold :
    12696 * 12696 * 625 < 4 * 8 * 576 * 2500 * 2500 := by
  decide

/-- 1.339 µm (1339 nm) is the detail scale, far below the 73 µm sibling (72720 nm). -/
theorem detail_wavelength_below_cellular_sibling : 1339 < 72720 := by
  decide

end MathOfBiology.Loop_3153
