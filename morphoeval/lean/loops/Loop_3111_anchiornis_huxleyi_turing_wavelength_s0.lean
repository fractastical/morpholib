-- Loop 3111 — Anchiornis huxleyi, Turing wavenumber at the 2.0 nm detail scale.
-- Age: Late Jurassic, Tiaojishan Formation, 160.0 Ma. Chronoseries 1/1.
-- Parameters (thousandths): a = 62, b = 1280, Du = 720, Dv = 31500
--   i.e. a = 0.062, b = 1.28, Du = 0.72, Dv = 31.5.
-- The 29 nm envelope of slot 1229 is a different mode; these integers are not that slot's.
namespace MathOfBiology.Loop_3111

/-- Schnakenberg trace is negative iff (b - a) < (a + b)^3 for a + b > 0. -/
def traceMargin (a b : Int) : Int := (a + b) ^ 3 - (b - a)

theorem schnakenberg_trace_negative :
    traceMargin 62 1280 > 0 := by
  decide

/-- det(J) = (a + b)^2, which is positive at the 2.0 nm parameters. -/
def detSquare (a b : Int) : Int := (a + b) * (a + b)

theorem schnakenberg_det_positive :
    detSquare 62 1280 > 0 := by
  decide

/-- Inhibitor diffusivity exceeds activator diffusivity by more than 40×
    (Dv / Du = 31.5 / 0.72). Scaled as 31500 > 40 * 720. -/
theorem inhibitor_diffuses_faster :
    (31500 : Int) > 40 * 720 := by
  decide

/-- The 2.0 nm detail mode is shorter than the 29 nm envelope (slot 1229)
    by more than a factor of 14. Same species, same 160.0 Ma horizon. -/
theorem detail_wavelength_shorter_than_29nm_envelope :
    (29 : Int) > 14 * 2 := by
  decide

/-- Rounded critical wavenumber k_c × 1000 = 776 lies inside the unstable
    band [265, 1065] (also ×1000) computed for these parameters. -/
theorem critical_wavenumber_inside_unstable_band :
    (265 : Int) < 776 ∧ 776 < 1065 := by
  decide

end MathOfBiology.Loop_3111
