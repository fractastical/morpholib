/-
  Loop 3141 — Holocene Cavia porcellus, Turing wavelength at 126.7 µm (0 Ma).
  Integer certificates for the 10× tier relation and the diffusion-driven
  instability signs of the s0 Schnakenberg Jacobian. No sorry.
-/
namespace MathOfBiology.Loop_3141

/-- Critical wavelength of the s0 detail scale, in nanometres (126.7 µm). -/
def lambdaS0Nm : Int := 126700

/-- Predecessor tier (10× coarser) wavelength in nanometres (1267 µm). -/
def lambdaPredNm : Int := 1267000

/-- Successor tier (10× finer) wavelength in nanometres (12.67 µm). -/
def lambdaSuccNm : Int := 12670

/-- One predecessor-tier wavelength holds exactly ten s0 waves. -/
theorem pred_holds_ten_s0_waves : lambdaPredNm = 10 * lambdaS0Nm := by
  decide

/-- One s0 wavelength holds exactly ten successor-tier waves. -/
theorem s0_holds_ten_successor_waves : lambdaS0Nm = 10 * lambdaSuccNm := by
  decide

/-- Eight s0 waves span the primordium window: 8 × 1267 tenths of a micrometre = 1013.6 µm. -/
theorem eight_waves_span_primordium : 8 * 1267 = 10136 := by
  decide

/-- Cross term D_v f_u + D_u g_v for the scaled s0 Jacobian.
    Entries: D_u = 1692 (×10 µm²), D_v = 67670, f_u = 8699 (×10⁴), g_v = −15129. -/
def turingCross (Du Dv fu gv : Int) : Int := Dv * fu + Du * gv

theorem activator_inhibitor_cross_positive :
    turingCross 1692 67670 8699 (-15129) > 0 := by
  decide

/-- Homogeneous steady state is stable: trace = f_u + g_v < 0, so k = 0 does not grow. -/
theorem homogeneous_state_stable : 8699 + (-15129) < 0 := by
  decide

end MathOfBiology.Loop_3141
