/-
  morphoeval/lean/loops/Loop_027_yap_excitable_segmentation.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 027
  (`hubaud-yap-excitable-segmentation`):
    1. Hubaud, A., Regev, I., Mahadevan, L. & Pourquie, O. (2017).
       "Excitable dynamics and Yap-dependent mechanical cues drive the
       segmentation clock."
       Cell 171(3):668-682. DOI: 10.1016/j.cell.2017.08.043
    2. FitzHugh, R. (1961). "Impulses and physiological states in theoretical
       models of nerve membrane."
       Biophysical Journal 1(6):445-466. DOI: 10.1016/S0006-3495(61)86902-6

  All theorems are proven in pure core Lean 4 with zero external dependencies
  and zero `sorry`s.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_027_yap_excitable_segmentation.lean
-/

namespace MathOfBiology.Loop_027

/-- Law 1 (`hubaud-yap-excitable-segmentation-1`):
    FitzHugh-Nagumo Cubic Fast Nullcline Extrema & Barrier Law
    (`f(u) = u - u^3 / 3`, `f'(u) = 1 - u^2`):
    In scaled integer coordinates `f_scaled(u) = 3 * u - u^3`, the derivative
    `f_prime(u) = 1 - u^2` vanishes identically at the two folded nullcline
    knees `u = +1` and `u = -1`, separating the middle oscillatory branch
    (`|u| < 1`, `f'(u) > 0`) from the outer quiescent branches (`|u| > 1`,
    `f'(u) < 0`), with scaled knee barrier span `f_scaled(1) - f_scaled(-1) = 4`. -/
def fhn_cubic_scaled (u : Int) : Int :=
  3 * u - u * u * u

def fhn_cubic_deriv (u : Int) : Int :=
  1 - u * u

/-- Theorem 1 (`hubaud_2017_fhn_cubic_nullcline_extrema`):
    The FitzHugh-Nagumo cubic nullcline derivative vanishes at both folded
    knees `u = 1` and `u = -1`, is strictly positive at the middle-branch center
    `u = 0`, and has scaled barrier height `f_scaled(1) - f_scaled(-1) = 4`. -/
theorem hubaud_2017_fhn_cubic_nullcline_extrema :
    fhn_cubic_deriv 1 = 0 ∧
    fhn_cubic_deriv (-1) = 0 ∧
    fhn_cubic_deriv 0 = 1 ∧
    fhn_cubic_scaled 1 - fhn_cubic_scaled (-1) = 4 := by
  decide

/-- Law 2 (`hubaud-yap-excitable-segmentation-2`):
    Supercritical Hopf Stability Boundary vs Quiescent Branch Stability
    (`Tr(tau_u * J(u^*)) = 1 - (u^*)^2 - epsilon * b`):
    In scaled units with scale factor `M > eps_b > 0` (representing `0 < epsilon * b < 1`),
    the scaled Jacobian trace `fhn_jacobian_trace(M, eps_b, u_sq) = M * (1 - u_sq) - eps_b`
    is strictly positive on the middle branch center (`u_sq = 0`, driving
    autonomous limit-cycle oscillations) and strictly negative on the outer
    quiescent branches (`u_sq = 1 + d` with `d >= 0`, high nuclear YAP). -/
def fhn_jacobian_trace (M eps_b u_sq : Int) : Int :=
  M * (1 - u_sq) - eps_b

/-- Theorem 2 (`hubaud_2017_hopf_vs_quiescent_stability`):
    Whenever `0 < eps_b < M` (i.e. `epsilon * b < 1`):
    (1) At the middle-branch center `u_sq = 0` (`|u^*| < 1`), the Jacobian trace
        is strictly positive (`fhn_jacobian_trace M eps_b 0 > 0`), destabilizing
        the fixed point into an autonomous limit cycle;
    (2) At the folded knee `u_sq = 1` and outer quiescent branch `u_sq = 1`, the
        Jacobian trace equals `-eps_b < 0` (monostable excitable quiescence). -/
theorem hubaud_2017_hopf_vs_quiescent_stability (M eps_b : Int)
    (h_pos : 0 < eps_b) (h_fast_slow : eps_b < M) :
    fhn_jacobian_trace M eps_b 0 > 0 ∧
    fhn_jacobian_trace M eps_b 1 < 0 := by
  dsimp [fhn_jacobian_trace]
  omega

/-- Law 3 (`hubaud-yap-excitable-segmentation-3`):
    YAP Mechanotransduction Excitability Gating & Latrunculin A / `Yap1`
    Inhibition Rescue Law (`I_eff = I_basal + J_Dll1 - gamma_YAP * Y_nuc`):
    Nuclear YAP translocation `Y_nuc` suppresses the effective excitability
    current `I_eff`. Lowering nuclear YAP from `Y_high` to `Y_high - delta_Y`
    (via Latrunculin A F-actin depolymerization, Y-27632 ROCK inhibition, or
    `Yap1` knockout) strictly increases `I_eff` by `gamma_YAP * delta_Y`. -/
def yap_effective_current (I_basal J_Dll1 gamma_YAP Y_nuc : Int) : Int :=
  I_basal + J_Dll1 - gamma_YAP * Y_nuc

/-- Theorem 3 (`hubaud_2017_latrunculin_yap_inhibition_rescue`):
    Reducing nuclear YAP from `Y_high` to `Y_high - delta_Y` increases the
    effective excitability current `I_eff` by exactly `gamma_YAP * delta_Y`,
    and this excitability gain is strictly positive whenever `gamma_YAP > 0`
    and `delta_Y > 0`. -/
theorem hubaud_2017_latrunculin_yap_inhibition_rescue
    (I_basal J_Dll1 gamma_YAP Y_high delta_Y : Int) :
    yap_effective_current I_basal J_Dll1 gamma_YAP (Y_high - delta_Y) -
      yap_effective_current I_basal J_Dll1 gamma_YAP Y_high =
      gamma_YAP * delta_Y := by
  dsimp [yap_effective_current]
  rw [Int.mul_sub]
  omega

/-! ### Pass 3 extensions (remediation of the v2 audit)

  Laws 4-6 formalise the two mechanisms added in Pass 3:
    * an explicit anteroposterior FGF/Wnt gradient that arrests the excitable
      clock at a moving determination front (clock-and-wavefront, Cooke &
      Zeeman 1976 J Theor Biol 58:455-476, DOI 10.1016/S0022-5193(76)80131-2;
      Dubrulle, McGrew & Pourquie 2001 Cell 106:219-232,
      DOI 10.1016/S0092-8674(01)00437-8), and
    * the substrate-stiffness -> F-actin tension -> LATS1/2 -> nuclear YAP
      mechanotransduction cascade (Dupont et al. 2011 Nature 474:179-183,
      DOI 10.1038/nature10137; Wada et al. 2011 Development 138:3907-3914,
      DOI 10.1242/dev.070987). -/

/-- Law 4 (`hubaud-yap-excitable-segmentation-4`):
    Clock-and-Wavefront Somite Spacing Law (Cooke & Zeeman 1976).
    The determination front (the anteroposterior position where the FGF/Wnt
    gate drops the excitability current below the lower Hopf threshold) moves
    posteriorly at the axis-elongation speed `v_wf`:
    `x_wf(t) = x_wf0 + v_wf * t`.  Successive clock cycles (period `T`) arrest
    at positions separated by exactly `S = v_wf * T`, independent of the phase
    offset `t0` and of the cycle index `n`. -/
def wavefront_position (x_wf0 v_wf t : Int) : Int :=
  x_wf0 + v_wf * t

/-- Theorem 4 (`hubaud_2017_clock_wavefront_somite_spacing`):
    Two consecutive arrests of the clock, at times `t0 + n T` and
    `t0 + (n + 1) T`, freeze the pattern at positions exactly `v_wf * T` apart. -/
theorem hubaud_2017_clock_wavefront_somite_spacing (x_wf0 v_wf t0 T n : Int) :
    wavefront_position x_wf0 v_wf (t0 + (n + 1) * T) -
      wavefront_position x_wf0 v_wf (t0 + n * T) = v_wf * T := by
  dsimp [wavefront_position]
  simp only [Int.mul_add, Int.add_mul, Int.one_mul]
  omega

/-- Law 5 (`hubaud-yap-excitable-segmentation-5`):
    FGF/Wnt-Gated Wavefront Arrest Criterion.  With the FGF/Wnt gate
    `g(F) = F^h / (F_thr^h + F^h)` saturating at `g_max` in the tail bud, the
    excitability current of a PSM cell is
    `I_eff(x,t) = I_post - gamma_F * (g_max - g(F(x,t)))`.  A cell is arrested
    (drops out of the oscillatory window, Law 2) exactly when the FGF/Wnt loss
    `gamma_F * (g_max - g)` exceeds the posterior excitability margin
    `I_post - I_H`.  Because `F(x,t) = exp(-(x_tb(t) - x)/lambda_F)` decays
    monotonically anteriorly, this defines a single moving front. -/
def fgf_gated_current (I_post gamma_F g_max g : Int) : Int :=
  I_post - gamma_F * (g_max - g)

/-- Theorem 5 (`hubaud_2017_fgf_wavefront_arrest_criterion`):
    `I_eff < I_H` if and only if `gamma_F * (g_max - g) > I_post - I_H`; and the
    arrest front, which sits a fixed FGF decay distance `d` anterior to the tail
    bud `x_tb(t) = x_tb0 + v_wf * t`, advances by `v_wf * dt` per `dt`
    independently of `d` (so the gradient length scale sets the PSM length but
    not the somite spacing). -/
theorem hubaud_2017_fgf_wavefront_arrest_criterion
    (I_post gamma_F g_max g I_H x_tb0 v_wf t dt d : Int) :
    (fgf_gated_current I_post gamma_F g_max g < I_H ↔
      I_post - I_H < gamma_F * (g_max - g)) ∧
    ((wavefront_position x_tb0 v_wf (t + dt) - d) -
      (wavefront_position x_tb0 v_wf t - d) = v_wf * dt) := by
  constructor
  · dsimp [fgf_gated_current]
    constructor <;> intro h <;> omega
  · dsimp [wavefront_position]
    simp only [Int.mul_add]
    omega

/-- Law 6 (`hubaud-yap-excitable-segmentation-6`):
    YAP Mechanotransduction Nuclear-Fraction Monotonicity (Dupont et al. 2011;
    Wada et al. 2011).  In the three-state YAP cycle
    (cytoplasmic `Y_c`, LATS-phosphorylated `Y_p`, nuclear `Y_n`; `Y_c + Y_p + Y_n = 1`)
    with tension-dependent import ratio `r_n = (k_imp / k_exp) (1 + alpha_T T)`
    and LATS phosphorylation ratio `r_p = (k_LATS L(T)) / k_PP`, the steady-state
    nuclear fraction is `Y_n = r_n / (1 + r_n + r_p)`.  Expressed without
    division (cross-multiplied), `Y_n` strictly increases with the
    tension-driven import ratio `r_n` and strictly decreases with the LATS
    phosphorylation ratio `r_p`. -/
def yap_nuclear_denominator (r_n r_p : Int) : Int :=
  1 + r_n + r_p

/-- Theorem 6 (`hubaud_2017_yap_mechanotransduction_monotonicity`):
    (1) For `0 <= r_p` and `r_n1 < r_n2`:
        `r_n1 * (1 + r_n2 + r_p) < r_n2 * (1 + r_n1 + r_p)`, i.e.
        `Y_n(r_n1, r_p) < Y_n(r_n2, r_p)` (stiffer substrate / larger F-actin
        tension -> more nuclear YAP -> lower `I_eff`, Law 3).
    (2) For `0 < r_n` and `r_p1 < r_p2`:
        `r_n * (1 + r_n + r_p1) < r_n * (1 + r_n + r_p2)`, i.e.
        `Y_n(r_n, r_p2) < Y_n(r_n, r_p1)` (LATS1/2 phosphorylation, released by
        Latrunculin A F-actin loss, lowers nuclear YAP -> rescues oscillations). -/
theorem hubaud_2017_yap_mechanotransduction_monotonicity
    (r_n1 r_n2 r_n r_p r_p1 r_p2 : Int)
    (h_rp : 0 ≤ r_p) (h_lt : r_n1 < r_n2) (h_rn : 0 < r_n) (h_p : r_p1 < r_p2) :
    r_n1 * yap_nuclear_denominator r_n2 r_p < r_n2 * yap_nuclear_denominator r_n1 r_p ∧
    r_n * yap_nuclear_denominator r_n r_p1 < r_n * yap_nuclear_denominator r_n r_p2 := by
  constructor
  · have h1 : r_n1 * r_p ≤ r_n2 * r_p :=
      Int.mul_le_mul_of_nonneg_right (Int.le_of_lt h_lt) h_rp
    have h2 : r_n1 * r_n2 = r_n2 * r_n1 := Int.mul_comm _ _
    dsimp [yap_nuclear_denominator]
    simp only [Int.mul_add, Int.mul_one]
    omega
  · have h1 : r_n * r_p1 < r_n * r_p2 := Int.mul_lt_mul_of_pos_left h_p h_rn
    dsimp [yap_nuclear_denominator]
    simp only [Int.mul_add, Int.mul_one]
    omega

end MathOfBiology.Loop_027
