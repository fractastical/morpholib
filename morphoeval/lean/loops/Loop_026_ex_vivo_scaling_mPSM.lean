/-
  morphoeval/lean/loops/Loop_026_ex_vivo_scaling_mPSM.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 026
  (`lauschke-ex-vivo-scaling-mPSM`):
    1. Lauschke, V. M., Tsiairis, C. D., Francois, P. & Aulehla, A. (2013).
       "Scaling of embryonic patterning based on phase-gradient encoding."
       Nature 493(7430):101-105. DOI: 10.1038/nature11804
    2. Aulehla, A. et al. (2008). "A beta-catenin gradient links the clock and
       wavefront systems in mouse embryo segmentation."
       Nature Cell Biology 10(2):186-193. DOI: 10.1038/ncb1679

  All theorems are proven in pure core Lean 4 with zero external dependencies
  and zero `sorry`s.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_026_ex_vivo_scaling_mPSM.lean
-/

namespace MathOfBiology.Loop_026

/-- Law 1 (`lauschke-ex-vivo-scaling-mPSM-1`):
    Lauschke et al. (2013) Proportional Segment Scaling Law under Invariant
    Total Phase Span `Delta_phi = Phi_0`:
    Because the total phase difference across the active monolayer presomitic
    mesoderm (mPSM) radius `R` is invariant (`Delta_phi_total = Phi_0`), each
    newly specified segment width `S(alpha, R) = alpha * R` scales strictly
    linearly with the current residual mPSM radius `R` (`alpha ≈ 0.11`). -/
def scaled_segment_width (alpha R : Nat) : Nat :=
  alpha * R

/-- Theorem 1 (`lauschke_2013_proportional_segment_scaling`):
    Scaling the residual mPSM explant radius `R` by any factor `k` scales the
    newly formed segment width `S(alpha, k * R)` by `k`, and the segment width
    is strictly additive across tissue sub-radii `R1 + R2`. -/
theorem lauschke_2013_proportional_segment_scaling (alpha k R R1 R2 : Nat) :
    scaled_segment_width alpha (k * R) = k * scaled_segment_width alpha R ∧
    scaled_segment_width alpha (R1 + R2) =
      scaled_segment_width alpha R1 + scaled_segment_width alpha R2 := by
  constructor
  · dsimp [scaled_segment_width]
    rw [← Nat.mul_assoc alpha k R, Nat.mul_comm alpha k, Nat.mul_assoc k alpha R]
  · dsimp [scaled_segment_width]
    exact Nat.left_distrib alpha R1 R2

/-- Law 2 (`lauschke-ex-vivo-scaling-mPSM-2`):
    Lauschke et al. (2013) Geometric Progression of Residual mPSM Radius
    (`R_{n+1} = (1 - alpha) * R_n`):
    In scaled integer units with total scale `M` and proportional fraction `alpha`,
    after budding a segment of scaled width `S_n = alpha * R_n`, the remaining
    unsegmented mPSM radius is `R_{n+1} = (M - alpha) * R_n`. -/
def residual_mpsm_radius_step (M alpha R_n : Int) : Int :=
  (M - alpha) * R_n

def budded_segment_step (alpha R_n : Int) : Int :=
  alpha * R_n

/-- Theorem 2 (`lauschke_2013_geometric_residual_radius_conservation`):
    At every oscillation cycle `n`, the sum of the newly budded peripheral
    segment `S_n = alpha * R_n` and the contracted residual mPSM radius
    `R_{n+1} = (M - alpha) * R_n` exactly equals the prior scaled radius `M * R_n`. -/
theorem lauschke_2013_geometric_residual_radius_conservation (M alpha R_n : Int) :
    residual_mpsm_radius_step M alpha R_n + budded_segment_step alpha R_n = M * R_n := by
  dsimp [residual_mpsm_radius_step, budded_segment_step]
  rw [Int.sub_mul, Int.sub_add_cancel]

/-- Law 3 (`lauschke-ex-vivo-scaling-mPSM-3`):
    Lauschke et al. (2013) Phase-Wave Velocity Proportionality Law
    (`v_wave(R) ∝ R / T_osc`):
    When a full `2*pi` phase wave sweeps across residual mPSM radius `R` over
    central pacemaker period `T_osc`, the kinematic wave displacement per
    oscillation cycle is proportional to `R`, satisfying the kinematic balance
    `v_wave * T_osc - lambda_ratio * R = 0`. -/
def phase_wave_kinematic_residual (v_wave T_osc lambda_ratio R : Int) : Int :=
  v_wave * T_osc - lambda_ratio * R

/-- Theorem 3 (`lauschke_2013_phase_wave_velocity_scaling`):
    When the radial phase-wave velocity `v_wave = lambda_ratio * c` scales
    linearly with residual mPSM radius `R = T_osc * c`, the kinematic phase-wave
    residual vanishes identically for any explant scale factor `k`. -/
theorem lauschke_2013_phase_wave_velocity_scaling (lambda_ratio T_osc c : Int) :
    phase_wave_kinematic_residual (lambda_ratio * c) T_osc lambda_ratio (T_osc * c) = 0 := by
  dsimp [phase_wave_kinematic_residual]
  rw [Int.mul_assoc lambda_ratio c T_osc, Int.mul_comm c T_osc]
  omega

/-! ### Pass 3 additions (2026-10-19, audit v3)

  Laws 4-6 formalise the two mechanisms added in the Pass 3 remediation of
  `2014-lauschke-ex-vivo-scaling-mPSM.py`:
    * an explicit FGF/Wnt source-sink signalling gradient on the shrinking
      mPSM disc whose half-maximum length scales with the residual radius
      (Law 4), driving a dynamic relaxation of the Notch phase-wave wavelength
      (Law 5; Lauschke et al. 2013 Fig. 4, Beaupeux & Francois 2016);
    * a disordered cell-reaggregation lattice of Kuramoto-Sakaguchi phase
      oscillators (Tsiairis & Aulehla 2016; Kuramoto 1984) in which target-wave
      pacemakers emerge at local frequency maxima and the fastest pacemaker
      wins (Law 6). -/

/-- Law 4 (`lauschke-ex-vivo-scaling-mPSM-4`):
    FGF/Wnt source-sink signalling gradient on the active mPSM disc.
    In the diffusion-dominated (sink-controlled) limit the quasi-static
    profile with a central source `W(0) = W_0` and a peripheral sink
    `W(R) = 0` is linear, `W(r) = W_0 * (R - r) / R`; in scaled integer units
    the unnormalised profile is `W_0 * (R - r)`. -/
def source_sink_gradient (W0 R r : Int) : Int :=
  W0 * (R - r)

/-- Theorem 4 (`lauschke_2013_fgf_wnt_gradient_scale_covariance`):
    Scaling the residual mPSM radius `R` and the sampling position `r` by the
    same factor `k` scales the source-sink signalling profile by `k`, so the
    position of any iso-concentration level set (e.g. the half-maximum length
    `l_half = R / 2`) is proportional to `R`: the gradient shrinks with the
    tissue (the "shrinking FGF/Wnt gradient" that drives wavelength adaptation). -/
theorem lauschke_2013_fgf_wnt_gradient_scale_covariance (W0 k R r : Int) :
    source_sink_gradient W0 (k * R) (k * r) = k * source_sink_gradient W0 R r := by
  dsimp [source_sink_gradient]
  rw [← Int.mul_sub, Int.mul_left_comm]

/-- Law 5 (`lauschke-ex-vivo-scaling-mPSM-5`):
    Dynamic wavelength adaptation `d lambda / dt = -(lambda - c * l_half(t)) / tau`.
    Discretised with step `h / M` (`0 <= h <= M`), the wavelength error
    `e_n = lambda_n - lambda_target` obeys `M * e_{n+1} = (M - h) * e_n`. -/
def wavelength_adaptation_error_step (M h e : Int) : Int :=
  (M - h) * e

/-- Theorem 5 (`lauschke_2013_wavelength_adaptation_contraction`):
    For any relaxation step `0 <= h <= M` the scaled wavelength error after
    one adaptation step is no larger in magnitude than before it, so the Notch
    phase-wave wavelength relaxes monotonically toward the target set by the
    instantaneous FGF/Wnt gradient length (and the total phase span toward
    `Phi_0`). -/
theorem lauschke_2013_wavelength_adaptation_contraction (M h e : Int)
    (h0 : 0 ≤ h) (hM : h ≤ M) :
    Int.natAbs (wavelength_adaptation_error_step M h e) ≤ Int.natAbs (M * e) := by
  dsimp [wavelength_adaptation_error_step]
  rw [Int.natAbs_mul, Int.natAbs_mul]
  apply Nat.mul_le_mul_right
  have h1 : 0 ≤ M - h := Int.sub_nonneg_of_le hM
  have h2 : 0 ≤ M := Int.le_trans h0 hM
  have h3 : M - h ≤ M := Int.sub_le_self M h0
  have := Int.natAbs_of_nonneg h1
  have := Int.natAbs_of_nonneg h2
  omega

/-- Law 6 (`lauschke-ex-vivo-scaling-mPSM-6`):
    Kuramoto (1984) target-wave dispersion relation for the nonlinear phase
    diffusion equation `theta_t = omega + D lap theta + beta |grad theta|^2`
    that is the continuum limit of the Kuramoto-Sakaguchi reaggregate lattice:
    a pacemaker with frequency excess `beta * k^2` emits a target wave of
    wavenumber `k`. -/
def target_wave_frequency_excess (beta k : Nat) : Nat :=
  beta * k * k

/-- Theorem 6 (`tsiairis_2016_pacemaker_dispersion_monotone`):
    The dispersion relation is monotone in `k`: a faster pacemaker emits a
    shorter (larger-`k`) target wave, which is why, in the disordered
    reaggregate lattice, the local frequency maximum with the largest excess
    entrains its neighbours and the slower foci are progressively eliminated
    (Tsiairis & Aulehla 2016 multiple-foci consolidation). -/
theorem tsiairis_2016_pacemaker_dispersion_monotone (beta k1 k2 : Nat) (h : k1 ≤ k2) :
    target_wave_frequency_excess beta k1 ≤ target_wave_frequency_excess beta k2 := by
  dsimp [target_wave_frequency_excess]
  exact Nat.mul_le_mul (Nat.mul_le_mul_left beta h) h

end MathOfBiology.Loop_026
