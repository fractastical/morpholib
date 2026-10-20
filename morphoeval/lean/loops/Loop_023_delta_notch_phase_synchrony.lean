/-
  morphoeval/lean/loops/Loop_023_delta_notch_phase_synchrony.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 023
  (`riedel-delta-notch-phase-synchrony`):
    1. Riedel-Kruse, I. H., Müller, C. & Oates, A. C. (2007).
       "Synchrony dynamics during initiation, failure, and rescue of the
       segmentation clock." Science 317(5846):1911-1915.
       DOI: 10.1126/science.1142538
    2. Morelli, L. G., Ares, S., Herrgen, L., Schröter, C., Jülicher, F. &
       Oates, A. C. (2009). "Delayed coupling theory of vertebrate segmentation."
       HFSP Journal 3(1):55-66.
       DOI: 10.2976/1.3027088
    3. Herrgen, L., Ares, S., Morelli, L. G., Schröter, C., Jülicher, F. &
       Oates, A. C. (2010). "Intercellular coupling regulates the period of the
       segmentation clock." Current Biology 20(14):1244-1253.
       DOI: 10.1016/j.cub.2010.06.034

  All theorems are proven in pure core Lean 4 with zero external dependencies
  and zero `sorry`s.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_023_delta_notch_phase_synchrony.lean
-/

namespace MathOfBiology.Loop_023

/-- Law 1 (`riedel-delta-notch-phase-synchrony-1`):
    Riedel-Kruse et al. (2007) & Morelli et al. (2009) Delayed Kuramoto Collective
    Frequency Shift Law:
    In the delayed Kuramoto phase-oscillator lattice
      `dtheta_i/dt = omega_0 + (kappa / z) * sum_j sin(theta_j(t - tau_c) - theta_i(t))`,
    the spatially uniform synchronized state `theta_i(t) = Omega * t` satisfies
    the collective frequency balance
      `Omega = omega_0 - kappa * sin_delay_phase`,
    where `sin_delay_phase` represents `sin(Omega * tau_c)`. -/
def delayed_kuramoto_collective_frequency (omega_0 kappa sin_delay_phase : Int) : Int :=
  omega_0 - kappa * sin_delay_phase

/-- Theorem 1 (`riedel_2007_collective_frequency_shift_and_dapt_limit`):
    (i) Under complete Notch blockade (`kappa = 0`, e.g. saturating DAPT treatment
        or *bea*/*des*/*aei*/*mib* loss-of-function), the collective frequency equals
        the intrinsic single-cell autonomous frequency `omega_0`;
    (ii) Reversing the sign of the delay phase coupling term `sin_delay_phase`
         symmetrically shifts the collective frequency above and below `omega_0`
         so that their sum is `2 * omega_0`. -/
theorem riedel_2007_collective_frequency_shift_and_dapt_limit (omega_0 kappa sin_delay_phase : Int) :
    delayed_kuramoto_collective_frequency omega_0 0 sin_delay_phase = omega_0 ∧
    delayed_kuramoto_collective_frequency omega_0 kappa sin_delay_phase +
      delayed_kuramoto_collective_frequency omega_0 kappa (-sin_delay_phase) = 2 * omega_0 := by
  constructor
  · dsimp [delayed_kuramoto_collective_frequency]
    omega
  · dsimp [delayed_kuramoto_collective_frequency]
    rw [Int.mul_neg]
    omega

/-- Law 2 (`riedel-delta-notch-phase-synchrony-2`):
    Morelli et al. (2009) Delayed Coupling Linear Stability Stiffness Criterion:
    Linearizing the delayed Kuramoto lattice around `theta_i(t) = Omega * t + delta_theta_i(t)`
    yields the effective intercellular restoring stiffness
      `kappa_eff = kappa * cos_delay_phase`,
    which evolves a pairwise phase perturbation `delta_theta` over one coupling step as
      `delta_theta_next = (1 - kappa * cos_delay_phase) * delta_theta`. -/
def delayed_kuramoto_effective_stiffness (kappa cos_delay_phase : Int) : Int :=
  kappa * cos_delay_phase

def delayed_phase_perturbation_step (kappa cos_delay_phase delta_theta : Int) : Int :=
  (1 - delayed_kuramoto_effective_stiffness kappa cos_delay_phase) * delta_theta

/-- Theorem 2 (`morelli_2009_in_phase_stability_vs_antiphase_instability`):
    (i) In the in-phase attractive delay window (`kappa = 1, cos_delay_phase = 1`,
        so `kappa_eff = 1 > 0`), any phase perturbation `delta_theta` contracts
        to zero in one coupling step;
    (ii) In the repulsive anti-phase delay window (`kappa = 1, cos_delay_phase = -1`,
         so `kappa_eff = -1 < 0`), the perturbation doubles (`2 * delta_theta`),
         destabilizing in-phase synchrony. -/
theorem morelli_2009_in_phase_stability_vs_antiphase_instability (delta_theta : Int) :
    delayed_phase_perturbation_step 1 1 delta_theta = 0 ∧
    delayed_phase_perturbation_step 1 (-1) delta_theta = 2 * delta_theta := by
  constructor
  · dsimp [delayed_phase_perturbation_step, delayed_kuramoto_effective_stiffness]
    omega
  · dsimp [delayed_phase_perturbation_step, delayed_kuramoto_effective_stiffness]

/-- Law 3 (`riedel-delta-notch-phase-synchrony-3`):
    Riedel-Kruse et al. (2007) Anterior Limit of Defects (ALD) & Posterior Rescue
    Point (PRP) Pulse-Chase Law:
    When DAPT blocks Notch cleavage at somite stage `s_in` and is washed out at
    somite stage `s_out = s_in + dapt_duration`, defective somites appear after the
    desynchronization lag `Delta_n_ALD` (`ALD = s_in + Delta_n_ALD`) and coherent
    segmentation recovers after the resynchronization lag `Delta_n_PRP`
    (`PRP = s_out + Delta_n_PRP`). -/
def anterior_limit_of_defects (s_in Delta_n_ALD : Nat) : Nat :=
  s_in + Delta_n_ALD

def posterior_rescue_point (s_out Delta_n_PRP : Nat) : Nat :=
  s_out + Delta_n_PRP

/-- Theorem 3 (`riedel_2007_ald_prp_pulse_chase_window_invariance`):
    (i) Shifting the DAPT pulse onset `s_in` posteriorly by `delta_s` shifts
        `ALD` by exactly `delta_s` with unit slope;
    (ii) When the desynchronization lag and resynchronization lag match
         (`Delta_n_ALD = Delta_n_PRP = Delta_lag`), the defective somite span
         `PRP - ALD` equals the DAPT exposure duration `dapt_duration`. -/
theorem riedel_2007_ald_prp_pulse_chase_window_invariance (s_in delta_s dapt_duration Delta_lag : Nat) :
    anterior_limit_of_defects (s_in + delta_s) Delta_lag =
      anterior_limit_of_defects s_in Delta_lag + delta_s ∧
    posterior_rescue_point (s_in + dapt_duration) Delta_lag -
      anterior_limit_of_defects s_in Delta_lag = dapt_duration := by
  constructor
  · dsimp [anterior_limit_of_defects]
    omega
  · dsimp [posterior_rescue_point, anterior_limit_of_defects]
    omega

/-! ### Pass 3 laws (2026-10-19): kinematic phase waves, Mib1 ubiquitination gate, emergent defect span -/

/-- Law 4 (`riedel-delta-notch-phase-synchrony-4`):
    Morelli et al. (2009) Kinematic Phase-Wave Law for an advected frequency profile.
    In the frame co-moving with the tailbud a cell at position `x` oscillates at the local
    effective frequency `omega_eff x` while the pattern oscillates at the collective frequency
    `Omega`; over one advection step of duration `dt` the cell accumulates the phase lag
      `delta_phi = (Omega - omega_eff) * dt`,
    so the local wave number is `k = (Omega - omega_eff) / v` with `v` the advection speed. -/
def kinematic_phase_lag (Omega omega_eff dt : Int) : Int :=
  (Omega - omega_eff) * dt

/-- Theorem 4 (`morelli_2009_kinematic_wave_phase_lag_law`):
    (i) A uniform frequency profile (`omega_eff = Omega`) carries no kinematic wave
        (zero phase lag per step, `k = 0`);
    (ii) for a non-negative step `dt`, a slower cell (`omega_1 <= omega_2`) accumulates at least
         as much lag as a faster one, so the lag is monotone decreasing in the local frequency:
         the anterior (FGF8/Wnt-slowed) PSM lags the tailbud and waves travel tail -> front;
    (iii) the phase lag accumulated over two successive advection steps is additive, which is the
          discrete form of `Phi(x) = Int k dx` and of `N_w = (1 / 2 pi) * Int k dx`. -/
theorem morelli_2009_kinematic_wave_phase_lag_law
    (Omega omega_1 omega_2 dt dt_1 dt_2 : Int) (h_omega : omega_1 ≤ omega_2) (h_dt : 0 ≤ dt) :
    kinematic_phase_lag Omega Omega dt = 0 ∧
    kinematic_phase_lag Omega omega_2 dt ≤ kinematic_phase_lag Omega omega_1 dt ∧
    kinematic_phase_lag Omega omega_1 dt_1 + kinematic_phase_lag Omega omega_1 dt_2 =
      kinematic_phase_lag Omega omega_1 (dt_1 + dt_2) := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [kinematic_phase_lag]
    omega
  · dsimp [kinematic_phase_lag]
    apply Int.mul_le_mul_of_nonneg_right _ h_dt
    omega
  · dsimp [kinematic_phase_lag]
    rw [Int.mul_add]

/-- Law 5 (`riedel-delta-notch-phase-synchrony-5`):
    Itoh et al. (2003) Mind bomb (Mib1) DeltaC ubiquitination gate. The steady-state fraction of
    Mib1-ubiquitinated, signalling-competent DeltaC is
      `u_ss = k_ub * M / (k_ub * M + k_endo)`,
    represented here by its numerator and denominator in `Nat` so that fraction comparisons are
    cross-multiplications. The coupling is `kappa = kappa_max * u_ss * g`. -/
def mib1_ubiquitinated_numerator (k_ub M : Nat) : Nat :=
  k_ub * M

def mib1_ubiquitinated_denominator (k_ub k_endo M : Nat) : Nat :=
  k_ub * M + k_endo

/-- Theorem 5 (`itoh_2003_mib1_ubiquitination_gate_monotone`):
    (i) In the *mib* null (`M = 0`) no DeltaC is ubiquitinated (`u_ss = 0`), so the Delta-Notch
        coupling vanishes exactly as under saturating DAPT;
    (ii) `u_ss <= 1` (numerator <= denominator), so `kappa(t)` never exceeds `kappa_max`;
    (iii) `u_ss` is monotone non-decreasing in the Mib1 activity `M` (`M_1 <= M_2` implies
          `u_ss(M_1) <= u_ss(M_2)`, stated as the cross-multiplied inequality
          `num(M_1) * den(M_2) <= num(M_2) * den(M_1)`). -/
theorem itoh_2003_mib1_ubiquitination_gate_monotone
    (k_ub k_endo M_1 M_2 : Nat) (h : M_1 ≤ M_2) :
    mib1_ubiquitinated_numerator k_ub 0 = 0 ∧
    mib1_ubiquitinated_numerator k_ub M_1 ≤ mib1_ubiquitinated_denominator k_ub k_endo M_1 ∧
    mib1_ubiquitinated_numerator k_ub M_1 * mib1_ubiquitinated_denominator k_ub k_endo M_2 ≤
      mib1_ubiquitinated_numerator k_ub M_2 * mib1_ubiquitinated_denominator k_ub k_endo M_1 := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [mib1_ubiquitinated_numerator]
  · dsimp [mib1_ubiquitinated_numerator, mib1_ubiquitinated_denominator]
    exact Nat.le_add_right _ _
  · dsimp [mib1_ubiquitinated_numerator, mib1_ubiquitinated_denominator]
    have h1 : k_ub * M_1 * k_endo ≤ k_ub * M_2 * k_endo :=
      Nat.mul_le_mul_right k_endo (Nat.mul_le_mul_left k_ub h)
    have e1 : k_ub * M_1 * (k_ub * M_2 + k_endo) = k_ub * M_1 * (k_ub * M_2) + k_ub * M_1 * k_endo :=
      Nat.mul_add _ _ _
    have e2 : k_ub * M_2 * (k_ub * M_1 + k_endo) = k_ub * M_2 * (k_ub * M_1) + k_ub * M_2 * k_endo :=
      Nat.mul_add _ _ _
    have e3 : k_ub * M_1 * (k_ub * M_2) = k_ub * M_2 * (k_ub * M_1) := by
      rw [Nat.mul_comm (k_ub * M_1) (k_ub * M_2)]
    rw [e1, e2, e3]
    exact Nat.add_le_add_left h1 _

/-- Law 6 (`riedel-delta-notch-phase-synchrony-6`):
    Emergent ALD / PRP read-out. `s_cross` (`s_recross`) is the somite stage at which the
    determination-front lateral order parameter first falls below (recovers above) `R_crit`;
    the morphological boundary index follows after the determination-to-boundary lag
    `Delta_n_det`. -/
def emergent_anterior_limit_of_defects (s_cross Delta_n_det : Nat) : Nat :=
  s_cross + Delta_n_det

def emergent_posterior_rescue_point (s_recross Delta_n_det : Nat) : Nat :=
  s_recross + Delta_n_det

/-- Theorem 6 (`riedel_2007_emergent_defect_span_independent_of_determination_lag`):
    (i) The emergent defective span `PRP - ALD` equals the coherence-loss span
        `s_recross - s_cross`, i.e. it is independent of the maturation lag `Delta_n_det`;
    (ii) the emergent ALD is never anterior to the coherence-loss stage (`s_cross <= ALD`);
    (iii) changing the lag by `d` shifts ALD and PRP by exactly `d` each. -/
theorem riedel_2007_emergent_defect_span_independent_of_determination_lag
    (s_cross s_recross Delta_n_det d : Nat) (h : s_cross ≤ s_recross) :
    emergent_posterior_rescue_point s_recross Delta_n_det -
      emergent_anterior_limit_of_defects s_cross Delta_n_det = s_recross - s_cross ∧
    s_cross ≤ emergent_anterior_limit_of_defects s_cross Delta_n_det ∧
    emergent_anterior_limit_of_defects s_cross (Delta_n_det + d) =
      emergent_anterior_limit_of_defects s_cross Delta_n_det + d ∧
    emergent_posterior_rescue_point s_recross (Delta_n_det + d) =
      emergent_posterior_rescue_point s_recross Delta_n_det + d := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · dsimp [emergent_posterior_rescue_point, emergent_anterior_limit_of_defects]
    omega
  · dsimp [emergent_anterior_limit_of_defects]
    omega
  · dsimp [emergent_anterior_limit_of_defects]
    omega
  · dsimp [emergent_posterior_rescue_point]
    omega

end MathOfBiology.Loop_023
