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

end MathOfBiology.Loop_023
