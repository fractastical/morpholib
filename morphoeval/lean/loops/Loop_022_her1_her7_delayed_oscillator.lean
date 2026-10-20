/-
  morphoeval/lean/loops/Loop_022_her1_her7_delayed_oscillator.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 022
  (`lewis-her1-her7-delayed-oscillator`):
    1. Lewis, J. (2003). "Autoinhibition with transcriptional delay: a simple
       mechanism for the zebrafish somitogenesis oscillator."
       Current Biology 13(16):1398-1408.
       DOI: 10.1016/S0960-9822(03)00534-7
    2. Giudicelli, F., Ozbudak, E. M., Wright, G. J. & Lewis, J. (2007).
       "Setting the tempo in development: an investigation of the zebrafish
       somite clock mechanism." PLoS Biology 5(6):e150.
       DOI: 10.1371/journal.pbio.0050150
    3. Herrgen, L., Ares, S., Morelli, L. G., Schroter, C., Julicher, F. &
       Oates, A. C. (2010). "Intercellular coupling regulates the period of the
       segmentation clock." Current Biology 20(14):1244-1253.
       DOI: 10.1016/j.cub.2010.06.034

  All theorems are proven in pure core Lean 4 with zero external dependencies
  and zero `sorry`s.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_022_her1_her7_delayed_oscillator.lean
-/

namespace MathOfBiology.Loop_022

/-- Law 1 (`lewis-her1-her7-delayed-oscillator-1`):
    Lewis (2003) Autoinhibitory Delay-Loop Invariance & Protein Translation Balance:
    In the single-cell *her1*/*her7* delay differential equations
      `dp/dt = a * m(t - T_p) - b * p(t)`
      `dm/dt = f(p(t - T_m)) - c * m(t)`,
    at steady state (or under cycloheximide translation block `a = 0`), the protein
    drift residual `a * m_delayed - b * p` vanishes when `p = a * c_eq` and
    `m_delayed = b * c_eq`, and the total feedback loop delay `T_tot = T_m + T_p`
    is invariant under redistributing delay `delta` between transcriptional delay
    `T_m` and translational delay `T_p`. -/
def lewis_protein_dde_residual (a m_delayed b p : Int) : Int :=
  a * m_delayed - b * p

def lewis_total_loop_delay (T_m T_p : Int) : Int :=
  T_m + T_p

/-- Theorem 1 (`lewis_2003_protein_steady_state_and_delay_invariance`):
    (i) Along the steady-state proportionality `m_delayed = b * q` and `p = a * q`,
        the Her1/Her7 protein DDE drift `a * m_delayed - b * p` vanishes identically;
    (ii) Redistributing delay `delta` from transcription (`T_m + delta`) to
         translation (`T_p - delta`) preserves the total feedback loop delay
         `lewis_total_loop_delay (T_m + delta) (T_p - delta) = lewis_total_loop_delay T_m T_p`. -/
theorem lewis_2003_protein_steady_state_and_delay_invariance (a b q T_m T_p delta : Int) :
    lewis_protein_dde_residual a (b * q) b (a * q) = 0 ∧
    lewis_total_loop_delay (T_m + delta) (T_p - delta) = lewis_total_loop_delay T_m T_p := by
  constructor
  · dsimp [lewis_protein_dde_residual]
    rw [← Int.mul_assoc a b q, ← Int.mul_assoc b a q, Int.mul_comm a b]
    omega
  · dsimp [lewis_total_loop_delay]
    omega

/-- Law 2 (`lewis-her1-her7-delayed-oscillator-2`):
    Lewis (2003) Box 1 Delay-Driven Oscillation Period Law:
    Above the supercritical Hopf bifurcation (`T_m + T_p > T_crit`), the oscillation
    period `T_osc` of the *her1*/*her7* autoinhibitory loop is twice the sum of the
    transcriptional delay `T_m`, translational delay `T_p`, and mRNA/protein decay
    relaxation times `tau_m + tau_p`:
      `T_osc = 2 * (T_m + T_p + tau_m + tau_p)`. -/
def lewis_oscillation_period (T_m T_p tau_m tau_p : Nat) : Nat :=
  2 * (T_m + T_p + tau_m + tau_p)

/-- Theorem 2 (`lewis_2003_oscillation_period_delay_sensitivity`):
    Increasing transcriptional splicing/export delay `T_m` by `delta_Tm` (or
    intercellular Notch coupling delay) increases the segmentation clock period
    `lewis_oscillation_period` by exactly `2 * delta_Tm`, and reducing intron delay
    by `delta_intron ≤ T_m` shortens the period by `2 * delta_intron`. -/
theorem lewis_2003_oscillation_period_delay_sensitivity (T_m T_p tau_m tau_p delta_Tm : Nat) :
    lewis_oscillation_period (T_m + delta_Tm) T_p tau_m tau_p =
      lewis_oscillation_period T_m T_p tau_m tau_p + 2 * delta_Tm := by
  dsimp [lewis_oscillation_period]
  omega

/-- Law 3 (`lewis-her1-her7-delayed-oscillator-3`):
    Lewis (2003) DeltaC-Notch Intercellular Phase-Locking Restoring Force vs
    *bea*/*des*/*aei* Uncoupled Phase Drift:
    For two neighboring PSM oscillators with phase difference `Delta_phi` and
    intrinsic frequency mismatch `delta_omega`, intercellular DeltaC-Notch coupling
    with strength `kappa` evolves the linearized phase difference over step `dt` as
    `Delta_phi_next = (1 - kappa) * Delta_phi + delta_omega`. -/
def notch_phase_difference_step (kappa Delta_phi delta_omega : Int) : Int :=
  (1 - kappa) * Delta_phi + delta_omega

/-- Theorem 3 (`lewis_2003_notch_coupling_synchronization_contraction`):
    When identical neighboring PSM oscillators (`delta_omega = 0`) experience
    unit normalizing DeltaC-Notch coupling (`kappa = 1`), any initial phase
    perturbation `Delta_phi` contracts to zero in one coupling timescale, whereas
    in Notch pathway mutants (`kappa = 0`, *beamter* / *deadly seven* / *after eight*),
    phase error accumulates linearly with frequency noise `Delta_phi + delta_omega`. -/
theorem lewis_2003_notch_coupling_synchronization_contraction (Delta_phi delta_omega : Int) :
    notch_phase_difference_step 1 Delta_phi 0 = 0 ∧
    notch_phase_difference_step 0 Delta_phi delta_omega = Delta_phi + delta_omega := by
  constructor
  · dsimp [notch_phase_difference_step]
    omega
  · dsimp [notch_phase_difference_step]
    omega

/-! ### Pass 3 additions: 2D DDE lattice with DeltaC/NICD kinetics and the
    Schröter et al. (2012) Her1/Her7/Hes6 dimer truth table.
    4. Schröter, C., Ares, S., Morelli, L. G., Isakova, A., Koo, B.-K., Jülicher, F. &
       Oates, A. C. (2012). "Topology and dynamics of the zebrafish segmentation clock
       core circuit." PLoS Biology 10(7):e1001364. DOI: 10.1371/journal.pbio.1001364 -/

/-- Law 4a (`lewis-her1-her7-delayed-oscillator-4`):
    Notch-gated her transcription factor `G_N = (1 - eps_N) + eps_N * s`, where `s` is the
    saturating NICD occupancy `N^h / (N_0^h + N^h)` (scaled to integers here) and `eps_N`
    is the fraction of her transcription that depends on Notch signalling. -/
def notch_transcription_gate (eps_N s : Int) : Int :=
  (1 - eps_N) + eps_N * s

/-- Theorem 4a (`lewis_2003_notch_gate_mutant_independence`):
    In Notch pathway mutants (*bea*/*des*/*aei*, `eps_N = 0`) her transcription is
    independent of the NICD signal `s` (gate identically 1), while under full Notch
    dependence (`eps_N = 1`) the gate equals the NICD occupancy itself. -/
theorem lewis_2003_notch_gate_mutant_independence (s : Int) :
    notch_transcription_gate 0 s = 1 ∧ notch_transcription_gate 1 s = s := by
  constructor
  · dsimp [notch_transcription_gate]
    omega
  · dsimp [notch_transcription_gate]
    omega

/-- Law 4b: DeltaC presented to a cell is the mean over its 4 von Neumann neighbours
    `Dbar = (D_up + D_down + D_left + D_right) / 4`. -/
def deltac_neighbour_mean (D_up D_down D_left D_right : Nat) : Nat :=
  (D_up + D_down + D_left + D_right) / 4

/-- Theorem 4b (`deltac_lattice_uniform_neighbour_mean_fixed_point`):
    A spatially uniform (synchronised) DeltaC field is a fixed point of the lattice
    coupling: every cell sees exactly its own DeltaC level, so Notch coupling exerts no
    differential drive on an already synchronised sheet. -/
theorem deltac_lattice_uniform_neighbour_mean_fixed_point (D : Nat) :
    deltac_neighbour_mean D D D D = D := by
  dsimp [deltac_neighbour_mean]
  have h : D + D + D + D = 4 * D := by omega
  rw [h, Nat.mul_div_cancel_left D (by decide : 0 < 4)]

/-- Law 5a (`lewis-her1-her7-delayed-oscillator-5`):
    Total repressive dimer occupancy under the DNA-binding truth table `B_AB ∈ {0,1}`:
    `Rep = B_11 [Her1:Her1] + B_76 [Her7:Hes6] + B_17 [Her1:Her7] + B_16 [Her1:Hes6]
          + B_77 [Her7:Her7] + B_66 [Hes6:Hes6]`, with mass-action `[A:B] = A * B`
    (dissociation constants absorbed into the units). -/
def schroter_dimer_repression (B11 B76 B17 B16 B77 B66 p1 p7 h6 : Nat) : Nat :=
  B11 * (p1 * p1) + B76 * (p7 * h6) + B17 * (p1 * p7) + B16 * (p1 * h6) + B77 * (p7 * p7) + B66 * (h6 * h6)

/-- Theorem 5a (`schroter_2012_dimer_truth_table_repression`):
    Under the Schröter et al. (2012) truth table (Her1:Her1 and Her7:Hes6 bind DNA; the
    other four dimers do not) the repression functional reduces to
    `p1^2 + p7 * h6`, and in a *hes6*-null background (`h6 = 0`) the Her7 branch is silent:
    repression is carried by the Her1 homodimer alone (`p1^2`). -/
theorem schroter_2012_dimer_truth_table_repression (p1 p7 h6 : Nat) :
    schroter_dimer_repression 1 1 0 0 0 0 p1 p7 h6 = p1 * p1 + p7 * h6 ∧
    schroter_dimer_repression 1 1 0 0 0 0 p1 p7 0 = p1 * p1 := by
  constructor
  · dsimp [schroter_dimer_repression]
    omega
  · dsimp [schroter_dimer_repression]
    omega

/-- Law 5b: the effective loop delay of a cell whose repression is shared between the
    Her7:Hes6 fast branch (weight `w`, delay `T7`) and the Her1 homodimer slow branch
    (weight `v`, delay `T1`) is the convex combination `(w T7 + v T1) / (w + v)`;
    we state it in cleared-denominator form. -/
def hes6_weighted_loop_delay (w v T7 T1 : Nat) : Nat :=
  w * T7 + v * T1

/-- Theorem 5b (`schroter_2012_hes6_gradient_period_interpolation`):
    For `T7 ≤ T1`, the Hes6-weighted loop delay is bracketed by the pure her7 and pure her1
    delays: `(w + v) T7 ≤ w T7 + v T1 ≤ (w + v) T1`. Since the Box 1 period is monotone in the
    loop delay, the posterior (Hes6-high, `w` large) period is bounded below by the her7 branch
    and the anterior (Hes6-low) period is bounded above by the her1 branch — the AP period
    gradient lives inside `[T_her7, T_her1]`. -/
theorem schroter_2012_hes6_gradient_period_interpolation (w v T7 T1 : Nat) (h : T7 ≤ T1) :
    (w + v) * T7 ≤ hes6_weighted_loop_delay w v T7 T1 ∧
    hes6_weighted_loop_delay w v T7 T1 ≤ (w + v) * T1 := by
  constructor
  · dsimp [hes6_weighted_loop_delay]
    rw [Nat.add_mul]
    exact Nat.add_le_add_left (Nat.mul_le_mul_left v h) _
  · dsimp [hes6_weighted_loop_delay]
    rw [Nat.add_mul]
    exact Nat.add_le_add_right (Nat.mul_le_mul_left w h) _

end MathOfBiology.Loop_022
