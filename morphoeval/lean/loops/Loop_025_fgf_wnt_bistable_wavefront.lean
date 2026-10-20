/-
  morphoeval/lean/loops/Loop_025_fgf_wnt_bistable_wavefront.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 025
  (`baker-fgf-wnt-bistable-wavefront`):
    1. Goldbeter, A., Gonze, D. & Pourquie, O. (2007). "Sharp developmental
       thresholds defined through bistability by antagonistic gradients of
       retinoic acid and FGF signaling." Dev. Cell / Dev. Dyn. 236(6):1495-1508.
       DOI: 10.1016/j.devcel.2007.05.015 / 10.1002/dvdy.21193
    2. Baker, R. E., Schnell, S. & Maini, P. K. (2006). "A mathematical
       investigation of a clock and wavefront model for somitogenesis."
       Journal of Theoretical Biology 239(4):413-429.
       DOI: 10.1016/j.jtbi.2005.07.019

  All theorems are proven in pure core Lean 4 with zero external dependencies
  and zero `sorry`s.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_025_fgf_wnt_bistable_wavefront.lean
-/

namespace MathOfBiology.Loop_025

/-- Law 1 (`baker-fgf-wnt-bistable-wavefront-1`):
    Goldbeter, Gonze & Pourquie (2007) CYP26A1-Mediated Retinoic Acid (RA)
    Degradation Steady-State Balance:
    The local reaction rate for anterior Retinoic Acid `R` under RALDH2 synthesis
    `v_sc`, FGF8/Wnt-induced CYP26A1 enzymatic degradation `k_d1 * C_F * R`, and
    basal degradation `k_d2 * R` is `v_sc - (k_d1 * C_F + k_d2) * R`. -/
def ra_cyp26_kinetic_residual (v_sc k_d1 C_F k_d2 R : Int) : Int :=
  v_sc - (k_d1 * C_F + k_d2) * R

/-- Theorem 1 (`goldbeter_2007_ra_cyp26_steady_balance`):
    When RALDH2 synthesis matches total CYP26A1 + basal clearance
    `v_sc = (k_d1 * C_F + k_d2) * R`, the steady-state RA reaction residual
    vanishes identically, and elevating FGF-induced CYP26A1 activity by `delta_C`
    decreases the net RA accumulation rate by `-k_d1 * delta_C * R`. -/
theorem goldbeter_2007_ra_cyp26_steady_balance (k_d1 C_F delta_C k_d2 R : Int) :
    ra_cyp26_kinetic_residual ((k_d1 * C_F + k_d2) * R) k_d1 C_F k_d2 R = 0 ∧
    ra_cyp26_kinetic_residual ((k_d1 * C_F + k_d2) * R) k_d1 (C_F + delta_C) k_d2 R =
      -(k_d1 * delta_C * R) := by
  constructor
  · dsimp [ra_cyp26_kinetic_residual]
    omega
  · dsimp [ra_cyp26_kinetic_residual]
    rw [Int.mul_add k_d1 C_F delta_C]
    have h1 : (k_d1 * C_F + k_d1 * delta_C + k_d2) * R =
        (k_d1 * C_F + k_d2) * R + k_d1 * delta_C * R := by
      rw [Int.add_right_comm (k_d1 * C_F) (k_d1 * delta_C) k_d2]
      rw [Int.add_mul (k_d1 * C_F + k_d2) (k_d1 * delta_C) R]
    rw [h1]
    omega

/-- Law 2 (`baker-fgf-wnt-bistable-wavefront-2`):
    Goldbeter, Gonze & Pourquie (2007) Saddle-Node Bistable Hysteresis Window:
    Mutual antagonism between anterior RA (`R`) and posterior FGF8/Wnt (`F`)
    creates an S-shaped response curve `M(F)` bounded by lower saddle-node
    threshold `M_1` and upper saddle-node threshold `M_2`, with bistable window
    width `Delta_M = M_2 - M_1`. -/
def hysteresis_window_width (M_1 M_2 : Nat) : Nat :=
  M_2 - M_1

/-- Theorem 2 (`goldbeter_2007_hysteresis_window_width`):
    When mutual inhibition generates an upper saddle-node threshold
    `M_2 = M_1 + delta_M` with `delta_M > 0`, the bistable hysteresis width is
    strictly positive (`hysteresis_window_width M_1 (M_1 + delta_M) = delta_M > 0`),
    whereas in the CYP26-null monostable ablation (`M_2 = M_1`) the hysteresis
    width vanishes identically. -/
theorem goldbeter_2007_hysteresis_window_width (M_1 delta_M : Nat) (h : 0 < delta_M) :
    hysteresis_window_width M_1 (M_1 + delta_M) = delta_M ∧
    0 < hysteresis_window_width M_1 (M_1 + delta_M) ∧
    hysteresis_window_width M_1 M_1 = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [hysteresis_window_width]
    omega
  · dsimp [hysteresis_window_width]
    omega
  · dsimp [hysteresis_window_width]
    omega

/-- Law 3 (`baker-fgf-wnt-bistable-wavefront-3`):
    Baker, Schnell & Maini (2006) / Goldbeter et al. (2007) Opposing RA-FGF
    Determination Front Shift & Somite Size Symmetry Law:
    A wavefront shift `delta_x` over one segmentation clock period `T_c` yields
    somite size `S = v_g * T_c - delta_x`. Anterior shifts (`+delta_x > 0` under
    Vitamin A Deficiency VAD or ectopic FGF8) compress somites, whereas posterior
    shifts (`-delta_x < 0` under exogenous RA or SU5402) enlarge somites. -/
def bistable_somite_length (v_g T_c delta_x : Int) : Int :=
  v_g * T_c - delta_x

/-- Theorem 3 (`baker_2006_ra_fgf_opposing_somite_shift`):
    Symmetric opposing shifts `+delta_x` (VAD / FGF8 elevation) and `-delta_x`
    (exogenous RA / FGFR inhibition) of the bistable determination front preserve
    twice the wild-type somite length `2 * (v_g * T_c)` and produce a somite size
    splitting equal to `2 * delta_x`. -/
theorem baker_2006_ra_fgf_opposing_somite_shift (v_g T_c delta_x : Int) :
    bistable_somite_length v_g T_c delta_x +
      bistable_somite_length v_g T_c (-delta_x) = 2 * (v_g * T_c) ∧
    bistable_somite_length v_g T_c (-delta_x) -
      bistable_somite_length v_g T_c delta_x = 2 * delta_x := by
  constructor
  · dsimp [bistable_somite_length]
    omega
  · dsimp [bistable_somite_length]
    omega

/-- Law 4 (`baker-fgf-wnt-bistable-wavefront-4`, Pass 3):
    Lewis (2003) Delayed Autoinhibition Oscillator Period Law:
    A single-gene negative-feedback loop with total transcription + translation
    delay `T = T_m + T_p` and mRNA / protein half-life terms `d = 1/b + 1/c`
    oscillates with period `T_osc ~ 2 * (T + d)` (Lewis 2003, Current Biology
    13:1398-1408, Eq. 3). Intercellular Notch-Delta coupling with delay `T_N`
    phase-locks neighbouring PSM cells; in-phase locking (synchrony) occurs when
    `T_N` is commensurate with one clock period. -/
def lewis_delayed_oscillator_period (T d : Nat) : Nat :=
  2 * (T + d)

/-- Theorem 4 (`lewis_2003_delayed_oscillator_period`):
    The Lewis period is monotone in the feedback delay: increasing the delay by
    `delta_T` lengthens the period by exactly `2 * delta_T`; the period is always
    at least twice the feedback delay; and the chick preset
    `T = 34 min, d = 8 min` (T_m + T_p = 31.5 + 2.8 rounded; 1/b + 1/c = 2/0.23)
    yields `T_osc = 84 min` within the observed 90 min chick clock. -/
theorem lewis_2003_delayed_oscillator_period (T d delta_T : Nat) :
    lewis_delayed_oscillator_period (T + delta_T) d =
      lewis_delayed_oscillator_period T d + 2 * delta_T ∧
    2 * T ≤ lewis_delayed_oscillator_period T d ∧
    lewis_delayed_oscillator_period 34 8 = 84 := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [lewis_delayed_oscillator_period]
    omega
  · dsimp [lewis_delayed_oscillator_period]
    omega
  · rfl

/-- Law 5 (`baker-fgf-wnt-bistable-wavefront-5`, Pass 3):
    Goldbeter, Gonze & Pourquie (2007) / Lewis (2003) Saltatory Cohort Arrest
    Additivity: when the segmentation clock gates the bistable RA-FGF switch,
    PSM cells arrest in discrete cohorts of `n` cells, one cohort per clock
    cycle. With cell spacing `a`, each cohort spans a segment `S = n * a`, so
    `k` consecutive cohorts span `k * S` of paraxial mesoderm. -/
def cohort_axial_extent (n a k : Int) : Int :=
  k * (n * a)

/-- Theorem 5 (`goldbeter_2007_saltatory_cohort_additivity`):
    Adding one more clock cycle adds exactly one cohort segment `n * a`, and the
    Notch-ablated limit of single-cell "cohorts" (`n = 1`) reduces the per-cycle
    axial increment to one cell spacing `a` (continuous rather than saltatory
    arrest). -/
theorem goldbeter_2007_saltatory_cohort_additivity (n a k : Int) :
    cohort_axial_extent n a (k + 1) - cohort_axial_extent n a k = n * a ∧
    cohort_axial_extent 1 a (k + 1) - cohort_axial_extent 1 a k = a := by
  constructor
  · dsimp [cohort_axial_extent]
    rw [Int.add_mul k 1 (n * a), Int.one_mul]
    omega
  · dsimp [cohort_axial_extent]
    rw [Int.add_mul k 1 (1 * a), Int.one_mul, Int.one_mul]
    omega

/-- Law 6 (`baker-fgf-wnt-bistable-wavefront-6`, Pass 3):
    Vermot & Pourquie (2005) Retinoic Acid-Buffered Left-Right Symmetry of the
    Determination Front: a lateralizing bias `b` (from the embryonic LR cascade)
    acts on the two bilateral paraxial mesoderm strips as `+b` on one side and
    `-b` on the other; anterior RA buffers it to an effective bias `b_eff`.
    The left-right front offset is the difference of the two half-biases
    `(x_0 + b_eff) - (x_0 - b_eff) = 2 * b_eff` for the delayed vs. advanced
    strip; with full RA buffering (`b_eff = 0`) the fronts coincide, and in
    VAD (`b_eff > 0`) a one-somite left-right desynchronisation appears. -/
def lr_front_offset (x_0 b_eff : Int) : Int :=
  (x_0 + b_eff) - (x_0 - b_eff)

/-- Theorem 6 (`vermot_2005_ra_buffered_lr_symmetry`):
    Full RA buffering abolishes the left-right offset, an unbuffered bias
    produces an offset of `2 * b_eff` independent of the shared wild-type
    front position `x_0`, and situs inversion (`-b_eff`) exactly negates it. -/
theorem vermot_2005_ra_buffered_lr_symmetry (x_0 b_eff : Int) :
    lr_front_offset x_0 0 = 0 ∧
    lr_front_offset x_0 b_eff = 2 * b_eff ∧
    lr_front_offset x_0 (-b_eff) = -(lr_front_offset x_0 b_eff) := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [lr_front_offset]
    omega
  · dsimp [lr_front_offset]
    omega
  · dsimp [lr_front_offset]
    omega

end MathOfBiology.Loop_025
