/-
  morphoeval/lean/loops/Loop_024_fgf8_wavefront_somitogenesis.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 024
  (`dubrulle-fgf8-wavefront-somitogenesis`):
    1. Dubrulle, J., McGrew, M. J. & Pourquie, O. (2001). "FGF signaling controls
       somite boundary position and regulates segmentation clock control of
       spatiotemporal Hox gene activation." Cell 106(2):219-232.
       DOI: 10.1016/S0092-8674(01)00437-8
    2. Dubrulle, J. & Pourquie, O. (2004). "fgf8 mRNA decay establishes a gradient
       that couples axial elongation to patterning in the vertebrate embryo."
       Nature 427(6973):419-422.
       DOI: 10.1038/nature02332

  All theorems are proven in pure core Lean 4 with zero external dependencies
  and zero `sorry`s.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_024_fgf8_wavefront_somitogenesis.lean
-/

namespace MathOfBiology.Loop_024

/-- Law 1 (`dubrulle-fgf8-wavefront-somitogenesis-1`):
    Dubrulle & Pourquie (2004) Cell-Autonomous `fgf8` mRNA Decay Advection-Reaction
    Characteristic Balance:
    Along a tailbud-derived PSM cell trajectory `x(t) = v_g * t`, the steady-state
    spatial advection flux `v_g * dm_dx` balances intracellular mRNA decay `-k_d * m`.
    Equivalently, the characteristic decay length `lambda_m = v_g / k_d` scales
    linearly with tailbud elongation speed `v_g`. -/
def fgf8_advection_decay_residual (v_g dm_dx k_d m : Int) : Int :=
  v_g * dm_dx + k_d * m

/-- Theorem 1 (`dubrulle_2004_advection_decay_characteristic_balance`):
    When spatial gradient slope satisfies `dm_dx = -k_d * c` and concentration is
    `m = v_g * c` along the characteristic `lambda_m = v_g / k_d`, the steady-state
    advection-decay PDE residual `v_g * dm_dx + k_d * m` vanishes identically. -/
theorem dubrulle_2004_advection_decay_characteristic_balance (v_g k_d c : Int) :
    fgf8_advection_decay_residual v_g (-(k_d * c)) k_d (v_g * c) = 0 := by
  dsimp [fgf8_advection_decay_residual]
  rw [Int.mul_neg, ← Int.mul_assoc v_g k_d c, ← Int.mul_assoc k_d v_g c, Int.mul_comm v_g k_d]
  omega

/-- Law 2 (`dubrulle-fgf8-wavefront-somitogenesis-2`):
    Dubrulle et al. (2001, 2004) Determination Front Positioning Law:
    The distance `x_star` from the posterior tailbud to the determination front
    (where `m(x_star) = m_crit`) is `x_star = lambda_m * L_log`, where `lambda_m`
    is the characteristic `fgf8` decay length `v_g / k_d` and `L_log = ln(m_0 / m_crit)`
    is the logarithmic transcription-to-threshold ratio. -/
def determination_front_position (lambda_m L_log : Nat) : Nat :=
  lambda_m * L_log

/-- Theorem 2 (`dubrulle_2001_determination_front_scaling`):
    Scaling tailbud elongation velocity (and hence `lambda_m`) by factor `k` scales
    the determination front position `x_star` by `k`, and increasing tailbud `fgf8`
    overexpression log-ratio by `delta_L` shifts `x_star` anteriorly (away from the
    tailbud) by `lambda_m * delta_L`. -/
theorem dubrulle_2001_determination_front_scaling (k lambda_m L_log delta_L : Nat) :
    determination_front_position (k * lambda_m) L_log =
      k * determination_front_position lambda_m L_log ∧
    determination_front_position lambda_m (L_log + delta_L) =
      determination_front_position lambda_m L_log + lambda_m * delta_L := by
  constructor
  · simp [determination_front_position, Nat.mul_assoc]
  · simp [determination_front_position, Nat.left_distrib]

/-- Law 3 (`dubrulle-fgf8-wavefront-somitogenesis-3`):
    Dubrulle et al. (2001) Ectopic FGF8 Bead / SU5402 Somite Size Perturbation &
    Conservation Law:
    A transient anterior shift `+delta_x` of the determination front (induced by an
    ectopic FGF8-soaked bead) produces a compressed anterior somite
    `S_ant = v_g * T_c - delta_x`, followed by a compensatory enlarged posterior
    somite `S_post = v_g * T_c + delta_x` when the wavefront catches up behind the
    bead, conserving total two-cycle segmented length `S_ant + S_post = 2 * v_g * T_c`. -/
def perturbed_somite_length (v_g T_c delta_x : Int) : Int :=
  v_g * T_c - delta_x

/-- Theorem 3 (`dubrulle_2001_fgf8_bead_somite_pair_conservation`):
    The sum of the compressed anterior somite `perturbed_somite_length v_g T_c delta_x`
    and the compensatory enlarged posterior somite `perturbed_somite_length v_g T_c (-delta_x)`
    around an ectopic FGF8 bead equals exactly twice the wild-type somite length
    `2 * (v_g * T_c)` for any wavefront shift `delta_x`. -/
theorem dubrulle_2001_fgf8_bead_somite_pair_conservation (v_g T_c delta_x : Int) :
    perturbed_somite_length v_g T_c delta_x +
      perturbed_somite_length v_g T_c (-delta_x) = 2 * (v_g * T_c) := by
  dsimp [perturbed_somite_length]
  omega

end MathOfBiology.Loop_024
