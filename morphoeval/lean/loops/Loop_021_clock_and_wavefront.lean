/-
  morphoeval/lean/loops/Loop_021_clock_and_wavefront.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 021
  (`cooke-clock-and-wavefront`):
    Cooke, J. & Zeeman, E. C. (1976). "A clock and wavefront model for control
    of the number of repeated structures during animal morphogenesis."
    Journal of Theoretical Biology 58(2):455-476.
    DOI: 10.1016/S0022-5193(76)80131-2

  All theorems are proven in pure core Lean 4 with zero external dependencies
  and zero `sorry`s.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_021_clock_and_wavefront.lean
-/

namespace MathOfBiology.Loop_021

/-- Law 1 (`cooke-clock-and-wavefront-1`):
    Cooke & Zeeman (1976) Somite Wavelength Law:
    The anteroposterior length `S` of a single segmented somite equals the
    product of the wavefront regression speed `v` and the intracellular
    segmentation clock period `T_c`: `S = v * T_c`. -/
def somite_length (v T_c : Nat) : Nat :=
  v * T_c

/-- Total axial length `L(N, v, T_c)` spanned by `N` serially repeated somites
    formed at wavefront velocity `v` and clock period `T_c`. -/
def segmented_axis_length (N v T_c : Nat) : Nat :=
  N * somite_length v T_c

/-- Theorem 1 (`cooke_zeeman_1976_wavelength_period_scaling`):
    Doubling the intracellular segmentation clock period `2 * T_c` at fixed
    wavefront speed `v` doubles individual somite length `S(v, 2 * T_c) = 2 * S(v, T_c)`,
    and scaling wavefront velocity by `k` scales somite length by `k`. -/
theorem cooke_zeeman_1976_wavelength_period_scaling (k v T_c : Nat) :
    somite_length (k * v) T_c = k * somite_length v T_c ∧
    somite_length v (2 * T_c) = 2 * somite_length v T_c := by
  constructor
  · simp [somite_length, Nat.mul_assoc]
  · simp [somite_length, Nat.mul_left_comm]

/-- Law 2 (`cooke-clock-and-wavefront-2`):
    Cooke (1975) & Cooke & Zeeman (1976) Embryo Size-Invariance Scaling Theorem:
    When a surgically reduced miniature *Xenopus laevis* embryo scales its body
    axis length `L' = scale * L_0` by scaling the wavefront regression speed
    proportionally (`v' = scale * v_0`) over the same total developmental window
    `T_total = N * T_c`, the total segmented axis length satisfies
    `segmented_axis_length N (scale * v_0) T_c = scale * segmented_axis_length N v_0 T_c`,
    preserving the invariant somite count `N` across embryo sizes. -/
theorem cooke_1975_miniature_embryo_size_invariance (scale N v_0 T_c : Nat) :
    segmented_axis_length N (scale * v_0) T_c =
      scale * segmented_axis_length N v_0 T_c := by
  simp [segmented_axis_length, somite_length, Nat.mul_assoc, Nat.mul_left_comm]

/-- Law 3 (`cooke-clock-and-wavefront-3`):
    Thom-Zeeman Cusp Catastrophe Equilibrium Manifold (`chi^3 - a * chi - b = 0`):
    Prior to wavefront arrival (`a = -1`, symmetric unsegmented presomitic
    mesoderm with `b = 0`), the unique integer equilibrium is `chi = 0`; after
    wavefront passage (`a = 1`, `b = 0`), the pre-somitic state bifurcates into
    the symmetric differentiated attractor pair `chi = 1` and `chi = -1`. -/
def cusp_catastrophe_residual (chi a b : Int) : Int :=
  chi * chi * chi - a * chi - b

theorem zeeman_1976_cusp_wavefront_bifurcation :
    cusp_catastrophe_residual 0 (-1) 0 = 0 ∧
    cusp_catastrophe_residual 1 1 0 = 0 ∧
    cusp_catastrophe_residual (-1) 1 0 = 0 := by
  decide

/-! ## Pass 3 (2026-10-19) — endogenous FGF8/RA determination front and the
    anterior/posterior half-somite cleft rule coupled to the 2D vertex sheet. -/

/-- Law 4 (`cooke-clock-and-wavefront-4`):
    Goldbeter, Gonze & Pourquie (2007) FGF8/RA bistable determination front driven
    by tailbud elongation (Dubrulle & Pourquie 2004 fgf8 mRNA decay gradient).
    The tailbud sits at `x_tb(t) = x_tb0 + v_tb * t`; fgf8 mRNA decays with the
    distance from the tailbud, so the saddle-node threshold `M_F = M_lo` of the
    FGF/RA toggle is crossed at a fixed offset `D = lambda_F * ln(1 / M_lo)`
    behind the tailbud. The front position is therefore
    `x_w(t) = x_tb0 + v_tb * t - D`. -/
def tailbud_position (x_tb0 v_tb t : Int) : Int :=
  x_tb0 + v_tb * t

def determination_front (x_tb0 v_tb D t : Int) : Int :=
  tailbud_position x_tb0 v_tb t - D

/-- Theorem 4 (`goldbeter_2007_front_velocity_equals_tailbud_elongation`):
    For ANY bistability offset `D` (i.e. any toggle parameters, mRNA decay length
    or threshold), the determination front advances by exactly `v_tb * (t₂ - t₁)`
    between two times, so its velocity equals the tailbud elongation speed and is
    independent of the toggle's internal parameters; consequently the somite
    length of Law 1 becomes `S = v_tb * T_c` with no prescribed wavefront speed. -/
theorem goldbeter_2007_front_velocity_equals_tailbud_elongation
    (x_tb0 v_tb D t₁ t₂ : Int) :
    determination_front x_tb0 v_tb D t₂ - determination_front x_tb0 v_tb D t₁ =
      v_tb * (t₂ - t₁) := by
  unfold determination_front tailbud_position
  rw [Int.mul_sub]
  omega

/-- Corollary: the somite length produced by an endogenous front over one clock
    period `T_c` (front displacement between `t` and `t + T_c`) equals Law 1's
    `somite_length v_tb T_c` once the integers are read as naturals. -/
theorem endogenous_front_somite_length (x_tb0 v_tb D t T_c : Int) :
    determination_front x_tb0 v_tb D (t + T_c) - determination_front x_tb0 v_tb D t =
      v_tb * T_c := by
  rw [goldbeter_2007_front_velocity_equals_tailbud_elongation]
  have h : t + T_c - t = T_c := by omega
  rw [h]

/-- Law 5 (`cooke-clock-and-wavefront-5`):
    Anterior/posterior half-somite compartments (Meinhardt 1986; Saga & Takeda 2001).
    The cusp of Law 3 alternates `+1` (anterior half-somite) and `-1` (posterior
    half-somite) along the axis. `compartments N` is the resulting sequence of `2N`
    half-somite states for `N` somites. -/
def compartments : Nat → List Int
  | 0 => []
  | n + 1 => 1 :: (-1) :: compartments n

/-- Count posterior(-1) -> anterior(+1) interfaces (inter-somitic clefts) in a
    sequence read anterior to posterior. -/
def count_clefts : List Int → Nat
  | (-1) :: 1 :: rest => 1 + count_clefts (1 :: rest)
  | _ :: rest => count_clefts rest
  | [] => 0

/-- Count anterior(+1) -> posterior(-1) interfaces (intra-somitic compartment
    boundaries, which do NOT form clefts). -/
def count_intra : List Int → Nat
  | 1 :: (-1) :: rest => 1 + count_intra ((-1) :: rest)
  | _ :: rest => count_intra rest
  | [] => 0

theorem compartments_length (N : Nat) : (compartments N).length = 2 * N := by
  induction N with
  | zero => rfl
  | succ n ih => simp [compartments, ih]; omega

theorem count_clefts_cons_neg_compartments (n : Nat) :
    count_clefts ((-1) :: compartments (n + 1)) = 1 + count_clefts (compartments (n + 1)) := by
  rfl

theorem count_clefts_compartments_zero : count_clefts (compartments 0) = 0 := rfl

/-- Theorem 5 (`meinhardt_1986_half_somite_cleft_count`):
    `N + 1` somites' worth of alternating half-somite compartments contain exactly
    `N` inter-somitic clefts (posterior -> anterior interfaces) and `N + 1`
    intra-somitic compartment boundaries, so counting only the `-1 -> +1`
    interfaces (as the vertex sheet's differential-tension rule does) recovers
    the somite number `N + 1 = clefts + 1` while every `+1 -> -1` interface is
    correctly left tension-free. -/
theorem meinhardt_1986_half_somite_cleft_count (N : Nat) :
    count_clefts (compartments (N + 1)) = N ∧
    count_intra (compartments (N + 1)) = N + 1 := by
  induction N with
  | zero => constructor <;> rfl
  | succ n ih =>
    obtain ⟨hc, hi⟩ := ih
    constructor
    · show count_clefts (1 :: (-1) :: compartments (n + 1)) = n + 1
      have h1 : count_clefts (1 :: (-1) :: compartments (n + 1)) =
          count_clefts ((-1) :: compartments (n + 1)) := rfl
      rw [h1, count_clefts_cons_neg_compartments, hc]
      omega
    · show count_intra (1 :: (-1) :: compartments (n + 1)) = n + 1 + 1
      have h1 : count_intra (1 :: (-1) :: compartments (n + 1)) =
          1 + count_intra ((-1) :: compartments (n + 1)) := rfl
      have h2 : count_intra ((-1) :: compartments (n + 1)) =
          count_intra (compartments (n + 1)) := rfl
      rw [h1, h2, hi]
      omega

/-- Law 5b: Farhadifar et al. (2007) vertex-model line-tension rule used on the
    sheet: an edge carries the baseline tension `Lambda_0` unless its anterior cell
    is a posterior half-somite (`-1`) and its posterior cell an anterior half-somite
    (`+1`), in which case it carries `Lambda_0 + dLambda`. -/
def edge_tension (Lambda_0 dLambda : Nat) (ant post : Int) : Nat :=
  if ant = -1 ∧ post = 1 then Lambda_0 + dLambda else Lambda_0

/-- Theorem 5b (`farhadifar_2007_cleft_tension_selectivity`): only the
    posterior -> anterior interface is loaded; the intra-somitic (+1 -> -1)
    interface, same-state interfaces and uncommitted (0) interfaces all keep
    the baseline tension. -/
theorem farhadifar_2007_cleft_tension_selectivity (Lambda_0 dLambda : Nat) :
    edge_tension Lambda_0 dLambda (-1) 1 = Lambda_0 + dLambda ∧
    edge_tension Lambda_0 dLambda 1 (-1) = Lambda_0 ∧
    edge_tension Lambda_0 dLambda 1 1 = Lambda_0 ∧
    edge_tension Lambda_0 dLambda (-1) (-1) = Lambda_0 ∧
    edge_tension Lambda_0 dLambda 0 1 = Lambda_0 := by
  simp [edge_tension]

end MathOfBiology.Loop_021
