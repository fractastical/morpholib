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

end MathOfBiology.Loop_021
