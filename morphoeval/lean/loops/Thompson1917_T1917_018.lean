/-
  morphoeval/lean/loops/Thompson1917_T1917_018.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 213
  (Thompson 1917 claim T1917-018, Ch. VII "The Forms of Tissues, or Cell-Aggregates",
  pp. 293-345, Figs. 100-135):

    "Cells in contact, like soap films, meet three-at-a-time at ~120 degrees and in
     other respects obey Plateau's laws of film configuration."

  The theorems below pin the algebraic skeleton of the claim that the script
  `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` tests on
  real epithelia:
    T1  Thompson's Fig. 113 tension cases (p. 303) in integer degrees;
    T2  Lami force balance at the 120-degree geometry holds iff the three tensions
        are equal (the converse Thompson uses on p. 297);
    T3  the per-vertex angle-sum identity forces the MEAN angle to be 120, so the
        empirical content of the claim is the dispersion (why the script scores
        D_v = RMS(theta - 120) and not the mean);
    T4  Euler / three-valent bookkeeping of a periodic froth (mean 6 sides, V = 2F)
        and the topological charge of a T1 pentagon-heptagon pair;
    T5  three regular n-gons close a vertex iff n = 6 (Plateau 120 degrees and the
        honeycomb are the same statement, pp. 319-321);
    T6  Plateau's four-fold-vertex instability, certified numerically at 1e-4 scale:
        the two-Y Steiner partition of a unit square (1 + sqrt 3) is shorter than the
        X partition (2 sqrt 2), pp. 309-310 (Lamarle 1864; Taylor 1976).

  Pure core Lean 4 (zero external imports, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Thompson1917_T1917_018.lean
-/

namespace MathOfBiology.Loop_213

/-! ## T1. Thompson (1917) Fig. 113 tension cases, p. 303 -/

/-- The three angles `alpha beta gamma` (integer degrees) at a planar tri-junction close
    the full turn: `alpha + beta + gamma = 360`. -/
def closes_full_turn (alpha beta gamma : Int) : Prop :=
  alpha + beta + gamma = 360

/-- Theorem 1 (`thompson_fig113_three_cases`):
    For any tri-junction closing the full turn,
    (1) if `alpha = beta` and the partition tension is negligible so that the two outer films
        are collinear (`gamma = 180`), then `alpha = beta = 90` (Thompson case (1));
    (2) if the three angles are all equal (`T = T' = t`), each is `120` (case (3));
    (3) if `alpha = beta` (`T = T'`), the opposite angle `gamma` is below `120` exactly when
        `alpha` and `beta` exceed `120` -- Thompson's case (2) "say, for instance, 100 degrees"
        for alpha, beta corresponds to `gamma = 160 > 120`. -/
theorem thompson_fig113_three_cases (alpha beta gamma : Int)
    (h : closes_full_turn alpha beta gamma) :
    (alpha = beta → gamma = 180 → alpha = 90 ∧ beta = 90) ∧
    (alpha = beta → beta = gamma → alpha = 120 ∧ beta = 120 ∧ gamma = 120) ∧
    (alpha = beta → (gamma < 120 ↔ 120 < alpha)) ∧
    (closes_full_turn 100 100 160) := by
  unfold closes_full_turn at h ⊢
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro hab hg
    omega
  · intro hab hbg
    omega
  · intro hab
    omega
  · decide

/-! ## T2. Lami balance at the 120-degree geometry (pp. 296-297) -/

/-- Horizontal component of the net tension at a vertex whose three films point along
    `0`, `120` and `240` degrees, written with the unit directions `(2, 0)`, `(-1, r)`,
    `(-1, -r)` scaled by two (`r` stands for `sqrt 3`, which never needs to be evaluated):
    `2 T1 - T2 - T3`. -/
def lami_balance_x (T1 T2 T3 : Int) : Int :=
  2 * T1 - T2 - T3

/-- Vertical component of the same net tension: `r * (T2 - T3)`. -/
def lami_balance_y (r T2 T3 : Int) : Int :=
  r * (T2 - T3)

/-- Theorem 2 (`lami_balance_iff_equal_tensions`):
    With a non-degenerate vertical unit (`r ≠ 0`), the three films meeting at `120` degrees
    are in force balance iff their tensions are equal: Thompson's "when three similar liquid
    surfaces meet together, they always do so at an angle of 120 degrees" (p. 297) and its
    converse, which is what makes a measured 120-degree junction evidence of equal tensions.
    In particular equal tensions `T` always balance. -/
theorem lami_balance_iff_equal_tensions (r T1 T2 T3 : Int) (hr : r ≠ 0) :
    (lami_balance_x T1 T2 T3 = 0 ∧ lami_balance_y r T2 T3 = 0) ↔ (T1 = T2 ∧ T2 = T3) := by
  unfold lami_balance_x lami_balance_y
  constructor
  · intro h
    have hy : r = 0 ∨ T2 - T3 = 0 := Int.mul_eq_zero.mp h.2
    have h23 : T2 - T3 = 0 := by
      cases hy with
      | inl h0 => exact absurd h0 hr
      | inr h0 => exact h0
    have hx := h.1
    omega
  · intro h
    refine ⟨by omega, ?_⟩
    have : T2 - T3 = 0 := by omega
    rw [this, Int.mul_zero]

theorem equal_tensions_balance (r T : Int) :
    lami_balance_x T T T = 0 ∧ lami_balance_y r T T = 0 := by
  unfold lami_balance_x lami_balance_y
  constructor
  · omega
  · rw [Int.sub_self, Int.mul_zero]

/-! ## T3. The angle-sum identity: the mean is 120 by construction -/

/-- Signed deviation of an angle from Plateau's `120` degrees. -/
def plateau_deviation (theta : Int) : Int :=
  theta - 120

/-- Theorem 3 (`tri_junction_mean_is_always_120`):
    At every planar tri-junction the three deviations from 120 sum to zero, i.e. the mean
    angle is 120 identically, whatever the tensions; and if two deviations vanish so does the
    third. Hence a test of Thompson's claim must score the DISPERSION (the script's
    `D_v = sqrt(mean (theta - 120)^2)` and `|theta - 120| <= 15` tolerance) rather than the
    mean, which can never fail. -/
theorem tri_junction_mean_is_always_120 (alpha beta gamma : Int)
    (h : closes_full_turn alpha beta gamma) :
    plateau_deviation alpha + plateau_deviation beta + plateau_deviation gamma = 0 ∧
    (plateau_deviation alpha = 0 → plateau_deviation beta = 0 → gamma = 120) ∧
    (plateau_deviation alpha = plateau_deviation beta →
      plateau_deviation gamma = -2 * plateau_deviation alpha) := by
  unfold closes_full_turn at h
  unfold plateau_deviation
  refine ⟨by omega, ?_, ?_⟩
  · intro ha hb
    omega
  · intro hab
    omega

/-! ## T4. Euler / three-valent bookkeeping of a froth (pp. 310, 319) -/

/-- Theorem 4 (`three_valent_froth_euler_bookkeeping`):
    In a periodic (toroidal) three-valent cellular network with `V` vertices, `E` edges and
    `F` cells (`V - E + F = 0`, `3 V = 2 E`): the total side count `2 E` equals `6 F` (mean
    polygon number six, Thompson's "hexagonal symmetry"), there are exactly two vertices per
    cell (`V = 2 F`), and a T1 exchange turning two hexagons into a pentagon-heptagon pair
    conserves total coordination (`5 + 7 = 6 + 6`), so three-fold meeting is preserved by
    the only topological move available to a froth. -/
theorem three_valent_froth_euler_bookkeeping (V E F : Int)
    (h_euler : V - E + F = 0) (h_trivalent : 3 * V = 2 * E) :
    2 * E = 6 * F ∧ V = 2 * F ∧ (5 + 7 : Int) = 6 + 6 := by
  refine ⟨by omega, by omega, by decide⟩

/-! ## T5. Three regular n-gons close a vertex iff n = 6 (pp. 319-321) -/

/-- `n` times the interior angle of a regular `n`-gon, in degrees: `(n - 2) * 180`. -/
def n_times_interior_angle (n : Int) : Int :=
  (n - 2) * 180

/-- Theorem 5 (`three_regular_polygons_close_vertex_iff_hexagon`):
    Three equal regular `n`-gons meeting at a vertex close the full turn,
    `3 * ((n - 2) * 180 / n) = 360`, i.e. `3 * (n - 2) * 180 = 360 * n`, iff `n = 6`;
    the hexagon's interior angle is `120` and the honeycomb is the unique regular
    realisation of Plateau's rule. Squares (`n = 4`) would need four per vertex
    (`4 * 90 = 360`), which Plateau's laws forbid for films. -/
theorem three_regular_polygons_close_vertex_iff_hexagon (n : Int) :
    (3 * n_times_interior_angle n = 360 * n ↔ n = 6) ∧
    n_times_interior_angle 6 = 6 * 120 ∧
    n_times_interior_angle 4 = 4 * 90 := by
  unfold n_times_interior_angle
  refine ⟨?_, by decide, by decide⟩
  constructor
  · intro h
    omega
  · intro h
    subst h
    decide

/-! ## T6. Plateau's four-fold vertex instability (pp. 309-310; Lamarle 1864; Taylor 1976) -/

/-- Integer surrogate for `10^4 * sqrt 3` (an upper rational bound). -/
def sqrt3_1e4_upper : Int := 17321

/-- Integer surrogate for `10^4 * sqrt 2` (a lower rational bound). -/
def sqrt2_1e4_lower : Int := 14142

/-- Theorem 6 (`steiner_two_y_beats_x_partition`):
    For a unit square, the four-fold "X" partition has total film length `2 sqrt 2`, while the
    two three-fold "Y" (Steiner) partition has length `1 + sqrt 3`. With certified rational
    brackets `sqrt 3 ≤ 1.7321` (`17321^2 ≥ 3 * 10^8`) and `sqrt 2 ≥ 1.4142`
    (`14142^2 ≤ 2 * 10^8`), the Steiner length is strictly shorter:
    `10^4 + 17321 < 2 * 14142`.  A four-fold film vertex therefore cannot be a minimiser and
    decays into two 120-degree vertices -- Plateau's second law, which the script checks as
    the `f3 >= 0.90` three-fold fraction. -/
theorem steiner_two_y_beats_x_partition :
    sqrt3_1e4_upper * sqrt3_1e4_upper ≥ 3 * 100000000 ∧
    sqrt2_1e4_lower * sqrt2_1e4_lower ≤ 2 * 100000000 ∧
    10000 + sqrt3_1e4_upper < 2 * sqrt2_1e4_lower := by
  unfold sqrt3_1e4_upper sqrt2_1e4_lower
  decide

/-- Supporting lemma (`square_of_sum_expansion`): the algebraic identity behind the
    squared comparison `(1 + sqrt 3)^2 = 4 + 2 sqrt 3 < 8 = (2 sqrt 2)^2`, i.e.
    `(k + r)^2 = k^2 + 2 k r + r^2`, in exact integer arithmetic. -/
theorem square_of_sum_expansion (k r : Int) :
    (k + r) * (k + r) = k * k + 2 * (k * r) + r * r := by
  have h1 : (k + r) * (k + r) = k * (k + r) + r * (k + r) := Int.add_mul k r (k + r)
  have h2 : k * (k + r) = k * k + k * r := Int.mul_add k k r
  have h3 : r * (k + r) = r * k + r * r := Int.mul_add r k r
  have h4 : r * k = k * r := Int.mul_comm r k
  rw [h1, h2, h3, h4]
  omega

end MathOfBiology.Loop_213
