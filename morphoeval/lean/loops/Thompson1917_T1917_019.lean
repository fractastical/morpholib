/-
  morphoeval/lean/loops/Thompson1917_T1917_019.lean
  Formal Lean 4 verification of the topological and isoperimetric laws behind
  D'Arcy Thompson (1917) On Growth and Form, Ch. VII "The Forms of Tissues, or
  Cell-Aggregates", pp. 293-345, Figs. 120-127 — claim T1917-019:
  "Tissue cells tessellate approximately as hexagons / as a froth, the
  configuration that partitions area with least perimeter."
  (`thompson1917-t1917-019`, Slot 214).

  Six laws are formalised, each tied to an equation E1-E6 of
  datasets/thompson_1917/t1917-019/t1917_019_the_forms_of_tissues_or_cell_a_reference.json:
    Law 1  Euler trivalent-froth side sum        sum n_i = 6F - 6 chi   (Thompson p. 325)
    Law 2  Second moment mu2 and the honeycomb   mu2 = 0  <->  all hexagons
    Law 3  Least-perimeter regular tiling         P_6^4 = 192 A^2 < P_4^4 = 256 A^2 < P_3^4 = 432 A^2
    Law 4  Lewis's law total-area invariance      total area = A_6 F on a torus for any slope
    Law 5  Aboav-Weaire sum rule                  sum n m(n) = F (36 + mu2), independent of a
    Law 6  Gibson division topology               s1 + s2 = n + 4, mean six is preserved

  All definitions and theorems are self-contained in standard Lean 4 core
  (zero external imports; every theorem is fully proved).
-/

namespace MathOfBiology.Loop_214

-- ============================================================================
-- Law 1 (`thompson1917-t1917-019-1`, E1):
-- Euler trivalent-froth side sum — "on the average hexagonal" (Thompson p. 325)
-- ============================================================================
/-- A planar cell aggregate in which every junction is trivalent (Thompson p. 309,
    p. 323: "only three partitions ... meet in a point") has `3 V = 2 E`; every edge
    is a side of exactly two cells, so the total side count is `sum_n = 2 E`; and
    Euler's relation `V - E + F = chi` holds for the surface the tissue covers.
    Then the total number of polygon sides is `6 F - 6 chi` — independent of the
    cell positions, sizes, or the mechanics that produced them. This is the exact
    content of Thompson's remark that irregular cells are "on the average hexagonal,
    but some will have fewer and some more sides than six". -/
def trivalent_side_sum_rhs (F chi : Int) : Int :=
  6 * F - 6 * chi

theorem trivalent_side_sum (V E F chi sum_n : Int)
    (h_trivalent : 3 * V = 2 * E)
    (h_two_cells_per_edge : sum_n = 2 * E)
    (h_euler : V - E + F = chi) :
    sum_n = trivalent_side_sum_rhs F chi := by
  simp only [trivalent_side_sum_rhs]
  omega

/-- On a periodic (toroidal, `chi = 0`) tissue — the boundary condition used by the
    Voronoi/Lloyd froth model in the loop script — the mean side number is exactly 6:
    `sum_n = 6 F`. -/
theorem torus_mean_six (V E F sum_n : Int)
    (h_trivalent : 3 * V = 2 * E)
    (h_two_cells_per_edge : sum_n = 2 * E)
    (h_euler : V - E + F = 0) :
    sum_n = 6 * F := by
  omega

/-- A closed trivalent tissue built only of 5-, 6- and 7-gons on a sphere (`chi = 2`)
    must carry exactly twelve more pentagons than heptagons (Euler's twelve-pentagon
    theorem) — the embryo-as-froth case of Thompson pp. 340-345. -/
theorem sphere_pentagon_excess (n5 n6 n7 V E : Int)
    (h_trivalent : 3 * V = 2 * E)
    (h_sides : 5 * n5 + 6 * n6 + 7 * n7 = 2 * E)
    (h_euler : V - E + (n5 + n6 + n7) = 2) :
    n5 - n7 = 12 := by
  omega


-- ============================================================================
-- Law 2 (`thompson1917-t1917-019-2`, E2):
-- Second moment of the polygon distribution and the honeycomb criterion
-- ============================================================================
/-- For a three-class tissue (`n5` pentagons, `n6` hexagons, `n7` heptagons) the
    un-normalised second moment `F * mu2 = sum_i (n_i - 6)^2` is simply `n5 + n7`. -/
def second_moment_numerator (n5 n7 : Nat) : Nat :=
  n5 * 1 + n7 * 1

/-- `F * mu2` equals `sum n_i^2 - 36 F` whenever the mean side number is 6
    (expanding `(n_i - 6)^2 = n_i^2 - 12 n_i + 36` and using `sum n_i = 6 F`),
    stated for the three-class decomposition. -/
theorem second_moment_identity (n5 n6 n7 : Int)
    (h_mean_six : 5 * n5 + 6 * n6 + 7 * n7 = 6 * (n5 + n6 + n7)) :
    25 * n5 + 36 * n6 + 49 * n7 - 36 * (n5 + n6 + n7) = n5 + n7 := by
  omega

/-- The second moment vanishes if and only if the tissue is a pure honeycomb
    (no pentagons and no heptagons). Thompson's strict reading of "hexagons" is
    exactly the `mu2 = 0` corner of the froth morphospace; every measured tissue
    with `mu2 > 0` is a froth, not a honeycomb. -/
theorem honeycomb_iff_zero_second_moment (n5 n7 : Nat) :
    second_moment_numerator n5 n7 = 0 ↔ (n5 = 0 ∧ n7 = 0) := by
  simp only [second_moment_numerator, Nat.mul_one]
  omega

/-- On a torus with only 5/6/7-gons the pentagon and heptagon counts balance
    (`n5 = n7`), so `F * mu2 = 2 n5`: the second moment counts defect *pairs*. -/
theorem torus_defect_pairs (n5 n6 n7 V E : Int)
    (h_trivalent : 3 * V = 2 * E)
    (h_sides : 5 * n5 + 6 * n6 + 7 * n7 = 2 * E)
    (h_euler : V - E + (n5 + n6 + n7) = 0) :
    n5 = n7 ∧ n5 + n7 = 2 * n5 := by
  omega


-- ============================================================================
-- Law 3 (`thompson1917-t1917-019-3`, E3):
-- Least-perimeter regular tiling — "the closely packed hexagons gave the minimal
-- extent of boundary in a plane" (Thompson p. 330; Hales 2001)
-- ============================================================================
/-- A regular `n`-gon of area `A` has perimeter `P_n = 2 sqrt(n A tan(pi/n))`, so
    `P_n^4 = 16 n^2 tan^2(pi/n) A^2`. For the three regular polygons that tile the
    plane the trigonometric factor is exact: `tan^2(pi/3) = 3`, `tan^2(pi/4) = 1`,
    `tan^2(pi/6) = 1/3`, giving the integer fourth-power perimeters
    `P_3^4 = 432 A^2`, `P_4^4 = 256 A^2`, `P_6^4 = 192 A^2`. -/
def fourth_power_perimeter_triangle (A_sq : Nat) : Nat := 432 * A_sq
def fourth_power_perimeter_square (A_sq : Nat) : Nat := 256 * A_sq
def fourth_power_perimeter_hexagon (A_sq : Nat) : Nat := 192 * A_sq

/-- For every positive cell area the hexagon has strictly the least perimeter of the
    three regular tilings, and the ordering hexagon < square < triangle is scale-free.
    The excess-perimeter ratios are `(256/192)^(1/4) = 1.0746` (square) and
    `(432/192)^(1/4) = 1.2247` (triangle): a square epithelium would spend 7.5 % more
    wall than a honeycomb of the same cell areas, a triangular one 22.5 % more. -/
theorem hexagon_least_fourth_power_perimeter (A_sq : Nat) (h_pos : 0 < A_sq) :
    fourth_power_perimeter_hexagon A_sq < fourth_power_perimeter_square A_sq ∧
    fourth_power_perimeter_square A_sq < fourth_power_perimeter_triangle A_sq ∧
    fourth_power_perimeter_hexagon 1 = 192 ∧
    fourth_power_perimeter_square 1 = 256 ∧
    fourth_power_perimeter_triangle 1 = 432 := by
  refine ⟨?_, ?_, rfl, rfl, rfl⟩
  · simp only [fourth_power_perimeter_hexagon, fourth_power_perimeter_square]
    omega
  · simp only [fourth_power_perimeter_square, fourth_power_perimeter_triangle]
    omega

/-- In a honeycomb every edge is shared by two cells, so the total wall length of
    `F` equal hexagons is `L = F P_6 / 2`, and `16 L^4 = F^4 P_6^4 = 192 F^4 A^2`, i.e.
    `L^4 = 12 F^4 A^2 = 12 (F A)^2 F^2`: the Hales honeycomb constant `12^(1/4) = 1.8612`
    is the least possible wall length per unit area per unit cell count. The loop
    script's `excess_perimeter` is `L_measured / L_honeycomb(same areas) - 1 >= 0`. -/
def honeycomb_sixteen_L4 (F4 A_sq : Nat) : Nat :=
  F4 * fourth_power_perimeter_hexagon A_sq

theorem honeycomb_edge_length_identity (F4 A_sq : Nat) :
    honeycomb_sixteen_L4 F4 A_sq = 16 * (12 * (F4 * A_sq)) := by
  simp only [honeycomb_sixteen_L4, fourth_power_perimeter_hexagon]
  have h16 : (16 : Nat) * 12 = 192 := by decide
  rw [Nat.mul_left_comm F4 192 A_sq, ← Nat.mul_assoc 16 12 (F4 * A_sq), h16]


-- ============================================================================
-- Law 4 (`thompson1917-t1917-019-4`, E4):
-- Lewis's law total-area invariance
-- ============================================================================
/-- Lewis (1928): the mean area of an `n`-sided cell is affine in `n`,
    `A_n = A_6 + k (n - 6)` (slope `k = A_6 / 4` in Lewis's own fit). Summing over all
    cells, `total_area = A_6 F + k (sum_n - 6 F)`. -/
def lewis_total_area (A6 k F sum_n : Int) : Int :=
  A6 * F + k * (sum_n - 6 * F)

/-- On a periodic tissue (`sum_n = 6 F`, Law 1) the Lewis correction term vanishes for
    EVERY slope `k`: the mean cell area equals the hexagon-class area `A_6` no matter
    how strongly size and topology are correlated. Lewis's law therefore redistributes
    area among polygon classes without changing the tissue's mean cell size — the
    size-topology coupling is a zero-sum reshuffle around the hexagon. -/
theorem lewis_total_area_invariance (A6 k F sum_n : Int)
    (h_torus : sum_n = 6 * F) :
    lewis_total_area A6 k F sum_n = A6 * F := by
  simp only [lewis_total_area]
  rw [h_torus]
  simp

/-- Lewis's per-class mean area `A_n = A_6 + k (n - 6)`. -/
def lewis_class_area (A6 k n : Int) : Int :=
  A6 + k * (n - 6)

/-- With Lewis's own slope `A_6 = 4 k` (i.e. `A_n = A_6 (n - 2) / 4`) the pentagon
    class has `3/4` and the heptagon class `5/4` of the hexagon area, and the pentagon
    and heptagon deficits cancel pairwise — the torus defect pairs of Law 2 carry no
    net area. -/
theorem lewis_quarter_slope_classes (A6 k : Int) (h_slope : A6 = 4 * k) :
    4 * lewis_class_area A6 k 5 = 3 * A6 ∧
    4 * lewis_class_area A6 k 7 = 5 * A6 ∧
    lewis_class_area A6 k 5 + lewis_class_area A6 k 7 = 2 * A6 := by
  simp only [lewis_class_area]
  have h5 : (5 : Int) - 6 = -1 := by decide
  have h7 : (7 : Int) - 6 = 1 := by decide
  rw [h5, h7]
  omega


-- ============================================================================
-- Law 5 (`thompson1917-t1917-019-5`, E5):
-- Aboav-Weaire sum rule
-- ============================================================================
/-- Aboav (1970) / Weaire: the mean side number of the neighbours of an `n`-gon obeys
    `n m(n) = (6 - a) n + (6 a + mu2)`. Summing `n_i m(n_i)` over the `F` cells gives
    `(6 - a) sum_n + F (6a + mu2)`. -/
def aboav_weaire_sum (a mu2 F sum_n : Int) : Int :=
  (6 - a) * sum_n + F * (6 * a + mu2)

/-- When `sum_n = 6 F` (Law 1) the Aboav parameter `a` cancels identically:
    `sum_i n_i m(n_i) = F (36 + mu2)`. This is Weaire's sum rule — it makes `mu2` the
    single parameter a froth needs beyond Euler, and it is the identity the loop script
    uses as a consistency check on the segmented corneal endothelium (the measured
    `sum n m(n) / F` must equal `36 + mu2` up to boundary effects). -/
theorem aboav_weaire_sum_rule (a mu2 F sum_n : Int)
    (h_torus : sum_n = 6 * F) :
    aboav_weaire_sum a mu2 F sum_n = F * (36 + mu2) := by
  simp only [aboav_weaire_sum]
  rw [h_torus]
  have e1 : (6 - a) * (6 * F) = 36 * F - 6 * a * F := by
    rw [Int.sub_mul]
    have : a * (6 * F) = 6 * a * F := by
      rw [Int.mul_comm a (6 * F), Int.mul_assoc 6 F a, Int.mul_comm F a, ← Int.mul_assoc 6 a F]
    rw [this]
    omega
  have e2 : F * (6 * a + mu2) = 6 * a * F + F * mu2 := by
    rw [Int.mul_add, Int.mul_comm F (6 * a)]
  have e3 : F * (36 + mu2) = 36 * F + F * mu2 := by
    rw [Int.mul_add, Int.mul_comm F 36]
  rw [e1, e2, e3]
  omega

/-- Honeycomb check: with `mu2 = 0` every neighbour of every (hexagonal) cell is a
    hexagon, so `sum n m(n) = 36 F` for any `a`. -/
theorem aboav_weaire_honeycomb (a F : Int) :
    aboav_weaire_sum a 0 F (6 * F) = 36 * F := by
  have h := aboav_weaire_sum_rule a 0 F (6 * F) rfl
  rw [h]
  omega


-- ============================================================================
-- Law 6 (`thompson1917-t1917-019-6`, E6):
-- Gibson (2006) division topology — why a growing epithelium is a froth of
-- 5-, 6- and 7-gons rather than a honeycomb
-- ============================================================================
/-- When an `n`-sided cell divides, the new wall joins two points on two different
    old sides. Daughter 1 inherits `k` undivided sides plus two half-sides plus the
    new wall (`k + 3` sides); daughter 2 inherits the remaining `n - 2 - k` undivided
    sides plus two half-sides plus the new wall. Their side counts always sum to
    `n + 4`. -/
def daughter1_sides (k : Nat) : Nat := k + 3
def daughter2_sides (n k : Nat) : Nat := (n - 2 - k) + 3

theorem division_daughter_side_sum (n k : Nat) (h_k : k ≤ n - 2) (h_n : 2 ≤ n) :
    daughter1_sides k + daughter2_sides n k = n + 4 := by
  simp only [daughter1_sides, daughter2_sides]
  omega

/-- The two neighbours whose shared sides were cut each gain one side. Hence one
    division adds `(n + 4) - n + 2 = 6` sides and one cell, so if the tissue had
    `sum_n = 6 F` before, it has `6 (F + 1)` after: the mean of six is an invariant of
    division — the froth stays a froth. -/
def sides_after_division (sum_n n : Nat) : Nat := sum_n - n + (n + 4) + 2

theorem division_preserves_mean_six (F sum_n n : Nat)
    (h_mean : sum_n = 6 * F) (h_n_le : n ≤ sum_n) :
    sides_after_division sum_n n = 6 * (F + 1) := by
  simp only [sides_after_division]
  omega

/-- A hexagon splitting symmetrically (`k = 2`) yields two pentagons; so a honeycomb
    that divides even once is no longer a honeycomb (`mu2` becomes positive by Law 2).
    Thompson's "approximately hexagons" cannot describe a proliferating tissue exactly. -/
theorem hexagon_symmetric_division_gives_pentagons :
    daughter1_sides 2 = 5 ∧ daughter2_sides 6 2 = 5 ∧
    second_moment_numerator 2 0 = 2 := by
  refine ⟨rfl, rfl, rfl⟩

end MathOfBiology.Loop_214
