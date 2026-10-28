/-
  morphoeval/lean/loops/Thompson1917_T1917_058.lean
  Formal Lean 4 verification for Thompson (1917) On Growth and Form, Ch. XVII,
  pp. 770-773, Figs. 404-408 (claim T1917-058, loop `thompson1917-t1917-058`, slot 247):

    "The human skull, inscribed in Cartesian coordinates and deformed, gives the skulls
     of chimpanzee, baboon and dog: ape/human skull difference is a whole-diagram
     transformation (notably of the facial/cranial proportions)."

  The script thompson/claims/t1917-058_on-the-theory-of-transformations.py reconstructs
  Thompson's deformed net as a planar map T and decomposes it into an affine ("rectangular
  box stretch") part and a non-affine (bending) part.  The theorems below pin down, in
  integer arithmetic, the algebraic facts the script's tests rely on:

    Law 1  An affine map sends every unit cell of the Cartesian net to a parallelogram with
           the SAME edge vectors (a, c), (b, d) and the SAME signed double area a*d - b*c.
           Hence straight net lines stay straight and parallel, and the cell-area field is
           flat.  Thompson's curved lines (Fig. 405) and graded cell areas (p. 771) are
           therefore evidence of a non-affine component - the test E3/E4 of the script.
    Law 2  Thompson's "approximately logarithmic order" of area increase (p. 771) is the
           model log lambda(i, j) = alpha + beta_i * i + beta_j * j.  Its increments along
           the net are constant, and it is strictly monotone from behind forwards
           (beta_i > 0) and from above downwards (beta_j < 0) - the sign test E4.
    Law 3  The intensity family T_k = Id + k (T - Id) ("the baboon differs only in an
           increased intensity or degree of deformation", p. 771): T_0 = Id, T_1 = T, the
           displacement scales linearly with k, and for a dilating box stretch the
           cell-area ratio (1 + k(a-1)) (1 + k(d-1)) is monotone non-decreasing in k.
    Law 4  Thompson's dog criterion (p. 773): a "rectangular conformation" has a constant
           height profile, a "triangular" one a strictly decreasing profile towards the
           front, so the anterior/posterior taper ratio is 1 for the rectangle and < 1
           for the triangle - the taper index of the script's silhouette segmentation.

  All definitions and theorems are self-contained in standard Lean 4 core (no Mathlib,
  zero `sorry`s).  Checked with `lean morphoeval/lean/loops/Thompson1917_T1917_058.lean`
  (Lean 4.15.0).
-/

namespace MathOfBiology.Loop_247

-- ============================================================================
-- Law 1 (`thompson1917-t1917-058-1`):
-- Affine maps keep the Cartesian net straight and its cell-area field flat
-- ============================================================================

/-- Affine ("box stretch + shear") image of the net node (i, j):
    X = a i + b j + e,  Y = c i + d j + f  (Fig. 404 abscissa i = 0..5, ordinate j = 0..4). -/
def affine_x (a b e i j : Int) : Int := a * i + b * j + e
def affine_y (c d f i j : Int) : Int := c * i + d * j + f

/-- Signed double area (shoelace) of the image of the unit cell with lower-left corner
    (i, j), computed from its two edge vectors at that corner. -/
def cell_double_area (a b c d e f i j : Int) : Int :=
  (affine_x a b e (i + 1) j - affine_x a b e i j) * (affine_y c d f i (j + 1) - affine_y c d f i j)
  - (affine_x a b e i (j + 1) - affine_x a b e i j) * (affine_y c d f (i + 1) j - affine_y c d f i j)

/-- Under any affine map every unit cell of the net has the same edge vectors (a, c) along
    the abscissa and (b, d) along the ordinate - net lines remain straight and parallel -
    and the same signed double area a d - b c, independent of (i, j).  A pure box stretch
    (b = c = 0) therefore has the constant cell-area ratio a * d. -/
theorem affine_net_cells_congruent (a b c d e f i j : Int) :
    affine_x a b e (i + 1) j - affine_x a b e i j = a ∧
    affine_y c d f (i + 1) j - affine_y c d f i j = c ∧
    affine_x a b e i (j + 1) - affine_x a b e i j = b ∧
    affine_y c d f i (j + 1) - affine_y c d f i j = d ∧
    cell_double_area a b c d e f i j = a * d - b * c ∧
    cell_double_area a 0 0 d e f i j = a * d := by
  have hx1 : affine_x a b e (i + 1) j - affine_x a b e i j = a := by
    simp [affine_x, Int.mul_add]; omega
  have hy1 : affine_y c d f (i + 1) j - affine_y c d f i j = c := by
    simp [affine_y, Int.mul_add]; omega
  have hx2 : affine_x a b e i (j + 1) - affine_x a b e i j = b := by
    simp [affine_x, Int.mul_add]; omega
  have hy2 : affine_y c d f i (j + 1) - affine_y c d f i j = d := by
    simp [affine_y, Int.mul_add]; omega
  have harea : cell_double_area a b c d e f i j = a * d - b * c := by
    unfold cell_double_area
    rw [hx1, hy1, hx2, hy2]
  have hbox : cell_double_area a 0 0 d e f i j = a * d := by
    have h := harea
    -- specialise the general identity to b = 0, c = 0
    have h0 : cell_double_area a 0 0 d e f i j = a * d - 0 * 0 := by
      unfold cell_double_area
      have hx1' : affine_x a 0 e (i + 1) j - affine_x a 0 e i j = a := by
        simp [affine_x, Int.mul_add]; omega
      have hy1' : affine_y 0 d f (i + 1) j - affine_y 0 d f i j = 0 := by
        simp [affine_y, Int.mul_add]
      have hx2' : affine_x a 0 e i (j + 1) - affine_x a 0 e i j = 0 := by
        simp [affine_x, Int.mul_add]
      have hy2' : affine_y 0 d f i (j + 1) - affine_y 0 d f i j = d := by
        simp [affine_y, Int.mul_add]; omega
      rw [hx1', hy1', hx2', hy2']
    simpa using h0
  exact ⟨hx1, hy1, hx2, hy2, harea, hbox⟩

/-- Numerical anchor used by the script's self-test: the box stretch a = 3, d = 2 gives
    double area 6 in the cell (0,0) and in the cell (4,3) alike (flat Jacobian field). -/
theorem affine_net_flat_area_example :
    cell_double_area 3 0 0 2 0 0 0 0 = 6 ∧ cell_double_area 3 0 0 2 0 0 4 3 = 6 := by
  decide

-- ============================================================================
-- Law 2 (`thompson1917-t1917-058-2`):
-- Thompson's "approximately logarithmic order" of cell-area increase
-- ============================================================================

/-- Log cell-area model: log lambda(i, j) = alpha + beta_i * i + beta_j * j
    (i: behind -> forwards, j: below -> above, all scaled to integers). -/
def log_area (alpha beta_i beta_j i j : Int) : Int := alpha + beta_i * i + beta_j * j

/-- (1) The increments of log lambda along the two net directions are the constants
    beta_i and beta_j (geometric progression of cell areas = Thompson's "logarithmic
    order"); (2) with beta_i > 0 the areas strictly increase from behind forwards;
    (3) with beta_j < 0 they strictly increase from above downwards; (4) the step-wise
    area RATIO lambda(i+1, j) / lambda(i, j) = exp(beta_i) is therefore also constant,
    stated here as equality of consecutive log-differences. -/
theorem log_area_gradient (alpha beta_i beta_j i j : Int) :
    log_area alpha beta_i beta_j (i + 1) j - log_area alpha beta_i beta_j i j = beta_i ∧
    log_area alpha beta_i beta_j i (j + 1) - log_area alpha beta_i beta_j i j = beta_j ∧
    (0 < beta_i → ∀ di : Int, 0 < di →
        log_area alpha beta_i beta_j i j < log_area alpha beta_i beta_j (i + di) j) ∧
    (beta_j < 0 → ∀ dj : Int, 0 < dj →
        log_area alpha beta_i beta_j i (j + dj) < log_area alpha beta_i beta_j i j) ∧
    (log_area alpha beta_i beta_j (i + 2) j - log_area alpha beta_i beta_j (i + 1) j =
      log_area alpha beta_i beta_j (i + 1) j - log_area alpha beta_i beta_j i j) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · unfold log_area
    have h := Int.mul_add beta_i i 1
    omega
  · unfold log_area
    have h := Int.mul_add beta_j j 1
    omega
  · intro hb di hdi
    unfold log_area
    have h := Int.mul_add beta_i i di
    have hpos : 0 < beta_i * di := Int.mul_pos hb hdi
    omega
  · intro hb dj hdj
    unfold log_area
    have h := Int.mul_add beta_j j dj
    have hneg : beta_j * dj < 0 := Int.mul_neg_of_neg_of_pos hb hdj
    omega
  · unfold log_area
    have h1 := Int.mul_add beta_i i 1
    have h2 := Int.mul_add beta_i i 2
    omega

-- ============================================================================
-- Law 3 (`thompson1917-t1917-058-3`):
-- The intensity family T_k = Id + k (T - Id) ("same order, increased intensity")
-- ============================================================================

/-- One axis of the box-stretch intensity family: T_k(x) = x + k (a x - x). -/
def intensity_map (a k x : Int) : Int := x + k * (a * x - x)

/-- Scale factor of T_k along one axis: 1 + k (a - 1). -/
def intensity_scale (a k : Int) : Int := 1 + k * (a - 1)

/-- Cell-area ratio of the box-stretch intensity family: (1 + k(a-1)) (1 + k(d-1)). -/
def intensity_area (a d k : Int) : Int := intensity_scale a k * intensity_scale d k

/-- (1) k = 0 is the identity (human net) and k = 1 is T itself (chimpanzee net);
    (2) the displacement T_k(x) - x is exactly k times the displacement of T;
    (3) T_k is the box stretch with factor 1 + k (a - 1);
    (4) for a dilating stretch (a >= 1, d >= 1) the cell-area ratio is monotone
        non-decreasing in the intensity k >= 0 - "increased intensity" of the same
        transformation can only enlarge the facial cells further, never reverse them. -/
theorem intensity_family (a d k k' x : Int) :
    intensity_map a 0 x = x ∧
    intensity_map a 1 x = a * x ∧
    intensity_map a k x - x = k * (intensity_map a 1 x - x) ∧
    intensity_map a k x = intensity_scale a k * x ∧
    (1 ≤ a → 1 ≤ d → 0 ≤ k → k ≤ k' → intensity_area a d k ≤ intensity_area a d k') := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · unfold intensity_map
    omega
  · unfold intensity_map
    omega
  · unfold intensity_map
    have h : x + 1 * (a * x - x) - x = a * x - x := by omega
    rw [h]
    omega
  · unfold intensity_map intensity_scale
    have h1 : k * (a * x - x) = k * ((a - 1) * x) := by
      rw [Int.sub_mul, Int.one_mul]
    have h2 : (1 + k * (a - 1)) * x = x + k * (a - 1) * x := by
      rw [Int.add_mul, Int.one_mul]
    rw [h1, h2, Int.mul_assoc]
  · intro ha hd hk hkk
    unfold intensity_area intensity_scale
    have ha1 : 0 ≤ a - 1 := by omega
    have hd1 : 0 ≤ d - 1 := by omega
    have hsa : k * (a - 1) ≤ k' * (a - 1) := Int.mul_le_mul_of_nonneg_right hkk ha1
    have hsd : k * (d - 1) ≤ k' * (d - 1) := Int.mul_le_mul_of_nonneg_right hkk hd1
    have hk' : 0 ≤ k' := Int.le_trans hk hkk
    have hc : 0 ≤ 1 + k * (d - 1) := by
      have := Int.mul_nonneg hk hd1
      omega
    have hb : 0 ≤ 1 + k' * (a - 1) := by
      have := Int.mul_nonneg hk' ha1
      omega
    have hA : 1 + k * (a - 1) ≤ 1 + k' * (a - 1) := by omega
    have hD : 1 + k * (d - 1) ≤ 1 + k' * (d - 1) := by omega
    exact Int.mul_le_mul hA hD hc hb

/-- Numerical anchor: with a = 3, d = 2 (chimpanzee-like facial dilation) the area ratio
    grows 1 -> 6 -> 15 for k = 0, 1, 2 - the ordering the script tests on the real
    human / chimpanzee / baboon specimens (C2, C4). -/
theorem intensity_family_example :
    intensity_area 3 2 0 = 1 ∧ intensity_area 3 2 1 = 6 ∧ intensity_area 3 2 2 = 15 := by
  decide

-- ============================================================================
-- Law 4 (`thompson1917-t1917-058-4`):
-- Rectangular vs triangular conformation (Thompson's dog criterion, p. 773)
-- ============================================================================

/-- Height profile of a rectangular conformation: constant h0. -/
def rect_height (h0 _x : Int) : Int := h0

/-- Height profile of a triangular conformation tapering forwards with slope s > 0. -/
def tri_height (h0 s x : Int) : Int := h0 - s * x

/-- For any posterior station xb and anterior station xf > xb: the rectangle has equal
    heights (taper ratio 1), the triangle is strictly lower in front (taper ratio < 1),
    and the deficit grows linearly with the distance xf - xb.  This is the sign of the
    silhouette taper index that distinguishes the dog net from the ape nets. -/
theorem rectangular_vs_triangular (h0 s xb xf : Int) (hs : 0 < s) (hx : xb < xf) :
    rect_height h0 xf = rect_height h0 xb ∧
    tri_height h0 s xf < tri_height h0 s xb ∧
    tri_height h0 s xb - tri_height h0 s xf = s * (xf - xb) := by
  refine ⟨rfl, ?_, ?_⟩
  · unfold tri_height
    have : 0 < s * (xf - xb) := Int.mul_pos hs (by omega)
    have h2 : s * (xf - xb) = s * xf - s * xb := Int.mul_sub s xf xb
    omega
  · unfold tri_height
    have h2 : s * (xf - xb) = s * xf - s * xb := Int.mul_sub s xf xb
    omega

end MathOfBiology.Loop_247
