/-
  morphoeval/lean/loops/Thompson1917_T1917_024.lean
  Loop thompson1917-t1917-024 (campaign slot 219) — Thompson, D.W. (1917) On Growth and Form,
  Ch. IX "On Concretions, Spicules, and Spicular Skeletons", pp. 443-449, Fig. 214.

  Claim T1917-024: sponge spicule forms (triradiates, tetraradiates) are the forms physical
  deposition takes within the cell field that secretes them.  Thompson's operational version
  (after Dreyer 1892 and Plateau 1873) is that the spicule rays lie along the partitions of
  three (or four) contiguous cells, so that their axes obey the force balance of a Plateau
  border — the Neumann tension triangle — and therefore meet at 120 deg when the "specific
  tensions" are equal, and at "two equal angles and one unequal" when they are not.

  Integer formalisation (Lean 4 core only, no Mathlib, every theorem fully proved, no
  placeholder proofs; checked with `lean` 4.35.0, exit 0).  Trigonometric functions are
  avoided by working with the law of cosines in integer numerator / denominator form:

      cos(theta_1) = cosNum g1 g2 g3 / cosDen g2 g3,
      cosNum g1 g2 g3 = g1^2 - g2^2 - g3^2,   cosDen g2 g3 = 2 g2 g3,          (E3)

  where theta_1 is the ray angle OPPOSITE interface 1 (between rays 2 and 3) and g_i are the
  interface tensions.  Script: thompson/claims/t1917-024_on-concretions-spicules-and-spicular-
  skeletons.py (functions `neumann_angles`, `sagittal_angles`, `tension_ratio_from_angles`).

  Theorems
    1. `plateau_equiangular_junction`      (E4, P1, Fig. 214 F): equal tensions give
       cos theta = -1/2 for every pair, i.e. the 120 deg Plateau junction, for every g > 0.
    2. `sagittal_paired_angles_equal`      (E5, P2): with one cell-cell tension c and two
       cell-medium tensions m the two angles adjacent to the cell-cell ray have identical
       cosine numerators and denominators — "two equal angles".
    3. `sagittal_equiangular_iff_equal_tensions` (E6): the unpaired angle equals the paired
       angle IFF c = m; any inequality of specific tensions breaks equiangularity (Thompson
       p. 445), via the exact factorisation
          cosNum c m m * cosDen c m - cosNum m c m * cosDen m m = 2 c m (c - m)(c + 2m).
    4. `sagittal_unpaired_angle_ordering_and_T_limit` (E5a, F4): for 0 < c < m the unpaired
       angle is strictly larger than the paired angles (the sagittal form opens towards the
       T shape, never towards a narrow fork), the unpaired cosine is strictly increasing in c,
       and the two boundary cases are exact: c = 0 gives cos = -1 (180 deg, rays "nearly in a
       straight line", p. 443) and c = 2m gives cos = +1 (0 deg, cells detach, no spicule).
    5. `tetrahedral_and_orthogonal_ray_systems` (E7, E8, P3): the four body-diagonal rays of
       a cube are in exact force balance with pairwise dot product -1 on squared norm 3
       (cos = -1/3, the tetrahedral 109.47 deg of Fig. 214 E), and the six axial rays are in
       force balance with adjacent dot products 0 (the 90 deg hexactine network).
-/

namespace MathOfBiology.Loop_219

/-- Law-of-cosines numerator for the angle opposite interface 1 (E3). -/
def cosNum (g1 g2 g3 : Int) : Int := g1 * g1 - g2 * g2 - g3 * g3

/-- Law-of-cosines denominator for the angle opposite interface 1 (E3). -/
def cosDen (g2 g3 : Int) : Int := 2 * (g2 * g3)

-- ============================================================================
-- Theorem 1: equal tensions -> 120 deg Plateau junction (E4, P1, Fig. 214 F)
-- ============================================================================

/-- For any positive tension g, each of the three cosines equals -1/2 exactly
    (2 * num + den = 0 with den > 0), so all three ray angles are 120 deg. -/
theorem plateau_equiangular_junction (g : Int) (hg : 0 < g) :
    2 * cosNum g g g + cosDen g g = 0 ∧ 0 < cosDen g g := by
  unfold cosNum cosDen
  have hsq : 0 < g * g := Int.mul_pos hg hg
  constructor
  · omega
  · omega

-- ============================================================================
-- Theorem 2: sagittal spicule has two equal angles (E5, P2, Fig. 214 A, C)
-- ============================================================================

/-- Ray 1 lies in the cell-cell partition (tension c); rays 2 and 3 lie in the
    cell-medium surface (tension m each).  The angles opposite rays 2 and 3 have
    identical cosine numerators and denominators: "two equal angles". -/
theorem sagittal_paired_angles_equal (c m : Int) :
    cosNum m c m = cosNum m m c ∧ cosDen c m = cosDen m c := by
  unfold cosNum cosDen
  constructor
  · omega
  · rw [Int.mul_comm c m]

-- ============================================================================
-- Theorem 3: equiangular iff equal specific tensions (E6, Thompson p. 445)
-- ============================================================================

/-- Exact factorisation of the cross-multiplied cosine difference
    cos(theta_unpaired) - cos(theta_paired) (times the positive denominators). -/
theorem sagittal_cosine_difference_factorisation (c m : Int) :
    cosNum c m m * cosDen c m - cosNum m c m * cosDen m m
      = 2 * ((c * m) * ((c - m) * (c + 2 * m))) := by
  unfold cosNum cosDen
  have hfac : (c - m) * (c + 2 * m) = c * c + c * m - 2 * (m * m) := by
    rw [Int.sub_mul, Int.mul_add, Int.mul_add]
    have h1 : c * (2 * m) = 2 * (c * m) := Int.mul_left_comm c 2 m
    have h2 : m * c = c * m := Int.mul_comm m c
    have h3 : m * (2 * m) = 2 * (m * m) := Int.mul_left_comm m 2 m
    rw [h1, h2, h3]; omega
  rw [hfac]
  have hsq : (c * c) * (m * m) = (c * m) * (c * m) := by ac_rfl
  generalize ha : c * c = a at *
  generalize hb : m * m = b at *
  generalize hu : c * m = u at *
  have e1 : (a - b - b) * (2 * u) = 2 * (a * u) - 4 * (b * u) := by
    rw [Int.sub_mul, Int.sub_mul]
    have : a * (2 * u) = 2 * (a * u) := Int.mul_left_comm a 2 u
    have : b * (2 * u) = 2 * (b * u) := Int.mul_left_comm b 2 u
    omega
  have e2 : (b - a - b) * (2 * b) = -(2 * (a * b)) := by
    rw [Int.sub_mul, Int.sub_mul]
    have : a * (2 * b) = 2 * (a * b) := Int.mul_left_comm a 2 b
    have : b * (2 * b) = 2 * (b * b) := Int.mul_left_comm b 2 b
    omega
  have e3 : u * (a + u - 2 * b) = a * u + u * u - 2 * (b * u) := by
    rw [Int.mul_sub, Int.mul_add]
    have : u * a = a * u := Int.mul_comm u a
    have : u * (2 * b) = 2 * (u * b) := Int.mul_left_comm u 2 b
    have : u * b = b * u := Int.mul_comm u b
    omega
  rw [e1, e2, e3]
  omega

/-- The unpaired angle equals the paired angles iff the cell-cell tension equals the
    cell-medium tension.  (Both denominators are positive, so equality of the
    cross-products is equality of the cosines.) -/
theorem sagittal_equiangular_iff_equal_tensions (c m : Int) (hc : 0 < c) (hm : 0 < m) :
    cosNum c m m * cosDen c m = cosNum m c m * cosDen m m ↔ c = m := by
  have hkey := sagittal_cosine_difference_factorisation c m
  have hcm : 0 < c * m := Int.mul_pos hc hm
  constructor
  · intro h
    have h0 : 2 * ((c * m) * ((c - m) * (c + 2 * m))) = 0 := by omega
    have h1 : (c * m) * ((c - m) * (c + 2 * m)) = 0 := by omega
    rcases Int.mul_eq_zero.mp h1 with h2 | h2
    · omega
    · rcases Int.mul_eq_zero.mp h2 with h3 | h3
      · omega
      · omega
  · intro h
    subst h
    rfl

-- ============================================================================
-- Theorem 4: ordering, monotonicity and the T-shaped limit (E5a, F4, p. 443)
-- ============================================================================

/-- (i) For 0 < c < m the unpaired angle is strictly LARGER than the paired angles
    (cosine strictly smaller): the sagittal spicule opens towards (180, 90, 90).
    (ii) The unpaired cosine numerator is strictly increasing in c on c >= 0, so the
    unpaired angle closes monotonically as the cell-cell tension grows.
    (iii) c = 0 gives cos = -1 exactly (T shape); c = 2m gives cos = +1 exactly (detachment). -/
theorem sagittal_unpaired_angle_ordering_and_T_limit (c c' m : Int)
    (hc : 0 < c) (hcm : c < m) (hcc' : c < c') :
    cosNum c m m * cosDen c m < cosNum m c m * cosDen m m
    ∧ cosNum c m m < cosNum c' m m
    ∧ cosNum 0 m m = -(cosDen m m)
    ∧ cosNum (2 * m) m m = cosDen m m := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · have hkey := sagittal_cosine_difference_factorisation c m
    have hm : 0 < m := by omega
    have hpos : 0 < c * m := Int.mul_pos hc hm
    have hneg : c - m < 0 := by omega
    have hsum : 0 < c + 2 * m := by omega
    have h1 : (c - m) * (c + 2 * m) < 0 := Int.mul_neg_of_neg_of_pos hneg hsum
    have h2 : (c * m) * ((c - m) * (c + 2 * m)) < 0 := Int.mul_neg_of_pos_of_neg hpos h1
    omega
  · unfold cosNum
    have hd : 0 < c' - c := by omega
    have hs : 0 < c' + c := by omega
    have hprod : 0 < (c' - c) * (c' + c) := Int.mul_pos hd hs
    have hexp : (c' - c) * (c' + c) = c' * c' - c * c := by
      rw [Int.sub_mul, Int.mul_add, Int.mul_add]
      have : c * c' = c' * c := Int.mul_comm c c'
      omega
    omega
  · unfold cosNum cosDen
    omega
  · unfold cosNum cosDen
    have : (2 * m) * (2 * m) = 4 * (m * m) := by
      rw [Int.mul_assoc, Int.mul_left_comm m 2 m, ← Int.mul_assoc]
      omega
    omega

-- ============================================================================
-- Theorem 5: tetrahedral (4-ray) and orthogonal (6-ray) systems (E7, E8, P3)
-- ============================================================================

/-- Integer 3-vectors. -/
structure V3 where
  x : Int
  y : Int
  z : Int
deriving DecidableEq, Repr

def V3.add (a b : V3) : V3 := ⟨a.x + b.x, a.y + b.y, a.z + b.z⟩
def V3.dot (a b : V3) : Int := a.x * b.x + a.y * b.y + a.z * b.z
def V3.zero : V3 := ⟨0, 0, 0⟩

/-- The four body-diagonal ray directions of Thompson's tetractine (Fig. 214 E). -/
def tet1 : V3 := ⟨1, 1, 1⟩
def tet2 : V3 := ⟨1, -1, -1⟩
def tet3 : V3 := ⟨-1, 1, -1⟩
def tet4 : V3 := ⟨-1, -1, 1⟩

/-- The six axial ray directions of a hexactine. -/
def ex : V3 := ⟨1, 0, 0⟩
def ey : V3 := ⟨0, 1, 0⟩
def ez : V3 := ⟨0, 0, 1⟩
def nex : V3 := ⟨-1, 0, 0⟩
def ney : V3 := ⟨0, -1, 0⟩
def nez : V3 := ⟨0, 0, -1⟩

/-- Tetractine: equal-tension force balance (E1) holds exactly, every pair of rays has
    dot product -1 on squared norm 3, i.e. cos theta = -1/3 (109.47 deg, "about 109 deg",
    p. 446).  Hexactine: force balance holds, every adjacent pair is orthogonal (cos = 0,
    90 deg) and every opposite pair is antiparallel. -/
theorem tetrahedral_and_orthogonal_ray_systems :
    -- force balance of the four tetrahedral rays
    (tet1.add tet2).add (tet3.add tet4) = V3.zero
    -- squared norms and pairwise dot products: cos = -1/3 for all six pairs
    ∧ tet1.dot tet1 = 3 ∧ tet2.dot tet2 = 3 ∧ tet3.dot tet3 = 3 ∧ tet4.dot tet4 = 3
    ∧ tet1.dot tet2 = -1 ∧ tet1.dot tet3 = -1 ∧ tet1.dot tet4 = -1
    ∧ tet2.dot tet3 = -1 ∧ tet2.dot tet4 = -1 ∧ tet3.dot tet4 = -1
    -- force balance of the six axial rays
    ∧ ((ex.add ey).add ez).add ((nex.add ney).add nez) = V3.zero
    -- adjacent rays orthogonal (cos = 0), opposite rays antiparallel (cos = -1)
    ∧ ex.dot ey = 0 ∧ ex.dot ez = 0 ∧ ey.dot ez = 0
    ∧ ex.dot nex = -1 ∧ ey.dot ney = -1 ∧ ez.dot nez = -1 := by
  decide

end MathOfBiology.Loop_219
