/-
  morphoeval/lean/loops/Thompson1917_T1917_040.lean
  Formal Lean 4 verification for loop `thompson1917-t1917-040` (slot 232):
  Thompson, D'Arcy W. (1917) On Growth and Form, Ch. XVII — the Dürer/Camper facial grids.

  Thompson's transformation types on pp. 729-731 (Figs. 353-358) and his Dürer heads
  (Fig. 364 p. 740, Fig. 366 p. 742) are all *whole-grid* coordinate deformations:
    type (1) proportional stretch   (x, y) ↦ (x, c·y)
    type (3) simple shear / oblique (x, y) ↦ (x + k·y, y)      with k = -cot ω
    and their closure, the affine map  X' = A·X + t.
  The Camper footnote (p. 742) claims the facial angle "is the index to a general
  deformation": under a shear every facial line of the same rise acquires the same
  extra run k·dy, so one number (k, equivalently ω) indexes the whole deformation.

  Everything below is self-contained standard Lean 4 core over `Int` (integer-scaled
  coordinates, as in the other loop files); every proof is complete.  The theorems mirror the
  functions `shear`, `stretch`, `affine_fit`, `facial_line_tan` in
  thompson/claims/t1917-040_on-the-theory-of-transformations.py.
-/

namespace MathOfBiology.Loop_232

-- ============================================================================
-- Definitions (Thompson 1917 pp. 729-730, E1 / E3 / E4 of the reference JSON)
-- ============================================================================

/-- Type (3) simple shear, x-coordinate: `x' = x + k·y` (oblique net, k = -cot ω). -/
def shearX (k x y : Int) : Int := x + k * y

/-- Type (1) proportional stretch along y: `y' = c·y` (Fig. 353: square → oblong). -/
def stretchY (c y : Int) : Int := c * y

/-- 2×2 determinant `a·d − b·c` of the linear part of a whole-grid deformation. -/
def det2 (a b c d : Int) : Int := a * d - b * c

/-- General affine (whole-grid) deformation, x-coordinate: `x' = a·x + b·y + t`. -/
def affX (a b t x y : Int) : Int := a * x + b * y + t

/-- Horizontal run of a facial line of direction `(dx, dy)` after a shear `k`
    (Camper footnote, E5): `dx' = dx + k·dy`, the rise `dy` is unchanged. -/
def facialRun (k dx dy : Int) : Int := dx + k * dy

-- ============================================================================
-- Law 1 — the shears form a group indexed by one number k (Thompson type 3)
-- ============================================================================
/-- Composing two oblique nets is the oblique net with `k₁ + k₂`; the shear with
    `-k` undoes `k`; the linear part `[[1, k], [0, 1]]` has determinant 1, so the
    deformation preserves area (Dürer's grids keep every cell's area). -/
theorem shear_group_law (k₁ k₂ k x y : Int) :
    shearX k₂ (shearX k₁ x y) y = shearX (k₁ + k₂) x y ∧
    shearX (-k) (shearX k x y) y = x ∧
    det2 1 k 0 1 = 1 := by
  refine ⟨?_, ?_, ?_⟩
  · simp [shearX, Int.add_mul]; omega
  · simp [shearX, Int.neg_mul]; omega
  · simp [det2]

-- ============================================================================
-- Law 2 — a shear is a *whole-grid* deformation: rows keep their lengths,
--         verticals all tilt by the same amount (Fig. 357 / Fig. 366)
-- ============================================================================
/-- Two landmarks on the same grid row keep their horizontal separation, and a
    vertical segment of height `h` acquires exactly the run `k·h`, whatever its
    abscissa `x`.  Hence the deformation is read off the net, not off the parts. -/
theorem shear_whole_grid (k x x₁ x₂ y h : Int) :
    shearX k x₁ y - shearX k x₂ y = x₁ - x₂ ∧
    shearX k x (y + h) - shearX k x y = k * h := by
  constructor
  · simp [shearX]; omega
  · simp [shearX, Int.mul_add]; omega

-- ============================================================================
-- Law 3 — Camper: the facial angle is the index of the general deformation
-- ============================================================================
/-- (i) Every facial line of the same rise `dy` changes its run by the same amount,
    independent of its original inclination `dx`; (ii) the run — hence tan of the
    facial angle — is strictly monotone in the shear `k` for a line that rises
    (`0 < dy`); (iii) the undeformed net (`k = 0`) leaves the facial line alone.
    So the facial angle and the shear determine each other: one index, one deformation. -/
theorem camper_facial_angle_index (k k₁ k₂ dx₁ dx₂ dy : Int)
    (hk : k₁ < k₂) (hdy : 0 < dy) :
    facialRun k dx₁ dy - dx₁ = facialRun k dx₂ dy - dx₂ ∧
    facialRun k₁ dx₁ dy < facialRun k₂ dx₁ dy ∧
    facialRun 0 dx₁ dy = dx₁ := by
  refine ⟨?_, ?_, ?_⟩
  · simp [facialRun]; omega
  · have := Int.mul_lt_mul_of_pos_right hk hdy
    simp [facialRun]; omega
  · simp [facialRun]

-- ============================================================================
-- Law 4 — proportional stretch (Thompson type 1): circle → ellipse, composition
-- ============================================================================
/-- Thompson p. 729-730: substituting `x₁, c·y₁` turns `x² + y² = a²` into
    `x₁² + c²·y₁² = a²`.  Stretches compose multiplicatively (two successive
    oblongs are one oblong), and the ox → sheep cannon-bone factor 3/2 (p. 730)
    appears as `stretchY 3 2 = 6 = 2·3`. -/
theorem stretch_circle_to_ellipse (x y₁ c c₁ c₂ : Int) :
    x * x + stretchY c y₁ * stretchY c y₁ = x * x + (c * c) * (y₁ * y₁) ∧
    stretchY c₁ (stretchY c₂ y₁) = stretchY (c₁ * c₂) y₁ ∧
    stretchY 3 2 = 6 := by
  refine ⟨?_, ?_, by decide⟩
  · simp [stretchY, Int.mul_comm, Int.mul_left_comm]
  · simp [stretchY, Int.mul_assoc]

-- ============================================================================
-- Law 5 — the whole-grid (affine) family: landmark *differences* transform
--         linearly, and stretch ∘ shear is again one affine map (closure)
-- ============================================================================
/-- (i) Under one affine map the displacement between two landmarks depends only on
    the linear part `(a, b)`, never on the translation `t`: this is what "the form
    varies as a whole" means for a landmark configuration (E4).
    (ii) Shearing a stretched net is the single affine map with `b = k·c` — the
    Dürer grid of Fig. 366 is one member of the family, not a composition of
    independent local changes. -/
theorem affine_whole_grid_closure (a b t k c x₁ y₁ x₂ y₂ : Int) :
    affX a b t x₁ y₁ - affX a b t x₂ y₂ = a * (x₁ - x₂) + b * (y₁ - y₂) ∧
    shearX k x₁ (stretchY c y₁) = affX 1 (k * c) 0 x₁ y₁ := by
  constructor
  · simp [affX, Int.mul_sub]; omega
  · simp [shearX, stretchY, affX, Int.mul_assoc]

-- ============================================================================
-- Law 6 — Thompson's oblique transposition is invertible iff cosec ω ≠ 0
-- ============================================================================
/-- p. 730: `x = ξ − η·cot ω, y = η·cosec ω`.  With integer-scaled `ct = cot ω`,
    `cs = cosec ω`, the linear part `[[1, -ct], [0, cs]]` has determinant `cs`, so the
    net can be read back (the comparison of forms is two-way) exactly when `cs ≠ 0`,
    i.e. whenever the oblique axes are not degenerate. -/
theorem oblique_transposition_invertible (ct cs : Int) (h : cs ≠ 0) :
    det2 1 (-ct) 0 cs = cs ∧ det2 1 (-ct) 0 cs ≠ 0 := by
  constructor
  · simp [det2]
  · simp [det2]; exact h

end MathOfBiology.Loop_232
