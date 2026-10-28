/-
  morphoeval/lean/loops/Thompson1917_T1917_054.lean
  Slot 243 · thompson1917-t1917-054 · Thompson (1917) On Growth and Form, Ch. XVII,
  pp. 757-759, Figs. 388-393: the pelves of Archaeopteryx and Apatornis (after Heilmann)
  and the three "intermediate and equidistant co-ordinate systems ... interpolated between
  them" (p. 759).

  Governing equations (datasets/thompson_1917/t1917-054/..._reference.json):
    E2  phi_t(x) = (1 - t) x + t phi(x),           t in {0, 1/4, 1/2, 3/4, 1}
    E5  affine case: phi_t(x) = ((1-t) I + t A) x + t b
    E6  corresponding points: phi_t(a_i) = (1 - t) a_i + t b_i, equal steps of (b_i - a_i)/4
    E7  best-fit stage t* = <D - A, B - A> / |B - A|^2

  Everything is stated over Int with the stage t = k/4 cleared of denominators:
    interp4 k a b = 4 * phi_{k/4}(a)  when phi(a) = b.
  All proofs are self-contained Lean 4 core (no Mathlib, no imports, no placeholder proofs).
  Mirrors `interp_stage`, `affine_interp_matrix`, `best_fit_t` and
  `check_no_fold` in thompson/claims/t1917-054_on-the-theory-of-transformations.py.
-/
namespace MathOfBiology.Loop_243

/-- E2/E6 with t = k/4, denominators cleared: `interp4 k a b = (4 - k) a + k b = 4 * phi_{k/4}(a)`
    for a point `a` of the Archaeopteryx grid whose image in the Apatornis grid is `b`. -/
def interp4 (k a b : Int) : Int := (4 - k) * a + k * b

/-- Scaled 2x2 interpolated affine matrix `(4 - k) I + k A` (E5), given `A = [[a11, a12], [a21, a22]]`. -/
def interpMat11 (k a11 : Int) : Int := (4 - k) + k * a11
def interpMat12 (k a12 : Int) : Int := k * a12
def interpMat21 (k a21 : Int) : Int := k * a21
def interpMat22 (k a22 : Int) : Int := (4 - k) + k * a22

/-- Determinant of the scaled interpolated affine matrix (E5). -/
def interpDet (k a11 a12 a21 a22 : Int) : Int :=
  interpMat11 k a11 * interpMat22 k a22 - interpMat12 k a12 * interpMat21 k a21

-- ============================================================================
-- Theorem 1 (E2, E6): endpoints and equidistance of Thompson's stages.
-- Stage 0 is Archaeopteryx, stage 4 is Apatornis, consecutive stages differ by the
-- same displacement (b - a), and the middle stage is the exact midpoint.
-- ============================================================================
theorem interp_endpoints_and_equidistant (k a b : Int) :
    interp4 0 a b = 4 * a ∧
    interp4 4 a b = 4 * b ∧
    interp4 (k + 1) a b - interp4 k a b = b - a ∧
    2 * interp4 2 a b = interp4 0 a b + interp4 4 a b := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [interp4]
  · simp [interp4]
  · simp only [interp4, Int.sub_mul, Int.add_mul, Int.one_mul]
    omega
  · simp only [interp4]
    omega

-- ============================================================================
-- Theorem 2 (E2): the interpolated stage is frame-independent.
-- Rescaling and translating both endpoint drawings by the same similarity
-- (c, d) rescales and translates the interpolated stage identically, so the
-- Procrustes frame chosen for Fig. 393 cannot change the predicted form.
-- ============================================================================
theorem interp_translate (k x y d : Int) :
    interp4 k (x + d) (y + d) = interp4 k x y + 4 * d := by
  simp only [interp4, Int.mul_add, Int.sub_mul]
  omega

theorem interp_scale (k x y c : Int) :
    interp4 k (c * x) (c * y) = c * interp4 k x y := by
  simp only [interp4]
  rw [Int.mul_add, Int.mul_left_comm c (4 - k) x, Int.mul_left_comm c k y]

theorem interp_similarity_equivariant (k a b c d : Int) :
    interp4 k (c * a + d) (c * b + d) = c * interp4 k a b + 4 * d := by
  rw [interp_translate, interp_scale]

-- ============================================================================
-- Theorem 3 (E1 at every stage, 1-D form): no folding of the interpolated grid.
-- If a correspondence is order-preserving at both ends (a1 < a2 and b1 < b2 —
-- the 1-D statement that both grids have positive Jacobian), every intermediate
-- stage 0 ≤ k ≤ 4 is order-preserving as well: the interpolated co-ordinate
-- systems of Fig. 390 are genuine (non-folded) grids, and every interpolated
-- co-ordinate lies between the two endpoint co-ordinates.
-- ============================================================================
theorem interp_no_fold (k a1 a2 b1 b2 : Int) (hk0 : 0 ≤ k) (hk4 : k ≤ 4)
    (ha : a1 < a2) (hb : b1 < b2) :
    interp4 k a1 b1 < interp4 k a2 b2 := by
  have hA : 0 ≤ (4 - k) * (a2 - a1) := Int.mul_nonneg (by omega) (by omega)
  have hB : 0 ≤ k * (b2 - b1) := Int.mul_nonneg hk0 (by omega)
  have hpos : 0 < (4 - k) * (a2 - a1) + k * (b2 - b1) := by
    rcases Int.lt_or_le 0 k with hkpos | hkz
    · have : 0 < k * (b2 - b1) := Int.mul_pos hkpos (by omega)
      omega
    · have hk : k = 0 := by omega
      subst hk
      have : 0 < (4 - 0) * (a2 - a1) := Int.mul_pos (by decide) (by omega)
      omega
  have hexp : interp4 k a2 b2 - interp4 k a1 b1 = (4 - k) * (a2 - a1) + k * (b2 - b1) := by
    simp only [interp4, Int.mul_sub]
    omega
  omega

theorem interp_between (k a b : Int) (hk0 : 0 ≤ k) (hk4 : k ≤ 4) (hab : a ≤ b) :
    4 * a ≤ interp4 k a b ∧ interp4 k a b ≤ 4 * b := by
  have h1 : 0 ≤ k * (b - a) := Int.mul_nonneg hk0 (by omega)
  have h2 : 0 ≤ (4 - k) * (b - a) := Int.mul_nonneg (by omega) (by omega)
  simp only [interp4, Int.sub_mul, Int.mul_sub] at *
  omega

-- ============================================================================
-- Theorem 4 (E7): soundness of the best-fit stage statistic.
-- If a drawn intermediate D is exactly the k-th interpolated stage of (A, B),
-- then the closed-form projection numerator 4 <D - A, B - A> equals k |B - A|^2,
-- i.e. t* = k/4 exactly (2-D, coordinates (Ax, Ay) etc.).  The script's
-- `best_fit_t` therefore recovers the nominal stage for a perfect interpolant,
-- so a non-zero equidistance residual |t* - k/4| measures the drawing, not the test.
-- ============================================================================
theorem best_fit_stage_sound (k Ax Ay Bx By Dx Dy : Int)
    (hx : 4 * Dx = interp4 k Ax Bx) (hy : 4 * Dy = interp4 k Ay By) :
    4 * ((Dx - Ax) * (Bx - Ax) + (Dy - Ay) * (By - Ay)) =
      k * ((Bx - Ax) * (Bx - Ax) + (By - Ay) * (By - Ay)) := by
  have ex : 4 * (Dx - Ax) = k * (Bx - Ax) := by
    simp only [interp4, Int.sub_mul] at hx
    simp only [Int.mul_sub]
    omega
  have ey : 4 * (Dy - Ay) = k * (By - Ay) := by
    simp only [interp4, Int.sub_mul] at hy
    simp only [Int.mul_sub]
    omega
  calc 4 * ((Dx - Ax) * (Bx - Ax) + (Dy - Ay) * (By - Ay))
      = (4 * (Dx - Ax)) * (Bx - Ax) + (4 * (Dy - Ay)) * (By - Ay) := by
        rw [Int.mul_add, Int.mul_assoc, Int.mul_assoc]
    _ = (k * (Bx - Ax)) * (Bx - Ax) + (k * (By - Ay)) * (By - Ay) := by
        rw [ex, ey]
    _ = k * ((Bx - Ax) * (Bx - Ax) + (By - Ay) * (By - Ay)) := by
        rw [Int.mul_add, Int.mul_assoc, Int.mul_assoc]

-- ============================================================================
-- Theorem 5 (E5): Thompson's "simple" affine transformations under interpolation.
-- (i) endpoints: det at k = 0 is 16 = 4^2 det I, det at k = 4 is 16 det A;
-- (ii) a pure shear A = [[1, s], [0, 1]] keeps det = 16 at EVERY stage (area is
--      preserved along the whole shear interpolation);
-- (iii) a pure scaling A = diag(p, q) with p, q ≥ 1 has strictly positive
--      determinant at every stage k ≥ 0 (orientation is never reversed).
-- ============================================================================
theorem four_mul_four_mul (x y : Int) : (4 * x) * (4 * y) = 16 * (x * y) := by
  rw [Int.mul_assoc, Int.mul_left_comm x 4 y, ← Int.mul_assoc]
  rfl

theorem affine_interp_det_endpoints (a11 a12 a21 a22 : Int) :
    interpDet 0 a11 a12 a21 a22 = 16 ∧
    interpDet 4 a11 a12 a21 a22 = 16 * (a11 * a22 - a12 * a21) := by
  refine ⟨?_, ?_⟩
  · simp [interpDet, interpMat11, interpMat12, interpMat21, interpMat22]
  · have h11 : interpMat11 4 a11 = 4 * a11 := by simp [interpMat11]
    have h22 : interpMat22 4 a22 = 4 * a22 := by simp [interpMat22]
    have h12 : interpMat12 4 a12 = 4 * a12 := by simp [interpMat12]
    have h21 : interpMat21 4 a21 = 4 * a21 := by simp [interpMat21]
    unfold interpDet
    rw [h11, h22, h12, h21, four_mul_four_mul, four_mul_four_mul, Int.mul_sub]

theorem shear_interp_preserves_area (k s : Int) :
    interpDet k 1 s 0 1 = 16 := by
  simp [interpDet, interpMat11, interpMat12, interpMat21, interpMat22]

theorem scaling_interp_orientation_preserving (k p q : Int)
    (hk0 : 0 ≤ k) (hp : 1 ≤ p) (hq : 1 ≤ q) :
    0 < interpDet k p 0 0 q := by
  have hkp : k ≤ k * p := by
    have := Int.mul_le_mul_of_nonneg_left hp hk0
    simpa using this
  have hkq : k ≤ k * q := by
    have := Int.mul_le_mul_of_nonneg_left hq hk0
    simpa using this
  have h11 : 0 < interpMat11 k p := by
    simp only [interpMat11]; omega
  have h22 : 0 < interpMat22 k q := by
    simp only [interpMat22]; omega
  have hprod : 0 < interpMat11 k p * interpMat22 k q := Int.mul_pos h11 h22
  simp only [interpDet, interpMat12, interpMat21, Int.mul_zero, Int.sub_zero]
  exact hprod

end MathOfBiology.Loop_243
