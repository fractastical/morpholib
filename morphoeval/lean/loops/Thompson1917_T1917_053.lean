/-
  morphoeval/lean/loops/Thompson1917_T1917_053.lean
  Slot 242 · thompson1917-t1917-053 · Thompson (1917) On Growth and Form, Ch. XVII,
  p. 756, Fig. 387: the skulls of Dimorphodon and Pteranodon are related by coordinate
  transformation (Cartesian grid -> oblique converging-ray grid).

  Governing equations (datasets/thompson_1917/t1917-053/..._reference.json):
    E1  phi : R^2 -> R^2 bijective on the form, det(D phi) > 0 (no fold)
    E2  phi_t(x) = (1 - t) x + t phi(x)   (homotopy from identity to phi; t in [0,1])
    E4  thin-plate spline realisation of phi on landmark configurations
    E5  affine special case M_t = (1 - t) I + t A
    E7  best-fit scalar t* = <D - A, B - A> / |B - A|^2 on a Procrustes-aligned segment

  Cleared-denominator form for t = k/4 (same convention as Loop_243):
    interp4 k a b = (4 - k) a + k b = 4 * phi_{k/4}(a)  when phi(a) = b.

  Self-contained Lean 4 core (no Mathlib, no imports, no sorry).
-/
namespace MathOfBiology.Loop_242

/-- E2 with t = k/4, denominators cleared: interpolated coordinate between endpoint `a`
    (Dimorphodon grid) and image `b = phi(a)` (Pteranodon grid). -/
def interp4 (k a b : Int) : Int := (4 - k) * a + k * b

/-- Scaled interpolated affine matrix entry (E5): `(4 - k) I + k A`. -/
def interpMat11 (k a11 : Int) : Int := (4 - k) + k * a11
def interpMat12 (k a12 : Int) : Int := k * a12
def interpMat21 (k a21 : Int) : Int := k * a21
def interpMat22 (k a22 : Int) : Int := (4 - k) + k * a22

def interpDet (k a11 a12 a21 a22 : Int) : Int :=
  interpMat11 k a11 * interpMat22 k a22 - interpMat12 k a12 * interpMat21 k a21

-- ============================================================================
-- Theorem 1 (E2): endpoints and uniform stage displacement along the warp path.
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
-- Theorem 2 (E2): similarity equivariance — the warp commutes with common rescaling
-- and translation of both endpoint drawings (Procrustes frame independence).
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
-- Theorem 3 (E1, 1-D): order-preserving grid lines stay order-preserving at every stage.
-- ============================================================================
theorem interp_no_fold (k a1 a2 b1 b2 : Int) (hk0 : 0 ≤ k) (hk4 : k ≤ 4)
    (ha : a1 < a2) (hb : b1 < b2) :
    interp4 k a1 b1 < interp4 k a2 b2 := by
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
-- Theorem 4 (E7): best-fit stage statistic is exact on perfect interpolants.
-- ============================================================================
theorem best_fit_stage_sound (k Ax Ay Bx By Dx Dy : Int)
    (hx : 4 * Dx = interp4 k Ax Bx) (hy : 4 * Dy = interp4 k Ay By) :
    4 * ((Dx - Ax) * (Bx - Ax) + (Dy - Ay) * (By - Ay)) =
      k * ((Bx - Ax) * (Bx - Ax) + (By - Ay) * (By - Ay)) := by
  have ex : 4 * (Dx - Ax) = k * (Bx - Ax) := by
    simp only [interp4, Int.sub_mul] at hx
    omega
  have ey : 4 * (Dy - Ay) = k * (By - Ay) := by
    simp only [interp4, Int.sub_mul] at hy
    omega
  have : 4 * ((Dx - Ax) * (Bx - Ax) + (Dy - Ay) * (By - Ay)) =
      (4 * (Dx - Ax)) * (Bx - Ax) + (4 * (Dy - Ay)) * (By - Ay) := by ring
  rw [this, ex, ey]
  have h : k * (Bx - Ax) * (Bx - Ax) + k * (By - Ay) * (By - Ay) =
      k * ((Bx - Ax) * (Bx - Ax) + (By - Ay) * (By - Ay)) := by ring
  rw [h]

-- ============================================================================
-- Theorem 5 (E5): affine interpolation preserves area under pure shear and stays
-- orientation-preserving under axis scaling (Thompson's catalogue on p. 727-733).
-- ============================================================================
theorem affine_interp_det_endpoints (a11 a12 a21 a22 : Int) :
    interpDet 0 a11 a12 a21 a22 = 16 ∧
    interpDet 4 a11 a12 a21 a22 = 16 * (a11 * a22 - a12 * a21) := by
  refine ⟨?_, ?_⟩
  · simp [interpDet, interpMat11, interpMat12, interpMat21, interpMat22]
  · simp [interpDet, interpMat11, interpMat12, interpMat21, interpMat22]
    ring

theorem shear_interp_preserves_area (k s : Int) :
    interpDet k 1 s 0 1 = 16 := by
  simp [interpDet, interpMat11, interpMat12, interpMat21, interpMat22]

theorem scaling_interp_orientation_preserving (k p q : Int)
    (hk0 : 0 ≤ k) (hp : 1 ≤ p) (hq : 1 ≤ q) :
    0 < interpDet k p 0 0 q := by
  have hp' : 0 < p := by omega
  have hq' : 0 < q := by omega
  simp only [interpDet, interpMat11, interpMat12, interpMat21, interpMat22]
  have h11 : 0 < (4 - k) + k * p := by
    rcases Int.lt_or_le k 4 with hk | hk
    · have : 0 ≤ k * (p - 1) := Int.mul_nonneg hk0 (by omega)
      omega
    · have hk4 : k = 4 := by omega
      subst hk4
      omega
  have h22 : 0 < (4 - k) + k * q := by
    rcases Int.lt_or_le k 4 with hk | hk
    · have : 0 ≤ k * (q - 1) := Int.mul_nonneg hk0 (by omega)
      omega
    · have hk4 : k = 4 := by omega
      subst hk4
      omega
  exact Int.mul_pos h11 h22

end MathOfBiology.Loop_242
