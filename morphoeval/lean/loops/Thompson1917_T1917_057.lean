/-
  morphoeval/lean/loops/Thompson1917_T1917_057.lean
  Slot 246 · thompson1917-t1917-057 · Thompson (1917) On Growth and Form, Ch. XVII,
  p. 769, Fig. 403: human scapulae (after Dwight) comparable by the coordinate method.

  Governing equations (datasets/thompson_1917/t1917-057/..._reference.json):
    E2  phi_t(x) = (1 - t) x + t phi(x)
    E5  affine case: phi_t(x) = ((1-t) I + t A) x + t b
    E6  corresponding landmarks: phi_t(a_i) = (1 - t) a_i + t b_i

  Stated over Int with stage t = k/4 cleared of denominators (same algebra as T1917-054).
  Self-contained Lean 4 core (no Mathlib, no imports, 0 sorry).
-/
namespace MathOfBiology.Loop_246

def interp4 (k a b : Int) : Int := (4 - k) * a + k * b

def interpMat11 (k a11 : Int) : Int := (4 - k) + k * a11
def interpMat12 (k a12 : Int) : Int := k * a12
def interpMat21 (k a21 : Int) : Int := k * a21
def interpMat22 (k a22 : Int) : Int := (4 - k) + k * a22

def interpDet (k a11 a12 a21 a22 : Int) : Int :=
  interpMat11 k a11 * interpMat22 k a22 - interpMat12 k a12 * interpMat21 k a21

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

theorem interp_no_fold (k a1 a2 b1 b2 : Int) (hk0 : 0 ≤ k) (hk4 : k ≤ 4)
    (ha : a1 < a2) (hb : b1 < b2) :
    interp4 k a1 b1 < interp4 k a2 b2 := by
  have hpos : 0 < (4 - k) * (a2 - a1) + k * (b2 - b1) := by
    rcases Int.lt_or_le 0 k with hkpos | hkz
    · have : 0 < k * (b2 - b1) := Int.mul_pos hkpos (by omega)
      omega
    · subst hkz; have : 0 < 4 * (a2 - a1) := Int.mul_pos (by decide) (by omega); omega
  have hexp : interp4 k a2 b2 - interp4 k a1 b1 = (4 - k) * (a2 - a1) + k * (b2 - b1) := by
    simp only [interp4, Int.mul_sub]; omega
  omega

theorem best_fit_stage_sound (k Ax Ay Bx By Dx Dy : Int)
    (hx : 4 * Dx = interp4 k Ax Bx) (hy : 4 * Dy = interp4 k Ay By) :
    4 * ((Dx - Ax) * (Bx - Ax) + (Dy - Ay) * (By - Ay)) =
      k * ((Bx - Ax) * (Bx - Ax) + (By - Ay) * (By - Ay)) := by
  have ex : 4 * (Dx - Ax) = k * (Bx - Ax) := by simp only [interp4, Int.sub_mul] at hx; simp only [Int.mul_sub]; omega
  have ey : 4 * (Dy - Ay) = k * (By - Ay) := by simp only [interp4, Int.sub_mul] at hy; simp only [Int.mul_sub]; omega
  calc 4 * ((Dx - Ax) * (Bx - Ax) + (Dy - Ay) * (By - Ay))
      = (4 * (Dx - Ax)) * (Bx - Ax) + (4 * (Dy - Ay)) * (By - Ay) := by rw [Int.mul_add, Int.mul_assoc, Int.mul_assoc]
    _ = (k * (Bx - Ax)) * (Bx - Ax) + (k * (By - Ay)) * (By - Ay) := by rw [ex, ey]
    _ = k * ((Bx - Ax) * (Bx - Ax) + (By - Ay) * (By - Ay)) := by rw [Int.mul_add, Int.mul_assoc, Int.mul_assoc]

theorem shear_interp_preserves_area (k s : Int) :
    interpDet k 1 s 0 1 = 16 := by
  simp [interpDet, interpMat11, interpMat12, interpMat21, interpMat22]

end MathOfBiology.Loop_246
