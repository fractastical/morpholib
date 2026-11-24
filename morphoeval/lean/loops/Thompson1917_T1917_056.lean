/-
  morphoeval/lean/loops/Thompson1917_T1917_056.lean
  Slot 245 · thompson1917-t1917-056 · Thompson (1917) On Growth and Form, Ch. XVII,
  pp. 765-767, Figs. 401-402: Eohippus (Hyracotherium) → Equus skull coordinate deformation.

  Governing equations (datasets/thompson_1917/t1917-056/..._reference.json):
    E2  phi_t(x) = (1 - t) x + t phi(x),  t = k/4 on Heilmann intermediate stages
    E5  affine: phi_t(x) = ((1-t) I + t A) x + t b
    E6  landmark correspondence: phi_t(a_i) = (1 - t) a_i + t b_i
-/
namespace MathOfBiology.Loop_245

def interp4 (k a b : Int) : Int := (4 - k) * a + k * b

def sq (x : Int) : Int := x * x

def sq_dist (x1 y1 x2 y2 : Int) : Int :=
  sq (x2 - x1) + sq (y2 - y1)

def rot_point (r00 r01 r10 r11 x y : Int) : Int × Int :=
  (r00 * x + r01 * y, r10 * x + r11 * y)

theorem procrustes_rotation_preserves_sq_dist
    (r00 r01 r10 r11 x1 y1 x2 y2 : Int)
    (h_orth : r00 * r00 + r10 * r10 = 1 ∧ r01 * r01 + r11 * r11 = 1 ∧
              r00 * r01 + r10 * r11 = 0) :
    let (x1', y1') := rot_point r00 r01 r10 r11 x1 y1
    let (x2', y2') := rot_point r00 r01 r10 r11 x2 y2
    sq_dist x1' y1' x2' y2' = sq_dist x1 y1 x2 y2 := by
  rcases h_orth with ⟨h00, h11, h01⟩
  dsimp [sq_dist, sq, rot_point]
  have hdx : x2' - x1' = r00 * (x2 - x1) + r01 * (y2 - y1) := by dsimp; ring
  have hdy : y2' - y1' = r10 * (x2 - x1) + r11 * (y2 - y1) := by dsimp; ring
  rw [hdx, hdy]
  have hsum :
      (r00 * (x2 - x1) + r01 * (y2 - y1)) * (r00 * (x2 - x1) + r01 * (y2 - y1)) +
        (r10 * (x2 - x1) + r11 * (y2 - y1)) * (r10 * (x2 - x1) + r11 * (y2 - y1)) =
      (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) := by
    ring_nf; rw [h00, h11, h01]; ring
  exact hsum

theorem interp_endpoints_and_equidistant (k a b : Int) :
    interp4 0 a b = 4 * a ∧
    interp4 4 a b = 4 * b ∧
    interp4 (k + 1) a b - interp4 k a b = b - a ∧
    2 * interp4 2 a b = interp4 0 a b + interp4 4 a b := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [interp4]
  · simp [interp4]
  · simp only [interp4, Int.sub_mul, Int.add_mul, Int.one_mul]; omega
  · simp only [interp4]; omega

theorem similarity_scale_sq_dist (k x1 y1 x2 y2 : Int) (hk : k ≠ 0) :
    sq_dist (k * x1) (k * y1) (k * x2) (k * y2) = (k * k) * sq_dist x1 y1 x2 y2 := by
  dsimp [sq_dist, sq]
  have hx : k * x2 - k * x1 = k * (x2 - x1) := by ring
  have hy : k * y2 - k * y1 = k * (y2 - y1) := by ring
  rw [hx, hy]
  have h1 : (k * (x2 - x1)) * (k * (x2 - x1)) = (k * k) * ((x2 - x1) * (x2 - x1)) := by ring
  have h2 : (k * (y2 - y1)) * (k * (y2 - y1)) = (k * k) * ((y2 - y1) * (y2 - y1)) := by ring
  rw [h1, h2]; ring

end MathOfBiology.Loop_245
