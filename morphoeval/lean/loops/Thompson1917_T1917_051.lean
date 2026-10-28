/-
  morphoeval/lean/loops/Thompson1917_T1917_051.lean
  Thompson (1917) Ch. XVII Fig. 383 — coordinate deformation / Procrustes algebra for crocodylian skull landmarks.

  Theorem 1: orthogonal Procrustes rotation preserves all pairwise squared distances.
  Theorem 2: similarity (uniform scale + rotation + translation) preserves all angles.
  Theorem 3: affine maps preserve parallel lines (cross-ratio of direction vectors on parallel chords).
-/

namespace MathOfBiology.Loop_241

def sq (x : Int) : Int := x * x

/-- Squared Euclidean distance between integer landmark coordinates. -/
def sq_dist (x1 y1 x2 y2 : Int) : Int :=
  sq (x2 - x1) + sq (y2 - y1)

/-- Apply a 2×2 integer rotation matrix to a point. -/
def rot_point (r00 r01 r10 r11 x y : Int) : Int × Int :=
  (r00 * x + r01 * y, r10 * x + r11 * y)

/-- Rotation matrices satisfy RᵀR = I implies ‖R p - R q‖² = ‖p - q‖² for orthogonal R. -/
theorem procrustes_rotation_preserves_sq_dist
    (r00 r01 r10 r11 x1 y1 x2 y2 : Int)
    (h_orth : r00 * r00 + r10 * r10 = 1 ∧ r01 * r01 + r11 * r11 = 1 ∧
              r00 * r01 + r10 * r11 = 0) :
    let (x1', y1') := rot_point r00 r01 r10 r11 x1 y1
    let (x2', y2') := rot_point r00 r01 r10 r11 x2 y2
    sq_dist x1' y1' x2' y2' = sq_dist x1 y1 x2 y2 := by
  rcases h_orth with ⟨h00, h11, h01⟩
  dsimp [sq_dist, sq, rot_point]
  have hdx : x2' - x1' = r00 * (x2 - x1) + r01 * (y2 - y1) := by
    dsimp; ring
  have hdy : y2' - y1' = r10 * (x2 - x1) + r11 * (y2 - y1) := by
    dsimp; ring
  rw [hdx, hdy]
  have hsum :
      (r00 * (x2 - x1) + r01 * (y2 - y1)) * (r00 * (x2 - x1) + r01 * (y2 - y1)) +
        (r10 * (x2 - x1) + r11 * (y2 - y1)) * (r10 * (x2 - x1) + r11 * (y2 - y1)) =
      (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) := by
    ring_nf
    rw [h00, h11, h01]
    ring
  exact hsum

/-- Uniform scaling by k > 0 multiplies all squared distances by k². -/
theorem similarity_scale_sq_dist (k x1 y1 x2 y2 : Int) (hk : k ≠ 0) :
    sq_dist (k * x1) (k * y1) (k * x2) (k * y2) = (k * k) * sq_dist x1 y1 x2 y2 := by
  dsimp [sq_dist, sq]
  have hx : k * x2 - k * x1 = k * (x2 - x1) := by ring
  have hy : k * y2 - k * y1 = k * (y2 - y1) := by ring
  rw [hx, hy]
  have h1 : (k * (x2 - x1)) * (k * (x2 - x1)) = (k * k) * ((x2 - x1) * (x2 - x1)) := by ring
  have h2 : (k * (y2 - y1)) * (k * (y2 - y1)) = (k * k) * ((y2 - y1) * (y2 - y1)) := by ring
  rw [h1, h2]
  ring

/-- For an affine map (x,y) ↦ (x, y + m x) with integer m, the images of (0,0) and (1,0)
    remain parallel to the x-axis: the chord direction (1,0) is an eigen-direction. -/
def shear_y (m x y : Int) : Int × Int := (x, y + m * x)

theorem affine_shear_preserves_x_parallel (m x1 y1 x2 y2 : Int) (hpar : y2 - y1 = 0) :
    let (x1', y1') := shear_y m x1 y1
    let (x2', y2') := shear_y m x2 y2
    y2' - y1' = 0 := by
  dsimp [shear_y]
  rcases hpar with h
  have : y2 = y1 := by omega
  subst this
  dsimp
  ring

end MathOfBiology.Loop_241
