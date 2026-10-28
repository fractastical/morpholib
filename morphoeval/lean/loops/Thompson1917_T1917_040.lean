/-
  morphoeval/lean/loops/Thompson1917_T1917_040.lean
  Thompson (1917) Ch. XVII Figs. 353-358 / Dürer-Camper profile grids — coordinate deformation algebra.

  Theorem 1: orthogonal rotation preserves squared Euclidean distance (Procrustes step).
  Theorem 2: uniform vertical scale y' = k y leaves x unchanged (Fig. 353 stretch).
  Theorem 3: horizontal shear (x, y) ↦ (x + m y, y) preserves y-coordinates (parallel to Frankfort line).
-/

namespace MathOfBiology.Loop_232

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
    ring_nf
    rw [h00, h11, h01]
    ring
  exact hsum

def vert_scale (k x y : Int) : Int × Int := (x, k * y)

theorem vertical_stretch_preserves_x (k x y : Int) :
    (vert_scale k x y).1 = x := by
  dsimp [vert_scale]

def horiz_shear (m x y : Int) : Int × Int := (x + m * y, y)

theorem shear_preserves_y (m x1 y1 x2 y2 : Int) :
    let (x1', y1') := horiz_shear m x1 y1
    let (x2', y2') := horiz_shear m x2 y2
    y2' - y1' = y2 - y1 := by
  dsimp [horiz_shear]
  ring

end MathOfBiology.Loop_232
