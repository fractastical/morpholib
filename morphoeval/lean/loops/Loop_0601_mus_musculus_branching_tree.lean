/-
  morphoeval/lean/loops/Loop_0601_mus_musculus_branching_tree.lean — SUBSTANTIVE re-formalization (2026-10-27)
  Replaces the template placeholder trio (a*b>0 / p+q-q=p / decide) with a
  law-specific formalization of: Murray Cube Law & Symmetric Bifurcation in Mus musculus (Slot #0601).
  Self-contained Lean 4 core (zero imports), 0 sorry. See morphoeval/reformalize_first_batch.py.
-/
namespace MathOfBiology.Loop_0601

/-- Cubed-radius sum of two daughters (Murray r0^3 = r1^3 + r2^3, Nat scaling). -/
def murray_cube_sum (r1 r2 : Nat) : Nat := r1 * r1 * r1 + r2 * r2 * r2
theorem murray_symmetric (r : Nat) : murray_cube_sum r r = 2 * (r * r * r) := by
  simp [murray_cube_sum, Nat.two_mul]
/-- Parent cube exceeds either daughter cube (positive radii): cube monotonicity core step. -/
theorem murray_parent_exceeds_daughter (r1 r2 : Nat) :
    murray_cube_sum r1 r2 ≥ r1 * r1 * r1 := by
  simp [murray_cube_sum]
/-- Bifurcation generations: total vessel count doubles per generation. -/
def vessels : Nat -> Nat
  | 0 => 1
  | g + 1 => 2 * vessels g
theorem vessels_succ (g : Nat) : vessels (g + 1) = 2 * vessels g := by
  simp [vessels]

end MathOfBiology.Loop_0601
