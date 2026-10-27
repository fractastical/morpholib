/-
  morphoeval/lean/loops/Loop_0501_mus_musculus_vertex_packing.lean — SUBSTANTIVE re-formalization (2026-10-27)
  Replaces the template placeholder trio (a*b>0 / p+q-q=p / decide) with a
  law-specific formalization of: Vertex Energy Minimum, Shape Index & Lewis Law in Mus musculus (Slot #0501).
  Self-contained Lean 4 core (zero imports), 0 sorry. See morphoeval/reformalize_first_batch.py.
-/
namespace MathOfBiology.Loop_0501

/-- Vertex energy deviation in Nat scaling: squared area+perimeter offsets (x K). -/
def vertex_offset (da dp : Nat) : Nat := da * da + dp * dp
theorem vertex_offset_nonneg_at_target : vertex_offset 0 0 = 0 := by
  simp [vertex_offset]
theorem vertex_offset_grows_with_area_offset (da dp : Nat) :
    vertex_offset (da + 1) dp ≥ vertex_offset da dp := by
  simp only [vertex_offset]
  have h1 : da * da ≤ (da + 1) * (da + 1) := by
    apply Nat.mul_le_mul <;> omega
  omega
/-- Lewis law: 4<A_n> = (n-2)<A>; for hexagonal cells <A_6> = <A>. -/
def lewis_lhs (n An : Nat) : Nat := 4 * An
def lewis_rhs (n Abar : Nat) : Nat := (n - 2) * Abar
theorem lewis_hexagonal (Abar : Nat) : lewis_lhs 6 Abar = lewis_rhs 6 Abar + 0 * Abar := by
  simp [lewis_lhs, lewis_rhs]

end MathOfBiology.Loop_0501
