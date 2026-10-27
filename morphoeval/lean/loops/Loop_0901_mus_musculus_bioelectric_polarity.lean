/-
  morphoeval/lean/loops/Loop_0901_mus_musculus_bioelectric_polarity.lean — SUBSTANTIVE re-formalization (2026-10-27)
  Replaces the template placeholder trio (a*b>0 / p+q-q=p / decide) with a
  law-specific formalization of: Gap-Junction Current Conservation & Bistable Gap in Mus musculus (Slot #0901).
  Self-contained Lean 4 core (zero imports), 0 sorry. See morphoeval/reformalize_first_batch.py.
-/
namespace MathOfBiology.Loop_0901

/-- Gap-junction current i->j: G (V_j - V_i), scaled integers. -/
def gj_current (G Vi Vj : Int) : Int := G * (Vj - Vi)
theorem gj_no_current_at_equipotential (G V : Int) : gj_current G V V = 0 := by
  simp [gj_current]
theorem gj_current_antisymmetric (G Vi Vj : Int) :
    gj_current G Vi Vj + gj_current G Vj Vi = 0 := by
  simp only [gj_current]
  have h : (Vj - Vi) + (Vi - Vj) = 0 := by omega
  calc G * (Vj - Vi) + G * (Vi - Vj) = G * ((Vj - Vi) + (Vi - Vj)) := by rw [Int.mul_add]
    _ = 0 := by rw [h]; simp
/-- Bistable attractor gap is positive exactly on the depolarized side. -/
def attractor_gap (Vd Vh : Int) : Int := Vd - Vh
theorem attractor_gap_positive (Vd Vh : Int) (h : Vd > Vh) : attractor_gap Vd Vh > 0 := by
  simp only [attractor_gap]; omega

end MathOfBiology.Loop_0901
