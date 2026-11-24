/-
  morphoeval/lean/loops/Thompson1917_T1917_035.lean
  Formal Lean 4 verification for Thompson, D.W. (1917) On Growth and Form, Ch. XV
  pp. 652-669, Fig. 328 (claim T1917-035, slot 229, `thompson1917-t1917-035`):

    "Certain hollow structures (e.g. sea-urchin tests diagrammed in Fig. 328, skulls and other
     shells) take forms explicable as membranes/shells under pressure and tension."

  Axisymmetric meridian  y(x) = p(x) sqrt(1 - x^2),  p(x) = c0 + c1 x + c2 x^2 + c3 x^3,
  Thompson envelope  p_n + T (1/r + 1/r') = P  (p. 658).  Zero imports, zero sorry.
-/

namespace MathOfBiology.Loop_229

def prestonPoly (c0 c1 c2 c3 x : Int) : Int :=
  c0 + c1 * x + c2 * (x * x) + c3 * (x * x * x)

def prestonHalfBreadthSq (c0 c1 c2 c3 x : Int) : Int :=
  prestonPoly c0 c1 c2 c3 x * prestonPoly c0 c1 c2 c3 x * (1 - x * x)

def poleRadius (c0 c1 c2 c3 s : Int) : Int :=
  prestonPoly c0 c1 c2 c3 s * prestonPoly c0 c1 c2 c3 s

theorem preston_pole_symmetry_and_sphere (c0 c1 c2 c3 x : Int) :
    prestonPoly c0 c1 c2 c3 1 - prestonPoly c0 c1 c2 c3 (-1) = 2 * (c1 + c3) ∧
    (c1 + c3 = 0 → poleRadius c0 c1 c2 c3 1 = poleRadius c0 c1 c2 c3 (-1)) ∧
    poleRadius c0 (-c1) c2 (-c3) 1 = poleRadius c0 c1 c2 c3 (-1) ∧
    prestonPoly 1 0 0 0 x = 1 ∧
    prestonHalfBreadthSq 1 0 0 0 x + x * x = 1 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp [prestonPoly]; omega
  · intro h
    have hp : prestonPoly c0 c1 c2 c3 1 = prestonPoly c0 c1 c2 c3 (-1) := by
      simp [prestonPoly]; omega
    simp [poleRadius, hp]
  · simp [poleRadius, prestonPoly]
  · simp [prestonPoly]
  · simp [prestonHalfBreadthSq, prestonPoly]

def prestonVolume315 (c0 c1 c2 c3 : Int) : Int :=
  105 * (c0 * c0) + 21 * (c1 * c1 + 2 * (c0 * c2)) + 9 * (c2 * c2 + 2 * (c1 * c3)) + 5 * (c3 * c3)

theorem preston_volume_sphere_scaling (c0 c1 c2 c3 : Int) :
    prestonVolume315 1 0 0 0 = 105 ∧
    prestonVolume315 (2 * c0) 0 0 0 = 4 * prestonVolume315 c0 0 0 0 ∧
    prestonVolume315 c0 (-c1) c2 (-c3) = prestonVolume315 c0 c1 c2 c3 := by
  refine ⟨by decide, ?_, ?_⟩
  · simp [prestonVolume315, Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
  · simp [prestonVolume315, Int.neg_mul, Int.mul_neg, Int.neg_neg]

def laplaceResidual (pn T r r' P : Int) : Int :=
  pn * (r * r') + T * (r + r') - P * (r * r')

def poleTwiceTension (P R : Int) : Int :=
  P * R

theorem thompson_envelope_pole_equilibrium (T R P Rb Rp : Int)
    (hR : R ≠ 0) (hP : 0 < P) (hRb : Rp < Rb) :
    laplaceResidual 0 T R R P = T * (R + R) - P * (R * R) ∧
    (laplaceResidual 0 T R R P = 0 → P * R = 2 * T) ∧
    poleTwiceTension P Rp < poleTwiceTension P Rb := by
  refine ⟨?_, ?_, ?_⟩
  · simp [laplaceResidual]
  · intro h
    have h' : T * (R + R) = P * (R * R) := by
      simp [laplaceResidual] at h; omega
    apply Int.eq_of_mul_eq_mul_right hR
    rw [Int.mul_assoc, ← h', Int.mul_add, Int.two_mul, Int.add_mul]
  · exact Int.mul_lt_mul_of_pos_left hRb hP

def coneBalance (p a p' a' : Int) : Prop :=
  p * a = p' * a'

theorem thompson_cone_pressure_order (p a p' a' : Int)
    (hbal : coneBalance p a p' a') (ha : 0 ≤ a) (hp' : 0 < p') (hlt : a' < a) :
    p < p' := by
  refine Decidable.byContradiction fun hcon => ?_
  have hge : p' ≤ p := Int.not_lt.mp hcon
  have h1 : p' * a ≤ p * a := Int.mul_le_mul_of_nonneg_right hge ha
  have h2 : p' * a' < p' * a := Int.mul_lt_mul_of_pos_left hlt hp'
  have h3 : p' * a' < p * a := Int.lt_of_lt_of_le h2 h1
  unfold coneBalance at hbal; rw [hbal] at h3; exact Int.lt_irrefl _ h3

def shellBreadth (c0 k : Int) : Int := 2 * (c0 * k)
def shellLength (k : Int) : Int := 2 * k

theorem shell_oblate_member_midpoint_max (c0 k x : Int)
    (hc : c0 ≤ 1) (hk : 0 ≤ k) (hx : 0 ≤ x * x) (hc0 : 0 ≤ c0 * c0) :
    shellBreadth c0 k ≤ shellLength k ∧
    prestonHalfBreadthSq c0 0 0 0 x ≤ c0 * c0 ∧
    prestonHalfBreadthSq c0 0 0 0 0 = c0 * c0 := by
  refine ⟨?_, ?_, ?_⟩
  · unfold shellBreadth shellLength
    have h1 : c0 * k ≤ 1 * k := Int.mul_le_mul_of_nonneg_right hc hk
    omega
  · simp [prestonHalfBreadthSq, prestonPoly, Int.mul_sub]
    have h1 : 0 ≤ c0 * c0 * (x * x) := Int.mul_nonneg hc0 hx
    omega
  · simp [prestonHalfBreadthSq, prestonPoly]

end MathOfBiology.Loop_229
