/-
  morphoeval/lean/loops/Thompson1917_T1917_034.lean
  Formal Lean 4 verification for Thompson, D.W. (1917) On Growth and Form, Ch. XV
  "On the Shapes of Eggs, and of Certain Other Hollow Structures", pp. 652-669
  (claim T1917-034, slot 228, `thompson1917-t1917-034`):

    "Birds' (and other hard-shelled) eggs are solids of revolution whose meridian curves
     form a definite, species-varying family of ovoids, comparable geometrically across
     species."

  The meridian family is written in the Preston (1953) four-parameter form used by
  Biggins, Thompson & Birkhead (2018, Ecol. Evol. 8:9728, Dryad doi:10.5061/dryad.8kv2b20):

      y(x) = p(x) * sqrt(1 - x^2),   p(x) = c0 + c1 x + c2 x^2 + c3 x^3,   -1 <= x <= 1,

  with the axis of revolution along x and the half-length normalised to 1.  Thompson's
  envelope (Laplace) equation  p_n + T (1/r + 1/r') = P  (p. 658) is formalised in
  cleared-denominator integer form.

  All definitions and theorems are self-contained in Lean 4 core (zero imports, no admitted goals).
-/

namespace MathOfBiology.Loop_228

-- ============================================================================
-- Law 1 (`thompson1917-t1917-034-1`):
-- Preston ovoid polynomial, pole radii and the polar-asymmetry criterion
-- ============================================================================
/-- Preston's meridian polynomial `p(x) = c0 + c1 x + c2 x^2 + c3 x^3`.  Near a pole
    `x = ±1` the half-breadth satisfies `y^2 = p(x)^2 (1 - x^2) ≈ 2 p(±1)^2 (1 ∓ x)`, so each
    pole is locally a sphere of radius `R_± = p(±1)^2` (in half-length units).  Hence
    1. the pole polynomial values differ by exactly `2 (c1 + c3)`: the two poles have equal
       osculating radii (Thompson's "symmetrical ends", grebe / pelican / cormorant) iff the
       odd coefficients cancel, `c1 + c3 = 0`;
    2. reflecting the egg end-for-end (`x ↦ -x`, i.e. `c1 ↦ -c1`, `c3 ↦ -c3`) swaps the
       pole radii, so the polar-asymmetry index `max(R₋ ⁄ R₊, R₊ ⁄ R₋)` is a reflection
       invariant and "which end is pointed" carries no shape information;
    3. the unit sphere is the member `c0 = 1, c1 = c2 = c3 = 0`, for which `p ≡ 1` and
       `y^2 + x^2 = 1`. -/
def prestonPoly (c0 c1 c2 c3 x : Int) : Int :=
  c0 + c1 * x + c2 * (x * x) + c3 * (x * x * x)

/-- Squared half-breadth numerator `p(x)^2 (1 - x^2)` of the Preston ovoid (half-length 1). -/
def prestonHalfBreadthSq (c0 c1 c2 c3 x : Int) : Int :=
  prestonPoly c0 c1 c2 c3 x * prestonPoly c0 c1 c2 c3 x * (1 - x * x)

/-- Pole osculating-circle radius `R_s = p(s)^2` at the pole `s = ±1`. -/
def poleRadius (c0 c1 c2 c3 s : Int) : Int :=
  prestonPoly c0 c1 c2 c3 s * prestonPoly c0 c1 c2 c3 s

theorem preston_pole_asymmetry_criterion (c0 c1 c2 c3 x : Int) :
    prestonPoly c0 c1 c2 c3 1 - prestonPoly c0 c1 c2 c3 (-1) = 2 * (c1 + c3) ∧
    (c1 + c3 = 0 → poleRadius c0 c1 c2 c3 1 = poleRadius c0 c1 c2 c3 (-1)) ∧
    poleRadius c0 (-c1) c2 (-c3) 1 = poleRadius c0 c1 c2 c3 (-1) ∧
    prestonPoly 1 0 0 0 x = 1 ∧
    prestonHalfBreadthSq 1 0 0 0 x + x * x = 1 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp [prestonPoly]
    omega
  · intro h
    have hp : prestonPoly c0 c1 c2 c3 1 = prestonPoly c0 c1 c2 c3 (-1) := by
      simp [prestonPoly]
      omega
    simp [poleRadius, hp]
  · simp [poleRadius, prestonPoly]
  · simp [prestonPoly]
  · simp [prestonHalfBreadthSq, prestonPoly]

-- ============================================================================
-- Law 2 (`thompson1917-t1917-034-2`):
-- Exact volume of the Preston solid of revolution and its reflection invariance
-- ============================================================================
/-- The egg is the solid of revolution of `y(x)` about the x-axis, so
    `V = π (L/2)^3 ∫_{-1}^{1} p(x)^2 (1 - x^2) dx`.  Expanding `p^2` and integrating the even
    monomials (`∫ x^{2k}(1 - x^2) dx = 4 / ((2k+1)(2k+3))`) gives the exact closed form
      `315 V / (4 π (L/2)^3) = 105 c0^2 + 21 (c1^2 + 2 c0 c2) + 9 (c2^2 + 2 c1 c3) + 5 c3^2`.
    1. the unit sphere (`c0 = 1`) gives `105`, i.e. `V = (4/315)·105·π = 4π/3`;
    2. the prolate ellipsoid `c1 = c2 = c3 = 0` has `V ∝ c0^2` (doubling the breadth
       quadruples the volume at fixed length);
    3. the volume is invariant under the end-for-end reflection `c1 ↦ -c1, c3 ↦ -c3`:
       the solid of revolution does not care which pole is the pointed one. -/
def prestonVolume315 (c0 c1 c2 c3 : Int) : Int :=
  105 * (c0 * c0) + 21 * (c1 * c1 + 2 * (c0 * c2)) + 9 * (c2 * c2 + 2 * (c1 * c3)) + 5 * (c3 * c3)

theorem preston_volume_sphere_ellipsoid_reflection (c0 c1 c2 c3 : Int) :
    prestonVolume315 1 0 0 0 = 105 ∧
    prestonVolume315 (2 * c0) 0 0 0 = 4 * prestonVolume315 c0 0 0 0 ∧
    prestonVolume315 c0 (-c1) c2 (-c3) = prestonVolume315 c0 c1 c2 c3 := by
  refine ⟨by decide, ?_, ?_⟩
  · simp [prestonVolume315, Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
  · simp [prestonVolume315, Int.neg_mul, Int.mul_neg, Int.neg_neg]

-- ============================================================================
-- Law 3 (`thompson1917-t1917-034-3`):
-- Thompson's envelope equation  p_n + T (1/r + 1/r') = P  at the poles
-- ============================================================================
/-- Thompson (p. 658): "the equation of the envelope is `p_n + T (1/r + 1/r') = P`".
    Multiplying by `r r'` gives the cleared integer form
      `laplaceResidual = p_n r r' + T (r + r') - P r r'`  (= 0 in equilibrium).
    At a pole both principal radii coincide, `r = r' = R`, and with Thompson's assumption
    `p_n = 0` at the poles the residual reduces to `T (R + R) - P R^2 = R (2 T - P R)`, so
    equilibrium with `R ≠ 0` forces `P R = 2 T`.  Consequences:
    1. a uniform normal pressure gives a constant mean curvature, i.e. the sphere
       (Thompson p. 659: "In the case of the spherical egg it is obvious that p_n is
       everywhere equal") — any two points with the same `p_n` and the same `T` have the same
       `R`;
    2. with one internal pressure `P > 0` and `p_n = 0` at both poles, the pole tension
       `T = P R / 2` is *proportional to the pole radius*: the blunt pole (larger `R`)
       carries the larger tension.  Thompson's side-remark that "the tension T is greatest at
       and near the small end" (p. 659) therefore has the sign reversed relative to his own
       equation; the claim under test (shape family) is unaffected. -/
def laplaceResidual (pn T r r' P : Int) : Int :=
  pn * (r * r') + T * (r + r') - P * (r * r')

/-- `2 T` at a pole of radius `R` under internal pressure `P` with `p_n = 0`. -/
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
      simp [laplaceResidual] at h
      omega
    apply Int.eq_of_mul_eq_mul_right hR
    rw [Int.mul_assoc, ← h', Int.mul_add, Int.two_mul, Int.add_mul]
  · exact Int.mul_lt_mul_of_pos_left hRb hP

-- ============================================================================
-- Law 4 (`thompson1917-t1917-034-4`):
-- Thompson's cone-ended-cylinder equilibrium  p cos²θ = p' cos²θ'
-- ============================================================================
/-- Thompson (p. 657): "a cylinder having conical ends, of semi-angles θ and θ'
    respectively, remains in equilibrium in the tube, apart from friction, if
    `p cos²θ = p' cos²θ'`, so that at the more tapered end where θ is small p is small."
    With `a = N cos²θ`, `a' = N cos²θ'` as integer-scaled direction cosines, the more tapered
    end has the *larger* `cos²` (`a' < a`), and the balance `p a = p' a'` with `0 < p'` and
    `0 ≤ a` forces `p < p'`: the external pressure needed at the pointed end is strictly
    smaller than at the blunt end.  This is the sign Thompson needs for his statement that the
    conical egg, blunt end foremost, is "directly favourable to the movement of the egg within
    the oviduct" (p. 658). -/
def coneBalance (p a p' a' : Int) : Prop :=
  p * a = p' * a'

theorem thompson_cone_equilibrium_tapered_end_low_pressure (p a p' a' : Int)
    (hbal : coneBalance p a p' a') (ha : 0 ≤ a) (hp' : 0 < p') (hlt : a' < a) :
    p < p' := by
  refine Decidable.byContradiction fun hcon => ?_
  have hge : p' ≤ p := Int.not_lt.mp hcon
  have h1 : p' * a ≤ p * a := Int.mul_le_mul_of_nonneg_right hge ha
  have h2 : p' * a' < p' * a := Int.mul_lt_mul_of_pos_left hlt hp'
  have h3 : p' * a' < p * a := Int.lt_of_lt_of_le h2 h1
  unfold coneBalance at hbal
  rw [hbal] at h3
  exact Int.lt_irrefl _ h3

-- ============================================================================
-- Law 5 (`thompson1917-t1917-034-5`):
-- Prolate-never-oblate and the Elongation index of the ellipsoid sub-family
-- ============================================================================
/-- Thompson (p. 652): "the egg, while often pointed, is never flattened or discoidal; it is a
    prolate, but never an oblate, spheroid", i.e. Elongation `E = L / B ≥ 1`.  For the
    ellipsoid sub-family `y = c0 sqrt(1 - x^2)` the maximum half-breadth is `c0` (attained at
    the mid-point `x = 0`), so with integer-scaled breadth `B = 2 c0 k` and length `L = 2 k`
    (`k` the half-length in pixels or mm), prolateness `B ≤ L` is exactly `c0 ≤ 1`, and the
    squared half-breadth never exceeds its mid-point value, `c0^2 (1 - x^2) ≤ c0^2` whenever
    `x^2 ≥ 0`.  Both facts are stated and proved for the Preston ellipsoid member. -/
def ellipsoidBreadth (c0 k : Int) : Int :=
  2 * (c0 * k)

def ellipsoidLength (k : Int) : Int :=
  2 * k

theorem thompson_prolate_never_oblate_ellipsoid (c0 k x : Int)
    (hc : c0 ≤ 1) (hk : 0 ≤ k) (hx : 0 ≤ x * x) (hc0 : 0 ≤ c0 * c0) :
    ellipsoidBreadth c0 k ≤ ellipsoidLength k ∧
    prestonHalfBreadthSq c0 0 0 0 x ≤ c0 * c0 ∧
    prestonHalfBreadthSq c0 0 0 0 0 = c0 * c0 := by
  refine ⟨?_, ?_, ?_⟩
  · unfold ellipsoidBreadth ellipsoidLength
    have h1 : c0 * k ≤ 1 * k := Int.mul_le_mul_of_nonneg_right hc hk
    omega
  · simp [prestonHalfBreadthSq, prestonPoly, Int.mul_sub]
    have h1 : 0 ≤ c0 * c0 * (x * x) := Int.mul_nonneg hc0 hx
    omega
  · simp [prestonHalfBreadthSq, prestonPoly]

end MathOfBiology.Loop_228
