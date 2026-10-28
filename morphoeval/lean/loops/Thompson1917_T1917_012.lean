/-
  morphoeval/lean/loops/Thompson1917_T1917_012.lean
  Loop 209 (slot thompson1917-t1917-012) — Thompson (1917) Ch. IV, Figs. 58-59:
  early cleavage conforms to a bipolar field with two *similar* poles.

  Standalone Lean 4 (no imports; omega / simp / rw only).  Integer-scaled meridional
  plane, mirroring `thompson/claims/t1917-012_on-the-internal-form-and-structure-of-the-cell.py`:

  * Two poles at (±a, 0).  Squared-distance factors d₋ = (x+a)² + y², d₊ = (x−a)² + y².
  * The like-pole potential proxy P = d₋ · d₊ is even in x (reflection symmetry of the field).
  * Discrete equatorial stationarity: P(1, y) = P(−1, y) ⇒ central difference at x = 0 vanishes.
  * Cleavage orthogonality (Thompson p. 190): polar axis (1,0) and equatorial normal (0,1) are orthogonal.
  * Pole-exchange symmetry: swapping x ↦ −x exchanges the roles of the two poles.
-/
namespace MathOfBiology.Loop_209

/-- Squared distance from the pole at `(−a, 0)`. -/
def dMinus (a x y : Int) : Int := (x + a) * (x + a) + y * y

/-- Squared distance from the pole at `(+a, 0)`. -/
def dPlus (a x y : Int) : Int := (x - a) * (x - a) + y * y

/-- Even potential proxy `P = d₋ · d₊` for the two-like-pole field (logarithmic potential in the script). -/
def P (a x y : Int) : Int := dMinus a x y * dPlus a x y

/-- E4: reflection symmetry `P(−x, y) = P(x, y)`. -/
theorem potential_even (a x y : Int) : P a (-x) y = P a x y := by
  unfold P dMinus dPlus
  have h1 : (-x + a) * (-x + a) = (x - a) * (x - a) := by
    have e1 := Int.neg_add a x
    have e2 := Int.add_comm a (-x)
    have e3 := Int.neg_add x a
    omega
  have h2 : (-x - a) * (-x - a) = (x + a) * (x + a) := by
    have e1 := Int.neg_sub x a
    omega
  have e1 := Int.mul_comm (dMinus a (-x) y) (dPlus a (-x) y)
  omega

/-- Roles of the two poles swap under `x ↦ −x`. -/
theorem pole_exchange (a x y : Int) :
    dMinus a (-x) y = dPlus a x y ∧ dPlus a (-x) y = dMinus a x y := by
  unfold dMinus dPlus
  have e1 := Int.neg_sub x a
  have e2 := Int.neg_add a x
  omega

/-- E4 corollary: the two squared-distance factors swap under reflection. -/
theorem factors_swap (a x y : Int) :
    dMinus a x y * dPlus a x y = dPlus a (-x) y * dMinus a (-x) y := by
  have h := potential_even a x y
  unfold P at h
  exact h

/-- Discrete equatorial stationarity at `x = 0`: `P(1, y) = P(−1, y)`. -/
theorem equatorial_value_match (a y : Int) : P a 1 y = P a (-1) y := by
  have h := potential_even a 1 y
  simpa using h.symm

/-- Central difference of an even function at the origin vanishes (integer form). -/
theorem discrete_equatorial_stationary (a y : Int) : P a 1 y - P a (-1) y = 0 := by
  have h := equatorial_value_match a y
  omega

/-- E3: equatorial cleavage plane normal `(0, 1)` is orthogonal to polar axis `(1, 0)`. -/
theorem cleavage_orthogonal_to_polar_axis :
    (1 : Int) * 0 + 0 * 1 = 0 := by
  omega

/-- For `a > 0`, poles are distinct and lie on the polar axis. -/
theorem pole_separation (a : Int) (ha : 0 < a) : 0 < a + a := by
  omega

/-- On the polar axis between the poles (`y = 0`, `|x| < a`), both distance factors are positive. -/
theorem between_poles_positive (a x : Int) (ha : 0 < a) (hx : -a < x) (hx' : x < a) :
    0 < dMinus a x 0 ∧ 0 < dPlus a x 0 := by
  unfold dMinus dPlus
  have hm : 0 < x + a := by omega
  have hp : 0 < a - x := by omega
  have e1 : 0 < (x + a) * (x + a) := by
    have h := Int.mul_pos hm hm
    simpa using h
  have e2 : 0 < (x - a) * (x - a) := by
    have hneg : x - a = -(a - x) := by omega
    rw [hneg, Int.neg_mul, Int.neg_neg]
    have h := Int.mul_pos hp hp
    simpa using h
  exact ⟨e1, e2⟩

/-- Like-pole symmetry group: reflecting twice is identity on `P`. -/
theorem reflect_twice (a x y : Int) : P a (-(-x)) y = P a x y := by
  have h1 : -(-x) = x := Int.neg_neg x
  rw [h1]

end MathOfBiology.Loop_209
