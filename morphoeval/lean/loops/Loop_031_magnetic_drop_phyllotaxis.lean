/-
  morphoeval/lean/loops/Loop_031_magnetic_drop_phyllotaxis.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 031:
  Douady & Couder (1992) Phys. Rev. Lett. 68:2098 & (1996) J. Theor. Biol. 178:255
  (`douady-magnetic-drop-phyllotaxis`)

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_031_magnetic_drop_phyllotaxis.lean
-/

namespace MathOfBiology.Loop_031

-- ============================================================================
-- Law 1 (`douady-magnetic-drop-phyllotaxis-1`):
-- Radial Advection & Squared Chord Distance Extremum at Opposition (1,1)
-- ============================================================================

/-- Linear radial advection `r_k = R_0 + k * G` (with scaled rim radius `R_0 = 1`)
    and squared Euclidean chord distance `d_k^2(c) = 1 + r_k^2 - 2 * r_k * c`
    as a function of integer cosine proxy `c ∈ {-1, 0, 1}` (`c = 1` at conjunction
    `Δφ = 0`, `c = -1` at opposition `Δφ = π`). -/
def linear_advected_radius (k G : Int) : Int :=
  1 + k * G

def chord_distance_sq (r_k c : Int) : Int :=
  1 + r_k * r_k - 2 * r_k * c

/-- Theorem 1 (`douady_couder_opposition_maximizes_chord_distance`):
    At conjunction (`c = 1, Δφ = 0`), `d_k^2(1) = (r_k - 1)^2`;
    at opposition (`c = -1, Δφ = π`), `d_k^2(-1) = (r_k + 1)^2`;
    and the squared chord distance gain at opposition over conjunction is
    `d_k^2(-1) - d_k^2(1) = 4 * r_k`, which for `r_1 = 1 + G` with `G ≥ 1`
    is at least `8 > 0`. Thus for a single dominant preceding drop (`k = 1`),
    any strictly decreasing repulsive potential `V(d^2)` attains its global minimum
    at opposition `Δφ = π` (the distichous `(1, 1)` mode at large `G`). -/
theorem douady_couder_opposition_maximizes_chord_distance (G : Int) :
    chord_distance_sq (linear_advected_radius 1 G) (-1) -
      chord_distance_sq (linear_advected_radius 1 G) 1 =
      4 * (1 + G) ∧
    chord_distance_sq 3 (-1) = 16 ∧
    chord_distance_sq 3 1 = 4 := by
  refine ⟨?_, by decide, by decide⟩
  simp [chord_distance_sq, linear_advected_radius]
  omega


-- ============================================================================
-- Law 2 (`douady-magnetic-drop-phyllotaxis-2`):
-- Supercritical Pitchfork Bifurcation `(1,1) -> (1,2)` Curvature Sign Change
-- ============================================================================

/-- Cross-multiplied numerator of the second derivative of the 2-drop dipole
    repulsion energy `E_2(φ) = d_1(φ)^{-3} + d_2(φ)^{-3}` evaluated at the
    alternating distichous state `φ = 0` (`φ_{n-1} = π` at `r_1`, `φ_{n-2} = 0` at `r_2`):
      `d^2 E_2 / dφ^2 |_{φ=0} = 3*r_1/(r_1 + 1)^5 - 3*r_2/(r_2 - 1)^5`.
    Clearing positive denominators `(r_1 + 1)^5 * (r_2 - 1)^5 > 0` yields the exact
    stability discriminant:
      `C(r_1, r_2) = 3 * r_1 * (r_2 - 1)^5 - 3 * r_2 * (r_1 + 1)^5`. -/
def pitchfork_curvature_discriminant (r1 r2 : Int) : Int :=
  3 * r1 * (r2 - 1) ^ 5 - 3 * r2 * (r1 + 1) ^ 5

/-- Theorem 2 (`douady_couder_pitchfork_bifurcation_sign_change`):
    Under linear advection `r_1 = 1 + G, r_2 = 1 + 2*G`:
    1. At `G = 3` (`r_1 = 4, r_2 = 7`), `C(4, 7) = 27687 > 0`, so `d^2 E_2 / dφ^2 > 0`
       and the alternating distichous `(1, 1)` mode (`Δφ = π`) is a strict local
       energy minimum (stable).
    2. At `G = 2` (`r_1 = 3, r_2 = 5`), `r_1 + 1 = r_2 - 1 = 4` and `C(3, 5) = -6144 < 0`,
       so `d^2 E_2 / dφ^2 < 0`: opposition `Δφ = π` loses stability as a local energy
       maximum and bifurcates via a supercritical pitchfork at `G_{c,1} ∈ (2, 3)`
       (`G_{c,1} ≈ 2.51456`) into two chiral spiral branches `(1, 2)` (`Δφ ≠ π`). -/
theorem douady_couder_pitchfork_bifurcation_sign_change :
    pitchfork_curvature_discriminant (linear_advected_radius 1 3) (linear_advected_radius 2 3) = 27687 ∧
    pitchfork_curvature_discriminant (linear_advected_radius 1 3) (linear_advected_radius 2 3) > 0 ∧
    pitchfork_curvature_discriminant (linear_advected_radius 1 2) (linear_advected_radius 2 2) = -6144 ∧
    pitchfork_curvature_discriminant (linear_advected_radius 1 2) (linear_advected_radius 2 2) < 0 := by
  decide


-- ============================================================================
-- Law 3 (`douady-magnetic-drop-phyllotaxis-3`):
-- Imperfect Pitchfork Parastichy Cascade & Unimodular Cassini Invariant
-- ============================================================================

/-- Generalized phyllotactic parastichy recurrence `S_0 = a, S_1 = b, S_{k+2} = S_{k+1} + S_k`.
    Setting `(a, b) = (1, 1)` generates the main Fibonacci branch `(1, 1, 2, 3, 5, 8, 13, 21)`,
    while setting `(a, b) = (1, 3)` generates the secondary Lucas branch `(1, 3, 4, 7, 11, 18)`. -/
def parastichy_seq (a b : Nat) : Nat → Nat
  | 0 => a
  | 1 => b
  | n + 2 => parastichy_seq a b (n + 1) + parastichy_seq a b n

/-- Cassini-Bezout determinant `D(p, q) = p^2 + p*q - q^2` associated with an opposing
    contact parastichy pair `(p, q)` on the cylindrical/conical van Iterson lattice. -/
def cassini_det (p q : Int) : Int :=
  p * p + p * q - q * q

/-- Theorem 3 (`douady_couder_fibonacci_vs_lucas_parastichy_cascades`):
    Verifies that:
    1. One rising phyllotaxis transition `(p, q) ↦ (q, p + q)` negates `cassini_det`
       (`cassini_det q (p + q) + cassini_det p q = 0`), preserving `|cassini_det|`
       across every imperfect pitchfork transition;
    2. The main Fibonacci branch `(a=1, b=1)` cascades through the exact consecutive pairs
       `(1,1) -> (1,2) -> (2,3) -> (3,5) -> (5,8) -> (8,13)` with `|cassini_det| = 1`;
    3. The secondary Lucas branch `(a=1, b=3)` cascades through `(1,3) -> (3,4) -> (4,7) -> (7,11)`
       with `|cassini_det| = 5`. -/
theorem douady_couder_fibonacci_vs_lucas_parastichy_cascades (p q : Int) :
    cassini_det q (p + q) + cassini_det p q = 0 ∧
    (parastichy_seq 1 1 1, parastichy_seq 1 1 2) = (1, 2) ∧
    (parastichy_seq 1 1 2, parastichy_seq 1 1 3) = (2, 3) ∧
    (parastichy_seq 1 1 3, parastichy_seq 1 1 4) = (3, 5) ∧
    (parastichy_seq 1 1 4, parastichy_seq 1 1 5) = (5, 8) ∧
    (parastichy_seq 1 1 5, parastichy_seq 1 1 6) = (8, 13) ∧
    cassini_det 5 8 = 1 ∧
    (parastichy_seq 1 3 0, parastichy_seq 1 3 1) = (1, 3) ∧
    (parastichy_seq 1 3 1, parastichy_seq 1 3 2) = (3, 4) ∧
    (parastichy_seq 1 3 2, parastichy_seq 1 3 3) = (4, 7) ∧
    (parastichy_seq 1 3 3, parastichy_seq 1 3 4) = (7, 11) ∧
    cassini_det 4 7 = -5 := by
  refine ⟨?_, by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide⟩
  simp [cassini_det, Int.add_mul, Int.mul_add]
  omega

end MathOfBiology.Loop_031
