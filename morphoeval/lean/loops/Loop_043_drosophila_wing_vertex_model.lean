/-
  morphoeval/lean/loops/Loop_043_drosophila_wing_vertex_model.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 043:
  Farhadifar, Röper, Aigouy, Eaton & Jülicher (2007) Current Biology 17:2095-2104 —
  Drosophila Wing Disc Epithelial Vertex Mechanics, Lewis's Law (1928),
  Euler 3-Valent Topological Charge Conservation, and Larval vs. Pupal Hexagonal Packing
  (`farhadifar-drosophila-wing-vertex-model`).

  Pure core Lean 4 (zero external imports, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_043_drosophila_wing_vertex_model.lean
-/

namespace MathOfBiology.Loop_043

/-- Twice the single-cell apical vertex energy functional from Farhadifar et al. (2007)
    Current Biology 17(24):2095-2104 Eq. (1):
      `2 * E_cell(K, A, A0, Lambda, l_sum, Gamma, L) = K * (A - A0)^2 + 2 * Lambda * l_sum + Gamma * L^2`
    expressed in exact integer arithmetic. -/
def two_vertex_cell_energy (K A A0 Lambda l_sum Gamma L : Int) : Int :=
  K * (A - A0) * (A - A0) + 2 * Lambda * l_sum + Gamma * L * L

/-- Twice the isolated area-elasticity energy penalty `K * (A - A0)^2` from Farhadifar et al. (2007) Eq. (1). -/
def two_area_elastic_energy (K A A0 : Int) : Int :=
  K * (A - A0) * (A - A0)

/-- Theorem 1 (`farhadifar_area_energy_equilibrium_zero`):
    At the preferred target apical area `A = A^{(0)}`, the quadratic area-elasticity penalty
    `K * (A - A^{(0)})^2` vanishes identically for any area modulus `K`, and the total cell
    energy reduces strictly to the junctional line-tension and perimeter-contractility terms
    `2 * Λ * l_sum + Γ * L^2`. -/
theorem farhadifar_area_energy_equilibrium_zero (K A0 Lambda l_sum Gamma L : Int) :
    two_area_elastic_energy K A0 A0 = 0 ∧
    two_vertex_cell_energy K A0 A0 Lambda l_sum Gamma L = 2 * Lambda * l_sum + Gamma * L * L := by
  constructor
  · dsimp [two_area_elastic_energy]
    rw [Int.sub_self, Int.mul_zero]
  · dsimp [two_vertex_cell_energy]
    omega

/-- Four times the normalized mean polygon area `4 * (<A_n> / <A>) = n - 2` from Lewis's Law
    (Lewis 1928 Anat. Rec. 38:341-376; Farhadifar et al. 2007 Fig. 3C). -/
def lewis_four_area_ratio (n : Int) : Int :=
  n - 2

/-- Theorem 2 (`lewis_law_hexagon_normalization_and_monotonicity`):
    Proves that under Lewis's Law `<A_n>/<A> = (n - 2)/4`:
    (1) Hexagonal cells (`n = 6`) have unit normalized area (`4 * (<A_6>/<A>) = 4`);
    (2) Exact integer ratios hold across `n = 4..9` (`2, 3, 4, 5, 6, 7` fourths);
    (3) Area increases strictly monotonically with polygon side number (`n1 < n2 → lewis(n1) < lewis(n2)`)
        with constant finite increment `lewis(n + 1) - lewis(n) = 1` (i.e. `+1/4` per side). -/
theorem lewis_law_hexagon_normalization_and_monotonicity (n n1 n2 : Int) (h : n1 < n2) :
    lewis_four_area_ratio 6 = 4 ∧
    lewis_four_area_ratio 4 = 2 ∧
    lewis_four_area_ratio 5 = 3 ∧
    lewis_four_area_ratio 7 = 5 ∧
    lewis_four_area_ratio 8 = 6 ∧
    lewis_four_area_ratio 9 = 7 ∧
    lewis_four_area_ratio n1 < lewis_four_area_ratio n2 ∧
    lewis_four_area_ratio (n + 1) - lewis_four_area_ratio n = 1 := by
  dsimp [lewis_four_area_ratio]
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, by omega, by omega⟩

/-- Theorem 3 (`euler_three_valent_average_six_sides`):
    In any periodic (toroidal) 3-valent epithelial vertex network with `V` tri-cellular vertices,
    `E` junctional edges, and `F` polygonal cells satisfying Euler's relation `V - E + F = 0`
    and 3-valent incidence `3 * V = 2 * E`, the total sum of polygon sides across all `F` cells
    (`sum_sides = 2 * E`) equals `6 * F` (so the mean polygon number is `<n> = 6`), and a T1
    intercalation converting two adjacent hexagons `(6, 6)` into a pentagon-heptagon dislocation
    pair `(5, 7)` strictly conserves total topological coordination (`5 + 7 = 6 + 6 = 12`). -/
theorem euler_three_valent_average_six_sides (V E F : Int)
    (h_euler : V - E + F = 0) (h_trivalent : 3 * V = 2 * E) :
    2 * E = 6 * F ∧ (5 + 7 : Int) = 6 + 6 := by
  constructor
  · omega
  · decide

/-- Per-mille (`1000 * P(n)`) polygon-class probabilities for `n ∈ {4, 5, 6, 7, 8, 9}` in:
    (a) Proliferating 3rd-instar Drosophila larval wing disc (Farhadifar et al. 2007 Fig. 4B):
        `P = (30, 280, 455, 200, 30, 5)` per mille;
    (b) Post-mitotic pupal wing after PCP junctional remodeling (Classen et al. 2005 Dev. Cell):
        `P = (5, 110, 780, 100, 5, 0)` per mille. -/
def p1000_wing_4 : Int := 30
def p1000_wing_5 : Int := 280
def p1000_wing_6 : Int := 455
def p1000_wing_7 : Int := 200
def p1000_wing_8 : Int := 30
def p1000_wing_9 : Int := 5

def p1000_pupal_4 : Int := 5
def p1000_pupal_5 : Int := 110
def p1000_pupal_6 : Int := 780
def p1000_pupal_7 : Int := 100
def p1000_pupal_8 : Int := 5
def p1000_pupal_9 : Int := 0

/-- Second central topological moment `1000 * μ_2 = ∑_{n=4}^9 1000 * P(n) * (n - 6)^2` measuring
    deviation from a regular hexagonal honeycomb. -/
def mu2_1000_wing : Int :=
  p1000_wing_4 * 4 + p1000_wing_5 * 1 + p1000_wing_6 * 0 +
  p1000_wing_7 * 1 + p1000_wing_8 * 4 + p1000_wing_9 * 9

def mu2_1000_pupal : Int :=
  p1000_pupal_4 * 4 + p1000_pupal_5 * 1 + p1000_pupal_6 * 0 +
  p1000_pupal_7 * 1 + p1000_pupal_8 * 4 + p1000_pupal_9 * 9

/-- Theorem 4 (`farhadifar_2007_wing_vs_pupal_distribution_identities`):
    Proves that both the proliferating larval wing disc distribution and the post-mitotic
    pupal wing distribution are normalized (`∑ P(n) = 1000` per mille), that post-mitotic
    junctional annealing strictly increases the hexagonal fraction (`P_pupal(6) = 780 > P_wing(6) = 455`),
    and that topological disorder `μ_2 = <(n - 6)^2>` drops by more than a factor of two
    (`1000 * μ_2,pupal = 250 < 1000 * μ_2,wing = 765`). -/
theorem farhadifar_2007_wing_vs_pupal_distribution_identities :
    (p1000_wing_4 + p1000_wing_5 + p1000_wing_6 + p1000_wing_7 + p1000_wing_8 + p1000_wing_9 = 1000) ∧
    (p1000_pupal_4 + p1000_pupal_5 + p1000_pupal_6 + p1000_pupal_7 + p1000_pupal_8 + p1000_pupal_9 = 1000) ∧
    (p1000_pupal_6 > p1000_wing_6) ∧
    (mu2_1000_wing = 765) ∧
    (mu2_1000_pupal = 250) ∧
    (2 * mu2_1000_pupal < mu2_1000_wing) := by
  decide

end MathOfBiology.Loop_043
