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

/-! ## Pass 3: planar-cell-polarity (Frizzled/Strabismus) bond-asymmetry tensor -/

/-- Quadratic form `n . Q . n` of a traceless symmetric 2x2 PCP bond tensor
    `Q = [[a, b], [b, -a]]` contracted with a bond normal `n = (c, s)` (integer-scaled
    components), i.e. `a * c^2 - a * s^2 + 2 * b * c * s`.  For uniform PD-aligned
    polarity `(a, b) = (1, 0)` this is `c^2 - s^2 = cos 2 theta_n`, the Fz/Stbm bond
    asymmetry `a_ab = 2 c_ab - 1` used in `pcp_bond_asymmetry` of
    `2007-farhadifar-drosophila-wing-vertex-model.py`. -/
def pcp_bond_quadratic_form (a b c s : Int) : Int :=
  a * (c * c) - a * (s * s) + 2 * b * (c * s)

/-- Trace of the PCP bond tensor `Q = [[a, b], [b, -a]]`. -/
def pcp_bond_tensor_trace (a : Int) : Int :=
  a + (-a)

/-- Per-mille bond line tension `1000 * Lambda_ab / Lambda_bar = 1000 + eps1000 * a_ab`
    for asymmetry `a_ab in {-1, 0, +1}` scaled as `a1000 in {-1000, 0, 1000}` (divided by 1000). -/
def pcp_bond_tension_permille (eps1000 a1000 : Int) : Int :=
  1000 + eps1000 * a1000 / 1000

/-- Theorem 5 (`pcp_bond_tensor_traceless_and_perpendicular_cancellation`):
    (1) the PCP bond tensor is traceless (`tr Q = 0`), so the Fz/Stbm asymmetry has zero
        isotropic part and cannot change the mean line tension of an isotropic network;
    (2) rotating the bond normal by 90 degrees, `(c, s) -> (-s, c)`, flips the sign of the
        contraction: `n . Q . n + n_perp . Q . n_perp = 0`.  PCP therefore stiffens
        AP-oriented junctions by exactly as much as it softens PD-oriented ones
        (`eps_pcp > 0`), which is the mechanism behind the oriented T1 bias in Section 2b;
    (3) for the uniform-polarity tensor `(a, b) = (1, 0)` and a unit normal with integer
        components the contraction reduces to `c^2 - s^2` (`cos 2 theta_n`);
    (4) with `0 < eps < 1000` per mille the AP bond tension `1000 + eps` strictly exceeds the
        PD bond tension `1000 - eps`, and both stay positive (no negative-tension Case II
        bond is created by PCP alone). -/
theorem pcp_bond_tensor_traceless_and_perpendicular_cancellation
    (a b c s eps1000 : Int) (h0 : 0 < eps1000) (h1 : eps1000 < 1000) :
    pcp_bond_tensor_trace a = 0 ∧
    pcp_bond_quadratic_form a b c s + pcp_bond_quadratic_form a b (-s) c = 0 ∧
    pcp_bond_quadratic_form 1 0 c s = c * c - s * s ∧
    pcp_bond_tension_permille eps1000 1000 = 1000 + eps1000 ∧
    pcp_bond_tension_permille eps1000 (-1000) = 1000 - eps1000 ∧
    pcp_bond_tension_permille eps1000 (-1000) < pcp_bond_tension_permille eps1000 1000 ∧
    0 < pcp_bond_tension_permille eps1000 (-1000) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · dsimp [pcp_bond_tensor_trace]
    omega
  · dsimp [pcp_bond_quadratic_form]
    have hsc : s * c = c * s := Int.mul_comm s c
    simp only [Int.neg_mul, Int.mul_neg, Int.neg_neg, hsc]
    omega
  · dsimp [pcp_bond_quadratic_form]
    omega
  · dsimp [pcp_bond_tension_permille]
    rw [Int.mul_ediv_cancel _ (by decide)]
  · dsimp [pcp_bond_tension_permille]
    have h : eps1000 * (-1000) = (-eps1000) * 1000 := by
      rw [Int.mul_neg, Int.neg_mul]
    rw [h, Int.mul_ediv_cancel _ (by decide)]
    omega
  · dsimp [pcp_bond_tension_permille]
    have h : eps1000 * (-1000) = (-eps1000) * 1000 := by
      rw [Int.mul_neg, Int.neg_mul]
    rw [h, Int.mul_ediv_cancel _ (by decide), Int.mul_ediv_cancel _ (by decide)]
    omega
  · dsimp [pcp_bond_tension_permille]
    have h : eps1000 * (-1000) = (-eps1000) * 1000 := by
      rw [Int.mul_neg, Int.neg_mul]
    rw [h, Int.mul_ediv_cancel _ (by decide)]
    omega

/-- Etournay et al. (2015) eLife shear-decomposition bookkeeping used in Section 2b:
    the imposed PD tissue shear `total` equals the change in mean cell elongation
    `elong` plus the rearrangement (T1) residual `rearr` (per-mille integers). -/
def shear_rearrangement_residual (total elong : Int) : Int :=
  total - elong

/-- Theorem 6 (`hinge_contraction_shear_decomposition_balance`):
    Under a fixed imposed hinge-contraction shear `total`, (1) the decomposition closes
    exactly (`elong + rearr = total`); (2) if PCP coupling lowers the residual cell
    elongation (`elong_pcp < elong_ctrl`, as measured for `eps_pcp = 0.5` vs the `eps_pcp = 0`
    control) then the rearrangement share strictly increases; (3) if the boundary strain is
    area-preserving pure shear with integer-scaled stretch `p/q` along PD and `q/p` along AP,
    the box area `Lx * Ly` is conserved: `(p * Lx) * (q * Ly) = (p * q) * (Lx * Ly)`
    (the `1/λ` compression exactly undoes the `λ` extension). -/
theorem hinge_contraction_shear_decomposition_balance
    (total elong_pcp elong_ctrl p q Lx Ly : Int) (h : elong_pcp < elong_ctrl) :
    elong_pcp + shear_rearrangement_residual total elong_pcp = total ∧
    shear_rearrangement_residual total elong_ctrl < shear_rearrangement_residual total elong_pcp ∧
    (p * Lx) * (q * Ly) = (p * q) * (Lx * Ly) := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [shear_rearrangement_residual]
    omega
  · dsimp [shear_rearrangement_residual]
    omega
  · rw [Int.mul_assoc, Int.mul_assoc, ← Int.mul_assoc Lx q Ly, Int.mul_comm Lx q, Int.mul_assoc]

end MathOfBiology.Loop_043
