/-
  morphoeval/lean/loops/Loop_perrard_vespinae_wing_mechanics.lean
  Formal Lean 4 verification of the Perrard, Baylac, Carpenter & Villemant (2014) J. Evol. Biol. 27:2665-2675,
  Perrard, Pickett, Villemant, Kojima & Carpenter (2013) J. Hymenoptera Res. 32:1-15, and
  Combes & Daniel (2003) J. Exp. Biol. 206:2979-2997 Vespinae Wing-Venation Mechanics & Phylogeny Laws:

    1. Theorem 1 (`antiparallel_vein_dipole_seesaw_theorem`):
       Universal proof that whenever two adjacent vein-compartment landmarks `(d_a, d_b)` undergo
       antiparallel displacement (`d_b = -d_a`, `d_a != 0`, as in the Vespa prestigma `LM12 <-> LM5`
       with `r = -0.80` or basal vein M `LM13 <-> LM14` with `r = -0.89`), a global smooth mode `(1, 1)`
       anchored to the perimeter endpoint `a` inflates the squared residual at interior endpoint `b`
       from `d_b^2` to `4 * d_a^2 > d_b^2` (strictly negative explained variance numerator `-3 * d_a^2 < 0`
       for all `d_a != 0`), whereas the 2-mode span `{global (1, 1), dipole (1, -1)}` achieves exact
       zero residual error for all `(d_a, d_b)`.
    2. Theorem 2 (`combes_daniel_quartic_spar_stiffness_and_pinned_orthogonality`):
       Universal proof that tubular vein second moment of area `I_vein(D) = D^4` scales by `16x` upon
       doubling diameter `2 * D` for all `D : Int` (Combes & Daniel 2003 leading-edge `C+Sc+R1` spar
       canalization vs interior cross-veins), and that any pinned-perimeter interior cross-vein mode
       `(0, u_int)` is strictly orthogonal (`dot = 0`) and decouples perimeter variance for all
       `u_per, u_int : Int`.
    3. Theorem 3 (`resolved_sister_vs_polytomy_nearest_neighbor_error`):
       Universal proof that resolving a phylogenetic polytomy (Perrard et al. 2013 Total-Evidence tree
       vs OpenTree 15-way Vespa polytomy) cancels shared clade displacement `c` between sister species
       (`(c + e) - (c - e) = 2 * e`), whereas copying across an unresolved polytomy branch with opposite
       clade shift `-c` adds `4 * c^2 + 8 * (c * e)` to the squared prediction error (`strictly larger`
       for all `c > 0, e >= 0`).

  All theorems bind universally quantified variables (Rule 4) and compile in Lean 4 core with zero `sorry`s.
-/

namespace MathOfBiology.Loop_PerrardVespinaeWingMechanics

def sq_int (z : Int) : Int := z * z

-- ============================================================================
-- Theorem 1: Antiparallel Vein-Dipole Least-Squares Seesaw Theorem (D3 & D4)
-- ============================================================================
/-- Explained-variance numerator `d_b^2 - (d_b - pred_b)^2` at interior landmark `b`.
    When `explained_var_num d_b pred_b < 0`, the model prediction `pred_b` is strictly worse
    than predicting zero change (`R^2_b < 0`). -/
def explained_var_num (d_b pred_b : Int) : Int :=
  sq_int d_b - sq_int (d_b - pred_b)

/-- Total 2-landmark residual after fitting the symmetric global mode `(1, 1)` and the
    antiparallel vein-dipole mode `(1, -1)` with integer weights `w_sym = d_a + d_b` and
    `w_dip = d_a - d_b` (scaled by 2x to remain in exact integer arithmetic). -/
def dipole_basis_residual_2x (d_a d_b : Int) : Int :=
  let w_sym := d_a + d_b
  let w_dip := d_a - d_b
  sq_int (2 * d_a - (w_sym + w_dip)) + sq_int (2 * d_b - (w_sym - w_dip))

theorem antiparallel_vein_dipole_seesaw_theorem (d_a d_b : Int) :
    explained_var_num (-d_a) d_a = -3 * sq_int d_a ∧
    (d_a ≠ 0 → explained_var_num (-d_a) d_a < 0) ∧
    dipole_basis_residual_2x d_a d_b = 0 := by
  have h_ident : explained_var_num (-d_a) d_a = -3 * sq_int d_a := by
    dsimp [explained_var_num, sq_int]
    have h1 : (-d_a) * (-d_a) = d_a * d_a := Int.neg_mul_neg d_a d_a
    have h2 : -d_a - d_a = -(2 * d_a) := by omega
    rw [h1, h2, Int.neg_mul_neg]
    have h3 : (2 * d_a) * (2 * d_a) = 4 * (d_a * d_a) := by
      calc (2 * d_a) * (2 * d_a) = 2 * 2 * (d_a * d_a) := by ac_rfl
      _ = 4 * (d_a * d_a) := by rfl
    rw [h3]
    omega
  refine ⟨h_ident, ?_, ?_⟩
  · intro hne
    rw [h_ident]
    dsimp [sq_int]
    have hpos : 0 < d_a * d_a := by
      cases Int.lt_or_gt_of_ne hne with
      | inl hlt => exact Int.mul_pos_of_neg_of_neg hlt hlt
      | inr hgt => exact Int.mul_pos hgt hgt
    omega
  · dsimp [dipole_basis_residual_2x, sq_int]
    have h1 : 2 * d_a - (d_a + d_b + (d_a - d_b)) = 0 := by omega
    have h2 : 2 * d_b - (d_a + d_b - (d_a - d_b)) = 0 := by omega
    rw [h1, h2]
    rfl

-- ============================================================================
-- Theorem 2: Combes-Daniel Quartic Vein-Spar Stiffness & Pinned-Perimeter Orthogonality (D6)
-- ============================================================================
def vein_flexural_rigidity (D : Int) : Int :=
  sq_int (sq_int D)

def two_tier_dot (u_per1 u_int1 u_per2 u_int2 : Int) : Int :=
  u_per1 * u_per2 + u_int1 * u_int2

theorem combes_daniel_quartic_spar_stiffness_and_pinned_orthogonality (D u_per u_int : Int) :
    vein_flexural_rigidity (2 * D) = 16 * vein_flexural_rigidity D ∧
    two_tier_dot u_per 0 0 u_int = 0 ∧
    two_tier_dot u_per u_int u_per u_int = sq_int u_per + sq_int u_int := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [vein_flexural_rigidity, sq_int]
    have h2 : (2 * D) * (2 * D) = 4 * (D * D) := by
      calc (2 * D) * (2 * D) = 2 * 2 * (D * D) := by ac_rfl
      _ = 4 * (D * D) := by rfl
    rw [h2]
    calc (4 * (D * D)) * (4 * (D * D)) = 4 * 4 * ((D * D) * (D * D)) := by ac_rfl
    _ = 16 * ((D * D) * (D * D)) := by rfl
  · simp [two_tier_dot]
  · rfl

-- ============================================================================
-- Theorem 3: Resolved Sister-Species vs Polytomy Nearest-Neighbor Error Law (D1, D2, D5)
-- ============================================================================
def sister_copy_sq_err (c e : Int) : Int :=
  sq_int ((c + e) - (c - e))

def polytomy_copy_sq_err (c e : Int) : Int :=
  sq_int ((c + e) - (-c - e))

theorem resolved_sister_vs_polytomy_nearest_neighbor_error (c e : Int) :
    sister_copy_sq_err c e = 4 * sq_int e ∧
    polytomy_copy_sq_err c e - sister_copy_sq_err c e = 4 * sq_int c + 8 * (c * e) ∧
    (0 < c → 0 ≤ e → sister_copy_sq_err c e < polytomy_copy_sq_err c e) := by
  have h_sis : sister_copy_sq_err c e = 4 * sq_int e := by
    dsimp [sister_copy_sq_err, sq_int]
    have h : c + e - (c - e) = 2 * e := by omega
    rw [h]
    calc (2 * e) * (2 * e) = 2 * 2 * (e * e) := by ac_rfl
    _ = 4 * (e * e) := by rfl
  have h_diff : polytomy_copy_sq_err c e - sister_copy_sq_err c e = 4 * sq_int c + 8 * (c * e) := by
    rw [h_sis]
    dsimp [polytomy_copy_sq_err, sq_int]
    have h2 : c + e - (-c - e) = 2 * (c + e) := by omega
    rw [h2]
    have h3 : (2 * (c + e)) * (2 * (c + e)) = 4 * ((c + e) * (c + e)) := by
      calc (2 * (c + e)) * (2 * (c + e)) = 2 * 2 * ((c + e) * (c + e)) := by ac_rfl
      _ = 4 * ((c + e) * (c + e)) := by rfl
    have h4 : (c + e) * (c + e) = c * c + 2 * (c * e) + e * e := by
      calc (c + e) * (c + e) = c * c + c * e + (e * c + e * e) := by
            rw [Int.add_mul, Int.mul_add, Int.mul_add]
      _ = c * c + 2 * (c * e) + e * e := by
            rw [Int.mul_comm e c]
            omega
    rw [h3, h4]
    omega
  refine ⟨h_sis, h_diff, ?_⟩
  · intro hc he
    have hcc : 0 < sq_int c := Int.mul_pos hc hc
    have hce : 0 ≤ c * e := Int.mul_nonneg (Int.le_of_lt hc) he
    omega

end MathOfBiology.Loop_PerrardVespinaeWingMechanics
