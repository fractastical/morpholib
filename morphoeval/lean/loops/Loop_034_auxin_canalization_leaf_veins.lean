/-
  morphoeval/lean/loops/Loop_034_auxin_canalization_leaf_veins.lean
  Formal Lean 4 verification of Mitchison (1980) Proc. R. Soc. Lond. B 207:79-109,
  Mitchison (1981) Phil. Trans. R. Soc. Lond. B 295:461-471,
  Feugier, Mochizuki & Iwasa (2005) J. Theor. Biol. 236:366-375, and
  Sachs (1981) Adv. Bot. Res. 9:151-262 governing quadratic flux-canalization
  symmetry breaking, canalized vein concentration drainage (S_vein < S_parenchyma),
  and PIN1 carrier-allocated polar auxin flux conservation (`mitchison-canalization-auxin-flux`, Slot 034).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_034

-- ============================================================================
-- Law 1 (`mitchison-canalization-auxin-flux-1`):
-- Mitchison (1980) Linear vs Quadratic Flux-Canalization Symmetry Breaking Law
-- ============================================================================
/-- In Mitchison (1980) Proc. R. Soc. Lond. B 207:79-109, two parallel pathways with
    conductivities `D1` and `D2` sharing a concentration drop `dS` carry fluxes
    `phi1 = D1 * dS` and `phi2 = D2 * dS`.
    - Under linear feedback (`p = 1`): `dD_k/dt = alpha * (D_k * dS) + beta - gamma * D_k`.
      The cross-multiplied relative growth rate `D2 * dD1_lin - D1 * dD2_lin` equals
      `beta * (D2 - D1)`, which damps any conductivity asymmetry (`D1 > D2`) back to
      symmetric diffuse flow (`D1 = D2`).
    - Under quadratic feedback (`p = 2`): `dD_k/dt = alpha * (D_k * dS)^2 + beta - gamma * D_k`.
      The cross-multiplied relative growth rate `D2 * dD1_quad - D1 * dD2_quad` factors as
      `(D1 - D2) * (alpha * dS * dS * D1 * D2 - beta)`, amplifying any asymmetry `D1 > D2`
      above the critical flux threshold into a 1-cell-wide canalized vascular strand. -/
def linear_canalization_rhs (alpha beta gamma dS D : Int) : Int :=
  alpha * (D * dS) + beta - gamma * D

def quadratic_canalization_rhs (alpha beta gamma dS D : Int) : Int :=
  alpha * (D * dS) * (D * dS) + beta - gamma * D

theorem mitchison_1980_quadratic_vs_linear_symmetry_breaking
    (alpha beta gamma dS D1 D2 : Int) :
    D2 * linear_canalization_rhs alpha beta gamma dS D1 -
      D1 * linear_canalization_rhs alpha beta gamma dS D2 = beta * (D2 - D1) ∧
    D2 * quadratic_canalization_rhs alpha beta gamma dS D1 -
      D1 * quadratic_canalization_rhs alpha beta gamma dS D2 =
      (D1 - D2) * (alpha * dS * dS * D1 * D2 - beta) := by
  constructor
  · dsimp [linear_canalization_rhs]
    simp [Int.mul_add, Int.mul_sub, Int.mul_comm, Int.mul_left_comm]
    omega
  · dsimp [quadratic_canalization_rhs]
    simp [Int.mul_add, Int.mul_sub, Int.mul_comm, Int.mul_left_comm]
    omega


-- ============================================================================
-- Law 2 (`mitchison-canalization-auxin-flux-2`):
-- Mitchison (1980) Canalized Vein Auxin Concentration Drainage Law (S_vein < S_parenchyma)
-- ============================================================================
/-- At steady state (`dD/dt = 0`) under quadratic feedback (`p = 2`), a pathway carrying
    flux `phi` has scaled conductivity `D_scaled(phi) = alpha * phi^2 + beta` (`gamma * D`).
    Since local auxin concentration required to drive flux `phi` through conductivity `D`
    is `S = phi / D`, clearing positive denominators `D_scaled(phi_vein) * D_scaled(phi_par)`
    gives the scaled concentration difference between parenchyma (`phi_par`) and the
    canalized vein (`phi_vein`):
      `S_par_scaled - S_vein_scaled = phi_par * D_scaled(phi_vein) - phi_vein * D_scaled(phi_par)`
                                   `= (phi_vein - phi_par) * (alpha * phi_vein * phi_par - beta)`.
    Thus whenever `phi_vein > phi_par` and `alpha * phi_vein * phi_par > beta`, auxin
    concentration inside the canalized vein drops strictly below the surrounding parenchyma
    (`S_vein < S_par`), draining lateral auxin into the differentiating vascular strand. -/
def steady_conductivity_scaled (alpha beta phi : Int) : Int :=
  alpha * phi * phi + beta

def drainage_conc_diff_scaled (alpha beta phi_vein phi_par : Int) : Int :=
  phi_par * steady_conductivity_scaled alpha beta phi_vein -
    phi_vein * steady_conductivity_scaled alpha beta phi_par

theorem mitchison_1980_vein_concentration_drainage (alpha beta phi_vein phi_par : Int) :
    drainage_conc_diff_scaled alpha beta phi_vein phi_par =
      (phi_vein - phi_par) * (alpha * phi_vein * phi_par - beta) := by
  dsimp [drainage_conc_diff_scaled, steady_conductivity_scaled]
  simp [Int.mul_add, Int.mul_sub, Int.mul_comm, Int.mul_left_comm]
  omega


-- ============================================================================
-- Law 3 (`mitchison-canalization-auxin-flux-3`):
-- Mitchison (1981) & Feugier et al. (2005) Polar Carrier-Allocated Flux Conservation Law
-- ============================================================================
/-- In Mitchison (1981) Phil. Trans. R. Soc. B 295:461-471 and Feugier, Mochizuki & Iwasa
    (2005) J. Theor. Biol. 236:366-375, net intercellular auxin flux from cell `i` to `j`
    combining polar PIN1 efflux carriers (`P_ij`, `P_ji`) and passive diffusion (`D0`) is:
      `phi_{i->j} = P_ij * S_i - P_ji * S_j + D0 * (S_i - S_j)`.
    This theorem proves that:
    1. Pairwise intercellular flux is strictly antisymmetric (`phi_{i->j} + phi_{j->i} = 0`),
       guaranteeing exact auxin mass conservation across the leaf lamina graph.
    2. At homogeneous auxin concentration (`S_i = S_j = S`), net flux reduces purely to
       polar carrier asymmetry `(P_ij - P_ji) * S`. -/
def polar_canalization_flux (P_ij P_ji D0 S_i S_j : Int) : Int :=
  P_ij * S_i - P_ji * S_j + D0 * (S_i - S_j)

theorem feugier_2005_polar_flux_conservation (P_ij P_ji D0 S_i S_j S : Int) :
    polar_canalization_flux P_ij P_ji D0 S_i S_j +
      polar_canalization_flux P_ji P_ij D0 S_j S_i = 0 ∧
    polar_canalization_flux P_ij P_ji D0 S S = (P_ij - P_ji) * S := by
  constructor
  · dsimp [polar_canalization_flux]
    simp [Int.mul_sub]
    omega
  · dsimp [polar_canalization_flux]
    simp [Int.sub_mul]

-- ============================================================================
-- Law 4 (`mitchison-canalization-auxin-flux-4`, Pass 3):
-- Saturating flux-feedback on a growing lamina — boundedness and monotonicity
-- ============================================================================
/- Pass 3 growing-lamina solver uses the soft-saturating Mitchison (1980) feedback
      `F(phi) = alpha * phi^2 / (1 + phi^2 / phi_s^2)`.
    Clearing the positive denominator `phi_s^2 + phi^2`, the feedback numerator
    `alpha * phi^2 * phi_s^2` never exceeds `alpha * phi_s^2 * (phi_s^2 + phi^2)`, i.e.
    `F(phi) <= alpha * phi_s^2` for every flux — this is what lets the midrib flux grow 4x per
    cell-division stage without the wall conductance diverging (Pass 2 hard cap removed). -/
/-- Squares of integers are non-negative (helper; core Lean 4 only). -/
theorem int_mul_self_nonneg (x : Int) : 0 ≤ x * x := by
  rcases Int.le_total 0 x with hx | hx
  · exact Int.mul_nonneg hx hx
  · exact Int.mul_nonneg_of_nonpos_of_nonpos hx hx

def saturating_feedback_num (alpha phi phis : Int) : Int :=
  alpha * (phi * phi) * (phis * phis)

def saturating_feedback_den (phi phis : Int) : Int :=
  phis * phis + phi * phi

theorem saturating_canalization_feedback_bounded (alpha phi phis : Int) (h : 0 ≤ alpha) :
    saturating_feedback_num alpha phi phis ≤ alpha * (phis * phis) * saturating_feedback_den phi phis := by
  dsimp [saturating_feedback_num, saturating_feedback_den]
  have h1 : 0 ≤ alpha * (phis * phis) * (phis * phis) :=
    Int.mul_nonneg (Int.mul_nonneg h (int_mul_self_nonneg phis)) (int_mul_self_nonneg phis)
  have h2 : alpha * (phis * phis) * (phis * phis + phi * phi) =
      alpha * (phis * phis) * (phis * phis) + alpha * (phi * phi) * (phis * phis) := by
    rw [Int.mul_add]
    simp [Int.mul_comm, Int.mul_left_comm]
  rw [h2]
  omega

/-- The saturating feedback is monotone in the flux magnitude: for `0 <= phi1 <= phi2` the
    cross-multiplied comparison `phi1^2 (phi_s^2 + phi2^2) <= phi2^2 (phi_s^2 + phi1^2)` holds,
    so a wall carrying more flux always receives at least as much reinforcement — the
    winner-take-all ordering of Law 1 survives saturation (strand hierarchy is preserved on
    the growing lamina). -/
theorem saturating_canalization_feedback_monotone (phi1 phi2 phis : Int)
    (h0 : 0 ≤ phi1) (h12 : phi1 ≤ phi2) :
    phi1 * phi1 * saturating_feedback_den phi2 phis ≤ phi2 * phi2 * saturating_feedback_den phi1 phis := by
  dsimp [saturating_feedback_den]
  have hsq : phi1 * phi1 ≤ phi2 * phi2 :=
    Int.mul_le_mul h12 h12 h0 (Int.le_trans h0 h12)
  have hs : 0 ≤ phis * phis := int_mul_self_nonneg phis
  have hA : phi1 * phi1 * (phis * phis) ≤ phi2 * phi2 * (phis * phis) :=
    Int.mul_le_mul_of_nonneg_right hsq hs
  have e1 : phi1 * phi1 * (phis * phis + phi2 * phi2) =
      phi1 * phi1 * (phis * phis) + phi1 * phi1 * (phi2 * phi2) := Int.mul_add _ _ _
  have e2 : phi2 * phi2 * (phis * phis + phi1 * phi1) =
      phi2 * phi2 * (phis * phis) + phi2 * phi2 * (phi1 * phi1) := Int.mul_add _ _ _
  have e3 : phi1 * phi1 * (phi2 * phi2) = phi2 * phi2 * (phi1 * phi1) := Int.mul_comm _ _
  rw [e1, e2, e3]
  omega

-- ============================================================================
-- Law 4b (`mitchison-canalization-auxin-flux-4b`, Pass 3):
-- Irreversible vascular differentiation is monotone (Sachs 1981)
-- ============================================================================
/-- The optional irreversible-differentiation rule of the growing-lamina solver updates the
    differentiated flag as `d' = d || (D > D_diff)`.  Once a wall is differentiated it stays
    differentiated, whatever the later conductance does: the set of differentiated walls is
    non-decreasing in time (the ablation shows this alone does not close loops). -/
def differentiation_update (d : Bool) (exceeds : Bool) : Bool :=
  d || exceeds

theorem irreversible_differentiation_monotone (d exceeds : Bool) (h : d = true) :
    differentiation_update d exceeds = true := by
  simp [differentiation_update, h]

end MathOfBiology.Loop_034
