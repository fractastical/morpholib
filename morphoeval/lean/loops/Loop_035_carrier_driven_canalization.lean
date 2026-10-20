/-
  morphoeval/lean/loops/Loop_035_carrier_driven_canalization.lean
  Formal Lean 4 verification of Feugier, Mochizuki & Iwasa (2005) J. Theor. Biol. 236(3):313-325
  and Scarpella et al. (2006) Genes & Development 20(8):1015-1027 governing hexagonal cell-level
  PIN1 efflux carrier pool conservation, winner-take-all open dendritic trees vs saturating
  multi-face co-polarization for closed reticulate venation loops, and pairwise polar auxin
  flux antisymmetry (`feugier-carrier-driven-canalization`, Slot 035).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_035

-- ============================================================================
-- Law 1 (`feugier-carrier-driven-canalization-1`):
-- Feugier et al. (2005) Hexagonal Intracellular PIN1 Carrier Pool Conservation Law
-- ============================================================================
/-- In Feugier, Mochizuki & Iwasa (2005) J. Theor. Biol. 236:313-325 (Eqs. 4-6), each hexagonal
    leaf lamina cell `i` partitions a fixed total PIN1 carrier pool `P_total` between its cytoplasmic
    vesicle pool `P_cyt` and its six plasma-membrane faces `P_1, ..., P_6`:
      `P_cyt = P_total - (P_1 + P_2 + P_3 + P_4 + P_5 + P_6)`.
    Each membrane face evolves via flux-stimulated exocytosis `k1 * Phi_j * P_cyt` minus endocytosis
    `k2 * P_j`, while the cytoplasmic pool evolves via `dP_cyt/dt = sum_j (k2 * P_j - k1 * Phi_j * P_cyt)`.
    This theorem proves:
    1. Exact state conservation: `P_cyt + sum_{j=1}^6 P_j = P_total`.
    2. Exact dynamical rate conservation: `dP_cyt/dt + sum_{j=1}^6 dP_j/dt = 0`. -/
def cytoplasmic_pin_pool (P_total P1 P2 P3 P4 P5 P6 : Int) : Int :=
  P_total - (P1 + P2 + P3 + P4 + P5 + P6)

def membrane_face_rhs (k1 k2 Phi_j P_cyt P_j : Int) : Int :=
  k1 * Phi_j * P_cyt - k2 * P_j

def cytoplasmic_pool_rhs
    (k1 k2 Phi1 Phi2 Phi3 Phi4 Phi5 Phi6 P_cyt P1 P2 P3 P4 P5 P6 : Int) : Int :=
  (k2 * P1 - k1 * Phi1 * P_cyt) +
  (k2 * P2 - k1 * Phi2 * P_cyt) +
  (k2 * P3 - k1 * Phi3 * P_cyt) +
  (k2 * P4 - k1 * Phi4 * P_cyt) +
  (k2 * P5 - k1 * Phi5 * P_cyt) +
  (k2 * P6 - k1 * Phi6 * P_cyt)

theorem feugier_2005_hexagonal_pin_pool_conservation
    (P_total k1 k2 Phi1 Phi2 Phi3 Phi4 Phi5 Phi6 P_cyt P1 P2 P3 P4 P5 P6 : Int) :
    cytoplasmic_pin_pool P_total P1 P2 P3 P4 P5 P6 + (P1 + P2 + P3 + P4 + P5 + P6) = P_total ∧
    cytoplasmic_pool_rhs k1 k2 Phi1 Phi2 Phi3 Phi4 Phi5 Phi6 P_cyt P1 P2 P3 P4 P5 P6 +
      (membrane_face_rhs k1 k2 Phi1 P_cyt P1 +
       membrane_face_rhs k1 k2 Phi2 P_cyt P2 +
       membrane_face_rhs k1 k2 Phi3 P_cyt P3 +
       membrane_face_rhs k1 k2 Phi4 P_cyt P4 +
       membrane_face_rhs k1 k2 Phi5 P_cyt P5 +
       membrane_face_rhs k1 k2 Phi6 P_cyt P6) = 0 := by
  constructor
  · dsimp [cytoplasmic_pin_pool]
    omega
  · dsimp [cytoplasmic_pool_rhs, membrane_face_rhs]
    omega


-- ============================================================================
-- Law 2 (`feugier-carrier-driven-canalization-2`):
-- Feugier et al. (2005) & Scarpella et al. (2006) Winner-Take-All Open Tree vs
-- Saturating Co-Polarized Closed Loop Law
-- ============================================================================
/-- Consider two competing efflux faces `1` and `2` of cell `i` draining into downstream sinks
    with concentration drops `dS1, dS2` so that carrier-driven fluxes are `J1 = P1 * dS1` and
    `J2 = P2 * dS2`, competing for the shared cytoplasmic pool `P_cyt`.
    - Under Feugier Model A non-saturating quadratic exocytosis (`Phi(J) = J^2`), the cross-multiplied
      relative growth rate `P2 * dP1/dt - P1 * dP2/dt` at equal downstream drop `dS1 = dS2 = dS` is:
        `k1 * P_cyt * dS * dS * P1 * P2 * (P1 - P2)`,
      which strictly amplifies any carrier asymmetry `P1 > P2` into winner-take-all single-face
      polarization (`P2 -> 0`), forbidding bipolar bridge cells and yielding open dendritic trees (`Betti-1 = 0`).
    - Under Feugier Model B / Scarpella et al. (2006) saturating exocytosis (`Phi_1 = Phi_2 = Phi_max`
      when both faces exceed the saturation knee `K`), the cross-multiplied relative growth rate is:
        `P2 * (k1 * Phi_max * P_cyt - k2 * P1) - P1 * (k1 * Phi_max * P_cyt - k2 * P2)`
        `= k1 * Phi_max * P_cyt * (P2 - P1)`,
      which restores `P1 = P2` (`d(P1 - P2)/dt = -k2 * (P1 - P2)`), stabilizing bipolar PIN1 bridge
      cells that connect secondary strands into closed reticulate venation loops (`Betti-1 > 0`). -/
def model_a_quadratic_face_rhs (k1 k2 P_cyt dS P : Int) : Int :=
  k1 * (P * dS) * (P * dS) * P_cyt - k2 * P

def model_b_saturated_face_rhs (k1 k2 Phi_max P_cyt P : Int) : Int :=
  k1 * Phi_max * P_cyt - k2 * P

theorem feugier_2005_winner_take_all_vs_saturating_bipolar
    (k1 k2 Phi_max P_cyt dS P1 P2 : Int) :
    P2 * model_a_quadratic_face_rhs k1 k2 P_cyt dS P1 -
      P1 * model_a_quadratic_face_rhs k1 k2 P_cyt dS P2 =
      k1 * P_cyt * dS * dS * P1 * P2 * (P1 - P2) ∧
    P2 * model_b_saturated_face_rhs k1 k2 Phi_max P_cyt P1 -
      P1 * model_b_saturated_face_rhs k1 k2 Phi_max P_cyt P2 =
      k1 * Phi_max * P_cyt * (P2 - P1) := by
  constructor
  · dsimp [model_a_quadratic_face_rhs]
    simp [Int.mul_sub, Int.mul_comm, Int.mul_left_comm]
    omega
  · dsimp [model_b_saturated_face_rhs]
    simp [Int.mul_sub, Int.mul_comm, Int.mul_left_comm]
    omega


-- ============================================================================
-- Law 3 (`feugier-carrier-driven-canalization-3`):
-- Feugier et al. (2005) Directional Carrier-Mediated Auxin Flux Antisymmetry Law
-- ============================================================================
/-- In Feugier, Mochizuki & Iwasa (2005) J. Theor. Biol. 236:313-325 (Eq. 2), net auxin flux
    across the shared cell wall from hexagonal cell `i` to neighbor `j` is:
      `J_{i->j} = P_ij * S_i - P_ji * S_j + D * (S_i - S_j)`.
    This theorem proves that:
    1. Pairwise intercellular flux is strictly antisymmetric (`J_{i->j} + J_{j->i} = 0`),
       guaranteeing exact auxin conservation across every interior cell-wall interface.
    2. When `S_i = S_j = S`, passive diffusion vanishes and net flux equals `(P_ij - P_ji) * S`. -/
def carrier_auxin_flux (P_ij P_ji D S_i S_j : Int) : Int :=
  P_ij * S_i - P_ji * S_j + D * (S_i - S_j)

theorem feugier_2005_flux_antisymmetry_and_polarity (P_ij P_ji D S_i S_j S : Int) :
    carrier_auxin_flux P_ij P_ji D S_i S_j +
      carrier_auxin_flux P_ji P_ij D S_j S_i = 0 ∧
    carrier_auxin_flux P_ij P_ji D S S = (P_ij - P_ji) * S := by
  constructor
  · dsimp [carrier_auxin_flux]
    simp [Int.mul_sub]
    omega
  · dsimp [carrier_auxin_flux]
    simp [Int.sub_mul]

end MathOfBiology.Loop_035
