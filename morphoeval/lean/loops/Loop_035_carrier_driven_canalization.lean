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


-- ============================================================================
-- Law 4 (`feugier-carrier-driven-canalization-4`, Pass 3):
-- Lockhart (1965) J. Theor. Biol. 8:264-275 Yield-Threshold Biomechanical Growth Law
-- ============================================================================
/-- Lockhart's irreversible-expansion law, reduced to blade area in the Pass 3 growing-blade module:
      `(1/A) dA/dt = phi * max(0, P - Y)`,
    i.e. the relative growth rate is `phi * (P - Y)` when turgor `P` exceeds the wall yield threshold `Y`
    and exactly `0` otherwise. With `Y(t) = Y0 + kappa t` (wall hardening) the blade expands while
    `P > Y(t)` and arrests once `Y(t) >= P`. This theorem proves the three properties the script relies on:
    1. no growth at or below threshold (`P <= Y -> rate = 0`),
    2. strictly positive growth above threshold for positive extensibility (`Y < P -> 0 < rate`),
    3. monotonicity in turgor (`P <= P' -> rate P <= rate P'`). -/
def lockhart_growth_rate (phi P Y : Int) : Int :=
  if Y < P then phi * (P - Y) else 0

theorem lockhart_1965_yield_threshold_growth (phi P P' Y : Int) (hphi : 0 < phi) :
    (P ≤ Y → lockhart_growth_rate phi P Y = 0) ∧
    (Y < P → 0 < lockhart_growth_rate phi P Y) ∧
    (P ≤ P' → lockhart_growth_rate phi P Y ≤ lockhart_growth_rate phi P' Y) := by
  refine ⟨?_, ?_, ?_⟩
  · intro h
    unfold lockhart_growth_rate
    have : ¬ Y < P := Int.not_lt.mpr h
    simp [this]
  · intro h
    unfold lockhart_growth_rate
    simp [h]
    exact Int.mul_pos hphi (Int.sub_pos.mpr h)
  · intro h
    unfold lockhart_growth_rate
    by_cases h1 : Y < P
    · have h2 : Y < P' := Int.lt_of_lt_of_le h1 h
      simp [h1, h2]
      exact Int.mul_le_mul_of_nonneg_left (by omega) (Int.le_of_lt hphi)
    · by_cases h2 : Y < P'
      · simp [h1, h2]
        exact Int.mul_nonneg (Int.le_of_lt hphi) (by omega)
      · simp [h1, h2]


-- ============================================================================
-- Law 5 (`feugier-carrier-driven-canalization-5`, Pass 3):
-- Vein-Graph Growth-Event Betti-1 Bookkeeping Law
-- ============================================================================
/-- The procambial strand graph has cycle rank (first Betti number) `beta_1 = E - V + C`
    (edges - vertices + connected components). During blade growth each canalization event is one of:
    1. free-end extension of a strand into a newly born cell (`E+1, V+1, C`): `beta_1` unchanged;
    2. a new isolated preprocambial seed (`E, V+1, C+1`): `beta_1` unchanged;
    3. fusion of a strand with a vein of its own component — Scarpella et al. (2004) "subsequent fusion
       of freely ending domains" through a bipolar cell (`E+1, V, C`): `beta_1 + 1`, one areole closed;
    4. fusion joining two previously separate components (`E+1, V, C-1`): `beta_1` unchanged.
    Hence closed areoles can only be created by same-component fusion events, which is exactly what the
    commitment lock (Law 6) preserves and the open-tree regime (winner-take-all) forbids. -/
def betti1 (E V C : Int) : Int := E - V + C

theorem vein_graph_growth_event_betti1 (E V C : Int) :
    betti1 (E + 1) (V + 1) C = betti1 E V C ∧
    betti1 E (V + 1) (C + 1) = betti1 E V C ∧
    betti1 (E + 1) V C = betti1 E V C + 1 ∧
    betti1 (E + 1) V (C - 1) = betti1 E V C := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> (dsimp [betti1]; omega)


-- ============================================================================
-- Law 6 (`feugier-carrier-driven-canalization-6`, Pass 3):
-- Scarpella et al. (2004) Irreversible Procambial Commitment Lock Law
-- ============================================================================
/-- Once a membrane face has sustained `P >= P_commit` for the commitment delay `tau_commit`, the Pass 3
    module applies the irreversible lock `P <- max P P_commit` at every subsequent step (Scarpella, Francis &
    Berleth 2004: preprocambial domains that reach procambial identity are not lost). This theorem proves:
    1. a locked face never drops below the commitment level (`P_commit <= lock P`),
    2. the lock never lowers a face (`P <= lock P`), so Law 1 pool accounting is only ever rescaled, not reversed,
    3. the lock is the identity on faces already at or above threshold (`P_commit <= P -> lock P = P`),
    4. an unlocked face is untouched. -/
def commitment_lock (locked : Bool) (P P_commit : Int) : Int :=
  if locked then (if P < P_commit then P_commit else P) else P

theorem procambial_commitment_lock_monotone (P P_commit : Int) :
    P_commit ≤ commitment_lock true P P_commit ∧
    P ≤ commitment_lock true P P_commit ∧
    (P_commit ≤ P → commitment_lock true P P_commit = P) ∧
    commitment_lock false P P_commit = P := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold commitment_lock
    by_cases h : P < P_commit <;> simp [h] <;> omega
  · unfold commitment_lock
    by_cases h : P < P_commit <;> simp [h] <;> omega
  · intro h
    unfold commitment_lock
    have : ¬ P < P_commit := Int.not_lt.mpr h
    simp [this]
  · simp [commitment_lock]

end MathOfBiology.Loop_035
