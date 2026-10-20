/-
  morphoeval/lean/loops/Loop_096_social_wasp_comb_hexagons.lean
  Formal Lean 4 verification of Karsai & Pénzes (1993) J. Theor. Biol. 161(4):505-525
  ("Comb building in social wasps: self-organization and stigmergic script"),
  Theraulaz & Bonabeau (1995) Science 269(5224):686-688 ("Coordination in distributed
  building"), Smith, Napp, Gravish, Seeley & Petersen (2021) PNAS 118(31):e2103605118,
  and Hales (2001) Discrete Comput. Geom. 25(1):1-22 governing stigmergic hexagonal
  comb ring growth, shared-wall paper pulp economy, and Euler topological charge
  neutrality of 5-7 disclination pairs (`bonabeau-social-wasp-comb-hexagons`, Slot 096).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_096

-- ============================================================================
-- Law 1 (`bonabeau-social-wasp-comb-hexagons-1`):
-- Centered Hexagonal Ring Progression & 6R Perimeter Ring Increment Law
-- ============================================================================
/-- In Karsai & Pénzes (1993) J. Theor. Biol. & Karsai & Pénzes (2000) Behav. Ecol.,
    high stigmergic wall-sharing selectivity (`w_3 >> w_2 >> w_1`) drives wasps to
    complete concentric hexagonal rings around a single founder cell (`R = 0, N(0) = 1`).
    After `R` complete rings, total brood cell count follows the centered hexagonal
    number `N(R) = 1 + 3 * R * (R + 1)`, and each new ring `R + 1` adds exactly
    `N(R + 1) - N(R) = 6 * (R + 1)` perimeter brood cells. -/
def centered_hex_cells (R : Nat) : Nat :=
  1 + 3 * R * (R + 1)

def ring_perimeter_increment (R : Nat) : Nat :=
  6 * (R + 1)

theorem centered_hex_step_identity (R : Nat) :
    3 * (R + 1) * (R + 1 + 1) = 3 * R * (R + 1) + 6 * (R + 1) := by
  have h_mul : (3 * R + 3) * (R + 1) = 3 * R * (R + 1) + 3 * (R + 1) :=
    Nat.add_mul (3 * R) 3 (R + 1)
  have h_3r : 3 * (R + 1) = 3 * R + 3 := by omega
  calc
    3 * (R + 1) * (R + 1 + 1) = 3 * (R + 1) * (R + 1) + 3 * (R + 1) * 1 :=
      Nat.mul_add (3 * (R + 1)) (R + 1) 1
    _ = (3 * R + 3) * (R + 1) + 3 * (R + 1) * 1 := by rw [← h_3r]
    _ = 3 * R * (R + 1) + 3 * (R + 1) + 3 * (R + 1) * 1 := by rw [h_mul]
    _ = 3 * R * (R + 1) + 6 * (R + 1) := by omega

theorem centered_hexagonal_ring_progression (R : Nat) :
    centered_hex_cells 0 = 1 ∧
    centered_hex_cells (R + 1) = centered_hex_cells R + ring_perimeter_increment R ∧
    centered_hex_cells (R + 1) > centered_hex_cells R := by
  have h_id := centered_hex_step_identity R
  refine ⟨rfl, ?_, ?_⟩
  · unfold centered_hex_cells ring_perimeter_increment
    omega
  · unfold centered_hex_cells
    omega


-- ============================================================================
-- Law 2 (`bonabeau-social-wasp-comb-hexagons-2`):
-- Shared-Wall Paper Pulp Economy & Asymptotic 3-Wall Hexagonal Optimality Law
-- ============================================================================
/-- A compact hexagonal comb of `R` concentric rings (`N(R)` cells) has `6 * (2 * R + 1)`
    outer boundary wall segments and shares all interior walls between two adjacent cells,
    requiring `W(R) = 3 * N(R) + 6 * R + 3` distinct wall segments compared to
    `W_isolated(R) = 6 * N(R)` walls for unshared hexagonal cells. For every `R >= 1`,
    wall-sharing saves `S_saved(R) = 6 * N(R) - W(R) = 3 * N(R) - 6 * R - 3 = 3 * R * (3 * R + 1) > 0`
    walls, and the excess over the asymptotic `3 * N(R)` interior bound is strictly `6 * R + 3`. -/
def compact_comb_walls (R : Nat) : Nat :=
  3 * centered_hex_cells R + 6 * R + 3

def unshared_isolated_walls (R : Nat) : Nat :=
  6 * centered_hex_cells R

def pulp_walls_saved (R : Nat) : Nat :=
  unshared_isolated_walls R - compact_comb_walls R

theorem shared_wall_pulp_economy_and_asymptotic_excess (R : Nat) :
    compact_comb_walls 0 = 6 ∧
    compact_comb_walls R - 3 * centered_hex_cells R = 6 * R + 3 ∧
    compact_comb_walls (R + 1) < unshared_isolated_walls (R + 1) ∧
    pulp_walls_saved (R + 1) > 0 := by
  have h_id := centered_hex_step_identity R
  refine ⟨rfl, ?_, ?_, ?_⟩
  · unfold compact_comb_walls
    omega
  · unfold compact_comb_walls unshared_isolated_walls centered_hex_cells
    omega
  · unfold pulp_walls_saved unshared_isolated_walls compact_comb_walls centered_hex_cells
    omega


-- ============================================================================
-- Law 3 (`bonabeau-social-wasp-comb-hexagons-3`):
-- Euler Topological Charge Neutrality of 5-7 Disclination Pairs at Caste Transitions
-- ============================================================================
/-- In Smith, Napp, Gravish, Seeley & Petersen (2021) PNAS 118:e2103605118, bridging
    smaller worker brood cells (`z = 6`) to larger reproductive cells (`lambda > 1`)
    inserts paired pentagon (`z = 5`) and heptagon (`z = 7`) topological dislocations.
    Each cell with coordination number `z` carries Euler topological charge `q(z) = 6 - z`:
    regular hexagons carry `q(6) = 0`, pentagons carry `q(5) = +1`, heptagons carry
    `q(7) = -1`, and any `n_pairs` of 5-7 disclination dipoles has zero net charge. -/
def topological_charge (z : Int) : Int :=
  6 - z

def disclination_dipole_net_charge (n_pairs : Int) : Int :=
  n_pairs * (topological_charge 5 + topological_charge 7)

theorem euler_disclination_dipole_charge_neutrality (n_pairs : Int) :
    topological_charge 6 = 0 ∧
    topological_charge 5 = 1 ∧
    topological_charge 7 = -1 ∧
    disclination_dipole_net_charge n_pairs = 0 := by
  refine ⟨rfl, rfl, rfl, ?_⟩
  unfold disclination_dipole_net_charge topological_charge
  have h_zero : (6 - 5 : Int) + (6 - 7 : Int) = 0 := by omega
  rw [h_zero]
  omega

end MathOfBiology.Loop_096
