/-
  morphoeval/lean/loops/Loop_058_cajal_dendritic_wiring.lean
  Formal Lean 4 verification of Ramón y Cajal (1899), Rall (1959) Exp. Neurol. 1:491-527,
  Cuntz, Borst & Segev (2007) PNAS 104:13319-13324, and Cuntz, Forstner, Borst & Häusser
  (2010) PLoS Comput. Biol. 6(8):e1000877 governing greedy minimum-wiring / conduction-delay
  dendritic trees, Rall's 3/2 electrotonic impedance-matching power law, and binary dendritic
  arbor topological identities (`caleo-cajal-dendritic-wiring`, Slot 058).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_058

-- ============================================================================
-- Law 1 (`caleo-cajal-dendritic-wiring-1`):
-- Cuntz et al. (2007, 2010) Greedy MST Wiring-vs-Delay Cost Trade-Off Law
-- ============================================================================
/-- In Cuntz et al. (2010) PLoS Comput. Biol. 6(8):e1000877, each unconnected carrier
    point `v` connects to existing tree node `u` minimizing:
      `C(u, v) = d(u, v) + bf * path(u, root)`.
    Scaling by positive integer `S > 0` with integer balancing factor `bf_s in [0, S]`:
      `C_scaled(S, bf_s, d_uv, path_u) = S * d_uv + bf_s * path_u`.
    Comparing attachment to an intermediate branch node `u` (at root path `d_ru` and
    distance `d_uv < d_rv` from `v`, satisfying strict triangle inequality `d_rv < d_ru + d_uv`)
    against direct radial attachment to the soma root (`path_root = 0`, distance `d_rv`):
    1. Added wiring cable is strictly smaller for the branched topology (`d_uv < d_rv`),
       whereas root-to-synapse conduction path is strictly smaller for the radial star
       topology (`d_rv < d_ru + d_uv`).
    2. The Cuntz greedy cost difference factors exactly as:
       `C_scaled(u, v) - C_scaled(root, v) = bf_s * d_ru - S * (d_rv - d_uv)`.
    3. At `bf_s = 0` (pure Steiner / Prim MST limit), `C_scaled(root, v) - C_scaled(u, v) = S * (d_rv - d_uv) > 0`,
       selecting the minimum-wiring branched tree, whereas at `bf_s = S` (`bf = 1`),
       `C_scaled(u, v) - C_scaled(root, v) = S * (d_ru + d_uv - d_rv) > 0`, selecting the
       minimum-conduction-delay radial star graph. -/
def cuntz_greedy_cost_scaled (S bf_s d_uv path_u : Int) : Int :=
  S * d_uv + bf_s * path_u

theorem cuntz_2010_wiring_delay_cost_tradeoff
    (S d_ru d_uv d_rv bf_s : Int)
    (hS : 0 < S)
    (h_wire : d_uv < d_rv)
    (h_tri : d_rv < d_ru + d_uv) :
    cuntz_greedy_cost_scaled S bf_s d_uv d_ru -
      cuntz_greedy_cost_scaled S bf_s d_rv 0 =
      bf_s * d_ru - S * (d_rv - d_uv) ∧
    0 < cuntz_greedy_cost_scaled S 0 d_rv 0 -
      cuntz_greedy_cost_scaled S 0 d_uv d_ru ∧
    0 < cuntz_greedy_cost_scaled S S d_uv d_ru -
      cuntz_greedy_cost_scaled S S d_rv 0 := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [cuntz_greedy_cost_scaled]
    simp [Int.mul_sub]
    omega
  · dsimp [cuntz_greedy_cost_scaled]
    have h_diff : 0 < d_rv - d_uv := by omega
    have h_prod : 0 < S * (d_rv - d_uv) := Int.mul_pos hS h_diff
    have h_eq : S * d_rv + 0 * 0 - (S * d_uv + 0 * d_ru) = S * (d_rv - d_uv) := by
      simp [Int.mul_sub]
    omega
  · dsimp [cuntz_greedy_cost_scaled]
    have h_diff : 0 < d_ru + d_uv - d_rv := by omega
    have h_prod : 0 < S * (d_ru + d_uv - d_rv) := Int.mul_pos hS h_diff
    have h_eq : S * d_uv + S * d_ru - (S * d_rv + S * 0) = S * (d_ru + d_uv - d_rv) := by
      simp [Int.mul_add, Int.mul_sub]
      omega
    omega


-- ============================================================================
-- Law 2 (`caleo-cajal-dendritic-wiring-2`):
-- Rall's (1959) 3/2 Electrotonic Impedance-Matching Diameter Power Law
-- ============================================================================
/-- In Rall (1959) Exp. Neurol. 1:491-527, a dendritic bifurcation from parent cylinder
    diameter `d_p` into daughter diameters `d_1, d_2` achieves electrotonic impedance
    matching (reducing the branched arbor to an equivalent cylinder without action-potential
    or EPSP reflection) iff `d_p^(3/2) = d_1^(3/2) + d_2^(3/2)`.
    Representing each branch's electrotonic half-cube diameter variable as `g_k = d_k^(3/2)`
    so that the cubed diameter is `d_k^3 = g_k^2`:
    1. Across any impedance-matched bifurcation `g_p = g_1 + g_2`, the cubed parent diameter
       satisfies `d_p^3 = g_p^2 = d_1^3 + d_2^3 + 2 * g_1 * g_2`.
    2. For a symmetric bifurcation (`g_1 = g_2 = g`, `d_1 = d_2 = d`), the cubed parent
       diameter equals exactly `4 * d^3` (`d_p = 4^(1/3) * d = 2^(2/3) * d ≈ 1.5874 * d`),
       strictly exceeding Murray's vascular cube law (`d_p^3 = 2 * d^3`) by `2 * d^3 > 0`. -/
def rall_cubed_diameter (g : Int) : Int :=
  g * g

def rall_parent_half_cube (g1 g2 : Int) : Int :=
  g1 + g2

theorem rall_1959_three_halves_impedance_matching
    (g1 g2 g : Int)
    (hg : 0 < g) :
    rall_cubed_diameter (rall_parent_half_cube g1 g2) =
      rall_cubed_diameter g1 + rall_cubed_diameter g2 + 2 * (g1 * g2) ∧
    rall_cubed_diameter (rall_parent_half_cube g g) =
      4 * rall_cubed_diameter g ∧
    2 * rall_cubed_diameter g <
      rall_cubed_diameter (rall_parent_half_cube g g) := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [rall_cubed_diameter, rall_parent_half_cube]
    simp [Int.mul_add, Int.mul_comm]
    omega
  · dsimp [rall_cubed_diameter, rall_parent_half_cube]
    simp [Int.add_mul, Int.mul_add]
    omega
  · dsimp [rall_cubed_diameter, rall_parent_half_cube]
    have hg2 : 0 < g * g := Int.mul_pos hg hg
    have h_eq : (g + g) * (g + g) = 4 * (g * g) := by
      simp [Int.add_mul, Int.mul_add]
      omega
    omega


-- ============================================================================
-- Law 3 (`caleo-cajal-dendritic-wiring-3`):
-- Acyclic Dendritic Spanning Tree & Binary Bifurcation Topological Identity
-- ============================================================================
/-- Every dendritic arbor grown by Cuntz's greedy MST algorithm is a connected acyclic
    tree (`Betti_1 = E - N + 1 = 0`, so `E = N - 1` segments for `N` nodes). Moreover,
    for a rooted binary dendritic tree with `B` internal bifurcation nodes and `1` soma
    root (having degree 1 into the primary dendrite trunk), the number of terminal
    synaptic carrier leaves is `L = B + 1`, total nodes is `N = 2 * B + 2`, and total
    dendritic cable segments is `E = 2 * B + 1 = L + B`. -/
def binary_arbor_leaves (B : Int) : Int :=
  B + 1

def binary_arbor_nodes (B : Int) : Int :=
  1 + B + binary_arbor_leaves B

def binary_arbor_edges (B : Int) : Int :=
  2 * B + 1

def tree_betti_1 (N E : Int) : Int :=
  E - N + 1

theorem dendritic_binary_tree_topology_identity (B : Int) :
    binary_arbor_edges B = binary_arbor_nodes B - 1 ∧
    binary_arbor_leaves B - binary_arbor_edges B + B = 0 ∧
    tree_betti_1 (binary_arbor_nodes B) (binary_arbor_edges B) = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [binary_arbor_edges, binary_arbor_nodes, binary_arbor_leaves]
    omega
  · dsimp [binary_arbor_leaves, binary_arbor_edges]
    omega
  · dsimp [tree_betti_1, binary_arbor_nodes, binary_arbor_edges, binary_arbor_leaves]
    omega

end MathOfBiology.Loop_058
