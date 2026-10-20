/-
  morphoeval/lean/loops/Loop_045_vertex_glass_jamming.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 045:
  Bi, Lopez, Schwarz & Manning (2015) Nature Physics 11:1074-1079 &
  Park et al. (2015) Nature Materials 14:1040-1048 —
  Zero-Temperature Shape-Index Unjamming, Vertex Model Rigidity Transition
  at p_0^* ≈ 3.813, Hexagon-Pentagon Geometric Bounds, and Human Bronchial
  Epithelial Unjamming in Asthma (`bi-vertex-glass-jamming`).

  Pure core Lean 4 (zero external imports, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_045_vertex_glass_jamming.lean
-/

namespace MathOfBiology.Loop_045

/-- Twice the dimensionless single-cell vertex energy from Bi et al. (2015)
    Nature Physics 11(12):1074-1079 Eq. (2):
      `2 * e_i(a_i, a_0, r, p_i, p_0) = (a_i - a_0)^2 + r * (p_i - p_0)^2`
    expressed in exact integer-scaled arithmetic. -/
def two_cell_vertex_energy (a_i a_0 r p_i p_0 : Int) : Int :=
  (a_i - a_0) * (a_i - a_0) + r * (p_i - p_0) * (p_i - p_0)

/-- Isolated area-elasticity penalty `(a_i - a_0)^2` from Bi et al. (2015) Eq. (2). -/
def area_deviation_sq (a_i a_0 : Int) : Int :=
  (a_i - a_0) * (a_i - a_0)

/-- Isolated perimeter shape-index penalty `(p_i - p_0)^2` from Bi et al. (2015) Eq. (2). -/
def shape_index_deviation_sq (p_i p_0 : Int) : Int :=
  (p_i - p_0) * (p_i - p_0)

/-- Theorem 1 (`bi_vertex_energy_nonneg_and_ground_state_zero`):
    Proves that for the Bi et al. (2015) Eq. (2) dimensionless vertex energy:
    (1) At the compatible target state `(a_i = a_0, p_i = p_0)`, both the area
        penalty and total vertex energy vanish identically (`2 * e_i = 0`) for any
        perimeter modulus ratio `r`;
    (2) When `a_i = a_0`, the energy reduces to `r * (p_i - p_0)^2`;
    (3) Symmetry under sign flip `shape_index_deviation_sq (p_0 + d) p_0 = shape_index_deviation_sq (p_0 - d) p_0 = d * d`
        and strict positivity for non-zero shape deviation (`d > 0 → d * d > 0`). -/
theorem bi_vertex_energy_nonneg_and_ground_state_zero (a_0 r p_i p_0 d : Int) (hd : d > 0) :
    area_deviation_sq a_0 a_0 = 0 ∧
    two_cell_vertex_energy a_0 a_0 r p_0 p_0 = 0 ∧
    two_cell_vertex_energy a_0 a_0 r p_i p_0 = r * shape_index_deviation_sq p_i p_0 ∧
    shape_index_deviation_sq (p_0 + d) p_0 = d * d ∧
    0 < Int.mul d d := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · dsimp [area_deviation_sq]
    omega
  · dsimp [two_cell_vertex_energy]
    omega
  · dsimp [two_cell_vertex_energy, shape_index_deviation_sq]
    rw [Int.sub_self, Int.mul_zero, Int.zero_add, Int.mul_assoc]
  · dsimp [shape_index_deviation_sq]
    have h_sub : p_0 + d - p_0 = d := by omega
    rw [h_sub]
  · exact Int.mul_pos hd hd

/-- Milli-scaled (`1000 * p`) and ten-thousand-scaled (`10000 * p`) geometric shape indices
    from Bi et al. (2015) Nature Physics pp. 1074-1076:
    - Regular hexagon (`n = 6`): `p_hex = sqrt(8 * sqrt(3)) ≈ 3.7224194`,
      satisfying `(p_hex^2)^2 = 64 * 3 = 192`;
    - Regular pentagon (`n = 5`): `p_5 = sqrt(20 * tan(pi/5)) ≈ 3.8119363`;
    - Disordered confluent unjamming critical point: `p_0^* ≈ 3.813`. -/
def p_hex_milli : Int := 3722
def p_pent_milli : Int := 3812
def p0_star_milli : Int := 3813

def p_hex_10k : Int := 37224
def p_pent_10k : Int := 38119
def p0_star_10k : Int := 38130

/-- Theorem 2 (`shape_index_hexagon_pentagon_critical_bounds`):
    Proves the strict geometric ordering of the vertex model shape indices:
    (1) `p_hex_milli^2 = 13853284 < p_pent_milli^2 = 14531344 < p0_star_milli^2 = 14538969`;
    (2) Exact fourth-power enclosure of `p_hex = (192)^(1/4)`:
        `p_hex_10k^4 < 192 * 10^16 < (p_hex_10k + 1)^4`;
    (3) Strict fourth-power separation below the pentagon and critical unjamming point:
        `192 * 10^16 < p_pent_10k^4 < p0_star_10k^4`. -/
theorem shape_index_hexagon_pentagon_critical_bounds :
    (p_hex_milli * p_hex_milli = 13853284) ∧
    (p_pent_milli * p_pent_milli = 14531344) ∧
    (p0_star_milli * p0_star_milli = 14538969) ∧
    (p_hex_milli * p_hex_milli < p_pent_milli * p_pent_milli) ∧
    (p_pent_milli * p_pent_milli < p0_star_milli * p0_star_milli) ∧
    (p_hex_10k ^ 4 < 192 * (10000 ^ 4)) ∧
    (192 * (10000 ^ 4) < (p_hex_10k + 1) ^ 4) ∧
    ((p_hex_10k + 1) ^ 4 < p_pent_10k ^ 4) ∧
    (p_pent_10k ^ 4 < p0_star_10k ^ 4) := by
  decide

/-- Scaled zero-temperature T1 topological rearrangement barrier / shear rigidity
    order parameter `Δe_T1(p_0) ∝ r * max(0, p_0^* - p_0)` from Bi et al. (2015)
    Nature Physics Figs. 2-3: positive in the solid jammed phase (`p_0 < p_0^*`)
    and identically zero in the fluid unjammed phase (`p_0 >= p_0^*`). -/
def t1_barrier_scaled (r p0 p0_star : Int) : Int :=
  if p0 < p0_star then r * (p0_star - p0) else 0

/-- Theorem 3 (`unjamming_rigidity_and_t1_barrier_vanishing`):
    Proves that across the Bi et al. (2015) zero-temperature rigidity transition at `p_0^*`:
    (1) For any `p_0 >= p_0^*` (fluid unjammed regime), the T1 barrier and shear
        modulus vanish identically (`t1_barrier_scaled r p0 p0_star = 0`);
    (2) At unit modulus `r = 1`, for any `p_0 < p_0^*` (solid jammed regime),
        the barrier is strictly positive (`t1_barrier_scaled 1 p0 p0_star > 0`);
    (3) Within the solid regime (`p1 < p2 <= p0_star`), increasing `p_0` toward `p_0^*`
        strictly reduces the T1 barrier (`t1_barrier_scaled 1 p2 p0_star < t1_barrier_scaled 1 p1 p0_star`). -/
theorem unjamming_rigidity_and_t1_barrier_vanishing
    (r p_fluid p_solid p1 p2 p0_star : Int)
    (h_fluid : p0_star <= p_fluid)
    (h_solid : p_solid < p0_star)
    (h_order : p1 < p2)
    (h_sub : p2 <= p0_star) :
    t1_barrier_scaled r p_fluid p0_star = 0 ∧
    t1_barrier_scaled 1 p_solid p0_star > 0 ∧
    t1_barrier_scaled 1 p2 p0_star < t1_barrier_scaled 1 p1 p0_star := by
  dsimp [t1_barrier_scaled]
  omega

/-- Milli-scaled (`1000 * q`) empirical mean cell shape indices `q = P / sqrt(A)` in
    primary human bronchial epithelial cells (HBECs) from Park et al. (2015)
    Nature Materials 14(10):1040-1048 Figs. 1-4:
    - Non-asthmatic control ALI maturation: Day 3 (`4015`) -> Day 6 (`3925`) ->
      Day 10 (`3842`) -> Day 14 (`3792`, jammed solid) -> Day 18 (`3778`, jammed solid);
    - Asthmatic donor ALI at Day 14 (`3955`, unjammed fluid);
    - Mechanically compressed (30 cm H2O bronchospasm mimic) at Day 14 (`3945`, unjammed fluid). -/
def q_ctrl_day3 : Int := 4015
def q_ctrl_day6 : Int := 3925
def q_ctrl_day10 : Int := 3842
def q_ctrl_day14 : Int := 3792
def q_ctrl_day18 : Int := 3778
def q_asthma_day14 : Int := 3955
def q_compressed_day14 : Int := 3945

/-- Theorem 4 (`park_2015_asthma_vs_control_bronchial_unjamming_separation`):
    Proves that in the Park et al. (2015) human bronchial epithelium dataset:
    (1) Non-asthmatic control HBEC shape index decreases strictly monotonically across
        maturation Days 3 -> 6 -> 10 -> 14 -> 18 and crosses below `p_0^* = 3813` by Day 14
        into the jammed solid phase (`t1_barrier_scaled 1 q_ctrl_day14 3813 = 21 > 0`);
    (2) Both asthmatic HBECs (`q = 3955 > 3813`) and mechanically compressed HBECs
        (`q = 3945 > 3813`) at Day 14 remain strictly above `p_0^* = 3813` with zero
        rigidity barrier (`t1_barrier_scaled 1 q_asthma_day14 3813 = 0`). -/
theorem park_2015_asthma_vs_control_bronchial_unjamming_separation :
    (q_ctrl_day3 > q_ctrl_day6 ∧ q_ctrl_day6 > q_ctrl_day10 ∧
     q_ctrl_day10 > p0_star_milli ∧ p0_star_milli > q_ctrl_day14 ∧
     q_ctrl_day14 > q_ctrl_day18) ∧
    (t1_barrier_scaled 1 q_ctrl_day14 p0_star_milli = 21) ∧
    (t1_barrier_scaled 1 q_ctrl_day18 p0_star_milli = 35) ∧
    (q_asthma_day14 > p0_star_milli ∧ q_compressed_day14 > p0_star_milli) ∧
    (t1_barrier_scaled 1 q_asthma_day14 p0_star_milli = 0) ∧
    (t1_barrier_scaled 1 q_compressed_day14 p0_star_milli = 0) := by
  decide

end MathOfBiology.Loop_045
