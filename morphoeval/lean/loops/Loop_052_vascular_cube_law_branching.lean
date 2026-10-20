/-
  morphoeval/lean/loops/Loop_052_vascular_cube_law_branching.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 052:
  Murray (1926a, 1926b) Minimum-Work Vascular Cube Law & Optimal Bifurcation Angles
  (`murray-vascular-cube-law-branching`).

  Covers 5 governing mathematical laws in pure core Lean 4 (zero external imports,
  zero `sorry`s) -- Laws 4 and 5 were added in the Pass 3 remediation round:
    4. Fahraeus-Lindqvist Stationarity Shift: with radius-dependent viscosity
       `eta(r)` and `eps = d ln eta / d ln r`, the optimum satisfies
       `2 P_met = (4 - eps) P_visc`; `eps = 0` recovers Law 1 and `eps > 0` strictly lowers
       the metabolic share below Murray's `2 P_visc` (exponent drops below 3).
    5. Coupled Arteriovenous Cube Balance: `k_a C_a = k_v C_v` with `k_v < k_a` forces
       `C_a < C_v` (venous cubic radius sum exceeds arterial; AVR < 1).
    1. Murray (1926a) Poiseuille + Metabolic Stationarity Partition Law:
       `P(r) = A * r^{-4} + B * r^2` has stationarity residual `r^5 P'(r) = 2 B r^6 - 4 A = 0`,
       where metabolic blood-volume cost `B r^6 = 2 A` is strictly twice viscous Poiseuille
       pumping power `A`, and total optimal power `A + B r^6 = 3 A`.
    2. Murray (1926a) Cubic Radius Conservation & Multi-Generation Tree Invariance:
       Under `Q(r) = k_M * r^3`, parent-daughter flow balance `Q_0 = Q_1 + Q_2` preserves
       cubic radius sum `r_0^3 = r_1^3 + r_2^3` across single and recursive 2-generation
       arterial bifurcations `r_0^3 = r_{11}^3 + r_{12}^3 + r_{21}^3 + r_{22}^3`.
    3. Murray (1926b) Optimal Bifurcation Cosine Numerator Symmetry & Cancellation Law:
       For `N_1 = r_0^4 + r_1^4 - r_2^4` and `N_2 = r_0^4 + r_2^4 - r_1^4`, the asymmetric
       fourth-power terms cancel identically (`N_1 + N_2 = 2 r_0^4`) and symmetric daughters
       (`r_1 = r_2 = r`) satisfy `N_1(r_0, r, r) = N_2(r_0, r, r) = r_0^4`.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_052_vascular_cube_law_branching.lean
-/

namespace MathOfBiology.Loop_052

-- ============================================================================
-- Law 1 (`murray-vascular-cube-law-branching-1`):
-- Murray (1926a) Poiseuille Viscous Power + Metabolic Volume Cost Stationarity
-- ============================================================================

/-- Scaled stationarity residual `r^5 * dP/dr = 2 * p_met - 4 * p_visc` for
    Murray's power functional `P(r) = p_visc + p_met` where `p_visc ∝ r^{-4}` and `p_met ∝ r^2`. -/
def murray_stationarity_residual (p_visc p_met : Int) : Int :=
  2 * p_met - 4 * p_visc

/-- Total scaled vascular power `P_total = p_visc + p_met`. -/
def murray_total_power (p_visc p_met : Int) : Int :=
  p_visc + p_met

/-- Theorem 1 (`murray_stationarity_viscous_twice_metabolic`):
    At Murray's minimum-work radius where `p_met = 2 * p_visc`, the stationarity derivative
    vanishes identically (`murray_stationarity_residual p_visc (2 * p_visc) = 0`) and total
    vascular power equals exactly `3 * p_visc`. -/
theorem murray_stationarity_viscous_twice_metabolic (p_visc : Int) :
    murray_stationarity_residual p_visc (2 * p_visc) = 0 ∧
    murray_total_power p_visc (2 * p_visc) = 3 * p_visc := by
  simp [murray_stationarity_residual, murray_total_power]
  omega


-- ============================================================================
-- Law 2 (`murray-vascular-cube-law-branching-2`):
-- Murray (1926a) Cubic Radius Conservation Across Multi-Generation Bifurcations
-- ============================================================================

/-- Cubic vessel conductance proxy `r^3`. -/
def vessel_cube (r : Nat) : Nat :=
  r * r * r

/-- Murray's volumetric flow `Q(r) = k_M * r^3`. -/
def murray_flow (k_M r : Nat) : Nat :=
  k_M * vessel_cube r

/-- Theorem 2 (`murray_cube_law_generational_invariance`):
    (1) Volumetric flow across a bifurcation satisfies `Q(r_1) + Q(r_2) = k_M * (r_1^3 + r_2^3)`, and
    (2) Across a 2-generation arterial tree where `c_0 = c_1 + c_2`, `c_1 = c_11 + c_12`, and
        `c_2 = c_21 + c_22`, the root cubic radius `c_0` equals the sum of all four terminal
        granddaughter cubic radii `c_11 + c_12 + c_21 + c_22`. -/
theorem murray_cube_law_generational_invariance
    (k_M r1 r2 c0 c1 c2 c11 c12 c21 c22 : Nat)
    (h_gen1 : c0 = c1 + c2)
    (h_left : c1 = c11 + c12)
    (h_right : c2 = c21 + c22) :
    murray_flow k_M r1 + murray_flow k_M r2 = k_M * (vessel_cube r1 + vessel_cube r2) ∧
    c0 = c11 + c12 + c21 + c22 := by
  constructor
  · simp [murray_flow, Nat.left_distrib]
  · omega


-- ============================================================================
-- Law 3 (`murray-vascular-cube-law-branching-3`):
-- Murray (1926b) Optimal Bifurcation Cosine Numerator Symmetry & Cancellation
-- ============================================================================

/-- Fourth power `r^4` appearing in Murray's (1926b) optimal branching cosine law
    `cos(theta_1) = (r_0^4 + r_1^4 - r_2^4) / (2 * r_0^2 * r_1^2)`. -/
def pow4 (r : Int) : Int :=
  r * r * r * r

/-- Numerator `N_1(r_0, r_1, r_2) = r_0^4 + r_1^4 - r_2^4` of `cos(theta_1)`. -/
def murray_cos_num1 (r0 r1 r2 : Int) : Int :=
  pow4 r0 + pow4 r1 - pow4 r2

/-- Numerator `N_2(r_0, r_1, r_2) = r_0^4 + r_2^4 - r_1^4` of `cos(theta_2)`. -/
def murray_cos_num2 (r0 r1 r2 : Int) : Int :=
  pow4 r0 + pow4 r2 - pow4 r1

/-- Theorem 3 (`murray_bifurcation_cosine_symmetry`):
    (1) For any daughter radii `(r_1, r_2)`, the sum of the two branch cosine numerators
        cancels daughter asymmetry identically: `N_1 + N_2 = 2 * r_0^4`.
    (2) For symmetric daughters (`r_1 = r_2 = r`), both numerators equal `r_0^4` identically,
        proving symmetric deflection angles `theta_1 = theta_2`. -/
theorem murray_bifurcation_cosine_symmetry (r0 r1 r2 r : Int) :
    murray_cos_num1 r0 r1 r2 + murray_cos_num2 r0 r1 r2 = 2 * pow4 r0 ∧
    murray_cos_num1 r0 r r = pow4 r0 ∧
    murray_cos_num2 r0 r r = pow4 r0 := by
  simp [murray_cos_num1, murray_cos_num2]
  omega


-- ============================================================================
-- Law 4 (`murray-vascular-cube-law-branching-4`, Pass 3):
-- Fahraeus-Lindqvist Radius-Dependent Viscosity Shifts the Stationarity Partition
-- ============================================================================

/-- Scaled stationarity residual `r^5 * dP/dr = 2 * p_met - (4 - eps) * p_visc` of the
    generalised Murray functional `P(r) = p_visc(r) + p_met(r)` with `p_visc ∝ eta(r) r^{-4}`
    and logarithmic viscosity sensitivity `eps = d ln eta / d ln r` (`eps = 0` is Newtonian blood). -/
def fl_stationarity_residual (p_visc p_met eps : Int) : Int :=
  2 * p_met - (4 - eps) * p_visc

/-- Theorem 4 (`fahraeus_lindqvist_stationarity_shift`):
    (1) The Newtonian limit `eps = 0` recovers Murray's residual exactly;
    (2) the generalised optimum sits where `2 * p_met = (4 - eps) * p_visc`
        (residual vanishes identically there); and
    (3) whenever viscosity falls toward smaller vessels (`0 < eps`, the Fahraeus-Lindqvist
        regime) and `0 < p_visc`, the metabolic share at the optimum is *strictly below*
        Murray's `2 * p_visc`, i.e. the optimum is pushed toward narrower vessels and the
        effective exponent drops below 3. -/
theorem fahraeus_lindqvist_stationarity_shift (p_visc p_met eps : Int)
    (h_fl : 0 < eps) (h_pos : 0 < p_visc) :
    fl_stationarity_residual p_visc p_met 0 = murray_stationarity_residual p_visc p_met ∧
    fl_stationarity_residual (2 * p_visc) ((4 - eps) * p_visc) eps = 0 ∧
    (4 - eps) * p_visc < 4 * p_visc := by
  refine ⟨?_, ?_, ?_⟩
  · simp [fl_stationarity_residual, murray_stationarity_residual]
  · show 2 * ((4 - eps) * p_visc) - (4 - eps) * (2 * p_visc) = 0
    rw [Int.mul_left_comm]
    exact Int.sub_self _
  · exact Int.mul_lt_mul_of_pos_right (by omega) h_pos


-- ============================================================================
-- Law 5 (`murray-vascular-cube-law-branching-5`, Pass 3):
-- Coupled Arteriovenous Cube Balance on the Retinal Cup
-- ============================================================================

/-- Theorem 5 (`arteriovenous_coupled_cube_balance`):
    Closed retinal circulation forces arterial and venous Murray trees to carry equal flow,
    `k_a * C_a = k_v * C_v` where `C_a = ∑ r_tip^3` and `C_v = ∑ R_tip^3` are the terminal
    cubic radius sums.  If the venous conductance constant is smaller (`k_v < k_a`, i.e. lower
    venous wall shear `tau_v = sqrt(eta b_v) < tau_a`) and the arterial tree is non-trivial
    (`0 < C_a`), then the venous cubic sum strictly exceeds the arterial one, `C_a < C_v`:
    veins are necessarily wider than their paired arteries (clinical AVR < 1). -/
theorem arteriovenous_coupled_cube_balance (k_a k_v C_a C_v : Nat)
    (h_flow : k_a * C_a = k_v * C_v) (h_k : k_v < k_a) (h_pos : 0 < C_a) :
    C_a < C_v := by
  have h1 : k_v * C_a < k_a * C_a := Nat.mul_lt_mul_of_pos_right h_k h_pos
  rw [h_flow] at h1
  exact Nat.lt_of_mul_lt_mul_left h1

end MathOfBiology.Loop_052
