/-
  morphoeval/lean/loops/Loop_100_hydractinia_stolon_vascular_net.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 100:
  Buss, Anderson & Perry (2015) Biol. Bull. 228(2):146-158, Dudgeon et al.
  (1999) Am. Nat. 153(3):292-309 & Blackstone (1999) Syst. Biol. 48(1):75-85 —
  Gastrovascular Flow and Stolonal Network Morphogenesis in Hydractinia
  (`badin-hydractinia-stolon-vascular-net`).

  Covers 3 governing mathematical laws in pure core Lean 4 (zero external imports,
  zero `sorry`s):
    1. Buss, Anderson & Perry (2015) & Dudgeon et al. (1999) Hagen-Poiseuille
       Quartic Hydraulic Conductance vs Wall Shear Stress Scaling & Kirchhoff Law:
       In a chitin-sheathed stolon of lumen radius `r`, Hagen-Poiseuille hydraulic
       conductance scales as the fourth power `G(r) = r^4`, volumetric lumen flux
       numerator is `Q_ij = r^4 * (p_i - p_j)`, and wall shear stress numerator
       `tau_w propto Q_ij / r^3 = r * (p_i - p_j)` satisfies the exact algebraic
       identity `Q_ij = r^3 * tau_w_num`. Directed fluxes are strictly antisymmetric
       (`Q_ij + Q_ji = 0`) and conserve gastrovascular volume at every Kirchhoff node.
    2. Blackstone (1999) & Buss (2001) Planar Stolon Graph Euler Cyclomatic Loop
       Number (`b_1 = E - V + 1`) & Runner-to-Sheet Anastomosis Increment Law:
       An un-anastomosed runner-like colony tree (`Podocoryna` limit, `E = V - 1`)
       has zero topological loops (`b_1 = 0`), whereas `k > 0` tip-to-stolon
       anastomosis fusion events (`E = V - 1 + k`) create exactly `b_1 = k > 0`
       closed gastrovascular loops (`Hydractinia` stolonal sheet morph), strictly
       increasing planar meshedness numerator across the runner-to-sheet continuum.
    3. Buss, Anderson & Perry (2015) Polyp Contraction Arrest / Excision
       Shear-Starvation Stolon Regression Law:
       When polyp volume-displacement pumping is arrested (`Delta_V_p = 0`, e.g.,
       polyp excision or MgCl2 paralysis), nodal pressure gradient vanishes
       (`p_i = p_j`), driving lumen flow `Q_ij = 0` and wall shear stress
       `tau_w = 0 < tau_reg`, so stolon lumen radius change `dr/dt = shear_gain - k_d * r`
       is strictly negative (`dr/dt = -k_d * r < 0`) for any open lumen `r > 0`
       and positive decay rate `k_d > 0`, forcing stolon regression.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_100_hydractinia_stolon_vascular_net.lean
-/

namespace MathOfBiology.Loop_100

-- ============================================================================
-- Law 1 (`badin-hydractinia-stolon-vascular-net-1`):
-- Hagen-Poiseuille Quartic Conductance vs Cubic Shear Stress Scaling & Kirchhoff Law
-- ============================================================================

/-- Fourth-power Hagen-Poiseuille lumen hydraulic conductance factor `G(r) = r^4`. -/
def hagen_poiseuille_r4 (r : Int) : Int :=
  r * r * r * r

/-- Cubic lumen radius factor `r^3` relating volumetric flow `Q` to wall shear stress `tau_w`. -/
def lumen_radius_r3 (r : Int) : Int :=
  r * r * r

/-- Hagen-Poiseuille volumetric stolon flow numerator `Q_ij = r^4 * (p_i - p_j)`. -/
def stolon_poiseuille_flux (r pi pj : Int) : Int :=
  hagen_poiseuille_r4 r * (pi - pj)

/-- Wall shear stress numerator `tau_w_num = r * (p_i - p_j)` since `tau_w propto Q / r^3 = r * dp`. -/
def stolon_wall_shear_num (r pi pj : Int) : Int :=
  r * (pi - pj)

/-- Kirchhoff gastrovascular volume balance residual at a 3-stolon junction: `Q_in - Q_out1 - Q_out2`. -/
def kirchhoff_stolon_residual (q_in q_out1 q_out2 : Int) : Int :=
  q_in - q_out1 - q_out2

/-- Theorem 1 (`buss_dudgeon_poiseuille_quartic_conductance_and_shear_identity`):
    (1) Volumetric stolon flux equals cubic lumen radius times wall shear stress numerator:
        `stolon_poiseuille_flux r pi pj = lumen_radius_r3 r * stolon_wall_shear_num r pi pj`,
    (2) Doubling lumen radius from `r = 1` to `r = 2` increases hydraulic conductance 16-fold
        (`hagen_poiseuille_r4 2 = 16 * hagen_poiseuille_r4 1`),
    (3) Directed Hagen-Poiseuille stolon flux is strictly antisymmetric (`Q_ij + Q_ji = 0`),
    (4) Kirchhoff nodal volume conservation `q_in = q1 + q2` gives zero net residual. -/
theorem buss_dudgeon_poiseuille_quartic_conductance_and_shear_identity (r pi pj q1 q2 : Int) :
    stolon_poiseuille_flux r pi pj = lumen_radius_r3 r * stolon_wall_shear_num r pi pj ∧
    hagen_poiseuille_r4 2 = 16 * hagen_poiseuille_r4 1 ∧
    stolon_poiseuille_flux r pi pj + stolon_poiseuille_flux r pj pi = 0 ∧
    kirchhoff_stolon_residual (q1 + q2) q1 q2 = 0 := by
  refine ⟨?_, by decide, ?_, ?_⟩
  · simp [stolon_poiseuille_flux, hagen_poiseuille_r4, lumen_radius_r3, stolon_wall_shear_num, Int.mul_assoc]
  · simp [stolon_poiseuille_flux, Int.mul_sub]
    omega
  · simp [kirchhoff_stolon_residual]
    omega


-- ============================================================================
-- Law 2 (`badin-hydractinia-stolon-vascular-net-2`):
-- Planar Stolon Graph Euler Cyclomatic Loop Number & Runner-to-Sheet Anastomosis
-- ============================================================================

/-- First Betti / Euler cyclomatic loop number `b_1 = E - V + 1` of a connected planar stolon graph. -/
def stolon_cyclomatic_loops (V E : Int) : Int :=
  E - V + 1

/-- Planar meshedness numerator `E - V + 1` comparing runner (`k = 0`) vs sheet (`k > 0`) colonies. -/
def runner_to_sheet_loop_gain (V k_runner k_sheet : Int) : Int :=
  stolon_cyclomatic_loops V (V - 1 + k_sheet) - stolon_cyclomatic_loops V (V - 1 + k_runner)

/-- Theorem 2 (`blackstone_buss_runner_to_sheet_anastomosis_loop_law`):
    (1) Any un-anastomosed runner-like stolon tree (`E = V - 1`, `Podocoryna` tree limit)
        has zero closed gastrovascular loops (`stolon_cyclomatic_loops V (V - 1) = 0`),
    (2) After `k` tip-to-stolon anastomosis fusion events (`E = V - 1 + k`), the colony
        contains exactly `b_1 = k` closed gastrovascular loops (`stolon_cyclomatic_loops V (V - 1 + k) = k`),
    (3) Whenever a sheet-morph colony (`Hydractinia` or 2,4-DNP-treated `Podocoryna`)
        undergoes strictly more anastomoses than a runner colony (`k_runner < k_sheet`),
        its cyclomatic loop count and planar meshedness numerator are strictly greater
        (`0 < runner_to_sheet_loop_gain V k_runner k_sheet`). -/
theorem blackstone_buss_runner_to_sheet_anastomosis_loop_law (V k k_runner k_sheet : Int)
    (h_anast : k_runner < k_sheet) :
    stolon_cyclomatic_loops V (V - 1) = 0 ∧
    stolon_cyclomatic_loops V (V - 1 + k) = k ∧
    0 < runner_to_sheet_loop_gain V k_runner k_sheet := by
  refine ⟨?_, ?_, ?_⟩
  · simp [stolon_cyclomatic_loops]
    omega
  · simp [stolon_cyclomatic_loops]
    omega
  · simp [runner_to_sheet_loop_gain, stolon_cyclomatic_loops]
    omega


-- ============================================================================
-- Law 3 (`badin-hydractinia-stolon-vascular-net-3`):
-- Polyp Contraction Arrest / Excision Shear-Starvation Stolon Regression Law
-- ============================================================================

/-- Stolon lumen radius adaptation rate `dr/dt = shear_growth - k_d * r`. -/
def stolon_lumen_radius_rate (shear_growth k_d r : Int) : Int :=
  shear_growth - k_d * r

/-- Theorem 3 (`buss_polyp_arrest_shear_starvation_stolon_regression`):
    (1) Under polyp contraction arrest or polyp excision (`pi = pj`), volumetric lumen
        flow vanishes (`stolon_poiseuille_flux r pi pi = 0`),
    (2) Wall shear stress vanishes (`stolon_wall_shear_num r pi pi = 0`), falling strictly
        below any positive regression threshold `0 < tau_reg`,
    (3) With zero shear-driven lumen growth (`shear_growth = 0`), any open stolon lumen
        (`0 < r`) with positive perisarc regression rate (`0 < k_d`) experiences a strictly
        negative radius derivative (`stolon_lumen_radius_rate 0 k_d r < 0`), forcing
        stolon lumen collapse and regression. -/
theorem buss_polyp_arrest_shear_starvation_stolon_regression (r pi tau_reg k_d : Int)
    (htau : 0 < tau_reg) (hkd : 0 < k_d) (hr : 0 < r) :
    stolon_poiseuille_flux r pi pi = 0 ∧
    stolon_wall_shear_num r pi pi = 0 ∧
    stolon_wall_shear_num r pi pi < tau_reg ∧
    stolon_lumen_radius_rate 0 k_d r < 0 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [stolon_poiseuille_flux]
  · simp [stolon_wall_shear_num]
  · simp [stolon_wall_shear_num]
    exact htau
  · simp [stolon_lumen_radius_rate]
    have hmul : 0 < k_d * r := Int.mul_pos hkd hr
    omega

end MathOfBiology.Loop_100
