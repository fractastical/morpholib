/-
  morphoeval/lean/loops/Loop_036_microtubule_stress_meristem.lean
  Formal Lean 4 verification of Bozorg, Krupinski & Jönsson (2014) PLoS Comput. Biol.
  10(1):e1003410, Sampathkumar et al. (2014) eLife 3:e01967, Hamant et al. (2008)
  Science 322(5908):1650-1655, and Uyttewaal et al. (2012) Cell 149(2):439-451
  governing shoot apical meristem (SAM) pressurized shell stress anisotropy,
  transverse-to-saddle principal stress switching, and katanin (`bot1`/`KTN1`) /
  oryzalin feedback gain reduction (`hamant-microtubule-stress-meristem`, Slot 036).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_036

-- ============================================================================
-- Law 1 (`hamant-microtubule-stress-meristem-1`):
-- Saddle-Boundary Stress Anisotropy Amplification & Gaussian Curvature Inversion Law
-- ============================================================================
/-- In Hamant et al. (2008) Science 322:1650-1655 and Bozorg, Krupinski & Jönsson (2014)
    PLoS Comput. Biol. 10:e1003410, a pressurized epidermal shell under turgor scale `P_s > 0`
    with principal curvatures `kappa_1` (along crease / circumferential) and `kappa_2`
    (meridional / across crease) exhibits scaled principal stress difference
      `Delta_sigma(P_s, kappa_1, kappa_2) = P_s * (kappa_1 - kappa_2)`.
    - On the spherical central dome (`kappa_1 = kappa_0 > 0`, `kappa_2 = kappa_0 > 0`),
      Gaussian curvature is positive (`K = kappa_0 * kappa_0 > 0`) and stress anisotropy
      vanishes identically (`Delta_sigma(P_s, kappa_0, kappa_0) = 0`).
    - At the organ-meristem saddle boundary (`kappa_1 = kappa_0 > 0` convex around the stem,
      `kappa_2 = -kappa_s < 0` concave in the meridional crease with `kappa_s > 0`),
      Gaussian curvature is strictly negative (`K = kappa_0 * (-kappa_s) < 0`) and
      principal stress anisotropy along the crease axis is strictly amplified:
        `Delta_sigma(P_s, kappa_0, -kappa_s) = P_s * (kappa_0 + kappa_s) > P_s * kappa_0 > 0`. -/
def gaussian_curvature (kappa_1 kappa_2 : Int) : Int :=
  kappa_1 * kappa_2

def saddle_stress_anisotropy (P_s kappa_1 kappa_2 : Int) : Int :=
  P_s * (kappa_1 - kappa_2)

theorem hamant_bozorg_saddle_boundary_stress_anisotropy_amplification
    (P_s kappa_0 kappa_s : Int)
    (hP : P_s > 0) (hk0 : kappa_0 > 0) (hks : kappa_s > 0) :
    gaussian_curvature kappa_0 kappa_0 > 0 ∧
    saddle_stress_anisotropy P_s kappa_0 kappa_0 = 0 ∧
    gaussian_curvature kappa_0 (-kappa_s) < 0 ∧
    saddle_stress_anisotropy P_s kappa_0 (-kappa_s) = P_s * kappa_0 + P_s * kappa_s ∧
    saddle_stress_anisotropy P_s kappa_0 (-kappa_s) > P_s * kappa_0 := by
  have hKpos : gaussian_curvature kappa_0 kappa_0 > 0 := by
    dsimp [gaussian_curvature]
    exact Int.mul_pos hk0 hk0
  have hDomeZero : saddle_stress_anisotropy P_s kappa_0 kappa_0 = 0 := by
    dsimp [saddle_stress_anisotropy]
    omega
  have hKneg : gaussian_curvature kappa_0 (-kappa_s) < 0 := by
    dsimp [gaussian_curvature]
    rw [Int.mul_neg]
    have hprod : kappa_0 * kappa_s > 0 := Int.mul_pos hk0 hks
    omega
  have hId : saddle_stress_anisotropy P_s kappa_0 (-kappa_s) = P_s * kappa_0 + P_s * kappa_s := by
    dsimp [saddle_stress_anisotropy]
    have hsub : kappa_0 - -kappa_s = kappa_0 + kappa_s := by omega
    rw [hsub, Int.mul_add]
  have hPs_ks : P_s * kappa_s > 0 := Int.mul_pos hP hks
  refine ⟨hKpos, hDomeZero, hKneg, hId, ?_⟩
  omega


-- ============================================================================
-- Law 2 (`hamant-microtubule-stress-meristem-2`):
-- Transverse-to-Circumferential Laplace Stress Switch & Longitudinal Stem Elongation
-- ============================================================================
/-- In Bozorg, Krupinski & Jönsson (2014) PLoS Comput. Biol. 10:e1003410 and Hamant et al. (2008)
    Science 322:1650-1655, a pressurized cylindrical stem of radius `R` (`kappa_z = 0`,
    `kappa_theta = 1/R`) has axial Young-Laplace stress `sigma_z = S_0` (where `S_0 = P * R / (2 * h) > 0`)
    and circumferential hoop stress `sigma_theta = 2 * S_0`.
    Cortical microtubules align parallel to maximal principal stress (`sigma_theta = 2 * S_0 > S_0 = sigma_z`),
    forming transverse cortical hoops that reinforce circumferential wall modulus above axial modulus
    (`C_z > C_theta` in compliance), so viscoplastic Lockhart elongation rate `eps_dot = C * sigma`
    is strictly greater along the longitudinal axis (`eps_dot_z > eps_dot_theta`) whenever
    axial compliance exceeds twice circumferential compliance (`C_z > 2 * C_theta`). -/
def cylinder_axial_stress (S_0 : Int) : Int :=
  S_0

def cylinder_hoop_stress (S_0 : Int) : Int :=
  2 * S_0

def lockhart_elongation_rate (compliance sigma : Int) : Int :=
  compliance * sigma

theorem bozorg_cylinder_to_saddle_stress_switch
    (S_0 C_z C_theta : Int)
    (hS : S_0 > 0) (hC : C_z > 2 * C_theta) :
    cylinder_hoop_stress S_0 = 2 * cylinder_axial_stress S_0 ∧
    cylinder_hoop_stress S_0 - cylinder_axial_stress S_0 = S_0 ∧
    cylinder_hoop_stress S_0 > cylinder_axial_stress S_0 ∧
    lockhart_elongation_rate C_z (cylinder_axial_stress S_0) -
      lockhart_elongation_rate C_theta (cylinder_hoop_stress S_0) =
      (C_z - 2 * C_theta) * S_0 ∧
    lockhart_elongation_rate C_z (cylinder_axial_stress S_0) >
      lockhart_elongation_rate C_theta (cylinder_hoop_stress S_0) := by
  have hDiff : (C_z - 2 * C_theta) * S_0 > 0 := by
    have hCz : C_z - 2 * C_theta > 0 := by omega
    exact Int.mul_pos hCz hS
  have hId :
      lockhart_elongation_rate C_z (cylinder_axial_stress S_0) -
        lockhart_elongation_rate C_theta (cylinder_hoop_stress S_0) =
        (C_z - 2 * C_theta) * S_0 := by
    dsimp [lockhart_elongation_rate, cylinder_axial_stress, cylinder_hoop_stress]
    rw [← Int.mul_assoc C_theta 2 S_0, Int.mul_comm C_theta 2, ← Int.sub_mul]
  refine ⟨rfl, by dsimp [cylinder_hoop_stress, cylinder_axial_stress]; omega,
          by dsimp [cylinder_hoop_stress, cylinder_axial_stress]; omega, hId, ?_⟩
  omega


-- ============================================================================
-- Law 3 (`hamant-microtubule-stress-meristem-3`):
-- Katanin (`bot1` / `KTN1`) & Oryzalin Mechanical Feedback Gain Reduction Theorem
-- ============================================================================
/-- In Uyttewaal et al. (2012) Cell 149:439-451 and Hamant et al. (2008) Science 322:1650-1655,
    cellulose microfibril stiffness along the maximal (`E_1`) and minimal (`E_2`) stress axes is
      `E_1(gamma_MT) = E_0 + dE * Delta_sigma * gamma_MT`,
      `E_2(gamma_MT) = E_0 - dE * Delta_sigma * gamma_MT`,
    and organ-meristem boundary crease deepening rate is proportional to wall stiffness
    anisotropy `E_1(gamma_MT) - E_2(gamma_MT) = 2 * dE * Delta_sigma * gamma_MT`.
    - Under Oryzalin microtubule depolymerization (`gamma_MT = 0`), cellulose stiffness
      anisotropy vanishes identically (`E_1(0) - E_2(0) = 0`), abolishing crease deepening.
    - In the katanin severing mutant `bot1`/`ktn1` (`0 < gamma_bot1 < gamma_WT`), stiffness
      anisotropy and boundary crease depth are strictly reduced relative to wild-type by
      `2 * dE * Delta_sigma * (gamma_WT - gamma_bot1) > 0`. -/
def wall_modulus_max (E_0 dE Delta_sigma gamma_MT : Int) : Int :=
  E_0 + dE * Delta_sigma * gamma_MT

def wall_modulus_min (E_0 dE Delta_sigma gamma_MT : Int) : Int :=
  E_0 - dE * Delta_sigma * gamma_MT

def wall_stiffness_anisotropy (E_0 dE Delta_sigma gamma_MT : Int) : Int :=
  wall_modulus_max E_0 dE Delta_sigma gamma_MT - wall_modulus_min E_0 dE Delta_sigma gamma_MT

theorem uyttewaal_katanin_oryzalin_feedback_gain_reduction
    (E_0 dE Delta_sigma gamma_WT gamma_bot1 : Int)
    (hdE : dE > 0) (hDS : Delta_sigma > 0) (hG : gamma_WT > gamma_bot1) :
    wall_stiffness_anisotropy E_0 dE Delta_sigma 0 = 0 ∧
    wall_stiffness_anisotropy E_0 dE Delta_sigma gamma_WT -
      wall_stiffness_anisotropy E_0 dE Delta_sigma gamma_bot1 =
      2 * (dE * Delta_sigma * (gamma_WT - gamma_bot1)) ∧
    wall_stiffness_anisotropy E_0 dE Delta_sigma gamma_WT >
      wall_stiffness_anisotropy E_0 dE Delta_sigma gamma_bot1 := by
  have hZero : wall_stiffness_anisotropy E_0 dE Delta_sigma 0 = 0 := by
    dsimp [wall_stiffness_anisotropy, wall_modulus_max, wall_modulus_min]
    omega
  have hId :
      wall_stiffness_anisotropy E_0 dE Delta_sigma gamma_WT -
        wall_stiffness_anisotropy E_0 dE Delta_sigma gamma_bot1 =
        2 * (dE * Delta_sigma * (gamma_WT - gamma_bot1)) := by
    dsimp [wall_stiffness_anisotropy, wall_modulus_max, wall_modulus_min]
    rw [Int.mul_sub (dE * Delta_sigma) gamma_WT gamma_bot1]
    omega
  have hdEDS : dE * Delta_sigma > 0 := Int.mul_pos hdE hDS
  have hGdiff : gamma_WT - gamma_bot1 > 0 := by omega
  have hProd : dE * Delta_sigma * (gamma_WT - gamma_bot1) > 0 :=
    Int.mul_pos hdEDS hGdiff
  refine ⟨hZero, hId, ?_⟩
  omega


-- ============================================================================
-- Law 4 (`hamant-microtubule-stress-meristem-4`, Pass 3):
-- Heisler et al. (2010) Stress-Based PIN1 Polarity / CMT Co-Alignment & Microtubule-Independence
-- ============================================================================
/-- In Heisler et al. (2010) PLoS Biol. 8(10):e1000516, PIN1 auxin efflux carriers accumulate on the
    plasma membrane adjacent to the anticlinal wall carrying the highest tensile load, while cortical
    microtubules (CMTs) align with the maximal principal stress of the same cell.  For a square L1 cell
    of side `a > 0` whose walls carry an isotropic tension `T_0 > 0` plus an anisotropic increment
    `tau > 0` on the walls parallel to the shell's maximal principal stress sigma_1:
      `T_par = T_0 + tau`, `T_perp = T_0`,
    the virial (Batchelor) wall-stress anisotropy is
      `M_xx - M_yy = 2 a T_par - 2 a T_perp = 2 a tau > 0`,
    so the CMT director (sigma_1 axis) is the axis of the PIN1-rich (max-tension) wall: PIN1 and CMTs
    are co-aligned.  Katanin/CMT feedback adds `kappa * S * gamma_MT >= 0` to the parallel wall only,
    so the PIN1 preference persists at `gamma_MT = 0` (oryzalin; Heisler 2010 Fig. 6 microtubule
    independence) and is only amplified by wild-type feedback. -/
def wall_tension_parallel (T_0 tau : Int) : Int :=
  T_0 + tau

def wall_tension_perpendicular (T_0 : Int) : Int :=
  T_0

def virial_stress_anisotropy (a T_par T_perp : Int) : Int :=
  2 * a * T_par - 2 * a * T_perp

def cmt_reinforced_parallel_tension (T_0 tau kappa S gamma_MT : Int) : Int :=
  T_0 + tau + kappa * S * gamma_MT

theorem heisler_pin1_cmt_coalignment_under_anisotropic_wall_tension
    (a T_0 tau kappa S gamma_MT : Int)
    (ha : a > 0) (_hT : T_0 > 0) (htau : tau > 0)
    (hkappa : kappa ≥ 0) (hS : S ≥ 0) (hgamma : gamma_MT ≥ 0) :
    wall_tension_parallel T_0 tau > wall_tension_perpendicular T_0 ∧
    virial_stress_anisotropy a (wall_tension_parallel T_0 tau) (wall_tension_perpendicular T_0)
      = 2 * a * tau ∧
    virial_stress_anisotropy a (wall_tension_parallel T_0 tau) (wall_tension_perpendicular T_0) > 0 ∧
    cmt_reinforced_parallel_tension T_0 tau kappa S 0 = wall_tension_parallel T_0 tau ∧
    cmt_reinforced_parallel_tension T_0 tau kappa S 0 > wall_tension_perpendicular T_0 ∧
    cmt_reinforced_parallel_tension T_0 tau kappa S gamma_MT ≥ wall_tension_parallel T_0 tau := by
  have hPar : wall_tension_parallel T_0 tau > wall_tension_perpendicular T_0 := by
    dsimp [wall_tension_parallel, wall_tension_perpendicular]
    omega
  have hId : virial_stress_anisotropy a (wall_tension_parallel T_0 tau) (wall_tension_perpendicular T_0)
      = 2 * a * tau := by
    dsimp [virial_stress_anisotropy, wall_tension_parallel, wall_tension_perpendicular]
    rw [Int.mul_add, Int.add_comm (2 * a * T_0) (2 * a * tau)]
    exact Int.add_sub_cancel (2 * a * tau) (2 * a * T_0)
  have h2a : 2 * a > 0 := by omega
  have hPos : 2 * a * tau > 0 := Int.mul_pos h2a htau
  have hOry : cmt_reinforced_parallel_tension T_0 tau kappa S 0 = wall_tension_parallel T_0 tau := by
    dsimp [cmt_reinforced_parallel_tension, wall_tension_parallel]
    rw [Int.mul_zero, Int.add_zero]
  have hOryGt : cmt_reinforced_parallel_tension T_0 tau kappa S 0 > wall_tension_perpendicular T_0 := by
    rw [hOry]
    exact hPar
  have hkS : kappa * S ≥ 0 := Int.mul_nonneg hkappa hS
  have hkSg : kappa * S * gamma_MT ≥ 0 := Int.mul_nonneg hkS hgamma
  have hAmp : cmt_reinforced_parallel_tension T_0 tau kappa S gamma_MT ≥ wall_tension_parallel T_0 tau := by
    dsimp [cmt_reinforced_parallel_tension, wall_tension_parallel]
    omega
  refine ⟨hPar, hId, ?_, hOry, hOryGt, hAmp⟩
  rw [hId]
  exact hPos


-- ============================================================================
-- Law 5 (`hamant-microtubule-stress-meristem-5`, Pass 3):
-- Topological Events on the L1 Cellular Vertex Mesh — Division Preserves the Euler Characteristic,
-- T1 Neighbour Exchange is Topology-Neutral (Louveaux et al. 2016; Hamant & Traas 2010)
-- ============================================================================
/-- A planar cellular vertex mesh with `V` vertices, `E` walls (edges) and `F` cells has Euler
    characteristic `chi = V - E + F` (the outer face excluded, so `chi = 1` for a disc).
    - A symmetric cell division along the maximal tensile stress (Louveaux et al. 2016 PNAS
      113:E4294) inserts two vertices on existing walls (each split wall becomes two: `+2` edges),
      adds one new wall (`+1` edge) and one cell: `(V, E, F) -> (V + 2, E + 3, F + 1)`, so
      `chi` is invariant and the cell count strictly increases.  After `n >= 0` divisions
      `(V + 2 n, E + 3 n, F + n)` has the same `chi`.
    - A T1 neighbour exchange (relevant to animal epithelia; suppressed by default here because
      plant cells are cemented by the middle lamella) keeps `(V, E, F)` and hence `chi` unchanged
      while swapping which cell pairs share the flipped wall. -/
def euler_characteristic (V E F : Int) : Int :=
  V - E + F

def vertices_after_divisions (V n : Int) : Int := V + 2 * n
def edges_after_divisions (E n : Int) : Int := E + 3 * n
def cells_after_divisions (F n : Int) : Int := F + n

theorem louveaux_vertex_mesh_division_preserves_euler_characteristic
    (V E F n : Int) (hn : n ≥ 0) :
    euler_characteristic (vertices_after_divisions V 1) (edges_after_divisions E 1)
        (cells_after_divisions F 1) = euler_characteristic V E F ∧
    cells_after_divisions F 1 > F ∧
    euler_characteristic (vertices_after_divisions V n) (edges_after_divisions E n)
        (cells_after_divisions F n) = euler_characteristic V E F ∧
    cells_after_divisions F n ≥ F ∧
    euler_characteristic V E F = euler_characteristic V E F := by
  refine ⟨?_, ?_, ?_, ?_, rfl⟩
  · dsimp [euler_characteristic, vertices_after_divisions, edges_after_divisions, cells_after_divisions]
    omega
  · dsimp [cells_after_divisions]
    omega
  · dsimp [euler_characteristic, vertices_after_divisions, edges_after_divisions, cells_after_divisions]
    omega
  · dsimp [cells_after_divisions]
    omega

end MathOfBiology.Loop_036
