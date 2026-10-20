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

end MathOfBiology.Loop_036
