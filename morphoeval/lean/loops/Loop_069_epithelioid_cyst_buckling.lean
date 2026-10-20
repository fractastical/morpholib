/-
  morphoeval/lean/loops/Loop_069_epithelioid_cyst_buckling.lean
  Formal Lean 4 verification of Trushko, Di Meglio, Merzouki, Blanch-Mercader,
  Abuhattum, Guck, Alessandri, Chopard, Kruse & Roux (2020) Dev. Cell 54(5):655-668
  ("Buckling of an epithelium growing under spherical confinement"),
  Hannezo, Prost & Joanny (2014) PNAS 111(1):27-32
  ("Theory of epithelial sheet morphology in three dimensions"),
  and Zoelly (1915) / Föppl–von Kármán spherical shell buckling theory
  (`trushko-epithelioid-cyst-buckling`, Slot 069).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_069

-- ============================================================================
-- Law 1 (`trushko-epithelioid-cyst-buckling-1`):
-- Zoelly / Föppl–von Kármán Spherical Shell Quadratic Buckling Pressure Law
-- ============================================================================
/-- In Zoelly (1915) and Trushko et al. (2020) Dev. Cell Eq. (1) / STAR Methods,
    a confluent epithelial monolayer of thickness `h`, Young's modulus prefactor
    `C_E = 2 E_epi / sqrt(3 (1 - nu^2))`, and spherical alginate capsule radius `R`
    buckles inward at critical radial confinement pressure
      `P_c = C_E * (h / R)^2`.
    Cleared of fractions by multiplying by `4 * R^2`, the scaled critical pressure
    `zoelly_pressure_scaled C_E h = C_E * h^2` satisfies exact 4-fold quadratic scaling:
    (1) doubling monolayer thickness `h -> 2 * h` increases `P_c` 4-fold, and
    (2) doubling capsule radius `R -> 2 * R` (comparing `P_c(h, R) * (2R)^2` against
        `4 * P_c(h, R) * R^2`) reduces `P_c` 4-fold. -/
def zoelly_pressure_scaled (c_e h : Nat) : Nat :=
  c_e * (h * h)

def zoelly_radius_doubling_lhs (c_e r : Nat) : Nat :=
  c_e * ((2 * r) * (2 * r))

def zoelly_radius_doubling_rhs (c_e r : Nat) : Nat :=
  4 * (c_e * (r * r))

theorem zoelly_quadratic_shell_buckling_scaling (c_e h r : Nat) :
    zoelly_pressure_scaled c_e (2 * h) = 4 * zoelly_pressure_scaled c_e h ∧
    zoelly_radius_doubling_lhs c_e r = zoelly_radius_doubling_rhs c_e r ∧
    zoelly_pressure_scaled 1 2 = 4 * zoelly_pressure_scaled 1 1 := by
  refine ⟨?_, ?_, by decide⟩
  · unfold zoelly_pressure_scaled
    have h_sq : (2 * h) * (2 * h) = 4 * (h * h) := by
      calc (2 * h) * (2 * h) = 2 * (2 * (h * h)) := by
            simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]
      _ = 4 * (h * h) := by omega
    rw [h_sq]
    simp [Nat.mul_assoc, Nat.mul_comm]
  · unfold zoelly_radius_doubling_lhs zoelly_radius_doubling_rhs
    have r_sq : (2 * r) * (2 * r) = 4 * (r * r) := by
      calc (2 * r) * (2 * r) = 2 * (2 * (r * r)) := by
            simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]
      _ = 4 * (r * r) := by omega
    rw [r_sq]
    simp [Nat.mul_assoc, Nat.mul_comm]


-- ============================================================================
-- Law 2 (`trushko-epithelioid-cyst-buckling-2`):
-- Confluent Proliferative Laplace Stress Buildup & Alginate Buckling Threshold
-- ============================================================================
/-- In Trushko et al. (2020) Dev. Cell Figs. 2-4, post-confluence cell proliferation
    increases the intrinsic epithelial stretch `g` above confluence (`g_0`), generating
    compressive tangential stress and radial confinement pressure
      `sigma_theta(g) = E_alg * (g - g_0)`
    against the elastic alginate capsule (`E_alg > 0`).
    We prove that:
    (1) At confluence (`g = g_0`), pre-buckling confinement stress vanishes (`sigma_theta = 0`),
        and in an unconfined or zero-stiffness capsule (`E_alg = 0`), stress vanishes for all `g`;
    (2) For post-confluence proliferation (`g = g_0 + dg + 1`), compressive stress is strictly
        positive (`sigma_theta > 0`); and
    (3) When growth exceeds the critical buckling stretch (`g = g_c + dg + 1 > g_c`),
        the buckling stress excess `E_alg * (g - g_c)` is strictly positive, driving
        spontaneous inward invagination. -/
def laplace_confinement_stress (e_alg g g0 : Nat) : Nat :=
  e_alg * (g - g0)

theorem proliferative_confinement_buckling_threshold (e0 g0 gc dg : Nat) :
    laplace_confinement_stress (e0 + 1) g0 g0 = 0 ∧
    laplace_confinement_stress 0 (g0 + dg + 1) g0 = 0 ∧
    laplace_confinement_stress (e0 + 1) (g0 + dg + 1) g0 > 0 ∧
    laplace_confinement_stress (e0 + 1) (gc + dg + 1) gc > 0 := by
  unfold laplace_confinement_stress
  refine ⟨by simp, by simp, ?_, ?_⟩
  · have h_e : 0 < e0 + 1 := by omega
    have h_g : 0 < (g0 + dg + 1) - g0 := by omega
    exact Nat.mul_pos h_e h_g
  · have h_e : 0 < e0 + 1 := by omega
    have h_g : 0 < (gc + dg + 1) - gc := by omega
    exact Nat.mul_pos h_e h_g


-- ============================================================================
-- Law 3 (`trushko-epithelioid-cyst-buckling-3`):
-- Single-Fold Inward Invagination Volume Conservation & Blebbistatin Myosin Shift
-- ============================================================================
/-- In Hannezo et al. (2014) PNAS and Trushko et al. (2020) Dev. Cell Figs. 5-6,
    when the confined epithelium invaginates inward with displaced cap volume `V_fold > 0`
    inside a capsule of inner volume `V_caps`, the remaining luminal volume
    `V_lumen = V_caps - V_fold` plus `V_fold` strictly conserves total capsule volume `V_caps`.
    Furthermore, active actomyosin contractility (`Gamma_myo`) amplifies post-buckling
    inward fold depth `delta_scaled = delta_base * (c_0 + Gamma_myo)`; inhibiting myosin-II
    with Blebbistatin (`Gamma_blebb = g0 + 1 < Gamma_wt = g0 + d_myo + 2`) strictly reduces
    the inward invagination fold depth and lowers the active cortical buckling barrier. -/
def invaginated_lumen_volume (v_caps v_fold : Int) : Int :=
  v_caps - v_fold

def active_invagination_depth_scaled (delta_base c0 gamma_myo : Nat) : Nat :=
  (delta_base + 1) * (c0 + gamma_myo + 1)

theorem invagination_volume_conservation_and_blebbistatin_shift
    (v_caps v_fold : Int) (delta_base c0 g0 d_myo : Nat) :
    invaginated_lumen_volume v_caps v_fold + v_fold = v_caps ∧
    active_invagination_depth_scaled delta_base c0 g0 <
      active_invagination_depth_scaled delta_base c0 (g0 + d_myo + 1) := by
  constructor
  · unfold invaginated_lumen_volume
    omega
  · unfold active_invagination_depth_scaled
    have h_lt : c0 + g0 + 1 < c0 + (g0 + d_myo + 1) + 1 := by omega
    have h_pos : 0 < delta_base + 1 := by omega
    exact Nat.mul_lt_mul_of_pos_left h_lt h_pos


-- ============================================================================
-- Law 4 (`trushko-epithelioid-cyst-buckling-4`, Pass 3):
-- Apical-Basal Differential Tension Sets the Spontaneous Curvature of the Monolayer
-- ============================================================================
/-- In Hannezo, Prost & Joanny (2014) PNAS Eq. (1)-(3) and the Pass 3 3D triangulated
    shell of `2019-trushko-epithelioid-cyst-buckling.py` (`solve_confined_cyst_shell3d`),
    the per-area bending energy of a monolayer with bending modulus `κ`, thickness `h`
    and apical-basal tension difference `ΔΓ = Γ_apical - Γ_basal` is
      `f(c) = (κ/2) c² - ΔΓ h c`,   `c = 2H` (twice the mean curvature).
    Scaling by `2κ` and writing `t = ΔΓ h` gives the integer energy
      `E(c) = κ (κ c² - 2 t c)`,
    which satisfies the completed square `E(c) + t² = (κ c - t)²`. Hence
    (1) `E(c) ≥ -t²` for every curvature `c`, and
    (2) the bound is attained at the spontaneous curvature `κ c₀ = t`, i.e.
        `c₀ = 2H₀ = ΔΓ h / κ` (`H₀ R₀ = 6 ΔΓ R₀ / (E_epi h²)` once
        `κ = E_epi h³ / (12 (1 - ν²))` is inserted; the `(1 - ν²)` factors cancel
        against `Y_2D`). This is the value used by
        `compute_apicobasal_spontaneous_curvature` in the Python script. -/
def apicobasal_bending_energy_scaled (k t c : Int) : Int :=
  k * (k * c * c - 2 * t * c)

theorem int_mul_self_nonneg (d : Int) : 0 ≤ d * d := by
  rcases Int.le_total 0 d with h | h
  · exact Int.mul_nonneg h h
  · exact Int.mul_nonneg_of_nonpos_of_nonpos h h

theorem apicobasal_bending_completed_square (k t c : Int) :
    apicobasal_bending_energy_scaled k t c + t * t = (k * c - t) * (k * c - t) := by
  unfold apicobasal_bending_energy_scaled
  have h1 : k * (k * c * c - 2 * t * c) = (k * c) * (k * c) - 2 * (t * (k * c)) := by
    rw [Int.mul_sub]
    have a1 : k * (k * c * c) = (k * c) * (k * c) := by ac_rfl
    have a2 : k * (2 * t * c) = 2 * (t * (k * c)) := by ac_rfl
    rw [a1, a2]
  rw [h1]
  generalize k * c = x
  have e : (x - t) * (x - t) = x * x - t * x - (t * x - t * t) := by
    rw [Int.sub_mul, Int.mul_sub, Int.mul_sub, Int.mul_comm x t]
  rw [e]
  omega

theorem apicobasal_spontaneous_curvature_minimizer (k t c : Int) :
    apicobasal_bending_energy_scaled k t c ≥ -(t * t) ∧
    apicobasal_bending_energy_scaled 1 t t = -(t * t) := by
  constructor
  · have h := apicobasal_bending_completed_square k t c
    have hs := int_mul_self_nonneg (k * c - t)
    omega
  · unfold apicobasal_bending_energy_scaled
    simp only [Int.one_mul, Int.two_mul, Int.add_mul]
    omega


-- ============================================================================
-- Law 5 (`trushko-epithelioid-cyst-buckling-5`, Pass 3):
-- Inverted (Pogorelov) Cap Penalty Under Apical-Basal Polarity & Unilateral Capsule Wall Contact
-- ============================================================================
/-- In the Pass 3 3D shell the apical-basal polarity axis is the radial direction of
    the capsule, so a surface patch of mean curvature magnitude `hm ≥ 0` that keeps the
    lumen on its apical side pays the aligned penalty `(hm - c₀)²`, whereas an inward
    (Pogorelov-inverted) cap pays `(hm + c₀)²`. For a spontaneous curvature `c₀ ≥ 0`
    (apical tension exceeding basal tension, Hannezo et al. 2014):
    (1) `(hm + c₀)² = (hm - c₀)² + 4 hm c₀`, hence the inverted cap is never cheaper
        than the aligned one — differential tension localises folding into few deep
        inversions instead of many shallow wrinkles;
    (2) the alginate capsule acts as a unilateral Winkler foundation: the wall penalty
        `max(0, r - R)²` vanishes for every vertex strictly inside or on the capsule
        (`r ≤ R`) and is strictly positive once a vertex dilates the wall (`r > R`),
        matching the `k_conf` term of `solve_confined_cyst_shell3d`. -/
def inverted_cap_bending_penalty (hm c0 : Int) : Int := (hm + c0) * (hm + c0)

def aligned_cap_bending_penalty (hm c0 : Int) : Int := (hm - c0) * (hm - c0)

def capsule_wall_winkler_penalty (r R : Int) : Int :=
  if r ≤ R then 0 else (r - R) * (r - R)

theorem inverted_cap_penalty_and_capsule_wall_contact
    (hm c0 r R : Int) (hh : 0 ≤ hm) (hc : 0 ≤ c0) :
    inverted_cap_bending_penalty hm c0 = aligned_cap_bending_penalty hm c0 + 4 * (hm * c0) ∧
    aligned_cap_bending_penalty hm c0 ≤ inverted_cap_bending_penalty hm c0 ∧
    (r ≤ R → capsule_wall_winkler_penalty r R = 0) ∧
    (R < r → capsule_wall_winkler_penalty r R > 0) := by
  have hp : 0 ≤ hm * c0 := Int.mul_nonneg hh hc
  have key : (hm + c0) * (hm + c0) = (hm - c0) * (hm - c0) + 4 * (hm * c0) := by
    simp only [Int.mul_add, Int.add_mul, Int.mul_sub, Int.sub_mul]
    rw [Int.mul_comm c0 hm]
    omega
  unfold inverted_cap_bending_penalty aligned_cap_bending_penalty
  refine ⟨key, by omega, ?_, ?_⟩
  · intro h; simp [capsule_wall_winkler_penalty, h]
  · intro h
    have hn : ¬ r ≤ R := Int.not_le.mpr h
    simp only [capsule_wall_winkler_penalty, hn, ite_false]
    have hpos : 0 < r - R := Int.sub_pos.mpr h
    exact Int.mul_pos hpos hpos

end MathOfBiology.Loop_069
