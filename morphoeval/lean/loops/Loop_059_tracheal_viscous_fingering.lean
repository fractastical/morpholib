/-
  morphoeval/lean/loops/Loop_059_tracheal_viscous_fingering.lean
  Formal Lean 4 verification of Nelson, Gleghorn, Myers, Radisky & Stone (2017)
  Development 144(23):4328-4335 ("Microfluidic chest cavities reveal that transmural
  pressure controls the rate of lung development"), Kim et al. (2015) Dev. Cell
  34(6):668-681, and Unbekandt et al. (2008) Mech. Dev. 125:314-324 governing
  transmural pressure common-mode gauge invariance, Laplace circumferential epithelial
  wall stress monotonicity, and tracheal drainage / nifedipine hypoplasia arrest
  (`sugimura-tracheal-viscous-fingering`, Slot 059).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_059

-- ============================================================================
-- Law 1 (`sugimura-tracheal-viscous-fingering-1`):
-- Nelson et al. (2017) Microfluidic Chest Transmural Pressure Gauge Invariance Law
-- ============================================================================
/-- In Nelson et al. (2017) Development 144:4328-4335, a microfluidic chest cavity
    independently controls luminal airway pressure `P_lum` and pleural pressure `P_pl`.
    The effective morphogenetic driving pressure across the embryonic airway wall is
    the transmural pressure difference `ΔP(P_lum, P_pl) = P_lum - P_pl`:
    1. Common-mode gauge invariance: adding any hydrostatic shift `c` to both
       luminal and pleural compartments leaves `ΔP` strictly invariant:
         `ΔP(P_lum + c, P_pl + c) = ΔP(P_lum, P_pl)`.
    2. Positive luminal pressurization `(P, 0)` and negative pleural suction `(0, -P)`
       yield identical transmural pressure `ΔP = P`.
    3. Equal luminal and pleural pressurization `(P, P)` collapses `ΔP = 0`. -/
def transmural_pressure (P_lum P_pl : Int) : Int :=
  P_lum - P_pl

theorem nelson_2017_transmural_pressure_gauge_invariance
    (P_lum P_pl c P : Int) :
    transmural_pressure (P_lum + c) (P_pl + c) =
      transmural_pressure P_lum P_pl ∧
    transmural_pressure P 0 = transmural_pressure 0 (-P) ∧
    transmural_pressure P P = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [transmural_pressure]
    omega
  · dsimp [transmural_pressure]
    omega
  · dsimp [transmural_pressure]
    omega


-- ============================================================================
-- Law 2 (`sugimura-tracheal-viscous-fingering-2`):
-- Laplace–Young Circumferential Epithelial Wall Stress Monotonicity Law
-- ============================================================================
/-- By the thin-shell Laplace–Young law (`σ_θ = ΔP * R / h`), the scaled circumferential
    hoop tension across a distal epithelial lung bud of positive radius `R > 0` and
    wall thickness `h > 0` is proportional to `σ_scaled(ΔP, R) = ΔP * R`:
    1. Hoop stress is invariant under any common-mode pressure shift `(P_lum + c, P_pl + c)`.
    2. Whenever `ΔP_2 > ΔP_1` and `R > 0`, the epithelial wall hoop stress strictly
       increases: `σ_scaled(ΔP_1, R) < σ_scaled(ΔP_2, R)` with exact increment
       `(ΔP_2 - ΔP_1) * R > 0`. -/
def laplace_hoop_stress_scaled (dP R : Int) : Int :=
  dP * R

theorem laplace_epithelial_hoop_stress_monotonicity
    (P_lum P_pl c dP1 dP2 R : Int)
    (hR : 0 < R)
    (hdP : dP1 < dP2) :
    laplace_hoop_stress_scaled (transmural_pressure (P_lum + c) (P_pl + c)) R =
      laplace_hoop_stress_scaled (transmural_pressure P_lum P_pl) R ∧
    laplace_hoop_stress_scaled dP2 R - laplace_hoop_stress_scaled dP1 R =
      (dP2 - dP1) * R ∧
    laplace_hoop_stress_scaled dP1 R < laplace_hoop_stress_scaled dP2 R := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [laplace_hoop_stress_scaled, transmural_pressure]
    have h_eq : P_lum + c - (P_pl + c) = P_lum - P_pl := by omega
    rw [h_eq]
  · dsimp [laplace_hoop_stress_scaled]
    simp [Int.sub_mul]
  · dsimp [laplace_hoop_stress_scaled]
    have h_diff : 0 < dP2 - dP1 := by omega
    have h_prod : 0 < (dP2 - dP1) * R := Int.mul_pos h_diff hR
    have h_sub : (dP2 - dP1) * R = dP2 * R - dP1 * R := Int.sub_mul dP2 dP1 R
    omega


-- ============================================================================
-- Law 3 (`sugimura-tracheal-viscous-fingering-3`):
-- Tracheal Drainage & Nifedipine ASM Blockade Hypoplasia Arrest Law
-- ============================================================================
/-- In Nelson et al. (2017) & Kim et al. (2015), terminal bud bifurcation requires both
    super-threshold transmural distension `ΔP > ΔP_c` (`excess_dP = max(0, ΔP - ΔP_c)`)
    and active airway smooth muscle (ASM) Ca2+ peristaltic contractility `α_asm >= 0`.
    Representing the unnormalized quadratic Hill branching numerator as
      `Γ_num(α_asm, dP, dP_c) = α_asm * (max 0 (dP - dP_c))^2`:
    1. Tracheal drainage or sub-threshold pressure (`dP <= dP_c`) forces `Γ_num = 0`.
    2. Complete L-type Ca2+ channel ASM blockade (`α_asm = 0`, nifedipine limit) forces
       `Γ_num = 0` at every transmural pressure `dP`.
    3. Super-threshold physiological distension (`dP > dP_c`) with active ASM (`α_asm > 0`)
       yields strictly positive branching rate `0 < Γ_num(α_asm, dP, dP_c)`. -/
def excess_pressure_pos (dP dP_c : Int) : Int :=
  if dP <= dP_c then 0 else dP - dP_c

def branching_rate_numerator (alpha_asm dP dP_c : Int) : Int :=
  alpha_asm * (excess_pressure_pos dP dP_c * excess_pressure_pos dP dP_c)

theorem tracheal_drainage_and_nifedipine_hypoplasia_arrest
    (alpha_asm dP_sub dP_sup dP_c : Int)
    (h_alpha : 0 < alpha_asm)
    (h_sub : dP_sub <= dP_c)
    (h_sup : dP_c < dP_sup) :
    branching_rate_numerator alpha_asm dP_sub dP_c = 0 ∧
    branching_rate_numerator 0 dP_sup dP_c = 0 ∧
    0 < branching_rate_numerator alpha_asm dP_sup dP_c := by
  refine ⟨?_, ?_, ?_⟩
  · dsimp [branching_rate_numerator, excess_pressure_pos]
    simp [h_sub]
  · dsimp [branching_rate_numerator]
    omega
  · dsimp [branching_rate_numerator, excess_pressure_pos]
    have h_not : ¬(dP_sup <= dP_c) := by omega
    simp [h_not]
    have h_diff : 0 < dP_sup - dP_c := by omega
    have h_sq : 0 < (dP_sup - dP_c) * (dP_sup - dP_c) := Int.mul_pos h_diff h_diff
    exact Int.mul_pos h_alpha h_sq

end MathOfBiology.Loop_059
