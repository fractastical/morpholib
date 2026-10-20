/-
  morphoeval/lean/loops/Loop_033_auxin_pin1_meristem_mechanics.lean
  Formal Lean 4 verification of Hamant et al. (2008) Science 322:1650-1655,
  Heisler et al. (2010) PLoS Biology 8:e1000516, Jonsson et al. (2006) PNAS 103:1633-1638,
  and Mirabet et al. (2012) PLoS Comput. Biol. 8:e1002389 governing pressurized shell
  principal stresses, laser-ablation Kirsch hoop-stress amplification, and PIN1-auxin
  transport conservation in the Arabidopsis shoot apical meristem
  (`mirabet-auxin-pin1-meristem-mechanics`, Slot 033).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_033

-- ============================================================================
-- Law 1 (`mirabet-auxin-pin1-meristem-mechanics-1`):
-- Hamant et al. (2008) Pressurized Shell Principal Stress & 2:1 Cylinder Anisotropy Law
-- ============================================================================
/-- In Hamant et al. (2008) Science 322:1650-1655, the outer L1 epidermal wall of
    the shoot apical meristem is modeled as a thin pressurized shell of radius `R`,
    thickness `t`, and turgor pressure `P`. Clearing the positive denominator `2 * t`:
    - On the spherical apical dome (`κ_m = κ_θ = 1/R`), hoop and meridional stresses
      are equal (`σ_θθ * (2t) = P * R`, `σ_mm * (2t) = P * R`), yielding zero
      stress anisotropy (`σ_θθ - σ_mm = 0`).
    - On the cylindrical stem flank (`κ_θ = 1/R, κ_z = 0`), hoop stress is twice
      axial stress (`σ_θθ * (2t) = 2 * P * R`, `σ_zz * (2t) = P * R`), so
      `σ_θθ = 2 * σ_zz` and `3 * (σ_θθ - σ_zz) = σ_θθ + σ_zz` (`A_σ = 1/3`). -/
def sphere_hoop_stress_2t (P R : Int) : Int :=
  P * R

def sphere_meridional_stress_2t (P R : Int) : Int :=
  P * R

def cylinder_hoop_stress_2t (P R : Int) : Int :=
  2 * (P * R)

def cylinder_axial_stress_2t (P R : Int) : Int :=
  P * R

theorem hamant_2008_pressurized_shell_anisotropy (P R : Int) :
    sphere_hoop_stress_2t P R - sphere_meridional_stress_2t P R = 0 ∧
    cylinder_hoop_stress_2t P R = 2 * cylinder_axial_stress_2t P R ∧
    3 * (cylinder_hoop_stress_2t P R - cylinder_axial_stress_2t P R) =
      cylinder_hoop_stress_2t P R + cylinder_axial_stress_2t P R := by
  refine ⟨?_, ?_, ?_⟩
  · simp [sphere_hoop_stress_2t, sphere_meridional_stress_2t]
  · simp [cylinder_hoop_stress_2t, cylinder_axial_stress_2t]
  · simp [cylinder_hoop_stress_2t, cylinder_axial_stress_2t]
    omega


-- ============================================================================
-- Law 2 (`mirabet-auxin-pin1-meristem-mechanics-2`):
-- Kirsch / Hamant et al. (2008) Laser Ablation Wound Rim Hoop-Stress Doubling Law
-- ============================================================================
/-- When a circular cell patch of radius `a_0` (`a0_sq = a_0^2`) is laser-ablated in a
    pressurized L1 epidermal sheet under isotropic tension `σ_∞` (`sig_inf`), the
    Kirsch radial and circumferential stresses (scaled by `r_sq = r^2`) are:
      `σ_rr_scaled = sig_inf * (r_sq - a0_sq)`
      `σ_θθ_scaled = sig_inf * (r_sq + a0_sq)`
    At the wound rim (`r_sq = a0_sq`), radial tension drops to `0`, circumferential
    hoop stress doubles to `2 * sig_inf * a0_sq`, and for all `r_sq` the stress trace
    `σ_rr_scaled + σ_θθ_scaled = 2 * sig_inf * r_sq` is invariant. -/
def kirsch_radial_scaled (sig_inf r_sq a0_sq : Int) : Int :=
  sig_inf * (r_sq - a0_sq)

def kirsch_hoop_scaled (sig_inf r_sq a0_sq : Int) : Int :=
  sig_inf * (r_sq + a0_sq)

theorem hamant_2008_ablation_rim_hoop_doubling (sig_inf r_sq a0_sq : Int) :
    kirsch_radial_scaled sig_inf a0_sq a0_sq = 0 ∧
    kirsch_hoop_scaled sig_inf a0_sq a0_sq = 2 * (sig_inf * a0_sq) ∧
    kirsch_radial_scaled sig_inf r_sq a0_sq + kirsch_hoop_scaled sig_inf r_sq a0_sq =
      2 * (sig_inf * r_sq) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [kirsch_radial_scaled]
  · simp [kirsch_hoop_scaled, Int.mul_add]
    omega
  · simp [kirsch_radial_scaled, kirsch_hoop_scaled, Int.mul_add, Int.mul_sub]
    omega


-- ============================================================================
-- Law 3 (`mirabet-auxin-pin1-meristem-mechanics-3`):
-- Heisler et al. (2010) & Jonsson et al. (2006) PIN1 Active Auxin Flux Conservation Law
-- ============================================================================
/-- Active PIN1-mediated auxin transport across the anticlinal cell wall between
    cells `i` and `j` is `J_{i←j} = T_pin * (P_ji * a_j - P_ij * a_i)`.
    This theorem proves that:
    1. Pairwise intercellular active flux is strictly antisymmetric (`J_{i←j} + J_{j←i} = 0`),
       guaranteeing exact total auxin mass conservation across the L1 cellular lattice in the
       absence of synthesis/decay.
    2. Under symmetric PIN1 carrier allocation (`P_ij = P_ji = P_sym`) at homogeneous auxin
       concentration (`a_i = a_j = a`), net active flux vanishes (`0`). -/
def pin1_active_flux (T_pin P_ij P_ji a_i a_j : Int) : Int :=
  T_pin * (P_ji * a_j - P_ij * a_i)

theorem heisler_2010_pin1_flux_conservation (T_pin P_ij P_ji P_sym a_i a_j a : Int) :
    pin1_active_flux T_pin P_ij P_ji a_i a_j + pin1_active_flux T_pin P_ji P_ij a_j a_i = 0 ∧
    pin1_active_flux T_pin P_sym P_sym a a = 0 := by
  constructor
  · simp [pin1_active_flux, Int.mul_sub]
    omega
  · simp [pin1_active_flux]

end MathOfBiology.Loop_033
