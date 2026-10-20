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


-- ============================================================================
-- Law 4 (`mirabet-auxin-pin1-meristem-mechanics-4`, Pass 3):
-- Multi-layered L1/L2/L3 Pressurized Shell Load-Sharing Law (Hamant et al. 2008 SOM;
-- Kutschera & Niklas 2007 epidermal-growth-control; reduced compatible-strain layer stack)
-- ============================================================================
/-- In the Pass 3 three-layer (L1 epidermis / L2 sub-epidermis / L3 corpus) cell-centre
    mesh, the layers are glued by their shared anticlinal/periclinal walls, so under a
    common in-plane membrane strain `eps` each layer `j` carries the in-plane tension
    `N_j = k_j * eps` where `k_j` is its in-plane wall stiffness (the L1 outer periclinal
    wall is several-fold thicker, so `k_1 >> k_2, k_3`). This theorem proves that:
    1. the layer tensions sum to the total shell tension `(k_1 + k_2 + k_3) * eps`;
    2. the L1 load share equals its stiffness share in cross-multiplied form,
       `N_1 * (k_1 + k_2 + k_3) = k_1 * N_total` (i.e. `N_1 / N_total = k_1 / Σ k`);
    3. if the stack satisfies the spherical-dome Laplace-Young balance
       `2 * N_total = P * R`, then the summed layer tensions satisfy the same balance,
       so the single-layer thin-shell law of Law 1 is recovered as the aggregate. -/
def layer_membrane_tension (k eps : Int) : Int :=
  k * eps

theorem hamant_2008_multilayer_shell_load_sharing (k1 k2 k3 eps P R : Int)
    (h_laplace : 2 * ((k1 + k2 + k3) * eps) = P * R) :
    layer_membrane_tension k1 eps + layer_membrane_tension k2 eps +
        layer_membrane_tension k3 eps = (k1 + k2 + k3) * eps ∧
    layer_membrane_tension k1 eps * (k1 + k2 + k3) = k1 * ((k1 + k2 + k3) * eps) ∧
    2 * (layer_membrane_tension k1 eps + layer_membrane_tension k2 eps +
        layer_membrane_tension k3 eps) = P * R := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [layer_membrane_tension, Int.add_mul]
  · simp only [layer_membrane_tension]
    ac_rfl
  · simp only [layer_membrane_tension, Int.add_mul] at *
    omega


-- ============================================================================
-- Law 5 (`mirabet-auxin-pin1-meristem-mechanics-5`, Pass 3):
-- Louveaux et al. (2016) Tension-Aligned Anticlinal Division Plane & Symmetric
-- Volume Conservation Law (Reddy et al. 2004 L1/L2 anticlinal lineage; Sachs's rule)
-- ============================================================================
/-- In Louveaux et al. (2016) PNAS 113:E4294 the new cell wall of a symmetrically
    dividing L1/L2 cell is laid down along the direction of maximal tensile stress
    `e_1` and perpendicular to the surface (anticlinal). The Pass 3 mesh realises this
    by choosing the division-plane normal `n_div = n_s × e_1` where `n_s` is the local
    outward surface normal. With integer vector components, this theorem proves that
    `n_div` is orthogonal to both `n_s` and `e_1`, i.e. the new wall contains the
    surface normal (it is anticlinal) and contains the maximal-tension direction
    (it is tension-aligned), for every `n_s` and `e_1`. It also records Sachs's rule
    for the volume bookkeeping of the two daughters: `V_1 + V_2 = V_mother` with equal
    daughters implies `2 * V_1 = V_mother`. -/
def cross_x (a2 a3 b2 b3 : Int) : Int := a2 * b3 - a3 * b2
def cross_y (a1 a3 b1 b3 : Int) : Int := a3 * b1 - a1 * b3
def cross_z (a1 a2 b1 b2 : Int) : Int := a1 * b2 - a2 * b1
def dot3 (x1 x2 x3 y1 y2 y3 : Int) : Int := x1 * y1 + x2 * y2 + x3 * y3

theorem louveaux_2016_anticlinal_tension_division_plane
    (n1 n2 n3 e1 e2 e3 V1 V2 Vm : Int)
    (h_sum : V1 + V2 = Vm) (h_sym : V1 = V2) :
    dot3 (cross_x n2 n3 e2 e3) (cross_y n1 n3 e1 e3) (cross_z n1 n2 e1 e2) n1 n2 n3 = 0 ∧
    dot3 (cross_x n2 n3 e2 e3) (cross_y n1 n3 e1 e3) (cross_z n1 n2 e1 e2) e1 e2 e3 = 0 ∧
    2 * V1 = Vm := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [dot3, cross_x, cross_y, cross_z, Int.mul_sub, Int.mul_comm, Int.mul_left_comm]
    omega
  · simp only [dot3, cross_x, cross_y, cross_z, Int.mul_sub, Int.mul_comm, Int.mul_left_comm]
    omega
  · omega

end MathOfBiology.Loop_033
