/-
  morphoeval/lean/loops/Loop_001_hydra_activator_inhibitor.lean
  Formal Lean 4 verification of the Gierer & Meinhardt (1972) Kybernetik 12:30-39
  short-range autocatalytic activator / long-range lateral inhibitor morphogenesis laws
  for Hydra vulgaris (`gierer-hydra-activator-inhibitor`, Slot 001).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_001

-- ============================================================================
-- Law 1 (`gierer-hydra-activator-inhibitor-1`):
-- Gierer & Meinhardt (1972) Eqs. (15a)-(15b) Homogeneous Steady-State Equilibrium
-- ============================================================================
/-- Clearing the inhibitor denominator `h` in Gierer & Meinhardt's activator reaction
    term `R_a = c * rho * a^2 / h - mu_a * a` yields the numerator residual
    `(a * a) * mu_h - a * h_scaled`. The inhibitor reaction residual is
    `R_h = (a * a) * mu_h - mu_h * h_scaled`.
    For integer-scaled parameters with unit source density `rho = 1` and equal
    production coefficients `c = c_prime = 1`, the homogeneous steady state
    `(a_*, h_*) = (mu_h / mu_a, (mu_h / mu_a)^2)` scaled by `mu_a` is
    `(a = mu_h, h_scaled = mu_h * mu_h)`: both activator and inhibitor reaction
    residuals vanish identically for all `mu_h : Int`. -/
def gm_activator_num_residual (a h_scaled mu_h : Int) : Int :=
  (a * a) * mu_h - a * h_scaled

def gm_inhibitor_num_residual (a h_scaled mu_h : Int) : Int :=
  (a * a) * mu_h - mu_h * h_scaled

theorem gierer_meinhardt_homogeneous_steady_state (mu_h : Int) :
    gm_activator_num_residual mu_h (mu_h * mu_h) mu_h = 0 ∧
    gm_inhibitor_num_residual mu_h (mu_h * mu_h) mu_h = 0 := by
  simp [gm_activator_num_residual, gm_inhibitor_num_residual, Int.mul_assoc]


-- ============================================================================
-- Law 2 (`gierer-hydra-activator-inhibitor-2`):
-- Gierer & Meinhardt (1972) Inhibitor Turnover & Equal-Diffusion Stability Theorem
-- ============================================================================
/-- At the basal-free Gierer-Meinhardt steady state (`rho_0 = 0`), the homogeneous
    reaction Jacobian has diagonal entries `J_11 = +mu_a` (autocatalytic self-activation)
    and `J_22 = -mu_h` (inhibitor decay), with trace `tr(J_0) = mu_a - mu_h` and
    determinant `det(J_0) = mu_a * mu_h`.
    When inhibitor turnover exceeds activator turnover (`mu_h = mu_a + d_mu + 1 > mu_a`
    with `det0 = d_pos + 1 > 0`), the negated Fourier-mode trace under equal diffusion
    `D_a = D_h = D` at wavenumber squared `q = k^2` is `(d_mu + 1) + 2*D*q ≥ 1` and
    the Fourier-mode determinant `det_J(q) = (D*q)^2 + (D*q)*(d_mu + 1) + (d_pos + 1)`
    is strictly positive (`≥ d_pos + 1`), proving equal diffusion cannot destabilize
    the spatially uniform state. -/
def gm_equal_diff_neg_trace (d_mu D q : Nat) : Nat :=
  (d_mu + 1) + 2 * D * q

def gm_equal_diff_det (d_mu d_pos D q : Nat) : Nat :=
  (D * q) * (D * q) + (D * q) * (d_mu + 1) + (d_pos + 1)

theorem gierer_meinhardt_equal_diffusion_stable (d_mu d_pos D q : Nat) :
    gm_equal_diff_neg_trace d_mu D q ≥ 1 ∧
    gm_equal_diff_det d_mu d_pos D q ≥ d_pos + 1 := by
  simp [gm_equal_diff_neg_trace, gm_equal_diff_det]
  omega


-- ============================================================================
-- Law 3 (`gierer-hydra-activator-inhibitor-3`):
-- Gierer & Meinhardt (1972) Differential-Diffusion Lateral-Inhibition Instability
-- ============================================================================
/-- For the Gierer-Meinhardt reaction Jacobian `J_0 = [[mu_a, -b], [c_j, -mu_h]]`
    with `J_11 = mu_a`, `J_22 = -mu_h`, and `det(J_0) = mu_a * mu_h`, the Fourier
    mode determinant at wavenumber squared `q = k^2` is:
      `det_J(q) = (D_a * q - mu_a) * (D_h * q + mu_h) + 2 * mu_a * mu_h`.
    Using the integer-scaled canonical parameters (`mu_a = 2, mu_h = 3, q = 1`):
    1. Zero wavenumber (`q = 0`) gives `det_J(0) = +6 > 0` (homogeneous stability).
    2. Equal diffusion (`D_a = 1, D_h = 1`, ratio `d = 1 < d_c ≈ 5.828`) gives `det_J(1) = +8 > 0` (stable).
    3. Subcritical diffusion (`D_a = 1, D_h = 5`, ratio `d = 5 < d_c ≈ 5.828`) gives `det_J(1) = +4 > 0` (stable).
    4. Supercritical long-range inhibitor diffusion (`D_a = 1, D_h = 10`, ratio `d = 10 > d_c`)
       drives `det_J(1) = -1 < 0`, triggering a symmetry-breaking Turing-Gierer-Meinhardt peak!
    5. At the canonical diffusivity ratio `d = 30` (`D_a = 1, D_h = 30`), `det_J(1) = -21 < 0`. -/
def gm_fourier_det (mu_a mu_h D_a D_h q : Int) : Int :=
  (D_a * q - mu_a) * (D_h * q + mu_h) + 2 * mu_a * mu_h

theorem gierer_meinhardt_lateral_inhibition_instability :
    gm_fourier_det 2 3 1 1 0 = 6 ∧
    gm_fourier_det 2 3 1 1 1 = 8 ∧
    gm_fourier_det 2 3 1 5 1 = 4 ∧
    gm_fourier_det 2 3 1 10 1 = -1 ∧
    gm_fourier_det 2 3 1 30 1 = -21 := by
  decide


-- ============================================================================
-- Pass 3 — Meinhardt (1993, 2012) coupled three-system model on the 3D shell
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Law 4 (`gierer-hydra-activator-inhibitor-4`):
-- Firing threshold of a thresholded activator-inhibitor subsystem (basal inhibitor production)
-- ----------------------------------------------------------------------------
/-- A Meinhardt subsystem with drive `w`, autocatalysis `w (c b^2 / c_i + rho_act)`, decay
    `mu b`, inhibitor `c' b^2 - mu_inh c_i + rho_inh` has a homogeneous "off" state iff the
    small-`b` quadratic `(w c mu_inh / rho_inh) b^2 - mu b + w rho_act = 0` has real roots,
    i.e. iff its discriminant `mu^2 rho_inh - 4 w^2 c mu_inh rho_act ≥ 0`
    (`w ≤ w_c = (mu/2) sqrt(rho_inh / (c mu_inh rho_act))`). In truncated natural-number
    arithmetic the discriminant is antitone in the drive: a larger drive (higher source density,
    weaker head-activator suppression) can only move a subsystem closer to firing, never away.
    This is what pins the tentacle ring to the band `phi_t(s) > w_c` in the Python shell model. -/
def meinhardt_off_state_disc (mu w c mu_inh rho_act rho_inh : Nat) : Nat :=
  mu * mu * rho_inh - 4 * w * w * c * mu_inh * rho_act

theorem meinhardt_three_system_off_state_discriminant_antitone
    (mu c mu_inh rho_act rho_inh w₁ w₂ : Nat) (h : w₁ ≤ w₂) :
    meinhardt_off_state_disc mu w₂ c mu_inh rho_act rho_inh ≤
      meinhardt_off_state_disc mu w₁ c mu_inh rho_act rho_inh := by
  unfold meinhardt_off_state_disc
  apply Nat.sub_le_sub_left
  apply Nat.mul_le_mul _ (Nat.le_refl rho_act)
  apply Nat.mul_le_mul _ (Nat.le_refl mu_inh)
  apply Nat.mul_le_mul _ (Nat.le_refl c)
  apply Nat.mul_le_mul _ h
  exact Nat.mul_le_mul (Nat.le_refl 4) h

/-- Integer instance with the canonical shell coefficients ×1000 (`c = 50, μ_inh = 30,
    ρ_act = 5, ρ_inh = 2`; `w_c` is invariant under a common rescaling of the five coefficients)
    and `μ` additionally ×10 (`μ = 200`) so that the drive is measured in tenths: the off state
    exists at `w = 0.1` (discriminant 50000 > 0) but not at `w = 0.2` (truncated to 0), bracketing
    `w_c = 0.1633` returned by the Python `three_system_firing_threshold`. -/
theorem meinhardt_three_system_threshold_instance :
    meinhardt_off_state_disc 200 1 50 30 5 2 = 50000 ∧
    meinhardt_off_state_disc 200 2 50 30 5 2 = 0 := by
  decide

-- ----------------------------------------------------------------------------
-- Law 5 (`gierer-hydra-activator-inhibitor-5`):
-- Tentacle drive: promoted by source density, suppressed by the head activator
-- ----------------------------------------------------------------------------
/-- Integer-scaled tentacle drive `phi_t = sigma * K / (K + a)` (the Python factor
    `H_t(sigma) / (1 + kappa_a a)` with `K = 1/kappa_a` scaled to a natural number):
    monotone in the source density `sigma` and antitone in the head activator `a`.
    Hence tentacle activation is maximal where the source density is highest but the head
    activator is low — the sub-hypostomal border (Meinhardt 1993). -/
def meinhardt_tentacle_drive (sigma K a : Nat) : Nat :=
  sigma * K / (K + a)

theorem meinhardt_tentacle_drive_monotone (K a σ₁ σ₂ a₁ a₂ σ : Nat)
    (hσ : σ₁ ≤ σ₂) (ha : a₁ ≤ a₂) :
    meinhardt_tentacle_drive σ₁ K a ≤ meinhardt_tentacle_drive σ₂ K a ∧
    meinhardt_tentacle_drive σ (K + 1) a₂ ≤ meinhardt_tentacle_drive σ (K + 1) a₁ := by
  unfold meinhardt_tentacle_drive
  constructor
  · exact Nat.div_le_div_right (Nat.mul_le_mul hσ (Nat.le_refl K))
  · apply Nat.div_le_div_left
    · omega
    · omega

-- ----------------------------------------------------------------------------
-- Law 6 (`gierer-hydra-activator-inhibitor-6`):
-- Head–foot exclusion through the source density
-- ----------------------------------------------------------------------------
/-- The head drive is the source density itself (`sigma`), the foot drive is
    `K / (K + sigma^2)` (Python: `1 / (1 + (sigma/sigma_f)^2)`). They are anti-monotone: raising
    the source density can never favour the head and the foot at the same time, so the two
    structures segregate to opposite ends of the body column (Meinhardt 2012: the foot forms at
    the position of lowest source density). -/
def meinhardt_foot_drive (K sigma : Nat) : Nat :=
  K / (K + sigma * sigma)

theorem meinhardt_head_foot_source_density_exclusion (K σ₁ σ₂ : Nat) (h : σ₁ ≤ σ₂) :
    σ₁ ≤ σ₂ ∧ meinhardt_foot_drive (K + 1) σ₂ ≤ meinhardt_foot_drive (K + 1) σ₁ := by
  unfold meinhardt_foot_drive
  constructor
  · exact h
  · apply Nat.div_le_div_left
    · exact Nat.add_le_add_left (Nat.mul_le_mul h h) (K + 1)
    · omega

-- ----------------------------------------------------------------------------
-- Law 7 (`gierer-hydra-activator-inhibitor-7`):
-- Finite-difference Laplace–Beltrami operator on the surface of revolution
-- ----------------------------------------------------------------------------
/-- Row stencil of the shell Laplacian used by `CylindricalShellGrid`:
    axial part `r_{i+1/2} (u_{i+1} - u_i) - r_{i-1/2} (u_i - u_{i-1})` (later divided by `r_i`)
    and circumferential part `u_{j+1} + u_{j-1} - 2 u_j` (later divided by `r_i^2`).
    Both annihilate constant fields for every radius profile, so the operator conserves the
    homogeneous Gierer–Meinhardt state exactly and the thresholds above are not shifted by the
    tapering geometry. -/
def shell_axial_stencil (r_plus r_minus u_up u u_down : Int) : Int :=
  r_plus * (u_up - u) - r_minus * (u - u_down)

def shell_circumferential_stencil (u_next u u_prev : Int) : Int :=
  u_next + u_prev - 2 * u

theorem cylindrical_shell_laplacian_annihilates_constants (r_plus r_minus u : Int) :
    shell_axial_stencil r_plus r_minus u u u = 0 ∧
    shell_circumferential_stencil u u u = 0 := by
  constructor
  · simp [shell_axial_stencil]
  · unfold shell_circumferential_stencil
    omega

end MathOfBiology.Loop_001
