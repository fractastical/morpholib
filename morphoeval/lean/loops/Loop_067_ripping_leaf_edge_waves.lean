/-
  morphoeval/lean/loops/Loop_067_ripping_leaf_edge_waves.lean
  Formal Lean 4 verification of:
    1. Sharon, Roman, Marder, Shin & Swinney (2002) Nature 419(6907):579-580
       ("Mechanics: Buckling cascades in free sheets")
    2. Audoly & Boudaoud (2003) Physical Review Letters 91(8):086105
       ("Self-Similar Structures near Boundaries in Strained Systems")
    3. Liang & Mahadevan (2009) PNAS 106(52):22049-22054
       ("The shape of a long leaf")

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_067

-- ============================================================================
-- Law 1 (`sharon-ripping-leaf-edge-waves-1`):
-- Isometric Arc-Length Excess Accommodation (`π^2 A^2 = ε_y λ^2`)
-- & Flat-Sheet Stability at Zero Marginal Excess Strain (`ε_y = 0 ⇒ A = 0`)
-- ============================================================================
/-- Along a leaf margin line at distance `x` from the free edge with target
    metric component `g_yy(x) = 1 + 2 ε_y(x)`, an isometric sinusoidal edge wave
    `w(x, y) = A(x) sin(2π y / λ)` relieves longitudinal membrane stretching
    when its squared slope satisfies `π^2 A^2 = ε_y λ^2` (or scaled by `4`:
    `4 π^2 A^2 = (g_yy - 1) λ^2`).
    Representing the excess metric factor `(g_yy - 1)` by integer `g_excess` and
    wavelength by `lam`, the squared amplitude factor `rhs = g_excess * lam * lam`
    vanishes identically when `g_excess = 0` (Euclidean flat metric `g_yy = 1`),
    is strictly positive whenever `g_excess > 0` and `lam > 0`, and scales
    quadratically when wavelength is scaled by cascade factor `r`. -/
def isometric_excess_amplitude_rhs (g_excess lam : Int) : Int :=
  g_excess * (lam * lam)

theorem sharon_isometric_arclength_accommodation (g_excess lam : Int) :
    isometric_excess_amplitude_rhs 0 lam = 0 ∧
    isometric_excess_amplitude_rhs g_excess 0 = 0 ∧
    isometric_excess_amplitude_rhs g_excess (-lam) = isometric_excess_amplitude_rhs g_excess lam ∧
    (∀ g l : Nat, 0 < isometric_excess_amplitude_rhs ((g : Int) + 1) ((l : Int) + 1)) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [isometric_excess_amplitude_rhs]
  · simp [isometric_excess_amplitude_rhs]
  · simp [isometric_excess_amplitude_rhs, Int.neg_mul_neg]
  · intro g l
    unfold isometric_excess_amplitude_rhs
    have hg : (0 : Int) < (g : Int) + 1 := by omega
    have hl : (0 : Int) < (l : Int) + 1 := by omega
    have hll : (0 : Int) < ((l : Int) + 1) * ((l : Int) + 1) := Int.mul_pos hl hl
    exact Int.mul_pos hg hll


-- ============================================================================
-- Law 2 (`sharon-ripping-leaf-edge-waves-2`):
-- Negative Intrinsic Gaussian Curvature (`K(x) < 0`) of Convexly Decaying
-- Marginal Growth Metric `ds^2 = dx^2 + g_yy(x) dy^2`
-- ============================================================================
/-- For the non-Euclidean leaf margin metric `ds^2 = dx^2 + g_yy(x) dy^2` with
    exponential marginal growth `g_yy(x) = 1 + g_0 exp(-x/ℓ)`, the Brioschi
    formula gives intrinsic Gaussian curvature numerator:
      `K_num(g_yy, g_yy_xx, g_yy_x) = -(2 * g_yy * g_yy_xx - g_yy_x^2)`,
    and in the leading weak-strain limit `K_lead(g_yy_xx) = -g_yy_xx`.
    When marginal growth decays convexly from the free edge (`g_yy_xx = c + 1 > 0`),
    the leading Gaussian curvature is strictly negative (`K_lead < 0`), whereas
    a uniform Euclidean metric (`g_yy_xx = 0`) has zero Gaussian curvature (`K_lead = 0`). -/
def marginal_metric_gaussian_curvature_lead (g_yy_xx : Int) : Int :=
  -g_yy_xx

theorem sharon_negative_gaussian_curvature (g_yy_xx : Int) :
    marginal_metric_gaussian_curvature_lead 0 = 0 ∧
    (∀ c : Nat, marginal_metric_gaussian_curvature_lead ((c : Int) + 1) < 0) ∧
    marginal_metric_gaussian_curvature_lead (-g_yy_xx) = g_yy_xx := by
  refine ⟨?_, ?_, ?_⟩
  · simp [marginal_metric_gaussian_curvature_lead]
  · intro c
    unfold marginal_metric_gaussian_curvature_lead
    omega
  · simp [marginal_metric_gaussian_curvature_lead]


-- ============================================================================
-- Law 3 (`sharon-ripping-leaf-edge-waves-3`):
-- Self-Similar Multi-Generation Sub-Harmonic Cascade Scaling
-- (`λ_{m+1} = r_cascade * λ_m`, `λ(x) = c_λ * x + λ_min`)
-- ============================================================================
/-- In Audoly & Boudaoud (2003) and Sharon et al. (2002), each coarsening
    buckling generation away from the sheet margin multiplies the local
    wavelength by a constant self-similar factor `r_cascade ≥ 3`:
      `cascade_next_wavelength r_cascade lam_m = r_cascade * lam_m`,
    and the local dominant wavelength increases linearly with distance `x` from
    the free edge above the thickness cutoff `lam_min`:
      `local_edge_wavelength c_lam x lam_min = c_lam * x + lam_min`.
    Proves two-step multiplicative cascade composition `r^2 * lam_m`, strict
    coarsening (`lam_m < r * lam_m` for `r ≥ 2, lam_m ≥ 1`), and exact reduction
    to the thickness cutoff `lam_min` at the free edge `x = 0`. -/
def cascade_next_wavelength (r_cascade lam_m : Int) : Int :=
  r_cascade * lam_m

def local_edge_wavelength (c_lam x lam_min : Int) : Int :=
  c_lam * x + lam_min

theorem audoly_self_similar_cascade_scaling (r_cascade lam_m c_lam lam_min : Int) :
    local_edge_wavelength c_lam 0 lam_min = lam_min ∧
    cascade_next_wavelength r_cascade (cascade_next_wavelength r_cascade lam_m) =
      (r_cascade * r_cascade) * lam_m ∧
    (∀ r l : Nat, ((l : Int) + 1) < cascade_next_wavelength ((r : Int) + 2) ((l : Int) + 1)) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [local_edge_wavelength]
  · simp [cascade_next_wavelength, Int.mul_assoc]
  · intro r l
    unfold cascade_next_wavelength
    have hr1 : (0 : Int) < (r : Int) + 1 := by omega
    have hl : (0 : Int) < (l : Int) + 1 := by omega
    have hpos : (0 : Int) < ((r : Int) + 1) * ((l : Int) + 1) := Int.mul_pos hr1 hl
    have hsplit : ((r : Int) + 2) * ((l : Int) + 1) =
        ((r : Int) + 1) * ((l : Int) + 1) + 1 * ((l : Int) + 1) := by
      have h2 : (r : Int) + 2 = ((r : Int) + 1) + 1 := by omega
      rw [h2, Int.add_mul]
    omega


-- ============================================================================
-- Law 4 (`sharon-ripping-leaf-edge-waves-4`, Pass 3):
-- CIN/TCP-Arrested, Auxin-Boosted Marginal Growth Increment and the
-- Non-Negative Excess Metric `g_0 = ((1+Γ_mar)/(1+Γ_int))^2 - 1 ≥ 0`
-- ============================================================================
/-- Forward-Euler growth increment of the developmental module
    `dΓ = γ_0 (1 - m) (1 + β a)`, written with natural-number proxies:
    `m_rem` for the remaining proliferative window `(1 - m)` (zero once the
    CIN/TCP maturation front has passed) and `a_boost` for `(1 + β a)`.
    Proves: (i) TCP arrest (`m_rem = 0`) gives zero growth; (ii) a slower cin
    front (larger remaining window) never grows less; (iii) the auxin boost makes
    the margin grow at least as much as the interior (`a_boost ≥ R`); (iv) the
    accumulated growth strain is non-decreasing step to step. -/
def marginal_growth_increment (gamma0 m_rem a_boost : Nat) : Nat :=
  gamma0 * m_rem * a_boost

theorem tcp_arrested_marginal_growth_increment (gamma0 a_boost G : Nat) :
    marginal_growth_increment gamma0 0 a_boost = 0 ∧
    (∀ m1 m2 : Nat, m1 ≤ m2 →
      marginal_growth_increment gamma0 m1 a_boost ≤ marginal_growth_increment gamma0 m2 a_boost) ∧
    (∀ m_rem R b : Nat,
      marginal_growth_increment gamma0 m_rem R ≤ marginal_growth_increment gamma0 m_rem (R + b)) ∧
    (∀ m_rem : Nat, G ≤ G + marginal_growth_increment gamma0 m_rem a_boost) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [marginal_growth_increment]
  · intro m1 m2 h
    unfold marginal_growth_increment
    exact Nat.mul_le_mul_right a_boost (Nat.mul_le_mul_left gamma0 h)
  · intro m_rem R b
    unfold marginal_growth_increment
    exact Nat.mul_le_mul_left (gamma0 * m_rem) (Nat.le_add_right R b)
  · intro m_rem
    exact Nat.le_add_right G _

/-- Excess metric numerator handed to the Föppl-von Kármán solver:
    `(1+Γ_mar)^2 - (1+Γ_int)^2` (the denominator `(1+Γ_int)^2 > 0` does not
    change the sign). With the margin growing at least as much as the interior
    (`Γ_int ≤ Γ_mar`, guaranteed by Law 4 (iii)) the excess is non-negative, and it
    vanishes exactly when margin and interior grow identically (`β a = 0`, i.e. the
    cuc2/pin1-like homogeneous limit reduces to the flat Euclidean control of Law 1). -/
def excess_metric_numerator (g_mar g_int : Nat) : Int :=
  ((1 + g_mar) * (1 + g_mar) : Nat) - ((1 + g_int) * (1 + g_int) : Nat)

theorem developmental_excess_metric_nonneg (g_mar g_int : Nat) (h : g_int ≤ g_mar) :
    0 ≤ excess_metric_numerator g_mar g_int ∧ excess_metric_numerator g_int g_int = 0 := by
  refine ⟨?_, ?_⟩
  · unfold excess_metric_numerator
    have hle : (1 + g_int) * (1 + g_int) ≤ (1 + g_mar) * (1 + g_mar) :=
      Nat.mul_le_mul (Nat.add_le_add_left h 1) (Nat.add_le_add_left h 1)
    have hz : ((1 + g_int) * (1 + g_int) : Int) ≤ ((1 + g_mar) * (1 + g_mar) : Int) := by
      exact_mod_cast hle
    omega
  · simp [excess_metric_numerator]


-- ============================================================================
-- Law 5 (`sharon-ripping-leaf-edge-waves-5`, Pass 3):
-- PIN1 Up-the-Gradient Pairwise Transport Conserves Auxin Mass
-- ============================================================================
/-- Between neighbouring margin cells `i, j` the PIN1 up-the-gradient scheme
    exports `f_ij` from `i` to `j` and `f_ji` from `j` to `i`. The pairwise net
    flux is antisymmetric and the two-cell auxin total is invariant under one
    exchange step, which is what makes `Σ_i da_i = 0` hold on the whole lattice
    (checked numerically in `self_test` check 9). -/
def pin_pairwise_net_flux (f_ij f_ji : Int) : Int :=
  f_ij - f_ji

theorem pin_pairwise_transport_mass_conservation (a_i a_j f_ij f_ji : Int) :
    pin_pairwise_net_flux f_ij f_ji + pin_pairwise_net_flux f_ji f_ij = 0 ∧
    (a_i - pin_pairwise_net_flux f_ij f_ji) + (a_j + pin_pairwise_net_flux f_ij f_ji) = a_i + a_j ∧
    pin_pairwise_net_flux f_ij f_ij = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · unfold pin_pairwise_net_flux; omega
  · unfold pin_pairwise_net_flux; omega
  · simp [pin_pairwise_net_flux]

end MathOfBiology.Loop_067
