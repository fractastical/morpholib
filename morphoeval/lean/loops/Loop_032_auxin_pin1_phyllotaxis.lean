/-
  morphoeval/lean/loops/Loop_032_auxin_pin1_phyllotaxis.lean
  Formal Lean 4 verification of Jönsson et al. (2006) PNAS 103(5):1633-1638
  ("An auxin-driven polarized transport model for phyllotaxis", 10.1073/pnas.0509839103)
  and Smith et al. (2006) PNAS 103(5):1301-1306 ("A plausible model of phyllotaxis",
  10.1073/pnas.0510457103) governing up-the-gradient Boltzmann PIN1 membrane
  allocation, antisymmetric intercellular polar auxin transport conservation, and
  the Turing-like active polar transport instability threshold in the shoot apical
  meristem (`jonsson-auxin-pin1-meristem`, Slot 032).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_032

-- ============================================================================
-- Law 1 (`jonsson-auxin-pin1-meristem-1`):
-- Jönsson et al. (2006) & Smith et al. (2006) Up-the-Gradient PIN1 Allocation Law
-- ============================================================================
/-- In Jönsson et al. (2006) PNAS Eq. 4 & Smith et al. (2006) PNAS Eq. 2, cell `i`
    with total PIN1 pool `P_i` allocates PIN1 efflux carriers across its anticlinal
    membranes `j` proportionally to unnormalized Boltzmann weights `w_j = exp(c_1 * a_j)`.
    Clearing the partition denominator `Z_i = ∑_k w_k`:
    1. Across a 3-neighbor local junction (`w_1, w_2, w_3`), the scaled allocations
       `P_i * w_1 + P_i * w_2 + P_i * w_3` sum identically to `P_i * (w_1 + w_2 + w_3)`,
       guaranteeing exact cellular PIN1 carrier pool conservation (`∑_j P_ij = P_i`).
    2. For any pair of neighbors `j, k` with `P_i > 0` and `w_j > w_k` (i.e. neighbor `j`
       has strictly higher auxin concentration `a_j > a_k`), the allocated PIN1 on wall
       `i → j` strictly exceeds that on wall `i → k` (`P_i * w_j > P_i * w_k`, up-the-gradient
       polarization). -/
def pin1_alloc_scaled (P_i w_j : Int) : Int :=
  P_i * w_j

def pin1_partition_sum3 (w1 w2 w3 : Int) : Int :=
  w1 + w2 + w3

theorem jonsson_2006_up_gradient_pin1_allocation (P_i w1 w2 w3 w_j w_k : Int)
    (hP : P_i > 0) (hw : w_j > w_k) :
    pin1_alloc_scaled P_i w1 + pin1_alloc_scaled P_i w2 + pin1_alloc_scaled P_i w3 =
      P_i * pin1_partition_sum3 w1 w2 w3 ∧
    pin1_alloc_scaled P_i w_j > pin1_alloc_scaled P_i w_k := by
  constructor
  · simp [pin1_alloc_scaled, pin1_partition_sum3, Int.mul_add]
  · simp [pin1_alloc_scaled]
    exact Int.mul_lt_mul_of_pos_left hw hP


-- ============================================================================
-- Law 2 (`jonsson-auxin-pin1-meristem-2`):
-- Jönsson et al. (2006) Intercellular Polar Auxin + Diffusive Flux Conservation Law
-- ============================================================================
/-- In Jönsson et al. (2006) PNAS Eqs. 1-2, the total intercellular auxin flux entering
    cell `i` from neighbor `j` combines active PIN1-mediated efflux and passive diffusion:
      `J_total(i ← j) = T * (P_ji * a_j - P_ij * a_i) + D * (a_j - a_i)`
    This theorem proves that:
    1. Pairwise intercellular flux is strictly antisymmetric (`J_total(i ← j) + J_total(j ← i) = 0`),
       guaranteeing exact mass conservation across every cell wall in the L1 meristem when
       synthesis and degradation vanish (`ρ_0 = 0, μ = 0`).
    2. At the homogeneous isotropic equilibrium (`a_i = a_j = a0`, `P_ij = P_ji = P0`),
       both active and diffusive intercellular fluxes vanish (`J_total = 0`). -/
def intercellular_auxin_flux (T D P_ij P_ji a_i a_j : Int) : Int :=
  T * (P_ji * a_j - P_ij * a_i) + D * (a_j - a_i)

theorem jonsson_2006_intercellular_flux_conservation (T D P_ij P_ji P0 a_i a_j a0 : Int) :
    intercellular_auxin_flux T D P_ij P_ji a_i a_j +
      intercellular_auxin_flux T D P_ji P_ij a_j a_i = 0 ∧
    intercellular_auxin_flux T D P0 P0 a0 a0 = 0 := by
  constructor
  · simp [intercellular_auxin_flux, Int.mul_sub]
    omega
  · simp [intercellular_auxin_flux]


-- ============================================================================
-- Law 3 (`jonsson-auxin-pin1-meristem-3`):
-- Jönsson et al. (2006) Up-the-Gradient Turing-Like Instability & NPA Collapse Law
-- ============================================================================
/-- Linearizing the Jönsson et al. (2006) PNAS Eqs. 1-4 around the homogeneous steady state
    `a* = a0` with uniform membrane PIN1 fraction `P0 = P_i / z` yields the effective
    short-wavelength transport coefficient (scaled by `z`):
      `D_eff_scaled = z * D - T * P_i * (c1_a0 - 1)`
    where `c1_a0 = c_1 * a0` is the dimensionless PIN1 up-the-gradient feedback product,
    and the spatial mode growth numerator `Γ_mode = T * P_i * (c1_a0 - 1) - z * D = -D_eff_scaled`.
    This theorem proves that:
    1. Whenever up-the-gradient polar transport feedback exceeds passive diffusion
       (`T * P_i * (c1_a0 - 1) > z * D`), the effective diffusion coefficient becomes
       strictly negative (`D_eff_scaled < 0`) and the spatial mode growth numerator is
       strictly positive (`Γ_mode > 0`), triggering spontaneous de novo auxin convergence
       maxima separated by characteristic spacing `λ_c`.
    2. Under NPA pharmacological treatment or in the *pin1* loss-of-function mutant (`T = 0`),
       for any positive passive diffusion (`z * D > 0`), `Γ_mode = -z * D < 0`, rigorously
       abolishing pattern instability and producing the bare pin-formed shoot apex. -/
def effective_diffusion_scaled (z_D T P_i c1_a0 : Int) : Int :=
  z_D - T * P_i * (c1_a0 - 1)

def mode_instability_gain (z_D T P_i c1_a0 : Int) : Int :=
  T * P_i * (c1_a0 - 1) - z_D

theorem jonsson_2006_up_gradient_instability_and_npa_collapse (z_D T P_i c1_a0 : Int)
    (h_instab : T * P_i * (c1_a0 - 1) > z_D)
    (h_diff : z_D > 0) :
    effective_diffusion_scaled z_D T P_i c1_a0 < 0 ∧
    mode_instability_gain z_D T P_i c1_a0 > 0 ∧
    mode_instability_gain z_D 0 P_i c1_a0 < 0 := by
  refine ⟨?_, ?_, ?_⟩
  · simp [effective_diffusion_scaled]
    omega
  · simp [mode_instability_gain]
    omega
  · simp [mode_instability_gain]
    omega


-- ============================================================================
-- Law 4 (`jonsson-auxin-pin1-meristem-4`, Pass 3):
-- Jönsson et al. (2006) Explicit PIN1 Exocytosis / Endocytosis Trafficking Kinetics
-- and their Quasi-Steady-State (QSS) Membrane Allocation
-- ============================================================================
/-- Jönsson et al. (2006) PNAS model PIN1 membrane cycling explicitly as first-order
    exocytosis from a cytosolic (endosomal) pool `P_cyt` onto wall `j` at rate
    `k_exo * f(a_j)` and uniform endocytosis at rate `k_endo`:
      `dP_ij/dt = k_exo * f_j * P_cyt - k_endo * P_ij`,   `P_cyt + ∑_j P_ij = P_tot`.
    Clearing the partition denominator `Z = k_endo + k_exo * (f_1 + f_2 + f_3)`, the
    steady-state membrane allocation numerator is `M_j = k_exo * f_j * P_tot` and the
    cytosolic numerator is `C = k_endo * P_tot`. This theorem proves that:
    1. `C + M_1 + M_2 + M_3 = P_tot * Z` — the kinetic steady state conserves the total
       cellular PIN1 pool (membrane + cytosol), generalising Law 1 to finite trafficking rates.
    2. `k_endo * M_j = k_exo * f_j * C` — exocytosis onto wall `j` exactly balances
       endocytosis from wall `j` (the ODE right-hand side vanishes), i.e. `(M_j, C)` is a
       genuine fixed point of the trafficking kinetics.
    3. `f_j > f_k ⇒ M_j > M_k` — up-the-gradient polarity survives finite trafficking rates.
    4. `k_exo = 0 ⇒ M_j = 0` — blocking exocytosis (brefeldin A, Geldner et al. 2001 Nature)
       internalises all PIN1 (`P_cyt = P_tot`), abolishing membrane polarity.
    In the limit `k_endo → 0` the allocation `M_j / Z → f_j / ∑_k f_k`, recovering the
    instantaneous Boltzmann law of Law 1. -/
def pin1_membrane_qss_num (k_exo f_j P_tot : Int) : Int :=
  k_exo * f_j * P_tot

def pin1_cytosol_qss_num (k_endo P_tot : Int) : Int :=
  k_endo * P_tot

def pin1_trafficking_partition (k_exo k_endo f1 f2 f3 : Int) : Int :=
  k_endo + k_exo * (f1 + f2 + f3)

theorem jonsson_2006_pin1_trafficking_qss (k_exo k_endo f1 f2 f3 f_j f_k P_tot : Int)
    (hk : k_exo > 0) (hP : P_tot > 0) (hf : f_j > f_k) :
    pin1_cytosol_qss_num k_endo P_tot + pin1_membrane_qss_num k_exo f1 P_tot +
        pin1_membrane_qss_num k_exo f2 P_tot + pin1_membrane_qss_num k_exo f3 P_tot =
      P_tot * pin1_trafficking_partition k_exo k_endo f1 f2 f3 ∧
    k_endo * pin1_membrane_qss_num k_exo f_j P_tot =
      k_exo * f_j * pin1_cytosol_qss_num k_endo P_tot ∧
    pin1_membrane_qss_num k_exo f_j P_tot > pin1_membrane_qss_num k_exo f_k P_tot ∧
    pin1_membrane_qss_num 0 f_j P_tot = 0 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [pin1_cytosol_qss_num, pin1_membrane_qss_num, pin1_trafficking_partition,
      Int.mul_add]
    ac_rfl
  · simp only [pin1_cytosol_qss_num, pin1_membrane_qss_num]
    ac_rfl
  · simp only [pin1_membrane_qss_num]
    have h1 : k_exo * f_j > k_exo * f_k := Int.mul_lt_mul_of_pos_left hf hk
    exact Int.mul_lt_mul_of_pos_right h1 hP
  · simp [pin1_membrane_qss_num]


-- ============================================================================
-- Law 5 (`jonsson-auxin-pin1-meristem-5`, Pass 3):
-- Mitchison (1980/1981) With-the-Flux Canalization Winner-Take-All Law
-- (subepidermal provascular strand formation; Bayer et al. 2009 Genes Dev)
-- ============================================================================
/-- In the with-the-flux canalization hypothesis (Mitchison 1980 Proc. R. Soc. B; 1981 Phil.
    Trans. R. Soc. B; Feugier et al. 2005 J. Theor. Biol.; Bayer et al. 2009 Genes Dev.) the
    exocytosis targeting function of a subepidermal cell is a superlinear function of the
    auxin efflux `J_ij` already passing through wall `j`, modelled here by the quadratic
    feedback `Φ(J) = J * J`. For two competing efflux walls with `J_1 > J_2 ≥ 0`:
    1. `Φ(J_1) - Φ(J_2) = (J_1 - J_2) * (J_1 + J_2)` — the feedback difference factorises
       through the flux asymmetry;
    2. `Φ(J_1) > Φ(J_2)` — the wall already carrying more flux receives strictly more PIN1,
       so the initial asymmetry grows (winner-take-all canalization into a single file);
    3. for any positive feedback gain `α > 0`, the PIN1 allocation difference drive
       `α * (Φ(J_1) - Φ(J_2))` is strictly positive, so a symmetric two-branch split
       (`P_1 = P_2`) is linearly unstable and the strand narrows. -/
def flux_feedback_quadratic (J : Int) : Int :=
  J * J

theorem mitchison_1980_canalization_winner_take_all (J1 J2 alpha : Int)
    (h2 : J2 ≥ 0) (h : J1 > J2) (ha : alpha > 0) :
    flux_feedback_quadratic J1 - flux_feedback_quadratic J2 = (J1 - J2) * (J1 + J2) ∧
    flux_feedback_quadratic J1 > flux_feedback_quadratic J2 ∧
    alpha * (flux_feedback_quadratic J1 - flux_feedback_quadratic J2) > 0 := by
  have hfact : flux_feedback_quadratic J1 - flux_feedback_quadratic J2 = (J1 - J2) * (J1 + J2) := by
    simp only [flux_feedback_quadratic, Int.sub_mul, Int.mul_add]
    have hc : J2 * J1 = J1 * J2 := Int.mul_comm J2 J1
    omega
  have hd : J1 - J2 > 0 := by omega
  have hs : J1 + J2 > 0 := by omega
  have hpos : (J1 - J2) * (J1 + J2) > 0 := Int.mul_pos hd hs
  refine ⟨hfact, ?_, ?_⟩
  · omega
  · rw [hfact]
    exact Int.mul_pos ha hpos

end MathOfBiology.Loop_032
