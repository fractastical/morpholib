/-
  morphoeval/lean/loops/Loop_003_mammalian_coat_markings.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 003:
  Murray (1981) J. Theor. Biol. 88(1):161-199 — "A pre-pattern formation
  mechanism for animal coat markings" (`murray-mammalian-coat-markings`).

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_003_mammalian_coat_markings.lean
-/

namespace MathOfBiology.Loop_003

-- ============================================================================
-- Law 1 (`murray-mammalian-coat-markings-1`):
-- Thomas (1975) / Murray (1981) Homogeneous Steady-State Reservoir Flux Balance
-- ============================================================================

/-- Nondimensionalized Thomas (1975) / Murray (1981) Eq. (4) reaction rates:
    `f(u, v) = (a - u) - h(u, v)` and `g(u, v) = alpha*(b - v) - h(u, v)`,
    where `h(u, v) = rho * u * v / (1 + u + K * u^2)`. -/
def thomas_react_f (a u h : Int) : Int :=
  (a - u) - h

def thomas_react_g (alpha_b_minus_v h : Int) : Int :=
  alpha_b_minus_v - h

/-- Theorem 1 (`thomas_steady_state_flux_balance`):
    At any homogeneous equilibrium `(u_0, v_0)` where `f(u_0, v_0) = 0` and
    `g(u, v) = 0`, the net substrate influx `a - u_0` equals both the enzymatic
    clearance `h` and the co-substrate influx `alpha * (b - v_0)`. -/
theorem thomas_steady_state_flux_balance (a u alpha_b_minus_v h : Int)
    (hf : thomas_react_f a u h = 0)
    (hg : thomas_react_g alpha_b_minus_v h = 0) :
    a - u = alpha_b_minus_v ∧ a - u = h := by
  unfold thomas_react_f at hf
  unfold thomas_react_g at hg
  omega


-- ============================================================================
-- Law 2 (`murray-mammalian-coat-markings-2`):
-- Thomas Substrate-Inhibition Autocatalysis Threshold (`K * u_0^2 > 1`)
-- ============================================================================

/-- Scaled numerator of `∂f/∂u = -1 - ∂h/∂u` at steady state `(u_0, v_0)`:
    `denom^2 * f_u = -denom^2 - rho * v_0 * (1 - K * u_0^2)`
                   `= rho * v_0 * (K_u0_sq - 1) - denom^2`. -/
def thomas_fu_scaled_num (denom_sq rho_v0 K_u0_sq : Int) : Int :=
  rho_v0 * (K_u0_sq - 1) - denom_sq

/-- Theorem 2 (`thomas_substrate_inhibition_autocatalysis`):
    (1) Without substrate inhibition (`K * u_0^2 = 0`), `denom^2 * f_u` is strictly
        negative (`≤ -2`) for all `denom_sq ≥ 1` and `rho_v0 ≥ 1`, so self-activation
        (`f_u > 0`) and Turing instability cannot occur.
    (2) With substrate inhibition active at Murray's (1981) Table 1 canonical regime
        (`u_0 ≈ 10`, `K * u_0^2 = 10`, `denom^2 = 21^2 = 441`, `rho * v_0 = 172`),
        `thomas_fu_scaled_num 441 172 10 = 1107 > 0`, yielding positive activator feedback. -/
theorem thomas_substrate_inhibition_autocatalysis (denom_sq rho_v0 : Int)
    (hd : denom_sq ≥ 1) (hrv : rho_v0 ≥ 1) :
    thomas_fu_scaled_num denom_sq rho_v0 0 ≤ -2 ∧
    thomas_fu_scaled_num 441 172 10 = 1107 := by
  unfold thomas_fu_scaled_num
  omega


-- ============================================================================
-- Law 3 (`murray-mammalian-coat-markings-3`):
-- Murray (1981) Conical Tail Circumferential-to-Axial Eigenmode Transition
-- ("A spotted animal can have a striped tail, NOT vice versa")
-- ============================================================================

/-- On a cylinder of radius `r`, an eigenmode with 10x-scaled axial squared wavenumber
    `q_ax_10` and circumferential mode `m` has `10 * r^2`-scaled effective wavenumber
    `q_eff = q_ax_10 * r^2 + 10 * m^2`. It lies within Murray's unstable Turing band
    `[q_min_10, q_max_10]` (e.g., `[5, 9]` representing `[0.524, 0.965]`) iff
    `q_min_10 * r^2 ≤ q_ax_10 * r^2 + 10 * m^2 ≤ q_max_10 * r^2`. -/
def in_turing_band_r2 (q_min_10 q_max_10 q_ax_10 m r : Nat) : Bool :=
  decide (q_min_10 * r * r ≤ q_ax_10 * r * r + 10 * m * m ∧
          q_ax_10 * r * r + 10 * m * m ≤ q_max_10 * r * r)

/-- Theorem 3 (`murray_conical_tail_spot_to_ring_transition`):
    For Murray's (1981) 10x-scaled Turing band `[q_min_10, q_max_10] = [5, 9]` and
    critical axial mode `q_ax_10 = 7`:
    - At the broad proximal tail base (`r = 3`), both the `m = 0` ring mode (`63 ∈ [45, 81]`)
      and the `m = 1` circumferential spot mode (`73 ∈ [45, 81]`) lie inside the Turing band (`true`).
    - At the narrow distal tail tip (`r = 1`), the `m = 0` axial ring mode remains
      unstable (`7 ∈ [5, 9]`, `true`), whereas the `m = 1` (`17 > 9`) and `m = 2` (`47 > 9`)
      circumferential spot modes exceed the upper Turing cutoff and are excluded (`false`). -/
theorem murray_conical_tail_spot_to_ring_transition :
    in_turing_band_r2 5 9 7 0 3 = true ∧
    in_turing_band_r2 5 9 7 1 3 = true ∧
    in_turing_band_r2 5 9 7 0 1 = true ∧
    in_turing_band_r2 5 9 7 1 1 = false ∧
    in_turing_band_r2 5 9 7 2 1 = false := by
  decide


-- ============================================================================
-- Law 4 (`murray-mammalian-coat-markings-4`, Pass 3):
-- P1 Surface-FEM Cotangent Laplace-Beltrami: Constants in the Kernel & Mass Conservation
-- ============================================================================

/-- One row of the P1 cotangent stiffness matrix restricted to a single triangle
    `(i, j, k)` (Pinkall & Polthier 1993; Dziuk 1988):
    `(K u)_i = w_ij * (u_i - u_j) + w_ik * (u_i - u_k)`,
    where `w_ij = cot(angle opposite edge ij) / 2` etc.  Any scaling of the weights
    (e.g. by `2 * area`) is absorbed into the integer `w`s. -/
def fem_cot_row (w_ij w_ik u_i u_j u_k : Int) : Int :=
  w_ij * (u_i - u_j) + w_ik * (u_i - u_k)

/-- Theorem 4 (`fem_cotangent_laplacian_conserves_mass`):
    (1) a constant field `c` is annihilated by every stiffness row (`K 1 = 0`, so the
        homogeneous steady state `(u_0, v_0)` is an exact discrete equilibrium and the
        natural boundary condition is zero flux), and
    (2) the three vertex rows of a triangle with edge weights `w_ab, w_bc, w_ca` sum to zero
        for ANY field `(u_a, u_b, u_c)` -- the discrete divergence theorem: the lumped-mass
        IMEX diffusion step `(M + dt K) u' = M u` preserves `sum_i M_i u_i`.
    Numerically twinned by `verify_fem_cylinder_spectrum` (|K 1| < 1e-15) and self-test 10. -/
theorem fem_cotangent_laplacian_conserves_mass (w_ab w_bc w_ca u_a u_b u_c c : Int) :
    fem_cot_row w_ab w_ca c c c = 0 ∧
    fem_cot_row w_ab w_ca u_a u_b u_c
      + fem_cot_row w_ab w_bc u_b u_a u_c
      + fem_cot_row w_ca w_bc u_c u_a u_b = 0 := by
  unfold fem_cot_row
  simp only [Int.mul_sub]
  omega


-- ============================================================================
-- Law 5 (`murray-mammalian-coat-markings-5`, Pass 3):
-- Full Circumferential Exclusion Radius r_excl = 1 / k_+ (n = 0 longitudinal-stripe mode)
-- ============================================================================

/-- Theorem 5 (`murray_tail_full_exclusion_requires_circumference_below_lambda_plus`):
    Murray's `r_crit` criterion tests only the fastest-growing axial wavenumber `q_ax = k_c`.
    Because Neumann ends admit the axially uniform mode `n = 0` (`q_ax = 0`), an `m ≥ 1`
    "longitudinal stripe" `0 + 10 m^2` is still inside the 10x-scaled band `[5, 9] * r^2`
    whenever `5 r^2 ≤ 10 m^2 ≤ 9 r^2`, i.e. for `r^2 ∈ [10 m^2 / 9, 2 m^2]`:
    - `m = 1, r = 1`: `10 > 9` -> excluded (`false`): circumference below lambda_+ = 2π/k_+;
    - the `m = 0`, `q_ax = 7` ring at `r = 1` is unstable (`true`), so rings survive there;
    - `m = 3, r = 4`: `80 ≤ 90 ≤ 144` -> the `n = 0` longitudinal mode IS unstable (`true`),
      whereas `m = 3, r = 3`: `90 > 81` -> excluded (`false`): the cutoff is `r^2 q_max < 10 m^2`;
    - `m = 1, r = 2`: `n = 0` is BELOW the band (`20 > 10`, `false`) but with axial `q_ax = 4`
      the same `m = 1` mode is unstable (`20 ≤ 26 ≤ 36`, `true`) -- between `r_excl = 1/k_+`
      and `r_crit` the tail still carries `m ≥ 1` modes of long axial wavelength.
    Hence every `m ≥ 1` mode is excluded only when `r < m / k_+` (for `m = 1`: `r < 1/k_+`);
    the v2 mode count (which started at `n = 1`) silently missed exactly these modes. -/
theorem murray_tail_full_exclusion_requires_circumference_below_lambda_plus :
    in_turing_band_r2 5 9 0 1 1 = false ∧
    in_turing_band_r2 5 9 7 0 1 = true ∧
    in_turing_band_r2 5 9 0 3 4 = true ∧
    in_turing_band_r2 5 9 0 3 3 = false ∧
    in_turing_band_r2 5 9 0 1 2 = false ∧
    in_turing_band_r2 5 9 4 1 2 = true := by
  decide

end MathOfBiology.Loop_003
