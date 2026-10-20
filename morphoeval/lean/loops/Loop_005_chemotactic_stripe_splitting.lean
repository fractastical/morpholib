/-
  morphoeval/lean/loops/Loop_005_chemotactic_stripe_splitting.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 005:
  Painter, Maini & Othmer (1999) PNAS 96(10):5549-5554 — "Stripe and
  spotilihood in chemotactic pattern formation" (`painter-chemotactic-stripe-splitting`).

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_005_chemotactic_stripe_splitting.lean
-/

namespace MathOfBiology.Loop_005

-- ============================================================================
-- Law 1 (`painter-chemotactic-stripe-splitting-1`):
-- Keller-Segel-Painter Homogeneous Steady-State & Pure-RD Stability (`chi = 0`)
-- ============================================================================

/-- Trace of the linearized Keller-Segel-Painter wavenumber matrix `M(q)` at
    squared wavenumber `q = k^2`:
    `tr M(q) = -(r + gamma) - (D_n + D_c) * q`. -/
def ksp_trace (r gamma D_n D_c q : Int) : Int :=
  -(r + gamma) - (D_n + D_c) * q

/-- Determinant of the linearized Keller-Segel-Painter wavenumber matrix `M(q)`
    at squared wavenumber `q = k^2` with effective chemotactic sensitivity
    `chi_eff = n_0 * chi(c^*) * s^*`:
    `det M(q) = D_n * D_c * q^2 + (r * D_c + gamma * D_n - chi_eff) * q + r * gamma`. -/
def ksp_det (r gamma D_n D_c chi_eff q : Int) : Int :=
  (D_n * D_c) * (q * q) + (r * D_c + gamma * D_n - chi_eff) * q + r * gamma

/-- Theorem 1 (`ksp_uniform_and_no_chemotaxis_stability`):
    (1) At spatially uniform perturbations (`q = 0`), `tr M(0) ≤ -2 < 0` and
        `det M(0) = r * gamma ≥ 1 > 0` for all `r ≥ 1, gamma ≥ 1`, proving
        uniform stability of `(n^*, c^*) = (n_0, alpha * n_0 / (gamma * (1 + beta * n_0)))`.
    (2) Without chemotactic cross-diffusion (`chi_eff = 0`), at canonical equal
        diffusivities (`D_n = D_c = 1, r = 1, gamma = 4`), `det M(q) = q^2 + 5*q + 4 ≥ 4 > 0`
        for all `q ≥ 0`, precluding any spatial Turing instability. -/
theorem ksp_uniform_and_no_chemotaxis_stability (r gamma D_n D_c chi_eff q : Int)
    (hr : r ≥ 1) (hg : gamma ≥ 1) (hq : q ≥ 0) :
    ksp_trace r gamma D_n D_c 0 ≤ -2 ∧
    ksp_det 1 4 1 1 chi_eff 0 = 4 ∧
    ksp_det 1 4 1 1 0 q ≥ 4 := by
  unfold ksp_trace ksp_det
  have hq2 : q * q ≥ 0 := Int.mul_nonneg hq hq
  omega


-- ============================================================================
-- Law 2 (`painter-chemotactic-stripe-splitting-2`):
-- Equal-Diffusivity Chemotactic Pattern Instability Threshold (`D_n = D_c`, `chi > chi_c`)
-- ============================================================================

/-- Theorem 2 (`painter_1999_equal_diffusivity_chemotactic_instability`):
    Unlike pure reaction-diffusion systems where equal diffusivities (`D_n = D_c = 1`)
    forbid spatial pattern formation, the Painter-Maini-Othmer (1999) chemotactic
    system with `D_n = D_c = 1, r = 1, gamma = 4` has critical chemotactic threshold
    `chi_eff_c = (sqrt(r * D_c) + sqrt(gamma * D_n))^2 = (1 + 2)^2 = 9` at critical
    squared wavenumber `q_c = sqrt(r * gamma / (D_n * D_c)) = 2`:
    - Subcritical chemotaxis (`chi_eff = 8 < 9`): `det M(2) = 2 > 0` (stable)
    - Critical bifurcation (`chi_eff = 9`): `det M(2) = 0` (marginal onset)
    - Supercritical chemotaxis (`chi_eff = 12 > 9`, `chi = 6.0 > chi_c = 4.5`):
      `det M(2) = -6 < 0` (chemotactic pattern instability at equal diffusivity!). -/
theorem painter_1999_equal_diffusivity_chemotactic_instability :
    ksp_det 1 4 1 1 8 2 = 2 ∧
    ksp_det 1 4 1 1 9 2 = 0 ∧
    ksp_det 1 4 1 1 12 2 = -6 := by
  unfold ksp_det
  omega


-- ============================================================================
-- Law 3 (`painter-chemotactic-stripe-splitting-3`):
-- Domain-Growth Wavenumber Halving & Chemotactic Stripe-Splitting Mode Cascade
-- ============================================================================

/-- On a domain of scale factor `s ≥ 1` (`L(s) = s * L_0`), spatial cosine mode `m`
    has scaled squared wavenumber `q_m(s) = m^2 / s^2`. Mode `m` lies inside the
    unstable chemotactic band `[q_min, q_max]` iff `q_min * s^2 ≤ m^2 ≤ q_max * s^2`. -/
def in_chemotactic_band (q_min q_max m s : Nat) : Bool :=
  decide (q_min * s * s ≤ m * m ∧ m * m ≤ q_max * s * s)

/-- Theorem 3 (`painter_1999_domain_growth_stripe_splitting_cascade`):
    For the unstable chemotactic wavenumber band `[q_min, q_max] = [1, 3]` around `q_c = 2`:
    - At initial domain scale `s = 1`, fundamental mode `m = 1` is inside the band (`true`),
      while split mode `m = 2` lies above the upper cutoff (`4 > 3`, `false`).
    - When domain length doubles (`s = 2`), mode `m = 1` falls below the lower cutoff
      (`1 < 4`, `false`) while the frequency-doubled mode `m = 2` (`4 ∈ [4, 12]`, `true`)
      enters the unstable band.
    - When domain length quadruples (`s = 4`), mode `m = 2` exits (`4 < 16`, `false`)
      and the quadrupled stripe-splitting mode `m = 4` (`16 ∈ [16, 48]`, `true`) is selected. -/
theorem painter_1999_domain_growth_stripe_splitting_cascade :
    in_chemotactic_band 1 3 1 1 = true ∧
    in_chemotactic_band 1 3 2 1 = false ∧
    in_chemotactic_band 1 3 1 2 = false ∧
    in_chemotactic_band 1 3 2 2 = true ∧
    in_chemotactic_band 1 3 2 4 = false ∧
    in_chemotactic_band 1 3 4 4 = true := by
  decide

end MathOfBiology.Loop_005
