/-
  morphoeval/lean/loops/Loop_005_chemotactic_stripe_splitting.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 005:
  Painter, Maini & Othmer (1999) PNAS 96(10):5549-5554 — "Stripe formation in
  juvenile Pomacanthus explained by a generalized Turing mechanism with chemotaxis"
  (doi:10.1073/pnas.96.10.5549, PMID 10318921; `painter-chemotactic-stripe-splitting`).

  Laws 1-3 formalise the generic Keller-Segel (1970) / Painter-Hillen (2011)
  chemotaxis-logistic system used for the Myxococcus xanthus rippling comparison
  (equal-diffusivity chemotactic instability and domain-growth mode cascade).
  Laws 4-6 (Pass 3) formalise the actual PNAS 1999 model: the Kondo-Asai (1995)
  piecewise-linear Turing kinetics driving chemotactic iridophores on a growing
  Pomacanthus flank (3 -> 6 -> 12 stripe doubling), and the slime-trail
  orientation-tensor mobility law used for the Myxococcus extension.

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


-- ============================================================================
-- Law 4 (`painter-chemotactic-stripe-splitting-4`, Pass 3):
-- Painter, Maini & Othmer (1999) Fig. 3 Kondo-Asai Turing kinetics are
-- diffusion-driven unstable and select the three-stripe mode on L = 11
-- ============================================================================

/-- Trace of the Kondo-Asai kinetic Jacobian `J = [[f_u, f_v], [g_u, g_v]]`. -/
def turing_trace (fu gv : Int) : Int := fu + gv

/-- Determinant of the kinetic Jacobian. -/
def turing_det (fu fv gu gv : Int) : Int := fu * gv - fv * gu

/-- Diffusion-weighted trace `h = D_v f_u + D_u g_v`; Turing instability needs
    `h > 0` and `h^2 > 4 D_u D_v det J`, with `k_c^2 = h / (2 D_u D_v)`. -/
def turing_h (Du Dv fu gv : Int) : Int := Dv * fu + Du * gv

/-- Theorem 4a (`painter_1999_pomacanthus_turing_instability`):
    For the PNAS 1999 Fig. 3 kinetics `f = 0.08u - 0.08v + 0.05 - 0.03u`,
    `g = 0.1u - 0.15 - 0.06v` (Kondo & Asai 1995), linearised about
    `(u*, v*) = (3, 2.5)` the Jacobian is `[[0.05, -0.08], [0.1, -0.06]]`.
    Scaling the Jacobian by 100 and the diffusivities `D_u = 0.007, D_v = 0.1`
    by 1000 gives integer data `(5, -8, 10, -6)` and `(7, 100)`:
    `tr J = -1 < 0`, `det J = 50 > 0` (kinetically stable),
    `h = 458 > 0` and `h^2 = 209764 > 4 D_u D_v det J = 140000`
    (diffusion-driven instability), i.e. `k_c^2 = 0.00458 / 0.0014 = 3.271`. -/
theorem painter_1999_pomacanthus_turing_instability :
    turing_trace 5 (-6) = -1 ∧
    turing_det 5 (-8) 10 (-6) = 50 ∧
    turing_h 7 100 5 (-6) = 458 ∧
    turing_h 7 100 5 (-6) * turing_h 7 100 5 (-6) > 4 * 7 * 100 * turing_det 5 (-8) 10 (-6) := by
  unfold turing_trace turing_det turing_h
  decide

/-- Zero-flux (Neumann) half-wave mode `m` on `L = 11` has `k^2 = m^2 pi^2 / 121`;
    scaled by `1000 * 121` with `pi^2 ~ 9.870`: `q(m) = 9870 m^2`. -/
def neumann_mode_q (m : Nat) : Nat := m * m * 9870

/-- Unstable Turing band `[k_-^2, k_+^2] = [1.385, 5.158]` of the Fig. 3 kinetics, scaled by `1000 * 121`. -/
def pomacanthus_band_lo : Nat := 1385 * 121
def pomacanthus_band_hi : Nat := 5158 * 121

/-- Fastest-growing squared wavenumber `k_c^2 = 3.271`, scaled by `1000 * 121`. -/
def pomacanthus_q_c : Nat := 3271 * 121

/-- Distance of Neumann mode `m` from the fastest-growing wavenumber. -/
def dist_to_qc (m : Nat) : Nat :=
  if neumann_mode_q m ≤ pomacanthus_q_c then pomacanthus_q_c - neumann_mode_q m
  else neumann_mode_q m - pomacanthus_q_c

/-- Theorem 4b (`pomacanthus_three_stripe_mode_selection`):
    On the initial Fig. 3 domain `L(0) = 11` only Neumann half-wave modes
    `m ∈ {5, 6, 7}` lie in the unstable band (`m = 4` below, `m = 8` above), and
    `m = 6` (three full wavelengths, i.e. the three-stripe juvenile pattern of
    Fig. 1a) is the mode closest to `k_c^2`. -/
theorem pomacanthus_three_stripe_mode_selection :
    (pomacanthus_band_lo ≤ neumann_mode_q 5 ∧ neumann_mode_q 7 ≤ pomacanthus_band_hi) ∧
    neumann_mode_q 4 < pomacanthus_band_lo ∧ pomacanthus_band_hi < neumann_mode_q 8 ∧
    dist_to_qc 6 < dist_to_qc 5 ∧ dist_to_qc 6 < dist_to_qc 7 := by
  decide


-- ============================================================================
-- Law 5 (`painter-chemotactic-stripe-splitting-5`, Pass 3):
-- Kondo-Asai (1995) stripe-doubling law on a uniformly growing surface
-- ============================================================================

/-- Theorem 5 (`kondo_asai_stripe_doubling_mode_selection`):
    Under uniform isotropic growth by scale `s` the Laplace-Beltrami operator of
    the flank surface scales as `1/s^2`, so a meridional mode with `m` stripes has
    `q_m(s) = m^2 / s^2` in units where the 3-stripe mode at `s = 1` sits at `q = 9`.
    (1) The wavelength-preserving mode is exactly `m = 3 s` (`m^2 = 9 s^2 ⟺ m = 3 s`),
        which is the Kondo-Asai observation that stripe number doubles with body length
        while spacing is maintained (3 -> 6 -> 12).
    (2) With the Fig. 3 unstable band `[k_-^2, k_+^2] / k_c^2 * 9 = [3, 28]`:
        at `s = 1` mode 3 is unstable but 6 is not; at `s = 2` mode 3 has left the band
        and mode 6 is selected; at `s = 4` mode 6 has left and mode 12 is selected. -/
theorem kondo_asai_stripe_doubling_mode_selection (m s : Nat) :
    (m * m = 9 * (s * s) ↔ m = 3 * s) ∧
    in_chemotactic_band 3 28 3 1 = true ∧ in_chemotactic_band 3 28 6 1 = false ∧
    in_chemotactic_band 3 28 3 2 = false ∧ in_chemotactic_band 3 28 6 2 = true ∧
    in_chemotactic_band 3 28 6 4 = false ∧ in_chemotactic_band 3 28 12 4 = true := by
  refine ⟨?_, by decide, by decide, by decide, by decide, by decide, by decide⟩
  have h : 9 * (s * s) = (3 * s) * (3 * s) := by
    rw [Nat.mul_mul_mul_comm]
  rw [h]
  exact Nat.mul_self_inj


-- ============================================================================
-- Law 6 (`painter-chemotactic-stripe-splitting-6`, Pass 3):
-- Slime-trail orientation tensor: anisotropic mobility lowers the
-- Keller-Segel chemotactic threshold along the trail director
-- ============================================================================

/-- Theorem 6 (`slime_trail_anisotropic_mobility_lowers_chemotactic_threshold`):
    With the mobility tensor `M = I + a Q` of the slime-trail nematic order tensor
    `Q`, the cell diffusivity and chemotactic sensitivity along the trail director
    are both multiplied by `m = 1 + a S/2 ≥ 1`. The equal-diffusivity threshold of
    Law 2 becomes `chi_eff_c(m) = (sqrt(r D_c) + sqrt(gamma m D_n))^2 / m`, which for
    `r = D_c = D_n = 1, gamma = 4` and a perfect-square mobility `m = j^2` equals
    `(1 + 2 j)^2 / j^2`. Since `(1 + 2 j)^2 < 9 j^2` for every `j ≥ 2` (with equality
    at `j = 1`, the isotropic value `chi_eff_c = 9`), trail-following strictly lowers
    the chemotactic threshold: rippling ridges can be sustained below `chi_c = 4.5`. -/
theorem slime_trail_anisotropic_mobility_lowers_chemotactic_threshold (j : Nat) (hj : 2 ≤ j) :
    (1 + 2 * j) * (1 + 2 * j) < 9 * (j * j) ∧ (1 + 2 * 1) * (1 + 2 * 1) = 9 * (1 * 1) := by
  refine ⟨?_, by decide⟩
  have h1 : 2 * j ≤ j * j := Nat.mul_le_mul_right j hj
  have h2 : (1 + 2 * j) * (1 + 2 * j) = 1 + 4 * j + 4 * (j * j) := by
    have : 2 * j * (2 * j) = 4 * (j * j) := by
      rw [Nat.mul_assoc, Nat.mul_left_comm j 2 j, ← Nat.mul_assoc]
    simp only [Nat.add_mul, Nat.mul_add, Nat.one_mul, Nat.mul_one, this]
    omega
  omega

end MathOfBiology.Loop_005
