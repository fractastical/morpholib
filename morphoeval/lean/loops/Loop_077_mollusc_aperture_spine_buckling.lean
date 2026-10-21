/-
  morphoeval/lean/loops/Loop_077_mollusc_aperture_spine_buckling.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 077:
  Moulton, Goriely & Chirat (2015) J. Theor. Biol. 364:420-430,
  Chirat, Moulton & Goriely (2013) PNAS 110:6015-6020 &
  Moulton, Goriely & Chirat (2012) J. Theor. Biol. 311:69-79
  Hierarchical Aperture Spine Folding, Secondary Fractal Frond Serrations &
  Episodic Varix Periodicity (`moulton-mollusc-aperture-spine-buckling`).

  Zero external dependencies, zero `sorry`s.
  Verify with:
    lean morphoeval/lean/loops/Loop_077_mollusc_aperture_spine_buckling.lean
-/

namespace MathOfBiology.Loop_077

-- ============================================================================
-- Law 1: Multi-Generational Fractal Fronding Wavenumber Cascade
-- ============================================================================

/-- Cubic flexural thickness `h^3` governing mantle bending rigidity `D_i = E*h_i^3/12`
    and fourth-power wavenumber `k_i^* = (3 * K_s / (E * h_i^3))^(1/4)`. -/
def frond_cubic_thickness (h : Nat) : Nat :=
  h * h * h

/-- Fourth-power wavenumber dispersion residual `E * h_i^3 * (k_i^*)^4 - 3 * K_s`
    governing the critical buckling wavenumber `k_i^* = (3 * K_s / (E * h_i^3))^(1/4)`
    at each hierarchical spine generation `i ∈ {1, 2, 3}`. -/
def frond_wavenumber_residual (E_h3 k4 K_s : Int) : Int :=
  E_h3 * k4 - 3 * K_s

/-- Wavenumber cascade gap `h_1^3 - h_2^3` measuring the fourth-power wavenumber
    amplification `(k_2^*)^4 - (k_1^*)^4` when mantle margin thickness thins from
    `h_1` (primary spine stem) to `h_2 < h_1` (secondary frond margin) and
    `h_3 < h_2` (tertiary micro-crenulation margin). -/
def frond_cascade_gap (h_coarse h_fine : Int) : Int :=
  h_coarse * h_coarse * h_coarse - h_fine * h_fine * h_fine

/-- Theorem 1 (`moulton_secondary_frond_cascade`):
    (1) At the critical wavenumber `E * h_i^3 * k_i^4 = 3 * K_s`, the dispersion
        residual vanishes identically (`frond_wavenumber_residual E_h3 (3 * K_s) (E_h3 * K_s) = 0`).
    (2) When distal mantle thickness decreases across hierarchical spine generations
        (`h_3 = 1 < h_2 = 2 < h_1 = 6`), the cubic flexural thickness strictly
        decreases (`h_3^3 < h_2^3 < h_1^3`), driving strictly positive wavenumber
        cascade gaps (`k_3^* > k_2^* > k_1^*`) in *Chicoreus ramosus*. -/
theorem moulton_secondary_frond_cascade (E_h3 K_s : Int) :
    frond_wavenumber_residual E_h3 (3 * K_s) (E_h3 * K_s) = 0 ∧
    frond_cubic_thickness 1 < frond_cubic_thickness 2 ∧
    frond_cubic_thickness 2 < frond_cubic_thickness 6 ∧
    frond_cascade_gap 6 1 = 215 ∧
    frond_cascade_gap 6 2 = 208 := by
  refine ⟨?_, by decide, by decide, by decide, by decide⟩
  simp [frond_wavenumber_residual, Int.mul_assoc, Int.mul_comm]

-- ============================================================================
-- Law 2: Tubular vs Open-Gutter Spine Folding Closure Criterion
-- ============================================================================

/-- Scaled elastica loop basal neck gap `Δx_neck(γ) = w_base * (γ_fold - γ)`
    governing self-contact closure of a buckled hollow spine:
    - For `γ_c < γ < γ_fold`, `Δx_neck > 0`: open-gutter / flared frond apex (*Chicoreus*).
    - At `γ = γ_fold`, `Δx_neck = 0`: incipient self-contact kiss point.
    - For `γ > γ_fold`, `Δx_neck < 0`: self-contact seals the hollow tubular spine base
      (*Murex pecten* siphonal comb spines). -/
def spine_neck_gap (w_base gamma_fold gamma : Int) : Int :=
  w_base * (gamma_fold - gamma)

/-- Boolean indicator of tubular self-contact closure (`Δx_neck ≤ 0`). -/
def is_tubular_sealed (w_base gamma_fold gamma : Int) : Bool :=
  decide (spine_neck_gap w_base gamma_fold gamma ≤ 0)

/-- Theorem 2 (`moulton_tubular_spine_closure_criterion`):
    Along a buckled elastica spine loop of positive basal width factor `w_base > 0`:
    (1) At the critical folding stretch `γ = γ_fold`, the basal neck gap vanishes
        identically (`spine_neck_gap w_base γ_fold γ_fold = 0`).
    (2) Below `γ_fold` (`γ = 130 < 142`), the spine remains an open
        gutter (`Δx_neck > 0`, `is_tubular_sealed = false`) flaring at the distal apex.
    (3) Above `γ_fold` (`γ = 168 > 142`), self-contact closes the base
        (`Δx_neck ≤ 0`, `is_tubular_sealed = true`), forming a sealed hollow tube. -/
theorem moulton_tubular_spine_closure_criterion (w_base gamma_fold : Int) :
    spine_neck_gap w_base gamma_fold gamma_fold = 0 ∧
    spine_neck_gap 4 142 130 = 48 ∧
    is_tubular_sealed 4 142 130 = false ∧
    is_tubular_sealed 4 142 142 = true ∧
    spine_neck_gap 4 142 168 = -104 ∧
    is_tubular_sealed 4 142 168 = true := by
  refine ⟨?_, by decide, by decide, by decide, by decide, by decide⟩
  simp [spine_neck_gap]

-- ============================================================================
-- Law 3: Episodic Varix Rest-Period Spine Periodicity Law
-- ============================================================================

/-- Squared spine amplitude `A_spine^2(γ, γ_c) = A_0^2 * (γ - γ_c)` over natural
    saturating growth units: during inter-varix spiral shell growth (`γ ≤ γ_c`),
    excess mantle compression vanishes and `A_spine^2 = 0` (smooth inter-varix whorl);
    during episodic rest-period varix bursts (`γ > γ_c`), `A_spine^2 > 0`. -/
def episodic_varix_amplitude_sq (A0_sq gamma gamma_c : Nat) : Nat :=
  A0_sq * (gamma - gamma_c)

/-- Triradiate varix angular phase invariant `3 * Δφ_varix = 360°` for 3 episodic
    varix bursts per 360° shell whorl (`Δφ_varix = 120°`, as in *Murex pecten* and
    *Chicoreus ramosus*). -/
def triradiate_whorl_period_deg (n_varices_per_whorl delta_phi_deg : Nat) : Nat :=
  n_varices_per_whorl * delta_phi_deg

/-- Theorem 3 (`moulton_episodic_varix_rest_period`):
    (1) At or below the critical buckling threshold (`γ = γ_c` or `γ = 0 ≤ γ_c`),
        the spine amplitude vanishes identically (`A_spine^2 = 0`), producing smooth
        inter-varix shell whorls (subcritical negative control).
    (2) During supercritical varix bursts (`γ = γ_c + dγ + 1`),
        `A_spine^2 = A_0^2 * (dγ + 1) > 0`.
    (3) Three episodic varices spaced at `Δφ = 120°` complete an exact `360°` whorl
        phase lock (`3 * 120 = 360`), aligning successive varix spine rows across whorls. -/
theorem moulton_episodic_varix_rest_period (A0_sq gamma_c dgamma : Nat) :
    episodic_varix_amplitude_sq A0_sq gamma_c gamma_c = 0 ∧
    episodic_varix_amplitude_sq A0_sq 0 gamma_c = 0 ∧
    episodic_varix_amplitude_sq A0_sq (gamma_c + (dgamma + 1)) gamma_c = A0_sq * (dgamma + 1) ∧
    triradiate_whorl_period_deg 3 120 = 360 := by
  refine ⟨?_, ?_, ?_, by decide⟩
  · simp [episodic_varix_amplitude_sq]
  · simp [episodic_varix_amplitude_sq]
  · simp [episodic_varix_amplitude_sq]

-- ============================================================================
-- Law 4 (Pass 3): Oblique Spiral-Cord / Varix Intersection Registration
-- ============================================================================

/-- Number of spiral-cord / varix intersections along a margin of length `L` when
    cords of spacing `lam_c` (measured normal to the cords) cross the margin at
    obliquity `psi` with `cos psi = cos_num / cos_den`: the spacing of the intersections
    along the varix is `d = lam_c / cos psi = lam_c * cos_den / cos_num`, so the count is
    `L / d = L * cos_num / (lam_c * cos_den)` (natural-number division). -/
def cord_varix_intersections (L lam_c cos_num cos_den : Nat) : Nat :=
  L * cos_num / (lam_c * cos_den)

/-- Free (untemplated) spine count `N_1 = L / lambda_1` of the Moulton–Chirat law. -/
def free_spine_count (L lam1 : Nat) : Nat :=
  L / lam1

/-- Theorem 4 (`cord_varix_intersection_registration`):
    (1) Whenever the oblique cord spacing projected onto the varix equals the free
        buckling wavelength (`lam_c * cos_den = lam1 * cos_num`, i.e. `lam_c = lam1 cos psi`)
        the number of cord/varix intersections equals the free spine count, giving a 1:1
        cord-templated spine row (the centre of the locking tongue).
    (2) Concrete 3-4-5 instance: `lam_c = 4`, `cos psi = 4/5`, `L = 80` gives intersection
        spacing `5` along the varix, `16` intersections, and `16` free spines with `lam1 = 5`.
    (3) Rotating the same cords to `cos psi = 3/5` (steeper obliquity) changes the
        intersection count to `12 ≠ 16`: registration is lost without re-locking. -/
theorem cord_varix_intersection_registration
    (L lam1 lam_c cos_num cos_den : Nat) (h : lam_c * cos_den = lam1 * cos_num)
    (hc : 0 < cos_num) :
    cord_varix_intersections L lam_c cos_num cos_den = free_spine_count L lam1 ∧
    cord_varix_intersections 80 4 4 5 = 16 ∧
    free_spine_count 80 5 = 16 ∧
    cord_varix_intersections 80 4 3 5 = 12 := by
  refine ⟨?_, by decide, by decide, by decide⟩
  unfold cord_varix_intersections free_spine_count
  rw [h]
  exact Nat.mul_div_mul_right L lam1 hc

-- ============================================================================
-- Law 5 (Pass 3): Ontogenetic Spine-Count Conservation Exponent
-- ============================================================================

/-- Four-times the log–log exponent of the primary spine count across instars,
    `4 * (1 - 3 beta / 4) = 4 - 3 beta`, written for a rational allometric exponent
    `beta = beta_num / beta_den` as the integer `4 * beta_den - 3 * beta_num`
    (so the true exponent is this value divided by `4 * beta_den`). Here
    `h_1 ∝ L^beta` and `N_1 = L / lambda_1(h_1) ∝ L^(1 - 3 beta / 4)`. -/
def spine_count_exponent_num (beta_num beta_den : Int) : Int :=
  4 * beta_den - 3 * beta_num

/-- Theorem 5 (`ontogenetic_spine_count_conservation`):
    (1) The spine count is conserved across instars (zero exponent) iff `3 beta = 4 beta_den`,
        i.e. `beta = 4/3` (mantle thickness must grow faster than isometrically).
    (2) Isometric growth `beta = 1` gives a positive exponent (`4 - 3 = 1`, i.e. `N_1 ∝ L^(1/4)`,
        spines are added across instars); `beta = 3/2` gives a negative exponent
        (`4*2 - 3*3 = -1`, spines are lost). -/
theorem ontogenetic_spine_count_conservation (beta_num beta_den : Int) :
    (spine_count_exponent_num beta_num beta_den = 0 ↔ 3 * beta_num = 4 * beta_den) ∧
    spine_count_exponent_num 4 3 = 0 ∧
    spine_count_exponent_num 1 1 = 1 ∧
    spine_count_exponent_num 3 2 = -1 := by
  refine ⟨?_, by decide, by decide, by decide⟩
  unfold spine_count_exponent_num
  omega

end MathOfBiology.Loop_077
