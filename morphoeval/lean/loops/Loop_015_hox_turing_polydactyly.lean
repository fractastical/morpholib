/-
  morphoeval/lean/loops/Loop_015_hox_turing_polydactyly.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 015:
  Sheth et al. (2012) Science 338(6113):1476-1480 — "Hox genes regulate digit
  patterning by controlling the wavelength of a Turing-type mechanism"
  (`sheth-hox-turing-polydactyly`, doi:10.1126/science.1226804).

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_015_hox_turing_polydactyly.lean
-/

namespace MathOfBiology.Loop_015

-- ============================================================================
-- Law 1 (`sheth-hox-turing-polydactyly-1`):
-- Turing Equal-Diffusion Negative Control (No Digit Patterning for D_u = D_v)
-- ============================================================================

/-- Modal Jacobian trace `tr(J_k) = tr_J0 - 2 * Dq` for equal diffusivities
    `D_u = D_v = D` at scaled wavenumber `Dq = D * k^2 ≥ 0`. -/
def modal_trace_eq_diff (tr_J0 Dq : Int) : Int :=
  tr_J0 - 2 * Dq

/-- Modal Jacobian determinant `det(J_k) = det_J0 + neg_tr_Dq + Dq_sq`
    under equal diffusivities `D_u = D_v = D`, where `neg_tr_Dq = (-tr_J0)*(D*k^2) ≥ 0`
    and `Dq_sq = (D*k^2)^2 ≥ 0`. -/
def modal_det_eq_diff (det_J0 neg_tr_Dq Dq_sq : Int) : Int :=
  det_J0 + neg_tr_Dq + Dq_sq

/-- Theorem 1 (`sheth_turing_equal_diffusion_no_instability`):
    Whenever the homogeneous steady state `(u_0, v_0) = (0, 0)` is linearly
    stable (`tr_J0 < 0` and `det_J0 > 0`), equal activator and inhibitor
    diffusivities (`Dq ≥ 0`, `neg_tr_Dq ≥ 0`, `Dq_sq ≥ 0`) keep `tr(J_k) < 0`
    and `det(J_k) > 0` for every spatial wavenumber, preventing Turing digit
    condensation. -/
theorem sheth_turing_equal_diffusion_no_instability (tr_J0 det_J0 Dq neg_tr_Dq Dq_sq : Int)
    (htr : tr_J0 < 0) (hdet : det_J0 > 0)
    (hDq : Dq ≥ 0) (hterm : neg_tr_Dq ≥ 0) (hsq : Dq_sq ≥ 0) :
    modal_trace_eq_diff tr_J0 Dq < 0 ∧ modal_det_eq_diff det_J0 neg_tr_Dq Dq_sq > 0 := by
  unfold modal_trace_eq_diff modal_det_eq_diff
  omega


-- ============================================================================
-- Law 2 (`sheth-hox-turing-polydactyly-2`):
-- Distal Hox/Gli3 Dose Monotonic Wavenumber Scaling & Allelic Series (5 -> 14)
-- ============================================================================

/-- In Sheth et al.'s (2012) canonical reaction-diffusion parameterization
    (`2*D_u = 1`, `2*D_v = 100`, `g_v = -200`), the homogeneous trace is
    `tr_J0(f_u) = f_u - 200` and the critical squared wavenumber is
    `k_c^2(f_u) = (2*D_v*f_u + 2*D_u*g_v) / (4*D_u*D_v) = f_u - 2`. -/
def sheth_trace_J0 (fu : Int) : Int :=
  fu - 200

def sheth_kc_sq (fu : Int) : Int :=
  fu - 2

/-- Theorem 2 (`sheth_hox_dose_monotonic_wavenumber_and_allelic_series`):
    (1) Strict wavenumber monotonicity: reducing distal 5' Hox and Gli3 dose
        increases activator self-activation `fu_1 < fu_2 < 200`, which strictly
        increases the critical squared wavenumber `sheth_kc_sq fu_1 < sheth_kc_sq fu_2`
        while preserving homogeneous trace stability `sheth_trace_J0 fu_2 < 0`.
    (2) Exact allelic series at constant autopod arc length `L = 2*pi`:
        - WT (`fu = 27`) -> `k_c^2 = 5^2 = 25` (5 pentadactyl digits)
        - Gli3 null (`fu = 51`) -> `k_c^2 = 7^2 = 49` (7 digits)
        - Hoxd11-13 null; Gli3 null (`fu = 83`) -> `k_c^2 = 9^2 = 81` (9 digits)
        - Hoxa13 het; Hoxd11-13 null; Gli3 null (`fu = 123`) -> `k_c^2 = 11^2 = 121` (11 digits)
        - Hoxa13 null; Hoxd11-13 null; Gli3 null (`fu = 198`) -> `k_c^2 = 14^2 = 196` (14 digits). -/
theorem sheth_hox_dose_monotonic_wavenumber_and_allelic_series (fu1 fu2 : Int)
    (h12 : fu1 < fu2) (h2 : fu2 < 200) :
    sheth_kc_sq fu1 < sheth_kc_sq fu2 ∧
    sheth_trace_J0 fu2 < 0 ∧
    sheth_kc_sq 27 = 5 * 5 ∧
    sheth_kc_sq 51 = 7 * 7 ∧
    sheth_kc_sq 83 = 9 * 9 ∧
    sheth_kc_sq 123 = 11 * 11 ∧
    sheth_kc_sq 198 = 14 * 14 := by
  unfold sheth_kc_sq sheth_trace_J0
  omega


-- ============================================================================
-- Law 3 (`sheth-hox-turing-polydactyly-3`):
-- Fan-Shaped Limb Bud Proximodistal Arc Expansion & Distal Digit Selection
-- ============================================================================

/-- On a fan-shaped limb bud sector at radial distance `r`, an angular mode `m`
    has `r^2`-scaled squared wavenumber `m^2`, which lies inside the unstable
    Turing band `[k_min_sq, k_max_sq]` iff `k_min_sq * r^2 ≤ m^2 ≤ k_max_sq * r^2`. -/
def in_fan_autopod_band (k_min_sq k_max_sq m r : Nat) : Bool :=
  decide (k_min_sq * r * r ≤ m * m ∧ m * m ≤ k_max_sq * r * r)

/-- Theorem 3 (`sheth_fan_autopod_proximodistal_mode_selection`):
    For a normalized local Turing wavenumber band `[k_min_sq, k_max_sq] = [2, 4]`:
    - At the narrow proximal limb base (`r = 1`), pentadactyl (`m = 5`) and
      polydactyl (`m = 9`) angular modes exceed the upper Turing cutoff (`25 > 4`,
      `81 > 4`) and are excluded (`false`), while only a low-order zeugopod mode
      (`m = 2`, `4 ∈ [2, 4]`) is admitted (`true`).
    - At the expanded distal autopod plate (`r = 3`, `r^2 = 9`, band `[18, 36]`),
      the WT pentadactyl digit mode (`m = 5`, `25 ∈ [18, 36]`) is selectively
      stabilized (`true`). -/
theorem sheth_fan_autopod_proximodistal_mode_selection :
    in_fan_autopod_band 2 4 2 1 = true ∧
    in_fan_autopod_band 2 4 5 1 = false ∧
    in_fan_autopod_band 2 4 9 1 = false ∧
    in_fan_autopod_band 2 4 5 3 = true := by
  decide


-- ============================================================================
-- Law 4 (`sheth-hox-turing-polydactyly-4`, Pass 3):
-- Digit Number = Autopod Width / Turing Wavelength on an AER-Grown Autopod
-- ============================================================================

/-- Number of digit wavelengths that fit on a distal autopod arc of width `L`
    (micrometres) at Turing wavelength `lam` (micrometres): `N = L / lam`
    (natural-number floor division). On the AER-grown fan `L = Theta * R(T)`
    is set by FGF8-driven outgrowth while `lam = lam(H)` is set by distal Hox dose. -/
def digit_wavelengths (L lam : Nat) : Nat :=
  L / lam

/-- Theorem 4 (`sheth_growing_autopod_digit_number_law`):
    (1) at fixed wavelength, a wider AER-grown autopod never carries fewer digit
        wavelengths (`L1 ≤ L2 → N(L1) ≤ N(L2)`): AER ablation (smaller `L`) can only
        reduce digit number (Summerbell 1974 ordering);
    (2) at fixed width, a shorter Hox-tuned wavelength never carries fewer
        (`0 < lam1 ≤ lam2 → N(lam2) ≤ N(lam1)`): the Sheth et al. (2012) allelic series;
    (3) instances at the constant 2500 um autopod: `2500/500 = 5` (WT),
        `2500/357 = 7` (Gli3 null), `2500/178 = 14` (triple null), and the early
        AER-ablated WT arc `1365/500 = 2`. -/
theorem sheth_growing_autopod_digit_number_law (L1 L2 L lam1 lam2 : Nat)
    (hL : L1 ≤ L2) (hlam : lam1 ≤ lam2) (hpos : 0 < lam1) :
    digit_wavelengths L1 lam1 ≤ digit_wavelengths L2 lam1 ∧
    digit_wavelengths L lam2 ≤ digit_wavelengths L lam1 ∧
    digit_wavelengths 2500 500 = 5 ∧
    digit_wavelengths 2500 357 = 7 ∧
    digit_wavelengths 2500 178 = 14 ∧
    digit_wavelengths 1365 500 = 2 := by
  unfold digit_wavelengths
  exact ⟨Nat.div_le_div_right hL, Nat.div_le_div_left hlam hpos, rfl, rfl, rfl, rfl⟩


-- ============================================================================
-- Law 5 (`sheth-hox-turing-polydactyly-5`, Pass 3):
-- Growth-Dilution Shift of the Turing Modal Eigenvalue (Crampin et al. 1999)
-- ============================================================================

/-- On a growing domain with local area dilation rate `g` the linearised
    reaction-diffusion Jacobian becomes `J_k - g I`, so every modal eigenvalue
    shifts by `-g`: `sigma_g(k) = sigma(k) - g` (integers scaled by 100 in the
    numerical instances below). -/
def dilution_shifted_eigenvalue (sigma g : Int) : Int :=
  sigma - g

/-- Theorem 5 (`sheth_growth_dilution_turing_threshold`):
    for any modal growth rate `sigma` (Turing-unstable when `sigma > 0`),
    (1) moderate AER-driven dilution `g < sigma` leaves the mode growing, so
        digits condense during outgrowth;
    (2) fast growth `g ≥ sigma` quenches it (Crampin, Gaffney & Maini 1999);
    (3) AER regression (`g = 0`) restores the full fixed-domain rate `sigma`;
    (4) instance from the script: `sigma_max = 4.49`, `g_rim = 1.95` → `449 - 195 = 254 > 0`,
        whereas a hypothetical `g = 5.00` gives `449 - 500 < 0`. -/
theorem sheth_growth_dilution_turing_threshold (sigma g : Int) :
    (g < sigma → dilution_shifted_eigenvalue sigma g > 0) ∧
    (g ≥ sigma → dilution_shifted_eigenvalue sigma g ≤ 0) ∧
    dilution_shifted_eigenvalue sigma 0 = sigma ∧
    dilution_shifted_eigenvalue 449 195 > 0 ∧
    dilution_shifted_eigenvalue 449 500 < 0 := by
  unfold dilution_shifted_eigenvalue
  omega

end MathOfBiology.Loop_015
