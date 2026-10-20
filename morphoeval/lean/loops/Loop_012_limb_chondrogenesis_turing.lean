/-
  morphoeval/lean/loops/Loop_012_limb_chondrogenesis_turing.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 012:
  Newman & Frisch (1979) Science 205(4407):662-668 — "Dynamics of skeletal
  pattern formation in developing chick limb"
  (`newman-limb-chondrogenesis-turing`, doi:10.1126/science.462174).

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_012_limb_chondrogenesis_turing.lean
-/

namespace MathOfBiology.Loop_012

-- ============================================================================
-- Law 1 (`newman-limb-chondrogenesis-turing-1`):
-- Turing Equal-Diffusion / Hyaluronate-Excess Negative Control (D_u = D_v)
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

/-- Theorem 1 (`newman_turing_equal_diffusion_no_chondrogenesis`):
    Whenever the homogeneous prechondrogenic mesenchyme steady state `(u_0, v_0) = (0, 0)`
    is linearly stable (`tr_J0 < 0` and `det_J0 > 0`), equal activator and inhibitor
    diffusivities (`Dq ≥ 0`, `neg_tr_Dq ≥ 0`, `Dq_sq ≥ 0`) keep `tr(J_k) < 0`
    and `det(J_k) > 0` for every spatial wavenumber, preventing prechondrogenic
    fibronectin/cartilage condensation. -/
theorem newman_turing_equal_diffusion_no_chondrogenesis (tr_J0 det_J0 Dq neg_tr_Dq Dq_sq : Int)
    (htr : tr_J0 < 0) (hdet : det_J0 > 0)
    (hDq : Dq ≥ 0) (hterm : neg_tr_Dq ≥ 0) (hsq : Dq_sq ≥ 0) :
    modal_trace_eq_diff tr_J0 Dq < 0 ∧ modal_det_eq_diff det_J0 neg_tr_Dq Dq_sq > 0 := by
  unfold modal_trace_eq_diff modal_det_eq_diff
  omega


-- ============================================================================
-- Law 2 (`newman-limb-chondrogenesis-turing-2`):
-- Proximodistal Progress-Zone Depth Reduction & 1 -> 2 -> 3 -> 4 Mode Series
-- ============================================================================

/-- In Newman & Frisch's (1979) reaction-diffusion / standing-wave parameterization
    (`2*D_u = 1`, `2*D_v = 100`, `g_v = -200`), the homogeneous trace is
    `tr_J0(f_u) = f_u - 200` and the selected anteroposterior squared mode number
    is `m^2 = k_AP^2(f_u) = (2*D_v*f_u + 2*D_u*g_v) / (4*D_u*D_v) = f_u - 2`. -/
def newman_trace_J0 (fu : Int) : Int :=
  fu - 200

def newman_mode_sq (fu : Int) : Int :=
  fu - 2

/-- Theorem 2 (`newman_proximodistal_mode_transition_series`):
    (1) Strict wavenumber monotonicity: as apical progress-zone depth `L_PD(t)`
        shrinks and hyaluronidase reduces effective fibronectin diffusivity `D_eff(t)`,
        effective self-activation increases (`fu_1 < fu_2 < 200`), strictly increasing
        `newman_mode_sq fu_1 < newman_mode_sq fu_2` while keeping `newman_trace_J0 fu_2 < 0`.
    (2) Exact Hamburger-Hamilton proximodistal skeletal mode progression:
        - HH20 Stylopod (`fu = 3`, `L_PD = 500 um`) -> `m^2 = 1^2 = 1` (1 humerus / femur)
        - HH24 Zeugopod (`fu = 6`, `L_PD = 250 um`) -> `m^2 = 2^2 = 4` (2 radius + ulna)
        - HH27 Wing Autopod (`fu = 11`, `L_PD = 140 um`) -> `m^2 = 3^2 = 9` (3 wing digits II-IV)
        - HH28 Leg Autopod (`fu = 18`, `L_PD = 115 um`) -> `m^2 = 4^2 = 16` (4 pedal digits I-IV). -/
theorem newman_proximodistal_mode_transition_series (fu1 fu2 : Int)
    (h12 : fu1 < fu2) (h2 : fu2 < 200) :
    newman_mode_sq fu1 < newman_mode_sq fu2 ∧
    newman_trace_J0 fu2 < 0 ∧
    newman_mode_sq 3 = 1 * 1 ∧
    newman_mode_sq 6 = 2 * 2 ∧
    newman_mode_sq 11 = 3 * 3 ∧
    newman_mode_sq 18 = 4 * 4 := by
  unfold newman_mode_sq newman_trace_J0
  omega


-- ============================================================================
-- Law 3 (`newman-limb-chondrogenesis-turing-3`):
-- Apical Ectodermal Ridge (AER) Excision Proximodistal Truncation Law
-- ============================================================================

/-- Cumulative chick wing skeletal elements formed across the 3 proximodistal
    compartments (`m = 1` stylopod, `m = 2` zeugopod, `m = 3` autopod) when the
    Apical Ectodermal Ridge (AER) is excised after `stage` compartments (`0..3`). -/
def aer_wing_elements (stage : Nat) : Nat :=
  (if stage ≥ 1 then 1 else 0) +
  (if stage ≥ 2 then 2 else 0) +
  (if stage ≥ 3 then 3 else 0)

/-- Theorem 3 (`newman_aer_excision_proximodistal_truncation`):
    Surgical removal of the Apical Ectodermal Ridge (Saunders 1948; Summerbell et al. 1973;
    Newman & Frisch 1979) produces a strictly monotonic proximodistal skeletal truncation
    hierarchy:
    - Pre-stylopod ablation (`stage = 0`) -> `0` elements
    - HH19-20 early AER removal (`stage = 1`) -> `1` element (stylopod humerus only)
    - HH23-24 intermediate AER removal (`stage = 2`) -> `3` elements (1 humerus + 2 radius/ulna)
    - Intact AER wild-type (`stage = 3`) -> `6` elements (1 humerus + 2 radius/ulna + 3 digits). -/
theorem newman_aer_excision_proximodistal_truncation :
    aer_wing_elements 0 = 0 ∧
    aer_wing_elements 1 = 1 ∧
    aer_wing_elements 2 = 3 ∧
    aer_wing_elements 3 = 6 ∧
    aer_wing_elements 0 < aer_wing_elements 1 ∧
    aer_wing_elements 1 < aer_wing_elements 2 ∧
    aer_wing_elements 2 < aer_wing_elements 3 := by
  decide


-- ============================================================================
-- Pass 3 (FGF8-graded moving-boundary 3D wing bud, Section 4b of the script)
-- ============================================================================

-- ============================================================================
-- Law 4 (`newman-limb-chondrogenesis-turing-4`):
-- 3D Helmholtz Mode Budget  k_c^2 = k_AP^2 + k_DV^2 + k_PD^2
-- ============================================================================

/-- Squared anteroposterior mode energy left over from the critical wavenumber once the
    dorsoventral core hump (`kDV_sq = (π/L_DV)^2`) and the proximodistal progress-zone
    mode (`kPD_sq`, the lowest eigenvalue of the PD operator on the active zone, ≈ (π/L_PD)^2)
    are paid for (Newman & Frisch 1979 Eq. 4, all quantities in integer-scaled units). -/
def helmholtz_ap_mode_sq (kc_sq kDV_sq kPD_sq : Int) : Int :=
  kc_sq - kDV_sq - kPD_sq

/-- Theorem 4 (`newman_helmholtz_3d_mode_bound`):
    (1) The AP mode energy is exactly the Helmholtz budget `kc_sq - kDV_sq - kPD_sq`, so any
        selected AP mode `m_sq` satisfies `m_sq + kDV_sq + kPD_sq = kc_sq`.
    (2) Monotone selection: between two progress-zone states with the SAME DV geometry, if the
        critical wavenumber rises (`kc1 ≤ kc2`, hyaluronidase raising f_u) while the PD mode
        energy does not rise (`kPD2 ≤ kPD1`), the admitted AP mode energy cannot fall; and a
        strict rise in `kc` gives a strict rise in AP mode energy (`1 -> 2 -> 3` elements).
    (3) Dirichlet apical boundary: increasing the PD mode energy at fixed `kc` strictly lowers
        the AP budget -- a shallower zone (larger `kPD_sq`) at fixed kinetics admits only
        SMALLER AP mode numbers, which is why the Table 1 series needs `f_u` to rise with
        shrinking `L_PD`. -/
theorem newman_helmholtz_3d_mode_bound (kc1 kc2 kDV kPD1 kPD2 : Int)
    (hkc : kc1 ≤ kc2) (hPD : kPD2 ≤ kPD1) :
    helmholtz_ap_mode_sq kc1 kDV kPD1 + kDV + kPD1 = kc1 ∧
    helmholtz_ap_mode_sq kc1 kDV kPD1 ≤ helmholtz_ap_mode_sq kc2 kDV kPD2 ∧
    (kc1 < kc2 → helmholtz_ap_mode_sq kc1 kDV kPD1 < helmholtz_ap_mode_sq kc2 kDV kPD2) ∧
    (kPD1 < kPD1 + 1 → helmholtz_ap_mode_sq kc1 kDV (kPD1 + 1) < helmholtz_ap_mode_sq kc1 kDV kPD1) := by
  unfold helmholtz_ap_mode_sq
  omega


-- ============================================================================
-- Law 5 (`newman-limb-chondrogenesis-turing-5`):
-- FGF8 Progress-Zone Depth Is Set By The FGF8 RANGE, Not By AER Output
-- ============================================================================

/-- Proximodistal position (distance from the condensed base, integer-scaled) at which the
    quasi-steady AER gradient `F(z) = F_AER exp(-(L - z)/λ_F)` crosses a log-threshold
    `lnF_thr`:  `z = L - λ_F (ln F_AER - ln F_thr)`.  Written with logarithms so the
    exponential is linear in integer-scaled units. -/
def fgf8_threshold_position (L lam lnF_AER lnF_thr : Int) : Int :=
  L - lam * (lnF_AER - lnF_thr)

/-- Active progress-zone depth: distance between the exit threshold (`lnF_exit`, proximal,
    where tissue freezes) and the apical threshold (`lnF_ap`, distal, below which FGF8 no
    longer keeps mesenchyme refractory). -/
def fgf8_progress_zone_depth (L lam lnF_AER lnF_ap lnF_exit : Int) : Int :=
  fgf8_threshold_position L lam lnF_AER lnF_ap - fgf8_threshold_position L lam lnF_AER lnF_exit

/-- Theorem 5 (`fgf8_progress_zone_depth_independent_of_aer_output`):
    For ANY AER output `lnF_AER` and bud length `L`, the active progress-zone depth equals
    `λ_F (ln F_ap - ln F_exit)` -- it depends only on the FGF8 range `λ_F` and the two
    thresholds.  Hence the shrinking Table 1 depths (500 -> 250 -> 140 μm) must come from a
    shrinking RANGE (the hyaluronidase clock `λ_F(t) = λ_0 exp(-t/τ_H)`), not from a change in
    the AER's FGF8 level; and halving the range halves the depth (`lam -> 2 lam` doubles it). -/
theorem fgf8_progress_zone_depth_independent_of_aer_output
    (L L' lam lnF_AER lnF_AER' lnF_ap lnF_exit : Int) :
    fgf8_progress_zone_depth L lam lnF_AER lnF_ap lnF_exit = lam * (lnF_ap - lnF_exit) ∧
    fgf8_progress_zone_depth L lam lnF_AER lnF_ap lnF_exit =
      fgf8_progress_zone_depth L' lam lnF_AER' lnF_ap lnF_exit ∧
    fgf8_progress_zone_depth L (2 * lam) lnF_AER lnF_ap lnF_exit =
      2 * fgf8_progress_zone_depth L lam lnF_AER lnF_ap lnF_exit := by
  unfold fgf8_progress_zone_depth fgf8_threshold_position
  refine ⟨?_, ?_, ?_⟩ <;> (simp only [Int.mul_sub, Int.mul_assoc]; omega)


-- ============================================================================
-- Law 6 (`newman-limb-chondrogenesis-turing-6`):
-- Hill-2 FGF8 Mitogenic Response Is Strictly Monotone (Graded Proliferation)
-- ============================================================================

/-- Hill-2 proliferation `S(F) = S_max F^2 / (K^2 + F^2)` compared across two FGF8 levels by
    cross-multiplication: `S(F1) < S(F2)  <=>  F1^2 (K^2 + F2^2) < F2^2 (K^2 + F1^2)`.
    The two sides of that inequality, in natural (integer-scaled) units. -/
def hill2_cross_lhs (K F1 F2 : Nat) : Nat := F1 * F1 * (K * K + F2 * F2)
def hill2_cross_rhs (K F1 F2 : Nat) : Nat := F2 * F2 * (K * K + F1 * F1)

/-- Theorem 6 (`fgf8_hill_proliferation_monotone`):
    For `K > 0` and `F1 < F2`, `S(F1) < S(F2)` (the cross-multiplied form), i.e. the Hill-2
    FGF8 mitogenic response is strictly increasing in FGF8 -- proliferation and hence the
    moving-boundary outgrowth velocity `v_tip = ∫ S dz` are graded monotonically with distance
    from the AER (Sun et al. 2002; Boehm et al. 2010). -/
theorem fgf8_hill_proliferation_monotone (K F1 F2 : Nat) (hK : 0 < K) (h12 : F1 < F2) :
    hill2_cross_lhs K F1 F2 < hill2_cross_rhs K F1 F2 := by
  unfold hill2_cross_lhs hill2_cross_rhs
  have hsq : F1 * F1 < F2 * F2 := Nat.mul_self_lt_mul_self h12
  have hKK : 0 < K * K := Nat.mul_pos hK hK
  -- F1² (K² + F2²) = F1² K² + F1² F2²  <  F2² K² + F2² F1² = F2² (K² + F1²)
  rw [Nat.mul_add, Nat.mul_add]
  have h1 : F1 * F1 * (K * K) < F2 * F2 * (K * K) := Nat.mul_lt_mul_of_pos_right hsq hKK
  have h2 : F1 * F1 * (F2 * F2) = F2 * F2 * (F1 * F1) := Nat.mul_comm _ _
  omega

end MathOfBiology.Loop_012
