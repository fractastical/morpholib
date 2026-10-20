/-
  morphoeval/lean/loops/Loop_014_tgfb_fibronectin_limb.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 014:
  Hentschel, Glimm, Glazier & Newman (2004) Proc. R. Soc. Lond. B 271(1549):1713-1722
  (`doi:10.1098/rspb.2004.2772`) & Kiskowski, Alber, Glimm, Hentschel, Glazier & Newman
  (2004) Dev. Biol. 271(2):372-387 (`doi:10.1016/j.ydbio.2004.03.038`) —
  "Dynamical mechanisms for skeletal pattern formation in the vertebrate limb"
  (`hentschel-tgfb-fibronectin-limb`).

  Pure core Lean 4 (zero external dependencies, zero `sorry`s).
  Verify with:
    lean morphoeval/lean/loops/Loop_014_tgfb_fibronectin_limb.lean
-/

namespace MathOfBiology.Loop_014

-- ============================================================================
-- Law 1 (`hentschel-tgfb-fibronectin-limb-1`):
-- Equal-Diffusion (D_a = D_i) & Anti-Fibronectin No-Haptotaxis (chi_F = 0) Controls
-- ============================================================================

/-- Modal Jacobian trace `tr(J_k) = tr_J0 - 2 * Dq` for equal TGF-beta and
    inhibitor diffusivities `D_a = D_i = D` at scaled wavenumber `Dq = D * k^2 ≥ 0`. -/
def tgfb_modal_trace_eq_diff (tr_J0 Dq : Int) : Int :=
  tr_J0 - 2 * Dq

/-- Modal Jacobian determinant `det(J_k) = det_J0 + neg_tr_Dq + Dq_sq`
    under equal diffusivities `D_a = D_i = D`. -/
def tgfb_modal_det_eq_diff (det_J0 neg_tr_Dq Dq_sq : Int) : Int :=
  det_J0 + neg_tr_Dq + Dq_sq

/-- Linearized mesenchymal cell-density growth rate `sigma_R(q) = (chi_F * S_F - D_R) * q - r_R`
    around `R_0 = 1`: when `chi_F = 0` (anti-fibronectin / RGD haptotaxis blockade),
    `sigma_R_no_hapto(D_R_q, r_R) = -D_R_q - r_R < 0`. -/
def sigma_R_no_hapto (D_R_q r_R : Int) : Int :=
  -D_R_q - r_R

/-- Theorem 1 (`hentschel_tgfb_inhibitor_haptotaxis_stability_controls`):
    (1) When the homogeneous TGF-beta / inhibitor steady state is stable (`tr_J0 < 0`,
        `det_J0 > 0`), equal diffusivities `D_a = D_i` keep `tr(J_k) < 0` and
        `det(J_k) > 0` for all `k^2 ≥ 0` (no chemical prepattern).
    (2) When mesenchymal haptotactic sensitivity `chi_F = 0` (anti-fibronectin / RGD
        peptide control, Kiskowski et al. 2004), `sigma_R_no_hapto(D_R_q, r_R) < 0` for
        all `D_R_q ≥ 0` and `r_R > 0`, preventing mesenchymal cell condensation even in
        the presence of a prepattern. -/
theorem hentschel_tgfb_inhibitor_haptotaxis_stability_controls
    (tr_J0 det_J0 Dq neg_tr_Dq Dq_sq D_R_q r_R : Int)
    (htr : tr_J0 < 0) (hdet : det_J0 > 0)
    (hDq : Dq ≥ 0) (hterm : neg_tr_Dq ≥ 0) (hsq : Dq_sq ≥ 0)
    (hDR : D_R_q ≥ 0) (hr : r_R > 0) :
    tgfb_modal_trace_eq_diff tr_J0 Dq < 0 ∧
    tgfb_modal_det_eq_diff det_J0 neg_tr_Dq Dq_sq > 0 ∧
    sigma_R_no_hapto D_R_q r_R < 0 := by
  unfold tgfb_modal_trace_eq_diff tgfb_modal_det_eq_diff sigma_R_no_hapto
  omega


-- ============================================================================
-- Law 2 (`hentschel-tgfb-fibronectin-limb-2`):
-- AER-FGF Zonation & Proximodistal 1 -> 2 -> 3 -> 4 Skeletal Mode Transition
-- ============================================================================

/-- In Hentschel et al. (2004) Proc. R. Soc. B Eqs. (3.1)-(3.12) (`2*D_a = 1`,
    `2*D_i = 100`, `g_i = -200`), the homogeneous trace is `tr_J0(f_a) = f_a - 200`
    and the FGF-modulated critical squared wavenumber in the apical active zone is
    `m^2(f_a) = (2*D_i*f_a + 2*D_a*g_i) / (4*D_a*D_i) = f_a - 2`. -/
def hentschel_trace_J0 (fa : Int) : Int :=
  fa - 200

def hentschel_mode_sq (fa : Int) : Int :=
  fa - 2

/-- Theorem 2 (`hentschel_fgf_zonation_proximodistal_mode_transition`):
    As the distal AER-FGF signal `c_FGF(z)` increases across the proximodistal zones
    of the developing chick limb bud (`fa_1 < fa_2 < 200`), the selected anteroposterior
    squared mode number `hentschel_mode_sq(fa)` strictly increases while the homogeneous
    steady state remains stable (`hentschel_trace_J0(fa_2) < 0`), reproducing the
    discrete skeletal condensation hierarchy:
    - HH20 Stylopod (`fa = 3`)       -> `m^2 = 1^2 = 1` (1 element: Humerus)
    - HH24 Zeugopod (`fa = 6`)       -> `m^2 = 2^2 = 4` (2 elements: Radius + Ulna)
    - HH27 Wing Autopod (`fa = 11`)  -> `m^2 = 3^2 = 9` (3 elements: Wing digits II, III, IV)
    - HH28 Leg / High-TGFb (`fa=18`) -> `m^2 = 4^2 = 16` (4 elements: Pedal digits I-IV). -/
theorem hentschel_fgf_zonation_proximodistal_mode_transition (fa1 fa2 : Int)
    (h12 : fa1 < fa2) (h2 : fa2 < 200) :
    hentschel_mode_sq fa1 < hentschel_mode_sq fa2 ∧
    hentschel_trace_J0 fa2 < 0 ∧
    hentschel_mode_sq 3 = 1 * 1 ∧
    hentschel_mode_sq 6 = 2 * 2 ∧
    hentschel_mode_sq 11 = 3 * 3 ∧
    hentschel_mode_sq 18 = 4 * 4 := by
  unfold hentschel_mode_sq hentschel_trace_J0
  omega


-- ============================================================================
-- Law 3 (`hentschel-tgfb-fibronectin-limb-3`):
-- Fibronectin Haptotactic Positive-Feedback Amplification & Condensation Gain
-- ============================================================================

/-- Net haptotactic flux gain `hapto_net_flux(chi_F_SF, D_R, q) = (chi_F_SF - D_R) * q`
    for mesenchymal cell density `R` climbing extracellular fibronectin gradients `grad(F)`
    with haptotactic coupling `chi_F_SF = chi_F * (k_F / gamma_F)` against random cell
    motility `D_R`. -/
def hapto_net_flux (chi_F_SF D_R q : Int) : Int :=
  (chi_F_SF - D_R) * q

/-- Theorem 3 (`hentschel_kiskowski_fibronectin_haptotaxis_amplification`):
    Whenever fibronectin-mediated haptotactic attraction exceeds random mesenchymal
    cell diffusion (`chi_F_SF > D_R ≥ 0`) at any non-zero spatial mode (`q ≥ 1`),
    the net haptotactic condensation flux is strictly positive (`hapto_net_flux > 0`)
    and strictly exceeds the zero-haptotaxis control (`hapto_net_flux 0 D_R q ≤ 0`),
    driving high-contrast precartilage cell condensations (`R > R_0`) colocalized with
    fibronectin (`F`) and TGF-beta (`c_a`) peaks. -/
theorem hentschel_kiskowski_fibronectin_haptotaxis_amplification
    (chi_F_SF D_R q : Int)
    (hchi : chi_F_SF > D_R) (hDR : D_R ≥ 0) (hq : q ≥ 1) :
    hapto_net_flux chi_F_SF D_R q > 0 ∧
    hapto_net_flux 0 D_R q ≤ 0 ∧
    hapto_net_flux chi_F_SF D_R q > hapto_net_flux 0 D_R q := by
  unfold hapto_net_flux
  have hdiff : chi_F_SF - D_R ≥ 1 := by omega
  have hpos : (chi_F_SF - D_R) * q ≥ 1 * 1 := Int.mul_le_mul hdiff hq (by omega) (by omega)
  have hneg : (0 - D_R) * q ≤ 0 := Int.mul_nonpos_of_nonpos_of_nonneg (by omega) (by omega)
  omega


-- ============================================================================
-- Law 4 (`hentschel-tgfb-fibronectin-limb-4`, Pass 3):
-- Glazier-Graner-Hogeweg Differential-Adhesion Sorting Condition
-- ============================================================================

/-- Twice the heterotypic surface tension of the GGH cellular Potts model,
    `2 γ_ld = 2 J_ld - J_ll - J_dd` (Glazier & Graner 1993 Phys. Rev. E 47:2128), for
    R1 ("light", `J_ll`), R2 ("dark", `J_dd`) and heterotypic (`J_ld`) contact energies. -/
def ggh_hetero_surface_tension2 (J_ll J_dd J_ld : Int) : Int :=
  2 * J_ld - J_ll - J_dd

/-- Energy change of replacing two heterotypic R1|R2 contacts by one R1|R1 and one R2|R2
    contact (the elementary sorting swap in the GGH Hamiltonian
    `H = Σ J(τ_i, τ_j)(1 - δ_{σ_i σ_j}) + λ_V Σ (v_σ - V_T)^2`). -/
def ggh_sorting_swap_dH (J_ll J_dd J_ld : Int) : Int :=
  (J_ll + J_dd) - 2 * J_ld

/-- Theorem 4 (`ggh_differential_adhesion_sorting_condition`):
    Whenever the heterotypic surface tension is positive (`2 J_ld > J_ll + J_dd`), the
    elementary sorting swap strictly lowers the GGH Hamiltonian (`ggh_sorting_swap_dH < 0`), so
    R2 prechondrogenic cells sort into compact condensations surrounded by R1 mesenchyme;
    for the Graner & Glazier (1992) values `(J_ll, J_dd, J_ld) = (14, 2, 11)` used in
    `simulate_ggh_cellular_potts_condensation` the swap energy is `-6`; and under the
    no-differential-adhesion ablation `J_ll = J_dd = J_ld = J` both the surface tension and the
    swap energy vanish (no sorting drive), matching the `hapto_only` / `ablation` regimes. -/
theorem ggh_differential_adhesion_sorting_condition (J_ll J_dd J_ld J : Int)
    (hsort : 2 * J_ld > J_ll + J_dd) :
    ggh_sorting_swap_dH J_ll J_dd J_ld < 0 ∧
    ggh_hetero_surface_tension2 J_ll J_dd J_ld > 0 ∧
    ggh_sorting_swap_dH 14 2 11 = -6 ∧
    ggh_hetero_surface_tension2 14 2 11 = 6 ∧
    ggh_sorting_swap_dH J J J = 0 ∧
    ggh_hetero_surface_tension2 J J J = 0 := by
  unfold ggh_sorting_swap_dH ggh_hetero_surface_tension2
  omega


-- ============================================================================
-- Law 5 (`hentschel-tgfb-fibronectin-limb-5`, Pass 3):
-- Metropolis Haptotactic Acceptance in the Coupled GGH Cell Lattice
-- ============================================================================

/-- Total GGH copy-attempt energy change for a cell extending into a target lattice site:
    adhesion term `dH_adh`, volume-constraint term `dH_vol`, and the haptotaxis term
    `-μ_F (F(target) - F(source)) = -μ_F · dF` (Savill & Hogeweg 1997). -/
def ggh_total_dH (dH_adh dH_vol mu_F dF : Int) : Int :=
  dH_adh + dH_vol - mu_F * dF

/-- Theorem 5 (`ggh_metropolis_haptotaxis_acceptance`):
    (1) If the haptotactic energy gain of an up-gradient extension (`μ_F ≥ 1`, `dF ≥ 1`) is at
        least the adhesion-plus-volume penalty, the total `ΔH ≤ 0`, so the Metropolis rule
        `P = min(1, exp(-ΔH/T))` accepts the copy with probability 1 — mesenchymal cells climb
        fibronectin gradients deterministically once `μ_F dF` dominates.
    (2) The anti-fibronectin control `μ_F = 0` removes the haptotactic term entirely:
        `ΔH = dH_adh + dH_vol`, so acceptance is governed by adhesion and volume alone (the
        `da_only` regime), and the haptotactic term is strictly favourable whenever `μ_F dF > 0`. -/
theorem ggh_metropolis_haptotaxis_acceptance (dH_adh dH_vol mu_F dF : Int)
    (hmu : mu_F ≥ 1) (hdF : dF ≥ 1) (hpen : dH_adh + dH_vol ≤ mu_F * dF) :
    ggh_total_dH dH_adh dH_vol mu_F dF ≤ 0 ∧
    ggh_total_dH dH_adh dH_vol 0 dF = dH_adh + dH_vol ∧
    ggh_total_dH dH_adh dH_vol mu_F dF < ggh_total_dH dH_adh dH_vol 0 dF := by
  unfold ggh_total_dH
  have hpos : mu_F * dF ≥ 1 * 1 := Int.mul_le_mul hmu hdF (by omega) (by omega)
  omega

end MathOfBiology.Loop_014
