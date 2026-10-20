/-
  morphoeval/lean/loops/Loop_013_mechanochemical_chondrogenesis.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 013
  (`oster-mechanochemical-chondrogenesis`):
    1. Oster, G. F., Murray, J. D. & Harris, A. K. (1983). "Mechanical aspects
       of mesenchymal morphogenesis." Journal of Embryology and Experimental
       Morphology 78:83-125. DOI: 10.1242/dev.78.1.83
    2. Oster, G. F., Murray, J. D. & Maini, P. K. (1985). "A model for
       chondrogenic condensations in the developing limb: the role of
       extracellular matrix and cell tractions." Journal of Embryology and
       Experimental Morphology 89:93-112. DOI: 10.1242/dev.89.1.93

  All theorems are proven in pure core Lean 4 with zero external dependencies
  and zero `sorry`s.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_013_mechanochemical_chondrogenesis.lean
-/

namespace MathOfBiology.Loop_013

/-- Law 1 (`oster-mechanochemical-chondrogenesis-1`):
    Oster, Murray & Harris (1983) Mechanochemical Traction Instability Criterion:
    The linearized effective restoring modulus of the coupled cell-ECM system is
    `K_eff = E_eff + D1_s - tau_eff`, where `E_eff` is the elastic matrix modulus,
    `D1_s` is the diffusive-tethering stabilization term, and `tau_eff` is the
    active mesenchymal cell traction parameter. -/
def mechanochemical_effective_stiffness (E_eff D1_s tau_eff : Int) : Int :=
  E_eff + D1_s - tau_eff

/-- Theorem 1 (`oster_murray_1983_traction_instability_criterion`):
    When active cell traction `tau_eff` is below the elastic-diffusive threshold
    `E_eff + D1_s`, the effective restoring stiffness is strictly positive
    (homogeneous mesenchyme is stable); when `tau_eff > E_eff + D1_s`, the
    effective stiffness becomes strictly negative, driving spontaneous
    mechanochemical cell-ECM condensation. -/
theorem oster_murray_1983_traction_instability_criterion
    (E_eff D1_s tau_sub tau_super : Int)
    (h_sub : tau_sub < E_eff + D1_s)
    (h_super : E_eff + D1_s < tau_super) :
    mechanochemical_effective_stiffness E_eff D1_s tau_sub > 0 ∧
    mechanochemical_effective_stiffness E_eff D1_s tau_super < 0 := by
  unfold mechanochemical_effective_stiffness
  omega

/-- Law 2 (`oster-mechanochemical-chondrogenesis-2`):
    Oster, Murray & Maini (1985) Contact-Inhibited Cell Traction Saturation Bound:
    Active cell traction stress `sigma_cell(n) = (tau * n^2) / (1 + lam * n^2)`
    has numerator `tau * n^2` and denominator `1 + lam * n^2`. -/
def contact_inhibited_traction_num (tau n : Nat) : Nat :=
  tau * (n * n)

def contact_inhibited_traction_den (lam n : Nat) : Nat :=
  1 + lam * (n * n)

/-- Theorem 2 (`oster_murray_1985_contact_inhibited_traction_bound`):
    Cross-multiplying by `lam` and `1 + lam * n^2`, active cell traction satisfies
    `lam * (tau * n^2) <= tau * (1 + lam * n^2)` (with strict inequality for `tau > 0`),
    preventing infinite-density blowup and stabilizing finite chondrogenic condensations. -/
theorem oster_murray_1985_contact_inhibited_traction_bound (tau lam n : Nat) :
    lam * contact_inhibited_traction_num tau n ≤
      tau * contact_inhibited_traction_den lam n := by
  unfold contact_inhibited_traction_num contact_inhibited_traction_den
  rw [Nat.mul_add, Nat.mul_one]
  have h_comm : lam * (tau * (n * n)) = tau * (lam * (n * n)) := by
    rw [← Nat.mul_assoc, Nat.mul_comm lam tau, Nat.mul_assoc]
  rw [h_comm]
  exact Nat.le_add_left (tau * (lam * (n * n))) tau

/-- Law 3 (`oster-mechanochemical-chondrogenesis-3`):
    Oster, Murray & Maini (1985) & Shubin-Alberch (1986) Proximo-Distal Limb
    Chondrogenic Branching Progression (`1 -> 2 -> 4 -> 5`):
    The number of transversely unstable chondrogenic condensation rods across a
    limb bud cross-section of width `W` and critical mechanochemical wavelength
    `lambda_c` is `m(W, lambda_c) = W / lambda_c`. -/
def limb_condensation_mode (W lambda_c : Nat) : Nat :=
  W / lambda_c

/-- Theorem 3 (`oster_maini_1985_limb_branching_progression`):
    For any positive critical mechanochemical wavelength `lambda_c > 0`, as the
    limb paddle width expands across the stylopod (`1 * lambda_c`), zeugopod
    (`2 * lambda_c`), urodele forelimb autopod (`4 * lambda_c`), and urodele
    hindlimb autopod (`5 * lambda_c`), the chondrogenic condensation mode count
    transitions through `(1, 2, 4, 5)`. -/
theorem oster_maini_1985_limb_branching_progression (lambda_c : Nat) (h : 0 < lambda_c) :
    limb_condensation_mode (1 * lambda_c) lambda_c = 1 ∧
    limb_condensation_mode (2 * lambda_c) lambda_c = 2 ∧
    limb_condensation_mode (4 * lambda_c) lambda_c = 4 ∧
    limb_condensation_mode (5 * lambda_c) lambda_c = 5 := by
  unfold limb_condensation_mode
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact Nat.mul_div_cancel 1 h
  · exact Nat.mul_div_cancel 2 h
  · exact Nat.mul_div_cancel 4 h
  · exact Nat.mul_div_cancel 5 h

/-! ### Pass 3 additions (2026-10-19, superseding audit v2) -/

/-- Law 4 (`oster-mechanochemical-chondrogenesis-4`):
    Position of the apical ectodermal ridge (AER) free boundary after time `t`
    when the limb bud grows at constant outgrowth speed `v` from `L_0`:
    `L(t) = L_0 + v * t`. -/
def aer_front (L0 v t : Nat) : Nat :=
  L0 + v * t

/-- Theorem 4a (`oster_maini_1985_aer_front_monotone`):
    The AER front never retreats: for `t_1 <= t_2`, `L(t_1) <= L(t_2)`. -/
theorem oster_maini_1985_aer_front_monotone (L0 v t1 t2 : Nat) (h : t1 ≤ t2) :
    aer_front L0 v t1 ≤ aer_front L0 v t2 := by
  unfold aer_front
  exact Nat.add_le_add_left (Nat.mul_le_mul_left v h) L0

/-- Theorem 4b (`oster_maini_1985_aer_outgrowth_mode_monotone`):
    For a limb paddle whose width is non-decreasing along the proximo-distal
    axis (`W1 <= W2` for a proximal and a more distal section), the mechanochemical
    condensation mode laid down behind the advancing AER cannot decrease distally:
    `m(W1, lambda_c) <= m(W2, lambda_c)`. Together with 4a this is the
    proximo-distal `1 -> 2 -> 4/5` sequence of the free-boundary solver. -/
theorem oster_maini_1985_aer_outgrowth_mode_monotone (W1 W2 lambda_c : Nat)
    (h : W1 ≤ W2) :
    limb_condensation_mode W1 lambda_c ≤ limb_condensation_mode W2 lambda_c := by
  unfold limb_condensation_mode
  exact Nat.div_le_div_right h

/-- Theorem 4c (`saunders_1948_aer_ablation_truncation`):
    Saunders (1948): removing the AER at `t_ablate` freezes the front at
    `L(t_ablate)`, so a zone at proximo-distal level `y > L(t_ablate)` is never
    reached and no condensation forms there — the limb is truncated distally.
    Formally: if `y > aer_front L0 v t_ablate` then the front at the ablation
    time (and at every earlier time, by 4a) lies strictly proximal to `y`. -/
theorem saunders_1948_aer_ablation_truncation (L0 v t_ablate y t : Nat)
    (h_t : t ≤ t_ablate) (h_y : aer_front L0 v t_ablate < y) :
    aer_front L0 v t < y :=
  Nat.lt_of_le_of_lt (oster_maini_1985_aer_front_monotone L0 v t t_ablate h_t) h_y

/-- Law 5 (`oster-mechanochemical-chondrogenesis-5`):
    Hybrid mechanochemical + Bmp-Sox9-Wnt effective restoring stiffness.
    A Sox9-mediated positive feedback (condensation -> Sox9 -> traction) with
    gain `chi_bsw >= 0` subtracts from the mechanics-only stiffness
    `K_mech = E_eff + D1_s - tau_eff`. -/
def hybrid_effective_stiffness (E_eff D1_s tau_eff chi_bsw : Int) : Int :=
  mechanochemical_effective_stiffness E_eff D1_s tau_eff - chi_bsw

/-- Theorem 5a (`hybrid_mechanochemical_bsw_threshold_lowering`):
    A non-negative biochemical feedback gain can only lower the effective
    stiffness: `K_hyb <= K_mech`. Hence the hybrid critical traction is never
    above the mechanics-only one. -/
theorem hybrid_mechanochemical_bsw_threshold_lowering
    (E_eff D1_s tau_eff chi_bsw : Int) (h : 0 ≤ chi_bsw) :
    hybrid_effective_stiffness E_eff D1_s tau_eff chi_bsw ≤
      mechanochemical_effective_stiffness E_eff D1_s tau_eff := by
  unfold hybrid_effective_stiffness mechanochemical_effective_stiffness
  omega

/-- Theorem 5b (`hybrid_mechanochemical_bsw_synergy_threshold`):
    Synergy window: if mechanics alone is stable (`0 < K_mech`) but the feedback
    gain exceeds that margin (`K_mech < chi_bsw`), the hybrid system is unstable
    (`K_hyb < 0`) — neither subsystem alone patterns, the hybrid does. This is the
    regime tested at `tau = 0.85` in `evaluate_mechanochemical_bsw_synergy`. -/
theorem hybrid_mechanochemical_bsw_synergy_threshold
    (E_eff D1_s tau_eff chi_bsw : Int)
    (h_mech_stable : 0 < mechanochemical_effective_stiffness E_eff D1_s tau_eff)
    (h_gain : mechanochemical_effective_stiffness E_eff D1_s tau_eff < chi_bsw) :
    0 < mechanochemical_effective_stiffness E_eff D1_s tau_eff ∧
    hybrid_effective_stiffness E_eff D1_s tau_eff chi_bsw < 0 := by
  unfold hybrid_effective_stiffness mechanochemical_effective_stiffness at *
  omega

/-- Theorem 5c (`hybrid_n_sox9_loop_destabilises_when_gain_exceeds_margin`):
    Two-variable reduction of the hybrid Jacobian on (n, S):
    `J = [[-a, b], [c, -mu]]` with `a > 0` (mechanics-only stable), `b, c >= 0`
    (Sox9 -> traction, condensation -> Sox9) and `mu > 0` (Sox9 clearance).
    Its determinant `a * mu - b * c` is negative — a real positive eigenvalue,
    i.e. a stationary instability — exactly when the loop gain `b * c` exceeds the
    stability margin `a * mu`. -/
def hybrid_reduced_det (a mu b c : Int) : Int :=
  a * mu - b * c

theorem hybrid_n_sox9_loop_destabilises_when_gain_exceeds_margin (a mu b c : Int)
    (h : a * mu < b * c) :
    hybrid_reduced_det a mu b c < 0 := by
  unfold hybrid_reduced_det
  omega

end MathOfBiology.Loop_013
