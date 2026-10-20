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

end MathOfBiology.Loop_013
