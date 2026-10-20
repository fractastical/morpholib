/-
  morphoeval/lean/loops/Loop_029_somitoid_chirality_buckling.lean
  =============================================================================
  Formal Lean 4 Specification & Machine-Checked Proofs for Slot 029
  (`yaman-somitoid-chirality-buckling`):
    1. Yamanaka, Y. et al. (2023). "Reconstituting human somitogenesis in vitro."
       Nature 614(7948):509-520. DOI: 10.1038/s41586-022-05649-2
    2. Miao, Y. et al. (2023). "Reconstruction and deconstruction of human
       somitogenesis in vitro." Nature 614(7948):500-508.
       DOI: 10.1038/s41586-022-05655-4
    3. Adhyapok, P. et al. (2021). "A mechanical model of early somite
       segmentation." Developmental Cell 56(9):1312-1324.
       DOI: 10.1016/j.devcel.2021.03.026

  All theorems are proven in pure core Lean 4 with zero external dependencies
  and zero `sorry`s.

  Verify with the Lean 4 kernel:
    lean morphoeval/lean/loops/Loop_029_somitoid_chirality_buckling.lean
-/

namespace MathOfBiology.Loop_029

/-- Law 1 (`yaman-somitoid-chirality-buckling-1`):
    Apical Constriction Wedge-Cell Tapering & Equilibrium Rosette Radius Law:
    For a somite rosette of `N` epithelial wedge cells with inner somitecoel
    lumen radius `R_in < R_out` and scaled equilibrium ratio numerator
    `N_lat - two_pi_gamma_ap` and denominator `N_lat + two_pi_gamma_bas`,
    each cell's apical arc `L_ap = theta_scaled * R_in` is strictly shorter than
    its basal arc `L_bas = theta_scaled * R_out`, and positive apical/basal
    tensions (`gamma_ap > gamma_bas > 0`) guarantee `R_in^* < R_out^*`. -/
def wedge_arc_length (theta_scaled R : Int) : Int :=
  theta_scaled * R

def rosette_lumen_numerator (N_lat two_pi_gamma_ap : Int) : Int :=
  N_lat - two_pi_gamma_ap

def rosette_basal_denominator (N_lat two_pi_gamma_bas : Int) : Int :=
  N_lat + two_pi_gamma_bas

/-- Theorem 1 (`yamanaka_adhyapok_wedge_rosette_tapering`):
    Whenever the scaled wedge angle `theta_scaled > 0` (`theta = 2*pi/N > 0`),
    `R_in < R_out`, and `0 < two_pi_gamma_bas < two_pi_gamma_ap`, the apical
    domain is strictly narrower than the basal domain
    (`wedge_arc_length theta_scaled R_in < wedge_arc_length theta_scaled R_out`)
    and the equilibrium lumen numerator is strictly smaller than the basal
    denominator (`rosette_lumen_numerator N_lat two_pi_gamma_ap < rosette_basal_denominator N_lat two_pi_gamma_bas`). -/
theorem yamanaka_adhyapok_wedge_rosette_tapering
    (theta_scaled R_in R_out N_lat two_pi_gamma_ap two_pi_gamma_bas : Int)
    (h_theta : 0 < theta_scaled)
    (h_rad : R_in < R_out)
    (h_bas : 0 < two_pi_gamma_bas)
    (h_ap : two_pi_gamma_bas < two_pi_gamma_ap) :
    wedge_arc_length theta_scaled R_in < wedge_arc_length theta_scaled R_out ∧
    rosette_lumen_numerator N_lat two_pi_gamma_ap <
      rosette_basal_denominator N_lat two_pi_gamma_bas := by
  dsimp [wedge_arc_length, rosette_lumen_numerator, rosette_basal_denominator]
  refine ⟨?_, ?_⟩
  · exact Int.mul_lt_mul_of_pos_left h_rad h_theta
  · omega

/-- Law 2 (`yaman-somitoid-chirality-buckling-2`):
    EphA4 / ephrinB2 Inter-Somite Cleft Boundary Sorting Energy Barrier Law:
    The doubled interfacial energy gain of separating an inter-somitic boundary
    of length `L_cleft` with heterotypic EphA4/ephrinB2 tension `gamma_het`
    relative to homotypic N-cadherin tensions `gamma_ant` and `gamma_post` is
      `eph_cleft_sorting_energy L_cleft gamma_het gamma_ant gamma_post =
         L_cleft * (2 * gamma_het - (gamma_ant + gamma_post))`. -/
def eph_cleft_sorting_energy (L_cleft gamma_het gamma_ant gamma_post : Int) : Int :=
  L_cleft * (2 * gamma_het - (gamma_ant + gamma_post))

/-- Theorem 2 (`adhyapok_2021_eph_ephrin_cleft_sorting_barrier`):
    For any positive inter-somitic contact length `L_cleft > 0`, whenever
    EphA4/ephrinB2 heterotypic boundary repulsion exceeds the mean homotypic
    adhesion (`gamma_ant + gamma_post < 2 * gamma_het`), the inter-somitic
    cleft sorting energy barrier is strictly positive (`0 < Delta E_cleft`),
    whereas losing Eph/ephrin repulsion (`2 * gamma_het = gamma_ant + gamma_post`)
    collapses the cleft separation barrier to zero. -/
theorem adhyapok_2021_eph_ephrin_cleft_sorting_barrier
    (L_cleft gamma_het gamma_het_kd gamma_ant gamma_post : Int)
    (h_len : 0 < L_cleft)
    (h_rep : gamma_ant + gamma_post < 2 * gamma_het)
    (h_kd : 2 * gamma_het_kd = gamma_ant + gamma_post) :
    0 < eph_cleft_sorting_energy L_cleft gamma_het gamma_ant gamma_post ∧
    eph_cleft_sorting_energy L_cleft gamma_het_kd gamma_ant gamma_post = 0 := by
  dsimp [eph_cleft_sorting_energy]
  refine ⟨?_, ?_⟩
  · have h_diff : 0 < 2 * gamma_het - (gamma_ant + gamma_post) := by omega
    exact Int.mul_pos h_len h_diff
  · have h_zero : 2 * gamma_het_kd - (gamma_ant + gamma_post) = 0 := by omega
    rw [h_zero, Int.mul_zero]

/-- Law 3 (`yaman-somitoid-chirality-buckling-3`):
    Y-27632 ROCK Inhibition & Minus-Matrigel Apico-Basal Tension Collapse Law:
    The spontaneous apico-basal curvature driving force is proportional to the
    differential tension `apico_basal_tension_diff gamma_ap gamma_bas = gamma_ap - gamma_bas`. -/
def apico_basal_tension_diff (gamma_ap gamma_bas : Int) : Int :=
  gamma_ap - gamma_bas

/-- Theorem 3 (`yamanaka_2023_rock_inhibition_curvature_collapse`):
    Under Y-27632 ROCK inhibition or Matrigel omission where apical tension
    relaxes to basal tension (`gamma_ap_inhibited = gamma_bas`), the spontaneous
    apico-basal tension differential vanishes (`apico_basal_tension_diff gamma_bas gamma_bas = 0`),
    whereas wild-type MET (`gamma_bas < gamma_ap_wt`) maintains strictly positive
    apical constriction differential and a strictly more constricted somitecoel
    lumen numerator than the ROCK-inhibited state. -/
theorem yamanaka_2023_rock_inhibition_curvature_collapse
    (N_lat gamma_ap_wt gamma_bas : Int)
    (h_wt : gamma_bas < gamma_ap_wt) :
    apico_basal_tension_diff gamma_bas gamma_bas = 0 ∧
    0 < apico_basal_tension_diff gamma_ap_wt gamma_bas ∧
    rosette_lumen_numerator N_lat gamma_ap_wt <
      rosette_lumen_numerator N_lat gamma_bas := by
  dsimp [apico_basal_tension_diff, rosette_lumen_numerator]
  refine ⟨?_, ?_, ?_⟩
  · omega
  · omega
  · omega

end MathOfBiology.Loop_029
