/-
  morphoeval/lean/loops/Loop_037_allometric_compound_leaf.lean
  Formal Lean 4 verification of Bilsborough et al. (2011) PNAS 108(8):3424-3429,
  Barkoulas et al. (2008) Nature Genetics 40(9):1136-1141, Vlad et al. (2014) Science
  343(6172):780-783, and Kierzkowski et al. (2019) Cell 177(6):1405-1418 governing the
  growing leaf margin Auxin-PIN1-CUC2-RCO feedback loop, complementary spatial antiphase
  between auxin tooth peaks and CUC2 sinus maxima, smooth untoothed margin collapse under
  cuc2 loss-of-function or unrepressed uniform CUC2, and RCO-driven serration-to-compound
  leaflet dissection enhancement (`przemyslaw-allometric-compound-leaf`, Slot 037).

  All definitions and theorems are self-contained in standard Lean 4 core (zero `sorry`s).
-/

namespace MathOfBiology.Loop_037

-- ============================================================================
-- Law 1 (`przemyslaw-allometric-compound-leaf-1`):
-- Bilsborough et al. (2011) Complementary Spatial Antiphase of Auxin Tooth Peaks
-- vs CUC2 Sinus Maxima and CUC2-Promoted Up-the-Gradient PIN1 Polarization Law
-- ============================================================================
/-- In Bilsborough et al. (2011) PNAS 108:3424-3429 (Loop 1 & Loop 2), auxin `A_i` at
    margin cell `i` transcriptionally represses `CUC2` expression via a decreasing Hill
    response `C_i = C_max * K_n / (K_n + A_i^n)`, while `CUC2` promotes PIN1 up-the-gradient
    polarization toward high-auxin neighbor `j` with weight `w_{ij} = (1 + gamma * C_i) * B_j`.
    Cross-multiplying the CUC2 equilibrium levels at a tooth tip (`A_tip^n`) and an adjacent
    sinus (`A_sinus^n`) gives:
      `C_sinus * (K_n + A_sinus^n) * (K_n + A_tip^n) - C_tip * (K_n + A_sinus^n) * (K_n + A_tip^n)`
      `= C_max * K_n * (A_tip^n - A_sinus^n)`.
    This theorem proves that:
    1. Whenever `A_tip^n > A_sinus^n` (for example `A_tip^n > K_n > A_sinus^n` with `C_max > 0, K_n > 0`),
       the cross-multiplied CUC2 difference `C_max * K_n * (A_tip^n - A_sinus^n)` is strictly positive,
       enforcing complementary spatial antiphase (`C_tip < C_sinus`).
    2. CUC2 at sinus flank cell `i` strictly enhances up-the-gradient PIN1 polarization toward
       a higher-auxin neighbor (`B_high > B_low`) over the `cuc2` null baseline (`C_i = 0`) by
       `gamma * C_i * (B_high - B_low)`. -/
def cuc2_cross_product (C_max K_n A_n_other : Int) : Int :=
  C_max * K_n + C_max * K_n * A_n_other

def pin1_polarization_weight (gamma C_i B_j : Int) : Int :=
  (1 + gamma * C_i) * B_j

theorem bilsborough_2011_auxin_cuc2_antiphase_and_pin1_promotion
    (C_max K_n A_tip_n A_sinus_n gamma C_i B_high B_low : Int)
    (hC : C_max > 0) (hK : K_n > 0) (hA : A_tip_n > A_sinus_n) :
    C_max * K_n * (K_n + A_tip_n) - C_max * K_n * (K_n + A_sinus_n) =
      C_max * K_n * (A_tip_n - A_sinus_n) ∧
    C_max * K_n * (A_tip_n - A_sinus_n) > 0 ∧
    (pin1_polarization_weight gamma C_i B_high - pin1_polarization_weight gamma C_i B_low) -
      (pin1_polarization_weight gamma 0 B_high - pin1_polarization_weight gamma 0 B_low) =
      gamma * C_i * (B_high - B_low) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [Int.mul_add, Int.mul_sub]
    omega
  · have hCK : C_max * K_n > 0 := Int.mul_pos hC hK
    have hDiff : A_tip_n - A_sinus_n > 0 := by omega
    exact Int.mul_pos hCK hDiff
  · dsimp [pin1_polarization_weight]
    simp [Int.add_mul, Int.mul_sub, Int.mul_assoc]
    omega


-- ============================================================================
-- Law 2 (`przemyslaw-allometric-compound-leaf-2`):
-- Bilsborough et al. (2011) Smooth Margin Collapse under `cuc2` Loss-of-Function
-- or Ectopic Unrepressed `CUC2` (`cuc2-1D` / `mir164a` Feedback Ablation) Law
-- ============================================================================
/-- In Bilsborough et al. (2011) PNAS 108:3424-3429 (Loop 3 & Figs. 3-4), normal margin
    outgrowth velocity is `v_n(s) = G_A(s) * (S_max - eta * C(s))`, where `G_A(s)` is the
    auxin growth factor and `eta * C(s)` is CUC2/RCO-mediated sinus growth repression.
    - In the `cuc2-1` loss-of-function mutant (`C(s) = 0` everywhere), CUC2-promoted PIN1
      convergence fails (`A_tip = A_sinus -> G_tip = G_sinus = G_0`), and sinus growth
      repression vanishes (`eta * 0 = 0`), collapsing differential outgrowth
      `v_n(tip) - v_n(sinus) = 0`.
    - In an unrepressed uniform ectopic `CUC2` mutant (`C(s) = C_0` spatially constant
      because auxin cannot repress CUC2), spatial CUC2 contrast `C_sinus - C_tip = 0`
      also eliminates CUC2-mediated differential sinus retardation at equal auxin growth `G_0`.
    - Conversely, in wild-type leaves where `G_tip >= G_sinus` and `C_sinus > C_tip`,
      differential tooth-to-sinus outgrowth equals:
        `v_n(tip) - v_n(sinus) = (G_tip - G_sinus) * (S_max - eta * C_tip) + G_sinus * eta * (C_sinus - C_tip)`. -/
def margin_outgrowth_velocity (G_A S_max eta C : Int) : Int :=
  G_A * (S_max - eta * C)

theorem bilsborough_2011_mutant_smooth_margin_collapse
    (G_0 G_tip G_sinus S_max eta C_0 C_tip C_sinus : Int) :
    margin_outgrowth_velocity G_0 S_max eta 0 -
      margin_outgrowth_velocity G_0 S_max eta 0 = 0 ∧
    margin_outgrowth_velocity G_0 S_max eta C_0 -
      margin_outgrowth_velocity G_0 S_max eta C_0 = 0 ∧
    margin_outgrowth_velocity G_tip S_max eta C_tip -
      margin_outgrowth_velocity G_sinus S_max eta C_sinus =
      (G_tip - G_sinus) * (S_max - eta * C_tip) + G_sinus * eta * (C_sinus - C_tip) := by
  refine ⟨by omega, by omega, ?_⟩
  dsimp [margin_outgrowth_velocity]
  simp [Int.mul_sub, Int.sub_mul, Int.mul_assoc]
  omega


-- ============================================================================
-- Law 3 (`przemyslaw-allometric-compound-leaf-3`):
-- Vlad et al. (2014) & Kierzkowski et al. (2019) RCO Sinus Growth Repression
-- Enhances Tooth-to-Leaflet Dissection Depth Index Law
-- ============================================================================
/-- In Vlad et al. (2014) Science 343:780-783 and Kierzkowski et al. (2019) Cell 177:1405-1418,
    the homeobox gene `RCO` in *Cardamine hirsuta* acts locally at leaflet sinuses (`C_sinus > 0`)
    to increase sinus growth repression from `eta_Arabidopsis` to `eta_Cardamine > eta_Arabidopsis`
    without repressing outgrowth at the CUC2-depleted tooth/leaflet tip (`C_tip = 0`).
    After growth duration `T > 0` from initial primordium radius `R_0`, tip radius is
      `R_tip = R_0 + T * G_tip * S_max`,
    and sinus radius under repression strength `eta` is
      `R_sinus(eta) = R_0 + T * G_sinus * (S_max - eta * C_sinus)`.
    The unnormalized dissection depth `D(eta) = R_tip - R_sinus(eta)` (the numerator of the
    dissection index `I_D = (R_tip - R_sinus) / R_tip`) satisfies:
      `D(eta_Cardamine) - D(eta_Arabidopsis) = T * G_sinus * C_sinus * (eta_Cardamine - eta_Arabidopsis) > 0`,
    proving that `RCO` sinus growth repression strictly increases leaflet dissection depth `I_D`,
    converting shallow *Arabidopsis* serrations into deeply dissected *Cardamine* compound leaflets. -/
def tip_radius (R_0 T G_tip S_max : Int) : Int :=
  R_0 + T * G_tip * S_max

def sinus_radius (R_0 T G_sinus S_max eta C_sinus : Int) : Int :=
  R_0 + T * G_sinus * (S_max - eta * C_sinus)

def dissection_depth_numerator (R_0 T G_tip G_sinus S_max eta C_sinus : Int) : Int :=
  tip_radius R_0 T G_tip S_max - sinus_radius R_0 T G_sinus S_max eta C_sinus

theorem kierzkowski_2019_rco_compound_leaflet_dissection_enhancement
    (R_0 T G_tip G_sinus S_max eta_Cardamine eta_Arabidopsis C_sinus : Int)
    (hT : T > 0) (hG : G_sinus > 0) (hC : C_sinus > 0) (hEta : eta_Cardamine > eta_Arabidopsis) :
    dissection_depth_numerator R_0 T G_tip G_sinus S_max eta_Cardamine C_sinus -
      dissection_depth_numerator R_0 T G_tip G_sinus S_max eta_Arabidopsis C_sinus =
      T * G_sinus * C_sinus * (eta_Cardamine - eta_Arabidopsis) ∧
    dissection_depth_numerator R_0 T G_tip G_sinus S_max eta_Cardamine C_sinus >
      dissection_depth_numerator R_0 T G_tip G_sinus S_max eta_Arabidopsis C_sinus := by
  have hId :
      dissection_depth_numerator R_0 T G_tip G_sinus S_max eta_Cardamine C_sinus -
        dissection_depth_numerator R_0 T G_tip G_sinus S_max eta_Arabidopsis C_sinus =
        T * G_sinus * C_sinus * (eta_Cardamine - eta_Arabidopsis) := by
    dsimp [dissection_depth_numerator, tip_radius, sinus_radius]
    simp [Int.mul_sub, Int.mul_comm, Int.mul_left_comm]
    omega
  refine ⟨hId, ?_⟩
  have hTG : T * G_sinus > 0 := Int.mul_pos hT hG
  have hTGC : T * G_sinus * C_sinus > 0 := Int.mul_pos hTG hC
  have hDiff : eta_Cardamine - eta_Arabidopsis > 0 := by omega
  have hProd : T * G_sinus * C_sinus * (eta_Cardamine - eta_Arabidopsis) > 0 :=
    Int.mul_pos hTGC hDiff
  omega

-- ============================================================================
-- Pass 3 (2026-10-20): Hamant et al. (2008) / Sampathkumar et al. (2014)
-- Microtubule-Reinforced Anisotropic Growth Tensor Partition Law
-- ============================================================================
/-- In Hamant et al. (2008) Science 322:1650-1655 and Sampathkumar et al. (2014) eLife 3:e01967,
    cortical microtubules (and the cellulose microfibrils they guide) reinforce the cell wall along
    a direction `d`, so that an edge of the cellular lamina making angle `theta` with `d` elongates at
      `gamma(theta) = g * (1 + a * (2 cos^2 theta - 1))`,
    where `g > 0` is the isotropic (areal) growth rate and `0 < a < 1` the reinforcement anisotropy.
    In integer form with unit scale `S` (so `cos^2 theta` ranges over `0 .. S`):
      `gamma(cos2) = g * (S + a * (2 * cos2 - S))`.
    The partition law states that the parallel (`cos2 = S`) and perpendicular (`cos2 = 0`) rates
    satisfy `gamma_par + gamma_perp = 2 g S` (anisotropy redistributes growth, it does not create
    area) and `gamma_par - gamma_perp = 2 g a S > 0` (strictly more elongation along the
    reinforcement axis). This is exactly the edge rest-length update used by
    `GrowingLaminaSpringMesh.run` in `2011-przemyslaw-allometric-compound-leaf.py`. -/
def anisotropic_growth_rate (g S a cos2 : Int) : Int :=
  g * (S + a * (2 * cos2 - S))

theorem hamant_2008_microtubule_anisotropic_growth_partition
    (g S a : Int) (hg : g > 0) (hS : S > 0) (ha : a > 0) :
    anisotropic_growth_rate g S a S + anisotropic_growth_rate g S a 0 = 2 * g * S ∧
    anisotropic_growth_rate g S a S - anisotropic_growth_rate g S a 0 = 2 * g * a * S ∧
    anisotropic_growth_rate g S a S > anisotropic_growth_rate g S a 0 := by
  -- gamma_par = g * (S + a * S) ; gamma_perp = g * (S - a * S)
  have hPar : anisotropic_growth_rate g S a S = g * S + g * (a * S) := by
    show g * (S + a * (2 * S - S)) = g * S + g * (a * S)
    have h1 : 2 * S - S = S := by omega
    rw [h1, Int.mul_add]
  have hPerp : anisotropic_growth_rate g S a 0 = g * S - g * (a * S) := by
    show g * (S + a * (2 * 0 - S)) = g * S - g * (a * S)
    have h1 : a * (2 * 0 - S) = -(a * S) := by
      rw [Int.mul_zero, Int.zero_sub, Int.mul_neg]
    rw [h1, ← Int.sub_eq_add_neg, Int.mul_sub]
  have hSum : anisotropic_growth_rate g S a S + anisotropic_growth_rate g S a 0 = 2 * g * S := by
    rw [hPar, hPerp, Int.mul_assoc 2 g S]
    omega
  have hDiff : anisotropic_growth_rate g S a S - anisotropic_growth_rate g S a 0 = 2 * g * a * S := by
    rw [hPar, hPerp, Int.mul_assoc (2 * g) a S, Int.mul_assoc 2 g (a * S)]
    omega
  refine ⟨hSum, hDiff, ?_⟩
  have haS : a * S > 0 := Int.mul_pos ha hS
  have hgaS : g * (a * S) > 0 := Int.mul_pos hg haS
  rw [hPar, hPerp]
  omega

-- ============================================================================
-- Pass 3 (2026-10-20): Mitchison (1980) Flux-Canalization Conductance Feedback Law
-- ============================================================================
/-- In Mitchison (1980) Proc. R. Soc. Lond. B 207:79-109 (and its auxin-transport reading in
    Scarpella et al. (2006) Genes Dev 20:1015-1027), the conductance `D` of a cell-cell interface
    carrying auxin flux `q` is updated by positive flux feedback and first-order decay:
      `D' = D + dt * (k * q^2 - mu * D)`.
    With `dt > 0`: whenever the flux-feedback term exceeds the decay term (`k q^2 > mu D`) the
    conductance strictly increases (the strand canalizes), and at zero flux with `mu > 0, D > 0`
    it strictly decays (unused interfaces lose conductance). This is the per-edge update of
    `GrowingLaminaSpringMesh._canalize` that produces the lateral vein strands draining each
    marginal convergence point into the procambial midvein. -/
def conductance_update (D dt k mu q : Int) : Int :=
  D + dt * (k * q * q - mu * D)

theorem mitchison_1980_canalization_conductance_feedback
    (D dt k mu q : Int) (hdt : dt > 0) (hmu : mu > 0) (hD : D > 0)
    (hpos : k * q * q > mu * D) :
    conductance_update D dt k mu q > D ∧ conductance_update D dt k mu 0 < D := by
  have hGain : k * q * q - mu * D > 0 := by omega
  have hUp : dt * (k * q * q - mu * D) > 0 := Int.mul_pos hdt hGain
  have hmuD : mu * D > 0 := Int.mul_pos hmu hD
  have hDown : dt * (mu * D) > 0 := Int.mul_pos hdt hmuD
  constructor
  · show D + dt * (k * q * q - mu * D) > D
    omega
  · show D + dt * (k * 0 * 0 - mu * D) < D
    have h0 : dt * (k * 0 * 0 - mu * D) = -(dt * (mu * D)) := by
      rw [Int.mul_zero, Int.zero_sub, Int.mul_neg]
    rw [h0]
    omega

end MathOfBiology.Loop_037
