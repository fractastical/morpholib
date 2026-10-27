/-
  morphoeval/lean/loops/Loop_prediction_DM_DPP_01_dpp_mobility_ratio.lean
  =============================================================================
  Machine-checked certificate for prediction DM-DPP-01 (Drosophila melanogaster Dpp).

  Source data (frozen in-repo, not edited by this loop):
    sequences/sequence_links.json → drosophila_melanogaster / dpp / UniProt P07713
      precursor_length_aa = 588
      protein_state.chains[0]: start 457, end 588, length_aa 132 (mature chain)

  Registered quantitative expectation (predictions/prediction_registry.json):
    D_precursor/D_mature = (132/588)^(1/3) ≈ 0.61 (size prior D ∝ MW^(-1/3))
    sqrt(D/k) length-scale factor ≈ 0.78 = (132/588)^(1/6)

  Proof strategy: exact Nat inequalities for cube-root and sixth-root bounds on
  rationals, checked by `decide` (complete proofs, no external libraries).

  Verify:
    lean morphoeval/lean/loops/Loop_prediction_DM_DPP_01_dpp_mobility_ratio.lean
-/

namespace MorphoEval.Prediction.DM_DPP_01

/-- Mature Dpp chain length from UniProt feature 457–588 inclusive (P07713). -/
def dm_dpp_01_mature_length_aa : Nat := 588 - 457 + 1

/-- Full precursor length (P07713). -/
def dm_dpp_01_precursor_length_aa : Nat := 588

theorem dm_dpp_01_mature_chain_length :
    dm_dpp_01_mature_length_aa = 132 := by
  decide

theorem dm_dpp_01_precursor_length :
    dm_dpp_01_precursor_length_aa = 588 := by
  decide

theorem dm_dpp_01_mature_shorter_than_precursor :
    dm_dpp_01_mature_length_aa < dm_dpp_01_precursor_length_aa := by
  decide

/-!
  Cube-root bounds for mobility ratio R = (132/588)^(1/3) = D_precursor/D_mature
  under the registered size prior (smaller species diffuses faster).

  For positive rationals, R > a/b  ↔  132/b³ > 588/a³  ↔  132·b³ > 588·a³ (with scale b=1000).
-/

/-- DM-DPP-01: D_precursor/D_mature exceeds 607/1000 (strict lower bound). -/
theorem dm_dpp_01_mobility_ratio_gt_607_per_1000 :
    588 * 607 ^ 3 < 132 * 1000 ^ 3 := by
  decide

/-- DM-DPP-01: D_precursor/D_mature is below 608/1000 (strict upper bound). -/
theorem dm_dpp_01_mobility_ratio_lt_608_per_1000 :
    132 * 1000 ^ 3 < 588 * 608 ^ 3 := by
  decide

/-- DM-DPP-01: bracket the registered nominal 0.61 (607/1000 < R < 610/1000). -/
theorem dm_dpp_01_mobility_ratio_lt_610_per_1000 :
    132 * 1000 ^ 3 < 588 * 610 ^ 3 := by
  decide

/-!
  Sixth-root bounds for morphogen length scale factor √(D/k) ratio:
  L = (132/588)^(1/6) (registered ≈ 0.78).
-/

/-- DM-DPP-01: length-scale ratio exceeds 779/1000. -/
theorem dm_dpp_01_length_scale_ratio_gt_779_per_1000 :
    588 * 779 ^ 6 < 132 * 1000 ^ 6 := by
  decide

/-- DM-DPP-01: length-scale ratio is at most 780/1000 (registered value). -/
theorem dm_dpp_01_length_scale_ratio_le_780_per_1000 :
    132 * 1000 ^ 6 ≤ 588 * 780 ^ 6 := by
  decide

end MorphoEval.Prediction.DM_DPP_01
