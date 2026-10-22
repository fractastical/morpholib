# Pre-registered blind predictions: Thompson 1917 phylogenetic prediction engine

Registered 2026-10-22 at commit `7889d9396de0` by [`thompson/preregister_blind_predictions.py`](../thompson/preregister_blind_predictions.py). Predictions: [`thompson_blind_predictions_registered.json`](../thompson_blind_predictions_registered.json) (SHA-256 `2e41cb1aaf1b21035f7d4970178861ee0ad6698a5b31ba35d8f1b87bca03b38b`), [`thompson_blind_predictions_registered.csv`](../thompson_blind_predictions_registered.csv) (SHA-256 `1d50205ffc2ed9954e707eba2e3db6aaa4871620df6cecf9e4cf766fe3046171`).

## What is predicted

Body-proportion ratios and the full 17-landmark configuration for **10,888 species** on the Rabosky et al. (2018) tree that are not among the 743 landmarked training species (nor the 7 digitised 1917 species). 366 targets lie outside the sampled crown clade and receive the root-state prediction. Models: Thompson-7 = Pagel lambda=0.959; full-30 = Pagel lambda=0.924; 400 Monte Carlo draws per species for 95% intervals (seed 1917).

| tau bin (Myr) | 0-5 | 5-10 | 10-25 | 25-50 | 50-80 | 80-120 | 120-1000 |
|---|---|---|---|---|---|---|---|
| n targets | 145 | 135 | 655 | 1011 | 2010 | 2323 | 4609 |

## Ratio definitions

- **BDmax_over_SL**: maximum body depth excluding fin webs / standard length. Landmark proxy: vertical extent of the 17 landmarks / (snout to caudal-fin insertion midpoint). PRIMARY.
- **CPD_over_SL**: caudal-peduncle depth / standard length. Landmark proxy: distance between dorsal (LM5) and ventral (LM6) caudal-fin insertions / SL. SECONDARY (the dataset's 'minimum' peduncle depth will be a little smaller than the insertion-to-insertion distance).
- **HL_over_SL**: head length / standard length. Landmark proxy: snout (mean LM1-2) to opercle landmark (LM11) / SL. EXPLORATORY (only if the dataset reports head length).
- **PD_over_SL**: predorsal length / standard length. Landmark proxy: snout to dorsal-fin origin (LM3), straight line / SL. EXPLORATORY.

## Hypotheses (fixed before any comparison data were opened)

- **H1** (primary): For target species with a training relative within 50 Myr (tau < 50), the full 30-D phylogenetic prediction of BDmax/SL has smaller absolute error than the grand-consensus value. Test: one-sided paired Wilcoxon signed-rank on |error| across species; also median |error| ratio. Expected: full30 median |error| < 0.8 x consensus median |error|.
- **H2** (primary): For tau < 50, the full 30-D prediction beats copying the nearest training relative's observed landmark ratio (same test). Expected margin smaller than H1; this held at every horizon < 50 Myr in the internal hold-out.
- **H3** (primary): After the pre-specified measurement-convention calibration (Section 'Calibration'), the 95% predictive intervals of BDmax/SL for tau < 50 contain the observed species value in 85-97% of species. Raw (uncalibrated) coverage is also reported.
- **H4** (primary): For tau >= 50, the full 30-D prediction does NOT beat the consensus by more than 10% in median |error| (the ~50 Myr horizon found internally and in the Thompson-pair analysis).
- **H5**: Median |error| of the full 30-D prediction increases monotonically across the tau bins [0,5), [5,10), [10,25), [25,50), [50,80), [80,120), [120,inf) (Spearman rho > 0 on bin medians; at least 5 of 6 consecutive steps non-decreasing).
- **H6**: The a-priori 7-Thompson-mode prediction is worse than the full 30-D prediction (median |error| larger) but still beats the consensus for tau < 50.
- **H7**: Secondary ratio CPD/SL: full 30-D beats consensus for tau < 50 (as H1). Exploratory HL/SL, PD/SL likewise if present in the dataset.

## Protocol

- **comparison dataset**: FishShapes v1 (Price, Larouche, Friedman, Corn, Wainwright & Martinez 2022, Ecology 103:e3829; Dryad). Linear measurements of museum specimens for ~6,000 teleost species. Never downloaded, opened, or used in this repository before the registration commit; the FishBase morphometric tables already in .thompson_cache/ are NOT the comparison dataset.
- **fallback dataset**: If FishShapes cannot be obtained, any independent published dataset of standard length and maximum body depth per species not previously used here, declared before opening it.
- **species matching**: Exact match of 'Genus_species' after normalising case, whitespace/underscore and stripping subspecies/author strings; no fuzzy matching. Synonyms are not resolved (a conservative choice that only loses power).
- **per species value**: Median across specimens of (measurement / standard length) computed per specimen. Specimens with missing SL or measurement are dropped; species need >= 1 specimen.
- **exclusions**: The 743 training species and the 7 digitised 1917 museum species are excluded from scoring (they may be used only for the convention calibration below). Species not on the Rabosky tree cannot be predicted and are excluded.
- **calibration**: Landmark-based ratios differ systematically from calliper ratios (e.g. landmark BDmax is the vertical landmark extent). A single linear map obs = alpha + beta * landmark_ratio is fitted by least squares on the TRAINING species present in the comparison dataset, using their OBSERVED landmark ratios (not predictions). It is applied to predicted point estimates and interval bounds of every method (prediction, consensus, nearest-relative copy, family mean alike), so it cannot favour one method. Raw results are reported alongside.
- **horizon tau**: tau = age (Myr) of the MRCA between the target and its nearest training species on the Rabosky tree, computed from the tree alone, before any data are seen. Species outside the sampled crown clade get tau = crown age and the root-state prediction.
- **error metric**: absolute error |predicted ratio - observed species ratio|; summarised by median within tau bins and overall; paired one-sided Wilcoxon signed-rank tests, alpha = 0.01 for primary hypotheses, with n reported.
- **no tuning**: No model, basis, bin edge, exclusion or metric may be changed after the comparison data are opened. Anything learned afterwards is exploratory and will be labelled as such in the results document.

## Status

Step 1 (this document) is complete once the files above are committed and their hashes are recorded in a nanopublication. Step 2 is `thompson/score_blind_predictions.py`, which verifies the hashes before reading any comparison data and writes `docs/THOMPSON_BLIND_TEST_RESULTS.md`.
