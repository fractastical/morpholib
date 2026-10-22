# Thompson 1917 prediction engine: dataset portfolio and forward confidence

Registered 2026-10-22 at commit `b63b56f22851` by `thompson/register_dataset_portfolio.py`. Anchor: the FishShapes v1 blind test (preregistration nanopub `RAilN4nbRHp3e3ZHqlbIgeL3UGPRV_KtQHtJD3gzVVdqw`, outcome `RAJancwGOPEkHzBoZXGv_vu_snW0Lff2FktE85Me1P-fI`).

This document states, before the comparisons are made, which datasets the engine should and should not have predictive power on, and how confident we are. **Simulated** probabilities come from resampling the FishShapes per-species errors in matching horizon bins (tau = time to nearest training species) for the species actually in scope on the Rabosky tree, then re-running the registered tests (2000 resamples, seed 1917). **Subjective** probabilities are exactly that; the reasoning is given beside each. Realised columns will be filled in by the scoring scripts; the registered probabilities are never edited.

## 1. Ray-finned fishes (the trained engine; predictions already locked at `dac08493`)

| Dataset | Class | Status | Species in scope / used | Within 50 Myr | Pred. error ratio vs consensus (<50 Myr) | vs nearest | 95% coverage | P(beats consensus) | P(beats nearest) | P(coverage 85-97%) | Realised |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FishShapes v1 (Price et al. 2022) - scored | A linear calliper traits, same clade | scored 2026-10-22 (anchor for all simulated confidences) | 5180 | 22% | 0.41 | 0.96 | 93% | - | - | - | H1 pass, H2 fail, H3 pass; far ratio 0.52 |
| FishBase picture-based morphometrics (MORPHMET table) | A linear traits, same clade, different measurement protocol | in hand: cached in .thompson_cache/fishbase_21_06 for unrelated analyses (run_three_dataset_validation.py); never compared with the engine; predictions locked at dac08493 before any comparison -> semi-blind | 5494 / 5494 | 27% | 0.41 [0.38, 0.44] | 0.95 [0.91, 1.01] | 93% [91, 94] | 90% | 39% | 90% | pending |
| Lake Tanganyika cichlid body-shape landmarks (Ronco et al. 2021) | B landmark geometric morphometrics from another lab, same clade, deep horizon | not downloaded | 750 / 150 | 0% | - | - | - | - | - | - | pending |
| Characiformes body-shape landmarks (Burns & Sidlauskas 2019) | C landmark data, same engine, beyond the horizon (Otophysi; tau > 120 Myr for essentially every species) | not downloaded | 485 / 500 | 0% | - | - | - | - | - | - | pending |

Deep-horizon profile (species >= 50 Myr from any training species), simulated. The P3 pass rule is that the realised ratio and coverage both land inside the widened intervals:

| Dataset | n >= 50 Myr (of used) | Pred. error ratio vs consensus | Pred. 95% coverage | P3 pass interval (ratio; coverage) | P(pass), subjective |
|---|---|---|---|---|---|
| FishBase picture-based morphometrics (MORPHMET table) | 4034 | 0.53 [0.51, 0.55] | 74% [73, 75] | 0.41-0.65; 63-85% | 60% |
| Lake Tanganyika cichlid body-shape landmarks (Ronco et al. 2021) | 150 | 0.77 [0.66, 0.88] | 68% [62, 74] | 0.56-0.98; 52-84% | 60% |
| Characiformes body-shape landmarks (Burns & Sidlauskas 2019) | 500 | 0.43 [0.40, 0.46] | 82% [80, 85] | 0.30-0.56; 70-95% | 60% |

Per-trait probabilities (trait factor applied; HL and PD carry an extra 0.8/0.9 because they have no empirical anchor):

| Dataset | Trait | P(beats consensus < 50 Myr) | P(H2 passes) | P(coverage calibrated) |
|---|---|---|---|---|
| fishbase_morphmet | BDmax_over_SL (BD/SL) | 90% | 39% | 90% |
| fishbase_morphmet | CPD_over_SL (CH/SL) | 90% | 39% | 90% |
| fishbase_morphmet | HL_over_SL (HL/SL) | 80% | 34% | 80% |
| fishbase_morphmet | PD_over_SL (PDL/SL) | 80% | 34% | 80% |
| ronco2021_tanganyika_cichlids | BDmax_over_SL (from landmark coordinates) | - | - | - |
| ronco2021_tanganyika_cichlids | HL_over_SL (from landmark coordinates) | - | - | - |
| burns_sidlauskas2019_characiformes | BDmax_over_SL (from landmark coordinates) | - | - | - |
| burns_sidlauskas2019_characiformes | PD_over_SL (from landmark coordinates) | - | - | - |

Fish datasets without a simulated anchor (subjective):

| Dataset | Class | Status | n | Prediction | P | Reasoning |
|---|---|---|---|---|---|---|
| FishBase BodyShapeI categorical body-shape class | F categorical trait derived from shape | in hand (cached); never compared with the engine | 8700 | ordinal class predicted from locked BDmax/SL with thresholds fitted on training species only; accuracy above the majority-class baseline (43% 'elongated') within 50 Myr | 80% | coarse 3-4 level class; the BDmax error of 0.038 within 50 Myr is far smaller than the class width (~0.1 BDmax/SL); FishBase classes are inconsistently assigned, which caps accuracy |
| FishBase caudal-fin aspect ratio (SWIMMING table) | G trait outside the landmark set (negative control) | in hand (cached); 109 target species, 21 within 50 Myr | 109 | NO skill: the 17-landmark set has no caudal-fin landmarks, so the engine should not beat the consensus on aspect ratio (ratio >= 0.9) | 70% | peduncle depth (LM5-LM6) correlates weakly with fin aspect ratio across teleosts, so some incidental skill is possible; n is tiny |

## 2. Other groups (same machinery, re-trained; all probabilities subjective)

The horizon rule being tested (P7): coverage collapses at about 0.28 x the clade's sampled crown age (fish: 50 of 177.5 Myr).

| Group | Phylogeny | Trait data | Trait type | n (order) | Crown (Ma) | Pred. horizon (Myr) | Blind design | P(beats consensus) | P(coverage 85-97%) | P(beats nearest) |
|---|---|---|---|---|---|---|---|---|---|---|
| insects: ants (Formicidae) | Economo EP et al. 2018 Nat Commun 9:1778 (species-level tree, ~14,000 tips, grafted) | GlobalAnts (Parr CL et al. 2017 Insect Conserv Divers 10:5-20): worker head width, head length, Weber's length, eye length, scape and femur length | linear morphometrics (worker caste) | 10^3 | ~150 | ~42 | train on species with records in the first GlobalAnts release; predict species added in later releases or in AntWeb specimen measurements | 85% | 55% | 35% |
| insects: butterflies (Papilionoidea) | Kawahara AY et al. 2023 Nat Ecol Evol (2,244 species, time-calibrated) | LepTraits 1.0 (Shirey V et al. 2022 Sci Data 9:382): wingspan / forewing length for 12,448 species | linear size trait | 10^3 | ~100 | ~28 | the 2,244 tree species are a small fraction of LepTraits; train on half the tree species, predict the other half, then predict species added to the tree by later studies | 85% | 55% | 35% |
| insects: Drosophilidae wings | Houle D, Bolstad GH, van der Linde K, Hansen TF 2017 Nature 548:447-450 (111 species, ~40 Myr) | wing landmarks / semilandmarks for the same 111 species (Houle et al. 2017 supplementary data) | landmark geometric morphometrics | 10^2 | ~40 | ~11 | leave-clade-out within the 111 species; a truly blind test needs new species digitised to the same scheme | 75% | 45% | 30% |
| birds | Jetz W et al. 2012 Nature 491:444 (9,993 species; birdtree.org) | AVONET (Tobias JA et al. 2022 Ecol Lett 25:581-597): 11 linear traits (beak length/width/depth, tarsus, wing, Kipp's distance, tail, mass) for 11,009 species; 3-D beak shape PCs for 2,028 species (Cooney CR et al. 2017 Nature 542:344) | linear morphometrics + 3-D landmark PCs | 10^4 | ~110 | ~31 | train the engine on the 2,028 Cooney beak-shape species, predict AVONET beak length/width/depth for the remaining ~9,000 species (independent measurements, independent team) | 95% | 60% | 40% |
| mammals | Upham NS et al. 2019 PLoS Biol 17:e3000494 (5,911 species) | PanTHERIA (Jones KE et al. 2009 Ecology 90:2648) -> COMBINE (Soria CD et al. 2021 Ecology 102:e03344): body mass, head-body length, and life history; cranial 3-D landmarks for 322 species (Goswami A et al. 2022 Science 378:377) | linear size traits + 3-D cranial landmarks | 10^3 | ~180 | ~51 | train on PanTHERIA (2009) species, predict species whose first measurement appears only in COMBINE (2021): genuinely blind on species not yet measured when the training compilation was made | 90% | 55% | 35% |
| squamate reptiles | Tonini JFR et al. 2016 Biol Conserv 204:23 (9,755 species) | Meiri S 2018 Glob Ecol Biogeogr 27:1168 'Traits of lizards of the world': maximum SVL, limb/leg state, clutch size, habitat | linear size + categorical | 10^3 | ~200 | ~56 | train on lizards, predict snakes' SVL? No - different body plan. Train on 2018 release, predict species added in the 2024 update of the same database | 85% | 50% | 35% |
| amphibians | Jetz W & Pyron RA 2018 Nat Ecol Evol 2:850 (7,238 species) | AmphiBIO (Oliveira BF et al. 2017 Sci Data 4:170123): body size, reproductive and ecological traits | linear size + categorical | 10^3 | ~300 | ~85 | train on AmphiBIO v1, predict species added in later releases | 80% | 50% | 30% |
| seed plants | Smith SA & Brown JW 2018 Am J Bot 105:302 (seed-plant megatree, ~80,000 tips) | Diaz S et al. 2016 Nature 529:167 (six traits: height, seed mass, LMA, leaf area, leaf N, stem density; ~46,000 species) and TRY (Kattge J et al. 2020 Glob Change Biol 26:119) | linear functional traits | 10^4 | ~330 | ~93 | train on Diaz et al. 2016 species, predict TRY-only species added after 2016 | 70% | 45% | 30% |

Reasoning per group:

- **insects: ants (Formicidae)** - worker size and Weber's length are strongly phylogenetically conserved in ants; caste polymorphism (majors/minors) and inconsistent measurement protocols add noise; in-hand Camponotus GM data (Zenodo 10.5281/zenodo.18807285) and Pheidole images in .thompson_cache/antscan offer a small first test
- **insects: butterflies (Papilionoidea)** - wing size has high phylogenetic signal; LepTraits values are literature-compiled ranges (noisy); essentially a one-trait test of the machinery
- **insects: Drosophilidae wings** - closest analogue to the fish engine (2-D landmarks on a time tree) and the dataset where a Thompson-style low-dimensional mode set is most likely to work; small n limits power, hence the lower probability
- **birds** - best-case group: complete tree, complete trait coverage, museum callipers, lambda ~ 0.9 for most AVONET traits; the beak-shape -> beak-calliper design mirrors the fish landmark -> calliper design exactly
- **mammals** - body mass is among the most conserved traits known (lambda > 0.95); the two-compilation design gives a clean blind split; COMBINE imputes some values and those must be excluded
- **squamate reptiles** - SVL is conserved but repeated limb loss and body elongation across squamates make shape far less predictable than size; a good place to test the horizon rule because the clade is old
- **amphibians** - size conserved at the family level, but miniaturisation is frequent and convergent; the 300 Myr crown implies, under the 0.28 rule, a horizon of ~85 Myr, which is a strong and testable claim
- **seed plants** - seed mass and height have high phylogenetic signal, leaf economics traits much less (lambda ~ 0.5-0.7); plasticity and intraspecific variance are large; the lowest-confidence group on the list

Joint probability that *every* listed group beats its consensus within the horizon (P6, product of the subjective numbers): 22%. Expected number of groups succeeding: 6.7 of 8.

## 3. Portfolio-level hypotheses (registered now)

| ID | Primary | Statement | Confidence |
|---|---|---|---|
| P1 | yes | For any dataset of linear body-proportion traits on ray-finned fishes (class A), the locked full-30 predictions beat the consensus (ratio < 0.8, paired Wilcoxon p < 0.01) for species within 50 Myr of a training species. | simulated per dataset below |
| P2 | yes | The nearest-relative margin does NOT hold up: on class A fish datasets the full-30 prediction improves on copying the nearest training species by less than 10% (H2-type test fails again). | simulated per dataset below (reported as P(H2 passes); P2 is supported if it does not) |
| P3 | yes | Beyond 50 Myr the behaviour is bin-specific, not uniform: the realised BDmax error ratio vs consensus and the realised 95% coverage of each deep-horizon dataset fall inside the simulated intervals given below (ratio +/- 0.10, coverage +/- 10 points). Concretely: cichlids (80-120 Myr) ~0.77 with ~68% coverage; characiforms (>120 Myr, root-state predictions) ~0.43 with ~82% coverage. | subjective 0.6: intervals are simulated, but the tolerance and the transfer across landmark schemes are judgement calls |
| P4 | no | Head-length and predorsal ratios (HL/SL, PD/SL) show weaker skill than BDmax/SL on the same species (error ratio vs consensus at least 0.1 higher). | subjective 0.6: the jaw/head module had a worse process fit than the body module in the engine; FishBase picture measurements of HL are noisy |
| P5 | no | Traits outside the landmark set (caudal-fin aspect ratio) show no skill (ratio vs consensus >= 0.9). | subjective 0.7 |
| P6 | yes | The same machinery (AICc-selected process model on a time tree, Gaussian conditional prediction, horizon-binned calibration) re-trained on another clade beats that clade's consensus within the horizon for every group listed. | subjective, per group (p_beats_consensus); product over groups is the probability all succeed |
| P7 | no | The horizon beyond which 95% coverage collapses below 75% falls at 0.28 +/- 50% of the clade's sampled crown age (fish: 50 of 177.5 Myr) in every group tested. | subjective 0.4: one-point extrapolation; it is registered because it is cheap to falsify |
| P8 | no | Where a Thompson-style low-dimensional a-priori mode set exists (Drosophila wings; bird beaks), it beats the consensus but is worse than the full shape space, as in fish (H6). | subjective 0.65 |

## 4. Sub-prediction protocol (applies to every item)

- **step_1**: Before downloading or opening a dataset: if its traits are already in the locked registration (BDmax, CPD, HL, PD over SL for all 10,888 Rabosky targets, commit dac08493) nothing new is generated; otherwise generate species-level predictions, SHA-256 them, commit, and publish a preregistration nanopub citing this portfolio nanopub.
- **step_2**: Download; record checksum; match species by exact name; per-species median over specimens/pictures.
- **step_3**: Calibrate the measurement convention with a linear map fitted on TRAINING species only (as in the FishShapes protocol); apply to all methods.
- **step_4**: Score exactly the registered tests (paired one-sided Wilcoxon, alpha = 0.01; tau bins 0-5, 5-10, 10-25, 25-50, 50-80, 80-120, >120 Myr as registered; 95% coverage).
- **step_5**: Publish an outcome nanopub citing the preregistration and this portfolio; update the realised columns of docs/THOMPSON_DATASET_PORTFOLIO.md without changing the registered probabilities.
- **for_other_clades**: Fit the engine on the training split of that clade's tree, lock predictions for every other tip, then obtain the comparison values. Horizon bins are scaled to the clade's crown age.

## 5. What this is and is not

- The simulated probabilities assume the FishShapes error profile transfers across measurement protocols for the same trait on the same tree. That is the assumption being tested; if FishBase MORPHMET comes out far from the simulated interval, the simulation itself is falsified.
- The subjective probabilities for other groups are priors, written down so they can be scored (Brier) once the groups are tested. They are deliberately not dressed up as outputs of a model.
- Nothing in this document is updated after a dataset is scored except the 'Realised' column.
