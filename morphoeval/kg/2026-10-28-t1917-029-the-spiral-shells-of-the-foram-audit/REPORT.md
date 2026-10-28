# MorphoEval report — audit `2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit`

Rubric **0.1.0** · checkpoint `0eb517d23d79fca6826d903b09d425781dd017b4` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-029: logarithmic-spiral chamber succession with constant ratio vs Shi (2021) fusulinid half-whorl series (`thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py`) | 72 | 45 | 50 | 62 | 85 | 70 | 70 | 64.7 | 51.4 | 75.0 | 62.5 | **62.5** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-029: logarithmic-spiral chamber succession with constant ratio vs Shi (2021) fusulinid half-whorl series  —  overall 62.5 (mostly faithful)

v1 implements Thompson's model faithfully as far as he states it (constant ratio, constant angle, limiting cases, mass-proportional volume law) and tests the central quantitative claim on a correctly matched public dataset: 106 fusulinid specimens give median r = 0.995 for ln D vs half-whorl stage, the geometric law beats arithmetic growth in 100 % of specimens, and Triticites reproduces von Moller's Fusulina ratio (1.239 +/- 0.021 vs 1.225 per half-whorl; 1.53 vs 1.5 per whorl). Weaknesses: (1) the real specimen photograph is shown but not segmented or measured, so specimen-match rests on eyeballing; (2) no parameter sweep demonstrates that W, d, overlap and T control the model's form monotonically — the four morphs are hand-picked points; (3) no negative control (shuffled stages / synthetic arithmetic series / random-walk ratios) is run through the verdict rules; (4) the panel-G finding that pseudoschwagerine ratios rise to ~1.45 and fall to ~1.1 through ontogeny (systematic, not a 'small secondary fluctuation') is visible but untested — T2 uses only a CV threshold, so the verdict machinery cannot register the most interesting deviation from Thompson's law; (5) T4 threshold is uncalibrated (for n ~ 96 interior points the expected max/median gap of a continuous distribution is ~6.6, so 6.34 is the null expectation); (6) goodness-of-fit lacks KS/normality of residuals and confidence intervals; (7) the GIF model shell uses an arbitrary a0 = 0.5 that does not resemble Nummulina; (8) environment versions differ from requirements.lock.txt.

- **Model fidelity to source: 72** — E1-E5 are implemented exactly as Thompson states them and verified numerically against the Lean theorems in self_test; the generative model (E6) is Thompson's verbal construction made explicit and its limiting cases (d = 0 Nodosarian, rho = 1 Orbitolitid) are correct. Docked because chamber radius a0 and overlap are unconstrained free choices with no source value, the Textularian translation T is implemented but never exercised or shown, and the Raup (1966) W-D-T correspondence is asserted in a docstring rather than demonstrated.  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L89–140; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L602–641; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L109–127
- **Morphological realism / specimen match: 45** — Panel A's four morphs are recognisably Globigerina-like, Cyclammina-like (Fig. 313 style 20-degree chambers), Triloculina-like and an Orbitolitid ring, and are visually distinct from one another (credited). Panel E shows a genuine fusulinid axial section next to panel F's Dryad outlines, and the two are the same kind of object (fusiform nested whorls). But nothing is overlaid or measured on the photograph, the model shells are never compared with Fig. 309 Nummulina or with the thin section, and the GIF's 'Nummulina-like' shell (a0 = 0.5, d = 24 degrees) is a thick ring of overlapping discs that does not resemble either specimen image.  
  evidence: `docs/media/t1917_029_the_spiral_shells_of_the_foram_comparison.png` L0–0; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L429–434; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L491–491
- **Parameter expressivity: 50** — Only four hand-picked (W, d, a0) points are rendered; there is no sweep across W or d, no morphometric read-out (coiling tightness, chamber overlap fraction, number of chambers per whorl, aspect ratio) as a function of parameter, and so monotonic control is asserted (and proven in Lean for size) but not shown for form. The Nodosarian branch of chamber_series is exercised only in self_test.  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L367–377; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L633–637
- **Morphometric validation: 62** — Strong core: 106 specimens x ~13 stages, per-specimen r / RMSE / ratio CV / BIC vs arithmetic, genus means with SD, and an explicit comparison with Thompson's printed 1.225 and 1.5 (T3 passes at 1.239 / 1.53). Thompson's own tables are refitted and his stated ratios recovered. Docked because: no confidence intervals or KS/normality test on residuals; T2 (CV <= 0.15) cannot detect the systematic ontogenetic hump of the pseudoschwagerines seen in panel G; T4 uses an uncalibrated threshold whose null expectation (~6.6 for n ~ 96) exceeds the cut-off, so the v1 FAIL is an artefact of the statistic, not of the data; no negative control.  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L143–175; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L309–339; `datasets/thompson_1917/t1917-029/result.json` L0–0; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L230–241
- **Source and data traceability: 85** — Dryad DOI, Crossref-verified primary article (author list corrected to single author, Paleobiology 47(1):115-133), IA scan with verified leaf mapping, Commons file with SHA-1 and licence, SHA-256 for every vendored file in the reference JSON and scout manifest, .gitattributes protecting the CRLF bytes of the Dryad files. Docked because von Moller (1878) and van Iterson (1907) have no resolvable identifiers (only Thompson's citation), and Przibram (1913) is cited without DOI.  
  evidence: `datasets/thompson_1917/t1917-029/t1917_029_the_spiral_shells_of_the_foram_reference.json` L85–118; `datasets/thompson_1917/t1917-029/paper_citations.bib` L12–40; `datasets/thompson_1917/t1917-029/scout_manifest.json` L1–60
- **Code-to-model traceability: 70** — Every model function cites the page / equation it implements (E1-E6), Thompson's tables are constants with page comments, and the Lean file names the theorem for each law. Docked because the analysis-side choices (PCA frame from the mid-ontogeny outline, 0.85 scale-break threshold, 5-95 % interior trimming, thresholds in T1-T4) are documented in comments but not tied to any source or pre-registration beyond the scout manifest's generic T1-T4 list.  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L76–83; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L215–229
- **Reproducibility package: 70** — Deterministic (SEED only affects jitter in panel D), --self-test covers model identities, the refits and the loader, the run takes 2.4 s and reproduced byte-identical outputs twice. Docked because the environment used (numpy 2.5.4 / matplotlib 3.11.2 / Pillow 13.0.0) is not the lockfile's, scipy is imported nowhere yet listed in the README environment, and there is no regression assertion on the headline numbers (a silent change of the longest-segment rule would pass --self-test).  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L602–641; `datasets/thompson_1917/t1917-029/README.md` L46–60; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L644–688

Remediation:
- {'id': 'R1', 'item': "specimen-match / morphometric-validation (L429-L434): segment the fusulinid thin section — locate the proloculus, take sector-median radial intensity profiles along the minor axis, detect dark whorl walls, fit ln r_j vs j, and compare the resulting whorl ratio with the Dryad Triticites and Thompson's 1.5; overlay detected walls on panel E."}
- {'id': 'R2', 'item': 'parameter-expressivity (L367-L377): add a (W, d) sweep of the generative model with morphometric read-outs (chambers per whorl, chamber overlap fraction, coiling tightness r_last/r_first, aspect ratio) and show monotonic control; exercise the translation T (Textularian) branch.'}
- {'id': 'R3', 'item': 'morphometric-validation (L309-L339): add a negative/ablation control — run the verdict rules on (a) within-specimen shuffled stage order, (b) synthetic arithmetic-growth series with matched noise, (c) random-walk ratios — and report that T1/T3 reject them.'}
- {'id': 'R4', 'item': 'morphometric-validation (L318): replace the CV-only T2 with a test of systematic ratio drift (quadratic term in ln D vs k with its t-statistic / p-value per genus, plus the stage of maximum ratio) so the pseudoschwagerine inflation hump in panel G is scored, not just plotted.'}
- {'id': 'R5', 'item': 'morphometric-validation (L329): calibrate T4 with a Monte-Carlo null — the largest interior gap of n samples from a continuous (normal / KDE) fit to the pooled ratios — and report a p-value instead of the fixed 6.0 threshold.'}
- {'id': 'R6', 'item': 'morphometric-validation (L143-L175): add goodness-of-fit statistics — KS test of log residuals against normal, bootstrap 95 % CI of the genus mean ratios, RMSE in both spaces — and expose them in measure_morphoeval().'}
- {'id': 'R7', 'item': "specimen-match (L491): choose GIF/figure model parameters from the data (W from the Triticites fit, d and a0 fitted to Fig. 309's visible chamber count per whorl) and draw the model section next to the Fig. 309 crop."}
- {'id': 'R8', 'item': 'repro-package (L644-L688): stamp numpy/matplotlib/Pillow versions and the script version into the metrics JSON; add headline-number regression assertions to --self-test; drop the unused scipy mention from the README environment list or use scipy for the KS test.'}
- {'id': 'R9', 'item': 'source-traceability (paper_citations.bib L12-L40): add resolvable URLs (BHL / IA) for von Moller 1878 and van Iterson 1907, and a DOI or URL for Przibram 1913.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-029: logarithmic-spiral chamber succession with constant ratio vs Shi (2021) fusulinid half-whorl series
- **spec_fingerprint** (semantic model + inputs): `e8dfa9fdde138927298f774fce032da52edc892f40295eed80203fdda26888b1`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-029/fusulinid_axial_thin_section_commons_NEON.jpg` (`39f54718bbc8afa2…`), `datasets/thompson_1917/t1917-029/scout_manifest.json` (`1420d9e34b22dcc7…`), `datasets/thompson_1917/t1917-029/shi2020_dryad_sn02v6x26_Landmarks.dat` (`f7bb3fe73332d58c…`), `datasets/thompson_1917/t1917-029/shi2020_dryad_sn02v6x26_Table_1.csv` (`b5ddc6561b74b26d…`), `datasets/thompson_1917/t1917-029/t1917_029_the_spiral_shells_of_the_foram_reference.json` (`cdd3787ee526a2a9…`), `datasets/thompson_1917/t1917-029/thompson1917_fig309_nummulina_antiquior_vonmoller_p591.jpg` (`e7e37b2c3ea0a6f3…`)
- `datasets/thompson_1917/t1917-029/result.json` — artifact_sha256 `738c11d51999982a…`, 3,684 B, 
- `docs/media/t1917_029_the_spiral_shells_of_the_foram_comparison.png` — artifact_sha256 `3ac36d67d48be41b…`, 557,632 B, 1760×1100 px

| measurement | value | reference | method |
|---|---:|---|---|
| Triticites half-whorl growth ratio (mean of per-specimen log-linear fits) | 1.2385 ratio | 1.225 | slope of ln D_k on half-whorl index k per specimen, exp(slope), mean over Triticites specimens (Dryad doi:10.5061/dryad.sn02v6x26) *(sd 0.021; whorl ratio 1.534 vs Thompson 1.5)* |
| Pooled median Pearson r of ln D_k vs k (geometric succession) | 0.9953 r | 1.0 | per-specimen Pearson correlation of ln(test diameter) with half-whorl stage, median over 106 specimens |
| Pooled median CV of successive half-whorl ratios | 0.076 CV | — | std/mean of D_{k+1}/D_k within specimen, median over specimens — dispersion of the 'constant' ratio *(Thompson p. 595: ratio 'not strictly constant ... small secondary fluctuation')* |
| Fraction of specimens where geometric beats arithmetic growth (BIC) | 1.0 fraction | — | BIC of log-linear vs linear-in-k model on ln D residuals; alternative-model control |
| Pooled half-whorl ratio range across 5 genera | 0.2317 ratio | — | max-min of per-specimen rho; largest gap/median gap = 6.34 (continuity of the series vs gaps) |
| von Moller F. cylindrica half-whorl ratio refit | 1.2243 ratio | 1.225 | log-linear refit of the 8 breadths Thompson prints on p. 593 |
| van Iterson Triloculina chamber ratio refit | 1.3223 ratio | 1.323 | log-linear refit of chambers 2-10 breadths Thompson prints on p. 595 |
| Pseudoschwagerina half-whorl growth ratio (mean) | 1.2985 ratio | — | as above, per genus *(sd 0.048; ratio-vs-stage trend -0.0101/stage)* |
| Sphaeroschwagerina half-whorl growth ratio (mean) | 1.3289 ratio | — | as above, per genus *(sd 0.026; ratio-vs-stage trend -0.0016/stage)* |
| Zellia half-whorl growth ratio (mean) | 1.2952 ratio | — | as above, per genus *(sd 0.035; ratio-vs-stage trend -0.0088/stage)* |
| Robustoschwagerina half-whorl growth ratio (mean) | 1.3181 ratio | — | as above, per genus *(sd 0.044; ratio-vs-stage trend -0.0035/stage)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scipy: 1.18.1
- lean: 4.35.0
- lockfile: requirements.lock.txt (NOT the versions used here — pip resolved newer numpy/matplotlib/Pillow)

Pass 1 commit on branch cursor/t1917-029-foram-spiral-shells-eb42 (pushed); all cited loci exist at this commit.
