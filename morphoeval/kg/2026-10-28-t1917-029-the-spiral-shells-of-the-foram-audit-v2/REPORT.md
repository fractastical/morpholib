# MorphoEval report — audit `2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2`

Rubric **0.1.0** · checkpoint `12d9d136ee5632ff58f859a3df2645e58ace3576` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit -> 2026-10-28-t1917-029-the-spiral-shells-of-the-foram-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-029: logarithmic-spiral chamber succession with constant ratio vs Shi (2021) fusulinid half-whorl series (Pass 2) (`thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py`) | 78 | 68 | 80 | 80 | 92 | 78 | 78 | 78.7 | 72.5 | 82.7 | 77.4 | **77.4** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-029: logarithmic-spiral chamber succession with constant ratio vs Shi (2021) fusulinid half-whorl series (Pass 2)  —  overall 77.4 (mostly faithful)

Pass 2 closes the Pass 1 gaps with measurements rather than assertions. The real fusulinid thin section yields W = 1.49 +/- 0.10 from 13 detected whorl walls, matching Thompson's von Moller value 1.5 and the Dryad Triticites 1.53; Thompson's own Fig. 309 Nummulina yields W = 1.61 +/- 0.18 from 17 wall ratios, a different constant angle (85.7 vs 86.1 deg), which is exactly Thompson's statement that genera differ in the rate of growth. The (W, d, T) sweep shows d alone sets chamber overlap (Globigerina -> Cyclammina), W alone sets coiling tightness, and T sets the helicoid aspect, all monotonically. The verdict machinery now registers the most interesting biology: the early/mid/late drift statistic fails all four pseudoschwagerine genera at 15 % (Triticites 4.6 %), so the pre-registered verdict is MIXED (3/4) — the logarithmic-spiral succession and Thompson's numbers are supported for his exemplar group, but 'near-constant ratio' is not a universal law of foraminiferal shells. Controls behave as designed (shuffled 1/4, arithmetic fails T1, exact-geometric 4/4; the i.i.d. control is a boundary case that passes by construction). Remaining weaknesses: KS rejects normality of residuals (reported, not hidden); the Fig. 309 estimate is coarse; chamber-level (septal) claims remain untested on new specimens; model a0/overlap remain unconstrained by any source value; the environment differs from the lockfile although it is now stamped in the metrics.

- **Model fidelity to source: 78** — Unchanged core (E1-E6 exactly as on pp. 593-598, numerically mirrored to the Lean theorems in self_test). Improved: the translation branch is exercised and read out (helicoid aspect), the limiting cases are rendered in the sweep, and the model is anchored to measured values (Fig. 309 first-wall radius and its own CV whorl ratio) rather than hand-picked numbers. Still docked for a0 = (W-1)/2 and d = 22.5 deg being analyst choices with no source value, and for the Raup (1966) correspondence remaining a docstring remark.  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L108–158; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L1085–1145; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L684–690
- **Morphological realism / specimen match: 68** — Both specimen images are now measured and overlaid: panel G shows 13 detected whorl walls on the fusulinid section, panel C shows 17 wall detections (8 sectors) on Fig. 309 with the data-fitted model spiral passing through them, and the GIF places the growing model beside Fig. 309 and the Dryad outlines. The sweep grid is recognisably Globigerina-like at d = 90-120, Cyclammina-like at d = 10-20 and Orbitolitid at W = 1.2. Docked because the model shells are circles (no chamber shape/overlap law), the Fig. 309 overlay is approximate (engraving is an oblique view), and Thompson's Figs. 311-317 genera are matched only by family resemblance.  
  evidence: `docs/media/t1917_029_the_spiral_shells_of_the_foram_comparison.png` L0–0; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L412–486; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L488–533; `docs/media/t1917_029_the_spiral_shells_of_the_foram_growth.gif` L0–0
- **Parameter expressivity: 80** — 5 x 6 (W, d) grid rendered as small multiples with four read-outs; monotonic checks pass for overlap vs d (for every W), tightness vs W (for every d) and helicoid aspect vs T, and are asserted in --self-test. Docked slightly because overlap is shown to be independent of W only implicitly (identical curves) and the chamber-count read-out is trivially 360/d.  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L161–206; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L703–754
- **Morphometric validation: 80** — 106 specimens x ~13 stages with per-specimen r / relative RMSE / ratio CV / BIC, genus means with bootstrap 95 % CIs, early/mid/late-third drift with Kruskal-Wallis p, KS of pooled residuals, Monte-Carlo gap test, two independent CV whorl ratios, Thompson's printed series refitted, and four controls through the identical rules — the pre-registered T1-T4 now discriminate (real data 3/4 vs shuffled 1/4 vs exact-geometric 4/4). Docked because the i.i.d. control is a boundary case rather than a true negative, KS rejects normality (honestly reported), and the T2 10 % drift limit is a pre-registered convention rather than a value Thompson stated.  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L321–409; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L550–582; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L588–663; `datasets/thompson_1917/t1917-029/result.json` L0–0; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L655–657
- **Source and data traceability: 92** — All nine bib entries now resolve (IA scans for von Moller 1878 and van Iterson 1907, Crossref DOI for Przibram 1913 with the volume/page correction of Thompson's footnote), Dryad DOI + digests, Crossref-verified Shi 2021, IA leaf mapping, Commons SHA-1/licence, every vendored file SHA-256-pinned, environment stamped. Docked for the Fig. 309 pin correction being documented only in the reference JSON note (no versioned pin history).  
  evidence: `datasets/thompson_1917/t1917-029/paper_citations.bib` L12–41; `datasets/thompson_1917/t1917-029/t1917_029_the_spiral_shells_of_the_foram_reference.json` L85–118
- **Code-to-model traceability: 78** — Every model function cites its page/equation; the verdict thresholds are named module constants with the pre-registration source in a comment; the KG YAML code loci were updated to the v2 line ranges; the CV thresholds are stated in the docstrings. Docked because the analysis-side conventions (PCA frame, 0.85 break threshold, 5-95 % trimming, 10 % drift) are documented but not sourced to any publication.  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L86–102; `morphoeval/math_kg_loops/thompson1917-t1917-029.yaml` L58–59; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L285–310
- **Reproducibility package: 78** — Deterministic (seeded bootstrap / MC / jitter), byte-identical outputs across runs, --self-test with headline regression asserts (Triticites ratio in [1.22, 1.26], 106 specimens, 14 breaks, r > 0.99, CV walls >= 5 per side, controls), --fast flag, environment stamped into metrics and result JSON, documented invocation in the README. Docked because the versions still differ from requirements.lock.txt and the run depends on the pinned proloculus centres in the reference JSON (a change there silently changes the CV numbers).  
  evidence: `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L1127–1142; `thompson/claims/t1917-029_the-spiral-shells-of-the-foraminifera.py` L969–975; `datasets/thompson_1917/t1917-029/README.md` L46–75

Remediation:
- {'id': 'R10', 'item': 'morphometric-validation: replace the i.i.d. boundary control with a true negative for T2 — a monotone-drift series (ratio rising linearly 1.1 -> 1.4) — so every test has a control that fails it.'}
- {'id': 'R11', 'item': "specimen-match: give model chambers Thompson's overlap/shape law (p. 602: Globigerina spheres vs Cyclammina overlapped) instead of circles, and fit a0/overlap to the Fig. 309 chamber outlines."}
- {'id': 'R12', 'item': 'morphometric-validation: test the septal-chamber-level claim on a modern specimen (e.g. a Peneroplis or Elphidium micrograph with visible sutures) so the Miliolid/rotaline part of the claim is measured, not only modelled.'}
- {'id': 'R13', 'item': 'repro-package: run under requirements.lock.txt (or update the lockfile) so the stamped environment and the lockfile agree; version the CV pins (proloculus centres) in the reference JSON with a changelog.'}
- {'id': 'R14', 'item': 'independent second assessor (blinded) before the MIXED verdict is quoted outside this loop.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-029: logarithmic-spiral chamber succession with constant ratio vs Shi (2021) fusulinid half-whorl series (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `a65e5f2d0b7aad60f22deae47babc008c2fa65221e8f2cdd854123ba8884a9f7`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-029/fusulinid_axial_thin_section_commons_NEON.jpg` (`39f54718bbc8afa2…`), `datasets/thompson_1917/t1917-029/scout_manifest.json` (`1420d9e34b22dcc7…`), `datasets/thompson_1917/t1917-029/shi2020_dryad_sn02v6x26_Landmarks.dat` (`f7bb3fe73332d58c…`), `datasets/thompson_1917/t1917-029/shi2020_dryad_sn02v6x26_Table_1.csv` (`b5ddc6561b74b26d…`), `datasets/thompson_1917/t1917-029/t1917_029_the_spiral_shells_of_the_foram_reference.json` (`f109a70536020eeb…`), `datasets/thompson_1917/t1917-029/thompson1917_fig309_nummulina_antiquior_vonmoller_p591.jpg` (`e7e37b2c3ea0a6f3…`)
- `datasets/thompson_1917/t1917-029/result.json` — artifact_sha256 `ce66eac681e9affd…`, 6,977 B, 
- `docs/media/t1917_029_the_spiral_shells_of_the_foram_comparison.png` — artifact_sha256 `c266574c8781e151…`, 883,361 B, 1800×1300 px
- `docs/media/t1917_029_the_spiral_shells_of_the_foram_growth.gif` — artifact_sha256 `6df0512f746268e7…`, 872,268 B, 1300×480 px, 20 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Triticites half-whorl growth ratio (mean of per-specimen log-linear fits) | 1.2385 ratio | 1.225 | slope of ln D_k on half-whorl index k per specimen, exp(slope), mean over 40 Triticites specimens (Dryad doi:10.5061/dryad.sn02v6x26); bootstrap 95 % CI *(sd 0.021; 95 % CI [1.232, 1.245]; whorl ratio 1.534 vs Thompson 1.5)* |
| Pooled median Pearson r of ln D_k vs k (geometric succession) | 0.9953 r | 1.0 | per-specimen Pearson correlation of ln(test diameter) with half-whorl stage, median over 106 specimens |
| KS p-value of standardized ln-residuals vs N(0,1) | 0.0007 p-value | — | two-sided Kolmogorov-Smirnov test on 1364 pooled standardized residuals of the log-linear fits (D = 0.054) |
| Pooled median relative RMSE of the geometric fit | 0.0932 fraction | — | sqrt(mean(((D - D_hat)/D)^2)) per specimen, median |
| Pooled median CV of successive half-whorl ratios | 0.076 CV | — | std/mean of D_{k+1}/D_k within specimen, median over specimens — dispersion of the 'constant' ratio *(Thompson p. 595: ratio 'not strictly constant ... small secondary fluctuation')* |
| Triticites ontogenetic ratio drift (early/mid/late thirds) | 0.0463 fraction | 0.1 | (max - min)/mean of the three stage-third mean ratios pooled over the genus; Kruskal-Wallis p = 1.4e-14 |
| Fraction of specimens where geometric beats arithmetic growth (BIC) | 1.0 fraction | — | BIC of log-linear vs linear-in-k model on ln D residuals; alternative-model control (arithmetic-growth negative control: 0.00) |
| Monte-Carlo p of the largest interior gap in the pooled ratio distribution | 0.7402 p-value | — | 4000 normal-null simulations of n=106 ratios; observed gap 0.0079 vs null median 0.0094 (continuity of the genus series) |
| CV whorl ratio from the fusulinid thin section (joint up/down fit) | 1.4936 ratio | 1.5 | dark whorl walls detected on sector-median radial profiles (13 walls), ln r vs whorl index with half-whorl offset; sector-width sensitivity sd 0.105 *(r = 0.952; Dryad Triticites whorl ratio 1.534)* |
| CV whorl ratio of Thompson's Fig. 309 Nummulina antiquior (engraving) | 1.6085 ratio | — | geometric mean of 17 consecutive bright-wall radius ratios in 8 radial sectors (sd 0.182); coarse: oblique shaded engraving *(alpha = 85.7 deg; differs from the fusulinid 1.5 as Thompson's 'genera differ in constant angle' (p. 601) requires)* |
| von Moller F. cylindrica half-whorl ratio refit | 1.2243 ratio | 1.225 | log-linear refit of the 8 breadths Thompson prints on p. 593 |
| van Iterson Triloculina chamber ratio refit | 1.3223 ratio | 1.323 | log-linear refit of chambers 2-10 breadths Thompson prints on p. 595 |
| Shuffled-stage negative control: median r | -0.027 r | — | verdict rules re-run on within-specimen shuffled stage order (null control; T1 must fail) |
| Pseudoschwagerina half-whorl growth ratio (mean) | 1.2985 ratio | — | as above, per genus; bootstrap 95 % CI *(CI [1.274, 1.322]; drift 14.8% (KW p=1.5e-17))* |
| Sphaeroschwagerina half-whorl growth ratio (mean) | 1.3289 ratio | — | as above, per genus; bootstrap 95 % CI *(CI [1.317, 1.344]; drift 15.2% (KW p=1.3e-15))* |
| Zellia half-whorl growth ratio (mean) | 1.2952 ratio | — | as above, per genus; bootstrap 95 % CI *(CI [1.278, 1.313]; drift 15.5% (KW p=9.0e-18))* |
| Robustoschwagerina half-whorl growth ratio (mean) | 1.3181 ratio | — | as above, per genus; bootstrap 95 % CI *(CI [1.302, 1.335]; drift 14.7% (KW p=1.5e-23))* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- lean: 4.35.0
- lockfile: requirements.lock.txt (NOT the versions used here; the versions actually used are stamped in t1917_029_metrics.json#environment)

Pass 2 commit on branch cursor/t1917-029-foram-spiral-shells-eb42 (pushed); all cited loci exist at this commit.
