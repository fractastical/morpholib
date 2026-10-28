# MorphoEval report — audit `2026-10-28-t1917-019-the-forms-of-tissues-or-cell-a-audit-v2`

Rubric **0.1.0** · checkpoint `f1b27eb65` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-019-the-forms-of-tissues-or-cell-a-audit -> 2026-10-28-t1917-019-the-forms-of-tissues-or-cell-a-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Ch. VII — tissue cells as hexagons / froth vs. human corneal endothelium (Pass 2: segmented specimen, sweep, controls, statistics) (`thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py`) | 78 | 72 | 80 | 74 | 88 | 80 | 80 | 78.7 | 72.8 | 82.7 | 77.5 | **77.5** | mostly faithful |

## Findings and remediation

### Thompson (1917) Ch. VII — tissue cells as hexagons / froth vs. human corneal endothelium (Pass 2: segmented specimen, sweep, controls, statistics)  —  overall 77.5 (mostly faithful)

Specimen (85 interior cells of 126): <n> = 5.894 [95 % CI 5.78, 6.01], window-bias corrected 6.08; p6 = 0.659 [0.55, 0.77], p5 = 0.224, p7 = 0.118, mu2 = 0.34; 100 % trivalent junctions; median shape index 3.96 (regular hexagon 3.72, square 4.00); area CV 0.285 vs clinical norm 0.315; excess perimeter 7.7 % raw, 8.6 % after the rasterised-froth pipeline correction; Lewis lambda 0.18 (Lewis 1928: 0.25); Aboav a = 0.74 (froths 1.0-1.5), neighbour-shuffle control a = 6.1. Thompson's Fig. 125: A (compressed) p6 = 0.89, B (relaxed) p6 = 0.97. Best topology fit: fully relaxed CVT froth (delta = 1.0, Lloyd = 20), TV = 0.054, r = 0.99, chi-square p = 0.42; joint topology+geometry fit (delta = 0.2, Lloyd = 0) cannot match the tissue's area CV (model 0.18) — the froth model class reproduces the tissue's topology but not its size dispersion. Specimen vs Gibson 2006 proliferating spectrum: r = 0.94, RMSE 0.10, chi-square p = 0.005 (significantly more hexagonal). Verdict: SUPPORTED on Thompson's own exemplar tissue (C1 mean six, C2 least perimeter within 10 %, C3 p6 modal and >= 0.5), qualified for proliferating epithelia.

- **Model fidelity to source: 78** — The froth laws (E1-E5) are implemented exactly and proved in Lean; the claim is now actually tested — Euler mean six, trivalency, honeycomb bound and hexagon fraction are all measured on tissue and on Thompson's own Fig. 125. The division chain (E6) reproduces Gibson's spectrum to 0.03 and its ablation shows cleavage geometry matters. Remaining gap: least perimeter is modelled by Lloyd relaxation (a proxy), and the froth model class cannot match topology and area dispersion simultaneously (joint fit), which the script records rather than hides.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L198–257; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L316–360; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L486–514; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L151–163; `morphoeval/lean/loops/Thompson1917_T1917_019.lean` L115–148
- **Morphological realism / specimen match: 72** — Side by side: the segmented corneal endothelium coloured by polygon class, the digitised Fig. 125 A/B, and the best-fit froth. The tissue reads as a mildly disordered hexagonal froth and the best-fit froth matches its polygon spectrum (TV 0.054, chi-square p = 0.42); but the relaxed CVT froth is visibly more uniform in cell size (area CV 0.10 vs 0.285), the joint fit shows no sweep point matches both, and the comparison rests on a single specimen.  
  evidence: `docs/media/t1917_019_the_forms_of_tissues_or_cell_a_comparison.png` L0–0; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L363–397; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L534–557; `datasets/thompson_1917/t1917-019/result.json` L0–0
- **Parameter expressivity: 80** — 50-point delta x Lloyd sweep rendered (p6 and excess perimeter vs delta for four Lloyd depths; (p6, mu2) morphospace with tissue, plate, Gibson, Poisson and honeycomb points); monotonicity asserted in --self-test; --delta, --lloyd, --nx exposed on the CLI; the GIF traces p6 and excess perimeter during relaxation against the specimen's p6. Lattice size and cleavage probability p_cut are not swept.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L473–483; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L1069–1077; `docs/media/t1917_019_the_forms_of_tissues_or_cell_a_growth.gif` L0–0; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L266–270
- **Morphometric validation: 74** — 16 measurements, 14 measured on the specimen or the 1917 plate, each with a literature reference: p6 0.659 vs clinical norm 0.616 (within 0.55-0.70), area CV 0.285 vs 0.315, <n> 5.89 vs 6, trivalent 1.00 vs 1, Lewis lambda 0.18 vs 0.25, Aboav a 0.74 vs 1.2, excess perimeter 8.6 % vs the Hales bound 0; bootstrap 95 % CIs; r / RMSE / TV / KS and chi-square against Gibson, best froth, honeycomb and Poisson null; pipeline control quantifies pixel-perimeter and finite-window bias. Limits: n = 1, no um scale, and the measured excess perimeter conflates polygon irregularity with wall waviness.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L933–1007; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L438–470; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L746–764; `datasets/thompson_1917/t1917-019/result.json` L0–0; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L721–731
- **Source and data traceability: 88** — Unchanged provenance (IA scan + page rule, 7 page-pinned quotes, 12 Crossref-verified DOIs, Commons specimens with author/licence/sha1/SHA-256) plus result.json now pins the dataset digest and states the dataset-match rationale; Fig. 125 crop boxes documented from the ink-density rule positions. Specimen scale still unknown.  
  evidence: `datasets/thompson_1917/t1917-019/result.json` L0–0; `datasets/thompson_1917/t1917-019/t1917_019_the_forms_of_tissues_or_cell_a_reference.json` L1–40; `datasets/thompson_1917/t1917-019/paper_citations.bib` L1–30
- **Code-to-model traceability: 80** — Module docstring maps C1-C3 and steps 1-7 to Thompson pages and equations E1-E6; every function docstring names its law; math_kg code_locus/lean_locus pass the integrity and formula-alignment gates; Fig. 125 crop boxes carry their derivation comment. Segmentation hyper-parameters (blueness 40, min contact 4 px, min_distance) are documented but not derived from the image.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L1–50; `morphoeval/math_kg_loops/thompson1917-t1917-019.yaml` L1–40; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L92–97
- **Reproducibility package: 80** — Deterministic (seed asserted; identical metrics on repeated runs), --self-test 2.0 s covering laws, controls and segmentation sanity, full run 24 s, pinned dependencies (requirements.txt / .lock), CLI documented, dataset README with invocation. The comparison PNG is force-added past the repo's *.png ignore rule.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L1010–1066; `datasets/thompson_1917/t1917-019/README.md` L1–60

Remediation:
- {'id': 'R7', 'item': 'Independent cross-provider second assessor — still builder-scored'}
- {'id': 'R8', 'item': 'More specimens: >= 5 corneal endothelia and at least one segmented proliferating epithelium (e.g. Drosophila wing disc) to test the Gibson regime directly rather than from the published spectrum'}
- {'id': 'R9', 'item': 'Attach a um/px scale (specular-microscopy field or scale-bar specimen) to report cell density and absolute areas'}
- {'id': 'R10', 'item': 'Replace Lloyd relaxation by a Plateau / vertex-model energy descent (Farhadifar 2007) so that least perimeter is minimised, not proxied'}
- {'id': 'R11', 'item': 'Separate wall waviness from polygon irregularity in the excess perimeter (vertex-to-vertex polygonal perimeter vs Crofton perimeter)'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=3.0]** `specimen_comparison` → `specimen-match`: Polygon-class overlay of the segmented corneal endothelium next to the model froth at the best-fit delta
- **[RESOLVED · P=4.0]** `morphometric_validation` → `morphometric-validation`: p6, <n>, mu2, CV of area and excess perimeter measured on tissue, with reference norms
- **[RESOLVED · P=0.5]** `parameter_sweep` → `parameter-expressivity`: delta x Lloyd sweep with p6 / mu2 / excess perimeter curves
- **[OPEN · P=0.5]** `independent_assessor` → `model-fidelity`: Cross-provider second assessor
- **[OPEN · P=4.0]** `specimen_comparison` → `morphometric-validation`: Segmented proliferating epithelium to test the Gibson 2006 regime on data rather than from the literature spectrum

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Ch. VII — tissue cells as hexagons / froth vs. human corneal endothelium (Pass 2: segmented specimen, sweep, controls, statistics)
- **spec_fingerprint** (semantic model + inputs): `573eaa78956bb7a95427321ba84764635768abd58c2fd7fd76e7660dc905fc59`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-019/elodea_canadensis_leaf_specimen.jpg` (`ffa66c5d5bfa1afb…`), `datasets/thompson_1917/t1917-019/human_corneal_endothelium_specimen.jpg` (`9daed5fb546e11bf…`), `datasets/thompson_1917/t1917-019/paper_citations.bib` (`0f7f54da3056b65d…`), `datasets/thompson_1917/t1917-019/scout_manifest.json` (`9edbd03ddd14f395…`), `datasets/thompson_1917/t1917-019/t1917_019_the_forms_of_tissues_or_cell_a_reference.json` (`85abd22429855187…`), `datasets/thompson_1917/t1917-019/thompson_1917_p322_fig125_126_froth_elodea.jpg` (`1a31755fe8c43ca4…`)
- `datasets/thompson_1917/t1917-019/result.json` — artifact_sha256 `69a794a8aa303dba…`, 3,555 B, 
- `docs/media/t1917_019_the_forms_of_tissues_or_cell_a_comparison.png` — artifact_sha256 `12cce483342dadf1…`, 783,069 B, 1600×1300 px
- `docs/media/t1917_019_the_forms_of_tissues_or_cell_a_growth.gif` — artifact_sha256 `4308f85616c2df66…`, 1,526,947 B, 900×480 px, 25 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen mean polygon side number <n> (interior cells) | 5.8941 sides | 6.0 | nucleus-marker watershed on human corneal endothelium, 85 interior cells; bootstrap 95% CI [5.78, 6.01]; finite-window bias from raster control -0.183 (window-corrected <n> = 6.077) |
| specimen hexagon fraction p6 | 0.6588 fraction | 0.616 | interior-cell neighbour count = 6; bootstrap 95% CI [0.55, 0.76] *(honeycomb (Thompson strict) = 1.0; Gibson 2006 proliferating wing disc = 0.464; Poisson-Voronoi null = 0.25)* |
| specimen second moment mu2 of polygon distribution | 0.3412 sides^2 | 0.0 | sum p_n (n-6)^2 over interior cells; CI [0.24, 0.45] *(Poisson-Voronoi null control mu2 = 1.99; best-fit froth mu2 = 0.34)* |
| specimen trivalent junction fraction | 1.0 fraction | 1.0 | 2x2 label windows with 3 distinct labels / (3 or >=4) |
| specimen excess perimeter over honeycomb bound (bias-corrected) | 8.5883 % | 0.0 | sum P_i / sum 2*12^(1/4) sqrt(A_i) - 1 on Crofton perimeters, minus rasterised-froth pipeline bias -0.91 % (raw 7.68 %) *(best-fit froth model excess perimeter = 1.69 %)* |
| specimen median shape index P/sqrt(A) | 3.9641 dimensionless | 3.7224 | Crofton perimeter / sqrt(pixel area), interior cells |
| specimen coefficient of variation of cell area (polymegethism) | 0.2845 CV | 0.315 | std/mean of interior pixel areas |
| specimen Lewis slope lambda (<A_n>/<A> = 1 + lambda (n-6)) | 0.1814 1/side | 0.25 | linear fit over classes [5, 6, 7], r = 0.9997346364418448 |
| specimen Aboav-Weaire parameter a | 0.7368 dimensionless | 1.2 | fit n m(n) = (6-a) n + (6a+mu2), r = 0.999009636440421; neighbour-shuffle control a = 0.09 |
| specimen Aboav-Weaire sum-rule residual (1/F) sum n m(n) - (36 + mu2) | -1.8941 sides^2 | 0.0 | Lean Law 5 aboav_weaire_sum_rule consistency check (exact on a torus; finite-window specimen) |
| Pearson r specimen vs Gibson 2006 polygon distribution (n = 4..8) | 0.9436 r | 1.0 | RMSE = 0.101, TV = 0.195, KS D = 0.100, chi-square p = 0.00499 |
| Pearson r specimen vs best-fit froth model polygon distribution | 0.9914 r | 1.0 | delta = 1.0, Lloyd = 20; RMSE = 0.032, TV = 0.054, chi-square p = 0.423 |
| total-variation distance specimen vs honeycomb (= 1 - p6) | 0.3412 TV | 0.0 | 0.5 sum |p_n - delta_n6|; Thompson strict reading baseline |
| Thompson Fig. 125 A (compressed soap froth) hexagon fraction, digitised | 0.8889 fraction | 0.616 | line-drawing watershed on IA leaf n341 crop [560, 1096, 1400, 1306], 63 interior cells, <n> = 5.78 |
| pipeline control: p6 recovery error on rasterised froth | 0.0127 fraction | 0.0 | recovered - exact p6 for delta = 0.35, 202 interior cells; honeycomb raster recovered p6 = 1.000 |
| division-topology chain fixed-point p6 (through-centre cleavage reconstruction) | 0.4336 fraction | 0.464 | mean-field Gibson 2006 Markov chain, p_cut = 1/3; uniform-random cleavage ablation p6 = 0.126 |

## Environment at checkpoint


Branch cursor/t1917-019-epithelial-polygon-packing-4281, Pass 2 commit. Environment: Python 3.12.3, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, Pillow 13.0.0, scikit-image 0.26.0; Lean 4.35.0 for the companion module. Full run 24 s, --self-test 2.0 s (includes segmentation sanity checks).
