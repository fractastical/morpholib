# MorphoEval report — audit `2026-10-28-t1917-019-the-forms-of-tissues-or-cell-a-audit`

Rubric **0.1.0** · checkpoint `afd867b4c` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Ch. VII — tissue cells as hexagons / froth vs. human corneal endothelium (Pass 1 baseline) (`thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py`) | 62 | 35 | 55 | 30 | 85 | 72 | 75 | 59.7 | 33.1 | 77.5 | 54.4 | **54.4** | partially faithful |

## Findings and remediation

### Thompson (1917) Ch. VII — tissue cells as hexagons / froth vs. human corneal endothelium (Pass 1 baseline)  —  overall 54.4 (partially faithful)

v1 builds a periodic Voronoi froth morphospace (hexagonal lattice + disorder delta, optional Lloyd relaxation) and computes the froth laws exactly: torus <n> = 6.000, honeycomb mu2 = 0 / excess perimeter 0, P^4 ordering 192 < 256 < 432, Lewis slope, Aboav-Weaire a with the sum rule exact, and a through-centre-cleavage division chain whose fixed point (p6 = 0.434, p5 = 0.267, p7 = 0.221) reproduces Gibson 2006 (0.464 / 0.289 / 0.208) to within 0.03. At the canonical delta = 0.35 the model froth has p6 = 0.343, mu2 = 1.17, excess perimeter 8.95 %. The corneal endothelium specimen and Thompson's Fig. 125 soap froth are shown side by side but NOT segmented, so nothing in v1 measures the claim on tissue.

- **Model fidelity to source: 62** — The froth laws the claim rests on (E1 Euler mean six, E2 mu2, E3 Hales bound, E4 Lewis, E5 Aboav-Weaire) are implemented exactly on Voronoi polygons and asserted in --self-test; the division chain (E6) reproduces Gibson's spectrum to 0.03. But "least perimeter" is modelled only by Lloyd relaxation (a proxy), and the model is never confronted with tissue, so the claim itself is not yet tested.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L98–124; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L159–212; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L226–276; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L127–138; `morphoeval/lean/loops/Thompson1917_T1917_019.lean` L37–68
- **Morphological realism / specimen match: 35** — The vendored corneal endothelium (Thompson's own exemplar tissue, p. 321) and Fig. 125 are displayed but no overlay, segmentation or polygon colouring is produced for either; the model froth at delta = 0.35 (p6 = 0.34) is visibly more disordered than the corneal mosaic. Side-by-side resemblance is therefore judged by eye only and is weak.  
  evidence: `docs/media/t1917_019_the_forms_of_tissues_or_cell_a_comparison.png` L0–0; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L298–320; `datasets/thompson_1917/t1917-019/human_corneal_endothelium_specimen.jpg` L0–0
- **Parameter expressivity: 55** — Disorder delta monotonically lowers p6 and raises mu2 / excess perimeter, and Lloyd relaxation reverses it — both asserted in --self-test and shown in the GIF — but only one delta is rendered in the figure, no sweep panel exists, and Lloyd iterations / lattice size are not exposed on the CLI.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L468–479; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L494–499; `docs/media/t1917_019_the_forms_of_tissues_or_cell_a_growth.gif` L0–0
- **Morphometric validation: 30** — Eight quantitative measurements are emitted, each with a literature reference (Gibson 2006 p6, Lewis slope 1/4, Aboav a = 1.2, Hales bound 0), but all are computed on the MODEL; no morphometric is measured on the specimen or on the 1917 plate, so the comparison is model-vs-literature, not output-vs-specimen.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L415–451; `datasets/thompson_1917/t1917-019/t1917_019_the_forms_of_tissues_or_cell_a_metrics.json` L0–0
- **Source and data traceability: 85** — IA scan pinned by identifier and leaf->page rule, seven primary quotes with page numbers, twelve Crossref-verified DOIs, two CC BY-SA 4.0 Commons specimens with author, licence URL, Commons sha1 and SHA-256, three public-domain plates SHA-256-pinned. Scale (um/px) of the specimen is unknown.  
  evidence: `datasets/thompson_1917/t1917-019/t1917_019_the_forms_of_tissues_or_cell_a_reference.json` L1–40; `datasets/thompson_1917/t1917-019/paper_citations.bib` L1–30; `datasets/thompson_1917/t1917-019/scout_manifest.json` L60–100
- **Code-to-model traceability: 72** — Module docstring maps C1-C3 to Thompson pages and E1-E6; each function docstring names its equation and Lean law; the math_kg YAML pins code_locus and lean_locus (integrity + formula-alignment gates pass). Thompson's own text is cited by page but Fig. 125 crop boxes are hard-coded without derivation.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L1–40; `morphoeval/math_kg_loops/thompson1917-t1917-019.yaml` L1–30; `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L66–68
- **Reproducibility package: 75** — Deterministic (seed 20261028, asserted), --self-test in 0.3 s, full run 3.0 s, dependencies in requirements.txt / requirements.lock.txt, documented invocation in the dataset README. The comparison PNG is gitignored by the repo's *.png rule and had to be force-added; no CLI for lattice size or Lloyd iterations.  
  evidence: `thompson/claims/t1917-019_the-forms-of-tissues-or-cell-aggregates.py` L454–491; `datasets/thompson_1917/t1917-019/README.md` L1–60

Remediation:
- {'id': 'R1', 'item': 'Segment the corneal endothelium specimen (blue-channel ridge threshold -> distance-transform watershed), count neighbours of interior cells, measure <n>, mu2, p_n, p6, junction valence, shape index, area CV and excess perimeter on the tissue'}
- {'id': 'R2', 'item': 'Render a delta x Lloyd parameter-regime sweep and fit delta to the specimen polygon distribution'}
- {'id': 'R3', 'item': 'Add negative/ablation controls: Poisson-Voronoi null, uniform-random cleavage ablation of the division chain, Aboav-Weaire neighbour-shuffle control, rasterised-honeycomb pipeline positive control (pixel perimeter bias calibration)'}
- {'id': 'R4', 'item': 'Goodness-of-fit statistics: Pearson r and RMSE between specimen and model/Gibson/honeycomb p_n, chi-square / KS-type distance, bootstrap CI on p6 and <n>, CV of area vs Sheng & Bullimore 2007 norm'}
- {'id': 'R5', 'item': "Digitise Thompson's Fig. 125 A/B through the same pipeline as the physical froth reference"}
- {'id': 'R6', 'item': 'Write datasets/thompson_1917/t1917-019/result.json with the verdict; fix legend/honeycomb-bar overlap; expose --lloyd and --nx on the CLI'}
- {'id': 'R7', 'item': 'Independent cross-provider second assessor — still builder-scored'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=3.0]** `specimen_comparison` → `specimen-match`: Polygon-class overlay of the segmented corneal endothelium next to the model froth at the best-fit delta
- **[OPEN · P=4.0]** `morphometric_validation` → `morphometric-validation`: p6, <n>, mu2, CV of area and excess perimeter measured on tissue, with reference norms
- **[OPEN · P=0.5]** `parameter_sweep` → `parameter-expressivity`: delta x Lloyd sweep with p6 / mu2 / excess perimeter curves
- **[OPEN · P=0.5]** `independent_assessor` → `model-fidelity`: Cross-provider second assessor

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Ch. VII — tissue cells as hexagons / froth vs. human corneal endothelium (Pass 1 baseline)
- **spec_fingerprint** (semantic model + inputs): `340ff36032325b8ea801c38e36dff91aa9d34352293527c48827c7e2bd5fb74c`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-019/elodea_canadensis_leaf_specimen.jpg` (`ffa66c5d5bfa1afb…`), `datasets/thompson_1917/t1917-019/human_corneal_endothelium_specimen.jpg` (`9daed5fb546e11bf…`), `datasets/thompson_1917/t1917-019/paper_citations.bib` (`0f7f54da3056b65d…`), `datasets/thompson_1917/t1917-019/scout_manifest.json` (`9edbd03ddd14f395…`), `datasets/thompson_1917/t1917-019/t1917_019_the_forms_of_tissues_or_cell_a_reference.json` (`85abd22429855187…`), `datasets/thompson_1917/t1917-019/thompson_1917_p322_fig125_126_froth_elodea.jpg` (`1a31755fe8c43ca4…`)
- `datasets/thompson_1917/t1917-019/human_corneal_endothelium_specimen.jpg` — artifact_sha256 `9daed5fb546e11bf…`, 668,732 B, 1290×1031 px
- `datasets/thompson_1917/t1917-019/t1917_019_the_forms_of_tissues_or_cell_a_metrics.json` — artifact_sha256 `77a03998547244f4…`, 5,808 B, 
- `docs/media/t1917_019_the_forms_of_tissues_or_cell_a_comparison.png` — artifact_sha256 `592edcce78d6504c…`, 834,756 B, 1540×990 px
- `docs/media/t1917_019_the_forms_of_tissues_or_cell_a_growth.gif` — artifact_sha256 `6eb20a2e07bf9609…`, 1,319,286 B, 500×540 px, 25 frames

| measurement | value | reference | method |
|---|---:|---|---|
| froth model mean polygon side number <n> (torus) | 6.0 sides | 6.0 | periodic Voronoi, delta=0.35, 400 cells; Lean Law 1 torus_mean_six |
| froth model hexagon fraction p6 | 0.3425 fraction | 0.464 | polygon_statistics on model froth |
| froth model second moment mu2 | 1.17 sides^2 | 0.0 | sum p_n (n-6)^2; honeycomb control = 0 *(honeycomb control mu2 = 0.000; Poisson-Voronoi null ~ 1.78)* |
| froth model excess perimeter over honeycomb bound | 8.9492 % | 0.0 | sum P_i / sum 2*12^(1/4) sqrt(A_i) - 1 (E3, Hales 2001) *(honeycomb control = -0.00 %)* |
| froth model Lewis slope lambda | 0.1747 1/side | 0.25 | linear fit of <A_n>/<A> vs n |
| froth model Aboav-Weaire a | 0.9447 dimensionless | 1.2 | fit n m(n) = (6-a) n + (6a + mu2) |
| Aboav-Weaire sum-rule residual (1/F) sum n m(n) - (36 + mu2) | 0.0 sides^2 | 0.0 | Lean Law 5 aboav_weaire_sum_rule consistency check |
| division-topology chain fixed-point p6 (reconstruction) | 0.4336 fraction | 0.464 | mean-field Gibson 2006 Markov chain, p_cut = 1/3 |

## Environment at checkpoint


Branch cursor/t1917-019-epithelial-polygon-packing-4281, Pass 1 commit. Environment: Python 3.12.3, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, Pillow 13.0.0, scikit-image 0.26.0 (unused in v1); Lean 4.35.0 for the companion module. Full run 3.0 s, --self-test 0.3 s.
