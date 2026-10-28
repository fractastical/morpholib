# MorphoEval report — audit `2026-10-28-t1917-018-the-forms-of-tissues-or-cell-a-audit-v2`

Rubric **0.1.0** · checkpoint `18879252d` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-28-t1917-018-the-forms-of-tissues-or-cell-a-audit -> 2026-10-28-t1917-018-the-forms-of-tissues-or-cell-a-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-018 — Plateau tri-junction (120 degree) claim: Lami balance, periodic froth relaxation, junction-angle statistics (`thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py`) | 88 | 84 | 86 | 88 | 90 | 88 | 88 | 87.3 | 85.5 | 88.7 | 87.0 | **87.0** | faithful |

## Findings and remediation

### Thompson (1917) T1917-018 — Plateau tri-junction (120 degree) claim: Lami balance, periodic froth relaxation, junction-angle statistics  —  overall 87.0 (faithful)

Pass 2 (v2) evaluation (overall 87.0/100, up from 40.2/100 in Pass 1; +46.8 points). The remediated script SHA-256-verifies all four vendored specimens, segments the human corneal endothelium (126 cells, 215 vertices), the onion epidermis (47 cells, 68 vertices) and Thompson's own Figs. 103 and 124 (13 and 28 vertices), and scores tri-junction angles with two estimators against a-priori criteria. Every detected vertex in every specimen is three-fold (f3 = 1.000; Plateau's second law). In the isotropic corneal monolayer the chord-angle distribution has median |theta - 120| = 7.2 degrees [bootstrap 95 % CI 6.3, 7.7], SD 12.5 degrees, 81 % within +/-15 degrees, and is statistically indistinguishable from the relaxed equal-tension froth model (two-sample KS D = 0.045, p = 0.60; histogram r = 0.98) while far from the Poisson-Voronoi null (KS D = 0.27; SD 29.5 degrees). The anisotropic onion epidermis and Thompson's Fig. 124 behave as he predicted (still three-fold, angles spread; best-fit anisotropy eps = 0.2 and 0.3), and his Fig. 103 is drawn with median |theta - 120| = 2.3 degrees. Verdict: SUPPORTED. Remaining weaknesses: the branch estimator puts the cornea at 15.9 degrees (estimator sensitivity at the tolerance edge), the onion segmentation under-counts small cells, Fig. 124's four boundary T-junctions are not separated from interior ones, and the model is straight-edged (no film curvature).

- **Model fidelity to source: 88** — `lami_angles_from_tensions` / `tensions_from_angles` / `thompson_fig113_cases` (L140-172) implement Thompson's triangle of forces exactly, `steiner_vs_cross_lengths` (L174-177) Lamarle's four-fold instability, and `PeriodicNetwork.energy_gradient` (L284-311) the Farhadifar-type vertex energy with orientation-dependent tension and an analytic gradient verified against finite differences in the self-test (L1267-1319). The equal-tension Plateau limit (Gamma = eps = 0) is now reached with combinatorial T1 exchanges (`t1_shortest_edge`, L345-406) and a gradient-norm convergence criterion (`relax_froth`, L440-465): the relaxed froth is three-valent with mean six sides, SD 9.9 degrees. What keeps it from the top band: edges are straight, so the model cannot reproduce the curved-film 120 degrees of a real froth in a disordered topology (documented in the claim boundary), and the 3D tetrahedral law is not simulated.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L140–177; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L284–311; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L345–465
- **Morphological realism / specimen match: 84** — All four specimens are loaded, verified (`verify_inputs`, L124-137) and segmented with pipelines matched to their imaging (`segment_specimen`, L532-572: nucleus-seeded watershed on a Sato ridge for the alizarin-red corneal mosaic, Sato wall mask + components for the magenta-walled onion, skeleton graph of the closed ink for Thompson's drawings, `skeleton_graph_analysis` L681-778), and the comparison figure (`render_figures`, L988-1109) shows each specimen with its segmentation and detected vertices beside the model and the nulls. The cornea segmentation is clean (126 cells, one nucleus each). Deductions: the onion pipeline drops small cells (47 of roughly 60 visible), and Fig. 124's four boundary T-junctions on the strip's outer walls are scored together with interior vertices.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L532–572; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L681–778; `docs/media/t1917_018_the_forms_of_tissues_or_cell_a_comparison.png` L0–0
- **Parameter expressivity: 86** — `run_model_suite` (L864-908) sweeps the tension anisotropy eps over seven values (SD rises monotonically 9.9 -> 11.8 -> 17.4 -> 25.7 -> 36.1 -> 43.8 -> 51.5 degrees, i.e. eps visibly controls the Fig. 124 distortion), the Lami partition-tension ratio t/T continuously, and contrasts the Plateau limit with the Farhadifar Case I perimeter-contractility comparator; `parse_args` (L1321-1339) exposes --eps-sweep, --lambda-bar, --gamma-bar, --seed, --steps, --n-cells, --bootstrap, --tolerance, --quick and --no-figures. K and the T1 threshold remain module constants.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L864–908; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L1321–1339
- **Morphometric validation: 88** — `plateau_statistics`, `bootstrap_vertex_ci`, `two_sample_ks` and `histogram_fit` (L180-232) deliver SD, RMS deviation, median |theta - 120| with vertex-resampled 95 % CIs, fraction within the a-priori +/-15 degree tolerance, CV, two-sample KS D/p, histogram Pearson r and RMSE; `goodness_of_fit` (L910-935) fits eps by KS distance and compares each specimen with the Plateau froth, the Poisson-Voronoi null and the Farhadifar comparator; `decide_verdict` (L937-973) applies the pre-registered criteria and records the branch-estimator robustness. The cornea result (KS p = 0.60 vs the froth, D = 0.27 vs Voronoi) is a genuine quantitative match. Deduction: the two estimators disagree on the cornea (7.2 vs 15.9 degrees), so the angle criterion is estimator-sensitive at the tolerance edge; no literature junction-angle distribution is available to pin as an external reference value.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L180–232; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L910–973; `datasets/thompson_1917/t1917-018/result.json` L1–40
- **Source and data traceability: 90** — The script loads the reference package and SHA-256-verifies every specimen and both IA leaves at start-up (`verify_inputs`, L124-137; also in `--self-test`); result.json records the dataset names, digests, the IA leaf numbers, the Commons authors/licences and the Dryad DATASET_MISMATCH with its metadata digests. The docstring and reference.json carry page-pinned Thompson quotes and resolvable DOIs/URLs for every source. Minor gap: Plateau (1873) and Lamarle (1864) are cited by archive URL, not DOI.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L111–137; `datasets/thompson_1917/t1917-018/t1917_018_the_forms_of_tissues_or_cell_a_reference.json` L1–60
- **Code-to-model traceability: 88** — Section banners and function names map onto the equation labels E1/E2/E3/E5 of the reference package; the six Lean theorems of `Thompson1917_T1917_018.lean` (Fig. 113 cases, Lami iff equal tensions, angle-sum identity, Euler/T1 bookkeeping, hexagon closure, Steiner inequality) compile with Lean 4.35.0 and 0 sorry, and `math_kg_loops/thompson1917-t1917-018.yaml` pins each law to v2 line ranges. The self-test checks the Euler invariant (mean six sides) after T1s. Deduction: the segmentation heuristics (thresholds 0.18, 0.6 x Otsu, 0.3 x median area, 15 px ink distance) are documented in docstrings but not derived from a source.  
  evidence: `morphoeval/lean/loops/Thompson1917_T1917_018.lean` L39–229; `morphoeval/math_kg_loops/thompson1917-t1917-018.yaml` L1–140
- **Reproducibility package: 88** — Deterministic seeds, a `--self-test` (L1267-1319) covering Lami cases, inverse Lami, SHA-256 pins, analytic gradient vs finite difference, T1 three-valence and Euler invariant, determinism and a segmentation smoke test; the default run takes 12 s and writes result.json plus a full metrics JSON that `measure_morphoeval()` (L1216-1265) consumes so MorphoEval measurements are byte-identical to the run. Deductions: no environment lock file for this loop beyond the audit's recorded versions, and the PNG/GIF bytes depend on the Matplotlib version.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L1267–1319; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L1216–1265

Remediation:
- Pass 3: add a curved-edge (circular-arc) froth model or a Surface-Evolver-style relaxation so the Plateau limit reaches 120 degrees exactly, and compare specimen tangent angles against it.
- Pass 3: separate Fig. 124's outer-wall T-junctions from interior vertices (Thompson's p. 314 exemption) and improve the onion pipeline with nucleus markers to recover the small peripheral cells.
- Pass 3: digitise a published tricellular-junction angle histogram (e.g. Drosophila notum / wing disc) as an external reference value and add a second, independent assessor.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-018 — Plateau tri-junction (120 degree) claim: Lami balance, periodic froth relaxation, junction-angle statistics
- **spec_fingerprint** (semantic model + inputs): `b6891b9d48370d1a3d2785b75836612aaca82ba408f906fd75dc438b1a976266`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-018/allium_cepa_epidermis_specimen.jpg` (`12d2cff76378adae…`), `datasets/thompson_1917/t1917-018/human_corneal_endothelium_specimen.jpg` (`9daed5fb546e11bf…`), `datasets/thompson_1917/t1917-018/ongrowthform1917thom_0318.jp2` (`fb329d9ddbe640d8…`), `datasets/thompson_1917/t1917-018/ongrowthform1917thom_0341.jp2` (`38d6d1a9bc86ad90…`), `datasets/thompson_1917/t1917-018/paper_citations.bib` (`ef63468ff8403071…`), `datasets/thompson_1917/t1917-018/scout_manifest.json` (`50b930d12b0f97e0…`), `datasets/thompson_1917/t1917-018/t1917_018_the_forms_of_tissues_or_cell_a_reference.json` (`213d7c371e7f98fe…`), `datasets/thompson_1917/t1917-018/thompson1917_fig103_maize_parenchyma.png` (`d1c3c3aa92d21321…`), `datasets/thompson_1917/t1917-018/thompson1917_fig124_girardia_epidermis.png` (`3a33c7b1b82aad70…`)
- `docs/media/t1917_018_the_forms_of_tissues_or_cell_a_comparison.png` — artifact_sha256 `94aa66cc8749d83f…`, 1,042,338 B, 2100×1102 px

| measurement | value | reference | method |
|---|---:|---|---|
| corneal endothelium: three-fold vertex fraction f3 | 1.0 fraction | 1.0 | vertices with exactly 3 cells / all detected vertices (N=215) |
| corneal endothelium: chord median |theta - 120| | 7.2419 deg | <= 15.0 | median absolute deviation of chord angles at 3-fold vertices (N3=181) |
| corneal endothelium: chord angle SD | 12.5182 deg | 0.0 | pooled SD of chord angles at 3-fold vertices |
| corneal endothelium: fraction within +/-15 deg | 0.814 fraction | — | chord estimator |
| corneal endothelium: branch-estimator median |theta - 120| | 15.9393 deg | <= 15.0 | mean-offset wall directions within R=22px |
| onion epidermis: three-fold vertex fraction f3 | 1.0 fraction | 1.0 | vertices with exactly 3 cells / all detected vertices (N=68) |
| onion epidermis: chord median |theta - 120| | 15.4118 deg | <= 15.0 | median absolute deviation of chord angles at 3-fold vertices (N3=46) |
| onion epidermis: chord angle SD | 22.7529 deg | 0.0 | pooled SD of chord angles at 3-fold vertices |
| onion epidermis: fraction within +/-15 deg | 0.471 fraction | — | chord estimator |
| onion epidermis: branch-estimator median |theta - 120| | 8.3286 deg | <= 15.0 | mean-offset wall directions within R=35px |
| Thompson Fig. 103: three-fold vertex fraction f3 | 1.0 fraction | 1.0 | vertices with exactly 3 cells / all detected vertices (N=13) |
| Thompson Fig. 103: chord median |theta - 120| | 2.3106 deg | <= 15.0 | median absolute deviation of chord angles at 3-fold vertices (N3=11) |
| Thompson Fig. 103: chord angle SD | 3.1386 deg | 0.0 | pooled SD of chord angles at 3-fold vertices |
| Thompson Fig. 103: fraction within +/-15 deg | 1.0 fraction | — | chord estimator |
| Thompson Fig. 103: branch-estimator median |theta - 120| | 3.6423 deg | <= 15.0 | mean-offset wall directions within R=65px |
| Thompson Fig. 124: three-fold vertex fraction f3 | 1.0 fraction | 1.0 | vertices with exactly 3 cells / all detected vertices (N=28) |
| Thompson Fig. 124: chord median |theta - 120| | 13.1891 deg | <= 15.0 | median absolute deviation of chord angles at 3-fold vertices (N3=20) |
| Thompson Fig. 124: chord angle SD | 28.2649 deg | 0.0 | pooled SD of chord angles at 3-fold vertices |
| Thompson Fig. 124: fraction within +/-15 deg | 0.5833 fraction | — | chord estimator |
| Thompson Fig. 124: branch-estimator median |theta - 120| | 14.4382 deg | <= 15.0 | mean-offset wall directions within R=23px |
| corneal endothelium: KS D vs Plateau froth | 0.0449 dimensionless | — | two-sample KS, chord angles vs relaxed Gamma=eps=0 froth |
| corneal endothelium: KS D vs Poisson-Voronoi null | 0.2667 dimensionless | — | two-sample KS |
| corneal endothelium: best-fit tension anisotropy eps | 0.0 dimensionless | — | argmin KS D over the eps sweep |
| corneal endothelium: histogram Pearson r at best eps | 0.9831 dimensionless | — | 5-degree bins |
| corneal endothelium: histogram RMSE at best eps | 0.0016 density | — | 5-degree bins |
| Plateau froth model: junction-angle SD | 9.9458 deg | 0.0 | relaxed Gamma=eps=0 froth, 24 T1s, converged=True |
| Poisson-Voronoi null: junction-angle SD | 29.4814 deg | — | unrelaxed periodic Voronoi seed |
| Farhadifar Case I: junction-angle SD | 12.2974 deg | — | Lambda_bar=0.12, Gamma_bar=0.04 relaxed |
| eps sweep: SD at eps=0.6 minus SD at eps=0 | 41.5049 deg | — | monotone dispersion gain with tension anisotropy |
| Lami equal-tension junction angle | 120.0 deg | 120.0 | E1 with T1=T2=T3 |
| Steiner / cross length ratio (unit square) | 0.9659 dimensionless | 0.9659 | (1+sqrt3)/(2 sqrt2) |
| verdict (supported=1, mixed=0.5, refuted=0) | 1.0 code | — | pre-registered criteria in reference.json |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scikit-image: 0.26.0

Pass 2 (v2) evaluation after the 12-step closed remediation loop. Every Pass 1 remediation item (specimen segmentation + SHA-256 verification; equal-tension Plateau limit with T1 exchanges and anisotropy sweep; controls, goodness-of-fit and result.json) is implemented and exercised by the default run (12 s) and by `--self-test`.
