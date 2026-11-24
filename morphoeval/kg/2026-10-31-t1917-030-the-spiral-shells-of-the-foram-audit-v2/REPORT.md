# MorphoEval report — audit `2026-10-31-t1917-030-the-spiral-shells-of-the-foram-audit-v2`

Rubric **0.1.0** · checkpoint `35d30f4cf` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-31-t1917-030-the-spiral-shells-of-the-foram-audit -> 2026-10-31-t1917-030-the-spiral-shells-of-the-foram-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-030 — unilocular Lagena Delaunay fit (Pass 2) (`thompson/claims/t1917-030_the-spiral-shells-of-the-foraminifera.py`) | 78 | 64 | 68 | 62 | 88 | 80 | 76 | 74.7 | 63.2 | 81.1 | 72.0 | **72.0** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-030 — unilocular Lagena Delaunay fit (Pass 2)  —  overall 72.0 (mostly faithful)

Pass 2 adds free-form floor gate, ellipsoid null and sphere-cylinder ablation, B/t parameter sweep, CV nRMSE and KS statistics. Verdict mixed: 8/26 compatible; flask-like 8/21; spindles 0/5.

- **Model fidelity to source: 78** — Full E7/E8 composite catalogue with comparators and sweep invariants.  
  evidence: `thompson/claims/t1917-030_the-spiral-shells-of-the-foraminifera.py` L437–525; `morphoeval/lean/loops/Thompson1917_T1917_030.lean` L156–221
- **Morphological realism / specimen match: 64** — Fig. 84 + Fig. 19 CV segmentation with scale bars; polythalamous Fig. 8 controls.  
  evidence: `thompson/claims/t1917-030_the-spiral-shells-of-the-foraminifera.py` L324–370; `datasets/thompson_1917/t1917-030/kireenko_2022_bdj_e87457_fig19.jpg` L0–0
- **Parameter expressivity: 68** — SWEEP_B and SWEEP_T exported; GIF animates B and flask neck elongation.  
  evidence: `thompson/claims/t1917-030_the-spiral-shells-of-the-foraminifera.py` L117–118; `docs/media/t1917_030_the_spiral_shells_of_the_foram_growth.gif` L0–0
- **Morphometric validation: 62** — Median Pearson r 0.95, KS and CV exported; verdict mixed with spindle refutation.  
  evidence: `datasets/thompson_1917/t1917-030/result.json` L36–55; `datasets/thompson_1917/t1917-030/t1917_030_metrics.json` L1–40
- **Source and data traceability: 88** — Pre-scouted Fig. 19 verified MATCH; IA Fig. 84 + Ch. XII leaf pinned.  
  evidence: `datasets/thompson_1917/t1917-030/scout_manifest.json` L10–35
- **Code-to-model traceability: 80** — Reference JSON E1-E8 and math_kg yaml cross-walk.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-030.yaml` L54–90
- **Reproducibility package: 76** — Deterministic self-test; ~25s full run; measure_morphoeval hook.  
  evidence: `thompson/claims/t1917-030_the-spiral-shells-of-the-foraminifera.py` L629–676

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-030 — unilocular Lagena Delaunay fit (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `b34282bb5042ed2c7e96a316685b10531cd3439cb7059e3c291735bdd26880b9`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-030/kireenko_2022_bdj_e87457_fig8_station5656mc.jpg` (`3b57acc0dfc98374…`), `datasets/thompson_1917/t1917-030/paper_citations.bib` (`102e5d153f118cc9…`), `datasets/thompson_1917/t1917-030/scout_manifest.json` (`dfb385784d2f08b9…`), `datasets/thompson_1917/t1917-030/thompson_1917_fig84_lagena_after_brady.png` (`f6cd3abb75217bd0…`), `datasets/thompson_1917/t1917-030/thompson_1917_leaf0276_p256_fig84.jp2` (`ed440583a8b144d5…`)
- `datasets/thompson_1917/t1917-030/kireenko_2022_bdj_e87457_fig19.jpg` — artifact_sha256 `ec66de1789890c24…`, 3,782,013 B, 2540×3668 px
- `docs/media/t1917_030_the_spiral_shells_of_the_foram_comparison.png` — artifact_sha256 `4e3e73d537f17875…`, 1,410,938 B, 2200×2000 px
- `docs/media/t1917_030_the_spiral_shells_of_the_foram_growth.gif` — artifact_sha256 `102e3c47804d57e7…`, 267,994 B, 640×320 px, 36 frames

| measurement | value | reference | method |
|---|---:|---|---|
| unilocular test silhouettes analysed (13 drawings + 5 photographs) | 26 count | 18 | Fig. 84 background-normalised segmentation; Kireenko plate Otsu segmentation |
| Delaunay-compatible unilocular tests | 8 count | 26 | best of E7/E8 with nRMSE <= max(0.06, 1.5 x free-form) and asymmetry <= 0.12 *(verdict mixed; fraction 0.31)* |
| median best Delaunay RMSE / r_max (tests) | 0.0648137754769045 dimensionless | 0.04147649925072913 | RMSE over all silhouette rows *(reference = free-form degree-6 noise floor)* |
| median ellipsoid null RMSE / r_max (tests) | 0.08726676657752529 dimensionless | 0.0648137754769045 | null model vs Delaunay baseline |
| median sphere+cylinder ablation RMSE / r_max (tests) | 0.13677483254980716 dimensionless | 0.0648137754769045 | unduloid shoulder ablated (C0 join) vs Delaunay baseline |
| tests where Delaunay beats the ellipsoid null | 11 count | 26 | nRMSE comparison |
| median 2-fold CV RMSE / r_max (best Delaunay) | 0.06490496491680446 dimensionless | — | fit on even rows, evaluate on odd rows and vice versa |
| median KS statistic observed vs fitted radii (best Delaunay) | 0.27769522144522146 dimensionless | — | scipy.stats.ks_2samp |
| median Pearson r observed vs fitted radius | 0.9520018991514708 dimensionless | — | np.corrcoef |
| median bilateral asymmetry index mean|r_L - r_R| / r_max | 0.07986088775493386 dimensionless | 0.0 | half-widths about the least-squares axis *(0 for a perfect surface of revolution)* |
| flask-like tests (L/W <= pi) Delaunay-compatible | 8 count | 21 | subset within the unduloid bead aspect bound |
| spindle tests (L/W > pi) Delaunay-compatible | 0 count | 5 | subset outside the bead aspect bound (E6, Lean bead_aspect_bound) |
| polythalamous negative controls Delaunay-compatible | 1 count | 0 | Globobulimina 64a, Quinqueloculina 66a fitted with the same pipeline *(control median nRMSE 0.042)* |
| Lagena sp. test length from 100 um scale bar | 223.51046698872784 um | 300.0 | scale-bar calibrated silhouette length (61a, 62a) *(order-of-magnitude check: Lagena tests are typically 0.15-0.6 mm (Jones 1984 key to unilocular hyaline forams))* |
| fitted Kenmotsu B range | 0.26-0.87 dimensionless | — | best catalogue member per specimen |

## Environment at checkpoint

