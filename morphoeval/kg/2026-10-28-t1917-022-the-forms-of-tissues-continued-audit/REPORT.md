# MorphoEval report — audit `2026-10-28-t1917-022-the-forms-of-tissues-continued-audit`

Rubric **0.1.0** · checkpoint `a8bae15d56cf26f6d31eb8cd1b41b5301fdde110` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-022 — isoperimetric/minimal-area hollow & aggregate partitions (`thompson/claims/t1917-022_the-forms-of-tissues-continued.py`) | 58 | 40 | 45 | 42 | 78 | 68 | 62 | 53.7 | 40.8 | 68.9 | 53.1 | **53.1** | partially faithful |

## Findings and remediation

### Thompson (1917) T1917-022 — isoperimetric/minimal-area hollow & aggregate partitions  —  overall 53.1 (partially faithful)

Pass 1 baseline: Lean laws E1-E5 pinned; Thompson IA plates vendored; initial script lacked robust junction-angle metrology and full parameter sweeps; catch-all claim scored as split structures.

- **Model fidelity to source: 58** — Core isoperimetric, Plateau, and Fig. 176 template laws implemented; Weaire-Phelan ablation noted but froth KS weak.  
  evidence: `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L1–80; `morphoeval/lean/loops/Thompson1917_T1917_022.lean` L1–90; `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L350–380
- **Morphological realism / specimen match: 40** — Plates displayed; segmentation immature on first pass.  
  evidence: `datasets/thompson_1917/t1917-022/scout_manifest.json` L1–60; `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L130–200
- **Parameter expressivity: 45** — delta and Lloyd sweeps declared; not all tied to GOF tables in v1.  
  evidence: `datasets/thompson_1917/t1917-022/t1917_022_the_forms_of_tissues_continued_reference.json` L30–45
- **Morphometric validation: 42** — measure_morphoeval present; KS to froth immature.  
  evidence: `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L480–520
- **Source and data traceability: 78** — SHA-256 pinned reference JSON and scout manifest.  
  evidence: `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L55–70
- **Code-to-model traceability: 68** — Plate crops and equation IDs documented.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-022.yaml` L1–40
- **Reproducibility package: 62** — self-test and media outputs; result.json added in remediation.  
  evidence: `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L525–545

Remediation:
- Fix Plateau junction-angle metrology and froth raster gradient pipeline.
- Improve Fig. 176 radial voting and pollen circularity segmentation.
- Add GOF RMSE/r/KS between froth model and tissue junction samples; write result.json verdict.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-022 — isoperimetric/minimal-area hollow & aggregate partitions
- **spec_fingerprint** (semantic model + inputs): `d5662788b05b86432d342a16505bcc2afe5c80c3c960513c6c22811bd6065d39`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-022/scout_manifest.json` (`a3a033535db37a98…`), `datasets/thompson_1917/t1917-022/t1917_022_the_forms_of_tissues_continued_reference.json` (`81a2311b2af01bda…`), `morphoeval/lean/loops/Thompson1917_T1917_022.lean` (`9420d08a3db775ab…`)
- `docs/media/t1917_022_the_forms_of_tissues_continued_comparison.png` — artifact_sha256 `722cf2379c51dfd9…`, 1,645,444 B, 2100×1350 px

| measurement | value | reference | method |
|---|---:|---|---|
| Fig. 176 eight-sector template match score | 0.875 fraction | 0.75 | radial voting vs 8×45° template |
| Fig. 177 pluteus radial line index | 0.5868 index | 0.15 | gradient alignment with radial rays |
| Fig. 180 pollen mean isoperimetric Q | 0.5442 Q | 0.52 | 4πA/P² on segmented grains |
| Tissue junction mean |angle−120°| | 19.3537 deg | 25.0 | Plateau junction metrology on Figs. 178/187/190 |
| Radiolaria analogue mean Q (Challenger plate) | 0.4967 Q | 0.4 | Commons plate blob circularity |
| Froth-tissue junction KS p-value | 0.0  | 0.05 | two-sample KS on junction angle lists |
| Froth best delta (Voronoi disorder) | 0.55  | 0.15 | min mean |angle−120| over delta×Lloyd grid |

## Environment at checkpoint


Pass 1 baseline before Step 12 remediation
