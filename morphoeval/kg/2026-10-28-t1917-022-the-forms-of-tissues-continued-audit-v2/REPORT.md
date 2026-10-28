# MorphoEval report — audit `2026-10-28-t1917-022-the-forms-of-tissues-continued-audit-v2`

Rubric **0.1.0** · checkpoint `a8bae15d56cf26f6d31eb8cd1b41b5301fdde110` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib · supersession chain `2026-10-28-t1917-022-the-forms-of-tissues-continued-audit -> 2026-10-28-t1917-022-the-forms-of-tissues-continued-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-022 — isoperimetric/minimal-area hollow & aggregate partitions (`thompson/claims/t1917-022_the-forms-of-tissues-continued.py`) | 72 | 68 | 70 | 65 | 82 | 74 | 78 | 71.3 | 66.9 | 78.3 | 71.6 | **71.6** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-022 — isoperimetric/minimal-area hollow & aggregate partitions  —  overall 71.6 (mostly faithful)

Pass 2: 6/7 structure-split tests pass; verdict mixed (catch-all claim + Weaire-Phelan foam caveat). Fig. 176 eight-fold match 0.88, pollen mean Q≈0.54, junctions within 25° of 120° on 35%+ samples, radiolarian analogue Q≈0.50; froth KS still marginal.

- **Model fidelity to source: 72** — Lean-backed laws plus Voronoi froth sweep with Lloyd relaxation and ablation on delta.  
  evidence: `morphoeval/lean/loops/Thompson1917_T1917_022.lean` L24–88; `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L210–280
- **Morphological realism / specimen match: 68** — Six Thompson crops segmented; two Commons radiolarian analogues measured.  
  evidence: `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L290–360; `datasets/thompson_1917/t1917-022/result.json` L1–40
- **Parameter expressivity: 70** — delta×Lloyd grid and pollen threshold sweep executed.  
  evidence: `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L44–52
- **Morphometric validation: 65** — GOF RMSE/r/KS exported; T6 froth KS still fails.  
  evidence: `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L365–395; `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L400–430
- **Source and data traceability: 82** — Seven SHA-256 vendored files; scout MATCH + analogue labels.  
  evidence: `datasets/thompson_1917/t1917-022/scout_manifest.json` L40–90
- **Code-to-model traceability: 74** — PLATE_CROPS pinned; math_kg_loops yaml links Lean theorems.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-022.yaml` L1–35
- **Reproducibility package: 78** — self-test ~2 min; comparison PNG + GIF; result.json verdict.  
  evidence: `thompson/claims/t1917-022_the-forms-of-tissues-continued.py` L525–570

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


Pass 2 after Step 12 remediation (segmentation, sweeps, result.json)
