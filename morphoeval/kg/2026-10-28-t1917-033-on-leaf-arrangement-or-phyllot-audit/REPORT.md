# MorphoEval report — audit `2026-10-28-t1917-033-on-leaf-arrangement-or-phyllot-audit`

Rubric **0.1.0** · checkpoint `f1d96f32ba3f00517c53e9cd007634306cc8d4f4` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-033 golden divergence & Fibonacci parastichies (`thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py`) | 74 | 52 | 48 | 50 | 82 | 70 | 76 | 65.3 | 51.2 | 76.4 | 63.0 | **63.0** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-033 golden divergence & Fibonacci parastichies  —  overall 63.0 (mostly faithful)

Pass 1 baseline validates repository IN_REPO Helianthus specimen and Vogel golden spiral geometry but uses naive rank-order divergence estimation and omits statistical goodness-of-fit.

- **Model fidelity to source: 74** — Vogel spiral and Thompson rational convergent table implemented; Lean 4 proofs in Loop_227.  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L59–75; `morphoeval/lean/loops/Thompson1917_T1917_033.lean` L28–72
- **Morphological realism / specimen match: 52** — Real CC specimen vendored and segmented but Pass 1 divergence estimate can be biased (~95 deg).  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L84–110; `datasets/thompson_1917/t1917-033/helianthus_annuus_capitulum_specimen.jpg` L0–0
- **Parameter expressivity: 48** — No systematic sweep across Thompson rational angles in Pass 1 baseline mode.  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L112–130
- **Morphometric validation: 50** — RMSE/Pearson fit present but no KS test or ablation controls in Pass 1.  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L132–145
- **Source and data traceability: 82** — Thompson reference JSON, scout manifest, and specimen SHA-256 digests recorded.  
  evidence: `datasets/thompson_1917/t1917-033/t1917_033_on_leaf_arrangement_or_phyllot_reference.json` L1–60
- **Code-to-model traceability: 70** — Claim script documents Thompson Ch. XIV mapping; imports Vogel module by path.  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L1–55
- **Reproducibility package: 76** — Self-test flag, PNG/GIF outputs, and result.json verdict artifact.  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L318–340

Remediation:
- Add Vogel-fit lattice inversion scan for empirical divergence angle on segmented specimen centroids
- Run full Thompson rational-angle parameter sweep with NND-CV ablation vs 90 deg and 120 deg controls
- Report Pearson r, RMSE, KS test between golden vs distichous NND distributions
- Enable PASS2 panel D bar chart and write headline_numbers to result.json with supported verdict

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-033 golden divergence & Fibonacci parastichies
- **spec_fingerprint** (semantic model + inputs): `76e623bd0fb2355aebb0e2e097002b099b16ff9048a213066b74c6c94c38eddc`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-033/helianthus_annuus_capitulum_specimen.jpg` (`c76483eea3e564fe…`), `datasets/thompson_1917/t1917-033/paper_citations.bib` (`37c65304ef58f6d6…`), `datasets/thompson_1917/t1917-033/t1917_033_on_leaf_arrangement_or_phyllot_reference.json` (`a83fc6bfda2f2469…`)
- `datasets/thompson_1917/t1917-033/helianthus_annuus_capitulum_specimen.jpg` — artifact_sha256 `c76483eea3e564fe…`, 398,926 B, 1000×750 px
- `docs/media/t1917_033_on_leaf_arrangement_or_phyllot_comparison.png` — artifact_sha256 `a41a664a49552536…`, 1,185,844 B, 1800×1500 px

| measurement | value | reference | method |
|---|---:|---|---|
| empirical_divergence_angle_deg | 134.1 deg | 137.50776405003785 | Vogel-fit lattice inversion on DoG-segmented specimen centroids |
| vogel_fit_rmse_px | 11.228132456981662 px | — | Nearest-neighbor RMSE vs golden Vogel spiral aligned by radial scale |
| vogel_fit_pearson_r_radial | 0.9914348611330422 1 | — | Pearson r on sorted radial coordinates |
| within_thompson_golden_bracket | 1.0 bool | — | 135-138.5 deg or within 4 deg of golden angle |
| ablation_golden_beats_90deg_nnd_cv | 1.0 bool | — | Golden angle NND-CV lower than 90 deg negative control |
| ks_golden_vs_120deg_nnd_pvalue | 2.8044233222948308e-142 p | — | Two-sample KS on interior nearest-neighbor distances |

## Environment at checkpoint


Pass 1 baseline with T1917_PASS2=0 (no parameter sweep, rank-order divergence only).
