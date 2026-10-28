# MorphoEval report — audit `2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit`

Rubric **0.1.0** · checkpoint `b0060c02` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Ch. XIII — bovid horns as logarithmic spirals wound on a cone (Pass 1) (`thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py`) | 62 | 35 | 45 | 48 | 78 | 70 | 60 | 56.3 | 39.9 | 68.7 | 53.5 | **53.5** | partially faithful |

## Findings and remediation

### Thompson (1917) Ch. XIII — bovid horns as logarithmic spirals wound on a cone (Pass 1)  —  overall 53.5 (partially faithful)

Fig. 320 outer edge (550 points, 278 deg sweep) fitted by a free-pole logarithmic spiral: alpha_proj = 85.9 deg, whorl ratio 1.57, Pearson r(ln rho, theta) = 0.91, RMSE 16.7 px (4.6 % of mean radius). Log spiral beats circle (delta-AIC 246) and Archimedean spiral (delta-AIC 12). Thompson's alpha = 75 deg (R = 5.4) is NOT reproduced by the projection (alpha_proj - 75 = +10.9 deg). The v1 verdict logic returns "refuted" because both the r >= 0.95 and the |alpha - 75| <= 5 deg criteria fail; that logic ignores the sin(beta) cot(alpha) degeneracy and is itself a Pass-1 defect. Overall 48.9/100.

- **Model fidelity to source: 62** — E1-E3 are implemented exactly as Thompson states them (k = sin(beta) cot(alpha), R = e^(2 pi sin beta cot alpha), p. 558) and the gnomonic tube (E4) is rendered; the Lean module proves the whorl-ratio invariance, breadth law, projection non-identifiability and growth-velocity lemmas with 0 sorry. Deductions: the fit is done to a 2D projection with beta = 90 deg assumed when converting k to alpha, so the quantity compared to Thompson's 75 deg is alpha_proj, not alpha; E4 (breadth shares k) is formalised but not measured; E5 (velocity ratio) is only a rendering, not a fitted quantity.  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L70–117; `morphoeval/lean/loops/Thompson1917_T1917_031.lean` L32–101; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L444–473
- **Morphological realism / specimen match: 35** — The rendered comparison overlays the fitted spiral on Thompson's own figure and the overlay follows the dorsal and posterior edge closely, but (i) no real specimen is used in Pass 1 — the CC BY-SA Argali photograph is vendored and pinned but not analysed; (ii) the ventral end of the digitised edge includes the mandible junction (ln rho rises at the last 15 deg); (iii) the 3D helicoid panel uses beta = 60 deg chosen by hand, not fitted; (iv) only one species is shown, so the "distinct morphologies distinct" test is not exercised.  
  evidence: `docs/media/t1917_031_the_shapes_of_horns_and_of_tee_comparison.png` L0–0; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L291–311; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L172–189
- **Parameter expressivity: 45** — The alpha sweep panel (65-85 deg) and the model_regime_table (alpha x beta -> k, R) show the monotone effect of both parameters, and the self-test checks R(90) = 1 and that an acute cone shrinks R. But nothing is swept against the data: no beta sweep is rendered, no breadth ratio or handedness control, and no CLI parameters.  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L322–339; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L395–404
- **Morphometric validation: 48** — Quantitative: alpha_proj, R, Pearson r, RMSE and two delta-AIC values are computed on the digitised figure and compared with Thompson's published 75 deg and with explicit null models; measure_morphoeval returns 6 rows with reference values. Missing: no uncertainty (bootstrap CI, KS test), no breadth / inner-edge gnomonic test (E4), n = 1 drawing, no real specimen, and the registry Dryad table is pinned but not even summarised.  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L275–284; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L341–364; `datasets/thompson_1917/t1917-031/t1917_031_metrics.json` L0–0
- **Source and data traceability: 78** — Thompson's text is pinned to the IA scan (identifier, leaf mapping verified against page_numbers.json, OCR line range), every vendored image and the Dryad table carry SHA-256 digests re-verified at runtime, the Commons licence/author/URL are recorded, the Dryad DOI/version/zip digest are recorded, and the registry-dataset mismatch is stated explicitly. Deductions: the Dryad primary article DOI is in the .bib but the OpenAlex hits are not DOI-verified against Crossref; Brooke 1875 has no DOI.  
  evidence: `datasets/thompson_1917/t1917-031/t1917_031_the_shapes_of_horns_and_of_tee_reference.json` L0–0; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L120–139; `datasets/thompson_1917/t1917-031/scout_manifest.json` L0–0
- **Code-to-model traceability: 70** — The module docstring maps E1-E5 to Thompson's pages; each model function cites its equation id; reference.json carries the thompson_locus for each equation; the Lean file cites pages and equation ids. Deductions: the figure-digitisation steps (sector choice, farthest-point rule) are documented in reference.json but not tied to any Thompson procedure (he measured whorl breadth ratios, Ch. XI p. 558), and verdict thresholds are asserted in reference.json without derivation.  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L1–52; `morphoeval/math_kg_loops/thompson1917-t1917-031.yaml` L0–0
- **Reproducibility package: 60** — Deterministic (seed 1917 used only in the self-test synthetic controls; the fit itself is deterministic), --self-test passes in ~1.3 s, full run ~3 s, inputs SHA-verified at runtime, outputs written to documented paths, invocation documented. Deductions: dependency versions are not pinned per script (requirements.txt lower bounds only), the checkpoint commit does not include the v1.0.1 list-API change, and skimage emits deprecation warnings that will break on 0.28.  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L475–540; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L147–158

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor from a different provider — this round is builder-scored'}
- {'id': 'R2', 'item': "Analyse the real Ovis ammon specimen photograph (HSV segmentation of the horn sheath) and Thompson's Capra sinaitica figure so that two species and one real specimen are fitted"}
- {'id': 'R3', 'item': "Add the projection-invariant gnomonic test (E4): inner edge and breadth profile along rays from the fitted pole; compare k_breadth with k_edge; report the implied cone semi-angle beta for Thompson's alpha = 75 deg"}
- {'id': 'R4', 'item': 'Add uncertainty: bootstrap CI on alpha and k, KS test on standardised residuals, relative RMSE; add synthetic positive/negative controls and a shuffled-angle ablation to the measured rows'}
- {'id': 'R5', 'item': 'Replace the alpha-only verdict logic with one that separates (a) spiral-vs-null model comparison, (b) alpha/beta identifiability and (c) gnomonic self-similarity, and label the Dryad table as a non-shape analogue in the result'}
- {'id': 'R6', 'item': 'Trim the mandible junction from the Fig. 320 ventral sector automatically (monotone ln rho criterion) instead of relying on the hand-set sector end'}
- {'id': 'R7', 'item': "Render a beta sweep (acute vs blunt cone, Thompson's O. poli vs O. ammon contrast) alongside the alpha sweep"}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Ch. XIII — bovid horns as logarithmic spirals wound on a cone (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `78b39928d6410ad23f00651145d4b7faa4e8de550bae97e1a0f3272bacd16b08`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-031/dryad_qz612jmb4_dataset_Tidiereetal2020.txt` (`8e13941af7964f2d…`), `datasets/thompson_1917/t1917-031/ovis_ammon_argali_skull_horns_specimen.jpg` (`f8c17c09c7ebe13e…`), `datasets/thompson_1917/t1917-031/paper_citations.bib` (`eb8c1ca49ca4877d…`), `datasets/thompson_1917/t1917-031/scout_manifest.json` (`c6e9d7fbfcf1ea70…`), `datasets/thompson_1917/t1917-031/t1917_031_the_shapes_of_horns_and_of_tee_reference.json` (`64826ef38cca003b…`), `datasets/thompson_1917/t1917-031/thompson_1917_fig318_ram_horns_diagram.jpg` (`18f820edfb2de26c…`), `datasets/thompson_1917/t1917-031/thompson_1917_fig319_capra_sinaitica_head.jpg` (`55131cbad1f81417…`), `datasets/thompson_1917/t1917-031/thompson_1917_fig320_ovis_ammon_head.jpg` (`4d6a6093ce49a70a…`)
- `datasets/thompson_1917/t1917-031/scout_manifest.json` — artifact_sha256 `c6e9d7fbfcf1ea70…`, 12,723 B, 
- `datasets/thompson_1917/t1917-031/t1917_031_metrics.json` — artifact_sha256 `0fd36968bffa49b2…`, 5,953 B, 
- `datasets/thompson_1917/t1917-031/t1917_031_the_shapes_of_horns_and_of_tee_reference.json` — artifact_sha256 `64826ef38cca003b…`, 12,893 B, 
- `docs/media/t1917_031_the_shapes_of_horns_and_of_tee_comparison.png` — artifact_sha256 `32b7ee29bfb996ac…`, 406,769 B, 1760×990 px
- `morphoeval/math_kg_loops/thompson1917-t1917-031.yaml` — artifact_sha256 `9c695b9a4114bace…`, 7,997 B, 

| measurement | value | reference | method |
|---|---:|---|---|
| fig320 projected spiral angle alpha_proj | 85.91 deg | 75.0 | free-pole nonlinear least-squares log-spiral fit to the digitised outer horn edge of Thompson (1917) Fig. 320; alpha = atan(1/|k|) |
| fig320 whorl ratio R = exp(2 pi |k|) | 1.567 ratio | 5.385 | E3 from the fitted pitch; baseline R = 1 is the circle null |
| fig320 Pearson r of ln(rho) vs theta | 0.9097 r | 1.0 | linearity of ln(rho) in growth angle; 1.0 = exact log spiral |
| fig320 log-spiral RMSE | 16.71 px | — | radial RMSE of the fit over 550 edge points (278 deg sweep); relative RMSE 0.046 |
| fig320 delta AIC circle minus log-spiral | 246.1 AIC | 10.0 | alternative-model comparison vs the circle null (no gnomonic growth); >= 10 counts as decisive |
| fig320 delta AIC Archimedean minus log-spiral | 12.0 AIC | 10.0 | alternative-model comparison vs an Archimedean (arithmetic) spiral |

## Environment at checkpoint


Cloud loop thompson1917-t1917-031 (slot 225) on branch cursor/t1917-031-bovid-horn-spiral-1bd0, based on origin/main fc7454cd8. Environment: Python 3.12.3, numpy/scipy/scikit-image/Pillow/ matplotlib from requirements.txt, Lean 4.35.0 (elan). Script v1.0.1-pass1 (measure_morphoeval list API added on top of the committed v1.0.0 baseline; otherwise identical).
