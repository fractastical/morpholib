# MorphoEval report — audit `2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2`

Rubric **0.1.0** · checkpoint `430ab132` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit -> 2026-10-28-t1917-031-the-shapes-of-horns-and-of-tee-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Ch. XIII — bovid horns as logarithmic spirals wound on a cone (Pass 2) (`thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py`) | 72 | 58 | 70 | 66 | 82 | 74 | 64 | 71.3 | 61.0 | 72.7 | 67.6 | **67.6** | mostly faithful |

## Findings and remediation

### Thompson (1917) Ch. XIII — bovid horns as logarithmic spirals wound on a cone (Pass 2)  —  overall 67.6 (mostly faithful)

Fig. 320 (Ovis ammon drawing, 521 pts, 269 deg): alpha_proj 84.2 deg [83.7, 84.6], R 1.90, r 0.968, RMSE 3.7 %, log spiral beats circle by dAIC 425 and Archimedean by 22; beta implied by alpha = 75 deg is 22 deg. Real Argali horn (HSV sheath mask, 478 pts, 215 deg): alpha_proj 74.7 deg [73.6, 75.6], R 5.57, r 0.995, RMSE 2.6 %, dAIC circle 666, Archimedean +2 — matches Thompson's 75 deg, but dropping the ventral/tip arc makes the remaining arc circle-like (sensitivity range 74.7-87.0 deg). Capra sinaitica half-whorl: alpha not identifiable (53-86 deg across sectors). Gnomonic test fails on Fig. 320 (|k_breadth| 0.32 vs |k_outer| 0.13), inconclusive on the photo. Controls: synthetic alpha 75 recovered to 75.0, circle null -> circle, shuffled ablation r 0.97 -> 0.01. Verdict mixed. Overall 67.6/100.

- **Model fidelity to source: 72** — E1-E4 implemented as Thompson states them; E2 is now used in the right direction (implied beta from k_proj rather than treating alpha_proj as alpha), and the Lean module proves the non-identifiability that this rests on plus the Pass-2 data pins (alpha match, dAIC, E4 failure) with 0 sorry. The gnomonic test implements E4 literally (breadth along rays). Still 2D: no 3D centreline, so beta is inferred, never measured; E5 remains a rendering.  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L85–141; `morphoeval/lean/loops/Thompson1917_T1917_031.lean` L32–153; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L478–549; `datasets/thompson_1917/t1917-031/result.json` L0–0
- **Morphological realism / specimen match: 58** — A real Ovis ammon horn is now segmented (HSV sheath mask excludes the bleached skull) and the fitted spiral tracks its outer edge over 215 deg with 2.6 % RMSE; the overlay on Thompson's Fig. 320 tracks the dorsal/posterior edge; the E2 helicoid is rendered with the Fig.-320-implied beta. Deductions: n = 1 real specimen in perspective view, the spiral estimate on it is driven by the tip arc (sensitivity flag), the Capra figure is a halftone whose half-whorl cannot pin alpha, and the visual match of the helicoid to the photograph is qualitative.  
  evidence: `docs/media/t1917_031_the_shapes_of_horns_and_of_tee_comparison.png` L0–0; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L207–219; `datasets/thompson_1917/t1917-031/t1917_031_metrics.json` L0–0
- **Parameter expressivity: 70** — alpha sweep (65-85) and beta sweep at alpha = 75 (15-90) are rendered and tabulated (model_regime_table alpha x beta -> k_proj, R, alpha_proj); the self-test checks the monotone effect of beta; --bootstrap is a CLI parameter; trim_ends / sectors / thresholds are data-driven from reference.json. Breadth ratio and handedness are still fixed.  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L613–661; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L805–824
- **Morphometric validation: 66** — Per source: alpha_proj with bootstrap 95 % CI, R, Pearson r, RMSE (abs and relative), KS p-value, dAIC vs two nulls, implied beta, k_breadth vs k_outer; plus positive/negative controls, a shuffled ablation, sector-sensitivity ranges and a species contrast — 23 measurement rows with reference values. Deductions: the bootstrap CI (+/-0.5 deg) understates the model uncertainty revealed by the sector sensitivity (up to 12 deg); the gnomonic test is pole-dependent and inconclusive on the photo; the KS test rejects normality on Fig. 320 (ring corrugations); n = 1 real specimen.  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L439–471; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L552–583; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L664–709; `datasets/thompson_1917/t1917-031/result.json` L0–0
- **Source and data traceability: 82** — Unchanged pins (IA leaf mapping, 5 SHA-256 inputs re-verified at runtime, Commons licence/author, Dryad DOI + version + zip digest); the Dryad mismatch is now carried through to result.json as an explicit dataset_mismatch_flag, and the Dryad table is summarised (log-log slopes) so its content is visible. The .gitattributes keeps the vendored table byte-exact. Deductions as in v1 (no Crossref re-verification of secondary DOIs; Brooke 1875 lacks a DOI).  
  evidence: `datasets/thompson_1917/t1917-031/t1917_031_the_shapes_of_horns_and_of_tee_reference.json` L0–0; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L585–610
- **Code-to-model traceability: 74** — Docstring maps E1-E5 and tests T1-T4 to pages; every fit/segmentation step is a named function with its equation id; reference.json carries the loci; the math-KG YAML links five laws to Lean declarations and script line ranges. Deductions: the ray-based inner-edge rule and the trimming rule are the script's own constructions, not Thompson's whorl-breadth procedure (Ch. XI p. 558), and this is only stated in comments.  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L1–64; `morphoeval/math_kg_loops/thompson1917-t1917-031.yaml` L0–0
- **Reproducibility package: 64** — Deterministic (seed 1917 for bootstrap/controls), --self-test (24 assertions incl. a trim test, E2 inversion, controls, real-data sanity) passes, full run 11 s, inputs SHA-verified, documented invocation with --bootstrap/--no-gif, checkpoint commit 430ab132 contains the scored script. Deductions: no per-script dependency pins; skimage deprecation path; the ephemeral nanopub signing key is loop-local (sign-only, not published).  
  evidence: `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L949–992; `thompson/claims/t1917-031_the-shapes-of-horns-and-of-teeth-or-tusks-with-a.py` L995–1017

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor (different provider) — still builder-scored'}
- {'id': 'R8', 'item': '3D centrelines (MorphoSource CT or photogrammetry) for >= 3 Argali / Ovis specimens to measure alpha and beta separately and test the cone/torsion component'}
- {'id': 'R9', 'item': 'Orthographic lateral photographs of isolated horn sheaths (museum specimens) so the sector-sensitivity of alpha_proj disappears and the breadth test becomes conclusive'}
- {'id': 'R10', 'item': 'Replace the ray-based breadth test with a pole-free local test (breadth vs radius of curvature along the outer edge)'}
- {'id': 'R11', 'item': 'Fit the Capra sinaitica horn only with a constrained pole (e.g. on the horn-base normal) or use a full-whorl Capra falconeri specimen so the goat-vs-sheep contrast is identifiable'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Ch. XIII — bovid horns as logarithmic spirals wound on a cone (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `860e7e35af4127f92fad446ceb9389d004fdb7dfa2fe98cf3f8ad3f9657429c2`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-031/dryad_qz612jmb4_dataset_Tidiereetal2020.txt` (`8e13941af7964f2d…`), `datasets/thompson_1917/t1917-031/ovis_ammon_argali_skull_horns_specimen.jpg` (`f8c17c09c7ebe13e…`), `datasets/thompson_1917/t1917-031/paper_citations.bib` (`eb8c1ca49ca4877d…`), `datasets/thompson_1917/t1917-031/scout_manifest.json` (`c6e9d7fbfcf1ea70…`), `datasets/thompson_1917/t1917-031/t1917_031_the_shapes_of_horns_and_of_tee_reference.json` (`0395050a08978a58…`), `datasets/thompson_1917/t1917-031/thompson_1917_fig318_ram_horns_diagram.jpg` (`18f820edfb2de26c…`), `datasets/thompson_1917/t1917-031/thompson_1917_fig319_capra_sinaitica_head.jpg` (`55131cbad1f81417…`), `datasets/thompson_1917/t1917-031/thompson_1917_fig320_ovis_ammon_head.jpg` (`4d6a6093ce49a70a…`)
- `datasets/thompson_1917/t1917-031/result.json` — artifact_sha256 `b497cdf5262491b9…`, 6,709 B, 
- `datasets/thompson_1917/t1917-031/t1917_031_metrics.json` — artifact_sha256 `348b7f53a93ee9f8…`, 24,322 B, 
- `datasets/thompson_1917/t1917-031/t1917_031_the_shapes_of_horns_and_of_tee_reference.json` — artifact_sha256 `0395050a08978a58…`, 15,637 B, 
- `docs/media/t1917_031_the_shapes_of_horns_and_of_tee_comparison.png` — artifact_sha256 `832658753c97c6a0…`, 971,711 B, 2000×1300 px
- `morphoeval/math_kg_loops/thompson1917-t1917-031.yaml` — artifact_sha256 `095adf2f45493d44…`, 8,428 B, 

| measurement | value | reference | method |
|---|---:|---|---|
| Fig.320 Ovis ammon: projected spiral angle alpha_proj | 84.15 deg | 75.0 | free-pole NLS log-spiral fit to the digitised outer edge (521 pts, 269 deg sweep); bootstrap 95% CI [83.7, 84.5] deg |
| Fig.320 Ovis ammon: whorl ratio R = exp(2 pi |k|) | 1.903 ratio | 5.385 | E3 from the fitted pitch; baseline R = 1 is the circle null |
| Fig.320 Ovis ammon: Pearson r of ln(rho) vs theta | 0.9679 r | 1.0 | linearity of ln rho in growth angle; KS p-value of standardised residuals vs N(0,1) = 0.00282; RMSE 13.6 px (3.7% of mean radius) |
| Fig.320 Ovis ammon: delta AIC circle minus log-spiral | 425.1 AIC | 10.0 | alternative-model comparison vs the circle null (no gnomonic growth); >= 10 decisive |
| Fig.320 Ovis ammon: delta AIC Archimedean minus log-spiral | 21.7 AIC | 10.0 | alternative-model comparison vs an Archimedean spiral |
| Fig.320 Ovis ammon: cone semi-angle beta implied by Thompson's alpha = 75 deg | 22.5 deg | — | E2 inverted: sin(beta) = k_proj tan(75 deg); identifiability caveat (Lean alpha_beta_not_identifiable_from_projection) |
| Fig.320 Ovis ammon: gnomonic test k_breadth vs k_outer (fixed pole) | -0.3206 1/rad | -0.131 | log-linear slope of breadth along rays; k_outer = -0.1310; E4 predicts equality (|diff|/|k_outer| <= 0.25); conclusive |
| Fig.319 Capra sinaitica: projected spiral angle alpha_proj | 86.06 deg | 70.0 | free-pole NLS log-spiral fit to the digitised outer edge (263 pts, 334 deg sweep); bootstrap 95% CI [85.5, 86.9] deg |
| Fig.319 Capra sinaitica: whorl ratio R = exp(2 pi |k|) | 1.542 ratio | 5.385 | E3 from the fitted pitch; baseline R = 1 is the circle null |
| Fig.319 Capra sinaitica: Pearson r of ln(rho) vs theta | 0.8895 r | 1.0 | linearity of ln rho in growth angle; KS p-value of standardised residuals vs N(0,1) = 0.0328; RMSE 29.4 px (7.6% of mean radius) |
| Fig.319 Capra sinaitica: delta AIC circle minus log-spiral | 104.2 AIC | 10.0 | alternative-model comparison vs the circle null (no gnomonic growth); >= 10 decisive |
| Fig.319 Capra sinaitica: delta AIC Archimedean minus log-spiral | -5.0 AIC | 10.0 | alternative-model comparison vs an Archimedean spiral |
| Fig.319 Capra sinaitica: cone semi-angle beta implied by Thompson's alpha = 75 deg | 14.9 deg | — | E2 inverted: sin(beta) = k_proj tan(75 deg); identifiability caveat (Lean alpha_beta_not_identifiable_from_projection) |
| Argali specimen photo: projected spiral angle alpha_proj | 74.72 deg | 75.0 | free-pole NLS log-spiral fit to the digitised outer edge (478 pts, 215 deg sweep); bootstrap 95% CI [73.6, 75.4] deg |
| Argali specimen photo: whorl ratio R = exp(2 pi |k|) | 5.566 ratio | 5.385 | E3 from the fitted pitch; baseline R = 1 is the circle null |
| Argali specimen photo: Pearson r of ln(rho) vs theta | 0.9953 r | 1.0 | linearity of ln rho in growth angle; KS p-value of standardised residuals vs N(0,1) = 0.343; RMSE 10.0 px (2.6% of mean radius) |
| Argali specimen photo: delta AIC circle minus log-spiral | 666.2 AIC | 10.0 | alternative-model comparison vs the circle null (no gnomonic growth); >= 10 decisive |
| Argali specimen photo: delta AIC Archimedean minus log-spiral | 2.0 AIC | 10.0 | alternative-model comparison vs an Archimedean spiral |
| Argali specimen photo: gnomonic test k_breadth vs k_outer (fixed pole) | -0.0157 1/rad | -0.2166 | log-linear slope of breadth along rays; k_outer = -0.2166; E4 predicts equality (|diff|/|k_outer| <= 0.25); INCONCLUSIVE (pole inside sheath or no exponential trend in breadth) |
| positive control: synthetic alpha = 75 deg spiral, recovered alpha | 74.89 deg | 75.0 | noisy synthetic log spiral, unknown pole; pole error 0.3 px |
| negative control: noisy circle, fitted |k| | 0.0002 1/rad | 0.0 | circle null must be classified as circle by AIC (best = circle) |
| ablation: Fig.320 radii shuffled against angles, Pearson r | 0.0536 r | 0.0 | spiral structure must vanish under shuffling (real r = 0.968) |
| species contrast: alpha_proj Capra minus Ovis | 6.62 deg | -5.0 | Thompson predicts goats more open (smaller alpha) than Ovis ammon: negative value supports |

## Environment at checkpoint


Cloud loop thompson1917-t1917-031 (slot 225), branch cursor/t1917-031-bovid-horn-spiral-1bd0 on origin/main fc7454cd8. Python 3.12.3, numpy/scipy/scikit-image/Pillow/matplotlib per requirements.txt, Lean 4.35.0. Script v2.0.0-pass2; full run 11 s, --self-test 11 s (bootstrap 20 per source).
