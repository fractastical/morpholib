# MorphoEval report — audit `2026-10-28-t1917-010-the-rate-of-growth-audit`

Rubric **0.1.0** · checkpoint `7295001d7` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) regeneration as a lawful growth process — Durbin/Ellis tadpole tail curves (`thompson/claims/t1917-010_the-rate-of-growth.py`) | 70 | 35 | 40 | 55 | 78 | 74 | 52 | 60.0 | 42.5 | 66.5 | 55.0 | **55.0** | partially faithful |

## Findings and remediation

### Thompson (1917) regeneration as a lawful growth process — Durbin/Ellis tadpole tail curves  —  overall 55.0 (partially faithful)

Pass 1 baseline. The script loads four SHA-256-pinned tables digitised from Thompson's own pages, fits the growth laws Thompson discusses (E2 hyperbolic gloss vs E3 exponential approach; E4 sigmoid), recovers the Fig. 37 velocity exponent (n = 1.31, not 2), confirms Ellis's amount-removed scaling (r = 0.993 day 9, 0.9997 plateau), replicates the Fig. 40 isochrones (beta = 0.35 vs 1/3) and writes result.json (supported, two caveats). It does not sweep parameter regimes, run any negative or ablation control, use the vendored specimen photograph, or report KS / permutation statistics, and the Dryad CSV digest pin breaks under .gitattributes LF normalisation on a fresh clone.

- **Model fidelity to source: 70** — The exact laws Thompson states are implemented (E2 L_inf - c/t from V t^2 = const; E3 exponential approach from Fig. 37; E4 sigmoid; E5 L_n = p_n R; E6 cubic-parabola isochrones) and mirrored by 17 kernel-checked Lean theorems. The script does not yet fit the free-exponent power law to the integrated curve, so Thompson's n ~ 2 claim is only tested on the increment series.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L136–153; `thompson/claims/t1917-010_the-rate-of-growth.py` L186–195; `thompson/claims/t1917-010_the-rate-of-growth.py` L239–254; `morphoeval/lean/loops/Thompson1917_T1917_010.lean` L1–215; `thompson/claims/t1917-010_the-rate-of-growth.py` L175–184
- **Morphological realism / specimen match: 35** — A CC-BY lateral photograph of Ellis's species is vendored and the p. 148 page scan is shown in the figure, but the specimen is never segmented or measured; no tail-fraction morphometric is compared with Thompson's 65 % (Ellis) / 62 % (Durbin) tail proportion, and the GIF tadpoles are schematic.  
  evidence: `datasets/thompson_1917/t1917-010/lithobates_clamitans_tadpole_specimen.jpg` L0–0; `thompson/claims/t1917-010_the-rate-of-growth.py` L364–369; `thompson/claims/t1917-010_the-rate-of-growth.py` L375–386
- **Parameter expressivity: 40** — Fits expose k, L_inf, t0, n and beta but no panel sweeps a parameter across its regime (e.g. decay exponent n from 0.5 to 3, or amputation level R toward the 75 % critical point) to show monotone control of the output.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L428–442; `thompson/claims/t1917-010_the-rate-of-growth.py` L239–254
- **Morphometric validation: 55** — Quantitative comparisons with the printed tables are made (r, RMSE, exponent, inflection day, plateau fraction) and exported by measure_morphoeval, but there is no negative/ablation control (e.g. permutation of amount-removed labels, constant-velocity null), no KS or coefficient-of-variation statistic, and no specimen-derived morphometric.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L206–223; `thompson/claims/t1917-010_the-rate-of-growth.py` L510–526; `thompson/claims/t1917-010_the-rate-of-growth.py` L445–464
- **Source and data traceability: 78** — IA scan identifier with page-offset rule, five vendored page scans with original download digests, OCR excerpt, Durbin/Ellis DOIs, Dryad DOI + mismatch rationale, iNaturalist observation URL and licence are all recorded; the Dryad CSV digest, however, is pinned on CRLF bytes that git normalises to LF, so the pin is not reproducible from a clone.  
  evidence: `datasets/thompson_1917/t1917-010/t1917_010_the_rate_of_growth_reference.json` L60–110; `datasets/thompson_1917/t1917-010/scout_manifest.json` L1–60; `datasets/thompson_1917/t1917-010/dryad_n6n6495_fitzpatrick_et_al_data.csv` L1–38
- **Code-to-model traceability: 74** — Each loader names the Thompson page and original Durbin/Ellis table; each law docstring cites the equation id and page; math_kg yaml links six laws to Lean declarations and the script. Panel titles cite figure numbers. No per-function mapping table in the README yet.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L82–131; `morphoeval/math_kg_loops/thompson1917-t1917-010.yaml` L1–80
- **Reproducibility package: 52** — --self-test with numeric guards, fixed seed and deterministic measure_morphoeval exist, but the Dryad digest pin fails after git LF normalisation (fresh clone -> verify_sha256 raises), no dependency pins are stated in the header, and GIF label text is clipped.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L529–549; `thompson/claims/t1917-010_the-rate-of-growth.py` L64–68; `thompson/claims/t1917-010_the-rate-of-growth.py` L388–413

Remediation:
- Normalise the Dryad CSV to LF, re-pin its digest and record the original download digest separately so verify_sha256 passes from a fresh clone
- Add a parameter-regime sweep: decay exponent n in [0.5, 3] -> RMSE of the integrated law on Exp. II (minimum vs Thompson's n = 2), and amputation level R -> isochrone time with the 75 % asymptote
- Add negative/ablation controls: exact permutation test of Ellis amount-removed labels, constant-velocity (linear) null and R-independent null, with p-values
- Segment the vendored L. clamitans specimen (threshold + principal axis + width profile) to measure tail fraction of total length and compare with Thompson's 65 % / 62 %; overlay the mask on the figure
- Report KS statistics on standardised residuals, coefficient of variation of the restored fraction, and fit a free-exponent generalised power law to the integrated curve
- Fix GIF label clipping and the Exp. IV negative y-axis floor; add dependency pins in the script header
- Write superseding Pass 2 audit after Step 12

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) regeneration as a lawful growth process — Durbin/Ellis tadpole tail curves
- **spec_fingerprint** (semantic model + inputs): `de10fcf1506d2328367848e4d263a6f76c4a536627abf0558ae32051c5aa7ba2`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-010/lithobates_clamitans_tadpole_specimen.jpg` (`4a90d205c4f3ec5e…`), `datasets/thompson_1917/t1917-010/t1917_010_the_rate_of_growth_reference.json` (`8cf23ca99668fed8…`), `datasets/thompson_1917/t1917-010/thompson_p139_durbin_expII_regeneration.csv` (`ee7b4283ace77101…`), `datasets/thompson_1917/t1917-010/thompson_p141_durbin_expII_interpolated.csv` (`b44610d153e0e230…`), `datasets/thompson_1917/t1917-010/thompson_p144_durbin_expIV_regeneration.csv` (`cc4848894d2310d1…`), `datasets/thompson_1917/t1917-010/thompson_p148_ellis_tail_regeneration.csv` (`3b61cc051e28ff22…`), `morphoeval/lean/loops/Thompson1917_T1917_010.lean` (`36af9a4ae4b3f5de…`)
- `datasets/thompson_1917/t1917-010/lithobates_clamitans_tadpole_specimen.jpg` — artifact_sha256 `4a90d205c4f3ec5e…`, 64,739 B, 1024×683 px
- `docs/media/t1917_010_the_rate_of_growth_comparison.png` — artifact_sha256 `e7b927671098f62a…`, 488,733 B, 1950×1170 px

| measurement | value | reference | method |
|---|---:|---|---|
| ellis_day9_pearson_r | 0.993038 dimensionless | — | Pearson r, mm regenerated vs mm removed (n=4) |
| ellis_plateau_pearson_r | 0.999738 dimensionless | — | Pearson r at day 32 |
| ellis_plateau_fraction_mean | 0.46 fraction | — | mean of 44,44,48,48 % |
| ellis_isochrone_beta_mean | 0.346577 dimensionless | — | log-log slope of t_q vs R, q=10..40 |
| durbin_expII_exp_approach_rmse | 0.089353 mm | — | curve_fit L_inf(1-exp(-k(t-t0))) |
| durbin_expII_hyperbolic_rmse | 0.460299 mm | — | curve_fit L_inf - c/t |
| durbin_velocity_power_exponent | 1.311573 dimensionless | — | OLS log V vs log t (Thompson claims 2) |
| durbin_velocity_exp_k | 0.100879 1/day | — | OLS log V vs t |
| durbin_expIV_inflection_day | 6.349334 day | — | Gompertz ln(b)/c (Thompson: ~7) |
| durbin_expIV_gompertz_r | 0.996195 dimensionless | — | Pearson r fitted vs observed |

## Environment at checkpoint


Pass 1 baseline (commit 7295001d7) before Step 12 remediation. Scored against the v1 script as committed: no parameter-regime sweep, no negative/ablation control, no computer-vision use of the vendored specimen, no KS / permutation statistics, and a SHA-256 pin that breaks on fresh clone.
