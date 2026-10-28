# MorphoEval report — audit `2026-10-28-t1917-010-the-rate-of-growth-audit-v2`

Rubric **0.1.0** · checkpoint `6f3b0adec` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-010-the-rate-of-growth-audit -> 2026-10-28-t1917-010-the-rate-of-growth-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) regeneration as a lawful growth process — Durbin/Ellis tadpole tail curves (`thompson/claims/t1917-010_the-rate-of-growth.py`) | 84 | 68 | 80 | 80 | 88 | 84 | 80 | 82.7 | 72.5 | 83.7 | 78.9 | **78.9** | mostly faithful |

## Findings and remediation

### Thompson (1917) regeneration as a lawful growth process — Durbin/Ellis tadpole tail curves  —  overall 78.9 (mostly faithful)

Pass 2 remediated assessment. All Pass 1 remediation items are closed: a decay-exponent sweep shows the integrated power law is best at n = 1.2 (free fit 1.15) and 5x worse at Thompson's n = 2; an amputation-level sweep through the Fig. 40 law shows the pure cubic parabola already fits the four series (RMSE 0.11 d) and that the 75 % divergence is not demanded by data below 57 % (free gamma = 0.00); exact permutation tests give p = 1/24 for both the day-9 and plateau amount-removed correlations; R-independent and constant-velocity nulls are 4.8x / 19x / 13.5x worse than the fitted laws; KS p = 0.47 / 0.92 on residuals; the vendored L. clamitans specimen is segmented (tail fraction 0.655 vs Thompson's 0.65 / 0.62) and overlaid; every exported measurement that has a literature value carries reference_value; the Dryad digest is pinned on LF bytes so verify_sha256 passes from a clone; GIF clipping and the Exp. IV axis floor are fixed. Verdict: supported, with two documented caveats against Thompson's own glosses (V t^2 exponent; extrapolated 75 % asymptote).

- **Model fidelity to source: 84** — Every law Thompson states (E1-E7) is implemented as written, now including the generalised power decay with free exponent and the asymptotic isochrone family; the Lean module kernel-checks E1, E2, E3, E5, E6, E7 with 0 sorry. Remaining gap: Thompson's 'latent period' before day 3 is a fitted t0, not modelled.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L147–165; `thompson/claims/t1917-010_the-rate-of-growth.py` L271–296; `thompson/claims/t1917-010_the-rate-of-growth.py` L197–206; `morphoeval/lean/loops/Thompson1917_T1917_010.lean` L1–215; `thompson/claims/t1917-010_the-rate-of-growth.py` L186–195
- **Morphological realism / specimen match: 68** — The vendored CC-BY photograph of Ellis's species is segmented, the mask and body/tail boundary are overlaid on the figure, and the measured tail fraction (0.655) matches Thompson's printed 65 %; the p. 148 page scan sits beside the Fig. 40 replica. Limits: one specimen, a schematic GIF, and no image of a regenerating tail.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L365–435; `thompson/claims/t1917-010_the-rate-of-growth.py` L578–600; `datasets/thompson_1917/t1917-010/lithobates_clamitans_tadpole_specimen.jpg` L0–0; `thompson/claims/t1917-010_the-rate-of-growth.py` L609–648
- **Parameter expressivity: 80** — Two sweeps with visible monotone control: RMSE vs decay exponent n (minimum 1.2, Thompson's 2 marked) and t_20 vs amputation level R for gamma in {0, 0.25, 0.5, 1} with the 75 % asymptote; the Fig. 40 replica shows all four q-isochrones. Not swept: k / t0 of the exponential law.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L276–291; `thompson/claims/t1917-010_the-rate-of-growth.py` L298–323; `thompson/claims/t1917-010_the-rate-of-growth.py` L540–576
- **Morphometric validation: 80** — Fits are compared against Thompson's own printed tables with r, RMSE, exact permutation p-values, null-model RMSE ratios, KS statistics and the plateau CV; eight measurements carry explicit reference values (Thompson's n = 2, day 7, 1/3, 65 %, 0.46, r = 1). Weakness: n = 4 series for the Ellis tests (permutation floor p = 1/24) and n = 7-10 points per Durbin series.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L326–362; `thompson/claims/t1917-010_the-rate-of-growth.py` L775–813; `datasets/thompson_1917/t1917-010/result.json` L1–80; `thompson/claims/t1917-010_the-rate-of-growth.py` L326–333
- **Source and data traceability: 88** — IA scan + page-offset rule, five page scans with original download digests, OCR excerpt, Durbin/Ellis DOIs and the Durbin original scan used to identify the tables, Dryad DOI with mismatch rationale, iNaturalist observation/photo ids and licence; all 12 digests now reproducible from a clone.  
  evidence: `datasets/thompson_1917/t1917-010/t1917_010_the_rate_of_growth_reference.json` L1–200; `datasets/thompson_1917/t1917-010/scout_manifest.json` L1–90; `thompson/claims/t1917-010_the-rate-of-growth.py` L75–79
- **Code-to-model traceability: 84** — Loaders cite page and original table; each law carries its equation id and page; panels cite figure numbers; the math KG yaml maps six laws to Lean declarations and the script; the dataset README tabulates equations E1-E7. No line-level mapping table for the Pass 2 statistics functions.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L93–144; `morphoeval/math_kg_loops/thompson1917-t1917-010.yaml` L1–80; `datasets/thompson_1917/t1917-010/README.md` L1–70
- **Reproducibility package: 80** — Header pins the known-good environment, deterministic fits, --self-test with 25 guards including Pass 2 controls and CV, --no-gif switch, digest verification reproducible from a clone; GIF and axis defects fixed. No lockfile hash of the exact wheel set is recorded in result.json.  
  evidence: `thompson/claims/t1917-010_the-rate-of-growth.py` L22–33; `thompson/claims/t1917-010_the-rate-of-growth.py` L816–855; `thompson/claims/t1917-010_the-rate-of-growth.py` L622–626

Remediation:
- Add a second CC-licensed specimen (ideally a regenerating tadpole tail) to move specimen-match beyond a single photograph
- Digitise Ellis's per-individual data from the J. Exp. Zool. scan to raise n beyond the four printed series means
- Record the installed wheel versions in result.json for a byte-level environment pin

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) regeneration as a lawful growth process — Durbin/Ellis tadpole tail curves
- **spec_fingerprint** (semantic model + inputs): `bffecc9effec0ed61f93d27b11061055edc430bd0988d9f85d96335fb77a665c`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-010/lithobates_clamitans_tadpole_specimen.jpg` (`4a90d205c4f3ec5e…`), `datasets/thompson_1917/t1917-010/t1917_010_the_rate_of_growth_reference.json` (`afba32ab058719d6…`), `datasets/thompson_1917/t1917-010/thompson_p139_durbin_expII_regeneration.csv` (`ee7b4283ace77101…`), `datasets/thompson_1917/t1917-010/thompson_p141_durbin_expII_interpolated.csv` (`b44610d153e0e230…`), `datasets/thompson_1917/t1917-010/thompson_p144_durbin_expIV_regeneration.csv` (`cc4848894d2310d1…`), `datasets/thompson_1917/t1917-010/thompson_p148_ellis_tail_regeneration.csv` (`3b61cc051e28ff22…`), `morphoeval/lean/loops/Thompson1917_T1917_010.lean` (`36af9a4ae4b3f5de…`)
- `datasets/thompson_1917/t1917-010/lithobates_clamitans_tadpole_specimen.jpg` — artifact_sha256 `4a90d205c4f3ec5e…`, 64,739 B, 1024×683 px
- `docs/media/t1917_010_the_rate_of_growth_comparison.png` — artifact_sha256 `cd24ca6582e91fdb…`, 734,418 B, 1950×1755 px

| measurement | value | reference | method |
|---|---:|---|---|
| ellis_day9_pearson_r | 0.993038 dimensionless | 1.0 | Pearson r, mm regenerated vs mm removed (n=4); exact permutation control p=0.042 |
| ellis_plateau_pearson_r | 0.999738 dimensionless | 1.0 | Pearson r at day-32 plateau; R-independent null RMSE / proportional-model RMSE = 19.2 |
| ellis_plateau_fraction_mean | 0.46 fraction | 0.46 | mean of 44,44,48,48 % with CV=0.043 |
| ellis_isochrone_beta_mean | 0.346577 dimensionless | 0.3333 | log-log slope of t_q vs R, q=10..40 (cubic parabola <=> 1/3) |
| durbin_expII_exp_approach_rmse | 0.089353 mm | — | curve_fit L_inf(1-exp(-k(t-t0))); KS resid D=0.30 p=0.47; constant-velocity null RMSE x13.5 |
| durbin_expII_hyperbolic_rmse | 0.460299 mm | — | curve_fit L_inf - c/t (Thompson's V t^2 = const integrated) vs exponential alternative |
| durbin_velocity_power_exponent | 1.311573 dimensionless | 2.0 | OLS log V vs log t on Thompson's interpolated increments; sweep RMSE minimum at n=1.20 |
| durbin_velocity_exp_k | 0.100879 1/day | — | OLS log V vs t (Fig. 37 straight line), r=-0.991 |
| durbin_expIV_inflection_day | 6.349334 day | 7.0 | Gompertz ln(b)/c; argmax of printed daily increments = day 7 |
| durbin_expIV_gompertz_r | 0.996195 dimensionless | — | Pearson r fitted vs observed (10 points) |
| sweep_R_isochrone_gamma | 0.003165 dimensionless | — | asymptote exponent in t_q = a R^(1/3) (75/(75-R))^gamma, monotone=True |
| specimen_tail_fraction | 0.655462 fraction | 0.65 | CV threshold + principal axis + width profile on vendored L. clamitans photo (tail = terminal monotone taper; boundary at half-decay of the posterior body bulge) |

## Environment at checkpoint


Pass 2 after Step 12 remediation of every Pass 1 remediation item: decay-exponent and amputation-level sweeps, exact permutation + constant-velocity + R-independent controls, specimen CV tail fraction, KS and CV statistics, reference values on measurements, LF-normalised Dryad digest, GIF/axis fixes.
