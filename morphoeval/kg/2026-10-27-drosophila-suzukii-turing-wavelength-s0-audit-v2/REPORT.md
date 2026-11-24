# MorphoEval report — audit `2026-10-27-drosophila-suzukii-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-suzukii-turing-wavelength-s0-audit -> 2026-10-27-drosophila-suzukii-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila suzukii Turing onset wavelength at 2.046 µm (0 Ma, detail scale) (`models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py`) | 88 | 82 | 90 | 88 | 91 | 88 | 90 | 88.7 | 84.2 | 89.8 | 87.2 | **87.2** | faithful |

## Findings and remediation

### Drosophila suzukii Turing onset wavelength at 2.046 µm (0 Ma, detail scale)  —  overall 87.2 (faithful)

Pass 2. Eigenvector-initialized mode growth matches analytical sigma(k) with Pearson r = 0.9999999995 and RMSE = 4.16e-4 on the real-eigenvalue band. Diffusivity sweep r = 0.997 and is monotonic. Equal-diffusion and subobscura std(u) are 1e-15, drawn flat on the focal color scale. A stripe seed on the 24.552 µm window locks to spacing 2.046 µm (CV 0.056). The noise-seeded field remains a labyrinth (CV 0.37, KS 0.63). Specimen Otsu metrics differ from their priors. lambda_c = 2.046e-6 m, age 0 Ma. Fastest-growing wavelength is 2.116 µm, slightly longer than onset, as expected just above threshold (margin > 1 at diffusion ratio 12.6).

- **Model fidelity to source: 88** — The script implements Schnakenberg kinetics, the onset formula k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), and sigma(k) from the reaction-diffusion matrix. D_u is solved so lambda_c = 2.046 µm exactly (self-test tolerance 1e-9). Numerical sigma now follows the least-stable eigenvector, which is the comparison the linear law requires. The rates themselves are calibrated, not taken from a yellow/ebony measurement, so this is not a 95.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L88–114; `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L141–183; `morphoeval/lean/loops/Loop_3207_drosophila_suzukii_turing_wavelength_s0.lean` L28–60
- **Morphological realism / specimen match: 82** — The plate (male wing spot, female ovipositor, 1 mm bar) sits beside the suzukii labyrinth, a flat subobscura field on the same color scale, and a longer-wavelength takahashii field. The null is visibly flat rather than a stretched roundoff pattern. The GIF is 12 frames at 960×640 and ends on the adult plate, so the fitness comparison sees the animal; the carrier itself is a microstructure, not a second portrait of the fly.  
  evidence: `docs/media/drosophila_suzukii_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L386–400; `docs/media/drosophila_suzukii_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Scaling both diffusivities by s moves wavelength as sqrt(s). Six scales from 0.81 to 1.96 give sweep Pearson r = 0.997 against the fastest PDE mode, and the analytical curve is strictly increasing. Takahashii is the s = 1.44 point (lambda_c 2.4552 µm). Subobscura removes the band by f_u < 0 without touching D_u. Equal diffusion is a separate off-switch.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L226–259; `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L569–592
- **Morphometric validation: 88** — Dispersion Pearson r = 0.9999999995, RMSE = 4.16e-4 per time unit (reference for r is the null of 0, not a copy of the result). Stripe-seed mean spacing is 2.046 µm with CV 0.056, against the fastest-growing wavelength 2.116 µm, so the value is not the reference. Labyrinth CV stays 0.37 and KS stays 0.63; that gap is reported, not tuned away. Specimen foreground 0.203 vs prior 0.40, melanin 0.028 vs 0.10, left-minus-right darkness -5.28 vs 0, abdominal troughs 7 vs 6, coherency 0.375 vs 0.50. Ablation std ratio is about 1e-15 versus a reference of 1.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L594–687; `datasets/drosophila_suzukii_turing_wavelength_s0/drosophila_suzukii_turing_wavelength_s0_metrics.json` L1–80; `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L323–372
- **Source and data traceability: 91** — Seven DOIs were confirmed on OpenAlex during scouting and written to paper_citations.bib. The reference JSON states scale 2.046e-6 m and age 0 Ma, pins the specimen SHA-256, and separates empirical claims from calibrated parameters. Scout manifest holds the OpenAlex and Wikimedia hits. The gallery fallback is the McEvey plate already used as the species image.  
  evidence: `datasets/drosophila_suzukii_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/drosophila_suzukii_turing_wavelength_s0/README.md` L1–25; `datasets/drosophila_suzukii_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 88** — onset_kc2, growth_rate, the eigenvector initializer, and the chronoseries constants are the same objects the Lean file proves as sign facts (trace < 0, det > 0, 12 vs 10 ocelli, f_u numerator -17). The math-kg yaml points at those declarations. Comments name Turing 1952 and Schnakenberg 1979 at the formula, not only in the module docstring.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L88–104; `morphoeval/lean/loops/Loop_3207_drosophila_suzukii_turing_wavelength_s0.lean` L1–62
- **Reproducibility package: 90** — python3 models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py --self-test checks lambda_c, both null controls, Pearson r > 0.98, and the monotonic sweep. The full command rewrites the PNG, the 12-frame GIF, and the metrics JSON from seed 3207. Specimen SHA-256 is checked before any integration. No extra packages beyond the repo scientific stack.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L569–592; `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L468–476

Remediation:
- Still one assessor. A second provider has not scored this audit.
- Still one plate. Abdominal trough count 7 versus a tergite prior of 6 is an ROI method on n = 1, not a population.
- The McEvey plate does not resolve 2.046 µm. Organism-level Otsu numbers are not a carrier measurement.
- Noise-seeded labyrinth spacing KS remains 0.63. The stripe seed shows the onset mode is selectable; it does not make the nonlinear labyrinth periodic.
- D_u, D_v and the takahashii factor 1.44 are calibrated hypotheses, not measured diffusivities.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila suzukii Turing onset wavelength at 2.046 µm (0 Ma, detail scale)
- **spec_fingerprint** (semantic model + inputs): `8f4a40558dfdd1a363ec347c2232dbab086e21c6d2b0baf5b5a36feb603bb029`
- **input_dataset_sha256**: `datasets/drosophila_suzukii_turing_wavelength_s0/drosophila_suzukii_specimen.jpg` (`a4180479fce43c90…`), `datasets/drosophila_suzukii_turing_wavelength_s0/drosophila_suzukii_turing_wavelength_s0_reference.json` (`6dcb8c0b6f42ebaf…`)
- `docs/media/drosophila_suzukii_turing_wavelength_s0_comparison.png` — artifact_sha256 `e240d3ff629f62b8…`, 562,769 B, 1584×984 px
- `docs/media/drosophila_suzukii_turing_wavelength_s0_growth.gif` — artifact_sha256 `af431f926cbdfcaa…`, 2,925,121 B, 960×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_pearson_r | 0.9999999994750002 1 | 0.0 | Pearson r of analytical sigma(k) vs eigenvector-initialized PDE growth on real-eigenvalue modes |
| dispersion_rmse | 0.0004161109110393154 1/tu | 0.0 | RMSE of sigma(k) on the same real-eigenvalue band |
| stripe_spacing_cv | 0.37092840999770954 1 | 0.0 | CV of mid-row activator peak spacings on the saturated suzukii field |
| stripe_spacing_ks | 0.6273324427089965 1 | 0.0 | KS distance of labyrinth peak spacings vs Normal(lambda_sigma, sample sd) |
| stripe_seed_spacing_mean_um | 2.046 µm | 2.115864226653975 | Mean peak spacing of a critical-mode stripe seed versus the fastest-growing wavelength |
| onset_wavelength_um | 2.046 µm | 2.115864226653975 | Onset lambda_c versus the fastest-growing wavelength (they differ off threshold) |
| specimen_foreground_area_fraction | 0.20345703125 1 | 0.4 | Otsu foreground fraction of the McEvey plate; reference is a prior, not the measurement |
| specimen_melanin_area_fraction | 0.02822265625 1 | 0.1 | Fraction of pixels darker than gray 70 inside the Otsu body; reference is a prior |
| specimen_left_minus_right_darkness | -5.277725108575623 gray | 0.0 | Male (left) vs female (right) foreground gray. Reference 0 is the no-dimorphism null. |
| specimen_abdominal_band_count | 7 1 | 6 | Troughs in the female abdominal ROI. Reference 6 is the typical tergite count, not this ROI count. |
| specimen_structure_tensor_coherency | 0.3745278547842805 1 | 0.5 | Mean structure-tensor coherency of the plate. Reference 0.50 is isotropic, not the photo. |
| ablation_std_ratio | 2.8509323824832185e-15 1 | 1.0 | Equal-diffusion control std(u) / focal std(u). Reference 1 would mean the control failed. |

## Environment at checkpoint


Pass 2 working tree. Slot-3207 files land in this PR; parent commit does not yet contain them.
