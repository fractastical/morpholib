# MorphoEval report — audit `2026-10-27-drosophila-suzukii-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila suzukii Turing onset wavelength at 2.046 µm (0 Ma, detail scale) (`models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py`) | 84 | 74 | 82 | 78 | 90 | 84 | 86 | 83.3 | 75.5 | 86.8 | 81.2 | **81.2** | faithful |

## Findings and remediation

### Drosophila suzukii Turing onset wavelength at 2.046 µm (0 Ma, detail scale)  —  overall 81.2 (faithful)

Pass 1. Schnakenberg kinetics with the Turing onset wavenumber calibrated so lambda_c = 2.046 µm at age 0 Ma. A pure-u mode kick gave dispersion Pearson r = 0.970, under the 0.98 gate. The noise-seeded field is a labyrinth (spacing CV 0.37). Subobscura (f_u < 0) and equal diffusion stay flat, but the first figure autoscaled that roundoff into fake stripes. Specimen Otsu metrics are real and not equal to their priors. The McEvey plate does not resolve 2.046 µm.

- **Model fidelity to source: 84** — Onset wavenumber, Jacobian trace/determinant, and sigma(k) match Turing 1952 and Schnakenberg 1979, and D_u is solved so lambda_c is 2.046 µm. The first numerical sigma used a pure-u kick, which does not isolate the unstable eigenvector, so the dispersion correlation was 0.970 rather than the law. Kinetics are phenomenological.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L88–114; `morphoeval/lean/loops/Loop_3207_drosophila_suzukii_turing_wavelength_s0.lean` L36–55
- **Morphological realism / specimen match: 74** — The comparison figure places the McEvey male/female plate beside the carrier. The noise-seeded field is a melanin-colored labyrinth, not a portrait of the fly. The growth GIF ends on the real plate (12 frames, 960×640). The first null-control panel autoscaled 1e-15 roundoff into visible stripes, which misrepresents subobscura.  
  evidence: `docs/media/drosophila_suzukii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_suzukii_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 82** — Diffusivity scale s moves the analytical wavelength as sqrt(s), and the fastest PDE mode follows it (sweep r = 0.997, monotonic). Subobscura turns the band off by flipping the sign of f_u. The stripe-seed lock to the 12-wave window mode is not yet part of this pass's scored claim.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L226–259; `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L61–63
- **Morphometric validation: 78** — Specimen foreground fraction 0.203 (prior 0.40), melanin fraction 0.028 (prior 0.10), left-right darkness -5.3 gray (null 0), abdominal troughs 7 (tergite prior 6), coherency 0.375 (isotropic prior 0.50). None equals its reference. The pure-u dispersion estimate did not clear Pearson r > 0.98 (r = 0.970). Labyrinth spacing CV is 0.37 and KS against lambda_sigma is 0.63.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L323–370; `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L185–223
- **Source and data traceability: 90** — DOIs for Turing 1952, Schnakenberg 1979, Gompel 2005, Prud'homme 2006, Atallah 2014, Koshikawa 2020 and Kondo 2010 were resolved on OpenAlex. The specimen SHA-256 is pinned. The 2.046 µm scale and 0 Ma age are stated in the dataset README and reference JSON. Wikimedia CC BY candidates are in the scout manifest.  
  evidence: `datasets/drosophila_suzukii_turing_wavelength_s0/paper_citations.bib` L1–72; `datasets/drosophila_suzukii_turing_wavelength_s0/drosophila_suzukii_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — onset_kc2, growth_rate and the Schnakenberg steady state are commented with the Turing/Murray and Schnakenberg sources. The Lean module records the same sign facts (trace, determinant, ocellus count, subobscura f_u). The pure-u measurement path is the step that does not yet match the formula.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L70–96; `morphoeval/math_kg_loops/delphi10000-drosophila-suzukii-turing-wavelength-s0.yaml` L40–67
- **Reproducibility package: 86** — --self-test checks the onset wavelength, the Turing flags, the predecessor and the equal-diffusion ablation. Seed 3207. Specimen hash is checked. Dependencies are the repo scientific stack (numpy, scipy, Pillow, matplotlib). The Pearson r > 0.98 assertion was added because the first estimator missed it.  
  evidence: `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L569–592; `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` L697–734

Remediation:
- Replace the pure-u mode kick with least-stable-eigenvector initial data so analytical and numerical sigma(k) clear Pearson r > 0.98.
- Add the diffusivity sweep, the equal-diffusion ablation, and the subobscura f_u < 0 control to the self-test assertions, not only to the notebook narrative.
- Segment the McEvey plate (Otsu foreground, melanin fraction, abdominal troughs, structure-tensor coherency) and keep every specimen value different from its reference.
- Draw the null controls on the focal color scale. Autoscaling 1e-15 roundoff drew fake stripes on subobscura.
- Seed a critical-mode stripe and report its spacing against lambda_c. The noise-seeded labyrinth CV is 0.37.
- Single assessor only. The plate cannot resolve 2.046 µm; do not treat organism-level Otsu numbers as a carrier measurement.

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


Pass 1 working tree. New slot-3207 files are not in this parent commit; loci are the working tree.
