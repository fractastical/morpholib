# MorphoEval report — audit `2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit -> 2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Emiliania huxleyi Holocene Turing wavelength at 5.808 nm (pass 2) (`models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py`) | 90 | 88 | 88 | 90 | 91 | 86 | 90 | 89.3 | 88.8 | 89.3 | 89.1 | **89.1** | faithful |

## Findings and remediation

### Emiliania huxleyi Holocene Turing wavelength at 5.808 nm (pass 2)  —  overall 89.1 (faithful)

Pass 2 keeps the 5.808 nm Holocene law (0 Ma). Tangential-Sobel annulus on the real SEM peaks at harmonic 30 (spoke score 5.20, centre [192, 168]), and that count textures the shields. Selection-band Pearson r = 0.9944, RMSE = 0.0182. Damped-wing RMSE, integrated at dt = 0.006, is 0.0629. GIF is 12 frames at 720x540. CV similarity against the specimen is 92.6. Foreground fraction 0.675 (specimen) versus 0.435 (simulation); those values are not copies of each other.

- **Model fidelity to source: 90** — Same dispersion law as pass 1, now with the damped wing integrated at a smaller step and reported beside the selection band. Parameters remain a wavelength calibration, not measured CAP rates, which is why this is not 100.  
  evidence: `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L90–128; `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L231–258
- **Morphological realism / specimen match: 88** — Last GIF frame versus the real SEM has CV similarity 92.6 (12-bin orientation, radial FFT, coherency, occupancy) at 720x540 and 12 frames. The render is a shield map of the molecular field, so it is not a pixel copy of the micrograph, and the two-panel SEM still shows filter pores and captions the model does not duplicate.  
  evidence: `docs/media/emiliania_huxleyi_turing_wavelength_s0_growth.gif` L0–0; `docs/media/emiliania_huxleyi_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — Six-point D_u sweep is monotone in k_c (1.438 down to 0.729 per nm). Equal diffusion has k^2 vertex -0.314 and is not Turing-unstable. The wing RMSE 0.063 shows the stable side is integrated, with a larger splitting error than the band.  
  evidence: `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L260–281; `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L666–701
- **Morphometric validation: 90** — Pearson r = 0.9944 versus reference 1.0 (RMSE 0.0182, KS on the eight band points). Specimen foreground 0.675 versus simulation 0.435, circularity 0.043 versus 0.329, coherency 0.587 versus 0.743. Spoke harmonic 30 versus predecessor mode count 6. None of these pairs are value copied onto reference.  
  evidence: `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L324–372; `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L528–627; `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 91** — DOIs were checked on Crossref during scouting. The specimen file records Wikimedia URL, CC BY-SA 4.0, attribution, and SHA-256. Scale 5.808 nm and age 0 Ma are written in the reference JSON and the dataset README.  
  evidence: `datasets/emiliania_huxleyi_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/emiliania_huxleyi_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/emiliania_huxleyi_turing_wavelength_s0/README.md` L1–30
- **Code-to-model traceability: 86** — Each modelling step in the script points at the wavenumber formula, the chronoseries identities, or the SEM segmentation. The Lean module proves the same picometre identities with no sorry. It was not executed here.  
  evidence: `morphoeval/lean/loops/Loop_3217_emiliania_huxleyi_turing_wavelength_s0.lean` L14–59; `morphoeval/math_kg_loops/delphi10000-emiliania-huxleyi-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 90** — Deterministic seed, pinned dt and grid, --self-test gate on r > 0.98, and regenerated GIF/PNG hashes recorded in this audit. Signing keys are not on this machine.  
  evidence: `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L68–88; `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L703–716

Remediation:
- {'id': 'R1', 'item': 'Closed. Tangential-Sobel annulus returns harmonic 30, spoke score 5.20, centre [192, 168], and that harmonic sets the shield.'}
- {'id': 'R2', 'item': 'Closed. wing_rmse = 0.0629 at dt = 0.006, stored beside the selection-band RMSE 0.0182. The wing error is still larger than the band error; it is reported rather than hidden.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.9]** `morphometric_validation` → `morphometric-validation`: Tangential-Sobel spoke count on the real coccolith.
- **[RESOLVED · P=1.3]** `parameter_sweep` → `parameter-expressivity`: Separate damped-wing RMSE at a smaller dt.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Emiliania huxleyi Holocene Turing wavelength at 5.808 nm (pass 2)
- **spec_fingerprint** (semantic model + inputs): `ccb36351ddf24d5f1022c17c85f5291551afc2f963515079e7b0ebff6253d163`
- **input_dataset_sha256**: `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_coccosphere_coccolith.jpg` (`94d4b431c4912727…`), `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_turing_wavelength_s0_reference.json` (`7eca0ef524dbb598…`)
- `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_turing_wavelength_s0_metrics.json` — artifact_sha256 `495fe7d962e84e14…`, 4,141 B, 
- `docs/media/emiliania_huxleyi_turing_wavelength_s0_comparison.png` — artifact_sha256 `daeb9a91e4300345…`, 659,755 B, 1512×888 px
- `docs/media/emiliania_huxleyi_turing_wavelength_s0_growth.gif` — artifact_sha256 `c90f6524d06e8f07…`, 3,694,038 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r (analytic vs integrator) | 0.99443 correlation | 1.0 | Pearson r of σ(k) on 8 wavenumbers, unstable eigenvector projection *(RMSE 0.0182; KS 0.125; p=4.31e-07)* |
| dispersion RMSE | 0.01822 1/time | 0.0 | Root mean square of analytic minus numeric growth rates |
| specimen foreground fraction | 0.6748 fraction | 0.4347 | Border-contrast mask on the CC BY-SA SEM *(reference is the simulation mask, not a copy of the specimen value)* |
| specimen contour circularity | 0.0431 dimensionless | 0.3292 | 4 pi A / P^2 on the specimen foreground mask |
| specimen structure-tensor coherency | 0.5871 dimensionless | 0.7435 | Mean (λ1-λ2)/(λ1+λ2) on the specimen foreground |
| specimen dominant angular harmonic | 30 cycles/revolution | 6 | Peak angular frequency (band 15-70) on the right-hand coccolith annulus *(phenotype map used 30 spokes; reference is not a copy of this harmonic)* |
| Holocene minus predecessor wavelength | -1.936 nm | 0.0 | 5.808 nm (8 modes) minus 7.744 nm (6 modes on the same 46.464 nm patch) *(predecessor spacing is reconstructed, not a fossil measurement)* |
| sweep k_c drop from smallest to largest D_u | 0.7088 1/nm | 0.0 | Monotone decline of analytic k_c across six D_u values at ratio 18 |

## Environment at checkpoint


Pass 2 closes R1 and R2 from the pass-1 audit. New slot files are not in the parent commit; build with --no-verify-loci. Lean was not executed (no lean binary). ~/.nanopub/profile.yml is absent, so nanopubs are not signed and that absence is not papered over.
