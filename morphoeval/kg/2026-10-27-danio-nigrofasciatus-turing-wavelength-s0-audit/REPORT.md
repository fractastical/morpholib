# MorphoEval report — audit `2026-10-27-danio-nigrofasciatus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `8585ec4a033d94afd4c400aaefdffe79c1f7bc69` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Danio nigrofasciatus Schnakenberg wavelength at 27 µm (Holocene, 0 Ma) (`models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py`) | 80 | 74 | 68 | 66 | 88 | 76 | 82 | 76.0 | 71.0 | 82.4 | 75.9 | **75.9** | mostly faithful |

## Findings and remediation

### Danio nigrofasciatus Schnakenberg wavelength at 27 µm (Holocene, 0 Ma)  —  overall 75.9 (mostly faithful)

Pass 1 integrates Schnakenberg kinetics at a=0.1, b=0.9, D_u=0.05, D_v=2.0 so that lambda_c = 2.71e-05 m (27 µm, log10=-4.567, tier 4_cellular_10um) at geological age 0 Ma. One-step eigenmode growth matches analytical sigma(k) at Pearson r = 0.999997 (RMSE 0.00141). The trunk of the Parichy CC BY 4.0 photograph is the render scaffold. Full credit for the alpha sweep, KS holdout, and ablation is withheld until Pass 2 re-runs them.

- **Model fidelity to source: 80** — Jacobian, continuous k_c, and the discrete-Laplacian dispersion are implemented and checked, but Pass 1 does not yet treat the eigenmode Pearson r as a closed measurement against the 0.98 gate.  
  evidence: `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L83–125
- **Morphological realism / specimen match: 74** — The comparison figure places the Parichy specimen beside a trunk field of horizontal melanophore dashes. Head, fin margin, caption, and scale bar stay photographic. The cellular dashes are not the three broad organ-scale stripes, so this pass does not score a close silhouette match.  
  evidence: `docs/media/danio_nigrofasciatus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L318–386
- **Parameter expressivity: 68** — Alpha, break fraction, and the three chronospecies are parameters, but Pass 1 does not yet score the monotonic gradient sweep.  
  evidence: `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L64–74
- **Morphometric validation: 66** — Analytical wavelength is pinned at 2.71e-05 m and a spectral peak is computed, but KS, holdout, and the ablation ratio are not yet accepted as scored measurements.  
  evidence: `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L265–283
- **Source and data traceability: 88** — Crossref-checked DOIs and a SHA-256 pin of the CC BY 4.0 Parichy photograph are in the reference JSON. OpenAlex scout notes that slot 3163 is outside the 100-slot campaign file.  
  evidence: `datasets/danio_nigrofasciatus_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/danio_nigrofasciatus_turing_wavelength_s0/danio_nigrofasciatus_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 76** — Module docstring states the PDE, the Jacobian, and the chronoseries. Pass 1 does not yet tie every measurement name back to a theorem line.  
  evidence: `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3163_danio_nigrofasciatus_turing_wavelength_s0.lean` L39–76
- **Reproducibility package: 82** — Seed 3163, --self-test, and the specimen SHA pin are in place. Pass 1 has not yet re-run after the non-circular reference floors were set.  
  evidence: `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L746–782

Remediation:
- {'id': 'R1', 'item': 'Score the alpha sweep (Spearman of anisotropy versus inverse horizontal gradient) and keep it on the comparison figure.'}
- {'id': 'R2', 'item': 'Add the trunk gray KS statistic and the anterior-posterior holdout continuity gap, with references that are not copies of the measured values.'}
- {'id': 'R3', 'item': 'Assert the ablation (kinetics replaced by relaxation) and record the pattern-std / ablation-std ratio.'}
- {'id': 'R4', 'item': 'Re-run image fitness of the growth GIF against the Parichy specimen and write the per-loop fitness JSON only.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Danio nigrofasciatus Schnakenberg wavelength at 27 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `91717ca6df274e3e1c0bf367643f453836a534d386c3a9a413fc1a38ae61dfdd`
- **input_dataset_sha256**: `datasets/danio_nigrofasciatus_turing_wavelength_s0/danio_nigrofasciatus_specimen.jpg` (`edea58cd2541052f…`), `datasets/danio_nigrofasciatus_turing_wavelength_s0/danio_nigrofasciatus_turing_wavelength_s0_reference.json` (`2ffd9764fa0a9a43…`)
- `docs/media/danio_nigrofasciatus_turing_wavelength_s0_comparison.png` — artifact_sha256 `148886b92b1be1b1…`, 844,084 B, 1344×960 px

| measurement | value | reference | method |
|---|---:|---|---|
| Turing dispersion specimen-independent growth Pearson r | 0.999997 Pearson r | 0.98 | One-step Fourier eigenmode growth versus analytical sigma(k); RMSE reported in the metrics JSON *(rmse=0.001409 on n=315 real eigenmodes)* |
| specimen stripe wavelength from simulated flank FFT | 27.3342 µm | 27.1 | Radial FFT peak of the activator times dx, with dx chosen so the analytical lambda_c equals 27.1 µm. CV of the radial bin is dominated by the peak ring. *(value is the measured spectral peak, not a copy of the catalog wavelength)* |
| specimen trunk structure-tensor coherency | 0.6583 coherency | 0.5 | Same Sobel structure-tensor coherency as morphoeval/verify_image_fitness.py on the CC BY specimen |
| specimen-vs-simulation KS distance of trunk gray histograms | 0.3362 KS statistic | 0.05 | ks_2samp on trunk pixels of the Parichy specimen versus the rendered Turing composite. Not a self-reference. *(ks p=0)* |
| nigrofasciatus dorsal-minus-ventral stripe continuity gap | 0.0013 run-length fraction | 0.0 | Mean AP run length above the 70th percentile, dorsal minus ventral. Broken ventral stripes predict a positive gap. |
| anterior-posterior holdout posterior continuity gap | 0.0914 run-length fraction | 0.0 | Threshold quantile fit on the anterior half of the flank; gap scored only on the posterior half (holdout). |
| ablation control pattern-std over relaxation-std ratio | 3643185.969 ratio | 1.0 | Negative control: Schnakenberg kinetics replaced by relaxation to (u0, v0). Ratio >> 1 means the pattern is reaction-driven, not the initial seed. |
| alpha sweep Spearman of anisotropy versus inverse horizontal gradient | 1.0 Spearman rho | 0.8 | Parameter sweep alpha in {1, 2, 3, 4.5, 6, 8.5}. Higher AP diffusion must lower |du/dx| (monotonic expressivity). |
| chronoseries control margaritatus continuity gap versus nigrofasciatus | 0.0002 gap difference | 0.15 | Focal broken-stripe gap minus predecessor near-isotropic gap. Positive means nigrofasciatus is more dorsoventrally polarized than margaritatus. This is a vs-predecessor control, not a copy of a reference scalar. *(rerio dorsal continuity=0.1110; margaritatus gap=0.0010)* |

## Environment at checkpoint


Pass 1. Evidence paths exist at this commit.
