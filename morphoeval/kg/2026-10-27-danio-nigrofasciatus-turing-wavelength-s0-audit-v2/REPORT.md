# MorphoEval report — audit `2026-10-27-danio-nigrofasciatus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `8585ec4a033d94afd4c400aaefdffe79c1f7bc69` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-danio-nigrofasciatus-turing-wavelength-s0-audit -> 2026-10-27-danio-nigrofasciatus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Danio nigrofasciatus Schnakenberg wavelength at 27 µm (Holocene, 0 Ma), Pass 2 (`models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py`) | 88 | 83 | 88 | 82 | 91 | 84 | 90 | 88.0 | 82.6 | 88.7 | 86.1 | **86.1** | faithful |

## Findings and remediation

### Danio nigrofasciatus Schnakenberg wavelength at 27 µm (Holocene, 0 Ma), Pass 2  —  overall 86.1 (faithful)

Pass 2 re-run. Dispersion Pearson r = 0.999997 (RMSE 0.00141, n = 315 real eigenmodes) against the analytical sigma(k), above the 0.98 gate and not equal to that gate. Measured spectral wavelength 27.33 µm versus the catalog 27.1 µm (2.71e-05 m, log10=-4.567) at age 0 Ma. Alpha sweep Spearman = 1.0 against a 0.8 floor. Ablation ratio of pattern std to relaxation std is about 3.6e6. Trunk KS versus the Parichy photograph is 0.336. GIF is 12 frames at 1120x480. Chronoseries at this scale, all 0 Ma: margaritatus spots (alpha 1.05), nigrofasciatus broken ventral stripes (alpha 6 / 1.15), rerio continuous stripes (alpha 8.5). The photograph's dorsoventral darkness profile is not fit (r = 0.023) and is not claimed.

- **Model fidelity to source: 88** — The script implements Schnakenberg kinetics, the continuous critical wavenumber, and the discrete 5-point dispersion. One-step eigenmode growth matches analytical sigma(k) at r = 0.999997. Anisotropy is an extension of the isotropic law, so this is not a line-by-line copy of Nakamasu's cell-contact rules.  
  evidence: `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L83–178; `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L180–233
- **Morphological realism / specimen match: 83** — Side-by-side, the Parichy fish (eye, fins, caption, scale bar) is preserved and the trunk carries the cellular melanophore field. The chronoseries strip separates margaritatus spots, nigrofasciatus broken stripes, and rerio continuous stripes. The trunk field is a fine dash pattern, not a repaint of the three broad body stripes, which is the right object at 27 µm and a limit on silhouette match.  
  evidence: `docs/media/danio_nigrofasciatus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/danio_nigrofasciatus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L372–414
- **Parameter expressivity: 88** — Alpha in {1, 2, 3, 4.5, 6, 8.5} lowers horizontal gradient energy monotonically (Spearman 1.0 versus a 0.8 floor). Dorsal continuity rises over the same sweep. The three chronospecies use different alphas at the same 27 µm and the same 0 Ma.  
  evidence: `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L285–316; `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L388–414
- **Morphometric validation: 82** — Measured wavelength 27.33 µm versus catalog 27.1 µm. Dispersion RMSE 0.00141. Trunk KS 0.336 versus a 0.05 identity floor. Posterior holdout gap is scored with a threshold fit only on the anterior half. The dorsoventral profile correlation is 0.023 and is reported as a miss, which caps this dimension.  
  evidence: `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L417–457; `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L628–740
- **Source and data traceability: 91** — DOIs were checked on Crossref for Turing, Schnakenberg, Nakamasu, Parichy 2006, and the specimen's eLife paper. The photograph is CC BY 4.0, David M. Parichy, SHA-256 pinned. Scout manifest records that slot 3163 does not collide with slot 1203.  
  evidence: `datasets/danio_nigrofasciatus_turing_wavelength_s0/paper_citations.bib` L1–93; `datasets/danio_nigrofasciatus_turing_wavelength_s0/README.md` L1–30; `datasets/danio_nigrofasciatus_turing_wavelength_s0/scout_manifest.json` L1–45
- **Code-to-model traceability: 84** — Docstring, Lean theorems (wavelength product, anisotropy order, diffusion threshold), and math-kg loci point at the same equations. The integer Lean proofs witness the inequalities; they do not re-derive the real eigenmode calculation.  
  evidence: `morphoeval/lean/loops/Loop_3163_danio_nigrofasciatus_turing_wavelength_s0.lean` L39–76; `morphoeval/math_kg_loops/delphi10000-danio-nigrofasciatus-turing-wavelength-s0.yaml` L1–40; `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L1–40
- **Reproducibility package: 90** — python3 models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py --self-test is deterministic at seed 3163, checks r > 0.98, age 0 Ma, GIF frames >= 10, and minimum side >= 480. Specimen SHA is verified before integration. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib).  
  evidence: `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L459–469; `models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py` L746–786

Remediation:
- {'id': 'R1', 'item': 'Closed in this pass: alpha sweep Spearman is measured and drawn on the comparison figure.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Closed in this pass: trunk KS and anterior-posterior holdout are in measure_morphoeval, with reference floors that differ from the values.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Closed in this pass: ablation ratio is asserted in --self-test (pattern std / relaxation std > 3).', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Closed in this pass: image fitness is computed from the growth GIF against the Parichy specimen and written only to the per-loop fitness JSON.', 'status': 'resolved'}
- {'id': 'R5', 'item': 'Still open: one specimen, and the dorsoventral darkness profile of that photograph is not reproduced (r = 0.023). A second fish and an explicit iridophore variable would be the next experiment. Not scored as done.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Danio nigrofasciatus Schnakenberg wavelength at 27 µm (Holocene, 0 Ma), Pass 2
- **spec_fingerprint** (semantic model + inputs): `91717ca6df274e3e1c0bf367643f453836a534d386c3a9a413fc1a38ae61dfdd`
- **input_dataset_sha256**: `datasets/danio_nigrofasciatus_turing_wavelength_s0/danio_nigrofasciatus_specimen.jpg` (`edea58cd2541052f…`), `datasets/danio_nigrofasciatus_turing_wavelength_s0/danio_nigrofasciatus_turing_wavelength_s0_reference.json` (`2ffd9764fa0a9a43…`)
- `docs/media/danio_nigrofasciatus_turing_wavelength_s0_comparison.png` — artifact_sha256 `148886b92b1be1b1…`, 844,084 B, 1344×960 px
- `docs/media/danio_nigrofasciatus_turing_wavelength_s0_growth.gif` — artifact_sha256 `568733fb3c23a087…`, 2,793,763 B, 1120×480 px, 12 frames

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


Pass 2. Evidence paths exist at this commit.
