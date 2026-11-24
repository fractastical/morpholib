# MorphoEval report — audit `2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `8c1db788ac0382399c6119652ec88de045be6725` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit -> 2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cyanobacteria mat Turing ocellus wavelength at 64.19 µm (Holocene, 0 Ma) (`models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py`) | 91 | 82 | 89 | 86 | 90 | 86 | 89 | 90.3 | 83.5 | 88.5 | 87.1 | **87.1** | faithful |

## Findings and remediation

### Cyanobacteria mat Turing ocellus wavelength at 64.19 µm (Holocene, 0 Ma)  —  overall 87.1 (faithful)

Re-run of the same checkpoint after the four Pass-1 gaps were closed in the script: narrow-band sweep (r = 0.999475), equal-diffusion ablation (sigma_max = -0.0588, RMS ratio 0.047), plate segmentation (137 colonies versus 65 spots, KS = 0.453), and eigenvector dispersion fit (r = 0.999986, RMSE = 0.00558) with the rejected bin-mean control still reported (r = 0.5288). cv_sim_similarity remains 90.7 on a 12-frame 640x520 GIF. The size distributions still disagree; that is measured, not tuned away. No second 64 µm micrograph and no second assessor. Overall 87.1.

- **Model fidelity to source: 91** — Pass 2: the script integrates Schnakenberg kinetics with the Turing dispersion sigma(k), spectral diffusion, and an eigenvector probe of eleven exact Fourier modes. Slot parameters (a, b, Dv/Du) = (0.11, 0.83, 16) are not the Anabaena heterocyst set. The factor 1.08 on both diffusivities is an empirical finite-amplitude correction measured on this domain (scale 1.00 locks to mode 9 / 57 µm; 1.08 locks to mode 8 / 64.19 µm; 1.30 locks to mode 7). It is not a diffusivity reported for a cyanobacterial mat.  
  evidence: `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L14–28; `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L122–155; `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L164–208; `morphoeval/lean/loops/Loop_3158_cyanobacteria_mat_turing_wavelength_s0.lean` L15–20; `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L379–422
- **Morphological realism / specimen match: 82** — Pass 2: the GIF last frame scores cv_sim_similarity 90.7 against the culture-plate photograph (12 frames, 640x520). Beige colonies on a gold ground are the right pattern class for this plate. The photograph is a macroscopic plate, not a 64 µm field, and the agar illumination in the display transfer is not a biological field. KS = 0.453 on median-normalised diameters: the size distributions do not match, and that disagreement is left in the figure.  
  evidence: `docs/media/cyanobacteria_mat_turing_wavelength_s0_comparison.png` L0–0; `docs/media/cyanobacteria_mat_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L547–584; `datasets/cyanobacteria_mat_turing_wavelength_s0/cyanobacteria_mat_turing_wavelength_s0_metrics.json` L0–0
- **Parameter expressivity: 89** — Pass 2: diffusivity scale gamma in (0.5, 1.0, 1.5, 2.0) moves the narrow-band wavelength with Pearson r = 0.9995 against lambda * sqrt(gamma). An earlier integer-shell summary pinned gamma = 1.5 and gamma = 2.0 to the same mode and scored r = 0.967; the narrow-band estimator is the one reported. a and b are not swept.  
  evidence: `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L272–281; `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L424–455; `datasets/cyanobacteria_mat_turing_wavelength_s0/cyanobacteria_mat_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 86** — Pass 2: saturated dominant shell is 64.19 µm; the linear k_* wavelength is 66.617 µm. Eigenvector probe versus analytical sigma(k): Pearson r = 0.999986, RMSE = 0.00558 (11 modes). The rejected bin-mean estimator scores r = 0.529 on the same diffusivities. Plate segmentation finds 137 colonies (area CV 1.642) against 65 spots (area CV 0.295); two-sample KS = 0.453. Those specimen numbers are not copies of the simulation. n = 1 plate, and the plate is not at 64 µm.  
  evidence: `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L313–377; `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L485–545; `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L732–833; `datasets/cyanobacteria_mat_turing_wavelength_s0/cyanobacteria_mat_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Pass 2: Turing 1952, Schnakenberg 1979, Gierer and Meinhardt 1972, Petroff 2010, Shepard and Sumner 2010, and Polerecky et al. 2009 were resolved on OpenAlex or Crossref. The culture-plate SHA-256 is pinned and checked at runtime. Crossref and OpenAlex expose only the family name Armitage for 10.3389/fmicb.2012.00293, so that paper is not given guessed coauthors and is not used as a parameter source. The gallery photograph has no separate licence sidecar.  
  evidence: `datasets/cyanobacteria_mat_turing_wavelength_s0/paper_citations.bib` L1–77; `datasets/cyanobacteria_mat_turing_wavelength_s0/cyanobacteria_mat_turing_wavelength_s0_reference.json` L1–40; `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L881–884
- **Code-to-model traceability: 86** — Pass 2: the module docstring states the PDE, the dispersion, the chronoseries, and the claim boundary. Lean theorems mirror the trace sign, the eight-wavelength domain, the diffusion ratio 16, det J > 0, and Holocene versus the 3480 Ma Dresser photographs. There is no line-by-line map onto numbered equations of Turing 1952.  
  evidence: `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L1–68; `morphoeval/lean/loops/Loop_3158_cyanobacteria_mat_turing_wavelength_s0.lean` L14–52; `datasets/cyanobacteria_mat_turing_wavelength_s0/README.md` L1–30
- **Reproducibility package: 89** — Pass 2: seed 3158, --self-test, and a non-interactive default render. Self-test checks the trace, the Turing inequality, the 64.19 µm shell, dispersion r > 0.98, RMSE < 0.05, ablation sigma_max < 0, and the specimen digest. Dependencies are the repository requirements.txt, not a per-script lock. One assessor scored this audit.  
  evidence: `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L835–869; `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` L871–928; `datasets/cyanobacteria_mat_turing_wavelength_s0/README.md` L18–31

Remediation:
- {'id': 'R1', 'item': 'Closed: narrow-band sweep Pearson r = 0.999475 against lambda * sqrt(gamma).'}
- {'id': 'R2', 'item': 'Closed: equal-diffusion sigma_max = -0.0588 and RMS ratio 0.047, printed on the comparison figure.'}
- {'id': 'R3', 'item': 'Closed: plate versus spot KS = 0.453 on median-normalised diameters. The distributions differ; the statistic is not a self-copy.'}
- {'id': 'R4', 'item': 'Closed: eigenvector probe Pearson r = 0.999986, RMSE = 0.00558. Rejected bin-mean r = 0.5288 remains in the metrics.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cyanobacteria mat Turing ocellus wavelength at 64.19 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `17c14d868777fb5a50af2e088b23ad6bf8798c31500d2ec6dec6508dc89f427c`
- **input_dataset_sha256**: `datasets/cyanobacteria_mat_turing_wavelength_s0/cyanobacteria_mat_holocene_culture_plate.jpg` (`858930a5ad520ee4…`), `datasets/cyanobacteria_mat_turing_wavelength_s0/cyanobacteria_mat_turing_wavelength_s0_reference.json` (`d166e23ba481aac8…`)
- `datasets/cyanobacteria_mat_turing_wavelength_s0/cyanobacteria_mat_turing_wavelength_s0_metrics.json` — artifact_sha256 `d64bd13650faeb19…`, 3,108 B, 
- `docs/media/cyanobacteria_mat_turing_wavelength_s0_comparison.png` — artifact_sha256 `29d2a0dd6b33420d…`, 619,956 B, 1512×984 px
- `docs/media/cyanobacteria_mat_turing_wavelength_s0_growth.gif` — artifact_sha256 `2df1fd841d11f51f…`, 4,613,466 B, 640×520 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected pigment-ocellus wavelength | 64.19 µm | 66.617 | integer Fourier shell of the saturated activator with the most power *(reference is the linear k_* wavelength; the slot assignment is 64.19 µm)* |
| dispersion pearson r | 0.999986 Pearson r | 0.95 | eigenvector-initialised cosine modes n=4..14 versus analytical σ(k) *(RMSE of σ is 0.0056; reference 0.95 is an acceptance floor, not the measured r)* |
| rejected bin-mean dispersion pearson r | 0.5288 Pearson r | 0.98 | radial bin-mean of a broadband initial condition; a negative methodological control *(this estimator is biased; the eigenvector probe is the reported goodness-of-fit)* |
| dispersion RMSE | 0.0056 1/time | 0.0 | root mean square of σ_measured(k) - σ_analytic(k) *(reference 0 would be a perfect linear match)* |
| equal-diffusion control sigma_max | -0.0588 1/time | 0.0 | max_k σ(k) with D_u = D_v = D_u(Turing); negative means the control decays *(RMS ratio final/initial = 0.047314456136288555)* |
| specimen colony count | 137 colonies | 65 | high-pass threshold on the culture-plate photograph; reference is the simulated spot count *(specimen area fraction 0.14587959238401715; spot area fraction 0.2684461805555556)* |
| specimen colony area CV | 1.6418 CV | 0.2947 | std/mean of segmented colony areas; reference is the simulated spot-area CV *(scale-free shape statistic; the plate is not a 64 µm micrograph)* |
| segmented spot equivalent-diameter KS | 0.4526 KS D | 0.0 | two-sample KS on median-normalised equivalent diameters, specimen colonies vs activator peaks *(p = 1.0608521524716083e-08; reference 0 would mean identical distributions)* |
| diffusivity-sweep wavelength pearson r | 0.999475 Pearson r | 0.9 | measured λ(γ) versus λ_* √γ for γ in {0.5, 1, 1.5, 2} *(reference 0.9 is the expressivity floor, not the measured correlation)* |

## Environment at checkpoint


Loop 3158 source, specimen, Lean module, and media are at this commit. Audits cite it.
