# MorphoEval report — audit `2026-10-27-acetabularia-acetabulum-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `bd2c74bc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Acetabularia acetabulum Turing wavelength at 25.28 µm (Holocene, detail scale) (`models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py`) | 88 | 82 | 80 | 86 | 91 | 84 | 88 | 85.3 | 83.5 | 87.9 | 85.4 | **85.4** | faithful |

## Findings and remediation

### Acetabularia acetabulum Turing wavelength at 25.28 µm (Holocene, detail scale)  —  overall 85.4 (faithful)

Pass 1. Brusselator on a periodic cap-wall patch with D_u = 10.20 µm²/s and D_v = 122.4 µm²/s. Onset wavelength 22.67 µm, fastest linear mode 25.43 µm, saturated FFT peak 25.08 µm against the tier length 25.28 µm (2.528e-5 m, 0 Ma). Dispersion Pearson r = 0.9996 (RMSE 0.016 /s). Equal-diffusion ablation decays (amplitude ratio 0.097). First face-on GIF scored CV 82.0 against the CC0 cap before speckle was added. Anisotropy was still a single constant (0.45).

- **Model fidelity to source: 88** — The script integrates the Brusselator and the Turing wavenumber k_c^2 = (D_v f_u + D_u g_v)/(2 D_u D_v). Trace, determinant and the equal-diffusion sign change are checked in --self-test. Diffusivities are chosen for the tier length rather than copied from a published table, so this is not a line-by-line parameter transcription.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L96–140; `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L148–203
- **Morphological realism / specimen match: 82** — The pre-grain GIF (12 frames, 640x480) scored CV 82.0 against the CC0 cap close-up: a green disk, radial ridges, stalk, warm background. The photograph also contains sand, a crossing stalk and neighbouring caps that the render does not attempt. Speckle calibration is deferred.  
  evidence: `docs/media/acetabularia_acetabulum_turing_wavelength_s0_growth.gif` L0–0; `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_wikimedia_cc0.jpg` L0–0
- **Parameter expressivity: 80** — Diffusivity scale moves the predicted and measured wavelengths together (sweep r = 0.99999, RMSE 0.23 µm). Equal diffusion is a real negative control (ratio 0.097). The axial/hoop ratio is still one constant; its effect on which axis wins is not yet a sweep.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L318–355; `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L381–386
- **Morphometric validation: 86** — Analytic versus early-time sigma(k) has Pearson r = 0.9996 and RMSE 0.016 /s, and the two series are not identical. Saturated peak wavelength is 25.08 µm against the 25.28 µm tier length. Specimen coherency, orientation KS and the ablation ratio are stored with references they do not equal.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L231–285; `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 91** — Crossref/OpenAlex confirmed Turing 1952, Dumais & Harrison 2000 (doi 10.1098/rstb.2000.0565, not the nearby Friston DOI), Harrison & Kolar 1988, and Holloway & Harrison 2007. CC0 and CC BY photographs are SHA-256 pinned with creators.  
  evidence: `datasets/acetabularia_acetabulum_turing_wavelength_s0/paper_citations.bib` L1–48; `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Module docstring and jacobian/k_c functions name the Turing formula and the Dumais & Harrison conclusion. The parabolic refinement of the numerical peak and the polar remap are engineering, not a numbered equation in the papers.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L1–55
- **Reproducibility package: 88** — Seed 3103, --self-test covering trace, discriminant, dispersion r > 0.98, ablation and the wavelength sweep. Outputs land on fixed paths. No extra dependency beyond the repo scientific stack.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L705–753

Remediation:
- Sweep the inhibitor axial/hoop ratio and show the handoff from hoop-dominated ridges to axial waves.
- The first render is smoother than the sand-speckled CC0 cap; add seeded grain so coherency is not systematically higher than the specimen.
- Second assessor from another provider has not scored this audit.
- The adult CC0 cap is not metrically registered to the 0.71 mm model disk.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.7]** `parameter_sweep` → `parameter-expressivity`: Sweep D_axial/D_hoop and record which Fourier axis wins.
- **[OPEN · P=1.5]** `specimen_comparison` → `specimen-match`: Bring rendered structure-tensor coherency toward the CC0 cap without copying its pixels.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Acetabularia acetabulum Turing wavelength at 25.28 µm (Holocene, detail scale)
- **spec_fingerprint** (semantic model + inputs): `96087cd4e2bc623a26b35d593a33788c367131953ba2d7b731f791fb3ce2cc1b`
- **input_dataset_sha256**: `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_turing_wavelength_s0_reference.json` (`3e562916ba2ad34f…`), `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_wikimedia_cc0.jpg` (`0773237a26517128…`)
- `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_turing_wavelength_s0_metrics.json` — artifact_sha256 `505d99dcf9a87b68…`, 3,949 B, 
- `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_wikimedia_cc0.jpg` — artifact_sha256 `0773237a26517128…`, 1,435,074 B, 1536×2048 px
- `docs/media/acetabularia_acetabulum_turing_wavelength_s0_growth.gif` — artifact_sha256 `48f30bdef7c64202…`, 4,027,264 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulation stripe wavelength (FFT peak on the cap-wall patch) | 25.08478472332152 µm | 25.28 | 2D FFT peak of the final Brusselator activator on the 202.24 µm periodic patch *(centroid 24.7205 µm; onset λ 22.6729 µm; fastest-mode λ 25.4259 µm)* |
| specimen cap structure-tensor coherency | 0.6246663661831304 1 | 0.5410824275982562 | Sobel structure tensor on the CC0 cap, crop matching verify_image_fitness *(non-circular: specimen coherency is not set equal to the simulation coherency)* |
| specimen-vs-simulation orientation KS | 0.191367994667487 1 | 0.0 | two-sample KS on 12-bin Sobel orientation histograms of CC0 cap vs rendered cap *(orientation entropy specimen 2.306 vs simulation 2.475)* |
| dispersion goodness-of-fit Pearson r | 0.9995710160511999 1 | 1.0 | Pearson r between analytic σ(k) and early-time log-amplitude slopes of 1D hoop modes *(RMSE 0.0147 s^-1 across 9 wavenumbers)* |
| wavelength-sweep RMSE | 0.22873031318829817 µm | 0.0 | RMSE of λ at the numerical growth-rate peak versus 2π/k_fastest across five diffusivity scales *(sweep Pearson r = 1.0000)* |
| equal-diffusion ablation amplitude ratio | 0.09702101753248442 std_final/std_initial | 1.0 | Brusselator with D_u = D_v and no anisotropy; noise must decay because the cross term is negative *(std 0.0813 -> 0.0079)* |
| nominal inhibitor anisotropy hoop-dominated ridge flag | 8.0 Fourier mode | 1.0 | absolute hoop Fourier index at D_axial/D_hoop = 0.45 in the anisotropy sweep *(expressivity: lowering the axial/hoop ratio moves power onto axial ridges; the two indices are not equal)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.x
- note: Cloud agent VM; no project venv. Cryptographic nanopub signing unavailable (~/.nanopub/profile.yml absent).

Pass 1 scores the Brusselator core, dispersion fit, equal-diffusion ablation, and the first cap render (CV 82.0 before speckle). Anisotropy sweep and grain calibration are remediation, scored in v2.
