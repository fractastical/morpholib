# MorphoEval report — audit `2026-10-27-acetabularia-acetabulum-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `bd2c74bc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-acetabularia-acetabulum-turing-wavelength-s0-audit -> 2026-10-27-acetabularia-acetabulum-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Acetabularia acetabulum Turing wavelength at 25.28 µm (Holocene, detail scale) (`models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py`) | 91 | 88 | 90 | 91 | 92 | 88 | 90 | 90.7 | 89.1 | 90.1 | 89.9 | **89.9** | faithful |

## Findings and remediation

### Acetabularia acetabulum Turing wavelength at 25.28 µm (Holocene, detail scale)  —  overall 89.9 (faithful)

Pass 2. Same Brusselator and tier length (25.28 µm, 2.528e-5 m, 0 Ma). Inhibitor D_axial/D_hoop swept over {0.35, 0.45, 0.70, 1.00, 1.60}: hoop mode wins at 0.45 (|mode| 8 vs 1) and the axial mode wins at 1.60 (|mode| 8 vs 1). Seeded grain drops rendered coherency from 0.87 to 0.54, beside the CC0 cap at 0.62. GIF CV similarity 87.3 (12 frames, 640x480). Dispersion Pearson r = 0.9996, wavelength-sweep Pearson r = 0.99999, RMSE 0.23 µm. Ablation ratio 0.097. Orientation KS = 0.191. Saturated peak 25.08 µm versus tier 25.28 µm.

- **Model fidelity to source: 91** — Pass 2 keeps the published Brusselator and the Turing wavenumber, and now shows that the reconstructed anisotropy is the term that chooses the axis. The diffusivities remain a reconstruction aimed at the tier length, which is why this is not a higher score.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L118–140; `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L357–379
- **Morphological realism / specimen match: 88** — After seeded grain, the 12-frame 640x480 GIF scores CV 87.3 against the CC0 cap (Sobel orientation, radial FFT, structure-tensor coherency, occupancy). Rendered coherency is 0.54 and the specimen is 0.62. The adult photo's sand and neighbouring caps are still absent, so this is not a registered overlay.  
  evidence: `docs/media/acetabularia_acetabulum_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L428–435
- **Parameter expressivity: 90** — Wavelength tracks diffusivity (r = 0.99999 across five scales). The anisotropy sweep is monotonic in the intended sense: ratios below 1 are hoop-dominated and 1.60 is axially dominated. Equal diffusion decays. The face-on GIF shows the nominal ratio only.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L318–379; `docs/media/acetabularia_acetabulum_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 91** — Dispersion Pearson r = 0.9996 with RMSE 0.016 /s (numeric slopes are not a copy of the analytic curve). Wavelength-sweep RMSE is 0.23 µm. Orientation KS against the CC0 cap is 0.191. Ablation ratio is 0.097 against a null of 1. Saturated peak 25.08 µm against tier reference 25.28 µm.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L619–700; `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — Unchanged DOIs, now also pointing at the v2 claim boundary. Specimen SHA-256 and CC0/CC BY creators are in the reference JSON and README.  
  evidence: `datasets/acetabularia_acetabulum_turing_wavelength_s0/README.md` L1–30; `datasets/acetabularia_acetabulum_turing_wavelength_s0/paper_citations.bib` L1–48
- **Code-to-model traceability: 88** — Pass 2 anisotropy sweep and grain are commented at the functions that implement them, and the docstring still carries the PDE and the chronoseries statement. The polar sampler remains a rendering choice.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L357–379; `morphoeval/lean/loops/Loop_3103_acetabularia_acetabulum_turing_wavelength_s0.lean` L14–53
- **Reproducibility package: 90** — --self-test now also checks that ratio 1.6 flips the dominant axis. Seed, paths and the metrics JSON are stable across the pass-2 rerun.  
  evidence: `models/2026-delphi10000-acetabularia-acetabulum-turing-wavelength-s0.py` L705–753

Remediation:
- A second assessor from another provider has not scored these dimensions.
- The 354 µm model cap is not a landmark registration of the adult CC0 photograph.
- The axial/hoop ratio is bounded by the sweep but was not measured on a cell.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.7]** `parameter_sweep` → `parameter-expressivity`: Sweep D_axial/D_hoop and record which Fourier axis wins.
- **[RESOLVED · P=1.5]** `specimen_comparison` → `specimen-match`: Bring rendered structure-tensor coherency toward the CC0 cap without copying its pixels.
- **[OPEN · P=1.1]** `independent_assessor` → `model-fidelity`: Independent second assessor has not reviewed the pass-2 scores.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Acetabularia acetabulum Turing wavelength at 25.28 µm (Holocene, detail scale)
- **spec_fingerprint** (semantic model + inputs): `96087cd4e2bc623a26b35d593a33788c367131953ba2d7b731f791fb3ce2cc1b`
- **input_dataset_sha256**: `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_turing_wavelength_s0_reference.json` (`3e562916ba2ad34f…`), `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_wikimedia_cc0.jpg` (`0773237a26517128…`)
- `datasets/acetabularia_acetabulum_turing_wavelength_s0/acetabularia_acetabulum_turing_wavelength_s0_metrics.json` — artifact_sha256 `505d99dcf9a87b68…`, 3,949 B, 
- `docs/media/acetabularia_acetabulum_turing_wavelength_s0_comparison.png` — artifact_sha256 `ee3d388a3cc6c973…`, 540,666 B, 1232×902 px
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
- note: Pass 2 on the same VM. Nanopub profile absent, so the KG is unsigned.

Pass 2 adds the anisotropy sweep and seeded cap grain. CV against the CC0 photograph is 87.3. Second assessor still open.
