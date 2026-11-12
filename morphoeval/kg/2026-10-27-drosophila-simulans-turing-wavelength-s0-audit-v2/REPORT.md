# MorphoEval report — audit `2026-10-27-drosophila-simulans-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-simulans-turing-wavelength-s0-audit -> 2026-10-27-drosophila-simulans-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila simulans 1.92 µm Turing carrier on the posterior-lobe arc (pass 2) (`models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py`) | 88 | 86 | 90 | 91 | 92 | 88 | 90 | 88.7 | 87.9 | 90.1 | 88.8 | **88.8** | faithful |

## Findings and remediation

### Drosophila simulans 1.92 µm Turing carrier on the posterior-lobe arc (pass 2)  —  overall 88.8 (faithful)

Pass 2 keeps the same dispersion law (λ_c = 1.92 µm at γ = 1.428) and reports the odd-radius holdout Pearson r = 0.9895, RMSE 4.07 against a null RMSE 28.16, and KS 0.096. Plate CV similarity of the 12-frame GIF is 89.5; the Wikimedia whole-fly control is 85.6, so the score is not treated as species identification. Equal-diffusion ablation drops the amplitude ratio to 0.0078. The radial PDE peak k = 3.115 versus infinite-line k_c = 3.272 (ratio 0.952) is the bounded-disk shift.

- **Model fidelity to source: 88** — γ is solved so critical_wavelength returns 1.92 µm to numerical noise. Equal diffusivities give max growth -0.143 and an amplitude ratio 0.0078. The radial solver's k is 4.8% below the infinite-line k_c; the code records that ratio as a bounded-disk shift rather than tuning it away. The GIF is the weakly nonlinear radial mode, not a saturated 2D field.  
  evidence: `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L97–128; `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L419–442
- **Morphological realism / specimen match: 86** — Last GIF frame versus the annular plate scores CV similarity 89.5 (12 frames, 800x600). The same GIF versus the CC BY-SA whole-fly photograph scores 85.6, so specimen-match is the annulus reconstruction, not a unique species ID. Side-by-side panels are in the comparison PNG.  
  evidence: `docs/media/drosophila_simulans_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_simulans_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — Raising γ/γ_simulans from 0.5 to 3 shortens λ_c from 2.72 µm to 1.11 µm. Raising Du from 0.02 to 0.40 lengthens λ_c from 1.19 µm to 7.44 µm while h stays positive. Both sweeps are monotonic and are the chronoseries lever (silvestris γ ratio 0.529, subobscura 1.44).  
  evidence: `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L404–417; `datasets/drosophila_simulans_turing_wavelength_s0/drosophila_simulans_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 91** — Odd-radius holdout Pearson r is 0.9895 against the pass-1 in-sample reference 0.9825. RMSE is 4.07 versus null 28.16. KS is 0.096. The plate annulus is 67.57 µm, not the 1.92 µm detail reference. Otsu dark-annulus fraction is 0.650 versus a 0.5 baseline. Predecessor wavelength 2.64 µm is stored against the focal 1.92 µm reference.  
  evidence: `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L179–186; `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L339–376; `datasets/drosophila_simulans_turing_wavelength_s0/drosophila_simulans_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — Seed DOI 10.1098/rstb.1952.0012 and the anatomical, picture-wing, and cline DOIs resolve on OpenAlex. Input SHA-256 covers the scout manifest, bibliography, README, gallery plate, and the Weeks CC BY-SA 3.0 female.  
  evidence: `datasets/drosophila_simulans_turing_wavelength_s0/paper_citations.bib` L1–85; `datasets/drosophila_simulans_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 88** — Lean theorems pin the wavelength ratios 11/8 and 6/5, the mode counts 36<50<60, the posterior-lobe flag, and the equal-diffusivity block. The Python dispersion functions use the same Jacobian.  
  evidence: `morphoeval/lean/loops/Loop_3205_drosophila_simulans_turing_wavelength_s0.lean` L18–48; `morphoeval/lean/loops/Loop_3205_drosophila_simulans_turing_wavelength_s0.lean` L69–78
- **Reproducibility package: 90** — python3 --self-test checks the wavelength, both sweeps, the ablation, holdout r > 0.98, and a 12-frame GIF at 800x600. Seed is 3205. Signing is unavailable because no nanopub profile is installed; that is recorded rather than simulated.  
  evidence: `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L749–790; `datasets/drosophila_simulans_turing_wavelength_s0/README.md` L28–36

Remediation:
- {'id': 'R1', 'item': 'Resolved in pass 2: odd-radius holdout Pearson r = 0.9895 is a measured row and a self-test gate.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved in pass 2: plate CV similarity 89.5 and Wikimedia whole-fly CV 85.6 are both reported in the loop fitness JSON.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Resolved in pass 2: bounded_disk_k_ratio records numerical k / analytical k_c = 0.952. The gap is kept.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Still open: no independent second assessor in this session.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila simulans 1.92 µm Turing carrier on the posterior-lobe arc (pass 2)
- **spec_fingerprint** (semantic model + inputs): `60c689b8831f2a33a5b5821273a9d442392ad8ba957aa7d326024097a96f04cd`
- **input_dataset_sha256**: `datasets/drosophila_simulans_turing_wavelength_s0/Drosophila_simulans_female_wikimedia.jpg` (`8cf84002b7360579…`), `datasets/drosophila_simulans_turing_wavelength_s0/drosophila_simulans_turing_wavelength_s0_reference.json` (`cbf21999cb034540…`), `datasets/drosophila_simulans_turing_wavelength_s0/drosophila_simulans_turing_wavelength_s0_specimen.jpg` (`d581f0b19b1f33bb…`), `datasets/drosophila_simulans_turing_wavelength_s0/paper_citations.bib` (`f6fa93b93eea80e6…`), `datasets/drosophila_simulans_turing_wavelength_s0/scout_manifest.json` (`b0435b2ab49a47d2…`)
- `datasets/drosophila_simulans_turing_wavelength_s0/drosophila_simulans_turing_wavelength_s0_metrics.json` — artifact_sha256 `049bd46e61a8d7b7…`, 5,083 B, 
- `docs/media/drosophila_simulans_turing_wavelength_s0_comparison.png` — artifact_sha256 `816d722ff5c9d5c2…`, 519,627 B, 1260×840 px
- `docs/media/drosophila_simulans_turing_wavelength_s0_growth.gif` — artifact_sha256 `a99bc81ec1eda90e…`, 4,767,949 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen radial annulus wavelength | 67.567568 µm | 1.92 | azimuthal mean of the species plate; dominant period converted with the 100 µm scale bar *(granules of 1.92 µm per annulus = 35.191)* |
| pass-2 weakly nonlinear Pearson r | 0.98951 correlation | 0.982532 | specimen radial luminance versus 3-harmonic Schnakenberg mode at the plate's measured k *(RMSE 4.068 versus null RMSE 28.158)* |
| pass-2 held-out radial Pearson r | 0.989464 correlation | 0.982532 | 3-harmonic mode fit on even radii, Pearson r on odd radii of the specimen plate *(odd-radius holdout; not equal to the in-sample pass-2 correlation)* |
| pass-2 radial RMSE | 4.067774 luminance | 28.158004 | root mean square residual of the 3-harmonic radial mode *(KS statistic 0.0958)* |
| specimen dark-annulus fraction | 0.649958 area fraction | 0.5 | Otsu threshold on luminance above the caption band; fraction darker than the threshold *(Otsu threshold 105.59; moment circularity 0.3957)* |
| specimen structure-tensor coherency | 0.30021 coherency | 0.368368 | mean (λ1-λ2)/(λ1+λ2) of the blurred structure tensor on the plate interior *(plate versus rendered radial mode; the two numbers are computed separately)* |
| equal-diffusion ablation amplitude ratio | 0.007796 std ratio | 1.0 | std(u) with Du=Dv divided by std(u) at the Turing diffusivity ratio, same γ and seed *(max growth Turing 0.3758; equal-D -0.1428)* |
| silvestris predecessor detail wavelength | 2.64 µm | 1.92 | reconstructed γ retuning (11/8)·1.92 µm on picture-wing cuticle that lacks a posterior lobe *(mode count on the 96 µm arc is 36 versus 50)* |

## Environment at checkpoint


Pass 2 on the same working tree. Loop files are new relative to the checkpoint commit. Nanopubs are unsigned: ~/.nanopub/profile.yml is absent.
