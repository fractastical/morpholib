# MorphoEval report — audit `2026-10-27-drosophila-simulans-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila simulans 1.92 µm Turing carrier on the posterior-lobe arc (pass 1) (`models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py`) | 80 | 76 | 84 | 74 | 90 | 84 | 86 | 81.3 | 75.2 | 86.8 | 80.5 | **80.5** | faithful |

## Findings and remediation

### Drosophila simulans 1.92 µm Turing carrier on the posterior-lobe arc (pass 1)  —  overall 80.5 (faithful)

Pass 1 implements Schnakenberg kinetics with γ fixed so λ_c = 1.92 µm, measures a 67.57 µm annulus on the species plate (111 px = 100 µm), and fits one radial harmonic (Pearson r = 0.9825). Sweeps and the ablation exist in the script but the headline goodness-of-fit is still the in-sample single harmonic. Overall is withheld below the pass-2 bar.

- **Model fidelity to source: 80** — The dispersion formula recovers 1.92 µm exactly and the radial PDE runs, but pass 1 still treats the 4.8% gap between numerical k (3.115) and infinite-line k_c (3.272) as an uninterpreted residual. The painted field is a radial harmonic, not the PDE state.  
  evidence: `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L97–111; `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L130–177
- **Morphological realism / specimen match: 76** — The comparison figure places the plate beside the last GIF frame, but pass 1 has not yet separated plate CV from the whole-fly control. The match is a radial reconstruction of the annulus, not a cell-resolved lobe.  
  evidence: `docs/media/drosophila_simulans_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_simulans_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 84** — γ multipliers 0.5 to 3 shorten λ_c monotonically, and Du from 0.02 to 0.40 lengthens it while the diffusion drive stays positive. Pass 1 does not yet make that sweep the scored expressivity claim.  
  evidence: `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L404–417
- **Morphometric validation: 74** — Single-harmonic in-sample Pearson r is 0.9825 and the plate annulus is 67.57 µm against the 1.92 µm detail law. The odd-radius holdout is computed later in the file but is not the pass-1 headline, and KS/CV segmentation are not yet the basis of the score.  
  evidence: `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L259–280; `datasets/drosophila_simulans_turing_wavelength_s0/drosophila_simulans_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — OpenAlex-resolved DOIs for Turing, Schnakenberg, Glassford, Hagen, Huey, and Edwards. Plate SHA-256 and the CC BY-SA 3.0 Weeks female are pinned in the reference JSON.  
  evidence: `datasets/drosophila_simulans_turing_wavelength_s0/paper_citations.bib` L1–78; `datasets/drosophila_simulans_turing_wavelength_s0/drosophila_simulans_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — jacobian, gamma_for_wavelength, and the Lean wavelength-order theorems name the same quantities. Section-level pins into Turing 1952 are not reproduced beyond the dispersion formula.  
  evidence: `morphoeval/lean/loops/Loop_3205_drosophila_simulans_turing_wavelength_s0.lean` L34–37; `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L72–90
- **Reproducibility package: 86** — Seed 3205, --self-test, and hashed inputs. Pass 1 self-test already renders the GIF. No second machine and no nanopub profile on this runner.  
  evidence: `models/2026-delphi10000-drosophila-simulans-turing-wavelength-s0.py` L749–780

Remediation:
- {'id': 'R1', 'item': 'Make the odd-radius holdout the headline goodness-of-fit, not only the in-sample single harmonic (pass-1 r = 0.9825).'}
- {'id': 'R2', 'item': 'Report computer-vision similarity of the GIF against the annular plate and, separately, against the Wikimedia whole-fly photograph.'}
- {'id': 'R3', 'item': 'Interpret the numerical-versus-analytical wavenumber gap (3.115 vs 3.272) as the bounded-disk shift instead of leaving it uninterpreted.'}
- {'id': 'R4', 'item': 'Independent second assessor is not available in this cloud session.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila simulans 1.92 µm Turing carrier on the posterior-lobe arc (pass 1)
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


Pass 1 working tree on cursor/drosophila-simulans-turing-wavelength-9215. Loop files are new and are not in that commit.
