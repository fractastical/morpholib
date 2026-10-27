# MorphoEval report — audit `2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `bd8176806dee28731b94a3866aa6609c1d9d4ba2` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Clytia hemisphaerica Schnakenberg wavelength at 8.64 µm (Pass 1) (`models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py`) | 76 | 78 | 60 | 64 | 90 | 80 | 84 | 70.7 | 72.8 | 84.9 | 75.8 | **75.8** | mostly faithful |

## Findings and remediation

### Clytia hemisphaerica Schnakenberg wavelength at 8.64 µm (Pass 1)  —  overall 75.8 (mostly faithful)

Pass 1 integrates Schnakenberg kinetics with Du = 1 µm²/s and Dv/Du = 20, and γ is already pinned near 2.227 s⁻¹ so k_* sits on 8.64 µm. The growth-rate comparison used an activator-only kick fit from t = 0. That probe is off the unstable eigenspace: Pearson r = 0.506 against Jacobian σ(k), which fails the 0.98 gate. The planula render and the equal-diffusion integrator exist in the file but were not yet the reported morphometric table. Holocene age is 0 Ma; the chronoseries is 1/1.

- **Model fidelity to source: 76** — The Jacobian and the PDE are the Schnakenberg system, but Pass 1 reads σ from an activator-only perturbation, so the numerical growth rates are not the unstable eigenvalue.  
  evidence: `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L164–212
- **Morphological realism / specimen match: 78** — The GIF places pigment grain inside a segmented day-2 planula on the photograph's background. The outline is more jagged than the DIC halo, so this is a recognisable larva rather than a matched micrograph.  
  evidence: `docs/media/clytia_hemisphaerica_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L378–414
- **Parameter expressivity: 60** — Du, the diffusion ratio, and γ are defined, but Pass 1 does not yet report the ratio sweep as a measurement. Wavelength control is asserted by the pin, not shown across the ratio.  
  evidence: `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L73–90
- **Morphometric validation: 64** — Pass-1 Pearson r between σ_numeric and σ_analytic is 0.506 (RMSE large relative to the band). That is below 0.98. Specimen aspect, circularity, and KS were not yet the recorded comparison.  
  evidence: `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L214–240
- **Source and data traceability: 90** — DOIs resolve on OpenAlex. The planula file is Momose's CC BY-SA 4.0 Commons micrograph, SHA-256 pinned. Slot 3150 is absent from the 100-slot campaign manifest; that absence is recorded rather than papered over.  
  evidence: `datasets/clytia_hemisphaerica_turing_wavelength_s0/paper_citations.bib` L1–89
- **Code-to-model traceability: 80** — The module docstring maps the PDE and the Pass-1 probe defect to the dispersion relation. Murray's scale argument is cited for the mode-count contrast, not section-pinned.  
  evidence: `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L1–47
- **Reproducibility package: 84** — Seed 3150, --self-test, and pinned γ. Pass 1 self-test already refuses a transient probe that accidentally looks calibrated, but the reported fit itself does not clear 0.98.  
  evidence: `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L505–543

Remediation:
- Refit σ from the analytic unstable eigenvector on the second half of each Fourier trajectory, and require Pearson r > 0.98 against the discrete-Laplacian dispersion.
- Put the equal-diffusion ablation (Dv = Du) in the measurement table next to the wild-type std(u).
- Segment the Momose planula and report aspect ratio, circularity, and an interior KS statistic whose value is not equal to its reference.
- Sweep Dv/Du at fixed γ and record that the selected wavelength moves monotonically.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Clytia hemisphaerica Schnakenberg wavelength at 8.64 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `96a665cf0d7c09f92284588fc71976c4d562666ee4ba1270067ce2e4f5d4c878`
- **input_dataset_sha256**: `datasets/clytia_hemisphaerica_turing_wavelength_s0/clytia_hemisphaerica_specimen.jpg` (`95bffe69b1854823…`), `datasets/clytia_hemisphaerica_turing_wavelength_s0/clytia_hemisphaerica_turing_wavelength_s0_reference.json` (`8e95bb9584ac6852…`), `datasets/clytia_hemisphaerica_turing_wavelength_s0/paper_citations.bib` (`045903087c0ddff1…`), `datasets/clytia_hemisphaerica_turing_wavelength_s0/scout_manifest.json` (`014d315226f357c5…`)
- `docs/media/clytia_hemisphaerica_turing_wavelength_s0_comparison.png` — artifact_sha256 `20a127f4c9842022…`, 447,751 B, 1152×1008 px

| measurement | value | reference | method |
|---|---:|---|---|
| selected Turing wavelength lambda_* | 8.6464 µm | 8.64 | argmax_k Re σ(k) of the Schnakenberg Jacobian at Du=1 µm²/s, Dv/Du=20 *(γ is pinned so k_* realises the assigned 8.64 µm detail length)* |
| dispersion-probe Pearson r (eigenvector, pass 2) | 0.999301 1 | 0.98 | PDE Fourier-mode growth vs Jacobian σ at the discrete Laplacian wavenumber *(RMSE 0.0141 s⁻¹; gate is r > 0.98)* |
| pass-1 transient probe Pearson r | 0.506133 1 | 0.98 | activator-only initial data, growth fit from t=0 (biased eigenspace) *(below the gate; this is why pass 2 changes the probe)* |
| dispersion-probe RMSE | 0.01406 s^-1 | 0.0 | root mean square of σ_numeric − σ_analytic over the probed band *(zero would be an exact eigenmode of the discrete Laplacian)* |
| equal-diffusion ablation std(u) | 0.0013248623997750928 1 | 0.0192708407717446 | Dv=Du integrated on the same grid and horizon as wild type *(reference_value is the wild-type std; ablation must sit far below it)* |
| specimen contour aspect ratio | 3.2304 1 | 1.0 | bounding-box height/width of the segmented day-2 planula mask *(null reference is a circular body; the planula is elongate)* |
| specimen contour circularity | 0.3711 1 | 1.0 | 4πA/P² on the segmented planula outline *(a circle would score 1; the larva does not)* |
| planula versus medusa-bell mode count at 8.64 µm | 28 stripes | 231 | floor(L/λ_*) with reconstructed L_planula=250 µm *(reference is floor(2000 µm / λ_*) on a medusa bell; detail scale holds fewer repeats. Age delta vs the 1/1 Holocene series is 0 Ma)* |
| simulation interior gray std | 17.3421 gray | 17.1185 | std of the GIF last frame luminance, including background *(reference is the first frame; growth must move the luminance distribution)* |
| specimen interior luminance KS statistic | 0.1908 1 | 0.0 | two-sample KS between specimen and simulation interior gray values *(0 would be a pixel copy; a positive KS is a real distributional distance)* |

## Environment at checkpoint


Pass 1 scored the activator-only dispersion probe before the eigenvector refit.
