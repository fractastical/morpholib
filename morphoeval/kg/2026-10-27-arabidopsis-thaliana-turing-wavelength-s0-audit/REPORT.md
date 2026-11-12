# MorphoEval report — audit `2026-10-27-arabidopsis-thaliana-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `7dbcb5f4ae37938efcf50ef7ff5969863bb80dbe` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Arabidopsis thaliana Turing wavenumber at 111.6 µm (`models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py`) | 89 | 82 | 85 | 84 | 90 | 84 | 86 | 87.7 | 82.8 | 86.8 | 85.4 | **85.4** | faithful |

## Findings and remediation

### Arabidopsis thaliana Turing wavenumber at 111.6 µm  —  overall 85.4 (faithful)

Schnakenberg reduction of Bouyer TTG1 trapping/depletion on a 446.4 µm primordium patch, with the Turing wavenumber fixed at 111.6 µm (0.0001116 m) and diffusivity ratio 8.4. Holocene extant, chronoseries 1/1, age 0 Ma. Continuum dispersion versus the discrete Laplacian has Pearson r = 0.99991. Equal diffusion collapses the field. A noisy run raises the high-k spectral tail by a factor 1.70.

- **Model fidelity to source: 89** — The script implements Schnakenberg kinetics, the steady-state Jacobian, and the Turing wavenumber law with Dv/Du = 8.4 so that λ_c is 111.6 µm. Continuum σ(k) is checked against the discrete Laplacian on the same 128 grid (r = 0.99991, RMSE = 0.00336). Equal diffusion is integrated at the geometric-mean diffusivity and the field standard deviation falls to about 1e-14. The stochastic term is a stand-in for Di Patti noise, not their master equation, which is why this is not a 95.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L79–150; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L158–204; `morphoeval/lean/loops/Loop_3117_arabidopsis_thaliana_turing_wavelength_s0.lean` L10–43
- **Morphological realism / specimen match: 82** — The comparison figure places Heiti Paves' CC BY-SA trichome SEM beside the activator field and a blend of the two. The field is a spot lattice on a 446.4 µm patch; it does not grow the branched hair. The growth GIF ends on that SEM so the differentiated trichome is what the last frame shows. The mislabeled Aeonium gallery file was rejected.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L291–323; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L394–465; `docs/media/arabidopsis_thaliana_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 85** — With Du held at the 111.6 µm value, Dv/Du runs through 1, 4, 6, 8.4, 12 and 20. Amplitude stays near zero below the Turing threshold and rises past d = 8.4. The wavelength panel draws λ(argmax σ) only where σ_max is positive, so the equal-diffusion control is not drawn as a fake 558 µm mode. That ratio is the knob for this slot; it is not copied from a millimetre-scale sibling.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L325–361; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L415–465
- **Morphometric validation: 84** — The nonlinear FFT wavelength is 111.50 µm against the law value 111.6 µm. Nearest-neighbour spacing of 21 spots is 101.11 µm (CV 0.108), with a 500-draw bootstrap 95% interval 96.60–105.81 µm. A KS test against Normal(111.6, 16.74) gives D = 0.495, which does not match that normal and is reported as a failure of that reference distribution rather than forced through. Specimen cyan fraction 0.094 is not the simulated peak fraction 0.284. Dispersion RMSE is 0.00336.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L206–254; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L477–608; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L732–758
- **Source and data traceability: 90** — Four DOIs are cited. Both images are CC BY-SA 3.0 Wikimedia files with authors (Heiti Paves; Emmanuel Boutet) and SHA-256 pins. The scout manifest records that slot 3117 is not in the 100-slot campaign file.  
  evidence: `datasets/arabidopsis_thaliana_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/arabidopsis_thaliana_turing_wavelength_s0/arabidopsis_thaliana_turing_wavelength_s0_reference.json` L1–40; `datasets/arabidopsis_thaliana_turing_wavelength_s0/scout_manifest.json` L1–20
- **Code-to-model traceability: 84** — Jacobian, wavenumber, integrator, sweep, and specimen segmentation are separate functions. Lean theorems match the sign of the Turing numerator and the mode counts 0, 4, and 16. They do not encode the floating-point Jacobian, which is the right claim boundary for a dependency-free Lean file.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L123–156; `morphoeval/math_kg_loops/delphi10000-arabidopsis-thaliana-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 86** — Seed 3117, pinned specimen hashes checked at startup, --self-test, and a metrics cache. The self-test passed on Python 3.12 with the environment versions above. Signing could not be run.  
  evidence: `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L72–77; `models/2026-delphi10000-arabidopsis-thaliana-turing-wavelength-s0.py` L732–758; `docs/media/arabidopsis_thaliana_turing_wavelength_s0_growth.gif` L0–0

Remediation:
- Nanopub signing is blocked: ~/.nanopub/profile.yml is not on this machine.
- Di Patti 2023 radial-spectrum bins were not in the OpenAlex record, so the fat-tail test is only the scalar tail ratio 1.70.
- Do not convert the Boutet epidermis dominant mode (480 px) into micrometres.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=0.9]** `provenance_lock` → `repro-package`: Sign the pass-1 nanopubs once a nanopub profile exists.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Arabidopsis thaliana Turing wavenumber at 111.6 µm
- **spec_fingerprint** (semantic model + inputs): `c3c61462f42b93b82ab11c814de08014335238d29e5879ad1c5373a3aac7d878`
- **input_dataset_sha256**: `datasets/arabidopsis_thaliana_turing_wavelength_s0/arabidopsis_thaliana_epidermis_stomata.jpg` (`4b7c1062f6ebfd6e…`), `datasets/arabidopsis_thaliana_turing_wavelength_s0/arabidopsis_thaliana_trichome_specimen.jpg` (`41f6702ee4687dd9…`), `datasets/arabidopsis_thaliana_turing_wavelength_s0/arabidopsis_thaliana_turing_wavelength_s0_reference.json` (`baa7ea4e84cec45e…`), `datasets/arabidopsis_thaliana_turing_wavelength_s0/paper_citations.bib` (`fb11f832295b852c…`), `datasets/arabidopsis_thaliana_turing_wavelength_s0/scout_manifest.json` (`ecd8d065fc5e5570…`)
- `docs/media/arabidopsis_thaliana_turing_wavelength_s0_comparison.png` — artifact_sha256 `008bc69512d76ece…`, 663,781 B, 1512×1008 px
- `docs/media/arabidopsis_thaliana_turing_wavelength_s0_growth.gif` — artifact_sha256 `fd77056db04b255c…`, 2,480,014 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| leaf primordium stripe wavelength | 111.4951 µm | 111.6 | parabolic sub-bin radial FFT of activator u *(nonlinear peak vs law λ_c; relative error -0.0009)* |
| dispersion pearson r continuum vs grid | 0.999906 1 | 1.0 | Pearson r of σ(k) on 18 wavenumbers *(rmse=0.003364; not a val==ref self-score of the wavelength)* |
| dispersion RMSE | 0.003364 1/time | 0.0 | RMSE between continuum σ(k) and the discrete-Laplacian symbol on the 128 grid *(rmse of the growth-rate curve)* |
| specimen trichome cyan area fraction | 0.0943 1 | 0.2844 | cyan threshold on the Paves SEM (B>0.42, B>R+0.12) after opening/closing *(specimen segmentation vs simulation occupancy; the two fractions are different objects)* |
| specimen trichome coherency | 0.5816 1 | 0.7152 | structure-tensor coherency on the trichome SEM *(non-circular specimen vs simulation coherency)* |
| specimen epidermis foreground fraction | 0.5675 1 | 0.3988 | border-contrast foreground on Boutet epidermis cryo-SEM *(two real specimens; neither equals the model scalar)* |
| equal-diffusion control amplitude ratio | 0.0 1 | 0.0 | std(u|D_u=D_v) / std(u|d=8.4) after the same horizon *(negative control; ratio near 0 means the ablation killed the spots)* |
| trichome nearest-neighbour spacing | 101.1051 µm | 111.6 | nearest-neighbour distances of local maxima of u *(cv=0.1075; ks=0.4948 against N(λ_c, (0.15 λ_c)^2); bootstrap 95% CI [96.60, 105.81] µm)* |
| spacing KS statistic | 0.4948 1 | 0.2968 | Kolmogorov-Smirnov D of NN distances vs Normal(111.6, 16.74) *(p=2.85e-05; value is D, reference is the critical value)* |
| stochastic spectral tail ratio | 1.698 1 | 1.0 | high-k power fraction (k>2 k_c) of the noisy run divided by the deterministic run *(ratio > 1 supports a fatter stochastic tail; this is not their birth-death master equation)* |
| cell-scale predecessor mode count | 0 1 | 4 | floor(18.6 µm / 111.6 µm) on the pavement-cell predecessor at 0 Ma *(same Holocene age; the cell is subcritical)* |
| trichome spacing bootstrap CI half-width | 4.6059 µm | 101.1051 | percentile bootstrap, 500 resamples of nearest-neighbour distances *(95% CI [96.60, 105.81] µm; half-width is not the mean, so the metric is not self-referential)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of loop 3117. Slot is outside campaign_100, so literature was queried directly.
