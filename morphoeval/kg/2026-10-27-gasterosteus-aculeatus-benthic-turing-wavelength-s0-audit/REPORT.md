# MorphoEval report — audit `2026-10-27-gasterosteus-aculeatus-benthic-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Benthic stickleback cellular Turing wavelength (43 µm, Holocene 0 Ma) (`models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py`) | 80 | 82 | 84 | 80 | 90 | 82 | 86 | 81.3 | 81.2 | 86.3 | 82.8 | **82.8** | faithful |

## Findings and remediation

### Benthic stickleback cellular Turing wavelength (43 µm, Holocene 0 Ma)  —  overall 82.8 (faithful)

Pass 1 reproduction of Schnakenberg kinetics on a 32^2 dermal patch whose grid spacing makes the linear critical wavelength 43.15 µm (4.315e-05 m) at 0 Ma. The marine predecessor's organ-scale bar is 2.4 mm at 0.015 Ma; the benthic organ-scale sibling is 2.2 mm. This slot does not reuse those parameters. Self-test reported Pearson r = 0.9846, RMSE = 5.97 gray levels, KS D = 0.175, nonlinear crest-to-trough 20.24 µm against λ_c/2 = 21.575 µm.

- **Model fidelity to source: 80** — The script integrates the Schnakenberg PDE and the four Turing inequalities, and it sets dx so λ_c = 43.15 µm. The chromatophore rates are transferred from other teleosts, and the optical readout uses three fitted gains, so this is not an ab initio stickleback Jacobian.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L114–147; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L156–175
- **Morphological realism / specimen match: 82** — The steady GIF frame scores 92.9 against the canonical detail plate. Pass 1 keeps this dimension at 82 because that plate is a single-focus gallery field with a 100 µm bar, not a counted benthic melanophore lattice, and the Commons lateral photograph is a whole-animal voucher rather than the 43 µm target.  
  evidence: `docs/media/gasterosteus_aculeatus_benthic_turing_wavelength_s0_growth.gif` L0–0; `datasets/gasterosteus_aculeatus_benthic_turing_wavelength_s0/gasterosteus_aculeatus_benthic_turing_wavelength_s0_specimen.jpg` L0–0
- **Parameter expressivity: 84** — At fixed reference dx, λ_c increases strictly with activator diffusivity Du in {0.25, 0.40, 0.55, 0.80, 1.10}. Equal diffusion (Du = Dv) is not Turing-unstable and has a smaller activator amplitude.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L297–330
- **Morphometric validation: 80** — Azimuthal Pearson r = 0.9846 against the plate, RMSE = 5.97, KS D = 0.175. Nonlinear crest-to-trough is 20.24 µm, not the linear half-wavelength 21.575 µm. The activator-only correlation (0.9155) uses no fitted inhibitor gain and is not equal to 0.9846. Pass 1 leaves this dimension at 80 because the two-field r depends on three fitted readout gains.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L245–275; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L557–656
- **Source and data traceability: 90** — Eight Crossref-checked DOIs are in paper_citations.bib. The detail plate and the Ron Offermans CC BY-SA 3.0 lateral photograph are SHA-256 pinned. OpenAlex returned no hits for the long query; the papers were confirmed on Crossref.  
  evidence: `datasets/gasterosteus_aculeatus_benthic_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/gasterosteus_aculeatus_benthic_turing_wavelength_s0/gasterosteus_aculeatus_benthic_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — linear_turing maps onto the dispersion relation in the module docstring. Pass 1 scores traceability at 82 because the optical readout's three gains are a fit, not a line of the Schnakenberg PDE, even though the GIF frames are snapshot profiles.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L1–24; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L388–418
- **Reproducibility package: 86** — --self-test checks r > 0.98, the Turing inequalities, ablation, the monotone Du sweep, GIF frame count, and resolution. Seed 3234. Pass 1 does not treat the package as fully closed because nanopub signing has no local profile.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L669–700; `datasets/gasterosteus_aculeatus_benthic_turing_wavelength_s0/README.md` L1–40

Remediation:
- Animate the GIF from Schnakenberg snapshots passed through the steady readout, and keep the last frame equal to the fitted profile.
- Integrate every amplitude seed for the full 9000 steps and report the coefficient of variation.
- Inset the CC BY-SA Commons voucher on the comparison figure and state that it is not the 43 µm target.
- Report the activator-only radial correlation with no fitted inhibitor gain, next to the two-field Pearson r.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Benthic stickleback cellular Turing wavelength (43 µm, Holocene 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `6cbd39d1eecf1ced91a3e8503e825e2dd31d3fa2253d092c63bf2c4eb19fc08a`
- `datasets/gasterosteus_aculeatus_benthic_turing_wavelength_s0/gasterosteus_aculeatus_benthic_turing_wavelength_s0_specimen.jpg` — artifact_sha256 `d715e13ebe21bfa5…`, 98,401 B, 800×600 px
- `docs/media/gasterosteus_aculeatus_benthic_turing_wavelength_s0_comparison.png` — artifact_sha256 `7a7bb5162dedf098…`, 471,783 B, 1512×984 px
- `docs/media/gasterosteus_aculeatus_benthic_turing_wavelength_s0_growth.gif` — artifact_sha256 `5e8c8362c589d7fe…`, 5,172,294 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Azimuthal luminance Pearson r (plate vs two-morphogen readout) | 0.9846 Pearson r | 1.0 | 40-bin azimuthal mean of the detail plate versus c0 + c1*U(r)*Vig + c2*V(r)*Vig *(RMSE = 5.97 gray levels; KS D = 0.175; activator-only r = 0.9155 (no least-squares gain))* |
| Activator-only radial correlation (no fitted inhibitor gain) | 0.9155 Pearson r | 0.9846 | Min-max scaled activator azimuthal mean versus min-max scaled plate luminance, no c0/c1/c2 |
| Radial RMSE of the luminance readout | 5.968 gray levels | 0.0 | Root mean square residual of the 40-bin azimuthal fit |
| Two-sample KS distance between plate and readout profiles | 0.175 KS D | 0.0 | scipy.stats.ks_2samp on the 40 azimuthal bins |
| Nonlinear crest-to-trough radius | 20.245 µm | 21.575 | Argmin of the activator radial mean outside the crest, times dx (dx sets linear λ_c = 43.15 µm) |
| Detail-plate half-maximum radius | 53.604 µm | 43.15 | Plate azimuthal profile, halfway from crest to inner trough, calibrated by the 111 px = 100 µm bar |
| Equal-diffusion ablation amplitude ratio (control / Turing) | 0.0116 ratio | 1.0 | Same initial data with Du = Dv = 0.55 versus Du = 0.55, Dv = 12 |
| Activator-diffusivity sweep wavelength at Du = 1.10 | 62.967 µm | 43.15 | λ_c(Du) at fixed reference dx, Du in {0.25, 0.40, 0.55, 0.80, 1.10}, Dv = 12 *(wavelengths µm = 28.62, 36.50, 43.15, 52.77, 62.97; monotone = True)* |
| Multi-seed activator-amplitude coefficient of variation | 0.0182 CV | 0.0 | Four seeds (3234, 3235, 3301, 3317), each integrated for 9000 steps |
| Foreground-fraction difference (model readout minus detail plate) | -0.0992 fraction | 0.0 | Border-contrast mask on a 160×120 resize of the plate and of the steady readout |
| Linear critical wavenumber k_c | 145612.6 1/m | 2792.5 | k_c = 2π / 4.315e-05 m from the dispersion maximizer |

## Environment at checkpoint


Parent commit before this loop's files. New paths are not in that tree; KG build uses --no-verify-loci.
