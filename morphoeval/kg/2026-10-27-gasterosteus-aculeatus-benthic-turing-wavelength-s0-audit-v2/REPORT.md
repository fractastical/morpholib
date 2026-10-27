# MorphoEval report — audit `2026-10-27-gasterosteus-aculeatus-benthic-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-gasterosteus-aculeatus-benthic-turing-wavelength-s0-audit -> 2026-10-27-gasterosteus-aculeatus-benthic-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Benthic stickleback cellular Turing wavelength (43 µm, Holocene 0 Ma) (`models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py`) | 88 | 90 | 88 | 91 | 92 | 88 | 90 | 88.0 | 90.4 | 90.1 | 89.6 | **89.6** | faithful |

## Findings and remediation

### Benthic stickleback cellular Turing wavelength (43 µm, Holocene 0 Ma)  —  overall 89.6 (faithful)

Pass 2 keeps the Schnakenberg patch at λ_c = 43.15 µm (4.315e-05 m, 0 Ma) and closes the pass-1 remediations. The GIF is twelve Schnakenberg snapshots rendered with the steady readout; the last frame is the fitted profile (computer-vision similarity 92.9, 800×600). Self-test: Pearson r = 0.9846, RMSE = 5.97, KS D = 0.175, crest-to-trough 20.24 µm versus λ_c/2 = 21.575 µm. Du sweep wavelengths increase; equal diffusion does not pattern. Nanopub signing was not available (~/.nanopub/profile.yml absent).

- **Model fidelity to source: 88** — Pass 2 still uses the published Schnakenberg reaction and the Turing discriminant, with dx fixed by λ_c = 43.15 µm. GIF frames now come from PDE snapshots through that readout rather than an independent crossfade. Rates remain a transfer from other teleosts, which caps the score below 95.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L114–147; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L388–418
- **Morphological realism / specimen match: 90** — Last GIF frame versus the detail plate scores 92.9 on the 12-bin orientation, 24-bin radial FFT, and structure-tensor coherency comparison (800×600, 12 frames). The comparison figure insets the Commons voucher and labels it as the whole-animal photograph, not the 43 µm target. The detail plate itself is the repository canonical single-focus field.  
  evidence: `docs/media/gasterosteus_aculeatus_benthic_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gasterosteus_aculeatus_benthic_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L426–432
- **Parameter expressivity: 88** — λ_c at the reference dx rises strictly across Du = 0.25, 0.40, 0.55, 0.80, 1.10, with the reference Du = 0.55 sitting on 43.15 µm. The equal-diffusion control is linearly stable to diffusion-driven instability and has a smaller amplitude.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L297–330
- **Morphometric validation: 91** — Two-field azimuthal Pearson r = 0.9846 (reference ideal 1.0), RMSE = 5.97 gray levels (reference 0), KS D = 0.175 (reference 0). Crest-to-trough 20.24 µm is not λ_c/2 = 21.575 µm. Plate half-max radius is calibrated by the 111 px = 100 µm bar and is not 43.15 µm. Activator-only r, with no fitted inhibitor gain, is reported separately and is not equal to 0.9846. Foreground fractions of the plate and the readout are computed apart.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L245–275; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L557–656; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L488–510
- **Source and data traceability: 92** — BibTeX DOIs were checked on Crossref. Specimen SHA-256 digests are verified at startup. The Commons file records creator Ron Offermans and CC BY-SA 3.0. Scout found no slug collision with gasterosteus-aculeatus-benthic-turing-wavelength-s0; the organ-scale sibling is a different slug.  
  evidence: `datasets/gasterosteus_aculeatus_benthic_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/gasterosteus_aculeatus_benthic_turing_wavelength_s0/README.md` L1–50; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L78–100
- **Code-to-model traceability: 88** — Each modelling step is a named function tied to the dispersion relation, the radial readout, the Du sweep, or the ablation. Lean theorems in MathOfBiology.Loop_3234 prove the wavelength ordering, the wavenumber inversion, and the integerised discriminant with no sorry.  
  evidence: `morphoeval/lean/loops/Loop_3234_gasterosteus_aculeatus_benthic_turing_wavelength_s0.lean` L16–50; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L114–147
- **Reproducibility package: 90** — python3 --self-test is deterministic under seed 3234, checks r > 0.98, ablation, monotone sweep, and a GIF of at least 10 frames and 480 px. All four amplitude seeds run 9000 steps. Dependencies are the repository scientific stack. Signing could not be run.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L333–343; `models/2026-delphi10000-gasterosteus-aculeatus-benthic-turing-wavelength-s0.py` L669–710; `datasets/gasterosteus_aculeatus_benthic_turing_wavelength_s0/README.md` L36–52

Remediation:
- Replace transferred Du, Dv, and gamma with a Jacobian measured on benthic stickleback chromatophores if such a count is published. Not possible from the sources in this pack.
- A second, independent assessor has not scored this audit.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Benthic stickleback cellular Turing wavelength (43 µm, Holocene 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `6cbd39d1eecf1ced91a3e8503e825e2dd31d3fa2253d092c63bf2c4eb19fc08a`
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


Parent commit. New loop files are added in the loop commit; locus check is skipped with --no-verify-loci because those paths are not in the parent tree.
