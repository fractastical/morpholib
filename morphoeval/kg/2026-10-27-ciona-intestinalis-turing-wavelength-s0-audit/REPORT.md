# MorphoEval report — audit `2026-10-27-ciona-intestinalis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `25e28a4514fe1c49bc401d10321c28551fd40114` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ciona intestinalis Turing wavelength at 51 µm (5.056e-05 m), Holocene 0 Ma (`models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py`) | 86 | 78 | 84 | 78 | 88 | 84 | 82 | 85.3 | 78.0 | 84.5 | 82.2 | **82.2** | faithful |

## Findings and remediation

### Ciona intestinalis Turing wavelength at 51 µm (5.056e-05 m), Holocene 0 Ma  —  overall 82.2 (faithful)

Pass 1. Schnakenberg activator-inhibitor kinetics on a periodic 4-wavelength patch, with gamma set so lambda_c = 5.056e-05 m (tier label 51 µm, log10 = -4.296, tier 4_cellular_10um, detail). Age 0 Ma, chronoseries 1/1. Critical-mode log-amplitude Pearson r = 0.9994 on the linear window. Numerical radial-FFT wavelength 50.05 µm against the 50.56 µm pin. Equal-diffusion amplitude ratio 1.98e-5 and gamma=0 ratio 2.80e-4, against a Turing ratio 2.95. YPM IZ 082346 (CC0) body length 34.0 mm on the 10 mm bar; adult pigment spacing 873 µm, which is not the detail-tier wavelength. The image-fitness JSON for the loop is not assembled yet.

- **Model fidelity to source: 86** — The script integrates the Schnakenberg parenthesis with a semi-implicit Fourier step and sets gamma from the Turing formula so lambda_c matches 5.056e-05 m to numerical tolerance. Periodic boundaries match the Fourier symbol. u and v are not dressed up as named genes.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L88–131; `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L140–202
- **Morphological realism / specimen match: 78** — The comparison figure places YPM IZ 082346 beside the body mask, the stripe field, and a specimen-conditioned render. CV similarity of the 12-frame 560×779 GIF against that photograph saturates the scorer cap at 96. The loop fitness JSON that combines CV with these rubric subscores is not written yet, so this pass does not treat the visual match as closed.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L298–308; `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L424–444; `docs/media/ciona_intestinalis_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 84** — Analytical lambda falls monotonically as Dv/Du runs through 4, 6, 8, 10, 16, 24 (57.0 µm down to 48.6 µm), crossing the pin at ratio 10. Short integrations at ratio 4 do not amplify; that is reported rather than hidden. Equal diffusion and gamma=0 both decay.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L252–295
- **Morphometric validation: 78** — Pearson r of log|A(k_c)| versus t is 0.9994 (RMSE on the log-amplitude residual is stored). Radial FFT wavelength is 50.05 µm against the 50.56 µm pin. Body length 34.0 mm, circularity 0.248 against a circle null of 1, KS statistic 0.434 against a zero-difference null. Adult spacing 873 µm is a different scale. The fitness file that lists these as non-circular metrics is still missing.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L204–250; `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L311–363; `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L446–463
- **Source and data traceability: 88** — DOIs were checked on Crossref/OpenAlex. The specimen file is SHA-256 pinned (bacc05b7…) with photographer, CC0, catalog number, and GBIF dataset DOI 10.15468/0lkr3w. Slot 3148 is outside campaign_100_slots.yaml; the scout manifest says so.  
  evidence: `datasets/ciona_intestinalis_turing_wavelength_s0/paper_citations.bib` L1–20; `datasets/ciona_intestinalis_turing_wavelength_s0/ciona_intestinalis_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — The module docstring states the PDE, the wavenumber formula, the periodic boundary, and the claim that u and v are not named genes. gamma_for_wavelength is the line that enforces the pin. Lean Loop_3148 proves the integer form of the trace, determinant, cross term, discriminant, equal-diffusion ablation, and the scale sandwich.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3148_ciona_intestinalis_turing_wavelength_s0.lean` L22–55
- **Reproducibility package: 82** — --self-test checks the sign of the Turing conditions, the wavelength pin, monotonic lambda(ratio), Pearson r > 0.98, and both ablations. Seed 3148. The fitness JSON writer exists but has not been run for this pass.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L729–760; `datasets/ciona_intestinalis_turing_wavelength_s0/README.md` L1–24

Remediation:
- {'id': 'R1', 'item': 'Write morphoeval/runs/2026-10-27-ciona-intestinalis-turing-wavelength-s0-loop-image-fitness.json with image_fitness = 0.55*CV + 0.30*specimen-match + 0.15*morphometric-validation, n_frames, resolution, and non-circular specimen metrics (value != reference).'}
- {'id': 'R2', 'item': 'Keep the dispersion panel zoomed on the unstable band so sigma(k_c) > 0 is visible, and quote the equal-diffusion amplitude ratio in scientific notation rather than as 0.00.'}
- {'id': 'R3', 'item': 'Publish the adult pigment spacing (about 873 µm) next to the 50.56 µm pin as an explicit scale-transfer control in the pass-2 measurements sidecar.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `specimen_comparison` → `specimen-match`: Assemble the loop image-fitness JSON from the measured CV similarity and the pass-2 rubric subscores.
- **[OPEN · P=1.1]** `morphometric_validation` → `morphometric-validation`: Record adult spacing versus the 51 µm pin as a scale-transfer control with value != reference.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ciona intestinalis Turing wavelength at 51 µm (5.056e-05 m), Holocene 0 Ma
- **spec_fingerprint** (semantic model + inputs): `cc8077f6425fa5fa7578c008956e84c51354ee6ba24c3850292df68d0c32e0f9`
- **input_dataset_sha256**: `datasets/ciona_intestinalis_turing_wavelength_s0/Ciona_intestinalis_YPM_IZ_082346.jpg` (`bacc05b7298b188c…`), `datasets/ciona_intestinalis_turing_wavelength_s0/ciona_intestinalis_turing_wavelength_s0_reference.json` (`ba8af50f6e980b7c…`), `datasets/ciona_intestinalis_turing_wavelength_s0/paper_citations.bib` (`6b138fabe89be7a0…`)
- `docs/media/ciona_intestinalis_turing_wavelength_s0_comparison.png` — artifact_sha256 `cc584dd710e4101b…`, 277,777 B, 1512×984 px
- `docs/media/ciona_intestinalis_turing_wavelength_s0_growth.gif` — artifact_sha256 `e120696368881972…`, 1,385,890 B, 560×779 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe_wavelength_um | 50.05183748686624 µm | 50.559999999999995 | Radial FFT peak of activator u; parabolic sub-bin interpolation. RMSE not used on the peak. *(Numerical peak is not forced equal to the pin.)* |
| critical_mode_log_amplitude_pearson_r | 0.999390494829923 1 | 0.98 | Pearson r and RMSE of log|A(k_c)| versus t in the linear window *(Gate is r > 0.98; the fitted r is a separate number.)* |
| specimen_foreground_fraction | 0.2977914554670529 1 | 0.5 | Largest connected component on the CC0 YPM photograph *(Specimen segmentation, not a self-copy of a reference.)* |
| specimen_contour_circularity | 0.24771186677158408 1 | 1.0 | Contour circularity of the body mask *(Vase outline is far from a circle.)* |
| specimen_body_length_mm | 33.95454545454545 mm | 10.0 | Bounding-box height times µm/px from the white 10 mm bar *(Adult body versus the bar length.)* |
| adult_pigment_spacing_um | 872.7272727272726 µm | 50.559999999999995 | FFT of the high-pass red-channel transect inside the body mask *(Adult photograph resolves millimetre speckles, not the 51 µm law. Scale-transfer control.)* |
| equal_diffusion_amplitude_ratio | 1.9757219312794065e-05 1 | 2.9452267353620942 | Negative control: D_u = D_v amplitude ratio versus the Turing ratio *(Equal diffusion must not amplify the critical mode.)* |
| pure_diffusion_amplitude_ratio | 0.00027972213368463464 1 | 1.0 | Ablation control, γ = 0, amplitude ratio *(Decay ratio is below 1.)* |
| ks_specimen_vs_activator | 0.43400000000000005 1 | 0.0 | Two-sample KS (and p-value) between body luminance and mapped activator; CV of the wavelength sweep is stored separately *(Distributions are not the same object; KS is not zero.)* |
| sweep_wavelength_cv | 0.05455172534481563 1 | 0.0 | Coefficient of variation of analytical λ across D_v/D_u *(λ moves when the diffusion ratio moves.)* |

## Environment at checkpoint

- python: 3.12.3
- note: numpy, scipy, pillow, matplotlib. No ~/.nanopub/profile.yml on this runner.

Pass 1. Schnakenberg wavelength at 5.056e-05 m on Ciona intestinalis, Holocene 0 Ma, chronoseries 1/1.
