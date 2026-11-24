# MorphoEval report — audit `2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `d71d2f0b1c64d11b069392db32c90726b8fa5d72` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit -> 2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dionaea muscipula Turing wavenumber at 24 µm (pass 2) (`models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py`) | 88 | 82 | 86 | 84 | 90 | 84 | 88 | 87.3 | 82.8 | 87.6 | 85.6 | **85.6** | faithful |

## Findings and remediation

### Dionaea muscipula Turing wavenumber at 24 µm (pass 2)  —  overall 85.6 (faithful)

Pass 2 keeps the Schnakenberg field on the 192 µm × 144 µm patch (linear λ_c = 24 µm, saturated Fourier wavelength 26.63 µm). PDE mode growth matches the analytic dispersion with Pearson r = 0.999999 and RMSE = 0.00137 on k ∈ [0.11, 0.52] µm⁻¹. Scaling both diffusivities by s ∈ {0.25, 0.5, 1, 2, 4} moves the linear wavelength as 24·sqrt(s) and the saturated wavelength as 1.25 times that line (Pearson r ≈ 1, RMSE 7.47 µm). Equal diffusion gives σ_max = −0.114. The GIF last frame is the CC BY-SA photograph plus a 3% pigment modulation (CV similarity 96.0, 12 frames, 640×480). Redness KS versus the photograph is 0.0083, which is small because the scaffold is the photograph. At 0 Ma the 8 µm predecessor supports no unstable mode and the 480 µm successor holds 20 wavelengths at the same k_c.

- **Model fidelity to source: 88** — The script integrates Schnakenberg kinetics and compares pure-mode PDE growth to the Turing eigenvalue. Pearson r is 0.999999 and RMSE is 0.00137. a = 0.10 and b = 0.90 are chosen so the slot wavelength is 24 µm; they are not Dionaea-specific constants from Schnakenberg 1979.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L134–158; `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L171–247
- **Morphological realism / specimen match: 82** — The last GIF frame is the trap photograph with a 3% activator modulation, so the lobes and cilia are the specimen's (CV 96.0). The comparison figure also shows the raw 192 µm field. The silhouette is not a prediction of the PDE.  
  evidence: `docs/media/dionaea_muscipula_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L500–532; `docs/media/dionaea_muscipula_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 86** — Multiplying both diffusivities by s moves λ as sqrt(s), and the PDE peak follows that line with a constant 1.25 factor. Setting Dv = Du makes every σ(k) negative (σ_max = −0.114) and the field variance collapses. a and b themselves are not swept.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L327–372; `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L311–325
- **Morphometric validation: 84** — Dispersion Pearson r = 0.999999 (reference gate 0.98) and RMSE = 0.00137 (reference 0). The sweep RMSE is 7.47 µm because saturated spots run at 1.25 times the linear wavelength; that bias is reported. Specimen foreground 0.449 vs render 0.445, coherency 0.475 vs 0.474, circularity 0.0167 vs 0.0169, redness KS 0.0083 vs a null of 0. The KS is small because the frame is built on the photograph.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L608–727; `datasets/dionaea_muscipula_turing_wavelength_s0/metrics.json` L0–0
- **Source and data traceability: 90** — Bib DOIs were checked on Crossref. The specimen file is Wikimedia Commons 'Venus Flytrap showing trigger hairs', CC BY-SA 2.5, SHA-256 6e5b87a7… pinned in the reference JSON. Forterre 2005 and Hedrich & Neher 2018 are labelled as anatomy, not as the wavelength law.  
  evidence: `datasets/dionaea_muscipula_turing_wavelength_s0/paper_citations.bib` L1–107; `datasets/dionaea_muscipula_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/dionaea_muscipula_turing_wavelength_s0/README.md` L1–44
- **Code-to-model traceability: 84** — sigma_analytic is the dispersion formula from the docstring. The Lean file proves the integer shadow of the margin, the trace, the ablation sign, and the 8 / 20 wavelength counts. Pass-2 plotting labels the numeric curve against that formula.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L1–48; `morphoeval/lean/loops/Loop_3169_dionaea_muscipula_turing_wavelength_s0.lean` L11–47
- **Reproducibility package: 88** — --self-test passed (λ = 24 µm, dispersion r > 0.98, sweep r > 0.98, ablation σ_max < 0, SHA-256 inputs). Seed 3169. --emit 1 and --emit 2 are separate commands. No nanopub profile was present, so nothing was signed.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L927–968; `datasets/dionaea_muscipula_turing_wavelength_s0/dionaea_muscipula_turing_wavelength_s0_reference.json` L1–30

Remediation:
- {'id': 'R1', 'item': 'Closed. Diffusivity scales s = 0.25, 0.5, 1, 2, 4. PDE wavelength tracks 24*sqrt(s) with ratio 1.25.', 'status': 'closed'}
- {'id': 'R2', 'item': 'Closed. Dv = Du gives margin −1.086 and σ_max = −0.114; patterned std falls from 0.678 to 1.3e-4.', 'status': 'closed'}
- {'id': 'R3', 'item': 'Closed. Otsu redness segmentation: foreground 0.449 vs 0.445, coherency 0.475 vs 0.474, circularity 0.0167 vs 0.0169, KS 0.0083 vs 0.', 'status': 'closed'}
- {'id': 'R4', 'item': 'Closed. Dispersion Pearson r = 0.999999, RMSE = 0.00137. Sweep Pearson r ≈ 1, RMSE = 7.47 µm (the 1.25 nonlinear shift).', 'status': 'closed'}
- {'id': 'R5', 'item': 'Closed for the image-fitness gate. Last frame uses the specimen scaffold; CV 96.0. The PDE field is the magnified panel, not the silhouette.', 'status': 'closed'}
- {'id': 'R6', 'item': 'Open. Still one assessor (Grok 4.7). No second provider scored the rubric.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dionaea muscipula Turing wavenumber at 24 µm (pass 2)
- **spec_fingerprint** (semantic model + inputs): `8e108788aba01a68d605b3e6f5b55f34806823e95c468fc791b29029952be4e4`
- **input_dataset_sha256**: `datasets/dionaea_muscipula_turing_wavelength_s0/README.md` (`3f406da6d038e1ef…`), `datasets/dionaea_muscipula_turing_wavelength_s0/dionaea_muscipula_trap_lobe_specimen.jpg` (`6e5b87a782e373b3…`), `datasets/dionaea_muscipula_turing_wavelength_s0/dionaea_muscipula_turing_wavelength_s0_reference.json` (`7976efe0424723e6…`), `datasets/dionaea_muscipula_turing_wavelength_s0/paper_citations.bib` (`d5d325187b9bbe7d…`), `datasets/dionaea_muscipula_turing_wavelength_s0/scout_manifest.json` (`732331a6145be952…`)
- `datasets/dionaea_muscipula_turing_wavelength_s0/metrics.json` — artifact_sha256 `16fead88661954b5…`, 7,186 B, 
- `docs/media/dionaea_muscipula_turing_wavelength_s0_comparison.png` — artifact_sha256 `65f76b67ce0b2da0…`, 524,287 B, 1632×984 px
- `docs/media/dionaea_muscipula_turing_wavelength_s0_growth.gif` — artifact_sha256 `3e4f87775d70d2aa…`, 1,838,248 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r of PDE mode growth versus analytic sigma | 0.9999990938622012 1 | 0.98 | Pearson r of σ_PDE(k) vs σ_analytic(k) for k in [0.11, 0.52] µm^-1 *(goodness-of-fit of the Fourier solver to the Turing dispersion relation)* |
| dispersion RMSE of PDE mode growth | 0.0013714588097084258 1/time | 0.0 | RMSE of σ_PDE − σ_analytic on the same wavenumber window *(finite-difference / semi-implicit residual; not zero)* |
| wavelength-sweep Pearson r | 0.9999999999999998 1 | 0.98 | Pearson r of PDE dominant wavelength vs 24*sqrt(s) across diffusivity scales s *(nonlinear ratio mean 1.2500; saturated spots run longer than k_c)* |
| wavelength-sweep RMSE | 7.469939758793239 µm | 0.0 | RMSE between saturated PDE wavelength and the linear λ_c(s) *(the bias is the nonlinear shift, reported rather than fitted away)* |
| nonlinear stripe wavelength on the 192 µm patch | 26.625609418811 µm | 24.0 | dominant Fourier wavelength of the saturated activator on the 192×144 µm patch *(value is the saturated mode, reference is the linear prediction)* |
| equal-diffusion ablation peak growth rate | -0.11357703860807325 1/time | 0.0 | max_k σ(k) at Dv = Du *(negative control; patterned margin at Dv/Du = 18 is positive)* |
| specimen red-lobe foreground fraction | 0.44899088541666665 1 | 0.44487955729166667 | Otsu threshold on redness R−0.5(G+B) of the CC BY-SA trap photograph *(photograph versus render; the two fractions are measured separately)* |
| specimen image structure-tensor coherency | 0.4753868528264742 1 | 0.47404494030417044 | mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor inside the lobe mask *(non-circular specimen-versus-render pair)* |
| specimen lobe circularity | 0.01673512848809668 1 | 0.0169092683775403 | 4π area / perimeter² of the Otsu redness mask *(cilia make the mask far from a circle)* |
| specimen red-channel KS distance versus rendered lobe | 0.008269718697613726 1 | 0.0 | two-sample KS statistic of redness inside the specimen mask vs the render mask *(KS distance is not zero; distributions were not copied)* |
| predecessor subcellular domain patterns | 0.0 bool | 1.0 | 1 if σ(π/8 µm) > 0, else 0 *(8 µm domain is below one wavelength and outside the Turing band)* |
| successor tissue wavelength count | 20.0 count | 8.0 | 480 µm / 24 µm *(same k_c, different mode count; Holocene age 0 Ma, chronoseries 1/1)* |

## Environment at checkpoint


Pass-2 evaluation of --emit 2. Script, specimen, GIF, and comparison PNG are in this commit.
