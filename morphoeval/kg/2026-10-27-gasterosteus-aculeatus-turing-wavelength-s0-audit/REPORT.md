# MorphoEval report — audit `2026-10-27-gasterosteus-aculeatus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `4fdc4af8f8347d55b64a1de2f47c8a0931a60888` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Holocene Gasterosteus aculeatus cellular Turing wavelength (56 µm) (`models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py`) | 84 | 80 | 78 | 70 | 90 | 84 | 86 | 82.0 | 76.2 | 86.8 | 81.1 | **81.1** | faithful |

## Findings and remediation

### Holocene Gasterosteus aculeatus cellular Turing wavelength (56 µm)  —  overall 81.1 (faithful)

Pass 1 implements Schnakenberg kinetics whose analytic Holocene wavelength is 56.00 µm, with marine a=0.28 at 61.81 µm and benthic a=0.14 at 52.23 µm on the same dx. Gamma sweep Pearson r against analytic λ_c was 0.9973 with RMSE 4.28 µm. Equal diffusion decays. The growth GIF scores cv_sim_similarity 88.4 against the Krisp specimen. Two readouts are not yet trustworthy: nearest-bin radial FFT and a percentile darkness that sits at ~0.35 on both images.

- **Model fidelity to source: 84** — Jacobian, Turing conditions, k_c and the equal-diffusion control match the Schnakenberg linearisation, and four Lean theorems check the cleared integers (trace −58/75, det 108/75, discriminant 107296/625, and the 61.81 > 56.00 > 52.23 µm contraction). The parameters are reconstructed to the assigned 56 µm scale; Greenwood 2011 and Miller 2007 do not publish this PDE.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L67–107; `morphoeval/lean/loops/Loop_3233_gasterosteus_aculeatus_turing_wavelength_s0.lean` L20–48; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/gasterosteus_aculeatus_turing_wavelength_s0_reference.json` L78–95
- **Morphological realism / specimen match: 80** — The 12-frame 960×548 growth GIF scores 88.4 on the canonical Sobel / radial-FFT / coherency comparison to the CC BY 4.0 Krisp photograph. The composite keeps the real background and paints bars inside the segmented body. It is not a pixel copy (column-luminance r = 0.851). The cellular panel is a spot lattice at 56 µm, which the photograph cannot resolve.  
  evidence: `docs/media/gasterosteus_aculeatus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/gasterosteus_aculeatus_turing_wavelength_s0_growth.gif` L0–0; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/pass1_run_metrics.json` L1–40
- **Parameter expressivity: 78** — γ in {3,4,6,8,10} moves analytic and measured wavelength together (r = 0.9973) in the direction λ ∝ γ^(−1/2). Equal diffusion (Du = Dv) yields activator std ~0 against pattern std 0.81. The nonlinear marine/Holocene/benthic separation is compressed relative to the analytic 61.81 / 56.00 / 52.23 µm triple, and the sweep RMSE is 4.28 µm.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L320–338; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/pass1_run_metrics.json` L1–25; `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L113–141
- **Morphometric validation: 70** — Sweep Pearson r and the ablation KS are real comparisons with references that the values do not copy. The body-darkness fraction in this pass was a within-image percentile, so both images scored ~0.35 (pass1_run_metrics.json photo_dark_fraction 0.350 vs render 0.350). Posterior flank r is only 0.372. Those two readouts are the remediation.  
  evidence: `datasets/gasterosteus_aculeatus_turing_wavelength_s0/pass1_run_metrics.json` L1–45; `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L676–694
- **Source and data traceability: 90** — Seed DOI Greenwood 2011 was the top OpenAlex hit for the slot query. Turing 1952, Schnakenberg 1979, Miller 2007, Tapanes 2024 and Nakamasu 2009 were checked against Crossref or OpenAlex on 2026-10-27. The specimen is Holger Krisp, CC BY 4.0, SHA-256 69b13a9c…b67f. The gallery JPEG was rejected because it is a synthetic field.  
  evidence: `datasets/gasterosteus_aculeatus_turing_wavelength_s0/paper_citations.bib` L1–98; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/gasterosteus_aculeatus_turing_wavelength_s0_reference.json` L60–77; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 84** — Each modelling step maps to a formula in the module docstring and the reference JSON. The Lean module cites the same cleared integers the script's Jacobian produces at a=1/5, b=1, Du=1, Dv=30. There is no equation number in the stickleback genetics papers to pin, and that limit is stated.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L1–32; `morphoeval/math_kg_loops/delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.yaml` L1–55
- **Reproducibility package: 86** — Seed 3233, dt = 0.004 < 1/(4 Dv) = 0.0083, --self-test checks the analytic triple, sweep r > 0.98, GIF frames and resolution. SHA-256 of the specimen is checked at runtime. No per-slot lockfile beyond the cloud image.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L676–716; `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L180–186

Remediation:
- R1. Replace the within-image percentile darkness with an absolute luminance threshold so specimen and render fractions are not forced near 0.35.
- R2. Replace nearest-bin radial FFT with parabolic sub-bin interpolation so nonlinear wavelengths are not quantized in ~5 µm steps.
- R3. Re-measure marine, Holocene and benthic peaks after that interpolation and state whether the saturated benthic pattern actually shortens.
- R4. Leave the posterior flank half out of the cosine fit; do not raise posterior r by fitting the test segment.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Holocene Gasterosteus aculeatus cellular Turing wavelength (56 µm)
- **spec_fingerprint** (semantic model + inputs): `3db7c83138922a03aef82aaf803ecbe744b83029257f1146668b5acc01d7c2cb`
- **input_dataset_sha256**: `datasets/gasterosteus_aculeatus_turing_wavelength_s0/gasterosteus_aculeatus_turing_wavelength_s0_reference.json` (`a53e439d8af1eeb7…`), `datasets/gasterosteus_aculeatus_turing_wavelength_s0/gasterosteus_aculeatus_turing_wavelength_s0_specimen.jpg` (`69b13a9c5037e468…`), `datasets/gasterosteus_aculeatus_turing_wavelength_s0/paper_citations.bib` (`6e834bc83f029f3b…`), `datasets/gasterosteus_aculeatus_turing_wavelength_s0/scout_manifest.json` (`bc4c0aa8b21f50ae…`)
- `docs/media/gasterosteus_aculeatus_turing_wavelength_s0_comparison.png` — artifact_sha256 `8d5a2170cfe64f0e…`, 544,837 B, 1632×984 px
- `docs/media/gasterosteus_aculeatus_turing_wavelength_s0_growth.gif` — artifact_sha256 `8beaa3da6da52666…`, 5,427,300 B, 960×548 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Holocene nonlinear cellular wavelength versus analytic k_c target | 54.991 µm | 56.0 | Radial FFT peak of the saturated Schnakenberg activator, times dx anchored at analytic 56 µm *(analytic 56.0000 µm; nonlinear peak is not forced equal to the target)* |
| Marine predecessor nonlinear cellular wavelength | 60.2511 µm | 61.8063 | Same grid and dx, a=0.28 (higher Kitlg direction), seed 3234 *(Predecessor at this cellular tier, not the 2400 µm organ envelope)* |
| Benthic successor nonlinear cellular wavelength | 57.9014 µm | 52.235 | Same grid and dx, a=0.14 (lower Kitlg direction), seed 3235 *(Successor at this cellular tier, not the 2200 µm organ envelope)* |
| Gamma-sweep Pearson r of nonlinear wavelength versus analytic λ_c | 0.9938 Pearson r | 1.0 | γ in [3.0, 4.0, 6.0, 8.0, 10.0]; pearsonr(analytic λ(γ), radial-FFT λ(γ)) *(RMSE 4.8793 µm; measured [88.55, 73.969, 54.991, 48.522, 44.618])* |
| Gamma-sweep RMSE of wavelength | 4.8793 µm | 0.0 | sqrt mean square deviation of nonlinear λ from analytic λ across the γ sweep |
| Equal-diffusion ablation activator standard deviation | 0.0 concentration | 0.8068 | Du = Dv = 1, same γ, seed, and step count as the patterned run *(pattern std 0.8068; equal-diffusion drive is negative so the field decays)* |
| KS statistic, patterned activator versus equal-diffusion control | 0.6849 KS D | 0.0 | scipy.stats.ks_2samp on strided activator samples *(p = 0.000e+00)* |
| Specimen mask melanin area fraction versus rendered composite | 0.0974 fraction of body pixels with luminance < 95 | 0.266 | Absolute grey threshold 95 inside the colour-distance body mask, last composite versus the photograph *(Pass-1 percentile darkness collapsed to ~0.35 on both images; this absolute threshold does not)* |
| Specimen versus render luminance KS inside the body mask | 0.1795 KS D | 0.0 | ks_2samp of specimen luminance and last-frame luminance on the segmented body *(p = 0.000e+00; column-luminance Pearson r = 0.8510, RMSE = 4.471 grey levels)* |
| Held-out posterior flank Pearson r of the bar envelope | 0.3716 Pearson r | 0.6972 | Cosine frequency and phase fit on the anterior half of the flank pigment profile; r on the posterior half *(Organ-scale bars on this photograph. This is not the 56 µm carrier.)* |
| Specimen body area fraction | 0.1746 fraction of frame | 0.2 | Largest component of a border-colour distance mask, closed and filled |

## Environment at checkpoint


Script, specimen, Lean module, and media at this commit. Audit text and nanopubs follow in the next commit.
