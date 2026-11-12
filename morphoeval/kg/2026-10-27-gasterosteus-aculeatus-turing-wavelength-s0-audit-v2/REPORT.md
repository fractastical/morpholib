# MorphoEval report — audit `2026-10-27-gasterosteus-aculeatus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `4fdc4af8f8347d55b64a1de2f47c8a0931a60888` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-gasterosteus-aculeatus-turing-wavelength-s0-audit -> 2026-10-27-gasterosteus-aculeatus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Holocene Gasterosteus aculeatus cellular Turing wavelength (56 µm) (`models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py`) | 88 | 84 | 86 | 84 | 91 | 86 | 88 | 87.3 | 84.0 | 88.5 | 86.3 | **86.3** | faithful |

## Findings and remediation

### Holocene Gasterosteus aculeatus cellular Turing wavelength (56 µm)  —  overall 86.3 (faithful)

Pass 2 keeps the same Schnakenberg carrier (Holocene analytic λ_c = 56.00 µm; marine 61.81 µm; benthic 52.23 µm). Parabolic radial-FFT interpolation moves the Holocene nonlinear peak to 54.99 µm. Gamma-sweep Pearson r is 0.9938 (RMSE 4.88 µm). Absolute luminance < 95 gives specimen melanin area 0.266 versus render 0.097. The saturated benthic peak is 57.90 µm, so the nonlinear lattice does not follow the analytic contraction. Posterior bar r stays 0.372 because that half was not fit. GIF cv_sim_similarity remains 88.4 at 960×548, 12 frames.

- **Model fidelity to source: 88** — The script integrates the Schnakenberg PDE and evaluates the onset formula k_c² = γ (Dv f_u + Du g_v) / (2 Du Dv) for three values of a and for the equal-diffusion control, which is stable. Lean checks the cleared trace, determinant, discriminant and the analytic contraction. Deduction: a, Du, Dv and γ are chosen to place the Holocene carrier on the assigned 56 µm, and the stickleback papers are genetic rather than a published parameter table.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L67–107; `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L113–141; `morphoeval/lean/loops/Loop_3233_gasterosteus_aculeatus_turing_wavelength_s0.lean` L20–48; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/gasterosteus_aculeatus_turing_wavelength_s0_reference.json` L78–96
- **Morphological realism / specimen match: 84** — cv_sim_similarity against the real Krisp specimen is 88.4 (12 frames, 960×548). The comparison figure shows the photograph, the body mask, and the barred composite on that background. Column-luminance r is 0.851, not 1. Melanin area inside the mask is 0.266 on the specimen and 0.097 on the render, so the model is too pale, and the held-out bar profile only reaches r = 0.372. Those gaps keep this below the low 90s.  
  evidence: `docs/media/gasterosteus_aculeatus_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gasterosteus_aculeatus_turing_wavelength_s0_comparison.png` L0–0; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/run_metrics.json` L1–40
- **Parameter expressivity: 86** — Raising γ from 3 to 10 shortens both analytic and nonlinear wavelengths (Pearson r = 0.9938). Setting Du = Dv collapses the activator standard deviation from 0.807 to ~0. Changing a from 0.28 to 0.20 to 0.14 shortens analytic λ from 61.81 to 56.00 to 52.23 µm. The saturated benthic FFT peak does not follow that last step (57.90 µm), which is reported rather than tuned away.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L320–338; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/run_metrics.json` L1–30; `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L36–52
- **Morphometric validation: 84** — Pass-1 remediation is in the code. Parabolic interpolation puts the Holocene nonlinear peak at 54.99 µm against the 56.00 µm reference (not equal). Absolute luminance < 95 separates specimen melanin area 0.266 from render 0.097. KS D is 0.685 versus the equal-diffusion field and 0.179 versus specimen luminance inside the mask. Sweep r = 0.9938 with RMSE 4.88 µm. Posterior r remains 0.372 because R4 forbids fitting that half. The benthic nonlinear peak (57.90 µm) misses the analytic 52.23 µm reference; that miss is the score cap.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L164–178; `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L387–396; `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L248–281; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/run_metrics.json` L1–45
- **Source and data traceability: 91** — Same pinned DOIs and SHA-256 specimen as pass 1, plus Kondo & Miura 2010 as the review that states the wavelength-selection framework used to read k_c. Scout manifest records that slot 3233 does not collide with the 2.4 mm marine or 2.2 mm benthic sibling scripts.  
  evidence: `datasets/gasterosteus_aculeatus_turing_wavelength_s0/paper_citations.bib` L22–40; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/scout_manifest.json` L5–40; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/README.md` L1–30
- **Code-to-model traceability: 86** — Dispersion, simulator, sweep, segmentation, held-out cosine and the GIF compositor are separate functions cited by the math KG. Pass-2 measurement changes (interpolation, absolute threshold) are local to those functions.  
  evidence: `morphoeval/math_kg_loops/delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.yaml` L20–55; `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L143–178
- **Reproducibility package: 88** — --self-test passed after the pass-2 readout change: analytic triple exact, sweep r = 0.9938 > 0.98, ablation std below 5% of the pattern, GIF 12 frames at 960×548, specimen SHA checked. Pass-1 metrics file is kept beside the new run_metrics.json. Nanopub signing was not available (~/.nanopub/profile.yml absent).  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-turing-wavelength-s0.py` L676–694; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/run_metrics.json` L0–0; `datasets/gasterosteus_aculeatus_turing_wavelength_s0/pass1_run_metrics.json` L0–0

Remediation:
- Resolved R1: melanin area uses luminance < 95 (specimen 0.266, render 0.097).
- Resolved R2: dominant_wavelength_cells interpolates the radial peak.
- Resolved R3: re-measured peaks are marine 60.25, Holocene 54.99, benthic 57.90 µm. The saturated benthic pattern does not shorten past the Holocene peak.
- Resolved R4: posterior r remains 0.372; the posterior half is still not in the fit.
- Open: nonlinear benthic radial peak does not track analytic 52.23 µm. Not adjusted by changing a or the FFT window.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Holocene Gasterosteus aculeatus cellular Turing wavelength (56 µm)
- **spec_fingerprint** (semantic model + inputs): `3db7c83138922a03aef82aaf803ecbe744b83029257f1146668b5acc01d7c2cb`
- **input_dataset_sha256**: `datasets/gasterosteus_aculeatus_turing_wavelength_s0/gasterosteus_aculeatus_turing_wavelength_s0_reference.json` (`a53e439d8af1eeb7…`), `datasets/gasterosteus_aculeatus_turing_wavelength_s0/gasterosteus_aculeatus_turing_wavelength_s0_specimen.jpg` (`69b13a9c5037e468…`), `datasets/gasterosteus_aculeatus_turing_wavelength_s0/paper_citations.bib` (`6e834bc83f029f3b…`), `datasets/gasterosteus_aculeatus_turing_wavelength_s0/scout_manifest.json` (`bc4c0aa8b21f50ae…`)
- `datasets/gasterosteus_aculeatus_turing_wavelength_s0/pass1_run_metrics.json` — artifact_sha256 `a92a516a7b932544…`, 904 B, 
- `datasets/gasterosteus_aculeatus_turing_wavelength_s0/run_metrics.json` — artifact_sha256 `72a164112b7f6b5a…`, 904 B, 
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
