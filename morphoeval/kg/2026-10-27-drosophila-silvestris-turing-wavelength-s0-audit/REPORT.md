# MorphoEval report — audit `2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0dc967f22ddaa01ca809b3701e8da57313f6c44d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila silvestris Turing wavenumber at 2.995 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py`) | 86 | 74 | 80 | 76 | 90 | 84 | 88 | 84.0 | 74.8 | 87.6 | 81.4 | **81.4** | faithful |

## Findings and remediation

### Drosophila silvestris Turing wavenumber at 2.995 µm (Holocene, 0 Ma)  —  overall 81.4 (faithful)

Pass 1. Schnakenberg kinetics on a periodic 8-wavelength patch with lambda_c = 2.995e-6 m at gamma = 1 and diffusivity ratio 20. The Kilohana CC BY-SA voucher is the species image. The linearised sweep matched closed-form k_c with Pearson r = 0.9894, but the wavelength correlation was 0.9878 because the peak snapped to a single Fourier mode. The dark-pixel mask had 139 components and covered the body, not just wing spots. Equal diffusion was stable (sigma_max = -0.586). This slot does not claim that the photographed wing spots are a 3 µm Turing pattern (Gompel et al. 2005).

- **Model fidelity to source: 86** — Schnakenberg steady state, Jacobian, and k_c^2 = gamma (D_v f_u + D_u g_v) / (2 D_u D_v) are implemented and checked (trace < 0, det > 0, lambda = 2.995 µm at gamma = 1). The pass-1 wavelength readout is a single Fourier bin, so the numerical lambda sits systematically off the curve.  
  evidence: `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L82–124; `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L161–164
- **Morphological realism / specimen match: 74** — The comparison figure shows the real Kilohana fly, and the GIF scored CV similarity 90.5 against that photograph. The pass-1 mask painted the body and legs (139 components), and the caption band left the last-frame orientation KS at 0.153. The free activator field is a spot lattice, not the animal.  
  evidence: `docs/media/drosophila_silvestris_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L272–273
- **Parameter expressivity: 80** — Analytical lambda falls strictly as 1/sqrt(gamma) across 12 reaction scales. Numerical k correlates at r = 0.9894, but discrete-mode plateaus keep the wavelength correlation at 0.9878, just under the 0.98 gate.  
  evidence: `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L176–188
- **Morphometric validation: 76** — Equal-diffusion ablation gives sigma_max = -0.586 against +0.142 in the ocellus regime. Wavenumber ratio versus D. serrata is 30.80 (bound 30). KS versus the free field is 0.109. Wavelength RMSE and the speckled mask are not yet a clean goodness-of-fit.  
  evidence: `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L205–216
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Murray, Carson & Bryant, Gompel, and Kondo & Miura. Specimen SHA-256 dd936942183615043876595eb722f77a3c7edbc4fa886bd87ca6bd9eedd4d81c, CC BY-SA 3.0, Wikimedia user KarlM, Kilohana.  
  evidence: `datasets/drosophila_silvestris_turing_wavelength_s0/paper_citations.bib` L1–86; `datasets/drosophila_silvestris_turing_wavelength_s0/README.md` L1–28
- **Code-to-model traceability: 84** — k_c, the Jacobian, and the chronoseries ratios are named in the script header and in the Lean module. The pass-1 figure still mixes the voucher, the raw mask, and the activator without a separate nonlinear-onset panel.  
  evidence: `morphoeval/lean/loops/Loop_3204_drosophila_silvestris_turing_wavelength_s0.lean` L36–54
- **Reproducibility package: 88** — SHA-256 pins, seeds 7 and 3204, and --self-test. Pass 1 self-test passed on the argmax sweep. Dataset hashes are checked before any run.  
  evidence: `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L44–79; `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L639–670

Remediation:
- R1. Replace the argmax Fourier peak with a power-weighted centroid so wavelength Pearson r clears 0.98 (pass-1 lambda r = 0.9878, k r = 0.9894).
- R2. Restrict the pigment mask to dorsal wing-band islands. The pass-1 raw dark mask had 139 components and painted the body.
- R3. Make the GIF terminal frame the unmodified voucher. The pass-1 caption band left last-frame orientation KS at 0.153.
- R4. Report the nonlinear onset wavenumber against closed-form k_c, not only the linearised sweep.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila silvestris Turing wavenumber at 2.995 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `94f67ddc05494c16449850f8c780e88f2c89119a3224682e596b88bad888be9c`
- **input_dataset_sha256**: `datasets/drosophila_silvestris_turing_wavelength_s0/drosophila_silvestris_turing_wavelength_s0_reference.json` (`9f4f77b1d9bee8df…`), `datasets/drosophila_silvestris_turing_wavelength_s0/drosophila_silvestris_turing_wavelength_s0_specimen.jpg` (`dd93694218361504…`), `datasets/drosophila_silvestris_turing_wavelength_s0/paper_citations.bib` (`4ff27448f5d20b31…`)
- `docs/media/drosophila_silvestris_turing_wavelength_s0_comparison.png` — artifact_sha256 `0830930fc3e69be6…`, 594,499 B, 1248×984 px
- `docs/media/drosophila_silvestris_turing_wavelength_s0_growth.gif` — artifact_sha256 `d7debf773cd5bad2…`, 1,766,654 B, 720×576 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen foreground fraction (Kilohana voucher segmentation) | 0.5705 area fraction | 0.5 | border-median contrast mask, close, fill holes *(pigment fraction 0.0100, red-eye fraction 0.0125, circularity 0.0195)* |
| specimen pigment connected-component count | 17 components | 25 | gray < 0.22 inside foreground, red-eye pixels removed, 8-connected label *(Pass-1 raw dark-foreground components 705; pass-2 wing-band size filter keeps this count. It is not the tibial cilia count.)* |
| specimen structure-tensor coherency | 0.5176 coherency | 0.4 | mean (λ1-λ2)/(λ1+λ2) of the Gaussian-smoothed structure tensor on the foreground *(foreground aspect ratio (width/height) 1.251)* |
| Pearson r of FFT wavenumber versus dispersion k_c across gamma | 0.9981 Pearson r | 1.0 | 12-point sweep of reaction scale gamma; dominant Fourier mode of the linearised PDE versus closed-form k_c *(lambda Pearson r 0.9972; RMSE 0.2084 µm; reference-table r 1.0000)* |
| RMSE of numerical versus analytical wavelength | 0.2084 µm | 0.0 | root mean square of lambda_FFT - lambda_dispersion over the gamma sweep *(at gamma nearest 1, numerical lambda 3.3983 µm versus 2.995 µm)* |
| KS distance of Sobel orientation, specimen versus free Turing field | 0.1085 KS | 0.0 | 12-bin magnitude-weighted Sobel orientation histograms, discrete Kolmogorov-Smirnov *(last-frame versus specimen KS 0.0119)* |
| equal-diffusion ablation max growth rate | -0.5857 1/time | 0.0 | max_k Re sigma(k) at D_v = D_u (d = 1), same D_u as the ocellus regime *(ocellus d=20 sigma_max 0.1421; stripe contrasts d=8 -0.2639, d=10 -0.1479)* |
| wavenumber ratio versus Drosophila serrata predecessor | 30.8047 k_this/k_pred | 30.0 | lambda_serrata / lambda_silvestris with lambda_serrata = 9.226e-5 m and lambda_silvestris = 2.995e-6 m *(successor ratio k_this/k_simulans 27.5659 versus bound 27)* |
| nonlinear early-frame wavenumber over linear k_c | 0.8486 k_nl/k_c | 1.0 | power-weighted spectral centroid of nonlinear snapshot 3, divided by the closed-form k_c at gamma=1 *(k_c 6.2832, early 5.3318, saturated 5.7067)* |

## Environment at checkpoint

- python: 3.12.3
- note: numpy, scipy, Pillow, matplotlib. No ~/.nanopub/profile.yml on this runner.

Pass-1 checkpoint. Linearised Schnakenberg sweep used the discrete argmax peak.
