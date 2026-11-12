# MorphoEval report — audit `2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0dc967f22ddaa01ca809b3701e8da57313f6c44d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit -> 2026-10-27-drosophila-silvestris-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila silvestris Turing wavenumber at 2.995 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py`) | 91 | 86 | 92 | 90 | 91 | 88 | 92 | 91.3 | 87.5 | 90.6 | 89.6 | **89.6** | faithful |

## Findings and remediation

### Drosophila silvestris Turing wavenumber at 2.995 µm (Holocene, 0 Ma)  —  overall 89.6 (faithful)

Pass 2 closes the pass-1 readout, mask, and terminal-frame gaps. Power-weighted spectral centroids track the dispersion curve with Pearson r = 0.9981 (wavenumber) and 0.9972 (wavelength), RMSE 0.208 µm. The dorsal wing-band mask keeps 17 dark islands (the raw dark mask had 139). The GIF has 12 frames at 720x576; its last frame is the Kilohana voucher, and computer-vision similarity against that photograph is 96.0 with last-frame orientation KS 0.012. Equal diffusion remains stable. k_this/k_serrata = 30.80 and k_this/k_simulans = 27.57 at age 0 Ma. The 2.995 µm wavelength is the assigned detail tier, not a measurement from Carson & Bryant (1979) and not an identification of the photographed wing spots (Gompel et al. 2005).

- **Model fidelity to source: 91** — The Jacobian, critical wavenumber, and equal-diffusion stability condition match Turing (1952) and Schnakenberg (1979). Pass 2 adds the centroid estimator and the nonlinear onset ratio without changing the kinetics. Remaining gap: the nonlinear field is a numerical integration, not a matched asymptotic spot law.  
  evidence: `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L82–124; `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L167–184; `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L219–251
- **Morphological realism / specimen match: 86** — Side-by-side Kilohana voucher and a wing-band mask of 17 components (raw dark mask 139, the pass-1 failure). GIF last frame is the unmodified photograph: CV similarity 96.0, last-frame orientation KS 0.012, 12 frames, 720x576. The free activator panel remains a subcellular spot lattice, which is the model and not a portrait of the fly.  
  evidence: `docs/media/drosophila_silvestris_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L269–287; `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L411–416
- **Parameter expressivity: 92** — Analytical wavelength is strictly decreasing in gamma. Numerical centroids give Pearson r = 0.9981 on k and 0.9972 on lambda across the same 12 scales. Equal diffusion (d = 1) is a negative control with sigma_max < 0, while d = 20 is unstable.  
  evidence: `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L176–203; `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L205–216
- **Morphometric validation: 90** — Wavelength Pearson r = 0.9972 and RMSE = 0.208 µm against the dispersion curve. Specimen foreground fraction, wing-spot count, and coherency differ from their reference priors (not self-copies). Orientation KS of the free field versus the photograph is reported separately from the last-frame KS. The wing mask is still a luminance heuristic.  
  evidence: `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L498–590; `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L470–473
- **Source and data traceability: 91** — Eight citable works, all with DOIs checked on Crossref. Specimen file, license, artist, and SHA-256 are in the reference JSON and README. Scale 3.0 µm and age 0 Ma are stated there. The assigned wavelength is labeled as assigned.  
  evidence: `datasets/drosophila_silvestris_turing_wavelength_s0/paper_citations.bib` L45–86; `datasets/drosophila_silvestris_turing_wavelength_s0/README.md` L1–36
- **Code-to-model traceability: 88** — Lean theorems pin the nanometre ordering, the factor-of-30 wavenumber gap, the eight-mode patch, and the diffusivity-ratio inequality. The script's k_c function is the same relation. Pass-1 argmax remains in the file as the estimator the centroid replaced.  
  evidence: `morphoeval/lean/loops/Loop_3204_drosophila_silvestris_turing_wavelength_s0.lean` L27–56; `morphoeval/math_kg_loops/delphi10000-drosophila-silvestris-turing-wavelength-s0.yaml` L36–70
- **Reproducibility package: 92** — --self-test checks dataset hashes, Turing inequalities, Pearson r > 0.98 on both k and lambda, GIF frame count and resolution, and the equal-diffusion sign. Seeds are fixed. The pass-2 self-test passed.  
  evidence: `models/2026-delphi10000-drosophila-silvestris-turing-wavelength-s0.py` L639–678

Remediation:
- R5 (open). No independent second assessor has scored this audit.
- R6 (open). The wing-spot mask is a luminance and size heuristic (17 components), not a homologised spot map.
- R7 (open). Wavelength RMSE is 0.208 µm because the Fourier lattice is discrete even after the centroid.
- R8 (open). Nanopublications are unsigned: ~/.nanopub/profile.yml is absent. Do not treat the TriG as signed.

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
- note: numpy, scipy, Pillow, matplotlib. Signing profile absent.

Pass-2 re-score of the centroid sweep, wing-band mask, and clean terminal frame.
