# MorphoEval report — audit `2026-10-27-drosophila-bipectinata-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-bipectinata-turing-wavelength-s0-audit -> 2026-10-27-drosophila-bipectinata-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila bipectinata Turing wavelength at 1.92 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py`) | 86 | 80 | 90 | 88 | 86 | 84 | 88 | 87.3 | 83.0 | 86.3 | 85.3 | **85.3** | faithful |

## Findings and remediation

### Drosophila bipectinata Turing wavelength at 1.92 µm (Holocene, 0 Ma)  —  overall 85.3 (faithful)

Pass 2 closes the four Pass-1 remediations. The seven-point D_u sweep is monotonic (self-test). Equal diffusion with an 0.8% white-noise seed amplifies by 0.0388 while the Turing run amplifies by 89.6. Specimen foreground fraction is 0.3233 at the verifier threshold and differs from the percentile-85 mask. Even-mode Pearson r is 0.99478 against a reference of 1.0; odd-mode RMSE is 0.01287 against 0; KS is 0.125 against 0. The last GIF frame matches the pinned specimen at CV similarity 93.0. Weighted overall is 85.3.

- **Model fidelity to source: 86** — The code implements Schnakenberg kinetics, the Turing wavenumber, and a discrete dispersion relation whose eigenmode growth correlates with the formula at r = 0.99478. gamma is derived so the continuous focal wavelength is 1.92 µm. Fidelity is not higher because that wavelength is the assigned rung rather than a constant published for this species.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L68–102; `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L151–219; `morphoeval/lean/loops/Loop_3178_drosophila_bipectinata_turing_wavelength_s0.lean` L18–42
- **Morphological realism / specimen match: 80** — The 13-frame GIF grows a melanin mask from the activator on the pinned gallery photograph. Direct comparison of the last frame to that photograph gives CV similarity 93.0 (12-bin orientation, 24-bin radial FFT, structure tensor). The score stays at 80 because the JPEG is a gallery fallback, not a micrograph at 1.92 µm, and the overlay inherits the photograph's low-frequency layout.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L345–368; `docs/media/drosophila_bipectinata_turing_wavelength_s0_growth.gif` L0–0; `datasets/drosophila_bipectinata_turing_wavelength_s0/drosophila_bipectinata_specimen.jpg` L0–0
- **Parameter expressivity: 90** — du_sweep samples seven diffusivities from phi 0.49 to 2.25. self_test asserts that lambda is strictly increasing. The chronoseries places biarmipes at 2.40 µm, the focal species at 1.92 µm, and buzzatii at 1.536 µm, which is the sqrt(D_u) control the law claims. The comparison figure states the sweep endpoints.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L125–134; `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L580–595; `docs/media/drosophila_bipectinata_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Even-mode Pearson r = 0.99478 (reference 1.0). Held-out odd-mode RMSE = 0.01287 s^-1 (reference 0). KS distance = 0.125 (reference 0). Nonlinear wavelength 2.56 µm is not set equal to the linear 1.92 µm reference. Ablation amplification 0.0388 versus Turing amplification 89.6. Specimen foreground at the verifier threshold is 0.3233 versus the percentile-85 mask, and coherency is 0.755 versus an isotropic baseline of 0.5.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L461–575; `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L293–338
- **Source and data traceability: 86** — Citations resolve on OpenAlex to Turing 1952, Schnakenberg 1979, Kopp & True 2002, Mishra & Singh 2006, Gompel et al. 2005, and Kondo & Miura 2010. The scout manifest records the rejected catalog DOI, the empty GBIF media list (usageKey 5074478), and the gallery fallback. The specimen hash is checked at load.  
  evidence: `datasets/drosophila_bipectinata_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/drosophila_bipectinata_turing_wavelength_s0/paper_citations.bib` L1–70; `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L59–66
- **Code-to-model traceability: 84** — Each measurement method string names the equation or the segmentation rule it implements. The Lean theorems encode the integer diffusivity identities used by chronoseries(). The nonlinear wavelength is traced to the radial mode of the saturated field rather than hidden inside the linear solve.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L221–291; `morphoeval/math_kg_loops/delphi10000-drosophila-bipectinata-turing-wavelength-s0.yaml` L16–48
- **Reproducibility package: 88** — python3 models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py --self-test regenerates the PNG and GIF from seed 3178 and checks wavelength order, Pearson r > 0.98, ablation, sweep monotonicity, GIF frames and resolution, Lean theorem count, and non-circular measurements. The run reported here exited SELF_TEST_OK.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L580–626; `datasets/drosophila_bipectinata_turing_wavelength_s0/README.md` L1–40

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila bipectinata Turing wavelength at 1.92 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `d5f22a094ff81c9bfb8cec455feea78ef577fb724b6dfb1571cde0d7f6d97c6c`
- `datasets/drosophila_bipectinata_turing_wavelength_s0/drosophila_bipectinata_specimen.jpg` — artifact_sha256 `bba52df453cd96fd…`, 96,717 B, 800×600 px
- `docs/media/drosophila_bipectinata_turing_wavelength_s0_comparison.png` — artifact_sha256 `97b0621eba033a3b…`, 738,331 B, 1248×984 px
- `docs/media/drosophila_bipectinata_turing_wavelength_s0_growth.gif` — artifact_sha256 `1c0014b74396d41a…`, 3,266,791 B, 640×480 px, 13 frames

| measurement | value | reference | method |
|---|---:|---|---|
| focal linear Turing wavelength against the 86 µm sibling rung | 1.92 µm | 86.0 | k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) with gamma solved at D_u = 0.0504 µm^2/s, delta = 31.5 *(the solved focal wavelength is the 1.92 µm rung; 86 µm is the other slot)* |
| nonlinear selected pigment wavelength on the 15.36 µm patch | 2.56 µm | 1.92 | dominant radial Fourier mode of the saturated Schnakenberg activator, L/m* *(saturation shifts the peak off the linear critical mode; value is not forced equal to 1.92)* |
| even-mode dispersion Pearson r of numerical eigenmode growth vs discrete sigma(k) | 0.99478 Pearson r | 1.0 | pure unstable-eigenvector initial data, log-amplitude slope versus the discrete-Laplacian eigenvalue *(all-mode r = 0.99772; odd-mode held-out RMSE = 0.01287)* |
| held-out odd-mode dispersion RMSE | 0.01287 1/s | 0.0 | RMSE of sigma on odd Fourier modes, which are not the Pearson reporting set *(KS distance between analytical and numerical sigma lists is recorded separately)* |
| KS distance between analytical and numerical growth-rate lists | 0.125 KS statistic | 0.0 | scipy.stats.ks_2samp on the 16-mode sigma vectors *(a zero KS would mean the lists are the same sample; the solver residual keeps it positive)* |
| predecessor Drosophila biarmipes linear wavelength at 0 Ma | 2.4 µm | 1.92 | same gamma and delta, D_u scaled by 25/16 (wing-spot clade, less tethered activator) *(age is 0 Ma for both; the change at this rung is diffusivity, not geological time)* |
| successor Drosophila buzzatii linear wavelength at 0 Ma | 1.536 µm | 1.92 | same gamma and delta, D_u scaled by 16/25 *(reconstructed chronoseries contrast, not a measured buzzatii granule spacing)* |
| equal-diffusion ablation amplification of activator std | 0.0388 std_final/std_0 | 89.6033 | Schnakenberg integration at delta = D_v/D_u = 1 for 6 s *(control must stay near or below 1 while the Turing run amplifies)* |
| Du sweep wavelength at phi = 2.25 versus phi = 0.49 | 2.88 µm | 1.344 | seven-point sweep of D_u at fixed gamma and delta; reported value is lambda(phi=2.25) *(lambda increases monotonically with sqrt(D_u))* |
| specimen image foreground fraction at contrast percentile 45 | 0.3233 area fraction | 0.153 | border-median contrast mask on the SHA-pinned gallery JPEG, percentile 45, closing and hole fill *(circularity p45 = 0.0462, coherency = 0.755)* |
| specimen structure-tensor coherency | 0.755 coherency | 0.5 | mean (lambda1-lambda2)/(lambda1+lambda2) of the Sobel structure tensor, sigma = 2.5 *(the gallery frame is anisotropic; this is not a self-copy of the reference)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scipy: 1.18.1

Pass 2 re-ran --self-test after the four Pass-1 remediations were in the script: seven-point D_u sweep, white-noise equal-diffusion ablation, two-threshold specimen segmentation, and even/odd dispersion GoF. SELF_TEST_OK. Even-mode Pearson r = 0.99478. Odd-mode RMSE = 0.01287. KS = 0.125. Turing amplification = 89.60 versus ablation amplification = 0.0388. Nonlinear wavelength = 2.56 µm against linear 1.92 µm. GIF 640×480, 13 frames. CV similarity of the last frame to the pinned specimen = 93.0.
