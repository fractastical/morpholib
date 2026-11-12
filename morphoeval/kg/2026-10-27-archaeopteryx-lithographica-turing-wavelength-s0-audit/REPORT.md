# MorphoEval report — audit `2026-10-27-archaeopteryx-lithographica-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength on Archaeopteryx lithographica at 75 µm (Tithonian, 150.0 Ma) (`models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py`) | 88 | 72 | 86 | 87 | 90 | 84 | 88 | 87.3 | 77.6 | 87.6 | 83.5 | **83.5** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength on Archaeopteryx lithographica at 75 µm (Tithonian, 150.0 Ma)  —  overall 83.5 (faithful)

Periodic Schnakenberg system with a=0.112, b=0.836, Du=0.052, Dv=1.48, gamma=0.039, anisotropy 1.22, seed 3120. Analytical lambda_c = 11.991 cells, scaled so the physical wavelength is 7.478e-05 m (75 µm) at 150.0 Ma. Grid FFT wavelength is 12.00 cells (relative error 0.00072). Wing-transect Pearson r = 0.9803, nRMSE = 0.198, KS = 0.053. Equal-diffusion ablation std ratio = 0.00081. Gamma sweep is strictly decreasing. Predicted neighbor wavelengths at the same dx: Anchiornis 95.33 µm (160.8 Ma), Microraptor 57.05 µm (120 Ma). The Berlin-slab scan does not resolve 75 µm.

- **Model fidelity to source: 88** — dispersion() implements the Schnakenberg Jacobian and k_c^2 = (Dv fu + Du gv) / (2 Du Dv), and the grid FFT peak (12.00 cells) matches the analytical wavelength (11.991 cells). The numeric tuple is a reconstruction for this scale, not a table copied from a feather paper.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L63–93; `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L101–138
- **Morphological realism / specimen match: 72** — The comparison figure places the Berlin slab beside a stripe field and a wing overlay, and the GIF keeps the fossil recognizable (cv_sim_similarity 96). The overlay darkening is hard to see against slab contrast, so the pigment stripes are not yet a readable pattern on the vane.  
  evidence: `docs/media/archaeopteryx_lithographica_turing_wavelength_s0_comparison.png` L0–0; `docs/media/archaeopteryx_lithographica_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 86** — lambda_c(gamma) for gamma in {0.020, 0.039, 0.080, 0.150, 0.250} is strictly decreasing and is plotted. The chronoseries bars use the same dx and three gamma values.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L178–184; `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L448–468
- **Morphometric validation: 87** — Wing transect Pearson r = 0.9803, nRMSE = 0.198, KS = 0.053, inter-peak CV = 0.194. Equal-diffusion control collapses the field (std ratio 0.00081). Fundamental-only r = 0.807, so the headline r is not a one-sine fit. The transect is the scan, not a 75 µm measurement.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L232–270; `datasets/archaeopteryx_lithographica_turing_wavelength_s0/fit_metrics.json` L0–0
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Prum and Williamson, Carney, Manning, Foth, and both Li papers. The Berlin slab is public domain and SHA-256 pinned.  
  evidence: `datasets/archaeopteryx_lithographica_turing_wavelength_s0/paper_citations.bib` L1–96; `datasets/archaeopteryx_lithographica_turing_wavelength_s0/archaeopteryx_lithographica_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — The module docstring and dispersion() name the PDE, the steady state, and k_c. Lean Loop_3120 records the wavelength order and the diffusivity ratio as Nat facts checked by the self-test.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L1–18; `morphoeval/lean/loops/Loop_3120_archaeopteryx_lithographica_turing_wavelength_s0.lean` L20–56
- **Reproducibility package: 88** — --self-test checks Turing conditions, r > 0.98, ablation, monotonic sweep, Lean integers, GIF frames and size, and cv_sim_similarity >= 58. Seed 3120 and the specimen hash are pinned. Pass 1 is a single assessor.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L594–613

Remediation:
- {'id': 'R1', 'item': 'The wing overlay is too faint against slab contrast. Raise stripe contrast on the segmented vane and keep cv_sim_similarity >= 70.'}
- {'id': 'R2', 'item': 'Print the fundamental-only Pearson r (0.807) on the comparison figure next to the 3-harmonic r so the flexible fit is visible.'}
- {'id': 'R3', 'item': 'Annotate the segmented wing-mask fraction on the figure. Pass 1 leaves it only in fit_metrics.json.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength on Archaeopteryx lithographica at 75 µm (Tithonian, 150.0 Ma)
- **spec_fingerprint** (semantic model + inputs): `6913098e47054812a2f1cf896b7e10fbf923735b2080ce927a3087eecb641c84`
- **input_dataset_sha256**: `datasets/archaeopteryx_lithographica_turing_wavelength_s0/archaeopteryx_lithographica_berlin_slab.jpg` (`2e307a2774e08571…`), `datasets/archaeopteryx_lithographica_turing_wavelength_s0/archaeopteryx_lithographica_turing_wavelength_s0_reference.json` (`32deca0baa44c4ef…`)
- `datasets/archaeopteryx_lithographica_turing_wavelength_s0/fit_metrics.json` — artifact_sha256 `1648889b00d2326f…`, 1,601 B, 
- `docs/media/archaeopteryx_lithographica_turing_wavelength_s0_comparison.png` — artifact_sha256 `4c1546e3a950825e…`, 758,256 B, 1496×924 px
- `docs/media/archaeopteryx_lithographica_turing_wavelength_s0_growth.gif` — artifact_sha256 `a735340e24845d2e…`, 10,303,307 B, 960×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Archaeopteryx lithographica specimen wing-vane optical-density Pearson r | 0.9803 Pearson r | 0.95 | gaussian sigma=3 column-mean of Berlin-slab ROI rows 450:850, cols 750:1200; quartic + 3 harmonics of the dominant FFT frequency *(fundamental-only r=0.8066; N=450; scan does not resolve 75 µm)* |
| specimen wing transect normalized RMSE | 0.1977 RMSE / sigma | 0.35 | sqrt(mean((OD - prediction)^2)) / std(OD) on the same wing transect *(KS two-sample statistic=0.0533)* |
| specimen versus fitted transect two-sample KS statistic | 0.0533 KS distance | 0.25 | scipy.stats.ks_2samp on the wing optical-density profile and the quartic-plus-harmonics prediction |
| Turing pigment stripe wavelength lambda_c at the Archaeopteryx cellular scale | 7.478e-05 m | 9e-07 | lambda_cells(gamma=0.039) * dx with dx chosen so this wavelength equals 7.478e-05 m *(measured/analytical grid relative error=0.00072)* |
| equal-diffusion ablation spatial-std ratio (Du=Dv control) | 0.000815 std(u_ablation)/std(u_turing) | 1.0 | Du=Dv=0.45, no critical-mode seed, 1800 steps, versus the Turing run *(std_turing=0.4968, std_ablation=0.000405)* |
| Anchiornis huxleyi predecessor cellular stripe wavelength | 9.533e-05 m | 7.478e-05 | same dx and kinetics, gamma=0.024 (slower symmetric-vane melanocyte kinetics) |
| Microraptor gui successor cellular stripe wavelength | 5.705e-05 m | 7.478e-05 | same dx and kinetics, gamma=0.067 (faster kinetics, iridescent melanosome stacks) |
| specimen slab structure-tensor coherency | 0.4665 coherency | 0.3 | border-contrast mask and Sobel structure tensor on the Berlin slab resized to 160x120 *(foreground_fraction=0.6664; wing_mask_fraction=0.0768; interpeak CV=0.1943)* |
| gamma sweep wavelength ratio lambda(0.020)/lambda(0.250) | 3.5355 ratio | 1.0 | analytical lambda_c across gamma in {0.020, 0.039, 0.080, 0.150, 0.250} *(strictly decreasing in gamma)* |

## Environment at checkpoint


Pass 1 working tree on 0c76d15d. Slot files are new and are not in that commit.
