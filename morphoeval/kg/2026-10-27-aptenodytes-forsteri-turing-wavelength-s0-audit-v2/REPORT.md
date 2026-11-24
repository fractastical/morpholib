# MorphoEval report — audit `2026-10-27-aptenodytes-forsteri-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `95244113b1348dfeec9466cf094db251e2807d75` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-aptenodytes-forsteri-turing-wavelength-s0-audit -> 2026-10-27-aptenodytes-forsteri-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg wavelength at the 1.178 mm shell-thickness tier of Aptenodytes forsteri (`models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py`) | 90 | 84 | 88 | 91 | 90 | 88 | 90 | 89.3 | 86.6 | 89.5 | 88.3 | **88.3** | faithful |

## Findings and remediation

### Schnakenberg wavelength at the 1.178 mm shell-thickness tier of Aptenodytes forsteri  —  overall 88.3 (faithful)

Pass 2 overall 88.3. Remediation items from pass 1 are closed in the script: robust display, scale sweep, equal-diffusion ablation, specimen segmentation, and goodness-of-fit (r=0.9979, RMSE=0.0301 mm, KS=0.143). The GIF terminal frame matches the MHNT egg (CV 84.7, 16 frames, 640 px). The generative wavelength remains 1.178 mm and is not forced onto the 0.66 mm pore measurement. No second assessor. Nanopub signing profile was absent.

- **Model fidelity to source: 90** — Schnakenberg kinetics, Jacobian, sigma(k), and the diffusivity rescale that puts the fastest mode on 10^(-2.929) m are in the script and checked by --self-test (trace negative, determinant positive, sigma(0) negative, equal-diffusion sigma_max negative, lambda within 1.5 percent of the tier). The system is not a line-by-line copy of a feather-follicle parameter table.  
  evidence: `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L94–123; `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L477–498; `morphoeval/lean/loops/Loop_3116_aptenodytes_forsteri_turing_wavelength_s0.lean` L20–37
- **Morphological realism / specimen match: 84** — Direct CV comparison of the 640x480, 16-frame GIF against the MHNT egg is 84.7. The last frame is that photograph. The middle panel, after median-to-p99.5 scaling, shows a spot-labyrinth on a shell patch, and the caption states the photograph pore spacing (0.66 mm) next to the model wavelength. The score stays below 90 because the generative field is not a pore-by-pore copy of the shell.  
  evidence: `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L362–368; `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L345–359; `docs/media/aptenodytes_forsteri_turing_wavelength_s0_comparison.png` L0–0; `docs/media/aptenodytes_forsteri_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Scale factors 0.8, 8/9, 1, 8/7, 4/3, 1.6 and 2 move lambda along the 1:1 line in the comparison figure (r=0.9979). Equal diffusion is the negative control (amplitude ratio 0.0039 versus a growing Turing run).  
  evidence: `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L208–230; `docs/media/aptenodytes_forsteri_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 91** — Dispersion lambda 1.1767 mm versus PDE 1.1685 mm. Pearson r 0.9979, RMSE 0.0301 mm, KS 0.143. Photograph pore wavelength 0.663 mm against published shell thickness 1.11 mm. Simulated patch foreground 0.450, coherency 0.388, circularity 0.030 against specimen 0.535, 0.384, 0.732. No metric copies its reference. The KS sample has only seven scales.  
  evidence: `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L506–599; `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L252–313
- **Source and data traceability: 90** — DOIs for Turing, Prum and Williamson, Kondo and Miura, Schnakenberg, and Tyler. Egg size from the 2008 Polar Research URL, with authors left blank because the page does not list them. Specimen SHA-256 70b89e84383c36f84caa716b75b158e562e4d78d22c697e0a96668a6bfef4245, CC BY-SA 4.0, Didier Descouens.  
  evidence: `datasets/aptenodytes_forsteri_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/aptenodytes_forsteri_turing_wavelength_s0/aptenodytes_forsteri_turing_wavelength_s0_reference.json` L40–72; `datasets/aptenodytes_forsteri_turing_wavelength_s0/README.md` L1–36
- **Code-to-model traceability: 88** — Each modelling step is a named function cited from the math-kg loop file. The Lean module has six theorems and no sorry. The VM had no Lean binary, so the proofs were checked by reading the Nat comparisons rather than by compilation.  
  evidence: `morphoeval/lean/loops/Loop_3116_aptenodytes_forsteri_turing_wavelength_s0.lean` L20–47; `morphoeval/math_kg_loops/delphi10000-aptenodytes-forsteri-turing-wavelength-s0.yaml` L12–53
- **Reproducibility package: 90** — Deterministic seed, --self-test, and documented invocation. Pass 2 re-ran the script after the display fix. Signing was not possible: ~/.nanopub/profile.yml is absent, and that is recorded rather than simulated.  
  evidence: `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L602–634; `datasets/aptenodytes_forsteri_turing_wavelength_s0/README.md` L28–40

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=2.5]** `specimen_comparison` → `specimen-match`: Median-to-p99.5 scaling shows the spot lattice, and the caption states pore FFT 0.66 mm beside lambda_c.
- **[RESOLVED · P=2.5]** `morphometric_validation` → `morphometric-validation`: Pearson r 0.9979, RMSE 0.0301 mm, KS 0.143 on seven commensurate scales. Law and PDE values differ.
- **[RESOLVED · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Equal-diffusion amplitude ratio 0.0039 is in the figure caption and in the sweep.
- **[RESOLVED · P=2.5]** `specimen_comparison` → `morphometric-validation`: Specimen foreground 0.535, coherency 0.384, circularity 0.732 versus simulated patch 0.450, 0.388, 0.030.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg wavelength at the 1.178 mm shell-thickness tier of Aptenodytes forsteri
- **spec_fingerprint** (semantic model + inputs): `8258983b5a6bcf70e050b83d480003ff7d17bbb90cda5744e49bb2bec31fced1`
- **input_dataset_sha256**: `datasets/aptenodytes_forsteri_turing_wavelength_s0/aptenodytes_forsteri_specimen.jpg` (`70b89e84383c36f8…`)
- `docs/media/aptenodytes_forsteri_turing_wavelength_s0_comparison.png` — artifact_sha256 `4a0ff4289fae61fc…`, 259,363 B, 1464×648 px
- `docs/media/aptenodytes_forsteri_turing_wavelength_s0_growth.gif` — artifact_sha256 `43100eded47e3a77…`, 2,034,727 B, 640×480 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen_pore_wavelength_mm | 0.6628 mm | 1.11 | radial FFT of high-pass shell texture; scale bar from egg length 121.3 mm; CV of wavelength vs shell thickness *(Photograph pore spacing is not set equal to the model wavelength.)* |
| specimen_shell_coherency | 0.3878 1 | 0.3841 | structure-tensor coherency on the simulated patch versus the segmented specimen |
| specimen_segment_foreground_fraction | 0.45 1 | 0.5347 | Otsu-style percentile segmentation of the activator field versus specimen foreground |
| specimen_shell_circularity | 0.0299 1 | 0.7323 | 4πA/P² on the simulated spot mask versus the egg silhouette |
| turing_wavelength_pde_mm | 1.1685 mm | 1.1767 | Fourier peak of the s=1 PDE field; RMSE against the dispersion law |
| wavelength_sweep_pearson_r | 0.9979 1 | 0.98 | Pearson r between dispersion λ(s) and PDE λ(s); RMSE reported alongside |
| wavelength_sweep_rmse_mm | 0.0301 mm | 0.15 | RMSE of PDE wavelength against the dispersion prediction across commensurate scales |
| wavelength_sweep_ks_statistic | 0.1429 1 | 0.05 | two-sample KS statistic between the law series and the PDE series |
| equal_diffusion_ablation_amplitude_ratio | 0.0039 1 | 1.0 | negative control: D_u = D_v, amplitude ratio versus the Turing run (ratio baseline 1) *(control vs Turing δ=42)* |

## Environment at checkpoint


Same tree as pass 1 plus the robust display. Lean 4 was not executed (no lean binary).
