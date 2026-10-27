# MorphoEval report — audit `2026-10-27-aptenodytes-forsteri-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `95244113b1348dfeec9466cf094db251e2807d75` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg wavelength at the 1.178 mm shell-thickness tier of Aptenodytes forsteri (`models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py`) | 86 | 72 | 84 | 80 | 88 | 84 | 86 | 85.3 | 75.0 | 86.1 | 81.4 | **81.4** | faithful |

## Findings and remediation

### Schnakenberg wavelength at the 1.178 mm shell-thickness tier of Aptenodytes forsteri  —  overall 81.4 (faithful)

Pass 1 overall 81.4. The dispersion law, ablation and specimen segmentation are implemented and the self-test reports r=0.9979, RMSE=0.0301 mm, ablation amplitude ratio 0.0039, photograph pore spacing 0.663 mm. Specimen-match stays at 72 because that pore spacing is not lambda_c and the first display scale hid the lattice.

- **Model fidelity to source: 86** — The script implements Schnakenberg kinetics, the Jacobian trace and determinant, sigma(k), and a diffusivity rescale that places the fastest mode on 10^(-2.929) m. Equal diffusion is predicted stable. The kinetics are the standard Schnakenberg form, not a feather-specific parameter table from Prum and Williamson.  
  evidence: `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L108–123; `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L69–86; `morphoeval/lean/loops/Loop_3116_aptenodytes_forsteri_turing_wavelength_s0.lean` L30–37
- **Morphological realism / specimen match: 72** — The growth GIF is 16 frames at 640x480 and ends on the MHNT egg (CV similarity 84.7 against that photograph). The generative panel is a labyrinth at about 1.18 mm, while the egg texture measures 0.66 mm. Those are different patterns. Pass 1 does not treat the terminal frame as proof that the PDE drew the pores.  
  evidence: `docs/media/aptenodytes_forsteri_turing_wavelength_s0_growth.gif` L0–0; `docs/media/aptenodytes_forsteri_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L252–299
- **Parameter expressivity: 84** — Seven commensurate scale factors move the dispersion wavelength, and the PDE peak follows them. Setting D_u = D_v drops the amplitude ratio to 0.0039. The figure must still show that sweep and the ablation in the caption before this score is raised.  
  evidence: `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L208–230; `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L477–503
- **Morphometric validation: 80** — Self-test on this run: dispersion lambda 1.1767 mm, PDE lambda 1.1685 mm, Pearson r 0.9979, RMSE 0.0301 mm, KS 0.143, pore FFT 0.663 mm against shell thickness 1.11 mm. n=7 for the KS sample is small. Values are not copied onto their references. Pass 1 holds the score at 80 until those numbers are the measurement record.  
  evidence: `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L506–599; `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L302–313
- **Source and data traceability: 88** — Turing, Prum and Williamson, Kondo and Miura, and Schnakenberg have DOIs. The egg dimensions are tied to the 2008 Polar Research URL; that page does not name authors, and none were invented. The specimen is Descouens CC BY-SA 4.0 with a SHA-256 pin. The species-gallery spiral was rejected.  
  evidence: `datasets/aptenodytes_forsteri_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/aptenodytes_forsteri_turing_wavelength_s0/aptenodytes_forsteri_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Dispersion, sweep, ablation, segmentation and rendering are separate functions. Six Lean theorems encode the shell-thickness window, the diffusion-ratio threshold, the integer margin, equal-diffusion failure, the successor egg length, and the Holocene single node. Lean was not run here.  
  evidence: `morphoeval/lean/loops/Loop_3116_aptenodytes_forsteri_turing_wavelength_s0.lean` L20–47; `morphoeval/math_kg_loops/delphi10000-aptenodytes-forsteri-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 86** — --self-test, seed 3116, and fixed output paths. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib). No second machine was used.  
  evidence: `models/2026-delphi10000-aptenodytes-forsteri-turing-wavelength-s0.py` L602–635; `datasets/aptenodytes_forsteri_turing_wavelength_s0/README.md` L1–40

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `specimen_comparison` → `specimen-match`: Min-max colour scale collapsed the activator to one hot pixel. Rescale from the median to the 99.5th percentile so the spot lattice is visible, and keep the 0.66 mm pore measurement beside lambda_c.
- **[OPEN · P=2.5]** `morphometric_validation` → `morphometric-validation`: Publish Pearson r, RMSE and KS for the commensurate-scale sweep, with law and PDE values not copied onto each other.
- **[OPEN · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Equal-diffusion ablation and the scale sweep must be visible in the comparison figure, not only in the self-test log.
- **[OPEN · P=2.5]** `specimen_comparison` → `morphometric-validation`: Segment the real MHNT egg and compare foreground, coherency and circularity to the simulated patch without setting value equal to reference.

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


Loop 3116 files are pinned at this commit. Lean 4 was not executed in this VM (no lean binary).
