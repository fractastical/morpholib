# MorphoEval report — audit `2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `bd8176806dee28731b94a3866aa6609c1d9d4ba2` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit -> 2026-10-27-clytia-hemisphaerica-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Clytia hemisphaerica Schnakenberg wavelength at 8.64 µm (Pass 2) (`models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py`) | 90 | 82 | 88 | 90 | 92 | 86 | 88 | 89.3 | 85.0 | 88.8 | 87.4 | **87.4** | faithful |

## Findings and remediation

### Clytia hemisphaerica Schnakenberg wavelength at 8.64 µm (Pass 2)  —  overall 87.4 (faithful)

Pass 2 closes the Pass-1 probe defect. With the analytic eigenvector and a second-half fit, PDE growth rates match Jacobian σ(k) at Pearson r = 0.9993 and RMSE = 0.014 s⁻¹. Selected wavelength is 8.646 µm (assigned 8.64 µm, log10 = -5.063) at age 0 Ma. Equal-diffusion std(u) = 0.00132 versus wild-type 0.0193. Dv/Du from 10 to 36 increases λ_*. Segmented planula aspect is 3.23 against a circular null of 1, circularity 0.371, interior KS 0.191 against 0. CV similarity of the 12-frame 640 px GIF against the Momose micrograph is 82.1. A feathered outline was tried and rejected: it dropped CV to 60.5.

- **Model fidelity to source: 90** — Schnakenberg Jacobian, onset k_c, and k_* = argmax Re σ(k) are implemented directly. γ is solved so k_* equals the assigned wavelength. The Pass-2 probe sits on the unstable eigenvector.  
  evidence: `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L101–162; `morphoeval/lean/loops/Loop_3150_clytia_hemisphaerica_turing_wavelength_s0.lean` L12–44
- **Morphological realism / specimen match: 82** — Computer-vision similarity against the real planula is 82.1 (12-bin orientation, 24-bin radial FFT, structure-tensor coherency). The body, background, and endoderm shading read as the same larva. The contour is still harder than the DIC halo; feathering that edge cut CV to 60.5 and was not kept.  
  evidence: `docs/media/clytia_hemisphaerica_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L378–414
- **Parameter expressivity: 88** — At fixed γ, raising Dv/Du from 10 to 36 increases λ_* monotonically (ratio-sweep Pearson 0.99-class, flag monotonic_increasing). The equal-diffusion end of the axis is the ablation, not another patterned state.  
  evidence: `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L278–294; `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L242–276
- **Morphometric validation: 90** — Pass-2 Pearson r = 0.999301, RMSE = 0.01406 s⁻¹, both with references that are not copies of the value. Planula aspect 3.2304 versus a circular null of 1, circularity 0.3711 versus 1, interior KS 0.1908 versus 0. Ablation std 0.00132 versus wild-type 0.0193.  
  evidence: `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L546–660; `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L297–337
- **Source and data traceability: 92** — Citations carry resolvable DOIs. Specimen SHA-256 95bffe69b1854823dd8042a28e0f79657379a0d5d0c6cf1f672202b230648ba2, creator Tsuyoshi Momose, CC BY-SA 4.0, Commons file dated 2021-04-22.  
  evidence: `datasets/clytia_hemisphaerica_turing_wavelength_s0/paper_citations.bib` L1–89; `datasets/clytia_hemisphaerica_turing_wavelength_s0/clytia_hemisphaerica_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — Each reported law in the math-KG record points at a Lean theorem and a Python locus: margin, k_*, mode count, sqrt(D) scaling, and the Holocene age. The 250 µm planula length is labelled as a reconstruction.  
  evidence: `morphoeval/math_kg_loops/delphi10000-clytia-hemisphaerica-turing-wavelength-s0.yaml` L1–65; `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L1–47
- **Reproducibility package: 88** — --self-test checks λ* within 0.05 µm, Pass-2 r > 0.98, Pass-1 r still below 0.96, ablation decay, monotonic sweep, and GIF geometry when the file exists. Seed is 3150. No nanopub profile is present, so nothing is signed.  
  evidence: `models/2026-delphi10000-clytia-hemisphaerica-turing-wavelength-s0.py` L505–543

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Clytia hemisphaerica Schnakenberg wavelength at 8.64 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `96a665cf0d7c09f92284588fc71976c4d562666ee4ba1270067ce2e4f5d4c878`
- **input_dataset_sha256**: `datasets/clytia_hemisphaerica_turing_wavelength_s0/clytia_hemisphaerica_specimen.jpg` (`95bffe69b1854823…`), `datasets/clytia_hemisphaerica_turing_wavelength_s0/clytia_hemisphaerica_turing_wavelength_s0_reference.json` (`8e95bb9584ac6852…`), `datasets/clytia_hemisphaerica_turing_wavelength_s0/paper_citations.bib` (`045903087c0ddff1…`), `datasets/clytia_hemisphaerica_turing_wavelength_s0/scout_manifest.json` (`014d315226f357c5…`)
- `docs/media/clytia_hemisphaerica_turing_wavelength_s0_growth.gif` — artifact_sha256 `0f928ee2c59844b3…`, 661,371 B, 640×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected Turing wavelength lambda_* | 8.6464 µm | 8.64 | argmax_k Re σ(k) of the Schnakenberg Jacobian at Du=1 µm²/s, Dv/Du=20 *(γ is pinned so k_* realises the assigned 8.64 µm detail length)* |
| dispersion-probe Pearson r (eigenvector, pass 2) | 0.999301 1 | 0.98 | PDE Fourier-mode growth vs Jacobian σ at the discrete Laplacian wavenumber *(RMSE 0.0141 s⁻¹; gate is r > 0.98)* |
| pass-1 transient probe Pearson r | 0.506133 1 | 0.98 | activator-only initial data, growth fit from t=0 (biased eigenspace) *(below the gate; this is why pass 2 changes the probe)* |
| dispersion-probe RMSE | 0.01406 s^-1 | 0.0 | root mean square of σ_numeric − σ_analytic over the probed band *(zero would be an exact eigenmode of the discrete Laplacian)* |
| equal-diffusion ablation std(u) | 0.0013248623997750928 1 | 0.0192708407717446 | Dv=Du integrated on the same grid and horizon as wild type *(reference_value is the wild-type std; ablation must sit far below it)* |
| specimen contour aspect ratio | 3.2304 1 | 1.0 | bounding-box height/width of the segmented day-2 planula mask *(null reference is a circular body; the planula is elongate)* |
| specimen contour circularity | 0.3711 1 | 1.0 | 4πA/P² on the segmented planula outline *(a circle would score 1; the larva does not)* |
| planula versus medusa-bell mode count at 8.64 µm | 28 stripes | 231 | floor(L/λ_*) with reconstructed L_planula=250 µm *(reference is floor(2000 µm / λ_*) on a medusa bell; detail scale holds fewer repeats. Age delta vs the 1/1 Holocene series is 0 Ma)* |
| simulation interior gray std | 17.3421 gray | 17.1185 | std of the GIF last frame luminance, including background *(reference is the first frame; growth must move the luminance distribution)* |
| specimen interior luminance KS statistic | 0.1908 1 | 0.0 | two-sample KS between specimen and simulation interior gray values *(0 would be a pixel copy; a positive KS is a real distributional distance)* |

## Environment at checkpoint


Pass 2 uses the eigenvector probe, the ablation, the ratio sweep, and the planula segmentation already in this commit.
