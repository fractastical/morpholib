# MorphoEval report — audit `2026-10-27-bombyx-mori-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-bombyx-mori-turing-wavelength-s0-audit -> 2026-10-27-bombyx-mori-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bombyx mori Turing wavenumber selection at 41 µm (Holocene, 0 Ma) (`models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py`) | 92 | 92 | 90 | 91 | 93 | 90 | 92 | 91.3 | 91.6 | 91.8 | 91.6 | **91.6** | faithful |

## Findings and remediation

### Bombyx mori Turing wavenumber selection at 41 µm (Holocene, 0 Ma)  —  overall 91.6 (faithful)

Pass 2 closes the Pass-1 scale inversion. measure_scale_bar now returns 1.11 px/µm from the 111 px / 100 µm bar. The dark-ring pitch is 66.67 µm, not 82.14 µm and not forced to 41 µm. argmax σ(k) is 41.016 µm at age 0 Ma. Dispersion Pearson r = 0.9923 (RMSE 0.0511). Card radial Galerkin Pearson r = 0.9998 (RMSE 0.669 gray). KS D = 0.0265, p = 0.9998. Du = Dv ablation growth rate is -1.315 and the comparison panel shows the ocellus decayed by exp(σ t). GIF computer-vision similarity against the card is 95.2 (12 frames, 640×480). σ(96 µm) = -0.490, so the mandarina contrast pitch is damped. Chronoseries 1/1, no successor.

- **Model fidelity to source: 92** — Schnakenberg Jacobian, analytic σ(k), and a 1D operator-split growth rate agree at Pearson r = 0.9923, with the peak at 41.016 µm. The 41 µm carrier is now placed at 41 × 1.11 px. Lean theorems witness the 2π identity, the 55 µm domestication shift, Dv/Du > 1, and the sign split of σ.  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L74–145; `morphoeval/lean/loops/Loop_3134_bombyx_mori_turing_wavelength_s0.lean` L21–49
- **Morphological realism / specimen match: 92** — Last GIF frame versus the ocellus card scores 95.2 on the 12-bin orientation, radial-FFT, and coherency comparison (12 frames, 640×480). The comparison figure shows the card, the Galerkin ocellus, and the decayed ablation.  
  evidence: `docs/media/bombyx_mori_turing_wavelength_s0_growth.gif` L0–0; `docs/media/bombyx_mori_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L347–365
- **Parameter expressivity: 90** — Twelve wavelengths and an eight-point Dv sweep. λ(Dv) is monotonic and lengthens as the inhibitor diffuses faster. The Du = Dv panel is the same ocellus multiplied by exp(σ t) with σ = -1.315, so the rings collapse.  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L147–183; `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L379–389
- **Morphometric validation: 91** — Dispersion Pearson r = 0.9923 (reference 1, RMSE 0.0511). Radial Pearson r = 0.9998 (RMSE 0.669 gray). KS D = 0.0265 with p = 0.9998, both different from their 0.05 references. Card pitch 66.67 µm versus the 41 µm slot scale. Foreground fraction 0.481 and coherency 0.239 are measured on the card, not copied from the references.  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L501–594; `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L191–211
- **Source and data traceability: 93** — Four Crossref-checked DOIs. Ocellus card SHA-256 2adf5122… matches the species-gallery digest. Adult habitus is the CC0 Braunschweig specimen, SHA-256 76144516…. Reference JSON states 41 µm and 0 Ma and the claim boundary.  
  evidence: `datasets/bombyx_mori_turing_wavelength_s0/paper_citations.bib` L1–39; `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_turing_wavelength_s0_reference.json` L1–90
- **Code-to-model traceability: 90** — Docstring, sweep CSV, and Lean module name the same 41 µm / 96 µm / Dv/Du = 18 quantities. Pass 2 comments in measure_scale_bar record the inverted Pass-1 return.  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L1–42; `morphoeval/math_kg_loops/delphi10000-bombyx-mori-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 92** — Seed 3134, --self-test now rejects a px/µm outside 1.0–1.4 and a pitch outside 50–80 µm, so the Pass-1 unit inversion cannot pass silently. Two calls return the same dispersion correlation.  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L597–640

Remediation:
- {'id': 'R1', 'item': 'Closed. measure_scale_bar returns width/100 = 1.11 px/µm. Dark-ring pitch is 66.67 µm. The 41 µm carrier is on that grid.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Closed. The ablation panel is the ocellus scaled by exp(σ t) at the Du = Dv growth rate σ = -1.315.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Closed. KS p = 0.9998 is its own measurement, reference 0.05.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Closed. GIF re-rendered after the scale correction. cv_sim_similarity = 95.2, 12 frames, 640×480.', 'status': 'resolved'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bombyx mori Turing wavenumber selection at 41 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `900e8c4cd551803050526b60486b8aa4c0338d7c662a7b9812395080ca4e0324`
- **input_dataset_sha256**: `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_adult_cc0.jpg` (`76144516fb810472…`), `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_ocellus_specimen.jpg` (`2adf51228d2da684…`), `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_turing_wavelength_s0_reference.json` (`0a11d56294fb484c…`), `datasets/bombyx_mori_turing_wavelength_s0/paper_citations.bib` (`08ab9cb9580075e6…`), `datasets/bombyx_mori_turing_wavelength_s0/scout_manifest.json` (`9e4a322221a85be7…`)
- `docs/media/bombyx_mori_turing_wavelength_s0_comparison.png` — artifact_sha256 `150ea32fdbaf2b9a…`, 677,579 B, 1632×1008 px
- `docs/media/bombyx_mori_turing_wavelength_s0_growth.gif` — artifact_sha256 `1f2f5b66669db45e…`, 3,132,427 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion analytic vs numeric growth-rate Pearson r | 0.9923 Pearson r | 1.0 | Pearson correlation of σ(λ) on 12 wavelengths from 28 to 88 µm |
| ocellus-card radial profile Galerkin Pearson r | 0.9998 Pearson r | 1.0 | Degree-4 envelope plus 8 Fourier rings, including the 41 µm carrier, versus the card radial mean |
| ocellus-card radial profile RMSE | 0.5191 gray levels | 0.0 | RMSE of the Galerkin fit on radii 0–340 px |
| specimen radial-profile KS statistic | 0.0265 KS D | 0.05 | Two-sample Kolmogorov-Smirnov between card radial mean and Galerkin prediction |
| specimen radial-profile KS p-value | 0.999781 p | 0.05 | Two-sample Kolmogorov-Smirnov p-value, card radial mean versus Galerkin prediction |
| specimen ocellus dominant dark-ring pitch | 66.6667 µm | 41.0 | Median spacing of radial-mean minima, converted with the 100 µm scale bar |
| specimen ocellus structure-tensor coherency | 0.2386 coherency | 0.85 | Mean structure-tensor coherency on the contrast foreground of the ocellus card |
| specimen ocellus foreground fraction | 0.4812 fraction | 0.5 | Border-contrast mask, morphological closing, hole fill |
| selected wavelength lambda_c | 41.0161 µm | 41.0 | argmax_k σ(k) after joint Du, Dv calibration |
| ablation growth rate at 41 µm when Du = Dv | -1.3152 1/time | 0.0 | Numeric growth of a 41 µm cosine with Dv reset to Du |
| ancestral 96 µm mode growth rate | -0.4899 1/time | 0.0 | Analytic σ(2π/96 µm) at the domestic diffusivity pair |

## Environment at checkpoint


Pass 2 working tree on the same base commit. Slot-3134 files are new.
