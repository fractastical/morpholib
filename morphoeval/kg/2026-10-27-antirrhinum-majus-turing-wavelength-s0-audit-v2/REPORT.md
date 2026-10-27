# MorphoEval report — audit `2026-10-27-antirrhinum-majus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-antirrhinum-majus-turing-wavelength-s0-audit -> 2026-10-27-antirrhinum-majus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Antirrhinum majus Turing ocellus wavelength at 259 µm (Holocene, 0 Ma) (`models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py`) | 90 | 82 | 93 | 90 | 90 | 88 | 90 | 91.0 | 85.0 | 89.5 | 88.1 | **88.1** | faithful |

## Findings and remediation

### Antirrhinum majus Turing ocellus wavelength at 259 µm (Holocene, 0 Ma)  —  overall 88.1 (faithful)

Pass 2 of the 259.13 µm ocellus (Holocene, 0 Ma). The radial power spectrum is now binned in physical wavenumber. At unit diffusivity the readback is 255.1 µm against λ_c = 259.13 µm (1.6% relative, was about 7% on integer pixel bins). The diffusivity sweep Pearson r is 0.99973 and RMSE(k) is 0.000263 µm⁻¹. Equal-diffusion σ_max stays −0.25 h⁻¹. Gallery pitch remains 77.8 µm (KS = 1 against the simulated gaps). The corner in σ(k) is marked as the unstable-band edge. GIF 800×600, 12 frames, CV similarity 83.6. Overall 88.1. No second assessor (R2).

- **Model fidelity to source: 90** — Same Jacobian, dispersion, cubic split-step, and ablation as Pass 1, with k_c defined as argmax σ(k). The Pass 1 readback bias is closed: physical-k bins recover 255.1 µm against 259.13 µm. Lean still checks the integer margin and the equal-diffusion closure, with zero sorry. The rates remain a reconstruction for this slot, not a snapdragon fit.  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L139–187; `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L197–229; `morphoeval/lean/loops/Loop_3114_antirrhinum_majus_turing_wavelength_s0.lean` L32–42
- **Morphological realism / specimen match: 82** — Last frame versus the gallery field scores 83.6 on the shared orientation / radial-FFT / coherency comparator. Both are soft green radial ocelli in an 800×600 frame. The physical pitches differ (77.8 µm on the bar, 259.13 µm in the model) and the audit does not treat that difference as a match. The CC BY-SA flower is the organ-scale voucher, not the fitness target.  
  evidence: `docs/media/antirrhinum_majus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/antirrhinum_majus_turing_wavelength_s0_growth.gif` L0–0; `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_metrics.json` L0–0
- **Parameter expressivity: 93** — Joint scale of D_u and D_v from 0.35 to 4 moves λ from about 153 µm to about 518 µm. With physical-k bins, Pearson r is 0.99973 and RMSE(k) is 0.000263 µm⁻¹. The ablation D_u = D_v = 640 stays at σ_max = −0.25 and is not on that curve.  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L261–285; `docs/media/antirrhinum_majus_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 90** — Gallery bar 111 px = 100 µm, ring pitch 77.8 µm, foreground fraction 0.684, coherency 0.236. None of these equals its reference (the references are λ_c, 0.5, and 0.25). KS of the gap samples is 1.0. Pass 2 cuts the unit-scale wavelength bias from about 7% to 1.6% (255.1 vs 259.13 µm) and lowers RMSE(k) from 0.00114 to 0.000263.  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L429–494; `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L519–541; `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L728–755
- **Source and data traceability: 90** — Unchanged from Pass 1. Five DOIs checked on Crossref after OpenAlex returned 429. SHA-256 pins on the gallery field and the CC BY-SA Sète flower. The gallery asset is not a DOI-backed micrograph.  
  evidence: `datasets/antirrhinum_majus_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_reference.json` L48–70
- **Code-to-model traceability: 88** — Docstring, sigma_of_k2, k_c_and_sigma, and the physical-k binning docstring name the same law. The band-edge annotation in the figure points at the sign change of σ(k). Lean integers match D_u and D_v.  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L43–54; `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L570–586
- **Reproducibility package: 90** — --self-test passes after the binning change (r = 0.99973, RMSE 0.000263, ablation negative, bar 111 px, λ_c within 1% of 259.2). Seed 3114. Re-render writes the same GIF size. No per-script lockfile.  
  evidence: `models/2026-delphi10000-antirrhinum-majus-turing-wavelength-s0.py` L779–817; `datasets/antirrhinum_majus_turing_wavelength_s0/README.md` L28–34

Remediation:
- {'id': 'R1', 'item': 'Closed in Pass 2. Physical-k bins. Unit-scale readback 255.1 µm vs 259.13 µm (1.6%).', 'status': 'closed'}
- {'id': 'R2', 'item': 'Open. Single assessor; no second provider has scored this script.', 'status': 'open'}
- {'id': 'R3', 'item': 'Closed in Pass 2. The dispersion panel marks the σ=0 crossing as the band edge.', 'status': 'closed'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Antirrhinum majus Turing ocellus wavelength at 259 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `9aa96d9174bbc264f2f19508483526998fd8a90a8bb5b00ee7c2a039134c8ca4`
- **input_dataset_sha256**: `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_sete_cc-by-sa.jpg` (`4d2cd99ca49de264…`), `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_reference.json` (`0918abca4751ad6b…`), `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_specimen.jpg` (`f3e9bf78a4453bc1…`)
- `datasets/antirrhinum_majus_turing_wavelength_s0/antirrhinum_majus_turing_wavelength_s0_metrics.json` — artifact_sha256 `3785b1019911d440…`, 4,674 B, 
- `docs/media/antirrhinum_majus_turing_wavelength_s0_comparison.png` — artifact_sha256 `1e0af27325255c7e…`, 539,359 B, 1664×1092 px
- `docs/media/antirrhinum_majus_turing_wavelength_s0_growth.gif` — artifact_sha256 `7ec4b0240db6d0c7…`, 3,276,935 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical critical wavelength lambda_c | 259.1268 µm | 259.2 | 2π / argmax_k σ(k) with D_u=640, D_v=4341 µm²/h *(D_v=4341 places argmax σ(k) 0.02 µm under the assigned 259.2 µm)* |
| diffusivity-scale sweep Pearson r of measured vs analytical k_c | 0.99973 correlation | 0.98 | 8-point joint D_u, D_v scale sweep, Fourier-space linear evolution, radial PSD peak *(RMSE(k) = 0.000263 µm⁻¹)* |
| diffusivity-scale sweep RMSE of wavenumber | 0.000263 µm⁻¹ | 0.005 | sqrt mean square error between analytical k_c and Fourier peak |
| equal-diffusion ablation sigma_max | -0.25 h⁻¹ | 0.0 | same linear solver with D_u = D_v = 640 *(spatial std ratio after growth = 0.0041)* |
| specimen ocellus ring spacing | 77.7778 µm | 259.1268 | radial gray peaks × (100 µm / detected scale-bar pixels) on the gallery field *(CV = 0.047, bar = 111 px, coherency = 0.236)* |
| specimen image foreground fraction | 0.6837 fraction | 0.5 | contrast-to-border mask on the gallery ocellus |
| specimen image structure-tensor coherency | 0.2359 dimensionless | 0.25 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the mask |
| domain-growth mode-count Pearson r | 1.0 correlation | 0.98 | N = k_measured L / 2π versus L/λ_c from one wavelength to a 10 mm lobe *(N(10 mm) = 38.06)* |
| successor petal mode count at 10 mm | 38.591 wavelengths | 1.0 | 10000 µm / λ_c |

## Environment at checkpoint


Pass 2 working tree on top of 0c76d15d. ~/.nanopub/profile.yml is absent, so the TriG files are unsigned.
