# MorphoEval report — audit `2026-10-27-drosophila-erecta-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-erecta-turing-wavelength-s0-audit -> 2026-10-27-drosophila-erecta-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg–Turing critical wavelength for Drosophila erecta at 2.262 µm (Pass 2) (`models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py`) | 88 | 82 | 90 | 86 | 92 | 86 | 90 | 88.7 | 83.5 | 89.6 | 86.9 | **86.9** | faithful |

## Findings and remediation

### Schnakenberg–Turing critical wavelength for Drosophila erecta at 2.262 µm (Pass 2)  —  overall 86.9 (faithful)

Dispersion wavelength is 2.262 µm at a=0.17, b=0.93, Dv/Du=16, age 0 Ma. Gallery annulus Pearson r=0.99803, RMSE=1.966 gray levels, KS=0.064. Annulus radius 97.30 µm versus the elegans primordium anchor 101.1 µm. δ sweep λ falls 2.847, 2.415, 2.310, 2.262, 2.217, 2.183 µm. Critical-mode std ratio 3.36; equal-diffusion ratio 7.5e-5; ablation ratio 2.1e-7. Dark/light production shift 1.24 exceeds wavelength shift 0.30. Wavenumber versus elegans is ×44.69. GIF CV similarity against the gallery image is 82.0 (16 frames, 640×480).

- **Model fidelity to source: 88** — k_c^2 = (Dv fu + Du gv)/(2 Du Dv) is solved so the physical wavelength is 2.262 µm (Du=3.987e-14 m^2/s, δ=16). Trace=-0.519 and det=1.210, so the homogeneous state is stable until diffusion. The disk-mode envelope is an extra radial fit on top of that law, which is why this is not 100.  
  evidence: `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L78–111; `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L124–188
- **Morphological realism / specimen match: 82** — Direct CV similarity of the 16-frame 640×480 GIF against the gallery specimen is 82.0 (Sobel 12-bin, radial FFT, coherency, occupancy). The annulus reads as the same bright core and dark ring. The image is the project gallery field, not a wild fly, so the score stays below the mid-90s.  
  evidence: `docs/media/drosophila_erecta_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_erecta_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — At fixed Du, λ(δ) for δ=4,8,12,16,24,40 is 2.847, 2.415, 2.310, 2.262, 2.217, 2.183 µm, strictly decreasing. Dark versus light production shift is 1.235 while the wavelength shift is 0.305. Equal diffusion and autocatalysis removal both kill the pattern.  
  evidence: `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L351–392
- **Morphometric validation: 86** — Annulus Pearson r=0.99803 against reference ceiling 1.0 (RMSE=1.966, so the series are not copies). KS=0.064 against anchor 0.5. Segmented ring radius 97.30 µm against the elegans primordium anchor 101.1 µm. PDE patch grows by 3.36× at the critical mode; the negative controls fall below 1e-4.  
  evidence: `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L236–258; `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L482–551
- **Source and data traceability: 92** — DOIs resolve for Turing 1952, Schnakenberg 1979, Gierer & Meinhardt 1972, Yassin et al. 2016, Kondo & Miura 2010, and Aguilar-Hidalgo et al. 2016. Specimen SHA-256 ef1ddb03… is pinned. The empty Wikimedia/GBIF search is recorded in the scout manifest.  
  evidence: `datasets/drosophila_erecta_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/drosophila_erecta_turing_wavelength_s0/scout_manifest.json` L1–80
- **Code-to-model traceability: 86** — critical_kc2 cites the Turing quotient. Lean theorems pin 2262 nm < 101100 nm and 100600 nm, the δ=16 threshold 16×691 > 1210, and the equal-diffusion failure. The script's self-test checks those same inequalities.  
  evidence: `morphoeval/lean/loops/Loop_3183_drosophila_erecta_turing_wavelength_s0.lean` L16–47; `morphoeval/math_kg_loops/delphi10000-drosophila-erecta-turing-wavelength-s0.yaml` L1–58
- **Reproducibility package: 90** — Seed 3183, --self-test, and a deterministic GIF (16 frames, 640×480). Self-test passed on the dispersion identity, r>0.98, both controls, and the Lean module. Dependencies are the repo scientific stack (numpy, scipy, Pillow, matplotlib).  
  evidence: `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` L553–616; `datasets/drosophila_erecta_turing_wavelength_s0/README.md` L1–47

Remediation:
- {'id': 'R1', 'item': 'Resolved in this pass: δ sweep is monotonic (2.847 µm at δ=4 down to 2.183 µm at δ=40).', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved in this pass: equal-diffusion std ratio 7.5e-5 and ablation ratio 2.1e-7 versus critical-mode ratio 3.36.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Resolved in this pass: Pearson r=0.99803, RMSE=1.966, KS=0.064, ring 97.30 µm ≠ 101.1 µm anchor, GIF CV=82.0.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'No second assessor from another provider. Still open.', 'status': 'open'}
- {'id': 'R5', 'item': 'No CC wild photograph. Gallery fallback remains the specimen. Still open.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg–Turing critical wavelength for Drosophila erecta at 2.262 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `944cf6527af7b0d5c21d4cf33a0403c90ad4f3ca2ab561dcf67e43e4ef7022f3`
- **input_dataset_sha256**: `datasets/drosophila_erecta_turing_wavelength_s0/drosophila_erecta_specimen.jpg` (`ef1ddb031fefd5f8…`), `datasets/drosophila_erecta_turing_wavelength_s0/drosophila_erecta_turing_wavelength_s0_reference.json` (`db334ff62f6ae794…`), `datasets/drosophila_erecta_turing_wavelength_s0/paper_citations.bib` (`2b2bacd184e5772d…`)
- `docs/media/drosophila_erecta_turing_wavelength_s0_comparison.png` — artifact_sha256 `e24b0d85978700ec…`, 637,721 B, 1380×984 px
- `docs/media/drosophila_erecta_turing_wavelength_s0_growth.gif` — artifact_sha256 `b30de403594b50fd…`, 2,568,655 B, 640×480 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen annulus radius versus elegans primordium wavelength | 97.2973 µm | 101.1 | argmin of the gallery radial luminance profile, times the drawn 100 µm scale bar *(The annulus is the ~100 µm ring in the gallery image. It is not the 2.262 µm detail wavelength.)* |
| PDE detail wavelength versus assigned 2.262 µm target | 2.262 µm | 2.3 | FFT peak of the grown Schnakenberg patch, scaled by sqrt(Du) *(relative error vs 2.262 µm = 0.0000; RMSE of the radial fit = 1.966)* |
| specimen radial-profile Pearson r | 0.99803 correlation | 1.0 | Pearson r between the gallery annulus profile and the damped J0 disk mode *(RMSE=1.966 gray levels; KS=0.0640)* |
| specimen radial-profile KS distance | 0.064 KS statistic | 0.5 | two-sample KS between empirical annulus samples and the disk-mode prediction *(ks statistic is nonzero, so the fit is not a self-copy of the samples)* |
| equal-diffusion control versus critical-mode growth ratio | 0.0001 std ratio | 3.3599 | perturbation std after the run divided by the initial std; δ=1 negative control *(control decays; the δ=16 patch grows. Autocatalysis ablation ratio = 0.0000)* |
| diffusion-ratio sweep endpoint wavelength ratio | 1.3042 λ(δ=4) / λ(δ=40) | 1.0 | analytic k_c at fixed Du across δ = 4, 8, 12, 16, 24, 40 *(monotonic decreasing control of wavelength by inhibitor range)* |

## Environment at checkpoint


Parent commit at audit time. Slot 3183 files are new in the working tree. Nanopubs are unsigned because ~/.nanopub/profile.yml is absent.
