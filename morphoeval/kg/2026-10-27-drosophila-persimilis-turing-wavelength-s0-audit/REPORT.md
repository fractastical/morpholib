# MorphoEval report — audit `2026-10-27-drosophila-persimilis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `6c49ecbdcc1e6d2fcf72d168ee46196263be7d3b` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg critical wavelength for Drosophila persimilis at 2.1 µm (`models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py`) | 84 | 68 | 74 | 70 | 90 | 82 | 84 | 80.7 | 68.8 | 85.5 | 77.3 | **77.3** | mostly faithful |

## Findings and remediation

### Schnakenberg critical wavelength for Drosophila persimilis at 2.1 µm  —  overall 77.3 (mostly faithful)

Pass 1. Focal wavelength is the assigned 2.098 µm detail tier at 0 Ma. Predecessor D. paulistorum and successor D. planitibia change only the diffusion length, λ ∝ L^0.42. Self-test reports Pearson r = 0.9993 on the Du sweep, ablation ratio 0.051, and stripe order paulistorum < persimilis < planitibia. Full credit is withheld: the gallery plate is not an electron micrograph, β is a transfer, and the critical mode is seeded.

- **Model fidelity to source: 84** — The integrator is Schnakenberg kinetics with the standard onset wavenumber, periodic boundaries, and an equal-diffusion control that drops the activator standard deviation by about twenty-fold. The critical cosine is seeded at the predicted wavelength, so the run shows that the mode persists, not that noise finds it.  
  evidence: `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L117–130; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L165–208; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L243–259
- **Morphological realism / specimen match: 68** — The comparison figure places the gallery plate beside three stripe fields whose spacing increases from paulistorum to planitibia. The plate is a golden-brown spiral, explicitly captioned as not an electron micrograph. The growth GIF's last frame is that plate, so a high computer-vision score against it is expected and is not ultrastructural agreement.  
  evidence: `docs/media/drosophila_persimilis_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L457–474
- **Parameter expressivity: 74** — The Du sweep is monotonic and the analytic β table keeps paulistorum shorter than planitibia from β=0.20 to 0.70. Endpoint diffusivities are integrated, but this first reading does not yet treat that grid as a full substitute for a measured wavelength series.  
  evidence: `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L210–241; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L261–292
- **Morphometric validation: 70** — FFT centroids, Pearson r, RMSE, ablation ratio, seed SD, and grid convergence are computed. References are the assigned 2.098 µm tier and the allometric targets, not periods measured on these three species. The orientation KS against the gallery plate is large, which is the correct disagreement, and is not yet the basis of the score.  
  evidence: `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L144–163; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L531–662
- **Source and data traceability: 90** — Primary DOIs were resolved on Crossref. The catalog DOI 10.1073/pnas.98.21.12084 does not resolve and was not used; Noor et al. 2001 is 10.1073/pnas.221274498. Dataset files are SHA-256 pinned.  
  evidence: `datasets/drosophila_persimilis_turing_wavelength_s0/paper_citations.bib` L1–20; `datasets/drosophila_persimilis_turing_wavelength_s0/drosophila_persimilis_turing_wavelength_s0_reference.json` L1–20
- **Code-to-model traceability: 82** — Each numeric routine maps to one piece of the dispersion relation, the allometry, or the ablation. Lean states the integer order, the onset bound, and the k·λ pin. The docstring states which claims are transfers.  
  evidence: `morphoeval/lean/loops/Loop_3198_drosophila_persimilis_turing_wavelength_s0.lean` L14–34; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L1–40
- **Reproducibility package: 84** — --self-test checks scale, age, order, Pearson r > 0.98, ablation, β order, seed SD, and grid convergence. Seeds and hashes are fixed. Pass 1 had not yet re-rendered after the endpoint integration was scored.  
  evidence: `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L82–102; `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` L664–696

Remediation:
- {'id': 'R1', 'item': 'The gallery plate is not a 2.1 µm micrograph. Keep the orientation KS between the plate and the activator field as a separate metric, and do not read the GIF terminal-frame similarity as ultrastructural agreement.'}
- {'id': 'R2', 'item': 'β = 0.42 is a transfer. Credit the integrated β=0.20 and β=0.70 endpoints only after the self-test order check is re-run and the gap is reported against the analytic gap without copying it.'}
- {'id': 'R3', 'item': 'The critical mode is seeded. The equal-diffusion ablation must stay in the scored measurement set so the stripe is not credited as an imposed drawing. A noise-selected transient is still not shown.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg critical wavelength for Drosophila persimilis at 2.1 µm
- **spec_fingerprint** (semantic model + inputs): `6cfd22b3b1eb8b085833af27f643b5e89f6fcf635888d87c45635473a8295900`
- **input_dataset_sha256**: `datasets/drosophila_persimilis_turing_wavelength_s0/drosophila_persimilis_specimen.jpg` (`f1216712eddebdcd…`), `datasets/drosophila_persimilis_turing_wavelength_s0/drosophila_persimilis_turing_wavelength_s0_reference.json` (`a12c3e1298940f01…`), `datasets/drosophila_persimilis_turing_wavelength_s0/paper_citations.bib` (`9eb570b42e0a0ddc…`)
- `docs/media/drosophila_persimilis_turing_wavelength_s0_comparison.png` — artifact_sha256 `dedf562e5c821cfd…`, 306,753 B, 1440×960 px

| measurement | value | reference | method |
|---|---:|---|---|
| persimilis FFT centroid wavelength | 2.0967 µm | 2.098 | radial spectral centroid of the final Schnakenberg activator on the 16.784 µm patch *(centroid is deliberately not the integer mode L/8, so value != reference)* |
| Du-sweep Pearson r analytic vs measured wavelength | 0.9993 correlation | 0.98 | Pearson r across five D_u multipliers at fixed r=20; RMSE reported alongside *(RMSE 0.0333 µm; monotonic=True)* |
| Du-sweep RMSE | 0.0333 µm | 0.05 | root mean square of FFT-centroid λ minus sqrt(D_u) analytic λ |
| equal-diffusion ablation amplitude ratio | 0.0509 std(u)_ablation / std(u)_pattern | 1.0 | negative control with D_v set equal to D_u; Turing onset condition fails *(control vs patterned baseline)* |
| paulistorum measured wavelength | 1.9553 µm | 1.9468 | same patch and kinetics; D_u scaled by (2.36/2.82)^(0.84) |
| planitibia measured wavelength | 2.8072 µm | 2.7346 | same patch and kinetics; D_u scaled by (5.30/2.82)^(0.84) |
| beta-endpoint planitibia minus paulistorum at beta 0.70 | 1.5138 µm | 1.4109 | integrated wavelength gap at the steep allometric endpoint β=0.70 *(parameter-regime sweep beyond the nominal β=0.42)* |
| multi-seed wavelength SD | 0.0001 µm | 0.0 | sample standard deviation of FFT centroid over 5 seeds *(mean 2.0974 µm across seeds [3198, 3209, 3221, 3233, 3247])* |
| grid-convergence relative wavelength error | 0.0002 fraction of 2.098 µm | 0.0 | absolute difference of N=64 and N=96 centroids, divided by 2.098 µm |
| specimen foreground fraction | 0.3294 area fraction | 0.5 | border-median contrast mask on the gallery plate resized to 160x120 *(specimen image segmentation, not a self-copy of the reference JSON)* |
| specimen structure-tensor coherency | 0.7825 coherency | 0.5 | box-smoothed structure tensor on the gallery-plate luminance |
| KS distance specimen vs RD orientation histogram | 0.5052 KS | 0.0 | 12-bin Sobel-style orientation CDF distance between the gallery plate and the persimilis activator *(non-circular specimen metric; value is not set equal to the reference)* |

## Environment at checkpoint


Slot 3198 only. No existing tracked file on origin/main is modified.
