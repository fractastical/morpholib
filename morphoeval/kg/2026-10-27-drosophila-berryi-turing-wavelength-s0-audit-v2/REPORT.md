# MorphoEval report — audit `2026-10-27-drosophila-berryi-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-berryi-turing-wavelength-s0-audit -> 2026-10-27-drosophila-berryi-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila berryi detail-scale Turing wavelength at 1.754 µm (19 Ma) (`models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py`) | 90 | 82 | 90 | 92 | 90 | 88 | 90 | 90.0 | 85.8 | 89.5 | 88.1 | **88.1** | faithful |

## Findings and remediation

### Drosophila berryi detail-scale Turing wavelength at 1.754 µm (19 Ma)  —  overall 88.1 (faithful)

Pass 2 keeps the Schnakenberg critical wavenumber calibrated to 1.754 µm at 19 Ma (Pearson r = 0.999998, RMSE = 0.000250 on nine modes). The equal-diffusion control is now the real part of the complex eigenvalue: measured envelope -0.4051 versus analytical -0.4099. A six-point D_v sweep lengthens lambda_sigma monotonically from 1.960 µm to 2.614 µm. The early 2D field locks to lambda_sigma = 2.253 µm. Detail wavelengths at fixed mode number N = 55.87 are 2.040 µm (succini, 44 Ma), 1.754 µm (berryi, 19 Ma) and 1.718 µm (affinis, 0 Ma). The gallery plate remains an amber spiral, not a 1.8 µm micrograph.

- **Model fidelity to source: 90** — The script evaluates the Schnakenberg Jacobian, the classical k_c formula, and the leading eigenvalue, including the complex pair that appears when D_v = D_u. Pass 2 seeds that complex mode and fits its envelope instead of a real cosine that aliased the oscillation.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L95–168; `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L335–391
- **Morphological realism / specimen match: 82** — Comparison figure shows the gallery plate, the amber Schnakenberg field, the dispersion with the corrected control, and the chronoseries bars. Computer-vision similarity of the 12-frame 720x540 GIF against the plate is 96.0 because the terminal frame is the plate. Specimen-match stays at 82 because the plate is not a 1.8 µm micrograph of berryi pigment; foreground fraction 0.3445 versus an a priori target 0.55 records that gap.  
  evidence: `docs/media/drosophila_berryi_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_berryi_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L440–475
- **Parameter expressivity: 90** — Nine wavenumbers trace sigma(k). The D_v sweep from 0.90 to 3.40 lengthens lambda_sigma monotonically from 1.960 µm to 2.614 µm. Equal diffusion and deletion of u^2 v both decay. The chronoseries metric is a separate axis from this diffusivity sweep.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L393–416; `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L187–268
- **Morphometric validation: 92** — Pearson r = 0.999998 and RMSE = 0.000250 against the eigenvalue curve (KS statistic 0.111 versus an a priori mismatch threshold 0.45). Equal-diffusion envelope -0.4051 versus analytical -0.4099 (error 0.0048). The 2D FFT peak locks to lambda_sigma = 2.253 µm. Specimen foreground, coherency and radial wavelength use reference targets that are not copies of the measured values.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L585–710; `datasets/drosophila_berryi_turing_wavelength_s0/drosophila_berryi_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Crossref DOIs for Turing 1952, Schnakenberg 1979, Murray 1981, Kondo and Miura 2010, Grimaldi 1995 and Tamura 2003. Grimaldi 1987 is pinned by its Biodiversity Heritage Library URL and OpenAlex id. Specimen SHA-256 f00982007d1d2ba67d3795b961e75b6b8b53f24d68c137ff2fe186fc07c4ffa5.  
  evidence: `datasets/drosophila_berryi_turing_wavelength_s0/paper_citations.bib` L1–77; `datasets/drosophila_berryi_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 88** — Docstrings name the Turing formula, the complex-mode control, the D_v sweep and the chronoseries. Lean theorems in namespace MathOfBiology.Loop_3173 encode the wavelength, the Miocene gap, the diffusion-ratio sign and the mode count, with no sorry.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L1–55; `morphoeval/lean/loops/Loop_3173_drosophila_berryi_turing_wavelength_s0.lean` L8–43
- **Reproducibility package: 90** — --self-test checks the 1.754 µm calibration, chronoseries order, Pearson r > 0.98, the equal-diffusion rate error below 0.02, ablation decay, the monotone D_v sweep and the specimen hash. Seed 3173. One command writes the PNG, the GIF and the metrics JSON.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L772–808; `datasets/drosophila_berryi_turing_wavelength_s0/README.md` L1–34

Remediation:
- {'id': 'R1', 'item': 'Closed in Pass 2. Equal-diffusion envelope is -0.4051 versus analytical Re(sigma) -0.4099.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Closed in Pass 2. Early 2D FFT peak equals lambda_sigma (2.253 µm) and the D_v sweep is monotone.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Open. No second assessor. The gallery plate is not a 1.8 µm fossil micrograph; specimen-match is not raised to the GIF crop score.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila berryi detail-scale Turing wavelength at 1.754 µm (19 Ma)
- **spec_fingerprint** (semantic model + inputs): `fe2e74bbe1194adf9f66d7de47fdd4b5119a2272757767687404f3140401fa54`
- `datasets/drosophila_berryi_turing_wavelength_s0/drosophila_berryi_turing_wavelength_s0_metrics.json` — artifact_sha256 `e47bedf3e9c031e9…`, 4,977 B, 
- `docs/media/drosophila_berryi_turing_wavelength_s0_comparison.png` — artifact_sha256 `7dc94265a3154b3e…`, 420,283 B, 1344×1008 px
- `docs/media/drosophila_berryi_turing_wavelength_s0_growth.gif` — artifact_sha256 `4d88e755064ab8d3…`, 3,521,552 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| modal growth-rate Pearson r of sigma(k) | 0.999998 Pearson r | 1.0 | Pearson r, RMSE and KS two-sample between predicted and measured growth rates across 9 wavenumbers *(RMSE=0.000250; KS statistic=0.1111; p=1)* |
| modal growth-rate RMSE | 0.00025 1/time | 0.0 | RMSE of measured minus predicted sigma(k) *(dispersion RMSE)* |
| dispersion KS statistic | 0.1111 KS | 0.45 | KS two-sample of predicted versus measured sigma samples *(p=1)* |
| classical critical wavelength lambda_c | 1.754 µm | 1.754 | lambda_c = 2 pi / k_c with k_c from the Turing formula, unit calibrated to 1.754 µm *(k_c=3.582204e+06 1/m at 19 Ma)* |
| fastest-growing wavelength lambda_sigma | 2.2525 µm | 1.754 | argmax_k sigma(k) converted by the berryi metric *(not equal to lambda_c; the difference is the trace penalty)* |
| equal-diffusion negative-control growth rate | -0.4051 1/time | -0.4099 | envelope fit of a complex-mode seed with D_v set equal to D_u (negative control) *(rate error=0.0048; decays=True; Pass 2 replaces the cosine projection that aliased the oscillation)* |
| two-dimensional field wavelength versus lambda_sigma | 2.2525 µm | 2.2525 | radial FFT peak of the periodic Schnakenberg field during linear growth *(early-time 2D selection; late coarsening is not required for this lock)* |
| inhibitor-diffusivity sweep monotone flag | 1.0 flag | 1.0 | six-point D_v sweep of the fastest-growing wavelength at fixed D_u *(parameter expressivity of the inhibitor diffusivity; distinct from the chronospecies metric)* |
| autocatalysis-ablation growth rate | -1.1649 1/time | 0.0 | ablation control: reaction replaced by linear relaxation f=a-u, g=b-v *(analytical sigma=-1.1600; decays=True)* |
| specimen amber-spiral foreground fraction | 0.3445 fraction | 0.55 | border-contrast foreground segmentation of Drosophila_berryi_specimen.jpg *(gallery plate is an amber spiral, not a 1.8 µm SEM)* |
| specimen structure-tensor coherency | 0.981 coherency | 0.45 | mean structure-tensor coherency on the gallery plate *(CV orientation feature of the canonical specimen)* |
| specimen radial wavelength norm | 0.2 fraction of frame | 0.08 | radial FFT peak of the gallery plate divided by the frame *(the printed bar is 100 µm; this slot's law is 1.754 µm, so the ratio is not 1)* |
| chronoseries detail-wavelength ratio berryi over succini | 0.8596 ratio | 0.8596 | detail lambda(berryi) / detail lambda(succini) *(Miocene contraction versus the 44 Ma predecessor)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2. Parent commit does not contain these new files; build uses --no-verify-loci.
