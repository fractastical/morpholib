# MorphoEval report — audit `2026-10-27-drosophila-berryi-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila berryi detail-scale Turing wavelength at 1.754 µm (19 Ma) (`models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py`) | 86 | 76 | 82 | 84 | 88 | 84 | 88 | 84.7 | 79.0 | 86.9 | 83.1 | **83.1** | faithful |

## Findings and remediation

### Drosophila berryi detail-scale Turing wavelength at 1.754 µm (19 Ma)  —  overall 83.1 (faithful)

Pass 1 Schnakenberg model of the Drosophila berryi detail-scale Turing wavelength. Classical lambda_c is calibrated to 1.754 µm at 19 Ma. Modal growth rates match the Jacobian eigenvalue with Pearson r = 0.999998 and RMSE = 0.000250. The chronoseries at fixed mode number N = 55.87 shortens from 2.040 µm (succini, 44 Ma) to 1.754 µm (berryi) to 1.718 µm (affinis, 0 Ma). Equal-diffusion and u^2 v ablation both decay, but the equal-diffusion rate is not yet quantitative. The gallery plate is an amber spiral, not a 1.8 µm micrograph.

- **Model fidelity to source: 86** — Implements the Schnakenberg Jacobian and the Turing formula k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), plus the leading eigenvalue sigma(k). The 1.754 µm unit is a calibration of that formula, not a new equation. Nonlinear coarsening of the 2D field is not yet part of the tested law.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L95–168; `morphoeval/lean/loops/Loop_3173_drosophila_berryi_turing_wavelength_s0.lean` L10–40
- **Morphological realism / specimen match: 76** — The comparison figure places the gallery plate beside the amber-rendered Schnakenberg field. The plate is a project amber spiral with a 100 µm printed bar, not a 1.8 µm micrograph, so the visual match is a computer-vision comparison against the canonical asset rather than a fossil SEM. The GIF is 12 frames at 720x540 and its last frame is the plate.  
  evidence: `docs/media/drosophila_berryi_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_berryi_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 82** — Nine wavenumbers are integrated and plotted against sigma(k). Equal diffusivities and deletion of u^2 v are separate controls and both decay. The chronoseries table varies the metric at fixed kinetics. An explicit D_v grid of lambda_sigma is not yet drawn.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L187–268; `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L335–369
- **Morphometric validation: 84** — Pearson r of measured versus predicted sigma(k) is 0.999998 with RMSE 0.000250 on nine wavenumbers (KS statistic 0.111). Specimen foreground fraction 0.3445, coherency 0.981 and radial wavelength norm 0.20 are compared with independent reference targets 0.55, 0.45 and 0.08. The equal-diffusion control has the correct sign but the measured rate (-1.32) misses the eigenvalue (-0.41), so the control is not yet a quantitative fit.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L506–614; `datasets/drosophila_berryi_turing_wavelength_s0/drosophila_berryi_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 88** — Crossref-checked DOIs for Turing 1952, Schnakenberg 1979, Murray 1981, Kondo and Miura 2010, Grimaldi 1995 and Tamura 2003. Grimaldi 1987 has no Crossref DOI; the Biodiversity Heritage Library landing page and OpenAlex id are recorded. The specimen SHA-256 is pinned.  
  evidence: `datasets/drosophila_berryi_turing_wavelength_s0/paper_citations.bib` L1–68; `datasets/drosophila_berryi_turing_wavelength_s0/drosophila_berryi_turing_wavelength_s0_reference.json` L1–74
- **Code-to-model traceability: 84** — Module docstring and function docstrings name the Turing formula, the Schnakenberg steady state, the periodic boundary and the chronoseries construction. Lean declarations are named in the math-kg loop file. Line comments do not yet cite Grimaldi page numbers, because the wavelength is not in that paper.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L1–55; `morphoeval/math_kg_loops/delphi10000-drosophila-berryi-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 88** — Seed 3173, --self-test covering the discriminant, the 1.754 µm calibration, chronoseries order, Pearson r > 0.98, both controls, and the specimen hash. Matplotlib Agg backend. Single-command render writes the PNG, the GIF and the metrics JSON.  
  evidence: `models/2026-delphi10000-drosophila-berryi-turing-wavelength-s0.py` L670–703; `datasets/drosophila_berryi_turing_wavelength_s0/README.md` L1–34

Remediation:
- {'id': 'R1', 'item': 'Equal-diffusion control matches the sign of sigma but not the rate (-1.32 measured vs -0.41 eigenvalue). Refit with a smaller step and an earlier window.'}
- {'id': 'R2', 'item': 'Late-time nonlinear coarsening of the 2D Schnakenberg field is not measured against lambda_sigma.'}
- {'id': 'R3', 'item': 'No second assessor. The gallery plate is not a 1.8 µm fossil micrograph; keep that limit explicit rather than raising specimen-match to match the GIF crop score.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila berryi detail-scale Turing wavelength at 1.754 µm (19 Ma)
- **spec_fingerprint** (semantic model + inputs): `fe2e74bbe1194adf9f66d7de47fdd4b5119a2272757767687404f3140401fa54`
- `datasets/drosophila_berryi_turing_wavelength_s0/drosophila_berryi_turing_wavelength_s0_metrics.json` — artifact_sha256 `77dc23b7d635de03…`, 3,970 B, 
- `docs/media/drosophila_berryi_turing_wavelength_s0_comparison.png` — artifact_sha256 `564dff560a1b2d9d…`, 430,729 B, 1344×1008 px
- `docs/media/drosophila_berryi_turing_wavelength_s0_growth.gif` — artifact_sha256 `4d88e755064ab8d3…`, 3,521,552 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| modal growth-rate Pearson r of sigma(k) | 0.999998 Pearson r | 1.0 | Pearson r, RMSE and KS two-sample between predicted and measured growth rates across 9 wavenumbers *(RMSE=0.000250; KS statistic=0.1111; p=1)* |
| modal growth-rate RMSE | 0.00025 1/time | 0.0 | RMSE of measured minus predicted sigma(k) *(dispersion RMSE)* |
| dispersion KS statistic | 0.1111 KS | 0.45 | KS two-sample of predicted versus measured sigma samples *(p=1)* |
| classical critical wavelength lambda_c | 1.754 µm | 1.754 | lambda_c = 2 pi / k_c with k_c from the Turing formula, unit calibrated to 1.754 µm *(k_c=3.582204e+06 1/m at 19 Ma)* |
| fastest-growing wavelength lambda_sigma | 2.2525 µm | 1.754 | argmax_k sigma(k) converted by the berryi metric *(not equal to lambda_c; the difference is the trace penalty)* |
| equal-diffusion negative-control growth rate | -1.3219 1/time | 0.0 | pure-mode integration with D_v set equal to D_u (negative control) *(analytical sigma=-0.4099; decays=True)* |
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

Pass 1 on the loop branch before the coarsening and equal-diffusion rate fix. New files are not in the parent commit; loci checked with --no-verify-loci.
