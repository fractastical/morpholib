# MorphoEval report — audit `2026-10-27-diplomystus-dentatus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Diplomystus dentatus detail-tier Turing wavelength (184.3 µm, 51.5 Ma) (`models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py`) | 90 | 78 | 88 | 86 | 90 | 84 | 86 | 89.3 | 81.0 | 86.8 | 85.2 | **85.2** | faithful |

## Findings and remediation

### Diplomystus dentatus detail-tier Turing wavelength (184.3 µm, 51.5 Ma)  —  overall 85.2 (faithful)

Pass 1. Schnakenberg reaction-diffusion on a no-flux flank, with D_u chosen so the Turing onset wavelength is 184.3 µm at 51.5 Ma. The rendered field's radial peak is 186.0 µm. Dispersion Pearson r is 1.000 (RMSE 0.00033) and the diffusivity sweep Pearson r is 0.997. Equal diffusion collapses a linear perturbation (std ratio 0.031). A 20 µm chromatophore is sub-wavelength and its lowest mode decays. The Descouens Fossil Lake photograph is the specimen; its major-axis spacing is an organ-scale contrast, not a fit to 184.3 µm. The growth GIF ends on that photograph (12 frames, 960×640).

- **Model fidelity to source: 90** — The script integrates the Schnakenberg system and the onset formula, and the discrete Laplacian reproduces analytic growth rates at r=1.000. Kinetics are a transfer, not a Diplomystus measurement, so this is not a perfect source identity.  
  evidence: `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L1–40; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L62–102; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L125–237
- **Morphological realism / specimen match: 78** — The comparison figure places the real Fossil Lake specimen beside a 2.2 mm flank field with a 184.3 µm scale bar. The field is a labyrinth of ocelli, not a drawing of the ribs. The GIF carries that field on the fish and ends on the photograph.  
  evidence: `docs/media/diplomystus_dentatus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L469–594
- **Parameter expressivity: 88** — Scaling both diffusivities by s^2 moves the measured wavelength along the predicted line (sweep r=0.997) from about 130 µm to about 280 µm. Equal diffusion is the off-switch and does not produce a wavelength.  
  evidence: `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L300–318; `docs/media/diplomystus_dentatus_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 86** — Radial FFT wavelength 186.0 µm versus 184.3 µm, dispersion RMSE 0.00033, sweep RMSE reported, KS between fossil and field orientations, and a segmented flank spacing that is intentionally not equal to 0.1843 mm.  
  evidence: `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L239–266; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L326–425; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L596–676
- **Source and data traceability: 90** — Crossref-resolved DOIs for Turing, Schnakenberg, Murray, Kondo, Nakamasu, and Smith; Grande 1982 via the AMNH handle; specimen SHA-256, creator, and CC BY-SA 4.0 URL.  
  evidence: `datasets/diplomystus_dentatus_turing_wavelength_s0/paper_citations.bib` L1–71; `datasets/diplomystus_dentatus_turing_wavelength_s0/diplomystus_dentatus_turing_wavelength_s0_reference.json` L1–40; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L84–92
- **Code-to-model traceability: 84** — The module docstring and analytic_sigma map onto the onset formula. Lean states the scale inequalities (detail versus 7.56 mm, half-wave predecessor, sixteen waves in the window). Not every numerical choice cites a numbered equation in Turing 1952.  
  evidence: `morphoeval/lean/loops/Loop_3170_diplomystus_dentatus_turing_wavelength_s0.lean` L21–44; `morphoeval/math_kg_loops/delphi10000-diplomystus-dentatus-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 86** — Deterministic seed 3170, --self-test gates on Pearson r>0.98, wavelength error, ablation ratio, and predecessor sigma. SHA-256 pins. No separate pinned requirements file beyond the repo stack.  
  evidence: `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L53–70; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L683–722; `datasets/diplomystus_dentatus_turing_wavelength_s0/README.md` L1–40

Remediation:
- {'id': 'R1', 'item': 'The flank field is an isotropic labyrinth. A dorsoventral anisotropy trial is still needed to see whether rib-aligned stripes can be selected without leaving the 184.3 µm onset wavelength.'}
- {'id': 'R2', 'item': 'The 48 cm caption calibration of flank spacing has no uncertainty. Report a resampled interval.'}
- {'id': 'R3', 'item': 'Single builder-assessor. Do not invent a second provider score.'}
- {'id': 'R4', 'item': 'The equal-diffusion ablation is linearised. Add a small-amplitude nonlinear control that stays inside the fixed-point basin.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Diplomystus dentatus detail-tier Turing wavelength (184.3 µm, 51.5 Ma)
- **spec_fingerprint** (semantic model + inputs): `dca96e708aacc37802c4b54ab60381864ab6fced4c5a2aa1dbd4e8eeec3dd5e9`
- `docs/media/diplomystus_dentatus_turing_wavelength_s0_comparison.png` — artifact_sha256 `337592b6425e02b4…`, 1,059,751 B, 1736×1176 px

| measurement | value | reference | method |
|---|---:|---|---|
| discrete-Laplacian dispersion Pearson r | 1.0 1 | 1.0 | Pearson r between analytic σ(k) using k^2=(2-2cos q)/dx^2 and growth of pure modes in the explicit periodic scheme *(n=11 modes, RMSE=0.00033 per time unit)* |
| dispersion growth-rate RMSE | 0.00033 1/time | 0.0 | RMSE of measured versus analytic σ on the same discrete modes *(nonzero because of the explicit Euler transient and the quadratic term at finite amplitude)* |
| flank radial-FFT wavelength | 186.0387 µm | 184.3 | power-weighted radial FFT peak of the Neumann Schnakenberg field; D_u solved so onset λ_c=184.3 µm *(relative error 0.0094; D_u=2.694010e-10 m^2/s, D_v/D_u=9.17)* |
| diffusivity-sweep wavelength Pearson r | 0.997235 1 | 1.0 | Pearson r of predicted s*184.3 µm versus measured λ when both diffusivities scale by s^2, s in {0.72,0.88,1,1.18,1.46} *(sweep RMSE 5.67 µm)* |
| equal-diffusion ablation std ratio | 0.031474 1 | 1.0 | final/initial std of a linearised perturbation when D_v is set equal to D_u (Turing band removed) *(control: ratio near 0 means the homogeneous state is restored; ratio 1 would be a failed ablation)* |
| 20 µm predecessor-mode growth rate | -5.958628 1/time | 0.0 | analytic σ at the lowest Neumann wavenumber k=π/L on a 20 µm chromatophore *(negative σ: the cell-scale predecessor cannot carry the 184.3 µm wave)* |
| specimen flank vertical-element spacing | 6.329 mm | 0.1843 | high-pass FFT along the PCA major axis of the segmented Descouens fish; 48 cm caption length calibrates the axis *(reference is the 184.3 µm detail wave expressed in mm. The fossil transect resolves organ-scale ribs/scutes, not the primordium wave. Value differs from the reference on purpose.)* |
| specimen mask aspect ratio | 1.4743 1 | 1.0 | major/minor PCA extents of the segmented fossil versus a circular null *(foreground fraction 0.934, circularity 0.119)* |
| specimen-versus-field orientation KS statistic | 0.135 1 | 0.0 | two-sample KS on Sobel orientation of the fossil versus the simulated flank field *(p=1.45e-30. The fossil is dominated by sub-vertical ribs; the ocellus field is not a copy of that histogram.)* |

## Environment at checkpoint


Parent commit at the start of the loop. New slot files are not in 0c76d15d; loci were checked with --no-verify-loci.
