# MorphoEval report — audit `2026-10-27-diplomystus-dentatus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-diplomystus-dentatus-turing-wavelength-s0-audit -> 2026-10-27-diplomystus-dentatus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Diplomystus dentatus detail-tier Turing wavelength (184.3 µm, 51.5 Ma) (`models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py`) | 90 | 78 | 90 | 90 | 90 | 86 | 88 | 90.0 | 82.5 | 88.1 | 86.4 | **86.4** | faithful |

## Findings and remediation

### Diplomystus dentatus detail-tier Turing wavelength (184.3 µm, 51.5 Ma)  —  overall 86.4 (faithful)

Pass 2. Same Schnakenberg onset wavelength, 184.3 µm at 51.5 Ma. Radial peak remains 186.0 µm (dispersion r=1.000, RMSE=0.00033; diffusivity sweep r=0.997). Remediation: a Dy/Dx=1.8 dorsoventral trial selected 212 µm (15% off the onset law) and was rejected, so the isotropic field is kept. A 1e-5 cubic equal-diffusion perturbation decays (std ratio 0.118), in addition to the linearised ratio 0.031. Five minor-axis transects on the Descouens fish give a central spacing of 6.33 mm with a placement std of 11.7 mm, so the fossil spacing is millimetre-scale and is not the 0.1843 mm wave. The assessor is still a single agent.

- **Model fidelity to source: 90** — The script integrates the Schnakenberg system and the onset formula, and the discrete Laplacian reproduces analytic growth rates at r=1.000. Kinetics are a transfer, not a Diplomystus measurement, so this is not a perfect source identity.  
  evidence: `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L1–40; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L62–102; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L125–237
- **Morphological realism / specimen match: 78** — The comparison figure places the real Fossil Lake specimen beside a 2.2 mm flank field with a 184.3 µm scale bar. The field is a labyrinth of ocelli, not a drawing of the ribs. The GIF carries that field on the fish and ends on the photograph.  
  evidence: `docs/media/diplomystus_dentatus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L469–594
- **Parameter expressivity: 90** — The s^2 diffusivity sweep still tracks predicted wavelength (r=0.997). A separate Dy/Dx=1.8 trial moves the peak to 212 µm and is rejected, which is the expressivity result: anisotropy changes the selected scale and is not free.  
  evidence: `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L347–365; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L312–339
- **Morphometric validation: 90** — Wavelength 186.0 versus 184.3 µm, dispersion RMSE 0.00033, sweep r=0.997, KS 0.135, flank spacing 6.33 mm versus 0.1843 mm, and a five-transect placement std of 11.7 mm. The anisotropy trial (212 µm) is reported as a failed alternative, not folded into the fit.  
  evidence: `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L373–401; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L658–760
- **Source and data traceability: 90** — Crossref-resolved DOIs for Turing, Schnakenberg, Murray, Kondo, Nakamasu, and Smith; Grande 1982 via the AMNH handle; specimen SHA-256, creator, and CC BY-SA 4.0 URL.  
  evidence: `datasets/diplomystus_dentatus_turing_wavelength_s0/paper_citations.bib` L1–71; `datasets/diplomystus_dentatus_turing_wavelength_s0/diplomystus_dentatus_turing_wavelength_s0_reference.json` L1–40; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L84–92
- **Code-to-model traceability: 86** — Onset formula, discrete dispersion probe, rejected anisotropy, and both ablations are separate functions. Lean still carries the scale inequalities with zero sorry. Turing 1952 section numbers are not cited line by line.  
  evidence: `morphoeval/lean/loops/Loop_3170_diplomystus_dentatus_turing_wavelength_s0.lean` L21–44; `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L294–339
- **Reproducibility package: 88** — Seed 3170, SHA-256 pins, and --self-test now also gates the nonlinear ablation ratio. The anisotropy rejection is deterministic. Still no loop-local requirements pin beyond the repo stack.  
  evidence: `models/2026-delphi10000-diplomystus-dentatus-turing-wavelength-s0.py` L769–809

Remediation:
- {'id': 'R3', 'item': 'Still a single builder-assessor. A second provider was not available; no second score was invented.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Diplomystus dentatus detail-tier Turing wavelength (184.3 µm, 51.5 Ma)
- **spec_fingerprint** (semantic model + inputs): `dca96e708aacc37802c4b54ab60381864ab6fced4c5a2aa1dbd4e8eeec3dd5e9`
- `docs/media/diplomystus_dentatus_turing_wavelength_s0_comparison.png` — artifact_sha256 `f34d3f2bca1546c3…`, 1,060,255 B, 1736×1176 px

| measurement | value | reference | method |
|---|---:|---|---|
| discrete-Laplacian dispersion Pearson r | 1.0 1 | 1.0 | Pearson r between analytic σ(k) using k^2=(2-2cos q)/dx^2 and growth of pure modes in the explicit periodic scheme *(n=11 modes, RMSE=0.00033 per time unit)* |
| dispersion growth-rate RMSE | 0.00033 1/time | 0.0 | RMSE of measured versus analytic σ on the same discrete modes *(nonzero because of the explicit Euler transient and the quadratic term at finite amplitude)* |
| flank radial-FFT wavelength | 186.0387 µm | 184.3 | power-weighted radial FFT peak of the Neumann Schnakenberg field; D_u solved so onset λ_c=184.3 µm *(relative error 0.0094; D_u=2.694010e-10 m^2/s, D_v/D_u=9.17)* |
| diffusivity-sweep wavelength Pearson r | 0.997235 1 | 1.0 | Pearson r of predicted s*184.3 µm versus measured λ when both diffusivities scale by s^2, s in {0.72,0.88,1,1.18,1.46} *(sweep RMSE 5.67 µm)* |
| equal-diffusion ablation std ratio | 0.031474 1 | 1.0 | final/initial std of a linearised perturbation when D_v is set equal to D_u (Turing band removed) *(control: ratio near 0 means the homogeneous state is restored; ratio 1 would be a failed ablation)* |
| nonlinear equal-diffusion ablation std ratio | 0.118462 1 | 1.0 | final/initial std of a 1e-5 cubic perturbation with D_v = D_u and dt=0.02 *(stays inside the fixed-point basin; distinct from the linearised ablation)* |
| rejected dorsoventral anisotropy wavelength | 212.413 µm | 184.3 | radial FFT after Dy/Dx = 1.8; accepted only if relative error < 0.08 *(accepted=False. The isotropic field is kept because the trial leaves the onset wavelength.)* |
| specimen flank spacing resample std | 11.6676 mm | 0.0 | std of high-pass spacing across five transects offset along the minor axis (±0, 4, 8 px) *(placement uncertainty on the 48 cm caption calibration; not a second copy of the mean spacing)* |
| 20 µm predecessor-mode growth rate | -5.958628 1/time | 0.0 | analytic σ at the lowest Neumann wavenumber k=π/L on a 20 µm chromatophore *(negative σ: the cell-scale predecessor cannot carry the 184.3 µm wave)* |
| specimen flank vertical-element spacing | 6.329 mm | 0.1843 | high-pass FFT along the PCA major axis of the segmented Descouens fish; 48 cm caption length calibrates the axis *(reference is the 184.3 µm detail wave expressed in mm. The fossil transect resolves organ-scale ribs/scutes, not the primordium wave. Value differs from the reference on purpose.)* |
| specimen mask aspect ratio | 1.4743 1 | 1.0 | major/minor PCA extents of the segmented fossil versus a circular null *(foreground fraction 0.934, circularity 0.119)* |
| specimen-versus-field orientation KS statistic | 0.135 1 | 0.0 | two-sample KS on Sobel orientation of the fossil versus the simulated flank field *(p=1.45e-30. The fossil is dominated by sub-vertical ribs; the ocellus field is not a copy of that histogram.)* |

## Environment at checkpoint


Pass 2 on the same parent commit 0c76d15d. New files are not in that commit; loci checked with --no-verify-loci. No nanopub profile, graph unsigned.
