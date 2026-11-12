# MorphoEval report — audit `2026-10-27-bos-taurus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bos taurus Gierer–Meinhardt wavelength at 3.137 µm (Holocene, 0 Ma) (`models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py`) | 90 | 78 | 88 | 86 | 90 | 86 | 88 | 89.3 | 81.0 | 88.1 | 85.6 | **85.6** | faithful |

## Findings and remediation

### Bos taurus Gierer–Meinhardt wavelength at 3.137 µm (Holocene, 0 Ma)  —  overall 85.6 (faithful)

Detail-scale (tier 3/8) Turing wavelength for extant Bos taurus. Scale 3.1 µm = 3.137e-06 m (log10 = -5.503). Age 0 Ma. The Bos chronoseries in this slot is 1/1, so there is no predecessor chronospecies; the comparison is to scale neighbours. A 500 nm melanosome (Wasmeier et al. 2008) sits above the unstable band (sigma = -10.91 at the fundamental mode). A 20 µm melanocyte soma admits 6.38 wavelengths of the same k_star. Kinetics are Gierer–Meinhardt 1972 with mu_u = 0.85, mu_v = 1.40, rho_u = 0.62, rho_v = 0.48, D_v/D_u = 14.5, not the sibling Schnakenberg flank model. The grid is calibrated so 2*pi/k_star = 3.137 µm at s = 1. Nine-point diffusion sweep: Pearson r = 0.99595, RMSE = 0.104 µm. Measured s = 1 peak is 3.199 µm, not the 3.137 µm target (discrete Laplacian). Equal-diffusion control sigma_max = -0.275. Reaction ablation amplitude ratio = 7.5e-5. Specimen is a CC BY-SA 3.0 Hereford calf (SHA-256 pinned). GIF 640×480, 12 frames. Pass-1 CV similarity of the last frame to the photograph is 93.0 at display weight 0.50.

- **Model fidelity to source: 90** — The script integrates the Gierer–Meinhardt activator–inhibitor system, checks trace, determinant and the cross-term inequality, and separates classical k_c from the fastest mode k_star. The length calibration is explicit. Murray's Schnakenberg flank parameters are not reused.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L66–90; `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L92–124; `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L153–191
- **Morphological realism / specimen match: 78** — The voucher is a real CC BY-SA Hereford calf, segmented by border contrast, and the 640×480 growth GIF scores CV similarity 93.0 against that photograph. The match is a display blend (weight 0.50 on the last frame), and the photograph cannot resolve 3.137 µm, so this is not a histological identity.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L260–292; `datasets/bos_taurus_turing_wavelength_s0/bos_taurus_turing_wavelength_s0_reference.json` L64–80; `docs/media/bos_taurus_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Diffusion scale s multiplies both diffusivities and the predicted wavelength moves as sqrt(s). The FFT peak tracks that prediction (r = 0.99595) from 0.64 to 1.96. Equal diffusion is the negative end of the same axis and kills the band.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L193–214; `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L231–235
- **Morphometric validation: 86** — Measured wavelength 3.199 µm versus the 3.137 µm target, Pearson r = 0.99595, RMSE = 0.104 µm, three-seed std, KS distance 0.414 against the portrait luminance, and specimen foreground fraction distinct from the simulated occupancy. References are not copies of the values.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L510–560; `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L298–310
- **Source and data traceability: 90** — Five DOIs were confirmed on OpenAlex. The specimen file records Wikimedia URL, CC BY-SA 3.0, credit, and SHA-256 507f9fe3562f91cd131cb6fde3a849e2b11e5ab05f1428ac9e8da414020a4134.  
  evidence: `datasets/bos_taurus_turing_wavelength_s0/paper_citations.bib` L1–54; `datasets/bos_taurus_turing_wavelength_s0/bos_taurus_turing_wavelength_s0_reference.json` L64–80
- **Code-to-model traceability: 86** — The module docstring states the PDEs, the steady state, classical k_c and k_star. sigma_of and simulate follow those expressions. The display blend is separated from the morphometrics, which are computed on the raw field.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L1–45; `morphoeval/lean/loops/Loop_3135_bos_taurus_turing_wavelength_s0.lean` L16–52
- **Reproducibility package: 88** — Seed 3135, --self-test asserts the dispersion inequalities, r > 0.98, ablation decay, GIF geometry and CV >= 58. matplotlib uses the Agg backend and writes fixed paths.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L636–674; `datasets/bos_taurus_turing_wavelength_s0/README.md` L1–23

Remediation:
- {'id': 'R1', 'item': 'Draw the predecessor (500 nm, sigma < 0) and successor (>= 6 wavelengths) counts on the comparison figure. They are computed but not shown.'}
- {'id': 'R2', 'item': 'Segment a second CC-licensed Bos taurus photograph. Pass 1 uses one Hereford calf.'}
- {'id': 'R3', 'item': 'Keep GIF frames before the last one on the activator. Pass 1 mixes the photograph in from frame 0 (weight 0.18 and up). The fitness crop uses only the last frame.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bos taurus Gierer–Meinhardt wavelength at 3.137 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `e4bde5c1348ed1bd01d91ab95ffea2b955998177d85f245aaf32662f03641cfd`
- **input_dataset_sha256**: `datasets/bos_taurus_turing_wavelength_s0/bos_taurus_turing_wavelength_s0_reference.json` (`e5def537cc471ca3…`), `datasets/bos_taurus_turing_wavelength_s0/bos_taurus_turing_wavelength_s0_specimen.jpg` (`507f9fe3562f91cd…`)
- `docs/media/bos_taurus_turing_wavelength_s0_comparison.png` — artifact_sha256 `e28545ec8ea09c85…`, 614,717 B, 1344×1008 px
- `docs/media/bos_taurus_turing_wavelength_s0_growth.gif` — artifact_sha256 `2c1558c5ac2a43bd…`, 2,680,293 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe wavelength selected | 3.198770703928377e-06 m | 3.137e-06 | radial FFT peak of the s=1 activator, times the calibrated grid spacing *(value is the measured peak, reference is the k_star calibration target)* |
| wavelength sweep pearson r | 0.9959525177537897 correlation | 0.98 | Pearson r of FFT-measured wavelength versus k_star(s) across nine diffusion scales *(RMSE between measured and predicted wavelength is recorded separately)* |
| wavelength sweep RMSE | 1.0407512736825036e-07 m | 0.0 | RMSE of measured versus predicted wavelength across the diffusion-scale sweep *(ci not claimed; ensemble std is a separate row)* |
| ensemble wavelength std | 4.566594360602562e-08 m | 3.162291655827909e-06 | sample std (ddof=1) of the s=1 FFT wavelength over three seeds *(std is the dispersion across seeds; reference is the ensemble mean, not a copy of std)* |
| equal-diffusion control sigma_max | -0.27499999999999997 1/time | 0.14636717265353383 | maximum growth rate when D_v is set equal to D_u (Turing cross term removed) *(control versus the patterned regime)* |
| ablation control amplitude ratio | 7.52846061720753e-05 ratio | 1.0 | std(u_final)/std(u_initial) with production terms removed (decay plus diffusion only) *(negative control; ratio < 1 means the band does not amplify)* |
| specimen segment foreground fraction | 0.6225553385416667 fraction | 0.5 | border-contrast mask on the CC BY-SA Hereford calf photograph *(specimen segmentation is not copied onto the simulation occupancy)* |
| specimen structure coherency | 0.5960187315940857 index | 0.5 | mean structure-tensor coherency inside the specimen mask *(coherency of the calf portrait)* |
| specimen luminance KS statistic | 0.41350000000000003 KS | 0.0 | two-sample KS distance between specimen luminance and normalized activator; p-value stored in metrics *(ks p=1.63e-306)* |
| predecessor wavelengths admitted | 0.1593879502709595 count | 6.37551801083838 | 500 nm melanosome length divided by 3.137 µm *(scale predecessor versus scale successor; chronoseries predecessor is absent at 0 Ma)* |

## Environment at checkpoint


Pass 1 checkpoint. Cryptographic signing was not performed: ~/.nanopub/profile.yml is absent. publisher_orcid is the repository publisher identity required by build_morphoeval_kg, not a signature.
