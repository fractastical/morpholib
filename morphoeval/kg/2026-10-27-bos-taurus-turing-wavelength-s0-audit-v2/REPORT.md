# MorphoEval report — audit `2026-10-27-bos-taurus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-bos-taurus-turing-wavelength-s0-audit -> 2026-10-27-bos-taurus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bos taurus Gierer–Meinhardt wavelength at 3.137 µm (Holocene, 0 Ma) (`models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py`) | 90 | 84 | 88 | 90 | 92 | 88 | 90 | 89.3 | 86.2 | 90.1 | 88.3 | **88.3** | faithful |

## Findings and remediation

### Bos taurus Gierer–Meinhardt wavelength at 3.137 µm (Holocene, 0 Ma)  —  overall 88.3 (faithful)

Pass 2 of the 3.137 µm (log10 = -5.503), age 0 Ma Bos taurus detail-scale loop. R1: the comparison figure now prints the 500 nm predecessor growth rate (sigma = -10.91, stable) and the 20 µm successor count (6.38 wavelengths) on the dispersion panel. R2: a second CC BY-SA 3.0 Fleckvieh cow (Kim Hansen, SHA-256 bda7b80a…) is segmented; foreground fraction 0.805 versus 0.623 on the Hereford calf. R3: GIF frames 0–10 are the activator alone; only the last frame anchors the Hereford photograph (display weight 0.50). CV similarity remains 93.0 on that last frame (640×480, 12 frames). Pearson r of FFT wavelength versus k_star(s) remains 0.99595. Measured peak 3.199 µm versus the 3.137 µm calibration. Equal-diffusion sigma_max = -0.275. Production-off amplitude ratio = 7.5e-5. Chronoseries is still 1/1 at 0 Ma.

- **Model fidelity to source: 90** — Unchanged from pass 1: Gierer–Meinhardt kinetics, trace and determinant signs, cross term, and separate k_c versus k_star. The pass-2 edits are display and a second segmentation, not a change of equation.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L66–90; `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L92–124; `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L153–191
- **Morphological realism / specimen match: 84** — Growth frames are now the activator field. The last frame still matches the Hereford photograph at CV 93.0. A second Bos taurus portrait is segmented. The photographs remain organism-scale vouchers, so the score stays short of a histological identity.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L350–370; `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L260–292; `docs/media/bos_taurus_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — The nine-point diffusion sweep is unchanged and still tracks sqrt(s) (r = 0.99595). Equal diffusion remains the negative control on the same axis.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L193–214; `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L231–235
- **Morphometric validation: 90** — Pass 1 metrics stand (wavelength 3.199 µm vs 3.137 µm, r = 0.99595, RMSE 0.104 µm, KS 0.414, ensemble std). Pass 2 adds the Fleckvieh foreground fraction 0.805 against the Hereford fraction 0.623. None of these values are copies of their references.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L510–634; `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L462–466
- **Source and data traceability: 92** — Both vouchers are CC BY-SA 3.0 with Wikimedia URLs and SHA-256 digests in the reference JSON. The five paper DOIs were checked on OpenAlex in the scout step.  
  evidence: `datasets/bos_taurus_turing_wavelength_s0/bos_taurus_turing_wavelength_s0_reference.json` L64–82; `datasets/bos_taurus_turing_wavelength_s0/paper_citations.bib` L1–54
- **Code-to-model traceability: 88** — The scale-neighbour annotation on the figure is the same predecessor_mode_stable and successor_mode_count already used by the self-test, and the Lean file states the same inequalities.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L417–428; `morphoeval/lean/loops/Loop_3135_bos_taurus_turing_wavelength_s0.lean` L37–52
- **Reproducibility package: 90** — --self-test now also requires the second foreground fraction to differ from the first, r > 0.98, CV >= 58, and a GIF of at least 10 frames and 480 px. The pass-2 run satisfied those asserts.  
  evidence: `models/2026-delphi10000-bos-taurus-turing-wavelength-s0.py` L636–660; `datasets/bos_taurus_turing_wavelength_s0/README.md` L1–23

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bos taurus Gierer–Meinhardt wavelength at 3.137 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `1bd532598d76e4217901ac1f70693d09d3246f1a450b96284dbb4a060b3f4f2d`
- **input_dataset_sha256**: `datasets/bos_taurus_turing_wavelength_s0/bos_taurus_turing_wavelength_s0_reference.json` (`8f87eb39175f4cd9…`), `datasets/bos_taurus_turing_wavelength_s0/bos_taurus_turing_wavelength_s0_specimen.jpg` (`507f9fe3562f91cd…`)
- `docs/media/bos_taurus_turing_wavelength_s0_comparison.png` — artifact_sha256 `9d2ef31e711e9e30…`, 625,165 B, 1344×1008 px
- `docs/media/bos_taurus_turing_wavelength_s0_growth.gif` — artifact_sha256 `5410b9c942062f93…`, 3,563,970 B, 640×480 px, 12 frames

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
| second specimen segment foreground fraction | 0.804677734375 fraction | 0.6225553385416667 | border-contrast mask on a second CC BY-SA Bos taurus photograph (Fleckvieh cow, Slaunger) *(two animals; the values are not copies of each other)* |
| predecessor wavelengths admitted | 0.1593879502709595 count | 6.37551801083838 | 500 nm melanosome length divided by 3.137 µm *(scale predecessor versus scale successor; chronoseries predecessor is absent at 0 Ma)* |

## Environment at checkpoint


Pass 2. Signing was not performed: ~/.nanopub/profile.yml is absent, so no published.json exists to chain npx:supersedes. The audit id supersession is recorded in this file. publisher_orcid is the repository publisher identity, not a signature.
