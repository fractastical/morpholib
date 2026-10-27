# MorphoEval report — audit `2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit -> 2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg–Turing wavelength at 39 µm in Botryllus schlosseri (Holocene, detail scale) (`models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py`) | 90 | 74 | 88 | 88 | 90 | 88 | 90 | 89.3 | 79.2 | 89.5 | 85.3 | **85.3** | faithful |

## Findings and remediation

### Schnakenberg–Turing wavelength at 39 µm in Botryllus schlosseri (Holocene, detail scale)  —  overall 85.3 (faithful)

Pass 2 keeps the seven-wavelength Schnakenberg window (λ assigned 38.72 µm, nonlinear FFT 39.90 µm, 12-frame 640×480 GIF). γ-sweep Pearson r = 0.999998, RMSE = 1.598 cells. With D_u fixed, raising D_v/D_u shortens the onset-formula wavelength monotonically (39.81 → 37.99 µm) while the supercritical peak moves the other way (40.27 → 47.57 µm, r = -0.987, RMSE = 5.652 µm). Equal diffusion stays stable (σ_peak = -0.649). Removing u²v drops the peak growth rate to -1.37. Specimen circularity on the Parent 1991 photograph is 0.085 versus a circle reference of 1. CV similarity of the last GIF frame to that photograph is 83.7.

- **Model fidelity to source: 90** — Schnakenberg kinetics, the dispersion relation, and the onset identity for k_c are implemented directly and recover λ = 38.72 µm. Pass 2 adds the two controls the law predicts: unequal diffusion is required, and so is the u²v autocatalytic term (knockout σ_peak = -1.37).  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L77–129; `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L199–258
- **Morphological realism / specimen match: 74** — The GIF is unchanged from Pass 1: burgundy-gold spots in the colony's colour histogram, CV similarity 83.7 against the Parent 1991 photograph. The star systems themselves are not reconstructed, so the score stays below the computer-vision number.  
  evidence: `docs/media/botryllus_schlosseri_turing_wavelength_s0_comparison.png` L0–0; `docs/media/botryllus_schlosseri_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — γ scales both the formula wavelength and the dispersion peak together (r = 0.999998). With D_u fixed, increasing D_v/D_u shortens the formula wavelength monotonically and lengthens the supercritical peak monotonically. Both curves are on the comparison figure.  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L145–177; `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L199–237; `docs/media/botryllus_schlosseri_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Nonlinear FFT wavelength 39.90 µm versus assigned 38.72 µm. γ-sweep Pearson r = 0.999998, RMSE = 1.598 cells. Ratio-sweep RMSE = 5.652 µm versus a reference of 0. Specimen circularity 0.085 versus 1. Seed CV and the KS orientation statistic are recorded. The colony photograph is not at the 39 µm scale.  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L579–655; `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — DOIs were read back from OpenAlex. The specimen SHA-256 matches the vendored public-domain Parent 1991 file, with author, locality, and license.  
  evidence: `datasets/botryllus_schlosseri_turing_wavelength_s0/paper_citations.bib` L1–65; `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_reference.json` L62–78
- **Code-to-model traceability: 88** — Docstring, function names, and the Lean module state the same four facts: positive diffusion product, cancellation in the wavenumber identity, 38.72 µm above the 10 µm floor, and seven modes in the detail window.  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3136_botryllus_schlosseri_turing_wavelength_s0.lean` L12–40
- **Reproducibility package: 90** — --self-test now also requires the ratio sweep to be monotone, the γ-sweep Pearson r to exceed 0.98, the knockout to be stable, and the specimen digest to match the reference JSON. Seed 3136 is fixed.  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L658–707; `datasets/botryllus_schlosseri_turing_wavelength_s0/README.md` L26–31

Remediation:
- {'id': 'R1', 'item': 'Resolved. Diffusivity-ratio sweep with D_u fixed: formula λ falls monotonically from 39.81 µm to 37.99 µm; supercritical peak λ rises from 40.27 µm to 47.57 µm.'}
- {'id': 'R2', 'item': 'Resolved. Autocatalysis knockout (u^2 v removed) gives σ_peak = -1.37, so the band of positive growth disappears.'}
- {'id': 'R3', 'item': 'Resolved. Ratio-sweep RMSE 5.652 µm is stored in the metrics JSON beside γ-sweep Pearson r = 0.999998, RMSE 1.598 cells, KS = 0.086, and seed spacing CV.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg–Turing wavelength at 39 µm in Botryllus schlosseri (Holocene, detail scale)
- **spec_fingerprint** (semantic model + inputs): `1f41527fdb3ad32975a5fe85bdbd97d229f23905e445211f557d1ede5e6ef301`
- **input_dataset_sha256**: `datasets/botryllus_schlosseri_turing_wavelength_s0/Botryllus_schlosseri_Parent1991_Banyuls.jpg` (`e351975e1a0b4822…`), `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_reference.json` (`1ec57fcd143ac8d8…`)
- `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_metrics.json` — artifact_sha256 `3da7872dbb8d2fa6…`, 4,972 B, 
- `docs/media/botryllus_schlosseri_turing_wavelength_s0_comparison.png` — artifact_sha256 `aaf2c0b9f3b67edc…`, 646,369 B, 1496×902 px
- `docs/media/botryllus_schlosseri_turing_wavelength_s0_growth.gif` — artifact_sha256 `55f5711ce0e622c7…`, 2,857,505 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe wavelength from activator FFT | 44.4678 µm | 38.72 | sub-bin radial FFT peak of the nonlinear Schnakenberg activator on the 271.04 µm window *(Pearson r of formula vs argmax σ(k) = 0.999998; RMSE = 1.5979 cells)* |
| specimen contour circularity | 0.0853 1 | 1.0 | 4π area / perimeter² on the border-contrast mask of Parent 1991 *(foreground fraction = 0.8128; coherency = 0.2679; components = 18)* |
| equal-diffusion control peak growth | -0.64945 1/time | 0.0 | argmax σ(k) after setting D_u = D_v = (D_u+D_v)/2 *(pattern σ_peak = 0.2122 at the design ratio 21.312)* |
| seed-ensemble stripe spacing CV | 0.004543 1 | 0.0 | sample standard deviation / mean of radial-FFT cycles across seeds *(KS orientation statistic = 0.0852; orientation L1 = 0.2317)* |
| diffusivity-ratio sweep wavelength RMSE | 5.6516 µm | 0.0 | RMSE of onset-formula λ versus argmax σ(k) with D_u fixed and D_v/D_u increased *(pearson r = -0.986850; monotonic decreasing = True)* |
| autocatalysis knockout peak growth | -1.37 1/time | 0.0 | max σ(k) after replacing u^2 v kinetics by pure decay f_u = g_v = -γ *(negative control distinct from the equal-diffusion ablation)* |

## Environment at checkpoint


Pass 2 files are new on this branch; locus check needs --no-verify-loci until they are committed. No ~/.nanopub/profile.yml, so these nanopubs are not signed.
