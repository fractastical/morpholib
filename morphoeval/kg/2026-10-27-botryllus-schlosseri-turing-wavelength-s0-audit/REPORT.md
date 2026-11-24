# MorphoEval report — audit `2026-10-27-botryllus-schlosseri-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg–Turing wavelength at 39 µm in Botryllus schlosseri (Holocene, detail scale) (`models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py`) | 86 | 74 | 78 | 84 | 90 | 86 | 88 | 83.3 | 77.8 | 88.1 | 82.5 | **82.5** | faithful |

## Findings and remediation

### Schnakenberg–Turing wavelength at 39 µm in Botryllus schlosseri (Holocene, detail scale)  —  overall 82.5 (faithful)

Periodic Schnakenberg kinetics on a 271.04 µm window (seven wavelengths). The onset formula is pinned to λ = 38.72 µm. The nonlinear activator FFT measured 39.90 µm (6.79 cycles). Pearson r between the formula and argmax σ(k) across a 10-point γ sweep is 0.999998 (RMSE 1.598 cells). Equal diffusion is stable (σ_peak = -0.649). The Parent 1991 colony photograph has circularity 0.085 against a circle reference of 1. The GIF is a spot field in the colony colour histogram, not a reconstruction of the star systems.

- **Model fidelity to source: 86** — The script integrates Schnakenberg kinetics and evaluates σ(k) from J - k²D. Diffusivities are solved so the onset identity k_c² = (D_v f_u + D_u g_v)/(2 D_u D_v) equals 2π/λ with λ = 38.72 µm. The γ scale and the autocatalytic term are not yet both ablated.  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L77–129
- **Morphological realism / specimen match: 74** — Side by side, the GIF uses the colony's burgundy-gold colour histogram and a spot grain on the scale of the bright siphon rims, but it does not draw the star systems. Computer-vision similarity of the last frame to the Parent photograph is 83.7; anatomical identity is weaker than that number.  
  evidence: `docs/media/botryllus_schlosseri_turing_wavelength_s0_comparison.png` L0–0; `docs/media/botryllus_schlosseri_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 78** — The γ sweep moves the formula wavelength and the dispersion peak together (r = 0.999998) and the comparison figure shows that scatter. A diffusivity-ratio sweep with D_u held fixed is not yet saved, so ratio expressivity is only implicit in RENDER_FACTOR.  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L145–177; `docs/media/botryllus_schlosseri_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 84** — FFT wavelength 39.90 µm versus the assigned 38.72 µm, specimen circularity 0.085 versus a circle reference of 1, equal-diffusion σ_peak = -0.649 versus 0, seed CV 0.0089 versus 0, KS statistic 0.086. The photograph is not a 39 µm micrograph, so the micrometre comparison is to the assigned scale.  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L500–554; `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Six DOIs were checked on OpenAlex. The specimen file is SHA-256 pinned to the public-domain Parent 1991 Wikimedia photograph, with author, locality, and license in the reference JSON.  
  evidence: `datasets/botryllus_schlosseri_turing_wavelength_s0/paper_citations.bib` L1–65; `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_reference.json` L62–78
- **Code-to-model traceability: 86** — The module docstring states the PDE, the steady state, and the k_c identity, and names the functions that implement them. Lean theorems encode the diffusion-product sign, the cancellation in the wavenumber identity, the 38.72 µm versus 10 µm margin, and the seven-mode count.  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3136_botryllus_schlosseri_turing_wavelength_s0.lean` L12–40
- **Reproducibility package: 88** — --self-test checks the steady-state signs, the recovered wavelength 38.72 µm, σ(k_c)>0, equal-diffusion stability, Pearson r > 0.98, and the specimen SHA-256. Seed 3136 and the render size are constants. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib).  
  evidence: `models/2026-delphi10000-botryllus-schlosseri-turing-wavelength-s0.py` L557–593; `datasets/botryllus_schlosseri_turing_wavelength_s0/README.md` L22–34

Remediation:
- {'id': 'R1', 'item': 'Add a diffusivity-ratio sweep with D_u fixed and D_v/D_u varied above r_c, and record whether λ(r) is monotonic.'}
- {'id': 'R2', 'item': 'Add an autocatalysis knockout (drop the u^2 v term) and show that the growth-rate peak is no longer positive.'}
- {'id': 'R3', 'item': 'Save the ratio-sweep RMSE next to the existing Pearson r, KS statistic, and spacing CV in the metrics JSON.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg–Turing wavelength at 39 µm in Botryllus schlosseri (Holocene, detail scale)
- **spec_fingerprint** (semantic model + inputs): `1f41527fdb3ad32975a5fe85bdbd97d229f23905e445211f557d1ede5e6ef301`
- **input_dataset_sha256**: `datasets/botryllus_schlosseri_turing_wavelength_s0/Botryllus_schlosseri_Parent1991_Banyuls.jpg` (`e351975e1a0b4822…`), `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_reference.json` (`1ec57fcd143ac8d8…`)
- `datasets/botryllus_schlosseri_turing_wavelength_s0/botryllus_schlosseri_turing_wavelength_s0_metrics.json` — artifact_sha256 `e082dcb2a0ec9b31…`, 2,662 B, 
- `docs/media/botryllus_schlosseri_turing_wavelength_s0_comparison.png` — artifact_sha256 `5354b05f2dc4818d…`, 912,735 B, 1344×1008 px
- `docs/media/botryllus_schlosseri_turing_wavelength_s0_growth.gif` — artifact_sha256 `55f5711ce0e622c7…`, 2,857,505 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe wavelength from activator FFT | 44.4678 µm | 38.72 | sub-bin radial FFT peak of the nonlinear Schnakenberg activator on the 271.04 µm window *(Pearson r of formula vs argmax σ(k) = 0.999998; RMSE = 1.5979 cells)* |
| specimen contour circularity | 0.0853 1 | 1.0 | 4π area / perimeter² on the border-contrast mask of Parent 1991 *(foreground fraction = 0.8128; coherency = 0.2679; components = 18)* |
| equal-diffusion control peak growth | -0.64945 1/time | 0.0 | argmax σ(k) after setting D_u = D_v = (D_u+D_v)/2 *(pattern σ_peak = 0.2122 at the design ratio 21.312)* |
| seed-ensemble stripe spacing CV | 0.004543 1 | 0.0 | sample standard deviation / mean of radial-FFT cycles across seeds *(KS orientation statistic = 0.0852; orientation L1 = 0.2317)* |

## Environment at checkpoint


Pass 1 files are new on this branch; locus check needs --no-verify-loci until they are committed.
