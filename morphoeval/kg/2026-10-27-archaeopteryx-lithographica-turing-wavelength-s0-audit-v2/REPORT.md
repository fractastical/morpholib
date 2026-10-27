# MorphoEval report — audit `2026-10-27-archaeopteryx-lithographica-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-archaeopteryx-lithographica-turing-wavelength-s0-audit -> 2026-10-27-archaeopteryx-lithographica-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength on Archaeopteryx lithographica at 75 µm (Tithonian, 150.0 Ma) (`models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py`) | 88 | 83 | 88 | 90 | 90 | 84 | 90 | 88.0 | 85.6 | 88.4 | 87.2 | **87.2** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength on Archaeopteryx lithographica at 75 µm (Tithonian, 150.0 Ma)  —  overall 87.2 (faithful)

Pass 2 keeps the same Schnakenberg system (a=0.112, b=0.836, Du=0.052, Dv=1.48, gamma=0.039, anisotropy 1.22, seed 3120) and the same 7.478e-05 m wavelength at 150.0 Ma. The wing window is now a full soft mask (fraction 0.080) and the critical mode is mixed onto that window at alpha 0.78, so vertical activator bars are readable on the vane. Grid FFT wavelength remains 12.00 cells versus 11.991 analytical (relative error 0.00072). Wing-transect 3-harmonic Pearson r = 0.9803, fundamental-only r = 0.807, nRMSE = 0.198, KS = 0.053. Equal-diffusion ablation std ratio = 0.00081. Neighbor predictions at the same dx: Anchiornis 95.33 µm (160.8 Ma), Microraptor 57.05 µm (120 Ma).

- **Model fidelity to source: 88** — Unchanged from Pass 1. dispersion() implements the Schnakenberg Jacobian and k_c^2 = (Dv fu + Du gv) / (2 Du Dv). The grid FFT peak (12.00 cells) matches 11.991 analytical cells. The numeric tuple is a reconstruction for this scale.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L63–93; `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L101–138
- **Morphological realism / specimen match: 83** — Pass 2 paints the critical mode into the wing window at alpha 0.78. The wing crop of the GIF shows broad vertical light and dark bars on the fossil texture, and the comparison figure's overlay panel shows the same patch. The rest of the Berlin slab is unchanged, so the animal remains recognizable (cv_sim_similarity 96.0). This is still a slab-scale photograph, not a 75 µm micrograph.  
  evidence: `docs/media/archaeopteryx_lithographica_turing_wavelength_s0_comparison.png` L0–0; `docs/media/archaeopteryx_lithographica_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L373–416
- **Parameter expressivity: 88** — lambda_c(gamma) on {0.020, 0.039, 0.080, 0.150, 0.250} is strictly decreasing and plotted. The chronoseries bars use one dx and three gamma values. The bars now visible on the wing are the gamma=0.039 mode that the sweep moves.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L178–184; `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L156–176
- **Morphometric validation: 90** — 3-harmonic Pearson r = 0.9803, fundamental-only r = 0.807 (now printed on the figure), nRMSE = 0.198, KS = 0.053, inter-peak CV = 0.194. Equal-diffusion control std ratio = 0.00081. Wing-mask fraction 0.080 is on the figure. The transect remains the scan, not a 75 µm caliper measurement.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L242–286; `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L455–458
- **Source and data traceability: 90** — Unchanged. Crossref-checked DOIs and a SHA-256-pinned public-domain Berlin slab.  
  evidence: `datasets/archaeopteryx_lithographica_turing_wavelength_s0/paper_citations.bib` L1–96; `datasets/archaeopteryx_lithographica_turing_wavelength_s0/archaeopteryx_lithographica_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Docstring and dispersion() name the PDE and k_c. Lean Loop_3120 checks the wavelength order and the diffusivity ratio. Pass 2 comments in feather_mask and render_gif say what changed relative to the faint Pass-1 mix.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L1–18; `morphoeval/lean/loops/Loop_3120_archaeopteryx_lithographica_turing_wavelength_s0.lean` L20–56
- **Reproducibility package: 90** — --self-test still passes after the overlay change: r > 0.98, ablation, monotonic sweep, Lean integers 9533/7478/5705, 12 frames at 960x720, cv_sim_similarity 96.0. Seed 3120 and the specimen hash are pinned.  
  evidence: `models/2026-delphi10000-archaeopteryx-lithographica-turing-wavelength-s0.py` L611–630

Remediation:
- {'id': 'R1', 'item': 'Resolved in Pass 2. Wing mix alpha is 0.78 inside the soft wing window; vertical bars are visible on the vane and cv_sim_similarity stayed 96.0.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved in Pass 2. The transect panel prints 3-harmonic r=0.9803 and fundamental r=0.807.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Resolved in Pass 2. The slab panel is titled with wing mask fraction 0.080.', 'status': 'resolved'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength on Archaeopteryx lithographica at 75 µm (Tithonian, 150.0 Ma)
- **spec_fingerprint** (semantic model + inputs): `6913098e47054812a2f1cf896b7e10fbf923735b2080ce927a3087eecb641c84`
- **input_dataset_sha256**: `datasets/archaeopteryx_lithographica_turing_wavelength_s0/archaeopteryx_lithographica_berlin_slab.jpg` (`2e307a2774e08571…`), `datasets/archaeopteryx_lithographica_turing_wavelength_s0/archaeopteryx_lithographica_turing_wavelength_s0_reference.json` (`32deca0baa44c4ef…`)
- `docs/media/archaeopteryx_lithographica_turing_wavelength_s0_comparison.png` — artifact_sha256 `4417000cdfb1483e…`, 752,917 B, 1496×924 px
- `docs/media/archaeopteryx_lithographica_turing_wavelength_s0_growth.gif` — artifact_sha256 `ec2fb7e3ba74959f…`, 10,309,092 B, 960×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Archaeopteryx lithographica specimen wing-vane optical-density Pearson r | 0.9803 Pearson r | 0.95 | gaussian sigma=3 column-mean of Berlin-slab ROI rows 450:850, cols 750:1200; quartic + 3 harmonics of the dominant FFT frequency *(fundamental-only r=0.8066; N=450; scan does not resolve 75 µm)* |
| specimen wing transect normalized RMSE | 0.1977 RMSE / sigma | 0.35 | sqrt(mean((OD - prediction)^2)) / std(OD) on the same wing transect *(KS two-sample statistic=0.0533)* |
| specimen versus fitted transect two-sample KS statistic | 0.0533 KS distance | 0.25 | scipy.stats.ks_2samp on the wing optical-density profile and the quartic-plus-harmonics prediction |
| Turing pigment stripe wavelength lambda_c at the Archaeopteryx cellular scale | 7.478e-05 m | 9e-07 | lambda_cells(gamma=0.039) * dx with dx chosen so this wavelength equals 7.478e-05 m *(measured/analytical grid relative error=0.00072)* |
| equal-diffusion ablation spatial-std ratio (Du=Dv control) | 0.000815 std(u_ablation)/std(u_turing) | 1.0 | Du=Dv=0.45, no critical-mode seed, 1800 steps, versus the Turing run *(std_turing=0.4968, std_ablation=0.000405)* |
| Anchiornis huxleyi predecessor cellular stripe wavelength | 9.533e-05 m | 7.478e-05 | same dx and kinetics, gamma=0.024 (slower symmetric-vane melanocyte kinetics) |
| Microraptor gui successor cellular stripe wavelength | 5.705e-05 m | 7.478e-05 | same dx and kinetics, gamma=0.067 (faster kinetics, iridescent melanosome stacks) |
| specimen slab structure-tensor coherency | 0.4665 coherency | 0.3 | border-contrast mask and Sobel structure tensor on the Berlin slab resized to 160x120 *(foreground_fraction=0.6664; wing_mask_fraction=0.0799; interpeak CV=0.1943)* |
| gamma sweep wavelength ratio lambda(0.020)/lambda(0.250) | 3.5355 ratio | 1.0 | analytical lambda_c across gamma in {0.020, 0.039, 0.080, 0.150, 0.250} *(strictly decreasing in gamma)* |

## Environment at checkpoint


Pass 2 working tree on 0c76d15d. Slot files are new and are not in that commit.
