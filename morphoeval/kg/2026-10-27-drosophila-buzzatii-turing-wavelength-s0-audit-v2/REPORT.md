# MorphoEval report — audit `2026-10-27-drosophila-buzzatii-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `00782eec` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-buzzatii-turing-wavelength-s0-audit -> 2026-10-27-drosophila-buzzatii-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg–Turing wavelength at 2.908 µm in Drosophila buzzatii (Pass 2) (`models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py`) | 91 | 85 | 91 | 89 | 91 | 88 | 90 | 91.0 | 86.5 | 89.8 | 88.8 | **88.8** | faithful |

## Findings and remediation

### Schnakenberg–Turing wavelength at 2.908 µm in Drosophila buzzatii (Pass 2)  —  overall 88.8 (faithful)

Pass 2 re-ran the script. Analytical lambda is 2.908 µm at 0 Ma. Discrete-mode sweep Pearson r = 0.99995 (RMSE 0.962 µm). Saturated spots sit at 3.53 µm. Ablation growth is negative. KS radial between the card and the pure field is 0.458. Image fitness against the canonical card is 85.4 with CV similarity 84.6 on a 12-frame 720×540 GIF. On a 1.70 mm chord the mode counts are 584 (buzzatii), 19 (bipectinata), and 8 (cyrtoloma).

- **Model fidelity to source: 91** — Schnakenberg kinetics, the four Turing inequalities, and the critical-wavenumber formula are implemented directly. gamma is solved for this slot's 2.908 µm target. The 102 µm sibling's power-law coefficients are not used.  
  evidence: `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L86–120; `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L182–223; `morphoeval/lean/loops/Loop_3179_drosophila_buzzatii_turing_wavelength_s0.lean` L28–48
- **Morphological realism / specimen match: 85** — Side by side, the pure field is a spot lattice (micro-ocelli) and the GIF paints that density onto the canonical card, from a blank frame (activator std 0.010) to a spotted frame (std 0.926). The card itself is a radial texture, not a wing micrograph, so the score stays below a photographic match.  
  evidence: `docs/media/drosophila_buzzatii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_buzzatii_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 91** — Raising gamma shortens both the analytical and the discrete-mode wavelength (panel D, r = 0.99995). Setting D_v = D_u flips max Re(sigma) from positive to -0.050 (panel F).  
  evidence: `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L152–179; `docs/media/drosophila_buzzatii_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 89** — Pearson r = 0.99995 and RMSE = 0.962 µm on the dispersion sweep. Nonlinear wavelength 3.53 µm versus linear 2.908 µm. Specimen coherency differs from field coherency. KS radial = 0.458 against a reference of 0. None of the specimen metrics are value == reference.  
  evidence: `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L226–317; `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L534–627
- **Source and data traceability: 91** — DOIs for Turing, Schnakenberg, Norry 1997, and Cortese 2002. Specimen SHA-256 21d216f74bbb27351f7455fa27e4496514b9adc5b31a51f85c6fcd9a283d39a4. Scout notes the failed Commons/iNaturalist/GBIF search and the gallery fallback.  
  evidence: `datasets/drosophila_buzzatii_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/drosophila_buzzatii_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/drosophila_buzzatii_turing_wavelength_s0/README.md` L1–20
- **Code-to-model traceability: 88** — Docstring, Lean theorems, and the math-kg loop file name the same wavenumber, diffusion ratio, and mode-count statements, with line-level code loci.  
  evidence: `morphoeval/math_kg_loops/delphi10000-drosophila-buzzatii-turing-wavelength-s0.yaml` L24–55; `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L320–333
- **Reproducibility package: 90** — Deterministic seed 3179, --self-test, pinned specimen hash, and a re-run that reprints r, ablation, KS, and the GIF frame count.  
  evidence: `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L630–716

Remediation:
- {'id': 'R1', 'item': 'KS radial is 0.458 (reference 0). Closed in measure_morphoeval.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Card segmentation reports foreground, circularity, and coherency, each unequal to its reference.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Panel F shows max Re(sigma) for D_v/D_u = 29 and for D_v = D_u.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Saturated wavelength 3.53 µm is reported beside the linear target 2.908 µm (RMSE on the sweep is 0.962 µm).', 'status': 'resolved'}
- {'id': 'R5', 'item': 'Still one assessor, and the canonical image is a species card rather than a wing micrograph. A second assessor and a CC adult photograph would be the next real gain; neither was available in this run.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg–Turing wavelength at 2.908 µm in Drosophila buzzatii (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `08fa3b6fbda37d14034e66aaa32be6a4f2884d2974f79734f2b5414d7f6640c2`
- **input_dataset_sha256**: `datasets/drosophila_buzzatii_turing_wavelength_s0/drosophila_buzzatii_turing_wavelength_s0_reference.json` (`61a3792c4a139711…`), `datasets/drosophila_buzzatii_turing_wavelength_s0/drosophila_buzzatii_turing_wavelength_s0_specimen.jpg` (`21d216f74bbb2735…`)
- `docs/media/drosophila_buzzatii_turing_wavelength_s0_comparison.png` — artifact_sha256 `e036de5b2cd1da21…`, 469,642 B, 1512×984 px
- `docs/media/drosophila_buzzatii_turing_wavelength_s0_growth.gif` — artifact_sha256 `338bf89ea58e68dc…`, 3,467,062 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_wavelength_pearson_r | 0.999952 correlation | 0.98 | pearsonr of analytical λ_c versus the most unstable discrete Fourier mode across 14 γ scales |
| dispersion_wavelength_rmse_um | 0.9619 µm | 0.0 | RMSE between closed-form λ_c and the discrete-mode λ on the same γ sweep |
| simulated_ocellus_stripe_wavelength_um | 3.5324 µm | 2.908 | parabolic-interpolated radial FFT peak of the saturated Schnakenberg activator |
| specimen_segment_foreground_fraction | 0.3785 fraction | 0.5 | border-median contrast mask on the canonical species card, 320×240 |
| specimen_structure_tensor_coherency | 0.6748 index | 0.2483 | mean structure-tensor coherency inside the specimen mask |
| specimen_segment_circularity | 0.026 index | 1.0 | 4π area / perimeter² of the specimen foreground mask |
| specimen_vs_field_ks_radial | 0.4583 KS | 0.0 | two-sample KS on 24-bin log radial spectra of the card versus the pure field |
| equal_diffusion_ablation_max_growth | -0.049563 1/time | 0.0 | max Re(σ) of the discrete dispersion with D_v set equal to D_u |
| buzzatii_wing_chord_mode_count | 584 modes | 8 | floor(1700 µm / 2.908 µm) on a representative wing chord |

## Environment at checkpoint

- python: 3.12
- numpy: 2.5.4
- scipy: 1.18.1

Pass 2 of slot 3179, checkpoint 00782eec. Signing was not available: ~/.nanopub/profile.yml is absent, so the nanopubs are unsigned.
