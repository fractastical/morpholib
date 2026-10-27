# MorphoEval report — audit `2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0e0828eed603126b2ece796473b70906c8bb4076` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit -> 2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg most-unstable wavelength for D. mojavensis epicuticle at 2.4 µm (0 Ma) (`models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py`) | 90 | 82 | 90 | 88 | 90 | 84 | 88 | 90.0 | 84.2 | 87.6 | 87.0 | **87.0** | faithful |

## Findings and remediation

### Schnakenberg most-unstable wavelength for D. mojavensis epicuticle at 2.4 µm (0 Ma)  —  overall 87.0 (faithful)

Pass 2 closes R1 and R2. No-flux and periodic patches select the same radial-FFT bin (2.41388 µm) while the saturated std moves from 1.01072 (periodic) to 0.99918 (Neumann). Saturated FFT wavelengths track the square law: D. miranda 3.2825 µm vs analytic 3.2364 µm, focal small-grid 2.4139 µm, D. novamexicana 1.9754 µm vs analytic 1.9477 µm, in that order. Dispersion Pearson r remains 0.99323 (RMSE 0.02405). GIF unchanged: 14 frames, 640×480, CV 87.1. Image fitness 85.7 = 0.55*87.1 + 0.30*82 + 0.15*88. Age 0 Ma. Scale 2.4 µm (2.38e-06 m).

- **Model fidelity to source: 90** — Pass 2 adds the no-flux Laplacian beside the periodic integrator. Both select the same radial bin (2.41388 µm), so the wavelength is set by the dispersion relation rather than by the boundary. Absolute Du, Dv remain a calibration to 2.38e-06 m.  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L102–156; `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L879–923
- **Morphological realism / specimen match: 82** — Unchanged render: 14-frame 640×480 GIF, pattern weight 0.18, CV similarity 87.1 against the canonical plate. The score stays below the CV number because the plate is a stylized spiral with a 100 µm bar, not a 2.4 µm micrograph. Pass 2 did not retune the blend.  
  evidence: `docs/media/drosophila_mojavensis_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_mojavensis_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — Analytic λ★(s) is monotone, and the saturated FFT now follows it at the two chronospecies scales: miranda 3.2825 vs 3.2364 µm, novamexicana 1.9754 vs 1.9477 µm, with the focal small-grid bin in between. Equal-diffusion ablation still collapses.  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L926–963; `datasets/drosophila_mojavensis_turing_wavelength_s0/metrics.json` L154–165
- **Morphometric validation: 88** — Pearson r=0.99323, RMSE=0.02405. Plate vs habitus foreground 0.3806 vs 0.5455, coherency 0.8155 vs 0.5838, KS(activator, plate)=0.858. Chronospecies FFT errors are about 1.4% (miranda and novamexicana). Neumann std 0.99918 vs periodic 1.01072. None of these pairs have value == reference.  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L205–224; `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L706–812
- **Source and data traceability: 90** — Unchanged from Pass 1. Crossref-checked DOIs, SHA-256 pins, and an explicit rejection of the sibling slot's mis-assigned DOI 10.1534/genetics.106.059444.  
  evidence: `datasets/drosophila_mojavensis_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/drosophila_mojavensis_turing_wavelength_s0/drosophila_mojavensis_turing_wavelength_s0_reference.json` L1–30
- **Code-to-model traceability: 84** — Pass 2 functions carry docstrings stating the no-flux stencil and the square-law grid. Lean still has four theorems and no sorry. The GIF weight remains a documented render choice.  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L879–963; `morphoeval/lean/loops/Loop_3194_drosophila_mojavensis_turing_wavelength_s0.lean` L14–42
- **Reproducibility package: 88** — --self-test passes, including the new order miranda FFT > focal FFT > novamexicana FFT within 15% of the analytic wavelengths, and value != reference on every morphometric row. Seed 3194. Same explicit Euler.  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L815–854; `datasets/drosophila_mojavensis_turing_wavelength_s0/README.md` L1–28

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.3]** `parameter_sweep` → `parameter-expressivity`: Saturated FFT wavelength at the predecessor and successor diffusivity scales.
- **[RESOLVED · P=1.1]** `model_equation` → `model-fidelity`: Neumann boundary integrated and compared with the periodic patch.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg most-unstable wavelength for D. mojavensis epicuticle at 2.4 µm (0 Ma)
- **spec_fingerprint** (semantic model + inputs): `b1989635e560992257bdc385b54a49911ad758777fefbeab3e3411ca482c2805`
- `docs/media/drosophila_mojavensis_turing_wavelength_s0_comparison.png` — artifact_sha256 `33eeba72316e26ac…`, 491,888 B, 1512×984 px
- `docs/media/drosophila_mojavensis_turing_wavelength_s0_growth.gif` — artifact_sha256 `98d86591b76e9ff0…`, 3,810,451 B, 640×480 px, 14 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe_wavelength_um | 2.41388 µm | 2.38 | radial FFT peak of the saturated Schnakenberg activator on a periodic 2.4 µm-calibrated grid |
| dispersion_pearson_r | 0.99323 1 | 1.0 | Pearson r of analytic σ(k) vs growth rate measured from the PDE, nine wavenumbers |
| dispersion_rmse | 0.02405 1/time | 0.0 | RMSE of the same σ(k) pairs |
| specimen_plate_foreground_fraction | 0.3806 1 | 0.5455 | border-contrast segmentation of the canonical plate; reference is the Wikimedia habitus |
| specimen_plate_coherency | 0.8155 1 | 0.5838 | structure-tensor coherency of the canonical plate vs the habitus drawing |
| specimen_plate_circularity | 0.0538 1 | 0.264 | 4πA/P² of the plate foreground; habitus is the reference |
| segment_ks_activator_vs_plate | 0.858 1 | 0.0 | two-sample KS between normalized activator intensities and plate luminance |
| ablation_to_pattern_std_ratio | 0.0 1 | 1.0 | std(u) with Dv=Du divided by std(u) at R=80; reference 1 would mean the control failed to flatten |
| chronoseries_predecessor_wavelength_um | 3.23644 µm | 2.37959 | D. miranda λ = λ★ * 3237/2380 at 0 Ma; reference is the focal 2.4 µm wavelength |
| chronoseries_successor_wavelength_um | 1.94766 µm | 2.37959 | D. novamexicana λ = λ★ * 1948/2380 at 0 Ma; reference is the focal wavelength |
| neumann_stripe_wavelength_um | 2.41388 µm | 2.38 | no-flux radial-FFT wavelength; the periodic small-grid run falls in the same bin, so the reference is the catalog 2.38 µm rather than that identical bin |
| neumann_pattern_std | 0.99918 1 | 1.01072 | saturated activator std on the no-flux patch versus the periodic patch at the same diffusivity |
| predecessor_fft_wavelength_um | 3.28251 µm | 3.23644 | saturated FFT at the D. miranda diffusivity scale; reference is the analytic square-law wavelength |
| successor_fft_wavelength_um | 1.97539 µm | 1.94766 | saturated FFT at the D. novamexicana diffusivity scale; reference is the analytic square-law wavelength |

## Environment at checkpoint


Loop commit that adds the slot 3194 files. Origin URL in this checkout carries an access token, so the blob-locus check is skipped with a note. ORCID is the publisher already used by morphoeval/kg/2026-10-19-turing-audit because ~/.nanopub/profile.yml is absent. Signing was not performed.
