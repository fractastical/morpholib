# MorphoEval report — audit `2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg most-unstable wavelength for D. mojavensis epicuticle at 2.4 µm (0 Ma) (`models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py`) | 86 | 80 | 84 | 82 | 90 | 82 | 86 | 85.3 | 80.8 | 86.3 | 83.8 | **83.8** | faithful |

## Findings and remediation

### Schnakenberg most-unstable wavelength for D. mojavensis epicuticle at 2.4 µm (0 Ma)  —  overall 83.8 (faithful)

Pass 1. Schnakenberg kinetics a=0.37, b=0.71, γ=4, R=Dv/Du=80. Diffusivities are scaled so argmax σ(k) is 2.380 µm (catalog 2.4 µm = 2.38e-06 m). PDE-measured σ(k) vs analytic σ(k): Pearson r=0.99323, RMSE=0.02405 (n=9). Saturated radial-FFT wavelength 2.414 µm. Equal-diffusion ablation std ratio 2.5e-6. Analytic λ★(s) is monotone on s in {0.67, 0.85, 1, 1.40, 1.85}. Chronoseries at 0 Ma, λ ∝ √D: D. miranda 3.236 µm, D. mojavensis 2.380 µm, D. novamexicana 1.948 µm. GIF 14 frames, 640×480, CV similarity 87.1 against the canonical gallery plate. Pattern weight 0.18 chosen from {0.06, 0.10, 0.14, 0.18}.

- **Model fidelity to source: 86** — The script integrates Schnakenberg kinetics and the Turing linearisation, and it separates the Murray leading-order k_c from the true argmax of σ(k). Absolute Du, Dv are calibrated to the catalog length 2.38e-06 m rather than taken from a measured epicuticular diffusivity. Periodic boundaries only.  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L88–111; `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L122–156; `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L249–296
- **Morphological realism / specimen match: 80** — The 14-frame 640×480 GIF ends on the canonical plate with a 0.18 activator modulation. Computer-vision similarity to that plate is 87.1. The plate is a stylized ochre spiral with a 100 µm bar, not a 2.4 µm micrograph, so the match is to the repository specimen asset rather than to a published SEM. The public-domain habitus (225×99) is segmented but not the fitness target.  
  evidence: `docs/media/drosophila_mojavensis_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_mojavensis_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L382–414
- **Parameter expressivity: 84** — λ★(s) increases monotonically for diffusivity scales 0.67, 0.85, 1, 1.40, 1.85 at fixed Jacobian, which is the claimed square-root control. The saturated FFT wavelength is measured only at s=1 (2.414 µm vs λ★ 2.380 µm). The equal-diffusion ablation collapses the amplitude (std ratio 2.5e-6).  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L322–332; `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L298–320; `datasets/drosophila_mojavensis_turing_wavelength_s0/metrics.json` L29–120
- **Morphometric validation: 82** — Pearson r=0.99323 and RMSE=0.02405 between analytic and PDE growth rates across nine wavenumbers. Plate foreground fraction 0.3806 vs habitus 0.5455, coherency 0.8155 vs 0.5838, circularity 0.0538 vs 0.264, KS(activator, plate)=0.858, all with value ≠ reference. Nonlinear wavelength error is 1.4% (2.414 vs 2.380). No second boundary condition yet.  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L205–224; `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L474–508; `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L226–247
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing 1952, Schnakenberg 1979, Matzkin 2006, Matzkin 2012, Allan & Matzkin 2019, and the Drosophila 12 Genomes paper. Specimen SHA-256 pins match the files on disk. The sibling slot's DOI 10.1534/genetics.106.059444 was checked and rejected (it is Hill & Otto 2007).  
  evidence: `datasets/drosophila_mojavensis_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/drosophila_mojavensis_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/drosophila_mojavensis_turing_wavelength_s0/drosophila_mojavensis_turing_wavelength_s0_reference.json` L1–30
- **Code-to-model traceability: 82** — Jacobian, dispersion, calibration, integrator, ablation, and chronoseries square law are each a named function with the equation in the docstring. The GIF weight search is a render fit, documented in metrics.json, not a new biological parameter.  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3194_drosophila_mojavensis_turing_wavelength_s0.lean` L14–42; `morphoeval/math_kg_loops/delphi10000-drosophila-mojavensis-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 86** — --self-test passed: r>0.98, ablation collapse, monotone sweep, GIF ≥10 frames and ≥480 px, CV≥58, and every morphometric value differs from its reference. Seed 3194. Deterministic explicit Euler. No pinned conda lock for this slot beyond the repository requirements.txt.  
  evidence: `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` L815–854; `datasets/drosophila_mojavensis_turing_wavelength_s0/README.md` L1–28

Remediation:
- {'id': 'R1', 'item': 'Add a no-flux (Neumann) boundary control and report its saturated wavelength next to the periodic λ.'}
- {'id': 'R2', 'item': 'Measure the saturated FFT wavelength at the predecessor and successor diffusivity scales, not only the analytic λ★(s).'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `parameter_sweep` → `parameter-expressivity`: Saturated FFT wavelength at s≠1 is not yet measured.
- **[OPEN · P=1.1]** `model_equation` → `model-fidelity`: Neumann boundary has not been integrated; only the periodic Laplacian is exercised.

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

## Environment at checkpoint


Pass 1 working tree on top of 0c76d15. New slot files are not in that commit; build uses --no-verify-loci. Commit pin is updated when the loop commit exists.
