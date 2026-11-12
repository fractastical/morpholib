# MorphoEval report — audit `2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `d71d2f0b1c64d11b069392db32c90726b8fa5d72` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dionaea muscipula Turing wavenumber at 24 µm (pass 1, unfitted field) (`models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py`) | 76 | 46 | 40 | 34 | 86 | 70 | 78 | 64.0 | 41.5 | 78.5 | 59.4 | **59.4** | partially faithful |

## Findings and remediation

### Dionaea muscipula Turing wavenumber at 24 µm (pass 1, unfitted field)  —  overall 59.4 (partially faithful)

Pass 1 runs Schnakenberg kinetics on a 192 µm by 144 µm adaxial patch with Du and Dv chosen so the linear wavelength is 24 µm, and writes a full-bleed red activator GIF (12 frames, 640×480). Computer-vision similarity of that GIF to the CC BY-SA trap photograph is 73.0. The saturated Fourier wavelength on the patch is 26.63 µm. No diffusivity sweep, no equal-diffusion ablation, and no specimen-versus-render segmentation are in the emitted metrics. The only saved comparison is λ = 24 µm against λ = 24 µm.

- **Model fidelity to source: 76** — Emit 1 integrates the Schnakenberg PDE whose linearisation is the Turing dispersion relation, with Du and Dv set so λ_c = 24 µm. The numeric σ(k) check is in the file but is not part of the pass-1 output.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L115–158; `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L264–309
- **Morphological realism / specimen match: 46** — The pass-1 GIF is a red spot field. CV similarity to the trap photograph is 73.0 because both are red and filled, but the lobes, midrib and marginal cilia are absent.  
  evidence: `docs/media/dionaea_muscipula_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L472–489
- **Parameter expressivity: 40** — a, b and Dv/Du are fixed constants. Emit 1 does not vary them, so the figure cannot show that wavelength tracks the diffusivity scale.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L816–843
- **Morphometric validation: 34** — The only measurement is the analytic wavelength 24.0 µm against the reference 24.0 µm. That comparison is true by construction and carries no specimen residual, RMSE, or KS statistic.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L593–606; `datasets/dionaea_muscipula_turing_wavelength_s0/metrics.json` L0–0
- **Source and data traceability: 86** — Crossref-verified DOIs for Turing, Schnakenberg, Kondo & Miura, and Forterre are in the bib file. The trap photograph is the Wikimedia Commons file under CC BY-SA 2.5, SHA-256 pinned.  
  evidence: `datasets/dionaea_muscipula_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/dionaea_muscipula_turing_wavelength_s0/README.md` L1–20
- **Code-to-model traceability: 70** — The module docstring writes the PDE, the steady state and the dispersion formula, and sigma_analytic implements that formula. Pass 1 does not yet point a plotted numeric curve back at the formula.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L1–48; `morphoeval/lean/loops/Loop_3169_dionaea_muscipula_turing_wavelength_s0.lean` L15–32
- **Reproducibility package: 78** — --self-test checks λ_c, the sign of the Turing margin, and the pinned SHA-256 inputs. Seed 3169. Emit 1 itself does not re-check the dispersion Pearson statistic.  
  evidence: `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` L927–966; `datasets/dionaea_muscipula_turing_wavelength_s0/dionaea_muscipula_turing_wavelength_s0_reference.json` L1–40

Remediation:
- {'id': 'R1', 'item': 'Add a diffusivity-scale sweep and show that the PDE wavelength tracks 24*sqrt(s).'}
- {'id': 'R2', 'item': 'Add the equal-diffusion ablation (Dv = Du) as a negative control with σ_max < 0.'}
- {'id': 'R3', 'item': 'Segment the real specimen and the render (foreground, coherency, circularity, KS). Values must not be copies of the references.'}
- {'id': 'R4', 'item': 'Report Pearson r and RMSE for σ_PDE(k) against σ_analytic(k), and keep r > 0.98.'}
- {'id': 'R5', 'item': "Rebuild the GIF so the last frame carries the trap's cilia and lobe geometry; pass-1 CV is 73.0 and image fitness stays under 70 at these audit subscores."}
- {'id': 'R6', 'item': 'No second assessor. Left open: this loop is one agent.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dionaea muscipula Turing wavenumber at 24 µm (pass 1, unfitted field)
- **spec_fingerprint** (semantic model + inputs): `8e108788aba01a68d605b3e6f5b55f34806823e95c468fc791b29029952be4e4`
- **input_dataset_sha256**: `datasets/dionaea_muscipula_turing_wavelength_s0/README.md` (`3f406da6d038e1ef…`), `datasets/dionaea_muscipula_turing_wavelength_s0/dionaea_muscipula_trap_lobe_specimen.jpg` (`6e5b87a782e373b3…`), `datasets/dionaea_muscipula_turing_wavelength_s0/dionaea_muscipula_turing_wavelength_s0_reference.json` (`7976efe0424723e6…`), `datasets/dionaea_muscipula_turing_wavelength_s0/paper_citations.bib` (`d5d325187b9bbe7d…`), `datasets/dionaea_muscipula_turing_wavelength_s0/scout_manifest.json` (`732331a6145be952…`)
- `datasets/dionaea_muscipula_turing_wavelength_s0/metrics.json` — artifact_sha256 `cb633255c4f5a7eb…`, 851 B, 
- `docs/media/dionaea_muscipula_turing_wavelength_s0_growth.gif` — artifact_sha256 `3c8a8601e273489a…`, 1,467,858 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytic critical wavelength | 24.0 µm | 24.0 | 2*pi/k_c from the Schnakenberg Jacobian at Dv/Du = 18 *(value equals reference by calibration; self-referential)* |

## Environment at checkpoint


Pass-1 evaluation of --emit 1. The GIF bytes hashed in the pass-1 measurement sidecar were replaced by the pass-2 render in this same commit; the sidecar keeps the pass-1 hashes.
