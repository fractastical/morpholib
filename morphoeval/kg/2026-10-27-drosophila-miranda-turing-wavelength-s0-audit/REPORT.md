# MorphoEval report — audit `2026-10-27-drosophila-miranda-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila miranda detail-scale Turing wavelength (1.9 µm, 0 Ma) (`models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py`) | 88 | 82 | 64 | 70 | 90 | 84 | 80 | 80.0 | 77.5 | 84.4 | 80.3 | **80.3** | faithful |

## Findings and remediation

### Drosophila miranda detail-scale Turing wavelength (1.9 µm, 0 Ma)  —  overall 80.3 (faithful)

Pass 1 executes the Schnakenberg field (FFT wavelength 1.864 µm) and a 12-frame 640×480 GIF at fixed overlay depth 0.22. The pass-1 measurement set stops at wavelength and trace. Sweep, ablation, specimen segmentation, and dispersion Pearson/RMSE are the remediation items.

- **Model fidelity to source: 88** — The pass-1 run integrates Schnakenberg kinetics with a pseudospectral Laplacian and the Turing linearisation (trace -0.16, det 1, Turing margin positive). Kinetics are the published Schnakenberg form; the diffusivities are solved for this slot rather than copied from a miranda table.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L72–98; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L145–194; `morphoeval/lean/loops/Loop_3193_drosophila_miranda_turing_wavelength_s0.lean` L11–45
- **Morphological realism / specimen match: 82** — The GIF is 12 frames at 640×480. The last frame is the gallery plate modulated by the activator. Pass 1 uses a fixed depth and does not yet score plate segmentation, so specimen-match stays below the pass-2 value. The pure field is a spot lattice, not a portrait, and the plate is not an EM.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L365–386; `docs/media/drosophila_miranda_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L507–508
- **Parameter expressivity: 64** — diffusivity_sweep exists, but the pass-1 figure is a single row and measure_morphoeval returns before the sweep correlation. Expressivity is not credited until that sweep is on the figure and in the sidecar.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L225–238; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L430–431
- **Morphometric validation: 70** — Pass 1 reports FFT wavelength 1.864 µm against the 1.9 µm label and the trace against a stability prior. Pearson r, RMSE, KS, ablation ratio, and plate morphometrics are computed in memory and then dropped.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L196–209; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L487–508
- **Source and data traceability: 90** — Turing 1952, Schnakenberg 1979, Shoup 1966, and Bachtrog 2013 are cited with DOIs resolved on OpenAlex. The specimen SHA-256 is pinned. Slot 3193 is outside campaign_100, so scout_literature.py --slot 3193 cannot load a campaign row; the manifest records the DOI lookup that replaced it.  
  evidence: `datasets/drosophila_miranda_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_miranda_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 84** — Dispersion, kinetics, ablation, and the chronoseries each have a Lean theorem and a Python function. Pass 1 does not yet point the figure at the dispersion curve, so the trace from equation to pixel is incomplete.  
  evidence: `morphoeval/math_kg_loops/delphi10000-drosophila-miranda-turing-wavelength-s0.yaml` L40–80; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L40–66
- **Reproducibility package: 80** — --self-test checks the Lean integer witnesses, wavelength, Pearson r > 0.98, ablation, and sweep monotonicity. Pass 1's public measurement sidecar does not yet carry those checks, so the package is not fully evidenced.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L608–650; `datasets/drosophila_miranda_turing_wavelength_s0/README.md` L1–40

Remediation:
- Add the D_u parameter sweep (λ versus √D_u) to the figure and to measure_morphoeval.
- Add the equal-diffusion negative control and report the activator std ratio.
- Segment the real gallery plate (foreground, coherency, circularity, orientation KS) and emit non-circular specimen metrics.
- Report dispersion goodness-of-fit (Pearson r, RMSE) between continuum σ(k) and the explicit-Euler symbol.
- Choose overlay depth by a CV sweep instead of the fixed pass-1 depth 0.22.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `parameter_sweep` → `parameter-expressivity`: Pass 1 does not score the five-point D_u sweep.
- **[OPEN · P=1.5]** `morphometric_validation` → `morphometric-validation`: Pass 1 measurement set has no Pearson r, RMSE, KS, or ablation ratio.
- **[OPEN · P=1.3]** `specimen_comparison` → `specimen-match`: Pass 1 does not emit plate segmentation metrics.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila miranda detail-scale Turing wavelength (1.9 µm, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `d72df4485d107fdd15f1f53825f9d2fc9fb6708a151467f2a51b6d54ce339010`
- **input_dataset_sha256**: `datasets/drosophila_miranda_turing_wavelength_s0/Drosophila_miranda_specimen.jpg` (`fc43a0abd39eea68…`), `datasets/drosophila_miranda_turing_wavelength_s0/README.md` (`144d0cc17c40ba2e…`), `datasets/drosophila_miranda_turing_wavelength_s0/drosophila_miranda_turing_wavelength_s0_reference.json` (`03f46b7405f11c6b…`), `datasets/drosophila_miranda_turing_wavelength_s0/paper_citations.bib` (`ba21cc5176da73b5…`), `datasets/drosophila_miranda_turing_wavelength_s0/scout_manifest.json` (`e67dc1eb5bd042c5…`)
- `docs/media/drosophila_miranda_turing_wavelength_s0_comparison.png` — artifact_sha256 `d7278ebb5fca3418…`, 576,855 B, 1464×912 px
- `docs/media/drosophila_miranda_turing_wavelength_s0_growth.gif` — artifact_sha256 `5a6e72f6e8037c60…`, 3,156,821 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| miranda detail-scale selected wavelength | 1.864 µm | 1.9 | radial FFT peak of the saturated Schnakenberg activator on a 22.368 µm periodic field *(FFT wavelength 1.8640 µm versus the 1.9 µm label (target 1.864 µm))* |
| Turing trace at the Schnakenberg steady state | -0.16 1/time | -0.2 | tr J = f_u + g_v at (u0, v0) = (1, 0.92) *(trace is negative, so the uniform state is stable without diffusion)* |

## Environment at checkpoint


Loop files are added on top of 0c76d15d. Pass 1 figure was rendered with --pass 1 and then replaced by the pass-2 figure; this audit scores the pass-1 measurement path.
