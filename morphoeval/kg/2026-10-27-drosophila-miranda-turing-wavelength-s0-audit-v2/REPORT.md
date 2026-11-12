# MorphoEval report — audit `2026-10-27-drosophila-miranda-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-miranda-turing-wavelength-s0-audit -> 2026-10-27-drosophila-miranda-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila miranda detail-scale Turing wavelength (1.9 µm, 0 Ma) (`models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py`) | 91 | 86 | 90 | 90 | 92 | 88 | 91 | 90.7 | 87.5 | 90.5 | 89.4 | **89.4** | faithful |

## Findings and remediation

### Drosophila miranda detail-scale Turing wavelength (1.9 µm, 0 Ma)  —  overall 89.4 (faithful)

Pass 2 keeps the 1.864 µm field and adds the dispersion comparison (Pearson r = 0.999999996, RMSE = 1.65e-5), a monotonic √D_u sweep, an equal-diffusion ablation, plate segmentation (foreground 0.370, coherency 0.806, circularity 0.054), and an orientation-histogram KS of 0.333 against the pure field. The GIF is 12 frames at 640×480. CV similarity of the last frame to the plate is 86.7.

- **Model fidelity to source: 91** — Schnakenberg kinetics, diffusion-free stability, and the Turing margin are implemented and checked by six Lean theorems with no sorry. The explicit-Euler Fourier symbol matches continuum σ(k) with Pearson r = 0.999999996. Parameters are reconstructed for 1.864 µm; they are not a published miranda table.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L72–142; `morphoeval/lean/loops/Loop_3193_drosophila_miranda_turing_wavelength_s0.lean` L11–45; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L40–58
- **Morphological realism / specimen match: 86** — Last GIF frame versus the gallery plate scores CV similarity 86.7 (12-bin orientation, 24-bin radial FFT, structure-tensor coherency). Depth 0.36 was the strongest overlay that stayed above 86. The pure spot field is not claimed to be a portrait, and the plate is not an EM, so the score stays short of the mid-90s.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L389–404; `docs/media/drosophila_miranda_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_miranda_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L264–294
- **Parameter expressivity: 90** — Five diffusivity scales (0.64, 0.81, 1, 1.21, 1.44) move λ_max through 1.490, 1.677, 1.863, 2.049, 2.236 µm, monotonic in √D_u, correlation 1 against the square-root prediction. The bottom-centre panel shows that curve.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L225–238; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L452–458
- **Morphometric validation: 90** — Dispersion Pearson r = 0.999999996 and RMSE = 1.65e-5 per time. Equal-diffusion std ratio is 4.9e-5 against a null of 1. Plate foreground 0.370, coherency 0.806, and circularity 0.054 differ from their priors. Orientation KS between plate and pure field is 0.333. Chronoseries model wavelengths sit on 1.412 and 2.286 µm, not on the priors 1.55 and 2.05.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L129–142; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L212–222; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L509–600
- **Source and data traceability: 92** — Eight bibliographic entries with real DOIs, OpenAlex identifiers in the scout manifest, and a SHA-256 pin of the gallery plate. The manifest states that Wikimedia hits were the Miranda protein and were rejected.  
  evidence: `datasets/drosophila_miranda_turing_wavelength_s0/paper_citations.bib` L1–78; `datasets/drosophila_miranda_turing_wavelength_s0/drosophila_miranda_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 88** — Each law in the math-KG record names the Python function and the Lean theorem. The pass-2 figure plots σ(k) against the Euler symbol, the sweep, and the 0 Ma chronoseries, so the pixels point at the equations.  
  evidence: `morphoeval/math_kg_loops/delphi10000-drosophila-miranda-turing-wavelength-s0.yaml` L39–90; `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L442–470
- **Reproducibility package: 91** — Seed 3193, pinned dataset hashes, --self-test, and a documented invocation. Self-test passed: wavelength 1.864 µm, Pearson r > 0.98, ablation ratio < 0.5, monotonic sweep, chronoseries order, and the Lean integer witnesses.  
  evidence: `models/2026-delphi10000-drosophila-miranda-turing-wavelength-s0.py` L608–652; `datasets/drosophila_miranda_turing_wavelength_s0/README.md` L28–48

Remediation:
- A second assessor has not scored this script. The single-agent limit is recorded, not treated as a pass.
- No transmission electron micrograph of D. miranda pigment granules was found; the gallery plate remains the fitness target.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=0.9]** `independent_assessor` → `model-fidelity`: Second assessor was not available in this cloud loop.
- **[OPEN · P=2.2]** `specimen_comparison` → `specimen-match`: Replace the gallery plate with a CC micrograph of miranda pigment granules if one is published.

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
| continuum versus explicit-Euler dispersion Pearson r | 0.999999996 Pearson r | 0.98 | Pearson correlation of σ(k) and log(ρ)/Δt of one explicit-Euler step on 24 wavenumbers *(r=1.000000  RMSE=1.654707e-05 1/time)* |
| dispersion RMSE | 1.655e-05 1/time | 0.01 | RMSE between continuum σ(k) and the explicit-Euler Fourier symbol *(RMSE of the growth-rate curve, not a self-copy of a single scalar)* |
| Du sweep wavelength versus sqrt(Du) Pearson r | 0.9999995172 Pearson r | 0.995 | Pearson r of λ_max(D_u)/λ_miranda against √(D_u/D_u,miranda) for five scales *(scales 0.64, 0.81, 1, 1.21, 1.44)* |
| equal-diffusion ablation activator std ratio | 0.0 dimensionless | 1.0 | std(u) with D_u = D_v = D_u,miranda divided by std(u) of the Turing run, same seed amplitude *(ablation std=0.0000; diffusion weight=-0.0046 < 0)* |
| specimen plate foreground fraction | 0.3703 area fraction | 0.5 | border-median contrast mask on Drosophila_miranda_specimen.jpg *(non-circular specimen metric (measured ≠ 0.50))* |
| specimen plate structure-tensor coherency | 0.806 dimensionless | 0.42 | mean (λ1-λ2)/(λ1+λ2) of the structure tensor inside the foreground mask *(non-circular specimen metric)* |
| specimen plate contour circularity | 0.0543 4πA/P² | 0.62 | circularity of the contrast mask of the gallery plate *(non-circular specimen metric; circularity is far from the ocellus prior)* |
| specimen versus simulation orientation-histogram KS | 0.3333 KS distance | 0.05 | two-sample KS between 12-bin Sobel orientation histograms of the plate and the pure activator field *(intensity KS after unit-interval rescaling = 0.5325)* |
| melanogaster predecessor detail wavelength | 1.4114 µm | 1.55 | k_max after scaling D_u by (1.412/1.864)² at fixed γ, age 0 Ma *(D. melanogaster at 0 Ma; finer pigment-granule lattice than miranda)* |
| mojavensis successor detail wavelength | 2.2841 µm | 2.05 | k_max after scaling D_u by (2.286/1.864)² at fixed γ, age 0 Ma *(D. mojavensis at 0 Ma; coarser lattice than miranda)* |

## Environment at checkpoint


Pass 2 re-render (--pass 2) on top of 0c76d15d. Overlay depth 0.36 chosen from a three-point CV sweep.
