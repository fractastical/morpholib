# MorphoEval report — audit `2026-10-27-drosophila-serrata-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-serrata-turing-wavelength-s0-audit -> 2026-10-27-drosophila-serrata-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila serrata Schnakenberg Turing wavelength at 2.123 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py`) | 91 | 86 | 88 | 90 | 92 | 88 | 90 | 90.0 | 87.5 | 90.1 | 89.0 | **89.0** | faithful |

## Findings and remediation

### Drosophila serrata Schnakenberg Turing wavelength at 2.123 µm (Holocene, 0 Ma)  —  overall 89.0 (faithful)

Pass 2 keeps the Pass 1 Schnakenberg model (Pearson r = 0.99998, λ = 2134/2123/2102 nm, equal-diffusion σ_max = -0.231, GIF CV similarity 94.1). It adds k_max = argmax σ(k), with k_max/k_c = 0.975, and reports the nonlinear FFT peak relative to that mode. The comparison figure states the gallery plate's 100 µm bar is not the 2.123 µm model scale.

- **Model fidelity to source: 91** — Same Schnakenberg linearization as Pass 1 (RK2 vs analytic r = 0.99998). Pass 2 separates band-center k_c from the fastest mode k_max (k_max/k_c = 0.975) and checks the nonlinear FFT peak against k_max, which is the comparison the dispersion relation actually predicts off onset.  
  evidence: `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L137–179; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L239–244; `morphoeval/lean/loops/Loop_3203_drosophila_serrata_turing_wavelength_s0.lean` L43–46
- **Morphological realism / specimen match: 86** — GIF CV similarity remains 94.1 against the gallery plate (12 frames, 640×520). The comparison figure now says the plate's printed bar is 100 µm and the model wavelength is 2.123 µm, so the match is a texture match to an ocellus-like field. It is still not a voucher micrograph at the detail scale, which is why this stays below 90.  
  evidence: `docs/media/drosophila_serrata_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L441–449; `docs/media/drosophila_serrata_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — λ(δ) still shortens across the seven-point sweep and across sechellia, serrata, silvestris. The new k_max marker moves when δ moves, because the fastest mode tracks the band.  
  evidence: `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L273–336; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L420–426
- **Morphometric validation: 90** — Pass 1 metrics stand (r = 0.99998 vs 1, RMSE 0.00169 vs 0, orientation KS 0.071 vs 0.5, sechellia−serrata gap 11.14 nm vs 0). Pass 2 adds k_measured/k_max, reference 1, value near 0.976: the residual is one diagonal lattice step on the 64² grid (Δ|k| = k_c/8), not a second wavelength. The CHC cline's nanometre span remains a reconstruction.  
  evidence: `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L548–566; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L455–465
- **Source and data traceability: 92** — Unchanged from Pass 1. Crossref-checked DOIs, SHA-256 pins, and an explicit negative search of Wikimedia, iNaturalist, and GBIF for a single-animal CC photograph.  
  evidence: `datasets/drosophila_serrata_turing_wavelength_s0/paper_citations.bib` L1–87; `datasets/drosophila_serrata_turing_wavelength_s0/drosophila_serrata_turing_wavelength_s0_reference.json` L1–30
- **Code-to-model traceability: 88** — k_max_of is a separate function from kc_of, and the figure caption names both lines. Lean integers are still asserted in --self-test, which now also gates k/k_max.  
  evidence: `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L239–244; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L625–660
- **Reproducibility package: 90** — --self-test passed 18 checks after the Pass 2 edit, including dataset hashes, r > 0.98, r < 1, equal-diffusion stability, GIF frames and resolution, and CV similarity >= 58. Seed 3203. No nanopub profile is present on this machine, so the graphs are unsigned.  
  evidence: `datasets/drosophila_serrata_turing_wavelength_s0/README.md` L40–55; `models/2026-delphi10000-drosophila-serrata-turing-wavelength-s0.py` L64–78

Remediation:
- R2 remains open: a second assessor from another provider did not score this loop. Scores are single-agent.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=0.0]** `independent_assessor` → `model-fidelity`: Second assessor from another provider has not scored this script.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila serrata Schnakenberg Turing wavelength at 2.123 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `a3c4b3d65dab3b0ebfbf973810a2da1a980cfa0eca08496d6d9c3676418453c9`
- `docs/media/drosophila_serrata_turing_wavelength_s0_comparison.png` — artifact_sha256 `69278c46b9775589…`, 387,124 B, 1344×1008 px
- `docs/media/drosophila_serrata_turing_wavelength_s0_growth.gif` — artifact_sha256 `2fbf667000d2f0d9…`, 3,313,138 B, 640×520 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| RK2 vs analytic Turing dispersion Pearson r | 0.99998 Pearson r | 1.0 | Pearson correlation of σ(k) from the closed-form eigenvalue and from RK2 integration of the linearized mode, 12 wavenumbers *(RMSE = 0.00168968)* |
| dispersion RMSE between RK2 and analytic growth rate | 0.00169 1/time | 0.0 | RMSE of σ_RK2(k) against σ_analytic(k) on k in [0.55, 1.45] k_c |
| serrata nonlinear patch measured-to-critical wavenumber ratio | 0.952 k_measured / k_c | 1.0 | FFT peak of the periodic Schnakenberg patch divided by the band-center k_c *(equal-diffusion control std = 0.0000 vs pattern std = 0.0600)* |
| serrata FFT peak over fastest linear mode k_max | 0.9761 k_measured / k_max | 1.0 | Same FFT peak divided by k_max = argmax σ(k). The 64² lattice quantizes |k| in steps of k_c/8. *(k_max/k_c = 0.9753)* |
| equal-diffusion ablation maximum growth rate | -0.23102 1/time | 0.0 | max_k σ(k) with Dv = Du (negative control) *(control vs Turing: equal diffusion is linearly stable)* |
| specimen image structure-tensor coherency | 0.775 coherency | 0.1177 | verify_image_fitness.compute_specimen_cv_signature on the pinned gallery plate *(specimen foreground fraction = 0.3762)* |
| specimen vs simulation orientation KS distance | 0.0714 KS | 0.5 | Kolmogorov-Smirnov distance between 12-bin Sobel orientation histograms of the specimen crop and the GIF last-frame crop |
| chronoseries wavelength gap sechellia minus serrata | 11.14 nm | 0.0 | analytic λ(δ) at fixed Du; predecessor factor 1.025× onset, serrata 1.085× onset *(silvestris-serrata = -20.590 nm; neighbor k/k_c sechellia 0.981, silvestris 0.901)* |
| east-coast reconstructed CHC-cline wavelength span | -4.532 nm | 11.0 | λ(δ(ξ)) with δ(ξ) = δ_serrata (1 + 0.012 ξ), ξ from Cairns to Melbourne *(direction follows Frentiu & Chenoweth 2010; nanometre values are reconstructed)* |
| GIF specimen-match cv_sim_similarity | 94.1 score | 58.0 | Sobel orientation, radial FFT and structure-tensor coherency via compute_sim_vs_specimen_cv_similarity *(n_frames = 12, size = 640x520)* |

## Environment at checkpoint


Pass 2 working tree. New slot files are uncommitted. Built with --no-verify-loci.
