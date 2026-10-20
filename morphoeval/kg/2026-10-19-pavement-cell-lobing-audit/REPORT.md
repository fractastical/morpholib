# MorphoEval report — audit `2026-10-19-pavement-cell-lobing-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/39

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sapala et al. (2018) & Fu et al. (2005) ROP2-ROP6 interdigitated puzzle-shaped pavement cell lobing and Largest Empty Circle (LEC) turgor stress scaling (`2017-runions-pavement-cell-lobing.py`) | 54 | 45 | 52 | 52 | 68 | 58 | 54 | 53.3 | 47.6 | 59.7 | 53.0 | **53.0** | partially faithful |

## Findings and remediation

### Sapala et al. (2018) & Fu et al. (2005) ROP2-ROP6 interdigitated puzzle-shaped pavement cell lobing and Largest Empty Circle (LEC) turgor stress scaling  —  overall 53.0 (partially faithful)

Pass 1 (v1) baseline evaluation of `2017-runions-pavement-cell-lobing.py` (weighted score: 53.6/100). The Pass 1 script evaluates only an empirical curve fit for Largest Empty Circle (LEC) radius and a static cosine contour without solving coupled ROP2-ROP6 anticlinal boundary reaction-diffusion PDEs, Sapala et al. (2018) spring-remodeling cell wall mechanics, exact Euclidean distance-transform LEC radii, ktn1 / rop2 rop4 mutant and monocot parallel-file comparisons, or CV segmentation of the vendored Arabidopsis leaf epidermis micrograph.

- **Model fidelity to source: 54** — Computes an analytical curve for R_LEC(A) and a static 6-lobe cosine curve (L25-45), but omits the coupled ROP2-RIC4 / ROP6-RIC1-KTN1 mutual-inhibition reaction-diffusion PDEs across shared anticlinal walls, Sapala et al. (2018) mechanical spring-remodeling lobe growth, and distance-transform Largest Empty Circle (LEC) evaluation on evolving cell polygons.  
  evidence: `2017-runions-pavement-cell-lobing.py` L25–45
- **Morphological realism / specimen match: 45** — Vendors `datasets/pavement_cell_lobing/arabidopsis_pavement_cell_specimen.jpg` on disk, but Pass 1 does not load, segment, or measure cell contours, lobeyness, solidity, or LEC radii from the Arabidopsis leaf epidermis micrograph.  
  evidence: `2017-runions-pavement-cell-lobing.py` L68–92
- **Parameter expressivity: 52** — Evaluates a single static parameter point without sweeping cell area growth across genotypes (wild-type Col-0, rop2 rop4 RNAi, ktn1/ric1 mutant, and monocot narrow parallel files) or exposing CLI parameter overrides.  
  evidence: `2017-runions-pavement-cell-lobing.py` L25–45
- **Morphometric validation: 52** — Exposes only 2 analytical ratios in `measure_morphoeval()` without PDE cross-correlation metrics, statistical p-values, lobeyness/solidity genotype comparisons, or empirical specimen segmentation metrics.  
  evidence: `2017-runions-pavement-cell-lobing.py` L48–65
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `pavement_cell_lobing_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/pavement_cell_lobing/pavement_cell_lobing_reference.json` L1–145
- **Code-to-model traceability: 58** — Header cites Sapala et al. (2018) and Fu et al. (2005), but lacks explicit equation-to-code-line and Lean 4 theorem mappings.  
  evidence: `2017-runions-pavement-cell-lobing.py` L1–10
- **Reproducibility package: 54** — Runs deterministically to generate PNG and GIF outputs, but lacks a comprehensive 8-check `--self-test` suite, SHA-256 dataset verification, and mutant negative controls.  
  evidence: `2017-runions-pavement-cell-lobing.py` L108–120

Remediation:
- Implement coupled ROP2-RIC4 and ROP6-RIC1-KTN1 mutual-inhibition cortical reaction-diffusion PDEs across shared anticlinal walls (Fu et al. 2005 Cell; Lin et al. 2013 Curr Biol), verifying intracortical anticorrelation and trans-wall lobe-neck antiphase complementarity.
- Implement the Sapala et al. (2018) eLife mechanical spring-remodeling puzzle cell growth solver across cell area A in [250, 4500] um^2 with exact Euclidean distance-transform Largest Empty Circle (R_LEC) computation across wild-type Col-0, rop2 rop4 RNAi, ktn1/ric1 mutants, and parallel-file monocot epidermis.
- Implement automated computer-vision segmentation of the SHA-256-locked Arabidopsis thaliana leaf epidermis micrograph (`arabidopsis_pavement_cell_specimen.jpg`) to extract interlocking pavement cell contours, lobeyness L = P_perim / P_hull, solidity S = A / A_hull, lobe count, and inscribed R_LEC circles.
- Add runtime SHA-256 dataset verification, 9 reference-anchored morphometric measurements with statistical p-values and negative controls, and an 8-point deterministic `--self-test` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sapala et al. (2018) & Fu et al. (2005) ROP2-ROP6 interdigitated puzzle-shaped pavement cell lobing and Largest Empty Circle (LEC) turgor stress scaling
- **spec_fingerprint** (semantic model + inputs): `972460a5749d7d30868f5de9c61bcfd278fa143f89d629090a75b7aa4ebdc287`
- **input_dataset_sha256**: `datasets/pavement_cell_lobing/arabidopsis_pavement_cell_specimen.jpg` (`4b7c1062f6ebfd6e…`), `datasets/pavement_cell_lobing/paper_citations.bib` (`e5d0ea8ec26365c9…`), `datasets/pavement_cell_lobing/pavement_cell_lobing_reference.json` (`de4e6daab5eae4a0…`), `datasets/pavement_cell_lobing/scout_manifest.json` (`72c917ec10ac1a55…`)
- `docs/media/pavement_cell_lobing_comparison.png` — artifact_sha256 `8e3b33e9aa4c563a…`, 74,721 B, 1260×540 px

| measurement | value | reference | method |
|---|---:|---|---|
| wt_mature_lec_radius_at_4500um2 | 11.444 um | 11.4 | Analytical asymptotic neck radius R_LEC(A) |
| isotropic_vs_puzzle_stress_ratio_at_4500um2 | 3.307 ratio | 3.32 | Ratio of unlobed isotropic circle stress to puzzle-lobed cell stress at A = 4500 um^2 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2017-runions-pavement-cell-lobing.py` prior to Step 12 closed-loop remediation.
