# MorphoEval report — audit `2026-10-19-roman-esco-fractal-inflorescence-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/40

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Azpeitia et al. (2021) & Prusinkiewicz et al. (2007) TFL1-LFY-AP1/CAL-SOC1 floral GRN perturbation and accelerating plastochron Romanesco fractal inflorescence morphogenesis (`2021-pringle-roman-esco-fractal-inflorescence.py`) | 53 | 45 | 52 | 52 | 68 | 58 | 54 | 52.7 | 47.6 | 59.7 | 52.8 | **52.8** | partially faithful |

## Findings and remediation

### Azpeitia et al. (2021) & Prusinkiewicz et al. (2007) TFL1-LFY-AP1/CAL-SOC1 floral GRN perturbation and accelerating plastochron Romanesco fractal inflorescence morphogenesis  —  overall 52.8 (partially faithful)

Pass 1 (v1) baseline evaluation of `2021-pringle-roman-esco-fractal-inflorescence.py` (weighted score: 53.5/100). The Pass 1 baseline computes only a 1D quadratic polynomial for primordium count and a 2D single-order Vogel spiral without solving the coupled TFL1-LFY-AP1/CAL-SOC1 floral gene-regulatory network ODEs, 3D multi-order recursive meristem trees across wild-type raceme, Arabidopsis ap1 cal, Brassica Cauliflower (beta=0), and Brassica Romanesco (beta>0) genotypes, 3D box-counting fractal dimension D_f, Fibonacci parastichy transitions (8,13) -> (13,21), lfy mutant negative controls, or computer-vision segmentation of the vendored Romanesco Brassica oleracea specimen.

- **Model fidelity to source: 53** — Evaluates only a 1D quadratic curve for N(T) and a flat 2D single-order golden-angle spiral (L24-39), omitting the coupled TFL1-LFY-AP1/CAL-SOC1 nonlinear Hill ODE system, 3D recursive multi-order meristem tree generation, and 3D box-counting fractal dimension analysis.  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L24–39
- **Morphological realism / specimen match: 45** — Vendors `datasets/roman_esco_fractal_inflorescence/romanesco_brassica_oleracea_specimen.jpg` on disk, but Pass 1 does not load, segment, or measure hierarchical floret cones, Fibonacci parastichy spirals, or self-similar scaling on the specimen photograph.  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L59–86
- **Parameter expressivity: 52** — Hardcodes a single beta = 0.085 comparison without sweeping plastochron acceleration beta across [0.0, 0.11], comparing the 4 canonical genotypes (WT raceme, Arabidopsis ap1 cal, Brassica Cauliflower, Brassica Romanesco) plus lfy loss-of-function control, or exposing CLI parameter flags.  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L24–39
- **Morphometric validation: 52** — Reports only 2 analytical numbers in `measure_morphoeval()` without GRN steady-state validation, 3D fractal dimension D_f, Fibonacci parastichy contact numbers, Pearson correlations/p-values across the plastochron sweep, or specimen CV metrics.  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L42–56
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `roman_esco_fractal_inflorescence_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/roman_esco_fractal_inflorescence/roman_esco_fractal_inflorescence_reference.json` L1–165
- **Code-to-model traceability: 58** — Brief header cites Azpeitia et al. (2021) and Prusinkiewicz et al. (2007), but lacks explicit equation-to-code-line and Lean 4 theorem mappings.  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L1–7
- **Reproducibility package: 54** — Generates PNG and GIF deterministically, but has only a 2-assertion `--self-test` without SHA-256 dataset verification, GRN perturbation checks, or specimen CV checks.  
  evidence: `2021-pringle-roman-esco-fractal-inflorescence.py` L103–108

Remediation:
- Implement the coupled TFL1-LFY-AP1/CAL-SOC1 nonlinear Hill ODE gene-regulatory network solver (Azpeitia et al. 2021 Science; Prusinkiewicz et al. 2007 Science) across wild-type raceme (g_AP1=1), ap1 cal / curd mutants (g_AP1=0), and lfy loss-of-function negative control.
- Implement the 3D multi-order recursive phyllotactic meristem tree solver across 4 hierarchical orders (n=4) and sweep plastochron acceleration beta in [0.0, 0.11], computing conical aspect ratio H/R, 3D box-counting fractal dimension D_f, and Voronoi/Delaunay Fibonacci contact parastichy pairs (8,13) -> (13,21).
- Implement automated computer-vision segmentation of the SHA-256-locked Romanesco Brassica oleracea specimen photograph (`romanesco_brassica_oleracea_specimen.jpg`) to extract the whole-curd conical aspect ratio, multi-scale secondary/tertiary floret cones, empirical self-similar scale ratio lambda, and 2D/3D fractal box-counting dimension.
- Add runtime SHA-256 dataset verification, 9 reference-anchored morphometric measurements with statistical p-values and negative controls, and an 8-point deterministic `--self-test` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Azpeitia et al. (2021) & Prusinkiewicz et al. (2007) TFL1-LFY-AP1/CAL-SOC1 floral GRN perturbation and accelerating plastochron Romanesco fractal inflorescence morphogenesis
- **spec_fingerprint** (semantic model + inputs): `3b35051e2ba55236a89c013359c0e36df27af50f133cd223b665f430f960cce0`
- **input_dataset_sha256**: `datasets/roman_esco_fractal_inflorescence/paper_citations.bib` (`4906c890ff71618e…`), `datasets/roman_esco_fractal_inflorescence/roman_esco_fractal_inflorescence_reference.json` (`b580780d055808ce…`), `datasets/roman_esco_fractal_inflorescence/romanesco_brassica_oleracea_specimen.jpg` (`5fba6c320b478d6c…`), `datasets/roman_esco_fractal_inflorescence/scout_manifest.json` (`d684b2df5ef44398…`)
- `docs/media/roman_esco_fractal_inflorescence_comparison.png` — artifact_sha256 `f04c92d85381a93b…`, 74,902 B, 1300×585 px

| measurement | value | reference | method |
|---|---:|---|---|
| romanesco_primordia_at_T20 | 37.0 count | 37.0 | 1D analytical polynomial r0*T + 0.5*beta*T^2 |
| romanesco_cone_aspect_ratio | 1.4797000000000002 ratio | 1.48 | Linear aspect ratio fit in beta |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2021-pringle-roman-esco-fractal-inflorescence.py` prior to Step 12 closed-loop remediation.
