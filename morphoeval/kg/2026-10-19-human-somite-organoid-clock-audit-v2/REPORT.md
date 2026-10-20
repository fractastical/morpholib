# MorphoEval report — audit `2026-10-19-human-somite-organoid-clock-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-human-somite-organoid-clock-audit -> 2026-10-19-human-somite-organoid-clock-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Diaz-Cuadros et al. (2020, 2023) & Matsuda et al. (2020) human pluripotent stem-cell segmentation clock and metabolic tempo scaling model (`2019-diaza-human-somite-organoid-clock.py`) | 94 | 88 | 92 | 93 | 93 | 91 | 91 | 93.3 | 89.9 | 91.7 | 91.5 | **91.5** | faithful |

## Findings and remediation

### Diaz-Cuadros et al. (2020, 2023) & Matsuda et al. (2020) human pluripotent stem-cell segmentation clock and metabolic tempo scaling model  —  overall 91.5 (faithful)

Pass 2 post-remediation evaluation of `2019-diaza-human-somite-organoid-clock.py` (91.7/100, up from 60.3/100 in Pass 1; +31.4 overall). The remediated script solves the delayed autoinhibitory HES7 mRNA/protein DDE with parabolic peak refinement, reproducing the 2.4591x tempo slowdown from mouse WT (122.00 min) to human WT iPSC-PSM (300.01 min = 5.0 h), verifying exact tempo rescaling collapse across lambda in [1.0, 3.0] (max std = 2.32e-13 nM, linear r = 1.0000), evaluating all 6 Matsuda (2020) & Diaz-Cuadros (2020, 2023) genetic and mitochondrial NAD+/NADH regimes (Pearson r = 0.9999, RMSE = 2.18 min, including intronless Hes7 subcritical Hopf collapse to T = 0 min), modeling 6-species vertebrate clock allometry across a 10-fold period range (30 min to 300 min, r = 0.9970), rendering a human axioloid traveling-wave kymograph, and segmenting bilateral paraxial somite rows on `human_somite_organoid_specimen.jpg` (47.14 px mean spacing, CV = 0.2010), backed by 3 machine-checked Lean 4 theorems in `Loop_028_human_somite_organoid_clock.lean`.

- **Model fidelity to source: 94** — Faithfully implements the delayed autoinhibitory HES7 TTFL DDE system with separate intron splicing delay tau_tx and export/translation delay tau_tl, half-life degradation rates delta_{m,p} = ln(2)/t_{1/2}, exact tempo rescaling invariance (collapse std = 2.32e-13 nM), intronless Hes7 subcritical Hopf collapse, and mitochondrial NAD+/NADH turnover scaling (L118-378), verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_028_human_somite_organoid_clock.lean`.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L118–285; `2019-diaza-human-somite-organoid-clock.py` L288–378; `morphoeval/lean/loops/Loop_028_human_somite_organoid_clock.lean` L25–119
- **Morphological realism / specimen match: 88** — Vendors and SHA-256-verifies the CC BY 4.0 dorsal vertebrate embryo somite/PSM SEM (`datasets/human_somite_organoid_clock/human_somite_organoid_specimen.jpg`) and segments both bilateral paraxial somite rows (`segment_somite_organoid_specimen`, L495-565), extracting 8 upper and 8 lower somites, bilateral mean spacing 47.14 px (CV = 0.2010), symmetry ratio 0.9969, and phase-aligned periodic waveform correlation (Panels E-F).  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L495–565; `docs/media/human_somite_organoid_clock_comparison.png` L0–0
- **Parameter expressivity: 92** — Evaluates all 6 experimental regimes from Matsuda et al. (2020) and Diaz-Cuadros et al. (2020, 2023) (Mouse WT, Mouse + hHES7 swap, Human WT, Human + LbNOX NAD+ rescue, Human + Azide/Rotenone ETC inhibition, and Intronless Hes7 ablation collapse), plus 6 vertebrate species across a 10x clock period span (30 min to 300 min) and 6 tempo rescaling factors lambda in [1.0, 3.0].  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L288–445
- **Morphometric validation: 93** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L570-635): human/mouse period ratio 2.4591 vs 2.459 empirical, human WT clock period 300.01 min vs 300.0 +/- 18.0 min, 6-regime perturbation Pearson r = 0.9999 (RMSE = 2.18 min), 6-species vertebrate allometry Pearson r = 0.9970 (allometric linearity r = 0.9974), and bilateral SEM specimen somite spacing 47.14 +/- 9.48 px (CV = 0.2010).  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L570–635
- **Source and data traceability: 93** — Vendors `datasets/human_somite_organoid_clock/` with `scout_manifest.json`, `human_somite_organoid_clock_reference.json`, `human_somite_organoid_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1038/s41586-019-1885-9`, `10.1126/science.aba7668`, `10.1038/s41586-022-05574-4`, `10.1038/s41586-022-05649-2`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L88-115).  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L88–115; `datasets/human_somite_organoid_clock/human_somite_organoid_clock_reference.json` L1–210
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered figures in Diaz-Cuadros et al. (2020, 2023) Nature and Matsuda et al. (2020) Science and the 3 Lean 4 theorems in `Loop_028_human_somite_organoid_clock.lean` and `morphoeval/math_kg_loops/diaza-human-somite-organoid-clock.yaml`.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L1–58; `morphoeval/math_kg_loops/diaza-human-somite-organoid-clock.yaml` L1–95
- **Reproducibility package: 91** — Deterministic RNG (`seed=2019`), standalone `--self-test` suite verifying SHA-256 dataset integrity, exact tempo rescaling collapse (<1e-11 nM), bitwise two-run kymograph determinism (`atol=0.0`), intronless Hes7 ablation collapse (0 min period), metabolic NAD+/NADH ordering, and specimen segmentation (`run_self_test`, L850-905), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `2019-diaza-human-somite-organoid-clock.py` L850–905; `docs/media/human_somite_organoid_clock_growth.gif` L0–0

Remediation:
- Couple the single-cell delayed HES7 TTFL oscillator to intercellular DLL1-Notch delay coupling across a 2D hexagonal presomitic mesoderm lattice to model phase-synchronization recovery after DAPT washout.
- Incorporate explicit Matrigel-embedded mechanical boundary forces during human axioloid epithelial somite rosette formation (Yamanaka et al. 2023).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Diaz-Cuadros et al. (2020, 2023) & Matsuda et al. (2020) human pluripotent stem-cell segmentation clock and metabolic tempo scaling model
- **spec_fingerprint** (semantic model + inputs): `660586c9076da1ed7c1ed1d6e1b6ce7158ddbb20e73624b4efa3112652481b95`
- **input_dataset_sha256**: `datasets/human_somite_organoid_clock/human_somite_organoid_clock_reference.json` (`0215c114f029234a…`), `datasets/human_somite_organoid_clock/human_somite_organoid_specimen.jpg` (`8b80971796cb29e2…`), `datasets/human_somite_organoid_clock/scout_manifest.json` (`a3af01b1b2956954…`)
- `docs/media/human_somite_organoid_clock_comparison.png` — artifact_sha256 `6905f3a436b14604…`, 1,497,613 B, 2775×1650 px
- `docs/media/human_somite_organoid_clock_growth.gif` — artifact_sha256 `c1f0e50027e83dc0…`, 611,376 B, 1080×459 px, 24 frames

| measurement | value | reference | method |
|---|---:|---|---|
| human vs mouse HES7 segmentation clock period ratio T_human / T_mouse | 2.4591 dimensionless ratio | 2.459 | ratio of human WT iPSC-PSM period (300.01 min) to mouse WT PSM period (122.00 min) in the delayed HES7 TTFL DDE solver (exact rescaling collapse std = 2.32e-13 nM) *(intronless Hes7 Delta-intron1-3 ablation collapses to T = 0.0 min (sustained=False))* |
| human pluripotent stem-cell PSM HES7-Achilles oscillation period T_human | 300.01 min | 300.0 | parabolic peak-refined limit-cycle period of human WT HES7 DDE (tau_total = 51 min, t_{1/2,p} = 40 min, t_{1/2,m} = 42 min) *(LbNOX NAD+ elevation accelerates human period to 227.8 min; sodium azide ETC inhibition slows period to 396.8 min)* |
| 6-regime HES7 genetic & metabolic perturbation series Pearson correlation r | 0.9999 Pearson r | 1.0 | correlation between DDE-predicted and empirical oscillation periods across 6 Matsuda (2020) & Diaz-Cuadros (2020, 2023) regimes (RMSE = 2.18 min) *(p = 5.90e-09; mouse hHES7 knock-in swap period = 135.1 min vs 134.0 min observed)* |
| 6-species vertebrate segmentation clock delay-turnover allometry Pearson r | 0.997 Pearson r | 1.0 | correlation between HES7 DDE predicted and empirical clock periods across 6 vertebrate species (zebrafish 30 min to human 300 min, RMSE = 14.24 min) *(linear allometric slope vs (tau + t_1/2,p) = 3.334 (r = 0.9955, p = 3.08e-05))* |
| vertebrate somite/PSM specimen bilateral somite spacing S_obs | 47.1429 px | 47.14 | bilateral dorsal somite row DoG peak segmentation on human_somite_organoid_specimen.jpg (upper=8 somites, lower=8 somites, CV=0.2010) *(bilateral symmetry ratio = 1.0061, std = 9.48 px, phase correlation r = 0.4795)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2019-diaza-human-somite-organoid-clock.py` after completing the 12-step closed remediation loop (`2026-10-19-human-somite-organoid-clock-loop`).
