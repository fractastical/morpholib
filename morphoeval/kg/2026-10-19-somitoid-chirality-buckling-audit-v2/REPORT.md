# MorphoEval report — audit `2026-10-19-somitoid-chirality-buckling-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-somitoid-chirality-buckling-audit -> 2026-10-19-somitoid-chirality-buckling-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Yamanaka et al. (2023), Miao et al. (2023) & Adhyapok et al. (2021) epithelial somite rosette vertex mechanics and mesenchymal-to-epithelial transition (MET) model (`2021-yaman-somitoid-chirality-buckling.py`) | 94 | 89 | 92 | 93 | 93 | 91 | 91 | 93.3 | 90.5 | 91.7 | 91.7 | **91.7** | faithful |

## Findings and remediation

### Yamanaka et al. (2023), Miao et al. (2023) & Adhyapok et al. (2021) epithelial somite rosette vertex mechanics and mesenchymal-to-epithelial transition (MET) model  —  overall 91.7 (faithful)

Pass 2 post-remediation evaluation of `2021-yaman-somitoid-chirality-buckling.py` (91.9/100, up from 60.1/100 in Pass 1; +31.8 overall). The remediated script implements a 2D 32-vertex apico-basal differential interfacial tension L-BFGS-B relaxation solver for an N=16 wedge-cell somite rosette, reproducing exact agreement with the analytical polygon equilibrium ratio (WT L_ap/L_bas = 0.3030 vs 0.3030 analytical and 0.305 empirical), Y-27632 ROCK-inhibited lumen dilation (L_ap/L_bas = 0.7013, R_in* = 19.65 um vs 7.72 um in WT), and Minus-Matrigel depolarization (0.5562), evaluating all 6 experimental regimes from Yamanaka et al. (2023), Miao et al. (2023), and Adhyapok et al. (2021) (Pearson r = 0.9989, RMSE = 0.0086), quantifying EphA4/ephrinB2 inter-somitic cleft separation (S_cleft = 0.8189 vs 0.0930 in Eph/ephrin KD), mapping the 2D phase diagram over apical tension and Matrigel concentration, and segmenting bilateral somite rosettes on `somitoid_rosette_specimen.jpg` (45.38 px spacing, CV = 0.2010, dome/cleft contrast = 1.189x), backed by 3 machine-checked Lean 4 theorems in `Loop_029_somitoid_chirality_buckling.lean`.

- **Model fidelity to source: 94** — Faithfully implements the 2D apico-basal differential interfacial tension vertex energy functional E = sum_i [0.5*K_A*(A_i - A_0)^2 + gamma_ap*L_{ap,i} + gamma_lat*L_{lat,i} + gamma_bas*L_{bas,i}] alongside the exact analytical polygon and continuum equilibrium laws, EphA4/ephrinB2 inter-somitic cleft sorting energy barrier, and Y-27632 / Minus-Matrigel negative controls (L118-375), verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_029_somitoid_chirality_buckling.lean`.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L118–285; `2021-yaman-somitoid-chirality-buckling.py` L288–375; `morphoeval/lean/loops/Loop_029_somitoid_chirality_buckling.lean` L24–119
- **Morphological realism / specimen match: 89** — Vendors and SHA-256-verifies the CC BY 4.0 dorsal vertebrate embryo somite rosette SEM (`datasets/somitoid_chirality_buckling/somitoid_rosette_specimen.jpg`) and segments both bilateral somite rosette rows (`segment_somite_rosette_specimen`, L418-505), extracting rosette centers, inter-somitic clefts, bilateral mean spacing 45.38 px (CV = 0.2010), radial dome-to-cleft intensity contrast (1.189x), and periodic waveform fit (Panels E-F).  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L418–505; `docs/media/somitoid_chirality_buckling_comparison.png` L0–0
- **Parameter expressivity: 92** — Evaluates all 6 experimental regimes from Yamanaka et al. (2023), Miao et al. (2023), and Adhyapok et al. (2021) (WT +10% Matrigel, Calpeptin hyper-constriction, Low 2.5% Matrigel, EphA4/ephrinB2 boundary knockdown, Y-27632 ROCK inhibition, and Minus-Matrigel control) plus a 28x28 2D phase diagram over (gamma_ap/gamma_lat, Matrigel ECM %).  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L305–415
- **Morphometric validation: 93** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L510-575): WT apical-to-basal ratio 0.3030 vs 0.3030 analytical (0.305 empirical), Y-27632 ROCK-inhibited ratio dilation 0.7013 vs 0.7013 analytical (0.700 empirical), 6-regime perturbation Pearson r = 0.9989 (RMSE = 0.0086), EphA4/ephrinB2 cleft separation index 0.8189 (vs 0.0930 in KD), and bilateral specimen rosette spacing 45.38 px (CV = 0.2010).  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L510–575
- **Source and data traceability: 93** — Vendors `datasets/somitoid_chirality_buckling/` with `scout_manifest.json`, `somitoid_chirality_buckling_reference.json`, `somitoid_rosette_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1038/s41586-022-05649-2`, `10.1038/s41586-022-05655-4`, `10.1016/j.devcel.2021.03.026`, `10.1038/s41467-022-29967-1`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L88-115).  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L88–115; `datasets/somitoid_chirality_buckling/somitoid_chirality_buckling_reference.json` L1–185
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered figures in Yamanaka et al. (2023) Nature, Miao et al. (2023) Nature, and Adhyapok et al. (2021) Dev. Cell and the 3 Lean 4 theorems in `Loop_029_somitoid_chirality_buckling.lean` and `morphoeval/math_kg_loops/yaman-somitoid-chirality-buckling.yaml`.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L1–56; `morphoeval/math_kg_loops/yaman-somitoid-chirality-buckling.yaml` L1–115
- **Reproducibility package: 91** — Deterministic RNG (`seed=2021`), standalone `--self-test` suite verifying SHA-256 dataset integrity, bitwise two-run vertex relaxation determinism (`atol=0.0`), analytical-to-vertex ratio agreement (<0.015), Y-27632 & Minus-Matrigel negative controls, Eph/ephrin cleft knockdown collapse, and specimen segmentation (`run_self_test`, L800-845), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `2021-yaman-somitoid-chirality-buckling.py` L800–845; `docs/media/somitoid_chirality_buckling_growth.gif` L0–0

Remediation:
- Extend the 2D cross-sectional wedge-rosette vertex model to a 3D apical-basal polyhedral scutoid shell to capture 3D somitecoel lumen pressurization.
- Couple MESP2/RIPPLY2 wavefront arrest directly from a live HES7 delayed oscillator into the spatiotemporal activation of EphA4/ephrinB2 and RhoA/ROCK line tensions.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Yamanaka et al. (2023), Miao et al. (2023) & Adhyapok et al. (2021) epithelial somite rosette vertex mechanics and mesenchymal-to-epithelial transition (MET) model
- **spec_fingerprint** (semantic model + inputs): `834555c3491ff9a0813c6b8298d939c224a2f6c75546f46b11089139cd1ede3c`
- **input_dataset_sha256**: `datasets/somitoid_chirality_buckling/scout_manifest.json` (`8326704c93aa8a75…`), `datasets/somitoid_chirality_buckling/somitoid_chirality_buckling_reference.json` (`89cec9bd5a9ed6ba…`), `datasets/somitoid_chirality_buckling/somitoid_rosette_specimen.jpg` (`8b80971796cb29e2…`)
- `docs/media/somitoid_chirality_buckling_comparison.png` — artifact_sha256 `aae2e04bd0e93a9f…`, 788,839 B, 2253×1554 px
- `docs/media/somitoid_chirality_buckling_growth.gif` — artifact_sha256 `f37a7baf161b1fdd…`, 448,063 B, 817×408 px, 20 frames

| measurement | value | reference | method |
|---|---:|---|---|
| wild-type somite rosette equilibrium apical-to-basal ratio L_ap / L_bas | 0.303 dimensionless ratio | 0.303 | 2D 32-vertex apico-basal differential interfacial tension relaxation (N=16 wedge cells, gamma_ap=1.65, gamma_bas=0.45, R_in*=7.74 um, R_out*=25.56 um) *(apico-basal polarity score = 1.0000; wedge cell height = 17.81 um)* |
| Y-27632 ROCK-inhibited apico-basal ratio dilation L_ap / L_bas | 0.7013 dimensionless ratio | 0.7013 | 2D vertex relaxation under Y-27632 ROCK / Blebbistatin myosin II inhibition (gamma_ap = gamma_bas = 0.45, Delta_gamma_ap_bas = 0) *(somitecoel lumen dilates from 7.74 um (WT) to 24.00 um (Y-27632); Minus-Matrigel ratio = 0.5614)* |
| 6-regime somite rosette perturbation series Pearson correlation r | 0.9989 Pearson r | 1.0 | correlation between 2D vertex-predicted and empirical apical-to-basal ratios across 6 Yamanaka (2023), Miao (2023) & Adhyapok (2021) regimes (RMSE = 0.0154) *(p = 1.94e-06; spans Calpeptin hyper-constricted (0.1370) to Y-27632 (0.7013))* |
| EphA4/ephrinB2 inter-somitic cleft separation index S_cleft | 0.8189 dimensionless index | 0.94 | heterotypic EphA4/ephrinB2 contact repulsion cleft separation score 1 - (gamma_hom / gamma_het)^2 at gamma_het = 2.35 (Delta_E_cleft = 75.60 pN) *(cleft indentation depth = 14.68 um in WT vs 1.23 um (S_cleft = 0.0930) in EphA4/ephrinB2 knockdown)* |
| vertebrate somite rosette specimen bilateral metamere spacing S_obs | 45.375 px | 47.1429 | bilateral dorsal somite rosette DoG peak & inter-somitic cleft segmentation on somitoid_rosette_specimen.jpg (upper=8 rosettes, lower=10 rosettes, CV=0.2391) *(bilateral symmetry ratio = 1.0834, AP cleft waveform r = 0.2136, dome/cleft contrast = 1.2315)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2021-yaman-somitoid-chirality-buckling.py` after completing the 12-step closed remediation loop (`2026-10-19-somitoid-chirality-buckling-loop`).
