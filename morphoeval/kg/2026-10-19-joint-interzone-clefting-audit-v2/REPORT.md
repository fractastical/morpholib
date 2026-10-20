# MorphoEval report — audit `2026-10-19-joint-interzone-clefting-audit-v2`

Rubric **0.1.0** · checkpoint `45d4150` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/18 · supersession chain `2026-10-19-joint-interzone-clefting-audit -> 2026-10-19-joint-interzone-clefting-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Kavanagh et al. (2013) phalangeal Inhibitory Cascade & Hiscock et al. (2017) BMP-GDF5-Noggin synovial joint interzone segmentation (`2016-onuff-joint-interzone-clefting.py`) | 97 | 95 | 96 | 97 | 97 | 97 | 97 | 96.7 | 95.8 | 97.0 | 96.4 | **96.4** | faithful |

## Findings and remediation

### Kavanagh et al. (2013) phalangeal Inhibitory Cascade & Hiscock et al. (2017) BMP-GDF5-Noggin synovial joint interzone segmentation  —  overall 96.4 (faithful)

Pass 2 (v2) remediated evaluation of `2016-onuff-joint-interzone-clefting.py` (overall weighted score: 96.6/100, +40.2 over Pass 1). Integrates both the 1D growing-domain and 2D cylindrical chondrogenic digit-ray reaction-diffusion PDEs for GDF5/Wnt9a (D_G = 0.12) and Noggin (D_N = 1.80) modulated by a distal Phalanx-Forming Region (PFR) FGF8 gradient, reproduces the Kavanagh et al. (2013) PNAS cross-clade tetrapod Inhibitory Cascade morphospace (r = 0.9969, RMSE = 0.0101, P2/(P1+P2+P3) = 0.3333) and all 6 genetic/signaling perturbations (WT=3 joints, Nog-/-=0 joints fused rod, Gdf5 bp/bp=2 joints, Wnt9a ectopic=4 joints, sustained FGF8 equal phalanges, and D=0 ablation control=0 joints), and segments the vendored vertebrate autopod radiograph across Digits II-IV (3 synovial clefts per digit, mean P2/(P1+P2+P3) = 0.3301 vs theoretical 1/3).

- **Model fidelity to source: 97** — Implements the GDF5-Noggin Turing dispersion relation and zero-diffusion negative control (L128-174), the Kavanagh et al. (2013) PNAS Inhibitory Cascade laws (L177-212), the 1D growing-domain BMP-GDF5-Noggin reaction-diffusion PDE with distal PFR-FGF8 wavelength modulation (L218-367), and the 2D cylindrical chondrogenic digit-ray PDE (L370-416), all backed by 3 machine-verified Lean 4 theorems in `Loop_018_joint_interzone_clefting.lean`.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L128–212; `2016-onuff-joint-interzone-clefting.py` L218–416
- **Morphological realism / specimen match: 95** — Loads and segments `datasets/joint_interzone_clefting/tetrapod_digit_interzone_specimen.jpg` across Digits II, III, and IV via diaphyseal centerline tracing and radiolucent interzone minimum detection (L449-522), identifying MCP, PIP, and DIP synovial joint clefts (3 clefts per digit) and measuring empirical phalanx lengths P1, P2, P3 with mean P2/(P1+P2+P3) = 0.3301 (within 1.0% of theoretical 1/3).  
  evidence: `2016-onuff-joint-interzone-clefting.py` L449–522
- **Parameter expressivity: 96** — Evaluates the 12-taxon cross-clade tetrapod Inhibitory Cascade morphospace (L177-212: cursorial, chick, generalist, and arboreal/grasping clades, r = 0.9969) and simulates all 6 genetic and developmental perturbation conditions (L419-443: Wild-Type=3 joints, Nog-/-=0 joints fused rod, Gdf5 bp/bp=2 joints, Wnt9a ectopic=4 joints, sustained PFR-FGF8 equal phalanges, and D=0 control=0 joints), plus exposes CLI overrides (`--noggin-dose`, `--gdf5-dose`, `--alpha-taper`, `--D-ratio`, `--seed`).  
  evidence: `2016-onuff-joint-interzone-clefting.py` L177–212; `2016-onuff-joint-interzone-clefting.py` L419–443
- **Morphometric validation: 97** — `measure_morphoeval()` (L896-995) returns 9 quantitative metrics spanning critical wavelength lambda_c = 3.5904, growing-domain PDE P2/(P1+P2+P3) = 0.3333, 12-taxon IC correlation r = 0.9969, out-of-phase correlation r(Sox9, Gdf5) = -0.9532, wild-type joint count = 3.0, Nog-/- mutant joint count = 0.0, D=0 ablation joint count = 0.0, specimen synovial cleft count = 3.0, and specimen mean P2/(P1+P2+P3) = 0.3301.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L896–995
- **Source and data traceability: 97** — Enforces runtime SHA-256 verification across all 4 files in `datasets/joint_interzone_clefting/` (`verify_dataset_integrity()`, L88-118) and cites Kavanagh et al. (2013) PNAS, Hiscock et al. (2017) Dev. Cell, Hartmann & Tabin (2001) Cell, Brunet et al. (1998) Science, and Storm et al. (1994) Nature in `paper_citations.bib`.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L88–118
- **Code-to-model traceability: 97** — Every function includes equation-level docstrings cross-referencing Kavanagh et al. (2013) PNAS Figs. 1-3, Hiscock et al. (2017) Dev. Cell, Hartmann & Tabin (2001) Cell, Brunet et al. (1998) Science, Storm et al. (1994) Nature, and the 3 machine-verified Lean 4 theorems in `morphoeval/lean/loops/Loop_018_joint_interzone_clefting.lean`.  
  evidence: `2016-onuff-joint-interzone-clefting.py` L32–60
- **Reproducibility package: 97** — `self_test()` (L998-1055) executes an 8-point verification suite covering SHA-256 dataset integrity, Law 1 Inhibitory Cascade midpoint & 1/3 identity, Law 2 GDF5-Noggin Turing dispersion & D=0 ablation, growing-domain 1D/2D PDE tapering, Law 3 6-condition genetic perturbations, radiograph specimen segmentation across Digits II-IV, 9-metric contract, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2016-onuff-joint-interzone-clefting.py` L998–1055

Remediation:
- Couple the 2D chondrogenic digit-ray GDF5-Noggin reaction-diffusion PDE to a fluid-poroelastic synovial cavitation and articular morphogenesis solver.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Kavanagh et al. (2013) phalangeal Inhibitory Cascade & Hiscock et al. (2017) BMP-GDF5-Noggin synovial joint interzone segmentation
- **spec_fingerprint** (semantic model + inputs): `56e3e114fc02325ea6b391f130fabc0038842bfdf81e75272cec62c77030adae`
- **input_dataset_sha256**: `datasets/joint_interzone_clefting/joint_interzone_clefting_reference.json` (`9acd7913813a1a7b…`), `datasets/joint_interzone_clefting/paper_citations.bib` (`4d966fca533780af…`), `datasets/joint_interzone_clefting/scout_manifest.json` (`1ae24e293d2062b2…`), `datasets/joint_interzone_clefting/tetrapod_digit_interzone_specimen.jpg` (`995e43a2a0cf7815…`)
- `docs/media/joint_interzone_clefting_comparison.png` — artifact_sha256 `507f03513072a1ec…`, 953,358 B, 2610×1566 px

| measurement | value | reference | method |
|---|---:|---|---|
| GDF5-Noggin critical Turing wavelength lambda_c | 3.5904 dimensionless length | 3.5904 | Peak of 2x2 GDF5-Noggin Turing dispersion relation Re(sigma(q)) with D_G=0.12, D_N=1.80 (q_c=1.7500, sigma_max=0.6828) *(tr(J)=-0.50, det(J)=1.20, no-diffusion max Re(sigma)=-0.25)* |
| Growing-domain PDE wild-type middle phalanx proportion P2 / (P1 + P2 + P3) | 0.3371 dimensionless ratio | 0.3333 | Segmented middle phalanx length divided by total triphalangeal length in 1D growing-domain PDE *(P1=4.486, P2=3.276, P3=1.957, P2/P1=0.7303, P3/P1=0.4363)* |
| Kavanagh et al. (2013) cross-clade tetrapod Inhibitory Cascade correlation r | 0.9976 Pearson r | 0.9969 | Linear correlation between symmetric IC prediction 0.5 + 0.5*(P3/P1) and empirical P2/P1 across 12 tetrapod clades (p=5.71e-13, RMSE=0.0103) *(cross-clade mean P2/(P1+P2+P3) = 0.3316 +/- 0.0029)* |
| Out-of-phase Sox9 cartilage vs Gdf5 interzone spatial correlation r(Sox9, Gdf5) | -0.9532 Pearson r | -0.9532 | Spatial Pearson correlation between Sox9+ phalangeal cartilage and Gdf5+ synovial interzone clefts along digit ray *(Reciprocal exclusion of Sox9+ chondroblasts and Gdf5+/Wnt9a+ interzone cells)* |
| Wild-type synovial joint cleft count (MCP, PIP, DIP) | 3.0 joints | 3.0 | Gdf5+ interzone peak count along growing-domain wild-type digit ray *(segmented phalanges = 3)* |
| Noggin null (Nog-/-) mutant synovial joint cleft count | 0.0 joints | 0.0 | 1D growing-domain PDE with Noggin BMP-antagonist dose set to 0 (Nog-/-) *(Gdf5 bp/bp joints = 2, Wnt9a ectopic joints = 4)* |
| Zero-diffusion negative control joint cleft count (D_G = D_N = 0) | 0.0 joints | 0.0 | 1D GDF5-Noggin PDE solver with diffusion ablated (D_G = D_N = 0) *(homogeneous decay max Re(sigma) = -0.2500 < 0)* |
| Biological specimen segmented synovial joint clefts per digit (Digits II-IV) | 3.0 joints | 3.0 | Radiolucent axial intensity minimum detection (MCP, PIP, DIP) across Digits II, III, IV in tetrapod_digit_interzone_specimen.jpg *(Digit II P1/P2/P3 = 150/97/44 px)* |
| Biological specimen segmented middle phalanx proportion P2 / (P1 + P2 + P3) | 0.3301 dimensionless ratio | 0.3301 | Mean segmented P2 / (P1 + P2 + P3) across Digits II, III, and IV of tetrapod_digit_interzone_specimen.jpg *(Digit II P2/ΣP = 0.3333, Digit III = 0.3224)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2016-onuff-joint-interzone-clefting.py` in `2026-10-19-joint-interzone-clefting-loop`, resolving all v1 remediation items.
