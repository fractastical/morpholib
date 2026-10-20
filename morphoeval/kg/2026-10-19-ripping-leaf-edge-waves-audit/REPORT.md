# MorphoEval report — audit `2026-10-19-ripping-leaf-edge-waves-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/67

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sharon et al. (2002) Nature / Audoly & Boudaoud (2003) PRL Non-Euclidean Edge Buckling Cascades in Ruffled Leaf Margins (`2002-sharon-ripping-leaf-edge-waves.py`) | 56 | 46 | 52 | 55 | 68 | 61 | 63 | 54.7 | 49.4 | 64.1 | 55.4 | **55.4** | partially faithful |

## Findings and remediation

### Sharon et al. (2002) Nature / Audoly & Boudaoud (2003) PRL Non-Euclidean Edge Buckling Cascades in Ruffled Leaf Margins  —  overall 55.4 (partially faithful)

Pass 1 (v1) baseline evaluation of `2002-sharon-ripping-leaf-edge-waves.py` (overall weighted score: 56.2/100). The baseline evaluates the analytical isometric aspect ratio A/lambda and plots a kinematic sinusoidal superposition, but lacks variational L-BFGS-B minimization of the 2D non-Euclidean Föppl-von Kármán thin-sheet energy E = E_stretch + E_bend across sheet thickness t and marginal growth profile g_yy(x), omits the Brioschi negative Gaussian curvature profile K(x) < 0, omits the Liang & Mahadevan (2009) saddle-to-ripple transition and multi-system taxon comparison, and does not segment the vendored Lactuca sativa 'Lollo Bionda' ruffled leaf specimen image.

- **Model fidelity to source: 56** — Evaluates analytical g_yy(x) and single-mode aspect ratio (`target_metric_gyy`, `analytical_aspect_ratio`, L26-35), but uses a fixed kinematic sine sum (`synthesize_baseline_surface`, L38-58) without L-BFGS-B minimization of the non-Euclidean Föppl-von Kármán stretching + bending energy or Brioschi Gaussian curvature K(x) verification.  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L26–58
- **Morphological realism / specimen match: 46** — Does not load or segment the vendored Lactuca sativa 'Lollo Bionda' specimen photograph `datasets/ripping_leaf_edge_waves/lactuca_sativa_edge_waves_specimen.jpg` and renders only a 2-panel baseline plot.  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L61–86
- **Parameter expressivity: 52** — Evaluates only a single default parameter tuple without sweeping inward distance x, sheet thickness t (Liang & Mahadevan 2009 saddle-to-ripple transition), or CLI parameter overrides.  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L38–58
- **Morphometric validation: 55** — `measure_morphoeval()` (L109-133) returns only 2 basic analytical scalars without multi-generation cascade wavelength/amplitude correlations, self-similar collapse verification, or specimen image segmentation metrics.  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L109–133
- **Source and data traceability: 68** — Vendors `datasets/ripping_leaf_edge_waves/` with 4 DOIs in `paper_citations.bib`, `scout_manifest.json`, reference JSON, and Lactuca sativa specimen JPEG, but the baseline script does not enforce runtime SHA-256 verification.  
  evidence: `datasets/ripping_leaf_edge_waves/paper_citations.bib` L1–39
- **Code-to-model traceability: 61** — Cites Sharon et al. (2002), Audoly & Boudaoud (2003), and Liang & Mahadevan (2009) in module docstring, but lacks function-level equation mappings to the 3 Lean 4 theorems in `Loop_067_ripping_leaf_edge_waves.lean`.  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L1–11
- **Reproducibility package: 63** — Provides `--self-test` (`L136-141`), but checks only two analytical values without SHA-256 dataset verification, multi-generation cascade relaxation checks, or bitwise solver determinism checks.  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L136–141

Remediation:
- Implement a variational L-BFGS-B energy minimizer for the 2D non-Euclidean Föppl-von Kármán thin-sheet energy functional E = E_stretch + E_bend across sheet thickness t and marginal growth profile g_yy(x) = 1 + g_0 exp(-x/ell), reproducing the multi-generation Fourier buckling cascade spectrum and self-similar collapse of w(x, y)/lambda(x).
- Compute the exact Brioschi intrinsic Gaussian curvature K(x) < 0 profile, evaluate the Liang & Mahadevan (2009) saddle-to-ripple transition across dimensionless strain-to-thickness parameter Gamma = g_0*(ell/t)^2, and verify the flat Euclidean metric negative control (g_0 = 0 => w = 0, K = 0).
- Implement computer-vision segmentation of `datasets/ripping_leaf_edge_waves/lactuca_sativa_edge_waves_specimen.jpg` (*Lactuca sativa* cv. 'Lollo Bionda'), extracting the ruffled canopy boundary and multi-scale marginal wave wavelengths & amplitudes across coarse lobes and fine micro-ruffles.
- Enforce runtime SHA-256 verification of all 4 dataset files, expand `measure_morphoeval()` to 9 quantitative reference-anchored metrics, and upgrade the figure to a 6-panel synthesis layout.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sharon et al. (2002) Nature / Audoly & Boudaoud (2003) PRL Non-Euclidean Edge Buckling Cascades in Ruffled Leaf Margins
- **spec_fingerprint** (semantic model + inputs): `7f611592e1fd4b993235be4db6e1a4477fa6a36fa0f4a6d7d8002d143a1ff98e`
- **input_dataset_sha256**: `datasets/ripping_leaf_edge_waves/lactuca_sativa_edge_waves_specimen.jpg` (`921d920a85d530f7…`), `datasets/ripping_leaf_edge_waves/paper_citations.bib` (`d9c45c59e0864ec3…`), `datasets/ripping_leaf_edge_waves/ripping_leaf_edge_waves_reference.json` (`7791c76d9fb5bd78…`), `datasets/ripping_leaf_edge_waves/scout_manifest.json` (`9d72416d0542e21c…`)
- `docs/media/ripping_leaf_edge_waves_comparison.png` — artifact_sha256 `378af2a5aeb7eb13…`, 191,980 B, 1430×624 px

| measurement | value | reference | method |
|---|---:|---|---|
| Sharon et al. (2002) isometric edge wave aspect ratio A/lambda (g_0=0.64) | 0.1801 dimensionless | 0.1801 | Analytical single-mode isometric arc-length relation A/lambda = sqrt(g_0)/(sqrt(2)*pi) |
| Audoly & Boudaoud (2003) sub-harmonic cascade wavelength ratio r_cascade | 3.5 dimensionless | 3.5 | Configured sub-harmonic wavelength ratio lambda_m / lambda_{m+1} |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2002-sharon-ripping-leaf-edge-waves.py` prior to Step 12 remediation.
