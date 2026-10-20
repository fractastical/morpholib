# MorphoEval report — audit `2026-10-19-chiral-seed-pod-opening-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/65

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Armon et al. (2011) Science / Efrati et al. (2009) JMPS Incompatible Shell Elasticity & Chiral Seed Pod Opening (`2007-liang-chiral-seed-pod-opening.py`) | 57 | 45 | 51 | 54 | 68 | 60 | 62 | 55.0 | 48.4 | 63.5 | 54.9 | **54.9** | partially faithful |

## Findings and remediation

### Armon et al. (2011) Science / Efrati et al. (2009) JMPS Incompatible Shell Elasticity & Chiral Seed Pod Opening  —  overall 54.9 (partially faithful)

Pass 1 (v1) baseline evaluation of `2007-liang-chiral-seed-pod-opening.py` (overall weighted score: 55.8/100). The baseline evaluates the analytical Föppl-von Kármán critical width w_tilde_c and renders a kinematic helical surface, but lacks variational L-BFGS-B minimization of the non-Euclidean incompatible plate energy E = E_stretch + E_bend across w_tilde in [0.5, 10], omits the cutting-angle chirality sweep theta in [0 deg, 90 deg] (including the theta = 0 deg achiral circular ring negative control), omits the multi-specimen comparison, and does not segment the vendored Bauhinia variegata specimen image.

- **Model fidelity to source: 57** — Evaluates analytical critical width w_tilde_c and mean curvature H(w_tilde) (`critical_fvk_width`, `analytical_mean_curvature`, L29-39), but does not minimize the non-Euclidean incompatible shell energy functional E_stretch + E_bend via variational relaxation or verify Gauss incompatibility det(b_0) = -kappa_0^2 < 0.  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L29–77
- **Morphological realism / specimen match: 45** — Does not load or segment the vendored Bauhinia variegata specimen photograph `datasets/chiral_seed_pod_opening/bauhinia_chiral_seed_pod_specimen.jpg` and renders only a 2-panel baseline plot.  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L80–104
- **Parameter expressivity: 51** — Evaluates only a single default parameter tuple without sweeping dimensionless width w_tilde in [0.5, 10], cutting angle theta in [0 deg, 90 deg] (including theta = 0 deg achiral ring control), or CLI parameter overrides.  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L42–55
- **Morphometric validation: 54** — `measure_morphoeval()` (L124-147) returns only 2 basic analytical checks without pitch P(w_tilde) and radius R(w_tilde) transition correlations, cutting-angle chirality validation, or specimen image segmentation metrics.  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L124–147
- **Source and data traceability: 68** — Vendors `datasets/chiral_seed_pod_opening/` with 4 DOIs in `paper_citations.bib`, `scout_manifest.json`, reference JSON, and Bauhinia variegata specimen JPEG, but the baseline script does not enforce runtime SHA-256 verification.  
  evidence: `datasets/chiral_seed_pod_opening/paper_citations.bib` L1–39
- **Code-to-model traceability: 60** — Cites Armon et al. (2011) Science and Efrati et al. (2009) JMPS in module docstring, but lacks function-level equation mappings to the 3 Lean 4 theorems in `Loop_065_chiral_seed_pod_opening.lean`.  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L1–13
- **Reproducibility package: 62** — Provides `--self-test` (`L150-155`), but checks only two analytical values without SHA-256 dataset verification, negative controls, or bitwise solver determinism checks.  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L150–155

Remediation:
- Implement a variational L-BFGS-B energy minimizer for the non-Euclidean incompatible plate energy functional E[H, D, phi] = E_stretch + E_bend across dimensionless width w_tilde in [0.5, 10], reproducing both the narrow Pure-Twist Helicoid regime (w_tilde <= w_tilde_c, H = 0, R = 0) and the wide Cylindrical Helical Ribbon regime (w_tilde > w_tilde_c, H -> (1-nu)*kappa_0/2).
- Implement the cutting-angle chirality sweep across theta in [0 deg, 90 deg], verifying that theta = 45 deg maximizes helical twist tau = D*sin(2*theta) while theta = 0 deg and theta = 90 deg serve as achiral circular ring controls (tau = 0, pitch P = 0), plus zero-shrinkage (kappa_0 = 0) flat pod control.
- Implement computer-vision segmentation of `datasets/chiral_seed_pod_opening/bauhinia_chiral_seed_pod_specimen.jpg` (*Bauhinia variegata* MHNT.BOT.2011.3.22), isolating the dehisced twisted valves, intact pre-dehisced flat pod, and helical twist crossings/lobes.
- Enforce runtime SHA-256 verification of all 4 dataset files, expand `measure_morphoeval()` to 9 quantitative reference-anchored metrics, and upgrade the figure to a 6-panel synthesis layout.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Armon et al. (2011) Science / Efrati et al. (2009) JMPS Incompatible Shell Elasticity & Chiral Seed Pod Opening
- **spec_fingerprint** (semantic model + inputs): `0721ebc9afa69617d711a2d91c13ef54394e8e1f4a186543c993ead8f2e0eebf`
- **input_dataset_sha256**: `datasets/chiral_seed_pod_opening/bauhinia_chiral_seed_pod_specimen.jpg` (`047f357961bc58c5…`), `datasets/chiral_seed_pod_opening/chiral_seed_pod_opening_reference.json` (`9ba8847c46f708bb…`), `datasets/chiral_seed_pod_opening/paper_citations.bib` (`b3a693ac6a538b04…`), `datasets/chiral_seed_pod_opening/scout_manifest.json` (`05711e5e7df6cb17…`)
- `docs/media/chiral_seed_pod_opening_comparison.png` — artifact_sha256 `565a347e18f3d082…`, 179,007 B, 1400×630 px

| measurement | value | reference | method |
|---|---:|---|---|
| Critical Föppl-von Kármán pitchfork width w_tilde_c (nu=0.30) | 2.1609 dimensionless | 2.1609 | Analytical threshold w_tilde_c = (16*(1+nu)/sqrt(1-nu^2))^(1/4) |
| Wide valve mean curvature ratio H/kappa_0 (W=1.80 cm) | 0.3309 dimensionless | 0.3304 | Analytical pitchfork branch evaluation |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2007-liang-chiral-seed-pod-opening.py` prior to Step 12 remediation.
