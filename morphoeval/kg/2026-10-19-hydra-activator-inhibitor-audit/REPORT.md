# MorphoEval report — audit `2026-10-19-hydra-activator-inhibitor-audit`

Rubric **0.1.0** · checkpoint `3cb5ea9` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gierer & Meinhardt (1972) Hydra vulgaris activator-inhibitor morphogenesis (`1972-gierer-hydra-activator-inhibitor.py`) | 62 | 38 | 44 | 45 | 68 | 64 | 65 | 56.0 | 40.6 | 65.7 | 52.8 | **52.8** | partially faithful |

## Findings and remediation

### Gierer & Meinhardt (1972) Hydra vulgaris activator-inhibitor morphogenesis  —  overall 52.8 (partially faithful)

Pass 1 (v1) baseline evaluation of `1972-gierer-hydra-activator-inhibitor.py` (55.0/100). The v1 baseline implements the 1D Gierer-Meinhardt (1972) Eqs. (15a)-(15b) activator-inhibitor PDE on a 60-cell apico-basal column and computes the analytical dispersion relation sigma(k) and critical diffusivity ratio d_c = (sqrt(2)+1)^2 = 5.8284, but omits surgical bisection head regeneration (Gierer & Meinhardt 1972 Fig. 3), periodic sub-hypostomal tentacle-ring morphogenesis (Meinhardt 1993), equal-diffusion negative control (D_a = D_h), allometric domain-scaling peak insertion, and automated computer-vision segmentation of the vendored Hydra vulgaris specimen image.

- **Model fidelity to source: 62** — Implements Gierer & Meinhardt (1972) Eqs. (15a)-(15b) 1D PDE with zero-flux Neumann boundary conditions (L85-134) and exact linear stability analysis (L38-82, d_c = 5.8284, a_* = 1.5, h_* = 2.25). However, v1 omits Gierer & Meinhardt's signature surgical bisection and head regeneration experiment (Fig. 3), omits periodic circular ring morphogenesis for the tentacle whorl (Meinhardt 1993), and runs no equal-diffusion negative control (D_a = D_h).  
  evidence: `1972-gierer-hydra-activator-inhibitor.py` L38–82; `1972-gierer-hydra-activator-inhibitor.py` L85–134
- **Morphological realism / specimen match: 38** — Displays `datasets/hydra_activator_inhibitor/hydra_vulgaris_specimen.jpg` in Panel 3 (L164-168), but performs zero computer-vision segmentation of the translucent Hydra body column, apical hypostome, or the 6 radiating tentacles, and does not simulate the periodic 6-tentacle ring to match the specimen.  
  evidence: `1972-gierer-hydra-activator-inhibitor.py` L164–168; `docs/media/hydra_activator_inhibitor_comparison.png` L0–0
- **Parameter expressivity: 44** — Simulates only a single parameter point (d = D_h/D_a = 30, N = 60) with no diffusivity-ratio sweep across d_c = 5.8284, no domain-length sweep L, and no CLI parameter overrides beyond `--self-test` (L227-237).  
  evidence: `1972-gierer-hydra-activator-inhibitor.py` L227–237
- **Morphometric validation: 45** — `measure_morphoeval()` reports only 3 basic scalars (d_c, a_*, and apical peak index, L190-215) without uncertainty metrics (CV, std, RMSE, Pearson r), without negative-control contrast, and without specimen tentacle count / angular spacing comparison.  
  evidence: `1972-gierer-hydra-activator-inhibitor.py` L190–215
- **Source and data traceability: 68** — Vendors `datasets/hydra_activator_inhibitor/` with `hydra_activator_inhibitor_reference.json`, `hydra_vulgaris_specimen.jpg`, `scout_manifest.json`, and `paper_citations.bib`, but v1 script does not verify SHA-256 digests of the dataset files at runtime or assert JSON-to-code parameter parity in `self_test()`.  
  evidence: `1972-gierer-hydra-activator-inhibitor.py` L34–36
- **Code-to-model traceability: 64** — Docstring cites Gierer & Meinhardt (1972) Eqs. (15a)-(15b), but lacks explicit mappings to Gierer & Meinhardt (1972) Fig. 2-3 (regeneration), Meinhardt (1993) (tentacle ring), and the 3 Lean 4 theorems in `Loop_001_hydra_activator_inhibitor.lean`.  
  evidence: `1972-gierer-hydra-activator-inhibitor.py` L1–18
- **Reproducibility package: 65** — Includes deterministic `np.random.default_rng(1972)` and a basic `self_test()` (L218-225), but does not test bitwise two-run determinism, SHA-256 dataset locks, equal-diffusion control decay, or surgical head regeneration recovery.  
  evidence: `1972-gierer-hydra-activator-inhibitor.py` L218–225

Remediation:
- Implement Gierer & Meinhardt (1972) Fig. 3 surgical bisection and head regeneration experiment (cutting the 60-cell column at x = 30 after apical peak formation and verifying autonomous regeneration of a new apical hypostome peak in the basal fragment once long-range inhibitor decays).
- Implement Meinhardt (1993) periodic circular tentacle-ring reaction-diffusion simulation (C = 50 cells, periodic BCs) producing 6 evenly spaced tentacle primordia, paired with an equal-diffusion negative control (D_a = D_h = 0.015, d = 1.0 < d_c = 5.8284) and a 6-point diffusivity-ratio sweep across d in {1.0, 3.0, 5.0, 8.0, 15.0, 30.0}.
- Implement automated computer-vision segmentation of `datasets/hydra_activator_inhibitor/hydra_vulgaris_specimen.jpg` to extract the Hydra polyp silhouette, apico-basal body axis, apical hypostome centroid, and angular positions/spacing CV of the 6 radiating tentacles, plus an allometric domain-length scaling sweep L in {20, 35, 50, 65, 80, 95} with Pearson r and RMSE.
- Add runtime SHA-256 verification of all files in `datasets/hydra_activator_inhibitor/`, expand `measure_morphoeval()` to 8 quantitative measurements with reference values, CV/RMSE/r uncertainty, and negative-control delta, expose full CLI flags, and upgrade the comparison figure to a 9-panel publication synthesis.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Gierer & Meinhardt (1972) Hydra vulgaris activator-inhibitor morphogenesis
- **spec_fingerprint** (semantic model + inputs): `c9d51c7c93ac67b33d6bf4ce3308cc6670cda412367b820e85de0adf71aa9719`
- **input_dataset_sha256**: `datasets/hydra_activator_inhibitor/hydra_activator_inhibitor_reference.json` (`f9d8a0be5130d952…`), `datasets/hydra_activator_inhibitor/hydra_vulgaris_specimen.jpg` (`007d127bcfa8501d…`), `datasets/hydra_activator_inhibitor/paper_citations.bib` (`49b5c0a5531a00fe…`), `datasets/hydra_activator_inhibitor/scout_manifest.json` (`b57d0d923283daaf…`)
- `docs/media/hydra_activator_inhibitor_comparison.png` — artifact_sha256 `72f7922212aac99c…`, 484,460 B, 1820×585 px

| measurement | value | reference | method |
|---|---:|---|---|
| Gierer-Meinhardt critical diffusivity ratio d_c = (sqrt(2)+1)^2 | 5.8284 D_h / D_a | 5.8284 | Analytical Turing-Gierer-Meinhardt threshold (sqrt(2)+1)^2 at mu_h > mu_a |
| Homogeneous steady-state activator concentration a_* (rho_0 = 0) | 1.5 concentration units | 1.5 | a_* = (c / c_prime) * (mu_h / mu_a) |
| Apical hypostome peak cell index along polar Hydra column (N=60) | 0 cell index (0=apical head) | 0 | argmax_x a(x, T) on 60-cell column with source gradient slope -0.15 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1972-gierer-hydra-activator-inhibitor.py` prior to Step 12 remediation in `2026-10-19-hydra-activator-inhibitor-loop`.
