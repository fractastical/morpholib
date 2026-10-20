# MorphoEval report — audit `2026-10-19-root-auxin-reflux-fountain-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/38

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Grieneisen et al. (2007) Nature & Mähönen et al. (2014) Nature reverse-fountain root tip auxin reflux capacitor, Quiescent Center exponential accumulation, and PLETHORA meristem zonality (`2014-grieneisen-root-auxin-reflux-fountain.py`) | 55 | 45 | 52 | 54 | 68 | 58 | 54 | 54.0 | 48.4 | 59.7 | 53.5 | **53.5** | partially faithful |

## Findings and remediation

### Grieneisen et al. (2007) Nature & Mähönen et al. (2014) Nature reverse-fountain root tip auxin reflux capacitor, Quiescent Center exponential accumulation, and PLETHORA meristem zonality  —  overall 53.5 (partially faithful)

Pass 1 (v1) baseline evaluation of `2014-grieneisen-root-auxin-reflux-fountain.py` (weighted score: 54.4/100). The Pass 1 script implements only a 1D 2-stream (stele + epidermis) finite-difference toy model at WT and k_lat = 0 without the full 2D 5-tissue root tip cellular lattice (stele, pericycle/endodermis, cortex, epidermis/LRC, and columella/QC), without Mähönen et al. (2014) PLETHORA (PLT1/2) transcription factor synthesis/diffusion and Meristem/Elongation/Differentiation zone boundary positioning, without the shoot-decapitation autonomy test (J_in = 0), pin2/eir1 mutant, NPA polar transport blockade, or QC laser-ablation proximal regeneration shift, and without empirical CV segmentation of the vendored Arabidopsis primary root meristem confocal micrograph.

- **Model fidelity to source: 55** — Integrates a simplified 1D 2-file advection-diffusion loop (L24-66), omitting the 2D 5-tissue radial-axial root tip cellular lattice, Mähönen et al. (2014) PLETHORA (PLT1/2) dosage gradient PDE and zone boundary thresholds (Theta_MZ, Theta_EZ), shoot-decapitation capacitor autonomy (J_in = 0), and QC laser-ablation proximal regeneration.  
  evidence: `2014-grieneisen-root-auxin-reflux-fountain.py` L24–66
- **Morphological realism / specimen match: 45** — Vendors `datasets/root_auxin_reflux_fountain/arabidopsis_root_meristem_specimen.jpg` on disk, but Pass 1 does not load, segment, or extract meristematic nuclear files, mitotic division counts, or axial nuclear density profiles from the confocal specimen.  
  evidence: `2014-grieneisen-root-auxin-reflux-fountain.py` L78–118
- **Parameter expressivity: 52** — Evaluates only WT (k_lat=0.18) and k_lat=0 without sweeping the 6 canonical experimental perturbations (WT, shoot decapitation, pin2/eir1, lateral-PIN ablation, NPA blockade, QC laser ablation) or exposing CLI flags.  
  evidence: `2014-grieneisen-root-auxin-reflux-fountain.py` L69–75
- **Morphometric validation: 54** — Exposes only 2 scalar values in `measure_morphoeval()` without reference comparisons, Pearson r/p-values, counter-current flux conservation error, shoot-decapitation retention, PLT meristem zone boundaries, or specimen confocal morphometrics.  
  evidence: `2014-grieneisen-root-auxin-reflux-fountain.py` L69–75
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `root_auxin_reflux_fountain_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/root_auxin_reflux_fountain/root_auxin_reflux_fountain_reference.json` L1–145
- **Code-to-model traceability: 58** — Module docstring cites Grieneisen et al. (2007) and Mähönen et al. (2014), but lacks explicit equation-to-code and Lean 4 theorem mappings (`Loop_038_root_auxin_reflux_fountain.lean`).  
  evidence: `2014-grieneisen-root-auxin-reflux-fountain.py` L1–7
- **Reproducibility package: 54** — Includes a minimal 1-assertion `--self-test`, but lacks SHA-256 dataset verification, NPA and shoot-decapitation controls, and multi-law invariant checks.  
  evidence: `2014-grieneisen-root-auxin-reflux-fountain.py` L121–126

Remediation:
- Upgrade to a 2D 5-tissue root tip cellular lattice solver (stele, pericycle/endodermis, cortex, epidermis/LRC, and columella/QC) integrating Grieneisen et al. (2007) anisotropic PIN1/PIN2/PIN3/PIN4/PIN7 polar transport and verifying exact vascular-epidermal counter-current flux conservation.
- Couple the 2D auxin reflux capacitor to Mähönen et al. (2014) PLETHORA (PLT1/2) transcription factor synthesis, intercellular movement, and growth dilution to predict Meristematic Zone (z_MZ), Elongation Zone (z_EZ), and Differentiation Zone boundaries across all 6 experimental regimes (Wild-Type, shoot decapitation J_in=0 autonomy, pin2/eir1, lateral-PIN ablation, NPA blockade, and QC laser-ablation proximal regeneration).
- Segment the SHA-256-verified 3-panel Arabidopsis thaliana primary root meristem confocal micrograph (`arabidopsis_root_meristem_specimen.jpg`) to extract root meristem area fraction, DAPI nuclear file count (113), Alexa Fluor 488 mitotic division foci (125), intercellular nuclear spacing, and apical-to-proximal nuclear density ratio.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Grieneisen et al. (2007) Nature & Mähönen et al. (2014) Nature reverse-fountain root tip auxin reflux capacitor, Quiescent Center exponential accumulation, and PLETHORA meristem zonality
- **spec_fingerprint** (semantic model + inputs): `5f9ff984dd3572c3a2bb1124d68b38a94c8ae92664796e5dd367404918a3bd5f`
- **input_dataset_sha256**: `datasets/root_auxin_reflux_fountain/arabidopsis_root_meristem_specimen.jpg` (`1b962c21e75c8e72…`), `datasets/root_auxin_reflux_fountain/paper_citations.bib` (`069ef06b9ad6bdfb…`), `datasets/root_auxin_reflux_fountain/root_auxin_reflux_fountain_reference.json` (`1fc68ce109094fc5…`), `datasets/root_auxin_reflux_fountain/scout_manifest.json` (`285fff11a61df708…`)
- `docs/media/root_auxin_reflux_fountain_comparison.png` — artifact_sha256 `24bbb9e18ef0ef8c…`, 60,812 B, 1400×588 px

| measurement | value | reference | method |
|---|---:|---|---|
| wt_qc_amplification_ratio | 7.5784 ratio | — | Pass 1 1D two-stream stele-epidermis reflux peak-to-shoot ratio |
| no_lat_qc_amplification_ratio | 1.0 ratio | — | Pass 1 1D stele-epidermis ratio at k_lat = 0 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2014-grieneisen-root-auxin-reflux-fountain.py` prior to Step 12 closed-loop remediation.
