# MorphoEval report — audit `2026-10-19-auxin-canalization-leaf-veins-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/34

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Mitchison (1980, 1981) & Feugier, Mochizuki & Iwasa (2005) with-the-flux auxin canalization, Sachs wound bypass regeneration, and open vs closed leaf venation loops (`1997-mitchison-canalization-auxin-flux.py`) | 56 | 46 | 52 | 54 | 68 | 58 | 54 | 54.7 | 49.0 | 59.7 | 53.9 | **53.9** | partially faithful |

## Findings and remediation

### Mitchison (1980, 1981) & Feugier, Mochizuki & Iwasa (2005) with-the-flux auxin canalization, Sachs wound bypass regeneration, and open vs closed leaf venation loops  —  overall 53.9 (partially faithful)

Pass 1 (v1) baseline evaluation of `1997-mitchison-canalization-auxin-flux.py` (weighted score: 54.5/100). The Pass 1 script integrates only a minimal 2-channel ODE model at p=1 and p=2 without a 2D cellular leaf lamina graph, vein concentration drainage verification (S_vein < S_parenchyma), Sachs (1981) transverse stem incision bypass regeneration, Feugier et al. (2005) open-tree vs closed-areole venation loop topology, or segmentation of the cleared dicot leaf venation specimen.

- **Model fidelity to source: 56** — Integrates a 2-channel Mitchison ODE for p=1 and p=2 (L21-48), but omits 2D cellular lamina PDE/network integration, explicit proof and numerical verification of concentration drainage inside the canalized strand (S_vein < S_parenchyma), Sachs (1981) transverse stem wound bypass regeneration, and Feugier et al. (2005) PIN1 carrier-allocation open-tree vs closed-areole loop bifurcation.  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L21–48
- **Morphological realism / specimen match: 46** — Vendors `datasets/auxin_canalization_leaf_veins/leaf_venation_canalization_specimen.jpg` on disk, but Pass 1 does not load, segment, skeletonize, or extract secondary vein branching angles or closed areole loops from the cleared leaf specimen.  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L71–108
- **Parameter expressivity: 52** — Evaluates only p=1.0 and p=2.0 in a 2-channel toy system without sweeping the feedback exponent p across [0.5, 2.5], stem incision geometries, NPA polar transport inhibition, or open vs closed carrier-allocation regimes.  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L51–68
- **Morphometric validation: 54** — Exposes only 2 scalar measurements in `measure_morphoeval()` without statistical p-values, vein concentration drop ratios, wound bypass tortuosity, Betti-1 loop counts, or specimen venation morphometrics.  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L51–68
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `auxin_canalization_leaf_veins_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/auxin_canalization_leaf_veins/auxin_canalization_leaf_veins_reference.json` L1–130
- **Code-to-model traceability: 58** — Module docstring cites Mitchison (1980), but lacks explicit equation-to-code and Lean 4 theorem mappings.  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L1–10
- **Reproducibility package: 54** — Generates PNG and GIF outputs deterministically, but lacks a `--self-test` suite, SHA-256 integrity checks, and NPA transport inhibitor negative controls.  
  evidence: `1997-mitchison-canalization-auxin-flux.py` L71–112

Remediation:
- Implement a 2D cellular leaf lamina canalization PDE/network solver sweeping feedback exponent p in [0.5, 2.5], demonstrating diffuse flow for p <= 1 and 1-cell-wide winner-take-all vascular strands with S_vein < S_parenchyma for p = 2.
- Implement Sachs (1969, 1981) transverse stem incision wound-bypass simulation on a 2D cellular epicotyl grid, plus an NPA polar transport inhibitor negative control.
- Implement Feugier, Mochizuki & Iwasa (2005) PIN1 carrier-allocation dynamics comparing open hierarchical branching trees (Betti-1 = 0) against closed-areole venation loops (Betti-1 > 0).
- Segment and skeletonize the vendored cleared dicot leaf venation specimen (`leaf_venation_canalization_specimen.jpg`) to extract the midrib, secondary vein branching angles, and closed polygonal areole loops, with SHA-256 verification and `--self-test` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Mitchison (1980, 1981) & Feugier, Mochizuki & Iwasa (2005) with-the-flux auxin canalization, Sachs wound bypass regeneration, and open vs closed leaf venation loops
- **spec_fingerprint** (semantic model + inputs): `b1e95723b4cbd5f2bb79b0150dbb4ab3fc046d7542c48fe2c7644bb6175993d9`
- **input_dataset_sha256**: `datasets/auxin_canalization_leaf_veins/auxin_canalization_leaf_veins_reference.json` (`55e0f743ada83918…`), `datasets/auxin_canalization_leaf_veins/leaf_venation_canalization_specimen.jpg` (`7f67774708f8e952…`), `datasets/auxin_canalization_leaf_veins/paper_citations.bib` (`95f1b35cf42a8ae7…`), `datasets/auxin_canalization_leaf_veins/scout_manifest.json` (`94a1d6a85fe649c5…`)
- `docs/media/auxin_canalization_leaf_veins_comparison.png` — artifact_sha256 `565e6e4b22ecd61b…`, 58,343 B, 1200×504 px

| measurement | value | reference | method |
|---|---:|---|---|
| linear_p1_winner_flux_share | 0.5404 fraction | — | 2-channel Mitchison ODE at p=1.0 |
| quadratic_p2_winner_flux_share | 0.9904 fraction | — | 2-channel Mitchison ODE at p=2.0 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1997-mitchison-canalization-auxin-flux.py` prior to Step 12 closed-loop remediation.
