# MorphoEval report — audit `2026-10-19-coral-calice-septa-insertion-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/74 · supersession chain `2026-10-19-coral-calice-septa-insertion-audit -> 2026-10-19-coral-calice-septa-insertion-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Milne-Edwards & Haime (1848), Kunth (1869) & Chindapol et al. (2013) Scleractinian coral calice septal insertion & advection-diffusion morphogenesis (`2013-thomas-coral-calice-septa-insertion.py`) | 93 | 90 | 90 | 91 | 94 | 92 | 91 | 92.0 | 90.4 | 92.3 | 91.4 | **91.4** | faithful |

## Findings and remediation

### Milne-Edwards & Haime (1848), Kunth (1869) & Chindapol et al. (2013) Scleractinian coral calice septal insertion & advection-diffusion morphogenesis  —  overall 91.4 (faithful)

Pass 2 (v2) remediated evaluation of `2013-thomas-coral-calice-septa-insertion.py` (consensus score 91.6/100, up from 52.4/100 in Pass 1). Implements the Milne-Edwards & Haime (1848) hexameral septal insertion law N_hex(k) = 6 * 2^(k-1) across cycles k=1..6 with decreasing cycle radial reach alpha_k, Kunth's Law (1869) Paleozoic Rugosa serial 4-fossula insertion N_rugosa(m) = 6 + 4*m, Chindapol et al. (2013) Péclet advection-diffusion boundary-layer symmetry breaking across Pe in {0, 1, 4, 16, 36} (with Pe=0 pure-diffusion symmetry control A_1=0.0000 and r=0.9964 vs sqrt(Pe)), 7-species cross-taxa Scleractinian scaling (r=0.9988, log2 slope=1.0000), and automated polar ridge segmentation of n=40 radiating septa (9.00 deg spacing, major/minor reach ratio 1.4368) on the SHA-256-locked Caryophyllia quadragenaria calice specimen, backed by 4 machine-checked Lean 4 theorems.

- **Model fidelity to source: 93** — Faithfully implements Milne-Edwards & Haime (1848) hexameral septal doubling N_hex(k) = 6 * 2^(k-1) triggered at perimeter threshold 2*pi*R_k = N_hex(k)*lambda_s, decreasing radial reach L_k/R = alpha_k, Kunth's Law (1869) Paleozoic Rugosa serial 4-fossula insertion N_rugosa(m) = 6 + 4*m, and Chindapol et al. (2013) Péclet advection-diffusion boundary-layer flux modulation with a Pe = 0 pure-diffusion radial symmetry negative control. Backed by 4 machine-checked Lean 4 theorems in `Loop_074_coral_calice_septa_insertion.lean`.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L137–348; `2013-thomas-coral-calice-septa-insertion.py` L354–414; `morphoeval/lean/loops/Loop_074_coral_calice_septa_insertion.lean` L21–152
- **Morphological realism / specimen match: 90** — Vendors, SHA-256-verifies, and performs automated polar-coordinate background-subtracted ridge segmentation (`segment_caryophyllia_specimen`, L445-518) on the *Caryophyllia (Caryophyllia) quadragenaria* calice micrograph (`datasets/coral_calice_septa_insertion/scleractinian_coral_septa_specimen.jpg`), detecting n=40 radiating septa (mean angular spacing 9.00 deg matching 360/40 = 9.0 deg, CV=0.165) and quantifying major (S1+S2, L/R=0.4741) vs minor (S3+S4, L/R=0.3299) inward radial reach ratio 1.4368.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L445–518; `docs/media/coral_calice_septa_insertion_comparison.png` L0–0
- **Parameter expressivity: 90** — Simulates Scleractinian cyclic insertion across 5 developmental cycles (R=0.55 to 8.20 mm, N=6 to 96 septa), contrasts against Paleozoic Rugosa stages m=0..4 (N=6 to 22 septa), executes a 5-point Péclet advection-diffusion sweep across Pe in {0, 1, 4, 16, 36}, evaluates 7 Scleractinian species across 6 families, and exposes `--R-mm`, `--lambda-s-mm`, `--Pe`, `--seed`, and `--skip-gif` CLI flags.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L396–441; `2013-thomas-coral-calice-septa-insertion.py` L856–886
- **Morphometric validation: 91** — Exposes 9 reference-anchored morphometric measurements in `measure_morphoeval()` covering 7-species septa correlation (r=0.9988), log2 cycle doubling slope (1.0000), Scleractinia-to-Rugosa septa ratio (2.6667), primary-to-quaternary reach ratio (2.5161), Pe=0 dipolar asymmetry negative control (0.0000), 5-point Pe sweep correlation with sqrt(Pe) (r=0.9964), and 3 specimen segmentation morphometrics (40 septa, 9.00 deg spacing, 1.4368 major/minor reach ratio), achieving quick_gate = 100/100.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L715–804
- **Source and data traceability: 94** — Vendors `datasets/coral_calice_septa_insertion/` with `coral_calice_septa_insertion_reference.json`, `scleractinian_coral_septa_specimen.jpg` (CC BY 3.0, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (5 DOIs: `10.5962/bhl.title.11574`, `10.1007/978-3-662-04339-4`, `10.1371/journal.pcbi.1002849`, `10.1127/zdgg/21/1869/647`, `10.3897/zookeys.227.3612`), and `README.md`, verified at runtime by `verify_dataset_integrity()` (L102-128).  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L88–134; `datasets/coral_calice_septa_insertion/coral_calice_septa_insertion_reference.json` L1–130
- **Code-to-model traceability: 92** — Module and function docstrings map every equation and computational step directly to Milne-Edwards & Haime (1848), Kunth (1869), Kaandorp & Kübler (2001), Chindapol et al. (2013) PLoS Comput. Biol., Cairns & Kitahara (2012) ZooKeys, and the 4 machine-checked Lean 4 theorems in `Loop_074_coral_calice_septa_insertion.lean`.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L1–68
- **Reproducibility package: 91** — Deterministic execution (`seed=2013`), 8-check `self_test()` suite (L807-853) verifying SHA-256 dataset provenance, JSON-to-code parity, Milne-Edwards doubling, Kunth Rugosa divergence, Pe=0 radial symmetry control, 7-species scaling, Caryophyllia specimen segmentation parity, and bitwise two-run determinism (`np.allclose(..., atol=0.0)`), plus synchronized 6-panel PNG and 2-panel co-evolving Scleractinia-vs-Rugosa growth GIF.  
  evidence: `2013-thomas-coral-calice-septa-insertion.py` L807–853

Remediation:
- Extend the 2D corallite calice simulator to a 3D accreting turbinate/trochoid corallum mesh resolving internal endothecal dissepiments and synapticulae alongside radial septa.
- Couple the boundary-layer Péclet flux model to oscillatory wave surge (Keulegan-Carpenter number) to compare unidirectional current polarization against bidirectional reef surge.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Milne-Edwards & Haime (1848), Kunth (1869) & Chindapol et al. (2013) Scleractinian coral calice septal insertion & advection-diffusion morphogenesis
- **spec_fingerprint** (semantic model + inputs): `838f1452734ffec4b95f30098de1ca90f5c6e961cce22175354f682fa6865b61`
- **input_dataset_sha256**: `datasets/coral_calice_septa_insertion/coral_calice_septa_insertion_reference.json` (`9984d41d12285bd0…`), `datasets/coral_calice_septa_insertion/scleractinian_coral_septa_specimen.jpg` (`904035c936843361…`), `datasets/coral_calice_septa_insertion/scout_manifest.json` (`325761c7cb0d58db…`)
- `docs/media/coral_calice_septa_insertion_comparison.png` — artifact_sha256 `d46e2020411ac23d…`, 1,000,915 B, 2415×1551 px

| measurement | value | reference | method |
|---|---:|---|---|
| 7-species Scleractinian Milne-Edwards septa count correlation | 0.9988 Pearson r | 0.9985 | Correlation of predicted Milne-Edwards septa N_hex(k)=6*2^(k-1) vs observed mean septa across 7 Scleractinian species (p=9.99e-08) *(Exact log2 doubling slope = 1.0000 septa/cycle)* |
| Milne-Edwards hexameral log2 doubling slope d(log2(N/6))/dk across cycles k=1..5 | 1.0 doublings/cycle | 1.0 | OLS regression slope of log2(N_hex(k)/6) vs (k - 1) across S1..S6 *(S4 septa = 48, S5 septa = 96)* |
| Cycle k=4 Scleractinia (N=48) to Paleozoic Rugosa stage m=3 (N=18) septa ratio | 2.6667 ratio (N_hex(4) / N_rugosa(3)) | 2.6667 | Ratio of simulated Scleractinian 4th-cycle septa (48) to Kunth's Law Rugosa stage m=3 septa (6 + 4*3 = 18) *(Scleractinia N_hex(4)=48 vs Rugosa N_rugosa(3)=18)* |
| Primary-to-quaternary septal radial reach hierarchy ratio alpha_1 / alpha_4 | 2.5161 ratio (alpha_1 / alpha_4) | 2.5161 | Ratio of primary protosepta inward radial reach (alpha_1=0.78) to quaternary cycle reach (alpha_4=0.31) in simulated calice *(Strictly decreasing reach hierarchy alpha_1 > alpha_2 > alpha_3 > alpha_4 > alpha_5)* |
| Hydrostatic pure-diffusion control (Pe=0) upstream-downstream dipolar flux asymmetry | 0.0 asymmetry index A_1 | 0.0 | Upstream-minus-downstream normalized flux asymmetry (j_up - j_down)/(j_up + j_down) at Pe = 0 (hydrostatic negative control) *(Exact radial symmetry preserved at Pe = 0)* |
| 5-point Péclet sweep dipolar flux asymmetry correlation with sqrt(Pe) (Chindapol et al. 2013) | 0.9964 Pearson r | 0.999 | Correlation of dipolar calcification flux asymmetry A_1(Pe) with boundary-layer thinning factor sqrt(Pe) across Pe in {0, 1, 4, 16, 36} (p=2.54e-04) *(Pe=36 upstream/downstream flux ratio = 13.7297)* |
| Specimen segmented outer radiating septa count (scleractinian_coral_septa_specimen.jpg) | 40.0 septa | 40.0 | Automated polar-coordinate background-subtracted ridge peak detection on Caryophyllia quadragenaria calice micrograph (R=204.5 px) *(Mean angular spacing = 9.00 deg (CV=0.1699))* |
| Specimen inter-septal angular spacing delta_theta (scleractinian_coral_septa_specimen.jpg) | 9.0 deg | 9.0 | Mean angular spacing across 40 segmented radiating septa around the 360 deg calice perimeter *(Angular spacing CV = 0.1699)* |
| Specimen major (S1+S2) to minor (S3+S4) septal radial reach ratio (scleractinian_coral_septa_specimen.jpg) | 1.4368 ratio (L_major / L_minor) | 1.4368 | Ratio of segmented inward radial reach for major septa (0.4741 R) vs minor septa (0.3299 R) in Caryophyllia quadragenaria micrograph *(Major reach = 0.4741 R vs Minor reach = 0.3299 R)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2013-thomas-coral-calice-septa-insertion.py` superseding `2026-10-19-coral-calice-septa-insertion-audit`.
