# MorphoEval report — audit `2026-10-18-conus-audit`

Rubric **0.1.0** · checkpoint `fb234bc` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasProbability` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for ProbKnow template conformance). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Wolfram (1984) Rule 30 & Meinhardt–Klingler (1987) Conus textile Pigmentation (`1984-wolfram-conus-textile-rule30.py`) | 89 | 84 | 85 | 88 | 90 | 89 | 89 | 87.7 | 85.5 | 89.3 | 87.3 | **87.3** | faithful |

## Findings and remediation

### Wolfram (1984) Rule 30 & Meinhardt–Klingler (1987) Conus textile Pigmentation  —  overall 87.3 (faithful)

Side-by-side generative reproduction and 1D space-time kymograph analysis of Wolfram (1984, 2002) Rule 30 cellular automata and Meinhardt & Klingler's (1987) activator-substrate-hormone reaction-diffusion PDE mapped onto a 3D Raup (1966) Conus textile shell mesh. Validated against OEIS A051023 (128/128 bits), Erica Jen's (1990) left-bijective inverse and diagonal period-doubling theorem (0 bit errors; dislocations at k = 0, 3, 8, 29, 400), and segmented OSU Conus textile specimen photography (H/W error 0.07%, 2D silhouette IoU 0.8353, white-tent chord CV 0.7069 vs specimen 0.6403).

- **Model fidelity to source: 89** — Implements both the discrete Wolfram Rule 30 cellular automaton (in both Boolean XOR/OR and GF(2) Algebraic Normal Form) and the continuous Meinhardt-Klingler (1987) 3-species activator-substrate-hormone mantle-edge PDE, plus Erica Jen's (1990) left-bijective spatial inverse, left-cone diagonal recursion d_k(t+1) = d_{k-2}(t) XOR (d_{k-1}(t) OR d_k(t)), and asymmetric Boolean Jacobian damage-cone analysis.  
  evidence: `1984-wolfram-conus-textile-rule30.py` L140–273; `1984-wolfram-conus-textile-rule30.py` L276–422; `1984-wolfram-conus-textile-rule30.py` L627–704
- **Morphological realism / specimen match: 84** — Directly segments two Wikimedia Commons Conus textile photographs (`stjohn_osu_24399204286`, CC BY 2.0, and `ling_commons_010`, Public Domain). The 3D Raup Conus mesh matches the OSU specimen's height-to-width ratio within 0.07% (1.8342 vs 1.8330) and achieves a 2D silhouette IoU of 0.8353. Rule 30 reproduces the full-shell white-tent chord-length CV (0.7069 vs specimen 0.6403, KS = 0.2389), while the Meinhardt-Klingler PDE matches the multi-scale close-up dorsal whorl dispersion (CV = 1.1050 vs 1.1310).  
  evidence: `1984-wolfram-conus-textile-rule30.py` L837–855; `conus_textile_rule30_comparison.png` L0–0
- **Parameter expressivity: 85** — Contrasts Rule 30 against Rule 90 (linear XOR / Sierpinski negative control) and the continuous Meinhardt-Klingler PDE across intact growth and pinned mantle-edge lesion perturbations, sweeps diagonal index k across 415 left-cone diagonals and 25 damage-cone random seeds, and compares 1D space-time defect-front kymographs across four repository morphogenetic paradigms.  
  evidence: `1984-wolfram-conus-textile-rule30.py` L425–620; `docs/media/conus_rule30_kymograph_defect_fronts.png` L0–0
- **Morphometric validation: 88** — Seven quantitative morphometrics and algebraic invariants are verified in `self_test()` and exported to `conus_textile_rule30_metrics.csv` and `conus_rule30_defect_fronts.csv`: 128/128 OEIS A051023 bit match, 0 bit errors on Jen's left-bijective inverse and diagonal recursion, exact dislocation indices k in {0, 3, 8, 29, 400}, damage-cone velocities v_R = +1.0000 and v_L = -0.2604, shell H/W = 1.8342 (ref 1.8330), silhouette IoU = 0.8353, and white-chord CV = 0.7069 (theory 1/sqrt(2) = 0.7071, specimen 0.6403).  
  evidence: `1984-wolfram-conus-textile-rule30.py` L799–913; `conus_textile_rule30_metrics.csv` L0–0; `conus_rule30_defect_fronts.csv` L0–0
- **Source and data traceability: 90** — Cites Wolfram (1983, 1984, 2002), the 2019 Wolfram Rule 30 Prize, OEIS A051023, Jen (1990), Meinhardt & Klingler (1987), Raup (1966), and both Wikimedia Commons specimen photographs in `datasets/conus/paper_citations.bib`, with SHA-256 locks verified on `conus_textile_reference.json` and both specimen JPEGs.  
  evidence: `datasets/conus/paper_citations.bib` L1–95; `datasets/conus/conus_textile_reference.json` L1–80
- **Code-to-model traceability: 89** — All figure titles, legends, and annotations in both 6-panel figures are dynamically formatted from the dictionary returned by `self_test()`, preventing hardcoded label drift between computed metrics and plotted text.  
  evidence: `1984-wolfram-conus-textile-rule30.py` L973–1148; `1984-wolfram-conus-textile-rule30.py` L1285–1504
- **Reproducibility package: 89** — Deterministic RNG seeds (`42`, `100..124`, `2025`), fast `--self-test-only` entrypoint, automatic specimen verification/caching with SHA-256 checks, and complete PNG, GIF, and CSV outputs.  
  evidence: `1984-wolfram-conus-textile-rule30.py` L1507–1560; `docs/media/conus_textile_rule30.gif` L0–0

Remediation:
- Extend 3D Conus shell texture comparison from 2D orthographic silhouette IoU to unwrapped cylindrical UV registration against 360-degree photogrammetry of a single Conus textile shell.
- Include the 4-variable Meinhardt (1995) branched-travelling-wave system with explicit point-arrest hormone pulses for large triangular tent cascades.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Wolfram (1984) Rule 30 & Meinhardt–Klingler (1987) Conus textile Pigmentation
- **spec_fingerprint** (semantic model + inputs): `6baf9ffa515c0c89775df3b58f398571f7894e64d20534778dff610ad08bc313`
- **input_dataset_sha256**: `datasets/conus/conus_textile_reference.json` (`dfcb500d87a39390…`)
- `conus_rule30_defect_fronts.csv` — artifact_sha256 `ed19157163ba7735…`, 1,205 B, 
- `conus_textile_rule30_comparison.png` — artifact_sha256 `b204b836cd08e75a…`, 611,089 B, 2640×1632 px
- `conus_textile_rule30_metrics.csv` — artifact_sha256 `da34a299061268e9…`, 644 B, 
- `docs/media/conus_rule30_kymograph_defect_fronts.png` — artifact_sha256 `61ea4422526b1008…`, 509,522 B, 2688×1664 px
- `docs/media/conus_textile_rule30.gif` — artifact_sha256 `0f2462a3e14406d8…`, 590,905 B, 640×340 px, 36 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Rule 30 center-column exact match to OEIS A051023 (first 128 bits) | 128 bits matched | 128 | exact bitwise comparison of a_{t,0} (t=0..127) against OEIS A051023 |
| Jen (1990) left-bijective inverse & diagonal recursion bit errors | 0 bit errors | 0 | sum of bit reconstruction errors across a_{t,i-1} = a_{t+1,i} XOR (a_{t,i} OR a_{t,i+1}) and d_k(t+1) = d_{k-2}(t) XOR (d_{k-1}(t) OR d_k(t)) *(first 5 period-doubling dislocation diagonals k = [0, 3, 8, 29, 400] (P = 1, 2, 4, 8, 16))* |
| Rule 30 Prize #2 center-column density c_1024 | 0.4785 fraction of 1s | 0.5 | cumulative fraction (1/N) sum_{t=0}^{1023} a_{t,0} on single-seed light-cone *(normalized 4-bit block Shannon entropy H_4/4 = 0.99640 bits/bit)* |
| left-cone periodic crystal lock-in front velocity v_lock | -0.1962 cells/step | -0.2081 | spatial drift velocity of period-locked diagonal boundary (1/slope - 1 from T_lock(k) = 1.2441*k + 7.89 over k in [30, 400]) *(damage-cone velocities: v_R = +1.0000±0.0000, v_L = -0.2604±0.1024 cells/step)* |
| 3D Raup Conus textile shell height-to-width ratio (H/W) | 1.8342 ratio (H/W) | 1.833 | projected axial height / maximum shoulder diameter of 3D Raup Conus mesh (W=1.42, D=0.05, T=3.15) |
| 2D shell silhouette IoU vs segmented Conus textile specimen | 0.8353 IoU (intersection/union) | 1.0 | pixel overlap between projected 3D Raup Conus mesh and segmented OSU Conus textile specimen silhouette (256x140 grid) |
| Rule 30 white-tent chord-length CV (σ/μ) | 0.7075 CV (std/mean) | 0.6403 | coefficient of variation of horizontal unpigmented run lengths w on periodic mantle ring (230x370) *(Meinhardt-Klingler PDE CV = 1.1045 (vs close-up specimen ling_commons_010 CV = 1.1310); KS(Rule30, specimen) = 0.2623)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

First formal 7-axis MorphoEval audit of `1984-wolfram-conus-textile-rule30.py`, covering Wolfram Rule 30 (1984/2019 Prize Problems), Erica Jen's (1990) left-cone period-doubling dislocation theorem, Meinhardt & Klingler's (1987) mantle-edge activator-substrate-hormone PDE, 3D Raup Conus shell texture mapping, and 1D space-time defect-front kymograph tooling.
