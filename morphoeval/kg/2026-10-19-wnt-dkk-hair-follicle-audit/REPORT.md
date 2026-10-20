# MorphoEval report — audit `2026-10-19-wnt-dkk-hair-follicle-audit`

Rubric **0.1.0** · checkpoint `3cb5ea9` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sick et al. (2006) WNT/DKK activator-inhibitor hair follicle spacing morphogenesis (`2002-sick-wnt-dkk-hair-follicle.py`) | 63 | 38 | 44 | 46 | 68 | 64 | 65 | 56.7 | 41.0 | 65.7 | 53.1 | **53.1** | partially faithful |

## Findings and remediation

### Sick et al. (2006) WNT/DKK activator-inhibitor hair follicle spacing morphogenesis  —  overall 53.1 (partially faithful)

Pass 1 (v1) baseline evaluation of `2002-sick-wnt-dkk-hair-follicle.py` (55.5/100). The v1 baseline implements the 2D saturating Meinhardt-Gierer WNT/DKK reaction-diffusion equations on a 56x56 periodic grid and computes the linear dispersion relation sigma(k) and critical diffusivity ratio d_c = (sqrt(2)+1)^2 = 5.8284, but omits primary-to-secondary inter-follicular wave insertion upon 2x epidermal expansion (Sick et al. 2006 Figs. 3-4), omits K14-Wnt overexpression high-density clustering and Dkk1 transgenic enlargement/blockade phenotypes, runs no equal-diffusion negative control (D_w = D_h), and performs no computer-vision segmentation of the vendored Mus musculus follicle placode specimen image.

- **Model fidelity to source: 63** — Implements the 2D saturating WNT/DKK activator-inhibitor PDE (L66-109) and linear dispersion curve (L34-55), but omits Sick et al.'s (2006) primary -> secondary inter-follicular wave insertion upon 2x epidermal expansion, omits K14-Wnt overexpression and Dkk1 transgenic phenotypes, and runs no equal-diffusion negative control (D_w = D_h).  
  evidence: `2002-sick-wnt-dkk-hair-follicle.py` L34–55; `2002-sick-wnt-dkk-hair-follicle.py` L66–109
- **Morphological realism / specimen match: 38** — Displays `datasets/wnt_dkk_hair_follicle/hair_follicle_placode_specimen.jpg` raw in Panel 3 (L126-128), performing zero computer-vision segmentation of the in situ hybridization follicle placodes or inter-placode nearest-neighbor spacing.  
  evidence: `2002-sick-wnt-dkk-hair-follicle.py` L126–128; `docs/media/wnt_dkk_hair_follicle_comparison.png` L0–0
- **Parameter expressivity: 44** — Simulates only a single wild-type parameter point (d = 40, rho_w = 0.001, rho_h = 0) with no transgenic parameter sweep across rho_w (K14-Wnt) or rho_h (Dkk1), no domain expansion sweep, and no CLI parameter overrides beyond `--self-test` (L180-188).  
  evidence: `2002-sick-wnt-dkk-hair-follicle.py` L180–188
- **Morphometric validation: 46** — `measure_morphoeval()` reports only 3 basic scalars (d_c, w_*, and wild-type placode count, L141-169) without uncertainty metrics (CV, std, RMSE, Pearson r), without negative-control contrast, and without specimen nearest-neighbor spacing or secondary wave insertion validation.  
  evidence: `2002-sick-wnt-dkk-hair-follicle.py` L141–169
- **Source and data traceability: 68** — Vendors `datasets/wnt_dkk_hair_follicle/` with `wnt_dkk_hair_follicle_reference.json`, `hair_follicle_placode_specimen.jpg`, `scout_manifest.json`, and `paper_citations.bib`, but v1 script does not verify SHA-256 digests of the dataset files at runtime or assert JSON-to-code parameter parity in `self_test()`.  
  evidence: `2002-sick-wnt-dkk-hair-follicle.py` L26–29
- **Code-to-model traceability: 64** — Docstring cites Sick et al. (2006) and Andl et al. (2002), but lacks explicit mappings to Sick et al. (2006) Figs. 2-4 (primary/secondary waves, K14-Wnt, Dkk1) and the 3 Lean 4 theorems in `Loop_006_wnt_dkk_hair_follicle.lean`.  
  evidence: `2002-sick-wnt-dkk-hair-follicle.py` L1–12
- **Reproducibility package: 65** — Includes deterministic `np.random.default_rng(2006)` and a basic `self_test()` (L172-178), but does not test bitwise two-run determinism, SHA-256 dataset locks, equal-diffusion control decay, or secondary inter-follicular wave insertion.  
  evidence: `2002-sick-wnt-dkk-hair-follicle.py` L172–178

Remediation:
- Implement Sick et al. (2006) Figs. 3-4 primary wave (E14.5) -> 2x epidermal expansion -> secondary inter-follicular wave insertion (E16.5) showing de novo placode formation at the uninhibited midpoints between primary follicles.
- Implement Sick et al. (2006) Fig. 2-4 & Andl et al. (2002) transgenic phenotype suite in 2D: (a) Wild-Type regular placode lattice, (b) K14-Wnt overexpression high-density clustered placodes (elevated rho_w), (c) saturated enlarged/ringing follicles (elevated kappa & moderate DKK), (d) Dkk1 transgenic blockade (high rho_h -> 0 placodes), and (e) equal-diffusion negative control (D_w = D_h = 0.01 -> 0 placodes).
- Implement automated computer-vision segmentation of `datasets/wnt_dkk_hair_follicle/hair_follicle_placode_specimen.jpg` (Mus musculus E12.5-E13.5 in situ hybridization placode array) to extract individual dark-purple stained follicle placode centroids, areas, and nearest-neighbor spacing CV, plus an allometric domain-length scaling sweep.
- Add runtime SHA-256 verification of all files in `datasets/wnt_dkk_hair_follicle/`, expand `measure_morphoeval()` to 7 quantitative measurements with reference values, CV/RMSE/r uncertainty, and negative-control delta, expose full CLI flags, and upgrade the comparison figure to a 9-panel publication synthesis.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sick et al. (2006) WNT/DKK activator-inhibitor hair follicle spacing morphogenesis
- **spec_fingerprint** (semantic model + inputs): `b89c42d7be6f1e2a68cf568ccc651fde825a3fca205d98fa93823f3a02d4cc25`
- **input_dataset_sha256**: `datasets/wnt_dkk_hair_follicle/hair_follicle_placode_specimen.jpg` (`f48a7d76ae5984e5…`), `datasets/wnt_dkk_hair_follicle/paper_citations.bib` (`473c1747b6bcb451…`), `datasets/wnt_dkk_hair_follicle/scout_manifest.json` (`39fe775496e89c51…`), `datasets/wnt_dkk_hair_follicle/wnt_dkk_hair_follicle_reference.json` (`e224bca31b91571c…`)
- `docs/media/wnt_dkk_hair_follicle_comparison.png` — artifact_sha256 `7c67c960f1efad08…`, 202,559 B, 1620×528 px

| measurement | value | reference | method |
|---|---:|---|---|
| WNT/DKK critical diffusivity ratio d_c = (sqrt(2)+1)^2 | 5.8284 D_h / D_w | 5.8284 | Analytical Turing-Meinhardt threshold (3 + 2*sqrt(2)) |
| Unsaturated steady-state WNT activator concentration w_* | 1.5 concentration units | 1.5 | Exact steady-state formula w_* = (c_w / c_h) * (mu_h / mu_w) |
| Wild-type 2D primary wave hair follicle placode count | 39 placodes | 16 | Connected components of WNT activator field w(x,y) > mean + 0.85*std on 56x56 grid |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2002-sick-wnt-dkk-hair-follicle.py` prior to Step 12 remediation in `2026-10-19-wnt-dkk-hair-follicle-loop`.
