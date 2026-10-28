# MorphoEval report — audit `2026-10-28-t1917-018-the-forms-of-tissues-or-cell-a-audit`

Rubric **0.1.0** · checkpoint `88f1e6cf4` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-018 — Plateau tri-junction (120 degree) claim: Lami balance, periodic froth relaxation, junction-angle statistics (`thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py`) | 55 | 18 | 32 | 30 | 58 | 62 | 52 | 47.3 | 22.5 | 56.7 | 40.2 | **40.2** | partially faithful |

## Findings and remediation

### Thompson (1917) T1917-018 — Plateau tri-junction (120 degree) claim: Lami balance, periodic froth relaxation, junction-angle statistics  —  overall 40.2 (partially faithful)

Pass 1 (v1) evaluation (overall 40.2/100). The baseline implements Lami balance with Thompson's three Fig. 113 cases, the Steiner-vs-cross inequality, the angle-sum identity statistics, and a periodic Voronoi froth relaxed under a Farhadifar-type energy at a single parameter point (relaxed junction-angle SD 19.0 deg vs 29.5 deg for the Voronoi seed), with a 3-panel figure and a relaxation GIF. It does NOT load, SHA-256-verify, or segment any of the four vendored specimens, does not sweep tension anisotropy or the partition-tension ratio, has no negative/ablation control beyond the unrelaxed seed, computes no goodness-of-fit, writes no result.json, and its froth relaxation is unconverged (SD 19 deg is far from the Plateau limit) because perimeter contractility makes edge tensions unequal and no T1 exchanges are allowed.

- **Model fidelity to source: 55** — `lami_angles_from_tensions` (L61-75) implements the triangle-of-forces law exactly as Thompson states it (pp. 296-297) and `thompson_fig113_cases` (L78-84) reproduces cases (1)-(3) (90 / 100 / 120 deg); `steiner_vs_cross_lengths` (L87-89) encodes Lamarle's four-fold instability. However the simulated froth (L182-220) uses Farhadifar's Gamma_bar = 0.04 perimeter term so edge tensions are unequal (not the equal-tension soap froth Thompson invokes), runs a fixed 400 steps with no convergence check and no T1 transitions, and the single-regime result (SD 19 deg) therefore mis-represents the Plateau limit it is meant to embody.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L61–89; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L182–220
- **Morphological realism / specimen match: 18** — Four specimens are vendored and SHA-256-pinned in `datasets/thompson_1917/t1917-018/` (corneal endothelium, onion epidermis, Thompson Figs. 103 and 124) but v1 never opens any of them: `render_outputs` (L243-309) draws a Lami curve, a synthetic froth and a synthetic histogram only. There is no side-by-side with a real epithelium or with Thompson's own figures.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L243–309; `docs/media/t1917_018_the_forms_of_tissues_or_cell_a_comparison.png` L0–0
- **Parameter expressivity: 32** — The Lami curve sweeps the partition-tension ratio t/T continuously from 0 to 2 (L258-271), which is genuine expressivity for E1. The froth model, however, is run at a single (Lambda_bar, Gamma_bar) = (0.12, 0.04) point; the tension-anisotropy parameter eps of E5 (the regime that distinguishes Thompson's Fig. 103 from Fig. 124) is not implemented at all, and the CLI exposes only `--self-test` (L374-377).  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L258–271; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L374–377
- **Morphometric validation: 30** — `plateau_statistics` (L93-105) computes the right dispersion statistics (SD, RMS deviation, median |theta-120|, fraction within +/-15 deg) and `measure_morphoeval` (L312-345) reports seven quantities, but every one of them is measured on the synthetic froth or the Lami formula; nothing is measured on a specimen, so there is no r, RMSE, KS test or confidence interval against real data and no comparison with the stated a-priori tolerance on a real tissue.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L93–105; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L312–345
- **Source and data traceability: 58** — The module docstring (L1-35) cites Thompson's chapter, pages, figures, the IA identifier and page-to-leaf rule, and Farhadifar et al. (2007); the dataset directory carries a complete reference.json, bib file and scout manifest with every digest. But the script does not load `t1917_018_the_forms_of_tissues_or_cell_a_reference.json` or verify any SHA-256 at runtime, so the provenance lives beside the code rather than in it.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L1–35; `datasets/thompson_1917/t1917-018/t1917_018_the_forms_of_tissues_or_cell_a_reference.json` L1–20
- **Code-to-model traceability: 62** — Functions map one-to-one onto the equation labels E1/E2/E3/E5 used in the reference package and the six Lean theorems of `Thompson1917_T1917_018.lean` (Fig. 113 cases, Lami iff equal tensions, angle-sum identity, Euler bookkeeping, hexagon closure, Steiner inequality) are machine-checked with 0 sorry; `math_kg_loops/thompson1917-t1917-018.yaml` links each law to code loci. Some of those loci name functions that only exist in the planned v2 (`tensions_from_angles`, `polygon_class_distribution`, `square_lattice_control`).  
  evidence: `morphoeval/lean/loops/Thompson1917_T1917_018.lean` L39–207; `morphoeval/math_kg_loops/thompson1917-t1917-018.yaml` L37–143
- **Reproducibility package: 52** — Deterministic `np.random.default_rng(1917)` seed, a `--self-test` with nine assertions including a determinism check (L348-371), and outputs rendered to fixed paths. Missing: runtime SHA-256 verification of inputs, a machine-readable result.json, any CLI parameters, and a recorded convergence criterion for the relaxation.  
  evidence: `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L348–371; `thompson/claims/t1917-018_the-forms-of-tissues-or-cell-aggregates.py` L374–386

Remediation:
- Load and SHA-256-verify all four vendored specimens against reference.json; segment the cell-border network of the corneal endothelium and onion epidermis (border channel -> watershed from nucleus/centre markers) and binarise Thompson's Figs. 103 and 124; detect tri-cellular vertices from label images and measure the three interior angles per vertex by circular label sampling at a stated radius; report f3, SD, RMS deviation, median |theta-120| and fraction within the a-priori +/-15 deg tolerance per specimen, with bootstrap CIs.
- Implement the equal-tension soap-froth limit (Gamma_bar = 0) with a convergence criterion and T1 edge exchanges, add the orientation-dependent line tension Lambda(phi) = Lambda0 (1 + eps cos 2 phi) of E5 and sweep eps over 0..0.6 (7 points) to show monotone angle dispersion (Fig. 124 regime); add a partition-tension-ratio sweep of E1; expose --eps, --lambda-bar, --gamma-bar, --seed, --steps and --quick CLI flags.
- Add negative/ablation controls (Poisson-Voronoi null with matched vertex count, square/brick lattice with four-fold vertices, shuffled-angle null) and goodness-of-fit between the fitted-eps froth angle distribution and each specimen (Pearson r and RMSE of binned histograms, two-sample KS statistic, coefficient of variation); write datasets/thompson_1917/t1917-018/result.json with the verdict and headline numbers, and extend measure_morphoeval() with specimen-side measurements.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-018 — Plateau tri-junction (120 degree) claim: Lami balance, periodic froth relaxation, junction-angle statistics
- **spec_fingerprint** (semantic model + inputs): `19f6f11c004310d9d3822d28b8f9f71011ff87e8a31986733b91a36effb75d46`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-018/allium_cepa_epidermis_specimen.jpg` (`12d2cff76378adae…`), `datasets/thompson_1917/t1917-018/human_corneal_endothelium_specimen.jpg` (`9daed5fb546e11bf…`), `datasets/thompson_1917/t1917-018/ongrowthform1917thom_0318.jp2` (`fb329d9ddbe640d8…`), `datasets/thompson_1917/t1917-018/ongrowthform1917thom_0341.jp2` (`38d6d1a9bc86ad90…`), `datasets/thompson_1917/t1917-018/paper_citations.bib` (`ef63468ff8403071…`), `datasets/thompson_1917/t1917-018/scout_manifest.json` (`50b930d12b0f97e0…`), `datasets/thompson_1917/t1917-018/t1917_018_the_forms_of_tissues_or_cell_a_reference.json` (`213d7c371e7f98fe…`), `datasets/thompson_1917/t1917-018/thompson1917_fig103_maize_parenchyma.png` (`d1c3c3aa92d21321…`), `datasets/thompson_1917/t1917-018/thompson1917_fig124_girardia_epidermis.png` (`3a33c7b1b82aad70…`)
- `docs/media/t1917_018_the_forms_of_tissues_or_cell_a_comparison.png` — artifact_sha256 `694426affc0cfe18…`, 180,777 B, 1650×528 px

| measurement | value | reference | method |
|---|---:|---|---|
| Lami equal-tension junction angle | 120.0 deg | 120.0 | cos(alpha) = (T^2 - T^2 - T^2)/(2 T T) with T1 = T2 = T3 (E1) |
| Lami negligible-partition angle | 90.0 deg | 90.0 | E1 with t -> 0 (Fig. 113 case 1) |
| Steiner / cross length ratio (unit square) | 0.9659 dimensionless | 0.9659 | (1 + sqrt 3) / (2 sqrt 2) (E2) |
| relaxed froth junction-angle SD | 18.9948 deg | — | pooled SD of all tri-junction angles after 400 gradient steps of E5 (seed 1917) |
| Poisson-Voronoi seed junction-angle SD (negative control) | 29.4814 deg | — | pooled SD of tri-junction angles of the unrelaxed Voronoi seed |
| relaxed froth fraction of angles within +/-15 deg of 120 | 0.6806 fraction | — | |theta - 120| <= 15 over all junction angles |
| relaxed froth three-fold vertex fraction | 1.0 fraction | 1.0 | vertices with exactly three incident edges / all vertices |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scikit-image: 0.26.0

Pass 1 (v1) evaluation of the baseline script before Pass 2 remediation. The claim is a dated Thompson (1917) claim validated against Thompson's own figures and real specimens, not a new-model scout; the pre-scouted Dryad dataset (doi:10.5061/dryad.mn8cs) was verified as a DATASET_MISMATCH (Ca2+ ratio imaging, no junction label) and is not used.
