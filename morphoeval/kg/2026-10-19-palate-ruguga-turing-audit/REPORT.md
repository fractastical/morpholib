# MorphoEval report — audit `2026-10-19-palate-ruguga-turing-audit`

Rubric **0.1.0** · checkpoint `45d4150` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/8

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Economou et al. (2012) FGF10/FGF7-SHH activator-inhibitor Turing stripe mechanism at mammalian palatal growth zones (`2012-economou-palate-ruguga-turing.py`) | 58 | 35 | 47 | 55 | 74 | 68 | 68 | 54.3 | 42.5 | 70.0 | 54.3 | **54.3** | partially faithful |

## Findings and remediation

### Economou et al. (2012) FGF10/FGF7-SHH activator-inhibitor Turing stripe mechanism at mammalian palatal growth zones  —  overall 54.3 (partially faithful)

Pass 1 (v1) baseline evaluation of `2012-economou-palate-ruguga-turing.py` (overall weighted score: 56.2/100). The v1 script constructs the 2-component FGF-SHH activator-inhibitor reaction Jacobian J_0, evaluates the Turing dispersion relation Re(sigma(q)), and integrates a static 1D domain producing 9 in-phase FGF and SHH peaks (r_AH > 0.95), but lacks a 2D growing palatal shelf PDE with a localized ruga-8 anterior growth zone driving sequential `8 -> 1..7b` stripe intercalation, omits the coronal vs sagittal surgical cuts and cyclopamine/FGF10-bead perturbations, and does not segment the vendored mammalian hard palate rugae specimen.

- **Model fidelity to source: 58** — Implements the 2x2 FGF-SHH activator-inhibitor Jacobian J_0 (L36-43), dispersion curve Re(sigma(q)) (L46-85), and 1D reaction-diffusion solver (L88-145), backed by 3 Lean 4 theorems in `Loop_008_palate_ruguga_turing.lean`. However, it integrates only a static fixed-length 1D domain without modeling 2D anteroposterior growth-zone elongation anterior to ruga 8 (`8 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 7b`) or the equal-diffusion (D_A = D_H) negative control.  
  evidence: `2012-economou-palate-ruguga-turing.py` L36–85; `2012-economou-palate-ruguga-turing.py` L88–145
- **Morphological realism / specimen match: 35** — Although `datasets/palate_ruguga_turing/palate_rugae_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or displays the mammalian hard palate transverse rugae specimen image.  
  evidence: `2012-economou-palate-ruguga-turing.py` L148–210
- **Parameter expressivity: 47** — Simulates only the wild-type 1D parameter set (L88-145) and plots tabular elongation values without simulating the 9-stage sequential growth-zone intercalation trajectory or the 6 surgical and pharmacological perturbation conditions (coronal cut across ruga-8 growth zone causing bifurcation/spots, sagittal cut preserving parallel stripes, cyclopamine SHH blockade, exogenous FGF10 bead / Spry2/4 hyperactivation, and equal-diffusion control), and exposes no CLI parameter overrides (L275-290).  
  evidence: `2012-economou-palate-ruguga-turing.py` L275–290
- **Morphometric validation: 55** — `measure_morphoeval()` (L234-267) returns only 3 baseline metrics (lambda_c, r(FGF, SHH), and 1D wild-type rugae count) without evaluating the 9-stage sequential growth-zone intercalation regression (Pearson r, p-value, RMSE), coronal vs sagittal cut stripe anisotropy / bifurcation index, cyclopamine / FGF10 perturbation counts, or computer-vision segmentation of left and right palatal shelf rugae in the biological specimen.  
  evidence: `2012-economou-palate-ruguga-turing.py` L234–267
- **Source and data traceability: 74** — Loads `datasets/palate_ruguga_turing/palate_ruguga_turing_reference.json` and cites Economou et al. (2012) `10.1038/ng.1090` (L1-11), but does not perform runtime SHA-256 verification of the 4 dataset files.  
  evidence: `2012-economou-palate-ruguga-turing.py` L32–33
- **Code-to-model traceability: 68** — Functions compute J_0, Re(sigma(q)), and 1D reaction-diffusion dynamics (L36-145), but lack explicit equation-level docstrings cross-referencing Economou et al. (2012) Figs. 1-4, Pantalacci et al. (2008), Welsh & O'Brien (2009), and the 3 Lean 4 theorems in `Loop_008_palate_ruguga_turing.lean`.  
  evidence: `2012-economou-palate-ruguga-turing.py` L36–145
- **Reproducibility package: 68** — Provides `--self-test` and `--skip-gif` flags (L275-290), but `self_test()` runs only 3 basic assertions without verifying SHA-256 dataset integrity, sequential growth-zone intercalation (`8 -> 1..7b`), coronal vs sagittal surgical cut phenotypes, cyclopamine/FGF10 perturbations, equal-diffusion negative control, specimen segmentation, or two-run bitwise determinism.  
  evidence: `2012-economou-palate-ruguga-turing.py` L270–278

Remediation:
- Implement a 1D & 2D growing palatal shelf reaction-diffusion PDE solver with a localized ruga-8 anterior growth zone that sequentially intercalates rugae in the developmental order `8 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 7b` alongside the equal-diffusion (D_A = D_H = 0.2) negative control.
- Simulate the full 6-condition surgical and pharmacological perturbation suite of Economou et al. (2012) Figs. 2-4: Wild-Type intact palate (9 transverse rugae), Sagittal surgical cut (AP-parallel incision preserving stripe orientation), Coronal surgical cut across the ruga-8 growth zone (releasing SHH inhibition to trigger ruga bifurcation and spots), Cyclopamine SHH blockade (broadened/merged rugae), Exogenous FGF10 bead / Spry2-/-;Spry4+/- FGF hyperactivation (high-density ectopic rugae/repulsion), and Equal-Diffusion negative control (0 rugae).
- Implement computer-vision DoG ridge segmentation of `datasets/palate_ruguga_turing/palate_rugae_specimen.jpg` across both the left palatal shelf (`[320, 220, 420, 1760]`, 15 rugae) and right palatal shelf (`[500, 220, 600, 1760]`, 16 rugae) to quantify empirical transverse ruga count, inter-ruga AP wavelength spacing, and CV.
- Add runtime SHA-256 verification of all 4 files in `datasets/palate_ruguga_turing/`, expose full CLI parameter overrides (`--a11`, `--a12`, `--a21`, `--a22`, `--D-ratio`, `--growth-rate`, `--seed`), and expand `measure_morphoeval()` to 9 metrics and `self_test()` to 8 comprehensive checks.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Economou et al. (2012) FGF10/FGF7-SHH activator-inhibitor Turing stripe mechanism at mammalian palatal growth zones
- **spec_fingerprint** (semantic model + inputs): `2d8b918612b8f3a25ba7f27439207d03bf556187e2f521ce737d04a76d0c39ef`
- **input_dataset_sha256**: `datasets/palate_ruguga_turing/palate_rugae_specimen.jpg` (`a38f6c053ffe9048…`), `datasets/palate_ruguga_turing/palate_ruguga_turing_reference.json` (`e76e670dd50c86fe…`), `datasets/palate_ruguga_turing/paper_citations.bib` (`fd020997fb3c2db0…`), `datasets/palate_ruguga_turing/scout_manifest.json` (`f0a345a5c9039e78…`)
- `docs/media/palate_ruguga_turing_comparison.png` — artifact_sha256 `84f9f694133a1960…`, 306,096 B, 1570×1154 px

| measurement | value | reference | method |
|---|---:|---|---|
| FGF-SHH critical Turing wavelength lambda_c | 4.553 dimensionless length | 3.2232 | Peak of 2-component FGF-SHH dispersion relation Re(sigma(q)) |
| In-phase FGF vs SHH spatial correlation r(FGF, SHH) | 0.988 Pearson r | 0.96 | Spatial Pearson correlation between 1D steady-state FGF and SHH profiles |
| 1D palatal transect wild-type rugae count | 9.0 rugae | 9.0 | 1D SHH peak detection along palatal shelf domain L=29.0 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2012-economou-palate-ruguga-turing.py` prior to Step 12 closed-loop remediation in `2026-10-19-palate-ruguga-turing-loop`.
