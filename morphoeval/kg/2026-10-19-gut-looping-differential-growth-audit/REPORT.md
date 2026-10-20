# MorphoEval report — audit `2026-10-19-gut-looping-differential-growth-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/62

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Savin et al. (2011) Kirchhoff elastic rod + stretched neo-Hookean dorsal mesentery gut looping morphogenesis (`2011-savin-gut-looping-differential-growth.py`) | 58 | 35 | 48 | 56 | 74 | 68 | 68 | 54.7 | 42.9 | 70.0 | 54.5 | **54.5** | partially faithful |

## Findings and remediation

### Savin et al. (2011) Kirchhoff elastic rod + stretched neo-Hookean dorsal mesentery gut looping morphogenesis  —  overall 54.5 (partially faithful)

Pass 1 (v1) baseline evaluation of `2011-savin-gut-looping-differential-growth.py` (overall weighted score: 56.4/100). The v1 script implements Savin et al.'s (2011) annular tube inertia I_t = (pi/4)*(R_t^4 - R_i^4), characteristic length scale ell_* = (E_t*I_t / (E_m*h_m))^(1/3), and Chick E8-E16 wavelength/radius scaling, but uses a synthetic sinusoidal 3D curve rather than minimizing a discrete Kirchhoff elastic rod + neo-Hookean mesentery energy functional, omits the 4-species comparative collapse (zebra finch, quail, chick, mouse) and BMP2/Noggin/surgical-ablation controls, and does not segment the vendored biological mesentery/intestine specimen image.

- **Model fidelity to source: 58** — Computes the exact annular second moment of area I_t = (pi/4)*(R_t^4 - R_i^4) and cube-root scaling length ell_* = (E_t*I_t / (E_m*h_m))^(1/3) (L35-43), backed by 3 Lean 4 theorems in `Loop_062_gut_looping_differential_growth.lean`. However, `generate_parametric_gut_curve` (L46-65) prescribes a hardcoded 3D sinusoid rather than minimizing the coupled Kirchhoff rod bending + axial stretching + neo-Hookean mesenteric traction energy functional, and no negative ablation control (surgical mesentery dissection / epsilon = 0) is simulated.  
  evidence: `2011-savin-gut-looping-differential-growth.py` L35–43; `2011-savin-gut-looping-differential-growth.py` L46–65
- **Morphological realism / specimen match: 35** — Although `datasets/gut_looping_differential_growth/intestine_mesentery_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or plots the biological specimen image, and compares only against Chick E8-E16 tabular values without cross-species data from Zebra finch, Quail, and Mouse.  
  evidence: `2011-savin-gut-looping-differential-growth.py` L68–92; `docs/media/gut_looping_differential_growth_comparison.png` L0–0
- **Parameter expressivity: 48** — Evaluates 5 embryonic stages of Chick (E8-E16) across differential strain epsilon in [1.26, 3.72] (L68-92), but lacks the 14-point 4-species morphospace sweep (Zebra finch, Quail, Chick, Mouse), omits Nerurkar et al. (2017) BMP2 and Noggin perturbations plus surgical mesentery ablation (epsilon = 0), and exposes no CLI parameters for epsilon, E_t/E_m, R_t, h_m, or seed (L250-256).  
  evidence: `2011-savin-gut-looping-differential-growth.py` L250–256
- **Morphometric validation: 56** — `measure_morphoeval()` (L205-238) reports Chick E12 predicted vs observed wavelength and radius plus Chick E8-E16 Pearson r and RMSE, but does not measure physical rod curvature kappa(s) or zero-crossing loop wavelength from a relaxed 3D Kirchhoff rod, does not evaluate cross-species collapse statistics across all 14 vertebrate data points, and extracts zero computer-vision metrics from the specimen image.  
  evidence: `2011-savin-gut-looping-differential-growth.py` L205–238
- **Source and data traceability: 74** — Reads `datasets/gut_looping_differential_growth/gut_looping_differential_growth_reference.json` and cites Savin et al. (2011) `10.1038/nature10277` (L4-13), but does not verify SHA-256 digests of the reference JSON, BibTeX, scout manifest, or specimen image at runtime.  
  evidence: `2011-savin-gut-looping-differential-growth.py` L24–28
- **Code-to-model traceability: 68** — Functions document the inertia and cube-root scaling formulas (L35-44), but lack explicit mappings to Savin et al. (2011) Figs. 1-4, Supplementary Sec. S2-S4, Nerurkar et al. (2017) PNAS Fig. 2-4, and the 3 Lean 4 theorems in `Loop_062_gut_looping_differential_growth.lean`.  
  evidence: `2011-savin-gut-looping-differential-growth.py` L35–65
- **Reproducibility package: 68** — Includes `--self-test` and `--skip-gif` flags and deterministic execution (L241-265), but `self_test()` performs only 3 basic assertions without checking SHA-256 dataset integrity, zero-strain ablation uncoiling, cross-species scaling collapse, or two-run bitwise determinism.  
  evidence: `2011-savin-gut-looping-differential-growth.py` L241–248

Remediation:
- Replace the synthetic sinusoidal curve with a genuine 3D discrete Kirchhoff elastic rod + stretched neo-Hookean dorsal mesentery energy minimizer (L-BFGS-B relaxation of axial stretching + bending + mesenteric radial sheet traction + steric repulsion) and extract emergent loop wavelength, curvature kappa(s), and tortuosity directly from the relaxed 3D centerline.
- Evaluate the full 14-point cross-species scaling collapse across Zebra finch (Taeniopygia guttata), Quail (Coturnix japonica), Chick (Gallus gallus), and Mouse (Mus musculus) (Savin et al. 2011 Fig. 4e-f) alongside Nerurkar et al. (2017) PNAS BMP2 gain-of-function, Noggin loss-of-function, and surgical mesentery ablation (epsilon = 0) negative controls.
- Implement computer-vision segmentation of `datasets/gut_looping_differential_growth/intestine_mesentery_specimen.jpg` to segment the yellow-orange dorsal mesentery fan and pinkish-red intestinal loop ridges, quantifying empirical inter-loop spacing, CV, and loop count.
- Add runtime SHA-256 verification of all 4 files in `datasets/gut_looping_differential_growth/`, expose full CLI parameters (`--epsilon`, `--E-ratio`, `--R-t`, `--h-m`, `--seed`), and expand `self_test()` and `measure_morphoeval()` to cover all physical, cross-species, ablation, and specimen metrics.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Savin et al. (2011) Kirchhoff elastic rod + stretched neo-Hookean dorsal mesentery gut looping morphogenesis
- **spec_fingerprint** (semantic model + inputs): `643f6989deff1f1d47f576fe1e43c257f747c5e39321e10638b972d0ac0dc491`
- **input_dataset_sha256**: `datasets/gut_looping_differential_growth/gut_looping_differential_growth_reference.json` (`f73121bb4140187f…`), `datasets/gut_looping_differential_growth/intestine_mesentery_specimen.jpg` (`e12fee9ac1fb5f3d…`), `datasets/gut_looping_differential_growth/paper_citations.bib` (`21703c48df51dd52…`), `datasets/gut_looping_differential_growth/scout_manifest.json` (`0723b42b4d825087…`)
- `docs/media/gut_looping_differential_growth_comparison.png` — artifact_sha256 `a509fa903edf6a28…`, 242,127 B, 1413×1054 px

| measurement | value | reference | method |
|---|---:|---|---|
| Chick E12 predicted gut loop wavelength lambda | 3.1132 mm | 3.12 | c_lambda * (E_t * I_t / (E_m * h_m))^(1/3) at Chick E12 *(Chick E8-E16 wavelength RMSE = 0.0142 mm, Pearson r = 1.0000)* |
| Chick E12 predicted gut loop radius of curvature R | 1.1495 mm | 1.15 | c_R * (E_t * I_t / (E_m * h_m))^(1/3) at Chick E12 *(Observed vs predicted ratio lambda/R = 2.7083)* |
| Chick E8-E16 wavelength scaling Pearson r | 1.0 correlation | 1.0 | Pearson correlation between predicted c_lambda * ell_* and measured Chick E8-E16 wavelengths *(RMSE = 0.0142 mm across 5 stages)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2011-savin-gut-looping-differential-growth.py` prior to Step 12 closed-loop remediation in `2026-10-19-gut-looping-differential-growth-loop`.
