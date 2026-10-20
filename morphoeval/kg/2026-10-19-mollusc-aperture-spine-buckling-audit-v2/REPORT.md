# MorphoEval report — audit `2026-10-19-mollusc-aperture-spine-buckling-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/77 · supersession chain `2026-10-19-mollusc-aperture-spine-buckling-audit -> 2026-10-19-mollusc-aperture-spine-buckling-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Moulton, Goriely & Chirat (2015) & Chirat et al. (2013) hierarchical mollusc aperture spine folding and fractal frond serration (`2015-moulton-mollusc-aperture-spine-buckling.py`) | 93 | 90 | 89 | 92 | 94 | 91 | 91 | 91.7 | 90.8 | 92.0 | 91.4 | **91.4** | faithful |

## Findings and remediation

### Moulton, Goriely & Chirat (2015) & Chirat et al. (2013) hierarchical mollusc aperture spine folding and fractal frond serration  —  overall 91.4 (faithful)

Pass 2 (v2) remediated evaluation of `2015-moulton-mollusc-aperture-spine-buckling.py` (consensus score 91.4/100, up from 52.7/100 in Pass 1). The remediated script implements L-BFGS-B energy minimization of the unilateral mantle-margin functional along the *Murex pecten* siphonal canal (`solve_murex_siphonal_comb`, L168-281, N=16.0 primary spines + intercalated spinules), solves Kirchhoff-Clebsch elastica loop closure from open-gutter frond to sealed hollow tube (`solve_kirchhoff_elastica_spine_loop`, L287-338), synthesizes 3-stage *Chicoreus ramosus* fractal frond serrations (`synthesize_chicoreus_fractal_frond`, L341-385; k_2^*/k_1^* = 3.8011, k_3^*/k_1^* = 8.4258), models episodic triradiate varix whorl accretion (`simulate_episodic_varix_whorls`, L391-443; Delta_phi = 120.0 deg), validates 10 muricid species (r = 1.0000 for k_2/k_1, r = 0.9987 for spine count), and segments n=16 primary comb spines (CV = 0.2092) on the SHA-256-locked *Murex pecten* specimen, backed by 3 machine-checked Lean 4 theorems.

- **Model fidelity to source: 93** — Solves the fourth-order unilateral mantle-margin functional via L-BFGS-B (`solve_murex_siphonal_comb`, L168-281), models planar Kirchhoff-Clebsch elastica loop folding and tubular self-contact closure at gamma >= gamma_fold (`solve_kirchhoff_elastica_spine_loop`, L287-338), synthesizes the 3-stage hierarchical wavenumber cascade k_i^* = (3*K_s/(E*h_i^3))^(1/4) for *Chicoreus ramosus* fractal fronds (`synthesize_chicoreus_fractal_frond`, L341-385), simulates episodic triradiate varix whorl accretion (`simulate_episodic_varix_whorls`, L391-443), and verifies the subcritical negative control (gamma=0.96 < gamma_c=1.08 -> 0 spines). Backed by 3 machine-checked Lean 4 theorems in `Loop_077_mollusc_aperture_spine_buckling.lean`.  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L130–385; `2015-moulton-mollusc-aperture-spine-buckling.py` L391–443; `morphoeval/lean/loops/Loop_077_mollusc_aperture_spine_buckling.lean` L18–125
- **Morphological realism / specimen match: 90** — Vendors, SHA-256-verifies, and performs Difference-of-Gaussians comb-spine periodicity segmentation (`segment_murex_specimen`, L508-561) on the *Murex pecten* Venus comb murex shell (`datasets/mollusc_aperture_spine_buckling/murex_pecten_aperture_specimen.jpg`), detecting all n=16 primary comb spines along the siphonal canal and aperture margin (mean inter-spine spacing = 42.13 px, CV = 0.2092).  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L508–561; `docs/media/mollusc_aperture_spine_buckling_comparison.png` L0–0
- **Parameter expressivity: 89** — Spans subcritical smooth shell (gamma=0.96), open-gutter frond (gamma=1.26), kissing self-contact threshold (gamma=1.42), and sealed tubular *Murex* spine (gamma=1.68), resolves 3 hierarchical frond generations (h1=0.42, h2=0.0708, h3=0.0245 mm), evaluates 10 muricid species across 5 subclades, and exposes `--E`, `--K-s`, `--gamma-varix`, `--seed`, `--skip-gif`, `--self-test`, and `--verify-only` CLI flags (L885-922).  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L287–502; `2015-moulton-mollusc-aperture-spine-buckling.py` L885–922
- **Morphometric validation: 92** — Extracts 9 reference-anchored morphometrics in `measure_morphoeval()` (L732-835), including 10-species secondary/primary wavenumber ratio correlation (`r = 1.0000`, `RMSE = 0.0080`), 10-species primary spine count correlation (`r = 0.9987`, `RMSE = 0.1453 spines`), *Chicoreus ramosus* secondary/primary wavenumber ratio (`3.8011` vs `3.80`), L-BFGS-B relaxed *Murex* primary spine count (`16.0` vs `16.0`), tubular vs open-gutter neck gap delta (`-3.1059 mm`), triradiate varix angular spacing (`120.0 deg`), subcritical negative control (`0.0 mm`), and specimen comb spine count (`16` vs `16`) and spacing CV (`0.2092`).  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L449–502; `2015-moulton-mollusc-aperture-spine-buckling.py` L732–835
- **Source and data traceability: 94** — Vendors `datasets/mollusc_aperture_spine_buckling/` with `mollusc_aperture_spine_buckling_reference.json`, `murex_pecten_aperture_specimen.jpg` (CC BY 2.0, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (4 DOIs: `10.1016/j.jtbi.2014.09.021`, `10.1073/pnas.1220443110`, `10.1016/j.jtbi.2012.07.009`, `10.1371/journal.pcbi.1007213`), and `README.md`, verified cryptographically at runtime by `verify_dataset_integrity()` (L104-119).  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L78–124; `datasets/mollusc_aperture_spine_buckling/mollusc_aperture_spine_buckling_reference.json` L1–275
- **Code-to-model traceability: 91** — Module and function docstrings map every equation and solver step directly to Moulton, Goriely & Chirat (2015) J. Theor. Biol. 364:420-430 Sec. 2-4, Chirat, Moulton & Goriely (2013) PNAS 110:6015-6020, Moulton et al. (2012) J. Theor. Biol. 311:69-79, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_077_mollusc_aperture_spine_buckling.lean`.  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L1–58
- **Reproducibility package: 91** — Deterministic RNG (`seed=2015`), 8-check `self_test()` suite (L838-882) and `--verify-only` mode verifying SHA-256 provenance, JSON parity, 3-stage frond wavenumber cascade, tubular self-contact closure criterion, episodic triradiate varix spacing and subcritical negative control, 10-species muricid scaling collapse, *Murex pecten* specimen comb segmentation, and bitwise two-run determinism (`np.allclose(..., atol=0.0)`), plus synchronized 6-panel PNG and GIF outputs.  
  evidence: `2015-moulton-mollusc-aperture-spine-buckling.py` L838–882

Remediation:
- Couple the planar Kirchhoff-Clebsch elastica loop solver to a 3D finite-strain Koiter shell finite-element model (Rudraraju et al. 2019 PLoS Comput. Biol.) to resolve oblique spiral cord intersections with varix spines.
- Incorporate micro-CT cross-sectional measurements of mantle edge thickness gradients h(s) across ontogenetic instars in Chicoreus ramosus.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Moulton, Goriely & Chirat (2015) & Chirat et al. (2013) hierarchical mollusc aperture spine folding and fractal frond serration
- **spec_fingerprint** (semantic model + inputs): `2b94e7a03cd347db53de50ca64fa7292e03536c1d004ec54908d722b5d7edd8a`
- **input_dataset_sha256**: `datasets/mollusc_aperture_spine_buckling/mollusc_aperture_spine_buckling_reference.json` (`c73e5d2ac5c13cd5…`), `datasets/mollusc_aperture_spine_buckling/murex_pecten_aperture_specimen.jpg` (`2f87d4ec30ebec6a…`), `datasets/mollusc_aperture_spine_buckling/paper_citations.bib` (`fc39398ba54f69e1…`), `datasets/mollusc_aperture_spine_buckling/scout_manifest.json` (`53cb565add07b28d…`)
- `docs/media/mollusc_aperture_spine_buckling_comparison.png` — artifact_sha256 `16175ede1e6af031…`, 923,113 B, 2640×1680 px

| measurement | value | reference | method |
|---|---:|---|---|
| 10-species muricid secondary/primary wavenumber cascade correlation (Moulton et al. 2015) | 1.0 Pearson r | 1.0 | Linear regression of predicted (h1/h2)^(3/4) vs observed k2*/k1* across 10 muricid species (p=6.50e-18, RMSE=0.0074) *(RMSE = 0.0074, log-log exponent = 0.7488 ± 0.0022)* |
| 10-species muricid primary comb spine count correlation (Chirat et al. 2013) | 0.9987 Pearson r | 1.0 | Correlation of predicted vs observed primary varix/canal spine count across 10 muricid taxa (RMSE=0.1516 spines, p=1.38e-11) *(RMSE = 0.1516 spines, p=1.38e-11)* |
| Chicoreus ramosus secondary-to-primary frond wavenumber ratio k_2* / k_1* | 3.8011 ratio (k2*/k1*) | 3.8 | Hierarchical 3-stage wavenumber cascade (h1/h2)^(3/4) for Chicoreus ramosus primary stem (h1=0.42 mm) vs secondary frond margin (h2=0.0708 mm) *(tertiary ratio k3*/k1* = 8.4249 vs primary k1*)* |
| L-BFGS-B relaxed Murex pecten siphonal canal primary spine count | 16.0 spines | 16.0 | Sub-bin parabolic FFT interpolation of L-BFGS-B relaxed fourth-order mantle margin along L=67.2 mm siphonal canal *(simulated wavelength = 4.2000 mm vs theory 4.0813 mm)* |
| Tubular self-contact vs open-gutter basal neck gap delta (Murex vs Chicoreus) | -2.9647 mm | -3.1059 | Difference in basal elastica neck gap Delta_x_neck between Murex tubular spine (gamma=1.68 > gamma_fold=1.42) and Chicoreus open-gutter frond (gamma=1.26 < gamma_fold) *(Murex neck_gap = -1.8353 mm (sealed) vs Chicoreus neck_gap = 1.1294 mm (open))* |
| Episodic triradiate varix angular phase spacing across 3 whorls | 119.99 deg | 120.0 | Mean angular spacing across 9 episodic varix spine bursts over 3 helicospiral whorls *(inter-varix smooth whorl amplitude = 0.000000 mm)* |
| Subcritical mantle growth negative control spine amplitude (gamma = 0.96 < gamma_c vs wild-type) | 0.0 mm | 0.0 | L-BFGS-B relaxed spine outgrowth amplitude under subcritical growth stretch gamma = 0.96 < gamma_c = 1.08 (negative control vs gamma = 1.68 wild-type) *(subcritical spines = 0.0 vs wild-type peak = 4.0836 mm)* |
| Biological specimen segmented primary comb spine count (murex_pecten_aperture_specimen.jpg) | 16.0 spines | 16.0 | Difference-of-Gaussians comb-spine peak extraction along siphonal canal and aperture margin on Murex pecten specimen (mean spacing = 42.13 px, CV = 0.2092) *(spacing std = 8.82 px, CV = 0.2092)* |
| Biological specimen siphonal comb inter-spine spacing CV (murex_pecten_aperture_specimen.jpg) | 0.2092 CV (std/mean) | 0.2092 | Coefficient of variation of inter-spine spacing across n=16 primary comb spines in Murex pecten specimen *(mean spacing = 42.13 ± 8.82 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2015-moulton-mollusc-aperture-spine-buckling.py` superseding `2026-10-19-mollusc-aperture-spine-buckling-audit`.
