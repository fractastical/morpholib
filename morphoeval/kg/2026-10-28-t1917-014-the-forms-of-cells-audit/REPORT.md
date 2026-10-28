# MorphoEval report — audit `2026-10-28-t1917-014-the-forms-of-cells-audit`

Rubric **0.1.0** · checkpoint `d762bb9c6` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-014 — Lagena outlines vs. the Plateau/Delaunay catalogue (Pass 1) (`thompson/claims/t1917-014_the-forms-of-cells.py`) | 72 | 45 | 50 | 48 | 85 | 75 | 70 | 64.7 | 46.1 | 76.3 | 60.8 | **60.8** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-014 — Lagena outlines vs. the Plateau/Delaunay catalogue (Pass 1)  —  overall 60.8 (mostly faithful)

v1 implements the Delaunay catalogue exactly (Kenmotsu E4, neck-ratio E5, period E6, catenoid), builds Thompson's 'unduloid + portion of a sphere' as a C1 composite (E7) plus a pure unduloid bead (E8), segments all 13 Lagena of Fig. 84 and fits both models. Result: 6/13 Delaunay-compatible at the 0.06 gate; flask-like specimens (L/W <= pi, n = 8) median nRMSE 0.049, spindles (L/W > pi, n = 5) median 0.209; median bilateral asymmetry 0.058. Missing: photograph of a real specimen, null and free-form comparators, statistics (KS / CV), parameter sweep, negative control, verdict file.

- **Model fidelity to source: 72** — The catalogue is implemented from the first-integral / Kenmotsu form with the correct limits (cylinder, unduloid, spheres, nodoid, catenoid), self-tested against the root identities and the elliptic-integral period, and 18 Lean theorems check the integer algebra (roots, envelope, trichotomy, bead bound, flask join). Thompson's flask qualification is honoured by the C1 composite. Deductions: the flask's sphere-unduloid join is restricted to the one-parameter family of C1 joins (no C2 matching), the neck is a plain cylinder with no apertural lip, and the nodoid is classified but never fitted.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L95–150; `thompson/claims/t1917-014_the-forms-of-cells.py` L154–184; `morphoeval/lean/loops/Thompson1917_T1917_014.lean` L32–221; `thompson/claims/t1917-014_the-forms-of-cells.py` L154–176
- **Morphological realism / specimen match: 45** — Side by side (panel D), the fitted composites trace the flask and globular Lagena closely and visibly fail on the slender spindles, which is the honest result of the test. But the only 'specimens' in Pass 1 are 1917 line drawings after Brady; the vendored CC-BY photograph (Kireenko 2022) is not analysed, so no real specimen has been compared yet.  
  evidence: `docs/media/t1917_014_the_forms_of_cells_comparison.png` L0–0; `thompson/claims/t1917-014_the-forms-of-cells.py` L187–208; `datasets/thompson_1917/t1917-014/kireenko_2022_bdj_e87457_fig8_station5656mc.jpg` L0–0
- **Parameter expressivity: 50** — B visibly and monotonically controls the neck ratio in the GIF (cylinder to string of spheres) and the flask neck length grows with L_neck; the fitted B per specimen spans 0.30-0.85. No systematic sweep over (B, t, L_neck) is tabulated, the nodoid branch (B > 1) and the catenoid are only drawn, and no CLI exposes the parameters.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L430–462; `thompson/claims/t1917-014_the-forms-of-cells.py` L260–278
- **Morphometric validation: 48** — Quantitative morphometrics are computed on every silhouette (L/W, asymmetry index, fitted B, neck ratio, nRMSE, Pearson r) and the bead aspect bound gives an a priori prediction (L/W <= pi) that is checked. There is no null model (ellipsoid), no free-form reference to calibrate the noise floor, no KS test or cross-validation, and no literature measurement of Lagena dimensions.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L211–238; `datasets/thompson_1917/t1917-014/t1917_014_metrics.json` L0–0; `thompson/claims/t1917-014_the-forms-of-cells.py` L250–258
- **Source and data traceability: 85** — The claim text, page and figure numbers are pinned to the Internet Archive leaf (0276 = p. 256) with SHA-256 of the JP2 and of the derived crop, re-verified at run time; Delaunay 1841 and Kenmotsu 1980 are cited with DOIs/URLs; the pre-scouted Dryad dataset is documented as a mismatch with pinned metadata. Deduction: Kenmotsu's DOI and Brady 1884 are cited in the .bib but not resolved against Crossref in this round.  
  evidence: `datasets/thompson_1917/t1917-014/t1917_014_the_forms_of_cells_reference.json` L0–0; `thompson/claims/t1917-014_the-forms-of-cells.py` L85–92; `datasets/thompson_1917/t1917-014/scout_manifest.json` L0–0
- **Code-to-model traceability: 75** — Every function docstring names the equation id (E3-E8) and the Thompson page it implements; the Lean module and math KG YAML cite the exact script line ranges. The decision threshold 0.06 is a stated judgement, not derived from a noise model.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L1–45; `morphoeval/math_kg_loops/thompson1917-t1917-014.yaml` L0–0; `thompson/claims/t1917-014_the-forms-of-cells.py` L72–73
- **Reproducibility package: 70** — Deterministic (no RNG), --self-test passes in 3 s, the full run in 7 s, inputs SHA-pinned, dependency versions recorded here. Deductions: no result.json verdict file, no pinned per-script requirements, PNG outputs are gitignored by the repo and had to be force-added.  
  evidence: `thompson/claims/t1917-014_the-forms-of-cells.py` L465–499; `thompson/claims/t1917-014_the-forms-of-cells.py` L502–522

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor (different provider) — this round is builder-scored'}
- {'id': 'R2', 'item': 'Segment the real specimens in the vendored Kireenko (2022) CC-BY plate (Lagena 61a/62a, Oolina 63, Fissurina 57a/58a) and fit them with the same pipeline; report sizes in um from the 100 um scale bars'}
- {'id': 'R3', 'item': 'Add the ellipsoid null model and a free-form (degree-6 r^2 polynomial) reference so the Delaunay residual is judged against the attainable noise floor, not a fixed 0.06'}
- {'id': 'R4', 'item': 'Add statistics: KS test between observed and fitted radius distributions, 2-fold cross-validated RMSE, Pearson r per model'}
- {'id': 'R5', 'item': 'Negative/ablation control: fit the Delaunay models to polythalamous tests on the same plate (Globobulimina 64a, Quinqueloculina 66a), which the claim does not cover and which should fit worse'}
- {'id': 'R6', 'item': 'Parameter-regime sweep: tabulate neck ratio, period and bead aspect ratio over B in [0, 1.5] (cylinder -> unduloid -> spheres -> nodoid) and the flask join feasibility over t'}
- {'id': 'R7', 'item': 'Write datasets/thompson_1917/t1917-014/result.json with the claim verdict (supported | refuted | mixed | untestable) and headline numbers'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-014 — Lagena outlines vs. the Plateau/Delaunay catalogue (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `ce11ca3725361e46a837bbcd24fc0d74234556b6328d06d635521b2bb17dd6e7`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-014/kireenko_2022_bdj_e87457_fig8_station5656mc.jpg` (`3b57acc0dfc98374…`), `datasets/thompson_1917/t1917-014/paper_citations.bib` (`a21c514d2fded5d2…`), `datasets/thompson_1917/t1917-014/scout_manifest.json` (`53e6b01a4f0e6102…`), `datasets/thompson_1917/t1917-014/t1917_014_the_forms_of_cells_reference.json` (`0a5ac92486643991…`), `datasets/thompson_1917/t1917-014/thompson_1917_fig84_lagena_after_brady.png` (`c02d29af384791df…`), `datasets/thompson_1917/t1917-014/thompson_1917_leaf0276_p256_fig84.jp2` (`ed440583a8b144d5…`)
- `datasets/thompson_1917/t1917-014/kireenko_2022_bdj_e87457_fig8_station5656mc.jpg` — artifact_sha256 `3b57acc0dfc98374…`, 2,933,295 B, 2480×3508 px
- `datasets/thompson_1917/t1917-014/scout_manifest.json` — artifact_sha256 `53e6b01a4f0e6102…`, 9,226 B, 
- `datasets/thompson_1917/t1917-014/t1917_014_metrics.json` — artifact_sha256 `7666625021d3e0ea…`, 11,462 B, 
- `datasets/thompson_1917/t1917-014/t1917_014_the_forms_of_cells_reference.json` — artifact_sha256 `0a5ac92486643991…`, 12,067 B, 
- `docs/media/t1917_014_the_forms_of_cells_comparison.png` — artifact_sha256 `500c908720bfbd2b…`, 634,068 B, 2200×1430 px
- `morphoeval/math_kg_loops/thompson1917-t1917-014.yaml` — artifact_sha256 `810e31a9e1b42103…`, 10,374 B, 

| measurement | value | reference | method |
|---|---:|---|---|
| Fig. 84 Lagena silhouettes segmented | 13 count | 13 | background-normalised threshold + closing + hole fill + opening, area > 3000 px *(13 specimens drawn by Thompson after Brady (1884))* |
| Delaunay-compatible specimens (best catalogue fit nRMSE <= 0.06) | 6 count | 13 | least-squares fit of E7 flask and E8 bead to the meridional radius profile *(reference = all 13 if the claim held for every figured Lagena)* |
| median best-fit RMSE / r_max | 0.06000318800002959 dimensionless | 0.06 | RMSE of the better of flask/bead over all rows of the silhouette |
| median best-fit RMSE / r_max, flask-like (L/W <= pi) | 0.049057303447845646 dimensionless | 0.06 | subset within the unduloid bead aspect bound |
| median best-fit RMSE / r_max, spindles (L/W > pi) | 0.20931565350576947 dimensionless | 0.06 | subset outside the unduloid bead aspect bound (E6) |
| median Pearson r (observed vs fitted radius) | 0.9844391764593691 dimensionless | — | np.corrcoef |
| median bilateral asymmetry index mean|r_L - r_R| / r_max | 0.058297686972462674 dimensionless | 0.0 | half-widths about the least-squares symmetry axis *(0 for a perfect surface of revolution)* |
| fitted Kenmotsu B range | 0.30-0.85 dimensionless | — | best catalogue member per specimen |

## Environment at checkpoint

- python: 3.12
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scikit-image: 0.26.0
- lean: 4.35.0

Branch cursor/t1917-014-the-forms-of-cells-8998; Pass 1 baseline committed as d762bb9c6 on top of origin/main fd43e5aba. Lean module kernel-checked with `lean morphoeval/lean/loops/Thompson1917_T1917_014.lean` (exit 0, 18 theorems, 0 sorry).
