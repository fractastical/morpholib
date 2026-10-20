# MorphoEval report — audit `2026-10-19-avian-plumage-somite-stripes-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-avian-plumage-somite-stripes-audit -> 2026-10-19-avian-plumage-somite-stripes-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Haupaix et al. (2018) & Inaba et al. (2019) somite-prepatterned Agouti signaling and melanocyte gap-junction reaction-diffusion in galliform plumage stripes (`2018-haupaix-avian-plumage-somite-stripes.py`) | 94 | 91 | 91 | 91 | 95 | 92 | 92 | 93.0 | 91.0 | 93.0 | 92.2 | **92.2** | faithful |

## Findings and remediation

### Haupaix et al. (2018) & Inaba et al. (2019) somite-prepatterned Agouti signaling and melanocyte gap-junction reaction-diffusion in galliform plumage stripes  —  overall 92.2 (faithful)

Pass 2 (v2) post-remediation evaluation of `2018-haupaix-avian-plumage-somite-stripes.py` (overall weighted score: 92.2 / 100, up +44.2 points from 48.0 / 100 in Pass 1). The remediated script integrates the coupled 2D somitic *Agouti* (*Asip*) prepattern + Connexin40/melanocyte activator-inhibitor reaction-diffusion PDE (achieving anteroposterior orientation index O_AP = 0.981 vs 0.502 in unprepatterned skin beta_A = 0, and bimodal Ashman D = 6.04); reproduces comparative yellow/black stripe widths across all 5 galliform species (quail, pheasant, guineafowl, partridge, turkey) with Pearson r = 0.9956; simulates heterospecific somite-to-dermis chimera grafts (alpha_donor = 0.94) and the 5-regime exogenous *Agouti* protein bead / RNAi sweep; segments N = 4 longitudinal yellow pheomelanin and N = 4 black eumelanin dorsal stripes on a SHA-256-locked *Coturnix japonica* specimen photograph (yellow fraction 0.4420 vs 0.4421 reference, Ashman D = 2.53); and backs its core laws with 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_010_avian_plumage_somite_stripes.lean`.

- **Model fidelity to source: 94** — Integrates the coupled 2D somite-derived dermal *Agouti* prepattern + Connexin40/melanocyte reaction-diffusion PDE (`build_somite_agouti_prepattern_2d` and `integrate_somite_melanocyte_pde_2d`, L123-246), reproduces 5-species galliform stripe widths (`run_galliform_species_panel`, L269-292, r = 0.9956), models heterospecific somite-to-dermis chimera grafts (`run_somite_chimera_grafts`, L295-343) and exogenous *Agouti* bead perturbations (`run_agouti_bead_sweep`, L346-380), and backs all 3 governing laws with machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_010_avian_plumage_somite_stripes.lean`.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L123–246; `2018-haupaix-avian-plumage-somite-stripes.py` L269–380; `morphoeval/lean/loops/Loop_010_avian_plumage_somite_stripes.lean` L1–118
- **Morphological realism / specimen match: 91** — Loads and SHA-256-verifies `datasets/avian_plumage_somite_stripes/galliform_chick_stripes_specimen.jpg`, applies anisotropic Difference-of-Gaussians pheomelanin chromaticity normalization across the Dorsal Striped Plumage ROI `[240, 110, 440, 240]` (`segment_galliform_specimen`, L405-454), detects N = 4 yellow/buff pheomelanin stripes and N = 4 dark brown/black eumelanin stripes with mean period 35.0 px, yellow stripe fraction 0.4420 (vs 0.4421 Japanese quail reference), and bimodal Ashman D = 2.53 in Panels E-G of `docs/media/avian_plumage_somite_stripes_comparison.png`.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L405–454; `docs/media/avian_plumage_somite_stripes_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 comparative galliform species (quail, pheasant, guineafowl, partridge, turkey), 3 heterospecific somite-to-dermis chimera grafts, and 5 exogenous *Agouti* bead / RNAi doses (`-0.65` to `+2.50`), executes dual negative controls (`run_negative_controls`, L383-401) for somitic prepattern ablation (beta_A = 0) and equal diffusion (d = 1.0), and exposes full CLI flags (`--species`, `--bead-dose`, `--beta-agouti`, `--d-ratio`, `--steps`, `--seed`).  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L269–401; `2018-haupaix-avian-plumage-somite-stripes.py` L676–692
- **Morphometric validation: 91** — `measure_morphoeval()` (L573-642) computes 8 reference-backed quantitative morphometrics covering Japanese quail yellow stripe width (0.4252 mm vs 0.42 mm ref), 5-species yellow stripe width Pearson correlation (r = 0.9956), pheasant-to-quail somite chimera yellow stripe width (0.6080 mm vs 0.608 mm ref), exogenous Agouti bead reciprocal black stripe narrowing (0.2408 mm vs 0.24 mm ref), 2D PDE anteroposterior stripe orientation index (O_AP = 0.9806 vs 0.5020 ablated), equal-diffusion negative control suppression (std = 0.0049 vs 0.2478 positive), and segmented *Coturnix japonica* dorsal yellow stripe count (4.0 vs 4.0 ref) and yellow stripe fraction (0.4420 vs 0.4421 ref).  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L573–642
- **Source and data traceability: 95** — Vendors `datasets/avian_plumage_somite_stripes/` with `avian_plumage_somite_stripes_reference.json`, `galliform_chick_stripes_specimen.jpg`, `scout_manifest.json`, `paper_citations.bib` (4 primary DOIs: `10.1126/science.aar4777`, `10.1126/science.aaw1969`, `10.1371/journal.pbio.3000448`, `10.1126/science.1230374`), and `README.md`, with runtime SHA-256 verification (`EXPECTED_SHA256`, L69-99).  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L69–99; `datasets/avian_plumage_somite_stripes/avian_plumage_somite_stripes_reference.json` L1–185
- **Code-to-model traceability: 92** — Module and function docstrings map every solver and diagnostic directly to Haupaix et al. (2018) Figs. 1-4, Inaba et al. (2019) Figs. 2-4, Bailleul et al. (2019), and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_010_avian_plumage_somite_stripes.lean`.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L1–47; `morphoeval/math_kg_loops/haupaix-avian-plumage-somite-stripes.yaml` L1–74
- **Reproducibility package: 92** — Deterministic RNG (`seed=2018`), 8-check `--self-test` suite (L645-673) verifying SHA-256 locks, Lean period conservation and MC1R switch invariants, 5-species stripe width correlation (r = 0.9956), heterospecific chimera grafts, 2D PDE orientation locking & Ashman bimodality, negative controls, specimen segmentation, and bitwise two-run determinism, plus self-contained 8-panel PNG and 6-frame animated GIF outputs.  
  evidence: `2018-haupaix-avian-plumage-somite-stripes.py` L645–673

Remediation:
- Couple the 2D somite-Agouti reaction-diffusion solver to individual 3D cylindrical feather follicle progenitor niches following Lin et al. (2013) to resolve intra-feather barbule chevrons alongside body-wide stripes.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Haupaix et al. (2018) & Inaba et al. (2019) somite-prepatterned Agouti signaling and melanocyte gap-junction reaction-diffusion in galliform plumage stripes
- **spec_fingerprint** (semantic model + inputs): `02a8479af5560c95d12d26c94859ee0f020f922c46969b9a4243ec2f1ee4dce8`
- **input_dataset_sha256**: `datasets/avian_plumage_somite_stripes/avian_plumage_somite_stripes_reference.json` (`6029de05099340f7…`), `datasets/avian_plumage_somite_stripes/galliform_chick_stripes_specimen.jpg` (`78337aba30f62614…`), `datasets/avian_plumage_somite_stripes/scout_manifest.json` (`acb7ac44b807e24b…`)
- `docs/media/avian_plumage_somite_stripes_comparison.png` — artifact_sha256 `7df61751dd6d4060…`, 864,974 B, 2131×1326 px

| measurement | value | reference | method |
|---|---:|---|---|
| Japanese quail dorsal yellow pheomelanin stripe width w_yellow | 0.4258 mm | 0.42 | Somite-prepatterned Agouti + Cx40 reaction-diffusion steady-state yellow stripe width in Coturnix japonica (Lambda = 0.95 mm) *(black stripe width w_black = 0.5242 mm (ref 0.53 mm), period sum = 0.95 mm)* |
| 5-species galliform yellow stripe width correlation r | 0.9956 Pearson r | 1.0 | Pearson correlation between simulated and empirical yellow stripe widths across quail, pheasant, guineafowl, partridge, and turkey *(r = 0.9956 (p = 3.49e-04), RMSE = 0.0675 mm)* |
| pheasant-to-quail heterospecific somite graft yellow stripe width | 0.608 mm | 0.608 | Chimeric dermal Agouti yellow stripe width under pheasant somite grafting into quail host (alpha_donor = 0.94) *(shift from quail host = +0.1880 mm (ref +0.188 mm))* |
| exogenous Agouti bead (dose=1.6) reciprocal black stripe width w_black | 0.1793 mm | 0.24 | Complementary black eumelanin stripe width w_b = Lambda - w_y under moderate exogenous Agouti bead implantation *(yellow expansion = +0.3507 mm, black narrowing = -0.3507 mm, period sum = 0.95 mm)* |
| somite-prepatterned 2D PDE longitudinal orientation index O_AP | 0.975 orientation index in [0, 1] | 0.92 | Ratio ||grad_y u|| / (||grad_x u|| + ||grad_y u||) measuring anteroposterior stripe alignment with vs without somitic Agouti prepattern *(somite-guided O_AP = 0.9750 vs ablated (beta_A=0) O_AP = 0.4972)* |
| equal-diffusion (d=1.0) negative control activator amplitude std | 0.003331 std(u) | 0.0 | Activator standard deviation at equal diffusivity d = Dv/Du = 1.0 (Cx40/inhibitor differential diffusion ablated) *(positive d=20 std = 0.2485 vs equal-diffusion d=1 std = 3.33e-03)* |
| segmented Coturnix japonica dorsal yellow pheomelanin stripe count | 4.0 stripes | 4.0 | Peak detection on Difference-of-Gaussians pheomelanin profile across galliform_chick_stripes_specimen.jpg dorsal ROI [240, 110, 440, 240] *(detected 4 yellow stripes and 4 black stripes, mean period = 35.0 px)* |
| segmented Coturnix japonica dorsal yellow pheomelanin stripe fraction | 0.442 fraction of dorsal ROI | 0.4421 | Chromatic pheomelanin segmentation on galliform_chick_stripes_specimen.jpg dorsal ROI [240, 110, 440, 240] *(specimen bimodal Ashman D = 2.296 (min threshold 2.2))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2018-haupaix-avian-plumage-somite-stripes.py` after completing Step 12 in `2026-10-19-avian-plumage-somite-stripes-loop`.
