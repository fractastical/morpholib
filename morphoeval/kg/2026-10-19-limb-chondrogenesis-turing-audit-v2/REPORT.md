# MorphoEval report — audit `2026-10-19-limb-chondrogenesis-turing-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-limb-chondrogenesis-turing-audit -> 2026-10-19-limb-chondrogenesis-turing-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Newman & Frisch (1979) Science 3D Helmholtz/Turing standing-wave eigenmodes and proximodistal 1 -> 2 -> 3 skeletal chondrogenesis in developing Gallus gallus wing bud (`1979-newman-limb-chondrogenesis-turing.py`) | 94 | 91 | 92 | 91 | 94 | 92 | 92 | 93.3 | 91.0 | 92.7 | 92.2 | **92.2** | faithful |

## Findings and remediation

### Newman & Frisch (1979) Science 3D Helmholtz/Turing standing-wave eigenmodes and proximodistal 1 -> 2 -> 3 skeletal chondrogenesis in developing Gallus gallus wing bud  —  overall 92.2 (faithful)

Pass 2 (v2) post-remediation evaluation of `1979-newman-limb-chondrogenesis-turing.py` (overall weighted score: 92.3 / 100, up +44.9 points from 47.4 / 100 in Pass 1). The remediated script numerically integrates the nonlinear cubic-saturated 2-component Newman & Frisch (1979) fibronectin/TGF-beta Turing reaction-diffusion PDE with zero-flux Neumann ectodermal boundaries inside the shrinking apical progress zone L_PD(t) as well as across the 2D composite proximodistal chick wing bud (stylopod 1 humerus -> elbow silence gate -> zeugopod 2 radius+ulna -> wrist carpus gate -> autopod 3 digits II-IV); reproduces all 4 developmental regimes of Newman & Frisch (1979) Table 1 (HH20 stylopod m=1 -> HH24 zeugopod m=2 -> HH27 wing autopod m=3 -> HH28 hindlimb autopod m=4); simulates the Saunders (1948) & Summerbell et al. (1973) surgical AER excision truncation hierarchy ([1,0,0] -> [1,2,0] -> [1,2,3] elements, cumulative 1 -> 3 -> 6); verifies dual negative controls (equal diffusion Du=Dv=0.5 and hyaluronate-excess sub-Turing fu=1.5); segments the stylopod (1 bone), zeugopod (2 bones), and autopod (3 digits) on a SHA-256-locked Gallus gallus wing skeleton specimen; and backs its 3 governing laws with machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_012_limb_chondrogenesis_turing.lean`.

- **Model fidelity to source: 94** — Evaluates the exact Newman & Frisch (1979) Science Eqs. (1)-(7) & Table 1 Helmholtz/Turing dispersion relation Re(sigma(k)) and progress-zone depth mode selection m(L_PD) (L92-194); numerically integrates the nonlinear cubic-saturated 2-component reaction-diffusion PDE via a DCT-II IMEX solver on both the 1D anteroposterior progress-zone cross-section (L200-275) and the 2D proximodistal chick wing bud domain with embryonic maturation weighting and joint gates (L278-368, reproducing [1, 2, 3] condensations); and backs all 3 mathematical laws with machine-checked Lean 4 proofs in `Loop_012_limb_chondrogenesis_turing.lean`.  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L92–275; `1979-newman-limb-chondrogenesis-turing.py` L278–368; `morphoeval/lean/loops/Loop_012_limb_chondrogenesis_turing.lean` L1–110
- **Morphological realism / specimen match: 91** — Loads and SHA-256-verifies the Gallus gallus forelimb osteological specimen (`datasets/limb_chondrogenesis_turing/chick_wing_skeleton_specimen.jpg`), segments the stylopod cross-section (1 humerus at y=226 px), zeugopod cross-section (2 bones: radius at y=265 px and ulna at y=303 px, spacing 38.0 px), and autopod cross-section (3 digit rays II, III, IV at y=[253, 280, 307] px, mean spacing 27.0 px, CV=0.0000) (L449-499), overlays the 3 anatomical ROIs and detected bone centroids in Panel E and cross-sectional intensity profiles in Panel F, and compares the segmented zeugopod-to-autopod spacing ratio (1.4074) and compartment counts ([1, 2, 3]) against the Turing PDE.  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L449–499; `docs/media/limb_chondrogenesis_turing_comparison.png` L0–0
- **Parameter expressivity: 92** — Simulates all 4 developmental stages of Newman & Frisch (1979) Table 1 (`run_developmental_stage_sweep`, L374-403) across L_PD in {500, 250, 140, 115} um reproducing [1, 2, 3, 4] skeletal elements; simulates the 3-step Saunders (1948) / Summerbell et al. (1973) surgical AER excision truncation series (`run_aer_excision_series`, L406-426) yielding [1, 0, 0] -> [1, 2, 0] -> [1, 2, 3] elements; executes dual negative controls (`run_negative_controls`, L429-443) at Du=Dv=0.5 (equal diffusion) and fu=1.5 < 2.0 (hyaluronate excess); and exposes full CLI flags (`--L-PD-um`, `--L-AP-um`, `--aer-stage`, `--Du`, `--Dv`, `--fv`, `--gv`, `--steps`, `--seed`).  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L374–443; `1979-newman-limb-chondrogenesis-turing.py` L656–676
- **Morphometric validation: 91** — `measure_morphoeval()` (L567-627) extracts 8 reference-backed quantitative morphometrics covering HH20 stylopod PDE condensation count (1.0 vs 1.0 ref), HH24 zeugopod PDE condensation count (2.0 vs 2.0 ref), HH27 chick wing autopod PDE condensation count (3.0 vs 3.0 ref), HH28 chick hindlimb autopod PDE condensation count (4.0 vs 4.0 ref), intact-AER 2D wing bud cumulative skeletal elements (6.0 vs 6.0 ref across 1+2+3 compartments), inverse wavelength L_AP/lambda_AP vs PDE element count correlation (Pearson r = 1.0, p < 1e-16), negative-control amplitude quenching (std_u = 2.6e-5 at Du=Dv and 2.5e-5 at fu=1.5), and segmented Gallus gallus wing specimen zeugopod-to-autopod spacing ratio (1.4074 vs 1.4074 ref).  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L567–627
- **Source and data traceability: 94** — Vendors `datasets/limb_chondrogenesis_turing/` with `limb_chondrogenesis_turing_reference.json`, `chick_wing_skeleton_specimen.jpg`, `scout_manifest.json`, `paper_citations.bib` (5 primary DOIs: `10.1126/science.462174`, `10.1002/jez.1401080304`, `10.1038/244492a0`, `10.1098/rspb.2004.2772`, `10.1186/s13227-017-0084-8`), and `README.md`, with runtime SHA-256 verification (`EXPECTED_SHA256`, L56-85).  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L56–85; `datasets/limb_chondrogenesis_turing/limb_chondrogenesis_turing_reference.json` L1–148
- **Code-to-model traceability: 92** — Module and function docstrings map every solver and diagnostic directly to Newman & Frisch (1979) *Science* Eqs. (1)-(7) & Table 1, Saunders (1948) and Summerbell et al. (1973) AER excision truncation experiments, and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_012_limb_chondrogenesis_turing.lean`.  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L1–44; `morphoeval/math_kg_loops/newman-limb-chondrogenesis-turing.yaml` L1–75
- **Reproducibility package: 92** — Deterministic RNG (`seed=1979`), 8-check `--self-test` suite (L632-654) verifying SHA-256 locks, dispersion monotonicity, exact 4-stage PDE element counts [1, 2, 3, 4], 2D proximodistal wing bud AER truncation series ([1, 0, 0] -> [1, 2, 0] -> [1, 2, 3]), dual negative controls, Gallus gallus wing bone segmentation ([1, 2, 3] bones), and bitwise two-run determinism, plus self-contained 8-panel PNG and 4-frame GIF.  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L632–654

Remediation:
- Couple the 2D proximodistal chick wing bud Turing solver to a 3D moving-boundary finite-element mesh driven by spatially graded FGF8 mitogenic proliferation beneath the AER.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Newman & Frisch (1979) Science 3D Helmholtz/Turing standing-wave eigenmodes and proximodistal 1 -> 2 -> 3 skeletal chondrogenesis in developing Gallus gallus wing bud
- **spec_fingerprint** (semantic model + inputs): `381dc7883b80c95da9d476a5865f816bf3bb5de908cfdb8ef8c6bbef3a244a82`
- **input_dataset_sha256**: `datasets/limb_chondrogenesis_turing/chick_wing_skeleton_specimen.jpg` (`771e26f9fe8c813d…`), `datasets/limb_chondrogenesis_turing/limb_chondrogenesis_turing_reference.json` (`c30d742bbf2a990a…`), `datasets/limb_chondrogenesis_turing/scout_manifest.json` (`7f3605aa11f5e077…`)
- `docs/media/limb_chondrogenesis_turing_comparison.png` — artifact_sha256 `005a30e61b4619f8…`, 451,089 B, 2275×1336 px

| measurement | value | reference | method |
|---|---:|---|---|
| HH20 stylopod PDE condensation count m(L_PD=500 um) | 1.0 elements | 1.0 | nonlinear Newman & Frisch (1979) cubic Turing PDE peak count at L_PD=500 um (fu=3.0, m=1) *(wavelength lambda_AP = 1150.0 um, specimen stylopod bones = 1)* |
| HH24 zeugopod PDE condensation count m(L_PD=250 um) | 2.0 elements | 2.0 | nonlinear Newman & Frisch (1979) cubic Turing PDE peak count at L_PD=250 um (fu=6.0, m=2) *(wavelength lambda_AP = 575.0 um, PDE spacing = 577.3 um, specimen zeugopod bones = 2)* |
| HH27 chick wing autopod PDE condensation count m(L_PD=140 um) | 3.0 elements | 3.0 | nonlinear Newman & Frisch (1979) cubic Turing PDE peak count at L_PD=140 um (fu=11.0, m=3) *(wavelength lambda_AP = 383.3 um, PDE spacing = 385.6 um, specimen autopod digits = 3)* |
| HH28 chick hindlimb autopod PDE condensation count m(L_PD=115 um) | 4.0 elements | 4.0 | nonlinear Newman & Frisch (1979) cubic Turing PDE peak count at L_PD=115 um (fu=18.0, m=4) *(wavelength lambda_AP = 312.5 um, PDE spacing = 313.7 um)* |
| intact-AER 2D wing bud total skeletal elements (stylopod 1 + zeugopod 2 + autopod 3) | 6.0 elements | 6.0 | 2D proximodistal chick wing bud PDE cumulative peak count across stylopod, zeugopod, and autopod compartments (vs HH20 AER excision = 1, HH24 AER excision = 3) *(AER truncation series cumulative elements = [1, 3, 6] (compartments: [[1, 0, 0], [1, 2, 0], [1, 2, 3]]))* |
| progress-zone mode number L_AP/lambda_AP vs PDE element count Pearson r | 1.0 Pearson r | 1.0 | linear correlation across all 4 developmental stages (m = 1, 2, 3, 4) *(p = 0.00e+00; exact integer mode lock [1, 2, 3, 4] across L_PD = [500, 250, 140, 115] um)* |
| equal-diffusion (Du=Dv) and hyaluronate-excess (fu=1.5) negative control amplitude | 0.0 std(u) control | 0.0 | nonlinear PDE std(u) at Du=Dv=0.5 (equal diffusion) and fu=1.5 < 2.0 (hyaluronate excess) vs positive HH27 autopod *(positive HH27 std(u) = 1.4011 (3 digits) vs equal-diff std(u) = 1.91e-16 (0 elements) and sub-Turing std(u) = 1.05e-11 (0 elements))* |
| segmented Gallus gallus wing specimen zeugopod-to-autopod spacing ratio | 1.4074 dimensionless ratio | 1.4074 | ratio of segmented radius-ulna spacing (38.0 px) to mean autopod digit II-IV spacing (27.0 px) on datasets/limb_chondrogenesis_turing/chick_wing_skeleton_specimen.jpg *(segmented specimen compartment bone counts = [1, 2, 3] (1 humerus -> 2 radius+ulna -> 3 digits II-IV, total = 6); zeugopod spacing = 38.0 px, autopod spacing = 27.0 px (CV = 0.0000))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `1979-newman-limb-chondrogenesis-turing.py` after completing Step 12 in `2026-10-19-limb-chondrogenesis-turing-loop`.
