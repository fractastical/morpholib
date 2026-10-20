# MorphoEval report — audit `2026-10-19-cortical-gyrification-folding-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/61

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Richman et al. (1975) & Tallinen et al. (2014, 2016) neo-Hookean constrained cortical expansion and sulcal creasing instability (`1975-richman-cortical-gyrification-buckling.py`) | 57 | 35 | 48 | 56 | 74 | 68 | 68 | 54.0 | 42.9 | 70.0 | 54.4 | **54.4** | partially faithful |

## Findings and remediation

### Richman et al. (1975) & Tallinen et al. (2014, 2016) neo-Hookean constrained cortical expansion and sulcal creasing instability  —  overall 54.4 (partially faithful)

Pass 1 (v1) baseline evaluation of `1975-richman-cortical-gyrification-buckling.py` (overall weighted score: 56.2/100). The v1 script implements Richman et al.'s (1975) and Tallinen et al.'s (2014) cube-root sulcal wavelength law lambda = 2*pi*h*(mu_c/(3*mu_s))^(1/3) and evaluates human fetal stages GW22-GW38, but prescribes a synthetic harmonic contour rather than minimizing a coupled neo-Hookean cortical shell + subcortical white-matter elastic energy functional, omits the 11-species mammalian scaling collapse and pathological polymicrogyria / lissencephaly / zero-growth controls, and does not segment the vendored coronal human cerebral cortex specimen image.

- **Model fidelity to source: 57** — Computes the exact Richman-Tallinen sulcal wavelength scaling lambda = 2*pi*h*(mu_c/(3*mu_s))^(1/3) and enforces the critical threshold g_c = 1.18 (L34-44), backed by 3 Lean 4 theorems in `Loop_061_cortical_gyrification_folding.lean`. However, `generate_parametric_coronal_contour` (L46-66) uses a synthetic rectified-sine ansatz rather than relaxing a coupled 2D neo-Hookean cortical shell + soft white-matter core energy functional with sulcal self-contact, and omits polymicrogyric, lissencephalic, and zero-growth (g=1.0) controls.  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L34–44; `1975-richman-cortical-gyrification-buckling.py` L46–66
- **Morphological realism / specimen match: 35** — Although `datasets/cortical_gyrification_folding/cerebral_cortex_gyri_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or displays the coronal brain specimen image, and compares only against human fetal GW22-GW38 tabular numbers without the 11-species mammalian dataset.  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L69–94; `docs/media/cortical_gyrification_folding_comparison.png` L0–0
- **Parameter expressivity: 48** — Evaluates 5 human fetal stages (GW22-GW38, g in [1.12, 2.15], L69-94), but omits the 11-species mammalian morphospace sweep, lacks Richman et al.'s (1975) polymicrogyria (h=0.85 mm) vs lissencephaly (h=6.50 mm) vs normogyria (h=2.60 mm) comparison, and exposes no CLI overrides for g, h, mu_c/mu_s, R_brain, or seed (L198-206).  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L198–206
- **Morphometric validation: 56** — `measure_morphoeval()` (L156-188) reports 3 analytical human fetal metrics, but does not compute true contour arc-length perimeter P_pial / P_hull from a relaxed elastic cortical shell, does not evaluate 11-species mammalian wavelength or GI correlations, and extracts zero computer-vision metrics from the coronal brain slice specimen.  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L156–188
- **Source and data traceability: 74** — Reads `datasets/cortical_gyrification_folding/cortical_gyrification_folding_reference.json` and cites Richman et al. (1975) and Tallinen et al. (2014, 2016) (L4-11), but does not perform runtime SHA-256 verification of the 4 dataset files.  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L37–38
- **Code-to-model traceability: 68** — Documents the sulcal wavelength formula (L40-44), but lacks explicit mappings to Richman et al. (1975) Table 1, Tallinen et al. (2014) Eq. 1 & Fig. 1-4, Tallinen et al. (2016) Fig. 2-4, and the 3 Lean 4 theorems in `Loop_061_cortical_gyrification_folding.lean`.  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L40–66
- **Reproducibility package: 68** — Includes `--self-test` and `--skip-gif` flags (L190-209), but `self_test()` runs only 3 basic assertions without checking SHA-256 dataset integrity, lissencephalic/ablation invariants, 11-species mammalian collapse, or two-run bitwise determinism.  
  evidence: `1975-richman-cortical-gyrification-buckling.py` L190–196

Remediation:
- Replace the parametric coronal curve with a genuine 2D neo-Hookean multiplicative-growth (F = F_e G) cortical shell + soft subcortical white-matter core elastic energy minimizer (L-BFGS-B relaxation of tangential compression, bending stiffness B_c ~ mu_c h^3, white-matter radial/shear foundation stiffness, and sulcal self-contact) and measure emergent GI = P_pial / P_hull and sulcal wavelength directly from the relaxed contour.
- Evaluate the full 11-species mammalian comparative scaling collapse (Mouse, Rat, Ferret, Marmoset, Cat, Bottlenose dolphin, Capuchin, Rhesus macaque, Sheep, Chimpanzee, Human) alongside Richman et al. (1975) polymicrogyria (h = 0.85 mm), lissencephaly (h = 6.50 mm, g < g_c), and zero-growth (g = 1.00) negative controls.
- Implement computer-vision segmentation of `datasets/cortical_gyrification_folding/cerebral_cortex_gyri_specimen.jpg` to isolate the cortical gray-matter ribbon, white-matter core, outer convex hull perimeter P_hull, pial perimeter P_pial, empirical specimen Gyrification Index GI, and inter-sulcal wavelength CV.
- Add runtime SHA-256 verification of all 4 files in `datasets/cortical_gyrification_folding/`, expose full CLI parameters (`--growth-g`, `--h-cortex`, `--mu-ratio`, `--R-brain`, `--seed`), and expand `self_test()` and `measure_morphoeval()` to 9 quantitative reference-anchored metrics.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Richman et al. (1975) & Tallinen et al. (2014, 2016) neo-Hookean constrained cortical expansion and sulcal creasing instability
- **spec_fingerprint** (semantic model + inputs): `b7b9a6e07287630e2e4e6277f4842e9ba61e2c8da24f1f7145e6787809fe3f06`
- **input_dataset_sha256**: `datasets/cortical_gyrification_folding/cerebral_cortex_gyri_specimen.jpg` (`ff9a25e993fd5aad…`), `datasets/cortical_gyrification_folding/cortical_gyrification_folding_reference.json` (`cee775390f805000…`), `datasets/cortical_gyrification_folding/paper_citations.bib` (`56bc1d0beedebe37…`), `datasets/cortical_gyrification_folding/scout_manifest.json` (`7f5410ff8ad2c1ee…`)
- `docs/media/cortical_gyrification_folding_comparison.png` — artifact_sha256 `a3439bd2af0d3988…`, 141,810 B, 1430×1105 px

| measurement | value | reference | method |
|---|---:|---|---|
| Human GW38 predicted sulcal wavelength lambda_pred | 11.3269 mm | 11.34 | Analytical Richman-Tallinen scaling lambda = 2*pi*h*(mu_c/(3*mu_s))^(1/3) at h=2.60 mm |
| Human GW38 predicted gyrification index GI_pred | 2.5591 P_pial / P_hull | 2.56 | Parametric coronal folding power law at GW38 (g=2.15) |
| Human fetal GW26-GW38 sulcal wavelength correlation | 1.0 Pearson r | 1.0 | Linear correlation across GW26-GW38 fetal stages |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `1975-richman-cortical-gyrification-buckling.py` prior to Step 12 closed-loop remediation in `2026-10-19-cortical-gyrification-folding-loop`.
