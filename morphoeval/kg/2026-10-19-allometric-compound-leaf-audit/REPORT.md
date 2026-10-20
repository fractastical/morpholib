# MorphoEval report — audit `2026-10-19-allometric-compound-leaf-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/37

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bilsborough et al. (2011), Barkoulas et al. (2008), Vlad et al. (2014) & Kierzkowski et al. (2019) growing leaf margin PIN1-Auxin-CUC2-RCO feedback loop and serration vs compound leaflet dissection (`2011-przemyslaw-allometric-compound-leaf.py`) | 55 | 45 | 52 | 54 | 68 | 58 | 54 | 54.0 | 48.4 | 59.7 | 53.5 | **53.5** | partially faithful |

## Findings and remediation

### Bilsborough et al. (2011), Barkoulas et al. (2008), Vlad et al. (2014) & Kierzkowski et al. (2019) growing leaf margin PIN1-Auxin-CUC2-RCO feedback loop and serration vs compound leaflet dissection  —  overall 53.5 (partially faithful)

Pass 1 (v1) baseline evaluation of `2011-przemyslaw-allometric-compound-leaf.py` (weighted score: 54.4/100). The Pass 1 script integrates only a static 1D periodic ring of 48 margin cells for wild-type and `cuc2` loss-of-function without a growing 1D/2D leaf blade contour PDE, without intercalary tooth/leaflet insertion when inter-peak arc-length exceeds critical wavelength lambda_c, without the full 6-genotype morphospace comparison (`cuc2-1`, `cuc2-1D` unrepressed uniform CUC2, *Arabidopsis* WT Col-0, `mir164a-4`, *Cardamine* `rco` mutant, and *Cardamine hirsuta* WT compound leaflets with RCO sinus growth repression), and without empirical CV segmentation of the vendored *Cardamine hirsuta* specimen photograph.

- **Model fidelity to source: 55** — Integrates a static 1D ring of 48 epidermal cells with up-the-gradient PIN1 polarization and Auxin-CUC2 mutual antagonism (L22-55), but omits the growing 2D leaf margin contour PDE v_n(s) = v_0 [A(s)^m/(K_g^m+A(s)^m)] (1 - eta C(s)), intercalary tooth/leaflet insertion at critical wavelength lambda_c, unrepressed ectopic `cuc2-1D` control, and *Cardamine hirsuta* RCO sinus growth repression.  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L22–55
- **Morphological realism / specimen match: 45** — Vendors `datasets/allometric_compound_leaf/cardamine_hirsuta_compound_leaf_specimen.jpg` on disk, but Pass 1 does not load, segment, or extract leaflet lobe count, dissection perimeter ratio, or radial tip-to-sinus dissection index I_D from the specimen image.  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L68–111
- **Parameter expressivity: 52** — Evaluates only gamma_cuc2 = 2.4 and 0.0 on a static ring without sweeping the 6-genotype *Arabidopsis*/*Cardamine* morphospace, growth stages P3-P6, or exposing CLI parameter overrides.  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L58–65
- **Morphometric validation: 54** — Exposes only 3 scalar numbers in `measure_morphoeval()` without reference comparisons, statistical p-values, dissection depth indices I_D, intercalary wavelength scaling, or specimen leaflet morphometrics.  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L58–65
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `allometric_compound_leaf_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/allometric_compound_leaf/allometric_compound_leaf_reference.json` L1–148
- **Code-to-model traceability: 58** — Module docstring cites Bilsborough et al. (2011), but lacks explicit equation-to-code and Lean 4 theorem mappings (`Loop_037_allometric_compound_leaf.lean`).  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L1–5
- **Reproducibility package: 54** — Includes a 1-assertion `--self-test`, but lacks SHA-256 dataset integrity verification, `cuc2-1D` ectopic unrepressed negative control, and multi-law invariant checks.  
  evidence: `2011-przemyslaw-allometric-compound-leaf.py` L114–118

Remediation:
- Implement a growing 1D/2D leaf margin contour PDE solver coupling Loop 1 (CUC2-promoted PIN1 up-the-gradient polarization rho_{ij} propto (1 + gamma_CUC2 C_i) exp(beta A_j)), Loop 2 (Auxin transcriptional repression of CUC2, C_i = C_max / (1 + (A_i/K_A)^n)), and Loop 3 (differential normal margin outgrowth v_n(s) = v_0 [A(s)^m/(K_g^m + A(s)^m)] (1 - eta_sinus C(s))) with intercalary cell division and new auxin peak insertion whenever inter-peak arc-length exceeds critical wavelength lambda_c across developmental stages P3-P6.
- Simulate the full 6-genotype morphospace panel: `cuc2-1` loss-of-function smooth margin (C = 0), `cuc2-1D` unrepressed uniform CUC2 negative control (C = C_0), *Arabidopsis thaliana* Col-0 wild-type shallow serration (eta = 0.42), `mir164a-4` deep serration (eta = 0.65), *Cardamine hirsuta* `rco` mutant simplified lobes (eta = 0.48), and *Cardamine hirsuta* wild-type pinnate compound leaflets with RCO sinus growth repression (eta = 0.88).
- Segment the SHA-256-verified *Cardamine hirsuta* compound leaf specimen (`cardamine_hirsuta_compound_leaf_specimen.jpg`) to extract foliage area fraction, 32 compound leaflet lobes, perimeter dissection ratio P/(2*sqrt(pi*A)) = 4.0939, and radial tip-to-sinus dissection depth index I_D = 0.5202, and expose an 8-check deterministic `--self-test` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bilsborough et al. (2011), Barkoulas et al. (2008), Vlad et al. (2014) & Kierzkowski et al. (2019) growing leaf margin PIN1-Auxin-CUC2-RCO feedback loop and serration vs compound leaflet dissection
- **spec_fingerprint** (semantic model + inputs): `f7e327f643238b6973cf1043ff1d25ce8c9a027859b313633da11bdc256da7cd`
- **input_dataset_sha256**: `datasets/allometric_compound_leaf/allometric_compound_leaf_reference.json` (`fa4f9b265dd83379…`), `datasets/allometric_compound_leaf/cardamine_hirsuta_compound_leaf_specimen.jpg` (`1c268fecb3660b31…`), `datasets/allometric_compound_leaf/paper_citations.bib` (`8245ac5ad5a33db0…`), `datasets/allometric_compound_leaf/scout_manifest.json` (`46ea1539febdb312…`)
- `docs/media/allometric_compound_leaf_comparison.png` — artifact_sha256 `f3c1906ac6a9bf0f…`, 86,936 B, 1540×630 px

| measurement | value | reference | method |
|---|---:|---|---|
| wt_auxin_contrast | 0.3055141114033264 auxin_units | — | 1D static ring WT auxin contrast |
| cuc2_auxin_contrast | 0.30551411140321183 auxin_units | — | 1D static ring cuc2=0 auxin contrast |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2011-przemyslaw-allometric-compound-leaf.py` prior to Step 12 closed-loop remediation.
