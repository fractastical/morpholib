# MorphoEval report — audit `2026-10-28-t1917-024-on-concretions-spicules-and-sp-audit`

Rubric **0.1.0** · checkpoint `0643f406e28a6ba3f9600caef4642142fc8be064` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) spicule rays as Plateau borders of the secreting cell field vs Grantia SEM ray angles (Pass 1) (`thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py`) | 72 | 58 | 45 | 48 | 82 | 70 | 66 | 63.0 | 54.2 | 72.4 | 62.3 | **62.3** | mostly faithful |

## Findings and remediation

### Thompson (1917) spicule rays as Plateau borders of the secreting cell field vs Grantia SEM ray angles (Pass 1)  —  overall 62.3 (mostly faithful)

Neumann tension triangle (E1-E3) with the equiangular point (E4), the one-parameter sagittal family cos(theta_pair) = -rho/2 (E5), tetrahedral/orthogonal limits (E7/E8) and the curvilinear interstice (E9); six Lean theorems kernel-checked. Radial-arc ray extraction on the SEM plate finds 8 three-ray spicules and 3 monaxons (self-tested). Angles: all-angle SD 15.5 deg; closest-pair difference median 4.2 deg (mean 5.7); unpaired deviation mean 27.0 deg; ratio 4.7; fitted rho spans 0.37 (T-shaped atrial tetractine) to 1.43 (Y-shaped cortical triactine); 3/8 spicules equiangular within 10 deg. Verdict is withheld until Pass 2 adds a null control: in Pass 1 the small paired difference has no reference distribution. Overall (weighted) 62.3/100.

- **Model fidelity to source: 72** — The force balance, law of cosines, equiangular point, sagittal family and its inverse, tetrahedral and orthogonal limits and the interstice area are implemented exactly as Thompson states them and six Lean theorems check the algebra (equiangular iff c = m via the factorisation 2cm(c-m)(c+2m); ordering and T-limit; tetrahedral by decide). Deductions: Thompson's own mechanism is adsorption/deposition along Plateau borders of a cell FOAM — the script never simulates a foam or a cell doublet, it only uses the junction force balance; ray curvature (Fig. 214 C) and non-planarity (p. 444) are outside the model; the growth GIF's three-cell field is illustrative, not derived from the tensions.  
  evidence: `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L71–111; `morphoeval/lean/loops/Thompson1917_T1917_024.lean` L59–229; `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L296–307
- **Morphological realism / specimen match: 58** — Side by side, each measured spicule (ray axes re-rendered) is matched by the E5 model at its fitted rho: the Y-shaped cortical triactine (rho 1.43), the sagittal tubar triactines (rho ~0.74) and the T-shaped atrial tetractine (rho 0.37) are all visibly reproduced; the SEM overlay shows the detected axes on the real spicules. Deductions: the model panel for the rho > 1 spicule is mis-oriented (basal-ray picker), rays are drawn straight and tapered while the real laterals of c and e curve, the tetractines' apical ray is not rendered, and only one specimen image is compared.  
  evidence: `docs/media/t1917_024_on_concretions_spicules_and_sp_comparison.png` L0–0; `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L423–435
- **Parameter expressivity: 45** — rho is the single morphospace parameter and it monotonically moves the form from T (rho 0) through equiangular (rho 1) towards detachment (rho 2); the GIF shows three values and the angle-plane panel draws the whole curve. But there is no sweep with measured outputs, no CLI control of rho, threshold or radial parameters, no exploration of the general three-tension model (gamma_1, gamma_2, gamma_3 free), and the (pair, unpaired) plane is tautological (sum = 360) so the curve conveys nothing the sorted angles do not already contain.  
  evidence: `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L309–343; `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L393–414
- **Morphometric validation: 48** — Identical ray-axis morphometrics are measured on every real spicule (near-junction and chord axes, three angles, paired difference, unpaired deviation, fitted rho) and summarised in measure_morphoeval (median paired difference 4.2 deg vs unpaired deviation 27 deg). Deductions: n = 8 from one plate; no null distribution for the paired difference (a random partition of 360 deg also has a smallest gap); no goodness-of-fit (r, RMSE, KS) against the model; no second specimen population and Thompson's Fig. 214 is pinned but not digitised; near- vs chord-axis sensitivity is recorded but not analysed; measurement error from projection is unquantified.  
  evidence: `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L175–270; `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L438–462; `datasets/thompson_1917/t1917-024/measurements/sem_ray_angles.json` L0–0
- **Source and data traceability: 82** — All seven DOIs Crossref-verified during the loop; the specimen is a CC BY 3.0 Commons file with artist, credit, licence URL, Commons SHA-1 and SHA-256 recorded and the digest re-verified at run time; Thompson's page is pinned to the IA leaf with the leaf/page rule, with the OCR excerpt stored; the pre-assigned Dryad dataset's API metadata and per-file MD5s are pinned and the mismatch is reasoned in the scout manifest. Deduction: the Dryad payload itself was not fetched (5.8 GB), and the Haeckel (1872) and Dreyer (1892) primary sources are cited without DOIs/URLs.  
  evidence: `datasets/thompson_1917/t1917-024/paper_citations.bib` L1–128; `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L223–226; `datasets/thompson_1917/t1917-024/scout_manifest.json` L1–112
- **Code-to-model traceability: 70** — The module docstring states P1-P3 and E1-E8 with Thompson page numbers; every model function's docstring names its equation; the Lean header maps each theorem to an equation and to script function names; the KG yaml links code loci by function name. Deductions: code loci are by function name, not line range; the segmentation thresholds (90, 400, 1500, fill 0.9) are not traced to anything but the plate.  
  evidence: `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L1–50; `morphoeval/math_kg_loops/thompson1917-t1917-024.yaml` L60–131
- **Reproducibility package: 66** — Deterministic (no stochastic step in v1; seed set anyway), --self-test passes in < 1 s and the full run in ~3 s, documented invocation, outputs to docs/media and the dataset dir, environment versions recorded in this audit. Deductions: the v1 measure_morphoeval hook returned a dict and crashed measure_outputs.py (fixed in the hotfix, but the interface was not tested), no per-script dependency pin, scikit-image deprecation warnings are emitted, no CLI parameters, and the audit checkpoint is a branch commit not yet on main.  
  evidence: `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L465–490; `thompson/claims/t1917-024_on-concretions-spicules-and-spicular-skeletons.py` L493–518

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor (different provider) — this round is builder-scored'}
- {'id': 'R2', 'item': "Null control for the paired-difference statistic: Monte Carlo Dirichlet(1,1,1) random partitions of 360 deg (smallest-gap distribution) with a one-sided p-value and KS test; otherwise 'two equal angles' is unsupported"}
- {'id': 'R3', 'item': 'Parameter sweep: rho from 0 to 2 with measured outputs (angle triplets re-measured from rendered spicules by the same radial method — closes the measurement loop) and the general three-tension model'}
- {'id': 'R4', 'item': "Second specimen population: segment the CC BY-SA Grantia micrograph (many free triradiates) for the P1 120-deg mode, and digitise Thompson's own Fig. 214 (pinned IA leaf 465) with the same method"}
- {'id': 'R5', 'item': 'Goodness-of-fit: Pearson r and RMSE of model-predicted vs measured angle triplets, CV of angles, near-axis vs chord-axis sensitivity'}
- {'id': 'R6', 'item': 'Fix the basal-ray orientation heuristic for rho > 1 spicules and render the tetractine apical ray / ray curvature honestly (or state the limit)'}
- {'id': 'R7', 'item': "Ablation control: a 'crystallographic' alternative (Haeckel's view Thompson argues against — discrete 60/90/120 deg axes) vs the continuous E5 family, scored on the same data"}
- {'id': 'R8', 'item': 'Expose rho, threshold and radial parameters on the CLI; silence/upgrade the scikit-image deprecations'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) spicule rays as Plateau borders of the secreting cell field vs Grantia SEM ray angles (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `77e8201d27dd51cb520268fe921b47c6a9cc21b65c7f28e750eaa978978c6f12`
- `datasets/thompson_1917/t1917-024/measurements/sem_ray_angles.json` — artifact_sha256 `36caa2b660ca1147…`, 18,618 B, 
- `docs/media/t1917_024_on_concretions_spicules_and_sp_comparison.png` — artifact_sha256 `3f92607c5d3f21ac…`, 324,123 B, 1870×792 px

| measurement | value | reference | method |
|---|---:|---|---|
| three-ray spicules measured on the SEM plate | 8 count | — | radial-arc ray extraction after Otsu-free fixed threshold (90/255), closing(2), area >= 1500, fill <= 0.9 *(3 monaxon diactines/trichoxea detected as 2-ray needles)* |
| closest-pair angle difference, median | 4.208 deg | 0.0 | |theta_2 - theta_3| of the two closest sorted ray angles per spicule (near-junction axes), median over spicules *(mean 5.72 deg)* |
| unpaired-angle deviation from the paired angle, mean | 27.034 deg | — | |theta_unpaired - mean(theta_pair)| per spicule, mean over spicules *(ratio unpaired deviation / paired difference = 4.72 (no null distribution in Pass 1))* |
| all ray angles, SD | 15.478 deg | — | sample std of the 3 n sorted angles (mean is 120 by construction) *(fraction of spicules equiangular within 10 deg = 0.375)* |
| fitted tension ratio rho, range | 1.059 ratio span | — | rho = -2 cos(theta_pair) per spicule (inverse of E5) *(rho from 0.37 (T-shaped atrial tetractine) to 1.43 (Y-shaped cortical triactine))* |
| tetrahedral inter-ray angle (E7) | 109.4712 deg | 109.0 | arccos(-1/3) |
| interstice area coefficient (E9) | 0.161254 r^2 | 0.161254 | sqrt(3) - pi/2 for three mutually tangent equal circles |

## Environment at checkpoint


Pass-1 commit on branch cursor/t1917-024-sponge-spicules-ab2b: dataset dir, Lean module, KG law yaml, v1 script and rendered outputs; plus one hotfix committed with this audit (measure_morphoeval returned a dict and crashed measure_outputs.py at Step 8 — now a list of measurement records; rendered outputs byte-identical). Environment: Python 3.12.3, numpy 1.26.4, scipy 1.18.1, scikit-image 0.26.0, matplotlib 3.11.2, Pillow 13.0.0; Lean 4.35.0.
