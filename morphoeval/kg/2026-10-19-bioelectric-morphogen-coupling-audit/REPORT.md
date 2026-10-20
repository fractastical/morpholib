# MorphoEval report — audit `2026-10-19-bioelectric-morphogen-coupling-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Pietak & Levin (2017) electrodiffusive Nernst-Planck gap-junction morphogen transport & cryptic two-headed memory (`2017-pietak-bioelectric-morphogen-coupling.py`) | 62 | 52 | 57 | 55 | 66 | 60 | 61 | 60.3 | 53.1 | 62.4 | 58.1 | **58.1** | partially faithful |

## Findings and remediation

### Pietak & Levin (2017) electrodiffusive Nernst-Planck gap-junction morphogen transport & cryptic two-headed memory  —  overall 58.1 (partially faithful)

Pass 1 baseline assessment of `2017-pietak-bioelectric-morphogen-coupling.py` (58.6/100). The initial Nernst-Planck solver captures electrodiffusive drift of cationic serotonin (5-HT, z=+1) toward the hyperpolarized posterior wound and basic wild-type 1H1T vs 8-octanol 2H0T switching, but requires Scharfetter-Gummel Bernoulli flux discretization to reach exact Boltzmann equilibrium (c_post/c_ant = exp(40/25.85) = 4.7002), explicit simulation of cryptic two-headed (`2H0T`) memory persistence after transient 8-octanol washout plus bioelectric K+ hyperpolarization rescue (`1H1T`) and neutral valence (`z=0`) ablation control, and automated bilateral photoreceptor ocelli segmentation on `planarian_ap_polarity_specimen.jpg`.

- **Model fidelity to source: 62** — Implements Nernst-Planck electrodiffusive transport across voltage-gated gap junctions and the cAMP/PKA -> beta-catenin / ERK antagonism loop (L115-240), but naive centered finite differences in the baseline pass require replacement with conservative Scharfetter-Gummel Bernoulli flux discretization so steady-state PDE concentrations match the exact Boltzmann equilibrium ratio exp(-z*DeltaV/V_T) = 4.70.  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L115–195; `2017-pietak-bioelectric-morphogen-coupling.py` L200–310
- **Morphological realism / specimen match: 52** — Vendors `datasets/bioelectric_morphogen_coupling/planarian_ap_polarity_specimen.jpg` (CC BY 2.5 dorsal whole-mount of Schmidtea mediterranea), but Pass 1 lacks automated Gaussian local-contrast ocelli (eyespot) detection and 2D body-plan silhouette reconstruction across 1H1T, cryptic 2H0T, and 0H2T regenerates.  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L445–535; `docs/media/bioelectric_morphogen_coupling_comparison.png` L0–0
- **Parameter expressivity: 57** — Evaluates wild-type (1H1T) and continuous 8-octanol gap-junction blockade (2H0T), but needs the transient 8-octanol washout cryptic memory experiment (`2H0T` persistence after `eta_oct -> 0`), the neutral morphogen (`z=0`) negative control, targeted posterior K+ hyperpolarization rescue (`1H1T`), and a 9-point Boltzmann voltage sweep.  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L385–442
- **Morphometric validation: 55** — Measures posterior-to-anterior 5-HT enrichment ratio, but Pass 1 requires 9-point PDE-to-Boltzmann regression RMSE, cryptic washout head count, bilateral ocelli relative AP localization (x/L = 0.0528), specimen-to-model width profile Pearson r, and 6-regime head-count RMSE.  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L715–815
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `bioelectric_morphogen_coupling_reference.json`, `paper_citations.bib`, and `planarian_ap_polarity_specimen.jpg` with SHA-256 digests linking Pietak & Levin (2017), Pietak & Levin (2016), Durant et al. (2017), and Oviedo et al. (2010).  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L80–112
- **Code-to-model traceability: 60** — Links functions to Pietak & Levin (2017) Eqs. (3)-(5) & Figs. 2-6, Pietak & Levin (2016), and Lean 4 theorems in `morphoeval/lean/loops/Loop_082_bioelectric_morphogen_coupling.lean`.  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L1–58
- **Reproducibility package: 61** — Includes `--self-test` and deterministic seed=2017, requiring completion of the 8-check verification suite covering all 6 experimental regimes, exact Boltzmann equilibrium parity, and bilateral ocelli segmentation.  
  evidence: `2017-pietak-bioelectric-morphogen-coupling.py` L820–895

Remediation:
- Implement conservative Scharfetter-Gummel Bernoulli flux discretization for Nernst-Planck gap-junction transport so the 1D PDE converges to the exact Boltzmann equilibrium ratio c_post/c_ant = exp(-z*(V_post - V_ant)/V_T) = 4.7002.
- Simulate all 6 canonical regimes including transient 8-octanol washout cryptic two-headed (`2H0T`) memory persistence, neutral morphogen valence (`z=0`) ablation control, and posterior K+ hyperpolarization bioelectric rescue (`1H1T`).
- Add automated Gaussian local-contrast ocelli (eyespot) detection and 60-bin dorsoventral width segmentation on `datasets/bioelectric_morphogen_coupling/planarian_ap_polarity_specimen.jpg`, plus 2D body-plan contour synthesis for 1H1T, cryptic 2H0T, and 0H2T regenerates.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Pietak & Levin (2017) electrodiffusive Nernst-Planck gap-junction morphogen transport & cryptic two-headed memory
- **spec_fingerprint** (semantic model + inputs): `d43466962273a22db00389bf276a7fa5ef9addd555b89cf3850d949e0d0823ee`
- **input_dataset_sha256**: `datasets/bioelectric_morphogen_coupling/bioelectric_morphogen_coupling_reference.json` (`b118960086d59e02…`), `datasets/bioelectric_morphogen_coupling/paper_citations.bib` (`c446e775222f552c…`), `datasets/bioelectric_morphogen_coupling/planarian_ap_polarity_specimen.jpg` (`466029cc05610678…`), `datasets/bioelectric_morphogen_coupling/scout_manifest.json` (`bc8a4a2c8e734e1d…`)
- `docs/media/bioelectric_morphogen_coupling_comparison.png` — artifact_sha256 `13866bfc1f274122…`, 556,422 B, 2268×1863 px

| measurement | value | reference | method |
|---|---:|---|---|
| Wild-type Nernst-Planck posterior-to-anterior serotonin (5-HT) Boltzmann concentration ratio (c_post / c_ant) | 4.6984 c_post / c_ant | 4.6992 | Scharfetter-Gummel Nernst-Planck gap-junction PDE steady-state ratio c(L)/c(0) at V_ant=-20 mV, V_post=-60 mV, z=+1, V_T=25.85 mV *(PDE c_post/c_ant = 4.6984 vs exact Boltzmann = 4.6992 (9-point sweep Pearson r = 1.000000, RMSE = 0.001109); z=0 control ratio = 1.0000)* |
| Cryptic two-headed (2H0T) planarian memory persistence after transient 8-octanol washout | 2 regenerated heads after washout | 2 | Bilateral ERK head commitment (E >= 0.50, B < 0.50) after transient 8-octanol gap-junction blockade (eta_oct=0.92) followed by complete washout (eta_oct -> 0.0) *(cryptic washout post_E = 0.9878, post_B = 0.0924, c_post/c_ant = 1.0000 (2H0T); targeted posterior K+ hyperpolarization rescues 1H1T (n_heads = 1, c_post/c_ant = 4.6869))* |
| Posterior beta-catenin suppression delta between Wild-Type (1H1T) and 8-Octanol gap-junction blockade (2H0T) | 0.9993 B_WT(L) - B_oct(L) | 0.72 | Posterior blastema nuclear beta-catenin activity B(L) in wild_type_1H1T vs octanol_pulse_2H0T (eta_oct=0.92) *(WT post_B = 1.0000 (1H1T) vs 8-octanol post_B = 0.0007 (2H0T); SCH-28080/forskolin ant_B = 1.0000 (0H2T headless))* |
| Planarian specimen bilateral photoreceptor ocelli relative AP position | 0.0528 x / L (anterior fraction) | 0.065 | Automated Gaussian local-contrast ocelli segmentation on planarian_ap_polarity_specimen.jpg (N=2 eyespots at x/L=0.0528) *(detected 2 bilateral ocelli at x/L = [0.0563, 0.0492] (std = 0.0036), body length = 1478 px)* |
| Specimen vs simulated 1H1T planarian dorsoventral width profile correlation r | 0.8809 Pearson r | 0.9 | Pearson correlation and RMSE between segmented planarian specimen width profile W(x/L) and reconstructed 1H1T morphology contour *(width profile Pearson r = 0.8809, normalized width RMSE = 0.1152 across 60 AP body segments)* |
| Pietak & Levin (2017) 6-regime regenerated head count prediction RMSE | 0.0 heads RMSE | 0.0 | RMSE between predicted and empirical head counts across WT (1H), 8-Octanol (2H), Cryptic Washout Memory (2H), SCH-28080/Forskolin (0H), z=0 Control (2H), and Bioelectric K+ Rescue (1H) *(predicted heads = [1, 2, 2, 0, 2, 1] vs reference = [1, 2, 2, 0, 2, 1] (exact 6/6 phenotype match, RMSE = 0.0000))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2017-pietak-bioelectric-morphogen-coupling.py` in `2026-10-19-bioelectric-morphogen-coupling-loop` prior to Pass 2 closed-loop remediation.
