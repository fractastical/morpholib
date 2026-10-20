# MorphoEval report — audit `2026-10-19-planarian-hk-atpase-anterior-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Beane et al. (2011) planarian H,K-ATPase bioelectric V_mem anterior polarity switch (`2011-beane-planarian-hk-atpase-anterior.py`) | 63 | 53 | 58 | 55 | 66 | 61 | 62 | 61.3 | 53.8 | 63.1 | 58.8 | **58.8** | partially faithful |

## Findings and remediation

### Beane et al. (2011) planarian H,K-ATPase bioelectric V_mem anterior polarity switch  —  overall 58.8 (partially faithful)

Pass 1 baseline assessment of `2011-beane-planarian-hk-atpase-anterior.py` (59.2/100). The initial bioelectric ODE captures bistable V_mem attractors (-65 mV and -20 mV) and basic wild-type 1H1T vs SCH-28080 0H2T switching, but requires nonlinear Bell-form voltage-gated innexin gap-junction coupling G_GJ(dV/dx) so the anterior depolarized blastema self-isolates from the hyperpolarized trunk, automated bilateral photoreceptor ocelli segmentation on `schmidtea_mediterranea_specimen.jpg`, and an explicit nifedipine L-type Ca_v channel blockade negative control alongside 8-octanol gap-junction blockade.

- **Model fidelity to source: 63** — Implements the cubic bistable membrane current I_ion(V) = g*(V - V_hyper)*(V - V_thresh)*(V - V_depol) and voltage-gated Ca2+ activation gate Ca_ss(V) (L118-245), but constant linear gap-junction diffusion in the baseline pass requires replacement with nonlinear voltage-gated innexin conductance G_GJ(DeltaV) = (1 - eta_oct)*G_0 / (1 + (DeltaV/V_0)^2) to prevent trunk hyperpolarization from draining the anterior blastema.  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L118–195; `2011-beane-planarian-hk-atpase-anterior.py` L200–290
- **Morphological realism / specimen match: 53** — Vendors `datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg` (CC BY 2.5 dorsal whole-mount of Schmidtea mediterranea), but Pass 1 lacks automated Gaussian local-contrast ocelli (eyespot) detection and 2D body-plan silhouette reconstruction across 1H1T, 2H0T, and 0H2T regenerates.  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L445–535; `docs/media/planarian_hk_atpase_anterior_comparison.png` L0–0
- **Parameter expressivity: 58** — Evaluates wild-type (1H1T), SCH-28080 (0H2T), and ivermectin (2H0T), but needs both the 8-octanol gap-junction blockade (2H0T) and the nifedipine L-type Ca_v channel blockade negative control (depolarized V_mem with 0 heads) plus a 2D (I_HK, I_IVM) phase diagram.  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L380–442
- **Morphometric validation: 55** — Measures anteroposterior V_mem delta, but Pass 1 requires bilateral ocelli relative AP localization (x/L = 0.0528), specimen-to-model dorsoventral width profile Pearson r and RMSE, and 5-regime head-count RMSE.  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L715–815
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `planarian_hk_atpase_anterior_reference.json`, `paper_citations.bib`, and `schmidtea_mediterranea_specimen.jpg` with SHA-256 digests linking Beane et al. (2011), Oviedo et al. (2010), and Pietak & Levin (2017).  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L80–112
- **Code-to-model traceability: 61** — Links functions to Beane et al. (2011) Figs. 1-6, Pietak & Levin (2017), and Lean 4 theorems in `morphoeval/lean/loops/Loop_081_planarian_hk_atpase_anterior.lean`.  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L1–56
- **Reproducibility package: 62** — Includes `--self-test` and deterministic seed=2011, requiring completion of the 8-check verification suite covering all 5 experimental regimes and bilateral ocelli segmentation.  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L820–895

Remediation:
- Couple the bistable membrane potential PDE to nonlinear Bell-form voltage-gated gap-junction conductance G_GJ(DeltaV) = (1 - eta_oct)*G_0 / (1 + (DeltaV/V_0)^2) so anterior depolarization self-isolates from the hyperpolarized trunk.
- Add automated Gaussian local-contrast ocelli (eyespot) detection and 60-bin dorsoventral width segmentation on `datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg`, plus 2D body-plan contour synthesis for 1H1T, 2H0T, and 0H2T regenerates.
- Include both 8-octanol gap-junction blockade (2H0T) and nifedipine L-type Ca_v channel blockade negative control (depolarized V_mem -> 0H2T) alongside a 2D (I_HK, I_IVM) phase diagram.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Beane et al. (2011) planarian H,K-ATPase bioelectric V_mem anterior polarity switch
- **spec_fingerprint** (semantic model + inputs): `c0d35653d229caf892ce95227098738e04e01130c0c0eb1c24a63345a769e0d8`
- **input_dataset_sha256**: `datasets/planarian_hk_atpase_anterior/paper_citations.bib` (`4978cdbc97ad314e…`), `datasets/planarian_hk_atpase_anterior/planarian_hk_atpase_anterior_reference.json` (`ad7dfa156bbb79fe…`), `datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg` (`466029cc05610678…`), `datasets/planarian_hk_atpase_anterior/scout_manifest.json` (`9de72183ede698f0…`)
- `docs/media/planarian_hk_atpase_anterior_comparison.png` — artifact_sha256 `23e52bef2eb4a12b…`, 586,158 B, 2268×1863 px

| measurement | value | reference | method |
|---|---:|---|---|
| Wild-type planarian anteroposterior V_mem polarization delta (V_ant - V_post) | 50.5062 mV | 45.0 | Coupled bistable V_mem + voltage-gated gap-junction PDE steady-state difference V_ant(0) - V_post(L) in wild_type_1H1T *(WT V_ant = -14.49 mV, V_post = -65.00 mV (1H1T); SCH-28080 H,K-ATPase inhibition collapses delta_V to -5.05 mV (0H2T))* |
| H,K-ATPase inhibition (SCH-28080 / RNAi) vs Wild-Type anterior head commitment suppression delta | 0.9467 H_WT(0) - H_SCH28080(0) | 0.9 | Anterior blastema head gene expression H(0) in wild_type_1H1T vs hk_atpase_inhibited_0H2T (SCH-28080) *(WT ant_H = 0.9468 (n_heads=1) vs SCH-28080 ant_H = 0.0001 (n_heads=0); nifedipine Ca_v-block negative control ant_H = 0.0155 (n_heads=0))* |
| Ivermectin depolarization & 8-octanol gap-junction blockade double-head (2H0T) induction count | 2 regenerated heads | 2 | Bilateral blastema head count (H >= 0.50) under ivermectin GluCl depolarization (I_IVM=22) and 8-octanol gap-junction blockade (eta_oct=0.92) *(ivermectin post_H = 0.9823 (2H0T, V_post = -13.57 mV); 8-octanol post_H = 0.9464 (2H0T, n_heads = 2))* |
| Schmidtea mediterranea specimen bilateral photoreceptor ocelli count & relative AP position | 0.0528 x / L (anterior fraction) | 0.065 | Automated Gaussian local-contrast ocelli detection on schmidtea_mediterranea_specimen.jpg (N=2 eyespots at x/L=0.0528, inter-ocelli spacing=67.1 px) *(detected 2 bilateral ocelli at x/L = [0.0563, 0.0492] (std = 0.0036), body length = 1478 px, aspect ratio = 3.723)* |
| Specimen vs simulated 1H1T planarian dorsoventral width profile correlation r | 0.8884 Pearson r | 0.9 | Pearson correlation and RMSE between segmented Schmidtea mediterranea width profile W(x/L) and reconstructed 1H1T morphology contour *(width profile Pearson r = 0.8884, normalized width RMSE = 0.1111 across 60 AP body segments)* |
| Beane et al. (2011) 5-regime regenerated head count prediction RMSE | 0.0 heads RMSE | 0.0 | RMSE between predicted and empirical head counts across WT (1H), SCH-28080 (0H), Ivermectin (2H), 8-Octanol (2H), and Nifedipine control (0H) *(predicted heads = [1, 0, 2, 2, 0] vs reference = [1, 0, 2, 2, 0] (exact 5/5 phenotype match, RMSE = 0.0000, gj_isolation_ratio = 0.1000))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2011-beane-planarian-hk-atpase-anterior.py` in `2026-10-19-planarian-hk-atpase-anterior-loop` prior to Pass 2 closed-loop remediation.
