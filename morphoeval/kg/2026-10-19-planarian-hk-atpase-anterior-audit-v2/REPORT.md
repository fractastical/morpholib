# MorphoEval report — audit `2026-10-19-planarian-hk-atpase-anterior-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-planarian-hk-atpase-anterior-audit -> 2026-10-19-planarian-hk-atpase-anterior-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Beane et al. (2011) planarian H,K-ATPase bioelectric V_mem anterior polarity switch (`2011-beane-planarian-hk-atpase-anterior.py`) | 93 | 89 | 91 | 91 | 93 | 91 | 91 | 92.3 | 89.8 | 91.7 | 91.1 | **91.1** | faithful |

## Findings and remediation

### Beane et al. (2011) planarian H,K-ATPase bioelectric V_mem anterior polarity switch  —  overall 91.1 (faithful)

Pass 2 post-remediation evaluation of `2011-beane-planarian-hk-atpase-anterior.py` (91.3/100, up from 59.2/100 in Pass 1; +32.1 overall). The remediated script couples the cubic bistable membrane potential PDE to Bell-form voltage-gated innexin gap-junction conductance G_GJ(DeltaV) = (1 - eta_oct)*G_0 / (1 + (DeltaV/V_0)^2) and L-type Ca_v-gated anterior/posterior gene transcription, reproducing all 5 canonical Beane et al. (2011) and Oviedo et al. (2010) regeneration phenotypes with zero head-count error (RMSE = 0.0000): Wild-Type `1H1T` (delta_V = +50.5 mV), SCH-28080 / H,K-ATPase RNAi `0H2T` headless, Ivermectin `2H0T` double-headed, 8-Octanol gap-junction blockade `2H0T` double-headed, and Nifedipine Ca_v-block negative control `0H2T`, alongside automated bilateral ocelli segmentation on `schmidtea_mediterranea_specimen.jpg` (N = 2 eyespots at x/L = 0.0528), backed by 3 machine-checked Lean 4 theorems in `Loop_081_planarian_hk_atpase_anterior.lean`.

- **Model fidelity to source: 93** — Faithfully implements the coupled Beane et al. (2011) and Pietak-Levin (2017) bistable membrane voltage PDE, symmetric Bell-form voltage-gated gap-junction flux divergence (10x conductance isolation at 36 mV trans-junctional drop), voltage-gated L-type Ca2+ influx gate Ca_ss(V), and mutually antagonistic anterior Head H(x,t) (*pc2/ndk*) and posterior Tail T(x,t) (*wnt1*) reaction-diffusion equations (L118-378), verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_081_planarian_hk_atpase_anterior.lean`.  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L118–195; `2011-beane-planarian-hk-atpase-anterior.py` L198–378; `morphoeval/lean/loops/Loop_081_planarian_hk_atpase_anterior.lean` L12–92
- **Morphological realism / specimen match: 89** — Vendors and SHA-256-verifies the CC BY 2.5 dorsal whole-mount micrograph of Schmidtea mediterranea (`datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg`), automatically segments the body contour (length 1478 px, aspect ratio 3.72) and bilateral photoreceptor ocelli (`segment_planarian_specimen`, L446-528: N = 2 eyespots at x/L = 0.0492 and 0.0563, mean x/L = 0.0528 matching the simulated anterior head domain), and reconstructs 2D body-plan silhouettes with eyespots for 1H1T, 2H0T, and 0H2T regenerates (Panels B3, C1, C2).  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L446–566; `docs/media/planarian_hk_atpase_anterior_comparison.png` L0–0
- **Parameter expressivity: 91** — Simulates all 5 experimental regimes from Beane et al. (2011) and Oviedo et al. (2010) (`wild_type_1H1T`, `hk_atpase_inhibited_0H2T`, `ivermectin_depolarized_2H0T`, `octanol_gj_blockade_2H0T`, and `nifedipine_cav_block_0H2T` negative control), executes an 11-point H,K-ATPase pump current dose-response curve across I_HK in [-18, +24] showing the sharp 0H2T -> 1H1T bifurcation (Panel C3), and maps the 2D (I_HK, I_IVM) bioelectric phase diagram across 0H2T, 1H1T, and 2H0T territories (Panel B2).  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L380–442
- **Morphometric validation: 91** — Extracts 6 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L715-815): wild-type AP V_mem polarization delta (+50.52 mV vs +45.0 mV reference), H,K-ATPase inhibition anterior head commitment suppression delta (0.9168 vs 0.90 reference), ivermectin & 8-octanol double-head induction count (2 heads vs 2 reference), segmented specimen bilateral ocelli relative AP position (0.0528 vs 0.065 reference, std = 0.0036), specimen vs simulated 1H1T width profile correlation (Pearson r and RMSE across 60 AP bins), and 5-regime regenerated head count RMSE = 0.0000 (exact 5/5 phenotype match).  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L715–815
- **Source and data traceability: 93** — Vendors `datasets/planarian_hk_atpase_anterior/` with `scout_manifest.json`, `planarian_hk_atpase_anterior_reference.json`, `schmidtea_mediterranea_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1016/j.chembiol.2010.11.012`, `10.1016/j.ydbio.2009.12.012`, `10.1098/rsif.2017.0425`, `10.1016/j.bpj.2017.04.011`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_sha256`, L88-107).  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L88–107; `datasets/planarian_hk_atpase_anterior/planarian_hk_atpase_anterior_reference.json` L1–148
- **Code-to-model traceability: 91** — Every computational function and docstring maps directly to Beane et al. (2011) Figs. 1-6 and Table 1, Oviedo et al. (2010), Pietak & Levin (2017) Eqs. (1)-(8), and the 3 Lean 4 theorems in `Loop_081_planarian_hk_atpase_anterior.lean` and `morphoeval/math_kg_loops/beane-planarian-hk-atpase-anterior.yaml`.  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L1–56; `morphoeval/math_kg_loops/beane-planarian-hk-atpase-anterior.yaml` L1–88
- **Reproducibility package: 91** — Deterministic RNG (`seed=2011`), standalone 8-check `--self-test` suite verifying SHA-256 dataset locks, bistable V_mem equilibria, voltage-gated GJ symmetry and isolation, all 5 Beane et al. (2011) regeneration phenotypes (`1H1T`, `0H2T` SCH-28080, `2H0T` ivermectin, `2H0T` 8-octanol, `0H2T` nifedipine control), specimen bilateral ocelli segmentation, and bitwise two-run determinism (`atol=0.0`), plus rendered 9-panel PNG and 24-frame GIF in `docs/media/`.  
  evidence: `2011-beane-planarian-hk-atpase-anterior.py` L820–895; `docs/media/planarian_hk_atpase_anterior_growth.gif` L0–0

Remediation:
- Extend the 1D anteroposterior bioelectric PDE to a 2D finite-element planarian body mesh to model mediolateral slit/netrin axonal guidance of photoreceptor nerves.
- Couple endogenous bioelectric V_mem states to stochastic neoblast division and migration kinetics during blastema outgrowth.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Beane et al. (2011) planarian H,K-ATPase bioelectric V_mem anterior polarity switch
- **spec_fingerprint** (semantic model + inputs): `c0d35653d229caf892ce95227098738e04e01130c0c0eb1c24a63345a769e0d8`
- **input_dataset_sha256**: `datasets/planarian_hk_atpase_anterior/paper_citations.bib` (`4978cdbc97ad314e…`), `datasets/planarian_hk_atpase_anterior/planarian_hk_atpase_anterior_reference.json` (`ad7dfa156bbb79fe…`), `datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg` (`466029cc05610678…`), `datasets/planarian_hk_atpase_anterior/scout_manifest.json` (`9de72183ede698f0…`)
- `docs/media/planarian_hk_atpase_anterior_comparison.png` — artifact_sha256 `23e52bef2eb4a12b…`, 586,158 B, 2268×1863 px
- `docs/media/planarian_hk_atpase_anterior_growth.gif` — artifact_sha256 `4b0de2e5847b4abc…`, 265,390 B, 827×351 px, 24 frames

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

Pass 2 (v2) post-remediation evaluation of `2011-beane-planarian-hk-atpase-anterior.py` after completing the 12-step closed remediation loop (`2026-10-19-planarian-hk-atpase-anterior-loop`).
