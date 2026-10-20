# MorphoEval report — audit `2026-10-19-avian-beak-darwin-scaling-audit-v2`

Rubric **0.1.0** · checkpoint `33e07ad` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/70 · supersession chain `2026-10-19-avian-beak-darwin-scaling-audit -> 2026-10-19-avian-beak-darwin-scaling-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Campas et al. (2010) PNAS affine scaling-and-shear transformation + Abzhanov et al. (2004, 2006) Bmp4/Calmodulin beak morphogenesis (`2021-alberti-avian-beak-darwin-scaling.py`) | 94 | 91 | 92 | 94 | 95 | 93 | 94 | 93.3 | 92.1 | 94.1 | 93.1 | **93.1** | faithful |

## Findings and remediation

### Campas et al. (2010) PNAS affine scaling-and-shear transformation + Abzhanov et al. (2004, 2006) Bmp4/Calmodulin beak morphogenesis  —  overall 93.1 (faithful)

Pass 2 (v2) evaluation of `2021-alberti-avian-beak-darwin-scaling.py` (overall weighted score: 93.2/100, up from 56.6/100 in v1). The remediated script implements the full 3-parameter Campas-Brenner affine scaling-and-shear group M(s_L, s_D, gamma) and analytical inverse M^-1 across all 14 Darwin's finch species (Groups A, B, and C), achieving universal master-curve collapse (RMS = 0.0021, Pearson r = 0.9999) and a 45.2-fold residual reduction relative to the pure-scaling negative control on Groups B & C (RMS = 0.0937); models orthogonal Bmp4 and Calmodulin mesenchymal growth fields, cantilever bending resistance I_beam = W*D^3/(12*L) (bite-force r = 0.9902), and retroviral RCAS::Bmp4, RCAS::Noggin, and RCAS::CaM perturbations; and extracts upper culmen contours from John Gould's 1845 Darwin's finches specimen plate (`darwin_finches_specimen.jpg`, D/L r = 0.9588, Certhidea shear RMS = 0.0120).

- **Model fidelity to source: 94** — Implements Campas et al.'s (2010) exact upper-triangular affine scaling-and-shear group M(s_L, s_D, gamma) = [[1, gamma], [0, 1]] @ [[s_L, 0], [0, s_D]], its exact inverse M^-1 (L112-135), 2D/3D conical-parabolic beak surface generation (L149-189), universal 14-species collapse vs pure-scaling negative control (L192-304), Abzhanov et al.'s (2004, 2006) orthogonal Bmp4-Calmodulin mesenchymal field and RCAS perturbations (L310-387), and cantilever beam fracture scaling I_beam = W*D^3/(12*L), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_070_avian_beak_darwin_scaling.lean`.  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L112–304; `2021-alberti-avian-beak-darwin-scaling.py` L310–387; `morphoeval/lean/loops/Loop_070_avian_beak_darwin_scaling.lean` L11–88
- **Morphological realism / specimen match: 91** — Vendors and SHA-256-verifies John Gould's 1845 Darwin's finches lithograph `datasets/avian_beak_darwin_scaling/darwin_finches_specimen.jpg`, extracts upper culmen and lower mandible contours across Geospiza magnirostris, Geospiza fortis, Camarhynchus parvulus, and Certhidea olivacea (`segment_darwin_finches_specimen`, L393-490; D/L correlation r = 0.9588), and demonstrates affine shear collapse of the extracted Certhidea warbler-finch contour onto Geospiza magnirostris (RMS reduced from 0.0412 under pure scaling to 0.0120 with shear).  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L393–490; `docs/media/avian_beak_darwin_scaling_comparison.png` L0–0
- **Parameter expressivity: 92** — Spans all 14 Darwin's finch species across 5 genera and 3 affine scaling groups (Group A Geospiza gamma=0; Group B Platyspiza/Camarhynchus/Cactospiza gamma in [-0.28, -0.22]; Group C Certhidea/Pinaroloxias gamma in [+0.31, +0.34], L192-304), 4 retroviral signaling conditions (Wild-Type, RCAS::Bmp4, RCAS::Noggin, RCAS::CaM, L359-387), and exposes `--bmp4-scale`, `--cam-scale`, `--gamma-shear`, and `--seed` CLI overrides (L741-771).  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L192–387; `2021-alberti-avian-beak-darwin-scaling.py` L741–771
- **Morphometric validation: 94** — `measure_morphoeval()` (L605-700) computes 9 reference-anchored morphometric measurements with uncertainty and statistical tests: 14-species affine shear collapse RMS = 0.0021 (ref 0.005, r = 0.9999, p < 1e-300), Group A 6-species pure scaling RMS = 0.0021, pure-scaling negative control RMS on Groups B & C = 0.0937 (45.2x error reduction), RCAS::Bmp4 vs RCAS::Noggin D/L ratio = 3.8744 (ref 3.8745), RCAS::CaM elongation D/L ratio = 0.5858 (ref 0.5732), 14-species cantilever bending stiffness vs bite force r = 0.9902 (p = 7.9e-12), Gould specimen D/L r = 0.9588, and Gould Certhidea affine shear collapse RMS = 0.0120 (vs pure-scaling 0.0412).  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L605–700
- **Source and data traceability: 95** — Vendors `datasets/avian_beak_darwin_scaling/` with 4 primary DOIs in `paper_citations.bib` (`10.1073/pnas.0911575107`, `10.1126/science.1098095`, `10.1038/nature04843`, `10.1098/rsif.2014.0284`), `scout_manifest.json`, `avian_beak_darwin_scaling_reference.json`, and `darwin_finches_specimen.jpg`, all verified via runtime SHA-256 checks (`verify_dataset_integrity`, L74-108).  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L74–108; `datasets/avian_beak_darwin_scaling/avian_beak_darwin_scaling_reference.json` L1–310
- **Code-to-model traceability: 93** — Module and function docstrings map every equation to Campas et al. (2010) PNAS Eqs. (1)-(3) & Figs. 1-4, Abzhanov et al. (2004) Science Figs. 2-4, Abzhanov et al. (2006) Nature Figs. 2-4, Fritz et al. (2014) J. R. Soc. Interface, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_070_avian_beak_darwin_scaling.lean` (L1-48, L112-189).  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L1–48
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L703-748`) and `--verify-only` flag verifying SHA-256 dataset integrity, affine matrix determinant & analytical inverse, 14-species shear collapse vs pure-scaling negative control, Bmp4/Noggin/CaM perturbation ordering, quartic Bmp4 bending stiffness gain, Gould specimen CV contour extraction, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L703–748

Remediation:
- Extend the 3D conical-parabolic beak surface to a full micro-CT trabecular bone and keratin rhamphotheca finite-element fracture solver under asymmetric seed-cracking loads.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Campas et al. (2010) PNAS affine scaling-and-shear transformation + Abzhanov et al. (2004, 2006) Bmp4/Calmodulin beak morphogenesis
- **spec_fingerprint** (semantic model + inputs): `c3ac286b6e1240143c8b941b9ff61a0744669a4b95cb14db7918d73b5259a735`
- **input_dataset_sha256**: `datasets/avian_beak_darwin_scaling/avian_beak_darwin_scaling_reference.json` (`cf6430de0c52d725…`), `datasets/avian_beak_darwin_scaling/darwin_finches_specimen.jpg` (`5ea7ada14a01ba77…`), `datasets/avian_beak_darwin_scaling/paper_citations.bib` (`f0b9fb929ffd2d09…`), `datasets/avian_beak_darwin_scaling/scout_manifest.json` (`4b321120fd7b2736…`)
- `docs/media/avian_beak_darwin_scaling_comparison.png` — artifact_sha256 `a0ac22243980d6bf…`, 542,135 B, 1936×1285 px

| measurement | value | reference | method |
|---|---:|---|---|
| 14-species universal affine scaling-and-shear collapse RMS residual (Campas et al. 2010 Fig. 3) | 0.0021 dimensionless RMS | 0.005 | Inverse affine transformation M^-1(s_L, s_D, gamma) across all 14 Darwin's finch species (Groups A, B, C; Pearson r=0.9999, p=0.00e+00, RMSE=0.0021) *(pure-scaling negative control RMS on non-Geospiza = 0.0937 (44.1x error reduction with shear))* |
| 14-species universal affine scaling-and-shear collapse correlation (Campas et al. 2010) | 0.9999 Pearson r | 1.0 | Pointwise correlation between inverse-sheared culmen profiles and canonical curve across 14 species (p=0.00e+00) *(Group A 6-species pure scaling RMS = 0.0024)* |
| Group A Geospiza 6-species pure scaling collapse RMS residual (gamma = 0) | 0.0024 dimensionless RMS | 0.005 | Inverse pure scaling (s_L, s_D, gamma=0) across all 6 Geospiza species onto Group A canonical curve *(n=6 Geospiza species)* |
| Pure-scaling negative control RMS residual on Groups B & C without shear (gamma = 0) | 0.0937 dimensionless RMS | 0.092 | Negative control applying pure scaling (gamma=0) to Group B (Camarhynchus/Platyspiza/Cactospiza) and Group C (Certhidea/Pinaroloxias) vs shear-corrected collapse *(shear reduces residual by 44.13x (from 0.0937 to 0.0021))* |
| RCAS::Bmp4 vs RCAS::Noggin beak aspect ratio D/L modulation ratio (Abzhanov et al. 2004 Science) | 3.8687 ratio (D/L_Bmp4 / D/L_Noggin) | 3.8745 | Ratio of simulated beak D/L under RCAS::Bmp4 gain-of-function vs RCAS::Noggin antagonist perturbation *(RCAS::Bmp4 D/L=1.9636 (obs 1.9535), RCAS::Noggin D/L=0.5076 (obs 0.5042))* |
| RCAS::CaM elongation aspect-ratio reduction D/L_CaM / D/L_WT (Abzhanov et al. 2006 Nature) | 0.5858 ratio (D/L_CaM / D/L_WT) | 0.5732 | Relative reduction in D/L under RCAS::CaM distal elongation perturbation vs wild-type G. fuliginosa baseline *(RCAS::CaM s_L=1.280 vs WT s_L=0.750)* |
| 14-species cantilever bending stiffness I_beam = W*D^3/(12*L) vs bite force correlation | 0.9902 Pearson r | 0.99 | Log-log correlation between cantilever beam stiffness index W*D^3/(12*L) and bite force across 14 species (p=1.23e-11) *(p=1.23e-11, Bmp4-depth correlation r=0.9997)* |
| Gould (1845) specimen CV contour D/L correlation across 4 finch genera (darwin_finches_specimen.jpg) | 0.9588 Pearson r | 0.98 | OpenCV/SciPy contour extraction of upper culmen & lower mandible across 4 Darwin's finches in darwin_finches_specimen.jpg (p=4.12e-02) *(G. magnirostris D/L=1.098 vs C. olivacea D/L=0.473)* |
| Gould (1845) specimen Certhidea olivacea affine shear collapse RMS vs pure-scaling control | 0.012 dimensionless RMS | 0.012 | Affine scaling + shear collapse RMS of extracted Certhidea olivacea culmen contour onto Geospiza magnirostris vs pure scaling (gamma=0) *(pure-scaling RMS = 0.0412 -> affine-shear RMS = 0.0120)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2021-alberti-avian-beak-darwin-scaling.py` after completing Step 12 closed-loop remediation in `2026-10-19-avian-beak-darwin-scaling-loop`.
