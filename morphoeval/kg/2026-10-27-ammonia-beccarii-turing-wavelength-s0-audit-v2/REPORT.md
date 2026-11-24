# MorphoEval report — audit `2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `a46634a7d52b65a84c884100329f1df74d7ae443` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit -> 2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ammonia beccarii Turing wavelength at 31 µm, Holocene 0 Ma (`models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py`) | 88 | 82 | 86 | 88 | 90 | 86 | 87 | 87.3 | 84.2 | 87.7 | 86.2 | **86.2** | faithful |

## Findings and remediation

### Ammonia beccarii Turing wavelength at 31 µm, Holocene 0 Ma  —  overall 86.2 (faithful)

Pass 2 of the 31.37 µm (3.137e-05 m) Schnakenberg wavelength on a Holocene (0 Ma) Ammonia beccarii test face. Closes pass-1 R1 and R2. The 500 µm bar is 127 px on the 720-wide plate (3.937 µm/px). The gray>0.08 bounding boxes average 769.7 µm and the gray>0.32 cores average 761.8 µm, so the rim is not the discrepancy: these six tests are larger than the caption's rounded ~600 µm. FFT wavelengths are 31.18 µm (96²), 32.46 µm (128²), and 32.88 µm (160²), all within 5% of λ_c and not forced to equal it. Dispersion monodromy r = 0.9999997. γ sweep r = 0.995. GIF 12 frames at 720×540, CV similarity 92.2. Chronoseries 1/1 still has no predecessor and no successor.

- **Model fidelity to source: 88** — Unchanged from pass 1. The Schnakenberg Jacobian, Turing inequality, k_c, and λ_c = 31.37 µm are implemented directly. Linearised Fourier monodromy matches σ(k) at r = 0.9999997. Equal diffusion gives σ(k_c) = −0.055. The kinetics are not a published foraminiferal calcification model. Lean theorems remain source-level only.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L1–36; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L130–210; `morphoeval/lean/loops/Loop_3108_ammonia_beccarii_turing_wavelength_s0.lean` L14–44
- **Morphological realism / specimen match: 82** — Same render as pass 1. The last GIF frame is the Hillewaert plate with the activator grown inside the six tests. CV similarity against the photograph is 92.2 at 12 frames and 720×540. The spot lattice is the model. Pores at 0.564 µm are not resolved on this plate, so the match is the shell geometry plus a wavelength-scale texture, not a pore-by-pore copy.  
  evidence: `docs/media/ammonia_beccarii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/ammonia_beccarii_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L450–470
- **Parameter expressivity: 86** — Unchanged mechanism, re-run on the pass-2 script. γ factors 0.70–1.80 move measured FFT wavelengths from 38.7 to 23.0 µm against predictions 37.5 to 23.4 µm (r = 0.995, RMSE 0.94 µm). Equal diffusion and γ = 0 both kill the pattern. a and b stay fixed.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L282–327; `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 88** — Closes R1 and R2. Halo bounding boxes average 769.7 µm (CV 0.08 on the six values) and rim-excluded cores average 761.8 µm, against the caption reference 600 µm: the two masks agree with each other and both disagree with the rounded caption, using the plate's own 500 µm bar (127 px). Grid FFT peaks are 31.18, 32.46, and 32.88 µm on 96², 128², and 160², reference 31.37 µm, spread 1.7 µm. Dispersion r = 0.9999997. Sweep r = 0.995, RMSE 0.94 µm. Spot-spacing CV 0.14, KS distance 0.27 versus Normal(31.37 µm). None of these measured values is copied onto its reference.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L329–421; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L604–610; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L659–800
- **Source and data traceability: 90** — Unchanged sources. Crossref-resolved DOIs for Turing, Schnakenberg, Hayward, Schönfeld, Petersen, and Subba & Ghosh. Specimen SHA-256 pinned and checked at runtime. The gallery Caryophyllia file was rejected. AFM pores remain Ammonia cf. beccarii, not these six tests.  
  evidence: `datasets/ammonia_beccarii_turing_wavelength_s0/paper_citations.bib` L1–84; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L53–91
- **Code-to-model traceability: 86** — The new rim-excluded diameter and the three-grid FFT block sit next to the segmentation and the production run they modify. Equation functions are unchanged. The Lean file still encodes the margin, the 19-wavelength count, the pore/detail/test order, and the Holocene terminal.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L375–390; `morphoeval/math_kg_loops/delphi10000-ammonia-beccarii-turing-wavelength-s0.yaml` L1–71
- **Reproducibility package: 87** — --self-test passes on the pass-2 script and now also requires the rim-excluded diameter to differ from the halo diameter and every grid wavelength to lie within 15% of 31.37 µm. Seed 3108 and the SHA-256 gate are unchanged. No per-script lockfile. Lean was not executed.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L630–660; `datasets/ammonia_beccarii_turing_wavelength_s0/README.md` L1–38

Remediation:
- {'id': 'R1', 'item': "Closed. Halo boxes average 769.7 µm and gray>0.32 cores average 761.8 µm on the 500 µm bar. The caption's ~600 µm is a round number, not the size of these six tests."}
- {'id': 'R2', 'item': 'Closed. FFT wavelengths are 31.18 µm (96²), 32.46 µm (128²), and 32.88 µm (160²). All sit within 5% of 31.37 µm; the finer grids run slightly long and were not adjusted to match.'}
- {'id': 'R3', 'item': 'Still open. Only grok-4.7 scored this audit. A second provider was not available in this run, and no second score was invented.'}
- {'id': 'R4', 'item': 'Still open. Lean 4 is not installed in this environment, so Loop_3108 was not kernel-checked.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ammonia beccarii Turing wavelength at 31 µm, Holocene 0 Ma
- **spec_fingerprint** (semantic model + inputs): `5c78fef63a2342ef4c8a82571c976eb61e437c0ace429ce768a5b9e0809b609e`
- **input_dataset_sha256**: `datasets/ammonia_beccarii_turing_wavelength_s0/Ammonia_beccarii_Hillewaert_2012.jpg` (`c15a90be96e2b088…`), `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_reference.json` (`6a6a4fb506092514…`), `datasets/ammonia_beccarii_turing_wavelength_s0/paper_citations.bib` (`5ac93aba84d88754…`), `datasets/ammonia_beccarii_turing_wavelength_s0/scout_manifest.json` (`12dc86d6408aac7a…`)
- `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_metrics.json` — artifact_sha256 `94e75fe8dd4bb167…`, 3,163 B, 
- `docs/media/ammonia_beccarii_turing_wavelength_s0_comparison.png` — artifact_sha256 `2f66f42e479d98dc…`, 813,963 B, 1344×1032 px
- `docs/media/ammonia_beccarii_turing_wavelength_s0_growth.gif` — artifact_sha256 `2fea87438ec20b71…`, 1,666,604 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| linear dispersion Pearson r vs numerical Fourier growth | 1.0 Pearson r | 1.0 | linearised Schnakenberg PDE on exact Fourier modes versus σ(k) from det(γJ − k²D − σI)=0 *(RMSE = 2.47123e-05 1/τ over 0.045454545454545525 KS; n modes in metrics JSON)* |
| dispersion RMSE | 2.5e-05 1/τ | 0.0 | root mean square of analytical minus numerical σ(k) |
| dispersion KS statistic | 0.0455 KS distance | 0.0 | two-sample KS between analytical and numerical σ samples *(p=1)* |
| gamma sweep predicted vs measured wavelength Pearson r | 0.9949 Pearson r | 1.0 | six γ factors; linear λ_c(γ) versus FFT wavelength of the saturated field *(RMSE 0.943 µm; predicted [37.49432147481993, 34.025542308856224, 31.37, 28.636761048228433, 26.05136703011823, 23.381817484722802]; measured [38.729183245398886, 34.26880997948951, 31.18235201808591, 27.779020210405303, 24.35939745297053, 23.03280501731194])* |
| saturated spot wavelength | 32.876 µm | 31.37 | parabolic-refined radial FFT peak of the nonlinear activator on the 600 µm face |
| equal-diffusion control sigma_max | -0.054621 1/τ | 0.0 | σ(k_c) with Dv reset to Du; Turing requires Dv f_u + Du g_v > 0 *(negative control: equal diffusion must decay)* |
| diffusion ablation std ratio | 0.1165 std_final/std_initial | 1.0 | γ = 0 pure diffusion of a noise seed for 400 steps, versus the patterned run *(ablation control)* |
| specimen segmented mean test diameter | 769.685 µm | 600.0 | longer side of the gray>0.08 bounding box of each test, scaled by the detected 500 µm bar (includes the bright rim) *(n=6; halo diameters=[736.220472440945, 795.2755905511812, 696.8503937007874, 889.7637795275591, 748.0314960629921, 751.9685039370079]; CV=0.07917380912620081)* |
| specimen rim-excluded test diameter | 761.811 µm | 600.0 | longer side of the gray>0.32 core inside each segmented test, same 500 µm scale bar *(cores=[732.2834645669292, 791.3385826771654, 692.9133858267717, 877.9527559055118, 740.1574803149606, 736.220472440945]; halo mean=769.6850393700788)* |
| grid convergence FFT wavelength at 128 | 32.4622 µm | 31.37 | radial FFT peak on a 128² face at the same γ as the 160² production run *(also 96²=31.18235201808591, 160²=32.87596870707274)* |
| specimen structure-tensor coherency | 0.5762 coherency | 0.4869 | mean (λ1−λ2)/(λ1+λ2) inside the segmented tests, Gaussian σ = 1.6 px on the 720×540 plate |
| specimen foreground fraction | 0.4203 area fraction | 0.4404 | fraction of the 720×540 plate covered by tests with circularity > 0.45 after opening and hole fill |
| segmented test count | 6 count | — | connected components on the Hillewaert plate after rejecting the scale bar and caption by area and circularity *(the plate shows six tests; the count is measured, not copied from a parameter)* |
| spot spacing CV | 0.1407 CV | 0.0 | std/mean of nearest-neighbour distances between activator peaks *(mean spacing 29.20313748718024 µm; KS vs Normal(31.37 µm) = 0.2699510911364592)* |

## Environment at checkpoint


Pass-2 script adds the rim-excluded diameter and the 96/128/160 grid comparison.
