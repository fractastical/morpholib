# MorphoEval report — audit `2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `a491e0e4871f38b92696ac7909d8a014270fb92c` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ammonia beccarii Turing wavelength at 31 µm, Holocene 0 Ma (`models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py`) | 88 | 82 | 86 | 84 | 90 | 86 | 86 | 87.3 | 82.8 | 87.3 | 85.5 | **85.5** | faithful |

## Findings and remediation

### Ammonia beccarii Turing wavelength at 31 µm, Holocene 0 Ma  —  overall 85.5 (faithful)

Schnakenberg reaction-diffusion on a 600 µm periodic face. γ is solved so the linear critical wavelength is 31.37 µm (3.137e-05 m, log10 −4.503), the 31 µm detail tier. Dv/Du = 10.4 keeps the unstable band at 26.9–39.1 µm. The pore diameter 0.564 µm and the whole-test mode 600 µm fall outside that band. Chronoseries Ammonia 1/1 at 0 Ma has no other age. A six-point γ sweep tracks λ with Pearson r = 0.995. The linearised Fourier monodromy matches σ(k) at r = 1.000. The Hillewaert CC BY-SA plate is segmented (6 tests) and the activator is grown inside those tests across 12 frames at 720×540. Bounding-box diameters average 770 µm against the caption's ~600 µm.

- **Model fidelity to source: 88** — The script implements the Schnakenberg steady state, Jacobian, Turing inequality, critical wavenumber k_c² = γ (Dv f_u + Du g_v) / (2 Du Dv), and λ_c = 2π/k_c, with γ solved so λ_c = 31.37 µm. The linearised Fourier monodromy matches analytical σ at Pearson r = 0.9999997 (RMSE about 2.5e-5 /τ after rounding in the sidecar). Equal diffusion gives σ(k_c) = −0.055. Four Lean theorems encode the margin, the 19-wavelength count, the pore/detail/test ordering, and the Holocene terminal; they were not kernel-checked because Lean 4 is not installed. The kinetics are Schnakenberg's, not a foraminiferal calcification law.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L1–36; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L93–157; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L159–214; `morphoeval/lean/loops/Loop_3108_ammonia_beccarii_turing_wavelength_s0.lean` L14–44
- **Morphological realism / specimen match: 82** — The growth GIF's last frame is the Hillewaert plate with the simulated activator grown inside the six segmented tests. Computer-vision similarity of that frame against the photograph is 92.2 (12-bin Sobel orientation, radial FFT, structure-tensor coherency, occupancy). The shells, black background, rose-bengal individual, and 500 µm bar are the photograph. The spot lattice is the model, not a resolved natural pustule field: pores at 0.564 µm are below this plate's sampling.  
  evidence: `docs/media/ammonia_beccarii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/ammonia_beccarii_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L441–460
- **Parameter expressivity: 86** — γ factors 0.70–1.80 move the predicted wavelength from 37.5 to 23.4 µm and the measured FFT wavelength from 38.7 to 23.0 µm (Pearson r = 0.995, RMSE 0.94 µm), in the direction λ ∝ γ^(−1/2). Setting Dv = Du yields σ(k_c) < 0. Setting γ = 0 drops the noise standard deviation to 0.12 of its initial value. a and b are fixed, not swept.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L282–327; `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 84** — Linear σ matches the monodromy at r = 0.9999997. The γ sweep matches at r = 0.995 (RMSE 0.94 µm). The 160² saturated FFT peak is 32.88 µm against the linear target 31.37 µm (4.8% longer; not a copy of the reference). Spot-spacing CV is 0.14 and the KS distance to Normal(31.37 µm, band half-width) is 0.27. The plate yields 6 tests. Bounding-box diameters average 769.7 µm against the caption ~600 µm (reference 600, value not equal). Foreground fraction 0.420 versus the full-resolution signature 0.440. Coherency at σ = 1.6 px is 0.576 versus 0.487 at σ = 2.8 px. The diameter offset is the open gap.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L329–430; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L637–750; `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_reference.json` L0–0
- **Source and data traceability: 90** — Turing 1952, Schnakenberg 1979, Hayward 2004, Schönfeld 2021, Petersen 2016, and Subba & Ghosh 2023 were resolved on the Crossref API. The specimen file is SHA-256 c15a90be…, CC BY-SA 4.0, Hans Hillewaert, with the Commons URL in the reference JSON. Runtime verification checks that digest. The species-gallery JPEG labeled Ammonia_beccarii.jpg is a Caryophyllia coral and was rejected. Deduction: AFM pores are Ammonia cf. beccarii, not these six tests.  
  evidence: `datasets/ammonia_beccarii_turing_wavelength_s0/paper_citations.bib` L1–84; `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L53–91; `datasets/ammonia_beccarii_turing_wavelength_s0/scout_manifest.json` L0–0
- **Code-to-model traceability: 86** — jacobian, sigma_of_k2, gamma_for_wavelength, linear_mode_check, simulate_field, wavelength_sweep, and ablation_controls each implement one stated equation. The Lean module cites the same integer encodings (744769*5 > 208*15625, 600000/31370 = 19, pore < detail < test, series 1/1 at age 0). No equation number from Turing 1952 is quoted in a comment next to the stencil.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L100–157; `morphoeval/math_kg_loops/delphi10000-ammonia-beccarii-turing-wavelength-s0.yaml` L1–71
- **Reproducibility package: 86** — Seed 3108, pinned SHA-256 inputs, --self-test (dispersion r > 0.98, sweep r > 0.98, band membership, age 0, chronoseries 1/1, GIF ≥ 10 frames and ≥ 480 px). The pass-1 run printed PASS with saturated λ = 32.88 µm, 6 tests, 12 frames at 720×540. Dependencies are the environment's numpy/scipy/Pillow/matplotlib, not a per-script lockfile. Lean was not executed.  
  evidence: `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` L600–635; `datasets/ammonia_beccarii_turing_wavelength_s0/README.md` L1–38

Remediation:
- {'id': 'R1', 'item': 'Bounding-box diameter averages 770 µm against the caption ~600 µm because gray>0.08 includes the bright rim. Remeasure on a rim-excluded mask and report both.'}
- {'id': 'R2', 'item': 'The 160² saturated FFT wavelength is 32.88 µm, 4.8% above λ_c = 31.37 µm. Show grid convergence against the 96² sweep point (31.18 µm at the same γ).'}
- {'id': 'R3', 'item': 'Single assessor (grok-4.7). No second provider scored this pass.'}
- {'id': 'R4', 'item': 'Lean 4 is not installed; the four theorems were not kernel-checked.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ammonia beccarii Turing wavelength at 31 µm, Holocene 0 Ma
- **spec_fingerprint** (semantic model + inputs): `5c78fef63a2342ef4c8a82571c976eb61e437c0ace429ce768a5b9e0809b609e`
- **input_dataset_sha256**: `datasets/ammonia_beccarii_turing_wavelength_s0/Ammonia_beccarii_Hillewaert_2012.jpg` (`c15a90be96e2b088…`), `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_reference.json` (`6a6a4fb506092514…`), `datasets/ammonia_beccarii_turing_wavelength_s0/paper_citations.bib` (`5ac93aba84d88754…`), `datasets/ammonia_beccarii_turing_wavelength_s0/scout_manifest.json` (`12dc86d6408aac7a…`)
- `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_metrics.json` — artifact_sha256 `58b2c53868af1f04…`, 2,797 B, 
- `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_reference.json` — artifact_sha256 `6a6a4fb506092514…`, 7,302 B, 
- `datasets/ammonia_beccarii_turing_wavelength_s0/scout_manifest.json` — artifact_sha256 `12dc86d6408aac7a…`, 9,698 B, 
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
| specimen segmented mean test diameter | 769.685 µm | 600.0 | equivalent circular diameter of each filled test on the resized plate, scaled by the detected 500 µm bar *(n=6; diameters=[736.220472440945, 795.2755905511812, 696.8503937007874, 889.7637795275591, 748.0314960629921, 751.9685039370079]; CV=0.07917380912620081)* |
| specimen structure-tensor coherency | 0.5762 coherency | 0.4869 | mean (λ1−λ2)/(λ1+λ2) inside the segmented tests, Gaussian σ = 1.6 px on the 720×540 plate |
| specimen foreground fraction | 0.4203 area fraction | 0.4404 | fraction of the 720×540 plate covered by tests with circularity > 0.45 after opening and hole fill |
| segmented test count | 6 count | — | connected components on the Hillewaert plate after rejecting the scale bar and caption by area and circularity *(the plate shows six tests; the count is measured, not copied from a parameter)* |
| spot spacing CV | 0.1407 CV | 0.0 | std/mean of nearest-neighbour distances between activator peaks *(mean spacing 29.20313748718024 µm; KS vs Normal(31.37 µm) = 0.2699510911364592)* |

## Environment at checkpoint


Pass-1 code, Hillewaert plate, Lean module, and rendered GIF/PNG.
