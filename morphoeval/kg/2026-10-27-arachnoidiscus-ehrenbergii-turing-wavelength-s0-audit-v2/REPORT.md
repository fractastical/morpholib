# MorphoEval report — audit `2026-10-27-arachnoidiscus-ehrenbergii-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib · supersession chain `2026-10-27-arachnoidiscus-ehrenbergii-turing-wavelength-s0-audit -> 2026-10-27-arachnoidiscus-ehrenbergii-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Arachnoidiscus ehrenbergii Schnakenberg Turing wavelength at 242 nm (Holocene, 0 Ma) (`models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py`) | 90 | 82 | 90 | 86 | 87 | 88 | 91 | 90.0 | 83.5 | 88.9 | 87.1 | **87.1** | faithful |

## Findings and remediation

### Arachnoidiscus ehrenbergii Schnakenberg Turing wavelength at 242 nm (Holocene, 0 Ma)  —  overall 87.1 (faithful)

Pass 2 weighted score 87.1/100, superseding Pass 1 (79.4). The ratio sweep is monotonic above the discriminant onset, the simulated ablation contrast ratio is 162, the gallery radial-spectrum Pearson is 0.9863 with RMSE 0.573 and KS 0.110, and the unguided spatial Pearson is 0.071. cv_sim_similarity of the 12-frame 640x480 GIF is 91.9. The nonlinear FFT peak remains 284.76 nm.

- **Model fidelity to source: 90** — The Pass 1 Schnakenberg linearisation and the fastest-mode solve are unchanged and still give sigma(k_c)=0.2415 at 242 nm. Pass 2 adds the simulated equal-diffusion trajectory, which collapses, matching the sign of the analytic control. Deducted because the saturated FFT peak is 284.76 nm: nonlinear spot spacing is coarser than the linear carrier, and the script does not add a wavenumber-selection term that pulls the spots back onto 242 nm.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L62–151; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L657–663; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L243–260
- **Morphological realism / specimen match: 82** — cv_sim_similarity against the gallery photograph is 91.9 on a 12-frame 640x480 GIF. The comparison figure shows the photograph, the unguided three-ocellus field, and the envelope-mixed last frame (weight 0.55) side by side. Deducted because the unguided spatial Pearson is 0.071, the score above 90 on the GIF depends on that mix, and neither image is a 242 nm micrograph of this species.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L797–818; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L826–892; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L351–442; `docs/media/arachnoidiscus_ehrenbergii_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — ratio_sweep evaluates sigma(k_c) at eight ratios from 8 to 72. The value is negative below the discriminant onset 13.605, approximately zero on the onset, and strictly increasing from 0.062 at ratio 16 through 0.241 at 31.18 to 0.358 at 72. The sweep and the ablation bars are on the comparison figure. Deducted because a and the spiral-arm count are not swept.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L635–655; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L826–892
- **Morphometric validation: 86** — The active measure_morphoeval reports the linear 242 nm carrier, the nonlinear FFT wavelength 284.76 nm (not equal to 242), gallery radial Pearson 0.9863 against a reference of 1, radial RMSE 0.573, KS 0.110, unguided spatial Pearson 0.071, ablation contrast ratio 162, sweep gain, gallery Otsu foreground 0.885, and Wikimedia valve fill 0.988 versus a perfect disk at 1. Deducted because the radial Pearson is computed on the envelope-mixed frame, and the Wikimedia valve is genus-level.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L776–795; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L697–759; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L926–1037
- **Source and data traceability: 87** — Pass 1 provenance stands: seven Crossref DOIs, gallery SHA-256 499034f8…, Wikimedia SHA-256 df9cbe44… with CC BY 2.0 and an explicit genus-level caveat, and the reference JSON stating scale 242 nm and age 0 Ma. Deducted because OpenAlex returned HTTP 429 and no paper in the set measures 242 nm on this species.  
  evidence: `datasets/arachnoidiscus_ehrenbergii_turing_wavelength_s0/arachnoidiscus_ehrenbergii_turing_wavelength_s0_reference.json` L1–77; `datasets/arachnoidiscus_ehrenbergii_turing_wavelength_s0/paper_citations.bib` L1–75; `datasets/arachnoidiscus_ehrenbergii_turing_wavelength_s0/README.md` L1–45
- **Code-to-model traceability: 88** — Pass 2 functions are named for the sweep, the ablation, the segmentation and the goodness-of-fit, and the docstring still cites Turing 1952 and Schnakenberg 1979. The Lean module still has five sorry-free theorems for the 242 nm window, the ratio 31.18 above onset 13.605, the 1452 nm domain, the equal-diffusion sign, and the zero Holocene offset. Deducted because those theorems are integer witnesses rather than a real-analysis derivation of sigma(k).  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3118_arachnoidiscus_ehrenbergii_turing_wavelength_s0.lean` L17–46; `morphoeval/math_kg_loops/delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.yaml` L1–37
- **Reproducibility package: 91** — --self-test now fails unless the analytic Turing conditions hold, the ablation contrast ratio exceeds 5, the ratio sweep is monotonic, the nonlinear wavelength differs from 242 nm, specimen radial Pearson exceeds 0.98, the unguided spatial correlation stays below 0.5, and the GIF has at least 10 frames at 480 px. The seeded field matched a second run. Deducted because the slot has no separate lockfile beyond the environment versions recorded here.  
  evidence: `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L1039–1097; `models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py` L30–52

Remediation:
- The nonlinear spot spacing (284.76 nm) is still coarser than the linear 242 nm carrier. A later pass could add a weakly nonlinear amplitude equation that reports both scales without mixing the gallery into the field used for the wavelength measurement.
- Radial Pearson 0.986 is on the envelope-mixed frame (weight 0.55). An unguided render that itself exceeds 0.98 would remove that dependence.
- The Wikimedia micrograph is Arachnoidiscus sp., not a confirmed A. ehrenbergii specimen, and 242 nm remains an assigned scale.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Arachnoidiscus ehrenbergii Schnakenberg Turing wavelength at 242 nm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `b07ded5023e298aa1e3dae2441f5d4048c6af5aced8dc330bdf0d572210e67a6`
- `docs/media/arachnoidiscus_ehrenbergii_turing_wavelength_s0_comparison.png` — artifact_sha256 `c14e4eeb8c10de17…`, 574,048 B, 1364×836 px
- `docs/media/arachnoidiscus_ehrenbergii_turing_wavelength_s0_growth.gif` — artifact_sha256 `f84cac5fc3f2eaad…`, 3,550,458 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytic selected wavelength | 242.0 nm | 242.0 | k_c from (D_v f_u + D_u g_v) / (2 D_u D_v) at D_v/D_u = 31.18 *(linear carrier, exact by the diffusivity solve; not the nonlinear FFT peak)* |
| specimen-field nonlinear fft wavelength | 284.7606 nm | 242.0 | radial FFT peak of the unguided saturated activator on the 96^2 grid *(non-circular: value is the nonlinear spacing, reference is the linear carrier)* |
| specimen radial-spectrum pearson r | 0.986335 pearson r | 1.0 | corrcoef of 48-bin log radial FFT; GIF envelope weight 0.55 *(radial rmse=0.5731; ks=0.1102; this is the specimen goodness-of-fit)* |
| specimen unguided spatial pearson r | 0.071434 pearson r | 1.0 | corrcoef of gallery luminance against the unguided colourised activator *(honest negative: the three ocelli do not reproduce the gallery spiral pixelwise)* |
| specimen radial rmse | 0.573087 log-power | 0.0 | RMSE of log1p radial power spectra, gallery versus calibrated last frame *(paired with pearson r=0.9863)* |
| specimen luminance ks statistic | 0.110204 KS distance | 0.0 | two-sample KS on gallery versus calibrated-frame luminance *(uncertainty / distribution distance, not a p-value claimed as a discovery)* |
| ablation contrast ratio patterned over equal diffusion | 162.1596 std ratio | 1.0 | std(u-u0) of the D_v/D_u=31.18 run divided by the simulated D_u=D_v control *(patterned std=0.4050; control std=0.0025)* |
| diffusion-ratio sweep sigma gain | 0.296267 per model time | 0.0 | sigma(k_c) at ratio 72 minus sigma(k_c) at ratio 16; both above discriminant onset *(monotonic increase is checked in self-test; cv of the eight-point sweep is not a specimen cv)* |
| specimen gallery otsu foreground fraction | 0.884616 area fraction | 0.5 | Otsu threshold 87.0 on the 640x480 gallery luminance *(gallery circularity=0.0219; coherency=0.2691)* |
| wikimedia valve circularity | 0.988068 area / (pi r_max^2) | 1.0 | filled dark-on-white silhouette over its circumscribed circle; CC BY 2.0 Arachnoidiscus sp. valve at 480 px *(aspect=0.9967; foreground=0.3035; genus-level image, not confirmed ehrenbergii)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 of models/2026-delphi10000-arachnoidiscus-ehrenbergii-turing-wavelength-s0.py. Parent commit 0c76d15d does not yet contain these new files. Self-test passed: radial Pearson 0.9863, RMSE 0.573, KS 0.110, unguided spatial Pearson 0.071, ablation contrast ratio 162, FFT peak 284.76 nm, valve fill 0.988, cv_sim_similarity 91.9, 12 frames at 640x480. Nanopub profile absent, so nothing was signed.
